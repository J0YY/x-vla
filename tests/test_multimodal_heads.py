"""Tensor-pure multimodal action heads (flow-matching + product-routing).

The linear+MSE action head emits E[a|obs] and collapses on multimodal targets (the
0% closed-loop root cause). These tests assert the two multimodal heads (1) stay in
the tensor-network class (their BilinearFFN cores fold exactly), (2) run inside ChiVLA
forward+backward for every head type, and (3) actually escape mode-collapse on a
conditional-bimodal target where the linear head cannot.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from xvla.models.vla import ChiVLA, VLAConfig
from xvla.nn.flow_action import FlowMatchingActionHead
from xvla.nn.normalization import PerTokenRmsNorm, RationalNorm
from xvla.nn.product_routing import ProductRoutingHead


def _synth(dim, H, d_a, device, Wfeat, base, delta, B):
    """Conditional-bimodal chunk: context picks mode prob; latent coin picks the mode."""
    c = torch.randint(0, 2, (B,), device=device)
    h = F.one_hot(c, 2).float() @ Wfeat
    p_plus = torch.where(c == 1, 0.7, 0.3)
    s = torch.where(torch.rand(B, device=device) < p_plus, 1.0, -1.0)
    chunk = base[c] + s[:, None, None] * delta[None]
    chunk[:, :, -1] = s[:, None]                       # gripper dim = mode sign
    chunk = chunk + 0.05 * torch.randn_like(chunk)
    return h, chunk, s


def test_flow_head_shapes_and_decode():
    torch.manual_seed(0)
    head = FlowMatchingActionHead(dim=16, action_dim=7, horizon=4, depth=2, time_degree=3)
    h = torch.randn(5, 16)
    v = head.velocity(torch.randn(5, 28), torch.rand(5, 1), h)
    assert v.shape == (5, 28)
    assert head.decode(h, n_steps=4).shape == (5, 4, 7)
    assert head.sample(h, n_steps=4, n_samples=3).shape == (5, 3, 4, 7)
    loss = head.fm_loss(h, torch.randn(5, 4, 7))
    loss.backward()
    assert torch.isfinite(loss)


def test_product_head_shapes_and_modes():
    torch.manual_seed(0)
    head = ProductRoutingHead(dim=16, action_dim=7, horizon=4, n_factors=3)
    h = torch.randn(5, 16)
    assert head.decode(h).shape == (5, 4, 7)
    assert head.enumerate_modes(h).shape == (5, 8, 4, 7)   # 2^3 modes
    loss, best = head.loss(h, torch.randn(5, 4, 7))
    loss.backward()
    assert torch.isfinite(loss) and best.shape == (5,)


def test_chi_vla_all_head_types_forward_backward():
    for head in ["linear", "flow", "product", "quantile"]:
        torch.manual_seed(0)
        cfg = VLAConfig(image_size=32, patch_size=8, vit_dim=48, vit_layers=2, vit_heads=4,
                        vocab_size=16, max_instr_len=8, state_dim=8, n_embodiments=1,
                        dim=64, n_layers=2, n_heads=4, action_horizon=4, action_dim=7,
                        action_head=head, flow_steps=4, n_factors=2)
        m = ChiVLA(cfg)
        B = 4
        a, loss = m(torch.rand(B, 3, 32, 32), torch.randint(0, 16, (B, 8)),
                    torch.randn(B, 8), torch.zeros(B, dtype=torch.long),
                    target_actions=torch.randn(B, 4, 7))
        assert a.shape == (B, 4, 7)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_convolutional_vision_uses_its_requested_grid():
    cfg = VLAConfig(image_size=32, patch_size=8, vit_dim=48, vit_layers=2, vit_heads=4,
                    vocab_size=16, max_instr_len=8, state_dim=8, n_embodiments=1,
                    dim=64, n_layers=2, n_heads=4, action_horizon=4, action_dim=7,
                    vision_encoder="conv", conv_grid=2)
    model = ChiVLA(cfg)
    actions, _ = model(torch.rand(2, 3, 32, 32), torch.randint(0, 16, (2, 8)),
                       torch.randn(2, 8), torch.zeros(2, dtype=torch.long))
    assert model.vision.features(torch.rand(2, 3, 32, 32)).shape == (2, 4, 48)
    assert actions.shape == (2, 4, 7)


def test_rational_norm_tracks_per_token_rmsnorm():
    torch.manual_seed(0)
    ref = PerTokenRmsNorm()
    norm = RationalNorm(variant="pade")
    x = torch.randn(64, 16)
    norm.train()
    norm(x)
    norm.freeze()
    got = norm(x)
    want = ref(x)
    rel = (got - want).norm() / want.norm()
    assert rel < 0.01


def test_multimodal_heads_escape_collapse():
    """Linear head collapses (gripper ~0, never commits); flow & product commit."""
    torch.manual_seed(0)
    dim, H, d_a, steps = 24, 4, 7, 800
    m_flat = H * d_a
    Wfeat = torch.randn(2, dim)
    base = torch.randn(2, H, d_a) * 0.3
    delta = torch.randn(H, d_a); delta = delta / delta.norm() * 3.0

    def batch(B):
        return _synth(dim, H, d_a, "cpu", Wfeat, base, delta, B)

    # linear baseline collapses
    lin = nn.Linear(dim, m_flat)
    opt = torch.optim.Adam(lin.parameters(), 3e-3)
    for _ in range(steps):
        h, chunk, s = batch(128)
        loss = F.mse_loss(lin(h), chunk.reshape(-1, m_flat))
        opt.zero_grad(); loss.backward(); opt.step()
    h, chunk, s = batch(1024)
    g_lin = lin(h).reshape(-1, H, d_a)[:, :, -1].mean(1)
    lin_commit = (g_lin.abs() > 0.5).float().mean().item()
    assert lin_commit < 0.2, f"linear should collapse, commit={lin_commit}"

    # flow head commits and covers both modes
    flow = FlowMatchingActionHead(dim, d_a, H, depth=2, time_degree=3)
    opt = torch.optim.Adam(flow.parameters(), 3e-3)
    for _ in range(steps):
        h, chunk, s = batch(128)
        loss = flow.fm_loss(h, chunk)
        opt.zero_grad(); loss.backward(); opt.step()
    h, chunk, s = batch(1024)
    samp = flow.sample(h, n_steps=16, n_samples=6)
    g = samp[:, 0, :, -1].mean(1)
    assert (g.abs() > 0.5).float().mean().item() > 0.6, "flow should commit"

    # product head commits
    prod = ProductRoutingHead(dim, d_a, H, n_factors=3)
    opt = torch.optim.Adam(prod.parameters(), 3e-3)
    for _ in range(steps):
        h, chunk, s = batch(128)
        loss, _ = prod.loss(h, chunk)
        opt.zero_grad(); loss.backward(); opt.step()
    h, chunk, s = batch(1024)
    g = prod.decode(h)[:, :, -1].mean(1)
    assert (g.abs() > 0.5).float().mean().item() > 0.6, "product should commit"


def test_bilinear_cores_fold_exactly():
    """Every gate/expert/velocity core is a BilinearFFN → folds to its dense core."""
    from xvla.nn.bilinear import BilinearFFN
    torch.manual_seed(0)
    prod = ProductRoutingHead(dim=6, action_dim=3, horizon=2, n_factors=2).double()
    x = torch.randn(4, 6).double()
    xb = torch.cat([torch.ones(4, 1).double(), x], 1)
    for core in [prod.center, *prod.factors, *prod.gates]:
        T = core.dense_core()
        y = torch.einsum("oij,bi,bj->bo", T, xb, xb)
        assert torch.allclose(core(x), y, atol=1e-10)
