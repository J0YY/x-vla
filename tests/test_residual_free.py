"""Tests for the residual-free (strict funnel) option.

Why this flag exists: ODT needs a *narrow separator*, meaning no path may route
around a bond. A token-wise residual stream ``x_{l+1} = x_l + block(x_l)`` carries
the full ``N*d`` layer state past any bottleneck, so with it the minimum separator
is the whole layer state regardless of the mixing mechanism. It is also what turns
the attention core's Kronecker rank from 1 into 2, which then compounds as ``2^S``
over sublayers and is what puts exact decomposition out of reach at depth.

So `residual=False` is the architectural precondition for the decomposition, and
these tests pin down that it is really off, that nothing else moved, and that the
identity path genuinely has to travel *inside* the cores instead.
"""

from __future__ import annotations

import torch

from xvla.models.vla import ChiVLA, VLAConfig
from xvla.nn.block import ChiTransformer, ChiTransformerBlock


def _blk(residual: bool, n_layers: int = 8) -> ChiTransformerBlock:
    torch.manual_seed(0)
    return ChiTransformerBlock(dim=16, n_heads=2, n_layers=n_layers,
                               causal=True, norm="scalar_rbn", qk_norm="scalar_rbn",
                               residual=residual)


def test_residual_gain_is_one_when_there_is_no_residual():
    """1/sqrt(2L) is a perturbation scale. Without a residual the branch IS the
    signal, so that gain would attenuate it to nothing across depth."""
    assert abs(float(_blk(True).attn_gain) - (2.0 * 8) ** -0.5) < 1e-12
    assert float(_blk(False).attn_gain) == 1.0
    assert float(_blk(False).ffn_gain) == 1.0


def test_no_identity_path_survives_a_zeroed_branch():
    """The decisive structural property: kill the learned branches and a
    residual block still passes its input through, a funnel block emits nothing."""
    x = torch.randn(2, 5, 16)

    keep = _blk(True)
    with torch.no_grad():
        keep.attn_gain.zero_()
        keep.ffn_gain.zero_()
    # Residual: input reaches the output untouched.
    assert torch.allclose(keep(x), x, atol=1e-6)

    funnel = _blk(False)
    with torch.no_grad():
        funnel.attn_gain.zero_()
        funnel.ffn_gain.zero_()
    # Funnel: no route around the (now dead) cores at all.
    assert torch.allclose(funnel(x), torch.zeros_like(x), atol=1e-6)


def test_funnel_block_is_not_an_additive_perturbation_of_its_input():
    x = torch.randn(2, 5, 16)
    assert not torch.allclose(_blk(False)(x), x, atol=1e-3)


def test_residual_flag_reaches_every_block_in_the_stack():
    for flag in (True, False):
        stack = ChiTransformer(dim=16, n_layers=3, n_heads=2, causal=True,
                               norm="scalar_rbn", qk_norm="scalar_rbn", residual=flag)
        assert stack.residual is flag
        assert [b.residual for b in stack.blocks] == [flag] * 3


def test_vla_plumbs_residual_to_backbone_and_vision_independently():
    def cfg(**kw):
        return VLAConfig(image_size=16, patch_size=8, vit_dim=32, vit_layers=1, vit_heads=2,
                         vocab_size=12, max_instr_len=4, state_dim=3, n_embodiments=1,
                         dim=32, n_layers=2, n_heads=2, action_horizon=2, action_dim=3,
                         action_head="linear", norm="scalar_rbn", qk_norm="scalar_rbn", **kw)

    # Default: residual everywhere (legacy behaviour must be untouched).
    m = ChiVLA(cfg())
    assert all(b.residual for b in m.backbone.blocks)
    assert all(b.residual for b in m.vision.blocks.blocks)

    # residual=False propagates to vision too, since vit_residual inherits.
    m = ChiVLA(cfg(residual=False))
    assert not any(b.residual for b in m.backbone.blocks)
    assert not any(b.residual for b in m.vision.blocks.blocks)

    # ...and vision can be pinned independently, to isolate the backbone change.
    m = ChiVLA(cfg(residual=False, vit_residual=True))
    assert not any(b.residual for b in m.backbone.blocks)
    assert all(b.residual for b in m.vision.blocks.blocks)


def test_residual_free_vla_runs_and_trains_a_step():
    cfg = VLAConfig(image_size=16, patch_size=8, vit_dim=32, vit_layers=1, vit_heads=2,
                    vocab_size=12, max_instr_len=4, state_dim=3, n_embodiments=1,
                    dim=32, n_layers=2, n_heads=2, action_horizon=2, action_dim=3,
                    action_head="linear", norm="scalar_rbn", qk_norm="scalar_rbn",
                    residual=False)
    torch.manual_seed(0)
    model = ChiVLA(cfg)
    batch = (torch.rand(4, 3, 16, 16), torch.randint(0, 12, (4, 4)),
             torch.randn(4, 3), torch.zeros(4, dtype=torch.long))
    act = torch.randn(4, 2, 3)

    out, loss = model(*batch, target_actions=act)
    assert out.shape == (4, 2, 3)
    assert torch.isfinite(loss), "residual-free forward produced a non-finite loss"

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed through the funnel"
    assert all(torch.isfinite(g).all() for g in grads)
    # Gradient actually reaches the earliest layer, which is the real risk without
    # a residual highway.
    first = model.backbone.blocks[0].ffn.left.weight
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert float(first.grad.abs().max()) > 0.0
