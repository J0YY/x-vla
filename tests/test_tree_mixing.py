"""Regression tests for the separator-friendly token mixers and the binding probe.

Written during the review of `xvla/nn/tree_mixing.py`, `xvla/nn/block.py` and
`scripts/binding_probe_v1.py`. Every assertion here is an invariant the binding
probe's conclusion depends on: if one of these breaks, the probe silently
measures something other than what it claims to measure.

The two `xfail(strict=True)` tests record KNOWN BUGS found in that review. They
are strict so that fixing the bug turns them into a loud XPASS rather than
quietly passing unnoticed — delete the marker (not the test) when you fix it.
"""

from __future__ import annotations

import math

import pytest
import torch

from xvla.models.vla import ChiVLA, VLAConfig
from xvla.nn.tree_mixing import (
    ButterflyBilinearMix,
    FixedTokenMix,
    _shift_butterfly,
    _shift_prefix,
)
from xvla.train.synth_vla import COLOR_TOK0, SHAPE_TOK0, make_batch


def _dependency_matrix(mod, n, d=8, seed=0):
    """dep[i, j] = 1 iff out[0, i] has a nonzero gradient w.r.t. in[0, j]."""
    torch.manual_seed(seed)
    x = torch.randn(1, n, d, requires_grad=True)
    y = mod(x)
    dep = torch.zeros(n, n)
    for i in range(n):
        g = torch.autograd.grad(y[0, i].sum(), x, retain_graph=True)[0]
        dep[i] = (g[0].abs().sum(-1) > 1e-12).float()
    return dep


# ---------------------------------------------------------------- shift helpers


def test_shift_prefix_semantics():
    x = torch.arange(1, 7, dtype=torch.float32).view(1, 6, 1)
    assert _shift_prefix(x, 0).flatten().tolist() == [1, 2, 3, 4, 5, 6]
    assert _shift_prefix(x, 2).flatten().tolist() == [0, 0, 1, 2, 3, 4]
    # offset >= n must zero out entirely, never wrap or clamp to index 0
    assert _shift_prefix(x, 6).flatten().tolist() == [0] * 6
    assert _shift_prefix(x, 99).flatten().tolist() == [0] * 6


def test_shift_prefix_never_self_partners():
    """A token must never be merged with itself: that would square it (see the
    module docstring's stated rationale for zero- rather than clamp-padding)."""
    x = torch.randn(1, 11, 3)
    for off in (1, 2, 4, 8):
        s = _shift_prefix(x, off)
        for p in range(11):
            assert not torch.equal(s[0, p], x[0, p]) or torch.all(s[0, p] == 0)


def test_shift_butterfly_semantics():
    x = torch.arange(1, 9, dtype=torch.float32).view(1, 8, 1)
    assert _shift_butterfly(x, 1).flatten().tolist() == [2, 1, 4, 3, 6, 5, 8, 7]
    assert _shift_butterfly(x, 4).flatten().tolist() == [5, 6, 7, 8, 1, 2, 3, 4]
    # involution on a power-of-two length
    for off in (1, 2, 4):
        assert torch.equal(_shift_butterfly(_shift_butterfly(x, off), off), x)


# ---------------------------------------------------------------- FixedTokenMix


def test_fixedmix_causal_is_strictly_causal():
    dep = _dependency_matrix(FixedTokenMix(8, 6, causal=True), 6)
    assert dep.triu(1).sum() == 0, "information leaked from later to earlier tokens"
    assert (dep[torch.tril(torch.ones(6, 6)) > 0] > 0).all(), "should reach every j <= i"


def test_fixedmix_noncausal_honours_a_binary_mask():
    from xvla.nn.attention import causal_mask

    mod = FixedTokenMix(4, 5, causal=False)
    mask = causal_mask(5)
    x = torch.randn(1, 5, 4)
    x2 = x.clone()
    x2[0, 1:] += 10.0
    a, b = mod(x, mask=mask), mod(x2, mask=mask)
    assert torch.allclose(a[0, 0], b[0, 0], atol=1e-6)


def test_fixedmix_rejects_oversized_input():
    mod = FixedTokenMix(4, 5)
    with pytest.raises(ValueError):
        mod(torch.randn(1, 6, 4))


def test_fixedmix_train_eval_agree():
    mod = FixedTokenMix(16, 8)
    x = torch.randn(2, 8, 16)
    mod.train()
    a = mod(x)
    mod.eval()
    assert torch.equal(a, mod(x)), "mixer must be stateless (no norm/dropout)"


# --------------------------------------------------------- ButterflyBilinearMix


def test_butterfly_prefix_is_strictly_causal_and_reaches_everything():
    n = 87  # the joint backbone's seq_len for the binding probe's config
    mod = ButterflyBilinearMix(8, n, width=4, causal=True)
    assert mod.n_stages == math.ceil(math.log2(n)) == 7
    dep = _dependency_matrix(mod, n)
    assert dep.triu(1).sum() == 0, "prefix merge leaked information backwards in time"
    assert (dep[torch.tril(torch.ones(n, n)) > 0] > 0).all(), "log2(N) stages must reach all j <= i"
    # the action-query positions in particular must see the whole prompt
    assert dep[-1].sum() == n


def test_butterfly_xor_is_all_to_all():
    dep = _dependency_matrix(ButterflyBilinearMix(8, 8, width=4, causal=False), 8)
    assert (dep > 0).all()


def test_butterfly_returns_the_residual_delta_not_the_state():
    """`forward` returns the ACCUMULATED merges, i.e. exactly `h_final - x`.

    The surrounding block computes `x = x + gain * mixer(norm(x))`, so this is
    what makes the log-N stages compose as one residual branch rather than
    double-counting the input.
    """
    torch.manual_seed(0)
    mod = ButterflyBilinearMix(8, 8, width=4, causal=False)
    x = torch.randn(2, 8, 8)
    h = x
    for s in range(mod.n_stages):
        partner = _shift_butterfly(h, 1 << s)
        ell, r = mod.left[s](h), mod.right[s](partner)
        merged = mod.term_norm * (
            mod.up[s](ell * r) + mod.up_self[s](ell) + mod.up_partner[s](r))
        h = h + mod.stage_gain * merged
    assert torch.allclose(mod(x), h - x, atol=1e-5)


def test_butterfly_train_eval_agree():
    mod = ButterflyBilinearMix(16, 8, width=4)
    x = torch.randn(2, 8, 16)
    mod.train()
    a = mod(x)
    mod.eval()
    assert torch.equal(a, mod(x))


def test_shift_butterfly_never_self_partners():
    """Regression: boundary tokens must get a ZERO partner, never themselves.

    A self-partner makes the merge compute up((L x_p) * (R x_p)), a squared
    self-term. Was a real bug, hitting 41/87 tokens at the last stage of n=87.
    """
    n = 87
    x = torch.randn(1, n, 3)
    for s in range(math.ceil(math.log2(n))):
        s_x = _shift_butterfly(x, 1 << s)
        self_partnered = [p for p in range(n) if torch.equal(s_x[0, p], x[0, p])]
        assert not self_partnered, f"stage {s}: {len(self_partnered)} tokens partner themselves"


# ------------------------------------------------------------------- plumbing


@pytest.mark.parametrize("attn", ["bilinear", "softmax"])
def test_vit_config_inherits_attn_unchanged(attn):
    """Existing configs must be byte-identical to the pre-`vit_attn` behaviour."""
    cfg = VLAConfig(attn=attn)
    assert cfg.vit_config().attn == attn
    assert cfg.vit_config().mix_width == cfg.mix_width


def test_vit_attn_can_be_pinned_independently():
    cfg = VLAConfig(attn="butterfly", mix_width=16, vit_attn="bilinear", vit_mix_width=64)
    assert cfg.vit_config().attn == "bilinear"
    assert cfg.vit_config().mix_width == 64
    assert cfg.attn == "butterfly" and cfg.mix_width == 16


def test_block_requires_max_tokens_for_the_new_mixers():
    from xvla.nn.block import ChiTransformerBlock

    for attn in ("fixedmix", "butterfly"):
        with pytest.raises(ValueError, match="max_tokens"):
            ChiTransformerBlock(8, 2, attn=attn)


@pytest.mark.parametrize("arm", [
    dict(attn="bilinear"),
    dict(attn="fixedmix"),
    dict(attn="butterfly", mix_width=16),
])
def test_chivla_builds_and_runs_for_every_mixing_arm(arm):
    cfg = VLAConfig(image_size=32, patch_size=8, vit_dim=16, vit_layers=1, vit_heads=2,
                    vocab_size=16, max_instr_len=4, state_dim=8, n_embodiments=2,
                    dim=16, n_layers=2, n_heads=2, action_horizon=2, action_dim=7,
                    vit_attn="bilinear", **arm)
    model = ChiVLA(cfg)
    img = torch.randn(2, 3, 32, 32)
    instr = torch.randint(0, 16, (2, 4))
    state = torch.randn(2, 8)
    emb = torch.zeros(2, dtype=torch.long)
    act, loss = model(img, instr, state, emb, target_actions=torch.randn(2, 2, 7))
    assert act.shape == (2, 2, 7) and torch.isfinite(loss)
    # the ViT is held at bilinear in every arm, which is what the probe relies on
    from xvla.nn.attention import BilinearAttention
    assert isinstance(model.vision.blocks.blocks[0].attn, BilinearAttention)


# --------------------------------------------------- binding probe methodology


def _probe_mod():
    import importlib.util
    import pathlib

    p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "binding_probe_v1.py"
    spec = importlib.util.spec_from_file_location("binding_probe_v1", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_labels_are_the_instruction_selected_object():
    bp = _probe_mod()
    torch.manual_seed(0)
    pool = make_batch(24, "cpu")
    lab = bp.labels_from_pool(pool)
    for b in range(24):
        t = int(pool["tgt_idx"][b])
        col, sh = int(pool["obj_pairs"][b, t, 0]), int(pool["obj_pairs"][b, t, 1])
        # the instruction really names object t
        assert int(pool["instr"][b, 2]) == COLOR_TOK0 + col
        assert int(pool["instr"][b, 3]) == SHAPE_TOK0 + sh
        # and the label really is that object's position
        assert abs(float(lab["tgt_x"][b]) - float(pool["all_pos"][b, t, 0])) < 1e-9
        assert abs(float(lab["tgt_y"][b]) - float(pool["all_pos"][b, t, 1])) < 1e-9
        # and the demo action really points at it
        expected = (pool["all_pos"][b, t] - pool["state"][b, :2]) / 4
        assert torch.allclose(pool["actions"][b, 0, :2], expected, atol=1e-6)


def test_ridge_r2_is_calibrated():
    bp = _probe_mod()
    torch.manual_seed(1)
    xtr = torch.randn(400, 5, dtype=torch.float64)
    w = torch.randn(5, dtype=torch.float64)
    xte = torch.randn(200, 5, dtype=torch.float64)
    # exactly linear target with a large offset -> R^2 ~ 1
    assert bp.ridge_r2(xtr, xtr @ w + 3.0, xte, xte @ w + 3.0) > 0.999
    # unpredictable target -> R^2 <= ~0
    ytr = 0.5 + 0.28 * torch.randn(400, dtype=torch.float64)
    yte = 0.5 + 0.28 * torch.randn(200, dtype=torch.float64)
    assert bp.ridge_r2(xtr, ytr, xte, yte) < 0.1


def test_probe_features_capture_the_pre_backbone_state():
    """The forward-pre-hook on `model.backbone` must capture the true input bond.

    A stale or mutated capture would make `input_mean` (the claimed 'chance
    floor') meaningless while still producing plausible-looking numbers.
    """
    bp = _probe_mod()
    torch.manual_seed(0)
    cfg = VLAConfig(image_size=32, patch_size=8, vit_dim=16, vit_layers=1, vit_heads=2,
                    vocab_size=16, max_instr_len=4, state_dim=8, n_embodiments=2,
                    dim=16, n_layers=1, n_heads=2, action_horizon=2, action_dim=7)
    model = ChiVLA(cfg)
    pool = make_batch(12, "cpu")
    pool["instr"] = pool["instr"][:, :4]
    feats = bp.collect_features(model, pool, batch=5)   # 3 uneven chunks

    model.eval()
    with torch.no_grad():
        B = 12
        vis = model._visual_tokens(pool["img"])
        x = torch.cat([vis, model.bos.expand(B, -1, -1), model.tok_emb(pool["instr"]),
                       model.state_proj(pool["state"])[:, None],
                       model.embodiment_emb(pool["embodiment"])[:, None],
                       model.action_queries.expand(B, -1, -1)], dim=1)
        x = x + model.pos_emb[:, : x.shape[1]]
    assert torch.allclose(feats["input_mean"], x.double().mean(1), atol=1e-6)
    for v in feats.values():
        assert v.dtype == torch.float64 and v.shape[0] == 12


def test_probe_gate_thresholds_are_attainable():
    """A pre-registered threshold no run can meet is a silent kill switch.

    The original gate compared `mse_shuffled - mse` (a 7-dim mean) against an
    absolute 0.01, while the entire language-blind budget on this task is
    blind_mse ~ 0.001, so no model could ever validate the positive control and
    every run reported UNINFORMATIVE. The gate is now a RATIO, which is
    scale-free and therefore attainable.
    """
    bp = _probe_mod()
    assert "min_reference_grounding_gap" not in bp.GATE, (
        "the unreachable absolute-gap gate is back")
    ratio = bp.GATE["min_reference_shuffle_ratio"]
    assert 1.0 < ratio < 100.0, f"shuffle-ratio gate {ratio} is not a sane ratio"
    # A ratio gate is dimensionless, so it cannot be priced out by blind_mse.
    torch.manual_seed(1234)
    pool = make_batch(512, "cpu")
    assert pool["blind_mse"] > 0


def test_stated_probe_floor_matches_the_measured_centroid_floor():
    """The pre-registered 0.60 bar must be justified against the RIGHT floor.

    0.224 is the `input_mean` floor. At the `post_aq` readout the language-blind
    centroid is already close to fully linear, so the relevant floor is the
    centroid-implied 1/k = 0.31 for k=3 objects. The gate documentation and the
    per-arm `min_margin_over_centroid` check both depend on this number.
    """
    bp = _probe_mod()
    torch.manual_seed(1234)
    tr, te = make_batch(4096, "cpu"), make_batch(2048, "cpu")
    ltr, lte = bp.labels_from_pool(tr), bp.labels_from_pool(te)
    r2 = bp.ridge_r2(tr["all_pos"].mean(1).double(), ltr["tgt_x"].double(),
                     te["all_pos"].mean(1).double(), lte["tgt_x"].double())
    assert 0.25 < r2 < 0.40, f"expected the ~1/3 centroid floor, measured {r2:.3f}"
    assert bp.GATE["min_probe_r2"] > r2 + 0.15, (
        f"gate {bp.GATE['min_probe_r2']} leaves too little headroom over the "
        f"measured centroid floor {r2:.3f}")


def test_evaluate_gate_does_not_crash_without_the_reference_arm():
    bp = _probe_mod()
    rec = dict(arm="butterfly_w64", seed=0, mse=0.1, grounding_gap=0.001,
               probes={"post_aq": {"tgt_x": 0.9, "tgt_y": 0.9}})
    out = bp.evaluate_gate([rec])
    assert out["_positive_control"]["valid"] is False
    assert out["_verdict"]["passed"] is False
    assert out["butterfly_w64"]["mse_ratio_vs_reference"] is None
