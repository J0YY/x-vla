"""Tests for the normalization swap that makes a trained policy ODT-compatible."""

from __future__ import annotations

import pytest
import torch

from xvla.models.vla import ChiVLA, VLAConfig
from xvla.nn.normalization import RationalNorm, RmsBatchNorm
from xvla.train.norm_swap import (
    calibrate_scalar_norms,
    swap_norm_weights,
    tail_ratio_report,
)


def _cfg(norm: str) -> VLAConfig:
    return VLAConfig(image_size=16, patch_size=8, vit_dim=32, vit_layers=1, vit_heads=2,
                     vocab_size=12, max_instr_len=4, state_dim=3, n_embodiments=1,
                     dim=32, n_layers=2, n_heads=2, action_horizon=2, action_dim=3,
                     action_head="linear", norm=norm, qk_norm=norm)


def _batch(cfg: VLAConfig, bs: int = 4):
    return (
        torch.rand(bs, 3, cfg.image_size, cfg.image_size),
        torch.randint(0, cfg.vocab_size, (bs, cfg.max_instr_len)),
        torch.randn(bs, cfg.state_dim),
        torch.zeros(bs, dtype=torch.long),
    )


def test_swap_transfers_every_weight_and_flags_only_norm_buffers():
    src = ChiVLA(_cfg("rational"))
    dst = ChiVLA(_cfg("scalar_rbn"))
    report = swap_norm_weights(dst, src.state_dict())

    # Every reported key must be a norm buffer, never a weight.
    assert report.transferred_tensors > 0
    assert report.transferred_elements > 0
    for key in report.missing_keys + report.unexpected_keys:
        leaf = key.rsplit(".", 1)[-1]
        assert leaf in {"running_rms", "running_ms", "initialized",
                        "_calib_sum", "_calib_count", "pa", "pb"}, key

    # And the weights really did land: compare tensors that exist in both.
    src_state, dst_state = src.state_dict(), dst.state_dict()
    shared = [k for k in src_state if k in dst_state
              and k.rsplit(".", 1)[-1] not in {"running_rms", "running_ms", "initialized",
                                               "_calib_sum", "_calib_count", "pa", "pb"}]
    assert shared
    for k in shared:
        assert torch.equal(src_state[k], dst_state[k]), k


def test_swap_refuses_to_drop_a_real_weight():
    """A shape/name change in a learned tensor must raise, not silently pass."""
    src = ChiVLA(_cfg("rational"))
    dst = ChiVLA(_cfg("scalar_rbn"))
    state = src.state_dict()
    weight_key = next(k for k in state if k.endswith("state_proj.weight"))
    state[f"{weight_key}_renamed"] = state.pop(weight_key)
    with pytest.raises(RuntimeError, match="would drop non-norm tensors"):
        swap_norm_weights(dst, state)


def test_swap_rejects_a_target_that_is_still_nonpolynomial():
    src = ChiVLA(_cfg("rational"))
    dst = ChiVLA(_cfg("per_token"))
    with pytest.raises(RuntimeError, match="non-polynomial normalization"):
        swap_norm_weights(dst, src.state_dict())
    # ...but is allowed when the caller explicitly opts out of the requirement.
    dst2 = ChiVLA(_cfg("per_token"))
    report = swap_norm_weights(dst2, src.state_dict(), require_polynomial=False)
    assert report.remaining_nonpolynomial_sites


def test_target_model_has_no_nonpolynomial_norm_after_swap():
    dst = ChiVLA(_cfg("scalar_rbn"))
    swap_norm_weights(dst, ChiVLA(_cfg("rational")).state_dict())
    assert not [m for m in dst.modules() if isinstance(m, RationalNorm)]
    assert [m for m in dst.modules() if isinstance(m, RmsBatchNorm)]


def test_calibration_freezes_every_exercised_site_and_is_deterministic_at_eval():
    cfg = _cfg("scalar_rbn")
    model = ChiVLA(cfg)
    torch.manual_seed(0)
    batch = _batch(cfg)
    report = calibrate_scalar_norms(model, lambda: batch, iters=3)

    sites = dict((n, m) for n, m in model.named_modules() if isinstance(m, RmsBatchNorm))
    assert report.frozen
    assert set(report.frozen) | set(report.unexercised) == set(sites)
    for name, c in report.frozen.items():
        m = sites[name]
        assert m.frozen and not m.calibrating and bool(m.initialized)
        assert float(m.running_rms) > 0 and c > 0
    for name in report.unexercised:
        assert not sites[name].calibrating, name

    # Frozen means eval is a fixed scalar divide: same input, same output, and
    # unaffected by other batches passing through.
    model.eval()
    with torch.no_grad():
        first, _ = model(*batch)
        model(*_batch(cfg, bs=7))
        second, _ = model(*batch)
    assert torch.allclose(first, second, atol=0, rtol=0)


def test_calibration_reports_the_dead_vision_norm_site():
    """`ChiVLA` never applies `vision.norm_out`, so it must be reported, not faked.

    An export that folded this site would fold an uncalibrated 1.0, so silence
    here would be a correctness bug in any downstream ODT claim.
    """
    cfg = _cfg("scalar_rbn")
    model = ChiVLA(cfg)
    batch = _batch(cfg)
    report = calibrate_scalar_norms(model, lambda: batch, iters=2)

    assert "vision.norm_out" in report.unexercised
    assert "vision.norm_out" not in report.frozen
    # Left at its loaded value rather than given a fabricated constant.
    site = dict(model.named_modules())["vision.norm_out"]
    assert float(site.running_rms) == 1.0
    assert not bool(site.initialized)


def test_calibration_raises_when_nothing_was_exercised():
    model = ChiVLA(_cfg("scalar_rbn"))
    with pytest.raises(RuntimeError, match="exercised none"):
        calibrate_scalar_norms(model, lambda: _batch(_cfg("scalar_rbn")), iters=0)


def test_tail_ratio_reports_every_site_and_leaves_no_hooks_behind():
    cfg = _cfg("scalar_rbn")
    model = ChiVLA(cfg)
    batch = _batch(cfg)
    calibrate_scalar_norms(model, lambda: batch, iters=3)

    n_hooks_before = sum(len(m._forward_hooks)
                         for m in model.modules() if isinstance(m, RmsBatchNorm))
    stats = tail_ratio_report(model, lambda: batch, iters=2)
    n_hooks_after = sum(len(m._forward_hooks)
                        for m in model.modules() if isinstance(m, RmsBatchNorm))

    assert n_hooks_after == n_hooks_before
    assert stats
    exercised = [n for n, s in stats.items() if s["n_batches"] > 0]
    assert exercised
    for name in exercised:
        s = stats[name]
        assert s["n_batches"] == 2, name
        assert s["max_ratio"] > 0
        assert s["max_ratio"] >= s["mean_ratio"]
    # The dead vision site shows up with zero batches rather than a fake ratio.
    assert stats["vision.norm_out"]["n_batches"] == 0


def test_tail_ratio_is_near_one_when_the_scalar_matches_the_per_token_rms():
    """Sanity anchor: on data whose per-token RMS is uniform, the ratio is ~1.

    This is the interpretive claim the diagnostic rests on, so it gets a test.
    """
    mod = RmsBatchNorm()
    x = torch.full((4, 6, 8), 3.0)          # every token has identical RMS
    mod.calibrating = True
    mod(x)
    c = float(mod._calib_sum) / float(mod._calib_count)
    mod.running_rms.fill_(c)
    mod.calibrating, mod.frozen = False, True

    per_token = torch.sqrt(x.float().pow(2).mean(dim=-1))
    ratio = per_token / (float(mod.running_rms) + mod.eps)
    assert abs(float(ratio.max()) - 1.0) < 1e-5
