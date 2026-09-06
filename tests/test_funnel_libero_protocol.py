"""Protocol regression tests for the funnel LIBERO runner.

These exist because an audit caught a disqualifying bug that produced no error and no
visible symptom: the runner looked task language up by the OFFICIAL LIBERO benchmark
index while indexing with the HuggingFace lerobot `task_index` from the frames cache.
Those index spaces differ on 9 of the 10 LIBERO-Object tasks, so the model trained on a
fixed permutation that scrambled instruction against image, and was then evaluated
unscrambled. Every closed-loop number from such a run is contaminated.

The tests are deliberately source-level where they have to be: LIBERO and MuJoCo are not
installed in the local environment, so the runner cannot be imported here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from athena.post_norm_intervention_v1_common import DATASET_TO_OFFICIAL_TASK

RUNNER = Path(__file__).resolve().parents[1] / "athena" / "run_funnel_libero.py"


def test_frozen_task_mapping_is_a_permutation_with_one_fixed_point():
    """The mapping is a repo-frozen constant. Pin its shape so a silent edit is caught."""
    assert sorted(DATASET_TO_OFFICIAL_TASK) == list(range(10))
    assert sorted(DATASET_TO_OFFICIAL_TASK.values()) == list(range(10))
    fixed = [d for d, o in DATASET_TO_OFFICIAL_TASK.items() if d == o]
    # Exactly one task index means the same thing in both spaces. If this ever becomes
    # ten, the two orderings have converged and the remap is a no-op; if it becomes zero
    # or some other number, the frozen constant changed and every past result is suspect.
    assert fixed == [3], f"expected only task 3 to be a fixed point, got {fixed}"


def test_runner_builds_its_task_table_through_the_frozen_mapping():
    src = RUNNER.read_text(encoding="utf-8")
    assert "DATASET_TO_OFFICIAL_TASK" in src, (
        "runner must remap the dataset task index to the official benchmark index")
    # The naive form is the bug: keyed by official index, consumed with the dataset index.
    naive = re.search(r"tasks\s*=\s*\{\s*ti\s*:\s*suite\.get_task\(ti\)", src)
    assert naive is None, "runner reintroduced the unmapped task table"


def test_runner_latches_success_and_does_not_abort_on_env_done():
    """Success must be sticky.

    LIBERO recomputes `_check_success` every step, so reading the reward of the last
    executed step in a chunk scores a policy that succeeds and then nudges the object
    back out as a failure. The canonical protocol breaks the instant reward > 0.
    """
    src = RUNNER.read_text(encoding="utf-8")
    assert "ok, t = False, 0" in src, "success flag is not latched"
    assert "succ += int(ok)" in src, "episode is not scored on the latched flag"
    assert "while not ok and t <" in src, (
        "episode loop must exit on latched success, not on env `done`")
    assert "if r > 0:\n                            ok = True" in src or \
           re.search(r"if r > 0:\s*\n\s*ok = True", src), "reward does not latch"


def test_runner_seeds_the_environment_before_reset():
    src = RUNNER.read_text(encoding="utf-8")
    assert re.search(r"env\.seed\(ti \* 100 \+ ep\)", src), (
        "canonical protocol seeds the env as ti*100+ep before reset")


def test_runner_records_provenance():
    """A multi-job claim must be checkable after the fact."""
    src = RUNNER.read_text(encoding="utf-8")
    for field in ("frames_sha256", "dataset_to_official_task", "mujoco_gl",
                  "gpu_name", "matmul_tf32"):
        assert field in src, f"result JSON does not record {field}"


def test_truncation_curve_has_a_matched_dense_anchor():
    """The eps=0 point must use the same episode count as the truncated points.

    Comparing a 500-trial dense anchor against 100-trial truncated points is the exact
    protocol error this repo previously retracted a headline over.
    """
    src = RUNNER.read_text(encoding="utf-8")
    assert "is_dense_anchor" in src, "truncation curve lacks a matched dense anchor"


def test_gripper_index_is_last_not_hardcoded_six():
    src = RUNNER.read_text(encoding="utf-8")
    assert "a[6]" not in src, "gripper index must be a[-1], not a hardcoded a[6]"
    assert "a[-1] = 1.0 if a[-1] > 0 else -1.0" in src


def test_runner_persists_results_before_it_can_crash():
    """Hours of rollouts must not be discardable by a later formatting error.

    A full matrix was lost exactly this way: the eps=None full-rank identity control
    reached an f-string `{eps:.3f}`, raised TypeError, and seven jobs threw away
    training plus a 500-trial rollout plus the ODT export they had already completed.
    """
    src = RUNNER.read_text(encoding="utf-8")
    assert "def _save(" in src, "runner has no incremental save helper"
    assert src.count("_save(result)") >= 3, (
        "results must be persisted after the closed-loop number and after each "
        "truncation row, not only at the very end")
    assert ".partial" in src and "os.replace" in src, (
        "saves must be atomic so a crash mid-write cannot corrupt the result")


def test_truncation_label_handles_the_full_rank_control():
    """`eps` is None for the full-rank identity row; formatting it as a float raises."""
    src = RUNNER.read_text(encoding="utf-8")
    assert 'label = "full-rank" if eps is None' in src
    assert "eps {eps:.3f}: remove" not in src, "unguarded float format on a nullable eps"


def test_control_arm_does_not_inherit_the_scalar_norm_constant_fill():
    """The arms get separate const_fill knobs.

    NOTE the original justification for this ("const_fill collapses a per_token model")
    was retracted: the collapse is beta-independent (DC = 1.000000 at beta 0, 0.5, 1, 2),
    so it is a training dynamic, not the fill. The separate knob is still correct, but
    the asymmetry must be recorded rather than justified by a false mechanism.
    """
    src = RUNNER.read_text(encoding="utf-8")
    assert "control_const_fill" in src, (
        "the non-decomposable arm must get its own const_fill, not the scalar-norm one")


def test_partial_truncation_curve_cannot_look_complete():
    """The eps grid is ASCENDING and jobs run on a preemptible partition.

    Without a completion marker, a run preempted mid-curve emits a well-formed curve
    showing capability preserved at every tested epsilon, having dropped exactly the
    rows that degrade. That is an optimistic, plausible, wrong result.
    """
    src = RUNNER.read_text(encoding="utf-8")
    assert "complete=False" in src, "no completion marker in the base result dict"
    assert 'result["complete"] = True' in src, "completion is never asserted"
    assert "truncation_eps_grid" in src, (
        "the intended grid must be recorded so a short curve is detectable")


def test_runner_records_what_actually_built_the_model():
    """Two runs at different const_fill must not have identical provenance.

    const_fill sets the const:linear:quadratic ratio AND inflates the DC term that
    realized_eps is normalised against, so it is part of the epsilon axis.
    """
    src = RUNNER.read_text(encoding="utf-8")
    for field in ("const_fill=args.const_fill", "effective_const_fill",
                  "truncation_trials=args.truncation_trials"):
        assert field in src, f"result JSON does not record {field}"


def test_guards_are_not_nan_permeable():
    """max(nan, x) is nan and nan > threshold is False, so NaN slips every guard."""
    src = RUNNER.read_text(encoding="utf-8")
    assert "not math.isfinite(out_dc)" in src, "collapse gate is NaN-permeable"
    assert "not math.isfinite(exact)" in src, "exactness guard is NaN-permeable"
    assert "allow_nan=False" in src, "NaN would be written as invalid JSON"


def test_cli_const_fill_default_matches_the_retuned_model_default():
    """The CLI shipped 10.0 while the model documents 2.0 as the retuned value, so
    every seed ran at the beta a 3-seed sweep had rejected."""
    import re as _re
    from xvla.models.chi_vla_funnel import ChiVLAFunnelConfig
    src = RUNNER.read_text(encoding="utf-8")
    m = _re.search(r'"--const-fill", type=float, default=([0-9.]+)', src)
    assert m, "could not find the --const-fill default"
    assert float(m.group(1)) == ChiVLAFunnelConfig().const_fill, (
        f"CLI default {m.group(1)} disagrees with the model default "
        f"{ChiVLAFunnelConfig().const_fill}")


def test_zero_const_fill_still_gets_the_divisor_and_gain():
    """An early return at beta<=0 gave the control arm the raw init scale and no
    jacobian gain, a third undocumented difference between the arms.

    The divisor 0.5*(beta**2 + 1) is correct at beta=0 too (it reduces to 0.5), so it
    must be applied there rather than skipped.
    """
    import torch
    from xvla.models.chi_vla_funnel import ChiVLAFunnel, ChiVLAFunnelConfig

    def block_rms(beta):
        torch.manual_seed(0)
        cfg = ChiVLAFunnelConfig(
            image_size=16, in_chans=1, vocab_size=4, max_instr_len=2, state_dim=2,
            n_embodiments=1, dim=64, n_layers=1, action_horizon=1, action_dim=2,
            const_fill=beta)
        model = ChiVLAFunnel(cfg).eval()
        with torch.no_grad():
            h = torch.randn(256, cfg.dim)
            return float(model.layers[0](model.norms[0](h)).pow(2).mean().sqrt())

    # With the divisor applied at every beta the block output stays near 1/sqrt(2)
    # rather than blowing up with beta**2 or sitting at the raw init scale.
    for beta in (0.0, 1.0, 2.0, 10.0):
        rms = block_rms(beta)
        assert 0.3 < rms < 1.6, f"beta={beta}: block RMS {rms:.3f} outside intended range"
