"""Pure negative and resampling-oracle tests for the frozen Stage 7 protocol."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from athena.aggregate_post_norm_intervention_v1 import (
    crossed_bootstrap_distribution,
    crossed_resample_mean,
)
from athena.post_norm_intervention_v1_common import (
    CACHE,
    CHECKPOINTS,
    CLAIM_BOUNDARY,
    CONDITIONS,
    DATASET_TASKS,
    DATASET_TO_OFFICIAL_TASK,
    LIBERO_RUNTIME,
    PREP_CHECK_KEYS,
    PROMPTS,
    PROTOCOL,
    RUN_CHECK_KEYS,
    SCHEMA_RUN,
    SCIENTIFIC_GATE_RESULT_KEYS,
    TASK_METADATA,
    canonical_sha256,
    rollout_condition_order,
    validate_attenuation_inventory,
    validate_check_inventory,
    validate_smoke_prerequisite,
)
from xvla.train.post_norm_intervention import HorizonLowRankProjector
from xvla.train.post_norm_intervention import attenuate_queries


def test_crossed_resample_oracle_shares_episode_draws_across_checkpoints() -> None:
    cube = np.fromfunction(lambda checkpoint, task, episode: 100 * checkpoint + 10 * task + episode, (3, 10, 10))
    checkpoints = np.asarray([2, 0], dtype=np.int64)
    tasks = np.asarray([1, 3], dtype=np.int64)
    shared = np.asarray([[0, 2], [4, 5]], dtype=np.int64)
    actual = crossed_resample_mean(cube, checkpoints, tasks, shared)
    expected_values = [
        cube[checkpoint, task, episode]
        for task, episodes in zip(tasks, shared)
        for checkpoint in checkpoints
        for episode in episodes
    ]
    assert actual == float(np.mean(expected_values))


def test_crossed_bootstrap_constant_cube_is_exact_and_deterministic() -> None:
    cube = np.full((3, 10, 10), 0.125, dtype=np.float64)
    left = crossed_bootstrap_distribution(cube, draws=17, seed=991)
    right = crossed_bootstrap_distribution(cube, draws=17, seed=991)
    np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(left, np.full(17, 0.125))


def test_check_inventory_rejects_missing_extra_nonboolean_and_failed() -> None:
    passing = {name: True for name in RUN_CHECK_KEYS}
    validate_check_inventory(passing, RUN_CHECK_KEYS, require_pass=True)
    with pytest.raises(RuntimeError, match="inventory"):
        validate_check_inventory({name: True for name in list(RUN_CHECK_KEYS)[:-1]}, RUN_CHECK_KEYS, require_pass=True)
    with pytest.raises(RuntimeError, match="inventory"):
        validate_check_inventory({**passing, "unexpected": True}, RUN_CHECK_KEYS, require_pass=True)
    with pytest.raises(RuntimeError, match="exact boolean"):
        validate_check_inventory({**passing, next(iter(RUN_CHECK_KEYS)): 1}, RUN_CHECK_KEYS, require_pass=True)
    failed = {name: True for name in PREP_CHECK_KEYS}
    failed[next(iter(PREP_CHECK_KEYS))] = False
    with pytest.raises(RuntimeError, match="failed"):
        validate_check_inventory(failed, PREP_CHECK_KEYS, require_pass=True)
    validate_check_inventory(
        {name: True for name in SCIENTIFIC_GATE_RESULT_KEYS},
        SCIENTIFIC_GATE_RESULT_KEYS,
        require_pass=False,
    )
    with pytest.raises(RuntimeError, match="inventory"):
        validate_check_inventory(
            {name: True for name in list(SCIENTIFIC_GATE_RESULT_KEYS)[:-1]},
            SCIENTIFIC_GATE_RESULT_KEYS,
            require_pass=False,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_rank_projector_rejects_unapproved_float_dtypes(dtype: torch.dtype) -> None:
    state = torch.ones(2, 3, 1, dtype=dtype)
    analysis = torch.full((2, 1, 3), 1 / 3, dtype=dtype)
    with pytest.raises(ValueError, match="float32 or float64"):
        HorizonLowRankProjector(state, analysis, "bad_dtype")


def test_attenuation_rejects_unapproved_activation_dtype() -> None:
    state = torch.tensor([[[1.0], [0.0]]], dtype=torch.float32)
    operator = HorizonLowRankProjector(state, state.transpose(1, 2), "valid")
    with pytest.raises(ValueError, match="queries must use float32 or float64"):
        attenuate_queries(
            torch.ones(2, 1, 2, dtype=torch.float16),
            torch.zeros(1, 2, dtype=torch.float16),
            operator,
            alpha=0.5,
        )


def test_canonical_hash_detects_nested_protocol_tampering() -> None:
    value = {"protocol": {"rank": 1, "conditions": ["a", "b"]}, "passed": True}
    digest = canonical_sha256(value)
    tampered = {"protocol": {"rank": 2, "conditions": ["a", "b"]}, "passed": True}
    assert canonical_sha256(tampered) != digest


def test_attenuation_inventory_allows_continuous_control_strengths() -> None:
    alphas = {name: 0.333 for name in CONDITIONS}
    alphas["balanced_top"] = 0.5
    validate_attenuation_inventory(alphas)
    alphas["balanced_top"] = 0.333
    with pytest.raises(RuntimeError, match="primary attenuation"):
        validate_attenuation_inventory(alphas)


def _valid_smoke() -> tuple[dict, dict, str]:
    freeze_sha = "f" * 64
    bundle_sha = "b" * 64
    model_sha = "m" * 64
    sources = {"xvla/example.py": "s" * 64}
    freeze = {
        "prepared": {"0": {"bundle_sha256": bundle_sha, "model_state_sha256": model_sha}},
        "source_hashes": sources,
    }
    rows = []
    for task in (0, 1):
        for episode in (0, 1):
            conditions = {
                name: {"all_gripper_actions_committed": True}
                for name in ("baseline", *CONDITIONS)
            }
            rows.append({
                "checkpoint_seed": 0,
                "task": task,
                "episode": episode,
                "reset_seed": task * 100 + episode,
                "condition_order": rollout_condition_order(0, task, episode),
                "conditions": conditions,
            })
    smoke = {
        "schema": SCHEMA_RUN,
        "mode": "strict_smoke",
        "seed": 0,
        "task_bounds": [0, 2],
        "episode_bounds": [0, 2],
        "protocol": PROTOCOL,
        "claim_boundary": CLAIM_BOUNDARY,
        "identity": {
            "checkpoint_sha256": CHECKPOINTS[0]["sha256"],
            "cache_sha256": CACHE["sha256"],
            "bundle_sha256": bundle_sha,
            "freeze_sha256": freeze_sha,
            "smoke_sha256": None,
            "model_state_sha256": model_sha,
            "source_hashes": sources,
        },
        "runtime": PROTOCOL["runtime"],
        "task_identity": {
            "official_prompts": PROMPTS,
            "dataset_tasks": DATASET_TASKS,
            "dataset_to_official_task": DATASET_TO_OFFICIAL_TASK,
            "dataset_metadata": TASK_METADATA,
            "libero_runtime": LIBERO_RUNTIME,
        },
        "alpha_zero_maximum_action_error": 0.0,
        "checks": {name: True for name in RUN_CHECK_KEYS},
        "rows": rows,
        "elapsed_seconds": 1.0,
    }
    return smoke, freeze, freeze_sha


def test_smoke_prerequisite_requires_exact_panel_and_provenance() -> None:
    smoke, freeze, freeze_sha = _valid_smoke()
    validate_smoke_prerequisite(smoke, freeze, freeze_sha)

    duplicate, freeze, freeze_sha = _valid_smoke()
    duplicate["rows"][3] = dict(duplicate["rows"][0])
    with pytest.raises(RuntimeError, match="duplication|panel"):
        validate_smoke_prerequisite(duplicate, freeze, freeze_sha)

    wrong_source, freeze, freeze_sha = _valid_smoke()
    wrong_source["identity"]["source_hashes"] = {"xvla/example.py": "x" * 64}
    with pytest.raises(RuntimeError, match="identity"):
        validate_smoke_prerequisite(wrong_source, freeze, freeze_sha)
