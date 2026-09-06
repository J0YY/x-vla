"""Static and pure validation tests for the frozen Stage 7 v2 protocol."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from athena.aggregate_post_norm_intervention_v2 import (
    crossed_bootstrap_distribution, crossed_resample_mean,
)
from athena.post_norm_intervention_v2_common import (
    CACHE, CHECKPOINTS, CLAIM_BOUNDARY, CONDITIONS, DATASET_TASKS,
    DATASET_TO_OFFICIAL_TASK, ENGINEERING_GATES, FEASIBILITY_POOL_SIZE,
    HAAR_CONDITIONS, LIBERO_RUNTIME, PROMPTS,
    MINIMUM_FEASIBILITY_ELIGIBLE, PREP_CHECK_KEYS, PRIMARY, PROTOCOL,
    RUN_CHECK_KEYS, SCIENTIFIC_GATE_RESULT_KEYS, STRUCTURED_DESCRIPTIVE,
    SAMPLING_POOL_SIZE, SCHEMA_RUN, STAGE5_RESULT_PATH, STAGE5_RESULT_SHA256,
    TASK_METADATA, canonical_sha256, condition_order, file_sha256,
    model_state_sha256, reject_v1_schema, validate_alpha_inventory,
    validate_check_inventory, validate_smoke_prerequisite,
    validate_rollout_row, validate_structured_descriptive,
    validate_structured_metadata_order,
)
from athena.prepare_post_norm_intervention_v2 import model_state_sha256 as prepare_model_hash
from athena.run_post_norm_intervention_v2 import model_state_sha256 as rollout_model_hash
from athena.freeze_post_norm_intervention_v2 import validate_structured_descriptive as freeze_structured_validator


ROOT = Path(__file__).resolve().parents[1]


def test_exact_deployed_inventory_excludes_structured_controls() -> None:
    assert CONDITIONS == (PRIMARY, *HAAR_CONDITIONS)
    assert len(HAAR_CONDITIONS) == 5
    assert not set(CONDITIONS) & set(STRUCTURED_DESCRIPTIVE)
    assert PROTOCOL["descriptive_controls_are_never_deployed"] is True
    assert (FEASIBILITY_POOL_SIZE, MINIMUM_FEASIBILITY_ELIGIBLE, SAMPLING_POOL_SIZE) == (16_384, 32, 65_536)
    assert PROTOCOL["selection_uses_rollout_outcomes"] is False
    assert PROTOCOL["selection_uses_simulator"] is False
    runner = (ROOT / "athena/run_post_norm_intervention_v2.py").read_text()
    assert "STRUCTURED_DESCRIPTIVE" not in runner


def test_v2_validators_reject_v1_and_unversioned_schemas() -> None:
    reject_v1_schema({"schema": "xvla-post-norm-intervention-rollout-v2"})
    with pytest.raises(RuntimeError, match="rejects v1"):
        reject_v1_schema({"schema": "xvla-post-norm-intervention-rollout-v1"})
    with pytest.raises(RuntimeError, match="rejects v1"):
        reject_v1_schema({})


def test_exact_check_inventories_and_bool_types() -> None:
    for inventory in (PREP_CHECK_KEYS, RUN_CHECK_KEYS, SCIENTIFIC_GATE_RESULT_KEYS):
        passing = {key: True for key in inventory}
        validate_check_inventory(passing, inventory, require_pass=True)
        with pytest.raises(RuntimeError, match="inventory"):
            validate_check_inventory({key: True for key in list(inventory)[:-1]}, inventory, require_pass=False)
        broken = dict(passing)
        broken[next(iter(inventory))] = 1
        with pytest.raises(RuntimeError, match="exact boolean"):
            validate_check_inventory(broken, inventory, require_pass=False)


def test_alpha_inventory_requires_exact_order_and_primary_grid() -> None:
    valid = {name: 0.5 for name in CONDITIONS}
    validate_alpha_inventory(valid)
    reversed_items = dict(reversed(list(valid.items())))
    validate_alpha_inventory(json.loads(json.dumps(reversed_items, sort_keys=True)))
    invalid = dict(valid)
    invalid[PRIMARY] = 0.333
    with pytest.raises(RuntimeError, match="grid"):
        validate_alpha_inventory(invalid)


def test_condition_order_is_deterministic_complete_and_v2_specific() -> None:
    left = condition_order(1, 2, 43)
    assert left == condition_order(1, 2, 43)
    assert set(left) == {"baseline", *CONDITIONS}
    assert len(left) == len(set(left))


def test_crossed_resampling_oracle_and_determinism() -> None:
    cube = np.fromfunction(lambda checkpoint, task, episode: 100 * checkpoint + 10 * task + episode, (3, 10, 10))
    checkpoints = np.asarray([2, 0])
    tasks = np.asarray([1, 3])
    episodes = np.asarray([[0, 2], [4, 5]])
    expected = np.mean([
        cube[checkpoint, task, episode]
        for task, draws in zip(tasks, episodes)
        for checkpoint in checkpoints for episode in draws
    ])
    assert crossed_resample_mean(cube, checkpoints, tasks, episodes) == expected
    constant = np.full((3, 10, 10), .125)
    left = crossed_bootstrap_distribution(constant, draws=19, seed=7)
    right = crossed_bootstrap_distribution(constant, draws=19, seed=7)
    np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(left, np.full(19, .125))


def test_canonical_hash_detects_nested_protocol_tampering() -> None:
    value = {"protocol": PROTOCOL, "passed": True}
    changed = {"protocol": {**PROTOCOL, "selection_uses_rollout_outcomes": True}, "passed": True}
    assert canonical_sha256(value) != canonical_sha256(changed)


def test_model_state_hash_is_shared_across_both_call_sites() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2)).double()
    assert prepare_model_hash is model_state_sha256
    assert rollout_model_hash is model_state_sha256
    assert prepare_model_hash(model) == rollout_model_hash(model)
    before = model_state_sha256(model)
    with torch.no_grad():
        model[0].weight[0, 0] += 1
    assert model_state_sha256(model) != before


def _structured_record() -> dict:
    diagnostics = {
        "aggregate_rms": 1.0, "continuous_rms": 0.9, "gripper_rms": 0.2,
        "gripper_flip_fraction": 0.01, "gripper_margin_reachable_fraction": 0.02,
    }
    return {
        name: {
            "deployed": False, "rank_per_horizon": 1,
            "selection_full_attenuation": dict(diagnostics),
            "certification_full_attenuation": dict(diagnostics),
        }
        for name in STRUCTURED_DESCRIPTIVE
    }


def test_structured_controls_survive_serialization_and_freeze_validation() -> None:
    roundtrip = json.loads(json.dumps(_structured_record(), sort_keys=True))
    validate_structured_descriptive(roundtrip)
    freeze_structured_validator(roundtrip)
    reordered = dict(reversed(list(roundtrip.items())))
    validate_structured_descriptive(reordered)
    validate_structured_metadata_order(list(STRUCTURED_DESCRIPTIVE))
    with pytest.raises(RuntimeError, match="metadata order"):
        validate_structured_metadata_order(list(reversed(STRUCTURED_DESCRIPTIVE)))
    extra = copy.deepcopy(roundtrip)
    extra[STRUCTURED_DESCRIPTIVE[0]]["unexpected"] = 1
    with pytest.raises(RuntimeError, match="field inventory"):
        validate_structured_descriptive(extra)


def _valid_v2_smoke() -> tuple[dict, dict, str]:
    freeze_sha = "f" * 64
    sources = {"xvla/example.py": "e" * 64}
    bundle_sha = "b" * 64
    model_sha = "c" * 64
    freeze = {"prepared": {"0": {"bundle_sha256": bundle_sha, "model_state_sha256": model_sha}}, "source_hashes": sources}
    rows = []
    for task in (0, 1):
        for episode in (0, 1):
            order = condition_order(0, task, episode)
            conditions = {
                name: {
                    "success": False, "steps": 8, "success_step": None,
                    "action_chunks": 1, "action_chunk_sha256": ["a" * 64],
                    "executed_actions_sha256": "d" * 64,
                    "all_gripper_actions_committed": True,
                    "positive_gripper_actions": 4, "negative_gripper_actions": 4,
                }
                for name in order
            }
            rows.append({
                "checkpoint_seed": 0, "task": task, "episode": episode,
                "reset_seed": task * 100 + episode, "init_state_sha256": "1" * 64,
                "start_physical_input_sha256": "2" * 64,
                "condition_order": order, "conditions": conditions,
            })
    smoke = {
        "schema": SCHEMA_RUN, "mode": "strict_smoke", "seed": 0,
        "task_bounds": [0, 2], "episode_bounds": [0, 2], "protocol": PROTOCOL,
        "claim_boundary": CLAIM_BOUNDARY,
        "identity": {
            "checkpoint_sha256": CHECKPOINTS[0]["sha256"], "cache_sha256": CACHE["sha256"],
            "bundle_sha256": bundle_sha, "freeze_sha256": freeze_sha,
            "smoke_sha256": None, "model_state_sha256": model_sha, "source_hashes": sources,
        },
        "runtime": PROTOCOL["runtime"],
        "task_identity": {
            "official_prompts": PROMPTS, "dataset_tasks": DATASET_TASKS,
            "dataset_to_official_task": DATASET_TO_OFFICIAL_TASK,
            "dataset_metadata": TASK_METADATA, "libero_runtime": LIBERO_RUNTIME,
        },
        "alpha_zero_maximum_action_error": 0.0,
        "checks": {name: True for name in RUN_CHECK_KEYS},
        "rows": rows, "elapsed_seconds": 1.0,
    }
    return smoke, freeze, freeze_sha


def test_smoke_validator_checks_exact_nested_inventories_hashes_and_metrics() -> None:
    smoke, freeze, freeze_sha = _valid_v2_smoke()
    smoke = json.loads(json.dumps(smoke, sort_keys=True))
    validate_smoke_prerequisite(smoke, freeze, freeze_sha)
    mutations = []
    value = copy.deepcopy(smoke); value["rows"][0]["extra"] = 1; mutations.append(value)
    value = copy.deepcopy(smoke); value["rows"][0]["conditions"]["baseline"]["extra"] = 1; mutations.append(value)
    value = copy.deepcopy(smoke); value["rows"][0]["conditions"]["baseline"]["executed_actions_sha256"] = "z" * 64; mutations.append(value)
    value = copy.deepcopy(smoke); value["rows"][0]["conditions"]["baseline"]["success"] = True; mutations.append(value)
    value = copy.deepcopy(smoke); value["alpha_zero_maximum_action_error"] = float("nan"); mutations.append(value)
    value = copy.deepcopy(smoke); value["elapsed_seconds"] = float("inf"); mutations.append(value)
    for malformed in mutations:
        with pytest.raises(RuntimeError):
            validate_smoke_prerequisite(malformed, freeze, freeze_sha)


def test_full_artifact_survives_production_sorted_json_roundtrip() -> None:
    smoke, _, _ = _valid_v2_smoke()
    row = copy.deepcopy(smoke["rows"][0])
    row["checkpoint_seed"] = 2
    row["task"] = 4
    row["episode"] = 45
    row["reset_seed"] = 445
    row["condition_order"] = condition_order(2, 4, 45)
    row["conditions"] = {name: copy.deepcopy(next(iter(row["conditions"].values()))) for name in row["condition_order"]}
    artifact = {"schema": SCHEMA_RUN, "mode": "full", "rows": [row]}
    roundtrip = json.loads(json.dumps(artifact, sort_keys=True))
    assert list(roundtrip["rows"][0]["conditions"]) == sorted(roundtrip["rows"][0]["conditions"])
    validate_rollout_row(roundtrip["rows"][0], 2)


def test_stage5_result_is_pinned_to_accepted_bytes() -> None:
    assert PROTOCOL["accepted_stage5_result_sha256"] == STAGE5_RESULT_SHA256
    assert file_sha256(ROOT / STAGE5_RESULT_PATH) == STAGE5_RESULT_SHA256


def test_freeze_requires_exact_runtime_and_task_identity() -> None:
    source = (ROOT / "athena/freeze_post_norm_intervention_v2.py").read_text()
    assert 'record.get("runtime") != PROTOCOL["runtime"]' in source
    assert 'validate_task_identity(record.get("task_identity"))' in source
    assert 'record.get("protocol") != PROTOCOL' in source


def test_v1_files_are_preserved_and_v2_paths_are_separate() -> None:
    for name in (
        "post_norm_intervention_v1_common.py", "prepare_post_norm_intervention_v1.py",
        "freeze_post_norm_intervention_v1.py", "run_post_norm_intervention_v1.py",
        "aggregate_post_norm_intervention_v1.py",
    ):
        assert (ROOT / "athena" / name).exists()
    v2_text = "\n".join(path.read_text() for path in (ROOT / "athena").glob("*post_norm_intervention_v2*"))
    assert "post_norm_intervention_v1_bundle" not in v2_text
    assert "post_norm_intervention_v1_smoke" not in v2_text


def test_freeze_precedes_every_simulator_job_and_preparation_has_no_env() -> None:
    launcher = (ROOT / "athena/launch_post_norm_intervention_v2.sh").read_text()
    assert 'smoke_job="$(sbatch --parsable --dependency="afterok:${freeze_job}"' in launcher
    assert 'full_job="$(sbatch --parsable --dependency="afterok:${smoke_job}"' in launcher
    preparation = (ROOT / "athena/prepare_post_norm_intervention_v2.py").read_text()
    freeze = (ROOT / "athena/freeze_post_norm_intervention_v2.py").read_text()
    assert "OffScreenRenderEnv" not in preparation
    assert "OffScreenRenderEnv" not in freeze
