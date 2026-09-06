from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import athena.aggregate_product_rational_results as aggregate
import athena.eval_product_rational_checkpoint as evaluator
import athena.prepare_product_rational_launch as prepare
import athena.product_rational_protocol as protocol
import athena.train_product_rational_checkpoint as trainer
from scripts.odt_direct_only_compliance import (
    DirectOnlyComplianceError,
    EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT,
    EXPECTED_RUNTIME_GUARD_ENTRYPOINTS,
    audit_direct_only_launch,
)
from xvla.models.vla import ChiVLA
from xvla.nn.normalization import RationalNorm


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_task_map_and_vocabulary_are_exact():
    class Task:
        def __init__(self, language: str):
            self.language = language

    class Suite:
        n_tasks = 10

        def get_task(self, index: int):
            return Task(protocol.OFFICIAL_TASK_LANGUAGES[index])

    tasks = protocol.task_languages(Suite())
    assert tasks == protocol.OFFICIAL_TASK_LANGUAGES
    assert protocol.canonical_sha256(tasks) == protocol.OFFICIAL_TASK_LANGUAGES_SHA256
    vocab, encode = protocol.build_vocab(tasks)
    assert protocol.canonical_sha256(vocab) == protocol.VOCAB_SHA256
    assert len(vocab) == 26
    assert len(encode(tasks[0])) == protocol.MAX_INSTRUCTION_LENGTH
    changed = Suite()
    changed.get_task = lambda index: Task(
        "changed" if index == 0 else protocol.OFFICIAL_TASK_LANGUAGES[index]
    )
    with pytest.raises(RuntimeError, match="frozen map"):
        protocol.task_languages(changed)


def test_evaluation_image_preprocessing_is_exact_rot180_and_single_scale():
    raw = np.zeros((protocol.RESOLUTION, protocol.RESOLUTION, 3), dtype=np.uint8)
    raw[0, 0] = (255, 128, 1)
    image = evaluator.preprocess_agentview_image({"agentview_image": raw})
    assert image.shape == (3, protocol.RESOLUTION, protocol.RESOLUTION)
    assert image.dtype == torch.float32
    assert torch.equal(image[:, 0, 0], torch.zeros(3))
    assert torch.equal(
        image[:, -1, -1], torch.tensor([1.0, 128.0 / 255.0, 1.0 / 255.0])
    )


def test_evaluation_environment_normalizes_torch_version_to_literal_string(
    monkeypatch,
):
    class VersionSubclass(str):
        pass

    monkeypatch.setattr(evaluator.torch, "__version__", VersionSubclass("2.7.1+cu126"))
    environment = evaluator.evaluation_environment_record("Quadro RTX 6000")
    assert type(environment["torch"]) is str
    assert environment["torch"] == "2.7.1+cu126"


def test_robot_state_uses_official_robosuite_quaternion_conversion():
    transform_utils = pytest.importorskip("robosuite.utils.transform_utils")
    quaternions = [
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    ]
    generator = np.random.default_rng(20260905)
    for _ in range(16):
        quaternion = generator.normal(size=4)
        quaternion /= float((quaternion * quaternion).sum() ** 0.5)
        quaternions.append(quaternion)
    for quaternion in quaternions:
        observation = {
            "robot0_eef_pos": np.asarray([0.1, -0.2, 0.3]),
            "robot0_eef_quat": quaternion,
            "robot0_gripper_qpos": np.asarray([0.4, -0.5]),
        }
        expected = np.concatenate(
            (
                observation["robot0_eef_pos"],
                transform_utils.quat2axisangle(quaternion),
                observation["robot0_gripper_qpos"],
            )
        ).astype(np.float32)
        assert np.array_equal(evaluator.robot_state(observation), expected)


def test_settle_transition_rejects_nonfinite_success_and_done():
    observation = {
        "agentview_image": np.zeros(
            (protocol.RESOLUTION, protocol.RESOLUTION, 3), dtype=np.uint8
        ),
        "robot0_eef_pos": np.zeros(3, dtype=np.float64),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float64),
    }
    evaluator.validate_settle_transition(
        observation, 0.0, False, task_index=0, episode_index=0, settle_index=0
    )
    nonfinite = dict(observation)
    nonfinite["robot0_eef_pos"] = np.asarray([float("nan"), 0.0, 0.0])
    with pytest.raises(RuntimeError, match="not entirely finite"):
        evaluator.validate_settle_transition(
            nonfinite, 0.0, False, task_index=0, episode_index=0, settle_index=0
        )
    with pytest.raises(RuntimeError, match="non-finite"):
        evaluator.validate_settle_transition(
            observation,
            float("nan"),
            False,
            task_index=0,
            episode_index=0,
            settle_index=0,
        )
    for reward, done in ((1.0, False), (0.0, True)):
        with pytest.raises(RuntimeError, match="success or termination"):
            evaluator.validate_settle_transition(
                observation,
                reward,
                done,
                task_index=0,
                episode_index=0,
                settle_index=0,
            )


def test_policy_transition_validator_rejects_every_malformed_terminal_surface():
    observation = {
        "agentview_image": np.zeros(
            (protocol.RESOLUTION, protocol.RESOLUTION, 3), dtype=np.uint8
        ),
        "robot0_eef_pos": np.zeros(3, dtype=np.float64),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float64),
    }
    observed, reward, done, info = evaluator.validate_environment_transition(
        (observation, 1.0, True, {}),
        phase="policy",
        task_index=0,
        episode_index=0,
        transition_index=0,
    )
    assert observed is observation
    assert reward == 1.0
    assert done is True
    assert info == {}
    _, _, numpy_done, _ = evaluator.validate_environment_transition(
        (observation, 0.0, np.bool_(False), {}),
        phase="policy",
        task_index=0,
        episode_index=0,
        transition_index=1,
    )
    assert type(numpy_done) is bool and numpy_done is False
    mutations = [
        ((observation, float("nan"), False, {}), "non-finite"),
        ((observation, np.asarray([0.0]), False, {}), "numeric scalar"),
        ((observation, 0.0, 0, {}), "scalar bool"),
        ((observation, 0.0, np.asarray(False), {}), "scalar bool"),
        ((observation, 0.0, np.asarray([False]), {}), "scalar bool"),
        ((observation, 0.0, False), "exact 4-tuple"),
    ]
    wrong_shape = dict(observation)
    wrong_shape["agentview_image"] = np.zeros((1, 1, 3), dtype=np.uint8)
    mutations.append(((wrong_shape, 0.0, False, {}), "shape or dtype"))
    terminal_nonfinite = dict(observation)
    terminal_nonfinite["robot0_eef_pos"] = np.asarray(
        [float("nan"), 0.0, 0.0], dtype=np.float64
    )
    mutations.append(((terminal_nonfinite, 1.0, True, {}), "not entirely finite"))
    for transition, message in mutations:
        with pytest.raises(RuntimeError, match=message):
            evaluator.validate_environment_transition(
                transition,
                phase="policy",
                task_index=0,
                episode_index=0,
                transition_index=0,
            )


def _valid_aggregate_evaluation_payload(monkeypatch):
    task_start, task_end = protocol.SHARDS[0]
    evaluation = protocol.evaluation_protocol(False, task_start, task_end)
    row_hashes = {
        task: [f"{task * 100 + index:064x}" for index in range(50)]
        for task in range(task_start, task_end)
    }
    for task, rows in row_hashes.items():
        monkeypatch.setitem(
            aggregate.OFFICIAL_INIT_STATE_ROW_HASH_LIST_SHA256,
            task,
            protocol.canonical_sha256(rows),
        )
    episodes = [
        {
            "seed": 0,
            "task_index": task,
            "episode_index": index,
            "initial_state_sha256": row_hashes[task][index],
            "settle_steps_completed": evaluation["settle_steps"],
            "settle_all_observations_and_rewards_finite": True,
            "settle_success_or_done_count": 0,
            "validated_settle_transition_count": evaluation["settle_steps"],
            "validated_policy_transition_count": evaluation["max_steps"],
            "success": False,
            "done": False,
            "steps": evaluation["max_steps"],
            "policy_chunk_count": evaluation["max_steps"]
            // evaluation["execution_horizon"],
            "elapsed_s": 1.0,
        }
        for task in range(task_start, task_end)
        for index in range(50)
    ]
    identities = [
        (
            row["seed"],
            row["task_index"],
            row["episode_index"],
            row["initial_state_sha256"],
        )
        for row in episodes
    ]
    chunks = sum(row["policy_chunk_count"] for row in episodes)
    sources = {"athena/example.py": "0" * 64}
    static = {
        "scope": "transitive_local_import_closure",
        "entrypoints": sorted(sources),
        "source_sha256": sources,
        "source_count": len(sources),
        "direct_qr_call_sites": 0,
        "direct_qr_required": False,
        "guarded_dormant_spectral_norm_sites": [],
        "duplicate_top_level_definition_sites": [],
        "prohibited_self_overlap_sites": [],
        "prohibited_calls_found": [],
        "call_site_count": 1,
    }
    task_mapping = {"exact": True}
    payload = {
        "schema": f"{protocol.SCHEMA}_evaluation_result",
        "recipe_version": protocol.RECIPE_VERSION,
        "mode": "full",
        "seed": 0,
        "checkpoint": "/frozen/checkpoint.pt",
        "checkpoint_sha256": "1" * 64,
        "metadata": "/frozen/metadata.json",
        "metadata_sha256": "2" * 64,
        "training_result": "/frozen/training.json",
        "training_result_sha256": "3" * 64,
        "precalibration_ema_state": "/frozen/precalibration.pt",
        "precalibration_ema_state_sha256": "4" * 64,
        "deployment_completion": "/frozen/completion.json",
        "deployment_completion_sha256": "5" * 64,
        "calibration_resume_proof_sha256": None,
        "training_result_closed": True,
        "source_manifest_sha256": "6" * 64,
        "manifest_bundle_sha256": "7" * 64,
        "protocol": evaluation,
        "task_protocol": {
            str(task): protocol.EXPECTED_TASK_PROTOCOL[task]
            for task in range(task_start, task_end)
        },
        "task_mapping": task_mapping,
        "episode_count": len(episodes),
        "success_count": 0,
        "overall": 0.0,
        "per_task": {
            str(task): 0.0 for task in range(task_start, task_end)
        },
        "episodes": episodes,
        "episode_identity_sha256": protocol.canonical_sha256(identities),
        "decode_boundary_proof": {
            "checks": {
                "public_action_equals_fixed_sign_decode": True,
                "finite_center": True,
                "finite_factors": True,
                "finite_gates": True,
                "finite_actions": True,
            },
            "zero_gate_logit_count": 0,
            "positive_sign_count": 0,
            "total_sign_count": protocol.N_FACTORS,
            "positive_sign_fraction": 0.0,
            "fixed_sign_rule": "+1 iff gate_logit > 0, else -1",
            "component_shapes": {
                "center": [1, protocol.ACTION_HORIZON * protocol.ACTION_DIM],
                "factors": [
                    1,
                    protocol.N_FACTORS,
                    protocol.ACTION_HORIZON * protocol.ACTION_DIM,
                ],
                "gates": [1, protocol.N_FACTORS],
            },
        },
        "decode_counts": {
            "zero_gate_logits": 0,
            "positive_signs": 0,
            "total_signs": chunks * protocol.N_FACTORS,
            "public_component_mismatch": 0,
        },
        "policy_chunk_count": chunks,
        "validated_settle_transition_count": len(episodes)
        * evaluation["settle_steps"],
        "validated_policy_transition_count": sum(row["steps"] for row in episodes),
        "validated_transition_count": len(episodes) * evaluation["settle_steps"]
        + sum(row["steps"] for row in episodes),
        "nonfinite_count": 0,
        "decode_failure_count": 0,
        "elapsed_s": 10.0,
        "environment": {
            "gpu": "Quadro RTX 6000",
            "inference_dtype": "float32",
            "matmul_precision": "highest",
            "python": "3.10.19",
            "torch": "2.7.1+cu126",
            "cuda": "12.6",
            "numpy": "1.26.4",
            "libero": "0.1.0",
            "robosuite": "1.4.1",
            "mujoco": "3.5.0",
            "mujoco_gl": "egl",
        },
        "transitive_direct_only_static_audit": static,
        "capability_simulator_boundary": protocol.CAPABILITY_SIMULATOR_BOUNDARY,
        "simulator_external_to_odt": True,
        "odt_runtime_compliance_claimed": False,
        "canonical_odt_runtime_guard_installed": False,
        "canonical_odt_numerical_compliance_claimed": False,
        "external_simulator_outside_weight_only_odt_closure": True,
        "claim_boundary": protocol.DEPLOYMENT_CLAIM_BOUNDARY,
    }
    arguments = {
        "path": Path("/frozen/eval.json"),
        "seed": 0,
        "task_start": task_start,
        "task_end": task_end,
        "manifest": {
            "manifest_bundle_sha256": "7" * 64,
            "source_closure": sources,
        },
        "manifest_sha": "6" * 64,
        "expected_task_mapping": task_mapping,
    }
    return payload, arguments


def test_aggregate_evaluation_consumer_rejects_protocol_mutations(monkeypatch):
    payload, arguments = _valid_aggregate_evaluation_payload(monkeypatch)
    assert aggregate.validate_evaluation_result(payload, **arguments) == payload[
        "episodes"
    ]
    mutations = []
    extra = json.loads(json.dumps(payload))
    extra["unreviewed"] = True
    mutations.append((extra, "top-level fields differ"))
    bool_top_seed = json.loads(json.dumps(payload))
    bool_top_seed["seed"] = False
    mutations.append((bool_top_seed, "failed frozen gates"))
    malformed = json.loads(json.dumps(payload))
    malformed["episodes"][0]["unreviewed"] = True
    mutations.append((malformed, "malformed episode"))
    bool_identity = json.loads(json.dumps(payload))
    bool_identity["episodes"][0]["seed"] = False
    mutations.append((bool_identity, "non-integer episode identity"))
    bool_success_count = json.loads(json.dumps(payload))
    bool_success_count["success_count"] = False
    mutations.append((bool_success_count, "success summary differs"))
    bool_per_task = json.loads(json.dumps(payload))
    bool_per_task["per_task"]["0"] = False
    mutations.append((bool_per_task, "per-task summary differs"))
    bool_settle_count = json.loads(json.dumps(payload))
    bool_settle_count["episodes"][0]["settle_success_or_done_count"] = False
    mutations.append((bool_settle_count, "non-integer transition counts"))
    bool_decode_count = json.loads(json.dumps(payload))
    bool_decode_count["decode_counts"]["public_component_mismatch"] = False
    mutations.append((bool_decode_count, "failed frozen gates"))
    zero_steps = json.loads(json.dumps(payload))
    zero_steps["episodes"][0]["steps"] = 0
    mutations.append((zero_steps, "invalid step count"))
    early = json.loads(json.dumps(payload))
    early["episodes"][0]["steps"] = 1
    early["episodes"][0]["policy_chunk_count"] = 1
    mutations.append((early, "early-stop semantics"))
    settle = json.loads(json.dumps(payload))
    settle["episodes"][0]["settle_success_or_done_count"] = 1
    mutations.append((settle, "settle sentinel"))
    policy_transition_count = json.loads(json.dumps(payload))
    policy_transition_count["episodes"][0]["validated_policy_transition_count"] -= 1
    mutations.append((policy_transition_count, "settle sentinel"))
    shard_transition_count = json.loads(json.dumps(payload))
    shard_transition_count["validated_transition_count"] -= 1
    mutations.append((shard_transition_count, "decode-count arithmetic differs"))
    order = json.loads(json.dumps(payload))
    order["episodes"][0], order["episodes"][1] = (
        order["episodes"][1],
        order["episodes"][0],
    )
    mutations.append((order, "episode order differs"))
    row_identity = json.loads(json.dumps(payload))
    row_identity["episodes"][0]["initial_state_sha256"] = "f" * 64
    row_identity["episode_identity_sha256"] = protocol.canonical_sha256(
        [
            (
                row["seed"],
                row["task_index"],
                row["episode_index"],
                row["initial_state_sha256"],
            )
            for row in row_identity["episodes"]
        ]
    )
    mutations.append((row_identity, "row authority differs"))
    decode = json.loads(json.dumps(payload))
    decode["decode_counts"]["total_signs"] += 1
    mutations.append((decode, "decode-count arithmetic differs"))
    mapping = json.loads(json.dumps(payload))
    mapping["task_mapping"] = {"exact": False}
    mutations.append((mapping, "failed frozen gates"))
    static_mutation = json.loads(json.dumps(payload))
    del static_mutation["transitive_direct_only_static_audit"][
        "duplicate_top_level_definition_sites"
    ]
    mutations.append((static_mutation, "failed frozen gates"))
    static_zero_alias = json.loads(json.dumps(payload))
    static_zero_alias["transitive_direct_only_static_audit"][
        "direct_qr_call_sites"
    ] = False
    mutations.append((static_zero_alias, "failed frozen gates"))
    decode_check_alias = json.loads(json.dumps(payload))
    decode_check_alias["decode_boundary_proof"]["checks"][
        "finite_center"
    ] = 1
    mutations.append((decode_check_alias, "decode boundary schema differs"))
    component_shape_alias = json.loads(json.dumps(payload))
    component_shape_alias["decode_boundary_proof"]["component_shapes"]["center"][0] = True
    mutations.append((component_shape_alias, "decode boundary schema differs"))
    environment_mutation = json.loads(json.dumps(payload))
    environment_mutation["environment"]["torch"] = "2.7.1"
    mutations.append((environment_mutation, "failed frozen gates"))
    for candidate, message in mutations:
        with pytest.raises(RuntimeError, match=message):
            aggregate.validate_evaluation_result(candidate, **arguments)


def test_aggregate_stored_guards_reject_numeric_boolean_aliases():
    sources = {"athena/example.py": "0" * 64}
    static = {
        "scope": "transitive_local_import_closure",
        "entrypoints": sorted(sources),
        "source_sha256": sources,
        "source_count": 1,
        "direct_qr_call_sites": 0,
        "direct_qr_required": False,
        "guarded_dormant_spectral_norm_sites": [],
        "duplicate_top_level_definition_sites": [],
        "prohibited_self_overlap_sites": [],
        "prohibited_calls_found": [],
        "call_site_count": 4,
    }
    runtime = {
        "installed": True,
        "patched_entrypoints": sorted(EXPECTED_RUNTIME_GUARD_ENTRYPOINTS),
        "patched_entrypoint_count": EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT,
        "allowed_call_count": 0,
        "allowed_calls": [],
        "prohibited_attempt_count": 0,
        "prohibited_attempts": [],
    }
    assert aggregate.stored_static_audit_closed(static, sources)
    assert aggregate.stored_runtime_guard_closed(runtime)
    assert protocol.static_audit_record_is_closed(
        static,
        expected_source_closure=sources,
        direct_qr_required=False,
        direct_qr_call_sites=0,
    )
    assert protocol.runtime_guard_record_is_closed(
        runtime,
        expected_entrypoints=EXPECTED_RUNTIME_GUARD_ENTRYPOINTS,
        expected_entrypoint_count=EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT,
    )
    for field, alias in (("source_count", True), ("direct_qr_call_sites", False)):
        candidate = copy.deepcopy(static)
        candidate[field] = alias
        assert not aggregate.stored_static_audit_closed(candidate, sources)
        assert not protocol.static_audit_record_is_closed(
            candidate,
            expected_source_closure=sources,
            direct_qr_required=False,
            direct_qr_call_sites=0,
        )
    for field, alias in (
        ("patched_entrypoint_count", True),
        ("allowed_call_count", False),
        ("prohibited_attempt_count", False),
    ):
        candidate = copy.deepcopy(runtime)
        candidate[field] = alias
        assert not aggregate.stored_runtime_guard_closed(candidate)
        assert not protocol.runtime_guard_record_is_closed(
            candidate,
            expected_entrypoints=EXPECTED_RUNTIME_GUARD_ENTRYPOINTS,
            expected_entrypoint_count=EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT,
        )


def test_raw_capability_evidence_reconstruction_executes_and_rejects_forged_pair(
    tmp_path: Path, monkeypatch
):
    run_root = tmp_path / "run"

    def write_json(path: Path, payload: dict) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        return protocol.file_sha256(path)

    sources = {"athena/example.py": "0" * 64}
    static = {
        "scope": "transitive_local_import_closure",
        "entrypoints": sorted(sources),
        "source_sha256": sources,
        "source_count": len(sources),
        "direct_qr_call_sites": 0,
        "direct_qr_required": False,
        "guarded_dormant_spectral_norm_sites": [],
        "duplicate_top_level_definition_sites": [],
        "prohibited_self_overlap_sites": [],
        "prohibited_calls_found": [],
        "call_site_count": 1,
    }
    runtime = {
        "installed": True,
        "patched_entrypoints": sorted(EXPECTED_RUNTIME_GUARD_ENTRYPOINTS),
        "patched_entrypoint_count": EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT,
        "allowed_call_count": 0,
        "allowed_calls": [],
        "prohibited_attempt_count": 0,
        "prohibited_attempts": [],
    }
    manifest = {
        "schema": f"{protocol.SCHEMA}_source_manifest",
        "source_closure": sources,
        "manifest_bundle_sha256": "a" * 64,
    }
    manifest_path = protocol.source_manifest_path(run_root)
    manifest_sha = write_json(manifest_path, manifest)
    task_rows = {
        task: [protocol.canonical_sha256([task, episode]) for episode in range(50)]
        for task in range(10)
    }
    for task, rows in task_rows.items():
        monkeypatch.setitem(
            aggregate.OFFICIAL_INIT_STATE_ROW_HASH_LIST_SHA256,
            task,
            protocol.canonical_sha256(rows),
        )
    task_mapping = protocol.expected_task_mapping_record()
    artifact_hashes: dict[str, dict[str, str]] = {}
    evaluation_hashes: dict[str, str] = {}
    seed_summaries: dict[str, dict] = {}
    for seed in protocol.SEEDS:
        checkpoint = protocol.checkpoint_path(run_root, seed, False)
        precalibration = protocol.precalibration_state_path(run_root, seed, False)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        precalibration.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        precalibration.write_bytes(f"precalibration-{seed}".encode())
        checkpoint_sha = protocol.file_sha256(checkpoint)
        precalibration_sha = protocol.file_sha256(precalibration)
        metadata_path = protocol.metadata_path(run_root, seed, False)
        training_path = protocol.training_result_path(run_root, seed, False)
        completion_path = protocol.deployment_completion_path(run_root, seed, False)
        metadata_payload = {
            "schema": protocol.SCHEMA,
            "seed": seed,
            "mode": "full",
            "checkpoint": checkpoint.as_posix(),
            "checkpoint_sha256": checkpoint_sha,
            "training_only_precalibration_ema_state": {"sha256": precalibration_sha},
            "source_manifest": {
                "sha256": manifest_sha,
                "manifest_bundle_sha256": manifest["manifest_bundle_sha256"],
            },
            "source_snapshot_start": sources,
            "source_snapshot_end": sources,
            "transitive_direct_only_static_audit": static,
            "end_transitive_direct_only_static_audit": static,
            "claim_boundary": protocol.DEPLOYMENT_CLAIM_BOUNDARY,
        }
        metadata_sha = write_json(metadata_path, metadata_payload)
        training_payload = {
            "schema": f"{protocol.SCHEMA}_training_result",
            "seed": seed,
            "mode": "full",
            "checkpoint": checkpoint.as_posix(),
            "checkpoint_sha256": checkpoint_sha,
            "metadata": metadata_path.as_posix(),
            "metadata_sha256": metadata_sha,
            "training_only_precalibration_ema_state": {"sha256": precalibration_sha},
            "source_manifest_sha256": manifest_sha,
            "manifest_bundle_sha256": manifest["manifest_bundle_sha256"],
            "transitive_direct_only_static_audit": static,
            "runtime_direct_only_guard_at_import": runtime,
            "runtime_direct_only_guard_final": runtime,
            "claim_boundary": protocol.DEPLOYMENT_CLAIM_BOUNDARY,
        }
        training_sha = write_json(training_path, training_payload)
        completion_payload = {
            "schema": protocol.DEPLOYMENT_COMPLETION_SCHEMA,
            "complete": True,
            "seed": seed,
            "mode": "full",
            "checkpoint": checkpoint.as_posix(),
            "checkpoint_sha256": checkpoint_sha,
            "metadata": metadata_path.as_posix(),
            "metadata_sha256": metadata_sha,
            "training_result": training_path.as_posix(),
            "training_result_sha256": training_sha,
            "precalibration_ema_state": precalibration.as_posix(),
            "precalibration_ema_state_sha256": precalibration_sha,
            "source_manifest_sha256": manifest_sha,
            "manifest_bundle_sha256": manifest["manifest_bundle_sha256"],
        }
        completion_payload["artifact_bundle_sha256"] = protocol.canonical_sha256(
            aggregate._completion_bound_fields(completion_payload)
        )
        completion_sha = write_json(completion_path, completion_payload)
        seed_artifacts = {
            "checkpoint": checkpoint_sha,
            "metadata": metadata_sha,
            "training_result": training_sha,
            "precalibration_ema_state": precalibration_sha,
            "deployment_completion": completion_sha,
        }
        artifact_hashes[str(seed)] = seed_artifacts
        for task_start, task_end in protocol.SHARDS:
            evaluation_protocol = protocol.evaluation_protocol(False, task_start, task_end)
            episodes = [
                {
                    "seed": seed,
                    "task_index": task,
                    "episode_index": episode,
                    "initial_state_sha256": task_rows[task][episode],
                    "settle_steps_completed": evaluation_protocol["settle_steps"],
                    "settle_all_observations_and_rewards_finite": True,
                    "settle_success_or_done_count": 0,
                    "validated_settle_transition_count": evaluation_protocol[
                        "settle_steps"
                    ],
                    "validated_policy_transition_count": 1,
                    "success": True,
                    "done": True,
                    "steps": 1,
                    "policy_chunk_count": 1,
                    "elapsed_s": 1.0,
                }
                for task in range(task_start, task_end)
                for episode in range(50)
            ]
            chunks = len(episodes)
            settle_count = len(episodes) * evaluation_protocol["settle_steps"]
            identities = [
                (
                    row["seed"],
                    row["task_index"],
                    row["episode_index"],
                    row["initial_state_sha256"],
                )
                for row in episodes
            ]
            eval_payload = {
                "schema": f"{protocol.SCHEMA}_evaluation_result",
                "recipe_version": protocol.RECIPE_VERSION,
                "mode": "full",
                "seed": seed,
                "checkpoint": checkpoint.as_posix(),
                "checkpoint_sha256": checkpoint_sha,
                "metadata": metadata_path.as_posix(),
                "metadata_sha256": metadata_sha,
                "training_result": training_path.as_posix(),
                "training_result_sha256": training_sha,
                "precalibration_ema_state": precalibration.as_posix(),
                "precalibration_ema_state_sha256": precalibration_sha,
                "deployment_completion": completion_path.as_posix(),
                "deployment_completion_sha256": completion_sha,
                "calibration_resume_proof_sha256": None,
                "training_result_closed": True,
                "source_manifest_sha256": manifest_sha,
                "manifest_bundle_sha256": manifest["manifest_bundle_sha256"],
                "protocol": evaluation_protocol,
                "task_protocol": {
                    str(task): protocol.EXPECTED_TASK_PROTOCOL[task]
                    for task in range(task_start, task_end)
                },
                "task_mapping": task_mapping,
                "episode_count": len(episodes),
                "success_count": len(episodes),
                "overall": 1.0,
                "per_task": {str(task): 1.0 for task in range(task_start, task_end)},
                "episodes": episodes,
                "episode_identity_sha256": protocol.canonical_sha256(identities),
                "decode_boundary_proof": {
                    "checks": {
                        "public_action_equals_fixed_sign_decode": True,
                        "finite_center": True,
                        "finite_factors": True,
                        "finite_gates": True,
                        "finite_actions": True,
                    },
                    "zero_gate_logit_count": 0,
                    "positive_sign_count": 0,
                    "total_sign_count": protocol.N_FACTORS,
                    "positive_sign_fraction": 0.0,
                    "fixed_sign_rule": "+1 iff gate_logit > 0, else -1",
                    "component_shapes": {
                        "center": [1, protocol.ACTION_HORIZON * protocol.ACTION_DIM],
                        "factors": [
                            1,
                            protocol.N_FACTORS,
                            protocol.ACTION_HORIZON * protocol.ACTION_DIM,
                        ],
                        "gates": [1, protocol.N_FACTORS],
                    },
                },
                "decode_counts": {
                    "zero_gate_logits": 0,
                    "positive_signs": 0,
                    "total_signs": chunks * protocol.N_FACTORS,
                    "public_component_mismatch": 0,
                },
                "policy_chunk_count": chunks,
                "validated_settle_transition_count": settle_count,
                "validated_policy_transition_count": len(episodes),
                "validated_transition_count": settle_count + len(episodes),
                "nonfinite_count": 0,
                "decode_failure_count": 0,
                "elapsed_s": 10.0,
                "environment": {
                    "gpu": "Quadro RTX 6000",
                    "inference_dtype": "float32",
                    "matmul_precision": "highest",
                    "python": "3.10.19",
                    "torch": "2.7.1+cu126",
                    "cuda": "12.6",
                    "numpy": "1.26.4",
                    "libero": "0.1.0",
                    "robosuite": "1.4.1",
                    "mujoco": "3.5.0",
                    "mujoco_gl": "egl",
                },
                "transitive_direct_only_static_audit": static,
                "capability_simulator_boundary": protocol.CAPABILITY_SIMULATOR_BOUNDARY,
                "simulator_external_to_odt": True,
                "odt_runtime_compliance_claimed": False,
                "canonical_odt_runtime_guard_installed": False,
                "canonical_odt_numerical_compliance_claimed": False,
                "external_simulator_outside_weight_only_odt_closure": True,
                "claim_boundary": protocol.DEPLOYMENT_CLAIM_BOUNDARY,
            }
            eval_path = protocol.evaluation_result_path(
                run_root, seed, task_start, task_end, False
            )
            evaluation_hashes[eval_path.as_posix()] = write_json(eval_path, eval_payload)
        seed_summaries[str(seed)] = {
            "success_count": 500,
            "episode_count": 500,
            "overall": 1.0,
            "per_task": {str(task): 1.0 for task in range(10)},
        }
    conditions = {key: True for key in aggregate._CAPABILITY_CONDITION_KEYS}
    transition_counts = {
        "settle": 15_000,
        "policy": 1_500,
        "total": 16_500,
        "expected_settle": 15_000,
        "episode_step_total": 1_500,
    }
    aggregate_payload = {
        "schema": f"{protocol.SCHEMA}_three_seed_aggregate",
        "recipe_version": protocol.RECIPE_VERSION,
        "source_manifest_sha256": manifest_sha,
        "manifest_bundle_sha256": manifest["manifest_bundle_sha256"],
        "evaluation_result_sha256": evaluation_hashes,
        "training_artifact_sha256": artifact_hashes,
        "seeds": list(protocol.SEEDS),
        "preregistered_primary_seed": 0,
        "preregistered_primary_floor": protocol.PRIMARY_CAPABILITY_FLOOR,
        "seed_summaries": seed_summaries,
        "three_seed_mean": 1.0,
        "three_seed_population_std": 0.0,
        "capability_gate_conditions": conditions,
        "capability_gate_pass": True,
        "validated_transition_counts": transition_counts,
        "task_protocol": {
            str(task): protocol.EXPECTED_TASK_PROTOCOL[task] for task in range(10)
        },
        "transitive_direct_only_static_audit": static,
        "runtime_direct_only_guard_at_import": runtime,
        "runtime_direct_only_guard_final": runtime,
        "claim_boundary": protocol.DEPLOYMENT_CLAIM_BOUNDARY,
    }
    aggregate_path = protocol.aggregate_result_path(run_root)
    aggregate_sha = write_json(aggregate_path, aggregate_payload)
    seed0 = artifact_hashes["0"]
    gate_payload = {
        "schema": f"{protocol.SCHEMA}_seed0_capability_gate",
        "recipe_version": protocol.RECIPE_VERSION,
        "preregistered_before_evaluation": True,
        "primary_seed": 0,
        "required_overall": protocol.PRIMARY_CAPABILITY_FLOOR,
        "observed_overall": 1.0,
        "conditions": conditions,
        "approved_for_capable_global_claim": True,
        "capability_condition_satisfied_for_future_composite_attestation": True,
        "exact_odt_composite_attestation_approved": False,
        "checkpoint": protocol.checkpoint_path(run_root, 0, False).as_posix(),
        "checkpoint_sha256": seed0["checkpoint"],
        "metadata_sha256": seed0["metadata"],
        "training_result_sha256": seed0["training_result"],
        "precalibration_ema_state_sha256": seed0["precalibration_ema_state"],
        "deployment_completion_sha256": seed0["deployment_completion"],
        "aggregate": aggregate_path.as_posix(),
        "aggregate_sha256": aggregate_sha,
        "source_manifest_sha256": manifest_sha,
        "manifest_bundle_sha256": manifest["manifest_bundle_sha256"],
        "does_not_select_best_seed_post_hoc": True,
        "exact_odt_result_required_separately": True,
        "claim_boundary": protocol.DEPLOYMENT_CLAIM_BOUNDARY,
    }
    gate_path = protocol.seed0_capability_gate_path(run_root)
    write_json(gate_path, gate_payload)
    result = aggregate.validate_capability_aggregate_evidence(
        run_root, manifest=manifest, manifest_sha256=manifest_sha
    )
    assert result["validated"] is True
    assert result["seed_summaries"] == seed_summaries
    aggregate.verify_capability_evidence_snapshots(result["snapshots"])

    truthy_nonbool = json.loads(json.dumps(aggregate_payload))
    truthy_nonbool["capability_gate_conditions"][
        "primary_seed_is_preregistered_seed0"
    ] = 1
    write_json(aggregate_path, truthy_nonbool)
    with pytest.raises(RuntimeError, match="raw reconstruction"):
        aggregate.validate_capability_aggregate_evidence(
            run_root, manifest=manifest, manifest_sha256=manifest_sha
        )

    aggregate_sha = write_json(aggregate_path, aggregate_payload)
    gate_payload["aggregate_sha256"] = aggregate_sha
    write_json(gate_path, gate_payload)
    gate_path.write_text('{"schema": NaN}\n')
    with pytest.raises(RuntimeError, match="strict finite JSON"):
        aggregate.validate_capability_aggregate_evidence(
            run_root, manifest=manifest, manifest_sha256=manifest_sha
        )

    write_json(gate_path, gate_payload)
    forged = dict(aggregate_payload)
    forged["three_seed_mean"] = 0.99
    aggregate_path.write_text(json.dumps(forged, sort_keys=True) + "\n")
    with pytest.raises(RuntimeError, match="raw reconstruction"):
        aggregate.validate_capability_aggregate_evidence(
            run_root, manifest=manifest, manifest_sha256=manifest_sha
        )


def test_training_and_deployment_topology_and_export_are_closed():
    vocab_size = 26
    training = ChiVLA(protocol.make_product_config(vocab_size, deployment=False))
    topology = protocol.validate_product_model(
        training, deployment=False, require_initialized_norms=False
    )
    assert topology["pade_site_count"] == 73
    assert topology["product_component_root_euclidean_width"] == 284
    assert topology["product_component_root_projective_width"] == 285
    assert topology["vision_encoder_class"] == "ChiViT"
    assert topology["vision_block_count"] == 4
    assert topology["joint_block_count"] == 8
    assert topology["all_vision_and_joint_blocks_residual"] is True
    state, stripped = protocol.deployment_state_from_training(training)
    assert stripped == ("teacher_head.bias", "teacher_head.weight")
    assert not any(key.startswith(("teacher_head.", "value_head.")) for key in state)
    deployment = ChiVLA(protocol.make_product_config(vocab_size, deployment=True))
    protocol.strict_load_deployment_state(deployment, state)
    deployed = protocol.validate_product_model(
        deployment, deployment=True, require_initialized_norms=False
    )
    assert deployed["linear_action_fallback_present"] is False
    assert deployed["training_only_teacher_present"] is False


def test_deployment_equivalence_records_exact_raw_and_public_output_digest():
    torch.manual_seed(20260905)
    training = ChiVLA(protocol.make_product_config(26, deployment=False))
    deployment_state, _ = protocol.deployment_state_from_training(training)
    deployment = ChiVLA(protocol.make_product_config(26, deployment=True))
    protocol.strict_load_deployment_state(deployment, deployment_state)
    image = torch.zeros(1, 3, protocol.RESOLUTION, protocol.RESOLUTION)
    instruction = torch.zeros(
        1, protocol.MAX_INSTRUCTION_LENGTH, dtype=torch.long
    )
    state = torch.zeros(1, protocol.STATE_DIM)
    proof = trainer.prove_deployment_equivalence(
        training, deployment, image, instruction, state
    )
    assert len(proof["output_tensor_sha256"]) == 64
    assert all(proof["bitwise_checks"].values())
    assert proof["max_abs_action_error"] == 0.0
    assert proof["max_abs_gate_error"] == 0.0


def test_topology_and_pade_mutations_fail_closed():
    model = ChiVLA(protocol.make_product_config(26, deployment=True))
    model.vision.blocks.blocks[0].residual = False
    with pytest.raises(RuntimeError, match="remapped, linearized, or non-residual"):
        protocol.validate_product_model(
            model, deployment=True, require_initialized_norms=False
        )
    model.vision.blocks.blocks[0].residual = True
    sites = protocol.active_rational_norms(model)
    sites[0][1].pb[1].add_(1.0)
    with pytest.raises(RuntimeError, match="changed the frozen"):
        protocol.validate_product_model(
            model, deployment=True, require_initialized_norms=False
        )
    sites[0][1].pb[1].sub_(1.0)
    with pytest.raises(RuntimeError, match="never initialized"):
        protocol.validate_product_model(
            model, deployment=True, require_initialized_norms=True
        )
    for _, site in sites:
        site.initialized.fill_(True)
        site.running_ms.fill_(1.25)
        site.freeze()
    protocol.validate_product_model(
        model, deployment=True, require_initialized_norms=True
    )
    sites[0][1].running_ms.fill_(protocol.PRODUCT_RUNNING_MS_IDENTITY_FLOOR / 2)
    with pytest.raises(RuntimeError, match="failed before forward"):
        sites[0][1](torch.ones(1, 4))
    sites[0][1].running_ms.fill_(1.25)
    sites[-1][1].running_ms.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="invalid running mean-square"):
        protocol.validate_product_model(
            model, deployment=True, require_initialized_norms=True
        )


def test_strict_deployment_load_rejects_training_only_and_missing_state():
    training = ChiVLA(protocol.make_product_config(26, deployment=False))
    state, _ = protocol.deployment_state_from_training(training)
    deployment = ChiVLA(protocol.make_product_config(26, deployment=True))
    contaminated = dict(state)
    contaminated["teacher_head.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="training-only"):
        protocol.strict_load_deployment_state(deployment, contaminated)
    missing = dict(state)
    missing.pop(next(key for key in missing if key.startswith("product_head.factors.0")))
    with pytest.raises(RuntimeError):
        protocol.strict_load_deployment_state(deployment, missing)


class CalibrationProbe(nn.Module):
    def __init__(self, *, storage_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.register_buffer("running_ms", torch.ones((), dtype=storage_dtype))
        self.register_buffer("initialized", torch.ones((), dtype=torch.bool))
        self.frozen = False
        self.eps = 1e-6

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class CalibrationToy(nn.Module):
    def __init__(
        self,
        *,
        first_dtype: torch.dtype = torch.float32,
        second_dtype: torch.dtype = torch.float32,
        compute_dtype: torch.dtype = torch.float64,
        multiplier: float = 1.0,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.first = CalibrationProbe(storage_dtype=first_dtype)
        self.second = CalibrationProbe(storage_dtype=second_dtype)
        self.raw_component = nn.Identity()
        self.compute_dtype = compute_dtype
        self.multiplier = multiplier

    def forward(self, image, instruction, state, embodiment):
        del instruction, state, embodiment
        value = image.flatten(1).to(self.compute_dtype) * self.weight.to(
            self.compute_dtype
        )
        first = self.first(value)
        second = self.second(first * self.multiplier)
        return self.raw_component(second), None


def install_calibration_toy(monkeypatch, model: CalibrationToy) -> None:
    monkeypatch.setattr(
        trainer,
        "active_rational_norms",
        lambda candidate: (("first", candidate.first), ("second", candidate.second)),
    )
    monkeypatch.setattr(
        trainer,
        "active_product_component_modules",
        lambda candidate: (("product_center", candidate.raw_component),),
    )
    monkeypatch.setattr(trainer, "FINAL_CALIBRATION_PASSES", 1)
    monkeypatch.setattr(trainer, "FINAL_CALIBRATION_BATCH_SIZE", 2)
    monkeypatch.setattr(trainer, "FINAL_CALIBRATION_AUDIT_MAX_RELATIVE_CHANGE", 100.0)
    monkeypatch.setattr(
        trainer,
        "install_product_running_ms_fail_closed_guards",
        lambda _model: {"guarded_module_count": 2},
    )


def calibration_inputs(dtype: torch.dtype = torch.float32):
    image = torch.arange(1, 17, dtype=dtype).reshape(4, 1, 2, 2)
    instruction = torch.zeros(4, 1, dtype=torch.long)
    state = torch.zeros(4, 1)
    return image, instruction, state


def test_checkpoint_publisher_serializes_once(monkeypatch, tmp_path: Path):
    calls = 0
    original = torch.save

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "save", counted)
    output = tmp_path / "checkpoint.pt"
    state = {"weight": torch.arange(4)}
    protocol.publish_checkpoint_exclusive(output, state)
    assert calls == 1
    assert torch.equal(torch.load(output, weights_only=True)["weight"], state["weight"])


def test_simultaneous_calibration_commits_together_and_audits(monkeypatch):
    model = CalibrationToy()
    model.first.running_ms.fill_(2.0)
    model.second.running_ms.fill_(3.0)
    initializers = {"first": 2.0, "second": 3.0}
    install_calibration_toy(monkeypatch, model)
    monkeypatch.setattr(trainer, "FINAL_CALIBRATION_PASSES", 2)
    image, instruction, state = calibration_inputs(torch.uint8)
    report = trainer.calibrate_live_rational_norms_simultaneously(
        model, image, instruction, state
    )
    assert report["update_passes"] == 2
    assert report["final_no_commit_audit_pass"] is True
    assert report["expected_calls_per_site"] == 6
    assert report["all_initialized_and_frozen"] is True
    assert report["initializer"] == protocol.CALIBRATION_INITIALIZER_LABEL
    assert report["initializer_running_ms_sha256"] == protocol.canonical_sha256(
        initializers
    )
    assert report["old_training_buffers_discarded"] is False
    assert report["simultaneous_snapshot_commit"] is True
    assert report["commit_protocol"] == (
        "all_candidates_validated_before_sequential_commit_with_exception_rollback"
    )
    assert report["commit_exception_rollback"] is True
    assert report["observation_copy_dtype"] == "float64"
    assert report["observation_copy_before_square"] is True
    assert report["accumulator_dtype"] == "float64"
    assert all(site.frozen and bool(site.initialized) for site in (model.first, model.second))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), 0.0])
def test_calibration_rejects_invalid_post_ema_initializer_without_mutation(
    monkeypatch, bad_value: float
):
    model = CalibrationToy()
    install_calibration_toy(monkeypatch, model)
    model.second.running_ms.fill_(bad_value)
    before = model.first.running_ms.clone()
    image, instruction, state = calibration_inputs()
    with pytest.raises(
        RuntimeError,
        match=r"invalid_post_ema_initializer.*pass_index=-1.*batch_index=-1.*site='second'",
    ):
        trainer.calibrate_live_rational_norms_simultaneously(
            model, image, instruction, state
        )
    assert torch.equal(model.first.running_ms, before)
    assert model.first.frozen is False


def test_calibration_reports_float64_overflow_site_and_batch(monkeypatch):
    model = CalibrationToy(multiplier=1e200)
    install_calibration_toy(monkeypatch, model)
    image, instruction, state = calibration_inputs(torch.float64)
    with pytest.raises(
        RuntimeError,
        match=(
            r"nonfinite_float64_square_sum.*pass_index=0.*batch_index=0.*"
            r"site='second'.*finite_amax"
        ),
    ):
        trainer.calibrate_live_rational_norms_simultaneously(
            model, image, instruction, state
        )


def test_finite_float32_above_square_limit_is_copied_before_square(monkeypatch):
    model = CalibrationToy(
        first_dtype=torch.float64,
        second_dtype=torch.float64,
        compute_dtype=torch.float32,
    )
    install_calibration_toy(monkeypatch, model)
    image = torch.full((4, 1, 2, 2), 1.0e22, dtype=torch.float32)
    instruction = torch.zeros(4, 1, dtype=torch.long)
    state = torch.zeros(4, 1)
    report = trainer.calibrate_live_rational_norms_simultaneously(
        model, image, instruction, state
    )
    first_candidate = report["pass_records"][0]["candidate_running_ms"]["first"]
    assert first_candidate > torch.finfo(torch.float32).max
    assert np.isfinite(first_candidate)
    assert report["observation_copy_dtype"] == "float64"
    assert report["observation_copy_before_square"] is True


class NonfiniteRawComponent(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.tensor(float("nan"), dtype=value.dtype, device=value.device)


class MaskedNonfiniteProductToy(CalibrationToy):
    def __init__(self):
        super().__init__()
        self.raw_component = NonfiniteRawComponent()

    def forward(self, image, instruction, state, embodiment):
        del instruction, state, embodiment
        value = image.flatten(1).to(self.compute_dtype) * self.weight.to(
            self.compute_dtype
        )
        first = self.first(value)
        second = self.second(first)
        raw_gate = self.raw_component(second)
        public = torch.where(raw_gate > 0, second, second)
        assert bool(torch.isfinite(public).all())
        return public, None


def test_nonfinite_raw_product_component_fails_even_when_public_action_is_finite(
    monkeypatch,
):
    model = MaskedNonfiniteProductToy()
    install_calibration_toy(monkeypatch, model)
    monkeypatch.setattr(
        trainer,
        "active_product_component_modules",
        lambda candidate: (("product_gate_0", candidate.raw_component),),
    )
    image, instruction, state = calibration_inputs()
    with pytest.raises(
        RuntimeError,
        match=r"nonfinite_raw_product_component.*pass_index=0.*batch_index=0.*site='product_gate_0'",
    ):
        trainer.calibrate_live_rational_norms_simultaneously(
            model, image, instruction, state
        )


class CoupledRationalCalibrationToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = RationalNorm(variant="pade", deg=2)
        self.second = RationalNorm(variant="pade", deg=2)
        self.raw_component = nn.Identity()
        for value, norm in ((2.0, self.first), (3.0, self.second)):
            norm.running_ms.fill_(value)
            norm.initialized.fill_(True)

    def forward(self, image, instruction, state, embodiment):
        del instruction, state, embodiment
        first = self.first(image.flatten(1))
        second = self.second(first * 0.75 + 0.125)
        return self.raw_component(second), None


def _manual_frozen_rational(norm: RationalNorm, value: torch.Tensor, anchor: float):
    mean_square = value.pow(2).mean(dim=-1, keepdim=True) + norm.eps
    anchor_tensor = torch.tensor(anchor, dtype=value.dtype, device=value.device)
    normalized = mean_square / anchor_tensor
    numerator = sum(norm.pa[index] * normalized ** index for index in range(norm.deg + 1))
    denominator = sum(norm.pb[index] * normalized ** index for index in range(norm.deg + 1))
    return value * (numerator / denominator) * torch.rsqrt(anchor_tensor)


def _manual_calibration_candidate(value: torch.Tensor, norm: RationalNorm) -> float:
    total = None
    count = 0
    for start in range(0, value.shape[0], 2):
        copied = value[start : start + 2].detach().to(torch.float64)
        contribution = copied.square().sum(dtype=torch.float64)
        total = contribution if total is None else total + contribution
        count += copied.numel()
    candidate = total / count + torch.tensor(norm.eps, dtype=torch.float64)
    return float(candidate.to(norm.running_ms.dtype).item())


def test_coupled_rational_sites_match_manual_jacobi_after_every_pass(monkeypatch):
    model = CoupledRationalCalibrationToy()
    monkeypatch.setattr(
        trainer,
        "active_rational_norms",
        lambda candidate: (("first", candidate.first), ("second", candidate.second)),
    )
    monkeypatch.setattr(
        trainer,
        "active_product_component_modules",
        lambda candidate: (("product_center", candidate.raw_component),),
    )
    monkeypatch.setattr(trainer, "FINAL_CALIBRATION_PASSES", 2)
    monkeypatch.setattr(trainer, "FINAL_CALIBRATION_BATCH_SIZE", 2)
    monkeypatch.setattr(trainer, "FINAL_CALIBRATION_AUDIT_MAX_RELATIVE_CHANGE", 100.0)
    monkeypatch.setattr(
        trainer,
        "install_product_running_ms_fail_closed_guards",
        lambda _model: {"guarded_module_count": 2},
    )
    image, instruction, state = calibration_inputs()
    normalized_image = image.float().div(255).flatten(1)
    anchors = {"first": 2.0, "second": 3.0}
    manual_records = []
    for pass_index in range(3):
        first_input = normalized_image
        first_output = _manual_frozen_rational(
            model.first, first_input, anchors["first"]
        )
        second_input = first_output * 0.75 + 0.125
        candidates = {
            "first": _manual_calibration_candidate(first_input, model.first),
            "second": _manual_calibration_candidate(second_input, model.second),
        }
        manual_records.append(candidates)
        if pass_index < 2:
            anchors = candidates
    report = trainer.calibrate_live_rational_norms_simultaneously(
        model, image, instruction, state
    )
    assert len(report["pass_records"]) == len(manual_records)
    for observed, expected in zip(report["pass_records"], manual_records, strict=True):
        assert observed["candidate_running_ms"] == expected
        assert observed["site_call_count"] == {"first": 2, "second": 2}
        assert observed["product_component_call_count"] == {"product_center": 2}
    assert report["final_running_ms"] == manual_records[-2]
    monkeypatch.setattr(protocol, "EXPECTED_PADE_SITE_COUNT", 2)
    monkeypatch.setattr(protocol, "FINAL_CALIBRATION_PASSES", 2)
    monkeypatch.setattr(protocol, "FINAL_CALIBRATION_BATCH_SIZE", 2)
    monkeypatch.setattr(
        protocol, "FINAL_CALIBRATION_AUDIT_MAX_RELATIVE_CHANGE", 100.0
    )
    monkeypatch.setattr(protocol, "PRODUCT_COMPONENT_SITE_NAMES", ("product_center",))
    attestation = protocol.validate_simultaneous_calibration_attestation(
        report,
        initial_values={"first": 2.0, "second": 3.0},
        final_values=report["final_running_ms"],
        sample_count=image.shape[0],
    )
    assert attestation["validated"] is True
    tampered = json.loads(json.dumps(report))
    tampered_record = tampered["pass_records"][0]
    tampered_record["relative_change_by_site"]["first"] += 1.0e-9
    tampered_record["relative_change_by_site_sha256"] = protocol.canonical_sha256(
        tampered_record["relative_change_by_site"]
    )
    with pytest.raises(RuntimeError, match="residual arithmetic differs"):
        protocol.validate_simultaneous_calibration_attestation(
            tampered,
            initial_values={"first": 2.0, "second": 3.0},
            final_values=report["final_running_ms"],
            sample_count=image.shape[0],
        )
    missing = json.loads(json.dumps(report))
    del missing["initializer_min"]
    with pytest.raises(RuntimeError, match="top-level fields differ"):
        protocol.validate_simultaneous_calibration_attestation(
            missing,
            initial_values={"first": 2.0, "second": 3.0},
            final_values=report["final_running_ms"],
            sample_count=image.shape[0],
        )
    extra = json.loads(json.dumps(report))
    extra["unreviewed_claim"] = True
    with pytest.raises(RuntimeError, match="top-level fields differ"):
        protocol.validate_simultaneous_calibration_attestation(
            extra,
            initial_values={"first": 2.0, "second": 3.0},
            final_values=report["final_running_ms"],
            sample_count=image.shape[0],
        )
    semantic_tampers = {
        "initializer_min": report["initializer_min"] + 1.0,
        "running_ms_required_min": report["running_ms_required_min"] * 2.0,
        "historical_floor_is_identity_for_every_accepted_forward": False,
        "invalid_anchor_rejected_before_frozen_forward": False,
        "initializer_and_every_candidate_floor_verified_before_use_or_commit": False,
    }
    for key, replacement in semantic_tampers.items():
        tampered_semantic = json.loads(json.dumps(report))
        tampered_semantic[key] = replacement
        with pytest.raises(RuntimeError, match="fixed gates failed"):
            protocol.validate_simultaneous_calibration_attestation(
                tampered_semantic,
                initial_values={"first": 2.0, "second": 3.0},
                final_values=report["final_running_ms"],
                sample_count=image.shape[0],
            )


def test_invalid_late_candidate_cannot_partially_commit(monkeypatch):
    model = CalibrationToy(second_dtype=torch.float16, multiplier=1e5)
    install_calibration_toy(monkeypatch, model)
    first_before = model.first.running_ms.clone()
    second_before = model.second.running_ms.clone()
    image, instruction, state = calibration_inputs()
    with pytest.raises(
        RuntimeError,
        match=r"invalid_candidate_scale.*pass_index=0.*site='second'",
    ):
        trainer.calibrate_live_rational_norms_simultaneously(
            model, image, instruction, state
        )
    assert torch.equal(model.first.running_ms, first_before)
    assert torch.equal(model.second.running_ms, second_before)


def test_commit_exception_restores_every_running_ms_from_snapshot(monkeypatch):
    model = CalibrationToy()
    model.first.running_ms.fill_(2.0)
    model.second.running_ms.fill_(3.0)
    install_calibration_toy(monkeypatch, model)
    before = {
        "first": model.first.running_ms.clone(),
        "second": model.second.running_ms.clone(),
    }

    def fail_guard_refresh(_model):
        raise RuntimeError("injected guard refresh failure")

    monkeypatch.setattr(
        trainer, "install_product_running_ms_fail_closed_guards", fail_guard_refresh
    )
    image, instruction, state = calibration_inputs()
    with pytest.raises(RuntimeError, match="all live running_ms buffers were restored"):
        trainer.calibrate_live_rational_norms_simultaneously(
            model, image, instruction, state
        )
    assert torch.equal(model.first.running_ms, before["first"])
    assert torch.equal(model.second.running_ms, before["second"])
    assert bool(model.first.initialized.item())
    assert bool(model.second.initialized.item())


def test_post_ema_validation_checks_parameters_buffers_and_initializers(monkeypatch):
    model = CalibrationToy()
    install_calibration_toy(monkeypatch, model)
    report = trainer.validate_post_ema_training_state(model)
    assert report["all_parameter_and_buffer_tensors_finite"] is True
    assert report["all_active_initializers_finite_positive_initialized"] is True
    model.weight.data.fill_(float("inf"))
    with pytest.raises(RuntimeError, match="Post-EMA parameter/buffer.*weight"):
        trainer.validate_post_ema_training_state(model)
    model.weight.data.fill_(1.0)
    model.second.initialized.fill_(False)
    with pytest.raises(RuntimeError, match="initializer.*site='second'"):
        trainer.validate_post_ema_training_state(model)


def test_real_model_retains_73_training_anchors_and_covers_all_raw_components(
    monkeypatch,
):
    model = ChiVLA(protocol.make_product_config(26, deployment=False))
    active = protocol.active_rational_norms(model)
    all_norms = [module for module in model.modules() if isinstance(module, RationalNorm)]
    assert len(all_norms) == protocol.EXPECTED_TOTAL_PADE_MODULE_COUNT == 74
    assert len(active) == protocol.EXPECTED_PADE_SITE_COUNT == 73
    active_ids = {id(module) for _, module in active}
    inactive = [module for module in all_norms if id(module) not in active_ids]
    assert inactive == [model.vision.norm_out]
    assert bool(inactive[0].initialized.item()) is False
    anchors = {}
    for index, (name, norm) in enumerate(active):
        value = 1.0 + index / 100.0
        norm.running_ms.fill_(value)
        norm.initialized.fill_(True)
        anchors[name] = float(norm.running_ms.item())
    validation = trainer.validate_post_ema_training_state(model)
    assert validation["initializer_running_ms"] == anchors
    assert validation["running_ms_fail_closed_guard"]["guarded_module_count"] == 74
    monkeypatch.setattr(trainer, "FINAL_CALIBRATION_PASSES", 1)
    monkeypatch.setattr(trainer, "FINAL_CALIBRATION_BATCH_SIZE", 1)
    monkeypatch.setattr(trainer, "FINAL_CALIBRATION_AUDIT_MAX_RELATIVE_CHANGE", 1.0e20)
    image = torch.zeros(1, 3, protocol.RESOLUTION, protocol.RESOLUTION)
    instruction = torch.zeros(1, protocol.MAX_INSTRUCTION_LENGTH, dtype=torch.long)
    state = torch.zeros(1, protocol.STATE_DIM)
    dormant_calls = 0

    def count_dormant(_module, _inputs):
        nonlocal dormant_calls
        dormant_calls += 1

    dormant_handle = model.vision.norm_out.register_forward_pre_hook(count_dormant)
    try:
        report = trainer.calibrate_live_rational_norms_simultaneously(
            model, image, instruction, state
        )
    finally:
        dormant_handle.remove()
    assert report["initializer_running_ms"] == anchors
    assert set(report["site_call_count"].values()) == {2}
    assert tuple(report["product_component_call_count"]) == (
        protocol.PRODUCT_COMPONENT_SITE_NAMES
    )
    assert set(report["product_component_call_count"].values()) == {2}
    assert report["raw_product_components_finite_every_batch"] is True
    assert dormant_calls == 0


def test_calibration_state_transition_allows_only_73_live_running_ms_buffers():
    model = ChiVLA(protocol.make_product_config(26, deployment=False))
    sites = protocol.active_rational_norms(model)
    for index, (_, norm) in enumerate(sites):
        norm.running_ms.fill_(1.0 + index / 100.0)
        norm.initialized.fill_(True)
        norm.freeze()
    precalibration = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    for index, (_, norm) in enumerate(sites):
        norm.running_ms.fill_(2.0 + index / 100.0)
    calibrated = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    deployment, _ = protocol.deployment_state_from_training(model)
    names = tuple(name for name, _ in sites)
    proof = protocol.calibration_state_transition_proof(
        precalibration,
        calibrated,
        deployment,
        active_site_names=names,
    )
    assert proof["only_active_running_ms_changed_during_calibration"] is True
    assert proof["all_learned_parameters_bitwise_unchanged_during_calibration"] is True
    assert proof["all_noncalibration_buffers_bitwise_unchanged_during_calibration"] is True
    assert proof["changed_training_state_keys"] == sorted(
        f"{name}.running_ms" for name in names
    )
    recomputed = protocol.validate_precalibration_to_deployment_transition(
        precalibration,
        deployment,
        active_site_names=names,
        claimed_proof=proof,
    )
    assert recomputed == proof["precalibration_to_deployment"]

    changed_parameter = dict(calibrated)
    parameter_key = next(key for key in changed_parameter if key.endswith(".weight"))
    changed_parameter[parameter_key] = changed_parameter[parameter_key] + 1
    with pytest.raises(RuntimeError, match="non-running_ms"):
        protocol.calibration_state_transition_proof(
            precalibration,
            changed_parameter,
            deployment,
            active_site_names=names,
        )

    changed_dormant = dict(deployment)
    changed_dormant["vision.norm_out.running_ms"] = (
        changed_dormant["vision.norm_out.running_ms"] + 1
    )
    with pytest.raises(RuntimeError, match="unauthorized state"):
        protocol.validate_precalibration_to_deployment_transition(
            precalibration,
            changed_dormant,
            active_site_names=names,
            claimed_proof=proof,
        )


def test_training_only_precalibration_publication_is_bound_and_deployment_rejected(
    tmp_path: Path,
):
    output = tmp_path / "training_only" / "precalibration.pt"
    state = {"weight": torch.arange(4, dtype=torch.float32)}
    initializer_values = {}
    for index in range(protocol.EXPECTED_PADE_SITE_COUNT):
        site = f"site_{index}"
        running = torch.tensor(1.0 + index / 100.0)
        value = float(running.item())
        initializer_values[site] = value
        state[f"{site}.running_ms"] = running
        state[f"{site}.initialized"] = torch.tensor(True)
    authorities = {
        "seed": 1,
        "mode": "full",
        "step_count": protocol.FULL_STEPS,
        "source_manifest_sha256": "1" * 64,
        "manifest_bundle_sha256": "2" * 64,
        "source_bundle_sha256": "3" * 64,
        "training_config_sha256": "4" * 64,
    }
    validation = {
        "all_parameter_and_buffer_tensors_finite": True,
        "all_active_initializers_finite_positive_initialized": True,
        "initializer": protocol.CALIBRATION_INITIALIZER_LABEL,
        "initializer_running_ms": initializer_values,
        "initializer_running_ms_sha256": protocol.canonical_sha256(
            initializer_values
        ),
    }
    training_metrics = {
        "completed_steps": protocol.FULL_STEPS,
        "last_loss": 0.25,
        "training_elapsed_s": 12.0,
        "peak_allocated_gb": 2.0,
        "peak_reserved_gb": 3.0,
    }
    identity = protocol.publish_training_only_precalibration_state_exclusive(
        output,
        state,
        post_ema_state_validation=validation,
        training_metrics=training_metrics,
        **authorities,
    )
    loaded = protocol.load_training_only_precalibration_state(
        output,
        expected_file_sha256=identity["sha256"],
        expected_seed=authorities["seed"],
        expected_mode=authorities["mode"],
        expected_step_count=authorities["step_count"],
        expected_source_manifest_sha256=authorities["source_manifest_sha256"],
        expected_manifest_bundle_sha256=authorities["manifest_bundle_sha256"],
        expected_source_bundle_sha256=authorities["source_bundle_sha256"],
        expected_training_config_sha256=authorities["training_config_sha256"],
    )
    assert torch.equal(loaded["training_state"]["weight"], state["weight"])
    assert loaded["artifact_role"] == protocol.PRECALIBRATION_ARTIFACT_ROLE
    assert loaded["deployment_eligible"] is False
    with pytest.raises(FileExistsError):
        protocol.publish_training_only_precalibration_state_exclusive(
            output,
            state,
            post_ema_state_validation=validation,
            training_metrics=training_metrics,
            **authorities,
        )
    deployment = ChiVLA(protocol.make_product_config(26, deployment=True))
    with pytest.raises(RuntimeError, match="training-only pre-calibration"):
        protocol.strict_load_deployment_state(deployment, loaded)
    internally_tampered = dict(loaded)
    internally_tampered["training_state"] = dict(loaded["training_state"])
    internally_tampered["training_state"]["weight"] = (
        internally_tampered["training_state"]["weight"] + 1
    )
    tampered_output = tmp_path / "internally_tampered.pt"
    torch.save(internally_tampered, tampered_output)
    with pytest.raises(RuntimeError, match="tensor state digest differs"):
        protocol.load_training_only_precalibration_state(
            tampered_output,
            expected_file_sha256=protocol.file_sha256(tampered_output),
            expected_seed=authorities["seed"],
            expected_mode=authorities["mode"],
            expected_step_count=authorities["step_count"],
            expected_source_manifest_sha256=authorities["source_manifest_sha256"],
            expected_manifest_bundle_sha256=authorities["manifest_bundle_sha256"],
            expected_source_bundle_sha256=authorities["source_bundle_sha256"],
            expected_training_config_sha256=authorities["training_config_sha256"],
        )
    output.chmod(0o644)
    with output.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RuntimeError, match="file digest differs"):
        protocol.load_training_only_precalibration_state(
            output,
            expected_file_sha256=identity["sha256"],
            expected_seed=authorities["seed"],
            expected_mode=authorities["mode"],
            expected_step_count=authorities["step_count"],
            expected_source_manifest_sha256=authorities["source_manifest_sha256"],
            expected_manifest_bundle_sha256=authorities["manifest_bundle_sha256"],
            expected_source_bundle_sha256=authorities["source_bundle_sha256"],
            expected_training_config_sha256=authorities["training_config_sha256"],
        )


def test_direct_and_resumed_calibration_are_bitwise_identical(monkeypatch, tmp_path: Path):
    direct = CalibrationToy()
    direct.first.running_ms.fill_(2.0)
    direct.second.running_ms.fill_(3.0)
    install_calibration_toy(monkeypatch, direct)
    monkeypatch.setattr(protocol, "EXPECTED_PADE_SITE_COUNT", 2)
    post_ema = trainer.validate_post_ema_training_state(direct)
    metrics = {
        "completed_steps": 10,
        "last_loss": 0.5,
        "training_elapsed_s": 1.0,
        "peak_allocated_gb": 0.0,
        "peak_reserved_gb": 0.0,
    }
    output = tmp_path / "training_only" / "resume.pt"
    identity = protocol.publish_training_only_precalibration_state_exclusive(
        output,
        direct.state_dict(),
        seed=0,
        mode="smoke",
        step_count=10,
        source_manifest_sha256="1" * 64,
        manifest_bundle_sha256="2" * 64,
        source_bundle_sha256="3" * 64,
        training_config_sha256="4" * 64,
        post_ema_state_validation=post_ema,
        training_metrics=metrics,
    )
    loaded = protocol.load_training_only_precalibration_state(
        output,
        expected_file_sha256=identity["sha256"],
        expected_seed=0,
        expected_mode="smoke",
        expected_step_count=10,
        expected_source_manifest_sha256="1" * 64,
        expected_manifest_bundle_sha256="2" * 64,
        expected_source_bundle_sha256="3" * 64,
        expected_training_config_sha256="4" * 64,
    )
    resumed = CalibrationToy()
    resumed.load_state_dict(loaded["training_state"], strict=True)
    assert protocol.tensor_state_sha256(resumed.state_dict()) == loaded[
        "training_state_sha256"
    ]
    image, instruction, state = calibration_inputs()
    direct_report = trainer.calibrate_live_rational_norms_simultaneously(
        direct, image, instruction, state
    )
    resumed_report = trainer.calibrate_live_rational_norms_simultaneously(
        resumed, image, instruction, state
    )
    embodiment = torch.zeros(image.shape[0], dtype=torch.long)
    direct_output, _ = direct(image, instruction, state, embodiment)
    resumed_output, _ = resumed(image, instruction, state, embodiment)
    assert torch.equal(direct_output, resumed_output)
    assert direct_report == resumed_report
    assert all(
        torch.equal(direct.state_dict()[key], value)
        for key, value in resumed.state_dict().items()
    )


def test_deployment_completion_marker_rejects_partial_and_tampered_publication(
    tmp_path: Path,
):
    run_root = tmp_path / "run"
    checkpoint = protocol.checkpoint_path(run_root, 0, True)
    metadata = protocol.metadata_path(run_root, 0, True)
    training_result = protocol.training_result_path(run_root, 0, True)
    precalibration = protocol.precalibration_state_path(run_root, 0, True)
    completion = protocol.deployment_completion_path(run_root, 0, True)
    for path, content in (
        (checkpoint, b"checkpoint"),
        (metadata, b"metadata"),
        (precalibration, b"precalibration"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    with pytest.raises(RuntimeError, match="missing training_result"):
        protocol.publish_deployment_completion_exclusive(
            completion,
            seed=0,
            mode="smoke",
            checkpoint=checkpoint,
            metadata=metadata,
            training_result=training_result,
            precalibration_ema_state=precalibration,
            source_manifest_sha256="1" * 64,
            manifest_bundle_sha256="2" * 64,
        )
    with pytest.raises(RuntimeError, match="missing or nonphysical"):
        protocol.load_and_validate_deployment_completion(
            completion,
            expected_seed=0,
            expected_mode="smoke",
            expected_checkpoint=checkpoint,
            expected_metadata=metadata,
            expected_training_result=training_result,
            expected_precalibration_ema_state=precalibration,
            expected_source_manifest_sha256="1" * 64,
            expected_manifest_bundle_sha256="2" * 64,
        )
    training_result.parent.mkdir(parents=True, exist_ok=True)
    training_result.write_bytes(b"result")
    published = protocol.publish_deployment_completion_exclusive(
        completion,
        seed=0,
        mode="smoke",
        checkpoint=checkpoint,
        metadata=metadata,
        training_result=training_result,
        precalibration_ema_state=precalibration,
        source_manifest_sha256="1" * 64,
        manifest_bundle_sha256="2" * 64,
    )
    loaded = protocol.load_and_validate_deployment_completion(
        completion,
        expected_seed=0,
        expected_mode="smoke",
        expected_checkpoint=checkpoint,
        expected_metadata=metadata,
        expected_training_result=training_result,
        expected_precalibration_ema_state=precalibration,
        expected_source_manifest_sha256="1" * 64,
        expected_manifest_bundle_sha256="2" * 64,
    )
    assert loaded == published
    metadata.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="metadata_sha256"):
        protocol.load_and_validate_deployment_completion(
            completion,
            expected_seed=0,
            expected_mode="smoke",
            expected_checkpoint=checkpoint,
            expected_metadata=metadata,
            expected_training_result=training_result,
            expected_precalibration_ema_state=precalibration,
            expected_source_manifest_sha256="1" * 64,
            expected_manifest_bundle_sha256="2" * 64,
        )


@pytest.mark.parametrize(
    ("resume", "digest"),
    [
        (True, None),
        (False, "1" * 64),
        (True, "A" * 64),
        (True, "1" * 63),
    ],
)
def test_resume_cli_requires_exact_external_lowercase_sha(resume, digest):
    with pytest.raises(ValueError):
        trainer.validate_resume_request(
            resume_precalibration=resume,
            precalibration_sha256=digest,
        )
    trainer.validate_resume_request(
        resume_precalibration=False, precalibration_sha256=None
    )
    trainer.validate_resume_request(
        resume_precalibration=True, precalibration_sha256="1" * 64
    )
    with pytest.raises(ValueError, match="Resume proof requires"):
        trainer.validate_resume_request(
            resume_precalibration=False,
            precalibration_sha256=None,
            resume_proof_only=True,
        )
    trainer.validate_resume_request(
        resume_precalibration=True,
        precalibration_sha256="1" * 64,
        resume_proof_only=True,
    )


def test_same_allocation_resume_proof_requires_exact_slurm_gpu_identity(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")
    current = trainer.slurm_gpu_allocation_record(
        gpu="NVIDIA RTX A6000", compute_capability=(8, 6)
    )
    trainer.validate_same_slurm_gpu_allocation(current, dict(current))
    for key, bad_value in (
        ("slurm_job_id", "54321"),
        ("slurm_job_gpus", "1"),
        ("cuda_visible_devices", "GPU-def"),
        ("gpu", "Quadro RTX 6000"),
        ("compute_capability", [7, 5]),
    ):
        changed = dict(current)
        changed[key] = bad_value
        with pytest.raises(RuntimeError, match="fresh smoke GPU allocation"):
            trainer.validate_same_slurm_gpu_allocation(current, changed)
    for missing_key in (
        "SLURM_JOB_ID",
        "SLURM_JOB_GPUS",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(missing_key)
        missing = trainer.slurm_gpu_allocation_record(
            gpu="NVIDIA RTX A6000", compute_capability=(8, 6)
        )
        with pytest.raises(RuntimeError, match="fresh smoke GPU allocation"):
            trainer.validate_same_slurm_gpu_allocation(missing, missing)
        monkeypatch.setenv(
            missing_key,
            {
                "SLURM_JOB_ID": "12345",
                "SLURM_JOB_GPUS": "0",
                "CUDA_VISIBLE_DEVICES": "GPU-abc",
            }[missing_key],
        )


def test_resume_proof_orchestration_requires_version_tracked_tensors():
    trainer.require_version_tracked_resume_context()
    with torch.no_grad():
        trainer.require_version_tracked_resume_context()
        source = ChiVLA(protocol.make_product_config(26, deployment=True))
        state = {key: value.detach().clone() for key, value in source.state_dict().items()}
        deployment = ChiVLA(protocol.make_product_config(26, deployment=True))
        protocol.strict_load_deployment_state(deployment, state)
        guarded = [
            module for module in deployment.modules() if isinstance(module, RationalNorm)
        ]
        assert len(guarded) == protocol.EXPECTED_TOTAL_PADE_MODULE_COUNT
        for module in guarded:
            assert type(module.running_ms._version) is int
    with torch.inference_mode(), pytest.raises(
        RuntimeError, match="version-tracked tensors"
    ):
        trainer.require_version_tracked_resume_context()


def test_resume_runtime_guard_uses_canonical_json_sequence_types():
    live_shape = {
        "installed": True,
        "patched_entrypoints": ["example"],
        "patched_entrypoint_count": 1,
        "allowed_call_count": 0,
        "allowed_calls": [],
        "prohibited_attempt_count": 0,
        "prohibited_attempts": (),
    }
    normalized = trainer.canonical_json_runtime_guard_record(live_shape)
    assert normalized["patched_entrypoints"] == ["example"]
    assert normalized["allowed_calls"] == []
    assert normalized["prohibited_attempts"] == []
    assert type(normalized["prohibited_attempts"]) is list
    assert live_shape["prohibited_attempts"] == ()
    malformed = dict(live_shape)
    malformed["prohibited_attempts"] = ""
    with pytest.raises(RuntimeError, match="not a sequence"):
        trainer.canonical_json_runtime_guard_record(malformed)


def test_resume_proof_validator_is_exact_and_recomputes_all_external_joins():
    source = {"athena/example.py": "0" * 64}
    static_audit = {
        "scope": "transitive_local_import_closure",
        "entrypoints": ["athena/example.py"],
        "source_sha256": source,
        "source_count": 1,
        "direct_qr_call_sites": 0,
        "direct_qr_required": False,
        "guarded_dormant_spectral_norm_sites": [],
        "duplicate_top_level_definition_sites": [],
        "prohibited_self_overlap_sites": [],
        "prohibited_calls_found": [],
        "call_site_count": 1,
    }
    runtime_guard = {
        "installed": True,
        "patched_entrypoints": sorted(EXPECTED_RUNTIME_GUARD_ENTRYPOINTS),
        "patched_entrypoint_count": EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT,
        "allowed_call_count": 0,
        "allowed_calls": [],
        "prohibited_attempt_count": 0,
        "prohibited_attempts": [],
    }
    allocation = {
        "gpu": "NVIDIA RTX A6000",
        "compute_capability": [8, 6],
        "slurm_job_id": "12345",
        "slurm_job_gpus": "0",
        "cuda_visible_devices": "GPU-abc",
    }
    replay = {
        "panel_size": 1,
        "bitwise_checks": {"public": True},
        "output_tensor_sha256": "a" * 64,
    }
    attestation = {"validated": True}
    transition = {"schema": "transition"}
    payload = {
        "schema": protocol.CALIBRATION_RESUME_PROOF_SCHEMA,
        "seed": 0,
        "mode": "smoke",
        "fresh_deployment_completion_sha256": "1" * 64,
        "fresh_deployment_completion_bundle_sha256": "2" * 64,
        "fresh_checkpoint_sha256": "3" * 64,
        "fresh_metadata_sha256": "4" * 64,
        "fresh_training_result_sha256": "5" * 64,
        "precalibration_ema_state_sha256": "6" * 64,
        "precalibration_training_state_sha256": "7" * 64,
        "external_sha_authority_supplied": True,
        "optimizer_steps_executed": 0,
        "same_slurm_gpu_allocation_as_fresh_smoke": True,
        "allocation": allocation,
        "calibration_elapsed_s": 1.0,
        "fresh_and_resume_calibration_report_bitwise_equal": True,
        "fresh_and_resume_final_running_ms_bitwise_equal": True,
        "fresh_and_resume_deployment_state_bitwise_equal": True,
        "fresh_and_resume_raw_components_and_actions_bitwise_equal": True,
        "fresh_output_tensor_sha256": "a" * 64,
        "resume_output_tensor_sha256": "a" * 64,
        "calibration_attestation": attestation,
        "calibration_state_transition": transition,
        "deployment_equivalence": replay,
        "removed_training_only_keys": ["teacher_head.bias", "teacher_head.weight"],
        "source_manifest_sha256": "8" * 64,
        "manifest_bundle_sha256": "9" * 64,
        "preflight_certificate_sha256": "b" * 64,
        "source_snapshot_start": source,
        "source_snapshot_end": source,
        "transitive_direct_only_static_audit": static_audit,
        "runtime_direct_only_guard": runtime_guard,
        "recovery_scope": protocol.CALIBRATION_RECOVERY_SCOPE,
        "cross_version_recovery_supported": False,
    }
    expected = {
        "expected_source_manifest_sha256": "8" * 64,
        "expected_manifest_bundle_sha256": "9" * 64,
        "expected_preflight_sha256": "b" * 64,
        "expected_source_closure": source,
        "expected_static_audit": static_audit,
        "expected_checkpoint_sha256": "3" * 64,
        "expected_metadata_sha256": "4" * 64,
        "expected_training_result_sha256": "5" * 64,
        "expected_precalibration_sha256": "6" * 64,
        "expected_precalibration_training_state_sha256": "7" * 64,
        "expected_completion_sha256": "1" * 64,
        "expected_completion_bundle_sha256": "2" * 64,
        "expected_training_environment": allocation,
        "expected_calibration_attestation": attestation,
        "expected_state_transition": transition,
        "expected_deployment_equivalence": replay,
    }
    validated = protocol.validate_same_allocation_calibration_resume_proof(
        payload, **expected
    )
    assert validated["validated"] is True
    assert all(validated["conditions"].values())
    extra = dict(payload, unbound=True)
    with pytest.raises(RuntimeError, match="fields differ"):
        protocol.validate_same_allocation_calibration_resume_proof(extra, **expected)
    changed = dict(payload, fresh_checkpoint_sha256="c" * 64)
    with pytest.raises(RuntimeError, match="artifact_hashes"):
        protocol.validate_same_allocation_calibration_resume_proof(changed, **expected)
    for field, replacement in (
        ("seed", False),
        ("optimizer_steps_executed", False),
    ):
        alias = json.loads(json.dumps(payload))
        alias[field] = replacement
        with pytest.raises(RuntimeError, match="strict gates"):
            protocol.validate_same_allocation_calibration_resume_proof(alias, **expected)
    for section, field, replacement in (
        ("deployment_equivalence", "panel_size", True),
        ("calibration_attestation", "validated", 1),
        ("transitive_direct_only_static_audit", "source_count", True),
        ("transitive_direct_only_static_audit", "direct_qr_call_sites", False),
        ("runtime_direct_only_guard", "allowed_call_count", False),
        ("runtime_direct_only_guard", "prohibited_attempt_count", False),
    ):
        alias = json.loads(json.dumps(payload))
        alias[section][field] = replacement
        with pytest.raises(RuntimeError, match="strict gates"):
            protocol.validate_same_allocation_calibration_resume_proof(alias, **expected)
    bitwise_alias = json.loads(json.dumps(payload))
    bitwise_alias["deployment_equivalence"]["bitwise_checks"]["public"] = 1
    with pytest.raises(RuntimeError, match="strict gates"):
        protocol.validate_same_allocation_calibration_resume_proof(
            bitwise_alias, **expected
        )
    for section, mutation in (
        (
            "transitive_direct_only_static_audit",
            lambda value: value.pop("duplicate_top_level_definition_sites"),
        ),
        (
            "transitive_direct_only_static_audit",
            lambda value: value.update(call_site_count=2),
        ),
        (
            "runtime_direct_only_guard",
            lambda value: value.pop("allowed_call_count"),
        ),
        (
            "runtime_direct_only_guard",
            lambda value: value.update(patched_entrypoint_count=0),
        ),
        (
            "runtime_direct_only_guard",
            lambda value: value.update(prohibited_attempts=["blocked"]),
        ),
    ):
        nested = json.loads(json.dumps(payload))
        mutation(nested[section])
        with pytest.raises(RuntimeError, match="strict gates"):
            protocol.validate_same_allocation_calibration_resume_proof(
                nested, **expected
            )


def test_resume_output_contract_allows_only_fixed_precalibration_artifact(tmp_path: Path):
    run_root = tmp_path / "run"
    artifact = protocol.precalibration_state_path(run_root, 0, True)
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"recovery")
    trainer.assert_own_outputs_absent(
        run_root, 0, True, resume_precalibration=True
    )
    with pytest.raises(FileExistsError, match="pre-calibration output"):
        trainer.assert_own_outputs_absent(
            run_root, 0, True, resume_precalibration=False
        )
    protocol.checkpoint_path(run_root, 0, True).parent.mkdir(parents=True)
    protocol.checkpoint_path(run_root, 0, True).write_bytes(b"deployment")
    with pytest.raises(FileExistsError, match="existing seed outputs"):
        trainer.assert_own_outputs_absent(
            run_root, 0, True, resume_precalibration=True
        )


def test_product_bilinear_path_does_not_import_unstaged_tree_module():
    code = """
import sys
from xvla.nn.block import ChiTransformerBlock
assert 'xvla.nn.tree_mixing' not in sys.modules
ChiTransformerBlock(8, 2, ffn_rank=12, attn='bilinear', ffn='bilinear')
assert 'xvla.nn.tree_mixing' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1"},
    )


def test_direct_call_requirement_can_only_disable_that_one_gate(tmp_path: Path):
    clean = tmp_path / "clean.py"
    clean.write_text("value = 1\n")
    with pytest.raises(DirectOnlyComplianceError, match="no_direct_qr_call_site"):
        audit_direct_only_launch(tmp_path, (clean,))
    report = audit_direct_only_launch(tmp_path, (clean,), require_direct_qr=False)
    assert report["direct_qr_required"] is False
    assert report["direct_qr_call_sites"] == 0
    assert report["prohibited_calls_found"] == []
    blocked_name = "".join(("s", "v", "d"))
    blocked = tmp_path / "blocked.py"
    blocked.write_text(f"import torch\ndef f(x):\n    return torch.linalg.{blocked_name}(x)\n")
    with pytest.raises(DirectOnlyComplianceError):
        audit_direct_only_launch(tmp_path, (blocked,), require_direct_qr=False)


def test_full_python_and_launch_closures_are_clean_and_exact():
    report = audit_direct_only_launch(
        ROOT,
        tuple(ROOT / relative for relative in protocol.SOURCE_CLOSURE),
        require_direct_qr=False,
    )
    assert report["source_sha256"] == protocol.source_snapshot()
    assert report["prohibited_calls_found"] == []
    assert report["prohibited_self_overlap_sites"] == []
    assert report["guarded_dormant_spectral_norm_sites"] == []
    assert "xvla/nn/tree_mixing.py" not in report["source_sha256"]
    assert prepare.audit_text_launch_closure()["blocked_route_occurrences"] == []
    assert trainer._EXPECTED_SOURCE_CLOSURE == protocol.SOURCE_CLOSURE
    assert evaluator._EXPECTED_SOURCE_CLOSURE == protocol.SOURCE_CLOSURE
    assert aggregate._EXPECTED_SOURCE_CLOSURE == protocol.SOURCE_CLOSURE
    assert trainer._EXPECTED_AUTHENTICATED_INPUTS == protocol.AUTHENTICATED_INPUTS
    assert evaluator._EXPECTED_AUTHENTICATED_INPUTS == protocol.AUTHENTICATED_INPUTS
    assert aggregate._EXPECTED_AUTHENTICATED_INPUTS == protocol.AUTHENTICATED_INPUTS
    assert trainer._EXPECTED_LAUNCH_CLOSURE == protocol.LAUNCH_CLOSURE
    assert evaluator._EXPECTED_LAUNCH_CLOSURE == protocol.LAUNCH_CLOSURE
    assert aggregate._EXPECTED_LAUNCH_CLOSURE == protocol.LAUNCH_CLOSURE


def test_manifest_roundtrip_binds_every_exact_section(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    written = protocol.write_source_manifest_exclusive(manifest_path, ROOT)
    assert protocol.load_and_verify_source_manifest(manifest_path, ROOT) == written
    assert tuple(sorted(written["source_closure"])) == tuple(
        sorted(protocol.SOURCE_CLOSURE)
    )
    assert tuple(sorted(written["authenticated_inputs"])) == tuple(
        sorted(protocol.AUTHENTICATED_INPUTS)
    )
    assert tuple(sorted(written["launch_closure"])) == tuple(
        sorted(protocol.LAUNCH_CLOSURE)
    )
    incomplete = dict(written)
    incomplete["source_closure"] = {
        "athena/train_product_rational_checkpoint.py": "0" * 64
    }
    bad_path = tmp_path / "incomplete.json"
    bad_path.write_text(json.dumps(incomplete))
    with pytest.raises(RuntimeError, match="source_closure differs"):
        trainer._verify_preimport_source_manifest(
            ["--source-manifest", str(bad_path)]
        )


def test_cli_authority_options_reject_duplicates_and_abbreviations(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--run-root",
            "a",
            "--run-root",
            "b",
            "--source-manifest",
            "m",
            "--mode",
            "smoke",
            "--preflight-only",
        ],
    )
    with pytest.raises(SystemExit):
        trainer.parse_args()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--run-root",
            "a",
            "--source-manifest",
            "m",
            "--mode",
            "smoke",
            "--seed",
            "0",
            "--task-start",
            "0",
            "--task-end",
            "1",
            "--task-end",
            "1",
        ],
    )
    with pytest.raises(SystemExit):
        evaluator.parse_args()
    monkeypatch.setattr(sys, "argv", ["train", "--run-r", "a"])
    with pytest.raises(SystemExit):
        trainer.parse_args()


def test_exclusive_json_rejects_broken_link(tmp_path: Path):
    output = tmp_path / "result.json"
    output.symlink_to(tmp_path / "missing-target")
    with pytest.raises(FileExistsError):
        protocol.write_json_exclusive(output, {"ok": True})


def test_launch_shell_files_parse():
    for relative in protocol.LAUNCH_CLOSURE:
        subprocess.run(["bash", "-n", str(ROOT / relative)], check=True)
    train_wrapper = (
        ROOT / "athena/slurm_product_rational_train.sbatch"
    ).read_text()
    assert "--resume-proof-only" in train_wrapper
    assert 'if [[ "$mode" == smoke ]]' in train_wrapper
    submit = (ROOT / "athena/submit_product_rational_v2.sh").read_text()
    assert "--time=02:00:00" in submit


def test_locked_manifest_and_writable_gate_submission_contract(tmp_path: Path):
    manifest = tmp_path / "manifest"
    gates = tmp_path / "gates"
    manifest.mkdir(mode=0o755)
    gates.mkdir(mode=0o755)
    manifest.chmod(0o555)
    blocked = subprocess.run(["mkdir", str(manifest / "submission.lock")])
    assert blocked.returncode != 0
    subprocess.run(["mkdir", str(gates / "submission.lock")], check=True)
    submit = (ROOT / "athena/submit_product_rational_v2.sh").read_text()
    assert 'mkdir "$run_root/gates/submission.lock"' in submit
    assert 'mkdir "$run_root/manifest/submission.lock"' not in submit
    stage = (ROOT / "athena/stage_product_rational_launch.py").read_text()
    assert '"training_only"' in stage


def test_protocol_has_no_duplicate_top_level_definitions():
    import ast

    tree = ast.parse((ROOT / "athena/product_rational_protocol.py").read_text())
    definitions: dict[str, list[int]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.setdefault(node.name, []).append(node.lineno)
    assert {name: lines for name, lines in definitions.items() if len(lines) > 1} == {}
