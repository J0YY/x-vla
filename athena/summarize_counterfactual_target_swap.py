#!/usr/bin/env python3
"""Strictly validate and score the frozen counterfactual BDDL target-swap runs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from athena.counterfactual_target_swap_common import (
    CACHE,
    CHECKPOINTS,
    FROZEN_GATES,
    MANIFEST_PATH,
    PROMPTS,
    PROTOCOL,
    PROVENANCE,
    RECEPTACLE_SYMBOL,
    SCHEMA_RUN,
    SCHEMA_SUMMARY,
    SMOKE_RESULT_PATH,
    SUMMARY_RESULT_PATH,
    decode_array,
    file_sha256,
    hash_array,
    load_json,
    source_hashes,
    validate_hex_digest,
    validate_libero_runtime,
    validate_manifest,
    validate_specificity_summary,
    write_json,
)
from athena.run_xvla_experiment import load_suite, official_init_states, task_languages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--specificity-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_physical(value: dict[str, Any], expected_bodies: set[str]) -> str:
    validate_hex_digest(value.get("combined_sha256"), "physical input")
    components = value.get("component_sha256")
    expected_components = {
        "mujoco_state",
        "raw_agentview_image",
        "model_image",
        "model_robot_state",
        "all_robot_observations",
        "all_object_observations",
    }
    if not isinstance(components, dict) or set(components) != expected_components:
        raise RuntimeError("Physical component digest set is incomplete")
    for label, digest in components.items():
        validate_hex_digest(digest, f"physical component {label}")
    per_object = value.get("per_object_sha256")
    if not isinstance(per_object, dict) or set(per_object) != expected_bodies:
        raise RuntimeError("Per-object physical digest set is incomplete")
    for body, digest in per_object.items():
        validate_hex_digest(digest, f"object {body}")
    robot_fields = value.get("robot_field_names")
    if not isinstance(robot_fields, list) or not robot_fields or len(set(robot_fields)) != len(robot_fields):
        raise RuntimeError("Robot physical field catalog is invalid")
    object_fields = value.get("object_field_names")
    if not isinstance(object_fields, dict) or set(object_fields) != expected_bodies:
        raise RuntimeError("Object physical field catalog is invalid")
    for body, fields in object_fields.items():
        if not isinstance(fields, list) or f"{body}_pos" not in fields or f"{body}_quat" not in fields:
            raise RuntimeError(f"Object {body} field catalog lacks position or quaternion")
    return str(value["combined_sha256"])


def validate_rollout(run: dict[str, Any], prompt_id: int, instruction_ids: list[int], expected_start: dict[str, Any], expected_bodies: set[str]) -> bool:
    if int(run.get("prompt_id", -1)) != prompt_id or run.get("instruction_ids") != instruction_ids:
        raise RuntimeError("Rollout prompt identity differs from frozen catalog")
    if run.get("start_physical") != expected_start:
        raise RuntimeError("Paired rollout start physical state differs")
    validate_physical(run["start_physical"], expected_bodies)
    validate_physical(run["final_physical"], expected_bodies)
    steps = int(run.get("steps", -1))
    if not 1 <= steps <= PROTOCOL["max_steps"]:
        raise RuntimeError("Rollout step count is outside the frozen horizon")
    actions = decode_array(run["executed_actions"])
    rewards = decode_array(run["rewards"])
    dones = decode_array(run["dones"])
    if actions.dtype != np.dtype("<f8") or actions.shape != (steps, 7) or not np.isfinite(actions).all():
        raise RuntimeError("Executed action trace is malformed")
    if rewards.dtype != np.dtype("<f8") or rewards.shape != (steps,) or not np.isfinite(rewards).all():
        raise RuntimeError("Reward trace is malformed")
    if dones.dtype != np.dtype("|b1") or dones.shape != (steps,):
        raise RuntimeError("Done trace is malformed")
    chunks = run.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise RuntimeError("Action-chunk trace is absent")
    reproduced = []
    expected_start_step = 0
    for chunk_row in chunks:
        if int(chunk_row.get("start_step", -1)) != expected_start_step:
            raise RuntimeError("Action-chunk start-step mapping is discontinuous")
        executed_rows = int(chunk_row.get("executed_rows", -1))
        if not 1 <= executed_rows <= PROTOCOL["execution_horizon"]:
            raise RuntimeError("Action-chunk executed-row count is invalid")
        chunk = decode_array(chunk_row["prediction"])
        if chunk.dtype != np.dtype("<f4") or chunk.shape != (8, 7) or not np.isfinite(chunk).all():
            raise RuntimeError("Predicted action chunk is malformed")
        for action in chunk[:executed_rows]:
            row = np.asarray(action, dtype=np.float64).copy()
            row[-1] = 1.0 if row[-1] > 0 else -1.0
            reproduced.append(row)
        expected_start_step += executed_rows
    reproduced_actions = np.asarray(reproduced, dtype=np.float64).reshape(-1, 7)
    if expected_start_step != steps or not np.array_equal(reproduced_actions, actions):
        raise RuntimeError("Executed action trace does not reproduce from chunks")
    if not np.all(np.isin(actions[:, -1], [-1.0, 1.0])):
        raise RuntimeError("Executed gripper actions were not thresholded to signs")
    success = bool(np.any(rewards > 0))
    if run.get("success") is not success:
        raise RuntimeError("Success flag does not reproduce from reward trace")
    success_step = int(np.argmax(rewards > 0) + 1) if success else None
    if run.get("success_step") != success_step:
        raise RuntimeError("Success step does not reproduce from reward trace")
    if np.any(dones[:-1]):
        raise RuntimeError("Rollout continued after an environment done signal")
    if steps < PROTOCOL["max_steps"] and not success and not bool(dones[-1]):
        raise RuntimeError("Unsuccessful rollout stopped before horizon without done")
    return success


def validate_result_identity(result: dict[str, Any], mode: str, manifest_sha: str, specificity_sha: str, provenance_sha: str, manifest_libero_runtime: dict[str, Any], expected_sources: dict[str, str]) -> int:
    if result.get("schema") != SCHEMA_RUN or result.get("mode") != mode:
        raise RuntimeError(f"Result is not a {mode} target-swap run")
    if result.get("protocol") != PROTOCOL or result.get("frozen_gates") != FROZEN_GATES:
        raise RuntimeError("Run protocol or gates differ")
    prerequisite = result.get("prerequisite_gate", {})
    if prerequisite != {
        "specificity_summary_sha256": specificity_sha,
        "identity_validated": True,
        "overall_pass": True,
        "claim_eligible": True,
    }:
        raise RuntimeError("Run prerequisite specificity gate is invalid")
    identity = result.get("identity", {})
    seed = int(identity.get("checkpoint_seed", -1))
    if seed not in CHECKPOINTS:
        raise RuntimeError("Run checkpoint seed is invalid")
    expected_checkpoint = CHECKPOINTS[seed]
    if Path(str(identity.get("checkpoint", ""))).name != Path(expected_checkpoint["path"]).name or identity.get("checkpoint_sha256") != expected_checkpoint["sha256"]:
        raise RuntimeError("Run checkpoint identity differs")
    if identity.get("cache_sha256") != CACHE["sha256"]:
        raise RuntimeError("Run cache identity differs")
    if Path(str(identity.get("cache", ""))).name != Path(CACHE["path"]).name:
        raise RuntimeError("Run cache path differs")
    if str(identity.get("provenance_job_id")) != PROVENANCE["job_id"]:
        raise RuntimeError("Run provenance job identity differs")
    if identity.get("provenance_result_sha256") != provenance_sha or Path(str(identity.get("provenance_result", ""))).name != Path(PROVENANCE["path"]).name:
        raise RuntimeError("Run provenance result identity differs")
    if identity.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("Run manifest identity differs")
    if identity.get("source_sha256_start") != expected_sources or identity.get("source_sha256_end") != expected_sources:
        raise RuntimeError("Run source identities are stale")
    runtime = result.get("runtime", {})
    if runtime.get("gpu") != PROTOCOL["gpu_name"]:
        raise RuntimeError("Run GPU identity differs from frozen A6000")
    if runtime.get("gpu_capability") != PROTOCOL["gpu_compute_capability"]:
        raise RuntimeError("Run GPU compute capability differs from frozen A6000")
    if runtime.get("environment", {}).get("XVLA_FROZEN_EVALUATION_GPU_FAMILY") != PROTOCOL["gpu_family"]:
        raise RuntimeError("Run GPU-family environment declaration differs")
    if runtime.get("environment", {}).get("MUJOCO_GL") != "egl" or runtime.get("environment", {}).get("PYOPENGL_PLATFORM") != "egl":
        raise RuntimeError("Run rendering environment differs")
    if runtime.get("environment", {}).get("CUBLAS_WORKSPACE_CONFIG") != PROTOCOL["cublas_workspace_config"]:
        raise RuntimeError("Run deterministic cuBLAS workspace configuration differs")
    if result.get("evaluation", {}).get("matmul_precision") != "highest":
        raise RuntimeError("Run matmul precision is not highest")
    if runtime.get("deterministic_algorithms") is not True or runtime.get("cudnn_benchmark") is not False:
        raise RuntimeError("Run deterministic execution flags differ")
    for label in ("python", "torch", "torch_build", "cuda", "numpy", "pillow", "mujoco", "robosuite"):
        if not isinstance(runtime.get(label), str) or not runtime[label]:
            raise RuntimeError(f"Run runtime identity is missing {label}")
    if {label: runtime[label] for label in PROTOCOL["runtime_versions"]} != PROTOCOL["runtime_versions"]:
        raise RuntimeError("Run software versions differ from the frozen runtime")
    if runtime.get("libero_source_start") != manifest_libero_runtime or runtime.get("libero_source_end") != manifest_libero_runtime:
        raise RuntimeError("Run LIBERO source identity differs from the manifest")
    if not runtime.get("environment", {}).get("CUDA_VISIBLE_DEVICES") or not runtime.get("environment", {}).get("SLURM_JOB_ID"):
        raise RuntimeError("Run scheduler or CUDA device identity is absent")
    return seed


def bootstrap_interval(values: dict[tuple[int, int, int], np.ndarray]) -> list[float]:
    """Fixed task/checkpoint strata, canonical episodes resampled as five-pair clusters."""
    expected = {(checkpoint, task, episode) for checkpoint in range(3) for task in range(10) for episode in range(40, 50)}
    if set(values) != expected or any(array.shape != (5,) for array in values.values()):
        raise RuntimeError("Bootstrap inputs do not form 3 x 10 x 10 episode clusters of five pairs")
    draws = int(PROTOCOL["bootstrap_draws"])
    rng = np.random.default_rng(int(PROTOCOL["bootstrap_seed"]))
    means = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        stratum_means = []
        for checkpoint in range(3):
            for task in range(10):
                sampled = rng.integers(40, 50, size=10)
                stratum_means.append(float(np.mean([values[(checkpoint, task, int(episode))] for episode in sampled])))
        means[draw] = float(np.mean(stratum_means))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if len(args.result) != 15 or len(set(args.result)) != 15:
        raise ValueError("Exactly fifteen distinct full shard results are required")
    root = Path(__file__).resolve().parents[1]
    if args.manifest.resolve() != (root / MANIFEST_PATH).resolve() or args.smoke.resolve() != (root / SMOKE_RESULT_PATH).resolve() or args.output.resolve() != (root / SUMMARY_RESULT_PATH).resolve():
        raise RuntimeError("Summary manifest, smoke, or output path is not the frozen namespace")
    if args.specificity_summary.resolve() != (root / "results/local_instruction_specificity_v2_summary.json").resolve():
        raise RuntimeError("Specificity prerequisite path is not frozen")
    import torch

    summary_runtime_versions = {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "torch_build": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "pillow": importlib.metadata.version("Pillow"),
        "mujoco": importlib.metadata.version("mujoco"),
        "robosuite": importlib.metadata.version("robosuite"),
    }
    if summary_runtime_versions != PROTOCOL["runtime_versions"]:
        raise RuntimeError(f"Summary runtime versions differ: {summary_runtime_versions}")
    summary_sources_start = source_hashes(
        root,
        (
            "athena/preflight_counterfactual_target_swap.py",
            "athena/run_counterfactual_target_swap.py",
            "athena/summarize_counterfactual_target_swap.py",
        ),
    )
    run_sources = {key: value for key, value in summary_sources_start.items() if key != "athena/summarize_counterfactual_target_swap.py"}
    specificity = validate_specificity_summary(args.specificity_summary)
    manifest = validate_manifest(args.manifest, root)
    libero_summary_start = validate_libero_runtime()
    manifest_sha = file_sha256(args.manifest)
    specificity_sha = file_sha256(args.specificity_summary)
    provenance_sha = str(manifest["provenance"]["live_sha256"])
    suite = load_suite("libero_object")
    if task_languages(suite) != PROMPTS:
        raise RuntimeError("Live Object task-language order differs")
    canonical_hashes = {
        (task, episode): hash_array(np.asarray(official_init_states(suite, task)[episode]))
        for task in range(10)
        for episode in range(40, 50)
    }
    mapping_lookup = {(int(row["task_index"]), int(row["counterfactual_prompt_id"])): row for row in manifest["mappings"]}
    bodies_by_task = {
        task: set(next(row for key, row in mapping_lookup.items() if key[0] == task)["rewrite_validation"]["object_symbols"].values()) | {RECEPTACLE_SYMBOL}
        for task in range(10)
    }

    smoke = load_json(args.smoke)
    smoke_seed = validate_result_identity(smoke, "strict_smoke", manifest_sha, specificity_sha, provenance_sha, manifest["libero_runtime_end"], run_sources)
    if smoke_seed != 0:
        raise RuntimeError("Strict smoke did not use seed zero")
    evaluation = smoke.get("evaluation", {})
    if (int(evaluation.get("task_start", -1)), int(evaluation.get("task_end", -1)), int(evaluation.get("episode_start", -1)), int(evaluation.get("eps_per_task", -1)), int(evaluation.get("smoke_mapping_count", -1))) != (0, 10, 40, 1, 50):
        raise RuntimeError("Strict smoke coverage differs from the frozen protocol")
    seen_smoke = set()
    original_smoke_physical = {}
    seen_smoke_controls = set()
    for row in smoke.get("smoke_original_inputs", []):
        task = int(row["task_index"])
        episode = int(row["episode"])
        if task not in range(10) or episode != 40 or task in seen_smoke_controls:
            raise RuntimeError("Smoke original-control identity is invalid or duplicate")
        seen_smoke_controls.add(task)
        if int(row["init_state_index"]) != episode or int(row["reset_seed"]) != task * 100 + episode or row["init_state_sha256"] != canonical_hashes[(task, episode)]:
            raise RuntimeError("Smoke original-control canonical state differs")
        expected_original_sha = next(mapping_lookup[(task, distractor)]["original_bddl_sha256"] for distractor in range(10) if (task, distractor) in mapping_lookup)
        if row["original_bddl_sha256"] != expected_original_sha:
            raise RuntimeError("Smoke original-control BDDL identity differs")
        validate_physical(row["physical_input"], bodies_by_task[task])
        original_smoke_physical[task] = row["physical_input"]
    if seen_smoke_controls != set(range(10)):
        raise RuntimeError("Smoke did not cover all ten original-goal physical inputs")
    for row in smoke.get("smoke_mappings", []):
        task = int(row["task_index"])
        distractor = int(row["counterfactual_prompt_id"])
        key = (task, distractor)
        if key not in mapping_lookup or key in seen_smoke or int(row["episode"]) != 40:
            raise RuntimeError("Smoke mapping identity is invalid or duplicate")
        seen_smoke.add(key)
        if row.get("rewritten_bddl_sha256") != mapping_lookup[key]["rewritten_bddl_sha256"] or row.get("physical_matches_original_control") is not True:
            raise RuntimeError("Smoke BDDL or physical-match validation failed")
        physical = row["physical_input"]
        validate_physical(physical, bodies_by_task[task])
        if physical != original_smoke_physical[task]:
            raise RuntimeError("Smoke target swap changed the original physical input")
        chunks = row.get("finite_action_chunks", {})
        if set(chunks) != {str(task), str(distractor)}:
            raise RuntimeError("Smoke prompt-pair coverage is incomplete")
        for encoded_chunk in chunks.values():
            chunk = decode_array(encoded_chunk)
            if chunk.dtype != np.dtype("<f4") or chunk.shape != (8, 7) or not np.isfinite(chunk).all():
                raise RuntimeError("Smoke action chunk is malformed")
    if seen_smoke != set(mapping_lookup):
        raise RuntimeError("Strict smoke did not cover every task/distractor mapping")

    from athena.run_xvla_experiment import build_encoder, build_vocab

    vocab, _ = build_vocab(PROMPTS)
    encode = build_encoder(vocab)
    instruction_ids = {prompt: encode(text) for prompt, text in PROMPTS.items()}
    seen_shards = set()
    seen_controls = set()
    seen_pairs = set()
    cross_checkpoint_starts = {}
    control_success = defaultdict(list)
    matching_success = defaultdict(list)
    original_prompt_success = defaultdict(list)
    differences_by_cluster: dict[tuple[int, int, int], list[tuple[int, float]]] = defaultdict(list)
    result_identities = []
    seen_gpu_job_ids = {str(smoke["runtime"]["environment"]["SLURM_JOB_ID"])}
    expected_runtime_signature = {key: smoke["runtime"][key] for key in ("python", "torch", "torch_build", "cuda", "numpy", "pillow", "mujoco", "robosuite", "gpu", "gpu_capability")}
    for path in args.result:
        result = load_json(path)
        seed = validate_result_identity(result, "full", manifest_sha, specificity_sha, provenance_sha, manifest["libero_runtime_end"], run_sources)
        runtime_signature = {key: result["runtime"][key] for key in expected_runtime_signature}
        if runtime_signature != expected_runtime_signature:
            raise RuntimeError("Full run software or GPU runtime differs from strict smoke")
        slurm_job_id = str(result["runtime"]["environment"]["SLURM_JOB_ID"])
        if slurm_job_id in seen_gpu_job_ids:
            raise RuntimeError("GPU result reused a smoke or full Slurm job identity")
        seen_gpu_job_ids.add(slurm_job_id)
        eval_row = result.get("evaluation", {})
        task_start = int(eval_row.get("task_start", -1))
        task_end = int(eval_row.get("task_end", -1))
        shard = (seed, task_start, task_end)
        expected_result_path = root / f"results/counterfactual_target_swap_v1_s{seed}_t{task_start}_{task_end}.json"
        if path.resolve() != expected_result_path.resolve():
            raise RuntimeError("Full result path is not the frozen seed/shard namespace")
        if [task_start, task_end] not in PROTOCOL["task_shards"] or shard in seen_shards:
            raise RuntimeError("Full result has an unexpected or duplicate task shard")
        seen_shards.add(shard)
        if (int(eval_row.get("episode_start", -1)), int(eval_row.get("eps_per_task", -1)), int(eval_row.get("expected_rollouts", -1)), int(eval_row.get("completed_rollouts", -1)), int(eval_row.get("completed_pairs", -1)), int(eval_row.get("completed_controls", -1))) != (40, 10, 220, 220, 100, 20):
            raise RuntimeError("Full shard trial counts differ from the frozen protocol")
        for row in result.get("original_goal_controls", []):
            task = int(row["task_index"])
            episode = int(row["episode"])
            key = (seed, task, episode)
            if key in seen_controls or not task_start <= task < task_end or episode not in range(40, 50):
                raise RuntimeError("Original-goal control identity is invalid or duplicate")
            seen_controls.add(key)
            if int(row["init_state_index"]) != episode or int(row["reset_seed"]) != task * 100 + episode or row["init_state_sha256"] != canonical_hashes[(task, episode)]:
                raise RuntimeError("Original-goal control canonical state identity differs")
            expected_original_sha = next(mapping_lookup[(task, distractor)]["original_bddl_sha256"] for distractor in range(10) if (task, distractor) in mapping_lookup)
            if row["original_bddl_sha256"] != expected_original_sha:
                raise RuntimeError("Original-goal control BDDL identity differs")
            run = row["rollout"]
            start = run["start_physical"]
            validate_physical(start, bodies_by_task[task])
            cross_key = (task, episode)
            if cross_key in cross_checkpoint_starts and cross_checkpoint_starts[cross_key] != start:
                raise RuntimeError("Canonical physical input differs across checkpoints")
            cross_checkpoint_starts[cross_key] = start
            success = validate_rollout(run, task, instruction_ids[task], start, bodies_by_task[task])
            control_success[seed].append(float(success))
        for row in result.get("paired_counterfactual_trials", []):
            task = int(row["task_index"])
            episode = int(row["episode"])
            distractor = int(row["counterfactual_prompt_id"])
            key = (seed, task, episode, distractor)
            mapping_key = (task, distractor)
            if key in seen_pairs or mapping_key not in mapping_lookup or not task_start <= task < task_end or episode not in range(40, 50):
                raise RuntimeError("Counterfactual pair identity is invalid or duplicate")
            seen_pairs.add(key)
            if int(row["original_prompt_id"]) != task or int(row["init_state_index"]) != episode or int(row["reset_seed"]) != task * 100 + episode or row["init_state_sha256"] != canonical_hashes[(task, episode)]:
                raise RuntimeError("Counterfactual pair canonical identity differs")
            if row["rewritten_bddl_sha256"] != mapping_lookup[mapping_key]["rewritten_bddl_sha256"]:
                raise RuntimeError("Counterfactual pair BDDL identity differs")
            expected_order = ["matching_counterfactual_prompt", "original_prompt"] if (task + distractor + episode + seed) % 2 == 0 else ["original_prompt", "matching_counterfactual_prompt"]
            if row.get("execution_order") != expected_order:
                raise RuntimeError("Counterfactual condition execution order differs")
            conditions = row.get("conditions", {})
            if set(conditions) != {"matching_counterfactual_prompt", "original_prompt"}:
                raise RuntimeError("Counterfactual prompt condition pair is incomplete")
            expected_start = cross_checkpoint_starts[(task, episode)]
            if row.get("physical_input_sha256") != expected_start["combined_sha256"]:
                raise RuntimeError("Counterfactual physical input digest differs from original control")
            match = validate_rollout(conditions["matching_counterfactual_prompt"], distractor, instruction_ids[distractor], expected_start, bodies_by_task[task])
            original = validate_rollout(conditions["original_prompt"], task, instruction_ids[task], expected_start, bodies_by_task[task])
            matching_success[(seed, task)].append(float(match))
            original_prompt_success[(seed, task)].append(float(original))
            differences_by_cluster[(seed, task, episode)].append((distractor, float(match) - float(original)))
        result_identities.append({"path": str(path), "sha256": file_sha256(path), "slurm_job_id": slurm_job_id, "seed": seed, "task_start": task_start, "task_end": task_end})

    expected_shards = {(seed, start, end) for seed in range(3) for start, end in PROTOCOL["task_shards"]}
    expected_controls = {(seed, task, episode) for seed in range(3) for task in range(10) for episode in range(40, 50)}
    expected_pairs = {(seed, task, episode, distractor) for seed in range(3) for (task, distractor) in mapping_lookup for episode in range(40, 50)}
    if seen_shards != expected_shards or seen_controls != expected_controls or seen_pairs != expected_pairs:
        raise RuntimeError("Full results do not cover the frozen shard, control, and pair sets exactly")
    if len(seen_gpu_job_ids) != 16:
        raise RuntimeError("Expected one smoke and fifteen distinct full GPU job identities")

    clustered_differences = {}
    for key, rows in differences_by_cluster.items():
        rows.sort()
        if len(rows) != 5 or len({distractor for distractor, _ in rows}) != 5:
            raise RuntimeError(f"Episode cluster {key} does not contain five distractors")
        clustered_differences[key] = np.asarray([value for _, value in rows], dtype=np.float64)
    interval = bootstrap_interval(clustered_differences)
    checkpoint_rows = {}
    for seed in range(3):
        match_values = np.asarray([value for task in range(10) for value in matching_success[(seed, task)]], dtype=np.float64)
        original_values = np.asarray([value for task in range(10) for value in original_prompt_success[(seed, task)]], dtype=np.float64)
        controls = np.asarray(control_success[seed], dtype=np.float64)
        if match_values.shape != (500,) or original_values.shape != (500,) or controls.shape != (100,):
            raise RuntimeError("Per-checkpoint result counts are incomplete")
        advantage = float(np.mean(match_values - original_values))
        checkpoint_rows[str(seed)] = {
            "matching_counterfactual_success": float(match_values.mean()),
            "original_prompt_on_counterfactual_goal_success": float(original_values.mean()),
            "paired_advantage": advantage,
            "original_goal_control_success": float(controls.mean()),
            "positive_paired_advantage": advantage > 0,
            "original_goal_control_gate": float(controls.mean()) >= FROZEN_GATES["minimum_original_goal_control_success_every_checkpoint"],
        }

    pooled_task_match = {}
    pooled_task_original = {}
    pooled_task_advantage = {}
    for task in range(10):
        match = np.asarray([value for seed in range(3) for value in matching_success[(seed, task)]], dtype=np.float64)
        original = np.asarray([value for seed in range(3) for value in original_prompt_success[(seed, task)]], dtype=np.float64)
        pooled_task_match[str(task)] = float(match.mean())
        pooled_task_original[str(task)] = float(original.mean())
        pooled_task_advantage[str(task)] = float(np.mean(match - original))
    matching_task_macro = float(np.mean(list(pooled_task_match.values())))
    pooled_advantage = float(np.mean(list(pooled_task_advantage.values())))
    positive_tasks = sum(value > 0 for value in pooled_task_advantage.values())
    checks = {
        "matching_counterfactual_task_macro_at_least_0p25": matching_task_macro >= FROZEN_GATES["minimum_matching_counterfactual_task_macro_success"],
        "pooled_paired_advantage_at_least_0p15": pooled_advantage >= FROZEN_GATES["minimum_pooled_paired_advantage"],
        "paired_advantage_positive_every_checkpoint": all(row["positive_paired_advantage"] for row in checkpoint_rows.values()),
        "at_least_eight_positive_pooled_task_advantages": positive_tasks >= FROZEN_GATES["minimum_positive_pooled_task_advantages"],
        "stratified_paired_bootstrap_lower_above_zero": interval[0] > FROZEN_GATES["task_checkpoint_stratified_bootstrap_lower_strictly_above"],
        "original_goal_control_at_least_0p60_every_checkpoint": all(row["original_goal_control_gate"] for row in checkpoint_rows.values()),
    }
    overall_pass = all(checks.values())
    summary_sources_end = source_hashes(root, ("athena/preflight_counterfactual_target_swap.py", "athena/run_counterfactual_target_swap.py", "athena/summarize_counterfactual_target_swap.py"))
    if summary_sources_start != summary_sources_end:
        raise RuntimeError("Repository sources changed while target-swap summary was active")
    libero_summary_end = validate_libero_runtime()
    if libero_summary_start != libero_summary_end:
        raise RuntimeError("LIBERO sources changed while target-swap summary was active")
    output = {
        "schema": SCHEMA_SUMMARY,
        "protocol": PROTOCOL,
        "frozen_gates": FROZEN_GATES,
        "identity_validated": True,
        "diagnostic_complete": True,
        "overall_pass": overall_pass,
        "claim_eligible": overall_pass,
        "gate_checks": checks,
        "prerequisite_specificity": {
            "summary": str(args.specificity_summary),
            "sha256": specificity_sha,
            "identity_validated": specificity["identity_validated"],
            "overall_pass": specificity["overall_pass"],
            "claim_eligible": specificity["claim_eligible"],
        },
        "manifest": {"path": str(args.manifest), "sha256": manifest_sha, "mapping_count": len(manifest["mappings"])},
        "smoke": {"path": str(args.smoke), "sha256": file_sha256(args.smoke), "mapping_count": len(seen_smoke)},
        "full_results": sorted(result_identities, key=lambda row: (row["seed"], row["task_start"])),
        "source_sha256_start": summary_sources_start,
        "source_sha256_end": summary_sources_end,
        "libero_source_start": libero_summary_start,
        "libero_source_end": libero_summary_end,
        "checkpoint_results": checkpoint_rows,
        "pooled": {
            "paired_trials": len(seen_pairs),
            "original_goal_controls": len(seen_controls),
            "matching_counterfactual_task_macro_success": matching_task_macro,
            "paired_advantage": pooled_advantage,
            "positive_task_advantages": positive_tasks,
            "task_matching_success": pooled_task_match,
            "task_original_prompt_on_counterfactual_goal_success": pooled_task_original,
            "task_paired_advantage": pooled_task_advantage,
            "task_checkpoint_stratified_episode_cluster_bootstrap_interval": interval,
            "bootstrap_draws": PROTOCOL["bootstrap_draws"],
        },
        "runtime": summary_runtime_versions,
        "claim_if_passed": "On the prespecified familiar-object LIBERO-Object scenes with synthetic BDDL goal swaps, matching the instruction to the counterfactual goal increases paired closed-loop BDDL completion relative to retaining the original instruction.",
        "claim_boundaries": "The result is restricted to known task vocabulary, familiar Object-suite scenes, co-present distractors, and synthetic goal rewrites. Canonical episodes 40 through 49 were included in the broader prerequisite local-specificity state set, so this is a new closed-loop outcome and intervention on overlapping states, not an independent-state replication. It does not establish paraphrase robustness, unseen-object or broad compositional grounding, cross-suite or physical-robot generalization, or a benefit unique to tensor decomposability.",
        "interval_scope": "The task/checkpoint-stratified interval resamples canonical episodes as clusters of all five paired distractors within every fixed task and checkpoint. It is a finite-protocol stability interval, not a population confidence interval.",
    }
    write_json(args.output, output)
    print("RESULT", json.dumps({"identity_validated": True, "overall_pass": overall_pass, "pooled_paired_advantage": pooled_advantage, "bootstrap_interval": interval}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
