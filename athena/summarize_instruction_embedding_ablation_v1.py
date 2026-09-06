#!/usr/bin/env python3
"""Strictly validate and score instruction-embedding ablation v1."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from athena.instruction_embedding_ablation_v1_common import (
    CACHE,
    CHECKPOINTS,
    CONDITIONS,
    DESIGN_PROVENANCE,
    FROZEN_GATES,
    MANIFEST_PATH,
    PROMPTS,
    PROTOCOL,
    PROVENANCE,
    SCHEMA_RUN,
    SCHEMA_SUMMARY,
    SMOKE_RESULT_PATH,
    SUMMARY_RESULT_PATH,
    TASK_SHARDS,
    condition_order,
    decode_array,
    file_sha256,
    hash_array,
    one_sided_paired_exact_p,
    source_hashes,
    validate_manifest,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def expected_instruction_catalog() -> dict[int, dict[str, Any]]:
    from athena.run_xvla_experiment import (
        build_encoder,
        build_vocab,
        load_suite,
        task_languages,
    )

    suite = load_suite("libero_object")
    languages = task_languages(suite)
    if languages != PROMPTS:
        raise RuntimeError("Live Object prompt catalog differs")
    vocab, _ = build_vocab(languages)
    encode = build_encoder(vocab)
    result = {
        task: {
            "prompt_id": task,
            "prompt_text": text,
            "instruction_ids": encode(text),
        }
        for task, text in languages.items()
    }
    if any(
        len(row["instruction_ids"]) != PROTOCOL["instruction_token_count"]
        for row in result.values()
    ):
        raise RuntimeError("Expected prompt catalog does not encode to 32 positions")
    return result


def validate_rollout(rollout: dict[str, Any]) -> bool:
    actions = decode_array(rollout["executed_actions"])
    rewards = decode_array(rollout["rewards"])
    dones = decode_array(rollout["dones"])
    steps = int(rollout["steps"])
    if actions.shape != (steps, 7) or rewards.shape != (steps,) or dones.shape != (steps,):
        raise RuntimeError("Trajectory array shapes differ from the recorded step count")
    if not np.isfinite(actions).all() or not np.isfinite(rewards).all():
        raise RuntimeError("Trajectory contains nonfinite values")
    if steps < 1 or steps > PROTOCOL["max_steps"]:
        raise RuntimeError("Trajectory step count is outside the frozen range")
    success = bool(np.any(rewards > 0))
    if bool(rollout["success"]) != success:
        raise RuntimeError("Success does not reproduce from rewards")
    expected_step = int(np.argmax(rewards > 0) + 1) if success else None
    if rollout["success_step"] != expected_step:
        raise RuntimeError("Success step does not reproduce")
    reconstructed: list[np.ndarray] = []
    expected_start_step = 0
    for chunk_row in rollout["chunks"]:
        if int(chunk_row["start_step"]) != expected_start_step:
            raise RuntimeError("Chunk start steps are not contiguous")
        chunk = decode_array(chunk_row["prediction"])
        if chunk.shape != (8, 7) or not np.isfinite(chunk).all():
            raise RuntimeError("A predicted action chunk is malformed")
        executed_rows = int(chunk_row["executed_rows"])
        if executed_rows < 1 or executed_rows > 8:
            raise RuntimeError("Executed chunk-row count is invalid")
        for action in chunk[:executed_rows]:
            row = np.asarray(action, dtype=np.float64).copy()
            row[-1] = 1.0 if row[-1] > 0 else -1.0
            reconstructed.append(row)
        expected_start_step += executed_rows
    reconstructed_array = np.asarray(reconstructed, dtype=np.float64).reshape(-1, 7)
    if expected_start_step != steps or not np.array_equal(reconstructed_array, actions):
        raise RuntimeError("Executed actions do not reproduce exactly from saved chunks")
    return success


def validate_identity(
    result: dict[str, Any],
    path: Path,
    mode: str,
    manifest_sha: str,
    provenance_sha: str,
    current_sources: dict[str, str],
) -> int:
    if result.get("schema") != SCHEMA_RUN or result.get("mode") != mode:
        raise RuntimeError(f"{path} is not a {mode} lexical-ablation result")
    if result.get("protocol") != PROTOCOL or result.get("frozen_gates") != FROZEN_GATES:
        raise RuntimeError(f"{path} protocol or gates differ")
    if result.get("design_provenance") != DESIGN_PROVENANCE:
        raise RuntimeError(f"{path} design provenance differs")
    identity = result["identity"]
    seed = int(identity["checkpoint_seed"])
    if seed not in CHECKPOINTS:
        raise RuntimeError(f"{path} checkpoint seed differs")
    checkpoint = CHECKPOINTS[seed]
    if (
        Path(identity["checkpoint"]).name != Path(checkpoint["path"]).name
        or identity["checkpoint_sha256"] != checkpoint["sha256"]
    ):
        raise RuntimeError(f"{path} checkpoint identity differs")
    if (
        identity["cache_sha256"] != CACHE["sha256"]
        or str(identity["provenance_job_id"]) != PROVENANCE["job_id"]
        or identity["provenance_result_sha256"] != provenance_sha
        or identity["manifest_sha256"] != manifest_sha
    ):
        raise RuntimeError(f"{path} cache, provenance, or manifest identity differs")
    if (
        identity["source_sha256_start"] != current_sources
        or identity["source_sha256_end"] != current_sources
    ):
        raise RuntimeError(f"{path} source closure differs")
    if identity["model_state_sha256_start"] != identity["model_state_sha256_end"]:
        raise RuntimeError(f"{path} model state changed")
    if (
        identity["protected_component_sha256_start"]
        != identity["protected_component_sha256_end"]
    ):
        raise RuntimeError(f"{path} a protected model component changed")
    expected_protected = {
        "learned_common_bos",
        "token_embedding_weights",
        "action_queries",
        "joint_position_embeddings",
        "vision_position_embeddings",
        "state_projection_weight",
        "state_projection_bias",
    }
    if set(identity["protected_component_sha256_start"]) != expected_protected:
        raise RuntimeError(f"{path} protected-component hash set differs")
    runtime = result["runtime"]
    for key, expected in PROTOCOL["runtime_versions"].items():
        if runtime.get(key) != expected:
            raise RuntimeError(f"{path} runtime {key} differs")
    if (
        runtime.get("gpu") != PROTOCOL["gpu_name"]
        or runtime.get("gpu_capability") != PROTOCOL["gpu_compute_capability"]
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("cudnn_benchmark") is not False
        or runtime.get("libero_source_start") != runtime.get("libero_source_end")
    ):
        raise RuntimeError(f"{path} numerical runtime differs")
    return seed


def validate_input_contract(
    value: dict[str, Any], mapping: dict[str, Any], instruction_ids: list[int]
) -> None:
    expected = {
        "physical_input_sha256": mapping["settled_physical_input_sha256"],
        "model_image_sha256": mapping["settled_model_image_sha256"],
        "model_robot_state_sha256": mapping["settled_model_state_sha256"],
        "instruction_ids_sha256": hash_array(
            np.asarray([instruction_ids], dtype=np.int64)
        ),
        "instruction_id_shape": [1, PROTOCOL["instruction_token_count"]],
        "sequence_length_preserved": True,
        "vision_path_preserved": True,
        "state_path_preserved": True,
        "embodiment_path_preserved": True,
        "common_learned_bos_preserved": True,
        "positions_preserved": True,
        "action_queries_preserved": True,
    }
    if value != expected:
        raise RuntimeError("Condition input contract differs from the frozen manifest")


def validate_ablation_audit(
    audit: dict[str, Any],
    condition: str,
    instruction_ids: list[int],
    expected_calls: int,
) -> None:
    if (
        audit.get("protected_component_sha256_before")
        != audit.get("protected_component_sha256_after")
    ):
        raise RuntimeError("A protected component changed within a condition")
    if condition == "full_lexical_embeddings":
        if audit != {
            "enabled": False,
            "mechanism": "deployed_model_unmodified",
            "hook_calls": 0,
            "preexisting_hook_count": 0,
            "post_context_hook_count": 0,
            "all_returned_outputs_exactly_zero": None,
            "calls": [],
            "protected_component_sha256_before": audit[
                "protected_component_sha256_before"
            ],
            "protected_component_sha256_after": audit[
                "protected_component_sha256_after"
            ],
        }:
            raise RuntimeError("Full-path audit differs from an unmodified model call")
        return
    if (
        audit.get("enabled") is not True
        or audit.get("mechanism")
        != "scoped_post_lookup_forward_hook_returning_torch_zeros_like"
        or int(audit.get("hook_calls", -1)) != expected_calls
        or int(audit.get("instruction_token_positions_zeroed_per_call", -1)) != 32
        or int(audit.get("preexisting_hook_count", -1)) != 0
        or int(audit.get("post_context_hook_count", -1)) != 0
        or audit.get("all_returned_outputs_exactly_zero") is not True
        or len(audit.get("calls", [])) != expected_calls
    ):
        raise RuntimeError("Ablated-path hook audit differs")
    expected_ids_sha = hash_array(np.asarray([instruction_ids], dtype=np.int64))
    for index, call in enumerate(audit["calls"]):
        shape = call.get("pre_ablation_output_shape")
        if (
            int(call.get("call_index", -1)) != index
            or call.get("input_ids_shape") != [1, 32]
            or call.get("input_ids_sha256") != expected_ids_sha
            or not isinstance(shape, list)
            or len(shape) != 3
            or shape[0:2] != [1, 32]
            or shape[2] <= 0
            or call.get("pre_ablation_output_dtype") != "torch.float32"
            or not np.isfinite(float(call.get("pre_ablation_output_max_abs", np.nan)))
            or float(call.get("pre_ablation_output_max_abs", 0.0)) <= 0.0
            or float(call.get("returned_output_max_abs", np.nan)) != 0.0
            or int(call.get("returned_nonzero_elements", -1)) != 0
            or int(call.get("zeroed_token_positions", -1)) != 32
        ):
            raise RuntimeError("A hook call record is malformed")
        expected_zero_sha = hash_array(np.zeros(shape, dtype=np.float32))
        if call.get("returned_output_sha256") != expected_zero_sha:
            raise RuntimeError("A hook return does not hash to an exact all-zero tensor")


def paired_counts(records: list[dict[str, Any]]) -> dict[str, float | int | str]:
    full = sum(int(row["full_lexical_embeddings"]) for row in records)
    ablated = sum(int(row["zeroed_lexical_embeddings"]) for row in records)
    full_only = sum(
        int(row["full_lexical_embeddings"] and not row["zeroed_lexical_embeddings"])
        for row in records
    )
    ablated_only = sum(
        int(row["zeroed_lexical_embeddings"] and not row["full_lexical_embeddings"])
        for row in records
    )
    trials = len(records)
    return {
        "paired_checkpoint_state_pairs": trials,
        "full_lexical_successes": full,
        "zeroed_lexical_successes": ablated,
        "full_lexical_success_rate": full / trials,
        "zeroed_lexical_success_rate": ablated / trials,
        "paired_success_gap": (full - ablated) / trials,
        "full_only": full_only,
        "zeroed_only": ablated_only,
        "pooled_pair_mcnemar_one_sided_exact_p": one_sided_paired_exact_p(
            full_only, ablated_only
        ),
        "mcnemar_scope": (
            f"exact over {trials} checkpoint-state pairs; not cluster-robust inference"
        ),
    }


def task_stratified_state_cluster_bootstrap(
    records: list[dict[str, Any]], draws: int, seed: int
) -> list[float]:
    by_state: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_state[(int(row["task"]), int(row["episode"]))].append(row)
    expected = {(task, episode) for task in range(10) for episode in range(30, 40)}
    if set(by_state) != expected or any(len(rows) != 3 for rows in by_state.values()):
        raise RuntimeError("Bootstrap requires 100 states by three checkpoints")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        total = 0.0
        count = 0
        for task in range(10):
            sampled_episodes = rng.integers(30, 40, size=10)
            for episode in sampled_episodes:
                for row in by_state[(task, int(episode))]:
                    total += float(row["full_lexical_embeddings"]) - float(
                        row["zeroed_lexical_embeddings"]
                    )
                    count += 1
        if count != 300:
            raise RuntimeError("Bootstrap draw did not retain 300 checkpoint outcomes")
        means[draw] = total / count
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    expected_executable = Path(PROTOCOL["summary_python_executable"]).resolve()
    declared_executable = os.environ.get("XVLA_SUMMARY_PYTHON_EXECUTABLE")
    if declared_executable != PROTOCOL["summary_python_executable"]:
        raise RuntimeError("Summary wrapper did not declare the frozen Python executable")
    if Path(sys.executable).resolve() != expected_executable:
        raise RuntimeError(
            f"Summary Python executable differs: {Path(sys.executable).resolve()}"
        )
    if platform.python_version() != PROTOCOL["runtime_versions"]["python"]:
        raise RuntimeError("Summary Python version differs")
    if (
        np.__version__ != PROTOCOL["runtime_versions"]["numpy"]
        or importlib.metadata.version("numpy")
        != PROTOCOL["runtime_versions"]["numpy"]
    ):
        raise RuntimeError("Summary NumPy version differs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != PROTOCOL["summary_cuda_visible_devices"]:
        raise RuntimeError("CPU summary must have an empty CUDA_VISIBLE_DEVICES value")
    if args.output.resolve() != (root / SUMMARY_RESULT_PATH).resolve():
        raise RuntimeError("Summary output path differs")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.smoke.resolve() != (root / SMOKE_RESULT_PATH).resolve():
        raise RuntimeError("Smoke path differs")
    if args.manifest.resolve() != (root / MANIFEST_PATH).resolve():
        raise RuntimeError("Manifest path differs")
    if len(args.result) != 15 or len(set(args.result)) != 15:
        raise ValueError("Exactly fifteen distinct full-result files are required")

    current_sources = source_hashes(root)
    manifest = validate_manifest(args.manifest, root)
    manifest_sha = file_sha256(args.manifest)
    mappings = {
        (int(row["task_index"]), int(row["episode"])): row
        for row in manifest["mappings"]
    }
    catalog = expected_instruction_catalog()

    smoke = load_json(args.smoke)
    smoke_seed = validate_identity(
        smoke,
        args.smoke,
        "strict_smoke",
        manifest_sha,
        manifest["provenance"]["live_sha256"],
        current_sources,
    )
    if smoke_seed != 0 or smoke["evaluation"] != {
        "task_start": 0,
        "task_end": 10,
        "episode_start": 30,
        "eps_per_task": 1,
        "expected_rollouts": 0,
        "completed_rollouts": 0,
        "completed_checkpoint_state_pairs": 0,
        "smoke_states": 10,
        "matmul_precision": "highest",
    }:
        raise RuntimeError("Strict smoke evaluation matrix differs")
    if len(smoke["smoke_rows"]) != 10 or smoke["checkpoint_state_pairs"]:
        raise RuntimeError("Strict smoke row matrix differs")
    for row in smoke["smoke_rows"]:
        task = int(row["task_index"])
        episode = int(row["episode"])
        mapping = mappings[(task, episode)]
        expected_prompt = catalog[task]
        if (
            episode != 30
            or row["physical_input_sha256"] != mapping["settled_physical_input_sha256"]
            or row["condition_order"] != condition_order(0, task, episode)
            or set(row["conditions"]) != set(CONDITIONS)
        ):
            raise RuntimeError("Smoke state identity, order, or condition set differs")
        contracts = []
        for condition, value in row["conditions"].items():
            for key in ("prompt_id", "prompt_text", "instruction_ids"):
                if value[key] != expected_prompt[key]:
                    raise RuntimeError("Smoke prompt identity differs")
            validate_input_contract(
                value["input_contract"], mapping, expected_prompt["instruction_ids"]
            )
            contracts.append(value["input_contract"])
            validate_ablation_audit(
                value["ablation_audit"],
                condition,
                expected_prompt["instruction_ids"],
                int(condition == "zeroed_lexical_embeddings"),
            )
            chunk = decode_array(value["action_chunk"])
            if chunk.shape != (8, 7) or not np.isfinite(chunk).all():
                raise RuntimeError("Smoke action chunk is malformed")
        if contracts[0] != contracts[1]:
            raise RuntimeError("Smoke conditions do not preserve an identical input contract")

    expected_files = {
        root
        / f"results/instruction_embedding_ablation_v1_s{seed}_t{start}_{end}.json"
        for seed in range(3)
        for start, end in TASK_SHARDS
    }
    if {path.resolve() for path in args.result} != {
        path.resolve() for path in expected_files
    }:
        raise RuntimeError("Full-result path matrix differs")

    records: list[dict[str, Any]] = []
    result_identities: dict[str, dict[str, str]] = {}
    observed_keys: set[tuple[int, int, int]] = set()
    for path in args.result:
        result = load_json(path)
        seed = validate_identity(
            result,
            path,
            "full",
            manifest_sha,
            manifest["provenance"]["live_sha256"],
            current_sources,
        )
        evaluation = result["evaluation"]
        bounds = (int(evaluation["task_start"]), int(evaluation["task_end"]))
        if bounds not in TASK_SHARDS or (
            int(evaluation["episode_start"]),
            int(evaluation["eps_per_task"]),
            int(evaluation["expected_rollouts"]),
            int(evaluation["completed_rollouts"]),
            int(evaluation["completed_checkpoint_state_pairs"]),
            int(evaluation["smoke_states"]),
            evaluation["matmul_precision"],
        ) != (30, 10, 40, 40, 20, 0, "highest"):
            raise RuntimeError(f"{path} full evaluation matrix differs")
        if result["smoke_rows"] or len(result["checkpoint_state_pairs"]) != 20:
            raise RuntimeError(f"{path} full row matrix differs")
        result_identities[str(path)] = {"sha256": file_sha256(path)}
        for row in result["checkpoint_state_pairs"]:
            task = int(row["task_index"])
            episode = int(row["episode"])
            key = (seed, task, episode)
            if (
                key in observed_keys
                or not (bounds[0] <= task < bounds[1])
                or episode not in range(30, 40)
            ):
                raise RuntimeError("Full row identity is duplicate or outside its shard")
            observed_keys.add(key)
            mapping = mappings[(task, episode)]
            expected_prompt = catalog[task]
            if (
                int(row["init_state_index"]) != episode
                or int(row["reset_seed"]) != task * 100 + episode
                or row["init_state_sha256"] != mapping["init_state_sha256"]
                or row["original_bddl_sha256"] != mapping["original_bddl_sha256"]
                or row["physical_input_sha256"]
                != mapping["settled_physical_input_sha256"]
                or row["condition_order"] != condition_order(seed, task, episode)
                or set(row["conditions"]) != set(CONDITIONS)
            ):
                raise RuntimeError("Full row identity differs from the frozen manifest")
            starts: list[dict[str, Any]] = []
            contracts = []
            success_row: dict[str, Any] = {
                "seed": seed,
                "task": task,
                "episode": episode,
            }
            for condition, value in row["conditions"].items():
                for field in ("prompt_id", "prompt_text", "instruction_ids"):
                    if value[field] != expected_prompt[field]:
                        raise RuntimeError("Full condition prompt identity differs")
                validate_input_contract(
                    value["input_contract"], mapping, expected_prompt["instruction_ids"]
                )
                contracts.append(value["input_contract"])
                rollout = value["rollout"]
                if (
                    rollout["instruction_ids"] != expected_prompt["instruction_ids"]
                    or rollout["prompt_id"] != expected_prompt["prompt_id"]
                ):
                    raise RuntimeError("Rollout prompt identity differs")
                expected_calls = (
                    len(rollout["chunks"])
                    if condition == "zeroed_lexical_embeddings"
                    else 0
                )
                validate_ablation_audit(
                    value["ablation_audit"],
                    condition,
                    expected_prompt["instruction_ids"],
                    expected_calls,
                )
                starts.append(rollout["start_physical"])
                success_row[condition] = validate_rollout(rollout)
            if starts[0] != starts[1] or contracts[0] != contracts[1]:
                raise RuntimeError("Paired conditions do not share an identical input")
            records.append(success_row)

    expected_keys = {
        (seed, task, episode)
        for seed in range(3)
        for task in range(10)
        for episode in range(30, 40)
    }
    if observed_keys != expected_keys or len(records) != 300:
        raise RuntimeError("The complete seed-task-episode matrix is absent")

    pooled = paired_counts(records)
    by_checkpoint = {
        str(seed): paired_counts([row for row in records if row["seed"] == seed])
        for seed in range(3)
    }
    by_task = {
        str(task): paired_counts([row for row in records if row["task"] == task])
        for task in range(10)
    }
    full_pooled = float(pooled["full_lexical_success_rate"])
    full_by_checkpoint = {
        str(seed): float(by_checkpoint[str(seed)]["full_lexical_success_rate"])
        for seed in range(3)
    }
    task_gaps = {
        str(task): float(by_task[str(task)]["paired_success_gap"])
        for task in range(10)
    }
    task_consistency = {
        "strictly_positive_tasks": sum(gap > 0 for gap in task_gaps.values()),
        "negative_tasks": sum(gap < 0 for gap in task_gaps.values()),
        "gaps_by_task": task_gaps,
    }
    bootstrap = task_stratified_state_cluster_bootstrap(
        records,
        int(FROZEN_GATES["task_stratified_state_cluster_bootstrap_draws"]),
        int(FROZEN_GATES["task_stratified_state_cluster_bootstrap_seed"]),
    )
    gates = {
        "full_lexical_success_pooled_at_least_0p70": full_pooled >= 0.70,
        "full_lexical_success_every_checkpoint_at_least_0p60": all(
            value >= 0.60 for value in full_by_checkpoint.values()
        ),
        "pooled_success_gap_at_least_0p20": float(pooled["paired_success_gap"])
        >= 0.20,
        "success_gap_positive_every_checkpoint": all(
            float(by_checkpoint[str(seed)]["paired_success_gap"]) > 0
            for seed in range(3)
        ),
        "one_sided_paired_exact_p_at_most_0p01": float(
            pooled["pooled_pair_mcnemar_one_sided_exact_p"]
        )
        <= 0.01,
        "at_least_eight_strictly_positive_tasks": task_consistency[
            "strictly_positive_tasks"
        ]
        >= 8,
        "no_negative_task_gap": task_consistency["negative_tasks"] == 0,
        "task_stratified_state_cluster_bootstrap_lower_above_zero": bootstrap[0]
        > float(
            FROZEN_GATES[
                "task_stratified_state_cluster_bootstrap_lower_strictly_above"
            ]
        ),
    }
    overall_pass = all(gates.values())
    summary_runtime = {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "declared_python_executable": declared_executable,
        "numpy": np.__version__,
        "numpy_distribution": importlib.metadata.version("numpy"),
        "cpu_count_visible": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
    }
    output = {
        "schema": SCHEMA_SUMMARY,
        "protocol": PROTOCOL,
        "frozen_gates": FROZEN_GATES,
        "design_provenance": DESIGN_PROVENANCE,
        "identity_validated": True,
        "overall_pass": overall_pass,
        "claim_eligible": overall_pass,
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "smoke": str(args.smoke),
        "smoke_sha256": file_sha256(args.smoke),
        "input_results": result_identities,
        "summary_source_sha256": file_sha256(Path(__file__).resolve()),
        "summary_runtime": summary_runtime,
        "full_lexical_success_pooled": full_pooled,
        "full_lexical_success_by_checkpoint": full_by_checkpoint,
        "pooled": pooled,
        "by_checkpoint": by_checkpoint,
        "by_task": by_task,
        "task_consistency": task_consistency,
        "task_stratified_state_cluster_bootstrap_interval": bootstrap,
        "bootstrap_scope": (
            "Each draw resamples ten episode states within every task and retains "
            "all three checkpoint outcomes for each sampled state."
        ),
        "gates": gates,
        "claim_if_passed": (
            "Across three fixed chi-ViT policies and 300 paired checkpoint-state "
            "evaluations on 100 fresh canonical LIBERO-Object states, zeroing all "
            "32 learned instruction-token embedding outputs after lookup reduces "
            "closed-loop success by at least 20 percentage points."
        ),
        "claim_boundaries": (
            "This establishes necessity of the learned lexical-embedding path only "
            "for familiar LIBERO-Object tasks, prompts, objects, scenes, and the "
            "three fixed policies. It does not establish semantic grounding, "
            "paraphrase robustness, unseen-object or compositional generalization, "
            "cross-suite or physical transfer, or a benefit unique to tensor "
            "decomposability. The exact paired test is not cluster-robust inference."
        ),
        "elapsed_s": time.perf_counter() - started,
    }
    for path, identity in result_identities.items():
        if file_sha256(Path(path)) != identity["sha256"]:
            raise RuntimeError("A full result changed before summary serialization")
    if source_hashes(root) != current_sources:
        raise RuntimeError("Sources changed during summary")
    write_json(args.output, output)
    print(
        "RESULT",
        json.dumps(
            {
                "overall_pass": overall_pass,
                "full_lexical_success": full_pooled,
                "zeroed_lexical_success": pooled["zeroed_lexical_success_rate"],
                "paired_success_gap": pooled["paired_success_gap"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
