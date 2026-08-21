#!/usr/bin/env python3
"""Strictly validate and score instruction necessity v1."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from athena.instruction_necessity_common import (
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


def expected_instruction_catalog() -> dict[str, dict[str, Any]]:
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
    return {
        "bos_only": {"prompt_id": "bos_only", "prompt_text": "", "instruction_ids": encode("")},
        **{
            str(prompt_id): {
                "prompt_id": prompt_id,
                "prompt_text": text,
                "instruction_ids": encode(text),
            }
            for prompt_id, text in languages.items()
        },
    }


def validate_rollout(rollout: dict[str, Any]) -> bool:
    actions = decode_array(rollout["executed_actions"])
    rewards = decode_array(rollout["rewards"])
    dones = decode_array(rollout["dones"])
    steps = int(rollout["steps"])
    if actions.shape != (steps, 7) or rewards.shape != (steps,) or dones.shape != (steps,):
        raise RuntimeError("Trajectory array shapes differ from the recorded step count")
    if not np.isfinite(actions).all() or not np.isfinite(rewards).all():
        raise RuntimeError("Trajectory contains nonfinite values")
    success = bool(np.any(rewards > 0))
    if bool(rollout["success"]) != success:
        raise RuntimeError("Success does not reproduce from rewards")
    expected_step = int(np.argmax(rewards > 0) + 1) if success else None
    if rollout["success_step"] != expected_step:
        raise RuntimeError("Success step does not reproduce")
    reconstructed = []
    for chunk_row in rollout["chunks"]:
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
    reconstructed_array = np.asarray(reconstructed, dtype=np.float64).reshape(-1, 7)
    if not np.array_equal(reconstructed_array, actions):
        raise RuntimeError("Executed actions do not reproduce from saved chunks")
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
        raise RuntimeError(f"{path} is not a {mode} instruction-necessity result")
    if result.get("protocol") != PROTOCOL or result.get("frozen_gates") != FROZEN_GATES:
        raise RuntimeError(f"{path} protocol or gates differ")
    if result.get("design_provenance") != DESIGN_PROVENANCE:
        raise RuntimeError(f"{path} design provenance differs")
    identity = result["identity"]
    seed = int(identity["checkpoint_seed"])
    if seed not in CHECKPOINTS:
        raise RuntimeError(f"{path} checkpoint seed differs")
    checkpoint = CHECKPOINTS[seed]
    if Path(identity["checkpoint"]).name != Path(checkpoint["path"]).name or identity["checkpoint_sha256"] != checkpoint["sha256"]:
        raise RuntimeError(f"{path} checkpoint identity differs")
    if (
        identity["cache_sha256"] != CACHE["sha256"]
        or str(identity["provenance_job_id"]) != PROVENANCE["job_id"]
        or identity["provenance_result_sha256"] != provenance_sha
    ):
        raise RuntimeError(f"{path} cache or provenance identity differs")
    if identity["manifest_sha256"] != manifest_sha:
        raise RuntimeError(f"{path} manifest identity differs")
    if identity["source_sha256_start"] != current_sources or identity["source_sha256_end"] != current_sources:
        raise RuntimeError(f"{path} source closure differs")
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


def paired_counts(records: list[dict[str, Any]], control: str) -> dict[str, float | int]:
    correct = sum(int(row["correct_prompt"]) for row in records)
    control_success = sum(int(row[control]) for row in records)
    correct_only = sum(int(row["correct_prompt"] and not row[control]) for row in records)
    control_only = sum(int(row[control] and not row["correct_prompt"]) for row in records)
    trials = len(records)
    return {
        "trials": trials,
        "correct_successes": correct,
        "control_successes": control_success,
        "correct_success_rate": correct / trials,
        "control_success_rate": control_success / trials,
        "paired_gap": (correct - control_success) / trials,
        "correct_only": correct_only,
        "control_only": control_only,
        "one_sided_exact_p": one_sided_paired_exact_p(correct_only, control_only),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
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
        "episode_start": 40,
        "eps_per_task": 1,
        "expected_rollouts": 0,
        "completed_rollouts": 0,
        "completed_paired_states": 0,
        "smoke_states": 10,
        "matmul_precision": "highest",
    }:
        raise RuntimeError("Strict smoke evaluation matrix differs")
    if len(smoke["smoke_rows"]) != 10 or smoke["paired_states"]:
        raise RuntimeError("Strict smoke row matrix differs")
    for row in smoke["smoke_rows"]:
        task = int(row["task_index"])
        episode = int(row["episode"])
        mapping = mappings[(task, episode)]
        if episode != 40 or row["physical_input_sha256"] != mapping["settled_physical_input_sha256"]:
            raise RuntimeError("Smoke physical identity differs")
        if row["condition_order"] != condition_order(0, task, episode) or set(row["conditions"]) != set(CONDITIONS):
            raise RuntimeError("Smoke condition order or set differs")
        expected_ids = {
            "correct_prompt": str(task),
            "visible_distractor_prompt": str(mapping["selected_distractor_prompt_id"]),
            "bos_only": "bos_only",
        }
        for condition, value in row["conditions"].items():
            expected = catalog[expected_ids[condition]]
            for key in ("prompt_id", "prompt_text", "instruction_ids"):
                if value[key] != expected[key]:
                    raise RuntimeError("Smoke prompt identity differs")
            chunk = decode_array(value["action_chunk"])
            if chunk.shape != (8, 7) or not np.isfinite(chunk).all():
                raise RuntimeError("Smoke action chunk is malformed")

    expected_files = {
        root / f"results/instruction_necessity_v1_s{seed}_t{start}_{end}.json"
        for seed in range(3)
        for start, end in TASK_SHARDS
    }
    if {path.resolve() for path in args.result} != {path.resolve() for path in expected_files}:
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
            int(evaluation["completed_paired_states"]),
            int(evaluation["smoke_states"]),
            evaluation["matmul_precision"],
        ) != (40, 10, 60, 60, 20, 0, "highest"):
            raise RuntimeError(f"{path} full evaluation matrix differs")
        if result["smoke_rows"] or len(result["paired_states"]) != 20:
            raise RuntimeError(f"{path} full row matrix differs")
        result_identities[str(path)] = {"sha256": file_sha256(path)}
        for row in result["paired_states"]:
            task = int(row["task_index"])
            episode = int(row["episode"])
            key = (seed, task, episode)
            if key in observed_keys or not (bounds[0] <= task < bounds[1]) or episode not in range(40, 50):
                raise RuntimeError("Full row identity is duplicate or outside its shard")
            observed_keys.add(key)
            mapping = mappings[(task, episode)]
            if (
                int(row["init_state_index"]) != episode
                or int(row["reset_seed"]) != task * 100 + episode
                or row["init_state_sha256"] != mapping["init_state_sha256"]
                or row["original_bddl_sha256"] != mapping["original_bddl_sha256"]
                or int(row["selected_distractor_prompt_id"]) != int(mapping["selected_distractor_prompt_id"])
                or row["physical_input_sha256"] != mapping["settled_physical_input_sha256"]
                or row["condition_order"] != condition_order(seed, task, episode)
                or set(row["conditions"]) != set(CONDITIONS)
            ):
                raise RuntimeError("Full row identity differs from the frozen manifest")
            expected_ids = {
                "correct_prompt": str(task),
                "visible_distractor_prompt": str(mapping["selected_distractor_prompt_id"]),
                "bos_only": "bos_only",
            }
            starts = []
            success_row: dict[str, Any] = {"seed": seed, "task": task, "episode": episode}
            for condition, value in row["conditions"].items():
                expected = catalog[expected_ids[condition]]
                for field in ("prompt_id", "prompt_text", "instruction_ids"):
                    if value[field] != expected[field]:
                        raise RuntimeError("Full condition prompt identity differs")
                rollout = value["rollout"]
                if rollout["instruction_ids"] != expected["instruction_ids"] or rollout["prompt_id"] != expected["prompt_id"]:
                    raise RuntimeError("Rollout prompt identity differs")
                starts.append(rollout["start_physical"])
                success_row[condition] = validate_rollout(rollout)
            if not all(start == starts[0] for start in starts[1:]):
                raise RuntimeError("Paired conditions do not share an identical physical input")
            records.append(success_row)
    expected_keys = {
        (seed, task, episode)
        for seed in range(3)
        for task in range(10)
        for episode in range(40, 50)
    }
    if observed_keys != expected_keys or len(records) != 300:
        raise RuntimeError("The complete seed-task-episode matrix is absent")

    pooled = {
        control: paired_counts(records, control)
        for control in ("visible_distractor_prompt", "bos_only")
    }
    by_checkpoint = {
        str(seed): {
            control: paired_counts([row for row in records if row["seed"] == seed], control)
            for control in ("visible_distractor_prompt", "bos_only")
        }
        for seed in range(3)
    }
    by_task = {
        str(task): {
            control: paired_counts([row for row in records if row["task"] == task], control)
            for control in ("visible_distractor_prompt", "bos_only")
        }
        for task in range(10)
    }
    correct_pooled = sum(int(row["correct_prompt"]) for row in records) / len(records)
    correct_by_checkpoint = {
        str(seed): sum(int(row["correct_prompt"]) for row in records if row["seed"] == seed) / 100
        for seed in range(3)
    }
    task_consistency = {}
    for control in ("visible_distractor_prompt", "bos_only"):
        gaps = [by_task[str(task)][control]["paired_gap"] for task in range(10)]
        task_consistency[control] = {
            "strictly_positive_tasks": sum(float(gap) > 0 for gap in gaps),
            "negative_tasks": sum(float(gap) < 0 for gap in gaps),
            "gaps_by_task": {str(task): gaps[task] for task in range(10)},
        }

    gates = {
        "correct_success_pooled_at_least_0p70": correct_pooled >= 0.70,
        "correct_success_every_checkpoint_at_least_0p60": all(
            value >= 0.60 for value in correct_by_checkpoint.values()
        ),
        "correct_minus_visible_distractor_at_least_0p20": pooled["visible_distractor_prompt"]["paired_gap"] >= 0.20,
        "correct_minus_bos_only_at_least_0p20": pooled["bos_only"]["paired_gap"] >= 0.20,
        "both_margins_positive_every_checkpoint": all(
            by_checkpoint[str(seed)][control]["paired_gap"] > 0
            for seed in range(3)
            for control in ("visible_distractor_prompt", "bos_only")
        ),
        "both_one_sided_paired_exact_p_at_most_0p01": all(
            pooled[control]["one_sided_exact_p"] <= 0.01
            for control in ("visible_distractor_prompt", "bos_only")
        ),
        "at_least_eight_strictly_positive_tasks_per_control": all(
            task_consistency[control]["strictly_positive_tasks"] >= 8
            for control in ("visible_distractor_prompt", "bos_only")
        ),
        "no_negative_task_gap_for_either_control": all(
            task_consistency[control]["negative_tasks"] == 0
            for control in ("visible_distractor_prompt", "bos_only")
        ),
    }
    overall_pass = all(gates.values())
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
        "failed_local_summary_sha256": manifest["failed_local_summary_sha256"],
        "smoke": str(args.smoke),
        "smoke_sha256": file_sha256(args.smoke),
        "input_results": result_identities,
        "summary_source_sha256": file_sha256(Path(__file__).resolve()),
        "correct_success_pooled": correct_pooled,
        "correct_success_by_checkpoint": correct_by_checkpoint,
        "pooled": pooled,
        "by_checkpoint": by_checkpoint,
        "by_task": by_task,
        "task_consistency": task_consistency,
        "gates": gates,
        "claim_if_passed": (
            "Across three fixed chi-ViT checkpoints and 300 paired canonical "
            "LIBERO-Object states, correct familiar instructions improve original-goal "
            "closed-loop success by at least 20 percentage points relative to both one "
            "outcome-independent visible-distractor instruction and BOS-only input."
        ),
        "claim_boundaries": (
            "This establishes instruction necessity only for familiar Object-suite prompts, "
            "objects, scenes, tasks, and checkpoints. It does not establish paraphrase "
            "robustness, unseen-object or compositional grounding, broad generalization, "
            "physical transfer, or a benefit unique to tensor decomposability. Episodes 40 "
            "through 49 overlap the failed local endpoint's states, while the 280-step "
            "original-goal success outcome is new. This endpoint was frozen after the local "
            "displacement-rank endpoint became impossible and is not a rescue reanalysis."
        ),
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
                "correct_success_pooled": correct_pooled,
                "visible_distractor_gap": pooled["visible_distractor_prompt"]["paired_gap"],
                "bos_only_gap": pooled["bos_only"]["paired_gap"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
