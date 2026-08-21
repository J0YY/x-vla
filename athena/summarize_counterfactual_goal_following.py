#!/usr/bin/env python3
"""Strictly validate and score counterfactual goal following v1."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from athena.counterfactual_goal_following_common import (
    CACHE,
    CHECKPOINTS,
    CONDITIONS,
    DESIGN_PROVENANCE,
    FROZEN_GATES,
    INSTRUCTION_SUMMARY_PATH,
    MANIFEST_PATH,
    PROMPTS,
    PROTOCOL,
    PROVENANCE,
    RECEPTACLE_SYMBOL,
    SCHEMA_RUN,
    SCHEMA_SUMMARY,
    SMOKE_RESULT_PATH,
    SUMMARY_RESULT_PATH,
    TASK_SHARDS,
    condition_order,
    decode_array,
    file_sha256,
    hash_array,
    one_sided_exact_mcnemar_p,
    one_sided_exact_signflip_p,
    source_hashes,
    validate_instruction_summary,
    validate_libero_runtime,
    validate_manifest,
    write_json,
)
from athena.summarize_counterfactual_target_swap import validate_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--instruction-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def instruction_catalog() -> dict[int, list[int]]:
    from athena.run_xvla_experiment import build_encoder, build_vocab, load_suite, task_languages

    languages = task_languages(load_suite("libero_object"))
    if languages != PROMPTS:
        raise RuntimeError("Live Object prompt catalog differs")
    vocab, _ = build_vocab(languages)
    encode = build_encoder(vocab)
    return {prompt_id: encode(text) for prompt_id, text in languages.items()}


def validate_result_identity(
    result: dict[str, Any],
    path: Path,
    mode: str,
    manifest_sha: str,
    instruction_sha: str,
    provenance_sha: str,
    sources: dict[str, str],
) -> int:
    if result.get("schema") != SCHEMA_RUN or result.get("mode") != mode:
        raise RuntimeError(f"{path} is not a {mode} goal-following result")
    if result.get("protocol") != PROTOCOL or result.get("frozen_gates") != FROZEN_GATES:
        raise RuntimeError(f"{path} protocol or gates differ")
    if result.get("design_provenance") != DESIGN_PROVENANCE:
        raise RuntimeError(f"{path} design provenance differs")
    identity = result.get("identity", {})
    seed = int(identity.get("checkpoint_seed", -1))
    if seed not in CHECKPOINTS:
        raise RuntimeError(f"{path} checkpoint seed differs")
    checkpoint = CHECKPOINTS[seed]
    if (
        Path(str(identity.get("checkpoint", ""))).name != Path(checkpoint["path"]).name
        or identity.get("checkpoint_sha256") != checkpoint["sha256"]
        or identity.get("cache_sha256") != CACHE["sha256"]
        or str(identity.get("provenance_job_id")) != PROVENANCE["job_id"]
        or identity.get("provenance_result_sha256") != provenance_sha
        or identity.get("instruction_summary_sha256") != instruction_sha
        or identity.get("instruction_summary_gate_validated") is not True
        or identity.get("manifest_sha256") != manifest_sha
        or identity.get("source_sha256_start") != sources
        or identity.get("source_sha256_end") != sources
    ):
        raise RuntimeError(f"{path} artifact or source identity differs")
    runtime = result.get("runtime", {})
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


def paired_metric(records: list[dict[str, Any]], matching: str, mismatching: str) -> dict[str, Any]:
    match = sum(int(row[matching]) for row in records)
    mismatch = sum(int(row[mismatching]) for row in records)
    match_only = sum(int(row[matching] and not row[mismatching]) for row in records)
    mismatch_only = sum(int(row[mismatching] and not row[matching]) for row in records)
    trials = len(records)
    return {
        "paired_four_arm_units": trials,
        "matching_successes": match,
        "mismatching_successes": mismatch,
        "matching_success_rate": match / trials,
        "mismatching_success_rate": mismatch / trials,
        "paired_gap": (match - mismatch) / trials,
        "matching_only": match_only,
        "mismatching_only": mismatch_only,
        "one_sided_exact_mcnemar_p": one_sided_exact_mcnemar_p(
            match_only, mismatch_only
        ),
        "mcnemar_scope": (
            f"exact over {trials} paired four-arm units; not cluster-robust inference; "
            "task-level sign-flip sensitivity is reported separately"
        ),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve() != (root / SUMMARY_RESULT_PATH).resolve() or args.output.exists():
        raise RuntimeError("Summary output path differs or already exists")
    if args.smoke.resolve() != (root / SMOKE_RESULT_PATH).resolve():
        raise RuntimeError("Smoke path differs")
    if args.manifest.resolve() != (root / MANIFEST_PATH).resolve():
        raise RuntimeError("Manifest path differs")
    if args.instruction_summary.resolve() != (root / INSTRUCTION_SUMMARY_PATH).resolve():
        raise RuntimeError("Instruction summary path differs")
    if len(args.result) != 15 or len(set(args.result)) != 15:
        raise RuntimeError("Exactly fifteen full results are required")
    sources_start = source_hashes(root)
    libero_start = validate_libero_runtime()
    instruction = validate_instruction_summary(args.instruction_summary, root)
    instruction_sha = file_sha256(args.instruction_summary)
    manifest = validate_manifest(args.manifest, root)
    manifest_sha = file_sha256(args.manifest)
    provenance_sha = manifest["provenance"]["live_sha256"]
    mappings = {
        (int(row["task_index"]), int(row["counterfactual_prompt_id"])): row
        for row in manifest["mappings"]
    }
    encoded = instruction_catalog()
    bodies_by_task = {
        task: {
            *(
                str(value)
                for value in mappings[(task, next(target for t, target in mappings if t == task))][
                    "rewrite_validation"
                ]["object_symbols"].values()
            ),
            RECEPTACLE_SYMBOL,
        }
        for task in range(10)
    }

    smoke = load_json(args.smoke)
    smoke_seed = validate_result_identity(
        smoke,
        args.smoke,
        "strict_smoke",
        manifest_sha,
        instruction_sha,
        provenance_sha,
        sources_start,
    )
    if smoke_seed != 0 or smoke["evaluation"] != {
        "task_start": 0,
        "task_end": 10,
        "episode_start": 20,
        "eps_per_task": 1,
        "expected_rollouts": 0,
        "completed_rollouts": 0,
        "completed_four_arm_units": 0,
        "smoke_units": 50,
        "matmul_precision": "highest",
    }:
        raise RuntimeError("Strict smoke evaluation matrix differs")
    if smoke["four_arm_units"] or len(smoke["smoke_units"]) != 50:
        raise RuntimeError("Strict smoke row matrix differs")
    seen_smoke = set()
    reference_starts: dict[tuple[int, int], dict[str, Any]] = {}
    for row in smoke["smoke_units"]:
        task = int(row["task_index"])
        episode = int(row["episode"])
        target = int(row["counterfactual_prompt_id"])
        key = (task, target)
        if episode != 20 or key not in mappings or key in seen_smoke:
            raise RuntimeError("Smoke mapping identity differs")
        seen_smoke.add(key)
        expected_order = condition_order(0, task, episode, target)
        if row["condition_order"] != expected_order or list(row["conditions"]) != expected_order:
            raise RuntimeError("Smoke condition order differs")
        if (task, episode) not in reference_starts:
            reference_starts[(task, episode)] = row["physical_input"]
        elif reference_starts[(task, episode)] != row["physical_input"]:
            raise RuntimeError("Smoke target rewrites changed the physical input")
        expected = {
            "original_goal_original_prompt": (task, task),
            "original_goal_counterfactual_prompt": (task, target),
            "counterfactual_goal_counterfactual_prompt": (target, target),
            "counterfactual_goal_original_prompt": (target, task),
        }
        for condition, value in row["conditions"].items():
            goal_id, prompt_id = expected[condition]
            if (
                int(value["goal_prompt_id"]) != goal_id
                or int(value["prompt_id"]) != prompt_id
                or value["instruction_ids"] != encoded[prompt_id]
            ):
                raise RuntimeError("Smoke condition identity differs")
            chunk = decode_array(value["action_chunk"])
            if chunk.dtype != np.dtype("<f4") or chunk.shape != (8, 7) or not np.isfinite(chunk).all():
                raise RuntimeError("Smoke action chunk differs")
    if seen_smoke != set(mappings):
        raise RuntimeError("Smoke does not cover all fifty mappings")

    expected_paths = {
        (root / f"results/counterfactual_goal_following_v1_s{seed}_t{start}_{end}.json").resolve()
        for seed in range(3)
        for start, end in TASK_SHARDS
    }
    if {path.resolve() for path in args.result} != expected_paths:
        raise RuntimeError("Full-result path matrix differs")

    records: list[dict[str, Any]] = []
    observed = set()
    result_identities = {}
    cross_result_starts: dict[tuple[int, int], dict[str, Any]] = {}
    for path in args.result:
        result = load_json(path)
        seed = validate_result_identity(
            result,
            path,
            "full",
            manifest_sha,
            instruction_sha,
            provenance_sha,
            sources_start,
        )
        evaluation = result["evaluation"]
        bounds = (int(evaluation["task_start"]), int(evaluation["task_end"]))
        if bounds not in TASK_SHARDS or (
            int(evaluation["episode_start"]),
            int(evaluation["eps_per_task"]),
            int(evaluation["expected_rollouts"]),
            int(evaluation["completed_rollouts"]),
            int(evaluation["completed_four_arm_units"]),
            int(evaluation["smoke_units"]),
            evaluation["matmul_precision"],
        ) != (20, 10, 400, 400, 100, 0, "highest"):
            raise RuntimeError("Full evaluation matrix differs")
        if result["smoke_units"] or len(result["four_arm_units"]) != 100:
            raise RuntimeError("Full row matrix differs")
        result_identities[str(path)] = {"sha256": file_sha256(path), "seed": seed, "bounds": list(bounds)}
        from athena.run_xvla_experiment import load_suite, official_init_states

        suite = load_suite("libero_object")
        canonical = official_init_states(suite, bounds[0])
        canonical_by_task = {task: official_init_states(suite, task) for task in range(bounds[0], bounds[1])}
        del canonical
        for row in result["four_arm_units"]:
            task = int(row["task_index"])
            episode = int(row["episode"])
            target = int(row["counterfactual_prompt_id"])
            key = (seed, task, episode, target)
            mapping = mappings.get((task, target))
            if (
                key in observed
                or mapping is None
                or not bounds[0] <= task < bounds[1]
                or episode not in range(20, 30)
            ):
                raise RuntimeError("Full four-arm identity differs")
            observed.add(key)
            if (
                int(row["init_state_index"]) != episode
                or int(row["reset_seed"]) != task * 100 + episode
                or row["init_state_sha256"] != hash_array(np.asarray(canonical_by_task[task][episode]))
                or int(row["original_prompt_id"]) != task
                or row["original_bddl_sha256"] != mapping["original_bddl_sha256"]
                or row["rewritten_bddl_sha256"] != mapping["rewritten_bddl_sha256"]
            ):
                raise RuntimeError("Full canonical or BDDL identity differs")
            order = condition_order(seed, task, episode, target)
            if row["condition_order"] != order or list(row["conditions"]) != order:
                raise RuntimeError("Full condition order differs")
            expected = {
                "original_goal_original_prompt": (task, task),
                "original_goal_counterfactual_prompt": (task, target),
                "counterfactual_goal_counterfactual_prompt": (target, target),
                "counterfactual_goal_original_prompt": (target, task),
            }
            starts = []
            record: dict[str, Any] = {
                "seed": seed,
                "task": task,
                "episode": episode,
                "target": target,
            }
            for condition, value in row["conditions"].items():
                goal_id, prompt_id = expected[condition]
                rollout = value["rollout"]
                if (
                    int(value["goal_prompt_id"]) != goal_id
                    or int(value["prompt_id"]) != prompt_id
                    or value["instruction_ids"] != encoded[prompt_id]
                ):
                    raise RuntimeError("Full arm identity differs")
                starts.append(rollout["start_physical"])
                record[condition] = validate_rollout(
                    rollout,
                    prompt_id,
                    encoded[prompt_id],
                    rollout["start_physical"],
                    bodies_by_task[task],
                )
            if not all(start == starts[0] for start in starts[1:]):
                raise RuntimeError("Four arms do not share an exact physical input")
            if row["physical_input_sha256"] != starts[0]["combined_sha256"]:
                raise RuntimeError("Four-arm physical input hash differs")
            state_key = (task, episode)
            if state_key not in cross_result_starts:
                cross_result_starts[state_key] = starts[0]
            elif cross_result_starts[state_key] != starts[0]:
                raise RuntimeError("Physical input differs across targets or checkpoints")
            records.append(record)
    expected = {
        (seed, task, episode, target)
        for seed in range(3)
        for task, target in mappings
        for episode in range(20, 30)
    }
    if observed != expected or len(records) != 1500 or len(cross_result_starts) != 100:
        raise RuntimeError("The complete 1,500-unit matrix is absent")

    original_names = ("original_goal_original_prompt", "original_goal_counterfactual_prompt")
    counter_names = (
        "counterfactual_goal_counterfactual_prompt",
        "counterfactual_goal_original_prompt",
    )
    original_pooled = paired_metric(records, *original_names)
    counter_pooled = paired_metric(records, *counter_names)

    def gap_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
        original = float(np.mean([int(row[original_names[0]]) - int(row[original_names[1]]) for row in rows]))
        counter = float(np.mean([int(row[counter_names[0]]) - int(row[counter_names[1]]) for row in rows]))
        return {"original_goal_gap": original, "counterfactual_goal_gap": counter, "symmetric_gap": (original + counter) / 2}

    pooled_gaps = gap_rows(records)
    by_checkpoint = {
        str(seed): {
            **gap_rows([row for row in records if row["seed"] == seed]),
            "original_goal_matching_success": float(np.mean([row[original_names[0]] for row in records if row["seed"] == seed])),
            "counterfactual_goal_matching_success": float(np.mean([row[counter_names[0]] for row in records if row["seed"] == seed])),
        }
        for seed in range(3)
    }
    by_task = {
        str(task): gap_rows([row for row in records if row["task"] == task])
        for task in range(10)
    }
    counterfactual_task_success = {
        str(task): float(np.mean([row[counter_names[0]] for row in records if row["task"] == task]))
        for task in range(10)
    }
    counterfactual_task_macro = float(np.mean(list(counterfactual_task_success.values())))

    task_sums: dict[str, list[int]] = {
        name: [] for name in ("original_goal", "counterfactual_goal", "symmetric")
    }
    for task in range(10):
        rows = [row for row in records if row["task"] == task]
        if len(rows) != 150:
            raise RuntimeError("Task exact-test cluster does not retain all 150 paired units")
        original_sum = sum(
            int(row[original_names[0]]) - int(row[original_names[1]]) for row in rows
        )
        counter_sum = sum(
            int(row[counter_names[0]]) - int(row[counter_names[1]]) for row in rows
        )
        task_sums["original_goal"].append(original_sum)
        task_sums["counterfactual_goal"].append(counter_sum)
        task_sums["symmetric"].append(original_sum + counter_sum)
    task_exact_p = {
        name: one_sided_exact_signflip_p(values) for name, values in task_sums.items()
    }
    task_consistency = {}
    for metric in ("original_goal_gap", "counterfactual_goal_gap", "symmetric_gap"):
        gaps = [by_task[str(task)][metric] for task in range(10)]
        task_consistency[metric] = {
            "strictly_positive_tasks": sum(gap > 0 for gap in gaps),
            "negative_tasks": sum(gap < 0 for gap in gaps),
            "gaps": {str(task): gaps[task] for task in range(10)},
        }
    by_target_pair = {
        f"{task}->{target}": gap_rows(
            [row for row in records if row["task"] == task and row["target"] == target]
        )
        for task, target in sorted(mappings)
    }

    gates = {
        "original_goal_matching_success_at_least_0p60_every_checkpoint": all(
            row["original_goal_matching_success"] >= 0.60 for row in by_checkpoint.values()
        ),
        "counterfactual_goal_matching_task_macro_at_least_0p25": counterfactual_task_macro >= 0.25,
        "counterfactual_goal_matching_at_least_0p20_every_checkpoint": all(
            row["counterfactual_goal_matching_success"] >= 0.20 for row in by_checkpoint.values()
        ),
        "original_goal_gap_at_least_0p20": pooled_gaps["original_goal_gap"] >= 0.20,
        "counterfactual_goal_gap_at_least_0p20": pooled_gaps["counterfactual_goal_gap"] >= 0.20,
        "symmetric_gap_at_least_0p20": pooled_gaps["symmetric_gap"] >= 0.20,
        "all_three_gaps_positive_every_checkpoint": all(
            row[metric] > 0
            for row in by_checkpoint.values()
            for metric in ("original_goal_gap", "counterfactual_goal_gap", "symmetric_gap")
        ),
        "both_directional_mcnemar_p_at_most_0p01": (
            original_pooled["one_sided_exact_mcnemar_p"] <= 0.01
            and counter_pooled["one_sided_exact_mcnemar_p"] <= 0.01
        ),
        "all_task_signflip_p_at_most_0p01": all(
            value <= 0.01 for value in task_exact_p.values()
        ),
        "at_least_eight_positive_tasks_each_metric": all(
            row["strictly_positive_tasks"] >= 8 for row in task_consistency.values()
        ),
        "no_negative_task_gap_each_metric": all(
            row["negative_tasks"] == 0 for row in task_consistency.values()
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
        "instruction_prerequisite": {
            "path": str(args.instruction_summary),
            "sha256": instruction_sha,
            "identity_validated": instruction["identity_validated"],
            "overall_pass": instruction["overall_pass"],
            "claim_eligible": instruction["claim_eligible"],
        },
        "manifest": {"path": str(args.manifest), "sha256": manifest_sha},
        "smoke": {"path": str(args.smoke), "sha256": file_sha256(args.smoke)},
        "full_results": result_identities,
        "source_sha256_start": sources_start,
        "source_sha256_end": source_hashes(root),
        "libero_source_start": libero_start,
        "libero_source_end": validate_libero_runtime(),
        "original_goal_direction": original_pooled,
        "counterfactual_goal_direction": counter_pooled,
        "pooled_gaps": pooled_gaps,
        "counterfactual_goal_matching_task_macro_success": counterfactual_task_macro,
        "counterfactual_goal_matching_success_by_task": counterfactual_task_success,
        "by_checkpoint": by_checkpoint,
        "by_task": by_task,
        "by_target_pair_descriptive": by_target_pair,
        "task_cluster_sums_retaining_all_checkpoints_states_and_targets": task_sums,
        "task_cluster_one_sided_exact_signflip_p": task_exact_p,
        "task_consistency": task_consistency,
        "gates": gates,
        "claim_if_passed": (
            "Across all prespecified familiar co-present target swaps, matching the prompt "
            "to either the original or counterfactual synthetic BDDL goal improves paired "
            "closed-loop success by at least 20 percentage points, with checkpoint- and "
            "task-consistent effects."
        ),
        "claim_boundaries": (
            "This establishes counterfactual goal following only for familiar vocabulary, "
            "objects, Object-suite scenes, synthetic target-only BDDL rewrites, and the three "
            "fixed checkpoints. Episodes 20 through 29 overlap the failed local-rank state set "
            "but not the trajectory-shift or instruction-necessity state sets. It does not "
            "establish paraphrase, policy-unseen, cross-suite, physical-robot, or broad "
            "compositional generalization, or a benefit unique to tensor decomposability."
            " Pooled McNemar tests are exact over paired four-arm units and are not "
            "cluster-robust. Task sign-flip tests retain all checkpoints, states, and "
            "targets together within each of ten task sums."
        ),
    }
    if output["source_sha256_start"] != output["source_sha256_end"] or output["libero_source_start"] != output["libero_source_end"]:
        raise RuntimeError("Sources changed during summary")
    for path_text, identity in result_identities.items():
        if file_sha256(Path(path_text)) != identity["sha256"]:
            raise RuntimeError("A full result changed during summary")
    write_json(args.output, output)
    print(
        "RESULT",
        json.dumps(
            {
                "overall_pass": overall_pass,
                "pooled_gaps": pooled_gaps,
                "task_exact_p": task_exact_p,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
