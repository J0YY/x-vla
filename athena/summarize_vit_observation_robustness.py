#!/usr/bin/env python3
"""Strict validation and aggregation for ViT observation robustness."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from athena.run_vit_observation_robustness import (
    CONDITIONS,
    CORRUPTIONS,
    FROZEN_CACHE,
    FROZEN_CHECKPOINTS,
    FROZEN_DATASET_METADATA,
    FROZEN_MANIFEST,
    FROZEN_PROVENANCE,
    PROTOCOL,
    SCHEMA,
    SHARDS,
    file_sha256,
    source_hashes,
    write_json,
)

SUMMARY_SCHEMA = "xvla-vit-observation-robustness-summary-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--provenance-result", type=Path, required=True)
    parser.add_argument("--capability-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} differs from the frozen value")


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def validate_live_inputs(args: argparse.Namespace) -> dict[int, Path]:
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite the robustness summary")
    if (
        args.cache.name != FROZEN_CACHE["basename"]
        or file_sha256(args.cache) != FROZEN_CACHE["sha256"]
    ):
        raise RuntimeError("Live cache identity differs")
    if (
        args.provenance_result.name != FROZEN_PROVENANCE["basename"]
        or file_sha256(args.provenance_result) != FROZEN_PROVENANCE["sha256"]
    ):
        raise RuntimeError("Live provenance identity differs")
    if (
        args.capability_manifest.name != FROZEN_MANIFEST["basename"]
        or file_sha256(args.capability_manifest) != FROZEN_MANIFEST["sha256"]
    ):
        raise RuntimeError("Live capability-manifest identity differs")
    if len(args.checkpoint) != 3:
        raise RuntimeError("Exactly three live checkpoint files are required")
    checkpoints = {}
    for path in args.checkpoint:
        seeds = [
            seed
            for seed, row in FROZEN_CHECKPOINTS.items()
            if row["basename"] == path.name
        ]
        if (
            len(seeds) != 1
            or file_sha256(path) != FROZEN_CHECKPOINTS[seeds[0]]["sha256"]
        ):
            raise RuntimeError(f"Live checkpoint identity differs: {path}")
        checkpoints[seeds[0]] = path
    require_equal(set(checkpoints), set(FROZEN_CHECKPOINTS), "checkpoint seed coverage")
    return checkpoints


def validate_result(
    path: Path,
    payload: dict[str, Any],
    expected_sources: dict[str, str],
) -> list[dict[str, Any]]:
    require_equal(payload.get("schema"), SCHEMA, f"{path} schema")
    require_equal(payload.get("protocol"), PROTOCOL, f"{path} protocol")
    require_equal(payload.get("mode"), "full", f"{path} mode")
    identity = payload.get("identity", {})
    seed = int(identity.get("checkpoint_seed", -1))
    if seed not in FROZEN_CHECKPOINTS:
        raise RuntimeError(f"{path} checkpoint seed is invalid")
    checkpoint = FROZEN_CHECKPOINTS[seed]
    require_equal(
        identity.get("checkpoint_basename"),
        checkpoint["basename"],
        f"{path} checkpoint",
    )
    require_equal(
        identity.get("checkpoint_sha256"),
        checkpoint["sha256"],
        f"{path} checkpoint SHA",
    )
    require_equal(
        identity.get("cache_sha256"), FROZEN_CACHE["sha256"], f"{path} cache SHA"
    )
    require_equal(
        identity.get("provenance_sha256"),
        FROZEN_PROVENANCE["sha256"],
        f"{path} provenance SHA",
    )
    require_equal(
        identity.get("provenance_job_id"),
        FROZEN_PROVENANCE["job_id"],
        f"{path} provenance job",
    )
    require_equal(
        identity.get("provenance_canonical_content_sha256"),
        FROZEN_PROVENANCE["canonical_content_sha256"],
        f"{path} canonical content SHA",
    )
    require_equal(
        identity.get("capability_manifest_sha256"),
        FROZEN_MANIFEST["sha256"],
        f"{path} manifest SHA",
    )
    require_equal(
        identity.get("dataset_metadata"),
        FROZEN_DATASET_METADATA,
        f"{path} dataset metadata",
    )
    require_equal(
        identity.get("source_sha256"), expected_sources, f"{path} source hashes"
    )
    runtime = payload.get("runtime", {})
    require_equal(runtime.get("gpu"), PROTOCOL["gpu_name"], f"{path} GPU")
    require_equal(
        runtime.get("matmul_precision"),
        PROTOCOL["matmul_precision"],
        f"{path} matmul precision",
    )
    if not all(
        runtime.get(key)
        for key in ("torch", "cuda", "numpy", "mujoco", "robosuite", "libero")
    ):
        raise RuntimeError(f"{path} runtime identity is incomplete")

    shard = (int(payload.get("task_start", -1)), int(payload.get("task_end", -1)))
    if shard not in SHARDS:
        raise RuntimeError(f"{path} has an invalid task shard")
    require_equal(
        payload.get("episodes_per_task"),
        PROTOCOL["episodes_per_task"],
        f"{path} episode count",
    )
    expected_tasks = list(range(*shard))
    task_protocol = payload.get("task_protocol", {})
    require_equal(
        sorted(int(key) for key in task_protocol),
        expected_tasks,
        f"{path} task catalog",
    )
    for task in expected_tasks:
        task_row = task_protocol[str(task)]
        if set(task_row) != {
            "language",
            "problem_folder",
            "bddl_file",
            "bddl_sha256",
            "init_state_count",
            "init_states_sha256",
            "instruction_ids",
        }:
            raise RuntimeError(f"{path} task {task} protocol fields differ")
        if (
            not task_row["language"]
            or not task_row["problem_folder"]
            or not task_row["bddl_file"]
            or len(task_row["bddl_sha256"]) != 64
            or int(task_row["init_state_count"]) < PROTOCOL["episodes_per_task"]
            or len(task_row["init_states_sha256"]) != 64
            or not isinstance(task_row["instruction_ids"], list)
        ):
            raise RuntimeError(f"{path} task {task} protocol identity is invalid")

    rows = payload.get("episodes")
    expected_count = (
        len(expected_tasks) * PROTOCOL["episodes_per_task"] * len(CONDITIONS)
    )
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise RuntimeError(f"{path} episode row count differs")
    catalog = {}
    for row in rows:
        key = (
            int(row.get("task_index", -1)),
            int(row.get("episode", -1)),
            row.get("condition"),
        )
        if key in catalog:
            raise RuntimeError(f"{path} contains a duplicate episode condition")
        task, episode, condition = key
        if (
            task not in expected_tasks
            or episode not in range(PROTOCOL["episodes_per_task"])
            or condition not in CONDITIONS
        ):
            raise RuntimeError(f"{path} has an out-of-protocol episode key")
        if not isinstance(row.get("success"), bool):
            raise RuntimeError(f"{path} success is not Boolean")
        steps = int(row.get("steps", -1))
        decisions = int(row.get("decisions", -1))
        if not 0 < steps <= PROTOCOL["max_steps"] or not 0 < decisions <= math.ceil(
            steps / 8
        ):
            raise RuntimeError(f"{path} has invalid control counts")
        for field in (
            "canonical_init_state_sha256",
            "post_settle_simulator_state_sha256",
            "first_model_image_sha256",
            "model_image_stream_sha256",
            "executed_action_stream_sha256",
        ):
            value = row.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise RuntimeError(f"{path} has an invalid {field}")
        if not math.isfinite(float(row.get("elapsed_s", float("nan")))):
            raise RuntimeError(f"{path} has non-finite timing")
        catalog[key] = row
    for task in expected_tasks:
        for episode in range(PROTOCOL["episodes_per_task"]):
            paired = [catalog[(task, episode, condition)] for condition in CONDITIONS]
            require_equal(
                len({row["canonical_init_state_sha256"] for row in paired}),
                1,
                f"{path} paired canonical state {task}/{episode}",
            )
            require_equal(
                len({row["post_settle_simulator_state_sha256"] for row in paired}),
                1,
                f"{path} paired settled state {task}/{episode}",
            )
            require_equal(
                len({row["first_model_image_sha256"] for row in paired}),
                len(CONDITIONS),
                f"{path} distinct first condition images {task}/{episode}",
            )

    aggregate = payload.get("aggregate", {})
    require_equal(
        int(aggregate.get("trials", -1)), expected_count, f"{path} aggregate trials"
    )
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        successes = sum(row["success"] for row in selected)
        require_equal(
            aggregate.get("trials_by_condition", {}).get(condition),
            len(selected),
            f"{path} {condition} trials",
        )
        require_equal(
            aggregate.get("successes_by_condition", {}).get(condition),
            successes,
            f"{path} {condition} successes",
        )
        require_equal(
            aggregate.get("success_rate_by_condition", {}).get(condition),
            successes / len(selected),
            f"{path} {condition} rate",
        )
    if (
        not isinstance(payload.get("claim_boundary"), str)
        or not payload["claim_boundary"]
    ):
        raise RuntimeError(f"{path} claim boundary is absent")
    return [
        {**row, "checkpoint_seed": seed, "source_result": str(path)} for row in rows
    ]


def paired_bootstrap(rows: list[dict[str, Any]], corruption: str) -> dict[str, Any]:
    cells = []
    for seed in sorted(FROZEN_CHECKPOINTS):
        for task in range(10):
            catalog = {
                (int(row["episode"]), row["condition"]): float(row["success"])
                for row in rows
                if row["checkpoint_seed"] == seed and row["task_index"] == task
            }
            cells.append(
                np.asarray(
                    [
                        [catalog[(episode, "clean")], catalog[(episode, corruption)]]
                        for episode in range(10)
                    ],
                    dtype=np.float64,
                )
            )
    array = np.stack(cells)
    draws = int(PROTOCOL["bootstrap"]["draws"])
    condition_offset = CORRUPTIONS.index(corruption)
    rng = np.random.default_rng(int(PROTOCOL["bootstrap"]["seed"]) + condition_offset)
    indices = rng.integers(
        0, array.shape[1], size=(draws, array.shape[0], array.shape[1])
    )
    clean = np.take_along_axis(array[None, :, :, 0], indices, axis=2).mean(axis=(1, 2))
    shifted = np.take_along_axis(array[None, :, :, 1], indices, axis=2).mean(
        axis=(1, 2)
    )
    if bool((clean <= 0).any()):
        raise RuntimeError("Bootstrap encountered a zero clean denominator")
    retention = shifted / clean
    lo, hi = PROTOCOL["bootstrap"]["interval"]
    return {
        "draws": draws,
        "seed": int(PROTOCOL["bootstrap"]["seed"]) + condition_offset,
        "retention_mean": float(retention.mean()),
        "retention_interval": [
            float(np.quantile(retention, lo)),
            float(np.quantile(retention, hi)),
        ],
        "difference_interval": [
            float(np.quantile(shifted - clean, lo)),
            float(np.quantile(shifted - clean, hi)),
        ],
    }


def main() -> None:
    args = parse_args()
    validate_live_inputs(args)
    if len(args.result) != 12 or len({path.resolve() for path in args.result}) != 12:
        raise RuntimeError("Exactly twelve distinct full shard results are required")
    expected_sources = source_hashes()
    all_rows = []
    shard_coverage = set()
    result_identities = []
    task_protocol_catalog = {}
    for path in args.result:
        payload = load_json(path)
        rows = validate_result(path, payload, expected_sources)
        identity = (
            rows[0]["checkpoint_seed"],
            int(payload["task_start"]),
            int(payload["task_end"]),
        )
        if identity in shard_coverage:
            raise RuntimeError("Duplicate checkpoint/task shard")
        shard_coverage.add(identity)
        all_rows.extend(rows)
        for task, task_protocol in payload["task_protocol"].items():
            if task in task_protocol_catalog:
                require_equal(
                    task_protocol,
                    task_protocol_catalog[task],
                    f"task {task} protocol across checkpoints",
                )
            else:
                task_protocol_catalog[task] = task_protocol
        result_identities.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "seed": identity[0],
                "task_start": identity[1],
                "task_end": identity[2],
            }
        )
    expected_coverage = {
        (seed, start, end) for seed in FROZEN_CHECKPOINTS for start, end in SHARDS
    }
    require_equal(shard_coverage, expected_coverage, "full shard coverage")
    require_equal(
        set(task_protocol_catalog),
        {str(task) for task in range(10)},
        "task protocol coverage",
    )
    require_equal(len(all_rows), 1200, "total paired trial count")

    successes = defaultdict(int)
    trials = defaultdict(int)
    for row in all_rows:
        key = (row["checkpoint_seed"], row["condition"])
        successes[key] += int(row["success"])
        trials[key] += 1
    clean_total = sum(successes[(seed, "clean")] for seed in FROZEN_CHECKPOINTS)
    corruption_rows = {}
    gates = {}
    lower_gate = float(
        PROTOCOL["primary_gates"]["each_corruption_retention_ci_lower_at_least"]
    )
    checkpoint_gate = float(
        PROTOCOL["primary_gates"]["each_checkpoint_point_retention_at_least"]
    )
    for corruption in CORRUPTIONS:
        bootstrap = paired_bootstrap(all_rows, corruption)
        by_checkpoint = {}
        for seed in FROZEN_CHECKPOINTS:
            clean = successes[(seed, "clean")] / trials[(seed, "clean")]
            shifted = successes[(seed, corruption)] / trials[(seed, corruption)]
            by_checkpoint[str(seed)] = {
                "clean_successes": successes[(seed, "clean")],
                "shifted_successes": successes[(seed, corruption)],
                "trials": trials[(seed, "clean")],
                "clean_rate": clean,
                "shifted_rate": shifted,
                "point_retention": shifted / clean,
            }
        shifted_total = sum(
            successes[(seed, corruption)] for seed in FROZEN_CHECKPOINTS
        )
        corruption_rows[corruption] = {
            "clean_successes": clean_total,
            "shifted_successes": shifted_total,
            "trials_per_condition": 300,
            "point_retention": shifted_total / clean_total,
            "bootstrap": bootstrap,
            "by_checkpoint": by_checkpoint,
        }
        gates[f"{corruption}_ci_lower_at_least_{lower_gate}"] = (
            bootstrap["retention_interval"][0] >= lower_gate
        )
        gates[f"{corruption}_every_checkpoint_at_least_{checkpoint_gate}"] = all(
            row["point_retention"] >= checkpoint_gate for row in by_checkpoint.values()
        )
    gates["primary_observation_robustness_pass"] = all(gates.values())

    summary = {
        "schema": SUMMARY_SCHEMA,
        "protocol": PROTOCOL,
        "identity": {
            "checkpoints": FROZEN_CHECKPOINTS,
            "cache": FROZEN_CACHE,
            "provenance": FROZEN_PROVENANCE,
            "capability_manifest": FROZEN_MANIFEST,
            "dataset_metadata": FROZEN_DATASET_METADATA,
            "source_sha256": expected_sources,
            "raw_results": sorted(
                result_identities, key=lambda row: (row["seed"], row["task_start"])
            ),
        },
        "aggregate": {
            "checkpoints": 3,
            "tasks": 10,
            "paired_episodes_per_checkpoint_task": 10,
            "conditions": len(CONDITIONS),
            "total_closed_loop_trials": len(all_rows),
            "clean_successes": clean_total,
            "corruptions": corruption_rows,
        },
        "primary_gates": gates,
        "claim_if_passed": (
            "Across three immutable rational ViT policy checkpoints and all ten familiar Object "
            "tasks, each of three prespecified mild camera-stream shifts retains a paired-bootstrap "
            "lower bound of at least 75 percent of matched clean closed-loop success, and every "
            "checkpoint retains at least 70 percent pointwise."
        ),
        "boundaries": [
            "The perturbations are fixed mild synthetic camera-stream shifts, not adversarial attacks or a standardized broad robustness benchmark.",
            "All tasks, prompts, objects, scenes, and checkpoints are familiar LIBERO-Object training-domain entities.",
            "The experiment does not establish cross-suite, unseen-language, physical-robot, or architecture-comparative robustness.",
            "The bootstrap resamples paired episodes within 30 fixed checkpoint-task cells and does not estimate a population of tasks or checkpoints.",
        ],
    }
    if not gates["primary_observation_robustness_pass"]:
        write_json(args.output, summary)
        raise RuntimeError("Frozen observation-robustness gate did not pass")
    write_json(args.output, summary)
    print(
        json.dumps(
            {"primary_gates": gates, "clean_successes": clean_total},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
