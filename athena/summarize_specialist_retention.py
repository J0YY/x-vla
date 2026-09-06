#!/usr/bin/env python3
"""Score the frozen four-suite generalist-to-specialist retention gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
SHARDS = ((0, 3), (3, 6), (6, 8), (8, 10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/generalist_specialist_retention_summary.json"),
    )
    return parser.parse_args()


def result_path(
    results_dir: Path,
    suite: str,
    kind: str,
    start: int,
    end: int,
) -> Path:
    if kind == "generalist":
        name = f"generalist_chi_s0_{suite}_t{start}_{end}.json"
    elif suite == "libero_object":
        name = f"taskmap_chi_s0_t{start}_{end}.json"
    else:
        name = f"corrected_specialist_{suite}_chi_s0_t{start}_{end}.json"
    return results_dir / name


def load_policy(
    results_dir: Path,
    suite: str,
    kind: str,
) -> dict[str, Any]:
    task_outcomes = {task: [] for task in range(10)}
    seen = set()
    sources = []
    for start, end in SHARDS:
        path = result_path(results_dir, suite, kind, start, end)
        with path.open() as handle:
            result = json.load(handle)
        sources.append(str(path))
        expected_checkpoint = (
            "ckpt_generalist_chi_s0.pt"
            if kind == "generalist"
            else (
                "ckpt_taskmap_chi_s0.pt"
                if suite == "libero_object"
                else f"ckpt_indomain_{suite}_chi_s0.pt"
            )
        )
        identity = (
            result.get("mode"),
            result.get("architecture"),
            result.get("vision_encoder"),
            result.get("suite"),
            int(result.get("seed", -1)),
            result.get("matmul_precision"),
            Path(str(result.get("checkpoint", ""))).name,
        )
        expected_identity = (
            "capability",
            "chi",
            "vit",
            suite,
            0,
            "highest",
            expected_checkpoint,
        )
        if identity != expected_identity:
            raise RuntimeError(
                f"Identity mismatch in {path}: {identity} != {expected_identity}"
            )
        capability = result["capability"]
        expected_tasks = list(range(start, end))
        if capability.get("task_indices") != expected_tasks:
            raise RuntimeError(f"Task shard mismatch in {path}")
        if capability.get("eps_per_task") != 50:
            raise RuntimeError(f"Episode count mismatch in {path}")
        if capability.get("max_steps") != 280:
            raise RuntimeError(f"Horizon mismatch in {path}")
        if capability.get("canonical_init_states") is not True:
            raise RuntimeError(f"Noncanonical initialization in {path}")
        for episode in capability["episodes"]:
            task = int(episode["task_index"])
            episode_index = int(episode["episode"])
            key = (task, episode_index)
            if task not in expected_tasks or not 0 <= episode_index < 50:
                raise RuntimeError(f"Out-of-range episode in {path}")
            if key in seen:
                raise RuntimeError(f"Duplicate episode {key} in {path}")
            seen.add(key)
            task_outcomes[task].append(bool(episode["success"]))
    if len(seen) != 500 or any(len(rows) != 50 for rows in task_outcomes.values()):
        raise RuntimeError(f"Incomplete {kind} evaluation for {suite}")
    task_rates = {
        str(task): sum(outcomes) / len(outcomes)
        for task, outcomes in task_outcomes.items()
    }
    return {
        "task_success": task_rates,
        "macro_task_success": sum(task_rates.values()) / 10,
        "successes": sum(sum(outcomes) for outcomes in task_outcomes.values()),
        "trials": 500,
        "sources": sources,
    }


def main() -> None:
    args = parse_args()
    by_suite = {}
    for suite in SUITES:
        specialist = load_policy(args.results_dir, suite, "specialist")
        generalist = load_policy(args.results_dir, suite, "generalist")
        denominator = specialist["macro_task_success"]
        ratio = generalist["macro_task_success"] / denominator if denominator else 0.0
        by_suite[suite] = {
            "specialist": specialist,
            "generalist": generalist,
            "generalist_to_specialist_ratio": ratio,
        }
    specialist_macro = sum(
        row["specialist"]["macro_task_success"] for row in by_suite.values()
    ) / len(SUITES)
    generalist_macro = sum(
        row["generalist"]["macro_task_success"] for row in by_suite.values()
    ) / len(SUITES)
    pooled_ratio = generalist_macro / specialist_macro if specialist_macro else 0.0
    output = {
        "scope": {
            "suites": list(SUITES),
            "seed": 0,
            "episodes_per_task": 50,
            "max_steps": 280,
            "matmul_precision": "highest",
            "selection": "final EMA checkpoints only",
        },
        "by_suite": by_suite,
        "specialist_four_suite_macro": specialist_macro,
        "generalist_four_suite_macro": generalist_macro,
        "generalist_to_specialist_ratio": pooled_ratio,
        "gate": {
            "threshold": 0.85,
            "pass": pooled_ratio >= 0.85,
            "definition": "ratio of four-suite task-macro means",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
