#!/usr/bin/env python3
"""Aggregate the fixed six-checkpoint, four-suite generalist evaluation."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ARCHITECTURES = ("chi", "conventional")
SEEDS = (0, 1, 2)
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
SHARDS = ((0, 3), (3, 6), (6, 8), (8, 10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/generalist_summary.json"),
    )
    return parser.parse_args()


def load_checkpoint_results(
    results_dir: Path,
    architecture: str,
    seed: int,
) -> dict[str, Any]:
    task_rows: dict[tuple[str, int], list[bool]] = defaultdict(list)
    source_files = []
    seen_episodes = set()
    expected_checkpoint = f"ckpt_generalist_{architecture}_s{seed}.pt"
    expected_metadata = f"ckpt_generalist_{architecture}_s{seed}.json"
    for suite in SUITES:
        for start, end in SHARDS:
            path = (
                results_dir
                / f"generalist_{architecture}_s{seed}_{suite}_t{start}_{end}.json"
            )
            with path.open() as handle:
                result = json.load(handle)
            source_files.append(str(path))
            expected_tasks = list(range(start, end))
            identity = {
                "mode": result.get("mode"),
                "architecture": result.get("architecture"),
                "vision_encoder": result.get("vision_encoder"),
                "suite": result.get("suite"),
                "seed": result.get("seed"),
                "checkpoint": Path(str(result.get("checkpoint", ""))).name,
                "metadata": Path(str(result.get("model_metadata", ""))).name,
                "precision": result.get("matmul_precision"),
            }
            expected_identity = {
                "mode": "capability",
                "architecture": architecture,
                "vision_encoder": "vit",
                "suite": suite,
                "seed": seed,
                "checkpoint": expected_checkpoint,
                "metadata": expected_metadata,
                "precision": "highest",
            }
            if identity != expected_identity:
                raise RuntimeError(
                    f"Result identity mismatch in {path}: {identity} != {expected_identity}"
                )
            if result.get("evaluation_scope") != (
                "In-domain evaluation of a jointly trained multi-suite checkpoint"
            ):
                raise RuntimeError(f"Unexpected evaluation scope in {path}")
            metadata = result.get("training_metadata") or {}
            if metadata.get("format") != "xvla_multisuite_checkpoint_v1":
                raise RuntimeError(f"Invalid training metadata format in {path}")
            if metadata.get("architecture") != architecture:
                raise RuntimeError(f"Training architecture mismatch in {path}")
            if int(metadata.get("seed", -1)) != seed:
                raise RuntimeError(f"Training seed mismatch in {path}")
            if Path(str(metadata.get("checkpoint", ""))).name != expected_checkpoint:
                raise RuntimeError(f"Training checkpoint mismatch in {path}")
            if not metadata.get("checkpoint_sha256"):
                raise RuntimeError(f"Missing checkpoint hash in {path}")
            capability = result["capability"]
            protocol = {
                "task_indices": capability.get("task_indices"),
                "eps_per_task": capability.get("eps_per_task"),
                "max_steps": capability.get("max_steps"),
                "canonical_init_states": capability.get("canonical_init_states"),
            }
            expected_protocol = {
                "task_indices": expected_tasks,
                "eps_per_task": 50,
                "max_steps": 280,
                "canonical_init_states": True,
            }
            if protocol != expected_protocol:
                raise RuntimeError(
                    f"Evaluation protocol mismatch in {path}: "
                    f"{protocol} != {expected_protocol}"
                )
            expected_trials = len(expected_tasks) * 50
            if int(capability.get("trials", -1)) != expected_trials:
                raise RuntimeError(f"Trial count mismatch in {path}")
            shard_episodes = set()
            for episode in capability["episodes"]:
                task = int(episode["task_index"])
                episode_index = int(episode["episode"])
                if task not in expected_tasks or not 0 <= episode_index < 50:
                    raise RuntimeError(f"Out-of-range episode in {path}: {episode}")
                episode_key = (suite, task, episode_index)
                if episode_key in shard_episodes or episode_key in seen_episodes:
                    raise RuntimeError(f"Duplicate episode {episode_key} in {path}")
                shard_episodes.add(episode_key)
                seen_episodes.add(episode_key)
                key = (suite, task)
                task_rows[key].append(bool(episode["success"]))
            if len(shard_episodes) != expected_trials:
                raise RuntimeError(f"Episode row count mismatch in {path}")
            successes = sum(
                int(episode["success"]) for episode in capability["episodes"]
            )
            if int(capability.get("successes", -1)) != successes:
                raise RuntimeError(f"Success count mismatch in {path}")

    expected_keys = {(suite, task) for suite in SUITES for task in range(10)}
    if set(task_rows) != expected_keys:
        raise RuntimeError(
            f"Expected 40 suite-task cells for {architecture} seed {seed}, "
            f"got {len(task_rows)}"
        )
    for key, outcomes in task_rows.items():
        if len(outcomes) != 50:
            raise RuntimeError(f"Expected 50 trials for {key}, got {len(outcomes)}")

    task_rates = {
        f"{suite}:{task}": sum(outcomes) / len(outcomes)
        for (suite, task), outcomes in sorted(task_rows.items())
    }
    suite_rates = {
        suite: sum(task_rates[f"{suite}:{task}"] for task in range(10)) / 10
        for suite in SUITES
    }
    macro = sum(task_rates.values()) / len(task_rates)
    return {
        "macro_task_success": macro,
        "suite_macro_success": suite_rates,
        "task_success": task_rates,
        "tasks_at_least_0p50": sum(rate >= 0.50 for rate in task_rates.values()),
        "successes": sum(sum(outcomes) for outcomes in task_rows.values()),
        "trials": sum(len(outcomes) for outcomes in task_rows.values()),
        "source_files": source_files,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    by_architecture = {
        architecture: {
            str(seed): load_checkpoint_results(
                args.results_dir, architecture, seed
            )
            for seed in SEEDS
        }
        for architecture in ARCHITECTURES
    }

    aggregate = {}
    for architecture, checkpoint_rows in by_architecture.items():
        seed_macros = [
            row["macro_task_success"] for row in checkpoint_rows.values()
        ]
        aggregate[architecture] = {
            "seed_macro_task_success": seed_macros,
            "seed_mean_macro_task_success": mean(seed_macros),
            "seed_sample_std_macro_task_success": statistics.stdev(seed_macros),
            "suite_seed_mean_macro_success": {
                suite: mean(
                    [
                        row["suite_macro_success"][suite]
                        for row in checkpoint_rows.values()
                    ]
                )
                for suite in SUITES
            },
            "task_seed_mean_success": {
                f"{suite}:{task}": mean(
                    [
                        row["task_success"][f"{suite}:{task}"]
                        for row in checkpoint_rows.values()
                    ]
                )
                for suite in SUITES
                for task in range(10)
            },
        }
        aggregate[architecture]["tasks_seed_mean_at_least_0p50"] = sum(
            rate >= 0.50
            for rate in aggregate[architecture]["task_seed_mean_success"].values()
        )

    chi = aggregate["chi"]
    conventional = aggregate["conventional"]
    capability_gates = {
        "chi_seed_mean_macro_at_least_0p70": (
            chi["seed_mean_macro_task_success"] >= 0.70
        ),
        "every_chi_suite_at_least_0p50": all(
            rate >= 0.50 for rate in chi["suite_seed_mean_macro_success"].values()
        ),
        "at_least_32_of_40_tasks_at_0p50": (
            chi["tasks_seed_mean_at_least_0p50"] >= 32
        ),
        "every_chi_seed_at_least_0p60": all(
            row["macro_task_success"] >= 0.60
            for row in by_architecture["chi"].values()
        ),
    }
    capability_gates["primary_generalist_capability_pass"] = all(
        capability_gates.values()
    )
    conversion_gap = (
        chi["seed_mean_macro_task_success"]
        - conventional["seed_mean_macro_task_success"]
    )
    secondary_gates = {
        "chi_within_0p05_of_conventional": conversion_gap >= -0.05
    }

    output = {
        "scope": {
            "architectures": list(ARCHITECTURES),
            "seeds": list(SEEDS),
            "suites": list(SUITES),
            "tasks_per_suite": 10,
            "episodes_per_task": 50,
            "total_trials": 12000,
            "macro_averaging": "task first, then suite and checkpoint",
        },
        "by_architecture_and_seed": by_architecture,
        "aggregate": aggregate,
        "chi_minus_conventional_macro": conversion_gap,
        "capability_gates": capability_gates,
        "secondary_gates": secondary_gates,
        "all_preregistered_gates_pass": None,
        "specialist_retention_gate": {
            "status": "pending_locked_specialist_results",
            "reason": (
                "Requires locked, same-runtime seed-0 specialist results for all four "
                "suites and will be added without changing the generalist gates."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
