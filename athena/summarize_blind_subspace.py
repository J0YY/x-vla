#!/usr/bin/env python3
"""Summarize the frozen blind rank-96 and activation-energy controls."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


SEEDS = (0, 1, 2)
TASKS = (8, 9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/visual_blind_rank96_summary.json"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def load_visual(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)["visual_subspace"]


def episode_map(result: dict[str, Any]) -> dict[tuple[int, int], bool]:
    return {
        (int(row["task_index"]), int(row["episode"])): bool(row["success"])
        for row in result["episodes"]
    }


def retention(records: list[tuple[bool, bool, bool]]) -> dict[str, float | int]:
    full = sum(int(row[0]) for row in records)
    jacobian = sum(int(row[0] and row[1]) for row in records)
    activation = sum(int(row[0] and row[2]) for row in records)
    return {
        "full_successes": full,
        "jacobian_retained": jacobian,
        "activation_retained": activation,
        "jacobian_retention": jacobian / full if full else 0.0,
        "activation_retention": activation / full if full else 0.0,
        "jacobian_minus_activation": (jacobian - activation) / full if full else 0.0,
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap(
    strata: dict[tuple[int, int], list[tuple[bool, bool, bool]]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        replicate = []
        for sampled_seed in rng.choices(SEEDS, k=len(SEEDS)):
            for sampled_task in rng.choices(TASKS, k=len(TASKS)):
                rows = strata[(sampled_seed, sampled_task)]
                replicate.extend(rng.choices(rows, k=len(rows)))
        estimate = retention(replicate)
        if estimate["full_successes"]:
            differences.append(float(estimate["jacobian_minus_activation"]))
    return {
        "method": "paired checkpoint-task-episode cluster bootstrap",
        "samples_requested": samples,
        "samples_valid": len(differences),
        "seed": seed,
        "jacobian_minus_activation_ci95": [
            percentile(differences, 0.025),
            percentile(differences, 0.975),
        ],
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")

    strata: dict[tuple[int, int], list[tuple[bool, bool, bool]]] = {}
    activation_mse = defaultdict(list)
    action_mse = defaultdict(lambda: defaultdict(list))

    for seed in SEEDS:
        for task in TASKS:
            path = (
                args.results_dir
                / f"visual_taskmap_activation_s{seed}_r96_t{task}.json"
            )
            visual = load_visual(path)
            if visual.get("gram_task_index_space") != "official LIBERO task indices":
                raise RuntimeError(f"Uncorrected Gram task indexing in {path}")
            if visual.get("gram_task_range") != [0, 4]:
                raise RuntimeError(f"Unexpected Gram split in {path}")
            if visual.get("cache_task_metadata", {}).get(
                "ordering_matches_official"
            ) is not False:
                raise RuntimeError(f"Missing verified dataset task permutation in {path}")
            conditions = visual["by_condition"]
            full = episode_map(conditions["full"])
            jacobian = episode_map(conditions["causal_topk"])
            activation = episode_map(conditions["activation_energy_topk"])
            if full.keys() != jacobian.keys() or full.keys() != activation.keys():
                raise RuntimeError(f"Episode mismatch in {path}")
            strata[(seed, task)] = [
                (full[key], jacobian[key], activation[key]) for key in sorted(full)
            ]
            for condition, value in visual["offline_activation_reconstruction_mse"].items():
                if condition in {"causal_topk", "activation_energy_topk"}:
                    activation_mse[condition].append(float(value))
            for condition in ("causal_topk", "activation_energy_topk"):
                for group, value in visual["offline_mse_to_full"][condition].items():
                    action_mse[condition][group].append(float(value))

    all_records = [row for rows in strata.values() for row in rows]
    pooled = retention(all_records)
    by_checkpoint = {
        str(seed): retention(
            [row for task in TASKS for row in strata[(seed, task)]]
        )
        for seed in SEEDS
    }
    bootstrap = paired_bootstrap(
        strata,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )

    random_gate = {}
    random_signs = []
    for seed in SEEDS:
        seed_rows = []
        for task in TASKS:
            path = (
                args.results_dir
                / f"visual_taskmap_vit_s{seed}_r96_blind_t{task}.json"
            )
            visual = load_visual(path)
            if visual.get("gram_task_index_space") != "official LIBERO task indices":
                raise RuntimeError(f"Uncorrected Gram task indexing in {path}")
            if visual.get("gram_task_range") != [0, 4]:
                raise RuntimeError(f"Unexpected Gram split in {path}")
            conditions = visual["by_condition"]
            full = episode_map(conditions["full"])
            jacobian = episode_map(conditions["causal_topk"])
            random_conditions = sorted(
                name for name in conditions if name.startswith("random_topk")
            )
            for control_index, condition in enumerate(random_conditions):
                random_map = episode_map(conditions[condition])
                for key in sorted(full):
                    seed_rows.append(
                        (control_index, full[key], jacobian[key], random_map[key])
                    )
        seed_summary = {}
        for control_index in sorted({row[0] for row in seed_rows}):
            rows = [row for row in seed_rows if row[0] == control_index]
            full_successes = sum(int(row[1]) for row in rows)
            jacobian_retained = sum(int(row[1] and row[2]) for row in rows)
            random_retained = sum(int(row[1] and row[3]) for row in rows)
            difference = (
                (jacobian_retained - random_retained) / full_successes
                if full_successes
                else 0.0
            )
            seed_summary[str(control_index)] = {
                "full_successes": full_successes,
                "jacobian_retained": jacobian_retained,
                "random_retained": random_retained,
                "jacobian_minus_random": difference,
            }
            random_signs.append(difference > 0)
        random_gate[str(seed)] = seed_summary

    mean_activation_mse = {
        condition: sum(values) / len(values)
        for condition, values in activation_mse.items()
    }
    mean_action_mse = {
        condition: {
            group: sum(values) / len(values)
            for group, values in groups.items()
        }
        for condition, groups in action_mse.items()
    }

    gates = {
        "jacobian_retention_at_least_0p90": pooled["jacobian_retention"] >= 0.90,
        "jacobian_minus_activation_at_least_0p10": (
            pooled["jacobian_minus_activation"] >= 0.10
        ),
        "jacobian_beats_activation_every_checkpoint": all(
            row["jacobian_minus_activation"] > 0 for row in by_checkpoint.values()
        ),
        "bootstrap_lower_bound_above_zero": (
            bootstrap["jacobian_minus_activation_ci95"][0] > 0
        ),
        "activation_mse_no_greater_than_jacobian": (
            mean_activation_mse["activation_energy_topk"]
            <= mean_activation_mse["causal_topk"]
        ),
        "jacobian_beats_every_random_control": all(random_signs),
    }
    gates["action_specificity_pass"] = all(
        value
        for key, value in gates.items()
        if key != "jacobian_beats_every_random_control"
    )
    gates["blind_selectivity_pass"] = (
        pooled["jacobian_retention"] >= 0.85
        and gates["jacobian_beats_every_random_control"]
    )

    output = {
        "scope": {
            "seeds": list(SEEDS),
            "tasks": list(TASKS),
            "rank": 96,
            "conditions_same_allocation": True,
        },
        "pooled": pooled,
        "by_checkpoint": by_checkpoint,
        "bootstrap": bootstrap,
        "mean_offline_activation_reconstruction_mse": mean_activation_mse,
        "mean_offline_action_mse": mean_action_mse,
        "random_controls_by_checkpoint": random_gate,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
