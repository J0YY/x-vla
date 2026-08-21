#!/usr/bin/env python3
"""Aggregate sharded Athena experiment JSON files into a paper-ready summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials == 0:
        return [float("nan"), float("nan")]
    probability = successes / trials
    denominator = 1 + z * z / trials
    center = (probability + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(
        probability * (1 - probability) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return [center - radius, center + radius]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def aggregate_capability(paths: list[Path]) -> dict[str, Any] | None:
    if not paths:
        return None
    episodes = []
    profiles = []
    architecture = None
    for path in paths:
        result = load_json(path)
        architecture = result["architecture"]
        profiles.append(result["profile"])
        episodes.extend(result["capability"]["episodes"])
    by_task: dict[int, list[bool]] = {}
    for episode in episodes:
        by_task.setdefault(int(episode["task_index"]), []).append(bool(episode["success"]))
    successes = sum(int(episode["success"]) for episode in episodes)
    trials = len(episodes)
    tasks_covered = sorted(by_task)
    complete_ten_task_suite = tasks_covered == list(range(10)) and all(
        len(by_task[task]) == 50 for task in tasks_covered
    )
    return {
        "architecture": architecture,
        "shards": [str(path) for path in paths],
        "successes": successes,
        "trials": trials,
        "success_rate": successes / trials,
        "success_rate_wilson_95pct_ci": wilson_interval(successes, trials),
        "tasks_covered": tasks_covered,
        "complete_ten_task_suite": complete_ten_task_suite,
        "per_task": {
            str(task): {
                "successes": sum(values),
                "trials": len(values),
                "success_rate": mean(values),
            }
            for task, values in sorted(by_task.items())
        },
        "parameters": profiles[0]["parameters"],
        "latency_ms_batch1_median_across_shards": median(
            float(profile["latency_ms_batch1"]) for profile in profiles
        ),
        "latency_ms_batch1_by_shard": [
            float(profile["latency_ms_batch1"]) for profile in profiles
        ],
        "gpus": sorted({str(profile["gpu"]) for profile in profiles}),
    }


def same_hardware_capability_paths(
    results_dir: Path, architecture: str, hardware: str
) -> list[Path]:
    """Prefer tagged reruns, then reuse nonoverlapping base shards from the same GPU."""
    selected_by_tasks: dict[tuple[int, ...], Path] = {}
    tagged_architecture = "conv" if architecture == "conventional" else architecture
    candidates = [
        *sorted(
            results_dir.glob(
                f"matched_{hardware}_{tagged_architecture}_s0_t*.json"
            )
        ),
        *sorted(results_dir.glob(f"{architecture}_s0_t*.json")),
    ]
    hardware_token = hardware.lower().replace(" ", "")
    for path in candidates:
        result = load_json(path)
        gpu = str(result["profile"]["gpu"]).lower().replace(" ", "")
        if hardware_token not in gpu:
            continue
        task_key = tuple(
            sorted(
                {
                    int(episode["task_index"])
                    for episode in result["capability"]["episodes"]
                }
            )
        )
        selected_by_tasks.setdefault(task_key, path)
    return [selected_by_tasks[key] for key in sorted(selected_by_tasks)]


def aggregate_causal(paths: list[Path]) -> dict[str, Any] | None:
    if not paths:
        return None
    by_checkpoint = {}
    all_rows = []
    for path in paths:
        result = load_json(path)
        causal = result["causal"]
        label = Path(result["checkpoint"]).stem
        rows = causal["rows"]
        all_rows.extend(rows)
        by_checkpoint[label] = {
            "n_trials": causal["n_trials"],
            "mean_paired_preference_shift_m": causal["mean_paired_preference_shift_m"],
            "mean_paired_preference_shift_95pct_bootstrap_ci_m": causal[
                "mean_paired_preference_shift_95pct_bootstrap_ci_m"
            ],
            "fraction_shift_toward_counterfactual_named_object": causal[
                "fraction_shift_toward_counterfactual_named_object"
            ],
            "fraction_counterfactual_ended_closer_to_named_object": causal[
                "fraction_counterfactual_ended_closer_to_named_object"
            ],
            "true_instruction_bddl_success": causal["true_instruction_bddl_success"],
        }
    shifts = [float(row["paired_preference_shift"]) for row in all_rows]
    positive = [bool(row["shift_toward_counterfactual_named_object"]) for row in all_rows]
    return {
        "files": [str(path) for path in paths],
        "by_checkpoint": by_checkpoint,
        "pooled_trials": len(all_rows),
        "pooled_mean_paired_preference_shift_m": mean(shifts) if shifts else None,
        "pooled_fraction_shift_toward_counterfactual_named_object": (
            mean(positive) if positive else None
        ),
        "pooling_note": (
            "The pooled estimate is descriptive. Checkpoint-level estimates are the primary "
            "replication result because trials within a checkpoint share learned weights."
        ),
    }


def aggregate_exact(paths: list[Path]) -> dict[str, Any] | None:
    if not paths:
        return None
    runs = []
    for path in paths:
        result = load_json(path)["exact_attention"]
        runs.append(
            {
                "file": str(path),
                "modules_audited": result["modules_audited"],
                "heads_audited": result["heads_audited"],
                "max_abs_error": result["max_abs_error"],
                "max_relative_l2_error": result["max_relative_l2_error"],
                "all_modules_below_1e_minus_5": result["all_modules_below_1e_minus_5"],
                "all_modules_below_2e_minus_5": result["all_modules_below_2e_minus_5"],
            }
        )
    return {
        "runs": runs,
        "checkpoints": len(runs),
        "runs_below_absolute_2e_minus_5": sum(
            bool(run["all_modules_below_2e_minus_5"]) for run in runs
        ),
        "all_runs_relative_l2_below_3e_minus_7": all(
            float(run["max_relative_l2_error"]) < 3e-7 for run in runs
        ),
        "worst_max_abs_error": max(float(run["max_abs_error"]) for run in runs),
        "worst_max_relative_l2_error": max(
            float(run["max_relative_l2_error"]) for run in runs
        ),
        "heads_audited_total": sum(int(run["heads_audited"]) for run in runs),
        "threshold_note": (
            "The absolute 2e-5 indicator is retained per run. Relative error is the primary "
            "scale-aware reconstruction diagnostic, and no failed absolute indicator is hidden."
        ),
    }


def compare_capability(
    chi: dict[str, Any] | None, conventional: dict[str, Any] | None
) -> dict[str, Any] | None:
    if chi is None or conventional is None:
        return None
    chi_gpus = set(chi["gpus"])
    conventional_gpus = set(conventional["gpus"])
    same_single_gpu_class = len(chi_gpus) == 1 and chi_gpus == conventional_gpus
    same_tasks = chi["tasks_covered"] == conventional["tasks_covered"]
    hardware_controlled_complete = (
        same_single_gpu_class
        and same_tasks
        and chi["complete_ten_task_suite"]
        and conventional["complete_ten_task_suite"]
    )
    return {
        "chi_minus_conventional_success_rate": (
            chi["success_rate"] - conventional["success_rate"]
        ),
        "chi_to_conventional_parameter_ratio": (
            chi["parameters"] / conventional["parameters"]
        ),
        "chi_to_conventional_latency_ratio": (
            chi["latency_ms_batch1_median_across_shards"]
            / conventional["latency_ms_batch1_median_across_shards"]
        ),
        "same_single_gpu_class": same_single_gpu_class,
        "same_tasks": same_tasks,
        "hardware_controlled_complete": hardware_controlled_complete,
        "gpu_classes": sorted(chi_gpus | conventional_gpus),
        "protocol": (
            "Same skeleton, seed, cache, dimensions, action head, canonical initial states, "
            "task set, trial count, step cap, and Athena runner. The block family is the "
            "controlled difference. hardware_controlled_complete must be true before "
            "interpreting the comparison as a complete hardware-controlled estimate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    conventional_by_seed = {
        str(seed): aggregate_capability(
            sorted(args.results_dir.glob(f"conventional_s{seed}_t*.json"))
        )
        for seed in (0, 1, 2)
    }
    result = {
        "conventional_capability": conventional_by_seed["0"],
        "conventional_capability_by_seed": conventional_by_seed,
        "chi_s0_capability": aggregate_capability(
            sorted(args.results_dir.glob("chi_s0_t*.json"))
        ),
        "causal_intervention": aggregate_causal(
            sorted(args.results_dir.glob("causal_s*.json"))
        ),
        "exact_attention": aggregate_exact(
            sorted(args.results_dir.glob("exact_attention_s*.json"))
        ),
        "matched_training_profile": (
            load_json(args.results_dir / "training_profile_pair.json")
            if (args.results_dir / "training_profile_pair.json").exists()
            else None
        ),
        "matched_inference_profile": (
            load_json(args.results_dir / "inference_profile_pair.json")
            if (args.results_dir / "inference_profile_pair.json").exists()
            else None
        ),
        "matched_inference_profile_a6000": (
            load_json(args.results_dir / "inference_profile_pair_a6000.json")
            if (args.results_dir / "inference_profile_pair_a6000.json").exists()
            else None
        ),
    }
    result["same_hardware_capability"] = {}
    for hardware in ("a30", "a6000", "a40"):
        chi_hardware = aggregate_capability(
            same_hardware_capability_paths(args.results_dir, "chi", hardware)
        )
        conventional_hardware = aggregate_capability(
            same_hardware_capability_paths(
                results_dir=args.results_dir,
                architecture="conventional",
                hardware=hardware,
            )
        )
        if chi_hardware is not None or conventional_hardware is not None:
            result["same_hardware_capability"][hardware] = {
                "chi": chi_hardware,
                "conventional": conventional_hardware,
                "comparison": compare_capability(chi_hardware, conventional_hardware),
            }
    conventional = result["conventional_capability"]
    chi = result["chi_s0_capability"]
    result["matched_architecture_comparison"] = compare_capability(chi, conventional)
    complete_conventional = [
        item for item in conventional_by_seed.values() if item is not None and item["trials"] == 500
    ]
    if complete_conventional:
        rates = [float(item["success_rate"]) for item in complete_conventional]
        result["conventional_multiseed"] = {
            "complete_seeds": len(rates),
            "success_rates": rates,
            "mean_success_rate": mean(rates),
            "population_std_success_rate": pstdev(rates),
        }
    training_paths = sorted(args.results_dir.glob("train_conventional_s*.json"))
    if training_paths:
        training_runs = [load_json(path) for path in training_paths]
        result["conventional_training_runs"] = {
            "runs": training_runs,
            "mean_steps_per_s": mean(float(run["steps_per_s"]) for run in training_runs),
            "mean_peak_allocated_gb": mean(
                float(run["peak_allocated_gb"]) for run in training_runs
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
