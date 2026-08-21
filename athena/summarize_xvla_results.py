#!/usr/bin/env python3
"""Aggregate sharded Athena experiment JSON files into a paper-ready summary."""

from __future__ import annotations

import argparse
import json
import math
import re
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


def exact_paired_sign_pvalue(first_only: int, second_only: int) -> float:
    """Two-sided exact sign test over discordant paired binary outcomes."""
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(0, min(first_only, second_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


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


def canonical_shards(results_dir: Path, architecture: str, seed: int) -> list[Path]:
    """Return only canonical range shards, excluding targeted task diagnostics."""
    pattern = re.compile(rf"^{re.escape(architecture)}_s{seed}_t\d+_\d+\.json$")
    return [
        path
        for path in sorted(results_dir.glob(f"{architecture}_s{seed}_t*.json"))
        if pattern.match(path.name)
    ]


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


def aggregate_visual_subspace(paths: list[Path]) -> dict[str, Any] | None:
    if not paths:
        return None
    by_checkpoint = {}
    pooled: dict[str, dict[str, int]] = {}
    for path in paths:
        result = load_json(path)
        subspace = result["visual_subspace"]
        checkpoint = Path(result["checkpoint"]).stem
        by_checkpoint[checkpoint] = {
            "file": str(path),
            "rank": subspace["rank"],
            "ambient_dimension": subspace["ambient_dimension"],
            "gram_samples": subspace["gram_samples"],
            "offline_eval_samples": subspace.get("offline_eval_samples"),
            "offline_eval_disjoint_from_gram": subspace.get(
                "offline_eval_disjoint_from_gram", False
            ),
            "top_rank_spectral_mass": subspace["top_rank_spectral_mass"],
            "offline_random_to_causal_mse_ratio": subspace[
                "offline_random_to_causal_mse_ratio"
            ],
            "overall_by_condition": subspace["overall_by_condition"],
        }
        episodes_by_condition = {
            condition: {
                (int(row["task_index"]), int(row["episode"])): bool(row["success"])
                for row in capability["episodes"]
            }
            for condition, capability in subspace["by_condition"].items()
        }
        causal = episodes_by_condition.get("causal_topk", {})
        random_control = episodes_by_condition.get("random_topk", {})
        paired_keys = sorted(set(causal) & set(random_control))
        causal_only = sum(causal[key] and not random_control[key] for key in paired_keys)
        random_only = sum(random_control[key] and not causal[key] for key in paired_keys)
        by_checkpoint[checkpoint]["paired_causal_vs_random"] = {
            "paired_trials": len(paired_keys),
            "causal_only_successes": causal_only,
            "random_only_successes": random_only,
            "success_rate_difference": (
                subspace["overall_by_condition"]["causal_topk"]
                - subspace["overall_by_condition"]["random_topk"]
            ),
            "exact_two_sided_sign_pvalue": exact_paired_sign_pvalue(
                causal_only, random_only
            ),
        }
        for condition, capability in subspace["by_condition"].items():
            totals = pooled.setdefault(condition, {"successes": 0, "trials": 0})
            totals["successes"] += int(capability["successes"])
            totals["trials"] += int(capability["trials"])
    return {
        "checkpoints": len(by_checkpoint),
        "by_checkpoint": by_checkpoint,
        "pooled_by_condition": {
            condition: {
                **totals,
                "success_rate": totals["successes"] / totals["trials"],
                "success_rate_wilson_95pct_ci": wilson_interval(
                    totals["successes"], totals["trials"]
                ),
            }
            for condition, totals in pooled.items()
        },
        "pooling_note": (
            "Pooled condition rates are descriptive because trials within one checkpoint share "
            "learned weights. Checkpoint-level replication is the primary evidence."
        ),
    }


def aggregate_offline_diagnostics(paths: list[Path]) -> dict[str, Any] | None:
    if not paths:
        return None
    runs = []
    for path in paths:
        result = load_json(path)
        diagnostic = result["offline_diagnostic"]
        runs.append(
            {
                "file": str(path),
                "architecture": result["architecture"],
                "checkpoint": Path(result["checkpoint"]).stem,
                "seed": result["seed"],
                "sample_count": diagnostic["sample_count"],
                "overall": diagnostic["overall"],
                "per_task": diagnostic["per_task"],
            }
        )
    return {"runs": runs, "scope": load_json(paths[0])["offline_diagnostic"]["scope"]}


def aggregate_cross_suite(results_dir: Path) -> dict[str, Any] | None:
    suites = ("libero_spatial", "libero_goal", "libero_10")
    result = {}
    for suite in suites:
        by_architecture = {}
        for architecture, filename_architecture in (
            ("chi", "chi"),
            ("conventional", "conventional"),
        ):
            path = results_dir / f"{suite}_{filename_architecture}_s0.json"
            if path.exists():
                by_architecture[architecture] = aggregate_capability([path])
        if by_architecture:
            result[suite] = by_architecture
    return result or None


def aggregate_indomain_multisuite(results_dir: Path) -> dict[str, Any] | None:
    result = {}
    for suite in ("libero_spatial", "libero_goal", "libero_10"):
        by_architecture = {
            architecture: aggregate_capability(
                canonical_shards(
                    results_dir, f"indomain_{suite}_{architecture}", seed=0
                )
            )
            for architecture in ("chi", "conventional")
        }
        if any(item is not None for item in by_architecture.values()):
            result[suite] = {
                **by_architecture,
                "comparison": compare_capability(
                    by_architecture["chi"], by_architecture["conventional"]
                ),
            }
    return result or None


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
            canonical_shards(args.results_dir, "conventional", seed)
        )
        for seed in (0, 1, 2)
    }
    chi_by_seed = {
        str(seed): aggregate_capability(canonical_shards(args.results_dir, "chi", seed))
        for seed in (0, 1, 2)
    }
    result = {
        "conventional_capability": conventional_by_seed["0"],
        "conventional_capability_by_seed": conventional_by_seed,
        "chi_s0_capability": chi_by_seed["0"],
        "chi_capability_by_seed": chi_by_seed,
        "causal_intervention": aggregate_causal(
            sorted(args.results_dir.glob("causal_s*.json"))
        ),
        "exact_attention": aggregate_exact(
            sorted(args.results_dir.glob("exact_attention_s*.json"))
        ),
        "visual_subspace": aggregate_visual_subspace(
            [
                path
                for path in sorted(args.results_dir.glob("visual_subspace_s*.json"))
                if re.match(r"^visual_subspace_s\d+\.json$", path.name)
            ]
        ),
        "offline_checkpoint_diagnostics": aggregate_offline_diagnostics(
            sorted(args.results_dir.glob("offline_*.json"))
        ),
        "zero_shot_cross_suite": aggregate_cross_suite(args.results_dir),
        "indomain_multisuite": aggregate_indomain_multisuite(args.results_dir),
        "coefficient_surgery_discovery": (
            load_json(args.results_dir / "surgery_discovery_summary.json")
            if (args.results_dir / "surgery_discovery_summary.json").exists()
            else None
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
    complete_chi = [
        item for item in chi_by_seed.values() if item is not None and item["trials"] == 500
    ]
    if complete_chi:
        rates = [float(item["success_rate"]) for item in complete_chi]
        result["chi_multiseed"] = {
            "complete_seeds": len(rates),
            "success_rates": rates,
            "mean_success_rate": mean(rates),
            "population_std_success_rate": pstdev(rates),
        }
    result["matched_architecture_comparison_by_seed"] = {
        str(seed): compare_capability(chi_by_seed[str(seed)], conventional_by_seed[str(seed)])
        for seed in (0, 1, 2)
    }
    native_chi_by_seed = {
        str(seed): aggregate_capability(
            canonical_shards(args.results_dir, "native_chi", seed)
        )
        for seed in (0, 1, 2)
    }
    native_conventional_s0 = aggregate_capability(
        canonical_shards(args.results_dir, "native_conventional", 0)
    )
    if native_conventional_s0 is not None or any(
        item is not None for item in native_chi_by_seed.values()
    ):
        result["athena_native_provenance"] = {
            "conventional_s0": native_conventional_s0,
            "chi_by_seed": native_chi_by_seed,
            "seed0_comparison": compare_capability(
                native_chi_by_seed["0"], native_conventional_s0
            ),
            "scope": (
                "All checkpoints in this matrix are trained by the same Athena-native trainer. "
                "This separates training provenance from seed sensitivity in the legacy matrix."
            ),
        }
    task3_by_seed = {}
    for seed in (0, 1, 2):
        chi_targeted_path = args.results_dir / f"chi_s{seed}_task3.json"
        chi_targeted = (
            aggregate_capability([chi_targeted_path])
            if chi_targeted_path.exists()
            else None
        )
        chi_source = chi_targeted or chi_by_seed[str(seed)]
        conventional_source = conventional_by_seed[str(seed)]
        if chi_source is None and conventional_source is None:
            continue

        def task3_row(capability: dict[str, Any] | None) -> dict[str, Any] | None:
            if capability is None or "3" not in capability["per_task"]:
                return None
            row = capability["per_task"]["3"]
            return {
                **row,
                "success_rate_wilson_95pct_ci": wilson_interval(
                    int(row["successes"]), int(row["trials"])
                ),
            }

        chi_row = task3_row(chi_source)
        conventional_row = task3_row(conventional_source)
        task3_by_seed[str(seed)] = {
            "chi": chi_row,
            "conventional": conventional_row,
            "chi_minus_conventional_success_rate": (
                chi_row["success_rate"] - conventional_row["success_rate"]
                if chi_row is not None and conventional_row is not None
                else None
            ),
            "chi_source": (
                str(chi_targeted_path)
                if chi_targeted is not None
                else "full canonical shards"
            ),
        }
    result["targeted_task3_by_seed"] = task3_by_seed
    partial_paths = {
        "chi_rms": args.results_dir / "chi_rms_s0_task3.json",
        "conventional_rational": (
            args.results_dir / "conventional_rational_s0_task3.json"
        ),
    }
    partial_results = {
        architecture: aggregate_capability([path])
        for architecture, path in partial_paths.items()
        if path.exists()
    }
    if partial_results:
        result["normalization_partial_conversion_task3"] = partial_results
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
