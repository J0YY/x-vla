#!/usr/bin/env python3
"""Summarize the frozen blind rank-96 and activation-energy controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


SEEDS = (0, 1, 2)
TASKS = (8, 9)
CHECKPOINTS = {
    0: (
        "artifacts/ckpt_linear_rat_vit_s0_v2.pt",
        "96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9",
    ),
    1: (
        "artifacts/ckpt_linear_rat_vit_s1.pt",
        "cc0780b989165a80a449c03cbb3574e2f56b2e5747a64cf32d1fde418df8ec3c",
    ),
    2: (
        "artifacts/ckpt_linear_rat_vit_s2.pt",
        "413a770071bd8f16b924c5604f7c7a58567d2eb6c40c66aab202bcf046fec910",
    ),
}
CACHE_PATH = "artifacts/libero_frames_100000_64.pkl"
CACHE_SHA256 = "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662"
METADATA = {
    "repository": "lerobot/libero_object_image",
    "revision": "e1e080d7df1d0a359dff5c86c222e047549f447f",
    "metadata_file": "meta/tasks.parquet",
    "metadata_sha256": "34caee9641ae50bb4e077de306a7d0031753757882da8b1f117e7ea36a486b42",
    "dataset_to_official_task": {
        "0": 9,
        "1": 4,
        "2": 1,
        "3": 3,
        "4": 0,
        "5": 7,
        "6": 2,
        "7": 6,
        "8": 5,
        "9": 8,
    },
    "ordering_matches_official": False,
    "language_set_matches_official": True,
}
EVALUATION_PROTOCOL = {
    "res": 64,
    "horizon": 8,
    "num_steps_wait": 10,
    "exec_h": 8,
    "eps_per_task": 50,
    "max_steps": 400,
}
TASK_PROTOCOL_HASHES = {
    8: {
        "bddl_sha256": "674ceabc400b7b16d46e8eaf678709d4a633235c7e0a80233775044fe38b19b0",
        "init_states_sha256": "54593bb836dc12d321d9e7971e54034bbe01c02929cf0c09cb93b75038bedf5a",
    },
    9: {
        "bddl_sha256": "6298533e7bcfb83e40779a77fda39216e8cd53f22bf1af6388585dad0abb0b50",
        "init_states_sha256": "c198880f92818df55d0a8e53ab739bbfd36acfab236a276c7ac72a27a27baf0d",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/visual_taskmap_blind_rank96_summary.json"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, found {actual!r}")


def load_result(
    path: Path,
    *,
    seed: int,
    task: int,
    activation_control: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open() as handle:
        result = json.load(handle)

    checkpoint_path, _ = CHECKPOINTS[seed]
    expected_outer = {
        "mode": "visual_subspace",
        "architecture": "chi",
        "vision_encoder": "vit",
        "suite": "libero_object",
        "training_suite": "libero_object",
        "checkpoint": checkpoint_path,
        "cache": CACHE_PATH,
        "seed": seed,
        "matmul_precision": "high",
        "prediction_finite": True,
        "evaluation_protocol": EVALUATION_PROTOCOL,
    }
    for key, expected in expected_outer.items():
        require_equal(result.get(key), expected, f"{path}:{key}")
    require_equal(result.get("cache_task_metadata"), METADATA, f"{path}:metadata")

    visual = result.get("visual_subspace")
    if not isinstance(visual, dict):
        raise RuntimeError(f"Missing visual_subspace object in {path}")
    expected_visual = {
        "rank": 96,
        "ambient_dimension": 384,
        "gram_samples": 1024,
        "gram_task_range": [0, 4],
        "gram_task_index_space": "official LIBERO task indices",
        "cache_task_metadata": METADATA,
        "offline_eval_samples": 1024,
        "offline_eval_disjoint_from_gram": True,
        "gram_action_group": "all_balanced",
        "gram_probes": 4,
        "random_controls": 1 if activation_control else 3,
        "offline_only": False,
        "activation_energy_control": activation_control,
        "gradient_rows": 1_835_008,
        "activation_rows": 65_536 if activation_control else 0,
    }
    for key, expected in expected_visual.items():
        require_equal(visual.get(key), expected, f"{path}:visual_subspace.{key}")

    expected_conditions = (
        {"full", "causal_topk", "activation_energy_topk"}
        if activation_control
        else {"full", "causal_topk", "random_topk", "random_topk_1", "random_topk_2"}
    )
    conditions = visual.get("by_condition")
    if not isinstance(conditions, dict):
        raise RuntimeError(f"Missing by_condition object in {path}")
    require_equal(set(conditions), expected_conditions, f"{path}:condition names")
    require_equal(
        set(visual.get("subspace_rollout_conditions", [])),
        expected_conditions,
        f"{path}:declared rollout conditions",
    )
    for condition, rows in conditions.items():
        validate_condition(path, condition, rows, task)
    expected_overall = {
        condition: float(rows["successes"]) / float(rows["trials"])
        for condition, rows in conditions.items()
    }
    require_equal(
        visual.get("overall_by_condition"),
        expected_overall,
        f"{path}:overall_by_condition",
    )

    if activation_control:
        activation_reconstruction = visual.get(
            "offline_activation_reconstruction_mse"
        )
        action_mse = visual.get("offline_mse_to_full")
        if not isinstance(activation_reconstruction, dict) or not isinstance(
            action_mse, dict
        ):
            raise RuntimeError(f"{path}: missing activation-control offline metrics")
        for condition in ("causal_topk", "activation_energy_topk"):
            value = activation_reconstruction.get(condition)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise RuntimeError(
                    f"{path}: invalid offline activation MSE for {condition}"
                )
            groups = action_mse.get(condition)
            if not isinstance(groups, dict) or set(groups) != {
                "translation",
                "rotation",
                "gripper",
            }:
                raise RuntimeError(f"{path}: incomplete action MSE for {condition}")
            if any(
                not isinstance(group_value, (int, float))
                or not math.isfinite(group_value)
                or group_value < 0
                for group_value in groups.values()
            ):
                raise RuntimeError(f"{path}: invalid action MSE for {condition}")

    return result, visual


def validate_condition(
    path: Path,
    condition: str,
    result: dict[str, Any],
    task: int,
) -> None:
    require_equal(result.get("task_indices"), [task], f"{path}:{condition}:tasks")
    require_equal(result.get("trials"), 50, f"{path}:{condition}:trials")
    require_equal(result.get("eps_per_task"), 50, f"{path}:{condition}:eps")
    require_equal(result.get("max_steps"), 400, f"{path}:{condition}:max_steps")
    require_equal(result.get("num_steps_wait"), 10, f"{path}:{condition}:wait")
    require_equal(result.get("exec_h"), 8, f"{path}:{condition}:exec_h")
    require_equal(
        result.get("canonical_init_states"), True, f"{path}:{condition}:canonical"
    )
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 50:
        raise RuntimeError(f"{path}:{condition}: expected exactly 50 episode rows")
    keys = [(int(row["task_index"]), int(row["episode"])) for row in episodes]
    require_equal(keys, [(task, episode) for episode in range(50)], f"{path}:{condition}:episode matrix")
    if any(not isinstance(row.get("success"), bool) for row in episodes):
        raise RuntimeError(f"{path}:{condition}: success flags must be Boolean")
    successes = sum(int(row["success"]) for row in episodes)
    require_equal(result.get("successes"), successes, f"{path}:{condition}:successes")
    require_equal(result.get("overall"), successes / 50, f"{path}:{condition}:overall")
    protocol = result.get("task_protocol")
    if not isinstance(protocol, dict) or set(protocol) != {str(task)}:
        raise RuntimeError(f"{path}:{condition}: invalid task protocol keys")
    task_protocol = protocol[str(task)]
    for key, expected in TASK_PROTOCOL_HASHES[task].items():
        require_equal(
            task_protocol.get(key), expected, f"{path}:{condition}:task protocol {key}"
        )
    require_equal(
        task_protocol.get("init_state_count"),
        50,
        f"{path}:{condition}:task protocol init_state_count",
    )


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


def blind_retention(records: list[tuple[bool, bool]]) -> dict[str, float | int]:
    full = sum(int(row[0]) for row in records)
    jacobian = sum(int(row[0] and row[1]) for row in records)
    return {
        "full_successes": full,
        "jacobian_retained": jacobian,
        "jacobian_retention": jacobian / full if full else 0.0,
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
    summary_source_sha256 = file_sha256(Path(__file__).resolve())

    strata: dict[tuple[int, int], list[tuple[bool, bool, bool]]] = {}
    activation_mse = defaultdict(list)
    action_mse = defaultdict(lambda: defaultdict(list))
    input_identities: dict[str, dict[str, Any]] = {}

    for seed, (checkpoint_path, expected_sha256) in CHECKPOINTS.items():
        actual_sha256 = file_sha256(Path(checkpoint_path))
        require_equal(actual_sha256, expected_sha256, f"checkpoint SHA for seed {seed}")
    require_equal(file_sha256(Path(CACHE_PATH)), CACHE_SHA256, "cache SHA")

    for seed in SEEDS:
        for task in TASKS:
            path = (
                args.results_dir
                / f"visual_taskmap_activation_s{seed}_r96_t{task}.json"
            )
            input_sha256 = file_sha256(path)
            _, visual = load_result(
                path, seed=seed, task=task, activation_control=True
            )
            require_equal(file_sha256(path), input_sha256, f"{path}: changed while loading")
            input_identities[str(path)] = {"sha256": input_sha256}
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
    blind_strata: dict[tuple[int, int], list[tuple[bool, bool]]] = {}
    for seed in SEEDS:
        seed_rows = []
        for task in TASKS:
            path = (
                args.results_dir
                / f"visual_taskmap_vit_s{seed}_r96_blind_t{task}.json"
            )
            input_sha256 = file_sha256(path)
            _, visual = load_result(
                path, seed=seed, task=task, activation_control=False
            )
            require_equal(file_sha256(path), input_sha256, f"{path}: changed while loading")
            input_identities[str(path)] = {"sha256": input_sha256}
            conditions = visual["by_condition"]
            full = episode_map(conditions["full"])
            jacobian = episode_map(conditions["causal_topk"])
            blind_strata[(seed, task)] = [
                (full[key], jacobian[key]) for key in sorted(full)
            ]
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

    blind_pooled = blind_retention(
        [row for rows in blind_strata.values() for row in rows]
    )
    blind_by_checkpoint = {
        str(seed): blind_retention(
            [row for task in TASKS for row in blind_strata[(seed, task)]]
        )
        for seed in SEEDS
    }

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
        blind_pooled["jacobian_retention"] >= 0.85
        and gates["jacobian_beats_every_random_control"]
    )

    output = {
        "schema": "xvla-corrected-blind-subspace-summary-v2",
        "scope": {
            "seeds": list(SEEDS),
            "tasks": list(TASKS),
            "rank": 96,
            "conditions_same_allocation": True,
            "checkpoint_sha256": {
                str(seed): expected_sha256
                for seed, (_, expected_sha256) in CHECKPOINTS.items()
            },
            "cache_sha256": CACHE_SHA256,
            "cache_metadata": METADATA,
            "evaluation_protocol": EVALUATION_PROTOCOL,
        },
        "input_results": input_identities,
        "summary_source_sha256": summary_source_sha256,
        "blind_pooled": blind_pooled,
        "blind_by_checkpoint": blind_by_checkpoint,
        "pooled": pooled,
        "by_checkpoint": by_checkpoint,
        "bootstrap": bootstrap,
        "mean_offline_activation_reconstruction_mse": mean_activation_mse,
        "mean_offline_action_mse": mean_action_mse,
        "random_controls_by_checkpoint": random_gate,
        "gates": gates,
    }
    require_equal(
        file_sha256(Path(__file__).resolve()),
        summary_source_sha256,
        "summary source changed during execution",
    )
    for input_path, identity in input_identities.items():
        require_equal(
            file_sha256(Path(input_path)),
            identity["sha256"],
            f"{input_path}: changed before summary serialization",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
