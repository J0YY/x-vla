#!/usr/bin/env python3
"""Verify the corrected fixed-rank visual-bottleneck artifact with the standard library."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


ARTIFACT_DIR = Path(__file__).resolve().parent
ROOT = ARTIFACT_DIR.parents[1]
SEEDS = (0, 1, 2)
TASKS = (8, 9)
RANDOM_NAMES = ("random_topk", "random_topk_1", "random_topk_2")
EXPECTED_SUMMARY_SCHEMA = "xvla-corrected-blind-subspace-summary-v2"


class VerificationError(RuntimeError):
    """Raised when committed evidence differs from the frozen artifact."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"{path}: top-level value is not an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise VerificationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def require_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise VerificationError(f"{message}: expected {expected!r}, found {actual!r}")


def episode_map(condition: dict[str, Any], label: str) -> dict[tuple[int, int], bool]:
    rows = condition.get("episodes")
    require(isinstance(rows, list) and len(rows) == 50, f"{label}: incomplete episodes")
    output: dict[tuple[int, int], bool] = {}
    for row in rows:
        require(isinstance(row, dict), f"{label}: episode row is not an object")
        key = (int(row["task_index"]), int(row["episode"]))
        require(key not in output, f"{label}: duplicate episode {key}")
        require(isinstance(row.get("success"), bool), f"{label}: non-Boolean success")
        output[key] = row["success"]
    return output


def load_conditions(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    visual = payload.get("visual_subspace")
    require(isinstance(visual, dict), f"{path}: missing visual_subspace")
    require_equal(visual.get("rank"), 96, f"{path}: rank")
    require_equal(visual.get("gram_task_range"), [0, 4], f"{path}: discovery tasks")
    require_equal(
        visual.get("gram_task_index_space"),
        "official LIBERO task indices",
        f"{path}: task index space",
    )
    require_equal(visual.get("offline_eval_disjoint_from_gram"), True, f"{path}: cache split")
    conditions = visual.get("by_condition")
    require(isinstance(conditions, dict), f"{path}: missing conditions")
    return conditions


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def estimate(rows: list[tuple[bool, bool, bool, bool, bool, bool]]) -> dict[str, Any]:
    denominator = sum(int(row[0]) for row in rows)
    require(denominator > 0, "resample has no full-policy successes")
    retained = [sum(int(row[0] and row[index]) for row in rows) for index in range(1, 6)]
    return {
        "full_successes": denominator,
        "jacobian_retained": retained[0],
        "activation_retained": retained[1],
        "random_retained": retained[2:],
        "jacobian_retention": retained[0] / denominator,
        "activation_retention": retained[1] / denominator,
        "random_retention": [value / denominator for value in retained[2:]],
    }


def main() -> None:
    manifest = load_json(ARTIFACT_DIR / "manifest.json")
    require_equal(manifest.get("artifact_version"), 1, "artifact version")
    expected = manifest.get("expected")
    require(isinstance(expected, dict), "manifest has no expected block")

    summary_entry = manifest.get("summary")
    require(isinstance(summary_entry, dict), "manifest has no summary identity")
    summary_path = ROOT / str(summary_entry["path"])
    require_equal(sha256(summary_path), summary_entry["sha256"], "summary SHA-256")
    summary = load_json(summary_path)
    require_equal(summary.get("schema"), EXPECTED_SUMMARY_SCHEMA, "summary schema")
    require_equal(summary.get("scope", {}).get("tasks"), [8, 9], "summary tasks")
    require_equal(summary.get("scope", {}).get("rank"), 96, "summary rank")

    entries = manifest.get("raw_results")
    require(isinstance(entries, list) and len(entries) == 12, "manifest must list twelve raws")
    by_path: dict[str, str] = {}
    for entry in entries:
        require(isinstance(entry, dict), "raw identity is not an object")
        relative = str(entry["path"])
        require(relative not in by_path, f"duplicate raw identity {relative}")
        by_path[relative] = str(entry["sha256"])
        require_equal(sha256(ROOT / relative), entry["sha256"], f"{relative} SHA-256")

    strata: dict[tuple[int, int], list[tuple[bool, bool, bool, bool, bool, bool]]] = {}
    for seed in SEEDS:
        for task in TASKS:
            activation_relative = f"athena/results/visual_taskmap_activation_s{seed}_r96_t{task}.json"
            blind_relative = f"athena/results/visual_taskmap_vit_s{seed}_r96_blind_t{task}.json"
            require(activation_relative in by_path, f"missing {activation_relative}")
            require(blind_relative in by_path, f"missing {blind_relative}")
            activation = load_conditions(ROOT / activation_relative)
            blind = load_conditions(ROOT / blind_relative)
            full_activation = episode_map(activation["full"], f"{activation_relative}:full")
            full_blind = episode_map(blind["full"], f"{blind_relative}:full")
            require_equal(full_activation, full_blind, f"seed {seed} task {task}: full-policy pairing")
            keys = sorted(full_activation)
            require_equal(keys, [(task, episode) for episode in range(50)], f"seed {seed} task {task}: keys")
            jacobian = episode_map(activation["causal_topk"], f"{activation_relative}:jacobian")
            energy = episode_map(activation["activation_energy_topk"], f"{activation_relative}:energy")
            random_maps = [episode_map(blind[name], f"{blind_relative}:{name}") for name in RANDOM_NAMES]
            require(all(set(mapping) == set(keys) for mapping in [jacobian, energy, *random_maps]), "condition key mismatch")
            strata[(seed, task)] = [
                (
                    full_activation[key],
                    jacobian[key],
                    energy[key],
                    random_maps[0][key],
                    random_maps[1][key],
                    random_maps[2][key],
                )
                for key in keys
            ]

    point = estimate([row for rows in strata.values() for row in rows])
    for key in ("full_successes", "jacobian_retained", "activation_retained", "random_retained"):
        require_equal(point[key], expected[key], f"point estimate {key}")
    checkpoint_points = {
        seed: estimate(strata[(seed, 8)] + strata[(seed, 9)])
        for seed in SEEDS
    }
    checkpoint_random_signs = {
        str(seed): [
            value["jacobian_retention"] > random_value
            for random_value in value["random_retention"]
        ]
        for seed, value in checkpoint_points.items()
    }
    require(
        all(all(signs) for signs in checkpoint_random_signs.values()),
        "an action-Jacobian projector does not beat every random control within every checkpoint",
    )
    recomputed_selectivity_pass = (
        point["jacobian_retention"] >= 0.85
        and all(all(signs) for signs in checkpoint_random_signs.values())
    )
    require_equal(recomputed_selectivity_pass, expected["blind_selectivity_pass"], "recomputed selectivity gate")
    require_equal(summary["gates"]["blind_selectivity_pass"], recomputed_selectivity_pass, "summary selectivity gate")
    # A negative pooled Jacobian-minus-activation margin is sufficient to reject
    # the stronger action-specificity gate without trusting its stored Boolean.
    recomputed_specificity_pass = point["jacobian_retention"] - point["activation_retention"] >= 0.10
    require_equal(recomputed_specificity_pass, expected["action_specificity_pass"], "recomputed specificity gate")
    require_equal(summary["gates"]["action_specificity_pass"], recomputed_specificity_pass, "summary specificity gate")

    rng = random.Random(int(expected["bootstrap_seed"]))
    samples = int(expected["bootstrap_samples"])
    bootstrap: dict[str, list[float]] = {
        "jacobian_retention": [],
        "activation_retention": [],
        "activation_minus_random_0": [],
        "activation_minus_random_1": [],
        "activation_minus_random_2": [],
        "jacobian_minus_random_0": [],
        "jacobian_minus_random_1": [],
        "jacobian_minus_random_2": [],
    }
    for _ in range(samples):
        replicate: list[tuple[bool, bool, bool, bool, bool, bool]] = []
        for sampled_seed in rng.choices(SEEDS, k=len(SEEDS)):
            for sampled_task in rng.choices(TASKS, k=len(TASKS)):
                rows = strata[(sampled_seed, sampled_task)]
                replicate.extend(rng.choices(rows, k=len(rows)))
        value = estimate(replicate)
        bootstrap["jacobian_retention"].append(value["jacobian_retention"])
        bootstrap["activation_retention"].append(value["activation_retention"])
        for index, random_value in enumerate(value["random_retention"]):
            bootstrap[f"activation_minus_random_{index}"].append(value["activation_retention"] - random_value)
            bootstrap[f"jacobian_minus_random_{index}"].append(value["jacobian_retention"] - random_value)

    intervals = {
        name: [percentile(values, 0.025), percentile(values, 0.975)]
        for name, values in bootstrap.items()
    }
    require(all(math.isfinite(value) for interval in intervals.values() for value in interval), "non-finite interval")
    require(all(intervals[f"jacobian_minus_random_{index}"][0] > 0 for index in range(3)), "Jacobian-minus-random interval crosses zero")

    output = {
        "artifact_verified": True,
        "point": point,
        "checkpoint_random_signs": checkpoint_random_signs,
        "bootstrap": {
            "method": "paired checkpoint-task-episode cluster resampling",
            "samples": samples,
            "seed": expected["bootstrap_seed"],
            "intervals_95": intervals,
        },
        "claim_boundary": (
            "Rank 96 was fixed before corrected outcomes. Tasks 8 and 9 are absent from corrected "
            "projector construction, but were seen during policy training."
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except VerificationError as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
