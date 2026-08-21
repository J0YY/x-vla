#!/usr/bin/env python3
"""Verify the frozen ViT capability evidence using only the standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ARTIFACT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = ARTIFACT_DIR.parents[1]

EXPECTED_SCHEMA = "anonymous-vit-capability-artifact-v1"
EXPECTED_SHARDS = ((0, 3), (3, 6), (6, 8), (8, 10))
EXPECTED_TASKS = (
    "pick up the alphabet soup and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the orange juice and place it in the basket",
)
EXPECTED_PROTOCOL = {
    "architecture": "chi",
    "canonical_init_states": True,
    "episodes_per_task": 50,
    "execution_horizon": 8,
    "max_steps": 280,
    "settle_steps": 10,
    "shards": [list(shard) for shard in EXPECTED_SHARDS],
    "suite": "libero_object",
    "vision_encoder": "vit",
}
EXPECTED_CACHE_IDENTITY = {
    "path": "artifacts/libero_frames_100000_64.pkl",
    "provenance_path": "athena/results/cache_provenance_libero_object.json",
    "provenance_sha256": "1e3ed7eaeef317a221bb6649ea75ef68ff924b57797682e853b4bd90a138f4c4",
    "sha256": "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662",
}
EXPECTED_CHECKPOINTS = (
    {
        "path": "artifacts/ckpt_linear_rat_vit_s0_v2.pt",
        "seed": 0,
        "sha256": "96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9",
    },
    {
        "path": "artifacts/ckpt_linear_rat_vit_s1.pt",
        "seed": 1,
        "sha256": "cc0780b989165a80a449c03cbb3574e2f56b2e5747a64cf32d1fde418df8ec3c",
    },
    {
        "path": "artifacts/ckpt_linear_rat_vit_s2.pt",
        "seed": 2,
        "sha256": "413a770071bd8f16b924c5604f7c7a58567d2eb6c40c66aab202bcf046fec910",
    },
)
EXPECTED_MEMBER_TASK_SUCCESSES = {
    0: (37, 45, 49, 23, 48, 41, 49, 38, 49, 47),
    1: (43, 38, 49, 41, 46, 19, 50, 35, 34, 49),
    2: (32, 43, 49, 41, 50, 45, 47, 47, 49, 47),
}
EXPECTED_MEMBER_TOTALS = {0: 426, 1: 404, 2: 450}
EXPECTED_ENSEMBLE_TASK_SUCCESSES = (43, 47, 50, 43, 50, 42, 48, 46, 48, 50)
EXPECTED_ENSEMBLE_TOTAL = 467


class VerificationError(RuntimeError):
    """Raised when the evidence does not reproduce the frozen artifact."""


def _reject_constant(value: str) -> None:
    raise VerificationError(f"non-finite JSON constant {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"{path}: top-level JSON value is not an object")
    return value


def file_sha256(path: Path) -> str:
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


def require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise VerificationError(f"{label}: {actual!r} != {expected!r}")


def require_int(label: str, value: Any, *, lower: int | None = None, upper: int | None = None) -> int:
    require(type(value) is int, f"{label}: expected an integer")
    if lower is not None:
        require(value >= lower, f"{label}: {value} is below {lower}")
    if upper is not None:
        require(value <= upper, f"{label}: {value} exceeds {upper}")
    return value


def require_number(label: str, value: Any, *, nonnegative: bool = False) -> float:
    require(type(value) in (int, float), f"{label}: expected a number")
    number = float(value)
    require(math.isfinite(number), f"{label}: expected a finite number")
    if nonnegative:
        require(number >= 0.0, f"{label}: expected a nonnegative number")
    return number


def require_close(label: str, actual: Any, expected: Any) -> None:
    actual_value = require_number(label, actual)
    expected_value = require_number(f"{label} expected", expected)
    require(
        math.isclose(actual_value, expected_value, rel_tol=1e-12, abs_tol=1e-15),
        f"{label}: {actual_value!r} != {expected_value!r}",
    )


def safe_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    require(not candidate.is_absolute(), f"manifest path is absolute: {relative}")
    require(".." not in candidate.parts, f"manifest path escapes repository root: {relative}")
    return root / candidate


def expected_raw_paths() -> tuple[str, ...]:
    members = tuple(
        f"athena/results/chi_s{seed}_t{start}_{end}.json"
        for seed in range(3)
        for start, end in EXPECTED_SHARDS
    )
    ensemble = tuple(
        f"athena/results/ensemble_mean_t{start}_{end}.json"
        for start, end in EXPECTED_SHARDS
    )
    return members + ensemble


def verify_identity_manifest(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    require_equal("manifest schema", manifest.get("schema"), EXPECTED_SCHEMA)
    require_equal("manifest artifact version", manifest.get("artifact_version"), 1)
    require_equal("manifest protocol", manifest.get("protocol"), EXPECTED_PROTOCOL)
    require_equal("manifest task list", manifest.get("tasks"), list(EXPECTED_TASKS))
    require_equal(
        "manifest checkpoint identities",
        manifest.get("checkpoint_identities"),
        list(EXPECTED_CHECKPOINTS),
    )
    require_equal(
        "manifest cache identity",
        manifest.get("cache_identity"),
        EXPECTED_CACHE_IDENTITY,
    )

    provenance_path = safe_path(root, EXPECTED_CACHE_IDENTITY["provenance_path"])
    require_equal(
        "cache provenance result SHA-256",
        file_sha256(provenance_path),
        EXPECTED_CACHE_IDENTITY["provenance_sha256"],
    )
    provenance = load_json(provenance_path)
    require_equal("cache provenance suite", provenance.get("suite"), "libero_object")
    require_equal("cache provenance verified", provenance.get("verified"), True)
    provenance_cache = provenance.get("cache")
    require(isinstance(provenance_cache, dict), "cache provenance payload is absent")
    require_equal(
        "cache provenance path",
        provenance_cache.get("path"),
        EXPECTED_CACHE_IDENTITY["path"],
    )
    require_equal(
        "cache provenance SHA-256",
        provenance_cache.get("sha256"),
        EXPECTED_CACHE_IDENTITY["sha256"],
    )
    provenance_source = provenance.get("source")
    require(isinstance(provenance_source, dict), "cache source provenance is absent")
    require_equal("cache/source content match", provenance_source.get("content_hashes_match"), True)
    require_equal(
        "cache/source canonical identity",
        provenance_cache.get("canonical_content_sha256"),
        provenance_source.get("canonical_content_sha256"),
    )

    entries = manifest.get("raw_results")
    require(isinstance(entries, list), "manifest raw_results is not a list")
    require_equal("raw result count", len(entries), 16)
    require_equal(
        "raw result path order",
        tuple(entry.get("path") for entry in entries if isinstance(entry, dict)),
        expected_raw_paths(),
    )
    seen_paths: set[str] = set()
    for index, entry in enumerate(entries):
        require(isinstance(entry, dict), f"raw result entry {index} is not an object")
        relative = entry.get("path")
        digest = entry.get("sha256")
        require(isinstance(relative, str), f"raw result entry {index} has no path")
        require(relative not in seen_paths, f"duplicate raw result path {relative}")
        seen_paths.add(relative)
        require(isinstance(digest, str) and len(digest) == 64, f"invalid SHA-256 for {relative}")
        require_equal(f"SHA-256 {relative}", file_sha256(safe_path(root, relative)), digest)

        cohort = entry.get("cohort")
        start = require_int(f"{relative} task_start", entry.get("task_start"))
        end = require_int(f"{relative} task_end", entry.get("task_end"))
        require((start, end) in EXPECTED_SHARDS, f"{relative}: unexpected shard [{start}, {end})")
        if cohort == "member":
            seed = require_int(f"{relative} seed", entry.get("seed"), lower=0, upper=2)
            require_equal(
                f"{relative} path identity",
                relative,
                f"athena/results/chi_s{seed}_t{start}_{end}.json",
            )
        elif cohort == "ensemble":
            require("seed" not in entry, f"{relative}: ensemble manifest entry must not have a seed")
            require_equal(
                f"{relative} path identity",
                relative,
                f"athena/results/ensemble_mean_t{start}_{end}.json",
            )
        else:
            raise VerificationError(f"{relative}: invalid cohort {cohort!r}")
    return entries


def verify_capability_shard(
    result: dict[str, Any],
    entry: dict[str, Any],
    canonical_cache_stats: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    relative = str(entry["path"])
    cohort = str(entry["cohort"])
    start = int(entry["task_start"])
    end = int(entry["task_end"])
    expected_tasks = list(range(start, end))

    require_equal(f"{relative} architecture", result.get("architecture"), "chi")
    require_equal(f"{relative} cache", result.get("cache"), EXPECTED_CACHE_IDENTITY["path"])
    require_equal(f"{relative} vocabulary size", result.get("vocab_size"), 26)
    require_equal(f"{relative} prediction shape", result.get("prediction_shape"), [1, 8, 7])
    require_equal(f"{relative} finite prediction", result.get("prediction_finite"), True)
    for optional_field, expected in (
        ("suite", "libero_object"),
        ("training_suite", "libero_object"),
        ("vision_encoder", "vit"),
        ("evaluation_scope", "In-domain LIBERO-Object evaluation"),
    ):
        if optional_field in result:
            require_equal(f"{relative} {optional_field}", result[optional_field], expected)

    cache_stats = result.get("cache_stats")
    require(isinstance(cache_stats, dict), f"{relative}: cache statistics are absent")
    if canonical_cache_stats is not None:
        require_equal(f"{relative} cache statistics", cache_stats, canonical_cache_stats)

    checkpoint_paths = [identity["path"] for identity in EXPECTED_CHECKPOINTS]
    if cohort == "member":
        seed = int(entry["seed"])
        require_equal(f"{relative} mode", result.get("mode"), "capability")
        require_equal(f"{relative} seed", result.get("seed"), seed)
        require_equal(f"{relative} checkpoint", result.get("checkpoint"), checkpoint_paths[seed])
        require_equal(f"{relative} parameter count", result.get("profile", {}).get("parameters"), 20137352)
        require_close(
            f"{relative} parameter count in millions",
            result.get("profile", {}).get("parameters_millions"),
            20.137352,
        )
    else:
        require_equal(f"{relative} mode", result.get("mode"), "ensemble_capability")
        require_equal(f"{relative} anchor seed", result.get("seed"), 0)
        require_equal(f"{relative} checkpoint", result.get("checkpoint"), checkpoint_paths[0])
        require_equal(f"{relative} ensemble checkpoints", result.get("ensemble_checkpoints"), checkpoint_paths)
        require_equal(f"{relative} ensemble reduction", result.get("ensemble_reduction"), "mean")
        require_equal(f"{relative} pruning state", result.get("cp_pruning"), None)
        require_equal(f"{relative} parameter count", result.get("profile", {}).get("parameters"), 60412056)
        require_close(
            f"{relative} parameter count in millions",
            result.get("profile", {}).get("parameters_millions"),
            60.412056,
        )

    profile = result.get("profile")
    require(isinstance(profile, dict), f"{relative}: profile is absent")
    require(isinstance(profile.get("gpu"), str) and profile["gpu"], f"{relative}: GPU identity is absent")
    require_number(f"{relative} latency", profile.get("latency_ms_batch1"), nonnegative=True)

    capability = result.get("capability")
    require(isinstance(capability, dict), f"{relative}: capability payload is absent")
    require_equal(f"{relative} task indices", capability.get("task_indices"), expected_tasks)
    require_equal(f"{relative} episodes per task", capability.get("eps_per_task"), 50)
    require_equal(f"{relative} max steps", capability.get("max_steps"), 280)
    require_equal(f"{relative} settle steps", capability.get("num_steps_wait"), 10)
    require_equal(f"{relative} execution horizon", capability.get("exec_h"), 8)
    require_equal(f"{relative} canonical initial states", capability.get("canonical_init_states"), True)
    require_number(f"{relative} total elapsed time", capability.get("elapsed_s"), nonnegative=True)

    episodes = capability.get("episodes")
    require(isinstance(episodes, list), f"{relative}: episode rows are absent")
    require_equal(f"{relative} episode row count", len(episodes), 50 * len(expected_tasks))
    rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row_index, row in enumerate(episodes):
        require(isinstance(row, dict), f"{relative}: episode row {row_index} is not an object")
        require_equal(
            f"{relative} episode row {row_index} fields",
            set(row),
            {"task_index", "episode", "success", "steps", "elapsed_s"},
        )
        task = require_int(f"{relative} row {row_index} task", row.get("task_index"))
        episode = require_int(
            f"{relative} row {row_index} episode", row.get("episode"), lower=0, upper=49
        )
        require(task in expected_tasks, f"{relative}: row {row_index} has task {task}")
        key = (task, episode)
        require(key not in rows_by_key, f"{relative}: duplicate episode key {key}")
        require(type(row.get("success")) is bool, f"{relative}: success at {key} is not Boolean")
        steps = require_int(f"{relative} steps at {key}", row.get("steps"), lower=1, upper=280)
        if not row["success"]:
            require_equal(f"{relative} failed-episode steps at {key}", steps, 280)
        require_number(f"{relative} elapsed time at {key}", row.get("elapsed_s"), nonnegative=True)
        rows_by_key[key] = row

    expected_keys = {(task, episode) for task in expected_tasks for episode in range(50)}
    require_equal(f"{relative} task/episode matrix", set(rows_by_key), expected_keys)
    per_task_successes = {
        task: sum(int(rows_by_key[(task, episode)]["success"]) for episode in range(50))
        for task in expected_tasks
    }
    successes = sum(per_task_successes.values())
    trials = len(rows_by_key)
    require_equal(f"{relative} recorded successes", capability.get("successes"), successes)
    require_equal(f"{relative} recorded trials", capability.get("trials"), trials)
    require_close(f"{relative} recorded overall", capability.get("overall"), successes / trials)

    recorded_per_task = capability.get("per_task")
    require(isinstance(recorded_per_task, dict), f"{relative}: per-task summary is absent")
    require_equal(
        f"{relative} per-task instruction set",
        set(recorded_per_task),
        {EXPECTED_TASKS[task] for task in expected_tasks},
    )
    for task, task_successes in per_task_successes.items():
        require_close(
            f"{relative} task {task} success rate",
            recorded_per_task[EXPECTED_TASKS[task]],
            task_successes / 50,
        )
    return (
        {
            "cohort": cohort,
            "seed": entry.get("seed"),
            "task_start": start,
            "task_end": end,
            "successes": successes,
            "trials": trials,
            "per_task_successes": per_task_successes,
            "gpu": profile["gpu"],
        },
        cache_stats,
    )


def verify_frozen_outcomes(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    member_task_successes: dict[int, dict[int, int]] = defaultdict(dict)
    member_trials: dict[int, int] = defaultdict(int)
    ensemble_task_successes: dict[int, int] = {}
    ensemble_trials = 0
    for row in rows:
        if row["cohort"] == "member":
            seed = int(row["seed"])
            for task, successes in row["per_task_successes"].items():
                require(task not in member_task_successes[seed], f"duplicate member seed {seed} task {task}")
                member_task_successes[seed][task] = successes
            member_trials[seed] += int(row["trials"])
        else:
            for task, successes in row["per_task_successes"].items():
                require(task not in ensemble_task_successes, f"duplicate ensemble task {task}")
                ensemble_task_successes[task] = successes
            ensemble_trials += int(row["trials"])

    require_equal("member seed set", set(member_task_successes), {0, 1, 2})
    members = []
    for seed in range(3):
        ordered = tuple(member_task_successes[seed].get(task) for task in range(10))
        require_equal(f"seed {seed} per-task successes", ordered, EXPECTED_MEMBER_TASK_SUCCESSES[seed])
        successes = sum(ordered)
        trials = member_trials[seed]
        require_equal(f"seed {seed} total successes", successes, EXPECTED_MEMBER_TOTALS[seed])
        require_equal(f"seed {seed} total trials", trials, 500)
        members.append(
            {
                "seed": seed,
                "successes": successes,
                "trials": trials,
                "success_rate": successes / trials,
                "per_task_successes": list(ordered),
                "per_task_success_rates": [value / 50 for value in ordered],
            }
        )

    ordered_ensemble = tuple(ensemble_task_successes.get(task) for task in range(10))
    require_equal(
        "ensemble per-task successes", ordered_ensemble, EXPECTED_ENSEMBLE_TASK_SUCCESSES
    )
    ensemble_successes = sum(ordered_ensemble)
    require_equal("ensemble total successes", ensemble_successes, EXPECTED_ENSEMBLE_TOTAL)
    require_equal("ensemble total trials", ensemble_trials, 500)

    member_rates = [member["success_rate"] for member in members]
    member_mean = statistics.mean(member_rates)
    member_sample_sd = statistics.stdev(member_rates)
    ensemble_rate = ensemble_successes / ensemble_trials
    gain_over_mean_pp = (ensemble_rate - member_mean) * 100
    gain_over_best_pp = (ensemble_rate - max(member_rates)) * 100

    expected = manifest.get("expected")
    require(isinstance(expected, dict), "manifest expected outcome is absent")
    require_equal("manifest member outcome count", len(expected.get("members", [])), 3)
    for actual, recorded in zip(members, expected["members"]):
        require_equal(f"manifest seed {actual['seed']} identity", recorded.get("seed"), actual["seed"])
        require_equal(f"manifest seed {actual['seed']} successes", recorded.get("successes"), actual["successes"])
        require_equal(f"manifest seed {actual['seed']} trials", recorded.get("trials"), actual["trials"])
        require_equal(
            f"manifest seed {actual['seed']} per-task successes",
            recorded.get("per_task_successes"),
            actual["per_task_successes"],
        )
        require_close(
            f"manifest seed {actual['seed']} rate",
            recorded.get("success_rate"),
            actual["success_rate"],
        )
    require_close("manifest member mean", expected.get("member_success_rate_mean"), member_mean)
    require_close(
        "manifest member sample SD",
        expected.get("member_success_rate_sample_sd"),
        member_sample_sd,
    )

    expected_ensemble = expected.get("ensemble")
    require(isinstance(expected_ensemble, dict), "manifest ensemble outcome is absent")
    require_equal("manifest ensemble successes", expected_ensemble.get("successes"), ensemble_successes)
    require_equal("manifest ensemble trials", expected_ensemble.get("trials"), ensemble_trials)
    require_equal(
        "manifest ensemble per-task successes",
        expected_ensemble.get("per_task_successes"),
        list(ordered_ensemble),
    )
    require_close("manifest ensemble rate", expected_ensemble.get("success_rate"), ensemble_rate)
    require_close(
        "manifest ensemble gain over mean",
        expected_ensemble.get("gain_over_member_mean_percentage_points"),
        gain_over_mean_pp,
    )
    require_close(
        "manifest ensemble gain over best",
        expected_ensemble.get("gain_over_best_member_percentage_points"),
        gain_over_best_pp,
    )
    require_equal(
        "manifest rounded gain over mean",
        expected_ensemble.get("gain_over_member_mean_percentage_points_rounded_1dp"),
        round(gain_over_mean_pp, 1),
    )
    require_equal(
        "manifest rounded gain over best",
        expected_ensemble.get("gain_over_best_member_percentage_points_rounded_1dp"),
        round(gain_over_best_pp, 1),
    )

    task_member_mean_rates = [
        statistics.mean(member["per_task_success_rates"][task] for member in members)
        for task in range(10)
    ]
    return {
        "protocol": EXPECTED_PROTOCOL,
        "raw_result_hashes_verified": 16,
        "members": members,
        "member_success_rate_mean": member_mean,
        "member_success_rate_sample_sd": member_sample_sd,
        "ensemble": {
            "successes": ensemble_successes,
            "trials": ensemble_trials,
            "success_rate": ensemble_rate,
            "per_task_successes": list(ordered_ensemble),
            "per_task_success_rates": [value / 50 for value in ordered_ensemble],
            "gain_over_member_mean_percentage_points": gain_over_mean_pp,
            "gain_over_best_member_percentage_points": gain_over_best_pp,
        },
        "figure_data": {
            "task_labels": [
                "alphabet soup",
                "cream cheese",
                "salad dressing",
                "bbq sauce",
                "ketchup",
                "tomato sauce",
                "butter",
                "milk",
                "choc. pudding",
                "orange juice",
            ],
            "member_per_task_mean_success_rates": task_member_mean_rates,
            "ensemble_per_task_success_rates": [value / 50 for value in ordered_ensemble],
        },
    }


def verify_artifact(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    manifest = load_json(ARTIFACT_DIR / "manifest.json")
    entries = verify_identity_manifest(root, manifest)
    rows = []
    canonical_cache_stats: dict[str, Any] | None = None
    for entry in entries:
        result = load_json(safe_path(root, str(entry["path"])))
        row, cache_stats = verify_capability_shard(result, entry, canonical_cache_stats)
        if canonical_cache_stats is None:
            canonical_cache_stats = cache_stats
        rows.append(row)
    return verify_frozen_outcomes(manifest, rows)


def print_summary(summary: dict[str, Any]) -> None:
    print("PASS 16 immutable raw-result SHA-256 identities")
    print("PASS Object cache provenance and frozen checkpoint identities")
    print("PASS 2,000 complete canonical task/episode rows at a 280-step cap")
    member_text = ", ".join(
        f"seed {member['seed']} {member['successes']}/{member['trials']} "
        f"({100 * member['success_rate']:.1f}%)"
        for member in summary["members"]
    )
    print(f"Members: {member_text}")
    print(
        "Member mean: "
        f"{100 * summary['member_success_rate_mean']:.1f}% +/- "
        f"{100 * summary['member_success_rate_sample_sd']:.1f}% sample SD"
    )
    ensemble = summary["ensemble"]
    print(
        f"Mean ensemble: {ensemble['successes']}/{ensemble['trials']} "
        f"({100 * ensemble['success_rate']:.1f}%)"
    )
    print(
        "Gains: "
        f"+{ensemble['gain_over_member_mean_percentage_points']:.1f} points over member mean, "
        f"+{ensemble['gain_over_best_member_percentage_points']:.1f} points over best member"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="repository root")
    parser.add_argument(
        "--summary-json",
        action="store_true",
        help="emit the recomputed summary as JSON",
    )
    args = parser.parse_args()
    try:
        summary = verify_artifact(args.root.resolve())
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if args.summary_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
