#!/usr/bin/env python3
"""Verify the frozen four-suite breadth evidence using only the standard library."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


ARTIFACT_DIR = Path(__file__).resolve().parent
ROOT = ARTIFACT_DIR.parents[1]
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
SEEDS = (0, 1, 2)


class VerificationError(RuntimeError):
    """Raised when the evidence does not reproduce the frozen artifact."""


def reject_constant(value: str) -> None:
    raise VerificationError(f"non-finite JSON constant {value}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
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
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
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


def equal(label: str, actual: Any, expected: Any) -> None:
    require(actual == expected, f"{label}: {actual!r} != {expected!r}")


def close(label: str, actual: Any, expected: Any) -> None:
    require(type(actual) in (int, float), f"{label}: expected a number")
    require(type(expected) in (int, float), f"{label} expected: expected a number")
    require(math.isfinite(float(actual)), f"{label}: expected a finite number")
    require(
        math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-15),
        f"{label}: {actual!r} != {expected!r}",
    )


def safe_path(relative: str) -> Path:
    path = Path(relative)
    require(not path.is_absolute(), f"absolute evidence path: {relative}")
    require(".." not in path.parts, f"evidence path escapes repository root: {relative}")
    return ROOT / path


def verify_evidence_identities(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    equal("artifact schema", manifest.get("schema"), "anonymous-four-suite-breadth-artifact-v1")
    equal("artifact version", manifest.get("artifact_version"), 1)
    entries = manifest.get("evidence")
    require(isinstance(entries, dict), "artifact evidence map is absent")
    equal(
        "evidence keys",
        set(entries),
        {
            "ensemble_manifest",
            "ensemble_summary",
            "generalist_summary",
            "retention_summary",
            "train_chi_s0",
            "train_chi_s1",
            "train_chi_s2",
            "train_conventional_s0",
            "train_conventional_s1",
            "train_conventional_s2",
        },
    )
    loaded: dict[str, dict[str, Any]] = {}
    for name, entry in entries.items():
        require(isinstance(entry, dict), f"{name}: evidence entry is not an object")
        relative = entry.get("path")
        expected_digest = entry.get("sha256")
        require(isinstance(relative, str), f"{name}: evidence path is absent")
        require(isinstance(expected_digest, str) and len(expected_digest) == 64, f"{name}: invalid digest")
        path = safe_path(relative)
        equal(f"{name} SHA-256", file_sha256(path), expected_digest)
        text = path.read_text(encoding="utf-8")
        require("/work/joy" not in text and "/athenahomes/" not in text, f"{name}: identity leak")
        loaded[name] = load_json(path)
    return loaded


def verify_members(record: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    equal("ensemble manifest schema", record.get("schema"), "xvla_multisuite_mean_ensemble_manifest_v1")
    equal("ensemble freeze status", record.get("status"), "frozen_after_member_evaluations_before_ensemble_outcomes")
    scope = record.get("scope")
    require(isinstance(scope, dict), "ensemble scope is absent")
    equal("ensemble suites", scope.get("suites"), list(SUITES))
    equal("ensemble seeds", scope.get("seeds"), list(SEEDS))
    equal("ensemble trials", scope.get("ensemble_trials"), 2000)
    equal("ensemble inference cost", scope.get("inference_cost"), "three policy forward passes per query")

    members = record.get("member_results")
    require(isinstance(members, dict), "member results are absent")
    equal("member seeds", set(members), {str(seed) for seed in SEEDS})
    observed_macros: list[float] = []
    task_seed_values: dict[str, list[float]] = {f"{suite}:{task}": [] for suite in SUITES for task in range(10)}
    suite_seed_values: dict[str, list[float]] = {suite: [] for suite in SUITES}
    for seed in SEEDS:
        member = members[str(seed)]
        require(isinstance(member, dict), f"seed {seed}: member result is not an object")
        equal(f"seed {seed} trials", member.get("trials"), 2000)
        task_success = member.get("task_success")
        suite_success = member.get("suite_macro_success")
        require(isinstance(task_success, dict) and len(task_success) == 40, f"seed {seed}: task map is incomplete")
        require(isinstance(suite_success, dict) and set(suite_success) == set(SUITES), f"seed {seed}: suite map is incomplete")
        macro = statistics.fmean(float(task_success[key]) for key in sorted(task_success))
        close(f"seed {seed} recorded macro", member.get("macro_task_success"), macro)
        close(f"seed {seed} successes", member.get("successes"), macro * 2000)
        observed_macros.append(macro)
        for suite in SUITES:
            suite_tasks = [float(task_success[f"{suite}:{task}"]) for task in range(10)]
            suite_macro = statistics.fmean(suite_tasks)
            close(f"seed {seed} {suite} macro", suite_success[suite], suite_macro)
            suite_seed_values[suite].append(suite_macro)
            for task, value in enumerate(suite_tasks):
                task_seed_values[f"{suite}:{task}"].append(value)

    expected_macros = expected["macro_successes"]
    equal("expected member macro count", len(expected_macros), 3)
    for seed, (actual, wanted) in enumerate(zip(observed_macros, expected_macros, strict=True)):
        close(f"member macro seed {seed}", actual, wanted)
    member_mean = statistics.fmean(observed_macros)
    member_sd = statistics.stdev(observed_macros)
    close("member mean", member_mean, expected["mean"])
    close("member sample standard deviation", member_sd, expected["sample_standard_deviation"])
    close("manifest member mean", record.get("member_mean_macro_success"), member_mean)
    recorded_macros = record.get("member_macro_successes")
    require(isinstance(recorded_macros, list), "manifest member macros are absent")
    equal("manifest member macro count", len(recorded_macros), 3)
    for seed, (recorded, observed) in enumerate(zip(recorded_macros, observed_macros, strict=True)):
        close(f"manifest member macro seed {seed}", recorded, observed)

    suite_seed_means = {suite: statistics.fmean(values) for suite, values in suite_seed_values.items()}
    for suite, wanted in expected["suite_seed_means"].items():
        close(f"member seed-mean suite {suite}", suite_seed_means[suite], wanted)
    tasks_at_least = sum(statistics.fmean(values) >= 0.5 for values in task_seed_values.values())
    equal("member seed-mean task threshold count", tasks_at_least, expected["tasks_with_seed_mean_at_least_0p50"])
    return {"members": members, "mean": member_mean, "sample_sd": member_sd, "suite_means": suite_seed_means}


def verify_training_records(
    records: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, int]:
    identity = summary.get("identity")
    require(isinstance(identity, dict), "generalist summary identity is absent")
    parameters: dict[str, int] = {}
    for architecture in ("chi", "conventional"):
        expected_parameters = expected[f"{architecture}_parameters"]
        for seed in SEEDS:
            record = records[f"train_{architecture}_s{seed}"]
            equal(f"{architecture} seed {seed} architecture", record.get("architecture"), architecture)
            equal(f"{architecture} seed {seed} seed", record.get("seed"), seed)
            equal(f"{architecture} seed {seed} steps", record.get("steps"), 160000)
            equal(f"{architecture} seed {seed} batch size", record.get("batch_size"), 256)
            equal(f"{architecture} seed {seed} suite batch size", record.get("suite_batch_size"), 64)
            close(f"{architecture} seed {seed} learning rate", record.get("lr"), 8e-4)
            close(f"{architecture} seed {seed} EMA", record.get("ema_decay"), 0.999)
            equal(f"{architecture} seed {seed} horizon", record.get("horizon"), 8)
            equal(f"{architecture} seed {seed} resolution", record.get("res"), 64)
            equal(f"{architecture} seed {seed} encoder", record.get("vision_encoder"), "vit")
            equal(
                f"{architecture} seed {seed} suites",
                record.get("training_suites"),
                list(SUITES),
            )
            equal(
                f"{architecture} seed {seed} suite sampling",
                record.get("sampling"),
                "equal examples per suite per optimizer update",
            )
            equal(f"{architecture} seed {seed} parameters", record.get("parameters"), expected_parameters)
            equal(
                f"{architecture} seed {seed} checkpoint",
                record.get("checkpoint_sha256"),
                summary["by_architecture_and_seed"][architecture][str(seed)]["checkpoint_sha256"],
            )
            equal(
                f"{architecture} seed {seed} manifest",
                record.get("manifest_sha256"),
                identity["manifest_start_sha256"],
            )
            equal(
                f"{architecture} seed {seed} trainer",
                record.get("trainer_sha256"),
                identity["trainer_sha256"],
            )
            source = record.get("source_identity")
            require(isinstance(source, dict), f"{architecture} seed {seed}: source identity is absent")
            equal(
                f"{architecture} seed {seed} trainer closure",
                source.get("trainer_start_sha256"),
                source.get("trainer_end_sha256"),
            )
            equal(
                f"{architecture} seed {seed} imported-source closure",
                source.get("imported_sources_start"),
                source.get("imported_sources_end"),
            )
            equal(
                f"{architecture} seed {seed} imported-bundle closure",
                source.get("imported_bundle_start_sha256"),
                source.get("imported_bundle_end_sha256"),
            )
            equal(
                f"{architecture} seed {seed} manifest closure",
                source.get("manifest_start_sha256"),
                source.get("manifest_end_sha256"),
            )
            equal(
                f"{architecture} seed {seed} cache closure",
                source.get("source_cache_hashes_start"),
                source.get("source_cache_hashes_end"),
            )
        parameters[architecture] = expected_parameters
    return parameters


def verify_matched_comparison(
    record: dict[str, Any],
    members: dict[str, Any],
    records: dict[str, dict[str, Any]],
    expected: dict[str, Any],
) -> dict[str, Any]:
    scope = record.get("scope")
    require(isinstance(scope, dict), "generalist comparison scope is absent")
    equal("generalist architectures", scope.get("architectures"), ["chi", "conventional"])
    equal("generalist seeds", scope.get("seeds"), list(SEEDS))
    equal("generalist suites", scope.get("suites"), list(SUITES))
    equal("generalist tasks per suite", scope.get("tasks_per_suite"), 10)
    equal("generalist episodes per task", scope.get("episodes_per_task"), 50)
    equal("generalist total trials", scope.get("total_trials"), 12000)
    equal(
        "generalist claim boundary",
        scope.get("claim_boundary"),
        "In-domain capability of one jointly trained checkpoint per seed across all 40 tasks. "
        "This is not zero-shot suite or task generalization.",
    )

    identity = record.get("identity")
    require(isinstance(identity, dict), "generalist comparison identity is absent")
    equal("generalist manifest closure", identity.get("manifest_start_sha256"), identity.get("manifest_end_sha256"))
    equal("generalist summary closure", identity.get("summary_start_sha256"), identity.get("summary_end_sha256"))
    equal(
        "generalist summary source",
        identity.get("summary_start_sha256"),
        file_sha256(safe_path("athena/summarize_multisuite_generalist.py")),
    )
    parameters = verify_training_records(records, record, expected)

    observed: dict[str, dict[str, Any]] = {}
    for architecture in ("chi", "conventional"):
        seed_macros: list[float] = []
        suite_seed_values = {suite: [] for suite in SUITES}
        task_seed_values = {f"{suite}:{task}": [] for suite in SUITES for task in range(10)}
        for seed in SEEDS:
            result = record["by_architecture_and_seed"][architecture][str(seed)]
            equal(f"{architecture} seed {seed} trials", result.get("trials"), 2000)
            source_files = result.get("source_files")
            require(isinstance(source_files, list), f"{architecture} seed {seed}: sources are absent")
            equal(f"{architecture} seed {seed} source count", len(source_files), 16)
            equal(
                f"{architecture} seed {seed} unique sources",
                len({item["path"] for item in source_files}),
                16,
            )
            task_success = result.get("task_success")
            suite_success = result.get("suite_macro_success")
            require(isinstance(task_success, dict) and len(task_success) == 40, f"{architecture} seed {seed}: task map is incomplete")
            require(isinstance(suite_success, dict) and set(suite_success) == set(SUITES), f"{architecture} seed {seed}: suite map is incomplete")
            macro = statistics.fmean(float(task_success[key]) for key in sorted(task_success))
            close(f"{architecture} seed {seed} recorded macro", result.get("macro_task_success"), macro)
            close(f"{architecture} seed {seed} successes", result.get("successes"), 2000 * macro)
            threshold_count = sum(float(value) >= 0.5 for value in task_success.values())
            equal(f"{architecture} seed {seed} threshold count", result.get("tasks_at_least_0p50"), threshold_count)
            seed_macros.append(macro)
            for suite in SUITES:
                suite_macro = statistics.fmean(float(task_success[f"{suite}:{task}"]) for task in range(10))
                close(f"{architecture} seed {seed} {suite} macro", suite_success[suite], suite_macro)
                suite_seed_values[suite].append(suite_macro)
                for task in range(10):
                    task_seed_values[f"{suite}:{task}"].append(float(task_success[f"{suite}:{task}"]))
            if architecture == "chi":
                equal(f"chi seed {seed} task outcomes", task_success, members[str(seed)]["task_success"])

        mean = statistics.fmean(seed_macros)
        sample_sd = statistics.stdev(seed_macros)
        suite_means = {suite: statistics.fmean(values) for suite, values in suite_seed_values.items()}
        task_means = {key: statistics.fmean(values) for key, values in task_seed_values.items()}
        tasks_at_least = sum(value >= 0.5 for value in task_means.values())
        aggregate = record["aggregate"][architecture]
        for seed, (actual, recorded) in enumerate(zip(seed_macros, aggregate["seed_macro_task_success"], strict=True)):
            close(f"{architecture} aggregate seed {seed}", recorded, actual)
        close(f"{architecture} aggregate mean", aggregate.get("seed_mean_macro_task_success"), mean)
        close(f"{architecture} aggregate sample standard deviation", aggregate.get("seed_sample_std_macro_task_success"), sample_sd)
        for suite in SUITES:
            close(f"{architecture} aggregate {suite}", aggregate["suite_seed_mean_macro_success"][suite], suite_means[suite])
        recorded_task_means = aggregate.get("task_seed_mean_success")
        require(
            isinstance(recorded_task_means, dict) and set(recorded_task_means) == set(task_means),
            f"{architecture}: aggregate task means are incomplete",
        )
        for key, value in task_means.items():
            close(f"{architecture} aggregate task mean {key}", recorded_task_means[key], value)
        equal(f"{architecture} aggregate threshold count", aggregate.get("tasks_seed_mean_at_least_0p50"), tasks_at_least)
        observed[architecture] = {
            "macros": seed_macros,
            "mean": mean,
            "sample_sd": sample_sd,
            "suite_means": suite_means,
            "tasks_at_least": tasks_at_least,
        }

    conventional = observed["conventional"]
    for seed, (actual, wanted) in enumerate(zip(conventional["macros"], expected["conventional_macro_successes"], strict=True)):
        close(f"expected conventional seed {seed}", actual, wanted)
    close("expected conventional mean", conventional["mean"], expected["conventional_mean"])
    close(
        "expected conventional sample standard deviation",
        conventional["sample_sd"],
        expected["conventional_sample_standard_deviation"],
    )
    for suite in SUITES:
        close(
            f"expected conventional {suite}",
            conventional["suite_means"][suite],
            expected["conventional_suite_seed_means"][suite],
        )
    equal(
        "expected conventional task threshold count",
        conventional["tasks_at_least"],
        expected["conventional_tasks_with_seed_mean_at_least_0p50"],
    )
    delta = observed["chi"]["mean"] - conventional["mean"]
    close("recorded chi minus conventional macro", record.get("chi_minus_conventional_macro"), delta)
    close("expected chi minus conventional macro", delta, expected["chi_minus_conventional_macro"])
    comparison_pass = delta >= -expected["maximum_allowed_chi_deficit"]
    equal(
        "matched comparison gate",
        record.get("secondary_gates"),
        {"chi_within_0p05_of_conventional": comparison_pass},
    )
    require(comparison_pass, "matched comparison gate did not pass")
    return {
        "chi_mean": observed["chi"]["mean"],
        "conventional_mean": conventional["mean"],
        "delta": delta,
        "chi_parameters": parameters["chi"],
        "conventional_parameters": parameters["conventional"],
    }


def verify_ensemble(record: dict[str, Any], manifest_record: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    equal("ensemble summary manifest SHA-256", record.get("manifest", {}).get("sha256"), file_sha256(safe_path("athena/results/multisuite_mean_ensemble_v1_manifest.json")))
    equal("ensemble summary source count", len(record.get("source_files", [])), 16)
    source_paths = [item.get("path") for item in record["source_files"]]
    equal("ensemble summary unique sources", len(set(source_paths)), 16)
    equal("ensemble summary trials", record.get("trials"), expected["trials"])
    equal("ensemble summary successes", record.get("successes"), expected["successes"])
    task_success = record.get("task_success")
    require(isinstance(task_success, dict) and len(task_success) == 40, "ensemble task map is incomplete")
    macro = statistics.fmean(float(task_success[key]) for key in sorted(task_success))
    close("ensemble recorded macro", record.get("macro_task_success"), macro)
    close("ensemble expected macro", macro, expected["macro_task_success"])
    close("ensemble count-derived macro", macro, record["successes"] / record["trials"])
    suite_success = record.get("suite_macro_success")
    require(isinstance(suite_success, dict) and set(suite_success) == set(SUITES), "ensemble suite map is incomplete")
    for suite in SUITES:
        suite_macro = statistics.fmean(float(task_success[f"{suite}:{task}"]) for task in range(10))
        close(f"ensemble {suite} macro", suite_success[suite], suite_macro)
        close(f"ensemble expected {suite} macro", suite_macro, expected["suite_macro_success"][suite])
    threshold_count = sum(float(value) >= 0.5 for value in task_success.values())
    equal("ensemble task threshold count", threshold_count, expected["tasks_at_least_0p50"])
    equal("ensemble recorded task threshold count", record.get("tasks_at_least_0p50"), threshold_count)
    member_mean = float(manifest_record["member_mean_macro_success"])
    uplift = macro - member_mean
    close("ensemble member mean", record.get("member_mean_macro_success"), member_mean)
    close("ensemble uplift", record.get("uplift_over_member_mean"), uplift)
    close("ensemble expected uplift", uplift, expected["uplift_over_member_mean"])
    frozen_gates = manifest_record.get("frozen_gates")
    equal("ensemble frozen gates", frozen_gates, expected["frozen_gates"])
    computed_gates = {
        "every_suite_macro_at_least": min(float(value) for value in suite_success.values())
        >= frozen_gates["every_suite_macro_at_least"],
        "macro_task_success_at_least": macro >= frozen_gates["macro_task_success_at_least"],
        "tasks_at_least_0p50_at_least": threshold_count
        >= frozen_gates["tasks_at_least_0p50_at_least"],
        "uplift_over_member_mean_at_least": uplift
        >= frozen_gates["uplift_over_member_mean_at_least"],
    }
    computed_gates["ensemble_capability_pass"] = all(computed_gates.values())
    equal("ensemble gates", record.get("gates"), computed_gates)
    return {"macro": macro, "suite_success": suite_success, "tasks_at_least": threshold_count, "uplift": uplift}


def verify_retention(record: dict[str, Any], members: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    scope = record.get("scope")
    require(isinstance(scope, dict), "retention scope is absent")
    equal("retention status", scope.get("status"), "separate_frozen_retention_certificate")
    equal("retention seed", scope.get("seed"), 0)
    equal("retention suites", scope.get("suites"), list(SUITES))
    equal("generalist capability summary modified", record.get("generalist_capability_summary_modified"), False)
    by_suite = record.get("by_suite")
    require(isinstance(by_suite, dict) and set(by_suite) == set(SUITES), "retention suite map is incomplete")
    generalist_tasks: list[float] = []
    specialist_tasks: list[float] = []
    seed0 = members["0"]
    for suite in SUITES:
        suite_record = by_suite[suite]
        generalist = [float(suite_record["generalist_task_success"][str(task)]) for task in range(10)]
        specialist = [float(suite_record["specialist"]["task_success"][str(task)]) for task in range(10)]
        close(f"retention {suite} generalist macro", suite_record["generalist_macro_task_success"], statistics.fmean(generalist))
        close(f"retention {suite} specialist macro", suite_record["specialist"]["macro_task_success"], statistics.fmean(specialist))
        close(f"retention {suite} ratio", suite_record["generalist_to_specialist_ratio"], statistics.fmean(generalist) / statistics.fmean(specialist))
        for task, value in enumerate(generalist):
            close(f"retention/member seed0 {suite}:{task}", value, seed0["task_success"][f"{suite}:{task}"])
        generalist_tasks.extend(generalist)
        specialist_tasks.extend(specialist)
    generalist_macro = statistics.fmean(generalist_tasks)
    specialist_macro = statistics.fmean(specialist_tasks)
    ratio = generalist_macro / specialist_macro
    close("retention generalist macro", record.get("generalist_four_suite_task_macro"), generalist_macro)
    close("retention specialist macro", record.get("specialist_four_suite_task_macro"), specialist_macro)
    close("retention ratio", record.get("generalist_to_specialist_ratio"), ratio)
    close("retention expected generalist macro", generalist_macro, expected["generalist_four_suite_task_macro"])
    close("retention expected specialist macro", specialist_macro, expected["specialist_four_suite_task_macro"])
    close("retention expected ratio", ratio, expected["generalist_to_specialist_ratio"])
    gate = record.get("gate")
    require(isinstance(gate, dict), "retention gate is absent")
    close("retention gate threshold", gate.get("threshold"), expected["threshold"])
    equal("retention gate pass", gate.get("pass"), ratio >= expected["threshold"])
    return {"generalist": generalist_macro, "specialist": specialist_macro, "ratio": ratio}


def main() -> int:
    try:
        artifact_manifest = load_json(ARTIFACT_DIR / "manifest.json")
        records = verify_evidence_identities(artifact_manifest)
        expected = artifact_manifest.get("expected")
        require(isinstance(expected, dict), "expected-result map is absent")
        member_result = verify_members(records["ensemble_manifest"], expected["members"])
        comparison_result = verify_matched_comparison(
            records["generalist_summary"],
            member_result["members"],
            records,
            expected["matched_comparison"],
        )
        ensemble_result = verify_ensemble(records["ensemble_summary"], records["ensemble_manifest"], expected["ensemble"])
        retention_result = verify_retention(records["retention_summary"], member_result["members"], expected["retention"])
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("PASS: frozen four-suite breadth evidence verified")
    print(
        "members: "
        + ", ".join(f"{100 * value:.2f}%" for value in records["ensemble_manifest"]["member_macro_successes"])
        + f"; mean {100 * member_result['mean']:.2f}% ± {100 * member_result['sample_sd']:.2f}%"
    )
    print(
        f"matched conventional comparison: {100 * comparison_result['chi_mean']:.2f}% vs "
        f"{100 * comparison_result['conventional_mean']:.2f}%, "
        f"{100 * comparison_result['delta']:+.2f} points"
    )
    print(
        f"ensemble: {100 * ensemble_result['macro']:.2f}%, "
        f"+{100 * ensemble_result['uplift']:.2f} points, "
        f"{ensemble_result['tasks_at_least']}/40 tasks at or above 50%"
    )
    print(
        f"seed-0 retention: {100 * retention_result['generalist']:.2f}% / "
        f"{100 * retention_result['specialist']:.2f}% = {100 * retention_result['ratio']:.2f}%"
    )
    print("scope: familiar training tasks only; ensemble uses three forward passes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
