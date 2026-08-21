#!/usr/bin/env python3
"""Verify immutable structural-certificate evidence with the Python standard library."""

from __future__ import annotations

import argparse
import cmath
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ARTIFACT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = ARTIFACT_DIR.parents[1]


class VerificationError(RuntimeError):
    """Raised when evidence does not reproduce the immutable manifest."""


def load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise VerificationError(f"{path}: non-finite JSON constant {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read canonical JSON {path}: {exc}") from exc
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


def require_close(label: str, actual: Any, expected: Any) -> None:
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{label}: value is not numeric") from exc
    if not (
        math.isfinite(actual_value)
        and math.isfinite(expected_value)
        and math.isclose(actual_value, expected_value, rel_tol=1e-12, abs_tol=1e-15)
    ):
        raise VerificationError(f"{label}: {actual_value!r} != {expected_value!r}")


def safe_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    require(not candidate.is_absolute(), f"manifest path is absolute: {relative}")
    require(".." not in candidate.parts, f"manifest path escapes the root: {relative}")
    return root / candidate


def verify_manifest_hashes(root: Path, manifest: dict[str, Any]) -> dict[str, str]:
    require_equal(
        "manifest schema",
        manifest.get("schema"),
        "anonymous-structural-certificate-artifact-v1",
    )
    entries = manifest.get("files")
    require(isinstance(entries, list) and len(entries) == 8, "manifest must identify eight files")
    hashes: dict[str, str] = {}
    for entry in entries:
        require(isinstance(entry, dict), "manifest file entry is not an object")
        relative = entry.get("path")
        expected = entry.get("sha256")
        require(isinstance(relative, str), "manifest file path is absent")
        require(isinstance(expected, str) and len(expected) == 64, f"bad SHA-256 for {relative}")
        require(relative not in hashes, f"duplicate manifest path: {relative}")
        actual = file_sha256(safe_path(root, relative))
        require_equal(f"SHA-256 {relative}", actual, expected)
        hashes[relative] = actual
    return hashes


def error_value(label: str, row: dict[str, Any]) -> float:
    try:
        maximum = float(row["max_abs_error"])
        relative = float(row["relative_l2_error"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError(f"{label}: malformed error row") from exc
    numbers_finite = math.isfinite(maximum) and math.isfinite(relative)
    require_equal(f"{label} finite flag", row.get("finite"), numbers_finite)
    require(numbers_finite and maximum >= 0.0 and relative >= 0.0, f"{label}: invalid error")
    return relative


def verify_conv(
    root: Path,
    config: dict[str, Any],
    manifest_hashes: dict[str, str],
) -> dict[str, Any]:
    raw_paths = config["raw_results"]
    require_equal("Conv raw-result count", len(raw_paths), 3)
    summary = load_json(safe_path(root, config["summary"]))
    require_equal("Conv summary schema", summary.get("schema"), config["summary_schema"])
    protocol = summary.get("protocol")
    require(isinstance(protocol, dict), "Conv summary protocol is absent")
    gate = float(config["gate"]["relative_l2_error_at_most"])
    require_close("Conv protocol gate", protocol.get("relative_l2_gate"), gate)

    summary_rows = {int(row["seed"]): row for row in summary.get("checkpoints", [])}
    require_equal("Conv summary seeds", set(summary_rows), {0, 1, 2})
    recomputed = []
    for relative in raw_paths:
        result = load_json(safe_path(root, relative))
        require_equal(f"{relative} schema", result.get("schema"), config["raw_schema"])
        require_equal(f"{relative} mode", result.get("mode"), "full")
        require_equal(f"{relative} protocol", result.get("protocol"), protocol)
        seed = int(result.get("identity", {}).get("checkpoint_seed", -1))
        require(seed in {0, 1, 2}, f"{relative}: invalid seed {seed}")
        require(seed not in {row["seed"] for row in recomputed}, f"duplicate Conv seed {seed}")

        inputs = result.get("inputs")
        modules = result.get("module_rows")
        require(isinstance(inputs, list), f"{relative}: inputs are absent")
        require(isinstance(modules, list), f"{relative}: module rows are absent")
        expected_modules = len(inputs) * int(protocol["joint_modules_per_input"])
        require_equal(f"{relative} module count", len(modules), expected_modules)
        module_errors = []
        head_errors = []
        for module_index, module in enumerate(modules):
            require(isinstance(module, dict), f"{relative}: malformed module {module_index}")
            module_errors.append(error_value(f"{relative} module {module_index}", module))
            heads = module.get("heads")
            require(isinstance(heads, list), f"{relative}: heads are absent at module {module_index}")
            require_equal(
                f"{relative} head count at module {module_index}",
                len(heads),
                int(protocol["heads_per_module"]),
            )
            for head_index, head in enumerate(heads):
                require(isinstance(head, dict), f"{relative}: malformed head row")
                require_equal(
                    f"{relative} head index at module {module_index}",
                    int(head.get("head_index", -1)),
                    head_index,
                )
                head_errors.append(
                    error_value(f"{relative} module {module_index} head {head_index}", head)
                )

        row = {
            "seed": seed,
            "inputs_audited": len(inputs),
            "module_evaluations": len(modules),
            "head_evaluations": len(head_errors),
            "all_finite": True,
            "max_module_relative_l2_error": max(module_errors),
            "max_head_relative_l2_error": max(head_errors),
        }
        row["max_relative_l2_error"] = max(
            row["max_module_relative_l2_error"], row["max_head_relative_l2_error"]
        )
        row["passed"] = row["all_finite"] and row["max_relative_l2_error"] <= gate
        aggregate = result.get("aggregate", {})
        for field in (
            "inputs_audited",
            "module_evaluations",
            "head_evaluations",
            "all_finite",
            "passed",
        ):
            require_equal(f"{relative} aggregate {field}", aggregate.get(field), row[field])
        for field in (
            "max_module_relative_l2_error",
            "max_head_relative_l2_error",
            "max_relative_l2_error",
        ):
            require_close(f"{relative} aggregate {field}", aggregate.get(field), row[field])
        require_close(f"{relative} aggregate gate", aggregate.get("relative_l2_gate"), gate)

        recorded = summary_rows[seed]
        require_equal(f"Conv summary seed {seed} file hash", recorded.get("source_file_sha256"), manifest_hashes[relative])
        require_equal(f"Conv summary seed {seed} checkpoint SHA", recorded.get("checkpoint_sha256"), result["identity"]["checkpoint_sha256"])
        for field in (
            "inputs_audited",
            "module_evaluations",
            "head_evaluations",
            "all_finite",
            "passed",
        ):
            require_equal(f"Conv summary seed {seed} {field}", recorded.get(field), row[field])
        for field in (
            "max_module_relative_l2_error",
            "max_head_relative_l2_error",
            "max_relative_l2_error",
        ):
            require_close(f"Conv summary seed {seed} {field}", recorded.get(field), row[field])
        recomputed.append(row)

    aggregate = {
        "checkpoints_audited": 3,
        "inputs_audited": sum(row["inputs_audited"] for row in recomputed),
        "module_evaluations": sum(row["module_evaluations"] for row in recomputed),
        "head_evaluations": sum(row["head_evaluations"] for row in recomputed),
        "all_finite": all(row["all_finite"] for row in recomputed),
        "max_relative_l2_error": max(row["max_relative_l2_error"] for row in recomputed),
        "relative_l2_gate": gate,
        "all_three_checkpoints_pass": all(row["passed"] for row in recomputed),
    }
    recorded_aggregate = summary.get("aggregate", {})
    for field, expected in aggregate.items():
        if isinstance(expected, float):
            require_close(f"Conv summary aggregate {field}", recorded_aggregate.get(field), expected)
        else:
            require_equal(f"Conv summary aggregate {field}", recorded_aggregate.get(field), expected)
    require_equal(
        "Conv immutable expected outcome",
        aggregate["all_three_checkpoints_pass"],
        config["expected_outcome"],
    )
    return aggregate


def quadratic_roots(coefficients: list[float]) -> list[complex]:
    require_equal("denominator coefficient count", len(coefficients), 3)
    c, b, a = coefficients
    require(a != 0.0, "denominator leading coefficient is zero")
    discriminant = cmath.sqrt(complex(b * b - 4.0 * a * c, 0.0))
    return [(-b - discriminant) / (2.0 * a), (-b + discriminant) / (2.0 * a)]


def verify_rational_site(
    label: str,
    site: dict[str, Any],
    gates: dict[str, float],
) -> dict[str, Any]:
    require_equal(f"{label} variant", site.get("variant"), "pade")
    require_equal(f"{label} degree", int(site.get("degree", -1)), 2)
    require(site.get("initialized") is True, f"{label}: site was not initialized")
    require(site.get("evaluator_frozen_flag") is True, f"{label}: evaluator was not frozen")
    rows = int(site.get("normalization_rows", -1))
    in_range = int(site.get("rows_v_in_0p1_10", -1))
    local_rows = int(site.get("rows_local_scale_error_at_or_below_threshold", -1))
    nonfinite = int(site.get("nonfinite_rows", -1))
    require(rows > 0 and 0 <= in_range <= rows, f"{label}: invalid range counts")
    require(0 <= local_rows <= rows, f"{label}: invalid local-error counts")
    require_equal(f"{label} nonfinite rows", nonfinite, 0)
    range_fraction = in_range / rows
    local_fraction = local_rows / rows
    require_close(f"{label} range fraction", site.get("fraction_rows_v_in_0p1_10"), range_fraction)
    require_close(
        f"{label} local-error fraction",
        site.get("fraction_rows_local_scale_error_at_or_below_threshold"),
        local_fraction,
    )
    require_close(
        f"{label} local-error threshold",
        site.get("local_scale_relative_error_threshold"),
        gates["secondary_local_scale_relative_error_threshold"],
    )

    certificate = site.get("denominator_certificate", {})
    raw_coefficients = certificate.get("denominator_coefficients_float64")
    require(isinstance(raw_coefficients, list), f"{label}: denominator coefficients are absent")
    coefficients = [float(value) for value in raw_coefficients]
    positive_coefficients = all(math.isfinite(value) and value > 0.0 for value in coefficients)
    roots = quadratic_roots(coefficients)
    positive_real_roots = sum(abs(root.imag) <= 1e-12 and root.real >= 0.0 for root in roots)
    recorded_roots = certificate.get("denominator_roots_float64")
    require(isinstance(recorded_roots, list) and len(recorded_roots) == 2, f"{label}: roots are absent")
    actual_roots = sorted(roots, key=lambda value: (value.real, value.imag))
    expected_roots = sorted(
        [complex(float(row["real"]), float(row["imag"])) for row in recorded_roots],
        key=lambda value: (value.real, value.imag),
    )
    for index, (actual, expected) in enumerate(zip(actual_roots, expected_roots)):
        require_close(f"{label} root {index} real", actual.real, expected.real)
        require_close(f"{label} root {index} imag", actual.imag, expected.imag)
    certificate_pass = positive_coefficients and positive_real_roots == 0
    require_equal(f"{label} positive-coefficient flag", certificate.get("coefficients_strictly_positive"), positive_coefficients)
    require_equal(f"{label} positive-real-root count", certificate.get("positive_real_root_count"), positive_real_roots)
    require_equal(f"{label} certificate pass", certificate.get("certificate_pass"), certificate_pass)
    denominator_min = float(site.get("observed_denominator_min", float("nan")))
    require(math.isfinite(denominator_min) and denominator_min > 0.0, f"{label}: observed denominator is not positive")
    primary = certificate_pass and denominator_min > 0.0 and nonfinite == 0
    range_pass = range_fraction >= gates["secondary_each_site_fraction_rows_v_in_0p1_10_at_least"]
    local_pass = local_fraction >= gates["secondary_each_site_fraction_local_scale_error_at_or_below_threshold_at_least"]
    require_equal(f"{label} primary denominator gate", site.get("primary_denominator_pass"), primary)
    require_equal(f"{label} secondary range gate", site.get("secondary_range_pass"), range_pass)
    require_equal(f"{label} secondary local gate", site.get("secondary_local_error_pass"), local_pass)
    return {
        "rows": rows,
        "in_range": in_range,
        "local_rows": local_rows,
        "range_fraction": range_fraction,
        "local_fraction": local_fraction,
        "denominator_min": denominator_min,
        "v_min": float(site["observed_v_min"]),
        "v_max": float(site["observed_v_max"]),
        "p99": float(site["local_scale_relative_error_p99"]),
        "maximum": float(site["local_scale_relative_error_max"]),
        "primary": primary,
        "range_pass": range_pass,
        "local_pass": local_pass,
    }


def verify_rational(
    root: Path,
    config: dict[str, Any],
    manifest_hashes: dict[str, str],
) -> dict[str, Any]:
    raw_paths = config["raw_results"]
    require_equal("RationalNorm raw-result count", len(raw_paths), 3)
    summary = load_json(safe_path(root, config["summary"]))
    require_equal("RationalNorm summary schema", summary.get("schema"), config["summary_schema"])
    gates = config["gates"]
    for field, expected in gates.items():
        require_close(f"RationalNorm summary frozen gate {field}", summary.get("frozen_gates", {}).get(field), expected)
    summary_rows = {int(row["seed"]): row for row in summary.get("checkpoints", [])}
    require_equal("RationalNorm summary seeds", set(summary_rows), {0, 1, 2})

    recomputed = []
    for relative in raw_paths:
        result = load_json(safe_path(root, relative))
        require_equal(f"{relative} schema", result.get("schema"), config["raw_schema"])
        require_equal(f"{relative} mode", result.get("mode"), "full")
        for field, expected in gates.items():
            require_close(f"{relative} frozen gate {field}", result.get("frozen_gates", {}).get(field), expected)
        seed = int(result.get("checkpoint", {}).get("seed", -1))
        require(seed in {0, 1, 2}, f"{relative}: invalid seed {seed}")
        require(seed not in {row["seed"] for row in recomputed}, f"duplicate RationalNorm seed {seed}")
        sites = result.get("rational_sites")
        require(isinstance(sites, list) and len(sites) == 53, f"{relative}: expected 53 sites")
        site_rows = [
            verify_rational_site(f"{relative} site {index}", site, gates)
            for index, site in enumerate(sites)
        ]
        rows = sum(site["rows"] for site in site_rows)
        in_range = sum(site["in_range"] for site in site_rows)
        local_rows = sum(site["local_rows"] for site in site_rows)
        aggregate = result.get("aggregate", {})
        require_equal(f"{relative} aggregate row count", aggregate.get("normalization_rows"), rows)
        require_equal(f"{relative} aggregate in-range count", aggregate.get("rows_v_in_0p1_10"), in_range)
        require_equal(f"{relative} aggregate local-error count", aggregate.get("rows_local_scale_error_at_or_below_threshold"), local_rows)
        require_close(f"{relative} aggregate range fraction", aggregate.get("fraction_rows_v_in_0p1_10"), in_range / rows)
        require_close(f"{relative} aggregate local fraction", aggregate.get("fraction_rows_local_scale_error_at_or_below_threshold"), local_rows / rows)

        action = result.get("action_reference", {})
        physical_diff = float(action["physical_squared_difference_sum"])
        physical_reference = float(action["physical_squared_reference_sum"])
        normalized_diff = float(action["normalized_squared_difference_sum"])
        normalized_reference = float(action["normalized_squared_reference_sum"])
        require(physical_diff >= 0.0 and physical_reference > 0.0, f"{relative}: invalid physical action sums")
        require(normalized_diff >= 0.0 and normalized_reference > 0.0, f"{relative}: invalid normalized action sums")
        physical_nrmse = math.sqrt(physical_diff / physical_reference)
        normalized_nrmse = math.sqrt(normalized_diff / normalized_reference)
        action_nrmse = max(physical_nrmse, normalized_nrmse)
        require_close(f"{relative} physical NRMSE", action.get("physical_nrmse"), physical_nrmse)
        require_close(f"{relative} normalized NRMSE", action.get("normalized_nrmse"), normalized_nrmse)
        require_close(f"{relative} primary NRMSE", action.get("primary_nrmse"), action_nrmse)
        primary_denominator = all(site["primary"] for site in site_rows)
        primary_action = action_nrmse <= gates["primary_action_nrmse_at_most"]
        secondary_range = all(site["range_pass"] for site in site_rows)
        secondary_local = all(site["local_pass"] for site in site_rows)
        outcomes = {
            "primary_action_nrmse_pass": primary_action,
            "primary_denominator_certificate_pass": primary_denominator,
            "primary_pass": primary_denominator and primary_action,
            "secondary_local_scale_error_pass": secondary_local,
            "secondary_pass": secondary_range and secondary_local,
            "secondary_range_pass": secondary_range,
        }
        require_equal(f"{relative} gates", result.get("gates"), outcomes)

        row = {
            "seed": seed,
            "normalization_rows": rows,
            "fraction_rows_v_in_0p1_10": in_range / rows,
            "minimum_site_fraction_rows_v_in_0p1_10": min(site["range_fraction"] for site in site_rows),
            "fraction_rows_local_scale_error_at_or_below_threshold": local_rows / rows,
            "minimum_site_fraction_rows_local_scale_error_at_or_below_threshold": min(site["local_fraction"] for site in site_rows),
            "observed_v_min": min(site["v_min"] for site in site_rows),
            "observed_v_max": max(site["v_max"] for site in site_rows),
            "observed_denominator_min": min(site["denominator_min"] for site in site_rows),
            "local_scale_relative_error_p99": float(aggregate["local_scale_relative_error_p99"]),
            "local_scale_relative_error_max": max(site["maximum"] for site in site_rows),
            "action_nrmse": action_nrmse,
            "physical_action_nrmse": physical_nrmse,
            "normalized_action_nrmse": normalized_nrmse,
            "physical_action_squared_difference_sum": physical_diff,
            "physical_action_squared_reference_sum": physical_reference,
            "normalized_action_squared_difference_sum": normalized_diff,
            "normalized_action_squared_reference_sum": normalized_reference,
            "gates": outcomes,
        }
        recorded = summary_rows[seed]
        require_equal(f"RationalNorm summary seed {seed} checkpoint", recorded.get("checkpoint"), result.get("checkpoint"))
        require_equal(f"RationalNorm summary seed {seed} model parameters", recorded.get("model_parameters"), result.get("model", {}).get("parameters"))
        for field in (
            "normalization_rows",
            "gates",
        ):
            require_equal(f"RationalNorm summary seed {seed} {field}", recorded.get(field), row[field])
        for field in (
            "fraction_rows_v_in_0p1_10",
            "minimum_site_fraction_rows_v_in_0p1_10",
            "fraction_rows_local_scale_error_at_or_below_threshold",
            "minimum_site_fraction_rows_local_scale_error_at_or_below_threshold",
            "observed_v_min",
            "observed_v_max",
            "observed_denominator_min",
            "local_scale_relative_error_p99",
            "local_scale_relative_error_max",
            "action_nrmse",
            "physical_action_nrmse",
            "normalized_action_nrmse",
            "physical_action_squared_difference_sum",
            "physical_action_squared_reference_sum",
            "normalized_action_squared_difference_sum",
            "normalized_action_squared_reference_sum",
        ):
            require_close(f"RationalNorm summary seed {seed} {field}", recorded.get(field), row[field])
        summary_hashes = summary.get("inputs", {}).get("result_sha256", {})
        matching_hashes = [value for path, value in summary_hashes.items() if Path(path).name == Path(relative).name]
        require_equal(f"RationalNorm summary seed {seed} source hash count", len(matching_hashes), 1)
        require_equal(f"RationalNorm summary seed {seed} source hash", matching_hashes[0], manifest_hashes[relative])
        recomputed.append(row)

    physical_diff = sum(row["physical_action_squared_difference_sum"] for row in recomputed)
    physical_reference = sum(row["physical_action_squared_reference_sum"] for row in recomputed)
    normalized_diff = sum(row["normalized_action_squared_difference_sum"] for row in recomputed)
    normalized_reference = sum(row["normalized_action_squared_reference_sum"] for row in recomputed)
    aggregate = {
        "checkpoints": 3,
        "selected_samples": 3 * int(summary["protocol"]["full_samples_per_checkpoint"]),
        "rational_sites_per_checkpoint": 53,
        "normalization_rows": sum(row["normalization_rows"] for row in recomputed),
        "minimum_fraction_rows_v_in_0p1_10": min(row["minimum_site_fraction_rows_v_in_0p1_10"] for row in recomputed),
        "minimum_fraction_rows_local_scale_error_at_or_below_threshold": min(row["minimum_site_fraction_rows_local_scale_error_at_or_below_threshold"] for row in recomputed),
        "maximum_local_scale_relative_error_p99": max(row["local_scale_relative_error_p99"] for row in recomputed),
        "minimum_observed_denominator": min(row["observed_denominator_min"] for row in recomputed),
        "observed_v_min": min(row["observed_v_min"] for row in recomputed),
        "observed_v_max": max(row["observed_v_max"] for row in recomputed),
        "pooled_physical_action_nrmse": math.sqrt(physical_diff / physical_reference),
        "pooled_normalized_action_nrmse": math.sqrt(normalized_diff / normalized_reference),
        "maximum_checkpoint_action_nrmse": max(row["action_nrmse"] for row in recomputed),
    }
    recorded_aggregate = summary.get("aggregate", {})
    for field, expected in aggregate.items():
        if isinstance(expected, float):
            require_close(f"RationalNorm summary aggregate {field}", recorded_aggregate.get(field), expected)
        else:
            require_equal(f"RationalNorm summary aggregate {field}", recorded_aggregate.get(field), expected)
    outcomes = {
        "all_three_primary_pass": all(row["gates"]["primary_pass"] for row in recomputed),
        "all_three_secondary_pass": all(row["gates"]["secondary_pass"] for row in recomputed),
    }
    require_equal("RationalNorm summary gates", summary.get("gates"), outcomes)
    require_equal("RationalNorm immutable primary outcome", outcomes["all_three_primary_pass"], config["expected_primary_outcome"])
    require_equal("RationalNorm immutable secondary outcome", outcomes["all_three_secondary_pass"], config["expected_secondary_outcome"])
    return {**aggregate, **outcomes}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Repository root containing athena/results. Defaults to the verifier's repository.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_json(ARTIFACT_DIR / "manifest.json")
        hashes = verify_manifest_hashes(args.root.resolve(), manifest)
        conv = verify_conv(args.root.resolve(), manifest["certificates"]["conv_joint_attention"], hashes)
        rational = verify_rational(args.root.resolve(), manifest["certificates"]["rational_norm"], hashes)
    except (KeyError, TypeError, ValueError, VerificationError) as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"HASHES PASS: {len(hashes)} immutable JSON files")
    print(
        "CONV JOINT ATTENTION PASS: "
        f"{conv['head_evaluations']} head-input cases, "
        f"max relative L2 {conv['max_relative_l2_error']:.12g} <= "
        f"{conv['relative_l2_gate']:.12g}"
    )
    print(
        "RATIONALNORM PRIMARY PASS: "
        f"{rational['normalization_rows']} rows, "
        f"max checkpoint action NRMSE {rational['maximum_checkpoint_action_nrmse']:.12g} <= 0.001"
    )
    print(
        "RATIONALNORM SECONDARY FAIL (EXPECTED): "
        "the preregistered approximation-coverage gate is not claimed"
    )
    print("VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
