#!/usr/bin/env python3
"""Strictly validate the frozen three-checkpoint RationalNorm certificate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from athena.run_rational_norm_safety import (
    FROZEN_CACHE,
    FROZEN_CHECKPOINTS,
    FROZEN_DATASET_METADATA_SHA256,
    FROZEN_DATASET_TO_OFFICIAL_TASK,
    FROZEN_GATES,
    FROZEN_PROVENANCE_JOB_ID,
    PROTOCOL,
    SCHEMA,
    expected_rational_site_names,
    file_sha256,
    imported_source_hashes,
    load_cache_and_statistics,
    select_cache_samples,
    update_array_digest,
    validate_provenance,
    write_json,
)


SUMMARY_SCHEMA = "xvla-rational-norm-safety-summary-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--provenance-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def require_close(label: str, actual: float, expected: float, atol: float = 1e-12) -> None:
    if not math.isfinite(float(actual)) or not math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=atol
    ):
        raise RuntimeError(f"{label} does not reproduce: {actual} != {expected}")


def decode_float32_array(encoded: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    if not isinstance(encoded, str):
        raise RuntimeError(f"{label} encoding is absent")
    try:
        payload = base64.b64decode(encoded.encode("ascii"), validate=True)
    except Exception as exc:
        raise RuntimeError(f"{label} is not canonical base64") from exc
    expected_bytes = int(np.prod(shape)) * np.dtype("<f4").itemsize
    if len(payload) != expected_bytes:
        raise RuntimeError(f"{label} byte count differs")
    if base64.b64encode(payload).decode("ascii") != encoded:
        raise RuntimeError(f"{label} base64 representation is not canonical")
    return np.frombuffer(payload, dtype="<f4").copy().reshape(shape)


def validate_root_certificate(site: dict[str, Any]) -> None:
    certificate = site["denominator_certificate"]
    pa = np.asarray(certificate["numerator_coefficients_float64"], dtype=np.float64)
    pb = np.asarray(certificate["denominator_coefficients_float64"], dtype=np.float64)
    if pa.shape != (3,) or pb.shape != (3,) or not np.isfinite(pa).all():
        raise RuntimeError("Rational coefficient catalog is malformed")
    if not np.isfinite(pb).all() or not np.all(pb > 0.0):
        raise RuntimeError("Denominator coefficients are not strictly positive")
    roots = np.roots(pb[::-1])
    recorded_roots = np.asarray(
        [complex(float(row["real"]), float(row["imag"])) for row in certificate["denominator_roots_float64"]]
    )
    if roots.shape != recorded_roots.shape:
        raise RuntimeError("Denominator root count differs")
    expected_sorted = sorted(roots, key=lambda value: (float(value.real), float(value.imag)))
    recorded_sorted = sorted(
        recorded_roots, key=lambda value: (float(value.real), float(value.imag))
    )
    if not np.allclose(expected_sorted, recorded_sorted, rtol=0.0, atol=1e-12):
        raise RuntimeError("Denominator roots do not reproduce from coefficients")
    positive_real_count = sum(
        abs(float(root.imag)) <= 1e-12 and float(root.real) >= 0.0 for root in roots
    )
    if int(certificate["positive_real_root_count"]) != positive_real_count:
        raise RuntimeError("Positive-real-root count does not reproduce")
    expected_pass = bool(np.all(pb > 0.0) and positive_real_count == 0)
    if bool(certificate["coefficients_strictly_positive"]) != bool(np.all(pb > 0.0)):
        raise RuntimeError("Positive-coefficient certificate flag differs")
    if bool(certificate["certificate_pass"]) != expected_pass or not expected_pass:
        raise RuntimeError("Denominator positive-half-line certificate failed")


def validate_site(
    site: dict[str, Any],
    expected_name: str,
    batch_count: int,
) -> None:
    if site.get("name") != expected_name:
        raise RuntimeError("Rational site order or identity differs")
    if site.get("variant") != "pade" or int(site.get("degree", -1)) != 2:
        raise RuntimeError(f"Rational configuration differs at {expected_name}")
    if site.get("initialized") is not True or site.get("evaluator_frozen_flag") is not True:
        raise RuntimeError(
            f"Rational site is not initialized and evaluator-frozen at {expected_name}"
        )
    if not math.isfinite(float(site.get("running_ms", float("nan")))) or float(
        site["running_ms"]
    ) <= 0.0:
        raise RuntimeError(f"Running mean-square is invalid at {expected_name}")
    if int(site.get("invocations", -1)) != batch_count:
        raise RuntimeError(f"Invocation count differs at {expected_name}")
    rows = int(site.get("normalization_rows", -1))
    in_range = int(site.get("rows_v_in_0p1_10", -1))
    if rows <= 0 or not 0 <= in_range <= rows:
        raise RuntimeError(f"Row counts are invalid at {expected_name}")
    if int(site.get("nonfinite_rows", -1)) != 0:
        raise RuntimeError(f"Nonfinite rational rows occurred at {expected_name}")
    local_error_threshold = float(
        site.get("local_scale_relative_error_threshold", float("nan"))
    )
    require_close(
        f"local-error threshold at {expected_name}",
        local_error_threshold,
        FROZEN_GATES["secondary_local_scale_relative_error_threshold"],
    )
    local_error_at_or_below = int(
        site.get("rows_local_scale_error_at_or_below_threshold", -1)
    )
    if not 0 <= local_error_at_or_below <= rows:
        raise RuntimeError(f"Local-error threshold count is invalid at {expected_name}")
    require_close(
        f"range fraction at {expected_name}",
        site["fraction_rows_v_in_0p1_10"],
        in_range / rows,
    )
    require_close(
        f"local-error threshold fraction at {expected_name}",
        site["fraction_rows_local_scale_error_at_or_below_threshold"],
        local_error_at_or_below / rows,
    )
    finite_nonnegative = (
        "observed_v_min",
        "observed_v_max",
        "observed_denominator_min",
        "observed_denominator_max",
        "local_scale_relative_error_p50",
        "local_scale_relative_error_p95",
        "local_scale_relative_error_p99",
        "local_scale_relative_error_max",
        "fp32_vs_fp64_scale_relative_error_rms",
        "fp32_vs_fp64_scale_relative_error_max",
    )
    for field in finite_nonnegative:
        value = float(site[field])
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"Invalid {field} at {expected_name}")
    if float(site["observed_v_min"]) > float(site["observed_v_max"]):
        raise RuntimeError(f"Observed v range is reversed at {expected_name}")
    if float(site["observed_denominator_min"]) <= 0.0:
        raise RuntimeError(f"Observed denominator is not positive at {expected_name}")
    if float(site["observed_denominator_min"]) > float(site["observed_denominator_max"]):
        raise RuntimeError(f"Observed denominator range is reversed at {expected_name}")
    if not (
        float(site["local_scale_relative_error_p50"])
        <= float(site["local_scale_relative_error_p95"])
        <= float(site["local_scale_relative_error_p99"])
        <= float(site["local_scale_relative_error_max"])
    ):
        raise RuntimeError(f"Local-error quantiles are not monotone at {expected_name}")
    validate_root_certificate(site)
    denominator_pass = (
        bool(site["denominator_certificate"]["certificate_pass"])
        and float(site["observed_denominator_min"]) > 0.0
        and int(site["nonfinite_rows"]) == 0
    )
    if bool(site["primary_denominator_pass"]) != denominator_pass:
        raise RuntimeError(f"Denominator gate differs at {expected_name}")
    if bool(site["secondary_range_pass"]) != (
        float(site["fraction_rows_v_in_0p1_10"])
        >= FROZEN_GATES[
            "secondary_each_site_fraction_rows_v_in_0p1_10_at_least"
        ]
    ):
        raise RuntimeError(f"Per-site range gate differs at {expected_name}")
    if bool(site["secondary_local_error_pass"]) != (
        float(site["fraction_rows_local_scale_error_at_or_below_threshold"])
        >= FROZEN_GATES[
            "secondary_each_site_fraction_local_scale_error_at_or_below_threshold_at_least"
        ]
    ):
        raise RuntimeError(f"Per-site local-error gate differs at {expected_name}")


def validate_result(
    result: dict[str, Any],
    expected_seed: int,
    frames: list[Any] | tuple[Any, ...],
    stats: dict[str, Any],
    live_source_hashes: dict[str, str],
    live_provenance_sha256: str,
) -> dict[str, Any]:
    expected_checkpoint = FROZEN_CHECKPOINTS[expected_seed]
    if result.get("schema") != SCHEMA or result.get("mode") != "full":
        raise RuntimeError("Result schema or mode differs from the frozen full protocol")
    if result.get("protocol") != PROTOCOL or result.get("frozen_gates") != FROZEN_GATES:
        raise RuntimeError("Protocol or gates differ from their frozen definitions")
    checkpoint = result["checkpoint"]
    if int(checkpoint.get("seed", -1)) != expected_seed:
        raise RuntimeError("Checkpoint seed differs")
    if checkpoint.get("basename") != expected_checkpoint["basename"]:
        raise RuntimeError("Checkpoint basename differs")
    if checkpoint.get("sha256") != expected_checkpoint["sha256"]:
        raise RuntimeError("Checkpoint SHA-256 differs")
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != expected_checkpoint[
        "sha256"
    ]:
        raise RuntimeError("Live checkpoint is absent or differs from its frozen SHA-256")
    cache = result["cache"]
    if cache.get("sha256") != FROZEN_CACHE["sha256"] or int(
        cache.get("frames", -1)
    ) != FROZEN_CACHE["frames"]:
        raise RuntimeError("Frozen cache identity differs")
    if cache.get("expected_provenance_job_id") != FROZEN_PROVENANCE_JOB_ID:
        raise RuntimeError("Provenance job identity differs")
    if cache.get("provenance_result_sha256") != live_provenance_sha256:
        raise RuntimeError("Provenance result file changed after the certificate run")
    if cache.get("content_hashes_match") is not True:
        raise RuntimeError("Cache/source canonical content hashes do not match")
    if cache.get("canonical_content_sha256") != cache.get("source_content_sha256"):
        raise RuntimeError("Recorded cache/source canonical digests differ")
    if int(cache.get("normalization_samples", -1)) != int(stats["sample_count"]):
        raise RuntimeError("Cache normalization sample count differs")
    expected_task_metadata = {
        "repository": "lerobot/libero_object_image",
        "revision": "e1e080d7df1d0a359dff5c86c222e047549f447f",
        "metadata_file": "meta/tasks.parquet",
        "metadata_sha256": FROZEN_DATASET_METADATA_SHA256,
        "dataset_to_official_task": FROZEN_DATASET_TO_OFFICIAL_TASK,
        "ordering_matches_official": False,
        "language_set_matches_official": True,
    }
    if cache.get("cache_task_metadata") != expected_task_metadata:
        raise RuntimeError("Recorded Object task metadata differs from the frozen identity")
    if result.get("implementation", {}).get("source_sha256") != live_source_hashes:
        raise RuntimeError("Runner or imported source changed after certificate execution")

    expected_selection = select_cache_samples(
        frames,
        expected_checkpoint["sha256"],
        PROTOCOL["full_samples_per_checkpoint"],
    )
    if result.get("selection") != expected_selection:
        raise RuntimeError("SHA-256-ranked sample selection does not reproduce")
    if len({row["cache_offset"] for row in expected_selection["records"]}) != 4096:
        raise RuntimeError("Selected cache offsets are not unique")

    model = result["model"]
    expected_sites = expected_rational_site_names()
    if model.get("architecture") != "chi" or model.get("vision_encoder") != "conv":
        raise RuntimeError("Model family differs from the frozen certificate")
    if int(model.get("rational_site_count", -1)) != 53:
        raise RuntimeError("Rational site count is not 53")
    if model.get("rational_site_names") != expected_sites:
        raise RuntimeError("Rational site catalog differs")
    if int(model.get("parameters", -1)) <= 0:
        raise RuntimeError("Model parameter count is invalid")
    sites = result.get("rational_sites", [])
    if len(sites) != 53:
        raise RuntimeError("Rational site results are incomplete")
    aggregate = result["aggregate"]
    if int(aggregate.get("sample_count", -1)) != 4096:
        raise RuntimeError("Full result does not contain 4,096 selected samples")
    if int(aggregate.get("batch_size", -1)) != 32:
        raise RuntimeError("Batch size differs from the frozen value")
    expected_batches = 128
    if int(aggregate.get("batch_count", -1)) != expected_batches:
        raise RuntimeError("Batch count differs from the frozen value")
    for site, expected_name in zip(sites, expected_sites):
        validate_site(site, expected_name, expected_batches)

    total_rows = sum(int(site["normalization_rows"]) for site in sites)
    total_in_range = sum(int(site["rows_v_in_0p1_10"]) for site in sites)
    total_local_error_at_or_below_threshold = sum(
        int(site["rows_local_scale_error_at_or_below_threshold"])
        for site in sites
    )
    total_nonfinite = sum(int(site["nonfinite_rows"]) for site in sites)
    if int(aggregate.get("normalization_rows", -1)) != total_rows:
        raise RuntimeError("Aggregate rational row count does not reproduce")
    if int(aggregate.get("rows_v_in_0p1_10", -1)) != total_in_range:
        raise RuntimeError("Aggregate in-range row count does not reproduce")
    if int(aggregate.get("nonfinite_rows", -1)) != total_nonfinite or total_nonfinite != 0:
        raise RuntimeError("Aggregate nonfinite row count differs")
    require_close(
        "aggregate range fraction",
        aggregate["fraction_rows_v_in_0p1_10"],
        total_in_range / total_rows,
    )
    if int(
        aggregate.get("rows_local_scale_error_at_or_below_threshold", -1)
    ) != total_local_error_at_or_below_threshold:
        raise RuntimeError("Aggregate local-error threshold count does not reproduce")
    require_close(
        "aggregate local-error threshold fraction",
        aggregate["fraction_rows_local_scale_error_at_or_below_threshold"],
        total_local_error_at_or_below_threshold / total_rows,
    )
    require_close(
        "aggregate observed v minimum",
        aggregate["observed_v_min"],
        min(float(site["observed_v_min"]) for site in sites),
    )
    require_close(
        "aggregate observed v maximum",
        aggregate["observed_v_max"],
        max(float(site["observed_v_max"]) for site in sites),
    )
    require_close(
        "aggregate denominator minimum",
        aggregate["observed_denominator_min"],
        min(float(site["observed_denominator_min"]) for site in sites),
    )
    aggregate_p99 = float(aggregate["local_scale_relative_error_p99"])
    aggregate_max = float(aggregate["local_scale_relative_error_max"])
    if not math.isfinite(aggregate_p99) or not 0.0 <= aggregate_p99 <= aggregate_max:
        raise RuntimeError("Aggregate local-error statistics are invalid")
    if aggregate_max != max(float(site["local_scale_relative_error_max"]) for site in sites):
        raise RuntimeError("Aggregate local-error maximum does not reproduce")
    if bool(aggregate["all_sites_secondary_range_pass"]) != all(
        bool(site["secondary_range_pass"]) for site in sites
    ):
        raise RuntimeError("All-site range gate differs")
    if bool(aggregate["all_sites_secondary_local_error_pass"]) != all(
        bool(site["secondary_local_error_pass"]) for site in sites
    ):
        raise RuntimeError("All-site local-error gate differs")

    action = result["action_reference"]
    expected_action_shape = (PROTOCOL["full_samples_per_checkpoint"], 8, 7)
    if action.get("normalized_action_array_shape") != list(expected_action_shape):
        raise RuntimeError("Stored normalized action shape declaration differs")
    if action.get("normalized_action_array_dtype") != "<f4":
        raise RuntimeError("Stored normalized action dtype differs")
    if action.get("normalized_action_array_encoding") != "base64 contiguous C-order bytes":
        raise RuntimeError("Stored normalized action encoding differs")
    baseline_normalized32 = decode_float32_array(
        action.get("baseline_normalized_actions_float32_base64"),
        expected_action_shape,
        "baseline normalized actions",
    )
    reference_normalized32 = decode_float32_array(
        action.get("fp64_rational_reference_normalized_actions_float32_base64"),
        expected_action_shape,
        "reference normalized actions",
    )
    if not (
        np.isfinite(baseline_normalized32).all()
        and np.isfinite(reference_normalized32).all()
    ):
        raise RuntimeError("Stored normalized action array contains nonfinite values")
    baseline_normalized = baseline_normalized32.astype(np.float64)
    reference_normalized = reference_normalized32.astype(np.float64)
    normalized_difference = baseline_normalized - reference_normalized
    normalized_diff_sumsq = float(
        np.sum(normalized_difference * normalized_difference)
    )
    normalized_reference_sumsq = float(
        np.sum(reference_normalized * reference_normalized)
    )
    action_mean = np.asarray(stats["action_mean"], dtype=np.float64)
    action_std = np.asarray(stats["action_std"], dtype=np.float64)
    baseline_physical = baseline_normalized * action_std + action_mean
    reference_physical = reference_normalized * action_std + action_mean
    physical_difference = baseline_physical - reference_physical
    physical_diff_sumsq = float(np.sum(physical_difference * physical_difference))
    physical_reference_sumsq = float(
        np.sum(reference_physical * reference_physical)
    )
    if physical_reference_sumsq <= 0.0 or normalized_reference_sumsq <= 0.0:
        raise RuntimeError("Stored action reference energy is not positive")
    physical_nrmse = math.sqrt(physical_diff_sumsq / physical_reference_sumsq)
    normalized_nrmse = math.sqrt(
        normalized_diff_sumsq / normalized_reference_sumsq
    )
    expected_nrmse = max(physical_nrmse, normalized_nrmse)
    require_close(
        "physical squared difference sum",
        action["physical_squared_difference_sum"],
        physical_diff_sumsq,
    )
    require_close(
        "physical squared reference sum",
        action["physical_squared_reference_sum"],
        physical_reference_sumsq,
    )
    require_close(
        "normalized squared difference sum",
        action["normalized_squared_difference_sum"],
        normalized_diff_sumsq,
    )
    require_close(
        "normalized squared reference sum",
        action["normalized_squared_reference_sum"],
        normalized_reference_sumsq,
    )
    require_close("physical action NRMSE", action["physical_nrmse"], physical_nrmse)
    require_close(
        "normalized action NRMSE", action["normalized_nrmse"], normalized_nrmse
    )
    require_close("primary action NRMSE", action["primary_nrmse"], expected_nrmse)
    if int(action.get("physical_action_values", -1)) != int(
        np.prod(expected_action_shape)
    ):
        raise RuntimeError("Physical action value count differs")
    baseline_digest = hashlib.sha256()
    reference_digest = hashlib.sha256()
    update_array_digest(baseline_digest, baseline_physical)
    update_array_digest(reference_digest, reference_physical)
    if action.get("baseline_physical_actions_sha256") != baseline_digest.hexdigest():
        raise RuntimeError("Baseline physical-action digest does not reproduce")
    if action.get(
        "fp64_rational_reference_physical_actions_sha256"
    ) != reference_digest.hexdigest():
        raise RuntimeError("Reference physical-action digest does not reproduce")
    action_max_absolute_difference = float(
        np.max(np.abs(physical_difference), initial=0.0)
    )
    require_close(
        "action maximum absolute difference",
        action["max_absolute_difference"],
        action_max_absolute_difference,
    )

    expected_primary_denominator = all(
        bool(site["primary_denominator_pass"]) for site in sites
    )
    expected_primary_action = expected_nrmse <= FROZEN_GATES[
        "primary_action_nrmse_at_most"
    ]
    expected_secondary_range = all(bool(site["secondary_range_pass"]) for site in sites)
    expected_secondary_error = all(
        bool(site["secondary_local_error_pass"]) for site in sites
    )
    gates = result["gates"]
    expected_gate_values = {
        "primary_denominator_certificate_pass": expected_primary_denominator,
        "primary_action_nrmse_pass": expected_primary_action,
        "primary_pass": expected_primary_denominator and expected_primary_action,
        "secondary_range_pass": expected_secondary_range,
        "secondary_local_scale_error_pass": expected_secondary_error,
        "secondary_pass": expected_secondary_range and expected_secondary_error,
    }
    if gates != expected_gate_values:
        raise RuntimeError("Frozen gate outcomes do not reproduce")
    return {
        "seed": expected_seed,
        "checkpoint": checkpoint,
        "model_parameters": int(model["parameters"]),
        "selection": {
            key: expected_selection[key]
            for key in (
                "selected_count",
                "rank_order_sha256",
                "selected_offsets_sha256",
                "selected_inputs_sha256",
            )
        },
        "normalization_rows": total_rows,
        "fraction_rows_v_in_0p1_10": total_in_range / total_rows,
        "minimum_site_fraction_rows_v_in_0p1_10": min(
            float(site["fraction_rows_v_in_0p1_10"]) for site in sites
        ),
        "fraction_rows_local_scale_error_at_or_below_threshold": (
            total_local_error_at_or_below_threshold / total_rows
        ),
        "minimum_site_fraction_rows_local_scale_error_at_or_below_threshold": min(
            float(site["fraction_rows_local_scale_error_at_or_below_threshold"])
            for site in sites
        ),
        "observed_v_min": aggregate["observed_v_min"],
        "observed_v_max": aggregate["observed_v_max"],
        "observed_denominator_min": aggregate["observed_denominator_min"],
        "local_scale_relative_error_p99": aggregate_p99,
        "local_scale_relative_error_max": aggregate_max,
        "action_nrmse": expected_nrmse,
        "physical_action_nrmse": physical_nrmse,
        "normalized_action_nrmse": normalized_nrmse,
        "action_max_absolute_difference": action_max_absolute_difference,
        "physical_action_squared_difference_sum": physical_diff_sumsq,
        "physical_action_squared_reference_sum": physical_reference_sumsq,
        "normalized_action_squared_difference_sum": normalized_diff_sumsq,
        "normalized_action_squared_reference_sum": normalized_reference_sumsq,
        "gates": expected_gate_values,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    source_hashes_at_start = imported_source_hashes()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if len(args.result) != 3 or len(set(args.result)) != 3:
        raise ValueError("Exactly three distinct full result files are required")
    validate_provenance(args.provenance_result, args.cache, FROZEN_CACHE["sha256"])
    live_provenance_sha256 = file_sha256(args.provenance_result)
    if args.cache.name != FROZEN_CACHE["basename"]:
        raise RuntimeError("Live Object cache basename differs")
    if file_sha256(args.cache) != FROZEN_CACHE["sha256"]:
        raise RuntimeError("Live Object cache differs from the frozen SHA-256")
    frames, stats = load_cache_and_statistics(
        args.cache, PROTOCOL["action_horizon"]
    )
    result_sha256_at_start = {
        str(path): file_sha256(path) for path in args.result
    }
    loaded = [load_json(path) for path in args.result]
    by_seed = {}
    for result in loaded:
        seed = int(result.get("checkpoint", {}).get("seed", -1))
        if seed not in FROZEN_CHECKPOINTS or seed in by_seed:
            raise RuntimeError("Result seeds must be exactly 0, 1, and 2")
        by_seed[seed] = result
    if set(by_seed) != set(FROZEN_CHECKPOINTS):
        raise RuntimeError("Result seeds must be exactly 0, 1, and 2")
    checkpoint_rows = [
        validate_result(
            by_seed[seed],
            seed,
            frames,
            stats,
            source_hashes_at_start,
            live_provenance_sha256,
        )
        for seed in sorted(by_seed)
    ]
    if len({row["model_parameters"] for row in checkpoint_rows}) != 1:
        raise RuntimeError("Model parameter count differs across frozen checkpoints")
    primary_pass = all(row["gates"]["primary_pass"] for row in checkpoint_rows)
    secondary_pass = all(row["gates"]["secondary_pass"] for row in checkpoint_rows)
    pooled_physical_diff_sumsq = sum(
        row["physical_action_squared_difference_sum"] for row in checkpoint_rows
    )
    pooled_physical_reference_sumsq = sum(
        row["physical_action_squared_reference_sum"] for row in checkpoint_rows
    )
    pooled_normalized_diff_sumsq = sum(
        row["normalized_action_squared_difference_sum"] for row in checkpoint_rows
    )
    pooled_normalized_reference_sumsq = sum(
        row["normalized_action_squared_reference_sum"] for row in checkpoint_rows
    )
    source_hashes_at_end = imported_source_hashes()
    if source_hashes_at_end != source_hashes_at_start:
        raise RuntimeError("Summary or imported source changed during aggregation")
    if {
        str(path): file_sha256(path) for path in args.result
    } != result_sha256_at_start:
        raise RuntimeError("A result file changed during aggregation")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "protocol": PROTOCOL,
        "frozen_gates": FROZEN_GATES,
        "inputs": {
            "result_files": [str(path) for path in args.result],
            "result_sha256": result_sha256_at_start,
            "cache": str(args.cache),
            "cache_sha256": FROZEN_CACHE["sha256"],
            "provenance_result": str(args.provenance_result),
            "provenance_result_sha256": live_provenance_sha256,
            "expected_provenance_job_id": FROZEN_PROVENANCE_JOB_ID,
            "source_sha256": source_hashes_at_start,
            "source_unchanged_during_summary": True,
        },
        "checkpoints": checkpoint_rows,
        "aggregate": {
            "checkpoints": 3,
            "selected_samples": 3 * PROTOCOL["full_samples_per_checkpoint"],
            "rational_sites_per_checkpoint": 53,
            "normalization_rows": sum(
                row["normalization_rows"] for row in checkpoint_rows
            ),
            "minimum_fraction_rows_v_in_0p1_10": min(
                row["minimum_site_fraction_rows_v_in_0p1_10"]
                for row in checkpoint_rows
            ),
            "minimum_fraction_rows_local_scale_error_at_or_below_threshold": min(
                row[
                    "minimum_site_fraction_rows_local_scale_error_at_or_below_threshold"
                ]
                for row in checkpoint_rows
            ),
            "maximum_local_scale_relative_error_p99": max(
                row["local_scale_relative_error_p99"] for row in checkpoint_rows
            ),
            "minimum_observed_denominator": min(
                row["observed_denominator_min"] for row in checkpoint_rows
            ),
            "observed_v_min": min(row["observed_v_min"] for row in checkpoint_rows),
            "observed_v_max": max(row["observed_v_max"] for row in checkpoint_rows),
            "pooled_physical_action_nrmse": math.sqrt(
                pooled_physical_diff_sumsq / pooled_physical_reference_sumsq
            ),
            "pooled_normalized_action_nrmse": math.sqrt(
                pooled_normalized_diff_sumsq / pooled_normalized_reference_sumsq
            ),
            "maximum_checkpoint_action_nrmse": max(
                row["action_nrmse"] for row in checkpoint_rows
            ),
        },
        "gates": {
            "all_three_primary_pass": primary_pass,
            "all_three_secondary_pass": secondary_pass,
        },
        "claim_if_primary_passed": (
            "Across three frozen capable convolutional checkpoints and 4,096 "
            "deterministically selected Object cache inputs per checkpoint, all 53 "
            "deployed rational-normalizer sites have positive denominators by a "
            "coefficient and root certificate, and replacing only their arithmetic "
            "with a float64 rational reference changes physical action outputs by at "
            "most the preregistered 1e-3 NRMSE gate."
        ),
        "additional_claim_if_secondary_passed": (
            "At every deployed site, at least 99 percent of observed normalizer rows "
            "fall inside v in [0.1,10], and at least 99 percent have local RMS-scale "
            "relative error at or below 3.4 percent."
        ),
        "scope": (
            "The certificate is bounded to the pinned LIBERO-Object cache, three fixed "
            "convolutional checkpoints, and the observed normalizer inputs. The positive-"
            "coefficient denominator result holds for every v>=0, while approximation "
            "and end-to-end error measurements are empirical."
        ),
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(args.output, summary)
    print("RESULT", json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not primary_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
