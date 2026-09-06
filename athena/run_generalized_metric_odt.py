#!/usr/bin/env python3
"""Athena validation of exact k-ary metric recursions and clone estimators."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import math
from pathlib import Path
from statistics import mean, median

import torch

from xvla.train.generalized_metric_odt import (
    balanced_metric_odt,
    estimate_forward_metric,
    estimate_shared_environment,
    forward_metric,
    shared_environment,
)


DTYPE = torch.float64
ARITIES = (2, 3, 5)
BUDGETS = (256, 1024, 4096, 16384)


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)
    return float((torch.linalg.vector_norm(actual - expected) / denominator).item())


def top_subspace_similarity(left: torch.Tensor, right: torch.Tensor, rank: int = 1) -> float:
    return float(torch.linalg.matrix_norm(left[:, :rank].T @ right[:, :rank]).square().item() / rank)


def random_spd(generator: torch.Generator, dimension: int) -> torch.Tensor:
    factor = torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
    return factor @ factor.T / dimension + 0.4 * torch.eye(dimension, dtype=DTYPE)


def run_seed(seed: int, dimension: int, output_dimension: int) -> dict[str, object]:
    rows: dict[str, object] = {}
    for arity in ARITIES:
        generator = torch.Generator().manual_seed(100_000 * arity + seed)
        core = torch.randn(
            (output_dimension,) + (dimension,) * arity,
            generator=generator,
            dtype=DTYPE,
        ) / (dimension ** (arity / 2))
        upstream = random_spd(generator, dimension)
        inputs = (upstream,) * arity
        downstream = random_spd(generator, output_dimension)
        exact_forward = forward_metric(core, inputs)
        exact_backward = shared_environment(core, downstream, inputs)
        exact_balanced = balanced_metric_odt(upstream, exact_backward)
        top_gap = float((exact_balanced.eigenvalues[0] - exact_balanced.eigenvalues[1]).item())
        top_gap_relative = top_gap / max(float(exact_balanced.eigenvalues[0].item()), 1e-300)
        top_gap_resolved = top_gap_relative > 0.05

        budgets: dict[str, object] = {}
        for budget in BUDGETS:
            mc_generator = torch.Generator().manual_seed(
                10_000_000 * arity + 10_000 * seed + budget
            )
            estimated_forward = estimate_forward_metric(
                core, inputs, samples=budget, generator=mc_generator
            )
            estimated_backward = estimate_shared_environment(
                core,
                downstream,
                inputs,
                samples=budget,
                generator=mc_generator,
            )
            estimated_balanced = balanced_metric_odt(upstream, estimated_backward)
            budgets[str(budget)] = {
                "forward_relative_frobenius_error": relative_error(
                    estimated_forward, exact_forward
                ),
                "backward_relative_frobenius_error": relative_error(
                    estimated_backward, exact_backward
                ),
                "balanced_spectrum_relative_error": relative_error(
                    estimated_balanced.eigenvalues, exact_balanced.eigenvalues
                ),
                "top_balanced_subspace_similarity": (
                    top_subspace_similarity(
                        estimated_balanced.whitened_vectors,
                        exact_balanced.whitened_vectors,
                    )
                    if top_gap_resolved
                    else None
                ),
            }
        rows[str(arity)] = {
            "exact_forward_trace": float(torch.trace(exact_forward).item()),
            "exact_backward_trace": float(torch.trace(exact_backward).item()),
            "exact_balanced_spectrum": exact_balanced.eigenvalues.tolist(),
            "top_gap_relative": top_gap_relative,
            "top_gap_resolved": top_gap_resolved,
            "budgets": budgets,
        }
    return {"seed": seed, "arities": rows}


def negative_controls(samples: int) -> dict[str, float]:
    forward_core = torch.eye(2, dtype=DTYPE).unsqueeze(0)
    forward_metrics = (torch.eye(2, dtype=DTYPE),) * 2
    exact_forward = forward_metric(forward_core, forward_metrics)
    independent_forward = estimate_forward_metric(
        forward_core,
        forward_metrics,
        samples=samples,
        generator=torch.Generator().manual_seed(700),
    )
    reused_forward = estimate_forward_metric(
        forward_core,
        forward_metrics,
        samples=samples,
        generator=torch.Generator().manual_seed(701),
        independent_clones=False,
    )

    backward_core = torch.ones(1, 1, 1, 1, dtype=DTYPE)
    backward_metrics = (torch.ones(1, 1, dtype=DTYPE),) * 3
    output_metric = torch.ones(1, 1, dtype=DTYPE)
    exact_backward = shared_environment(backward_core, output_metric, backward_metrics)
    independent_backward = estimate_shared_environment(
        backward_core,
        output_metric,
        backward_metrics,
        samples=samples,
        generator=torch.Generator().manual_seed(702),
    )
    reused_backward = estimate_shared_environment(
        backward_core,
        output_metric,
        backward_metrics,
        samples=samples,
        generator=torch.Generator().manual_seed(703),
        independent_contexts=False,
    )
    return {
        "exact_forward": float(exact_forward.item()),
        "independent_forward": float(independent_forward.item()),
        "reused_forward": float(reused_forward.item()),
        "independent_forward_relative_error": relative_error(
            independent_forward, exact_forward
        ),
        "reused_forward_relative_error": relative_error(reused_forward, exact_forward),
        "independent_backward_relative_error": relative_error(
            independent_backward, exact_backward
        ),
        "reused_backward_relative_error": relative_error(reused_backward, exact_backward),
        "exact_backward": float(exact_backward.item()),
        "independent_backward": float(independent_backward.item()),
        "reused_backward": float(reused_backward.item()),
    }


def analytic_variance_controls(
    *, replicas: int = 512, samples: int = 4096
) -> dict[str, object]:
    """Check the known Gaussian scalar k-clone variance law."""

    rows: dict[str, object] = {}
    generator = torch.Generator().manual_seed(710)
    for arity in ARITIES:
        probes = torch.randn(replicas, samples, arity, generator=generator, dtype=DTYPE)
        estimates = probes.square().prod(dim=2).mean(dim=1)
        empirical_scaled_mse = float((samples * (estimates - 1.0).square().mean()).item())
        analytic_scaled_mse = float(3**arity - 1)
        rows[str(arity)] = {
            "replicas": replicas,
            "samples_per_replica": samples,
            "empirical_samples_times_mse": empirical_scaled_mse,
            "analytic_samples_times_mse": analytic_scaled_mse,
            "empirical_over_analytic": empirical_scaled_mse / analytic_scaled_mse,
        }
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for arity in ARITIES:
        budget_rows: dict[str, object] = {}
        for budget in BUDGETS:
            metrics = {}
            for field in (
                "forward_relative_frobenius_error",
                "backward_relative_frobenius_error",
                "balanced_spectrum_relative_error",
                "top_balanced_subspace_similarity",
            ):
                values = [
                    float(value)
                    for row in rows
                    for value in [row["arities"][str(arity)]["budgets"][str(budget)][field]]
                    if value is not None
                ]
                ordered = sorted(values)
                percentile_95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
                metrics[field] = {
                    "mean": mean(values),
                    "median": median(values),
                    "maximum": max(values),
                    "minimum": min(values),
                    "percentile_95": percentile_95,
                    "count": len(values),
                }
            budget_rows[str(budget)] = metrics
        slopes = {}
        log_budgets = [math.log(value) for value in BUDGETS[-3:]]
        centered_x = [value - mean(log_budgets) for value in log_budgets]
        denominator = sum(value * value for value in centered_x)
        for field in (
            "forward_relative_frobenius_error",
            "backward_relative_frobenius_error",
            "balanced_spectrum_relative_error",
        ):
            log_errors = [
                math.log(max(float(budget_rows[str(budget)][field]["median"]), 1e-300))
                for budget in BUDGETS[-3:]
            ]
            centered_y = [value - mean(log_errors) for value in log_errors]
            slopes[field] = sum(x * y for x, y in zip(centered_x, centered_y)) / denominator
        summary[str(arity)] = {
            "budgets": budget_rows,
            "resolved_top_gap_count": sum(
                bool(row["arities"][str(arity)]["top_gap_resolved"]) for row in rows
            ),
            "median_log_error_slope_last_three_budgets": slopes,
        }
    return summary


def evaluate_frozen_gates(
    summary: dict[str, object],
    controls: dict[str, float],
    variance_controls: dict[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    for arity in ARITIES:
        first = summary[str(arity)]["budgets"][str(BUDGETS[0])]
        last = summary[str(arity)]["budgets"][str(BUDGETS[-1])]
        for field in (
            "forward_relative_frobenius_error",
            "backward_relative_frobenius_error",
            "balanced_spectrum_relative_error",
        ):
            if not float(last[field]["median"]) < float(first[field]["median"]):
                failures.append(f"arity {arity}: {field} median did not improve")
        if float(last["forward_relative_frobenius_error"]["median"]) >= 0.10:
            failures.append(f"arity {arity}: final forward median error is at least 0.10")
        if float(last["backward_relative_frobenius_error"]["median"]) >= 0.10:
            failures.append(f"arity {arity}: final backward median error is at least 0.10")
        if float(last["balanced_spectrum_relative_error"]["median"]) >= 0.10:
            failures.append(f"arity {arity}: final spectrum median error is at least 0.10")
        if float(last["top_balanced_subspace_similarity"]["median"]) <= 0.90:
            failures.append(f"arity {arity}: final top-subspace median is at most 0.90")
        if int(summary[str(arity)]["resolved_top_gap_count"]) < 12:
            failures.append(f"arity {arity}: fewer than 12 seeds have a resolved top eigengap")
        if float(last["forward_relative_frobenius_error"]["percentile_95"]) >= 0.15:
            failures.append(f"arity {arity}: final forward 95th percentile is at least 0.15")
        for field in (
            "backward_relative_frobenius_error",
            "balanced_spectrum_relative_error",
        ):
            if float(last[field]["percentile_95"]) >= 0.10:
                failures.append(f"arity {arity}: final {field} 95th percentile is at least 0.10")
        for field, slope in summary[str(arity)]["median_log_error_slope_last_three_budgets"].items():
            if float(slope) > -0.25:
                failures.append(f"arity {arity}: {field} convergence slope is above -0.25")
    if controls["independent_forward_relative_error"] >= 0.05:
        failures.append("independent forward negative-control estimate missed its oracle")
    if controls["reused_forward_relative_error"] <= 1.0:
        failures.append("reused forward clones did not expose a large tied-moment bias")
    if controls["independent_backward_relative_error"] >= 0.08:
        failures.append("independent backward negative-control estimate missed its oracle")
    if controls["reused_backward_relative_error"] <= 0.20:
        failures.append("reused backward contexts did not expose tied-moment bias")
    if abs(controls["independent_backward"] - 3.0) >= 0.08:
        failures.append("independent scalar backward estimate missed expectation 3")
    if abs(controls["reused_backward"] - 9.0) >= 0.30:
        failures.append("reused scalar backward estimate missed tied expectation 9")
    for arity, row in variance_controls.items():
        ratio = float(row["empirical_over_analytic"])
        if not 0.75 < ratio < 1.25:
            failures.append(f"arity {arity}: empirical Gaussian variance law ratio outside (0.75, 1.25)")
    return {"passed": not failures, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--output-dimension", type=int, default=3)
    parser.add_argument("--negative-control-samples", type=int, default=262144)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("athena/results/generalized_metric_odt_v1.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result {args.output}")
    if args.seeds < 1 or args.dimension < 2 or args.output_dimension < 1:
        raise ValueError("invalid experiment dimensions")

    rows = [run_seed(seed, args.dimension, args.output_dimension) for seed in range(args.seeds)]
    summary = summarize(rows)
    controls = negative_controls(args.negative_control_samples)
    variance_controls = analytic_variance_controls()
    gates = evaluate_frozen_gates(summary, controls, variance_controls)
    manifest_path = Path("athena/generalized_metric_odt_sources.sha256")
    payload = {
        "schema": "xvla-generalized-metric-odt-v1",
        "configuration": {
            "seeds": args.seeds,
            "arities": list(ARITIES),
            "budgets": list(BUDGETS),
            "dimension": args.dimension,
            "output_dimension": args.output_dimension,
            "negative_control_samples": args.negative_control_samples,
            "dtype": "float64",
        },
        "provenance": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source_manifest": manifest_path.as_posix(),
            "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "source_tree_sha256": os.environ.get("XVLA_SOURCE_TREE_SHA256"),
            "pip_freeze_sha256": os.environ.get("XVLA_PIP_FREEZE_SHA256"),
        },
        "frozen_acceptance_gates": gates,
        "negative_controls": controls,
        "analytic_variance_controls": variance_controls,
        "summary": summary,
        "per_seed": rows,
        "claim_boundary": (
            "The exact recursion and Monte Carlo validation concern a declared typed multilinear "
            "core and coefficient-space second moments. Shared_environment is the sum of separate "
            "role environments and a certificate objective for shared projection. It is not an "
            "exact simultaneous-projection loss and it is not yet whole-policy VLA ODT."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"gates": gates, "negative_controls": controls, "analytic_variance_controls": variance_controls, "summary": summary}, indent=2))
    print(f"wrote {args.output}")
    if not gates["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
