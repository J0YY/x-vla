#!/usr/bin/env python3
"""Athena exactness audit for the complete typed attention compiler."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path
from statistics import mean, median

import torch

from xvla.nn.attention import BilinearAttention
from xvla.train.complete_attention_odt import (
    compile_bilinear_attention,
    evaluate_compiled_attention,
    pullback_route_metric,
)


DTYPE = torch.float64


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)
    return float((torch.linalg.vector_norm(actual - expected) / denominator).item())


def run_seed(seed: int) -> dict[str, object]:
    torch.manual_seed(seed)
    dimension = 6
    heads = 3
    positions = 4
    causal = bool(seed % 2)
    row_scale = "invsqrt" if seed % 3 else "inv"
    score_scale = "d_h2" if seed % 5 else "d_h"
    module = BilinearAttention(
        dimension,
        heads,
        causal=causal,
        row_scale=row_scale,
        score_scale=score_scale,
        qk_norm="none",
    ).double().eval()
    with torch.no_grad():
        for branch in (module.wq1, module.wk1, module.wq2, module.wk2, module.wv, module.wo):
            branch.bias.normal_()
    tokens = torch.randn(5, positions, dimension, dtype=DTYPE)
    compiled = compile_bilinear_attention(
        module,
        n_query=positions,
        query_types=("vision", "language", "state", "action"),
        source_types=("vision", "language", "state", "action"),
    )
    expected = module(tokens, method="explicit")
    actual = evaluate_compiled_attention(compiled, tokens)

    output_dimension = positions * dimension
    raw = torch.randn(output_dimension, output_dimension, dtype=DTYPE)
    output_metric = raw @ raw.T / output_dimension
    route_metric = pullback_route_metric(compiled, output_metric)
    route_vector = torch.randn(compiled.output_assembly.shape[1], dtype=DTYPE)
    output = compiled.output_assembly @ route_vector
    route_energy = route_vector @ route_metric @ route_vector
    output_energy = output @ output_metric @ output

    return {
        "seed": seed,
        "causal": causal,
        "row_scale": row_scale,
        "score_scale": score_scale,
        "route_count": len(compiled.routes),
        "compiler_relative_error": relative_error(actual, expected),
        "compiler_maximum_absolute_error": float((actual - expected).abs().max().item()),
        "route_energy_relative_error": float(
            ((route_energy - output_energy).abs() / output_energy.abs().clamp_min(1e-300)).item()
        ),
        "route_metric_symmetry_error": relative_error(route_metric, route_metric.T),
        "route_metric_minimum_eigenvalue": float(torch.linalg.eigvalsh(route_metric)[0].item()),
    }


def cancellation_fixture() -> dict[str, float]:
    torch.manual_seed(1000)
    module = BilinearAttention(2, 2, qk_norm="none").double().eval()
    with torch.no_grad():
        for branch in (module.wq1, module.wk1, module.wq2, module.wk2, module.wv):
            branch.weight[1].copy_(branch.weight[0])
            branch.bias[1].copy_(branch.bias[0])
        module.wo.weight.zero_()
        module.wo.weight[0] = torch.tensor([1.0, -1.0], dtype=DTYPE)
        module.wo.bias.zero_()
    compiled = compile_bilinear_attention(module, n_query=1)
    metric = pullback_route_metric(compiled, torch.eye(2, dtype=DTYPE))[1:, 1:]
    features = torch.ones(2, dtype=DTYPE)
    return {
        "combined_energy": float((features @ metric @ features).item()),
        "diagonal_surrogate_energy": float(
            (features @ torch.diag(torch.diag(metric)) @ features).item()
        ),
        "cross_head_entry": float(metric[0, 1].item()),
    }


def real_width_factorization_fixture() -> dict[str, object]:
    torch.manual_seed(1001)
    module = BilinearAttention(384, 12, qk_norm="none").double().eval()
    compiled = compile_bilinear_attention(
        module,
        n_query=1,
        dense_oracle_maximum_elements=1000,
    )
    tokens = torch.randn(2, 1, 384, dtype=DTYPE)
    actual = evaluate_compiled_attention(compiled, tokens)
    expected = module(tokens)
    return {
        "factor_shape": list(compiled.head_factors[0].shape),
        "dense_head_core_materialized": compiled.head_cores is not None,
        "dense_output_assembly_materialized": compiled.output_assembly is not None,
        "relative_error": relative_error(actual, expected),
        "maximum_absolute_error": float((actual - expected).abs().max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("athena/results/complete_attention_odt_v1.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result {args.output}")
    if args.seeds < 1:
        raise ValueError("seeds must be positive")
    rows = [run_seed(seed) for seed in range(args.seeds)]
    cancellation = cancellation_fixture()
    real_width = real_width_factorization_fixture()
    gates = {
        "maximum_compiler_relative_error_below_1e-11": max(
            float(row["compiler_relative_error"]) for row in rows
        ) < 1e-11,
        "maximum_compiler_absolute_error_below_1e-10": max(
            float(row["compiler_maximum_absolute_error"]) for row in rows
        ) < 1e-10,
        "maximum_route_energy_relative_error_below_1e-11": max(
            float(row["route_energy_relative_error"]) for row in rows
        ) < 1e-11,
        "maximum_route_metric_symmetry_error_below_1e-13": max(
            float(row["route_metric_symmetry_error"]) for row in rows
        ) < 1e-13,
        "cancellation_combined_energy_below_1e-12": abs(cancellation["combined_energy"]) < 1e-12,
        "cancellation_diagonal_energy_above_1": cancellation["diagonal_surrogate_energy"] > 1.0,
        "cancellation_cross_head_entry_negative": cancellation["cross_head_entry"] < 0.0,
        "real_width_dense_core_not_materialized": not real_width["dense_head_core_materialized"],
        "real_width_dense_assembly_not_materialized": not real_width["dense_output_assembly_materialized"],
        "real_width_relative_error_below_1e-10": real_width["relative_error"] < 1e-10,
    }
    manifest = Path("athena/complete_attention_odt_sources.sha256")
    payload = {
        "schema": "xvla-complete-attention-odt-v1",
        "configuration": {"seeds": args.seeds, "dtype": "float64"},
        "provenance": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source_manifest": manifest.as_posix(),
            "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "source_tree_sha256": os.environ.get("XVLA_SOURCE_TREE_SHA256"),
            "pip_freeze_sha256": os.environ.get("XVLA_PIP_FREEZE_SHA256"),
        },
        "frozen_acceptance_gates": {
            "passed": all(gates.values()),
            "checks": gates,
        },
        "summary": {
            "maximum_compiler_relative_error": max(float(row["compiler_relative_error"]) for row in rows),
            "median_compiler_relative_error": median(float(row["compiler_relative_error"]) for row in rows),
            "maximum_compiler_absolute_error": max(float(row["compiler_maximum_absolute_error"]) for row in rows),
            "maximum_route_energy_relative_error": max(float(row["route_energy_relative_error"]) for row in rows),
            "mean_route_count": mean(int(row["route_count"]) for row in rows),
        },
        "cancellation_fixture": cancellation,
        "real_width_factorization_fixture": real_width,
        "per_seed": rows,
        "claim_boundary": (
            "This is an exact factorized forward compiler at real width for one fixed-mask "
            "strict-polynomial bilinear attention operation. The exact dense route metric is "
            "validated only under the declared tiny-oracle materialization limit. Joint "
            "factorized pullback through all route cores remains a separate stage. This is not "
            "yet a canonicalized whole-policy ODT and does not treat rational normalization as "
            "ordinary polynomial ODT."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"gates": payload["frozen_acceptance_gates"], "summary": payload["summary"], "cancellation": cancellation, "real_width": real_width}, indent=2))
    print(f"wrote {args.output}")
    if not payload["frozen_acceptance_gates"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
