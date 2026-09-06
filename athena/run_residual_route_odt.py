#!/usr/bin/env python3
"""Athena exactness audit for residual route metrics and covariances."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path
from statistics import median

import torch

from xvla.nn.attention import BilinearAttention
from xvla.nn.bilinear import BilinearFFN
from xvla.train.complete_attention_odt import (
    attention_route_features,
    compile_bilinear_attention,
)
from xvla.train.residual_route_odt import (
    compile_bilinear_ffn_residual_core,
    compile_residual_route_assembly,
    pullback_residual_feature_metric,
    pushforward_residual_covariance,
    residual_output_route_metric,
)
from xvla.train.generalized_metric_odt import forward_metric, role_environment


DTYPE = torch.float64


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)
    return float((torch.linalg.vector_norm(actual - expected) / denominator).item())


def run_seed(seed: int) -> dict[str, float | int]:
    torch.manual_seed(seed)
    dimension = 4
    positions = 3
    gain = float(torch.empty(()).uniform_(-1.2, 1.2).item())
    module = BilinearAttention(
        dimension, 2, causal=bool(seed % 2), qk_norm="none"
    ).double().eval()
    with torch.no_grad():
        for branch in (module.wq1, module.wk1, module.wq2, module.wk2, module.wv, module.wo):
            branch.bias.normal_()
    attention = compile_bilinear_attention(module, n_query=positions)
    assert attention.output_assembly is not None
    residual = compile_residual_route_assembly(attention.output_assembly, gain=gain)
    tokens = torch.randn(5, positions, dimension, dtype=DTYPE)
    features = attention_route_features(attention, tokens)
    feature_routes = torch.cat((torch.ones(5, 1, dtype=DTYPE), features), dim=1)
    all_routes = torch.cat((tokens.flatten(1), feature_routes), dim=1)
    actual = all_routes @ residual.assembly.T
    expected = (tokens + gain * module(tokens)).flatten(1)

    output_dimension = positions * dimension
    raw = torch.randn(output_dimension, output_dimension, dtype=DTYPE)
    output_metric = raw @ raw.T / output_dimension
    pulled = pullback_residual_feature_metric(residual, output_metric)
    route = torch.randn(residual.assembly.shape[1], dtype=DTYPE)
    output = residual.assembly @ route
    route_energy = route @ pulled @ route
    output_energy = output @ output_metric @ output

    base = torch.randn(1000, dimension, dtype=DTYPE)
    linear = torch.randn(dimension, dimension, dtype=DTYPE) / dimension**0.5
    update = base @ linear.T
    base_covariance = base.T @ base / base.shape[0]
    update_covariance = update.T @ update / update.shape[0]
    cross = base.T @ update / base.shape[0]
    propagated = pushforward_residual_covariance(
        base_covariance, update_covariance, cross, gain=gain
    )
    direct = (base + gain * update).T @ (base + gain * update) / base.shape[0]

    ffn = BilinearFFN(dimension, rank=7, out_dim=dimension, down_bias=True).double().eval()
    with torch.no_grad():
        ffn.left.bias.normal_()
        ffn.right.bias.normal_()
        ffn.down.bias.normal_()
    attention_state = tokens + gain * module(tokens)
    ffn_gain = float(torch.empty(()).uniform_(-0.8, 0.8).item())
    ffn_core = compile_bilinear_ffn_residual_core(ffn, gain=ffn_gain)
    homogeneous = torch.cat(
        (torch.ones(*attention_state.shape[:-1], 1, dtype=DTYPE), attention_state), dim=-1
    )
    compiled_nested = torch.einsum(
        "oij,bni,bnj->bno", ffn_core, homogeneous, homogeneous
    )
    expected_nested = attention_state + ffn_gain * ffn(attention_state)
    expected_homogeneous = torch.cat(
        (torch.ones(*expected_nested.shape[:-1], 1, dtype=DTYPE), expected_nested), dim=-1
    )
    nested_replay_error = relative_error(compiled_nested, expected_homogeneous)

    flat_homogeneous = homogeneous.reshape(-1, dimension + 1)
    ffn_input_metric = flat_homogeneous.T @ flat_homogeneous / flat_homogeneous.shape[0]
    endpoint = torch.randn(dimension, dimension + 1, dtype=DTYPE)
    endpoint[:, 0] = 0.0
    ffn_output_metric = endpoint.T @ endpoint
    ffn_output_coefficient_metric = forward_metric(
        ffn_core, (ffn_input_metric, ffn_input_metric)
    )
    parent_energy = torch.trace(ffn_output_metric @ ffn_output_coefficient_metric)
    energy_errors = []
    for role in range(2):
        environment = role_environment(
            ffn_core,
            ffn_output_metric,
            (ffn_input_metric, ffn_input_metric),
            role,
        )
        child_energy = torch.trace(environment @ ffn_input_metric)
        energy_errors.append(
            float(((child_energy - parent_energy).abs() / parent_energy.abs().clamp_min(1e-300)).item())
        )
    return {
        "seed": seed,
        "forward_relative_error": relative_error(actual, expected),
        "metric_energy_relative_error": float(
            ((route_energy - output_energy).abs() / output_energy.abs().clamp_min(1e-300)).item()
        ),
        "covariance_relative_error": relative_error(propagated, direct),
        "nested_attention_ffn_replay_relative_error": nested_replay_error,
        "nested_ffn_maximum_role_energy_relative_error": max(energy_errors),
    }


def cancellation_fixture() -> dict[str, float]:
    gain = 0.4
    metric = torch.eye(3, dtype=DTYPE)
    route_metric = residual_output_route_metric(metric, gain=gain)
    base = torch.tensor([1.0, -2.0, 0.5], dtype=DTYPE)
    update = -base / gain
    routes = torch.cat((base, update))
    return {
        "full_energy": float((routes @ route_metric @ routes).item()),
        "diagonal_surrogate_energy": float(
            (routes @ torch.diag(torch.diag(route_metric)) @ routes).item()
        ),
        "cross_block_error": relative_error(route_metric[:3, 3:], gain * metric),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("athena/results/residual_route_odt_v1.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result {args.output}")
    rows = [run_seed(seed) for seed in range(args.seeds)]
    cancellation = cancellation_fixture()
    checks = {
        "maximum_forward_relative_error_below_1e-11": max(float(row["forward_relative_error"]) for row in rows) < 1e-11,
        "maximum_metric_energy_relative_error_below_1e-11": max(float(row["metric_energy_relative_error"]) for row in rows) < 1e-11,
        "maximum_covariance_relative_error_below_1e-11": max(float(row["covariance_relative_error"]) for row in rows) < 1e-11,
        "maximum_nested_replay_relative_error_below_1e-11": max(float(row["nested_attention_ffn_replay_relative_error"]) for row in rows) < 1e-11,
        "maximum_nested_role_energy_error_below_1e-11": max(float(row["nested_ffn_maximum_role_energy_relative_error"]) for row in rows) < 1e-11,
        "cancellation_full_energy_below_1e-12": abs(cancellation["full_energy"]) < 1e-12,
        "cancellation_diagonal_energy_positive": cancellation["diagonal_surrogate_energy"] > 1.0,
        "cross_block_error_below_1e-13": cancellation["cross_block_error"] < 1e-13,
    }
    manifest = Path("athena/residual_route_odt_sources.sha256")
    payload = {
        "schema": "xvla-residual-route-odt-v1",
        "configuration": {"seeds": args.seeds, "dtype": "float64"},
        "provenance": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source_manifest": manifest.as_posix(),
            "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "source_tree_sha256": os.environ.get("XVLA_SOURCE_TREE_SHA256"),
            "pip_freeze_sha256": os.environ.get("XVLA_PIP_FREEZE_SHA256"),
        },
        "frozen_acceptance_gates": {"passed": all(checks.values()), "checks": checks},
        "summary": {
            "maximum_forward_relative_error": max(float(row["forward_relative_error"]) for row in rows),
            "median_forward_relative_error": median(float(row["forward_relative_error"]) for row in rows),
            "maximum_metric_energy_relative_error": max(float(row["metric_energy_relative_error"]) for row in rows),
            "maximum_covariance_relative_error": max(float(row["covariance_relative_error"]) for row in rows),
            "maximum_nested_attention_ffn_replay_relative_error": max(float(row["nested_attention_ffn_replay_relative_error"]) for row in rows),
            "maximum_nested_ffn_role_energy_relative_error": max(float(row["nested_ffn_maximum_role_energy_relative_error"]) for row in rows),
        },
        "cancellation_fixture": cancellation,
        "per_seed": rows,
        "claim_boundary": (
            "This validates exact linear residual-route assembly, downstream metric cross blocks, "
            "upstream cross covariances, and a nested attention-to-FFN replay. The FFN core is the "
            "explicitly declared tied-input symmetric polynomial quotient, not the ordered CP "
            "topology tensor. This is not a joint factorized pullback through shared attention "
            "inputs and is not global ODT of a VLA."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"gates": payload["frozen_acceptance_gates"], "summary": payload["summary"], "cancellation": cancellation}, indent=2))
    print(f"wrote {args.output}")
    if not payload["frozen_acceptance_gates"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
