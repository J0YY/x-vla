#!/usr/bin/env python3
"""Run the N=3,D=2 typed local-bond projective DAG experiment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from xvla.nn.block import ChiTransformerBlock
from xvla.nn.normalization import RationalNorm
from xvla.train.typed_projective_dag_oracle import (
    CLAIM_BOUNDARY,
    FIXED_GAUGE,
    TypedProjectiveDAG,
    aggregate_clone_environments,
    assert_typed_norm_buffers_unchanged,
    brute_first_role_scaled_environment,
    canonicalize_typed_projective_dag,
    compile_typed_projective_block,
    diagonalize_typed_projective_dag_full_rank,
    evaluate_scaled_typed_pair,
    evaluate_typed_pair_unscaled_negative_control,
    evaluate_typed_quotient,
    explicit_clone_typed_environments,
    fix_typed_projective_gauge,
    gauge_shared_typed_node,
    maximum_aggregate_environment_offdiagonal_ratio,
    omit_quotient_cross_terms_negative_control,
    quotient_pair_output_metric,
    rescale_typed_projective_output,
    reverse_typed_projective_environments,
    scaled_matrix_relative_error,
    scaled_projective_relative_error,
    scaled_projective_statistics,
    shared_parent_occurrence_count,
    source_typed_action,
    typed_dag_shape_statistics,
    typed_pair_coefficient_norm,
)


DTYPE = torch.float64


def make_oracle(seed: int, *, tokens: int, dimension: int, rank: int):
    torch.manual_seed(seed)
    block = ChiTransformerBlock(
        dim=dimension,
        n_heads=1,
        ffn_rank=rank,
        n_layers=1,
        causal=True,
        norm="rational",
        qk_norm="rational",
        residual=True,
    ).double().eval()
    action_head = nn.Linear(dimension, 1, bias=True).double().eval()
    generator = torch.Generator().manual_seed(10_000 + seed)
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                module.weight.copy_(
                    0.04
                    + 0.12
                    * torch.rand(module.weight.shape, generator=generator, dtype=DTYPE)
                )
                if module.bias is not None:
                    module.bias.copy_(
                        -0.08
                        + 0.16
                        * torch.rand(module.bias.shape, generator=generator, dtype=DTYPE)
                    )
                    module.bias.add_(torch.where(module.bias >= 0, 0.025, -0.025))
        action_head.weight.copy_(
            torch.linspace(0.19, 0.31, dimension, dtype=DTYPE).reshape(1, dimension)
        )
        action_head.bias.fill_(-0.13)
        for index, norm in enumerate(
            module for module in block.modules() if isinstance(module, RationalNorm)
        ):
            norm.running_ms.fill_(0.95 + 0.09 * index)
            norm.initialized.fill_(True)
            norm.frozen = True
    positions = torch.stack(
        [
            torch.linspace(
                -0.045 + 0.017 * token,
                0.055 + 0.013 * token,
                dimension,
                dtype=DTYPE,
            )
            for token in range(tokens)
        ]
    )
    return compile_typed_projective_block(block, action_head, positions)


def inputs(tokens: int, dimension: int) -> torch.Tensor:
    values = torch.linspace(-0.32, 0.41, 4 * tokens * dimension, dtype=DTYPE)
    raw = values.reshape(4, tokens, dimension)
    raw[1].copy_(raw[1].flip(0))
    raw[2].mul_(-0.7)
    raw[3].zero_()
    return raw


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)
    return float((torch.linalg.vector_norm(actual - expected) / scale).item())


def run(seed: int, rank: int) -> dict[str, object]:
    timings: dict[str, float] = {}
    started = time.perf_counter()
    oracle = make_oracle(seed, tokens=3, dimension=2, rank=rank)
    timings["compile_seconds"] = time.perf_counter() - started
    assert_typed_norm_buffers_unchanged(oracle)
    raw = inputs(3, 2)
    source = source_typed_action(oracle, raw)
    quotient = evaluate_typed_quotient(oracle.network, raw)
    reference = evaluate_scaled_typed_pair(oracle.network, raw)

    started = time.perf_counter()
    canonical = canonicalize_typed_projective_dag(oracle.network)
    timings["shared_bottom_up_rq_seconds"] = time.perf_counter() - started
    canonical_pair = evaluate_scaled_typed_pair(canonical.network, raw)

    started = time.perf_counter()
    environments = reverse_typed_projective_environments(canonical.network)
    timings["reverse_shared_environment_seconds"] = time.perf_counter() - started
    recursive, brute = brute_first_role_scaled_environment(canonical.network)

    started = time.perf_counter()
    diagonal = diagonalize_typed_projective_dag_full_rank(canonical.network)
    timings["shared_full_rank_evd_seconds"] = time.perf_counter() - started
    diagonal_pair = evaluate_scaled_typed_pair(diagonal.network, raw)
    post_evd_aggregate_offdiagonal = maximum_aggregate_environment_offdiagonal_ratio(
        diagonal.network
    )

    label = "position.token2"
    transform = torch.tensor(
        ((1.15, 0.12, -0.04), (-0.08, 0.91, 0.09), (0.06, -0.11, 1.07)),
        dtype=DTYPE,
    )
    gauged = gauge_shared_typed_node(oracle.network, label, transform)
    gauged_pair = evaluate_scaled_typed_pair(gauged, raw)
    gauged_canonical = canonicalize_typed_projective_dag(gauged)
    missing_parent = gauge_shared_typed_node(
        oracle.network, label, transform, omit_parent_occurrence=0
    )
    missing_parent_error = relative_error(
        evaluate_typed_quotient(missing_parent, raw), quotient
    )
    # Preserve the first, borderline-sensitive probe in the artifact.  The q2
    # Padé repeated leg below is the independently swept, load-bearing gate.
    initial_missing_leg = canonicalize_typed_projective_dag(
        oracle.network,
        omit_factor_at=("ffn_pre_norm.mean_square_homogeneous", 1),
    )
    initial_missing_leg_error = relative_error(
        evaluate_typed_quotient(initial_missing_leg.network, raw), quotient
    )
    missing_leg = canonicalize_typed_projective_dag(
        oracle.network,
        omit_factor_at=("attention.q2.norm.q2.pade_homogeneous", 0),
    )
    missing_leg_error = relative_error(
        evaluate_typed_quotient(missing_leg.network, raw), quotient
    )

    rescaled = rescale_typed_projective_output(oracle.network, -6.5)
    refixed = fix_typed_projective_gauge(rescaled)
    fixed_metadata_rejected = False
    forged = TypedProjectiveDAG(
        rescaled.root,
        rescaled.head,
        rescaled.token_count,
        rescaled.feature_dimension,
        FIXED_GAUGE,
    )
    try:
        canonicalize_typed_projective_dag(forged)
    except ValueError:
        fixed_metadata_rejected = True
    indefinite_output_metric_rejected = False
    try:
        reverse_typed_projective_environments(
            canonical.network,
            torch.tensor(((1.0, 0.0), (0.0, -0.1)), dtype=DTYPE),
        )
    except ValueError:
        indefinite_output_metric_rejected = True

    metric_pair = reference.mantissa[1]
    metric = quotient_pair_output_metric(metric_pair)
    radial_scale = torch.linalg.vector_norm(metric) * torch.linalg.vector_norm(metric_pair)
    radial_error = float((torch.linalg.vector_norm(metric @ metric_pair) / radial_scale).item())
    missing_cross = omit_quotient_cross_terms_negative_control(metric)
    missing_cross_error = float(
        (torch.linalg.vector_norm(missing_cross @ metric_pair) / radial_scale).item()
    )

    zero = raw.new_zeros(1, 3, 2)
    naive_zero = evaluate_typed_pair_unscaled_negative_control(oracle.network, zero)
    scaled_zero = evaluate_scaled_typed_pair(oracle.network, zero)

    started = time.perf_counter()
    clone_oracle = make_oracle(seed + 100, tokens=2, dimension=1, rank=1)
    clone_canonical = canonicalize_typed_projective_dag(clone_oracle.network)
    reverse_small = reverse_typed_projective_environments(clone_canonical.network)
    explicit_small = explicit_clone_typed_environments(clone_canonical.network)
    aggregate_small = aggregate_clone_environments(explicit_small)
    reverse_small_by_identity = {
        record.node_identity: (record.gram, record.binary_exponent)
        for record in reverse_small
    }
    clone_environment_error = max(
        scaled_matrix_relative_error(reverse_small_by_identity[identity], expected)
        for identity, expected in aggregate_small.items()
    )
    timings["n2_d1_explicit_clone_comparison_seconds"] = time.perf_counter() - started

    shape = typed_dag_shape_statistics(oracle.network)
    basis_change = max(
        relative_error(diagonal.network.root.core, canonical.network.root.core),
        relative_error(diagonal.network.head, canonical.network.head),
    )
    row = {
        "seed": seed,
        "source_quotient_relative_error": relative_error(quotient, source),
        "canonical_pair_relative_error": scaled_projective_relative_error(
            canonical_pair, reference
        ),
        "full_rank_pair_relative_error": scaled_projective_relative_error(
            diagonal_pair, reference
        ),
        "hidden_shared_gauge_pair_relative_error": scaled_projective_relative_error(
            gauged_pair, reference
        ),
        "hidden_shared_gauge_recanonical_relative_error": scaled_projective_relative_error(
            evaluate_scaled_typed_pair(gauged_canonical.network, raw), reference
        ),
        "fixed_gauge_refix_pair_relative_error": scaled_projective_relative_error(
            evaluate_scaled_typed_pair(refixed, raw), reference
        ),
        "maximum_isometry_error": max(canonical.isometry_errors),
        "maximum_factorization_error": max(canonical.factorization_errors),
        "factorization_method_by_node": list(canonical.factorization_methods),
        "all_nodes_direct_reduced_rq": all(
            method == "direct_reduced_rq"
            for method in canonical.factorization_methods
        ),
        "maximum_diagonalization_error": max(diagonal.diagonalization_errors),
        "post_evd_recontracted_aggregate_offdiagonal_ratio": (
            post_evd_aggregate_offdiagonal
        ),
        "brute_first_role_relative_error": scaled_matrix_relative_error(recursive, brute),
        "explicit_clone_aggregate_relative_error": clone_environment_error,
        "missing_one_parent_inverse_relative_error": missing_parent_error,
        "initial_ffn_missing_R_leg_relative_error": initial_missing_leg_error,
        "missing_one_repeated_R_leg_relative_error": missing_leg_error,
        "quotient_radial_null_relative_error": radial_error,
        "missing_quotient_cross_terms_relative_error": missing_cross_error,
        "full_rank_basis_change": basis_change,
        "fixed_metadata_forgery_rejected": fixed_metadata_rejected,
        "indefinite_output_metric_rejected": indefinite_output_metric_rejected,
        "shared_gauge_parent_occurrences": shared_parent_occurrence_count(
            oracle.network, label
        ),
        "fixed_coefficient_norm": float(typed_pair_coefficient_norm(oracle.network).item()),
        "unfixed_log10_absolute_denominator_at_zero": (
            oracle.unfixed_log10_absolute_denominator_at_zero
        ),
        "old_anchor_refix_predicted_head_log10_maximum": (
            oracle.old_anchor_refix_predicted_head_log10_maximum
        ),
        "naive_zero_denominator_underflowed": bool(naive_zero[0, 1] == 0),
        "scaled_zero_is_finite": bool(torch.isfinite(scaled_zero.mantissa).all()),
        "scaled_zero_binary_exponent": int(scaled_zero.binary_exponent[0].item()),
        "pair_statistics": scaled_projective_statistics(reference),
        "shape": shape,
        "reverse_environment_count": len(environments),
        "n2_d1_explicit_clone_occurrence_count": len(explicit_small),
        "n2_d1_unique_environment_count": len(reverse_small),
        "timings": timings,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rank < 1:
        raise ValueError("--rank must be positive")
    row = run(args.seed, args.rank)
    gates = {
        "real_block_source_replay": row["source_quotient_relative_error"] < 1e-9,
        "shared_bottom_up_rq_replay": row["canonical_pair_relative_error"] < 3e-8,
        "shared_full_rank_evd_replay": row["full_rank_pair_relative_error"] < 4e-8,
        "row_isometry": row["maximum_isometry_error"] < 4e-10,
        "rq_factorization": row["maximum_factorization_error"] < 4e-10,
        "all_nodes_direct_reduced_rq": bool(row["all_nodes_direct_reduced_rq"]),
        "evd_diagonalization": row["maximum_diagonalization_error"] < 4e-10,
        "post_evd_recontracted_aggregate_is_diagonal": row[
            "post_evd_recontracted_aggregate_offdiagonal_ratio"
        ] < 4e-10,
        "reverse_environment_matches_brute": row["brute_first_role_relative_error"] < 1e-10,
        "reverse_shared_equals_explicit_clone_sum": row[
            "explicit_clone_aggregate_relative_error"
        ] < 3e-9,
        "hidden_shared_gauge_replay": row[
            "hidden_shared_gauge_pair_relative_error"
        ] < 1e-9,
        "hidden_shared_gauge_recanonicalization": row[
            "hidden_shared_gauge_recanonical_relative_error"
        ] < 4e-8,
        "every_parent_inverse_is_required": row[
            "missing_one_parent_inverse_relative_error"
        ] > 1e-5,
        "every_repeated_R_leg_is_required": row[
            "missing_one_repeated_R_leg_relative_error"
        ] > 1e-5,
        "fixed_gauge_refix": row["fixed_gauge_refix_pair_relative_error"] < 1e-10,
        "fixed_gauge_forgery_rejected": bool(row["fixed_metadata_forgery_rejected"]),
        "indefinite_output_metric_rejected": bool(
            row["indefinite_output_metric_rejected"]
        ),
        "quotient_radial_null": row["quotient_radial_null_relative_error"] < 1e-10,
        "quotient_cross_terms_required": row[
            "missing_quotient_cross_terms_relative_error"
        ] > 1e-4,
        "naive_underflow_scaled_away": bool(row["naive_zero_denominator_underflowed"])
        and bool(row["scaled_zero_is_finite"]),
        "unit_fixed_coefficient_norm": abs(row["fixed_coefficient_norm"] - 1.0) < 1e-10,
        "no_fused_hidden_bond": row["shape"]["maximum_local_bond_dimension"]
        < row["shape"]["fused_token_feature_dimension"],
        "shared_dag_not_clone_tree": row["n2_d1_explicit_clone_occurrence_count"]
        > 20 * row["n2_d1_unique_environment_count"],
        "full_rank_basis_absorbed": row["full_rank_basis_change"] > 1e-8,
    }
    result = {
        "schema": "xvla-typed-projective-dag-odt-v1",
        "object": {
            "tokens": 3,
            "width": 2,
            "heads": 1,
            "depth": 1,
            "ffn_rank": args.rank,
            "fixed_position": True,
            "pade_sites": 6,
            "residuals": "attention_and_ffn_preserved",
            "division_sites_inside_compiled_dag": 0,
            "numeric_chart": "nodewise_mantissa_plus_binary_exponent",
            "odt": "clone_unfolded_syntactic_metric_shared_RQ_aggregate_environment_full_rank_EVD",
            "metric_scope": "aggregate_occurrence_Gram_not_each_occurrence_Gram",
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "row": row,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    if not result["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
