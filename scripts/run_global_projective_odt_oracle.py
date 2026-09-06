#!/usr/bin/env python3
"""Run the exact global Padé residual-attention projective ODT oracle."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from xvla.nn.block import ChiTransformerBlock
from xvla.nn.normalization import RationalNorm
from xvla.train.global_projective_odt_oracle import (
    CLAIM_BOUNDARY,
    assert_norm_buffers_unchanged,
    brute_identity_role_environment,
    canonical_projective_environments,
    canonicalize_projective_tree,
    compile_global_projective_block,
    diagonalize_projective_tree_full_rank,
    drop_first_projective_add_cross_term,
    evaluate_projective_pair_unscaled_negative_control,
    evaluate_quotient,
    evaluate_scaled_projective_pair,
    fix_projective_gauge,
    gauge_tree_occurrence,
    node_at_path,
    omit_quotient_cross_terms_negative_control,
    quotient_pair_output_metric,
    rescale_projective_output,
    scale_symmetric_message,
    scaled_projective_relative_error,
    scaled_projective_statistics,
    source_action,
)


DTYPE = torch.float64


def make_oracle(seed: int, rank: int):
    torch.manual_seed(seed)
    block = ChiTransformerBlock(
        dim=1,
        n_heads=1,
        ffn_rank=rank,
        n_layers=1,
        causal=True,
        norm="rational",
        qk_norm="rational",
        residual=True,
    ).double().eval()
    action_head = nn.Linear(1, 1, bias=True).double().eval()
    generator = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                module.weight.copy_(
                    0.08
                    + 0.16
                    * torch.rand(module.weight.shape, generator=generator, dtype=DTYPE)
                )
                if module.bias is not None:
                    module.bias.copy_(
                        -0.12
                        + 0.24
                        * torch.rand(module.bias.shape, generator=generator, dtype=DTYPE)
                    )
                    module.bias.add_(torch.where(module.bias >= 0, 0.04, -0.04))
        action_head.weight.fill_(0.31)
        action_head.bias.fill_(-0.17)
        for index, norm in enumerate(
            module for module in block.modules() if isinstance(module, RationalNorm)
        ):
            norm.running_ms.fill_(1.05 + 0.07 * index)
            norm.initialized.fill_(True)
            norm.frozen = True
    return compile_global_projective_block(block, action_head)


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)
    return float((torch.linalg.vector_norm(actual - expected) / scale).item())


def run_seed(seed: int, rank: int) -> dict[str, object]:
    raw = torch.tensor(
        [[-0.35, 0.20], [0.10, -0.25], [0.30, 0.40], [-0.15, -0.10], [0.0, 0.0]],
        dtype=DTYPE,
    )
    timings: dict[str, float] = {}
    started = time.perf_counter()
    oracle = make_oracle(seed, rank)
    timings["compile_seconds"] = time.perf_counter() - started
    assert_norm_buffers_unchanged(oracle)

    source = source_action(oracle, raw)
    pair = evaluate_scaled_projective_pair(oracle.network, raw)
    pair_statistics = scaled_projective_statistics(pair)
    quotient = evaluate_quotient(oracle.network, raw)
    source_error = relative_error(quotient, source)

    started = time.perf_counter()
    canonical = canonicalize_projective_tree(oracle.network)
    timings["bottom_up_rq_seconds"] = time.perf_counter() - started
    canonical_pair = evaluate_scaled_projective_pair(canonical.network, raw)

    started = time.perf_counter()
    environments = canonical_projective_environments(canonical.network)
    timings["top_down_environment_seconds"] = time.perf_counter() - started
    paths_unique = len({record.path for record in environments}) == len(environments)
    root_node = node_at_path(canonical.network, ())
    root_record = next(record for record in environments if record.path == ())
    brute_unscaled = brute_identity_role_environment(root_node.core, root_record.gram, 0)
    brute_first, brute_first_exponent = scale_symmetric_message(
        brute_unscaled, root_record.binary_exponent
    )
    recursive_first = next(record for record in environments if record.path == (0,))
    brute_environment_error = relative_error(recursive_first.gram, brute_first)
    brute_environment_exponent_match = (
        recursive_first.binary_exponent == brute_first_exponent
    )

    started = time.perf_counter()
    diagonal = diagonalize_projective_tree_full_rank(canonical.network)
    timings["full_rank_evd_absorption_seconds"] = time.perf_counter() - started
    diagonal_pair = evaluate_scaled_projective_pair(diagonal.network, raw)
    root_core_change = relative_error(diagonal.network.root.core, canonical.network.root.core)
    root_head_change = relative_error(diagonal.network.head, canonical.network.head)

    transform = torch.tensor(((1.3, 0.25), (-0.15, 0.8)), dtype=DTYPE)
    gauged = gauge_tree_occurrence(oracle.network, (0, 0), transform)
    hidden_gauge_pair_error = scaled_projective_relative_error(
        evaluate_scaled_projective_pair(gauged, raw), pair
    )
    gauged_canonical = canonicalize_projective_tree(gauged)
    hidden_gauge_canonical_error = scaled_projective_relative_error(
        evaluate_scaled_projective_pair(gauged_canonical.network, raw), pair
    )

    scaled = rescale_projective_output(oracle.network, -7.25)
    refixed = fix_projective_gauge(scaled)
    metric_pair = pair.mantissa[2]
    metric = quotient_pair_output_metric(metric_pair)
    radial_scale = (
        torch.linalg.vector_norm(metric)
        * torch.linalg.vector_norm(metric_pair)
    ).clamp_min(torch.finfo(DTYPE).tiny)
    radial_residual = float(
        (torch.linalg.vector_norm(metric @ metric_pair) / radial_scale).item()
    )
    missing_cross = omit_quotient_cross_terms_negative_control(metric)
    missing_cross_residual = float(
        (torch.linalg.vector_norm(missing_cross @ metric_pair) / radial_scale).item()
    )
    broken = drop_first_projective_add_cross_term(
        oracle.network, label_contains="ffn.residual_add"
    )
    broken_error = relative_error(evaluate_quotient(broken, raw), quotient)
    zero = raw.new_zeros(1, raw.shape[1])
    naive_zero_pair = evaluate_projective_pair_unscaled_negative_control(
        oracle.network, zero
    )
    scaled_zero_pair = evaluate_scaled_projective_pair(oracle.network, zero)

    return {
        "seed": seed,
        "source_quotient_relative_error": source_error,
        "canonical_pair_relative_error": scaled_projective_relative_error(
            canonical_pair, pair
        ),
        "full_rank_diagonal_pair_relative_error": scaled_projective_relative_error(
            diagonal_pair, pair
        ),
        "fixed_gauge_rescale_pair_relative_error": scaled_projective_relative_error(
            evaluate_scaled_projective_pair(refixed, raw), pair
        ),
        "maximum_isometry_error": max(canonical.isometry_errors),
        "maximum_rq_factorization_error": max(canonical.factorization_errors),
        "factorization_method_by_node": list(canonical.factorization_methods),
        "all_nodes_direct_reduced_rq": all(
            method == "direct_reduced_rq"
            for method in canonical.factorization_methods
        ),
        "maximum_evd_diagonalization_error": max(diagonal.diagonalization_errors),
        "brute_environment_relative_error": brute_environment_error,
        "brute_environment_exponent_match": brute_environment_exponent_match,
        "hidden_gauge_pair_relative_error": hidden_gauge_pair_error,
        "hidden_gauge_canonical_relative_error": hidden_gauge_canonical_error,
        "root_core_full_rank_basis_change": root_core_change,
        "root_head_full_rank_basis_change": root_head_change,
        "unique_node_count": canonical.unique_node_count,
        "clone_unfolded_occurrence_count": canonical.occurrence_count,
        "environment_count": len(environments),
        "environment_paths_unique": paths_unique,
        "environment_binary_exponent_minimum": min(
            record.binary_exponent for record in environments
        ),
        "environment_binary_exponent_maximum": max(
            record.binary_exponent for record in environments
        ),
        "unfixed_log10_absolute_denominator_at_zero": (
            oracle.unfixed_log10_absolute_denominator_at_zero
        ),
        "old_anchor_refix_predicted_head_log10_maximum": (
            oracle.old_anchor_refix_predicted_head_log10_maximum
        ),
        "fixed_head_maximum_absolute": oracle.fixed_head_maximum_absolute,
        "fixed_coefficient_norm": oracle.fixed_coefficient_norm,
        "pair_statistics": pair_statistics,
        "r3_naive_zero_denominator_underflowed": bool(
            naive_zero_pair[0, 1] == 0
        ),
        "scaled_zero_pair_is_finite": bool(
            torch.isfinite(scaled_zero_pair.mantissa).all()
        ),
        "scaled_zero_pair_binary_exponent": int(
            scaled_zero_pair.binary_exponent[0].item()
        ),
        "quotient_radial_null_relative_error": radial_residual,
        "missing_cross_term_radial_relative_error": missing_cross_residual,
        "dropped_residual_cross_product_relative_error": broken_error,
        "timings": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seeds <= 0 or args.rank <= 0:
        raise ValueError("--seeds and --rank must be positive")
    rows = [run_seed(seed, args.rank) for seed in range(args.seeds)]

    maxima = {
        key: max(float(row[key]) for row in rows)
        for key in (
            "source_quotient_relative_error",
            "canonical_pair_relative_error",
            "full_rank_diagonal_pair_relative_error",
            "fixed_gauge_rescale_pair_relative_error",
            "maximum_isometry_error",
            "maximum_rq_factorization_error",
            "maximum_evd_diagonalization_error",
            "brute_environment_relative_error",
            "hidden_gauge_pair_relative_error",
            "hidden_gauge_canonical_relative_error",
            "quotient_radial_null_relative_error",
        )
    }
    minima = {
        key: min(float(row[key]) for row in rows)
        for key in (
            "missing_cross_term_radial_relative_error",
            "dropped_residual_cross_product_relative_error",
        )
    }
    gates = {
        "global_source_replay": maxima["source_quotient_relative_error"] < 1e-9,
        "bottom_up_rq_replay": maxima["canonical_pair_relative_error"] < 1e-8,
        "full_rank_evd_replay": maxima["full_rank_diagonal_pair_relative_error"] < 3e-8,
        "row_isometry": maxima["maximum_isometry_error"] < 3e-10,
        "rq_factorization": maxima["maximum_rq_factorization_error"] < 3e-10,
        "all_nodes_direct_reduced_rq": all(
            bool(row["all_nodes_direct_reduced_rq"]) for row in rows
        ),
        "evd_diagonalization": maxima["maximum_evd_diagonalization_error"] < 3e-10,
        "fixed_projective_gauge": maxima["fixed_gauge_rescale_pair_relative_error"] < 1e-10,
        "hidden_nonorthogonal_gauge": maxima["hidden_gauge_pair_relative_error"] < 1e-9,
        "hidden_gauge_recanonicalization": maxima[
            "hidden_gauge_canonical_relative_error"
        ] < 2e-8,
        "recursive_environment_matches_brute": maxima[
            "brute_environment_relative_error"
        ] < 1e-10
        and all(bool(row["brute_environment_exponent_match"]) for row in rows),
        "quotient_radial_null": maxima["quotient_radial_null_relative_error"] < 1e-10,
        "cross_terms_are_load_bearing": minima["missing_cross_term_radial_relative_error"] > 1e-4,
        "residual_cross_product_is_load_bearing": minima[
            "dropped_residual_cross_product_relative_error"
        ] > 1e-6,
        "all_denominators_observed_nonzero": min(
            float(row["pair_statistics"]["minimum_relative_denominator"])
            for row in rows
        ) > 1e-12,
        "r3_underflow_reproduced_and_scaled_away": all(
            bool(row["r3_naive_zero_denominator_underflowed"])
            and bool(row["scaled_zero_pair_is_finite"])
            for row in rows
        ),
        "fixed_coefficient_gauge_is_unit": max(
            abs(float(row["fixed_coefficient_norm"]) - 1.0) for row in rows
        ) < 1e-10,
        "environment_covers_clone_unfolding": all(
            row["environment_count"] == row["clone_unfolded_occurrence_count"]
            for row in rows
        ),
        "environment_paths_are_unique": all(bool(row["environment_paths_unique"]) for row in rows),
        "full_rank_evd_basis_was_absorbed": min(
            max(
                float(row["root_core_full_rank_basis_change"]),
                float(row["root_head_full_rank_basis_change"]),
            )
            for row in rows
        ) > 1e-8,
    }
    result = {
        "schema": "xvla-global-projective-odt-oracle-v2-scaled-messages",
        "object": {
            "tokens": 2,
            "width": 1,
            "heads": 1,
            "depth": 1,
            "ffn_rank": args.rank,
            "normalization": "deployed_pade_rational_at_pre_attention_q1_k1_q2_k2_pre_ffn",
            "residuals": "attention_and_ffn_preserved",
            "division_sites_inside_compiled_tree": 0,
            "final_decode": "numerator_over_denominator",
            "fixed_projective_gauge": "unit_coefficient_norm_and_D_at_zero_positive",
            "numeric_message_chart": "mantissa_plus_binary_exponent",
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "dtype": "float64",
        "seeds": args.seeds,
        "maxima": maxima,
        "minima": minima,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "rows": rows,
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
