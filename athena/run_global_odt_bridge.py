#!/usr/bin/env python3
"""Run exact tiny experiments at the attention-to-action global ODT boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
from pathlib import Path
from statistics import mean, median

import torch

from xvla.train.canonical_odt import (
    apply_bond_gauge,
    canonical_environments,
    canonicalize_homogeneous,
    explicit_bond_gram,
    explicit_downstream_tensor,
    materialize_tree_coefficient,
    sorted_eigensystem,
)
from xvla.train.global_odt_bridge import (
    ATTENTION_LEGS,
    attention_coefficient_embedding,
    attention_occurrence_gram,
    build_bilinear_attention_core,
    coefficient_occurrence_gram,
    evaluate_attention_core,
    make_attention_action_tree,
    shared_attention_gram,
    symmetrize_tied_attention_core,
    trace_before_square_source_gram,
    verify_one_occurrence_tail,
)


DTYPE = torch.float64


def random_attention_weights(seed: int, dimension: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    scale = dimension**-0.5
    return {
        name: scale * torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
        for name in ("Wq1", "Wk1", "Wq2", "Wk2", "Wv", "Wo")
    }


def cp_action_core(seed: int, dimension: int, actions: int, rank: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(10_000 + seed)
    left = torch.randn(rank, dimension, generator=generator, dtype=DTYPE) / math.sqrt(dimension)
    right = torch.randn(rank, dimension, generator=generator, dtype=DTYPE) / math.sqrt(dimension)
    down = torch.randn(actions, rank, generator=generator, dtype=DTYPE) / math.sqrt(rank)
    core = torch.einsum("ar,ri,rj->aij", down, left, right)
    return 0.5 * (core + core.transpose(1, 2))


def project_axis(coefficient: torch.Tensor, projector: torch.Tensor, axis: int) -> torch.Tensor:
    projected = torch.tensordot(projector, coefficient, dims=([1], [axis]))
    return projected.movedim(0, axis)


def top_projector(gram: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    values, vectors = sorted_eigensystem(gram)
    basis = vectors[:, :rank]
    return basis @ basis.T, basis


def relative_frobenius_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)
    return float((torch.linalg.vector_norm(actual - expected) / denominator).item())


def projector_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    rank = left.shape[1]
    return float(torch.linalg.matrix_norm(left.T @ right).square().item() / rank)


def cancellation_fixture() -> dict[str, float]:
    core = torch.zeros(1, 2, 2, 2, 2, 2, dtype=DTYPE)
    core[0, 0, 0, 0, 0, 0] = 1.0
    core[0, 1, 1, 0, 0, 0] = -1.0
    historical = trace_before_square_source_gram(core)
    ordinary = attention_occurrence_gram(core, "key_1")
    return {
        "trace_before_square_energy": float(torch.trace(historical).item()),
        "ordinary_mode_energy": float(torch.trace(ordinary).item()),
    }


def run_seed(
    seed: int,
    dimension: int,
    actions: int,
    rank: int,
    attention_components: int,
) -> dict[str, object]:
    typed = torch.stack(
        [
            build_bilinear_attention_core(
                random_attention_weights(seed * attention_components + component, dimension)
            )
            for component in range(attention_components)
        ]
    ).sum(dim=0)
    tied = symmetrize_tied_attention_core(typed)
    embedding = attention_coefficient_embedding(
        typed, scale=dimension**-2, include_tensor_unit=False
    )
    action_core = cp_action_core(seed, dimension, actions, rank)
    network = make_attention_action_tree(embedding, action_core)

    generator = torch.Generator().manual_seed(20_000 + seed)
    max_typed_tied_function_error = 0.0
    for _ in range(16):
        query = torch.randn(dimension, generator=generator, dtype=DTYPE)
        source = torch.randn(dimension, generator=generator, dtype=DTYPE)
        difference = evaluate_attention_core(typed, query, source) - evaluate_attention_core(
            tied, query, source
        )
        max_typed_tied_function_error = max(
            max_typed_tied_function_error, float(difference.abs().max().item())
        )

    raw_coefficient = materialize_tree_coefficient(network)
    canonical = canonicalize_homogeneous(network)
    canonical_coefficient = materialize_tree_coefficient(canonical.network)
    global_hidden = canonical_environments(canonical.network)[0]
    explicit_hidden = explicit_bond_gram(canonical.network, 0)
    hidden_values, hidden_vectors = sorted_eigensystem(global_hidden)
    hidden_cut_tensor = explicit_downstream_tensor(canonical.network, 0)
    tail = verify_one_occurrence_tail(
        hidden_cut_tensor, global_hidden, hidden_vectors[:, : max(1, dimension // 2)]
    )

    left, _ = torch.linalg.qr(
        torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
    )
    right, _ = torch.linalg.qr(
        torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
    )
    scales = torch.linspace(0.55, 1.8, dimension, dtype=DTYPE)
    gauge = left @ torch.diag(scales) @ right.T
    gauged = apply_bond_gauge(network, 0, gauge)
    gauged_coefficient = materialize_tree_coefficient(gauged)
    gauged_canonical = canonicalize_homogeneous(gauged).network
    gauged_values, _ = sorted_eigensystem(canonical_environments(gauged_canonical)[0])

    input_dimension = typed.shape[1]
    full_typed = raw_coefficient.reshape((actions,) + (input_dimension,) * 10)
    source_axis = 3
    exact_source = coefficient_occurrence_gram(full_typed, source_axis)
    local_typed = attention_occurrence_gram(typed, "key_1")
    local_tied = attention_occurrence_gram(tied, "key_1")
    historical = trace_before_square_source_gram(tied)

    cut_rows: dict[str, dict[str, float]] = {}
    exact_projectors: dict[int, torch.Tensor] = {}
    # The bias-free core has zero support on the homogeneous coordinate, so
    # ranks at or above ``dimension`` are numerically full rank and are not a
    # meaningful comparison between selectors.
    for kept_rank in range(1, dimension):
        exact_projector, exact_basis = top_projector(exact_source, kept_rank)
        exact_projectors[kept_rank] = exact_projector
        candidate_rows = {}
        for name, gram in (
            ("global", exact_source),
            ("local_typed", local_typed),
            ("local_tied", local_tied),
            ("trace_before_square", historical),
        ):
            projector, basis = top_projector(gram, kept_rank)
            approximation = project_axis(full_typed, projector, source_axis)
            error = torch.linalg.vector_norm(full_typed - approximation).square()
            candidate_rows[name] = {
                "squared_error": float(error.item()),
                "relative_squared_error": float(
                    (error / torch.linalg.vector_norm(full_typed).square()).item()
                ),
                "subspace_similarity_to_global": projector_similarity(exact_basis, basis),
            }
        cut_rows[str(kept_rank)] = candidate_rows

    source_legs = ("key_1", "key_2", "value")
    shared = shared_attention_gram(typed, source_legs)
    shared_rank = max(1, dimension - 1)
    shared_projector, _ = top_projector(shared, shared_rank)
    simultaneous = typed
    for leg in source_legs:
        simultaneous = project_axis(
            simultaneous, shared_projector, 1 + ATTENTION_LEGS.index(leg)
        )
    simultaneous_error = torch.linalg.vector_norm(typed - simultaneous).square()
    separate_upper = typed.new_zeros(())
    for leg in source_legs:
        separately_projected = project_axis(
            typed, shared_projector, 1 + ATTENTION_LEGS.index(leg)
        )
        separate_upper += torch.linalg.vector_norm(typed - separately_projected).square()
    bound_tolerance = 1e-10 * max(float(separate_upper.item()), 1.0)
    if float(simultaneous_error.item()) > float(separate_upper.item()) + bound_tolerance:
        raise AssertionError(
            "simultaneous shared projection exceeded the separate-occurrence certificate"
        )

    typed_source_values, typed_source_vectors = sorted_eigensystem(local_typed)
    tied_source_values, tied_source_vectors = sorted_eigensystem(local_tied)
    object_rank = 1
    typed_object_basis = typed_source_vectors[:, :object_rank]
    tied_object_basis = tied_source_vectors[:, :object_rank]

    return {
        "seed": seed,
        "attention_components": attention_components,
        "core_shape": list(typed.shape),
        "typed_vs_tied": {
            "maximum_diagonal_function_error": max_typed_tied_function_error,
            "relative_coefficient_difference": relative_frobenius_error(tied, typed),
            "source_spectrum_typed": typed_source_values.tolist(),
            "source_spectrum_tied": tied_source_values.tolist(),
            "source_subspace_similarity": projector_similarity(
                typed_object_basis, tied_object_basis
            ),
        },
        "global_hidden_odt": {
            "coefficient_reconstruction_relative_error": relative_frobenius_error(
                canonical_coefficient, raw_coefficient
            ),
            "recursive_vs_explicit_gram_relative_error": relative_frobenius_error(
                global_hidden, explicit_hidden
            ),
            "maximum_isometry_error": max(canonical.isometry_errors),
            "maximum_factorization_error": max(canonical.factorization_errors),
            "one_occurrence_tail_relative_difference": tail.relative_difference,
            "spectrum": hidden_values.tolist(),
            "gauge_coefficient_relative_error": relative_frobenius_error(
                gauged_coefficient, raw_coefficient
            ),
            "gauge_spectrum_relative_error": relative_frobenius_error(
                gauged_values, hidden_values
            ),
        },
        "same_source_cut": cut_rows,
        "shared_projector": {
            "rank": shared_rank,
            "simultaneous_squared_error": float(simultaneous_error.item()),
            "separate_error_upper_bound": float(separate_upper.item()),
            "simultaneous_over_upper_bound": float(
                (simultaneous_error / separate_upper.clamp_min(torch.finfo(DTYPE).tiny)).item()
            ),
        },
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    hidden_fields = (
        "coefficient_reconstruction_relative_error",
        "recursive_vs_explicit_gram_relative_error",
        "maximum_isometry_error",
        "maximum_factorization_error",
        "one_occurrence_tail_relative_difference",
        "gauge_coefficient_relative_error",
        "gauge_spectrum_relative_error",
    )
    exactness = {
        field: max(float(row["global_hidden_odt"][field]) for row in rows)
        for field in hidden_fields
    }
    ranks = rows[0]["same_source_cut"].keys()
    same_cut: dict[str, object] = {}
    for kept_rank in ranks:
        candidates = rows[0]["same_source_cut"][kept_rank].keys()
        rank_row: dict[str, object] = {}
        for candidate in candidates:
            relative_errors = [
                float(row["same_source_cut"][kept_rank][candidate]["relative_squared_error"])
                for row in rows
            ]
            similarities = [
                float(row["same_source_cut"][kept_rank][candidate]["subspace_similarity_to_global"])
                for row in rows
            ]
            global_errors = [
                float(row["same_source_cut"][kept_rank]["global"]["squared_error"])
                for row in rows
            ]
            errors = [
                float(row["same_source_cut"][kept_rank][candidate]["squared_error"])
                for row in rows
            ]
            ratios = [value / max(reference, 1e-300) for value, reference in zip(errors, global_errors)]
            rank_row[candidate] = {
                "mean_relative_squared_error": mean(relative_errors),
                "median_relative_squared_error": median(relative_errors),
                "mean_subspace_similarity_to_global": mean(similarities),
                "median_error_over_global": median(ratios),
                "fraction_strictly_worse_than_global": mean(
                    float(value > reference * (1.0 + 1e-10))
                    for value, reference in zip(errors, global_errors)
                ),
            }
        same_cut[kept_rank] = rank_row

    return {
        "maximum_exactness_errors": exactness,
        "same_source_cut": same_cut,
        "typed_vs_tied": {
            "maximum_diagonal_function_error": max(
                float(row["typed_vs_tied"]["maximum_diagonal_function_error"])
                for row in rows
            ),
            "median_relative_coefficient_difference": median(
                float(row["typed_vs_tied"]["relative_coefficient_difference"])
                for row in rows
            ),
            "median_source_subspace_similarity": median(
                float(row["typed_vs_tied"]["source_subspace_similarity"])
                for row in rows
            ),
        },
        "shared_projector": {
            "maximum_simultaneous_over_upper_bound": max(
                float(row["shared_projector"]["simultaneous_over_upper_bound"])
                for row in rows
            ),
            "median_simultaneous_over_upper_bound": median(
                float(row["shared_projector"]["simultaneous_over_upper_bound"])
                for row in rows
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--actions", type=int, default=3)
    parser.add_argument("--cp-rank", type=int, default=6)
    parser.add_argument("--attention-components", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("athena/results/global_odt_bridge_v1.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result {args.output}")
    if (
        args.seeds < 1
        or args.dimension < 2
        or args.actions < 1
        or args.cp_rank < 1
        or args.attention_components < 1
    ):
        raise ValueError(
            "seeds, actions, cp-rank, and attention-components must be positive, "
            "and dimension at least two"
        )

    rows = [
        run_seed(
            seed,
            args.dimension,
            args.actions,
            args.cp_rank,
            args.attention_components,
        )
        for seed in range(args.seeds)
    ]
    manifest_path = Path("athena/global_odt_bridge_sources.sha256")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    payload = {
        "schema": "xvla-global-odt-bridge-v1",
        "configuration": {
            "seeds": args.seeds,
            "dimension": args.dimension,
            "homogeneous_input_dimension": args.dimension + 1,
            "actions": args.actions,
            "action_cp_rank": args.cp_rank,
            "attention_components": args.attention_components,
            "dtype": "float64",
        },
        "provenance": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source_manifest": manifest_path.as_posix(),
            "source_manifest_sha256": manifest_sha256,
            "xvla_python_tree_sha256": os.environ.get("XVLA_SOURCE_TREE_SHA256"),
            "pip_freeze_sha256": os.environ.get("XVLA_PIP_FREEZE_SHA256"),
        },
        "cancellation_fixture": cancellation_fixture(),
        "summary": summarize(rows),
        "candidate_object_notes": {
            "global": (
                "Exact ordinary mode Gram at one named K1 occurrence of the complete ordered "
                "attention-to-action topology tensor."
            ),
            "local_typed": (
                "Ordinary K1 mode Gram of the same summed ordered typed attention contribution, "
                "evaluated using the global ordered topology-tensor error."
            ),
            "local_tied": (
                "K1 mode Gram of the tied symmetrized polynomial quotient, transferred into the "
                "ordered topology-tensor objective. This is an object-compatibility diagnostic."
            ),
            "trace_before_square": (
                "Historical query-traced marginal-response Gram of the tied quotient, transferred "
                "into the ordered topology-tensor objective. This is an object-compatibility diagnostic."
            ),
        },
        "per_seed": rows,
        "claim_boundary": (
            "This is exact global Dooms-style ODT for a declared one-core tree whose embedding "
            "is one bias-free query-source contribution formed as a sum of typed bilinear-attention "
            "components and whose root is a bilinear action readout. It has no source sum, causal "
            "mask, normalization, or residual. It is not global ODT of the shared-state VLA."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
