#!/usr/bin/env python3
"""Run direct-RQ implicit sparse ODT at tiny or pinned production width."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Install the process-wide numerical guard before importing any model,
# compiler, or checkpoint helper.  The static audit follows this runner's full
# local import closure and records the exact source hashes in the result.
from scripts.odt_direct_only_compliance import (
    DIRECT_RQ_RUNTIME_CALL,
    PRODUCTION_ALGORITHM3_RUNTIME_CALL,
    PRODUCTION_CLONE_QR_RUNTIME_CALL,
    PRODUCTION_SHARED_CLONE_ALGORITHM3_RUNTIME_CALL,
    STREAMED_QR_RUNTIME_CALL,
    TRIANGULAR_RUNTIME_CALL,
    assert_direct_only_runtime_guard,
    audit_direct_only_launch,
    canonical_direct_only_entrypoints,
    direct_only_runtime_report,
    install_direct_only_runtime_guard,
    single_cli_option,
)

single_cli_option(sys.argv[1:], "--lane")
_TRANSITIVE_STATIC_AUDIT = audit_direct_only_launch(
    PROJECT_ROOT, canonical_direct_only_entrypoints(PROJECT_ROOT, Path(__file__))
)
_RUNTIME_GUARD_AT_IMPORT = install_direct_only_runtime_guard()

import torch
import torch.nn as nn

from scripts.run_production_factorized_odt_stress import (
    EXPECTED_CHECKPOINT_SHA256,
    production_config,
)
from xvla.models.vla import ChiVLA
from xvla.nn.block import ChiTransformerBlock
from xvla.nn.normalization import RationalNorm
from xvla.train.implicit_sparse_projective_odt import (
    BOUNDED_EXPLICIT_RQ_METHOD,
    CLAIM_BOUNDARY,
    DIRECT_Q_PROVENANCE_TOLERANCE,
    DIRECT_RQ_METHOD,
    STREAMED_DIRECT_Q_PROVENANCE_METHOD,
    UNARY_RQ_METHOD,
    assert_norm_buffers_unchanged,
    audit_algorithm1_factorization_calls,
    canonicalize_implicit_dag_direct_rq,
    canonicalize_with_explicit_clone_step_trace,
    compare_shared_and_explicit_clone_environments,
    compile_implicit_projective_block_boundary,
    diagonalize_implicit_dag_full_rank,
    diagonalize_shared_and_explicit_clone_independently,
    evaluate_boundary_quotient,
    implicit_shape_statistics,
    old_project_then_add_topology_is_rejected,
    reverse_implicit_environments,
    source_block_boundary,
    telemetry_dict,
    tiny_dense_direct_rq_equivalence,
)


DTYPE = torch.float64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = expected.norm().clamp_min(torch.finfo(expected.dtype).tiny)
    return float(((actual - expected).norm() / scale).item())


def _module_state_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _initialize_norms(block: ChiTransformerBlock) -> None:
    with torch.no_grad():
        for index, module in enumerate(
            item for item in block.modules() if isinstance(item, RationalNorm)
        ):
            module.running_ms.fill_(0.91 + 0.12 * index)
            module.initialized.fill_(True)
            module.frozen = True


def _tiny_oracle(seed: int):
    torch.manual_seed(seed)
    block = ChiTransformerBlock(
        dim=1,
        n_heads=1,
        ffn_rank=1,
        n_layers=1,
        causal=True,
        norm="rational",
        qk_norm="rational",
        residual=True,
    ).double().eval()
    generator = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                module.weight.copy_(
                    0.07
                    + 0.11
                    * torch.rand(module.weight.shape, generator=generator, dtype=DTYPE)
                )
                if module.bias is not None:
                    module.bias.copy_(
                        -0.05
                        + 0.10
                        * torch.rand(module.bias.shape, generator=generator, dtype=DTYPE)
                    )
    _initialize_norms(block)
    positions = torch.tensor(((-0.04,), (0.055,)), dtype=DTYPE)
    return compile_implicit_projective_block_boundary(block, positions)


def _tiny_inputs() -> torch.Tensor:
    return torch.tensor(
        ((-0.31, 0.14), (0.22, -0.27), (0.35, 0.19), (0.0, 0.0)),
        dtype=DTYPE,
    ).reshape(4, 2, 1)


def _algorithm1_summary(oracle, raw: torch.Tensor, *, block_size: int, clone: bool):
    source = source_block_boundary(oracle, raw)
    raw_quotient = evaluate_boundary_quotient(oracle.network, raw)
    started = time.perf_counter()
    canonical = canonicalize_implicit_dag_direct_rq(
        oracle.network,
        block_size=block_size,
        telemetry=oracle.telemetry,
    )
    canonical_seconds = time.perf_counter() - started
    canonical_quotient = evaluate_boundary_quotient(canonical.network, raw)
    steps = canonical.steps
    condition_proxies = tuple(step.diagonal_condition_proxy for step in steps)
    maximum_condition_proxy = max(condition_proxies)
    result = {
        "source_replay_relative_error": _relative(raw_quotient, source),
        "canonical_replay_relative_error": _relative(canonical_quotient, raw_quotient),
        "maximum_factorization_relative_error": max(
            step.factorization_relative_error for step in steps
        ),
        "minimum_diagonal_to_maximum_entry": min(
            step.minimum_diagonal_to_maximum_entry for step in steps
        ),
        "maximum_diagonal_condition_proxy": (
            maximum_condition_proxy
            if math.isfinite(maximum_condition_proxy)
            else None
        ),
        "nonfinite_diagonal_condition_proxy_steps": sum(
            not math.isfinite(value) for value in condition_proxies
        ),
        "all_nodes_full_row_rank": all(step.full_row_rank for step in steps),
        "all_streamed_nodes_full_row_rank": all(
            step.full_row_rank
            for step in steps
            if step.method == DIRECT_RQ_METHOD
        ),
        "bounded_retained_q_rank_deficient_steps": sum(
            step.method == BOUNDED_EXPLICIT_RQ_METHOD and not step.full_row_rank
            for step in steps
        ),
        "factorization_method_by_node": [step.method for step in steps],
        "all_nodes_direct_reduced_rq": all(
            step.method
            in {DIRECT_RQ_METHOD, UNARY_RQ_METHOD, BOUNDED_EXPLICIT_RQ_METHOD}
            for step in steps
        ),
        "shared_step_count": len(steps),
        "expected_parent_occurrence_pushes": sum(
            step.expected_parent_occurrences for step in steps
        ),
        "actual_parent_occurrence_pushes": sum(
            step.parent_occurrences_pushed for step in steps
        ),
        "all_parent_occurrence_pushes_complete": all(
            step.expected_parent_occurrences == step.parent_occurrences_pushed
            for step in steps
        ),
        "direct_q_provenance_verified_steps": sum(
            step.direct_q_provenance_verified for step in steps
        ),
        "direct_q_expected_columns": sum(
            int(step.unfolding_shape[1]) for step in steps
        ),
        "direct_q_columns_compared": sum(
            step.direct_q_columns_compared for step in steps
        ),
        "streamed_direct_q_steps": sum(
            step.method == DIRECT_RQ_METHOD for step in steps
        ),
        "streamed_direct_q_provenance_certificates": sum(
            step.direct_q_provenance_method
            == STREAMED_DIRECT_Q_PROVENANCE_METHOD
            for step in steps
        ),
        "expected_streamed_direct_q_replay_qr_calls": sum(
            step.direct_q_provenance_stage_count
            for step in steps
            if step.method == DIRECT_RQ_METHOD
        ),
        "maximum_direct_q_compact_relative_error": max(
            step.direct_q_compact_relative_error for step in steps
        ),
        "maximum_direct_q_reconstruction_relative_error": max(
            step.direct_q_reconstruction_relative_error for step in steps
        ),
        "all_direct_q_step_provenance_complete": all(
            step.direct_q_provenance_verified
            and step.direct_q_provenance_method
            and step.direct_q_columns_compared == int(step.unfolding_shape[1])
            and step.direct_q_compact_relative_error
            <= DIRECT_Q_PROVENANCE_TOLERANCE
            and step.direct_q_reconstruction_relative_error
            <= DIRECT_Q_PROVENANCE_TOLERANCE
            and step.direct_q_conditioning_accepted
            for step in steps
        ),
        "all_streamed_direct_q_step_ledgers_complete": all(
            step.direct_q_provenance_method
            == STREAMED_DIRECT_Q_PROVENANCE_METHOD
            and step.direct_q_provenance_stage_count
            == step.direct_q_provenance_column_blocks_compared
            == step.streamed_column_blocks
            and step.direct_q_minimum_pivot_to_maximum_entry
            > step.direct_q_conditioning_threshold
            and (step.direct_q_retained_transition_elements > 0)
            == (step.direct_q_provenance_stage_count > 1)
            for step in steps
            if step.method == DIRECT_RQ_METHOD
        ),
        "canonical_seconds": canonical_seconds,
        "shape": implicit_shape_statistics(oracle.network),
        "telemetry": telemetry_dict(oracle.telemetry),
    }
    trace_object = None
    if clone:
        started = time.perf_counter()
        dense = tiny_dense_direct_rq_equivalence(
            oracle.network, raw, block_size=block_size
        )
        trace = canonicalize_with_explicit_clone_step_trace(
            oracle.network, raw, block_size=block_size
        )
        trace_object = trace
        result["tiny_dense_equivalence"] = {
            key: getattr(dense, key) for key in dense.__dataclass_fields__
        }
        result["explicit_clone_trace"] = {
            "seconds": time.perf_counter() - started,
            "shared_steps": len(trace.records),
            "explicit_clone_factorizations": len(trace.clone_factorization_methods),
            "maximum_factor_relative_error": max(
                record.factor_relative_error for record in trace.records
            ),
            "maximum_q_core_relative_error": max(
                record.q_core_relative_error for record in trace.records
            ),
            "maximum_transformed_parent_occurrence_relative_error": max(
                record.transformed_parent_occurrence_relative_error
                for record in trace.records
            ),
            "maximum_supported_reconstruction_relative_error": max(
                record.supported_reconstruction_relative_error for record in trace.records
            ),
            "maximum_shared_partial_replay_relative_error": max(
                record.shared_function_replay_error for record in trace.records
            ),
            "maximum_clone_partial_replay_relative_error": max(
                record.clone_function_replay_error for record in trace.records
            ),
            "maximum_shared_clone_relative_error": max(
                record.shared_clone_function_relative_error for record in trace.records
            ),
            "maximum_boundary_head_relative_error": max(
                record.boundary_head_relative_error for record in trace.records
            ),
            "all_shared_parent_occurrences_pushed": all(
                record.expected_shared_parent_occurrences
                == record.pushed_shared_parent_occurrences
                for record in trace.records
            ),
            "all_clone_parent_occurrences_pushed": all(
                record.expected_clone_parent_occurrences
                == record.pushed_clone_parent_occurrences
                for record in trace.records
            ),
            "all_clone_factorizations_independent_dense_direct_rq": all(
                "independent_dense" in method
                for method in trace.clone_factorization_methods
            ),
            "literal_full_rank_RQ_comparisons": sum(
                record.literal_rq_occurrences_compared for record in trace.records
            ),
            "null_gauges_checked_by_reconstruction": sum(
                record.null_gauge_occurrences_checked_by_reconstruction
                for record in trace.records
            ),
        }
    return result, canonical, trace_object


def _load_production_oracle(args):
    if args.checkpoint is None:
        raise ValueError("production lanes require --checkpoint")
    checkpoint_sha = _sha256(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA-256 differs from the pinned capable checkpoint")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vocab_size = int(state["tok_emb.weight"].shape[0])
    model = ChiVLA(production_config(vocab_size))
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model = model.to(device=device, dtype=DTYPE).eval()
    if args.lane == "vision":
        block = model.vision.blocks.blocks[0]
        positions = model.vision.pos_emb[0]
        selected = 0
    else:
        block = model.backbone.blocks[0]
        positions = model.pos_emb[0]
        selected = positions.shape[0] - 1
    oracle = compile_implicit_projective_block_boundary(
        block,
        positions,
        selected_token=selected,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 9100)
    raw = torch.randn(
        args.samples,
        positions.shape[0],
        positions.shape[1],
        generator=generator,
        dtype=DTYPE,
        device=device,
    ) * 0.05
    return oracle, raw, checkpoint_sha


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--lane", choices=("tiny", "vision", "joint"), default="tiny")
    parser.add_argument("--stage", choices=("algorithm1", "full"), default="algorithm1")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.samples < 1 or args.block_size < 1:
        raise ValueError("samples and block size must be positive")

    factorization_audit = audit_algorithm1_factorization_calls()
    checkpoint_sha = None
    if args.lane == "tiny":
        oracle = _tiny_oracle(args.seed)
        raw = _tiny_inputs()
    else:
        oracle, raw, checkpoint_sha = _load_production_oracle(args)
    source_digest_before = _module_state_digest(oracle.block)
    assert_norm_buffers_unchanged(oracle)
    row, canonical, trace = _algorithm1_summary(
        oracle,
        raw,
        block_size=args.block_size,
        clone=args.lane == "tiny",
    )
    # This diagnostic intentionally enters the compact streamed chart and
    # fails its conditioning gate before the triangular solve.  The public
    # structural route retains direct Q for the same bounded deficient shape,
    # so compact-chart rejection is not a topology or exactness obstruction.
    row["old_project_then_add_compact_chart_rejection_diagnostic"] = (
        old_project_then_add_topology_is_rejected(
            12, 3, like=oracle.network.head
        )
    )
    exact_runtime_calls = {
        DIRECT_RQ_RUNTIME_CALL,
        STREAMED_QR_RUNTIME_CALL,
    }
    if row["streamed_direct_q_steps"]:
        exact_runtime_calls.add(TRIANGULAR_RUNTIME_CALL)
    if args.lane == "tiny":
        exact_runtime_calls.add(PRODUCTION_CLONE_QR_RUNTIME_CALL)
    source_digest_after_algorithm1 = _module_state_digest(oracle.block)
    telemetry = row["telemetry"]
    gates = {
        "source_replay": row["source_replay_relative_error"] < 2e-9,
        "canonical_replay": row["canonical_replay_relative_error"] < 3e-8,
        "direct_RQ_factorization": row["maximum_factorization_relative_error"] < 3e-9,
        "all_nodes_direct_reduced_rq": bool(row["all_nodes_direct_reduced_rq"]),
        "all_streamed_nodes_full_row_rank": bool(
            row["all_streamed_nodes_full_row_rank"]
        ),
        "all_production_parent_occurrences_pushed": bool(
            row["all_parent_occurrence_pushes_complete"]
        )
        and row["actual_parent_occurrence_pushes"]
        == row["shape"]["edge_occurrences"] + 1,
        "algorithm1_static_call_audit": not factorization_audit["prohibited_calls_found"],
        "transitive_direct_only_static_audit": not _TRANSITIVE_STATIC_AUDIT[
            "prohibited_calls_found"
        ],
        "one_local_direct_RQ_per_shared_node": telemetry[
            "local_direct_rq_factorizations"
        ]
        == row["shared_step_count"],
        "every_step_has_direct_q_provenance": bool(
            row["all_direct_q_step_provenance_complete"]
        )
        and row["direct_q_provenance_verified_steps"]
        == row["shared_step_count"]
        and row["direct_q_columns_compared"]
        == row["direct_q_expected_columns"],
        "streamed_direct_q_provenance_ledger": bool(
            row["all_streamed_direct_q_step_ledgers_complete"]
        )
        and row["streamed_direct_q_provenance_certificates"]
        == row["streamed_direct_q_steps"],
        "direct_q_telemetry_ledger": telemetry[
            "direct_q_provenance_certificates"
        ]
        == row["shared_step_count"]
        and telemetry["streamed_direct_q_provenance_certificates"]
        == row["streamed_direct_q_steps"]
        and telemetry["direct_q_columns_compared"]
        == row["direct_q_expected_columns"]
        and telemetry["streamed_direct_q_replay_qr_factorizations"]
        == row["expected_streamed_direct_q_replay_qr_calls"],
        "householder_kernel_call_ledger": telemetry[
            "householder_qr_kernel_calls"
        ]
        >= telemetry["local_direct_rq_factorizations"],
        "shared_network_contains_no_dense_clone_core": row["shape"][
            "dense_clone_nodes"
        ]
        == 0,
        "fused_cp_ffn_once": telemetry["fused_ffn_primitive_count"] == 1,
        "no_ephemeral_ffn_bond_diagonalization": telemetry[
            "ephemeral_ffn_bond_diagonalizations"
        ]
        == 0,
        "no_fused_token_feature_bond": row["shape"]["maximum_local_bond_dimension"]
        < row["shape"]["fused_token_feature_dimension"],
        "source_block_state_unchanged": source_digest_before
        == source_digest_after_algorithm1,
    }
    if args.lane == "tiny":
        dense = row["tiny_dense_equivalence"]
        trace_summary = row["explicit_clone_trace"]
        gates.update(
            {
                "tiny_streamed_matches_dense_direct_RQ": dense[
                    "maximum_factor_relative_error"
                ]
                < 3e-9
                and dense["maximum_q_core_relative_error"] < 3e-8,
                "tiny_dense_reconstruction": dense[
                    "maximum_reconstruction_relative_error"
                ]
                < 3e-9,
                "every_step_partial_replay": dense[
                    "maximum_partial_replay_relative_error"
                ]
                < 3e-8,
                "shared_matches_no_memo_clone_every_step": trace_summary[
                    "maximum_shared_clone_relative_error"
                ]
                < 3e-8
                and trace_summary["maximum_factor_relative_error"] < 3e-9
                and trace_summary["maximum_q_core_relative_error"] < 3e-8,
                "transformed_parent_occurrences_match": trace_summary[
                    "maximum_transformed_parent_occurrence_relative_error"
                ]
                < 3e-8,
                "all_R_parent_occurrences_pushed": trace_summary[
                    "all_shared_parent_occurrences_pushed"
                ]
                and trace_summary["all_clone_parent_occurrences_pushed"],
                "clone_oracle_factorization_is_independent_dense_QR": trace_summary[
                    "all_clone_factorizations_independent_dense_direct_rq"
                ],
                "clone_occurrence_expansion_is_real": trace_summary[
                    "explicit_clone_factorizations"
                ]
                > 20 * trace_summary["shared_steps"],
            }
        )
    if args.lane != "tiny":
        gates["pinned_capable_checkpoint"] = checkpoint_sha == EXPECTED_CHECKPOINT_SHA256

    if args.stage == "full":
        exact_runtime_calls.add(PRODUCTION_ALGORITHM3_RUNTIME_CALL)
        if args.lane == "tiny":
            exact_runtime_calls.add(
                PRODUCTION_SHARED_CLONE_ALGORITHM3_RUNTIME_CALL
            )
        started = time.perf_counter()
        environments = reverse_implicit_environments(canonical.network)
        algorithm2_seconds = time.perf_counter() - started
        started = time.perf_counter()
        diagonal = diagonalize_implicit_dag_full_rank(
            canonical.network, replay_inputs=raw
        )
        algorithm3_seconds = time.perf_counter() - started
        full = {
            "shared_algorithm2_environment_count": len(environments),
            "shared_algorithm2_child_messages_emitted": sum(
                record.child_messages_emitted for record in environments
            ),
            "shared_algorithm2_seconds": algorithm2_seconds,
            "shared_algorithm3_seconds": algorithm3_seconds,
            "shared_algorithm3_replay_relative_error": diagonal.replay_relative_error,
            "shared_algorithm3_recontracted_offdiagonal_ratio": (
                diagonal.maximum_recontracted_offdiagonal_ratio
            ),
            "shared_algorithm3_push_ledger": [
                diagonal.pushed_parent_occurrences,
                diagonal.expected_parent_occurrences,
            ],
        }
        gates["shared_algorithm2_covers_every_node"] = len(environments) == row[
            "shape"
        ]["unique_nodes"]
        gates["shared_algorithm2_emits_every_child_message"] = sum(
            record.child_messages_emitted for record in environments
        ) == row["shape"]["edge_occurrences"]
        gates["shared_algorithm3_replay"] = diagonal.replay_relative_error < 3e-8
        gates["shared_algorithm3_recontracted_diagonal"] = (
            diagonal.maximum_recontracted_offdiagonal_ratio < 3e-8
        )
        gates["shared_algorithm3_pushes_every_occurrence_and_root_head"] = (
            diagonal.pushed_parent_occurrences == diagonal.expected_parent_occurrences
            and diagonal.pushed_parent_occurrences
            == row["shape"]["edge_occurrences"] + 1
        )
        if args.lane == "tiny":
            if trace is None:
                raise RuntimeError("tiny full stage lost its explicit clone trace")
            comparison = compare_shared_and_explicit_clone_environments(
                trace.shared_network, trace.explicit_clone_network
            )
            independent = diagonalize_shared_and_explicit_clone_independently(
                trace.shared_network,
                trace.explicit_clone_network,
                raw,
            )
            missing = diagonalize_shared_and_explicit_clone_independently(
                trace.shared_network,
                trace.explicit_clone_network,
                raw,
                omit_clone_parent=("pre_attention.token0.projective_output", 0),
            )
            full.update(
                {
                    "algorithm2_shared_clone_aggregate_relative_error": comparison[
                        "maximum_aggregate_relative_error"
                    ],
                    "algorithm2_explicit_clone_environment_count": comparison[
                        "explicit_clone_environment_count"
                    ],
                    "algorithm3_independent_pre_environment_relative_error": (
                        independent.pre_evd_aggregate_environment_relative_error
                    ),
                    "algorithm3_independent_eigenvalue_relative_error": (
                        independent.eigenvalue_relative_error
                    ),
                    "algorithm3_independent_shared_replay_relative_error": (
                        independent.shared_replay_relative_error
                    ),
                    "algorithm3_independent_clone_replay_relative_error": (
                        independent.clone_replay_relative_error
                    ),
                    "algorithm3_independent_shared_clone_relative_error": (
                        independent.shared_clone_relative_error
                    ),
                    "algorithm3_independent_shared_offdiagonal_ratio": (
                        independent.maximum_shared_aggregate_offdiagonal_ratio
                    ),
                    "algorithm3_independent_clone_offdiagonal_ratio": (
                        independent.maximum_clone_aggregate_offdiagonal_ratio
                    ),
                    "algorithm3_shared_push_ledger": [
                        independent.pushed_shared_parent_occurrences,
                        independent.expected_shared_parent_occurrences,
                    ],
                    "algorithm3_clone_push_ledger": [
                        independent.pushed_clone_parent_occurrences,
                        independent.expected_clone_parent_occurrences,
                    ],
                    "algorithm3_missing_occurrence_clone_replay_relative_error": (
                        missing.clone_replay_relative_error
                    ),
                    "algorithm3_missing_occurrence_count": missing.omitted_clone_occurrences,
                    "algorithm3_targeted_reused_clone_parent_occurrences": (
                        missing.targeted_clone_parent_occurrences
                    ),
                }
            )
            gates.update(
                {
                    "algorithm2_shared_equals_independent_clone_aggregate": comparison[
                        "maximum_aggregate_relative_error"
                    ]
                    < 3e-8,
                    "algorithm3_independent_eigenvalues_match": independent.eigenvalue_relative_error
                    < 3e-8,
                    "algorithm3_independent_shared_and_clone_replay": max(
                        independent.shared_replay_relative_error,
                        independent.clone_replay_relative_error,
                        independent.shared_clone_relative_error,
                    )
                    < 3e-8,
                    "algorithm3_independent_recontracted_aggregates_diagonal": max(
                        independent.maximum_shared_aggregate_offdiagonal_ratio,
                        independent.maximum_clone_aggregate_offdiagonal_ratio,
                    )
                    < 3e-8,
                    "algorithm3_every_shared_and_clone_occurrence_pushed": (
                        independent.pushed_shared_parent_occurrences
                        == independent.expected_shared_parent_occurrences
                        and independent.pushed_clone_parent_occurrences
                        == independent.expected_clone_parent_occurrences
                    ),
                    "algorithm3_missing_one_clone_occurrence_is_load_bearing": (
                        missing.omitted_clone_occurrences == 1
                        and missing.targeted_clone_parent_occurrences > 1
                        and missing.pushed_clone_parent_occurrences
                        == missing.expected_clone_parent_occurrences - 1
                        and missing.clone_replay_relative_error > 1e-6
                    ),
                }
            )
        row["full_algorithms"] = full

    source_digest_after_all = _module_state_digest(oracle.block)
    gates["source_block_state_unchanged_after_all_algorithms"] = (
        source_digest_before == source_digest_after_all
    )
    runtime_compliance = assert_direct_only_runtime_guard(
        direct_only_runtime_report(),
        exact_allowed_calls=exact_runtime_calls,
    )
    gates["transitive_direct_only_runtime_audit"] = True

    result = {
        "schema": "xvla-implicit-sparse-direct-rq-odt-v1",
        "stage": (
            "Algorithm1 per-step trace"
            if args.stage == "algorithm1"
            else "Algorithms1-3 direct-RQ explicit-contraction full-rank-EVD trace"
        ),
        "lane": args.lane,
        "checkpoint_sha256": checkpoint_sha,
        "claim_boundary": CLAIM_BOUNDARY,
        "factorization_audit": factorization_audit,
        "transitive_static_audit": _TRANSITIVE_STATIC_AUDIT,
        "runtime_direct_only_guard_at_import": _RUNTIME_GUARD_AT_IMPORT,
        "runtime_direct_only_guard_final": runtime_compliance,
        "source_block_state_sha256_before": source_digest_before,
        "source_block_state_sha256_after": source_digest_after_all,
        "row": row,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "peak_rss_mb": _peak_rss_mb(),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n")
    print(encoded)
    if not result["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
