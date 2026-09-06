"""Optional reports and source audits, outside the mathematical sweep."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from xvla.train.odt_engine_v2.constants import (
    COMPACT_RELATIVE_VALUE_FLOORS,
    COMPACT_TRACE_RETENTION_TARGETS,
    MAXIMUM_BOUNDED_EXPLICIT_Q_ELEMENTS,
    MAXIMUM_BOUNDED_EXPLICIT_UNFOLDING_ELEMENTS,
)

from xvla.train.odt_engine_v2.factorization import (
    _bounded_explicit_q_allowed_shape,
)

from xvla.train.odt_engine_v2.graph import (
    _walk_unique,
)

from xvla.train.odt_engine_v2.types import (
    CPBinaryCore,
    CanonicalStepRecord,
    CompactSpectrumRecord,
    DenseCloneCore,
    ImplicitProjectiveDAG,
    MaterializationTelemetry,
    ReducedQRBinaryCore,
    Tensor,
    UnaryCore,
)


def _canonical_step_record(
    step, node, source, factor, diagnostics, absorption, expected,
    local_error, exponent_delta, replay_error, replayed,
) -> CanonicalStepRecord:
    """Package scalar evidence separately from the numerical sweep."""
    direct = asdict(diagnostics)
    provenance = diagnostics.direct_q_provenance
    direct.pop("direct_q_provenance")
    direct.pop("factor_normalization_binary_exponent")
    return CanonicalStepRecord(
        step=step, uid=node.uid, label=node.label, **direct,
        per_step_function_replay_error=replay_error,
        expected_parent_occurrences=expected,
        parent_occurrences_pushed=absorption.pushed_occurrences,
        source_core_binary_exponent=int(source.binary_exponent),
        factor_binary_exponent=int(factor.binary_exponent),
        canonical_core_binary_exponent=int(node.core.binary_exponent),
        local_scaled_reconstruction_relative_error=local_error,
        local_reconstruction_exponent_delta=int(exponent_delta),
        maximum_absorption_scaled_relative_error=absorption.maximum_scaled_relative_error,
        maximum_absorption_exponent_delta=int(absorption.maximum_exponent_delta),
        scale_sensitive_occurrences_verified=absorption.verified_occurrences,
        per_step_function_replay_performed=replayed,
        direct_q_provenance_method=provenance.method,
        direct_q_provenance_verified=provenance.verified,
        direct_q_provenance_stage_count=provenance.stage_count,
        direct_q_provenance_column_blocks_compared=provenance.column_blocks_compared,
        direct_q_columns_compared=provenance.columns_compared,
        direct_q_compact_relative_error=provenance.compact_q_relative_error,
        direct_q_reconstruction_relative_error=provenance.direct_q_reconstruction_relative_error,
        direct_q_retained_transition_elements=provenance.retained_transition_elements,
        direct_q_minimum_pivot_to_maximum_entry=provenance.minimum_pivot_to_maximum_entry,
        direct_q_conditioning_threshold=provenance.conditioning_threshold,
        direct_q_conditioning_accepted=provenance.conditioning_accepted,
    )


def _compact_spectrum_record(
    uid: int,
    label: str,
    values: Tensor,
    environment_binary_exponent: int,
) -> CompactSpectrumRecord:
    """Summarize one descending Algorithm 3 spectrum without retaining it."""

    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("Algorithm 3 spectrum must be a nonempty vector")
    dimension = int(values.numel())
    raw_values = tuple(float(value) for value in values.tolist())
    if any(not math.isfinite(value) for value in raw_values):
        raise ValueError("Algorithm 3 spectrum must contain only finite values")
    absolute_maximum = max(abs(value) for value in raw_values)
    roundoff = (
        4096.0
        * torch.finfo(values.dtype).eps
        * float(dimension)
        * absolute_maximum
    )
    if any(value < -roundoff for value in raw_values):
        raise RuntimeError(
            "Algorithm 3 environment has a materially negative eigenvalue"
        )
    nonnegative = tuple(max(0.0, value) for value in raw_values)
    total = math.fsum(nonnegative)
    zero_trace = total <= roundoff
    leading = nonnegative[0]
    negative_roundoff_mass = math.fsum(
        -value for value in raw_values if value < 0.0
    )

    retention_ranks: list[int] = []
    tail_mantissas: list[float] = []
    tie_extensions: list[int] = []
    if zero_trace:
        retention_ranks = [1 for _ in COMPACT_TRACE_RETENTION_TARGETS]
        tail_mantissas = [0.0 for _ in COMPACT_TRACE_RETENTION_TARGETS]
        tie_extensions = [0 for _ in COMPACT_TRACE_RETENTION_TARGETS]
    else:
        cumulative: list[float] = []
        partial = 0.0
        for value in nonnegative:
            partial += value
            cumulative.append(partial)
        for target in COMPACT_TRACE_RETENTION_TARGETS:
            threshold = total * float(target)
            raw_rank = next(
                (
                    index
                    for index, value in enumerate(cumulative, start=1)
                    if value >= threshold
                ),
                dimension,
            )
            boundary = nonnegative[raw_rank - 1]
            tie_tolerance = max(roundoff, absolute_maximum * 1e-12)
            tied_rank = sum(
                value >= boundary - tie_tolerance for value in nonnegative
            )
            tied_rank = min(dimension, max(raw_rank, tied_rank))
            retention_ranks.append(tied_rank)
            tie_extensions.append(tied_rank - raw_rank)
            tail_mantissas.append(max(0.0, total - cumulative[tied_rank - 1]))

    relative_floor_ranks = tuple(
        max(
            1,
            sum(
                value > (absolute_maximum * float(floor) + roundoff)
                for value in nonnegative
            ),
        )
        for floor in COMPACT_RELATIVE_VALUE_FLOORS
    )
    return CompactSpectrumRecord(
        uid=int(uid),
        label=label,
        dimension=dimension,
        environment_binary_exponent=int(environment_binary_exponent),
        leading_value_mantissa=leading,
        trace_mantissa=total,
        negative_roundoff_mass_mantissa=negative_roundoff_mass,
        trace_retention_ranks=tuple(retention_ranks),
        trace_tail_mantissas=tuple(tail_mantissas),
        relative_floor_ranks=relative_floor_ranks,
        tie_extensions=tuple(tie_extensions),
        zero_trace=zero_trace,
    )


def implicit_shape_statistics(network: ImplicitProjectiveDAG) -> dict[str, int]:
    nodes = _walk_unique(network.root)
    cp = [node.core for node in nodes if isinstance(node.core, CPBinaryCore)]
    unary = [node.core for node in nodes if isinstance(node.core, UnaryCore)]
    rectangular = [
        node.core for node in nodes if isinstance(node.core, ReducedQRBinaryCore)
    ]
    dense_clones = [node.core for node in nodes if isinstance(node.core, DenseCloneCore)]
    fused_raw_dimension = (
        sum(spec.width for spec in network.physical_sources) + 1
        if network.physical_sources
        else network.token_count * network.feature_dimension + 1
    )
    return {
        "unique_nodes": len(nodes),
        "edge_occurrences": sum(len(node.children) for node in nodes),
        "cp_binary_nodes": len(cp),
        "reduced_q_binary_nodes": len(rectangular),
        "dense_clone_nodes": len(dense_clones),
        "unary_nodes": len(unary),
        "heterogeneous_physical_sources": len(network.physical_sources),
        "maximum_local_bond_dimension": max(node.output_dimension for node in nodes),
        "maximum_cp_rank": max((core.cp_rank for core in cp), default=0),
        "maximum_reduced_q_elements": max(
            (core.q_rows.numel() for core in rectangular), default=0
        ),
        "fused_token_feature_dimension": fused_raw_dimension,
    }


def predict_direct_rq_route_inventory(
    network: ImplicitProjectiveDAG,
) -> dict[str, object]:
    """Predict every shape after child R absorption, rejecting unsupported storage.

    Tied children use h(h+1)/2 QR coordinates, but the retained Q is stored on
    the original h*h ordered axes. Neither budget may be silently substituted
    for the other. No numerical rank or pivot decision occurs here.
    """
    nodes = _walk_unique(network.root)
    widths = {}
    count = q_total = cp_total = positive_delta = maximum_unfolding = maximum_q = changed = 0
    shapes: dict[str, int] = defaultdict(int)
    all_shapes: dict[str, int] = defaultdict(int)
    all_q_total = all_maximum_unfolding = all_maximum_q = unary_count = 0
    for node in nodes:
        core = node.core
        if isinstance(core, (ReducedQRBinaryCore, DenseCloneCore)):
            raise ValueError("route prediction requires a pre-Algorithm-1 production DAG")
        inputs = tuple(widths[id(child)] for child in node.children) if node.children else core.input_dimensions
        changed += int(inputs != core.input_dimensions)
        rows, stored = core.output_dimension, math.prod(inputs)
        columns = stored
        if isinstance(core, CPBinaryCore):
            if len(inputs) != 2:
                raise ValueError("predicted CP route does not have two inputs")
            if len(node.children) == 2 and node.children[0] is node.children[1]:
                if inputs[0] != inputs[1]:
                    raise ValueError("tied child dimensions differ")
                columns = inputs[0] * (inputs[0] + 1) // 2
        elif not isinstance(core, UnaryCore) or len(inputs) != 1:
            raise ValueError("predicted unary route does not have one input")
        if not _bounded_explicit_q_allowed_shape(rows, columns, storage_columns=stored):
            raise ValueError(
                "unsupported retained direct-Q route before factorization: "
                f"uid={node.uid} label={node.label!r} shape={rows}x{columns} "
                f"stored_columns={stored}"
            )
        widths[id(node)] = min(rows, columns)
        all_shapes[f"{rows}x{columns}"] += 1
        all_q_total += widths[id(node)] * stored
        all_maximum_unfolding = max(all_maximum_unfolding, rows * columns)
        all_maximum_q = max(all_maximum_q, widths[id(node)] * stored)
        if isinstance(core, UnaryCore):
            unary_count += 1
            continue
        q_elements = widths[id(node)] * stored
        cp_elements = core.cp_rank * (rows + sum(inputs))
        count += 1
        q_total += q_elements
        cp_total += cp_elements
        positive_delta += max(0, q_elements - cp_elements)
        maximum_unfolding = max(maximum_unfolding, rows * columns)
        maximum_q = max(maximum_q, q_elements)
        shapes[f"{rows}x{columns}"] += 1
    return {
        "schema": "exact_postorder_structural_direct_rq_routes_v2",
        "bounded_explicit_unfolding_element_limit": MAXIMUM_BOUNDED_EXPLICIT_UNFOLDING_ELEMENTS,
        "bounded_explicit_q_element_limit": MAXIMUM_BOUNDED_EXPLICIT_Q_ELEMENTS,
        "bounded_retained_q_candidate_count": count,
        "bounded_retained_q_total_elements": q_total,
        "bounded_retained_q_replaced_cp_total_elements": cp_total,
        "bounded_retained_q_signed_storage_delta_elements": q_total - cp_total,
        "bounded_retained_q_positive_storage_delta_elements": positive_delta,
        "bounded_retained_q_maximum_unfolding_elements": maximum_unfolding,
        "bounded_retained_q_maximum_elements": maximum_q,
        "streamed_cp_candidate_count": 0,
        "streamed_tall_unrepresentable_count": 0,
        "streamed_tall_unrepresentable_shape_counts": {},
        "bounded_retained_q_shape_counts": dict(sorted(shapes.items())),
        "nodes_with_propagated_input_shape_change": changed,
        "predicted_root_output_dimension": widths[id(network.root)],
        "simulated_unique_node_count": len(nodes),
        "unary_node_count": unary_count,
        "all_node_retained_q_total_elements": all_q_total,
        "all_node_maximum_unfolding_elements": all_maximum_unfolding,
        "all_node_maximum_retained_q_elements": all_maximum_q,
        "all_node_unfolding_shape_counts": dict(sorted(all_shapes.items())),
    }


def telemetry_dict(telemetry: MaterializationTelemetry) -> dict[str, int | float]:
    return asdict(telemetry)


def audit_algorithm1_factorization_calls() -> dict[str, object]:
    """Audit the actual primary numerical source closure, never only a facade."""
    from scripts.odt_direct_only_compliance import audit_direct_only_launch

    path = Path(__file__).resolve()
    report = audit_direct_only_launch(path.parents[3], (path.with_name("core.py"),))
    return {
        **report,
        "scope": "primary_transitive_source_closure",
        "static_direct_qr_call_sites": report["direct_qr_call_sites"],
        "direct_qr_required": True,
    }
