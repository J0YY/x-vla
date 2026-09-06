"""Independent clone comparisons and corruption controls. Not imported by the primary engine."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import torch


from xvla.train.odt_engine_v2.constants import (
    LOCAL_SCALE_CERTIFICATION_TOLERANCE,
    LIFT_ID,
)

from xvla.train.odt_engine_v2.core import (
    _push_factor_to_parents,
    _push_factor_to_parents_with_scale_ledger,
    reverse_implicit_environments,
)

from xvla.train.odt_engine_v2.factorization import (
    _direct_rq_core,
    _retained_direct_q_provenance,
)

from xvla.train.odt_engine_v2.diagnostics import _canonical_step_record

from xvla.train.odt_engine_v2.graph import (
    _clone_network,
    _parent_occurrences,
    _validate_network,
    _validate_symmetric_lift,
    _walk_unique,
    evaluate_projective_boundary,
    validate_canonical_exponent_normal_form,
)

from xvla.train.odt_engine_v2.ops import _absorb_input_factor, _add_scaled_matrices, _apply_output_basis, _core_clone, _ldexp, _offdiagonal_ratio, _strip_power_of_two

from xvla.train.odt_engine_v2.types import (
    CPBinaryCore,
    CanonicalCloneTrace,
    CanonicalStepRecord,
    CloneStepRecord,
    DenseCloneCore,
    DirectRQDiagnostics,
    EnvironmentRecord,
    ImplicitCore,
    ImplicitNode,
    ImplicitProjectiveDAG,
    IndependentCloneEVDTrace,
    MaterializationTelemetry,
    RawPhysicalInput,
    ReducedQRBinaryCore,
    ScaledMatrix,
    Tensor,
    TinyDenseEquivalence,
    UnaryCore,
)

from xvla.train.odt_engine_v2.validation import (
    _local_direct_rq_scaled_reconstruction_audit,
    _projective_batch_relative_error,
    _scaled_matrix_relative_error,
    _validate_direct_q_provenance,
)


def _independent_dense_clone_rq(core, telemetry, *, tied_inputs=False):
    """Independent direct QR with the same declared symmetric coordinate order."""
    dense = _tiny_core_mantissa(core)
    rows = dense.reshape(dense.shape[0], -1)
    if tied_inputs:
        if dense.ndim != 3 or dense.shape[1] != dense.shape[2]:
            raise ValueError("clone tied inputs require a square binary core")
        width = dense.shape[1]
        scale = max(float(dense.abs().max()), torch.finfo(dense.dtype).tiny)
        if float((dense - dense.transpose(1, 2)).abs().max()) > 1e-12 * scale:
            raise ValueError("clone raw tied core is not symmetric")
        columns = []
        for i in range(width):
            for j in range(i, width):
                columns.append((dense[:, i, j] / 2 + dense[:, j, i] / 2)
                               * (1.0 if i == j else math.sqrt(2.0)))
        unfolding = torch.stack(columns, dim=1)
    else:
        unfolding = rows
    q, upper = torch.linalg.qr(unfolding.T, mode="reduced")
    signs = torch.where(torch.diagonal(upper) < 0, -torch.ones_like(torch.diagonal(upper)),
                        torch.ones_like(torch.diagonal(upper)))
    q, upper = q * signs[None, :], signs[:, None] * upper
    if tied_inputs:
        expanded = dense.new_zeros((q.shape[1], width, width))
        cursor = 0
        for i in range(width):
            for j in range(i, width):
                expanded[:, i, j] = expanded[:, j, i] = q.T[:, cursor] / (1.0 if i == j else math.sqrt(2.0))
                cursor += 1
        q_rows = expanded.reshape(q.shape[1], -1)
    else:
        q_rows = q.T
    factor = upper.T
    scale = rows.abs().max().clamp_min(torch.finfo(rows.dtype).tiny)
    error = float(((factor @ q_rows - rows).abs().max() / scale).item())
    factor_mantissa, exponent = _strip_power_of_two(factor)
    telemetry.local_direct_rq_factorizations += 1
    telemetry.householder_qr_kernel_calls += 1
    telemetry.direct_q_provenance_certificates += 1
    telemetry.direct_q_columns_compared += rows.shape[1]
    provenance = _retained_direct_q_provenance("independent_symmetric_clone_qr_v2", rows.shape[1], error)
    return (ScaledMatrix(factor_mantissa, core.binary_exponent + exponent),
            DenseCloneCore(q_rows.reshape((q_rows.shape[0],) + tuple(dense.shape[1:])), "explicit_clone::" + core.kind, 0),
            DirectRQDiagnostics("independent_symmetric_clone_qr_v2", error, 1, tuple(unfolding.shape),
                                0.0, 0.0, False, q_rows.shape[0] < rows.shape[0], True, exponent, provenance))


def _tiny_core_mantissa(core: ImplicitCore, maximum_elements: int = 100_000) -> Tensor:
    elements = core.output_dimension * math.prod(core.input_dimensions)
    if elements > maximum_elements:
        raise ValueError("step-trace core comparison is restricted to the tiny oracle")
    if isinstance(core, UnaryCore):
        return core.matrix
    if isinstance(core, DenseCloneCore):
        return core.tensor
    if isinstance(core, ReducedQRBinaryCore):
        return core.q_rows.reshape(
            core.output_dimension, core.left_dimension, core.right_dimension
        )
    return torch.einsum(
        "ot,ti,tj->oij",
        core.output_factor,
        core.left_factor,
        core.right_factor,
    )


def _tiny_core_relative_error(actual: ImplicitCore, expected: ImplicitCore) -> float:
    left = _tiny_core_mantissa(actual).reshape(1, -1)
    right = _tiny_core_mantissa(expected).reshape(1, -1)
    return _scaled_matrix_relative_error(
        ScaledMatrix(left, actual.binary_exponent),
        ScaledMatrix(right, expected.binary_exponent),
    )


@torch.no_grad()
def canonicalize_with_explicit_clone_step_trace(
    network: ImplicitProjectiveDAG,
    replay_inputs: RawPhysicalInput,
    *,
    block_size: int = 512,
) -> CanonicalCloneTrace:
    """Compare shared Algorithm 1 with a no-memo clone after every RQ step.

    The explicit graph duplicates every syntactic occurrence.  For each shared
    node in bottom-up order, every corresponding clone occurrence is factored
    independently.  The gate compares positive-diagonal R factors, Q cores,
    every transformed parent occurrence, and full function replay before the
    next shared node is touched.
    """

    _validate_symmetric_lift(network)
    shared = _clone_network(network, unfold=False)
    clone = _clone_network(network, unfold=True)
    shared_parents = _parent_occurrences(shared.root)
    clone_parents = _parent_occurrences(clone.root)
    shared_schedule = _walk_unique(shared.root)
    shared.claim_boundary = clone.claim_boundary = LIFT_ID + ": explicit shared-clone validation"
    clone_nodes = _walk_unique(clone.root)
    clones_by_origin: dict[int, list[ImplicitNode]] = defaultdict(list)
    for node in clone_nodes:
        if node.origin_uid is None:
            raise ValueError("explicit clone node lost its origin uid")
        clones_by_origin[node.origin_uid].append(node)
    reference = evaluate_projective_boundary(network, replay_inputs)
    shared_meter = MaterializationTelemetry()
    clone_meter = MaterializationTelemetry()
    records: list[CloneStepRecord] = []
    shared_steps: list[CanonicalStepRecord] = []
    clone_methods: list[str] = []

    for step, shared_node in enumerate(shared_schedule):
        shared_source_core = shared_node.core
        factor, q_core, diagnostics = _direct_rq_core(
            shared_source_core, shared_meter, block_size=block_size,
            tied_inputs=len(shared_node.children) == 2 and shared_node.children[0] is shared_node.children[1],
        )
        provenance = diagnostics.direct_q_provenance
        if provenance is None:
            raise RuntimeError("shared clone trace lost direct-Q provenance")
        _validate_direct_q_provenance(provenance, require_verified=True)
        local_reconstruction_error, local_exponent_delta = (
            _local_direct_rq_scaled_reconstruction_audit(
                shared_source_core,
                factor,
                q_core,
                block_size=max(1, block_size),
                telemetry=shared_meter,
                factor_normalization_binary_exponent=(
                    diagnostics.factor_normalization_binary_exponent
                ),
            )
        )
        if (
            local_reconstruction_error > LOCAL_SCALE_CERTIFICATION_TOLERANCE
            or local_exponent_delta != 0
        ):
            raise RuntimeError("shared clone trace failed local scale reconstruction")
        shared_node.core = q_core
        shared_absorption = _push_factor_to_parents_with_scale_ledger(
            shared_node, factor, shared_parents, shared
        )
        shared_pushed = shared_absorption.pushed_occurrences
        candidates = clones_by_origin.get(shared_node.uid, [])
        if not candidates:
            raise ValueError("explicit clone trace missed a shared origin")
        factor_error = 0.0
        q_error = 0.0
        literal_compared = 0
        null_gauge_checked = 0
        supported_reconstruction_error = diagnostics.factorization_relative_error
        clone_pushed = 0
        for candidate in candidates:
            clone_source_core = candidate.core
            clone_factor, clone_q, clone_diagnostics = _independent_dense_clone_rq(
                clone_source_core, clone_meter,
                tied_inputs=len(shared_node.children) == 2 and shared_node.children[0] is shared_node.children[1],
            )
            clone_local_error, clone_exponent_delta = (
                _local_direct_rq_scaled_reconstruction_audit(
                clone_source_core,
                clone_factor,
                clone_q,
                block_size=max(1, block_size),
                telemetry=clone_meter,
                factor_normalization_binary_exponent=(
                    clone_diagnostics.factor_normalization_binary_exponent
                ),
                )
            )
            if (
                clone_local_error > LOCAL_SCALE_CERTIFICATION_TOLERANCE
                or clone_exponent_delta != 0
            ):
                raise RuntimeError("explicit clone trace failed local scale reconstruction")
            candidate.core = clone_q
            clone_pushed += _push_factor_to_parents(
                candidate, clone_factor, clone_parents, clone
            )
            clone_methods.append(clone_diagnostics.method)
            supported_reconstruction_error = max(
                supported_reconstruction_error,
                clone_diagnostics.factorization_relative_error,
            )
            literal_compared += 1
            factor_error = max(factor_error, _scaled_matrix_relative_error(clone_factor, factor))
            q_error = max(q_error, _tiny_core_relative_error(clone_q, q_core))

        parent_error = 0.0
        shared_parent_nodes = {parent.uid: parent for parent, _ in shared_parents.get(id(shared_node), [])}
        for parent_uid, shared_parent in shared_parent_nodes.items():
            for clone_parent in clones_by_origin.get(parent_uid, []):
                parent_error = max(
                    parent_error,
                    _tiny_core_relative_error(clone_parent.core, shared_parent.core),
                )

        shared_value = evaluate_projective_boundary(shared, replay_inputs)
        clone_value = evaluate_projective_boundary(clone, replay_inputs)
        shared_replay = _projective_batch_relative_error(shared_value, reference)
        clone_replay = _projective_batch_relative_error(clone_value, reference)
        shared_clone = _projective_batch_relative_error(shared_value, clone_value)
        head_error = _scaled_matrix_relative_error(
            ScaledMatrix(shared.head, shared.head_binary_exponent),
            ScaledMatrix(clone.head, clone.head_binary_exponent),
        )
        expected_shared_pushes = len(shared_parents.get(id(shared_node), [])) or 1
        expected_clone_pushes = sum(
            len(clone_parents.get(id(candidate), [])) or 1 for candidate in candidates
        )
        records.append(
            CloneStepRecord(
                step,
                shared_node.uid,
                shared_node.label,
                len(candidates),
                factor_error,
                q_error,
                parent_error,
                shared_replay,
                clone_replay,
                shared_clone,
                literal_compared,
                null_gauge_checked,
                supported_reconstruction_error,
                expected_shared_pushes,
                shared_pushed,
                expected_clone_pushes,
                clone_pushed,
                head_error,
            )
        )
        shared_steps.append(_canonical_step_record(
            step, shared_node, shared_source_core, factor, diagnostics, shared_absorption,
            expected_shared_pushes, local_reconstruction_error, local_exponent_delta,
            shared_replay, True,
        ))
        if max(factor_error, q_error, parent_error, head_error) > 2e-8:
            raise RuntimeError("explicit-clone canonicalization differs from the shared tensor lift")

    shared.algorithm1_direct_rq_complete = True
    shared.algorithm1_scale_ledger_complete = True
    shared.algorithm1_certified_head_binary_exponent = int(
        shared.head_binary_exponent
    )
    clone.algorithm1_direct_rq_complete = True
    clone.algorithm1_scale_ledger_complete = True
    clone.algorithm1_certified_head_binary_exponent = int(
        clone.head_binary_exponent
    )
    validate_canonical_exponent_normal_form(shared)
    validate_canonical_exponent_normal_form(clone)
    return CanonicalCloneTrace(
        shared,
        clone,
        tuple(records),
        tuple(shared_steps),
        tuple(clone_methods),
    )


@torch.no_grad()
def tiny_dense_direct_rq_equivalence(
    network: ImplicitProjectiveDAG,
    replay_inputs: RawPhysicalInput,
    *,
    block_size: int = 512,
) -> TinyDenseEquivalence:
    """Compare streamed Algorithm 1 to independent dense QR at each tiny step."""

    _validate_symmetric_lift(network)
    work = _clone_network(network, unfold=False)
    parents = _parent_occurrences(work.root)
    reference = evaluate_projective_boundary(work, replay_inputs)
    streamed_meter = MaterializationTelemetry()
    dense_meter = MaterializationTelemetry()
    factor_error = 0.0
    q_error = 0.0
    reconstruction_error = 0.0
    replay_error = 0.0
    full_steps = 0
    null_steps = 0
    for node in _walk_unique(work.root):
        original = _core_clone(node.core)
        factor, canonical, diagnostics = _direct_rq_core(
            original, streamed_meter, block_size=block_size,
            tied_inputs=len(node.children) == 2 and node.children[0] is node.children[1],
        )
        dense_factor, dense_q, dense_diagnostics = _independent_dense_clone_rq(
            original, dense_meter,
            tied_inputs=len(node.children) == 2 and node.children[0] is node.children[1],
        )
        reconstruction_error = max(
            reconstruction_error,
            diagnostics.factorization_relative_error,
            dense_diagnostics.factorization_relative_error,
        )
        full_steps += 1
        factor_error = max(factor_error, _scaled_matrix_relative_error(factor, dense_factor))
        q_error = max(q_error, _tiny_core_relative_error(canonical, dense_q))
        node.core = canonical
        _push_factor_to_parents(node, factor, parents, work)
        replay_error = max(
            replay_error,
            _projective_batch_relative_error(
                evaluate_projective_boundary(work, replay_inputs), reference
            ),
        )
    return TinyDenseEquivalence(
        len(_walk_unique(work.root)),
        0,  # Legacy numerical-rank counters are unassessed, not inferred.
        0,
        factor_error,
        q_error,
        reconstruction_error,
        replay_error,
    )


@torch.no_grad()
def compare_shared_and_explicit_clone_environments(
    shared_network: ImplicitProjectiveDAG,
    explicit_clone_network: ImplicitProjectiveDAG,
) -> dict[str, float | int]:
    """Aggregate no-memo clone messages and compare Algorithm 2 exactly."""

    shared = reverse_implicit_environments(shared_network)
    clone = reverse_implicit_environments(explicit_clone_network)
    clone_origin = {
        node.uid: node.origin_uid for node in _walk_unique(explicit_clone_network.root)
    }
    grouped: dict[int, list[ScaledMatrix]] = defaultdict(list)
    for record in clone:
        origin = clone_origin.get(record.uid)
        if origin is None:
            raise ValueError("clone environment record has no origin")
        grouped[origin].append(ScaledMatrix(record.mantissa, record.binary_exponent))
    shared_by_uid = {
        record.uid: ScaledMatrix(record.mantissa, record.binary_exponent)
        for record in shared
    }
    if set(grouped) != set(shared_by_uid):
        raise ValueError("shared and clone environment origins disagree")
    error = 0.0
    for uid, messages in grouped.items():
        aggregate = messages[0] if len(messages) == 1 else _add_scaled_matrices(messages)
        error = max(error, _scaled_matrix_relative_error(aggregate, shared_by_uid[uid]))
    return {
        "maximum_aggregate_relative_error": error,
        "shared_unique_environment_count": len(shared),
        "explicit_clone_environment_count": len(clone),
        "origins_compared": len(grouped),
    }


def _aggregate_clone_environment_records(
    network: ImplicitProjectiveDAG,
    records: Sequence[EnvironmentRecord],
) -> dict[int, ScaledMatrix]:
    origin_by_uid = {node.uid: node.origin_uid for node in _walk_unique(network.root)}
    grouped: dict[int, list[ScaledMatrix]] = defaultdict(list)
    for record in records:
        origin = origin_by_uid.get(record.uid)
        if origin is None:
            raise ValueError("explicit clone environment has no origin")
        grouped[origin].append(ScaledMatrix(record.mantissa, record.binary_exponent))
    return {
        uid: messages[0] if len(messages) == 1 else _add_scaled_matrices(messages)
        for uid, messages in grouped.items()
    }


def _eigenvalue_relative_error(
    first_values: ScaledMatrix,
    second_values: ScaledMatrix,
) -> float:
    center = max(first_values.binary_exponent, second_values.binary_exponent)
    first = _ldexp(
        torch.diagonal(first_values.mantissa),
        first_values.binary_exponent - center,
    )
    second = _ldexp(
        torch.diagonal(second_values.mantissa),
        second_values.binary_exponent - center,
    )
    scale = first.norm().clamp_min(torch.finfo(first.dtype).tiny)
    return float(((first - second).norm() / scale).item())


@torch.no_grad()
def diagonalize_shared_and_explicit_clone_independently(
    shared_network: ImplicitProjectiveDAG,
    explicit_clone_network: ImplicitProjectiveDAG,
    replay_inputs: RawPhysicalInput,
    *,
    omit_clone_parent: tuple[str, int] | None = None,
) -> IndependentCloneEVDTrace:
    """Independent Algorithm 3 on shared and no-memo dense-clone contractions."""

    _validate_network(shared_network)
    _validate_network(explicit_clone_network)
    shared = _clone_network(shared_network, unfold=False)
    clone = _clone_network(explicit_clone_network, unfold=False)
    shared_reference = evaluate_projective_boundary(shared, replay_inputs)
    clone_reference = evaluate_projective_boundary(clone, replay_inputs)

    shared_records = reverse_implicit_environments(shared)
    clone_records = reverse_implicit_environments(clone)
    shared_messages = {
        record.uid: ScaledMatrix(record.mantissa, record.binary_exponent)
        for record in shared_records
    }
    clone_messages = _aggregate_clone_environment_records(clone, clone_records)
    if set(shared_messages) != set(clone_messages):
        raise ValueError("independent clone EVD origins disagree with shared nodes")
    pre_environment_error = max(
        _scaled_matrix_relative_error(clone_messages[uid], shared_messages[uid])
        for uid in shared_messages
    )

    shared_bases: dict[int, Tensor] = {}
    clone_bases: dict[int, Tensor] = {}
    eigenvalue_error = 0.0
    for uid in shared_messages:
        shared_message = shared_messages[uid]
        clone_message = clone_messages[uid]
        shared_values, shared_vectors = torch.linalg.eigh(
            0.5 * (shared_message.mantissa + shared_message.mantissa.T)
        )
        clone_values, clone_vectors = torch.linalg.eigh(
            0.5 * (clone_message.mantissa + clone_message.mantissa.T)
        )
        shared_order = torch.argsort(shared_values, descending=True)
        clone_order = torch.argsort(clone_values, descending=True)
        shared_values = shared_values[shared_order]
        clone_values = clone_values[clone_order]
        shared_vectors = shared_vectors[:, shared_order]
        clone_vectors = clone_vectors[:, clone_order]
        shared_bases[uid] = shared_vectors
        clone_bases[uid] = clone_vectors
        value_error = _eigenvalue_relative_error(
            ScaledMatrix(torch.diag(shared_values), shared_message.binary_exponent),
            ScaledMatrix(torch.diag(clone_values), clone_message.binary_exponent),
        )
        eigenvalue_error = max(eigenvalue_error, value_error)

    shared_parents = _parent_occurrences(shared.root)
    clone_parents = _parent_occurrences(clone.root)
    clone_by_origin: dict[int, list[ImplicitNode]] = defaultdict(list)
    for node in _walk_unique(clone.root):
        if node.origin_uid is None:
            raise ValueError("independent clone EVD node lost its origin")
        clone_by_origin[node.origin_uid].append(node)

    expected_shared = pushed_shared = 0
    expected_clone = pushed_clone = 0
    omitted = 0
    targeted = 0
    target_occurrence = 0
    for shared_node in _walk_unique(shared.root):
        shared_basis = shared_bases[shared_node.uid]
        shared_node.core = _apply_output_basis(shared_node.core, shared_basis.T)
        occurrences = shared_parents.get(id(shared_node), [])
        if occurrences:
            expected_shared += len(occurrences)
            for parent, role in occurrences:
                parent.core = _absorb_input_factor(
                    parent.core,
                    role,
                    ScaledMatrix(shared_basis, 0),
                    normalize=False,
                )
                pushed_shared += 1
        else:
            expected_shared += 1
            shared.head = shared.head @ shared_basis
            pushed_shared += 1

        clone_basis = clone_bases[shared_node.uid]
        candidates = clone_by_origin.get(shared_node.uid, [])
        if not candidates:
            raise ValueError("independent clone EVD missed a shared origin")
        for candidate in candidates:
            candidate.core = _apply_output_basis(candidate.core, clone_basis.T)
            occurrences = clone_parents.get(id(candidate), [])
            if occurrences:
                expected_clone += len(occurrences)
                for parent, role in occurrences:
                    is_target = (
                        omit_clone_parent is not None
                        and shared_node.label == omit_clone_parent[0]
                    )
                    should_omit = is_target and target_occurrence == omit_clone_parent[1]
                    if is_target:
                        target_occurrence += 1
                        targeted += 1
                    if should_omit:
                        omitted += 1
                        continue
                    parent.core = _absorb_input_factor(
                        parent.core,
                        role,
                        ScaledMatrix(clone_basis, 0),
                        normalize=False,
                    )
                    pushed_clone += 1
            else:
                expected_clone += 1
                clone.head = clone.head @ clone_basis
                pushed_clone += 1
    if omit_clone_parent is not None and omitted != 1:
        raise ValueError("independent clone EVD omission did not resolve exactly once")

    shared_value = evaluate_projective_boundary(shared, replay_inputs)
    clone_value = evaluate_projective_boundary(clone, replay_inputs)
    shared_replay = _projective_batch_relative_error(shared_value, shared_reference)
    clone_replay = _projective_batch_relative_error(clone_value, clone_reference)
    shared_clone = _projective_batch_relative_error(shared_value, clone_value)
    shared_after = reverse_implicit_environments(shared)
    clone_after = reverse_implicit_environments(clone)
    shared_after_messages = {
        record.uid: ScaledMatrix(record.mantissa, record.binary_exponent)
        for record in shared_after
    }
    clone_after_messages = _aggregate_clone_environment_records(clone, clone_after)
    shared_offdiagonal = max(
        (_offdiagonal_ratio(value.mantissa) for value in shared_after_messages.values()),
        default=0.0,
    )
    clone_offdiagonal = max(
        (_offdiagonal_ratio(value.mantissa) for value in clone_after_messages.values()),
        default=0.0,
    )
    return IndependentCloneEVDTrace(
        shared,
        clone,
        pre_environment_error,
        eigenvalue_error,
        shared_replay,
        clone_replay,
        shared_clone,
        shared_offdiagonal,
        clone_offdiagonal,
        expected_shared,
        pushed_shared,
        expected_clone,
        pushed_clone,
        omitted,
        targeted,
    )


@torch.no_grad()
def drop_first_residual_add_cross_term(
    network: ImplicitProjectiveDAG,
    *,
    label: str = "ffn.residual_add",
) -> ImplicitProjectiveDAG:
    work = _clone_network(network, unfold=False)
    matches = [node for node in _walk_unique(work.root) if node.label == label]
    if len(matches) != 1 or not isinstance(matches[0].core, CPBinaryCore):
        raise ValueError("residual-add negative control did not resolve one CP node")
    node = matches[0]
    if node.core.kind != "sparse_projective_vector_add":
        raise ValueError("negative-control target is not a projective add")
    output = node.core.output_factor.clone()
    output[:, 0] = 0.0
    node.core = CPBinaryCore(
        output,
        node.core.left_factor,
        node.core.right_factor,
        node.core.kind,
        node.core.binary_exponent,
        node.core.direct_q_provenance,
    )
    return work


def old_project_then_add_topology_is_rejected(*args, **kwargs):
    raise ValueError("retired solver-dependent control; use the symmetric bounded-QR regression")


def multiblock_tsqr_dense_equivalence(*args, **kwargs):
    raise ValueError("retired TSQR solve route; unsupported retained-Householder shapes fail closed")
