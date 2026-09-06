"""Shared-basis extension of Dooms on a declared symmetric tensor lift.

Direct C=LQ and every-occurrence absorption preserve the fixed ordered tensor.
Explicit downstream contraction gives H_v=sum_p E_p by message linearity.
Full eigenbasis gauges cancel at every producer-consumer occurrence.
The occurrence sum ranks independent single-cut losses. It does not optimize
simultaneous tied truncation, diagonalize every clone cut, or rank action loss.
See README.md and research/odt_reference/SHARED_DAG.md for the contract.
"""

from __future__ import annotations

import math
from typing import Callable, Iterator, Sequence

import torch

from xvla.train.odt_engine_v2.constants import LIFT_ID, LOCAL_SCALE_CERTIFICATION_TOLERANCE
from xvla.train.odt_engine_v2.diagnostics import _canonical_step_record, _compact_spectrum_record
from xvla.train.odt_engine_v2.factorization import _direct_rq_core
from xvla.train.odt_engine_v2.graph import (
    _clone_network, _parent_occurrences, _topological_root_first,
    _validate_network, _validate_symmetric_lift, _walk_unique,
    evaluate_projective_boundary, validate_canonical_exponent_normal_form,
)
from xvla.train.odt_engine_v2.ops import (
    _absorb_input_factor, _accumulate_scaled_message, _apply_output_basis,
    _offdiagonal_ratio, _role_environment, _scale_symmetric, _strip_power_of_two,
)
from xvla.train.odt_engine_v2.types import (
    AbsorptionScaleLedger, CPBinaryCore, CanonicalImplicitDAG, CanonicalStepRecord,
    CompactSpectrumRecord, DiagonalImplicitDAG, EnvironmentRecord, ImplicitNode,
    ImplicitProjectiveDAG, MaterializationTelemetry, RawPhysicalInput, ScaledMatrix,
    Tensor, _validate_real_finite,
)
from xvla.train.odt_engine_v2.validation import (
    _absorption_scale_audit, _local_direct_rq_scaled_reconstruction_audit,
    _projective_batch_relative_error, _scaled_matrix_audit,
    _validate_direct_q_provenance,
)


def _clear_certificate(network: ImplicitProjectiveDAG) -> None:
    network.algorithm1_direct_rq_complete = False
    network.algorithm1_scale_ledger_complete = False
    network.algorithm1_certified_head_binary_exponent = None


def _certify(network: ImplicitProjectiveDAG) -> None:
    network.algorithm1_direct_rq_complete = True
    network.algorithm1_scale_ledger_complete = True
    network.algorithm1_certified_head_binary_exponent = int(network.head_binary_exponent)
    validate_canonical_exponent_normal_form(network)


def _require_lift(network: ImplicitProjectiveDAG) -> None:
    if not network.claim_boundary.startswith(LIFT_ID + ":"):
        raise ValueError("ODT environments require the validated symmetric ordered lift")


def _push_factor_to_parents_with_scale_ledger(
    node: ImplicitNode,
    factor: ScaledMatrix,
    parents: dict[int, list[tuple[ImplicitNode, int]]],
    network: ImplicitProjectiveDAG,
) -> AbsorptionScaleLedger:
    """Absorb the direct factor in every syntactic slot and certify its scale."""
    occurrences = parents.get(id(node), [])
    maximum_error = 0.0
    maximum_exponent_delta = 0
    for parent, role in occurrences:
        before = parent.core
        after = _absorb_input_factor(before, role, factor)
        error, exponent_delta = _absorption_scale_audit(before, after, role, factor)
        if (not math.isfinite(error)
                or error > LOCAL_SCALE_CERTIFICATION_TOLERANCE or exponent_delta != 0):
            raise RuntimeError("direct-RQ parent absorption failed its scale-sensitive local ledger")
        parent.core = after
        maximum_error = max(maximum_error, error)
        maximum_exponent_delta = max(maximum_exponent_delta, abs(int(exponent_delta)))
    if occurrences:
        count = len(occurrences)
        return AbsorptionScaleLedger(count, count, maximum_error, maximum_exponent_delta)
    if node is not network.root:
        raise ValueError("non-root implicit node has no parent occurrence")
    if network.head.shape[1] != factor.mantissa.shape[0]:
        raise ValueError("root factor does not match boundary head")
    before = ScaledMatrix(network.head, int(network.head_binary_exponent))
    head, stripped = _strip_power_of_two(before.mantissa @ factor.mantissa)
    after = ScaledMatrix(head, before.binary_exponent + factor.binary_exponent + stripped)
    expected = ScaledMatrix(
        before.mantissa @ factor.mantissa, before.binary_exponent + factor.binary_exponent,
    )
    error, exponent_delta = _scaled_matrix_audit(after, expected)
    if (not math.isfinite(error)
            or error > LOCAL_SCALE_CERTIFICATION_TOLERANCE or exponent_delta != 0):
        raise RuntimeError("direct-RQ root absorption failed its scale-sensitive local ledger")
    network.head, network.head_binary_exponent = after.mantissa, int(after.binary_exponent)
    return AbsorptionScaleLedger(1, 1, error, abs(int(exponent_delta)))


def _push_factor_to_parents(
    node: ImplicitNode,
    factor: ScaledMatrix,
    parents: dict[int, list[tuple[ImplicitNode, int]]],
    network: ImplicitProjectiveDAG,
) -> int:
    return _push_factor_to_parents_with_scale_ledger(node, factor, parents, network).pushed_occurrences


def _replay_error(network, inputs, reference) -> float:
    error = _projective_batch_relative_error(evaluate_projective_boundary(network, inputs), reference)
    if not math.isfinite(error):
        raise RuntimeError("canonical ODT replay returned a nonfinite error")
    return error


@torch.no_grad()
def canonicalize_implicit_dag_direct_rq(
    network: ImplicitProjectiveDAG,
    *,
    block_size: int = 4096,
    replay_inputs: RawPhysicalInput | None = None,
    telemetry: MaterializationTelemetry | None = None,
    replay_each_step: bool = True,
    copy_network: bool = True,
    step_callback: Callable[
        [ImplicitProjectiveDAG, ImplicitNode, ScaledMatrix, CanonicalStepRecord], None,
    ] | None = None,
) -> CanonicalImplicitDAG:
    """Direct RQ, every-occurrence absorption, and one local scale ledger.

    Raw tied-core symmetry is checked before any child factor can hide skew.
    Failed in-place sweeps remain explicitly uncertified. Replay errors retain
    their historical reporting semantics, with nonfinite results rejected.
    """
    try:
        _validate_network(network)
        _validate_symmetric_lift(network)
    except BaseException:
        if not copy_network:
            _clear_certificate(network)
        raise
    work = _clone_network(network, unfold=False) if copy_network else network
    _clear_certificate(work)
    if not work.claim_boundary.startswith(LIFT_ID + ":"):
        work.claim_boundary = LIFT_ID + ": " + work.claim_boundary
    meter = MaterializationTelemetry() if telemetry is None else telemetry
    records: list[CanonicalStepRecord] = []
    try:
        parents = _parent_occurrences(work.root)
        reference = None if replay_inputs is None else evaluate_projective_boundary(work, replay_inputs)
        evaluations = int(reference is not None)
        for step, node in enumerate(_walk_unique(work.root)):
            source = node.core
            try:
                factor, canonical, diagnostics = _direct_rq_core(
                    source, meter, block_size=block_size,
                    tied_inputs=len(node.children) == 2 and node.children[0] is node.children[1],
                )
            except Exception as error:
                raise RuntimeError(
                    "Algorithm 1 direct-RQ step failed: "
                    f"step={step} uid={node.uid} label={node.label!r} "
                    f"kind={source.kind!r} unfolding={source.output_dimension}x"
                    f"{math.prod(source.input_dimensions)}"
                ) from error
            provenance = diagnostics.direct_q_provenance
            if provenance is None:
                raise RuntimeError("direct-RQ step did not return direct-Q provenance")
            _validate_direct_q_provenance(provenance, require_verified=True)
            if provenance.columns_compared != math.prod(source.input_dimensions):
                raise RuntimeError("direct-Q provenance did not cover every local unfolding column")
            if isinstance(canonical, CPBinaryCore) and canonical.direct_q_provenance != provenance:
                raise RuntimeError("compact canonical CP core lost its direct-Q provenance")
            local_error, exponent_delta = _local_direct_rq_scaled_reconstruction_audit(
                source, factor, canonical, block_size=max(1, block_size), telemetry=meter,
                factor_normalization_binary_exponent=diagnostics.factor_normalization_binary_exponent,
            )
            if (not math.isfinite(local_error)
                    or local_error > LOCAL_SCALE_CERTIFICATION_TOLERANCE or exponent_delta != 0):
                raise RuntimeError("direct-RQ local scaled reconstruction failed before parent absorption")
            node.core = canonical
            absorption = _push_factor_to_parents_with_scale_ledger(node, factor, parents, work)
            expected = len(parents.get(id(node), [])) or 1
            if absorption.pushed_occurrences != expected or absorption.verified_occurrences != expected:
                raise RuntimeError("direct-RQ absorption ledger did not verify every occurrence")
            replayed = reference is not None and replay_each_step
            error = _replay_error(work, replay_inputs, reference) if replayed else 0.0
            evaluations += int(replayed)
            record = _canonical_step_record(
                step, node, source, factor, diagnostics, absorption,
                expected, local_error, exponent_delta, error, replayed,
            )
            records.append(record)
            if step_callback is not None:
                step_callback(work, node, factor, record)
        final_error = 0.0
        if reference is not None:
            # A callback can change the final state after its per-step replay.
            final_error = _replay_error(work, replay_inputs, reference)
            evaluations += 1
        _certify(work)
        return CanonicalImplicitDAG(work, tuple(records), meter, final_error, evaluations)
    except BaseException:
        _clear_certificate(work)
        raise


def _environment_records(
    network: ImplicitProjectiveDAG, meter: MaterializationTelemetry,
) -> Iterator[tuple[ImplicitNode, EnvironmentRecord]]:
    """Contract the two downstream copies, emitting all child messages before yield.

    This ordering permits the consumer to gauge the yielded node immediately:
    no unfinished contraction depends on that node's old tensor afterward.
    Each record is the sum over clone occurrences of one shared origin, not
    the environment of an individual cut. Aggregation is linear in messages.
    """
    _require_lift(network)
    nodes = _topological_root_first(network.root)
    incoming = {id(node): 0 for node in nodes}
    for parent in nodes:
        for child in parent.children:
            incoming[id(child)] += 1
    root = torch.einsum("oa,ob->ab", network.head, network.head)
    pending = {id(network.root): _scale_symmetric(root, 2 * int(network.head_binary_exponent))}
    emitted = 0
    for node in nodes:
        aggregate = pending.pop(id(node), None)
        if aggregate is None:
            raise ValueError("implicit reverse traversal found an unreachable node")
        for role, child in enumerate(node.children):
            contracted = _role_environment(node.core, aggregate.mantissa, role, meter)
            message = _scale_symmetric(contracted, aggregate.binary_exponent + 2 * int(node.core.binary_exponent))
            _accumulate_scaled_message(pending, id(child), message)
        count = len(node.children)
        emitted += count
        yield node, EnvironmentRecord(
            node.uid, node.label, aggregate.mantissa, aggregate.binary_exponent,
            incoming[id(node)], count,
        )
    if pending or emitted != sum(incoming.values()):
        raise RuntimeError("Algorithm 2 did not emit every edge occurrence exactly once")


@torch.no_grad()
def reverse_implicit_environments(
    network: ImplicitProjectiveDAG,
    *,
    telemetry: MaterializationTelemetry | None = None,
    retain_records: bool = True,
    record_callback: Callable[[EnvironmentRecord], None] | None = None,
) -> tuple[EnvironmentRecord, ...]:
    validate_canonical_exponent_normal_form(network)
    meter = MaterializationTelemetry() if telemetry is None else telemetry
    records = []
    for _, record in _environment_records(network, meter):
        if record_callback is not None:
            record_callback(record)
        if retain_records:
            records.append(record)
    return tuple(records)


def _environment_eigensystem(record: EnvironmentRecord) -> tuple[Tensor, Tensor]:
    matrix = record.mantissa
    _validate_real_finite(matrix, "contracted ODT environment")
    if matrix.ndim != 2 or not matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("contracted ODT environment must be nonempty and square")
    scale = max(float(matrix.abs().max()), torch.finfo(matrix.dtype).tiny)
    if float((matrix / scale - matrix.T / scale).abs().max()) > 1e-12:
        raise ValueError("contracted ODT environment is not symmetric")
    symmetric = matrix / 2 + matrix.T / 2
    values, vectors = torch.linalg.eigh(symmetric)
    _validate_real_finite(values, "ODT environment eigenvalues")
    _validate_real_finite(vectors, "ODT environment eigenvectors")
    if float(values[0]) / scale < -1e-12 * len(matrix):
        raise ValueError("contracted ODT environment is materially indefinite")
    order = torch.argsort(values, descending=True)
    return values[order], vectors[:, order]


def _apply_gauge(node, vectors, parents, network) -> int:
    node.core = _apply_output_basis(node.core, vectors.T)
    occurrences = parents.get(id(node), [])
    basis = ScaledMatrix(vectors, 0)
    for parent, role in occurrences:
        parent.core = _absorb_input_factor(parent.core, role, basis, normalize=False)
    if occurrences:
        return len(occurrences)
    if node is not network.root:
        raise ValueError("non-root node has no EVD parent occurrence")
    network.head = network.head @ vectors
    return 1


def _retained_environment_records(network, records, nodes, parents):
    by_uid = {}
    for record in records:
        if record.uid in by_uid:
            raise ValueError("precomputed Algorithm 2 environments repeat a node")
        by_uid[record.uid] = record
    if set(by_uid) != {node.uid for node in nodes}:
        raise ValueError("precomputed Algorithm 2 environments do not match the network")
    for node in nodes:
        record = by_uid.pop(node.uid)
        if (record.mantissa.shape != (node.output_dimension, node.output_dimension)
                or record.label != node.label
                or record.incoming_occurrences != len(parents.get(id(node), []))
                or record.child_messages_emitted != len(node.children)):
            raise ValueError("precomputed Algorithm 2 environment metadata differs")
        yield node, record


@torch.no_grad()
def diagonalize_implicit_dag_full_rank(
    network: ImplicitProjectiveDAG,
    *,
    replay_inputs: RawPhysicalInput | None = None,
    telemetry: MaterializationTelemetry | None = None,
    copy_network: bool = True,
    precomputed_environments: Sequence[EnvironmentRecord] | None = None,
    stream_pre_evd_environments: bool = False,
    retain_eigenvalues: bool = True,
    retain_post_evd_environments: bool = True,
    compact_spectrum_callback: Callable[[CompactSpectrumRecord], None] | None = None,
) -> DiagonalImplicitDAG:
    """One contracted-environment traversal, one EVD, one every-occurrence gauge.

    Retention changes collection/order only, never the mathematical operations.
    Streaming emits a node's child messages before changing its gauge. Retained
    mode preserves the historical bottom-up spectrum/callback order.
    Precomputed records must come from this exact unchanged canonical network.
    Metadata checks cannot authenticate that caller-supplied numerical premise.
    """
    validate_canonical_exponent_normal_form(network)
    _require_lift(network)
    if compact_spectrum_callback is not None and retain_eigenvalues:
        raise ValueError("compact spectrum streaming requires full-spectrum retention to be disabled")
    if precomputed_environments is not None and stream_pre_evd_environments:
        raise ValueError("precomputed and streamed pre-EVD environments are mutually exclusive")
    work = _clone_network(network, unfold=False) if copy_network else network
    _clear_certificate(work)
    meter = MaterializationTelemetry() if telemetry is None else telemetry
    try:
        reference = None if replay_inputs is None else evaluate_projective_boundary(work, replay_inputs)
        nodes = _walk_unique(work.root)
        parents = _parent_occurrences(work.root)
        source = _environment_records(work, meter)
        if not stream_pre_evd_environments:
            records = precomputed_environments if precomputed_environments is not None else (record for _, record in source)
            source = _retained_environment_records(work, records, nodes, parents)
        spectra = []
        messages = count = pushed = expected = 0
        for node, record in source:
            values, vectors = _environment_eigensystem(record)
            if compact_spectrum_callback is not None:
                compact_spectrum_callback(_compact_spectrum_record(node.uid, node.label, values, record.binary_exponent))
            if retain_eigenvalues:
                spectra.append(ScaledMatrix(torch.diag(values), record.binary_exponent))
            pushed += _apply_gauge(node, vectors, parents, work)
            expected += len(parents.get(id(node), [])) or 1
            messages += record.child_messages_emitted
            count += 1
        if pushed != expected or count != len(nodes) or messages != sum(len(node.children) for node in nodes):
            raise RuntimeError("Algorithm 3 did not process every node and incident occurrence")
        replay_error = 0.0 if reference is None else _replay_error(work, replay_inputs, reference)
        recontracted = _environment_records(work, meter)
        if retain_post_evd_environments:
            recontracted = tuple(recontracted)
        maximum_offdiagonal = 0.0
        for _, record in recontracted:
            ratio = _offdiagonal_ratio(record.mantissa)
            if not math.isfinite(ratio):
                raise RuntimeError("recontracted ODT environment diagnostic is nonfinite")
            maximum_offdiagonal = max(maximum_offdiagonal, ratio)
        _certify(work)
        return DiagonalImplicitDAG(work, tuple(spectra), replay_error, maximum_offdiagonal, expected, pushed, messages, count, retain_eigenvalues)
    except BaseException:
        _clear_certificate(work)
        raise
