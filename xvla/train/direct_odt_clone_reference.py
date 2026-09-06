"""Independent bounded no-memo reference for direct-RQ Dooms ODT.

This module accepts the production graph dataclasses only as a serialization
schema.  It immediately unfolds the shared DAG into a private dense occurrence
tree.  All subsequent cloning, evaluation, R pushes, environment contractions,
aggregation, and gauge pushes are implemented here without production helpers.

Algorithm 1 materializes each bounded local unfolding and computes ``M = R Q``
only through a direct reduced ``QR(M.T)``.  A deficient unfolding retains the
literal reduced Q, including the Householder completion chosen by QR.  Algorithm
2 contracts downstream tensor-network environments on the occurrence tree.
Algorithm 3 eigendecomposes only those contracted, per-origin environments and
pushes the resulting square gauge through every clone occurrence.

The reference is intentionally bounded and test-oriented.  It is an oracle for
the clone-unfolded syntactic coefficient metric, not a production compiler.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from xvla.train.implicit_sparse_projective_odt import (
    CPBinaryCore,
    DenseCloneCore,
    ImplicitNode,
    ImplicitProjectiveDAG,
    ReducedQRBinaryCore,
    UnaryCore,
)


Tensor = torch.Tensor
MAX_LOCAL_ELEMENTS = 4_000_000
MAX_OCCURRENCES = 100_000


def _require_real_finite(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point() or value.is_complex():
        raise TypeError(f"{name} must be a real floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _binary_shift(value: Tensor, exponent: int) -> Tensor:
    return torch.ldexp(
        value,
        torch.tensor(exponent, dtype=torch.int64, device=value.device),
    )


def _bounded_mantissa(value: Tensor) -> tuple[Tensor, int]:
    _require_real_finite(value, "reference tensor")
    maximum = value.abs().max()
    if float(maximum.item()) == 0.0:
        return value.clone(), 0
    _, exponent = torch.frexp(maximum)
    integer = int(exponent.item())
    return _binary_shift(value, -integer), integer


@dataclass(frozen=True)
class ReferenceScaledMatrix:
    mantissa: Tensor
    binary_exponent: int = 0


@dataclass(frozen=True)
class ReferenceScaledBatch:
    mantissa: Tensor
    binary_exponent: Tensor


@dataclass(eq=False)
class DenseOccurrence:
    occurrence_id: int
    origin_uid: int
    label: str
    tensor: Tensor
    binary_exponent: int
    children: tuple["DenseOccurrence", ...]
    physical_token: int | None
    physical_source_key: str | None

    @property
    def output_dimension(self) -> int:
        return int(self.tensor.shape[0])

    @property
    def input_dimensions(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.tensor.shape[1:])


@dataclass
class DenseOccurrenceNetwork:
    root: DenseOccurrence
    head: Tensor
    head_binary_exponent: int
    token_count: int
    feature_dimension: int
    selected_token: int
    mask: Tensor
    physical_sources: tuple[Any, ...]
    algorithm1_complete: bool = False


@dataclass(frozen=True)
class ReferenceRQStep:
    step: int
    origin_uid: int
    label: str
    occurrence_count: int
    unfolding_shape: tuple[int, int]
    factors: tuple[ReferenceScaledMatrix, ...]
    q_tensors: tuple[Tensor, ...]
    maximum_factorization_relative_error: float
    minimum_diagonal_to_maximum_entry: float
    literal_q_chart_resolved: bool
    expected_factor_pushes: int
    completed_factor_pushes: int
    omitted_factor_pushes: int
    network_snapshot: DenseOccurrenceNetwork
    homogeneous_output: ReferenceScaledBatch
    projective_coordinates: Tensor
    projective_replay_relative_error: float


@dataclass
class ReferenceCanonicalization:
    network: DenseOccurrenceNetwork
    steps: tuple[ReferenceRQStep, ...]
    origin_schedule: tuple[tuple[int, str], ...]
    initial_projective_coordinates: Tensor


@dataclass(frozen=True)
class OccurrenceEnvironment:
    occurrence_id: int
    origin_uid: int
    label: str
    mantissa: Tensor
    binary_exponent: int
    child_messages_emitted: int


@dataclass(frozen=True)
class ProductionStepComparison:
    step: int
    origin_uid: int
    label: str
    maximum_q_core_relative_error: float
    production_factorization_relative_error: float
    reference_factorization_relative_error: float
    production_pushes_complete: bool
    reference_pushes_complete: bool


@dataclass(frozen=True)
class ReferenceNetworkComparison:
    occurrence_count: int
    maximum_core_relative_error: float
    head_relative_error: float


@dataclass
class ReferenceDiagonalization:
    network: DenseOccurrenceNetwork
    eigenvalues_by_origin: dict[int, ReferenceScaledMatrix]
    gauges_by_origin: dict[int, Tensor]
    initial_projective_coordinates: Tensor
    final_projective_coordinates: Tensor
    replay_relative_error: float
    maximum_recontracted_offdiagonal_ratio: float
    expected_gauge_pushes: int
    completed_gauge_pushes: int
    omitted_gauge_pushes: int
    aggregates_after: dict[int, ReferenceScaledMatrix]


def _schema_core_dense(core: Any, maximum_elements: int) -> Tensor:
    if isinstance(core, UnaryCore):
        dense = core.matrix.clone()
    elif isinstance(core, CPBinaryCore):
        dense = torch.einsum(
            "or,ri,rj->oij",
            core.output_factor,
            core.left_factor,
            core.right_factor,
        )
    elif isinstance(core, ReducedQRBinaryCore):
        dense = core.q_rows.reshape(
            core.output_dimension,
            core.left_dimension,
            core.right_dimension,
        ).clone()
    elif isinstance(core, DenseCloneCore):
        dense = core.tensor.clone()
    else:
        raise TypeError(f"unsupported production core schema {type(core).__name__}")
    if dense.ndim not in (2, 3):
        raise ValueError("bounded reference supports only unary and binary cores")
    if dense.numel() > maximum_elements:
        raise ValueError(
            f"bounded reference core has {dense.numel()} elements, limit is {maximum_elements}"
        )
    _require_real_finite(dense, "materialized reference core")
    return dense


def _schema_origin_schedule(root: ImplicitNode) -> tuple[tuple[int, str], ...]:
    completed: set[int] = set()
    active: set[int] = set()
    result: list[tuple[int, str]] = []

    def visit(node: ImplicitNode) -> None:
        identity = id(node)
        if identity in active:
            raise ValueError("production schema contains a cycle")
        if identity in completed:
            return
        active.add(identity)
        for child in node.children:
            visit(child)
        active.remove(identity)
        completed.add(identity)
        origin = node.uid if node.origin_uid is None else node.origin_uid
        result.append((origin, node.label))

    visit(root)
    if len({uid for uid, _ in result}) != len(result):
        raise ValueError("production schema reuses an origin uid for distinct nodes")
    return tuple(result)


def unfold_occurrences_no_memo(
    network: ImplicitProjectiveDAG,
    *,
    maximum_local_elements: int = MAX_LOCAL_ELEMENTS,
    maximum_occurrences: int = MAX_OCCURRENCES,
) -> DenseOccurrenceNetwork:
    """Expand every syntactic occurrence recursively, without a clone memo."""

    _require_real_finite(network.head, "production boundary head")
    if (
        isinstance(maximum_occurrences, bool)
        or not isinstance(maximum_occurrences, int)
        or maximum_occurrences < 1
    ):
        raise ValueError("maximum occurrence count must be a positive integer")
    next_occurrence = 0
    active: set[int] = set()

    def expand(node: ImplicitNode) -> DenseOccurrence:
        nonlocal next_occurrence
        identity = id(node)
        if identity in active:
            raise ValueError("production schema contains a cycle")
        if next_occurrence >= maximum_occurrences:
            raise ValueError(
                "no-memo occurrence expansion exceeds the declared limit "
                f"of {maximum_occurrences}"
            )
        occurrence_id = next_occurrence
        next_occurrence += 1
        active.add(identity)
        children = tuple(expand(child) for child in node.children)
        active.remove(identity)
        tensor = _schema_core_dense(node.core, maximum_local_elements)
        if children:
            if len(children) != tensor.ndim - 1:
                raise ValueError("schema node arity does not match its dense core")
            if tuple(child.output_dimension for child in children) != tuple(tensor.shape[1:]):
                raise ValueError("schema child dimensions do not match its dense core")
        origin = node.uid if node.origin_uid is None else node.origin_uid
        return DenseOccurrence(
            occurrence_id,
            origin,
            node.label,
            tensor,
            int(node.core.binary_exponent),
            children,
            node.physical_token,
            node.physical_source_key,
        )

    root = expand(network.root)
    return DenseOccurrenceNetwork(
        root=root,
        head=network.head.clone(),
        head_binary_exponent=int(network.head_binary_exponent),
        token_count=int(network.token_count),
        feature_dimension=int(network.feature_dimension),
        selected_token=int(network.selected_token),
        mask=network.mask.clone(),
        physical_sources=tuple(network.physical_sources),
        algorithm1_complete=bool(network.algorithm1_direct_rq_complete),
    )


def _copy_occurrence_network(network: DenseOccurrenceNetwork) -> DenseOccurrenceNetwork:
    def copy_node(node: DenseOccurrence) -> DenseOccurrence:
        return DenseOccurrence(
            node.occurrence_id,
            node.origin_uid,
            node.label,
            node.tensor.clone(),
            node.binary_exponent,
            tuple(copy_node(child) for child in node.children),
            node.physical_token,
            node.physical_source_key,
        )

    return DenseOccurrenceNetwork(
        copy_node(network.root),
        network.head.clone(),
        network.head_binary_exponent,
        network.token_count,
        network.feature_dimension,
        network.selected_token,
        network.mask.clone(),
        network.physical_sources,
        network.algorithm1_complete,
    )


def _occurrences_postorder(root: DenseOccurrence) -> tuple[DenseOccurrence, ...]:
    result: list[DenseOccurrence] = []

    def visit(node: DenseOccurrence) -> None:
        for child in node.children:
            visit(child)
        result.append(node)

    visit(root)
    return tuple(result)


def _occurrence_parents(
    root: DenseOccurrence,
) -> dict[int, tuple[DenseOccurrence, int]]:
    parents: dict[int, tuple[DenseOccurrence, int]] = {}
    for parent in _occurrences_postorder(root):
        for role, child in enumerate(parent.children):
            if child.occurrence_id in parents:
                raise ValueError("no-memo occurrence appeared below two parents")
            parents[child.occurrence_id] = (parent, role)
    return parents


def _prepare_physical_sources(
    network: DenseOccurrenceNetwork,
    raw_input: Tensor | Mapping[str, Tensor] | Sequence[Tensor],
) -> tuple[int, Tensor | dict[str, Tensor]]:
    if not network.physical_sources:
        if not isinstance(raw_input, Tensor):
            raise TypeError("homogeneous reference input must be one tensor")
        expected = (network.token_count, network.feature_dimension)
        if raw_input.ndim != 3 or tuple(raw_input.shape[1:]) != expected:
            raise ValueError("homogeneous reference input has the wrong shape")
        if raw_input.dtype != network.head.dtype or raw_input.device != network.head.device:
            raise ValueError("homogeneous reference input has the wrong dtype or device")
        return int(raw_input.shape[0]), raw_input

    specs = network.physical_sources
    if isinstance(raw_input, Tensor):
        raise TypeError("heterogeneous reference input must be a mapping or sequence")
    if isinstance(raw_input, Mapping):
        if set(raw_input) != {spec.key for spec in specs}:
            raise ValueError("heterogeneous reference keys do not match the source registry")
        values = {spec.key: raw_input[spec.key] for spec in specs}
    else:
        sequence = tuple(raw_input)
        if len(sequence) != len(specs):
            raise ValueError("heterogeneous reference sequence has the wrong length")
        values = {spec.key: value for spec, value in zip(specs, sequence)}
    batch_size: int | None = None
    for spec in specs:
        value = values[spec.key]
        if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != spec.width:
            raise ValueError(f"heterogeneous source {spec.key!r} has the wrong shape")
        if value.dtype != network.head.dtype or value.device != network.head.device:
            raise ValueError("heterogeneous source has the wrong dtype or device")
        if batch_size is None:
            batch_size = int(value.shape[0])
        elif value.shape[0] != batch_size:
            raise ValueError("heterogeneous source batch sizes differ")
    if batch_size is None or batch_size < 1:
        raise ValueError("reference input batch must be nonempty")
    return batch_size, values


def _apply_dense_core(core: Tensor, inputs: tuple[Tensor, ...]) -> Tensor:
    if core.ndim == 2 and len(inputs) == 1:
        return inputs[0] @ core.T
    if core.ndim == 3 and len(inputs) == 2:
        return torch.einsum("oij,bi,bj->bo", core, inputs[0], inputs[1])
    raise ValueError("dense occurrence core and input arity disagree")


def _normalize_projective_rows(value: Tensor) -> Tensor:
    maximum = value.abs().amax(dim=1)
    if bool((maximum == 0).any()) or not bool(torch.isfinite(maximum).all()):
        raise ValueError("reference evaluation produced zero or nonfinite coordinates")
    _, exponent = torch.frexp(maximum)
    return torch.ldexp(value, -exponent[:, None])


def _normalize_scaled_rows(value: Tensor, exponent: Tensor) -> ReferenceScaledBatch:
    maximum = value.abs().amax(dim=1)
    if bool((maximum == 0).any()) or not bool(torch.isfinite(maximum).all()):
        raise ValueError("reference evaluation produced zero or nonfinite coordinates")
    _, local = torch.frexp(maximum)
    return ReferenceScaledBatch(torch.ldexp(value, -local[:, None]), exponent + local)


@torch.no_grad()
def reference_evaluate_projective(
    network: DenseOccurrenceNetwork,
    raw_input: Tensor | Mapping[str, Tensor] | Sequence[Tensor],
) -> Tensor:
    """Evaluate every occurrence recursively without memoization."""

    batch_size, prepared = _prepare_physical_sources(network, raw_input)

    def evaluate(node: DenseOccurrence) -> Tensor:
        if node.children:
            value = _apply_dense_core(
                node.tensor,
                tuple(evaluate(child) for child in node.children),
            )
        else:
            if node.physical_source_key is not None:
                if not isinstance(prepared, dict):
                    raise ValueError("heterogeneous leaf received homogeneous input")
                raw_value = prepared[node.physical_source_key]
            else:
                if node.physical_token is None or not isinstance(prepared, Tensor):
                    raise ValueError("reference leaf has no physical source")
                raw_value = prepared[:, node.physical_token]
            homogeneous = torch.cat(
                (raw_value, raw_value.new_ones(batch_size, 1)), dim=1
            )
            value = _apply_dense_core(node.tensor, (homogeneous,))
        return _normalize_projective_rows(value)

    root = evaluate(network.root)
    return _normalize_projective_rows(root @ network.head.T)


@torch.no_grad()
def reference_evaluate_homogeneous_scaled(
    network: DenseOccurrenceNetwork,
    raw_input: Tensor | Mapping[str, Tensor] | Sequence[Tensor],
) -> ReferenceScaledBatch:
    """Evaluate homogeneous coordinates with an explicit binary scale ledger."""

    batch_size, prepared = _prepare_physical_sources(network, raw_input)

    def evaluate(node: DenseOccurrence) -> ReferenceScaledBatch:
        if node.children:
            children = tuple(evaluate(child) for child in node.children)
            value = _apply_dense_core(
                node.tensor, tuple(child.mantissa for child in children)
            )
            exponent = torch.stack(
                tuple(child.binary_exponent for child in children)
            ).sum(dim=0)
        else:
            if node.physical_source_key is not None:
                if not isinstance(prepared, dict):
                    raise ValueError("heterogeneous leaf received homogeneous input")
                raw_value = prepared[node.physical_source_key]
            else:
                if node.physical_token is None or not isinstance(prepared, Tensor):
                    raise ValueError("reference leaf has no physical source")
                raw_value = prepared[:, node.physical_token]
            homogeneous = torch.cat(
                (raw_value, raw_value.new_ones(batch_size, 1)), dim=1
            )
            value = _apply_dense_core(node.tensor, (homogeneous,))
            exponent = torch.zeros(
                batch_size, dtype=torch.int64, device=network.head.device
            )
        return _normalize_scaled_rows(value, exponent + node.binary_exponent)

    root = evaluate(network.root)
    output = root.mantissa @ network.head.T
    return _normalize_scaled_rows(
        output, root.binary_exponent + network.head_binary_exponent
    )


def projective_relative_error(actual: Tensor, expected: Tensor) -> float:
    if actual.shape != expected.shape or actual.ndim != 2 or actual.shape[1] < 2:
        raise ValueError("projective comparison needs equally shaped coordinate matrices")
    actual_denominator = actual[:, -1:]
    expected_denominator = expected[:, -1:]
    epsilon = 100.0 * torch.finfo(actual.dtype).eps
    actual_chart = actual_denominator.abs() / actual.abs().amax(
        dim=1, keepdim=True
    ).clamp_min(torch.finfo(actual.dtype).tiny)
    expected_chart = expected_denominator.abs() / expected.abs().amax(
        dim=1, keepdim=True
    ).clamp_min(torch.finfo(expected.dtype).tiny)
    if bool((actual_chart <= epsilon).any() or (expected_chart <= epsilon).any()):
        raise ValueError("projective comparison encountered a zero denominator")
    left = actual[:, :-1] * expected_denominator
    right = expected[:, :-1] * actual_denominator
    residual = (left - right).norm(dim=1)
    scale = (
        left.norm(dim=1)
        + right.norm(dim=1)
        + (actual_denominator[:, 0] * expected_denominator[:, 0]).abs()
    ).clamp_min(torch.finfo(actual.dtype).tiny)
    return float((residual / scale).max().item())


def scaled_matrix_relative_error(
    actual: ReferenceScaledMatrix,
    expected: ReferenceScaledMatrix,
) -> float:
    if actual.mantissa.shape != expected.mantissa.shape:
        raise ValueError("scaled matrices have different shapes")
    center = max(actual.binary_exponent, expected.binary_exponent)
    left = _binary_shift(actual.mantissa, actual.binary_exponent - center)
    right = _binary_shift(expected.mantissa, expected.binary_exponent - center)
    scale = right.norm().clamp_min(torch.finfo(right.dtype).tiny)
    return float(((left - right).norm() / scale).item())


def scaled_batch_relative_error(
    actual: ReferenceScaledBatch,
    expected: ReferenceScaledBatch,
) -> float:
    if actual.mantissa.shape != expected.mantissa.shape:
        raise ValueError("scaled batches have different shapes")
    maximum = 0.0
    for index in range(actual.mantissa.shape[0]):
        maximum = max(
            maximum,
            scaled_matrix_relative_error(
                ReferenceScaledMatrix(
                    actual.mantissa[index : index + 1],
                    int(actual.binary_exponent[index].item()),
                ),
                ReferenceScaledMatrix(
                    expected.mantissa[index : index + 1],
                    int(expected.binary_exponent[index].item()),
                ),
            ),
        )
    return maximum


def reference_evaluate_quotient(
    network: DenseOccurrenceNetwork,
    raw_input: Tensor | Mapping[str, Tensor] | Sequence[Tensor],
) -> Tensor:
    pair = reference_evaluate_projective(network, raw_input)
    denominator = pair[:, -1:]
    relative = denominator.abs() / pair.abs().amax(dim=1, keepdim=True).clamp_min(
        torch.finfo(pair.dtype).tiny
    )
    if float(relative.min().item()) <= 100.0 * torch.finfo(pair.dtype).eps:
        raise ValueError("reference quotient denominator is numerically zero")
    return pair[:, :-1] / denominator


def physical_relative_error(actual: Tensor, expected: Tensor) -> float:
    if actual.shape != expected.shape:
        raise ValueError("physical tensors have different shapes")
    scale = expected.norm().clamp_min(torch.finfo(expected.dtype).tiny)
    return float(((actual - expected).norm() / scale).item())


def _independent_dense_clone_rq(
    tensor: Tensor,
    binary_exponent: int,
) -> tuple[ReferenceScaledMatrix, Tensor, float, float, bool]:
    """Return direct reduced RQ of one dense occurrence unfolding."""

    matrix = tensor.reshape(tensor.shape[0], -1)
    q_columns, upper = torch.linalg.qr(matrix.T, mode="reduced")
    signs = torch.where(
        torch.diagonal(upper) < 0,
        -torch.ones((), dtype=matrix.dtype, device=matrix.device),
        torch.ones((), dtype=matrix.dtype, device=matrix.device),
    )
    q_columns = q_columns * signs[None, :]
    upper = signs[:, None] * upper
    factor = upper.T
    q_rows = q_columns.T
    reconstructed = factor @ q_rows
    scale = matrix.norm().clamp_min(torch.finfo(matrix.dtype).tiny)
    error = float(((reconstructed - matrix).norm() / scale).item())
    diagonal = torch.diagonal(factor).abs()
    maximum = factor.abs().max().clamp_min(torch.finfo(factor.dtype).tiny)
    minimum_ratio = (
        float((diagonal.min() / maximum).item()) if diagonal.numel() else 0.0
    )
    literal_chart = minimum_ratio > 100.0 * torch.finfo(factor.dtype).eps
    factor_mantissa, local_exponent = _bounded_mantissa(factor)
    q_tensor = q_rows.reshape((q_rows.shape[0],) + tuple(tensor.shape[1:]))
    return (
        ReferenceScaledMatrix(
            factor_mantissa, binary_exponent + local_exponent
        ),
        q_tensor,
        error,
        minimum_ratio,
        literal_chart,
    )


def _absorb_occurrence_input(
    parent: DenseOccurrence,
    role: int,
    factor: ReferenceScaledMatrix,
    *,
    normalize: bool,
) -> None:
    if not 0 <= role < parent.tensor.ndim - 1:
        raise ValueError("reference parent role is out of range")
    axis = role + 1
    if parent.tensor.shape[axis] != factor.mantissa.shape[0]:
        raise ValueError("reference parent and factor dimensions disagree")
    transformed = (
        parent.tensor.movedim(axis, -1) @ factor.mantissa
    ).movedim(-1, axis)
    stripped = 0
    if normalize:
        transformed, stripped = _bounded_mantissa(transformed)
    parent.tensor = transformed
    parent.binary_exponent += factor.binary_exponent + stripped


def _selector_matches(selector: str | int, uid: int, label: str) -> bool:
    return selector == uid if isinstance(selector, int) else selector == label


@torch.no_grad()
def canonicalize_clone_reference_direct_rq(
    network: ImplicitProjectiveDAG,
    replay_inputs: Tensor | Mapping[str, Tensor] | Sequence[Tensor],
    *,
    maximum_local_elements: int = MAX_LOCAL_ELEMENTS,
    omit_factor_occurrence: tuple[str | int, int] | None = None,
) -> ReferenceCanonicalization:
    """Run Algorithm 1 on every occurrence, grouped by shared origin."""

    work = unfold_occurrences_no_memo(
        network, maximum_local_elements=maximum_local_elements
    )
    schedule = _schema_origin_schedule(network.root)
    original = reference_evaluate_projective(work, replay_inputs)
    parents = _occurrence_parents(work.root)
    records: list[ReferenceRQStep] = []
    omission_count = 0

    for step, (origin_uid, label) in enumerate(schedule):
        candidates = tuple(
            node
            for node in _occurrences_postorder(work.root)
            if node.origin_uid == origin_uid
        )
        if not candidates:
            raise RuntimeError("reference schedule lost an origin")
        factors: list[ReferenceScaledMatrix] = []
        q_tensors: list[Tensor] = []
        factorization_error = 0.0
        minimum_ratio = math.inf
        literal_chart = True
        expected_pushes = 0
        completed_pushes = 0
        shape: tuple[int, int] | None = None
        for occurrence_index, candidate in enumerate(candidates):
            current_shape = (
                candidate.output_dimension,
                math.prod(candidate.input_dimensions),
            )
            if shape is None:
                shape = current_shape
            elif current_shape != shape:
                raise ValueError("clone occurrences of one origin have different unfoldings")
            factor, q_tensor, error, ratio, resolved = _independent_dense_clone_rq(
                candidate.tensor, candidate.binary_exponent
            )
            candidate.tensor = q_tensor
            candidate.binary_exponent = 0
            factors.append(factor)
            q_tensors.append(q_tensor.clone())
            factorization_error = max(factorization_error, error)
            minimum_ratio = min(minimum_ratio, ratio)
            literal_chart = literal_chart and resolved
            expected_pushes += 1

            omit = (
                omit_factor_occurrence is not None
                and _selector_matches(
                    omit_factor_occurrence[0], origin_uid, label
                )
                and occurrence_index == omit_factor_occurrence[1]
            )
            if omit:
                if factor.mantissa.shape[0] != factor.mantissa.shape[1]:
                    raise ValueError("omitted reference R push requires a square factor")
                omission_count += 1
                continue
            parent = parents.get(candidate.occurrence_id)
            if parent is None:
                if candidate is not work.root:
                    raise ValueError("nonroot clone occurrence has no parent")
                if work.head.shape[1] != factor.mantissa.shape[0]:
                    raise ValueError("root reference factor does not match the head")
                transformed = work.head @ factor.mantissa
                transformed, stripped = _bounded_mantissa(transformed)
                work.head = transformed
                work.head_binary_exponent += factor.binary_exponent + stripped
            else:
                _absorb_occurrence_input(
                    parent[0], parent[1], factor, normalize=True
                )
            completed_pushes += 1

        coordinates = reference_evaluate_projective(work, replay_inputs)
        snapshot = _copy_occurrence_network(work)
        homogeneous = reference_evaluate_homogeneous_scaled(snapshot, replay_inputs)
        records.append(
            ReferenceRQStep(
                step,
                origin_uid,
                label,
                len(candidates),
                shape if shape is not None else (0, 0),
                tuple(factors),
                tuple(q_tensors),
                factorization_error,
                minimum_ratio,
                literal_chart,
                expected_pushes,
                completed_pushes,
                expected_pushes - completed_pushes,
                snapshot,
                homogeneous,
                coordinates.clone(),
                projective_relative_error(coordinates, original),
            )
        )

    if omit_factor_occurrence is not None and omission_count != 1:
        raise ValueError("requested reference R omission did not resolve exactly once")
    work.algorithm1_complete = omit_factor_occurrence is None
    return ReferenceCanonicalization(work, tuple(records), schedule, original)


def _schema_unique_nodes(root: ImplicitNode) -> tuple[ImplicitNode, ...]:
    completed: set[int] = set()
    active: set[int] = set()
    result: list[ImplicitNode] = []

    def visit(node: ImplicitNode) -> None:
        identity = id(node)
        if identity in active:
            raise ValueError("production schema contains a cycle")
        if identity in completed:
            return
        active.add(identity)
        for child in node.children:
            visit(child)
        active.remove(identity)
        completed.add(identity)
        result.append(node)

    visit(root)
    return tuple(result)


def compare_production_algorithm1_steps(
    production_result: Any,
    reference_result: ReferenceCanonicalization,
) -> tuple[ProductionStepComparison, ...]:
    """Compare every production Algorithm 1 Q core to all matching clones."""

    production_network = production_result.network
    production_steps = tuple(production_result.steps)
    if len(production_steps) != len(reference_result.steps):
        raise ValueError("production and reference Algorithm 1 step counts differ")
    nodes = {node.uid: node for node in _schema_unique_nodes(production_network.root)}
    comparisons: list[ProductionStepComparison] = []
    for production_step, reference_step in zip(
        production_steps, reference_result.steps
    ):
        if (
            production_step.step != reference_step.step
            or production_step.uid != reference_step.origin_uid
            or production_step.label != reference_step.label
        ):
            raise ValueError("production and reference Algorithm 1 schedules differ")
        node = nodes.get(reference_step.origin_uid)
        if node is None:
            raise ValueError("production result is missing a reference origin")
        production_dense = _schema_core_dense(node.core, MAX_LOCAL_ELEMENTS)
        maximum_q_error = 0.0
        for q_tensor in reference_step.q_tensors:
            maximum_q_error = max(
                maximum_q_error,
                scaled_matrix_relative_error(
                    ReferenceScaledMatrix(
                        production_dense.reshape(1, -1),
                        int(node.core.binary_exponent),
                    ),
                    ReferenceScaledMatrix(q_tensor.reshape(1, -1), 0),
                ),
            )
        comparisons.append(
            ProductionStepComparison(
                reference_step.step,
                reference_step.origin_uid,
                reference_step.label,
                maximum_q_error,
                float(production_step.factorization_relative_error),
                reference_step.maximum_factorization_relative_error,
                production_step.parent_occurrences_pushed
                == production_step.expected_parent_occurrences,
                reference_step.completed_factor_pushes
                == reference_step.expected_factor_pushes,
            )
        )
    return tuple(comparisons)


def compare_dense_occurrence_networks(
    actual: DenseOccurrenceNetwork,
    expected: DenseOccurrenceNetwork,
) -> ReferenceNetworkComparison:
    """Compare all dense clone cores and the head with binary scales intact."""

    actual_nodes = _occurrences_postorder(actual.root)
    expected_nodes = _occurrences_postorder(expected.root)
    if len(actual_nodes) != len(expected_nodes):
        raise ValueError("dense occurrence networks have different sizes")
    maximum_core_error = 0.0
    for actual_node, expected_node in zip(actual_nodes, expected_nodes):
        if (
            actual_node.occurrence_id != expected_node.occurrence_id
            or actual_node.origin_uid != expected_node.origin_uid
            or actual_node.label != expected_node.label
            or actual_node.tensor.shape != expected_node.tensor.shape
        ):
            raise ValueError("dense occurrence network structures differ")
        maximum_core_error = max(
            maximum_core_error,
            scaled_matrix_relative_error(
                ReferenceScaledMatrix(
                    actual_node.tensor.reshape(1, -1),
                    actual_node.binary_exponent,
                ),
                ReferenceScaledMatrix(
                    expected_node.tensor.reshape(1, -1),
                    expected_node.binary_exponent,
                ),
            ),
        )
    head_error = scaled_matrix_relative_error(
        ReferenceScaledMatrix(actual.head, actual.head_binary_exponent),
        ReferenceScaledMatrix(expected.head, expected.head_binary_exponent),
    )
    return ReferenceNetworkComparison(
        len(actual_nodes), maximum_core_error, head_error
    )


def _scale_symmetric(value: Tensor, exponent: int = 0) -> ReferenceScaledMatrix:
    symmetric = 0.5 * (value + value.T)
    mantissa, local = _bounded_mantissa(symmetric)
    if not bool((mantissa != 0).any()):
        return ReferenceScaledMatrix(mantissa, 0)
    return ReferenceScaledMatrix(mantissa, exponent + local)


def _sum_scaled_messages(
    messages: Sequence[ReferenceScaledMatrix],
) -> ReferenceScaledMatrix:
    if not messages:
        raise ValueError("cannot aggregate an empty environment collection")
    total = messages[0].mantissa.new_zeros(messages[0].mantissa.shape)
    for message in messages:
        if message.mantissa.shape != total.shape:
            raise ValueError("environment message shapes differ")
    nonzero = tuple(
        message for message in messages if bool((message.mantissa != 0).any())
    )
    if not nonzero:
        return ReferenceScaledMatrix(total, 0)
    center = max(message.binary_exponent for message in nonzero)
    for message in nonzero:
        relative_exponent = int(message.binary_exponent - center)
        if relative_exponent < torch.iinfo(torch.int64).min:
            raise ValueError(
                "nonzero reference Algorithm 2 environment message exponent is "
                "outside the direct scaled-contraction ledger"
            )
        shifted = _binary_shift(message.mantissa, relative_exponent)
        if bool(((message.mantissa != 0) & (shifted == 0)).any()):
            raise ValueError(
                "nonzero reference Algorithm 2 environment message underflowed "
                "during scaled contraction"
            )
        total = total + shifted
    return _scale_symmetric(total, center)


def _role_environment(core: Tensor, downstream: Tensor, role: int) -> Tensor:
    """Explicit downstream TN contraction with the sibling identity metric."""

    if core.ndim == 2:
        if role != 0:
            raise ValueError("unary occurrence has only role zero")
        result = torch.einsum("oi,op,pj->ij", core, downstream, core)
    elif core.ndim == 3 and role == 0:
        result = torch.einsum("oia,op,pja->ij", core, downstream, core)
    elif core.ndim == 3 and role == 1:
        result = torch.einsum("oai,op,paj->ij", core, downstream, core)
    else:
        raise ValueError("binary occurrence role is out of range")
    return 0.5 * (result + result.T)


@torch.no_grad()
def contract_clone_environments(
    network: DenseOccurrenceNetwork,
) -> tuple[OccurrenceEnvironment, ...]:
    """Contract Algorithm 2 separately down every no-memo occurrence path."""

    if not network.algorithm1_complete:
        raise ValueError("reference Algorithm 2 requires completed direct RQ")
    output_metric = torch.eye(
        network.head.shape[0],
        dtype=network.head.dtype,
        device=network.head.device,
    )
    root_dense = torch.einsum(
        "ao,ab,bp->op", network.head, output_metric, network.head
    )
    root_message = _scale_symmetric(
        root_dense, 2 * network.head_binary_exponent
    )
    records: list[OccurrenceEnvironment] = []

    def descend(node: DenseOccurrence, message: ReferenceScaledMatrix) -> None:
        records.append(
            OccurrenceEnvironment(
                node.occurrence_id,
                node.origin_uid,
                node.label,
                message.mantissa,
                message.binary_exponent,
                len(node.children),
            )
        )
        for role, child in enumerate(node.children):
            contracted = _role_environment(node.tensor, message.mantissa, role)
            child_message = _scale_symmetric(
                contracted,
                message.binary_exponent + 2 * node.binary_exponent,
            )
            descend(child, child_message)

    descend(network.root, root_message)
    if sum(record.child_messages_emitted for record in records) != len(records) - 1:
        raise RuntimeError("reference Algorithm 2 missed an occurrence edge")
    return tuple(records)


def aggregate_environments_by_origin(
    records: Sequence[OccurrenceEnvironment],
) -> dict[int, ReferenceScaledMatrix]:
    grouped: dict[int, list[ReferenceScaledMatrix]] = defaultdict(list)
    for record in records:
        grouped[record.origin_uid].append(
            ReferenceScaledMatrix(record.mantissa, record.binary_exponent)
        )
    return {
        origin_uid: _sum_scaled_messages(messages)
        for origin_uid, messages in grouped.items()
    }


def contract_and_aggregate_clone_environments(
    network: DenseOccurrenceNetwork,
) -> dict[int, ReferenceScaledMatrix]:
    return aggregate_environments_by_origin(contract_clone_environments(network))


def _offdiagonal_ratio(value: Tensor) -> float:
    offdiagonal = value - torch.diag(torch.diagonal(value))
    scale = value.norm().clamp_min(torch.finfo(value.dtype).tiny)
    return float((offdiagonal.norm() / scale).item())


@torch.no_grad()
def diagonalize_shared_and_explicit_clone_independently(
    canonical: ReferenceCanonicalization,
    replay_inputs: Tensor | Mapping[str, Tensor] | Sequence[Tensor],
    *,
    omit_gauge_occurrence: tuple[str | int, int] | None = None,
) -> ReferenceDiagonalization:
    """Run Algorithms 2 and 3 using only the dense occurrence-tree oracle."""

    if not canonical.network.algorithm1_complete:
        raise ValueError("reference Algorithm 3 requires completed direct RQ")
    work = _copy_occurrence_network(canonical.network)
    initial = reference_evaluate_projective(work, replay_inputs)
    aggregates = contract_and_aggregate_clone_environments(work)
    gauges: dict[int, Tensor] = {}
    eigenvalues: dict[int, ReferenceScaledMatrix] = {}
    for origin_uid, _ in canonical.origin_schedule:
        environment = aggregates[origin_uid]
        symmetric = 0.5 * (environment.mantissa + environment.mantissa.T)
        values, vectors = torch.linalg.eigh(symmetric)
        order = torch.argsort(values, descending=True)
        values = values[order]
        vectors = vectors[:, order]
        gauges[origin_uid] = vectors
        eigenvalues[origin_uid] = ReferenceScaledMatrix(
            torch.diag(values), environment.binary_exponent
        )

    parents = _occurrence_parents(work.root)
    expected_pushes = 0
    completed_pushes = 0
    omission_count = 0
    for origin_uid, label in canonical.origin_schedule:
        basis = gauges[origin_uid]
        candidates = tuple(
            node
            for node in _occurrences_postorder(work.root)
            if node.origin_uid == origin_uid
        )
        for occurrence_index, candidate in enumerate(candidates):
            candidate.tensor = torch.tensordot(
                basis.T, candidate.tensor, dims=([1], [0])
            )
            expected_pushes += 1
            parent = parents.get(candidate.occurrence_id)
            omit = (
                parent is not None
                and omit_gauge_occurrence is not None
                and _selector_matches(
                    omit_gauge_occurrence[0], origin_uid, label
                )
                and occurrence_index == omit_gauge_occurrence[1]
            )
            if omit:
                omission_count += 1
                continue
            if parent is None:
                if candidate is not work.root:
                    raise ValueError("nonroot gauge occurrence has no parent")
                work.head = work.head @ basis
            else:
                _absorb_occurrence_input(
                    parent[0],
                    parent[1],
                    ReferenceScaledMatrix(basis, 0),
                    normalize=False,
                )
            completed_pushes += 1

    if omit_gauge_occurrence is not None and omission_count != 1:
        raise ValueError("requested reference gauge omission did not resolve exactly once")
    final = reference_evaluate_projective(work, replay_inputs)
    aggregates_after = contract_and_aggregate_clone_environments(work)
    maximum_offdiagonal = max(
        (_offdiagonal_ratio(value.mantissa) for value in aggregates_after.values()),
        default=0.0,
    )
    return ReferenceDiagonalization(
        work,
        eigenvalues,
        gauges,
        initial,
        final,
        projective_relative_error(final, initial),
        maximum_offdiagonal,
        expected_pushes,
        completed_pushes,
        omission_count,
        aggregates_after,
    )


def compare_production_eigenvalues(
    production_result: Any,
    reference_result: ReferenceDiagonalization,
    origin_schedule: Sequence[tuple[int, str]],
) -> dict[int, float]:
    """Compare every production Algorithm 3 spectrum to the clone contraction."""

    production_values = tuple(production_result.eigenvalues)
    if len(production_values) != len(origin_schedule):
        raise ValueError("production and reference Algorithm 3 step counts differ")
    errors: dict[int, float] = {}
    for (origin_uid, _), production_value in zip(
        origin_schedule, production_values
    ):
        expected = reference_result.eigenvalues_by_origin[origin_uid]
        errors[origin_uid] = scaled_matrix_relative_error(
            ReferenceScaledMatrix(
                production_value.mantissa,
                int(production_value.binary_exponent),
            ),
            expected,
        )
    return errors


__all__ = [
    "DenseOccurrence",
    "DenseOccurrenceNetwork",
    "MAX_LOCAL_ELEMENTS",
    "MAX_OCCURRENCES",
    "OccurrenceEnvironment",
    "ProductionStepComparison",
    "ReferenceCanonicalization",
    "ReferenceDiagonalization",
    "ReferenceRQStep",
    "ReferenceNetworkComparison",
    "ReferenceScaledBatch",
    "ReferenceScaledMatrix",
    "aggregate_environments_by_origin",
    "canonicalize_clone_reference_direct_rq",
    "compare_production_algorithm1_steps",
    "compare_dense_occurrence_networks",
    "compare_production_eigenvalues",
    "contract_and_aggregate_clone_environments",
    "contract_clone_environments",
    "diagonalize_shared_and_explicit_clone_independently",
    "physical_relative_error",
    "projective_relative_error",
    "reference_evaluate_homogeneous_scaled",
    "reference_evaluate_projective",
    "reference_evaluate_quotient",
    "scaled_batch_relative_error",
    "scaled_matrix_relative_error",
    "unfold_occurrences_no_memo",
]
