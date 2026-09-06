"""Graph validation, traversal and decoded replay, separate from ODT sweeps."""

from __future__ import annotations

import math
from collections.abc import Mapping
from collections import defaultdict

import torch

from xvla.train.odt_engine_v2.constants import (
    MAXIMUM_RECTANGULAR_Q_ELEMENTS,
    STREAMED_DIRECT_Q_PROVENANCE_METHOD,
)

from xvla.train.odt_engine_v2.ops import (
    _core_apply,
    _core_arity,
    _core_clone,
    _normalize_batch,
    _normalize_projective_batch,
)

from xvla.train.odt_engine_v2.types import (
    CPBinaryCore,
    DenseCloneCore,
    ImplicitNode,
    ImplicitProjectiveDAG,
    PhysicalSourceSpec,
    RawPhysicalInput,
    ReducedQRBinaryCore,
    ScaledBatch,
    Tensor,
    UnaryCore,
    _validate_real_finite,
)

from xvla.train.odt_engine_v2.validation import (
    _validate_direct_q_provenance,
)
from xvla.train.odt_engine_v2.factorization import _validate_tied_input_symmetry


def _walk_unique(root: ImplicitNode) -> tuple[ImplicitNode, ...]:
    result: list[ImplicitNode] = []
    state: dict[int, int] = {}

    def visit(node: ImplicitNode) -> None:
        marker = state.get(id(node), 0)
        if marker == 1:
            raise ValueError("implicit projective graph contains a cycle")
        if marker == 2:
            return
        state[id(node)] = 1
        for child in node.children:
            visit(child)
        state[id(node)] = 2
        result.append(node)

    visit(root)
    return tuple(result)


def _validate_symmetric_lift(network: ImplicitProjectiveDAG) -> None:
    """Check the declared raw lift before child reductions can conceal skew."""
    _validate_network(network)
    for node in _walk_unique(network.root):
        if len(node.children) == 2 and node.children[0] is node.children[1]:
            _validate_tied_input_symmetry(node.core)


def _validate_network(network: ImplicitProjectiveDAG) -> None:
    if network.token_count < 1 or network.feature_dimension < 1:
        raise ValueError("implicit projective dimensions must be positive")
    if not 0 <= network.selected_token < network.token_count:
        raise ValueError("implicit selected token is invalid")
    _validate_real_finite(network.head, "implicit boundary head")
    if isinstance(network.head_binary_exponent, bool) or not isinstance(
        network.head_binary_exponent, int
    ):
        raise ValueError("implicit head exponent must be an integer")
    if not isinstance(network.algorithm1_scale_ledger_complete, bool):
        raise ValueError("Algorithm 1 scale-ledger state must be boolean")
    certified_head_exponent = network.algorithm1_certified_head_binary_exponent
    if certified_head_exponent is not None and (
        isinstance(certified_head_exponent, bool)
        or not isinstance(certified_head_exponent, int)
    ):
        raise ValueError("certified Algorithm 1 head exponent must be an integer")
    if network.head.ndim != 2 or network.head.shape[1] != network.root.output_dimension:
        raise ValueError("implicit boundary head/root dimensions disagree")
    if network.mask.shape != (network.token_count, network.token_count):
        raise ValueError("implicit fixed mask has the wrong shape")
    _validate_real_finite(network.mask, "implicit fixed mask")
    if network.mask.dtype != network.head.dtype or network.mask.device != network.head.device:
        raise ValueError("implicit mask and head must share dtype and device")
    if not bool(((network.mask == 0) | (network.mask == 1)).all()):
        raise ValueError("implicit fixed mask must be binary")
    source_specs: dict[str, PhysicalSourceSpec] = {}
    for spec in network.physical_sources:
        if spec.key in source_specs:
            raise ValueError("heterogeneous physical source keys must be unique")
        source_specs[spec.key] = spec
    nodes = _walk_unique(network.root)
    uid_owner: dict[int, int] = {}
    dtype = network.head.dtype
    device = network.head.device
    for node in nodes:
        owner = uid_owner.setdefault(node.uid, id(node))
        if owner != id(node):
            raise ValueError("distinct shared nodes must have unique uids")
        if isinstance(node.core.binary_exponent, bool) or not isinstance(
            node.core.binary_exponent, int
        ):
            raise ValueError("implicit core exponent must be an integer")
        if isinstance(node.core, UnaryCore):
            if node.core.matrix.ndim != 2:
                raise ValueError("implicit unary core must be a matrix")
            tensors = (node.core.matrix,)
        elif isinstance(node.core, DenseCloneCore):
            if node.core.tensor.ndim not in (2, 3):
                raise ValueError("dense clone core must be order two or three")
            tensors = (node.core.tensor,)
        elif isinstance(node.core, ReducedQRBinaryCore):
            if node.core.q_rows.ndim != 2:
                raise ValueError("reduced-QR binary rows must be a matrix")
            if node.core.left_dimension < 1 or node.core.right_dimension < 1:
                raise ValueError("reduced-QR binary input dimensions must be positive")
            if node.core.q_rows.shape[1] != (
                node.core.left_dimension * node.core.right_dimension
            ):
                raise ValueError("reduced-QR binary rows/input dimensions disagree")
            if node.core.q_rows.numel() > MAXIMUM_RECTANGULAR_Q_ELEMENTS:
                raise ValueError("reduced-QR binary core exceeds its bounded size")
            tensors = (node.core.q_rows,)
        else:
            if (
                node.core.output_factor.ndim != 2
                or node.core.left_factor.ndim != 2
                or node.core.right_factor.ndim != 2
            ):
                raise ValueError("implicit CP factors must be matrices")
            rank = node.core.output_factor.shape[1]
            if rank < 1 or node.core.left_factor.shape[0] != rank or node.core.right_factor.shape[0] != rank:
                raise ValueError("implicit CP factors must have one positive common rank")
            tensors = (
                node.core.output_factor,
                node.core.left_factor,
                node.core.right_factor,
            )
            if node.core.direct_q_provenance is not None:
                _validate_direct_q_provenance(
                    node.core.direct_q_provenance,
                    require_verified=False,
                )
        for value in tensors:
            _validate_real_finite(value, f"implicit core {node.label}")
            if value.dtype != dtype or value.device != device:
                raise ValueError("all implicit cores must share head dtype and device")
        if node.children:
            if node.physical_token is not None or node.physical_source_key is not None:
                raise ValueError("only physical leaves may select a raw source")
            if len(node.children) != _core_arity(node.core):
                raise ValueError("implicit node/core arity mismatch")
            if tuple(child.output_dimension for child in node.children) != node.core.input_dimensions:
                raise ValueError("implicit child/core bond mismatch")
        else:
            if source_specs:
                if node.physical_token is not None or node.physical_source_key not in source_specs:
                    raise ValueError("heterogeneous leaf must select one declared physical source")
                expected_physical_input = source_specs[node.physical_source_key].width + 1
            else:
                if (
                    node.physical_source_key is not None
                    or node.physical_token is None
                    or not 0 <= node.physical_token < network.token_count
                ):
                    raise ValueError("implicit leaf must select one physical token")
                expected_physical_input = network.feature_dimension + 1
            if isinstance(node.core, UnaryCore):
                physical_input = node.core.matrix.shape[1]
            elif isinstance(node.core, DenseCloneCore) and node.core.tensor.ndim == 2:
                physical_input = node.core.tensor.shape[1]
            else:
                physical_input = -1
            if physical_input != expected_physical_input:
                raise ValueError("implicit physical leaf has the wrong homogeneous width")
    if source_specs:
        reached = {node.physical_source_key for node in nodes if not node.children}
        if reached != set(source_specs):
            raise ValueError("heterogeneous source registry must match reachable physical leaves")


def validate_canonical_exponent_normal_form(
    network: ImplicitProjectiveDAG,
) -> dict[str, int | bool]:
    """Validate the scale normal form required by Algorithms 2 and 3.

    Algorithm 1 moves every local binary exponent upward.  Thus every
    canonical core must have exponent zero, while the sole remaining global
    exponent lives at the boundary head and must equal the value sealed by the
    local reconstruction/absorption ledger.
    """

    _validate_network(network)
    if not network.algorithm1_direct_rq_complete:
        raise ValueError(
            "canonical exponent normal form requires completed direct-RQ Algorithm 1"
        )
    if not network.algorithm1_scale_ledger_complete:
        raise ValueError(
            "canonical exponent normal form requires a completed scale ledger"
        )
    nodes = _walk_unique(network.root)
    invalid = tuple(
        (node.uid, node.label, int(node.core.binary_exponent))
        for node in nodes
        if node.core.binary_exponent != 0
    )
    if invalid:
        uid, label, exponent = invalid[0]
        raise ValueError(
            "canonical core exponent is not zero: "
            f"uid={uid} label={label!r} exponent={exponent}"
        )
    certified = network.algorithm1_certified_head_binary_exponent
    if certified is None or network.head_binary_exponent != certified:
        raise ValueError(
            "canonical boundary-head exponent differs from the Algorithm 1 scale ledger"
        )
    cp_certificates = 0
    cp_columns = 0
    streamed_certificates = 0
    streamed_columns = 0
    for node in nodes:
        if not isinstance(node.core, CPBinaryCore):
            continue
        provenance = node.core.direct_q_provenance
        if provenance is None:
            raise ValueError(
                "canonical compact CP Q lacks direct TSQR Q provenance: "
                f"uid={node.uid} label={node.label!r}"
            )
        _validate_direct_q_provenance(provenance, require_verified=True)
        expected_columns = math.prod(node.core.input_dimensions)
        if provenance.columns_compared != expected_columns:
            raise ValueError(
                "canonical compact CP Q provenance does not cover every column"
            )
        cp_certificates += 1
        cp_columns += provenance.columns_compared
        if provenance.method == STREAMED_DIRECT_Q_PROVENANCE_METHOD:
            streamed_certificates += 1
            streamed_columns += provenance.columns_compared
    return {
        "core_count": len(nodes),
        "all_core_exponents_zero": True,
        "head_binary_exponent": int(network.head_binary_exponent),
        "head_exponent_matches_ledger": True,
        "cp_direct_q_certificates": cp_certificates,
        "cp_direct_q_columns": cp_columns,
        "streamed_compact_cp_direct_q_certificates": streamed_certificates,
        "streamed_compact_cp_direct_q_columns": streamed_columns,
        "all_streamed_compact_cp_q_has_direct_q_provenance": True,
    }


def _topological_root_first(root: ImplicitNode) -> tuple[ImplicitNode, ...]:
    return tuple(reversed(_walk_unique(root)))


def _parent_occurrences(
    root: ImplicitNode,
) -> dict[int, list[tuple[ImplicitNode, int]]]:
    parents: dict[int, list[tuple[ImplicitNode, int]]] = defaultdict(list)
    for parent in _walk_unique(root):
        for role, child in enumerate(parent.children):
            parents[id(child)].append((parent, role))
    return parents


def _clone_network(network: ImplicitProjectiveDAG, *, unfold: bool) -> ImplicitProjectiveDAG:
    memo: dict[int, ImplicitNode] = {}
    next_clone_uid = 0

    def clone(node: ImplicitNode) -> ImplicitNode:
        nonlocal next_clone_uid
        if not unfold and id(node) in memo:
            return memo[id(node)]
        instance_uid = next_clone_uid if unfold else node.uid
        if unfold:
            next_clone_uid += 1
        result = ImplicitNode(
            instance_uid,
            node.label,
            _core_clone(node.core),
            tuple(clone(child) for child in node.children),
            node.physical_token,
            node.uid if node.origin_uid is None else node.origin_uid,
            node.physical_source_key,
        )
        if not unfold:
            memo[id(node)] = result
        return result

    return ImplicitProjectiveDAG(
        root=clone(network.root),
        head=network.head.clone(),
        head_binary_exponent=network.head_binary_exponent,
        token_count=network.token_count,
        feature_dimension=network.feature_dimension,
        selected_token=network.selected_token,
        mask=network.mask.clone(),
        claim_boundary=network.claim_boundary,
        algorithm1_direct_rq_complete=network.algorithm1_direct_rq_complete,
        physical_sources=network.physical_sources,
        algorithm1_scale_ledger_complete=network.algorithm1_scale_ledger_complete,
        algorithm1_certified_head_binary_exponent=(
            network.algorithm1_certified_head_binary_exponent
        ),
    )


@torch.no_grad()
def _prepare_physical_input(
    network: ImplicitProjectiveDAG,
    raw_input: RawPhysicalInput,
) -> tuple[int, Tensor | dict[str, Tensor]]:
    if not network.physical_sources:
        if not isinstance(raw_input, Tensor):
            raise TypeError("homogeneous implicit DAG input must be one tensor")
        if raw_input.ndim != 3 or tuple(raw_input.shape[1:]) != (
            network.token_count,
            network.feature_dimension,
        ):
            raise ValueError("raw input shape does not match implicit DAG")
        if raw_input.dtype != network.head.dtype or raw_input.device != network.head.device:
            raise ValueError("raw input must share the implicit DAG dtype and device")
        return int(raw_input.shape[0]), raw_input

    specs = network.physical_sources
    if isinstance(raw_input, Tensor):
        raise TypeError("heterogeneous implicit DAG input must be a mapping or tuple")
    if isinstance(raw_input, Mapping):
        if set(raw_input) != {spec.key for spec in specs}:
            raise ValueError("raw physical-source mapping keys do not match the registry")
        values = {spec.key: raw_input[spec.key] for spec in specs}
    else:
        sequence = tuple(raw_input)
        if len(sequence) != len(specs):
            raise ValueError("raw physical-source tuple length does not match the registry")
        values = {spec.key: value for spec, value in zip(specs, sequence)}
    batch_size: int | None = None
    for spec in specs:
        value = values[spec.key]
        if not isinstance(value, Tensor) or value.ndim != 2:
            raise ValueError("each heterogeneous physical source must be a batch matrix")
        if value.shape[1] != spec.width:
            raise ValueError(f"physical source {spec.key!r} has the wrong width")
        if value.dtype != network.head.dtype or value.device != network.head.device:
            raise ValueError("physical sources must share the implicit DAG dtype and device")
        if batch_size is None:
            batch_size = int(value.shape[0])
        elif value.shape[0] != batch_size:
            raise ValueError("heterogeneous physical sources have different batch sizes")
    if batch_size is None or batch_size < 1:
        raise ValueError("heterogeneous physical input must contain a nonempty batch")
    return batch_size, values


@torch.no_grad()
def evaluate_scaled_boundary(
    network: ImplicitProjectiveDAG,
    raw_input: RawPhysicalInput,
) -> ScaledBatch:
    _validate_network(network)
    batch_size, prepared = _prepare_physical_input(network, raw_input)
    memo: dict[int, ScaledBatch] = {}

    def evaluate(node: ImplicitNode) -> ScaledBatch:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        if node.children:
            children = tuple(evaluate(child) for child in node.children)
            value = _core_apply(node.core, tuple(child.mantissa for child in children))
            exponent = torch.stack(tuple(child.binary_exponent for child in children)).sum(0)
        else:
            if node.physical_source_key is not None:
                if not isinstance(prepared, dict):
                    raise ValueError("heterogeneous leaf received homogeneous input")
                raw_value = prepared[node.physical_source_key]
            else:
                if node.physical_token is None or not isinstance(prepared, Tensor):
                    raise ValueError("implicit leaf has no physical token")
                raw_value = prepared[:, node.physical_token]
            physical = torch.cat(
                (
                    raw_value,
                    raw_value.new_ones(batch_size, 1),
                ),
                dim=1,
            )
            value = _core_apply(node.core, (physical,))
            exponent = torch.zeros(
                batch_size, dtype=torch.int64, device=network.head.device
            )
        exponent = exponent + node.core.binary_exponent
        result = _normalize_batch(value, exponent)
        memo[id(node)] = result
        return result

    root = evaluate(network.root)
    output = root.mantissa @ network.head.T
    return _normalize_batch(output, root.binary_exponent + network.head_binary_exponent)


@torch.no_grad()
def evaluate_projective_boundary(
    network: ImplicitProjectiveDAG,
    raw_input: RawPhysicalInput,
) -> Tensor:
    """Evaluate bounded homogeneous coordinates, discarding common scales.

    Every primitive is homogeneous in each input.  Consequently each child,
    core, and head binary exponent contributes only one common scalar to its
    downstream projective value and cannot affect the quotient.  Omitting that
    scalar ledger prevents the exponentially growing polynomial degree of a
    deep policy from overflowing a fixed-width sample exponent while retaining
    the exact rational function.
    """

    _validate_network(network)
    batch_size, prepared = _prepare_physical_input(network, raw_input)
    memo: dict[int, Tensor] = {}

    def evaluate(node: ImplicitNode) -> Tensor:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        if node.children:
            children = tuple(evaluate(child) for child in node.children)
            value = _core_apply(node.core, children)
        else:
            if node.physical_source_key is not None:
                if not isinstance(prepared, dict):
                    raise ValueError("heterogeneous leaf received homogeneous input")
                raw_value = prepared[node.physical_source_key]
            else:
                if node.physical_token is None or not isinstance(prepared, Tensor):
                    raise ValueError("implicit leaf has no physical token")
                raw_value = prepared[:, node.physical_token]
            physical = torch.cat(
                (raw_value, raw_value.new_ones(batch_size, 1)), dim=1
            )
            value = _core_apply(node.core, (physical,))
        result = _normalize_projective_batch(value)
        memo[id(node)] = result
        return result

    root = evaluate(network.root)
    return _normalize_projective_batch(root @ network.head.T)


@torch.no_grad()
def evaluate_boundary_quotient(
    network: ImplicitProjectiveDAG,
    raw_input: RawPhysicalInput,
) -> Tensor:
    pair = evaluate_projective_boundary(network, raw_input)
    denominator = pair[:, -1]
    relative = denominator.abs() / pair.abs().amax(dim=1).clamp_min(
        torch.finfo(pair.dtype).tiny
    )
    if float(relative.min().item()) <= 100.0 * torch.finfo(pair.dtype).eps:
        raise ValueError("implicit projective denominator is numerically zero")
    return pair[:, :-1] / denominator[:, None]
