"""Exact typed projective-DAG ODT oracle for a real width-two Chi block.

This prototype keeps every hidden bond local.  A scalar rational value uses a
two-dimensional ``[numerator, denominator]`` bond and an ``m``-vector rational
value uses an ``m + 1`` dimensional ``[numerator_vector, denominator]`` bond.
Attention is compiled from typed affine, norm, dot, product, vector-scale, and
add primitives.  No layer core on a fused token-feature state is formed.

The computation graph is a DAG because residual streams and projections share
upstream values.  Its metric is the Dooms-style clone-unfolded *syntactic*
coefficient metric.  It is not the polynomial coefficient metric obtained by
first identifying every repeated raw variable.  Algorithm 1 canonicalizes a
shared node once and absorbs its factor into every parent role.  Algorithm 2
aggregates all clone-occurrence environments in reverse topological order,
with binary-exponent alignment.  Algorithm 3 applies one common full-rank
eigenbasis to a shared node and every one of its parent roles.  It diagonalizes
that summed occurrence Gram, not necessarily each occurrence Gram separately.
All quotient evaluation remains projective until the single final action
decode.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from string import ascii_lowercase
from typing import Iterable

import torch
import torch.nn as nn

from xvla.nn.attention import BilinearAttention, causal_mask
from xvla.nn.bilinear import BilinearFFN
from xvla.nn.block import ChiTransformerBlock
from xvla.nn.normalization import RationalNorm
from xvla.train.canonical_odt import reduced_rq_rows
from xvla.train.generalized_metric_odt import role_environment
from xvla.train.global_projective_odt_oracle import (
    ScaledProjectiveEvaluation,
    add_scaled_symmetric_messages,
    brute_identity_role_environment,
    omit_quotient_cross_terms_negative_control,
    quotient_pair_output_metric,
    scale_symmetric_message,
    scaled_projective_relative_error,
    scaled_projective_statistics,
)


Tensor = torch.Tensor
OBJECT_KIND = "typed_local_bond_projective_dag"
FIXED_GAUGE = "typed_pair_unit_clone_coefficient_norm_D_at_zero_positive"
UNFIXED_GAUGE = "typed_pair_constant_scalar_gauge_unfixed"
CLAIM_BOUNDARY = (
    "Exact full-rank common-basis ODT of the clone-unfolded syntactic metric "
    "for one fixed-gauge, typed local-bond projective ChiTransformerBlock DAG. "
    "The common basis diagonalizes the aggregate occurrence Gram. This is not "
    "an identified-variable polynomial metric or quotient-intrinsic ODT. This "
    "is a one-block construction, not a complete VLA traversal or compression "
    "certificate."
)


@dataclass(frozen=True)
class TypedProjectiveNode:
    label: str
    core: Tensor
    children: tuple["TypedProjectiveNode", ...] = ()
    physical_token: int | None = None
    prefix_metric: Tensor | None = field(default=None, repr=False, compare=False)

    @property
    def output_dimension(self) -> int:
        return int(self.core.shape[0])

    @property
    def arity(self) -> int:
        return len(self.children)


@dataclass(frozen=True)
class TypedProjectiveDAG:
    root: TypedProjectiveNode
    head: Tensor
    token_count: int
    feature_dimension: int
    gauge: str = UNFIXED_GAUGE
    object_kind: str = OBJECT_KIND
    claim_boundary: str = CLAIM_BOUNDARY


@dataclass(frozen=True)
class TypedDAGOracle:
    network: TypedProjectiveDAG
    block: ChiTransformerBlock
    action_head: nn.Linear
    positions: Tensor
    token_index: int
    mask: Tensor
    norm_buffer_snapshot: tuple[tuple[str, Tensor], ...]
    unfixed_log10_absolute_denominator_at_zero: float
    old_anchor_refix_predicted_head_log10_maximum: float
    fixed_coefficient_norm: float


@dataclass(frozen=True)
class CanonicalTypedDAG:
    network: TypedProjectiveDAG
    isometry_errors: tuple[float, ...]
    factorization_errors: tuple[float, ...]
    unique_node_count: int
    edge_occurrence_count: int
    factorization_methods: tuple[str, ...]


@dataclass(frozen=True)
class TypedEnvironment:
    node_identity: int
    label: str
    gram: Tensor
    binary_exponent: int
    incoming_occurrences: int


@dataclass(frozen=True)
class CloneEnvironment:
    path: tuple[int, ...]
    node_identity: int
    label: str
    gram: Tensor
    binary_exponent: int


@dataclass(frozen=True)
class DiagonalTypedDAG:
    network: TypedProjectiveDAG
    eigenvalues: tuple[Tensor, ...]
    eigenvalue_binary_exponents: tuple[int, ...]
    diagonalization_errors: tuple[float, ...]
    labels: tuple[str, ...]


def _validate_real_finite(value: Tensor, *, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point() or value.is_complex():
        raise TypeError(f"{name} must have a real floating dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _walk_unique(root: TypedProjectiveNode) -> tuple[TypedProjectiveNode, ...]:
    result: list[TypedProjectiveNode] = []
    state: dict[int, int] = {}

    def visit(node: TypedProjectiveNode) -> None:
        marker = state.get(id(node), 0)
        if marker == 1:
            raise ValueError("typed projective graph contains a cycle")
        if marker == 2:
            return
        state[id(node)] = 1
        result.append(node)
        for child in node.children:
            visit(child)
        state[id(node)] = 2

    visit(root)
    return tuple(result)


def _edge_occurrence_count(root: TypedProjectiveNode) -> int:
    return sum(len(node.children) for node in _walk_unique(root))


def _topological_nodes(root: TypedProjectiveNode) -> tuple[TypedProjectiveNode, ...]:
    nodes = _walk_unique(root)
    by_identity = {id(node): node for node in nodes}
    incoming = {identity: 0 for identity in by_identity}
    for node in nodes:
        for child in node.children:
            incoming[id(child)] += 1
    ready = deque(node for node in nodes if incoming[id(node)] == 0)
    ordered: list[TypedProjectiveNode] = []
    while ready:
        node = ready.popleft()
        ordered.append(node)
        for child in node.children:
            incoming[id(child)] -= 1
            if incoming[id(child)] == 0:
                ready.append(child)
    if len(ordered) != len(nodes):
        raise ValueError("typed projective graph is cyclic")
    if not ordered or ordered[0] is not root:
        raise ValueError("typed projective graph must have exactly one reachable root")
    return tuple(ordered)


def _validate_network(network: TypedProjectiveDAG) -> None:
    if network.object_kind != OBJECT_KIND:
        raise ValueError("unsupported typed projective object kind")
    if network.gauge not in {FIXED_GAUGE, UNFIXED_GAUGE}:
        raise ValueError("unknown typed projective gauge")
    if network.token_count < 2 or network.feature_dimension < 1:
        raise ValueError("typed projective dimensions must be positive")
    _validate_real_finite(network.head, name="typed projective head")
    if network.head.ndim != 2 or network.head.shape != (
        2,
        network.root.output_dimension,
    ):
        raise ValueError("head must map the root bond to one projective scalar pair")
    dtype = network.head.dtype
    device = network.head.device
    for node in _walk_unique(network.root):
        if not node.label:
            raise ValueError("typed nodes require labels")
        _validate_real_finite(node.core, name=f"core {node.label!r}")
        if node.core.dtype != dtype or node.core.device != device:
            raise ValueError("all typed cores and the head must share dtype and device")
        if node.children:
            if node.physical_token is not None:
                raise ValueError("only physical leaves may select a token")
            if node.core.ndim != node.arity + 1:
                raise ValueError(f"core arity mismatch at {node.label!r}")
            expected = tuple(child.output_dimension for child in node.children)
            if tuple(node.core.shape[1:]) != expected:
                raise ValueError(f"child bond mismatch at {node.label!r}")
        else:
            if node.physical_token is None or not 0 <= node.physical_token < network.token_count:
                raise ValueError("every leaf must select one valid physical token")
            if node.core.shape != (
                node.output_dimension,
                network.feature_dimension + 1,
            ):
                raise ValueError("physical leaf core has the wrong shape")
        if node.prefix_metric is not None:
            _validate_real_finite(node.prefix_metric, name=f"prefix metric {node.label!r}")
            if node.prefix_metric.shape != (node.output_dimension, node.output_dimension):
                raise ValueError("typed prefix metric has the wrong shape")
    _topological_nodes(network.root)


def _coefficient_prefix_metric(
    core: Tensor,
    children: tuple[TypedProjectiveNode, ...],
) -> Tensor:
    rows = core.reshape(core.shape[0], -1)
    if not children:
        metric = rows @ rows.T
    else:
        product = core.new_ones(1, 1)
        for child in children:
            if child.prefix_metric is None:
                raise ValueError("typed construction requires normalized child metrics")
            product = torch.kron(product, child.prefix_metric)
        metric = rows @ product @ rows.T
    return 0.5 * (metric + metric.T)


def _normalized_node(
    label: str,
    core: Tensor,
    children: tuple[TypedProjectiveNode, ...] = (),
    *,
    physical_token: int | None = None,
) -> TypedProjectiveNode:
    metric = _coefficient_prefix_metric(core, children)
    norm = torch.diagonal(metric).sum().sqrt()
    if not bool(torch.isfinite(norm)) or float(norm.item()) == 0.0:
        raise ValueError(f"typed node {label!r} has zero or nonfinite coefficient norm")
    return TypedProjectiveNode(
        label,
        core / norm,
        children,
        physical_token,
        metric / norm.square(),
    )


def _einsum_evaluate(core: Tensor, values: tuple[Tensor, ...]) -> Tensor:
    labels = "".join(label for label in ascii_lowercase if label not in {"b", "o"})
    if len(values) > len(labels):
        raise ValueError("typed core arity exceeds the einsum label budget")
    roles = labels[: len(values)]
    equation = f"o{roles}," + ",".join(f"b{role}" for role in roles) + "->bo"
    return torch.einsum(equation, core, *values)


def _scaled_coordinates(value: Tensor, inherited: Tensor) -> ScaledProjectiveEvaluation:
    scale = value.abs().amax(dim=1)
    if bool((scale == 0).any()) or not bool(torch.isfinite(scale).all()):
        raise ValueError("typed projective contraction produced zero/nonfinite coordinates")
    _, exponent = torch.frexp(scale)
    return ScaledProjectiveEvaluation(
        torch.ldexp(value, -exponent[:, None]),
        inherited + exponent,
    )


@torch.no_grad()
def evaluate_scaled_typed_pair(
    network: TypedProjectiveDAG,
    raw_input: Tensor,
) -> ScaledProjectiveEvaluation:
    _validate_network(network)
    _validate_real_finite(raw_input, name="raw_input")
    expected = (raw_input.shape[0], network.token_count, network.feature_dimension)
    if raw_input.ndim != 3 or raw_input.shape != expected:
        raise ValueError(
            "raw_input must have shape (batch, token_count, feature_dimension)"
        )
    if raw_input.dtype != network.head.dtype or raw_input.device != network.head.device:
        raise ValueError("raw input and typed network must share dtype and device")
    memo: dict[int, ScaledProjectiveEvaluation] = {}

    def evaluate(node: TypedProjectiveNode) -> ScaledProjectiveEvaluation:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        if node.children:
            children = tuple(evaluate(child) for child in node.children)
            value = _einsum_evaluate(node.core, tuple(child.mantissa for child in children))
            inherited = torch.stack(tuple(child.binary_exponent for child in children)).sum(0)
        else:
            physical = torch.cat(
                (
                    raw_input[:, node.physical_token],
                    raw_input.new_ones(raw_input.shape[0], 1),
                ),
                dim=1,
            )
            value = physical @ node.core.T
            inherited = torch.zeros(value.shape[0], dtype=torch.int64, device=value.device)
        result = _scaled_coordinates(value, inherited)
        memo[id(node)] = result
        return result

    root = evaluate(network.root)
    return _scaled_coordinates(root.mantissa @ network.head.T, root.binary_exponent)


@torch.no_grad()
def evaluate_typed_quotient(network: TypedProjectiveDAG, raw_input: Tensor) -> Tensor:
    pair = evaluate_scaled_typed_pair(network, raw_input).mantissa
    relative = pair[:, -1].abs() / pair.abs().amax(dim=1).clamp_min(
        torch.finfo(pair.dtype).tiny
    )
    if float(relative.min().item()) <= 100.0 * torch.finfo(pair.dtype).eps:
        raise ValueError("typed projective denominator is too small relative to its pair")
    return pair[:, 0] / pair[:, 1]


@torch.no_grad()
def evaluate_typed_pair_unscaled_negative_control(
    network: TypedProjectiveDAG,
    raw_input: Tensor,
) -> Tensor:
    _validate_network(network)
    memo: dict[int, Tensor] = {}

    def evaluate(node: TypedProjectiveNode) -> Tensor:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        if node.children:
            value = _einsum_evaluate(node.core, tuple(evaluate(child) for child in node.children))
        else:
            physical = torch.cat(
                (
                    raw_input[:, node.physical_token],
                    raw_input.new_ones(raw_input.shape[0], 1),
                ),
                dim=1,
            )
            value = physical @ node.core.T
        memo[id(node)] = value
        return value

    return evaluate(network.root) @ network.head.T


def _physical_leaf(token: int, dimension: int, like: Tensor) -> TypedProjectiveNode:
    return _normalized_node(
        f"input.token{token}",
        torch.eye(dimension + 1, dtype=like.dtype, device=like.device),
        physical_token=token,
    )


def _affine(
    value: TypedProjectiveNode,
    weight: Tensor,
    bias: Tensor,
    label: str,
) -> TypedProjectiveNode:
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise ValueError("typed affine weight/bias shape mismatch")
    if value.output_dimension != weight.shape[1] + 1:
        raise ValueError("typed affine input bond mismatch")
    core = weight.new_zeros(weight.shape[0] + 1, weight.shape[1] + 1)
    core[:-1, :-1] = weight
    core[:-1, -1] = bias
    core[-1, -1] = 1.0
    return _normalized_node(label, core, (value,))


def _scale_vector(
    value: TypedProjectiveNode,
    scale: Tensor | float,
    label: str,
) -> TypedProjectiveNode:
    dimension = value.output_dimension - 1
    scalar = torch.as_tensor(scale, dtype=value.core.dtype, device=value.core.device).reshape(())
    weight = torch.eye(dimension, dtype=value.core.dtype, device=value.core.device) * scalar
    return _affine(value, weight, value.core.new_zeros(dimension), label)


def _add_vectors(
    left: TypedProjectiveNode,
    right: TypedProjectiveNode,
    label: str,
) -> TypedProjectiveNode:
    if left.output_dimension != right.output_dimension:
        raise ValueError("typed vector addition needs equal bond dimensions")
    size = left.output_dimension
    denominator = size - 1
    core = left.core.new_zeros(size, size, size)
    for index in range(denominator):
        core[index, index, denominator] = 1.0
        core[index, denominator, index] = 1.0
    core[denominator, denominator, denominator] = 1.0
    return _normalized_node(label, core, (left, right))


def _dot_vectors(
    left: TypedProjectiveNode,
    right: TypedProjectiveNode,
    label: str,
) -> TypedProjectiveNode:
    if left.output_dimension != right.output_dimension:
        raise ValueError("typed dot product needs equal vector widths")
    denominator = left.output_dimension - 1
    core = left.core.new_zeros(2, denominator + 1, denominator + 1)
    for index in range(denominator):
        core[0, index, index] = 1.0
    core[1, denominator, denominator] = 1.0
    return _normalized_node(label, core, (left, right))


def _multiply_scalars(
    left: TypedProjectiveNode,
    right: TypedProjectiveNode,
    label: str,
) -> TypedProjectiveNode:
    if left.output_dimension != 2 or right.output_dimension != 2:
        raise ValueError("typed scalar product needs two scalar pairs")
    core = left.core.new_zeros(2, 2, 2)
    core[0, 0, 0] = 1.0
    core[1, 1, 1] = 1.0
    return _normalized_node(label, core, (left, right))


def _scale_vector_by_scalar(
    scalar: TypedProjectiveNode,
    vector: TypedProjectiveNode,
    label: str,
) -> TypedProjectiveNode:
    if scalar.output_dimension != 2:
        raise ValueError("first typed product input must be a scalar pair")
    size = vector.output_dimension
    denominator = size - 1
    core = vector.core.new_zeros(size, 2, size)
    for index in range(denominator):
        core[index, 0, index] = 1.0
    core[denominator, 1, denominator] = 1.0
    return _normalized_node(label, core, (scalar, vector))


def _hadamard_vectors(
    left: TypedProjectiveNode,
    right: TypedProjectiveNode,
    label: str,
) -> TypedProjectiveNode:
    if left.output_dimension != right.output_dimension:
        raise ValueError("typed Hadamard product needs equal vector widths")
    size = left.output_dimension
    denominator = size - 1
    core = left.core.new_zeros(size, size, size)
    for index in range(denominator):
        core[index, index, index] = 1.0
    core[denominator, denominator, denominator] = 1.0
    return _normalized_node(label, core, (left, right))


def _homogeneous_pade_core(module: RationalNorm, like: Tensor) -> Tensor:
    degree = int(module.deg)
    core = like.new_zeros((2,) + (2,) * degree)
    for power in range(degree + 1):
        indices = (0,) * power + (1,) * (degree - power)
        permutations = tuple(set(itertools.permutations(indices)))
        for output, coefficients in enumerate((module.pa, module.pb)):
            coefficient = coefficients[power].to(dtype=like.dtype, device=like.device)
            for permutation in permutations:
                core[(output, *permutation)] = coefficient / len(permutations)
    return core


def _pade_norm(
    value: TypedProjectiveNode,
    module: RationalNorm,
    label: str,
) -> TypedProjectiveNode:
    if not isinstance(module, RationalNorm) or module.variant != "pade":
        raise ValueError("typed normalization requires a Padé RationalNorm")
    dimension = value.output_dimension - 1
    if dimension < 1 or not bool(module.initialized):
        raise ValueError("typed normalization requires initialized vector input")
    scale = module.running_ms.to(dtype=value.core.dtype, device=value.core.device).clamp_min(1e-12)
    st_core = value.core.new_zeros(2, dimension + 1, dimension + 1)
    for index in range(dimension):
        st_core[0, index, index] = 1.0 / dimension
    st_core[0, dimension, dimension] = float(module.eps)
    st_core[1, dimension, dimension] = scale
    st = _normalized_node(f"{label}.mean_square_homogeneous", st_core, (value, value))
    pade = _normalized_node(
        f"{label}.pade_homogeneous",
        _homogeneous_pade_core(module, value.core),
        (st,) * int(module.deg),
    )
    output_core = value.core.new_zeros(dimension + 1, dimension + 1, 2)
    for index in range(dimension):
        output_core[index, index, 0] = torch.rsqrt(scale)
    output_core[dimension, dimension, 1] = 1.0
    return _normalized_node(f"{label}.projective_output", output_core, (value, pade))


def _norm_snapshot(block: ChiTransformerBlock) -> tuple[tuple[str, Tensor], ...]:
    result: list[tuple[str, Tensor]] = []
    for name, module in block.named_modules():
        if isinstance(module, RationalNorm):
            result.extend(
                (
                    (f"{name}.running_ms", module.running_ms.detach().clone()),
                    (f"{name}.initialized", module.initialized.detach().clone()),
                    (f"{name}.pa", module.pa.detach().clone()),
                    (f"{name}.pb", module.pb.detach().clone()),
                )
            )
    return tuple(result)


@torch.no_grad()
def typed_pair_coefficient_metric(network: TypedProjectiveDAG) -> Tensor:
    """Return the independent-clone syntactic coefficient Gram.

    Repeated child occurrences contribute independent coefficient axes, as in
    the tree object used by canonical ODT.  This is deliberately not the Gram
    of a polynomial after algebraically identifying all repeated raw inputs.
    """

    _validate_network(network)
    memo: dict[int, Tensor] = {}

    def prefix(node: TypedProjectiveNode) -> Tensor:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        rows = node.core.reshape(node.output_dimension, -1)
        if node.children:
            product = rows.new_ones(1, 1)
            for child in node.children:
                product = torch.kron(product, prefix(child))
            metric = rows @ product @ rows.T
        else:
            metric = rows @ rows.T
        metric = 0.5 * (metric + metric.T)
        memo[id(node)] = metric
        return metric

    output = network.head @ prefix(network.root) @ network.head.T
    return 0.5 * (output + output.T)


@torch.no_grad()
def typed_pair_coefficient_norm(network: TypedProjectiveDAG) -> Tensor:
    return torch.diagonal(typed_pair_coefficient_metric(network)).sum().sqrt()


@torch.no_grad()
def fix_typed_projective_gauge(network: TypedProjectiveDAG) -> TypedProjectiveDAG:
    _validate_network(network)
    norm = typed_pair_coefficient_norm(network)
    if not bool(torch.isfinite(norm)) or float(norm.item()) == 0.0:
        raise ValueError("cannot fix zero/nonfinite typed projective gauge")
    normalized = TypedProjectiveDAG(
        network.root,
        network.head / norm,
        network.token_count,
        network.feature_dimension,
        UNFIXED_GAUGE,
    )
    anchor = network.head.new_zeros(
        1, network.token_count, network.feature_dimension
    )
    denominator = evaluate_scaled_typed_pair(normalized, anchor).mantissa[0, 1]
    if not bool(torch.isfinite(denominator)) or float(denominator.item()) == 0.0:
        raise ValueError("typed gauge anchor denominator is zero/nonfinite")
    sign = 1.0 if float(denominator.item()) > 0.0 else -1.0
    fixed = TypedProjectiveDAG(
        normalized.root,
        normalized.head * sign,
        normalized.token_count,
        normalized.feature_dimension,
        FIXED_GAUGE,
    )
    _require_numeric_fixed_gauge(fixed)
    return fixed


@torch.no_grad()
def rescale_typed_projective_output(
    network: TypedProjectiveDAG,
    scale: float,
) -> TypedProjectiveDAG:
    _validate_network(network)
    scalar = network.head.new_tensor(scale)
    if not bool(torch.isfinite(scalar)) or float(scalar.abs().item()) == 0.0:
        raise ValueError("typed projective scale must be finite and nonzero")
    return TypedProjectiveDAG(
        network.root,
        network.head * scalar,
        network.token_count,
        network.feature_dimension,
        UNFIXED_GAUGE,
    )


def _require_numeric_fixed_gauge(network: TypedProjectiveDAG) -> None:
    if network.gauge != FIXED_GAUGE:
        raise ValueError("typed canonical ODT requires an explicitly fixed projective gauge")
    norm = typed_pair_coefficient_norm(network)
    tolerance = 50_000.0 * torch.finfo(norm.dtype).eps
    if abs(float(norm.item()) - 1.0) > tolerance:
        raise ValueError("typed fixed-gauge metadata has nonunit coefficient norm")
    anchor = network.head.new_zeros(
        1, network.token_count, network.feature_dimension
    )
    denominator = evaluate_scaled_typed_pair(network, anchor).mantissa[0, 1]
    if not bool(torch.isfinite(denominator)) or float(denominator.item()) <= 0.0:
        raise ValueError("typed fixed-gauge metadata has nonpositive D(0)")


def _validate_source(
    block: ChiTransformerBlock,
    action_head: nn.Linear,
    positions: Tensor,
) -> tuple[int, int]:
    if block.training or action_head.training:
        raise ValueError("source block and action head must be in eval mode")
    if not block.residual:
        raise ValueError("typed oracle requires both residual routes")
    if not isinstance(block.attn, BilinearAttention) or block.attn.n_heads != 1:
        raise ValueError("typed prototype requires one BilinearAttention head")
    dimension = block.attn.dim
    if dimension < 1 or block.attn.qk_norm != "rational":
        raise ValueError("typed prototype requires rational Q/K normalization")
    if not isinstance(block.ffn, BilinearFFN) or block.ffn.dim != dimension:
        raise ValueError("typed prototype requires a full BilinearFFN")
    if not isinstance(block.rbn_attn, RationalNorm) or not isinstance(block.rbn_ffn, RationalNorm):
        raise ValueError("both block pre-norms must be RationalNorm")
    if not isinstance(action_head, nn.Linear) or action_head.in_features != dimension:
        raise ValueError("action head input width must match the block")
    if action_head.out_features != 1 or action_head.bias is None:
        raise ValueError("typed prototype requires one biased scalar action head")
    _validate_real_finite(positions, name="positions")
    if positions.ndim != 2 or positions.shape[1] != dimension or positions.shape[0] < 2:
        raise ValueError("positions must have shape (N,D) with N at least two")
    tensors = tuple(block.parameters()) + tuple(action_head.parameters()) + (positions,)
    dtypes = {tensor.dtype for tensor in tensors if tensor.is_floating_point()}
    devices = {tensor.device for tensor in tensors}
    if dtypes != {torch.float64} or len(devices) != 1:
        raise ValueError("typed exact prototype requires float64 on one device")
    norms = tuple(module for module in block.modules() if isinstance(module, RationalNorm))
    if len(norms) != 6:
        raise ValueError("typed prototype requires all six Padé norm module sites")
    for index, module in enumerate(norms):
        if module.variant != "pade" or not bool(module.initialized):
            raise ValueError(f"Padé site {index} must be initialized")
    return int(positions.shape[0]), dimension


@torch.no_grad()
def compile_typed_projective_block(
    block: ChiTransformerBlock,
    action_head: nn.Linear,
    positions: Tensor,
    *,
    token_index: int | None = None,
    mask: Tensor | None = None,
) -> TypedDAGOracle:
    token_count, dimension = _validate_source(block, action_head, positions)
    selected = token_count - 1 if token_index is None else token_index
    if selected != token_count - 1:
        raise ValueError("causal typed prototype selects the last token to retain every source")
    like = action_head.weight
    fixed_mask = (
        causal_mask(token_count, dtype=like.dtype, device=like.device)
        if mask is None
        else mask.to(dtype=like.dtype, device=like.device)
    )
    if fixed_mask.shape != (token_count, token_count):
        raise ValueError("typed mask shape mismatch")
    if not bool(((fixed_mask == 0) | (fixed_mask == 1)).all()):
        raise ValueError("typed mask must be binary")
    if not bool(fixed_mask[selected].all()):
        raise ValueError("selected token must see every source route")

    raw = tuple(_physical_leaf(token, dimension, like) for token in range(token_count))
    positioned = tuple(
        _affine(
            raw[token],
            torch.eye(dimension, dtype=like.dtype, device=like.device),
            positions[token],
            f"position.token{token}",
        )
        for token in range(token_count)
    )
    normalized = tuple(
        _pade_norm(value, block.rbn_attn, f"attention_pre_norm.token{token}")
        for token, value in enumerate(positioned)
    )

    def project(linear: nn.Linear, token: int, label: str) -> TypedProjectiveNode:
        if linear.bias is None:
            raise ValueError(f"projection {label} unexpectedly lacks its bias")
        return _affine(normalized[token], linear.weight, linear.bias, label)

    queries: dict[str, TypedProjectiveNode] = {}
    for name, linear, norm in (
        ("q1", block.attn.wq1, block.attn.rn_q1),
        ("q2", block.attn.wq2, block.attn.rn_q2),
    ):
        queries[name] = _pade_norm(
            project(linear, selected, f"attention.{name}.affine.q{selected}"),
            norm,
            f"attention.{name}.norm.q{selected}",
        )
    keys: dict[tuple[str, int], TypedProjectiveNode] = {}
    for name, linear, norm in (
        ("k1", block.attn.wk1, block.attn.rn_k1),
        ("k2", block.attn.wk2, block.attn.rn_k2),
    ):
        for source in range(token_count):
            keys[(name, source)] = _pade_norm(
                project(linear, source, f"attention.{name}.affine.k{source}"),
                norm,
                f"attention.{name}.norm.k{source}",
            )
    values = tuple(
        project(block.attn.wv, source, f"attention.v.affine.source{source}")
        for source in range(token_count)
    )

    routes: list[TypedProjectiveNode] = []
    visible = fixed_mask[selected].sum()
    row_scale = torch.rsqrt(visible) if block.attn.row_scale == "invsqrt" else visible.reciprocal()
    fixed_scale = row_scale / float(block.attn._score_denom)
    for source in range(token_count):
        score1 = _dot_vectors(
            queries["q1"], keys[("k1", source)], f"attention.route{source}.score1"
        )
        score2 = _dot_vectors(
            queries["q2"], keys[("k2", source)], f"attention.route{source}.score2"
        )
        score = _multiply_scalars(score1, score2, f"attention.route{source}.score_product")
        route = _scale_vector_by_scalar(
            score, values[source], f"attention.route{source}.score_times_value"
        )
        routes.append(
            _scale_vector(route, fixed_scale, f"attention.route{source}.fixed_scale")
        )
    attention_value = routes[0]
    for source, route in enumerate(routes[1:], start=1):
        attention_value = _add_vectors(
            attention_value, route, f"attention.route{source}.cross_multiplied_sum"
        )
    attention = _affine(
        attention_value,
        block.attn.wo.weight,
        block.attn.wo.bias,
        "attention.output_affine",
    )
    attention = _scale_vector(attention, block.attn_gain, "attention.residual_gain")
    after_attention = _add_vectors(
        positioned[selected], attention, "attention.residual_add"
    )

    ffn_input = _pade_norm(after_attention, block.rbn_ffn, "ffn_pre_norm")
    left = _affine(ffn_input, block.ffn.left.weight, block.ffn.left.bias, "ffn.left")
    right = _affine(ffn_input, block.ffn.right.weight, block.ffn.right.bias, "ffn.right")
    product = _hadamard_vectors(left, right, "ffn.hadamard")
    down_bias = (
        block.ffn.down.weight.new_zeros(block.ffn.out_dim)
        if block.ffn.down.bias is None
        else block.ffn.down.bias
    )
    ffn = _affine(product, block.ffn.down.weight, down_bias, "ffn.down")
    ffn = _scale_vector(ffn, block.ffn_gain, "ffn.residual_gain")
    after_ffn = _add_vectors(after_attention, ffn, "ffn.residual_add")
    action = _affine(after_ffn, action_head.weight, action_head.bias, "action_head")

    raw_network = TypedProjectiveDAG(
        action,
        torch.eye(2, dtype=like.dtype, device=like.device),
        token_count,
        dimension,
        UNFIXED_GAUGE,
    )
    anchor = like.new_zeros(1, token_count, dimension)
    unfixed_anchor = evaluate_scaled_typed_pair(raw_network, anchor)
    denominator = unfixed_anchor.mantissa[0, 1].abs()
    if float(denominator.item()) == 0.0:
        raise ValueError("typed raw DAG has structurally zero D(0)")
    log10_denominator = float(
        unfixed_anchor.binary_exponent[0].item() * math.log10(2.0)
        + math.log10(float(denominator.item()))
    )
    predicted_head = math.log10(float(raw_network.head.abs().max().item())) - log10_denominator
    fixed = fix_typed_projective_gauge(raw_network)
    return TypedDAGOracle(
        fixed,
        block,
        action_head,
        positions.detach().clone(),
        selected,
        fixed_mask.detach().clone(),
        _norm_snapshot(block),
        log10_denominator,
        predicted_head,
        float(typed_pair_coefficient_norm(fixed).item()),
    )


@torch.no_grad()
def source_typed_action(oracle: TypedDAGOracle, raw_input: Tensor) -> Tensor:
    tokens = raw_input + oracle.positions
    output = oracle.block(tokens, mask=oracle.mask, method="explicit")
    return oracle.action_head(output[:, oracle.token_index]).squeeze(-1)


@torch.no_grad()
def assert_typed_norm_buffers_unchanged(oracle: TypedDAGOracle) -> None:
    current = dict(_norm_snapshot(oracle.block))
    for name, expected in oracle.norm_buffer_snapshot:
        if name not in current or not torch.equal(current[name], expected):
            raise RuntimeError(f"typed compiler changed normalization buffer {name}")


def _relative_error(actual: Tensor, expected: Tensor) -> float:
    scale = torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)
    return float((torch.linalg.vector_norm(actual - expected) / scale).item())


def _row_isometry_error(rows: Tensor) -> float:
    identity = torch.eye(rows.shape[0], dtype=rows.dtype, device=rows.device)
    return float(torch.linalg.matrix_norm(rows @ rows.T - identity).item())


def _supported_reduced_rq_rows(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Direct reduced RQ, including rectangular and singular unfoldings.

    Canonicalization never decides numerical support.  Any SVD/rank diagnostic
    belongs to a separate compression audit, after the exact full-rank gauge.
    """

    return reduced_rq_rows(matrix)


def _apply_input_factor(core: Tensor, role: int, factor: Tensor) -> Tensor:
    axis = role + 1
    return (core.movedim(axis, -1) @ factor).movedim(-1, axis)


def _apply_output_factor(core: Tensor, factor: Tensor) -> Tensor:
    return torch.tensordot(factor, core, dims=([1], [0]))


@torch.no_grad()
def canonicalize_typed_projective_dag(
    network: TypedProjectiveDAG,
    *,
    omit_factor_at: tuple[str, int] | None = None,
) -> CanonicalTypedDAG:
    """Shared-node Dooms Algorithm 1, optionally with one adversarial omission."""

    _validate_network(network)
    _require_numeric_fixed_gauge(network)
    memo: dict[int, tuple[TypedProjectiveNode, Tensor]] = {}
    active: set[int] = set()
    isometry_errors: list[float] = []
    factorization_errors: list[float] = []
    omitted = False

    def visit(node: TypedProjectiveNode) -> tuple[TypedProjectiveNode, Tensor]:
        nonlocal omitted
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        if id(node) in active:
            raise ValueError("cycle during typed bottom-up RQ")
        active.add(id(node))
        child_results = tuple(visit(child) for child in node.children)
        children = tuple(result[0] for result in child_results)
        transformed = node.core.detach().clone()
        for role, (_, factor) in enumerate(child_results):
            should_omit = (
                omit_factor_at is not None
                and not omitted
                and node.label == omit_factor_at[0]
                and role == omit_factor_at[1]
            )
            if should_omit:
                if factor.shape[0] != factor.shape[1]:
                    raise ValueError("negative-control omission requires a square child factor")
                omitted = True
            else:
                transformed = _apply_input_factor(transformed, role, factor)
        rows = transformed.reshape(transformed.shape[0], -1)
        factor, isometric_rows = _supported_reduced_rq_rows(rows)
        core = isometric_rows.reshape((isometric_rows.shape[0],) + transformed.shape[1:])
        canonical = TypedProjectiveNode(
            node.label,
            core,
            children,
            node.physical_token,
            torch.eye(core.shape[0], dtype=core.dtype, device=core.device),
        )
        isometry_errors.append(_row_isometry_error(isometric_rows))
        factorization_errors.append(_relative_error(factor @ isometric_rows, rows))
        result = (canonical, factor)
        memo[id(node)] = result
        active.remove(id(node))
        return result

    root, factor = visit(network.root)
    if omit_factor_at is not None and not omitted:
        raise ValueError("requested typed one-leg omission was not found")
    result_network = TypedProjectiveDAG(
        root,
        network.head @ factor,
        network.token_count,
        network.feature_dimension,
        UNFIXED_GAUGE if omit_factor_at is not None else network.gauge,
    )
    _validate_network(result_network)
    if omit_factor_at is None:
        _require_numeric_fixed_gauge(result_network)
    return CanonicalTypedDAG(
        result_network,
        tuple(isometry_errors),
        tuple(factorization_errors),
        len(memo),
        _edge_occurrence_count(root),
        tuple("direct_reduced_rq" for _ in isometry_errors),
    )


def _require_row_isometric(network: TypedProjectiveDAG) -> None:
    _validate_network(network)
    _require_numeric_fixed_gauge(network)
    for node in _walk_unique(network.root):
        rows = node.core.reshape(node.output_dimension, -1)
        if _row_isometry_error(rows) > 3e-8 * max(1.0, math.sqrt(rows.shape[0])):
            raise ValueError(f"typed node {node.label!r} is not row-isometric")


def _incoming_occurrences(root: TypedProjectiveNode) -> dict[int, int]:
    result = {id(node): 0 for node in _walk_unique(root)}
    for node in _walk_unique(root):
        for child in node.children:
            result[id(child)] += 1
    return result


def _validated_output_metric(
    network: TypedProjectiveDAG,
    output_metric: Tensor | None,
) -> Tensor:
    if output_metric is None:
        return torch.eye(2, dtype=network.head.dtype, device=network.head.device)
    _validate_real_finite(output_metric, name="typed output metric")
    if output_metric.shape != (2, 2):
        raise ValueError("typed output metric must have shape (2,2)")
    if output_metric.dtype != network.head.dtype or output_metric.device != network.head.device:
        raise ValueError("typed output metric and network must share dtype and device")
    symmetric = 0.5 * (output_metric + output_metric.T)
    scale = max(float(symmetric.abs().max().item()), torch.finfo(symmetric.dtype).tiny)
    tolerance = 100.0 * torch.finfo(symmetric.dtype).eps * scale
    if float((output_metric - output_metric.T).abs().max().item()) > tolerance:
        raise ValueError("typed output metric must be symmetric")
    if float(torch.linalg.eigvalsh(symmetric)[0].item()) < -tolerance:
        raise ValueError("typed output metric must be positive semidefinite")
    return symmetric


@torch.no_grad()
def reverse_typed_projective_environments(
    network: TypedProjectiveDAG,
    output_metric: Tensor | None = None,
) -> tuple[TypedEnvironment, ...]:
    """Reverse-topological shared-node form of Dooms Algorithm 2."""

    _require_row_isometric(network)
    output_metric = _validated_output_metric(network, output_metric)
    root_raw = network.head.T @ output_metric @ network.head
    root_message = scale_symmetric_message(root_raw)
    pending: dict[int, list[tuple[Tensor, int]]] = defaultdict(list)
    pending[id(network.root)].append(root_message)
    incoming = _incoming_occurrences(network.root)
    records: list[TypedEnvironment] = []
    for node in _topological_nodes(network.root):
        messages = pending.get(id(node), [])
        if not messages:
            raise ValueError(f"typed node {node.label!r} received no environment")
        gram, exponent = (
            messages[0] if len(messages) == 1 else add_scaled_symmetric_messages(messages)
        )
        records.append(
            TypedEnvironment(id(node), node.label, gram, exponent, incoming[id(node)])
        )
        identities = tuple(
            torch.eye(child.output_dimension, dtype=node.core.dtype, device=node.core.device)
            for child in node.children
        )
        for role, child in enumerate(node.children):
            raw = role_environment(node.core, gram, identities, role)
            pending[id(child)].append(scale_symmetric_message(raw, exponent))
    if len(records) != len(_walk_unique(network.root)):
        raise ValueError("typed reverse environment traversal missed a node")
    return tuple(records)


@torch.no_grad()
def explicit_clone_typed_environments(
    network: TypedProjectiveDAG,
    output_metric: Tensor | None = None,
) -> tuple[CloneEnvironment, ...]:
    """Path-enumerating clone oracle used only for the tiny D=1 regression."""

    _require_row_isometric(network)
    output_metric = _validated_output_metric(network, output_metric)
    root = scale_symmetric_message(network.head.T @ output_metric @ network.head)
    records: list[CloneEnvironment] = []

    def descend(
        node: TypedProjectiveNode,
        gram: Tensor,
        exponent: int,
        path: tuple[int, ...],
    ) -> None:
        records.append(CloneEnvironment(path, id(node), node.label, gram, exponent))
        identities = tuple(
            torch.eye(child.output_dimension, dtype=node.core.dtype, device=node.core.device)
            for child in node.children
        )
        for role, child in enumerate(node.children):
            raw = role_environment(node.core, gram, identities, role)
            child_gram, child_exponent = scale_symmetric_message(raw, exponent)
            descend(child, child_gram, child_exponent, path + (role,))

    descend(network.root, root[0], root[1], ())
    return tuple(records)


def aggregate_clone_environments(
    records: Iterable[CloneEnvironment],
) -> dict[int, tuple[Tensor, int]]:
    grouped: dict[int, list[tuple[Tensor, int]]] = defaultdict(list)
    for record in records:
        grouped[record.node_identity].append((record.gram, record.binary_exponent))
    return {
        identity: (
            values[0] if len(values) == 1 else add_scaled_symmetric_messages(values)
        )
        for identity, values in grouped.items()
    }


def scaled_matrix_relative_error(
    actual: tuple[Tensor, int],
    expected: tuple[Tensor, int],
) -> float:
    if actual[0].shape != expected[0].shape:
        raise ValueError("scaled matrix shapes differ")
    center = max(actual[1], expected[1])
    actual_shift = torch.tensor(actual[1] - center, dtype=torch.int64, device=actual[0].device)
    expected_shift = torch.tensor(expected[1] - center, dtype=torch.int64, device=expected[0].device)
    left = torch.ldexp(actual[0], actual_shift)
    right = torch.ldexp(expected[0], expected_shift)
    return _relative_error(left, right)


@torch.no_grad()
def maximum_aggregate_environment_offdiagonal_ratio(
    network: TypedProjectiveDAG,
    output_metric: Tensor | None = None,
) -> float:
    """Recontract and measure aggregate-Gram diagonality in the stored bases."""

    records = reverse_typed_projective_environments(network, output_metric)
    maximum = 0.0
    for record in records:
        diagonal = torch.diag(torch.diagonal(record.gram))
        scale = torch.linalg.matrix_norm(record.gram).clamp_min(
            torch.finfo(record.gram.dtype).tiny
        )
        ratio = float((torch.linalg.matrix_norm(record.gram - diagonal) / scale).item())
        maximum = max(maximum, ratio)
    return maximum


@torch.no_grad()
def diagonalize_typed_projective_dag_full_rank(
    network: TypedProjectiveDAG,
    output_metric: Tensor | None = None,
) -> DiagonalTypedDAG:
    environments = reverse_typed_projective_environments(network, output_metric)
    bases: dict[int, Tensor] = {}
    eigenvalues: list[Tensor] = []
    exponents: list[int] = []
    errors: list[float] = []
    labels: list[str] = []
    for record in environments:
        values, vectors = torch.linalg.eigh(0.5 * (record.gram + record.gram.T))
        order = torch.argsort(values, descending=True)
        values = values[order].clamp_min(0)
        vectors = vectors[:, order]
        diagonal = vectors.T @ record.gram @ vectors
        off_diagonal = diagonal - torch.diag(torch.diagonal(diagonal))
        scale = torch.linalg.matrix_norm(record.gram).clamp_min(
            torch.finfo(record.gram.dtype).tiny
        )
        errors.append(float((torch.linalg.matrix_norm(off_diagonal) / scale).item()))
        bases[record.node_identity] = vectors
        eigenvalues.append(values)
        exponents.append(record.binary_exponent)
        labels.append(record.label)

    memo: dict[int, TypedProjectiveNode] = {}

    def rotate(node: TypedProjectiveNode) -> TypedProjectiveNode:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        children = tuple(rotate(child) for child in node.children)
        core = node.core.detach().clone()
        for role, child in enumerate(node.children):
            core = _apply_input_factor(core, role, bases[id(child)])
        core = _apply_output_factor(core, bases[id(node)].T)
        result = TypedProjectiveNode(
            node.label,
            core,
            children,
            node.physical_token,
            torch.eye(core.shape[0], dtype=core.dtype, device=core.device),
        )
        memo[id(node)] = result
        return result

    root = rotate(network.root)
    rotated = TypedProjectiveDAG(
        root,
        network.head @ bases[id(network.root)],
        network.token_count,
        network.feature_dimension,
        network.gauge,
    )
    _require_row_isometric(rotated)
    return DiagonalTypedDAG(
        rotated,
        tuple(eigenvalues),
        tuple(exponents),
        tuple(errors),
        tuple(labels),
    )


def find_typed_node(network: TypedProjectiveDAG, label: str) -> TypedProjectiveNode:
    matches = tuple(node for node in _walk_unique(network.root) if node.label == label)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one typed node labeled {label!r}")
    return matches[0]


@torch.no_grad()
def gauge_shared_typed_node(
    network: TypedProjectiveDAG,
    label: str,
    transform: Tensor,
    *,
    omit_parent_occurrence: int | None = None,
) -> TypedProjectiveDAG:
    """Gauge one shared node and every parent role, or omit one role adversarially."""

    _validate_network(network)
    target = find_typed_node(network, label)
    if transform.shape != (target.output_dimension, target.output_dimension):
        raise ValueError("typed hidden gauge shape mismatch")
    _validate_real_finite(transform, name="typed hidden gauge")
    inverse = torch.linalg.inv(transform)
    memo: dict[int, TypedProjectiveNode] = {}
    occurrence = 0
    omitted = False

    def rewrite(node: TypedProjectiveNode) -> TypedProjectiveNode:
        nonlocal occurrence, omitted
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        children = tuple(rewrite(child) for child in node.children)
        core = node.core.detach().clone()
        if id(node) == id(target):
            core = _apply_output_factor(core, transform)
        for role, child in enumerate(node.children):
            if id(child) != id(target):
                continue
            skip = omit_parent_occurrence is not None and occurrence == omit_parent_occurrence
            occurrence += 1
            if skip:
                omitted = True
            else:
                core = _apply_input_factor(core, role, inverse)
        result = TypedProjectiveNode(node.label, core, children, node.physical_token)
        memo[id(node)] = result
        return result

    root = rewrite(network.root)
    head = network.head.detach().clone()
    if id(target) == id(network.root):
        if omit_parent_occurrence is None:
            head = head @ inverse
        elif omit_parent_occurrence == 0:
            omitted = True
    if occurrence == 0 and id(target) != id(network.root):
        raise ValueError("typed hidden gauge target has no parent occurrences")
    if omit_parent_occurrence is not None and not omitted:
        raise ValueError("requested parent occurrence omission was not found")
    result = TypedProjectiveDAG(
        root,
        head,
        network.token_count,
        network.feature_dimension,
        UNFIXED_GAUGE if omit_parent_occurrence is not None else network.gauge,
    )
    _validate_network(result)
    if omit_parent_occurrence is None:
        _require_numeric_fixed_gauge(result)
    return result


def shared_parent_occurrence_count(network: TypedProjectiveDAG, label: str) -> int:
    target = find_typed_node(network, label)
    return sum(
        1
        for node in _walk_unique(network.root)
        for child in node.children
        if id(child) == id(target)
    )


def typed_dag_shape_statistics(network: TypedProjectiveDAG) -> dict[str, int]:
    """Return local-bond graph sizes without unfolding shared occurrences."""

    _validate_network(network)
    nodes = _walk_unique(network.root)
    incoming = _incoming_occurrences(network.root)
    vector_adds = tuple(
        node
        for node in nodes
        if node.core.ndim == 3
        and node.core.shape[0] == node.core.shape[1] == node.core.shape[2]
        and ("residual_add" in node.label or "cross_multiplied_sum" in node.label)
    )
    return {
        "unique_node_count": len(nodes),
        "edge_occurrence_count": sum(len(node.children) for node in nodes),
        "shared_node_count": sum(count > 1 for count in incoming.values()),
        "maximum_local_bond_dimension": max(node.output_dimension for node in nodes),
        "maximum_materialized_core_entries": max(node.core.numel() for node in nodes),
        "maximum_materialized_vector_add_entries": max(
            (node.core.numel() for node in vector_adds), default=0
        ),
        "dense_cubic_vector_add_core_count": len(vector_adds),
        "fused_token_feature_dimension": (
            network.token_count * network.feature_dimension + 1
        ),
    }


def brute_first_role_scaled_environment(
    network: TypedProjectiveDAG,
) -> tuple[tuple[Tensor, int], tuple[Tensor, int]]:
    """Return recursive and independent-loop first-role messages."""

    records = reverse_typed_projective_environments(network)
    root_record = next(record for record in records if record.node_identity == id(network.root))
    if not network.root.children:
        raise ValueError("typed root has no first role")
    child = network.root.children[0]
    recursive = next(record for record in records if record.node_identity == id(child))
    brute_raw = brute_identity_role_environment(network.root.core, root_record.gram, 0)
    brute = scale_symmetric_message(brute_raw, root_record.binary_exponent)
    return (recursive.gram, recursive.binary_exponent), brute


__all__ = [
    "CLAIM_BOUNDARY",
    "FIXED_GAUGE",
    "OBJECT_KIND",
    "UNFIXED_GAUGE",
    "CanonicalTypedDAG",
    "CloneEnvironment",
    "DiagonalTypedDAG",
    "TypedDAGOracle",
    "TypedEnvironment",
    "TypedProjectiveDAG",
    "TypedProjectiveNode",
    "aggregate_clone_environments",
    "assert_typed_norm_buffers_unchanged",
    "brute_first_role_scaled_environment",
    "canonicalize_typed_projective_dag",
    "compile_typed_projective_block",
    "diagonalize_typed_projective_dag_full_rank",
    "evaluate_scaled_typed_pair",
    "evaluate_typed_pair_unscaled_negative_control",
    "evaluate_typed_quotient",
    "explicit_clone_typed_environments",
    "find_typed_node",
    "fix_typed_projective_gauge",
    "gauge_shared_typed_node",
    "maximum_aggregate_environment_offdiagonal_ratio",
    "omit_quotient_cross_terms_negative_control",
    "quotient_pair_output_metric",
    "rescale_typed_projective_output",
    "reverse_typed_projective_environments",
    "scaled_matrix_relative_error",
    "scaled_projective_relative_error",
    "scaled_projective_statistics",
    "shared_parent_occurrence_count",
    "source_typed_action",
    "typed_pair_coefficient_metric",
    "typed_pair_coefficient_norm",
    "typed_dag_shape_statistics",
]
