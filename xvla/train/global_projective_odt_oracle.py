"""Exact global projective ODT oracle for one tiny residual attention block.

The deployed Padé ``RationalNorm`` is a rational map.  This module therefore
does not evaluate a quotient at each normalization site.  It clone-unfolds the
complete selected-token policy into one heterogeneous multilinear arithmetic
tree whose root is a projective pair ``[N, D]``.  Every internal operation is a
fixed multilinear core.  The only input-dependent division is the final
``N / D`` used to compare with the source model.

The oracle deliberately targets ``N=2, D=1, H=1``.  That is large enough to
retain a causal cross-token attention route, both residual additions, all
linear biases, the bilinear FFN, and all six kinds of Padé normalization sites,
while keeping a dense full-rank Dooms sweep auditable.

Ordinary coefficient-space ODT is applied only after fixing the constant
projective gauge by requiring unit coefficient Frobenius norm and choosing the
sign so ``D(x=0)>0``.  This removes a global nonzero scalar rescaling without
undoing the stabilizing internal gauges.  It does not quotient arbitrary
nonconstant common polynomial factors, so the resulting Euclidean eigensystem
is canonical for the compiled pair representation, not intrinsically canonical
for the rational function.
Full-rank RQ and EVD basis absorption nevertheless preserve both polynomials
and hence their quotient exactly wherever the denominator is nonzero.
"""

from __future__ import annotations

import itertools
import math
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


Tensor = torch.Tensor
OBJECT_KIND = "global_clone_unfolded_projective_polynomial_tree"
FIXED_GAUGE = "constant_scalar_gauge_unit_coefficient_norm_D_at_zero_positive"
UNFIXED_GAUGE = "constant_scalar_projective_gauge_unfixed"
CLAIM_BOUNDARY = (
    "Exact full-rank canonical ODT of a fixed-gauge numerator/denominator "
    "polynomial tree for one tiny deployed Padé residual-attention block. "
    "This is not a quotient-intrinsic rational ODT and makes no compression claim."
)


@dataclass(frozen=True)
class MultilinearTreeNode:
    """One typed multilinear core in a clone-unfolded coefficient tree.

    A leaf has no children and ``core`` is its embedding from the common raw
    homogeneous input.  An internal node has one input axis per child.
    Reusing the same Python child object represents syntactic sharing only.
    Environment traversal still treats every occurrence as a separate cloned
    coefficient-space leg.
    """

    label: str
    core: Tensor
    children: tuple["MultilinearTreeNode", ...] = ()
    prefix_metric: Tensor | None = field(default=None, repr=False, compare=False)

    @property
    def output_dimension(self) -> int:
        return int(self.core.shape[0])

    @property
    def arity(self) -> int:
        return len(self.children)


@dataclass(frozen=True)
class ProjectivePolynomialTree:
    """A two-output polynomial tree evaluated as numerator over denominator."""

    root: MultilinearTreeNode
    head: Tensor
    raw_input_dimension: int
    gauge: str = UNFIXED_GAUGE
    object_kind: str = OBJECT_KIND
    claim_boundary: str = CLAIM_BOUNDARY


@dataclass(frozen=True)
class CanonicalProjectiveODT:
    """Bottom-up row-isometric tree and exact root head factors."""

    network: ProjectivePolynomialTree
    isometry_errors: tuple[float, ...]
    factorization_errors: tuple[float, ...]
    unique_node_count: int
    occurrence_count: int
    factorization_methods: tuple[str, ...]


@dataclass(frozen=True)
class EnvironmentRecord:
    path: tuple[int, ...]
    label: str
    gram: Tensor
    binary_exponent: int


@dataclass(frozen=True)
class FullRankDiagonalProjectiveODT:
    """Algorithm-3 full-rank bond rotations for the fixed-gauge pair tree."""

    network: ProjectivePolynomialTree
    eigenvalues: tuple[Tensor, ...]
    eigenvalue_binary_exponents: tuple[int, ...]
    diagonalization_errors: tuple[float, ...]
    paths: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class GlobalProjectiveBlockOracle:
    """Compiled tree together with the unchanged modules it represents."""

    network: ProjectivePolynomialTree
    block: ChiTransformerBlock
    action_head: nn.Linear
    token_index: int
    mask: Tensor
    norm_buffer_snapshot: tuple[tuple[str, Tensor], ...]
    unfixed_log10_absolute_denominator_at_zero: float
    old_anchor_refix_predicted_head_log10_maximum: float
    fixed_head_maximum_absolute: float
    fixed_coefficient_norm: float


@dataclass(frozen=True)
class ScaledProjectiveEvaluation:
    """Bounded pair coordinates plus an exact common binary exponent.

    The represented pair is ``ldexp(mantissa, binary_exponent)`` per batch item.
    ``ldexp`` changes only the floating-point exponent, so this bookkeeping adds
    no rounding beyond the core contraction itself.  It is an evaluator-side
    numerical chart choice, not an operation in the compiled multilinear tree,
    and it cannot change a quotient or any RQ/EVD coefficient tensor.
    """

    mantissa: Tensor
    binary_exponent: Tensor


def _validate_real_finite(tensor: Tensor, *, name: str) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(f"{name} must have a real floating dtype")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


def _walk_unique(root: MultilinearTreeNode) -> tuple[MultilinearTreeNode, ...]:
    seen: set[int] = set()
    result: list[MultilinearTreeNode] = []

    def visit(node: MultilinearTreeNode) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        result.append(node)
        for child in node.children:
            visit(child)

    visit(root)
    return tuple(result)


def _occurrence_count(root: MultilinearTreeNode) -> int:
    return 1 + sum(_occurrence_count(child) for child in root.children)


def _validate_network(network: ProjectivePolynomialTree) -> None:
    if network.object_kind != OBJECT_KIND:
        raise ValueError(f"unsupported object kind {network.object_kind!r}")
    if network.gauge not in {FIXED_GAUGE, UNFIXED_GAUGE}:
        raise ValueError("unknown projective scalar gauge state")
    if isinstance(network.raw_input_dimension, bool) or network.raw_input_dimension < 1:
        raise ValueError("raw_input_dimension must be positive")
    nodes = _walk_unique(network.root)
    reference_dtype = network.head.dtype
    reference_device = network.head.device
    _validate_real_finite(network.head, name="head")
    if network.head.ndim != 2 or network.head.shape != (2, network.root.output_dimension):
        raise ValueError("head must map the root bond to exactly [numerator, denominator]")
    for node in nodes:
        if not node.label:
            raise ValueError("every tree node needs a label")
        _validate_real_finite(node.core, name=f"core {node.label!r}")
        if node.core.dtype != reference_dtype or node.core.device != reference_device:
            raise ValueError("all cores and the head must share dtype and device")
        if node.prefix_metric is not None:
            _validate_real_finite(node.prefix_metric, name=f"prefix metric {node.label!r}")
            if node.prefix_metric.shape != (node.output_dimension, node.output_dimension):
                raise ValueError(f"prefix metric {node.label!r} has the wrong shape")
            if node.prefix_metric.dtype != reference_dtype or node.prefix_metric.device != reference_device:
                raise ValueError("prefix metrics must share core dtype and device")
        if node.children:
            if node.core.ndim != 1 + len(node.children):
                raise ValueError(f"core {node.label!r} arity does not match its children")
            expected = tuple(child.output_dimension for child in node.children)
            if tuple(node.core.shape[1:]) != expected:
                raise ValueError(f"core {node.label!r} input dimensions do not match its children")
        elif node.core.shape != (node.output_dimension, network.raw_input_dimension + 1):
            raise ValueError(f"leaf {node.label!r} is not a raw homogeneous embedding")


def _coefficient_prefix_metric(core: Tensor, children: tuple[MultilinearTreeNode, ...]) -> Tensor:
    rows = core.reshape(core.shape[0], -1)
    if not children:
        metric = rows @ rows.T
    else:
        product = core.new_ones(1, 1)
        for child in children:
            if child.prefix_metric is None:
                raise ValueError("normalized tree construction requires child prefix metrics")
            product = torch.kron(product, child.prefix_metric)
        metric = rows @ product @ rows.T
    return 0.5 * (metric + metric.T)


def _normalized_node(
    label: str,
    core: Tensor,
    children: tuple[MultilinearTreeNode, ...] = (),
) -> MultilinearTreeNode:
    """Apply an exact input-independent scalar projective gauge.

    Scaling every output coordinate of a homogeneous/projective node by the
    same nonzero constant does not alter any represented quotient.  Choosing
    the constant from the independent-clone coefficient prefix Gram keeps the
    bottom-up R factors bounded and prevents the cross-multiplied pair from
    overflowing before canonicalization.
    """

    metric = _coefficient_prefix_metric(core, children)
    norm = torch.diagonal(metric).sum().sqrt()
    if not bool(torch.isfinite(norm)) or float(norm.item()) == 0.0:
        raise ValueError(f"node {label!r} has a nonfinite or zero coefficient norm")
    scaled_core = core / norm
    scaled_metric = metric / norm.square()
    return MultilinearTreeNode(label, scaled_core, children, scaled_metric)


def _einsum_evaluate(core: Tensor, values: tuple[Tensor, ...]) -> Tensor:
    labels = "".join(label for label in ascii_lowercase if label not in {"b", "o"})
    if len(values) > len(labels):
        raise ValueError("tree node arity exceeds the einsum label budget")
    roles = labels[: len(values)]
    equation = f"o{roles}," + ",".join(f"b{role}" for role in roles) + "->bo"
    return torch.einsum(equation, core, *values)


def _scaled_coordinates(value: Tensor, inherited_exponent: Tensor) -> ScaledProjectiveEvaluation:
    local_scale = value.abs().amax(dim=1)
    if bool((local_scale == 0).any()) or not bool(torch.isfinite(local_scale).all()):
        raise ValueError("projective contraction produced a zero or nonfinite coordinate vector")
    _, local_exponent = torch.frexp(local_scale)
    return ScaledProjectiveEvaluation(
        torch.ldexp(value, -local_exponent[:, None]),
        inherited_exponent + local_exponent,
    )


@torch.no_grad()
def evaluate_scaled_projective_pair(
    network: ProjectivePolynomialTree,
    raw_input: Tensor,
) -> ScaledProjectiveEvaluation:
    """Contract in a bounded projective chart with a common log-scale ledger."""

    _validate_network(network)
    _validate_real_finite(raw_input, name="raw_input")
    if raw_input.ndim != 2 or raw_input.shape[1] != network.raw_input_dimension:
        raise ValueError("raw_input must have shape (batch, raw_input_dimension)")
    if raw_input.dtype != network.head.dtype or raw_input.device != network.head.device:
        raise ValueError("raw_input and network must share dtype and device")
    homogeneous = torch.cat((raw_input.new_ones(raw_input.shape[0], 1), raw_input), dim=1)
    memo: dict[int, ScaledProjectiveEvaluation] = {}

    def evaluate(node: MultilinearTreeNode) -> ScaledProjectiveEvaluation:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        if node.children:
            children = tuple(evaluate(child) for child in node.children)
            value = _einsum_evaluate(node.core, tuple(child.mantissa for child in children))
            inherited = torch.stack(tuple(child.binary_exponent for child in children)).sum(0)
        else:
            value = homogeneous @ node.core.T
            inherited = torch.zeros(value.shape[0], dtype=torch.int64, device=value.device)
        result = _scaled_coordinates(value, inherited)
        memo[id(node)] = result
        return result

    root = evaluate(network.root)
    output = root.mantissa @ network.head.T
    return _scaled_coordinates(output, root.binary_exponent)


@torch.no_grad()
def evaluate_projective_pair(network: ProjectivePolynomialTree, raw_input: Tensor) -> Tensor:
    """Materialize [N,D] only when its common scale fits the floating dtype."""

    scaled = evaluate_scaled_projective_pair(network, raw_input)
    finfo = torch.finfo(scaled.mantissa.dtype)
    maximum_exponent = math.frexp(float(finfo.max))[1]
    minimum_exponent = math.frexp(float(finfo.tiny))[1]
    if bool(
        ((scaled.binary_exponent > maximum_exponent) | (scaled.binary_exponent < minimum_exponent)).any()
    ):
        raise OverflowError(
            "projective pair common scale is outside floating range; use "
            "evaluate_scaled_projective_pair"
        )
    return torch.ldexp(scaled.mantissa, scaled.binary_exponent[:, None])


@torch.no_grad()
def evaluate_projective_pair_unscaled_negative_control(
    network: ProjectivePolynomialTree,
    raw_input: Tensor,
) -> Tensor:
    """Naïve contraction retained only to reproduce the r3 underflow failure."""

    _validate_network(network)
    if raw_input.ndim != 2 or raw_input.shape[1] != network.raw_input_dimension:
        raise ValueError("raw_input shape mismatch")
    homogeneous = torch.cat((raw_input.new_ones(raw_input.shape[0], 1), raw_input), dim=1)
    memo: dict[int, Tensor] = {}

    def evaluate(node: MultilinearTreeNode) -> Tensor:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        value = (
            _einsum_evaluate(node.core, tuple(evaluate(child) for child in node.children))
            if node.children
            else homogeneous @ node.core.T
        )
        memo[id(node)] = value
        return value

    return evaluate(network.root) @ network.head.T


@torch.no_grad()
def evaluate_quotient(
    network: ProjectivePolynomialTree,
    raw_input: Tensor,
    *,
    minimum_relative_denominator: float = 1e-14,
) -> Tensor:
    """Take the single allowed input-dependent division at the output boundary."""

    scaled = evaluate_scaled_projective_pair(network, raw_input)
    pair = scaled.mantissa
    denominator = pair[:, 1]
    if minimum_relative_denominator <= 0:
        raise ValueError("minimum_relative_denominator must be positive")
    scale = pair.abs().amax(dim=1).clamp_min(torch.finfo(pair.dtype).tiny)
    relative = denominator.abs() / scale
    threshold = max(minimum_relative_denominator, 100.0 * torch.finfo(pair.dtype).eps)
    if float(relative.min().item()) <= threshold:
        raise ValueError("observed final denominator is too small relative to its projective pair")
    return pair[:, 0] / denominator


def scaled_projective_relative_error(
    actual: ScaledProjectiveEvaluation,
    expected: ScaledProjectiveEvaluation,
) -> float:
    """Compare material pairs after a safe per-sample common exponent shift."""

    if actual.mantissa.shape != expected.mantissa.shape:
        raise ValueError("scaled pair shapes differ")
    if actual.binary_exponent.shape != expected.binary_exponent.shape:
        raise ValueError("scaled pair scale shapes differ")
    center = torch.maximum(actual.binary_exponent, expected.binary_exponent)
    actual_aligned = torch.ldexp(
        actual.mantissa, (actual.binary_exponent - center)[:, None]
    )
    expected_aligned = torch.ldexp(
        expected.mantissa, (expected.binary_exponent - center)[:, None]
    )
    numerator = torch.linalg.vector_norm(actual_aligned - expected_aligned, dim=1)
    denominator = torch.linalg.vector_norm(expected_aligned, dim=1).clamp_min(
        torch.finfo(expected.mantissa.dtype).tiny
    )
    return float((numerator / denominator).max().item())


def scaled_projective_statistics(
    evaluation: ScaledProjectiveEvaluation,
) -> dict[str, float | bool | int]:
    """Report log magnitudes and denominator conditioning without materializing scale."""

    absolute = evaluation.mantissa.abs()
    nonzero = absolute > 0
    if not bool(nonzero.any()):
        raise ValueError("scaled projective evaluation contains only zeros")
    coordinate_logs = evaluation.binary_exponent[:, None].to(absolute.dtype) + torch.where(
        nonzero,
        absolute.log2(),
        torch.full_like(absolute, float("inf")),
    )
    finite_coordinate_logs = coordinate_logs[nonzero]
    relative_denominator = absolute[:, 1] / absolute.amax(dim=1).clamp_min(
        torch.finfo(absolute.dtype).tiny
    )
    log10_two = math.log10(2.0)
    return {
        "count": int(evaluation.mantissa.shape[0]),
        "minimum_nonzero_log10_absolute_pair_coordinate": float(
            finite_coordinate_logs.min().item() * log10_two
        ),
        "maximum_log10_absolute_pair_coordinate": float(
            finite_coordinate_logs.max().item() * log10_two
        ),
        "minimum_relative_denominator": float(relative_denominator.min().item()),
        "maximum_relative_denominator": float(relative_denominator.max().item()),
        "all_denominator_mantissas_positive": bool((evaluation.mantissa[:, 1] > 0).all()),
    }


def _leaf_input_pair(index: int, raw_dimension: int, like: Tensor, label: str) -> MultilinearTreeNode:
    core = like.new_zeros(2, raw_dimension + 1)
    core[0, index + 1] = 1.0
    core[1, 0] = 1.0
    return _normalized_node(label, core)


def _leaf_constant_pair(
    value: Tensor | float,
    raw_dimension: int,
    like: Tensor,
    label: str,
) -> MultilinearTreeNode:
    scalar = torch.as_tensor(value, dtype=like.dtype, device=like.device).reshape(())
    core = like.new_zeros(2, raw_dimension + 1)
    core[0, 0] = scalar
    core[1, 0] = 1.0
    return _normalized_node(label, core)


def _pair_affine(
    pair: MultilinearTreeNode,
    weight: Tensor | float,
    bias: Tensor | float,
    label: str,
) -> MultilinearTreeNode:
    like = pair.core
    w = torch.as_tensor(weight, dtype=like.dtype, device=like.device).reshape(())
    b = torch.as_tensor(bias, dtype=like.dtype, device=like.device).reshape(())
    core = like.new_zeros(2, 2)
    core[0, 0] = w
    core[0, 1] = b
    core[1, 1] = 1.0
    return _normalized_node(label, core, (pair,))


def _pair_scale(pair: MultilinearTreeNode, scale: Tensor | float, label: str) -> MultilinearTreeNode:
    return _pair_affine(pair, scale, 0.0, label)


def _pair_product(
    left: MultilinearTreeNode,
    right: MultilinearTreeNode,
    label: str,
) -> MultilinearTreeNode:
    core = left.core.new_zeros(2, 2, 2)
    core[0, 0, 0] = 1.0
    core[1, 1, 1] = 1.0
    return _normalized_node(label, core, (left, right))


def _pair_add(
    left: MultilinearTreeNode,
    right: MultilinearTreeNode,
    label: str,
) -> MultilinearTreeNode:
    core = left.core.new_zeros(2, 2, 2)
    core[0, 0, 1] = 1.0
    core[0, 1, 0] = 1.0
    core[1, 1, 1] = 1.0
    return _normalized_node(label, core, (left, right))


def _homogeneous_pade_core(module: RationalNorm, like: Tensor) -> Tensor:
    degree = int(module.deg)
    if degree < 1:
        raise ValueError("global projective oracle requires Padé degree at least one")
    core = like.new_zeros((2,) + (2,) * degree)
    for power in range(degree + 1):
        indices = (0,) * power + (1,) * (degree - power)
        permutations = tuple(set(itertools.permutations(indices)))
        for output, coefficients in enumerate((module.pa, module.pb)):
            coefficient = coefficients[power].to(dtype=like.dtype, device=like.device)
            for permutation in permutations:
                core[(output, *permutation)] = coefficient / len(permutations)
    return core


def _pair_pade_norm(
    pair: MultilinearTreeNode,
    module: RationalNorm,
    label: str,
) -> MultilinearTreeNode:
    if not isinstance(module, RationalNorm) or module.variant != "pade":
        raise ValueError(f"{label} must be a Padé RationalNorm")
    if module.training and not module.frozen:
        raise ValueError(f"{label} must be eval-mode or frozen")
    if not bool(module.initialized):
        raise ValueError(f"{label} running_ms must be initialized")
    if module.pa.numel() != module.deg + 1 or module.pb.numel() != module.deg + 1:
        raise ValueError(f"{label} Padé buffer lengths disagree with degree")
    like = pair.core
    scale = module.running_ms.to(dtype=like.dtype, device=like.device).clamp_min(1e-12)

    # [S,T] = [N^2 + eps D^2, s0 D^2].
    st_core = like.new_zeros(2, 2, 2)
    st_core[0, 0, 0] = 1.0
    st_core[0, 1, 1] = float(module.eps)
    st_core[1, 1, 1] = scale
    st = _normalized_node(f"{label}.mean_square_homogeneous", st_core, (pair, pair))

    # [P_h(S,T), Q_h(S,T)] with the common T^degree denominator cancelled.
    pade_core = _homogeneous_pade_core(module, like)
    pade = _normalized_node(
        f"{label}.pade_homogeneous",
        pade_core,
        (st,) * int(module.deg),
    )

    # [N P_h / sqrt(s0), D Q_h].
    output_core = like.new_zeros(2, 2, 2)
    output_core[0, 0, 0] = torch.rsqrt(scale)
    output_core[1, 1, 1] = 1.0
    return _normalized_node(f"{label}.projective_output", output_core, (pair, pade))


def _norm_snapshot(block: ChiTransformerBlock) -> tuple[tuple[str, Tensor], ...]:
    snapshots: list[tuple[str, Tensor]] = []
    for name, module in block.named_modules():
        if isinstance(module, RationalNorm):
            snapshots.extend(
                (
                    (f"{name}.running_ms", module.running_ms.detach().clone()),
                    (f"{name}.initialized", module.initialized.detach().clone()),
                    (f"{name}.pa", module.pa.detach().clone()),
                    (f"{name}.pb", module.pb.detach().clone()),
                )
            )
    return tuple(snapshots)


@torch.no_grad()
def projective_pair_coefficient_metric(network: ProjectivePolynomialTree) -> Tensor:
    """Return the exact independent-clone coefficient Gram of [N,D]."""

    _validate_network(network)
    memo: dict[int, Tensor] = {}

    def prefix(node: MultilinearTreeNode) -> Tensor:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        rows = node.core.reshape(node.core.shape[0], -1)
        if not node.children:
            metric = rows @ rows.T
        else:
            product = node.core.new_ones(1, 1)
            for child in node.children:
                product = torch.kron(product, prefix(child))
            metric = rows @ product @ rows.T
        metric = 0.5 * (metric + metric.T)
        memo[id(node)] = metric
        return metric

    metric = network.head @ prefix(network.root) @ network.head.T
    return 0.5 * (metric + metric.T)


@torch.no_grad()
def projective_pair_coefficient_norm(network: ProjectivePolynomialTree) -> Tensor:
    metric = projective_pair_coefficient_metric(network)
    return torch.diagonal(metric).sum().sqrt()


def _validate_source_modules(block: ChiTransformerBlock, action_head: nn.Linear) -> None:
    if not isinstance(block, ChiTransformerBlock):
        raise TypeError("block must be ChiTransformerBlock")
    if block.training or action_head.training:
        raise ValueError("block and action head must be in eval mode")
    if not block.residual:
        raise ValueError("global projective oracle requires both residual routes")
    if not isinstance(block.attn, BilinearAttention) or block.attn.qk_norm != "rational":
        raise ValueError("attention must be BilinearAttention(qk_norm='rational')")
    if block.attn.dim != 1 or block.attn.n_heads != 1:
        raise ValueError("dense global oracle is intentionally restricted to D=H=1")
    if not isinstance(block.ffn, BilinearFFN) or block.ffn.dim != 1 or block.ffn.out_dim != 1:
        raise ValueError("FFN must be a scalar BilinearFFN")
    if not isinstance(block.rbn_attn, RationalNorm) or not isinstance(block.rbn_ffn, RationalNorm):
        raise ValueError("both block pre-norms must be Padé RationalNorm")
    if not isinstance(action_head, nn.Linear) or action_head.in_features != 1 or action_head.out_features != 1:
        raise ValueError("action_head must be a biased scalar Linear")
    if action_head.bias is None:
        raise ValueError("action_head bias must be retained")
    tensors = tuple(block.parameters()) + tuple(action_head.parameters())
    dtypes = {tensor.dtype for tensor in tensors if tensor.is_floating_point()}
    devices = {tensor.device for tensor in tensors}
    if dtypes != {torch.float64} or len(devices) != 1:
        raise ValueError("exact global oracle requires one float64 dtype and device")
    norms = [module for module in block.modules() if isinstance(module, RationalNorm)]
    if len(norms) != 6:
        raise ValueError("expected pre-attention, four Q/K, and pre-FFN RationalNorm modules")
    for index, norm in enumerate(norms):
        if norm.variant != "pade" or not bool(norm.initialized):
            raise ValueError(f"RationalNorm site {index} must be initialized Padé")


@torch.no_grad()
def compile_global_projective_block(
    block: ChiTransformerBlock,
    action_head: nn.Linear,
    *,
    token_index: int = 1,
    mask: Tensor | None = None,
) -> GlobalProjectiveBlockOracle:
    """Compile one complete selected-token policy into one global pair tree."""

    _validate_source_modules(block, action_head)
    if token_index != 1:
        raise ValueError("the N=2 causal oracle selects token_index=1 to retain both source routes")
    like = action_head.weight
    fixed_mask = causal_mask(2, dtype=like.dtype, device=like.device) if mask is None else mask
    _validate_real_finite(fixed_mask, name="mask")
    if fixed_mask.shape != (2, 2) or not bool(((fixed_mask == 0) | (fixed_mask == 1)).all()):
        raise ValueError("mask must be a fixed binary 2 by 2 tensor")
    if not bool(fixed_mask[token_index].all()):
        raise ValueError("selected query must retain both attention source routes")

    raw_dimension = 2
    raw = tuple(
        _leaf_input_pair(index, raw_dimension, like, f"input.token{index}")
        for index in range(2)
    )
    normalized = tuple(
        _pair_pade_norm(pair, block.rbn_attn, f"attention_pre_norm.token{index}")
        for index, pair in enumerate(raw)
    )

    def projection_pair(linear: nn.Linear, source: int, label: str) -> MultilinearTreeNode:
        return _pair_affine(
            normalized[source], linear.weight[0, 0], linear.bias[0], label
        )

    query_norms = {
        "q1": block.attn.rn_q1,
        "q2": block.attn.rn_q2,
    }
    key_norms = {
        "k1": block.attn.rn_k1,
        "k2": block.attn.rn_k2,
    }
    query_linears = {"q1": block.attn.wq1, "q2": block.attn.wq2}
    key_linears = {"k1": block.attn.wk1, "k2": block.attn.wk2}
    queries = {
        name: _pair_pade_norm(
            projection_pair(linear, token_index, f"attention.{name}.affine.q{token_index}"),
            query_norms[name],
            f"attention.{name}.norm.q{token_index}",
        )
        for name, linear in query_linears.items()
    }
    keys = {
        (name, source): _pair_pade_norm(
            projection_pair(linear, source, f"attention.{name}.affine.k{source}"),
            key_norms[name],
            f"attention.{name}.norm.k{source}",
        )
        for name, linear in key_linears.items()
        for source in range(2)
    }
    values = tuple(
        projection_pair(block.attn.wv, source, f"attention.v.affine.source{source}")
        for source in range(2)
    )

    attention = _leaf_constant_pair(
        block.attn.wo.bias[0], raw_dimension, like, "attention.output_bias"
    )
    visible = fixed_mask[token_index].sum()
    row_scale = torch.rsqrt(visible) if block.attn.row_scale == "invsqrt" else visible.reciprocal()
    fixed_scale = (
        block.attn.wo.weight[0, 0]
        * row_scale
        / float(block.attn._score_denom)
    )
    for source in range(2):
        factors = (
            queries["q1"],
            keys[("k1", source)],
            queries["q2"],
            keys[("k2", source)],
            values[source],
        )
        contribution = factors[0]
        for factor_index, factor in enumerate(factors[1:], start=1):
            contribution = _pair_product(
                contribution,
                factor,
                f"attention.route{source}.product{factor_index}",
            )
        contribution = _pair_scale(
            contribution, fixed_scale, f"attention.route{source}.fixed_scale"
        )
        attention = _pair_add(
            attention, contribution, f"attention.route{source}.cross_multiplied_sum"
        )

    attention = _pair_scale(attention, block.attn_gain, "attention.residual_gain")
    after_attention = _pair_add(raw[token_index], attention, "attention.residual_add")
    ffn_input = _pair_pade_norm(after_attention, block.rbn_ffn, "ffn_pre_norm")

    ffn_output = _leaf_constant_pair(
        0.0 if block.ffn.down.bias is None else block.ffn.down.bias[0],
        raw_dimension,
        like,
        "ffn.output_bias",
    )
    for rank in range(block.ffn.rank):
        left = _pair_affine(
            ffn_input,
            block.ffn.left.weight[rank, 0],
            block.ffn.left.bias[rank],
            f"ffn.left.rank{rank}",
        )
        right = _pair_affine(
            ffn_input,
            block.ffn.right.weight[rank, 0],
            block.ffn.right.bias[rank],
            f"ffn.right.rank{rank}",
        )
        product = _pair_product(left, right, f"ffn.product.rank{rank}")
        contribution = _pair_scale(
            product, block.ffn.down.weight[0, rank], f"ffn.down.rank{rank}"
        )
        ffn_output = _pair_add(
            ffn_output, contribution, f"ffn.rank{rank}.cross_multiplied_sum"
        )
    ffn_output = _pair_scale(ffn_output, block.ffn_gain, "ffn.residual_gain")
    after_ffn = _pair_add(after_attention, ffn_output, "ffn.residual_add")
    action = _pair_affine(
        after_ffn, action_head.weight[0, 0], action_head.bias[0], "action_head"
    )

    raw_network = ProjectivePolynomialTree(
        root=action,
        head=torch.eye(2, dtype=like.dtype, device=like.device),
        raw_input_dimension=raw_dimension,
        gauge=UNFIXED_GAUGE,
    )
    anchor = like.new_zeros(1, raw_dimension)
    unfixed_anchor = evaluate_scaled_projective_pair(raw_network, anchor)
    denominator_mantissa = unfixed_anchor.mantissa[0, 1].abs()
    if float(denominator_mantissa.item()) == 0.0:
        raise ValueError("normalized projective tree has structurally zero D(x=0)")
    unfixed_log10 = float(
        unfixed_anchor.binary_exponent[0].item() * math.log10(2.0)
        + math.log10(float(denominator_mantissa.item()))
    )
    old_refix_head_log10 = (
        math.log10(float(raw_network.head.abs().max().item())) - unfixed_log10
    )
    fixed = fix_projective_gauge(raw_network)
    return GlobalProjectiveBlockOracle(
        network=fixed,
        block=block,
        action_head=action_head,
        token_index=token_index,
        mask=fixed_mask.detach().clone(),
        norm_buffer_snapshot=_norm_snapshot(block),
        unfixed_log10_absolute_denominator_at_zero=unfixed_log10,
        old_anchor_refix_predicted_head_log10_maximum=old_refix_head_log10,
        fixed_head_maximum_absolute=float(fixed.head.abs().max().item()),
        fixed_coefficient_norm=float(projective_pair_coefficient_norm(fixed).item()),
    )


@torch.no_grad()
def source_action(oracle: GlobalProjectiveBlockOracle, raw_input: Tensor) -> Tensor:
    """Evaluate the unchanged PyTorch block and biased action head."""

    tokens = raw_input.reshape(raw_input.shape[0], 2, 1)
    output = oracle.block(tokens, mask=oracle.mask, method="explicit")
    return oracle.action_head(output[:, oracle.token_index]).squeeze(-1)


@torch.no_grad()
def assert_norm_buffers_unchanged(oracle: GlobalProjectiveBlockOracle) -> None:
    current = dict(_norm_snapshot(oracle.block))
    for name, expected in oracle.norm_buffer_snapshot:
        if name not in current or not torch.equal(current[name], expected):
            raise RuntimeError(f"normalization buffer changed during compilation: {name}")


@torch.no_grad()
def rescale_projective_output(
    network: ProjectivePolynomialTree,
    scale: Tensor | float,
) -> ProjectivePolynomialTree:
    """Apply a nonzero constant projective output gauge."""

    _validate_network(network)
    scalar = torch.as_tensor(scale, dtype=network.head.dtype, device=network.head.device).reshape(())
    if not bool(torch.isfinite(scalar)) or float(scalar.abs().item()) == 0.0:
        raise ValueError("projective scale must be finite and nonzero")
    return ProjectivePolynomialTree(
        network.root,
        network.head * scalar,
        network.raw_input_dimension,
        UNFIXED_GAUGE,
    )


@torch.no_grad()
def fix_projective_gauge(network: ProjectivePolynomialTree) -> ProjectivePolynomialTree:
    """Fix constant scale by unit coefficient norm and positive zero-anchor D."""

    _validate_network(network)
    norm = projective_pair_coefficient_norm(network)
    if not bool(torch.isfinite(norm)) or float(norm.item()) == 0.0:
        raise ValueError("cannot fix a zero or nonfinite projective coefficient norm")
    normalized_head = network.head / norm
    normalized = ProjectivePolynomialTree(
        network.root,
        normalized_head,
        network.raw_input_dimension,
        UNFIXED_GAUGE,
    )
    anchor = network.head.new_zeros(1, network.raw_input_dimension)
    denominator = evaluate_scaled_projective_pair(normalized, anchor).mantissa[0, 1]
    if not bool(torch.isfinite(denominator)) or float(denominator.item()) == 0.0:
        raise ValueError("cannot orient projective gauge because D(x=0) is zero or nonfinite")
    sign = 1.0 if float(denominator.item()) > 0.0 else -1.0
    fixed = ProjectivePolynomialTree(
        network.root,
        normalized_head * sign,
        network.raw_input_dimension,
        FIXED_GAUGE,
    )
    _require_numeric_fixed_gauge(fixed)
    return fixed


@torch.no_grad()
def _require_numeric_fixed_gauge(network: ProjectivePolynomialTree) -> None:
    if network.gauge != FIXED_GAUGE:
        raise ValueError("canonical ODT requires an explicitly fixed projective gauge")
    norm = projective_pair_coefficient_norm(network)
    tolerance = 20_000.0 * torch.finfo(norm.dtype).eps
    if abs(float(norm.item()) - 1.0) > tolerance:
        raise ValueError(
            "projective gauge metadata is inconsistent: coefficient norm is not one"
        )
    anchor = network.head.new_zeros(1, network.raw_input_dimension)
    denominator = evaluate_scaled_projective_pair(network, anchor).mantissa[0, 1]
    if not bool(torch.isfinite(denominator)) or float(denominator.item()) <= 0.0:
        raise ValueError(
            "projective gauge metadata is inconsistent: numeric D(x=0) is not positive"
        )


def _relative_error(actual: Tensor, expected: Tensor) -> float:
    scale = torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)
    return float((torch.linalg.vector_norm(actual - expected) / scale).item())


def _row_isometry_error(matrix: Tensor) -> float:
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    return float(torch.linalg.matrix_norm(matrix @ matrix.T - identity).item())


def _supported_reduced_rq_rows(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Direct reduced RQ, including rectangular and singular unfoldings.

    Numerical-rank truncation is deliberately absent.  This is the same
    QR-of-the-transpose primitive used by canonical ODT and keeps Algorithm 1
    distinct from any later compression decision.
    """

    return reduced_rq_rows(matrix)


def _apply_input_factor(core: Tensor, role: int, factor: Tensor) -> Tensor:
    axis = role + 1
    moved = core.movedim(axis, -1) @ factor
    return moved.movedim(-1, axis)


def _apply_output_factor(core: Tensor, factor: Tensor) -> Tensor:
    return torch.tensordot(factor, core, dims=([1], [0]))


def node_at_path(
    network: ProjectivePolynomialTree,
    path: tuple[int, ...],
) -> MultilinearTreeNode:
    """Resolve one clone occurrence by its role-index path from the root."""

    _validate_network(network)
    node = network.root
    for depth, role in enumerate(path):
        if isinstance(role, bool) or not isinstance(role, int) or not 0 <= role < len(node.children):
            raise ValueError(f"invalid role {role!r} at path depth {depth}")
        node = node.children[role]
    return node


@torch.no_grad()
def gauge_tree_occurrence(
    network: ProjectivePolynomialTree,
    path: tuple[int, ...],
    transform: Tensor,
) -> ProjectivePolynomialTree:
    """Apply one hidden nonorthogonal gauge and its exact parent inverse.

    A shared Python node is cloned only along the selected occurrence path.
    This ensures the test exercises clone-unfolded occurrence semantics rather
    than accidentally gauging every syntactically shared use.
    """

    _validate_network(network)
    target = node_at_path(network, path)
    if transform.shape != (target.output_dimension, target.output_dimension):
        raise ValueError("gauge transform shape mismatch")
    if transform.dtype != network.head.dtype or transform.device != network.head.device:
        raise ValueError("gauge transform and tree must share dtype and device")
    _validate_real_finite(transform, name="transform")
    inverse = torch.linalg.inv(transform)

    def rewrite(node: MultilinearTreeNode, suffix: tuple[int, ...]) -> MultilinearTreeNode:
        if not suffix:
            core = _apply_output_factor(node.core, transform)
            return MultilinearTreeNode(node.label, core, node.children)
        role = suffix[0]
        children = list(node.children)
        children[role] = rewrite(children[role], suffix[1:])
        # Only the target's direct parent needs the inverse.  That compensation
        # restores the parent's output exactly, so no higher ancestor changes.
        core = (
            _apply_input_factor(node.core, role, inverse)
            if len(suffix) == 1
            else node.core.detach().clone()
        )
        return MultilinearTreeNode(node.label, core, tuple(children))

    if not path:
        root = rewrite(network.root, ())
        head = network.head @ inverse
    else:
        root = rewrite(network.root, path)
        head = network.head.detach().clone()
    gauged = ProjectivePolynomialTree(
        root,
        head,
        network.raw_input_dimension,
        network.gauge,
    )
    _validate_network(gauged)
    return gauged


@torch.no_grad()
def brute_identity_role_environment(
    core: Tensor,
    downstream_metric: Tensor,
    role: int,
) -> Tensor:
    """Independent loop oracle for one identity-sibling role environment."""

    if core.ndim < 2 or downstream_metric.shape != (core.shape[0], core.shape[0]):
        raise ValueError("core or downstream metric shape mismatch")
    arity = core.ndim - 1
    if isinstance(role, bool) or not isinstance(role, int) or not 0 <= role < arity:
        raise ValueError("role index out of range")
    result = core.new_zeros(core.shape[role + 1], core.shape[role + 1])
    other_roles = tuple(index for index in range(arity) if index != role)
    other_ranges: Iterable[range] = tuple(range(core.shape[index + 1]) for index in other_roles)
    for left_index in range(result.shape[0]):
        for right_index in range(result.shape[1]):
            total = core.new_zeros(())
            for context in itertools.product(*other_ranges):
                left_inputs = [0] * arity
                right_inputs = [0] * arity
                left_inputs[role] = left_index
                right_inputs[role] = right_index
                for other_role, value in zip(other_roles, context):
                    left_inputs[other_role] = value
                    right_inputs[other_role] = value
                for output_left in range(core.shape[0]):
                    for output_right in range(core.shape[0]):
                        total = total + (
                            core[(output_left, *left_inputs)]
                            * downstream_metric[output_left, output_right]
                            * core[(output_right, *right_inputs)]
                        )
            result[left_index, right_index] = total
    return 0.5 * (result + result.T)


@torch.no_grad()
def canonicalize_projective_tree(network: ProjectivePolynomialTree) -> CanonicalProjectiveODT:
    """Dooms Algorithm 1 on the complete heterogeneous multilinear tree."""

    _validate_network(network)
    _require_numeric_fixed_gauge(network)
    isometry_errors: list[float] = []
    factorization_errors: list[float] = []
    memo: dict[int, tuple[MultilinearTreeNode, Tensor]] = {}

    def canonicalize(node: MultilinearTreeNode) -> tuple[MultilinearTreeNode, Tensor]:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        if node.children:
            child_results = tuple(canonicalize(child) for child in node.children)
            children = tuple(result[0] for result in child_results)
            transformed = node.core.detach().clone()
            for role, (_, factor) in enumerate(child_results):
                transformed = _apply_input_factor(transformed, role, factor)
        else:
            children = ()
            transformed = node.core.detach().clone()
        rows = transformed.reshape(transformed.shape[0], -1)
        factor, isometric_rows = _supported_reduced_rq_rows(rows)
        isometric_core = isometric_rows.reshape((isometric_rows.shape[0],) + transformed.shape[1:])
        canonical_node = MultilinearTreeNode(
            node.label,
            isometric_core,
            children,
            torch.eye(
                isometric_core.shape[0],
                dtype=isometric_core.dtype,
                device=isometric_core.device,
            ),
        )
        isometry_errors.append(_row_isometry_error(isometric_rows))
        factorization_errors.append(_relative_error(factor @ isometric_rows, rows))
        result = (canonical_node, factor)
        memo[id(node)] = result
        return result

    root, root_factor = canonicalize(network.root)
    canonical_network = ProjectivePolynomialTree(
        root,
        network.head @ root_factor,
        network.raw_input_dimension,
        network.gauge,
    )
    _validate_network(canonical_network)
    return CanonicalProjectiveODT(
        canonical_network,
        tuple(isometry_errors),
        tuple(factorization_errors),
        len(memo),
        _occurrence_count(canonical_network.root),
        tuple("direct_reduced_rq" for _ in isometry_errors),
    )


def _require_row_isometric_tree(network: ProjectivePolynomialTree) -> None:
    _validate_network(network)
    _require_numeric_fixed_gauge(network)
    for node in _walk_unique(network.root):
        rows = node.core.reshape(node.core.shape[0], -1)
        tolerance = 2e-8 * max(1.0, math.sqrt(rows.shape[0]))
        if _row_isometry_error(rows) > tolerance:
            raise ValueError(f"node {node.label!r} is not row-isometric")


def scale_symmetric_message(matrix: Tensor, inherited_exponent: int = 0) -> tuple[Tensor, int]:
    """Strip one exact power-of-two scale from a symmetric metric message.

    The represented matrix is ``ldexp(mantissa, binary_exponent)``.  This is
    the matrix analogue of :func:`evaluate_scaled_projective_pair` and mirrors
    exponent-stripping tensor-network contractions.  Multiplying a Gram by a
    positive scalar changes its eigenvalues but not its eigenvectors.
    """

    _validate_real_finite(matrix, name="symmetric metric message")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("symmetric metric message must be square")
    if isinstance(inherited_exponent, bool) or not isinstance(inherited_exponent, int):
        raise TypeError("inherited_exponent must be an integer")

    symmetric = 0.5 * (matrix + matrix.T)
    scale = symmetric.abs().max()
    if not bool(torch.isfinite(scale)):
        raise ValueError("metric message became nonfinite")
    if float(scale.item()) == 0.0:
        return symmetric, inherited_exponent
    _, local_exponent = torch.frexp(scale)
    exponent = int(local_exponent.item())
    shift = torch.tensor(-exponent, dtype=torch.int64, device=symmetric.device)
    return torch.ldexp(symmetric, shift), inherited_exponent + exponent


def add_scaled_symmetric_messages(
    messages: Iterable[tuple[Tensor, int]],
) -> tuple[Tensor, int]:
    """Add Gram messages after aligning their binary exponents.

    The current clone-unfolded oracle keeps occurrence environments separate,
    as Dooms Algorithm 2 requires.  A production compiler may tie equivalent
    occurrence bonds and must then sum their environments.  This helper makes
    that sum explicit and prevents a silent addition of incompatible mantissa
    charts.
    """

    material = tuple(messages)
    if not material:
        raise ValueError("at least one scaled symmetric message is required")
    reference, _ = material[0]
    _validate_real_finite(reference, name="scaled symmetric mantissa")
    if reference.ndim != 2 or reference.shape[0] != reference.shape[1]:
        raise ValueError("scaled symmetric mantissas must be square")
    maximum_exponent = max(exponent for _, exponent in material)
    aligned = reference.new_zeros(reference.shape)
    for matrix, exponent in material:
        _validate_real_finite(matrix, name="scaled symmetric mantissa")
        if matrix.shape != reference.shape or matrix.dtype != reference.dtype:
            raise ValueError("scaled symmetric messages must share shape and dtype")
        if matrix.device != reference.device:
            raise ValueError("scaled symmetric messages must share device")
        if isinstance(exponent, bool) or not isinstance(exponent, int):
            raise TypeError("scaled symmetric exponents must be integers")
        shift = torch.tensor(
            exponent - maximum_exponent, dtype=torch.int64, device=matrix.device
        )
        aligned = aligned + torch.ldexp(matrix, shift)
    return scale_symmetric_message(aligned, maximum_exponent)


@torch.no_grad()
def canonical_projective_environments(
    network: ProjectivePolynomialTree,
    output_metric: Tensor | None = None,
) -> tuple[EnvironmentRecord, ...]:
    """Dooms Algorithm 2 on every occurrence of the clone-unfolded tree."""

    _require_row_isometric_tree(network)
    if output_metric is None:
        output_metric = torch.eye(2, dtype=network.head.dtype, device=network.head.device)
    if output_metric.shape != (2, 2):
        raise ValueError("output_metric must have shape (2, 2)")
    output_metric = 0.5 * (output_metric + output_metric.T)
    if float(torch.linalg.eigvalsh(output_metric)[0].item()) < -1e-12:
        raise ValueError("output_metric must be positive semidefinite")
    records: list[EnvironmentRecord] = []

    def descend(
        node: MultilinearTreeNode,
        gram: Tensor,
        binary_exponent: int,
        path: tuple[int, ...],
    ) -> None:
        records.append(EnvironmentRecord(path, node.label, gram, binary_exponent))
        if not node.children:
            return
        identities = tuple(
            torch.eye(child.output_dimension, dtype=node.core.dtype, device=node.core.device)
            for child in node.children
        )
        for role, child in enumerate(node.children):
            child_gram = role_environment(node.core, gram, identities, role)
            child_mantissa, child_exponent = scale_symmetric_message(
                child_gram, binary_exponent
            )
            descend(child, child_mantissa, child_exponent, path + (role,))

    root_gram = network.head.T @ output_metric @ network.head
    root_mantissa, root_exponent = scale_symmetric_message(root_gram)
    descend(network.root, root_mantissa, root_exponent, ())
    return tuple(records)


@torch.no_grad()
def diagonalize_projective_tree_full_rank(
    network: ProjectivePolynomialTree,
    output_metric: Tensor | None = None,
) -> FullRankDiagonalProjectiveODT:
    """Dooms Algorithm 3 with every eigenvector retained at every bond."""

    records = canonical_projective_environments(network, output_metric=output_metric)
    eigenvalues: list[Tensor] = []
    eigenvalue_exponents: list[int] = []
    errors: list[float] = []
    paths: list[tuple[int, ...]] = []
    bases: dict[tuple[int, ...], Tensor] = {}
    for record in records:
        values, vectors = torch.linalg.eigh(0.5 * (record.gram + record.gram.T))
        order = torch.argsort(values, descending=True)
        values = values[order].clamp_min(0)
        vectors = vectors[:, order]
        diagonal = vectors.T @ record.gram @ vectors
        off_diagonal = diagonal - torch.diag(torch.diagonal(diagonal))
        scale = torch.linalg.matrix_norm(record.gram).clamp_min(torch.finfo(record.gram.dtype).tiny)
        errors.append(float((torch.linalg.matrix_norm(off_diagonal) / scale).item()))
        eigenvalues.append(values)
        eigenvalue_exponents.append(record.binary_exponent)
        paths.append(record.path)
        bases[record.path] = vectors

    def rotate(node: MultilinearTreeNode, path: tuple[int, ...]) -> MultilinearTreeNode:
        children = tuple(rotate(child, path + (role,)) for role, child in enumerate(node.children))
        core = node.core.detach().clone()
        for role in range(len(children)):
            core = _apply_input_factor(core, role, bases[path + (role,)])
        core = torch.tensordot(bases[path].T, core, dims=([1], [0]))
        return MultilinearTreeNode(
            node.label,
            core,
            children,
            torch.eye(core.shape[0], dtype=core.dtype, device=core.device),
        )

    root = rotate(network.root, ())
    diagonal_network = ProjectivePolynomialTree(
        root,
        network.head @ bases[()],
        network.raw_input_dimension,
        network.gauge,
    )
    _validate_network(diagonal_network)
    return FullRankDiagonalProjectiveODT(
        diagonal_network,
        tuple(eigenvalues),
        tuple(eigenvalue_exponents),
        tuple(errors),
        tuple(paths),
    )


def quotient_pair_output_metric(pair: Tensor) -> Tensor:
    """Exact quotient pullback on ambient [N,D], including both cross terms."""

    _validate_real_finite(pair, name="pair")
    if pair.shape != (2,):
        raise ValueError("pair must have shape (2,)")
    # Choose a harmless local scalar representative for this diagnostic so the
    # quotient Jacobian cannot overflow merely because the global polynomial
    # pair carries a very small common coordinate scale.
    scale = pair.abs().max()
    if float(scale.item()) == 0.0:
        raise ValueError("projective pair cannot be the zero vector")
    numerator, denominator = (pair / scale).unbind()
    if float(denominator.abs().item()) <= 100.0 * torch.finfo(pair.dtype).eps:
        raise ValueError("denominator is too small relative to the projective pair")
    jacobian = torch.stack((denominator.reciprocal(), -numerator / denominator.square()))
    return jacobian[:, None] @ jacobian[None, :]


def omit_quotient_cross_terms_negative_control(metric: Tensor) -> Tensor:
    """Delete the two N-D cross terms from a quotient pullback."""

    if metric.shape != (2, 2):
        raise ValueError("metric must have shape (2, 2)")
    result = metric.detach().clone()
    result[0, 1] = 0.0
    result[1, 0] = 0.0
    return result


@torch.no_grad()
def drop_first_projective_add_cross_term(
    network: ProjectivePolynomialTree,
    *,
    label_contains: str = "residual_add",
) -> ProjectivePolynomialTree:
    """Adversarially remove one numerator-denominator cross product."""

    _validate_network(network)
    changed = False

    def rewrite(node: MultilinearTreeNode) -> MultilinearTreeNode:
        nonlocal changed
        children = tuple(rewrite(child) for child in node.children)
        core = node.core.detach().clone()
        if not changed and label_contains in node.label:
            if core.shape != (2, 2, 2):
                raise ValueError("selected node is not a projective binary addition core")
            core[0, 0, 1] = 0.0
            changed = True
        return MultilinearTreeNode(node.label, core, children)

    root = rewrite(network.root)
    if not changed:
        raise ValueError(f"no projective add node contained {label_contains!r}")
    return ProjectivePolynomialTree(
        root,
        network.head.detach().clone(),
        network.raw_input_dimension,
        network.gauge,
    )


def denominator_statistics(pair: Tensor) -> dict[str, float | bool | int]:
    if pair.ndim != 2 or pair.shape[1] != 2:
        raise ValueError("pair must have shape (batch, 2)")
    denominator = pair[:, 1]
    scale = pair.abs().amax(dim=1).clamp_min(torch.finfo(pair.dtype).tiny)
    relative = denominator.abs() / scale
    return {
        "count": int(denominator.numel()),
        "minimum_absolute": float(denominator.abs().min().item()),
        "maximum_absolute": float(denominator.abs().max().item()),
        "minimum_relative_to_pair": float(relative.min().item()),
        "maximum_relative_to_pair": float(relative.max().item()),
        "all_positive": bool((denominator > 0).all()),
    }


__all__ = [
    "CLAIM_BOUNDARY",
    "FIXED_GAUGE",
    "OBJECT_KIND",
    "UNFIXED_GAUGE",
    "CanonicalProjectiveODT",
    "EnvironmentRecord",
    "FullRankDiagonalProjectiveODT",
    "GlobalProjectiveBlockOracle",
    "MultilinearTreeNode",
    "ProjectivePolynomialTree",
    "ScaledProjectiveEvaluation",
    "add_scaled_symmetric_messages",
    "assert_norm_buffers_unchanged",
    "brute_identity_role_environment",
    "canonical_projective_environments",
    "canonicalize_projective_tree",
    "compile_global_projective_block",
    "diagonalize_projective_tree_full_rank",
    "drop_first_projective_add_cross_term",
    "evaluate_projective_pair",
    "evaluate_projective_pair_unscaled_negative_control",
    "evaluate_quotient",
    "evaluate_scaled_projective_pair",
    "fix_projective_gauge",
    "gauge_tree_occurrence",
    "node_at_path",
    "omit_quotient_cross_terms_negative_control",
    "projective_pair_coefficient_metric",
    "projective_pair_coefficient_norm",
    "quotient_pair_output_metric",
    "rescale_projective_output",
    "scale_symmetric_message",
    "scaled_projective_relative_error",
    "scaled_projective_statistics",
    "source_action",
]
