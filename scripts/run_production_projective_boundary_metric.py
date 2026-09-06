#!/usr/bin/env python3
"""Exact factorized projective boundary-metric stress for one production block.

This runner deliberately stops at a single selected-token block map.  It loads
the pinned capable Padé checkpoint, compiles one real residual attention/FFN
block into a shared typed arithmetic DAG, replays that DAG against the unchanged
PyTorch module, and forms the Dooms-style clone-unfolded coefficient metric at
the block input boundary.  It does not materialize a dense whole-block core and
does not perform EVDs of ephemeral route bonds.

All primitive cores are either unary affine matrices or sparse multilinear term
lists.  Prefix Grams are propagated bottom-up and occurrence environments are
summed top-down with a mantissa plus binary-exponent ledger.  A built-in tiny
N=2,D=1 control materializes every primitive core and explicitly clone-unfolds
the graph, independently checking the factorized prefix and environment paths.

Scope is intentionally narrow: a passing production case is a one-block,
selected-token boundary metric.  It is not a twelve-block VLA traversal, a
quotient-intrinsic metric, an identified-variable polynomial metric, or a
compression certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_production_factorized_odt_stress import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    production_config,
)
from xvla.models.vla import ChiVLA  # noqa: E402
from xvla.nn.attention import causal_mask  # noqa: E402
from xvla.nn.block import ChiTransformerBlock  # noqa: E402
from xvla.nn.normalization import RationalNorm  # noqa: E402
from xvla.train.generalized_metric_odt import (  # noqa: E402
    forward_metric,
    role_environment,
)


DTYPE = torch.float64
CLAIM_BOUNDARY = (
    "Exact clone-unfolded syntactic coefficient metric at the input boundary "
    "of one selected-token production ChiTransformerBlock map. No dense block "
    "core and no ephemeral-bond EVD are formed. This is not an identified-"
    "variable or quotient-intrinsic metric, a complete VLA traversal, or a "
    "compression certificate."
)


@dataclass(frozen=True)
class ScaledMatrix:
    mantissa: torch.Tensor
    binary_exponent: int


@dataclass(frozen=True)
class SparseCore:
    output_dimension: int
    input_dimensions: tuple[int, ...]
    output_indices: torch.Tensor
    input_indices: tuple[torch.Tensor, ...]
    coefficients: torch.Tensor

    @property
    def arity(self) -> int:
        return len(self.input_dimensions)

    @property
    def term_count(self) -> int:
        return int(self.coefficients.numel())


@dataclass(eq=False)
class MetricNode:
    label: str
    output_dimension: int
    children: tuple["MetricNode", ...] = ()
    affine: torch.Tensor | None = None
    sparse: SparseCore | None = None
    physical_token: int | None = None

    @property
    def arity(self) -> int:
        return len(self.children)


@dataclass(frozen=True)
class CompiledBoundaryGraph:
    root: MetricNode
    block_root: MetricNode
    inputs: tuple[MetricNode, ...]
    block: ChiTransformerBlock
    token_index: int
    mask: torch.Tensor
    final_norm: RationalNorm | None
    action_head: nn.Linear | None
    stack: str
    block_index: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)
    return float((torch.linalg.vector_norm(actual - expected) / scale).item())


def _scale_matrix(matrix: torch.Tensor, inherited: int = 0) -> ScaledMatrix:
    symmetric = 0.5 * (matrix + matrix.T)
    if not bool(torch.isfinite(symmetric).all()):
        raise RuntimeError("metric contraction produced a nonfinite matrix")
    scale = symmetric.abs().amax()
    if float(scale.item()) == 0.0:
        return ScaledMatrix(symmetric, inherited)
    _, exponent = torch.frexp(scale)
    shift = int(exponent.item())
    return ScaledMatrix(torch.ldexp(symmetric, -shift), inherited + shift)


def _add_scaled(left: ScaledMatrix | None, right: ScaledMatrix) -> ScaledMatrix:
    if left is None:
        return right
    if left.mantissa.shape != right.mantissa.shape:
        raise RuntimeError("cannot add scaled metrics with different shapes")
    exponent = max(left.binary_exponent, right.binary_exponent)
    left_shift = torch.tensor(
        left.binary_exponent - exponent,
        dtype=torch.int64,
        device=left.mantissa.device,
    )
    right_shift = torch.tensor(
        right.binary_exponent - exponent,
        dtype=torch.int64,
        device=right.mantissa.device,
    )
    aligned = torch.ldexp(left.mantissa, left_shift) + torch.ldexp(
        right.mantissa, right_shift
    )
    return _scale_matrix(aligned, exponent)


def _scaled_matrix_error(actual: ScaledMatrix, expected: ScaledMatrix) -> float:
    exponent = max(actual.binary_exponent, expected.binary_exponent)
    a = torch.ldexp(
        actual.mantissa,
        torch.tensor(
            actual.binary_exponent - exponent,
            dtype=torch.int64,
            device=actual.mantissa.device,
        ),
    )
    b = torch.ldexp(
        expected.mantissa,
        torch.tensor(
            expected.binary_exponent - exponent,
            dtype=torch.int64,
            device=expected.mantissa.device,
        ),
    )
    return _relative_error(a, b)


def _validate_sparse(core: SparseCore) -> None:
    terms = core.term_count
    if core.arity < 1 or len(core.input_indices) != core.arity:
        raise RuntimeError("sparse core arity is invalid")
    if core.output_indices.shape != (terms,):
        raise RuntimeError("sparse core output indices have the wrong shape")
    if any(indices.shape != (terms,) for indices in core.input_indices):
        raise RuntimeError("sparse core input indices have the wrong shape")
    if core.output_indices.dtype != torch.long or any(
        indices.dtype != torch.long for indices in core.input_indices
    ):
        raise RuntimeError("sparse core indices must be int64")
    if not bool(torch.isfinite(core.coefficients).all()):
        raise RuntimeError("sparse core coefficients are nonfinite")
    if int(core.output_indices.min()) < 0 or int(core.output_indices.max()) >= core.output_dimension:
        raise RuntimeError("sparse core output index is out of range")
    for dimension, indices in zip(core.input_dimensions, core.input_indices):
        if int(indices.min()) < 0 or int(indices.max()) >= dimension:
            raise RuntimeError("sparse core input index is out of range")


def _sparse_core(
    output_dimension: int,
    input_dimensions: tuple[int, ...],
    terms: Iterable[tuple[int, tuple[int, ...], torch.Tensor | float]],
    like: torch.Tensor,
) -> SparseCore:
    material = tuple(terms)
    if not material:
        raise RuntimeError("a sparse primitive cannot have zero terms")
    outputs = torch.tensor(
        [term[0] for term in material], dtype=torch.long, device=like.device
    )
    inputs = tuple(
        torch.tensor(
            [term[1][role] for term in material],
            dtype=torch.long,
            device=like.device,
        )
        for role in range(len(input_dimensions))
    )
    coefficients = torch.stack(
        [
            torch.as_tensor(term[2], dtype=like.dtype, device=like.device).reshape(())
            for term in material
        ]
    )
    result = SparseCore(
        output_dimension,
        input_dimensions,
        outputs,
        inputs,
        coefficients,
    )
    _validate_sparse(result)
    return result


def _input_node(token: int, dimension: int) -> MetricNode:
    return MetricNode(f"input.token{token}", dimension + 1, physical_token=token)


def _affine_node(
    child: MetricNode,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    label: str,
) -> MetricNode:
    if weight.ndim != 2 or weight.shape[1] + 1 != child.output_dimension:
        raise RuntimeError(f"affine shape mismatch at {label}")
    resolved = weight.new_zeros(weight.shape[0]) if bias is None else bias
    if resolved.shape != (weight.shape[0],):
        raise RuntimeError(f"affine bias mismatch at {label}")
    matrix = weight.new_zeros(weight.shape[0] + 1, weight.shape[1] + 1)
    matrix[:-1, :-1] = weight
    matrix[:-1, -1] = resolved
    matrix[-1, -1] = 1.0
    return MetricNode(label, matrix.shape[0], (child,), affine=matrix)


def _scale_vector_node(child: MetricNode, scale: torch.Tensor | float, label: str) -> MetricNode:
    dimension = child.output_dimension - 1
    scalar = torch.as_tensor(scale, dtype=child_dtype(child), device=child_device(child))
    weight = torch.eye(dimension, dtype=scalar.dtype, device=scalar.device) * scalar
    return _affine_node(child, weight, None, label)


def child_dtype(node: MetricNode) -> torch.dtype:
    if node.affine is not None:
        return node.affine.dtype
    if node.sparse is not None:
        return node.sparse.coefficients.dtype
    for child in node.children:
        return child_dtype(child)
    return DTYPE


def child_device(node: MetricNode) -> torch.device:
    if node.affine is not None:
        return node.affine.device
    if node.sparse is not None:
        return node.sparse.coefficients.device
    for child in node.children:
        return child_device(child)
    return torch.device("cpu")


def _add_vectors_node(left: MetricNode, right: MetricNode, label: str) -> MetricNode:
    if left.output_dimension != right.output_dimension:
        raise RuntimeError("projective vector add requires equal dimensions")
    size = left.output_dimension
    denominator = size - 1
    terms: list[tuple[int, tuple[int, ...], float]] = []
    for index in range(denominator):
        terms.append((index, (index, denominator), 1.0))
        terms.append((index, (denominator, index), 1.0))
    terms.append((denominator, (denominator, denominator), 1.0))
    sparse = _sparse_core(size, (size, size), terms, first_tensor(left, right))
    return MetricNode(label, size, (left, right), sparse=sparse)


def _dot_node(left: MetricNode, right: MetricNode, label: str) -> MetricNode:
    if left.output_dimension != right.output_dimension:
        raise RuntimeError("projective dot requires equal dimensions")
    denominator = left.output_dimension - 1
    terms = [(0, (index, index), 1.0) for index in range(denominator)]
    terms.append((1, (denominator, denominator), 1.0))
    sparse = _sparse_core(2, (denominator + 1, denominator + 1), terms, first_tensor(left, right))
    return MetricNode(label, 2, (left, right), sparse=sparse)


def _multiply_scalars_node(left: MetricNode, right: MetricNode, label: str) -> MetricNode:
    if left.output_dimension != 2 or right.output_dimension != 2:
        raise RuntimeError("projective scalar product requires pair inputs")
    sparse = _sparse_core(
        2,
        (2, 2),
        ((0, (0, 0), 1.0), (1, (1, 1), 1.0)),
        first_tensor(left, right),
    )
    return MetricNode(label, 2, (left, right), sparse=sparse)


def _scale_vector_by_scalar_node(
    scalar: MetricNode,
    vector: MetricNode,
    label: str,
) -> MetricNode:
    if scalar.output_dimension != 2:
        raise RuntimeError("projective scalar-vector product requires a scalar pair")
    size = vector.output_dimension
    denominator = size - 1
    terms = [(index, (0, index), 1.0) for index in range(denominator)]
    terms.append((denominator, (1, denominator), 1.0))
    sparse = _sparse_core(2 if size == 2 else size, (2, size), terms, first_tensor(scalar, vector))
    return MetricNode(label, size, (scalar, vector), sparse=sparse)


def _hadamard_node(left: MetricNode, right: MetricNode, label: str) -> MetricNode:
    if left.output_dimension != right.output_dimension:
        raise RuntimeError("projective Hadamard product requires equal dimensions")
    size = left.output_dimension
    terms = [(index, (index, index), 1.0) for index in range(size)]
    sparse = _sparse_core(size, (size, size), terms, first_tensor(left, right))
    return MetricNode(label, size, (left, right), sparse=sparse)


def first_tensor(*nodes: MetricNode) -> torch.Tensor:
    for node in nodes:
        if node.affine is not None:
            return node.affine
        if node.sparse is not None:
            return node.sparse.coefficients
        for child in node.children:
            try:
                return first_tensor(child)
            except RuntimeError:
                pass
    raise RuntimeError("could not infer a tensor device for primitive construction")


def _pade_norm_node(value: MetricNode, module: RationalNorm, label: str) -> MetricNode:
    if module.variant != "pade" or module.deg != 2 or not bool(module.initialized):
        raise RuntimeError(f"{label} is not an initialized degree-two Padé norm")
    dimension = value.output_dimension - 1
    like = first_tensor(value)
    running = module.running_ms.to(dtype=like.dtype, device=like.device).clamp_min(1e-12)
    mean_terms: list[tuple[int, tuple[int, ...], torch.Tensor | float]] = [
        (0, (index, index), 1.0 / dimension) for index in range(dimension)
    ]
    mean_terms.extend(
        (
            (0, (dimension, dimension), float(module.eps)),
            (1, (dimension, dimension), running),
        )
    )
    mean_core = _sparse_core(
        2,
        (dimension + 1, dimension + 1),
        mean_terms,
        like,
    )
    mean = MetricNode(
        f"{label}.mean_square_homogeneous",
        2,
        (value, value),
        sparse=mean_core,
    )
    pade_terms: list[tuple[int, tuple[int, ...], torch.Tensor]] = []
    for output, coefficients in enumerate((module.pa, module.pb)):
        resolved = coefficients.to(dtype=like.dtype, device=like.device)
        pade_terms.extend(
            (
                (output, (1, 1), resolved[0]),
                (output, (0, 1), resolved[1] / 2.0),
                (output, (1, 0), resolved[1] / 2.0),
                (output, (0, 0), resolved[2]),
            )
        )
    pade_core = _sparse_core(2, (2, 2), pade_terms, like)
    pade = MetricNode(f"{label}.pade_homogeneous", 2, (mean, mean), sparse=pade_core)
    output_terms = [
        (index, (index, 0), torch.rsqrt(running)) for index in range(dimension)
    ]
    output_terms.append((dimension, (dimension, 1), 1.0))
    output_core = _sparse_core(
        dimension + 1,
        (dimension + 1, 2),
        output_terms,
        like,
    )
    return MetricNode(
        f"{label}.projective_output",
        dimension + 1,
        (value, pade),
        sparse=output_core,
    )


def _concat_heads_node(heads: tuple[MetricNode, ...], label: str) -> MetricNode:
    if not heads:
        raise RuntimeError("attention must contain at least one head")
    sizes = {head.output_dimension for head in heads}
    if len(sizes) != 1:
        raise RuntimeError("attention head projective dimensions differ")
    head_size = heads[0].output_dimension
    head_dimension = head_size - 1
    output_dimension = len(heads) * head_dimension + 1
    denominator_indices = [head_dimension] * len(heads)
    terms: list[tuple[int, tuple[int, ...], float]] = []
    for head_index in range(len(heads)):
        for feature in range(head_dimension):
            inputs = denominator_indices.copy()
            inputs[head_index] = feature
            terms.append((head_index * head_dimension + feature, tuple(inputs), 1.0))
    terms.append((output_dimension - 1, tuple(denominator_indices), 1.0))
    sparse = _sparse_core(
        output_dimension,
        tuple(head_size for _ in heads),
        terms,
        first_tensor(*heads),
    )
    return MetricNode(label, output_dimension, heads, sparse=sparse)


def _compile_selected_block(
    block: ChiTransformerBlock,
    token_count: int,
    token_index: int,
    mask: torch.Tensor,
    *,
    stack: str,
    block_index: int,
    final_norm: RationalNorm | None = None,
    action_head: nn.Linear | None = None,
) -> CompiledBoundaryGraph:
    if block.training or not block.residual:
        raise RuntimeError("production boundary compiler requires an eval residual block")
    if block.attn.qk_norm != "rational":
        raise RuntimeError("production boundary compiler requires rational Q/K norms")
    dimension = block.attn.dim
    heads = block.attn.n_heads
    head_dimension = block.attn.head_dim
    if not 0 <= token_index < token_count:
        raise RuntimeError("selected token is out of range")
    if mask.shape != (token_count, token_count):
        raise RuntimeError("block mask has the wrong shape")
    visible_sources = tuple(
        source for source in range(token_count) if float(mask[token_index, source].item()) != 0.0
    )
    if not visible_sources:
        raise RuntimeError("selected output token has no visible attention source")

    inputs = tuple(_input_node(token, dimension) for token in range(token_count))
    normalized = tuple(
        _pade_norm_node(value, block.rbn_attn, f"pre_attention_norm.token{token}")
        for token, value in enumerate(inputs)
    )
    row_count = mask[token_index].sum().clamp_min(1.0)
    row_scale = row_count.rsqrt() if block.attn.row_scale == "invsqrt" else row_count.reciprocal()
    fixed_scale = row_scale / float(block.attn._score_denom)
    head_outputs: list[MetricNode] = []
    for head in range(heads):
        start = head * head_dimension
        stop = start + head_dimension

        def projection(
            linear: nn.Linear,
            token: int,
            name: str,
        ) -> MetricNode:
            bias = None if linear.bias is None else linear.bias[start:stop]
            return _affine_node(
                normalized[token],
                linear.weight[start:stop],
                bias,
                f"attention.head{head}.{name}.token{token}",
            )

        q1 = _pade_norm_node(
            projection(block.attn.wq1, token_index, "q1_affine"),
            block.attn.rn_q1,
            f"attention.head{head}.q1_norm.token{token_index}",
        )
        q2 = _pade_norm_node(
            projection(block.attn.wq2, token_index, "q2_affine"),
            block.attn.rn_q2,
            f"attention.head{head}.q2_norm.token{token_index}",
        )
        routes: list[MetricNode] = []
        for source in visible_sources:
            k1 = _pade_norm_node(
                projection(block.attn.wk1, source, "k1_affine"),
                block.attn.rn_k1,
                f"attention.head{head}.k1_norm.token{source}",
            )
            k2 = _pade_norm_node(
                projection(block.attn.wk2, source, "k2_affine"),
                block.attn.rn_k2,
                f"attention.head{head}.k2_norm.token{source}",
            )
            value = projection(block.attn.wv, source, "value_affine")
            score1 = _dot_node(q1, k1, f"attention.head{head}.route{source}.score1")
            score2 = _dot_node(q2, k2, f"attention.head{head}.route{source}.score2")
            score = _multiply_scalars_node(
                score1,
                score2,
                f"attention.head{head}.route{source}.score_product",
            )
            route = _scale_vector_by_scalar_node(
                score,
                value,
                f"attention.head{head}.route{source}.score_times_value",
            )
            routes.append(
                _scale_vector_node(
                    route,
                    fixed_scale,
                    f"attention.head{head}.route{source}.fixed_scale",
                )
            )
        head_output = routes[0]
        for route_index, route in enumerate(routes[1:], start=1):
            head_output = _add_vectors_node(
                head_output,
                route,
                f"attention.head{head}.route_sum{route_index}",
            )
        head_outputs.append(head_output)
    concatenated = _concat_heads_node(tuple(head_outputs), "attention.concatenate_heads")
    attention = _affine_node(
        concatenated,
        block.attn.wo.weight,
        block.attn.wo.bias,
        "attention.output_affine",
    )
    attention = _scale_vector_node(attention, block.attn_gain, "attention.residual_gain")
    after_attention = _add_vectors_node(
        inputs[token_index], attention, "attention.residual_add"
    )
    ffn_input = _pade_norm_node(after_attention, block.rbn_ffn, "ffn_pre_norm")
    left = _affine_node(ffn_input, block.ffn.left.weight, block.ffn.left.bias, "ffn.left")
    right = _affine_node(ffn_input, block.ffn.right.weight, block.ffn.right.bias, "ffn.right")
    product = _hadamard_node(left, right, "ffn.hadamard")
    ffn = _affine_node(product, block.ffn.down.weight, block.ffn.down.bias, "ffn.down")
    ffn = _scale_vector_node(ffn, block.ffn_gain, "ffn.residual_gain")
    block_root = _add_vectors_node(after_attention, ffn, "ffn.residual_add")
    root = block_root
    if final_norm is not None:
        root = _pade_norm_node(root, final_norm, "final_norm")
    if action_head is not None:
        root = _affine_node(root, action_head.weight, action_head.bias, "action_head")
    return CompiledBoundaryGraph(
        root,
        block_root,
        inputs,
        block,
        token_index,
        mask,
        final_norm,
        action_head,
        stack,
        block_index,
    )


def _walk_unique(root: MetricNode) -> tuple[MetricNode, ...]:
    result: list[MetricNode] = []
    state: dict[int, int] = {}

    def visit(node: MetricNode) -> None:
        marker = state.get(id(node), 0)
        if marker == 1:
            raise RuntimeError("compiled metric graph contains a cycle")
        if marker == 2:
            return
        state[id(node)] = 1
        result.append(node)
        for child in node.children:
            visit(child)
        state[id(node)] = 2

    visit(root)
    return tuple(result)


def _topological_root_first(root: MetricNode) -> tuple[MetricNode, ...]:
    nodes = _walk_unique(root)
    incoming = {id(node): 0 for node in nodes}
    for node in nodes:
        for child in node.children:
            incoming[id(child)] += 1
    ready = deque(node for node in nodes if incoming[id(node)] == 0)
    ordered: list[MetricNode] = []
    while ready:
        node = ready.popleft()
        ordered.append(node)
        for child in node.children:
            incoming[id(child)] -= 1
            if incoming[id(child)] == 0:
                ready.append(child)
    if len(ordered) != len(nodes) or ordered[0] is not root:
        raise RuntimeError("compiled metric graph topology is invalid")
    return tuple(ordered)


def _sparse_prefix(core: SparseCore, metrics: tuple[torch.Tensor, ...]) -> torch.Tensor:
    coefficients = core.coefficients
    pair = coefficients[:, None] * coefficients[None, :]
    for indices, metric in zip(core.input_indices, metrics):
        pair = pair * metric[indices[:, None], indices[None, :]]
    flat_indices = (
        core.output_indices[:, None] * core.output_dimension
        + core.output_indices[None, :]
    ).reshape(-1)
    result = coefficients.new_zeros(core.output_dimension * core.output_dimension)
    result.index_add_(0, flat_indices, pair.reshape(-1))
    return result.reshape(core.output_dimension, core.output_dimension)


def _sparse_role_environment(
    core: SparseCore,
    downstream: torch.Tensor,
    input_metrics: tuple[torch.Tensor, ...],
    role: int,
) -> torch.Tensor:
    coefficients = core.coefficients
    pair = coefficients[:, None] * coefficients[None, :]
    pair = pair * downstream[
        core.output_indices[:, None], core.output_indices[None, :]
    ]
    for sibling, (indices, metric) in enumerate(zip(core.input_indices, input_metrics)):
        if sibling != role:
            pair = pair * metric[indices[:, None], indices[None, :]]
    dimension = core.input_dimensions[role]
    indices = core.input_indices[role]
    flat_indices = (indices[:, None] * dimension + indices[None, :]).reshape(-1)
    result = coefficients.new_zeros(dimension * dimension)
    result.index_add_(0, flat_indices, pair.reshape(-1))
    return result.reshape(dimension, dimension)


@torch.no_grad()
def form_prefix_metrics(
    root: MetricNode,
) -> tuple[dict[int, ScaledMatrix], dict[str, float | int]]:
    metrics: dict[int, ScaledMatrix] = {}
    min_exponent = 0
    max_exponent = 0
    maximum_pair_elements = 0

    def visit(node: MetricNode) -> ScaledMatrix:
        nonlocal min_exponent, max_exponent, maximum_pair_elements
        cached = metrics.get(id(node))
        if cached is not None:
            return cached
        if node.physical_token is not None:
            raw = torch.eye(
                node.output_dimension,
                dtype=first_tensor_from_root(root).dtype,
                device=first_tensor_from_root(root).device,
            )
            result = ScaledMatrix(raw, 0)
        elif node.affine is not None:
            child = visit(node.children[0])
            result = _scale_matrix(
                node.affine @ child.mantissa @ node.affine.T,
                child.binary_exponent,
            )
        elif node.sparse is not None:
            children = tuple(visit(child) for child in node.children)
            maximum_pair_elements = max(
                maximum_pair_elements, node.sparse.term_count**2
            )
            raw = _sparse_prefix(
                node.sparse,
                tuple(child.mantissa for child in children),
            )
            result = _scale_matrix(
                raw,
                sum(child.binary_exponent for child in children),
            )
        else:
            raise RuntimeError(f"node {node.label} has no primitive")
        metrics[id(node)] = result
        min_exponent = min(min_exponent, result.binary_exponent)
        max_exponent = max(max_exponent, result.binary_exponent)
        return result

    visit(root)
    return metrics, {
        "minimum_prefix_binary_exponent": min_exponent,
        "maximum_prefix_binary_exponent": max_exponent,
        "maximum_sparse_term_pair_elements": maximum_pair_elements,
    }


def first_tensor_from_root(root: MetricNode) -> torch.Tensor:
    for node in _walk_unique(root):
        if node.affine is not None:
            return node.affine
        if node.sparse is not None:
            return node.sparse.coefficients
    raise RuntimeError("metric graph has no tensor-valued primitive")


@torch.no_grad()
def form_boundary_environments(
    root: MetricNode,
    prefix: dict[int, ScaledMatrix],
    *,
    retain_all: bool = False,
) -> tuple[dict[int, ScaledMatrix], dict[str, float | int]]:
    like = first_tensor_from_root(root)
    pending: dict[int, ScaledMatrix] = {
        id(root): ScaledMatrix(
            torch.eye(root.output_dimension, dtype=like.dtype, device=like.device),
            0,
        )
    }
    retained: dict[int, ScaledMatrix] = {}
    min_exponent = 0
    max_exponent = 0
    messages = 0
    maximum_pair_elements = 0
    for node in _topological_root_first(root):
        downstream = pending.pop(id(node), None)
        if downstream is None:
            raise RuntimeError(f"node {node.label} received no reverse metric")
        if retain_all or node.physical_token is not None:
            retained[id(node)] = downstream
        min_exponent = min(min_exponent, downstream.binary_exponent)
        max_exponent = max(max_exponent, downstream.binary_exponent)
        if node.physical_token is not None:
            continue
        child_metrics = tuple(prefix[id(child)] for child in node.children)
        for role, child in enumerate(node.children):
            if node.affine is not None:
                if role != 0:
                    raise RuntimeError("unary affine node has a nonzero role")
                raw = node.affine.T @ downstream.mantissa @ node.affine
                inherited = downstream.binary_exponent
            elif node.sparse is not None:
                maximum_pair_elements = max(
                    maximum_pair_elements, node.sparse.term_count**2
                )
                raw = _sparse_role_environment(
                    node.sparse,
                    downstream.mantissa,
                    tuple(metric.mantissa for metric in child_metrics),
                    role,
                )
                inherited = downstream.binary_exponent + sum(
                    metric.binary_exponent
                    for sibling, metric in enumerate(child_metrics)
                    if sibling != role
                )
            else:
                raise RuntimeError(f"node {node.label} has no reverse primitive")
            message = _scale_matrix(raw, inherited)
            pending[id(child)] = _add_scaled(pending.get(id(child)), message)
            messages += 1
    if pending:
        raise RuntimeError("reverse metric traversal left unprocessed messages")
    return retained, {
        "minimum_environment_binary_exponent": min_exponent,
        "maximum_environment_binary_exponent": max_exponent,
        "reverse_edge_message_count": messages,
        "maximum_sparse_term_pair_elements": maximum_pair_elements,
    }


def _scaled_coordinates(value: torch.Tensor, inherited: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = value.abs().amax(dim=1)
    if bool((scale == 0).any()) or not bool(torch.isfinite(scale).all()):
        raise RuntimeError("projective evaluation produced zero/nonfinite coordinates")
    _, exponent = torch.frexp(scale)
    return torch.ldexp(value, -exponent[:, None]), inherited + exponent


@torch.no_grad()
def evaluate_graph(root: MetricNode, raw_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    memo: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def visit(node: MetricNode) -> tuple[torch.Tensor, torch.Tensor]:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        if node.physical_token is not None:
            value = torch.cat(
                (
                    raw_input[:, node.physical_token],
                    raw_input.new_ones(raw_input.shape[0], 1),
                ),
                dim=1,
            )
            inherited = torch.zeros(
                raw_input.shape[0], dtype=torch.int64, device=raw_input.device
            )
        elif node.affine is not None:
            child, inherited = visit(node.children[0])
            value = child @ node.affine.T
        elif node.sparse is not None:
            children = tuple(visit(child) for child in node.children)
            inherited = torch.stack(tuple(child[1] for child in children)).sum(0)
            term_values = node.sparse.coefficients[None, :].expand(raw_input.shape[0], -1)
            for indices, (child, _) in zip(node.sparse.input_indices, children):
                term_values = term_values * child[:, indices]
            value = raw_input.new_zeros(raw_input.shape[0], node.output_dimension)
            value.index_add_(1, node.sparse.output_indices, term_values)
        else:
            raise RuntimeError(f"node {node.label} has no evaluation primitive")
        result = _scaled_coordinates(value, inherited)
        memo[id(node)] = result
        return result

    return visit(root)


def quotient_vector(evaluation: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    pair = evaluation[0]
    denominator = pair[:, -1:]
    relative = denominator.abs() / pair.abs().amax(dim=1, keepdim=True).clamp_min(
        torch.finfo(pair.dtype).tiny
    )
    if float(relative.min().item()) <= 100.0 * torch.finfo(pair.dtype).eps:
        raise RuntimeError("projective output denominator is too small")
    return pair[:, :-1] / denominator


def _materialize_core(node: MetricNode, maximum_elements: int) -> torch.Tensor:
    if node.physical_token is not None:
        return torch.eye(
            node.output_dimension,
            dtype=first_tensor(node).dtype,
            device=first_tensor(node).device,
        )
    if node.affine is not None:
        return node.affine
    if node.sparse is None:
        raise RuntimeError("cannot materialize an untyped node")
    shape = (node.output_dimension, *node.sparse.input_dimensions)
    elements = math.prod(shape)
    if elements > maximum_elements:
        raise RuntimeError(
            f"tiny dense control attempted {elements} elements above cap {maximum_elements}"
        )
    core = node.sparse.coefficients.new_zeros(shape)
    core.index_put_(
        (node.sparse.output_indices, *node.sparse.input_indices),
        node.sparse.coefficients,
        accumulate=True,
    )
    return core


@torch.no_grad()
def explicit_dense_clone_control(
    root: MetricNode,
    *,
    maximum_elements: int = 100_000,
) -> dict[str, object]:
    factorized_prefix, _ = form_prefix_metrics(root)
    dense_prefix: dict[int, ScaledMatrix] = {}

    def prefix(node: MetricNode) -> ScaledMatrix:
        cached = dense_prefix.get(id(node))
        if cached is not None:
            return cached
        if node.physical_token is not None:
            result = ScaledMatrix(_materialize_core(node, maximum_elements), 0)
        else:
            children = tuple(prefix(child) for child in node.children)
            raw = forward_metric(
                _materialize_core(node, maximum_elements),
                tuple(child.mantissa for child in children),
            )
            result = _scale_matrix(
                raw, sum(child.binary_exponent for child in children)
            )
        dense_prefix[id(node)] = result
        return result

    prefix(root)
    prefix_error = max(
        _scaled_matrix_error(factorized_prefix[identity], expected)
        for identity, expected in dense_prefix.items()
    )
    like = first_tensor_from_root(root)
    clone_messages: dict[int, list[ScaledMatrix]] = defaultdict(list)
    clone_occurrences = 0

    def descend(node: MetricNode, downstream: ScaledMatrix) -> None:
        nonlocal clone_occurrences
        clone_occurrences += 1
        clone_messages[id(node)].append(downstream)
        if node.physical_token is not None:
            return
        child_metrics = tuple(dense_prefix[id(child)] for child in node.children)
        dense_core = _materialize_core(node, maximum_elements)
        for role, child in enumerate(node.children):
            raw = role_environment(
                dense_core,
                downstream.mantissa,
                tuple(metric.mantissa for metric in child_metrics),
                role,
            )
            inherited = downstream.binary_exponent + sum(
                metric.binary_exponent
                for sibling, metric in enumerate(child_metrics)
                if sibling != role
            )
            descend(child, _scale_matrix(raw, inherited))

    descend(
        root,
        ScaledMatrix(
            torch.eye(root.output_dimension, dtype=like.dtype, device=like.device),
            0,
        ),
    )
    clone_aggregate: dict[int, ScaledMatrix] = {}
    for identity, messages in clone_messages.items():
        total: ScaledMatrix | None = None
        for message in messages:
            total = _add_scaled(total, message)
        if total is None:
            raise RuntimeError("explicit clone aggregation lost a message")
        clone_aggregate[identity] = total
    reverse, _ = form_boundary_environments(root, factorized_prefix, retain_all=True)
    environment_error = max(
        _scaled_matrix_error(reverse[identity], expected)
        for identity, expected in clone_aggregate.items()
    )
    return {
        "maximum_factorized_vs_dense_prefix_relative_error": prefix_error,
        "maximum_reverse_vs_explicit_clone_relative_error": environment_error,
        "explicit_clone_occurrence_count": clone_occurrences,
        "unique_node_count": len(_walk_unique(root)),
    }


def _initialize_tiny_norms(block: ChiTransformerBlock, final_norm: RationalNorm) -> None:
    with torch.no_grad():
        for index, module in enumerate(
            module
            for owner in (block, final_norm)
            for module in owner.modules()
            if isinstance(module, RationalNorm)
        ):
            module.running_ms.fill_(0.9 + 0.07 * index)
            module.initialized.fill_(True)
            module.frozen = True


@torch.no_grad()
def run_tiny_control() -> dict[str, object]:
    torch.manual_seed(71)
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
    final_norm = RationalNorm(variant="pade").double().eval()
    action_head = nn.Linear(1, 1, bias=True).double().eval()
    generator = torch.Generator().manual_seed(7100)
    for module in block.modules():
        if isinstance(module, nn.Linear):
            module.weight.copy_(
                0.05 + 0.15 * torch.rand(module.weight.shape, generator=generator, dtype=DTYPE)
            )
            if module.bias is not None:
                module.bias.copy_(
                    -0.08
                    + 0.16 * torch.rand(module.bias.shape, generator=generator, dtype=DTYPE)
                )
    action_head.weight.fill_(0.27)
    action_head.bias.fill_(-0.11)
    _initialize_tiny_norms(block, final_norm)
    mask = causal_mask(2, dtype=DTYPE)
    graph = _compile_selected_block(
        block,
        2,
        1,
        mask,
        stack="tiny",
        block_index=0,
        final_norm=final_norm,
        action_head=action_head,
    )
    raw = torch.tensor(
        [[[-0.25], [0.31]], [[0.17], [-0.09]], [[0.0], [0.0]]], dtype=DTYPE
    )
    source_block = block(raw, mask=mask, method="explicit")[:, 1]
    source_tail = action_head(final_norm(source_block)).squeeze(-1)
    graph_block = quotient_vector(evaluate_graph(graph.block_root, raw))
    graph_tail = quotient_vector(evaluate_graph(graph.root, raw)).squeeze(-1)
    dense = explicit_dense_clone_control(graph.root)
    return {
        "block_source_replay_relative_error": _relative_error(graph_block, source_block),
        "tail_source_replay_relative_error": _relative_error(graph_tail, source_tail),
        **dense,
    }


def _state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def _graph_statistics(root: MetricNode) -> dict[str, int]:
    nodes = _walk_unique(root)
    incoming = {id(node): 0 for node in nodes}
    maximum_tensor = 0
    sparse_terms = 0
    dense_core_elements_avoided = 0
    for node in nodes:
        for child in node.children:
            incoming[id(child)] += 1
        if node.affine is not None:
            maximum_tensor = max(maximum_tensor, node.affine.numel())
        if node.sparse is not None:
            sparse_terms += node.sparse.term_count
            maximum_tensor = max(
                maximum_tensor,
                node.sparse.coefficients.numel(),
                node.sparse.output_indices.numel(),
                *(indices.numel() for indices in node.sparse.input_indices),
            )
            dense_core_elements_avoided += node.output_dimension * math.prod(
                node.sparse.input_dimensions
            )
    metric_elements = sum(node.output_dimension**2 for node in nodes)
    return {
        "unique_node_count": len(nodes),
        "edge_occurrence_count": sum(len(node.children) for node in nodes),
        "shared_node_count": sum(value > 1 for value in incoming.values()),
        "maximum_local_bond_dimension": max(node.output_dimension for node in nodes),
        "sparse_core_term_count": sparse_terms,
        "maximum_materialized_tensor_elements": maximum_tensor,
        "dense_primitive_core_elements_avoided": dense_core_elements_avoided,
        "prefix_metric_elements": metric_elements,
        "prefix_metric_bytes_float64": metric_elements * 8,
    }


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


@torch.no_grad()
def run_production(args: argparse.Namespace) -> dict[str, object]:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for a production case")
    checkpoint_sha = _sha256(args.checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise RuntimeError("checkpoint SHA-256 differs from the pinned capable checkpoint")
    device = _resolve_device(args.device)
    load_started = time.perf_counter()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vocabulary = int(state["tok_emb.weight"].shape[0])
    model = ChiVLA(production_config(vocabulary)).to(device=device, dtype=DTYPE).eval()
    model.load_state_dict(state, strict=True)
    del state
    before_digest = _state_digest(model)
    load_seconds = time.perf_counter() - load_started
    if args.stack == "vision":
        blocks = model.vision.blocks.blocks
        token_count = 64
        default_index = 63
        mask = torch.ones(token_count, token_count, dtype=DTYPE, device=device)
    else:
        blocks = model.backbone.blocks
        token_count = 107
        default_index = 106
        mask = causal_mask(token_count, dtype=DTYPE, device=device)
    if not 0 <= args.block_index < len(blocks):
        raise ValueError("--block-index is outside the selected production stack")
    token_index = default_index if args.token_index < 0 else args.token_index
    include_tail = bool(args.include_final_tail)
    if include_tail and not (args.stack == "joint" and args.block_index == 7 and token_index >= 99):
        raise ValueError(
            "--include-final-tail is valid only for an action-query token of joint block 7"
        )
    block = blocks[args.block_index]
    compile_started = time.perf_counter()
    graph = _compile_selected_block(
        block,
        token_count,
        token_index,
        mask,
        stack=args.stack,
        block_index=args.block_index,
        final_norm=model.norm_out if include_tail else None,
        action_head=model.action_head if include_tail else None,
    )
    compile_seconds = time.perf_counter() - compile_started
    shape = _graph_statistics(graph.root)
    if shape["unique_node_count"] > args.maximum_nodes:
        raise RuntimeError("compiled DAG exceeds --maximum-nodes")
    if shape["maximum_local_bond_dimension"] > args.maximum_bond_dimension:
        raise RuntimeError("compiled DAG exceeds --maximum-bond-dimension")
    if shape["prefix_metric_bytes_float64"] > args.maximum_prefix_metric_bytes:
        raise RuntimeError("prefix metric estimate exceeds --maximum-prefix-metric-bytes")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    raw = torch.randn(
        args.batch_size,
        token_count,
        block.attn.dim,
        generator=generator,
        dtype=DTYPE,
    ).to(device)
    replay_started = time.perf_counter()
    source_block = block(raw, mask=None if args.stack == "vision" else mask, method="explicit")[
        :, token_index
    ]
    graph_block = quotient_vector(evaluate_graph(graph.block_root, raw))
    if include_tail:
        source_root = model.action_head(model.norm_out(source_block))
    else:
        source_root = source_block
    graph_root = quotient_vector(evaluate_graph(graph.root, raw))
    replay_seconds = time.perf_counter() - replay_started

    prefix_started = time.perf_counter()
    prefix, prefix_stats = form_prefix_metrics(graph.root)
    prefix_seconds = time.perf_counter() - prefix_started
    if int(prefix_stats["maximum_sparse_term_pair_elements"]) > args.maximum_term_pair_elements:
        raise RuntimeError("sparse prefix kernel exceeds --maximum-term-pair-elements")
    environment_started = time.perf_counter()
    boundary, environment_stats = form_boundary_environments(graph.root, prefix)
    environment_seconds = time.perf_counter() - environment_started
    if int(environment_stats["maximum_sparse_term_pair_elements"]) > args.maximum_term_pair_elements:
        raise RuntimeError("sparse environment kernel exceeds --maximum-term-pair-elements")
    missing = [node.label for node in graph.inputs if id(node) not in boundary]
    if missing:
        raise RuntimeError(f"boundary traversal missed physical inputs: {missing[:3]}")
    boundary_rows = []
    for node in graph.inputs:
        message = boundary[id(node)]
        boundary_rows.append(
            {
                "token": node.physical_token,
                "binary_exponent": message.binary_exponent,
                "mantissa_frobenius_norm": float(
                    torch.linalg.matrix_norm(message.mantissa).item()
                ),
                "mantissa_trace": float(torch.trace(message.mantissa).item()),
                "finite": bool(torch.isfinite(message.mantissa).all()),
                "symmetric_relative_error": _relative_error(
                    message.mantissa,
                    0.5 * (message.mantissa + message.mantissa.T),
                ),
            }
        )
    after_digest = _state_digest(model)
    replay_error = _relative_error(graph_root, source_root)
    block_replay_error = _relative_error(graph_block, source_block)
    gates = {
        "pinned_capable_checkpoint": checkpoint_sha == args.expected_checkpoint_sha256,
        "one_actual_production_block": graph.block is block,
        "source_block_replay": block_replay_error < 3e-9,
        "selected_root_replay": replay_error < 3e-9,
        "all_input_boundary_metrics_formed": len(boundary_rows) == token_count,
        "all_input_boundary_metrics_finite": all(row["finite"] for row in boundary_rows),
        "all_input_boundary_metrics_symmetric": max(
            row["symmetric_relative_error"] for row in boundary_rows
        )
        < 1e-12,
        "source_weights_unchanged": before_digest == after_digest,
        "no_dense_block_core": shape["maximum_materialized_tensor_elements"]
        < args.maximum_term_pair_elements,
        "caps_respected": shape["unique_node_count"] <= args.maximum_nodes
        and shape["maximum_local_bond_dimension"] <= args.maximum_bond_dimension
        and shape["prefix_metric_bytes_float64"] <= args.maximum_prefix_metric_bytes,
    }
    return {
        "schema": "xvla-production-projective-boundary-metric-v1",
        "claim_boundary": CLAIM_BOUNDARY,
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "case": {
            "stack": args.stack,
            "block_index": args.block_index,
            "tokens": token_count,
            "width": int(block.attn.dim),
            "heads": int(block.attn.n_heads),
            "head_dimension": int(block.attn.head_dim),
            "ffn_rank": int(block.ffn.rank),
            "selected_token": token_index,
            "visible_sources": int(mask[token_index].sum().item()),
            "include_final_norm_and_action_head": include_tail,
            "depth_traversed": 1,
            "complete_vla_traversal": False,
        },
        "tiny_dense_clone_control": run_tiny_control(),
        "shape": shape,
        "prefix_metric": prefix_stats,
        "environment_metric": environment_stats,
        "boundary": {
            "count": len(boundary_rows),
            "minimum_binary_exponent": min(row["binary_exponent"] for row in boundary_rows),
            "maximum_binary_exponent": max(row["binary_exponent"] for row in boundary_rows),
            "rows": boundary_rows,
        },
        "errors": {
            "block_source_replay_relative": block_replay_error,
            "selected_root_source_replay_relative": replay_error,
        },
        "timings": {
            "load_seconds": load_seconds,
            "compile_seconds": compile_seconds,
            "sampled_replay_seconds": replay_seconds,
            "prefix_metric_seconds": prefix_seconds,
            "boundary_environment_seconds": environment_seconds,
        },
        "peak_rss_mb": _peak_rss_mb(),
        "device": str(device),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256
    )
    parser.add_argument("--stack", choices=("vision", "joint"), default="vision")
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--token-index", type=int, default=-1)
    parser.add_argument("--include-final-tail", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--maximum-nodes", type=int, default=100_000)
    parser.add_argument("--maximum-bond-dimension", type=int, default=2048)
    parser.add_argument("--maximum-prefix-metric-bytes", type=int, default=8_000_000_000)
    parser.add_argument("--maximum-term-pair-elements", type=int, default=2_000_000)
    parser.add_argument("--tiny-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    started = time.perf_counter()
    if args.tiny_only:
        tiny = run_tiny_control()
        gates = {
            "tiny_block_source_replay": tiny["block_source_replay_relative_error"] < 1e-10,
            "tiny_tail_source_replay": tiny["tail_source_replay_relative_error"] < 1e-10,
            "tiny_factorized_prefix_matches_dense": tiny[
                "maximum_factorized_vs_dense_prefix_relative_error"
            ]
            < 1e-10,
            "tiny_reverse_matches_explicit_clone": tiny[
                "maximum_reverse_vs_explicit_clone_relative_error"
            ]
            < 1e-9,
        }
        result: dict[str, object] = {
            "schema": "xvla-production-projective-boundary-metric-v1-tiny-control",
            "claim_boundary": CLAIM_BOUNDARY,
            "tiny_dense_clone_control": tiny,
            "gates": gates,
            "all_gates_pass": all(gates.values()),
        }
    else:
        result = run_production(args)
        tiny = result["tiny_dense_clone_control"]
        gates = result["gates"]
        gates.update(
            {
                "tiny_block_source_replay": tiny["block_source_replay_relative_error"]
                < 1e-10,
                "tiny_tail_source_replay": tiny["tail_source_replay_relative_error"]
                < 1e-10,
                "tiny_factorized_prefix_matches_dense": tiny[
                    "maximum_factorized_vs_dense_prefix_relative_error"
                ]
                < 1e-10,
                "tiny_reverse_matches_explicit_clone": tiny[
                    "maximum_reverse_vs_explicit_clone_relative_error"
                ]
                < 1e-9,
            }
        )
        result["all_gates_pass"] = all(gates.values())
    result["total_elapsed_seconds"] = time.perf_counter() - started
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite existing result {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
        print(
            json.dumps(
                {
                    "schema": result["schema"],
                    "all_gates_pass": result["all_gates_pass"],
                    "gates": result["gates"],
                    "output": str(args.output),
                    "total_elapsed_seconds": result["total_elapsed_seconds"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    if not result["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
