"""Exact primitive-graph Kronecker-closure audit for residual attention stacks.

This is the adversarial counterpart to ``run_kron_closure_audit.py``.  That
runner intentionally fuses affine constants, attention, and the residual route
into one arity-five whole-state core.  Here every operation remains a separate
typed node:

* position is a learned constant leaf followed by a tokenwise add,
* every Q/K/V, output, and FFN affine is a shared linear node plus a separate
  constant-bias leaf and add node,
* Q1, Q2, K1, K2, and V remain five ordered roles,
* branch gains and both residual additions remain explicit nodes, and
* the FFN stays as left/right affine, Hadamard product, and down projection.

The source model is unchanged.  The runner clone-unfolds its arithmetic DAG
conceptually, performs the repository's actual reduced-RQ bottom-up sweep, and
pushes each child factor into its exact typed parent role.  Full canonical
replay is required.  Operator-Schmidt spectra use the token-versus-feature
partition and are reported at several relative tolerances.  Fixed constant
leaves are reported both separately and as part of the all-node closure audit.

This polynomial audit uses the identity normalization setting.  Deployed Padé
normalization is rational and belongs to the separate projective-tree oracle;
it is not silently replaced or treated as a polynomial node here.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from string import ascii_lowercase

import torch
import torch.nn as nn

from xvla.nn.attention import causal_mask
from xvla.nn.block import ChiTransformerBlock
from xvla.train.canonical_odt import reduced_rq_rows


DTYPE = torch.float64


@dataclass(frozen=True)
class Layout:
    tokens: int
    features: int

    @property
    def size(self) -> int:
        return self.tokens * self.features


@dataclass(frozen=True)
class PrimitiveNode:
    label: str
    core: torch.Tensor
    layout: Layout
    children: tuple["PrimitiveNode", ...] = ()
    source_kind: str = "internal"


@dataclass(frozen=True)
class PrimitiveCase:
    name: str
    affine_biases: bool
    position_embedding: bool


CASES = (
    PrimitiveCase("primitive_nobias_nopos", False, False),
    PrimitiveCase("primitive_bias_nopos", True, False),
    PrimitiveCase("primitive_nobias_pos", False, True),
    PrimitiveCase("primitive_bias_pos", True, True),
)


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = expected.norm().clamp_min(torch.finfo(expected.dtype).tiny)
    return float(((actual - expected).norm() / scale).item())


def _tensor_sha256(tensors: tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _operator_spectrum(tensor: torch.Tensor, layouts: tuple[Layout, ...]) -> torch.Tensor:
    if tensor.ndim != len(layouts) or tuple(tensor.shape) != tuple(layout.size for layout in layouts):
        raise ValueError("tensor dimensions do not match token/feature layouts")
    split_shape = tuple(value for layout in layouts for value in (layout.tokens, layout.features))
    split = tensor.reshape(split_shape)
    token_axes = tuple(2 * index for index in range(len(layouts)))
    feature_axes = tuple(axis + 1 for axis in token_axes)
    matrix = split.permute(*(token_axes + feature_axes)).reshape(
        math.prod(layout.tokens for layout in layouts),
        math.prod(layout.features for layout in layouts),
    )
    return torch.linalg.svdvals(matrix)


def _operator_matrix(tensor: torch.Tensor, layouts: tuple[Layout, ...]) -> torch.Tensor:
    """Return the token-versus-feature Van Loan-Pitsianis rearrangement."""

    if tensor.ndim != len(layouts) or tuple(tensor.shape) != tuple(layout.size for layout in layouts):
        raise ValueError("tensor dimensions do not match token/feature layouts")
    split_shape = tuple(value for layout in layouts for value in (layout.tokens, layout.features))
    split = tensor.reshape(split_shape)
    token_axes = tuple(2 * index for index in range(len(layouts)))
    feature_axes = tuple(axis + 1 for axis in token_axes)
    return split.permute(*(token_axes + feature_axes)).reshape(
        math.prod(layout.tokens for layout in layouts),
        math.prod(layout.features for layout in layouts),
    )


def _spectrum_record(values: torch.Tensor, tolerances: tuple[float, ...]) -> dict[str, object]:
    if values.numel() == 0:
        normalized = values
    else:
        normalized = values / values[0].clamp_min(torch.finfo(values.dtype).tiny)
    return {
        "ranks": {
            f"{tolerance:.0e}": int((normalized > tolerance).sum().item())
            for tolerance in tolerances
        },
        "normalized_singular_values": [float(value.item()) for value in normalized[:16]],
    }


def _canonical_layout(original: Layout, dimension: int) -> Layout:
    if dimension == original.size:
        return original
    if dimension % original.tokens == 0:
        return Layout(original.tokens, dimension // original.tokens)
    return Layout(1, dimension)


def _structured_rank_one_rq(
    tensor: torch.Tensor,
    layouts: tuple[Layout, ...],
    *,
    structural_tolerance: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, Layout, dict[str, float]] | None:
    """Factor an exact token/feature product without QR degeneracy mixing.

    A generic QR of a wide block-diagonal matrix can choose arbitrary basis
    vectors after it encounters locally dependent columns.  That numerical
    gauge can destroy exact Kronecker structure even when the operator itself
    has Schmidt rank one.  The Kronecker theorem instead permits independent
    reduced RQ on the token and feature factors.  This helper takes that gauge
    whenever the actual transformed primitive is numerically rank one.
    """

    rearranged = _operator_matrix(tensor, layouts)
    left, values, right_adjoint = torch.linalg.svd(rearranged, full_matrices=False)
    if values.numel() == 0 or float(values[0].item()) == 0.0:
        return None
    if values.numel() > 1 and float((values[1] / values[0]).item()) > structural_tolerance:
        return None
    scale = values[0].sqrt()
    token_tensor = (left[:, 0] * scale).reshape(*(layout.tokens for layout in layouts))
    feature_tensor = (right_adjoint[0] * scale).reshape(
        *(layout.features for layout in layouts)
    )
    separated_rearranged = token_tensor.reshape(-1, 1) @ feature_tensor.reshape(1, -1)
    separated_operator_error = relative_error(separated_rearranged, rearranged)
    token_rows = token_tensor.reshape(layouts[0].tokens, -1)
    feature_rows = feature_tensor.reshape(layouts[0].features, -1)
    token_factor, token_quotient = reduced_rq_rows(token_rows)
    feature_factor, feature_quotient = reduced_rq_rows(feature_rows)
    canonical_layout = Layout(token_quotient.shape[0], feature_quotient.shape[0])
    factor = torch.einsum("ia,jb->ijab", token_factor, feature_factor).reshape(
        layouts[0].size, canonical_layout.size
    )

    token_q = token_quotient.reshape(
        canonical_layout.tokens, *(layout.tokens for layout in layouts[1:])
    )
    feature_q = feature_quotient.reshape(
        canonical_layout.features, *(layout.features for layout in layouts[1:])
    )
    outer = torch.tensordot(token_q, feature_q, dims=0)
    token_axis_count = len(layouts)
    permutation = [0, token_axis_count]
    for role in range(1, token_axis_count):
        permutation.extend((role, token_axis_count + role))
    quotient_core = outer.permute(*permutation).reshape(
        canonical_layout.size, *(layout.size for layout in layouts[1:])
    )
    rows = tensor.reshape(tensor.shape[0], -1)
    quotient = quotient_core.reshape(canonical_layout.size, -1)
    if relative_error(factor @ quotient, rows) > 5e-10:
        raise RuntimeError("structured rank-one RQ failed exact reconstruction")
    generic_factor, generic_quotient = reduced_rq_rows(rows)
    if generic_quotient.shape != quotient.shape:
        raise RuntimeError("structured and generic reduced RQ chose different bond dimensions")
    # Find the orthogonal right gauge relating the two factors.  In rank-
    # deficient cases the unused rows of Q are arbitrary, so comparing Q
    # itself is ill-posed.  Orthogonal Procrustes on R plus comparison after
    # projecting through R is the correct supported-space cross-check.
    gauge_left, _, gauge_right = torch.linalg.svd(
        factor.T @ generic_factor, full_matrices=False
    )
    gauge = gauge_left @ gauge_right
    gauge_identity = torch.eye(gauge.shape[0], dtype=gauge.dtype, device=gauge.device)
    diagnostics = {
        "separated_operator_relative_error": separated_operator_error,
        "generic_output_gram_relative_error": relative_error(
            factor @ factor.T, generic_factor @ generic_factor.T
        ),
        "generic_gauge_orthogonality_error": relative_error(
            gauge @ gauge.T, gauge_identity
        ),
        "generic_factor_gauge_alignment_error": relative_error(
            generic_factor, factor @ gauge
        ),
        "generic_supported_quotient_gauge_alignment_error": relative_error(
            factor @ quotient, factor @ gauge @ generic_quotient
        ),
    }
    return factor, quotient, canonical_layout, diagnostics


def _apply_input_factor(core: torch.Tensor, role: int, factor: torch.Tensor) -> torch.Tensor:
    axis = role + 1
    return (core.movedim(axis, -1) @ factor).movedim(-1, axis)


def _einsum_evaluate(core: torch.Tensor, values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    labels = "".join(label for label in ascii_lowercase if label not in {"b", "o"})
    roles = labels[: len(values)]
    equation = f"o{roles}," + ",".join(f"b{role}" for role in roles) + "->bo"
    return torch.einsum(equation, core, *values)


@torch.no_grad()
def evaluate_tree(root: PrimitiveNode, homogeneous_input: torch.Tensor) -> torch.Tensor:
    memo: dict[int, torch.Tensor] = {}

    def evaluate(node: PrimitiveNode) -> torch.Tensor:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        if node.source_kind == "input":
            value = homogeneous_input @ node.core.T
        elif node.source_kind == "constant":
            value = node.core.reshape(1, -1).expand(homogeneous_input.shape[0], -1)
        else:
            value = _einsum_evaluate(node.core, tuple(evaluate(child) for child in node.children))
        memo[id(node)] = value
        return value

    return evaluate(root)


def _tokenwise_unary(local: torch.Tensor, tokens: int) -> torch.Tensor:
    return torch.kron(torch.eye(tokens, dtype=local.dtype, device=local.device), local)


def _tokenwise_binary(local: torch.Tensor, tokens: int) -> torch.Tensor:
    token = local.new_zeros(tokens, tokens, tokens)
    index = torch.arange(tokens, device=local.device)
    token[index, index, index] = 1.0
    return torch.einsum("ijk,abc->iajbkc", token, local).reshape(
        tokens * local.shape[0], tokens * local.shape[1], tokens * local.shape[2]
    )


def _input_node(tokens: int, dimension: int) -> PrimitiveNode:
    layout = Layout(tokens, dimension + 1)
    return PrimitiveNode(
        "raw_homogeneous_input",
        torch.eye(layout.size, dtype=DTYPE),
        layout,
        source_kind="input",
    )


def _constant_node(values: torch.Tensor, label: str) -> PrimitiveNode:
    if values.ndim != 2:
        raise ValueError("constant values must have shape (tokens, features)")
    tokens, dimension = values.shape
    layout = Layout(tokens, dimension + 1)
    homogeneous = torch.cat((values.new_ones(tokens, 1), values), dim=1).reshape(-1)
    return PrimitiveNode(label, homogeneous, layout, source_kind="constant")


def _add_node(left: PrimitiveNode, right: PrimitiveNode, label: str) -> PrimitiveNode:
    if left.layout != right.layout:
        raise ValueError("homogeneous add requires matching typed layouts")
    feature = left.layout.features
    local = left.core.new_zeros(feature, feature, feature)
    local[0, 0, 0] = 1.0
    index = torch.arange(1, feature, device=local.device)
    local[index, index, 0] = 1.0
    local[index, 0, index] = 1.0
    return PrimitiveNode(
        label,
        _tokenwise_binary(local, left.layout.tokens),
        left.layout,
        (left, right),
    )


def _linear_node(child: PrimitiveNode, weight: torch.Tensor, label: str) -> PrimitiveNode:
    if weight.ndim != 2 or weight.shape[1] + 1 != child.layout.features:
        raise ValueError("linear weight does not match typed child feature dimension")
    local = weight.new_zeros(weight.shape[0] + 1, weight.shape[1] + 1)
    local[0, 0] = 1.0
    local[1:, 1:] = weight
    layout = Layout(child.layout.tokens, weight.shape[0] + 1)
    return PrimitiveNode(label, _tokenwise_unary(local, child.layout.tokens), layout, (child,))


def _affine_node(
    child: PrimitiveNode,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    label: str,
) -> PrimitiveNode:
    linear = _linear_node(child, weight, f"{label}.linear")
    resolved_bias = weight.new_zeros(weight.shape[0]) if bias is None else bias
    repeated = resolved_bias.reshape(1, -1).expand(child.layout.tokens, -1)
    constant = _constant_node(repeated, f"{label}.bias_constant")
    return _add_node(linear, constant, f"{label}.bias_add")


def _scale_node(child: PrimitiveNode, gain: torch.Tensor, label: str) -> PrimitiveNode:
    feature = child.layout.features
    local = child.core.new_zeros(feature, feature)
    local[0, 0] = 1.0
    local[1:, 1:] = torch.eye(feature - 1, dtype=local.dtype, device=local.device) * gain
    return PrimitiveNode(label, _tokenwise_unary(local, child.layout.tokens), child.layout, (child,))


def _hadamard_node(left: PrimitiveNode, right: PrimitiveNode, label: str) -> PrimitiveNode:
    if left.layout != right.layout:
        raise ValueError("Hadamard product requires matching typed layouts")
    feature = left.layout.features
    local = left.core.new_zeros(feature, feature, feature)
    index = torch.arange(feature, device=local.device)
    local[index, index, index] = 1.0
    return PrimitiveNode(
        label,
        _tokenwise_binary(local, left.layout.tokens),
        left.layout,
        (left, right),
    )


@torch.no_grad()
def _attention_contraction_node(
    q1: PrimitiveNode,
    q2: PrimitiveNode,
    k1: PrimitiveNode,
    k2: PrimitiveNode,
    value: PrimitiveNode,
    block: ChiTransformerBlock,
    label: str,
    maximum_elements: int,
) -> PrimitiveNode:
    roles = (q1, q2, k1, k2, value)
    if len({role.layout for role in roles}) != 1:
        raise ValueError("ordered attention roles must share a typed layout")
    layout = q1.layout
    tokens = layout.tokens
    dimension = layout.features - 1
    if block.attn.n_heads != 1 or block.attn.head_dim != dimension:
        raise ValueError("primitive dense oracle currently requires one full-width head")
    elements = layout.size ** 6
    if elements > maximum_elements:
        raise ValueError(f"attention contraction needs {elements} elements above cap")
    core = q1.core.new_zeros((layout.size,) * 6)
    for query in range(tokens):
        constant = query * layout.features
        core[(constant, constant, constant, constant, constant, constant)] = 1.0
    mask = causal_mask(tokens, dtype=core.dtype, device=core.device)
    row_scale = block.attn._row_scale_vec(mask)
    for query in range(tokens):
        for source in range(tokens):
            route_scale = row_scale[query] * mask[query, source] / block.attn._score_denom
            if float(route_scale.item()) == 0.0:
                continue
            for alpha, beta, channel in itertools.product(range(dimension), repeat=3):
                output_index = query * layout.features + channel + 1
                indices = (
                    query * layout.features + alpha + 1,
                    query * layout.features + beta + 1,
                    source * layout.features + alpha + 1,
                    source * layout.features + beta + 1,
                    source * layout.features + channel + 1,
                )
                core[(output_index, *indices)] += route_scale
    return PrimitiveNode(label, core, layout, roles)


def _make_blocks(seed: int, dimension: int, depth: int, affine_biases: bool) -> tuple[ChiTransformerBlock, ...]:
    torch.manual_seed(seed)
    blocks = tuple(
        ChiTransformerBlock(
            dim=dimension,
            n_heads=1,
            ffn_rank=max(3, 3 * dimension),
            n_layers=depth,
            causal=True,
            norm="none",
            qk_norm="none",
            residual=True,
        ).double().eval()
        for _ in range(depth)
    )
    generator = torch.Generator().manual_seed(10_000 + seed)
    with torch.no_grad():
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear) and module.bias is not None:
                    if affine_biases:
                        module.bias.normal_(std=0.2, generator=generator)
                    else:
                        module.bias.zero_()
    return blocks


@torch.no_grad()
def _compile_primitive_tree(
    blocks: tuple[ChiTransformerBlock, ...],
    position: torch.Tensor,
    maximum_elements: int,
) -> PrimitiveNode:
    tokens, dimension = position.shape
    state = _input_node(tokens, dimension)
    state = _add_node(state, _constant_node(position, "position.constant"), "position.add")
    for layer, block in enumerate(blocks):
        prefix = f"block{layer}"
        q1 = _affine_node(state, block.attn.wq1.weight, block.attn.wq1.bias, f"{prefix}.q1")
        q2 = _affine_node(state, block.attn.wq2.weight, block.attn.wq2.bias, f"{prefix}.q2")
        k1 = _affine_node(state, block.attn.wk1.weight, block.attn.wk1.bias, f"{prefix}.k1")
        k2 = _affine_node(state, block.attn.wk2.weight, block.attn.wk2.bias, f"{prefix}.k2")
        value = _affine_node(state, block.attn.wv.weight, block.attn.wv.bias, f"{prefix}.value")
        attention = _attention_contraction_node(
            q1, q2, k1, k2, value, block, f"{prefix}.attention_contraction", maximum_elements
        )
        attention = _affine_node(
            attention, block.attn.wo.weight, block.attn.wo.bias, f"{prefix}.output"
        )
        attention = _scale_node(attention, block.attn_gain, f"{prefix}.attention_gain")
        after_attention = _add_node(state, attention, f"{prefix}.attention_residual_add")

        left = _affine_node(
            after_attention, block.ffn.left.weight, block.ffn.left.bias, f"{prefix}.ffn_left"
        )
        right = _affine_node(
            after_attention, block.ffn.right.weight, block.ffn.right.bias, f"{prefix}.ffn_right"
        )
        product = _hadamard_node(left, right, f"{prefix}.ffn_hadamard")
        update = _affine_node(
            product, block.ffn.down.weight, block.ffn.down.bias, f"{prefix}.ffn_down"
        )
        update = _scale_node(update, block.ffn_gain, f"{prefix}.ffn_gain")
        state = _add_node(after_attention, update, f"{prefix}.ffn_residual_add")
    return state


@torch.no_grad()
def canonicalize_tree(
    root: PrimitiveNode,
    raw_layout: Layout,
    tolerances: tuple[float, ...],
) -> tuple[
    PrimitiveNode,
    torch.Tensor,
    tuple[dict[str, object], ...],
    dict[str, torch.Tensor],
]:
    records: list[dict[str, object]] = []
    memo: dict[int, tuple[PrimitiveNode, torch.Tensor]] = {}
    factors_by_label: dict[str, torch.Tensor] = {}

    def canonicalize(node: PrimitiveNode) -> tuple[PrimitiveNode, torch.Tensor]:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        child_results = tuple(canonicalize(child) for child in node.children)
        children = tuple(result[0] for result in child_results)
        transformed = node.core.detach().clone()
        for role, (_, factor) in enumerate(child_results):
            transformed = _apply_input_factor(transformed, role, factor)
        if node.source_kind == "input":
            raw_layouts = (node.layout, raw_layout)
            transformed_layouts = raw_layouts
        elif node.source_kind == "constant":
            raw_layouts = (node.layout,)
            transformed_layouts = raw_layouts
        else:
            raw_layouts = (node.layout,) + tuple(child.layout for child in node.children)
            transformed_layouts = (node.layout,) + tuple(child.layout for child in children)
        rows = transformed.reshape(transformed.shape[0], -1)
        structured = _structured_rank_one_rq(transformed, transformed_layouts)
        if structured is None:
            factor, quotient = reduced_rq_rows(rows)
            canonical_layout = _canonical_layout(node.layout, quotient.shape[0])
            rq_method = "generic_reduced_rq"
            structured_diagnostics = None
        else:
            factor, quotient, canonical_layout, structured_diagnostics = structured
            rq_method = "token_feature_rank_one_rq"
        quotient_core = quotient.reshape((quotient.shape[0],) + transformed.shape[1:])
        canonical = PrimitiveNode(
            node.label,
            quotient_core,
            canonical_layout,
            children,
            node.source_kind,
        )

        if node.source_kind == "input":
            quotient_layouts = (canonical_layout, raw_layout)
        elif node.source_kind == "constant":
            quotient_layouts = (canonical_layout,)
        else:
            quotient_layouts = (canonical_layout,) + tuple(child.layout for child in children)

        raw_spectrum = _operator_spectrum(node.core, raw_layouts)
        transformed_spectrum = _operator_spectrum(transformed, transformed_layouts)
        factor_spectrum = _operator_spectrum(factor, (node.layout, canonical_layout))
        quotient_spectrum = _operator_spectrum(quotient_core, quotient_layouts)
        identity = torch.eye(quotient.shape[0], dtype=quotient.dtype, device=quotient.device)
        records.append(
            {
                "label": node.label,
                "arity": len(node.children),
                "source_kind": node.source_kind,
                "rq_method": rq_method,
                "structured_rq_crosscheck": structured_diagnostics,
                "output_layout": [node.layout.tokens, node.layout.features],
                "canonical_layout": [canonical_layout.tokens, canonical_layout.features],
                "raw_core_operator_schmidt": _spectrum_record(raw_spectrum, tolerances),
                "transformed_core_operator_schmidt": _spectrum_record(
                    transformed_spectrum, tolerances
                ),
                "emitted_factor_operator_schmidt": _spectrum_record(factor_spectrum, tolerances),
                "quotient_core_operator_schmidt": _spectrum_record(quotient_spectrum, tolerances),
                "factorization_relative_error": relative_error(factor @ quotient, rows),
                "row_isometry_error": relative_error(quotient @ quotient.T, identity),
            }
        )
        if node.label in factors_by_label:
            raise ValueError(f"primitive node label is not unique: {node.label!r}")
        factors_by_label[node.label] = factor.detach().clone()
        result = (canonical, factor)
        memo[id(node)] = result
        return result

    canonical_root, root_factor = canonicalize(root)
    return canonical_root, root_factor, tuple(records), factors_by_label


def _low_rank_delta_record(
    factor: torch.Tensor,
    baseline: torch.Tensor,
    tolerances: tuple[float, ...],
) -> dict[str, object]:
    """Measure ``factor = baseline + U diag(s) V^T`` exactly by SVD."""

    if factor.shape != baseline.shape:
        return {
            "shape_match": False,
            "factor_shape": list(factor.shape),
            "baseline_shape": list(baseline.shape),
        }
    delta = factor - baseline
    values = torch.linalg.svdvals(delta)
    factor_scale = factor.norm().clamp_min(torch.finfo(factor.dtype).tiny)
    delta_scale = values[0] if values.numel() and float(values[0].item()) > 0.0 else None
    if delta_scale is None:
        ranks = {f"{tolerance:.0e}": 0 for tolerance in tolerances}
    else:
        ranks = {
            f"{tolerance:.0e}": int((values / delta_scale > tolerance).sum().item())
            for tolerance in tolerances
        }
    reconstruction_errors = {}
    for rank in (0, 1, 2, 4, 8):
        tail = values[rank:]
        reconstruction_errors[str(rank)] = float(
            (torch.linalg.vector_norm(tail) / factor_scale).item()
        )
    return {
        "shape_match": True,
        "ordinary_ranks": ranks,
        "delta_relative_frobenius_norm": float((delta.norm() / factor_scale).item()),
        "baseline_plus_best_rank_k_relative_errors": reconstruction_errors,
        "normalized_delta_singular_values": (
            [float(value.item()) for value in (values / delta_scale)[:16]]
            if delta_scale is not None
            else [0.0 for _ in values[:16]]
        ),
    }


@torch.no_grad()
def run_case(
    *,
    seed: int,
    tokens: int,
    dimension: int,
    depth: int,
    case: PrimitiveCase,
    tolerances: tuple[float, ...],
    maximum_elements: int,
) -> dict[str, object]:
    blocks = _make_blocks(seed, dimension, depth, case.affine_biases)
    source_tensors = tuple(tensor.detach() for block in blocks for tensor in block.state_dict().values())
    source_hash_before = _tensor_sha256(source_tensors)
    generator = torch.Generator().manual_seed(20_000 + seed)
    position = (
        torch.randn(tokens, dimension, generator=generator, dtype=DTYPE) * 0.3
        if case.position_embedding
        else torch.zeros(tokens, dimension, dtype=DTYPE)
    )
    root = _compile_primitive_tree(blocks, position, maximum_elements)
    raw = torch.randn(3, tokens, dimension, generator=generator, dtype=DTYPE)
    homogeneous = torch.cat((torch.ones(3, tokens, 1, dtype=DTYPE), raw), dim=-1).reshape(3, -1)
    primitive = evaluate_tree(root, homogeneous)
    source = raw + position
    mask = causal_mask(tokens, dtype=DTYPE)
    for block in blocks:
        source = block(source, mask=mask, method="explicit")
    primitive_features = primitive.reshape(3, tokens, dimension + 1)[..., 1:]
    primitive_constants = primitive.reshape(3, tokens, dimension + 1)[..., 0]

    canonical_root, root_factor, records, factors_by_label = canonicalize_tree(
        root, Layout(tokens, dimension + 1), tolerances
    )
    canonical = evaluate_tree(canonical_root, homogeneous) @ root_factor.T
    if not case.affine_biases and not case.position_embedding:
        baseline_factors = factors_by_label
    else:
        baseline_blocks = _make_blocks(seed, dimension, depth, False)
        baseline_root = _compile_primitive_tree(
            baseline_blocks,
            torch.zeros(tokens, dimension, dtype=DTYPE),
            maximum_elements,
        )
        _, _, _, baseline_factors = canonicalize_tree(
            baseline_root, Layout(tokens, dimension + 1), tolerances
        )
    for record in records:
        record["matched_bias_position_free_factor_delta"] = _low_rank_delta_record(
            factors_by_label[record["label"]],
            baseline_factors[record["label"]],
            tolerances,
        )
    tight_key = f"{min(tolerances):.0e}"
    factor_ranks = [
        int(record["emitted_factor_operator_schmidt"]["ranks"][tight_key])
        for record in records
    ]
    internal_records = [record for record in records if record["source_kind"] != "constant"]
    internal_factor_ranks = [
        int(record["emitted_factor_operator_schmidt"]["ranks"][tight_key])
        for record in internal_records
    ]
    position_records = [record for record in records if record["label"].startswith("position.")]
    bias_records = [record for record in records if ".bias_" in record["label"]]
    delta_records = [
        record["matched_bias_position_free_factor_delta"]
        for record in records
        if record["matched_bias_position_free_factor_delta"]["shape_match"]
    ]
    structured_records = [
        record["structured_rq_crosscheck"]
        for record in records
        if record["structured_rq_crosscheck"] is not None
    ]
    source_hash_after = _tensor_sha256(source_tensors)
    return {
        "case": case.name,
        "seed": seed,
        "tokens": tokens,
        "dimension": dimension,
        "depth": depth,
        "affine_biases": case.affine_biases,
        "position_embedding": case.position_embedding,
        "normalization": "identity_polynomial_scope",
        "primitive_node_count": len(records),
        "primitive_forward_relative_error": relative_error(primitive_features, source),
        "homogeneous_constant_maximum_absolute_error": float(
            (primitive_constants - 1.0).abs().max().item()
        ),
        "canonical_replay_relative_error": relative_error(canonical, primitive),
        "maximum_factorization_relative_error": max(
            float(record["factorization_relative_error"]) for record in records
        ),
        "maximum_row_isometry_error": max(float(record["row_isometry_error"]) for record in records),
        "source_unchanged": source_hash_before == source_hash_after,
        "tight_rank_tolerance": min(tolerances),
        "all_node_factor_ranks": factor_ranks,
        "nonconstant_node_factor_ranks": internal_factor_ranks,
        "all_node_rank_one_factor_closure": all(rank == 1 for rank in factor_ranks),
        "nonconstant_node_rank_one_factor_closure": all(
            rank == 1 for rank in internal_factor_ranks
        ),
        "maximum_all_node_factor_rank": max(factor_ranks),
        "maximum_nonconstant_node_factor_rank": max(internal_factor_ranks),
        "maximum_position_node_factor_rank": max(
            int(record["emitted_factor_operator_schmidt"]["ranks"][tight_key])
            for record in position_records
        ),
        "maximum_bias_node_factor_rank": max(
            int(record["emitted_factor_operator_schmidt"]["ranks"][tight_key])
            for record in bias_records
        ),
        "maximum_factor_delta_ordinary_rank": max(
            int(record["ordinary_ranks"][tight_key]) for record in delta_records
        ),
        "maximum_baseline_plus_rank2_relative_error": max(
            float(record["baseline_plus_best_rank_k_relative_errors"]["2"])
            for record in delta_records
        ),
        "maximum_baseline_plus_rank4_relative_error": max(
            float(record["baseline_plus_best_rank_k_relative_errors"]["4"])
            for record in delta_records
        ),
        "structured_rq_node_count": len(structured_records),
        "maximum_separated_operator_relative_error": max(
            float(record["separated_operator_relative_error"])
            for record in structured_records
        ),
        "maximum_structured_vs_generic_output_gram_relative_error": max(
            float(record["generic_output_gram_relative_error"])
            for record in structured_records
        ),
        "maximum_structured_vs_generic_gauge_crosscheck_error": max(
            max(
                float(record["generic_gauge_orthogonality_error"]),
                float(record["generic_factor_gauge_alignment_error"]),
                float(record["generic_supported_quotient_gauge_alignment_error"]),
            )
            for record in structured_records
        ),
        "records": records,
    }


def _parse_grid(value: str) -> tuple[tuple[int, int, int], ...]:
    result = []
    for item in value.split(","):
        fields = item.split("x")
        if len(fields) != 3:
            raise ValueError("grid entries must have NxDxL form")
        triple = tuple(int(field) for field in fields)
        if any(field <= 0 for field in triple):
            raise ValueError("grid values must be positive")
        result.append(triple)
    return tuple(result)


def _parse_tolerances(value: str) -> tuple[float, ...]:
    result = tuple(sorted({float(item) for item in value.split(",")}, reverse=True))
    if not result or any(not 0.0 < item < 1.0 for item in result):
        raise ValueError("rank tolerances must be distinct values strictly between zero and one")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="2x1x1,2x1x2,2x2x1,2x2x2,3x1x1,3x1x2")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rank-tolerances", default="1e-8,1e-10,1e-12")
    parser.add_argument("--maximum-elements", type=int, default=5_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    grid = _parse_grid(args.grid)
    tolerances = _parse_tolerances(args.rank_tolerances)
    if args.seeds <= 0 or args.maximum_elements <= 0:
        raise ValueError("seeds and maximum element cap must be positive")
    started = time.perf_counter()
    rows = [
        run_case(
            seed=seed,
            tokens=tokens,
            dimension=dimension,
            depth=depth,
            case=case,
            tolerances=tolerances,
            maximum_elements=args.maximum_elements,
        )
        for tokens, dimension, depth in grid
        for seed in range(args.seeds)
        for case in CASES
    ]
    gates = {
        "all_primitive_forwards_exact": max(
            float(row["primitive_forward_relative_error"]) for row in rows
        )
        < 1e-9,
        "all_homogeneous_constants_exact": max(
            float(row["homogeneous_constant_maximum_absolute_error"]) for row in rows
        )
        < 1e-12,
        "all_canonical_replays_exact": max(
            float(row["canonical_replay_relative_error"]) for row in rows
        )
        < 1e-9,
        "all_factorizations_exact": max(
            float(row["maximum_factorization_relative_error"]) for row in rows
        )
        < 1e-9,
        "all_rows_isometric": max(float(row["maximum_row_isometry_error"]) for row in rows) < 1e-9,
        "all_source_buffers_unchanged": all(bool(row["source_unchanged"]) for row in rows),
        "all_structured_operator_products_exact": max(
            float(row["maximum_separated_operator_relative_error"]) for row in rows
        )
        < 1e-9,
        "all_structured_rq_output_grams_match_generic": max(
            float(row["maximum_structured_vs_generic_output_gram_relative_error"])
            for row in rows
        )
        < 1e-9,
        "all_structured_rq_gauges_match_generic": max(
            float(row["maximum_structured_vs_generic_gauge_crosscheck_error"])
            for row in rows
        )
        < 1e-8,
        "all_four_affine_position_cases_present": {row["case"] for row in rows}
        == {case.name for case in CASES},
        "multi_tolerance_spectra_present": len(tolerances) >= 3,
    }
    by_case: dict[str, dict[str, object]] = {}
    for case in CASES:
        selected = [row for row in rows if row["case"] == case.name]
        by_case[case.name] = {
            "row_count": len(selected),
            "all_node_rank_one_closure_fraction": sum(
                bool(row["all_node_rank_one_factor_closure"]) for row in selected
            )
            / len(selected),
            "nonconstant_node_rank_one_closure_fraction": sum(
                bool(row["nonconstant_node_rank_one_factor_closure"]) for row in selected
            )
            / len(selected),
            "maximum_all_node_factor_rank": max(
                int(row["maximum_all_node_factor_rank"]) for row in selected
            ),
            "maximum_nonconstant_node_factor_rank": max(
                int(row["maximum_nonconstant_node_factor_rank"]) for row in selected
            ),
            "maximum_position_node_factor_rank": max(
                int(row["maximum_position_node_factor_rank"]) for row in selected
            ),
            "maximum_bias_node_factor_rank": max(
                int(row["maximum_bias_node_factor_rank"]) for row in selected
            ),
            "maximum_factor_delta_ordinary_rank": max(
                int(row["maximum_factor_delta_ordinary_rank"]) for row in selected
            ),
            "maximum_baseline_plus_rank2_relative_error": max(
                float(row["maximum_baseline_plus_rank2_relative_error"])
                for row in selected
            ),
            "maximum_baseline_plus_rank4_relative_error": max(
                float(row["maximum_baseline_plus_rank4_relative_error"])
                for row in selected
            ),
        }
    result = {
        "schema": "xvla-primitive-kron-closure-audit-v1",
        "object_kind": "ordered_typed_primitive_polynomial_arithmetic_dag",
        "claim_boundary": (
            "Exact identity-normalized residual attention+FFN primitive graph. "
            "This does not replace or polynomialize deployed Padé RationalNorm; "
            "that rational case is tested by the global projective oracle."
        ),
        "rank_definition": "operator-Schmidt rank across all token axes versus all feature axes",
        "rank_tolerances_relative": tolerances,
        "grid": grid,
        "seeds": args.seeds,
        "case_names": [case.name for case in CASES],
        "elapsed_seconds": time.perf_counter() - started,
        "gates": gates,
        "all_correctness_gates_pass": all(gates.values()),
        "finding_by_case": by_case,
        "rows": rows,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
        print(
            json.dumps(
                {
                    "schema": result["schema"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "gates": gates,
                    "finding_by_case": by_case,
                    "output": str(args.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
    if not result["all_correctness_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
