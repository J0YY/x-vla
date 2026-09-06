"""Typed Padé block compilation. No orthogonalization or environment eigendecomposition."""

from __future__ import annotations


import torch

import torch.nn as nn
from xvla.nn.attention import BilinearAttention, causal_mask
from xvla.nn.bilinear import BilinearFFN
from xvla.nn.block import ChiTransformerBlock
from xvla.nn.normalization import RationalNorm

from xvla.train.odt_engine_v2.ops import (
    _core_arity,
    _make_cp_core,
    _make_unary_core,
    _strip_power_of_two,
)

from xvla.train.odt_engine_v2.types import (
    ImplicitBlockOracle,
    ImplicitCore,
    ImplicitNode,
    ImplicitProjectiveDAG,
    MaterializationTelemetry,
    PhysicalSourceSpec,
    ProjectiveConstant,
    ProjectiveValue,
    Tensor,
    _validate_real_finite,
)


class _Builder:
    def __init__(self, like: Tensor, telemetry: MaterializationTelemetry):
        self.like = like
        self.telemetry = telemetry
        self.next_uid = 0
        self._physical_specs: dict[str, PhysicalSourceSpec] = {}
        self._physical_nodes: dict[str, ImplicitNode] = {}

    @property
    def physical_source_specs(self) -> tuple[PhysicalSourceSpec, ...]:
        return tuple(self._physical_specs.values())

    def _constant_coordinates(
        self,
        label: str,
        coordinates: Tensor,
        binary_exponent: int = 0,
    ) -> ProjectiveConstant:
        _validate_real_finite(coordinates, f"constant {label}")
        if coordinates.ndim != 1 or coordinates.numel() < 1:
            raise ValueError("constant projective coordinates must be a nonempty vector")
        mantissa, local = _strip_power_of_two(coordinates)
        self.telemetry.observe_persistent(mantissa)
        return ProjectiveConstant(label, mantissa, binary_exponent + local)

    def constant_pair(self, label: str, value: Tensor) -> ProjectiveConstant:
        _validate_real_finite(value, f"constant pair {label}")
        if value.ndim != 1:
            raise ValueError("constant pair value must be a vector")
        if value.dtype != self.like.dtype or value.device != self.like.device:
            raise ValueError("constant pair must share builder dtype and device")
        coordinates = torch.cat((value, value.new_ones(1)))
        return self._constant_coordinates(label, coordinates)

    def physical_pair(self, spec: PhysicalSourceSpec) -> ImplicitNode:
        existing = self._physical_nodes.get(spec.key)
        if existing is not None:
            if self._physical_specs[spec.key] != spec:
                raise ValueError("physical source key was reused with a different specification")
            return existing
        identity = torch.eye(
            spec.width + 1, dtype=self.like.dtype, device=self.like.device
        )
        result = self.unary(
            spec.label,
            identity,
            physical_source_key=spec.key,
            kind="heterogeneous_physical_homogeneous_embedding",
        )
        if not isinstance(result, ImplicitNode):
            raise RuntimeError("physical source construction unexpectedly folded")
        self._physical_specs[spec.key] = spec
        self._physical_nodes[spec.key] = result
        return result

    def node(
        self,
        label: str,
        core: ImplicitCore,
        children: tuple[ImplicitNode, ...] = (),
        *,
        physical_token: int | None = None,
        physical_source_key: str | None = None,
    ) -> ImplicitNode:
        if physical_token is not None and physical_source_key is not None:
            raise ValueError("a physical leaf cannot select two raw sources")
        physical_leaf = (
            physical_token is not None or physical_source_key is not None
        ) and not children
        if not physical_leaf and len(children) != _core_arity(core):
            raise ValueError(f"{label} child/core arity mismatch")
        if children and tuple(child.output_dimension for child in children) != core.input_dimensions:
            raise ValueError(f"{label} child/core dimensions mismatch")
        result = ImplicitNode(
            self.next_uid,
            label,
            core,
            children,
            physical_token,
            self.next_uid,
            physical_source_key,
        )
        self.next_uid += 1
        return result

    def unary(
        self,
        label: str,
        matrix: Tensor,
        child: ProjectiveValue | None = None,
        *,
        physical_token: int | None = None,
        physical_source_key: str | None = None,
        kind: str = "affine",
    ) -> ProjectiveValue:
        core = _make_unary_core(matrix, kind, self.telemetry)
        if isinstance(child, ProjectiveConstant):
            self.telemetry.constant_unary_folds += 1
            value = core.matrix @ child.mantissa
            return self._constant_coordinates(
                label,
                value,
                core.binary_exponent + child.binary_exponent,
            )
        children = () if child is None else (child,)
        return self.node(
            label,
            core,
            children,
            physical_token=physical_token,
            physical_source_key=physical_source_key,
        )

    def cp(
        self,
        label: str,
        output: Tensor,
        left: Tensor,
        right: Tensor,
        children: tuple[ProjectiveValue, ProjectiveValue],
        *,
        kind: str,
    ) -> ProjectiveValue:
        # Declare the symmetric ordered lift at compilation, never silently
        # change it inside Algorithm 1. This preserves tied-input evaluation.
        if children[0] is children[1]:
            output = torch.cat((output / 2, output / 2), dim=1)
            left, right = torch.cat((left, right)), torch.cat((right, left))
        core = _make_cp_core(output, left, right, kind, self.telemetry)
        left_child, right_child = children
        left_constant = isinstance(left_child, ProjectiveConstant)
        right_constant = isinstance(right_child, ProjectiveConstant)
        if left_constant and right_constant:
            self.telemetry.constant_cp_to_constant_folds += 1
            left_value = core.left_factor @ left_child.mantissa
            right_value = core.right_factor @ right_child.mantissa
            value = core.output_factor @ (left_value * right_value)
            return self._constant_coordinates(
                label,
                value,
                core.binary_exponent
                + left_child.binary_exponent
                + right_child.binary_exponent,
            )
        if left_constant or right_constant:
            self.telemetry.constant_cp_to_unary_folds += 1
            constant = left_child if left_constant else right_child
            variable = right_child if left_constant else left_child
            if not isinstance(constant, ProjectiveConstant) or not isinstance(
                variable, ImplicitNode
            ):
                raise RuntimeError("constant CP partial evaluation lost its typed children")
            constant_factor = (
                core.left_factor if left_constant else core.right_factor
            )
            variable_factor = (
                core.right_factor if left_constant else core.left_factor
            )
            weights = constant_factor @ constant.mantissa
            matrix = (core.output_factor * weights[None, :]) @ variable_factor
            unary = _make_unary_core(
                matrix, f"constant_folded::{kind}", self.telemetry
            )
            unary.binary_exponent += core.binary_exponent + constant.binary_exponent
            return self.node(label, unary, (variable,))
        return self.node(
            label,
            core,
            (left_child, right_child),
        )


def _identity_pair_leaf(builder: _Builder, token: int, dimension: int) -> ImplicitNode:
    identity = torch.eye(dimension + 1, dtype=builder.like.dtype, device=builder.like.device)
    result = builder.unary(
        f"input.token{token}",
        identity,
        physical_token=token,
        kind="physical_homogeneous_embedding",
    )
    if not isinstance(result, ImplicitNode):
        raise RuntimeError("physical token construction unexpectedly folded")
    return result


def _affine_pair(
    builder: _Builder,
    value: ProjectiveValue,
    weight: Tensor,
    bias: Tensor | None,
    label: str,
) -> ProjectiveValue:
    output, input_dimension = weight.shape
    if value.output_dimension != input_dimension + 1:
        raise ValueError(f"{label} affine input dimension mismatch")
    matrix = weight.new_zeros(output + 1, input_dimension + 1)
    matrix[:-1, :-1] = weight
    if bias is not None:
        if bias.shape != (output,):
            raise ValueError(f"{label} affine bias dimension mismatch")
        matrix[:-1, -1] = bias
    matrix[-1, -1] = 1.0
    return builder.unary(label, matrix, value, kind="projective_affine")


def _scale_pair_vector(
    builder: _Builder,
    value: ProjectiveValue,
    scale: Tensor | float,
    label: str,
) -> ProjectiveValue:
    dimension = value.output_dimension - 1
    scalar = torch.as_tensor(
        scale, dtype=builder.like.dtype, device=builder.like.device
    ).reshape(())
    weight = torch.eye(dimension, dtype=builder.like.dtype, device=builder.like.device) * scalar
    return _affine_pair(builder, value, weight, None, label)


def _add_pair_vectors(
    builder: _Builder,
    left_value: ProjectiveValue,
    right_value: ProjectiveValue,
    label: str,
) -> ProjectiveValue:
    if left_value.output_dimension != right_value.output_dimension:
        raise ValueError("projective add requires equal vector dimensions")
    size = left_value.output_dimension
    dimension = size - 1
    rank = 2 * dimension + 1
    output = builder.like.new_zeros(size, rank)
    left = builder.like.new_zeros(rank, size)
    right = builder.like.new_zeros(rank, size)
    term = 0
    for index in range(dimension):
        output[index, term] = 1.0
        left[term, index] = 1.0
        right[term, -1] = 1.0
        term += 1
        output[index, term] = 1.0
        left[term, -1] = 1.0
        right[term, index] = 1.0
        term += 1
    output[-1, term] = 1.0
    left[term, -1] = 1.0
    right[term, -1] = 1.0
    return builder.cp(
        label,
        output,
        left,
        right,
        (left_value, right_value),
        kind="sparse_projective_vector_add",
    )


def _dot_pair_vectors(
    builder: _Builder,
    left_value: ProjectiveValue,
    right_value: ProjectiveValue,
    label: str,
) -> ProjectiveValue:
    if left_value.output_dimension != right_value.output_dimension:
        raise ValueError("projective dot requires equal vector dimensions")
    size = left_value.output_dimension
    dimension = size - 1
    rank = dimension + 1
    output = builder.like.new_zeros(2, rank)
    left = builder.like.new_zeros(rank, size)
    right = builder.like.new_zeros(rank, size)
    for index in range(dimension):
        output[0, index] = 1.0
        left[index, index] = 1.0
        right[index, index] = 1.0
    output[1, -1] = 1.0
    left[-1, -1] = 1.0
    right[-1, -1] = 1.0
    return builder.cp(
        label,
        output,
        left,
        right,
        (left_value, right_value),
        kind="sparse_projective_dot",
    )


def _multiply_pair_scalars(
    builder: _Builder,
    left_value: ProjectiveValue,
    right_value: ProjectiveValue,
    label: str,
) -> ProjectiveValue:
    if left_value.output_dimension != 2 or right_value.output_dimension != 2:
        raise ValueError("projective scalar multiply needs scalar pairs")
    output = torch.eye(2, dtype=builder.like.dtype, device=builder.like.device)
    factors = output.clone()
    return builder.cp(
        label,
        output,
        factors,
        factors,
        (left_value, right_value),
        kind="sparse_projective_scalar_product",
    )


def _scale_pair_vector_by_scalar(
    builder: _Builder,
    scalar: ProjectiveValue,
    vector: ProjectiveValue,
    label: str,
) -> ProjectiveValue:
    if scalar.output_dimension != 2:
        raise ValueError("first scale input must be a projective scalar")
    size = vector.output_dimension
    output = torch.eye(size, dtype=builder.like.dtype, device=builder.like.device)
    scalar_factor = builder.like.new_zeros(size, 2)
    vector_factor = torch.eye(size, dtype=builder.like.dtype, device=builder.like.device)
    scalar_factor[:-1, 0] = 1.0
    scalar_factor[-1, 1] = 1.0
    return builder.cp(
        label,
        output,
        scalar_factor,
        vector_factor,
        (scalar, vector),
        kind="sparse_projective_scalar_times_vector",
    )


def _concatenate_pair_vectors(
    builder: _Builder,
    first: ProjectiveValue,
    second: ProjectiveValue,
    label: str,
) -> ProjectiveValue:
    first_dimension = first.output_dimension - 1
    second_dimension = second.output_dimension - 1
    output_dimension = first_dimension + second_dimension + 1
    rank = output_dimension
    output = torch.eye(
        output_dimension, dtype=builder.like.dtype, device=builder.like.device
    )
    left = builder.like.new_zeros(rank, first_dimension + 1)
    right = builder.like.new_zeros(rank, second_dimension + 1)
    for index in range(first_dimension):
        left[index, index] = 1.0
        right[index, -1] = 1.0
    offset = first_dimension
    for index in range(second_dimension):
        left[offset + index, -1] = 1.0
        right[offset + index, index] = 1.0
    left[-1, -1] = 1.0
    right[-1, -1] = 1.0
    return builder.cp(
        label,
        output,
        left,
        right,
        (first, second),
        kind="sparse_projective_vector_concat",
    )


def _mean_square_pair(
    builder: _Builder,
    value: ProjectiveValue,
    module: RationalNorm,
    label: str,
) -> ProjectiveValue:
    dimension = value.output_dimension - 1
    size = dimension + 1
    rank = dimension + 1
    output = builder.like.new_zeros(2, rank)
    factors = builder.like.new_zeros(rank, size)
    for index in range(dimension):
        output[0, index] = 1.0 / dimension
        factors[index, index] = 1.0
    scale = module.running_ms.to(dtype=builder.like.dtype, device=builder.like.device).clamp_min(1e-12)
    output[0, -1] = float(module.eps)
    output[1, -1] = scale
    factors[-1, -1] = 1.0
    return builder.cp(
        label,
        output,
        factors,
        factors.clone(),
        (value, value),
        kind="sparse_projective_mean_square",
    )


def _pade_pair(
    builder: _Builder,
    square: ProjectiveValue,
    module: RationalNorm,
    label: str,
) -> ProjectiveValue:
    if square.output_dimension != 2 or int(module.deg) != 2:
        raise ValueError("implicit production kernel requires degree-two Padé")
    pa = module.pa.to(dtype=builder.like.dtype, device=builder.like.device)
    pb = module.pb.to(dtype=builder.like.dtype, device=builder.like.device)
    # Ordered atoms TT, ST, TS, SS.  Splitting the mixed coefficient equally
    # preserves the symmetric clone-unfolded convention of the dense oracle.
    output = torch.stack(
        (
            torch.stack((pa[0], pb[0])),
            torch.stack((pa[1] / 2.0, pb[1] / 2.0)),
            torch.stack((pa[1] / 2.0, pb[1] / 2.0)),
            torch.stack((pa[2], pb[2])),
        ),
        dim=1,
    )
    left = builder.like.new_tensor(((0.0, 1.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    right = builder.like.new_tensor(((0.0, 1.0), (0.0, 1.0), (1.0, 0.0), (1.0, 0.0)))
    return builder.cp(
        label,
        output,
        left,
        right,
        (square, square),
        kind="sparse_projective_pade_degree2",
    )


def _pade_norm_pair(
    builder: _Builder,
    value: ProjectiveValue,
    module: RationalNorm,
    label: str,
) -> ProjectiveValue:
    if not isinstance(module, RationalNorm) or module.variant != "pade":
        raise ValueError(f"{label} is not a Padé RationalNorm")
    if not bool(module.initialized):
        raise ValueError(f"{label} has an uninitialized running mean square")
    square = _mean_square_pair(builder, value, module, f"{label}.mean_square")
    pade = _pade_pair(builder, square, module, f"{label}.pade")
    dimension = value.output_dimension - 1
    size = dimension + 1
    output = torch.eye(size, dtype=builder.like.dtype, device=builder.like.device)
    vector_factor = output.clone()
    scalar_factor = builder.like.new_zeros(size, 2)
    scalar_factor[:-1, 0] = torch.rsqrt(
        module.running_ms.to(dtype=builder.like.dtype, device=builder.like.device).clamp_min(1e-12)
    )
    scalar_factor[-1, 1] = 1.0
    return builder.cp(
        f"{label}.projective_output",
        output,
        vector_factor,
        scalar_factor,
        (value, pade),
        kind="sparse_projective_pade_vector_output",
    )


def _fused_cp_ffn_pair(
    builder: _Builder,
    value: ProjectiveValue,
    module: BilinearFFN,
    label: str,
) -> ProjectiveValue:
    dimension = value.output_dimension - 1
    if module.dim != dimension or module.out_dim != dimension:
        raise ValueError("fused CP FFN dimensions do not match its pair bond")
    rank = module.rank + 1
    size = dimension + 1
    left = builder.like.new_zeros(rank, size)
    right = builder.like.new_zeros(rank, size)
    left[:-1, :-1] = module.left.weight
    left[:-1, -1] = module.left.bias
    right[:-1, :-1] = module.right.weight
    right[:-1, -1] = module.right.bias
    left[-1, -1] = 1.0
    right[-1, -1] = 1.0
    output = builder.like.new_zeros(size, rank)
    output[:-1, :-1] = module.down.weight
    if module.down.bias is not None:
        output[:-1, -1] = module.down.bias
    output[-1, -1] = 1.0
    builder.telemetry.fused_ffn_primitive_count += 1
    return builder.cp(
        label,
        output,
        left,
        right,
        (value, value),
        kind="fused_cp_ffn_projective_pair",
    )


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


def _validate_block_source(
    block: ChiTransformerBlock,
    positions: Tensor,
) -> tuple[int, int, int]:
    if block.training or not block.residual:
        raise ValueError("implicit production kernel needs an eval-mode residual block")
    if not isinstance(block.attn, BilinearAttention) or block.attn.qk_norm != "rational":
        raise ValueError("implicit production kernel needs rational bilinear attention")
    if not isinstance(block.ffn, BilinearFFN):
        raise ValueError("implicit production kernel needs a BilinearFFN")
    if not isinstance(block.rbn_attn, RationalNorm) or not isinstance(block.rbn_ffn, RationalNorm):
        raise ValueError("both block pre-norms must be Padé RationalNorm modules")
    if positions.ndim != 2 or positions.shape[1] != block.attn.dim:
        raise ValueError("position tensor must have shape (tokens, width)")
    norms = tuple(module for module in block.modules() if isinstance(module, RationalNorm))
    if len(norms) != 6:
        raise ValueError("unchanged block must expose exactly six Padé module sites")
    for module in norms:
        if module.variant != "pade" or int(module.deg) != 2 or not bool(module.initialized):
            raise ValueError("every Padé site must be degree two and initialized")
    tensors = tuple(block.parameters()) + tuple(block.buffers()) + (positions,)
    floating = [tensor for tensor in tensors if tensor.is_floating_point()]
    if any(tensor.dtype != torch.float64 for tensor in floating):
        raise ValueError("implicit direct-RQ kernel requires float64 source tensors")
    devices = {tensor.device for tensor in floating}
    if len(devices) != 1:
        raise ValueError("implicit source tensors must share one device")
    return int(positions.shape[0]), block.attn.dim, block.attn.n_heads


@torch.no_grad()
def compile_implicit_projective_block_boundary(
    block: ChiTransformerBlock,
    positions: Tensor,
    *,
    selected_token: int | None = None,
    mask: Tensor | None = None,
) -> ImplicitBlockOracle:
    token_count, dimension, heads = _validate_block_source(block, positions)
    like = block.attn.wo.weight
    selected = token_count - 1 if selected_token is None else int(selected_token)
    if not 0 <= selected < token_count:
        raise ValueError("selected token is out of range")
    if mask is None:
        fixed_mask = (
            causal_mask(token_count, dtype=like.dtype, device=like.device)
            if block.attn.causal
            else like.new_ones(token_count, token_count)
        )
    else:
        fixed_mask = mask.to(dtype=like.dtype, device=like.device)
    if fixed_mask.shape != (token_count, token_count):
        raise ValueError("fixed attention mask has the wrong shape")
    if not bool(((fixed_mask == 0) | (fixed_mask == 1)).all()):
        raise ValueError("fixed attention mask must be binary")
    visible_sources = [
        source for source in range(token_count) if float(fixed_mask[selected, source]) == 1.0
    ]
    if not visible_sources:
        raise ValueError("selected attention row has no visible source")

    telemetry = MaterializationTelemetry()
    builder = _Builder(like, telemetry)
    raw = tuple(_identity_pair_leaf(builder, token, dimension) for token in range(token_count))
    identity = torch.eye(dimension, dtype=like.dtype, device=like.device)
    positioned = tuple(
        _affine_pair(builder, raw[token], identity, positions[token], f"position.token{token}")
        for token in range(token_count)
    )
    normalized = tuple(
        _pade_norm_pair(builder, value, block.rbn_attn, f"pre_attention.token{token}")
        for token, value in enumerate(positioned)
    )

    head_dimension = dimension // heads
    head_values: list[ImplicitNode] = []
    visible = like.new_tensor(float(len(visible_sources)))
    row_scale = visible.rsqrt() if block.attn.row_scale == "invsqrt" else visible.reciprocal()
    fixed_scale = row_scale / float(block.attn._score_denom)
    for head in range(heads):
        section = slice(head * head_dimension, (head + 1) * head_dimension)

        def project(linear: nn.Linear, token: int, name: str) -> ImplicitNode:
            return _affine_pair(
                builder,
                normalized[token],
                linear.weight[section],
                None if linear.bias is None else linear.bias[section],
                f"attention.head{head}.{name}.token{token}",
            )

        q1 = _pade_norm_pair(
            builder,
            project(block.attn.wq1, selected, "q1_affine"),
            block.attn.rn_q1,
            f"attention.head{head}.q1_norm.token{selected}",
        )
        q2 = _pade_norm_pair(
            builder,
            project(block.attn.wq2, selected, "q2_affine"),
            block.attn.rn_q2,
            f"attention.head{head}.q2_norm.token{selected}",
        )
        routes: list[ImplicitNode] = []
        for source in visible_sources:
            k1 = _pade_norm_pair(
                builder,
                project(block.attn.wk1, source, "k1_affine"),
                block.attn.rn_k1,
                f"attention.head{head}.k1_norm.token{source}",
            )
            k2 = _pade_norm_pair(
                builder,
                project(block.attn.wk2, source, "k2_affine"),
                block.attn.rn_k2,
                f"attention.head{head}.k2_norm.token{source}",
            )
            value = project(block.attn.wv, source, "value_affine")
            score1 = _dot_pair_vectors(
                builder, q1, k1, f"attention.head{head}.route{source}.score1"
            )
            score2 = _dot_pair_vectors(
                builder, q2, k2, f"attention.head{head}.route{source}.score2"
            )
            score = _multiply_pair_scalars(
                builder,
                score1,
                score2,
                f"attention.head{head}.route{source}.score_product",
            )
            route = _scale_pair_vector_by_scalar(
                builder,
                score,
                value,
                f"attention.head{head}.route{source}.score_times_value",
            )
            routes.append(
                _scale_pair_vector(
                    builder,
                    route,
                    fixed_scale,
                    f"attention.head{head}.route{source}.fixed_scale",
                )
            )
        head_value = routes[0]
        for route_index, route in enumerate(routes[1:], start=1):
            head_value = _add_pair_vectors(
                builder,
                head_value,
                route,
                f"attention.head{head}.route_add{route_index}",
            )
        head_values.append(head_value)

    attention = head_values[0]
    for head, head_value in enumerate(head_values[1:], start=1):
        attention = _concatenate_pair_vectors(
            builder, attention, head_value, f"attention.head_concat{head}"
        )
    attention = _affine_pair(
        builder,
        attention,
        block.attn.wo.weight,
        block.attn.wo.bias,
        "attention.output_affine",
    )
    attention = _scale_pair_vector(
        builder, attention, block.attn_gain, "attention.residual_gain"
    )
    after_attention = _add_pair_vectors(
        builder,
        positioned[selected],
        attention,
        "attention.residual_add",
    )
    ffn_input = _pade_norm_pair(builder, after_attention, block.rbn_ffn, "pre_ffn")
    ffn = _fused_cp_ffn_pair(builder, ffn_input, block.ffn, "ffn.fused_cp")
    ffn = _scale_pair_vector(builder, ffn, block.ffn_gain, "ffn.residual_gain")
    output = _add_pair_vectors(builder, after_attention, ffn, "ffn.residual_add")
    network = ImplicitProjectiveDAG(
        output,
        torch.eye(dimension + 1, dtype=like.dtype, device=like.device),
        0,
        token_count,
        dimension,
        selected,
        fixed_mask,
    )
    return ImplicitBlockOracle(
        network,
        block,
        positions.detach().clone(),
        telemetry,
        _norm_snapshot(block),
    )


@torch.no_grad()
def source_block_boundary(oracle: ImplicitBlockOracle, raw_input: Tensor) -> Tensor:
    value = raw_input + oracle.positions
    output = oracle.block(value, mask=oracle.network.mask, method="explicit")
    return output[:, oracle.network.selected_token]


@torch.no_grad()
def assert_norm_buffers_unchanged(oracle: ImplicitBlockOracle) -> None:
    current = dict(_norm_snapshot(oracle.block))
    for name, expected in oracle.norm_buffer_snapshot:
        if name not in current or not torch.equal(current[name], expected):
            raise RuntimeError(f"implicit compiler changed normalization buffer {name}")
