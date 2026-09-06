"""Full-policy input semantics for the canonical shared-DAG ODT compiler.

The decisive boundary starts at fixed, non-overlapping image patches, categorical
instruction and embodiment coordinates, and the continuous robot state.  Learned
position vectors, BOS, and action queries are fixed checkpoint constants.  This
module keeps those heterogeneous source spaces separate.  In particular, it never
pads a small source into the joint model width.

The compiler itself is added below these source helpers once the heterogeneous
direct-RQ primitives are available.  Keeping the source round trip independent is
intentional: production replay must be checked against ``ChiVLA.forward`` rather
than against a second hand-written implementation of the policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xvla.models.vit import ChiViT
from xvla.models.vla import ChiVLA
from xvla.nn.attention import BilinearAttention
from xvla.nn.bilinear import BilinearFFN
from xvla.nn.normalization import RationalNorm
from xvla.nn.product_routing import ProductRoutingHead
from xvla.nn.attention import causal_mask
from xvla.train.odt_engine_v2.compiler import (
    _Builder,
    _add_pair_vectors,
    _affine_pair,
    _concatenate_pair_vectors,
    _pade_norm_pair,
    _scale_pair_vector,
)
from xvla.train.odt_engine_v2.graph import (
    _validate_network,
    _walk_unique,
    evaluate_boundary_quotient,
)
from xvla.train.odt_engine_v2.types import (
    ImplicitNode,
    ImplicitProjectiveDAG,
    MaterializationTelemetry,
    PhysicalSourceSpec,
    ProjectiveConstant,
    ProjectiveValue,
)
from xvla.train.implicit_sparse_projective_odt_all_tokens import compile_block_tokens


LINEAR_FULL_VLA_CLAIM_BOUNDARY = (
    "Exact shared syntactic projective DAG from fixed non-overlapping patch "
    "coordinates, categorical instruction coordinates, robot state, and the "
    "checkpoint's fixed singleton embodiment through the unchanged single-branch "
    "ChiViT, visual projection, causal "
    "joint backbone, final Pade RationalNorm, and deterministic linear action "
    "head. The sole observable is the ordered action_horizon by action_dimension "
    "quotient. This is a clone-unfolded coefficient object, not compression and "
    "not an identified-variable or categorical-simplex intrinsic metric."
)

PRODUCT_COMPONENTS_FULL_VLA_CLAIM_BOUNDARY = (
    "Exact shared syntactic projective DAG from fixed non-overlapping patch "
    "coordinates, categorical instruction coordinates, robot state, and the "
    "checkpoint's fixed singleton embodiment through the unchanged single-branch "
    "ChiViT, visual projection, causal joint backbone, final Pade RationalNorm, "
    "pooled action-query state, and every BilinearFFN center, factor, and gate in "
    "the deployed ProductRoutingHead. The sole tensor observable contains all "
    "center, factor, and gate coordinates. Only the deterministic gate sign and "
    "signed factor sum remain in the unchanged out-of-graph controller decode."
)

PRODUCT_FIXED_ROUTE_FULL_VLA_CLAIM_BOUNDARY = (
    "Exact shared syntactic projective DAG from fixed non-overlapping patch "
    "coordinates, categorical instruction coordinates, robot state, and the "
    "checkpoint's fixed singleton embodiment through the unchanged single-branch "
    "ChiViT, visual projection, causal joint backbone, final Pade RationalNorm, "
    "pooled action-query state, and the deployed ProductRoutingHead center and "
    "factors for one externally fixed sign route. The sole observable is the "
    "ordered action_horizon by action_dimension quotient for that route."
)

# Backward-compatible name for the pinned linear production runner.
FULL_VLA_CLAIM_BOUNDARY = LINEAR_FULL_VLA_CLAIM_BOUNDARY

LINEAR_ACTION_OBSERVABLE = "linear_action"
PRODUCT_COMPONENTS_OBSERVABLE = "product_components"
PRODUCT_FIXED_ROUTE_OBSERVABLE = "product_fixed_route_action"


@dataclass(frozen=True)
class FullVLALayout:
    patch_count: int
    patch_width: int
    instruction_count: int
    vocabulary_width: int
    state_width: int
    embodiment_width: int
    action_horizon: int
    action_dimension: int
    vision_width: int
    joint_width: int
    joint_token_count: int
    bos_index: int
    instruction_start: int
    state_index: int
    embodiment_index: int
    action_start: int

    @property
    def physical_source_count(self) -> int:
        return self.patch_count + self.instruction_count + 1 + int(
            not self.embodiment_is_fixed
        )

    @property
    def embodiment_is_fixed(self) -> bool:
        return self.embodiment_width == 1

    @property
    def observable_width(self) -> int:
        return self.action_horizon * self.action_dimension


@dataclass(frozen=True)
class FullVLAPhysicalBatch:
    """A batch over heterogeneous physical source spaces, without padding."""

    patches: Tensor
    instruction_one_hot: Tensor
    state: Tensor
    embodiment_one_hot: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.patches.shape[0])


@dataclass
class FullVLAImplicitOracle:
    network: ImplicitProjectiveDAG
    model: ChiVLA
    layout: FullVLALayout
    telemetry: MaterializationTelemetry
    active_norm_snapshot: tuple[tuple[str, Tensor], ...]
    observable_kind: str
    observable_euclidean_width: int
    product_signs: tuple[float, ...] | None


def _require_real_finite(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point() or value.is_complex():
        raise TypeError(f"{name} must be a real floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def full_vla_layout(model: ChiVLA) -> FullVLALayout:
    """Validate the unchanged strict-policy topology and return exact token indices."""

    if model.training:
        raise ValueError("full VLA compilation requires an eval-mode source model")
    if not isinstance(model.vision, ChiViT) or model.cfg.vision_encoder != "vit":
        raise ValueError("full VLA compilation requires the unchanged ChiViT front end")
    if model.cfg.dual_vision or model.cfg.n_phases != 0:
        raise ValueError("full VLA compilation currently requires one vision branch and no phase tokens")
    if model.cfg.action_head == "linear":
        if not isinstance(getattr(model, "action_head", None), nn.Linear):
            raise ValueError("linear action-head configuration differs from the source model")
    elif model.cfg.action_head == "product":
        head = getattr(model, "product_head", None)
        if not isinstance(head, ProductRoutingHead):
            raise ValueError("product action-head configuration differs from the source model")
        if (
            head.dim != model.cfg.dim
            or head.action_dim != model.cfg.action_dim
            or head.horizon != model.cfg.action_horizon
            or head.m != model.cfg.action_horizon * model.cfg.action_dim
            or head.G != model.cfg.n_factors
            or len(head.factors) != head.G
            or len(head.gates) != head.G
        ):
            raise ValueError("ProductRoutingHead geometry differs from the source configuration")
        modules = (("center", head.center, head.m),) + tuple(
            (f"factor{index}", module, head.m)
            for index, module in enumerate(head.factors)
        ) + tuple(
            (f"gate{index}", module, 1)
            for index, module in enumerate(head.gates)
        )
        for label, module, output_width in modules:
            if (
                not isinstance(module, BilinearFFN)
                or module.dim != head.dim
                or module.out_dim != output_width
            ):
                raise ValueError(f"ProductRoutingHead {label} is not the deployed bilinear core")
    else:
        raise ValueError("full VLA compilation supports only unchanged linear or product action heads")
    patch = model.vision.patch
    patch_size = int(model.cfg.patch_size)
    if (
        patch.groups != 1
        or patch.in_channels != model.cfg.vit_config().in_chans
        or patch.out_channels != model.cfg.vit_dim
        or tuple(patch.kernel_size) != (patch_size, patch_size)
        or tuple(patch.stride) != (patch_size, patch_size)
        or tuple(patch.padding) != (0, 0)
        or tuple(patch.dilation) != (1, 1)
    ):
        raise ValueError("full VLA patch sources require the checkpoint's disjoint strided patches")
    if model.cfg.image_size % patch_size:
        raise ValueError("image size must be divisible by patch size")
    if model.vision.use_cls:
        raise ValueError("the pinned VLA path has no vision class token")
    if len(model.vision.blocks.blocks) != model.cfg.vit_layers:
        raise ValueError("vision block count differs from the source configuration")
    if len(model.backbone.blocks) != model.cfg.n_layers:
        raise ValueError("joint block count differs from the source configuration")
    for label, blocks, causal in (
        ("vision", model.vision.blocks.blocks, False),
        ("joint", model.backbone.blocks, True),
    ):
        for index, block in enumerate(blocks):
            if not block.residual:
                raise ValueError(f"{label} block {index} removed the residual route")
            if not isinstance(block.attn, BilinearAttention) or block.attn.causal != causal:
                raise ValueError(f"{label} block {index} has the wrong attention topology")
            if block.attn.qk_norm != "rational" or not isinstance(block.ffn, BilinearFFN):
                raise ValueError(f"{label} block {index} is not the rational bilinear checkpoint block")
            norms = tuple(item for item in block.modules() if isinstance(item, RationalNorm))
            if len(norms) != 6 or any(item.variant != "pade" for item in norms):
                raise ValueError(f"{label} block {index} does not expose all six Pade sites")
            if any(int(item.deg) != 2 or not bool(item.initialized) for item in norms):
                raise ValueError(f"{label} block {index} has an unusable Pade site")
    if not isinstance(model.norm_out, RationalNorm) or model.norm_out.variant != "pade":
        raise ValueError("the active final policy norm is not Pade RationalNorm")
    if int(model.norm_out.deg) != 2 or not bool(model.norm_out.initialized):
        raise ValueError("the active final policy norm is not initialized degree two")

    grid = model.cfg.image_size // patch_size
    patch_count = grid * grid
    instruction_start = patch_count + 1
    state_index = instruction_start + model.cfg.max_instr_len
    embodiment_index = state_index + 1
    action_start = embodiment_index + 1
    expected_tokens = action_start + model.cfg.action_horizon
    if model.seq_len != expected_tokens or model.pos_emb.shape[1] != expected_tokens:
        raise ValueError("joint token geometry differs from ChiVLA._encode")
    if model.vision.pos_emb.shape[1] != patch_count:
        raise ValueError("vision positional geometry differs from the patch grid")
    return FullVLALayout(
        patch_count=patch_count,
        patch_width=patch.in_channels * patch_size * patch_size,
        instruction_count=model.cfg.max_instr_len,
        vocabulary_width=model.cfg.vocab_size,
        state_width=model.cfg.state_dim,
        embodiment_width=model.cfg.n_embodiments,
        action_horizon=model.cfg.action_horizon,
        action_dimension=model.cfg.action_dim,
        vision_width=model.cfg.vit_dim,
        joint_width=model.cfg.dim,
        joint_token_count=expected_tokens,
        bos_index=patch_count,
        instruction_start=instruction_start,
        state_index=state_index,
        embodiment_index=embodiment_index,
        action_start=action_start,
    )


def patchize_images(model: ChiVLA, images: Tensor) -> Tensor:
    """Extract patches in the exact order consumed by the checkpoint Conv2d."""

    layout = full_vla_layout(model)
    _require_real_finite(images, "images")
    channels = model.cfg.vit_config().in_chans
    image_size = model.cfg.image_size
    if images.ndim != 4 or tuple(images.shape[1:]) != (channels, image_size, image_size):
        raise ValueError("image batch has the wrong shape")
    patch_size = model.cfg.patch_size
    grid = image_size // patch_size
    result = (
        images.reshape(images.shape[0], channels, grid, patch_size, grid, patch_size)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(images.shape[0], layout.patch_count, layout.patch_width)
    )
    return result


def unpatchize_images(model: ChiVLA, patches: Tensor) -> Tensor:
    """Inverse of :func:`patchize_images`, used only for the source-model oracle."""

    layout = full_vla_layout(model)
    _require_real_finite(patches, "patches")
    if patches.ndim != 3 or tuple(patches.shape[1:]) != (
        layout.patch_count,
        layout.patch_width,
    ):
        raise ValueError("patch batch has the wrong shape")
    channels = model.cfg.vit_config().in_chans
    patch_size = model.cfg.patch_size
    grid = model.cfg.image_size // patch_size
    return (
        patches.reshape(patches.shape[0], grid, grid, channels, patch_size, patch_size)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(patches.shape[0], channels, model.cfg.image_size, model.cfg.image_size)
    )


def physical_batch_from_model_inputs(
    model: ChiVLA,
    images: Tensor,
    instruction_ids: Tensor,
    state: Tensor,
    embodiment_ids: Tensor,
) -> FullVLAPhysicalBatch:
    """Convert ordinary model inputs to exact heterogeneous ODT source coordinates."""

    layout = full_vla_layout(model)
    if instruction_ids.dtype != torch.long or instruction_ids.ndim != 2:
        raise TypeError("instruction ids must be a rank-two long tensor")
    if tuple(instruction_ids.shape[1:]) != (layout.instruction_count,):
        raise ValueError("instruction ids have the wrong length")
    if embodiment_ids.dtype != torch.long or embodiment_ids.ndim != 1:
        raise TypeError("embodiment ids must be a rank-one long tensor")
    _require_real_finite(state, "state")
    if state.ndim != 2 or state.shape[1] != layout.state_width:
        raise ValueError("state batch has the wrong shape")
    batch = images.shape[0]
    if instruction_ids.shape[0] != batch or state.shape[0] != batch or embodiment_ids.shape[0] != batch:
        raise ValueError("model input batch dimensions disagree")
    if bool(((instruction_ids < 0) | (instruction_ids >= layout.vocabulary_width)).any()):
        raise ValueError("instruction id is outside the checkpoint vocabulary")
    if bool(((embodiment_ids < 0) | (embodiment_ids >= layout.embodiment_width)).any()):
        raise ValueError("embodiment id is outside the checkpoint table")
    dtype = images.dtype
    return FullVLAPhysicalBatch(
        patchize_images(model, images),
        F.one_hot(instruction_ids, num_classes=layout.vocabulary_width).to(dtype=dtype),
        state,
        F.one_hot(embodiment_ids, num_classes=layout.embodiment_width).to(dtype=dtype),
    )


def validate_physical_batch(model: ChiVLA, batch: FullVLAPhysicalBatch) -> None:
    layout = full_vla_layout(model)
    values = (
        ("patches", batch.patches, (layout.patch_count, layout.patch_width)),
        (
            "instruction_one_hot",
            batch.instruction_one_hot,
            (layout.instruction_count, layout.vocabulary_width),
        ),
        ("state", batch.state, (layout.state_width,)),
        ("embodiment_one_hot", batch.embodiment_one_hot, (layout.embodiment_width,)),
    )
    batch_sizes: set[int] = set()
    dtypes: set[torch.dtype] = set()
    devices: set[torch.device] = set()
    for name, value, trailing in values:
        _require_real_finite(value, name)
        if tuple(value.shape[1:]) != trailing:
            raise ValueError(f"{name} has the wrong heterogeneous source shape")
        batch_sizes.add(int(value.shape[0]))
        dtypes.add(value.dtype)
        devices.add(value.device)
    if len(batch_sizes) != 1 or len(dtypes) != 1 or len(devices) != 1:
        raise ValueError("heterogeneous physical sources disagree on batch, dtype, or device")
    for name, value in (
        ("instruction_one_hot", batch.instruction_one_hot),
        ("embodiment_one_hot", batch.embodiment_one_hot),
    ):
        if not bool(((value == 0) | (value == 1)).all()):
            raise ValueError(f"{name} must be exactly binary for source-model replay")
        if not bool((value.sum(dim=-1) == 1).all()):
            raise ValueError(f"{name} must contain exactly one active category")


def physical_source_mapping(
    model: ChiVLA, batch: FullVLAPhysicalBatch
) -> dict[str, Tensor]:
    """Map the batch to stable heterogeneous leaf keys used by the shared DAG."""

    validate_physical_batch(model, batch)
    layout = full_vla_layout(model)
    result = {
        **{
            f"image.patch{index}": batch.patches[:, index]
            for index in range(layout.patch_count)
        },
        **{
            f"instruction.token{index}": batch.instruction_one_hot[:, index]
            for index in range(layout.instruction_count)
        },
        "state": batch.state,
    }
    if not layout.embodiment_is_fixed:
        result["embodiment"] = batch.embodiment_one_hot
    if len(result) != layout.physical_source_count:
        raise RuntimeError("heterogeneous source map has the wrong cardinality")
    return result


def _active_norm_snapshot(model: ChiVLA) -> tuple[tuple[str, Tensor], ...]:
    result: list[tuple[str, Tensor]] = []
    modules: list[tuple[str, nn.Module]] = []
    modules.extend(
        (f"vision.blocks.{index}", block)
        for index, block in enumerate(model.vision.blocks.blocks)
    )
    modules.extend(
        (f"backbone.blocks.{index}", block)
        for index, block in enumerate(model.backbone.blocks)
    )
    modules.append(("norm_out", model.norm_out))
    for prefix, owner in modules:
        for name, module in owner.named_modules():
            if not isinstance(module, RationalNorm):
                continue
            site = prefix if not name else f"{prefix}.{name}"
            result.extend(
                (
                    (f"{site}.running_ms", module.running_ms.detach().clone()),
                    (f"{site}.initialized", module.initialized.detach().clone()),
                    (f"{site}.pa", module.pa.detach().clone()),
                    (f"{site}.pb", module.pb.detach().clone()),
                )
            )
    return tuple(result)


def assert_full_vla_norm_buffers_unchanged(oracle: FullVLAImplicitOracle) -> None:
    current = dict(_active_norm_snapshot(oracle.model))
    for name, expected in oracle.active_norm_snapshot:
        if name not in current or not torch.equal(current[name], expected):
            raise RuntimeError(f"full VLA compiler changed active normalization buffer {name}")


def _require_node(value: ProjectiveValue, label: str) -> ImplicitNode:
    if isinstance(value, ProjectiveConstant):
        raise ValueError(f"{label} unexpectedly remained independent of every physical source")
    return value


def _affine_physical_pair(
    builder: _Builder,
    spec: PhysicalSourceSpec,
    weight: Tensor,
    bias: Tensor | None,
    label: str,
) -> ProjectiveValue:
    return _affine_pair(builder, builder.physical_pair(spec), weight, bias, label)


def _balanced_pair_concat(
    builder: _Builder,
    pairs: tuple[ProjectiveValue, ...],
    label_prefix: str,
) -> ImplicitNode:
    if not pairs:
        raise ValueError("the full VLA observable needs at least one projective vector")
    level: list[ProjectiveValue] = list(pairs)
    depth = 0
    while len(level) > 1:
        following: list[ProjectiveValue] = []
        for index in range(0, len(level), 2):
            if index + 1 == len(level):
                following.append(level[index])
            else:
                following.append(
                    _concatenate_pair_vectors(
                        builder,
                        level[index],
                        level[index + 1],
                        f"{label_prefix}.level{depth}.pair{index // 2}",
                    )
                )
        level = following
        depth += 1
    return _require_node(level[0], label_prefix)


def _balanced_pair_sum(
    builder: _Builder,
    values: tuple[ProjectiveValue, ...],
    label_prefix: str,
) -> ProjectiveValue:
    if not values:
        raise ValueError("projective sum needs at least one value")
    level = list(values)
    depth = 0
    while len(level) > 1:
        following: list[ProjectiveValue] = []
        for index in range(0, len(level), 2):
            if index + 1 == len(level):
                following.append(level[index])
            else:
                following.append(
                    _add_pair_vectors(
                        builder,
                        level[index],
                        level[index + 1],
                        f"{label_prefix}.level{depth}.pair{index // 2}",
                    )
                )
        level = following
        depth += 1
    return level[0]


def _bilinear_ffn_output_pair(
    builder: _Builder,
    value: ProjectiveValue,
    module: BilinearFFN,
    label: str,
) -> ProjectiveValue:
    """Compile a BilinearFFN whose output width need not equal its input width."""

    dimension = value.output_dimension - 1
    if module.dim != dimension:
        raise ValueError(f"{label} BilinearFFN input dimension mismatch")
    rank = module.rank + 1
    input_size = dimension + 1
    output_size = module.out_dim + 1
    left = builder.like.new_zeros(rank, input_size)
    right = builder.like.new_zeros(rank, input_size)
    left[:-1, :-1] = module.left.weight
    left[:-1, -1] = module.left.bias
    right[:-1, :-1] = module.right.weight
    right[:-1, -1] = module.right.bias
    left[-1, -1] = 1.0
    right[-1, -1] = 1.0
    output = builder.like.new_zeros(output_size, rank)
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


def _normalize_product_signs(
    model: ChiVLA, signs: Tensor | tuple[float, ...] | list[float]
) -> tuple[Tensor, tuple[float, ...]]:
    if model.cfg.action_head != "product":
        raise ValueError("fixed product signs require a ProductRoutingHead source model")
    like = model.product_head.center.down.weight
    value = torch.as_tensor(signs, dtype=like.dtype, device=like.device)
    if value.ndim != 1 or value.shape[0] != model.product_head.G:
        raise ValueError("fixed product signs must have shape (n_factors,)")
    if not bool(torch.isfinite(value).all()) or not bool(
        ((value == -1) | (value == 1)).all()
    ):
        raise ValueError("fixed product signs must contain only exact -1 or +1 entries")
    frozen = tuple(float(item) for item in value.detach().cpu().tolist())
    return value, frozen


@torch.no_grad()
def compile_full_vla_projective_dag(
    model: ChiVLA,
    *,
    product_signs: Tensor | tuple[float, ...] | list[float] | None = None,
) -> FullVLAImplicitOracle:
    """Compile the complete unchanged deterministic policy into one shared DAG."""

    layout = full_vla_layout(model)
    fixed_sign_values: Tensor | None = None
    frozen_signs: tuple[float, ...] | None = None
    if model.cfg.action_head == "linear":
        if product_signs is not None:
            raise ValueError("a linear action head cannot consume fixed product signs")
        like = model.action_head.weight
        observable_kind = LINEAR_ACTION_OBSERVABLE
        claim_boundary = LINEAR_FULL_VLA_CLAIM_BOUNDARY
    else:
        like = model.product_head.center.down.weight
        if product_signs is None:
            observable_kind = PRODUCT_COMPONENTS_OBSERVABLE
            claim_boundary = PRODUCT_COMPONENTS_FULL_VLA_CLAIM_BOUNDARY
        else:
            fixed_sign_values, frozen_signs = _normalize_product_signs(
                model, product_signs
            )
            observable_kind = PRODUCT_FIXED_ROUTE_OBSERVABLE
            claim_boundary = PRODUCT_FIXED_ROUTE_FULL_VLA_CLAIM_BOUNDARY
    floating = [
        value
        for value in tuple(model.parameters()) + tuple(model.buffers())
        if value.is_floating_point()
    ]
    if like.dtype != torch.float64 or any(value.dtype != like.dtype for value in floating):
        raise ValueError("full VLA canonical compilation requires float64 source tensors")
    if any(value.device != like.device for value in floating):
        raise ValueError("full VLA source tensors must share one device")

    telemetry = MaterializationTelemetry()
    builder = _Builder(like, telemetry)
    patch_weight = model.vision.patch.weight.reshape(
        layout.vision_width, layout.patch_width
    )
    vision_tokens: tuple[ProjectiveValue, ...] = tuple(
        _affine_physical_pair(
            builder,
            PhysicalSourceSpec(
                f"image.patch{token}",
                f"input.image.patch{token}",
                layout.patch_width,
            ),
            patch_weight,
            model.vision.patch.bias + model.vision.pos_emb[0, token],
            f"vision.patch_affine_with_position.token{token}",
        )
        for token in range(layout.patch_count)
    )
    vision_mask = like.new_ones(layout.patch_count, layout.patch_count)
    for block_index, block in enumerate(model.vision.blocks.blocks):
        vision_tokens = compile_block_tokens(
            builder,
            block,
            vision_tokens,
            vision_mask,
            f"vision.block{block_index}",
        )

    joint_tokens: list[ProjectiveValue] = []
    for token, value in enumerate(vision_tokens):
        joint_tokens.append(
            _affine_pair(
                builder,
                value,
                model.vis_proj.weight,
                model.vis_proj.bias + model.pos_emb[0, token],
                f"joint.visual_projection_with_position.token{token}",
            )
        )
    joint_tokens.append(
        builder.constant_pair(
            "joint.constant.bos_with_position",
            model.bos[0, 0] + model.pos_emb[0, layout.bos_index],
        )
    )
    embedding_weight = model.tok_emb.weight.T
    for token in range(layout.instruction_count):
        joint_index = layout.instruction_start + token
        joint_tokens.append(
            _affine_physical_pair(
                builder,
                PhysicalSourceSpec(
                    f"instruction.token{token}",
                    f"input.instruction.token{token}",
                    layout.vocabulary_width,
                ),
                embedding_weight,
                model.pos_emb[0, joint_index],
                f"joint.instruction_embedding_with_position.token{token}",
            )
        )
    joint_tokens.append(
        _affine_physical_pair(
            builder,
            PhysicalSourceSpec("state", "input.state", layout.state_width),
            model.state_proj.weight,
            model.state_proj.bias + model.pos_emb[0, layout.state_index],
            "joint.state_projection_with_position",
        )
    )
    if layout.embodiment_is_fixed:
        joint_tokens.append(
            builder.constant_pair(
                "joint.constant.singleton_embodiment_with_position",
                model.embodiment_emb.weight[0]
                + model.pos_emb[0, layout.embodiment_index],
            )
        )
    else:
        joint_tokens.append(
            _affine_physical_pair(
                builder,
                PhysicalSourceSpec(
                    "embodiment", "input.embodiment", layout.embodiment_width
                ),
                model.embodiment_emb.weight.T,
                model.pos_emb[0, layout.embodiment_index],
                "joint.embodiment_embedding_with_position",
            )
        )
    for action in range(layout.action_horizon):
        joint_index = layout.action_start + action
        joint_tokens.append(
            builder.constant_pair(
                f"joint.constant.action_query{action}_with_position",
                model.action_queries[0, action] + model.pos_emb[0, joint_index],
            )
        )
    if len(joint_tokens) != layout.joint_token_count:
        raise RuntimeError("compiled joint token order differs from ChiVLA._encode")

    joint_mask = causal_mask(
        layout.joint_token_count, dtype=like.dtype, device=like.device
    )
    action_indices = tuple(
        range(layout.action_start, layout.action_start + layout.action_horizon)
    )
    joint_values: tuple[ProjectiveValue, ...] = tuple(joint_tokens)
    for block_index, block in enumerate(model.backbone.blocks):
        joint_values = compile_block_tokens(
            builder,
            block,
            joint_values,
            joint_mask,
            f"joint.block{block_index}",
            output_tokens=(
                action_indices
                if block_index == len(model.backbone.blocks) - 1
                else None
            ),
        )
    if len(joint_values) != layout.action_horizon:
        raise RuntimeError("final joint block did not return every action-query token")

    normalized_pairs: list[ProjectiveValue] = []
    for action, value in enumerate(joint_values):
        normalized_pairs.append(
            _pade_norm_pair(
                builder, value, model.norm_out, f"policy.final_norm.action{action}"
            )
        )

    if model.cfg.action_head == "linear":
        action_pairs = tuple(
            _affine_pair(
                builder,
                normalized,
                model.action_head.weight,
                model.action_head.bias,
                f"policy.action_head.action{action}",
            )
            for action, normalized in enumerate(normalized_pairs)
        )
        root = _balanced_pair_concat(
            builder, action_pairs, "policy.action_concat"
        )
        observable_euclidean_width = layout.observable_width
    else:
        pooled_sum = _balanced_pair_sum(
            builder, tuple(normalized_pairs), "policy.product.action_query_sum"
        )
        pooled = _scale_pair_vector(
            builder,
            pooled_sum,
            1.0 / layout.action_horizon,
            "policy.product.action_query_mean",
        )
        head = model.product_head
        center = _bilinear_ffn_output_pair(
            builder, pooled, head.center, "policy.product.center"
        )
        factors = tuple(
            _bilinear_ffn_output_pair(
                builder, pooled, module, f"policy.product.factor{index}"
            )
            for index, module in enumerate(head.factors)
        )
        if fixed_sign_values is None:
            gates = tuple(
                _bilinear_ffn_output_pair(
                    builder, pooled, module, f"policy.product.gate{index}"
                )
                for index, module in enumerate(head.gates)
            )
            root = _balanced_pair_concat(
                builder,
                (center,) + factors + gates,
                "policy.product.component_concat",
            )
            observable_euclidean_width = head.m * (head.G + 1) + head.G
        else:
            signed_factors = tuple(
                _scale_pair_vector(
                    builder,
                    factor,
                    fixed_sign_values[index],
                    f"policy.product.fixed_route.factor{index}",
                )
                for index, factor in enumerate(factors)
            )
            fixed_action = _balanced_pair_sum(
                builder,
                (center,) + signed_factors,
                "policy.product.fixed_route.sum",
            )
            root = _require_node(fixed_action, "fixed product-route action observable")
            observable_euclidean_width = layout.observable_width

    observable_pair_width = observable_euclidean_width + 1
    if root.output_dimension != observable_pair_width:
        raise RuntimeError("full VLA root has the wrong projective width")
    network = ImplicitProjectiveDAG(
        root=root,
        head=torch.eye(observable_pair_width, dtype=like.dtype, device=like.device),
        head_binary_exponent=0,
        token_count=layout.joint_token_count,
        feature_dimension=layout.joint_width,
        selected_token=layout.action_start,
        mask=joint_mask,
        claim_boundary=claim_boundary,
        physical_sources=builder.physical_source_specs,
    )
    _validate_network(network)
    oracle = FullVLAImplicitOracle(
        network=network,
        model=model,
        layout=layout,
        telemetry=telemetry,
        active_norm_snapshot=_active_norm_snapshot(model),
        observable_kind=observable_kind,
        observable_euclidean_width=observable_euclidean_width,
        product_signs=frozen_signs,
    )
    assert_full_vla_norm_buffers_unchanged(oracle)
    return oracle


@torch.no_grad()
def evaluate_full_vla_observable(
    oracle: FullVLAImplicitOracle, batch: FullVLAPhysicalBatch
) -> Tensor:
    """Evaluate the flat quotient stored at the shared DAG's sole root."""

    return evaluate_boundary_quotient(
        oracle.network, physical_source_mapping(oracle.model, batch)
    )


@torch.no_grad()
def evaluate_full_vla_quotient(
    oracle: FullVLAImplicitOracle, batch: FullVLAPhysicalBatch
) -> Tensor:
    """Evaluate the ordered action chunk, including the unchanged product decode."""

    flat = evaluate_full_vla_observable(oracle, batch)
    if oracle.observable_kind == PRODUCT_COMPONENTS_OBSERVABLE:
        head = oracle.model.product_head
        center = flat[:, : head.m]
        factor_end = head.m * (head.G + 1)
        factors = flat[:, head.m : factor_end].reshape(
            batch.batch_size, head.G, head.m
        )
        gates = flat[:, factor_end:]
        if gates.shape[1] != head.G:
            raise RuntimeError("product component observable has the wrong gate width")
        signs = torch.where(
            gates > 0,
            torch.ones_like(gates),
            -torch.ones_like(gates),
        )
        flat = head.action_for_signs(center, factors, signs)
    elif oracle.observable_kind not in {
        LINEAR_ACTION_OBSERVABLE,
        PRODUCT_FIXED_ROUTE_OBSERVABLE,
    }:
        raise ValueError("unknown full VLA observable kind")
    return flat.reshape(
        batch.batch_size,
        oracle.layout.action_horizon,
        oracle.layout.action_dimension,
    )


def full_vla_structure_statistics(oracle: FullVLAImplicitOracle) -> dict[str, object]:
    nodes = _walk_unique(oracle.network.root)
    leaf_nodes = tuple(node for node in nodes if node.physical_source_key is not None)
    return {
        "unique_nodes": len(nodes),
        "edge_occurrences": sum(len(node.children) for node in nodes),
        "physical_source_count": len(oracle.network.physical_sources),
        "physical_leaf_count": len(leaf_nodes),
        "physical_source_widths": {
            spec.key: spec.width for spec in oracle.network.physical_sources
        },
        "maximum_physical_source_width": max(
            spec.width for spec in oracle.network.physical_sources
        ),
        "root_projective_width": oracle.network.root.output_dimension,
        "observable_kind": oracle.observable_kind,
        "observable_euclidean_width": oracle.observable_euclidean_width,
        "product_signs": oracle.product_signs,
        "active_pade_sites": len(oracle.active_norm_snapshot) // 4,
    }


@torch.no_grad()
def source_full_vla_observable(
    oracle: FullVLAImplicitOracle, batch: FullVLAPhysicalBatch
) -> Tensor:
    """Evaluate the matching source-model observable without compiler code reuse."""

    model = oracle.model
    validate_physical_batch(model, batch)
    if oracle.observable_kind == LINEAR_ACTION_OBSERVABLE:
        return source_full_vla_output(model, batch).reshape(batch.batch_size, -1)
    images = unpatchize_images(model, batch.patches)
    instruction_ids = batch.instruction_one_hot.argmax(dim=-1)
    embodiment_ids = batch.embodiment_one_hot.argmax(dim=-1)
    pooled = model.pooled_features(
        images, instruction_ids, batch.state, embodiment_ids
    )
    center, factors, gates = model.product_head(pooled)
    if oracle.observable_kind == PRODUCT_COMPONENTS_OBSERVABLE:
        return torch.cat(
            (center, factors.reshape(batch.batch_size, -1), gates), dim=-1
        )
    if oracle.observable_kind == PRODUCT_FIXED_ROUTE_OBSERVABLE:
        if oracle.product_signs is None:
            raise RuntimeError("fixed product-route oracle lost its signs")
        signs = pooled.new_tensor(oracle.product_signs).expand(batch.batch_size, -1)
        return model.product_head.action_for_signs(center, factors, signs)
    raise ValueError("unknown full VLA observable kind")


@torch.no_grad()
def source_full_vla_output(model: ChiVLA, batch: FullVLAPhysicalBatch) -> Tensor:
    """Evaluate the decisive oracle with the unchanged public model forward path."""

    validate_physical_batch(model, batch)
    images = unpatchize_images(model, batch.patches)
    instruction_ids = batch.instruction_one_hot.argmax(dim=-1)
    embodiment_ids = batch.embodiment_one_hot.argmax(dim=-1)
    actions, _ = model(images, instruction_ids, batch.state, embodiment_ids)
    return actions


__all__ = [
    "FULL_VLA_CLAIM_BOUNDARY",
    "LINEAR_FULL_VLA_CLAIM_BOUNDARY",
    "PRODUCT_COMPONENTS_FULL_VLA_CLAIM_BOUNDARY",
    "PRODUCT_FIXED_ROUTE_FULL_VLA_CLAIM_BOUNDARY",
    "LINEAR_ACTION_OBSERVABLE",
    "PRODUCT_COMPONENTS_OBSERVABLE",
    "PRODUCT_FIXED_ROUTE_OBSERVABLE",
    "FullVLALayout",
    "FullVLAPhysicalBatch",
    "FullVLAImplicitOracle",
    "assert_full_vla_norm_buffers_unchanged",
    "compile_full_vla_projective_dag",
    "evaluate_full_vla_observable",
    "evaluate_full_vla_quotient",
    "full_vla_layout",
    "full_vla_structure_statistics",
    "patchize_images",
    "physical_batch_from_model_inputs",
    "physical_source_mapping",
    "source_full_vla_observable",
    "source_full_vla_output",
    "unpatchize_images",
    "validate_physical_batch",
]
