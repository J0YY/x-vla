"""Fail-closed export of a scalar-normalized χ-VLA to a polynomial graph.

The initial exporter supports one strict deployment class only: one χ-ViT
vision branch, bilinear attention and FFNs, calibrated frozen scalar RBN, no
phase branch, and a linear action head. Every deployed normalization scalar is
absorbed into its consumer weights. The standalone ViT final norm is a dead
path in ``ChiVLA._visual_tokens`` and is removed without being folded.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from xvla.models.vla import ChiVLA, VLAConfig
from xvla.models.vit import ChiViT
from xvla.nn.attention import BilinearAttention
from xvla.nn.bilinear import BilinearFFN
from xvla.nn.block import ChiTransformer, ChiTransformerBlock
from xvla.nn.normalization import HomotopyNorm, PerTokenRmsNorm, RationalNorm, RmsBatchNorm


Tensor = torch.Tensor


def _implementation_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "models/vit.py",
        "models/vla.py",
        "nn/attention.py",
        "nn/bilinear.py",
        "nn/block.py",
        "nn/normalization.py",
        "train/calibrate.py",
        "train/fold_vla.py",
    )
    digest = hashlib.sha256()
    for relative in relative_paths:
        digest.update(relative.encode("utf-8"))
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def _supported_model_dtype(model: nn.Module) -> torch.dtype:
    dtypes = {
        tensor.dtype
        for tensor in tuple(model.parameters()) + tuple(model.buffers())
        if tensor.is_floating_point() or tensor.is_complex()
    }
    if len(dtypes) != 1:
        raise ValueError(f"strict export requires one homogeneous floating dtype, got {dtypes}")
    dtype = next(iter(dtypes))
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("strict export supports only float32 and float64")
    return dtype


@dataclass(frozen=True)
class VLAFoldVerification:
    model: ChiVLA
    maximum_absolute_error: float
    maximum_relative_error: float
    remaining_norm_paths: tuple[str, ...]
    removed_dead_paths: tuple[str, ...]


def _strict_scale(norm: nn.Module, path: str) -> Tensor:
    if not isinstance(norm, RmsBatchNorm):
        raise TypeError(f"{path} must be RmsBatchNorm, got {type(norm).__name__}")
    if norm.calibrating or not norm.frozen or not bool(norm.initialized):
        raise ValueError(f"{path} must be calibrated, initialized, and frozen")
    scale = norm.scale.detach()
    if not bool(torch.isfinite(scale)) or float(scale.item()) <= 0:
        raise ValueError(f"{path} has an invalid fold scale")
    return scale


def _validate_supported(model: ChiVLA) -> None:
    if type(model) is not ChiVLA:
        raise TypeError("strict VLA export requires exactly ChiVLA")
    cfg = model.cfg
    required = {
        "vision_encoder": "vit",
        "dual_vision": False,
        "attn": "bilinear",
        "ffn": "bilinear",
        "norm": "scalar_rbn",
        "qk_norm": "scalar_rbn",
        "action_head": "linear",
        "n_phases": 0,
        "distill_teacher": False,
        "value_head": False,
    }
    for name, expected in required.items():
        actual = getattr(cfg, name)
        if actual != expected:
            raise ValueError(
                f"strict VLA export requires {name}={expected!r}, got {actual!r}"
            )
    with torch.device("meta"):
        expected_model = ChiVLA(copy.deepcopy(cfg))
    expected_signature = {
        name: (tuple(tensor.shape), tensor.is_floating_point(), tensor.is_complex())
        for name, tensor in expected_model.state_dict().items()
    }
    observed_signature = {
        name: (tuple(tensor.shape), tensor.is_floating_point(), tensor.is_complex())
        for name, tensor in model.state_dict().items()
    }
    if observed_signature != expected_signature or model.seq_len != expected_model.seq_len:
        raise ValueError("strict VLA state shape signature does not match config")
    if model.vision.use_cls is not False or model.vision.cfg != cfg.vit_config():
        raise ValueError("vision runtime metadata does not match strict VLA config")
    if model.vision.blocks.dim != cfg.vit_dim or model.vision.blocks.causal is not False:
        raise ValueError("vision transformer runtime metadata is inconsistent")
    if model.backbone.dim != cfg.dim or model.backbone.causal is not True:
        raise ValueError("backbone transformer runtime metadata is inconsistent")
    expected_root_children = {
        "vision",
        "vis_proj",
        "tok_emb",
        "state_proj",
        "embodiment_emb",
        "backbone",
        "norm_out",
        "action_head",
    }
    observed_root_children = set(dict(model.named_children()))
    if observed_root_children != expected_root_children:
        raise TypeError(
            "strict VLA module inventory mismatch: "
            f"expected {sorted(expected_root_children)}, got {sorted(observed_root_children)}"
        )
    exact_modules = (
        (model.vision, ChiViT, "vision"),
        (model.vision.patch, nn.Conv2d, "vision.patch"),
        (model.vision.blocks, ChiTransformer, "vision.blocks"),
        (model.vision.head, nn.Linear, "vision.head"),
        (model.vis_proj, nn.Linear, "vis_proj"),
        (model.tok_emb, nn.Embedding, "tok_emb"),
        (model.state_proj, nn.Linear, "state_proj"),
        (model.embodiment_emb, nn.Embedding, "embodiment_emb"),
        (model.backbone, ChiTransformer, "backbone"),
        (model.action_head, nn.Linear, "action_head"),
    )
    for module, expected_type, path in exact_modules:
        if type(module) is not expected_type:
            raise TypeError(f"{path} must be exactly {expected_type.__name__}")
    if set(dict(model.vision.named_children())) != {"patch", "blocks", "norm_out", "head"}:
        raise TypeError("vision child-module inventory mismatch")
    if type(model.vision.blocks.blocks) is not nn.ModuleList or type(model.backbone.blocks) is not nn.ModuleList:
        raise TypeError("strict VLA transformer blocks must be exactly nn.ModuleList")
    if len(model.vision.blocks.blocks) != cfg.vit_layers:
        raise ValueError("vision block count does not match config")
    if len(model.backbone.blocks) != cfg.n_layers:
        raise ValueError("backbone block count does not match config")
    for stack_path, transformer in (("vision.blocks", model.vision.blocks), ("backbone", model.backbone)):
        for index, block in enumerate(transformer.blocks):
            prefix = f"{stack_path}.{index}"
            if type(block) is not ChiTransformerBlock:
                raise TypeError(f"{prefix} must be exactly ChiTransformerBlock")
            if set(dict(block.named_children())) != {"rbn_attn", "attn", "rbn_ffn", "ffn"}:
                raise TypeError(f"{prefix} child-module inventory mismatch")
            if type(block.attn) is not BilinearAttention:
                raise TypeError(f"{prefix}.attn must be exactly BilinearAttention")
            if type(block.ffn) is not BilinearFFN:
                raise TypeError(f"{prefix}.ffn must be exactly BilinearFFN")
            if type(block.rbn_attn) is not RmsBatchNorm or type(block.rbn_ffn) is not RmsBatchNorm:
                raise TypeError(f"{prefix} pre-branch norms must be exactly RmsBatchNorm")
            attention_children = {
                "wq1", "wk1", "wq2", "wk2", "wv", "wo",
                "rbn_q1", "rbn_k1", "rbn_q2", "rbn_k2", "rbn_v",
            }
            if set(dict(block.attn.named_children())) != attention_children:
                raise TypeError(f"{prefix}.attn child-module inventory mismatch")
            for projection_name in ("wq1", "wk1", "wq2", "wk2", "wv", "wo"):
                projection = getattr(block.attn, projection_name)
                if type(projection) is not nn.Linear or projection.bias is None:
                    raise TypeError(f"{prefix}.attn.{projection_name} must be an affine nn.Linear")
            for norm_name in ("rbn_q1", "rbn_k1", "rbn_q2", "rbn_k2", "rbn_v"):
                if type(getattr(block.attn, norm_name)) is not RmsBatchNorm:
                    raise TypeError(f"{prefix}.attn.{norm_name} must be exactly RmsBatchNorm")
            if set(dict(block.ffn.named_children())) != {"left", "right", "down"}:
                raise TypeError(f"{prefix}.ffn child-module inventory mismatch")
            for projection_name in ("left", "right", "down"):
                projection = getattr(block.ffn, projection_name)
                if type(projection) is not nn.Linear:
                    raise TypeError(f"{prefix}.ffn.{projection_name} must be exactly nn.Linear")
            if block.ffn.left.bias is None or block.ffn.right.bias is None or block.ffn.down.bias is not None:
                raise TypeError(f"{prefix}.ffn bias contract is unsupported")
    if type(model.norm_out) is not RmsBatchNorm or type(model.vision.norm_out) is not RmsBatchNorm:
        raise TypeError("source output norms must be exactly RmsBatchNorm")
    if type(model.action_head) is not nn.Linear:
        raise TypeError("strict VLA export requires action_head to be exactly nn.Linear")
    for path, module in model.named_modules():
        if isinstance(module, BilinearAttention):
            if module.qk_norm != "scalar_rbn":
                raise ValueError(f"{path}.qk_norm must be 'scalar_rbn'")
            if module.row_scale not in ("invsqrt", "inv"):
                raise ValueError(f"{path}.row_scale is unsupported")
            if module.score_scale not in ("d_h", "d_h2"):
                raise ValueError(f"{path}.score_scale is unsupported")
            if module.dim <= 0 or module.n_heads <= 0 or module.dim % module.n_heads:
                raise ValueError(f"{path} has an invalid head partition")
            expected_denominator = module.head_dim ** (
                2 if module.score_scale == "d_h2" else 1
            )
            if module._score_denom != expected_denominator:
                raise ValueError(f"{path} has inconsistent score denominator")
            if not math.isfinite(module.norm_eps) or module.norm_eps <= 0:
                raise ValueError(f"{path}.norm_eps must be finite and positive")
            expected_causal = path.startswith("backbone.")
            if module.causal is not expected_causal:
                raise ValueError(f"{path}.causal does not match its deployed stack")
    for name, buffer in model.named_buffers():
        if name.startswith("vision.norm_out."):
            continue
        if (buffer.is_floating_point() or buffer.is_complex()) and not bool(torch.isfinite(buffer).all()):
            raise ValueError(f"source VLA live buffer {name} contains nonfinite values")


@torch.no_grad()
def _fold_transformer(transformer: nn.Module, path: str) -> None:
    for index, block in enumerate(transformer.blocks):
        prefix = f"{path}.blocks.{index}"
        if not isinstance(block.attn, BilinearAttention):
            raise TypeError(f"{prefix}.attn must be BilinearAttention")
        if not isinstance(block.ffn, BilinearFFN):
            raise TypeError(f"{prefix}.ffn must be BilinearFFN")
        attention_scale = _strict_scale(block.rbn_attn, f"{prefix}.rbn_attn")
        for linear in (
            block.attn.wq1,
            block.attn.wk1,
            block.attn.wq2,
            block.attn.wk2,
            block.attn.wv,
        ):
            linear.weight.div_(attention_scale)
        block.rbn_attn = nn.Identity()

        if block.attn.qk_norm != "scalar_rbn":
            raise ValueError(f"{prefix}.attn qk_norm must be 'scalar_rbn'")
        branch_pairs = (
            (block.attn.wq1, "rbn_q1"),
            (block.attn.wk1, "rbn_k1"),
            (block.attn.wq2, "rbn_q2"),
            (block.attn.wk2, "rbn_k2"),
            (block.attn.wv, "rbn_v"),
        )
        for linear, name in branch_pairs:
            norm = getattr(block.attn, name)
            branch_scale = _strict_scale(norm, f"{prefix}.attn.{name}")
            linear.weight.div_(branch_scale)
            linear.bias.div_(branch_scale)
            delattr(block.attn, name)
        block.attn.qk_norm = "none"

        ffn_scale = _strict_scale(block.rbn_ffn, f"{prefix}.rbn_ffn")
        block.ffn.left.weight.div_(ffn_scale)
        block.ffn.right.weight.div_(ffn_scale)
        block.rbn_ffn = nn.Identity()


def remaining_rbn_paths(model: nn.Module) -> tuple[str, ...]:
    return tuple(name for name, module in model.named_modules() if isinstance(module, RmsBatchNorm))


def forbidden_normalization_paths(model: nn.Module) -> tuple[str, ...]:
    forbidden = (RmsBatchNorm, HomotopyNorm, PerTokenRmsNorm, RationalNorm)
    return tuple(name for name, module in model.named_modules() if isinstance(module, forbidden))


def validate_strict_vla_inputs(model: ChiVLA, batch: Mapping[str, Tensor]) -> None:
    """Validate the fixed-shape input family named by a strict export package."""

    required = {"img", "instr_ids", "state", "embodiment_id"}
    if set(batch) != required:
        raise ValueError(f"strict VLA input keys must be exactly {sorted(required)}")
    img, instr, state, embodiment = (batch[name] for name in ("img", "instr_ids", "state", "embodiment_id"))
    if img.ndim != 4 or img.shape[1:] != (3, model.cfg.image_size, model.cfg.image_size):
        raise ValueError("img has the wrong strict export shape")
    batch_size = img.shape[0]
    if batch_size <= 0:
        raise ValueError("strict VLA inputs require a positive batch size")
    if instr.shape != (batch_size, model.cfg.max_instr_len):
        raise ValueError("instr_ids has the wrong strict export shape")
    if state.shape != (batch_size, model.cfg.state_dim):
        raise ValueError("state has the wrong strict export shape")
    if embodiment.shape != (batch_size,):
        raise ValueError("embodiment_id has the wrong strict export shape")
    dtype = _supported_model_dtype(model)
    if img.dtype != dtype or state.dtype != dtype:
        raise TypeError("continuous inputs must match the strict model floating dtype")
    if not bool(torch.isfinite(img).all()) or not bool(torch.isfinite(state).all()):
        raise ValueError("continuous strict VLA inputs must be finite")
    if instr.dtype != torch.int64 or embodiment.dtype != torch.int64:
        raise TypeError("categorical inputs must use torch.int64")
    if bool((instr < 0).any()) or bool((instr >= model.cfg.vocab_size).any()):
        raise ValueError("instr_ids is outside the embedding vocabulary")
    if bool((embodiment < 0).any()) or bool((embodiment >= model.cfg.n_embodiments).any()):
        raise ValueError("embodiment_id is outside the embedding vocabulary")


def _strict_input_schema(model: ChiVLA, parameter_dtype: str) -> dict[str, Any]:
    return {
        "img": ["batch", 3, model.cfg.image_size, model.cfg.image_size],
        "instr_ids": ["batch", model.cfg.max_instr_len],
        "state": ["batch", model.cfg.state_dim],
        "embodiment_id": ["batch"],
        "continuous_dtype": parameter_dtype,
        "categorical_dtype": "int64",
        "token_count": model.seq_len,
        "validator": "xvla.train.fold_vla.validate_strict_vla_inputs",
    }


def _metadata_sha256(package: Mapping[str, Any]) -> str:
    excluded = {"source_state", "folded_state", "metadata_sha256"}
    payload = {key: package[key] for key in sorted(package) if key not in excluded}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@torch.no_grad()
def fold_vla(model: ChiVLA) -> ChiVLA:
    """Return a normalization-free polynomial copy of a supported χ-VLA."""

    _validate_supported(model)
    _supported_model_dtype(model)
    if model.training:
        raise ValueError("strict VLA export requires model.eval()")
    for parameter in model.parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise ValueError("source VLA contains nonfinite parameters")
    folded = copy.deepcopy(model).eval()
    _fold_transformer(folded.vision.blocks, "vision.blocks")
    # ChiVLA calls vision.features(apply_final_norm=False), so this module is a
    # dead standalone-classifier path. It may legitimately be uninitialized.
    if not isinstance(folded.vision.norm_out, RmsBatchNorm):
        raise TypeError("vision.norm_out must be RmsBatchNorm in the strict source model")
    folded.vision.norm_out = nn.Identity()

    _fold_transformer(folded.backbone, "backbone")
    output_scale = _strict_scale(folded.norm_out, "norm_out")
    folded.action_head.weight.div_(output_scale)
    folded.norm_out = nn.Identity()
    folded.cfg.norm = "none"
    folded.cfg.qk_norm = "none"
    folded.vision.cfg.norm = "none"
    folded.vision.cfg.qk_norm = "none"
    leftovers = forbidden_normalization_paths(folded)
    if leftovers:
        raise AssertionError(f"normalization modules remain after strict VLA fold: {leftovers}")
    for parameter in folded.parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise ValueError("folded VLA contains nonfinite parameters")
    return folded


@torch.no_grad()
def verify_vla_fold(
    model: ChiVLA,
    batch: Mapping[str, Tensor],
) -> VLAFoldVerification:
    """Fold a model and compare deployed actions on one validation batch."""

    required = ("img", "instr_ids", "state", "embodiment_id")
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"validation batch is missing: {', '.join(missing)}")
    if model.training:
        raise ValueError("strict VLA verification requires model.eval()")
    source = model
    validate_strict_vla_inputs(source, {name: batch[name] for name in required})
    folded = fold_vla(source).eval()
    arguments = {name: batch[name] for name in required}
    before, _ = source(**arguments)
    after, _ = folded(**arguments)
    difference = (after - before).abs()
    relative = torch.linalg.vector_norm(after - before) / torch.linalg.vector_norm(before).clamp_min(
        torch.finfo(before.dtype).tiny
    )
    return VLAFoldVerification(
        model=folded,
        maximum_absolute_error=float(difference.max().item()),
        maximum_relative_error=float(relative.item()),
        remaining_norm_paths=forbidden_normalization_paths(folded),
        removed_dead_paths=("vision.norm_out",),
    )


def _state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _snapshot_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _attention_runtime(model: ChiVLA) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "causal": module.causal,
            "row_scale": module.row_scale,
            "score_scale": module.score_scale,
            "score_denominator": module._score_denom,
            "norm_eps": module.norm_eps,
        }
        for name, module in model.named_modules()
        if isinstance(module, BilinearAttention)
    }


def _rbn_runtime(model: ChiVLA) -> dict[str, dict[str, float]]:
    return {
        name: {"eps": float(module.eps), "momentum": float(module.momentum)}
        for name, module in model.named_modules()
        if isinstance(module, RmsBatchNorm) and name != "vision.norm_out"
    }


def _restore_hashed_runtime(
    model: ChiVLA,
    attention_runtime: Mapping[str, Mapping[str, Any]],
    rbn_runtime: Mapping[str, Mapping[str, Any]],
) -> None:
    attentions = {
        name: module for name, module in model.named_modules() if isinstance(module, BilinearAttention)
    }
    if set(attentions) != set(attention_runtime):
        raise ValueError("attention runtime path inventory mismatch")
    for name, module in attentions.items():
        record = attention_runtime[name]
        if set(record) != {"causal", "row_scale", "score_scale", "score_denominator", "norm_eps"}:
            raise ValueError(f"{name} attention runtime schema mismatch")
        module.causal = record["causal"]
        module.row_scale = record["row_scale"]
        module.score_scale = record["score_scale"]
        module._score_denom = record["score_denominator"]
        module.norm_eps = record["norm_eps"]
    norms = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, RmsBatchNorm) and name != "vision.norm_out"
    }
    if set(norms) != set(rbn_runtime):
        raise ValueError("RBN runtime path inventory mismatch")
    for name, module in norms.items():
        record = rbn_runtime[name]
        if set(record) != {"eps", "momentum"}:
            raise ValueError(f"{name} RBN runtime schema mismatch")
        eps = float(record["eps"])
        momentum = float(record["momentum"])
        if not math.isfinite(eps) or eps <= 0 or not math.isfinite(momentum) or not 0 < momentum < 1:
            raise ValueError(f"{name} has invalid RBN runtime metadata")
        module.eps = eps
        module.momentum = momentum


@torch.no_grad()
def make_strict_vla_export_package(
    model: ChiVLA,
    *,
    source_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a reloadable calibrated-source plus folded-state export package."""

    _validate_supported(model)
    if model.training:
        raise ValueError("strict VLA export requires model.eval()")
    dtype = _supported_model_dtype(model)
    live_scales = {
        name: float(module.scale.detach().cpu().item())
        for name, module in model.named_modules()
        if isinstance(module, RmsBatchNorm) and name != "vision.norm_out"
    }
    # Validation is delegated to fold_vla, which checks every live path.
    folded = fold_vla(model)
    source_state = _snapshot_state(model)
    folded_state = _snapshot_state(folded)
    parameter_dtype = str(dtype).removeprefix("torch.")
    if source_checkpoint_sha256 is not None:
        if len(source_checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in source_checkpoint_sha256.lower()
        ):
            raise ValueError("source_checkpoint_sha256 must be 64 hexadecimal characters")
    package = {
        "schema": "xvla-strict-polynomial-export-v1",
        "source_config": asdict(model.cfg),
        "folded_config": asdict(folded.cfg),
        "source_state": source_state,
        "folded_state": folded_state,
        "source_state_sha256": _state_dict_sha256(source_state),
        "folded_state_sha256": _state_dict_sha256(folded_state),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "parameter_dtype": parameter_dtype,
        "folded_scales": live_scales,
        "attention_runtime": _attention_runtime(model),
        "rbn_runtime": _rbn_runtime(model),
        "removed_dead_paths": ["vision.norm_out"],
        "dead_paths_outside_deployed_graph": ["vision.norm_out", "vision.head"],
        "decomposition_class": "normalization_free_strict_polynomial_program",
        "topology_status": (
            "eligible_input_to_later_clone_unfolded_topology_compilation_for_the_declared_fixed_shape"
        ),
        "implementation_sha256": _implementation_sha256(),
        "torch_version": torch.__version__,
        "input_boundary": (
            "Polynomial graph begins after discrete embedding lookup unless categorical "
            "inputs are represented explicitly as one-hot vectors."
        ),
        "preprocessing_boundary": (
            "Image scaling and any state or action normalization are external to this "
            "package and must be recorded with a deployment checkpoint."
        ),
        "input_schema": _strict_input_schema(model, parameter_dtype),
        "decode_boundary": "Linear continuous action output, with no in-graph decode.",
    }
    package["metadata_sha256"] = _metadata_sha256(package)
    return package


@torch.no_grad()
def load_strict_vla_export_package(package: Mapping[str, Any]) -> ChiVLA:
    """Validate and load a strict export package into a folded χ-VLA."""

    if package.get("schema") != "xvla-strict-polynomial-export-v1":
        raise ValueError("unsupported strict VLA export schema")
    dtype_name = package.get("parameter_dtype")
    supported_dtypes = {"float32": torch.float32, "float64": torch.float64}
    if dtype_name not in supported_dtypes:
        raise ValueError(f"unsupported strict export parameter dtype {dtype_name!r}")
    dtype = supported_dtypes[dtype_name]
    required_keys = {
        "source_config",
        "folded_config",
        "source_state",
        "folded_state",
        "source_state_sha256",
        "folded_state_sha256",
        "folded_scales",
        "attention_runtime",
        "rbn_runtime",
        "implementation_sha256",
        "torch_version",
        "input_schema",
        "metadata_sha256",
    }
    missing = sorted(required_keys - set(package))
    if missing:
        raise ValueError(f"strict export package is missing keys: {missing}")
    if package["metadata_sha256"] != _metadata_sha256(package):
        raise ValueError("strict export metadata hash mismatch")
    if package["implementation_sha256"] != _implementation_sha256():
        raise ValueError("strict export implementation hash mismatch")
    if package["torch_version"] != torch.__version__:
        raise ValueError("strict export torch version mismatch")
    if _state_dict_sha256(package["source_state"]) != package["source_state_sha256"]:
        raise ValueError("source state hash mismatch")
    if _state_dict_sha256(package["folded_state"]) != package["folded_state_sha256"]:
        raise ValueError("folded state hash mismatch")
    source = ChiVLA(VLAConfig(**dict(package["source_config"]))).to(dtype=dtype).eval()
    _supported_model_dtype(source)
    if package["input_schema"] != _strict_input_schema(source, dtype_name):
        raise ValueError("strict export input schema mismatch")
    source.load_state_dict(package["source_state"], strict=True)
    if _state_dict_sha256(source.state_dict()) != package["source_state_sha256"]:
        raise ValueError("source state hash mismatch")
    for name, module in source.named_modules():
        if isinstance(module, RmsBatchNorm) and name != "vision.norm_out":
            module.frozen = True
            module.calibrating = False
    _restore_hashed_runtime(source, package["attention_runtime"], package["rbn_runtime"])
    if _attention_runtime(source) != package.get("attention_runtime"):
        raise ValueError("attention runtime metadata mismatch")
    observed_scales = {
        name: float(module.scale.detach().cpu().item())
        for name, module in source.named_modules()
        if isinstance(module, RmsBatchNorm) and name != "vision.norm_out"
    }
    if observed_scales != package.get("folded_scales"):
        raise ValueError("folded scale metadata mismatch")
    folded = fold_vla(source)
    if _state_dict_sha256(folded.state_dict()) != package["folded_state_sha256"]:
        raise ValueError("deterministic folded state hash mismatch")
    if asdict(folded.cfg) != dict(package["folded_config"]):
        raise ValueError("folded config metadata mismatch")
    expected = ChiVLA(VLAConfig(**dict(package["folded_config"]))).to(dtype=dtype).eval()
    expected.load_state_dict(package["folded_state"], strict=True)
    folded.load_state_dict(expected.state_dict(), strict=True)
    return folded
