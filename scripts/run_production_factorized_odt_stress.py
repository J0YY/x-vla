#!/usr/bin/env python3
"""Compile and replay a production χ-VLA as a bounded operator-form TN descriptor.

This stress test deliberately does not claim quotient-aware RQ closure.  It proves
that the capable Padé checkpoint can be represented and replayed at production
shape without deleting vision, residuals, or rational normalization, and without
materializing a dense attention core or a route table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from xvla.models.vit import ChiViT
from xvla.models.vla import ChiVLA, VLAConfig
from xvla.nn.attention import BilinearAttention, causal_mask
from xvla.nn.bilinear import BilinearFFN
from xvla.nn.normalization import RationalNorm


EXPECTED_CHECKPOINT_SHA256 = "96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9"
DTYPE = torch.float64


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = expected.norm().clamp_min(torch.finfo(expected.dtype).tiny)
    return float(((actual - expected).norm() / scale).item())


def maximum_absolute_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).abs().max().item())


def production_config(vocab_size: int) -> VLAConfig:
    return VLAConfig(
        image_size=64,
        patch_size=8,
        vit_dim=192,
        vit_layers=4,
        vit_heads=8,
        vit_ffn_rank=576,
        vocab_size=vocab_size,
        max_instr_len=32,
        state_dim=8,
        n_embodiments=1,
        dim=384,
        n_layers=8,
        n_heads=12,
        ffn_rank=1152,
        attn="bilinear",
        vit_attn="bilinear",
        ffn="bilinear",
        norm="rational",
        qk_norm="rational",
        residual=True,
        vit_residual=True,
        action_horizon=8,
        action_dim=7,
        action_head="linear",
    )


def cloned(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().cpu()
    if value.is_floating_point():
        value = value.to(DTYPE)
    return value.clone()


def compile_linear(module: nn.Linear) -> dict[str, Any]:
    return {
        "kind": "affine",
        "weight": cloned(module.weight),
        "bias": None if module.bias is None else cloned(module.bias),
    }


def compile_norm(module: RationalNorm, label: str) -> dict[str, Any]:
    if not isinstance(module, RationalNorm) or module.variant != "pade":
        raise RuntimeError(f"{label} is not a deployed Padé RationalNorm")
    if module.training:
        raise RuntimeError(f"{label} must be in eval mode")
    if not bool(module.initialized):
        raise RuntimeError(f"{label} is not initialized")
    if module.deg != 2 or module.pa.numel() != 3 or module.pb.numel() != 3:
        raise RuntimeError(f"{label} is not a degree-2 over degree-2 Padé node")
    if not bool(torch.isfinite(module.running_ms)) or float(module.running_ms) <= 0.0:
        raise RuntimeError(f"{label} has an invalid running mean-square")
    if not bool(torch.isfinite(module.pa).all() and torch.isfinite(module.pb).all()):
        raise RuntimeError(f"{label} has nonfinite coefficients")
    return {
        "kind": "projective_pade_norm",
        "label": label,
        "degree": int(module.deg),
        "eps": float(module.eps),
        "running_ms": cloned(module.running_ms),
        "pa": cloned(module.pa),
        "pb": cloned(module.pb),
        "pair_degree_upper_bound": {"numerator": 5, "denominator": 4},
    }


def compile_attention(module: BilinearAttention, label: str) -> dict[str, Any]:
    if not isinstance(module, BilinearAttention) or module.qk_norm != "rational":
        raise RuntimeError(f"{label} is not bilinear attention with rational QK norms")
    return {
        "kind": "factorized_bilinear_attention",
        "label": label,
        "dim": int(module.dim),
        "heads": int(module.n_heads),
        "head_dim": int(module.head_dim),
        "causal": bool(module.causal),
        "row_scale": module.row_scale,
        "score_denominator": float(module._score_denom),
        "linears": {
            name: compile_linear(getattr(module, name))
            for name in ("wq1", "wk1", "wq2", "wk2", "wv", "wo")
        },
        "qk_norms": {
            name: compile_norm(getattr(module, name), f"{label}.{name}")
            for name in ("rn_q1", "rn_k1", "rn_q2", "rn_k2")
        },
        "dense_head_core": None,
        "dense_output_assembly": None,
        "route_table": None,
    }


def compile_ffn(module: BilinearFFN, label: str) -> dict[str, Any]:
    if not isinstance(module, BilinearFFN) or module.down.bias is not None:
        raise RuntimeError(f"{label} is not a bias-free-down BilinearFFN")
    return {
        "kind": "cp_bilinear_ffn",
        "label": label,
        "dim": int(module.dim),
        "rank": int(module.rank),
        "left": compile_linear(module.left),
        "right": compile_linear(module.right),
        "down": compile_linear(module.down),
        "dense_core": None,
    }


def compile_block(block: nn.Module, label: str, token_count: int) -> dict[str, Any]:
    if not bool(block.residual):
        raise RuntimeError(f"{label} residual route is disabled")
    attention = compile_attention(block.attn, f"{label}.attention")
    visible_pairs = token_count * (token_count + 1) // 2 if attention["causal"] else token_count**2
    route_count = visible_pairs * attention["heads"]
    return {
        "kind": "residual_rational_attention_ffn_block",
        "label": label,
        "token_count": token_count,
        "arities": [5, 2],
        "residuals": {"attention": True, "ffn": True},
        "pre_attention_norm": compile_norm(block.rbn_attn, f"{label}.pre_attention_norm"),
        "attention": attention,
        "attention_gain": cloned(block.attn_gain).reshape(()),
        "pre_ffn_norm": compile_norm(block.rbn_ffn, f"{label}.pre_ffn_norm"),
        "ffn": compile_ffn(block.ffn, f"{label}.ffn"),
        "ffn_gain": cloned(block.ffn_gain).reshape(()),
        "route_count": route_count,
        "route_feature_dimension": route_count * attention["head_dim"],
        "route_representation": "implicit fixed-mask index generator",
        "dense_attention_core_elements": attention["dim"] * (attention["dim"] + 1) ** 5,
        "dense_whole_state_attention_elements": (1 + token_count * attention["dim"]) ** 6,
    }


def compile_policy(model: ChiVLA) -> dict[str, Any]:
    if not isinstance(model.vision, ChiViT) or model.cfg.vision_encoder != "vit":
        raise RuntimeError("production stress requires the capable ViT checkpoint")
    if model.cfg.dual_vision or model.cfg.n_phases != 0:
        raise RuntimeError("production stress requires one vision branch and no phase tokens")
    if model.seq_len != 107 or model.vision.cfg.num_patches != 64:
        raise RuntimeError("checkpoint token geometry differs from Nvision=64 and Njoint=107")
    if len(model.vision.blocks.blocks) != 4 or len(model.backbone.blocks) != 8:
        raise RuntimeError("checkpoint depth differs from four vision and eight joint blocks")
    if not isinstance(model.action_head, nn.Linear) or model.cfg.action_head != "linear":
        raise RuntimeError("checkpoint must use the capable deterministic linear action head")
    patch = model.vision.patch
    descriptor = {
        "kind": "production_factorized_rational_vla_descriptor",
        "patch": {
            "kind": "strided_affine_patch_projection",
            "weight": cloned(patch.weight),
            "bias": cloned(patch.bias),
            "stride": tuple(int(value) for value in patch.stride),
            "padding": tuple(int(value) for value in patch.padding),
        },
        "vision_position": cloned(model.vision.pos_emb),
        "vision_blocks": [
            compile_block(block, f"vision.blocks.{index}", 64)
            for index, block in enumerate(model.vision.blocks.blocks)
        ],
        "visual_projection": compile_linear(model.vis_proj),
        "bos": cloned(model.bos),
        "token_embedding": cloned(model.tok_emb.weight),
        "state_projection": compile_linear(model.state_proj),
        "embodiment_embedding": cloned(model.embodiment_emb.weight),
        "action_queries": cloned(model.action_queries),
        "joint_position": cloned(model.pos_emb),
        "joint_blocks": [
            compile_block(block, f"backbone.blocks.{index}", 107)
            for index, block in enumerate(model.backbone.blocks)
        ],
        "final_norm": compile_norm(model.norm_out, "norm_out"),
        "action_head": compile_linear(model.action_head),
        "inactive_checkpoint_paths": {
            "vision.norm_out": "not called by ChiVLA._visual_tokens",
            "vision.head": "standalone classification head is outside the VLA path",
        },
        "canonical_rq_status": (
            "not_run: quotient-aware factorized RQ and environment propagation are the next stage"
        ),
    }
    return descriptor


def descriptor_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from descriptor_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from descriptor_tensors(child)


def tensor_bytes(value: Any) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in descriptor_tensors(value))


def linear_forward(descriptor: dict[str, Any], value: torch.Tensor) -> torch.Tensor:
    return F.linear(value, descriptor["weight"], descriptor["bias"])


def norm_forward(descriptor: dict[str, Any], value: torch.Tensor) -> torch.Tensor:
    mean_square = value.square().mean(dim=-1, keepdim=True) + descriptor["eps"]
    scale = descriptor["running_ms"].clamp_min(1e-12)
    normalized_square = mean_square / scale
    numerator = sum(
        descriptor["pa"][degree] * normalized_square**degree
        for degree in range(descriptor["degree"] + 1)
    )
    denominator = sum(
        descriptor["pb"][degree] * normalized_square**degree
        for degree in range(descriptor["degree"] + 1)
    )
    if not bool(torch.isfinite(denominator).all()) or bool((denominator == 0).any()):
        raise RuntimeError(f"{descriptor['label']} produced an invalid denominator")
    return value * (numerator / denominator) * torch.rsqrt(scale)


def attention_forward(
    descriptor: dict[str, Any], value: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    batch, tokens, dimension = value.shape
    heads = descriptor["heads"]
    head_dim = descriptor["head_dim"]

    def project(name: str) -> torch.Tensor:
        projected = linear_forward(descriptor["linears"][name], value)
        return projected.reshape(batch, tokens, heads, head_dim).transpose(1, 2)

    q1 = norm_forward(descriptor["qk_norms"]["rn_q1"], project("wq1"))
    k1 = norm_forward(descriptor["qk_norms"]["rn_k1"], project("wk1"))
    q2 = norm_forward(descriptor["qk_norms"]["rn_q2"], project("wq2"))
    k2 = norm_forward(descriptor["qk_norms"]["rn_k2"], project("wk2"))
    projected_value = project("wv")
    if mask is None:
        fixed_mask = value.new_ones(tokens, tokens)
    else:
        fixed_mask = mask.to(dtype=value.dtype, device=value.device)
    first = q1 @ k1.transpose(-1, -2)
    second = q2 @ k2.transpose(-1, -2)
    pattern = (first * second) / descriptor["score_denominator"]
    pattern = pattern * fixed_mask
    visible = fixed_mask.sum(dim=-1).clamp_min(1.0)
    row_scale = visible.rsqrt() if descriptor["row_scale"] == "invsqrt" else visible.reciprocal()
    output = row_scale.view(1, 1, tokens, 1) * (pattern @ projected_value)
    output = output.transpose(1, 2).reshape(batch, tokens, dimension)
    return linear_forward(descriptor["linears"]["wo"], output)


def ffn_forward(descriptor: dict[str, Any], value: torch.Tensor) -> torch.Tensor:
    left = linear_forward(descriptor["left"], value)
    right = linear_forward(descriptor["right"], value)
    return linear_forward(descriptor["down"], left * right)


def block_forward(
    descriptor: dict[str, Any], value: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    normalized = norm_forward(descriptor["pre_attention_norm"], value)
    attention = attention_forward(descriptor["attention"], normalized, mask)
    after_attention = value + descriptor["attention_gain"] * attention
    normalized_ffn = norm_forward(descriptor["pre_ffn_norm"], after_attention)
    branch = ffn_forward(descriptor["ffn"], normalized_ffn)
    return after_attention + descriptor["ffn_gain"] * branch


def descriptor_forward(
    descriptor: dict[str, Any],
    image: torch.Tensor,
    instruction: torch.Tensor,
    state: torch.Tensor,
    embodiment: torch.Tensor,
) -> torch.Tensor:
    patch = descriptor["patch"]
    vision = F.conv2d(
        image,
        patch["weight"],
        patch["bias"],
        stride=patch["stride"],
        padding=patch["padding"],
    ).flatten(2).transpose(1, 2)
    vision = vision + descriptor["vision_position"]
    for block in descriptor["vision_blocks"]:
        vision = block_forward(block, vision, None)
    visual = linear_forward(descriptor["visual_projection"], vision)
    batch = image.shape[0]
    joint = torch.cat(
        (
            visual,
            descriptor["bos"].expand(batch, -1, -1),
            F.embedding(instruction, descriptor["token_embedding"]),
            linear_forward(descriptor["state_projection"], state)[:, None],
            F.embedding(embodiment, descriptor["embodiment_embedding"])[:, None],
            descriptor["action_queries"].expand(batch, -1, -1),
        ),
        dim=1,
    )
    joint = joint + descriptor["joint_position"][:, : joint.shape[1]]
    mask = causal_mask(joint.shape[1], dtype=joint.dtype, device=joint.device)
    for block in descriptor["joint_blocks"]:
        joint = block_forward(block, joint, mask)
    joint = norm_forward(descriptor["final_norm"], joint)
    return linear_forward(descriptor["action_head"], joint[:, -8:])


def state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(array.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def block_summaries(descriptor: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for stack in ("vision", "joint"):
        for index, block in enumerate(descriptor[f"{stack}_blocks"]):
            attention = block["attention"]
            rows.append(
                {
                    "stack": stack,
                    "index": index,
                    "tokens": block["token_count"],
                    "width": attention["dim"],
                    "heads": attention["heads"],
                    "head_dim": attention["head_dim"],
                    "causal": attention["causal"],
                    "arities": block["arities"],
                    "residuals": block["residuals"],
                    "pade_sites": 6,
                    "route_count": block["route_count"],
                    "route_feature_dimension": block["route_feature_dimension"],
                    "ffn_cp_rank": block["ffn"]["rank"],
                    "factor_bytes": tensor_bytes(block),
                    "dense_attention_core_elements_avoided": block[
                        "dense_attention_core_elements"
                    ],
                    "dense_whole_state_attention_elements_avoided": block[
                        "dense_whole_state_attention_elements"
                    ],
                    "dense_core_materialized": False,
                    "dense_output_assembly_materialized": False,
                    "route_table_materialized": False,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result {args.output}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    started = time.perf_counter()
    checkpoint_sha256 = file_sha256(args.checkpoint)
    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise RuntimeError("checkpoint SHA-256 differs from the pinned capable checkpoint")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vocab_size = int(state["tok_emb.weight"].shape[0])
    model = ChiVLA(production_config(vocab_size)).double().eval()
    model.load_state_dict(state, strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise RuntimeError("checkpoint contains a nonfinite tensor")
    load_seconds = time.perf_counter() - started
    before_digest = state_digest(model)

    compile_started = time.perf_counter()
    descriptor = compile_policy(model)
    compile_seconds = time.perf_counter() - compile_started
    rows = block_summaries(descriptor)

    generator = torch.Generator().manual_seed(args.seed)
    image = torch.randn(args.batch_size, 3, 64, 64, generator=generator, dtype=DTYPE)
    instruction = torch.randint(
        0, vocab_size, (args.batch_size, 32), generator=generator
    )
    state_input = torch.randn(args.batch_size, 8, generator=generator, dtype=DTYPE)
    embodiment = torch.zeros(args.batch_size, dtype=torch.long)
    replay_started = time.perf_counter()
    with torch.inference_mode():
        direct = model(image, instruction, state_input, embodiment)[0]
        replay = descriptor_forward(
            descriptor, image, instruction, state_input, embodiment
        )
    replay_seconds = time.perf_counter() - replay_started
    after_digest = state_digest(model)

    active_norms = []
    for name, module in model.named_modules():
        if isinstance(module, RationalNorm) and name != "vision.norm_out":
            active_norms.append(name)
    vision_rows = [row for row in rows if row["stack"] == "vision"]
    joint_rows = [row for row in rows if row["stack"] == "joint"]
    direct_error = relative_error(replay, direct)
    absolute_error = maximum_absolute_error(replay, direct)
    all_descriptor_tensors = list(descriptor_tensors(descriptor))
    maximum_tensor_elements = max(tensor.numel() for tensor in all_descriptor_tensors)
    gates = {
        "pinned_capable_checkpoint": checkpoint_sha256 == args.expected_checkpoint_sha256,
        "full_vision_inventory": len(vision_rows) == 4,
        "full_joint_inventory": len(joint_rows) == 8,
        "six_pade_sites_per_block": all(row["pade_sites"] == 6 for row in rows),
        "all_73_active_pade_sites_present": len(active_norms) == 73,
        "all_residuals_preserved": all(
            row["residuals"] == {"attention": True, "ffn": True} for row in rows
        ),
        "vision_routes_exact": all(row["route_count"] == 32768 for row in vision_rows),
        "joint_routes_exact": all(row["route_count"] == 69336 for row in joint_rows),
        "no_dense_core_or_route_table": all(
            not row["dense_core_materialized"]
            and not row["dense_output_assembly_materialized"]
            and not row["route_table_materialized"]
            for row in rows
        ),
        "bounded_descriptor_tensors": maximum_tensor_elements < 1_000_000,
        "sampled_full_policy_replay": direct_error < 2e-10 and absolute_error < 2e-10,
        "source_buffers_unchanged": before_digest == after_digest,
        "finite_actions": bool(torch.isfinite(direct).all() and torch.isfinite(replay).all()),
    }
    result = {
        "schema": "xvla-production-factorized-odt-compile-stress-v1",
        "object_kind": "operator_form_projective_tensor_network_descriptor",
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "configuration": {
            "vision": {"tokens": 64, "width": 192, "heads": 8, "blocks": 4},
            "joint": {"tokens": 107, "width": 384, "heads": 12, "blocks": 8},
            "dtype": "float64 replay from checkpoint weights",
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "inventory": {
            "active_pade_sites": len(active_norms),
            "inactive_pade_sites": ["vision.norm_out"],
            "vision_route_count_total": sum(row["route_count"] for row in vision_rows),
            "joint_route_count_total": sum(row["route_count"] for row in joint_rows),
            "all_attention_route_count_total": sum(row["route_count"] for row in rows),
            "factor_bytes": tensor_bytes(descriptor),
            "maximum_descriptor_tensor_elements": maximum_tensor_elements,
            "descriptor_tensor_count": len(all_descriptor_tensors),
            "blocks": rows,
        },
        "sampled_descriptor_replay": {
            "action_shape": list(direct.shape),
            "relative_error": direct_error,
            "maximum_absolute_error": absolute_error,
            "direct_norm": float(direct.norm().item()),
            "replay_norm": float(replay.norm().item()),
        },
        "timings_seconds": {
            "load": load_seconds,
            "compile_descriptor": compile_seconds,
            "full_policy_direct_plus_descriptor_replay": replay_seconds,
        },
        "peak_rss_mb": peak_rss_mb(),
        "source_state_sha256_before": before_digest,
        "source_state_sha256_after": after_digest,
        "canonical_rq_status": descriptor["canonical_rq_status"],
        "claim_boundary": (
            "This is a complete production-shape factorized compile and sampled replay of the "
            "pinned capable Padé checkpoint. It is not quotient-aware global RQ closure. The "
            "separate dense and tiny projective oracles test canonical messages and exact RQ."
        ),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n")
    print(encoded)
    if not result["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
