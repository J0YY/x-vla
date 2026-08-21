#!/usr/bin/env python3
"""Frozen fixed-input certificate for the complete learned ViT VLA forward map.

The deployed reference is one unmodified PyTorch float32 model call. The independent
path starts from the same prepared model tensors and evaluates every learned operation
on the deployed path with NumPy float64. It never calls a learned module's forward
method. See VIT_FULL_LEARNED_FORWARD_PROTOCOL.md for the frozen claim boundary.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
import torch.nn as nn

from athena.libero_dataset_metadata import load_dataset_task_languages
from athena.run_conv_joint_block_certificate import (
    array64,
    block_identity,
    errors,
    linear_numpy,
    rational_norm_numpy,
    reconstruct_block,
)
from athena.run_vit_modality_contribution_audit import (
    FROZEN_CACHE,
    FROZEN_CAPABILITY_MANIFEST,
    FROZEN_CHECKPOINTS,
    FROZEN_DATASET_METADATA,
    FROZEN_PROVENANCE_BASENAME,
    FROZEN_PROVENANCE_JOB_ID,
    cache_record_identity,
    hash_array,
    tensorize_records,
    validate_capability_manifest,
    validate_dataset_metadata,
    validate_provenance,
    write_json,
)
from athena.run_xvla_experiment import (
    build_vocab,
    file_sha256,
    load_cache_statistics,
    load_suite,
    make_config,
    task_languages,
)
from xvla.models.vit import ChiViT
from xvla.models.vla import ChiVLA
from xvla.nn.attention import BilinearAttention, causal_mask
from xvla.nn.bilinear import BilinearFFN
from xvla.nn.normalization import RationalNorm


SCHEMA = "xvla-vit-full-learned-forward-certificate-v1"
PROTOCOL_ID = b"xvla-vit-full-learned-forward-certificate-v1\0"
FULL_TASK_QUOTAS = {str(task): 2 if task < 6 else 1 for task in range(10)}
SMOKE_TASK_QUOTAS = {str(task): 1 if task == 0 else 0 for task in range(10)}
STAGE_NAMES = (
    "patch_projection_and_vision_position",
    "vision_block_0",
    "vision_block_1",
    "vision_block_2",
    "vision_block_3",
    "visual_projection",
    "joint_input_embeddings",
    "joint_block_0",
    "joint_block_1",
    "joint_block_2",
    "joint_block_3",
    "joint_block_4",
    "joint_block_5",
    "joint_block_6",
    "joint_block_7",
    "final_rational_norm",
    "linear_action_head",
)
STAGE_SHAPES = {
    **{
        name: [64, 192]
        for name in (
            "patch_projection_and_vision_position",
            "vision_block_0",
            "vision_block_1",
            "vision_block_2",
            "vision_block_3",
        )
    },
    "visual_projection": [64, 384],
    **{
        name: [107, 384]
        for name in (
            "joint_input_embeddings",
            "joint_block_0",
            "joint_block_1",
            "joint_block_2",
            "joint_block_3",
            "joint_block_4",
            "joint_block_5",
            "joint_block_6",
            "joint_block_7",
            "final_rational_norm",
        )
    },
    "linear_action_head": [8, 7],
}
PROTOCOL = {
    "suite": "libero_object",
    "architecture": "chi",
    "vision_encoder": "vit",
    "resolution": 64,
    "action_horizon": 8,
    "action_dim": 7,
    "matmul_precision": "highest",
    "full_inputs_per_checkpoint": 16,
    "full_official_task_quotas": FULL_TASK_QUOTAS,
    "smoke_inputs_per_checkpoint": 1,
    "smoke_official_task_quotas": SMOKE_TASK_QUOTAS,
    "input_selection": (
        "Within each official task, lexicographically smallest "
        "SHA256(protocol_id || checkpoint_sha256 || canonical_cache_record_sha256), "
        "with fixed 2/1 full quotas and a one-input component-complete smoke"
    ),
    "prepared_input_boundary": (
        "exact float32 image and normalized-state tensors, integer instruction IDs, "
        "and embodiment ID presented to the learned model"
    ),
    "vision_patch_tokens": 64,
    "vision_width": 192,
    "vision_blocks": 4,
    "joint_sequence_length": 107,
    "joint_width": 384,
    "joint_blocks": 8,
    "stages": list(STAGE_NAMES),
    "stage_shapes": STAGE_SHAPES,
    "reference": "one unmodified deployed PyTorch float32 ChiVLA forward call",
    "reconstruction": (
        "sequential NumPy float64 evaluation from saved weights and buffers of patch "
        "projection, four vision blocks, all deployed learned embeddings and projections, "
        "eight joint blocks, final RationalNorm, and the linear action head"
    ),
    "end_to_end_action_relative_l2_gate": 1e-3,
    "gate_rationale": (
        "Allows accumulated float32-versus-float64 operation-order error across twelve "
        "sequential transformer blocks while remaining a strict action-output fidelity test"
    ),
}
EXPECTED_RUNTIME = {
    "gpu": "NVIDIA RTX A6000",
    "torch_version": "2.7.1+cu126",
    "numpy_version": "1.26.4",
    "cuda_version": "12.6",
    "matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": True,
    "deterministic_algorithms": True,
    "cublas_workspace_config": ":4096:8",
}
SOURCE_BUNDLE_RELATIVE = "athena/VIT_FULL_LEARNED_FORWARD_SOURCE_BUNDLE.sha256"
SOURCE_FILES = (
    "athena/run_vit_full_learned_forward_certificate.py",
    "athena/summarize_vit_full_learned_forward_certificate.py",
    "athena/VIT_FULL_LEARNED_FORWARD_PROTOCOL.md",
    "athena/run_vit_modality_contribution_audit.py",
    "athena/run_conv_joint_block_certificate.py",
    "athena/run_conv_joint_ffn_certificate.py",
    "athena/run_conv_joint_attention_certificate.py",
    "athena/libero_dataset_metadata.py",
    "athena/run_xvla_experiment.py",
    "xvla/models/vla.py",
    "xvla/models/vit.py",
    "xvla/nn/attention.py",
    "xvla/nn/baselines.py",
    "xvla/nn/bilinear.py",
    "xvla/nn/block.py",
    "xvla/nn/flow_action.py",
    "xvla/nn/homogeneous.py",
    "xvla/nn/normalization.py",
    "xvla/nn/product_routing.py",
    "xvla/nn/projector.py",
    "xvla/nn/quantile_action.py",
    "xvla/train/exact_odt_attention_proto.py",
    "athena/results/exact_attention_frozen_sources/xvla_models_vla.py.b64",
    "athena/results/exact_attention_frozen_sources/xvla_models_vit.py.b64",
    "athena/results/exact_attention_frozen_sources/xvla_nn_product_routing.py.b64",
)
SOURCE_SNAPSHOTS = {
    "xvla/models/vla.py": "athena/results/exact_attention_frozen_sources/xvla_models_vla.py.b64",
    "xvla/models/vit.py": "athena/results/exact_attention_frozen_sources/xvla_models_vit.py.b64",
    "xvla/nn/product_routing.py": "athena/results/exact_attention_frozen_sources/xvla_nn_product_routing.py.b64",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-seed", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--provenance-result", type=Path, required=True)
    parser.add_argument("--expected-provenance-job-id", required=True)
    parser.add_argument("--capability-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("strict_smoke", "full"), required=True)
    return parser.parse_args()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def source_hashes() -> dict[str, str]:
    root = repository_root()
    bundle_path = root / SOURCE_BUNDLE_RELATIVE
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Frozen source bundle is absent: {bundle_path}")
    declared = {}
    for line_number, line in enumerate(bundle_path.read_text().splitlines(), start=1):
        if not line:
            continue
        pieces = line.split("  ", 1)
        if len(pieces) != 2:
            raise RuntimeError(f"Malformed source bundle line {line_number}")
        expected, relative = pieces
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise RuntimeError(f"Malformed source SHA on bundle line {line_number}")
        if relative in declared:
            raise RuntimeError(f"Duplicate source bundle path: {relative}")
        declared[relative] = expected
    if set(declared) != set(SOURCE_FILES):
        raise RuntimeError("Frozen source bundle path set differs from SOURCE_FILES")
    result = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source file is absent: {path}")
        actual = file_sha256(path)
        if actual != declared[relative]:
            raise RuntimeError(f"Frozen source bundle SHA differs: {relative}")
        result[relative] = actual
    result[SOURCE_BUNDLE_RELATIVE] = file_sha256(bundle_path)
    for source_relative, snapshot_relative in SOURCE_SNAPSHOTS.items():
        snapshot_path = root / snapshot_relative
        try:
            encoded = "".join(snapshot_path.read_text(encoding="ascii").split())
            payload = base64.b64decode(encoded, validate=True)
        except (OSError, UnicodeError, binascii.Error) as exc:
            raise RuntimeError(f"Invalid source snapshot: {snapshot_relative}") from exc
        if hashlib.sha256(payload).hexdigest() != result[source_relative]:
            raise RuntimeError(f"Source snapshot does not bind live source: {source_relative}")
    return result


def runtime_identity() -> dict[str, Any]:
    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "python_packages": {
            package: importlib.metadata.version(package)
            for package in ("pillow", "huggingface-hub", "pandas")
        },
    }


def validate_runtime(runtime: dict[str, Any]) -> None:
    mismatches = [
        key for key, expected in EXPECTED_RUNTIME.items()
        if runtime.get(key) != expected
    ]
    if mismatches:
        raise RuntimeError("Frozen runtime differs at: " + ", ".join(mismatches))


def validate_frozen_inputs(
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    expected = FROZEN_CHECKPOINTS.get(args.checkpoint_seed)
    if expected is None:
        raise ValueError("Checkpoint seed must be 0, 1, or 2")
    errors_found = []
    if args.checkpoint.name != expected["basename"]:
        errors_found.append("checkpoint basename differs")
    if args.expected_checkpoint_sha256 != expected["sha256"]:
        errors_found.append("declared checkpoint SHA differs")
    if not args.checkpoint.is_file() or file_sha256(args.checkpoint) != expected["sha256"]:
        errors_found.append("live checkpoint SHA differs")
    if args.cache.name != FROZEN_CACHE["basename"]:
        errors_found.append("cache basename differs")
    if args.expected_cache_sha256 != FROZEN_CACHE["sha256"]:
        errors_found.append("declared cache SHA differs")
    if args.expected_provenance_job_id != FROZEN_PROVENANCE_JOB_ID:
        errors_found.append("provenance job identity differs")
    if args.provenance_result.name != FROZEN_PROVENANCE_BASENAME:
        errors_found.append("provenance result basename differs")
    if args.output.exists():
        errors_found.append("output already exists")
    if errors_found:
        raise RuntimeError("Frozen identity validation failed: " + "; ".join(errors_found))
    capability = validate_capability_manifest(args.capability_manifest)
    member_rates = [row["success_rate"] for row in capability["expected"]["members"]]
    if member_rates != [0.852, 0.808, 0.9]:
        raise RuntimeError("Capability member rates differ from the frozen manifest")
    provenance = validate_provenance(
        args.provenance_result, args.cache, args.expected_cache_sha256
    )
    return expected["sha256"], provenance, capability


def validated_official_tasks(
    capability: dict[str, Any], external_official_tasks: dict[int, str]
) -> dict[int, str]:
    frozen_task_list = capability.get("tasks")
    if not isinstance(frozen_task_list, list) or len(frozen_task_list) != 10:
        raise RuntimeError("Capability manifest does not contain ten frozen task strings")
    frozen_tasks = {
        task_index: str(instruction)
        for task_index, instruction in enumerate(frozen_task_list)
    }
    if external_official_tasks != frozen_tasks:
        raise RuntimeError("External LIBERO task strings differ from the capability manifest")
    return frozen_tasks


def rank_cache_inputs(
    records: list[Any], checkpoint_sha256: str, quotas: dict[str, int]
) -> list[dict[str, Any]]:
    mapping = FROZEN_DATASET_METADATA["dataset_to_official_task"]
    ranked: dict[int, list[dict[str, Any]]] = {task: [] for task in range(10)}
    for cache_row_index, record in enumerate(records):
        identity = cache_record_identity(record, cache_row_index, mapping)
        digest = hashlib.sha256()
        digest.update(PROTOCOL_ID)
        digest.update(checkpoint_sha256.encode("ascii") + b"\0")
        digest.update(identity["canonical_record_sha256"].encode("ascii"))
        identity["selection_sha256"] = digest.hexdigest()
        ranked[int(identity["official_task_index"])].append(identity)
    selected = []
    for official_task_index in range(10):
        rows = sorted(
            ranked[official_task_index],
            key=lambda row: (row["selection_sha256"], row["cache_row_index"]),
        )
        quota = int(quotas[str(official_task_index)])
        if len(rows) < quota:
            raise RuntimeError(f"Official task {official_task_index} lacks selected inputs")
        for within_task_rank, row in enumerate(rows[:quota]):
            row["selection_rank_within_official_task"] = within_task_rank
            selected.append(row)
    for selection_rank, row in enumerate(selected):
        row["selection_rank"] = selection_rank
    return selected


def tensor_identity(tensor: torch.Tensor) -> dict[str, Any]:
    array = tensor.detach().cpu().numpy()
    return {
        "shape": list(array.shape),
        "dtype": str(tensor.dtype),
        "sha256": hash_array(array),
    }


def state_dict_identity(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    tensors = {name: tensor_identity(state_dict[name]) for name in sorted(state_dict)}
    digest = hashlib.sha256()
    for name, identity in tensors.items():
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(json.dumps(identity, sort_keys=True).encode("ascii") + b"\0")
    return {
        "tensor_count": len(tensors),
        "scalar_count": sum(int(tensor.numel()) for tensor in state_dict.values()),
        "identity_sha256": digest.hexdigest(),
        "tensors": tensors,
    }


def validate_model(model: ChiVLA) -> dict[str, Any]:
    if not isinstance(model.vision, ChiViT) or model.cfg.vision_encoder != "vit":
        raise RuntimeError("Certificate requires the ViT policy")
    if model.cfg.dual_vision or model.cfg.n_phases != 0:
        raise RuntimeError("Certificate requires the single-vision no-phase policy")
    if model.cfg.action_head != "linear" or not isinstance(model.action_head, nn.Linear):
        raise RuntimeError("Certificate requires the deterministic linear action head")
    if model.vision.use_cls or model.vision.cfg.num_patches != 64:
        raise RuntimeError("Vision token layout differs")
    if len(model.vision.blocks.blocks) != PROTOCOL["vision_blocks"]:
        raise RuntimeError("Vision block count differs")
    if len(model.backbone.blocks) != PROTOCOL["joint_blocks"]:
        raise RuntimeError("Joint block count differs")
    if model.seq_len != PROTOCOL["joint_sequence_length"]:
        raise RuntimeError("Joint sequence length differs")
    if not isinstance(model.norm_out, RationalNorm):
        raise RuntimeError("Final norm is not RationalNorm")
    for stack_name, blocks in (
        ("vision", model.vision.blocks.blocks),
        ("joint", model.backbone.blocks),
    ):
        for block_index, block in enumerate(blocks):
            if not isinstance(block.rbn_attn, RationalNorm):
                raise RuntimeError(f"{stack_name} block {block_index} attention norm differs")
            if not isinstance(block.attn, BilinearAttention):
                raise RuntimeError(f"{stack_name} block {block_index} attention differs")
            if not isinstance(block.rbn_ffn, RationalNorm):
                raise RuntimeError(f"{stack_name} block {block_index} FFN norm differs")
            if not isinstance(block.ffn, BilinearFFN):
                raise RuntimeError(f"{stack_name} block {block_index} FFN differs")
            block_identity(block, block_index)
    patch = model.vision.patch
    if tuple(patch.kernel_size) != (8, 8) or tuple(patch.stride) != (8, 8):
        raise RuntimeError("Patch projection geometry differs")
    if patch.in_channels != 3 or patch.out_channels != PROTOCOL["vision_width"]:
        raise RuntimeError("Patch projection dimensions differ")
    if model.vis_proj.in_features != 192 or model.vis_proj.out_features != 384:
        raise RuntimeError("Visual projection dimensions differ")
    if model.action_head.in_features != 384 or model.action_head.out_features != 7:
        raise RuntimeError("Action head dimensions differ")
    return {
        "vision_blocks": [
            block_identity(block, index)
            for index, block in enumerate(model.vision.blocks.blocks)
        ],
        "joint_blocks": [
            block_identity(block, index)
            for index, block in enumerate(model.backbone.blocks)
        ],
        "unused_checkpoint_modules": {
            "vision.norm_out": "not called by ChiVLA._visual_tokens for the trained ViT path",
            "vision.head": "standalone image-classification head is not on the VLA path",
        },
    }


def make_model(vocab_size: int, stats: dict[str, Any], checkpoint: Path) -> ChiVLA:
    config = make_config(
        PROTOCOL["architecture"],
        vocab_size,
        stats["state_dim"],
        stats["action_dim"],
        PROTOCOL["resolution"],
        PROTOCOL["action_horizon"],
        vision_encoder=PROTOCOL["vision_encoder"],
    )
    model = ChiVLA(config).cuda()
    payload = torch.load(checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(payload, strict=True)
    model.eval()
    validate_model(model)
    return model


def patch_projection_numpy(image: np.ndarray, patch: nn.Conv2d) -> np.ndarray:
    if image.shape != (3, 64, 64):
        raise RuntimeError("Prepared image tensor has the wrong shape")
    weight = array64(patch.weight)
    bias = None if patch.bias is None else array64(patch.bias)
    kernel_height, kernel_width = patch.kernel_size
    stride_height, stride_width = patch.stride
    output_height = (image.shape[1] - kernel_height) // stride_height + 1
    output_width = (image.shape[2] - kernel_width) // stride_width + 1
    result = np.empty((output_height, output_width, patch.out_channels), dtype=np.float64)
    flattened_weight = weight.reshape(patch.out_channels, -1)
    for row in range(output_height):
        for column in range(output_width):
            patch_values = image[
                :,
                row * stride_height : row * stride_height + kernel_height,
                column * stride_width : column * stride_width + kernel_width,
            ].reshape(-1)
            result[row, column] = flattened_weight @ patch_values
            if bias is not None:
                result[row, column] += bias
    return result.reshape(output_height * output_width, patch.out_channels)


def stage_record(name: str, reference: np.ndarray, reconstruction: np.ndarray) -> dict[str, Any]:
    reference64 = np.asarray(reference, dtype=np.float64)
    reconstruction64 = np.asarray(reconstruction, dtype=np.float64)
    row = errors(reference64, reconstruction64)
    return {
        "name": name,
        "shape": list(reference64.shape),
        "deployed_float32_as_float64_sha256": hash_array(reference64),
        "reconstruction_float64_sha256": hash_array(reconstruction64),
        **row,
    }


def audit_input(
    model: ChiVLA,
    sample_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    input_identity: dict[str, Any],
) -> dict[str, Any]:
    image, instruction, state, embodiment = sample_tensors
    if image.shape != (1, 3, 64, 64) or instruction.shape != (1, 32):
        raise RuntimeError("Prepared model input shapes differ from the frozen protocol")
    stages = []
    with torch.inference_mode():
        deployed_patch_map = model.vision.patch(image)
        deployed_vision = deployed_patch_map.flatten(2).transpose(1, 2)
        deployed_vision = deployed_vision + model.vision.pos_emb
        reconstructed_vision = patch_projection_numpy(
            np.asarray(image[0].cpu().numpy(), dtype=np.float64), model.vision.patch
        ) + array64(model.vision.pos_emb[0])
        stages.append(
            stage_record(
                "patch_projection_and_vision_position",
                deployed_vision[0].cpu().numpy(),
                reconstructed_vision,
            )
        )

        vision_mask = np.ones((64, 64), dtype=np.float64)
        for block_index, block in enumerate(model.vision.blocks.blocks):
            deployed_vision = block(deployed_vision, mask=None, method="explicit")
            reconstructed_vision = reconstruct_block(
                block, reconstructed_vision, vision_mask
            )
            stages.append(
                stage_record(
                    f"vision_block_{block_index}",
                    deployed_vision[0].cpu().numpy(),
                    reconstructed_vision,
                )
            )

        deployed_visual = model.vis_proj(deployed_vision)
        reconstructed_visual = linear_numpy(reconstructed_vision, model.vis_proj)
        stages.append(
            stage_record(
                "visual_projection",
                deployed_visual[0].cpu().numpy(),
                reconstructed_visual,
            )
        )

        deployed_joint = torch.cat(
            [
                deployed_visual,
                model.bos.expand(1, -1, -1),
                model.tok_emb(instruction),
                model.state_proj(state)[:, None],
                model.embodiment_emb(embodiment)[:, None],
                model.action_queries.expand(1, -1, -1),
            ],
            dim=1,
        )
        deployed_joint = deployed_joint + model.pos_emb[:, : deployed_joint.shape[1]]
        instruction_ids = np.asarray(instruction[0].cpu().numpy(), dtype=np.int64)
        embodiment_ids = np.asarray(embodiment.cpu().numpy(), dtype=np.int64)
        reconstructed_joint = np.concatenate(
            [
                reconstructed_visual,
                array64(model.bos[0]),
                array64(model.tok_emb.weight)[instruction_ids],
                linear_numpy(np.asarray(state[0].cpu().numpy(), dtype=np.float64), model.state_proj)[None],
                array64(model.embodiment_emb.weight)[embodiment_ids],
                array64(model.action_queries[0]),
            ],
            axis=0,
        )
        reconstructed_joint = reconstructed_joint + array64(
            model.pos_emb[0, : reconstructed_joint.shape[0]]
        )
        if reconstructed_joint.shape != (107, 384):
            raise RuntimeError("Reconstructed joint input shape differs")
        stages.append(
            stage_record(
                "joint_input_embeddings",
                deployed_joint[0].cpu().numpy(),
                reconstructed_joint,
            )
        )

        deployed_mask = causal_mask(107, device=deployed_joint.device, dtype=deployed_joint.dtype)
        reconstructed_mask = np.tril(np.ones((107, 107), dtype=np.float64))
        for block_index, block in enumerate(model.backbone.blocks):
            deployed_joint = block(deployed_joint, mask=deployed_mask, method="explicit")
            reconstructed_joint = reconstruct_block(
                block, reconstructed_joint, reconstructed_mask
            )
            stages.append(
                stage_record(
                    f"joint_block_{block_index}",
                    deployed_joint[0].cpu().numpy(),
                    reconstructed_joint,
                )
            )

        deployed_normalized = model.norm_out(deployed_joint)
        reconstructed_normalized = rational_norm_numpy(
            reconstructed_joint, model.norm_out
        )
        stages.append(
            stage_record(
                "final_rational_norm",
                deployed_normalized[0].cpu().numpy(),
                reconstructed_normalized,
            )
        )

        deployed_action = model.action_head(
            deployed_normalized[:, -PROTOCOL["action_horizon"] :]
        )
        reconstructed_action = linear_numpy(
            reconstructed_normalized[-PROTOCOL["action_horizon"] :],
            model.action_head,
        )
        stages.append(
            stage_record(
                "linear_action_head",
                deployed_action[0].cpu().numpy(),
                reconstructed_action,
            )
        )
        direct_action, direct_loss = model(image, instruction, state, embodiment)
        if direct_loss is not None or not torch.equal(direct_action, deployed_action):
            raise RuntimeError("Manual deployed traversal differs from the model forward call")

    if tuple(row["name"] for row in stages) != STAGE_NAMES:
        raise RuntimeError("Stage traversal differs from the frozen protocol")
    if any(row["shape"] != STAGE_SHAPES[row["name"]] for row in stages):
        raise RuntimeError("Stage shape differs from the frozen protocol")
    deployed_action64 = np.asarray(deployed_action[0].cpu().numpy(), dtype=np.float64)
    reconstructed_action64 = np.asarray(reconstructed_action, dtype=np.float64)
    end_to_end = errors(deployed_action64, reconstructed_action64)
    return {
        "selection_rank": input_identity["selection_rank"],
        "selection_sha256": input_identity["selection_sha256"],
        "prepared_inputs": {
            "image_float32_sha256": hash_array(image[0].cpu().numpy()),
            "instruction_int64_sha256": hash_array(instruction[0].cpu().numpy()),
            "normalized_state_float32_sha256": hash_array(state[0].cpu().numpy()),
            "embodiment_int64_sha256": hash_array(embodiment.cpu().numpy()),
        },
        "manual_deployed_forward_exact": True,
        "stages": stages,
        "actions": {
            "shape": [8, 7],
            "deployed_float32_as_float64": deployed_action64.tolist(),
            "deployed_float32_as_float64_sha256": hash_array(deployed_action64),
            "reconstruction_float64": reconstructed_action64.tolist(),
            "reconstruction_float64_sha256": hash_array(reconstructed_action64),
        },
        "end_to_end_action": end_to_end,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    source_sha256 = source_hashes()
    checkpoint_sha256, provenance, capability = validate_frozen_inputs(args)
    checkpoint_file_sha256 = file_sha256(args.checkpoint)
    cache_file_sha256 = file_sha256(args.cache)
    provenance_file_sha256 = file_sha256(args.provenance_result)
    capability_file_sha256 = file_sha256(args.capability_manifest)

    if not torch.cuda.is_available():
        raise RuntimeError("The certificate runner requires an A6000 CUDA node")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_float32_matmul_precision(PROTOCOL["matmul_precision"])
    torch.use_deterministic_algorithms(True)
    runtime = runtime_identity()
    validate_runtime(runtime)

    with args.cache.open("rb") as handle:
        records = pickle.load(handle)
    if len(records) != FROZEN_CACHE["frames"]:
        raise RuntimeError("Live cache frame count differs")
    quotas = SMOKE_TASK_QUOTAS if args.mode == "strict_smoke" else FULL_TASK_QUOTAS
    selected = rank_cache_inputs(records, checkpoint_sha256, quotas)
    expected_count = (
        PROTOCOL["smoke_inputs_per_checkpoint"]
        if args.mode == "strict_smoke"
        else PROTOCOL["full_inputs_per_checkpoint"]
    )
    if len(selected) != expected_count:
        raise RuntimeError("Selected input count differs")

    suite = load_suite(PROTOCOL["suite"])
    external_official_tasks = task_languages(suite)
    official_tasks = validated_official_tasks(capability, external_official_tasks)
    cache_tasks, dataset_metadata = load_dataset_task_languages(
        PROTOCOL["suite"], official_tasks
    )
    validate_dataset_metadata(dataset_metadata)
    vocab, encode = build_vocab(cache_tasks)
    stats = load_cache_statistics(args.cache, PROTOCOL["action_horizon"])
    model = make_model(len(vocab), stats, args.checkpoint)
    component_identity = validate_model(model)
    checkpoint_state_identity = state_dict_identity(model.state_dict())
    tensors = tensorize_records(records, selected, encode, official_tasks, stats)

    input_rows = []
    evaluations = []
    for input_index, identity in enumerate(selected):
        instruction_text = official_tasks[int(identity["official_task_index"])]
        input_rows.append(
            {
                **identity,
                "instruction": instruction_text,
                "instruction_ids": encode(instruction_text),
            }
        )
        sample = tuple(tensor[input_index : input_index + 1] for tensor in tensors)
        row = audit_input(model, sample, identity)
        evaluations.append(row)
        print("VIT_FULL_LEARNED_FORWARD", json.dumps(row), flush=True)

    all_finite = all(
        bool(row["end_to_end_action"]["finite"])
        and all(bool(stage["finite"]) for stage in row["stages"])
        for row in evaluations
    )
    max_action_absolute_error = max(
        float(row["end_to_end_action"]["max_abs_error"]) for row in evaluations
    )
    max_action_relative_l2_error = max(
        float(row["end_to_end_action"]["relative_l2_error"]) for row in evaluations
    )
    max_stage_relative_l2_error = max(
        float(stage["relative_l2_error"])
        for row in evaluations
        for stage in row["stages"]
    )
    passed = (
        all_finite
        and all(row["manual_deployed_forward_exact"] is True for row in evaluations)
        and max_action_relative_l2_error
        <= PROTOCOL["end_to_end_action_relative_l2_gate"]
    )

    if source_hashes() != source_sha256:
        raise RuntimeError("Runner or imported source changed during the certificate")
    for label, path, expected_sha in (
        ("checkpoint", args.checkpoint, checkpoint_file_sha256),
        ("cache", args.cache, cache_file_sha256),
        ("provenance", args.provenance_result, provenance_file_sha256),
        ("capability manifest", args.capability_manifest, capability_file_sha256),
    ):
        if file_sha256(path) != expected_sha:
            raise RuntimeError(f"{label} changed during the certificate")
    if runtime_identity() != runtime:
        raise RuntimeError("Runtime identity changed during the certificate")

    result = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "mode": args.mode,
        "identity": {
            "checkpoint_seed": args.checkpoint_seed,
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_basename": args.checkpoint.name,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_state_identity": checkpoint_state_identity,
            "cache_path": str(args.cache),
            "cache_basename": args.cache.name,
            "cache_sha256": args.expected_cache_sha256,
            "cache_frames": len(records),
            "provenance_result": str(args.provenance_result),
            "provenance_result_sha256": provenance_file_sha256,
            "provenance_job_id": args.expected_provenance_job_id,
            "provenance_canonical_content_sha256": provenance["cache"][
                "canonical_content_sha256"
            ],
            "capability_manifest": str(args.capability_manifest),
            "capability_manifest_sha256": capability_file_sha256,
            "capability_member_success_rate": capability["expected"]["members"][
                args.checkpoint_seed
            ]["success_rate"],
            "dataset_metadata": dataset_metadata,
            "source_sha256": source_sha256,
        },
        "runtime": runtime,
        "component_identity": component_identity,
        "inputs": input_rows,
        "evaluations": evaluations,
        "aggregate": {
            "inputs_audited": len(input_rows),
            "stage_evaluations": len(evaluations) * len(STAGE_NAMES),
            "action_chunks": len(evaluations),
            "action_scalars": len(evaluations) * 8 * 7,
            "manual_deployed_forward_exact_for_all": all(
                row["manual_deployed_forward_exact"] is True for row in evaluations
            ),
            "all_finite": all_finite,
            "max_stage_relative_l2_error": max_stage_relative_l2_error,
            "max_action_absolute_error": max_action_absolute_error,
            "max_action_relative_l2_error": max_action_relative_l2_error,
            "end_to_end_action_relative_l2_gate": PROTOCOL[
                "end_to_end_action_relative_l2_gate"
            ],
            "passed": passed,
        },
        "elapsed_s": time.perf_counter() - started,
        "scope": (
            "Input-conditioned full learned-forward numerical replay from prepared model "
            "tensors through the learned patch projection, four vision blocks, deployed "
            "embeddings and projections, eight joint blocks, final RationalNorm, and linear "
            "action head. This is not one compact symbolic contraction or an input-general "
            "proof. It excludes preprocessing, fixed action unnormalization, chunk execution, "
            "simulator dynamics, decoding outside the linear head, and success evaluation."
        ),
    }
    write_json(args.output, result)
    print("RESULT", json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError(
            "Frozen full learned-forward gate failed: "
            f"finite={all_finite}, max action relative L2={max_action_relative_l2_error:.9g}"
        )


if __name__ == "__main__":
    main()
