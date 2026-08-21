#!/usr/bin/env python3
"""Frozen complete-joint-block certificate for convolutional χ-VLA checkpoints.

For each selected input and each of the eight joint blocks, this program compares
one deployed PyTorch float32 block call with an independent NumPy float64
reimplementation of both RationalNorm sites, bilinear attention, both residual
updates, residual gains, and the bilinear FFN. The reconstruction reads saved
weights and buffers but never calls a learned primitive's forward method.
"""

from __future__ import annotations

import argparse
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

from athena.libero_dataset_metadata import load_dataset_task_languages
from athena.run_conv_joint_attention_certificate import (
    FROZEN_CACHE,
    FROZEN_CHECKPOINTS,
    FROZEN_PROVENANCE_JOB_ID,
    PROTOCOL as ATTENTION_PROTOCOL,
    file_sha256,
    hash_array,
    rank_cache_inputs,
    validate_dataset_metadata,
    validate_provenance,
    write_json,
)
from athena.run_conv_joint_ffn_certificate import (
    PROTOCOL as FFN_PROTOCOL,
    factor_identity as companion_ffn_identity,
)
from athena.run_xvla_experiment import (
    build_vocab,
    load_cache_statistics,
    load_suite,
    make_config,
    task_languages,
    tensorize_sample,
)
from xvla.models.vla import ChiVLA
from xvla.nn.attention import BilinearAttention, causal_mask
from xvla.nn.bilinear import BilinearFFN
from xvla.nn.normalization import RationalNorm


SCHEMA = "xvla-conv-joint-block-certificate-v1"
PROTOCOL = {
    "suite": "libero_object",
    "architecture": "chi",
    "vision_encoder": "conv",
    "resolution": 64,
    "action_horizon": 8,
    "matmul_precision": "highest",
    "full_inputs_per_checkpoint": 16,
    "smoke_inputs": 1,
    "input_identity_source": "xvla-conv-joint-attention-certificate-v1",
    "input_selection": ATTENTION_PROTOCOL["input_selection"],
    "joint_blocks_per_input": 8,
    "sequence_length": 107,
    "model_width": 384,
    "attention_heads": 12,
    "attention_head_dim": 32,
    "ffn_cp_rank": 1152,
    "reference": (
        "one unmodified deployed PyTorch float32 ChiTransformerBlock call with the "
        "explicit attention path"
    ),
    "reconstruction": (
        "independent NumPy float64 RationalNorm, bilinear-attention, residual-gain, "
        "bilinear-FFN, and second-residual equations from saved tensors"
    ),
    "comparison": (
        "deployed float32 output cast to float64 versus the NumPy float64 reconstruction"
    ),
    "complete_block_relative_l2_gate": 1e-4,
    "gate_rationale": (
        "The comparison crosses float32 CUDA and float64 CPU implementations and composes "
        "multiple width-384, rank-1152, and length-107 reductions. A 1e-4 relative-L2 "
        "threshold is a strict numerical-equivalence gate while allowing normal float32 "
        "accumulation and operation-order error."
    ),
}
FROZEN_COMPANION_CERTIFICATES = {
    "attention": {
        "basename": "conv_joint_attention_certificate_v1_summary.json",
        "sha256": "3c380a899e5621d312cc45d62aff99ce883dd8dd029de56078e1cdc67838613f",
        "schema": "xvla-conv-joint-attention-certificate-summary-v1",
    },
    "rational_norm": {
        "basename": "rational_norm_safety_v1_summary.json",
        "sha256": "25c9dd61bfc805c321b12ba1b6575e4463fb76c1369de3b7a317dbdb524b17a8",
        "schema": "xvla-rational-norm-safety-summary-v1",
    },
}
SOURCE_FILES = (
    "athena/run_conv_joint_block_certificate.py",
    "athena/summarize_conv_joint_block_certificate.py",
    "athena/slurm_conv_joint_block_certificate.sbatch",
    "athena/slurm_summarize_conv_joint_block_certificate.sbatch",
    "athena/launch_conv_joint_block_certificate.sh",
    "athena/run_conv_joint_attention_certificate.py",
    "athena/run_conv_joint_ffn_certificate.py",
    "athena/run_rational_norm_safety.py",
    "athena/libero_dataset_metadata.py",
    "athena/run_xvla_experiment.py",
    "xvla/models/vla.py",
    "xvla/models/vit.py",
    "xvla/nn/attention.py",
    "xvla/nn/bilinear.py",
    "xvla/nn/block.py",
    "xvla/nn/normalization.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-seed", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--provenance-result", type=Path, required=True)
    parser.add_argument("--expected-provenance-job-id", required=True)
    parser.add_argument("--attention-summary", type=Path, required=True)
    parser.add_argument("--rational-norm-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("strict_smoke", "full"), required=True)
    return parser.parse_args()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def source_hashes() -> dict[str, str]:
    root = repository_root()
    hashes = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source file is absent: {path}")
        hashes[relative] = file_sha256(path)
    return hashes


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise RuntimeError(f"JSON result is not an object: {path}")
    return result


def validate_companion(path: Path, label: str) -> dict[str, Any]:
    frozen = FROZEN_COMPANION_CERTIFICATES[label]
    if path.name != frozen["basename"]:
        raise RuntimeError(f"{label} companion basename differs")
    if file_sha256(path) != frozen["sha256"]:
        raise RuntimeError(f"{label} companion SHA-256 differs")
    result = load_json(path)
    if result.get("schema") != frozen["schema"]:
        raise RuntimeError(f"{label} companion schema differs")
    if label == "attention":
        passed = result.get("aggregate", {}).get("all_three_checkpoints_pass")
    else:
        passed = result.get("gates", {}).get("all_three_primary_pass")
    if passed is not True:
        raise RuntimeError(f"{label} companion did not pass its primary gate")
    expected_checkpoints = {
        seed: (value["basename"], value["sha256"])
        for seed, value in FROZEN_CHECKPOINTS.items()
    }
    if label == "attention":
        actual_checkpoints = {
            int(row["seed"]): (row["checkpoint_basename"], row["checkpoint_sha256"])
            for row in result.get("checkpoints", [])
        }
    else:
        actual_checkpoints = {
            int(row["seed"]): (
                row["checkpoint"]["basename"],
                row["checkpoint"]["sha256"],
            )
            for row in result.get("checkpoints", [])
            if row.get("gates", {}).get("primary_pass") is True
        }
    if actual_checkpoints != expected_checkpoints:
        raise RuntimeError(f"{label} companion checkpoint identities differ")
    return {
        "path": str(path),
        "basename": path.name,
        "sha256": frozen["sha256"],
        "schema": frozen["schema"],
    }


def validate_frozen_inputs(
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    expected = FROZEN_CHECKPOINTS.get(args.checkpoint_seed)
    if expected is None:
        raise ValueError("Checkpoint seed must be 0, 1, or 2")
    errors = []
    if args.checkpoint.name != expected["basename"]:
        errors.append("checkpoint basename differs")
    if args.expected_checkpoint_sha256 != expected["sha256"]:
        errors.append("declared checkpoint SHA differs")
    if not args.checkpoint.is_file() or file_sha256(args.checkpoint) != expected["sha256"]:
        errors.append("live checkpoint SHA differs")
    if args.cache.name != FROZEN_CACHE["basename"]:
        errors.append("cache basename differs")
    if args.expected_cache_sha256 != FROZEN_CACHE["sha256"]:
        errors.append("declared cache SHA differs")
    if args.expected_provenance_job_id != FROZEN_PROVENANCE_JOB_ID:
        errors.append("provenance job identity differs")
    if args.output.exists():
        errors.append("output already exists")
    if errors:
        raise RuntimeError("Frozen identity validation failed: " + "; ".join(errors))
    provenance = validate_provenance(
        args.provenance_result, args.cache, args.expected_cache_sha256
    )
    companions = {
        "attention": validate_companion(args.attention_summary, "attention"),
        "rational_norm": validate_companion(
            args.rational_norm_summary, "rational_norm"
        ),
        "ffn_protocol": {
            "schema": "xvla-conv-joint-ffn-certificate-v1",
            "relative_l2_gate": FFN_PROTOCOL["relative_l2_gate"],
            "input_dim": FFN_PROTOCOL["input_dim"],
            "cp_rank": FFN_PROTOCOL["cp_rank"],
            "output_dim": FFN_PROTOCOL["output_dim"],
        },
    }
    return expected["sha256"], provenance, companions


def tensor_identity(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hash_array(tensor.detach().cpu().numpy()),
    }


def linear_identity(layer: Any) -> dict[str, Any]:
    identity = {"weight": tensor_identity(layer.weight)}
    identity["bias"] = None if layer.bias is None else tensor_identity(layer.bias)
    return identity


def rational_norm_identity(module: RationalNorm) -> dict[str, Any]:
    if module.variant != "pade" or module.deg != 2:
        raise RuntimeError("Complete-block certificate requires Padé [2/2] RationalNorm")
    return {
        "variant": module.variant,
        "degree": int(module.deg),
        "eps": float(module.eps),
        "running_ms": tensor_identity(module.running_ms),
        "initialized": tensor_identity(module.initialized),
        "numerator_coefficients": tensor_identity(module.pa),
        "denominator_coefficients": tensor_identity(module.pb),
    }


def attention_identity(module: BilinearAttention) -> dict[str, Any]:
    if module.qk_norm != "rational":
        raise RuntimeError("Complete-block certificate requires rational QK normalization")
    return {
        "dim": int(module.dim),
        "heads": int(module.n_heads),
        "head_dim": int(module.head_dim),
        "causal": bool(module.causal),
        "row_scale": module.row_scale,
        "score_scale": module.score_scale,
        "score_denominator": int(module._score_denom),
        "linears": {
            name: linear_identity(getattr(module, name))
            for name in ("wq1", "wk1", "wq2", "wk2", "wv", "wo")
        },
        "qk_norms": {
            name: rational_norm_identity(getattr(module, name))
            for name in ("rn_q1", "rn_k1", "rn_q2", "rn_k2")
        },
    }


def ffn_identity(module: BilinearFFN) -> dict[str, Any]:
    identity = {
        "left_weight_shape": list(module.left.weight.shape),
        "left_bias_shape": list(module.left.bias.shape),
        "right_weight_shape": list(module.right.weight.shape),
        "right_bias_shape": list(module.right.bias.shape),
        "down_weight_shape": list(module.down.weight.shape),
        "down_bias_present": module.down.bias is not None,
        "parameter_count": sum(
            tensor.numel()
            for tensor in (
                module.left.weight,
                module.left.bias,
                module.right.weight,
                module.right.bias,
                module.down.weight,
            )
        ),
        "sha256": {
            "left_weight": hash_array(module.left.weight.detach().cpu().numpy()),
            "left_bias": hash_array(module.left.bias.detach().cpu().numpy()),
            "right_weight": hash_array(module.right.weight.detach().cpu().numpy()),
            "right_bias": hash_array(module.right.bias.detach().cpu().numpy()),
            "down_weight": hash_array(module.down.weight.detach().cpu().numpy()),
        },
    }
    if identity != companion_ffn_identity(module):
        raise RuntimeError("FFN factor identity differs from the frozen FFN certificate")
    return identity


def block_identity(block: Any, block_index: int) -> dict[str, Any]:
    if not isinstance(block.rbn_attn, RationalNorm):
        raise RuntimeError(f"Joint block {block_index} attention norm is not RationalNorm")
    if not isinstance(block.rbn_ffn, RationalNorm):
        raise RuntimeError(f"Joint block {block_index} FFN norm is not RationalNorm")
    if not isinstance(block.attn, BilinearAttention):
        raise RuntimeError(f"Joint block {block_index} attention is not bilinear")
    if not isinstance(block.ffn, BilinearFFN):
        raise RuntimeError(f"Joint block {block_index} FFN is not bilinear")
    return {
        "block_index": block_index,
        "attention_norm": rational_norm_identity(block.rbn_attn),
        "attention": attention_identity(block.attn),
        "attention_gain": tensor_identity(block.attn_gain),
        "ffn_norm": rational_norm_identity(block.rbn_ffn),
        "ffn": ffn_identity(block.ffn),
        "ffn_gain": tensor_identity(block.ffn_gain),
    }


def array64(tensor: torch.Tensor) -> np.ndarray:
    return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float64)


def rational_norm_numpy(array: np.ndarray, module: RationalNorm) -> np.ndarray:
    if module.variant != "pade" or module.deg != 2:
        raise RuntimeError("Independent reconstruction requires Padé [2/2] RationalNorm")
    running_ms = float(module.running_ms.detach().cpu())
    if not math.isfinite(running_ms) or running_ms <= 0.0:
        raise RuntimeError("RationalNorm running mean-square is invalid")
    numerator_coefficients = array64(module.pa)
    denominator_coefficients = array64(module.pb)
    mean_square = np.square(array).mean(axis=-1, keepdims=True) + float(module.eps)
    scaled = mean_square / max(running_ms, 1e-12)
    numerator = sum(
        numerator_coefficients[degree] * scaled**degree
        for degree in range(module.deg + 1)
    )
    denominator = sum(
        denominator_coefficients[degree] * scaled**degree
        for degree in range(module.deg + 1)
    )
    if not np.isfinite(denominator).all() or np.any(denominator == 0.0):
        raise RuntimeError("RationalNorm reconstruction denominator is invalid")
    return array * (numerator / denominator) * max(running_ms, 1e-12) ** -0.5


def linear_numpy(array: np.ndarray, layer: Any) -> np.ndarray:
    output = array @ array64(layer.weight).T
    if layer.bias is not None:
        output = output + array64(layer.bias)
    return output


def attention_numpy(
    array: np.ndarray, module: BilinearAttention, mask: np.ndarray
) -> np.ndarray:
    sequence_length = array.shape[0]

    def project(name: str) -> np.ndarray:
        projected = linear_numpy(array, getattr(module, name))
        return projected.reshape(sequence_length, module.n_heads, module.head_dim).transpose(1, 0, 2)

    q1 = rational_norm_numpy(project("wq1"), module.rn_q1)
    k1 = rational_norm_numpy(project("wk1"), module.rn_k1)
    q2 = rational_norm_numpy(project("wq2"), module.rn_q2)
    k2 = rational_norm_numpy(project("wk2"), module.rn_k2)
    value = project("wv")
    first_scores = q1 @ k1.transpose(0, 2, 1)
    second_scores = q2 @ k2.transpose(0, 2, 1)
    pattern = (first_scores * second_scores) / float(module._score_denom)
    visible = np.maximum(mask.sum(axis=-1), 1.0)
    row_scale = visible**-0.5 if module.row_scale == "invsqrt" else visible**-1.0
    heads = row_scale[None, :, None] * ((pattern * mask[None, :, :]) @ value)
    concatenated = heads.transpose(1, 0, 2).reshape(sequence_length, module.dim)
    return linear_numpy(concatenated, module.wo)


def ffn_numpy(array: np.ndarray, module: BilinearFFN) -> np.ndarray:
    if module.down.bias is not None:
        raise RuntimeError("Complete-block certificate requires a bias-free FFN down projection")
    left = linear_numpy(array, module.left)
    right = linear_numpy(array, module.right)
    return (left * right) @ array64(module.down.weight).T


def reconstruct_block(block: Any, hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    normalized_attention = rational_norm_numpy(hidden, block.rbn_attn)
    attention_output = attention_numpy(normalized_attention, block.attn, mask)
    after_attention = hidden + float(block.attn_gain.detach().cpu()) * attention_output
    normalized_ffn = rational_norm_numpy(after_attention, block.rbn_ffn)
    ffn_output = ffn_numpy(normalized_ffn, block.ffn)
    return after_attention + float(block.ffn_gain.detach().cpu()) * ffn_output


def errors(reference: np.ndarray, reconstruction: np.ndarray) -> dict[str, float | bool]:
    if reference.shape != reconstruction.shape:
        raise RuntimeError("Reference and reconstruction shapes differ")
    difference = reference - reconstruction
    finite = bool(
        np.isfinite(reference).all()
        and np.isfinite(reconstruction).all()
        and np.isfinite(difference).all()
    )
    if not finite:
        return {
            "finite": False,
            "max_abs_error": float("inf"),
            "relative_l2_error": float("inf"),
        }
    return {
        "finite": True,
        "max_abs_error": float(np.max(np.abs(difference))),
        "relative_l2_error": float(
            np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-30)
        ),
    }


def make_model(vocab_size: int, stats: dict[str, Any], checkpoint: Path) -> ChiVLA:
    config = make_config(
        "chi",
        vocab_size,
        stats["state_dim"],
        stats["action_dim"],
        PROTOCOL["resolution"],
        PROTOCOL["action_horizon"],
        vision_encoder="conv",
    )
    model = ChiVLA(config).cuda()
    state_dict = torch.load(checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if model.cfg.vision_encoder != "conv":
        raise RuntimeError("Loaded model is not convolutional")
    if len(model.backbone.blocks) != PROTOCOL["joint_blocks_per_input"]:
        raise RuntimeError("Joint block count differs from the frozen protocol")
    identities = [
        block_identity(block, block_index)
        for block_index, block in enumerate(model.backbone.blocks)
    ]
    if any(row["attention"]["dim"] != PROTOCOL["model_width"] for row in identities):
        raise RuntimeError("Joint block width differs from the frozen protocol")
    if any(row["attention"]["heads"] != PROTOCOL["attention_heads"] for row in identities):
        raise RuntimeError("Joint attention head count differs from the frozen protocol")
    if any(row["ffn"]["left_weight_shape"][0] != PROTOCOL["ffn_cp_rank"] for row in identities):
        raise RuntimeError("Joint FFN rank differs from the frozen protocol")
    return model


def initial_hidden(
    model: ChiVLA,
    sample_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    image, instruction, state, embodiment = sample_tensors
    visual_tokens = model._visual_tokens(image)
    hidden = torch.cat(
        [
            visual_tokens,
            model.bos.expand(image.shape[0], -1, -1),
            model.tok_emb(instruction),
            model.state_proj(state)[:, None],
            model.embodiment_emb(embodiment)[:, None],
            model.action_queries.expand(image.shape[0], -1, -1),
        ],
        dim=1,
    )
    return hidden + model.pos_emb[:, : hidden.shape[1]]


def audit_input(
    model: ChiVLA,
    sample_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    input_identity: dict[str, Any],
) -> list[dict[str, Any]]:
    with torch.inference_mode():
        hidden = initial_hidden(model, sample_tensors)
        if hidden.shape != (1, PROTOCOL["sequence_length"], PROTOCOL["model_width"]):
            raise RuntimeError("Joint transformer input shape differs from the frozen protocol")
        mask = causal_mask(hidden.shape[1], device=hidden.device, dtype=hidden.dtype)
        mask_numpy = np.asarray(mask.detach().cpu().numpy(), dtype=np.float64)
        rows = []
        for block_index, block in enumerate(model.backbone.blocks):
            block_input = np.asarray(hidden[0].detach().cpu().numpy(), dtype=np.float64)
            reconstruction = reconstruct_block(block, block_input, mask_numpy)
            deployed_output = block(hidden, mask=mask, method="explicit")
            reference = np.asarray(
                deployed_output[0].detach().cpu().numpy(), dtype=np.float64
            )
            module_errors = errors(reference, reconstruction)
            row = {
                "selection_rank": input_identity["selection_rank"],
                "selection_sha256": input_identity["selection_sha256"],
                "block_index": block_index,
                "stack": "joint",
                "sequence_length": int(hidden.shape[1]),
                "model_width": int(hidden.shape[2]),
                "block_identity": block_identity(block, block_index),
                "input_sha256": hash_array(block_input),
                "deployed_float32_output_sha256": hash_array(
                    deployed_output[0].detach().cpu().numpy()
                ),
                "reconstruction_float64_output_sha256": hash_array(reconstruction),
                **module_errors,
            }
            rows.append(row)
            print("CONV_JOINT_BLOCK_EXACT", json.dumps(row), flush=True)
            hidden = deployed_output
    return rows


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    source_sha256 = source_hashes()
    checkpoint_sha256, provenance, companions = validate_frozen_inputs(args)
    provenance_result_sha256 = file_sha256(args.provenance_result)
    input_count = (
        PROTOCOL["smoke_inputs"]
        if args.mode == "strict_smoke"
        else PROTOCOL["full_inputs_per_checkpoint"]
    )
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_float32_matmul_precision(PROTOCOL["matmul_precision"])
    if not torch.cuda.is_available():
        raise RuntimeError("The certificate runner requires a CUDA compute node")

    print(f"Loading frozen cache {args.cache}", flush=True)
    with args.cache.open("rb") as handle:
        records = pickle.load(handle)
    if len(records) != FROZEN_CACHE["frames"]:
        raise RuntimeError("Live cache frame count differs from the frozen count")
    selected = rank_cache_inputs(records, checkpoint_sha256, input_count)

    suite = load_suite(PROTOCOL["suite"])
    official_tasks = task_languages(suite)
    cache_tasks, dataset_metadata = load_dataset_task_languages(
        PROTOCOL["suite"], official_tasks
    )
    validate_dataset_metadata(dataset_metadata)
    vocab, encode = build_vocab(cache_tasks)
    stats = load_cache_statistics(args.cache, PROTOCOL["action_horizon"])
    model = make_model(len(vocab), stats, args.checkpoint)
    block_identities = [
        block_identity(block, block_index)
        for block_index, block in enumerate(model.backbone.blocks)
    ]

    input_rows = []
    module_rows = []
    for identity in selected:
        record = records[identity["cache_row_index"]]
        task_index = int(record[5])
        sample = {
            "image": np.asarray(record[2]),
            "task": task_index,
            "state": np.asarray(record[3], dtype=np.float32),
        }
        input_rows.append(
            {
                **identity,
                "instruction": cache_tasks[task_index],
                "instruction_ids": encode(cache_tasks[task_index]),
            }
        )
        tensors = tensorize_sample(sample, encode, cache_tasks, stats)
        module_rows.extend(audit_input(model, tensors, identity))

    all_finite = all(bool(row["finite"]) for row in module_rows)
    max_absolute_error = max(float(row["max_abs_error"]) for row in module_rows)
    max_relative_l2_error = max(
        float(row["relative_l2_error"]) for row in module_rows
    )
    passed = (
        all_finite
        and max_relative_l2_error <= PROTOCOL["complete_block_relative_l2_gate"]
    )

    if source_hashes() != source_sha256:
        raise RuntimeError("Runner or imported source changed during the certificate")
    if file_sha256(args.checkpoint) != checkpoint_sha256:
        raise RuntimeError("Checkpoint changed during the certificate")
    if file_sha256(args.cache) != args.expected_cache_sha256:
        raise RuntimeError("Cache changed during the certificate")
    if file_sha256(args.provenance_result) != provenance_result_sha256:
        raise RuntimeError("Provenance result changed during the certificate")
    if validate_companion(args.attention_summary, "attention") != companions["attention"]:
        raise RuntimeError("Attention companion changed during the certificate")
    if (
        validate_companion(args.rational_norm_summary, "rational_norm")
        != companions["rational_norm"]
    ):
        raise RuntimeError("RationalNorm companion changed during the certificate")

    result = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "mode": args.mode,
        "identity": {
            "checkpoint_seed": args.checkpoint_seed,
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_basename": args.checkpoint.name,
            "checkpoint_sha256": checkpoint_sha256,
            "cache_path": str(args.cache),
            "cache_basename": args.cache.name,
            "cache_sha256": args.expected_cache_sha256,
            "cache_frames": len(records),
            "provenance_result": str(args.provenance_result),
            "provenance_result_sha256": provenance_result_sha256,
            "provenance_job_id": args.expected_provenance_job_id,
            "provenance_canonical_content_sha256": provenance["cache"][
                "canonical_content_sha256"
            ],
            "dataset_metadata": dataset_metadata,
            "companion_certificates": companions,
            "source_sha256": source_sha256,
        },
        "runtime": {
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "python_packages": {
                package: importlib.metadata.version(package)
                for package in ("pillow", "huggingface-hub", "pandas")
            },
        },
        "inputs": input_rows,
        "unique_block_identities": block_identities,
        "module_rows": module_rows,
        "aggregate": {
            "inputs_audited": len(input_rows),
            "unique_blocks": len(block_identities),
            "complete_block_evaluations": len(module_rows),
            "token_rows": sum(row["sequence_length"] for row in module_rows),
            "scalar_outputs": sum(
                row["sequence_length"] * row["model_width"] for row in module_rows
            ),
            "all_finite": all_finite,
            "max_absolute_error": max_absolute_error,
            "max_relative_l2_error": max_relative_l2_error,
            "relative_l2_gate": PROTOCOL["complete_block_relative_l2_gate"],
            "passed": passed,
        },
        "elapsed_s": time.perf_counter() - started,
        "scope": (
            "Independent input-conditioned numerical reproduction of every complete joint "
            "transformer block, including its two RationalNorm sites, rational-QK bilinear "
            "attention, two residual gains, bilinear FFN, and residual composition. The "
            "deployed reference is a single unmodified float32 block call. This does not "
            "reconstruct the convolutional vision stem, embeddings, final action head, or one "
            "closed-form contraction of the complete policy."
        ),
    }
    write_json(args.output, result)
    print("RESULT", json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError(
            f"Frozen complete-block gate failed: finite={all_finite}, max relative L2="
            f"{max_relative_l2_error:.9g}"
        )


if __name__ == "__main__":
    main()
