#!/usr/bin/env python3
"""Frozen exact-reconstruction certificate for Conv χ-VLA joint FFNs.

The audit reuses the checkpoint identities and deterministic cache inputs from
the Conv joint-attention certificate.  It reconstructs every bilinear FFN in
the eight-block joint transformer from its saved CP factors with an independent
NumPy float64 contraction and compares that result with an explicit PyTorch
float64 evaluation of the same affine-factor equation.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
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
from athena.run_xvla_experiment import (
    build_vocab,
    load_cache_statistics,
    load_suite,
    make_config,
    task_languages,
    tensorize_sample,
)
from xvla.models.vla import ChiVLA
from xvla.nn.attention import causal_mask
from xvla.nn.bilinear import BilinearFFN


SCHEMA = "xvla-conv-joint-ffn-certificate-v1"
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
    "joint_modules_per_input": 8,
    "sequence_length": 107,
    "input_dim": 384,
    "cp_rank": 1152,
    "output_dim": 384,
    "down_bias": False,
    "parameters_per_unique_module": 1_329_408,
    "reference": (
        "explicit PyTorch float64 affine left, affine right, elementwise product, "
        "and down projection evaluated from the trained weights"
    ),
    "reconstruction": (
        "independent NumPy float64 CP-factor contraction from saved left, right, "
        "and down factors without materializing a dense third-order core"
    ),
    "relative_l2_gate": 1e-10,
}
SOURCE_FILES = (
    "athena/run_conv_joint_ffn_certificate.py",
    "athena/summarize_conv_joint_ffn_certificate.py",
    "athena/slurm_conv_joint_ffn_certificate.sbatch",
    "athena/slurm_summarize_conv_joint_ffn_certificate.sbatch",
    "athena/launch_conv_joint_ffn_certificate.sh",
    "athena/run_conv_joint_attention_certificate.py",
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


def validate_frozen_inputs(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
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
    return expected["sha256"], provenance


def parameter_sha256(tensor: torch.Tensor) -> str:
    return hash_array(tensor.detach().cpu().numpy())


def factor_identity(ffn: BilinearFFN) -> dict[str, Any]:
    if ffn.down.bias is not None:
        raise RuntimeError("Frozen certificate requires a bias-free down projection")
    tensors = {
        "left_weight": ffn.left.weight,
        "left_bias": ffn.left.bias,
        "right_weight": ffn.right.weight,
        "right_bias": ffn.right.bias,
        "down_weight": ffn.down.weight,
    }
    if any(tensor is None for tensor in tensors.values()):
        raise RuntimeError("A required CP factor or affine bias is absent")
    return {
        "left_weight_shape": list(ffn.left.weight.shape),
        "left_bias_shape": list(ffn.left.bias.shape),
        "right_weight_shape": list(ffn.right.weight.shape),
        "right_bias_shape": list(ffn.right.bias.shape),
        "down_weight_shape": list(ffn.down.weight.shape),
        "down_bias_present": False,
        "parameter_count": sum(tensor.numel() for tensor in tensors.values()),
        "sha256": {
            name: parameter_sha256(tensor) for name, tensor in tensors.items()
        },
    }


def independent_reconstruction(
    ffn: BilinearFFN, normalized_input: torch.Tensor
) -> np.ndarray:
    """Contract the stored CP factors with NumPy, independent of module.forward."""
    if ffn.down.bias is not None:
        raise RuntimeError("Frozen certificate requires down_bias=False")
    x = normalized_input[0].detach().double().cpu().numpy()
    left_weight = ffn.left.weight.detach().double().cpu().numpy()
    left_bias = ffn.left.bias.detach().double().cpu().numpy()
    right_weight = ffn.right.weight.detach().double().cpu().numpy()
    right_bias = ffn.right.bias.detach().double().cpu().numpy()
    down_weight = ffn.down.weight.detach().double().cpu().numpy()

    left = np.einsum("td,rd->tr", x, left_weight, optimize=False) + left_bias
    right = np.einsum("td,rd->tr", x, right_weight, optimize=False) + right_bias
    output = np.einsum(
        "tr,or->to", left * right, down_weight, optimize=False
    )
    return np.asarray(output, dtype=np.float64)


def torch_reference(ffn: BilinearFFN, normalized_input: torch.Tensor) -> np.ndarray:
    """Evaluate the affine-factor equation explicitly with PyTorch float64."""
    with torch.inference_mode():
        x = normalized_input.double()
        left = torch.nn.functional.linear(
            x, ffn.left.weight.detach().double(), ffn.left.bias.detach().double()
        )
        right = torch.nn.functional.linear(
            x, ffn.right.weight.detach().double(), ffn.right.bias.detach().double()
        )
        output = torch.nn.functional.linear(
            left * right,
            ffn.down.weight.detach().double(),
            None,
        )
    return output[0].detach().cpu().numpy()


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
    if len(model.backbone.blocks) != PROTOCOL["joint_modules_per_input"]:
        raise RuntimeError("Joint FFN count differs from the frozen protocol")
    for block_index, block in enumerate(model.backbone.blocks):
        ffn = block.ffn
        if not isinstance(ffn, BilinearFFN):
            raise RuntimeError(f"Joint block {block_index} is not a BilinearFFN")
        if (
            ffn.dim != PROTOCOL["input_dim"]
            or ffn.rank != PROTOCOL["cp_rank"]
            or ffn.out_dim != PROTOCOL["output_dim"]
            or ffn.down.bias is not None
        ):
            raise RuntimeError(f"Joint block {block_index} FFN shape differs")
        if factor_identity(ffn)["parameter_count"] != PROTOCOL[
            "parameters_per_unique_module"
        ]:
            raise RuntimeError(f"Joint block {block_index} parameter count differs")
    return model


def audit_input(
    model: ChiVLA,
    sample_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    input_identity: dict[str, Any],
) -> list[dict[str, Any]]:
    image, instruction, state, embodiment = sample_tensors
    with torch.inference_mode():
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
        hidden = hidden + model.pos_emb[:, : hidden.shape[1]]
        mask = causal_mask(hidden.shape[1], device=hidden.device, dtype=hidden.dtype)

        rows = []
        for block_index, block in enumerate(model.backbone.blocks):
            normalized_attention = block.rbn_attn(hidden)
            after_attention = hidden + block.attn_gain * block.attn(
                normalized_attention, mask=mask
            )
            normalized_ffn = block.rbn_ffn(after_attention)
            reference = torch_reference(block.ffn, normalized_ffn)
            reconstruction = independent_reconstruction(block.ffn, normalized_ffn)
            module_errors = errors(reference, reconstruction)
            row = {
                "selection_rank": input_identity["selection_rank"],
                "selection_sha256": input_identity["selection_sha256"],
                "block_index": block_index,
                "stack": "joint",
                "sequence_length": int(normalized_ffn.shape[1]),
                "input_dim": int(block.ffn.dim),
                "cp_rank": int(block.ffn.rank),
                "output_dim": int(block.ffn.out_dim),
                "factor_identity": factor_identity(block.ffn),
                "reference_output_sha256": hash_array(reference),
                "reconstruction_output_sha256": hash_array(reconstruction),
                **module_errors,
            }
            rows.append(row)
            print("CONV_JOINT_FFN_EXACT", json.dumps(row), flush=True)
            hidden = after_attention + block.ffn_gain * block.ffn(normalized_ffn)
    return rows


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    source_sha256 = source_hashes()
    checkpoint_sha256, provenance = validate_frozen_inputs(args)
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
        instruction_ids = encode(cache_tasks[task_index])
        input_rows.append(
            {
                **identity,
                "instruction": cache_tasks[task_index],
                "instruction_ids": instruction_ids,
            }
        )
        tensors = tensorize_sample(sample, encode, cache_tasks, stats)
        module_rows.extend(audit_input(model, tensors, identity))

    all_finite = all(bool(row["finite"]) for row in module_rows)
    max_absolute_error = max(float(row["max_abs_error"]) for row in module_rows)
    max_relative_l2_error = max(
        float(row["relative_l2_error"]) for row in module_rows
    )
    passed = all_finite and max_relative_l2_error <= PROTOCOL["relative_l2_gate"]
    unique_factor_identities = [
        factor_identity(block.ffn) for block in model.backbone.blocks
    ]

    if source_hashes() != source_sha256:
        raise RuntimeError("Runner or imported source changed during the certificate")
    if file_sha256(args.checkpoint) != checkpoint_sha256:
        raise RuntimeError("Checkpoint changed during the certificate")
    if file_sha256(args.cache) != args.expected_cache_sha256:
        raise RuntimeError("Cache changed during the certificate")
    if file_sha256(args.provenance_result) != provenance_result_sha256:
        raise RuntimeError("Provenance result changed during the certificate")

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
        "unique_module_factors": unique_factor_identities,
        "module_rows": module_rows,
        "aggregate": {
            "inputs_audited": len(input_rows),
            "unique_modules": len(unique_factor_identities),
            "module_evaluations": len(module_rows),
            "token_rows": sum(row["sequence_length"] for row in module_rows),
            "scalar_outputs": sum(
                row["sequence_length"] * row["output_dim"] for row in module_rows
            ),
            "unique_module_parameters": sum(
                row["parameter_count"] for row in unique_factor_identities
            ),
            "all_finite": all_finite,
            "max_absolute_error": max_absolute_error,
            "max_relative_l2_error": max_relative_l2_error,
            "relative_l2_gate": PROTOCOL["relative_l2_gate"],
            "passed": passed,
        },
        "elapsed_s": time.perf_counter() - started,
        "scope": (
            "Independent layerwise numerical reproduction from saved CP factors for every "
            "bilinear FFN in the eight-block joint transformer on the frozen inputs in one "
            "convolutional χ-VLA checkpoint. This does not reconstruct the three bilinear-"
            "convolution vision stages, attention branches, normalization sites, residual "
            "composition, linear embeddings and projections, action head, or the complete "
            "end-to-end policy."
        ),
    }
    write_json(args.output, result)
    print("RESULT", json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError(
            f"Frozen numerical gate failed: finite={all_finite}, max relative L2="
            f"{max_relative_l2_error:.9g}"
        )


if __name__ == "__main__":
    main()
