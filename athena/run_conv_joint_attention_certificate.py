#!/usr/bin/env python3
"""Frozen exact-reconstruction certificate for legacy convolutional χ-VLA attention.

This program audits only the eight learned joint bilinear-attention modules.  The
convolutional vision stem contains no attention modules.  Inputs are selected from
the provenance-verified LIBERO-Object cache by a checkpoint-conditioned SHA-256
ranking fixed in ``PROTOCOL``.
"""

from __future__ import annotations

import argparse
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

from athena.libero_dataset_metadata import load_dataset_task_languages
from athena.run_xvla_experiment import (
    build_vocab,
    file_sha256,
    load_cache_statistics,
    load_suite,
    make_config,
    task_languages,
    tensorize_sample,
)
from xvla.models.vla import ChiVLA
from xvla.nn.attention import causal_mask


SCHEMA = "xvla-conv-joint-attention-certificate-v1"
PROTOCOL = {
    "suite": "libero_object",
    "architecture": "chi",
    "vision_encoder": "conv",
    "resolution": 64,
    "action_horizon": 8,
    "matmul_precision": "highest",
    "full_inputs_per_checkpoint": 16,
    "smoke_inputs": 1,
    "input_selection": (
        "lexicographically smallest SHA256(protocol_id || checkpoint_sha256 || "
        "canonical_cache_record_sha256) over every record in the frozen cache"
    ),
    "joint_modules_per_input": 8,
    "heads_per_module": 12,
    "reference": (
        "explicit PyTorch float64 attention equation evaluated from the trained weights"
    ),
    "reconstruction": (
        "independent NumPy float64 affine projections, rational QK normalization, "
        "bilinear head contractions, concatenation, and output projection"
    ),
    "relative_l2_gate": 1e-6,
}
PROTOCOL_ID = b"xvla-conv-joint-attention-certificate-v1\0"
FROZEN_CACHE = {
    "basename": "libero_frames_100000_64.pkl",
    "sha256": "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662",
    "frames": 66984,
}
FROZEN_PROVENANCE_JOB_ID = "830988"
FROZEN_DATASET_METADATA = {
    "repository": "lerobot/libero_object_image",
    "revision": "e1e080d7df1d0a359dff5c86c222e047549f447f",
    "metadata_file": "meta/tasks.parquet",
    "metadata_sha256": "34caee9641ae50bb4e077de306a7d0031753757882da8b1f117e7ea36a486b42",
    "dataset_to_official_task": {
        "0": 9,
        "1": 4,
        "2": 1,
        "3": 3,
        "4": 0,
        "5": 7,
        "6": 2,
        "7": 6,
        "8": 5,
        "9": 8,
    },
    "ordering_matches_official": False,
    "language_set_matches_official": True,
}
FROZEN_CHECKPOINTS = {
    0: {
        "basename": "ckpt_linear_rat_conv_s0.pt",
        "sha256": "4f9f3eef4bd661934b7c66af117995f02bdfc777368f52464f03d229ccb3e72d",
    },
    1: {
        "basename": "ckpt_linear_rat_conv_s1_matched.pt",
        "sha256": "9d8df0c30583222490536b21aac040b47c5f89556aa23c08265605f7c2bc8cb6",
    },
    2: {
        "basename": "ckpt_linear_rat_conv_s2_matched.pt",
        "sha256": "fc2e0bfa1a2ea2737ed0af13b168cd2562e4b0e36b159d0f98f3c5038131ace5",
    },
}
SOURCE_FILES = (
    "athena/run_conv_joint_attention_certificate.py",
    "athena/summarize_conv_joint_attention_certificate.py",
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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def hash_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def cache_record_identity(record: Any, cache_row_index: int) -> dict[str, Any]:
    if len(record) != 6:
        raise RuntimeError(f"Cache row {cache_row_index} does not have six fields")
    episode_index = int(record[0])
    frame_index = int(record[1])
    task_index = int(record[5])
    image = np.asarray(record[2])
    state = np.asarray(record[3])
    action = np.asarray(record[4])
    if image.shape != (64, 64, 3) or image.dtype != np.uint8:
        raise RuntimeError(f"Cache row {cache_row_index} has an invalid image")
    if state.shape != (8,) or action.shape != (7,):
        raise RuntimeError(f"Cache row {cache_row_index} has invalid state/action shape")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise RuntimeError(f"Cache row {cache_row_index} contains non-finite values")
    if task_index not in range(10):
        raise RuntimeError(f"Cache row {cache_row_index} has invalid task {task_index}")

    image_sha256 = hash_array(image)
    state_sha256 = hash_array(state)
    action_sha256 = hash_array(action)
    digest = hashlib.sha256()
    digest.update(PROTOCOL_ID)
    for label, value in (
        ("cache_row_index", cache_row_index),
        ("episode_index", episode_index),
        ("frame_index", frame_index),
        ("task_index", task_index),
    ):
        digest.update(label.encode("ascii") + b"\0")
        digest.update(int(value).to_bytes(8, "little", signed=True))
    for label, value in (
        ("image_sha256", image_sha256),
        ("state_sha256", state_sha256),
        ("action_sha256", action_sha256),
    ):
        digest.update(label.encode("ascii") + b"\0" + value.encode("ascii") + b"\0")
    return {
        "cache_row_index": cache_row_index,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "task_index": task_index,
        "image_sha256": image_sha256,
        "state_sha256": state_sha256,
        "action_sha256": action_sha256,
        "canonical_record_sha256": digest.hexdigest(),
    }


def rank_cache_inputs(
    records: list[Any], checkpoint_sha256: str, count: int
) -> list[dict[str, Any]]:
    if count < 1 or count > len(records):
        raise ValueError("Input count is outside the cache")
    ranked = []
    for cache_row_index, record in enumerate(records):
        identity = cache_record_identity(record, cache_row_index)
        digest = hashlib.sha256()
        digest.update(PROTOCOL_ID)
        digest.update(checkpoint_sha256.encode("ascii") + b"\0")
        digest.update(identity["canonical_record_sha256"].encode("ascii"))
        identity["selection_sha256"] = digest.hexdigest()
        ranked.append(identity)
    ranked.sort(key=lambda row: (row["selection_sha256"], row["cache_row_index"]))
    selected = ranked[:count]
    for selection_rank, row in enumerate(selected):
        row["selection_rank"] = selection_rank
    return selected


def validate_provenance(
    path: Path, cache: Path, expected_cache_sha256: str
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Object provenance result is absent: {path}")
    result = json.loads(path.read_text())
    cache_result = result.get("cache", {})
    source_result = result.get("source", {})
    metadata = result.get("metadata", {})
    mismatch_counts = result.get("comparison", {}).get("mismatch_counts", {})
    errors = []
    if result.get("schema") != "xvla-cache-provenance-v1":
        errors.append("wrong provenance schema")
    if result.get("suite") != "libero_object" or result.get("verified") is not True:
        errors.append("Object provenance did not pass")
    if cache_result.get("sha256") != expected_cache_sha256:
        errors.append("provenance cache SHA differs from the frozen SHA")
    if Path(str(cache_result.get("path", ""))).name != FROZEN_CACHE["basename"]:
        errors.append("provenance cache basename differs")
    if int(cache_result.get("frames", -1)) != FROZEN_CACHE["frames"]:
        errors.append("provenance frame count differs")
    if int(cache_result.get("resolution", -1)) != PROTOCOL["resolution"]:
        errors.append("provenance resolution differs")
    if source_result.get("repository") != FROZEN_DATASET_METADATA["repository"]:
        errors.append("provenance source repository differs")
    if source_result.get("revision") != FROZEN_DATASET_METADATA["revision"]:
        errors.append("provenance source revision differs")
    if source_result.get("content_hashes_match") is not True:
        errors.append("cache/source canonical content hashes do not match")
    if cache_result.get("canonical_content_sha256") != source_result.get(
        "canonical_content_sha256"
    ):
        errors.append("cache/source canonical hashes are absent or unequal")
    if metadata.get("sha256") != FROZEN_DATASET_METADATA["metadata_sha256"]:
        errors.append("provenance metadata SHA differs")
    if metadata.get("dataset_to_official_task") != FROZEN_DATASET_METADATA[
        "dataset_to_official_task"
    ]:
        errors.append("provenance task mapping differs")
    if not mismatch_counts or any(int(value) != 0 for value in mismatch_counts.values()):
        errors.append("provenance mismatch counts are absent or nonzero")
    if file_sha256(cache) != expected_cache_sha256:
        errors.append("live cache SHA differs from the frozen SHA")
    if errors:
        raise RuntimeError("Invalid frozen Object provenance: " + "; ".join(errors))
    return result


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


def validate_dataset_metadata(metadata: dict[str, Any]) -> None:
    if metadata != FROZEN_DATASET_METADATA:
        raise RuntimeError("Pinned dataset task metadata differs from the frozen identity")


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


def rational_norm_numpy(array: np.ndarray, module: Any) -> np.ndarray:
    if getattr(module, "variant", None) != "pade":
        raise RuntimeError("Certificate requires Padé RationalNorm")
    running_ms = float(module.running_ms.detach().cpu())
    if not math.isfinite(running_ms) or running_ms <= 0.0:
        raise RuntimeError("RationalNorm running mean-square is invalid")
    pa = module.pa.detach().double().cpu().numpy()
    pb = module.pb.detach().double().cpu().numpy()
    mean_square = np.square(array).mean(axis=-1, keepdims=True) + float(module.eps)
    scaled = mean_square / max(running_ms, 1e-12)
    numerator = sum(pa[degree] * scaled**degree for degree in range(module.deg + 1))
    denominator = sum(pb[degree] * scaled**degree for degree in range(module.deg + 1))
    if not np.isfinite(denominator).all() or np.any(denominator == 0.0):
        raise RuntimeError("RationalNorm reconstruction denominator is invalid")
    return array * (numerator / denominator) * max(running_ms, 1e-12) ** -0.5


def independent_reconstruction(
    attention: Any, normalized_input: torch.Tensor, mask: torch.Tensor
) -> tuple[list[np.ndarray], np.ndarray]:
    if attention.qk_norm != "rational":
        raise RuntimeError("Certificate requires rational QK normalization")
    x = normalized_input[0].detach().double().cpu().numpy()

    def weight(linear: Any) -> np.ndarray:
        return linear.weight.detach().double().cpu().numpy()

    def bias(linear: Any) -> np.ndarray:
        return linear.bias.detach().double().cpu().numpy()

    projections = {
        "q1": x @ weight(attention.wq1).T + bias(attention.wq1),
        "k1": x @ weight(attention.wk1).T + bias(attention.wk1),
        "q2": x @ weight(attention.wq2).T + bias(attention.wq2),
        "k2": x @ weight(attention.wk2).T + bias(attention.wk2),
        "v": x @ weight(attention.wv).T + bias(attention.wv),
    }
    mask_array = mask.detach().double().cpu().numpy()
    visible = np.maximum(mask_array.sum(axis=-1), 1.0)
    row_scale = visible**-0.5 if attention.row_scale == "invsqrt" else visible**-1.0
    head_outputs = []
    for head_index in range(attention.n_heads):
        start = head_index * attention.head_dim
        stop = (head_index + 1) * attention.head_dim
        head_slice = slice(start, stop)
        q1 = rational_norm_numpy(projections["q1"][:, head_slice], attention.rn_q1)
        k1 = rational_norm_numpy(projections["k1"][:, head_slice], attention.rn_k1)
        q2 = rational_norm_numpy(projections["q2"][:, head_slice], attention.rn_q2)
        k2 = rational_norm_numpy(projections["k2"][:, head_slice], attention.rn_k2)
        value = projections["v"][:, head_slice]
        pattern = ((q1 @ k1.T) * (q2 @ k2.T)) / attention._score_denom
        head_outputs.append(row_scale[:, None] * ((pattern * mask_array) @ value))
    concatenated = np.concatenate(head_outputs, axis=-1)
    output = concatenated @ weight(attention.wo).T + bias(attention.wo)
    return head_outputs, output


def torch_head_reference(
    attention: Any, normalized_input: torch.Tensor, mask: torch.Tensor
) -> tuple[list[np.ndarray], np.ndarray]:
    def linear(layer: Any, array: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(
            array,
            layer.weight.detach().double(),
            layer.bias.detach().double(),
        )

    def rational_norm(array: torch.Tensor, module: Any) -> torch.Tensor:
        mean_square = array.square().mean(dim=-1, keepdim=True) + float(module.eps)
        running_ms = module.running_ms.detach().double().clamp_min(1e-12)
        scaled = mean_square / running_ms
        numerator = sum(
            module.pa.detach().double()[degree] * scaled**degree
            for degree in range(module.deg + 1)
        )
        denominator = sum(
            module.pb.detach().double()[degree] * scaled**degree
            for degree in range(module.deg + 1)
        )
        if not torch.isfinite(denominator).all() or bool((denominator == 0).any()):
            raise RuntimeError("PyTorch float64 reference denominator is invalid")
        return array * (numerator / denominator) * torch.rsqrt(running_ms)

    with torch.inference_mode():
        x = normalized_input.double()
        batch, sequence_length, _ = x.shape

        def project(layer: Any) -> torch.Tensor:
            return linear(layer, x).view(
                batch,
                sequence_length,
                attention.n_heads,
                attention.head_dim,
            ).transpose(1, 2)

        q1 = rational_norm(project(attention.wq1), attention.rn_q1)
        k1 = rational_norm(project(attention.wk1), attention.rn_k1)
        q2 = rational_norm(project(attention.wq2), attention.rn_q2)
        k2 = rational_norm(project(attention.wk2), attention.rn_k2)
        value = project(attention.wv)
        mask64 = mask.double()
        a1 = q1 @ k1.transpose(-1, -2)
        a2 = q2 @ k2.transpose(-1, -2)
        pattern = (a1 * a2) / attention._score_denom
        row_scale = attention._row_scale_vec(mask64).view(1, 1, mask64.shape[0], 1)
        heads = row_scale * ((pattern * mask64) @ value)
        concatenated = heads.transpose(1, 2).reshape(
            batch, sequence_length, attention.dim
        )
        deployed = linear(attention.wo, concatenated)
    return (
        [heads[0, index].detach().double().cpu().numpy() for index in range(attention.n_heads)],
        deployed[0].detach().double().cpu().numpy(),
    )


def errors(reference: np.ndarray, reconstruction: np.ndarray) -> dict[str, float | bool]:
    difference = reference - reconstruction
    finite = bool(
        np.isfinite(reference).all()
        and np.isfinite(reconstruction).all()
        and np.isfinite(difference).all()
    )
    if not finite:
        return {"finite": False, "max_abs_error": float("inf"), "relative_l2_error": float("inf")}
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
        raise RuntimeError("Joint module count differs from the frozen protocol")
    for block in model.backbone.blocks:
        if block.attn.n_heads != PROTOCOL["heads_per_module"]:
            raise RuntimeError("Joint head count differs from the frozen protocol")
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
            normalized = block.rbn_attn(hidden)
            reconstructed_heads, reconstructed_output = independent_reconstruction(
                block.attn, normalized, mask
            )
            reference_heads, reference_output = torch_head_reference(
                block.attn, normalized, mask
            )
            head_rows = []
            for head_index, (reference, reconstruction) in enumerate(
                zip(reference_heads, reconstructed_heads)
            ):
                head_rows.append(
                    {
                        "head_index": head_index,
                        **errors(reference, reconstruction),
                    }
                )
            module_errors = errors(reference_output, reconstructed_output)
            row = {
                "selection_rank": input_identity["selection_rank"],
                "selection_sha256": input_identity["selection_sha256"],
                "block_index": block_index,
                "stack": "joint",
                "sequence_length": int(normalized.shape[1]),
                "dim": int(block.attn.dim),
                "head_dim": int(block.attn.head_dim),
                "head_count": int(block.attn.n_heads),
                **module_errors,
                "heads": head_rows,
            }
            rows.append(row)
            print("CONV_JOINT_EXACT", json.dumps(row), flush=True)
            hidden = block(hidden, mask=mask)
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
        input_row = {
            **identity,
            "instruction": cache_tasks[task_index],
            "instruction_ids": instruction_ids,
        }
        input_rows.append(input_row)
        tensors = tensorize_sample(sample, encode, cache_tasks, stats)
        module_rows.extend(audit_input(model, tensors, identity))

    all_error_rows = [
        {"finite": row["finite"], "relative_l2_error": row["relative_l2_error"]}
        for row in module_rows
    ]
    all_error_rows.extend(
        {"finite": head["finite"], "relative_l2_error": head["relative_l2_error"]}
        for row in module_rows
        for head in row["heads"]
    )
    all_finite = all(bool(row["finite"]) for row in all_error_rows)
    max_relative_l2_error = max(float(row["relative_l2_error"]) for row in all_error_rows)
    passed = all_finite and max_relative_l2_error <= PROTOCOL["relative_l2_gate"]
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
        "module_rows": module_rows,
        "aggregate": {
            "inputs_audited": len(input_rows),
            "module_evaluations": len(module_rows),
            "head_evaluations": sum(len(row["heads"]) for row in module_rows),
            "all_finite": all_finite,
            "max_module_relative_l2_error": max(
                float(row["relative_l2_error"]) for row in module_rows
            ),
            "max_head_relative_l2_error": max(
                float(head["relative_l2_error"])
                for row in module_rows
                for head in row["heads"]
            ),
            "max_relative_l2_error": max_relative_l2_error,
            "relative_l2_gate": PROTOCOL["relative_l2_gate"],
            "passed": passed,
        },
        "elapsed_s": time.perf_counter() - started,
        "scope": (
            "Independent layerwise numerical reproduction from learned weights for every "
            "joint bilinear-attention head on the frozen inputs in one convolutional χ-VLA "
            "checkpoint. This does not reconstruct the convolutional stem, FFN branches, "
            "residual stack, action head, or a single compact symbolic contraction for the "
            "whole policy."
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
