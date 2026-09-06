#!/usr/bin/env python3
"""Frozen modality decomposition of deployed ViT joint bilinear attention.

For each selected LIBERO-Object cache input, this audit partitions every
action-query attention update into an exhaustive set of source-token groups:
vision, instruction, robot state, and earlier/current action queries.  The
partition is performed before the learned output projection.  The same groups
are then carried through the linear output projection, with its learned bias
reported separately as the only source-independent constant.

The only favorable numerical gate is algebraic fidelity.  Contribution
magnitudes, signs, cosines, and prompt-permutation responses are descriptive.
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
import torch.nn.functional as F

from athena.libero_dataset_metadata import load_dataset_task_languages
from athena.run_xvla_experiment import (
    build_vocab,
    file_sha256,
    load_cache_statistics,
    load_suite,
    make_config,
    task_languages,
)
from xvla.models.vla import ChiVLA
from xvla.nn.attention import causal_mask

SCHEMA = "xvla-vit-modality-contribution-audit-v1"
PROTOCOL_ID = b"xvla-vit-modality-contribution-audit-v1\0"
SOURCE_GROUPS = ("vision", "instruction", "robot_state", "action_query")
FULL_TASK_QUOTAS = {str(task): 13 if task < 8 else 12 for task in range(10)}
SMOKE_TASK_QUOTAS = {str(task): 1 for task in range(10)}
PROTOCOL = {
    "suite": "libero_object",
    "architecture": "chi",
    "vision_encoder": "vit",
    "resolution": 64,
    "action_horizon": 8,
    "matmul_precision": "highest",
    "analysis_batch_size": 8,
    "full_inputs_per_checkpoint": 128,
    "full_official_task_quotas": FULL_TASK_QUOTAS,
    "smoke_inputs_per_checkpoint": 10,
    "smoke_official_task_quotas": SMOKE_TASK_QUOTAS,
    "input_selection": (
        "Within each official task, lexicographically smallest SHA256(protocol_id || "
        "checkpoint_sha256 || canonical_cache_record_sha256), with fixed 13/12 full "
        "quotas and one-per-task smoke quotas"
    ),
    "joint_blocks_per_input": 8,
    "heads_per_block": 12,
    "token_layout": {
        "vision": {"start": 0, "stop": 64, "tokens": 64},
        "instruction": {
            "start": 64,
            "stop": 97,
            "tokens": 33,
            "contents": "one learned BOS token plus 32 encoded instruction positions",
        },
        "robot_state": {
            "start": 97,
            "stop": 99,
            "tokens": 2,
            "contents": "one projected 8D robot-state token plus one embodiment token",
        },
        "action_query": {"start": 99, "stop": 107, "tokens": 8},
        "sequence_length": 107,
    },
    "source_groups": list(SOURCE_GROUPS),
    "head_reference": (
        "float64 direct bilinear-attention contraction on the deployed normalized input"
    ),
    "module_reference": (
        "deployed float32 attention output compared with the float64 grouped contraction; "
        "the output-projection bias is audited as a separate constant and the block's "
        "residual gain is verified on the reconstructed update"
    ),
    "metrics": {
        "coherent_energy_fraction": "||sum_token c_source||_F^2 / sum_source ||sum_token c_source||_F^2",
        "token_energy_fraction": "sum_visible_token ||c_token||_2^2 / sum_source sum_visible_token ||c_token||_2^2",
        "per_source_token_energy_fraction": (
            "(sum_visible_token ||c_token||_2^2 / visible_source_token_incidences) / "
            "sum_source (sum_visible_token ||c_token||_2^2 / visible_source_token_incidences)"
        ),
        "signed_projection_fraction": "<c_source, full_update> / ||full_update||_F^2",
        "cosine_with_total": "<c_source, full_update> / (||c_source||_F ||full_update||_F)",
    },
    "relative_l2_gate": 1e-6,
    "favorable_magnitude_gate": None,
    "prompt_permutation": {
        "enabled": True,
        "observations_per_checkpoint": 10,
        "selection": "lowest-ranked main-audit input in each official task",
        "prompts": "all ten official LIBERO-Object prompts on every fixed observation",
        "interpretation": "descriptive same-observation characterization, not grounding",
        "favorable_threshold": None,
    },
}
FROZEN_CACHE = {
    "basename": "libero_frames_100000_64.pkl",
    "sha256": "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662",
    "frames": 66984,
}
FROZEN_PROVENANCE_JOB_ID = "830988"
FROZEN_PROVENANCE_BASENAME = "cache_provenance_libero_object.json"
FROZEN_CAPABILITY_MANIFEST = {
    "basename": "manifest.json",
    "sha256": "c23a9ba8bf791d55b0a731743fdac7810395d0030fdd6225ee98b7ec6da42453",
}
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
        "basename": "ckpt_linear_rat_vit_s0_v2.pt",
        "sha256": "96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9",
    },
    1: {
        "basename": "ckpt_linear_rat_vit_s1.pt",
        "sha256": "cc0780b989165a80a449c03cbb3574e2f56b2e5747a64cf32d1fde418df8ec3c",
    },
    2: {
        "basename": "ckpt_linear_rat_vit_s2.pt",
        "sha256": "413a770071bd8f16b924c5604f7c7a58567d2eb6c40c66aab202bcf046fec910",
    },
}
SOURCE_FILES = (
    "athena/run_vit_modality_contribution_audit.py",
    "athena/summarize_vit_modality_contribution_audit.py",
    "athena/slurm_vit_modality_contribution_audit.sbatch",
    "athena/slurm_summarize_vit_modality_contribution_audit.sbatch",
    "athena/launch_vit_modality_contribution_audit.sh",
    "athena/VIT_MODALITY_CONTRIBUTION_PROTOCOL.md",
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
    parser.add_argument("--capability-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("strict_smoke", "full"), required=True)
    return parser.parse_args()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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


def hash_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def validate_dataset_metadata(metadata: dict[str, Any]) -> None:
    if metadata != FROZEN_DATASET_METADATA:
        raise RuntimeError(
            "Pinned dataset task metadata differs from the frozen identity"
        )


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
    if result.get("suite") != PROTOCOL["suite"] or result.get("verified") is not True:
        errors.append("Object provenance did not pass")
    if cache_result.get("sha256") != expected_cache_sha256:
        errors.append("provenance cache SHA differs")
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
        errors.append("cache/source canonical hashes did not match")
    if cache_result.get("canonical_content_sha256") != source_result.get(
        "canonical_content_sha256"
    ):
        errors.append("cache/source canonical content hashes differ")
    if metadata.get("sha256") != FROZEN_DATASET_METADATA["metadata_sha256"]:
        errors.append("provenance metadata SHA differs")
    if (
        metadata.get("dataset_to_official_task")
        != FROZEN_DATASET_METADATA["dataset_to_official_task"]
    ):
        errors.append("provenance task map differs")
    if not mismatch_counts or any(
        int(value) != 0 for value in mismatch_counts.values()
    ):
        errors.append("provenance mismatch counts are absent or nonzero")
    if file_sha256(cache) != expected_cache_sha256:
        errors.append("live cache SHA differs")
    if errors:
        raise RuntimeError("Invalid frozen Object provenance: " + "; ".join(errors))
    return result


def validate_capability_manifest(path: Path) -> dict[str, Any]:
    if path.name != FROZEN_CAPABILITY_MANIFEST["basename"]:
        raise RuntimeError("Capability manifest basename differs")
    if file_sha256(path) != FROZEN_CAPABILITY_MANIFEST["sha256"]:
        raise RuntimeError("Capability manifest SHA differs")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "anonymous-vit-capability-artifact-v1":
        raise RuntimeError("Capability manifest schema differs")
    expected_checkpoints = [
        {
            "path": f"artifacts/{FROZEN_CHECKPOINTS[seed]['basename']}",
            "seed": seed,
            "sha256": FROZEN_CHECKPOINTS[seed]["sha256"],
        }
        for seed in sorted(FROZEN_CHECKPOINTS)
    ]
    if manifest.get("checkpoint_identities") != expected_checkpoints:
        raise RuntimeError("Capability manifest checkpoint identities differ")
    cache_identity = manifest.get("cache_identity", {})
    if Path(str(cache_identity.get("path", ""))).name != FROZEN_CACHE["basename"]:
        raise RuntimeError("Capability manifest cache basename differs")
    if cache_identity.get("sha256") != FROZEN_CACHE["sha256"]:
        raise RuntimeError("Capability manifest cache SHA differs")
    return manifest


def validate_frozen_inputs(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    expected = FROZEN_CHECKPOINTS.get(args.checkpoint_seed)
    if expected is None:
        raise ValueError("Checkpoint seed must be 0, 1, or 2")
    errors = []
    if args.checkpoint.name != expected["basename"]:
        errors.append("checkpoint basename differs")
    if args.expected_checkpoint_sha256 != expected["sha256"]:
        errors.append("declared checkpoint SHA differs")
    if (
        not args.checkpoint.is_file()
        or file_sha256(args.checkpoint) != expected["sha256"]
    ):
        errors.append("live checkpoint SHA differs")
    if args.cache.name != FROZEN_CACHE["basename"]:
        errors.append("cache basename differs")
    if args.expected_cache_sha256 != FROZEN_CACHE["sha256"]:
        errors.append("declared cache SHA differs")
    if args.expected_provenance_job_id != FROZEN_PROVENANCE_JOB_ID:
        errors.append("provenance job identity differs")
    if args.provenance_result.name != FROZEN_PROVENANCE_BASENAME:
        errors.append("provenance result basename differs")
    if args.output.exists():
        errors.append("output already exists")
    if errors:
        raise RuntimeError("Frozen identity validation failed: " + "; ".join(errors))
    validate_capability_manifest(args.capability_manifest)
    provenance = validate_provenance(
        args.provenance_result, args.cache, args.expected_cache_sha256
    )
    return expected["sha256"], provenance


def cache_record_identity(
    record: Any, cache_row_index: int, dataset_to_official: dict[str, int]
) -> dict[str, Any]:
    if len(record) != 6:
        raise RuntimeError(f"Cache row {cache_row_index} does not have six fields")
    episode_index = int(record[0])
    frame_index = int(record[1])
    dataset_task_index = int(record[5])
    if str(dataset_task_index) not in dataset_to_official:
        raise RuntimeError(f"Cache row {cache_row_index} has an unmapped dataset task")
    official_task_index = int(dataset_to_official[str(dataset_task_index)])
    image = np.asarray(record[2])
    state = np.asarray(record[3])
    action = np.asarray(record[4])
    if image.shape != (64, 64, 3) or image.dtype != np.uint8:
        raise RuntimeError(f"Cache row {cache_row_index} has an invalid image")
    if state.shape != (8,) or action.shape != (7,):
        raise RuntimeError(
            f"Cache row {cache_row_index} has invalid state/action shape"
        )
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise RuntimeError(f"Cache row {cache_row_index} contains non-finite values")
    image_sha256 = hash_array(image)
    state_sha256 = hash_array(state)
    action_sha256 = hash_array(action)
    digest = hashlib.sha256()
    digest.update(PROTOCOL_ID)
    for label, value in (
        ("cache_row_index", cache_row_index),
        ("episode_index", episode_index),
        ("frame_index", frame_index),
        ("dataset_task_index", dataset_task_index),
        ("official_task_index", official_task_index),
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
        "dataset_task_index": dataset_task_index,
        "official_task_index": official_task_index,
        "image_sha256": image_sha256,
        "state_sha256": state_sha256,
        "action_sha256": action_sha256,
        "canonical_record_sha256": digest.hexdigest(),
    }


def rank_cache_inputs(
    records: list[Any], checkpoint_sha256: str, task_quotas: dict[str, int]
) -> list[dict[str, Any]]:
    dataset_to_official = FROZEN_DATASET_METADATA["dataset_to_official_task"]
    ranked_by_task: dict[int, list[dict[str, Any]]] = {task: [] for task in range(10)}
    for cache_row_index, record in enumerate(records):
        identity = cache_record_identity(record, cache_row_index, dataset_to_official)
        digest = hashlib.sha256()
        digest.update(PROTOCOL_ID)
        digest.update(checkpoint_sha256.encode("ascii") + b"\0")
        digest.update(identity["canonical_record_sha256"].encode("ascii"))
        identity["selection_sha256"] = digest.hexdigest()
        ranked_by_task[identity["official_task_index"]].append(identity)
    selected = []
    for official_task_index in range(10):
        rows = ranked_by_task[official_task_index]
        rows.sort(key=lambda row: (row["selection_sha256"], row["cache_row_index"]))
        quota = int(task_quotas[str(official_task_index)])
        if len(rows) < quota:
            raise RuntimeError(
                f"Official task {official_task_index} has {len(rows)} rows, needs {quota}"
            )
        for within_task_rank, row in enumerate(rows[:quota]):
            row["selection_rank_within_official_task"] = within_task_rank
            selected.append(row)
    for global_rank, row in enumerate(selected):
        row["selection_rank"] = global_rank
    return selected


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
    state_dict = torch.load(checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if model.cfg.vision_encoder != "vit" or model.cfg.n_phases != 0:
        raise RuntimeError(
            "Audit requires the deployed single-vision, no-phase ViT policy"
        )
    if model.cfg.dual_vision:
        raise RuntimeError("Audit does not permit a dual-vision token layout")
    if model.cfg.max_instr_len != 32 or model.cfg.action_horizon != 8:
        raise RuntimeError(
            "Instruction or action-query count differs from the frozen layout"
        )
    if model.vision.cfg.num_patches != 64 or model.vision.use_cls:
        raise RuntimeError(
            "ViT token count or CLS-token setting differs from the frozen layout"
        )
    if model.seq_len != PROTOCOL["token_layout"]["sequence_length"]:
        raise RuntimeError("Model sequence length differs from the frozen layout")
    if len(model.backbone.blocks) != PROTOCOL["joint_blocks_per_input"]:
        raise RuntimeError("Joint block count differs from the frozen protocol")
    for block in model.backbone.blocks:
        if block.attn.n_heads != PROTOCOL["heads_per_block"]:
            raise RuntimeError("Joint head count differs from the frozen protocol")
        if block.attn.qk_norm != "rational":
            raise RuntimeError("Audit requires deployed rational QK normalization")
    return model


def tensorize_records(
    records: list[Any],
    identities: list[dict[str, Any]],
    encode: Any,
    official_tasks: dict[int, str],
    stats: dict[str, Any],
    prompt_tasks: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if prompt_tasks is not None and len(prompt_tasks) != len(identities):
        raise ValueError("Prompt task count differs from the input count")
    images = []
    instructions = []
    states = []
    for index, identity in enumerate(identities):
        record = records[identity["cache_row_index"]]
        images.append(np.asarray(record[2]))
        prompt_task = (
            identity["official_task_index"]
            if prompt_tasks is None
            else prompt_tasks[index]
        )
        instructions.append(encode(official_tasks[int(prompt_task)]))
        states.append(
            (np.asarray(record[3], dtype=np.float32) - stats["state_mean"])
            / stats["state_std"]
        )
    image_tensor = (
        torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div(255).cuda()
    )
    instruction_tensor = torch.tensor(instructions, dtype=torch.long, device="cuda")
    state_tensor = torch.tensor(np.stack(states), dtype=torch.float32, device="cuda")
    embodiment = torch.zeros(len(identities), dtype=torch.long, device="cuda")
    return image_tensor, instruction_tensor, state_tensor, embodiment


def build_initial_hidden(
    model: ChiVLA,
    image: torch.Tensor,
    instruction: torch.Tensor,
    state: torch.Tensor,
    embodiment: torch.Tensor,
) -> torch.Tensor:
    batch = image.shape[0]
    components = [
        model._visual_tokens(image),
        model.bos.expand(batch, -1, -1),
        model.tok_emb(instruction),
        model.state_proj(state)[:, None],
        model.embodiment_emb(embodiment)[:, None],
        model.action_queries.expand(batch, -1, -1),
    ]
    hidden = torch.cat(components, dim=1)
    if hidden.shape[1] != PROTOCOL["token_layout"]["sequence_length"]:
        raise RuntimeError("Constructed hidden sequence length differs")
    return hidden + model.pos_emb[:, : hidden.shape[1]]


def confirm_deployed_token_layout(
    model: ChiVLA,
    sample_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    image, instruction, state, embodiment = (tensor[:1] for tensor in sample_tensors)
    manual = build_initial_hidden(model, image, instruction, state, embodiment)
    captured: dict[str, torch.Tensor] = {}

    def capture(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if "hidden" not in captured:
            captured["hidden"] = args[0].detach().clone()
            captured["mask"] = kwargs["mask"].detach().clone()

    handle = model.backbone.blocks[0].register_forward_pre_hook(
        capture, with_kwargs=True
    )
    try:
        with torch.inference_mode():
            model(image, instruction, state, embodiment)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("Failed to capture the deployed first joint-block input")
    difference = captured["hidden"] - manual
    max_abs = float(difference.abs().max().item())
    relative = float(
        torch.linalg.vector_norm(difference).item()
        / max(torch.linalg.vector_norm(captured["hidden"]).item(), 1e-30)
    )
    expected_mask = causal_mask(
        manual.shape[1], device=manual.device, dtype=manual.dtype
    )
    mask_exact = bool(torch.equal(captured["mask"], expected_mask))
    hidden_exact = bool(torch.equal(captured["hidden"], manual))
    if not hidden_exact or not mask_exact:
        raise RuntimeError(
            "Manual source-token grouping does not match the deployed joint sequence exactly"
        )
    layout = PROTOCOL["token_layout"]
    if [layout[group]["start"] for group in SOURCE_GROUPS] != [0, 64, 97, 99]:
        raise RuntimeError("Frozen source-group starts differ")
    if [layout[group]["stop"] for group in SOURCE_GROUPS] != [64, 97, 99, 107]:
        raise RuntimeError("Frozen source-group stops differ")
    return {
        "deployed_manual_hidden_exact": hidden_exact,
        "deployed_causal_mask_exact": mask_exact,
        "max_abs_error": max_abs,
        "relative_l2_error": relative,
        "sequence_length": int(manual.shape[1]),
        "source_groups": {group: layout[group] for group in SOURCE_GROUPS},
    }


def rational_norm_float64(array: torch.Tensor, module: Any) -> torch.Tensor:
    if getattr(module, "variant", None) != "pade":
        raise RuntimeError("Audit requires Padé RationalNorm")
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
        raise RuntimeError("RationalNorm denominator is non-finite or zero")
    return array * (numerator / denominator) * torch.rsqrt(running_ms)


def relative_errors(
    reference: torch.Tensor, reconstruction: torch.Tensor, reduce_dims: tuple[int, ...]
) -> dict[str, torch.Tensor]:
    difference = reference - reconstruction
    finite = (
        torch.isfinite(reference).all(dim=reduce_dims)
        & torch.isfinite(reconstruction).all(dim=reduce_dims)
        & torch.isfinite(difference).all(dim=reduce_dims)
    )
    max_abs = difference.abs().amax(dim=reduce_dims)
    relative = torch.linalg.vector_norm(
        difference, dim=reduce_dims
    ) / torch.linalg.vector_norm(reference, dim=reduce_dims).clamp_min(1e-30)
    return {"finite": finite, "max_abs_error": max_abs, "relative_l2_error": relative}


def source_metrics(
    grouped: torch.Tensor,
    total: torch.Tensor,
    token_energy: torch.Tensor | None,
    visible_incidences: list[int] | None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return vectorized metrics for grouped ``(B,H,G,Q,D)`` tensors."""

    coherent_energy = grouped.square().sum(dim=(-1, -2))
    coherent_denominator = coherent_energy.sum(dim=-1).clamp_min(1e-30)
    full_energy = total.square().sum(dim=(-1, -2)).clamp_min(1e-30)
    dot = (grouped * total.unsqueeze(2)).sum(dim=(-1, -2))
    cosine_denominator = torch.sqrt(
        coherent_energy.clamp_min(1e-30) * full_energy.unsqueeze(-1)
    )
    values: dict[str, torch.Tensor] = {
        "coherent_energy": coherent_energy,
        "coherent_energy_fraction": coherent_energy
        / coherent_denominator.unsqueeze(-1),
        "signed_projection_numerator": dot,
        "signed_projection_fraction": dot / full_energy.unsqueeze(-1),
        "cosine_denominator": cosine_denominator,
        "cosine_with_total": dot / cosine_denominator,
    }
    denominators: dict[str, torch.Tensor] = {
        "coherent_energy_denominator": coherent_denominator,
        "signed_projection_denominator": full_energy,
    }
    if token_energy is not None:
        if visible_incidences is None:
            raise ValueError("Visible incidences are required with token energy")
        incidence = token_energy.new_tensor(visible_incidences).view(1, 1, -1)
        mean_token_energy = token_energy / incidence
        token_energy_denominator = token_energy.sum(dim=-1).clamp_min(1e-30)
        per_token_denominator = mean_token_energy.sum(dim=-1).clamp_min(1e-30)
        values.update(
            {
                "token_energy": token_energy,
                "token_energy_fraction": token_energy
                / token_energy_denominator.unsqueeze(-1),
                "mean_visible_source_token_energy": mean_token_energy,
                "per_source_token_energy_fraction": mean_token_energy
                / per_token_denominator.unsqueeze(-1),
            }
        )
        denominators.update(
            {
                "token_energy_denominator": token_energy_denominator,
                "per_source_token_energy_denominator": per_token_denominator,
            }
        )
    return values, denominators


def decompose_attention_batch(
    attention: Any,
    normalized_input: torch.Tensor,
    mask: torch.Tensor,
    residual_gain: torch.Tensor,
) -> dict[str, Any]:
    batch, sequence_length, _ = normalized_input.shape
    layout = PROTOCOL["token_layout"]
    query_start = layout["action_query"]["start"]
    query_stop = layout["action_query"]["stop"]
    x = normalized_input.double()

    def project(layer: Any) -> torch.Tensor:
        return (
            F.linear(x, layer.weight.detach().double(), layer.bias.detach().double())
            .view(batch, sequence_length, attention.n_heads, attention.head_dim)
            .transpose(1, 2)
        )

    q1 = rational_norm_float64(project(attention.wq1), attention.rn_q1)
    k1 = rational_norm_float64(project(attention.wk1), attention.rn_k1)
    q2 = rational_norm_float64(project(attention.wq2), attention.rn_q2)
    k2 = rational_norm_float64(project(attention.wk2), attention.rn_k2)
    value = project(attention.wv)
    mask64 = mask.double()
    query_mask = mask64[query_start:query_stop]
    pattern = (
        (q1[:, :, query_start:query_stop] @ k1.transpose(-1, -2))
        * (q2[:, :, query_start:query_stop] @ k2.transpose(-1, -2))
        / attention._score_denom
    )
    row_scale = attention._row_scale_vec(mask64)[query_start:query_stop]
    weights = pattern * query_mask.view(1, 1, query_stop - query_start, sequence_length)
    weights = weights * row_scale.view(1, 1, query_stop - query_start, 1)
    token_contributions = weights.unsqueeze(-1) * value.unsqueeze(2)
    full_heads = weights @ value

    group_tensors = []
    token_energies = []
    visible_incidences = []
    for source in SOURCE_GROUPS:
        start = int(layout[source]["start"])
        stop = int(layout[source]["stop"])
        source_tokens = token_contributions[:, :, :, start:stop]
        group_tensors.append(source_tokens.sum(dim=3))
        token_energies.append(source_tokens.square().sum(dim=(-1, -2, -3)))
        visible_incidences.append(int(query_mask[:, start:stop].sum().item()))
    grouped_heads = torch.stack(group_tensors, dim=2)
    token_energy = torch.stack(token_energies, dim=-1)
    head_group_sum = grouped_heads.sum(dim=2)
    head_errors = relative_errors(full_heads, head_group_sum, (-1, -2))
    head_values, head_denominators = source_metrics(
        grouped_heads, full_heads, token_energy, visible_incidences
    )

    module_groups = []
    for group_index in range(len(SOURCE_GROUPS)):
        concatenated = (
            grouped_heads[:, :, group_index]
            .transpose(1, 2)
            .reshape(batch, query_stop - query_start, attention.dim)
        )
        module_groups.append(
            F.linear(concatenated, attention.wo.weight.detach().double(), bias=None)
        )
    grouped_modules = torch.stack(module_groups, dim=1)
    module_prebias = grouped_modules.sum(dim=1)
    output_bias = attention.wo.bias.detach().double().view(1, 1, -1)
    module_full = module_prebias + output_bias
    full_concatenated = full_heads.transpose(1, 2).reshape(
        batch, query_stop - query_start, attention.dim
    )
    reference_prebias = F.linear(
        full_concatenated, attention.wo.weight.detach().double(), bias=None
    )
    reference_full = reference_prebias + output_bias
    deployed_full = attention(normalized_input, mask=mask, method="explicit")[
        :, query_start:query_stop
    ].double()
    gain64 = residual_gain.detach().double()
    reconstructed_residual_update = gain64 * module_full
    deployed_residual_update = (
        residual_gain
        * attention(normalized_input, mask=mask, method="explicit")[
            :, query_start:query_stop
        ]
    ).double()
    module_prebias_errors = relative_errors(reference_prebias, module_prebias, (-1, -2))
    module_full_errors = relative_errors(reference_full, module_full, (-1, -2))
    deployed_errors = relative_errors(deployed_full, module_full, (-1, -2))
    residual_update_errors = relative_errors(
        deployed_residual_update, reconstructed_residual_update, (-1, -2)
    )
    module_values, module_denominators = source_metrics(
        grouped_modules.unsqueeze(1), reference_prebias.unsqueeze(1), None, None
    )

    all_tensors = (
        [
            token_contributions,
            full_heads,
            grouped_heads,
            module_prebias,
            module_full,
            deployed_full,
            reconstructed_residual_update,
            deployed_residual_update,
        ]
        + list(head_values.values())
        + list(head_denominators.values())
        + list(module_values.values())
        + list(module_denominators.values())
        + [
            tensor
            for errors in (
                head_errors,
                module_prebias_errors,
                module_full_errors,
                deployed_errors,
                residual_update_errors,
            )
            for tensor in errors.values()
        ]
    )
    if not all(bool(torch.isfinite(tensor).all()) for tensor in all_tensors):
        raise RuntimeError("A modality contribution tensor or metric is non-finite")
    return {
        "full_heads": full_heads,
        "grouped_heads": grouped_heads,
        "head_errors": head_errors,
        "head_values": head_values,
        "head_denominators": head_denominators,
        "visible_incidences": visible_incidences,
        "module_groups": grouped_modules,
        "module_values": {key: value[:, 0] for key, value in module_values.items()},
        "module_denominators": {
            key: value[:, 0] for key, value in module_denominators.items()
        },
        "module_prebias_errors": module_prebias_errors,
        "module_full_errors": module_full_errors,
        "deployed_errors": deployed_errors,
        "residual_update_errors": residual_update_errors,
        "residual_gain": float(gain64.item()),
        "output_bias_energy": float(output_bias.square().sum().item()),
    }


def scalar_error_row(
    errors: dict[str, torch.Tensor], batch_index: int, head_index: int | None = None
) -> dict[str, Any]:
    index: Any = batch_index if head_index is None else (batch_index, head_index)
    return {
        "finite": bool(errors["finite"][index].item()),
        "max_abs_error": float(errors["max_abs_error"][index].item()),
        "relative_l2_error": float(errors["relative_l2_error"][index].item()),
    }


def metric_catalog(
    values: dict[str, torch.Tensor],
    denominators: dict[str, torch.Tensor],
    batch_index: int,
    head_index: int | None,
    visible_incidences: list[int] | None,
) -> tuple[dict[str, Any], dict[str, float]]:
    source_rows = {}
    for source_index, source in enumerate(SOURCE_GROUPS):
        index: Any = (
            (batch_index, source_index)
            if head_index is None
            else (batch_index, head_index, source_index)
        )
        source_row = {
            key: float(value[index].item())
            for key, value in values.items()
            if key != "cosine_denominator"
        }
        source_row["cosine_denominator"] = float(
            values["cosine_denominator"][index].item()
        )
        if visible_incidences is not None:
            source_row["unique_source_tokens"] = int(
                PROTOCOL["token_layout"][source]["tokens"]
            )
            source_row["visible_source_token_incidences"] = int(
                visible_incidences[source_index]
            )
        source_rows[source] = source_row
    denominator_index: Any = (
        batch_index if head_index is None else (batch_index, head_index)
    )
    denominator_rows = {
        key: float(value[denominator_index].item())
        for key, value in denominators.items()
    }
    return source_rows, denominator_rows


def audit_batch(
    model: ChiVLA,
    sample_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    identities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    image, instruction, state, embodiment = sample_tensors
    with torch.inference_mode():
        hidden = build_initial_hidden(model, image, instruction, state, embodiment)
        mask = causal_mask(hidden.shape[1], device=hidden.device, dtype=hidden.dtype)
        rows = []
        for block_index, block in enumerate(model.backbone.blocks):
            normalized = block.rbn_attn(hidden)
            decomposition = decompose_attention_batch(
                block.attn, normalized, mask, block.attn_gain
            )
            for batch_index, identity in enumerate(identities):
                heads = []
                for head_index in range(PROTOCOL["heads_per_block"]):
                    source_rows, denominators = metric_catalog(
                        decomposition["head_values"],
                        decomposition["head_denominators"],
                        batch_index,
                        head_index,
                        decomposition["visible_incidences"],
                    )
                    heads.append(
                        {
                            "head_index": head_index,
                            "group_sum_error": scalar_error_row(
                                decomposition["head_errors"], batch_index, head_index
                            ),
                            "denominators": denominators,
                            "sources": source_rows,
                        }
                    )
                module_sources, module_denominators = metric_catalog(
                    decomposition["module_values"],
                    decomposition["module_denominators"],
                    batch_index,
                    None,
                    None,
                )
                rows.append(
                    {
                        "selection_rank": identity["selection_rank"],
                        "selection_sha256": identity["selection_sha256"],
                        "official_task_index": identity["official_task_index"],
                        "block_index": block_index,
                        "sequence_length": int(normalized.shape[1]),
                        "action_query_rows": [99, 107],
                        "model_dim": int(block.attn.dim),
                        "head_dim": int(block.attn.head_dim),
                        "head_count": int(block.attn.n_heads),
                        "heads": heads,
                        "module": {
                            "group_sum_prebias_error": scalar_error_row(
                                decomposition["module_prebias_errors"], batch_index
                            ),
                            "group_sum_plus_bias_error": scalar_error_row(
                                decomposition["module_full_errors"], batch_index
                            ),
                            "deployed_float32_vs_grouped_float64_error": scalar_error_row(
                                decomposition["deployed_errors"], batch_index
                            ),
                            "deployed_gain_scaled_update_error": scalar_error_row(
                                decomposition["residual_update_errors"], batch_index
                            ),
                            "attention_residual_gain": decomposition["residual_gain"],
                            "output_projection_bias_energy": decomposition[
                                "output_bias_energy"
                            ],
                            "denominators": module_denominators,
                            "sources": module_sources,
                        },
                    }
                )
            hidden = block(hidden, mask=mask)
            print(
                "MODALITY_PROGRESS",
                json.dumps(
                    {
                        "selection_ranks": [
                            row["selection_rank"] for row in identities
                        ],
                        "block_index": block_index,
                    }
                ),
                flush=True,
            )
    rows.sort(key=lambda row: (row["selection_rank"], row["block_index"]))
    return rows


def vector_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference_vector = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate_vector = np.asarray(candidate, dtype=np.float64).reshape(-1)
    difference = candidate_vector - reference_vector
    reference_norm = max(float(np.linalg.norm(reference_vector)), 1e-30)
    candidate_norm = max(float(np.linalg.norm(candidate_vector)), 1e-30)
    dot = float(np.dot(reference_vector, candidate_vector))
    return {
        "relative_l2_change": float(np.linalg.norm(difference) / reference_norm),
        "mean_absolute_change": float(np.mean(np.abs(difference))),
        "signed_projection": float(dot / (reference_norm * reference_norm)),
        "cosine": float(dot / (reference_norm * candidate_norm)),
        "reference_l2_denominator": reference_norm,
        "cosine_denominator": float(reference_norm * candidate_norm),
    }


def prompt_permutation_characterization(
    model: ChiVLA,
    records: list[Any],
    selected: list[dict[str, Any]],
    encode: Any,
    official_tasks: dict[int, str],
    stats: dict[str, Any],
) -> list[dict[str, Any]]:
    observations = [
        next(
            row
            for row in selected
            if row["official_task_index"] == task
            and row["selection_rank_within_official_task"] == 0
        )
        for task in range(10)
    ]
    action_mean = np.asarray(stats["action_mean"], dtype=np.float64)
    action_std = np.asarray(stats["action_std"], dtype=np.float64)
    predictions: dict[tuple[int, int], np.ndarray] = {}
    rows = []
    for observation in observations:
        identities = [observation for _ in range(10)]
        prompts = list(range(10))
        tensors = tensorize_records(
            records, identities, encode, official_tasks, stats, prompt_tasks=prompts
        )
        with torch.inference_mode():
            normalized, _ = model(*tensors)
        normalized_array = normalized.detach().double().cpu().numpy()
        physical_array = normalized_array * action_std.reshape(
            1, 1, -1
        ) + action_mean.reshape(1, 1, -1)
        for prompt_index in prompts:
            prediction = physical_array[prompt_index]
            if not np.isfinite(prediction).all():
                raise RuntimeError("Prompt-permutation prediction is non-finite")
            predictions[(observation["official_task_index"], prompt_index)] = prediction
    for observation in observations:
        observation_task = int(observation["official_task_index"])
        matched = predictions[(observation_task, observation_task)]
        for prompt_task in range(10):
            prediction = predictions[(observation_task, prompt_task)]
            instruction_ids = encode(official_tasks[prompt_task])
            rows.append(
                {
                    "observation_selection_rank": observation["selection_rank"],
                    "observation_selection_sha256": observation["selection_sha256"],
                    "observation_official_task_index": observation_task,
                    "prompt_official_task_index": prompt_task,
                    "prompt_matches_observation": prompt_task == observation_task,
                    "instruction": official_tasks[prompt_task],
                    "instruction_ids": instruction_ids,
                    "instruction_ids_sha256": hash_array(
                        np.asarray(instruction_ids, dtype=np.int64)
                    ),
                    "physical_action_prediction": prediction.tolist(),
                    "physical_action_prediction_sha256": hash_array(prediction),
                    "comparison_to_matched_prompt": vector_comparison(
                        matched, prediction
                    ),
                }
            )
    return rows


def collect_error_rows(module_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for module in module_rows:
        rows.extend(head["group_sum_error"] for head in module["heads"])
        rows.extend(
            module["module"][key]
            for key in (
                "group_sum_prebias_error",
                "group_sum_plus_bias_error",
                "deployed_float32_vs_grouped_float64_error",
                "deployed_gain_scaled_update_error",
            )
        )
    return rows


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    initial_source_hashes = source_hashes()
    checkpoint_sha256, provenance = validate_frozen_inputs(args)
    provenance_result_sha256 = file_sha256(args.provenance_result)
    capability_manifest_sha256 = file_sha256(args.capability_manifest)
    task_quotas = (
        PROTOCOL["smoke_official_task_quotas"]
        if args.mode == "strict_smoke"
        else PROTOCOL["full_official_task_quotas"]
    )
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_float32_matmul_precision(PROTOCOL["matmul_precision"])
    if not torch.cuda.is_available():
        raise RuntimeError("The modality audit requires a CUDA compute node")

    with args.cache.open("rb") as handle:
        records = pickle.load(handle)
    if len(records) != FROZEN_CACHE["frames"]:
        raise RuntimeError("Live cache frame count differs from the frozen count")
    selected = rank_cache_inputs(records, checkpoint_sha256, task_quotas)
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
    for identity in selected:
        official_task = int(identity["official_task_index"])
        input_rows.append(
            {
                **identity,
                "instruction": official_tasks[official_task],
                "instruction_ids": encode(official_tasks[official_task]),
            }
        )
    first_tensors = tensorize_records(
        records, selected[:1], encode, official_tasks, stats
    )
    layout_confirmation = confirm_deployed_token_layout(model, first_tensors)

    module_rows = []
    batch_size = int(PROTOCOL["analysis_batch_size"])
    for start in range(0, len(selected), batch_size):
        batch_identities = selected[start : start + batch_size]
        tensors = tensorize_records(
            records, batch_identities, encode, official_tasks, stats
        )
        module_rows.extend(audit_batch(model, tensors, batch_identities))
    module_rows.sort(key=lambda row: (row["selection_rank"], row["block_index"]))

    prompt_rows = []
    if args.mode == "full" and PROTOCOL["prompt_permutation"]["enabled"]:
        prompt_rows = prompt_permutation_characterization(
            model, records, selected, encode, official_tasks, stats
        )

    error_rows = collect_error_rows(module_rows)
    all_finite = all(row["finite"] is True for row in error_rows)
    max_relative_l2_error = max(float(row["relative_l2_error"]) for row in error_rows)
    prompt_all_finite = all(
        math.isfinite(value)
        for row in prompt_rows
        for value in row["comparison_to_matched_prompt"].values()
    )
    passed = (
        all_finite
        and prompt_all_finite
        and max_relative_l2_error <= PROTOCOL["relative_l2_gate"]
    )
    if source_hashes() != initial_source_hashes:
        raise RuntimeError("Runner or imported source changed during the audit")
    if file_sha256(args.checkpoint) != checkpoint_sha256:
        raise RuntimeError("Checkpoint changed during the audit")
    if file_sha256(args.cache) != args.expected_cache_sha256:
        raise RuntimeError("Cache changed during the audit")
    if file_sha256(args.provenance_result) != provenance_result_sha256:
        raise RuntimeError("Provenance result changed during the audit")
    if file_sha256(args.capability_manifest) != capability_manifest_sha256:
        raise RuntimeError("Capability manifest changed during the audit")

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
            "capability_manifest": str(args.capability_manifest),
            "capability_manifest_sha256": capability_manifest_sha256,
            "dataset_metadata": dataset_metadata,
            "official_task_languages": official_tasks,
            "source_sha256": initial_source_hashes,
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
        "layout_confirmation": layout_confirmation,
        "inputs": input_rows,
        "module_rows": module_rows,
        "prompt_permutation_rows": prompt_rows,
        "aggregate": {
            "inputs_audited": len(input_rows),
            "official_task_counts": {
                str(task): sum(row["official_task_index"] == task for row in input_rows)
                for task in range(10)
            },
            "module_evaluations": len(module_rows),
            "head_evaluations": sum(len(row["heads"]) for row in module_rows),
            "source_group_evaluations": sum(
                len(head["sources"]) for row in module_rows for head in row["heads"]
            ),
            "prompt_permutation_evaluations": len(prompt_rows),
            "all_finite": all_finite,
            "prompt_all_finite": prompt_all_finite,
            "max_relative_l2_error": max_relative_l2_error,
            "relative_l2_gate": PROTOCOL["relative_l2_gate"],
            "passed": passed,
        },
        "elapsed_s": time.perf_counter() - started,
        "scope": (
            "This is an input-conditioned exact additive decomposition of joint-attention "
            "updates on deterministic training-domain cache inputs. BOS is grouped with the "
            "instruction and embodiment with robot state so the four source groups exhaust the "
            "deployed sequence. The output-projection bias is a separately audited constant. "
            "Magnitude, sign, cosine, and official-prompt permutation statistics are descriptive. "
            "They do not establish grounding, causal necessity, human-nameable features, or "
            "end-to-end policy decomposition."
        ),
    }
    write_json(args.output, result)
    print(
        "RESULT_AGGREGATE", json.dumps(result["aggregate"], sort_keys=True), flush=True
    )
    if not passed:
        raise RuntimeError(
            f"Frozen fidelity gate failed: finite={all_finite}, prompt_finite={prompt_all_finite}, "
            f"max relative L2={max_relative_l2_error:.9g}"
        )


if __name__ == "__main__":
    main()
