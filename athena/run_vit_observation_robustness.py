#!/usr/bin/env python3
"""Paired closed-loop camera-shift robustness for frozen ViT policies."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
from PIL import Image

from athena.libero_dataset_metadata import load_dataset_task_languages
from athena.run_xvla_experiment import (
    array_sha256,
    build_robot_state,
    build_vocab,
    file_sha256,
    load_cache_statistics,
    load_suite,
    make_config,
    official_init_states,
    task_languages,
)
from xvla.models.vla import ChiVLA

SCHEMA = "xvla-vit-observation-robustness-v1"
PROTOCOL_ID = b"xvla-vit-observation-robustness-v1\0"
CONDITIONS = (
    "clean",
    "brightness_0p85",
    "gaussian_4_over_255",
    "translate_down_right_1px",
)
CORRUPTIONS = CONDITIONS[1:]
SHARDS = ((0, 3), (3, 6), (6, 8), (8, 10))
PROTOCOL = {
    "suite": "libero_object",
    "architecture": "chi",
    "vision_encoder": "vit",
    "resolution": 64,
    "action_horizon": 8,
    "execution_horizon": 8,
    "max_steps": 280,
    "settle_steps": 10,
    "canonical_init_states": True,
    "episodes_per_task": 10,
    "task_indices": list(range(10)),
    "conditions": list(CONDITIONS),
    "condition_definitions": {
        "clean": "the deployed rotated and resized RGB input",
        "brightness_0p85": "multiply post-resize RGB intensities by 0.85 and clip to [0,1]",
        "gaussian_4_over_255": (
            "add deterministic independent RGB Gaussian noise with sigma 4/255 and clip to [0,1]"
        ),
        "translate_down_right_1px": (
            "translate the post-resize RGB image one pixel down and right with edge replication"
        ),
    },
    "pairing": "same checkpoint, official task, canonical episode, seed, and post-settle simulator state",
    "matmul_precision": "highest",
    "gpu_name": "NVIDIA RTX A6000",
    "bootstrap": {
        "draws": 20000,
        "seed": 20260821,
        "unit": "paired episodes resampled within each of 30 fixed checkpoint-task cells",
        "interval": [0.025, 0.975],
    },
    "primary_gates": {
        "each_corruption_retention_ci_lower_at_least": 0.75,
        "each_checkpoint_point_retention_at_least": 0.70,
    },
    "favorable_condition_selection": None,
}
FROZEN_CACHE = {
    "basename": "libero_frames_100000_64.pkl",
    "sha256": "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662",
    "frames": 66984,
}
FROZEN_PROVENANCE = {
    "basename": "cache_provenance_libero_object.json",
    "sha256": "1e3ed7eaeef317a221bb6649ea75ef68ff924b57797682e853b4bd90a138f4c4",
    "job_id": "830988",
    "canonical_content_sha256": "01bc724b9bf8c158b983e34b82bbb40dce9dba2cb86dd9be2a5bf8910be341c7",
}
FROZEN_MANIFEST = {
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
    "athena/run_vit_observation_robustness.py",
    "athena/summarize_vit_observation_robustness.py",
    "athena/slurm_vit_observation_robustness.sbatch",
    "athena/slurm_summarize_vit_observation_robustness.sbatch",
    "athena/launch_vit_observation_robustness.sh",
    "athena/VIT_OBSERVATION_ROBUSTNESS_PROTOCOL.md",
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
    parser.add_argument("--provenance-result", type=Path, required=True)
    parser.add_argument("--capability-manifest", type=Path, required=True)
    parser.add_argument("--task-start", type=int, required=True)
    parser.add_argument("--task-end", type=int, required=True)
    parser.add_argument("--mode", choices=("strict_smoke", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def source_hashes() -> dict[str, str]:
    root = repository_root()
    return {relative: file_sha256(root / relative) for relative in SOURCE_FILES}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = FROZEN_CHECKPOINTS.get(args.checkpoint_seed)
    if checkpoint is None:
        raise RuntimeError("Checkpoint seed must be 0, 1, or 2")
    errors = []
    if args.checkpoint.name != checkpoint["basename"]:
        errors.append("checkpoint basename")
    if args.expected_checkpoint_sha256 != checkpoint["sha256"]:
        errors.append("declared checkpoint SHA")
    if (
        not args.checkpoint.is_file()
        or file_sha256(args.checkpoint) != checkpoint["sha256"]
    ):
        errors.append("live checkpoint SHA")
    if args.cache.name != FROZEN_CACHE["basename"]:
        errors.append("cache basename")
    if not args.cache.is_file() or file_sha256(args.cache) != FROZEN_CACHE["sha256"]:
        errors.append("live cache SHA")
    if args.provenance_result.name != FROZEN_PROVENANCE["basename"]:
        errors.append("provenance basename")
    if file_sha256(args.provenance_result) != FROZEN_PROVENANCE["sha256"]:
        errors.append("provenance SHA")
    if args.capability_manifest.name != FROZEN_MANIFEST["basename"]:
        errors.append("manifest basename")
    if file_sha256(args.capability_manifest) != FROZEN_MANIFEST["sha256"]:
        errors.append("manifest SHA")
    if (args.task_start, args.task_end) not in SHARDS:
        errors.append("task shard")
    if (
        args.output.exists()
        or args.output.with_suffix(args.output.suffix + ".tmp").exists()
    ):
        errors.append("output already exists")
    if errors:
        raise RuntimeError("Frozen input validation failed: " + ", ".join(errors))

    provenance = json.loads(args.provenance_result.read_text())
    if (
        provenance.get("schema") != "xvla-cache-provenance-v1"
        or provenance.get("suite") != PROTOCOL["suite"]
        or provenance.get("verified") is not True
        or provenance.get("cache", {}).get("sha256") != FROZEN_CACHE["sha256"]
        or provenance.get("cache", {}).get("frames") != FROZEN_CACHE["frames"]
        or provenance.get("cache", {}).get("canonical_content_sha256")
        != FROZEN_PROVENANCE["canonical_content_sha256"]
        or provenance.get("source", {}).get("content_hashes_match") is not True
        or provenance.get("metadata", {}).get("dataset_to_official_task")
        != FROZEN_DATASET_METADATA["dataset_to_official_task"]
        or any(provenance.get("comparison", {}).get("mismatch_counts", {}).values())
    ):
        raise RuntimeError(
            "Object provenance payload differs from the frozen passing identity"
        )

    manifest = json.loads(args.capability_manifest.read_text())
    expected_checkpoints = [
        {
            "path": f"artifacts/{FROZEN_CHECKPOINTS[seed]['basename']}",
            "seed": seed,
            "sha256": FROZEN_CHECKPOINTS[seed]["sha256"],
        }
        for seed in sorted(FROZEN_CHECKPOINTS)
    ]
    if (
        manifest.get("schema") != "anonymous-vit-capability-artifact-v1"
        or manifest.get("checkpoint_identities") != expected_checkpoints
        or manifest.get("cache_identity", {}).get("sha256") != FROZEN_CACHE["sha256"]
    ):
        raise RuntimeError("Capability manifest differs from the frozen identity")
    return provenance, manifest


def validate_dataset_metadata(metadata: dict[str, Any]) -> None:
    if metadata != FROZEN_DATASET_METADATA:
        raise RuntimeError("Pinned Object dataset task metadata differs")


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
    model.load_state_dict(
        torch.load(checkpoint, map_location="cuda", weights_only=True), strict=True
    )
    model.eval()
    if model.cfg.vision_encoder != "vit" or model.cfg.n_phases != 0:
        raise RuntimeError("Robustness audit requires the frozen no-phase ViT policy")
    return model


def perturb_image(array: np.ndarray, condition: str, noise_seed: int) -> np.ndarray:
    image = np.asarray(array, dtype=np.float32) / 255.0
    if condition == "clean":
        transformed = image
    elif condition == "brightness_0p85":
        transformed = np.clip(image * 0.85, 0.0, 1.0)
    elif condition == "gaussian_4_over_255":
        rng = np.random.default_rng(noise_seed)
        transformed = np.clip(
            image + rng.normal(0.0, 4.0 / 255.0, image.shape), 0.0, 1.0
        )
    elif condition == "translate_down_right_1px":
        padded = np.pad(image, ((1, 0), (1, 0), (0, 0)), mode="edge")
        transformed = padded[: image.shape[0], : image.shape[1]]
    else:
        raise ValueError(f"Unknown condition {condition}")
    return np.ascontiguousarray(transformed, dtype=np.float32)


def deterministic_noise_seed(
    checkpoint_sha256: str,
    task_index: int,
    episode: int,
    decision: int,
    condition: str,
) -> int:
    digest = hashlib.sha256()
    digest.update(PROTOCOL_ID)
    for value in (
        checkpoint_sha256,
        str(task_index),
        str(episode),
        str(decision),
        condition,
    ):
        digest.update(value.encode("utf-8") + b"\0")
    return int.from_bytes(digest.digest()[:8], "little")


@torch.inference_mode()
def predict_chunk(
    model: ChiVLA,
    obs: dict[str, Any],
    instruction: torch.Tensor,
    stats: dict[str, Any],
    condition: str,
    noise_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_state = build_robot_state(obs, stats["state_dim"])
    rotated = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    resized = np.asarray(
        Image.fromarray(rotated).resize(
            (PROTOCOL["resolution"], PROTOCOL["resolution"])
        )
    ).copy()
    transformed = perturb_image(resized, condition, noise_seed)
    image = torch.from_numpy(transformed).permute(2, 0, 1).unsqueeze(0).cuda()
    state = torch.tensor(
        (raw_state - stats["state_mean"]) / stats["state_std"],
        dtype=torch.float32,
        device="cuda",
    ).unsqueeze(0)
    embodiment = torch.zeros(1, dtype=torch.long, device="cuda")
    prediction, _ = model(image, instruction, state, embodiment)
    action_mean = torch.tensor(stats["action_mean"], dtype=torch.float32, device="cuda")
    action_std = torch.tensor(stats["action_std"], dtype=torch.float32, device="cuda")
    action = (prediction[0] * action_std + action_mean).float().cpu().numpy()
    if not np.isfinite(action).all():
        raise RuntimeError("Policy prediction is non-finite")
    return action, transformed


def simulator_state_hash(environment: Any) -> str:
    state = environment.sim.get_state()
    flattened = (
        state.flatten() if hasattr(state, "flatten") else np.asarray(state).reshape(-1)
    )
    return array_sha256(np.asarray(flattened, dtype=np.float64))


def rollout_condition(
    model: ChiVLA,
    suite: Any,
    task_index: int,
    episode: int,
    condition: str,
    instruction: torch.Tensor,
    stats: dict[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = suite.get_task(task_index)
    bddl_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    environment = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=PROTOCOL["resolution"],
        camera_widths=PROTOCOL["resolution"],
    )
    environment.seed(task_index * 100 + episode)
    observation = environment.reset()
    init_states = official_init_states(suite, task_index)
    init_state = init_states[episode % len(init_states)]
    observation = environment.set_init_state(init_state)
    dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    for _ in range(PROTOCOL["settle_steps"]):
        observation, _, _, _ = environment.step(dummy_action)
    initial_state_sha256 = simulator_state_hash(environment)

    image_stream = hashlib.sha256()
    action_stream = hashlib.sha256()
    first_image_sha256 = None
    success = False
    steps = 0
    decisions = 0
    started = time.perf_counter()
    while steps < PROTOCOL["max_steps"] and not success:
        noise_seed = deterministic_noise_seed(
            checkpoint_sha256, task_index, episode, decisions, condition
        )
        chunk, transformed = predict_chunk(
            model, observation, instruction, stats, condition, noise_seed
        )
        image_stream.update(transformed.tobytes())
        current_image_sha256 = hashlib.sha256(transformed.tobytes()).hexdigest()
        if first_image_sha256 is None:
            first_image_sha256 = current_image_sha256
        decisions += 1
        for offset in range(PROTOCOL["execution_horizon"]):
            action = np.asarray(chunk[offset], dtype=np.float64).copy()
            action[-1] = 1.0 if action[-1] > 0 else -1.0
            action_stream.update(action.tobytes())
            observation, reward, done, _ = environment.step(action.tolist())
            steps += 1
            success = bool(reward > 0)
            if done or success or steps >= PROTOCOL["max_steps"]:
                break
    environment.close()
    return {
        "task_index": task_index,
        "episode": episode,
        "condition": condition,
        "success": success,
        "steps": steps,
        "decisions": decisions,
        "canonical_init_state_sha256": array_sha256(np.asarray(init_state)),
        "post_settle_simulator_state_sha256": initial_state_sha256,
        "first_model_image_sha256": first_image_sha256,
        "model_image_stream_sha256": image_stream.hexdigest(),
        "executed_action_stream_sha256": action_stream.hexdigest(),
        "elapsed_s": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    initial_sources = source_hashes()
    provenance, _ = validate_inputs(args)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_float32_matmul_precision(PROTOCOL["matmul_precision"])
    if not torch.cuda.is_available():
        raise RuntimeError("Closed-loop robustness requires CUDA")
    if torch.cuda.get_device_name(0) != PROTOCOL["gpu_name"]:
        raise RuntimeError("Closed-loop robustness requires the frozen A6000 runtime")

    suite = load_suite(PROTOCOL["suite"])
    official_tasks = task_languages(suite)
    cache_tasks, dataset_metadata = load_dataset_task_languages(
        PROTOCOL["suite"], official_tasks
    )
    validate_dataset_metadata(dataset_metadata)
    vocab, encode = build_vocab(cache_tasks)
    stats = load_cache_statistics(args.cache, PROTOCOL["action_horizon"])
    model = make_model(len(vocab), stats, args.checkpoint)
    episodes_per_task = (
        1 if args.mode == "strict_smoke" else PROTOCOL["episodes_per_task"]
    )
    task_end = args.task_start + 1 if args.mode == "strict_smoke" else args.task_end

    task_protocol = {}
    rows = []
    from libero.libero import get_libero_path

    for task_index in range(args.task_start, task_end):
        task = suite.get_task(task_index)
        init_states = official_init_states(suite, task_index)
        bddl_path = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        task_protocol[str(task_index)] = {
            "language": str(task.language),
            "problem_folder": str(task.problem_folder),
            "bddl_file": str(task.bddl_file),
            "bddl_sha256": file_sha256(bddl_path),
            "init_state_count": len(init_states),
            "init_states_sha256": array_sha256(init_states),
            "instruction_ids": encode(task.language),
        }
        instruction = torch.tensor(
            [encode(task.language)], dtype=torch.long, device="cuda"
        )
        for episode in range(episodes_per_task):
            paired_state_hash = None
            for condition in CONDITIONS:
                row = rollout_condition(
                    model,
                    suite,
                    task_index,
                    episode,
                    condition,
                    instruction,
                    stats,
                    args.expected_checkpoint_sha256,
                )
                if paired_state_hash is None:
                    paired_state_hash = row["post_settle_simulator_state_sha256"]
                elif row["post_settle_simulator_state_sha256"] != paired_state_hash:
                    raise RuntimeError(
                        "Paired conditions do not share the post-settle state"
                    )
                rows.append(row)
                print("EPISODE", json.dumps(row, sort_keys=True), flush=True)

    if source_hashes() != initial_sources:
        raise RuntimeError("Source changed during robustness evaluation")
    if file_sha256(args.checkpoint) != args.expected_checkpoint_sha256:
        raise RuntimeError("Checkpoint changed during robustness evaluation")
    if file_sha256(args.cache) != FROZEN_CACHE["sha256"]:
        raise RuntimeError("Cache changed during robustness evaluation")
    if file_sha256(args.provenance_result) != FROZEN_PROVENANCE["sha256"]:
        raise RuntimeError("Provenance changed during robustness evaluation")

    condition_successes = {
        condition: sum(row["success"] for row in rows if row["condition"] == condition)
        for condition in CONDITIONS
    }
    condition_trials = {
        condition: sum(row["condition"] == condition for row in rows)
        for condition in CONDITIONS
    }
    result = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "mode": args.mode,
        "identity": {
            "checkpoint_seed": args.checkpoint_seed,
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_basename": args.checkpoint.name,
            "checkpoint_sha256": args.expected_checkpoint_sha256,
            "cache_path": str(args.cache),
            "cache_sha256": FROZEN_CACHE["sha256"],
            "provenance_path": str(args.provenance_result),
            "provenance_sha256": FROZEN_PROVENANCE["sha256"],
            "provenance_job_id": FROZEN_PROVENANCE["job_id"],
            "provenance_canonical_content_sha256": provenance["cache"][
                "canonical_content_sha256"
            ],
            "capability_manifest_path": str(args.capability_manifest),
            "capability_manifest_sha256": FROZEN_MANIFEST["sha256"],
            "dataset_metadata": dataset_metadata,
            "source_sha256": initial_sources,
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": np.__version__,
            "pillow": importlib.metadata.version("pillow"),
            "mujoco": importlib.metadata.version("mujoco"),
            "robosuite": importlib.metadata.version("robosuite"),
            "libero": importlib.metadata.version("libero"),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "matmul_precision": torch.get_float32_matmul_precision(),
        },
        "task_start": args.task_start,
        "task_end": task_end,
        "episodes_per_task": episodes_per_task,
        "task_protocol": task_protocol,
        "episodes": rows,
        "aggregate": {
            "trials": len(rows),
            "successes_by_condition": condition_successes,
            "trials_by_condition": condition_trials,
            "success_rate_by_condition": {
                condition: condition_successes[condition] / condition_trials[condition]
                for condition in CONDITIONS
            },
        },
        "elapsed_s": time.perf_counter() - started,
        "claim_boundary": (
            "This is a paired closed-loop test of three prespecified mild camera-stream shifts "
            "on familiar LIBERO-Object tasks, prompts, checkpoints, and canonical initial states. "
            "It is not an adversarial-robustness, unseen-task, language-generalization, cross-suite, "
            "physical-robot, or architecture-comparison result."
        ),
    }
    if not all(math.isfinite(float(row["elapsed_s"])) for row in rows):
        raise RuntimeError("An episode timing is non-finite")
    write_json(args.output, result)
    print(
        "RESULT_AGGREGATE", json.dumps(result["aggregate"], sort_keys=True), flush=True
    )


if __name__ == "__main__":
    main()
