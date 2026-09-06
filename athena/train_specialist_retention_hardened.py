#!/usr/bin/env python3
"""Train one frozen seed-0 χ-ViT specialist for the retention certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from athena.libero_dataset_metadata import load_dataset_task_languages
from athena.run_xvla_experiment import build_vocab, load_suite, make_config, task_languages
from athena.specialist_retention_protocol import (
    ARCHITECTURE,
    EMA_DECAY,
    FULL_BATCH_SIZE,
    FULL_STEPS,
    HORIZON,
    LEARNING_RATE,
    MODEL_SOURCE_HASHES,
    RECIPE_VERSION,
    RESOLUTION,
    SEED,
    SMOKE_BATCH_SIZE,
    SMOKE_STEPS,
    SUITES,
    SUITE_SPECS,
    TRAIN_LM_SHA256,
    VISION_ENCODER,
    bundle_sha256,
    checkpoint_path,
    file_sha256,
    metadata_path,
    source_snapshot,
    training_result_path,
    validate_cache_and_provenance,
    validate_source_snapshot,
    validate_task_metadata,
    validate_work_cache_environment,
)
from xvla.models.vla import ChiVLA
from xvla.train.train_lm import TrainConfig, _lr_at


PROTOCOL_SOURCE_SHA256 = "0bee62f9296496c4ee0943a0b74860c927c69d436bac468c7dd947500b7700eb"
TRAINING_IMPORTED_SOURCES = {
    **MODEL_SOURCE_HASHES,
    "athena/specialist_retention_protocol.py": PROTOCOL_SOURCE_SHA256,
    "xvla/train/train_lm.py": TRAIN_LM_SHA256,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing stale temporary output {temporary}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def validate_frame_cache(
    suite: str,
    frames: list[Any],
    tasks: dict[int, str],
    encode,
) -> dict[str, Any]:
    spec = SUITE_SPECS[suite]
    if len(frames) != spec["frame_count"]:
        raise RuntimeError(
            f"{suite} cache has {len(frames)} frames, expected {spec['frame_count']}"
        )
    episodes: dict[int, list[Any]] = defaultdict(list)
    for frame in frames:
        if len(frame) < 6:
            raise RuntimeError(f"Malformed {suite} cache frame")
        image = np.asarray(frame[2])
        state = np.asarray(frame[3])
        action = np.asarray(frame[4])
        dataset_task = int(frame[5])
        if image.shape != (RESOLUTION, RESOLUTION, 3):
            raise RuntimeError(f"Invalid {suite} image shape {image.shape}")
        if state.shape != (8,) or action.shape != (7,):
            raise RuntimeError(
                f"Invalid {suite} state/action shapes {state.shape}/{action.shape}"
            )
        if dataset_task not in tasks:
            raise RuntimeError(f"Invalid {suite} dataset task {dataset_task}")
        episodes[int(frame[0])].append(frame)

    images: list[np.ndarray] = []
    instructions: list[list[int]] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode_id, episode_frames in episodes.items():
        episode_frames.sort(key=lambda item: int(item[1]))
        episode_tasks = {int(frame[5]) for frame in episode_frames}
        if len(episode_tasks) != 1:
            raise RuntimeError(f"{suite} episode {episode_id} crosses dataset tasks")
        for index in range(len(episode_frames) - HORIZON):
            frame = episode_frames[index]
            expected_indices = [int(frame[1]) + offset for offset in range(HORIZON)]
            observed_indices = [
                int(episode_frames[index + offset][1]) for offset in range(HORIZON)
            ]
            if observed_indices != expected_indices:
                raise RuntimeError(
                    f"{suite} episode {episode_id} has a noncontiguous horizon window"
                )
            dataset_task = int(frame[5])
            images.append(np.asarray(frame[2], dtype=np.uint8))
            instructions.append(encode(tasks[dataset_task]))
            states.append(np.asarray(frame[3], dtype=np.float32))
            actions.append(
                np.stack(
                    [episode_frames[index + offset][4] for offset in range(HORIZON)]
                ).astype(np.float32)
            )
    if len(images) != spec["sample_count"]:
        raise RuntimeError(
            f"{suite} cache produced {len(images)} samples, expected {spec['sample_count']}"
        )
    state_array = np.stack(states)
    action_array = np.stack(actions)
    state_mean = state_array.astype(np.float64).mean(axis=0)
    state_std = state_array.astype(np.float64).std(axis=0) + 1e-6
    action_mean = action_array.astype(np.float64).mean(axis=(0, 1))
    action_std = action_array.astype(np.float64).std(axis=(0, 1)) + 1e-6
    normalization = {
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
    }
    return {
        "images": np.stack(images),
        "instructions": np.asarray(instructions, dtype=np.int64),
        "states": ((state_array - state_mean) / state_std).astype(np.float32),
        "actions": ((action_array - action_mean) / action_std).astype(np.float32),
        "sample_count": len(images),
        "state_dim": 8,
        "action_dim": 7,
        "normalization": normalization,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    smoke = args.mode == "smoke"
    steps = SMOKE_STEPS if smoke else FULL_STEPS
    batch_size = SMOKE_BATCH_SIZE if smoke else FULL_BATCH_SIZE
    checkpoint_output = checkpoint_path(args.suite, smoke)
    metadata_output = metadata_path(args.suite, smoke)
    result_output = training_result_path(args.suite, smoke)
    for output in (checkpoint_output, metadata_output, result_output):
        if output.exists() or output.with_suffix(output.suffix + ".tmp").exists():
            raise FileExistsError(f"Refusing to overwrite hardened output {output}")

    trainer_start_sha256 = file_sha256(Path(__file__))
    expected_trainer_sha256 = os.environ.get("XVLA_FROZEN_SPECIALIST_TRAINER_SHA256")
    if expected_trainer_sha256 is None:
        raise RuntimeError("XVLA_FROZEN_SPECIALIST_TRAINER_SHA256 is required")
    if trainer_start_sha256 != expected_trainer_sha256:
        raise RuntimeError("Specialist trainer does not match its frozen wrapper SHA-256")
    imported_start = source_snapshot(TRAINING_IMPORTED_SOURCES)
    validate_source_snapshot(imported_start, TRAINING_IMPORTED_SOURCES)
    cache_environment_start = validate_work_cache_environment()
    cache_identity_start = validate_cache_and_provenance(args.suite)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.set_float32_matmul_precision("high")
    suite = load_suite(args.suite)
    official_tasks = task_languages(suite)
    dataset_tasks, task_metadata_start = load_dataset_task_languages(
        args.suite, official_tasks
    )
    validate_task_metadata(args.suite, task_metadata_start)
    vocab, encode = build_vocab(dataset_tasks)
    spec = SUITE_SPECS[args.suite]
    cache_path = Path(spec["cache_path"])
    with cache_path.open("rb") as handle:
        frames = pickle.load(handle)
    data = validate_frame_cache(args.suite, frames, dataset_tasks, encode)
    del frames

    config = make_config(
        ARCHITECTURE,
        len(vocab),
        data["state_dim"],
        data["action_dim"],
        RESOLUTION,
        HORIZON,
        vision_encoder=VISION_ENCODER,
    )
    model = ChiVLA(config).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.05
    )
    schedule = TrainConfig(
        train_bin="",
        val_bin="",
        lr=LEARNING_RATE,
        max_steps=steps,
        warmup_frac=0.05,
    )
    ema = {
        name: parameter.detach().clone().float()
        for name, parameter in model.named_parameters()
    }
    images = torch.from_numpy(data["images"]).permute(0, 3, 1, 2).cuda()
    instructions = torch.from_numpy(data["instructions"]).cuda()
    states = torch.from_numpy(data["states"]).cuda()
    actions = torch.from_numpy(data["actions"]).cuda()
    embodiments = torch.zeros(batch_size, dtype=torch.long, device="cuda")
    del data["images"], data["instructions"], data["states"], data["actions"]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    model.train()
    last_loss = None
    for step_index in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = _lr_at(step_index, schedule)
        indices = torch.randint(data["sample_count"], (batch_size,), device="cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(
                images[indices].float().div(255),
                instructions[indices],
                states[indices],
                embodiments,
                target_actions=actions[indices],
                progress=step_index / max(steps - 1, 1),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                ema[name].mul_(EMA_DECAY).add_(
                    parameter.detach().float(), alpha=1.0 - EMA_DECAY
                )
        last_loss = float(loss.detach())
        if step_index % 1000 == 0 or step_index + 1 == steps:
            print(
                f"STEP {step_index + 1}/{steps} loss={last_loss:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f}",
                flush=True,
            )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(ema[name].to(parameter.dtype))
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - started
    if last_loss is None or not np.isfinite(last_loss):
        raise RuntimeError(f"Final training loss is not finite: {last_loss}")
    nonfinite = [
        name for name, parameter in model.named_parameters()
        if not torch.isfinite(parameter).all()
    ]
    if nonfinite:
        raise RuntimeError(f"Non-finite trained parameters: {nonfinite}")

    task_metadata_end_tasks, task_metadata_end = load_dataset_task_languages(
        args.suite, official_tasks
    )
    if task_metadata_end_tasks != dataset_tasks:
        raise RuntimeError("Dataset task languages changed during specialist training")
    validate_task_metadata(args.suite, task_metadata_end)
    trainer_end_sha256 = file_sha256(Path(__file__))
    imported_end = source_snapshot(TRAINING_IMPORTED_SOURCES)
    validate_source_snapshot(imported_end, TRAINING_IMPORTED_SOURCES)
    cache_environment_end = validate_work_cache_environment()
    cache_identity_end = validate_cache_and_provenance(args.suite)
    if trainer_end_sha256 != trainer_start_sha256:
        raise RuntimeError("Specialist trainer changed during training")
    if imported_end != imported_start:
        raise RuntimeError("Imported specialist sources changed during training")
    if cache_environment_end != cache_environment_start:
        raise RuntimeError("Cache environment changed during training")
    if cache_identity_end != cache_identity_start:
        raise RuntimeError("Cache or provenance identity changed during training")
    if task_metadata_end != task_metadata_start:
        raise RuntimeError("Task mapping metadata changed during training")

    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = checkpoint_output.with_suffix(checkpoint_output.suffix + ".tmp")
    torch.save(model.state_dict(), temporary_checkpoint)
    temporary_checkpoint.replace(checkpoint_output)
    checkpoint_sha256 = file_sha256(checkpoint_output)
    vocab_serialized = json.dumps(vocab, sort_keys=True, separators=(",", ":"))
    vocab_sha256 = hashlib.sha256(vocab_serialized.encode()).hexdigest()
    source_identity = {
        "expected_trainer_sha256": expected_trainer_sha256,
        "trainer_start_sha256": trainer_start_sha256,
        "trainer_end_sha256": trainer_end_sha256,
        "imported_sources_start": imported_start,
        "imported_sources_end": imported_end,
        "imported_bundle_start_sha256": bundle_sha256(imported_start),
        "imported_bundle_end_sha256": bundle_sha256(imported_end),
        "cache_identity_start": cache_identity_start,
        "cache_identity_end": cache_identity_end,
        "task_metadata_start": task_metadata_start,
        "task_metadata_end": task_metadata_end,
        "cache_environment_start": cache_environment_start,
        "cache_environment_end": cache_environment_end,
    }
    training_environment = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "autocast_dtype": "bfloat16",
    }
    metadata = {
        "format": "xvla_specialist_retention_checkpoint_v1",
        "recipe_version": RECIPE_VERSION,
        "mode": args.mode,
        "architecture": ARCHITECTURE,
        "vision_encoder": VISION_ENCODER,
        "suite": args.suite,
        "seed": SEED,
        "steps": steps,
        "batch_size": batch_size,
        "lr": LEARNING_RATE,
        "ema_decay": EMA_DECAY,
        "res": RESOLUTION,
        "horizon": HORIZON,
        "state_dim": data["state_dim"],
        "action_dim": data["action_dim"],
        "sample_count": data["sample_count"],
        "checkpoint": checkpoint_output.as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "cache": spec["cache_path"],
        "cache_sha256": spec["cache_sha256"],
        "provenance": cache_identity_start,
        "dataset_tasks": {str(key): value for key, value in dataset_tasks.items()},
        "official_tasks": {str(key): value for key, value in official_tasks.items()},
        "task_metadata": task_metadata_start,
        "vocab": vocab,
        "vocab_sha256": vocab_sha256,
        "normalization": data["normalization"],
        "parameters": model.num_params(),
        "trainer_sha256": trainer_start_sha256,
        "source_identity": source_identity,
        "training_environment": training_environment,
    }
    atomic_json(metadata_output, metadata)
    result = {
        "format": "xvla_specialist_retention_training_result_v1",
        "suite": args.suite,
        "mode": args.mode,
        "architecture": ARCHITECTURE,
        "vision_encoder": VISION_ENCODER,
        "seed": SEED,
        "steps": steps,
        "batch_size": batch_size,
        "lr": LEARNING_RATE,
        "ema_decay": EMA_DECAY,
        "elapsed_s": elapsed_s,
        "steps_per_s": steps / elapsed_s,
        "examples_per_s": steps * batch_size / elapsed_s,
        "last_loss": last_loss,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "checkpoint": checkpoint_output.as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "metadata": metadata_output.as_posix(),
        "cache": spec["cache_path"],
        "cache_sha256": spec["cache_sha256"],
        "task_metadata": task_metadata_start,
        "source_identity": source_identity,
        "training_environment": training_environment,
    }
    atomic_json(result_output, result)
    print("RESULT", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
