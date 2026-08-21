#!/usr/bin/env python3
"""Train one fixed-recipe four-suite LIBERO generalist on Athena."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from athena.run_xvla_experiment import build_encoder, make_config
from xvla.models.vla import ChiVLA
from xvla.train.train_lm import TrainConfig, _lr_at


EXPECTED_SUITES = (
    "libero_object",
    "libero_spatial",
    "libero_goal",
    "libero_10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture", choices=("chi", "conventional"), required=True
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    parser.add_argument("--vision-encoder", choices=("vit", "conv"), default="vit")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=160000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--recovery-output", type=Path)
    parser.add_argument("--recovery-interval", type=int, default=10000)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "xvla_multisuite_v1":
        raise ValueError(f"Unsupported manifest format in {path}")
    if tuple(manifest["training_suites"]) != EXPECTED_SUITES:
        raise ValueError(
            f"Expected suite order {EXPECTED_SUITES}, got {manifest['training_suites']}"
        )
    return manifest


def load_suite_samples(
    suite_name: str,
    manifest: dict[str, Any],
    encode,
) -> dict[str, Any]:
    source = manifest["source_caches"][suite_name]
    cache_path = Path(source["path"])
    print(f"Loading {suite_name} samples from {cache_path}", flush=True)
    actual_hash = file_sha256(cache_path)
    if actual_hash != source["sha256"]:
        raise RuntimeError(
            f"Cache hash mismatch for {suite_name}: {actual_hash} != {source['sha256']}"
        )
    with cache_path.open("rb") as handle:
        frames = pickle.load(handle)
    episodes: dict[int, list[Any]] = defaultdict(list)
    for frame in frames:
        episodes[int(frame[0])].append(frame)

    horizon = int(manifest["horizon"])
    task_offset = int(manifest["suite_task_offsets"][suite_name])
    languages = manifest["task_languages"]
    images: list[np.ndarray] = []
    instructions: list[list[int]] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode_frames in episodes.values():
        episode_frames.sort(key=lambda item: int(item[1]))
        for index in range(len(episode_frames) - horizon):
            frame = episode_frames[index]
            global_task = task_offset + int(frame[5])
            images.append(np.asarray(frame[2], dtype=np.uint8))
            instructions.append(encode(languages[str(global_task)]))
            states.append(np.asarray(frame[3], dtype=np.float32))
            actions.append(
                np.stack(
                    [episode_frames[index + offset][4] for offset in range(horizon)]
                ).astype(np.float32)
            )
    if not images:
        raise RuntimeError(f"No horizon-{horizon} samples in {cache_path}")
    expected_count = int(source["sample_count"])
    if len(images) != expected_count:
        raise RuntimeError(
            f"Manifest expected {expected_count} samples for {suite_name}, got {len(images)}"
        )

    normalization = manifest["normalization"]
    state_mean = np.asarray(normalization["state_mean"], dtype=np.float32)
    state_std = np.asarray(normalization["state_std"], dtype=np.float32)
    action_mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    action_std = np.asarray(normalization["action_std"], dtype=np.float32)
    state_array = np.stack(states)
    action_array = np.stack(actions)
    data = {
        "images": np.stack(images),
        "instructions": np.asarray(instructions, dtype=np.int64),
        "states": ((state_array - state_mean) / state_std).astype(np.float32),
        "actions": ((action_array - action_mean) / action_std).astype(np.float32),
        "sample_count": len(images),
    }
    del frames, episodes, images, instructions, states, actions
    return data


def sample_balanced_batch(
    suite_data: dict[str, dict[str, Any]],
    per_suite: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fields: dict[str, list[np.ndarray]] = {
        "images": [],
        "instructions": [],
        "states": [],
        "actions": [],
    }
    for suite_name in EXPECTED_SUITES:
        data = suite_data[suite_name]
        indices = rng.integers(0, data["sample_count"], size=per_suite)
        for field in fields:
            fields[field].append(data[field][indices])
    images = (
        torch.from_numpy(np.concatenate(fields["images"]))
        .permute(0, 3, 1, 2)
        .to(device="cuda", dtype=torch.float32)
        .div_(255)
    )
    instructions = torch.from_numpy(np.concatenate(fields["instructions"])).cuda()
    states = torch.from_numpy(np.concatenate(fields["states"])).cuda()
    actions = torch.from_numpy(np.concatenate(fields["actions"])).cuda()
    return images, instructions, states, actions


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size % len(EXPECTED_SUITES) != 0:
        raise ValueError("batch-size must be divisible by four for balanced suite sampling")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.recovery_interval <= 0:
        raise ValueError("recovery-interval must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)

    manifest_hash = file_sha256(args.manifest)
    manifest = load_manifest(args.manifest)
    vocab = {str(token): int(index) for token, index in manifest["vocab"].items()}
    encode = build_encoder(vocab)
    suite_data = {
        suite_name: load_suite_samples(suite_name, manifest, encode)
        for suite_name in EXPECTED_SUITES
    }
    sample_counts = {
        suite_name: int(data["sample_count"])
        for suite_name, data in suite_data.items()
    }
    print(f"Balanced suite samples: {sample_counts}", flush=True)

    config = make_config(
        args.architecture,
        len(vocab),
        int(manifest["state_dim"]),
        int(manifest["action_dim"]),
        int(manifest["res"]),
        int(manifest["horizon"]),
        vision_encoder=args.vision_encoder,
    )
    model = ChiVLA(config).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05
    )
    schedule = TrainConfig(
        train_bin="", val_bin="", lr=args.lr, max_steps=args.steps, warmup_frac=0.05
    )
    ema = {
        name: parameter.detach().clone().float()
        for name, parameter in model.named_parameters()
    }
    embodiments = torch.zeros(args.batch_size, dtype=torch.long, device="cuda")
    per_suite = args.batch_size // len(EXPECTED_SUITES)

    start_step = 0
    recovered_elapsed = 0.0
    if args.recovery_output is not None and args.recovery_output.exists():
        recovery = torch.load(
            args.recovery_output, map_location="cpu", weights_only=False
        )
        recovery_identity = (
            recovery.get("architecture"),
            recovery.get("vision_encoder"),
            recovery.get("seed"),
            recovery.get("steps"),
            recovery.get("manifest_sha256"),
        )
        expected_identity = (
            args.architecture,
            args.vision_encoder,
            args.seed,
            args.steps,
            manifest_hash,
        )
        if recovery_identity != expected_identity:
            raise RuntimeError(
                f"Recovery identity mismatch: {recovery_identity} != {expected_identity}"
            )
        model.load_state_dict(recovery["model"], strict=True)
        optimizer.load_state_dict(recovery["optimizer"])
        ema = {name: value.cuda() for name, value in recovery["ema"].items()}
        rng.bit_generator.state = recovery["numpy_rng_state"]
        torch.set_rng_state(recovery["torch_rng_state"])
        torch.cuda.set_rng_state_all(recovery["cuda_rng_state"])
        start_step = int(recovery["completed_steps"])
        recovered_elapsed = float(recovery.get("elapsed_s", 0.0))
        if not 0 <= start_step < args.steps:
            raise RuntimeError(f"Invalid recovered step {start_step}")
        print(f"Resuming from completed step {start_step}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    model.train()
    last_loss = None
    for step_index in range(start_step, args.steps):
        for group in optimizer.param_groups:
            group["lr"] = _lr_at(step_index, schedule)
        images, instructions, states, actions = sample_balanced_batch(
            suite_data, per_suite, rng
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(
                images,
                instructions,
                states,
                embodiments,
                target_actions=actions,
                progress=step_index / max(args.steps - 1, 1),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                ema[name].mul_(args.ema_decay).add_(
                    parameter.detach().float(), alpha=1.0 - args.ema_decay
                )
        last_loss = float(loss.detach())
        if step_index % 1000 == 0:
            print(
                f"STEP {step_index}/{args.steps} loss={last_loss:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f}",
                flush=True,
            )
        completed_steps = step_index + 1
        if (
            args.recovery_output is not None
            and completed_steps < args.steps
            and completed_steps % args.recovery_interval == 0
        ):
            torch.cuda.synchronize()
            atomic_torch_save(
                args.recovery_output,
                {
                    "architecture": args.architecture,
                    "vision_encoder": args.vision_encoder,
                    "seed": args.seed,
                    "steps": args.steps,
                    "manifest_sha256": manifest_hash,
                    "completed_steps": completed_steps,
                    "elapsed_s": recovered_elapsed + time.perf_counter() - started,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema": ema,
                    "numpy_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all(),
                },
            )
            print(f"RECOVERY {completed_steps} -> {args.recovery_output}", flush=True)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(ema[name].to(parameter.dtype))
    torch.cuda.synchronize()
    elapsed = recovered_elapsed + time.perf_counter() - started

    args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = args.checkpoint_output.with_suffix(".tmp")
    torch.save(model.state_dict(), temporary_checkpoint)
    temporary_checkpoint.replace(args.checkpoint_output)
    checkpoint_hash = file_sha256(args.checkpoint_output)

    metadata = dict(manifest)
    metadata.update(
        {
            "format": "xvla_multisuite_checkpoint_v1",
            "source_manifest": str(args.manifest),
            "architecture": args.architecture,
            "vision_encoder": args.vision_encoder,
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "suite_batch_size": per_suite,
            "lr": args.lr,
            "ema_decay": args.ema_decay,
            "checkpoint": str(args.checkpoint_output),
            "checkpoint_sha256": checkpoint_hash,
            "manifest_sha256": manifest_hash,
        }
    )
    atomic_json(args.metadata_output, metadata)

    result = {
        "architecture": args.architecture,
        "vision_encoder": args.vision_encoder,
        "training_suites": list(EXPECTED_SUITES),
        "sampling": "equal examples per suite per optimizer update",
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "suite_batch_size": per_suite,
        "sample_counts": sample_counts,
        "parameters": model.num_params(),
        "elapsed_s": elapsed,
        "steps_per_s": args.steps / elapsed,
        "examples_per_s": args.steps * args.batch_size / elapsed,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "last_loss": last_loss,
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint": str(args.checkpoint_output),
        "checkpoint_sha256": checkpoint_hash,
        "metadata": str(args.metadata_output),
        "manifest": str(args.manifest),
        "ema_decay": args.ema_decay,
    }
    atomic_json(args.result_output, result)
    print("RESULT", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
