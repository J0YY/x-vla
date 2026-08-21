#!/usr/bin/env python3
"""Train one χ-VLA or parameter-matched conventional checkpoint on Athena."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from athena.libero_dataset_metadata import file_sha256, load_dataset_task_languages
from athena.run_xvla_experiment import build_vocab, load_suite, make_config, task_languages
from xvla.models.vla import ChiVLA
from xvla.train.train_lm import TrainConfig, _lr_at


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("chi", "conventional", "chi_rms", "conventional_rational"),
        required=True,
    )
    parser.add_argument(
        "--suite",
        choices=("libero_object", "libero_spatial", "libero_goal", "libero_10"),
        default="libero_object",
    )
    parser.add_argument(
        "--vision-encoder",
        choices=("vit", "conv"),
        default="vit",
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--init-suite",
        choices=("libero_object", "libero_spatial", "libero_goal", "libero_10"),
        default="libero_object",
        help="Suite whose vocabulary was used by --init-checkpoint.",
    )
    return parser.parse_args()


def load_training_data(args, tasks, encode):
    with args.cache.open("rb") as handle:
        frames = pickle.load(handle)
    episodes = defaultdict(list)
    for frame in frames:
        episodes[int(frame[0])].append(frame)
    samples = []
    for episode_frames in episodes.values():
        episode_frames.sort(key=lambda item: int(item[1]))
        for index in range(len(episode_frames) - args.horizon):
            frame = episode_frames[index]
            samples.append(
                (
                    np.asarray(frame[2], dtype=np.uint8),
                    int(frame[5]),
                    np.asarray(frame[3], dtype=np.float32),
                    np.stack(
                        [episode_frames[index + offset][4] for offset in range(args.horizon)]
                    ).astype(np.float32),
                )
            )
    actions_array = np.stack([sample[3] for sample in samples])
    states_array = np.stack([sample[2] for sample in samples])
    action_mean = actions_array.mean((0, 1))
    action_std = actions_array.std((0, 1)) + 1e-6
    state_mean = states_array.mean(0)
    state_std = states_array.std(0) + 1e-6
    images = (
        torch.tensor(np.stack([sample[0] for sample in samples]))
        .permute(0, 3, 1, 2)
        .float()
        .div(255)
        .cuda()
    )
    instructions = torch.tensor(
        [encode(tasks[sample[1]]) for sample in samples], dtype=torch.long, device="cuda"
    )
    states = torch.tensor(
        (states_array - state_mean) / state_std, dtype=torch.float32, device="cuda"
    )
    actions = torch.tensor(
        (actions_array - action_mean) / action_std, dtype=torch.float32, device="cuda"
    )
    return {
        "images": images,
        "instructions": instructions,
        "states": states,
        "actions": actions,
        "sample_count": len(samples),
        "state_dim": int(states_array.shape[-1]),
        "action_dim": int(actions_array.shape[-1]),
        "normalization": {
            "action_mean": action_mean.tolist(),
            "action_std": action_std.tolist(),
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
        },
    }


def initialize_from_checkpoint(
    model: ChiVLA,
    checkpoint: Path,
    source_suite: str,
    target_vocab: dict[str, int],
) -> dict[str, object]:
    """Load a same-architecture checkpoint and remap shared vocabulary rows."""
    state_dict = torch.load(checkpoint, map_location="cuda", weights_only=True)
    source_embedding = state_dict.pop("tok_emb.weight")
    incompatibility = model.load_state_dict(state_dict, strict=False)
    if incompatibility.missing_keys != ["tok_emb.weight"]:
        raise RuntimeError(
            f"Unexpected missing warm-start keys: {incompatibility.missing_keys}"
        )
    if incompatibility.unexpected_keys:
        raise RuntimeError(
            f"Unexpected warm-start keys: {incompatibility.unexpected_keys}"
        )
    source_official = task_languages(load_suite(source_suite))
    source_tasks, source_task_metadata = load_dataset_task_languages(
        source_suite, source_official
    )
    source_vocab, _ = build_vocab(source_tasks)
    if source_embedding.shape[0] != len(source_vocab):
        raise RuntimeError(
            f"Source embedding has {source_embedding.shape[0]} rows, "
            f"but pinned source vocabulary has {len(source_vocab)} tokens"
        )
    shared_tokens = sorted(set(source_vocab) & set(target_vocab))
    with torch.no_grad():
        for token in shared_tokens:
            model.tok_emb.weight[target_vocab[token]].copy_(
                source_embedding[source_vocab[token]]
            )
    return {
        "source_suite": source_suite,
        "source_checkpoint_sha256": file_sha256(checkpoint),
        "source_vocab_size": len(source_vocab),
        "target_vocab_size": len(target_vocab),
        "shared_vocab_tokens": len(shared_tokens),
        "shared_vocab_fraction": len(shared_tokens) / max(len(target_vocab), 1),
        "shared_tokens": shared_tokens,
        "source_task_metadata": source_task_metadata,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    suite = load_suite(args.suite)
    official_tasks = task_languages(suite)
    tasks, task_metadata = load_dataset_task_languages(args.suite, official_tasks)
    vocab, encode = build_vocab(tasks)
    data = load_training_data(args, tasks, encode)
    cache_hash = file_sha256(args.cache)
    vocab_serialized = json.dumps(vocab, sort_keys=True, separators=(",", ":"))
    vocab_hash = hashlib.sha256(vocab_serialized.encode()).hexdigest()
    config = make_config(
        args.architecture,
        len(vocab),
        data["state_dim"],
        data["action_dim"],
        args.res,
        args.horizon,
        vision_encoder=args.vision_encoder,
    )
    model = ChiVLA(config).cuda()
    initialization = None
    if args.init_checkpoint is not None:
        initialization = initialize_from_checkpoint(
            model,
            args.init_checkpoint,
            args.init_suite,
            vocab,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05
    )
    schedule = TrainConfig(
        train_bin="", val_bin="", lr=args.lr, max_steps=args.steps, warmup_frac=0.05
    )
    ema = {name: parameter.detach().clone().float() for name, parameter in model.named_parameters()}
    embodiments = torch.zeros(args.batch_size, dtype=torch.long, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    model.train()
    last_loss = None
    for step_index in range(args.steps):
        for group in optimizer.param_groups:
            group["lr"] = _lr_at(step_index, schedule)
        indices = torch.randint(data["sample_count"], (args.batch_size,), device="cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(
                data["images"][indices],
                data["instructions"][indices],
                data["states"][indices],
                embodiments,
                target_actions=data["actions"][indices],
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
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(ema[name].to(parameter.dtype))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = args.checkpoint_output.with_suffix(".tmp")
    torch.save(model.state_dict(), temporary_checkpoint)
    temporary_checkpoint.replace(args.checkpoint_output)
    result = {
        "architecture": args.architecture,
        "vision_encoder": args.vision_encoder,
        "suite": args.suite,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "res": args.res,
        "horizon": args.horizon,
        "samples": data["sample_count"],
        "parameters": model.num_params(),
        "elapsed_s": elapsed,
        "steps_per_s": args.steps / elapsed,
        "examples_per_s": args.steps * args.batch_size / elapsed,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "last_loss": last_loss,
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint": str(args.checkpoint_output),
        "cache": str(args.cache),
        "cache_sha256": cache_hash,
        "vocab": vocab,
        "vocab_sha256": vocab_hash,
        "normalization": data["normalization"],
        "ema_decay": args.ema_decay,
        "init_checkpoint": (
            str(args.init_checkpoint) if args.init_checkpoint is not None else None
        ),
        "initialization": initialization,
        "task_metadata": task_metadata,
    }
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    args.result_output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
