#!/usr/bin/env python3
"""Train one χ-VLA or parameter-matched conventional checkpoint on Athena."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

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
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    suite = load_suite()
    tasks = task_languages(suite)
    vocab, encode = build_vocab(tasks)
    data = load_training_data(args, tasks, encode)
    config = make_config(
        args.architecture,
        len(vocab),
        data["state_dim"],
        data["action_dim"],
        args.res,
        args.horizon,
    )
    model = ChiVLA(config).cuda()
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
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
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
        "ema_decay": args.ema_decay,
    }
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    args.result_output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
