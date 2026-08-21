#!/usr/bin/env python3
"""Profile χ and conventional training steps on one matched batch and GPU."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from athena.run_xvla_experiment import (
    build_vocab,
    load_cache_statistics,
    load_suite,
    make_config,
    task_languages,
)
from xvla.models.vla import ChiVLA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chi-checkpoint", type=Path, required=True)
    parser.add_argument("--conventional-checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    return parser.parse_args()


def load_fixed_batch(args, tasks, encode, stats):
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
                    frame[2],
                    int(frame[5]),
                    frame[3],
                    np.stack(
                        [episode_frames[index + offset][4] for offset in range(args.horizon)]
                    ),
                )
            )
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(samples), size=args.batch_size, replace=False)
    selected = [samples[int(index)] for index in indices]
    images = (
        torch.tensor(np.stack([sample[0] for sample in selected]))
        .permute(0, 3, 1, 2)
        .float()
        .div(255)
        .cuda()
    )
    instructions = torch.tensor(
        [encode(tasks[sample[1]]) for sample in selected], dtype=torch.long, device="cuda"
    )
    states = torch.tensor(
        (np.stack([sample[2] for sample in selected]) - stats["state_mean"])
        / stats["state_std"],
        dtype=torch.float32,
        device="cuda",
    )
    actions = torch.tensor(
        (np.stack([sample[3] for sample in selected]) - stats["action_mean"])
        / stats["action_std"],
        dtype=torch.float32,
        device="cuda",
    )
    embodiments = torch.zeros(args.batch_size, dtype=torch.long, device="cuda")
    return images, instructions, states, embodiments, actions


def profile_architecture(args, architecture, checkpoint, vocab_size, stats, batch):
    torch.manual_seed(args.seed)
    config = make_config(
        architecture,
        vocab_size,
        stats["state_dim"],
        stats["action_dim"],
        args.res,
        args.horizon,
    )
    model = ChiVLA(config).cuda()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True), strict=True)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=8e-4, betas=(0.9, 0.95), weight_decay=0.05
    )
    images, instructions, states, embodiments, actions = batch

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(
                images,
                instructions,
                states,
                embodiments,
                target_actions=actions,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return float(loss.detach())

    for _ in range(args.warmup_steps):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = []
    for _ in range(args.profile_steps):
        losses.append(step())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "architecture": architecture,
        "parameters": model.num_params(),
        "elapsed_s": elapsed,
        "steps": args.profile_steps,
        "steps_per_s": args.profile_steps / elapsed,
        "examples_per_s": args.profile_steps * args.batch_size / elapsed,
        "ms_per_step": 1000.0 * elapsed / args.profile_steps,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "loss_first": losses[0],
        "loss_last": losses[-1],
    }
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    suite = load_suite()
    tasks = task_languages(suite)
    vocab, encode = build_vocab(tasks)
    stats = load_cache_statistics(args.cache, args.horizon)
    batch = load_fixed_batch(args, tasks, encode, stats)
    conventional = profile_architecture(
        args,
        "conventional",
        args.conventional_checkpoint,
        len(vocab),
        stats,
        batch,
    )
    chi = profile_architecture(
        args, "chi", args.chi_checkpoint, len(vocab), stats, batch
    )
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "profile_steps": args.profile_steps,
        "fixed_batch_seed": args.seed,
        "conventional": conventional,
        "chi": chi,
        "chi_to_conventional_ms_per_step_ratio": (
            chi["ms_per_step"] / conventional["ms_per_step"]
        ),
        "chi_to_conventional_peak_allocated_ratio": (
            chi["peak_allocated_gb"] / conventional["peak_allocated_gb"]
        ),
        "scope": (
            "Steady-state forward, backward, gradient clipping, and AdamW update cost on the "
            "same fixed batch and physical GPU. This is not a time-to-convergence comparison."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
