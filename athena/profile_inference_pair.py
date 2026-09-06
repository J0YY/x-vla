#!/usr/bin/env python3
"""Benchmark eager and compiled χ/conventional inference on one physical GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from athena.run_xvla_experiment import (
    build_vocab,
    load_cache_statistics,
    load_suite,
    make_config,
    task_languages,
    tensorize_sample,
)
from xvla.models.vla import ChiVLA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chi-checkpoint", type=Path, required=True)
    parser.add_argument("--conventional-checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--profile-iters", type=int, default=200)
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    return parser.parse_args()


@torch.inference_mode()
def benchmark(callable_model, sample, warmup: int, iterations: int):
    for _ in range(warmup):
        callable_model(*sample)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        callable_model(*sample)
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - started) / iterations


def profile_model(args, architecture, checkpoint, vocab_size, stats, sample):
    config = make_config(
        architecture,
        vocab_size,
        stats["state_dim"],
        stats["action_dim"],
        args.res,
        args.horizon,
    )
    model = ChiVLA(config).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True), strict=True)
    eager_ms = benchmark(model, sample, args.warmup_iters, args.profile_iters)
    compiled_ms = None
    compile_error = None
    compile_started = time.perf_counter()
    try:
        compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        compiled_ms = benchmark(compiled, sample, 5, args.profile_iters)
    except Exception as exc:
        compile_error = f"{type(exc).__name__}: {exc}"
    compile_elapsed = time.perf_counter() - compile_started
    return {
        "architecture": architecture,
        "parameters": model.num_params(),
        "eager_latency_ms_batch1": eager_ms,
        "compiled_latency_ms_batch1": compiled_ms,
        "compiled_to_eager_ratio": compiled_ms / eager_ms if compiled_ms is not None else None,
        "compile_and_warmup_elapsed_s": compile_elapsed,
        "compile_error": compile_error,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    suite = load_suite()
    tasks = task_languages(suite)
    vocab, encode = build_vocab(tasks)
    stats = load_cache_statistics(args.cache, args.horizon)
    sample = tensorize_sample(stats["first_sample"], encode, tasks, stats)
    conventional = profile_model(
        args,
        "conventional",
        args.conventional_checkpoint,
        len(vocab),
        stats,
        sample,
    )
    chi = profile_model(
        args, "chi", args.chi_checkpoint, len(vocab), stats, sample
    )
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "warmup_iters": args.warmup_iters,
        "profile_iters": args.profile_iters,
        "conventional": conventional,
        "chi": chi,
        "eager_chi_to_conventional_latency_ratio": (
            chi["eager_latency_ms_batch1"] / conventional["eager_latency_ms_batch1"]
        ),
        "compiled_chi_to_conventional_latency_ratio": (
            chi["compiled_latency_ms_batch1"] / conventional["compiled_latency_ms_batch1"]
            if chi["compiled_latency_ms_batch1"] is not None
            and conventional["compiled_latency_ms_batch1"] is not None
            else None
        ),
        "scope": (
            "Full image, instruction, state, backbone, and linear action-head forward pass at "
            "batch one on the same checkpoint pair and physical GPU."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
