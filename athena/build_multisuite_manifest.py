#!/usr/bin/env python3
"""Build fixed vocabulary and pooled normalization metadata for LIBERO suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from athena.run_xvla_experiment import build_vocab, load_suite, task_languages


DEFAULT_SUITES = (
    "libero_object",
    "libero_spatial",
    "libero_goal",
    "libero_10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-cache",
        action="append",
        required=True,
        help="Suite and cache path as SUITE=PATH, repeated in global-task order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--res", type=int, default=64)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_suite_caches(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected SUITE=PATH, received {value!r}")
        suite, path = value.split("=", 1)
        result.append((suite, Path(path)))
    suites = tuple(suite for suite, _ in result)
    if suites != DEFAULT_SUITES:
        raise ValueError(
            f"Suite order must be {DEFAULT_SUITES}, received {suites}"
        )
    return result


def cache_moments(path: Path, horizon: int) -> dict[str, Any]:
    with path.open("rb") as handle:
        frames = pickle.load(handle)
    episodes: dict[int, list[Any]] = defaultdict(list)
    image_shape = None
    state_dim = None
    action_dim = None
    for frame in frames:
        if len(frame) < 6:
            raise RuntimeError(f"Malformed frame in {path}")
        frame_image_shape = tuple(np.asarray(frame[2]).shape)
        frame_state_dim = int(np.asarray(frame[3]).shape[-1])
        frame_action_dim = int(np.asarray(frame[4]).shape[-1])
        if image_shape is None:
            image_shape = frame_image_shape
            state_dim = frame_state_dim
            action_dim = frame_action_dim
        if frame_image_shape != image_shape:
            raise RuntimeError(f"Inconsistent image shapes in {path}")
        if frame_state_dim != state_dim or frame_action_dim != action_dim:
            raise RuntimeError(f"Inconsistent state/action dimensions in {path}")
        episodes[int(frame[0])].append(frame)

    action_sum = None
    action_square_sum = None
    state_sum = None
    state_square_sum = None
    action_count = 0
    state_count = 0
    sample_count = 0
    local_tasks = set()
    for episode_frames in episodes.values():
        episode_frames.sort(key=lambda item: int(item[1]))
        for index in range(len(episode_frames) - horizon):
            frame = episode_frames[index]
            expected_indices = [int(frame[1]) + offset for offset in range(horizon)]
            actual_indices = [
                int(episode_frames[index + offset][1]) for offset in range(horizon)
            ]
            if actual_indices != expected_indices:
                raise RuntimeError(
                    f"Noncontiguous horizon window in {path}: "
                    f"{actual_indices} != {expected_indices}"
                )
            action = np.stack(
                [episode_frames[index + offset][4] for offset in range(horizon)]
            ).astype(np.float64)
            state = np.asarray(frame[3], dtype=np.float64)
            if action_sum is None:
                action_sum = np.zeros(action.shape[-1], dtype=np.float64)
                action_square_sum = np.zeros_like(action_sum)
                state_sum = np.zeros(state.shape[-1], dtype=np.float64)
                state_square_sum = np.zeros_like(state_sum)
            action_sum += action.sum(axis=0)
            action_square_sum += np.square(action).sum(axis=0)
            state_sum += state
            state_square_sum += np.square(state)
            action_count += action.shape[0]
            state_count += 1
            sample_count += 1
            local_tasks.add(int(frame[5]))
    if sample_count == 0:
        raise RuntimeError(f"No horizon-{horizon} samples in {path}")
    return {
        "frame_count": len(frames),
        "sample_count": sample_count,
        "local_tasks": sorted(local_tasks),
        "image_shape": list(image_shape),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_sum": action_sum,
        "action_square_sum": action_square_sum,
        "action_count": action_count,
        "state_sum": state_sum,
        "state_square_sum": state_square_sum,
        "state_count": state_count,
    }


def main() -> None:
    args = parse_args()
    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    suite_caches = parse_suite_caches(args.suite_cache)

    source_caches = {}
    suite_offsets = {}
    global_languages = {}
    action_sum = None
    action_square_sum = None
    state_sum = None
    state_square_sum = None
    action_count = 0
    state_count = 0
    task_offset = 0
    expected_state_dim = None
    expected_action_dim = None

    for suite_name, cache_path in suite_caches:
        print(f"Reading {suite_name} from {cache_path}", flush=True)
        moments = cache_moments(cache_path, args.horizon)
        languages = task_languages(load_suite(suite_name))
        if len(languages) != 10 or set(languages) != set(range(10)):
            raise RuntimeError(f"{suite_name} does not expose exactly tasks 0 through 9")
        if moments["local_tasks"] != list(range(10)):
            raise RuntimeError(
                f"{suite_name} cache task IDs are {moments['local_tasks']}, expected 0 through 9"
            )
        if moments["image_shape"] != [args.res, args.res, 3]:
            raise RuntimeError(
                f"{suite_name} image shape is {moments['image_shape']}, "
                f"expected {[args.res, args.res, 3]}"
            )
        if expected_state_dim is None:
            expected_state_dim = moments["state_dim"]
            expected_action_dim = moments["action_dim"]
        if (
            moments["state_dim"] != expected_state_dim
            or moments["action_dim"] != expected_action_dim
        ):
            raise RuntimeError(f"State/action dimensions differ in {suite_name}")
        suite_offsets[suite_name] = task_offset
        for local_task, language in languages.items():
            global_languages[task_offset + local_task] = language
        source_caches[suite_name] = {
            "path": str(cache_path),
            "sha256": file_sha256(cache_path),
            "frame_count": moments["frame_count"],
            "sample_count": moments["sample_count"],
            "local_tasks": moments["local_tasks"],
            "image_shape": moments["image_shape"],
            "state_dim": moments["state_dim"],
            "action_dim": moments["action_dim"],
        }
        if action_sum is None:
            action_sum = moments["action_sum"].copy()
            action_square_sum = moments["action_square_sum"].copy()
            state_sum = moments["state_sum"].copy()
            state_square_sum = moments["state_square_sum"].copy()
        else:
            action_sum += moments["action_sum"]
            action_square_sum += moments["action_square_sum"]
            state_sum += moments["state_sum"]
            state_square_sum += moments["state_square_sum"]
        action_count += moments["action_count"]
        state_count += moments["state_count"]
        task_offset += len(languages)

    action_mean = action_sum / action_count
    state_mean = state_sum / state_count
    action_variance = np.maximum(action_square_sum / action_count - action_mean**2, 0)
    state_variance = np.maximum(state_square_sum / state_count - state_mean**2, 0)
    vocab, _ = build_vocab(global_languages)

    manifest = {
        "format": "xvla_multisuite_v1",
        "training_suites": [suite for suite, _ in suite_caches],
        "suite_task_offsets": suite_offsets,
        "task_languages": {str(key): value for key, value in global_languages.items()},
        "source_caches": source_caches,
        "vocab": vocab,
        "normalization": {
            "action_mean": action_mean.tolist(),
            "action_std": (np.sqrt(action_variance) + 1e-6).tolist(),
            "state_mean": state_mean.tolist(),
            "state_std": (np.sqrt(state_variance) + 1e-6).tolist(),
        },
        "action_count": action_count,
        "state_count": state_count,
        "state_dim": int(state_mean.shape[0]),
        "action_dim": int(action_mean.shape[0]),
        "task_count": task_offset,
        "horizon": args.horizon,
        "res": args.res,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
