#!/usr/bin/env python3
"""Build the tuple cache consumed by the Athena χ-VLA trainer for one LIBERO suite."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image


DATASETS = {
    "libero_object": "lerobot/libero_object_image",
    "libero_spatial": "lerobot/libero_spatial_image",
    "libero_goal": "lerobot/libero_goal_image",
    "libero_10": "lerobot/libero_10_image",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(DATASETS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, default=100000)
    parser.add_argument("--res", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    dataset_name = DATASETS[args.suite]
    stream = load_dataset(dataset_name, split="train", streaming=True)
    frames = []
    task_counts: Counter[int] = Counter()
    episode_counts: Counter[int] = Counter()
    for example in stream:
        image = example["observation.images.image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        task_index = int(example["task_index"])
        episode_index = int(example["episode_index"])
        frames.append(
            (
                episode_index,
                int(example["frame_index"]),
                np.asarray(image.resize((args.res, args.res)), dtype=np.uint8),
                np.asarray(example["observation.state"], dtype=np.float32),
                np.asarray(example["action"], dtype=np.float32),
                task_index,
            )
        )
        task_counts[task_index] += 1
        episode_counts[episode_index] += 1
        if len(frames) % 10000 == 0:
            print(f"CACHE {args.suite}: {len(frames)}/{args.n_frames}", flush=True)
        if len(frames) >= args.n_frames:
            break
    if not frames:
        raise RuntimeError(f"No frames were streamed from {dataset_name}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(frames, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(args.output)

    result = {
        "suite": args.suite,
        "dataset": dataset_name,
        "frames": len(frames),
        "resolution": args.res,
        "episodes": len(episode_counts),
        "task_frame_counts": {str(key): value for key, value in sorted(task_counts.items())},
        "elapsed_s": time.perf_counter() - started,
        "cache": str(args.output),
    }
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    args.result_output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
