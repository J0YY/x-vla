#!/usr/bin/env python3
"""Reproduce and verify every frame in one pinned LIBERO tuple cache."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pickle
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import PIL
from datasets import Features
from datasets import Image as DatasetImage
from datasets import Sequence, Value, load_dataset
from PIL import Image

from athena.libero_dataset_metadata import (
    DATASETS,
    file_sha256,
    load_dataset_task_languages,
)


MISMATCH_FIELDS = (
    "episode_index",
    "frame_index",
    "task_id",
    "language_join",
    "state",
    "action",
    "resized_image",
    "source_missing_frame",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(DATASETS), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument(
        "--expected-cache-frames",
        type=int,
        required=True,
        help="Frozen frame count for the named cache artifact.",
    )
    parser.add_argument(
        "--max-cache-frames",
        type=int,
        default=100000,
        help="Hard audit bound. The verifier refuses a larger cache.",
    )
    parser.add_argument(
        "--mismatch-example-limit",
        type=int,
        default=20,
        help="Maximum diagnostic frame records retained in JSON.",
    )
    return parser.parse_args()


def _update_blob(digest: Any, label: str, value: bytes) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(struct.pack("<I", len(label_bytes)))
    digest.update(label_bytes)
    digest.update(struct.pack("<Q", len(value)))
    digest.update(value)


def _update_array(digest: Any, label: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    _update_blob(digest, f"{label}.dtype", array.dtype.str.encode("ascii"))
    _update_blob(
        digest,
        f"{label}.shape",
        np.asarray(array.shape, dtype="<i8").tobytes(),
    )
    _update_blob(digest, f"{label}.data", array.tobytes())


def _update_record(
    digest: Any,
    *,
    episode_index: int,
    frame_index: int,
    task_index: int,
    language: str,
    state: np.ndarray,
    action: np.ndarray,
    image: np.ndarray,
) -> None:
    _update_blob(digest, "record", b"libero-cache-v1")
    _update_blob(digest, "episode_index", struct.pack("<q", episode_index))
    _update_blob(digest, "frame_index", struct.pack("<q", frame_index))
    _update_blob(digest, "task_index", struct.pack("<q", task_index))
    _update_blob(digest, "task_language", language.encode("utf-8"))
    _update_array(digest, "state", state)
    _update_array(digest, "action", action)
    _update_array(digest, "resized_image", image)


def _exact_array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.ascontiguousarray(left).tobytes()
        == np.ascontiguousarray(right).tobytes()
    )


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape != right.shape or left.size == 0:
        return None
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    if not np.all(np.isfinite(difference)):
        return None
    return float(difference.max(initial=0.0))


def _language_for(task_index: int, languages: dict[int, str]) -> str:
    return languages.get(task_index, f"<missing-task-{task_index}>")


def _write_json(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.res <= 0:
        raise ValueError("--res must be positive")
    if args.max_cache_frames <= 0:
        raise ValueError("--max-cache-frames must be positive")
    if args.mismatch_example_limit < 0:
        raise ValueError("--mismatch-example-limit cannot be negative")
    if not args.cache.is_file():
        raise FileNotFoundError(args.cache)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    dataset_name, dataset_revision = DATASETS[args.suite]
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    official_languages = {
        task_index: str(suite.get_task(task_index).language)
        for task_index in range(suite.n_tasks)
    }
    dataset_languages, metadata = load_dataset_task_languages(
        args.suite,
        official_languages,
    )
    dataset_to_official = {
        int(key): int(value)
        for key, value in metadata["dataset_to_official_task"].items()
    }

    cache_sha256 = file_sha256(args.cache)
    with args.cache.open("rb") as handle:
        cache = pickle.load(handle)
    if not isinstance(cache, (list, tuple)):
        raise TypeError(f"Cache must be a list or tuple, got {type(cache).__name__}")
    cache_frames = len(cache)
    if cache_frames != args.expected_cache_frames:
        raise RuntimeError(
            f"{args.cache} has {cache_frames} frames, expected "
            f"{args.expected_cache_frames}"
        )
    if cache_frames > args.max_cache_frames:
        raise RuntimeError(
            f"Cache has {cache_frames} frames, exceeding hard audit bound "
            f"{args.max_cache_frames}"
        )

    source_features = Features(
        {
            "observation.images.image": DatasetImage(),
            "observation.state": Sequence(Value("float32"), length=8),
            "action": Sequence(Value("float32"), length=7),
            "episode_index": Value("int64"),
            "frame_index": Value("int64"),
            "task_index": Value("int64"),
        }
    )
    stream = load_dataset(
        dataset_name,
        revision=dataset_revision,
        split="train",
        streaming=True,
        features=source_features,
        columns=list(source_features),
    )
    source_iterator = iter(stream)
    cache_digest = hashlib.sha256()
    source_digest = hashlib.sha256()
    mismatch_counts = {field: 0 for field in MISMATCH_FIELDS}
    mismatch_examples: list[dict[str, Any]] = []
    max_abs_difference: dict[str, float | None] = {
        "state": 0.0,
        "action": 0.0,
        "resized_image": 0.0,
    }
    compared_frames = 0
    source_missing = False

    for offset, cached_record in enumerate(cache):
        if not isinstance(cached_record, (list, tuple)) or len(cached_record) != 6:
            raise RuntimeError(
                f"Cache record {offset} must be a six-field tuple, got "
                f"{type(cached_record).__name__} of length "
                f"{len(cached_record) if hasattr(cached_record, '__len__') else 'unknown'}"
            )
        try:
            source_record = next(source_iterator)
        except StopIteration:
            missing = cache_frames - offset
            mismatch_counts["source_missing_frame"] += missing
            source_missing = True
            if len(mismatch_examples) < args.mismatch_example_limit:
                mismatch_examples.append(
                    {
                        "cache_offset": offset,
                        "fields": ["source_missing_frame"],
                        "remaining_cached_frames": missing,
                    }
                )
            break

        cached_episode = int(cached_record[0])
        cached_frame = int(cached_record[1])
        cached_image = np.asarray(cached_record[2])
        cached_state = np.asarray(cached_record[3])
        cached_action = np.asarray(cached_record[4])
        cached_task = int(cached_record[5])

        source_episode = int(source_record["episode_index"])
        source_frame = int(source_record["frame_index"])
        source_task = int(source_record["task_index"])
        source_image_object = source_record["observation.images.image"]
        if not isinstance(source_image_object, Image.Image):
            source_image_object = Image.fromarray(np.asarray(source_image_object))
        source_image = np.asarray(
            source_image_object.resize((args.res, args.res)),
            dtype=np.uint8,
        )
        source_state = np.asarray(
            source_record["observation.state"],
            dtype=np.float32,
        )
        source_action = np.asarray(source_record["action"], dtype=np.float32)

        cached_language = _language_for(cached_task, dataset_languages)
        source_language = _language_for(source_task, dataset_languages)
        cached_official = dataset_to_official.get(cached_task)
        source_official = dataset_to_official.get(source_task)

        _update_record(
            cache_digest,
            episode_index=cached_episode,
            frame_index=cached_frame,
            task_index=cached_task,
            language=cached_language,
            state=cached_state,
            action=cached_action,
            image=cached_image,
        )
        _update_record(
            source_digest,
            episode_index=source_episode,
            frame_index=source_frame,
            task_index=source_task,
            language=source_language,
            state=source_state,
            action=source_action,
            image=source_image,
        )

        fields: list[str] = []
        details: dict[str, Any] = {}
        scalar_checks = (
            ("episode_index", cached_episode, source_episode),
            ("frame_index", cached_frame, source_frame),
            ("task_id", cached_task, source_task),
        )
        for field, cached_value, source_value in scalar_checks:
            if cached_value != source_value:
                mismatch_counts[field] += 1
                fields.append(field)
                details[field] = {
                    "cache": cached_value,
                    "source": source_value,
                }

        language_join_matches = (
            cached_task in dataset_languages
            and source_task in dataset_languages
            and cached_official is not None
            and source_official is not None
            and cached_language == source_language
            and cached_official == source_official
        )
        if not language_join_matches:
            mismatch_counts["language_join"] += 1
            fields.append("language_join")
            details["language_join"] = {
                "cache_dataset_task": cached_task,
                "source_dataset_task": source_task,
                "cache_language": cached_language,
                "source_language": source_language,
                "cache_official_task": cached_official,
                "source_official_task": source_official,
            }

        array_checks = (
            ("state", cached_state, source_state),
            ("action", cached_action, source_action),
            ("resized_image", cached_image, source_image),
        )
        for field, cached_value, source_value in array_checks:
            if not _exact_array_equal(cached_value, source_value):
                mismatch_counts[field] += 1
                fields.append(field)
                difference = _max_abs_difference(cached_value, source_value)
                if difference is not None:
                    previous = max_abs_difference[field]
                    max_abs_difference[field] = max(float(previous or 0.0), difference)
                details[field] = {
                    "cache_shape": list(cached_value.shape),
                    "source_shape": list(source_value.shape),
                    "cache_dtype": cached_value.dtype.str,
                    "source_dtype": source_value.dtype.str,
                    "max_abs_difference": difference,
                }

        if fields and len(mismatch_examples) < args.mismatch_example_limit:
            mismatch_examples.append(
                {
                    "cache_offset": offset,
                    "fields": fields,
                    "details": details,
                }
            )
        compared_frames += 1
        if compared_frames % 10000 == 0:
            print(
                f"VERIFY {args.suite}: {compared_frames}/{cache_frames}",
                flush=True,
            )

    source_has_additional_frame: bool | None = None
    if not source_missing:
        try:
            next(source_iterator)
            source_has_additional_frame = True
        except StopIteration:
            source_has_additional_frame = False

    cache_content_sha256 = cache_digest.hexdigest()
    source_content_sha256 = source_digest.hexdigest()
    content_hashes_match = (
        not source_missing
        and compared_frames == cache_frames
        and cache_content_sha256 == source_content_sha256
    )
    verified = (
        compared_frames == cache_frames
        and not source_missing
        and all(value == 0 for value in mismatch_counts.values())
        and content_hashes_match
        and bool(metadata["language_set_matches_official"])
    )
    result = {
        "schema": "xvla-cache-provenance-v1",
        "suite": args.suite,
        "verified": verified,
        "cache": {
            "path": str(args.cache),
            "sha256": cache_sha256,
            "frames": cache_frames,
            "resolution": args.res,
            "canonical_content_sha256": cache_content_sha256,
        },
        "source": {
            "repository": dataset_name,
            "revision": dataset_revision,
            "split": "train",
            "canonical_content_sha256": source_content_sha256,
            "content_hashes_match": content_hashes_match,
            "has_additional_frame_after_cache": source_has_additional_frame,
        },
        "metadata": {
            "repository": metadata["repository"],
            "revision": metadata["revision"],
            "file": metadata["metadata_file"],
            "sha256": metadata["metadata_sha256"],
            "dataset_to_official_task": metadata[
                "dataset_to_official_task"
            ],
            "language_set_matches_official": metadata[
                "language_set_matches_official"
            ],
        },
        "comparison": {
            "cached_frames_compared": compared_frames,
            "mismatch_counts": mismatch_counts,
            "max_abs_difference": max_abs_difference,
            "mismatch_examples": mismatch_examples,
        },
        "limits": {
            "max_cache_frames": args.max_cache_frames,
            "expected_cache_frames": args.expected_cache_frames,
            "source_records_read_max": cache_frames + 1,
            "source_extra_probe_records": 1,
            "mismatch_example_limit": args.mismatch_example_limit,
        },
        "implementation": {
            "verifier_source_sha256": file_sha256(Path(__file__).resolve()),
            "numpy_version": np.__version__,
            "pillow_version": PIL.__version__,
            "datasets_version": importlib.metadata.version("datasets"),
            "resize": (
                f"PIL.Image.resize(({args.res}, {args.res})) with library "
                "default resampling"
            ),
            "array_comparison": "exact dtype, shape, and contiguous bytes after cache casts",
            "canonical_record_schema": "libero-cache-v1",
            "source_feature_schema": repr(source_features),
        },
        "elapsed_s": time.perf_counter() - started,
    }
    _write_json(args.output, result)
    print("RESULT", json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not verified:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
