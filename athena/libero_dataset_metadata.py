#!/usr/bin/env python3
"""Pinned LeRobot task-language metadata for the four LIBERO suites."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download


DATASETS = {
    "libero_object": (
        "lerobot/libero_object_image",
        "e1e080d7df1d0a359dff5c86c222e047549f447f",
    ),
    "libero_spatial": (
        "lerobot/libero_spatial_image",
        "d86c0b94922572b3b657e1d1a3d01f0952ddeb46",
    ),
    "libero_goal": (
        "lerobot/libero_goal_image",
        "91a97115558b5b611200a432d9c82e4f30991b60",
    ),
    "libero_10": (
        "lerobot/libero_10_image",
        "7e324b526699f444044952c82ce3f438e8d300d0",
    ),
}


def normalize_language(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_task_languages(
    suite_name: str,
    official_languages: dict[int, str],
) -> tuple[dict[int, str], dict[str, Any]]:
    """Load the pinned dataset mapping and verify its language set against LIBERO."""
    repository, revision = DATASETS[suite_name]
    metadata_path = Path(
        hf_hub_download(
            repository,
            "meta/tasks.parquet",
            repo_type="dataset",
            revision=revision,
        )
    )
    frame = pd.read_parquet(metadata_path).reset_index()
    if not {"task_index", "task"}.issubset(frame.columns):
        raise RuntimeError(
            f"{repository} task metadata columns are {list(frame.columns)}"
        )
    dataset_languages = {
        int(row["task_index"]): str(row["task"])
        for _, row in frame.iterrows()
    }
    expected_indices = set(range(10))
    if set(dataset_languages) != expected_indices:
        raise RuntimeError(
            f"{repository} task IDs are {sorted(dataset_languages)}, expected 0 through 9"
        )
    if set(official_languages) != expected_indices:
        raise RuntimeError(
            f"Official {suite_name} task IDs are {sorted(official_languages)}"
        )

    official_by_language = {
        normalize_language(language): task_index
        for task_index, language in official_languages.items()
    }
    dataset_by_language = {
        normalize_language(language): task_index
        for task_index, language in dataset_languages.items()
    }
    if len(official_by_language) != 10 or len(dataset_by_language) != 10:
        raise RuntimeError(f"Duplicate normalized task languages in {suite_name}")
    if set(dataset_by_language) != set(official_by_language):
        missing = sorted(set(official_by_language) - set(dataset_by_language))
        extra = sorted(set(dataset_by_language) - set(official_by_language))
        raise RuntimeError(
            f"Dataset/official task-language mismatch for {suite_name}: "
            f"missing={missing}, extra={extra}"
        )

    dataset_to_official = {
        dataset_index: official_by_language[normalize_language(language)]
        for dataset_index, language in dataset_languages.items()
    }
    provenance = {
        "repository": repository,
        "revision": revision,
        "metadata_file": "meta/tasks.parquet",
        "metadata_sha256": file_sha256(metadata_path),
        "dataset_to_official_task": {
            str(key): value for key, value in sorted(dataset_to_official.items())
        },
        "ordering_matches_official": all(
            dataset_to_official[index] == index for index in range(10)
        ),
        "language_set_matches_official": True,
    }
    return dataset_languages, provenance
