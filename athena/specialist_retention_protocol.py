#!/usr/bin/env python3
"""Frozen identities for the hardened four-suite specialist-retention program."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
SHARDS = ((0, 3), (3, 6), (6, 8), (8, 10))
ARCHITECTURE = "chi"
VISION_ENCODER = "vit"
SEED = 0
FULL_STEPS = 40000
SMOKE_STEPS = 10
FULL_BATCH_SIZE = 256
SMOKE_BATCH_SIZE = 64
LEARNING_RATE = 8e-4
EMA_DECAY = 0.999
RESOLUTION = 64
HORIZON = 8
RECIPE_VERSION = "specialist_retention_hardened_v1"
EVALUATION_GPU_FAMILY = "a6000"
EVALUATION_GPU_NAME = "NVIDIA RTX A6000"
GENERALIST_MANIFEST_PATH = Path("artifacts/libero_all_manifest.json")
GENERALIST_MANIFEST_SHA256 = (
    "d0183b465c4d687a4b31f1fc2ca8c75a35786e164cd07f638ca47e347e21f4fd"
)
GENERALIST_SUMMARY_SOURCE_SHA256 = (
    "cd87e66f267472d9b5c934804958f802e176b17618fca7f3be463db4a441a865"
)
PROVENANCE_VERIFIER_SHA256 = (
    "9dc1c6d643d318afa2069c5244079d198beb3b7a20eb049e28f9e7103f887894"
)
PROVENANCE_MISMATCH_FIELDS = {
    "action",
    "episode_index",
    "frame_index",
    "language_join",
    "resized_image",
    "source_missing_frame",
    "state",
    "task_id",
}
CACHE_ENVIRONMENT_KEYS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME",
    "TORCH_HOME",
    "MPLCONFIGDIR",
)
EXPECTED_EVALUATION_RUNTIME = {
    "torch_version": "2.7.1+cu126",
    "cuda_version": "12.6",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": True,
    "libero_version": "0.1.0",
    "robosuite_version": "1.4.1",
    "mujoco_version": "3.5.0",
}


SUITE_SPECS: dict[str, dict[str, Any]] = {
    "libero_object": {
        "cache_path": "artifacts/libero_frames_100000_64.pkl",
        "cache_sha256": "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662",
        "frame_count": 66984,
        "sample_count": 63352,
        "repository": "lerobot/libero_object_image",
        "revision": "e1e080d7df1d0a359dff5c86c222e047549f447f",
        "metadata_sha256": "34caee9641ae50bb4e077de306a7d0031753757882da8b1f117e7ea36a486b42",
        "dataset_to_official_task": {
            "0": 9, "1": 4, "2": 1, "3": 3, "4": 0,
            "5": 7, "6": 2, "7": 6, "8": 5, "9": 8,
        },
        "canonical_content_sha256": "01bc724b9bf8c158b983e34b82bbb40dce9dba2cb86dd9be2a5bf8910be341c7",
        "provenance_path": "results/cache_provenance_libero_object.json",
        "provenance_sha256": "1e3ed7eaeef317a221bb6649ea75ef68ff924b57797682e853b4bd90a138f4c4",
    },
    "libero_spatial": {
        "cache_path": "artifacts/libero_spatial_frames_100000_64.pkl",
        "cache_sha256": "662234d52afff37d55b4b36377cc89151208fc0392147197e2cbec38a49b3fcb",
        "frame_count": 52970,
        "sample_count": 49514,
        "repository": "lerobot/libero_spatial_image",
        "revision": "d86c0b94922572b3b657e1d1a3d01f0952ddeb46",
        "metadata_sha256": "90ef6a4278e2f2307d64f1649ecd4f84ee481cff5007f15d8683f1f9cc917af3",
        "dataset_to_official_task": {
            "0": 6, "1": 4, "2": 5, "3": 7, "4": 0,
            "5": 3, "6": 8, "7": 1, "8": 2, "9": 9,
        },
        "canonical_content_sha256": "8755623bbaa9e7c691bc19c49b8fa62175d2495a1670b68b4813f36ddd58b186",
        "provenance_path": "results/cache_provenance_libero_spatial.json",
        "provenance_sha256": "fd38dc27518206a5b105f9909133947f4b36e5f2cc8e843f67e1db7898a52e6f",
    },
    "libero_goal": {
        "cache_path": "artifacts/libero_goal_frames_100000_64.pkl",
        "cache_sha256": "f0d3ff502f797a0d6e2b611f5be806b2660afcb9d78c883610486f5138a50550",
        "frame_count": 52042,
        "sample_count": 48618,
        "repository": "lerobot/libero_goal_image",
        "revision": "91a97115558b5b611200a432d9c82e4f30991b60",
        "metadata_sha256": "80f750568ef991a406e9e1ffcdfe638eaa2dcf452c5e69c1b911c7dae0de8986",
        "dataset_to_official_task": {
            "0": 8, "1": 9, "2": 3, "3": 6, "4": 2,
            "5": 5, "6": 7, "7": 1, "8": 4, "9": 0,
        },
        "canonical_content_sha256": "8ae09b3b59c72d17b53b074f16b9a06a105671ffb2ae28ec64da055d2403245a",
        "provenance_path": "results/cache_provenance_libero_goal.json",
        "provenance_sha256": "2fdb5c7fe0f5e8de272ad23f6e182c027ada0b97280aaeddac284074f479f430",
    },
    "libero_10": {
        "cache_path": "artifacts/libero_10_frames_100000_64.pkl",
        "cache_sha256": "571691dac732f9aa6c0837dbd3505e497b8dd18ea247b35ae10038c476eb44d0",
        "frame_count": 100000,
        "sample_count": 97008,
        "repository": "lerobot/libero_10_image",
        "revision": "7e324b526699f444044952c82ce3f438e8d300d0",
        "metadata_sha256": "10bb12686977333e729ef5c444cb9eb06377fff5b08a6f87f422dc6bf232205e",
        "dataset_to_official_task": {
            "0": 4, "1": 6, "2": 9, "3": 2, "4": 7,
            "5": 0, "6": 8, "7": 1, "8": 3, "9": 5,
        },
        "canonical_content_sha256": "708573972812d83297c10e18ca037095aa123175ea51742396fa36b96e5ea45c",
        "provenance_path": "results/cache_provenance_libero_10.json",
        "provenance_sha256": "56637d9ed74a947922b0102fb7c4a0488b4b1e978b59a97bb8e7d56e72e08800",
    },
}


MODEL_SOURCE_HASHES = {
    "athena/libero_dataset_metadata.py": "3e14b117ee72b010939c0fdd2c20417778b89a8691156553a0bfb2ad2d0b4edb",
    "athena/run_xvla_experiment.py": "91ae342892e32b0aa019a43ff06b5809dca2a49d4be20ce9a55fbad1b15cdbb3",
    "xvla/__init__.py": "a6eeee3f1c8c9eec78d2101a0a55761c49d24c3a107d32f065fde725ee193626",
    "xvla/models/__init__.py": "142c431a5637c1d97cc1cb8d0ada64b3904dba4d0eabf79a706be141190a71ec",
    "xvla/models/lm.py": "c77279e821bddf168a975b777bfe1f9293fd317085b7cb9a9df771b6fd569aa0",
    "xvla/models/vla.py": "bc276b0328b53cf1f54b42bcd75137df5785cb8625f51d2b09671081d7c4275f",
    "xvla/models/vit.py": "111049ad4c24179e294da8cccb732e3a416223f77b42a08ed56291febe4ddf87",
    "xvla/nn/__init__.py": "2751993d3f6782f60b55f72915744c55762beb02108f05a1c550b89ceb3cfc52",
    "xvla/nn/attention.py": "4f8e49d9dc25ee292b34daf60687f80f38a8f85548ef2905b3217e66dc8c27ec",
    "xvla/nn/baselines.py": "eb4d6ba5f3aa66589991ee3221017dc59092ec9fd35255f9124812420451a8d4",
    "xvla/nn/bilinear.py": "162929529750fe719c6b4199ae316ddb45f72f52de7bbbd07d329dfb5f8287c8",
    "xvla/nn/block.py": "7555c0b7b11c2592c97bf90ec7eeca6ecd0b76f58960df78eefa984386eea969",
    "xvla/nn/flow_action.py": "1108fd31c0091ce51afa342e798ab625514bc5a8064e45b8f0ce38790d71c1aa",
    "xvla/nn/homogeneous.py": "314ba488c638da18d74de213218e310ea75aafa4fbbc92d9cb838aaa5a31701b",
    "xvla/nn/normalization.py": "67406ca22083c28223023ce1efcfc448470654da8477a535bf4a5284cf7338a9",
    "xvla/nn/product_routing.py": "20da357c875e426b2341e1950736ac6cb88700cb1654e46a4b8a2943c666c9b2",
    "xvla/nn/projector.py": "6212e35fbab973de2e79d74bd5064635db12a2c5e48c8c7df7428cbf1e31aaed",
    "xvla/nn/quantile_action.py": "6bd1ecbe5491457fb9ae0154ea566c0ce24e43f102516699f0b3941d4b497c7e",
    "xvla/train/__init__.py": "b1a7a347ca176ef4626ecc1f1d3318dcff63aa5350a9f6df7b8eee19e6058f42",
    "xvla/train/balance.py": "c444c959dfef3935bbad316f514a3175de92a7f5c1df5dea3f1a5e3e146a6208",
    "xvla/train/calibrate.py": "f40387995adeca41a94d1d2c2d524b5fa4bfcd1f1bfda3fa6f7555f2648426f7",
    "xvla/train/data.py": "2b2dd182c654979aff8905097ba577d97c82a386d6d36220dd9e2a49b142c992",
    "xvla/train/exact_odt_attention_proto.py": "048e094f384ef6064cbd02c6d5668024a49ed8a218d4c318f7846e66d47e3238",
}
TRAIN_LM_SHA256 = "e6ecdcc12a9a252afb3b5fcb75de7ff2728a81183b6c05710fd0c23bf6e2f525"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_sha256(sources: dict[str, str]) -> str:
    payload = "".join(
        f"{path}\0{digest}\n" for path, digest in sorted(sources.items())
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def source_snapshot(expected: dict[str, str]) -> dict[str, str]:
    repository_root = Path(__file__).resolve().parents[1]
    return {path: file_sha256(repository_root / path) for path in expected}


def validate_source_snapshot(observed: dict[str, str], expected: dict[str, str]) -> None:
    if observed != expected:
        changed = {
            path: {"expected": expected.get(path), "observed": observed.get(path)}
            for path in sorted(set(expected) | set(observed))
            if expected.get(path) != observed.get(path)
        }
        raise RuntimeError(f"Frozen specialist source mismatch: {changed}")


def validate_task_metadata(suite: str, metadata: dict[str, Any]) -> None:
    spec = SUITE_SPECS[suite]
    expected = {
        "repository": spec["repository"],
        "revision": spec["revision"],
        "metadata_file": "meta/tasks.parquet",
        "metadata_sha256": spec["metadata_sha256"],
        "dataset_to_official_task": spec["dataset_to_official_task"],
        "ordering_matches_official": all(
            int(dataset) == official
            for dataset, official in spec["dataset_to_official_task"].items()
        ),
        "language_set_matches_official": True,
    }
    require_equal(metadata, expected, f"{suite} task metadata")


def validate_work_cache_environment() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for key in CACHE_ENVIRONMENT_KEYS:
        value = os.environ.get(key)
        if not value or not Path(value).is_absolute() or not value.startswith("/work/"):
            raise RuntimeError(f"{key} must be an absolute /work path, got {value!r}")
        snapshot[key] = value
    return snapshot


def validate_cache_and_provenance(suite: str) -> dict[str, Any]:
    spec = SUITE_SPECS[suite]
    cache_path = Path(spec["cache_path"])
    provenance_path = Path(spec["provenance_path"])
    require_equal(file_sha256(cache_path), spec["cache_sha256"], f"{suite} cache SHA-256")
    require_equal(
        file_sha256(provenance_path),
        spec["provenance_sha256"],
        f"{suite} provenance record SHA-256",
    )
    payload = json.loads(provenance_path.read_text())
    require_equal(payload.get("schema"), "xvla-cache-provenance-v1", f"{suite} provenance schema")
    require_equal(payload.get("suite"), suite, f"{suite} provenance suite")
    require_equal(payload.get("verified"), True, f"{suite} provenance verified")
    cache = payload.get("cache", {})
    for key, expected in (
        ("path", spec["cache_path"]),
        ("sha256", spec["cache_sha256"]),
        ("frames", spec["frame_count"]),
        ("canonical_content_sha256", spec["canonical_content_sha256"]),
    ):
        require_equal(cache.get(key), expected, f"{suite} provenance cache {key}")
    metadata = payload.get("metadata", {})
    require_equal(metadata.get("repository"), spec["repository"], f"{suite} metadata repository")
    require_equal(metadata.get("revision"), spec["revision"], f"{suite} metadata revision")
    require_equal(metadata.get("sha256"), spec["metadata_sha256"], f"{suite} metadata SHA-256")
    require_equal(
        metadata.get("dataset_to_official_task"),
        spec["dataset_to_official_task"],
        f"{suite} metadata task mapping",
    )
    require_equal(metadata.get("language_set_matches_official"), True, f"{suite} language identity")
    comparison = payload.get("comparison", {})
    require_equal(
        comparison.get("cached_frames_compared"),
        spec["frame_count"],
        f"{suite} compared frames",
    )
    mismatch_counts = comparison.get("mismatch_counts", {})
    require_equal(set(mismatch_counts), PROVENANCE_MISMATCH_FIELDS, f"{suite} mismatch fields")
    if any(value != 0 for value in mismatch_counts.values()):
        raise RuntimeError(f"{suite} provenance contains mismatches")
    source = payload.get("source", {})
    require_equal(source.get("repository"), spec["repository"], f"{suite} source repository")
    require_equal(source.get("revision"), spec["revision"], f"{suite} source revision")
    require_equal(source.get("content_hashes_match"), True, f"{suite} source content identity")
    require_equal(
        source.get("canonical_content_sha256"),
        spec["canonical_content_sha256"],
        f"{suite} source canonical hash",
    )
    require_equal(
        payload.get("implementation", {}).get("verifier_source_sha256"),
        PROVENANCE_VERIFIER_SHA256,
        f"{suite} verifier source",
    )
    return {
        "cache_path": spec["cache_path"],
        "cache_sha256": spec["cache_sha256"],
        "provenance_path": spec["provenance_path"],
        "provenance_sha256": spec["provenance_sha256"],
        "canonical_content_sha256": spec["canonical_content_sha256"],
        "metadata_sha256": spec["metadata_sha256"],
        "dataset_to_official_task": spec["dataset_to_official_task"],
    }


def checkpoint_path(suite: str, smoke: bool = False) -> Path:
    suffix = "_smoke" if smoke else ""
    return Path(f"artifacts/ckpt_specialist_retention_hardened_{suite}_chi_s0{suffix}.pt")


def metadata_path(suite: str, smoke: bool = False) -> Path:
    suffix = "_smoke" if smoke else ""
    return Path(f"artifacts/ckpt_specialist_retention_hardened_{suite}_chi_s0{suffix}.json")


def training_result_path(suite: str, smoke: bool = False) -> Path:
    suffix = "_smoke" if smoke else ""
    return Path(f"results/train_specialist_retention_hardened_{suite}_chi_s0{suffix}.json")


def evaluation_result_path(
    suite: str, start: int | None = None, end: int | None = None, smoke: bool = False
) -> Path:
    if smoke:
        return Path(f"results/specialist_retention_hardened_{suite}_chi_s0_eval_smoke.json")
    if start is None or end is None:
        raise ValueError("Full specialist result paths require a task shard")
    return Path(
        f"results/specialist_retention_hardened_{suite}_chi_s0_t{start}_{end}.json"
    )
