#!/usr/bin/env python3
"""Strictly aggregate the frozen six-checkpoint, four-suite generalist run."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any


ARCHITECTURES = ("chi", "conventional")
SEEDS = (0, 1, 2)
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
SHARDS = ((0, 3), (3, 6), (6, 8), (8, 10))
MANIFEST_PATH = Path("artifacts/libero_all_manifest.json")
MANIFEST_SHA256 = "d0183b465c4d687a4b31f1fc2ca8c75a35786e164cd07f638ca47e347e21f4fd"
EVALUATOR_SHA256 = "91ae342892e32b0aa019a43ff06b5809dca2a49d4be20ce9a55fbad1b15cdbb3"
TRAINER_SHA256 = "9b63cce90d6c99a3e4f9ecef1db2914352e0221f57ed0bbd3e159eb6c1534f98"
TRAIN_LM_SHA256 = "e6ecdcc12a9a252afb3b5fcb75de7ff2728a81183b6c05710fd0c23bf6e2f525"
EXPECTED_PARAMETERS = {"chi": 20154632, "conventional": 20155912}
EXPECTED_EVALUATION_GPU_FAMILY = "a6000"
EXPECTED_EVALUATION_GPU_NAME = "NVIDIA RTX A6000"
PROVENANCE_VERIFIER_SHA256 = "9dc1c6d643d318afa2069c5244079d198beb3b7a20eb049e28f9e7103f887894"
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
EXPECTED_SUITE_OFFSETS = {
    "libero_object": 0,
    "libero_spatial": 10,
    "libero_goal": 20,
    "libero_10": 30,
}
CACHES = {
    "libero_object": {
        "path": "artifacts/libero_frames_100000_64.pkl",
        "sha256": "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662",
        "frame_count": 66984,
        "sample_count": 63352,
        "repository": "lerobot/libero_object_image",
        "revision": "e1e080d7df1d0a359dff5c86c222e047549f447f",
        "metadata_sha256": "34caee9641ae50bb4e077de306a7d0031753757882da8b1f117e7ea36a486b42",
    },
    "libero_spatial": {
        "path": "artifacts/libero_spatial_frames_100000_64.pkl",
        "sha256": "662234d52afff37d55b4b36377cc89151208fc0392147197e2cbec38a49b3fcb",
        "frame_count": 52970,
        "sample_count": 49514,
        "repository": "lerobot/libero_spatial_image",
        "revision": "d86c0b94922572b3b657e1d1a3d01f0952ddeb46",
        "metadata_sha256": "90ef6a4278e2f2307d64f1649ecd4f84ee481cff5007f15d8683f1f9cc917af3",
    },
    "libero_goal": {
        "path": "artifacts/libero_goal_frames_100000_64.pkl",
        "sha256": "f0d3ff502f797a0d6e2b611f5be806b2660afcb9d78c883610486f5138a50550",
        "frame_count": 52042,
        "sample_count": 48618,
        "repository": "lerobot/libero_goal_image",
        "revision": "91a97115558b5b611200a432d9c82e4f30991b60",
        "metadata_sha256": "80f750568ef991a406e9e1ffcdfe638eaa2dcf452c5e69c1b911c7dae0de8986",
    },
    "libero_10": {
        "path": "artifacts/libero_10_frames_100000_64.pkl",
        "sha256": "571691dac732f9aa6c0837dbd3505e497b8dd18ea247b35ae10038c476eb44d0",
        "frame_count": 100000,
        "sample_count": 97008,
        "repository": "lerobot/libero_10_image",
        "revision": "7e324b526699f444044952c82ce3f438e8d300d0",
        "metadata_sha256": "10bb12686977333e729ef5c444cb9eb06377fff5b08a6f87f422dc6bf232205e",
    },
}
IMPORTED_SOURCES = {
    "athena/libero_dataset_metadata.py": "3e14b117ee72b010939c0fdd2c20417778b89a8691156553a0bfb2ad2d0b4edb",
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
TRAINER_IMPORTED_SOURCES = {
    **IMPORTED_SOURCES,
    "athena/run_xvla_experiment.py": EVALUATOR_SHA256,
    "xvla/train/train_lm.py": TRAIN_LM_SHA256,
}
EXPECTED_NUMERICAL_ENVIRONMENT = {
    "torch_version": "2.7.1+cu126",
    "cuda_version": "12.6",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": True,
    "libero_version": "0.1.0",
    "robosuite_version": "1.4.1",
    "mujoco_version": "3.5.0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/generalist_hardened_summary.json"),
    )
    return parser.parse_args()


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


def load_json_with_sha(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def validate_cache_table(metadata: dict[str, Any], label: str) -> None:
    sources = metadata.get("source_caches", {})
    require_equal(set(sources), set(SUITES), f"{label} source-cache suite set")
    for suite, expected in CACHES.items():
        source = sources[suite]
        for key in ("path", "sha256", "frame_count", "sample_count"):
            require_equal(source.get(key), expected[key], f"{label} {suite} {key}")
        task_metadata = source.get("task_metadata", {})
        for key in ("repository", "revision", "metadata_sha256"):
            require_equal(
                task_metadata.get(key), expected[key], f"{label} {suite} metadata {key}"
            )


def validate_training_metadata(
    metadata: dict[str, Any],
    architecture: str,
    seed: int,
    expected_checkpoint: str,
    live_manifest: dict[str, Any],
    label: str,
) -> None:
    expected = {
        "format": "xvla_multisuite_checkpoint_v1",
        "architecture": architecture,
        "vision_encoder": "vit",
        "seed": seed,
        "steps": 160000,
        "batch_size": 256,
        "suite_batch_size": 64,
        "lr": 8e-4,
        "ema_decay": 0.999,
        "res": 64,
        "horizon": 8,
        "state_dim": 8,
        "action_dim": 7,
        "task_count": 40,
        "training_suites": list(SUITES),
        "suite_task_offsets": EXPECTED_SUITE_OFFSETS,
        "source_manifest": MANIFEST_PATH.as_posix(),
        "manifest_sha256": MANIFEST_SHA256,
        "recipe_version": "multisuite_balanced_v1",
        "trainer_sha256": TRAINER_SHA256,
    }
    for key, value in expected.items():
        require_equal(metadata.get(key), value, f"{label} training metadata {key}")
    require_equal(
        Path(str(metadata.get("checkpoint", ""))).name,
        expected_checkpoint,
        f"{label} checkpoint filename",
    )
    checkpoint_sha = metadata.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
        raise RuntimeError(f"{label} invalid checkpoint SHA-256")
    for key, value in live_manifest.items():
        if key == "format":
            continue
        require_equal(metadata.get(key), value, f"{label} manifest field {key}")
    validate_cache_table(metadata, label)

    source = metadata.get("source_identity", {})
    require_equal(
        source.get("expected_trainer_sha256"), TRAINER_SHA256, f"{label} trainer pin"
    )
    require_equal(
        source.get("trainer_start_sha256"), TRAINER_SHA256, f"{label} trainer start"
    )
    require_equal(
        source.get("trainer_end_sha256"), TRAINER_SHA256, f"{label} trainer end"
    )
    require_equal(
        source.get("imported_sources_start"),
        TRAINER_IMPORTED_SOURCES,
        f"{label} trainer imported sources start",
    )
    require_equal(
        source.get("imported_sources_end"),
        TRAINER_IMPORTED_SOURCES,
        f"{label} trainer imported sources end",
    )
    expected_bundle = bundle_sha256(TRAINER_IMPORTED_SOURCES)
    require_equal(
        source.get("imported_bundle_start_sha256"),
        expected_bundle,
        f"{label} trainer bundle start",
    )
    require_equal(
        source.get("imported_bundle_end_sha256"),
        expected_bundle,
        f"{label} trainer bundle end",
    )
    require_equal(
        source.get("manifest_start_sha256"), MANIFEST_SHA256, f"{label} manifest start"
    )
    require_equal(
        source.get("manifest_end_sha256"), MANIFEST_SHA256, f"{label} manifest end"
    )
    expected_cache_hashes = {suite: CACHES[suite]["sha256"] for suite in SUITES}
    require_equal(
        source.get("source_cache_hashes_start"),
        expected_cache_hashes,
        f"{label} cache hashes start",
    )
    require_equal(
        source.get("source_cache_hashes_end"),
        expected_cache_hashes,
        f"{label} cache hashes end",
    )
    environment = metadata.get("training_environment", {})
    require_equal(environment.get("matmul_precision"), "high", f"{label} train precision")
    require_equal(environment.get("autocast_dtype"), "bfloat16", f"{label} train autocast")
    for key in ("torch_version", "cuda_version", "gpu"):
        if not environment.get(key):
            raise RuntimeError(f"{label} missing training environment {key}")


def validate_evaluation_source(
    result: dict[str, Any],
    metadata: dict[str, Any],
    suite: str,
    label: str,
) -> None:
    identity = result.get("source_identity", {})
    require_equal(
        identity.get("expected_evaluator_sha256"), EVALUATOR_SHA256, f"{label} evaluator pin"
    )
    start = identity.get("start", {})
    end = identity.get("end", {})
    require_equal(start, end, f"{label} evaluation start/end identity")
    require_equal(start.get("evaluator_sha256"), EVALUATOR_SHA256, f"{label} evaluator")
    require_equal(
        start.get("evaluation_gpu_family"),
        EXPECTED_EVALUATION_GPU_FAMILY,
        f"{label} evaluation GPU family",
    )
    require_equal(
        start.get("evaluation_gpu_name"),
        EXPECTED_EVALUATION_GPU_NAME,
        f"{label} evaluation GPU name",
    )
    require_equal(
        start.get("imported_sources"), IMPORTED_SOURCES, f"{label} evaluator imports"
    )
    require_equal(
        start.get("imported_bundle_sha256"),
        bundle_sha256(IMPORTED_SOURCES),
        f"{label} evaluator bundle",
    )
    require_equal(
        start.get("checkpoint_sha256"),
        metadata.get("checkpoint_sha256"),
        f"{label} evaluated checkpoint",
    )
    require_equal(start.get("manifest_sha256"), MANIFEST_SHA256, f"{label} manifest")
    require_equal(
        start.get("evaluation_cache_sha256"),
        CACHES[suite]["sha256"],
        f"{label} evaluation cache",
    )
    metadata_sha = start.get("model_metadata_sha256")
    if not isinstance(metadata_sha, str) or len(metadata_sha) != 64:
        raise RuntimeError(f"{label} invalid metadata SHA-256")


def load_live_artifact_identity(
    artifacts_dir: Path,
    architecture: str,
    seed: int,
    live_manifest: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_name = f"ckpt_generalist_hardened_{architecture}_s{seed}.pt"
    metadata_name = f"ckpt_generalist_hardened_{architecture}_s{seed}.json"
    checkpoint_path = artifacts_dir / checkpoint_name
    metadata_path = artifacts_dir / metadata_name
    checkpoint_sha256 = file_sha256(checkpoint_path)
    metadata, metadata_sha256 = load_json_with_sha(metadata_path)
    label = f"live {architecture} seed {seed} artifacts"
    validate_training_metadata(
        metadata,
        architecture,
        seed,
        checkpoint_name,
        live_manifest,
        label,
    )
    require_equal(
        metadata.get("checkpoint"), checkpoint_path.as_posix(), f"{label} checkpoint path"
    )
    require_equal(
        metadata.get("checkpoint_sha256"), checkpoint_sha256, f"{label} checkpoint hash"
    )
    if not zipfile.is_zipfile(checkpoint_path):
        raise RuntimeError(f"{label} checkpoint is not a PyTorch ZIP archive")
    with zipfile.ZipFile(checkpoint_path) as archive:
        members = sorted(archive.namelist())
        bad_members = [
            member
            for member in members
            if member.startswith("/") or ".." in Path(member).parts
        ]
        if bad_members or not any(member.endswith("/data.pkl") for member in members):
            raise RuntimeError(f"{label} checkpoint archive structure is invalid")
    archive_index_sha256 = hashlib.sha256("\n".join(members).encode()).hexdigest()
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_archive_index_sha256": archive_index_sha256,
        "metadata_path": metadata_path,
        "metadata_sha256": metadata_sha256,
        "metadata": metadata,
    }


def load_checkpoint_results(
    results_dir: Path,
    architecture: str,
    seed: int,
    live_manifest: dict[str, Any],
    live_checkpoint_sha256: str,
    live_metadata: dict[str, Any],
    live_metadata_sha256: str,
) -> dict[str, Any]:
    task_rows: dict[tuple[str, int], list[bool]] = defaultdict(list)
    task_protocol: dict[str, dict[str, Any]] = {}
    source_files: list[dict[str, str]] = []
    seen_episodes: set[tuple[str, int, int]] = set()
    expected_checkpoint = f"ckpt_generalist_hardened_{architecture}_s{seed}.pt"
    expected_metadata = f"ckpt_generalist_hardened_{architecture}_s{seed}.json"
    canonical_metadata: dict[str, Any] | None = None
    canonical_metadata_sha: str | None = None
    canonical_runtime: dict[str, Any] | None = None
    canonical_gpu: str | None = None

    for suite in SUITES:
        for start, end in SHARDS:
            path = results_dir / f"generalist_hardened_{architecture}_s{seed}_{suite}_t{start}_{end}.json"
            result, result_sha = load_json_with_sha(path)
            label = path.as_posix()
            source_files.append({"path": label, "sha256": result_sha})
            expected_tasks = list(range(start, end))
            identity = {
                "mode": result.get("mode"),
                "architecture": result.get("architecture"),
                "vision_encoder": result.get("vision_encoder"),
                "suite": result.get("suite"),
                "training_suite": result.get("training_suite"),
                "seed": result.get("seed"),
                "checkpoint": Path(str(result.get("checkpoint", ""))).name,
                "metadata": Path(str(result.get("model_metadata", ""))).name,
                "cache": str(result.get("cache", "")),
                "precision": result.get("matmul_precision"),
            }
            expected_identity = {
                "mode": "capability",
                "architecture": architecture,
                "vision_encoder": "vit",
                "suite": suite,
                "training_suite": suite,
                "seed": seed,
                "checkpoint": expected_checkpoint,
                "metadata": expected_metadata,
                "cache": CACHES[suite]["path"],
                "precision": "highest",
            }
            require_equal(identity, expected_identity, f"{label} result identity")
            require_equal(
                result.get("evaluation_scope"),
                "In-domain evaluation of a jointly trained multi-suite checkpoint",
                f"{label} evaluation scope",
            )
            require_equal(result.get("prediction_finite"), True, f"{label} finite prediction")
            require_equal(result.get("prediction_shape"), [1, 8, 7], f"{label} prediction shape")
            require_equal(
                result.get("profile", {}).get("parameters"),
                EXPECTED_PARAMETERS[architecture],
                f"{label} parameter count",
            )
            gpu = result.get("profile", {}).get("gpu")
            require_equal(gpu, EXPECTED_EVALUATION_GPU_NAME, f"{label} evaluation GPU")
            if canonical_gpu is None:
                canonical_gpu = gpu
            require_equal(gpu, canonical_gpu, f"{label} checkpoint evaluation GPU")
            runtime = result.get("numerical_environment")
            require_equal(runtime, EXPECTED_NUMERICAL_ENVIRONMENT, f"{label} runtime")
            if canonical_runtime is None:
                canonical_runtime = runtime
            require_equal(runtime, canonical_runtime, f"{label} checkpoint runtime")
            require_equal(
                result.get("evaluation_protocol"),
                {
                    "res": 64,
                    "horizon": 8,
                    "num_steps_wait": 10,
                    "exec_h": 8,
                    "eps_per_task": 50,
                    "max_steps": 280,
                },
                f"{label} evaluation protocol",
            )
            cache_stats = result.get("cache_stats", {})
            for key, expected in (
                ("frame_count", CACHES[suite]["frame_count"]),
                ("sample_count", CACHES[suite]["sample_count"]),
                ("state_dim", 8),
                ("action_dim", 7),
            ):
                require_equal(cache_stats.get(key), expected, f"{label} cache stats {key}")

            metadata = result.get("training_metadata") or {}
            validate_training_metadata(
                metadata, architecture, seed, expected_checkpoint, live_manifest, label
            )
            validate_evaluation_source(result, metadata, suite, label)
            require_equal(metadata, live_metadata, f"{label} live metadata payload")
            require_equal(
                metadata.get("checkpoint_sha256"),
                live_checkpoint_sha256,
                f"{label} live checkpoint SHA-256",
            )
            require_equal(
                result.get("cache_task_metadata"),
                metadata["source_caches"][suite]["task_metadata"],
                f"{label} task metadata",
            )
            require_equal(result.get("vocab_size"), len(metadata["vocab"]), f"{label} vocab")
            metadata_sha = result["source_identity"]["start"]["model_metadata_sha256"]
            require_equal(
                metadata_sha,
                live_metadata_sha256,
                f"{label} live metadata SHA-256",
            )
            if canonical_metadata is None:
                canonical_metadata = metadata
                canonical_metadata_sha = metadata_sha
            require_equal(metadata, canonical_metadata, f"{label} checkpoint metadata payload")
            require_equal(metadata_sha, canonical_metadata_sha, f"{label} metadata file SHA")

            capability = result.get("capability", {})
            require_equal(
                {
                    "task_indices": capability.get("task_indices"),
                    "eps_per_task": capability.get("eps_per_task"),
                    "max_steps": capability.get("max_steps"),
                    "num_steps_wait": capability.get("num_steps_wait"),
                    "exec_h": capability.get("exec_h"),
                    "canonical_init_states": capability.get("canonical_init_states"),
                },
                {
                    "task_indices": expected_tasks,
                    "eps_per_task": 50,
                    "max_steps": 280,
                    "num_steps_wait": 10,
                    "exec_h": 8,
                    "canonical_init_states": True,
                },
                f"{label} capability protocol",
            )
            shard_protocol = capability.get("task_protocol", {})
            require_equal(set(shard_protocol), {str(task) for task in expected_tasks}, f"{label} task protocol keys")
            for task in expected_tasks:
                protocol = shard_protocol[str(task)]
                dataset_to_official = metadata["source_caches"][suite][
                    "task_metadata"
                ]["dataset_to_official_task"]
                official_to_dataset = {
                    int(official): int(dataset)
                    for dataset, official in dataset_to_official.items()
                }
                dataset_task = official_to_dataset[task]
                expected_language = live_manifest["task_languages"][
                    str(EXPECTED_SUITE_OFFSETS[suite] + dataset_task)
                ]
                require_equal(
                    protocol.get("language"),
                    expected_language,
                    f"{label} task {task} language",
                )
                if int(protocol.get("init_state_count", -1)) < 50:
                    raise RuntimeError(f"{label} task {task} has fewer than 50 init states")
                for key in ("language", "problem_folder", "bddl_file", "bddl_sha256", "init_states_sha256"):
                    if not protocol.get(key):
                        raise RuntimeError(f"{label} task {task} protocol missing {key}")
                task_protocol[f"{suite}:{task}"] = protocol

            expected_trials = len(expected_tasks) * 50
            episodes = capability.get("episodes")
            if not isinstance(episodes, list):
                raise RuntimeError(f"{label} episodes are missing")
            require_equal(len(episodes), expected_trials, f"{label} episode row count")
            shard_episodes: set[tuple[str, int, int]] = set()
            shard_successes = 0
            per_task_successes = {task: 0 for task in expected_tasks}
            for episode in episodes:
                task = episode.get("task_index")
                episode_index = episode.get("episode")
                success = episode.get("success")
                steps = episode.get("steps")
                if type(task) is not int or task not in expected_tasks:
                    raise RuntimeError(f"{label} invalid task row {episode}")
                if type(episode_index) is not int or not 0 <= episode_index < 50:
                    raise RuntimeError(f"{label} invalid episode row {episode}")
                if type(success) is not bool:
                    raise RuntimeError(f"{label} success is not Boolean: {episode}")
                if type(steps) is not int or not 0 <= steps <= 280:
                    raise RuntimeError(f"{label} invalid step count: {episode}")
                episode_key = (suite, task, episode_index)
                if episode_key in shard_episodes or episode_key in seen_episodes:
                    raise RuntimeError(f"{label} duplicate episode {episode_key}")
                shard_episodes.add(episode_key)
                seen_episodes.add(episode_key)
                task_rows[(suite, task)].append(success)
                shard_successes += int(success)
                per_task_successes[task] += int(success)
            require_equal(capability.get("trials"), expected_trials, f"{label} trials")
            require_equal(capability.get("successes"), shard_successes, f"{label} successes")
            require_equal(
                capability.get("overall"),
                shard_successes / expected_trials,
                f"{label} overall success",
            )
            expected_per_task = {
                shard_protocol[str(task)]["language"]: per_task_successes[task] / 50
                for task in expected_tasks
            }
            require_equal(capability.get("per_task"), expected_per_task, f"{label} per-task rates")

    expected_keys = {(suite, task) for suite in SUITES for task in range(10)}
    require_equal(set(task_rows), expected_keys, f"{architecture} seed {seed} task cells")
    require_equal(len(seen_episodes), 2000, f"{architecture} seed {seed} episode coverage")
    for key, outcomes in task_rows.items():
        require_equal(len(outcomes), 50, f"{architecture} seed {seed} {key} trials")
    require_equal(len(task_protocol), 40, f"{architecture} seed {seed} task protocol coverage")

    task_rates = {
        f"{suite}:{task}": sum(outcomes) / len(outcomes)
        for (suite, task), outcomes in sorted(task_rows.items())
    }
    suite_rates = {
        suite: sum(task_rates[f"{suite}:{task}"] for task in range(10)) / 10
        for suite in SUITES
    }
    return {
        "macro_task_success": sum(task_rates.values()) / len(task_rates),
        "suite_macro_success": suite_rates,
        "task_success": task_rates,
        "tasks_at_least_0p50": sum(rate >= 0.50 for rate in task_rates.values()),
        "successes": sum(sum(outcomes) for outcomes in task_rows.values()),
        "trials": sum(len(outcomes) for outcomes in task_rows.values()),
        "checkpoint_sha256": canonical_metadata["checkpoint_sha256"],
        "model_metadata_sha256": canonical_metadata_sha,
        "training_environment": canonical_metadata["training_environment"],
        "evaluation_environment": canonical_runtime,
        "evaluation_gpu": canonical_gpu,
        "task_protocol": task_protocol,
        "source_files": source_files,
    }


def validate_cache_provenance(results_dir: Path) -> dict[str, Any]:
    validated: dict[str, Any] = {}
    for suite, expected in CACHES.items():
        path = results_dir / f"cache_provenance_{suite}.json"
        result, result_sha = load_json_with_sha(path)
        label = path.as_posix()
        require_equal(result.get("schema"), "xvla-cache-provenance-v1", f"{label} schema")
        require_equal(result.get("suite"), suite, f"{label} suite")
        require_equal(result.get("verified"), True, f"{label} verified")
        cache = result.get("cache", {})
        for key in ("path", "sha256", "frames"):
            expected_key = "frame_count" if key == "frames" else key
            require_equal(cache.get(key), expected[expected_key], f"{label} cache {key}")
        metadata = result.get("metadata", {})
        for key in ("repository", "revision", "sha256"):
            expected_key = "metadata_sha256" if key == "sha256" else key
            require_equal(metadata.get(key), expected[expected_key], f"{label} metadata {key}")
        comparison = result.get("comparison", {})
        require_equal(
            comparison.get("cached_frames_compared"),
            expected["frame_count"],
            f"{label} compared frames",
        )
        mismatch_counts = comparison.get("mismatch_counts", {})
        require_equal(
            set(mismatch_counts), PROVENANCE_MISMATCH_FIELDS, f"{label} mismatch fields"
        )
        if any(value != 0 for value in mismatch_counts.values()):
            raise RuntimeError(f"{label} contains cache/source mismatches")
        require_equal(
            result.get("implementation", {}).get("verifier_source_sha256"),
            PROVENANCE_VERIFIER_SHA256,
            f"{label} verifier source",
        )
        require_equal(
            result.get("metadata", {}).get("language_set_matches_official"),
            True,
            f"{label} language-set identity",
        )
        source = result.get("source", {})
        require_equal(source.get("repository"), expected["repository"], f"{label} source repo")
        require_equal(source.get("revision"), expected["revision"], f"{label} source revision")
        require_equal(
            source.get("content_hashes_match"),
            True,
            f"{label} content hashes",
        )
        require_equal(
            cache.get("canonical_content_sha256"),
            source.get("canonical_content_sha256"),
            f"{label} canonical content identity",
        )
        validated[suite] = {
            "result_sha256": result_sha,
            "cache_sha256": cache["sha256"],
            "verified": True,
        }
    return validated


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    require_equal(args.artifacts_dir.as_posix(), "artifacts", "hardened artifacts directory")
    summary_start_sha256 = file_sha256(Path(__file__))
    manifest_start_sha256 = file_sha256(MANIFEST_PATH)
    require_equal(manifest_start_sha256, MANIFEST_SHA256, "live frozen manifest")
    with MANIFEST_PATH.open() as handle:
        live_manifest = json.load(handle)
    validate_cache_table(live_manifest, "live manifest")
    cache_provenance = validate_cache_provenance(args.results_dir)

    live_artifacts = {
        architecture: {
            str(seed): load_live_artifact_identity(
                args.artifacts_dir, architecture, seed, live_manifest
            )
            for seed in SEEDS
        }
        for architecture in ARCHITECTURES
    }

    by_architecture = {
        architecture: {
            str(seed): load_checkpoint_results(
                args.results_dir,
                architecture,
                seed,
                live_manifest,
                live_artifacts[architecture][str(seed)]["checkpoint_sha256"],
                live_artifacts[architecture][str(seed)]["metadata"],
                live_artifacts[architecture][str(seed)]["metadata_sha256"],
            )
            for seed in SEEDS
        }
        for architecture in ARCHITECTURES
    }

    reference_task_protocol = by_architecture["chi"]["0"]["task_protocol"]
    reference_runtime = by_architecture["chi"]["0"]["evaluation_environment"]
    reference_gpu = by_architecture["chi"]["0"]["evaluation_gpu"]
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            row = by_architecture[architecture][str(seed)]
            require_equal(
                row["task_protocol"], reference_task_protocol, f"{architecture} seed {seed} task assets"
            )
            require_equal(
                row["evaluation_environment"], reference_runtime, f"{architecture} seed {seed} runtime"
            )
            require_equal(
                row["evaluation_gpu"], reference_gpu, f"{architecture} seed {seed} GPU"
            )

    aggregate: dict[str, Any] = {}
    for architecture, checkpoint_rows in by_architecture.items():
        seed_macros = [row["macro_task_success"] for row in checkpoint_rows.values()]
        aggregate[architecture] = {
            "seed_macro_task_success": seed_macros,
            "seed_mean_macro_task_success": mean(seed_macros),
            "seed_sample_std_macro_task_success": statistics.stdev(seed_macros),
            "suite_seed_mean_macro_success": {
                suite: mean([row["suite_macro_success"][suite] for row in checkpoint_rows.values()])
                for suite in SUITES
            },
            "task_seed_mean_success": {
                f"{suite}:{task}": mean(
                    [row["task_success"][f"{suite}:{task}"] for row in checkpoint_rows.values()]
                )
                for suite in SUITES
                for task in range(10)
            },
        }
        aggregate[architecture]["tasks_seed_mean_at_least_0p50"] = sum(
            rate >= 0.50 for rate in aggregate[architecture]["task_seed_mean_success"].values()
        )

    chi = aggregate["chi"]
    conventional = aggregate["conventional"]
    capability_gates = {
        "chi_seed_mean_macro_at_least_0p70": chi["seed_mean_macro_task_success"] >= 0.70,
        "every_chi_suite_at_least_0p50": all(
            rate >= 0.50 for rate in chi["suite_seed_mean_macro_success"].values()
        ),
        "at_least_32_of_40_tasks_at_0p50": chi["tasks_seed_mean_at_least_0p50"] >= 32,
        "every_chi_seed_at_least_0p60": all(
            row["macro_task_success"] >= 0.60 for row in by_architecture["chi"].values()
        ),
    }
    capability_gates["primary_generalist_capability_pass"] = all(capability_gates.values())
    conversion_gap = (
        chi["seed_mean_macro_task_success"]
        - conventional["seed_mean_macro_task_success"]
    )
    secondary_gates = {"chi_within_0p05_of_conventional": conversion_gap >= -0.05}

    summary_end_sha256 = file_sha256(Path(__file__))
    manifest_end_sha256 = file_sha256(MANIFEST_PATH)
    require_equal(summary_end_sha256, summary_start_sha256, "summary source start/end")
    require_equal(manifest_end_sha256, manifest_start_sha256, "manifest start/end")
    public_artifact_identity: dict[str, dict[str, Any]] = {}
    for architecture in ARCHITECTURES:
        public_artifact_identity[architecture] = {}
        for seed in SEEDS:
            artifact = live_artifacts[architecture][str(seed)]
            checkpoint_end_sha256 = file_sha256(artifact["checkpoint_path"])
            metadata_end_sha256 = file_sha256(artifact["metadata_path"])
            require_equal(
                checkpoint_end_sha256,
                artifact["checkpoint_sha256"],
                f"{architecture} seed {seed} checkpoint start/end",
            )
            require_equal(
                metadata_end_sha256,
                artifact["metadata_sha256"],
                f"{architecture} seed {seed} metadata start/end",
            )
            public_artifact_identity[architecture][str(seed)] = {
                "checkpoint_path": artifact["checkpoint_path"].as_posix(),
                "checkpoint_start_sha256": artifact["checkpoint_sha256"],
                "checkpoint_end_sha256": checkpoint_end_sha256,
                "checkpoint_archive_index_sha256": artifact[
                    "checkpoint_archive_index_sha256"
                ],
                "metadata_path": artifact["metadata_path"].as_posix(),
                "metadata_start_sha256": artifact["metadata_sha256"],
                "metadata_end_sha256": metadata_end_sha256,
            }
    output = {
        "scope": {
            "architectures": list(ARCHITECTURES),
            "seeds": list(SEEDS),
            "suites": list(SUITES),
            "tasks_per_suite": 10,
            "episodes_per_task": 50,
            "total_trials": 12000,
            "macro_averaging": "task first, then suite and checkpoint",
            "claim_boundary": (
                "In-domain capability of one jointly trained checkpoint per seed across "
                "all 40 tasks. This is not zero-shot suite or task generalization."
            ),
        },
        "identity": {
            "manifest_path": MANIFEST_PATH.as_posix(),
            "manifest_start_sha256": manifest_start_sha256,
            "manifest_end_sha256": manifest_end_sha256,
            "evaluator_sha256": EVALUATOR_SHA256,
            "trainer_sha256": TRAINER_SHA256,
            "summary_start_sha256": summary_start_sha256,
            "summary_end_sha256": summary_end_sha256,
            "evaluation_environment": reference_runtime,
            "evaluation_gpu": reference_gpu,
            "task_protocol": reference_task_protocol,
            "cache_provenance": cache_provenance,
            "live_artifacts": public_artifact_identity,
        },
        "by_architecture_and_seed": by_architecture,
        "aggregate": aggregate,
        "chi_minus_conventional_macro": conversion_gap,
        "capability_gates": capability_gates,
        "secondary_gates": secondary_gates,
        "all_preregistered_gates_pass": None,
        "specialist_retention_gate": {
            "status": "pending_locked_specialist_results",
            "reason": (
                "Requires locked, same-runtime seed-0 specialist results for all four "
                "suites and will be added without changing the generalist gates."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
