#!/usr/bin/env python3
"""Native Athena evaluation and profiling for the χ-VLA LIBERO checkpoints.

This runner deliberately has no Modal dependency. It reconstructs the exact model,
vocabulary, normalization statistics, observation transform, and canonical LIBERO
reset protocol used by ``modal_app.libero_rollout_head``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
from PIL import Image

from athena.libero_dataset_metadata import load_dataset_task_languages
from xvla.models.vla import ChiVLA, VLAConfig


FROZEN_GENERALIST_MANIFEST_SHA256 = (
    "d0183b465c4d687a4b31f1fc2ca8c75a35786e164cd07f638ca47e347e21f4fd"
)
FROZEN_GENERALIST_SUITES = (
    "libero_object",
    "libero_spatial",
    "libero_goal",
    "libero_10",
)
FROZEN_GENERALIST_SUITE_OFFSETS = {
    "libero_object": 0,
    "libero_spatial": 10,
    "libero_goal": 20,
    "libero_10": 30,
}
FROZEN_GENERALIST_EVALUATION_GPU_FAMILY = "a6000"
FROZEN_GENERALIST_EVALUATION_GPU_NAME = "NVIDIA RTX A6000"
FROZEN_GENERALIST_CACHES = {
    "libero_object": {
        "path": "artifacts/libero_frames_100000_64.pkl",
        "repository": "lerobot/libero_object_image",
        "sha256": "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662",
        "frame_count": 66984,
        "sample_count": 63352,
        "revision": "e1e080d7df1d0a359dff5c86c222e047549f447f",
        "metadata_sha256": "34caee9641ae50bb4e077de306a7d0031753757882da8b1f117e7ea36a486b42",
    },
    "libero_spatial": {
        "path": "artifacts/libero_spatial_frames_100000_64.pkl",
        "repository": "lerobot/libero_spatial_image",
        "sha256": "662234d52afff37d55b4b36377cc89151208fc0392147197e2cbec38a49b3fcb",
        "frame_count": 52970,
        "sample_count": 49514,
        "revision": "d86c0b94922572b3b657e1d1a3d01f0952ddeb46",
        "metadata_sha256": "90ef6a4278e2f2307d64f1649ecd4f84ee481cff5007f15d8683f1f9cc917af3",
    },
    "libero_goal": {
        "path": "artifacts/libero_goal_frames_100000_64.pkl",
        "repository": "lerobot/libero_goal_image",
        "sha256": "f0d3ff502f797a0d6e2b611f5be806b2660afcb9d78c883610486f5138a50550",
        "frame_count": 52042,
        "sample_count": 48618,
        "revision": "91a97115558b5b611200a432d9c82e4f30991b60",
        "metadata_sha256": "80f750568ef991a406e9e1ffcdfe638eaa2dcf452c5e69c1b911c7dae0de8986",
    },
    "libero_10": {
        "path": "artifacts/libero_10_frames_100000_64.pkl",
        "repository": "lerobot/libero_10_image",
        "sha256": "571691dac732f9aa6c0837dbd3505e497b8dd18ea247b35ae10038c476eb44d0",
        "frame_count": 100000,
        "sample_count": 97008,
        "revision": "7e324b526699f444044952c82ce3f438e8d300d0",
        "metadata_sha256": "10bb12686977333e729ef5c444cb9eb06377fff5b08a6f87f422dc6bf232205e",
    },
}

# Conservative closure of repository sources imported by the four-suite model.
# The evaluator and trainer verify this table both before and after their work.
FROZEN_GENERALIST_IMPORTED_SOURCES = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "smoke",
            "profile",
            "capability",
            "ensemble_capability",
            "cp_pruning",
            "offline_diagnostic",
            "causal",
            "visual_subspace",
            "exact_attention",
            "surgery",
        ),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--ensemble-checkpoint",
        type=Path,
        action="append",
        default=[],
        help="Additional checkpoints for ensemble_capability mode.",
    )
    parser.add_argument(
        "--ensemble-reduction", choices=("mean", "median"), default="mean"
    )
    parser.add_argument("--prune-fraction", type=float, default=0.5)
    parser.add_argument(
        "--prune-strategy", choices=("magnitude", "random"), default="magnitude"
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--model-metadata",
        type=Path,
        help=(
            "Optional training metadata containing a fixed vocabulary and pooled "
            "normalization statistics, used for multi-suite checkpoints."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        choices=("chi", "conventional", "chi_rms", "conventional_rational"),
        default="chi",
    )
    parser.add_argument("--vision-encoder", choices=("vit", "conv"), default="vit")
    parser.add_argument(
        "--suite",
        choices=("libero_object", "libero_spatial", "libero_goal", "libero_10"),
        default="libero_object",
    )
    parser.add_argument(
        "--training-suite",
        choices=("libero_object", "libero_spatial", "libero_goal", "libero_10"),
        default="libero_object",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int, default=10)
    parser.add_argument("--eps-per-task", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--exec-h", type=int, default=8)
    parser.add_argument("--profile-iters", type=int, default=200)
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
        help=(
            "PyTorch float32 matmul precision. Historical Modal evaluations used the "
            "PyTorch default, highest, while the initial Athena replication used high."
        ),
    )
    parser.add_argument("--causal-steps", type=int, default=80)
    parser.add_argument("--block-index", type=int, default=6)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--gram-samples", type=int, default=1024)
    parser.add_argument("--offline-eval-samples", type=int, default=1024)
    parser.add_argument("--gram-task-start", type=int, default=0)
    parser.add_argument("--gram-task-end", type=int, default=10)
    parser.add_argument(
        "--gram-action-group",
        choices=("gripper", "all_balanced"),
        default="gripper",
    )
    parser.add_argument("--gram-probes", type=int, default=4)
    parser.add_argument("--random-controls", type=int, default=1)
    parser.add_argument(
        "--activation-energy-control",
        action="store_true",
        help=(
            "Add a same-rank projector onto the top uncentered visual-activation "
            "energy directions as a gradient-free compression baseline."
        ),
    )
    parser.add_argument(
        "--subspace-rollout-conditions",
        default="",
        help=(
            "Optional comma-separated subset of visual-subspace conditions to "
            "evaluate closed loop. Empty evaluates every constructed condition."
        ),
    )
    parser.add_argument("--subspace-offline-only", action="store_true")
    parser.add_argument("--offline-samples", type=int, default=4096)
    parser.add_argument(
        "--surgery-conditions",
        default=(
            "baseline,keep_top,keep_random_normmatched,"
            "remove_top,remove_random_normmatched"
        ),
    )
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    return parser.parse_args()


def load_suite(name: str = "libero_object"):
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[name]()


def task_languages(suite) -> dict[int, str]:
    return {idx: str(suite.get_task(idx).language) for idx in range(suite.n_tasks)}


def build_vocab(tasks: dict[int, str]) -> tuple[dict[str, int], Any]:
    words: set[str] = set()
    for language in tasks.values():
        words.update(language.lower().replace(".", "").split())
    vocab = {"<pad>": 0, "<bos>": 1}
    for word in sorted(words):
        vocab[word] = len(vocab)

    return vocab, build_encoder(vocab)


def build_encoder(vocab: dict[str, int]):
    """Build the fixed whitespace-token encoder used by training and evaluation."""

    def encode(text: str, length: int = 32) -> list[int]:
        ids = [1] + [vocab.get(word, 0) for word in text.lower().replace(".", "").split()]
        return (ids[:length] + [0] * max(0, length - len(ids)))[:length]

    return encode


def load_model_metadata(path: Path) -> tuple[dict[str, int], dict[str, np.ndarray], dict[str, Any]]:
    with path.open() as handle:
        metadata = json.load(handle)
    vocab = {str(token): int(index) for token, index in metadata["vocab"].items()}
    normalization = metadata["normalization"]
    stats = {
        "action_mean": np.asarray(normalization["action_mean"], dtype=np.float32),
        "action_std": np.asarray(normalization["action_std"], dtype=np.float32),
        "state_mean": np.asarray(normalization["state_mean"], dtype=np.float32),
        "state_std": np.asarray(normalization["state_std"], dtype=np.float32),
    }
    return vocab, stats, metadata


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_bundle_sha256(sources: dict[str, str]) -> str:
    payload = "".join(
        f"{path}\0{digest}\n" for path, digest in sorted(sources.items())
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def generalist_imported_source_snapshot() -> dict[str, str]:
    repository_root = Path(__file__).resolve().parents[1]
    return {
        path: file_sha256(repository_root / path)
        for path in FROZEN_GENERALIST_IMPORTED_SOURCES
    }


def validate_generalist_imported_sources(sources: dict[str, str]) -> None:
    if sources != FROZEN_GENERALIST_IMPORTED_SOURCES:
        changed = {
            path: {
                "expected": FROZEN_GENERALIST_IMPORTED_SOURCES.get(path),
                "observed": sources.get(path),
            }
            for path in sorted(
                set(FROZEN_GENERALIST_IMPORTED_SOURCES) | set(sources)
            )
            if FROZEN_GENERALIST_IMPORTED_SOURCES.get(path) != sources.get(path)
        }
        raise RuntimeError(f"Frozen generalist source mismatch: {changed}")


def array_sha256(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_model_metadata(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    vocab: dict[str, int],
    fixed_stats: dict[str, np.ndarray],
    evaluation_stats: dict[str, Any],
    cache_task_metadata: dict[str, Any],
) -> None:
    """Reject stale or mismatched metadata before loading a generalist checkpoint."""
    errors = []
    expected_suites = FROZEN_GENERALIST_SUITES
    if metadata.get("format") != "xvla_multisuite_checkpoint_v1":
        errors.append("unsupported metadata format")
    if tuple(metadata.get("training_suites", ())) != expected_suites:
        errors.append("training suite order is not the frozen four-suite order")
    if args.suite not in metadata.get("source_caches", {}):
        errors.append(f"evaluation suite {args.suite} is absent from source metadata")
    if metadata.get("architecture") != args.architecture:
        errors.append("architecture does not match --architecture")
    if metadata.get("vision_encoder") != args.vision_encoder:
        errors.append("vision encoder does not match --vision-encoder")
    if int(metadata.get("seed", -1)) != args.seed:
        errors.append("training seed does not match --seed")
    if Path(str(metadata.get("checkpoint", ""))).name != args.checkpoint.name:
        errors.append("checkpoint filename does not match metadata")
    expected_checkpoint_hash = metadata.get("checkpoint_sha256")
    if not expected_checkpoint_hash:
        errors.append("checkpoint SHA-256 is missing")
    elif file_sha256(args.checkpoint) != expected_checkpoint_hash:
        errors.append("checkpoint SHA-256 does not match metadata")
    if int(metadata.get("res", -1)) != args.res:
        errors.append("image resolution does not match --res")
    if int(metadata.get("horizon", -1)) != args.horizon:
        errors.append("action horizon does not match --horizon")
    if int(metadata.get("state_dim", -1)) != evaluation_stats["state_dim"]:
        errors.append("state dimension does not match the evaluation cache")
    if int(metadata.get("action_dim", -1)) != evaluation_stats["action_dim"]:
        errors.append("action dimension does not match the evaluation cache")
    if int(metadata.get("task_count", -1)) != 40:
        errors.append("metadata does not contain exactly 40 tasks")
    if metadata.get("manifest_sha256") != FROZEN_GENERALIST_MANIFEST_SHA256:
        errors.append("metadata does not name the frozen generalist manifest")
    if metadata.get("recipe_version") != "multisuite_balanced_v1":
        errors.append("training recipe version is not multisuite_balanced_v1")
    expected_steps = 10 if "smoke" in args.checkpoint.stem else 160000
    expected_recipe = {
        "steps": expected_steps,
        "batch_size": 256,
        "suite_batch_size": 64,
        "lr": 8e-4,
        "ema_decay": 0.999,
        "res": 64,
        "horizon": 8,
        "state_dim": 8,
        "action_dim": 7,
    }
    for key, expected in expected_recipe.items():
        if metadata.get(key) != expected:
            errors.append(
                f"training metadata {key}={metadata.get(key)!r}, expected {expected!r}"
            )
    if vocab.get("<pad>") != 0 or vocab.get("<bos>") != 1:
        errors.append("reserved vocabulary identifiers are invalid")
    if sorted(vocab.values()) != list(range(len(vocab))):
        errors.append("vocabulary identifiers are not contiguous")
    if metadata.get("suite_task_offsets") != FROZEN_GENERALIST_SUITE_OFFSETS:
        errors.append("suite task offsets do not match the frozen offsets")
    if len(metadata.get("task_languages", {})) != 40:
        errors.append("task-language table does not contain 40 entries")
    if fixed_stats["state_mean"].shape != (evaluation_stats["state_dim"],):
        errors.append("state normalization has the wrong shape")
    if fixed_stats["state_std"].shape != (evaluation_stats["state_dim"],):
        errors.append("state scale has the wrong shape")
    if fixed_stats["action_mean"].shape != (evaluation_stats["action_dim"],):
        errors.append("action normalization has the wrong shape")
    if fixed_stats["action_std"].shape != (evaluation_stats["action_dim"],):
        errors.append("action scale has the wrong shape")
    if not all(np.isfinite(value).all() for value in fixed_stats.values()):
        errors.append("normalization contains non-finite values")
    if not all((fixed_stats[key] > 0).all() for key in ("state_std", "action_std")):
        errors.append("normalization scales must be positive")

    source_caches = metadata.get("source_caches", {})
    if set(source_caches) != set(expected_suites):
        errors.append("source-cache table does not contain exactly the four suites")
    for suite_name, expected in FROZEN_GENERALIST_CACHES.items():
        source = source_caches.get(suite_name, {})
        for key in ("path", "sha256", "frame_count", "sample_count"):
            if source.get(key) != expected[key]:
                errors.append(
                    f"{suite_name} source {key}={source.get(key)!r}, "
                    f"expected {expected[key]!r}"
                )
        task_metadata = source.get("task_metadata", {})
        for source_key, expected_key in (
            ("repository", "repository"),
            ("revision", "revision"),
            ("metadata_sha256", "metadata_sha256"),
        ):
            if task_metadata.get(source_key) != expected[expected_key]:
                errors.append(
                    f"{suite_name} task metadata {source_key} is not frozen"
                )
    expected_evaluation_cache = FROZEN_GENERALIST_CACHES[args.suite]
    if args.cache.as_posix() != expected_evaluation_cache["path"]:
        errors.append("evaluation cache path is not the frozen suite cache")
    elif file_sha256(args.cache) != expected_evaluation_cache["sha256"]:
        errors.append("evaluation cache SHA-256 does not match the frozen cache")
    if cache_task_metadata != source_caches.get(args.suite, {}).get("task_metadata"):
        errors.append("live task metadata does not match checkpoint metadata")

    manifest_path_value = metadata.get("source_manifest")
    if not manifest_path_value:
        errors.append("source manifest path is missing")
    else:
        manifest_path = Path(str(manifest_path_value))
        if manifest_path.as_posix() != "artifacts/libero_all_manifest.json":
            errors.append("source manifest path is not the frozen path")
        elif not manifest_path.is_file():
            errors.append("source manifest file is missing")
        else:
            live_manifest_sha = file_sha256(manifest_path)
            if live_manifest_sha != FROZEN_GENERALIST_MANIFEST_SHA256:
                errors.append("live manifest SHA-256 is not frozen")
            else:
                with manifest_path.open() as handle:
                    live_manifest = json.load(handle)
                for key, value in live_manifest.items():
                    if key == "format":
                        continue
                    if metadata.get(key) != value:
                        errors.append(
                            f"checkpoint metadata differs from live manifest at {key}"
                        )
    if errors:
        raise ValueError("Invalid model metadata: " + "; ".join(errors))


def load_cache_profile_sample(cache_path: Path, horizon: int) -> dict[str, Any]:
    """Load one valid sample without materializing all overlapping cache windows."""
    with cache_path.open("rb") as handle:
        frames = pickle.load(handle)
    episodes: dict[int, list[Any]] = defaultdict(list)
    for frame in frames:
        episodes[int(frame[0])].append(frame)
    for episode_frames in episodes.values():
        episode_frames.sort(key=lambda item: int(item[1]))
        if len(episode_frames) <= horizon:
            continue
        frame = episode_frames[0]
        action_chunk = np.stack(
            [episode_frames[offset][4] for offset in range(horizon)]
        ).astype(np.float32)
        state = np.asarray(frame[3], dtype=np.float32)
        return {
            "frame_count": len(frames),
            "sample_count": None,
            "state_dim": int(state.shape[-1]),
            "action_dim": int(action_chunk.shape[-1]),
            "first_sample": {
                "image": np.asarray(frame[2], dtype=np.uint8),
                "task": int(frame[5]),
                "state": state,
                "actions": action_chunk,
            },
        }
    raise RuntimeError(f"No {horizon}-step sample could be built from {cache_path}")


def load_cache_statistics(cache_path: Path, horizon: int) -> dict[str, Any]:
    with cache_path.open("rb") as handle:
        frames = pickle.load(handle)
    episodes: dict[int, list[Any]] = defaultdict(list)
    for frame in frames:
        episodes[int(frame[0])].append(frame)

    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    first_sample = None
    for episode_frames in episodes.values():
        episode_frames.sort(key=lambda item: int(item[1]))
        for index in range(len(episode_frames) - horizon):
            frame = episode_frames[index]
            action_chunk = np.stack(
                [episode_frames[index + offset][4] for offset in range(horizon)]
            ).astype(np.float32)
            state = np.asarray(frame[3], dtype=np.float32)
            if first_sample is None:
                first_sample = {
                    "image": np.asarray(frame[2], dtype=np.uint8),
                    "task": int(frame[5]),
                    "state": state,
                    "actions": action_chunk,
                }
            actions.append(action_chunk)
            states.append(state)

    if first_sample is None:
        raise RuntimeError(f"No {horizon}-step samples could be built from {cache_path}")
    action_array = np.stack(actions)
    state_array = np.stack(states)
    return {
        "frame_count": len(frames),
        "sample_count": len(actions),
        "action_mean": action_array.mean((0, 1)),
        "action_std": action_array.std((0, 1)) + 1e-6,
        "state_mean": state_array.mean(0),
        "state_std": state_array.std(0) + 1e-6,
        "state_dim": int(state_array.shape[-1]),
        "action_dim": int(action_array.shape[-1]),
        "first_sample": first_sample,
    }


def make_config(
    architecture: str,
    vocab_size: int,
    state_dim: int,
    action_dim: int,
    res: int,
    horizon: int,
    vision_encoder: str = "vit",
) -> VLAConfig:
    kwargs: dict[str, Any] = {
        "image_size": res,
        "patch_size": 8,
        "vit_dim": 192,
        "vit_layers": 4,
        "vit_heads": 8,
        "vocab_size": vocab_size,
        "max_instr_len": 32,
        "state_dim": state_dim,
        "n_embodiments": 1,
        "dim": 384,
        "n_layers": 8,
        "n_heads": 12,
        "action_horizon": horizon,
        "action_dim": action_dim,
        "action_head": "linear",
        "vision_encoder": vision_encoder,
    }
    if architecture in ("conventional", "conventional_rational"):
        kwargs.update(
            attn="softmax",
            ffn="swiglu",
            norm=("rational" if architecture == "conventional_rational" else "per_token"),
            qk_norm="none",
            ffn_rank=1408,
            vit_ffn_rank=704,
        )
    elif architecture == "chi_rms":
        kwargs.update(attn="bilinear", ffn="bilinear", norm="per_token", qk_norm="per_token")
    else:
        kwargs.update(attn="bilinear", ffn="bilinear", norm="rational", qk_norm="rational")
    return VLAConfig(**kwargs)


def load_model(
    args: argparse.Namespace,
    vocab_size: int,
    stats: dict[str, Any],
    checkpoint: Path | None = None,
) -> ChiVLA:
    config = make_config(
        args.architecture,
        vocab_size,
        stats["state_dim"],
        stats["action_dim"],
        args.res,
        args.horizon,
        vision_encoder=args.vision_encoder,
    )
    model = ChiVLA(config).cuda()
    checkpoint_path = args.checkpoint if checkpoint is None else checkpoint
    state_dict = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


class EnsemblePolicy(torch.nn.Module):
    """Average or coordinate-median action chunks from fixed trained checkpoints."""

    def __init__(self, models: list[ChiVLA], reduction: str) -> None:
        super().__init__()
        if len(models) < 2:
            raise ValueError("An ensemble requires at least two checkpoints")
        self.models = torch.nn.ModuleList(models)
        self.reduction = reduction
        self.cfg = models[0].cfg

    def forward(self, *inputs):
        predictions = [model(*inputs)[0] for model in self.models]
        stacked = torch.stack(predictions, dim=0)
        if self.reduction == "mean":
            combined = stacked.mean(dim=0)
        elif self.reduction == "median":
            combined = stacked.median(dim=0).values
        else:
            raise ValueError(f"Unsupported ensemble reduction: {self.reduction}")
        return combined, {"ensemble_members": len(self.models)}

    def num_params(self) -> int:
        return sum(model.num_params() for model in self.models)


@torch.no_grad()
def apply_cp_term_pruning(
    model: ChiVLA,
    fraction: float,
    strategy: str,
    seed: int,
) -> dict[str, Any]:
    """Remove complete rank-one terms from every bilinear FFN.

    The score is the product of the homogeneous left/right row norms and the
    down-projection column norm. It is invariant to the usual CP component
    rescaling gauge. Zeroing a down column removes that CP term exactly.
    """
    from xvla.nn.bilinear import BilinearFFN

    if not 0.0 < fraction < 1.0:
        raise ValueError("--prune-fraction must be strictly between 0 and 1")
    generator = np.random.default_rng(seed)
    modules: list[dict[str, Any]] = []
    total_terms = 0
    removed_terms = 0
    total_component_parameters = 0
    removed_component_parameters = 0

    for name, module in model.named_modules():
        if not isinstance(module, BilinearFFN):
            continue
        left = torch.cat([module.left.weight, module.left.bias[:, None]], dim=1)
        right = torch.cat([module.right.weight, module.right.bias[:, None]], dim=1)
        scores = (
            torch.linalg.vector_norm(left, dim=1)
            * torch.linalg.vector_norm(right, dim=1)
            * torch.linalg.vector_norm(module.down.weight, dim=0)
        )
        count = max(1, min(module.rank - 1, int(round(fraction * module.rank))))
        if strategy == "magnitude":
            indices = torch.argsort(scores)[:count]
        elif strategy == "random":
            selected = generator.choice(module.rank, size=count, replace=False)
            indices = torch.as_tensor(selected, dtype=torch.long, device=scores.device)
        else:
            raise ValueError(f"Unsupported pruning strategy: {strategy}")

        selected_scores = scores[indices]
        module.down.weight[:, indices] = 0
        component_parameters = module.dim + 1 + module.dim + 1 + module.out_dim
        total_terms += module.rank
        removed_terms += count
        total_component_parameters += module.rank * component_parameters
        removed_component_parameters += count * component_parameters
        modules.append(
            {
                "name": name,
                "rank": module.rank,
                "removed_terms": count,
                "component_parameters_per_term": component_parameters,
                "score_min": float(scores.min().item()),
                "score_median": float(scores.median().item()),
                "score_max": float(scores.max().item()),
                "selected_score_mean": float(selected_scores.mean().item()),
                "selected_score_max": float(selected_scores.max().item()),
            }
        )

    if not modules:
        raise RuntimeError("No BilinearFFN modules were found for CP-term pruning")
    return {
        "strategy": strategy,
        "requested_fraction": fraction,
        "modules_pruned": len(modules),
        "total_cp_terms": total_terms,
        "removed_cp_terms": removed_terms,
        "removed_cp_term_fraction": removed_terms / total_terms,
        "total_component_parameters": total_component_parameters,
        "removed_component_parameters": removed_component_parameters,
        "removed_component_parameter_fraction": (
            removed_component_parameters / total_component_parameters
        ),
        "whole_model_parameter_equivalent_fraction": (
            removed_component_parameters / model.num_params()
        ),
        "mask_seed": seed if strategy == "random" else None,
        "modules": modules,
        "scope": (
            "Complete rank-one terms are removed from every BilinearFFN present in the model. "
            + (
                "For the ViT policy this covers the vision and joint towers. "
                if model.cfg.vision_encoder == "vit"
                else "For the convolutional policy this covers the joint tower only. "
            )
            + "Parameter counts are compact-model equivalents. The serialized checkpoint is "
            "not physically compacted for this evaluation."
        ),
    }


def official_init_states(suite, task_index: int):
    """Load trusted packaged LIBERO states under PyTorch 2.6 without fallback."""
    original_torch_load = torch.load

    def trusted_load(*load_args, **load_kwargs):
        load_kwargs["weights_only"] = False
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = trusted_load
    try:
        states = suite.get_task_init_states(task_index)
    except Exception as exc:
        raise RuntimeError(
            f"Official LIBERO initial states failed to load for task {task_index}"
        ) from exc
    finally:
        torch.load = original_torch_load
    if states is None or len(states) == 0:
        raise RuntimeError(f"Official LIBERO initial states are empty for task {task_index}")
    return states


def tensorize_sample(
    sample: dict[str, Any],
    encode,
    tasks: dict[int, str],
    stats: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image = (
        torch.from_numpy(np.asarray(sample["image"]))
        .permute(2, 0, 1)
        .float()
        .div(255)
        .unsqueeze(0)
        .cuda()
    )
    instruction = torch.tensor([encode(tasks[sample["task"]])], dtype=torch.long, device="cuda")
    state = torch.tensor(
        (sample["state"] - stats["state_mean"]) / stats["state_std"],
        dtype=torch.float32,
        device="cuda",
    ).unsqueeze(0)
    embodiment = torch.zeros(1, dtype=torch.long, device="cuda")
    return image, instruction, state, embodiment


@torch.inference_mode()
def profile_model(
    model: Any,
    sample_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    iterations: int,
) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    for _ in range(20):
        model(*sample_tensors)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        model(*sample_tensors)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "parameters": model.num_params(),
        "parameters_millions": round(model.num_params() / 1e6, 6),
        "latency_ms_batch1": 1000.0 * elapsed / iterations,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "profile_iterations": iterations,
        "gpu": torch.cuda.get_device_name(0),
    }


def build_robot_state(obs: dict[str, Any], state_dim: int) -> np.ndarray:
    from robosuite.utils.transform_utils import quat2axisangle

    state = np.concatenate(
        [
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]
    ).astype(np.float32)
    if len(state) < state_dim:
        state = np.pad(state, (0, state_dim - len(state)))
    return state[:state_dim]


def forward_from_visual_tokens(
    model: ChiVLA,
    visual_tokens: torch.Tensor,
    instruction: torch.Tensor,
    state: torch.Tensor,
    embodiment: torch.Tensor,
) -> torch.Tensor:
    """Run the deployed linear policy from an intervened visual-token bond."""
    from xvla.nn.attention import causal_mask

    if model.cfg.action_head != "linear":
        raise ValueError("Visual-bond intervention currently requires the linear action head")
    batch_size = visual_tokens.shape[0]
    hidden = torch.cat(
        [
            visual_tokens,
            model.bos.expand(batch_size, -1, -1),
            model.tok_emb(instruction),
            model.state_proj(state)[:, None],
            model.embodiment_emb(embodiment)[:, None],
            model.action_queries.expand(batch_size, -1, -1),
        ],
        dim=1,
    )
    hidden = hidden + model.pos_emb[:, : hidden.shape[1]]
    mask = causal_mask(hidden.shape[1], device=hidden.device, dtype=hidden.dtype)
    hidden = model.norm_out(model.backbone(hidden, mask=mask))
    return model.action_head(hidden[:, -model.cfg.action_horizon :])


@torch.inference_mode()
def predict_chunk(
    model: Any,
    obs: dict[str, Any],
    instruction: torch.Tensor,
    stats: dict[str, Any],
    res: int,
    visual_projector: torch.Tensor | None = None,
) -> np.ndarray:
    raw_state = build_robot_state(obs, stats["state_dim"])
    rotated = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    image_array = np.asarray(Image.fromarray(rotated).resize((res, res))).copy()
    image = torch.from_numpy(image_array).permute(2, 0, 1).float().div(255).unsqueeze(0).cuda()
    state = torch.tensor(
        (raw_state - stats["state_mean"]) / stats["state_std"],
        dtype=torch.float32,
        device="cuda",
    ).unsqueeze(0)
    embodiment = torch.zeros(1, dtype=torch.long, device="cuda")
    if visual_projector is None:
        prediction, _ = model(image, instruction, state, embodiment)
    else:
        visual_tokens = model._visual_tokens(image) @ visual_projector.T
        prediction = forward_from_visual_tokens(
            model, visual_tokens, instruction, state, embodiment
        )
    action_mean = torch.tensor(stats["action_mean"], dtype=torch.float32, device="cuda")
    action_std = torch.tensor(stats["action_std"], dtype=torch.float32, device="cuda")
    return (prediction[0] * action_std + action_mean).float().cpu().numpy()


def run_capability(
    args: argparse.Namespace,
    model: Any,
    suite,
    tasks: dict[int, str],
    encode,
    stats: dict[str, Any],
    visual_projector: torch.Tensor | None = None,
) -> dict[str, Any]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_end = min(args.task_end, suite.n_tasks)
    if not 0 <= args.task_start < task_end:
        raise ValueError(f"Invalid task range [{args.task_start}, {task_end})")
    dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    per_task: dict[str, float] = {}
    episode_records: list[dict[str, Any]] = []
    task_protocol: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()

    for task_index in range(args.task_start, task_end):
        task = suite.get_task(task_index)
        bddl_path = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=args.res,
            camera_widths=args.res,
        )
        instruction = torch.tensor([encode(task.language)], dtype=torch.long, device="cuda")
        init_states = official_init_states(suite, task_index)
        task_protocol[str(task_index)] = {
            "language": str(task.language),
            "problem_folder": str(task.problem_folder),
            "bddl_file": str(task.bddl_file),
            "bddl_sha256": file_sha256(Path(bddl_path)),
            "init_state_count": len(init_states),
            "init_states_sha256": array_sha256(init_states),
        }
        successes = 0
        for episode in range(args.eps_per_task):
            env.seed(task_index * 100 + episode)
            obs = env.reset()
            obs = env.set_init_state(init_states[episode % len(init_states)])
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(dummy_action)

            success = False
            steps = 0
            episode_started = time.perf_counter()
            while steps < args.max_steps and not success:
                chunk = predict_chunk(
                    model,
                    obs,
                    instruction,
                    stats,
                    args.res,
                    visual_projector=visual_projector,
                )
                for offset in range(min(args.exec_h, args.horizon)):
                    action = chunk[offset].copy()
                    action[-1] = 1.0 if action[-1] > 0 else -1.0
                    obs, reward, done, _ = env.step(action.tolist())
                    steps += 1
                    success = bool(reward > 0)
                    if done or success or steps >= args.max_steps:
                        break
            successes += int(success)
            record = {
                "task_index": task_index,
                "episode": episode,
                "success": success,
                "steps": steps,
                "elapsed_s": time.perf_counter() - episode_started,
            }
            episode_records.append(record)
            print("EPISODE", json.dumps(record), flush=True)
        env.close()
        per_task[tasks[task_index]] = successes / args.eps_per_task
        print(
            f"TASK {task_index}: {tasks[task_index]}: {successes}/{args.eps_per_task}",
            flush=True,
        )

    total_successes = sum(int(record["success"]) for record in episode_records)
    return {
        "overall": total_successes / len(episode_records),
        "per_task": per_task,
        "episodes": episode_records,
        "successes": total_successes,
        "trials": len(episode_records),
        "task_indices": list(range(args.task_start, task_end)),
        "eps_per_task": args.eps_per_task,
        "max_steps": args.max_steps,
        "num_steps_wait": args.num_steps_wait,
        "exec_h": args.exec_h,
        "canonical_init_states": True,
        "task_protocol": task_protocol,
        "elapsed_s": time.perf_counter() - started,
    }


def readable_object_name(body_name: str) -> str:
    import re

    return re.sub(r"_\d+$", "", body_name).replace("_", " ")


def scene_object_bodies(obs: dict[str, Any]) -> list[str]:
    bodies = []
    for key in obs:
        if not key.endswith("_pos") or key.startswith("robot0"):
            continue
        body = key[: -len("_pos")]
        if "basket" in body or "_to_" in body or "eef" in body:
            continue
        bodies.append(body)
    return sorted(set(bodies))


def bootstrap_mean_ci(values: list[float], seed: int, draws: int = 10000) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(draws, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def run_causal_intervention(
    args: argparse.Namespace,
    model: ChiVLA,
    suite,
    tasks: dict[int, str],
    encode,
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Paired closed-loop language intervention with a trajectory-level proximity outcome."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_end = min(args.task_end, suite.n_tasks)
    dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    def make_environment(task) -> Any:
        bddl_path = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        return OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=args.res,
            camera_widths=args.res,
        )

    def rollout(
        task,
        init_state,
        seed: int,
        instruction_text: str,
        original_body: str,
        named_body: str,
    ) -> dict[str, Any]:
        environment = make_environment(task)
        environment.seed(seed)
        obs = environment.reset()
        obs = environment.set_init_state(init_state)
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = environment.step(dummy_action)
        instruction_ids = torch.tensor(
            [encode(instruction_text)], dtype=torch.long, device="cuda"
        )

        def distances(observation) -> tuple[float, float]:
            eef = np.asarray(observation["robot0_eef_pos"])
            original = np.asarray(observation[f"{original_body}_pos"])
            named = np.asarray(observation[f"{named_body}_pos"])
            return float(np.linalg.norm(eef - original)), float(np.linalg.norm(eef - named))

        original_distances = []
        named_distances = []
        first_original, first_named = distances(obs)
        original_distances.append(first_original)
        named_distances.append(first_named)
        steps = 0
        bddl_success = False
        while steps < args.causal_steps:
            chunk = predict_chunk(model, obs, instruction_ids, stats, args.res)
            for offset in range(min(args.exec_h, args.horizon, args.causal_steps - steps)):
                action = chunk[offset].copy()
                action[-1] = 1.0 if action[-1] > 0 else -1.0
                obs, reward, done, _ = environment.step(action.tolist())
                steps += 1
                bddl_success = bddl_success or bool(reward > 0)
                original_distance, named_distance = distances(obs)
                original_distances.append(original_distance)
                named_distances.append(named_distance)
                if done or steps >= args.causal_steps:
                    break
            if done:
                break
        environment.close()
        return {
            "dist_original_start": original_distances[0],
            "dist_original_end": original_distances[-1],
            "dist_named_start": named_distances[0],
            "dist_named_end": named_distances[-1],
            "preference_margin_start": original_distances[0] - named_distances[0],
            "preference_margin_end": original_distances[-1] - named_distances[-1],
            "steps": steps,
            "bddl_success": bddl_success,
        }

    for task_index in range(args.task_start, task_end):
        task = suite.get_task(task_index)
        init_states = official_init_states(suite, task_index)
        probe_env = make_environment(task)
        for episode in range(args.eps_per_task):
            seed = task_index * 100 + episode
            probe_env.seed(seed)
            probe_obs = probe_env.reset()
            probe_obs = probe_env.set_init_state(init_states[episode % len(init_states)])
            bodies = scene_object_bodies(probe_obs)
            original_body = next(
                (
                    body
                    for body in bodies
                    if readable_object_name(body) in task.language.lower()
                ),
                None,
            )
            distractors = [body for body in bodies if body != original_body]
            if original_body is None or not distractors:
                skipped.append(
                    {
                        "task_index": task_index,
                        "episode": episode,
                        "reason": "target or distractor could not be identified",
                        "bodies": bodies,
                    }
                )
                continue
            named_body = distractors[episode % len(distractors)]
            counterfactual_instruction = (
                f"pick up the {readable_object_name(named_body)} and place it in the basket"
            )
            true_run = rollout(
                task,
                init_states[episode % len(init_states)],
                seed,
                task.language,
                original_body,
                named_body,
            )
            counterfactual_run = rollout(
                task,
                init_states[episode % len(init_states)],
                seed,
                counterfactual_instruction,
                original_body,
                named_body,
            )
            preference_shift = (
                counterfactual_run["preference_margin_end"]
                - true_run["preference_margin_end"]
            )
            row = {
                "task_index": task_index,
                "episode": episode,
                "original_object": original_body,
                "counterfactual_object": named_body,
                "true_instruction": task.language,
                "counterfactual_instruction": counterfactual_instruction,
                "true_run": true_run,
                "counterfactual_run": counterfactual_run,
                "paired_preference_shift": preference_shift,
                "shift_toward_counterfactual_named_object": preference_shift > 0,
                "counterfactual_ended_closer_to_named_object": (
                    counterfactual_run["preference_margin_end"] > 0
                ),
            }
            rows.append(row)
            print("CAUSAL_TRIAL", json.dumps(row), flush=True)
        probe_env.close()

    shifts = [float(row["paired_preference_shift"]) for row in rows]
    positive = [float(row["shift_toward_counterfactual_named_object"]) for row in rows]
    ended_named = [float(row["counterfactual_ended_closer_to_named_object"]) for row in rows]
    true_success = [float(row["true_run"]["bddl_success"]) for row in rows]
    return {
        "n_trials": len(rows),
        "n_skipped": len(skipped),
        "task_indices": list(range(args.task_start, task_end)),
        "eps_per_task": args.eps_per_task,
        "causal_steps": args.causal_steps,
        "mean_paired_preference_shift_m": float(np.mean(shifts)) if shifts else None,
        "mean_paired_preference_shift_95pct_bootstrap_ci_m": bootstrap_mean_ci(
            shifts, args.seed
        ),
        "fraction_shift_toward_counterfactual_named_object": (
            float(np.mean(positive)) if positive else None
        ),
        "fraction_counterfactual_ended_closer_to_named_object": (
            float(np.mean(ended_named)) if ended_named else None
        ),
        "true_instruction_bddl_success": float(np.mean(true_success)) if true_success else None,
        "rows": rows,
        "skipped": skipped,
        "outcome_scope": (
            "The counterfactual outcome is a paired end-effector proximity shift under matched "
            "canonical simulator states. It is behavioral grounding evidence, not certified "
            "counterfactual LIBERO task success, because the BDDL predicate names only the "
            "original target."
        ),
    }


def run_exact_attention_audit(
    model: ChiVLA,
    sample_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    """Reconstruct all learned bilinear-attention modules from their weights."""
    import copy

    from xvla.nn.attention import causal_mask

    image, instruction, state, embodiment = sample_tensors

    def rational_norm(array: np.ndarray, module) -> np.ndarray:
        running_ms = float(module.running_ms.detach().cpu())
        pa = module.pa.detach().double().cpu().numpy()
        pb = module.pb.detach().double().cpu().numpy()
        mean_square = np.square(array).mean(axis=-1, keepdims=True) + float(module.eps)
        scaled = mean_square / max(running_ms, 1e-12)
        numerator = sum(pa[degree] * scaled**degree for degree in range(module.deg + 1))
        denominator = sum(pb[degree] * scaled**degree for degree in range(module.deg + 1))
        return array * (numerator / denominator) * max(running_ms, 1e-12) ** -0.5

    def reconstruct(attention, normalized_input: torch.Tensor, mask) -> np.ndarray:
        x = normalized_input[0].detach().double().cpu().numpy()

        def weight(linear) -> np.ndarray:
            return linear.weight.detach().double().cpu().numpy()

        def bias(linear) -> np.ndarray:
            return linear.bias.detach().double().cpu().numpy()

        projections = {
            "q1": x @ weight(attention.wq1).T + bias(attention.wq1),
            "k1": x @ weight(attention.wk1).T + bias(attention.wk1),
            "q2": x @ weight(attention.wq2).T + bias(attention.wq2),
            "k2": x @ weight(attention.wk2).T + bias(attention.wk2),
            "v": x @ weight(attention.wv).T + bias(attention.wv),
        }
        sequence_length = x.shape[0]
        if mask is None:
            mask_array = np.ones((sequence_length, sequence_length), dtype=np.float64)
        else:
            mask_array = mask.detach().double().cpu().numpy()
        visible = np.maximum(mask_array.sum(axis=-1), 1.0)
        row_scale = visible**-0.5 if attention.row_scale == "invsqrt" else visible**-1.0
        output = np.zeros((sequence_length, attention.dim), dtype=np.float64)
        for head_index in range(attention.n_heads):
            start = head_index * attention.head_dim
            stop = (head_index + 1) * attention.head_dim
            head_slice = slice(start, stop)
            q1 = projections["q1"][:, head_slice]
            k1 = projections["k1"][:, head_slice]
            q2 = projections["q2"][:, head_slice]
            k2 = projections["k2"][:, head_slice]
            value = projections["v"][:, head_slice]
            if attention.qk_norm == "rational":
                q1 = rational_norm(q1, attention.rn_q1)
                k1 = rational_norm(k1, attention.rn_k1)
                q2 = rational_norm(q2, attention.rn_q2)
                k2 = rational_norm(k2, attention.rn_k2)
            else:
                raise ValueError(
                    f"Exact audit currently requires rational QK norm, got {attention.qk_norm}"
                )
            pattern = ((q1 @ k1.T) * (q2 @ k2.T)) / attention._score_denom
            head_output = row_scale[:, None] * ((pattern * mask_array) @ value)
            output += head_output @ weight(attention.wo)[:, head_slice].T
        return output + bias(attention.wo)

    def audit_stack(stack_name: str, blocks, initial: torch.Tensor, mask) -> tuple[list[Any], Any]:
        rows = []
        hidden = initial
        for block_index, block in enumerate(blocks):
            normalized = block.rbn_attn(hidden)
            attention64 = copy.deepcopy(block.attn).double()
            mask64 = mask.double() if mask is not None else None
            with torch.inference_mode():
                deployed = attention64(normalized.double(), mask=mask64, method="explicit")
            reconstructed = reconstruct(block.attn, normalized, mask)
            deployed_array = deployed[0].detach().double().cpu().numpy()
            max_abs_error = float(np.max(np.abs(deployed_array - reconstructed)))
            relative_l2_error = float(
                np.linalg.norm(deployed_array - reconstructed)
                / max(np.linalg.norm(deployed_array), 1e-30)
            )
            row = {
                "stack": stack_name,
                "block_index": block_index,
                "heads": block.attn.n_heads,
                "dim": block.attn.dim,
                "sequence_length": int(normalized.shape[1]),
                "max_abs_error": max_abs_error,
                "relative_l2_error": relative_l2_error,
            }
            rows.append(row)
            print("EXACT_ATTENTION", json.dumps(row), flush=True)
            hidden = block(hidden, mask=mask)
        return rows, hidden

    with torch.inference_mode():
        vision_hidden = model.vision.patch(image)
        vision_hidden = vision_hidden.flatten(2).transpose(1, 2)
        vision_hidden = vision_hidden + model.vision.pos_emb
        vision_rows, _ = audit_stack(
            "vision", model.vision.blocks.blocks, vision_hidden, None
        )

        visual_tokens = model._visual_tokens(image)
        joint_hidden = torch.cat(
            [
                visual_tokens,
                model.bos.expand(image.shape[0], -1, -1),
                model.tok_emb(instruction),
                model.state_proj(state)[:, None],
                model.embodiment_emb(embodiment)[:, None],
                model.action_queries.expand(image.shape[0], -1, -1),
            ],
            dim=1,
        )
        joint_hidden = joint_hidden + model.pos_emb[:, : joint_hidden.shape[1]]
        joint_mask = causal_mask(
            joint_hidden.shape[1], device=joint_hidden.device, dtype=joint_hidden.dtype
        )
        joint_rows, _ = audit_stack(
            "joint", model.backbone.blocks, joint_hidden, joint_mask
        )

    rows = vision_rows + joint_rows
    return {
        "modules_audited": len(rows),
        "heads_audited": sum(int(row["heads"]) for row in rows),
        "vision_modules": len(vision_rows),
        "joint_modules": len(joint_rows),
        "max_abs_error": max(float(row["max_abs_error"]) for row in rows),
        "max_relative_l2_error": max(float(row["relative_l2_error"]) for row in rows),
        "all_modules_below_1e_minus_5": all(
            float(row["max_abs_error"]) < 1e-5 for row in rows
        ),
        "all_modules_below_2e_minus_5": all(
            float(row["max_abs_error"]) < 2e-5 for row in rows
        ),
        "rows": rows,
        "scope": (
            "This validates exact learned-weight reconstruction for every individual attention "
            "module in the vision and joint stacks. It remains a layerwise audit and does not "
            "materialize one compact symbolic contraction for the complete policy."
        ),
    }


def run_visual_subspace_intervention(
    args: argparse.Namespace,
    model: ChiVLA,
    suite,
    tasks: dict[int, str],
    cache_tasks: dict[int, str],
    cache_task_metadata: dict[str, Any],
    encode,
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Replicate the visual-bond causal intervention for one trained checkpoint."""
    if not 0 <= args.gram_task_start < args.gram_task_end <= suite.n_tasks:
        raise ValueError(
            f"Invalid Gram task range [{args.gram_task_start}, {args.gram_task_end})"
        )
    with args.cache.open("rb") as handle:
        frames = pickle.load(handle)
    episodes: dict[int, list[Any]] = defaultdict(list)
    for frame in frames:
        episodes[int(frame[0])].append(frame)
    samples = []
    dataset_to_official = {
        int(dataset_index): int(official_index)
        for dataset_index, official_index in cache_task_metadata[
            "dataset_to_official_task"
        ].items()
    }
    for episode_frames in episodes.values():
        episode_frames.sort(key=lambda item: int(item[1]))
        for index in range(len(episode_frames) - args.horizon):
            frame = episode_frames[index]
            dataset_task_index = int(frame[5])
            official_task_index = dataset_to_official[dataset_task_index]
            if (
                args.gram_task_start
                <= official_task_index
                < args.gram_task_end
            ):
                samples.append(
                    (
                        np.asarray(frame[2], dtype=np.uint8),
                        dataset_task_index,
                        official_task_index,
                        np.asarray(frame[3], dtype=np.float32),
                    )
                )
    if not samples:
        raise RuntimeError("No visual-bond samples could be built from the cache")
    if not 0 < args.rank < model.cfg.dim:
        raise ValueError(f"rank must be between 1 and {model.cfg.dim - 1}")

    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(samples))
    gram_count = min(args.gram_samples, len(samples) - 1)
    offline_eval_count = min(args.offline_eval_samples, len(samples) - gram_count)
    if gram_count <= 0 or offline_eval_count <= 0:
        raise RuntimeError("Visual-subspace analysis requires disjoint discovery and evaluation samples")
    gram_indices = permutation[:gram_count]
    offline_eval_indices = permutation[gram_count : gram_count + offline_eval_count]

    def tensorize(indices: np.ndarray):
        images = (
            torch.from_numpy(np.stack([samples[index][0] for index in indices]))
            .permute(0, 3, 1, 2)
            .float()
            .div(255)
            .cuda()
        )
        instructions = torch.tensor(
            [encode(cache_tasks[samples[index][1]]) for index in indices],
            dtype=torch.long,
            device="cuda",
        )
        states_array = np.stack([samples[index][3] for index in indices])
        states = torch.tensor(
            (states_array - stats["state_mean"]) / stats["state_std"],
            dtype=torch.float32,
            device="cuda",
        )
        embodiments = torch.zeros(len(indices), dtype=torch.long, device="cuda")
        return images, instructions, states, embodiments

    gram_images, gram_instructions, gram_states, gram_embodiments = tensorize(gram_indices)
    eval_images, eval_instructions, eval_states, eval_embodiments = tensorize(
        offline_eval_indices
    )

    if args.gram_probes <= 0:
        raise ValueError("gram_probes must be positive")
    if args.random_controls <= 0:
        raise ValueError("random_controls must be positive")
    gram = torch.zeros(model.cfg.dim, model.cfg.dim, dtype=torch.float64, device="cuda")
    activation_gram = torch.zeros_like(gram)
    component_grams = [
        torch.zeros_like(gram) for _ in range(model.cfg.action_dim)
    ]
    gradient_rows = 0
    activation_rows = 0
    gram_batch_size = 128
    probe_rng = torch.Generator(device="cuda").manual_seed(args.seed + 17011)
    for start in range(0, len(gram_indices), gram_batch_size):
        stop = min(start + gram_batch_size, len(gram_indices))
        with torch.no_grad():
            visual = model._visual_tokens(gram_images[start:stop])
            if args.activation_energy_control:
                flat_visual = visual.reshape(-1, model.cfg.dim).double()
                activation_gram += flat_visual.T @ flat_visual
                activation_rows += flat_visual.shape[0]
        visual = visual.detach().requires_grad_(True)
        with torch.enable_grad():
            prediction = forward_from_visual_tokens(
                model,
                visual,
                gram_instructions[start:stop],
                gram_states[start:stop],
                gram_embodiments[start:stop],
            )
            if args.gram_action_group == "gripper":
                selected = prediction[:, :, -1].sum()
                gradient = torch.autograd.grad(selected, visual)[0]
                flat_gradient = gradient.reshape(-1, model.cfg.dim).double()
                gram += flat_gradient.T @ flat_gradient
                gradient_rows += flat_gradient.shape[0]
            else:
                backward_index = 0
                backward_total = model.cfg.action_dim * args.gram_probes
                for action_index in range(model.cfg.action_dim):
                    for _ in range(args.gram_probes):
                        signs = torch.randint(
                            0,
                            2,
                            prediction.shape[:2],
                            generator=probe_rng,
                            device="cuda",
                            dtype=torch.int64,
                        ).to(prediction.dtype)
                        signs = signs.mul_(2).sub_(1)
                        selected = (
                            prediction[:, :, action_index] * signs
                        ).sum()
                        backward_index += 1
                        gradient = torch.autograd.grad(
                            selected,
                            visual,
                            retain_graph=backward_index < backward_total,
                        )[0]
                        flat_gradient = gradient.reshape(-1, model.cfg.dim).double()
                        component_grams[action_index] += (
                            flat_gradient.T @ flat_gradient
                        )
                        gradient_rows += flat_gradient.shape[0]
    if args.gram_action_group == "gripper":
        gram /= max(gradient_rows, 1)
    else:
        for component in component_grams:
            component /= component.trace().clamp_min(1e-30)
            gram += component
        gram /= len(component_grams)

    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.flip(0)
    top_basis = eigenvectors.flip(1)[:, : args.rank]
    top_projector = (top_basis @ top_basis.T).float()
    activation_eigenvalues = None
    activation_projector = None
    if args.activation_energy_control:
        activation_gram /= max(activation_rows, 1)
        activation_eigenvalues, activation_eigenvectors = torch.linalg.eigh(
            activation_gram
        )
        activation_eigenvalues = activation_eigenvalues.flip(0)
        activation_basis = activation_eigenvectors.flip(1)[:, : args.rank]
        activation_projector = (activation_basis @ activation_basis.T).float()
    random_projectors = []
    for control_index in range(args.random_controls):
        torch_rng = torch.Generator(device="cuda").manual_seed(
            args.seed + 4109 + 7919 * control_index
        )
        random_matrix = torch.randn(
            model.cfg.dim,
            args.rank,
            generator=torch_rng,
            dtype=torch.float64,
            device="cuda",
        )
        random_basis, _ = torch.linalg.qr(random_matrix)
        random_projectors.append((random_basis @ random_basis.T).float())

    condition_projectors: list[tuple[str, torch.Tensor | None]] = [
        ("full", None),
        ("causal_topk", top_projector),
    ]
    if activation_projector is not None:
        condition_projectors.append(("activation_energy_topk", activation_projector))
    for control_index, projector in enumerate(random_projectors):
        condition = "random_topk" if control_index == 0 else f"random_topk_{control_index}"
        condition_projectors.append((condition, projector))

    offline_predictions: dict[str, list[torch.Tensor]] = {
        condition: [] for condition, _ in condition_projectors
    }
    activation_reconstruction_sse = {
        condition: 0.0
        for condition, projector in condition_projectors
        if projector is not None
    }
    activation_reconstruction_elements = 0
    with torch.inference_mode():
        for start in range(0, len(offline_eval_indices), gram_batch_size):
            stop = min(start + gram_batch_size, len(offline_eval_indices))
            visual = model._visual_tokens(eval_images[start:stop])
            activation_reconstruction_elements += visual.numel()
            for condition, projector in condition_projectors:
                intervened = visual if projector is None else visual @ projector.T
                if projector is not None:
                    activation_reconstruction_sse[condition] += float(
                        torch.sum((intervened - visual).double().square())
                    )
                offline_predictions[condition].append(
                    forward_from_visual_tokens(
                        model,
                        intervened,
                        eval_instructions[start:stop],
                        eval_states[start:stop],
                        eval_embodiments[start:stop],
                    ).float()
                )
    concatenated = {
        condition: torch.cat(chunks) for condition, chunks in offline_predictions.items()
    }
    action_groups = {
        "translation": slice(0, 3),
        "rotation": slice(3, 6),
        "gripper": slice(6, 7),
    }
    offline_mse: dict[str, dict[str, float]] = {}
    for condition in offline_predictions:
        if condition == "full":
            continue
        offline_mse[condition] = {}
        for group, action_slice in action_groups.items():
            offline_mse[condition][group] = float(
                torch.mean(
                    (
                        concatenated[condition][:, :, action_slice]
                        - concatenated["full"][:, :, action_slice]
                    )
                    ** 2
                )
            )

    random_ratios = {
        condition: {
            group: offline_mse[condition][group]
            / max(offline_mse["causal_topk"][group], 1e-30)
            for group in action_groups
        }
        for condition in offline_mse
        if condition.startswith("random_topk")
    }
    activation_to_causal_ratios = None
    if "activation_energy_topk" in offline_mse:
        activation_to_causal_ratios = {
            group: offline_mse["activation_energy_topk"][group]
            / max(offline_mse["causal_topk"][group], 1e-30)
            for group in action_groups
        }
    by_condition = {}
    if not args.subspace_offline_only:
        requested_conditions = {
            value.strip()
            for value in args.subspace_rollout_conditions.split(",")
            if value.strip()
        }
        available_conditions = {
            condition for condition, _ in condition_projectors
        }
        unknown_conditions = requested_conditions - available_conditions
        if unknown_conditions:
            raise ValueError(
                "Unknown subspace rollout conditions: "
                + ", ".join(sorted(unknown_conditions))
            )
        for condition, projector in condition_projectors:
            if requested_conditions and condition not in requested_conditions:
                continue
            print(f"VISUAL_SUBSPACE condition={condition}", flush=True)
            by_condition[condition] = run_capability(
                args,
                model,
                suite,
                tasks,
                encode,
                stats,
                visual_projector=projector,
            )

    return {
        "rank": args.rank,
        "ambient_dimension": model.cfg.dim,
        "gram_samples": len(gram_indices),
        "gram_task_range": [args.gram_task_start, args.gram_task_end],
        "gram_task_index_space": "official LIBERO task indices",
        "cache_task_metadata": cache_task_metadata,
        "offline_eval_samples": len(offline_eval_indices),
        "offline_eval_disjoint_from_gram": True,
        "gradient_rows": gradient_rows,
        "activation_energy_control": args.activation_energy_control,
        "activation_rows": activation_rows,
        "gram_action_group": args.gram_action_group,
        "gram_probes": args.gram_probes,
        "random_controls": args.random_controls,
        "offline_only": args.subspace_offline_only,
        "subspace_rollout_conditions": sorted(by_condition),
        "top_eigenvalues": eigenvalues[:16].detach().cpu().tolist(),
        "top_rank_spectral_mass": float(
            eigenvalues[: args.rank].clamp_min(0).sum()
            / eigenvalues.clamp_min(0).sum().clamp_min(1e-30)
        ),
        "activation_top_eigenvalues": (
            activation_eigenvalues[:16].detach().cpu().tolist()
            if activation_eigenvalues is not None
            else None
        ),
        "activation_top_rank_spectral_mass": (
            float(
                activation_eigenvalues[: args.rank].clamp_min(0).sum()
                / activation_eigenvalues.clamp_min(0).sum().clamp_min(1e-30)
            )
            if activation_eigenvalues is not None
            else None
        ),
        "offline_activation_reconstruction_mse": {
            condition: squared_error
            / max(activation_reconstruction_elements, 1)
            for condition, squared_error in activation_reconstruction_sse.items()
        },
        "offline_mse_to_full": offline_mse,
        "offline_activation_to_causal_mse_ratio": activation_to_causal_ratios,
        "offline_random_to_causal_mse_ratio": random_ratios["random_topk"],
        "offline_random_controls_to_causal_mse_ratio": random_ratios,
        "offline_median_random_to_causal_mse_ratio": {
            group: float(
                np.median([ratios[group] for ratios in random_ratios.values()])
            )
            for group in action_groups
        },
        "overall_by_condition": {
            condition: float(result["overall"])
            for condition, result in by_condition.items()
        },
        "by_condition": by_condition,
        "discovery_scope": (
            f"The {args.gram_action_group} visual-bond subspace is estimated from the training cache "
            "after translating pinned dataset task IDs to official LIBERO task IDs. "
            "Offline reconstruction uses a disjoint cache sample, while closed-loop evaluation "
            "uses independent canonical simulator states. This is a data-driven causal "
            "bottleneck, not a weight-only whole-policy decomposition."
        ),
    }


def run_offline_checkpoint_diagnostic(
    args: argparse.Namespace,
    model: ChiVLA,
    tasks: dict[int, str],
    encode,
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Measure checkpoint fit on one fixed cache sample, including per-task failures."""
    with args.cache.open("rb") as handle:
        frames = pickle.load(handle)
    episodes: dict[int, list[Any]] = defaultdict(list)
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
    if not samples:
        raise RuntimeError("No offline diagnostic samples could be built from the cache")
    fixed_rng = np.random.default_rng(20260821)
    sample_indices = fixed_rng.choice(
        len(samples), size=min(args.offline_samples, len(samples)), replace=False
    )
    action_mean = torch.tensor(stats["action_mean"], dtype=torch.float32, device="cuda")
    action_std = torch.tensor(stats["action_std"], dtype=torch.float32, device="cuda")
    rows = []
    batch_size = 128
    with torch.inference_mode():
        for start in range(0, len(sample_indices), batch_size):
            selected_indices = sample_indices[start : start + batch_size]
            batch_samples = [samples[index] for index in selected_indices]
            images = (
                torch.from_numpy(np.stack([sample[0] for sample in batch_samples]))
                .permute(0, 3, 1, 2)
                .float()
                .div(255)
                .cuda()
            )
            instructions = torch.tensor(
                [encode(tasks[sample[1]]) for sample in batch_samples],
                dtype=torch.long,
                device="cuda",
            )
            states_array = np.stack([sample[2] for sample in batch_samples])
            states = torch.tensor(
                (states_array - stats["state_mean"]) / stats["state_std"],
                dtype=torch.float32,
                device="cuda",
            )
            target_raw = torch.tensor(
                np.stack([sample[3] for sample in batch_samples]),
                dtype=torch.float32,
                device="cuda",
            )
            target_normalized = (target_raw - action_mean) / action_std
            prediction_normalized, _ = model(
                images,
                instructions,
                states,
                torch.zeros(len(batch_samples), dtype=torch.long, device="cuda"),
            )
            prediction_raw = prediction_normalized * action_std + action_mean
            for offset, sample in enumerate(batch_samples):
                rows.append(
                    {
                        "task": sample[1],
                        "normalized_mse": float(
                            torch.mean(
                                (prediction_normalized[offset] - target_normalized[offset]) ** 2
                            )
                        ),
                        "arm_mae": float(
                            torch.mean(torch.abs(prediction_raw[offset, :, :6] - target_raw[offset, :, :6]))
                        ),
                        "gripper_mae": float(
                            torch.mean(torch.abs(prediction_raw[offset, :, 6] - target_raw[offset, :, 6]))
                        ),
                        "gripper_sign_accuracy": float(
                            torch.mean(
                                (torch.sign(prediction_raw[offset, :, 6]) == torch.sign(target_raw[offset, :, 6])).float()
                            )
                        ),
                        "prediction_arm_std": float(prediction_raw[offset, :, :6].std()),
                    }
                )

    def summarize(selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            key: float(np.mean([row[key] for row in selected_rows]))
            for key in (
                "normalized_mse",
                "arm_mae",
                "gripper_mae",
                "gripper_sign_accuracy",
                "prediction_arm_std",
            )
        }

    return {
        "sample_count": len(rows),
        "sample_seed": 20260821,
        "overall": summarize(rows),
        "per_task": {
            str(task_index): {
                "language": tasks[task_index],
                "samples": sum(row["task"] == task_index for row in rows),
                **summarize([row for row in rows if row["task"] == task_index]),
            }
            for task_index in sorted(tasks)
            if any(row["task"] == task_index for row in rows)
        },
        "scope": (
            "This diagnostic samples the training cache and is not a held-out generalization "
            "estimate. It tests checkpoint integrity and whether closed-loop collapse coexists "
            "with low behavior-cloning error."
        ),
    }


def run_weight_surgery(
    args: argparse.Namespace,
    model: ChiVLA,
    suite,
    tasks: dict[int, str],
    encode,
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Apply exact-weight attention subspaces directly to learned coefficient columns."""
    from xvla.train.exact_odt_attention_proto import build_gram_reduced_head

    if not 0 <= args.block_index < len(model.backbone.blocks):
        raise ValueError(f"block-index {args.block_index} is outside the joint stack")
    attention = model.backbone.blocks[args.block_index].attn
    if not hasattr(attention, "wq1"):
        raise ValueError("Weight surgery requires bilinear attention")
    if not 0 < args.rank < attention.dim:
        raise ValueError(f"rank must be between 1 and {attention.dim - 1}")

    matrix_names = ("wq1", "wk1", "wq2", "wk2", "wv")
    original_weights = {
        name: getattr(attention, name).weight.detach().clone() for name in matrix_names
    }

    def weight_and_bias(linear) -> tuple[np.ndarray, np.ndarray]:
        return (
            linear.weight.detach().double().cpu().numpy(),
            linear.bias.detach().double().cpu().numpy(),
        )

    q1_weight, q1_bias = weight_and_bias(attention.wq1)
    k1_weight, k1_bias = weight_and_bias(attention.wk1)
    q2_weight, q2_bias = weight_and_bias(attention.wq2)
    k2_weight, k2_bias = weight_and_bias(attention.wk2)
    value_weight, value_bias = weight_and_bias(attention.wv)
    output_weight = attention.wo.weight.detach().double().cpu().numpy()
    matrices = (q1_weight, k1_weight, q2_weight, k2_weight, value_weight)
    rng = np.random.default_rng(args.seed)
    identity = np.eye(attention.dim)
    projectors: dict[str, list[np.ndarray]] = {
        "keep_top": [],
        "keep_random": [],
        "remove_top": [],
        "remove_random": [],
    }
    norm_match_scales: dict[str, list[float]] = {
        "keep_random_normmatched": [],
        "remove_random_normmatched": [],
    }
    spectra = {}
    for head_index in range(attention.n_heads):
        start = head_index * attention.head_dim
        stop = (head_index + 1) * attention.head_dim
        head_slice = slice(start, stop)
        gram = build_gram_reduced_head(
            q1_weight[head_slice],
            q1_bias[head_slice],
            k1_weight[head_slice],
            k1_bias[head_slice],
            q2_weight[head_slice],
            q2_bias[head_slice],
            k2_weight[head_slice],
            k2_bias[head_slice],
            value_weight[head_slice],
            value_bias[head_slice],
            output_weight[:, head_slice],
        )
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        top_basis = eigenvectors[:, : args.rank]
        top_projector = top_basis @ top_basis.T
        random_basis, _ = np.linalg.qr(rng.normal(size=(attention.dim, args.rank)))
        random_projector = random_basis @ random_basis.T
        projectors["keep_top"].append(top_projector)
        projectors["keep_random"].append(random_projector)
        projectors["remove_top"].append(identity - top_projector)
        projectors["remove_random"].append(identity - random_projector)

        deltas = {}
        for projector_name, projector in (
            ("keep_top", top_projector),
            ("keep_random", random_projector),
            ("remove_top", identity - top_projector),
            ("remove_random", identity - random_projector),
        ):
            deltas[projector_name] = sum(
                float(np.square(matrix[head_slice] @ projector - matrix[head_slice]).sum())
                for matrix in matrices
            )
        norm_match_scales["keep_random_normmatched"].append(
            math.sqrt(deltas["keep_top"] / max(deltas["keep_random"], 1e-30))
        )
        norm_match_scales["remove_random_normmatched"].append(
            math.sqrt(deltas["remove_top"] / max(deltas["remove_random"], 1e-30))
        )
        nonnegative = np.clip(eigenvalues, 0, None)
        spectra[str(head_index)] = {
            "top_eigenvalues": eigenvalues[:8].tolist(),
            "top_rank_spectral_mass": float(
                nonnegative[: args.rank].sum() / max(nonnegative.sum(), 1e-30)
            ),
        }

    def restore_weights() -> None:
        with torch.no_grad():
            for name in matrix_names:
                getattr(attention, name).weight.copy_(original_weights[name])

    def apply_condition(condition: str) -> float:
        restore_weights()
        if condition == "baseline":
            return 0.0
        projector_condition = condition.replace("_normmatched", "")
        if projector_condition not in projectors:
            raise ValueError(f"Unknown surgery condition {condition}")
        squared_delta = 0.0
        squared_base = 0.0
        with torch.no_grad():
            for head_index in range(attention.n_heads):
                start = head_index * attention.head_dim
                stop = (head_index + 1) * attention.head_dim
                head_slice = slice(start, stop)
                projector = torch.tensor(
                    projectors[projector_condition][head_index],
                    dtype=original_weights["wq1"].dtype,
                    device="cuda",
                )
                interpolation = (
                    norm_match_scales[condition][head_index]
                    if condition in norm_match_scales
                    else 1.0
                )
                for name in matrix_names:
                    original = original_weights[name][head_slice]
                    projected = original @ projector
                    edited = original + interpolation * (projected - original)
                    getattr(attention, name).weight[head_slice].copy_(edited)
                    squared_delta += float((edited - original).float().pow(2).sum())
                    squared_base += float(original.float().pow(2).sum())
        return math.sqrt(squared_delta / max(squared_base, 1e-30))

    conditions = [item.strip() for item in args.surgery_conditions.split(",") if item.strip()]
    by_condition = {}
    relative_weight_change = {}
    try:
        for condition in conditions:
            relative_weight_change[condition] = apply_condition(condition)
            print(
                f"SURGERY {condition}: relative weight change "
                f"{relative_weight_change[condition]:.6f}",
                flush=True,
            )
            by_condition[condition] = run_capability(
                args, model, suite, tasks, encode, stats
            )
    finally:
        restore_weights()

    overall = {
        condition: float(result["overall"]) for condition, result in by_condition.items()
    }
    comparisons = {}
    if "keep_top" in overall and "keep_random_normmatched" in overall:
        comparisons["keep_top_minus_random_normmatched"] = (
            overall["keep_top"] - overall["keep_random_normmatched"]
        )
    if "remove_top" in overall and "remove_random_normmatched" in overall:
        comparisons["remove_random_normmatched_minus_top"] = (
            overall["remove_random_normmatched"] - overall["remove_top"]
        )
    return {
        "block_index": args.block_index,
        "rank": args.rank,
        "rank_fraction": args.rank / attention.dim,
        "conditions": conditions,
        "overall_by_condition": overall,
        "comparisons": comparisons,
        "relative_weight_change": relative_weight_change,
        "by_condition": by_condition,
        "spectra": spectra,
        "discovery_inputs": "trained weights only",
        "evaluation_split": {
            "task_start": args.task_start,
            "task_end": args.task_end,
            "eps_per_task": args.eps_per_task,
        },
        "scope": (
            "This is direct coefficient surgery with weight-derived subspaces. A discovery "
            "sweep must be followed by a held-out task confirmation before making a selective "
            "behavior-edit claim."
        ),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items() if key != "first_sample"}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        args.task_end = min(args.task_start + 1, args.task_end)
        args.eps_per_task = min(1, args.eps_per_task)
        args.max_steps = min(8, args.max_steps)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if not torch.cuda.is_available():
        raise RuntimeError("This Athena runner requires a CUDA compute node")

    generalist_source_start = None
    evaluator_start_sha256 = None
    expected_evaluator_sha256 = None
    evaluation_gpu_family = None
    if args.model_metadata is not None:
        evaluator_start_sha256 = file_sha256(Path(__file__))
        expected_evaluator_sha256 = os.environ.get("XVLA_FROZEN_EVALUATOR_SHA256")
        if expected_evaluator_sha256 is None:
            raise RuntimeError("XVLA_FROZEN_EVALUATOR_SHA256 is required")
        if evaluator_start_sha256 != expected_evaluator_sha256:
            raise RuntimeError(
                "Evaluator source does not match the Slurm wrapper's frozen SHA-256"
            )
        evaluation_gpu_family = os.environ.get(
            "XVLA_FROZEN_EVALUATION_GPU_FAMILY"
        )
        if evaluation_gpu_family != FROZEN_GENERALIST_EVALUATION_GPU_FAMILY:
            raise RuntimeError(
                "XVLA_FROZEN_EVALUATION_GPU_FAMILY must be "
                f"{FROZEN_GENERALIST_EVALUATION_GPU_FAMILY!r}"
            )
        if torch.cuda.get_device_name(0) != FROZEN_GENERALIST_EVALUATION_GPU_NAME:
            raise RuntimeError(
                "Generalist evaluation must run on "
                f"{FROZEN_GENERALIST_EVALUATION_GPU_NAME}"
            )
        imported_sources = generalist_imported_source_snapshot()
        validate_generalist_imported_sources(imported_sources)
        generalist_source_start = {
            "evaluator_sha256": evaluator_start_sha256,
            "evaluation_gpu_family": evaluation_gpu_family,
            "evaluation_gpu_name": torch.cuda.get_device_name(0),
            "imported_sources": imported_sources,
            "imported_bundle_sha256": source_bundle_sha256(imported_sources),
        }

    suite = load_suite(args.suite)
    tasks = task_languages(suite)
    cache_tasks, cache_task_metadata = load_dataset_task_languages(
        args.suite, tasks
    )
    model_metadata = None
    if args.model_metadata is not None:
        print(f"Loading one profile sample from {args.cache}", flush=True)
        stats = load_cache_profile_sample(args.cache, args.horizon)
        vocab, fixed_stats, model_metadata = load_model_metadata(args.model_metadata)
        validate_model_metadata(
            args,
            model_metadata,
            vocab,
            fixed_stats,
            stats,
            cache_task_metadata,
        )
        generalist_source_start.update(
            {
                "checkpoint_sha256": file_sha256(args.checkpoint),
                "model_metadata_sha256": file_sha256(args.model_metadata),
                "manifest_sha256": file_sha256(
                    Path(str(model_metadata["source_manifest"]))
                ),
                "evaluation_cache_sha256": file_sha256(args.cache),
            }
        )
        stats.update(fixed_stats)
        stats["sample_count"] = int(
            model_metadata["source_caches"][args.suite]["sample_count"]
        )
        encode = build_encoder(vocab)
    else:
        print(f"Loading cache statistics from {args.cache}", flush=True)
        stats = load_cache_statistics(args.cache, args.horizon)
        training_tasks = task_languages(load_suite(args.training_suite))
        vocab, encode = build_vocab(training_tasks)
    print(f"Tasks={len(tasks)} vocab={len(vocab)} samples={stats['sample_count']}", flush=True)
    ensemble_checkpoints = [args.checkpoint, *args.ensemble_checkpoint]
    if args.mode == "ensemble_capability":
        if len(ensemble_checkpoints) < 2:
            raise ValueError(
                "ensemble_capability requires at least one --ensemble-checkpoint"
            )
        member_models = [
            load_model(args, len(vocab), stats, checkpoint=checkpoint)
            for checkpoint in ensemble_checkpoints
        ]
        model: Any = EnsemblePolicy(member_models, args.ensemble_reduction).cuda().eval()
    else:
        if args.ensemble_checkpoint:
            raise ValueError(
                "--ensemble-checkpoint is only valid for ensemble_capability"
            )
        model = load_model(args, len(vocab), stats)
    cp_pruning = None
    if args.mode == "cp_pruning":
        if args.architecture != "chi":
            raise ValueError("cp_pruning requires architecture=chi")
        cp_pruning = apply_cp_term_pruning(
            model, args.prune_fraction, args.prune_strategy, args.seed
        )
    sample_tensors = tensorize_sample(
        stats["first_sample"], encode, cache_tasks, stats
    )
    with torch.inference_mode():
        prediction, _ = model(*sample_tensors)
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Checkpoint produced a non-finite prediction")

    profile = profile_model(model, sample_tensors, args.profile_iters)
    result: dict[str, Any] = {
        "mode": args.mode,
        "architecture": args.architecture,
        "vision_encoder": args.vision_encoder,
        "suite": args.suite,
        "training_suite": args.training_suite,
        "checkpoint": str(args.checkpoint),
        "ensemble_checkpoints": (
            [str(checkpoint) for checkpoint in ensemble_checkpoints]
            if args.mode == "ensemble_capability"
            else None
        ),
        "ensemble_reduction": (
            args.ensemble_reduction if args.mode == "ensemble_capability" else None
        ),
        "cp_pruning": cp_pruning,
        "cache": str(args.cache),
        "cache_task_metadata": cache_task_metadata,
        "model_metadata": str(args.model_metadata) if args.model_metadata else None,
        "seed": args.seed,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "numerical_environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "libero_version": installed_version("libero"),
            "robosuite_version": installed_version("robosuite"),
            "mujoco_version": installed_version("mujoco"),
        },
        "evaluation_protocol": {
            "res": args.res,
            "horizon": args.horizon,
            "num_steps_wait": args.num_steps_wait,
            "exec_h": args.exec_h,
            "eps_per_task": args.eps_per_task,
            "max_steps": args.max_steps,
        },
        "vocab_size": len(vocab),
        "cache_stats": stats,
        "profile": profile,
        "prediction_shape": list(prediction.shape),
        "prediction_finite": True,
        "prediction_sample": prediction[0].detach().float().cpu().tolist(),
        "evaluation_scope": (
            "In-domain evaluation of a jointly trained multi-suite checkpoint"
            if args.model_metadata is not None
            else (
                "In-domain suite evaluation"
                if args.suite == args.training_suite
                else (
                    f"Zero-shot cross-suite evaluation of a {args.training_suite}-trained checkpoint. "
                    "Vocabulary and normalization remain fixed to the training suite, with unseen "
                    "instruction words mapped to the padding identifier."
                )
            )
        ),
        "training_metadata": model_metadata,
    }

    if args.mode == "smoke":
        result["capability"] = run_capability(args, model, suite, tasks, encode, stats)
    elif args.mode == "capability":
        result["capability"] = run_capability(args, model, suite, tasks, encode, stats)
    elif args.mode == "ensemble_capability":
        result["capability"] = run_capability(args, model, suite, tasks, encode, stats)
    elif args.mode == "cp_pruning":
        if args.suite != "libero_object" or args.training_suite != "libero_object":
            raise ValueError("cp_pruning pilot currently requires Object training and evaluation")
        result["capability"] = run_capability(args, model, suite, tasks, encode, stats)
    elif args.mode == "offline_diagnostic":
        if args.suite != args.training_suite:
            raise ValueError("offline_diagnostic requires suite=training_suite")
        result["offline_diagnostic"] = run_offline_checkpoint_diagnostic(
            args, model, cache_tasks, encode, stats
        )
    elif args.mode == "causal":
        result["causal"] = run_causal_intervention(args, model, suite, tasks, encode, stats)
    elif args.mode == "visual_subspace":
        if args.architecture != "chi":
            raise ValueError("visual_subspace requires architecture=chi")
        if args.suite != "libero_object" or args.training_suite != "libero_object":
            raise ValueError("visual_subspace currently requires Object training and evaluation")
        result["visual_subspace"] = run_visual_subspace_intervention(
            args,
            model,
            suite,
            tasks,
            cache_tasks,
            cache_task_metadata,
            encode,
            stats,
        )
    elif args.mode == "exact_attention":
        if args.architecture != "chi":
            raise ValueError("exact_attention requires architecture=chi")
        result["exact_attention"] = run_exact_attention_audit(model, sample_tensors)
    elif args.mode == "surgery":
        if args.architecture != "chi":
            raise ValueError("surgery requires architecture=chi")
        result["surgery"] = run_weight_surgery(
            args, model, suite, tasks, encode, stats
        )

    if args.model_metadata is not None:
        evaluator_end_sha256 = file_sha256(Path(__file__))
        imported_sources_end = generalist_imported_source_snapshot()
        validate_generalist_imported_sources(imported_sources_end)
        generalist_source_end = {
            "evaluator_sha256": evaluator_end_sha256,
            "evaluation_gpu_family": evaluation_gpu_family,
            "evaluation_gpu_name": torch.cuda.get_device_name(0),
            "imported_sources": imported_sources_end,
            "imported_bundle_sha256": source_bundle_sha256(imported_sources_end),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "model_metadata_sha256": file_sha256(args.model_metadata),
            "manifest_sha256": file_sha256(
                Path(str(model_metadata["source_manifest"]))
            ),
            "evaluation_cache_sha256": file_sha256(args.cache),
        }
        if generalist_source_end != generalist_source_start:
            raise RuntimeError(
                "Generalist source, checkpoint, metadata, manifest, or cache "
                "changed during evaluation"
            )
        result["source_identity"] = {
            "expected_evaluator_sha256": expected_evaluator_sha256,
            "start": generalist_source_start,
            "end": generalist_source_end,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(result), indent=2) + "\n")
    temporary.replace(args.output)
    print("RESULT", json.dumps(json_ready(result), indent=2), flush=True)


if __name__ == "__main__":
    main()
