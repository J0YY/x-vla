#!/usr/bin/env python3
"""Certify deployed RationalNorm arithmetic on frozen LIBERO-Object inputs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from athena.libero_dataset_metadata import DATASETS, load_dataset_task_languages
from athena.run_xvla_experiment import (
    build_encoder,
    build_vocab,
    file_sha256,
    load_suite,
    make_config,
    task_languages,
)
from xvla.models.vla import ChiVLA
from xvla.nn.normalization import RationalNorm


SCHEMA = "xvla-rational-norm-safety-v1"
SELECTION_SCHEMA = "xvla-rational-norm-safety-sha256-ranking-v1"
PROTOCOL = {
    "suite": "libero_object",
    "architecture": "chi",
    "vision_encoder": "conv",
    "resolution": 64,
    "action_horizon": 8,
    "full_samples_per_checkpoint": 4096,
    "smoke_samples": 32,
    "batch_size": 32,
    "matmul_precision": "highest",
    "v_certificate_interval": [0.1, 10.0],
    "reference": (
        "Every RationalNorm scale is recomputed from the same float32 input values "
        "with float64 mean-square, polynomial, division, and root operations. The "
        "reference output is cast back to float32, and all other model operations "
        "remain unchanged."
    ),
    "primary_action_metric": (
        "maximum of normalized-model-output NRMSE and denormalized physical-action NRMSE"
    ),
}
FROZEN_GATES = {
    "primary_action_nrmse_at_most": 1.0e-3,
    "primary_all_denominator_certificates_pass": True,
    "secondary_each_site_fraction_rows_v_in_0p1_10_at_least": 0.99,
    "secondary_local_scale_relative_error_threshold": 0.034,
    "secondary_each_site_fraction_local_scale_error_at_or_below_threshold_at_least": 0.99,
    "all_three_checkpoints_must_pass": True,
}
FROZEN_CACHE = {
    "basename": "libero_frames_100000_64.pkl",
    "sha256": "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662",
    "frames": 66984,
}
FROZEN_PROVENANCE_JOB_ID = "830988"
FROZEN_CHECKPOINTS = {
    0: {
        "basename": "ckpt_linear_rat_conv_s0.pt",
        "sha256": "4f9f3eef4bd661934b7c66af117995f02bdfc777368f52464f03d229ccb3e72d",
    },
    1: {
        "basename": "ckpt_linear_rat_conv_s1_matched.pt",
        "sha256": "9d8df0c30583222490536b21aac040b47c5f89556aa23c08265605f7c2bc8cb6",
    },
    2: {
        "basename": "ckpt_linear_rat_conv_s2_matched.pt",
        "sha256": "fc2e0bfa1a2ea2737ed0af13b168cd2562e4b0e36b159d0f98f3c5038131ace5",
    },
}
FROZEN_DATASET_METADATA_SHA256 = (
    "34caee9641ae50bb4e077de306a7d0031753757882da8b1f117e7ea36a486b42"
)
FROZEN_DATASET_TO_OFFICIAL_TASK = {
    "0": 9,
    "1": 4,
    "2": 1,
    "3": 3,
    "4": 0,
    "5": 7,
    "6": 2,
    "7": 6,
    "8": 5,
    "9": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-seed", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--provenance-result", type=Path, required=True)
    parser.add_argument("--expected-provenance-job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("strict_smoke", "full"), required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
    )
    return parser.parse_args()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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


def cache_sample_sha256(record: Any, offset: int) -> str:
    if not isinstance(record, (list, tuple)) or len(record) != 6:
        raise RuntimeError(f"Cache record {offset} is not a six-field tuple")
    digest = hashlib.sha256()
    _update_blob(digest, "schema", b"xvla-rational-norm-model-input-v1")
    _update_blob(digest, "cache_offset", struct.pack("<q", offset))
    _update_blob(digest, "episode_index", struct.pack("<q", int(record[0])))
    _update_blob(digest, "frame_index", struct.pack("<q", int(record[1])))
    _update_blob(digest, "task_index", struct.pack("<q", int(record[5])))
    _update_array(digest, "image", np.asarray(record[2]))
    _update_array(digest, "state", np.asarray(record[3]))
    return digest.hexdigest()


def select_cache_samples(
    frames: list[Any] | tuple[Any, ...],
    checkpoint_sha256: str,
    sample_count: int,
) -> dict[str, Any]:
    if len(checkpoint_sha256) != 64:
        raise ValueError("Checkpoint SHA-256 is malformed")
    if sample_count <= 0 or sample_count > len(frames):
        raise ValueError("Requested sample count is outside the cache")
    ranked = []
    salt = bytes.fromhex(checkpoint_sha256)
    for offset, record in enumerate(frames):
        input_sha = cache_sample_sha256(record, offset)
        rank_digest = hashlib.sha256()
        rank_digest.update(SELECTION_SCHEMA.encode("ascii"))
        rank_digest.update(b"\0")
        rank_digest.update(salt)
        rank_digest.update(bytes.fromhex(input_sha))
        ranked.append((rank_digest.hexdigest(), offset, input_sha))
    ranked.sort(key=lambda item: (item[0], item[1]))
    selected = ranked[:sample_count]
    order_digest = hashlib.sha256()
    offset_digest = hashlib.sha256()
    input_digest = hashlib.sha256()
    records = []
    for rank_sha, offset, input_sha in selected:
        record = frames[offset]
        order_digest.update(bytes.fromhex(rank_sha))
        offset_digest.update(struct.pack("<q", offset))
        input_digest.update(bytes.fromhex(input_sha))
        records.append(
            {
                "rank_sha256": rank_sha,
                "cache_offset": offset,
                "input_sha256": input_sha,
                "episode_index": int(record[0]),
                "frame_index": int(record[1]),
                "task_index": int(record[5]),
            }
        )
    return {
        "schema": SELECTION_SCHEMA,
        "candidate_count": len(frames),
        "selected_count": sample_count,
        "checkpoint_sha256_salt": checkpoint_sha256,
        "ordering": "ascending SHA-256 digest, cache offset as collision tiebreaker",
        "rank_order_sha256": order_digest.hexdigest(),
        "selected_offsets_sha256": offset_digest.hexdigest(),
        "selected_inputs_sha256": input_digest.hexdigest(),
        "records": records,
    }


def validate_provenance(
    path: Path,
    cache: Path,
    expected_cache_sha256: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Object cache provenance result is absent: {path}")
    with path.open() as handle:
        result = json.load(handle)
    expected_repository, expected_revision = DATASETS["libero_object"]
    errors = []
    if result.get("schema") != "xvla-cache-provenance-v1":
        errors.append("wrong provenance schema")
    if result.get("suite") != "libero_object":
        errors.append("provenance suite is not libero_object")
    if result.get("verified") is not True:
        errors.append("provenance did not pass")
    cache_result = result.get("cache", {})
    if cache_result.get("sha256") != expected_cache_sha256:
        errors.append("provenance cache SHA differs from the frozen SHA")
    if int(cache_result.get("frames", -1)) != 66984:
        errors.append("provenance cache frame count is not 66,984")
    if int(cache_result.get("resolution", -1)) != 64:
        errors.append("provenance cache resolution is not 64")
    source = result.get("source", {})
    if source.get("repository") != expected_repository:
        errors.append("provenance repository is not the pinned Object repository")
    if source.get("revision") != expected_revision:
        errors.append("provenance revision is not the pinned Object revision")
    if source.get("content_hashes_match") is not True:
        errors.append("canonical cache and source content hashes differ")
    metadata = result.get("metadata", {})
    if metadata.get("repository") != expected_repository:
        errors.append("metadata repository is not the pinned Object repository")
    if metadata.get("revision") != expected_revision:
        errors.append("metadata revision is not the pinned Object revision")
    if metadata.get("sha256") != FROZEN_DATASET_METADATA_SHA256:
        errors.append("metadata SHA-256 differs from the frozen Object metadata")
    if metadata.get("language_set_matches_official") is not True:
        errors.append("metadata language set does not match official Object tasks")
    task_map = metadata.get("dataset_to_official_task", {})
    if task_map != FROZEN_DATASET_TO_OFFICIAL_TASK:
        errors.append("metadata task permutation differs from the frozen Object map")
    comparison = result.get("comparison", {})
    if int(comparison.get("cached_frames_compared", -1)) != 66984:
        errors.append("provenance did not compare every Object frame")
    mismatch_counts = comparison.get("mismatch_counts", {})
    if not mismatch_counts or any(int(value) != 0 for value in mismatch_counts.values()):
        errors.append("provenance mismatch counts are absent or nonzero")
    if not cache.is_file() or file_sha256(cache) != expected_cache_sha256:
        errors.append("live cache is absent or differs from its frozen SHA")
    if errors:
        raise RuntimeError("Invalid Object provenance: " + "; ".join(errors))
    return result


def validate_cache_task_metadata(metadata: dict[str, Any]) -> None:
    expected_repository, expected_revision = DATASETS["libero_object"]
    expected = {
        "repository": expected_repository,
        "revision": expected_revision,
        "metadata_file": "meta/tasks.parquet",
        "metadata_sha256": FROZEN_DATASET_METADATA_SHA256,
        "dataset_to_official_task": FROZEN_DATASET_TO_OFFICIAL_TASK,
        "ordering_matches_official": False,
        "language_set_matches_official": True,
    }
    if metadata != expected:
        raise RuntimeError("Pinned Object task metadata differs from the frozen identity")


def load_cache_and_statistics(
    cache: Path,
    horizon: int,
) -> tuple[list[Any] | tuple[Any, ...], dict[str, Any]]:
    with cache.open("rb") as handle:
        frames = pickle.load(handle)
    if not isinstance(frames, (list, tuple)) or len(frames) != 66984:
        raise RuntimeError("Frozen Object cache must contain exactly 66,984 records")
    episodes: dict[int, list[Any]] = defaultdict(list)
    for record in frames:
        if not isinstance(record, (list, tuple)) or len(record) != 6:
            raise RuntimeError("Cache contains a noncanonical record")
        episodes[int(record[0])].append(record)
    actions = []
    states = []
    for episode_frames in episodes.values():
        episode_frames.sort(key=lambda item: int(item[1]))
        for index in range(len(episode_frames) - horizon):
            actions.append(
                np.stack(
                    [episode_frames[index + offset][4] for offset in range(horizon)]
                ).astype(np.float32)
            )
            states.append(np.asarray(episode_frames[index][3], dtype=np.float32))
    if not actions:
        raise RuntimeError("No overlapping action chunks were reconstructed")
    action_array = np.stack(actions)
    state_array = np.stack(states)
    stats = {
        "frame_count": len(frames),
        "sample_count": len(actions),
        "action_mean": action_array.mean((0, 1)),
        "action_std": action_array.std((0, 1)) + 1e-6,
        "state_mean": state_array.mean(0),
        "state_std": state_array.std(0) + 1e-6,
        "state_dim": int(state_array.shape[-1]),
        "action_dim": int(action_array.shape[-1]),
    }
    return frames, stats


def expected_rational_site_names() -> list[str]:
    names = [f"vision.norms.{index}" for index in range(3)]
    names.append("vision.norm_out")
    for block in range(8):
        prefix = f"backbone.blocks.{block}"
        names.append(f"{prefix}.rbn_attn")
        names.extend(
            [
                f"{prefix}.attn.rn_q1",
                f"{prefix}.attn.rn_k1",
                f"{prefix}.attn.rn_q2",
                f"{prefix}.attn.rn_k2",
            ]
        )
        names.append(f"{prefix}.rbn_ffn")
    names.append("norm_out")
    return names


def denominator_certificate(module: RationalNorm) -> dict[str, Any]:
    if module.variant != "pade" or int(module.deg) != 2:
        raise RuntimeError("Certificate requires the deployed degree-2 Pade RationalNorm")
    pa = module.pa.detach().double().cpu().numpy()
    pb = module.pb.detach().double().cpu().numpy()
    if pa.shape != (3,) or pb.shape != (3,):
        raise RuntimeError("Rational coefficient shape differs from the frozen degree")
    roots = np.roots(pb[::-1])
    root_rows = [
        {"real": float(root.real), "imag": float(root.imag)} for root in roots
    ]
    positive_real_roots = [
        root
        for root in roots
        if abs(float(root.imag)) <= 1e-12 and float(root.real) >= 0.0
    ]
    coefficients_strictly_positive = bool(np.all(pb > 0.0))
    return {
        "numerator_coefficients_float64": pa.tolist(),
        "denominator_coefficients_float64": pb.tolist(),
        "denominator_roots_float64": root_rows,
        "positive_real_root_count": len(positive_real_roots),
        "positive_half_line_proof": (
            "Every denominator coefficient is strictly positive, so Q(v)>0 for all v>=0."
            if coefficients_strictly_positive
            else "Strictly positive coefficient proof is unavailable."
        ),
        "coefficients_strictly_positive": coefficients_strictly_positive,
        "certificate_pass": coefficients_strictly_positive and not positive_real_roots,
    }


def rational_scales(
    x: torch.Tensor,
    module: RationalNorm,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xf32 = x.float()
    ms32 = xf32.pow(2).mean(dim=-1, keepdim=True) + float(module.eps)
    s032 = module.running_ms.float().clamp_min(1e-12)
    v32 = ms32 / s032
    p32 = sum(module.pa[index] * v32**index for index in range(module.deg + 1))
    q32 = sum(module.pb[index] * v32**index for index in range(module.deg + 1))
    scale32 = (p32 / q32) * torch.rsqrt(s032)

    xf64 = x.double()
    ms64 = xf64.pow(2).mean(dim=-1, keepdim=True) + float(module.eps)
    s064 = module.running_ms.double().clamp_min(1e-12)
    v64 = ms64 / s064
    pa64 = module.pa.double()
    pb64 = module.pb.double()
    p64 = sum(pa64[index] * v64**index for index in range(module.deg + 1))
    q64 = sum(pb64[index] * v64**index for index in range(module.deg + 1))
    scale64 = (p64 / q64) * torch.rsqrt(s064)
    exact_rms_scale64 = torch.rsqrt(ms64)
    return v64, q64, scale32.double(), scale64, exact_rms_scale64


class SiteAccumulator:
    def __init__(self, name: str, module: RationalNorm):
        self.name = name
        self.module = module
        self.certificate = denominator_certificate(module)
        self.invocations = 0
        self.rows = 0
        self.in_range_rows = 0
        self.v_min = math.inf
        self.v_max = -math.inf
        self.q_min = math.inf
        self.q_max = -math.inf
        self.local_error_chunks: list[np.ndarray] = []
        self.local_error_at_or_below_threshold_rows = 0
        self.numeric_error_sumsq = 0.0
        self.numeric_error_max = 0.0
        self.nonfinite_rows = 0

    @torch.inference_mode()
    def update(self, x: torch.Tensor) -> None:
        v64, q64, scale32, scale64, exact64 = rational_scales(x, self.module)
        local_error = torch.abs(scale64 / exact64 - 1.0)
        numeric_error = torch.abs(scale32 / scale64 - 1.0)
        finite = (
            torch.isfinite(v64)
            & torch.isfinite(q64)
            & torch.isfinite(scale32)
            & torch.isfinite(scale64)
            & torch.isfinite(exact64)
            & torch.isfinite(local_error)
            & torch.isfinite(numeric_error)
        )
        row_count = int(v64.numel())
        finite_count = int(finite.sum().item())
        self.invocations += 1
        self.rows += row_count
        self.nonfinite_rows += row_count - finite_count
        if finite_count != row_count:
            raise RuntimeError(f"Nonfinite rational arithmetic at site {self.name}")
        self.in_range_rows += int(((v64 >= 0.1) & (v64 <= 10.0)).sum().item())
        self.v_min = min(self.v_min, float(v64.min().item()))
        self.v_max = max(self.v_max, float(v64.max().item()))
        self.q_min = min(self.q_min, float(q64.min().item()))
        self.q_max = max(self.q_max, float(q64.max().item()))
        numeric = numeric_error.reshape(-1)
        self.numeric_error_sumsq += float(torch.sum(numeric * numeric).item())
        self.numeric_error_max = max(
            self.numeric_error_max,
            float(numeric.max().item()),
        )
        self.local_error_at_or_below_threshold_rows += int(
            (
                local_error
                <= FROZEN_GATES["secondary_local_scale_relative_error_threshold"]
            )
            .sum()
            .item()
        )
        self.local_error_chunks.append(
            local_error.detach().float().cpu().numpy().reshape(-1).copy()
        )

    def finalize(self) -> tuple[dict[str, Any], np.ndarray]:
        if self.rows <= 0 or not self.local_error_chunks:
            raise RuntimeError(f"Rational site {self.name} was never observed")
        errors = np.concatenate(self.local_error_chunks)
        if len(errors) != self.rows:
            raise RuntimeError(f"Rational site {self.name} row accounting differs")
        quantiles = np.quantile(errors, [0.5, 0.95, 0.99])
        fraction = self.in_range_rows / self.rows
        p99 = float(quantiles[2])
        local_error_threshold = FROZEN_GATES[
            "secondary_local_scale_relative_error_threshold"
        ]
        rows_local_error_at_or_below_threshold = (
            self.local_error_at_or_below_threshold_rows
        )
        fraction_local_error_at_or_below_threshold = (
            rows_local_error_at_or_below_threshold / self.rows
        )
        result = {
            "name": self.name,
            "variant": self.module.variant,
            "degree": int(self.module.deg),
            "eps": float(self.module.eps),
            "running_ms": float(self.module.running_ms.detach().double().cpu().item()),
            "initialized": bool(self.module.initialized.detach().cpu().item()),
            "evaluator_frozen_flag": bool(self.module.frozen),
            "invocations": self.invocations,
            "normalization_rows": self.rows,
            "nonfinite_rows": self.nonfinite_rows,
            "observed_v_min": self.v_min,
            "observed_v_max": self.v_max,
            "observed_denominator_min": self.q_min,
            "observed_denominator_max": self.q_max,
            "rows_v_in_0p1_10": self.in_range_rows,
            "fraction_rows_v_in_0p1_10": fraction,
            "local_scale_relative_error_p50": float(quantiles[0]),
            "local_scale_relative_error_p95": float(quantiles[1]),
            "local_scale_relative_error_p99": p99,
            "local_scale_relative_error_max": float(errors.max(initial=0.0)),
            "local_scale_relative_error_threshold": local_error_threshold,
            "rows_local_scale_error_at_or_below_threshold": (
                rows_local_error_at_or_below_threshold
            ),
            "fraction_rows_local_scale_error_at_or_below_threshold": (
                fraction_local_error_at_or_below_threshold
            ),
            "fp32_vs_fp64_scale_relative_error_rms": math.sqrt(
                self.numeric_error_sumsq / self.rows
            ),
            "fp32_vs_fp64_scale_relative_error_max": self.numeric_error_max,
            "denominator_certificate": self.certificate,
            "primary_denominator_pass": (
                bool(self.certificate["certificate_pass"])
                and self.q_min > 0.0
                and self.nonfinite_rows == 0
            ),
            "secondary_range_pass": fraction
            >= FROZEN_GATES[
                "secondary_each_site_fraction_rows_v_in_0p1_10_at_least"
            ],
            "secondary_local_error_pass": fraction_local_error_at_or_below_threshold
            >= FROZEN_GATES[
                "secondary_each_site_fraction_local_scale_error_at_or_below_threshold_at_least"
            ],
        }
        self.local_error_chunks = []
        return result, errors


class RationalCollector:
    def __init__(self, modules: dict[str, RationalNorm]):
        self.enabled = False
        self.sites = {
            name: SiteAccumulator(name, module) for name, module in modules.items()
        }
        self.handles = []
        for name, module in modules.items():
            self.handles.append(
                module.register_forward_pre_hook(self._make_hook(name))
            )

    def _make_hook(self, name: str):
        def hook(_module: RationalNorm, inputs: tuple[torch.Tensor, ...]) -> None:
            if self.enabled:
                if len(inputs) != 1:
                    raise RuntimeError(f"Unexpected RationalNorm input arity at {name}")
                self.sites[name].update(inputs[0])

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class Float64RationalReference:
    def __init__(self, modules: dict[str, RationalNorm]):
        self.enabled = False
        self.handles = [
            module.register_forward_hook(self._make_hook(name))
            for name, module in modules.items()
        ]

    def _make_hook(self, name: str):
        def hook(
            module: RationalNorm,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> torch.Tensor | None:
            if not self.enabled:
                return None
            if len(inputs) != 1:
                raise RuntimeError(f"Unexpected RationalNorm input arity at {name}")
            _v, _q, _scale32, scale64, _exact64 = rational_scales(inputs[0], module)
            reference = inputs[0].double() * scale64
            if not torch.isfinite(reference).all():
                raise RuntimeError(f"Nonfinite float64 rational reference at {name}")
            return reference.to(dtype=output.dtype)

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def tensorize_batch(
    frames: list[Any] | tuple[Any, ...],
    selected_records: list[dict[str, Any]],
    cache_tasks: dict[int, str],
    encode: Any,
    stats: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    records = [frames[int(row["cache_offset"])] for row in selected_records]
    images = np.stack([np.asarray(record[2], dtype=np.uint8) for record in records])
    states = np.stack([np.asarray(record[3], dtype=np.float32) for record in records])
    tasks = [int(record[5]) for record in records]
    if images.shape[1:] != (64, 64, 3):
        raise RuntimeError(f"Unexpected cached image shape {images.shape}")
    if not np.isfinite(states).all():
        raise RuntimeError("Selected cache states are nonfinite")
    image_tensor = (
        torch.from_numpy(images)
        .permute(0, 3, 1, 2)
        .float()
        .div(255.0)
        .cuda(non_blocking=True)
    )
    instruction_tensor = torch.tensor(
        [encode(cache_tasks[task]) for task in tasks],
        dtype=torch.long,
        device="cuda",
    )
    state_tensor = torch.tensor(
        (states - stats["state_mean"]) / stats["state_std"],
        dtype=torch.float32,
        device="cuda",
    )
    embodiment_tensor = torch.zeros(len(records), dtype=torch.long, device="cuda")
    return image_tensor, instruction_tensor, state_tensor, embodiment_tensor


def update_array_digest(digest: Any, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.astype("<f8", copy=False).tobytes())


def encode_float32_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype="<f4")
    return base64.b64encode(contiguous.tobytes()).decode("ascii")


def imported_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "athena/launch_rational_norm_safety.sh",
        "athena/run_rational_norm_safety.py",
        "athena/summarize_rational_norm_safety.py",
        "athena/slurm_rational_norm_safety.sbatch",
        "athena/slurm_summarize_rational_norm_safety.sbatch",
        "athena/run_xvla_experiment.py",
        "athena/libero_dataset_metadata.py",
        "xvla/nn/normalization.py",
        "xvla/nn/attention.py",
        "xvla/nn/bilinear.py",
        "xvla/nn/block.py",
        "xvla/models/vit.py",
        "xvla/models/vla.py",
    )
    if len(paths) != len(set(paths)):
        raise RuntimeError("Rational certificate source catalog contains a duplicate path")
    return {path: file_sha256(root / path) for path in paths}


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    source_hashes_at_start = imported_source_hashes()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.mode == "strict_smoke":
        if args.sample_count != PROTOCOL["smoke_samples"]:
            raise ValueError("Strict smoke must use exactly one frozen 32-row batch")
    elif args.sample_count != PROTOCOL["full_samples_per_checkpoint"]:
        raise ValueError("Full certificate must use exactly 4,096 selected samples")
    if args.batch_size != PROTOCOL["batch_size"]:
        raise ValueError("Certificate batch size is frozen at 32")
    if args.res != 64 or args.horizon != 8:
        raise ValueError("Certificate requires resolution 64 and horizon 8")
    if args.matmul_precision != "highest":
        raise ValueError("Certificate requires highest float32 matmul precision")
    if args.checkpoint_seed not in FROZEN_CHECKPOINTS:
        raise ValueError("Checkpoint seed must be 0, 1, or 2")
    expected_checkpoint = FROZEN_CHECKPOINTS[args.checkpoint_seed]
    if args.checkpoint.name != expected_checkpoint["basename"]:
        raise RuntimeError("Checkpoint basename differs from the frozen seed identity")
    if args.expected_checkpoint_sha256 != expected_checkpoint["sha256"]:
        raise RuntimeError("Declared checkpoint SHA-256 differs from the frozen seed identity")
    if args.cache.name != FROZEN_CACHE["basename"]:
        raise RuntimeError("Cache basename differs from the frozen Object cache")
    if args.expected_cache_sha256 != FROZEN_CACHE["sha256"]:
        raise RuntimeError("Declared cache SHA-256 differs from the frozen Object cache")
    if str(args.expected_provenance_job_id) != FROZEN_PROVENANCE_JOB_ID:
        raise RuntimeError("Declared provenance job differs from the frozen job identity")
    if args.provenance_result.name != "cache_provenance_libero_object.json":
        raise RuntimeError("Provenance result basename differs from the frozen artifact")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    live_checkpoint_sha = file_sha256(args.checkpoint)
    if live_checkpoint_sha != args.expected_checkpoint_sha256:
        raise RuntimeError("Live checkpoint differs from its frozen SHA-256")
    provenance = validate_provenance(
        args.provenance_result,
        args.cache,
        args.expected_cache_sha256,
    )
    provenance_result_sha256 = file_sha256(args.provenance_result)

    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    np.random.seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("The rational safety runner requires a CUDA compute node")

    suite = load_suite("libero_object")
    official_tasks = task_languages(suite)
    cache_tasks, cache_task_metadata = load_dataset_task_languages(
        "libero_object", official_tasks
    )
    validate_cache_task_metadata(cache_task_metadata)
    vocab, _ = build_vocab(official_tasks)
    encode = build_encoder(vocab)
    print("Loading frozen Object cache and normalization statistics", flush=True)
    frames, stats = load_cache_and_statistics(args.cache, args.horizon)
    selection = select_cache_samples(
        frames,
        args.expected_checkpoint_sha256,
        args.sample_count,
    )

    config = make_config(
        "chi",
        len(vocab),
        stats["state_dim"],
        stats["action_dim"],
        args.res,
        args.horizon,
        vision_encoder="conv",
    )
    model = ChiVLA(config).cuda().eval()
    state_dict = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, RationalNorm)
    }
    expected_sites = expected_rational_site_names()
    if list(modules) != expected_sites:
        raise RuntimeError(
            "RationalNorm site catalog differs from the frozen 53-site architecture: "
            f"actual={list(modules)}"
        )
    if len(modules) != 53:
        raise RuntimeError("Frozen convolutional policy must contain 53 RationalNorm sites")
    for name, module in modules.items():
        module.freeze()
        if module.training or module.variant != "pade" or int(module.deg) != 2:
            raise RuntimeError(f"Unexpected evaluator normalizer configuration at {name}")
        if not torch.isfinite(module.running_ms) or float(module.running_ms.item()) <= 0.0:
            raise RuntimeError(f"Invalid running mean-square at {name}")
        if not bool(module.initialized.item()) or not module.frozen:
            raise RuntimeError(
                f"Rational normalizer is not initialized and evaluator-frozen at {name}"
            )

    collector = RationalCollector(modules)
    reference = Float64RationalReference(modules)
    baseline_normalized_chunks: list[np.ndarray] = []
    reference_normalized_chunks: list[np.ndarray] = []
    batch_count = math.ceil(args.sample_count / args.batch_size)
    try:
        with torch.inference_mode():
            for batch_index in range(batch_count):
                start = batch_index * args.batch_size
                end = min(args.sample_count, start + args.batch_size)
                batch_records = selection["records"][start:end]
                tensors = tensorize_batch(
                    frames, batch_records, cache_tasks, encode, stats
                )
                collector.enabled = True
                reference.enabled = False
                baseline = model(*tensors)[0]
                collector.enabled = False
                reference.enabled = True
                fp64_reference = model(*tensors)[0]
                reference.enabled = False
                if baseline.shape != fp64_reference.shape or baseline.shape[1:] != (8, 7):
                    raise RuntimeError("Action prediction shape differs from the frozen contract")
                baseline_normalized = baseline.detach().float().cpu().numpy().copy()
                reference_normalized = (
                    fp64_reference.detach().float().cpu().numpy().copy()
                )
                if not (
                    np.isfinite(baseline_normalized).all()
                    and np.isfinite(reference_normalized).all()
                ):
                    raise RuntimeError("Action output contains nonfinite values")
                baseline_normalized_chunks.append(baseline_normalized)
                reference_normalized_chunks.append(reference_normalized)
                print(
                    f"CERTIFICATE seed={args.checkpoint_seed} batch "
                    f"{batch_index + 1}/{batch_count}",
                    flush=True,
                )
    finally:
        collector.enabled = False
        reference.enabled = False
        collector.close()
        reference.close()

    baseline_normalized_actions = np.concatenate(
        baseline_normalized_chunks, axis=0
    ).astype(np.float32, copy=False)
    reference_normalized_actions = np.concatenate(
        reference_normalized_chunks, axis=0
    ).astype(np.float32, copy=False)
    expected_action_shape = (args.sample_count, 8, 7)
    if (
        baseline_normalized_actions.shape != expected_action_shape
        or reference_normalized_actions.shape != expected_action_shape
    ):
        raise RuntimeError("Stored normalized action array shape differs")
    baseline64 = baseline_normalized_actions.astype(np.float64)
    reference64 = reference_normalized_actions.astype(np.float64)
    normalized_difference = baseline64 - reference64
    normalized_action_diff_sumsq = float(
        np.sum(normalized_difference * normalized_difference)
    )
    normalized_action_reference_sumsq = float(np.sum(reference64 * reference64))
    action_mean = np.asarray(stats["action_mean"], dtype=np.float64)
    action_std = np.asarray(stats["action_std"], dtype=np.float64)
    baseline_physical = baseline64 * action_std + action_mean
    reference_physical = reference64 * action_std + action_mean
    physical_difference = baseline_physical - reference_physical
    action_diff_sumsq = float(np.sum(physical_difference * physical_difference))
    action_reference_sumsq = float(np.sum(reference_physical * reference_physical))
    action_max_abs_difference = float(
        np.max(np.abs(physical_difference), initial=0.0)
    )
    action_values = int(physical_difference.size)
    baseline_digest = hashlib.sha256()
    reference_digest = hashlib.sha256()
    update_array_digest(baseline_digest, baseline_physical)
    update_array_digest(reference_digest, reference_physical)

    site_results = []
    total_rows = sum(site.rows for site in collector.sites.values())
    all_local_errors = np.empty(total_rows, dtype=np.float32)
    cursor = 0
    for name in expected_sites:
        site_result, errors = collector.sites[name].finalize()
        if site_result["invocations"] != batch_count:
            raise RuntimeError(f"Rational site {name} was not invoked once per batch")
        next_cursor = cursor + len(errors)
        all_local_errors[cursor:next_cursor] = errors
        cursor = next_cursor
        site_results.append(site_result)
    if cursor != total_rows:
        raise RuntimeError("Aggregate rational row accounting differs")
    aggregate_p99 = float(np.quantile(all_local_errors, 0.99))
    aggregate_error_max = float(all_local_errors.max(initial=0.0))
    del all_local_errors
    total_in_range = sum(int(site["rows_v_in_0p1_10"]) for site in site_results)
    total_local_error_at_or_below_threshold = sum(
        int(site["rows_local_scale_error_at_or_below_threshold"])
        for site in site_results
    )
    total_nonfinite = sum(int(site["nonfinite_rows"]) for site in site_results)
    range_fraction = total_in_range / total_rows
    local_error_at_or_below_threshold_fraction = (
        total_local_error_at_or_below_threshold / total_rows
    )
    physical_action_nrmse = math.sqrt(
        action_diff_sumsq / max(action_reference_sumsq, 1e-300)
    )
    normalized_action_nrmse = math.sqrt(
        normalized_action_diff_sumsq
        / max(normalized_action_reference_sumsq, 1e-300)
    )
    action_nrmse = max(physical_action_nrmse, normalized_action_nrmse)
    primary_denominator_pass = all(
        bool(site["primary_denominator_pass"]) for site in site_results
    )
    primary_action_pass = action_nrmse <= FROZEN_GATES[
        "primary_action_nrmse_at_most"
    ]
    secondary_range_pass = all(
        bool(site["secondary_range_pass"]) for site in site_results
    )
    secondary_local_error_pass = all(
        bool(site["secondary_local_error_pass"]) for site in site_results
    )

    source_hashes_at_end = imported_source_hashes()
    if source_hashes_at_end != source_hashes_at_start:
        raise RuntimeError("Runner or imported source changed during certificate execution")
    if file_sha256(args.checkpoint) != live_checkpoint_sha:
        raise RuntimeError("Checkpoint changed during certificate execution")
    if file_sha256(args.cache) != args.expected_cache_sha256:
        raise RuntimeError("Cache changed during certificate execution")
    if file_sha256(args.provenance_result) != provenance_result_sha256:
        raise RuntimeError("Provenance result changed during certificate execution")
    result = {
        "schema": SCHEMA,
        "mode": args.mode,
        "protocol": PROTOCOL,
        "frozen_gates": FROZEN_GATES,
        "checkpoint": {
            "path": str(args.checkpoint),
            "basename": args.checkpoint.name,
            "seed": args.checkpoint_seed,
            "sha256": live_checkpoint_sha,
        },
        "cache": {
            "path": str(args.cache),
            "sha256": args.expected_cache_sha256,
            "frames": len(frames),
            "normalization_samples": int(stats["sample_count"]),
            "provenance_result": str(args.provenance_result),
            "provenance_result_sha256": provenance_result_sha256,
            "expected_provenance_job_id": str(args.expected_provenance_job_id),
            "canonical_content_sha256": provenance["cache"][
                "canonical_content_sha256"
            ],
            "source_repository": provenance["source"]["repository"],
            "source_revision": provenance["source"]["revision"],
            "source_content_sha256": provenance["source"][
                "canonical_content_sha256"
            ],
            "content_hashes_match": provenance["source"][
                "content_hashes_match"
            ],
            "cache_task_metadata": cache_task_metadata,
        },
        "selection": selection,
        "model": {
            "architecture": "chi",
            "vision_encoder": "conv",
            "rational_site_count": len(site_results),
            "rational_site_names": expected_sites,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "rational_sites": site_results,
        "aggregate": {
            "sample_count": args.sample_count,
            "batch_size": args.batch_size,
            "batch_count": batch_count,
            "normalization_rows": total_rows,
            "nonfinite_rows": total_nonfinite,
            "rows_v_in_0p1_10": total_in_range,
            "fraction_rows_v_in_0p1_10": range_fraction,
            "rows_local_scale_error_at_or_below_threshold": (
                total_local_error_at_or_below_threshold
            ),
            "fraction_rows_local_scale_error_at_or_below_threshold": (
                local_error_at_or_below_threshold_fraction
            ),
            "observed_v_min": min(site["observed_v_min"] for site in site_results),
            "observed_v_max": max(site["observed_v_max"] for site in site_results),
            "observed_denominator_min": min(
                site["observed_denominator_min"] for site in site_results
            ),
            "local_scale_relative_error_p99": aggregate_p99,
            "local_scale_relative_error_max": aggregate_error_max,
            "all_sites_secondary_range_pass": all(
                bool(site["secondary_range_pass"]) for site in site_results
            ),
            "all_sites_secondary_local_error_pass": all(
                bool(site["secondary_local_error_pass"]) for site in site_results
            ),
        },
        "action_reference": {
            "description": PROTOCOL["reference"],
            "physical_action_values": action_values,
            "physical_squared_difference_sum": action_diff_sumsq,
            "physical_squared_reference_sum": action_reference_sumsq,
            "physical_nrmse": physical_action_nrmse,
            "normalized_squared_difference_sum": normalized_action_diff_sumsq,
            "normalized_squared_reference_sum": normalized_action_reference_sumsq,
            "normalized_nrmse": normalized_action_nrmse,
            "primary_nrmse": action_nrmse,
            "max_absolute_difference": action_max_abs_difference,
            "normalized_action_array_shape": list(expected_action_shape),
            "normalized_action_array_dtype": "<f4",
            "normalized_action_array_encoding": "base64 contiguous C-order bytes",
            "baseline_normalized_actions_float32_base64": encode_float32_array(
                baseline_normalized_actions
            ),
            "fp64_rational_reference_normalized_actions_float32_base64": (
                encode_float32_array(reference_normalized_actions)
            ),
            "baseline_physical_actions_sha256": baseline_digest.hexdigest(),
            "fp64_rational_reference_physical_actions_sha256": (
                reference_digest.hexdigest()
            ),
        },
        "gates": {
            "primary_denominator_certificate_pass": primary_denominator_pass,
            "primary_action_nrmse_pass": primary_action_pass,
            "primary_pass": primary_denominator_pass and primary_action_pass,
            "secondary_range_pass": secondary_range_pass,
            "secondary_local_scale_error_pass": secondary_local_error_pass,
            "secondary_pass": secondary_range_pass and secondary_local_error_pass,
        },
        "implementation": {
            "source_sha256": source_hashes_at_start,
            "python_version": os.sys.version,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "datasets_version": importlib.metadata.version("datasets"),
        },
        "scope": (
            "This is a numerical-safety certificate for the deployed degree-2 Pade "
            "RationalNorm sites and their end-to-end arithmetic effect on three frozen "
            "capable convolutional checkpoints. The evaluator sets each loaded site to "
            "its frozen evaluation state before measurement. This does not assert that the "
            "Python-only frozen flag was serialized in the checkpoint. It does not measure "
            "task success, prove global whole-network rational simplification, or compare "
            "normalization architectures."
        ),
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(args.output, result)
    print("RESULT", json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not result["gates"]["primary_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
