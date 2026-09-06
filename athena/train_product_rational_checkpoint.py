#!/usr/bin/env python3
"""Train and export one exact ProductRoutingHead + Padé-rational VLA checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_RUNNING_AS_ENTRYPOINT = __name__ == "__main__"

_MANIFEST_SCHEMA = "xvla_product_pade_rational_vla_v2_source_manifest"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant is forbidden: {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Nonfinite JSON float is forbidden")
    return parsed
_EXPECTED_SOURCE_CLOSURE = (
    "athena/__init__.py",
    "athena/aggregate_product_rational_results.py",
    "athena/eval_product_rational_checkpoint.py",
    "athena/prepare_product_rational_launch.py",
    "athena/product_rational_protocol.py",
    "athena/stage_product_rational_launch.py",
    "athena/train_product_rational_checkpoint.py",
    "scripts/__init__.py",
    "scripts/odt_direct_only_compliance.py",
    "tests/test_product_rational_production_lane.py",
    "xvla/__init__.py",
    "xvla/models/__init__.py",
    "xvla/models/lm.py",
    "xvla/models/vit.py",
    "xvla/models/vla.py",
    "xvla/nn/__init__.py",
    "xvla/nn/attention.py",
    "xvla/nn/baselines.py",
    "xvla/nn/bilinear.py",
    "xvla/nn/block.py",
    "xvla/nn/flow_action.py",
    "xvla/nn/homogeneous.py",
    "xvla/nn/normalization.py",
    "xvla/nn/product_routing.py",
    "xvla/nn/projector.py",
    "xvla/nn/quantile_action.py",
)
_EXPECTED_AUTHENTICATED_INPUTS = (
    "athena/results/cache_provenance_libero_object.json",
)
_EXPECTED_LAUNCH_CLOSURE = (
    "athena/slurm_product_rational_aggregate.sbatch",
    "athena/slurm_product_rational_eval.sbatch",
    "athena/slurm_product_rational_preflight.sbatch",
    "athena/slurm_product_rational_train.sbatch",
    "athena/submit_product_rational_v2.sh",
)


def _verify_preimport_source_manifest(arguments: list[str]) -> dict[str, Any]:
    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument == "--source-manifest":
            if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
                raise RuntimeError("--source-manifest requires one value")
            values.append(arguments[index + 1])
        elif argument.startswith("--source-manifest="):
            values.append(argument.split("=", 1)[1])
        elif argument.startswith("--source-man"):
            raise RuntimeError("Abbreviated source-manifest options are forbidden")
    if len(values) != 1 or not values[0]:
        raise RuntimeError("Exactly one --source-manifest is required")
    manifest_path = Path(values[0])
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError("Pre-import source manifest is missing or nonphysical")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"Duplicate source-manifest key {key!r}")
            result[key] = value
        return result

    payload = json.loads(
        manifest_path.read_text(),
        object_pairs_hook=reject_duplicates,
        parse_constant=_reject_json_constant,
        parse_float=_finite_json_float,
    )
    if payload.get("schema") != _MANIFEST_SCHEMA:
        raise RuntimeError("Pre-import source manifest schema differs")

    def canonical_digest(value: Any) -> str:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    observed_sections: dict[str, dict[str, str]] = {}
    section_specs = (
        ("source_closure", _EXPECTED_SOURCE_CLOSURE, "source_bundle_sha256"),
        (
            "authenticated_inputs",
            _EXPECTED_AUTHENTICATED_INPUTS,
            "authenticated_inputs_bundle_sha256",
        ),
        ("launch_closure", _EXPECTED_LAUNCH_CLOSURE, "launch_bundle_sha256"),
    )
    for section, exact_paths, bundle_key in section_specs:
        expected = payload.get(section)
        if not isinstance(expected, dict) or tuple(sorted(expected)) != tuple(
            sorted(exact_paths)
        ):
            raise RuntimeError(f"Pre-import source manifest {section} differs")
        if payload.get(bundle_key) != canonical_digest(dict(sorted(expected.items()))):
            raise RuntimeError(f"Pre-import source manifest {bundle_key} differs")
        observed: dict[str, str] = {}
        for relative, expected_digest in expected.items():
            if (
                not isinstance(relative, str)
                or not isinstance(expected_digest, str)
                or len(expected_digest) != 64
                or any(char not in "0123456789abcdef" for char in expected_digest)
            ):
                raise RuntimeError("Pre-import source manifest contains an invalid entry")
            candidate = PROJECT_ROOT / relative
            if candidate.is_symlink():
                raise RuntimeError(f"Pre-import lane file is a link: {relative}")
            path = candidate.resolve()
            if path == PROJECT_ROOT or PROJECT_ROOT not in path.parents:
                raise RuntimeError("Pre-import source path escapes the project root")
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"Pre-import lane file is missing or nonphysical: {relative}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected_digest:
                raise RuntimeError(f"Pre-import lane file hash changed: {relative}")
            observed[relative] = digest
        observed_sections[section] = observed
    bound = {
        "schema": payload.get("schema"),
        "source_closure": payload.get("source_closure"),
        "source_bundle_sha256": payload.get("source_bundle_sha256"),
        "authenticated_inputs": payload.get("authenticated_inputs"),
        "authenticated_inputs_bundle_sha256": payload.get(
            "authenticated_inputs_bundle_sha256"
        ),
        "launch_closure": payload.get("launch_closure"),
        "launch_bundle_sha256": payload.get("launch_bundle_sha256"),
    }
    if payload.get("manifest_bundle_sha256") != canonical_digest(bound):
        raise RuntimeError("Pre-import aggregate manifest digest differs")
    return {
        "path": manifest_path.as_posix(),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_sha256": observed_sections["source_closure"],
        "authenticated_inputs_sha256": observed_sections["authenticated_inputs"],
        "launch_sha256": observed_sections["launch_closure"],
        "manifest_bundle_sha256": payload["manifest_bundle_sha256"],
    }


_PREIMPORT_SOURCE_VERIFICATION = (
    _verify_preimport_source_manifest(sys.argv[1:]) if _RUNNING_AS_ENTRYPOINT else None
)

if _RUNNING_AS_ENTRYPOINT:
    from scripts.odt_direct_only_compliance import (
        assert_direct_only_runtime_guard,
        audit_direct_only_launch,
        direct_only_runtime_report,
        install_direct_only_runtime_guard,
    )

    _TRANSITIVE_STATIC_AUDIT = audit_direct_only_launch(
        PROJECT_ROOT,
        tuple(PROJECT_ROOT / path for path in _EXPECTED_SOURCE_CLOSURE),
        require_direct_qr=False,
    )
    if _TRANSITIVE_STATIC_AUDIT["source_sha256"] != _PREIMPORT_SOURCE_VERIFICATION[
        "source_sha256"
    ]:
        raise RuntimeError("Authenticated Python closure differs from transitive static audit")
    for field in (
        "prohibited_calls_found",
        "prohibited_self_overlap_sites",
        "guarded_dormant_spectral_norm_sites",
        "duplicate_top_level_definition_sites",
    ):
        if _TRANSITIVE_STATIC_AUDIT[field] != []:
            raise RuntimeError(f"Product lane static audit did not close {field}")
    _RUNTIME_GUARD_AT_IMPORT = install_direct_only_runtime_guard()
else:
    _TRANSITIVE_STATIC_AUDIT = None
    _RUNTIME_GUARD_AT_IMPORT = None

import numpy as np
import torch

from athena.product_rational_protocol import (
    ACTION_DIM,
    ACTION_HORIZON,
    CACHE_FRAME_COUNT,
    CACHE_PATH,
    CACHE_SAMPLE_COUNT,
    CALIBRATION_INITIALIZER_LABEL,
    CALIBRATION_RECOVERY_SCOPE,
    CALIBRATION_RESUME_PROOF_SCHEMA,
    DEPLOYMENT_CLAIM_BOUNDARY,
    EMA_DECAY,
    FINAL_CALIBRATION_AUDIT_MAX_RELATIVE_CHANGE,
    FINAL_CALIBRATION_BATCH_SIZE,
    FINAL_CALIBRATION_PASSES,
    FULL_BATCH_SIZE,
    FULL_STEPS,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    MIN_LEARNING_RATE_RATIO,
    PRODUCT_RUNNING_MS_IDENTITY_FLOOR,
    PRODUCT_COMPONENT_SITE_NAMES,
    RECIPE_VERSION,
    SCHEMA,
    SEEDS,
    SMOKE_BATCH_SIZE,
    SMOKE_STEPS,
    STATE_DIM,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
    active_rational_norms,
    active_running_ms_values,
    assert_cache_identity,
    assert_outputs_absent,
    build_vocab,
    calibration_state_transition_proof,
    calibration_resume_proof_path,
    canonical_sha256,
    checkpoint_path,
    deployment_completion_path,
    config_record,
    deployment_state_from_training,
    file_sha256,
    json_type_exact_equal,
    load_and_verify_source_manifest,
    load_and_validate_deployment_completion,
    load_training_only_precalibration_state,
    load_suite,
    make_product_config,
    metadata_path,
    install_product_running_ms_fail_closed_guards,
    precalibration_state_path,
    preflight_result_path,
    publish_checkpoint_exclusive,
    publish_deployment_completion_exclusive,
    publish_training_only_precalibration_state_exclusive,
    read_json_object_physical,
    source_bundle_sha256,
    source_snapshot,
    strict_load_deployment_state,
    task_languages,
    tensor_state_sha256,
    training_recipe,
    training_result_path,
    validate_product_model,
    validate_same_allocation_calibration_resume_proof,
    validate_simultaneous_calibration_attestation,
    validate_preflight_certificate,
    validate_seed,
    validate_task_mapping,
    write_json_exclusive,
)
from xvla.models.vla import ChiVLA


def _option_count(arguments: list[str], option: str) -> int:
    return sum(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--preflight-result", type=Path)
    parser.add_argument("--resume-precalibration", action="store_true")
    parser.add_argument("--precalibration-sha256")
    parser.add_argument("--resume-proof-only", action="store_true")
    arguments = sys.argv[1:]
    parsed = parser.parse_args(arguments)
    for option in ("--run-root", "--source-manifest", "--mode"):
        if _option_count(arguments, option) != 1:
            parser.error(f"{option} must occur exactly once")
    for option in (
        "--seed",
        "--preflight-only",
        "--preflight-output",
        "--preflight-result",
        "--resume-precalibration",
        "--precalibration-sha256",
        "--resume-proof-only",
    ):
        if _option_count(arguments, option) > 1:
            parser.error(f"{option} may occur at most once")
    return parsed


def validate_resume_request(
    *,
    resume_precalibration: bool,
    precalibration_sha256: str | None,
    resume_proof_only: bool = False,
) -> None:
    if resume_precalibration != (precalibration_sha256 is not None):
        raise ValueError(
            "Calibration-only resume requires both --resume-precalibration and "
            "--precalibration-sha256, while fresh training requires neither"
        )
    if precalibration_sha256 is not None and (
        len(precalibration_sha256) != 64
        or any(character not in "0123456789abcdef" for character in precalibration_sha256)
    ):
        raise ValueError("Calibration-only resume SHA-256 must be lowercase hexadecimal")
    if resume_proof_only and not resume_precalibration:
        raise ValueError("Resume proof requires the calibration-only resume authority")


def slurm_gpu_allocation_record(
    *, gpu: str, compute_capability: tuple[int, int]
) -> dict[str, Any]:
    return {
        "gpu": gpu,
        "compute_capability": list(compute_capability),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def validate_same_slurm_gpu_allocation(
    current: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    exact_keys = {
        "gpu",
        "compute_capability",
        "slurm_job_id",
        "slurm_job_gpus",
        "cuda_visible_devices",
    }
    if (
        set(current) != exact_keys
        or any(
            not isinstance(current.get(key), str) or not current.get(key)
            for key in ("slurm_job_id", "slurm_job_gpus", "cuda_visible_devices")
        )
        or current.get("gpu") != "NVIDIA RTX A6000"
        or current.get("compute_capability") != [8, 6]
        or any(current.get(key) != reference.get(key) for key in exact_keys)
    ):
        raise RuntimeError("Resume proof did not run in the fresh smoke GPU allocation")


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def verify_transitive_static_audit_now() -> dict[str, Any]:
    if not _RUNNING_AS_ENTRYPOINT:
        raise RuntimeError("Static production audit requires the authenticated entrypoint")
    report = audit_direct_only_launch(
        PROJECT_ROOT,
        tuple(PROJECT_ROOT / path for path in _EXPECTED_SOURCE_CLOSURE),
        require_direct_qr=False,
    )
    if report["source_sha256"] != _PREIMPORT_SOURCE_VERIFICATION["source_sha256"]:
        raise RuntimeError("Authenticated Python source changed after imports")
    for field in (
        "prohibited_calls_found",
        "prohibited_self_overlap_sites",
        "guarded_dormant_spectral_norm_sites",
        "duplicate_top_level_definition_sites",
    ):
        if report[field] != []:
            raise RuntimeError(f"Product lane static audit did not close {field}")
    return report


def learning_rate_at(step: int, steps: int) -> float:
    warmup = max(1, int(WARMUP_FRACTION * steps))
    if step < warmup:
        return LEARNING_RATE * step / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LEARNING_RATE * (
        MIN_LEARNING_RATE_RATIO
        + (1.0 - MIN_LEARNING_RATE_RATIO) * coefficient
    )


def inspect_or_build_training_data(
    frames: list[Any],
    tasks: Mapping[int, str],
    encode: Any,
    *,
    materialize: bool,
) -> dict[str, Any]:
    if len(frames) != CACHE_FRAME_COUNT:
        raise RuntimeError(f"Cache frame count {len(frames)} != {CACHE_FRAME_COUNT}")
    episodes: dict[int, list[Any]] = defaultdict(list)
    task_frame_counts = {task: 0 for task in range(10)}
    for frame in frames:
        if not isinstance(frame, (tuple, list)) or len(frame) < 6:
            raise RuntimeError("Malformed cached frame record")
        episode_id = int(frame[0])
        frame_index = int(frame[1])
        dataset_task = int(frame[5])
        if dataset_task not in tasks:
            raise RuntimeError(f"Cached frame names unknown dataset task {dataset_task}")
        image = np.asarray(frame[2])
        state = np.asarray(frame[3])
        action = np.asarray(frame[4])
        if image.shape != (64, 64, 3) or image.dtype != np.uint8:
            raise RuntimeError(f"Cached image has invalid shape/dtype {image.shape}/{image.dtype}")
        if state.shape != (STATE_DIM,) or action.shape != (ACTION_DIM,):
            raise RuntimeError(f"Cached state/action shapes differ: {state.shape}/{action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise RuntimeError("Cached state or action is non-finite")
        if episode_id < 0 or frame_index < 0:
            raise RuntimeError("Cached episode/frame index is negative")
        episodes[episode_id].append(frame)
        task_frame_counts[dataset_task] += 1

    images: list[np.ndarray] = []
    instructions: list[list[int]] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    sample_count = 0
    task_sample_counts = {task: 0 for task in range(10)}
    for episode_id, episode_frames in episodes.items():
        episode_frames.sort(key=lambda item: int(item[1]))
        episode_tasks = {int(frame[5]) for frame in episode_frames}
        if len(episode_tasks) != 1:
            raise RuntimeError(f"Cache episode {episode_id} crosses dataset tasks")
        observed_indices = [int(frame[1]) for frame in episode_frames]
        if len(observed_indices) != len(set(observed_indices)):
            raise RuntimeError(f"Cache episode {episode_id} repeats a frame index")
        if observed_indices != list(range(observed_indices[0], observed_indices[0] + len(observed_indices))):
            raise RuntimeError(f"Cache episode {episode_id} is not frame-contiguous")
        dataset_task = next(iter(episode_tasks))
        for index in range(len(episode_frames) - ACTION_HORIZON):
            sample_count += 1
            task_sample_counts[dataset_task] += 1
            if not materialize:
                continue
            frame = episode_frames[index]
            images.append(np.asarray(frame[2], dtype=np.uint8))
            instructions.append(encode(tasks[dataset_task]))
            states.append(np.asarray(frame[3], dtype=np.float32))
            actions.append(
                np.stack(
                    [episode_frames[index + offset][4] for offset in range(ACTION_HORIZON)]
                ).astype(np.float32)
            )
    if sample_count != CACHE_SAMPLE_COUNT:
        raise RuntimeError(f"Cache sample count {sample_count} != {CACHE_SAMPLE_COUNT}")
    if any(value == 0 for value in task_sample_counts.values()):
        raise RuntimeError(f"At least one dataset task has no training sample: {task_sample_counts}")
    summary: dict[str, Any] = {
        "frame_count": len(frames),
        "episode_count": len(episodes),
        "sample_count": sample_count,
        "task_frame_counts": {str(key): value for key, value in task_frame_counts.items()},
        "task_sample_counts": {str(key): value for key, value in task_sample_counts.items()},
    }
    if not materialize:
        return summary
    state_array = np.stack(states)
    action_array = np.stack(actions)
    state_mean = state_array.astype(np.float64).mean(axis=0)
    state_std = state_array.astype(np.float64).std(axis=0) + 1e-6
    action_mean = action_array.astype(np.float64).mean(axis=(0, 1))
    action_std = action_array.astype(np.float64).std(axis=(0, 1)) + 1e-6
    summary.update(
        {
            "images": np.stack(images),
            "instructions": np.asarray(instructions, dtype=np.int64),
            "states": ((state_array - state_mean) / state_std).astype(np.float32),
            "actions": ((action_array - action_mean) / action_std).astype(np.float32),
            "normalization": {
                "state_mean": state_mean.tolist(),
                "state_std": state_std.tolist(),
                "action_mean": action_mean.tolist(),
                "action_std": action_std.tolist(),
            },
        }
    )
    return summary


def load_and_validate_cache(
    tasks: Mapping[int, str], encode: Any, *, materialize: bool
) -> dict[str, Any]:
    cache_path = Path(__file__).resolve().parents[1] / CACHE_PATH
    with cache_path.open("rb") as handle:
        frames = pickle.load(handle)
    if not isinstance(frames, list):
        raise RuntimeError("Pinned frame cache is not a list")
    result = inspect_or_build_training_data(frames, tasks, encode, materialize=materialize)
    del frames
    return result


def _tensor_diagnostics(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum().item())
    if finite_count:
        finite_values = detached if finite_count == detached.numel() else detached[finite]
        finite_amax: float | None = float(finite_values.abs().amax().item())
    else:
        finite_amax = None
    return {
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "numel": detached.numel(),
        "finite_count": finite_count,
        "all_finite": finite_count == detached.numel(),
        "finite_amax": finite_amax,
    }


@torch.inference_mode()
def validate_post_ema_training_state(model: ChiVLA) -> dict[str, Any]:
    """Validate every persistent tensor and each live calibration initializer."""

    state = model.state_dict()
    floating_tensor_count = 0
    floating_element_count = 0
    for name, value in state.items():
        if value.is_floating_point() or value.is_complex():
            floating_tensor_count += 1
            floating_element_count += value.numel()
            diagnostics = _tensor_diagnostics(value)
            if diagnostics["all_finite"] is not True:
                raise RuntimeError(
                    "Post-EMA parameter/buffer tensor is non-finite: "
                    f"name={name!r} diagnostics={diagnostics}"
                )
    sites = active_rational_norms(model)
    initializer_values: dict[str, float] = {}
    for name, norm in sites:
        diagnostics = _tensor_diagnostics(norm.running_ms)
        initialized = bool(norm.initialized.item())
        value = (
            float(norm.running_ms.item())
            if diagnostics["all_finite"] and norm.running_ms.numel() == 1
            else None
        )
        if (
            norm.running_ms.numel() != 1
            or diagnostics["all_finite"] is not True
            or value is None
            or value < PRODUCT_RUNNING_MS_IDENTITY_FLOOR
            or not initialized
        ):
            raise RuntimeError(
                "Post-EMA calibration initializer is invalid: "
                f"site={name!r} initialized={initialized} diagnostics={diagnostics} "
                f"value={value!r} required_min={PRODUCT_RUNNING_MS_IDENTITY_FLOOR}"
            )
        if norm.frozen:
            raise RuntimeError(
                f"Post-EMA calibration initializer site={name!r} was frozen during training"
            )
        initializer_values[name] = value
    guard_report = install_product_running_ms_fail_closed_guards(model)
    return {
        "phase": "after_ema_parameter_swap_before_calibration",
        "state_tensor_count": len(state),
        "floating_tensor_count": floating_tensor_count,
        "floating_element_count": floating_element_count,
        "all_parameter_and_buffer_tensors_finite": True,
        "live_site_count": len(sites),
        "all_active_initializers_finite_positive_initialized": True,
        "all_active_sites_unfrozen_during_training": True,
        "initializer": CALIBRATION_INITIALIZER_LABEL,
        "initializer_running_ms_sha256": canonical_sha256(initializer_values),
        "initializer_running_ms": initializer_values,
        "initializer_min": min(initializer_values.values()),
        "initializer_max": max(initializer_values.values()),
        "running_ms_required_min": PRODUCT_RUNNING_MS_IDENTITY_FLOOR,
        "historical_floor_is_identity_for_every_accepted_forward": True,
        "running_ms_fail_closed_guard": guard_report,
    }


def _calibration_error(
    reason: str,
    *,
    pass_index: int,
    batch_index: int,
    batch_start: int,
    batch_stop: int,
    site: str,
    diagnostics: Mapping[str, Any],
) -> RuntimeError:
    return RuntimeError(
        "Simultaneous rational calibration failed: "
        f"reason={reason} pass_index={pass_index} batch_index={batch_index} "
        f"batch_start={batch_start} batch_stop={batch_stop} site={site!r} "
        f"diagnostics={dict(diagnostics)}"
    )


def active_product_component_modules(model: ChiVLA) -> tuple[tuple[str, Any], ...]:
    """Return the nine raw Product component producers in fixed decode order."""

    head = model.product_head
    modules = (
        ("product_center", head.center),
        *((f"product_factor_{index}", module) for index, module in enumerate(head.factors)),
        *((f"product_gate_{index}", module) for index, module in enumerate(head.gates)),
    )
    if tuple(name for name, _ in modules) != PRODUCT_COMPONENT_SITE_NAMES:
        raise RuntimeError("Raw Product component module inventory differs")
    return modules


@torch.inference_mode()
def calibrate_live_rational_norms_simultaneously(
    model: ChiVLA,
    images: torch.Tensor,
    instructions: torch.Tensor,
    states: torch.Tensor,
) -> dict[str, Any]:
    """Calibrate from trained anchors using snapshot/commit whole-network passes."""

    if (
        images.shape[0] <= 0
        or instructions.shape[0] != images.shape[0]
        or states.shape[0] != images.shape[0]
    ):
        raise RuntimeError("Calibration inputs have inconsistent or empty batch axes")
    sites = active_rational_norms(model)
    product_components = active_product_component_modules(model)
    initializer_values: dict[str, float] = {}
    for name, norm in sites:
        diagnostics = _tensor_diagnostics(norm.running_ms)
        initialized = bool(norm.initialized.item())
        value = (
            float(norm.running_ms.item())
            if diagnostics["all_finite"] and norm.running_ms.numel() == 1
            else None
        )
        if (
            norm.running_ms.numel() != 1
            or diagnostics["all_finite"] is not True
            or value is None
            or value < PRODUCT_RUNNING_MS_IDENTITY_FLOOR
            or not initialized
        ):
            raise _calibration_error(
                "invalid_post_ema_initializer",
                pass_index=-1,
                batch_index=-1,
                batch_start=-1,
                batch_stop=-1,
                site=name,
                diagnostics={
                    "initialized": initialized,
                    "running_ms": diagnostics,
                    "value": value,
                    "required_min": PRODUCT_RUNNING_MS_IDENTITY_FLOOR,
                },
            )
        initializer_values[name] = value
    for _, norm in sites:
        norm.frozen = True
    model.eval()
    pass_records: list[dict[str, Any]] = []
    hook_calls = {name: 0 for name, _ in sites}
    product_component_calls = {name: 0 for name, _ in product_components}
    batch_count = (
        int(images.shape[0]) + FINAL_CALIBRATION_BATCH_SIZE - 1
    ) // FINAL_CALIBRATION_BATCH_SIZE
    for pass_index in range(FINAL_CALIBRATION_PASSES + 1):
        audit_only = pass_index == FINAL_CALIBRATION_PASSES
        sums: dict[str, torch.Tensor | None] = {name: None for name, _ in sites}
        counts = {name: 0 for name, _ in sites}
        pass_hook_calls = {name: 0 for name, _ in sites}
        batch_site_calls = {name: 0 for name, _ in sites}
        pass_component_calls = {name: 0 for name, _ in product_components}
        batch_component_calls = {name: 0 for name, _ in product_components}
        batch_context = {"index": -1, "start": -1, "stop": -1}
        handles = []
        for name, norm in sites:
            def capture(_module, inputs, *, site_name=name):
                if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
                    raise _calibration_error(
                        "invalid_hook_input",
                        pass_index=pass_index,
                        batch_index=batch_context["index"],
                        batch_start=batch_context["start"],
                        batch_stop=batch_context["stop"],
                        site=site_name,
                        diagnostics={"input_count": len(inputs)},
                    )
                value = inputs[0].detach().to(dtype=torch.float64)
                value_diagnostics = _tensor_diagnostics(value)
                if value_diagnostics["all_finite"] is not True:
                    raise _calibration_error(
                        "nonfinite_observation",
                        pass_index=pass_index,
                        batch_index=batch_context["index"],
                        batch_start=batch_context["start"],
                        batch_stop=batch_context["stop"],
                        site=site_name,
                        diagnostics={"observation": value_diagnostics},
                    )
                contribution = value.square().sum(dtype=torch.float64)
                contribution_diagnostics = _tensor_diagnostics(contribution)
                if contribution_diagnostics["all_finite"] is not True:
                    raise _calibration_error(
                        "nonfinite_float64_square_sum",
                        pass_index=pass_index,
                        batch_index=batch_context["index"],
                        batch_start=batch_context["start"],
                        batch_stop=batch_context["stop"],
                        site=site_name,
                        diagnostics={
                            "observation": value_diagnostics,
                            "contribution": contribution_diagnostics,
                        },
                    )
                previous = sums[site_name]
                updated = contribution if previous is None else previous + contribution
                updated_diagnostics = _tensor_diagnostics(updated)
                if updated_diagnostics["all_finite"] is not True:
                    raise _calibration_error(
                        "nonfinite_float64_accumulator",
                        pass_index=pass_index,
                        batch_index=batch_context["index"],
                        batch_start=batch_context["start"],
                        batch_stop=batch_context["stop"],
                        site=site_name,
                        diagnostics={
                            "observation": value_diagnostics,
                            "contribution": contribution_diagnostics,
                            "previous": None
                            if previous is None
                            else _tensor_diagnostics(previous),
                            "updated": updated_diagnostics,
                        },
                    )
                sums[site_name] = updated
                counts[site_name] += value.numel()
                hook_calls[site_name] += 1
                pass_hook_calls[site_name] += 1
                batch_site_calls[site_name] += 1

            handles.append(norm.register_forward_pre_hook(capture))
        for component_name, component in product_components:
            def capture_component(_module, _inputs, output, *, site_name=component_name):
                if not isinstance(output, torch.Tensor):
                    raise _calibration_error(
                        "invalid_raw_product_component_output",
                        pass_index=pass_index,
                        batch_index=batch_context["index"],
                        batch_start=batch_context["start"],
                        batch_stop=batch_context["stop"],
                        site=site_name,
                        diagnostics={"output_type": type(output).__name__},
                    )
                diagnostics = _tensor_diagnostics(output)
                if diagnostics["all_finite"] is not True:
                    raise _calibration_error(
                        "nonfinite_raw_product_component",
                        pass_index=pass_index,
                        batch_index=batch_context["index"],
                        batch_start=batch_context["start"],
                        batch_stop=batch_context["stop"],
                        site=site_name,
                        diagnostics={"output": diagnostics},
                    )
                product_component_calls[site_name] += 1
                pass_component_calls[site_name] += 1
                batch_component_calls[site_name] += 1

            handles.append(component.register_forward_hook(capture_component))
        try:
            for batch_index, start in enumerate(
                range(0, images.shape[0], FINAL_CALIBRATION_BATCH_SIZE)
            ):
                stop = min(start + FINAL_CALIBRATION_BATCH_SIZE, images.shape[0])
                batch_context.update(index=batch_index, start=start, stop=stop)
                for name in batch_site_calls:
                    batch_site_calls[name] = 0
                for name in batch_component_calls:
                    batch_component_calls[name] = 0
                embodiment = torch.zeros(
                    stop - start, dtype=torch.long, device=images.device
                )
                output, _ = model(
                    images[start:stop].float().div(255),
                    instructions[start:stop],
                    states[start:stop],
                    embodiment,
                )
                output_diagnostics = _tensor_diagnostics(output)
                if output_diagnostics["all_finite"] is not True:
                    raise _calibration_error(
                        "nonfinite_model_output",
                        pass_index=pass_index,
                        batch_index=batch_index,
                        batch_start=start,
                        batch_stop=stop,
                        site="model_output",
                        diagnostics={"output": output_diagnostics},
                    )
                wrong_batch_calls = {
                    name: count
                    for name, count in batch_site_calls.items()
                    if count != 1
                }
                if wrong_batch_calls:
                    first = next(iter(wrong_batch_calls))
                    raise _calibration_error(
                        "site_call_count_per_batch_differs",
                        pass_index=pass_index,
                        batch_index=batch_index,
                        batch_start=start,
                        batch_stop=stop,
                        site=first,
                        diagnostics={"site_call_count": wrong_batch_calls},
                    )
                wrong_component_calls = {
                    name: count
                    for name, count in batch_component_calls.items()
                    if count != 1
                }
                if wrong_component_calls:
                    first = next(iter(wrong_component_calls))
                    raise _calibration_error(
                        "raw_product_component_call_count_per_batch_differs",
                        pass_index=pass_index,
                        batch_index=batch_index,
                        batch_start=start,
                        batch_stop=stop,
                        site=first,
                        diagnostics={"component_call_count": wrong_component_calls},
                    )
        finally:
            for handle in handles:
                handle.remove()
        wrong_pass_calls = {
            name: count
            for name, count in pass_hook_calls.items()
            if count != batch_count
        }
        if wrong_pass_calls:
            first = next(iter(wrong_pass_calls))
            raise _calibration_error(
                "site_call_count_per_pass_differs",
                pass_index=pass_index,
                batch_index=batch_count - 1,
                batch_start=max(0, int(images.shape[0]) - FINAL_CALIBRATION_BATCH_SIZE),
                batch_stop=int(images.shape[0]),
                site=first,
                diagnostics={
                    "expected_calls": batch_count,
                    "site_call_count": wrong_pass_calls,
                },
            )
        wrong_pass_component_calls = {
            name: count
            for name, count in pass_component_calls.items()
            if count != batch_count
        }
        if wrong_pass_component_calls:
            first = next(iter(wrong_pass_component_calls))
            raise _calibration_error(
                "raw_product_component_call_count_per_pass_differs",
                pass_index=pass_index,
                batch_index=batch_count - 1,
                batch_start=max(0, int(images.shape[0]) - FINAL_CALIBRATION_BATCH_SIZE),
                batch_stop=int(images.shape[0]),
                site=first,
                diagnostics={
                    "expected_calls": batch_count,
                    "component_call_count": wrong_pass_component_calls,
                },
            )
        candidates: dict[str, torch.Tensor] = {}
        candidate_values: dict[str, float] = {}
        relative_changes: dict[str, float] = {}
        for name, norm in sites:
            total = sums[name]
            count = counts[name]
            if total is None or count <= 0:
                raise _calibration_error(
                    "site_not_exercised",
                    pass_index=pass_index,
                    batch_index=batch_count - 1,
                    batch_start=max(0, int(images.shape[0]) - FINAL_CALIBRATION_BATCH_SIZE),
                    batch_stop=int(images.shape[0]),
                    site=name,
                    diagnostics={"count": count, "total_present": total is not None},
                )
            candidate64 = total / count + torch.as_tensor(
                norm.eps, dtype=torch.float64, device=total.device
            )
            candidate_storage = candidate64.to(dtype=norm.running_ms.dtype)
            candidate_diagnostics = _tensor_diagnostics(candidate64)
            storage_diagnostics = _tensor_diagnostics(candidate_storage)
            candidate_value = (
                float(candidate64.item())
                if candidate_diagnostics["all_finite"]
                else None
            )
            if (
                candidate_diagnostics["all_finite"] is not True
                or storage_diagnostics["all_finite"] is not True
                or candidate_value is None
                or candidate_value < PRODUCT_RUNNING_MS_IDENTITY_FLOOR
                or float(candidate_storage.item()) < PRODUCT_RUNNING_MS_IDENTITY_FLOOR
            ):
                raise _calibration_error(
                    "invalid_candidate_scale",
                    pass_index=pass_index,
                    batch_index=batch_count - 1,
                    batch_start=max(0, int(images.shape[0]) - FINAL_CALIBRATION_BATCH_SIZE),
                    batch_stop=int(images.shape[0]),
                    site=name,
                    diagnostics={
                        "count": count,
                        "total": _tensor_diagnostics(total),
                        "candidate_float64": candidate_diagnostics,
                        "candidate_storage": storage_diagnostics,
                        "candidate_value": candidate_value,
                    },
                )
            previous64 = norm.running_ms.detach().to(dtype=torch.float64)
            committed_candidate64 = candidate_storage.detach().to(dtype=torch.float64)
            previous_value = float(previous64.item())
            committed_candidate_value = float(committed_candidate64.item())
            relative_value = abs(committed_candidate_value - previous_value) / abs(
                previous_value
            )
            if not math.isfinite(relative_value):
                raise _calibration_error(
                    "nonfinite_relative_change",
                    pass_index=pass_index,
                    batch_index=batch_count - 1,
                    batch_start=max(0, int(images.shape[0]) - FINAL_CALIBRATION_BATCH_SIZE),
                    batch_stop=int(images.shape[0]),
                    site=name,
                    diagnostics={
                        "previous": _tensor_diagnostics(previous64),
                        "candidate_storage_as_float64": _tensor_diagnostics(
                            committed_candidate64
                        ),
                        "relative_value": relative_value,
                    },
                )
            candidates[name] = candidate_storage
            candidate_values[name] = committed_candidate_value
            relative_changes[name] = relative_value
        if not audit_only:
            precommit = {
                name: (
                    norm.running_ms.detach().clone(),
                    norm.initialized.detach().clone(),
                    norm.frozen,
                )
                for name, norm in sites
            }
            try:
                for name, norm in sites:
                    norm.running_ms.copy_(candidates[name])
                    norm.initialized.fill_(True)
                    norm.frozen = True
                install_product_running_ms_fail_closed_guards(model)
            except Exception as error:
                rollback_failures = []
                for name, norm in sites:
                    try:
                        previous_running, previous_initialized, previous_frozen = precommit[
                            name
                        ]
                        norm.running_ms.copy_(previous_running)
                        norm.initialized.copy_(previous_initialized)
                        norm.frozen = previous_frozen
                    except Exception as rollback_error:
                        rollback_failures.append(
                            f"{name}: {type(rollback_error).__name__}: {rollback_error}"
                        )
                if rollback_failures:
                    raise RuntimeError(
                        "Calibration commit failed and rollback was incomplete: "
                        f"{rollback_failures}"
                    ) from error
                raise RuntimeError(
                    "Calibration commit failed after full-vector validation; all live "
                    "running_ms buffers were restored from the pre-commit snapshot"
                ) from error
        committed_values = {
            name: float(norm.running_ms.item()) for name, norm in sites
        }
        worst_site = max(relative_changes, key=relative_changes.__getitem__)
        pass_records.append(
            {
                "pass_index": pass_index,
                "audit_only_no_commit": audit_only,
                "sample_count": int(images.shape[0]),
                "batch_count": batch_count,
                "max_relative_scale_change": relative_changes[worst_site],
                "worst_relative_change_site": worst_site,
                "observation_copy_dtype": "float64",
                "observation_copy_before_square": True,
                "accumulator_dtype": "float64",
                "candidate_running_ms": candidate_values,
                "candidate_running_ms_sha256": canonical_sha256(candidate_values),
                "relative_change_by_site": relative_changes,
                "relative_change_by_site_sha256": canonical_sha256(relative_changes),
                "site_call_count": pass_hook_calls,
                "product_component_call_count": pass_component_calls,
                "committed_running_ms_sha256": canonical_sha256(committed_values),
            }
        )
    audit_residual = pass_records[-1]["max_relative_scale_change"]
    if audit_residual > FINAL_CALIBRATION_AUDIT_MAX_RELATIVE_CHANGE:
        worst_site = pass_records[-1]["worst_relative_change_site"]
        raise _calibration_error(
            "fixed_pass_audit_residual_exceeds_threshold",
            pass_index=FINAL_CALIBRATION_PASSES,
            batch_index=batch_count - 1,
            batch_start=max(0, int(images.shape[0]) - FINAL_CALIBRATION_BATCH_SIZE),
            batch_stop=int(images.shape[0]),
            site=worst_site,
            diagnostics={
                "observed_relative_change": audit_residual,
                "required_max_relative_change": (
                    FINAL_CALIBRATION_AUDIT_MAX_RELATIVE_CHANGE
                ),
                "candidate_running_ms": pass_records[-1][
                    "candidate_running_ms"
                ][worst_site],
                "committed_running_ms": float(
                    dict(sites)[worst_site].running_ms.item()
                ),
            },
        )
    expected_calls = (FINAL_CALIBRATION_PASSES + 1) * batch_count
    if set(hook_calls.values()) != {expected_calls}:
        raise RuntimeError(f"Fresh calibration site coverage differs: {hook_calls}")
    if set(product_component_calls.values()) != {expected_calls}:
        raise RuntimeError(
            "Fresh calibration raw Product component coverage differs: "
            f"{product_component_calls}"
        )
    final_values = {name: float(norm.running_ms.item()) for name, norm in sites}
    model.eval()
    return {
        "method": "trained_anchor_fixed_pass_simultaneous_end_to_end",
        "under_final_ema_parameters": True,
        "initializer": CALIBRATION_INITIALIZER_LABEL,
        "initializer_running_ms_sha256": canonical_sha256(initializer_values),
        "initializer_running_ms": initializer_values,
        "initializer_min": min(initializer_values.values()),
        "initializer_max": max(initializer_values.values()),
        "running_ms_required_min": PRODUCT_RUNNING_MS_IDENTITY_FLOOR,
        "historical_floor_is_identity_for_every_accepted_forward": True,
        "old_training_buffers_discarded": False,
        "all_sites_frozen_before_first_pass": True,
        "simultaneous_snapshot_commit": True,
        "commit_protocol": (
            "all_candidates_validated_before_sequential_commit_with_exception_rollback"
        ),
        "commit_exception_rollback": True,
        "intra_pass_exception_rollback_source": (
            "in_memory_precommit_running_ms_snapshot"
        ),
        "process_level_retry_source": (
            "immutable_training_only_precalibration_ema_state"
        ),
        "recovery_scope": CALIBRATION_RECOVERY_SCOPE,
        "cross_version_recovery_supported": False,
        "no_value_changing_clamp_or_fallback_on_accepted_path": True,
        "invalid_anchor_rejected_before_frozen_forward": True,
        "initializer_and_every_candidate_floor_verified_before_use_or_commit": True,
        "observation_copy_dtype": "float64",
        "observation_copy_before_square": True,
        "accumulator_dtype": "float64",
        "update_passes": FINAL_CALIBRATION_PASSES,
        "final_no_commit_audit_pass": True,
        "audit_max_relative_change": FINAL_CALIBRATION_AUDIT_MAX_RELATIVE_CHANGE,
        "audit_observed_max_relative_change": audit_residual,
        "batch_size": FINAL_CALIBRATION_BATCH_SIZE,
        "sample_count_per_pass": int(images.shape[0]),
        "live_site_count": len(sites),
        "site_names": [name for name, _ in sites],
        "site_names_sha256": canonical_sha256([name for name, _ in sites]),
        "site_call_count": hook_calls,
        "product_component_site_names": [name for name, _ in product_components],
        "product_component_site_names_sha256": canonical_sha256(
            [name for name, _ in product_components]
        ),
        "product_component_call_count": product_component_calls,
        "raw_product_components_finite_every_batch": True,
        "expected_calls_per_site": expected_calls,
        "pass_records": pass_records,
        "final_running_ms_sha256": canonical_sha256(final_values),
        "final_running_ms": final_values,
        "exact_fixed_point_convergence_claimed": False,
        "all_initialized_and_frozen": all(
            bool(norm.initialized) and norm.frozen for _, norm in sites
        ),
    }


def assert_own_outputs_absent(
    run_root: Path, seed: int, smoke: bool, *, resume_precalibration: bool
) -> None:
    deployment_paths = (
        checkpoint_path(run_root, seed, smoke),
        metadata_path(run_root, seed, smoke),
        training_result_path(run_root, seed, smoke),
        deployment_completion_path(run_root, seed, smoke),
    ) + ((calibration_resume_proof_path(run_root),) if smoke else ())
    collisions = [
        str(path)
        for path in deployment_paths
        if path.exists() or path.is_symlink()
    ]
    if collisions:
        raise FileExistsError(f"Refusing existing seed outputs: {collisions}")
    precalibration = precalibration_state_path(run_root, seed, smoke)
    if resume_precalibration:
        if not precalibration.is_file() or precalibration.is_symlink():
            raise RuntimeError(
                "Calibration-only resume requires the fixed physical pre-calibration artifact"
            )
    elif precalibration.exists() or precalibration.is_symlink():
        raise FileExistsError(
            f"Refusing existing training-only pre-calibration output: {precalibration}"
        )


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed is not None:
        raise ValueError("--seed is not accepted with --preflight-only")
    smoke = args.mode == "smoke"
    manifest = load_and_verify_source_manifest(args.source_manifest)
    if _PREIMPORT_SOURCE_VERIFICATION["source_sha256"] != manifest["source_closure"]:
        raise RuntimeError("Pre-import and post-import source verification differ")
    cache_identity = assert_cache_identity()
    suite = load_suite()
    official_tasks = task_languages(suite)
    dataset_tasks, task_metadata = validate_task_mapping(official_tasks)
    vocab, encode = build_vocab(dataset_tasks)
    cache_structure = load_and_validate_cache(dataset_tasks, encode, materialize=False)
    training_model = ChiVLA(make_product_config(len(vocab), deployment=False))
    deployment_model = ChiVLA(make_product_config(len(vocab), deployment=True))
    training_topology = validate_product_model(
        training_model, deployment=False, require_initialized_norms=False
    )
    deployment_state, stripped = deployment_state_from_training(training_model)
    strict_load_deployment_state(deployment_model, deployment_state)
    deployment_topology = validate_product_model(
        deployment_model, deployment=True, require_initialized_norms=False
    )
    state_equal = all(
        torch.equal(deployment_model.state_dict()[key], value)
        for key, value in deployment_state.items()
    )
    if not state_equal:
        raise RuntimeError("Static deployment export changed a retained state tensor")
    assert_outputs_absent(args.run_root, include_smoke=smoke)
    postimport_static_audit = verify_transitive_static_audit_now()
    final_runtime_guard = assert_direct_only_runtime_guard(
        direct_only_runtime_report(), exact_allowed_calls=()
    )
    if final_runtime_guard["prohibited_attempt_count"] != 0:
        raise RuntimeError("A prohibited numerical route was attempted during preflight")
    payload = {
        "schema": f"{SCHEMA}_preflight",
        "ready": True,
        "mode": args.mode,
        "seeds": list(SEEDS),
        "cache": cache_identity,
        "cache_structure": cache_structure,
        "task_metadata": task_metadata,
        "dataset_tasks": {str(key): value for key, value in dataset_tasks.items()},
        "official_tasks": {str(key): value for key, value in official_tasks.items()},
        "vocab_size": len(vocab),
        "vocab_sha256": canonical_sha256(vocab),
        "training_config": config_record(training_model.cfg),
        "deployment_config": config_record(deployment_model.cfg),
        "training_topology": training_topology,
        "deployment_topology": deployment_topology,
        "deployment_export": {
            "strict_load": True,
            "retained_state_bitwise_equal": True,
            "removed_training_only_keys": list(stripped),
            "removed_non_training_keys": [],
        },
        "training_recipe": training_recipe(smoke),
        "source_manifest": {
            "path": args.source_manifest.as_posix(),
            "sha256": file_sha256(args.source_manifest),
            "bundle_sha256": manifest["source_bundle_sha256"],
            "authenticated_inputs_bundle_sha256": manifest[
                "authenticated_inputs_bundle_sha256"
            ],
            "launch_bundle_sha256": manifest["launch_bundle_sha256"],
            "manifest_bundle_sha256": manifest["manifest_bundle_sha256"],
        },
        "source_snapshot": source_snapshot(),
        "preimport_source_verification": _PREIMPORT_SOURCE_VERIFICATION,
        "transitive_direct_only_static_audit": _TRANSITIVE_STATIC_AUDIT,
        "postimport_transitive_direct_only_static_audit": postimport_static_audit,
        "runtime_direct_only_guard_at_import": _RUNTIME_GUARD_AT_IMPORT,
        "runtime_direct_only_guard_final": final_runtime_guard,
        "claim_boundary": DEPLOYMENT_CLAIM_BOUNDARY,
    }
    if args.preflight_output is not None:
        write_json_exclusive(args.preflight_output, payload)
    return payload


@torch.inference_mode()
def prove_deployment_equivalence(
    training_model: ChiVLA,
    deployment_model: ChiVLA,
    images: torch.Tensor,
    instructions: torch.Tensor,
    states: torch.Tensor,
) -> dict[str, Any]:
    training_model.eval()
    deployment_model.eval()
    embodiments = torch.zeros(images.shape[0], dtype=torch.long, device=images.device)
    training_actions, _ = training_model(images, instructions, states, embodiments)
    deployment_actions, _ = deployment_model(images, instructions, states, embodiments)
    training_h = training_model.pooled_features(images, instructions, states, embodiments)
    center, factors, gates = training_model.product_head(training_h)
    signs = torch.where(gates > 0, 1.0, -1.0).to(training_h.dtype)
    explicit_actions = training_model.product_head.action_for_signs(
        center, factors, signs
    ).reshape(-1, ACTION_HORIZON, ACTION_DIM)
    deployment_h = deployment_model.pooled_features(images, instructions, states, embodiments)
    deploy_center, deploy_factors, deploy_gates = deployment_model.product_head(deployment_h)
    deploy_signs = torch.where(deploy_gates > 0, 1.0, -1.0).to(deployment_h.dtype)
    checks = {
        "training_public_equals_explicit_decode": torch.equal(training_actions, explicit_actions),
        "training_equals_deployment_actions": torch.equal(training_actions, deployment_actions),
        "training_equals_deployment_pooled_state": torch.equal(training_h, deployment_h),
        "training_equals_deployment_center": torch.equal(center, deploy_center),
        "training_equals_deployment_factors": torch.equal(factors, deploy_factors),
        "training_equals_deployment_gates": torch.equal(gates, deploy_gates),
        "training_equals_deployment_signs": torch.equal(signs, deploy_signs),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Deployment export inference proof failed: {checks}")
    finite_outputs = {
        "public_actions": training_actions,
        "pooled_state": training_h,
        "center": center,
        "factors": factors,
        "gates": gates,
    }
    nonfinite = [
        name
        for name, value in finite_outputs.items()
        if not bool(torch.isfinite(value).all())
    ]
    if nonfinite:
        raise RuntimeError(
            "Deployment equivalence panel produced non-finite tensors: "
            f"{nonfinite}"
        )
    output_tensor_sha256 = tensor_state_sha256(
        {
            "public_actions": training_actions,
            "pooled_state": training_h,
            "center": center,
            "factors": factors,
            "gates": gates,
            "signs": signs,
        }
    )
    return {
        "panel_size": images.shape[0],
        "bitwise_checks": checks,
        "output_tensor_sha256": output_tensor_sha256,
        "max_abs_action_error": float((training_actions - deployment_actions).abs().max()),
        "max_abs_gate_error": float((gates - deploy_gates).abs().max()),
        "positive_sign_fraction": float((signs > 0).float().mean()),
        "zero_gate_logit_count": int((gates == 0).sum()),
        "sign_rule": "+1 iff gate_logit > 0, else -1",
    }


def require_version_tracked_resume_context() -> None:
    """Fail before recovery if tensor version counters were globally disabled."""

    if torch.is_inference_mode_enabled():
        raise RuntimeError(
            "Calibration resume proof requires version-tracked tensors; "
            "global inference mode is forbidden for recovery orchestration"
        )


def canonical_json_runtime_guard_record(report: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize validated runtime-ledger sequences to their published JSON types."""

    normalized = dict(report)
    for field in ("patched_entrypoints", "allowed_calls", "prohibited_attempts"):
        value = normalized.get(field)
        if not isinstance(value, (list, tuple)):
            raise RuntimeError(f"Runtime guard {field} is not a sequence")
        normalized[field] = list(value)
    return normalized


@torch.no_grad()
def run_same_allocation_calibration_resume_proof(
    args: argparse.Namespace,
    *,
    model: ChiVLA,
    images: torch.Tensor,
    instructions: torch.Tensor,
    states: torch.Tensor,
    data: Mapping[str, Any],
    vocab: Mapping[str, int],
    manifest_start: Mapping[str, Any],
    manifest_sha256: str,
    sources_start: Mapping[str, str],
    training_config_record: Mapping[str, Any],
    preflight_certificate: Mapping[str, Any],
    compute_capability: tuple[int, int],
) -> dict[str, Any]:
    """Replay the real smoke recovery path in the same scheduled GPU allocation."""

    require_version_tracked_resume_context()
    output = calibration_resume_proof_path(args.run_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing calibration resume proof {output}")
    checkpoint = checkpoint_path(args.run_root, 0, True)
    metadata_file = metadata_path(args.run_root, 0, True)
    training_file = training_result_path(args.run_root, 0, True)
    precalibration_file = precalibration_state_path(args.run_root, 0, True)
    completion_file = deployment_completion_path(args.run_root, 0, True)
    completion = load_and_validate_deployment_completion(
        completion_file,
        expected_seed=0,
        expected_mode="smoke",
        expected_checkpoint=checkpoint,
        expected_metadata=metadata_file,
        expected_training_result=training_file,
        expected_precalibration_ema_state=precalibration_file,
        expected_source_manifest_sha256=manifest_sha256,
        expected_manifest_bundle_sha256=manifest_start["manifest_bundle_sha256"],
    )
    reference_metadata = read_json_object_physical(
        metadata_file, label="Fresh smoke metadata"
    )
    reference_training = read_json_object_physical(
        training_file, label="Fresh smoke training result"
    )
    loaded = load_training_only_precalibration_state(
        precalibration_file,
        expected_file_sha256=args.precalibration_sha256,
        expected_seed=0,
        expected_mode="smoke",
        expected_step_count=SMOKE_STEPS,
        expected_source_manifest_sha256=manifest_sha256,
        expected_manifest_bundle_sha256=manifest_start["manifest_bundle_sha256"],
        expected_source_bundle_sha256=source_bundle_sha256(sources_start),
        expected_training_config_sha256=training_config_record["sha256"],
    )
    incompatibility = model.load_state_dict(loaded["training_state"], strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"Resume-proof topology differs: {incompatibility}")
    restored_state_sha256 = tensor_state_sha256(model.state_dict())
    if restored_state_sha256 != loaded["training_state_sha256"]:
        raise RuntimeError("Resume proof changed the restored pre-calibration state")
    post_ema = validate_post_ema_training_state(model)
    if not json_type_exact_equal(
        post_ema, loaded["post_ema_state_validation"]
    ):
        raise RuntimeError("Resume-proof post-EMA validation differs")
    initial_values = active_running_ms_values(model)
    started = time.perf_counter()
    calibration = calibrate_live_rational_norms_simultaneously(
        model, images, instructions, states
    )
    final_values = active_running_ms_values(model)
    attestation = validate_simultaneous_calibration_attestation(
        calibration,
        initial_values=initial_values,
        final_values=final_values,
        sample_count=int(data["sample_count"]),
    )
    deployment_state, stripped = deployment_state_from_training(model)
    cpu_training_state = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    cpu_deployment_state = {
        key: value.detach().cpu() for key, value in deployment_state.items()
    }
    transition = calibration_state_transition_proof(
        loaded["training_state"],
        cpu_training_state,
        cpu_deployment_state,
        active_site_names=tuple(initial_values),
    )
    reference_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(reference_state, dict)
        or tuple(sorted(reference_state)) != tuple(sorted(cpu_deployment_state))
        or any(
            not torch.equal(reference_state[key], cpu_deployment_state[key])
            for key in reference_state
        )
    ):
        raise RuntimeError("Resume proof did not reproduce the fresh deployment state")
    deployment_model = ChiVLA(make_product_config(len(vocab), deployment=True)).cuda()
    strict_load_deployment_state(deployment_model, reference_state)
    panel_size = min(32, int(data["sample_count"]))
    panel_indices = torch.linspace(
        0, int(data["sample_count"]) - 1, panel_size, device="cuda"
    ).round().long()
    replay = prove_deployment_equivalence(
        model,
        deployment_model,
        images[panel_indices].float().div(255),
        instructions[panel_indices],
        states[panel_indices],
    )
    reference_calibration = reference_metadata.get(
        "post_ema_simultaneous_rational_calibration"
    )
    reference_attestation = reference_metadata.get(
        "post_ema_simultaneous_rational_calibration_attestation"
    )
    reference_transition = reference_metadata.get("calibration_state_transition")
    reference_replay = reference_metadata.get("deployment_export", {}).get(
        "inference_equivalence"
    )
    if (
        not json_type_exact_equal(reference_calibration, calibration)
        or not json_type_exact_equal(
            reference_training.get("post_ema_simultaneous_rational_calibration"),
            calibration,
        )
        or not json_type_exact_equal(reference_attestation, attestation)
        or not json_type_exact_equal(
            reference_training.get(
                "post_ema_simultaneous_rational_calibration_attestation"
            ),
            attestation,
        )
        or not json_type_exact_equal(reference_transition, transition)
        or not json_type_exact_equal(
            reference_training.get("calibration_state_transition"), transition
        )
        or not json_type_exact_equal(reference_replay, replay)
    ):
        raise RuntimeError(
            "Fresh and resumed calibration, state, or output attestations differ"
        )
    allocation = slurm_gpu_allocation_record(
        gpu=torch.cuda.get_device_name(0), compute_capability=compute_capability
    )
    reference_allocation = reference_metadata.get("training_environment", {})
    validate_same_slurm_gpu_allocation(allocation, reference_allocation)
    sources_end = source_snapshot()
    manifest_end = load_and_verify_source_manifest(args.source_manifest)
    static_end = verify_transitive_static_audit_now()
    if (
        sources_end != sources_start
        or manifest_end != manifest_start
        or file_sha256(args.source_manifest) != manifest_sha256
    ):
        raise RuntimeError("Authenticated source changed during resume proof")
    runtime = canonical_json_runtime_guard_record(
        assert_direct_only_runtime_guard(
            direct_only_runtime_report(), exact_allowed_calls=()
        )
    )
    payload = {
        "schema": CALIBRATION_RESUME_PROOF_SCHEMA,
        "seed": 0,
        "mode": "smoke",
        "fresh_deployment_completion_sha256": file_sha256(completion_file),
        "fresh_deployment_completion_bundle_sha256": completion[
            "artifact_bundle_sha256"
        ],
        "fresh_checkpoint_sha256": file_sha256(checkpoint),
        "fresh_metadata_sha256": file_sha256(metadata_file),
        "fresh_training_result_sha256": file_sha256(training_file),
        "precalibration_ema_state_sha256": args.precalibration_sha256,
        "precalibration_training_state_sha256": restored_state_sha256,
        "external_sha_authority_supplied": True,
        "optimizer_steps_executed": 0,
        "same_slurm_gpu_allocation_as_fresh_smoke": True,
        "allocation": allocation,
        "calibration_elapsed_s": time.perf_counter() - started,
        "fresh_and_resume_calibration_report_bitwise_equal": True,
        "fresh_and_resume_final_running_ms_bitwise_equal": True,
        "fresh_and_resume_deployment_state_bitwise_equal": True,
        "fresh_and_resume_raw_components_and_actions_bitwise_equal": True,
        "fresh_output_tensor_sha256": reference_replay["output_tensor_sha256"],
        "resume_output_tensor_sha256": replay["output_tensor_sha256"],
        "calibration_attestation": attestation,
        "calibration_state_transition": transition,
        "deployment_equivalence": replay,
        "removed_training_only_keys": list(stripped),
        "source_manifest_sha256": manifest_sha256,
        "manifest_bundle_sha256": manifest_start["manifest_bundle_sha256"],
        "preflight_certificate_sha256": preflight_certificate["sha256"],
        "source_snapshot_start": dict(sources_start),
        "source_snapshot_end": sources_end,
        "transitive_direct_only_static_audit": static_end,
        "runtime_direct_only_guard": runtime,
        "recovery_scope": CALIBRATION_RECOVERY_SCOPE,
        "cross_version_recovery_supported": False,
    }
    validate_same_allocation_calibration_resume_proof(
        payload,
        expected_source_manifest_sha256=manifest_sha256,
        expected_manifest_bundle_sha256=manifest_start["manifest_bundle_sha256"],
        expected_preflight_sha256=preflight_certificate["sha256"],
        expected_source_closure=sources_start,
        expected_static_audit=static_end,
        expected_checkpoint_sha256=file_sha256(checkpoint),
        expected_metadata_sha256=file_sha256(metadata_file),
        expected_training_result_sha256=file_sha256(training_file),
        expected_precalibration_sha256=args.precalibration_sha256,
        expected_precalibration_training_state_sha256=restored_state_sha256,
        expected_completion_sha256=file_sha256(completion_file),
        expected_completion_bundle_sha256=completion["artifact_bundle_sha256"],
        expected_training_environment=reference_allocation,
        expected_calibration_attestation=reference_attestation,
        expected_state_transition=reference_transition,
        expected_deployment_equivalence=reference_replay,
    )
    write_json_exclusive(output, payload)
    return payload


def main() -> None:
    args = parse_args()
    if not _RUNNING_AS_ENTRYPOINT or _PREIMPORT_SOURCE_VERIFICATION is None:
        raise RuntimeError("Product training must execute through its authenticated entrypoint")
    if args.preflight_only:
        if (
            args.preflight_result is not None
            or args.resume_precalibration
            or args.precalibration_sha256 is not None
            or args.resume_proof_only
        ):
            raise ValueError(
                "Preflight forbids training-result and calibration-only resume options"
            )
        payload = preflight(args)
        print("PREFLIGHT", json.dumps(payload, ensure_ascii=False), flush=True)
        return
    if args.preflight_output is not None:
        raise ValueError("--preflight-output requires --preflight-only")
    if args.seed is None:
        raise ValueError("Training requires --seed")
    validate_resume_request(
        resume_precalibration=args.resume_precalibration,
        precalibration_sha256=args.precalibration_sha256,
        resume_proof_only=args.resume_proof_only,
    )
    expected_preflight = preflight_result_path(args.run_root)
    if args.preflight_result != expected_preflight:
        raise ValueError(
            f"Training requires the frozen preflight result {expected_preflight}"
        )
    validate_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native CUDA bfloat16 support is required; no dtype fallback is allowed")
    compute_capability = torch.cuda.get_device_capability(0)
    if compute_capability[0] < 8:
        raise RuntimeError(f"Ampere-or-newer CUDA capability is required, got {compute_capability}")
    if torch.cuda.get_device_name(0) != "NVIDIA RTX A6000":
        raise RuntimeError(
            f"Frozen training device must be NVIDIA RTX A6000, got {torch.cuda.get_device_name(0)!r}"
        )
    smoke = args.mode == "smoke"
    if args.resume_proof_only and (not smoke or args.seed != 0):
        raise ValueError("Calibration resume proof is fixed to smoke seed0")
    steps = SMOKE_STEPS if smoke else FULL_STEPS
    batch_size = SMOKE_BATCH_SIZE if smoke else FULL_BATCH_SIZE
    if not args.resume_proof_only:
        assert_own_outputs_absent(
            args.run_root,
            args.seed,
            smoke,
            resume_precalibration=args.resume_precalibration,
        )

    manifest_start = load_and_verify_source_manifest(args.source_manifest)
    if _PREIMPORT_SOURCE_VERIFICATION["source_sha256"] != manifest_start["source_closure"]:
        raise RuntimeError("Pre-import and post-import source verification differ")
    manifest_sha256 = file_sha256(args.source_manifest)
    preflight_certificate = validate_preflight_certificate(
        args.preflight_result, args.source_manifest, manifest_start
    )
    sources_start = source_snapshot()
    cache_start = assert_cache_identity()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    suite = load_suite()
    official_tasks = task_languages(suite)
    dataset_tasks, task_metadata_start = validate_task_mapping(official_tasks)
    vocab, encode = build_vocab(dataset_tasks)
    data = load_and_validate_cache(dataset_tasks, encode, materialize=True)
    config = make_product_config(len(vocab), deployment=False)
    model = ChiVLA(config).cuda()
    training_topology = validate_product_model(
        model, deployment=False, require_initialized_norms=False
    )
    training_config_record = config_record(config)
    images = torch.from_numpy(data.pop("images")).permute(0, 3, 1, 2).cuda()
    instructions = torch.from_numpy(data.pop("instructions")).cuda()
    states = torch.from_numpy(data.pop("states")).cuda()
    action_array = data.pop("actions")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    execution_started = time.perf_counter()
    precalibration_output = precalibration_state_path(
        args.run_root, args.seed, smoke
    )
    if args.resume_proof_only:
        proof = run_same_allocation_calibration_resume_proof(
            args,
            model=model,
            images=images,
            instructions=instructions,
            states=states,
            data=data,
            vocab=vocab,
            manifest_start=manifest_start,
            manifest_sha256=manifest_sha256,
            sources_start=sources_start,
            training_config_record=training_config_record,
            preflight_certificate=preflight_certificate,
            compute_capability=compute_capability,
        )
        print("RESUME_PROOF", json.dumps(proof, ensure_ascii=False), flush=True)
        return
    if args.resume_precalibration:
        if action_array.shape[0] != data["sample_count"]:
            raise RuntimeError("Calibration-only resume observed malformed cached actions")
        loaded_precalibration = load_training_only_precalibration_state(
            precalibration_output,
            expected_file_sha256=args.precalibration_sha256,
            expected_seed=args.seed,
            expected_mode=args.mode,
            expected_step_count=steps,
            expected_source_manifest_sha256=manifest_sha256,
            expected_manifest_bundle_sha256=manifest_start["manifest_bundle_sha256"],
            expected_source_bundle_sha256=source_bundle_sha256(sources_start),
            expected_training_config_sha256=training_config_record["sha256"],
        )
        restored_state = loaded_precalibration["training_state"]
        incompatibility = model.load_state_dict(restored_state, strict=True)
        if incompatibility.missing_keys or incompatibility.unexpected_keys:
            raise RuntimeError(
                f"Calibration-only resume topology differs: {incompatibility}"
            )
        restored_state_sha256 = tensor_state_sha256(model.state_dict())
        if restored_state_sha256 != loaded_precalibration["training_state_sha256"]:
            raise RuntimeError("Calibration-only resume changed an exact state tensor")
        post_ema_state_validation = validate_post_ema_training_state(model)
        if (
            post_ema_state_validation
            != loaded_precalibration["post_ema_state_validation"]
        ):
            raise RuntimeError(
                "Calibration-only resume post-EMA validation differs from publication"
            )
        training_metrics = dict(loaded_precalibration["training_metrics"])
        precalibration_identity = {
            "path": precalibration_output.as_posix(),
            "sha256": args.precalibration_sha256,
            "schema": loaded_precalibration["schema"],
            "artifact_role": loaded_precalibration["artifact_role"],
            "deployment_eligible": loaded_precalibration["deployment_eligible"],
            "recovery_scope": loaded_precalibration["recovery_scope"],
            "cross_version_recovery_supported": loaded_precalibration[
                "cross_version_recovery_supported"
            ],
            "artifact_bundle_sha256": loaded_precalibration[
                "artifact_bundle_sha256"
            ],
            "training_state_sha256": loaded_precalibration[
                "training_state_sha256"
            ],
            "training_state_key_count": loaded_precalibration[
                "training_state_key_count"
            ],
            "training_state_keys_sha256": loaded_precalibration[
                "training_state_keys_sha256"
            ],
            "training_metrics": training_metrics,
        }
        del loaded_precalibration
        del restored_state
        calibration_resume_state_bitwise_exact = True
    else:
        actions = torch.from_numpy(action_array).cuda()
        embodiments = torch.zeros(batch_size, dtype=torch.long, device="cuda")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.95),
            weight_decay=WEIGHT_DECAY,
        )
        ema = {
            name: parameter.detach().clone().float()
            for name, parameter in model.named_parameters()
        }
        model.train()
        last_loss: float | None = None
        for step_index in range(steps):
            learning_rate = learning_rate_at(step_index, steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            indices = torch.randint(data["sample_count"], (batch_size,), device="cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(
                    images[indices].float().div(255),
                    instructions[indices],
                    states[indices],
                    embodiments,
                    target_actions=actions[indices],
                    progress=step_index / max(steps - 1, 1),
                )
            if loss is None or not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Training loss is invalid at step {step_index}: {loss}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP_NORM
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError(f"Gradient norm is non-finite at step {step_index}")
            optimizer.step()
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    ema[name].mul_(EMA_DECAY).add_(
                        parameter.detach().float(), alpha=1.0 - EMA_DECAY
                    )
            last_loss = float(loss.detach())
            if step_index % 1000 == 0 or step_index + 1 == steps:
                print(
                    f"STEP {step_index + 1}/{steps} loss={last_loss:.6f} "
                    f"lr={learning_rate:.8f}",
                    flush=True,
                )
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(ema[name].to(parameter.dtype))
        torch.cuda.synchronize()
        if last_loss is None or not np.isfinite(last_loss):
            raise RuntimeError(f"Final training loss is invalid: {last_loss}")
        training_metrics = {
            "completed_steps": steps,
            "last_loss": last_loss,
            "training_elapsed_s": time.perf_counter() - execution_started,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        }
        post_ema_state_validation = validate_post_ema_training_state(model)
        manifest_precalibration = load_and_verify_source_manifest(args.source_manifest)
        sources_precalibration = source_snapshot()
        if (
            manifest_precalibration != manifest_start
            or sources_precalibration != sources_start
            or file_sha256(args.source_manifest) != manifest_sha256
        ):
            raise RuntimeError(
                "Source identity changed before pre-calibration state publication"
            )
        precalibration_identity = publish_training_only_precalibration_state_exclusive(
            precalibration_output,
            model.state_dict(),
            seed=args.seed,
            mode=args.mode,
            step_count=steps,
            source_manifest_sha256=manifest_sha256,
            manifest_bundle_sha256=manifest_start["manifest_bundle_sha256"],
            source_bundle_sha256=source_bundle_sha256(sources_start),
            training_config_sha256=training_config_record["sha256"],
            post_ema_state_validation=post_ema_state_validation,
            training_metrics=training_metrics,
        )
        del optimizer
        del ema
        del actions
        calibration_resume_state_bitwise_exact = False
    precalibration_runtime_guard = assert_direct_only_runtime_guard(
        direct_only_runtime_report(), exact_allowed_calls=()
    )
    if precalibration_runtime_guard["prohibited_attempt_count"] != 0:
        raise RuntimeError(
            "A prohibited numerical route was attempted before calibration"
        )
    print(
        "PRECALIBRATION_RESUME" if args.resume_precalibration else "PRECALIBRATION_STATE",
        json.dumps(precalibration_identity, ensure_ascii=False),
        flush=True,
    )
    precalibration_running_ms_values = active_running_ms_values(model)
    calibration_report = calibrate_live_rational_norms_simultaneously(
        model, images, instructions, states
    )
    if calibration_report["all_initialized_and_frozen"] is not True:
        raise RuntimeError("Final simultaneous rational calibration did not close")
    final_running_ms_values = active_running_ms_values(model)
    calibration_attestation = validate_simultaneous_calibration_attestation(
        calibration_report,
        initial_values=precalibration_running_ms_values,
        final_values=final_running_ms_values,
        sample_count=data["sample_count"],
    )
    torch.cuda.synchronize()
    execution_elapsed_s = time.perf_counter() - execution_started
    elapsed_s = (
        float(training_metrics["training_elapsed_s"]) + execution_elapsed_s
        if args.resume_precalibration
        else execution_elapsed_s
    )
    last_loss = float(training_metrics["last_loss"])
    nonfinite_state = [
        name
        for name, value in model.state_dict().items()
        if value.is_floating_point() and not bool(torch.isfinite(value).all())
    ]
    if nonfinite_state:
        raise RuntimeError(f"Trained state contains non-finite tensors: {nonfinite_state}")
    training_topology_final = validate_product_model(
        model, deployment=False, require_initialized_norms=True
    )

    deployment_state, stripped_keys = deployment_state_from_training(model)
    deployment_config = make_product_config(len(vocab), deployment=True)
    deployment_model = ChiVLA(deployment_config).cuda()
    strict_load_deployment_state(deployment_model, deployment_state)
    deployment_topology = validate_product_model(
        deployment_model, deployment=True, require_initialized_norms=True
    )
    panel_size = min(32, data["sample_count"])
    panel_indices = torch.linspace(
        0, data["sample_count"] - 1, panel_size, device="cuda"
    ).round().long()
    export_proof = prove_deployment_equivalence(
        model,
        deployment_model,
        images[panel_indices].float().div(255),
        instructions[panel_indices],
        states[panel_indices],
    )
    state_copy_equal = all(
        torch.equal(deployment_model.state_dict()[key], value)
        for key, value in deployment_state.items()
    )
    if not state_copy_equal:
        raise RuntimeError("Strict deployment model differs from exported state")

    _, task_metadata_end = validate_task_mapping(official_tasks)
    cache_end = assert_cache_identity()
    sources_end = source_snapshot()
    end_static_audit = verify_transitive_static_audit_now()
    manifest_end = load_and_verify_source_manifest(args.source_manifest)
    if task_metadata_end != task_metadata_start:
        raise RuntimeError("Task mapping changed during training")
    if cache_end != cache_start:
        raise RuntimeError("Cache identity changed during training")
    if sources_end != sources_start or manifest_end != manifest_start:
        raise RuntimeError("Source identity changed during training")
    if file_sha256(args.source_manifest) != manifest_sha256:
        raise RuntimeError("Source manifest changed during training")
    verified_precalibration = load_training_only_precalibration_state(
        precalibration_output,
        expected_file_sha256=precalibration_identity["sha256"],
        expected_seed=args.seed,
        expected_mode=args.mode,
        expected_step_count=steps,
        expected_source_manifest_sha256=manifest_sha256,
        expected_manifest_bundle_sha256=manifest_start["manifest_bundle_sha256"],
        expected_source_bundle_sha256=source_bundle_sha256(sources_start),
        expected_training_config_sha256=training_config_record["sha256"],
    )
    if (
        verified_precalibration["artifact_bundle_sha256"]
        != precalibration_identity["artifact_bundle_sha256"]
    ):
        raise RuntimeError("Training-only pre-calibration artifact identity changed")
    checkpoint_output = checkpoint_path(args.run_root, args.seed, smoke)
    cpu_deployment_state = {
        key: value.detach().cpu() for key, value in deployment_state.items()
    }
    cpu_calibrated_training_state = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    calibration_state_transition = calibration_state_transition_proof(
        verified_precalibration["training_state"],
        cpu_calibrated_training_state,
        cpu_deployment_state,
        active_site_names=tuple(precalibration_running_ms_values),
    )
    del verified_precalibration
    del cpu_calibrated_training_state
    publish_checkpoint_exclusive(checkpoint_output, cpu_deployment_state)
    checkpoint_sha256 = file_sha256(checkpoint_output)
    reloaded_state = torch.load(checkpoint_output, map_location="cpu", weights_only=True)
    reloaded_model = ChiVLA(deployment_config)
    strict_load_deployment_state(reloaded_model, reloaded_state)
    validate_product_model(reloaded_model, deployment=True, require_initialized_norms=True)
    reloaded_equal = all(
        torch.equal(reloaded_model.state_dict()[key], value)
        for key, value in cpu_deployment_state.items()
    )
    if not reloaded_equal:
        raise RuntimeError("Serialized deployment checkpoint changed a state tensor")

    training_environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        **slurm_gpu_allocation_record(
            gpu=torch.cuda.get_device_name(0),
            compute_capability=compute_capability,
        ),
        "native_bfloat16": torch.cuda.is_bf16_supported(),
        "numpy": np.__version__,
        "libero": installed_version("libero"),
        "robosuite": installed_version("robosuite"),
        "mujoco": installed_version("mujoco"),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "autocast_dtype": "bfloat16",
    }
    metadata = {
        "schema": SCHEMA,
        "recipe_version": RECIPE_VERSION,
        "mode": args.mode,
        "seed": args.seed,
        "suite": "libero_object",
        "checkpoint": checkpoint_output.as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_state_keys_sha256": canonical_sha256(sorted(cpu_deployment_state)),
        "checkpoint_contains_deployment_state_only": True,
        "training_config": training_config_record,
        "deployment_config": config_record(deployment_config),
        "training_recipe": training_recipe(smoke),
        "training_topology_initial": training_topology,
        "training_topology_final": training_topology_final,
        "post_ema_state_validation": post_ema_state_validation,
        "training_only_precalibration_ema_state": precalibration_identity,
        "precalibration_runtime_direct_only_guard": precalibration_runtime_guard,
        "calibration_execution": {
            "resumed_from_training_only_precalibration_state": args.resume_precalibration,
            "resume_state_bitwise_exact": calibration_resume_state_bitwise_exact,
            "optimizer_steps_executed_in_this_process": 0
            if args.resume_precalibration
            else steps,
            "training_metrics": training_metrics,
            "current_process_elapsed_s": execution_elapsed_s,
        },
        "post_ema_simultaneous_rational_calibration": calibration_report,
        "post_ema_simultaneous_rational_calibration_attestation": (
            calibration_attestation
        ),
        "calibration_state_transition": calibration_state_transition,
        "deployment_topology": deployment_topology,
        "deployment_export": {
            "strict_load": True,
            "serialized_strict_reload": True,
            "retained_state_bitwise_equal": state_copy_equal and reloaded_equal,
            "removed_training_only_keys": list(stripped_keys),
            "removed_non_training_keys": [],
            "inference_equivalence": export_proof,
        },
        "training_parameter_count": model.num_params(),
        "deployment_parameter_count": deployment_model.num_params(),
        "cache": cache_start,
        "cache_structure": {
            key: data[key]
            for key in (
                "frame_count",
                "episode_count",
                "sample_count",
                "task_frame_counts",
                "task_sample_counts",
            )
        },
        "normalization": data["normalization"],
        "dataset_tasks": {str(key): value for key, value in dataset_tasks.items()},
        "official_tasks": {str(key): value for key, value in official_tasks.items()},
        "task_metadata": task_metadata_start,
        "vocab": vocab,
        "vocab_sha256": canonical_sha256(vocab),
        "source_manifest": {
            "path": args.source_manifest.as_posix(),
            "sha256": manifest_sha256,
            "bundle_sha256": source_bundle_sha256(sources_start),
            "authenticated_inputs_bundle_sha256": manifest_start[
                "authenticated_inputs_bundle_sha256"
            ],
            "launch_bundle_sha256": manifest_start["launch_bundle_sha256"],
            "manifest_bundle_sha256": manifest_start["manifest_bundle_sha256"],
        },
        "preflight_certificate": preflight_certificate,
        "source_snapshot_start": sources_start,
        "source_snapshot_end": sources_end,
        "transitive_direct_only_static_audit": _TRANSITIVE_STATIC_AUDIT,
        "end_transitive_direct_only_static_audit": end_static_audit,
        "training_environment": training_environment,
        "claim_boundary": DEPLOYMENT_CLAIM_BOUNDARY,
        "deployment_publication_requires_completion_marker": True,
    }
    metadata_output = metadata_path(args.run_root, args.seed, smoke)
    write_json_exclusive(metadata_output, metadata)
    metadata_sha256 = file_sha256(metadata_output)
    result = {
        "schema": f"{SCHEMA}_training_result",
        "recipe_version": RECIPE_VERSION,
        "mode": args.mode,
        "seed": args.seed,
        "checkpoint": checkpoint_output.as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "metadata": metadata_output.as_posix(),
        "metadata_sha256": metadata_sha256,
        "elapsed_s": elapsed_s,
        "training_elapsed_s": training_metrics["training_elapsed_s"],
        "current_process_elapsed_s": execution_elapsed_s,
        "resumed_from_training_only_precalibration_state": args.resume_precalibration,
        "resume_state_bitwise_exact": calibration_resume_state_bitwise_exact,
        "optimizer_steps_executed_in_this_process": 0
        if args.resume_precalibration
        else steps,
        "steps_per_s": steps / float(training_metrics["training_elapsed_s"]),
        "examples_per_s": steps
        * batch_size
        / float(training_metrics["training_elapsed_s"]),
        "last_loss": last_loss,
        "training_peak_allocated_gb": training_metrics["peak_allocated_gb"],
        "training_peak_reserved_gb": training_metrics["peak_reserved_gb"],
        "current_process_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "current_process_peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "peak_allocated_gb": max(
            float(training_metrics["peak_allocated_gb"]),
            torch.cuda.max_memory_allocated() / 1e9,
        ),
        "peak_reserved_gb": max(
            float(training_metrics["peak_reserved_gb"]),
            torch.cuda.max_memory_reserved() / 1e9,
        ),
        "deployment_export": metadata["deployment_export"],
        "post_ema_state_validation": post_ema_state_validation,
        "training_only_precalibration_ema_state": precalibration_identity,
        "post_ema_simultaneous_rational_calibration": calibration_report,
        "post_ema_simultaneous_rational_calibration_attestation": (
            calibration_attestation
        ),
        "calibration_state_transition": calibration_state_transition,
        "source_manifest_sha256": manifest_sha256,
        "manifest_bundle_sha256": manifest_start["manifest_bundle_sha256"],
        "preflight_certificate": preflight_certificate,
        "claim_boundary": DEPLOYMENT_CLAIM_BOUNDARY,
        "deployment_publication_requires_completion_marker": True,
    }
    final_runtime_guard = assert_direct_only_runtime_guard(
        direct_only_runtime_report(), exact_allowed_calls=()
    )
    if final_runtime_guard["prohibited_attempt_count"] != 0:
        raise RuntimeError("A prohibited numerical route was attempted during training")
    result["transitive_direct_only_static_audit"] = _TRANSITIVE_STATIC_AUDIT
    result["runtime_direct_only_guard_at_import"] = _RUNTIME_GUARD_AT_IMPORT
    result["runtime_direct_only_guard_final"] = final_runtime_guard
    result_output = training_result_path(args.run_root, args.seed, smoke)
    write_json_exclusive(result_output, result)
    completion_output = deployment_completion_path(
        args.run_root, args.seed, smoke
    )
    completion = publish_deployment_completion_exclusive(
        completion_output,
        seed=args.seed,
        mode=args.mode,
        checkpoint=checkpoint_output,
        metadata=metadata_output,
        training_result=result_output,
        precalibration_ema_state=precalibration_output,
        source_manifest_sha256=manifest_sha256,
        manifest_bundle_sha256=manifest_start["manifest_bundle_sha256"],
    )
    print("RESULT", json.dumps(result, ensure_ascii=False), flush=True)
    print("COMPLETION", json.dumps(completion, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
