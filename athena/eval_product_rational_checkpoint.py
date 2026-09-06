#!/usr/bin/env python3
"""Evaluate one immutable ProductRoutingHead + Padé-rational checkpoint shard."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

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

    def canonical_digest(value: Any) -> str:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    payload = json.loads(
        manifest_path.read_text(),
        object_pairs_hook=reject_duplicates,
        parse_constant=_reject_json_constant,
        parse_float=_finite_json_float,
    )
    if payload.get("schema") != _MANIFEST_SCHEMA:
        raise RuntimeError("Pre-import source manifest schema differs")
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
            if not path.is_file():
                raise RuntimeError(f"Pre-import lane file is missing: {relative}")
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
    from scripts.odt_direct_only_compliance import audit_direct_only_launch

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
else:
    _TRANSITIVE_STATIC_AUDIT = None

import numpy as np
import torch

from scripts.odt_direct_only_compliance import (
    EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT,
    EXPECTED_RUNTIME_GUARD_ENTRYPOINTS,
)
from athena.product_rational_protocol import (
    ACTION_DIM,
    ACTION_HORIZON,
    CAPABILITY_SIMULATOR_BOUNDARY,
    CACHE_SAMPLE_COUNT,
    CALIBRATION_INITIALIZER_LABEL,
    CALIBRATION_RECOVERY_SCOPE,
    CALIBRATION_RESUME_PROOF_SCHEMA,
    DEPLOYMENT_CLAIM_BOUNDARY,
    EVALUATION_MATMUL_PRECISION,
    EXPECTED_EVALUATION_ENVIRONMENT,
    EXPECTED_TASK_PROTOCOL,
    FINAL_CALIBRATION_AUDIT_MAX_RELATIVE_CHANGE,
    FINAL_CALIBRATION_PASSES,
    OFFICIAL_TASK_LANGUAGES,
    OFFICIAL_INIT_STATE_ROW_HASH_LIST_SHA256,
    PRECALIBRATION_ARTIFACT_ROLE,
    PRECALIBRATION_SCHEMA,
    PRODUCT_RUNNING_MS_IDENTITY_FLOOR,
    RECIPE_VERSION,
    RESOLUTION,
    SCHEMA,
    STATE_DIM,
    assert_cache_identity,
    active_rational_norms,
    active_running_ms_values,
    build_vocab,
    calibration_resume_proof_path,
    canonical_sha256,
    checkpoint_path,
    config_record,
    evaluation_protocol,
    evaluation_result_path,
    file_sha256,
    json_type_exact_equal,
    load_and_verify_source_manifest,
    load_and_validate_deployment_completion,
    load_training_only_precalibration_state,
    load_suite,
    make_product_config,
    metadata_path,
    deployment_completion_path,
    precalibration_state_path,
    preflight_result_path,
    read_json_object_physical,
    read_training_metadata,
    runtime_guard_record_is_closed,
    source_snapshot,
    source_bundle_sha256,
    static_audit_record_is_closed,
    strict_load_deployment_state,
    task_languages,
    tensor_state_sha256,
    training_recipe,
    training_result_path,
    validate_precalibration_to_deployment_transition,
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
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task-start", type=int, required=True)
    parser.add_argument("--task-end", type=int, required=True)
    arguments = sys.argv[1:]
    parsed = parser.parse_args(arguments)
    for option in (
        "--run-root",
        "--source-manifest",
        "--mode",
        "--seed",
        "--task-start",
        "--task-end",
    ):
        if _option_count(arguments, option) != 1:
            parser.error(f"{option} must occur exactly once")
    return parsed


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


def preprocess_agentview_image(observation: dict[str, Any]) -> torch.Tensor:
    """Apply the frozen 180-degree image convention and one [0, 1] scaling."""
    rotated = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    if rotated.shape != (RESOLUTION, RESOLUTION, 3):
        raise RuntimeError(f"Agent-view image has invalid shape {rotated.shape}")
    if rotated.dtype != np.uint8:
        raise RuntimeError(f"Agent-view image has invalid dtype {rotated.dtype}")
    image = torch.from_numpy(rotated.copy()).permute(2, 0, 1).float().div(255)
    if image.shape != (3, RESOLUTION, RESOLUTION):
        raise RuntimeError(f"Preprocessed image has invalid shape {tuple(image.shape)}")
    if not bool(torch.isfinite(image).all()) or not bool(
        ((image >= 0) & (image <= 1)).all()
    ):
        raise RuntimeError("Preprocessed image is outside the frozen [0, 1] range")
    return image


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def evaluation_environment_record(device_name: str) -> dict[str, Any]:
    """Normalize external version objects to the literal JSON string domain."""
    return {
        "gpu": device_name,
        "inference_dtype": "float32",
        "matmul_precision": torch.get_float32_matmul_precision(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "libero": installed_version("libero"),
        "robosuite": installed_version("robosuite"),
        "mujoco": installed_version("mujoco"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    }


def official_init_states(suite: Any, task_index: int) -> Any:
    original_load = torch.load

    def trusted_packaged_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = trusted_packaged_load
    try:
        states = suite.get_task_init_states(task_index)
    except Exception as exc:
        raise RuntimeError(
            f"Official packaged initial states failed for task {task_index}"
        ) from exc
    finally:
        torch.load = original_load
    if states is None or len(states) != 50:
        raise RuntimeError(
            f"Task {task_index} must expose exactly 50 packaged states, got "
            f"{None if states is None else len(states)}"
        )
    hashes = [array_sha256(np.asarray(state)) for state in states]
    if len(set(hashes)) != 50:
        raise RuntimeError(f"Task {task_index} packaged initial states are not unique")
    return states


def robot_state(observation: dict[str, Any]) -> np.ndarray:
    # Import the official conversion only when the external capability simulator is
    # actually running. Merely importing this module therefore does not pull the
    # simulator into the authenticated weight-only numerical closure.
    from robosuite.utils.transform_utils import quat2axisangle

    value = np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float64),
            quat2axisangle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64),
        )
    )
    if not np.isfinite(value).all() or value.size > STATE_DIM:
        raise RuntimeError("Robot state is non-finite or wider than the frozen state input")
    if value.size < STATE_DIM:
        value = np.pad(value, (0, STATE_DIM - value.size))
    return value.astype(np.float32)


def validate_environment_transition(
    transition: Any,
    *,
    phase: str,
    task_index: int,
    episode_index: int,
    transition_index: int,
) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    """Validate every simulator transition before its observation or outcome is used."""

    if phase not in {"settle", "policy"}:
        raise ValueError(f"Unknown transition phase {phase!r}")
    location = (
        f"phase={phase} task={task_index} episode={episode_index} "
        f"transition={transition_index}"
    )
    if type(transition) is not tuple or len(transition) != 4:
        raise RuntimeError(f"Simulator transition is not an exact 4-tuple: {location}")
    observation, reward, done, info = transition
    if type(info) is not dict:
        raise RuntimeError(f"Simulator transition info is not a dict: {location}")

    if not isinstance(observation, dict) or not observation:
        raise RuntimeError(f"Transition observation is missing or malformed: {location}")
    required = {
        "agentview_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    }
    if not required.issubset(observation):
        raise RuntimeError(
            "Transition observation lacks a model input: "
            f"{location} "
            f"missing={sorted(required - set(observation))}"
        )
    exact_model_inputs = {
        "agentview_image": ((RESOLUTION, RESOLUTION, 3), np.dtype(np.uint8)),
        "robot0_eef_pos": ((3,), np.dtype(np.float64)),
        "robot0_eef_quat": ((4,), np.dtype(np.float64)),
        "robot0_gripper_qpos": ((2,), np.dtype(np.float64)),
    }
    for name, (shape, dtype) in exact_model_inputs.items():
        value = observation[name]
        if type(value) is not np.ndarray or value.shape != shape or value.dtype != dtype:
            raise RuntimeError(
                "Transition model input shape or dtype differs: "
                f"{location} field={name} "
                f"shape={getattr(value, 'shape', None)!r} "
                f"dtype={getattr(value, 'dtype', None)!r}"
            )
    nonfinite_fields: list[str] = []
    nonnumeric_fields: list[str] = []
    for name, value in observation.items():
        array = np.asarray(value)
        if array.dtype.kind not in "biufc":
            nonnumeric_fields.append(str(name))
        elif not bool(np.isfinite(array).all()):
            nonfinite_fields.append(str(name))
    if nonnumeric_fields or nonfinite_fields:
        raise RuntimeError(
            "Transition observation is not entirely finite numeric data: "
            f"{location} "
            f"nonnumeric={sorted(nonnumeric_fields)} "
            f"nonfinite={sorted(nonfinite_fields)}"
        )
    reward_array = np.asarray(reward)
    if (
        isinstance(reward, (bool, np.bool_))
        or reward_array.ndim != 0
        or reward_array.dtype.kind not in "iuf"
    ):
        raise RuntimeError(f"Transition reward is not a non-boolean numeric scalar: {location}")
    reward_value = float(reward)
    if not np.isfinite(reward_value):
        raise RuntimeError(f"Transition reward is non-finite: {location} reward={reward_value!r}")
    if type(done) not in {bool, np.bool_}:
        raise RuntimeError(
            f"Transition done flag is not a scalar bool: {location} "
            f"type={type(done).__name__}"
        )
    done_value = bool(done)
    if phase == "settle" and (reward_value > 0.0 or done_value):
        raise RuntimeError(
            "Canonical initial state reached success or termination during dummy settling: "
            f"{location} reward={reward_value!r} done={done_value}"
        )
    return observation, reward_value, done_value, info


def validate_settle_transition(
    observation: Any,
    reward: Any,
    done: Any,
    *,
    task_index: int,
    episode_index: int,
    settle_index: int,
) -> None:
    """Compatibility wrapper for the strict unified transition validator."""

    validate_environment_transition(
        (observation, reward, done, {}),
        phase="settle",
        task_index=task_index,
        episode_index=episode_index,
        transition_index=settle_index,
    )


@torch.inference_mode()
def public_action_and_component_oracle(
    model: ChiVLA,
    image: torch.Tensor,
    instruction: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    captured_center: list[torch.Tensor] = []
    captured_factors: list[torch.Tensor | None] = [None] * model.product_head.G
    captured_gates: list[torch.Tensor | None] = [None] * model.product_head.G
    handles = [
        model.product_head.center.register_forward_hook(
            lambda _module, _inputs, output: captured_center.append(output)
        )
    ]
    for index, module in enumerate(model.product_head.factors):
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, *, position=index: captured_factors.__setitem__(
                    position, output
                )
            )
        )
    for index, module in enumerate(model.product_head.gates):
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, *, position=index: captured_gates.__setitem__(
                    position, output
                )
            )
        )
    embodiment = torch.zeros(image.shape[0], dtype=torch.long, device=image.device)
    try:
        public_actions, loss = model(image, instruction, state, embodiment)
    finally:
        for handle in handles:
            handle.remove()
    if loss is not None:
        raise RuntimeError("Deployment inference unexpectedly returned a loss")
    if (
        len(captured_center) != 1
        or any(value is None for value in captured_factors)
        or any(value is None for value in captured_gates)
    ):
        raise RuntimeError("Public Product forward did not exercise every component once")
    center = captured_center[0]
    factors = torch.stack([value for value in captured_factors if value is not None], dim=1)
    gates = torch.cat([value for value in captured_gates if value is not None], dim=-1)
    signs = torch.where(gates > 0, 1.0, -1.0).to(center.dtype)
    explicit = model.product_head.action_for_signs(center, factors, signs).reshape(
        -1, ACTION_HORIZON, ACTION_DIM
    )
    checks = {
        "public_action_equals_fixed_sign_decode": torch.equal(public_actions, explicit),
        "finite_center": bool(torch.isfinite(center).all()),
        "finite_factors": bool(torch.isfinite(factors).all()),
        "finite_gates": bool(torch.isfinite(gates).all()),
        "finite_actions": bool(torch.isfinite(public_actions).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Product decode boundary proof failed: {checks}")
    return public_actions, {
        "checks": checks,
        "zero_gate_logit_count": int((gates == 0).sum()),
        "positive_sign_count": int((signs > 0).sum()),
        "total_sign_count": signs.numel(),
        "positive_sign_fraction": float((signs > 0).float().mean()),
        "fixed_sign_rule": "+1 iff gate_logit > 0, else -1",
        "component_shapes": {
            "center": list(center.shape),
            "factors": list(factors.shape),
            "gates": list(gates.shape),
        },
    }


@torch.inference_mode()
def verify_decode_boundary(
    model: ChiVLA,
    image: torch.Tensor,
    instruction: torch.Tensor,
    state: torch.Tensor,
) -> dict[str, Any]:
    _, report = public_action_and_component_oracle(
        model, image, instruction, state
    )
    return report


@torch.inference_mode()
def predict_chunk(
    model: ChiVLA,
    observation: dict[str, Any],
    instruction: torch.Tensor,
    normalization: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, int]]:
    image = preprocess_agentview_image(observation).unsqueeze(0).cuda()
    raw_state = robot_state(observation)
    state = torch.from_numpy(
        ((raw_state - normalization["state_mean"]) / normalization["state_std"]).astype(
            np.float32
        )
    ).unsqueeze(0).cuda()
    public_action, oracle = public_action_and_component_oracle(
        model, image, instruction, state
    )
    normalized_action = public_action[0]
    output = (
        normalized_action.cpu().numpy() * normalization["action_std"]
        + normalization["action_mean"]
    )
    if not np.isfinite(output).all():
        raise RuntimeError("Denormalized action is non-finite")
    return output.astype(np.float32), {
        "zero_gate_logits": int(oracle["zero_gate_logit_count"]),
        "positive_signs": int(oracle["positive_sign_count"]),
        "total_signs": int(oracle["total_sign_count"]),
        "public_component_mismatch": 0,
    }


def validated_normalization(metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    source = metadata.get("normalization")
    if not isinstance(source, dict):
        raise RuntimeError("Training metadata omits normalization statistics")
    result = {
        "state_mean": np.asarray(source.get("state_mean"), dtype=np.float32),
        "state_std": np.asarray(source.get("state_std"), dtype=np.float32),
        "action_mean": np.asarray(source.get("action_mean"), dtype=np.float32),
        "action_std": np.asarray(source.get("action_std"), dtype=np.float32),
    }
    expected_shapes = {
        "state_mean": (STATE_DIM,),
        "state_std": (STATE_DIM,),
        "action_mean": (ACTION_DIM,),
        "action_std": (ACTION_DIM,),
    }
    for key, value in result.items():
        if value.shape != expected_shapes[key] or not np.isfinite(value).all():
            raise RuntimeError(f"Normalization statistic {key} is malformed")
        if key.endswith("std") and not bool((value > 0).all()):
            raise RuntimeError(f"Normalization statistic {key} is not strictly positive")
    return result


def verify_training_artifact(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    official_tasks: dict[int, str],
    vocab: dict[str, int],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    ChiVLA,
    dict[str, np.ndarray],
    dict[str, str],
]:
    smoke = args.mode == "smoke"
    expected_checkpoint = checkpoint_path(args.run_root, args.seed, smoke)
    expected_metadata = metadata_path(args.run_root, args.seed, smoke)
    expected_result = training_result_path(args.run_root, args.seed, smoke)
    expected_precalibration = precalibration_state_path(
        args.run_root, args.seed, smoke
    )
    expected_completion = deployment_completion_path(
        args.run_root, args.seed, smoke
    )
    expected_preflight = preflight_result_path(args.run_root)
    if not expected_result.is_file() or expected_result.is_symlink():
        raise RuntimeError("Exclusive training result is missing or nonphysical")
    if not expected_checkpoint.is_file() or expected_checkpoint.is_symlink():
        raise RuntimeError("Deployment checkpoint is missing or nonphysical")
    if not expected_precalibration.is_file() or expected_precalibration.is_symlink():
        raise RuntimeError(
            "Training-only pre-calibration EMA state is missing or nonphysical"
        )
    completion = load_and_validate_deployment_completion(
        expected_completion,
        expected_seed=args.seed,
        expected_mode=args.mode,
        expected_checkpoint=expected_checkpoint,
        expected_metadata=expected_metadata,
        expected_training_result=expected_result,
        expected_precalibration_ema_state=expected_precalibration,
        expected_source_manifest_sha256=file_sha256(args.source_manifest),
        expected_manifest_bundle_sha256=manifest["manifest_bundle_sha256"],
    )
    training_result = read_json_object_physical(
        expected_result, label="Training result"
    )
    metadata = read_training_metadata(expected_metadata)
    preflight = validate_preflight_certificate(
        expected_preflight, args.source_manifest, manifest
    )
    artifact_hashes = {
        "training_result": file_sha256(expected_result),
        "metadata": file_sha256(expected_metadata),
        "checkpoint": file_sha256(expected_checkpoint),
        "preflight": file_sha256(expected_preflight),
        "precalibration_ema_state": file_sha256(expected_precalibration),
        "deployment_completion": file_sha256(expected_completion),
    }
    resume_proof: dict[str, Any] | None = None
    if smoke:
        proof_path = calibration_resume_proof_path(args.run_root)
        if not proof_path.is_file() or proof_path.is_symlink():
            raise RuntimeError("Same-allocation calibration resume proof is missing")
        resume_proof = read_json_object_physical(
            proof_path, label="Same-allocation calibration resume proof"
        )
        artifact_hashes["calibration_resume_proof"] = file_sha256(proof_path)
    stored_static = training_result.get("transitive_direct_only_static_audit", {})
    stored_import_guard = training_result.get("runtime_direct_only_guard_at_import", {})
    stored_final_guard = training_result.get("runtime_direct_only_guard_final", {})
    calibration = metadata.get("post_ema_simultaneous_rational_calibration", {})
    post_ema_validation = metadata.get("post_ema_state_validation", {})
    metadata_precalibration = metadata.get(
        "training_only_precalibration_ema_state", {}
    )
    result_precalibration = training_result.get(
        "training_only_precalibration_ema_state", {}
    )
    training_config = make_product_config(len(vocab), deployment=False)
    expected_training_config_record = config_record(training_config)
    if not json_type_exact_equal(
        metadata.get("training_config"), expected_training_config_record
    ):
        raise RuntimeError("Training config differs before pre-calibration validation")
    loaded_precalibration = load_training_only_precalibration_state(
        expected_precalibration,
        expected_file_sha256=artifact_hashes["precalibration_ema_state"],
        expected_seed=args.seed,
        expected_mode=args.mode,
        expected_step_count=training_recipe(smoke)["steps"],
        expected_source_manifest_sha256=file_sha256(args.source_manifest),
        expected_manifest_bundle_sha256=manifest["manifest_bundle_sha256"],
        expected_source_bundle_sha256=source_bundle_sha256(
            manifest["source_closure"]
        ),
        expected_training_config_sha256=expected_training_config_record["sha256"],
    )
    loaded_precalibration_identity = {
        "path": expected_precalibration.as_posix(),
        "sha256": artifact_hashes["precalibration_ema_state"],
        "schema": loaded_precalibration["schema"],
        "artifact_role": loaded_precalibration["artifact_role"],
        "deployment_eligible": loaded_precalibration["deployment_eligible"],
        "artifact_bundle_sha256": loaded_precalibration["artifact_bundle_sha256"],
        "training_state_sha256": loaded_precalibration["training_state_sha256"],
        "training_state_key_count": loaded_precalibration["training_state_key_count"],
        "training_state_keys_sha256": loaded_precalibration[
            "training_state_keys_sha256"
        ],
        "training_metrics": loaded_precalibration["training_metrics"],
        "recovery_scope": loaded_precalibration["recovery_scope"],
        "cross_version_recovery_supported": loaded_precalibration[
            "cross_version_recovery_supported"
        ],
    }
    if not json_type_exact_equal(
        loaded_precalibration_identity, metadata_precalibration
    ):
        raise RuntimeError(
            "Independently loaded pre-calibration identity differs from metadata"
        )
    training_state_model = ChiVLA(training_config)
    incompatibility = training_state_model.load_state_dict(
        loaded_precalibration["training_state"], strict=True
    )
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"Pre-calibration training topology differs: {incompatibility}")
    if (
        tensor_state_sha256(training_state_model.state_dict())
        != loaded_precalibration["training_state_sha256"]
    ):
        raise RuntimeError("Pre-calibration model load changed an exact tensor")
    validate_product_model(
        training_state_model, deployment=False, require_initialized_norms=False
    )
    initial_running_ms_values = active_running_ms_values(training_state_model)
    invalid_initializers = [
        name
        for name, norm in active_rational_norms(training_state_model)
        if not bool(norm.initialized.item())
        or not bool(torch.isfinite(norm.running_ms).all())
        or norm.running_ms.numel() != 1
        or float(norm.running_ms.item()) < PRODUCT_RUNNING_MS_IDENTITY_FLOOR
    ]
    if invalid_initializers:
        raise RuntimeError(
            "Pre-calibration training state has invalid active initializers: "
            f"{invalid_initializers}"
        )
    if (
        not json_type_exact_equal(
            post_ema_validation,
            loaded_precalibration["post_ema_state_validation"],
        )
        or not json_type_exact_equal(
            training_result.get("post_ema_state_validation"),
            loaded_precalibration["post_ema_state_validation"],
        )
    ):
        raise RuntimeError("Post-EMA validation differs across immutable artifacts")
    if (
        canonical_sha256(initial_running_ms_values)
        != post_ema_validation.get("initializer_running_ms_sha256")
        or canonical_sha256(initial_running_ms_values)
        != calibration.get("initializer_running_ms_sha256")
    ):
        raise RuntimeError("Actual pre-calibration running_ms digest differs")
    if smoke:
        strict_resume_proof = validate_same_allocation_calibration_resume_proof(
            resume_proof,
            expected_source_manifest_sha256=file_sha256(args.source_manifest),
            expected_manifest_bundle_sha256=manifest["manifest_bundle_sha256"],
            expected_preflight_sha256=artifact_hashes["preflight"],
            expected_source_closure=manifest["source_closure"],
            expected_static_audit=_TRANSITIVE_STATIC_AUDIT,
            expected_checkpoint_sha256=artifact_hashes["checkpoint"],
            expected_metadata_sha256=artifact_hashes["metadata"],
            expected_training_result_sha256=artifact_hashes["training_result"],
            expected_precalibration_sha256=artifact_hashes[
                "precalibration_ema_state"
            ],
            expected_precalibration_training_state_sha256=(
                loaded_precalibration["training_state_sha256"]
            ),
            expected_completion_sha256=artifact_hashes["deployment_completion"],
            expected_completion_bundle_sha256=completion["artifact_bundle_sha256"],
            expected_training_environment=metadata["training_environment"],
            expected_calibration_attestation=metadata[
                "post_ema_simultaneous_rational_calibration_attestation"
            ],
            expected_state_transition=metadata["calibration_state_transition"],
            expected_deployment_equivalence=metadata["deployment_export"][
                "inference_equivalence"
            ],
        )
        proof_static = resume_proof.get("transitive_direct_only_static_audit", {})
        proof_runtime = resume_proof.get("runtime_direct_only_guard", {})
        proof_bitwise_checks = resume_proof.get("deployment_equivalence", {}).get(
            "bitwise_checks", {}
        )
        proof_allocation = resume_proof.get("allocation", {})
        allocation_keys = (
            "gpu",
            "compute_capability",
            "slurm_job_id",
            "slurm_job_gpus",
            "cuda_visible_devices",
        )
        reference_allocation = metadata.get("training_environment", {})
        reference_replay = metadata.get("deployment_export", {}).get(
            "inference_equivalence", {}
        )
        proof_checks = {
            "schema": resume_proof.get("schema")
            == CALIBRATION_RESUME_PROOF_SCHEMA,
            "strict_validator": strict_resume_proof.get("validated") is True
            and isinstance(strict_resume_proof.get("conditions"), dict)
            and all(
                type(value) is bool and value
                for value in strict_resume_proof["conditions"].values()
            ),
            "identity": type(resume_proof.get("seed")) is int
            and resume_proof.get("seed") == 0
            and resume_proof.get("mode") == "smoke",
            "completion": resume_proof.get("fresh_deployment_completion_sha256")
            == artifact_hashes["deployment_completion"]
            and resume_proof.get("fresh_deployment_completion_bundle_sha256")
            == completion["artifact_bundle_sha256"],
            "precalibration": resume_proof.get("precalibration_ema_state_sha256")
            == artifact_hashes["precalibration_ema_state"]
            and resume_proof.get("precalibration_training_state_sha256")
            == loaded_precalibration["training_state_sha256"],
            "external_authority": resume_proof.get("external_sha_authority_supplied")
            is True,
            "fresh_artifact_hashes": resume_proof.get("fresh_checkpoint_sha256")
            == artifact_hashes["checkpoint"]
            and resume_proof.get("fresh_metadata_sha256")
            == artifact_hashes["metadata"]
            and resume_proof.get("fresh_training_result_sha256")
            == artifact_hashes["training_result"],
            "zero_training": type(resume_proof.get("optimizer_steps_executed"))
            is int
            and resume_proof.get("optimizer_steps_executed") == 0,
            "same_allocation": resume_proof.get(
                "same_slurm_gpu_allocation_as_fresh_smoke"
            )
            is True
            and isinstance(proof_allocation, dict)
            and set(proof_allocation) == set(allocation_keys)
            and all(
                isinstance(proof_allocation.get(key), str)
                and bool(proof_allocation.get(key))
                for key in (
                    "slurm_job_id",
                    "slurm_job_gpus",
                    "cuda_visible_devices",
                )
            )
            and proof_allocation.get("gpu") == "NVIDIA RTX A6000"
            and json_type_exact_equal(
                proof_allocation.get("compute_capability"), [8, 6]
            )
            and all(
                json_type_exact_equal(
                    proof_allocation.get(key), reference_allocation.get(key)
                )
                for key in allocation_keys
            ),
            "exact_reports": resume_proof.get(
                "fresh_and_resume_calibration_report_bitwise_equal"
            )
            is True
            and resume_proof.get(
                "fresh_and_resume_final_running_ms_bitwise_equal"
            )
            is True,
            "exact_state": resume_proof.get(
                "fresh_and_resume_deployment_state_bitwise_equal"
            )
            is True
            and json_type_exact_equal(
                resume_proof.get("removed_training_only_keys"),
                ["teacher_head.bias", "teacher_head.weight"],
            ),
            "exact_raw_and_public": resume_proof.get(
                "fresh_and_resume_raw_components_and_actions_bitwise_equal"
            )
            is True
            and isinstance(reference_replay, dict)
            and bool(reference_replay)
            and isinstance(reference_replay.get("output_tensor_sha256"), str)
            and len(reference_replay["output_tensor_sha256"]) == 64
            and all(
                character in "0123456789abcdef"
                for character in reference_replay["output_tensor_sha256"]
            )
            and resume_proof.get("fresh_output_tensor_sha256")
            == resume_proof.get("resume_output_tensor_sha256")
            == reference_replay.get("output_tensor_sha256")
            and json_type_exact_equal(
                resume_proof.get("deployment_equivalence"), reference_replay
            )
            and isinstance(proof_bitwise_checks, dict)
            and bool(proof_bitwise_checks)
            and all(type(value) is bool and value for value in proof_bitwise_checks.values()),
            "calibration_attestation": json_type_exact_equal(
                resume_proof.get("calibration_attestation"),
                metadata.get("post_ema_simultaneous_rational_calibration_attestation"),
            ),
            "state_transition": json_type_exact_equal(
                resume_proof.get("calibration_state_transition"),
                metadata.get("calibration_state_transition"),
            ),
            "manifest": resume_proof.get("source_manifest_sha256")
            == file_sha256(args.source_manifest)
            and resume_proof.get("manifest_bundle_sha256")
            == manifest["manifest_bundle_sha256"]
            and resume_proof.get("preflight_certificate_sha256")
            == artifact_hashes["preflight"],
            "source_snapshots": json_type_exact_equal(
                resume_proof.get("source_snapshot_start"), manifest["source_closure"]
            )
            and json_type_exact_equal(
                resume_proof.get("source_snapshot_end"), manifest["source_closure"]
            ),
            "static": json_type_exact_equal(proof_static, _TRANSITIVE_STATIC_AUDIT),
            "runtime": isinstance(proof_runtime, dict)
            and type(proof_runtime.get("patched_entrypoint_count")) is int
            and type(proof_runtime.get("allowed_call_count")) is int
            and proof_runtime.get("allowed_call_count") == 0
            and json_type_exact_equal(proof_runtime.get("allowed_calls"), [])
            and type(proof_runtime.get("prohibited_attempt_count")) is int
            and proof_runtime.get("prohibited_attempt_count") == 0
            and json_type_exact_equal(proof_runtime.get("prohibited_attempts"), []),
            "scope": resume_proof.get("recovery_scope")
            == CALIBRATION_RECOVERY_SCOPE
            and resume_proof.get("cross_version_recovery_supported") is False,
        }
        failed_proof = sorted(
            name for name, passed in proof_checks.items() if not passed
        )
        if failed_proof:
            raise RuntimeError(
                f"Same-allocation calibration resume proof failed: {failed_proof}"
            )
    del training_state_model
    inference_export = metadata.get("deployment_export", {}).get(
        "inference_equivalence", {}
    )
    fixed = {
        "result_schema": training_result.get("schema")
        == f"{SCHEMA}_training_result",
        "result_recipe": training_result.get("recipe_version") == RECIPE_VERSION,
        "result_mode": training_result.get("mode") == args.mode,
        "result_seed": type(training_result.get("seed")) is int
        and training_result.get("seed") == args.seed,
        "result_checkpoint_path": training_result.get("checkpoint")
        == expected_checkpoint.as_posix(),
        "result_checkpoint_sha": training_result.get("checkpoint_sha256")
        == artifact_hashes["checkpoint"],
        "result_metadata_path": training_result.get("metadata")
        == expected_metadata.as_posix(),
        "result_metadata_sha": training_result.get("metadata_sha256")
        == artifact_hashes["metadata"],
        "mode": metadata.get("mode") == args.mode,
        "seed": type(metadata.get("seed")) is int
        and metadata.get("seed") == args.seed,
        "suite": metadata.get("suite") == "libero_object",
        "checkpoint_path": metadata.get("checkpoint") == expected_checkpoint.as_posix(),
        "source_snapshot_start": json_type_exact_equal(
            metadata.get("source_snapshot_start"), manifest["source_closure"]
        ),
        "source_snapshot_end": json_type_exact_equal(
            metadata.get("source_snapshot_end"), manifest["source_closure"]
        ),
        "manifest_sha": metadata.get("source_manifest", {}).get("sha256")
        == file_sha256(args.source_manifest),
        "manifest_bundle": metadata.get("source_manifest", {}).get(
            "manifest_bundle_sha256"
        )
        == manifest["manifest_bundle_sha256"],
        "official_tasks": json_type_exact_equal(
            metadata.get("official_tasks"),
            {str(key): value for key, value in official_tasks.items()},
        ),
        "vocab": json_type_exact_equal(metadata.get("vocab"), vocab),
        "vocab_sha": metadata.get("vocab_sha256") == canonical_sha256(vocab),
        "deployment_only": metadata.get("checkpoint_contains_deployment_state_only") is True,
        "retained_state_exact": metadata.get("deployment_export", {}).get(
            "retained_state_bitwise_equal"
        )
        is True,
        "deployment_output_digest": isinstance(inference_export, dict)
        and isinstance(inference_export.get("output_tensor_sha256"), str)
        and len(inference_export["output_tensor_sha256"]) == 64
        and all(
            character in "0123456789abcdef"
            for character in inference_export["output_tensor_sha256"]
        )
        and isinstance(inference_export.get("bitwise_checks"), dict)
        and bool(inference_export["bitwise_checks"])
        and all(
            value is True
            for value in inference_export["bitwise_checks"].values()
        ),
        "result_export_matches_metadata": json_type_exact_equal(
            training_result.get("deployment_export"),
            metadata.get("deployment_export"),
        ),
        "result_calibration_matches_metadata": json_type_exact_equal(
            training_result.get("post_ema_simultaneous_rational_calibration"),
            calibration,
        ),
        "result_state_transition_matches_metadata": json_type_exact_equal(
            training_result.get("calibration_state_transition"),
            metadata.get("calibration_state_transition"),
        ),
        "completion_marker_required_by_producer": metadata.get(
            "deployment_publication_requires_completion_marker"
        )
        is True
        and training_result.get(
            "deployment_publication_requires_completion_marker"
        )
        is True,
        "metadata_preflight": json_type_exact_equal(
            metadata.get("preflight_certificate"), preflight
        ),
        "result_preflight": json_type_exact_equal(
            training_result.get("preflight_certificate"), preflight
        ),
        "precalibration_identity_join": json_type_exact_equal(
            metadata_precalibration, result_precalibration
        ),
        "precalibration_path": metadata_precalibration.get("path")
        == expected_precalibration.as_posix(),
        "precalibration_sha": metadata_precalibration.get("sha256")
        == artifact_hashes["precalibration_ema_state"],
        "precalibration_schema": metadata_precalibration.get("schema")
        == PRECALIBRATION_SCHEMA,
        "precalibration_training_only": metadata_precalibration.get("artifact_role")
        == PRECALIBRATION_ARTIFACT_ROLE
        and metadata_precalibration.get("deployment_eligible") is False,
        "precalibration_recovery_scope": metadata_precalibration.get(
            "recovery_scope"
        )
        == loaded_precalibration_identity.get("recovery_scope")
        and metadata_precalibration.get("cross_version_recovery_supported") is False,
        "deployment_completion_marker": completion.get("complete") is True
        and isinstance(completion.get("artifact_bundle_sha256"), str)
        and len(completion["artifact_bundle_sha256"]) == 64,
        "post_ema_all_state_finite": post_ema_validation.get(
            "all_parameter_and_buffer_tensors_finite"
        )
        is True,
        "post_ema_initializer_valid": post_ema_validation.get(
            "all_active_initializers_finite_positive_initialized"
        )
        is True,
        "simultaneous_calibration_final_ema": calibration.get(
            "under_final_ema_parameters"
        )
        is True,
        "simultaneous_calibration_training_buffers_retained_as_initializer": calibration.get(
            "old_training_buffers_discarded"
        )
        is False
        and calibration.get("initializer")
        == CALIBRATION_INITIALIZER_LABEL,
        "simultaneous_calibration_float64_observation_copy": calibration.get(
            "observation_copy_dtype"
        )
        == "float64"
        and calibration.get("observation_copy_before_square") is True
        and calibration.get("accumulator_dtype") == "float64",
        "simultaneous_calibration_snapshot_commit": calibration.get(
            "simultaneous_snapshot_commit"
        )
        is True
        and calibration.get("commit_protocol")
        == "all_candidates_validated_before_sequential_commit_with_exception_rollback"
        and calibration.get("commit_exception_rollback") is True
        and calibration.get("intra_pass_exception_rollback_source")
        == "in_memory_precommit_running_ms_snapshot"
        and calibration.get("process_level_retry_source")
        == "immutable_training_only_precalibration_ema_state",
        "simultaneous_calibration_passes": type(calibration.get("update_passes"))
        is int
        and calibration.get("update_passes") == FINAL_CALIBRATION_PASSES,
        "simultaneous_calibration_audit": calibration.get(
            "final_no_commit_audit_pass"
        )
        is True
        and type(calibration.get("audit_observed_max_relative_change")) is float
        and math.isfinite(calibration.get("audit_observed_max_relative_change"))
        and calibration.get("audit_observed_max_relative_change")
        <= FINAL_CALIBRATION_AUDIT_MAX_RELATIVE_CHANGE,
        "simultaneous_calibration_coverage": calibration.get(
            "all_initialized_and_frozen"
        )
        is True,
        "stored_static": static_audit_record_is_closed(
            stored_static,
            expected_source_closure=manifest["source_closure"],
            direct_qr_required=False,
            direct_qr_call_sites=0,
        ),
        "stored_import_guard": runtime_guard_record_is_closed(
            stored_import_guard,
            expected_entrypoints=EXPECTED_RUNTIME_GUARD_ENTRYPOINTS,
            expected_entrypoint_count=EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT,
        ),
        "stored_final_guard": runtime_guard_record_is_closed(
            stored_final_guard,
            expected_entrypoints=EXPECTED_RUNTIME_GUARD_ENTRYPOINTS,
            expected_entrypoint_count=EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT,
        ),
    }
    failed = sorted(key for key, value in fixed.items() if not value)
    if failed:
        raise RuntimeError(f"Training metadata failed frozen evaluation gates: {failed}")
    observed_checkpoint_sha = artifact_hashes["checkpoint"]
    if observed_checkpoint_sha != metadata.get("checkpoint_sha256"):
        raise RuntimeError("Deployment checkpoint SHA-256 differs from metadata")
    state = torch.load(expected_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise RuntimeError("Deployment checkpoint does not contain a tensor state mapping")
    if canonical_sha256(sorted(state)) != metadata.get("checkpoint_state_keys_sha256"):
        raise RuntimeError("Deployment checkpoint state-key digest differs")
    validate_precalibration_to_deployment_transition(
        loaded_precalibration["training_state"],
        state,
        active_site_names=tuple(initial_running_ms_values),
        claimed_proof=metadata.get("calibration_state_transition", {}),
    )
    del loaded_precalibration
    config = make_product_config(len(vocab), deployment=True)
    if not json_type_exact_equal(
        config_record(config), metadata.get("deployment_config")
    ):
        raise RuntimeError("Deployment config differs from training metadata")
    model = ChiVLA(config)
    strict_load_deployment_state(model, state)
    model.cuda().eval()
    topology = validate_product_model(
        model, deployment=True, require_initialized_norms=True
    )
    if not json_type_exact_equal(topology, metadata.get("deployment_topology")):
        raise RuntimeError("Deployment topology certificate differs from training metadata")
    final_running_ms_values = active_running_ms_values(model)
    calibration_sample_count = metadata.get("cache_structure", {}).get(
        "sample_count"
    )
    if (
        type(calibration_sample_count) is not int
        or calibration_sample_count != CACHE_SAMPLE_COUNT
    ):
        raise RuntimeError("Calibration sample-count authority differs")
    calibration_attestation = validate_simultaneous_calibration_attestation(
        calibration,
        initial_values=initial_running_ms_values,
        final_values=final_running_ms_values,
        sample_count=calibration_sample_count,
    )
    if (
        not json_type_exact_equal(
            calibration_attestation,
            metadata.get("post_ema_simultaneous_rational_calibration_attestation"),
        )
        or not json_type_exact_equal(
            calibration_attestation,
            training_result.get(
                "post_ema_simultaneous_rational_calibration_attestation"
            ),
        )
    ):
        raise RuntimeError("Independent calibration attestation differs")
    return training_result, metadata, model, validated_normalization(metadata), artifact_hashes


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if not _RUNNING_AS_ENTRYPOINT or _PREIMPORT_SOURCE_VERIFICATION is None:
        raise RuntimeError("Product evaluation must use its authenticated entrypoint")
    validate_seed(args.seed)
    smoke = args.mode == "smoke"
    protocol = evaluation_protocol(smoke, args.task_start, args.task_end)
    output = evaluation_result_path(
        args.run_root, args.seed, args.task_start, args.task_end, smoke
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite evaluation output {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen evaluation lane")
    device_name = torch.cuda.get_device_name(0)
    if device_name != "Quadro RTX 6000":
        raise RuntimeError(
            f"Frozen FP32 evaluation device must be Quadro RTX 6000, got {device_name!r}"
        )
    torch.set_float32_matmul_precision(EVALUATION_MATMUL_PRECISION)
    if torch.get_float32_matmul_precision() != EVALUATION_MATMUL_PRECISION:
        raise RuntimeError("Evaluation matmul precision was not applied exactly")
    manifest_start = load_and_verify_source_manifest(args.source_manifest)
    if manifest_start["source_closure"] != _PREIMPORT_SOURCE_VERIFICATION["source_sha256"]:
        raise RuntimeError("Pre-import and post-import source verification differ")
    if source_snapshot() != _TRANSITIVE_STATIC_AUDIT["source_sha256"]:
        raise RuntimeError("Live Python closure differs from the authenticated static audit")
    manifest_sha = file_sha256(args.source_manifest)
    cache_start = assert_cache_identity()
    suite = load_suite()
    official_tasks = task_languages(suite)
    dataset_tasks, task_mapping = validate_task_mapping(official_tasks)
    vocab, encode = build_vocab(dataset_tasks)
    training_result, metadata, model, normalization, artifact_hashes = verify_training_artifact(
        args, manifest_start, official_tasks, vocab
    )
    synthetic_image = torch.zeros(1, 3, RESOLUTION, RESOLUTION, device="cuda")
    synthetic_instruction = torch.tensor(
        [encode(OFFICIAL_TASK_LANGUAGES[0])], dtype=torch.long, device="cuda"
    )
    synthetic_state = torch.zeros(1, STATE_DIM, device="cuda")
    decode_proof = verify_decode_boundary(
        model, synthetic_image, synthetic_instruction, synthetic_state
    )

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    episode_records: list[dict[str, Any]] = []
    task_protocol: dict[str, dict[str, Any]] = {}
    decode_counts = {
        "zero_gate_logits": 0,
        "positive_signs": 0,
        "total_signs": 0,
        "public_component_mismatch": 0,
    }
    policy_chunk_count = 0
    validated_settle_transition_count = 0
    validated_policy_transition_count = 0
    started = time.perf_counter()
    for task_index in range(args.task_start, args.task_end):
        task = suite.get_task(task_index)
        expected_task = EXPECTED_TASK_PROTOCOL[task_index]
        bddl_path = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        init_states = official_init_states(suite, task_index)
        init_state_row_hashes = [
            array_sha256(init_states[index]) for index in range(len(init_states))
        ]
        if canonical_sha256(init_state_row_hashes) != (
            OFFICIAL_INIT_STATE_ROW_HASH_LIST_SHA256[task_index]
        ):
            raise RuntimeError(
                f"Task {task_index} ordered initial-state row digest differs"
            )
        observed_task = {
            "language": str(task.language),
            "problem_folder": str(task.problem_folder),
            "bddl_file": str(task.bddl_file),
            "bddl_sha256": file_sha256(bddl_path),
            "init_state_count": len(init_states),
            "init_states_sha256": array_sha256(init_states),
        }
        if observed_task != expected_task:
            raise RuntimeError(
                f"Task {task_index} packaged protocol differs: {observed_task}"
            )
        task_protocol[str(task_index)] = observed_task
        instruction = torch.tensor(
            [encode(official_tasks[task_index])], dtype=torch.long, device="cuda"
        )
        environment = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=RESOLUTION,
            camera_widths=RESOLUTION,
        )
        try:
            for episode_index in range(protocol["episodes_per_task"]):
                environment.seed(task_index * 100 + episode_index)
                observation = environment.reset()
                observation = environment.set_init_state(init_states[episode_index])
                episode_settle_transitions = 0
                for settle_index in range(protocol["settle_steps"]):
                    observation, _, _, _ = validate_environment_transition(
                        environment.step(dummy_action),
                        phase="settle",
                        task_index=task_index,
                        episode_index=episode_index,
                        transition_index=settle_index,
                    )
                    episode_settle_transitions += 1
                    validated_settle_transition_count += 1
                success = False
                done = False
                steps = 0
                episode_policy_transitions = 0
                episode_policy_chunks = 0
                episode_started = time.perf_counter()
                while steps < protocol["max_steps"] and not success and not done:
                    chunk, counts = predict_chunk(
                        model, observation, instruction, normalization
                    )
                    policy_chunk_count += 1
                    episode_policy_chunks += 1
                    for key, value in counts.items():
                        decode_counts[key] += value
                    for offset in range(protocol["execution_horizon"]):
                        action = chunk[offset].copy()
                        action[-1] = 1.0 if action[-1] > 0 else -1.0
                        observation, reward, done, _ = validate_environment_transition(
                            environment.step(action.tolist()),
                            phase="policy",
                            task_index=task_index,
                            episode_index=episode_index,
                            transition_index=steps,
                        )
                        steps += 1
                        episode_policy_transitions += 1
                        validated_policy_transition_count += 1
                        success = reward > 0.0
                        if success or done or steps >= protocol["max_steps"]:
                            break
                record = {
                    "seed": args.seed,
                    "task_index": task_index,
                    "episode_index": episode_index,
                    "initial_state_sha256": init_state_row_hashes[episode_index],
                    "settle_steps_completed": protocol["settle_steps"],
                    "settle_all_observations_and_rewards_finite": True,
                    "settle_success_or_done_count": 0,
                    "validated_settle_transition_count": episode_settle_transitions,
                    "validated_policy_transition_count": episode_policy_transitions,
                    "success": success,
                    "done": bool(done),
                    "steps": steps,
                    "policy_chunk_count": episode_policy_chunks,
                    "elapsed_s": time.perf_counter() - episode_started,
                }
                episode_records.append(record)
                print("EPISODE", json.dumps(record), flush=True)
        finally:
            environment.close()
    if len(episode_records) != (
        (args.task_end - args.task_start) * protocol["episodes_per_task"]
    ):
        raise RuntimeError("Evaluation emitted the wrong number of episode records")
    episode_keys = [
        (record["seed"], record["task_index"], record["episode_index"])
        for record in episode_records
    ]
    episode_identities = [
        (
            record["seed"],
            record["task_index"],
            record["episode_index"],
            record["initial_state_sha256"],
        )
        for record in episode_records
    ]
    expected_episode_keys = [
        (args.seed, task_index, episode_index)
        for task_index in range(args.task_start, args.task_end)
        for episode_index in range(protocol["episodes_per_task"])
    ]
    if episode_keys != expected_episode_keys:
        raise RuntimeError("Evaluation emitted a noncanonical episode identity order")
    invalid_episode_termination = [
        (record["task_index"], record["episode_index"], record["steps"])
        for record in episode_records
        if not 1 <= record["steps"] <= protocol["max_steps"]
        or (
            record["steps"] < protocol["max_steps"]
            and not (record["success"] or record["done"])
        )
    ]
    if invalid_episode_termination:
        raise RuntimeError(
            "Evaluation episode step or early-stop semantics differ: "
            f"{invalid_episode_termination}"
        )
    success_count = sum(int(record["success"]) for record in episode_records)
    per_task = {
        str(task_index): sum(
            int(record["success"])
            for record in episode_records
            if record["task_index"] == task_index
        )
        / protocol["episodes_per_task"]
        for task_index in range(args.task_start, args.task_end)
    }
    manifest_end = load_and_verify_source_manifest(args.source_manifest)
    end_audit = audit_direct_only_launch(
        PROJECT_ROOT,
        tuple(PROJECT_ROOT / path for path in _EXPECTED_SOURCE_CLOSURE),
        require_direct_qr=False,
    )
    if manifest_end != manifest_start or file_sha256(args.source_manifest) != manifest_sha:
        raise RuntimeError("Authenticated manifest changed during evaluation")
    if end_audit["source_sha256"] != _PREIMPORT_SOURCE_VERIFICATION["source_sha256"]:
        raise RuntimeError("Transitive Python closure changed during evaluation")
    for field in (
        "prohibited_calls_found",
        "prohibited_self_overlap_sites",
        "guarded_dormant_spectral_norm_sites",
        "duplicate_top_level_definition_sites",
    ):
        if end_audit[field] != []:
            raise RuntimeError(f"Final product lane static audit did not close {field}")
    if assert_cache_identity() != cache_start:
        raise RuntimeError("Pinned cache identity changed during evaluation")
    artifact_hashes_end = {
        "training_result": file_sha256(
            training_result_path(args.run_root, args.seed, smoke)
        ),
        "metadata": file_sha256(metadata_path(args.run_root, args.seed, smoke)),
        "checkpoint": file_sha256(checkpoint_path(args.run_root, args.seed, smoke)),
        "preflight": file_sha256(preflight_result_path(args.run_root)),
        "precalibration_ema_state": file_sha256(
            precalibration_state_path(args.run_root, args.seed, smoke)
        ),
        "deployment_completion": file_sha256(
            deployment_completion_path(args.run_root, args.seed, smoke)
        ),
    }
    if smoke:
        artifact_hashes_end["calibration_resume_proof"] = file_sha256(
            calibration_resume_proof_path(args.run_root)
        )
    if artifact_hashes_end != artifact_hashes:
        raise RuntimeError("Training artifact identity changed during evaluation")
    environment = evaluation_environment_record(device_name)
    if not json_type_exact_equal(environment, EXPECTED_EVALUATION_ENVIRONMENT):
        raise RuntimeError(
            f"Capability simulator environment differs from its exact pin: {environment!r}"
        )
    payload = {
        "schema": f"{SCHEMA}_evaluation_result",
        "recipe_version": RECIPE_VERSION,
        "mode": args.mode,
        "seed": args.seed,
        "checkpoint": metadata["checkpoint"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "metadata": metadata_path(args.run_root, args.seed, smoke).as_posix(),
        "metadata_sha256": file_sha256(metadata_path(args.run_root, args.seed, smoke)),
        "training_result": training_result_path(
            args.run_root, args.seed, smoke
        ).as_posix(),
        "training_result_sha256": artifact_hashes["training_result"],
        "precalibration_ema_state": precalibration_state_path(
            args.run_root, args.seed, smoke
        ).as_posix(),
        "precalibration_ema_state_sha256": artifact_hashes[
            "precalibration_ema_state"
        ],
        "deployment_completion": deployment_completion_path(
            args.run_root, args.seed, smoke
        ).as_posix(),
        "deployment_completion_sha256": artifact_hashes[
            "deployment_completion"
        ],
        "calibration_resume_proof_sha256": artifact_hashes.get(
            "calibration_resume_proof"
        ),
        "training_result_closed": training_result.get("schema")
        == f"{SCHEMA}_training_result",
        "source_manifest_sha256": manifest_sha,
        "manifest_bundle_sha256": manifest_start["manifest_bundle_sha256"],
        "protocol": protocol,
        "task_protocol": task_protocol,
        "task_mapping": task_mapping,
        "episode_count": len(episode_records),
        "success_count": success_count,
        "overall": success_count / len(episode_records),
        "per_task": per_task,
        "episodes": episode_records,
        "episode_identity_sha256": canonical_sha256(episode_identities),
        "decode_boundary_proof": decode_proof,
        "decode_counts": decode_counts,
        "policy_chunk_count": policy_chunk_count,
        "validated_settle_transition_count": validated_settle_transition_count,
        "validated_policy_transition_count": validated_policy_transition_count,
        "validated_transition_count": validated_settle_transition_count
        + validated_policy_transition_count,
        "nonfinite_count": 0,
        "decode_failure_count": 0,
        "elapsed_s": time.perf_counter() - started,
        "environment": environment,
        "transitive_direct_only_static_audit": end_audit,
        "capability_simulator_boundary": CAPABILITY_SIMULATOR_BOUNDARY,
        "simulator_external_to_odt": True,
        "odt_runtime_compliance_claimed": False,
        "canonical_odt_runtime_guard_installed": False,
        "canonical_odt_numerical_compliance_claimed": False,
        "external_simulator_outside_weight_only_odt_closure": True,
        "claim_boundary": DEPLOYMENT_CLAIM_BOUNDARY,
    }
    write_json_exclusive(output, payload)
    return payload


def main() -> None:
    args = parse_args()
    payload = run_evaluation(args)
    print("RESULT", json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
