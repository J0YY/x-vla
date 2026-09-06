#!/usr/bin/env python3
"""Validate the paired, isolated MuJoCo package-version diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SUMMARY_SCHEMA_VERSION = "xvla-mujoco-version-control-summary-v2"
FORENSIC_SCHEMA_VERSION = "xvla-mujoco-version-forensics-v3"
SETUP_SCHEMA_VERSION = "xvla-mujoco-version-env-setup-v3"
EXPECTED_BASE_PREFIX = "/work/joy/safesae-openvla"
EXPECTED_OVERLAY_PREFIX = "/work/joy/xvla-mujoco-3.1.6-overlay"
EXPECTED_BASE_FREEZE_SHA256 = (
    "c7e80877c4d26ec8c1db88d68e3d610c0b5174a802192b6f5d195ba4582952a8"
)
EXPECTED_WHEEL_FILENAME = (
    "mujoco-3.1.6-cp310-cp310-manylinux_2_17_x86_64."
    "manylinux2014_x86_64.whl"
)
EXPECTED_WHEEL_SHA256 = (
    "f4c2bbc5025bd0a06e939397188be9dc4a6b7d98aa39176d43a302a5811c0b12"
)
EXPECTED_CACHE_PATH = "artifacts/libero_frames_100000_64.pkl"
EXPECTED_CACHE_SHA256 = (
    "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662"
)
EXPECTED_CHECKPOINT_SHA256 = {
    "artifacts/ckpt_linear_rat_vit_s1.pt": (
        "cc0780b989165a80a449c03cbb3574e2f56b2e5747a64cf32d1fde418df8ec3c"
    ),
    "artifacts/ckpt_linear_rat_vit_s0_v2.pt": (
        "96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9"
    ),
}
EXPECTED_ENVIRONMENT_VARIABLES = {
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "PYTHONNOUSERSITE": "1",
    "LD_LIBRARY_PATH": None,
}
EXPECTED_BASE_PYTHONPATH = "/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO"
EXPECTED_OVERLAY_PYTHONPATH = (
    f"{EXPECTED_OVERLAY_PREFIX}:{EXPECTED_BASE_PYTHONPATH}"
)
COMMON_PROTOCOL = {
    "mode": "capability",
    "architecture": "chi",
    "vision_encoder": "vit",
    "suite": "libero_object",
    "training_suite": "libero_object",
    "cache": EXPECTED_CACHE_PATH,
    "matmul_precision": "highest",
    "eps_per_task": 20,
    "max_steps": 280,
    "num_steps_wait": 10,
    "exec_h": 8,
    "res": 64,
    "horizon": 8,
    "profile_iters": 20,
}
EXPECTED = {
    ("base350", "vit_s1_task5"): {
        "mujoco": "3.5.0",
        "prefix": EXPECTED_BASE_PREFIX,
        "overlay": None,
        "seed": 1,
        "task": 5,
        "checkpoint": "artifacts/ckpt_linear_rat_vit_s1.pt",
    },
    ("base350", "vit_s0_task3"): {
        "mujoco": "3.5.0",
        "prefix": EXPECTED_BASE_PREFIX,
        "overlay": None,
        "seed": 0,
        "task": 3,
        "checkpoint": "artifacts/ckpt_linear_rat_vit_s0_v2.pt",
    },
    ("overlay316", "vit_s1_task5"): {
        "mujoco": "3.1.6",
        "prefix": EXPECTED_BASE_PREFIX,
        "overlay": EXPECTED_OVERLAY_PREFIX,
        "seed": 1,
        "task": 5,
        "checkpoint": "artifacts/ckpt_linear_rat_vit_s1.pt",
    },
    ("overlay316", "vit_s0_task3"): {
        "mujoco": "3.1.6",
        "prefix": EXPECTED_BASE_PREFIX,
        "overlay": EXPECTED_OVERLAY_PREFIX,
        "seed": 0,
        "task": 3,
        "checkpoint": "artifacts/ckpt_linear_rat_vit_s0_v2.pt",
    },
}
EXPECTED_HISTORICAL = {
    "precision_ab_vit_s1_task5_highest": {
        "sha256": "76ce226367eb377a13e63b9059bf90aee8d531e994bc4898f5203ab94c77d693",
        "checkpoint": "artifacts/ckpt_linear_rat_vit_s1.pt",
        "seed": 1,
        "task": 5,
        "successes": 6,
    },
    "precision_ab_vit_s0_task3_highest": {
        "sha256": "d4b084e989149186cfb6d004a7a85c350897bb6395315fa4d092dac5eb250b9d",
        "checkpoint": "artifacts/ckpt_linear_rat_vit_s0_v2.pt",
        "seed": 0,
        "task": 3,
        "successes": 9,
    },
}
MATERIAL_RECOVERY_MIN_SUCCESS_GAIN = 8
MATERIAL_RECOVERY_MAX_ONE_SIDED_MCNEMAR_P = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-setup", type=Path, required=True)
    parser.add_argument("--result", action="append", type=Path, required=True)
    parser.add_argument("--historical", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def directory_identity(root: Path) -> dict[str, Any]:
    manifest = {}
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.suffix in (".pyc", ".pyo")
            or "__pycache__" in path.parts
        ):
            continue
        manifest[str(path.relative_to(root))] = sha256_file(path)
    material = "\n".join(
        f"{relative}\0{value}" for relative, value in manifest.items()
    )
    return {
        "file_count": len(manifest),
        "tree_sha256": sha256_bytes(material.encode("utf-8")),
        "file_sha256": manifest,
    }


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def normalized_freeze_sha256(path: Path, *, without_mujoco: bool = False) -> str:
    lines = sorted(line.strip() for line in path.read_text().splitlines() if line.strip())
    if without_mujoco:
        lines = [
            line
            for line in lines
            if not line.lower().startswith(("mujoco==", "mujoco @"))
        ]
    return sha256_bytes("\n".join(lines).encode("utf-8"))


def hash_array(value: Any, dtype: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def path_is_within(path: str, prefix: str) -> bool:
    resolved = Path(path).resolve()
    expected = Path(prefix).resolve()
    return resolved == expected or expected in resolved.parents


def result_key(path: Path) -> tuple[str, str]:
    for version in ("base350", "overlay316"):
        prefix = f"{version}_"
        if path.stem.startswith(prefix):
            return version, path.stem[len(prefix) :]
    raise RuntimeError(f"Unexpected result name: {path}")


def validate_setup(path: Path) -> dict[str, Any]:
    setup = load(path)
    result_dir = path.parent
    expected_files = {
        "base_pip_freeze.txt": setup.get("base_pip_freeze_sha256"),
        "base_pip_freeze_after_overlay.txt": setup.get(
            "base_pip_freeze_after_overlay_sha256"
        ),
        "overlay_pip_freeze.txt": setup.get("overlay_pip_freeze_sha256"),
        "base_without_mujoco.txt": setup.get("base_without_mujoco_sha256"),
        "overlay_without_mujoco.txt": setup.get("overlay_without_mujoco_sha256"),
        "base_pip_check.txt": setup.get("base_pip_check_sha256"),
        "overlay_pip_check.txt": setup.get("overlay_pip_check_sha256"),
    }
    failed = []
    for filename, expected_sha256 in expected_files.items():
        candidate = result_dir / filename
        if not candidate.is_file() or sha256_file(candidate) != expected_sha256:
            failed.append(f"artifact:{filename}")

    wheel = result_dir / "wheels" / EXPECTED_WHEEL_FILENAME
    if not wheel.is_file() or sha256_file(wheel) != EXPECTED_WHEEL_SHA256:
        failed.append("frozen_wheel")
    for name in ("base_runtime_identity", "overlay_runtime_identity"):
        candidate = result_dir / f"{name.removesuffix('_identity')}_identity.json"
        if not candidate.is_file() or load(candidate) != setup.get(name):
            failed.append(name)

    overlay = Path(EXPECTED_OVERLAY_PREFIX)
    if not overlay.is_dir() or directory_identity(overlay) != setup.get(
        "overlay_directory_identity"
    ):
        failed.append("overlay_directory_identity")

    checks = {
        "schema": setup.get("schema_version") == SETUP_SCHEMA_VERSION,
        "setup_source": setup.get("setup_source_sha256")
        == sha256_file(Path("athena/slurm_setup_mujoco_version_control.sbatch")),
        "base_prefix": setup.get("base_prefix") == EXPECTED_BASE_PREFIX,
        "overlay_prefix": setup.get("overlay_prefix") == EXPECTED_OVERLAY_PREFIX,
        "base_freeze": setup.get("base_pip_freeze_sha256")
        == EXPECTED_BASE_FREEZE_SHA256,
        "base_freeze_unchanged": setup.get("base_pip_freeze_after_overlay_sha256")
        == EXPECTED_BASE_FREEZE_SHA256,
        "expected_base_freeze": setup.get("expected_base_pip_freeze_sha256")
        == EXPECTED_BASE_FREEZE_SHA256,
        "only_freeze_change": setup.get("only_pip_freeze_change_is_mujoco") is True,
        "without_mujoco_equal": setup.get("base_without_mujoco_sha256")
        == setup.get("overlay_without_mujoco_sha256"),
        "versions": setup.get("base_mujoco_version") == "3.5.0"
        and setup.get("overlay_mujoco_version") == "3.1.6",
        "pip_check_equal": setup.get("pip_check_status_and_output_equal") is True
        and setup.get("base_pip_check_status") == setup.get("overlay_pip_check_status")
        and setup.get("base_pip_check_sha256") == setup.get("overlay_pip_check_sha256"),
        "wheel_name": setup.get("mujoco_3_1_6_wheel_filename")
        == EXPECTED_WHEEL_FILENAME,
        "expected_wheel_sha256": setup.get("expected_mujoco_3_1_6_wheel_sha256")
        == EXPECTED_WHEEL_SHA256,
        "wheel_sha256": setup.get("mujoco_3_1_6_wheel_sha256")
        == EXPECTED_WHEEL_SHA256,
        "base_not_mutated": setup.get("base_environment_mutated") is False,
        "isolation_mechanism": setup.get("isolation_mechanism")
        == "base_python_with_package_only_overlay_first_on_pythonpath",
    }
    failed.extend(name for name, passed in checks.items() if not passed)
    base_runtime = setup.get("base_runtime_identity", {})
    overlay_runtime = setup.get("overlay_runtime_identity", {})
    for label, runtime, overlay_prefix, module_prefix, version in (
        ("base", base_runtime, None, EXPECTED_BASE_PREFIX, "3.5.0"),
        (
            "overlay",
            overlay_runtime,
            EXPECTED_OVERLAY_PREFIX,
            EXPECTED_OVERLAY_PREFIX,
            "3.1.6",
        ),
    ):
        distribution = runtime.get("mujoco_distribution", {})
        runtime_checks = {
            "base_prefix": runtime.get("base_prefix") == EXPECTED_BASE_PREFIX,
            "overlay_prefix": runtime.get("overlay_prefix") == overlay_prefix,
            "python_path": path_is_within(
                runtime.get("python_executable", "/"), EXPECTED_BASE_PREFIX
            ),
            "module_path": path_is_within(
                runtime.get("mujoco_module", "/"), module_prefix
            ),
            "runtime_version": runtime.get("mujoco_runtime_version") == version,
            "metadata_version": distribution.get("version") == version,
            "record": is_sha256(distribution.get("record_sha256")),
            "tree": is_sha256(distribution.get("distribution_tree_sha256")),
            "native": bool(distribution.get("native_library_sha256"))
            and all(
                path_is_within(native_path, module_prefix) and is_sha256(native_sha256)
                for native_path, native_sha256 in distribution.get(
                    "native_library_sha256", {}
                ).items()
            ),
        }
        failed.extend(
            f"{label}_runtime:{name}"
            for name, passed in runtime_checks.items()
            if not passed
        )
    if base_runtime.get("python_executable_sha256") != overlay_runtime.get(
        "python_executable_sha256"
    ):
        failed.append("python_executable_bytes")
    if base_runtime.get("python_version") != overlay_runtime.get("python_version"):
        failed.append("python_version")
    if failed:
        raise RuntimeError(f"Environment-setup integrity failure: {failed}")

    base_freeze = result_dir / "base_pip_freeze.txt"
    overlay_freeze = result_dir / "overlay_pip_freeze.txt"
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "setup": setup,
        "normalized_base_freeze_sha256": normalized_freeze_sha256(base_freeze),
        "normalized_overlay_freeze_sha256": normalized_freeze_sha256(overlay_freeze),
        "normalized_without_mujoco_sha256": normalized_freeze_sha256(
            base_freeze, without_mujoco=True
        ),
    }


def recompute_environment_identity_sha256(environment: dict[str, Any]) -> str:
    material = {
        key: environment[key]
        for key in (
            "python_executable_sha256",
            "python_version",
            "mujoco_overlay",
            "pip_freeze_sha256",
            "mujoco_install_record_sha256",
            "mujoco_distribution_tree_sha256",
            "mujoco_native_library_sha256",
            "mujoco_loaded_native_library_sha256",
            "forensic_runner_sha256",
            "evaluator_sha256",
            "torch_flags",
        )
    }
    return sha256_bytes(json.dumps(material, sort_keys=True).encode("utf-8"))


def validate_trajectory(
    path: Path,
    capability: dict[str, Any],
    forensic_episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    outcomes = capability["episodes"]
    if len(outcomes) != 20 or len(forensic_episodes) != 20:
        raise RuntimeError(f"Unexpected episode count in {path}")
    successes = 0
    total_queries = 0
    total_actions = 0
    for outcome, episode in zip(outcomes, forensic_episodes, strict=True):
        identity = (outcome["task_index"], outcome["episode"])
        forensic_identity = (episode["task_index"], episode["episode"])
        if identity != forensic_identity:
            raise RuntimeError(f"Outcome/forensic identity mismatch in {path}")
        if episode["success"] is not outcome["success"] or episode["steps"] != outcome["steps"]:
            raise RuntimeError(f"Outcome/forensic value mismatch in {path}")
        if not 0 < outcome["steps"] <= COMMON_PROTOCOL["max_steps"]:
            raise RuntimeError(f"Invalid rollout length in {path}")
        if not outcome["success"] and outcome["steps"] != COMMON_PROTOCOL["max_steps"]:
            raise RuntimeError(f"Failed rollout stopped before the frozen horizon in {path}")

        actions = episode["policy_actions"]
        transitions = episode["policy_transition_flags"]
        queries = episode["policy_queries"]
        if len(actions) != outcome["steps"] or len(transitions) != len(actions):
            raise RuntimeError(f"Incomplete action or transition trace in {path}")
        if len(queries) != math.ceil(len(actions) / COMMON_PROTOCOL["exec_h"]):
            raise RuntimeError(f"Unexpected policy-query schedule in {path}")

        for action in actions:
            values = action["values"]
            if len(values) != 7 or values[-1] not in (-1.0, 1.0):
                raise RuntimeError(f"Invalid executed action in {path}")
            if not all(math.isfinite(float(value)) for value in values):
                raise RuntimeError(f"Non-finite executed action in {path}")
            if action.get("sha256") != hash_array(values, np.float32):
                raise RuntimeError(f"Executed-action digest mismatch in {path}")

        reward_success = False
        for transition in transitions:
            if not isinstance(transition.get("done"), bool):
                raise RuntimeError(f"Invalid transition termination flag in {path}")
            reward = float(transition["reward"])
            if not math.isfinite(reward):
                raise RuntimeError(f"Non-finite transition reward in {path}")
            reward_success = reward_success or reward > 0
            for name in ("post_step_sim_state_sha256", "post_step_raw_image_sha256"):
                if not is_sha256(transition.get(name)):
                    raise RuntimeError(f"Invalid transition digest in {path}")
        if reward_success is not outcome["success"]:
            raise RuntimeError(f"Reward-derived outcome mismatch in {path}")

        expected_start = 0
        for query_index, query in enumerate(queries):
            if query.get("query_index") != query_index:
                raise RuntimeError(f"Policy-query index mismatch in {path}")
            if query.get("executed_action_start") != expected_start:
                raise RuntimeError(f"Non-contiguous policy-query schedule in {path}")
            executed_count = query.get("executed_action_count")
            if not isinstance(executed_count, int) or not 1 <= executed_count <= 8:
                raise RuntimeError(f"Invalid per-query action count in {path}")
            if query_index + 1 < len(queries) and executed_count != 8:
                raise RuntimeError(f"Short non-final action chunk in {path}")
            chunk = query["raw_action_chunk"]
            if len(chunk) != 8 or any(len(action) != 7 for action in chunk):
                raise RuntimeError(f"Invalid raw action chunk in {path}")
            if not all(
                math.isfinite(float(value)) for action in chunk for value in action
            ):
                raise RuntimeError(f"Non-finite raw action chunk in {path}")
            if query.get("raw_action_chunk_sha256") != hash_array(chunk, np.float32):
                raise RuntimeError(f"Raw-action-chunk digest mismatch in {path}")
            if query.get("instruction_token_sha256") != hash_array(
                query["instruction_token_ids"], np.int64
            ):
                raise RuntimeError(f"Instruction-token digest mismatch in {path}")
            for name in (
                "raw_image_sha256",
                "model_image_sha256",
                "robot_state_sha256",
                "sim_state_sha256",
            ):
                if not is_sha256(query.get(name)):
                    raise RuntimeError(f"Invalid query-input digest in {path}")

            for offset in range(executed_count):
                executed = np.asarray(actions[expected_start + offset]["values"], dtype=np.float32)
                raw = np.asarray(chunk[offset], dtype=np.float32)
                if not np.array_equal(executed[:6], raw[:6]):
                    raise RuntimeError(f"Chunk/action value mismatch in {path}")
                expected_gripper = np.float32(1.0 if raw[-1] > 0 else -1.0)
                if executed[-1] != expected_gripper:
                    raise RuntimeError(f"Chunk/action gripper mismatch in {path}")
            expected_start += executed_count
        if expected_start != len(actions):
            raise RuntimeError(f"Policy-query coverage mismatch in {path}")

        successes += int(outcome["success"])
        total_queries += len(queries)
        total_actions += len(actions)
    if successes != capability["successes"]:
        raise RuntimeError(f"Capability success total is not reproducible in {path}")
    if capability["overall"] != successes / capability["trials"]:
        raise RuntimeError(f"Capability success rate is not reproducible in {path}")
    return {
        "successes_recomputed": successes,
        "queries_recomputed": total_queries,
        "actions_recomputed": total_actions,
        "chunk_to_action_mapping_valid": True,
        "outcomes_recomputed_from_rewards": True,
    }


def validate_result(
    path: Path,
    result: dict[str, Any],
    setup_validation: dict[str, Any],
) -> dict[str, Any]:
    key = result_key(path)
    expected = EXPECTED[key]
    capability = result["capability"]
    forensics = result["version_isolation_forensics"]
    environment = forensics["environment"]
    frozen_inputs = forensics["frozen_inputs"]
    protocol = forensics["protocol"]
    expected_protocol = {
        **COMMON_PROTOCOL,
        "checkpoint": expected["checkpoint"],
        "seed": expected["seed"],
        "task_start": expected["task"],
        "task_end": expected["task"] + 1,
    }
    setup_runtime_key = (
        "base_runtime_identity" if key[0] == "base350" else "overlay_runtime_identity"
    )
    setup_runtime = setup_validation["setup"][setup_runtime_key]
    setup_distribution = setup_runtime["mujoco_distribution"]
    expected_normalized_freeze = setup_validation[
        "normalized_base_freeze_sha256"
        if key[0] == "base350"
        else "normalized_overlay_freeze_sha256"
    ]
    expected_environment_variables = {
        **EXPECTED_ENVIRONMENT_VARIABLES,
        "PYTHONPATH": (
            EXPECTED_BASE_PYTHONPATH
            if expected["overlay"] is None
            else EXPECTED_OVERLAY_PYTHONPATH
        ),
        "XVLA_MUJOCO_OVERLAY": expected["overlay"],
    }
    checks = {
        "forensic_schema": forensics.get("schema_version") == FORENSIC_SCHEMA_VERSION,
        "protocol": protocol == expected_protocol,
        "mode": result.get("mode") == "capability",
        "architecture": result.get("architecture") == "chi",
        "vision_encoder": result.get("vision_encoder") == "vit",
        "suite": result.get("suite") == "libero_object",
        "training_suite": result.get("training_suite") == "libero_object",
        "checkpoint": result.get("checkpoint") == expected["checkpoint"],
        "cache": result.get("cache") == EXPECTED_CACHE_PATH,
        "seed": result.get("seed") == expected["seed"],
        "precision": result.get("matmul_precision") == "highest",
        "tasks": capability.get("task_indices") == [expected["task"]],
        "trials": capability.get("trials") == 20,
        "eps_per_task": capability.get("eps_per_task") == 20,
        "max_steps": capability.get("max_steps") == 280,
        "wait_steps": capability.get("num_steps_wait") == 10,
        "exec_h": capability.get("exec_h") == 8,
        "episodes": [row.get("episode") for row in capability.get("episodes", [])]
        == list(range(20)),
        "canonical": capability.get("canonical_init_states") is True,
        "finite": result.get("prediction_finite") is True,
        "environment_schema": environment.get("schema_version")
        == FORENSIC_SCHEMA_VERSION,
        "conda_prefix": environment.get("conda_prefix") == expected["prefix"],
        "mujoco_overlay": environment.get("mujoco_overlay") == expected["overlay"],
        "python_path": path_is_within(
            environment.get("python_executable", "/"), expected["prefix"]
        ),
        "python_bytes": environment.get("python_executable_sha256")
        == setup_runtime.get("python_executable_sha256"),
        "python_version": environment.get("python_version")
        == setup_runtime.get("python_version"),
        "mujoco_metadata_version": environment.get("critical_packages", {}).get(
            "mujoco"
        )
        == expected["mujoco"],
        "mujoco_runtime_version": environment.get("mujoco_runtime_version")
        == expected["mujoco"],
        "mujoco_module": environment.get("mujoco_module")
        == setup_runtime.get("mujoco_module"),
        "mujoco_record": environment.get("mujoco_install_record_sha256")
        == setup_distribution.get("record_sha256"),
        "mujoco_tree": environment.get("mujoco_distribution_tree_sha256")
        == setup_distribution.get("distribution_tree_sha256"),
        "mujoco_native": environment.get("mujoco_native_library_sha256")
        == setup_distribution.get("native_library_sha256"),
        "mujoco_loaded_native": bool(
            environment.get("mujoco_loaded_native_library_sha256")
        )
        and all(
            path_is_within(
                native_path,
                EXPECTED_BASE_PREFIX
                if expected["overlay"] is None
                else EXPECTED_OVERLAY_PREFIX,
            )
            and setup_distribution.get("native_library_sha256", {}).get(native_path)
            == native_sha256
            for native_path, native_sha256 in environment.get(
                "mujoco_loaded_native_library_sha256", {}
            ).items()
        ),
        "pip_freeze": environment.get("pip_freeze_sha256")
        == expected_normalized_freeze,
        "pip_freeze_without_mujoco": environment.get(
            "pip_freeze_without_mujoco_sha256"
        )
        == setup_validation["normalized_without_mujoco_sha256"],
        "environment_variables": all(
            environment.get("environment_variables", {}).get(name) == value
            for name, value in expected_environment_variables.items()
        ),
        "environment_identity": environment.get("environment_identity_sha256")
        == recompute_environment_identity_sha256(environment),
        "forensic_count": len(forensics.get("episodes", [])) == 20,
        "cache_sha256": frozen_inputs.get("cache_sha256") == EXPECTED_CACHE_SHA256,
        "checkpoint_sha256": frozen_inputs.get("checkpoint_sha256")
        == EXPECTED_CHECKPOINT_SHA256[expected["checkpoint"]],
        "records_actions": forensics.get("records_all_executed_policy_actions") is True,
        "records_chunks": forensics.get("records_all_raw_action_chunks") is True,
        "source_stable": frozen_inputs.get("source_snapshot_stable") is True
        and frozen_inputs.get("source_sha256_start")
        == frozen_inputs.get("source_sha256_end")
        == frozen_inputs.get("source_sha256"),
        "runner_source": environment.get("forensic_runner_sha256")
        == frozen_inputs.get("source_sha256", {}).get(
            "athena/run_mujoco_version_forensics.py"
        ),
        "evaluator_source": environment.get("evaluator_sha256")
        == frozen_inputs.get("source_sha256", {}).get(
            "athena/run_xvla_experiment.py"
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Identity failure in {path}: {failed}")
    for episode in forensics["episodes"]:
        for name in (
            "canonical_init_state_sha256",
            "post_set_sim_state_sha256",
            "post_wait_sim_state_sha256",
            "initial_raw_image_sha256",
            "initial_model_image_sha256",
            "initial_robot_state_sha256",
        ):
            if not is_sha256(episode.get(name)):
                raise RuntimeError(f"Invalid episode-state digest in {path}")
    trajectory_validation = validate_trajectory(path, capability, forensics["episodes"])
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "successes": capability["successes"],
        "trials": capability["trials"],
        "environment_identity_sha256": environment["environment_identity_sha256"],
        "pip_freeze_sha256": environment["pip_freeze_sha256"],
        "pip_freeze_without_mujoco_sha256": environment[
            "pip_freeze_without_mujoco_sha256"
        ],
        "mujoco_install_record_sha256": environment[
            "mujoco_install_record_sha256"
        ],
        "mujoco_distribution_tree_sha256": environment[
            "mujoco_distribution_tree_sha256"
        ],
        "source_sha256": frozen_inputs["source_sha256"],
        "torch_flags": environment["torch_flags"],
        "environment_variables": environment["environment_variables"],
        "trajectory_validation": trajectory_validation,
    }


def paired_table(base: dict[str, Any], clone: dict[str, Any]) -> dict[str, int]:
    base_outcomes = base["capability"]["episodes"]
    clone_outcomes = clone["capability"]["episodes"]
    table = {
        "both_fail": 0,
        "base_only_success_regression": 0,
        "overlay_only_success_improvement": 0,
        "both_success": 0,
    }
    for base_row, clone_row in zip(base_outcomes, clone_outcomes, strict=True):
        if (base_row["task_index"], base_row["episode"]) != (
            clone_row["task_index"],
            clone_row["episode"],
        ):
            raise RuntimeError("Paired outcome identities differ across MuJoCo versions")
        if base_row["success"] and clone_row["success"]:
            table["both_success"] += 1
        elif base_row["success"]:
            table["base_only_success_regression"] += 1
        elif clone_row["success"]:
            table["overlay_only_success_improvement"] += 1
        else:
            table["both_fail"] += 1
    return table


def exact_one_sided_mcnemar_p(table: dict[str, int]) -> float:
    improvements = table["overlay_only_success_improvement"]
    regressions = table["base_only_success_regression"]
    discordant = improvements + regressions
    if discordant == 0:
        return 1.0
    numerator = sum(math.comb(discordant, k) for k in range(improvements, discordant + 1))
    return numerator / (2**discordant)


def add_tables(tables: list[dict[str, int]]) -> dict[str, int]:
    keys = (
        "both_fail",
        "base_only_success_regression",
        "overlay_only_success_improvement",
        "both_success",
    )
    return {key: sum(table[key] for table in tables) for key in keys}


def paired_forensics(base: dict[str, Any], clone: dict[str, Any]) -> dict[str, Any]:
    base_rows = base["version_isolation_forensics"]["episodes"]
    clone_rows = clone["version_isolation_forensics"]["episodes"]
    if [(row["task_index"], row["episode"]) for row in base_rows] != [
        (row["task_index"], row["episode"]) for row in clone_rows
    ]:
        raise RuntimeError("Paired episode identities differ across MuJoCo versions")
    fields = (
        "canonical_init_state_sha256",
        "post_set_sim_state_sha256",
        "post_wait_sim_state_sha256",
        "initial_raw_image_sha256",
        "initial_model_image_sha256",
        "initial_robot_state_sha256",
    )
    agreements = {
        field: sum(a[field] == b[field] for a, b in zip(base_rows, clone_rows, strict=True))
        for field in fields
    }
    if agreements["canonical_init_state_sha256"] != len(base_rows):
        raise RuntimeError("Canonical initial-state inputs differ across versions")
    exact_action_episodes = sum(
        a["policy_actions"] == b["policy_actions"]
        for a, b in zip(base_rows, clone_rows, strict=True)
    )
    first_action_equal = sum(
        bool(a["policy_actions"])
        and bool(b["policy_actions"])
        and a["policy_actions"][0] == b["policy_actions"][0]
        for a, b in zip(base_rows, clone_rows, strict=True)
    )
    mismatches = []
    for base_row, clone_row in zip(base_rows, clone_rows, strict=True):
        base_queries: dict[tuple[str, str, str], set[str]] = {}
        for query in base_row["policy_queries"]:
            key = (
                query["model_image_sha256"],
                query["robot_state_sha256"],
                query["instruction_token_sha256"],
            )
            base_queries.setdefault(key, set()).add(query["raw_action_chunk_sha256"])
        for query in clone_row["policy_queries"]:
            key = (
                query["model_image_sha256"],
                query["robot_state_sha256"],
                query["instruction_token_sha256"],
            )
            if key in base_queries and query["raw_action_chunk_sha256"] not in base_queries[key]:
                mismatches.append(
                    {"task_index": base_row["task_index"], "episode": base_row["episode"]}
                )
    if mismatches:
        raise RuntimeError("Identical policy inputs produced different action chunks")
    return {
        "episode_count": len(base_rows),
        "hash_agreements": agreements,
        "exact_full_action_trace_agreements": exact_action_episodes,
        "exact_first_action_agreements": first_action_equal,
        "identical_input_action_chunk_mismatches": mismatches,
    }


def validate_historical(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    expected = EXPECTED_HISTORICAL.get(path.stem)
    if expected is None:
        raise RuntimeError(f"Unexpected historical result name: {path}")
    capability = result.get("capability", {})
    checks = {
        "sha256": sha256_file(path) == expected["sha256"],
        "mode": result.get("mode") == "capability",
        "architecture": result.get("architecture") == "chi",
        "vision_encoder": result.get("vision_encoder") == "vit",
        "suite": result.get("suite") == "libero_object",
        "training_suite": result.get("training_suite") == "libero_object",
        "checkpoint": result.get("checkpoint") == expected["checkpoint"],
        "cache": result.get("cache") == EXPECTED_CACHE_PATH,
        "seed": result.get("seed") == expected["seed"],
        "precision": result.get("matmul_precision") == "highest",
        "finite": result.get("prediction_finite") is True,
        "task": capability.get("task_indices") == [expected["task"]],
        "successes": capability.get("successes") == expected["successes"],
        "trials": capability.get("trials") == 20,
        "episodes": [row.get("episode") for row in capability.get("episodes", [])]
        == list(range(20)),
        "success_count": sum(
            int(row.get("success", False)) for row in capability.get("episodes", [])
        )
        == expected["successes"],
        "eps_per_task": capability.get("eps_per_task") == 20,
        "max_steps": capability.get("max_steps") == 280,
        "wait_steps": capability.get("num_steps_wait") == 10,
        "exec_h": capability.get("exec_h") == 8,
        "canonical": capability.get("canonical_init_states") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Historical result integrity failure in {path}: {failed}")
    return {
        "path": str(path),
        "sha256": expected["sha256"],
        "seed": expected["seed"],
        "task": expected["task"],
        "successes": expected["successes"],
        "trials": 20,
    }


def main() -> None:
    args = parse_args()
    if len(args.result) != 4 or len(args.historical) != 2:
        raise ValueError("Expected four forensic results and two historical controls")
    summary_source_start = sha256_file(Path(__file__))
    input_paths = [args.environment_setup, *args.result, *args.historical]
    input_sha256_start = {str(path): sha256_file(path) for path in input_paths}

    setup_validation = validate_setup(args.environment_setup)
    results = {result_key(path): load(path) for path in args.result}
    if set(results) != set(EXPECTED):
        raise RuntimeError("Result set does not match the frozen four-condition design")
    validated = {
        "/".join(key): validate_result(path, results[key], setup_validation)
        for key, path in ((result_key(path), path) for path in args.result)
    }

    source_manifests = {
        json.dumps(
            result["version_isolation_forensics"]["frozen_inputs"]["source_sha256"],
            sort_keys=True,
        )
        for result in results.values()
    }
    if len(source_manifests) != 1:
        raise RuntimeError("Source hashes differ across version-isolation arms")
    torch_flags = {
        json.dumps(
            result["version_isolation_forensics"]["environment"]["torch_flags"],
            sort_keys=True,
        )
        for result in results.values()
    }
    if len(torch_flags) != 1:
        raise RuntimeError("Torch, GPU, CUDA, or numerical flags differ across arms")
    environment_controls = {
        json.dumps(
            {
                name: value
                for name, value in result["version_isolation_forensics"]["environment"][
                    "environment_variables"
                ].items()
                if name not in ("PYTHONPATH", "XVLA_MUJOCO_OVERLAY")
            },
            sort_keys=True,
        )
        for result in results.values()
    }
    if len(environment_controls) != 1:
        raise RuntimeError("Non-treatment runtime environment variables differ across arms")
    python_hashes = {
        result["version_isolation_forensics"]["environment"][
            "python_executable_sha256"
        ]
        for result in results.values()
    }
    if len(python_hashes) != 1:
        raise RuntimeError("Python executable bytes differ across overlay arms")

    slices: dict[str, Any] = {}
    base_total = 0
    clone_total = 0
    no_slice_decrease = True
    tables = []
    for task_label in ("vit_s1_task5", "vit_s0_task3"):
        base = results[("base350", task_label)]
        clone = results[("overlay316", task_label)]
        base_successes = base["capability"]["successes"]
        clone_successes = clone["capability"]["successes"]
        base_total += base_successes
        clone_total += clone_successes
        no_slice_decrease = no_slice_decrease and clone_successes >= base_successes
        table = paired_table(base, clone)
        tables.append(table)
        slices[task_label] = {
            "base350_successes": base_successes,
            "overlay316_successes": clone_successes,
            "success_gain": clone_successes - base_successes,
            "paired_outcome_table": table,
            "one_sided_exact_mcnemar_p": exact_one_sided_mcnemar_p(table),
            "paired_forensics": paired_forensics(base, clone),
        }

    historical = {
        path.stem: validate_historical(path, load(path)) for path in args.historical
    }
    if set(historical) != set(EXPECTED_HISTORICAL):
        raise RuntimeError("Historical result set does not match the frozen controls")

    aggregate_table = add_tables(tables)
    aggregate_p = exact_one_sided_mcnemar_p(aggregate_table)
    gain = clone_total - base_total
    effect_size_pass = gain >= MATERIAL_RECOVERY_MIN_SUCCESS_GAIN
    paired_inference_pass = aggregate_p <= MATERIAL_RECOVERY_MAX_ONE_SIDED_MCNEMAR_P
    material_recovery_pass = effect_size_pass and no_slice_decrease and paired_inference_pass

    setup_validation_end = validate_setup(args.environment_setup)
    summary_source_end = sha256_file(Path(__file__))
    input_sha256_end = {str(path): sha256_file(path) for path in input_paths}
    if (
        setup_validation != setup_validation_end
        or summary_source_start != summary_source_end
        or input_sha256_start != input_sha256_end
    ):
        raise RuntimeError("Summary source or inputs changed during validation")

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "diagnostic_complete": True,
        "integrity_validation_pass": True,
        "material_recovery_pass": material_recovery_pass,
        "summary_source_sha256_start": summary_source_start,
        "summary_source_sha256_end": summary_source_end,
        "input_sha256_start": input_sha256_start,
        "input_sha256_end": input_sha256_end,
        "environment_setup": {
            "path": setup_validation["path"],
            "sha256": setup_validation["sha256"],
            "exact_frozen_wheel": EXPECTED_WHEEL_FILENAME,
            "exact_frozen_wheel_sha256": EXPECTED_WHEEL_SHA256,
            "overlay_prefix": EXPECTED_OVERLAY_PREFIX,
            "base_environment_mutated": False,
            "isolation_mechanism": (
                "unchanged_base_python_with_package_only_overlay_first_on_pythonpath"
            ),
            "software_environment_isolated_to_mujoco_distribution": True,
        },
        "validated_results": validated,
        "validated_historical_controls": historical,
        "slices": slices,
        "aggregate": {
            "base350_successes": base_total,
            "overlay316_successes": clone_total,
            "trials_per_version": 40,
            "success_gain": gain,
            "success_rate_gain": gain / 40,
            "paired_outcome_table": aggregate_table,
            "one_sided_exact_mcnemar_p": aggregate_p,
        },
        "material_recovery_gate": {
            "minimum_success_gain": MATERIAL_RECOVERY_MIN_SUCCESS_GAIN,
            "minimum_success_rate_gain": MATERIAL_RECOVERY_MIN_SUCCESS_GAIN / 40,
            "requires_no_slice_decrease": True,
            "maximum_one_sided_exact_mcnemar_p": (
                MATERIAL_RECOVERY_MAX_ONE_SIDED_MCNEMAR_P
            ),
            "effect_size_pass": effect_size_pass,
            "no_slice_decrease_pass": no_slice_decrease,
            "paired_inference_pass": paired_inference_pass,
            "pass": material_recovery_pass,
        },
        "treatment_scope": (
            "The isolated software treatment is the complete MuJoCo Python wheel and "
            "its bundled native distribution. It jointly changes renderer and physics "
            "behavior and cannot support a solver-only attribution."
        ),
        "order_drift_control": {
            "run": False,
            "execution_order": [
                "base350/vit_s1_task5",
                "base350/vit_s0_task3",
                "overlay316/vit_s1_task5",
                "overlay316/vit_s0_task3",
            ],
            "claim_limit": (
                "The paired diagnostic uses one fixed base-then-overlay order. Same-allocation "
                "identity checks reduce but do not eliminate order drift. A counterbalanced "
                "or post-overlay base placebo is required before a definitive causal claim."
            ),
        },
        "panel_scope": (
            "This is a preregistered two-slice failure-panel diagnostic. A passing gate "
            "justifies a full locked replay but is not a suite-wide recovery estimate."
        ),
        "full_locked_replay_recommendation": (
            "queue_mujoco_3_1_6_full_locked_replay"
            if material_recovery_pass
            else "do_not_queue_from_this_control"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
