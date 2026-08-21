#!/usr/bin/env python3
"""Add environment and trajectory forensics to the frozen Athena evaluator."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import athena.run_xvla_experiment as evaluator


SCHEMA_VERSION = "xvla-mujoco-version-forensics-v1"
EPISODE_FORENSICS: list[dict[str, Any]] = []
ACTIVE_FORENSIC_RECORD: dict[str, Any] | None = None
NUM_WAIT_STEPS = 10
RESOLUTION = 64
SOURCE_FILES = (
    "athena/run_mujoco_version_forensics.py",
    "athena/run_xvla_experiment.py",
    "athena/libero_dataset_metadata.py",
    "xvla/models/vla.py",
    "xvla/models/vit.py",
    "xvla/nn/attention.py",
    "xvla/nn/bilinear.py",
    "xvla/nn/block.py",
    "xvla/nn/normalization.py",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def model_image(obs: dict[str, Any]) -> np.ndarray:
    rotated = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    return np.asarray(Image.fromarray(rotated).resize((RESOLUTION, RESOLUTION))).copy()


def pip_freeze() -> list[str]:
    output = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True
    )
    return sorted(line.strip() for line in output.splitlines() if line.strip())


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def install_record_sha256(name: str) -> str | None:
    distribution = importlib.metadata.distribution(name)
    for relative in distribution.files or []:
        if str(relative).endswith(".dist-info/RECORD"):
            path = Path(distribution.locate_file(relative))
            return sha256_file(path)
    return None


def patch_environment() -> None:
    global ACTIVE_FORENSIC_RECORD

    import libero.libero.envs as libero_envs

    original = libero_envs.OffScreenRenderEnv
    original_predict_chunk = evaluator.predict_chunk

    class RecordingOffScreenRenderEnv(original):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._forensic_record: dict[str, Any] | None = None
            self._steps_after_init = 0

        def set_init_state(self, init_state: Any):
            global ACTIVE_FORENSIC_RECORD

            obs = super().set_init_state(init_state)
            self._steps_after_init = 0
            self._forensic_record = {
                "canonical_init_state_sha256": hash_array(init_state),
                "post_set_sim_state_sha256": hash_array(self.sim.get_state().flatten()),
                "mujoco_model_options": {
                    name: float(getattr(self.sim.model.opt, name))
                    if name in ("timestep", "tolerance", "ls_tolerance", "noslip_tolerance")
                    else int(getattr(self.sim.model.opt, name))
                    for name in (
                        "timestep",
                        "integrator",
                        "solver",
                        "iterations",
                        "ls_iterations",
                        "noslip_iterations",
                        "tolerance",
                        "ls_tolerance",
                        "noslip_tolerance",
                    )
                    if hasattr(self.sim.model.opt, name)
                },
                "policy_queries": [],
                "policy_actions": [],
                "policy_transition_flags": [],
            }
            EPISODE_FORENSICS.append(self._forensic_record)
            ACTIVE_FORENSIC_RECORD = self._forensic_record
            return obs

        def step(self, action: Any):
            obs, reward, done, info = super().step(action)
            self._steps_after_init += 1
            record = self._forensic_record
            if record is None:
                raise RuntimeError("step called before canonical state recording")
            current_sim_state_sha256 = hash_array(self.sim.get_state().flatten())
            record["current_sim_state_sha256"] = current_sim_state_sha256
            if self._steps_after_init == NUM_WAIT_STEPS:
                record["post_wait_sim_state_sha256"] = current_sim_state_sha256
                record["initial_raw_image_sha256"] = hash_array(
                    obs["agentview_image"]
                )
                record["initial_model_image_sha256"] = hash_array(model_image(obs))
                record["initial_robot_state_sha256"] = hash_array(
                    evaluator.build_robot_state(obs, 8)
                )
            elif self._steps_after_init > NUM_WAIT_STEPS:
                action_array = np.asarray(action, dtype=np.float32)
                record["policy_actions"].append(
                    {
                        "values": action_array.tolist(),
                        "sha256": hash_array(action_array),
                    }
                )
                record["policy_transition_flags"].append(
                    {
                        "reward": float(reward),
                        "done": bool(done),
                        "post_step_sim_state_sha256": current_sim_state_sha256,
                        "post_step_raw_image_sha256": hash_array(
                            obs["agentview_image"]
                        ),
                    }
                )
            return obs, reward, done, info

    libero_envs.OffScreenRenderEnv = RecordingOffScreenRenderEnv

    def recording_predict_chunk(
        model: Any,
        obs: dict[str, Any],
        instruction: Any,
        stats: dict[str, Any],
        res: int,
        visual_projector: Any = None,
    ) -> np.ndarray:
        if ACTIVE_FORENSIC_RECORD is None:
            raise RuntimeError("Prediction called before canonical state recording")
        query = {
            "raw_image_sha256": hash_array(obs["agentview_image"]),
            "model_image_sha256": hash_array(model_image(obs)),
            "robot_state_sha256": hash_array(
                evaluator.build_robot_state(obs, int(stats["state_dim"]))
            ),
            "instruction_token_ids": instruction.detach().cpu().tolist(),
            "instruction_token_sha256": hash_array(instruction.detach().cpu().numpy()),
            "sim_state_sha256": ACTIVE_FORENSIC_RECORD["current_sim_state_sha256"],
        }
        chunk = original_predict_chunk(
            model,
            obs,
            instruction,
            stats,
            res,
            visual_projector=visual_projector,
        )
        chunk_array = np.asarray(chunk, dtype=np.float32)
        query["raw_action_chunk"] = chunk_array.tolist()
        query["raw_action_chunk_sha256"] = hash_array(chunk_array)
        ACTIVE_FORENSIC_RECORD["policy_queries"].append(query)
        return chunk

    evaluator.predict_chunk = recording_predict_chunk


def environment_identity() -> dict[str, Any]:
    import mujoco
    import torch

    freeze = pip_freeze()
    freeze_without_mujoco = [
        line for line in freeze if not line.lower().startswith("mujoco==")
    ]
    source_path = Path(__file__).resolve()
    evaluator_path = Path(evaluator.__file__).resolve()
    nvidia_smi = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,driver_version,name",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    identity = {
        "schema_version": SCHEMA_VERSION,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "python_executable": sys.executable,
        "python_executable_sha256": sha256_file(Path(sys.executable)),
        "python_version": sys.version,
        "pip_freeze": freeze,
        "pip_freeze_sha256": sha256_bytes("\n".join(freeze).encode("utf-8")),
        "pip_freeze_without_mujoco_sha256": sha256_bytes(
            "\n".join(freeze_without_mujoco).encode("utf-8")
        ),
        "critical_packages": {
            name: package_version(name)
            for name in ("torch", "pillow", "mujoco", "robosuite", "libero")
        },
        "mujoco_module": str(Path(mujoco.__file__).resolve()),
        "mujoco_install_record_sha256": install_record_sha256("mujoco"),
        "forensic_runner_sha256": sha256_file(source_path),
        "evaluator_sha256": sha256_file(evaluator_path),
        "torch_flags": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "nvidia_smi": nvidia_smi,
        },
        "environment_variables": {
            name: os.environ.get(name)
            for name in (
                "MUJOCO_GL",
                "PYOPENGL_PLATFORM",
                "PYTHONPATH",
                "PYTHONNOUSERSITE",
                "LD_LIBRARY_PATH",
            )
        },
    }
    identity_material = {
        key: identity[key]
        for key in (
            "python_executable_sha256",
            "python_version",
            "pip_freeze_sha256",
            "mujoco_install_record_sha256",
            "forensic_runner_sha256",
            "evaluator_sha256",
            "torch_flags",
        )
    }
    identity["environment_identity_sha256"] = sha256_bytes(
        json.dumps(identity_material, sort_keys=True).encode("utf-8")
    )
    return identity


def main() -> None:
    global NUM_WAIT_STEPS, RESOLUTION

    args = evaluator.parse_args()
    if args.mode not in ("smoke", "capability"):
        raise ValueError("Forensic wrapper supports only smoke and capability modes")
    NUM_WAIT_STEPS = args.num_steps_wait
    RESOLUTION = args.res
    if args.matmul_precision != "highest":
        raise ValueError("Version isolation is frozen at highest matrix precision")

    patch_environment()
    evaluator.main()

    result = json.loads(args.output.read_text())
    capability = result["capability"]
    episodes = capability["episodes"]
    if len(episodes) != len(EPISODE_FORENSICS):
        raise RuntimeError("Forensic episode count differs from evaluator result")
    for outcome, forensic in zip(episodes, EPISODE_FORENSICS, strict=True):
        forensic["task_index"] = outcome["task_index"]
        forensic["episode"] = outcome["episode"]
        forensic["success"] = outcome["success"]
        forensic["steps"] = outcome["steps"]
        if len(forensic["policy_actions"]) != outcome["steps"]:
            raise RuntimeError("Recorded action count differs from evaluator step count")
        required = (
            "post_wait_sim_state_sha256",
            "initial_raw_image_sha256",
            "initial_model_image_sha256",
            "initial_robot_state_sha256",
        )
        if any(key not in forensic for key in required):
            raise RuntimeError("Initial-state forensic record is incomplete")

    result["version_isolation_forensics"] = {
        "schema_version": SCHEMA_VERSION,
        "environment": environment_identity(),
        "episodes": EPISODE_FORENSICS,
        "records_all_executed_policy_actions": True,
        "records_all_raw_action_chunks": True,
        "frozen_inputs": {
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "cache_sha256": sha256_file(args.cache),
            "source_sha256": {
                path: sha256_file(Path(path)) for path in SOURCE_FILES
            },
        },
        "initial_image_definition": (
            "Observation after canonical reset and ten dummy settle actions. The model image "
            "applies the evaluator's 180-degree rotation and PIL resize."
        ),
    }
    temporary = args.output.with_suffix(args.output.suffix + ".forensics.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(
        "FORENSICS",
        json.dumps(
            {
                "output": str(args.output),
                "episodes": len(EPISODE_FORENSICS),
                "environment_identity_sha256": result["version_isolation_forensics"][
                    "environment"
                ]["environment_identity_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
