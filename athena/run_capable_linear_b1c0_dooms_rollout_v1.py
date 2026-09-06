#!/usr/bin/env python3
"""Paired closed-loop capability for the serialized Dooms-style VLA truncation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import secrets
import stat
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


FROZEN_ROOT = Path("/work/joy/x-vla-capable-linear-b1c0-odt-v2")
LANE_ROOT = Path("/work/joy/x-vla-capable-linear-b1c0-dooms-v1")
OUTPUT_ROOT = FROZEN_ROOT / "athena/results/capable_linear_b1c0_dooms_v1"
SWEEP_RESULT = OUTPUT_ROOT / "dooms_dimension_sweep.json"
ARTIFACT_MODULE = LANE_ROOT / "xvla/train/implicit_projective_dag_artifact.py"
LANE_LEDGER = LANE_ROOT / "stage.sha256"
FROZEN_LEDGER = FROZEN_ROOT / "athena/capable_linear_b1c0_stage.sha256"
EXPECTED_CHECKPOINT_SHA256 = (
    "b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee"
)
EXPECTED_ARTIFACT_MODULE_SHA256 = (
    "e302131d021ed8fac5b8ec5bb1d321b4f3ca035487f4ba7626159d914d0f4bd9"
)
MAX_STEPS = 280
SETTLE_STEPS = 10
EXECUTION_HORIZON = 8
ACTION_HORIZON = 8
ACTION_DIMENSION = 7
RESOLUTION = 64
SHARDS = ((0, 3), (3, 6), (6, 8), (8, 10))
UNIT_OUTPUT = LANE_ROOT / "results/rollout_unit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ledger(path: Path, root: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing physical ledger: {path}")
    result: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        fields = raw.strip().split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(f"malformed ledger line {number}: {path}")
        digest, relative = fields
        relative = relative.lstrip("*")
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative in result
        ):
            raise RuntimeError(f"invalid ledger line {number}: {path}")
        candidate = root / relative
        resolved = candidate.resolve(strict=True)
        if candidate.is_symlink() or resolved == root or root not in resolved.parents:
            raise RuntimeError(f"unsafe ledger member: {relative}")
        if sha256(resolved) != digest:
            raise RuntimeError(f"ledger member changed: {relative}")
        result[relative] = digest
    return result


def verify_roots() -> dict[str, Any]:
    lane = read_ledger(LANE_LEDGER, LANE_ROOT)
    frozen = read_ledger(FROZEN_LEDGER, FROZEN_ROOT)
    required = {
        Path(__file__).resolve().relative_to(LANE_ROOT).as_posix(),
        "xvla/train/implicit_projective_dag_artifact.py",
    }
    if not required.issubset(lane):
        raise RuntimeError("lane ledger omits rollout sources")
    if lane["xvla/train/implicit_projective_dag_artifact.py"] != (
        EXPECTED_ARTIFACT_MODULE_SHA256
    ):
        raise RuntimeError("serializer identity differs")
    if frozen.get("inputs/capable_linear_b1c0_checkpoint.pt") != (
        EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError("checkpoint identity differs")
    return {
        "lane_ledger_sha256": sha256(LANE_LEDGER),
        "frozen_stage_ledger_sha256": sha256(FROZEN_LEDGER),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }


def read_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON constant: {value}")
        ),
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def publish(path: Path, payload: Mapping[str, Any]) -> str:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing existing output: {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(12)}.tmp")
    linked = False
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if not stat.S_ISREG(os.lstat(temporary).st_mode):
            raise RuntimeError("temporary result is not a regular file")
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        os.chmod(path, 0o444, follow_symlinks=False)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if sha256(path) != digest or os.lstat(path).st_mode & 0o222:
            raise RuntimeError("published result identity differs")
        return digest
    except BaseException:
        if linked and os.path.lexists(path):
            path.unlink()
        raise
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


if FROZEN_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, FROZEN_ROOT.as_posix())

import numpy as np
import torch
from PIL import Image

import scripts.run_capable_linear_fresh_capability as capability
from xvla.train.implicit_sparse_projective_odt import evaluate_boundary_quotient
from xvla.train.implicit_sparse_projective_odt_vla import (
    physical_batch_from_model_inputs,
    physical_source_mapping,
)


def load_artifact_module() -> Any:
    if sha256(ARTIFACT_MODULE) != EXPECTED_ARTIFACT_MODULE_SHA256:
        raise RuntimeError("serializer changed before loading")
    name = "xvla_dooms_artifact_v1"
    specification = importlib.util.spec_from_file_location(name, ARTIFACT_MODULE)
    if specification is None or specification.loader is None:
        raise RuntimeError("serializer module could not be loaded")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def configure_runtime() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("closed-loop evaluation requires a CUDA EGL device")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    versions = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
    }
    if (
        versions["python"] != "3.10.19"
        or versions["numpy"] != "1.26.4"
        or versions["torch"] != "2.7.1+cu126"
        or versions["cuda"] != "12.6"
        or versions["mujoco_gl"] != "egl"
        or versions["pyopengl_platform"] != "egl"
    ):
        raise RuntimeError(f"runtime identity differs: {versions}")
    return versions


def load_sweep() -> tuple[dict[str, Any], Path, str]:
    value = read_object(SWEEP_RESULT)
    artifact = value.get("serialized_tensor_network")
    selection = value.get("physical_selection")
    if (
        value.get("schema") != "xvla_capable_linear_b1c0_dooms_dimension_sweep_v1"
        or value.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
        or value.get("all_full_rank_and_materialization_gates_pass") is not True
        or not isinstance(artifact, dict)
        or not isinstance(selection, dict)
        or selection.get("requested_actual_fraction_unique_bond_dimensions_removed")
        != 0.70
    ):
        raise RuntimeError("sweep result gates or identity differ")
    root = Path(str(artifact.get("path", "")))
    manifest_digest = artifact.get("manifest_sha256")
    if (
        root != OUTPUT_ROOT / "uniform_actual70_tensor_network"
        or root.is_symlink()
        or not root.is_dir()
        or type(manifest_digest) is not str
        or len(manifest_digest) != 64
        or sha256(root / "manifest.json") != manifest_digest
    ):
        raise RuntimeError("serialized network identity differs")
    return value, root, manifest_digest


def compressed_predict(
    network: Any,
    model: Any,
    observations: Sequence[dict[str, Any]],
    instructions: Sequence[str],
    normalization: Mapping[str, np.ndarray],
) -> np.ndarray:
    if not observations or len(observations) != len(instructions):
        raise ValueError("compressed batch inventory differs")
    image_arrays = []
    raw_states = []
    identifiers = []
    for observation, instruction in zip(observations, instructions):
        rotated = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
        image_arrays.append(
            np.asarray(
                Image.fromarray(rotated).resize((RESOLUTION, RESOLUTION)),
                dtype=np.uint8,
            ).copy()
        )
        raw_states.append(capability._robot_state(observation))
        identifiers.append(capability._encode(instruction))
    images = (
        torch.from_numpy(np.stack(image_arrays))
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float64)
        .div(255.0)
    )
    states_array = (
        np.stack(raw_states) - normalization["state_mean"]
    ) / normalization["state_std"]
    states = torch.tensor(states_array, dtype=torch.float64)
    instruction_ids = torch.tensor(identifiers, dtype=torch.long)
    embodiments = torch.zeros(len(observations), dtype=torch.long)
    physical = physical_batch_from_model_inputs(
        model, images, instruction_ids, states, embodiments
    )
    raw = physical_source_mapping(model, physical)
    started = time.perf_counter()
    normalized = evaluate_boundary_quotient(network, raw).reshape(
        len(observations), ACTION_HORIZON, ACTION_DIMENSION
    )
    elapsed = time.perf_counter() - started
    actions = normalized * torch.tensor(
        normalization["action_std"], dtype=torch.float64
    ) + torch.tensor(normalization["action_mean"], dtype=torch.float64)
    result = actions.detach().cpu().numpy()
    if result.shape != (
        len(observations),
        ACTION_HORIZON,
        ACTION_DIMENSION,
    ) or not np.isfinite(result).all():
        raise RuntimeError("compressed policy output is malformed")
    compressed_predict.last_elapsed_seconds = elapsed
    return result


compressed_predict.last_elapsed_seconds = 0.0


def make_environment(suite: Any, task_index: int, episode: int) -> tuple[Any, Any, Any]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = suite.get_task(task_index)
    bddl_path = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    states = capability._official_init_states(suite, task_index)
    capability._require_expected_task_record(
        task_index, capability._task_record(task, bddl_path, states)
    )
    simulation = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=RESOLUTION,
        camera_widths=RESOLUTION,
    )
    simulation.seed(task_index * 100 + episode)
    observation = simulation.reset()
    observation = simulation.set_init_state(states[episode])
    observation = capability._settle_environment(
        simulation,
        observation,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    )
    return simulation, observation, states[episode]


def run_source_episode(
    suite: Any,
    model: Any,
    normalization: Mapping[str, np.ndarray],
    task_index: int,
    episode: int,
    max_steps: int,
) -> dict[str, Any]:
    simulation, observation, initial_state = make_environment(
        suite, task_index, episode
    )
    task = suite.get_task(task_index)
    instruction = torch.tensor(
        [capability._encode(str(task.language))], dtype=torch.long, device="cuda"
    )
    started = time.perf_counter()
    try:
        success, steps, terminated = capability._execute_episode(
            simulation,
            model,
            observation,
            instruction,
            normalization,
            max_steps,
        )
    finally:
        simulation.close()
    return {
        "task_index": task_index,
        "episode": episode,
        "initial_state_sha256": capability._array_sha256(initial_state),
        "success": success,
        "steps": steps,
        "terminated_without_success": terminated and not success,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_compressed_group(
    suite: Any,
    network: Any,
    model: Any,
    normalization: Mapping[str, np.ndarray],
    pairs: Sequence[tuple[int, int]],
    max_steps: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        for task_index, episode in pairs:
            simulation, observation, initial_state = make_environment(
                suite, task_index, episode
            )
            task = suite.get_task(task_index)
            entries.append(
                {
                    "task_index": task_index,
                    "episode": episode,
                    "simulation": simulation,
                    "observation": observation,
                    "instruction": str(task.language),
                    "initial_state_sha256": capability._array_sha256(initial_state),
                    "success": False,
                    "terminated": False,
                    "steps": 0,
                    "inference_seconds": 0.0,
                    "started": time.perf_counter(),
                }
            )
        while True:
            active = [
                entry
                for entry in entries
                if not entry["success"]
                and not entry["terminated"]
                and entry["steps"] < max_steps
            ]
            if not active:
                break
            chunks = compressed_predict(
                network,
                model,
                [entry["observation"] for entry in active],
                [entry["instruction"] for entry in active],
                normalization,
            )
            per_entry_inference = compressed_predict.last_elapsed_seconds / len(active)
            for entry in active:
                entry["inference_seconds"] += per_entry_inference
            for offset in range(EXECUTION_HORIZON):
                for entry, chunk in zip(active, chunks):
                    if (
                        entry["success"]
                        or entry["terminated"]
                        or entry["steps"] >= max_steps
                    ):
                        continue
                    action = chunk[offset].copy()
                    action[-1] = 1.0 if action[-1] > 0 else -1.0
                    observation, reward, done, _information = entry["simulation"].step(
                        action.tolist()
                    )
                    entry["observation"] = observation
                    entry["steps"] += 1
                    entry["success"] = bool(reward > 0)
                    entry["terminated"] = bool(done)
        return [
            {
                "task_index": entry["task_index"],
                "episode": entry["episode"],
                "initial_state_sha256": entry["initial_state_sha256"],
                "success": entry["success"],
                "steps": entry["steps"],
                "terminated_without_success": (
                    entry["terminated"] and not entry["success"]
                ),
                "elapsed_seconds": time.perf_counter() - entry["started"],
                "allocated_inference_seconds": entry["inference_seconds"],
            }
            for entry in entries
        ]
    finally:
        for entry in entries:
            entry["simulation"].close()


def output_path(mode: str, task_start: int, task_end: int) -> Path:
    if mode == "smoke":
        return OUTPUT_ROOT / "closed_loop_smoke.json"
    if mode == "pilot":
        return OUTPUT_ROOT / "closed_loop_pilot.json"
    return OUTPUT_ROOT / f"closed_loop_t{task_start}_{task_end}.json"


def evaluate(mode: str, task_start: int, task_end: int) -> None:
    roots_before = verify_roots()
    runtime = configure_runtime()
    sweep, artifact_root, manifest_digest = load_sweep()
    artifact_module = load_artifact_module()
    load_started = time.perf_counter()
    network = artifact_module.load_implicit_projective_dag_artifact(
        artifact_root, expected_manifest_sha256=manifest_digest
    )
    load_seconds = time.perf_counter() - load_started
    training, normalization = capability._training_record()
    model, checkpoint = capability._load_model()
    if checkpoint.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("source baseline loaded a different checkpoint")
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    if suite.n_tasks != 10:
        raise RuntimeError("LIBERO-Object task count differs")
    if mode == "smoke":
        pairs = [(0, 0)]
        max_steps = 1
        batch_size = 1
    elif mode == "pilot":
        pairs = [(task, 0) for task in range(10)]
        max_steps = MAX_STEPS
        batch_size = 10
    else:
        if (task_start, task_end) not in SHARDS:
            raise RuntimeError("task shard differs from the fixed partition")
        pairs = [
            (task, episode)
            for task in range(task_start, task_end)
            for episode in range(1, 11)
        ]
        max_steps = MAX_STEPS
        batch_size = 10
    inactive_counter, inactive_handle = capability._arm_inactive_vision_norm_sentinel(
        model
    )
    source_records = []
    started = time.perf_counter()
    try:
        for task_index, episode in pairs:
            record = run_source_episode(
                suite,
                model,
                normalization,
                task_index,
                episode,
                max_steps,
            )
            source_records.append(record)
            print("SOURCE", json.dumps(record, sort_keys=True), flush=True)
        source_forward = model.forward

        def reject_source_forward(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("compressed execution attempted source-model forward")

        model.forward = reject_source_forward
        compressed_records: list[dict[str, Any]] = []
        try:
            for offset in range(0, len(pairs), batch_size):
                group = run_compressed_group(
                    suite,
                    network,
                    model,
                    normalization,
                    pairs[offset : offset + batch_size],
                    max_steps,
                )
                compressed_records.extend(group)
                for record in group:
                    print("COMPRESSED", json.dumps(record, sort_keys=True), flush=True)
        finally:
            model.forward = source_forward
    finally:
        inactive_handle.remove()
    if inactive_counter["calls"] != 0:
        raise RuntimeError("inactive vision normalization site executed")
    source_by_pair = {
        (record["task_index"], record["episode"]): record
        for record in source_records
    }
    compressed_by_pair = {
        (record["task_index"], record["episode"]): record
        for record in compressed_records
    }
    if set(source_by_pair) != set(pairs) or set(compressed_by_pair) != set(pairs):
        raise RuntimeError("paired episode inventory differs")
    paired = []
    for pair in pairs:
        source = source_by_pair[pair]
        compressed = compressed_by_pair[pair]
        if source["initial_state_sha256"] != compressed["initial_state_sha256"]:
            raise RuntimeError(f"initial state differs for pair {pair}")
        paired.append(
            {
                "task_index": pair[0],
                "episode": pair[1],
                "initial_state_sha256": source["initial_state_sha256"],
                "source_success": source["success"],
                "compressed_success": compressed["success"],
                "source_steps": source["steps"],
                "compressed_steps": compressed["steps"],
                "compressed_allocated_inference_seconds": compressed[
                    "allocated_inference_seconds"
                ],
            }
        )
    source_successes = sum(int(record["source_success"]) for record in paired)
    compressed_successes = sum(
        int(record["compressed_success"]) for record in paired
    )
    selection = sweep["physical_selection"]
    payload = {
        "schema": "xvla_capable_linear_b1c0_dooms_closed_loop_v1",
        "mode": mode,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "dimension_point": {
            "requested_fraction_unique_bond_dimensions_removed": 0.70,
            "actual_fraction_unique_bond_dimensions_removed": selection[
                "actual_fraction_unique_bond_dimensions_removed"
            ],
            "original_unique_bond_dimensions": selection[
                "original_unique_bond_dimensions"
            ],
            "retained_unique_bond_dimensions": selection[
                "retained_unique_bond_dimensions"
            ],
        },
        "protocol": {
            "suite": "libero_object",
            "task_start": min(pair[0] for pair in pairs),
            "task_end": max(pair[0] for pair in pairs) + 1,
            "episodes": sorted(set(pair[1] for pair in pairs)),
            "max_steps": max_steps,
            "execution_horizon": EXECUTION_HORIZON,
            "settle_steps": SETTLE_STEPS,
            "paired_same_checkpoint_and_initial_states": True,
            "compressed_batch_size": batch_size,
            "observation_transform": "agentview image rotated 180 degrees and resized to 64x64",
            "gripper_decode": "+1 iff predicted coordinate > 0, else -1",
        },
        "runtime": runtime,
        "artifact": {
            "path": artifact_root.as_posix(),
            "manifest_sha256": manifest_digest,
            "eager_load_seconds": load_seconds,
            "source_model_forward_disabled_during_compressed_execution": True,
        },
        "trials": len(paired),
        "source_successes": source_successes,
        "compressed_successes": compressed_successes,
        "source_accuracy": source_successes / len(paired),
        "compressed_accuracy": compressed_successes / len(paired),
        "accuracy_change": (compressed_successes - source_successes) / len(paired),
        "no_observed_accuracy_drop": compressed_successes >= source_successes,
        "source_only_successes": sum(
            int(record["source_success"] and not record["compressed_success"])
            for record in paired
        ),
        "compressed_only_successes": sum(
            int(record["compressed_success"] and not record["source_success"])
            for record in paired
        ),
        "episodes": paired,
        "roots": roots_before,
        "roots_unchanged_after": verify_roots() == roots_before,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Smoke is an execution check only. Pilot and shard rows are paired closed-loop "
            "LIBERO success measurements at the reported actual unique-bond dimension removal."
        ),
        "all_identity_and_execution_gates_pass": True,
    }
    if not payload["roots_unchanged_after"]:
        raise RuntimeError("source roots changed during evaluation")
    path = output_path(mode, task_start, task_end)
    digest = publish(path, payload)
    print(json.dumps({"result_sha256": digest, "path": path.as_posix()}), flush=True)
    del network, model
    gc.collect()


def aggregate() -> None:
    roots_before = verify_roots()
    inputs = [OUTPUT_ROOT / "closed_loop_pilot.json"] + [
        OUTPUT_ROOT / f"closed_loop_t{start}_{stop}.json" for start, stop in SHARDS
    ]
    records = []
    identities = []
    expected_by_path = {
        inputs[0]: {(task, 0) for task in range(10)},
        **{
            OUTPUT_ROOT / f"closed_loop_t{start}_{stop}.json": {
                (task, episode)
                for task in range(start, stop)
                for episode in range(1, 11)
            }
            for start, stop in SHARDS
        },
    }
    expected_mode_by_path = {
        inputs[0]: "pilot",
        **{
            OUTPUT_ROOT / f"closed_loop_t{start}_{stop}.json": "shard"
            for start, stop in SHARDS
        },
    }
    common_dimension_point: dict[str, Any] | None = None
    for path in inputs:
        value = read_object(path)
        if (
            value.get("schema")
            != "xvla_capable_linear_b1c0_dooms_closed_loop_v1"
            or value.get("mode") != expected_mode_by_path[path]
            or value.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
            or value.get("all_identity_and_execution_gates_pass") is not True
            or not isinstance(value.get("episodes"), list)
            or not isinstance(value.get("dimension_point"), dict)
        ):
            raise RuntimeError(f"closed-loop shard failed validation: {path}")
        dimension_point = value["dimension_point"]
        if common_dimension_point is None:
            common_dimension_point = dimension_point
        elif dimension_point != common_dimension_point:
            raise RuntimeError("closed-loop shards used different dimension points")
        shard_records = value["episodes"]
        observed_pairs = [
            (record.get("task_index"), record.get("episode"))
            for record in shard_records
            if isinstance(record, dict)
        ]
        expected_pairs = expected_by_path[path]
        if (
            len(observed_pairs) != len(shard_records)
            or len(observed_pairs) != len(expected_pairs)
            or len(set(observed_pairs)) != len(observed_pairs)
            or set(observed_pairs) != expected_pairs
        ):
            raise RuntimeError(f"closed-loop shard inventory differs: {path}")
        for record in shard_records:
            if (
                type(record.get("source_success")) is not bool
                or type(record.get("compressed_success")) is not bool
                or type(record.get("initial_state_sha256")) is not str
            ):
                raise RuntimeError(f"closed-loop shard record differs: {path}")
        records.extend(shard_records)
        identities.append({"path": path.as_posix(), "sha256": sha256(path)})
    expected = [(task, episode) for task in range(10) for episode in range(11)]
    records.sort(key=lambda record: (record["task_index"], record["episode"]))
    observed = [(record["task_index"], record["episode"]) for record in records]
    if observed != expected:
        raise RuntimeError("aggregate paired episode inventory differs")
    source_successes = sum(int(record["source_success"] is True) for record in records)
    compressed_successes = sum(
        int(record["compressed_success"] is True) for record in records
    )
    sweep = read_object(SWEEP_RESULT)
    selection = sweep["physical_selection"]
    expected_dimension_point = {
        "requested_fraction_unique_bond_dimensions_removed": 0.70,
        "actual_fraction_unique_bond_dimensions_removed": selection[
            "actual_fraction_unique_bond_dimensions_removed"
        ],
        "original_unique_bond_dimensions": selection[
            "original_unique_bond_dimensions"
        ],
        "retained_unique_bond_dimensions": selection[
            "retained_unique_bond_dimensions"
        ],
    }
    if common_dimension_point != expected_dimension_point:
        raise RuntimeError("closed-loop dimension point differs from the sweep")
    payload = {
        "schema": "xvla_capable_linear_b1c0_dooms_accuracy_curve_v1",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "dimensions_removed_axis": [
            {
                "actual_fraction_unique_bond_dimensions_removed": 0.0,
                "accuracy": source_successes / len(records),
                "successes": source_successes,
                "trials": len(records),
            },
            {
                "actual_fraction_unique_bond_dimensions_removed": selection[
                    "actual_fraction_unique_bond_dimensions_removed"
                ],
                "accuracy": compressed_successes / len(records),
                "successes": compressed_successes,
                "trials": len(records),
            },
        ],
        "paired_accuracy_change": (
            compressed_successes - source_successes
        ) / len(records),
        "no_observed_accuracy_drop": compressed_successes >= source_successes,
        "source_only_successes": sum(
            int(record["source_success"] and not record["compressed_success"])
            for record in records
        ),
        "compressed_only_successes": sum(
            int(record["compressed_success"] and not record["source_success"])
            for record in records
        ),
        "episodes": records,
        "inputs": identities,
        "roots": roots_before,
        "roots_unchanged_after": verify_roots() == roots_before,
        "all_gates_pass": True,
        "claim_boundary": (
            "This establishes a paired two-point capability comparison, not yet a dense "
            "closed-loop truncation curve. The full uniform grid is reported for action fidelity."
        ),
    }
    if not payload["roots_unchanged_after"]:
        raise RuntimeError("source roots changed during aggregation")
    digest = publish(OUTPUT_ROOT / "closed_loop_accuracy_curve.json", payload)
    print(json.dumps({"aggregate_sha256": digest}, sort_keys=True), flush=True)


def unit() -> None:
    roots = verify_roots()
    artifact_module = load_artifact_module()
    expected_capability_constants = {
        "resolution": capability.RESOLUTION,
        "action_horizon": capability.ACTION_HORIZON,
        "action_dimension": capability.ACTION_DIM,
        "execution_horizon": capability.EXECUTION_HORIZON,
        "settle_steps": capability.SETTLE_STEPS,
        "checkpoint_sha256": capability.CHECKPOINT_SHA256,
    }
    if expected_capability_constants != {
        "resolution": RESOLUTION,
        "action_horizon": ACTION_HORIZON,
        "action_dimension": ACTION_DIMENSION,
        "execution_horizon": EXECUTION_HORIZON,
        "settle_steps": SETTLE_STEPS,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    }:
        raise RuntimeError("rollout protocol differs from the frozen evaluator")
    for name in (
        "export_implicit_projective_dag_artifact",
        "load_implicit_projective_dag_artifact",
    ):
        if not callable(getattr(artifact_module, name, None)):
            raise RuntimeError(f"serializer API is missing: {name}")
    payload = {
        "schema": "xvla_capable_linear_b1c0_dooms_rollout_unit_v1",
        "capability_protocol": expected_capability_constants,
        "shards": [list(value) for value in SHARDS],
        "roots": roots,
        "all_gates_pass": True,
    }
    digest = publish(UNIT_OUTPUT, payload)
    print(json.dumps({"rollout_unit_sha256": digest}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("unit", "smoke", "pilot", "shard", "aggregate"),
        required=True,
    )
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.mode == "unit":
        unit()
    elif arguments.mode == "aggregate":
        aggregate()
    else:
        evaluate(arguments.mode, arguments.task_start, arguments.task_end)


if __name__ == "__main__":
    main()
