"""Paired, fixed-panel LIBERO execution of genuinely narrowed ODT DAGs.

The physical-input adapter is pure preprocessing.  It cannot call a source
policy, and the reduced branch does not construct a source model at all.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from xvla.models.vla import ChiVLA, VLAConfig
from xvla.nn.normalization import RationalNorm
from xvla.train.implicit_projective_dag_mapped import open_mapped_implicit_projective_dag_artifact


CHECKPOINT_SHA256 = "b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee"
TRAINING_SHA256 = "e6c07beb2efb7e5ebe55c92fabf694d97b50b9e399f50868eeef4af455e30fa7"
REMOVALS = (30, 40, 50, 60, 70, 80)
PANEL = tuple((task, episode) for task in range(10) for episode in range(2))
MAX_STEPS = 280
HORIZON = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    result = json.loads(path.read_text(), object_pairs_hook=object_pairs,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if type(result) is not dict:
        raise ValueError("JSON object required")
    return result


def array_sha256(value: Any) -> str:
    values = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(values.shape)).encode())
    digest.update(b"\0")
    digest.update(values.tobytes())
    return digest.hexdigest()


def configure() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_training(inputs: Path) -> dict:
    checkpoint = inputs / "capable_linear_b1c0_checkpoint.pt"
    training = inputs / "capable_linear_training.json"
    if sha256(checkpoint) != CHECKPOINT_SHA256 or sha256(training) != TRAINING_SHA256:
        raise RuntimeError("authenticated policy inputs changed")
    return read_json(training)


def load_source(inputs: Path, configuration: dict) -> ChiVLA:
    model = ChiVLA(VLAConfig(**configuration))
    state = torch.load(inputs / "capable_linear_b1c0_checkpoint.pt", map_location="cpu", weights_only=True)
    if len(state) != 540:
        raise RuntimeError("checkpoint tensor count differs")
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval()
    norms = {name: module for name, module in model.named_modules() if isinstance(module, RationalNorm)}
    active = {name: module for name, module in norms.items() if name != "vision.norm_out"}
    if len(norms) != 74 or len(active) != 73 or model.num_params() != 20137352:
        raise RuntimeError("source architecture inventory differs")
    if any(not bool(module.initialized) for module in active.values()):
        raise RuntimeError("uninitialized active Pade normalization")
    def inactive_norm_error(*_args):
        raise RuntimeError("inactive vision norm executed")
    model.vision.norm_out.register_forward_pre_hook(inactive_norm_error)
    return model


def encode(instruction: str, vocabulary: dict) -> list[int]:
    values = [1] + [vocabulary.get(word, 0) for word in instruction.lower().replace(".", "").split()]
    return (values[:32] + [0] * max(0, 32 - len(values)))[:32]


def model_inputs(observations: list[dict], instructions: list[str], training: dict) -> tuple:
    from robosuite.utils.transform_utils import quat2axisangle
    image_values = []
    state_values = []
    for observation in observations:
        pixels = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
        image_values.append(np.asarray(Image.fromarray(pixels).resize((64, 64))).copy())
        state = np.concatenate((np.asarray(observation["robot0_eef_pos"], dtype=np.float64),
            quat2axisangle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64))).astype(np.float32)
        if state.size < 8:
            state = np.pad(state, (0, 8 - state.size))
        state_values.append(state[:8])
    images = torch.from_numpy(np.stack(image_values)).permute(0, 3, 1, 2).to(torch.float64).div(255.)
    normalization = training["normalization"]
    states = torch.tensor((np.stack(state_values) - np.asarray(normalization["state_mean"], dtype=np.float32)) /
                          np.asarray(normalization["state_std"], dtype=np.float32), dtype=torch.float64)
    tokens = torch.tensor([encode(text, training["vocab"]) for text in instructions], dtype=torch.long)
    if not bool(torch.isfinite(states).all()):
        raise RuntimeError("nonfinite observed state")
    return images, tokens, states


def physical_inputs(images: torch.Tensor, tokens: torch.Tensor, states: torch.Tensor) -> dict:
    batch = images.shape[0]
    patches = images.reshape(batch, 3, 8, 8, 8, 8).permute(0, 2, 4, 1, 3, 5).reshape(batch, 64, 192)
    categories = torch.nn.functional.one_hot(tokens, num_classes=26).to(torch.float64)
    return {**{f"image.patch{index}": patches[:, index] for index in range(64)},
            **{f"instruction.token{index}": categories[:, index] for index in range(32)}, "state": states}


@torch.inference_mode()
def predict(source: Any, mapped: Any, observations: list[dict], instructions: list[str], training: dict) -> tuple[np.ndarray, dict]:
    images, tokens, states = model_inputs(observations, instructions, training)
    started = time.monotonic()
    if mapped is not None:
        if source is not None:
            raise RuntimeError("reduced executor must not own a source model")
        normalized, receipt = mapped.evaluate_boundary_quotient(physical_inputs(images, tokens, states), return_receipt=True)
        if not receipt.root_only_live or not receipt.all_refcounts_zero:
            raise RuntimeError("mapped occurrence lifetime ledger did not close")
        normalized = normalized.reshape(len(observations), HORIZON, 7)
        metrics = {"nodes": receipt.node_evaluations, "edges": receipt.edge_occurrences_emitted,
                   "peak_live_values": receipt.peak_live_values, "edge_sha256": receipt.edge_ledger_sha256}
    else:
        if source is None:
            raise RuntimeError("baseline source is missing")
        normalized, loss = source(images.float().cuda(), tokens.cuda(), states.float().cuda(),
                                  torch.zeros(len(observations), dtype=torch.long, device="cuda"))
        if loss is not None:
            raise RuntimeError("baseline forward returned a training loss")
        metrics = {}
    elapsed = time.monotonic() - started
    normalization = training["normalization"]
    actions = normalized * torch.tensor(normalization["action_std"], dtype=normalized.dtype, device=normalized.device) + torch.tensor(normalization["action_mean"], dtype=normalized.dtype, device=normalized.device)
    if tuple(actions.shape) != (len(observations), HORIZON, 7) or not bool(torch.isfinite(actions).all()):
        raise RuntimeError("policy action batch is malformed")
    return actions.cpu().numpy(), {**metrics, "elapsed_seconds": elapsed, "batch_size": len(observations)}


def transition(environment: Any, action: list[float]) -> tuple:
    observation, reward, done, information = environment.step(action)
    if type(done) not in (bool, np.bool_) or not np.isscalar(reward) or not math.isfinite(float(reward)):
        raise RuntimeError("simulator returned an invalid transition")
    return observation, float(reward), bool(done), information


def run_panel(source: Any, mapped: Any, training: dict, protocol: dict,
              progress: Callable[[dict], None], *, smoke: bool = False) -> list[dict]:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    pairs = ((0, 0),) if smoke else PANEL
    max_steps = 1 if smoke else MAX_STEPS
    entries = []
    try:
        for task_index, episode in pairs:
            task = suite.get_task(task_index)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            original_load = torch.load
            def packaged_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return original_load(*args, **kwargs)
            torch.load = packaged_load
            try:
                states = suite.get_task_init_states(task_index)
            finally:
                torch.load = original_load
            expected = protocol["tasks"][str(task_index)]
            if [str(task.language), str(task.bddl_file), sha256(bddl), array_sha256(states)] != expected:
                raise RuntimeError(f"task {task_index} identity differs")
            environment = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
            entry = {"task": task_index, "episode": episode, "environment": environment,
                "instruction": str(task.language), "initial_state_sha256": array_sha256(states[episode]),
                "steps": 0, "success": False, "terminated": False}
            entries.append(entry)
            environment.seed(task_index * 100 + episode)
            environment.reset()
            observation = environment.set_init_state(states[episode])
            for _ in range(10):
                observation, reward, done, _info = transition(environment, [0., 0., 0., 0., 0., 0., -1.])
                if done or reward > 0:
                    raise RuntimeError("episode terminated during settling")
            entry["observation"] = observation
        inference_index = 0
        while True:
            active = [entry for entry in entries if not entry["success"] and not entry["terminated"] and entry["steps"] < max_steps]
            if not active:
                break
            actions, metrics = predict(source, mapped, [entry["observation"] for entry in active],
                                        [entry["instruction"] for entry in active], training)
            for offset in range(HORIZON):
                for entry, chunk in zip(active, actions):
                    if entry["success"] or entry["terminated"] or entry["steps"] >= max_steps:
                        continue
                    action = chunk[offset].copy()
                    action[-1] = 1. if action[-1] > 0 else -1.
                    observation, reward, done, _info = transition(entry["environment"], action.tolist())
                    entry.update(observation=observation, steps=entry["steps"] + 1, success=reward > 0, terminated=done)
            inference_index += 1
            progress({"inference_index": inference_index, "metrics": metrics, "episodes": episode_records(entries)})
        return episode_records(entries)
    finally:
        for entry in entries:
            entry["environment"].close()


def episode_records(entries: list[dict]) -> list[dict]:
    fields = ("task", "episode", "initial_state_sha256", "steps", "success", "terminated")
    return [{key: entry[key] for key in fields} for entry in entries]
