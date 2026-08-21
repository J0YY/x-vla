#!/usr/bin/env python3
"""Fixed-observation local noun-control specificity on canonical LIBERO-Object states."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
from PIL import Image

from athena.libero_dataset_metadata import DATASETS
from athena.run_xvla_experiment import (
    build_encoder,
    build_robot_state,
    build_vocab,
    file_sha256,
    load_cache_statistics,
    load_suite,
    make_config,
    official_init_states,
    readable_object_name,
    scene_object_bodies,
    task_languages,
)
from xvla.models.vla import ChiVLA


SCHEMA = "xvla-local-instruction-specificity-v2"
PROTOCOL = {
    "suite": "libero_object",
    "task_start": 0,
    "task_end": 10,
    "primary_episode_start": 10,
    "primary_eps_per_task": 40,
    "smoke_episode_start": 0,
    "smoke_eps_per_task": 1,
    "settle_steps": 10,
    "action_horizon": 8,
    "resolution": 64,
    "matmul_precision": "highest",
    "language_conditions": "ten official Object prompts plus empty/BOS-only",
    "scene_objects": "one official target plus five co-present distractors",
    "primary_metric": (
        "mean over five distractors of "
        "0.5 * (delta_A-delta_B) dot (unit_A-unit_B)"
    ),
    "control_metric": (
        "for each of five target-distractor axes, rank its semantic score against "
        "12 ordered contrasts among four truly absent-object prompts, then average ranks"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--provenance-result", type=Path, required=True)
    parser.add_argument("--expected-provenance-job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-seed", type=int, required=True)
    parser.add_argument("--mode", choices=("strict_smoke", "full"), required=True)
    parser.add_argument("--task-start", type=int, required=True)
    parser.add_argument("--task-end", type=int, required=True)
    parser.add_argument("--episode-start", type=int, required=True)
    parser.add_argument("--eps-per-task", type=int, required=True)
    parser.add_argument("--expected-trials", type=int, required=True)
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--num-steps-wait", type=int, default=10)
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


def hash_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def hash_fields(fields: list[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, value in fields:
        name_bytes = name.encode("utf-8")
        value_bytes = hash_array(value).encode("ascii")
        digest.update(len(name_bytes).to_bytes(4, "little"))
        digest.update(name_bytes)
        digest.update(len(value_bytes).to_bytes(4, "little"))
        digest.update(value_bytes)
    return digest.hexdigest()


def validate_provenance(
    path: Path,
    cache: Path,
    expected_cache_sha256: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Object cache provenance result is absent: {path}")
    with path.open() as handle:
        result = json.load(handle)
    errors = []
    expected_repository, expected_revision = DATASETS["libero_object"]
    if result.get("schema") != "xvla-cache-provenance-v1":
        errors.append("wrong provenance schema")
    if result.get("suite") != "libero_object":
        errors.append("provenance suite is not libero_object")
    if result.get("verified") is not True:
        errors.append("provenance result is not verified")
    cache_result = result.get("cache", {})
    if cache_result.get("sha256") != expected_cache_sha256:
        errors.append("provenance cache SHA does not match the frozen SHA")
    if int(cache_result.get("frames", -1)) != 66984:
        errors.append("provenance cache frame count is not 66,984")
    if int(cache_result.get("resolution", -1)) != 64:
        errors.append("provenance cache resolution is not 64")
    source_result = result.get("source", {})
    if source_result.get("repository") != expected_repository:
        errors.append("provenance source repository is not pinned Object")
    if source_result.get("revision") != expected_revision:
        errors.append("provenance source revision is not pinned Object")
    if source_result.get("content_hashes_match") is not True:
        errors.append("cache/source canonical content hashes do not match")
    mismatch_counts = result.get("comparison", {}).get("mismatch_counts", {})
    if not mismatch_counts or any(int(value) != 0 for value in mismatch_counts.values()):
        errors.append("provenance contains nonzero or absent mismatch counts")
    if file_sha256(cache) != expected_cache_sha256:
        errors.append("live cache SHA does not match the frozen SHA")
    if errors:
        raise RuntimeError("Invalid Object cache provenance: " + "; ".join(errors))
    return result


def target_phrase(language: str) -> str:
    normalized = " ".join(language.lower().strip().rstrip(".").split())
    prefix = "pick up the "
    suffix = " and place it in the basket"
    if not normalized.startswith(prefix) or not normalized.endswith(suffix):
        raise RuntimeError(f"Unexpected LIBERO-Object instruction template: {language}")
    return normalized[len(prefix) : -len(suffix)]


def transformed_model_input(
    observation: dict[str, Any],
    stats: dict[str, Any],
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    rotated = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    image = np.asarray(
        Image.fromarray(rotated).resize((resolution, resolution)),
        dtype=np.uint8,
    ).copy()
    state = build_robot_state(observation, stats["state_dim"]).astype(np.float32)
    return image, state


def simulator_state_array(environment: Any) -> np.ndarray:
    state = environment.sim.get_state()
    if hasattr(state, "flatten") and callable(state.flatten):
        state = state.flatten()
    array = np.asarray(state)
    if array.dtype == object:
        raise RuntimeError("Simulator state cannot be represented as a numeric array")
    array = np.ascontiguousarray(array)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise RuntimeError("Simulator state is nonnumeric or nonfinite")
    if array.size == 0:
        raise RuntimeError("Simulator state is empty")
    return array


def physical_input_hash(
    environment: Any,
    observation: dict[str, Any],
    stats: dict[str, Any],
    resolution: int,
    body_names: list[str],
) -> tuple[str, dict[str, str], np.ndarray, np.ndarray]:
    image, state = transformed_model_input(observation, stats, resolution)
    simulator_state = simulator_state_array(environment)
    fields = [
        ("simulator_state", simulator_state),
        ("model_image", image),
        ("robot_state", state),
        ("eef", np.asarray(observation["robot0_eef_pos"], dtype=np.float64)),
    ]
    component_hashes = {
        "simulator_state": hash_array(simulator_state),
        "model_image": hash_array(image),
        "robot_state": hash_array(state),
        "eef": hash_array(np.asarray(observation["robot0_eef_pos"], dtype=np.float64)),
    }
    for body_name in sorted(body_names):
        position = np.asarray(observation[f"{body_name}_pos"], dtype=np.float64)
        fields.append((f"body:{body_name}", position))
        component_hashes[f"body:{body_name}"] = hash_array(position)
    return hash_fields(fields), component_hashes, image, state


@torch.inference_mode()
def predict_all_chunks(
    model: ChiVLA,
    image_array: np.ndarray,
    raw_state: np.ndarray,
    encoded_prompts: list[list[int]],
    stats: dict[str, Any],
) -> np.ndarray:
    count = len(encoded_prompts)
    image = (
        torch.from_numpy(image_array)
        .permute(2, 0, 1)
        .float()
        .div(255)
        .unsqueeze(0)
        .repeat(count, 1, 1, 1)
        .cuda()
    )
    state_array = (raw_state - stats["state_mean"]) / stats["state_std"]
    state = (
        torch.tensor(state_array, dtype=torch.float32, device="cuda")
        .unsqueeze(0)
        .repeat(count, 1)
    )
    instruction = torch.tensor(
        encoded_prompts,
        dtype=torch.long,
        device="cuda",
    )
    embodiment = torch.zeros(count, dtype=torch.long, device="cuda")
    prediction, _ = model(image, instruction, state, embodiment)
    action_mean = torch.tensor(
        stats["action_mean"], dtype=torch.float32, device="cuda"
    )
    action_std = torch.tensor(
        stats["action_std"], dtype=torch.float32, device="cuda"
    )
    chunks = prediction * action_std + action_mean
    if chunks.shape[1:] != (PROTOCOL["action_horizon"], 7):
        raise RuntimeError(f"Unexpected action-chunk shape {tuple(chunks.shape)}")
    if not torch.isfinite(chunks).all():
        raise RuntimeError("Non-finite action chunk")
    return chunks.float().cpu().numpy()


def make_environment(task: Any, resolution: int) -> Any:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    return OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=resolution,
        camera_widths=resolution,
    )


def reset_to_input(
    environment: Any,
    init_state: Any,
    seed: int,
    settle_steps: int,
) -> dict[str, Any]:
    dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    environment.seed(seed)
    observation = environment.reset()
    observation = environment.set_init_state(init_state)
    for _ in range(settle_steps):
        observation, _, done, _ = environment.step(dummy_action)
        if done:
            raise RuntimeError("Environment terminated during deterministic settling")
    return observation


def execute_fixed_chunk(
    environment: Any,
    chunk: np.ndarray,
    start_eef: np.ndarray,
) -> dict[str, Any]:
    start = np.asarray(start_eef, dtype=np.float64)
    observation = None
    for action_row in chunk:
        action = np.asarray(action_row, dtype=np.float64).copy()
        action[-1] = 1.0 if action[-1] > 0 else -1.0
        observation, _, done, _ = environment.step(action.tolist())
        if done:
            raise RuntimeError("Environment terminated within the fixed eight-action chunk")
    if observation is None:
        raise RuntimeError("No local actions were executed")
    end = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
    return {
        "start_eef": start.tolist(),
        "end_eef": end.tolist(),
        "displacement": (end - start).tolist(),
        "steps": int(len(chunk)),
    }


def unit_direction(start: np.ndarray, target: np.ndarray) -> np.ndarray:
    vector = target - start
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("Degenerate target direction")
    return vector / norm


def specificity_metrics(
    conditions: dict[str, dict[str, Any]],
    target_prompt_id: int,
    distractor_prompt_ids: list[int],
    absent_prompt_ids: list[int],
    unit_directions: dict[int, np.ndarray],
) -> dict[str, Any]:
    def displacement(prompt_id: int | str) -> np.ndarray:
        return np.asarray(
            conditions[str(prompt_id)]["execution"]["displacement"],
            dtype=np.float64,
        )

    def command(prompt_id: int | str) -> np.ndarray:
        return np.asarray(
            conditions[str(prompt_id)]["predicted_cumulative_xyz"],
            dtype=np.float64,
        )

    target_displacement = displacement(target_prompt_id)
    if len(distractor_prompt_ids) != 5 or len(set(distractor_prompt_ids)) != 5:
        raise RuntimeError("Expected five distinct co-present distractor prompts")
    if len(absent_prompt_ids) != 4 or len(set(absent_prompt_ids)) != 4:
        raise RuntimeError("Expected four distinct absent-object prompts")
    if set(unit_directions) != {target_prompt_id, *distractor_prompt_ids}:
        raise RuntimeError("Unit-direction catalog does not match present objects")

    distractor_comparisons = []
    for distractor_prompt_id in distractor_prompt_ids:
        direction_contrast = (
            unit_directions[target_prompt_id] - unit_directions[distractor_prompt_id]
        )
        distractor_displacement = displacement(distractor_prompt_id)
        semantic_score = 0.5 * float(
            np.dot(
                target_displacement - distractor_displacement,
                direction_contrast,
            )
        )
        command_score = 0.5 * float(
            np.dot(
                command(target_prompt_id) - command(distractor_prompt_id),
                direction_contrast,
            )
        )
        controls = []
        for left_id in absent_prompt_ids:
            for right_id in absent_prompt_ids:
                if left_id == right_id:
                    continue
                score = 0.5 * float(
                    np.dot(
                        displacement(left_id) - displacement(right_id),
                        direction_contrast,
                    )
                )
                controls.append(
                    {
                        "left_prompt_id": left_id,
                        "right_prompt_id": right_id,
                        "score_m": score,
                    }
                )
        if len(controls) != 12:
            raise RuntimeError(f"Expected 12 ordered controls, got {len(controls)}")
        control_scores = np.asarray(
            [row["score_m"] for row in controls], dtype=np.float64
        )
        specificity_rank = float(
            (
                np.count_nonzero(control_scores < semantic_score)
                + 0.5 * np.count_nonzero(control_scores == semantic_score)
            )
            / len(control_scores)
        )
        distractor_comparisons.append(
            {
                "distractor_prompt_id": distractor_prompt_id,
                "semantic_score_m": semantic_score,
                "specificity_rank": specificity_rank,
                "ordered_absent_controls": controls,
                "secondary_predicted_command_score": command_score,
                "direction_contrast": direction_contrast.tolist(),
            }
        )
    semantic_score = float(
        np.mean([row["semantic_score_m"] for row in distractor_comparisons])
    )
    command_score = float(
        np.mean(
            [
                row["secondary_predicted_command_score"]
                for row in distractor_comparisons
            ]
        )
    )
    specificity_rank = float(
        np.mean([row["specificity_rank"] for row in distractor_comparisons])
    )
    empty_displacement = displacement("empty")
    return {
        "semantic_score_m": semantic_score,
        "semantic_score_positive": bool(semantic_score > 0),
        "specificity_rank": specificity_rank,
        "secondary_predicted_command_score": command_score,
        "distractor_comparisons": distractor_comparisons,
        "no_language_diagnostic": {
            "empty_displacement": empty_displacement.tolist(),
            "empty_displacement_norm_m": float(np.linalg.norm(empty_displacement)),
            "target_vs_empty_displacement_difference_m": float(
                np.linalg.norm(target_displacement - empty_displacement)
            ),
            "mean_distractor_vs_empty_displacement_difference_m": float(
                np.mean(
                    [
                        np.linalg.norm(
                            displacement(distractor_prompt_id) - empty_displacement
                        )
                        for distractor_prompt_id in distractor_prompt_ids
                    ]
                )
            ),
        },
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    repository_root = Path(__file__).resolve().parents[1]
    runner_source_sha256 = file_sha256(Path(__file__).resolve())
    imported_source_sha256 = {
        relative: file_sha256(repository_root / relative)
        for relative in (
            "athena/run_xvla_experiment.py",
            "xvla/models/vla.py",
            "xvla/models/vit.py",
            "xvla/nn/attention.py",
            "xvla/nn/normalization.py",
        )
    }
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not args.checkpoint.is_file() or not args.cache.is_file():
        raise FileNotFoundError("Checkpoint or cache is absent")
    if args.res != PROTOCOL["resolution"]:
        raise ValueError("Resolution is frozen at 64")
    if args.horizon != PROTOCOL["action_horizon"]:
        raise ValueError("Action horizon is frozen at eight")
    if args.num_steps_wait != PROTOCOL["settle_steps"]:
        raise ValueError("Settling is frozen at ten steps")
    if args.matmul_precision != PROTOCOL["matmul_precision"]:
        raise ValueError("Matmul precision is frozen at highest")
    if args.mode == "full":
        if (
            args.task_start,
            args.task_end,
            args.episode_start,
            args.eps_per_task,
        ) != (0, 10, 10, 40):
            raise ValueError(
                "Full protocol is frozen at ten tasks and canonical episodes 10 to 49"
            )
    elif (
        args.task_start,
        args.task_end,
        args.episode_start,
        args.eps_per_task,
    ) != (0, 10, 0, 1):
        raise ValueError(
            "Strict smoke is frozen at all ten tasks and canonical episode zero"
        )
    if args.expected_trials != (args.task_end - args.task_start) * args.eps_per_task:
        raise ValueError("Expected trial count does not match task and episode bounds")

    random.seed(args.checkpoint_seed)
    np.random.seed(args.checkpoint_seed)
    torch.manual_seed(args.checkpoint_seed)
    torch.cuda.manual_seed_all(args.checkpoint_seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    checkpoint_sha256 = file_sha256(args.checkpoint)
    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise RuntimeError("Checkpoint SHA does not match the frozen identity")
    provenance = validate_provenance(
        args.provenance_result,
        args.cache,
        args.expected_cache_sha256,
    )
    provenance_sha256 = file_sha256(args.provenance_result)

    suite = load_suite("libero_object")
    tasks = task_languages(suite)
    if set(tasks) != set(range(10)):
        raise RuntimeError("LIBERO-Object must expose exactly ten official tasks")
    prompt_targets = {prompt_id: target_phrase(text) for prompt_id, text in tasks.items()}
    if len(set(prompt_targets.values())) != 10:
        raise RuntimeError("Official Object target phrases are not unique")
    target_to_prompt = {target: prompt_id for prompt_id, target in prompt_targets.items()}
    vocab, _ = build_vocab(tasks)
    encode = build_encoder(vocab)
    prompt_catalog = [
        {
            "prompt_id": prompt_id,
            "text": tasks[prompt_id],
            "target_phrase": prompt_targets[prompt_id],
            "instruction_ids": encode(tasks[prompt_id]),
        }
        for prompt_id in range(10)
    ]
    prompt_catalog.append(
        {
            "prompt_id": "empty",
            "text": "",
            "target_phrase": None,
            "instruction_ids": encode(""),
        }
    )
    encoded_prompts = [row["instruction_ids"] for row in prompt_catalog]

    print("Loading frozen Object cache statistics", flush=True)
    stats = load_cache_statistics(args.cache, args.horizon)
    if int(stats["frame_count"]) != 66984 or int(stats["sample_count"]) != 63352:
        raise RuntimeError("Frozen Object cache statistics have unexpected counts")
    config = make_config(
        "chi",
        len(vocab),
        stats["state_dim"],
        stats["action_dim"],
        args.res,
        args.horizon,
        vision_encoder="vit",
    )
    model = ChiVLA(config).cuda()
    state_dict = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    for preflight_task_index in range(args.task_start, args.task_end):
        preflight_states = official_init_states(suite, preflight_task_index)
        if len(preflight_states) < args.episode_start + args.eps_per_task:
            raise RuntimeError(
                f"Task {preflight_task_index} exposes {len(preflight_states)} canonical "
                f"states, fewer than required {args.episode_start + args.eps_per_task}"
            )

    rows = []
    for task_index in range(args.task_start, args.task_end):
        task = suite.get_task(task_index)
        init_states = official_init_states(suite, task_index)
        environment = make_environment(task, args.res)
        try:
            for episode in range(
                args.episode_start,
                args.episode_start + args.eps_per_task,
            ):
                init_state_index = episode
                init_state = init_states[init_state_index]
                reset_seed = task_index * 100 + episode
                probe_observation = reset_to_input(
                    environment,
                    init_state,
                    reset_seed,
                    args.num_steps_wait,
                )
                bodies = scene_object_bodies(probe_observation)
                if len(bodies) != 6:
                    raise RuntimeError(
                        f"Task {task_index} state {episode} has {len(bodies)} eligible "
                        f"objects, expected exactly six: {bodies}"
                    )
                original_body = next(
                    (
                        body
                        for body in bodies
                        if readable_object_name(body) in task.language.lower()
                    ),
                    None,
                )
                if original_body is None:
                    raise RuntimeError("Official target body cannot be identified")
                original_name = readable_object_name(original_body)
                body_to_prompt_id = {
                    body: target_to_prompt.get(readable_object_name(body))
                    for body in bodies
                }
                if original_name not in target_to_prompt or any(
                    prompt_id is None for prompt_id in body_to_prompt_id.values()
                ):
                    raise RuntimeError(
                        f"Scene objects do not map one-to-one to official prompts: "
                        f"{body_to_prompt_id}"
                    )
                target_prompt_id = target_to_prompt[original_name]
                if target_prompt_id != task_index:
                    raise RuntimeError(
                        f"Task {task_index} target mapped to prompt {target_prompt_id}"
                    )
                present_prompt_ids = sorted(
                    int(prompt_id) for prompt_id in body_to_prompt_id.values()
                )
                if len(set(present_prompt_ids)) != 6 or target_prompt_id not in present_prompt_ids:
                    raise RuntimeError(
                        f"Present prompt identities are invalid: {present_prompt_ids}"
                    )
                distractor_prompt_ids = [
                    prompt_id
                    for prompt_id in present_prompt_ids
                    if prompt_id != target_prompt_id
                ]
                distractor_bodies = [
                    next(
                        body
                        for body, prompt_id in body_to_prompt_id.items()
                        if int(prompt_id) == distractor_prompt_id
                    )
                    for distractor_prompt_id in distractor_prompt_ids
                ]
                absent_prompt_ids = [
                    prompt_id
                    for prompt_id in range(10)
                    if prompt_id not in present_prompt_ids
                ]
                if len(distractor_prompt_ids) != 5 or len(absent_prompt_ids) != 4:
                    raise RuntimeError(
                        "Expected five present distractors and four absent prompts"
                    )

                input_hash, component_hashes, image, raw_state = physical_input_hash(
                    environment,
                    probe_observation,
                    stats,
                    args.res,
                    bodies,
                )
                chunks = predict_all_chunks(
                    model,
                    image,
                    raw_state,
                    encoded_prompts,
                    stats,
                )
                start_eef = np.asarray(
                    probe_observation["robot0_eef_pos"], dtype=np.float64
                )
                original_position = np.asarray(
                    probe_observation[f"{original_body}_pos"], dtype=np.float64
                )
                prompt_positions = {
                    int(prompt_id): np.asarray(
                        probe_observation[f"{body}_pos"], dtype=np.float64
                    )
                    for body, prompt_id in body_to_prompt_id.items()
                }
                unit_directions = {
                    prompt_id: unit_direction(start_eef, position)
                    for prompt_id, position in prompt_positions.items()
                }

                conditions: dict[str, dict[str, Any]] = {}
                for condition_index, catalog_row in enumerate(prompt_catalog):
                    condition_observation = reset_to_input(
                        environment,
                        init_state,
                        reset_seed,
                        args.num_steps_wait,
                    )
                    clone_hash, clone_components, _, _ = physical_input_hash(
                        environment,
                        condition_observation,
                        stats,
                        args.res,
                        bodies,
                    )
                    if clone_hash != input_hash or clone_components != component_hashes:
                        raise RuntimeError(
                            f"Input clone mismatch for task {task_index}, episode {episode}, "
                            f"prompt {catalog_row['prompt_id']}"
                        )
                    chunk = np.asarray(chunks[condition_index], dtype=np.float32)
                    execution = execute_fixed_chunk(
                        environment,
                        chunk,
                        np.asarray(
                            condition_observation["robot0_eef_pos"], dtype=np.float64
                        ),
                    )
                    prompt_key = str(catalog_row["prompt_id"])
                    conditions[prompt_key] = {
                        "prompt_id": catalog_row["prompt_id"],
                        "prompt_text": catalog_row["text"],
                        "instruction_ids": catalog_row["instruction_ids"],
                        "input_hash": clone_hash,
                        "action_chunk": chunk.tolist(),
                        "action_chunk_sha256": hash_array(chunk),
                        "predicted_cumulative_xyz": chunk[:, :3].sum(axis=0).tolist(),
                        "execution": execution,
                    }

                metrics = specificity_metrics(
                    conditions,
                    target_prompt_id,
                    distractor_prompt_ids,
                    absent_prompt_ids,
                    unit_directions,
                )
                row = {
                    "episode_identity": {
                        "suite": "libero_object",
                        "task_index": task_index,
                        "episode": episode,
                        "init_state_index": init_state_index,
                        "reset_seed": reset_seed,
                        "init_state_sha256": hash_array(np.asarray(init_state)),
                        "task_language": task.language,
                    },
                    "scene": {
                        "eligible_bodies": bodies,
                        "target_body": original_body,
                        "distractor_bodies": distractor_bodies,
                        "body_to_prompt_id": body_to_prompt_id,
                        "target_prompt_id": target_prompt_id,
                        "present_prompt_ids": present_prompt_ids,
                        "distractor_prompt_ids": distractor_prompt_ids,
                        "absent_prompt_ids": absent_prompt_ids,
                        "start_eef": start_eef.tolist(),
                        "target_position": original_position.tolist(),
                        "prompt_positions": {
                            str(prompt_id): position.tolist()
                            for prompt_id, position in prompt_positions.items()
                        },
                        "unit_directions": {
                            str(prompt_id): direction.tolist()
                            for prompt_id, direction in unit_directions.items()
                        },
                    },
                    "model_input": {
                        "combined_sha256": input_hash,
                        "component_sha256": component_hashes,
                        "clone_count": len(prompt_catalog),
                        "all_clone_hashes_match": True,
                    },
                    "conditions": conditions,
                    "metrics": metrics,
                }
                rows.append(row)
                print(
                    "LOCAL_INSTRUCTION_TRIAL",
                    json.dumps(
                        {
                            "task_index": task_index,
                            "episode": episode,
                            "semantic_score_m": metrics["semantic_score_m"],
                            "specificity_rank": metrics["specificity_rank"],
                        }
                    ),
                    flush=True,
                )
        finally:
            environment.close()

    if len(rows) != args.expected_trials:
        raise RuntimeError(f"Produced {len(rows)} trials, expected {args.expected_trials}")
    semantic_scores = np.asarray(
        [row["metrics"]["semantic_score_m"] for row in rows], dtype=np.float64
    )
    specificity_ranks = np.asarray(
        [row["metrics"]["specificity_rank"] for row in rows], dtype=np.float64
    )
    result = {
        "schema": SCHEMA,
        "mode": args.mode,
        "protocol": PROTOCOL,
        "identity": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_seed": args.checkpoint_seed,
            "cache": str(args.cache),
            "cache_sha256": args.expected_cache_sha256,
            "cache_frames": int(stats["frame_count"]),
            "cache_samples": int(stats["sample_count"]),
            "provenance_result": str(args.provenance_result),
            "provenance_result_sha256": provenance_sha256,
            "provenance_job_id": str(args.expected_provenance_job_id),
            "provenance_verified": provenance["verified"],
            "dataset_repository": provenance["source"]["repository"],
            "dataset_revision": provenance["source"]["revision"],
            "metadata_revision": provenance["metadata"]["revision"],
            "metadata_sha256": provenance["metadata"]["sha256"],
            "runner_source_sha256": runner_source_sha256,
            "imported_source_sha256": imported_source_sha256,
        },
        "evaluation": {
            "task_start": args.task_start,
            "task_end": args.task_end,
            "episode_start": args.episode_start,
            "eps_per_task": args.eps_per_task,
            "expected_trials": args.expected_trials,
            "completed_trials": len(rows),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "numerical_environment": {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "numpy_version": np.__version__,
                "pillow_version": importlib.metadata.version("Pillow"),
                "gpu": torch.cuda.get_device_name(0),
            },
        },
        "prompt_catalog": prompt_catalog,
        "descriptive_only": {
            "mean_semantic_score_m": float(semantic_scores.mean()),
            "fraction_semantic_score_positive": float((semantic_scores > 0).mean()),
            "mean_specificity_rank": float(specificity_ranks.mean()),
        },
        "rows": rows,
        "outcome_scope": (
            "This is an eight-action, fixed-observation local-control intervention. It does "
            "not measure absolute motion toward either named object, bidirectional noun "
            "selection, BDDL success, counterfactual task completion, robust grounding, "
            "zero-shot language understanding, broad-suite generalization, a practically "
            "meaningful effect size, or a benefit unique to tensor decomposability."
        ),
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(args.output, result)
    print("RESULT", json.dumps(result["descriptive_only"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
