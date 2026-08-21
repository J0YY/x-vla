#!/usr/bin/env python3
"""Run the frozen, prerequisite-gated counterfactual BDDL target-swap experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from athena.counterfactual_target_swap_common import (
    CACHE,
    CHECKPOINTS,
    FROZEN_GATES,
    MANIFEST_PATH,
    PROMPTS,
    PROTOCOL,
    PROVENANCE,
    RECEPTACLE_SYMBOL,
    SCHEMA_RUN,
    SPECIFICITY,
    SMOKE_RESULT_PATH,
    decode_array,
    encode_array,
    file_sha256,
    hash_array,
    source_hashes,
    validate_manifest,
    validate_libero_runtime,
    validate_specificity_summary,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("strict_smoke", "full"), required=True)
    parser.add_argument("--checkpoint-seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--provenance-result", type=Path, required=True)
    parser.add_argument("--specificity-summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-start", type=int, required=True)
    parser.add_argument("--task-end", type=int, required=True)
    parser.add_argument("--episode-start", type=int, required=True)
    parser.add_argument("--eps-per-task", type=int, required=True)
    parser.add_argument("--expected-rollouts", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--exec-h", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--matmul-precision", default="highest")
    return parser.parse_args()


def strict_pre_cuda_validation(args: argparse.Namespace, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """All prerequisite and artifact checks happen before torch is imported."""
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.checkpoint_seed not in CHECKPOINTS:
        raise ValueError("Checkpoint seed is not one of the three frozen seeds")
    expected_checkpoint = CHECKPOINTS[args.checkpoint_seed]
    frozen_paths = {
        "checkpoint": root / expected_checkpoint["path"],
        "cache": root / CACHE["path"],
        "provenance": root / PROVENANCE["path"],
        "specificity": root / SPECIFICITY["path"],
    }
    observed_paths = {
        "checkpoint": args.checkpoint,
        "cache": args.cache,
        "provenance": args.provenance_result,
        "specificity": args.specificity_summary,
    }
    for label, expected in frozen_paths.items():
        if observed_paths[label].resolve() != expected.resolve():
            raise RuntimeError(f"{label} argument is not the frozen path")
    if args.manifest.resolve() != (root / MANIFEST_PATH).resolve():
        raise RuntimeError("Manifest argument is not the frozen path")
    if args.mode == "strict_smoke" and args.output.resolve() != (root / SMOKE_RESULT_PATH).resolve():
        raise RuntimeError("Smoke output is not the frozen fresh path")
    if file_sha256(args.checkpoint) != expected_checkpoint["sha256"]:
        raise RuntimeError("Checkpoint SHA differs from the frozen identity")
    if file_sha256(args.cache) != CACHE["sha256"]:
        raise RuntimeError("Cache SHA differs from the frozen identity")
    specificity = validate_specificity_summary(args.specificity_summary)
    manifest = validate_manifest(args.manifest, root)
    if manifest["specificity_summary_sha256"] != file_sha256(args.specificity_summary):
        raise RuntimeError("Manifest and live specificity summary identities differ")
    if manifest["cache"]["live_sha256"] != file_sha256(args.cache):
        raise RuntimeError("Manifest and live cache identities differ")
    if manifest["provenance"]["live_sha256"] != file_sha256(args.provenance_result):
        raise RuntimeError("Manifest and live provenance identities differ")
    frozen_scalars = (
        args.res,
        args.horizon,
        args.exec_h,
        args.max_steps,
        args.num_steps_wait,
        args.matmul_precision,
    )
    expected_scalars = (
        PROTOCOL["resolution"],
        PROTOCOL["action_horizon"],
        PROTOCOL["execution_horizon"],
        PROTOCOL["max_steps"],
        PROTOCOL["settle_steps"],
        PROTOCOL["matmul_precision"],
    )
    if frozen_scalars != expected_scalars:
        raise ValueError("Rollout arguments differ from the frozen protocol")
    if args.mode == "strict_smoke":
        if (args.checkpoint_seed, args.task_start, args.task_end, args.episode_start, args.eps_per_task, args.expected_rollouts) != (0, 0, 10, 40, 1, 0):
            raise ValueError("Strict smoke must cover all mappings at canonical episode 40 with seed 0")
    else:
        if [args.task_start, args.task_end] not in PROTOCOL["task_shards"]:
            raise ValueError("Full task range is not a frozen two-task shard")
        if (args.episode_start, args.eps_per_task) != (40, 10):
            raise ValueError("Full jobs are frozen at canonical episodes 40 through 49")
        expected_rollouts = (args.task_end - args.task_start) * 110
        if args.expected_rollouts != expected_rollouts:
            raise ValueError("Expected rollout count differs from shard protocol")
        expected_output = root / f"results/counterfactual_target_swap_v1_s{args.checkpoint_seed}_t{args.task_start}_{args.task_end}.json"
        if args.output.resolve() != expected_output.resolve():
            raise RuntimeError("Full output is not the frozen seed/shard path")
    required_environment = {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "XVLA_FROZEN_EVALUATION_GPU_FAMILY": PROTOCOL["gpu_family"],
        "CUBLAS_WORKSPACE_CONFIG": PROTOCOL["cublas_workspace_config"],
    }
    for name, expected in required_environment.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"{name} must equal {expected}")
    expected_runner = os.environ.get("XVLA_TARGET_SWAP_RUNNER_SHA256")
    expected_common = os.environ.get("XVLA_TARGET_SWAP_COMMON_SHA256")
    if expected_runner != file_sha256(Path(__file__).resolve()):
        raise RuntimeError("Runner source does not match wrapper declaration")
    if expected_common != file_sha256(Path(__file__).with_name("counterfactual_target_swap_common.py")):
        raise RuntimeError("Common source does not match wrapper declaration")
    return manifest, specificity


def hash_fields(fields: list[tuple[str, Any]], np: Any) -> str:
    digest = hashlib.sha256()
    for name, value in fields:
        contiguous = np.ascontiguousarray(value)
        name_bytes = name.encode("utf-8")
        value_digest = hash_array(contiguous).encode("ascii")
        digest.update(len(name_bytes).to_bytes(4, "little"))
        digest.update(name_bytes)
        digest.update(len(value_digest).to_bytes(4, "little"))
        digest.update(value_digest)
    return digest.hexdigest()


def transformed_input(observation: dict[str, Any], stats: dict[str, Any], res: int, np: Any, Image: Any, build_robot_state: Any) -> tuple[Any, Any]:
    image = np.asarray(
        Image.fromarray(np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])).resize((res, res)),
        dtype=np.uint8,
    ).copy()
    state = build_robot_state(observation, stats["state_dim"]).astype(np.float32)
    return image, state


def simulator_state(environment: Any, np: Any) -> Any:
    value = environment.sim.get_state()
    if hasattr(value, "flatten") and callable(value.flatten):
        value = value.flatten()
    array = np.ascontiguousarray(value)
    if array.dtype == object or array.size == 0 or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise RuntimeError("MuJoCo state is empty, nonnumeric, or nonfinite")
    return array


def physical_snapshot(environment: Any, observation: dict[str, Any], stats: dict[str, Any], res: int, body_names: list[str], np: Any, Image: Any, build_robot_state: Any) -> dict[str, Any]:
    image, state = transformed_input(observation, stats, res, np, Image, build_robot_state)
    sim = simulator_state(environment, np)
    raw_image = np.ascontiguousarray(observation["agentview_image"])
    fields = [
        ("mujoco_state", sim),
        ("raw_agentview_image", raw_image),
        ("model_image", image),
        ("model_robot_state", state),
    ]
    components = {
        "mujoco_state": hash_array(sim),
        "raw_agentview_image": hash_array(raw_image),
        "model_image": hash_array(image),
        "model_robot_state": hash_array(state),
    }
    robot_fields = []
    for key in sorted(key for key in observation if key.startswith("robot0") and np.asarray(observation[key]).dtype != object):
        value = np.ascontiguousarray(observation[key])
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise RuntimeError(f"Robot observation {key} is nonnumeric or nonfinite")
        fields.append((f"robot:{key}", value))
        robot_fields.append((key, value))
    if not robot_fields:
        raise RuntimeError("No numeric robot observation fields were found")
    components["all_robot_observations"] = hash_fields(robot_fields, np)
    object_components = {}
    for body in sorted(body_names):
        body_fields = []
        for key in sorted(key for key in observation if key.startswith(body + "_")):
            value = np.ascontiguousarray(observation[key])
            if value.dtype == object or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                raise RuntimeError(f"Object observation {key} is nonnumeric or nonfinite")
            fields.append((f"object:{key}", value))
            body_fields.append((key, value))
        required = {f"{body}_pos", f"{body}_quat"}
        if not required.issubset({name for name, _ in body_fields}):
            raise RuntimeError(f"Object {body} is missing position or quaternion observations")
        object_components[body] = hash_fields(body_fields, np)
    components["all_object_observations"] = hash_fields(
        [(body, np.frombuffer(bytes.fromhex(digest), dtype=np.uint8)) for body, digest in sorted(object_components.items())],
        np,
    )
    combined = hash_fields(fields, np)
    return {
        "combined_sha256": combined,
        "component_sha256": components,
        "per_object_sha256": object_components,
        "robot_field_names": [name for name, _ in robot_fields],
        "object_field_names": {
            body: sorted(key for key in observation if key.startswith(body + "_"))
            for body in sorted(body_names)
        },
        "model_image": image,
        "model_robot_state": state,
    }


def public_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in snapshot.items() if key not in {"model_image", "model_robot_state"}}


def reset_environment(environment: Any, init_state: Any, reset_seed: int, settle_steps: int, np: Any) -> dict[str, Any]:
    environment.seed(reset_seed)
    observation = environment.reset()
    observation = environment.set_init_state(init_state)
    dummy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    for _ in range(settle_steps):
        observation, reward, done, _ = environment.step(dummy)
        if bool(done) or float(reward) > 0:
            raise RuntimeError("Episode terminated or succeeded during deterministic settling")
    return observation


def predict_chunk(model: Any, image_array: Any, raw_state: Any, instruction_ids: list[int], stats: dict[str, Any], torch: Any, np: Any) -> Any:
    with torch.inference_mode():
        image = torch.from_numpy(image_array).permute(2, 0, 1).float().div(255).unsqueeze(0).cuda()
        normalized_state = (raw_state - stats["state_mean"]) / stats["state_std"]
        state = torch.tensor(normalized_state, dtype=torch.float32, device="cuda").unsqueeze(0)
        instruction = torch.tensor([instruction_ids], dtype=torch.long, device="cuda")
        embodiment = torch.zeros(1, dtype=torch.long, device="cuda")
        prediction, _ = model(image, instruction, state, embodiment)
        action_mean = torch.tensor(stats["action_mean"], dtype=torch.float32, device="cuda")
        action_std = torch.tensor(stats["action_std"], dtype=torch.float32, device="cuda")
        chunk = (prediction[0] * action_std + action_mean).float().cpu().numpy()
    if chunk.shape != (PROTOCOL["action_horizon"], 7) or not np.isfinite(chunk).all():
        raise RuntimeError("Predicted action chunk is malformed or nonfinite")
    return chunk


def rollout(environment: Any, init_state: Any, reset_seed: int, instruction_ids: list[int], prompt_id: int, model: Any, stats: dict[str, Any], body_names: list[str], args: argparse.Namespace, torch: Any, np: Any, Image: Any, build_robot_state: Any) -> dict[str, Any]:
    observation = reset_environment(environment, init_state, reset_seed, args.num_steps_wait, np)
    start = physical_snapshot(environment, observation, stats, args.res, body_names, np, Image, build_robot_state)
    chunks = []
    executed = []
    rewards = []
    dones = []
    steps = 0
    success = False
    while steps < args.max_steps and not success:
        if steps == 0:
            model_image = start["model_image"]
            model_state = start["model_robot_state"]
        else:
            model_image, model_state = transformed_input(
                observation,
                stats,
                args.res,
                np,
                Image,
                build_robot_state,
            )
        chunk = predict_chunk(
            model,
            model_image,
            model_state,
            instruction_ids,
            stats,
            torch,
            np,
        )
        executed_from_chunk = 0
        chunk_start = steps
        for offset in range(min(args.exec_h, args.horizon, args.max_steps - steps)):
            action = np.asarray(chunk[offset], dtype=np.float64).copy()
            action[-1] = 1.0 if action[-1] > 0 else -1.0
            observation, reward, done, _ = environment.step(action.tolist())
            executed.append(action)
            rewards.append(float(reward))
            dones.append(bool(done))
            steps += 1
            executed_from_chunk += 1
            success = success or float(reward) > 0
            if bool(done) or success or steps >= args.max_steps:
                break
        chunks.append({
            "start_step": chunk_start,
            "executed_rows": executed_from_chunk,
            "prediction": encode_array(np.asarray(chunk, dtype=np.float32)),
        })
        if dones[-1]:
            break
    actions_array = np.asarray(executed, dtype=np.float64).reshape(-1, 7)
    rewards_array = np.asarray(rewards, dtype=np.float64)
    dones_array = np.asarray(dones, dtype=np.bool_)
    if steps != len(actions_array) or steps != len(rewards_array) or steps != len(dones_array):
        raise RuntimeError("Trajectory lengths are inconsistent")
    reproduced = []
    for item in chunks:
        chunk = decode_array(item["prediction"])
        for action in chunk[: int(item["executed_rows"])]:
            row = np.asarray(action, dtype=np.float64).copy()
            row[-1] = 1.0 if row[-1] > 0 else -1.0
            reproduced.append(row)
    reproduced_array = np.asarray(reproduced, dtype=np.float64).reshape(-1, 7)
    if not np.array_equal(reproduced_array, actions_array):
        raise RuntimeError("Executed actions do not reproduce from action chunks")
    if success != bool(np.any(rewards_array > 0)):
        raise RuntimeError("Success does not reproduce from rewards")
    final = physical_snapshot(environment, observation, stats, args.res, body_names, np, Image, build_robot_state)
    return {
        "prompt_id": prompt_id,
        "instruction_ids": instruction_ids,
        "start_physical": public_snapshot(start),
        "chunks": chunks,
        "executed_actions": encode_array(actions_array),
        "rewards": encode_array(rewards_array),
        "dones": encode_array(dones_array),
        "steps": steps,
        "success": success,
        "success_step": int(np.argmax(rewards_array > 0) + 1) if success else None,
        "final_physical": public_snapshot(final),
    }


def make_environment(path: Path, res: int) -> Any:
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(bddl_file_name=str(path), camera_heights=res, camera_widths=res)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest, specificity = strict_pre_cuda_validation(args, root)
    sources_start = source_hashes(
        root,
        (
            "athena/preflight_counterfactual_target_swap.py",
            "athena/run_counterfactual_target_swap.py",
        ),
    )
    started = time.perf_counter()

    import numpy as np
    import torch
    from PIL import Image
    from athena.run_xvla_experiment import (
        build_encoder,
        build_robot_state,
        build_vocab,
        load_cache_statistics,
        load_suite,
        make_config,
        official_init_states,
        scene_object_bodies,
        task_languages,
    )
    from xvla.models.vla import ChiVLA

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("Torch did not retain highest float32 matmul precision")
    if not torch.are_deterministic_algorithms_enabled() or torch.backends.cudnn.benchmark:
        raise RuntimeError("Frozen deterministic torch execution was not retained")
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name != PROTOCOL["gpu_name"]:
        raise RuntimeError(f"Frozen A6000 required, observed {gpu_name}")
    torch.manual_seed(args.checkpoint_seed)
    torch.cuda.manual_seed_all(args.checkpoint_seed)
    np.random.seed(args.checkpoint_seed)
    runtime_versions = {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "torch_build": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "pillow": importlib.metadata.version("Pillow"),
        "mujoco": importlib.metadata.version("mujoco"),
        "robosuite": importlib.metadata.version("robosuite"),
    }
    if runtime_versions != PROTOCOL["runtime_versions"]:
        raise RuntimeError(f"Frozen software runtime differs: {runtime_versions}")
    libero_runtime_start = validate_libero_runtime()

    suite = load_suite("libero_object")
    languages = task_languages(suite)
    if languages != PROMPTS:
        raise RuntimeError("Live Object task-language order differs")
    vocab, _ = build_vocab(languages)
    encode = build_encoder(vocab)
    encoded = {prompt: encode(text) for prompt, text in PROMPTS.items()}
    stats = load_cache_statistics(args.cache, args.horizon)
    if (int(stats["frame_count"]), int(stats["sample_count"])) != (CACHE["frames"], CACHE["samples"]):
        raise RuntimeError("Frozen cache statistics differ")
    config = make_config("chi", len(vocab), stats["state_dim"], stats["action_dim"], args.res, args.horizon, vision_encoder="vit")
    model = ChiVLA(config).cuda()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cuda", weights_only=True), strict=True)
    model.eval()

    mapping_lookup = {(int(row["task_index"]), int(row["counterfactual_prompt_id"])): row for row in manifest["mappings"]}
    controls = []
    pairs = []
    smoke_mappings = []
    smoke_original_inputs = []
    task_indices = range(args.task_start, args.task_end)
    for task_index in task_indices:
        task = suite.get_task(task_index)
        init_states = official_init_states(suite, task_index)
        if len(init_states) < 50:
            raise RuntimeError(f"Task {task_index} has fewer than 50 canonical states")
        task_rows = sorted(
            (row for key, row in mapping_lookup.items() if key[0] == task_index),
            key=lambda row: int(row["counterfactual_prompt_id"]),
        )
        if len(task_rows) != 5:
            raise RuntimeError(f"Task {task_index} does not have five manifest distractors")
        grocery_body_names = [str(value) for value in task_rows[0]["rewrite_validation"]["object_symbols"].values()]
        body_names = [*grocery_body_names, RECEPTACLE_SYMBOL]
        original_path = Path(task_rows[0]["original_bddl_path"])
        original_environment = make_environment(original_path, args.res)
        reference_starts = {}
        try:
            for episode in range(args.episode_start, args.episode_start + args.eps_per_task):
                reset_seed = task_index * 100 + episode
                observation = reset_environment(original_environment, init_states[episode], reset_seed, args.num_steps_wait, np)
                observed_bodies = scene_object_bodies(observation)
                if sorted(observed_bodies) != sorted(grocery_body_names):
                    raise RuntimeError(f"Task {task_index} observation bodies differ from manifest")
                start = physical_snapshot(original_environment, observation, stats, args.res, body_names, np, Image, build_robot_state)
                reference_starts[episode] = public_snapshot(start)
                if args.mode == "strict_smoke":
                    smoke_original_inputs.append({
                        "task_index": task_index,
                        "episode": episode,
                        "init_state_index": episode,
                        "reset_seed": reset_seed,
                        "init_state_sha256": hash_array(np.asarray(init_states[episode])),
                        "original_bddl_sha256": task_rows[0]["original_bddl_sha256"],
                        "physical_input": reference_starts[episode],
                    })
                if args.mode == "full":
                    control = rollout(original_environment, init_states[episode], reset_seed, encoded[task_index], task_index, model, stats, body_names, args, torch, np, Image, build_robot_state)
                    if control["start_physical"] != reference_starts[episode]:
                        raise RuntimeError("Original-goal control reset did not reproduce")
                    controls.append({
                        "task_index": task_index,
                        "episode": episode,
                        "init_state_index": episode,
                        "reset_seed": reset_seed,
                        "init_state_sha256": hash_array(np.asarray(init_states[episode])),
                        "original_bddl_sha256": task_rows[0]["original_bddl_sha256"],
                        "rollout": control,
                    })
        finally:
            original_environment.close()

        for mapping in task_rows:
            distractor = int(mapping["counterfactual_prompt_id"])
            counterfactual_environment = make_environment(Path(mapping["rewritten_bddl_path"]), args.res)
            try:
                for episode in range(args.episode_start, args.episode_start + args.eps_per_task):
                    reset_seed = task_index * 100 + episode
                    observation = reset_environment(counterfactual_environment, init_states[episode], reset_seed, args.num_steps_wait, np)
                    start = physical_snapshot(counterfactual_environment, observation, stats, args.res, body_names, np, Image, build_robot_state)
                    if public_snapshot(start) != reference_starts[episode]:
                        raise RuntimeError(f"Counterfactual BDDL changed physical input for task {task_index}, prompt {distractor}, episode {episode}")
                    if args.mode == "strict_smoke":
                        chunks = {}
                        for prompt_id in (distractor, task_index):
                            chunk = predict_chunk(model, start["model_image"], start["model_robot_state"], encoded[prompt_id], stats, torch, np)
                            chunks[str(prompt_id)] = encode_array(np.asarray(chunk, dtype=np.float32))
                        smoke_mappings.append({
                            "task_index": task_index,
                            "episode": episode,
                            "counterfactual_prompt_id": distractor,
                            "rewritten_bddl_sha256": mapping["rewritten_bddl_sha256"],
                            "physical_input": public_snapshot(start),
                            "physical_matches_original_control": True,
                            "finite_action_chunks": chunks,
                        })
                        continue
                    order = [distractor, task_index] if (task_index + distractor + episode + args.checkpoint_seed) % 2 == 0 else [task_index, distractor]
                    condition_results = {}
                    for prompt_id in order:
                        condition = "matching_counterfactual_prompt" if prompt_id == distractor else "original_prompt"
                        run = rollout(counterfactual_environment, init_states[episode], reset_seed, encoded[prompt_id], prompt_id, model, stats, body_names, args, torch, np, Image, build_robot_state)
                        if run["start_physical"] != reference_starts[episode]:
                            raise RuntimeError("Paired counterfactual condition did not reproduce physical input")
                        condition_results[condition] = run
                    pairs.append({
                        "task_index": task_index,
                        "episode": episode,
                        "init_state_index": episode,
                        "reset_seed": reset_seed,
                        "init_state_sha256": hash_array(np.asarray(init_states[episode])),
                        "original_prompt_id": task_index,
                        "counterfactual_prompt_id": distractor,
                        "rewritten_bddl_sha256": mapping["rewritten_bddl_sha256"],
                        "execution_order": ["matching_counterfactual_prompt" if prompt == distractor else "original_prompt" for prompt in order],
                        "physical_input_sha256": reference_starts[episode]["combined_sha256"],
                        "conditions": condition_results,
                    })
            finally:
                counterfactual_environment.close()

    if args.mode == "strict_smoke":
        if len(smoke_mappings) != 50 or len(smoke_original_inputs) != 10 or controls or pairs:
            raise RuntimeError("Strict smoke did not cover exactly all 50 mappings")
    else:
        expected_pairs = (args.task_end - args.task_start) * 10 * 5
        expected_controls = (args.task_end - args.task_start) * 10
        if len(pairs) != expected_pairs or len(controls) != expected_controls:
            raise RuntimeError("Full shard trial counts are incomplete")
        if 2 * len(pairs) + len(controls) != args.expected_rollouts:
            raise RuntimeError("Full shard rollout count is incomplete")

    sources_end = source_hashes(root, ("athena/preflight_counterfactual_target_swap.py", "athena/run_counterfactual_target_swap.py"))
    if sources_start != sources_end:
        raise RuntimeError("Repository sources changed while target-swap runner was active")
    libero_runtime_end = validate_libero_runtime()
    if libero_runtime_start != libero_runtime_end:
        raise RuntimeError("LIBERO sources changed while target-swap runner was active")
    output = {
        "schema": SCHEMA_RUN,
        "mode": args.mode,
        "protocol": PROTOCOL,
        "frozen_gates": FROZEN_GATES,
        "prerequisite_gate": {
            "specificity_summary_sha256": file_sha256(args.specificity_summary),
            "identity_validated": specificity["identity_validated"],
            "overall_pass": specificity["overall_pass"],
            "claim_eligible": specificity["claim_eligible"],
        },
        "identity": {
            "checkpoint_seed": args.checkpoint_seed,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "cache": str(args.cache),
            "cache_sha256": file_sha256(args.cache),
            "provenance_result": str(args.provenance_result),
            "provenance_result_sha256": file_sha256(args.provenance_result),
            "provenance_job_id": PROVENANCE["job_id"],
            "manifest": str(args.manifest),
            "manifest_sha256": file_sha256(args.manifest),
            "source_sha256_start": sources_start,
            "source_sha256_end": sources_end,
        },
        "evaluation": {
            "task_start": args.task_start,
            "task_end": args.task_end,
            "episode_start": args.episode_start,
            "eps_per_task": args.eps_per_task,
            "expected_rollouts": args.expected_rollouts,
            "completed_rollouts": 2 * len(pairs) + len(controls),
            "completed_pairs": len(pairs),
            "completed_controls": len(controls),
            "smoke_mapping_count": len(smoke_mappings),
            "matmul_precision": torch.get_float32_matmul_precision(),
        },
        "runtime": {
            **runtime_versions,
            "libero_source_start": libero_runtime_start,
            "libero_source_end": libero_runtime_end,
            "gpu": gpu_name,
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "environment": {name: os.environ.get(name) for name in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "XVLA_FROZEN_EVALUATION_GPU_FAMILY")},
        },
        "smoke_mappings": smoke_mappings,
        "smoke_original_inputs": smoke_original_inputs,
        "original_goal_controls": controls,
        "paired_counterfactual_trials": pairs,
        "elapsed_s": time.perf_counter() - started,
        "claim_boundary": "This familiar-vocabulary, familiar-scene synthetic BDDL goal swap can test paired closed-loop instruction dependence only. Canonical episodes 40 through 49 were included in the broader prerequisite local-specificity state set, so this is a new outcome and intervention on overlapping states, not an independent-state replication. It cannot establish paraphrase robustness, unseen-object or broad compositional grounding, cross-suite or physical-robot generalization, or a benefit unique to tensor decomposability.",
    }
    write_json(args.output, output)
    print("RESULT", json.dumps({"mode": args.mode, "pairs": len(pairs), "controls": len(controls), "smoke_mappings": len(smoke_mappings)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
