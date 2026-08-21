#!/usr/bin/env python3
"""Native Athena evaluation and profiling for the χ-VLA LIBERO checkpoints.

This runner deliberately has no Modal dependency. It reconstructs the exact model,
vocabulary, normalization statistics, observation transform, and canonical LIBERO
reset protocol used by ``modal_app.libero_rollout_head``.
"""

from __future__ import annotations

import argparse
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

from xvla.models.vla import ChiVLA, VLAConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "smoke",
            "profile",
            "capability",
            "offline_diagnostic",
            "causal",
            "visual_subspace",
            "exact_attention",
            "surgery",
        ),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        choices=("chi", "conventional", "chi_rms", "conventional_rational"),
        default="chi",
    )
    parser.add_argument(
        "--suite",
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
    parser.add_argument("--causal-steps", type=int, default=80)
    parser.add_argument("--block-index", type=int, default=6)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--gram-samples", type=int, default=1024)
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

    def encode(text: str, length: int = 32) -> list[int]:
        ids = [1] + [vocab.get(word, 0) for word in text.lower().replace(".", "").split()]
        return (ids[:length] + [0] * max(0, length - len(ids)))[:length]

    return vocab, encode


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
        "vision_encoder": "vit",
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


def load_model(args: argparse.Namespace, vocab_size: int, stats: dict[str, Any]) -> ChiVLA:
    config = make_config(
        args.architecture,
        vocab_size,
        stats["state_dim"],
        stats["action_dim"],
        args.res,
        args.horizon,
    )
    model = ChiVLA(config).cuda()
    state_dict = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


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
    model: ChiVLA,
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
    model: ChiVLA,
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
    model: ChiVLA,
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
    encode,
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Replicate the visual-bond causal intervention for one trained checkpoint."""
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
                )
            )
    if not samples:
        raise RuntimeError("No visual-bond samples could be built from the cache")
    if not 0 < args.rank < model.cfg.dim:
        raise ValueError(f"rank must be between 1 and {model.cfg.dim - 1}")

    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(
        len(samples), size=min(args.gram_samples, len(samples)), replace=False
    )
    images = (
        torch.from_numpy(np.stack([samples[index][0] for index in sample_indices]))
        .permute(0, 3, 1, 2)
        .float()
        .div(255)
        .cuda()
    )
    instructions = torch.tensor(
        [encode(tasks[samples[index][1]]) for index in sample_indices],
        dtype=torch.long,
        device="cuda",
    )
    states_array = np.stack([samples[index][2] for index in sample_indices])
    states = torch.tensor(
        (states_array - stats["state_mean"]) / stats["state_std"],
        dtype=torch.float32,
        device="cuda",
    )
    embodiments = torch.zeros(len(sample_indices), dtype=torch.long, device="cuda")

    gram = torch.zeros(model.cfg.dim, model.cfg.dim, dtype=torch.float64, device="cuda")
    gradient_rows = 0
    gram_batch_size = 128
    for start in range(0, len(sample_indices), gram_batch_size):
        stop = min(start + gram_batch_size, len(sample_indices))
        with torch.no_grad():
            visual = model._visual_tokens(images[start:stop])
        visual = visual.detach().requires_grad_(True)
        with torch.enable_grad():
            prediction = forward_from_visual_tokens(
                model,
                visual,
                instructions[start:stop],
                states[start:stop],
                embodiments[start:stop],
            )
            selected = prediction[:, :, -1].sum()
            gradient = torch.autograd.grad(selected, visual)[0]
        flat_gradient = gradient.reshape(-1, model.cfg.dim).double()
        gram += flat_gradient.T @ flat_gradient
        gradient_rows += flat_gradient.shape[0]
    gram /= max(gradient_rows, 1)

    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.flip(0)
    top_basis = eigenvectors.flip(1)[:, : args.rank]
    top_projector = (top_basis @ top_basis.T).float()
    torch_rng = torch.Generator(device="cuda").manual_seed(args.seed + 4109)
    random_matrix = torch.randn(
        model.cfg.dim,
        args.rank,
        generator=torch_rng,
        dtype=torch.float64,
        device="cuda",
    )
    random_basis, _ = torch.linalg.qr(random_matrix)
    random_projector = (random_basis @ random_basis.T).float()

    offline_predictions: dict[str, list[torch.Tensor]] = {
        "full": [],
        "causal_topk": [],
        "random_topk": [],
    }
    with torch.inference_mode():
        for start in range(0, len(sample_indices), gram_batch_size):
            stop = min(start + gram_batch_size, len(sample_indices))
            visual = model._visual_tokens(images[start:stop])
            for condition, projector in (
                ("full", None),
                ("causal_topk", top_projector),
                ("random_topk", random_projector),
            ):
                intervened = visual if projector is None else visual @ projector.T
                offline_predictions[condition].append(
                    forward_from_visual_tokens(
                        model,
                        intervened,
                        instructions[start:stop],
                        states[start:stop],
                        embodiments[start:stop],
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
    for condition in ("causal_topk", "random_topk"):
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

    by_condition = {}
    for condition, projector in (
        ("full", None),
        ("causal_topk", top_projector),
        ("random_topk", random_projector),
    ):
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
        "gram_samples": len(sample_indices),
        "gradient_rows": gradient_rows,
        "gram_action_group": "gripper",
        "top_eigenvalues": eigenvalues[:16].detach().cpu().tolist(),
        "top_rank_spectral_mass": float(
            eigenvalues[: args.rank].clamp_min(0).sum()
            / eigenvalues.clamp_min(0).sum().clamp_min(1e-30)
        ),
        "offline_mse_to_full": offline_mse,
        "offline_random_to_causal_mse_ratio": {
            group: offline_mse["random_topk"][group]
            / max(offline_mse["causal_topk"][group], 1e-30)
            for group in action_groups
        },
        "overall_by_condition": {
            condition: float(result["overall"])
            for condition, result in by_condition.items()
        },
        "by_condition": by_condition,
        "discovery_scope": (
            "The gripper-sensitive visual-bond subspace is estimated from cached held-out "
            "activations for this checkpoint. It is a data-driven causal bottleneck, not a "
            "weight-only whole-policy decomposition."
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("This Athena runner requires a CUDA compute node")

    print(f"Loading cache statistics from {args.cache}", flush=True)
    stats = load_cache_statistics(args.cache, args.horizon)
    suite = load_suite(args.suite)
    tasks = task_languages(suite)
    training_tasks = task_languages(load_suite("libero_object"))
    vocab, encode = build_vocab(training_tasks)
    print(f"Tasks={len(tasks)} vocab={len(vocab)} samples={stats['sample_count']}", flush=True)
    model = load_model(args, len(vocab), stats)
    sample_tensors = tensorize_sample(stats["first_sample"], encode, training_tasks, stats)
    with torch.inference_mode():
        prediction, _ = model(*sample_tensors)
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Checkpoint produced a non-finite prediction")

    profile = profile_model(model, sample_tensors, args.profile_iters)
    result: dict[str, Any] = {
        "mode": args.mode,
        "architecture": args.architecture,
        "suite": args.suite,
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "seed": args.seed,
        "vocab_size": len(vocab),
        "cache_stats": stats,
        "profile": profile,
        "prediction_shape": list(prediction.shape),
        "prediction_finite": True,
        "prediction_sample": prediction[0].detach().float().cpu().tolist(),
        "evaluation_scope": (
            "In-domain LIBERO-Object evaluation"
            if args.suite == "libero_object"
            else (
                "Zero-shot cross-suite evaluation of a LIBERO-Object-trained checkpoint. "
                "Vocabulary and normalization remain fixed to LIBERO-Object, with unseen "
                "instruction words mapped to the padding identifier."
            )
        ),
    }

    if args.mode == "smoke":
        original_end = args.task_end
        original_eps = args.eps_per_task
        original_steps = args.max_steps
        args.task_end = min(args.task_start + 1, original_end)
        args.eps_per_task = min(1, original_eps)
        args.max_steps = min(8, original_steps)
        result["capability"] = run_capability(args, model, suite, tasks, encode, stats)
    elif args.mode == "capability":
        result["capability"] = run_capability(args, model, suite, tasks, encode, stats)
    elif args.mode == "offline_diagnostic":
        if args.suite != "libero_object":
            raise ValueError("offline_diagnostic currently requires suite=libero_object")
        result["offline_diagnostic"] = run_offline_checkpoint_diagnostic(
            args, model, tasks, encode, stats
        )
    elif args.mode == "causal":
        result["causal"] = run_causal_intervention(args, model, suite, tasks, encode, stats)
    elif args.mode == "visual_subspace":
        if args.architecture != "chi":
            raise ValueError("visual_subspace requires architecture=chi")
        if args.suite != "libero_object":
            raise ValueError("visual_subspace currently requires suite=libero_object")
        result["visual_subspace"] = run_visual_subspace_intervention(
            args, model, suite, tasks, encode, stats
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(result), indent=2) + "\n")
    temporary.replace(args.output)
    print("RESULT", json.dumps(json_ready(result), indent=2), flush=True)


if __name__ == "__main__":
    main()
