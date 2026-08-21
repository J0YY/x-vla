#!/usr/bin/env python3
"""Run the fresh prerequisite-gated 2x2 counterfactual goal-following endpoint."""

from __future__ import annotations

import argparse
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

from athena.counterfactual_goal_following_common import (
    CACHE,
    CHECKPOINTS,
    CONDITIONS,
    DESIGN_PROVENANCE,
    FROZEN_GATES,
    INSTRUCTION_SUMMARY_PATH,
    MANIFEST_PATH,
    PROMPTS,
    PROTOCOL,
    PROVENANCE,
    RECEPTACLE_SYMBOL,
    SCHEMA_RUN,
    SMOKE_RESULT_PATH,
    TASK_SHARDS,
    condition_order,
    encode_array,
    file_sha256,
    hash_array,
    source_hashes,
    validate_instruction_summary,
    validate_libero_runtime,
    validate_manifest,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("strict_smoke", "full"), required=True)
    parser.add_argument("--checkpoint-seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--provenance-result", type=Path, required=True)
    parser.add_argument("--instruction-summary", type=Path, required=True)
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
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.checkpoint_seed not in CHECKPOINTS:
        raise ValueError("Checkpoint seed is not frozen")
    checkpoint = CHECKPOINTS[args.checkpoint_seed]
    expected_paths = {
        "checkpoint": root / checkpoint["path"],
        "cache": root / CACHE["path"],
        "provenance": root / PROVENANCE["path"],
        "instruction summary": root / INSTRUCTION_SUMMARY_PATH,
        "manifest": root / MANIFEST_PATH,
    }
    observed_paths = {
        "checkpoint": args.checkpoint,
        "cache": args.cache,
        "provenance": args.provenance_result,
        "instruction summary": args.instruction_summary,
        "manifest": args.manifest,
    }
    for label, path in observed_paths.items():
        if path.resolve() != expected_paths[label].resolve():
            raise RuntimeError(f"{label} argument is not the frozen path")
    if file_sha256(args.checkpoint) != checkpoint["sha256"] or file_sha256(args.cache) != CACHE["sha256"]:
        raise RuntimeError("Checkpoint or cache identity differs")
    instruction = validate_instruction_summary(args.instruction_summary, root)
    manifest = validate_manifest(args.manifest, root)
    if manifest["instruction_summary_sha256"] != file_sha256(args.instruction_summary):
        raise RuntimeError("Manifest used a different instruction-necessity summary")
    observed_scalars = (
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
    if observed_scalars != expected_scalars:
        raise RuntimeError("Rollout arguments differ from the frozen protocol")
    if args.mode == "strict_smoke":
        observed = (
            args.checkpoint_seed,
            args.task_start,
            args.task_end,
            args.episode_start,
            args.eps_per_task,
            args.expected_rollouts,
        )
        if observed != (0, 0, 10, 20, 1, 0):
            raise RuntimeError("Strict smoke matrix differs")
        if args.output.resolve() != (root / SMOKE_RESULT_PATH).resolve():
            raise RuntimeError("Smoke output path differs")
    else:
        if (args.task_start, args.task_end) not in TASK_SHARDS:
            raise RuntimeError("Full task shard differs")
        if (args.episode_start, args.eps_per_task, args.expected_rollouts) != (20, 10, 400):
            raise RuntimeError("Full state or rollout count differs")
        expected_output = root / (
            f"results/counterfactual_goal_following_v1_s{args.checkpoint_seed}_"
            f"t{args.task_start}_{args.task_end}.json"
        )
        if args.output.resolve() != expected_output.resolve():
            raise RuntimeError("Full output path differs")
    required_environment = {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "CUBLAS_WORKSPACE_CONFIG": PROTOCOL["cublas_workspace_config"],
        "XVLA_FROZEN_EVALUATION_GPU_FAMILY": PROTOCOL["gpu_family"],
    }
    for name, expected in required_environment.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"{name} must equal {expected}")
    expected_common = os.environ.get("XVLA_GOAL_FOLLOWING_COMMON_SHA256")
    expected_runner = os.environ.get("XVLA_GOAL_FOLLOWING_RUNNER_SHA256")
    if expected_common != file_sha256(Path(__file__).with_name("counterfactual_goal_following_common.py")):
        raise RuntimeError("Common source differs from wrapper declaration")
    if expected_runner != file_sha256(Path(__file__).resolve()):
        raise RuntimeError("Runner source differs from wrapper declaration")
    return manifest, instruction


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest, instruction = strict_pre_cuda_validation(args, root)
    sources_start = source_hashes(root)
    runtime_start = validate_libero_runtime()
    started = time.perf_counter()

    import numpy as np
    import torch
    from PIL import Image
    from athena.run_counterfactual_target_swap import (
        make_environment,
        physical_snapshot,
        predict_chunk,
        public_snapshot,
        reset_environment,
        rollout,
    )
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
    if torch.get_float32_matmul_precision() != "highest" or not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Frozen deterministic torch execution was not retained")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_capability = list(torch.cuda.get_device_capability(0))
    if gpu_name != PROTOCOL["gpu_name"] or gpu_capability != PROTOCOL["gpu_compute_capability"]:
        raise RuntimeError("Frozen A6000 runtime is required")
    torch.manual_seed(args.checkpoint_seed)
    torch.cuda.manual_seed_all(args.checkpoint_seed)
    np.random.seed(args.checkpoint_seed)
    versions = {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "torch_build": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "pillow": importlib.metadata.version("Pillow"),
        "mujoco": importlib.metadata.version("mujoco"),
        "robosuite": importlib.metadata.version("robosuite"),
    }
    if versions != PROTOCOL["runtime_versions"]:
        raise RuntimeError(f"Frozen software runtime differs: {versions}")

    suite = load_suite("libero_object")
    languages = task_languages(suite)
    if languages != PROMPTS:
        raise RuntimeError("Live Object prompt catalog differs")
    vocab, _ = build_vocab(languages)
    encode = build_encoder(vocab)
    encoded = {prompt_id: encode(text) for prompt_id, text in languages.items()}
    stats = load_cache_statistics(args.cache, args.horizon)
    if (int(stats["frame_count"]), int(stats["sample_count"])) != (
        CACHE["frames"],
        CACHE["samples"],
    ):
        raise RuntimeError("Cache statistics differ")
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
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cuda", weights_only=True), strict=True
    )
    model.eval()

    by_task: dict[int, list[dict[str, Any]]] = {task: [] for task in range(10)}
    for mapping in manifest["mappings"]:
        by_task[int(mapping["task_index"])].append(mapping)
    quadruplets: list[dict[str, Any]] = []
    smoke_rows: list[dict[str, Any]] = []
    for task_index in range(args.task_start, args.task_end):
        task = suite.get_task(task_index)
        mappings = sorted(by_task[task_index], key=lambda row: int(row["counterfactual_prompt_id"]))
        if len(mappings) != 5:
            raise RuntimeError("Task manifest does not contain five targets")
        object_symbols = mappings[0]["rewrite_validation"]["object_symbols"]
        body_names = [str(value) for value in object_symbols.values()]
        extended_bodies = [*body_names, RECEPTACLE_SYMBOL]
        original_environment = make_environment(Path(mappings[0]["original_bddl_path"]), args.res)
        init_states = official_init_states(suite, task_index)
        references: dict[int, dict[str, Any]] = {}
        try:
            for episode in range(args.episode_start, args.episode_start + args.eps_per_task):
                reset_seed = task_index * 100 + episode
                observation = reset_environment(
                    original_environment,
                    init_states[episode],
                    reset_seed,
                    args.num_steps_wait,
                    np,
                )
                if sorted(scene_object_bodies(observation)) != sorted(body_names):
                    raise RuntimeError("Live object identities differ from the manifest")
                references[episode] = physical_snapshot(
                    original_environment,
                    observation,
                    stats,
                    args.res,
                    extended_bodies,
                    np,
                    Image,
                    build_robot_state,
                )

            for mapping in mappings:
                target = int(mapping["counterfactual_prompt_id"])
                counterfactual_environment = make_environment(
                    Path(mapping["rewritten_bddl_path"]), args.res
                )
                try:
                    for episode in range(args.episode_start, args.episode_start + args.eps_per_task):
                        reset_seed = task_index * 100 + episode
                        observation = reset_environment(
                            counterfactual_environment,
                            init_states[episode],
                            reset_seed,
                            args.num_steps_wait,
                            np,
                        )
                        counterfactual_start = physical_snapshot(
                            counterfactual_environment,
                            observation,
                            stats,
                            args.res,
                            extended_bodies,
                            np,
                            Image,
                            build_robot_state,
                        )
                        reference = references[episode]
                        if public_snapshot(counterfactual_start) != public_snapshot(reference):
                            raise RuntimeError("Goal rewrite changed the paired physical input")
                        specs = {
                            "original_goal_original_prompt": {
                                "environment": original_environment,
                                "goal_prompt_id": task_index,
                                "prompt_id": task_index,
                            },
                            "original_goal_counterfactual_prompt": {
                                "environment": original_environment,
                                "goal_prompt_id": task_index,
                                "prompt_id": target,
                            },
                            "counterfactual_goal_counterfactual_prompt": {
                                "environment": counterfactual_environment,
                                "goal_prompt_id": target,
                                "prompt_id": target,
                            },
                            "counterfactual_goal_original_prompt": {
                                "environment": counterfactual_environment,
                                "goal_prompt_id": target,
                                "prompt_id": task_index,
                            },
                        }
                        order = condition_order(
                            args.checkpoint_seed, task_index, episode, target
                        )
                        if args.mode == "strict_smoke":
                            conditions = {}
                            for condition in order:
                                spec = specs[condition]
                                chunk = predict_chunk(
                                    model,
                                    reference["model_image"],
                                    reference["model_robot_state"],
                                    encoded[spec["prompt_id"]],
                                    stats,
                                    torch,
                                    np,
                                )
                                conditions[condition] = {
                                    "goal_prompt_id": spec["goal_prompt_id"],
                                    "prompt_id": spec["prompt_id"],
                                    "instruction_ids": encoded[spec["prompt_id"]],
                                    "action_chunk": encode_array(
                                        np.asarray(chunk, dtype=np.float32)
                                    ),
                                }
                            smoke_rows.append(
                                {
                                    "task_index": task_index,
                                    "episode": episode,
                                    "counterfactual_prompt_id": target,
                                    "physical_input": public_snapshot(reference),
                                    "condition_order": order,
                                    "conditions": conditions,
                                }
                            )
                            continue

                        conditions = {}
                        for condition in order:
                            spec = specs[condition]
                            result = rollout(
                                spec["environment"],
                                init_states[episode],
                                reset_seed,
                                encoded[spec["prompt_id"]],
                                spec["prompt_id"],
                                model,
                                stats,
                                extended_bodies,
                                args,
                                torch,
                                np,
                                Image,
                                build_robot_state,
                            )
                            if result["start_physical"] != public_snapshot(reference):
                                raise RuntimeError("A factorial arm changed its physical input")
                            conditions[condition] = {
                                "goal_prompt_id": spec["goal_prompt_id"],
                                "prompt_id": spec["prompt_id"],
                                "instruction_ids": encoded[spec["prompt_id"]],
                                "rollout": result,
                            }
                        quadruplets.append(
                            {
                                "task_index": task_index,
                                "episode": episode,
                                "init_state_index": episode,
                                "reset_seed": reset_seed,
                                "init_state_sha256": hash_array(np.asarray(init_states[episode])),
                                "original_prompt_id": task_index,
                                "counterfactual_prompt_id": target,
                                "original_bddl_sha256": mapping["original_bddl_sha256"],
                                "rewritten_bddl_sha256": mapping["rewritten_bddl_sha256"],
                                "physical_input_sha256": reference["combined_sha256"],
                                "condition_order": order,
                                "conditions": conditions,
                            }
                        )
                finally:
                    counterfactual_environment.close()
        finally:
            original_environment.close()

    if args.mode == "strict_smoke":
        if len(smoke_rows) != 50 or quadruplets:
            raise RuntimeError("Strict smoke must contain fifty mappings")
        completed_rollouts = 0
    else:
        if len(quadruplets) != 100 or smoke_rows:
            raise RuntimeError("Full shard must contain one hundred four-arm units")
        completed_rollouts = 4 * len(quadruplets)
        if completed_rollouts != args.expected_rollouts:
            raise RuntimeError("Full rollout count differs")

    sources_end = source_hashes(root)
    runtime_end = validate_libero_runtime()
    if sources_start != sources_end or runtime_start != runtime_end:
        raise RuntimeError("Sources changed during evaluation")
    output = {
        "schema": SCHEMA_RUN,
        "mode": args.mode,
        "protocol": PROTOCOL,
        "frozen_gates": FROZEN_GATES,
        "design_provenance": DESIGN_PROVENANCE,
        "identity": {
            "checkpoint_seed": args.checkpoint_seed,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "cache": str(args.cache),
            "cache_sha256": file_sha256(args.cache),
            "provenance_result": str(args.provenance_result),
            "provenance_result_sha256": file_sha256(args.provenance_result),
            "provenance_job_id": PROVENANCE["job_id"],
            "instruction_summary": str(args.instruction_summary),
            "instruction_summary_sha256": file_sha256(args.instruction_summary),
            "instruction_summary_gate_validated": instruction["overall_pass"],
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
            "completed_rollouts": completed_rollouts,
            "completed_four_arm_units": len(quadruplets),
            "smoke_units": len(smoke_rows),
            "matmul_precision": torch.get_float32_matmul_precision(),
        },
        "runtime": {
            **versions,
            "gpu": gpu_name,
            "gpu_capability": gpu_capability,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "libero_source_start": runtime_start,
            "libero_source_end": runtime_end,
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "MUJOCO_GL",
                    "PYOPENGL_PLATFORM",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "CUDA_VISIBLE_DEVICES",
                    "SLURM_JOB_ID",
                    "XVLA_FROZEN_EVALUATION_GPU_FAMILY",
                )
            },
        },
        "smoke_units": smoke_rows,
        "four_arm_units": quadruplets,
        "elapsed_s": time.perf_counter() - started,
        "claim_boundary": (
            "A pass supports counterfactual goal following only for familiar Object-suite "
            "vocabulary, objects, scenes, synthetic target-only goal rewrites, and the three "
            "fixed checkpoints. It does not establish paraphrase, policy-unseen, cross-suite, "
            "or physical-robot generalization, or a benefit unique to decomposability."
        ),
    }
    write_json(args.output, output)
    print(
        "RESULT",
        json.dumps(
            {
                "mode": args.mode,
                "four_arm_units": len(quadruplets),
                "rollouts": completed_rollouts,
                "smoke_units": len(smoke_rows),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
