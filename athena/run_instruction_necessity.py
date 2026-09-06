#!/usr/bin/env python3
"""Run the prospectively frozen closed-loop instruction-necessity endpoint."""

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

from athena.instruction_necessity_common import (
    CACHE,
    CHECKPOINTS,
    CONDITIONS,
    DESIGN_PROVENANCE,
    FROZEN_GATES,
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


def strict_pre_cuda_validation(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.checkpoint_seed not in CHECKPOINTS:
        raise ValueError("Checkpoint seed is not frozen")
    expected_checkpoint = CHECKPOINTS[args.checkpoint_seed]
    expected_paths = {
        "checkpoint": root / expected_checkpoint["path"],
        "cache": root / CACHE["path"],
        "provenance": root / PROVENANCE["path"],
        "manifest": root / MANIFEST_PATH,
    }
    observed_paths = {
        "checkpoint": args.checkpoint,
        "cache": args.cache,
        "provenance": args.provenance_result,
        "manifest": args.manifest,
    }
    for label, path in observed_paths.items():
        if path.resolve() != expected_paths[label].resolve():
            raise RuntimeError(f"{label} argument is not the frozen path")
    if file_sha256(args.checkpoint) != expected_checkpoint["sha256"]:
        raise RuntimeError("Checkpoint SHA differs")
    if file_sha256(args.cache) != CACHE["sha256"]:
        raise RuntimeError("Cache SHA differs")
    manifest = validate_manifest(args.manifest, root)
    if manifest["cache"]["live_sha256"] != file_sha256(args.cache):
        raise RuntimeError("Manifest cache identity differs")
    if manifest["provenance"]["live_sha256"] != file_sha256(args.provenance_result):
        raise RuntimeError("Manifest provenance identity differs")
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
        raise ValueError("Rollout arguments differ from the frozen protocol")
    if args.mode == "strict_smoke":
        expected = (0, 0, 10, 40, 1, 0)
        observed = (
            args.checkpoint_seed,
            args.task_start,
            args.task_end,
            args.episode_start,
            args.eps_per_task,
            args.expected_rollouts,
        )
        if observed != expected:
            raise ValueError("Strict smoke must cover episode 40 of all ten tasks at seed 0")
        if args.output.resolve() != (root / SMOKE_RESULT_PATH).resolve():
            raise RuntimeError("Smoke output path differs")
    else:
        if (args.task_start, args.task_end) not in TASK_SHARDS:
            raise ValueError("Full task bounds are not a frozen shard")
        if (args.episode_start, args.eps_per_task, args.expected_rollouts) != (40, 10, 60):
            raise ValueError("Full shard must contain 20 states and 60 rollouts")
        expected_output = root / (
            f"results/instruction_necessity_v1_s{args.checkpoint_seed}_"
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
    expected_common = os.environ.get("XVLA_INSTRUCTION_NECESSITY_COMMON_SHA256")
    expected_runner = os.environ.get("XVLA_INSTRUCTION_NECESSITY_RUNNER_SHA256")
    if expected_common != file_sha256(Path(__file__).with_name("instruction_necessity_common.py")):
        raise RuntimeError("Common source differs from the wrapper declaration")
    if expected_runner != file_sha256(Path(__file__).resolve()):
        raise RuntimeError("Runner source differs from the wrapper declaration")
    return manifest


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = strict_pre_cuda_validation(args, root)
    sources_start = source_hashes(root)
    runtime_start = validate_libero_runtime()
    started = time.perf_counter()

    import numpy as np
    import torch
    from PIL import Image
    from libero.libero import get_libero_path
    from athena.run_counterfactual_target_swap import (
        physical_snapshot,
        predict_chunk,
        public_snapshot,
        reset_environment,
        rollout,
    )
    from athena.run_local_instruction_specificity import physical_input_hash
    from athena.run_xvla_experiment import (
        build_encoder,
        build_robot_state,
        build_vocab,
        load_cache_statistics,
        load_suite,
        make_config,
        official_init_states,
        task_languages,
    )
    from libero.libero.envs import OffScreenRenderEnv
    from xvla.models.vla import ChiVLA

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("Torch did not retain highest precision")
    if not torch.are_deterministic_algorithms_enabled() or torch.backends.cudnn.benchmark:
        raise RuntimeError("Deterministic execution settings differ")
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name != PROTOCOL["gpu_name"] or list(torch.cuda.get_device_capability(0)) != PROTOCOL["gpu_compute_capability"]:
        raise RuntimeError("The frozen A6000 runtime is required")
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
        raise RuntimeError(f"Software runtime differs: {versions}")

    suite = load_suite("libero_object")
    languages = task_languages(suite)
    if languages != PROMPTS:
        raise RuntimeError("Live Object prompt catalog differs")
    vocab, _ = build_vocab(languages)
    encode = build_encoder(vocab)
    encoded_prompts: dict[str, list[int]] = {
        "empty_instruction": encode(""),
        **{str(prompt_id): encode(text) for prompt_id, text in languages.items()},
    }
    if encoded_prompts["empty_instruction"] != [1] + [0] * 31:
        raise RuntimeError("Empty-instruction token semantics changed")
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

    mapping_lookup = {
        (int(row["task_index"]), int(row["episode"])): row
        for row in manifest["mappings"]
    }
    rows: list[dict[str, Any]] = []
    smoke_rows: list[dict[str, Any]] = []
    for task_index in range(args.task_start, args.task_end):
        task = suite.get_task(task_index)
        bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        environment = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=args.res,
            camera_widths=args.res,
        )
        init_states = official_init_states(suite, task_index)
        try:
            for episode in range(args.episode_start, args.episode_start + args.eps_per_task):
                mapping = mapping_lookup[(task_index, episode)]
                reset_seed = task_index * 100 + episode
                observation = reset_environment(
                    environment,
                    init_states[episode],
                    reset_seed,
                    args.num_steps_wait,
                    np,
                )
                body_names = list(mapping["eligible_bodies"])
                input_sha, components, image, raw_state = physical_input_hash(
                    environment, observation, stats, args.res, body_names
                )
                if input_sha != mapping["settled_physical_input_sha256"] or components != mapping["settled_component_sha256"]:
                    raise RuntimeError("Settled physical input differs from the manifest")
                extended_bodies = [*body_names, RECEPTACLE_SYMBOL]
                reference = public_snapshot(
                    physical_snapshot(
                        environment,
                        observation,
                        stats,
                        args.res,
                        extended_bodies,
                        np,
                        Image,
                        build_robot_state,
                    )
                )
                distractor_id = int(mapping["selected_distractor_prompt_id"])
                condition_specs = {
                    "correct_prompt": {
                        "prompt_id": task_index,
                        "prompt_text": languages[task_index],
                        "instruction_ids": encoded_prompts[str(task_index)],
                    },
                    "copresent_distractor_prompt": {
                        "prompt_id": distractor_id,
                        "prompt_text": languages[distractor_id],
                        "instruction_ids": encoded_prompts[str(distractor_id)],
                    },
                    "empty_instruction": {
                        "prompt_id": "empty_instruction",
                        "prompt_text": "",
                        "instruction_ids": encoded_prompts["empty_instruction"],
                    },
                }
                order = condition_order(args.checkpoint_seed, task_index, episode)
                if args.mode == "strict_smoke":
                    chunks = {}
                    for condition in order:
                        spec = condition_specs[condition]
                        chunk = predict_chunk(
                            model,
                            image,
                            raw_state,
                            spec["instruction_ids"],
                            stats,
                            torch,
                            np,
                        )
                        chunks[condition] = {
                            **spec,
                            "action_chunk": encode_array(np.asarray(chunk, dtype=np.float32)),
                        }
                    smoke_rows.append(
                        {
                            "task_index": task_index,
                            "episode": episode,
                            "physical_input_sha256": input_sha,
                            "condition_order": order,
                            "conditions": chunks,
                        }
                    )
                    continue

                conditions: dict[str, dict[str, Any]] = {}
                for condition in order:
                    spec = condition_specs[condition]
                    result = rollout(
                        environment,
                        init_states[episode],
                        reset_seed,
                        spec["instruction_ids"],
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
                    if result["start_physical"] != reference:
                        raise RuntimeError("A paired condition changed its physical input")
                    conditions[condition] = {**spec, "rollout": result}
                rows.append(
                    {
                        "task_index": task_index,
                        "episode": episode,
                        "init_state_index": episode,
                        "reset_seed": reset_seed,
                        "init_state_sha256": hash_array(np.asarray(init_states[episode])),
                        "original_bddl_sha256": mapping["original_bddl_sha256"],
                        "selected_distractor_prompt_id": distractor_id,
                        "physical_input_sha256": input_sha,
                        "condition_order": order,
                        "conditions": conditions,
                    }
                )
        finally:
            environment.close()

    if args.mode == "strict_smoke":
        if len(smoke_rows) != 10 or rows:
            raise RuntimeError("Strict smoke does not contain exactly ten state rows")
        completed_rollouts = 0
    else:
        if len(rows) != 20 or smoke_rows:
            raise RuntimeError(
                "Full shard does not contain exactly twenty checkpoint-state rows"
            )
        completed_rollouts = 3 * len(rows)
        if completed_rollouts != args.expected_rollouts:
            raise RuntimeError("Full rollout count differs")

    sources_end = source_hashes(root)
    runtime_end = validate_libero_runtime()
    if sources_start != sources_end or runtime_start != runtime_end:
        raise RuntimeError("Source identity changed during evaluation")
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
            "completed_checkpoint_state_pairs": len(rows),
            "smoke_states": len(smoke_rows),
            "matmul_precision": torch.get_float32_matmul_precision(),
        },
        "runtime": {
            **versions,
            "gpu": gpu_name,
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
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
        "smoke_rows": smoke_rows,
        "checkpoint_state_pairs": rows,
        "elapsed_s": time.perf_counter() - started,
        "claim_boundary": (
            "This tests instruction necessity for familiar Object-suite tasks, prompts, "
            "objects, scenes, and checkpoints. It does not test paraphrases, unseen objects, "
            "new task compositions, cross-suite or physical transfer, or a benefit unique to "
            "tensor decomposability. Episodes 40 through 49 overlap the failed local endpoint's "
            "state set, but the 280-step original-goal success outcome is new. This study was "
            "designed after that different local endpoint failed and is not its reanalysis. "
            "The co-present object is verified from simulator observation fields, not guaranteed "
            "visually unoccluded. Empty instruction means the empty-string tokenizer output, "
            "BOS plus 31 unmasked PAD tokens, together with the model's common learned BOS token."
        ),
    }
    write_json(args.output, output)
    print(
        "RESULT",
        json.dumps(
            {
                "mode": args.mode,
                "checkpoint_state_pairs": len(rows),
                "rollouts": completed_rollouts,
                "smoke_states": len(smoke_rows),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
