#!/usr/bin/env python3
"""Create the outcome-independent instruction-necessity manifest."""

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

from athena.instruction_necessity_common import (
    CACHE,
    CHECKPOINTS,
    DESIGN_PROVENANCE,
    FROZEN_GATES,
    LOCAL_SPECIFICITY_SUMMARY_PATH,
    MANIFEST_PATH,
    ORIGINAL_BDDL,
    PROMPTS,
    PROTOCOL,
    PROVENANCE,
    SCHEMA_MANIFEST,
    TARGETS,
    file_sha256,
    hash_array,
    prompt_target_to_id,
    select_visible_distractor,
    source_hashes,
    validate_failed_local_summary,
    validate_libero_runtime,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--provenance-result", type=Path, required=True)
    parser.add_argument("--failed-local-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def runtime_versions() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "torch_build": __import__("torch").__version__,
        "cuda": __import__("torch").version.cuda,
        "numpy": importlib.metadata.version("numpy"),
        "pillow": importlib.metadata.version("Pillow"),
        "mujoco": importlib.metadata.version("mujoco"),
        "robosuite": importlib.metadata.version("robosuite"),
    }


def validate_paths(args: argparse.Namespace, root: Path) -> None:
    expected = {
        "cache": root / CACHE["path"],
        "provenance": root / PROVENANCE["path"],
        "failed local summary": root / LOCAL_SPECIFICITY_SUMMARY_PATH,
        "output": root / MANIFEST_PATH,
    }
    observed = {
        "cache": args.cache,
        "provenance": args.provenance_result,
        "failed local summary": args.failed_local_summary,
        "output": args.output,
    }
    for label, path in observed.items():
        if path.resolve() != expected[label].resolve():
            raise RuntimeError(f"{label} path is not the frozen path")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if file_sha256(args.cache) != CACHE["sha256"]:
        raise RuntimeError("Cache SHA differs from the frozen identity")
    for seed, identity in CHECKPOINTS.items():
        checkpoint = root / identity["path"]
        if not checkpoint.is_file() or file_sha256(checkpoint) != identity["sha256"]:
            raise RuntimeError(f"Checkpoint identity failed for seed {seed}")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    import numpy as np
    from athena.preflight_counterfactual_target_swap import validate_provenance
    from athena.run_local_instruction_specificity import (
        make_environment,
        physical_input_hash,
        reset_to_input,
    )
    from athena.run_xvla_experiment import (
        load_cache_statistics,
        load_suite,
        official_init_states,
        readable_object_name,
        scene_object_bodies,
        task_languages,
    )
    from libero.libero import get_libero_path

    validate_paths(args, root)
    sources_start = source_hashes(root)
    local_summary = validate_failed_local_summary(args.failed_local_summary)
    provenance = validate_provenance(args.provenance_result, args.cache)
    runtime_start = validate_libero_runtime()
    versions = runtime_versions()
    if versions != PROTOCOL["runtime_versions"]:
        raise RuntimeError(f"Preflight runtime versions differ: {versions}")

    suite = load_suite("libero_object")
    languages = task_languages(suite)
    if languages != PROMPTS:
        raise RuntimeError("Live Object task-language catalog differs")
    stats = load_cache_statistics(args.cache, PROTOCOL["action_horizon"])
    if (int(stats["frame_count"]), int(stats["sample_count"])) != (
        CACHE["frames"],
        CACHE["samples"],
    ):
        raise RuntimeError("Cache statistics differ")
    target_to_prompt = prompt_target_to_id()
    mappings: list[dict[str, Any]] = []
    started = time.perf_counter()
    for task_index in range(10):
        task = suite.get_task(task_index)
        expected_bddl, expected_bddl_sha = ORIGINAL_BDDL[task_index]
        if task.bddl_file != expected_bddl or task.problem_folder != "libero_object":
            raise RuntimeError(f"Task {task_index} BDDL metadata differs")
        live_bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        if not live_bddl.is_file() or file_sha256(live_bddl) != expected_bddl_sha:
            raise RuntimeError(f"Task {task_index} live BDDL identity differs")
        init_states = official_init_states(suite, task_index)
        if len(init_states) < 50:
            raise RuntimeError(f"Task {task_index} has fewer than 50 canonical states")
        environment = make_environment(task, PROTOCOL["resolution"])
        try:
            for episode in range(40, 50):
                reset_seed = task_index * 100 + episode
                observation = reset_to_input(
                    environment,
                    init_states[episode],
                    reset_seed,
                    PROTOCOL["settle_steps"],
                )
                bodies = scene_object_bodies(observation)
                if len(bodies) != 6:
                    raise RuntimeError("A manifest state does not contain exactly six objects")
                body_to_prompt = {
                    body: target_to_prompt.get(readable_object_name(body)) for body in bodies
                }
                if any(value is None for value in body_to_prompt.values()):
                    raise RuntimeError("A visible object does not map to the prompt catalog")
                if len(set(body_to_prompt.values())) != 6:
                    raise RuntimeError("Visible objects do not map one-to-one to prompts")
                target_body = next(
                    (body for body, prompt_id in body_to_prompt.items() if prompt_id == task_index),
                    None,
                )
                if target_body is None or readable_object_name(target_body) != TARGETS[task_index]:
                    raise RuntimeError("Official target body cannot be identified")
                present_prompt_ids = sorted(int(value) for value in body_to_prompt.values())
                distractor, selector_digests = select_visible_distractor(
                    task_index, episode, present_prompt_ids
                )
                selected_body = next(
                    body for body, prompt_id in body_to_prompt.items() if prompt_id == distractor
                )
                input_sha, components, _, _ = physical_input_hash(
                    environment, observation, stats, PROTOCOL["resolution"], bodies
                )
                mappings.append(
                    {
                        "task_index": task_index,
                        "episode": episode,
                        "init_state_index": episode,
                        "reset_seed": reset_seed,
                        "init_state_sha256": hash_array(np.asarray(init_states[episode])),
                        "task_language": languages[task_index],
                        "original_bddl_file": expected_bddl,
                        "original_bddl_sha256": expected_bddl_sha,
                        "eligible_bodies": sorted(bodies),
                        "body_to_prompt_id": {
                            body: int(prompt_id) for body, prompt_id in sorted(body_to_prompt.items())
                        },
                        "present_prompt_ids": present_prompt_ids,
                        "target_body": target_body,
                        "selected_distractor_body": selected_body,
                        "selected_distractor_prompt_id": distractor,
                        "selected_distractor_prompt": languages[distractor],
                        "selector_digests": selector_digests,
                        "settled_physical_input_sha256": input_sha,
                        "settled_component_sha256": components,
                    }
                )
        finally:
            environment.close()
    expected_keys = {(task, episode) for task in range(10) for episode in range(40, 50)}
    observed_keys = {(row["task_index"], row["episode"]) for row in mappings}
    if len(mappings) != 100 or observed_keys != expected_keys:
        raise RuntimeError("Manifest does not contain the exact 100-state matrix")

    sources_end = source_hashes(root)
    runtime_end = validate_libero_runtime()
    if sources_start != sources_end or runtime_start != runtime_end:
        raise RuntimeError("Source identity changed during preflight")
    output = {
        "schema": SCHEMA_MANIFEST,
        "protocol": PROTOCOL,
        "frozen_gates": FROZEN_GATES,
        "design_provenance": DESIGN_PROVENANCE,
        "failed_local_summary": str(args.failed_local_summary),
        "failed_local_summary_sha256": file_sha256(args.failed_local_summary),
        "failed_local_summary_identity_validated": local_summary["identity_validated"],
        "failed_local_summary_overall_pass": local_summary["overall_pass"],
        "cache": {**CACHE, "live_sha256": file_sha256(args.cache)},
        "provenance": {
            **PROVENANCE,
            "live_sha256": file_sha256(args.provenance_result),
            "verified": provenance["verified"],
        },
        "checkpoints": CHECKPOINTS,
        "source_sha256_start": sources_start,
        "source_sha256_end": sources_end,
        "libero_runtime_start": runtime_start,
        "libero_runtime_end": runtime_end,
        "runtime_versions": versions,
        "mappings": mappings,
        "mapping_count": len(mappings),
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(args.output, output)
    print(
        "RESULT",
        json.dumps(
            {
                "mapping_count": len(mappings),
                "failed_endpoint_formally_recorded": local_summary["overall_pass"] is False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
