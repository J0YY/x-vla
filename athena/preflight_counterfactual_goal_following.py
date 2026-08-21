#!/usr/bin/env python3
"""Build the fresh goal-following BDDL manifest after instruction necessity passes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import tempfile
import time
from pathlib import Path

from athena.counterfactual_goal_following_common import (
    BDDL_DIRECTORY,
    CACHE,
    CHECKPOINTS,
    DESIGN_PROVENANCE,
    FROZEN_GATES,
    INSTRUCTION_SUMMARY_PATH,
    INSTRUCTION_SUMMARY_JOB,
    MANIFEST_PATH,
    ORIGINAL_BDDL,
    PROMPTS,
    PROTOCOL,
    PROVENANCE,
    SCHEMA_MANIFEST,
    file_sha256,
    inspect_original_bddl,
    rewrite_bddl,
    source_hashes,
    validate_instruction_summary,
    validate_libero_runtime,
    write_json,
)
from athena.preflight_counterfactual_target_swap import validate_provenance
from athena.run_xvla_experiment import load_suite, task_languages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction-summary", type=Path, required=True)
    parser.add_argument("--provenance-result", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bddl-dir", type=Path, required=True)
    return parser.parse_args()


def runtime_versions() -> dict[str, str | None]:
    torch = __import__("torch")
    return {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "torch_build": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": importlib.metadata.version("numpy"),
        "pillow": importlib.metadata.version("Pillow"),
        "mujoco": importlib.metadata.version("mujoco"),
        "robosuite": importlib.metadata.version("robosuite"),
    }


def validate_paths(args: argparse.Namespace, root: Path) -> None:
    expected = {
        "instruction summary": root / INSTRUCTION_SUMMARY_PATH,
        "provenance": root / PROVENANCE["path"],
        "cache": root / CACHE["path"],
        "output": root / MANIFEST_PATH,
        "BDDL directory": root / BDDL_DIRECTORY,
    }
    observed = {
        "instruction summary": args.instruction_summary,
        "provenance": args.provenance_result,
        "cache": args.cache,
        "output": args.output,
        "BDDL directory": args.bddl_dir,
    }
    for label, path in observed.items():
        if path.resolve() != expected[label].resolve():
            raise RuntimeError(f"{label} path is not frozen")
    if args.output.exists() or args.bddl_dir.exists():
        raise FileExistsError("Refusing to overwrite the fresh manifest namespace")
    if file_sha256(args.cache) != CACHE["sha256"]:
        raise RuntimeError("Cache identity differs")
    for seed, identity in CHECKPOINTS.items():
        checkpoint = root / identity["path"]
        if not checkpoint.is_file() or file_sha256(checkpoint) != identity["sha256"]:
            raise RuntimeError(f"Checkpoint identity differs for seed {seed}")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    validate_paths(args, root)
    sources_start = source_hashes(root)
    instruction = validate_instruction_summary(args.instruction_summary, root)
    provenance = validate_provenance(args.provenance_result, args.cache)
    runtime_start = validate_libero_runtime()
    versions = runtime_versions()
    if versions != PROTOCOL["runtime_versions"]:
        raise RuntimeError(f"Preflight runtime differs: {versions}")

    suite = load_suite("libero_object")
    languages = task_languages(suite)
    if languages != PROMPTS:
        raise RuntimeError("Live Object prompt catalog differs")
    from libero.libero import get_libero_path

    bddl_root = Path(get_libero_path("bddl_files"))
    args.bddl_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="goal-following-bddl-", dir=args.bddl_dir.parent))
    mappings = []
    try:
        for task_index in range(10):
            task = suite.get_task(task_index)
            expected_name, expected_sha = ORIGINAL_BDDL[task_index]
            if task.problem_folder != "libero_object" or task.bddl_file != expected_name:
                raise RuntimeError(f"Task {task_index} BDDL metadata differs")
            original_path = bddl_root / task.problem_folder / task.bddl_file
            if not original_path.is_file() or file_sha256(original_path) != expected_sha:
                raise RuntimeError(f"Task {task_index} original BDDL identity differs")
            original_text = original_path.read_text()
            scene = inspect_original_bddl(original_text, task_index)
            if len(scene["distractor_prompt_ids"]) != 5:
                raise RuntimeError("A task does not contain five co-present targets")
            for target in scene["distractor_prompt_ids"]:
                rewritten, rewrite_validation = rewrite_bddl(
                    original_text, task_index, int(target)
                )
                filename = f"task_{task_index:02d}_goal_prompt_{int(target):02d}.bddl"
                temporary_path = temporary / filename
                temporary_path.write_text(rewritten)
                if temporary_path.read_text() != rewritten:
                    raise RuntimeError("Rewritten BDDL did not round-trip")
                mappings.append(
                    {
                        "task_index": task_index,
                        "original_prompt_id": task_index,
                        "original_prompt": PROMPTS[task_index],
                        "counterfactual_prompt_id": int(target),
                        "counterfactual_prompt": PROMPTS[int(target)],
                        "present_prompt_ids": scene["present_prompt_ids"],
                        "original_bddl_file": expected_name,
                        "original_bddl_path": str(original_path.resolve()),
                        "original_bddl_sha256": expected_sha,
                        "rewritten_bddl_path": str((args.bddl_dir / filename).resolve()),
                        "rewritten_bddl_sha256": file_sha256(temporary_path),
                        "rewrite_validation": rewrite_validation,
                    }
                )
        if len(mappings) != 50 or len(
            {(row["task_index"], row["counterfactual_prompt_id"]) for row in mappings}
        ) != 50:
            raise RuntimeError("Manifest does not contain all fifty target rewrites")
        temporary.replace(args.bddl_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    sources_end = source_hashes(root)
    runtime_end = validate_libero_runtime()
    if sources_start != sources_end or runtime_start != runtime_end:
        raise RuntimeError("Sources changed during preflight")
    output = {
        "schema": SCHEMA_MANIFEST,
        "protocol": PROTOCOL,
        "frozen_gates": FROZEN_GATES,
        "design_provenance": DESIGN_PROVENANCE,
        "instruction_summary": str(args.instruction_summary),
        "instruction_summary_job_id": INSTRUCTION_SUMMARY_JOB,
        "instruction_summary_sha256": file_sha256(args.instruction_summary),
        "instruction_gate_fields": {
            "identity_validated": instruction["identity_validated"],
            "overall_pass": instruction["overall_pass"],
            "claim_eligible": instruction["claim_eligible"],
            "all_component_gates_pass": all(instruction["gates"].values()),
        },
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
                "instruction_necessity_gate_validated": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
