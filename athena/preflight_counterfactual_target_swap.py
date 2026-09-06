#!/usr/bin/env python3
"""Build the frozen BDDL target-swap manifest after the specificity gate passes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from athena.counterfactual_target_swap_common import (
    CACHE,
    BDDL_DIRECTORY,
    CHECKPOINTS,
    FROZEN_GATES,
    ORIGINAL_BDDL,
    PROMPTS,
    PROTOCOL,
    PROVENANCE,
    SCHEMA_MANIFEST,
    MANIFEST_PATH,
    SPECIFICITY,
    file_sha256,
    inspect_original_bddl,
    load_json,
    rewrite_bddl,
    source_hashes,
    validate_libero_runtime,
    validate_specificity_summary,
    write_json,
)
from athena.libero_dataset_metadata import DATASETS
from athena.run_xvla_experiment import load_suite, task_languages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specificity-summary", type=Path, required=True)
    parser.add_argument("--provenance-result", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bddl-dir", type=Path, required=True)
    return parser.parse_args()


def validate_declared_paths(args: argparse.Namespace, root: Path) -> None:
    expected = {
        "specificity-summary": root / SPECIFICITY["path"],
        "provenance-result": root / PROVENANCE["path"],
        "cache": root / CACHE["path"],
    }
    observed = {
        "specificity-summary": args.specificity_summary.resolve(),
        "provenance-result": args.provenance_result.resolve(),
        "cache": args.cache.resolve(),
    }
    for label, expected_path in expected.items():
        if observed[label] != expected_path.resolve():
            raise RuntimeError(f"{label} path is not the frozen path")
    if args.output.resolve() != (root / MANIFEST_PATH).resolve() or args.bddl_dir.resolve() != (root / BDDL_DIRECTORY).resolve():
        raise RuntimeError("Manifest output or BDDL directory is not the frozen fresh namespace")
    for seed, identity in CHECKPOINTS.items():
        checkpoint = root / identity["path"]
        if not checkpoint.is_file() or file_sha256(checkpoint) != identity["sha256"]:
            raise RuntimeError(f"Frozen checkpoint identity failed for seed {seed}")


def validate_provenance(path: Path, cache: Path) -> dict[str, Any]:
    result = load_json(path)
    source_repository, source_revision = DATASETS["libero_object"]
    checks = {
        "schema": result.get("schema") == "xvla-cache-provenance-v1",
        "suite": result.get("suite") == "libero_object",
        "verified": result.get("verified") is True,
        "cache_sha": result.get("cache", {}).get("sha256") == CACHE["sha256"],
        "cache_frames": int(result.get("cache", {}).get("frames", -1)) == CACHE["frames"],
        "cache_resolution": int(result.get("cache", {}).get("resolution", -1)) == PROTOCOL["resolution"],
        "repository": result.get("source", {}).get("repository") == source_repository == PROVENANCE["repository"],
        "revision": result.get("source", {}).get("revision") == source_revision == PROVENANCE["revision"],
        "metadata_sha": result.get("metadata", {}).get("sha256") == PROVENANCE["metadata_sha256"],
        "metadata_revision": result.get("metadata", {}).get("revision") == PROVENANCE["revision"],
        "canonical_content": result.get("source", {}).get("canonical_content_sha256") == PROVENANCE["canonical_content_sha256"]
        and result.get("cache", {}).get("canonical_content_sha256") == PROVENANCE["canonical_content_sha256"],
        "content_hashes": result.get("source", {}).get("content_hashes_match") is True,
        "mismatch_counts": bool(result.get("comparison", {}).get("mismatch_counts"))
        and all(int(value) == 0 for value in result["comparison"]["mismatch_counts"].values()),
        "live_cache": cache.is_file() and file_sha256(cache) == CACHE["sha256"],
    }
    failed = [label for label, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Object cache provenance validation failed: " + ", ".join(failed))
    return result


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.bddl_dir.exists():
        raise FileExistsError("Refusing to overwrite the target-swap manifest or BDDL directory")
    validate_declared_paths(args, root)
    sources_start = source_hashes(root, ("athena/preflight_counterfactual_target_swap.py",))
    specificity = validate_specificity_summary(args.specificity_summary)
    provenance = validate_provenance(args.provenance_result, args.cache)
    libero_runtime_start = validate_libero_runtime()
    runtime_versions = {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "torch_build": __import__("torch").__version__,
        "cuda": __import__("torch").version.cuda,
        "numpy": importlib.metadata.version("numpy"),
        "pillow": importlib.metadata.version("Pillow"),
        "mujoco": importlib.metadata.version("mujoco"),
        "robosuite": importlib.metadata.version("robosuite"),
    }
    if runtime_versions != PROTOCOL["runtime_versions"]:
        raise RuntimeError(f"Preflight runtime versions differ: {runtime_versions}")
    suite = load_suite("libero_object")
    languages = task_languages(suite)
    if languages != PROMPTS:
        raise RuntimeError("Live LIBERO-Object task order or language catalog differs")

    from libero.libero import get_libero_path

    bddl_root = Path(get_libero_path("bddl_files"))
    args.bddl_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix="target-swap-bddl-", dir=args.bddl_dir.parent))
    mappings = []
    try:
        for task_index in range(10):
            task = suite.get_task(task_index)
            expected_name, expected_sha = ORIGINAL_BDDL[task_index]
            if task.bddl_file != expected_name or task.problem_folder != "libero_object":
                raise RuntimeError(f"Task {task_index} BDDL path metadata differs")
            original_path = bddl_root / task.problem_folder / task.bddl_file
            if file_sha256(original_path) != expected_sha:
                raise RuntimeError(f"Task {task_index} original BDDL SHA differs")
            original_text = original_path.read_text()
            mapping = inspect_original_bddl(original_text, task_index)
            for distractor in mapping["distractor_prompt_ids"]:
                rewritten, validation = rewrite_bddl(original_text, task_index, distractor)
                filename = f"task_{task_index:02d}_goal_prompt_{distractor:02d}.bddl"
                rewritten_path = temporary_dir / filename
                rewritten_path.write_text(rewritten)
                if rewritten_path.read_text() != rewritten:
                    raise RuntimeError(f"Rewritten BDDL write did not round-trip for task {task_index}")
                mappings.append(
                    {
                        "task_index": task_index,
                        "original_prompt_id": task_index,
                        "original_prompt": PROMPTS[task_index],
                        "counterfactual_prompt_id": distractor,
                        "counterfactual_prompt": PROMPTS[distractor],
                        "original_bddl_path": str(original_path.resolve()),
                        "original_bddl_sha256": expected_sha,
                        "rewritten_bddl_path": str((args.bddl_dir / filename).resolve()),
                        "rewritten_bddl_sha256": file_sha256(rewritten_path),
                        "rewrite_validation": validation,
                    }
                )
        if len(mappings) != 50:
            raise RuntimeError(f"Expected 50 target-swap mappings, built {len(mappings)}")
        temporary_dir.replace(args.bddl_dir)
    except BaseException:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise

    sources_end = source_hashes(root, ("athena/preflight_counterfactual_target_swap.py",))
    if sources_start != sources_end:
        raise RuntimeError("Repository sources changed during target-swap preflight")
    libero_runtime_end = validate_libero_runtime()
    if libero_runtime_start != libero_runtime_end:
        raise RuntimeError("LIBERO sources changed during target-swap preflight")
    output = {
        "schema": SCHEMA_MANIFEST,
        "protocol": PROTOCOL,
        "frozen_gates": FROZEN_GATES,
        "specificity_gate_validated": True,
        "specificity_summary": str(args.specificity_summary),
        "specificity_summary_sha256": file_sha256(args.specificity_summary),
        "specificity_summary_job_id": SPECIFICITY["job_id"],
        "specificity_gate_fields": {
            "identity_validated": specificity["identity_validated"],
            "overall_pass": specificity["overall_pass"],
            "claim_eligible": specificity["claim_eligible"],
            "all_checkpoint_gates_pass": all(
                specificity["checkpoint_results"][str(seed)]["passes_frozen_gate"]
                for seed in range(3)
            ),
        },
        "cache": {
            **CACHE,
            "live_sha256": file_sha256(args.cache),
        },
        "provenance": {
            **PROVENANCE,
            "live_sha256": file_sha256(args.provenance_result),
            "verified": provenance["verified"],
        },
        "checkpoints": CHECKPOINTS,
        "source_sha256_start": sources_start,
        "source_sha256_end": sources_end,
        "libero_runtime_start": libero_runtime_start,
        "libero_runtime_end": libero_runtime_end,
        "runtime": {
            "libero_path": str(Path(__import__("libero").__file__).resolve()),
            **runtime_versions,
        },
        "mappings": mappings,
        "mapping_count": len(mappings),
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(args.output, output)
    print("RESULT", json.dumps({"mapping_count": len(mappings), "specificity_gate_validated": True}, sort_keys=True))


if __name__ == "__main__":
    main()
