#!/usr/bin/env python3
"""Freeze the three-member multi-suite ensemble inputs before ensemble outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from athena.summarize_multisuite_generalist import (
    CACHES,
    EXPECTED_EVALUATION_GPU_NAME,
    EXPECTED_NUMERICAL_ENVIRONMENT,
    MANIFEST_PATH,
    MANIFEST_SHA256,
    SEEDS,
    SUITES,
    file_sha256,
    load_checkpoint_results,
    load_live_artifact_identity,
    require_equal,
    validate_cache_provenance,
    validate_cache_table,
)


SCHEMA = "xvla_multisuite_mean_ensemble_manifest_v1"
ENSEMBLE_GATES = {
    "macro_task_success_at_least": 0.70,
    "every_suite_macro_at_least": 0.50,
    "tasks_at_least_0p50_at_least": 32,
    "uplift_over_member_mean_at_least": 0.00,
}
SOURCE_PATHS = (
    "athena/build_multisuite_ensemble_manifest.py",
    "athena/summarize_multisuite_ensemble.py",
    "athena/slurm_eval_multisuite_ensemble.sbatch",
    "athena/slurm_summarize_multisuite_ensemble.sbatch",
    "athena/launch_multisuite_ensemble.sh",
    "athena/run_xvla_experiment.py",
    "athena/summarize_multisuite_generalist.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/multisuite_mean_ensemble_v1_manifest.json"),
    )
    return parser.parse_args()


def bundle_sha256(files: dict[str, str]) -> str:
    canonical = "\n".join(f"{path}\0{files[path]}" for path in sorted(files))
    return hashlib.sha256(canonical.encode()).hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite frozen manifest {args.output}")
    require_equal(args.artifacts_dir.as_posix(), "artifacts", "artifacts directory")
    require_equal(args.results_dir.as_posix(), "results", "results directory")
    require_equal(file_sha256(MANIFEST_PATH), MANIFEST_SHA256, "generalist manifest")
    with MANIFEST_PATH.open() as handle:
        generalist_manifest = json.load(handle)
    validate_cache_table(generalist_manifest, "generalist manifest")
    cache_provenance = validate_cache_provenance(args.results_dir)

    source_start = {path: file_sha256(Path(path)) for path in SOURCE_PATHS}
    artifacts: dict[str, dict[str, Any]] = {}
    members: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        artifact = load_live_artifact_identity(
            args.artifacts_dir, "chi", seed, generalist_manifest
        )
        artifacts[str(seed)] = artifact
        member = load_checkpoint_results(
            args.results_dir,
            "chi",
            seed,
            generalist_manifest,
            artifact["checkpoint_sha256"],
            artifact["metadata"],
            artifact["metadata_sha256"],
        )
        members[str(seed)] = member

    reference_protocol = members["0"]["task_protocol"]
    reference_runtime = members["0"]["evaluation_environment"]
    reference_gpu = members["0"]["evaluation_gpu"]
    require_equal(reference_runtime, EXPECTED_NUMERICAL_ENVIRONMENT, "evaluation runtime")
    require_equal(reference_gpu, EXPECTED_EVALUATION_GPU_NAME, "evaluation GPU")
    for seed in SEEDS:
        member = members[str(seed)]
        require_equal(member["task_protocol"], reference_protocol, f"seed {seed} task protocol")
        require_equal(member["evaluation_environment"], reference_runtime, f"seed {seed} runtime")
        require_equal(member["evaluation_gpu"], reference_gpu, f"seed {seed} GPU")

    member_macros = [members[str(seed)]["macro_task_success"] for seed in SEEDS]
    member_mean = sum(member_macros) / len(member_macros)
    public_artifacts: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        artifact = artifacts[str(seed)]
        public_artifacts[str(seed)] = {
            "checkpoint_path": artifact["checkpoint_path"].as_posix(),
            "checkpoint_sha256": artifact["checkpoint_sha256"],
            "checkpoint_archive_index_sha256": artifact[
                "checkpoint_archive_index_sha256"
            ],
            "metadata_path": artifact["metadata_path"].as_posix(),
            "metadata_sha256": artifact["metadata_sha256"],
        }

    source_end = {path: file_sha256(Path(path)) for path in SOURCE_PATHS}
    require_equal(source_end, source_start, "manifest sources start/end")
    output = {
        "schema": SCHEMA,
        "status": "frozen_after_member_evaluations_before_ensemble_outcomes",
        "scope": {
            "architecture": "chi",
            "vision_encoder": "vit",
            "seeds": list(SEEDS),
            "suites": list(SUITES),
            "tasks_per_suite": 10,
            "episodes_per_task": 50,
            "ensemble_trials": 2000,
            "ensemble_reduction": "elementwise_mean_of_normalized_action_chunks",
            "inference_cost": "three policy forward passes per query",
            "claim_boundary": (
                "In-domain evaluation of one fixed three-checkpoint mean-prediction ensemble "
                "over the same 40 familiar tasks. This is not zero-shot generalization and "
                "does not alter the preregistered individual-checkpoint capability gates."
            ),
        },
        "protocol": {
            "resolution": 64,
            "action_horizon": 8,
            "settle_steps": 10,
            "execution_horizon": 8,
            "max_steps": 280,
            "matmul_precision": "highest",
            "canonical_episodes": list(range(50)),
            "gpu": EXPECTED_EVALUATION_GPU_NAME,
        },
        "frozen_gates": ENSEMBLE_GATES,
        "generalist_manifest": {
            "path": MANIFEST_PATH.as_posix(),
            "sha256": MANIFEST_SHA256,
        },
        "cache_provenance": cache_provenance,
        "task_protocol": reference_protocol,
        "evaluation_environment": reference_runtime,
        "evaluation_gpu": reference_gpu,
        "artifacts": public_artifacts,
        "member_results": {
            str(seed): {
                "macro_task_success": members[str(seed)]["macro_task_success"],
                "suite_macro_success": members[str(seed)]["suite_macro_success"],
                "task_success": members[str(seed)]["task_success"],
                "successes": members[str(seed)]["successes"],
                "trials": members[str(seed)]["trials"],
            "source_files": members[str(seed)]["source_files"],
            }
            for seed in SEEDS
        },
        "member_macro_successes": member_macros,
        "member_mean_macro_success": member_mean,
        "source_files": source_start,
        "source_bundle_sha256": bundle_sha256(source_start),
    }

    for seed in SEEDS:
        artifact = artifacts[str(seed)]
        require_equal(
            file_sha256(artifact["checkpoint_path"]),
            artifact["checkpoint_sha256"],
            f"seed {seed} checkpoint start/end",
        )
        require_equal(
            file_sha256(artifact["metadata_path"]),
            artifact["metadata_sha256"],
            f"seed {seed} metadata start/end",
        )
        for source in members[str(seed)]["source_files"]:
            require_equal(
                file_sha256(Path(source["path"])),
                source["sha256"],
                f"seed {seed} member result start/end",
            )
    require_equal(
        validate_cache_provenance(args.results_dir),
        cache_provenance,
        "cache provenance start/end",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
