#!/usr/bin/env python3
"""Strictly recompute the separate seed-0 generalist-to-specialist retention gate."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from athena.specialist_retention_protocol import (
    ARCHITECTURE,
    CACHE_ENVIRONMENT_KEYS,
    EMA_DECAY,
    EVALUATION_GPU_FAMILY,
    EVALUATION_GPU_NAME,
    EXPECTED_EVALUATION_RUNTIME,
    FULL_BATCH_SIZE,
    FULL_STEPS,
    GENERALIST_MANIFEST_PATH,
    GENERALIST_MANIFEST_SHA256,
    GENERALIST_SUMMARY_SOURCE_SHA256,
    HORIZON,
    LEARNING_RATE,
    MODEL_SOURCE_HASHES,
    RECIPE_VERSION,
    RESOLUTION,
    SEED,
    SHARDS,
    SUITES,
    SUITE_SPECS,
    TRAIN_LM_SHA256,
    VISION_ENCODER,
    bundle_sha256,
    checkpoint_path,
    evaluation_result_path,
    file_sha256,
    metadata_path,
    require_equal,
    source_snapshot,
    validate_cache_and_provenance,
    validate_source_snapshot,
    validate_task_metadata,
)
from athena.summarize_multisuite_generalist import (
    load_checkpoint_results as load_generalist_checkpoint_results,
    load_live_artifact_identity as load_generalist_artifact_identity,
    validate_cache_provenance as validate_generalist_cache_provenance,
    validate_cache_table as validate_generalist_cache_table,
)


PROTOCOL_SOURCE_SHA256 = "0bee62f9296496c4ee0943a0b74860c927c69d436bac468c7dd947500b7700eb"
TRAINER_SHA256 = "95e086b739359ba0cc653e3de13199c031ac22414bd217ff08113ea3c3129c27"
EVALUATOR_SHA256 = "fd28d56acb9e198c1dbcd207c8614d32a81d9ac50f3de3016fe9a2817bd19939"
EVALUATION_IMPORTED_SOURCES = {
    **MODEL_SOURCE_HASHES,
    "athena/specialist_retention_protocol.py": PROTOCOL_SOURCE_SHA256,
}
TRAINING_IMPORTED_SOURCES = {
    **EVALUATION_IMPORTED_SOURCES,
    "xvla/train/train_lm.py": TRAIN_LM_SHA256,
}
SUMMARY_IMPORTED_SOURCES = {
    "athena/specialist_retention_protocol.py": PROTOCOL_SOURCE_SHA256,
    "athena/summarize_multisuite_generalist.py": GENERALIST_SUMMARY_SOURCE_SHA256,
}
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/specialist_retention_hardened_summary.json"),
    )
    return parser.parse_args()


def load_json_with_sha(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def validate_consumed_sources_unchanged(
    source_files: list[dict[str, str]], label: str
) -> list[dict[str, str]]:
    end_files: list[dict[str, str]] = []
    for source in source_files:
        path = Path(source["path"])
        end_sha256 = file_sha256(path)
        require_equal(end_sha256, source["sha256"], f"{label} {path} start/end")
        end_files.append({"path": source["path"], "sha256": end_sha256})
    return end_files


def normalized_language(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def validate_live_training_metadata(
    metadata: dict[str, Any], suite: str, checkpoint_sha256: str
) -> None:
    spec = SUITE_SPECS[suite]
    expected = {
        "format": "xvla_specialist_retention_checkpoint_v1",
        "recipe_version": RECIPE_VERSION,
        "mode": "full",
        "architecture": ARCHITECTURE,
        "vision_encoder": VISION_ENCODER,
        "suite": suite,
        "seed": SEED,
        "steps": FULL_STEPS,
        "batch_size": FULL_BATCH_SIZE,
        "lr": LEARNING_RATE,
        "ema_decay": EMA_DECAY,
        "res": RESOLUTION,
        "horizon": HORIZON,
        "state_dim": 8,
        "action_dim": 7,
        "sample_count": spec["sample_count"],
        "checkpoint": checkpoint_path(suite).as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "cache": spec["cache_path"],
        "cache_sha256": spec["cache_sha256"],
        "trainer_sha256": TRAINER_SHA256,
    }
    for key, value in expected.items():
        require_equal(metadata.get(key), value, f"{suite} metadata {key}")
    require_equal(
        metadata.get("provenance"),
        validate_cache_and_provenance(suite),
        f"{suite} metadata provenance",
    )
    task_metadata = metadata.get("task_metadata", {})
    validate_task_metadata(suite, task_metadata)
    dataset_tasks = metadata.get("dataset_tasks", {})
    official_tasks = metadata.get("official_tasks", {})
    require_equal(set(dataset_tasks), {str(index) for index in range(10)}, f"{suite} dataset tasks")
    require_equal(set(official_tasks), {str(index) for index in range(10)}, f"{suite} official tasks")
    for dataset_index, official_index in spec["dataset_to_official_task"].items():
        require_equal(
            normalized_language(dataset_tasks[dataset_index]),
            normalized_language(official_tasks[str(official_index)]),
            f"{suite} task mapping {dataset_index}->{official_index}",
        )
    vocab = metadata.get("vocab", {})
    require_equal(vocab.get("<pad>"), 0, f"{suite} padding token")
    require_equal(vocab.get("<bos>"), 1, f"{suite} BOS token")
    require_equal(sorted(vocab.values()), list(range(len(vocab))), f"{suite} vocab IDs")
    vocab_payload = json.dumps(vocab, sort_keys=True, separators=(",", ":"))
    require_equal(
        metadata.get("vocab_sha256"),
        hashlib.sha256(vocab_payload.encode()).hexdigest(),
        f"{suite} vocab SHA-256",
    )
    normalization = metadata.get("normalization", {})
    for key, size in (
        ("state_mean", 8), ("state_std", 8), ("action_mean", 7), ("action_std", 7)
    ):
        values = normalization.get(key)
        if not isinstance(values, list) or len(values) != size:
            raise RuntimeError(f"{suite} normalization {key} is malformed")
        if any(not isinstance(value, (int, float)) for value in values):
            raise RuntimeError(f"{suite} normalization {key} is nonnumeric")
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"{suite} normalization {key} is non-finite")
        if key.endswith("std") and any(value <= 0 for value in values):
            raise RuntimeError(f"{suite} normalization {key} is not positive")
    parameters = metadata.get("parameters")
    if type(parameters) is not int or parameters <= 0:
        raise RuntimeError(f"{suite} parameter count is invalid")
    source = metadata.get("source_identity", {})
    for key in ("expected_trainer_sha256", "trainer_start_sha256", "trainer_end_sha256"):
        require_equal(source.get(key), TRAINER_SHA256, f"{suite} {key}")
    for key in ("imported_sources_start", "imported_sources_end"):
        require_equal(source.get(key), TRAINING_IMPORTED_SOURCES, f"{suite} {key}")
    expected_bundle = bundle_sha256(TRAINING_IMPORTED_SOURCES)
    for key in ("imported_bundle_start_sha256", "imported_bundle_end_sha256"):
        require_equal(source.get(key), expected_bundle, f"{suite} {key}")
    require_equal(source.get("cache_identity_start"), metadata["provenance"], f"{suite} cache start")
    require_equal(source.get("cache_identity_end"), metadata["provenance"], f"{suite} cache end")
    require_equal(source.get("task_metadata_start"), task_metadata, f"{suite} mapping start")
    require_equal(source.get("task_metadata_end"), task_metadata, f"{suite} mapping end")
    require_equal(
        source.get("cache_environment_start"),
        source.get("cache_environment_end"),
        f"{suite} trainer cache environment",
    )
    require_equal(
        set(source.get("cache_environment_start", {})),
        set(CACHE_ENVIRONMENT_KEYS),
        f"{suite} trainer cache environment keys",
    )
    for key, value in source.get("cache_environment_start", {}).items():
        if not value.startswith("/work/"):
            raise RuntimeError(f"{suite} trainer cache {key} is outside /work")
    environment = metadata.get("training_environment", {})
    require_equal(environment.get("matmul_precision"), "high", f"{suite} train precision")
    require_equal(environment.get("autocast_dtype"), "bfloat16", f"{suite} train autocast")
    for key in ("torch_version", "cuda_version", "gpu"):
        if not environment.get(key):
            raise RuntimeError(f"{suite} training environment lacks {key}")


def load_live_specialist_artifact(
    artifacts_dir: Path, suite: str
) -> dict[str, Any]:
    checkpoint = artifacts_dir / checkpoint_path(suite).name
    metadata_file = artifacts_dir / metadata_path(suite).name
    require_equal(checkpoint.as_posix(), checkpoint_path(suite).as_posix(), f"{suite} checkpoint path")
    require_equal(metadata_file.as_posix(), metadata_path(suite).as_posix(), f"{suite} metadata path")
    checkpoint_sha256 = file_sha256(checkpoint)
    metadata, metadata_sha256 = load_json_with_sha(metadata_file)
    validate_live_training_metadata(metadata, suite, checkpoint_sha256)
    if not zipfile.is_zipfile(checkpoint):
        raise RuntimeError(f"{suite} checkpoint is not a PyTorch ZIP archive")
    with zipfile.ZipFile(checkpoint) as archive:
        members = sorted(archive.namelist())
        bad_members = [
            member for member in members
            if member.startswith("/") or ".." in Path(member).parts
        ]
        if bad_members or not any(member.endswith("/data.pkl") for member in members):
            raise RuntimeError(f"{suite} checkpoint archive structure is invalid")
    return {
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_archive_index_sha256": hashlib.sha256(
            "\n".join(members).encode()
        ).hexdigest(),
        "metadata_path": metadata_file,
        "metadata_sha256": metadata_sha256,
        "metadata": metadata,
    }


def load_specialist_results(
    results_dir: Path, suite: str, artifact: dict[str, Any]
) -> dict[str, Any]:
    task_outcomes: dict[int, list[bool]] = defaultdict(list)
    seen_episodes: set[tuple[int, int]] = set()
    source_files: list[dict[str, str]] = []
    task_protocol: dict[str, Any] = {}
    metadata = artifact["metadata"]
    official_tasks = metadata["official_tasks"]
    canonical_runtime = None
    canonical_gpu = None
    for start, end in SHARDS:
        path = results_dir / evaluation_result_path(suite, start, end).name
        result, result_sha256 = load_json_with_sha(path)
        label = path.as_posix()
        source_files.append({"path": label, "sha256": result_sha256})
        expected_identity = {
            "format": "xvla_specialist_retention_evaluation_v1",
            "mode": "capability",
            "architecture": ARCHITECTURE,
            "vision_encoder": VISION_ENCODER,
            "suite": suite,
            "training_suite": suite,
            "seed": SEED,
            "checkpoint": checkpoint_path(suite).as_posix(),
            "model_metadata": metadata_path(suite).as_posix(),
            "cache": SUITE_SPECS[suite]["cache_path"],
            "matmul_precision": "highest",
        }
        for key, value in expected_identity.items():
            require_equal(result.get(key), value, f"{label} {key}")
        require_equal(result.get("prediction_finite"), True, f"{label} finite prediction")
        require_equal(result.get("prediction_shape"), [1, 8, 7], f"{label} prediction shape")
        require_equal(
            result.get("evaluation_protocol"),
            {
                "res": 64,
                "horizon": 8,
                "num_steps_wait": 10,
                "exec_h": 8,
                "eps_per_task": 50,
                "max_steps": 280,
                "task_start": start,
                "task_end": end,
            },
            f"{label} evaluation protocol",
        )
        require_equal(result.get("training_metadata"), metadata, f"{label} live metadata")
        runtime = result.get("numerical_environment")
        require_equal(runtime, EXPECTED_EVALUATION_RUNTIME, f"{label} runtime")
        gpu = result.get("profile", {}).get("gpu")
        require_equal(gpu, EVALUATION_GPU_NAME, f"{label} GPU")
        if canonical_runtime is None:
            canonical_runtime, canonical_gpu = runtime, gpu
        require_equal(runtime, canonical_runtime, f"{label} suite runtime")
        require_equal(gpu, canonical_gpu, f"{label} suite GPU")
        require_equal(
            result.get("profile", {}).get("parameters"),
            metadata["parameters"],
            f"{label} parameters",
        )
        identity = result.get("source_identity", {})
        require_equal(identity.get("expected_evaluator_sha256"), EVALUATOR_SHA256, f"{label} evaluator pin")
        start_identity = identity.get("start", {})
        require_equal(start_identity, identity.get("end", {}), f"{label} source start/end")
        expected_source = {
            "evaluator_sha256": EVALUATOR_SHA256,
            "evaluation_gpu_family": EVALUATION_GPU_FAMILY,
            "evaluation_gpu_name": EVALUATION_GPU_NAME,
            "imported_sources": EVALUATION_IMPORTED_SOURCES,
            "imported_bundle_sha256": bundle_sha256(EVALUATION_IMPORTED_SOURCES),
            "cache_identity": metadata["provenance"],
            "task_metadata": metadata["task_metadata"],
            "checkpoint_sha256": artifact["checkpoint_sha256"],
            "metadata_sha256": artifact["metadata_sha256"],
        }
        for key, value in expected_source.items():
            require_equal(start_identity.get(key), value, f"{label} source {key}")
        require_equal(
            set(start_identity.get("cache_environment", {})),
            set(CACHE_ENVIRONMENT_KEYS),
            f"{label} cache environment keys",
        )
        for key, value in start_identity.get("cache_environment", {}).items():
            if not value.startswith("/work/"):
                raise RuntimeError(f"{label} cache environment {key} is outside /work")

        capability = result.get("capability", {})
        expected_tasks = list(range(start, end))
        require_equal(
            {
                "task_indices": capability.get("task_indices"),
                "eps_per_task": capability.get("eps_per_task"),
                "max_steps": capability.get("max_steps"),
                "num_steps_wait": capability.get("num_steps_wait"),
                "exec_h": capability.get("exec_h"),
                "canonical_init_states": capability.get("canonical_init_states"),
            },
            {
                "task_indices": expected_tasks,
                "eps_per_task": 50,
                "max_steps": 280,
                "num_steps_wait": 10,
                "exec_h": 8,
                "canonical_init_states": True,
            },
            f"{label} capability protocol",
        )
        shard_protocol = capability.get("task_protocol", {})
        require_equal(set(shard_protocol), {str(task) for task in expected_tasks}, f"{label} protocol keys")
        for task in expected_tasks:
            protocol = shard_protocol[str(task)]
            require_equal(protocol.get("language"), official_tasks[str(task)], f"{label} task language")
            if int(protocol.get("init_state_count", -1)) < 50:
                raise RuntimeError(f"{label} task {task} has fewer than 50 init states")
            for key in ("language", "problem_folder", "bddl_file", "bddl_sha256", "init_states_sha256"):
                if not protocol.get(key):
                    raise RuntimeError(f"{label} task {task} protocol lacks {key}")
            task_protocol[str(task)] = protocol
        episodes = capability.get("episodes")
        expected_trials = len(expected_tasks) * 50
        if not isinstance(episodes, list):
            raise RuntimeError(f"{label} episodes are missing")
        require_equal(len(episodes), expected_trials, f"{label} episode rows")
        shard_successes = 0
        per_task_successes = {task: 0 for task in expected_tasks}
        for episode in episodes:
            task = episode.get("task_index")
            episode_index = episode.get("episode")
            success = episode.get("success")
            steps = episode.get("steps")
            if type(task) is not int or task not in expected_tasks:
                raise RuntimeError(f"{label} invalid task episode")
            if type(episode_index) is not int or not 0 <= episode_index < 50:
                raise RuntimeError(f"{label} invalid episode index")
            if type(success) is not bool:
                raise RuntimeError(f"{label} non-Boolean success")
            if type(steps) is not int or not 0 <= steps <= 280:
                raise RuntimeError(f"{label} invalid step count")
            key = (task, episode_index)
            if key in seen_episodes:
                raise RuntimeError(f"{label} duplicate episode {key}")
            seen_episodes.add(key)
            task_outcomes[task].append(success)
            shard_successes += int(success)
            per_task_successes[task] += int(success)
        require_equal(capability.get("trials"), expected_trials, f"{label} trials")
        require_equal(capability.get("successes"), shard_successes, f"{label} successes")
        require_equal(capability.get("overall"), shard_successes / expected_trials, f"{label} overall")
        expected_per_task = {
            official_tasks[str(task)]: per_task_successes[task] / 50
            for task in expected_tasks
        }
        require_equal(capability.get("per_task"), expected_per_task, f"{label} per-task")
    require_equal(len(seen_episodes), 500, f"{suite} episode coverage")
    require_equal(set(task_outcomes), set(range(10)), f"{suite} task coverage")
    for task, outcomes in task_outcomes.items():
        require_equal(len(outcomes), 50, f"{suite} task {task} trials")
    task_success = {
        str(task): sum(task_outcomes[task]) / 50 for task in range(10)
    }
    return {
        "task_success": task_success,
        "macro_task_success": sum(task_success.values()) / 10,
        "successes": sum(sum(rows) for rows in task_outcomes.values()),
        "trials": 500,
        "evaluation_environment": canonical_runtime,
        "evaluation_gpu": canonical_gpu,
        "task_protocol": task_protocol,
        "source_files": source_files,
    }


def main() -> None:
    args = parse_args()
    require_equal(args.artifacts_dir.as_posix(), "artifacts", "hardened artifacts directory")
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".tmp").exists():
        raise FileExistsError(f"Refusing to overwrite hardened summary {args.output}")
    summary_start_sha256 = file_sha256(Path(__file__))
    imported_start = source_snapshot(SUMMARY_IMPORTED_SOURCES)
    validate_source_snapshot(imported_start, SUMMARY_IMPORTED_SOURCES)
    require_equal(
        file_sha256(GENERALIST_MANIFEST_PATH),
        GENERALIST_MANIFEST_SHA256,
        "frozen generalist manifest",
    )
    live_manifest = json.loads(GENERALIST_MANIFEST_PATH.read_text())
    validate_generalist_cache_table(live_manifest, "live generalist manifest")
    generalist_cache_provenance = validate_generalist_cache_provenance(args.results_dir)
    specialist_cache_provenance = {
        suite: validate_cache_and_provenance(suite) for suite in SUITES
    }

    artifacts = {
        suite: load_live_specialist_artifact(args.artifacts_dir, suite)
        for suite in SUITES
    }
    specialists = {
        suite: load_specialist_results(args.results_dir, suite, artifacts[suite])
        for suite in SUITES
    }
    reference_runtime = specialists[SUITES[0]]["evaluation_environment"]
    reference_gpu = specialists[SUITES[0]]["evaluation_gpu"]
    for suite in SUITES:
        require_equal(specialists[suite]["evaluation_environment"], reference_runtime, f"{suite} runtime")
        require_equal(specialists[suite]["evaluation_gpu"], reference_gpu, f"{suite} GPU")

    generalist_artifact = load_generalist_artifact_identity(
        args.artifacts_dir, "chi", 0, live_manifest
    )
    generalist = load_generalist_checkpoint_results(
        args.results_dir,
        "chi",
        0,
        live_manifest,
        generalist_artifact["checkpoint_sha256"],
        generalist_artifact["metadata"],
        generalist_artifact["metadata_sha256"],
    )
    require_equal(generalist["evaluation_environment"], reference_runtime, "generalist/specialist runtime")
    require_equal(generalist["evaluation_gpu"], reference_gpu, "generalist/specialist GPU")

    for suite in SUITES:
        for task in range(10):
            require_equal(
                specialists[suite]["task_protocol"][str(task)],
                generalist["task_protocol"][f"{suite}:{task}"],
                f"{suite} task {task} generalist/specialist protocol",
            )

    by_suite: dict[str, Any] = {}
    for suite in SUITES:
        specialist_rate = specialists[suite]["macro_task_success"]
        generalist_rate = generalist["suite_macro_success"][suite]
        by_suite[suite] = {
            "specialist": specialists[suite],
            "generalist_task_success": {
                str(task): generalist["task_success"][f"{suite}:{task}"]
                for task in range(10)
            },
            "generalist_macro_task_success": generalist_rate,
            "generalist_to_specialist_ratio": (
                generalist_rate / specialist_rate if specialist_rate > 0 else None
            ),
        }
    specialist_macro = statistics.mean(
        specialists[suite]["macro_task_success"] for suite in SUITES
    )
    generalist_macro = generalist["macro_task_success"]
    ratio = generalist_macro / specialist_macro if specialist_macro > 0 else None
    gate_pass = ratio is not None and ratio >= 0.85

    summary_end_sha256 = file_sha256(Path(__file__))
    imported_end = source_snapshot(SUMMARY_IMPORTED_SOURCES)
    validate_source_snapshot(imported_end, SUMMARY_IMPORTED_SOURCES)
    require_equal(summary_end_sha256, summary_start_sha256, "summary source start/end")
    require_equal(imported_end, imported_start, "summary imports start/end")
    manifest_end_sha256 = file_sha256(GENERALIST_MANIFEST_PATH)
    require_equal(
        manifest_end_sha256,
        GENERALIST_MANIFEST_SHA256,
        "frozen generalist manifest start/end",
    )
    specialist_source_files_end = {
        suite: validate_consumed_sources_unchanged(
            specialists[suite]["source_files"], f"{suite} specialist results"
        )
        for suite in SUITES
    }
    generalist_source_files_end = validate_consumed_sources_unchanged(
        generalist["source_files"], "generalist results"
    )
    specialist_cache_provenance_end = {
        suite: validate_cache_and_provenance(suite) for suite in SUITES
    }
    require_equal(
        specialist_cache_provenance_end,
        specialist_cache_provenance,
        "specialist cache provenance start/end",
    )
    generalist_cache_provenance_end = validate_generalist_cache_provenance(
        args.results_dir
    )
    require_equal(
        generalist_cache_provenance_end,
        generalist_cache_provenance,
        "generalist cache provenance start/end",
    )
    public_artifacts: dict[str, Any] = {}
    for suite, artifact in artifacts.items():
        checkpoint_end_sha256 = file_sha256(artifact["checkpoint_path"])
        metadata_end_sha256 = file_sha256(artifact["metadata_path"])
        require_equal(checkpoint_end_sha256, artifact["checkpoint_sha256"], f"{suite} checkpoint start/end")
        require_equal(metadata_end_sha256, artifact["metadata_sha256"], f"{suite} metadata start/end")
        public_artifacts[suite] = {
            "checkpoint_path": artifact["checkpoint_path"].as_posix(),
            "checkpoint_start_sha256": artifact["checkpoint_sha256"],
            "checkpoint_end_sha256": checkpoint_end_sha256,
            "checkpoint_archive_index_sha256": artifact["checkpoint_archive_index_sha256"],
            "metadata_path": artifact["metadata_path"].as_posix(),
            "metadata_start_sha256": artifact["metadata_sha256"],
            "metadata_end_sha256": metadata_end_sha256,
            "evaluation_results_start": specialists[suite]["source_files"],
            "evaluation_results_end": specialist_source_files_end[suite],
        }
    generalist_checkpoint_end_sha256 = file_sha256(
        generalist_artifact["checkpoint_path"]
    )
    generalist_metadata_end_sha256 = file_sha256(generalist_artifact["metadata_path"])
    require_equal(
        generalist_checkpoint_end_sha256,
        generalist_artifact["checkpoint_sha256"],
        "generalist checkpoint start/end",
    )
    require_equal(
        generalist_metadata_end_sha256,
        generalist_artifact["metadata_sha256"],
        "generalist metadata start/end",
    )
    output = {
        "scope": {
            "status": "separate_frozen_retention_certificate",
            "suites": list(SUITES),
            "seed": 0,
            "specialist_training_updates": 40000,
            "episodes_per_task": 50,
            "tasks_per_suite": 10,
            "specialist_trials": 2000,
            "generalist_trials": 2000,
            "averaging": "arithmetic mean of all 40 task success rates",
            "claim_boundary": (
                "This seed-0 retention ratio is reported separately. It does not alter "
                "or enter the frozen multi-seed generalist capability gates."
            ),
        },
        "identity": {
            "summary_start_sha256": summary_start_sha256,
            "summary_end_sha256": summary_end_sha256,
            "imported_sources_start": imported_start,
            "imported_sources_end": imported_end,
            "trainer_sha256": TRAINER_SHA256,
            "evaluator_sha256": EVALUATOR_SHA256,
            "evaluation_gpu_family": EVALUATION_GPU_FAMILY,
            "evaluation_gpu": reference_gpu,
            "evaluation_environment": reference_runtime,
            "specialist_cache_provenance": specialist_cache_provenance,
            "generalist_cache_provenance": generalist_cache_provenance,
            "generalist_cache_provenance_end": generalist_cache_provenance_end,
            "manifest_start_sha256": GENERALIST_MANIFEST_SHA256,
            "manifest_end_sha256": manifest_end_sha256,
            "specialist_artifacts": public_artifacts,
            "generalist_artifact": {
                "checkpoint_start_sha256": generalist_artifact["checkpoint_sha256"],
                "checkpoint_end_sha256": generalist_checkpoint_end_sha256,
                "metadata_start_sha256": generalist_artifact["metadata_sha256"],
                "metadata_end_sha256": generalist_metadata_end_sha256,
                "evaluation_results_start": generalist["source_files"],
                "evaluation_results_end": generalist_source_files_end,
            },
        },
        "by_suite": by_suite,
        "specialist_four_suite_task_macro": specialist_macro,
        "generalist_four_suite_task_macro": generalist_macro,
        "generalist_to_specialist_ratio": ratio,
        "gate": {
            "name": "seed0_generalist_retains_at_least_85pct_of_specialists",
            "threshold": 0.85,
            "pass": gate_pass,
            "definition": "generalist 40-task macro divided by specialist 40-task macro",
        },
        "generalist_capability_summary_modified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
