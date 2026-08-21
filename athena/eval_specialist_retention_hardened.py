#!/usr/bin/env python3
"""Evaluate one frozen specialist on canonical LIBERO initial states."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from athena.libero_dataset_metadata import load_dataset_task_languages
from athena.run_xvla_experiment import (
    build_encoder,
    load_cache_profile_sample,
    load_model,
    load_suite,
    profile_model,
    run_capability,
    task_languages,
    tensorize_sample,
)
from athena.specialist_retention_protocol import (
    ARCHITECTURE,
    CACHE_ENVIRONMENT_KEYS,
    EMA_DECAY,
    EVALUATION_GPU_FAMILY,
    EVALUATION_GPU_NAME,
    FULL_BATCH_SIZE,
    FULL_STEPS,
    HORIZON,
    LEARNING_RATE,
    MODEL_SOURCE_HASHES,
    RECIPE_VERSION,
    RESOLUTION,
    SEED,
    SHARDS,
    SMOKE_BATCH_SIZE,
    SMOKE_STEPS,
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
    validate_work_cache_environment,
)


PROTOCOL_SOURCE_SHA256 = "0bee62f9296496c4ee0943a0b74860c927c69d436bac468c7dd947500b7700eb"
TRAINER_SHA256 = "95e086b739359ba0cc653e3de13199c031ac22414bd217ff08113ea3c3129c27"
EVALUATION_IMPORTED_SOURCES = {
    **MODEL_SOURCE_HASHES,
    "athena/specialist_retention_protocol.py": PROTOCOL_SOURCE_SHA256,
}
TRAINING_IMPORTED_SOURCES = {
    **EVALUATION_IMPORTED_SOURCES,
    "xvla/train/train_lm.py": TRAIN_LM_SHA256,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--task-start", type=int)
    parser.add_argument("--task-end", type=int)
    return parser.parse_args()


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_training_metadata(
    metadata: dict[str, Any],
    suite: str,
    smoke: bool,
    live_task_metadata: dict[str, Any],
    dataset_tasks: dict[int, str],
    official_tasks: dict[int, str],
) -> None:
    spec = SUITE_SPECS[suite]
    expected = {
        "format": "xvla_specialist_retention_checkpoint_v1",
        "recipe_version": RECIPE_VERSION,
        "mode": "smoke" if smoke else "full",
        "architecture": ARCHITECTURE,
        "vision_encoder": VISION_ENCODER,
        "suite": suite,
        "seed": SEED,
        "steps": SMOKE_STEPS if smoke else FULL_STEPS,
        "batch_size": SMOKE_BATCH_SIZE if smoke else FULL_BATCH_SIZE,
        "lr": LEARNING_RATE,
        "ema_decay": EMA_DECAY,
        "res": RESOLUTION,
        "horizon": HORIZON,
        "state_dim": 8,
        "action_dim": 7,
        "sample_count": spec["sample_count"],
        "checkpoint": checkpoint_path(suite, smoke).as_posix(),
        "cache": spec["cache_path"],
        "cache_sha256": spec["cache_sha256"],
        "trainer_sha256": TRAINER_SHA256,
    }
    for key, value in expected.items():
        require_equal(metadata.get(key), value, f"{suite} training metadata {key}")
    checkpoint_sha256 = metadata.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise RuntimeError(f"{suite} checkpoint SHA-256 is invalid")
    require_equal(
        metadata.get("provenance"),
        validate_cache_and_provenance(suite),
        f"{suite} training provenance",
    )
    validate_task_metadata(suite, metadata.get("task_metadata", {}))
    require_equal(metadata.get("task_metadata"), live_task_metadata, f"{suite} live task metadata")
    require_equal(
        metadata.get("dataset_tasks"),
        {str(key): value for key, value in dataset_tasks.items()},
        f"{suite} dataset task languages",
    )
    require_equal(
        metadata.get("official_tasks"),
        {str(key): value for key, value in official_tasks.items()},
        f"{suite} official task languages",
    )
    vocab = metadata.get("vocab", {})
    require_equal(vocab.get("<pad>"), 0, f"{suite} padding token")
    require_equal(vocab.get("<bos>"), 1, f"{suite} BOS token")
    require_equal(sorted(vocab.values()), list(range(len(vocab))), f"{suite} vocabulary IDs")
    serialized_vocab = json.dumps(vocab, sort_keys=True, separators=(",", ":"))
    require_equal(
        metadata.get("vocab_sha256"),
        hashlib.sha256(serialized_vocab.encode()).hexdigest(),
        f"{suite} vocabulary SHA-256",
    )
    normalization = metadata.get("normalization", {})
    for key, size in (
        ("state_mean", 8),
        ("state_std", 8),
        ("action_mean", 7),
        ("action_std", 7),
    ):
        value = np.asarray(normalization.get(key), dtype=np.float64)
        require_equal(value.shape, (size,), f"{suite} normalization {key} shape")
        if not np.isfinite(value).all():
            raise RuntimeError(f"{suite} normalization {key} is non-finite")
        if key.endswith("std") and not (value > 0).all():
            raise RuntimeError(f"{suite} normalization {key} is not positive")
    source = metadata.get("source_identity", {})
    require_equal(source.get("expected_trainer_sha256"), TRAINER_SHA256, f"{suite} trainer pin")
    require_equal(source.get("trainer_start_sha256"), TRAINER_SHA256, f"{suite} trainer start")
    require_equal(source.get("trainer_end_sha256"), TRAINER_SHA256, f"{suite} trainer end")
    require_equal(
        source.get("imported_sources_start"),
        TRAINING_IMPORTED_SOURCES,
        f"{suite} trainer sources start",
    )
    require_equal(
        source.get("imported_sources_end"),
        TRAINING_IMPORTED_SOURCES,
        f"{suite} trainer sources end",
    )
    training_bundle = bundle_sha256(TRAINING_IMPORTED_SOURCES)
    require_equal(
        source.get("imported_bundle_start_sha256"), training_bundle, f"{suite} trainer bundle start"
    )
    require_equal(
        source.get("imported_bundle_end_sha256"), training_bundle, f"{suite} trainer bundle end"
    )
    require_equal(
        source.get("cache_identity_start"),
        metadata.get("provenance"),
        f"{suite} trainer cache start",
    )
    require_equal(
        source.get("cache_identity_end"),
        metadata.get("provenance"),
        f"{suite} trainer cache end",
    )
    require_equal(source.get("task_metadata_start"), live_task_metadata, f"{suite} mapping start")
    require_equal(source.get("task_metadata_end"), live_task_metadata, f"{suite} mapping end")
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
            raise RuntimeError(f"{suite} trainer cache environment {key} is not /work-only")
    environment = metadata.get("training_environment", {})
    require_equal(environment.get("matmul_precision"), "high", f"{suite} train precision")
    require_equal(environment.get("autocast_dtype"), "bfloat16", f"{suite} train autocast")
    for key in ("torch_version", "cuda_version", "gpu"):
        if not environment.get(key):
            raise RuntimeError(f"{suite} training environment lacks {key}")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    smoke = args.mode == "smoke"
    if smoke:
        if args.task_start is not None or args.task_end is not None:
            raise ValueError("Smoke evaluation derives its frozen task range")
        task_start, task_end = 0, 1
        eps_per_task, max_steps, profile_iters = 1, 8, 2
    else:
        if (args.task_start, args.task_end) not in SHARDS:
            raise ValueError(f"Full evaluation requires one frozen shard, got {(args.task_start, args.task_end)}")
        task_start, task_end = args.task_start, args.task_end
        eps_per_task, max_steps, profile_iters = 50, 280, 20
    checkpoint = checkpoint_path(args.suite, smoke)
    metadata_file = metadata_path(args.suite, smoke)
    output = evaluation_result_path(args.suite, task_start, task_end, smoke)
    if output.exists() or output.with_suffix(output.suffix + ".tmp").exists():
        raise FileExistsError(f"Refusing to overwrite hardened evaluation {output}")

    evaluator_start_sha256 = file_sha256(Path(__file__))
    expected_evaluator_sha256 = os.environ.get("XVLA_FROZEN_SPECIALIST_EVALUATOR_SHA256")
    if expected_evaluator_sha256 is None:
        raise RuntimeError("XVLA_FROZEN_SPECIALIST_EVALUATOR_SHA256 is required")
    require_equal(
        evaluator_start_sha256,
        expected_evaluator_sha256,
        "specialist evaluator wrapper pin",
    )
    require_equal(
        os.environ.get("XVLA_FROZEN_SPECIALIST_GPU_FAMILY"),
        EVALUATION_GPU_FAMILY,
        "specialist evaluation GPU family",
    )
    require_equal(
        torch.cuda.get_device_name(0), EVALUATION_GPU_NAME, "specialist evaluation GPU"
    )
    imported_start = source_snapshot(EVALUATION_IMPORTED_SOURCES)
    validate_source_snapshot(imported_start, EVALUATION_IMPORTED_SOURCES)
    cache_environment_start = validate_work_cache_environment()
    cache_identity_start = validate_cache_and_provenance(args.suite)
    checkpoint_start_sha256 = file_sha256(checkpoint)
    metadata_start_sha256 = file_sha256(metadata_file)
    metadata = json.loads(metadata_file.read_text())

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("highest")
    suite = load_suite(args.suite)
    official_tasks = task_languages(suite)
    dataset_tasks, live_task_metadata_start = load_dataset_task_languages(
        args.suite, official_tasks
    )
    validate_task_metadata(args.suite, live_task_metadata_start)
    validate_training_metadata(
        metadata,
        args.suite,
        smoke,
        live_task_metadata_start,
        dataset_tasks,
        official_tasks,
    )
    require_equal(
        checkpoint_start_sha256,
        metadata["checkpoint_sha256"],
        f"{args.suite} live checkpoint hash",
    )
    stats = load_cache_profile_sample(Path(SUITE_SPECS[args.suite]["cache_path"]), HORIZON)
    require_equal(stats["frame_count"], SUITE_SPECS[args.suite]["frame_count"], "cache frames")
    stats.update(metadata["normalization"])
    stats["sample_count"] = metadata["sample_count"]
    encode = build_encoder(metadata["vocab"])
    runtime_args = argparse.Namespace(
        architecture=ARCHITECTURE,
        vision_encoder=VISION_ENCODER,
        checkpoint=checkpoint,
        res=RESOLUTION,
        horizon=HORIZON,
        task_start=task_start,
        task_end=task_end,
        eps_per_task=eps_per_task,
        max_steps=max_steps,
        num_steps_wait=10,
        exec_h=8,
    )
    model = load_model(runtime_args, len(metadata["vocab"]), stats)
    require_equal(model.num_params(), metadata["parameters"], f"{args.suite} parameters")
    sample_tensors = tensorize_sample(stats["first_sample"], encode, dataset_tasks, stats)
    with torch.inference_mode():
        prediction, _ = model(*sample_tensors)
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Specialist checkpoint produced a non-finite prediction")
    profile = profile_model(model, sample_tensors, profile_iters)
    require_equal(profile.get("gpu"), EVALUATION_GPU_NAME, "profile GPU")
    capability = run_capability(
        runtime_args, model, suite, official_tasks, encode, stats
    )
    numerical_environment = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "libero_version": installed_version("libero"),
        "robosuite_version": installed_version("robosuite"),
        "mujoco_version": installed_version("mujoco"),
    }

    live_task_languages_end, live_task_metadata_end = load_dataset_task_languages(
        args.suite, official_tasks
    )
    require_equal(live_task_languages_end, dataset_tasks, "dataset task languages start/end")
    validate_task_metadata(args.suite, live_task_metadata_end)
    evaluator_end_sha256 = file_sha256(Path(__file__))
    imported_end = source_snapshot(EVALUATION_IMPORTED_SOURCES)
    validate_source_snapshot(imported_end, EVALUATION_IMPORTED_SOURCES)
    cache_environment_end = validate_work_cache_environment()
    cache_identity_end = validate_cache_and_provenance(args.suite)
    checkpoint_end_sha256 = file_sha256(checkpoint)
    metadata_end_sha256 = file_sha256(metadata_file)
    start_identity = {
        "evaluator_sha256": evaluator_start_sha256,
        "evaluation_gpu_family": EVALUATION_GPU_FAMILY,
        "evaluation_gpu_name": EVALUATION_GPU_NAME,
        "imported_sources": imported_start,
        "imported_bundle_sha256": bundle_sha256(imported_start),
        "cache_environment": cache_environment_start,
        "cache_identity": cache_identity_start,
        "task_metadata": live_task_metadata_start,
        "checkpoint_sha256": checkpoint_start_sha256,
        "metadata_sha256": metadata_start_sha256,
    }
    end_identity = {
        "evaluator_sha256": evaluator_end_sha256,
        "evaluation_gpu_family": EVALUATION_GPU_FAMILY,
        "evaluation_gpu_name": torch.cuda.get_device_name(0),
        "imported_sources": imported_end,
        "imported_bundle_sha256": bundle_sha256(imported_end),
        "cache_environment": cache_environment_end,
        "cache_identity": cache_identity_end,
        "task_metadata": live_task_metadata_end,
        "checkpoint_sha256": checkpoint_end_sha256,
        "metadata_sha256": metadata_end_sha256,
    }
    require_equal(end_identity, start_identity, "specialist evaluation start/end identity")
    result = {
        "format": "xvla_specialist_retention_evaluation_v1",
        "mode": "smoke" if smoke else "capability",
        "architecture": ARCHITECTURE,
        "vision_encoder": VISION_ENCODER,
        "suite": args.suite,
        "training_suite": args.suite,
        "seed": SEED,
        "checkpoint": checkpoint.as_posix(),
        "model_metadata": metadata_file.as_posix(),
        "cache": SUITE_SPECS[args.suite]["cache_path"],
        "matmul_precision": torch.get_float32_matmul_precision(),
        "evaluation_protocol": {
            "res": RESOLUTION,
            "horizon": HORIZON,
            "num_steps_wait": 10,
            "exec_h": 8,
            "eps_per_task": eps_per_task,
            "max_steps": max_steps,
            "task_start": task_start,
            "task_end": task_end,
        },
        "prediction_finite": True,
        "prediction_shape": list(prediction.shape),
        "profile": profile,
        "numerical_environment": numerical_environment,
        "training_metadata": metadata,
        "capability": capability,
        "source_identity": {
            "expected_evaluator_sha256": expected_evaluator_sha256,
            "start": start_identity,
            "end": end_identity,
        },
        "claim_boundary": (
            "In-domain seed-0 specialist evaluation used only as the denominator "
            "for the separately frozen generalist-retention ratio."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    print("RESULT", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
