#!/usr/bin/env python3
"""Strict CPU summary for the frozen ViT full learned-forward certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from athena.libero_dataset_metadata import load_dataset_task_languages
from athena.run_vit_full_learned_forward_certificate import (
    EXPECTED_RUNTIME,
    FROZEN_CACHE,
    FROZEN_CAPABILITY_MANIFEST,
    FROZEN_CHECKPOINTS,
    FROZEN_DATASET_METADATA,
    FROZEN_PROVENANCE_JOB_ID,
    FULL_TASK_QUOTAS,
    PROTOCOL,
    SCHEMA,
    STAGE_NAMES,
    STAGE_SHAPES,
    rank_cache_inputs,
    source_hashes,
    state_dict_identity,
    validate_capability_manifest,
    validate_dataset_metadata,
    validate_model,
    validate_provenance,
    validated_official_tasks,
    write_json,
)
from athena.run_xvla_experiment import (
    build_vocab,
    file_sha256,
    load_cache_statistics,
    load_suite,
    make_config,
    task_languages,
)
from xvla.models.vla import ChiVLA


SUMMARY_SCHEMA = "xvla-vit-full-learned-forward-certificate-summary-v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
EXPECTED_CPU_RUNTIME = {
    "python_implementation": "CPython",
    "python_version": "3.10.19",
    "numpy_version": "1.26.4",
    "torch_version": "2.7.1+cu126",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--provenance-result", type=Path, required=True)
    parser.add_argument("--capability-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON value is not an object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} differs")


def require_close(label: str, actual: Any, expected: Any) -> None:
    actual_value = float(actual)
    expected_value = float(expected)
    if not (
        math.isfinite(actual_value)
        and math.isfinite(expected_value)
        and math.isclose(actual_value, expected_value, rel_tol=1e-12, abs_tol=1e-15)
    ):
        raise RuntimeError(f"{label} does not reproduce")


def require_sha256(label: str, value: Any) -> None:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not a SHA-256 digest")


def cpu_runtime_identity() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }


def checkpoint_paths(paths: list[Path]) -> dict[int, Path]:
    require(len(paths) == 3, "Exactly three checkpoint paths are required")
    result = {}
    for path in paths:
        matches = [
            seed for seed, frozen in FROZEN_CHECKPOINTS.items()
            if path.name == frozen["basename"]
        ]
        require(len(matches) == 1, f"Unexpected checkpoint path: {path}")
        seed = matches[0]
        require(seed not in result, f"Duplicate checkpoint seed {seed}")
        require_equal(
            f"checkpoint {seed} SHA",
            file_sha256(path),
            FROZEN_CHECKPOINTS[seed]["sha256"],
        )
        result[seed] = path
    require_equal("checkpoint seed set", set(result), {0, 1, 2})
    return result


def expected_inputs(
    records: list[Any], checkpoint_sha256: str, official_tasks: dict[int, str], encode: Any
) -> list[dict[str, Any]]:
    selected = rank_cache_inputs(records, checkpoint_sha256, FULL_TASK_QUOTAS)
    return [
        {
            **identity,
            "instruction": official_tasks[int(identity["official_task_index"])],
            "instruction_ids": encode(
                official_tasks[int(identity["official_task_index"])]
            ),
        }
        for identity in selected
    ]


def load_model_identity(
    checkpoint: Path, vocab_size: int, stats: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = make_config(
        PROTOCOL["architecture"],
        vocab_size,
        stats["state_dim"],
        stats["action_dim"],
        PROTOCOL["resolution"],
        PROTOCOL["action_horizon"],
        vision_encoder=PROTOCOL["vision_encoder"],
    )
    model = ChiVLA(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload, strict=True)
    model.eval()
    return state_dict_identity(model.state_dict()), validate_model(model)


def parse_action_array(label: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    require_equal(f"{label} shape", list(array.shape), [8, 7])
    require(bool(np.isfinite(array).all()), f"{label} is non-finite")
    return array


def canonical_array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def action_array_sha256(array: np.ndarray) -> str:
    return canonical_array_sha256(np.asarray(array, dtype=np.float64))


def recompute_action_errors(
    deployed: np.ndarray, reconstructed: np.ndarray
) -> dict[str, Any]:
    difference = deployed - reconstructed
    finite = bool(
        np.isfinite(deployed).all()
        and np.isfinite(reconstructed).all()
        and np.isfinite(difference).all()
    )
    if not finite:
        return {
            "finite": False,
            "max_abs_error": float("inf"),
            "relative_l2_error": float("inf"),
        }
    return {
        "finite": True,
        "max_abs_error": float(np.max(np.abs(difference))),
        "relative_l2_error": float(
            np.linalg.norm(difference) / max(np.linalg.norm(deployed), 1e-30)
        ),
    }


def validate_result(
    path: Path,
    result: dict[str, Any],
    checkpoint_by_seed: dict[int, Path],
    records: list[Any],
    official_tasks: dict[int, str],
    encode: Any,
    vocab_size: int,
    stats: dict[str, Any],
    provenance_sha256: str,
    provenance_canonical_sha256: str,
    capability_sha256: str,
    expected_sources: dict[str, str],
) -> dict[str, Any]:
    require_equal(f"{path} schema", result.get("schema"), SCHEMA)
    require_equal(f"{path} protocol", result.get("protocol"), PROTOCOL)
    require_equal(f"{path} mode", result.get("mode"), "full")
    identity = result.get("identity", {})
    seed = int(identity.get("checkpoint_seed", -1))
    require(seed in checkpoint_by_seed, f"{path} has an invalid seed")
    frozen = FROZEN_CHECKPOINTS[seed]
    require_equal(f"{path} checkpoint basename", identity.get("checkpoint_basename"), frozen["basename"])
    require_equal(f"{path} checkpoint SHA", identity.get("checkpoint_sha256"), frozen["sha256"])
    require_equal(f"{path} cache basename", identity.get("cache_basename"), FROZEN_CACHE["basename"])
    require_equal(f"{path} cache SHA", identity.get("cache_sha256"), FROZEN_CACHE["sha256"])
    require_equal(f"{path} cache frames", int(identity.get("cache_frames", -1)), FROZEN_CACHE["frames"])
    require_equal(f"{path} provenance SHA", identity.get("provenance_result_sha256"), provenance_sha256)
    require_equal(f"{path} provenance job", str(identity.get("provenance_job_id")), FROZEN_PROVENANCE_JOB_ID)
    require_equal(
        f"{path} provenance canonical SHA",
        identity.get("provenance_canonical_content_sha256"),
        provenance_canonical_sha256,
    )
    require_equal(f"{path} capability SHA", identity.get("capability_manifest_sha256"), capability_sha256)
    require_equal(
        f"{path} capability member rate",
        identity.get("capability_member_success_rate"),
        [0.852, 0.808, 0.9][seed],
    )
    require_equal(f"{path} dataset metadata", identity.get("dataset_metadata"), FROZEN_DATASET_METADATA)
    require_equal(f"{path} source hashes", identity.get("source_sha256"), expected_sources)
    runtime = result.get("runtime", {})
    for key, expected in EXPECTED_RUNTIME.items():
        require_equal(f"{path} runtime {key}", runtime.get(key), expected)
    require(isinstance(runtime.get("python_packages"), dict), f"{path} package runtime is absent")

    model_state, component_identity = load_model_identity(
        checkpoint_by_seed[seed], vocab_size, stats
    )
    require_equal(
        f"{path} checkpoint state identity",
        identity.get("checkpoint_state_identity"),
        model_state,
    )
    require_equal(
        f"{path} component identity",
        result.get("component_identity"),
        component_identity,
    )
    expected_input_rows = expected_inputs(
        records, frozen["sha256"], official_tasks, encode
    )
    require_equal(f"{path} selected inputs", result.get("inputs"), expected_input_rows)
    evaluations = result.get("evaluations")
    require(
        isinstance(evaluations, list) and len(evaluations) == 16,
        f"{path} must contain sixteen evaluations",
    )

    action_relatives = []
    action_absolutes = []
    stage_relatives = []
    for index, (evaluation, input_row) in enumerate(
        zip(evaluations, expected_input_rows, strict=True)
    ):
        require_equal(
            f"{path} evaluation {index} selection rank",
            int(evaluation.get("selection_rank", -1)),
            index,
        )
        require_equal(
            f"{path} evaluation {index} selection SHA",
            evaluation.get("selection_sha256"),
            input_row["selection_sha256"],
        )
        require(
            evaluation.get("manual_deployed_forward_exact") is True,
            f"{path} evaluation {index} manual traversal differs",
        )
        record = records[int(input_row["cache_row_index"])]
        image = (
            torch.from_numpy(np.asarray(record[2]))
            .permute(2, 0, 1)
            .float()
            .div(255)
            .numpy()
        )
        normalized_state = np.asarray(
            (
                np.asarray(record[3], dtype=np.float32) - stats["state_mean"]
            )
            / stats["state_std"],
            dtype=np.float32,
        )
        expected_prepared = {
            "image_float32_sha256": canonical_array_sha256(image),
            "instruction_int64_sha256": canonical_array_sha256(
                np.asarray(input_row["instruction_ids"], dtype=np.int64)
            ),
            "normalized_state_float32_sha256": canonical_array_sha256(
                normalized_state
            ),
            "embodiment_int64_sha256": canonical_array_sha256(
                np.zeros(1, dtype=np.int64)
            ),
        }
        prepared = evaluation.get("prepared_inputs", {})
        require_equal(
            f"{path} evaluation {index} prepared inputs",
            prepared,
            expected_prepared,
        )
        stages = evaluation.get("stages")
        require(isinstance(stages, list), f"{path} evaluation {index} stages are absent")
        require_equal(
            f"{path} evaluation {index} stage names",
            tuple(stage.get("name") for stage in stages),
            STAGE_NAMES,
        )
        for stage_index, stage in enumerate(stages):
            require(stage.get("finite") is True, f"{path} stage {index}:{stage_index} is non-finite")
            absolute = float(stage.get("max_abs_error", float("nan")))
            relative = float(stage.get("relative_l2_error", float("nan")))
            require(
                math.isfinite(absolute)
                and math.isfinite(relative)
                and absolute >= 0.0
                and relative >= 0.0,
                f"{path} stage {index}:{stage_index} has invalid errors",
            )
            require_equal(
                f"{path} stage {index}:{stage_index} shape",
                stage.get("shape"),
                STAGE_SHAPES[stage["name"]],
            )
            require_sha256(f"{path} stage {index}:{stage_index} reference SHA", stage.get("deployed_float32_as_float64_sha256"))
            require_sha256(f"{path} stage {index}:{stage_index} reconstruction SHA", stage.get("reconstruction_float64_sha256"))
            stage_relatives.append(relative)

        actions = evaluation.get("actions", {})
        require_equal(f"{path} action shape field {index}", actions.get("shape"), [8, 7])
        deployed = parse_action_array(
            f"{path} deployed action {index}", actions.get("deployed_float32_as_float64")
        )
        reconstructed = parse_action_array(
            f"{path} reconstruction action {index}", actions.get("reconstruction_float64")
        )
        require_equal(
            f"{path} deployed action SHA {index}",
            actions.get("deployed_float32_as_float64_sha256"),
            action_array_sha256(deployed),
        )
        require_equal(
            f"{path} reconstruction action SHA {index}",
            actions.get("reconstruction_float64_sha256"),
            action_array_sha256(reconstructed),
        )
        recomputed = recompute_action_errors(deployed, reconstructed)
        recorded = evaluation.get("end_to_end_action", {})
        require_equal(f"{path} action finite {index}", recorded.get("finite"), recomputed["finite"])
        require_close(f"{path} action max absolute {index}", recorded.get("max_abs_error"), recomputed["max_abs_error"])
        require_close(f"{path} action relative L2 {index}", recorded.get("relative_l2_error"), recomputed["relative_l2_error"])
        final_stage = stages[-1]
        require_equal(
            f"{path} final stage deployed SHA {index}",
            final_stage.get("deployed_float32_as_float64_sha256"),
            actions.get("deployed_float32_as_float64_sha256"),
        )
        require_equal(
            f"{path} final stage reconstruction SHA {index}",
            final_stage.get("reconstruction_float64_sha256"),
            actions.get("reconstruction_float64_sha256"),
        )
        require_close(f"{path} final stage max absolute {index}", final_stage.get("max_abs_error"), recomputed["max_abs_error"])
        require_close(f"{path} final stage relative L2 {index}", final_stage.get("relative_l2_error"), recomputed["relative_l2_error"])
        action_absolutes.append(float(recomputed["max_abs_error"]))
        action_relatives.append(float(recomputed["relative_l2_error"]))

    recomputed_aggregate = {
        "inputs_audited": 16,
        "stage_evaluations": 16 * len(STAGE_NAMES),
        "action_chunks": 16,
        "action_scalars": 16 * 8 * 7,
        "manual_deployed_forward_exact_for_all": True,
        "all_finite": True,
        "max_stage_relative_l2_error": max(stage_relatives),
        "max_action_absolute_error": max(action_absolutes),
        "max_action_relative_l2_error": max(action_relatives),
        "end_to_end_action_relative_l2_gate": PROTOCOL[
            "end_to_end_action_relative_l2_gate"
        ],
    }
    recomputed_aggregate["passed"] = (
        recomputed_aggregate["max_action_relative_l2_error"]
        <= PROTOCOL["end_to_end_action_relative_l2_gate"]
    )
    recorded_aggregate = result.get("aggregate", {})
    for field, expected in recomputed_aggregate.items():
        if isinstance(expected, float):
            require_close(f"{path} aggregate {field}", recorded_aggregate.get(field), expected)
        else:
            require_equal(f"{path} aggregate {field}", recorded_aggregate.get(field), expected)
    return {
        "seed": seed,
        "result_path": str(path),
        "result_sha256": file_sha256(path),
        "checkpoint_basename": frozen["basename"],
        "checkpoint_sha256": frozen["sha256"],
        "capability_member_success_rate": [0.852, 0.808, 0.9][seed],
        **recomputed_aggregate,
    }


def main() -> None:
    args = parse_args()
    cpu_runtime = cpu_runtime_identity()
    require_equal("summary CPU runtime", cpu_runtime, EXPECTED_CPU_RUNTIME)
    require(not args.output.exists(), "Refusing to overwrite the summary output")
    require(len(args.result) == 3, "Exactly three raw results are required")
    checkpoint_by_seed = checkpoint_paths(args.checkpoint)
    require_equal("cache basename", args.cache.name, FROZEN_CACHE["basename"])
    require_equal("cache SHA", file_sha256(args.cache), FROZEN_CACHE["sha256"])
    provenance = validate_provenance(
        args.provenance_result, args.cache, FROZEN_CACHE["sha256"]
    )
    capability = validate_capability_manifest(args.capability_manifest)
    capability_sha256 = file_sha256(args.capability_manifest)
    require_equal(
        "capability manifest SHA",
        capability_sha256,
        FROZEN_CAPABILITY_MANIFEST["sha256"],
    )
    provenance_sha256 = file_sha256(args.provenance_result)
    provenance_canonical_sha256 = provenance["cache"]["canonical_content_sha256"]
    sources_at_start = source_hashes()

    with args.cache.open("rb") as handle:
        records = pickle.load(handle)
    require_equal("cache frame count", len(records), FROZEN_CACHE["frames"])
    suite = load_suite(PROTOCOL["suite"])
    external_official_tasks = task_languages(suite)
    official_tasks = validated_official_tasks(capability, external_official_tasks)
    cache_tasks, dataset_metadata = load_dataset_task_languages(
        PROTOCOL["suite"], official_tasks
    )
    validate_dataset_metadata(dataset_metadata)
    vocab, encode = build_vocab(cache_tasks)
    stats = load_cache_statistics(args.cache, PROTOCOL["action_horizon"])

    rows = []
    seen_seeds = set()
    for path in args.result:
        row = validate_result(
            path,
            load_json(path),
            checkpoint_by_seed,
            records,
            official_tasks,
            encode,
            len(vocab),
            stats,
            provenance_sha256,
            provenance_canonical_sha256,
            capability_sha256,
            sources_at_start,
        )
        require(row["seed"] not in seen_seeds, f"Duplicate result seed {row['seed']}")
        seen_seeds.add(row["seed"])
        rows.append(row)
    require_equal("raw result seed set", seen_seeds, {0, 1, 2})
    rows.sort(key=lambda row: row["seed"])

    aggregate = {
        "checkpoints_audited": 3,
        "inputs_audited": sum(row["inputs_audited"] for row in rows),
        "stage_evaluations": sum(row["stage_evaluations"] for row in rows),
        "action_chunks": sum(row["action_chunks"] for row in rows),
        "action_scalars": sum(row["action_scalars"] for row in rows),
        "manual_deployed_forward_exact_for_all": all(
            row["manual_deployed_forward_exact_for_all"] for row in rows
        ),
        "all_finite": all(row["all_finite"] for row in rows),
        "max_stage_relative_l2_error": max(
            row["max_stage_relative_l2_error"] for row in rows
        ),
        "max_action_absolute_error": max(
            row["max_action_absolute_error"] for row in rows
        ),
        "max_action_relative_l2_error": max(
            row["max_action_relative_l2_error"] for row in rows
        ),
        "end_to_end_action_relative_l2_gate": PROTOCOL[
            "end_to_end_action_relative_l2_gate"
        ],
        "all_three_checkpoints_pass": all(row["passed"] for row in rows),
    }
    require_equal("summary source identity end", source_hashes(), sources_at_start)
    require_equal("summary CPU runtime end", cpu_runtime_identity(), cpu_runtime)
    require_equal("summary cache identity end", file_sha256(args.cache), FROZEN_CACHE["sha256"])
    require_equal("summary provenance identity end", file_sha256(args.provenance_result), provenance_sha256)
    require_equal("summary capability identity end", file_sha256(args.capability_manifest), capability_sha256)
    for seed, path in checkpoint_by_seed.items():
        require_equal(
            f"summary checkpoint {seed} identity end",
            file_sha256(path),
            FROZEN_CHECKPOINTS[seed]["sha256"],
        )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "protocol": PROTOCOL,
        "identity": {
            "source_sha256": sources_at_start,
            "cache_sha256": FROZEN_CACHE["sha256"],
            "provenance_result_sha256": provenance_sha256,
            "provenance_canonical_content_sha256": provenance_canonical_sha256,
            "capability_manifest_sha256": capability_sha256,
            "dataset_metadata": dataset_metadata,
            "checkpoint_sha256": {
                str(seed): frozen["sha256"]
                for seed, frozen in FROZEN_CHECKPOINTS.items()
            },
        },
        "checkpoints": rows,
        "aggregate": aggregate,
        "runtime": cpu_runtime,
        "scope": (
            "Input-conditioned full learned-forward numerical replay on 48 fixed corrected "
            "inputs across three capable ViT checkpoints. It is not a compact symbolic "
            "contraction, an input-general proof, or a closed-loop simulator certificate."
        ),
    }
    write_json(args.output, summary)
    print("RESULT", json.dumps(summary, indent=2), flush=True)
    if not aggregate["all_three_checkpoints_pass"]:
        raise RuntimeError("Frozen full learned-forward certificate did not pass")


if __name__ == "__main__":
    main()
