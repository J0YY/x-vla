#!/usr/bin/env python3
"""Strictly validate and aggregate the frozen ViT modality audit."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from athena.libero_dataset_metadata import load_dataset_task_languages
from athena.run_vit_modality_contribution_audit import (
    FROZEN_CACHE,
    FROZEN_CAPABILITY_MANIFEST,
    FROZEN_CHECKPOINTS,
    FROZEN_DATASET_METADATA,
    FROZEN_PROVENANCE_BASENAME,
    FROZEN_PROVENANCE_JOB_ID,
    PROTOCOL,
    SCHEMA,
    SOURCE_GROUPS,
    file_sha256,
    hash_array,
    rank_cache_inputs,
    source_hashes,
    validate_capability_manifest,
    validate_dataset_metadata,
    validate_provenance,
    vector_comparison,
    write_json,
)
from athena.run_xvla_experiment import build_vocab, load_suite, task_languages

SUMMARY_SCHEMA = "xvla-vit-modality-contribution-audit-summary-v1"
HEAD_METRICS = (
    "coherent_energy_fraction",
    "token_energy_fraction",
    "per_source_token_energy_fraction",
    "signed_projection_fraction",
    "cosine_with_total",
)
MODULE_METRICS = (
    "coherent_energy_fraction",
    "signed_projection_fraction",
    "cosine_with_total",
)
EXPECTED_VISIBLE_INCIDENCES = {
    "vision": 512,
    "instruction": 264,
    "robot_state": 16,
    "action_query": 36,
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
        return json.load(handle)


def assert_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} differs from the frozen value")


def assert_close(
    label: str, actual: Any, expected: Any, tolerance: float = 1e-10
) -> None:
    actual_value = float(actual)
    expected_value = float(expected)
    if not math.isfinite(actual_value) or not math.isfinite(expected_value):
        raise RuntimeError(f"{label} is non-finite")
    scale = max(1.0, abs(actual_value), abs(expected_value))
    if abs(actual_value - expected_value) > tolerance * scale:
        raise RuntimeError(
            f"{label} differs: actual={actual_value:.17g}, expected={expected_value:.17g}"
        )


def validate_error_row(label: str, row: dict[str, Any]) -> tuple[bool, float, float]:
    if set(row) != {"finite", "max_abs_error", "relative_l2_error"}:
        raise RuntimeError(f"{label} has an unexpected error schema")
    finite = row.get("finite") is True
    maximum = float(row.get("max_abs_error", float("nan")))
    relative = float(row.get("relative_l2_error", float("nan")))
    numerical_finite = math.isfinite(maximum) and math.isfinite(relative)
    if finite != numerical_finite or (
        numerical_finite and (maximum < 0 or relative < 0)
    ):
        raise RuntimeError(f"{label} has inconsistent error values")
    return finite, maximum, relative


def expected_inputs(
    records: list[Any],
    checkpoint_sha256: str,
    official_tasks: dict[int, str],
    encode: Any,
) -> list[dict[str, Any]]:
    rows = rank_cache_inputs(
        records, checkpoint_sha256, PROTOCOL["full_official_task_quotas"]
    )
    return [
        {
            **row,
            "instruction": official_tasks[int(row["official_task_index"])],
            "instruction_ids": encode(official_tasks[int(row["official_task_index"])]),
        }
        for row in rows
    ]


def validate_source_metrics(
    label: str,
    sources: dict[str, Any],
    denominators: dict[str, Any],
    include_token_metrics: bool,
) -> None:
    assert_equal(f"{label} source names", set(sources), set(SOURCE_GROUPS))
    required_denominators = {
        "coherent_energy_denominator",
        "signed_projection_denominator",
    }
    if include_token_metrics:
        required_denominators.update(
            {"token_energy_denominator", "per_source_token_energy_denominator"}
        )
    assert_equal(f"{label} denominator names", set(denominators), required_denominators)
    for name, value in denominators.items():
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise RuntimeError(f"{label} denominator {name} is not finite and positive")

    coherent_fraction_sum = 0.0
    signed_fraction_sum = 0.0
    token_fraction_sum = 0.0
    per_token_fraction_sum = 0.0
    for source in SOURCE_GROUPS:
        row = sources[source]
        required = {
            "coherent_energy",
            "coherent_energy_fraction",
            "signed_projection_numerator",
            "signed_projection_fraction",
            "cosine_denominator",
            "cosine_with_total",
        }
        if include_token_metrics:
            required.update(
                {
                    "token_energy",
                    "token_energy_fraction",
                    "mean_visible_source_token_energy",
                    "per_source_token_energy_fraction",
                    "unique_source_tokens",
                    "visible_source_token_incidences",
                }
            )
        assert_equal(f"{label} {source} metric names", set(row), required)
        for name, value in row.items():
            if name in {"unique_source_tokens", "visible_source_token_incidences"}:
                continue
            if not math.isfinite(float(value)):
                raise RuntimeError(f"{label} {source} {name} is non-finite")
        coherent = float(row["coherent_energy"])
        if coherent < 0:
            raise RuntimeError(f"{label} {source} coherent energy is negative")
        assert_close(
            f"{label} {source} coherent fraction",
            row["coherent_energy_fraction"],
            coherent / float(denominators["coherent_energy_denominator"]),
        )
        assert_close(
            f"{label} {source} signed projection",
            row["signed_projection_fraction"],
            float(row["signed_projection_numerator"])
            / float(denominators["signed_projection_denominator"]),
        )
        assert_close(
            f"{label} {source} cosine",
            row["cosine_with_total"],
            float(row["signed_projection_numerator"])
            / float(row["cosine_denominator"]),
        )
        coherent_fraction_sum += float(row["coherent_energy_fraction"])
        signed_fraction_sum += float(row["signed_projection_fraction"])
        if include_token_metrics:
            expected_tokens = int(PROTOCOL["token_layout"][source]["tokens"])
            assert_equal(
                f"{label} {source} unique token count",
                int(row["unique_source_tokens"]),
                expected_tokens,
            )
            assert_equal(
                f"{label} {source} visible incidences",
                int(row["visible_source_token_incidences"]),
                EXPECTED_VISIBLE_INCIDENCES[source],
            )
            token_energy = float(row["token_energy"])
            if token_energy < 0:
                raise RuntimeError(f"{label} {source} token energy is negative")
            assert_close(
                f"{label} {source} token fraction",
                row["token_energy_fraction"],
                token_energy / float(denominators["token_energy_denominator"]),
            )
            mean_token_energy = token_energy / EXPECTED_VISIBLE_INCIDENCES[source]
            assert_close(
                f"{label} {source} mean token energy",
                row["mean_visible_source_token_energy"],
                mean_token_energy,
            )
            assert_close(
                f"{label} {source} per-token fraction",
                row["per_source_token_energy_fraction"],
                mean_token_energy
                / float(denominators["per_source_token_energy_denominator"]),
            )
            token_fraction_sum += float(row["token_energy_fraction"])
            per_token_fraction_sum += float(row["per_source_token_energy_fraction"])
    assert_close(f"{label} coherent fractions sum", coherent_fraction_sum, 1.0)
    assert_close(f"{label} signed projections sum", signed_fraction_sum, 1.0, 1e-7)
    if include_token_metrics:
        assert_close(f"{label} token fractions sum", token_fraction_sum, 1.0)
        assert_close(f"{label} per-token fractions sum", per_token_fraction_sum, 1.0)


def validate_prompt_rows(
    path: Path,
    rows: list[dict[str, Any]],
    inputs: list[dict[str, Any]],
    official_tasks: dict[int, str],
    encode: Any,
) -> dict[str, list[float]]:
    if len(rows) != 100:
        raise RuntimeError(f"{path} has {len(rows)} prompt rows instead of 100")
    first_by_task = {
        task: next(
            row
            for row in inputs
            if row["official_task_index"] == task
            and row["selection_rank_within_official_task"] == 0
        )
        for task in range(10)
    }
    predictions: dict[tuple[int, int], np.ndarray] = {}
    row_catalog: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        observation_task = int(row.get("observation_official_task_index", -1))
        prompt_task = int(row.get("prompt_official_task_index", -1))
        key = (observation_task, prompt_task)
        if (
            key in row_catalog
            or observation_task not in range(10)
            or prompt_task not in range(10)
        ):
            raise RuntimeError(f"{path} prompt catalog is duplicate or out of range")
        expected_input = first_by_task[observation_task]
        assert_equal(
            f"{path} prompt observation rank",
            int(row.get("observation_selection_rank", -1)),
            expected_input["selection_rank"],
        )
        assert_equal(
            f"{path} prompt observation SHA",
            row.get("observation_selection_sha256"),
            expected_input["selection_sha256"],
        )
        assert_equal(
            f"{path} prompt match flag",
            row.get("prompt_matches_observation"),
            observation_task == prompt_task,
        )
        assert_equal(
            f"{path} prompt text", row.get("instruction"), official_tasks[prompt_task]
        )
        ids = encode(official_tasks[prompt_task])
        assert_equal(f"{path} prompt IDs", row.get("instruction_ids"), ids)
        assert_equal(
            f"{path} prompt ID SHA",
            row.get("instruction_ids_sha256"),
            hash_array(np.asarray(ids, dtype=np.int64)),
        )
        prediction = np.asarray(row.get("physical_action_prediction"), dtype=np.float64)
        if prediction.shape != (8, 7) or not np.isfinite(prediction).all():
            raise RuntimeError(f"{path} prompt prediction is invalid")
        assert_equal(
            f"{path} prompt prediction SHA",
            row.get("physical_action_prediction_sha256"),
            hash_array(prediction),
        )
        predictions[key] = prediction
        row_catalog[key] = row
    expected_keys = {
        (observation, prompt) for observation in range(10) for prompt in range(10)
    }
    assert_equal(f"{path} prompt key set", set(row_catalog), expected_keys)
    relative_changes = []
    mismatched_relative_changes = []
    for observation_task, prompt_task in sorted(expected_keys):
        matched = predictions[(observation_task, observation_task)]
        candidate = predictions[(observation_task, prompt_task)]
        expected = vector_comparison(matched, candidate)
        actual = row_catalog[(observation_task, prompt_task)].get(
            "comparison_to_matched_prompt", {}
        )
        assert_equal(f"{path} prompt comparison names", set(actual), set(expected))
        for name, value in expected.items():
            assert_close(
                f"{path} prompt comparison {observation_task}/{prompt_task}/{name}",
                actual[name],
                value,
            )
        relative_changes.append(float(actual["relative_l2_change"]))
        if observation_task != prompt_task:
            mismatched_relative_changes.append(float(actual["relative_l2_change"]))
    return {
        "including_matched": relative_changes,
        "mismatched_only": mismatched_relative_changes,
    }


def validate_result(
    path: Path,
    result: dict[str, Any],
    checkpoint_paths: dict[int, Path],
    records: list[Any],
    official_tasks: dict[int, str],
    encode: Any,
    provenance_sha256: str,
    provenance_canonical_sha256: str,
    source_sha256: dict[str, str],
) -> dict[str, Any]:
    assert_equal(f"{path} schema", result.get("schema"), SCHEMA)
    assert_equal(f"{path} protocol", result.get("protocol"), PROTOCOL)
    assert_equal(f"{path} mode", result.get("mode"), "full")
    identity = result.get("identity", {})
    seed = int(identity.get("checkpoint_seed", -1))
    if seed not in FROZEN_CHECKPOINTS:
        raise RuntimeError(f"{path} has an invalid checkpoint seed")
    checkpoint = FROZEN_CHECKPOINTS[seed]
    assert_equal(
        f"{path} checkpoint basename",
        identity.get("checkpoint_basename"),
        checkpoint["basename"],
    )
    assert_equal(
        f"{path} checkpoint path basename",
        Path(str(identity.get("checkpoint_path", ""))).name,
        checkpoint["basename"],
    )
    assert_equal(
        f"{path} checkpoint SHA",
        identity.get("checkpoint_sha256"),
        checkpoint["sha256"],
    )
    if file_sha256(checkpoint_paths[seed]) != checkpoint["sha256"]:
        raise RuntimeError(f"Live checkpoint {seed} SHA differs")
    assert_equal(
        f"{path} cache basename",
        identity.get("cache_basename"),
        FROZEN_CACHE["basename"],
    )
    assert_equal(
        f"{path} cache SHA", identity.get("cache_sha256"), FROZEN_CACHE["sha256"]
    )
    assert_equal(
        f"{path} cache frames",
        int(identity.get("cache_frames", -1)),
        FROZEN_CACHE["frames"],
    )
    assert_equal(
        f"{path} provenance basename",
        Path(str(identity.get("provenance_result", ""))).name,
        FROZEN_PROVENANCE_BASENAME,
    )
    assert_equal(
        f"{path} provenance SHA",
        identity.get("provenance_result_sha256"),
        provenance_sha256,
    )
    assert_equal(
        f"{path} provenance job",
        str(identity.get("provenance_job_id", "")),
        FROZEN_PROVENANCE_JOB_ID,
    )
    assert_equal(
        f"{path} provenance canonical SHA",
        identity.get("provenance_canonical_content_sha256"),
        provenance_canonical_sha256,
    )
    assert_equal(
        f"{path} capability manifest SHA",
        identity.get("capability_manifest_sha256"),
        FROZEN_CAPABILITY_MANIFEST["sha256"],
    )
    assert_equal(
        f"{path} dataset metadata",
        identity.get("dataset_metadata"),
        FROZEN_DATASET_METADATA,
    )
    assert_equal(
        f"{path} official languages",
        {
            int(key): value
            for key, value in identity.get("official_task_languages", {}).items()
        },
        official_tasks,
    )
    assert_equal(f"{path} source hashes", identity.get("source_sha256"), source_sha256)
    runtime = result.get("runtime", {})
    assert_equal(
        f"{path} matmul precision",
        runtime.get("matmul_precision"),
        PROTOCOL["matmul_precision"],
    )
    if not runtime.get("torch_version") or not runtime.get("gpu"):
        raise RuntimeError(f"{path} lacks runtime identity")

    layout = result.get("layout_confirmation", {})
    assert_equal(
        f"{path} deployed hidden equality",
        layout.get("deployed_manual_hidden_exact"),
        True,
    )
    assert_equal(
        f"{path} causal mask equality", layout.get("deployed_causal_mask_exact"), True
    )
    assert_close(f"{path} layout max error", layout.get("max_abs_error"), 0.0)
    assert_close(f"{path} layout relative error", layout.get("relative_l2_error"), 0.0)
    assert_equal(f"{path} layout length", int(layout.get("sequence_length", -1)), 107)
    assert_equal(
        f"{path} layout groups",
        layout.get("source_groups"),
        {group: PROTOCOL["token_layout"][group] for group in SOURCE_GROUPS},
    )

    inputs = expected_inputs(records, checkpoint["sha256"], official_tasks, encode)
    assert_equal(f"{path} selected inputs", result.get("inputs"), inputs)
    module_rows = result.get("module_rows")
    if not isinstance(module_rows, list) or len(module_rows) != 128 * 8:
        raise RuntimeError(f"{path} has an invalid module row count")
    error_relatives = []
    contribution_records = []
    module_records = []
    row_index = 0
    for input_row in inputs:
        for block_index in range(8):
            row = module_rows[row_index]
            row_index += 1
            prefix = f"{path} input {input_row['selection_rank']} block {block_index}"
            assert_equal(
                f"{prefix} selection rank",
                int(row.get("selection_rank", -1)),
                input_row["selection_rank"],
            )
            assert_equal(
                f"{prefix} selection SHA",
                row.get("selection_sha256"),
                input_row["selection_sha256"],
            )
            assert_equal(
                f"{prefix} official task",
                int(row.get("official_task_index", -1)),
                input_row["official_task_index"],
            )
            assert_equal(
                f"{prefix} block", int(row.get("block_index", -1)), block_index
            )
            assert_equal(
                f"{prefix} sequence length", int(row.get("sequence_length", -1)), 107
            )
            assert_equal(
                f"{prefix} query rows", row.get("action_query_rows"), [99, 107]
            )
            assert_equal(f"{prefix} model dim", int(row.get("model_dim", -1)), 384)
            assert_equal(f"{prefix} head dim", int(row.get("head_dim", -1)), 32)
            assert_equal(f"{prefix} head count", int(row.get("head_count", -1)), 12)
            heads = row.get("heads")
            if not isinstance(heads, list) or len(heads) != 12:
                raise RuntimeError(f"{prefix} has an invalid head catalog")
            for head_index, head in enumerate(heads):
                head_prefix = f"{prefix} head {head_index}"
                assert_equal(
                    f"{head_prefix} index", int(head.get("head_index", -1)), head_index
                )
                finite, _, relative = validate_error_row(
                    head_prefix, head.get("group_sum_error", {})
                )
                if not finite:
                    raise RuntimeError(f"{head_prefix} is non-finite")
                error_relatives.append(relative)
                validate_source_metrics(
                    head_prefix,
                    head.get("sources", {}),
                    head.get("denominators", {}),
                    True,
                )
                for source in SOURCE_GROUPS:
                    contribution_records.append(
                        {
                            "seed": seed,
                            "block": block_index,
                            "head": head_index,
                            "source": source,
                            **{
                                metric: float(head["sources"][source][metric])
                                for metric in HEAD_METRICS
                            },
                        }
                    )
            module = row.get("module", {})
            for error_name in (
                "group_sum_prebias_error",
                "group_sum_plus_bias_error",
                "deployed_float32_vs_grouped_float64_error",
                "deployed_gain_scaled_update_error",
            ):
                finite, _, relative = validate_error_row(
                    f"{prefix} {error_name}", module.get(error_name, {})
                )
                if not finite:
                    raise RuntimeError(f"{prefix} {error_name} is non-finite")
                error_relatives.append(relative)
            residual_gain = float(module.get("attention_residual_gain", float("nan")))
            if not math.isfinite(residual_gain):
                raise RuntimeError(f"{prefix} has a non-finite attention residual gain")
            bias_energy = float(
                module.get("output_projection_bias_energy", float("nan"))
            )
            if not math.isfinite(bias_energy) or bias_energy < 0:
                raise RuntimeError(f"{prefix} has an invalid output bias energy")
            validate_source_metrics(
                prefix + " module",
                module.get("sources", {}),
                module.get("denominators", {}),
                False,
            )
            for source in SOURCE_GROUPS:
                module_records.append(
                    {
                        "seed": seed,
                        "block": block_index,
                        "source": source,
                        **{
                            metric: float(module["sources"][source][metric])
                            for metric in MODULE_METRICS
                        },
                    }
                )

    prompt_relative_changes = validate_prompt_rows(
        path, result.get("prompt_permutation_rows", []), inputs, official_tasks, encode
    )
    maximum = max(error_relatives)
    passed = maximum <= float(PROTOCOL["relative_l2_gate"])
    aggregate = result.get("aggregate", {})
    assert_equal(
        f"{path} aggregate inputs", int(aggregate.get("inputs_audited", -1)), 128
    )
    assert_equal(
        f"{path} task counts",
        aggregate.get("official_task_counts"),
        PROTOCOL["full_official_task_quotas"],
    )
    assert_equal(
        f"{path} module count", int(aggregate.get("module_evaluations", -1)), 1024
    )
    assert_equal(
        f"{path} head count", int(aggregate.get("head_evaluations", -1)), 12288
    )
    assert_equal(
        f"{path} source count",
        int(aggregate.get("source_group_evaluations", -1)),
        49152,
    )
    assert_equal(
        f"{path} prompt count",
        int(aggregate.get("prompt_permutation_evaluations", -1)),
        100,
    )
    assert_equal(f"{path} finite flag", aggregate.get("all_finite"), True)
    assert_equal(f"{path} prompt finite flag", aggregate.get("prompt_all_finite"), True)
    assert_close(
        f"{path} maximum relative error",
        aggregate.get("max_relative_l2_error"),
        maximum,
    )
    assert_close(
        f"{path} fidelity gate",
        aggregate.get("relative_l2_gate"),
        PROTOCOL["relative_l2_gate"],
    )
    assert_equal(f"{path} pass flag", aggregate.get("passed"), passed)
    if not passed:
        raise RuntimeError(f"{path} fails the frozen algebraic fidelity gate")
    if not isinstance(result.get("scope"), str) or not result["scope"]:
        raise RuntimeError(f"{path} lacks a scope boundary")
    return {
        "seed": seed,
        "source_file": str(path),
        "source_file_sha256": file_sha256(path),
        "checkpoint_basename": checkpoint["basename"],
        "checkpoint_sha256": checkpoint["sha256"],
        "inputs_audited": 128,
        "module_evaluations": 1024,
        "head_evaluations": 12288,
        "source_group_evaluations": 49152,
        "prompt_permutation_evaluations": 100,
        "max_relative_l2_error": maximum,
        "passed": passed,
        "contribution_records": contribution_records,
        "module_records": module_records,
        "prompt_relative_changes": prompt_relative_changes,
    }


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("Cannot summarize an empty or non-finite distribution")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def grouped_distributions(
    records: list[dict[str, Any]],
    metrics: tuple[str, ...],
    key: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    catalog: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        catalog[key(row)].append(row)
    output = {}
    for group, rows in sorted(catalog.items()):
        output[group] = {
            source: {
                metric: distribution(
                    [row[metric] for row in rows if row["source"] == source]
                )
                for metric in metrics
            }
            for source in SOURCE_GROUPS
        }
    return output


def main() -> None:
    args = parse_args()
    if len(args.result) != 3 or len({path.resolve() for path in args.result}) != 3:
        raise RuntimeError("Exactly three distinct full results are required")
    if (
        len(args.checkpoint) != 3
        or len({path.resolve() for path in args.checkpoint}) != 3
    ):
        raise RuntimeError("Exactly three distinct checkpoint files are required")
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite existing summary: {args.output}")
    if (
        args.cache.name != FROZEN_CACHE["basename"]
        or file_sha256(args.cache) != FROZEN_CACHE["sha256"]
    ):
        raise RuntimeError("Summary cache identity differs")
    validate_capability_manifest(args.capability_manifest)
    provenance = validate_provenance(
        args.provenance_result, args.cache, FROZEN_CACHE["sha256"]
    )
    provenance_sha256 = file_sha256(args.provenance_result)
    provenance_canonical_sha256 = provenance["cache"]["canonical_content_sha256"]
    expected_source_hashes = source_hashes()
    checkpoint_paths = {}
    for path in args.checkpoint:
        seeds = [
            seed
            for seed, row in FROZEN_CHECKPOINTS.items()
            if row["basename"] == path.name
        ]
        if len(seeds) != 1:
            raise RuntimeError(f"Unexpected checkpoint path {path}")
        checkpoint_paths[seeds[0]] = path
    assert_equal(
        "checkpoint seed coverage", set(checkpoint_paths), set(FROZEN_CHECKPOINTS)
    )

    with args.cache.open("rb") as handle:
        records = pickle.load(handle)
    if len(records) != FROZEN_CACHE["frames"]:
        raise RuntimeError("Summary cache frame count differs")
    suite = load_suite(PROTOCOL["suite"])
    official_tasks = task_languages(suite)
    cache_tasks, dataset_metadata = load_dataset_task_languages(
        PROTOCOL["suite"], official_tasks
    )
    validate_dataset_metadata(dataset_metadata)
    _, encode = build_vocab(cache_tasks)

    by_seed = {}
    for path in args.result:
        row = validate_result(
            path,
            load_json(path),
            checkpoint_paths,
            records,
            official_tasks,
            encode,
            provenance_sha256,
            provenance_canonical_sha256,
            expected_source_hashes,
        )
        if row["seed"] in by_seed:
            raise RuntimeError(f"Duplicate result for seed {row['seed']}")
        by_seed[row["seed"]] = row
    assert_equal("full result seed coverage", set(by_seed), set(FROZEN_CHECKPOINTS))
    checkpoints = [by_seed[seed] for seed in sorted(by_seed)]
    contribution_records = [
        row
        for checkpoint in checkpoints
        for row in checkpoint.pop("contribution_records")
    ]
    module_records = [
        row for checkpoint in checkpoints for row in checkpoint.pop("module_records")
    ]
    prompt_change_catalog = [
        checkpoint.pop("prompt_relative_changes") for checkpoint in checkpoints
    ]
    prompt_changes = [
        value
        for catalog in prompt_change_catalog
        for value in catalog["including_matched"]
    ]
    mismatched_prompt_changes = [
        value
        for catalog in prompt_change_catalog
        for value in catalog["mismatched_only"]
    ]

    summary = {
        "schema": SUMMARY_SCHEMA,
        "protocol": PROTOCOL,
        "identity": {
            "cache_basename": FROZEN_CACHE["basename"],
            "cache_sha256": FROZEN_CACHE["sha256"],
            "cache_frames": FROZEN_CACHE["frames"],
            "provenance_result": str(args.provenance_result),
            "provenance_result_sha256": provenance_sha256,
            "provenance_job_id": FROZEN_PROVENANCE_JOB_ID,
            "provenance_canonical_content_sha256": provenance_canonical_sha256,
            "capability_manifest": str(args.capability_manifest),
            "capability_manifest_sha256": FROZEN_CAPABILITY_MANIFEST["sha256"],
            "dataset_metadata": FROZEN_DATASET_METADATA,
            "official_task_languages": official_tasks,
            "source_sha256": expected_source_hashes,
        },
        "checkpoints": checkpoints,
        "aggregate": {
            "checkpoints_audited": 3,
            "inputs_audited": 384,
            "module_evaluations": 3072,
            "head_evaluations": 36864,
            "source_group_evaluations": 147456,
            "prompt_permutation_evaluations": 300,
            "max_relative_l2_error": max(
                row["max_relative_l2_error"] for row in checkpoints
            ),
            "relative_l2_gate": PROTOCOL["relative_l2_gate"],
            "all_three_checkpoints_pass": all(row["passed"] for row in checkpoints),
        },
        "head_contribution_distributions": {
            "overall": grouped_distributions(
                contribution_records, HEAD_METRICS, lambda _row: "all"
            )["all"],
            "by_checkpoint": grouped_distributions(
                contribution_records, HEAD_METRICS, lambda row: f"seed_{row['seed']}"
            ),
            "by_layer": grouped_distributions(
                contribution_records, HEAD_METRICS, lambda row: f"block_{row['block']}"
            ),
            "by_head": grouped_distributions(
                contribution_records, HEAD_METRICS, lambda row: f"head_{row['head']}"
            ),
            "by_layer_head": grouped_distributions(
                contribution_records,
                HEAD_METRICS,
                lambda row: f"block_{row['block']}_head_{row['head']}",
            ),
        },
        "module_contribution_distributions": {
            "overall": grouped_distributions(
                module_records, MODULE_METRICS, lambda _row: "all"
            )["all"],
            "by_checkpoint": grouped_distributions(
                module_records, MODULE_METRICS, lambda row: f"seed_{row['seed']}"
            ),
            "by_layer": grouped_distributions(
                module_records, MODULE_METRICS, lambda row: f"block_{row['block']}"
            ),
        },
        "prompt_permutation_characterization": {
            "relative_l2_change_distribution_including_matched": distribution(
                prompt_changes
            ),
            "relative_l2_change_distribution_mismatched_only": distribution(
                mismatched_prompt_changes
            ),
            "interpretation": (
                "Same-observation action-prediction changes under familiar official prompt "
                "permutations are descriptive. They are not a grounding or counterfactual "
                "task-success test."
            ),
        },
        "claim_if_passed": (
            "On 128 deterministic, official-task-balanced, SHA-256-ranked Object-cache "
            "inputs for each of the three verified ViT checkpoint artifacts, the four frozen "
            "source-token groups reconstruct every action-query head update in all eight joint "
            "blocks, the corresponding projected module output, and its gain-scaled residual "
            "update with maximum relative L2 error at most 1e-6."
        ),
        "boundaries": [
            "BOS is assigned to instruction and embodiment to robot state so the four source groups exhaust the deployed 107-token sequence.",
            "The learned output-projection bias is source-independent and is audited separately rather than assigned arbitrarily to a modality.",
            "Energy fractions, per-visible-source-token energies, signed projections, and cosines are descriptive. No favorable magnitude threshold was frozen or tested.",
            "The official-prompt permutation uses fixed training-domain observations and familiar prompts. It is not evidence of language grounding, causal necessity, or closed-loop task success.",
            "This is an input-conditioned, layerwise attention decomposition. It does not decompose the ViT vision blocks, FFNs, residual composition, output normalization, action head, or the complete policy into one compact tensor.",
        ],
    }
    if not summary["aggregate"]["all_three_checkpoints_pass"]:
        raise RuntimeError("At least one checkpoint fails the frozen fidelity gate")
    write_json(args.output, summary)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
