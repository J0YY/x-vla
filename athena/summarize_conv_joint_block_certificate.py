#!/usr/bin/env python3
"""Strictly validate and aggregate the frozen complete joint-block certificate."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any

import torch

from athena.libero_dataset_metadata import load_dataset_task_languages
from athena.run_conv_joint_attention_certificate import (
    FROZEN_CACHE,
    FROZEN_CHECKPOINTS,
    FROZEN_DATASET_METADATA,
    FROZEN_PROVENANCE_JOB_ID,
    file_sha256,
    rank_cache_inputs,
    validate_dataset_metadata,
    validate_provenance,
    write_json,
)
from athena.run_conv_joint_block_certificate import (
    PROTOCOL,
    SCHEMA,
    block_identity,
    source_hashes,
    validate_companion,
)
from athena.run_conv_joint_ffn_certificate import PROTOCOL as FFN_PROTOCOL
from athena.run_xvla_experiment import (
    build_vocab,
    load_cache_statistics,
    load_suite,
    make_config,
    task_languages,
)
from xvla.models.vla import ChiVLA


SUMMARY_SCHEMA = "xvla-conv-joint-block-certificate-summary-v1"
FFN_SUMMARY_SCHEMA = "xvla-conv-joint-ffn-certificate-summary-v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result", type=Path, action="append", required=True,
        help="One full checkpoint certificate. Exactly three are required."
    )
    parser.add_argument(
        "--checkpoint", type=Path, action="append", required=True,
        help="One frozen checkpoint. Exactly three are required."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--provenance-result", type=Path, required=True)
    parser.add_argument("--attention-summary", type=Path, required=True)
    parser.add_argument("--rational-norm-summary", type=Path, required=True)
    parser.add_argument("--ffn-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise RuntimeError(f"JSON result is not an object: {path}")
    return result


def assert_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} differs from the frozen value")


def assert_float_equal(label: str, actual: Any, expected: Any) -> None:
    actual_value = float(actual)
    expected_value = float(expected)
    if not math.isfinite(actual_value) or actual_value != expected_value:
        raise RuntimeError(f"{label} does not reproduce exactly")


def assert_sha256(label: str, value: Any) -> None:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not a lowercase SHA-256 digest")


def checkpoint_path_map(paths: list[Path]) -> dict[int, Path]:
    if len(paths) != 3 or len({path.resolve() for path in paths}) != 3:
        raise RuntimeError("Exactly three distinct frozen checkpoint paths are required")
    by_seed = {}
    for path in paths:
        matched = [
            seed
            for seed, frozen in FROZEN_CHECKPOINTS.items()
            if path.name == frozen["basename"]
        ]
        if len(matched) != 1:
            raise RuntimeError(f"Unexpected checkpoint basename: {path}")
        seed = matched[0]
        if file_sha256(path) != FROZEN_CHECKPOINTS[seed]["sha256"]:
            raise RuntimeError(f"Frozen checkpoint SHA differs: {path}")
        by_seed[seed] = path
    if set(by_seed) != set(FROZEN_CHECKPOINTS):
        raise RuntimeError("Checkpoint paths do not cover seeds 0, 1, and 2")
    return by_seed


def expected_inputs(
    records: list[Any], checkpoint_sha256: str, cache_tasks: dict[int, str], encode: Any
) -> list[dict[str, Any]]:
    selected = rank_cache_inputs(
        records, checkpoint_sha256, PROTOCOL["full_inputs_per_checkpoint"]
    )
    return [
        {
            **identity,
            "instruction": cache_tasks[int(identity["task_index"])],
            "instruction_ids": encode(cache_tasks[int(identity["task_index"])]),
        }
        for identity in selected
    ]


def expected_block_identities(
    checkpoint: Path, vocab_size: int, stats: dict[str, Any]
) -> list[dict[str, Any]]:
    config = make_config(
        "chi",
        vocab_size,
        stats["state_dim"],
        stats["action_dim"],
        PROTOCOL["resolution"],
        PROTOCOL["action_horizon"],
        vision_encoder="conv",
    )
    model = ChiVLA(config)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if len(model.backbone.blocks) != PROTOCOL["joint_blocks_per_input"]:
        raise RuntimeError("Summary model joint block count differs")
    return [
        block_identity(block, block_index)
        for block_index, block in enumerate(model.backbone.blocks)
    ]


def validate_ffn_companion(path: Path) -> dict[str, Any]:
    if path.name != "conv_joint_ffn_certificate_v1_summary.json":
        raise RuntimeError("FFN companion basename differs")
    result = load_json(path)
    assert_equal("FFN companion schema", result.get("schema"), FFN_SUMMARY_SCHEMA)
    assert_equal("FFN companion protocol", result.get("protocol"), FFN_PROTOCOL)
    assert_equal(
        "FFN companion pass",
        result.get("aggregate", {}).get("all_three_checkpoints_pass"),
        True,
    )
    assert_equal(
        "FFN companion evaluation count",
        int(result.get("aggregate", {}).get("module_evaluations", -1)),
        3 * PROTOCOL["full_inputs_per_checkpoint"] * PROTOCOL["joint_blocks_per_input"],
    )
    actual_checkpoints = {
        int(row["seed"]): (row["checkpoint_basename"], row["checkpoint_sha256"])
        for row in result.get("checkpoints", [])
        if row.get("passed") is True
    }
    expected_checkpoints = {
        seed: (value["basename"], value["sha256"])
        for seed, value in FROZEN_CHECKPOINTS.items()
    }
    assert_equal("FFN companion checkpoints", actual_checkpoints, expected_checkpoints)
    return {
        "path": str(path),
        "basename": path.name,
        "sha256": file_sha256(path),
        "schema": FFN_SUMMARY_SCHEMA,
    }


def validate_error_row(label: str, row: dict[str, Any]) -> tuple[bool, float, float]:
    finite = row.get("finite") is True
    max_absolute = float(row.get("max_abs_error", float("nan")))
    relative = float(row.get("relative_l2_error", float("nan")))
    numbers_finite = math.isfinite(max_absolute) and math.isfinite(relative)
    if finite != numbers_finite:
        raise RuntimeError(f"{label} finite flag does not match its errors")
    if numbers_finite and (max_absolute < 0.0 or relative < 0.0):
        raise RuntimeError(f"{label} contains a negative error")
    return finite, max_absolute, relative


def validate_result(
    path: Path,
    result: dict[str, Any],
    checkpoint_paths: dict[int, Path],
    records: list[Any],
    cache_tasks: dict[int, str],
    encode: Any,
    vocab_size: int,
    stats: dict[str, Any],
    provenance_sha256: str,
    provenance_canonical_sha256: str,
    companion_certificates: dict[str, Any],
    expected_source_hashes: dict[str, str],
) -> dict[str, Any]:
    assert_equal(f"{path} schema", result.get("schema"), SCHEMA)
    assert_equal(f"{path} protocol", result.get("protocol"), PROTOCOL)
    assert_equal(f"{path} mode", result.get("mode"), "full")
    identity = result.get("identity", {})
    seed = int(identity.get("checkpoint_seed", -1))
    checkpoint = FROZEN_CHECKPOINTS.get(seed)
    if checkpoint is None or seed not in checkpoint_paths:
        raise RuntimeError(f"{path} has an invalid checkpoint seed")
    assert_equal(f"{path} checkpoint basename", identity.get("checkpoint_basename"), checkpoint["basename"])
    assert_equal(f"{path} checkpoint path basename", Path(str(identity.get("checkpoint_path", ""))).name, checkpoint["basename"])
    assert_equal(f"{path} checkpoint SHA", identity.get("checkpoint_sha256"), checkpoint["sha256"])
    assert_equal(f"{path} cache basename", identity.get("cache_basename"), FROZEN_CACHE["basename"])
    assert_equal(f"{path} cache SHA", identity.get("cache_sha256"), FROZEN_CACHE["sha256"])
    assert_equal(f"{path} cache frames", int(identity.get("cache_frames", -1)), FROZEN_CACHE["frames"])
    assert_equal(f"{path} provenance SHA", identity.get("provenance_result_sha256"), provenance_sha256)
    assert_equal(f"{path} provenance job", str(identity.get("provenance_job_id", "")), FROZEN_PROVENANCE_JOB_ID)
    assert_equal(f"{path} provenance canonical SHA", identity.get("provenance_canonical_content_sha256"), provenance_canonical_sha256)
    assert_equal(f"{path} dataset metadata", identity.get("dataset_metadata"), FROZEN_DATASET_METADATA)
    runner_companions = dict(companion_certificates)
    runner_companions.pop("ffn_summary")
    assert_equal(f"{path} companion identities", identity.get("companion_certificates"), runner_companions)
    assert_equal(f"{path} source hashes", identity.get("source_sha256"), expected_source_hashes)
    runtime = result.get("runtime", {})
    assert_equal(f"{path} matmul precision", runtime.get("matmul_precision"), PROTOCOL["matmul_precision"])
    if "A6000" not in str(runtime.get("gpu", "")):
        raise RuntimeError(f"{path} was not evaluated on the frozen A6000 path")

    inputs = expected_inputs(records, checkpoint["sha256"], cache_tasks, encode)
    assert_equal(f"{path} selected inputs", result.get("inputs"), inputs)
    block_identities = expected_block_identities(
        checkpoint_paths[seed], vocab_size, stats
    )
    assert_equal(
        f"{path} unique block identities",
        result.get("unique_block_identities"),
        block_identities,
    )
    module_rows = result.get("module_rows")
    if not isinstance(module_rows, list):
        raise RuntimeError(f"{path} module rows are absent")
    expected_evaluations = (
        PROTOCOL["full_inputs_per_checkpoint"] * PROTOCOL["joint_blocks_per_input"]
    )
    assert_equal(f"{path} complete block count", len(module_rows), expected_evaluations)

    finite_flags = []
    max_absolutes = []
    relatives = []
    row_index = 0
    for input_row in inputs:
        for block_index in range(PROTOCOL["joint_blocks_per_input"]):
            row = module_rows[row_index]
            row_index += 1
            assert_equal(f"{path} selection rank", int(row.get("selection_rank", -1)), input_row["selection_rank"])
            assert_equal(f"{path} selection SHA", row.get("selection_sha256"), input_row["selection_sha256"])
            assert_equal(f"{path} stack", row.get("stack"), "joint")
            assert_equal(f"{path} block index", int(row.get("block_index", -1)), block_index)
            assert_equal(f"{path} sequence length", int(row.get("sequence_length", -1)), PROTOCOL["sequence_length"])
            assert_equal(f"{path} model width", int(row.get("model_width", -1)), PROTOCOL["model_width"])
            assert_equal(f"{path} block identity", row.get("block_identity"), block_identities[block_index])
            for field in (
                "input_sha256",
                "deployed_float32_output_sha256",
                "reconstruction_float64_output_sha256",
            ):
                assert_sha256(f"{path} {field}", row.get(field))
            finite, max_absolute, relative = validate_error_row(
                f"{path} input {input_row['selection_rank']} block {block_index}", row
            )
            finite_flags.append(finite)
            max_absolutes.append(max_absolute)
            relatives.append(relative)

    token_rows = expected_evaluations * PROTOCOL["sequence_length"]
    scalar_outputs = token_rows * PROTOCOL["model_width"]
    all_finite = all(finite_flags)
    max_absolute = max(max_absolutes)
    max_relative = max(relatives)
    passed = all_finite and max_relative <= PROTOCOL["complete_block_relative_l2_gate"]
    aggregate = result.get("aggregate", {})
    expected_aggregate = {
        "inputs_audited": PROTOCOL["full_inputs_per_checkpoint"],
        "unique_blocks": PROTOCOL["joint_blocks_per_input"],
        "complete_block_evaluations": expected_evaluations,
        "token_rows": token_rows,
        "scalar_outputs": scalar_outputs,
        "all_finite": all_finite,
        "max_absolute_error": max_absolute,
        "max_relative_l2_error": max_relative,
        "relative_l2_gate": PROTOCOL["complete_block_relative_l2_gate"],
        "passed": passed,
    }
    for field, expected in expected_aggregate.items():
        if isinstance(expected, float):
            assert_float_equal(f"{path} aggregate {field}", aggregate.get(field), expected)
        else:
            assert_equal(f"{path} aggregate {field}", aggregate.get(field), expected)
    if not isinstance(result.get("scope"), str) or not result["scope"]:
        raise RuntimeError(f"{path} lacks a scope boundary")
    return {
        "seed": seed,
        "source_file": str(path),
        "source_file_sha256": file_sha256(path),
        "checkpoint_basename": checkpoint["basename"],
        "checkpoint_sha256": checkpoint["sha256"],
        "inputs_audited": PROTOCOL["full_inputs_per_checkpoint"],
        "unique_blocks": PROTOCOL["joint_blocks_per_input"],
        "complete_block_evaluations": expected_evaluations,
        "token_rows": token_rows,
        "scalar_outputs": scalar_outputs,
        "all_finite": all_finite,
        "max_absolute_error": max_absolute,
        "max_relative_l2_error": max_relative,
        "passed": passed,
    }


def main() -> None:
    args = parse_args()
    if len(args.result) != 3 or len({path.resolve() for path in args.result}) != 3:
        raise RuntimeError("Exactly three distinct full certificate results are required")
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite existing summary: {args.output}")
    checkpoints = checkpoint_path_map(args.checkpoint)
    if args.cache.name != FROZEN_CACHE["basename"] or file_sha256(args.cache) != FROZEN_CACHE["sha256"]:
        raise RuntimeError("Summary cache identity differs from the frozen cache")
    provenance = validate_provenance(args.provenance_result, args.cache, FROZEN_CACHE["sha256"])
    provenance_sha256 = file_sha256(args.provenance_result)
    provenance_canonical_sha256 = provenance["cache"]["canonical_content_sha256"]
    companion_certificates = {
        "attention": validate_companion(args.attention_summary, "attention"),
        "rational_norm": validate_companion(args.rational_norm_summary, "rational_norm"),
        "ffn_protocol": {
            "schema": "xvla-conv-joint-ffn-certificate-v1",
            "relative_l2_gate": FFN_PROTOCOL["relative_l2_gate"],
            "input_dim": FFN_PROTOCOL["input_dim"],
            "cp_rank": FFN_PROTOCOL["cp_rank"],
            "output_dim": FFN_PROTOCOL["output_dim"],
        },
        "ffn_summary": validate_ffn_companion(args.ffn_summary),
    }
    expected_source_hashes = source_hashes()

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
    vocab, encode = build_vocab(cache_tasks)
    stats = load_cache_statistics(args.cache, PROTOCOL["action_horizon"])

    by_seed = {}
    for path in args.result:
        row = validate_result(
            path,
            load_json(path),
            checkpoints,
            records,
            cache_tasks,
            encode,
            len(vocab),
            stats,
            provenance_sha256,
            provenance_canonical_sha256,
            companion_certificates,
            expected_source_hashes,
        )
        if row["seed"] in by_seed:
            raise RuntimeError(f"Duplicate result for seed {row['seed']}")
        by_seed[row["seed"]] = row
    if set(by_seed) != set(FROZEN_CHECKPOINTS):
        raise RuntimeError("Full results do not cover seeds 0, 1, and 2")

    checkpoint_rows = [by_seed[seed] for seed in sorted(by_seed)]
    all_finite = all(row["all_finite"] for row in checkpoint_rows)
    max_absolute = max(row["max_absolute_error"] for row in checkpoint_rows)
    max_relative = max(row["max_relative_l2_error"] for row in checkpoint_rows)
    passed = all(row["passed"] for row in checkpoint_rows)
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
            "dataset_metadata": FROZEN_DATASET_METADATA,
            "companion_certificates": companion_certificates,
            "source_sha256": expected_source_hashes,
        },
        "checkpoints": checkpoint_rows,
        "aggregate": {
            "checkpoints_audited": len(checkpoint_rows),
            "inputs_audited": sum(row["inputs_audited"] for row in checkpoint_rows),
            "unique_checkpoint_blocks": sum(row["unique_blocks"] for row in checkpoint_rows),
            "complete_block_evaluations": sum(row["complete_block_evaluations"] for row in checkpoint_rows),
            "token_rows": sum(row["token_rows"] for row in checkpoint_rows),
            "scalar_outputs": sum(row["scalar_outputs"] for row in checkpoint_rows),
            "all_finite": all_finite,
            "max_absolute_error": max_absolute,
            "max_relative_l2_error": max_relative,
            "relative_l2_gate": PROTOCOL["complete_block_relative_l2_gate"],
            "all_three_checkpoints_pass": passed,
        },
        "claim_if_passed": (
            "On the same 16 deterministic checkpoint-conditioned inputs used by the frozen "
            "attention and FFN certificates, an independent NumPy float64 equation reproduces "
            "all eight complete joint blocks of each of three convolutional χ-VLA checkpoints "
            "against single-call deployed PyTorch float32 outputs with maximum relative L2 "
            "error at most 1e-4."
        ),
        "boundaries": [
            "This is an input-conditioned numerical composition certificate, not a dense "
            "symbolic contraction or proof for every possible input.",
            "The comparison intentionally crosses the deployed float32 implementation and a "
            "float64 reconstruction, so the frozen threshold allows float32 reduction error.",
            "The certificate covers the eight complete joint blocks but not the convolutional "
            "vision stem, input embeddings, final action head, or out-of-graph action decoding.",
            "The selected inputs come from the provenance-verified Object training cache and do "
            "not establish cross-suite behavior or human-nameable mechanisms.",
        ],
    }
    write_json(args.output, summary)
    print("RESULT", json.dumps(summary, indent=2), flush=True)
    if not passed:
        raise RuntimeError(
            f"Frozen complete-block summary gate failed: finite={all_finite}, max relative L2="
            f"{max_relative:.9g}"
        )


if __name__ == "__main__":
    main()
