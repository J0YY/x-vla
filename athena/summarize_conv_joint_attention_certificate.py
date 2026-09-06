#!/usr/bin/env python3
"""Strictly validate and aggregate the frozen convolutional attention certificate."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

from athena.run_conv_joint_attention_certificate import (
    FROZEN_CACHE,
    FROZEN_CHECKPOINTS,
    FROZEN_DATASET_METADATA,
    FROZEN_PROVENANCE_JOB_ID,
    PROTOCOL,
    SCHEMA,
    file_sha256,
    rank_cache_inputs,
    source_hashes,
    validate_dataset_metadata,
    validate_provenance,
    write_json,
)
from athena.run_xvla_experiment import (
    build_vocab,
    load_suite,
    task_languages,
)
from athena.libero_dataset_metadata import load_dataset_task_languages


SUMMARY_SCHEMA = "xvla-conv-joint-attention-certificate-summary-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        action="append",
        required=True,
        help="One full checkpoint certificate. Exactly three are required.",
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--provenance-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def assert_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} differs from the frozen value")


def assert_float_equal(label: str, actual: Any, expected: Any) -> None:
    actual_value = float(actual)
    expected_value = float(expected)
    if not math.isfinite(actual_value) or actual_value != expected_value:
        raise RuntimeError(f"{label} does not reproduce exactly")


def validate_error_row(label: str, row: dict[str, Any]) -> tuple[bool, float, float]:
    finite = row.get("finite") is True
    max_abs = float(row.get("max_abs_error", float("nan")))
    relative = float(row.get("relative_l2_error", float("nan")))
    numbers_finite = math.isfinite(max_abs) and math.isfinite(relative)
    if finite != numbers_finite:
        raise RuntimeError(f"{label} finite flag does not match its errors")
    if numbers_finite and (max_abs < 0.0 or relative < 0.0):
        raise RuntimeError(f"{label} contains a negative error")
    return finite, max_abs, relative


def expected_input_rows(
    records: list[Any], checkpoint_sha256: str, cache_tasks: dict[int, str], encode: Any
) -> list[dict[str, Any]]:
    identities = rank_cache_inputs(
        records, checkpoint_sha256, PROTOCOL["full_inputs_per_checkpoint"]
    )
    rows = []
    for identity in identities:
        task_index = int(identity["task_index"])
        rows.append(
            {
                **identity,
                "instruction": cache_tasks[task_index],
                "instruction_ids": encode(cache_tasks[task_index]),
            }
        )
    return rows


def validate_result(
    path: Path,
    result: dict[str, Any],
    records: list[Any],
    cache_tasks: dict[int, str],
    encode: Any,
    provenance_sha256: str,
    provenance_canonical_sha256: str,
    expected_source_hashes: dict[str, str],
) -> dict[str, Any]:
    assert_equal(f"{path} schema", result.get("schema"), SCHEMA)
    assert_equal(f"{path} protocol", result.get("protocol"), PROTOCOL)
    assert_equal(f"{path} mode", result.get("mode"), "full")
    identity = result.get("identity", {})
    seed = int(identity.get("checkpoint_seed", -1))
    checkpoint = FROZEN_CHECKPOINTS.get(seed)
    if checkpoint is None:
        raise RuntimeError(f"{path} has an invalid checkpoint seed")
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
    assert_equal(f"{path} cache basename", identity.get("cache_basename"), FROZEN_CACHE["basename"])
    assert_equal(
        f"{path} cache path basename",
        Path(str(identity.get("cache_path", ""))).name,
        FROZEN_CACHE["basename"],
    )
    assert_equal(f"{path} cache SHA", identity.get("cache_sha256"), FROZEN_CACHE["sha256"])
    assert_equal(f"{path} cache frames", int(identity.get("cache_frames", -1)), FROZEN_CACHE["frames"])
    assert_equal(
        f"{path} provenance path basename",
        Path(str(identity.get("provenance_result", ""))).name,
        "cache_provenance_libero_object.json",
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
        f"{path} dataset metadata",
        identity.get("dataset_metadata"),
        FROZEN_DATASET_METADATA,
    )
    assert_equal(
        f"{path} source hashes",
        identity.get("source_sha256"),
        expected_source_hashes,
    )
    runtime = result.get("runtime", {})
    assert_equal(
        f"{path} matmul precision",
        runtime.get("matmul_precision"),
        PROTOCOL["matmul_precision"],
    )
    if not runtime.get("torch_version") or not runtime.get("gpu"):
        raise RuntimeError(f"{path} lacks its numerical runtime identity")

    expected_inputs = expected_input_rows(records, checkpoint["sha256"], cache_tasks, encode)
    assert_equal(f"{path} selected inputs", result.get("inputs"), expected_inputs)
    module_rows = result.get("module_rows")
    if not isinstance(module_rows, list):
        raise RuntimeError(f"{path} module rows are absent")
    expected_module_evaluations = (
        PROTOCOL["full_inputs_per_checkpoint"] * PROTOCOL["joint_modules_per_input"]
    )
    if len(module_rows) != expected_module_evaluations:
        raise RuntimeError(f"{path} has the wrong number of module evaluations")

    module_relatives = []
    head_relatives = []
    finite_flags = []
    row_index = 0
    for input_row in expected_inputs:
        for block_index in range(PROTOCOL["joint_modules_per_input"]):
            row = module_rows[row_index]
            row_index += 1
            assert_equal(
                f"{path} module selection rank",
                int(row.get("selection_rank", -1)),
                input_row["selection_rank"],
            )
            assert_equal(
                f"{path} module selection SHA",
                row.get("selection_sha256"),
                input_row["selection_sha256"],
            )
            assert_equal(f"{path} stack", row.get("stack"), "joint")
            assert_equal(f"{path} block index", int(row.get("block_index", -1)), block_index)
            assert_equal(f"{path} sequence length", int(row.get("sequence_length", -1)), 107)
            assert_equal(f"{path} model width", int(row.get("dim", -1)), 384)
            assert_equal(f"{path} head width", int(row.get("head_dim", -1)), 32)
            assert_equal(
                f"{path} head count",
                int(row.get("head_count", -1)),
                PROTOCOL["heads_per_module"],
            )
            module_finite, _, module_relative = validate_error_row(
                f"{path} input {input_row['selection_rank']} block {block_index}", row
            )
            finite_flags.append(module_finite)
            module_relatives.append(module_relative)
            heads = row.get("heads")
            if not isinstance(heads, list) or len(heads) != PROTOCOL["heads_per_module"]:
                raise RuntimeError(f"{path} has an invalid head catalog")
            for head_index, head in enumerate(heads):
                assert_equal(
                    f"{path} head index", int(head.get("head_index", -1)), head_index
                )
                head_finite, _, head_relative = validate_error_row(
                    f"{path} input {input_row['selection_rank']} block {block_index} "
                    f"head {head_index}",
                    head,
                )
                finite_flags.append(head_finite)
                head_relatives.append(head_relative)

    aggregate = result.get("aggregate", {})
    assert_equal(
        f"{path} aggregate input count",
        int(aggregate.get("inputs_audited", -1)),
        PROTOCOL["full_inputs_per_checkpoint"],
    )
    assert_equal(
        f"{path} aggregate module count",
        int(aggregate.get("module_evaluations", -1)),
        expected_module_evaluations,
    )
    assert_equal(
        f"{path} aggregate head count",
        int(aggregate.get("head_evaluations", -1)),
        expected_module_evaluations * PROTOCOL["heads_per_module"],
    )
    all_finite = all(finite_flags)
    max_module = max(module_relatives)
    max_head = max(head_relatives)
    max_relative = max(max_module, max_head)
    passed = all_finite and max_relative <= PROTOCOL["relative_l2_gate"]
    assert_equal(f"{path} aggregate finite flag", aggregate.get("all_finite"), all_finite)
    assert_float_equal(
        f"{path} aggregate module maximum",
        aggregate.get("max_module_relative_l2_error"),
        max_module,
    )
    assert_float_equal(
        f"{path} aggregate head maximum",
        aggregate.get("max_head_relative_l2_error"),
        max_head,
    )
    assert_float_equal(
        f"{path} aggregate maximum", aggregate.get("max_relative_l2_error"), max_relative
    )
    assert_float_equal(
        f"{path} numerical gate",
        aggregate.get("relative_l2_gate"),
        PROTOCOL["relative_l2_gate"],
    )
    assert_equal(f"{path} aggregate pass flag", aggregate.get("passed"), passed)
    if not isinstance(result.get("scope"), str) or not result["scope"]:
        raise RuntimeError(f"{path} lacks a scope boundary")
    return {
        "seed": seed,
        "source_file": str(path),
        "source_file_sha256": file_sha256(path),
        "checkpoint_basename": checkpoint["basename"],
        "checkpoint_sha256": checkpoint["sha256"],
        "inputs_audited": PROTOCOL["full_inputs_per_checkpoint"],
        "module_evaluations": expected_module_evaluations,
        "head_evaluations": expected_module_evaluations * PROTOCOL["heads_per_module"],
        "all_finite": all_finite,
        "max_module_relative_l2_error": max_module,
        "max_head_relative_l2_error": max_head,
        "max_relative_l2_error": max_relative,
        "passed": passed,
    }


def main() -> None:
    args = parse_args()
    if len(args.result) != 3:
        raise RuntimeError("Exactly three full certificate results are required")
    if len({path.resolve() for path in args.result}) != 3:
        raise RuntimeError("Certificate result paths must be distinct")
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite existing summary: {args.output}")
    if args.cache.name != FROZEN_CACHE["basename"]:
        raise RuntimeError("Summary cache basename differs from the frozen cache")
    if file_sha256(args.cache) != FROZEN_CACHE["sha256"]:
        raise RuntimeError("Summary cache SHA differs from the frozen cache")
    provenance = validate_provenance(args.provenance_result, args.cache, FROZEN_CACHE["sha256"])
    provenance_sha256 = file_sha256(args.provenance_result)
    provenance_canonical_sha256 = provenance["cache"]["canonical_content_sha256"]
    expected_source_hashes = source_hashes()

    with args.cache.open("rb") as handle:
        records = pickle.load(handle)
    if len(records) != FROZEN_CACHE["frames"]:
        raise RuntimeError("Summary cache frame count differs from the frozen count")
    suite = load_suite(PROTOCOL["suite"])
    official_tasks = task_languages(suite)
    cache_tasks, dataset_metadata = load_dataset_task_languages(
        PROTOCOL["suite"], official_tasks
    )
    validate_dataset_metadata(dataset_metadata)
    _, encode = build_vocab(cache_tasks)

    by_seed = {}
    for path in args.result:
        checkpoint_summary = validate_result(
            path,
            load_json(path),
            records,
            cache_tasks,
            encode,
            provenance_sha256,
            provenance_canonical_sha256,
            expected_source_hashes,
        )
        seed = checkpoint_summary["seed"]
        if seed in by_seed:
            raise RuntimeError(f"Duplicate result for seed {seed}")
        by_seed[seed] = checkpoint_summary
    if set(by_seed) != set(FROZEN_CHECKPOINTS):
        raise RuntimeError("Full results do not cover exactly checkpoint seeds 0, 1, and 2")

    checkpoints = [by_seed[seed] for seed in sorted(by_seed)]
    all_finite = all(row["all_finite"] for row in checkpoints)
    max_relative = max(row["max_relative_l2_error"] for row in checkpoints)
    passed = all(row["passed"] for row in checkpoints)
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
            "source_sha256": expected_source_hashes,
        },
        "checkpoints": checkpoints,
        "aggregate": {
            "checkpoints_audited": len(checkpoints),
            "inputs_audited": sum(row["inputs_audited"] for row in checkpoints),
            "module_evaluations": sum(row["module_evaluations"] for row in checkpoints),
            "head_evaluations": sum(row["head_evaluations"] for row in checkpoints),
            "all_finite": all_finite,
            "max_relative_l2_error": max_relative,
            "relative_l2_gate": PROTOCOL["relative_l2_gate"],
            "all_three_checkpoints_pass": passed,
        },
        "claim_if_passed": (
            "On 16 deterministic SHA-256-ranked LIBERO-Object cache inputs for each of three "
            "fixed legacy convolutional χ-VLA checkpoints, independent layerwise float64 "
            "reconstruction reproduced all eight joint attention modules and all twelve heads "
            "per module with finite outputs and maximum relative L2 error at most 1e-6."
        ),
        "boundaries": [
            "This is a same-checkpoint numerical identity, not a comparison with the ViT policy.",
            "The certificate covers joint bilinear attention only. The convolutional stem, "
            "FFNs, residual composition, output normalization, and action head are not rebuilt.",
            "The selected inputs come from the verified training-domain Object cache. They do "
            "not establish cross-suite behavior or robustness outside those fixed observations.",
            "The audit is layerwise and input-conditioned. It does not materialize one compact "
            "symbolic contraction for the complete policy.",
            "Passing establishes independent numerical reproduction from learned weights on "
            "the preregistered inputs. It is not a symbolic or exhaustive operator proof and "
            "does not establish that the recovered factors are human-nameable, causally "
            "selective, or uniquely useful for interpretation.",
        ],
    }
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
