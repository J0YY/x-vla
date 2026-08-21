#!/usr/bin/env python3
"""Strictly validate and aggregate the frozen Conv joint-FFN certificate."""

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
    hash_array,
    rank_cache_inputs,
    validate_dataset_metadata,
    validate_provenance,
    write_json,
)
from athena.run_conv_joint_ffn_certificate import (
    PROTOCOL,
    SCHEMA,
    file_sha256,
    source_hashes,
)
from athena.run_xvla_experiment import build_vocab, load_suite, task_languages


SUMMARY_SCHEMA = "xvla-conv-joint-ffn-certificate-summary-v1"
FROZEN_COMPANION_CERTIFICATES = {
    "attention": {
        "basename": "conv_joint_attention_certificate_v1_summary.json",
        "sha256": "3c380a899e5621d312cc45d62aff99ce883dd8dd029de56078e1cdc67838613f",
        "schema": "xvla-conv-joint-attention-certificate-summary-v1",
    },
    "rational_norm": {
        "basename": "rational_norm_safety_v1_summary.json",
        "sha256": "25c9dd61bfc805c321b12ba1b6575e4463fb76c1369de3b7a317dbdb524b17a8",
        "schema": "xvla-rational-norm-safety-summary-v1",
    },
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        action="append",
        required=True,
        help="One full checkpoint certificate. Exactly three are required.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="One frozen checkpoint. Exactly three are required.",
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--provenance-result", type=Path, required=True)
    parser.add_argument("--attention-summary", type=Path, required=True)
    parser.add_argument("--rational-norm-summary", type=Path, required=True)
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


def assert_sha256(label: str, value: Any) -> None:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not a lowercase SHA-256 digest")


def expected_input_rows(
    records: list[Any], checkpoint_sha256: str, cache_tasks: dict[int, str], encode: Any
) -> list[dict[str, Any]]:
    identities = rank_cache_inputs(
        records, checkpoint_sha256, PROTOCOL["full_inputs_per_checkpoint"]
    )
    return [
        {
            **identity,
            "instruction": cache_tasks[int(identity["task_index"])],
            "instruction_ids": encode(
                cache_tasks[int(identity["task_index"])]
            ),
        }
        for identity in identities
    ]


def tensor_identity(tensor: torch.Tensor) -> str:
    return hash_array(tensor.detach().cpu().numpy())


def expected_factor_identities(checkpoint: Path) -> list[dict[str, Any]]:
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    identities = []
    for block_index in range(PROTOCOL["joint_modules_per_input"]):
        prefix = f"backbone.blocks.{block_index}.ffn"
        names = {
            "left_weight": f"{prefix}.left.weight",
            "left_bias": f"{prefix}.left.bias",
            "right_weight": f"{prefix}.right.weight",
            "right_bias": f"{prefix}.right.bias",
            "down_weight": f"{prefix}.down.weight",
        }
        missing = [key for key in names.values() if key not in state_dict]
        if missing:
            raise RuntimeError(f"Checkpoint lacks FFN factors: {missing}")
        if f"{prefix}.down.bias" in state_dict:
            raise RuntimeError("Checkpoint has an unexpected FFN down bias")
        tensors = {name: state_dict[key] for name, key in names.items()}
        identity = {
            "left_weight_shape": list(tensors["left_weight"].shape),
            "left_bias_shape": list(tensors["left_bias"].shape),
            "right_weight_shape": list(tensors["right_weight"].shape),
            "right_bias_shape": list(tensors["right_bias"].shape),
            "down_weight_shape": list(tensors["down_weight"].shape),
            "down_bias_present": False,
            "parameter_count": sum(tensor.numel() for tensor in tensors.values()),
            "sha256": {
                name: tensor_identity(tensor) for name, tensor in tensors.items()
            },
        }
        assert_equal(
            f"checkpoint block {block_index} left shape",
            identity["left_weight_shape"],
            [PROTOCOL["cp_rank"], PROTOCOL["input_dim"]],
        )
        assert_equal(
            f"checkpoint block {block_index} left bias shape",
            identity["left_bias_shape"],
            [PROTOCOL["cp_rank"]],
        )
        assert_equal(
            f"checkpoint block {block_index} right shape",
            identity["right_weight_shape"],
            [PROTOCOL["cp_rank"], PROTOCOL["input_dim"]],
        )
        assert_equal(
            f"checkpoint block {block_index} right bias shape",
            identity["right_bias_shape"],
            [PROTOCOL["cp_rank"]],
        )
        assert_equal(
            f"checkpoint block {block_index} down shape",
            identity["down_weight_shape"],
            [PROTOCOL["output_dim"], PROTOCOL["cp_rank"]],
        )
        assert_equal(
            f"checkpoint block {block_index} parameter count",
            identity["parameter_count"],
            PROTOCOL["parameters_per_unique_module"],
        )
        identities.append(identity)
    return identities


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


def validate_result(
    path: Path,
    result: dict[str, Any],
    checkpoint_paths: dict[int, Path],
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
    if checkpoint is None or seed not in checkpoint_paths:
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
    assert_equal(
        f"{path} cache basename", identity.get("cache_basename"), FROZEN_CACHE["basename"]
    )
    assert_equal(
        f"{path} cache path basename",
        Path(str(identity.get("cache_path", ""))).name,
        FROZEN_CACHE["basename"],
    )
    assert_equal(f"{path} cache SHA", identity.get("cache_sha256"), FROZEN_CACHE["sha256"])
    assert_equal(
        f"{path} cache frames", int(identity.get("cache_frames", -1)), FROZEN_CACHE["frames"]
    )
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
        f"{path} dataset metadata", identity.get("dataset_metadata"), FROZEN_DATASET_METADATA
    )
    assert_equal(
        f"{path} source hashes", identity.get("source_sha256"), expected_source_hashes
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
    factors = expected_factor_identities(checkpoint_paths[seed])
    assert_equal(f"{path} unique factor catalog", result.get("unique_module_factors"), factors)
    module_rows = result.get("module_rows")
    if not isinstance(module_rows, list):
        raise RuntimeError(f"{path} module rows are absent")
    expected_module_evaluations = (
        PROTOCOL["full_inputs_per_checkpoint"] * PROTOCOL["joint_modules_per_input"]
    )
    if len(module_rows) != expected_module_evaluations:
        raise RuntimeError(f"{path} has the wrong number of module evaluations")

    finite_flags = []
    max_absolutes = []
    relatives = []
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
            assert_equal(
                f"{path} sequence length",
                int(row.get("sequence_length", -1)),
                PROTOCOL["sequence_length"],
            )
            assert_equal(
                f"{path} input width", int(row.get("input_dim", -1)), PROTOCOL["input_dim"]
            )
            assert_equal(f"{path} CP rank", int(row.get("cp_rank", -1)), PROTOCOL["cp_rank"])
            assert_equal(
                f"{path} output width",
                int(row.get("output_dim", -1)),
                PROTOCOL["output_dim"],
            )
            assert_equal(
                f"{path} factor identity", row.get("factor_identity"), factors[block_index]
            )
            assert_sha256(f"{path} reference output SHA", row.get("reference_output_sha256"))
            assert_sha256(
                f"{path} reconstruction output SHA",
                row.get("reconstruction_output_sha256"),
            )
            finite, max_abs, relative = validate_error_row(
                f"{path} input {input_row['selection_rank']} block {block_index}", row
            )
            finite_flags.append(finite)
            max_absolutes.append(max_abs)
            relatives.append(relative)

    expected_token_rows = expected_module_evaluations * PROTOCOL["sequence_length"]
    expected_scalar_outputs = expected_token_rows * PROTOCOL["output_dim"]
    expected_parameters = (
        PROTOCOL["joint_modules_per_input"] * PROTOCOL["parameters_per_unique_module"]
    )
    all_finite = all(finite_flags)
    max_absolute = max(max_absolutes)
    max_relative = max(relatives)
    passed = all_finite and max_relative <= PROTOCOL["relative_l2_gate"]
    aggregate = result.get("aggregate", {})
    assert_equal(
        f"{path} aggregate input count",
        int(aggregate.get("inputs_audited", -1)),
        PROTOCOL["full_inputs_per_checkpoint"],
    )
    assert_equal(
        f"{path} aggregate unique module count",
        int(aggregate.get("unique_modules", -1)),
        PROTOCOL["joint_modules_per_input"],
    )
    assert_equal(
        f"{path} aggregate module count",
        int(aggregate.get("module_evaluations", -1)),
        expected_module_evaluations,
    )
    assert_equal(
        f"{path} aggregate token rows",
        int(aggregate.get("token_rows", -1)),
        expected_token_rows,
    )
    assert_equal(
        f"{path} aggregate scalar outputs",
        int(aggregate.get("scalar_outputs", -1)),
        expected_scalar_outputs,
    )
    assert_equal(
        f"{path} aggregate unique parameters",
        int(aggregate.get("unique_module_parameters", -1)),
        expected_parameters,
    )
    assert_equal(f"{path} aggregate finite flag", aggregate.get("all_finite"), all_finite)
    assert_float_equal(
        f"{path} aggregate absolute maximum",
        aggregate.get("max_absolute_error"),
        max_absolute,
    )
    assert_float_equal(
        f"{path} aggregate relative maximum",
        aggregate.get("max_relative_l2_error"),
        max_relative,
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
        "unique_modules": PROTOCOL["joint_modules_per_input"],
        "module_evaluations": expected_module_evaluations,
        "token_rows": expected_token_rows,
        "scalar_outputs": expected_scalar_outputs,
        "unique_module_parameters": expected_parameters,
        "all_finite": all_finite,
        "max_absolute_error": max_absolute,
        "max_relative_l2_error": max_relative,
        "passed": passed,
    }


def validate_companion_certificates(
    attention_path: Path, rational_norm_path: Path
) -> dict[str, Any]:
    paths = {"attention": attention_path, "rational_norm": rational_norm_path}
    documents = {}
    for label, path in paths.items():
        frozen = FROZEN_COMPANION_CERTIFICATES[label]
        assert_equal(f"{label} summary basename", path.name, frozen["basename"])
        assert_equal(f"{label} summary SHA", file_sha256(path), frozen["sha256"])
        document = load_json(path)
        assert_equal(f"{label} summary schema", document.get("schema"), frozen["schema"])
        documents[label] = document

    attention = documents["attention"]
    assert_equal(
        "attention three-checkpoint pass",
        attention.get("aggregate", {}).get("all_three_checkpoints_pass"),
        True,
    )
    assert_equal(
        "attention modules per checkpoint",
        int(attention.get("aggregate", {}).get("module_evaluations", -1)),
        3 * PROTOCOL["full_inputs_per_checkpoint"] * PROTOCOL["joint_modules_per_input"],
    )
    attention_checkpoints = {
        int(row["seed"]): (row["checkpoint_basename"], row["checkpoint_sha256"])
        for row in attention.get("checkpoints", [])
    }
    expected_checkpoints = {
        seed: (value["basename"], value["sha256"])
        for seed, value in FROZEN_CHECKPOINTS.items()
    }
    assert_equal("attention checkpoint identities", attention_checkpoints, expected_checkpoints)

    rational = documents["rational_norm"]
    assert_equal(
        "rational primary three-checkpoint pass",
        rational.get("gates", {}).get("all_three_primary_pass"),
        True,
    )
    assert_equal(
        "rational sites per checkpoint",
        int(rational.get("aggregate", {}).get("rational_sites_per_checkpoint", -1)),
        53,
    )
    rational_checkpoints = {
        int(row["seed"]): (
            row["checkpoint"]["basename"],
            row["checkpoint"]["sha256"],
        )
        for row in rational.get("checkpoints", [])
        if row.get("gates", {}).get("primary_pass") is True
    }
    assert_equal("rational checkpoint identities", rational_checkpoints, expected_checkpoints)
    return {
        label: {
            "path": str(paths[label]),
            "sha256": FROZEN_COMPANION_CERTIFICATES[label]["sha256"],
            "schema": FROZEN_COMPANION_CERTIFICATES[label]["schema"],
        }
        for label in ("attention", "rational_norm")
    }


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


def main() -> None:
    args = parse_args()
    if len(args.result) != 3 or len({path.resolve() for path in args.result}) != 3:
        raise RuntimeError("Exactly three distinct full certificate results are required")
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite existing summary: {args.output}")
    checkpoints_by_seed = checkpoint_path_map(args.checkpoint)
    if args.cache.name != FROZEN_CACHE["basename"]:
        raise RuntimeError("Summary cache basename differs from the frozen cache")
    if file_sha256(args.cache) != FROZEN_CACHE["sha256"]:
        raise RuntimeError("Summary cache SHA differs from the frozen cache")
    provenance = validate_provenance(args.provenance_result, args.cache, FROZEN_CACHE["sha256"])
    provenance_sha256 = file_sha256(args.provenance_result)
    provenance_canonical_sha256 = provenance["cache"]["canonical_content_sha256"]
    companion_certificates = validate_companion_certificates(
        args.attention_summary, args.rational_norm_summary
    )
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
            checkpoints_by_seed,
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
    max_absolute = max(row["max_absolute_error"] for row in checkpoints)
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
            "companion_certificates": companion_certificates,
        },
        "checkpoints": checkpoints,
        "aggregate": {
            "checkpoints_audited": len(checkpoints),
            "inputs_audited": sum(row["inputs_audited"] for row in checkpoints),
            "unique_checkpoint_modules": sum(row["unique_modules"] for row in checkpoints),
            "module_evaluations": sum(row["module_evaluations"] for row in checkpoints),
            "token_rows": sum(row["token_rows"] for row in checkpoints),
            "scalar_outputs": sum(row["scalar_outputs"] for row in checkpoints),
            "unique_checkpoint_module_parameters": sum(
                row["unique_module_parameters"] for row in checkpoints
            ),
            "all_finite": all_finite,
            "max_absolute_error": max_absolute,
            "max_relative_l2_error": max_relative,
            "relative_l2_gate": PROTOCOL["relative_l2_gate"],
            "all_three_checkpoints_pass": passed,
        },
        "claim_if_passed": (
            "On the same 16 deterministic SHA-256-ranked LIBERO-Object cache inputs per "
            "checkpoint used by the attention certificate, independent NumPy float64 CP-factor "
            "contractions reconstruct every bilinear FFN in the eight-block joint transformer "
            "for three fixed convolutional χ-VLA checkpoints. This covers 384 module-input "
            "cases, 41,088 token rows, and 15,777,792 scalar outputs with finite values and "
            "maximum relative L2 error at most 1e-10."
        ),
        "combined_scope_if_passed": (
            "Together with the frozen passing attention and primary RationalNorm certificates "
            "on the same three checkpoint files, the certificate package covers all eight "
            "attention modules, all eight bilinear FFNs, and all 53 deployed RationalNorm sites "
            "per checkpoint. This is complete primitive-level coverage of the eight-block joint "
            "transformer plus every deployed RationalNorm site, not complete end-to-end policy "
            "reconstruction."
        ),
        "boundaries": [
            "This is a same-checkpoint numerical identity, not a comparison with the ViT policy.",
            "The FFN audit is layerwise and input-conditioned. It does not materialize a dense "
            "third-order core or one compact symbolic contraction for the complete policy.",
            "The combined scope does not reconstruct the three bilinear-convolution vision "
            "stages, residual composition, linear embeddings and projections, or action head.",
            "The selected FFN inputs come from the verified training-domain Object cache. They "
            "do not establish cross-suite behavior or robustness outside those observations.",
            "Passing establishes independent numerical reproduction from learned factors on "
            "preregistered inputs. It does not establish uniqueness, human-nameable factors, "
            "causal selectivity, or a complete symbolic proof of the end-to-end policy.",
        ],
    }
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("At least one frozen checkpoint failed the FFN certificate")


if __name__ == "__main__":
    main()
