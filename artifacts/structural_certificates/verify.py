#!/usr/bin/env python3
"""Verify immutable structural-certificate evidence with the Python standard library."""

from __future__ import annotations

import argparse
import base64
import binascii
import cmath
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any


ARTIFACT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = ARTIFACT_DIR.parents[1]


class VerificationError(RuntimeError):
    """Raised when evidence does not reproduce the immutable manifest."""


def load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise VerificationError(f"{path}: non-finite JSON constant {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read canonical JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"{path}: top-level JSON value is not an object")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise VerificationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise VerificationError(f"{label}: {actual!r} != {expected!r}")


def require_close(label: str, actual: Any, expected: Any) -> None:
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{label}: value is not numeric") from exc
    if not (
        math.isfinite(actual_value)
        and math.isfinite(expected_value)
        and math.isclose(actual_value, expected_value, rel_tol=1e-12, abs_tol=1e-15)
    ):
        raise VerificationError(f"{label}: {actual_value!r} != {expected_value!r}")


def safe_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    require(not candidate.is_absolute(), f"manifest path is absolute: {relative}")
    require(".." not in candidate.parts, f"manifest path escapes the root: {relative}")
    return root / candidate


def verify_manifest_hashes(root: Path, manifest: dict[str, Any]) -> dict[str, str]:
    require_equal(
        "manifest schema",
        manifest.get("schema"),
        "anonymous-structural-certificate-artifact-v2",
    )
    entries = manifest.get("files")
    require(
        isinstance(entries, list) and len(entries) == 22,
        "manifest must identify twenty-two files",
    )
    hashes: dict[str, str] = {}
    for entry in entries:
        require(isinstance(entry, dict), "manifest file entry is not an object")
        relative = entry.get("path")
        expected = entry.get("sha256")
        require(isinstance(relative, str), "manifest file path is absent")
        require(isinstance(expected, str) and len(expected) == 64, f"bad SHA-256 for {relative}")
        require(relative not in hashes, f"duplicate manifest path: {relative}")
        actual = file_sha256(safe_path(root, relative))
        require_equal(f"SHA-256 {relative}", actual, expected)
        hashes[relative] = actual
    return hashes


def error_value(label: str, row: dict[str, Any]) -> float:
    try:
        maximum = float(row["max_abs_error"])
        relative = float(row["relative_l2_error"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError(f"{label}: malformed error row") from exc
    numbers_finite = math.isfinite(maximum) and math.isfinite(relative)
    require_equal(f"{label} finite flag", row.get("finite"), numbers_finite)
    require(numbers_finite and maximum >= 0.0 and relative >= 0.0, f"{label}: invalid error")
    return relative


SOURCE_SNAPSHOTS = {
    "xvla/models/vla.py": "athena/results/exact_attention_frozen_sources/xvla_models_vla.py.b64",
    "xvla/models/vit.py": "athena/results/exact_attention_frozen_sources/xvla_models_vit.py.b64",
}

CONV_CACHE_SHA256 = "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662"
CONV_PROVENANCE_JOB_ID = "830988"
CONV_PROVENANCE_CONTENT_SHA256 = "01bc724b9bf8c158b983e34b82bbb40dce9dba2cb86dd9be2a5bf8910be341c7"
CONV_METADATA_SHA256 = "34caee9641ae50bb4e077de306a7d0031753757882da8b1f117e7ea36a486b42"
CONV_TASK_PERMUTATION = {
    "0": 9,
    "1": 4,
    "2": 1,
    "3": 3,
    "4": 0,
    "5": 7,
    "6": 2,
    "7": 6,
    "8": 5,
    "9": 8,
}
CONV_CHECKPOINT_SHA256 = {
    0: "4f9f3eef4bd661934b7c66af117995f02bdfc777368f52464f03d229ccb3e72d",
    1: "9d8df0c30583222490536b21aac040b47c5f89556aa23c08265605f7c2bc8cb6",
    2: "fc2e0bfa1a2ea2737ed0af13b168cd2562e4b0e36b159d0f98f3c5038131ace5",
}
CONV_SELECTION_SEQUENCE_SHA256 = {
    0: "d167baee5b55cb974f82f28045fc448e0478c456a9a5fe6dc9acc75369e0acfe",
    1: "38c763da8371744f8634bf816b4431818ce4c1e3d16342376df0210c7a78b33b",
    2: "0128f5e8940ad7e51a4ff05faafa8c3aa01c0eaf75c159a3de46584ac156377c",
}


def verify_conv_input_identity(
    label: str,
    identity: dict[str, Any],
    inputs: Any,
    seed: int,
) -> list[dict[str, Any]]:
    """Bind the three Conv certificates to one frozen checkpoint/cache input set."""
    require_equal(f"{label} checkpoint SHA", identity.get("checkpoint_sha256"), CONV_CHECKPOINT_SHA256[seed])
    require_equal(f"{label} cache SHA", identity.get("cache_sha256"), CONV_CACHE_SHA256)
    require_equal(f"{label} provenance job", str(identity.get("provenance_job_id")), CONV_PROVENANCE_JOB_ID)
    require_equal(
        f"{label} provenance content SHA",
        identity.get("provenance_canonical_content_sha256"),
        CONV_PROVENANCE_CONTENT_SHA256,
    )
    metadata = identity.get("dataset_metadata")
    require(isinstance(metadata, dict), f"{label}: dataset metadata is absent")
    require_equal(f"{label} metadata SHA", metadata.get("metadata_sha256"), CONV_METADATA_SHA256)
    require_equal(
        f"{label} task permutation",
        metadata.get("dataset_to_official_task"),
        CONV_TASK_PERMUTATION,
    )
    require(isinstance(inputs, list) and len(inputs) == 16, f"{label}: expected 16 inputs")
    require_equal(
        f"{label} input ranks",
        [int(row.get("selection_rank", -1)) for row in inputs],
        list(range(16)),
    )
    selections = [row.get("selection_sha256") for row in inputs]
    require(
        all(isinstance(value, str) and len(value) == 64 for value in selections),
        f"{label}: malformed selection identities",
    )
    require_equal(f"{label} sorted selections", selections, sorted(selections))
    sequence_digest = hashlib.sha256(("\n".join(selections) + "\n").encode("ascii")).hexdigest()
    require_equal(
        f"{label} selection sequence SHA",
        sequence_digest,
        CONV_SELECTION_SEQUENCE_SHA256[seed],
    )
    for index, row in enumerate(inputs):
        for field in (
            "canonical_record_sha256",
            "image_sha256",
            "state_sha256",
            "action_sha256",
        ):
            value = row.get(field)
            require(
                isinstance(value, str) and len(value) == 64,
                f"{label} input {index}: malformed {field}",
            )
    return inputs


def verify_source_identity(
    root: Path,
    label: str,
    source_map: Any,
) -> None:
    require(isinstance(source_map, dict) and source_map, f"{label}: source map is absent")
    for relative, expected in source_map.items():
        require(
            isinstance(relative, str)
            and isinstance(expected, str)
            and len(expected) == 64,
            f"{label}: malformed source identity",
        )
        source = safe_path(root, relative)
        if source.is_file() and file_sha256(source) == expected:
            continue
        snapshot_relative = SOURCE_SNAPSHOTS.get(relative)
        require(snapshot_relative is not None, f"{label}: missing or stale source {relative}")
        snapshot = safe_path(root, snapshot_relative)
        try:
            encoded = "".join(snapshot.read_text(encoding="ascii").split())
            payload = base64.b64decode(encoded, validate=True)
        except (OSError, UnicodeError, binascii.Error) as exc:
            raise VerificationError(f"{label}: invalid source snapshot {snapshot}") from exc
        require_equal(
            f"{label} snapshot source SHA-256 {relative}",
            hashlib.sha256(payload).hexdigest(),
            expected,
        )


def verify_conv(
    root: Path,
    config: dict[str, Any],
    manifest_hashes: dict[str, str],
) -> dict[str, Any]:
    raw_paths = config["raw_results"]
    require_equal("Conv raw-result count", len(raw_paths), 3)
    summary = load_json(safe_path(root, config["summary"]))
    require_equal("Conv summary schema", summary.get("schema"), config["summary_schema"])
    protocol = summary.get("protocol")
    require(isinstance(protocol, dict), "Conv summary protocol is absent")
    gate = float(config["gate"]["relative_l2_error_at_most"])
    require_close("Conv protocol gate", protocol.get("relative_l2_gate"), gate)
    require_equal("Conv protocol full inputs", protocol.get("full_inputs_per_checkpoint"), 16)
    require_equal("Conv protocol joint modules", protocol.get("joint_modules_per_input"), 8)
    require_equal("Conv protocol heads", protocol.get("heads_per_module"), 12)
    require_equal("Conv protocol precision", protocol.get("matmul_precision"), "highest")
    require_equal("Conv protocol suite", protocol.get("suite"), "libero_object")
    require_equal("Conv protocol vision encoder", protocol.get("vision_encoder"), "conv")
    summary_source_map = summary.get("identity", {}).get("source_sha256")
    verify_source_identity(root, "Conv attention summary", summary_source_map)

    summary_rows = {int(row["seed"]): row for row in summary.get("checkpoints", [])}
    require_equal("Conv summary seeds", set(summary_rows), {0, 1, 2})
    recomputed = []
    inputs_by_seed: dict[int, list[dict[str, Any]]] = {}
    for relative in raw_paths:
        result = load_json(safe_path(root, relative))
        require_equal(f"{relative} schema", result.get("schema"), config["raw_schema"])
        require_equal(f"{relative} mode", result.get("mode"), "full")
        require_equal(f"{relative} protocol", result.get("protocol"), protocol)
        identity = result.get("identity", {})
        require_equal(
            f"{relative} source map",
            identity.get("source_sha256"),
            summary_source_map,
        )
        seed = int(identity.get("checkpoint_seed", -1))
        require(seed in {0, 1, 2}, f"{relative}: invalid seed {seed}")
        require(seed not in {row["seed"] for row in recomputed}, f"duplicate Conv seed {seed}")

        inputs = verify_conv_input_identity(relative, identity, result.get("inputs"), seed)
        inputs_by_seed[seed] = inputs
        modules = result.get("module_rows")
        require(isinstance(modules, list), f"{relative}: module rows are absent")
        expected_modules = len(inputs) * int(protocol["joint_modules_per_input"])
        require_equal(f"{relative} module count", len(modules), expected_modules)
        module_errors = []
        head_errors = []
        for module_index, module in enumerate(modules):
            require(isinstance(module, dict), f"{relative}: malformed module {module_index}")
            input_index, block_index = divmod(module_index, 8)
            require_equal(
                f"{relative} module {module_index} rank",
                int(module.get("selection_rank", -1)),
                input_index,
            )
            require_equal(
                f"{relative} module {module_index} block",
                int(module.get("block_index", -1)),
                block_index,
            )
            require_equal(
                f"{relative} module {module_index} selection",
                module.get("selection_sha256"),
                inputs[input_index]["selection_sha256"],
            )
            require_equal(f"{relative} module {module_index} stack", module.get("stack"), "joint")
            require_equal(f"{relative} module {module_index} sequence", module.get("sequence_length"), 107)
            require_equal(f"{relative} module {module_index} width", module.get("dim"), 384)
            require_equal(f"{relative} module {module_index} head dim", module.get("head_dim"), 32)
            module_errors.append(error_value(f"{relative} module {module_index}", module))
            heads = module.get("heads")
            require(isinstance(heads, list), f"{relative}: heads are absent at module {module_index}")
            require_equal(
                f"{relative} head count at module {module_index}",
                len(heads),
                int(protocol["heads_per_module"]),
            )
            for head_index, head in enumerate(heads):
                require(isinstance(head, dict), f"{relative}: malformed head row")
                require_equal(
                    f"{relative} head index at module {module_index}",
                    int(head.get("head_index", -1)),
                    head_index,
                )
                head_errors.append(
                    error_value(f"{relative} module {module_index} head {head_index}", head)
                )

        row = {
            "seed": seed,
            "inputs_audited": len(inputs),
            "module_evaluations": len(modules),
            "head_evaluations": len(head_errors),
            "all_finite": True,
            "max_module_relative_l2_error": max(module_errors),
            "max_head_relative_l2_error": max(head_errors),
        }
        row["max_relative_l2_error"] = max(
            row["max_module_relative_l2_error"], row["max_head_relative_l2_error"]
        )
        row["passed"] = row["all_finite"] and row["max_relative_l2_error"] <= gate
        aggregate = result.get("aggregate", {})
        for field in (
            "inputs_audited",
            "module_evaluations",
            "head_evaluations",
            "all_finite",
            "passed",
        ):
            require_equal(f"{relative} aggregate {field}", aggregate.get(field), row[field])
        for field in (
            "max_module_relative_l2_error",
            "max_head_relative_l2_error",
            "max_relative_l2_error",
        ):
            require_close(f"{relative} aggregate {field}", aggregate.get(field), row[field])
        require_close(f"{relative} aggregate gate", aggregate.get("relative_l2_gate"), gate)

        recorded = summary_rows[seed]
        require_equal(f"Conv summary seed {seed} file hash", recorded.get("source_file_sha256"), manifest_hashes[relative])
        require_equal(f"Conv summary seed {seed} checkpoint SHA", recorded.get("checkpoint_sha256"), result["identity"]["checkpoint_sha256"])
        for field in (
            "inputs_audited",
            "module_evaluations",
            "head_evaluations",
            "all_finite",
            "passed",
        ):
            require_equal(f"Conv summary seed {seed} {field}", recorded.get(field), row[field])
        for field in (
            "max_module_relative_l2_error",
            "max_head_relative_l2_error",
            "max_relative_l2_error",
        ):
            require_close(f"Conv summary seed {seed} {field}", recorded.get(field), row[field])
        recomputed.append(row)

    aggregate = {
        "checkpoints_audited": 3,
        "inputs_audited": sum(row["inputs_audited"] for row in recomputed),
        "module_evaluations": sum(row["module_evaluations"] for row in recomputed),
        "head_evaluations": sum(row["head_evaluations"] for row in recomputed),
        "all_finite": all(row["all_finite"] for row in recomputed),
        "max_relative_l2_error": max(row["max_relative_l2_error"] for row in recomputed),
        "relative_l2_gate": gate,
        "all_three_checkpoints_pass": all(row["passed"] for row in recomputed),
    }
    recorded_aggregate = summary.get("aggregate", {})
    for field, expected in aggregate.items():
        if isinstance(expected, float):
            require_close(f"Conv summary aggregate {field}", recorded_aggregate.get(field), expected)
        else:
            require_equal(f"Conv summary aggregate {field}", recorded_aggregate.get(field), expected)
    require_equal(
        "Conv immutable expected outcome",
        aggregate["all_three_checkpoints_pass"],
        config["expected_outcome"],
    )
    aggregate["_inputs_by_seed"] = inputs_by_seed
    return aggregate


def verify_joint_module_certificate(
    root: Path,
    config: dict[str, Any],
    manifest_hashes: dict[str, str],
    certificate_name: str,
    expected_inputs_by_seed: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Recompute the FFN or complete-block fidelity certificate from raw rows."""
    raw_paths = config["raw_results"]
    require_equal(f"{certificate_name} raw-result count", len(raw_paths), 3)
    summary = load_json(safe_path(root, config["summary"]))
    require_equal(
        f"{certificate_name} summary schema",
        summary.get("schema"),
        config["summary_schema"],
    )
    protocol = summary.get("protocol")
    require(isinstance(protocol, dict), f"{certificate_name}: missing protocol")
    gate = float(config["gate"]["relative_l2_error_at_most"])
    count_field = (
        "module_evaluations"
        if certificate_name == "Conv joint FFN"
        else "complete_block_evaluations"
    )
    blocks_key = (
        "joint_modules_per_input"
        if certificate_name == "Conv joint FFN"
        else "joint_blocks_per_input"
    )
    protocol_gate = (
        "relative_l2_gate"
        if certificate_name == "Conv joint FFN"
        else "complete_block_relative_l2_gate"
    )
    require_close(f"{certificate_name} protocol gate", protocol.get(protocol_gate), gate)
    summary_rows = {int(row["seed"]): row for row in summary.get("checkpoints", [])}
    require_equal(f"{certificate_name} summary seeds", set(summary_rows), {0, 1, 2})
    summary_source_map = summary.get("identity", {}).get("source_sha256")
    verify_source_identity(root, f"{certificate_name} summary", summary_source_map)

    recomputed = []
    for relative in raw_paths:
        result = load_json(safe_path(root, relative))
        require_equal(f"{relative} schema", result.get("schema"), config["raw_schema"])
        require_equal(f"{relative} mode", result.get("mode"), "full")
        require_equal(f"{relative} protocol", result.get("protocol"), protocol)
        identity = result.get("identity", {})
        require_equal(
            f"{relative} source map",
            identity.get("source_sha256"),
            summary_source_map,
        )
        seed = int(identity.get("checkpoint_seed", -1))
        require(seed in {0, 1, 2}, f"{relative}: invalid seed {seed}")
        require(seed not in {row["seed"] for row in recomputed}, f"duplicate seed {seed}")
        inputs = verify_conv_input_identity(relative, identity, result.get("inputs"), seed)
        require_equal(
            f"{relative} exact attention input identities",
            inputs,
            expected_inputs_by_seed[seed],
        )
        rows = result.get("module_rows")
        blocks = int(protocol[blocks_key])
        require(isinstance(rows, list), f"{relative}: module rows are absent")
        require_equal(f"{relative} row count", len(rows), len(inputs) * blocks)
        relatives = []
        absolutes = []
        for index, row in enumerate(rows):
            input_index, block_index = divmod(index, blocks)
            require_equal(f"{relative} row {index} rank", int(row.get("selection_rank", -1)), input_index)
            require_equal(f"{relative} row {index} block", int(row.get("block_index", -1)), block_index)
            require_equal(
                f"{relative} row {index} selection",
                row.get("selection_sha256"),
                inputs[input_index]["selection_sha256"],
            )
            require_equal(f"{relative} row {index} stack", row.get("stack"), "joint")
            require_equal(f"{relative} row {index} sequence", int(row.get("sequence_length", -1)), 107)
            require(row.get("finite") is True, f"{relative} row {index} is non-finite")
            absolute = float(row.get("max_abs_error", float("nan")))
            relative_error = float(row.get("relative_l2_error", float("nan")))
            require(
                math.isfinite(absolute)
                and math.isfinite(relative_error)
                and absolute >= 0
                and relative_error >= 0,
                f"{relative} row {index} has invalid error values",
            )
            absolutes.append(absolute)
            relatives.append(relative_error)

        row = {
            "seed": seed,
            "inputs_audited": len(inputs),
            count_field: len(rows),
            "token_rows": len(rows) * 107,
            "scalar_outputs": len(rows) * 107 * 384,
            "max_absolute_error": max(absolutes),
            "max_relative_l2_error": max(relatives),
            "all_finite": True,
        }
        row["passed"] = row["max_relative_l2_error"] <= gate
        aggregate = result.get("aggregate", {})
        for field in (
            "inputs_audited",
            count_field,
            "token_rows",
            "scalar_outputs",
            "all_finite",
            "passed",
        ):
            require_equal(f"{relative} aggregate {field}", aggregate.get(field), row[field])
        for field in ("max_absolute_error", "max_relative_l2_error"):
            require_close(f"{relative} aggregate {field}", aggregate.get(field), row[field])
        require_close(f"{relative} aggregate gate", aggregate.get("relative_l2_gate"), gate)

        recorded = summary_rows[seed]
        require_equal(
            f"{certificate_name} seed {seed} source hash",
            recorded.get("source_file_sha256"),
            manifest_hashes[relative],
        )
        require_equal(
            f"{certificate_name} seed {seed} checkpoint SHA",
            recorded.get("checkpoint_sha256"),
            identity.get("checkpoint_sha256"),
        )
        for field in (
            "inputs_audited",
            count_field,
            "token_rows",
            "scalar_outputs",
            "all_finite",
            "passed",
        ):
            require_equal(f"{certificate_name} seed {seed} {field}", recorded.get(field), row[field])
        for field in ("max_absolute_error", "max_relative_l2_error"):
            require_close(f"{certificate_name} seed {seed} {field}", recorded.get(field), row[field])
        recomputed.append(row)

    output = {
        "checkpoints_audited": 3,
        "inputs_audited": sum(row["inputs_audited"] for row in recomputed),
        count_field: sum(row[count_field] for row in recomputed),
        "token_rows": sum(row["token_rows"] for row in recomputed),
        "scalar_outputs": sum(row["scalar_outputs"] for row in recomputed),
        "max_absolute_error": max(row["max_absolute_error"] for row in recomputed),
        "max_relative_l2_error": max(row["max_relative_l2_error"] for row in recomputed),
        "all_finite": True,
        "relative_l2_gate": gate,
        "all_three_checkpoints_pass": all(row["passed"] for row in recomputed),
    }
    recorded = summary.get("aggregate", {})
    for field, expected in output.items():
        if isinstance(expected, float):
            require_close(f"{certificate_name} summary {field}", recorded.get(field), expected)
        else:
            require_equal(f"{certificate_name} summary {field}", recorded.get(field), expected)
    require_equal(
        f"{certificate_name} immutable expected outcome",
        output["all_three_checkpoints_pass"],
        config["expected_outcome"],
    )
    return output


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> dict[str, Any]:
    require(values and all(math.isfinite(value) for value in values), "invalid distribution")
    count = len(values)
    mean = math.fsum(values) / count
    sample_sd = (
        math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (count - 1))
        if count > 1
        else 0.0
    )
    return {
        "count": count,
        "mean": mean,
        "sample_sd": sample_sd,
        "min": min(values),
        "q05": quantile(values, 0.05),
        "q25": quantile(values, 0.25),
        "median": quantile(values, 0.5),
        "q75": quantile(values, 0.75),
        "q95": quantile(values, 0.95),
        "max": max(values),
    }


def verify_distribution(label: str, actual: dict[str, Any], values: list[float]) -> None:
    expected = distribution(values)
    for field, value in expected.items():
        if isinstance(value, float):
            require_close(f"{label} {field}", actual.get(field), value)
        else:
            require_equal(f"{label} {field}", actual.get(field), value)


def verify_source_fractions(label: str, sources: dict[str, Any]) -> None:
    names = {"vision", "instruction", "robot_state", "action_query"}
    require_equal(f"{label} source names", set(sources), names)
    for metric in (
        "coherent_energy_fraction",
        "signed_projection_fraction",
    ):
        values = [float(sources[source][metric]) for source in names]
        require(all(math.isfinite(value) for value in values), f"{label} {metric} is non-finite")
        total = math.fsum(values)
        if metric == "signed_projection_fraction":
            require(
                math.isclose(total, 1.0, rel_tol=1e-7, abs_tol=1e-12),
                f"{label} {metric} does not sum to one",
            )
        else:
            require_close(f"{label} {metric} sum", total, 1.0)
    if "token_energy_fraction" in sources["vision"]:
        for metric in ("token_energy_fraction", "per_source_token_energy_fraction"):
            values = [float(sources[source][metric]) for source in names]
            require(all(math.isfinite(value) for value in values), f"{label} {metric} is non-finite")
            require_close(f"{label} {metric} sum", math.fsum(values), 1.0)


def flatten_action(value: Any) -> list[float]:
    require(isinstance(value, list) and len(value) == 8, "prompt action must have eight rows")
    output = []
    for row in value:
        require(isinstance(row, list) and len(row) == 7, "prompt action row must have seven values")
        numbers = [float(item) for item in row]
        require(all(math.isfinite(item) for item in numbers), "prompt action is non-finite")
        output.extend(numbers)
    return output


def float64_sha256(values: list[float]) -> str:
    digest = hashlib.sha256()
    digest.update(b"<f8")
    digest.update(struct.pack("<2q", 8, 7))
    digest.update(struct.pack(f"<{len(values)}d", *values))
    return digest.hexdigest()


def vector_comparison(reference: list[float], candidate: list[float]) -> dict[str, float]:
    difference = [candidate_value - reference_value for reference_value, candidate_value in zip(reference, candidate, strict=True)]
    reference_norm = max(math.sqrt(math.fsum(value * value for value in reference)), 1e-30)
    candidate_norm = max(math.sqrt(math.fsum(value * value for value in candidate)), 1e-30)
    dot = math.fsum(left * right for left, right in zip(reference, candidate, strict=True))
    return {
        "relative_l2_change": math.sqrt(math.fsum(value * value for value in difference)) / reference_norm,
        "mean_absolute_change": math.fsum(abs(value) for value in difference) / len(difference),
        "signed_projection": dot / (reference_norm * reference_norm),
        "cosine": dot / (reference_norm * candidate_norm),
        "reference_l2_denominator": reference_norm,
        "cosine_denominator": reference_norm * candidate_norm,
    }


def verify_vit_modality(
    root: Path,
    config: dict[str, Any],
    manifest_hashes: dict[str, str],
) -> dict[str, Any]:
    summary = load_json(safe_path(root, config["summary"]))
    require_equal("modality summary schema", summary.get("schema"), config["summary_schema"])
    protocol = summary.get("protocol")
    require(isinstance(protocol, dict), "modality summary protocol is absent")
    gate = float(config["gate"]["relative_l2_error_at_most"])
    require_close("modality protocol gate", protocol.get("relative_l2_gate"), gate)
    summary_rows = {int(row["seed"]): row for row in summary.get("checkpoints", [])}
    require_equal("modality summary seeds", set(summary_rows), {0, 1, 2})
    summary_source_map = summary.get("identity", {}).get("source_sha256")
    verify_source_identity(root, "modality summary", summary_source_map)
    source_names = ("vision", "instruction", "robot_state", "action_query")
    head_coherent = {source: [] for source in source_names}
    module_coherent = {source: [] for source in source_names}
    prompt_all = []
    prompt_mismatched = []
    recomputed = []

    for relative in config["raw_results"]:
        result = load_json(safe_path(root, relative))
        require_equal(f"{relative} schema", result.get("schema"), config["raw_schema"])
        require_equal(f"{relative} mode", result.get("mode"), "full")
        require_equal(f"{relative} protocol", result.get("protocol"), protocol)
        identity = result.get("identity", {})
        require_equal(
            f"{relative} source map",
            identity.get("source_sha256"),
            summary_source_map,
        )
        seed = int(identity.get("checkpoint_seed", -1))
        require(seed in {0, 1, 2}, f"{relative}: invalid seed {seed}")
        require(seed not in {row["seed"] for row in recomputed}, f"duplicate modality seed {seed}")
        require_equal(f"{relative} cache SHA", identity.get("cache_sha256"), "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662")
        require_equal(f"{relative} provenance job", str(identity.get("provenance_job_id")), "830988")
        layout = result.get("layout_confirmation", {})
        require(layout.get("deployed_manual_hidden_exact") is True, f"{relative}: hidden layout differs")
        require(layout.get("deployed_causal_mask_exact") is True, f"{relative}: causal mask differs")
        require_close(f"{relative} layout absolute error", layout.get("max_abs_error"), 0.0)
        require_close(f"{relative} layout relative error", layout.get("relative_l2_error"), 0.0)

        inputs = result.get("inputs")
        rows = result.get("module_rows")
        require(isinstance(inputs, list) and len(inputs) == 128, f"{relative}: expected 128 inputs")
        require_equal(f"{relative} input ranks", [int(row.get("selection_rank", -1)) for row in inputs], list(range(128)))
        task_counts = {str(task): 0 for task in range(10)}
        for row in inputs:
            task_counts[str(int(row.get("official_task_index", -1)))] += 1
        require_equal(f"{relative} official task quotas", task_counts, protocol["full_official_task_quotas"])
        require(isinstance(rows, list) and len(rows) == 1024, f"{relative}: expected 1024 module rows")
        errors = []
        for index, row in enumerate(rows):
            input_index, block_index = divmod(index, 8)
            require_equal(f"{relative} row {index} rank", int(row.get("selection_rank", -1)), input_index)
            require_equal(f"{relative} row {index} block", int(row.get("block_index", -1)), block_index)
            heads = row.get("heads")
            require(isinstance(heads, list) and len(heads) == 12, f"{relative} row {index}: expected 12 heads")
            for head_index, head in enumerate(heads):
                require_equal(f"{relative} row {index} head index", int(head.get("head_index", -1)), head_index)
                errors.append(error_value(f"{relative} row {index} head {head_index}", head.get("group_sum_error", {})))
                sources = head.get("sources", {})
                verify_source_fractions(f"{relative} row {index} head {head_index}", sources)
                for source in source_names:
                    head_coherent[source].append(float(sources[source]["coherent_energy_fraction"]))
            module = row.get("module", {})
            for error_name in (
                "group_sum_prebias_error",
                "group_sum_plus_bias_error",
                "deployed_float32_vs_grouped_float64_error",
                "deployed_gain_scaled_update_error",
            ):
                errors.append(error_value(f"{relative} row {index} {error_name}", module.get(error_name, {})))
            sources = module.get("sources", {})
            verify_source_fractions(f"{relative} row {index} module", sources)
            for source in source_names:
                module_coherent[source].append(float(sources[source]["coherent_energy_fraction"]))

        prompts = result.get("prompt_permutation_rows")
        require(isinstance(prompts, list) and len(prompts) == 100, f"{relative}: expected 100 prompt rows")
        predictions = {}
        comparisons = {}
        for row in prompts:
            observation = int(row.get("observation_official_task_index", -1))
            prompt = int(row.get("prompt_official_task_index", -1))
            require((observation, prompt) not in predictions, f"{relative}: duplicate prompt pair")
            action = flatten_action(row.get("physical_action_prediction"))
            require_equal(f"{relative} prompt action SHA", row.get("physical_action_prediction_sha256"), float64_sha256(action))
            predictions[(observation, prompt)] = action
            comparisons[(observation, prompt)] = row.get("comparison_to_matched_prompt", {})
        require_equal(f"{relative} prompt pairs", set(predictions), {(observation, prompt) for observation in range(10) for prompt in range(10)})
        for key, candidate in predictions.items():
            observation, prompt = key
            expected = vector_comparison(predictions[(observation, observation)], candidate)
            actual = comparisons[key]
            require_equal(f"{relative} prompt comparison fields", set(actual), set(expected))
            for field, value in expected.items():
                require_close(f"{relative} prompt {observation}/{prompt} {field}", actual.get(field), value)
            prompt_all.append(expected["relative_l2_change"])
            if observation != prompt:
                prompt_mismatched.append(expected["relative_l2_change"])

        maximum = max(errors)
        row = {
            "seed": seed,
            "inputs_audited": 128,
            "module_evaluations": 1024,
            "head_evaluations": 12288,
            "source_group_evaluations": 49152,
            "prompt_permutation_evaluations": 100,
            "max_relative_l2_error": maximum,
            "passed": maximum <= gate,
        }
        aggregate = result.get("aggregate", {})
        for field in (
            "inputs_audited",
            "module_evaluations",
            "head_evaluations",
            "source_group_evaluations",
            "prompt_permutation_evaluations",
            "passed",
        ):
            require_equal(f"{relative} aggregate {field}", aggregate.get(field), row[field])
        require(aggregate.get("all_finite") is True and aggregate.get("prompt_all_finite") is True, f"{relative}: aggregate is non-finite")
        require_close(f"{relative} aggregate maximum", aggregate.get("max_relative_l2_error"), maximum)
        recorded = summary_rows[seed]
        require_equal(f"modality seed {seed} source hash", recorded.get("source_file_sha256"), manifest_hashes[relative])
        require_equal(f"modality seed {seed} checkpoint SHA", recorded.get("checkpoint_sha256"), identity.get("checkpoint_sha256"))
        for field, value in row.items():
            if field == "seed":
                continue
            if isinstance(value, float):
                require_close(f"modality seed {seed} {field}", recorded.get(field), value)
            else:
                require_equal(f"modality seed {seed} {field}", recorded.get(field), value)
        recomputed.append(row)
        del result

    output = {
        "checkpoints_audited": 3,
        "inputs_audited": 384,
        "module_evaluations": 3072,
        "head_evaluations": 36864,
        "source_group_evaluations": 147456,
        "prompt_permutation_evaluations": 300,
        "max_relative_l2_error": max(row["max_relative_l2_error"] for row in recomputed),
        "relative_l2_gate": gate,
        "all_three_checkpoints_pass": all(row["passed"] for row in recomputed),
    }
    for field, value in output.items():
        actual = summary.get("aggregate", {}).get(field)
        if isinstance(value, float):
            require_close(f"modality summary {field}", actual, value)
        else:
            require_equal(f"modality summary {field}", actual, value)
    for source in source_names:
        verify_distribution(
            f"modality head coherent {source}",
            summary["head_contribution_distributions"]["overall"][source]["coherent_energy_fraction"],
            head_coherent[source],
        )
        verify_distribution(
            f"modality module coherent {source}",
            summary["module_contribution_distributions"]["overall"][source]["coherent_energy_fraction"],
            module_coherent[source],
        )
    prompt_summary = summary["prompt_permutation_characterization"]
    verify_distribution(
        "modality prompt changes including matched",
        prompt_summary["relative_l2_change_distribution_including_matched"],
        prompt_all,
    )
    verify_distribution(
        "modality prompt changes mismatched only",
        prompt_summary["relative_l2_change_distribution_mismatched_only"],
        prompt_mismatched,
    )
    require_equal("modality immutable expected outcome", output["all_three_checkpoints_pass"], config["expected_outcome"])
    return output


def quadratic_roots(coefficients: list[float]) -> list[complex]:
    require_equal("denominator coefficient count", len(coefficients), 3)
    c, b, a = coefficients
    require(a != 0.0, "denominator leading coefficient is zero")
    discriminant = cmath.sqrt(complex(b * b - 4.0 * a * c, 0.0))
    return [(-b - discriminant) / (2.0 * a), (-b + discriminant) / (2.0 * a)]


def verify_rational_site(
    label: str,
    site: dict[str, Any],
    gates: dict[str, float],
) -> dict[str, Any]:
    require_equal(f"{label} variant", site.get("variant"), "pade")
    require_equal(f"{label} degree", int(site.get("degree", -1)), 2)
    require(site.get("initialized") is True, f"{label}: site was not initialized")
    require(site.get("evaluator_frozen_flag") is True, f"{label}: evaluator was not frozen")
    rows = int(site.get("normalization_rows", -1))
    in_range = int(site.get("rows_v_in_0p1_10", -1))
    local_rows = int(site.get("rows_local_scale_error_at_or_below_threshold", -1))
    nonfinite = int(site.get("nonfinite_rows", -1))
    require(rows > 0 and 0 <= in_range <= rows, f"{label}: invalid range counts")
    require(0 <= local_rows <= rows, f"{label}: invalid local-error counts")
    require_equal(f"{label} nonfinite rows", nonfinite, 0)
    range_fraction = in_range / rows
    local_fraction = local_rows / rows
    require_close(f"{label} range fraction", site.get("fraction_rows_v_in_0p1_10"), range_fraction)
    require_close(
        f"{label} local-error fraction",
        site.get("fraction_rows_local_scale_error_at_or_below_threshold"),
        local_fraction,
    )
    require_close(
        f"{label} local-error threshold",
        site.get("local_scale_relative_error_threshold"),
        gates["secondary_local_scale_relative_error_threshold"],
    )

    certificate = site.get("denominator_certificate", {})
    raw_coefficients = certificate.get("denominator_coefficients_float64")
    require(isinstance(raw_coefficients, list), f"{label}: denominator coefficients are absent")
    coefficients = [float(value) for value in raw_coefficients]
    positive_coefficients = all(math.isfinite(value) and value > 0.0 for value in coefficients)
    roots = quadratic_roots(coefficients)
    positive_real_roots = sum(abs(root.imag) <= 1e-12 and root.real >= 0.0 for root in roots)
    recorded_roots = certificate.get("denominator_roots_float64")
    require(isinstance(recorded_roots, list) and len(recorded_roots) == 2, f"{label}: roots are absent")
    actual_roots = sorted(roots, key=lambda value: (value.real, value.imag))
    expected_roots = sorted(
        [complex(float(row["real"]), float(row["imag"])) for row in recorded_roots],
        key=lambda value: (value.real, value.imag),
    )
    for index, (actual, expected) in enumerate(zip(actual_roots, expected_roots)):
        require_close(f"{label} root {index} real", actual.real, expected.real)
        require_close(f"{label} root {index} imag", actual.imag, expected.imag)
    certificate_pass = positive_coefficients and positive_real_roots == 0
    require_equal(f"{label} positive-coefficient flag", certificate.get("coefficients_strictly_positive"), positive_coefficients)
    require_equal(f"{label} positive-real-root count", certificate.get("positive_real_root_count"), positive_real_roots)
    require_equal(f"{label} certificate pass", certificate.get("certificate_pass"), certificate_pass)
    denominator_min = float(site.get("observed_denominator_min", float("nan")))
    require(math.isfinite(denominator_min) and denominator_min > 0.0, f"{label}: observed denominator is not positive")
    primary = certificate_pass and denominator_min > 0.0 and nonfinite == 0
    range_pass = range_fraction >= gates["secondary_each_site_fraction_rows_v_in_0p1_10_at_least"]
    local_pass = local_fraction >= gates["secondary_each_site_fraction_local_scale_error_at_or_below_threshold_at_least"]
    require_equal(f"{label} primary denominator gate", site.get("primary_denominator_pass"), primary)
    require_equal(f"{label} secondary range gate", site.get("secondary_range_pass"), range_pass)
    require_equal(f"{label} secondary local gate", site.get("secondary_local_error_pass"), local_pass)
    return {
        "rows": rows,
        "in_range": in_range,
        "local_rows": local_rows,
        "range_fraction": range_fraction,
        "local_fraction": local_fraction,
        "denominator_min": denominator_min,
        "v_min": float(site["observed_v_min"]),
        "v_max": float(site["observed_v_max"]),
        "p99": float(site["local_scale_relative_error_p99"]),
        "maximum": float(site["local_scale_relative_error_max"]),
        "primary": primary,
        "range_pass": range_pass,
        "local_pass": local_pass,
    }


def verify_rational(
    root: Path,
    config: dict[str, Any],
    manifest_hashes: dict[str, str],
) -> dict[str, Any]:
    raw_paths = config["raw_results"]
    require_equal("RationalNorm raw-result count", len(raw_paths), 3)
    summary = load_json(safe_path(root, config["summary"]))
    require_equal("RationalNorm summary schema", summary.get("schema"), config["summary_schema"])
    summary_source_map = summary.get("inputs", {}).get("source_sha256")
    verify_source_identity(root, "RationalNorm summary", summary_source_map)
    gates = config["gates"]
    for field, expected in gates.items():
        require_close(f"RationalNorm summary frozen gate {field}", summary.get("frozen_gates", {}).get(field), expected)
    summary_rows = {int(row["seed"]): row for row in summary.get("checkpoints", [])}
    require_equal("RationalNorm summary seeds", set(summary_rows), {0, 1, 2})

    recomputed = []
    for relative in raw_paths:
        result = load_json(safe_path(root, relative))
        require_equal(f"{relative} schema", result.get("schema"), config["raw_schema"])
        require_equal(
            f"{relative} source map",
            result.get("implementation", {}).get("source_sha256"),
            summary_source_map,
        )
        require_equal(f"{relative} mode", result.get("mode"), "full")
        for field, expected in gates.items():
            require_close(f"{relative} frozen gate {field}", result.get("frozen_gates", {}).get(field), expected)
        seed = int(result.get("checkpoint", {}).get("seed", -1))
        require(seed in {0, 1, 2}, f"{relative}: invalid seed {seed}")
        require(seed not in {row["seed"] for row in recomputed}, f"duplicate RationalNorm seed {seed}")
        sites = result.get("rational_sites")
        require(isinstance(sites, list) and len(sites) == 53, f"{relative}: expected 53 sites")
        site_rows = [
            verify_rational_site(f"{relative} site {index}", site, gates)
            for index, site in enumerate(sites)
        ]
        rows = sum(site["rows"] for site in site_rows)
        in_range = sum(site["in_range"] for site in site_rows)
        local_rows = sum(site["local_rows"] for site in site_rows)
        aggregate = result.get("aggregate", {})
        require_equal(f"{relative} aggregate row count", aggregate.get("normalization_rows"), rows)
        require_equal(f"{relative} aggregate in-range count", aggregate.get("rows_v_in_0p1_10"), in_range)
        require_equal(f"{relative} aggregate local-error count", aggregate.get("rows_local_scale_error_at_or_below_threshold"), local_rows)
        require_close(f"{relative} aggregate range fraction", aggregate.get("fraction_rows_v_in_0p1_10"), in_range / rows)
        require_close(f"{relative} aggregate local fraction", aggregate.get("fraction_rows_local_scale_error_at_or_below_threshold"), local_rows / rows)

        action = result.get("action_reference", {})
        physical_diff = float(action["physical_squared_difference_sum"])
        physical_reference = float(action["physical_squared_reference_sum"])
        normalized_diff = float(action["normalized_squared_difference_sum"])
        normalized_reference = float(action["normalized_squared_reference_sum"])
        require(physical_diff >= 0.0 and physical_reference > 0.0, f"{relative}: invalid physical action sums")
        require(normalized_diff >= 0.0 and normalized_reference > 0.0, f"{relative}: invalid normalized action sums")
        physical_nrmse = math.sqrt(physical_diff / physical_reference)
        normalized_nrmse = math.sqrt(normalized_diff / normalized_reference)
        action_nrmse = max(physical_nrmse, normalized_nrmse)
        require_close(f"{relative} physical NRMSE", action.get("physical_nrmse"), physical_nrmse)
        require_close(f"{relative} normalized NRMSE", action.get("normalized_nrmse"), normalized_nrmse)
        require_close(f"{relative} primary NRMSE", action.get("primary_nrmse"), action_nrmse)
        primary_denominator = all(site["primary"] for site in site_rows)
        primary_action = action_nrmse <= gates["primary_action_nrmse_at_most"]
        secondary_range = all(site["range_pass"] for site in site_rows)
        secondary_local = all(site["local_pass"] for site in site_rows)
        outcomes = {
            "primary_action_nrmse_pass": primary_action,
            "primary_denominator_certificate_pass": primary_denominator,
            "primary_pass": primary_denominator and primary_action,
            "secondary_local_scale_error_pass": secondary_local,
            "secondary_pass": secondary_range and secondary_local,
            "secondary_range_pass": secondary_range,
        }
        require_equal(f"{relative} gates", result.get("gates"), outcomes)

        row = {
            "seed": seed,
            "normalization_rows": rows,
            "fraction_rows_v_in_0p1_10": in_range / rows,
            "minimum_site_fraction_rows_v_in_0p1_10": min(site["range_fraction"] for site in site_rows),
            "fraction_rows_local_scale_error_at_or_below_threshold": local_rows / rows,
            "minimum_site_fraction_rows_local_scale_error_at_or_below_threshold": min(site["local_fraction"] for site in site_rows),
            "observed_v_min": min(site["v_min"] for site in site_rows),
            "observed_v_max": max(site["v_max"] for site in site_rows),
            "observed_denominator_min": min(site["denominator_min"] for site in site_rows),
            "local_scale_relative_error_p99": float(aggregate["local_scale_relative_error_p99"]),
            "local_scale_relative_error_max": max(site["maximum"] for site in site_rows),
            "action_nrmse": action_nrmse,
            "physical_action_nrmse": physical_nrmse,
            "normalized_action_nrmse": normalized_nrmse,
            "physical_action_squared_difference_sum": physical_diff,
            "physical_action_squared_reference_sum": physical_reference,
            "normalized_action_squared_difference_sum": normalized_diff,
            "normalized_action_squared_reference_sum": normalized_reference,
            "gates": outcomes,
        }
        recorded = summary_rows[seed]
        require_equal(f"RationalNorm summary seed {seed} checkpoint", recorded.get("checkpoint"), result.get("checkpoint"))
        require_equal(f"RationalNorm summary seed {seed} model parameters", recorded.get("model_parameters"), result.get("model", {}).get("parameters"))
        for field in (
            "normalization_rows",
            "gates",
        ):
            require_equal(f"RationalNorm summary seed {seed} {field}", recorded.get(field), row[field])
        for field in (
            "fraction_rows_v_in_0p1_10",
            "minimum_site_fraction_rows_v_in_0p1_10",
            "fraction_rows_local_scale_error_at_or_below_threshold",
            "minimum_site_fraction_rows_local_scale_error_at_or_below_threshold",
            "observed_v_min",
            "observed_v_max",
            "observed_denominator_min",
            "local_scale_relative_error_p99",
            "local_scale_relative_error_max",
            "action_nrmse",
            "physical_action_nrmse",
            "normalized_action_nrmse",
            "physical_action_squared_difference_sum",
            "physical_action_squared_reference_sum",
            "normalized_action_squared_difference_sum",
            "normalized_action_squared_reference_sum",
        ):
            require_close(f"RationalNorm summary seed {seed} {field}", recorded.get(field), row[field])
        summary_hashes = summary.get("inputs", {}).get("result_sha256", {})
        matching_hashes = [value for path, value in summary_hashes.items() if Path(path).name == Path(relative).name]
        require_equal(f"RationalNorm summary seed {seed} source hash count", len(matching_hashes), 1)
        require_equal(f"RationalNorm summary seed {seed} source hash", matching_hashes[0], manifest_hashes[relative])
        recomputed.append(row)

    physical_diff = sum(row["physical_action_squared_difference_sum"] for row in recomputed)
    physical_reference = sum(row["physical_action_squared_reference_sum"] for row in recomputed)
    normalized_diff = sum(row["normalized_action_squared_difference_sum"] for row in recomputed)
    normalized_reference = sum(row["normalized_action_squared_reference_sum"] for row in recomputed)
    aggregate = {
        "checkpoints": 3,
        "selected_samples": 3 * int(summary["protocol"]["full_samples_per_checkpoint"]),
        "rational_sites_per_checkpoint": 53,
        "normalization_rows": sum(row["normalization_rows"] for row in recomputed),
        "minimum_fraction_rows_v_in_0p1_10": min(row["minimum_site_fraction_rows_v_in_0p1_10"] for row in recomputed),
        "minimum_fraction_rows_local_scale_error_at_or_below_threshold": min(row["minimum_site_fraction_rows_local_scale_error_at_or_below_threshold"] for row in recomputed),
        "maximum_local_scale_relative_error_p99": max(row["local_scale_relative_error_p99"] for row in recomputed),
        "minimum_observed_denominator": min(row["observed_denominator_min"] for row in recomputed),
        "observed_v_min": min(row["observed_v_min"] for row in recomputed),
        "observed_v_max": max(row["observed_v_max"] for row in recomputed),
        "pooled_physical_action_nrmse": math.sqrt(physical_diff / physical_reference),
        "pooled_normalized_action_nrmse": math.sqrt(normalized_diff / normalized_reference),
        "maximum_checkpoint_action_nrmse": max(row["action_nrmse"] for row in recomputed),
    }
    recorded_aggregate = summary.get("aggregate", {})
    for field, expected in aggregate.items():
        if isinstance(expected, float):
            require_close(f"RationalNorm summary aggregate {field}", recorded_aggregate.get(field), expected)
        else:
            require_equal(f"RationalNorm summary aggregate {field}", recorded_aggregate.get(field), expected)
    outcomes = {
        "all_three_primary_pass": all(row["gates"]["primary_pass"] for row in recomputed),
        "all_three_secondary_pass": all(row["gates"]["secondary_pass"] for row in recomputed),
    }
    require_equal("RationalNorm summary gates", summary.get("gates"), outcomes)
    require_equal("RationalNorm immutable primary outcome", outcomes["all_three_primary_pass"], config["expected_primary_outcome"])
    require_equal("RationalNorm immutable secondary outcome", outcomes["all_three_secondary_pass"], config["expected_secondary_outcome"])
    return {**aggregate, **outcomes}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Repository root containing athena/results. Defaults to the verifier's repository.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_json(ARTIFACT_DIR / "manifest.json")
        hashes = verify_manifest_hashes(args.root.resolve(), manifest)
        conv = verify_conv(args.root.resolve(), manifest["certificates"]["conv_joint_attention"], hashes)
        ffn = verify_joint_module_certificate(
            args.root.resolve(),
            manifest["certificates"]["conv_joint_ffn"],
            hashes,
            "Conv joint FFN",
            conv["_inputs_by_seed"],
        )
        block = verify_joint_module_certificate(
            args.root.resolve(),
            manifest["certificates"]["conv_joint_block"],
            hashes,
            "Conv complete joint block",
            conv["_inputs_by_seed"],
        )
        modality = verify_vit_modality(
            args.root.resolve(), manifest["certificates"]["vit_modality"], hashes
        )
        rational = verify_rational(args.root.resolve(), manifest["certificates"]["rational_norm"], hashes)
    except (KeyError, TypeError, ValueError, VerificationError) as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"HASHES PASS: {len(hashes)} immutable evidence files")
    print(
        "CONV JOINT ATTENTION PASS: "
        f"{conv['head_evaluations']} head-input cases, "
        f"max relative L2 {conv['max_relative_l2_error']:.12g} <= "
        f"{conv['relative_l2_gate']:.12g}"
    )
    print(
        "CONV JOINT FFN PASS: "
        f"{ffn['module_evaluations']} module-input cases, "
        f"max relative L2 {ffn['max_relative_l2_error']:.12g} <= "
        f"{ffn['relative_l2_gate']:.12g}"
    )
    print(
        "CONV COMPLETE JOINT BLOCK PASS: "
        f"{block['complete_block_evaluations']} block-input cases, "
        f"max relative L2 {block['max_relative_l2_error']:.12g} <= "
        f"{block['relative_l2_gate']:.12g}"
    )
    print(
        "VIT MODALITY DECOMPOSITION PASS: "
        f"{modality['head_evaluations']} head-input cases and "
        f"{modality['source_group_evaluations']} source-group evaluations, "
        f"max relative L2 {modality['max_relative_l2_error']:.12g} <= "
        f"{modality['relative_l2_gate']:.12g}"
    )
    print(
        "RATIONALNORM PRIMARY PASS: "
        f"{rational['normalization_rows']} rows, "
        f"max checkpoint action NRMSE {rational['maximum_checkpoint_action_nrmse']:.12g} <= 0.001"
    )
    print(
        "RATIONALNORM SECONDARY FAIL (EXPECTED): "
        "the preregistered approximation-coverage gate is not claimed"
    )
    print("VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
