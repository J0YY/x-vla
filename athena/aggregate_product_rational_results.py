#!/usr/bin/env python3
"""Aggregate the frozen three-seed ProductRoutingHead capability protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_RUNNING_AS_ENTRYPOINT = __name__ == "__main__"
_MANIFEST_SCHEMA = "xvla_product_pade_rational_vla_v2_source_manifest"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant is forbidden: {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Nonfinite JSON float is forbidden")
    return parsed
_EXPECTED_SOURCE_CLOSURE = (
    "athena/__init__.py",
    "athena/aggregate_product_rational_results.py",
    "athena/eval_product_rational_checkpoint.py",
    "athena/prepare_product_rational_launch.py",
    "athena/product_rational_protocol.py",
    "athena/stage_product_rational_launch.py",
    "athena/train_product_rational_checkpoint.py",
    "scripts/__init__.py",
    "scripts/odt_direct_only_compliance.py",
    "tests/test_product_rational_production_lane.py",
    "xvla/__init__.py",
    "xvla/models/__init__.py",
    "xvla/models/lm.py",
    "xvla/models/vit.py",
    "xvla/models/vla.py",
    "xvla/nn/__init__.py",
    "xvla/nn/attention.py",
    "xvla/nn/baselines.py",
    "xvla/nn/bilinear.py",
    "xvla/nn/block.py",
    "xvla/nn/flow_action.py",
    "xvla/nn/homogeneous.py",
    "xvla/nn/normalization.py",
    "xvla/nn/product_routing.py",
    "xvla/nn/projector.py",
    "xvla/nn/quantile_action.py",
)
_EXPECTED_AUTHENTICATED_INPUTS = (
    "athena/results/cache_provenance_libero_object.json",
)
_EXPECTED_LAUNCH_CLOSURE = (
    "athena/slurm_product_rational_aggregate.sbatch",
    "athena/slurm_product_rational_eval.sbatch",
    "athena/slurm_product_rational_preflight.sbatch",
    "athena/slurm_product_rational_train.sbatch",
    "athena/submit_product_rational_v2.sh",
)


def _verify_preimport_source_manifest(arguments: list[str]) -> dict[str, Any]:
    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument == "--source-manifest":
            if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
                raise RuntimeError("--source-manifest requires one value")
            values.append(arguments[index + 1])
        elif argument.startswith("--source-manifest="):
            values.append(argument.split("=", 1)[1])
        elif argument.startswith("--source-man"):
            raise RuntimeError("Abbreviated source-manifest options are forbidden")
    if len(values) != 1 or not values[0]:
        raise RuntimeError("Exactly one --source-manifest is required")
    manifest_path = Path(values[0])
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError("Pre-import source manifest is missing or nonphysical")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"Duplicate source-manifest key {key!r}")
            result[key] = value
        return result

    def canonical_digest(value: Any) -> str:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    payload = json.loads(
        manifest_path.read_text(),
        object_pairs_hook=reject_duplicates,
        parse_constant=_reject_json_constant,
        parse_float=_finite_json_float,
    )
    if payload.get("schema") != _MANIFEST_SCHEMA:
        raise RuntimeError("Pre-import source manifest schema differs")
    observed_sections: dict[str, dict[str, str]] = {}
    section_specs = (
        ("source_closure", _EXPECTED_SOURCE_CLOSURE, "source_bundle_sha256"),
        (
            "authenticated_inputs",
            _EXPECTED_AUTHENTICATED_INPUTS,
            "authenticated_inputs_bundle_sha256",
        ),
        ("launch_closure", _EXPECTED_LAUNCH_CLOSURE, "launch_bundle_sha256"),
    )
    for section, exact_paths, bundle_key in section_specs:
        expected = payload.get(section)
        if not isinstance(expected, dict) or tuple(sorted(expected)) != tuple(
            sorted(exact_paths)
        ):
            raise RuntimeError(f"Pre-import source manifest {section} differs")
        if payload.get(bundle_key) != canonical_digest(dict(sorted(expected.items()))):
            raise RuntimeError(f"Pre-import source manifest {bundle_key} differs")
        observed: dict[str, str] = {}
        for relative, expected_digest in expected.items():
            if (
                not isinstance(relative, str)
                or not isinstance(expected_digest, str)
                or len(expected_digest) != 64
                or any(char not in "0123456789abcdef" for char in expected_digest)
            ):
                raise RuntimeError("Pre-import source manifest contains an invalid entry")
            candidate = PROJECT_ROOT / relative
            if candidate.is_symlink():
                raise RuntimeError(f"Pre-import lane file is a link: {relative}")
            path = candidate.resolve()
            if path == PROJECT_ROOT or PROJECT_ROOT not in path.parents:
                raise RuntimeError("Pre-import source path escapes the project root")
            if not path.is_file():
                raise RuntimeError(f"Pre-import lane file is missing: {relative}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected_digest:
                raise RuntimeError(f"Pre-import lane file hash changed: {relative}")
            observed[relative] = digest
        observed_sections[section] = observed
    bound = {
        "schema": payload.get("schema"),
        "source_closure": payload.get("source_closure"),
        "source_bundle_sha256": payload.get("source_bundle_sha256"),
        "authenticated_inputs": payload.get("authenticated_inputs"),
        "authenticated_inputs_bundle_sha256": payload.get(
            "authenticated_inputs_bundle_sha256"
        ),
        "launch_closure": payload.get("launch_closure"),
        "launch_bundle_sha256": payload.get("launch_bundle_sha256"),
    }
    if payload.get("manifest_bundle_sha256") != canonical_digest(bound):
        raise RuntimeError("Pre-import aggregate manifest digest differs")
    return {
        "path": manifest_path.as_posix(),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_sha256": observed_sections["source_closure"],
        "authenticated_inputs_sha256": observed_sections["authenticated_inputs"],
        "launch_sha256": observed_sections["launch_closure"],
        "manifest_bundle_sha256": payload["manifest_bundle_sha256"],
    }


_PREIMPORT_SOURCE_VERIFICATION = (
    _verify_preimport_source_manifest(sys.argv[1:]) if _RUNNING_AS_ENTRYPOINT else None
)

if _RUNNING_AS_ENTRYPOINT:
    from scripts.odt_direct_only_compliance import (
        assert_direct_only_runtime_guard,
        audit_direct_only_launch,
        direct_only_runtime_report,
        install_direct_only_runtime_guard,
    )

    _TRANSITIVE_STATIC_AUDIT = audit_direct_only_launch(
        PROJECT_ROOT,
        tuple(PROJECT_ROOT / path for path in _EXPECTED_SOURCE_CLOSURE),
        require_direct_qr=False,
    )
    if _TRANSITIVE_STATIC_AUDIT["source_sha256"] != _PREIMPORT_SOURCE_VERIFICATION[
        "source_sha256"
    ]:
        raise RuntimeError("Authenticated Python closure differs from transitive static audit")
    for field in (
        "prohibited_calls_found",
        "prohibited_self_overlap_sites",
        "guarded_dormant_spectral_norm_sites",
        "duplicate_top_level_definition_sites",
    ):
        if _TRANSITIVE_STATIC_AUDIT[field] != []:
            raise RuntimeError(f"Product lane static audit did not close {field}")
    _RUNTIME_GUARD_AT_IMPORT = install_direct_only_runtime_guard()
else:
    _TRANSITIVE_STATIC_AUDIT = None
    _RUNTIME_GUARD_AT_IMPORT = None

from athena.product_rational_protocol import (
    ACTION_DIM,
    ACTION_HORIZON,
    CAPABILITY_SIMULATOR_BOUNDARY,
    DEPLOYMENT_COMPLETION_SCHEMA,
    DEPLOYMENT_CLAIM_BOUNDARY,
    EVALUATION_EPISODES_PER_TASK,
    EXPECTED_EVALUATION_ENVIRONMENT,
    EXPECTED_TASK_PROTOCOL,
    N_FACTORS,
    OFFICIAL_INIT_STATE_ROW_HASH_LIST_SHA256,
    OFFICIAL_TASK_LANGUAGES,
    PRIMARY_CAPABILITY_FLOOR,
    PRIMARY_CAPABILITY_SEED,
    RECIPE_VERSION,
    SCHEMA,
    SEEDS,
    SHARDS,
    aggregate_result_path,
    canonical_sha256,
    checkpoint_path,
    deployment_completion_path,
    expected_task_mapping_record,
    evaluation_protocol,
    evaluation_result_path,
    file_sha256,
    json_type_exact_equal,
    load_and_verify_source_manifest,
    load_and_validate_deployment_completion,
    metadata_path,
    precalibration_state_path,
    seed0_capability_gate_path,
    source_manifest_path,
    source_snapshot,
    training_result_path,
    validate_task_mapping,
    write_json_exclusive,
)


def _option_count(arguments: list[str], option: str) -> int:
    return sum(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    arguments = sys.argv[1:]
    parsed = parser.parse_args(arguments)
    for option in ("--run-root", "--source-manifest"):
        if _option_count(arguments, option) != 1:
            parser.error(f"{option} must occur exactly once")
    return parsed


def read_json_physical(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Required JSON artifact is missing or nonphysical: {path}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(),
        object_pairs_hook=reject_duplicates,
        parse_constant=_reject_json_constant,
        parse_float=_finite_json_float,
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    return value


_CAPABILITY_CONDITION_KEYS = {
    "primary_seed_is_preregistered_seed0",
    "primary_overall_at_least_floor",
    "all_three_seed_reports_present",
    "all_four_shards_per_seed_present_and_unique",
    "exactly_500_episodes_per_seed",
    "canonical_task_and_initial_state_protocol",
    "no_nonfinite_or_decode_failures",
    "all_simulator_transitions_strictly_validated",
    "no_best_seed_post_selection",
}
_TRAINING_ARTIFACT_KEYS = {
    "checkpoint",
    "metadata",
    "training_result",
    "precalibration_ema_state",
    "deployment_completion",
}
_SEED_SUMMARY_KEYS = {"success_count", "episode_count", "overall", "per_task"}
_AGGREGATE_TOP_LEVEL_KEYS = {
    "schema",
    "recipe_version",
    "source_manifest_sha256",
    "manifest_bundle_sha256",
    "evaluation_result_sha256",
    "training_artifact_sha256",
    "seeds",
    "preregistered_primary_seed",
    "preregistered_primary_floor",
    "seed_summaries",
    "three_seed_mean",
    "three_seed_population_std",
    "capability_gate_conditions",
    "capability_gate_pass",
    "validated_transition_counts",
    "task_protocol",
    "transitive_direct_only_static_audit",
    "runtime_direct_only_guard_at_import",
    "runtime_direct_only_guard_final",
    "claim_boundary",
}
_CAPABILITY_GATE_TOP_LEVEL_KEYS = {
    "schema",
    "recipe_version",
    "preregistered_before_evaluation",
    "primary_seed",
    "required_overall",
    "observed_overall",
    "conditions",
    "approved_for_capable_global_claim",
    "capability_condition_satisfied_for_future_composite_attestation",
    "exact_odt_composite_attestation_approved",
    "checkpoint",
    "checkpoint_sha256",
    "metadata_sha256",
    "training_result_sha256",
    "precalibration_ema_state_sha256",
    "deployment_completion_sha256",
    "aggregate",
    "aggregate_sha256",
    "source_manifest_sha256",
    "manifest_bundle_sha256",
    "does_not_select_best_seed_post_hoc",
    "exact_odt_result_required_separately",
    "claim_boundary",
}
_DEPLOYMENT_COMPLETION_KEYS = {
    "schema",
    "complete",
    "seed",
    "mode",
    "checkpoint",
    "checkpoint_sha256",
    "metadata",
    "metadata_sha256",
    "training_result",
    "training_result_sha256",
    "precalibration_ema_state",
    "precalibration_ema_state_sha256",
    "source_manifest_sha256",
    "manifest_bundle_sha256",
    "artifact_bundle_sha256",
}


def _physical_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_physical_bytes_once(
    path: Path, *, label: str
) -> tuple[bytes, str, tuple[int, ...]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or nonphysical: {path}")
    path_stat = os.lstat(path)
    if not stat.S_ISREG(path_stat.st_mode):
        raise RuntimeError(f"{label} is not a regular file: {path}")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        raw = handle.read()
        after = os.fstat(handle.fileno())
    identity = _physical_identity(before)
    if identity != _physical_identity(after):
        raise RuntimeError(f"{label} changed while being read: {path}")
    if (path_stat.st_dev, path_stat.st_ino) != (before.st_dev, before.st_ino):
        raise RuntimeError(f"{label} path was substituted while opening: {path}")
    return raw, hashlib.sha256(raw).hexdigest(), identity


def _read_json_physical_once(
    path: Path, *, label: str
) -> tuple[dict[str, Any], str, tuple[int, ...]]:
    raw, digest, identity = _read_physical_bytes_once(path, label=label)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not strict finite JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return payload, digest, identity


def _hash_physical_once(
    path: Path, *, label: str
) -> tuple[str, tuple[int, ...]]:
    _, digest, identity = _read_physical_bytes_once(path, label=label)
    return digest, identity


def verify_capability_evidence_snapshots(
    snapshots: list[tuple[Path, str, str, tuple[int, ...]]]
) -> None:
    """Require every capability authority to remain the exact opened inode/bytes."""

    for path, label, digest, identity in snapshots:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"{label} disappeared or became nonphysical: {path}")
        observed = os.lstat(path)
        if _physical_identity(observed) != identity:
            raise RuntimeError(f"{label} identity changed after authenticated read: {path}")
        if file_sha256(path) != digest:
            raise RuntimeError(f"{label} bytes changed after authenticated read: {path}")


def _all_literal_true(value: Any, *, exact_keys: set[str]) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == exact_keys
        and all(type(item) is bool and item for item in value.values())
    )


def _literal_bool_mapping(value: Any, *, exact_keys: set[str]) -> bool:
    """Accept an exact gate map without requiring every scientific gate to pass."""

    return (
        isinstance(value, dict)
        and set(value) == exact_keys
        and all(type(item) is bool for item in value.values())
    )


def _completion_bound_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "schema",
            "complete",
            "seed",
            "mode",
            "checkpoint",
            "checkpoint_sha256",
            "metadata",
            "metadata_sha256",
            "training_result",
            "training_result_sha256",
            "precalibration_ema_state",
            "precalibration_ema_state_sha256",
            "source_manifest_sha256",
            "manifest_bundle_sha256",
        )
    }


def _validate_completion_payload(
    payload: dict[str, Any],
    *,
    seed: int,
    paths: Mapping[str, Path],
    hashes: Mapping[str, str],
    manifest_sha256: str,
    manifest_bundle_sha256: str,
) -> None:
    if set(payload) != _DEPLOYMENT_COMPLETION_KEYS:
        raise RuntimeError(f"Seed {seed} completion marker fields differ")
    conditions = {
        "schema": payload.get("schema") == DEPLOYMENT_COMPLETION_SCHEMA,
        "complete": payload.get("complete") is True,
        "seed": type(payload.get("seed")) is int and payload.get("seed") == seed,
        "mode": payload.get("mode") == "full",
        "source_manifest": payload.get("source_manifest_sha256")
        == manifest_sha256,
        "manifest_bundle": payload.get("manifest_bundle_sha256")
        == manifest_bundle_sha256,
        "bundle": payload.get("artifact_bundle_sha256")
        == canonical_sha256(_completion_bound_fields(payload)),
    }
    for key in ("checkpoint", "metadata", "training_result", "precalibration_ema_state"):
        conditions[f"{key}_path"] = payload.get(key) == paths[key].as_posix()
        conditions[f"{key}_sha256"] = payload.get(f"{key}_sha256") == hashes[key]
    failed = sorted(name for name, passed in conditions.items() if not passed)
    if failed:
        raise RuntimeError(f"Seed {seed} completion marker failed gates: {failed}")


def stored_runtime_guard_closed(report: Any) -> bool:
    from scripts.odt_direct_only_compliance import (
        EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT,
        EXPECTED_RUNTIME_GUARD_ENTRYPOINTS,
    )

    return (
        isinstance(report, dict)
        and set(report)
        == {
            "installed",
            "patched_entrypoints",
            "patched_entrypoint_count",
            "allowed_call_count",
            "allowed_calls",
            "prohibited_attempt_count",
            "prohibited_attempts",
        }
        and report.get("installed") is True
        and json_type_exact_equal(
            report.get("patched_entrypoints"),
            sorted(EXPECTED_RUNTIME_GUARD_ENTRYPOINTS),
        )
        and type(report.get("patched_entrypoint_count")) is int
        and report.get("patched_entrypoint_count")
        == EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT
        and type(report.get("allowed_call_count")) is int
        and report.get("allowed_call_count") == 0
        and json_type_exact_equal(report.get("allowed_calls"), [])
        and type(report.get("prohibited_attempt_count")) is int
        and report.get("prohibited_attempt_count") == 0
        and json_type_exact_equal(report.get("prohibited_attempts"), [])
    )


def stored_static_audit_closed(report: Any, sources: dict[str, str]) -> bool:
    return (
        isinstance(report, dict)
        and set(report)
        == {
            "scope",
            "entrypoints",
            "source_sha256",
            "source_count",
            "direct_qr_call_sites",
            "direct_qr_required",
            "guarded_dormant_spectral_norm_sites",
            "duplicate_top_level_definition_sites",
            "prohibited_self_overlap_sites",
            "prohibited_calls_found",
            "call_site_count",
        }
        and report.get("scope") == "transitive_local_import_closure"
        and json_type_exact_equal(report.get("entrypoints"), sorted(sources))
        and json_type_exact_equal(report.get("source_sha256"), sources)
        and type(report.get("source_count")) is int
        and report.get("source_count") == len(sources)
        and type(report.get("direct_qr_call_sites")) is int
        and report.get("direct_qr_call_sites") == 0
        and report.get("direct_qr_required") is False
        and json_type_exact_equal(report.get("prohibited_calls_found"), [])
        and json_type_exact_equal(report.get("prohibited_self_overlap_sites"), [])
        and json_type_exact_equal(report.get("guarded_dormant_spectral_norm_sites"), [])
        and json_type_exact_equal(report.get("duplicate_top_level_definition_sites"), [])
        and type(report.get("call_site_count")) is int
        and report.get("call_site_count") >= 0
    )


def validate_evaluation_result(
    payload: dict[str, Any],
    *,
    path: Path,
    seed: int,
    task_start: int,
    task_end: int,
    manifest: dict[str, Any],
    manifest_sha: str,
    expected_task_mapping: dict[str, Any],
) -> list[dict[str, Any]]:
    protocol = evaluation_protocol(False, task_start, task_end)
    exact_top_level_keys = {
        "schema",
        "recipe_version",
        "mode",
        "seed",
        "checkpoint",
        "checkpoint_sha256",
        "metadata",
        "metadata_sha256",
        "training_result",
        "training_result_sha256",
        "precalibration_ema_state",
        "precalibration_ema_state_sha256",
        "deployment_completion",
        "deployment_completion_sha256",
        "calibration_resume_proof_sha256",
        "training_result_closed",
        "source_manifest_sha256",
        "manifest_bundle_sha256",
        "protocol",
        "task_protocol",
        "task_mapping",
        "episode_count",
        "success_count",
        "overall",
        "per_task",
        "episodes",
        "episode_identity_sha256",
        "decode_boundary_proof",
        "decode_counts",
        "policy_chunk_count",
        "validated_settle_transition_count",
        "validated_policy_transition_count",
        "validated_transition_count",
        "nonfinite_count",
        "decode_failure_count",
        "elapsed_s",
        "environment",
        "transitive_direct_only_static_audit",
        "capability_simulator_boundary",
        "simulator_external_to_odt",
        "odt_runtime_compliance_claimed",
        "canonical_odt_runtime_guard_installed",
        "canonical_odt_numerical_compliance_claimed",
        "external_simulator_outside_weight_only_odt_closure",
        "claim_boundary",
    }
    if set(payload) != exact_top_level_keys:
        raise RuntimeError(f"Evaluation {path} top-level fields differ")
    environment = payload.get("environment")
    environment_keys = {
        "gpu",
        "inference_dtype",
        "matmul_precision",
        "python",
        "torch",
        "cuda",
        "numpy",
        "libero",
        "robosuite",
        "mujoco",
        "mujoco_gl",
    }
    elapsed = payload.get("elapsed_s")
    fixed = {
        "schema": payload.get("schema") == f"{SCHEMA}_evaluation_result",
        "recipe": payload.get("recipe_version") == RECIPE_VERSION,
        "mode": payload.get("mode") == "full",
        "seed": type(payload.get("seed")) is int and payload.get("seed") == seed,
        "manifest_sha": payload.get("source_manifest_sha256") == manifest_sha,
        "manifest_bundle": payload.get("manifest_bundle_sha256")
        == manifest["manifest_bundle_sha256"],
        "protocol": canonical_sha256(payload.get("protocol"))
        == canonical_sha256(protocol),
        "task_mapping": canonical_sha256(payload.get("task_mapping"))
        == canonical_sha256(expected_task_mapping),
        "episode_count": type(payload.get("episode_count")) is int
        and payload.get("episode_count")
        == (task_end - task_start) * EVALUATION_EPISODES_PER_TASK,
        "training_result_closed": payload.get("training_result_closed") is True,
        "full_has_no_resume_proof": payload.get("calibration_resume_proof_sha256")
        is None,
        "elapsed": type(elapsed) is float
        and math.isfinite(float(elapsed))
        and float(elapsed) > 0.0,
        "nonfinite": type(payload.get("nonfinite_count")) is int
        and payload.get("nonfinite_count") == 0,
        "decode_failures": type(payload.get("decode_failure_count")) is int
        and payload.get("decode_failure_count") == 0,
        "public_component_replay": type(
            payload.get("decode_counts", {}).get("public_component_mismatch")
        )
        is int
        and payload.get("decode_counts", {}).get("public_component_mismatch") == 0,
        "decode_proof": payload.get("decode_boundary_proof", {})
        .get("checks", {})
        .get("public_action_equals_fixed_sign_decode")
        is True,
        "static_audit": stored_static_audit_closed(
            payload.get("transitive_direct_only_static_audit"),
            manifest["source_closure"],
        ),
        "simulator_boundary": payload.get("capability_simulator_boundary")
        == CAPABILITY_SIMULATOR_BOUNDARY,
        "odt_guard_not_claimed": payload.get("canonical_odt_runtime_guard_installed")
        is False
        and payload.get("canonical_odt_numerical_compliance_claimed") is False,
        "simulator_external": payload.get(
            "external_simulator_outside_weight_only_odt_closure"
        )
        is True
        and payload.get("simulator_external_to_odt") is True
        and payload.get("odt_runtime_compliance_claimed") is False,
        "gpu": payload.get("environment", {}).get("gpu") == "Quadro RTX 6000",
        "dtype": payload.get("environment", {}).get("inference_dtype") == "float32",
        "matmul": payload.get("environment", {}).get("matmul_precision") == "highest",
        "environment_schema": isinstance(environment, dict)
        and set(environment) == environment_keys
        and json_type_exact_equal(environment, EXPECTED_EVALUATION_ENVIRONMENT),
        "claim_boundary": payload.get("claim_boundary")
        == DEPLOYMENT_CLAIM_BOUNDARY,
    }
    failed = sorted(key for key, value in fixed.items() if not value)
    if failed:
        raise RuntimeError(f"Evaluation {path} failed frozen gates: {failed}")
    observed_task_protocol = payload.get("task_protocol")
    expected_task_protocol = {
        str(task): EXPECTED_TASK_PROTOCOL[task]
        for task in range(task_start, task_end)
    }
    if canonical_sha256(observed_task_protocol) != canonical_sha256(
        expected_task_protocol
    ):
        raise RuntimeError(f"Evaluation {path} task protocol differs")
    episodes = payload.get("episodes")
    expected_count = (task_end - task_start) * EVALUATION_EPISODES_PER_TASK
    if not isinstance(episodes, list) or len(episodes) != expected_count:
        raise RuntimeError(f"Evaluation {path} episode count differs")
    expected_ordered_keys = [
        (seed, task, episode)
        for task in range(task_start, task_end)
        for episode in range(EVALUATION_EPISODES_PER_TASK)
    ]
    observed_keys: list[tuple[int, int, int]] = []
    episode_schema = {
        "seed",
        "task_index",
        "episode_index",
        "initial_state_sha256",
        "settle_steps_completed",
        "settle_all_observations_and_rewards_finite",
        "settle_success_or_done_count",
        "validated_settle_transition_count",
        "validated_policy_transition_count",
        "success",
        "done",
        "steps",
        "policy_chunk_count",
        "elapsed_s",
    }
    for record in episodes:
        if not isinstance(record, dict) or set(record) != episode_schema:
            raise RuntimeError(f"Evaluation {path} has a malformed episode")
        if type(record.get("success")) is not bool or type(record.get("done")) is not bool:
            raise RuntimeError(f"Evaluation {path} has non-boolean outcome fields")
        key = (
            record.get("seed"),
            record.get("task_index"),
            record.get("episode_index"),
        )
        if (
            type(record.get("seed")) is not int
            or type(record.get("task_index")) is not int
            or type(record.get("episode_index")) is not int
        ):
            raise RuntimeError(f"Evaluation {path} has non-integer episode identity")
        observed_keys.append(key)
        steps = record.get("steps")
        chunks = record.get("policy_chunk_count")
        if type(steps) is not int or not 1 <= steps <= protocol["max_steps"]:
            raise RuntimeError(f"Evaluation {path} has invalid step count")
        if (
            type(chunks) is not int
            or chunks < 1
            or steps < (chunks - 1) * protocol["execution_horizon"] + 1
            or steps > chunks * protocol["execution_horizon"]
        ):
            raise RuntimeError(f"Evaluation {path} has invalid policy chunk count")
        if steps < protocol["max_steps"] and not (
            record["success"] or record["done"]
        ):
            raise RuntimeError(f"Evaluation {path} violates early-stop semantics")
        if (
            type(record.get("settle_steps_completed")) is not int
            or type(record.get("settle_success_or_done_count")) is not int
            or type(record.get("validated_settle_transition_count")) is not int
            or type(record.get("validated_policy_transition_count")) is not int
        ):
            raise RuntimeError(f"Evaluation {path} has non-integer transition counts")
        if (
            record.get("settle_steps_completed") != protocol["settle_steps"]
            or record.get("settle_all_observations_and_rewards_finite") is not True
            or record.get("settle_success_or_done_count") != 0
            or record.get("validated_settle_transition_count")
            != protocol["settle_steps"]
            or record.get("validated_policy_transition_count") != steps
        ):
            raise RuntimeError(f"Evaluation {path} failed the settle sentinel")
        episode_elapsed = record.get("elapsed_s")
        if (
            type(episode_elapsed) is not float
            or not math.isfinite(float(episode_elapsed))
            or float(episode_elapsed) <= 0.0
        ):
            raise RuntimeError(f"Evaluation {path} has invalid episode elapsed time")
        initial_hash = record.get("initial_state_sha256")
        if (
            not isinstance(initial_hash, str)
            or len(initial_hash) != 64
            or any(character not in "0123456789abcdef" for character in initial_hash)
        ):
            raise RuntimeError(f"Evaluation {path} has invalid initial-state identity")
    if observed_keys != expected_ordered_keys:
        raise RuntimeError(f"Evaluation {path} episode order differs")
    observed_identities = [
        (
            record["seed"],
            record["task_index"],
            record["episode_index"],
            record["initial_state_sha256"],
        )
        for record in episodes
    ]
    if payload.get("episode_identity_sha256") != canonical_sha256(
        observed_identities
    ):
        raise RuntimeError(f"Evaluation {path} episode identity digest differs")
    for task in range(task_start, task_end):
        row_hashes = [
            record["initial_state_sha256"]
            for record in episodes
            if record["task_index"] == task
        ]
        if canonical_sha256(row_hashes) != (
            OFFICIAL_INIT_STATE_ROW_HASH_LIST_SHA256[task]
        ):
            raise RuntimeError(
                f"Evaluation {path} initial-state row authority differs for task {task}"
            )
    total_chunks = sum(record["policy_chunk_count"] for record in episodes)
    total_settle_transitions = sum(
        record["validated_settle_transition_count"] for record in episodes
    )
    total_policy_transitions = sum(
        record["validated_policy_transition_count"] for record in episodes
    )
    decode_counts = payload.get("decode_counts")
    total_signs = total_chunks * N_FACTORS
    if (
        type(payload.get("policy_chunk_count")) is not int
        or type(payload.get("validated_settle_transition_count")) is not int
        or type(payload.get("validated_policy_transition_count")) is not int
        or type(payload.get("validated_transition_count")) is not int
        or payload.get("policy_chunk_count") != total_chunks
        or payload.get("validated_settle_transition_count")
        != total_settle_transitions
        or payload.get("validated_policy_transition_count")
        != total_policy_transitions
        or payload.get("validated_transition_count")
        != total_settle_transitions + total_policy_transitions
        or total_settle_transitions != len(episodes) * protocol["settle_steps"]
        or total_policy_transitions != sum(record["steps"] for record in episodes)
        or not isinstance(decode_counts, dict)
        or set(decode_counts)
        != {
            "zero_gate_logits",
            "positive_signs",
            "total_signs",
            "public_component_mismatch",
        }
        or type(decode_counts.get("zero_gate_logits")) is not int
        or not 0 <= decode_counts["zero_gate_logits"] <= total_signs
        or type(decode_counts.get("positive_signs")) is not int
        or not 0 <= decode_counts["positive_signs"] <= total_signs
        or type(decode_counts.get("total_signs")) is not int
        or decode_counts.get("total_signs") != total_signs
        or type(decode_counts.get("public_component_mismatch")) is not int
        or decode_counts.get("public_component_mismatch") != 0
    ):
        raise RuntimeError(f"Evaluation {path} decode-count arithmetic differs")
    decode_proof = payload.get("decode_boundary_proof")
    expected_decode_checks = {
        "public_action_equals_fixed_sign_decode": True,
        "finite_center": True,
        "finite_factors": True,
        "finite_gates": True,
        "finite_actions": True,
    }
    if (
        not isinstance(decode_proof, dict)
        or set(decode_proof)
        != {
            "checks",
            "zero_gate_logit_count",
            "positive_sign_count",
            "total_sign_count",
            "positive_sign_fraction",
            "fixed_sign_rule",
            "component_shapes",
        }
        or not json_type_exact_equal(
            decode_proof.get("checks"), expected_decode_checks
        )
        or type(decode_proof.get("total_sign_count")) is not int
        or decode_proof.get("total_sign_count") != N_FACTORS
        or type(decode_proof.get("zero_gate_logit_count")) is not int
        or not 0 <= decode_proof["zero_gate_logit_count"] <= N_FACTORS
        or type(decode_proof.get("positive_sign_count")) is not int
        or not 0 <= decode_proof["positive_sign_count"] <= N_FACTORS
        or type(decode_proof.get("positive_sign_fraction")) is not float
        or not math.isfinite(float(decode_proof["positive_sign_fraction"]))
        or float(decode_proof["positive_sign_fraction"])
        != decode_proof["positive_sign_count"] / N_FACTORS
        or decode_proof.get("fixed_sign_rule")
        != "+1 iff gate_logit > 0, else -1"
        or not json_type_exact_equal(
            decode_proof.get("component_shapes"),
            {
                "center": [1, ACTION_HORIZON * ACTION_DIM],
                "factors": [1, N_FACTORS, ACTION_HORIZON * ACTION_DIM],
                "gates": [1, N_FACTORS],
            },
        )
    ):
        raise RuntimeError(f"Evaluation {path} decode boundary schema differs")
    successes = sum(int(record["success"]) for record in episodes)
    if (
        type(payload.get("episode_count")) is not int
        or type(payload.get("success_count")) is not int
        or payload.get("success_count") != successes
        or type(payload.get("overall")) is not float
        or not math.isfinite(float(payload["overall"]))
        or float(payload["overall"]) != successes / len(episodes)
    ):
        raise RuntimeError(f"Evaluation {path} success summary differs")
    expected_per_task = {
        str(task): sum(
            int(record["success"])
            for record in episodes
            if record["task_index"] == task
        )
        / EVALUATION_EPISODES_PER_TASK
        for task in range(task_start, task_end)
    }
    observed_per_task = payload.get("per_task")
    if (
        not isinstance(observed_per_task, dict)
        or set(observed_per_task) != set(expected_per_task)
        or any(
            type(value) is not float
            or not math.isfinite(float(value))
            or float(value) != expected_per_task[key]
            for key, value in observed_per_task.items()
        )
    ):
        raise RuntimeError(f"Evaluation {path} per-task summary differs")
    return episodes


def validate_capability_aggregate_evidence(
    run_root: Path,
    *,
    manifest: dict[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    """Reconstruct the aggregate and gate from physical raw evaluation evidence.

    This is intentionally usable by the later exact-ODT composite.  It does not
    trust producer gate booleans.  It opens each source artifact once, binds the
    digest to those exact bytes, recomputes all 1,500 episode summaries, and
    returns inode snapshots that the caller must verify immediately before
    publishing its own attestation.
    """

    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != f"{SCHEMA}_source_manifest"
        or not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
    ):
        raise RuntimeError("Capability manifest authority is malformed")
    snapshots: list[tuple[Path, str, str, tuple[int, ...]]] = []

    def read_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
        payload, digest, identity = _read_json_physical_once(path, label=label)
        snapshots.append((path, label, digest, identity))
        return payload, digest

    def hash_binary(path: Path, label: str) -> str:
        digest, identity = _hash_physical_once(path, label=label)
        snapshots.append((path, label, digest, identity))
        return digest

    physical_manifest_path = source_manifest_path(run_root)
    physical_manifest, physical_manifest_sha = read_json(
        physical_manifest_path, "Capability source manifest"
    )
    if physical_manifest_sha != manifest_sha256 or not json_type_exact_equal(
        physical_manifest, manifest
    ):
        raise RuntimeError("Physical capability manifest differs from copied authority")

    expected_evaluations = {
        evaluation_result_path(run_root, seed, start, end, False).resolve()
        for seed in SEEDS
        for start, end in SHARDS
    }
    observed_evaluations = {
        path.resolve()
        for path in (run_root / "results").glob("eval_product_pade_rational_s*_t*.json")
        if not path.name.endswith("_smoke.json")
    }
    if observed_evaluations != expected_evaluations:
        raise RuntimeError("Raw capability evaluation inventory differs from the fixed 12 shards")

    expected_mapping = expected_task_mapping_record()
    evaluation_hashes: dict[str, str] = {}
    artifact_hashes: dict[str, dict[str, str]] = {}
    all_episodes: dict[int, list[dict[str, Any]]] = {seed: [] for seed in SEEDS}
    completed_shards: dict[int, set[tuple[int, int]]] = {}
    initial_panels: dict[tuple[int, int], tuple[str, ...]] = {}
    integrity_rows: list[dict[str, int]] = []

    for seed in SEEDS:
        artifact_paths = {
            "checkpoint": checkpoint_path(run_root, seed, False),
            "metadata": metadata_path(run_root, seed, False),
            "training_result": training_result_path(run_root, seed, False),
            "precalibration_ema_state": precalibration_state_path(run_root, seed, False),
            "deployment_completion": deployment_completion_path(run_root, seed, False),
        }
        metadata_payload, metadata_sha = read_json(
            artifact_paths["metadata"], f"Seed {seed} metadata"
        )
        training_payload, training_sha = read_json(
            artifact_paths["training_result"], f"Seed {seed} training result"
        )
        completion_payload, completion_sha = read_json(
            artifact_paths["deployment_completion"], f"Seed {seed} completion marker"
        )
        checkpoint_sha = hash_binary(
            artifact_paths["checkpoint"], f"Seed {seed} checkpoint"
        )
        precalibration_sha = hash_binary(
            artifact_paths["precalibration_ema_state"],
            f"Seed {seed} pre-calibration state",
        )
        seed_hashes = {
            "checkpoint": checkpoint_sha,
            "metadata": metadata_sha,
            "training_result": training_sha,
            "precalibration_ema_state": precalibration_sha,
            "deployment_completion": completion_sha,
        }
        artifact_hashes[str(seed)] = seed_hashes
        _validate_completion_payload(
            completion_payload,
            seed=seed,
            paths=artifact_paths,
            hashes=seed_hashes,
            manifest_sha256=manifest_sha256,
            manifest_bundle_sha256=manifest["manifest_bundle_sha256"],
        )
        training_checks = {
            "training_schema": training_payload.get("schema")
            == f"{SCHEMA}_training_result",
            "metadata_schema": metadata_payload.get("schema") == SCHEMA,
            "identity": type(training_payload.get("seed")) is int
            and type(metadata_payload.get("seed")) is int
            and training_payload.get("seed") == metadata_payload.get("seed") == seed
            and training_payload.get("mode") == metadata_payload.get("mode") == "full",
            "checkpoint": training_payload.get("checkpoint")
            == artifact_paths["checkpoint"].as_posix()
            and training_payload.get("checkpoint_sha256") == checkpoint_sha
            and metadata_payload.get("checkpoint")
            == artifact_paths["checkpoint"].as_posix()
            and metadata_payload.get("checkpoint_sha256") == checkpoint_sha,
            "metadata": training_payload.get("metadata")
            == artifact_paths["metadata"].as_posix()
            and training_payload.get("metadata_sha256") == metadata_sha,
            "precalibration": training_payload.get(
                "training_only_precalibration_ema_state", {}
            ).get("sha256")
            == metadata_payload.get("training_only_precalibration_ema_state", {}).get(
                "sha256"
            )
            == precalibration_sha,
            "source": training_payload.get("source_manifest_sha256")
            == manifest_sha256
            and training_payload.get("manifest_bundle_sha256")
            == manifest["manifest_bundle_sha256"]
            and metadata_payload.get("source_manifest", {}).get("sha256")
            == manifest_sha256
            and metadata_payload.get("source_manifest", {}).get(
                "manifest_bundle_sha256"
            )
            == manifest["manifest_bundle_sha256"],
            "source_snapshots": json_type_exact_equal(
                metadata_payload.get("source_snapshot_start"), manifest["source_closure"]
            )
            and json_type_exact_equal(
                metadata_payload.get("source_snapshot_end"), manifest["source_closure"]
            ),
            "training_static": stored_static_audit_closed(
                training_payload.get("transitive_direct_only_static_audit"),
                manifest["source_closure"],
            ),
            "metadata_static_start": stored_static_audit_closed(
                metadata_payload.get("transitive_direct_only_static_audit"),
                manifest["source_closure"],
            ),
            "metadata_static_end": stored_static_audit_closed(
                metadata_payload.get("end_transitive_direct_only_static_audit"),
                manifest["source_closure"],
            ),
            "runtime_import": stored_runtime_guard_closed(
                training_payload.get("runtime_direct_only_guard_at_import")
            ),
            "runtime_final": stored_runtime_guard_closed(
                training_payload.get("runtime_direct_only_guard_final")
            ),
            "claim_boundary": training_payload.get("claim_boundary")
            == metadata_payload.get("claim_boundary")
            == DEPLOYMENT_CLAIM_BOUNDARY,
        }
        failed_training = sorted(
            name for name, passed in training_checks.items() if not passed
        )
        if failed_training:
            raise RuntimeError(
                f"Seed {seed} raw training chain failed: {failed_training}"
            )

        seen_shards: set[tuple[int, int]] = set()
        for task_start, task_end in SHARDS:
            result_path = evaluation_result_path(
                run_root, seed, task_start, task_end, False
            )
            evaluation, evaluation_sha = read_json(
                result_path,
                f"Seed {seed} evaluation shard {task_start}:{task_end}",
            )
            evaluation_hashes[result_path.as_posix()] = evaluation_sha
            if (task_start, task_end) in seen_shards:
                raise RuntimeError(f"Seed {seed} repeats an evaluation shard")
            seen_shards.add((task_start, task_end))
            expected_paths = {
                "checkpoint": artifact_paths["checkpoint"].as_posix(),
                "metadata": artifact_paths["metadata"].as_posix(),
                "training_result": artifact_paths["training_result"].as_posix(),
                "precalibration_ema_state": artifact_paths[
                    "precalibration_ema_state"
                ].as_posix(),
                "deployment_completion": artifact_paths[
                    "deployment_completion"
                ].as_posix(),
            }
            if any(evaluation.get(key) != value for key, value in expected_paths.items()):
                raise RuntimeError(f"Seed {seed} evaluation artifact paths differ")
            if any(
                evaluation.get(f"{key}_sha256") != seed_hashes[key]
                for key in _TRAINING_ARTIFACT_KEYS
            ):
                raise RuntimeError(f"Seed {seed} evaluation artifact hashes differ")
            episodes = validate_evaluation_result(
                evaluation,
                path=result_path,
                seed=seed,
                task_start=task_start,
                task_end=task_end,
                manifest=manifest,
                manifest_sha=manifest_sha256,
                expected_task_mapping=expected_mapping,
            )
            integrity_rows.append(
                {
                    "nonfinite_count": evaluation["nonfinite_count"],
                    "decode_failure_count": evaluation["decode_failure_count"],
                    "public_component_mismatch": evaluation["decode_counts"][
                        "public_component_mismatch"
                    ],
                    "settle_success_or_done_count": sum(
                        record["settle_success_or_done_count"] for record in episodes
                    ),
                    "validated_settle_transition_count": evaluation[
                        "validated_settle_transition_count"
                    ],
                    "validated_policy_transition_count": evaluation[
                        "validated_policy_transition_count"
                    ],
                    "episode_step_count": sum(record["steps"] for record in episodes),
                }
            )
            all_episodes[seed].extend(episodes)
        if seen_shards != set(SHARDS):
            raise RuntimeError(f"Seed {seed} did not provide every frozen shard")
        completed_shards[seed] = seen_shards
        for task in range(10):
            records = sorted(
                (row for row in all_episodes[seed] if row["task_index"] == task),
                key=lambda row: row["episode_index"],
            )
            panel = tuple(row["initial_state_sha256"] for row in records)
            if (
                len(records) != EVALUATION_EPISODES_PER_TASK
                or len(set(panel)) != EVALUATION_EPISODES_PER_TASK
                or canonical_sha256(list(panel))
                != OFFICIAL_INIT_STATE_ROW_HASH_LIST_SHA256[task]
            ):
                raise RuntimeError(f"Seed {seed} task {task} initial-state panel differs")
            initial_panels[(seed, task)] = panel

    for task in range(10):
        reference = initial_panels[(SEEDS[0], task)]
        if any(initial_panels[(seed, task)] != reference for seed in SEEDS[1:]):
            raise RuntimeError(f"Task {task} initial-state panel differs across seeds")

    seed_summaries: dict[str, dict[str, Any]] = {}
    overalls: list[float] = []
    for seed in SEEDS:
        episodes = all_episodes[seed]
        if len(episodes) != 10 * EVALUATION_EPISODES_PER_TASK:
            raise RuntimeError(f"Seed {seed} aggregate episode count differs")
        identities = [
            (row["seed"], row["task_index"], row["episode_index"])
            for row in episodes
        ]
        if len(set(identities)) != len(identities):
            raise RuntimeError(f"Seed {seed} has duplicate aggregate episode identities")
        successes = sum(int(row["success"]) for row in episodes)
        overall = successes / len(episodes)
        overalls.append(overall)
        seed_summaries[str(seed)] = {
            "success_count": successes,
            "episode_count": len(episodes),
            "overall": overall,
            "per_task": {
                str(task): sum(
                    int(row["success"])
                    for row in episodes
                    if row["task_index"] == task
                )
                / EVALUATION_EPISODES_PER_TASK
                for task in range(10)
            },
        }
    seed0_overall = seed_summaries[str(PRIMARY_CAPABILITY_SEED)]["overall"]
    canonical_panel_protocol = (
        set(initial_panels)
        == {(seed, task) for seed in SEEDS for task in range(10)}
        and all(
            len(panel) == EVALUATION_EPISODES_PER_TASK
            and len(set(panel)) == EVALUATION_EPISODES_PER_TASK
            and canonical_sha256(list(panel))
            == OFFICIAL_INIT_STATE_ROW_HASH_LIST_SHA256[task]
            for (seed, task), panel in initial_panels.items()
        )
        and all(
            initial_panels[(seed, task)] == initial_panels[(SEEDS[0], task)]
            for seed in SEEDS[1:]
            for task in range(10)
        )
    )
    integrity_closed = (
        len(integrity_rows) == len(SEEDS) * len(SHARDS)
        and all(
            row["nonfinite_count"] == 0
            and row["decode_failure_count"] == 0
            and row["public_component_mismatch"] == 0
            and row["settle_success_or_done_count"] == 0
            and row["validated_settle_transition_count"] > 0
            and row["validated_policy_transition_count"]
            == row["episode_step_count"]
            for row in integrity_rows
        )
    )
    transition_counts = {
        "settle": sum(row["validated_settle_transition_count"] for row in integrity_rows),
        "policy": sum(row["validated_policy_transition_count"] for row in integrity_rows),
        "total": sum(
            row["validated_settle_transition_count"]
            + row["validated_policy_transition_count"]
            for row in integrity_rows
        ),
        "expected_settle": len(SEEDS)
        * 10
        * EVALUATION_EPISODES_PER_TASK
        * evaluation_protocol(False, *SHARDS[0])["settle_steps"],
        "episode_step_total": sum(row["episode_step_count"] for row in integrity_rows),
    }
    transitions_closed = (
        transition_counts["settle"] == transition_counts["expected_settle"]
        and transition_counts["policy"] == transition_counts["episode_step_total"]
        and transition_counts["total"]
        == transition_counts["settle"] + transition_counts["policy"]
    )
    conditions = {
        "primary_seed_is_preregistered_seed0": PRIMARY_CAPABILITY_SEED == 0,
        "primary_overall_at_least_floor": seed0_overall >= PRIMARY_CAPABILITY_FLOOR,
        "all_three_seed_reports_present": set(seed_summaries) == {"0", "1", "2"},
        "all_four_shards_per_seed_present_and_unique": completed_shards
        == {seed: set(SHARDS) for seed in SEEDS},
        "exactly_500_episodes_per_seed": all(
            summary["episode_count"] == 500 for summary in seed_summaries.values()
        ),
        "canonical_task_and_initial_state_protocol": canonical_panel_protocol,
        "no_nonfinite_or_decode_failures": integrity_closed,
        "all_simulator_transitions_strictly_validated": transitions_closed,
        "no_best_seed_post_selection": True,
    }
    if set(conditions) != _CAPABILITY_CONDITION_KEYS or not all(
        type(value) is bool for value in conditions.values()
    ):
        raise RuntimeError("Reconstructed capability condition schema differs")
    capability_pass = all(conditions.values())

    aggregate_path = aggregate_result_path(run_root)
    gate_path = seed0_capability_gate_path(run_root)
    aggregate, aggregate_sha = read_json(aggregate_path, "Capability aggregate")
    gate, gate_sha = read_json(gate_path, "Capability seed0 gate")
    if set(aggregate) != _AGGREGATE_TOP_LEVEL_KEYS:
        raise RuntimeError("Capability aggregate top-level fields differ")
    if set(gate) != _CAPABILITY_GATE_TOP_LEVEL_KEYS:
        raise RuntimeError("Capability gate top-level fields differ")
    if not stored_static_audit_closed(
        aggregate.get("transitive_direct_only_static_audit"), manifest["source_closure"]
    ):
        raise RuntimeError("Capability aggregate static audit differs")
    if not stored_runtime_guard_closed(aggregate.get("runtime_direct_only_guard_at_import")):
        raise RuntimeError("Capability aggregate import-time runtime guard differs")
    if not stored_runtime_guard_closed(aggregate.get("runtime_direct_only_guard_final")):
        raise RuntimeError("Capability aggregate final runtime guard differs")
    expected_aggregate = {
        "schema": f"{SCHEMA}_three_seed_aggregate",
        "recipe_version": RECIPE_VERSION,
        "source_manifest_sha256": manifest_sha256,
        "manifest_bundle_sha256": manifest["manifest_bundle_sha256"],
        "evaluation_result_sha256": evaluation_hashes,
        "training_artifact_sha256": artifact_hashes,
        "seeds": list(SEEDS),
        "preregistered_primary_seed": PRIMARY_CAPABILITY_SEED,
        "preregistered_primary_floor": PRIMARY_CAPABILITY_FLOOR,
        "seed_summaries": seed_summaries,
        "three_seed_mean": statistics.fmean(overalls),
        "three_seed_population_std": statistics.pstdev(overalls),
        "capability_gate_conditions": conditions,
        "capability_gate_pass": capability_pass,
        "validated_transition_counts": transition_counts,
        "task_protocol": {str(task): EXPECTED_TASK_PROTOCOL[task] for task in range(10)},
        "transitive_direct_only_static_audit": aggregate[
            "transitive_direct_only_static_audit"
        ],
        "runtime_direct_only_guard_at_import": aggregate[
            "runtime_direct_only_guard_at_import"
        ],
        "runtime_direct_only_guard_final": aggregate["runtime_direct_only_guard_final"],
        "claim_boundary": DEPLOYMENT_CLAIM_BOUNDARY,
    }
    if canonical_sha256(aggregate) != canonical_sha256(expected_aggregate):
        changed = sorted(
            key for key in _AGGREGATE_TOP_LEVEL_KEYS if aggregate.get(key) != expected_aggregate.get(key)
        )
        raise RuntimeError(f"Capability aggregate differs from raw reconstruction: {changed}")
    if not _literal_bool_mapping(
        aggregate["capability_gate_conditions"],
        exact_keys=_CAPABILITY_CONDITION_KEYS,
    ):
        raise RuntimeError("Capability aggregate conditions are not exact literal booleans")
    if type(aggregate["capability_gate_pass"]) is not bool:
        raise RuntimeError("Capability aggregate pass flag is not a literal boolean")

    seed0_artifacts = artifact_hashes[str(PRIMARY_CAPABILITY_SEED)]
    expected_gate = {
        "schema": f"{SCHEMA}_seed0_capability_gate",
        "recipe_version": RECIPE_VERSION,
        "preregistered_before_evaluation": True,
        "primary_seed": PRIMARY_CAPABILITY_SEED,
        "required_overall": PRIMARY_CAPABILITY_FLOOR,
        "observed_overall": seed0_overall,
        "conditions": conditions,
        "approved_for_capable_global_claim": capability_pass,
        "capability_condition_satisfied_for_future_composite_attestation": capability_pass,
        "exact_odt_composite_attestation_approved": False,
        "checkpoint": checkpoint_path(run_root, PRIMARY_CAPABILITY_SEED, False).as_posix(),
        "checkpoint_sha256": seed0_artifacts["checkpoint"],
        "metadata_sha256": seed0_artifacts["metadata"],
        "training_result_sha256": seed0_artifacts["training_result"],
        "precalibration_ema_state_sha256": seed0_artifacts["precalibration_ema_state"],
        "deployment_completion_sha256": seed0_artifacts["deployment_completion"],
        "aggregate": aggregate_path.as_posix(),
        "aggregate_sha256": aggregate_sha,
        "source_manifest_sha256": manifest_sha256,
        "manifest_bundle_sha256": manifest["manifest_bundle_sha256"],
        "does_not_select_best_seed_post_hoc": True,
        "exact_odt_result_required_separately": True,
        "claim_boundary": DEPLOYMENT_CLAIM_BOUNDARY,
    }
    if canonical_sha256(gate) != canonical_sha256(expected_gate):
        changed = sorted(
            key for key in _CAPABILITY_GATE_TOP_LEVEL_KEYS if gate.get(key) != expected_gate.get(key)
        )
        raise RuntimeError(f"Capability gate differs from raw reconstruction: {changed}")
    if not _literal_bool_mapping(gate["conditions"], exact_keys=_CAPABILITY_CONDITION_KEYS):
        raise RuntimeError("Capability gate conditions are not exact literal booleans")
    for key in (
        "preregistered_before_evaluation",
        "approved_for_capable_global_claim",
        "capability_condition_satisfied_for_future_composite_attestation",
        "exact_odt_composite_attestation_approved",
        "does_not_select_best_seed_post_hoc",
        "exact_odt_result_required_separately",
    ):
        if type(gate[key]) is not bool:
            raise RuntimeError(f"Capability gate field {key} is not a literal boolean")
    return {
        "validated": True,
        "aggregate": aggregate,
        "aggregate_sha256": aggregate_sha,
        "gate": gate,
        "gate_sha256": gate_sha,
        "seed_summaries": seed_summaries,
        "conditions": conditions,
        "evaluation_result_sha256": evaluation_hashes,
        "training_artifact_sha256": artifact_hashes,
        "snapshots": snapshots,
    }


def main() -> None:
    args = parse_args()
    if not _RUNNING_AS_ENTRYPOINT or _PREIMPORT_SOURCE_VERIFICATION is None:
        raise RuntimeError("Product aggregation must use its authenticated entrypoint")
    aggregate_path = aggregate_result_path(args.run_root)
    gate_path = seed0_capability_gate_path(args.run_root)
    for output in (aggregate_path, gate_path):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Refusing to overwrite aggregate output {output}")
    manifest_start = load_and_verify_source_manifest(args.source_manifest)
    manifest_sha = file_sha256(args.source_manifest)
    if manifest_start["source_closure"] != _PREIMPORT_SOURCE_VERIFICATION["source_sha256"]:
        raise RuntimeError("Pre-import and post-import source verification differ")
    if source_snapshot() != _TRANSITIVE_STATIC_AUDIT["source_sha256"]:
        raise RuntimeError("Live Python closure differs from the authenticated static audit")
    _, expected_task_mapping = validate_task_mapping(OFFICIAL_TASK_LANGUAGES)

    expected_full_evaluations = {
        evaluation_result_path(args.run_root, seed, start, end, False).resolve()
        for seed in SEEDS
        for start, end in SHARDS
    }
    observed_full_evaluations = {
        path.resolve()
        for path in (args.run_root / "results").glob(
            "eval_product_pade_rational_s*_t*.json"
        )
        if not path.name.endswith("_smoke.json")
    }
    if observed_full_evaluations != expected_full_evaluations:
        raise RuntimeError(
            "Full evaluation artifact inventory differs from the preregistered 12 paths: "
            f"missing={sorted(str(path) for path in expected_full_evaluations - observed_full_evaluations)}, "
            f"extra={sorted(str(path) for path in observed_full_evaluations - expected_full_evaluations)}"
        )

    evaluation_hashes: dict[str, str] = {}
    artifact_hashes: dict[str, dict[str, str]] = {}
    all_episodes: dict[int, list[dict[str, Any]]] = {seed: [] for seed in SEEDS}
    initial_panels: dict[tuple[int, int], tuple[str, ...]] = {}
    completed_shards: dict[int, set[tuple[int, int]]] = {}
    evaluation_integrity_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        checkpoint = checkpoint_path(args.run_root, seed, False)
        metadata = metadata_path(args.run_root, seed, False)
        training_result = training_result_path(args.run_root, seed, False)
        precalibration = precalibration_state_path(args.run_root, seed, False)
        completion = deployment_completion_path(args.run_root, seed, False)
        artifacts = {
            "checkpoint": checkpoint,
            "metadata": metadata,
            "training_result": training_result,
            "precalibration_ema_state": precalibration,
            "deployment_completion": completion,
        }
        for label, artifact in artifacts.items():
            if not artifact.is_file() or artifact.is_symlink():
                raise RuntimeError(f"Seed {seed} {label} is missing or nonphysical")
        artifact_hashes[str(seed)] = {
            label: file_sha256(artifact) for label, artifact in artifacts.items()
        }
        training_payload = read_json_physical(training_result)
        metadata_payload = read_json_physical(metadata)
        load_and_validate_deployment_completion(
            completion,
            expected_seed=seed,
            expected_mode="full",
            expected_checkpoint=checkpoint,
            expected_metadata=metadata,
            expected_training_result=training_result,
            expected_precalibration_ema_state=precalibration,
            expected_source_manifest_sha256=manifest_sha,
            expected_manifest_bundle_sha256=manifest_start[
                "manifest_bundle_sha256"
            ],
        )
        if (
            training_payload.get("schema") != f"{SCHEMA}_training_result"
            or type(training_payload.get("seed")) is not int
            or training_payload.get("seed") != seed
            or training_payload.get("mode") != "full"
            or training_payload.get("checkpoint_sha256")
            != artifact_hashes[str(seed)]["checkpoint"]
            or training_payload.get("metadata_sha256")
            != artifact_hashes[str(seed)]["metadata"]
            or metadata_payload.get("checkpoint_sha256")
            != artifact_hashes[str(seed)]["checkpoint"]
            or training_payload.get("training_only_precalibration_ema_state", {}).get(
                "sha256"
            )
            != artifact_hashes[str(seed)]["precalibration_ema_state"]
            or not json_type_exact_equal(
                metadata_payload.get("training_only_precalibration_ema_state"),
                training_payload.get("training_only_precalibration_ema_state"),
            )
            or not stored_static_audit_closed(
                training_payload.get("transitive_direct_only_static_audit"),
                manifest_start["source_closure"],
            )
            or not stored_runtime_guard_closed(
                training_payload.get("runtime_direct_only_guard_at_import")
            )
            or not stored_runtime_guard_closed(
                training_payload.get("runtime_direct_only_guard_final")
            )
        ):
            raise RuntimeError(f"Seed {seed} training artifact chain did not close")
        seen_shards: set[tuple[int, int]] = set()
        for task_start, task_end in SHARDS:
            result_path = evaluation_result_path(
                args.run_root, seed, task_start, task_end, False
            )
            payload = read_json_physical(result_path)
            relative = result_path.as_posix()
            evaluation_hashes[relative] = file_sha256(result_path)
            if (task_start, task_end) in seen_shards:
                raise RuntimeError(f"Seed {seed} repeats an evaluation shard")
            seen_shards.add((task_start, task_end))
            if (
                payload.get("checkpoint_sha256")
                != artifact_hashes[str(seed)]["checkpoint"]
                or payload.get("metadata_sha256")
                != artifact_hashes[str(seed)]["metadata"]
                or payload.get("training_result_sha256")
                != artifact_hashes[str(seed)]["training_result"]
                or payload.get("precalibration_ema_state_sha256")
                != artifact_hashes[str(seed)]["precalibration_ema_state"]
                or payload.get("deployment_completion_sha256")
                != artifact_hashes[str(seed)]["deployment_completion"]
            ):
                raise RuntimeError(f"Seed {seed} evaluation artifact chain differs")
            episodes = validate_evaluation_result(
                payload,
                path=result_path,
                seed=seed,
                task_start=task_start,
                task_end=task_end,
                manifest=manifest_start,
                manifest_sha=manifest_sha,
                expected_task_mapping=expected_task_mapping,
            )
            evaluation_integrity_rows.append(
                {
                    "nonfinite_count": payload["nonfinite_count"],
                    "decode_failure_count": payload["decode_failure_count"],
                    "public_component_mismatch": payload["decode_counts"][
                        "public_component_mismatch"
                    ],
                    "settle_success_or_done_count": sum(
                        record["settle_success_or_done_count"] for record in episodes
                    ),
                    "validated_settle_transition_count": payload[
                        "validated_settle_transition_count"
                    ],
                    "validated_policy_transition_count": payload[
                        "validated_policy_transition_count"
                    ],
                    "episode_step_count": sum(record["steps"] for record in episodes),
                }
            )
            all_episodes[seed].extend(episodes)
        if seen_shards != set(SHARDS):
            raise RuntimeError(f"Seed {seed} did not provide every frozen shard")
        completed_shards[seed] = seen_shards
        for task in range(10):
            task_records = sorted(
                (record for record in all_episodes[seed] if record["task_index"] == task),
                key=lambda record: record["episode_index"],
            )
            if len(task_records) != EVALUATION_EPISODES_PER_TASK:
                raise RuntimeError(f"Seed {seed} task {task} episode count differs")
            panel = tuple(record["initial_state_sha256"] for record in task_records)
            if len(set(panel)) != EVALUATION_EPISODES_PER_TASK:
                raise RuntimeError(f"Seed {seed} task {task} repeats an initial state")
            if canonical_sha256(list(panel)) != (
                OFFICIAL_INIT_STATE_ROW_HASH_LIST_SHA256[task]
            ):
                raise RuntimeError(
                    f"Seed {seed} task {task} initial-state row authority differs"
                )
            initial_panels[(seed, task)] = panel
    for task in range(10):
        reference = initial_panels[(SEEDS[0], task)]
        if any(initial_panels[(seed, task)] != reference for seed in SEEDS[1:]):
            raise RuntimeError(f"Task {task} packaged panel differs across seeds")

    seed_summaries: dict[str, dict[str, Any]] = {}
    overalls: list[float] = []
    for seed in SEEDS:
        episodes = all_episodes[seed]
        if len(episodes) != 10 * EVALUATION_EPISODES_PER_TASK:
            raise RuntimeError(f"Seed {seed} full evaluation count differs")
        keys = {
            (record["seed"], record["task_index"], record["episode_index"])
            for record in episodes
        }
        if len(keys) != len(episodes):
            raise RuntimeError(f"Seed {seed} has duplicate aggregate episode keys")
        successes = sum(int(record["success"]) for record in episodes)
        overall = successes / len(episodes)
        overalls.append(overall)
        seed_summaries[str(seed)] = {
            "success_count": successes,
            "episode_count": len(episodes),
            "overall": overall,
            "per_task": {
                str(task): sum(
                    int(record["success"])
                    for record in episodes
                    if record["task_index"] == task
                )
                / EVALUATION_EPISODES_PER_TASK
                for task in range(10)
            },
        }
    seed0_overall = seed_summaries[str(PRIMARY_CAPABILITY_SEED)]["overall"]
    canonical_panel_protocol = (
        set(initial_panels)
        == {(seed, task) for seed in SEEDS for task in range(10)}
        and all(
            len(panel) == EVALUATION_EPISODES_PER_TASK
            and len(set(panel)) == EVALUATION_EPISODES_PER_TASK
            and canonical_sha256(list(panel))
            == OFFICIAL_INIT_STATE_ROW_HASH_LIST_SHA256[task]
            for (seed, task), panel in initial_panels.items()
        )
        and all(
            initial_panels[(seed, task)] == initial_panels[(SEEDS[0], task)]
            for seed in SEEDS[1:]
            for task in range(10)
        )
    )
    evaluation_integrity_closed = (
        len(evaluation_integrity_rows) == len(SEEDS) * len(SHARDS)
        and all(
            row["nonfinite_count"] == 0
            and row["decode_failure_count"] == 0
            and row["public_component_mismatch"] == 0
            and row["settle_success_or_done_count"] == 0
            and row["validated_settle_transition_count"] > 0
            and row["validated_policy_transition_count"]
            == row["episode_step_count"]
            for row in evaluation_integrity_rows
        )
    )
    transition_counts = {
        "settle": sum(
            row["validated_settle_transition_count"]
            for row in evaluation_integrity_rows
        ),
        "policy": sum(
            row["validated_policy_transition_count"]
            for row in evaluation_integrity_rows
        ),
        "total": sum(
            row["validated_settle_transition_count"]
            + row["validated_policy_transition_count"]
            for row in evaluation_integrity_rows
        ),
        "expected_settle": len(SEEDS)
        * 10
        * EVALUATION_EPISODES_PER_TASK
        * evaluation_protocol(False, *SHARDS[0])["settle_steps"],
        "episode_step_total": sum(
            row["episode_step_count"] for row in evaluation_integrity_rows
        ),
    }
    transitions_closed = (
        transition_counts["settle"] == transition_counts["expected_settle"]
        and transition_counts["policy"] == transition_counts["episode_step_total"]
        and transition_counts["total"]
        == transition_counts["settle"] + transition_counts["policy"]
    )
    capability_conditions = {
        "primary_seed_is_preregistered_seed0": PRIMARY_CAPABILITY_SEED == 0,
        "primary_overall_at_least_floor": seed0_overall >= PRIMARY_CAPABILITY_FLOOR,
        "all_three_seed_reports_present": set(seed_summaries) == {"0", "1", "2"},
        "all_four_shards_per_seed_present_and_unique": completed_shards
        == {seed: set(SHARDS) for seed in SEEDS},
        "exactly_500_episodes_per_seed": all(
            summary["episode_count"] == 500 for summary in seed_summaries.values()
        ),
        "canonical_task_and_initial_state_protocol": canonical_panel_protocol,
        "no_nonfinite_or_decode_failures": evaluation_integrity_closed,
        "all_simulator_transitions_strictly_validated": transitions_closed,
        "no_best_seed_post_selection": True,
    }
    capability_pass = all(capability_conditions.values())
    end_audit = audit_direct_only_launch(
        PROJECT_ROOT,
        tuple(PROJECT_ROOT / path for path in _EXPECTED_SOURCE_CLOSURE),
        require_direct_qr=False,
    )
    if end_audit["source_sha256"] != _PREIMPORT_SOURCE_VERIFICATION["source_sha256"]:
        raise RuntimeError("Transitive Python closure changed during aggregation")
    for field in (
        "prohibited_calls_found",
        "prohibited_self_overlap_sites",
        "guarded_dormant_spectral_norm_sites",
        "duplicate_top_level_definition_sites",
    ):
        if end_audit[field] != []:
            raise RuntimeError(f"Final product lane static audit did not close {field}")
    if not json_type_exact_equal(
        load_and_verify_source_manifest(args.source_manifest), manifest_start
    ):
        raise RuntimeError("Authenticated manifest changed during aggregation")
    final_runtime_guard = assert_direct_only_runtime_guard(
        direct_only_runtime_report(), exact_allowed_calls=()
    )
    if final_runtime_guard["prohibited_attempt_count"] != 0:
        raise RuntimeError("A prohibited numerical route was attempted during aggregation")
    aggregate = {
        "schema": f"{SCHEMA}_three_seed_aggregate",
        "recipe_version": RECIPE_VERSION,
        "source_manifest_sha256": manifest_sha,
        "manifest_bundle_sha256": manifest_start["manifest_bundle_sha256"],
        "evaluation_result_sha256": evaluation_hashes,
        "training_artifact_sha256": artifact_hashes,
        "seeds": list(SEEDS),
        "preregistered_primary_seed": PRIMARY_CAPABILITY_SEED,
        "preregistered_primary_floor": PRIMARY_CAPABILITY_FLOOR,
        "seed_summaries": seed_summaries,
        "three_seed_mean": statistics.fmean(overalls),
        "three_seed_population_std": statistics.pstdev(overalls),
        "capability_gate_conditions": capability_conditions,
        "capability_gate_pass": capability_pass,
        "validated_transition_counts": transition_counts,
        "task_protocol": {str(task): EXPECTED_TASK_PROTOCOL[task] for task in range(10)},
        "transitive_direct_only_static_audit": end_audit,
        "runtime_direct_only_guard_at_import": _RUNTIME_GUARD_AT_IMPORT,
        "runtime_direct_only_guard_final": final_runtime_guard,
        "claim_boundary": DEPLOYMENT_CLAIM_BOUNDARY,
    }
    write_json_exclusive(aggregate_path, aggregate)
    aggregate_sha = file_sha256(aggregate_path)
    seed0_artifacts = artifact_hashes[str(PRIMARY_CAPABILITY_SEED)]
    gate = {
        "schema": f"{SCHEMA}_seed0_capability_gate",
        "recipe_version": RECIPE_VERSION,
        "preregistered_before_evaluation": True,
        "primary_seed": PRIMARY_CAPABILITY_SEED,
        "required_overall": PRIMARY_CAPABILITY_FLOOR,
        "observed_overall": seed0_overall,
        "conditions": capability_conditions,
        "approved_for_capable_global_claim": capability_pass,
        "capability_condition_satisfied_for_future_composite_attestation": capability_pass,
        "exact_odt_composite_attestation_approved": False,
        "checkpoint": checkpoint_path(
            args.run_root, PRIMARY_CAPABILITY_SEED, False
        ).as_posix(),
        "checkpoint_sha256": seed0_artifacts["checkpoint"],
        "metadata_sha256": seed0_artifacts["metadata"],
        "training_result_sha256": seed0_artifacts["training_result"],
        "precalibration_ema_state_sha256": seed0_artifacts[
            "precalibration_ema_state"
        ],
        "deployment_completion_sha256": seed0_artifacts[
            "deployment_completion"
        ],
        "aggregate": aggregate_path.as_posix(),
        "aggregate_sha256": aggregate_sha,
        "source_manifest_sha256": manifest_sha,
        "manifest_bundle_sha256": manifest_start["manifest_bundle_sha256"],
        "does_not_select_best_seed_post_hoc": True,
        "exact_odt_result_required_separately": True,
        "claim_boundary": DEPLOYMENT_CLAIM_BOUNDARY,
    }
    write_json_exclusive(gate_path, gate)
    independent = validate_capability_aggregate_evidence(
        args.run_root,
        manifest=manifest_start,
        manifest_sha256=manifest_sha,
    )
    if independent.get("validated") is not True:
        raise RuntimeError("Independent raw-evidence capability reconstruction failed")
    verify_capability_evidence_snapshots(independent["snapshots"])
    print(
        "RESULT",
        json.dumps(
            {
                "aggregate": aggregate_path.as_posix(),
                "aggregate_sha256": aggregate_sha,
                "gate": gate_path.as_posix(),
                "gate_sha256": file_sha256(gate_path),
                "seed0_overall": seed0_overall,
                "capability_gate_pass": capability_pass,
                "all_seed_overalls": overalls,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
