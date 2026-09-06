#!/usr/bin/env python3
"""Create a separate immutable Product checkpoint exact-ODT stage."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
_EXPECTED_PRESTAGE_ROOT = Path(
    "/work/joy/x-vla-product-pade-rational-odt-prestage-v3"
)
_PRESTAGE_MANIFEST_RELATIVE = Path(
    "athena/product_rational_exact_odt_prestage_manifest.json"
)
_PRESTAGE_LEDGER_RELATIVE = Path("prestage_ledger.json")
_PRESTAGE_SCHEMA = "xvla_product_pade_rational_exact_odt_v1_prestage_manifest_v3"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant is forbidden: {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Nonfinite JSON float is forbidden")
    return parsed


def _preimport_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preimport_canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _preimport_read_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or nonphysical: {path}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"Duplicate {label} key {key!r}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(),
        object_pairs_hook=reject_duplicates,
        parse_constant=_reject_json_constant,
        parse_float=_finite_json_float,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not an object")
    return payload


def _preimport_safe_member(relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or not value.parts or any(
        part in {"", ".", ".."} for part in value.parts
    ):
        raise RuntimeError(f"Unsafe prestage member {relative!r}")
    cursor = SOURCE_ROOT
    for part in value.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(f"Prestage member traverses a link: {relative}")
    resolved = cursor.resolve(strict=True)
    if resolved == SOURCE_ROOT or SOURCE_ROOT not in resolved.parents or not resolved.is_file():
        raise RuntimeError(f"Prestage member escapes or is missing: {relative}")
    return resolved


def _verify_prestage_preimport() -> dict[str, object]:
    if SOURCE_ROOT != _EXPECTED_PRESTAGE_ROOT:
        raise RuntimeError(
            f"Deferred staging must execute from {_EXPECTED_PRESTAGE_ROOT}, got {SOURCE_ROOT}"
        )
    manifest_path = SOURCE_ROOT / _PRESTAGE_MANIFEST_RELATIVE
    manifest = _preimport_read_object(manifest_path, "Prestage manifest")
    closure = manifest.get("closure")
    if manifest.get("schema") != _PRESTAGE_SCHEMA or not isinstance(closure, dict):
        raise RuntimeError("Prestage manifest schema or closure differs")
    minimum = {
        "athena/__init__.py",
        "athena/product_rational_exact_odt_protocol.py",
        "athena/stage_product_rational_exact_odt.py",
        "scripts/odt_direct_only_compliance.py",
    }
    if not minimum.issubset(closure):
        raise RuntimeError("Prestage manifest omits a staging authority module")
    for relative, expected in closure.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise RuntimeError("Malformed prestage manifest entry")
        if _preimport_file_sha256(_preimport_safe_member(relative)) != expected:
            raise RuntimeError(f"Prestage authority byte changed: {relative}")
    bundle = _preimport_canonical_sha256(dict(sorted(closure.items())))
    if manifest.get("bundle_sha256") != bundle:
        raise RuntimeError("Prestage authority bundle differs")
    ledger_path = SOURCE_ROOT / _PRESTAGE_LEDGER_RELATIVE
    ledger = _preimport_read_object(ledger_path, "Prestage ledger")
    if (
        ledger.get("schema") != f"{_PRESTAGE_SCHEMA}_ledger"
        or ledger.get("prestage_root") != SOURCE_ROOT.as_posix()
        or ledger.get("prestage_manifest") != _PRESTAGE_MANIFEST_RELATIVE.as_posix()
        or ledger.get("prestage_manifest_sha256")
        != _preimport_file_sha256(manifest_path)
        or ledger.get("prestage_bundle_sha256") != bundle
        or ledger.get("no_job_submitted") is not True
    ):
        raise RuntimeError("Prestage ledger differs from the frozen authority")
    return {
        "manifest": manifest,
        "manifest_sha256": _preimport_file_sha256(manifest_path),
        "ledger": ledger,
        "ledger_sha256": _preimport_file_sha256(ledger_path),
    }


_PREIMPORT_PRESTAGE = _verify_prestage_preimport()

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from athena.product_rational_exact_odt_protocol import (
    AUTHENTICATED_INPUTS,
    CAPABILITY_RUN_ROOT,
    CAPABILITY_STAGE_ROOT,
    DOOMS_REFERENCE_SHA256,
    LAUNCH_CLOSURE,
    MANIFEST_PATH,
    ODT_PRESTAGE_ROOT,
    ODT_STAGE_ROOT,
    ODT_SOURCE_OVERRIDES,
    PROGRESS_DIRECTORY,
    PRESTAGE_CLOSURE,
    PRESTAGE_LEDGER_PATH,
    PRESTAGE_MANIFEST_PATH,
    PRESTAGE_SCHEMA,
    GATES_DIRECTORY,
    RESULTS_DIRECTORY,
    SOURCE_CLOSURE,
    build_manifest,
    file_sha256,
    publish_json_exclusive,
    read_json_object,
)
from athena.product_rational_protocol import (
    AUTHENTICATED_INPUTS as CAPABILITY_AUTHENTICATED_INPUTS,
    CALIBRATION_RECOVERY_SCOPE,
    CALIBRATION_RESUME_PROOF_SCHEMA,
    LAUNCH_CLOSURE as CAPABILITY_LAUNCH_CLOSURE,
    SCHEMA as CAPABILITY_SCHEMA,
    SMOKE_STEPS,
    SOURCE_CLOSURE as CAPABILITY_SOURCE_CLOSURE,
    calibration_resume_proof_path,
    checkpoint_path,
    deployment_completion_path,
    load_and_validate_deployment_completion,
    load_training_only_precalibration_state,
    manifest_bundle_sha256 as capability_manifest_bundle_sha256,
    metadata_path,
    precalibration_state_path,
    read_json_object_physical,
    source_bundle_sha256 as capability_section_bundle_sha256,
    training_result_path,
    validate_same_allocation_calibration_resume_proof,
)
from scripts.odt_direct_only_compliance import audit_direct_only_launch


def _require_physical(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or nonphysical: {path}")


def _capability_manifest() -> tuple[Path, dict[str, object]]:
    path = (
        CAPABILITY_STAGE_ROOT
        / CAPABILITY_RUN_ROOT
        / "manifest/product_pade_rational_source_manifest.json"
    )
    payload = read_json_object(path, label="Capability source manifest")
    if payload.get("schema") != f"{CAPABILITY_SCHEMA}_source_manifest":
        raise RuntimeError("Capability source manifest schema differs")
    sections = (
        ("source_closure", CAPABILITY_SOURCE_CLOSURE, "source_bundle_sha256"),
        (
            "authenticated_inputs",
            CAPABILITY_AUTHENTICATED_INPUTS,
            "authenticated_inputs_bundle_sha256",
        ),
        ("launch_closure", CAPABILITY_LAUNCH_CLOSURE, "launch_bundle_sha256"),
    )
    for section, expected, bundle_key in sections:
        stored = payload.get(section)
        if not isinstance(stored, dict) or tuple(sorted(stored)) != tuple(sorted(expected)):
            raise RuntimeError(f"Capability manifest {section} differs")
        if payload.get(bundle_key) != capability_section_bundle_sha256(stored):
            raise RuntimeError(f"Capability manifest {bundle_key} differs")
        for relative, digest in stored.items():
            candidate = CAPABILITY_STAGE_ROOT / relative
            _require_physical(candidate, f"Capability {section} member")
            if file_sha256(candidate) != digest:
                raise RuntimeError(f"Capability stage byte drift: {relative}")
    if payload.get("manifest_bundle_sha256") != capability_manifest_bundle_sha256(
        payload
    ):
        raise RuntimeError("Capability aggregate manifest digest differs")
    return path, payload


def _copy_physical(source: Path, destination: Path) -> str:
    _require_physical(source, "Staging source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.is_symlink() or not destination.is_file():
        raise RuntimeError(f"Staged file is nonphysical: {destination}")
    source_digest = file_sha256(source)
    if file_sha256(destination) != source_digest:
        raise RuntimeError(f"Staged bytes differ: {destination}")
    destination.chmod(0o444)
    return source_digest


def _artifact_sources() -> dict[str, Path]:
    root = CAPABILITY_STAGE_ROOT / CAPABILITY_RUN_ROOT
    return {
        "inputs/capability_preflight.json": root
        / "gates/product_pade_rational_preflight.json",
        "inputs/capability_source_manifest.json": root
        / "manifest/product_pade_rational_source_manifest.json",
        "inputs/odt_prestage_ledger.json": SOURCE_ROOT / PRESTAGE_LEDGER_PATH,
        "inputs/odt_prestage_manifest.json": SOURCE_ROOT / PRESTAGE_MANIFEST_PATH,
        "inputs/smoke_checkpoint.pt": checkpoint_path(root, 0, True),
        "inputs/smoke_calibration_resume_proof.json": calibration_resume_proof_path(
            root
        ),
        "inputs/smoke_deployment_completion.json": deployment_completion_path(
            root, 0, True
        ),
        "inputs/smoke_metadata.json": metadata_path(root, 0, True),
        "inputs/smoke_precalibration_ema_state.pt": precalibration_state_path(
            root, 0, True
        ),
        "inputs/smoke_training_result.json": training_result_path(root, 0, True),
        "inputs/seed0_checkpoint.pt": root
        / "checkpoints/product_pade_rational_s0.pt",
        "inputs/seed0_deployment_completion.json": root
        / "gates/product_pade_rational_s0_deployment_complete.json",
        "inputs/seed0_metadata.json": root
        / "metadata/product_pade_rational_s0.json",
        "inputs/seed0_precalibration_ema_state.pt": root
        / "training_only/precalibration_ema_product_pade_rational_s0.pt",
        "inputs/seed0_training_result.json": root
        / "results/train_product_pade_rational_s0.json",
        "reference/dooms_xnets_2504.02667.pdf": SOURCE_ROOT
        / "reference/dooms_xnets_2504.02667.pdf",
    }


def _validate_smoke_calibration_resume_proof(
    proof_path: Path,
    *,
    capability_root: Path,
    capability_manifest: dict[str, object],
    capability_manifest_path: Path,
    capability_preflight_path: Path,
) -> dict[str, object]:
    """Bind the staged proof to the physical fresh-smoke producer chain."""

    smoke_checkpoint = checkpoint_path(capability_root, 0, True)
    smoke_metadata_path = metadata_path(capability_root, 0, True)
    smoke_training_path = training_result_path(capability_root, 0, True)
    smoke_precalibration_path = precalibration_state_path(capability_root, 0, True)
    smoke_completion_path = deployment_completion_path(capability_root, 0, True)
    for path, label in (
        (smoke_checkpoint, "Smoke checkpoint"),
        (smoke_metadata_path, "Smoke metadata"),
        (smoke_training_path, "Smoke training result"),
        (smoke_precalibration_path, "Smoke pre-calibration state"),
        (smoke_completion_path, "Smoke deployment completion"),
        (proof_path, "Smoke calibration resume proof"),
    ):
        _require_physical(path, label)
    smoke_completion = load_and_validate_deployment_completion(
        smoke_completion_path,
        expected_seed=0,
        expected_mode="smoke",
        expected_checkpoint=smoke_checkpoint,
        expected_metadata=smoke_metadata_path,
        expected_training_result=smoke_training_path,
        expected_precalibration_ema_state=smoke_precalibration_path,
        expected_source_manifest_sha256=file_sha256(capability_manifest_path),
        expected_manifest_bundle_sha256=str(
            capability_manifest["manifest_bundle_sha256"]
        ),
    )
    smoke_metadata = read_json_object_physical(
        smoke_metadata_path, label="Smoke metadata"
    )
    smoke_training = read_json_object_physical(
        smoke_training_path, label="Smoke training result"
    )
    proof = read_json_object_physical(
        proof_path, label="Smoke calibration resume proof"
    )
    smoke_precalibration = load_training_only_precalibration_state(
        smoke_precalibration_path,
        expected_file_sha256=file_sha256(smoke_precalibration_path),
        expected_seed=0,
        expected_mode="smoke",
        expected_step_count=SMOKE_STEPS,
        expected_source_manifest_sha256=file_sha256(capability_manifest_path),
        expected_manifest_bundle_sha256=str(
            capability_manifest["manifest_bundle_sha256"]
        ),
        expected_source_bundle_sha256=str(
            capability_manifest["source_bundle_sha256"]
        ),
        expected_training_config_sha256=smoke_metadata["training_config"]["sha256"],
    )
    capability_static_audit = audit_direct_only_launch(
        CAPABILITY_STAGE_ROOT,
        tuple(
            CAPABILITY_STAGE_ROOT / relative
            for relative in CAPABILITY_SOURCE_CLOSURE
        ),
        require_direct_qr=False,
    )
    strict_proof = validate_same_allocation_calibration_resume_proof(
        proof,
        expected_source_manifest_sha256=file_sha256(capability_manifest_path),
        expected_manifest_bundle_sha256=str(
            capability_manifest["manifest_bundle_sha256"]
        ),
        expected_preflight_sha256=file_sha256(capability_preflight_path),
        expected_source_closure=capability_manifest["source_closure"],
        expected_static_audit=capability_static_audit,
        expected_checkpoint_sha256=file_sha256(smoke_checkpoint),
        expected_metadata_sha256=file_sha256(smoke_metadata_path),
        expected_training_result_sha256=file_sha256(smoke_training_path),
        expected_precalibration_sha256=file_sha256(smoke_precalibration_path),
        expected_precalibration_training_state_sha256=smoke_precalibration[
            "training_state_sha256"
        ],
        expected_completion_sha256=file_sha256(smoke_completion_path),
        expected_completion_bundle_sha256=smoke_completion[
            "artifact_bundle_sha256"
        ],
        expected_training_environment=smoke_metadata["training_environment"],
        expected_calibration_attestation=smoke_metadata[
            "post_ema_simultaneous_rational_calibration_attestation"
        ],
        expected_state_transition=smoke_metadata["calibration_state_transition"],
        expected_deployment_equivalence=smoke_metadata["deployment_export"][
            "inference_equivalence"
        ],
    )
    proof_static = proof.get("transitive_direct_only_static_audit", {})
    proof_runtime = proof.get("runtime_direct_only_guard", {})
    proof_replay = proof.get("deployment_equivalence", {})
    proof_checks = proof_replay.get("bitwise_checks", {}) if isinstance(
        proof_replay, dict
    ) else {}
    expected_replay = smoke_metadata.get("deployment_export", {}).get(
        "inference_equivalence", {}
    )
    proof_allocation = proof.get("allocation", {})
    reference_allocation = smoke_metadata.get("training_environment", {})
    allocation_keys = {
        "gpu",
        "compute_capability",
        "slurm_job_id",
        "slurm_job_gpus",
        "cuda_visible_devices",
    }
    source_closure = capability_manifest["source_closure"]
    conditions = {
        "schema": proof.get("schema")
        == CALIBRATION_RESUME_PROOF_SCHEMA,
        "strict_validator": strict_proof.get("validated") is True
        and all(strict_proof.get("conditions", {}).values()),
        "identity": proof.get("seed") == 0 and proof.get("mode") == "smoke",
        "producer_schemas": smoke_metadata.get("schema") == CAPABILITY_SCHEMA
        and smoke_metadata.get("seed") == 0
        and smoke_metadata.get("mode") == "smoke"
        and smoke_training.get("schema")
        == f"{CAPABILITY_SCHEMA}_training_result"
        and smoke_training.get("seed") == 0
        and smoke_training.get("mode") == "smoke",
        "completion": proof.get("fresh_deployment_completion_sha256")
        == file_sha256(smoke_completion_path)
        and proof.get("fresh_deployment_completion_bundle_sha256")
        == smoke_completion["artifact_bundle_sha256"],
        "artifact_hashes": proof.get("fresh_checkpoint_sha256")
        == file_sha256(smoke_checkpoint)
        and proof.get("fresh_metadata_sha256") == file_sha256(smoke_metadata_path)
        and proof.get("fresh_training_result_sha256")
        == file_sha256(smoke_training_path)
        and proof.get("precalibration_ema_state_sha256")
        == file_sha256(smoke_precalibration_path),
        "external_sha_zero_training": proof.get("external_sha_authority_supplied")
        is True
        and proof.get("optimizer_steps_executed") == 0,
        "same_allocation": proof.get(
            "same_slurm_gpu_allocation_as_fresh_smoke"
        )
        is True
        and isinstance(proof_allocation, dict)
        and set(proof_allocation) == allocation_keys
        and all(
            isinstance(proof_allocation.get(key), str)
            and bool(proof_allocation.get(key))
            for key in (
                "slurm_job_id",
                "slurm_job_gpus",
                "cuda_visible_devices",
            )
        )
        and proof_allocation.get("gpu") == "NVIDIA RTX A6000"
        and proof_allocation.get("compute_capability") == [8, 6]
        and all(
            proof_allocation.get(key) == reference_allocation.get(key)
            for key in allocation_keys
        ),
        "exact_replay": proof.get(
            "fresh_and_resume_calibration_report_bitwise_equal"
        )
        is True
        and proof.get("fresh_and_resume_final_running_ms_bitwise_equal") is True
        and proof.get("fresh_and_resume_deployment_state_bitwise_equal") is True
        and proof.get(
            "fresh_and_resume_raw_components_and_actions_bitwise_equal"
        )
        is True
        and isinstance(expected_replay, dict)
        and bool(expected_replay)
        and isinstance(expected_replay.get("output_tensor_sha256"), str)
        and len(expected_replay["output_tensor_sha256"]) == 64
        and all(
            character in "0123456789abcdef"
            for character in expected_replay["output_tensor_sha256"]
        )
        and proof_replay == expected_replay
        and proof.get("fresh_output_tensor_sha256")
        == proof.get("resume_output_tensor_sha256")
        == expected_replay.get("output_tensor_sha256")
        and isinstance(proof_checks, dict)
        and bool(proof_checks)
        and all(value is True for value in proof_checks.values())
        and proof.get("removed_training_only_keys")
        == ["teacher_head.bias", "teacher_head.weight"],
        "attestations": proof.get("calibration_attestation")
        == smoke_metadata.get(
            "post_ema_simultaneous_rational_calibration_attestation"
        )
        == smoke_training.get(
            "post_ema_simultaneous_rational_calibration_attestation"
        )
        and proof.get("calibration_state_transition")
        == smoke_metadata.get("calibration_state_transition")
        == smoke_training.get("calibration_state_transition"),
        "source": proof.get("source_manifest_sha256")
        == file_sha256(capability_manifest_path)
        and proof.get("manifest_bundle_sha256")
        == capability_manifest["manifest_bundle_sha256"]
        and proof.get("preflight_certificate_sha256")
        == file_sha256(capability_preflight_path)
        and proof.get("source_snapshot_start")
        == proof.get("source_snapshot_end")
        == source_closure,
        "static": isinstance(proof_static, dict)
        and proof_static.get("source_sha256") == source_closure
        and proof_static.get("prohibited_calls_found") == []
        and proof_static.get("prohibited_self_overlap_sites") == []
        and proof_static.get("guarded_dormant_spectral_norm_sites") == [],
        "runtime": isinstance(proof_runtime, dict)
        and proof_runtime.get("installed") is True
        and proof_runtime.get("allowed_calls") == []
        and proof_runtime.get("prohibited_attempt_count") == 0,
        "scope": proof.get("recovery_scope") == CALIBRATION_RECOVERY_SCOPE
        and proof.get("cross_version_recovery_supported") is False,
    }
    failed = sorted(name for name, passed in conditions.items() if not passed)
    if failed:
        raise RuntimeError(
            "Same-allocation smoke calibration proof failed staging gates: "
            f"{failed}"
        )
    return {"payload": proof, "conditions": conditions}


def _validate_capability_artifact_chain(
    sources: dict[str, Path], capability_manifest: dict[str, object]
) -> None:
    for relative in AUTHENTICATED_INPUTS:
        _require_physical(sources[relative], relative)
    if file_sha256(sources["reference/dooms_xnets_2504.02667.pdf"]) != DOOMS_REFERENCE_SHA256:
        raise RuntimeError("Pinned Dooms reference SHA-256 differs")
    checkpoint_sha = file_sha256(sources["inputs/seed0_checkpoint.pt"])
    metadata_sha = file_sha256(sources["inputs/seed0_metadata.json"])
    preflight_sha = file_sha256(sources["inputs/capability_preflight.json"])
    source_manifest_sha = file_sha256(
        sources["inputs/capability_source_manifest.json"]
    )
    training = read_json_object(
        sources["inputs/seed0_training_result.json"], label="Seed0 training result"
    )
    metadata = read_json_object(
        sources["inputs/seed0_metadata.json"], label="Seed0 metadata"
    )
    completion = load_and_validate_deployment_completion(
        sources["inputs/seed0_deployment_completion.json"],
        expected_seed=0,
        expected_mode="full",
        expected_checkpoint=sources["inputs/seed0_checkpoint.pt"],
        expected_metadata=sources["inputs/seed0_metadata.json"],
        expected_training_result=sources["inputs/seed0_training_result.json"],
        expected_precalibration_ema_state=(
            CAPABILITY_STAGE_ROOT
            / CAPABILITY_RUN_ROOT
            / "training_only/precalibration_ema_product_pade_rational_s0.pt"
        ),
        expected_source_manifest_sha256=source_manifest_sha,
        expected_manifest_bundle_sha256=str(
            capability_manifest["manifest_bundle_sha256"]
        ),
    )
    capability_root = CAPABILITY_STAGE_ROOT / CAPABILITY_RUN_ROOT
    smoke_resume = _validate_smoke_calibration_resume_proof(
        sources["inputs/smoke_calibration_resume_proof.json"],
        capability_root=capability_root,
        capability_manifest=capability_manifest,
        capability_manifest_path=sources["inputs/capability_source_manifest.json"],
        capability_preflight_path=sources["inputs/capability_preflight.json"],
    )
    fixed = {
        "training_schema": training.get("schema")
        == f"{CAPABILITY_SCHEMA}_training_result",
        "training_seed0": training.get("seed") == 0,
        "training_full": training.get("mode") == "full",
        "training_checkpoint": training.get("checkpoint_sha256") == checkpoint_sha,
        "training_metadata": training.get("metadata_sha256") == metadata_sha,
        "training_manifest": training.get("source_manifest_sha256")
        == source_manifest_sha,
        "metadata_schema": metadata.get("schema")
        == CAPABILITY_SCHEMA,
        "metadata_seed0": metadata.get("seed") == 0,
        "metadata_full": metadata.get("mode") == "full",
        "metadata_checkpoint": metadata.get("checkpoint_sha256") == checkpoint_sha,
        "metadata_manifest": metadata.get("source_manifest", {}).get("sha256")
        == source_manifest_sha,
        "metadata_preflight": metadata.get("preflight_certificate", {}).get("sha256")
        == preflight_sha,
        "training_preflight": training.get("preflight_certificate", {}).get("sha256")
        == preflight_sha,
        "deployment_completion": completion.get("complete") is True,
        "smoke_calibration_resume_proof": all(
            smoke_resume["conditions"].values()
        ),
    }
    failed = sorted(name for name, passed in fixed.items() if not passed)
    if failed:
        raise RuntimeError(f"Capability seed0 artifact chain failed staging gates: {failed}")


def main() -> None:
    if SOURCE_ROOT != ODT_PRESTAGE_ROOT or SOURCE_ROOT != _EXPECTED_PRESTAGE_ROOT:
        raise RuntimeError(f"Staging must run from {ODT_PRESTAGE_ROOT}, got {SOURCE_ROOT}")
    prestage_manifest = _PREIMPORT_PRESTAGE["manifest"]
    if (
        PRESTAGE_SCHEMA != _PRESTAGE_SCHEMA
        or PRESTAGE_MANIFEST_PATH != _PRESTAGE_MANIFEST_RELATIVE
        or PRESTAGE_LEDGER_PATH != _PRESTAGE_LEDGER_RELATIVE
        or not isinstance(prestage_manifest, dict)
        or tuple(sorted(prestage_manifest.get("closure", {}))) != PRESTAGE_CLOSURE
    ):
        raise RuntimeError("Postimport prestage authority tuple differs")
    if os.path.lexists(ODT_STAGE_ROOT):
        raise FileExistsError(f"Refusing existing exact-ODT stage {ODT_STAGE_ROOT}")
    capability_manifest_path, capability_manifest = _capability_manifest()
    capability_sources = capability_manifest["source_closure"]
    if not isinstance(capability_sources, dict):
        raise RuntimeError("Capability source mapping is malformed")
    local_audit = audit_direct_only_launch(
        SOURCE_ROOT,
        tuple(SOURCE_ROOT / relative for relative in SOURCE_CLOSURE),
    )
    if local_audit["source_sha256"] != {
        relative: file_sha256(SOURCE_ROOT / relative) for relative in SOURCE_CLOSURE
    }:
        raise RuntimeError("Exact-ODT source tuple differs from the transitive audit")
    for field in (
        "prohibited_calls_found",
        "prohibited_self_overlap_sites",
        "guarded_dormant_spectral_norm_sites",
    ):
        if local_audit[field] != []:
            raise RuntimeError(f"Exact-ODT staging audit did not close {field}")
    if local_audit["source_sha256"] != {
        relative: prestage_manifest["closure"][relative]
        for relative in SOURCE_CLOSURE
    }:
        raise RuntimeError("Preimport authority and postimport static source map differ")
    for relative in CAPABILITY_SOURCE_CLOSURE:
        if (
            relative in SOURCE_CLOSURE
            and relative not in ODT_SOURCE_OVERRIDES
            and file_sha256(SOURCE_ROOT / relative) != capability_sources[relative]
        ):
            raise RuntimeError(f"Mutable source differs from trained capability bytes: {relative}")

    artifact_sources = _artifact_sources()
    _validate_capability_artifact_chain(artifact_sources, capability_manifest)
    ODT_STAGE_ROOT.mkdir(mode=0o755)
    staged: dict[str, str] = {}
    for relative in sorted(set(SOURCE_CLOSURE + LAUNCH_CLOSURE)):
        source = (
            CAPABILITY_STAGE_ROOT / relative
            if relative in capability_sources and relative not in ODT_SOURCE_OVERRIDES
            else SOURCE_ROOT / relative
        )
        staged[relative] = _copy_physical(source, ODT_STAGE_ROOT / relative)
    for relative, source in artifact_sources.items():
        staged[relative] = _copy_physical(source, ODT_STAGE_ROOT / relative)

    (ODT_STAGE_ROOT / RESULTS_DIRECTORY).mkdir(parents=True, exist_ok=True)
    (ODT_STAGE_ROOT / GATES_DIRECTORY).mkdir(parents=True, exist_ok=True)
    (ODT_STAGE_ROOT / PROGRESS_DIRECTORY).mkdir(parents=True, exist_ok=True)
    (ODT_STAGE_ROOT / "athena/logs").mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(ODT_STAGE_ROOT)
    manifest_output = ODT_STAGE_ROOT / MANIFEST_PATH
    manifest_sha = publish_json_exclusive(manifest_output, manifest)

    writable = {
        ODT_STAGE_ROOT / RESULTS_DIRECTORY,
        ODT_STAGE_ROOT / GATES_DIRECTORY,
        ODT_STAGE_ROOT / PROGRESS_DIRECTORY,
        ODT_STAGE_ROOT / "athena/logs",
    }
    ledger_payload = {
        "schema": f"{manifest['schema']}_stage",
        "stage_root": ODT_STAGE_ROOT.as_posix(),
        "capability_stage_root": CAPABILITY_STAGE_ROOT.as_posix(),
        "capability_source_manifest": capability_manifest_path.as_posix(),
        "capability_source_manifest_sha256": file_sha256(capability_manifest_path),
        "prestage_manifest_sha256": _PREIMPORT_PRESTAGE["manifest_sha256"],
        "prestage_ledger_sha256": _PREIMPORT_PRESTAGE["ledger_sha256"],
        "prestage_bundle_sha256": prestage_manifest["bundle_sha256"],
        "source_manifest": MANIFEST_PATH.as_posix(),
        "source_manifest_sha256": manifest_sha,
        "staged_sha256": staged,
        "static_audit": local_audit,
        "no_job_submitted": True,
    }
    ledger_path = ODT_STAGE_ROOT / "stage_ledger.json"
    encoded = (
        json.dumps(ledger_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    with ledger_path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    ledger_path.chmod(0o444)
    for path in sorted(
        (item for item in ODT_STAGE_ROOT.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        path.chmod(0o755 if path in writable else 0o555)
    ODT_STAGE_ROOT.chmod(0o555)
    print("STAGED", json.dumps(ledger_payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
