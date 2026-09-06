#!/usr/bin/env python3
"""Immutable paths and artifact helpers for the trained Product exact-ODT lane."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import stat
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "xvla_product_pade_rational_exact_odt_v1"
MANIFEST_SCHEMA = f"{SCHEMA}_source_manifest"
CAPABILITY_STAGE_ROOT = Path("/work/joy/x-vla-product-pade-rational-v2")
ODT_STAGE_ROOT = Path("/work/joy/x-vla-product-pade-rational-odt-v3")
ODT_PRESTAGE_ROOT = Path("/work/joy/x-vla-product-pade-rational-odt-prestage-v3")
MUTABLE_SOURCE_ROOT = Path("/work/joy/x-vla-workshop")
CAPABILITY_RUN_ROOT = Path("athena/results/product_pade_rational_v2")
MANIFEST_PATH = Path("athena/product_rational_exact_odt_sources.json")
PRESTAGE_MANIFEST_PATH = Path(
    "athena/product_rational_exact_odt_prestage_manifest.json"
)
PRESTAGE_LEDGER_PATH = Path("prestage_ledger.json")
RESULTS_DIRECTORY = Path("athena/results")
GATES_DIRECTORY = Path("athena/results/product_pade_rational_exact_odt_v1_gates")
PROGRESS_DIRECTORY = Path("athena/results/product_pade_rational_exact_odt_v1_progress")
SYNTHETIC_REPLAY_TASKS = (0, 3, 6, 9)
SYNTHETIC_REPLAY_SEED = 20260905
REPLAY_RELATIVE_TOLERANCE = 3e-8
CLONE_RELATIVE_TOLERANCE = 2e-10
OFFDIAGONAL_RELATIVE_TOLERANCE = 3e-10
PHYSICAL_MATERIALIZATION_TARGET = 0.999
DOOMS_REFERENCE_SHA256 = (
    "a97f59a91d90ffde56bc22bf0b3c79e0891ff88197ca9d7670331e4208277559"
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant is forbidden: {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Nonfinite JSON float is forbidden")
    return parsed
ODT_SOURCE_OVERRIDES = (
    "scripts/odt_direct_only_compliance.py",
)


# This exact tuple is independently duplicated in the authenticated runner before
# any project import. It includes the full frozen capability source closure plus
# the canonical compiler, direct-only oracle, and compression tests used here.
SOURCE_CLOSURE = (
    "athena/__init__.py",
    "athena/aggregate_product_rational_results.py",
    "athena/eval_product_rational_checkpoint.py",
    "athena/prepare_product_rational_launch.py",
    "athena/prepare_product_rational_exact_odt_prestage.py",
    "athena/product_rational_exact_odt_protocol.py",
    "athena/product_rational_protocol.py",
    "athena/run_product_rational_exact_odt.py",
    "athena/stage_product_rational_exact_odt.py",
    "athena/stage_product_rational_launch.py",
    "athena/train_product_rational_checkpoint.py",
    "scripts/__init__.py",
    "scripts/odt_direct_only_compliance.py",
    "tests/test_direct_odt_clone_reference.py",
    "tests/test_direct_odt_truncation.py",
    "tests/test_implicit_sparse_projective_odt_vla.py",
    "tests/test_product_rational_exact_odt_lane.py",
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
    "xvla/train/__init__.py",
    "xvla/train/direct_odt_clone_reference.py",
    "xvla/train/direct_odt_truncation.py",
    "xvla/train/implicit_sparse_projective_odt.py",
    "xvla/train/implicit_sparse_projective_odt_all_tokens.py",
    "xvla/train/implicit_sparse_projective_odt_vla.py",
)

AUTHENTICATED_INPUTS = (
    "inputs/capability_preflight.json",
    "inputs/capability_source_manifest.json",
    "inputs/odt_prestage_ledger.json",
    "inputs/odt_prestage_manifest.json",
    "inputs/smoke_checkpoint.pt",
    "inputs/smoke_calibration_resume_proof.json",
    "inputs/smoke_deployment_completion.json",
    "inputs/smoke_metadata.json",
    "inputs/smoke_precalibration_ema_state.pt",
    "inputs/smoke_training_result.json",
    "inputs/seed0_checkpoint.pt",
    "inputs/seed0_deployment_completion.json",
    "inputs/seed0_metadata.json",
    "inputs/seed0_precalibration_ema_state.pt",
    "inputs/seed0_training_result.json",
    "reference/dooms_xnets_2504.02667.pdf",
)

LAUNCH_CLOSURE = (
    "athena/slurm_product_rational_odt_stage.sbatch",
    "athena/slurm_product_rational_odt_composite.sbatch",
    "athena/slurm_product_rational_odt_full.sbatch",
    "athena/slurm_product_rational_odt_validate.sbatch",
    "athena/submit_product_rational_exact_odt.sh",
)

# This bundle exists before the trained seed is available.  The delayed staging
# job executes exclusively from this read-only tree, never from the mutable
# workshop checkout.
PRESTAGE_SCHEMA = f"{SCHEMA}_prestage_manifest_v3"
PRESTAGE_CLOSURE = tuple(
    sorted(
        set(
            SOURCE_CLOSURE
            + LAUNCH_CLOSURE
            + ("reference/dooms_xnets_2504.02667.pdf",)
        )
    )
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or nonphysical: {path}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"Duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(),
        object_pairs_hook=reject_duplicates,
        parse_constant=_reject_json_constant,
        parse_float=_finite_json_float,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return payload


def closure_snapshot(paths: tuple[str, ...], root: Path | None = None) -> dict[str, str]:
    base = repository_root() if root is None else root.resolve()
    result: dict[str, str] = {}
    for relative in paths:
        candidate = base / relative
        if candidate.is_symlink():
            raise RuntimeError(f"Closure member is a link: {relative}")
        path = candidate.resolve(strict=True)
        if path == base or base not in path.parents or not path.is_file():
            raise RuntimeError(f"Closure member escapes or is missing: {relative}")
        result[relative] = file_sha256(path)
    return result


def manifest_bound_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
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


def build_manifest(root: Path | None = None) -> dict[str, Any]:
    source = closure_snapshot(SOURCE_CLOSURE, root)
    inputs = closure_snapshot(AUTHENTICATED_INPUTS, root)
    launch = closure_snapshot(LAUNCH_CLOSURE, root)
    payload = {
        "schema": MANIFEST_SCHEMA,
        "source_closure": source,
        "source_bundle_sha256": canonical_sha256(dict(sorted(source.items()))),
        "authenticated_inputs": inputs,
        "authenticated_inputs_bundle_sha256": canonical_sha256(
            dict(sorted(inputs.items()))
        ),
        "launch_closure": launch,
        "launch_bundle_sha256": canonical_sha256(dict(sorted(launch.items()))),
    }
    payload["manifest_bundle_sha256"] = canonical_sha256(
        manifest_bound_fields(payload)
    )
    return payload


def verify_manifest(path: Path = MANIFEST_PATH, root: Path | None = None) -> dict[str, Any]:
    base = repository_root() if root is None else root.resolve()
    manifest_path = path if path.is_absolute() else base / path
    payload = read_json_object(manifest_path, label="Exact-ODT source manifest")
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError("Exact-ODT source manifest schema differs")
    for section, expected_paths, bundle_key in (
        ("source_closure", SOURCE_CLOSURE, "source_bundle_sha256"),
        (
            "authenticated_inputs",
            AUTHENTICATED_INPUTS,
            "authenticated_inputs_bundle_sha256",
        ),
        ("launch_closure", LAUNCH_CLOSURE, "launch_bundle_sha256"),
    ):
        stored = payload.get(section)
        if not isinstance(stored, dict) or tuple(sorted(stored)) != tuple(
            sorted(expected_paths)
        ):
            raise RuntimeError(f"Exact-ODT manifest {section} differs")
        if payload.get(bundle_key) != canonical_sha256(dict(sorted(stored.items()))):
            raise RuntimeError(f"Exact-ODT manifest {bundle_key} differs")
        if closure_snapshot(expected_paths, base) != stored:
            raise RuntimeError(f"Exact-ODT {section} bytes changed")
    if payload.get("manifest_bundle_sha256") != canonical_sha256(
        manifest_bound_fields(payload)
    ):
        raise RuntimeError("Exact-ODT aggregate manifest digest differs")
    return payload


def _validate_output(path: Path, *, directory: Path) -> Path:
    base = repository_root()
    raw = path if path.is_absolute() else base / path
    if os.path.lexists(raw):
        raise FileExistsError(f"Refusing to replace existing output {raw}")
    allowed_path = base / directory
    if allowed_path.is_symlink() or not allowed_path.is_dir():
        raise RuntimeError(f"Output directory is missing or nonphysical: {allowed_path}")
    allowed = allowed_path.resolve(strict=True)
    resolved = raw.resolve(strict=False)
    if resolved.parent != allowed or not resolved.name:
        raise RuntimeError(f"Output must be a direct child of {allowed}")
    return resolved


def validate_result_output(path: Path) -> Path:
    return _validate_output(path, directory=RESULTS_DIRECTORY)


def validate_gate_output(path: Path) -> Path:
    return _validate_output(path, directory=GATES_DIRECTORY)


def validate_progress_output(path: Path) -> Path:
    return _validate_output(path, directory=PROGRESS_DIRECTORY)


def publish_json_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode(
        "utf-8"
    )
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(12)}.tmp")
    linked = False
    source_stat: os.stat_result | None = None
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        source_stat = os.lstat(temporary)
        if not stat.S_ISREG(source_stat.st_mode):
            raise RuntimeError("Exclusive output temporary is not a regular file")
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        if file_sha256(path) != digest:
            raise RuntimeError("Published output bytes differ")
        os.chmod(path, 0o444, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return digest
    except BaseException:
        if linked and source_stat is not None and os.path.lexists(path):
            observed = os.lstat(path)
            if observed.st_dev == source_stat.st_dev and observed.st_ino == source_stat.st_ino:
                path.unlink()
        raise
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def input_path(name: str) -> Path:
    if name not in AUTHENTICATED_INPUTS:
        raise KeyError(name)
    return repository_root() / name


def preflight_gate_path() -> Path:
    return repository_root() / GATES_DIRECTORY / "seed0_checkpoint_integrity_gate.json"


def clone_oracle_result_path() -> Path:
    return repository_root() / RESULTS_DIRECTORY / "product_exact_odt_clone_oracle.json"


def clone_oracle_junit_path() -> Path:
    return repository_root() / RESULTS_DIRECTORY / "product_exact_odt_clone_oracle.junit.xml"


def lane_test_result_path() -> Path:
    return repository_root() / RESULTS_DIRECTORY / "product_exact_odt_lane_tests.json"


def lane_test_junit_path() -> Path:
    return repository_root() / RESULTS_DIRECTORY / "product_exact_odt_lane_tests.junit.xml"


def full_rank_progress_path() -> Path:
    return repository_root() / RESULTS_DIRECTORY / (
        "product_exact_odt_algorithm3_full_rank_complete_before_compression.json"
    )


def full_result_path() -> Path:
    return repository_root() / RESULTS_DIRECTORY / "product_exact_odt_full_with_compression.json"


def composite_result_path() -> Path:
    return repository_root() / RESULTS_DIRECTORY / "product_capable_global_odt_attestation.json"


def capability_aggregate_path() -> Path:
    return CAPABILITY_STAGE_ROOT / CAPABILITY_RUN_ROOT / "results" / (
        "product_pade_rational_three_seed_aggregate.json"
    )


def capability_gate_path() -> Path:
    return CAPABILITY_STAGE_ROOT / CAPABILITY_RUN_ROOT / "gates" / (
        "product_pade_rational_seed0_capability_gate.json"
    )
