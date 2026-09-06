"""Authenticate the canonical ODT smoke artifacts in a fresh CPU process.

This node is the release gate for the training array.  It deliberately reloads
the frozen checkpoint and stored matrix artifact instead of trusting producer
booleans or the GPU process that created them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from athena.summarize_svhn_canonical_odt import (
    EXPECTED_ALGEBRAIC_GATES,
    EXPECTED_CAPABILITY_GATES,
    EXPECTED_DATASET_FILES,
    PREFETCH_SCHEMA,
    SCHEMA,
    SMOKE_VALIDATION_SCHEMA,
    digest,
    export_checkpoint_homogeneous,
    regenerate_gauge_replay_sample,
    rehash_official_svhn,
    require_exact_true_gates,
    validate_gauge_matrix_artifact,
    validate_launch_binding,
    validate_protocol_and_measurements,
    validate_seed_provenance,
)
from athena.verify_canonical_odt_freeze import (
    SMOKE_REQUIRED_AUTHENTICATION_IDENTITIES,
    UNIT_CERTIFICATE_SCHEMA,
    reject_symlinks_in_directory,
    require_confined_path,
    require_physical_root,
    validate_required_authentication_decisions,
    verify as verify_source_freeze,
)


def _write_json_exclusive(path: Path, value: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _exact_input_inventory(results: Path, test_log: Path) -> list[str]:
    expected = {
        "launch.json",
        "canonical_odt_unit_certificate.json",
        "prefetch.json",
        "smoke.json",
        "smoke.pt",
        "smoke_gauge_matrices.npz",
        test_log.name,
    }
    children = list(results.iterdir())
    actual = {path.name for path in children}
    if actual != expected or len(children) != len(expected):
        raise RuntimeError(
            "smoke-validation input inventory mismatch: "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )
    for path in children:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"smoke-validation input is not a physical file: {path}")
    return sorted(expected)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = require_physical_root(args.root)
    results = reject_symlinks_in_directory(
        root, root / "results", label="results directory"
    )
    output = require_confined_path(
        root, args.output, label="smoke-validation receipt", allow_missing_leaf=True
    )
    if output != results / "smoke_validation.json":
        raise RuntimeError("smoke-validation output path differs from the canonical path")
    if output.exists() or output.is_symlink():
        raise RuntimeError("smoke-validation receipt already exists")
    if (
        os.environ.get("SLURM_JOB_PARTITION") != "compute"
        or os.environ.get("SLURM_ARRAY_JOB_ID") is not None
        or os.environ.get("SLURM_ARRAY_TASK_ID") is not None
    ):
        raise RuntimeError("smoke validation must run as a non-array Athena compute job")

    manifest_path = require_confined_path(
        root, root / "source_manifest.sha256", label="source manifest", kind="file"
    )
    frozen_entries = verify_source_freeze(root, manifest_path)
    manifest_text = manifest_path.read_text()
    manifest_sha256 = digest(manifest_path)
    launch_path = require_confined_path(
        root, results / "launch.json", label="launch record", kind="file"
    )
    launch = json.loads(launch_path.read_text())
    jobs = validate_launch_binding(
        launch,
        project_root=root,
        manifest_sha256=manifest_sha256,
        current_summary_job_id=os.environ.get("SLURM_JOB_ID"),
        current_node="smoke_validate",
    )

    certificate_path = require_confined_path(
        root,
        results / "canonical_odt_unit_certificate.json",
        label="unit-test certificate",
        kind="file",
    )
    certificate = json.loads(certificate_path.read_text())
    test_log = require_confined_path(
        root, Path(certificate.get("test_log_path", "")), label="unit-test log", kind="file"
    )
    if (
        certificate.get("schema") != UNIT_CERTIFICATE_SCHEMA
        or certificate.get("tests_passed") is not True
        or certificate.get("source_manifest_sha256") != manifest_sha256
        or certificate.get("source_entries") != frozen_entries
        or certificate.get("slurm_job_id") != jobs["tests"]
        or test_log.parent != results
        or certificate.get("test_log_sha256") != digest(test_log)
    ):
        raise RuntimeError("unit-test certificate does not authenticate the frozen source")

    prefetch_path = require_confined_path(
        root, results / "prefetch.json", label="prefetch receipt", kind="file"
    )
    prefetch = json.loads(prefetch_path.read_text())
    if prefetch != {
        "schema": PREFETCH_SCHEMA,
        "status": "complete",
        "source_manifest_sha256": manifest_sha256,
        "dataset_root": launch["data_root"],
        "dataset_files": EXPECTED_DATASET_FILES,
        "slurm_job_id": jobs["prefetch"],
    }:
        raise RuntimeError("prefetch receipt is not bound to the launch DAG")

    initial_inventory = _exact_input_inventory(results, test_log)
    dataset_root = Path(launch["data_root"]).resolve()
    dataset_files = rehash_official_svhn(dataset_root)
    replay_sample = regenerate_gauge_replay_sample(dataset_root)
    smoke_path = require_confined_path(
        root, results / "smoke.json", label="smoke result", kind="file"
    )
    smoke = json.loads(smoke_path.read_text())
    if (
        smoke.get("schema") != SCHEMA
        or smoke.get("status") != "complete"
        or smoke.get("seed") != 9173
        or smoke.get("protocol", {}).get("epochs") != 20
        or smoke.get("protocol", {}).get("gauge_trials") != 50
        or smoke.get("protocol", {}).get("minimum_full_model_accuracy") != 0.70
    ):
        raise RuntimeError("smoke protocol did not complete exactly as frozen")
    require_exact_true_gates(smoke, "algebraic_gates", EXPECTED_ALGEBRAIC_GATES)
    require_exact_true_gates(smoke, "capability_gates", EXPECTED_CAPABILITY_GATES)
    validate_protocol_and_measurements(smoke, smoke=True)
    validate_seed_provenance(
        smoke,
        project_root=root,
        manifest_path=manifest_path,
        manifest_text=manifest_text,
        manifest_sha256=manifest_sha256,
        frozen_entries=frozen_entries,
    )
    runtime = smoke["provenance"]["runtime"]
    if (
        runtime.get("slurm_job_id") != jobs["smoke"]
        or runtime.get("slurm_array_job_id") is not None
        or runtime.get("slurm_array_task_id") is not None
    ):
        raise RuntimeError("smoke result is not bound to its non-array DAG node")
    if smoke["provenance"].get("dataset_root") != str(dataset_root):
        raise RuntimeError("smoke dataset root differs from the launch binding")

    gauge_path, authentication = validate_gauge_matrix_artifact(
        smoke,
        project_root=root,
        expected_path=results / "smoke_gauge_matrices.npz",
        expected_raw_network=export_checkpoint_homogeneous(smoke, project_root=root),
        expected_replay_sample=replay_sample,
        include_authentication_details=True,
    )
    validate_required_authentication_decisions(
        authentication,
        expected_identities=SMOKE_REQUIRED_AUTHENTICATION_IDENTITIES,
    )
    artifact_hashes = {
        "launch_json": digest(launch_path),
        "unit_certificate_json": digest(certificate_path),
        "unit_test_log": digest(test_log),
        "prefetch_json": digest(prefetch_path),
        "smoke_json": digest(smoke_path),
        "smoke_checkpoint_pt": digest(smoke_path.with_suffix(".pt")),
        "smoke_gauge_matrices_npz": digest(gauge_path),
    }
    if verify_source_freeze(root, manifest_path) != frozen_entries:
        raise RuntimeError("source freeze changed during smoke validation")
    if rehash_official_svhn(dataset_root) != dataset_files:
        raise RuntimeError("official SVHN bytes changed during smoke validation")
    if _exact_input_inventory(results, test_log) != initial_inventory:
        raise RuntimeError("smoke-validation input inventory changed")
    if artifact_hashes != {
        "launch_json": digest(launch_path),
        "unit_certificate_json": digest(certificate_path),
        "unit_test_log": digest(test_log),
        "prefetch_json": digest(prefetch_path),
        "smoke_json": digest(smoke_path),
        "smoke_checkpoint_pt": digest(smoke_path.with_suffix(".pt")),
        "smoke_gauge_matrices_npz": digest(gauge_path),
    }:
        raise RuntimeError("smoke-validation input bytes changed")

    receipt = {
        "schema": SMOKE_VALIDATION_SCHEMA,
        "status": "complete",
        "source_manifest_sha256": manifest_sha256,
        "artifact_sha256": artifact_hashes,
        "input_inventory": initial_inventory,
        "dataset_files": dataset_files,
        "producer_slurm_job_id": jobs["smoke"],
        "validator_slurm_job_id": jobs["smoke_validate"],
        "required_projector_three_way_authentication": authentication,
    }
    _write_json_exclusive(output, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
