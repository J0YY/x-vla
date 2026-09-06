"""Adversarial protocol tests for the frozen canonical SVHN ODT pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import pytest
import numpy as np
import torch

import athena.run_svhn_canonical_odt as runner
import athena.launch_svhn_canonical_odt as launcher
import athena.summarize_svhn_canonical_odt as summary
import athena.validate_svhn_canonical_odt_smoke as smoke_validator
import athena.verify_canonical_odt_freeze as freeze


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_text(entries: dict[str, str]) -> str:
    return "".join(f"{value}  {key}\n" for key, value in sorted(entries.items()))


def _provenance_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project_root = tmp_path / "frozen"
    results = project_root / "results"
    results.mkdir(parents=True)
    source = project_root / "source.py"
    source.write_text("SOURCE = True\n")
    source_hash = _sha(source)
    entries = {"source.py": source_hash}
    manifest_path = project_root / "source_manifest.sha256"
    manifest_text = _manifest_text(entries)
    manifest_path.write_text(manifest_text)
    checkpoint = results / "seed_0.pt"
    checkpoint.write_bytes(b"checkpoint")
    dataset_root = tmp_path / "svhn"
    dataset_root.mkdir()
    monkeypatch.setattr(summary, "RESULT_SOURCE_PATHS", ("source.py",))
    fingerprint_payload = {
        "python": "3.11.0",
        "python_executable_invoked": "/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python",
        "python_executable_resolved": "/work/joy/safesae-openvla/bin/python3.10",
        "platform": "Linux-test",
        "torch": "test",
        "torchvision": "test",
        "numpy": "test",
        "cuda": "test",
        "cudnn": 1,
        "distributions": [["numpy", "test"], ["torch", "test"]],
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            {key: value for key, value in fingerprint_payload.items() if key != "platform"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    record = {
        "seed": 0,
        "provenance": {
            "source_sha256": entries.copy(),
            "frozen_source_manifest": {
                "path": str(manifest_path),
                "sha256": _sha(manifest_path),
                "text": manifest_text,
            },
            "dataset_root": str(dataset_root),
            "dataset_files": summary.EXPECTED_DATASET_FILES,
            "checkpoint": {
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "sha256": _sha(checkpoint),
            },
            "runtime": {
                **fingerprint_payload,
                "fingerprint_sha256": fingerprint,
                "gpu": {"name": "NVIDIA RTX A6000", "capability": [8, 6], "total_memory_bytes": 1},
                "scheduler": {
                    "job_id": "2", "array_job_id": "1", "array_task_id": "0", "partition": "gpu",
                    "constraint": None, "node_list": "node",
                },
                "slurm_job_id": "2",
                "slurm_array_job_id": "1",
                "slurm_array_task_id": "0",
                "deterministic_algorithms": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "float32_matmul_precision": "highest",
                "cublas_workspace_config": ":4096:8",
            },
        },
    }
    return project_root, manifest_path, manifest_text, entries, checkpoint, record


def _validate(record, project_root, manifest_path, manifest_text, entries):
    summary.validate_seed_provenance(
        record,
        project_root=project_root,
        manifest_path=manifest_path,
        manifest_text=manifest_text,
        manifest_sha256=_sha(manifest_path),
        frozen_entries=entries,
    )


def test_runner_and_summary_gate_inventories_are_identical():
    v3r1_algebraic_gates = {
        "sampled_logit_reconstruction_le_1e-8",
        "factorization_relative_le_1e-10",
        "row_isometry_le_1e-10",
        "gauge_sampled_single_bond_logit_replay_relative_le_1e-8",
        "gauge_singular_extrema_relative_le_1e-12",
        "gauge_condition_relative_le_1e-12",
        "gauge_nonsymmetry_relative_gt_1e-6",
        "gauge_coefficient_replay_relative_le_1e-8",
        "gauge_transport_orthogonality_le_1e-10",
        "gauge_gram_covariance_relative_le_1e-8",
        "gauge_spectrum_relative_le_1e-8",
        "gauge_projector_certificate_coverage_complete",
        "gauge_required_projector_overlap_support_defect_le_1e-8",
        "gauge_required_projectors_numerically_certifiable",
        "gauge_required_projectors_pass_nonvacuous_bound",
        "selected_canonical_eigengap_resolved",
        "all_bond_uniform_schedule_exact",
        "all_bond_trace_totals_relative_le_1e-8",
        "all_bond_full_rank_coefficient_reconstruction_le_1e-8",
        "all_bond_selected_canonical_eigengaps_resolved",
        "all_bond_coefficient_errors_within_hsvd_bounds_plus_declared_numerical_slack",
    }
    v3r2_additions = {
        "selected_canonical_roundoff_only_prefilter_pass",
        "all_bond_selected_canonical_roundoff_only_prefilter_pass",
    }
    assert runner.SCHEMA == "xvla-canonical-tree-odt-svhn-extension-v3r5"
    assert summary.SCHEMA == runner.SCHEMA
    assert summary.PREFETCH_SCHEMA == runner.PREFETCH_SCHEMA
    assert summary.SUMMARY_SCHEMA == "xvla-canonical-tree-odt-svhn-summary-v3r5"
    assert summary.SMOKE_VALIDATION_SCHEMA == "xvla-canonical-tree-odt-smoke-validation-v1"
    assert freeze.UNIT_CERTIFICATE_SCHEMA == "xvla-canonical-odt-unit-certificate-v4"
    assert launcher.LAUNCH_SCHEMA == "xvla-canonical-odt-athena-launch-v4"
    assert launcher.DAG_SCHEMA == "xvla-canonical-odt-athena-dag-v4"
    assert runner.EXPECTED_ALGEBRAIC_GATES == summary.EXPECTED_ALGEBRAIC_GATES
    assert runner.EXPECTED_ALGEBRAIC_GATES == v3r1_algebraic_gates | v3r2_additions
    assert runner.EXPECTED_CAPABILITY_GATES == summary.EXPECTED_CAPABILITY_GATES
    assert runner.SVHN_FILES == summary.EXPECTED_DATASET_FILES
    assert list(runner.PREDECESSOR_FAILURES) == summary.EXPECTED_PREDECESSOR_FAILURES
    assert summary.RAW_GAUGE_STREAM_RELATIVE_TOLERANCE == 1e-12
    assert summary.INDEPENDENT_MATRIX_RECONSTRUCTION_RELATIVE_TOLERANCE == 5e-8
    assert summary.NONREQUIRED_DIAGNOSTIC_PRIMITIVE_ABSOLUTE_TOLERANCE == 1e-8
    assert None not in runner.EXPECTED_ALGEBRAIC_GATES
    assert {
        path for path in freeze.RESULT_SOURCE_PATHS if not path.startswith("xvla/")
    } <= freeze.FIXED_SOURCE_PATHS
    assert summary.EXPECTED_SUMMARY_GATES == {
        "three_complete_seeds",
        "identical_source_hashes",
        "identical_dataset_hashes",
        "sampled_logit_reconstruction_le_1e-8",
        "factorization_relative_le_1e-10",
        "row_isometry_le_1e-10",
        "gauge_sampled_single_bond_logit_replay_relative_le_1e-8",
        "gauge_singular_extrema_relative_le_1e-12",
        "gauge_condition_relative_le_1e-12",
        "gauge_nonsymmetry_relative_gt_1e-6",
        "gauge_spectrum_relative_le_1e-8",
        "gauge_coefficient_replay_relative_le_1e-8",
        "gauge_projector_certificate_coverage_complete",
        "gauge_gram_covariance_relative_le_1e-8",
        "gauge_transport_orthogonality_le_1e-10",
        "gauge_required_projector_overlap_support_defect_le_1e-8",
        "gauge_required_projectors_numerically_certifiable",
        "gauge_required_projectors_pass_nonvacuous_bound",
        "selected_canonical_eigengap_resolved",
        "selected_canonical_roundoff_only_prefilter_pass",
        "all_seed_gate_inventories_exact_and_true",
        "all_seed_manifests_match_current_freeze",
        "official_svhn_hashes_match",
        "checkpoint_files_rehashed",
        "gauge_npz_artifacts_independently_recomputed",
        "identical_environment_fingerprints",
        "module_accuracy_floor_passed",
        "trained_checkpoint_50_gauge_preflight_passed",
        "all_bond_uniform_schedule_exact",
        "all_bond_trace_totals_relative_le_1e-8",
        "all_bond_full_rank_coefficient_reconstruction_le_1e-8",
        "all_bond_selected_canonical_eigengaps_resolved",
        "all_bond_selected_canonical_roundoff_only_prefilter_pass",
        "all_bond_coefficient_errors_within_hsvd_bounds_plus_declared_numerical_slack",
    }


def test_smoke_gauge_rank_inventory_is_nonvacuous_when_selection_is_full_rank():
    required, diagnostic = runner.required_gauge_ranks(
        (33, 33, 33, 33),
        selected_single_bond=1,
        selected_single_rank=33,
        bond_dims=(33, 33, 33, 33),
        smoke_diagnostic_rank=1,
    )
    assert diagnostic == 1
    assert required == ((1, 33), (1, 33), (1, 33), (1, 33))
    assert sum(rank < dimension for ranks, dimension in zip(required, (33,) * 4) for rank in ranks) == 4


def test_reported_seed_gauge_rank_inventory_does_not_add_smoke_diagnostic():
    required, diagnostic = runner.required_gauge_ranks(
        (33, 24, 17, 10),
        selected_single_bond=1,
        selected_single_rank=14,
        bond_dims=(33, 33, 33, 33),
        smoke_diagnostic_rank=None,
    )
    assert diagnostic is None
    assert required == ((33,), (14, 24), (17,), (10,))


def test_terminal_failure_preserves_authenticated_smoke_root_cause(
    tmp_path, monkeypatch
):
    results = tmp_path / "results"
    results.mkdir()
    manifest_path = tmp_path / "source_manifest.sha256"
    manifest_path.write_text("frozen manifest\n")
    manifest_sha = _sha(manifest_path)
    data_root = tmp_path / "svhn"
    data_root.mkdir()
    (results / "launch.json").write_text(json.dumps({"data_root": str(data_root)}))
    monkeypatch.setattr(summary, "verify_source_freeze", lambda root, path: {})
    monkeypatch.setattr(
        summary,
        "validate_launch_binding",
        lambda launch, **kwargs: {"smoke": "42"},
    )
    validated = []
    monkeypatch.setattr(
        summary,
        "validate_seed_provenance",
        lambda record, **kwargs: validated.append("provenance"),
    )
    monkeypatch.setattr(summary, "rehash_official_svhn", lambda root: {})
    monkeypatch.setattr(summary, "regenerate_gauge_replay_sample", lambda root: object())
    monkeypatch.setattr(summary, "export_checkpoint_homogeneous", lambda record, **kwargs: object())
    matrix_authentication = {"fixture": "three-way"}
    monkeypatch.setattr(
        summary,
        "validate_gauge_matrix_artifact",
        lambda record, **kwargs: (
            kwargs["expected_path"], matrix_authentication
        ),
    )
    monkeypatch.setattr(
        summary,
        "validate_protocol_and_measurements",
        lambda record, **kwargs: (
            validated.append(("measurements", kwargs)),
            record["algebraic_gates"],
            record["capability_gates"],
        )[1:],
    )
    smoke_path = results / "smoke.json"
    smoke_path.write_text(json.dumps({
        "schema": runner.SCHEMA,
        "status": "gate_failed",
        "seed": 9173,
        "provenance": {
            "frozen_source_manifest": {"sha256": manifest_sha},
            "runtime": {"slurm_job_id": "42"},
            "dataset_root": str(data_root),
        },
        "protocol": {"smoke_nontrivial_diagnostic_rank": 1},
        "algebraic_gates": {
            key: key != "gauge_required_projectors_numerically_certifiable"
            for key in summary.EXPECTED_ALGEBRAIC_GATES
        },
        "capability_gates": {
            key: True for key in summary.EXPECTED_CAPABILITY_GATES
        },
    }))
    root_cause = summary.authenticated_smoke_failure(tmp_path, manifest_sha)
    assert root_cause == {
        "stage": "smoke",
        "path": str(smoke_path),
        "sha256": _sha(smoke_path),
        "status": "gate_failed",
        "false_algebraic_gates": [
            "gauge_required_projectors_numerically_certifiable"
        ],
        "false_capability_gates": [],
        "required_projector_three_way_authentication": matrix_authentication,
    }
    assert validated == [
        "provenance",
        (
            "measurements",
            {
                "smoke": True,
                "expected_status": "gate_failed",
                "require_all_gates": False,
            },
        ),
    ]
    with pytest.raises(RuntimeError, match="source manifest changed"):
        summary.authenticated_smoke_failure(tmp_path, "b" * 64)


def test_exact_seed_result_paths_and_artifact_hash_inventory(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    seed_paths = []
    for seed in range(3):
        path = results / f"seed_{seed}.json"
        path.write_text(json.dumps({"seed": seed}))
        seed_paths.append(path)
    assert summary.exact_seed_result_paths(results) == seed_paths
    records = [{"seed": seed} for seed in range(3)]
    summary.require_seed_file_binding(records, seed_paths)
    with pytest.raises(RuntimeError, match="content does not match"):
        summary.require_seed_file_binding([records[1], records[0], records[2]], seed_paths)

    unexpected = results / "seed_3.json"
    unexpected.write_text("{}")
    with pytest.raises(RuntimeError, match="filenames must be exactly"):
        summary.exact_seed_result_paths(results)
    unexpected.unlink()

    launch = results / "launch.json"
    certificate = results / "certificate.json"
    test_log = results / "tests.log"
    smoke = results / "smoke.json"
    prefetch = results / "prefetch.json"
    smoke_validation = results / "smoke_validation.json"
    smoke_gauge = results / "smoke_gauge_matrices.npz"
    seed_gauges = []
    seed_checkpoints = []
    for seed in range(3):
        gauge = results / f"seed_{seed}_gauge_matrices.npz"
        gauge.write_bytes(f"gauge-{seed}".encode())
        seed_gauges.append(gauge)
        checkpoint = results / f"seed_{seed}.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        seed_checkpoints.append(checkpoint)
    for index, path in enumerate((
        launch, certificate, test_log, prefetch, smoke_validation, smoke, smoke_gauge
    )):
        path.write_text(f"artifact-{index}")
    smoke.with_suffix(".pt").write_bytes(b"smoke-checkpoint")
    manifest = summary.artifact_hash_manifest(
        launch_path=launch,
        certificate_path=certificate,
        test_log=test_log,
        prefetch_path=prefetch,
        smoke_validation_path=smoke_validation,
        smoke_path=smoke,
        smoke_gauge_path=smoke_gauge,
        records=records,
        seed_paths=seed_paths,
        seed_gauge_paths=seed_gauges,
    )
    assert manifest == {
        "launch_json": _sha(launch),
        "unit_certificate_json": _sha(certificate),
        "unit_test_log": _sha(test_log),
        "prefetch_json": _sha(prefetch),
        "smoke_validation_json": _sha(smoke_validation),
        "smoke_json": _sha(smoke),
        "smoke_checkpoint_pt": _sha(smoke.with_suffix(".pt")),
        "smoke_gauge_matrices_npz": _sha(smoke_gauge),
        "seed_json_by_seed": {str(seed): _sha(path) for seed, path in enumerate(seed_paths)},
        "gauge_matrices_npz_by_seed": {
            str(seed): _sha(path) for seed, path in enumerate(seed_gauges)
        },
        "checkpoint_pt_by_seed": {
            str(seed): _sha(path) for seed, path in enumerate(seed_checkpoints)
        },
    }
    assert summary.EXPECTED_EMPIRICAL_GATES == {
        "all_bond_selected_confirmation_noninferior_in_all_seeds",
        "all_bond_selected_odt_gt_local_with_resolved_boundaries_in_all_seeds",
        "all_bond_selected_haar_randomization_p_le_0p05_in_all_seeds",
    }


def _pre_summary_inventory_fixture(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    results = tmp_path / "results"
    results.mkdir()
    test_log = results / "canonical_odt_tests.log"
    expected_names = {
        "launch.json",
        "canonical_odt_unit_certificate.json",
        "prefetch.json",
        "smoke_validation.json",
        "smoke.json",
        "smoke.pt",
        "smoke_gauge_matrices.npz",
        test_log.name,
        *(f"seed_{seed}.json" for seed in range(3)),
        *(f"seed_{seed}.pt" for seed in range(3)),
        *(f"seed_{seed}_gauge_matrices.npz" for seed in range(3)),
    }
    for name in expected_names:
        (results / name).write_bytes(name.encode())
    return results, test_log, sorted(expected_names)


def test_pre_summary_result_inventory_accepts_exact_physical_files(tmp_path):
    results, test_log, expected_names = _pre_summary_inventory_fixture(tmp_path)
    assert summary.validate_pre_summary_result_inventory(results, test_log) == expected_names


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_pre_summary_result_inventory_rejects_extra_or_missing_files(tmp_path, mutation):
    results, test_log, _ = _pre_summary_inventory_fixture(tmp_path)
    if mutation == "extra":
        (results / "unplanned.json").write_text("{}")
    else:
        (results / "seed_2.pt").unlink()
    with pytest.raises(RuntimeError, match="artifact inventory mismatch"):
        summary.validate_pre_summary_result_inventory(results, test_log)


@pytest.mark.parametrize("replacement", ["symlink", "directory"])
def test_pre_summary_result_inventory_rejects_symlink_or_nonfile(tmp_path, replacement):
    results, test_log, _ = _pre_summary_inventory_fixture(tmp_path)
    target = results / "seed_1.pt"
    target.unlink()
    if replacement == "symlink":
        outside = tmp_path / "outside.pt"
        outside.write_bytes(b"checkpoint")
        target.symlink_to(outside)
    else:
        target.mkdir()
    with pytest.raises(RuntimeError, match="not a physical regular file"):
        summary.validate_pre_summary_result_inventory(results, test_log)


def test_confined_path_rejects_symlinked_leaf_and_parent(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    real_file = root / "real.py"
    real_file.write_text("pass\n")
    linked_file = root / "linked.py"
    linked_file.symlink_to(real_file)
    with pytest.raises(RuntimeError, match="symlink"):
        freeze.require_confined_path(root, linked_file, label="source", kind="file")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "escaped.py").write_text("pass\n")
    linked_directory = root / "linked_directory"
    linked_directory.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        freeze.require_confined_path(
            root, linked_directory / "escaped.py", label="source", kind="file"
        )


@pytest.mark.parametrize("directory_name", ["logs", "results"])
def test_symlinked_artifact_directories_are_rejected(tmp_path, directory_name):
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / directory_name).symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        freeze.reject_symlinks_in_directory(
            root, root / directory_name, label=f"{directory_name} directory"
        )


def _launch_fixture(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "run"
    root.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    record = launcher.make_launch_record(
        root=root.resolve(),
        data_root=data_root.resolve(),
        manifest_sha256="a" * 64,
        job_ids={
            "tests": "101",
            "prefetch": "102",
            "smoke": "103",
            "smoke_validate": "104",
            "train_array": "105",
            "summary": "106",
        },
    )
    return root, record


def _launcher_main_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create the smallest physical tree while mocking only Athena's boundary."""

    root = tmp_path / "run"
    (root / "xvla").mkdir(parents=True)
    (root / "xvla" / "module.py").write_text("VALUE = 1\n")
    data_root = tmp_path / "svhn"
    data_root.mkdir()
    monkeypatch.setattr(launcher, "FIXED_SOURCE_PATHS", set())
    verified = []

    def verify(project_root, manifest_path):
        verified.append((project_root, manifest_path, manifest_path.read_text()))

    monkeypatch.setattr(launcher, "verify", verify)
    monkeypatch.setattr(launcher, "verify_import_closure", lambda project_root: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_svhn_canonical_odt.py",
            "--run-root",
            str(root),
            "--data-root",
            str(data_root),
        ],
    )
    return root, data_root, verified


def test_launcher_main_success_writes_complete_record_before_releasing_hold(
    tmp_path, monkeypatch
):
    root, data_root, verified = _launcher_main_fixture(tmp_path, monkeypatch)
    submitted_commands = []
    events = []
    job_ids = iter(("101", "102", "103", "104", "105", "106"))

    def check_output(command, text):
        submitted_commands.append(command)
        job_id = next(job_ids)
        events.append(("submit", job_id))
        return f"{job_id};athena\n"

    def check_call(command):
        launch_path = root / "results" / "launch.json"
        assert launch_path.is_file()
        assert json.loads(launch_path.read_text())["dag"]["nodes"]["summary"][
            "job_id"
        ] == "106"
        events.append(("release", tuple(command)))

    cancellations = []
    monkeypatch.setattr(launcher.subprocess, "check_output", check_output)
    monkeypatch.setattr(launcher.subprocess, "check_call", check_call)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda command, check: cancellations.append((command, check)),
    )

    launcher.main()

    assert len(verified) == 1
    assert verified[0][0] == root.resolve()
    assert verified[0][1] == root / "source_manifest.sha256"
    assert "xvla/module.py" in verified[0][2]
    assert [event[0] for event in events] == ["submit"] * 6 + ["release"]
    assert events[-1] == ("release", ("scontrol", "release", "101"))
    assert "--hold" in submitted_commands[0]
    assert all("--hold" not in command for command in submitted_commands[1:])
    assert all("--kill-on-invalid-dep=yes" in command for command in submitted_commands)
    for index, upstream in enumerate(("101", "102", "103"), start=1):
        assert f"--dependency=afterany:{upstream}" in submitted_commands[index]
    assert "--dependency=afterok:104" in submitted_commands[4]
    assert "--dependency=afternotok:104?afterany:105" in submitted_commands[5]
    assert "--array=0-2" in submitted_commands[4]
    launch = json.loads((root / "results" / "launch.json").read_text())
    assert launch == launcher.make_launch_record(
        root=root.resolve(),
        data_root=data_root.resolve(),
        manifest_sha256=_sha(root / "source_manifest.sha256"),
        job_ids={
            "tests": "101",
            "prefetch": "102",
            "smoke": "103",
            "smoke_validate": "104",
            "train_array": "105",
            "summary": "106",
        },
    )
    assert not (tmp_path / "run.launch_failure.json").exists()
    assert cancellations == []
    assert (root / "source_manifest.sha256").stat().st_mode & 0o222 == 0
    assert (root / "results" / "launch.json").stat().st_mode & 0o222 == 0


def test_import_closure_uses_exact_athena_python_and_frozen_root(
    tmp_path, monkeypatch
):
    root = tmp_path.resolve()
    calls = []
    monkeypatch.setattr(launcher.sys, "executable", launcher.ATHENA_PYTHON)
    monkeypatch.setattr(
        launcher.subprocess,
        "check_call",
        lambda command, cwd, env: calls.append((command, cwd, env)),
    )
    launcher.verify_import_closure(root)
    assert len(calls) == 1
    command, cwd, environment = calls[0]
    assert command[0] == launcher.ATHENA_PYTHON
    assert command[1:3] == ["-B", "-c"]
    assert "import xvla" in command[3]
    assert "athena.run_svhn_canonical_odt" in command[3]
    assert "athena.validate_svhn_canonical_odt_smoke" in command[3]
    assert "athena.summarize_svhn_canonical_odt" in command[3]
    assert "athena.verify_canonical_odt_freeze" in command[3]
    assert cwd == root
    assert environment["PYTHONPATH"] == str(root)
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_import_closure_rejects_nonfrozen_python(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.sys, "executable", "/usr/bin/python3")
    with pytest.raises(RuntimeError, match="frozen Athena Python"):
        launcher.verify_import_closure(tmp_path)


def _smoke_validator_main_fixture(tmp_path, monkeypatch):
    root = tmp_path / "run"
    results = root / "results"
    data_root = tmp_path / "svhn"
    results.mkdir(parents=True)
    data_root.mkdir()
    manifest = root / "source_manifest.sha256"
    manifest.write_text("frozen\n")
    manifest_sha = _sha(manifest)
    launch = results / "launch.json"
    launch.write_text(json.dumps({"data_root": str(data_root)}))
    test_log = results / "canonical_odt_tests.log"
    test_log.write_text("passed\n")
    entries = {"source.py": "a" * 64}
    certificate = results / "canonical_odt_unit_certificate.json"
    certificate.write_text(json.dumps({
        "schema": freeze.UNIT_CERTIFICATE_SCHEMA,
        "tests_passed": True,
        "source_manifest_sha256": manifest_sha,
        "source_entries": entries,
        "slurm_job_id": "41",
        "test_log_path": str(test_log),
        "test_log_sha256": _sha(test_log),
    }))
    (results / "prefetch.json").write_text(json.dumps({
        "schema": smoke_validator.PREFETCH_SCHEMA,
        "status": "complete",
        "source_manifest_sha256": manifest_sha,
        "dataset_root": str(data_root),
        "dataset_files": smoke_validator.EXPECTED_DATASET_FILES,
        "slurm_job_id": "42",
    }))
    smoke = {
        "schema": smoke_validator.SCHEMA,
        "status": "complete",
        "seed": 9173,
        "protocol": {"epochs": 20, "gauge_trials": 50, "minimum_full_model_accuracy": 0.70},
        "algebraic_gates": {
            key: True for key in smoke_validator.EXPECTED_ALGEBRAIC_GATES
        },
        "capability_gates": {
            key: True for key in smoke_validator.EXPECTED_CAPABILITY_GATES
        },
        "provenance": {
            "runtime": {"slurm_job_id": "43", "slurm_array_job_id": None, "slurm_array_task_id": None},
            "dataset_root": str(data_root),
        },
    }
    (results / "smoke.json").write_text(json.dumps(smoke))
    (results / "smoke.pt").write_bytes(b"checkpoint")
    gauge = results / "smoke_gauge_matrices.npz"
    gauge.write_bytes(b"matrices")
    jobs = {"tests": "41", "prefetch": "42", "smoke": "43", "smoke_validate": "44"}
    authentication = {"fixture": "authentication"}
    calls = []
    monkeypatch.setenv("SLURM_JOB_PARTITION", "compute")
    monkeypatch.setenv("SLURM_JOB_ID", "44")
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.setattr(smoke_validator, "verify_source_freeze", lambda *args: entries)
    monkeypatch.setattr(smoke_validator, "validate_launch_binding", lambda *args, **kwargs: jobs)
    monkeypatch.setattr(smoke_validator, "rehash_official_svhn", lambda *args: smoke_validator.EXPECTED_DATASET_FILES)
    monkeypatch.setattr(smoke_validator, "regenerate_gauge_replay_sample", lambda *args: object())
    monkeypatch.setattr(smoke_validator, "export_checkpoint_homogeneous", lambda *args, **kwargs: object())
    monkeypatch.setattr(smoke_validator, "require_exact_true_gates", lambda *args: calls.append("gates"))
    monkeypatch.setattr(smoke_validator, "validate_protocol_and_measurements", lambda *args, **kwargs: calls.append("protocol"))
    monkeypatch.setattr(smoke_validator, "validate_seed_provenance", lambda *args, **kwargs: calls.append("provenance"))
    monkeypatch.setattr(
        smoke_validator,
        "validate_gauge_matrix_artifact",
        lambda *args, **kwargs: (gauge, authentication),
    )
    monkeypatch.setattr(
        smoke_validator,
        "validate_required_authentication_decisions",
        lambda value, *, expected_identities: calls.append(
            ("authentication", value, expected_identities)
        ) or list(expected_identities),
    )
    output = results / "smoke_validation.json"
    monkeypatch.setattr(sys, "argv", [
        "validate_svhn_canonical_odt_smoke.py", "--root", str(root),
        "--output", str(output),
    ])
    return root, output, authentication, calls


def test_smoke_validator_main_authenticates_then_writes_read_only_receipt(
    tmp_path, monkeypatch,
):
    _, output, authentication, calls = _smoke_validator_main_fixture(tmp_path, monkeypatch)
    smoke_validator.main()
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "complete"
    assert receipt["required_projector_three_way_authentication"] == authentication
    assert output.stat().st_mode & 0o222 == 0
    assert calls.count("protocol") == 1 and calls.count("provenance") == 1
    auth_call = next(call for call in calls if isinstance(call, tuple))
    assert auth_call == (
        "authentication", authentication,
        freeze.SMOKE_REQUIRED_AUTHENTICATION_IDENTITIES,
    )


def test_smoke_validator_main_writes_nothing_when_authentication_fails(
    tmp_path, monkeypatch,
):
    _, output, _, _ = _smoke_validator_main_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        smoke_validator,
        "validate_required_authentication_decisions",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("tampered decisions")),
    )
    with pytest.raises(RuntimeError, match="tampered decisions"):
        smoke_validator.main()
    assert not output.exists()


def test_smoke_validator_main_writes_nothing_when_inventory_changes(
    tmp_path, monkeypatch,
):
    _, output, _, _ = _smoke_validator_main_fixture(tmp_path, monkeypatch)
    original = smoke_validator._exact_input_inventory
    calls = 0

    def changed_inventory(*args):
        nonlocal calls
        calls += 1
        value = original(*args)
        return value if calls == 1 else [*value, "late-file"]

    monkeypatch.setattr(smoke_validator, "_exact_input_inventory", changed_inventory)
    with pytest.raises(RuntimeError, match="input inventory changed"):
        smoke_validator.main()
    assert not output.exists()


def test_launcher_main_import_closure_failure_is_transactional(tmp_path, monkeypatch):
    root, _, _ = _launcher_main_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        launcher,
        "verify_import_closure",
        lambda project_root: (_ for _ in ()).throw(
            RuntimeError("synthetic import closure failure")
        ),
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda command, text: pytest.fail(
            "import closure failure must precede every submission"
        ),
    )

    with pytest.raises(RuntimeError, match="synthetic import closure failure"):
        launcher.main()

    assert not (root / "source_manifest.sha256").exists()
    assert not (root / "results" / "launch.json").exists()
    receipt_path = tmp_path / "run.launch_failure.json"
    receipt_bytes = receipt_path.read_bytes()
    assert json.loads(receipt_bytes) == {
        "run_root": str(root.resolve()),
        "submitted_jobs": [],
        "error_type": "RuntimeError",
        "error": "synthetic import closure failure",
    }
    with pytest.raises(RuntimeError, match="prior launch failure receipt"):
        launcher.main()
    assert receipt_path.read_bytes() == receipt_bytes


def test_launcher_main_import_closure_cannot_contaminate_output_directories(
    tmp_path, monkeypatch
):
    root, _, _ = _launcher_main_fixture(tmp_path, monkeypatch)

    def contaminate(project_root):
        (project_root / "results" / "unexpected.txt").write_text("contamination")

    monkeypatch.setattr(launcher, "verify_import_closure", contaminate)
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda command, text: pytest.fail(
            "contaminated output directories must prevent every submission"
        ),
    )

    with pytest.raises(RuntimeError, match="contaminated the empty results directory"):
        launcher.main()

    assert not (root / "source_manifest.sha256").exists()
    assert not (root / "results" / "launch.json").exists()
    receipt = json.loads((tmp_path / "run.launch_failure.json").read_text())
    assert receipt["submitted_jobs"] == []
    assert receipt["error_type"] == "RuntimeError"


def test_launcher_main_freeze_verification_failure_rolls_back_manifest_and_receipts(
    tmp_path, monkeypatch
):
    root, _, _ = _launcher_main_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        launcher,
        "verify",
        lambda project_root, manifest_path: (_ for _ in ()).throw(
            RuntimeError("synthetic freeze verification failure")
        ),
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda command, text: pytest.fail("verification failure must precede submission"),
    )

    with pytest.raises(RuntimeError, match="synthetic freeze verification failure"):
        launcher.main()

    assert not (root / "source_manifest.sha256").exists()
    receipt = json.loads((tmp_path / "run.launch_failure.json").read_text())
    assert receipt["submitted_jobs"] == []
    assert receipt["error_type"] == "RuntimeError"
    with pytest.raises(RuntimeError, match="prior launch failure receipt"):
        launcher.main()


def test_launcher_main_first_submission_failure_rolls_back_manifest_and_receipts(
    tmp_path, monkeypatch
):
    root, _, _ = _launcher_main_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda command, text: (_ for _ in ()).throw(
            RuntimeError("synthetic first submission failure")
        ),
    )
    cancellations = []
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda command, check: cancellations.append((command, check)),
    )

    with pytest.raises(RuntimeError, match="synthetic first submission failure"):
        launcher.main()

    assert cancellations == []
    assert not (root / "source_manifest.sha256").exists()
    receipt = json.loads((tmp_path / "run.launch_failure.json").read_text())
    assert receipt["submitted_jobs"] == []
    assert receipt["error_type"] == "RuntimeError"
    with pytest.raises(RuntimeError, match="prior launch failure receipt"):
        launcher.main()


def test_launcher_main_intermediate_submit_failure_cancels_submitted_jobs_and_keeps_receipt(
    tmp_path, monkeypatch
):
    root, _, _ = _launcher_main_fixture(tmp_path, monkeypatch)
    responses = iter(("201;athena\n", "202;athena\n"))
    submission_count = 0

    def check_output(command, text):
        nonlocal submission_count
        submission_count += 1
        if submission_count == 3:
            raise RuntimeError("synthetic smoke submission failure")
        return next(responses)

    cancellations = []
    monkeypatch.setattr(launcher.subprocess, "check_output", check_output)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda command, check: cancellations.append((command, check)),
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "check_call",
        lambda command: pytest.fail("a failed DAG must never be released"),
    )

    with pytest.raises(RuntimeError, match="synthetic smoke submission failure"):
        launcher.main()

    failure_path = tmp_path / "run.launch_failure.json"
    receipt_bytes = failure_path.read_bytes()
    assert json.loads(receipt_bytes) == {
        "run_root": str(root.resolve()),
        "submitted_jobs": ["201", "202"],
        "error_type": "RuntimeError",
        "error": "synthetic smoke submission failure",
    }
    assert cancellations == [(["scancel", "201", "202"], False)]
    assert not (root / "results" / "launch.json").exists()

    with pytest.raises(RuntimeError, match="prior launch failure receipt"):
        launcher.main()
    assert failure_path.read_bytes() == receipt_bytes


def test_launcher_main_atomic_launch_failure_cancels_all_jobs_and_removes_partial_launch(
    tmp_path, monkeypatch
):
    root, _, _ = _launcher_main_fixture(tmp_path, monkeypatch)
    job_ids = iter(("301", "302", "303", "304", "305", "306"))
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda command, text: f"{next(job_ids)};athena\n",
    )
    cancellations = []
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda command, check: cancellations.append((command, check)),
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "check_call",
        lambda command: pytest.fail("a launch-write failure must never release the hold"),
    )
    original_atomic_json = launcher.atomic_json

    def fail_launch_write(path, value):
        if path.name == "launch.json":
            original_atomic_json(path, value)
            raise OSError("synthetic launch write failure")
        original_atomic_json(path, value)

    monkeypatch.setattr(launcher, "atomic_json", fail_launch_write)

    with pytest.raises(OSError, match="synthetic launch write failure"):
        launcher.main()

    assert cancellations == [
        (["scancel", "301", "302", "303", "304", "305", "306"], False)
    ]
    assert not (root / "results" / "launch.json").exists()
    receipt = json.loads((tmp_path / "run.launch_failure.json").read_text())
    assert receipt["submitted_jobs"] == ["301", "302", "303", "304", "305", "306"]
    assert receipt["error_type"] == "OSError"


def test_launcher_main_release_failure_cancels_all_jobs_and_removes_launch(
    tmp_path, monkeypatch
):
    root, _, _ = _launcher_main_fixture(tmp_path, monkeypatch)
    job_ids = iter(("401", "402", "403", "404", "405", "406"))
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda command, text: f"{next(job_ids)};athena\n",
    )
    release_commands = []

    def fail_release(command):
        release_commands.append(command)
        raise launcher.subprocess.CalledProcessError(1, command)

    cancellations = []
    monkeypatch.setattr(launcher.subprocess, "check_call", fail_release)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda command, check: cancellations.append((command, check)),
    )

    with pytest.raises(launcher.subprocess.CalledProcessError):
        launcher.main()

    assert release_commands == [["scontrol", "release", "401"]]
    assert cancellations == [
        (["scancel", "401", "402", "403", "404", "405", "406"], False)
    ]
    assert not (root / "results" / "launch.json").exists()
    receipt = json.loads((tmp_path / "run.launch_failure.json").read_text())
    assert receipt["submitted_jobs"] == ["401", "402", "403", "404", "405", "406"]
    assert receipt["error_type"] == "CalledProcessError"


def test_launch_v4_exact_dag_binds_terminal_aggregator(tmp_path):
    root, record = _launch_fixture(tmp_path)
    assert record["submission_protocol"] == {
        "first_node_submitted_held": True,
        "launch_record_written_before_release": True,
    }
    assert [edge["slurm_dependency"] for edge in record["dag"]["edges"]] == [
        "afterany", "afterany", "afterany", "afterok", "afternotok", "afterany"
    ]
    assert record["dag"]["terminal_dependency_expression"] == (
        "afternotok:104?afterany:105"
    )
    assert summary.validate_launch_binding(
        record,
        project_root=root,
        manifest_sha256="a" * 64,
        current_summary_job_id="106",
    ) == {
        "tests": "101",
        "prefetch": "102",
        "smoke": "103",
        "smoke_validate": "104",
        "train_array": "105",
        "summary": "106",
    }


def test_submit_uses_hold_afterany_and_rejects_malformed_job_ids(tmp_path, monkeypatch):
    commands = []

    def accepted(command, text):
        commands.append(command)
        return "12345;athena\n"

    monkeypatch.setattr(launcher.subprocess, "check_output", accepted)
    assert launcher.submit(
        tmp_path, "stage", "stage.sbatch", ["arg"], dependency="111", hold=True
    ) == "12345"
    assert "--dependency=afterany:111" in commands[0]
    assert "--hold" in commands[0]

    monkeypatch.setattr(
        launcher.subprocess, "check_output", lambda command, text: "not-a-job\n"
    )
    with pytest.raises(RuntimeError, match="invalid Slurm job id"):
        launcher.submit(tmp_path, "stage", "stage.sbatch", [])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record["dag"]["edges"].pop(), "dependency edges"),
        (
            lambda record: record["dag"]["nodes"]["train_array"].update(
                {"array": "0-3"}
            ),
            "specification",
        ),
        (
            lambda record: record["dag"]["nodes"]["summary"].update(
                {"job_id": "104"}
            ),
            "not unique",
        ),
    ],
)
def test_launch_v4_rejects_dag_tampering(tmp_path, mutation, message):
    root, record = _launch_fixture(tmp_path)
    mutation(record)
    with pytest.raises(RuntimeError, match=message):
        summary.validate_launch_binding(
            record,
            project_root=root,
            manifest_sha256="a" * 64,
            current_summary_job_id="106",
        )


def test_launch_v4_rejects_wrong_terminal_job(tmp_path):
    root, record = _launch_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="launch DAG node summary"):
        summary.validate_launch_binding(
            record,
            project_root=root,
            manifest_sha256="a" * 64,
            current_summary_job_id="999",
        )


def test_runner_binds_prefetch_and_array_tasks_to_the_launch_record(tmp_path, monkeypatch):
    root, record = _launch_fixture(tmp_path)
    results = root / "results"
    results.mkdir()
    (results / "launch.json").write_text(json.dumps(record))
    data_root = Path(record["data_root"])

    monkeypatch.setenv("SLURM_JOB_ID", "102")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "compute")
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_CONSTRAINTS", raising=False)
    runner.validate_athena_stage_binding(
        root.resolve(), data_root, "a" * 64, stage="prefetch", seed=None
    )

    monkeypatch.setenv("SLURM_JOB_ID", "106")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "gpu")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "105")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
    runner.validate_athena_stage_binding(
        root.resolve(), data_root, "a" * 64, stage="train_array", seed=2
    )
    with pytest.raises(RuntimeError, match="train-array Slurm binding"):
        runner.validate_athena_stage_binding(
            root.resolve(), data_root, "a" * 64, stage="train_array", seed=1
        )


@pytest.mark.parametrize("condition", [1.0, 10.0, 1000.0])
def test_balanced_gauge_has_frozen_singular_spectrum_and_is_nonsymmetric(condition):
    generator = torch.Generator().manual_seed(7401)
    gauge = runner.make_gauge(17, condition, generator, torch.device("cpu"))
    singular_values = torch.linalg.svdvals(gauge)
    half_log_condition = 0.5 * math.log10(condition)
    expected = torch.pow(
        10.0,
        torch.linspace(
            -half_log_condition,
            half_log_condition,
            17,
            dtype=torch.float64,
        ),
    ).flip(0)
    assert torch.allclose(singular_values, expected, rtol=1e-12, atol=1e-14)
    assert singular_values[0].item() == pytest.approx(condition ** 0.5, rel=1e-12)
    assert singular_values[-1].item() == pytest.approx(condition ** -0.5, rel=1e-12)
    assert (singular_values[0] / singular_values[-1]).item() == pytest.approx(
        condition, rel=1e-12
    )
    assert torch.linalg.matrix_norm(gauge - gauge.T).item() > 1e-6
    assert math.log10(singular_values[0].item()) == pytest.approx(
        -math.log10(singular_values[-1].item()), abs=1e-12
    )
    assert singular_values.log().sum().item() == pytest.approx(0.0, abs=1e-12)


def test_uniform_all_bond_schedule_is_exact_conservative_and_strictly_nested():
    schedule = runner.uniform_all_bond_schedule((33, 33, 33, 33))
    assert [item["target_removal_basis_points"] for item in schedule] == list(
        runner.ALL_BOND_TARGET_REMOVAL_BASIS_POINTS
    )
    assert [item["rank_tuple"][0] for item in schedule] == list(
        runner.ALL_BOND_EXPECTED_UNIFORM_RANKS
    )
    for item in schedule:
        assert len(set(item["rank_tuple"])) == 1
        realized_basis_points = 10_000 * (
            1.0 - sum(item["rank_tuple"]) / (4 * 33)
        )
        assert realized_basis_points <= item["target_removal_basis_points"] + 1e-12
    with pytest.raises(RuntimeError, match="four width-33 bonds"):
        runner.uniform_all_bond_schedule((33, 33, 33))


def test_dense_homogeneous_tree_storage_counts_physical_tensors():
    ranks = (33, 33, 33, 33)
    expected = 33 * 3073 + 3 * 33**3 + 10 * 33
    assert runner.dense_homogeneous_tree_storage(
        leaf_dimension=3073,
        output_dimension=10,
        ranks=ranks,
    ) == expected
    with pytest.raises(ValueError, match="positive integer"):
        runner.dense_homogeneous_tree_storage(
            leaf_dimension=3073,
            output_dimension=10,
            ranks=(33, 0, 33, 33),
        )


def test_resolved_frontier_skips_holes_but_blocks_post_failure_recovery():
    accuracies = [0.80, 0.795, 0.70, 0.799, 0.798]
    resolved = [True, False, True, True, True]
    points = [
        {
            "rank_tuple": [33 - index] * 4,
            "canonical_all_boundaries_resolved": is_resolved,
            "canonical_all_boundaries_roundoff_prefilter_pass": True,
            "accuracy": {
                "canonical_tree_odt": {
                    "discovery": accuracy,
                    "confirmation": 0.99,
                }
            },
        }
        for index, (accuracy, is_resolved) in enumerate(zip(accuracies, resolved))
    ]
    selected = runner.select_resolved_accuracy_frontier(points, 0.80)
    assert selected["eligible_schedule_indices"] == [0, 2, 3, 4]
    assert selected["selected_schedule_index"] == 0
    assert selected["first_failing_eligible_schedule_index"] == 2
    assert selected["selected_rank_tuple"] == [33, 33, 33, 33]
    assert selected["nontrivial_compression_selected"] is False
    assert selected["outcome"] == (
        "full_rank_selected_by_frozen_prefiltered_discovery_prefix_rule"
    )


def test_resolved_frontier_uses_discovery_only_and_advances_across_holes():
    points = [
        {
            "rank_tuple": [rank] * 4,
            "canonical_all_boundaries_resolved": is_resolved,
            "canonical_all_boundaries_roundoff_prefilter_pass": True,
            "accuracy": {
                "canonical_tree_odt": {
                    "discovery": discovery,
                    "confirmation": confirmation,
                }
            },
        }
        for rank, is_resolved, discovery, confirmation in (
            (33, True, 0.80, 0.80),
            (30, False, 0.10, 0.99),
            (27, True, 0.795, 0.00),
        )
    ]
    selected = runner.select_resolved_accuracy_frontier(points, 0.80)
    assert selected["eligible_schedule_indices"] == [0, 2]
    assert selected["selected_schedule_index"] == 2
    assert selected["selected_rank_tuple"] == [27, 27, 27, 27]
    assert selected["nontrivial_compression_selected"] is True
    assert selected["outcome"] == "nontrivial_rank_tuple_selected"


def test_all_bond_frontier_excludes_roundoff_ineligible_point_before_validation():
    points = [
        {
            "rank_tuple": [rank] * 4,
            "canonical_all_boundaries_resolved": True,
            "canonical_all_boundaries_roundoff_prefilter_pass": robust,
            "accuracy": {
                "canonical_tree_odt": {"discovery": 0.8, "confirmation": 0.8}
            },
        }
        for rank, robust in ((33, True), (30, False), (27, True))
    ]
    selected = runner.select_resolved_accuracy_frontier(points, 0.8)
    assert selected["eligible_schedule_indices"] == [0, 2]
    assert selected["selected_rank_tuple"] == [27, 27, 27, 27]
    assert selected["selection_gauge_trial_count"] == 0
    assert selected["held_out_validation_gauges_used_for_selection"] is False


def test_simultaneous_replays_are_diagnostic_but_single_bond_replay_is_gated():
    diagnostic = _protocol_measurement_fixture()
    diagnostic["gauge_audit"]["records"][0]["raw_coordinate_replay_relative"] = 1.0
    diagnostic["gauge_audit"]["max_raw_coordinate_replay_relative"] = 1.0
    diagnostic["gauge_audit"]["records"][0][
        "canonicalized_function_relative"
    ] = 1.0
    diagnostic["gauge_audit"]["max_canonicalized_function_relative"] = 1.0
    summary.validate_protocol_and_measurements(diagnostic, smoke=False)

    gated = _protocol_measurement_fixture()
    gated["gauge_audit"]["records"][0]["bond_realizations"][0][
        "sampled_single_bond_logit_replay_relative"
    ] = 1e-7
    gated["gauge_audit"]["records"][0][
        "maximum_sampled_single_bond_logit_replay_relative"
    ] = 1e-7
    gated["gauge_audit"]["max_sampled_single_bond_logit_replay_relative"] = 1e-7
    gated["algebraic_gates"][
        "gauge_sampled_single_bond_logit_replay_relative_le_1e-8"
    ] = False
    with pytest.raises(RuntimeError, match="disagree with measurements"):
        summary.validate_protocol_and_measurements(gated, smoke=False)


def test_gate_validation_rejects_missing_extra_false_and_none():
    record = {"seed": 0, "algebraic_gates": {key: True for key in summary.EXPECTED_ALGEBRAIC_GATES}}
    summary.require_exact_true_gates(
        record, "algebraic_gates", summary.EXPECTED_ALGEBRAIC_GATES
    )

    missing = {**record, "algebraic_gates": dict(record["algebraic_gates"])}
    missing["algebraic_gates"].pop(next(iter(summary.EXPECTED_ALGEBRAIC_GATES)))
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        summary.require_exact_true_gates(
            missing, "algebraic_gates", summary.EXPECTED_ALGEBRAIC_GATES
        )

    extra = {**record, "algebraic_gates": {**record["algebraic_gates"], "unknown": True}}
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        summary.require_exact_true_gates(
            extra, "algebraic_gates", summary.EXPECTED_ALGEBRAIC_GATES
        )

    for invalid in (False, None, 1):
        bad = {**record, "algebraic_gates": dict(record["algebraic_gates"])}
        bad["algebraic_gates"][next(iter(summary.EXPECTED_ALGEBRAIC_GATES))] = invalid
        with pytest.raises(RuntimeError, match="non-passing"):
            summary.require_exact_true_gates(
                bad, "algebraic_gates", summary.EXPECTED_ALGEBRAIC_GATES
            )


def test_seed_provenance_accepts_only_current_manifest_and_checkpoint(
    tmp_path, monkeypatch
):
    project_root, manifest_path, manifest_text, entries, _, record = _provenance_fixture(
        tmp_path, monkeypatch
    )
    _validate(record, project_root, manifest_path, manifest_text, entries)


def test_seed_provenance_rejects_stale_manifest(tmp_path, monkeypatch):
    project_root, manifest_path, manifest_text, entries, _, record = _provenance_fixture(
        tmp_path, monkeypatch
    )
    record["provenance"]["frozen_source_manifest"]["text"] += "0" * 64 + "  stale.py\n"
    with pytest.raises(RuntimeError, match="stale source manifest"):
        _validate(record, project_root, manifest_path, manifest_text, entries)


def test_seed_provenance_rejects_manifest_from_another_root(tmp_path, monkeypatch):
    project_root, manifest_path, manifest_text, entries, _, record = _provenance_fixture(
        tmp_path, monkeypatch
    )
    record["provenance"]["frozen_source_manifest"]["path"] = str(
        tmp_path / "another" / "source_manifest.sha256"
    )
    with pytest.raises(RuntimeError, match="outside this run root"):
        _validate(record, project_root, manifest_path, manifest_text, entries)


def test_seed_provenance_rejects_changed_source_hash_map(tmp_path, monkeypatch):
    project_root, manifest_path, manifest_text, entries, _, record = _provenance_fixture(
        tmp_path, monkeypatch
    )
    record["provenance"]["source_sha256"]["source.py"] = "0" * 64
    with pytest.raises(RuntimeError, match="source hash map mismatch"):
        _validate(record, project_root, manifest_path, manifest_text, entries)


def test_seed_provenance_rejects_nonofficial_dataset(tmp_path, monkeypatch):
    project_root, manifest_path, manifest_text, entries, _, record = _provenance_fixture(
        tmp_path, monkeypatch
    )
    record["provenance"]["dataset_files"] = dict(summary.EXPECTED_DATASET_FILES)
    record["provenance"]["dataset_files"].pop("test_32x32.mat")
    with pytest.raises(RuntimeError, match="official SVHN manifest mismatch"):
        _validate(record, project_root, manifest_path, manifest_text, entries)


def test_seed_provenance_rejects_checkpoint_outside_run(tmp_path, monkeypatch):
    project_root, manifest_path, manifest_text, entries, _, record = _provenance_fixture(
        tmp_path, monkeypatch
    )
    outside = tmp_path / "seed_0.pt"
    outside.write_bytes(b"checkpoint")
    record["provenance"]["checkpoint"]["path"] = str(outside)
    with pytest.raises(RuntimeError, match="outside this run"):
        _validate(record, project_root, manifest_path, manifest_text, entries)


def test_seed_provenance_rehashes_checkpoint(tmp_path, monkeypatch):
    project_root, manifest_path, manifest_text, entries, checkpoint, record = _provenance_fixture(
        tmp_path, monkeypatch
    )
    checkpoint.write_bytes(b"tampered!!")
    record["provenance"]["checkpoint"]["bytes"] = checkpoint.stat().st_size
    with pytest.raises(RuntimeError, match="checkpoint hash mismatch"):
        _validate(record, project_root, manifest_path, manifest_text, entries)


def test_seed_provenance_requires_environment_fingerprint(tmp_path, monkeypatch):
    project_root, manifest_path, manifest_text, entries, _, record = _provenance_fixture(
        tmp_path, monkeypatch
    )
    record["provenance"]["runtime"].pop("fingerprint_sha256")
    with pytest.raises(RuntimeError, match="runtime field inventory|environment fingerprint"):
        _validate(record, project_root, manifest_path, manifest_text, entries)


def test_seed_provenance_binds_array_parent_and_resolved_python(tmp_path, monkeypatch):
    project_root, manifest_path, manifest_text, entries, _, record = _provenance_fixture(
        tmp_path, monkeypatch
    )
    _validate(record, project_root, manifest_path, manifest_text, entries)

    wrong_parent = json.loads(json.dumps(record))
    wrong_parent["provenance"]["runtime"]["slurm_array_job_id"] = "999"
    with pytest.raises(RuntimeError, match="scheduler runtime inventory"):
        _validate(wrong_parent, project_root, manifest_path, manifest_text, entries)

    wrong_python = json.loads(json.dumps(record))
    wrong_runtime = wrong_python["provenance"]["runtime"]
    wrong_runtime["python_executable_resolved"] = "/wrong/python"
    fingerprint_keys = (
        "python", "python_executable_invoked", "python_executable_resolved",
        "torch", "torchvision", "numpy", "cuda", "cudnn", "distributions",
    )
    wrong_runtime["fingerprint_sha256"] = hashlib.sha256(
        json.dumps(
            {key: wrong_runtime[key] for key in fingerprint_keys},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    with pytest.raises(RuntimeError, match="deterministic runtime settings"):
        _validate(wrong_python, project_root, manifest_path, manifest_text, entries)


def test_seed_provenance_records_platform_but_excludes_it_from_fingerprint(
    tmp_path, monkeypatch
):
    project_root, manifest_path, manifest_text, entries, _, record = _provenance_fixture(
        tmp_path, monkeypatch
    )
    changed_platform = json.loads(json.dumps(record))
    changed_platform["provenance"]["runtime"]["platform"] = "Linux-another-node"
    _validate(changed_platform, project_root, manifest_path, manifest_text, entries)

    missing_platform = json.loads(json.dumps(record))
    missing_platform["provenance"]["runtime"].pop("platform")
    with pytest.raises(RuntimeError, match="runtime field inventory"):
        _validate(missing_platform, project_root, manifest_path, manifest_text, entries)


def test_official_svhn_validation_checks_name_size_md5_and_sha256(
    tmp_path, monkeypatch
):
    train = tmp_path / "train_32x32.mat"
    test = tmp_path / "test_32x32.mat"
    train.write_bytes(b"train")
    test.write_bytes(b"test")

    def metadata(path: Path):
        return {
            "bytes": path.stat().st_size,
            "md5": hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest(),
            "sha256": _sha(path),
        }

    expected = {
        "train_32x32.mat": metadata(train),
        "test_32x32.mat": metadata(test),
    }
    monkeypatch.setattr(runner, "SVHN_FILES", expected)
    assert runner.validated_svhn_manifest(tmp_path) == expected
    test.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        runner.validated_svhn_manifest(tmp_path)


def test_summary_rehashes_official_svhn_files(tmp_path, monkeypatch):
    train = tmp_path / "train_32x32.mat"
    test = tmp_path / "test_32x32.mat"
    train.write_bytes(b"train")
    test.write_bytes(b"test")

    def metadata(path: Path):
        return {
            "bytes": path.stat().st_size,
            "md5": hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest(),
            "sha256": _sha(path),
        }

    expected = {
        "train_32x32.mat": metadata(train),
        "test_32x32.mat": metadata(test),
    }
    monkeypatch.setattr(summary, "EXPECTED_DATASET_FILES", expected)
    assert summary.rehash_official_svhn(tmp_path) == expected
    train.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed before aggregation"):
        summary.rehash_official_svhn(tmp_path)


def test_affine_baseline_rank_zero_matches_canonical_rank_one_accounting():
    ranks = runner.affine_baseline_rank_grid([1, 2, 4, 33], learned_dimension=32)
    assert ranks == [0, 1, 3, 32]
    assert ranks[0] + 1 == 1
    assert [rank + 1 for rank in ranks] == [1, 2, 4, 33]


def test_choose_rank_respects_resolved_eigengap_inventory():
    curve = {"discovery": {"1": 0.99, "2": 0.99, "3": 1.0}}
    assert runner.choose_rank(curve, 1.0, allowed_ranks={2, 3}) == 2
    assert runner.choose_rank(curve, 1.0, allowed_ranks={3}) == 3
    with pytest.raises(ValueError, match="no allowed operating rank"):
        runner.choose_rank(curve, 1.0, allowed_ranks={4})


def test_roundoff_prefilter_closes_v3r1_1e_minus_8_vs_1e_minus_6_mismatch():
    floor = runner.roundoff_only_relative_gap_floor(33)
    assert isinstance(summary.expected_roundoff_relative_gap_floor(33), float)
    assert floor == pytest.approx(1.0363384255258677e-4, rel=1e-15)

    r4_relative_gap = 2.663504795369033e-8
    r4_spectrum = torch.tensor(
        [1.0, 1.0 - r4_relative_gap, *([0.0] * 31)], dtype=torch.float64
    )
    assert runner.eigengap_diagnostics(r4_spectrum, 1).resolved is True
    rejected = runner.roundoff_only_projector_prefilter(r4_spectrum, 1)
    independently_recomputed = summary.expected_roundoff_prefilter(
        r4_spectrum.tolist(), 1
    )
    summary.validate_finite_json_numbers(independently_recomputed)
    assert independently_recomputed == rejected
    assert rejected["relative_gap"] == pytest.approx(r4_relative_gap)
    assert rejected["passes"] is False
    assert rejected["bound_with_roundoff"] > runner.GAUGE_CERTIFICATE_MAX_BOUND

    below = floor * (1.0 - 1e-10)
    above = floor * (1.0 + 1e-10)
    below_spectrum = torch.tensor([1.0, 1.0 - below, *([0.0] * 31)], dtype=torch.float64)
    above_spectrum = torch.tensor([1.0, 1.0 - above, *([0.0] * 31)], dtype=torch.float64)
    assert runner.roundoff_only_projector_prefilter(below_spectrum, 1)["passes"] is False
    assert runner.roundoff_only_projector_prefilter(above_spectrum, 1)["passes"] is True


def test_rank_selection_is_analytic_and_cannot_postselect_on_validation_gauges(
    monkeypatch,
):
    monkeypatch.setattr(
        runner,
        "learned_gauge_audit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("held-out validation gauges entered rank selection")
        ),
    )
    spectrum = torch.tensor([1.0, 0.5, *([0.0] * 31)], dtype=torch.float64)
    curve = {"discovery": {"1": 0.995, "2": 0.90, "33": 1.0}}
    torch.manual_seed(1)
    first = runner.select_canonical_rank_prospectively(curve, 1.0, spectrum)
    torch.manual_seed(999999)
    second = runner.select_canonical_rank_prospectively(curve, 1.0, spectrum)
    assert first == second
    rank, audit = first
    assert rank == 1
    assert audit["nontrivial_compression_selected"] is True
    assert audit["selection_gauge_trial_count"] == 0
    assert audit["held_out_validation_gauge_seed"] == runner.GAUGE_SEED
    assert audit["held_out_validation_gauge_trial_count"] == 50
    assert audit["held_out_validation_gauges_used_for_selection"] is False


def test_rank_selection_reports_full_rank_fallback_without_compression_claim():
    spectrum = torch.tensor(
        [1.0] * 12 + [1.0 - 2.663504795369033e-8] * 21,
        dtype=torch.float64,
    )
    curve = {"discovery": {"12": 0.995, "33": 1.0}}
    rank, audit = runner.select_canonical_rank_prospectively(curve, 1.0, spectrum)
    assert rank == 33
    assert audit["joint_eligible_nontrivial_ranks"] == []
    assert audit["nontrivial_compression_selected"] is False
    assert audit["outcome"] == (
        "full_rank_fallback_no_nontrivial_accuracy_and_prefilter_eligible_rank"
    )
    assert audit["prefilter_by_rank"]["33"]["kind"] == "full_rank_identity_fallback"
    assert summary.expected_canonical_rank_selection(
        curve, 1.0, spectrum.tolist()
    ) == (rank, audit)


def _protocol_measurement_fixture(*, smoke: bool = False) -> dict:
    ranks = summary.SMOKE_RANK_GRID if smoke else summary.FULL_RANK_GRID
    haar_draws = 1 if smoke else 16
    protocol = {
        **summary.EXPECTED_PROTOCOL_COMMON,
        "rank_grid": ranks,
        "haar_draws": haar_draws,
        "gauge_required_ranks_by_bond": [[1], [1], [1], [1]],
        "smoke_nontrivial_diagnostic_rank": 1 if smoke else None,
        "discovery_indices_sha256": summary.EXPECTED_DISCOVERY_SHA256,
        "confirmation_indices_sha256": summary.EXPECTED_CONFIRMATION_SHA256,
    }
    canonical_keys = {str(rank): 0.8 for rank in ranks}
    affine_keys = {str(rank - 1): 0.8 for rank in ranks}
    curves = {}
    for method in summary.EXPECTED_METHODS:
        values = (
            affine_keys
            if method in {"legacy_raw_downstream_zero_anchor", "activation_pca_centered"}
            else canonical_keys
        )
        curves[method] = {"discovery": dict(values), "confirmation": dict(values)}
    spectrum = [float(value) for value in range(33, 0, -1)]
    base_spectra = [list(spectrum) for _ in range(4)]
    resolved_ranks = [
        [rank for rank in range(1, 33) if summary.expected_eigengap(spectrum, rank)["resolved"]]
        for _ in range(4)
    ]
    roundoff = float(10_000.0 * np.finfo(np.float64).eps * 33 * 33)
    arithmetic_slack = float(10_000.0 * np.finfo(np.float64).eps * 33)
    separation = float(1.0 - 2.0 * roundoff)
    certificate_bound = float(math.sqrt(2.0) * roundoff / separation)
    gauge_records = [
        {
            "trial": trial,
            "target_condition": 10.0 ** (3.0 * trial / 49.0),
            "target_minimum_singular_value": (
                10.0 ** (3.0 * trial / 49.0)
            ) ** -0.5,
            "target_maximum_singular_value": (
                10.0 ** (3.0 * trial / 49.0)
            ) ** 0.5,
            "minimum_realized_singular_value": (
                10.0 ** (3.0 * trial / 49.0)
            ) ** -0.5,
            "maximum_realized_singular_value": (
                10.0 ** (3.0 * trial / 49.0)
            ) ** 0.5,
            "minimum_realized_condition": 10.0 ** (3.0 * trial / 49.0),
            "maximum_realized_condition": 10.0 ** (3.0 * trial / 49.0),
            "minimum_realized_nonsymmetry_relative": 0.5,
            "bond_realizations": [
                {
                    "bond": bond,
                    "minimum_singular_value": (
                        10.0 ** (3.0 * trial / 49.0)
                    ) ** -0.5,
                    "maximum_singular_value": (
                        10.0 ** (3.0 * trial / 49.0)
                    ) ** 0.5,
                    "condition": 10.0 ** (3.0 * trial / 49.0),
                    "nonsymmetry_relative": 0.5,
                    "sampled_single_bond_logit_replay_relative": 0.0,
                    "prefix_overlap_minimum_singular_value": 1.0,
                    "prefix_overlap_maximum_singular_value": 1.0,
                    "prefix_overlap_span_defect": 0.0,
                    "transport_orthogonality": 0.0,
                    "raw_factor_transport_relative_diagnostic": 0.0,
                    "gram_covariance_relative": 0.0,
                    "spectrum_relative": 0.0,
                    "projector_certificates": [
                        {
                            "rank": rank,
                            "gap": 1.0,
                            "covariance_residual_operator": 0.0,
                            "base_eigensolver_residual_operator": 0.0,
                            "candidate_eigensolver_residual_operator": 0.0,
                            "base_eigenvector_orthogonality": 0.0,
                            "candidate_eigenvector_orthogonality": 0.0,
                            "transport_orthogonality": 0.0,
                            "roundoff_allowance": roundoff,
                            "eta": roundoff,
                            "separation_margin": separation,
                            "numerically_certifiable": True,
                            "normalized_projector_distance": 0.0,
                            "davis_kahan_bound": certificate_bound,
                            "bound_with_roundoff": certificate_bound + arithmetic_slack,
                            "passes_bound": True,
                            "base_supported_overlap_defect": 0.0,
                            "candidate_supported_overlap_defect": 0.0,
                            "polar_replacement_supported_defect": 0.0,
                            "maximum_supported_overlap_defect": 0.0,
                            "required_for_certificate": rank == 1,
                        }
                        for rank in resolved_ranks[bond]
                    ],
                }
                for bond in range(4)
            ],
            "raw_coordinate_replay_relative": 0.0,
            "canonicalized_function_relative": 0.0,
            "canonicalized_coefficient_replay_relative": 0.0,
            "maximum_sampled_single_bond_logit_replay_relative": 0.0,
            "max_prefix_overlap_span_defect": 0.0,
            "max_transport_orthogonality": 0.0,
            "max_raw_factor_transport_relative_diagnostic": 0.0,
            "max_gram_covariance_relative": 0.0,
            "max_spectrum_relative": 0.0,
            "projector_certificate_pair_count": sum(map(len, resolved_ranks)),
            "expected_projector_certificate_pair_count": sum(map(len, resolved_ranks)),
        }
        for trial in range(50)
    ]
    full_norm = sum(spectrum)
    all_bond_points = []
    for index, scheduled in enumerate(summary._all_bond_schedule()):
        rank = scheduled["rank_tuple"][0]
        ranks_tuple = scheduled["rank_tuple"]
        boundaries = [summary.expected_eigengap(spectrum, rank) for _ in range(4)]
        terms = [
            {
                "bond": bond,
                "rank": rank,
                "multiplicity": multiplicity,
                "discarded_trace": sum(spectrum[rank:]),
                "weighted_discarded_trace": multiplicity * sum(spectrum[rank:]),
            }
            for bond, multiplicity in enumerate(summary.ALL_BOND_MULTIPLICITIES)
        ]
        bound = sum(term["weighted_discarded_trace"] for term in terms)
        full_storage = summary._dense_storage([33] * 4)
        retained_storage = summary._dense_storage(ranks_tuple)
        all_bond_points.append({
            "schedule_index": index,
            "target_removal_basis_points": scheduled["target_removal_basis_points"],
            "uniform_rank": rank,
            "rank_tuple": ranks_tuple,
            "canonical_boundaries": boundaries,
            "canonical_all_boundaries_resolved": True,
            "canonical_roundoff_only_prefilters": [
                summary.expected_roundoff_prefilter(values, rank)
                for values in base_spectra
            ],
            "canonical_all_boundaries_roundoff_prefilter_pass": True,
            "local_boundaries": boundaries,
            "local_all_boundaries_resolved": True,
            "compression": {
                "unique_bond_coordinate_removal_fraction": 1.0 - sum(ranks_tuple) / 132,
                "occurrence_weighted_bond_coordinate_removal_fraction": 1.0 - sum(
                    m * r for m, r in zip(summary.ALL_BOND_MULTIPLICITIES, ranks_tuple)
                ) / (15 * 33),
                "dense_homogeneous_tn_full_storage_scalars": full_storage,
                "dense_homogeneous_tn_retained_storage_scalars": retained_storage,
                "dense_homogeneous_tn_storage_removal_fraction": 1.0 - retained_storage / full_storage,
            },
            "coefficient_certificate": {
                "object": "topology_specific_independent_clone_tree",
                "error_evaluation": "canonicalized_block_sparse_difference_tree",
                "full_coefficient_norm_squared": full_norm,
                "tail_terms": terms,
                "hsvd_upper_bound_squared": bound,
                "hsvd_relative_error_upper_bound": math.sqrt(bound / full_norm),
                "actual_coefficient_error_squared": 0.0,
                "actual_relative_coefficient_error": 0.0,
                "bound_numerical_slack": 1e-8 * bound + 1e-10 * full_norm,
                "actual_error_within_bound_plus_declared_numerical_slack": True,
            },
            "accuracy": {
                "canonical_tree_odt": {"discovery": 0.8, "confirmation": 0.8},
                "canonical_adjacent_local": {"discovery": 0.8, "confirmation": 0.8},
                "independent_nested_haar": {
                    "draw_count": 64,
                    "discovery": [0.7] * 64,
                    "confirmation": [0.7] * 64,
                    "discovery_mean": float(np.mean([0.7] * 64)),
                    "discovery_sample_standard_deviation": float(np.std([0.7] * 64, ddof=1)),
                    "confirmation_mean": float(np.mean([0.7] * 64)),
                    "confirmation_sample_standard_deviation": float(np.std([0.7] * 64, ddof=1)),
                },
            },
        })
    frontier = summary._all_bond_frontier(all_bond_points, 0.8)
    frontier.update({
        "selected_discovery_accuracy": 0.8,
        "selected_confirmation_accuracy": 0.8,
        "selected_confirmation_drop_from_full": 0.0,
        "selected_local_all_boundaries_resolved": True,
        "selected_confirmation_odt_minus_local": 0.0,
        "selected_confirmation_haar_at_least_canonical_count": 0,
        "selected_confirmation_haar_randomization_p": 1 / 65,
    })
    record = {
        "schema": summary.SCHEMA,
        "status": "complete",
        "claim_boundary": summary.EXPECTED_CLAIM_BOUNDARY,
        "seed": 9173 if smoke else 0,
        "predecessor_failures": [dict(item) for item in summary.EXPECTED_PREDECESSOR_FAILURES],
        "protocol": protocol,
        "training_discovery_accuracy_by_epoch": [0.8] * 20,
        "post_training_rbn_calibration": {
            "sites": [
                {"path": f"norms.{index}", "scale": 1.0, "batches": 50}
                for index in range(3)
            ],
            "excluded_paths": [],
        },
        "curves": curves,
        "canonical_rank_selection": summary.expected_canonical_rank_selection(
            curves["canonical_tree_odt"], 0.8, spectrum
        )[1],
        "haar_curves": {
            str(index): {"discovery": dict(canonical_keys), "confirmation": dict(canonical_keys)}
            for index in range(haar_draws)
        },
        "canonicalization": {
            "sampled_logit_reconstruction": {
                name: {
                    "max_absolute_logit_error": 0.0,
                    "relative_frobenius_logit_error": 0.0,
                }
                for name in ("homogeneous_export", "canonical_full", "full_rank_diagonalized")
            },
            "sampled_logit_reconstruction_count": 256,
            "sampled_logit_gate_max_absolute": 1e-8,
            "sampled_logit_gate_pass": True,
            "checkpoint_reload_max_absolute_logit_error": 0.0,
            "factorization_errors": [0.0] * 4,
            "isometry_errors": [0.0] * 4,
            "symmetry_errors": [0.0] * 3,
            "module_vs_homogeneous_discovery_abs": 0.0,
            "module_vs_canonical_discovery_abs": 0.0,
            "module_vs_homogeneous_confirmation_abs": 0.0,
            "module_vs_canonical_confirmation_abs": 0.0,
        },
        "gauge_audit": {
            "base_spectra_by_bond": base_spectra,
            "resolved_ranks_by_bond": resolved_ranks,
            "required_ranks_by_bond": [[1], [1], [1], [1]],
            "records": gauge_records,
            "max_raw_coordinate_replay_relative": 0.0,
            "max_canonicalized_function_relative": 0.0,
            "max_canonicalized_coefficient_replay_relative": 0.0,
            "max_sampled_single_bond_logit_replay_relative": 0.0,
            "max_singular_extrema_relative_error": 0.0,
            "max_condition_relative_error": 0.0,
            "minimum_nonsymmetry_relative": 0.5,
            "max_prefix_overlap_span_defect_diagnostic": 0.0,
            "max_transport_orthogonality": 0.0,
            "max_raw_factor_transport_relative_diagnostic": 0.0,
            "max_gram_covariance_relative": 0.0,
            "max_spectrum_relative": 0.0,
            "projector_certificate_count": 50 * sum(map(len, resolved_ranks)),
            "expected_required_projector_certificate_count": 50 * 4,
            "required_projector_certificate_count": 50 * 4,
            "projector_certificate_coverage_complete": True,
            "maximum_required_projector_overlap_support_defect": 0.0,
            "all_required_projectors_numerically_certifiable": True,
            "maximum_required_projector_bound_with_roundoff": (
                certificate_bound + arithmetic_slack
            ),
            "all_required_projectors_pass_bound": True,
            "matrix_artifact": {
                "path": "/not-used-by-record-only-validation.npz",
                "bytes": 1,
                "sha256": "0" * 64,
                "allow_pickle": False,
                "keys": {},
            },
        },
        "canonical_spectrum": spectrum,
        "canonical_trace_tail_fraction": {
            str(rank): sum(spectrum[rank:]) / sum(spectrum) for rank in ranks
        },
        "canonical_eigengap_diagnostics": {
            str(rank): summary.expected_eigengap(spectrum, rank) for rank in ranks
        },
        "references": {
            "module": {"discovery": 0.8, "confirmation": 0.8},
            "homogeneous_export": {"discovery": 0.8, "confirmation": 0.8},
            "canonical_full": {"discovery": 0.8, "confirmation": 0.8},
        },
        "simultaneous_uniform_all_bond_curve": {
            "claim_boundary": summary.EXPECTED_ALL_BOND_CLAIM_BOUNDARY,
            "target_removal_basis_points": list(summary.EXPECTED_PROTOCOL_COMMON["all_bond_target_removal_basis_points"]),
            "expected_uniform_ranks": list(summary.EXPECTED_PROTOCOL_COMMON["all_bond_expected_uniform_ranks"]),
            "bond_dimensions": [33] * 4,
            "bond_occurrence_multiplicities": list(summary.ALL_BOND_MULTIPLICITIES),
            "haar_draws": 64,
            "haar_construction": summary.EXPECTED_HAAR_CONSTRUCTION,
            "trace_totals": [full_norm] * 4,
            "local_spectra_by_bond": [list(values) for values in base_spectra],
            "maximum_trace_total_relative_disagreement": 0.0,
            "points": all_bond_points,
            "discovery_selection": frontier,
        },
        "provenance": {},
    }
    selected = {}
    for method, curve in curves.items():
        rank = min(int(value) for value in curve["discovery"])
        reference_rank = max(int(value) for value in curve["discovery"])
        selected[method] = {
            "rank": rank,
            "effective_affine_rank": (
                rank + 1
                if method in {"legacy_raw_downstream_zero_anchor", "activation_pca_centered"}
                else rank
            ),
            "discovery_accuracy": 0.8,
            "confirmation_accuracy": 0.8,
            "reference_rank": reference_rank,
            "reference_discovery_accuracy": 0.8,
            "confirmation_drop_from_module": 0.0,
            "eigengap_resolved": True if method == "canonical_tree_odt" else None,
        }
    record["discovery_selected_confirmation"] = selected
    record["haar_discovery_selected_confirmation"] = {
        str(index): {
            "rank": min(ranks),
            "discovery_accuracy": 0.8,
            "confirmation_accuracy": 0.8,
        }
        for index in range(haar_draws)
    }
    record["algebraic_gates"] = {name: True for name in summary.EXPECTED_ALGEBRAIC_GATES}
    record["capability_gates"] = {name: True for name in summary.EXPECTED_CAPABILITY_GATES}
    return record


def test_protocol_and_measurements_are_recomputed_fail_closed():
    record = _protocol_measurement_fixture()
    summary.validate_protocol_and_measurements(record, smoke=False)

    wrong_epochs = _protocol_measurement_fixture()
    wrong_epochs["protocol"]["epochs"] = 19
    with pytest.raises(RuntimeError, match="frozen experiment"):
        summary.validate_protocol_and_measurements(wrong_epochs, smoke=False)

    false_measurement = _protocol_measurement_fixture()
    false_measurement["canonicalization"]["factorization_errors"][0] = 1e-3
    with pytest.raises(RuntimeError, match="disagree with measurements"):
        summary.validate_protocol_and_measurements(false_measurement, smoke=False)

    wrong_ranks = _protocol_measurement_fixture()
    wrong_ranks["protocol"]["rank_grid"] = [1, 33]
    with pytest.raises(RuntimeError, match="rank grid"):
        summary.validate_protocol_and_measurements(wrong_ranks, smoke=False)

    calibration = _protocol_measurement_fixture()
    calibration["post_training_rbn_calibration"]["sites"][1]["batches"] = 49
    with pytest.raises(RuntimeError, match="scalar-RBN calibration"):
        summary.validate_protocol_and_measurements(calibration, smoke=False)


def test_gate_failed_smoke_measurements_are_independently_recomputed():
    record = _protocol_measurement_fixture(smoke=True)
    record["status"] = "gate_failed"
    record["canonicalization"]["sampled_logit_reconstruction"][
        "canonical_full"
    ]["max_absolute_logit_error"] = 2e-8
    record["canonicalization"]["sampled_logit_gate_pass"] = False
    record["algebraic_gates"][
        "sampled_logit_reconstruction_le_1e-8"
    ] = False
    algebraic, capability = summary.validate_protocol_and_measurements(
        record,
        smoke=True,
        expected_status="gate_failed",
        require_all_gates=False,
    )
    assert algebraic["sampled_logit_reconstruction_le_1e-8"] is False
    assert all(capability.values())
    with pytest.raises(RuntimeError, match="result identity"):
        summary.validate_protocol_and_measurements(record, smoke=True)


def test_smoke_failure_authenticator_uses_real_measurement_recomputation(
    tmp_path, monkeypatch
):
    results = tmp_path / "results"
    results.mkdir()
    manifest_path = tmp_path / "source_manifest.sha256"
    manifest_path.write_text("frozen manifest\n")
    manifest_sha = _sha(manifest_path)
    data_root = tmp_path / "svhn"
    data_root.mkdir()
    (results / "launch.json").write_text(json.dumps({"data_root": str(data_root)}))
    record = _protocol_measurement_fixture(smoke=True)
    record["status"] = "gate_failed"
    record["canonicalization"]["sampled_logit_reconstruction"][
        "canonical_full"
    ]["max_absolute_logit_error"] = 2e-8
    record["canonicalization"]["sampled_logit_gate_pass"] = False
    record["algebraic_gates"][
        "sampled_logit_reconstruction_le_1e-8"
    ] = False
    record["provenance"] = {
        "frozen_source_manifest": {"sha256": manifest_sha},
        "runtime": {"slurm_job_id": "42"},
        "dataset_root": str(data_root),
    }
    smoke_path = results / "smoke.json"
    smoke_path.write_text(json.dumps(record))
    monkeypatch.setattr(summary, "verify_source_freeze", lambda root, path: {})
    monkeypatch.setattr(
        summary,
        "validate_launch_binding",
        lambda launch, **kwargs: {"smoke": "42"},
    )
    monkeypatch.setattr(summary, "validate_seed_provenance", lambda record, **kwargs: None)
    monkeypatch.setattr(summary, "rehash_official_svhn", lambda root: {})
    monkeypatch.setattr(summary, "regenerate_gauge_replay_sample", lambda root: object())
    monkeypatch.setattr(summary, "export_checkpoint_homogeneous", lambda record, **kwargs: object())
    matrix_authentication = {"fixture": "three-way"}
    monkeypatch.setattr(
        summary,
        "validate_gauge_matrix_artifact",
        lambda record, **kwargs: (
            kwargs["expected_path"], matrix_authentication
        ),
    )
    authenticated = summary.authenticated_smoke_failure(tmp_path, manifest_sha)
    assert authenticated["false_algebraic_gates"] == [
        "sampled_logit_reconstruction_le_1e-8"
    ]
    assert authenticated["false_capability_gates"] == []
    assert authenticated["required_projector_three_way_authentication"] == (
        matrix_authentication
    )


def test_summary_failure_receipt_survives_malformed_smoke(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    (tmp_path / "source_manifest.sha256").write_text("frozen manifest\n")
    (results / "launch.json").write_text("{}")
    (results / "smoke.json").write_text("{malformed")
    monkeypatch.setattr(summary, "verify_source_freeze", lambda root, path: {})
    monkeypatch.setattr(
        summary,
        "validate_launch_binding",
        lambda launch, **kwargs: {"smoke": "42"},
    )
    summary.write_summary_failure_receipt(
        tmp_path, RuntimeError("terminal aggregation failed"), slurm_job_id="99"
    )
    receipt = json.loads((results / "summary_failure.json").read_text())
    assert receipt["schema"] == "xvla-canonical-tree-odt-svhn-summary-v3r5"
    assert receipt["status"] == "failed"
    assert receipt["error"] == "terminal aggregation failed"
    assert receipt["authenticated_upstream_failure"] is None
    assert receipt["upstream_authentication_error"]["error_type"] == "JSONDecodeError"
    assert receipt["authenticated_seed_failures"] == {
        "validated_seed_count": 0,
        "passing_seed_count": 0,
        "failed_seed_count": 0,
        "status_by_seed": {},
        "failures": [],
    }
    assert receipt["seed_authentication_error"] is None
    assert receipt["slurm_job_id"] == "99"


def test_gate_failed_seed_receipt_authenticates_json_checkpoint_npz_and_provenance(
    tmp_path, monkeypatch
):
    results = tmp_path / "results"
    results.mkdir()
    manifest = tmp_path / "source_manifest.sha256"
    manifest.write_text("frozen manifest\n")
    manifest_sha = _sha(manifest)
    data_root = tmp_path / "svhn"
    data_root.mkdir()
    (results / "launch.json").write_text(json.dumps({"data_root": str(data_root)}))
    for seed in range(3):
        checkpoint = results / f"seed_{seed}.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        gauge = results / f"seed_{seed}_gauge_matrices.npz"
        gauge.write_bytes(f"gauge-{seed}".encode())
        status = "gate_failed" if seed == 0 else "complete"
        (results / f"seed_{seed}.json").write_text(json.dumps({
            "schema": summary.SCHEMA,
            "seed": seed,
            "status": status,
            "canonical_rank_selection": {"selected_rank": 33, "outcome": "fixture"},
            "simultaneous_uniform_all_bond_curve": {
                "discovery_selection": {
                    "selected_rank_tuple": [33, 33, 33, 33],
                    "nontrivial_compression_selected": False,
                    "outcome": "full_rank_selected_by_frozen_prefiltered_discovery_prefix_rule",
                }
            },
            "provenance": {
                "dataset_root": str(data_root),
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": _sha(checkpoint),
                },
                "runtime": {
                    "slurm_array_job_id": "42",
                    "slurm_array_task_id": str(seed),
                    "slurm_job_id": str(100 + seed),
                },
            },
        }))

    calls = []
    monkeypatch.setattr(summary, "verify_source_freeze", lambda root, path: {})
    monkeypatch.setattr(
        summary,
        "validate_launch_binding",
        lambda launch, **kwargs: {"train_array": "42"},
    )
    monkeypatch.setattr(summary, "rehash_official_svhn", lambda root: calls.append("dataset"))
    monkeypatch.setattr(summary, "regenerate_gauge_replay_sample", lambda root: "sample")
    monkeypatch.setattr(
        summary,
        "validate_protocol_and_measurements",
        lambda record, **kwargs: (
            {key: not (record["seed"] == 0 and key == "selected_canonical_eigengap_resolved")
             for key in summary.EXPECTED_ALGEBRAIC_GATES},
            {key: True for key in summary.EXPECTED_CAPABILITY_GATES},
        ),
    )
    monkeypatch.setattr(
        summary,
        "validate_seed_provenance",
        lambda record, **kwargs: calls.append(("provenance", record["seed"])),
    )
    monkeypatch.setattr(
        summary,
        "export_checkpoint_homogeneous",
        lambda record, **kwargs: calls.append(("checkpoint", record["seed"])) or object(),
    )
    monkeypatch.setattr(
        summary,
        "validate_gauge_matrix_artifact",
        lambda record, **kwargs: (
            calls.append(("npz", record["seed"])),
            (kwargs["expected_path"], {"seed": record["seed"]}),
        )[1],
    )

    authenticated = summary.authenticated_seed_failures(tmp_path, manifest_sha)
    assert authenticated["validated_seed_count"] == 3
    assert authenticated["passing_seed_count"] == 2
    assert authenticated["failed_seed_count"] == 1
    assert authenticated["status_by_seed"] == {
        "0": "gate_failed", "1": "complete", "2": "complete"
    }
    failure = authenticated["failures"][0]
    assert failure["seed"] == 0
    assert failure["false_algebraic_gates"] == ["selected_canonical_eigengap_resolved"]
    assert failure["false_capability_gates"] == []
    assert failure["canonical_rank_selection"] == {
        "selected_rank": 33, "outcome": "fixture"
    }
    assert failure["all_bond_rank_selection"]["nontrivial_compression_selected"] is False
    assert failure["json"]["sha256"] == _sha(results / "seed_0.json")
    assert failure["checkpoint"]["sha256"] == _sha(results / "seed_0.pt")
    assert failure["gauge_npz"]["sha256"] == _sha(
        results / "seed_0_gauge_matrices.npz"
    )
    assert failure["required_projector_three_way_authentication"] == {"seed": 0}
    assert calls == [
        "dataset",
        ("provenance", 0), ("checkpoint", 0), ("npz", 0),
        ("provenance", 1), ("checkpoint", 1), ("npz", 1),
        ("provenance", 2), ("checkpoint", 2), ("npz", 2),
    ]

def test_seed_failure_authenticator_uses_real_full_measurement_recomputation(
    tmp_path, monkeypatch
):
    results = tmp_path / "results"
    results.mkdir()
    manifest = tmp_path / "source_manifest.sha256"
    manifest.write_text("frozen manifest\n")
    manifest_sha = _sha(manifest)
    data_root = tmp_path / "svhn"
    data_root.mkdir()
    (results / "launch.json").write_text(json.dumps({"data_root": str(data_root)}))
    for seed in range(3):
        record = _protocol_measurement_fixture()
        record["seed"] = seed
        if seed == 0:
            record["status"] = "gate_failed"
            record["canonicalization"]["sampled_logit_reconstruction"][
                "canonical_full"
            ]["max_absolute_logit_error"] = 2e-8
            record["canonicalization"]["sampled_logit_gate_pass"] = False
            record["algebraic_gates"][
                "sampled_logit_reconstruction_le_1e-8"
            ] = False
        checkpoint = results / f"seed_{seed}.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        gauge = results / f"seed_{seed}_gauge_matrices.npz"
        gauge.write_bytes(f"gauge-{seed}".encode())
        record["provenance"] = {
            "dataset_root": str(data_root),
            "checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint)},
            "runtime": {
                "slurm_array_job_id": "42",
                "slurm_array_task_id": str(seed),
                "slurm_job_id": str(100 + seed),
            },
        }
        (results / f"seed_{seed}.json").write_text(json.dumps(record))

    matrix_gate_modes = []
    monkeypatch.setattr(summary, "verify_source_freeze", lambda root, path: {})
    monkeypatch.setattr(
        summary,
        "validate_launch_binding",
        lambda launch, **kwargs: {"train_array": "42"},
    )
    monkeypatch.setattr(summary, "rehash_official_svhn", lambda root: {})
    monkeypatch.setattr(summary, "regenerate_gauge_replay_sample", lambda root: "sample")
    monkeypatch.setattr(summary, "validate_seed_provenance", lambda record, **kwargs: None)
    monkeypatch.setattr(summary, "export_checkpoint_homogeneous", lambda record, **kwargs: object())

    def validate_npz(record, **kwargs):
        matrix_gate_modes.append(
            (record["seed"], kwargs["require_all_matrix_gates"])
        )
        return kwargs["expected_path"], {"seed": record["seed"]}

    monkeypatch.setattr(summary, "validate_gauge_matrix_artifact", validate_npz)
    authenticated = summary.authenticated_seed_failures(tmp_path, manifest_sha)
    assert authenticated["status_by_seed"] == {
        "0": "gate_failed", "1": "complete", "2": "complete"
    }
    assert authenticated["failures"][0]["false_algebraic_gates"] == [
        "sampled_logit_reconstruction_le_1e-8"
    ]
    assert authenticated["failures"][0]["false_capability_gates"] == []
    assert matrix_gate_modes == [(0, False), (1, True), (2, True)]


def test_all_bond_curve_schedule_statistics_and_certificates_are_recomputed():
    record = _protocol_measurement_fixture()
    summary.validate_protocol_and_measurements(record, smoke=False)

    schedule = _protocol_measurement_fixture()
    schedule["simultaneous_uniform_all_bond_curve"]["points"][1]["rank_tuple"] = [29] * 4
    with pytest.raises(RuntimeError, match="rank schedule"):
        summary.validate_protocol_and_measurements(schedule, smoke=False)

    haar = _protocol_measurement_fixture()
    haar["simultaneous_uniform_all_bond_curve"]["points"][0]["accuracy"][
        "independent_nested_haar"
    ]["confirmation_mean"] += 1e-4
    with pytest.raises(RuntimeError, match="Haar confirmation mean"):
        summary.validate_protocol_and_measurements(haar, smoke=False)

    tail = _protocol_measurement_fixture()
    tail["simultaneous_uniform_all_bond_curve"]["points"][1][
        "coefficient_certificate"
    ]["tail_terms"][0]["discarded_trace"] += 1.0
    with pytest.raises(RuntimeError, match="discarded trace"):
        summary.validate_protocol_and_measurements(tail, smoke=False)

    frontier = _protocol_measurement_fixture()
    frontier["simultaneous_uniform_all_bond_curve"]["discovery_selection"][
        "selected_schedule_index"
    ] = 0
    with pytest.raises(RuntimeError, match="selection frontier"):
        summary.validate_protocol_and_measurements(frontier, smoke=False)

    local_spectrum = _protocol_measurement_fixture()
    local_spectrum["simultaneous_uniform_all_bond_curve"]["local_spectra_by_bond"][0][0] += 1.0
    with pytest.raises(RuntimeError, match="local boundary"):
        summary.validate_protocol_and_measurements(local_spectrum, smoke=False)


def test_independent_float64_reductions_use_tight_nonzero_tolerance():
    summary._close_number(1.0 + 5e-15, 1.0, label="last-bit reduction")
    with pytest.raises(RuntimeError, match="independent recomputation"):
        summary._close_number(1.0 + 1e-8, 1.0, label="material reduction drift")


def test_all_bond_local_empirical_check_requires_strict_odt_advantage():
    tied = _protocol_measurement_fixture()
    tied_validation = summary.validate_all_bond_curve(tied)
    assert tied_validation["selected_odt_gt_local_with_resolved_boundaries"] is False

    better = _protocol_measurement_fixture()
    selected_index = better["simultaneous_uniform_all_bond_curve"][
        "discovery_selection"
    ]["selected_schedule_index"]
    selected = better["simultaneous_uniform_all_bond_curve"]["points"][selected_index]
    selected["accuracy"]["canonical_adjacent_local"]["confirmation"] = 0.79
    better["simultaneous_uniform_all_bond_curve"]["discovery_selection"][
        "selected_confirmation_odt_minus_local"
    ] = 0.8 - 0.79
    better_validation = summary.validate_all_bond_curve(better)
    assert better_validation["selected_odt_gt_local_with_resolved_boundaries"] is True


def test_result_rejects_bad_predecessor_replay_hash_and_accuracy_domain():
    predecessor = _protocol_measurement_fixture()
    predecessor["predecessor_failures"][1]["observed_relative_error"] = 0.0
    with pytest.raises(RuntimeError, match="predecessor-failure lineage"):
        summary.validate_protocol_and_measurements(predecessor, smoke=False)

    replay = _protocol_measurement_fixture()
    replay["protocol"]["gauge_replay_sample_indices_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="frozen experiment"):
        summary.validate_protocol_and_measurements(replay, smoke=False)

    accuracy = _protocol_measurement_fixture()
    accuracy["curves"]["canonical_tree_odt"]["discovery"]["1"] = 1.1
    with pytest.raises(RuntimeError, match=r"\[0, 1\]"):
        summary.validate_protocol_and_measurements(accuracy, smoke=False)

    boolean = _protocol_measurement_fixture()
    boolean["curves"]["canonical_tree_odt"]["discovery"]["1"] = True
    with pytest.raises(RuntimeError, match="finite non-boolean"):
        summary.validate_protocol_and_measurements(boolean, smoke=False)

    protocol_boolean = _protocol_measurement_fixture()
    protocol_boolean["protocol"]["gradient_clip"] = True
    with pytest.raises(RuntimeError, match="frozen experiment"):
        summary.validate_protocol_and_measurements(protocol_boolean, smoke=False)


def test_every_projector_certificate_rank_is_authenticated_from_base_spectra():
    record = _protocol_measurement_fixture()
    record["gauge_audit"]["records"][0]["bond_realizations"][0][
        "projector_certificates"
    ].pop()
    record["gauge_audit"]["records"][0]["projector_certificate_pair_count"] -= 1
    with pytest.raises(RuntimeError, match="rank inventory"):
        summary.validate_protocol_and_measurements(record, smoke=False)

    boolean_rank = _protocol_measurement_fixture()
    boolean_rank["gauge_audit"]["resolved_ranks_by_bond"][0][0] = True
    with pytest.raises(RuntimeError, match="resolved ranks"):
        summary.validate_protocol_and_measurements(boolean_rank, smoke=False)


def test_full_rank_required_points_do_not_create_nontrivial_certificate_counts():
    gauge = _protocol_measurement_fixture()["gauge_audit"]
    gauge["required_ranks_by_bond"] = [[33], [33], [33], [33]]
    for trial in gauge["records"]:
        for bond in trial["bond_realizations"]:
            for certificate in bond["projector_certificates"]:
                certificate["required_for_certificate"] = False
    gauge.update(
        {
            "expected_required_projector_certificate_count": 0,
            "required_projector_certificate_count": 0,
            "maximum_required_projector_overlap_support_defect": None,
            "all_required_projectors_numerically_certifiable": False,
            "maximum_required_projector_bound_with_roundoff": None,
            "all_required_projectors_pass_bound": False,
        }
    )
    recomputed = summary.recompute_gauge_audit(gauge, 50)
    assert recomputed["expected_required_projector_certificate_count"] == 0
    assert recomputed["required_projector_certificate_count"] == 0


@pytest.mark.parametrize("key", sorted(summary.EXPECTED_PROTOCOL_COMMON))
def test_every_frozen_common_protocol_field_is_pinned(key):
    record = _protocol_measurement_fixture()
    record["protocol"][key] = "tampered"
    with pytest.raises(RuntimeError, match="frozen experiment"):
        summary.validate_protocol_and_measurements(record, smoke=False)


def test_protocol_inventory_and_haar_count_are_exact():
    missing = _protocol_measurement_fixture()
    missing["protocol"].pop("optimizer")
    with pytest.raises(RuntimeError, match="field inventory"):
        summary.validate_protocol_and_measurements(missing, smoke=False)

    wrong_haar = _protocol_measurement_fixture()
    wrong_haar["protocol"]["haar_draws"] = 15
    with pytest.raises(RuntimeError, match="Haar count"):
        summary.validate_protocol_and_measurements(wrong_haar, smoke=False)


def test_selected_endpoint_and_module_accuracy_deltas_are_recomputed():
    endpoint = _protocol_measurement_fixture()
    endpoint["discovery_selected_confirmation"]["canonical_tree_odt"][
        "confirmation_accuracy"
    ] = 0.9
    with pytest.raises(RuntimeError, match="selected endpoint"):
        summary.validate_protocol_and_measurements(endpoint, smoke=False)

    delta = _protocol_measurement_fixture()
    delta["canonicalization"]["module_vs_canonical_confirmation_abs"] = 0.1
    with pytest.raises(RuntimeError, match="accuracy delta"):
        summary.validate_protocol_and_measurements(delta, smoke=False)


def test_gauge_trial_inventory_and_aggregates_are_recomputed():
    missing = _protocol_measurement_fixture()
    missing["gauge_audit"]["records"].pop()
    with pytest.raises(RuntimeError, match="record count"):
        summary.validate_protocol_and_measurements(missing, smoke=False)

    extra = _protocol_measurement_fixture()
    extra["gauge_audit"]["records"].append(dict(extra["gauge_audit"]["records"][-1]))
    with pytest.raises(RuntimeError, match="record count"):
        summary.validate_protocol_and_measurements(extra, smoke=False)

    duplicate = _protocol_measurement_fixture()
    duplicate["gauge_audit"]["records"][1]["trial"] = 0
    with pytest.raises(RuntimeError, match="trial inventory"):
        summary.validate_protocol_and_measurements(duplicate, smoke=False)

    mismatched = _protocol_measurement_fixture()
    mismatched["gauge_audit"]["max_canonicalized_function_relative"] = 1e-12
    with pytest.raises(RuntimeError, match="gauge aggregates"):
        summary.validate_protocol_and_measurements(mismatched, smoke=False)

    negative = _protocol_measurement_fixture()
    negative["gauge_audit"]["records"][0]["max_spectrum_relative"] = -1e-12
    with pytest.raises(RuntimeError, match="gauge measurement"):
        summary.validate_protocol_and_measurements(negative, smoke=False)

    boolean = _protocol_measurement_fixture()
    boolean["gauge_audit"]["records"][0]["maximum_realized_condition"] = True
    with pytest.raises(RuntimeError, match="finite non-boolean number"):
        summary.validate_protocol_and_measurements(boolean, smoke=False)


def test_generator_realization_gates_are_recomputed_from_every_bond():
    singular = _protocol_measurement_fixture()
    singular["gauge_audit"]["records"][0]["bond_realizations"][0][
        "minimum_singular_value"
    ] *= 0.5
    singular["gauge_audit"]["records"][0]["bond_realizations"][0][
        "maximum_singular_value"
    ] *= 0.5
    singular["gauge_audit"]["records"][0][
        "minimum_realized_singular_value"
    ] *= 0.5
    singular["gauge_audit"]["max_singular_extrema_relative_error"] = 0.5
    singular["algebraic_gates"][
        "gauge_singular_extrema_relative_le_1e-12"
    ] = False
    with pytest.raises(RuntimeError, match="disagree with measurements"):
        summary.validate_protocol_and_measurements(singular, smoke=False)

    condition = _protocol_measurement_fixture()
    changed_minimum = 1.0 / math.sqrt(1.1)
    changed_maximum = math.sqrt(1.1)
    condition["gauge_audit"]["records"][0]["bond_realizations"][0][
        "minimum_singular_value"
    ] = changed_minimum
    condition["gauge_audit"]["records"][0]["bond_realizations"][0][
        "maximum_singular_value"
    ] = changed_maximum
    condition["gauge_audit"]["records"][0]["bond_realizations"][0][
        "condition"
    ] = changed_maximum / changed_minimum
    condition["gauge_audit"]["records"][0][
        "minimum_realized_singular_value"
    ] = changed_minimum
    condition["gauge_audit"]["records"][0][
        "maximum_realized_singular_value"
    ] = changed_maximum
    changed_condition = changed_maximum / changed_minimum
    condition["gauge_audit"]["records"][0][
        "maximum_realized_condition"
    ] = changed_condition
    condition["gauge_audit"]["max_condition_relative_error"] = abs(
        changed_condition - 1.0
    )
    condition["gauge_audit"]["max_singular_extrema_relative_error"] = max(
        abs(changed_minimum - 1.0), abs(changed_maximum - 1.0)
    )
    condition["algebraic_gates"]["gauge_condition_relative_le_1e-12"] = False
    condition["algebraic_gates"][
        "gauge_singular_extrema_relative_le_1e-12"
    ] = False
    with pytest.raises(RuntimeError, match="disagree with measurements"):
        summary.validate_protocol_and_measurements(condition, smoke=False)

    nonsymmetric = _protocol_measurement_fixture()
    nonsymmetric["gauge_audit"]["records"][0]["bond_realizations"][0][
        "nonsymmetry_relative"
    ] = 1e-9
    nonsymmetric["gauge_audit"]["records"][0][
        "minimum_realized_nonsymmetry_relative"
    ] = 1e-9
    nonsymmetric["gauge_audit"]["minimum_nonsymmetry_relative"] = 1e-9
    nonsymmetric["algebraic_gates"][
        "gauge_nonsymmetry_relative_gt_1e-6"
    ] = False
    with pytest.raises(RuntimeError, match="disagree with measurements"):
        summary.validate_protocol_and_measurements(nonsymmetric, smoke=False)


def test_selected_rank_and_eigengap_are_recomputed():
    nonminimal = _protocol_measurement_fixture()
    endpoint = nonminimal["discovery_selected_confirmation"]["canonical_tree_odt"]
    endpoint.update(
        {
            "rank": 2,
            "effective_affine_rank": 2,
            "discovery_accuracy": 0.8,
            "confirmation_accuracy": 0.8,
            "eigengap_resolved": True,
        }
    )
    with pytest.raises(RuntimeError, match="frozen discovery rule"):
        summary.validate_protocol_and_measurements(nonminimal, smoke=False)

    fabricated = _protocol_measurement_fixture()
    fabricated["canonical_eigengap_diagnostics"]["1"]["resolved"] = False
    with pytest.raises(RuntimeError, match="disagree with the canonical spectrum"):
        summary.validate_protocol_and_measurements(fabricated, smoke=False)


def test_canonicalization_inventory_and_trace_tails_are_exact():
    missing = _protocol_measurement_fixture()
    missing["canonicalization"].pop("symmetry_errors")
    with pytest.raises(RuntimeError, match="field inventory"):
        summary.validate_protocol_and_measurements(missing, smoke=False)

    wrong_length = _protocol_measurement_fixture()
    wrong_length["canonicalization"]["factorization_errors"].pop()
    with pytest.raises(RuntimeError, match="error inventory"):
        summary.validate_protocol_and_measurements(wrong_length, smoke=False)

    incomplete_reconstruction = _protocol_measurement_fixture()
    incomplete_reconstruction["canonicalization"]["sampled_logit_reconstruction"].pop(
        "full_rank_diagonalized"
    )
    with pytest.raises(RuntimeError, match="reconstruction inventory"):
        summary.validate_protocol_and_measurements(incomplete_reconstruction, smoke=False)

    wrong_tail = _protocol_measurement_fixture()
    wrong_tail["canonical_trace_tail_fraction"]["1"] += 1e-6
    with pytest.raises(RuntimeError, match="trace tails"):
        summary.validate_protocol_and_measurements(wrong_tail, smoke=False)


def test_output_paths_are_exactly_seed_scoped(tmp_path):
    project_root = tmp_path / "run"
    (project_root / "results").mkdir(parents=True)
    expected = project_root / "results" / "seed_2.json"
    assert runner.require_output_below_results(expected, project_root, "seed_2.json") == expected
    with pytest.raises(RuntimeError, match="outside this run root"):
        runner.require_output_below_results(
            tmp_path / "elsewhere" / "seed_2.json", project_root, "seed_2.json"
        )


def test_smoke_is_a_full_trained_gauge_preflight():
    text = Path("athena/slurm_svhn_canonical_odt_smoke.sbatch").read_text()
    assert "--epochs 20" in text
    assert "--gauge-trials 50" in text
    assert "--minimum-accuracy 0.70" in text


def test_reconstruction_is_explicitly_labeled_sampled():
    runner_text = Path("athena/run_svhn_canonical_odt.py").read_text()
    summary_text = Path("athena/summarize_svhn_canonical_odt.py").read_text()
    assert '"sampled_logit_reconstruction"' in runner_text
    assert '"exact_logit_reconstruction"' not in runner_text
    assert '"sampled_logit_reconstruction"' in summary_text


def test_checkpoint_homogeneous_export_loads_exact_frozen_rbn_state(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    config = summary.ChiMLPConfig(
        in_dim=3072,
        dim=32,
        n_layers=3,
        num_classes=10,
        norm="scalar_rbn",
    )
    torch.manual_seed(90210)
    model = summary.ChiMLP(config).double().eval()
    for index, norm in enumerate(model.norms):
        norm.running_rms.fill_(1.25 + index)
        norm.initialized.fill_(True)
        norm.freeze()
    torch.save(model.state_dict(), results / "seed_0.pt")
    expected = summary.export_homogeneous_network(
        model, require_frozen_rbn=True
    )
    actual = summary.export_checkpoint_homogeneous(
        {"seed": 0}, project_root=tmp_path
    )
    for expected_tensor, actual_tensor in zip(
        (expected.embedding, *expected.cores, expected.head),
        (actual.embedding, *actual.cores, actual.head),
    ):
        assert torch.equal(actual_tensor, expected_tensor)


def test_replay_sample_regeneration_uses_frozen_indices_and_svhn_transform(
    tmp_path, monkeypatch
):
    class FakeSVHN:
        def __init__(self, root, split, download, transform):
            assert Path(root) == tmp_path
            assert split == "test"
            assert download is False
            self.transform = transform

        def __len__(self):
            return 26032

        def __getitem__(self, index):
            image = np.full((32, 32, 3), index % 256, dtype=np.uint8)
            return self.transform(image), 0

    monkeypatch.setattr(summary.datasets, "SVHN", FakeSVHN)
    sample = summary.regenerate_gauge_replay_sample(tmp_path)
    generator = torch.Generator().manual_seed(20260902)
    indices = torch.randperm(26032, generator=generator)[:16].tolist()
    assert sample.shape == (16, 3072)
    means = torch.tensor((0.4377, 0.4438, 0.4728), dtype=torch.float32)
    standard_deviations = torch.tensor(
        (0.198, 0.201, 0.197), dtype=torch.float32
    )
    for row, index in zip(sample, indices):
        pixel = torch.tensor((index % 256) / 255.0, dtype=torch.float32)
        expected_channels = (pixel - means) / standard_deviations
        expected = expected_channels[:, None, None].expand(3, 32, 32).reshape(-1)
        assert torch.equal(row, expected.double())


def _gauge_npz_fixture(tmp_path: Path):
    generator = torch.Generator().manual_seed(9917)
    embedding = torch.randn(33, 3073, generator=generator, dtype=torch.float64) * 0.01
    embedding[0].zero_()
    embedding[0, 0] = 1.0
    cores = []
    for _ in range(3):
        core = torch.randn(33, 33, 33, generator=generator, dtype=torch.float64) * 0.01
        cores.append(0.5 * (core + core.transpose(1, 2)))
    head = torch.randn(10, 33, generator=generator, dtype=torch.float64) * 0.01
    head[:, 0] += 1.0
    raw = runner.HomogeneousChiTN(embedding, tuple(cores), head)
    canonical = runner.canonicalize_homogeneous(raw)
    systems = [runner.sorted_eigensystem(gram) for gram in runner.canonical_environments(canonical.network)]
    resolved = [
        [
            rank for rank in range(1, 33)
            if runner.eigengap_diagnostics(values, rank).resolved
        ]
        for values, _ in systems
    ]
    assert all(ranks for ranks in resolved)
    required = tuple((ranks[0],) for ranks in resolved)
    sample = torch.randn(16, 3072, generator=generator, dtype=torch.float64)
    records, matrices = runner.learned_gauge_audit(
        raw, canonical, sample, trials=1, required_ranks_by_bond=required
    )
    path = tmp_path / "seed_0_gauge_matrices.npz"

    def write_bundle(values):
        np.savez_compressed(path, **values)
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
            "allow_pickle": False,
            "keys": {
                key: {"dtype": str(value.dtype), "shape": list(value.shape)}
                for key, value in sorted(values.items())
            },
        }

    record = {
        "protocol": {
            "gauge_trials": 1,
            "gauge_projector_roundoff_multiplier": 10_000.0,
            "gauge_required_projector_maximum_bound": 1e-6,
        },
        "gauge_audit": {
            "base_spectra_by_bond": [
                [float(value) for value in values] for values, _ in systems
            ],
            "resolved_ranks_by_bond": resolved,
            "required_ranks_by_bond": [list(ranks) for ranks in required],
            "records": records,
            "matrix_artifact": write_bundle(matrices),
        },
    }
    return record, path, matrices, write_bundle


def _validate_gauge_fixture(record, path, matrices, project_root, **kwargs):
    expected_raw = runner.HomogeneousChiTN(
        torch.from_numpy(matrices["raw_network_embedding"].copy()),
        tuple(
            torch.from_numpy(matrices[f"raw_network_core_{index}"].copy())
            for index in range(3)
        ),
        torch.from_numpy(matrices["raw_network_head"].copy()),
    )
    return summary.validate_gauge_matrix_artifact(
        record,
        project_root=project_root,
        expected_path=path,
        expected_raw_network=expected_raw,
        expected_replay_sample=torch.from_numpy(
            matrices["replay_sample_flat"].copy()
        ),
        **kwargs,
    )


def test_gauge_npz_is_independently_recomputed_and_tamper_evident(tmp_path):
    record, path, matrices, write_bundle = _gauge_npz_fixture(tmp_path)
    assert _validate_gauge_fixture(record, path, matrices, tmp_path) == path

    tampered = {key: value.copy() for key, value in matrices.items()}
    tampered["trial_0_bond_0_raw_gauge"][0, 0] += 0.1
    record["gauge_audit"]["matrix_artifact"] = write_bundle(tampered)
    with pytest.raises(RuntimeError, match="gauge"):
        _validate_gauge_fixture(record, path, matrices, tmp_path)


def test_gate_failed_npz_authentication_recomputes_without_requiring_true_ceiling(
    tmp_path,
):
    record, path, matrices, _ = _gauge_npz_fixture(tmp_path)
    required_bounds = [
        certificate["bound_with_roundoff"]
        for trial in record["gauge_audit"]["records"]
        for bond in trial["bond_realizations"]
        for certificate in bond["projector_certificates"]
        if certificate["required_for_certificate"] is True
    ]
    assert required_bounds and all(bound is not None and bound > 0 for bound in required_bounds)
    record["protocol"]["gauge_required_projector_maximum_bound"] = (
        min(required_bounds) / 2.0
    )
    with pytest.raises(
        RuntimeError, match="independently reconstructed required-projector gate failed"
    ):
        _validate_gauge_fixture(record, path, matrices, tmp_path)
    assert _validate_gauge_fixture(
        record,
        path,
        matrices,
        tmp_path,
        require_all_matrix_gates=False,
    ) == path


def test_gauge_npz_rejects_certificate_and_polar_transport_tampering(tmp_path):
    record, path, matrices, write_bundle = _gauge_npz_fixture(tmp_path)
    certificate = record["gauge_audit"]["records"][0]["bond_realizations"][0][
        "projector_certificates"
    ][0]
    certificate["eta"] += 1e-4
    with pytest.raises(RuntimeError, match="internally inconsistent"):
        _validate_gauge_fixture(record, path, matrices, tmp_path)

    polar_root = tmp_path / "polar"
    polar_root.mkdir()
    record, path, matrices, write_bundle = _gauge_npz_fixture(polar_root)
    tampered = {key: value.copy() for key, value in matrices.items()}
    tampered["trial_0_bond_0_transport"][0, 0] += 1e-3
    record["gauge_audit"]["matrix_artifact"] = write_bundle(tampered)
    with pytest.raises(RuntimeError, match="polar factor"):
        _validate_gauge_fixture(record, path, matrices, path.parent)


def test_gauge_npz_rejects_self_consistent_overlap_and_candidate_gram_fork(tmp_path):
    """A mutually consistent stored fork must still bind to reconstructed weights."""

    record, path, matrices, write_bundle = _gauge_npz_fixture(tmp_path)
    tampered = {key: value.copy() for key, value in matrices.items()}
    orthogonal_fork = -np.eye(33, dtype=np.float64)
    base_gram = tampered["base_gram_bond_0"]
    tampered["trial_0_bond_0_overlap"] = orthogonal_fork
    tampered["trial_0_bond_0_transport"] = orthogonal_fork
    tampered["trial_0_bond_0_candidate_gram"] = (
        orthogonal_fork @ base_gram @ orthogonal_fork.T
    )
    entry = record["gauge_audit"]["records"][0]["bond_realizations"][0]
    entry.update(
        {
            "prefix_overlap_minimum_singular_value": 1.0,
            "prefix_overlap_maximum_singular_value": 1.0,
            "prefix_overlap_span_defect": 0.0,
            "transport_orthogonality": 0.0,
            "gram_covariance_relative": 0.0,
            "spectrum_relative": 0.0,
        }
    )
    record["gauge_audit"]["matrix_artifact"] = write_bundle(tampered)

    with pytest.raises(RuntimeError, match="independently reconstructed"):
        _validate_gauge_fixture(record, path, matrices, tmp_path)


@pytest.mark.parametrize("root_key", ["raw_network_embedding", "replay_sample_flat"])
def test_gauge_npz_roots_are_bound_to_checkpoint_export_and_frozen_data(
    tmp_path, root_key
):
    record, path, matrices, write_bundle = _gauge_npz_fixture(tmp_path)
    tampered = {key: value.copy() for key, value in matrices.items()}
    tampered[root_key].flat[0] += 1.0
    record["gauge_audit"]["matrix_artifact"] = write_bundle(tampered)
    expected_message = (
        "not derived from the checkpoint"
        if root_key == "raw_network_embedding"
        else "not the regenerated frozen discovery sample"
    )
    with pytest.raises(RuntimeError, match=expected_message):
        _validate_gauge_fixture(record, path, matrices, tmp_path)


def test_gauge_npz_raw_gauges_are_bound_to_frozen_generator_stream(tmp_path):
    record, path, matrices, write_bundle = _gauge_npz_fixture(tmp_path)
    tampered = {key: value.copy() for key, value in matrices.items()}
    key = "trial_0_bond_0_raw_gauge"
    tampered[key] = tampered[key].T.copy()
    record["gauge_audit"]["matrix_artifact"] = write_bundle(tampered)
    with pytest.raises(RuntimeError, match="raw gauge"):
        _validate_gauge_fixture(record, path, matrices, tmp_path)


@pytest.mark.parametrize(
    "field",
    ["raw_coordinate_replay_relative", "canonicalized_function_relative"],
)
def test_non_load_bearing_trial_replays_report_both_paths_without_scalar_gate(
    tmp_path, field
):
    record, path, matrices, _ = _gauge_npz_fixture(tmp_path)
    record["gauge_audit"]["records"][0][field] += 1e-4
    _, authentication = _validate_gauge_fixture(
        record, path, matrices, tmp_path, include_authentication_details=True
    )
    diagnostics = authentication["non_load_bearing_conditioning_diagnostics"]
    first = diagnostics["records"][0]
    assert first["producer"][field] == record["gauge_audit"]["records"][0][field]
    assert first["independent_checkpoint_npz_cpu"][field] != first["producer"][field]
    assert diagnostics["worst_of_both_maxima"][field] >= first["producer"][field]


def _synthetic_projector_certificate(*, gap, eta, required, arithmetic_slack):
    separation = gap - 2.0 * eta
    bound = math.sqrt(2.0) * eta / separation
    distance = bound / 2.0
    return {
        "rank": 31,
        "gap": gap,
        "covariance_residual_operator": eta / 4.0,
        "base_eigensolver_residual_operator": eta / 4.0,
        "candidate_eigensolver_residual_operator": eta / 4.0,
        "base_eigenvector_orthogonality": 0.0,
        "candidate_eigenvector_orthogonality": 0.0,
        "transport_orthogonality": 0.0,
        "roundoff_allowance": eta / 4.0,
        "eta": eta,
        "separation_margin": separation,
        "numerically_certifiable": True,
        "normalized_projector_distance": distance,
        "davis_kahan_bound": bound,
        "bound_with_roundoff": bound + arithmetic_slack,
        "passes_bound": True,
        "base_supported_overlap_defect": 1e-14,
        "candidate_supported_overlap_defect": 2e-14,
        "polar_replacement_supported_defect": 3e-14,
        "maximum_supported_overlap_defect": 3e-14,
        "required_for_certificate": required,
    }


def test_gpu_cpu_near_degenerate_diagnostic_is_conditioned_but_not_load_bearing():
    """Reproduce r7's rank-31 amplification without weakening required ranks."""

    arithmetic_slack = 10_000.0 * float(np.finfo(np.float64).eps) * 33
    reported = _synthetic_projector_certificate(
        gap=1.0624617079776385e-5,
        eta=1.8720747447351649e-10,
        required=False,
        arithmetic_slack=arithmetic_slack,
    )
    cpu = _synthetic_projector_certificate(
        gap=1.0624617079778072e-5,
        eta=1.87116427118161e-10,
        required=False,
        arithmetic_slack=arithmetic_slack,
    )
    assert abs(reported["davis_kahan_bound"] - cpu["davis_kahan_bound"]) > 1e-8
    assert summary._projector_bound_conditioning_tolerance(reported, cpu) > abs(
        reported["davis_kahan_bound"] - cpu["davis_kahan_bound"]
    )
    # A realized projector distance is not a rank-selection scalar.  Distinct
    # GPU/CPU eigensolver paths can legitimately produce two values anywhere
    # inside their own Davis--Kahan envelopes.  Both copies still have to
    # satisfy the inequality and agree on the boolean decision.
    reported["normalized_projector_distance"] = reported["bound_with_roundoff"]
    cpu["normalized_projector_distance"] = 0.0
    assert summary._nonrequired_projector_realization_tolerance(reported, cpu) >= abs(
        reported["normalized_projector_distance"] - cpu["normalized_projector_distance"]
    )
    summary._compare_projector_certificates(
        reported,
        cpu,
        arithmetic_slack=arithmetic_slack,
        label="r7 GPU/CPU non-required rank-31 diagnostic",
        required_scalar_policy="independent_gate_replication",
    )

    required_reported = dict(reported, required_for_certificate=True)
    required_cpu = dict(cpu, required_for_certificate=True)
    with pytest.raises(RuntimeError, match="required .* disagrees"):
        summary._compare_projector_certificates(
            required_reported,
            required_cpu,
            arithmetic_slack=arithmetic_slack,
            label="required near-degenerate certificate",
            required_scalar_policy="strict_stored_npz",
        )

    tampered = dict(reported, davis_kahan_bound=reported["davis_kahan_bound"] + 1e-5)
    with pytest.raises(RuntimeError, match="internally inconsistent"):
        summary._compare_projector_certificates(
            tampered,
            cpu,
            arithmetic_slack=arithmetic_slack,
            label="tampered non-required diagnostic",
            required_scalar_policy="independent_gate_replication",
        )

    shifted_eta = reported["eta"] + 9e-9
    shifted_separation = reported["gap"] - 2.0 * shifted_eta
    shifted_bound = math.sqrt(2.0) * shifted_eta / shifted_separation
    coordinated_tamper = dict(
        reported,
        eta=shifted_eta,
        separation_margin=shifted_separation,
        davis_kahan_bound=shifted_bound,
        bound_with_roundoff=shifted_bound + arithmetic_slack,
        normalized_projector_distance=shifted_bound / 2.0,
    )
    with pytest.raises(RuntimeError, match="eta is internally inconsistent"):
        summary._compare_projector_certificates(
            coordinated_tamper,
            cpu,
            arithmetic_slack=arithmetic_slack,
            label="coordinated non-required eta tamper",
            required_scalar_policy="independent_gate_replication",
        )

    overflowed = dict(reported, gap=1e308, eta=1e308, separation_margin=1e-308)
    with pytest.raises(RuntimeError, match="non-finite"):
        summary._projector_bound_conditioning_tolerance(overflowed, cpu)


def test_gpu_cpu_matrix_authentication_uses_strict_global_relative_error():
    expected = torch.eye(33, dtype=torch.float64)
    cpu = expected.clone()
    cpu[0, 1] = 3e-13
    summary._require_tensor_relative_close(
        cpu,
        expected,
        tolerance=summary.RAW_GAUGE_STREAM_RELATIVE_TOLERANCE,
        label="near-zero raw gauge coordinate",
    )
    reconstructed = expected.clone()
    reconstructed[0, 1] = 7e-8
    assert 1e-8 < summary._tensor_relative_error(reconstructed, expected) < 5e-8
    summary._require_tensor_relative_close(
        reconstructed,
        expected,
        tolerance=summary.INDEPENDENT_MATRIX_RECONSTRUCTION_RELATIVE_TOLERANCE,
        label="r7-scale GPU/CPU reconstructed matrix",
    )
    with pytest.raises(RuntimeError, match="relative error"):
        summary._require_tensor_relative_close(
            cpu + 1e-5,
            expected,
            tolerance=summary.INDEPENDENT_MATRIX_RECONSTRUCTION_RELATIVE_TOLERANCE,
            label="tampered independently reconstructed matrix",
        )


def test_r8_required_certificates_use_three_way_gate_replication():
    arithmetic_slack = 10_000.0 * float(np.finfo(np.float64).eps) * 33
    producer = _synthetic_projector_certificate(
        gap=2.100561082800236,
        eta=2.2637554344575855e-9,
        required=True,
        arithmetic_slack=arithmetic_slack,
    )
    independent = _synthetic_projector_certificate(
        gap=2.100561082800235,
        eta=1.565013149356465e-9,
        required=True,
        arithmetic_slack=arithmetic_slack,
    )
    assert abs(producer["eta"] - independent["eta"]) > 1e-10
    summary._compare_projector_certificates(
        producer,
        independent,
        arithmetic_slack=arithmetic_slack,
        label="r8 path-dependent required rank-1 recomputation",
        required_scalar_policy="independent_gate_replication",
    )
    producer_decision = summary._required_certificate_decision(
        producer, maximum_bound=1e-6
    )
    independent_decision = summary._required_certificate_decision(
        independent, maximum_bound=1e-6
    )
    assert producer_decision["all_frozen_gates_pass"] is True
    assert independent_decision["all_frozen_gates_pass"] is True
    passing_copies = {
        "producer": producer_decision,
        "stored_npz_cpu": dict(producer_decision),
        "independent_checkpoint_raw_gauge_cpu": independent_decision,
    }
    summary._validate_required_certificate_copies(
        passing_copies, require_all_gates=True
    )

    failing_bound = _synthetic_projector_certificate(
        gap=2.100561082800236,
        eta=2e-6,
        required=True,
        arithmetic_slack=arithmetic_slack,
    )
    failing_decision = summary._required_certificate_decision(
        failing_bound, maximum_bound=1e-6
    )
    assert failing_decision["passes_davis_kahan_inequality"] is True
    assert failing_decision["bound_with_roundoff_le_maximum"] is False
    for failing_copy in passing_copies:
        copies = {key: dict(value) for key, value in passing_copies.items()}
        copies[failing_copy] = failing_decision
        with pytest.raises(RuntimeError, match="one required-projector"):
            summary._validate_required_certificate_copies(
                copies, require_all_gates=True
            )


def test_required_authentication_validator_binds_exact_expected_identities(tmp_path):
    record, path, matrices, _ = _gauge_npz_fixture(tmp_path)
    _, authentication = _validate_gauge_fixture(
        record, path, matrices, tmp_path, include_authentication_details=True
    )
    identities = tuple(
        (item["trial"], item["bond"], item["rank"])
        for item in authentication["decision_inventory"]
    )
    assert freeze.validate_required_authentication_decisions(
        authentication, expected_identities=identities
    ) == list(identities)
    for wrong in (
        identities[:-1],
        (*identities, (999, 0, 1)),
        ((identities[0][0], identities[0][1], identities[0][2] + 1), *identities[1:]),
    ):
        with pytest.raises(RuntimeError, match="frozen scope"):
            freeze.validate_required_authentication_decisions(
                authentication, expected_identities=tuple(wrong)
            )


def test_v3r4_keeps_all_resolved_diagnostics_and_strict_required_certificates(tmp_path):
    record, path, matrices, _ = _gauge_npz_fixture(tmp_path)
    certificates = [
        certificate
        for bond in record["gauge_audit"]["records"][0]["bond_realizations"]
        for certificate in bond["projector_certificates"]
    ]
    assert len(certificates) == sum(
        len(ranks) for ranks in record["gauge_audit"]["resolved_ranks_by_bond"]
    )
    assert any(certificate["required_for_certificate"] is False for certificate in certificates)
    assert any(certificate["required_for_certificate"] is True for certificate in certificates)
    validated_path, authentication = _validate_gauge_fixture(
        record,
        path,
        matrices,
        tmp_path,
        include_authentication_details=True,
    )
    assert validated_path == path
    assert authentication["required_certificate_count"] == sum(
        len(ranks) for ranks in record["gauge_audit"]["required_ranks_by_bond"]
    )
    assert set(authentication["copies"]) == {
        "producer",
        "stored_npz_cpu",
        "independent_checkpoint_raw_gauge_cpu",
    }
    assert authentication["worst_of_all_copies"]["all_frozen_gates_pass"] is True
    assert len(authentication["decision_inventory"]) == (
        authentication["required_certificate_count"]
    )

    diagnostic_only = json.loads(json.dumps(record["gauge_audit"]))
    nonrequired = next(
        certificate
        for bond in diagnostic_only["records"][0]["bond_realizations"]
        for certificate in bond["projector_certificates"]
        if certificate["required_for_certificate"] is False
        and certificate["bound_with_roundoff"] is not None
    )
    nonrequired["normalized_projector_distance"] = (
        2.0 * nonrequired["bound_with_roundoff"]
    )
    nonrequired["passes_bound"] = False
    independently_reduced = summary.recompute_gauge_audit(diagnostic_only, 1)
    assert independently_reduced["all_required_projectors_pass_bound"] is True


def test_v3r4_all_full_rank_fallback_authenticates_without_vacuous_pass(tmp_path):
    record, path, matrices, _ = _gauge_npz_fixture(tmp_path)
    record["gauge_audit"]["required_ranks_by_bond"] = [[33] for _ in range(4)]
    for bond in record["gauge_audit"]["records"][0]["bond_realizations"]:
        for certificate in bond["projector_certificates"]:
            certificate["required_for_certificate"] = False
    validated_path, authentication = _validate_gauge_fixture(
        record,
        path,
        matrices,
        tmp_path,
        require_all_matrix_gates=False,
        include_authentication_details=True,
    )
    assert validated_path == path
    assert authentication["required_certificate_count"] == 0
    assert authentication["expected_required_certificate_count"] == 0
    assert authentication["decision_inventory"] == []
    assert authentication["worst_of_all_copies"] == {
        "maximum_bound_with_roundoff": None,
        "maximum_supported_overlap_defect": None,
        "all_frozen_gates_pass": False,
    }
    assert all(
        copy["certificate_count"] == 0
        and copy["all_frozen_gates_pass"] is False
        for copy in authentication["copies"].values()
    )
