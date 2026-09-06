"""Three-seed-ready SVHN evaluation for canonical homogeneous tree ODT.

The runner is native to Athena and intentionally separate from the historical
Modal experiment.  It compares topology ODT with local, legacy, PCA, and Haar
controls on fixed discovery and confirmation panels.  Its primary compression
record applies the complete Algorithm-3 projector tuple to all four bonds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as transforms

from xvla.models.chi_mlp import ChiMLP, ChiMLPConfig
from xvla.nn.normalization import RmsBatchNorm
from xvla.train.calibrate import calibrate_rbn_sequential
from xvla.train.canonical_odt import (
    HomogeneousChiTN,
    apply_bond_gauge,
    canonical_environments,
    canonical_prefix_overlaps,
    canonicalize_homogeneous,
    coefficient_error_squared,
    coefficient_inner_product,
    compress_homogeneous,
    discarded_trace,
    eigengap_diagnostics,
    export_homogeneous_network,
    hierarchical_tail_bound_squared,
    homogeneous_forward,
    local_environments,
    polar_orthogonal_transport,
    projector_perturbation_certificate,
    sorted_eigensystem,
    top_bases,
)
from xvla.train.odt import downstream_gram, export_cores, unroll_forward
try:
    from athena.verify_canonical_odt_freeze import (
        RESULT_SOURCE_PATHS,
        SMOKE_REQUIRED_AUTHENTICATION_IDENTITIES,
        UNIT_CERTIFICATE_SCHEMA,
        require_confined_path,
        require_physical_root,
        validate_required_authentication_decisions,
        verify as verify_source_freeze,
    )
except ModuleNotFoundError:
    from verify_canonical_odt_freeze import (
        RESULT_SOURCE_PATHS,
        SMOKE_REQUIRED_AUTHENTICATION_IDENTITIES,
        UNIT_CERTIFICATE_SCHEMA,
        require_confined_path,
        require_physical_root,
        validate_required_authentication_decisions,
        verify as verify_source_freeze,
    )


SCHEMA = "xvla-canonical-tree-odt-svhn-extension-v3r5"
PREFETCH_SCHEMA = "xvla-canonical-tree-odt-svhn-prefetch-v3"
LAUNCH_SCHEMA = "xvla-canonical-odt-athena-launch-v4"
DAG_SCHEMA = "xvla-canonical-odt-athena-dag-v4"
SMOKE_VALIDATION_SCHEMA = "xvla-canonical-tree-odt-smoke-validation-v1"
SPLIT_SEED = 20260902
GAUGE_SEED = 2026090200
GAUGE_REPLAY_SAMPLE_COUNT = 16
SELECTION_ACCURACY_TOLERANCE = 0.01
ALL_BOND_TARGET_REMOVAL_BASIS_POINTS = (
    0,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    9500,
    9700,
)
ALL_BOND_EXPECTED_DIMENSIONS = (33, 33, 33, 33)
ALL_BOND_EXPECTED_UNIFORM_RANKS = (33, 30, 27, 24, 20, 17, 14, 10, 7, 4, 2, 1)
ALL_BOND_HAAR_DRAWS = 64
ALL_BOND_HAAR_SEED = 700_000
ALL_BOND_BOUND_RELATIVE_SLACK = 1e-8
ALL_BOND_BOUND_ABSOLUTE_SLACK_FRACTION = 1e-10
GAUGE_CERTIFICATE_MAX_BOUND = 1e-6
GAUGE_ROUNDOFF_MULTIPLIER = 10_000.0
GAUGE_SELECTION_TRIALS = 0
GAUGE_VALIDATION_SEED = GAUGE_SEED
V1_PREDECESSOR_FAILURE = {
    "schema": "xvla-canonical-tree-odt-svhn-predecessor-failure-v1",
    "job_id": "834390",
    "result_schema": "xvla-canonical-tree-odt-svhn-v1",
    "immutable_root": "/work/joy/x-vla-odt-20260902-canonicalr1",
    "source_manifest_sha256": (
        "6497030d8f9b5b4848c37f5a64980fe22c6702102b3a2ce0105435c6da6609a3"
    ),
    "result_path": (
        "/work/joy/x-vla-odt-20260902-canonicalr1/results/smoke.json"
    ),
    "result_sha256": (
        "652a63705522e63a33f1b257bb44b979c8bff3baccc9dafbe6a38cfe7d5ecc20"
    ),
    "failed_gate": "gauge_function_relative_le_1e-8",
    "observed_relative_error": 4.201135487728006e-8,
    "threshold": 1e-8,
}
V2R3_PREDECESSOR_FAILURE = {
    "schema": "xvla-canonical-tree-odt-svhn-predecessor-failure-v2r3",
    "job_id": "834408",
    "result_schema": "xvla-canonical-tree-odt-svhn-v2",
    "immutable_root": "/work/joy/x-vla-odt-20260902-canonicalv2r3",
    "source_manifest_sha256": (
        "ef5d045356d3ac52f8455c69e66a663c93f52b50da68708bdadfcac4a185802a"
    ),
    "result_path": "/work/joy/x-vla-odt-20260902-canonicalv2r3/results/smoke.json",
    "result_sha256": (
        "c8a249a13023422e0a197a0e417ac3c8b09dfa416afe3754586c9f15367a69e5"
    ),
    "failed_gate": "gauge_resolved_projector_relative_le_1e-8",
    "observed_relative_error": 1.1474492274034916e-7,
    "threshold": 1e-8,
    "empirical_positive_control_failure": (
        "the discovery-selected uniform all-bond point retained full rank, so it "
        "did not establish a nontrivial ODT-over-local compression result"
    ),
}
V3R1_PREDECESSOR_FAILURE = {
    "schema": "xvla-canonical-tree-odt-svhn-predecessor-failure-v3r1",
    "immutable_root": "/work/joy/x-vla-canonical-v3r1-production-r4",
    "source_manifest_sha256": (
        "0e2ac659f7efd85fa8ce98b318aff6b0b364fe4644dafbade46001e5523b9a76"
    ),
    "launch_sha256": (
        "d48faa50a705073e7ee9bb0e0bc3b43cc4db55e39f904ac2535d3494399c659b"
    ),
    "job_ids": {
        "tests": "834518",
        "prefetch": "834519",
        "smoke": "834520",
        "train_array": "834521",
        "summary": "834522",
    },
    "passing_seed_count": 1,
    "seed_status": {"0": "gate_failed", "1": "complete", "2": "gate_failed"},
    "seed_result_sha256": {
        "0": "99b1a5a74fc766ac9d318ee2085754a0276c41656500abd1b2b8b678b1480bd1",
        "1": "c2fde01690dc03dfb25138baf449ff2d2e67cfec53db221f21d1e454c119730e",
        "2": "7e8aacd8611c622620eee005aa78f930cde0af045c1dcbd4ceb881f5bf6d7310",
    },
    "false_gate_by_seed": {
        "0": ["gauge_required_projectors_pass_nonvacuous_bound"],
        "1": [],
        "2": ["gauge_required_projectors_pass_nonvacuous_bound"],
    },
    "maximum_bound_with_roundoff_by_seed": {
        "0": 0.07145040829803825,
        "1": 5.781123711059455e-7,
        "2": 0.012674291232999077,
    },
    "seed_0_has_nonfull_accuracy_and_bound_eligible_rank": False,
    "summary_failure_sha256": (
        "4ae60f3e84c0077a41a4cb906e02f9f532018aa7079f0dd3e7c707466ef4f348"
    ),
}
V3R2_R5_PREDECESSOR_FAILURE = {
    "schema": "xvla-canonical-tree-odt-svhn-predecessor-failure-v3r2-r5",
    "immutable_root": "/work/joy/x-vla-canonical-v3r2-production-r5",
    "source_manifest_sha256": (
        "1a4cbce71b09a8550a52fdf11e1c0c224c694d7c2e7b09a6d58acc0dd3c0248b"
    ),
    "launch_sha256": (
        "82065d24b8c71352723ece635d4753643d195ed82c669a3d4c2a9bae345fdb2c"
    ),
    "job_ids": {
        "tests": "834531", "prefetch": "834532", "smoke": "834533",
        "train_array": "834534", "summary": "834535",
    },
    "classification": "engineering_preflight_failure_no_science",
    "unit_test_result": "131 passed, 58 failed",
    "failure": "NumPy float64 escaped the finite-JSON test fixture",
    "unit_test_log_sha256": (
        "72eb9a329411dc071367596dc022a758482670dcdbc496e88affddebdc72e90f"
    ),
    "summary_failure_sha256": (
        "0e373e83fa1577c526e76fb575de4c05325c088572f1de5ae367704e53d49f55"
    ),
}
V3R2_R6_PREDECESSOR_FAILURE = {
    "schema": "xvla-canonical-tree-odt-svhn-predecessor-failure-v3r2-r6",
    "immutable_root": "/work/joy/x-vla-canonical-v3r2-production-r6",
    "source_manifest_sha256": (
        "4bf4c30fab0926fd33b890ca189d299e18eb87346031036c5c58a2066d16baca"
    ),
    "launch_sha256": (
        "e16e8aea18e24b90573f83b7cc794f4d5fd6f36e41cd72f303678c26040b4e24"
    ),
    "job_ids": {
        "tests": "834538", "prefetch": "834539", "smoke": "834540",
        "train_array": "834541", "summary": "834542",
    },
    "classification": "engineering_preflight_failure_no_science",
    "unit_test_result": "188 passed, 1 failed",
    "failure": "four fail-closed protocol cases ran inside a monkeypatch scope",
    "unit_test_log_sha256": (
        "74619cb2151cafc2ed7ab6c07bef1e38fa0961cba74f4648d67a02e0501232f1"
    ),
    "summary_failure_sha256": (
        "caaa3633eb240fb5625b85138fb4e62c9f2a7dc2cb3852b79df55474de7e16c2"
    ),
}
V3R2_R7_PREDECESSOR_FAILURE = {
    "schema": "xvla-canonical-tree-odt-svhn-predecessor-failure-v3r2-r7",
    "immutable_root": "/work/joy/x-vla-canonical-v3r2-production-r7",
    "source_manifest_sha256": (
        "756427412047740f5411c0dfc373894fd071322a6995335ed4a62636b19e9f95"
    ),
    "launch_sha256": (
        "e71b7fa16b3e2eb7d380fc437bd90f467ed5048d8b7ebb45f17a667475bcee63"
    ),
    "job_ids": {
        "tests": "834545", "prefetch": "834546", "smoke": "834547",
        "train_array": "834548", "summary": "834549",
    },
    "classification": "engineering_authentication_failure_no_seed_science",
    "unit_test_result": "189 passed, 0 failed",
    "smoke_status": "complete",
    "smoke_json_sha256": (
        "8bb6c2d16b34b1993c5d65ced7047aeb3606c88a8c3a3d8d75817fad57d71d9f"
    ),
    "failure": (
        "CPU reconstruction rejected a non-required near-degenerate rank-31 "
        "Davis-Kahan diagnostic produced on GPU"
    ),
    "train_array_disposition": "cancelled_before_any_seed_result_completed",
    "summary_failure_sha256": (
        "456c85524a0e16925583c008052f776f9900b4dcb308d6fbaa01899b85539e81"
    ),
}
V3R3_R8_PREDECESSOR_FAILURE = {
    "schema": "xvla-canonical-tree-odt-svhn-predecessor-failure-v3r3-r8",
    "immutable_root": "/work/joy/x-vla-canonical-v3r3-production-r8",
    "source_manifest_sha256": (
        "47b994c6fe2661a007323f467835a892e43fd09cb144f2f3377844eabb7cdb20"
    ),
    "launch_sha256": (
        "925398f97a451190d9292b80b76a2aefb13e4437cc8b5ccb5b72210c9b07343b"
    ),
    "job_ids": {
        "tests": "834553", "prefetch": "834554", "smoke": "834555",
        "train_array": "834556", "summary": "834557",
    },
    "classification": "engineering_authentication_failure_no_seed_science",
    "unit_test_result": "192 passed, 0 failed",
    "smoke_status": "producer_complete_independent_authentication_failed",
    "smoke_json_sha256": (
        "478b5e543ce2a6e4b7d8b0541d88e7390472cb1a541d3789bceff526e9f8a885"
    ),
    "smoke_checkpoint_sha256": (
        "77d3609683865e4299f2811cc8e6bb7b5b3e4f97f234263cf3e0b07ed9bf571a"
    ),
    "smoke_gauge_npz_sha256": (
        "e8219293511951a9d539f216f1a20959286665f42cf0c448b9b3dd80a29f4b3c"
    ),
    "failure": (
        "strict equality of path-dependent required-rank residual magnitudes "
        "rejected an independently passing CPU reconstruction"
    ),
    "all_200_required_decisions_agree_and_pass": True,
    "maximum_required_bound_with_roundoff": {
        "producer": 1.5973597226370406e-9,
        "stored_npz_cpu": 1.5973578386563978e-9,
        "independent_cpu": 1.1269278804558074e-9,
    },
    "train_array_disposition": "cancelled_before_any_seed_result_completed",
    "summary_failure_sha256": (
        "21ea6a8a686bd707c198ce63553ed17e6627c33fd23e5007b138e92aef982cd0"
    ),
}
V3R4_R9_PREDECESSOR_FAILURE = {
    "schema": "xvla-canonical-tree-odt-svhn-predecessor-failure-v3r4-r9",
    "immutable_root": "/work/joy/x-vla-canonical-v3r4-production-r9",
    "source_manifest_sha256": (
        "45c159d6559c01ad008931b0a4ad7d08c144937ce74eddf3670ebd8133075e15"
    ),
    "launch_sha256": (
        "f7efc091a6f91dd36fee50b180db6ce8585726f5b72c97f941466323191b7798"
    ),
    "job_ids": {
        "tests": "834561", "prefetch": "834562", "smoke": "834563",
        "train_array": "834564", "summary": "834565",
    },
    "classification": "engineering_authentication_failure_no_seed_science",
    "unit_test_result": "194 passed, 0 failed",
    "smoke_status": "producer_complete_independent_authentication_failed",
    "smoke_json_sha256": (
        "f1c9d9a0eb0af33343b1950726eecad1a5a420ea51e286c07395fcc2678a53c9"
    ),
    "smoke_checkpoint_sha256": (
        "77d3609683865e4299f2811cc8e6bb7b5b3e4f97f234263cf3e0b07ed9bf571a"
    ),
    "smoke_gauge_npz_sha256": (
        "e8219293511951a9d539f216f1a20959286665f42cf0c448b9b3dd80a29f4b3c"
    ),
    "failure": (
        "cross-device scalar equality rejected two explicitly non-load-bearing "
        "simultaneous replay diagnostics although all 98,780 other comparisons and "
        "all 200 three-way required certificates passed"
    ),
    "login_cpu_mismatches": [
        {
            "trial": 49, "measurement": "raw_coordinate_replay_relative",
            "producer": 7.103200689715422e-8,
            "independent_cpu": 4.765571981717182e-8,
            "absolute_disagreement": 2.3376287079982402e-8,
        },
        {
            "trial": 47, "measurement": "canonicalized_function_relative",
            "producer": 3.587349506016325e-8,
            "independent_cpu": 1.594611819870762e-8,
            "absolute_disagreement": 1.992737686145563e-8,
        },
    ],
    "summary_compute_observation": (
        "an independent summary-node CPU path first disagreed at trial 43 raw replay"
    ),
    "all_200_required_decisions_agree_and_pass": True,
    "maximum_required_bound_with_roundoff": 1.5973597226370406e-9,
    "maximum_required_support_defect": 2.7873892849173025e-14,
    "train_array_disposition": "cancelled_before_any_seed_result_completed",
    "summary_failure_sha256": (
        "25dbf5a516cdfd7bdf68448b49ff076eb3ec911e04e9fac4053188007971cf83"
    ),
}
PREDECESSOR_FAILURES = (
    V1_PREDECESSOR_FAILURE,
    V2R3_PREDECESSOR_FAILURE,
    V3R1_PREDECESSOR_FAILURE,
    V3R2_R5_PREDECESSOR_FAILURE,
    V3R2_R6_PREDECESSOR_FAILURE,
    V3R2_R7_PREDECESSOR_FAILURE,
    V3R3_R8_PREDECESSOR_FAILURE,
    V3R4_R9_PREDECESSOR_FAILURE,
)
SOURCE_PATHS = RESULT_SOURCE_PATHS
SVHN_FILES = {
    "train_32x32.mat": {
        "bytes": 182040794,
        "md5": "e26dedcc434d2e4c54c9b2d4a06d8373",
        "sha256": "435e94d69a87fde4fd4d7f3dd208dfc32cb6ae8af2240d066de1df7508d083b8",
    },
    "test_32x32.mat": {
        "bytes": 64275384,
        "md5": "eb5a983be6a315427106f1b164d9cef3",
        "sha256": "cdce80dfb2a2c4c6160906d0bd7c68ec5a99d7ca4831afa54f09182025b6a75b",
    },
}
EXPECTED_ALGEBRAIC_GATES = {
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
    "selected_canonical_roundoff_only_prefilter_pass",
    "all_bond_uniform_schedule_exact",
    "all_bond_trace_totals_relative_le_1e-8",
    "all_bond_full_rank_coefficient_reconstruction_le_1e-8",
    "all_bond_selected_canonical_eigengaps_resolved",
    "all_bond_selected_canonical_roundoff_only_prefilter_pass",
    "all_bond_coefficient_errors_within_hsvd_bounds_plus_declared_numerical_slack",
}
EXPECTED_CAPABILITY_GATES = {
    "module_discovery_accuracy_ge_minimum",
    "module_confirmation_accuracy_ge_minimum",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate_athena_stage_binding(
    project_root: Path,
    data_root: Path,
    manifest_sha256: str,
    *,
    stage: str,
    seed: int | None,
) -> dict:
    """Bind a producer to the prospectively recorded Athena scheduler DAG."""

    launch_path = require_confined_path(
        project_root,
        project_root / "results" / "launch.json",
        label="launch record",
        kind="file",
    )
    launch = json.loads(launch_path.read_text())
    if set(launch) != {
        "schema", "source_manifest_sha256", "run_root", "data_root",
        "submission_protocol", "dag",
    } or (
        launch.get("schema") != LAUNCH_SCHEMA
        or launch.get("source_manifest_sha256") != manifest_sha256
        or launch.get("run_root") != str(project_root)
        or launch.get("data_root") != str(data_root)
        or launch.get("submission_protocol") != {
            "first_node_submitted_held": True,
            "launch_record_written_before_release": True,
        }
    ):
        raise RuntimeError("runner launch record identity differs")
    dag = launch.get("dag")
    order = ["tests", "prefetch", "smoke", "smoke_validate", "train_array", "summary"]
    expected_edges = [
        {"upstream": "tests", "downstream": "prefetch", "slurm_dependency": "afterany"},
        {"upstream": "prefetch", "downstream": "smoke", "slurm_dependency": "afterany"},
        {"upstream": "smoke", "downstream": "smoke_validate", "slurm_dependency": "afterany"},
        {"upstream": "smoke_validate", "downstream": "train_array", "slurm_dependency": "afterok"},
        {"upstream": "smoke_validate", "downstream": "summary", "slurm_dependency": "afternotok"},
        {"upstream": "train_array", "downstream": "summary", "slurm_dependency": "afterany"},
    ]
    if (
        not isinstance(dag, dict)
        or dag.get("schema") != DAG_SCHEMA
        or dag.get("node_order") != order
        or dag.get("terminal_node") != "summary"
        or dag.get("edges") != expected_edges
        or not isinstance(dag.get("nodes"), dict)
        or list(dag["nodes"]) != order
    ):
        raise RuntimeError("runner launch DAG identity differs")
    job_ids = [dag["nodes"][name].get("job_id") for name in order]
    if (
        any(not isinstance(job_id, str) or not job_id.isdecimal() for job_id in job_ids)
        or len(set(job_ids)) != len(job_ids)
    ):
        raise RuntimeError("runner launch DAG job ids are invalid")
    expected_terminal_expression = (
        f"afternotok:{dag['nodes']['smoke_validate']['job_id']}?"
        f"afterany:{dag['nodes']['train_array']['job_id']}"
    )
    if (
        set(dag) != {
            "schema", "node_order", "nodes", "edges", "terminal_node",
            "terminal_dependency_expression",
        }
        or dag.get("terminal_dependency_expression") != expected_terminal_expression
    ):
        raise RuntimeError("runner launch DAG terminal dependency differs")
    expected_specs = {
        "tests": ("athena/slurm_canonical_odt_tests.sbatch", [str(project_root)], None),
        "prefetch": (
            "athena/slurm_svhn_canonical_odt_prefetch.sbatch",
            [str(project_root), str(data_root)],
            None,
        ),
        "smoke": (
            "athena/slurm_svhn_canonical_odt_smoke.sbatch",
            [str(project_root), str(data_root)],
            None,
        ),
        "smoke_validate": (
            "athena/slurm_svhn_canonical_odt_smoke_validate.sbatch",
            [str(project_root)],
            None,
        ),
        "train_array": (
            "athena/slurm_svhn_canonical_odt.sbatch",
            [str(project_root), str(data_root)],
            "0-2",
        ),
        "summary": (
            "athena/slurm_svhn_canonical_odt_summary.sbatch",
            [str(project_root)],
            None,
        ),
    }
    for name, (script, arguments, array) in expected_specs.items():
        node = dag["nodes"][name]
        if set(node) != {"job_id", "script", "arguments", "array"} or (
            node["script"] != script
            or node["arguments"] != arguments
            or node["array"] != array
        ):
            raise RuntimeError(f"runner launch DAG node {name} specification differs")
    current_job = os.environ.get("SLURM_JOB_ID")
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    expected_job = dag["nodes"][stage]["job_id"]
    if stage == "train_array":
        if (
            type(seed) is not int
            or seed not in (0, 1, 2)
            or array_job != expected_job
            or array_task != str(seed)
            or not isinstance(current_job, str)
            or not current_job.isdecimal()
        ):
            raise RuntimeError("runner train-array Slurm binding differs")
    elif (
        current_job != expected_job
        or array_job is not None
        or array_task is not None
    ):
        raise RuntimeError(f"runner {stage} Slurm binding differs")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    constraint = os.environ.get("SLURM_JOB_CONSTRAINTS")
    if stage == "prefetch":
        if partition != "compute" or constraint not in (None, ""):
            raise RuntimeError("runner prefetch partition binding differs")
    elif partition != "gpu" or constraint not in (None, "", "a6000"):
        raise RuntimeError(f"runner {stage} GPU partition binding differs")
    return launch


def validate_unit_certificate(
    project_root: Path,
    frozen_entries: dict[str, str],
    manifest_sha256: str,
    launch: dict,
) -> dict:
    certificate_path = require_confined_path(
        project_root,
        project_root / "results" / "canonical_odt_unit_certificate.json",
        label="unit-test certificate",
        kind="file",
    )
    certificate = json.loads(certificate_path.read_text())
    if (
        certificate.get("schema") != UNIT_CERTIFICATE_SCHEMA
        or certificate.get("tests_passed") is not True
        or certificate.get("source_manifest_sha256") != manifest_sha256
        or certificate.get("source_entries") != frozen_entries
        or certificate.get("slurm_job_id")
        != launch["dag"]["nodes"]["tests"]["job_id"]
    ):
        raise RuntimeError("unit-test certificate does not match the frozen Athena launch")
    return certificate


def md5_digest(path: Path) -> str:
    value = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validated_svhn_manifest(root: Path) -> dict[str, dict[str, int | str]]:
    """Fail closed unless the two official SVHN archives are byte-identical."""

    result = {}
    for filename, expected in SVHN_FILES.items():
        path = root / filename
        if not path.is_file():
            raise RuntimeError(f"official SVHN file is missing: {path}")
        actual = {
            "bytes": path.stat().st_size,
            "md5": md5_digest(path),
            "sha256": digest(path),
        }
        if actual != expected:
            raise RuntimeError(
                f"official SVHN checksum mismatch for {filename}: {actual} != {expected}"
            )
        result[filename] = actual
    return result


def environment_provenance() -> dict:
    """Record a deterministic environment fingerprint plus scheduler context."""

    distributions = sorted(
        (
            (
                str(distribution.metadata.get("Name") or distribution.name or "UNKNOWN"),
                str(distribution.version),
            )
            for distribution in importlib.metadata.distributions()
        ),
        key=lambda item: (item[0].lower(), item[1]),
    )
    recorded_payload = {
        "python": platform.python_version(),
        "python_executable_invoked": os.path.abspath(sys.executable),
        "python_executable_resolved": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "distributions": distributions,
    }
    fingerprint_payload = {
        key: value for key, value in recorded_payload.items() if key != "platform"
    }
    encoded = json.dumps(
        fingerprint_payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        **recorded_payload,
        "fingerprint_sha256": hashlib.sha256(encoded).hexdigest(),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "scheduler": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "constraint": os.environ.get("SLURM_JOB_CONSTRAINTS"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        },
    }


def require_output_below_results(path: Path, project_root: Path, expected_name: str) -> Path:
    project_root = require_physical_root(project_root)
    expected_parent = require_confined_path(
        project_root,
        project_root / "results",
        label="results directory",
        kind="directory",
    )
    confined = require_confined_path(
        project_root,
        path,
        label=expected_name,
        allow_missing_leaf=True,
    )
    if confined.parent != expected_parent or confined.name != expected_name:
        raise RuntimeError(
            f"output must be exactly {expected_parent / expected_name}, got {confined}"
        )
    return confined


def make_datasets(root: Path, *, download: bool):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197)),
        ]
    )
    train = tv.datasets.SVHN(root, split="train", download=download, transform=transform)
    test = tv.datasets.SVHN(root, split="test", download=download, transform=transform)
    return train, test


def fixed_test_panels(test_dataset):
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    permutation = torch.randperm(len(test_dataset), generator=generator).tolist()
    discovery_indices = permutation[:4000]
    confirmation_indices = permutation[4000:]
    return (
        torch.utils.data.Subset(test_dataset, discovery_indices),
        torch.utils.data.Subset(test_dataset, confirmation_indices),
        discovery_indices,
        confirmation_indices,
    )


def load_panel(dataset, batch_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    images, labels = [], []
    for batch_images, batch_labels in loader:
        images.append(batch_images)
        labels.append(batch_labels)
    return torch.cat(images), torch.cat(labels)


@torch.no_grad()
def module_accuracy(model: ChiMLP, images: torch.Tensor, labels: torch.Tensor) -> float:
    correct = 0
    model_dtype = next(model.parameters()).dtype
    for start in range(0, len(images), 512):
        batch = images[start : start + 512].to(
            device="cuda", dtype=model_dtype, non_blocking=True
        )
        logits = model(batch)[0]
        correct += int((logits.argmax(-1).cpu() == labels[start : start + 512]).sum())
    return correct / len(images)


@torch.no_grad()
def network_accuracy(
    network: HomogeneousChiTN, images: torch.Tensor, labels: torch.Tensor
) -> float:
    correct = 0
    for start in range(0, len(images), 512):
        batch = images[start : start + 512].to(network.embedding.device, non_blocking=True)
        logits = homogeneous_forward(network, batch)
        correct += int((logits.argmax(-1).cpu() == labels[start : start + 512]).sum())
    return correct / len(images)


@torch.no_grad()
def sampled_reconstruction_diagnostics(
    model: ChiMLP,
    networks: dict[str, HomogeneousChiTN],
    images: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Compare logits, not just argmax agreement, on a frozen 256-image sample."""

    sample = images[:256].cuda(non_blocking=True).double()
    reference = model(sample)[0]
    reference_norm = torch.linalg.vector_norm(reference).clamp_min(1e-30)
    diagnostics = {}
    for name, network in networks.items():
        candidate = homogeneous_forward(network, sample)
        difference = candidate - reference
        diagnostics[name] = {
            "max_absolute_logit_error": float(difference.abs().max().item()),
            "relative_frobenius_logit_error": float(
                (torch.linalg.vector_norm(difference) / reference_norm).item()
            ),
        }
    return diagnostics


@torch.no_grad()
def legacy_accuracy(
    exported,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    bond: int,
    basis: torch.Tensor | None = None,
    center: torch.Tensor | None = None,
) -> float:
    embedding, cores, head = exported
    projector = None if basis is None else basis @ basis.T
    correct = 0
    for start in range(0, len(images), 512):
        batch = images[start : start + 512].to(embedding[0].device, non_blocking=True)
        if projector is None or center is None:
            logits = unroll_forward(
                embedding, cores, head, batch, proj=projector, proj_bond=bond
            )
        else:
            weight, bias = embedding
            state = batch.flatten(1).double() @ weight.T.double() + bias.double()
            for index, core in enumerate(cores, start=1):
                homogeneous = torch.cat(
                    [state.new_ones(state.shape[0], 1), state], dim=1
                )
                state = torch.einsum("oij,bi,bj->bo", core, homogeneous, homogeneous)
                if index == bond:
                    state = center + (state - center) @ projector.T
            head_weight, head_bias = head
            logits = state @ head_weight.T.double() + head_bias.double()
        correct += int((logits.argmax(-1).cpu() == labels[start : start + 512]).sum())
    return correct / len(images)


@torch.no_grad()
def collect_raw_bond_activations(
    exported, images: torch.Tensor, *, bond: int
) -> torch.Tensor:
    embedding, cores, _ = exported
    values = []
    for start in range(0, len(images), 512):
        batch = images[start : start + 512].to(embedding[0].device, non_blocking=True)
        weight, bias = embedding
        state = batch.flatten(1).double() @ weight.T.double() + bias.double()
        for index, core in enumerate(cores, start=1):
            homogeneous = torch.cat([state.new_ones(state.shape[0], 1), state], dim=1)
            state = torch.einsum("oij,bi,bj->bo", core, homogeneous, homogeneous)
            if index == bond:
                values.append(state.cpu())
                break
    return torch.cat(values)


def rank_grid(text: str, full_dimension: int) -> list[int]:
    parsed = sorted({int(value) for value in text.split(",") if value})
    parsed.append(full_dimension)
    result = sorted(set(parsed))
    if result[0] < 1 or result[-1] > full_dimension:
        raise ValueError(f"rank grid must lie within [1, {full_dimension}]")
    return result


def affine_baseline_rank_grid(ranks: list[int], learned_dimension: int) -> list[int]:
    """Include learned rank zero so effective affine rank one is representable."""

    return sorted({rank - 1 for rank in ranks if 1 <= rank <= learned_dimension + 1})


def canonical_curve(
    network: HomogeneousChiTN,
    vectors: torch.Tensor,
    ranks: list[int],
    discovery,
    confirmation,
    *,
    bond: int,
):
    result = {"discovery": {}, "confirmation": {}}
    for rank in ranks:
        compressed = compress_homogeneous(network, {bond: vectors[:, :rank]})
        result["discovery"][str(rank)] = network_accuracy(compressed, *discovery)
        result["confirmation"][str(rank)] = network_accuracy(compressed, *confirmation)
    return result


def legacy_curve(
    exported,
    vectors: torch.Tensor,
    ranks: list[int],
    discovery,
    confirmation,
    *,
    bond: int,
    center: torch.Tensor | None = None,
):
    result = {"discovery": {}, "confirmation": {}}
    for rank in ranks:
        basis = vectors[:, :rank]
        result["discovery"][str(rank)] = legacy_accuracy(
            exported, *discovery, bond=bond, basis=basis, center=center
        )
        result["confirmation"][str(rank)] = legacy_accuracy(
            exported, *confirmation, bond=bond, basis=basis, center=center
        )
    return result


def choose_rank(
    curve,
    reference: float,
    tolerance: float = SELECTION_ACCURACY_TOLERANCE,
    *,
    allowed_ranks: set[int] | None = None,
) -> int:
    eligible = [
        int(rank)
        for rank, accuracy in curve["discovery"].items()
        if accuracy >= reference - tolerance
        and (allowed_ranks is None or int(rank) in allowed_ranks)
    ]
    fallback = [
        int(rank) for rank in curve["discovery"]
        if allowed_ranks is None or int(rank) in allowed_ranks
    ]
    if not fallback:
        raise ValueError("no allowed operating rank is present in the curve")
    return min(eligible) if eligible else max(fallback)


def roundoff_only_relative_gap_floor(
    dimension: int,
    *,
    epsilon: float = torch.finfo(torch.float64).eps,
    roundoff_multiplier: float = GAUGE_ROUNDOFF_MULTIPLIER,
    maximum_bound: float = GAUGE_CERTIFICATE_MAX_BOUND,
) -> float:
    """Return the smallest relative gap that can satisfy the frozen envelope.

    This is a prospective, gauge-draw-free necessary condition.  It substitutes
    only the declared roundoff term into the exact frozen Davis-Kahan envelope:
    ``sqrt(2)*eta/(gap-2*eta) + arithmetic_slack <= maximum_bound``.
    It cannot make a rank pass held-out validation, but it prevents selection of
    a rank that is algebraically incapable of passing even before a gauge is
    drawn.
    """

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(roundoff_multiplier) or roundoff_multiplier < 0:
        raise ValueError("roundoff multiplier must be finite and nonnegative")
    if not math.isfinite(maximum_bound) or maximum_bound <= 0:
        raise ValueError("maximum bound must be finite and positive")
    arithmetic_slack = roundoff_multiplier * epsilon * dimension
    if maximum_bound <= arithmetic_slack:
        return math.inf
    return arithmetic_slack * (
        2.0 + math.sqrt(2.0) / (maximum_bound - arithmetic_slack)
    )


def roundoff_only_projector_prefilter(
    eigenvalues: torch.Tensor,
    rank: int,
    *,
    roundoff_multiplier: float = GAUGE_ROUNDOFF_MULTIPLIER,
    maximum_bound: float = GAUGE_CERTIFICATE_MAX_BOUND,
) -> dict:
    """Audit rank eligibility without observing any random validation gauge."""

    if eigenvalues.ndim != 1 or eigenvalues.numel() < 1:
        raise ValueError("eigenvalues must be a nonempty vector")
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("rank must be an integer")
    if not 1 <= rank <= eigenvalues.numel():
        raise ValueError("rank is outside the eigenspectrum")
    if not eigenvalues.is_floating_point() or not torch.isfinite(eigenvalues).all():
        raise ValueError("eigenvalues must be finite floating-point values")
    if torch.any(eigenvalues[:-1] < eigenvalues[1:]):
        raise ValueError("eigenvalues must be sorted in descending order")

    dimension = eigenvalues.numel()
    epsilon = torch.finfo(eigenvalues.dtype).eps
    relative_gap_floor = roundoff_only_relative_gap_floor(
        dimension,
        epsilon=epsilon,
        roundoff_multiplier=roundoff_multiplier,
        maximum_bound=maximum_bound,
    )
    if rank == dimension:
        return {
            "rank": rank,
            "kind": "full_rank_identity_fallback",
            "uses_random_gauge_draws": False,
            "relative_gap": None,
            "minimum_relative_gap_for_maximum_bound": relative_gap_floor,
            "roundoff_only_eta": None,
            "separation_margin": None,
            "bound_with_roundoff": None,
            "maximum_bound": maximum_bound,
            "passes": True,
        }

    scale = max(
        max(abs(float(value)) for value in eigenvalues),
        torch.finfo(eigenvalues.dtype).tiny,
    )
    gap = float(eigenvalues[rank - 1]) - float(eigenvalues[rank])
    relative_gap = gap / scale
    arithmetic_slack = roundoff_multiplier * epsilon * dimension
    eta = arithmetic_slack * scale
    separation = gap - 2.0 * eta
    bound = None
    if separation > 0:
        bound = math.sqrt(2.0) * eta / separation + arithmetic_slack
    passes = (
        relative_gap >= relative_gap_floor
        and bound is not None
        and bound <= maximum_bound
    )
    return {
        "rank": rank,
        "kind": "nontrivial_roundoff_only_necessary_condition",
        "uses_random_gauge_draws": False,
        "relative_gap": relative_gap,
        "minimum_relative_gap_for_maximum_bound": relative_gap_floor,
        "roundoff_only_eta": eta,
        "separation_margin": separation,
        "bound_with_roundoff": bound,
        "maximum_bound": maximum_bound,
        "passes": passes,
    }


def select_canonical_rank_prospectively(
    curve: dict,
    reference: float,
    eigenvalues: torch.Tensor,
) -> tuple[int, dict]:
    """Select using accuracy plus an analytic prefilter, never validation gauges."""

    rank_values = sorted(int(value) for value in curve["discovery"])
    if rank_values[-1] != eigenvalues.numel():
        raise ValueError("the rank curve must contain the full-rank endpoint")
    prefilter = {
        str(rank): roundoff_only_projector_prefilter(eigenvalues, rank)
        for rank in rank_values
    }
    allowed = {
        rank for rank in rank_values if prefilter[str(rank)]["passes"] is True
    }
    selected = choose_rank(curve, reference, allowed_ranks=allowed)
    minimum_accuracy = reference - SELECTION_ACCURACY_TOLERANCE
    accuracy_eligible = [
        rank
        for rank in rank_values
        if curve["discovery"][str(rank)] >= minimum_accuracy
    ]
    joint_nontrivial = [
        rank
        for rank in accuracy_eligible
        if rank < eigenvalues.numel() and rank in allowed
    ]
    outcome = (
        "nontrivial_rank_selected"
        if selected < eigenvalues.numel()
        else "full_rank_fallback_no_nontrivial_accuracy_and_prefilter_eligible_rank"
    )
    nontrivial_compression_selected = selected < eigenvalues.numel()
    return selected, {
        "policy": "analytic-roundoff-only-prefilter-v1",
        "selection_gauge_trial_count": GAUGE_SELECTION_TRIALS,
        "held_out_validation_gauge_seed": GAUGE_VALIDATION_SEED,
        "held_out_validation_gauge_trial_count": 50,
        "held_out_validation_gauges_used_for_selection": False,
        "minimum_discovery_accuracy": minimum_accuracy,
        "accuracy_eligible_ranks": accuracy_eligible,
        "roundoff_prefilter_eligible_ranks": sorted(allowed),
        "joint_eligible_nontrivial_ranks": joint_nontrivial,
        "prefilter_by_rank": prefilter,
        "selected_rank": selected,
        "nontrivial_compression_selected": nontrivial_compression_selected,
        "outcome": outcome,
    }


def uniform_all_bond_schedule(dimensions: tuple[int, ...]) -> list[dict]:
    """Return the frozen Figure-2-style schedule without using model outcomes.

    A target percentage is converted conservatively with a ceiling, so the
    realized curve never removes more coordinates than its nominal target.
    The Athena positive control intentionally requires four equal width-33
    bonds.  This makes the scalar horizontal axis unambiguous.  Algorithm 3
    itself permits nonuniform tuples, so this schedule remains an experimental
    convention rather than a joint-rank optimality claim.
    """

    if dimensions != ALL_BOND_EXPECTED_DIMENSIONS:
        raise RuntimeError(
            "the frozen uniform all-bond curve requires four width-33 bonds"
        )
    records = []
    for basis_points in ALL_BOND_TARGET_REMOVAL_BASIS_POINTS:
        retained_numerator = 10_000 - basis_points
        ranks = tuple(
            max(1, (retained_numerator * dimension + 9_999) // 10_000)
            for dimension in dimensions
        )
        records.append(
            {
                "target_removal_basis_points": basis_points,
                "rank_tuple": ranks,
            }
        )
    observed = tuple(record["rank_tuple"][0] for record in records)
    if (
        observed != ALL_BOND_EXPECTED_UNIFORM_RANKS
        or any(len(set(record["rank_tuple"])) != 1 for record in records)
        or any(
            records[index]["rank_tuple"][0]
            <= records[index + 1]["rank_tuple"][0]
            for index in range(len(records) - 1)
        )
    ):
        raise RuntimeError("the frozen uniform all-bond rank schedule changed")
    return records


def dense_homogeneous_tree_storage(
    *, leaf_dimension: int, output_dimension: int, ranks: tuple[int, ...]
) -> int:
    """Count scalars in the physically compressed dense homogeneous TN."""

    if not ranks or any(type(rank) is not int or rank < 1 for rank in ranks):
        raise ValueError("every dense-tree storage rank must be a positive integer")
    if leaf_dimension < 1 or output_dimension < 1:
        raise ValueError("leaf and output dimensions must be positive")
    total = ranks[0] * leaf_dimension
    for source, target in zip(ranks[:-1], ranks[1:]):
        total += target * source * source
    total += output_dimension * ranks[-1]
    return total


def select_resolved_accuracy_frontier(
    points: list[dict], reference_accuracy: float
) -> dict:
    """Select a robust frontier without exploiting later validation gauges.

    Unresolved or analytically impossible scheduled ranks are structural holes
    and are skipped.  The analytic eligibility field is computed without any
    random gauge draw.  Confirmation values and held-out gauge measurements are
    deliberately absent from this decision.
    """

    if not points:
        raise ValueError("the all-bond curve is empty")
    threshold = reference_accuracy - SELECTION_ACCURACY_TOLERANCE
    eligible_indices = [
        index
        for index, point in enumerate(points)
        if point["canonical_all_boundaries_resolved"] is True
        and point["canonical_all_boundaries_roundoff_prefilter_pass"] is True
    ]
    if not eligible_indices or eligible_indices[0] != 0:
        raise RuntimeError("the full-rank all-bond point must be resolved")
    selected_index = eligible_indices[0]
    first_failing_index = None
    for index in eligible_indices:
        if points[index]["accuracy"]["canonical_tree_odt"]["discovery"] < threshold:
            first_failing_index = index
            break
        selected_index = index
    selected_rank_tuple = list(points[selected_index]["rank_tuple"])
    full_rank_tuple = list(points[0]["rank_tuple"])
    nontrivial_compression_selected = selected_rank_tuple != full_rank_tuple
    return {
        "selection_rule": (
            "most compressed member of the resolved analytic-roundoff-prefiltered "
            "discovery prefix before the first eligible point more than one percentage "
            "point below full rank; held-out gauge trials are validation-only"
        ),
        "selection_gauge_trial_count": GAUGE_SELECTION_TRIALS,
        "held_out_validation_gauge_seed": GAUGE_VALIDATION_SEED,
        "held_out_validation_gauge_trial_count": 50,
        "held_out_validation_gauges_used_for_selection": False,
        "reference_discovery_accuracy": reference_accuracy,
        "minimum_discovery_accuracy": threshold,
        "eligible_schedule_indices": eligible_indices,
        "first_failing_eligible_schedule_index": first_failing_index,
        "selected_schedule_index": selected_index,
        "selected_rank_tuple": selected_rank_tuple,
        "nontrivial_compression_selected": nontrivial_compression_selected,
        "outcome": (
            "nontrivial_rank_tuple_selected"
            if nontrivial_compression_selected
            else "full_rank_selected_by_frozen_prefiltered_discovery_prefix_rule"
        ),
    }


def _finite_gap_record(values: torch.Tensor, rank: int) -> dict:
    record = vars(eigengap_diagnostics(values, rank))
    return {
        key: (value if not isinstance(value, float) or math.isfinite(value) else None)
        for key, value in record.items()
    }


@torch.no_grad()
def simultaneous_uniform_all_bond_curve(
    network: HomogeneousChiTN,
    global_systems,
    local_systems,
    discovery,
    confirmation,
    *,
    seed: int,
) -> dict:
    """Evaluate one fixed uniform rank at every bond in parallel.

    This is the physical all-projector contraction in Dooms et al. Algorithm 3
    for the topology-specific independent-clone coefficient tree.  Uniform
    percentage removal is a frozen plotting convention, not a proof that the
    common rank is the best allocation of a global storage budget.
    """

    dimensions = network.bond_dims
    schedule = uniform_all_bond_schedule(dimensions)
    multiplicities = tuple(2 ** (network.n_layers - bond) for bond in range(len(dimensions)))
    full_storage = dense_homogeneous_tree_storage(
        leaf_dimension=network.embedding.shape[1],
        output_dimension=network.head.shape[0],
        ranks=dimensions,
    )
    full_coefficient_norm_squared_tensor = coefficient_inner_product(network, network)
    full_coefficient_norm_squared = float(full_coefficient_norm_squared_tensor.item())
    if not math.isfinite(full_coefficient_norm_squared) or full_coefficient_norm_squared <= 0.0:
        raise RuntimeError("the full coefficient tree has nonpositive squared norm")

    trace_totals = [float(values.clamp_min(0).sum().item()) for values, _ in global_systems]
    trace_relative_disagreement = max(
        abs(value - full_coefficient_norm_squared) / full_coefficient_norm_squared
        for value in trace_totals
    )

    global_gap_records = {
        rank: [_finite_gap_record(values, rank) for values, _ in global_systems]
        for rank in ALL_BOND_EXPECTED_UNIFORM_RANKS
    }
    local_gap_records = {
        rank: [_finite_gap_record(values, rank) for values, _ in local_systems]
        for rank in ALL_BOND_EXPECTED_UNIFORM_RANKS
    }

    haar_bases = []
    for draw in range(ALL_BOND_HAAR_DRAWS):
        draw_bases = []
        for bond, dimension in enumerate(dimensions):
            generator = torch.Generator().manual_seed(
                ALL_BOND_HAAR_SEED + seed * 10_000 + draw * 100 + bond
            )
            matrix = torch.randn(
                dimension, dimension, generator=generator, dtype=torch.float64
            ).to(network.embedding.device)
            basis, _ = torch.linalg.qr(matrix)
            draw_bases.append(basis)
        haar_bases.append(tuple(draw_bases))

    points = []
    for schedule_index, scheduled in enumerate(schedule):
        ranks = scheduled["rank_tuple"]
        rank = ranks[0]
        global_boundaries = global_gap_records[rank]
        local_boundaries = local_gap_records[rank]
        global_resolved = all(item["resolved"] is True for item in global_boundaries)
        local_resolved = all(item["resolved"] is True for item in local_boundaries)
        global_roundoff_prefilters = [
            roundoff_only_projector_prefilter(values, bond_rank)
            for (values, _), bond_rank in zip(global_systems, ranks)
        ]
        global_roundoff_prefilter_pass = all(
            item["passes"] is True for item in global_roundoff_prefilters
        )

        is_full_rank = ranks == dimensions
        canonical_bases = top_bases(global_systems, ranks)
        local_bases = top_bases(local_systems, ranks)
        # Even the full-rank point physically absorbs every Algorithm-3 basis.
        # This keeps its coefficient certificate nonvacuous.
        canonical_compressed = compress_homogeneous(network, canonical_bases)
        local_compressed = (
            network if is_full_rank else compress_homogeneous(network, local_bases)
        )
        canonical_discovery = network_accuracy(canonical_compressed, *discovery)
        canonical_confirmation = network_accuracy(canonical_compressed, *confirmation)
        local_discovery = network_accuracy(local_compressed, *discovery)
        local_confirmation = network_accuracy(local_compressed, *confirmation)

        haar_discovery = []
        haar_confirmation = []
        for bases in haar_bases:
            compressed = (
                network
                if is_full_rank
                else compress_homogeneous(
                    network,
                    tuple(
                        basis[:, :bond_rank]
                        for basis, bond_rank in zip(bases, ranks)
                    ),
                )
            )
            haar_discovery.append(network_accuracy(compressed, *discovery))
            haar_confirmation.append(network_accuracy(compressed, *confirmation))

        tail_terms = []
        for bond, (((values, _), bond_rank), multiplicity) in enumerate(
            zip(zip(global_systems, ranks), multiplicities)
        ):
            tail = float(discarded_trace(values, bond_rank).item())
            tail_terms.append(
                {
                    "bond": bond,
                    "rank": bond_rank,
                    "multiplicity": multiplicity,
                    "discarded_trace": tail,
                    "weighted_discarded_trace": multiplicity * tail,
                }
            )
        bound_squared = float(
            hierarchical_tail_bound_squared(global_systems, ranks).item()
        )
        actual_error_squared = float(
            coefficient_error_squared(network, canonical_compressed).item()
        )
        bound_slack = (
            ALL_BOND_BOUND_RELATIVE_SLACK * bound_squared
            + ALL_BOND_BOUND_ABSOLUTE_SLACK_FRACTION
            * full_coefficient_norm_squared
        )
        retained_storage = dense_homogeneous_tree_storage(
            leaf_dimension=network.embedding.shape[1],
            output_dimension=network.head.shape[0],
            ranks=ranks,
        )
        unique_full = sum(dimensions)
        occurrence_full = sum(
            multiplicity * dimension
            for multiplicity, dimension in zip(multiplicities, dimensions)
        )
        point = {
            "schedule_index": schedule_index,
            "target_removal_basis_points": scheduled["target_removal_basis_points"],
            "uniform_rank": rank,
            "rank_tuple": list(ranks),
            "canonical_boundaries": global_boundaries,
            "canonical_all_boundaries_resolved": global_resolved,
            "canonical_roundoff_only_prefilters": global_roundoff_prefilters,
            "canonical_all_boundaries_roundoff_prefilter_pass": (
                global_roundoff_prefilter_pass
            ),
            "local_boundaries": local_boundaries,
            "local_all_boundaries_resolved": local_resolved,
            "compression": {
                "unique_bond_coordinate_removal_fraction": 1.0
                - sum(ranks) / unique_full,
                "occurrence_weighted_bond_coordinate_removal_fraction": 1.0
                - sum(
                    multiplicity * bond_rank
                    for multiplicity, bond_rank in zip(multiplicities, ranks)
                )
                / occurrence_full,
                "dense_homogeneous_tn_full_storage_scalars": full_storage,
                "dense_homogeneous_tn_retained_storage_scalars": retained_storage,
                "dense_homogeneous_tn_storage_removal_fraction": 1.0
                - retained_storage / full_storage,
            },
            "coefficient_certificate": {
                "object": "topology_specific_independent_clone_tree",
                "error_evaluation": "canonicalized_block_sparse_difference_tree",
                "full_coefficient_norm_squared": full_coefficient_norm_squared,
                "tail_terms": tail_terms,
                "hsvd_upper_bound_squared": bound_squared,
                "hsvd_relative_error_upper_bound": math.sqrt(
                    bound_squared / full_coefficient_norm_squared
                ),
                "actual_coefficient_error_squared": actual_error_squared,
                "actual_relative_coefficient_error": math.sqrt(
                    actual_error_squared / full_coefficient_norm_squared
                ),
                "bound_numerical_slack": bound_slack,
                "actual_error_within_bound_plus_declared_numerical_slack": actual_error_squared
                <= bound_squared + bound_slack,
            },
            "accuracy": {
                "canonical_tree_odt": {
                    "discovery": canonical_discovery,
                    "confirmation": canonical_confirmation,
                },
                "canonical_adjacent_local": {
                    "discovery": local_discovery,
                    "confirmation": local_confirmation,
                },
                "independent_nested_haar": {
                    "draw_count": ALL_BOND_HAAR_DRAWS,
                    "discovery": haar_discovery,
                    "confirmation": haar_confirmation,
                    "discovery_mean": float(np.mean(haar_discovery)),
                    "discovery_sample_standard_deviation": float(
                        np.std(haar_discovery, ddof=1)
                    ),
                    "confirmation_mean": float(np.mean(haar_confirmation)),
                    "confirmation_sample_standard_deviation": float(
                        np.std(haar_confirmation, ddof=1)
                    ),
                },
            },
        }
        points.append(point)

    reference_discovery = points[0]["accuracy"]["canonical_tree_odt"]["discovery"]
    selection = select_resolved_accuracy_frontier(points, reference_discovery)
    selected = points[selection["selected_schedule_index"]]
    selected_canonical_confirmation = selected["accuracy"]["canonical_tree_odt"][
        "confirmation"
    ]
    selected_haar_confirmation = selected["accuracy"]["independent_nested_haar"][
        "confirmation"
    ]
    haar_at_least_canonical = sum(
        value >= selected_canonical_confirmation for value in selected_haar_confirmation
    )
    selection.update(
        {
            "selected_discovery_accuracy": selected["accuracy"]["canonical_tree_odt"][
                "discovery"
            ],
            "selected_confirmation_accuracy": selected_canonical_confirmation,
            "selected_confirmation_drop_from_full": points[0]["accuracy"][
                "canonical_tree_odt"
            ]["confirmation"]
            - selected_canonical_confirmation,
            "selected_local_all_boundaries_resolved": selected[
                "local_all_boundaries_resolved"
            ],
            "selected_confirmation_odt_minus_local": selected_canonical_confirmation
            - selected["accuracy"]["canonical_adjacent_local"]["confirmation"],
            "selected_confirmation_haar_at_least_canonical_count": haar_at_least_canonical,
            "selected_confirmation_haar_randomization_p": (
                1 + haar_at_least_canonical
            )
            / (ALL_BOND_HAAR_DRAWS + 1),
        }
    )

    return {
        "claim_boundary": (
            "Simultaneous Algorithm-3 truncation of all four bonds in the canonical "
            "topology-specific independent-clone coefficient tree. Uniform percentage "
            "removal is a frozen evaluation convention, not a globally optimal rank allocation, "
            "a symmetric-polynomial quotient, an attention decomposition, or a VLA decomposition."
        ),
        "target_removal_basis_points": list(ALL_BOND_TARGET_REMOVAL_BASIS_POINTS),
        "expected_uniform_ranks": list(ALL_BOND_EXPECTED_UNIFORM_RANKS),
        "bond_dimensions": list(dimensions),
        "bond_occurrence_multiplicities": list(multiplicities),
        "haar_draws": ALL_BOND_HAAR_DRAWS,
        "haar_construction": (
            "one independently seeded square Gaussian QR basis per seed, draw, and bond; "
            "leading columns are nested across the complete frozen schedule"
        ),
        "trace_totals": trace_totals,
        "local_spectra_by_bond": [
            [float(value) for value in values.cpu()] for values, _ in local_systems
        ],
        "maximum_trace_total_relative_disagreement": trace_relative_disagreement,
        "points": points,
        "discovery_selection": selection,
    }


def make_gauge(dimension: int, condition: float, generator: torch.Generator, device):
    """Draw a nonsymmetric gauge with a prescribed target singular spectrum.

    The log singular values are centered at zero.  Consequently the target
    determinant magnitude is one, the largest and smallest singular values are
    reciprocal, and neither side of a transported bond is systematically given
    a larger numerical scale.
    """

    if dimension < 1:
        raise ValueError("gauge dimension must be positive")
    if not math.isfinite(condition) or condition < 1.0:
        raise ValueError("gauge condition must be finite and at least one")
    left, _ = torch.linalg.qr(
        torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    )
    right, _ = torch.linalg.qr(
        torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    )
    half_log_condition = 0.5 * math.log10(condition)
    exponents = torch.linspace(
        -half_log_condition,
        half_log_condition,
        dimension,
        dtype=torch.float64,
    )
    gauge = left @ torch.diag(torch.pow(10.0, exponents)) @ right.T
    return gauge.to(device)


@torch.no_grad()
def learned_gauge_audit(
    raw_network: HomogeneousChiTN,
    canonical,
    sample_images: torch.Tensor,
    *,
    trials: int,
    required_ranks_by_bond: tuple[tuple[int, ...], ...],
):
    """Audit functional gauge invariance with conditioned eigenspace certificates.

    Raw RQ factors can be ill-conditioned even when the canonical prefix
    functions agree.  They are retained only as a diagnostic.  Every
    load-bearing transport is instead the polar factor of the recursively
    contracted overlap between the two canonical prefix bases.
    """

    base_network = canonical.network
    if len(required_ranks_by_bond) != len(base_network.bond_dims):
        raise ValueError("one required-rank tuple is needed for every bond")
    for bond, (required, dimension) in enumerate(
        zip(required_ranks_by_bond, base_network.bond_dims)
    ):
        if any(type(rank) is not int or not 1 <= rank <= dimension for rank in required):
            raise ValueError(f"invalid required projector rank at bond {bond}")
    base_grams = canonical_environments(base_network)
    base_coefficient_norm_squared = float(
        coefficient_inner_product(base_network, base_network).item()
    )
    if not math.isfinite(base_coefficient_norm_squared) or base_coefficient_norm_squared <= 0:
        raise RuntimeError("base coefficient tree has nonpositive squared norm")
    images = sample_images.to(
        device=raw_network.embedding.device, dtype=raw_network.embedding.dtype
    )
    raw_base_logits = homogeneous_forward(raw_network, images)
    canonical_base_logits = homogeneous_forward(base_network, images)
    records = []
    matrix_artifacts = {
        f"base_gram_bond_{bond}": gram.detach().cpu().numpy()
        for bond, gram in enumerate(base_grams)
    }
    matrix_artifacts.update(
        {
            "replay_sample_flat": images.flatten(1).detach().cpu().numpy(),
            "raw_network_embedding": raw_network.embedding.detach().cpu().numpy(),
            "raw_network_head": raw_network.head.detach().cpu().numpy(),
            "base_canonical_embedding": base_network.embedding.detach().cpu().numpy(),
            "base_canonical_head": base_network.head.detach().cpu().numpy(),
        }
    )
    for layer, core in enumerate(raw_network.cores):
        matrix_artifacts[f"raw_network_core_{layer}"] = core.detach().cpu().numpy()
    for layer, core in enumerate(base_network.cores):
        matrix_artifacts[f"base_canonical_core_{layer}"] = core.detach().cpu().numpy()

    for trial in range(trials):
        condition = 10.0 ** (3.0 * trial / max(1, trials - 1))
        generator = torch.Generator().manual_seed(GAUGE_SEED + trial)
        gauged = raw_network
        gauges = []
        realized_minimum_singular_values = []
        realized_maximum_singular_values = []
        realized_conditions = []
        realized_nonsymmetry = []
        bond_realizations = []
        for bond, dimension in enumerate(raw_network.bond_dims):
            gauge = make_gauge(dimension, condition, generator, raw_network.embedding.device)
            gauges.append(gauge)
            # Preserve the actual sampled coordinate change, rather than only
            # scalar reductions of it.  The independent aggregator can then
            # recompute the singular extrema, condition, and nonsymmetry gates
            # from a non-pickle matrix artifact.
            matrix_artifacts[f"trial_{trial}_bond_{bond}_raw_gauge"] = (
                gauge.detach().cpu().numpy()
            )
            single_bond_gauged = apply_bond_gauge(raw_network, bond, gauge)
            single_bond_logits = homogeneous_forward(single_bond_gauged, images)
            sampled_single_bond_logit_replay_relative = float(
                (
                    torch.linalg.vector_norm(single_bond_logits - raw_base_logits)
                    / torch.linalg.vector_norm(raw_base_logits).clamp_min(1e-30)
                ).item()
            )
            singular_values = torch.linalg.svdvals(gauge)
            realized_maximum = float(singular_values[0].item())
            realized_minimum = float(singular_values[-1].item())
            realized_condition = float(
                (singular_values[0] / singular_values[-1]).item()
            )
            nonsymmetry = float(
                (
                    torch.linalg.matrix_norm(gauge - gauge.T)
                    / torch.linalg.matrix_norm(gauge).clamp_min(1e-30)
                ).item()
            )
            realized_maximum_singular_values.append(realized_maximum)
            realized_minimum_singular_values.append(realized_minimum)
            realized_conditions.append(realized_condition)
            realized_nonsymmetry.append(nonsymmetry)
            bond_realizations.append(
                {
                    "bond": bond,
                    "minimum_singular_value": realized_minimum,
                    "maximum_singular_value": realized_maximum,
                    "condition": realized_condition,
                    "nonsymmetry_relative": nonsymmetry,
                    "sampled_single_bond_logit_replay_relative": (
                        sampled_single_bond_logit_replay_relative
                    ),
                }
            )
            gauged = apply_bond_gauge(gauged, bond, gauge)

        raw_gauged_logits = homogeneous_forward(gauged, images)
        raw_coordinate_replay_relative = float(
            (
                torch.linalg.vector_norm(raw_gauged_logits - raw_base_logits)
                / torch.linalg.vector_norm(raw_base_logits).clamp_min(1e-30)
            ).item()
        )
        gauged_canonical = canonicalize_homogeneous(gauged)
        gauged_canonical_logits = homogeneous_forward(gauged_canonical.network, images)
        canonicalized_function_relative = float(
            (
                torch.linalg.vector_norm(
                    gauged_canonical_logits - canonical_base_logits
                )
                / torch.linalg.vector_norm(canonical_base_logits).clamp_min(1e-30)
            ).item()
        )
        gauged_grams = canonical_environments(gauged_canonical.network)
        overlaps = canonical_prefix_overlaps(gauged_canonical.network, base_network)
        coefficient_error = float(
            coefficient_error_squared(base_network, gauged_canonical.network).item()
        )
        coefficient_replay_relative = math.sqrt(
            max(0.0, coefficient_error) / base_coefficient_norm_squared
        )

        orthogonality = []
        prefix_span_defects = []
        gram_covariance = []
        spectrum_relative = []
        projector_pair_count = 0
        expected_projector_pair_count = 0
        raw_factor_transport_residual = []
        for bond, (base_gram, gauged_gram, overlap) in enumerate(
            zip(base_grams, gauged_grams, overlaps)
        ):
            transport = polar_orthogonal_transport(overlap)
            identity = torch.eye(
                transport.shape[0], dtype=transport.dtype, device=transport.device
            )
            overlap_left_defect = torch.linalg.matrix_norm(
                overlap @ overlap.T - identity, ord=2
            )
            overlap_right_defect = torch.linalg.matrix_norm(
                overlap.T @ overlap - identity, ord=2
            )
            prefix_span_defect = float(
                torch.maximum(overlap_left_defect, overlap_right_defect).item()
            )
            prefix_span_defects.append(prefix_span_defect)
            overlap_singular_values = torch.linalg.svdvals(overlap)

            # This raw-factor solve is intentionally non-load-bearing.  It
            # documents why coordinate-factor Procrustes is numerically worse
            # than contracting the canonical prefix functions themselves.
            base_factor = canonical.raw_from_canonical[bond]
            gauged_factor = gauged_canonical.raw_from_canonical[bond]
            target_factor = gauges[bond] @ base_factor
            raw_left, _, raw_right = torch.linalg.svd(
                gauged_factor.T @ target_factor, full_matrices=True
            )
            raw_factor_transport = raw_left @ raw_right
            factor_error = float(
                (
                    torch.linalg.matrix_norm(
                        gauged_factor @ raw_factor_transport - target_factor
                    )
                    / torch.linalg.matrix_norm(target_factor).clamp_min(1e-30)
                ).item()
            )
            raw_factor_transport_residual.append(factor_error)
            orthogonality_error = float(
                torch.linalg.matrix_norm(
                    transport @ transport.T - identity, ord=2
                ).item()
            )
            orthogonality.append(orthogonality_error)
            predicted = transport @ base_gram @ transport.T
            gram_error = float(
                (
                    torch.linalg.matrix_norm(gauged_gram - predicted, ord=2)
                    / torch.linalg.matrix_norm(gauged_gram, ord=2).clamp_min(1e-30)
                ).item()
            )
            gram_covariance.append(gram_error)
            base_values, base_vectors = sorted_eigensystem(base_gram)
            gauged_values, gauged_vectors = sorted_eigensystem(gauged_gram)
            spectrum_error = float(
                (
                    torch.linalg.vector_norm(base_values - gauged_values)
                    / torch.linalg.vector_norm(base_values).clamp_min(1e-30)
                ).item()
            )
            spectrum_relative.append(spectrum_error)
            resolved_ranks = [
                rank
                for rank in range(1, base_values.numel())
                if eigengap_diagnostics(base_values, rank).resolved
            ]
            expected_projector_pair_count += len(resolved_ranks)
            projector_certificates = []
            for audit_rank in resolved_ranks:
                certificate = projector_perturbation_certificate(
                    base_gram,
                    gauged_gram,
                    transport,
                    rank=audit_rank,
                )
                certificate_record = vars(certificate)
                base_projector = (
                    base_vectors[:, :audit_rank] @ base_vectors[:, :audit_rank].T
                )
                candidate_projector = (
                    gauged_vectors[:, :audit_rank]
                    @ gauged_vectors[:, :audit_rank].T
                )
                base_supported_defect = torch.linalg.matrix_norm(
                    base_projector
                    @ (identity - overlap.T @ overlap)
                    @ base_projector,
                    ord=2,
                )
                candidate_supported_defect = torch.linalg.matrix_norm(
                    candidate_projector
                    @ (identity - overlap @ overlap.T)
                    @ candidate_projector,
                    ord=2,
                )
                polar_replacement_defect = torch.linalg.matrix_norm(
                    (transport - overlap) @ base_projector,
                    ord=2,
                )
                certificate_record.update(
                    {
                        "base_supported_overlap_defect": float(
                            base_supported_defect.item()
                        ),
                        "candidate_supported_overlap_defect": float(
                            candidate_supported_defect.item()
                        ),
                        "polar_replacement_supported_defect": float(
                            polar_replacement_defect.item()
                        ),
                        "maximum_supported_overlap_defect": float(
                            torch.stack(
                                (
                                    base_supported_defect,
                                    candidate_supported_defect,
                                    polar_replacement_defect,
                                )
                            ).max().item()
                        ),
                    }
                )
                certificate_record["required_for_certificate"] = (
                    audit_rank in required_ranks_by_bond[bond]
                )
                projector_certificates.append(certificate_record)
            projector_pair_count += len(projector_certificates)

            matrix_prefix = f"trial_{trial}_bond_{bond}"
            matrix_artifacts[f"{matrix_prefix}_overlap"] = (
                overlap.detach().cpu().numpy()
            )
            matrix_artifacts[f"{matrix_prefix}_transport"] = (
                transport.detach().cpu().numpy()
            )
            matrix_artifacts[f"{matrix_prefix}_candidate_gram"] = (
                gauged_gram.detach().cpu().numpy()
            )
            bond_realizations[bond].update(
                {
                    "prefix_overlap_minimum_singular_value": float(
                        overlap_singular_values[-1].item()
                    ),
                    "prefix_overlap_maximum_singular_value": float(
                        overlap_singular_values[0].item()
                    ),
                    "prefix_overlap_span_defect": prefix_span_defect,
                    "transport_orthogonality": orthogonality_error,
                    "raw_factor_transport_relative_diagnostic": factor_error,
                    "gram_covariance_relative": gram_error,
                    "spectrum_relative": spectrum_error,
                    "projector_certificates": projector_certificates,
                }
            )

        records.append(
            {
                "trial": trial,
                "target_condition": condition,
                "target_minimum_singular_value": condition ** -0.5,
                "target_maximum_singular_value": condition ** 0.5,
                "minimum_realized_singular_value": min(
                    realized_minimum_singular_values
                ),
                "maximum_realized_singular_value": max(
                    realized_maximum_singular_values
                ),
                "minimum_realized_condition": min(realized_conditions),
                "maximum_realized_condition": max(realized_conditions),
                "minimum_realized_nonsymmetry_relative": min(realized_nonsymmetry),
                "bond_realizations": bond_realizations,
                "raw_coordinate_replay_relative": raw_coordinate_replay_relative,
                "canonicalized_function_relative": canonicalized_function_relative,
                "canonicalized_coefficient_replay_relative": coefficient_replay_relative,
                "maximum_sampled_single_bond_logit_replay_relative": max(
                    realization["sampled_single_bond_logit_replay_relative"]
                    for realization in bond_realizations
                ),
                "max_prefix_overlap_span_defect": max(prefix_span_defects),
                "max_transport_orthogonality": max(orthogonality),
                "max_raw_factor_transport_relative_diagnostic": max(
                    raw_factor_transport_residual
                ),
                "max_gram_covariance_relative": max(gram_covariance),
                "max_spectrum_relative": max(spectrum_relative),
                "projector_certificate_pair_count": projector_pair_count,
                "expected_projector_certificate_pair_count": expected_projector_pair_count,
            }
        )
    return records, matrix_artifacts


def required_gauge_ranks(
    selected_all_bond_ranks: tuple[int, ...],
    *,
    selected_single_bond: int,
    selected_single_rank: int,
    bond_dims: tuple[int, ...],
    smoke_diagnostic_rank: int | None,
) -> tuple[tuple[tuple[int, ...], ...], int | None]:
    """Bind gauge certificates to reported ranks and a nonvacuous smoke rank."""

    if len(selected_all_bond_ranks) != len(bond_dims):
        raise ValueError("selected rank tuple and bond dimensions differ")
    if selected_single_bond not in range(len(bond_dims)):
        raise ValueError("selected single-bond index is out of range")
    required_rank_sets = [set([rank]) for rank in selected_all_bond_ranks]
    required_rank_sets[selected_single_bond].add(selected_single_rank)
    diagnostic_rank = smoke_diagnostic_rank
    if diagnostic_rank is not None and diagnostic_rank != 1:
        raise ValueError("the frozen smoke diagnostic rank must be one")
    if diagnostic_rank is not None:
        for required, dimension in zip(required_rank_sets, bond_dims):
            if diagnostic_rank < dimension:
                required.add(diagnostic_rank)
    required = tuple(tuple(sorted(values)) for values in required_rank_sets)
    if any(
        not values or any(rank < 1 or rank > dimension for rank in values)
        for values, dimension in zip(required, bond_dims)
    ):
        raise ValueError("required gauge-projector rank is invalid")
    return required, diagnostic_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint-output", type=Path)
    parser.add_argument("--gauge-matrix-output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--rank-grid", default="1,2,4,6,8,12,16,24,32")
    parser.add_argument("--haar-draws", type=int, default=16)
    parser.add_argument("--gauge-trials", type=int, default=50)
    parser.add_argument("--minimum-accuracy", type=float, default=0.70)
    parser.add_argument("--smoke-nontrivial-diagnostic-rank", type=int)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    project_root = require_physical_root(Path(__file__).resolve().parents[1])
    if args.data_root.is_symlink():
        raise RuntimeError("data root must not be a symlink")
    lexical_data_root = Path(os.path.abspath(args.data_root))
    args.data_root = args.data_root.resolve(strict=True)
    if lexical_data_root != args.data_root or not args.data_root.is_dir():
        raise RuntimeError("data root must be an existing physical directory")
    manifest_path = project_root / "source_manifest.sha256"
    if not manifest_path.is_file():
        raise RuntimeError("frozen source manifest is required")
    frozen_entries = verify_source_freeze(project_root, manifest_path)
    manifest_sha256_at_start = digest(manifest_path)

    stage = "prefetch" if args.download_only else (
        "smoke" if args.seed == 9173 else "train_array"
    )
    if (
        (stage == "smoke" and args.smoke_nontrivial_diagnostic_rank != 1)
        or (stage != "smoke" and args.smoke_nontrivial_diagnostic_rank is not None)
    ):
        raise RuntimeError(
            "rank-one smoke diagnostic must be explicit and is forbidden outside smoke"
        )
    launch = validate_athena_stage_binding(
        project_root,
        args.data_root,
        manifest_sha256_at_start,
        stage=stage,
        seed=None if args.download_only else args.seed,
    )
    validate_unit_certificate(
        project_root, frozen_entries, manifest_sha256_at_start, launch
    )
    if stage == "smoke":
        prefetch_path = require_confined_path(
            project_root,
            project_root / "results" / "prefetch.json",
            label="prefetch receipt",
            kind="file",
        )
        prefetch = json.loads(prefetch_path.read_text())
        if prefetch != {
            "schema": PREFETCH_SCHEMA,
            "status": "complete",
            "source_manifest_sha256": manifest_sha256_at_start,
            "dataset_root": str(args.data_root),
            "dataset_files": SVHN_FILES,
            "slurm_job_id": launch["dag"]["nodes"]["prefetch"]["job_id"],
        }:
            raise RuntimeError("smoke predecessor prefetch receipt differs")
    elif stage == "train_array":
        smoke_path = require_confined_path(
            project_root,
            project_root / "results" / "smoke.json",
            label="smoke predecessor result",
            kind="file",
        )
        smoke = json.loads(smoke_path.read_text())
        smoke_runtime = smoke.get("provenance", {}).get("runtime", {})
        if (
            smoke.get("schema") != SCHEMA
            or smoke.get("status") != "complete"
            or smoke.get("seed") != 9173
            or smoke.get("provenance", {}).get("frozen_source_manifest", {}).get("sha256")
            != manifest_sha256_at_start
            or smoke.get("provenance", {}).get("dataset_root") != str(args.data_root)
            or smoke_runtime.get("slurm_job_id")
            != launch["dag"]["nodes"]["smoke"]["job_id"]
            or smoke_runtime.get("slurm_array_job_id") is not None
            or smoke_runtime.get("slurm_array_task_id") is not None
        ):
            raise RuntimeError("train predecessor smoke result differs")
        smoke_checkpoint = require_confined_path(
            project_root,
            project_root / "results" / "smoke.pt",
            label="smoke predecessor checkpoint",
            kind="file",
        )
        smoke_checkpoint_manifest = smoke.get("provenance", {}).get("checkpoint", {})
        smoke_gauge = require_confined_path(
            project_root,
            project_root / "results" / "smoke_gauge_matrices.npz",
            label="smoke predecessor gauge matrix artifact",
            kind="file",
        )
        smoke_gauge_manifest = smoke.get("gauge_audit", {}).get("matrix_artifact", {})
        if (
            smoke_checkpoint_manifest.get("path") != str(smoke_checkpoint)
            or smoke_checkpoint_manifest.get("bytes") != smoke_checkpoint.stat().st_size
            or smoke_checkpoint_manifest.get("sha256") != digest(smoke_checkpoint)
            or smoke_gauge_manifest.get("path") != str(smoke_gauge)
            or smoke_gauge_manifest.get("bytes") != smoke_gauge.stat().st_size
            or smoke_gauge_manifest.get("sha256") != digest(smoke_gauge)
            or smoke_gauge_manifest.get("allow_pickle") is not False
        ):
            raise RuntimeError("train predecessor smoke artifact manifest differs")
        validation_path = require_confined_path(
            project_root,
            project_root / "results" / "smoke_validation.json",
            label="fresh-process smoke-validation receipt",
            kind="file",
        )
        validation = json.loads(validation_path.read_text())
        certificate_path = require_confined_path(
            project_root,
            project_root / "results" / "canonical_odt_unit_certificate.json",
            label="unit-test certificate",
            kind="file",
        )
        certificate = json.loads(certificate_path.read_text())
        test_log = require_confined_path(
            project_root,
            Path(certificate.get("test_log_path", "")),
            label="unit-test log",
            kind="file",
        )
        prefetch_path = require_confined_path(
            project_root,
            project_root / "results" / "prefetch.json",
            label="prefetch receipt",
            kind="file",
        )
        expected_validation_keys = {
            "schema", "status", "source_manifest_sha256", "artifact_sha256",
            "input_inventory", "dataset_files", "producer_slurm_job_id",
            "validator_slurm_job_id", "required_projector_three_way_authentication",
        }
        expected_validation_hashes = {
            "launch_json": digest(project_root / "results" / "launch.json"),
            "unit_certificate_json": digest(certificate_path),
            "unit_test_log": digest(test_log),
            "prefetch_json": digest(prefetch_path),
            "smoke_json": digest(smoke_path),
            "smoke_checkpoint_pt": digest(smoke_checkpoint),
            "smoke_gauge_matrices_npz": digest(smoke_gauge),
        }
        expected_validation_inventory = sorted({
            "launch.json",
            "canonical_odt_unit_certificate.json",
            "prefetch.json",
            "smoke.json",
            "smoke.pt",
            "smoke_gauge_matrices.npz",
            test_log.name,
        })
        if (
            not isinstance(validation, dict)
            or set(validation) != expected_validation_keys
            or validation.get("schema") != SMOKE_VALIDATION_SCHEMA
            or validation.get("status") != "complete"
            or validation.get("source_manifest_sha256") != manifest_sha256_at_start
            or validation.get("artifact_sha256") != expected_validation_hashes
            or validation.get("input_inventory") != expected_validation_inventory
            or validation.get("dataset_files") != SVHN_FILES
            or validation.get("producer_slurm_job_id")
            != launch["dag"]["nodes"]["smoke"]["job_id"]
            or validation.get("validator_slurm_job_id")
            != launch["dag"]["nodes"]["smoke_validate"]["job_id"]
            or not isinstance(
                validation.get("required_projector_three_way_authentication"), dict
            )
        ):
            raise RuntimeError("train predecessor smoke-validation receipt differs")
        validate_required_authentication_decisions(
            validation["required_projector_three_way_authentication"],
            expected_identities=SMOKE_REQUIRED_AUTHENTICATION_IDENTITIES,
        )

    if not 0.0 <= args.minimum_accuracy <= 1.0:
        parser.error("--minimum-accuracy must lie in [0, 1]")
    train_dataset, test_dataset = make_datasets(args.data_root, download=args.download_only)
    dataset_files = validated_svhn_manifest(args.data_root)
    if args.download_only:
        verify_source_freeze(project_root, manifest_path)
        receipt_path = require_confined_path(
            project_root,
            project_root / "results" / "prefetch.json",
            label="prefetch receipt",
            allow_missing_leaf=True,
        )
        if receipt_path.exists() or receipt_path.is_symlink():
            raise RuntimeError("prefetch receipt already exists")
        receipt = {
            "schema": PREFETCH_SCHEMA,
            "status": "complete",
            "source_manifest_sha256": manifest_sha256_at_start,
            "dataset_root": str(args.data_root),
            "dataset_files": dataset_files,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        receipt_path.write_text(json.dumps(receipt, indent=2))
        print(
            json.dumps(
                {
                    "downloaded": True,
                    "train": len(train_dataset),
                    "test": len(test_dataset),
                    "dataset_files": dataset_files,
                }
            )
        )
        return
    if (
        args.output is None
        or args.checkpoint_output is None
        or args.gauge_matrix_output is None
    ):
        parser.error(
            "--output, --checkpoint-output, and --gauge-matrix-output are required "
            "unless --download-only is used"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("the training and evaluation run requires a CUDA node")
    expected_prefix = "smoke" if args.seed == 9173 else f"seed_{args.seed}"
    args.output = require_output_below_results(
        args.output, project_root, f"{expected_prefix}.json"
    )
    args.checkpoint_output = require_output_below_results(
        args.checkpoint_output, project_root, f"{expected_prefix}.pt"
    )
    args.gauge_matrix_output = require_output_below_results(
        args.gauge_matrix_output,
        project_root,
        f"{expected_prefix}_gauge_matrices.npz",
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        generator=train_generator,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )
    discovery_dataset, confirmation_dataset, discovery_indices, confirmation_indices = (
        fixed_test_panels(test_dataset)
    )
    discovery = load_panel(discovery_dataset)
    confirmation = load_panel(confirmation_dataset)

    config = ChiMLPConfig(
        in_dim=3072, dim=32, n_layers=3, num_classes=10, norm="scalar_rbn"
    )
    model = ChiMLP(config).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, weight_decay=0.05, betas=(0.9, 0.95)
    )

    epoch_accuracy = []
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(images, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        accuracy = module_accuracy(model, *discovery)
        epoch_accuracy.append(accuracy)
        print(f"seed={args.seed} epoch={epoch} discovery_accuracy={accuracy:.6f}", flush=True)

    calibration_images, _ = load_panel(
        torch.utils.data.Subset(train_dataset, range(50 * 256)), batch_size=256
    )
    calibration_report = calibrate_rbn_sequential(
        model,
        lambda index: calibration_images[index * 256 : (index + 1) * 256].cuda(
            non_blocking=True
        ),
        lambda module, batch: module(batch)[0],
        iters=50,
    )
    if len(calibration_report.sites) != config.n_layers:
        raise RuntimeError("not every scalar RBN was calibrated")
    model.eval()
    args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint_output)
    before_reload = model.eval()
    reload_sample = discovery[0][:256].cuda(non_blocking=True)
    with torch.no_grad():
        before_reload_logits = before_reload(reload_sample)[0]
    reloaded = ChiMLP(config).cuda().eval()
    reloaded.load_state_dict(
        torch.load(args.checkpoint_output, map_location="cuda", weights_only=True), strict=True
    )
    for norm in reloaded.norms:
        if not isinstance(norm, RmsBatchNorm) or not bool(norm.initialized):
            raise RuntimeError("reloaded scalar RBN is not initialized")
        norm.freeze()
    with torch.no_grad():
        reloaded_logits = reloaded(reload_sample)[0]
    checkpoint_reload_max_absolute = float(
        (reloaded_logits - before_reload_logits).abs().max().item()
    )
    if checkpoint_reload_max_absolute != 0.0:
        raise RuntimeError(
            f"serialized checkpoint reload changed logits by {checkpoint_reload_max_absolute:.3e}"
        )
    model = copy.deepcopy(reloaded).double().cuda().eval()

    raw_homogeneous = export_homogeneous_network(
        model, require_frozen_rbn=True
    )
    raw_homogeneous = HomogeneousChiTN(
        raw_homogeneous.embedding.cuda(),
        tuple(core.cuda() for core in raw_homogeneous.cores),
        raw_homogeneous.head.cuda(),
    )
    canonical = canonicalize_homogeneous(raw_homogeneous)
    global_grams = canonical_environments(canonical.network)
    local_grams = local_environments(canonical.network)
    bond = 1
    global_systems = tuple(sorted_eigensystem(gram) for gram in global_grams)
    global_values, global_vectors = global_systems[bond]
    _, local_vectors = sorted_eigensystem(local_grams[bond])
    ranks = rank_grid(args.rank_grid, canonical.network.bond_dims[bond])
    global_gap_objects = {
        rank: eigengap_diagnostics(global_values, rank) for rank in ranks
    }
    resolved_global_ranks = {
        rank for rank, diagnostics in global_gap_objects.items() if diagnostics.resolved
    }

    legacy_export = export_cores(model)
    legacy_export = (
        tuple(value.cuda() for value in legacy_export[0]),
        [value.cuda() for value in legacy_export[1]],
        tuple(value.cuda() for value in legacy_export[2]),
    )
    legacy_gram = downstream_gram(legacy_export[1], legacy_export[2], bond=1)
    _, legacy_vectors = sorted_eigensystem(legacy_gram)
    # A learned rank-zero affine baseline keeps only its fixed homogeneous
    # coordinate. This is the exact effective-affine-rank-one match for a
    # canonical rank-one projection and avoids silently dropping that endpoint.
    legacy_ranks = affine_baseline_rank_grid(ranks, legacy_vectors.shape[1])

    activations = collect_raw_bond_activations(legacy_export, discovery[0], bond=1)
    activation_center = activations.mean(0).cuda()
    centered = activations - activations.mean(0)
    activation_gram = centered.T @ centered / max(1, len(centered) - 1)
    _, pca_vectors = sorted_eigensystem(activation_gram.cuda())

    module_discovery = module_accuracy(model, *discovery)
    module_confirmation = module_accuracy(model, *confirmation)
    homogeneous_discovery = network_accuracy(raw_homogeneous, *discovery)
    canonical_discovery = network_accuracy(canonical.network, *discovery)
    homogeneous_confirmation = network_accuracy(raw_homogeneous, *confirmation)
    canonical_confirmation = network_accuracy(canonical.network, *confirmation)
    full_rotated = compress_homogeneous(
        canonical.network,
        top_bases(global_systems, canonical.network.bond_dims),
    )
    reconstruction = sampled_reconstruction_diagnostics(
        model,
        {
            "homogeneous_export": raw_homogeneous,
            "canonical_full": canonical.network,
            "full_rank_diagonalized": full_rotated,
        },
        discovery[0],
    )

    curves = {
        "canonical_tree_odt": canonical_curve(
            canonical.network,
            global_vectors,
            ranks,
            discovery,
            confirmation,
            bond=bond,
        ),
        "canonical_adjacent_local": canonical_curve(
            canonical.network,
            local_vectors,
            ranks,
            discovery,
            confirmation,
            bond=bond,
        ),
        "legacy_raw_downstream_zero_anchor": legacy_curve(
            legacy_export,
            legacy_vectors,
            legacy_ranks,
            discovery,
            confirmation,
            bond=1,
        ),
        "activation_pca_centered": legacy_curve(
            legacy_export,
            pca_vectors,
            legacy_ranks,
            discovery,
            confirmation,
            bond=1,
            center=activation_center,
        ),
    }

    haar = {}
    haar_selected = {}
    for draw in range(args.haar_draws):
        generator = torch.Generator().manual_seed(500000 + args.seed * 1000 + draw)
        matrix = torch.randn(
            canonical.network.bond_dims[bond],
            canonical.network.bond_dims[bond],
            generator=generator,
            dtype=torch.float64,
        ).cuda()
        basis, _ = torch.linalg.qr(matrix)
        haar[str(draw)] = canonical_curve(
            canonical.network, basis, ranks, discovery, confirmation, bond=bond
        )
        selected_rank = choose_rank(haar[str(draw)], canonical_discovery)
        haar_selected[str(draw)] = {
            "rank": selected_rank,
            "discovery_accuracy": haar[str(draw)]["discovery"][str(selected_rank)],
            "confirmation_accuracy": haar[str(draw)]["confirmation"][str(selected_rank)],
        }

    all_bond_curve = simultaneous_uniform_all_bond_curve(
        canonical.network,
        global_systems,
        tuple(sorted_eigensystem(gram) for gram in local_grams),
        discovery,
        confirmation,
        seed=args.seed,
    )

    references = {
        "module": {
            "discovery": module_discovery,
            "confirmation": module_confirmation,
        },
        "homogeneous_export": {
            "discovery": homogeneous_discovery,
            "confirmation": homogeneous_confirmation,
        },
        "canonical_full": {
            "discovery": canonical_discovery,
            "confirmation": canonical_confirmation,
        },
    }
    selected = {}
    canonical_rank_selection = None
    for name, curve in curves.items():
        reference_rank = max(int(rank) for rank in curve["discovery"])
        reference = curve["discovery"][str(reference_rank)]
        if name == "canonical_tree_odt":
            selected_rank, canonical_rank_selection = (
                select_canonical_rank_prospectively(curve, reference, global_values)
            )
        else:
            selected_rank = choose_rank(curve, reference)
        selected[name] = {
            "rank": selected_rank,
            "effective_affine_rank": (
                selected_rank + 1
                if name in {"legacy_raw_downstream_zero_anchor", "activation_pca_centered"}
                else selected_rank
            ),
            "discovery_accuracy": curve["discovery"][str(selected_rank)],
            "confirmation_accuracy": curve["confirmation"][str(selected_rank)],
            "reference_rank": reference_rank,
            "reference_discovery_accuracy": reference,
            "confirmation_drop_from_module": (
                module_confirmation - curve["confirmation"][str(selected_rank)]
            ),
            "eigengap_resolved": (
                global_gap_objects[selected_rank].resolved
                if name == "canonical_tree_odt" else None
            ),
        }
    if canonical_rank_selection is None:
        raise RuntimeError("canonical prospective rank-selection record was not produced")

    selected_all_bond_ranks = tuple(
        all_bond_curve["discovery_selection"]["selected_rank_tuple"]
    )
    required_ranks_by_bond, smoke_diagnostic_rank = required_gauge_ranks(
        selected_all_bond_ranks,
        selected_single_bond=bond,
        selected_single_rank=selected["canonical_tree_odt"]["rank"],
        bond_dims=canonical.network.bond_dims,
        smoke_diagnostic_rank=args.smoke_nontrivial_diagnostic_rank,
    )
    gauge_records, gauge_matrix_artifacts = learned_gauge_audit(
        raw_homogeneous,
        canonical,
        discovery[0][:GAUGE_REPLAY_SAMPLE_COUNT],
        trials=args.gauge_trials,
        required_ranks_by_bond=required_ranks_by_bond,
    )
    gauge_base_spectra = [
        [float(value) for value in values.cpu()] for values, _ in global_systems
    ]
    gauge_resolved_ranks_by_bond = [
        [
            rank
            for rank in range(1, values.numel())
            if eigengap_diagnostics(values, rank).resolved
        ]
        for values, _ in global_systems
    ]
    projector_certificates = [
        certificate
        for record in gauge_records
        for bond_record in record["bond_realizations"]
        for certificate in bond_record["projector_certificates"]
    ]
    required_projector_certificates = [
        certificate
        for certificate in projector_certificates
        if certificate["required_for_certificate"] is True
    ]
    expected_required_projector_certificate_count = args.gauge_trials * sum(
        sum(rank < dimension for rank in required)
        for required, dimension in zip(
            required_ranks_by_bond, canonical.network.bond_dims
        )
    )
    singular_extrema_relative_errors = [
        abs(realization[realized_key] - record[target_key]) / record[target_key]
        for record in gauge_records
        for realization in record["bond_realizations"]
        for realized_key, target_key in (
            ("minimum_singular_value", "target_minimum_singular_value"),
            ("maximum_singular_value", "target_maximum_singular_value"),
        )
    ]
    condition_relative_errors = [
        abs(realization["condition"] - record["target_condition"])
        / record["target_condition"]
        for record in gauge_records
        for realization in record["bond_realizations"]
    ]

    args.gauge_matrix_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.gauge_matrix_output, **gauge_matrix_artifacts)
    with np.load(args.gauge_matrix_output, allow_pickle=False) as matrix_bundle:
        if set(matrix_bundle.files) != set(gauge_matrix_artifacts):
            raise RuntimeError("gauge matrix bundle key inventory changed on round trip")
        for key in matrix_bundle.files:
            value = matrix_bundle[key]
            if value.dtype != np.float64 or not np.isfinite(value).all():
                raise RuntimeError(f"gauge matrix bundle entry {key} is invalid")
    gauge_matrix_manifest = {
        "path": str(args.gauge_matrix_output),
        "bytes": args.gauge_matrix_output.stat().st_size,
        "sha256": digest(args.gauge_matrix_output),
        "allow_pickle": False,
        "keys": {
            key: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
            for key, value in sorted(gauge_matrix_artifacts.items())
        },
    }

    missing_source_paths = [path for path in SOURCE_PATHS if not (project_root / path).is_file()]
    if missing_source_paths:
        raise RuntimeError(f"required source paths are missing: {missing_source_paths}")
    source_sha256 = {path: digest(project_root / path) for path in SOURCE_PATHS}
    spectrum = [float(value) for value in global_values.cpu()]
    nonnegative_spectrum = [max(0.0, value) for value in spectrum]
    trace_total = sum(nonnegative_spectrum)
    gap_diagnostics = {}
    for rank in ranks:
        diagnostic = vars(global_gap_objects[rank])
        gap_diagnostics[str(rank)] = {
            key: (value if not isinstance(value, float) or math.isfinite(value) else None)
            for key, value in diagnostic.items()
        }

    result = {
        "schema": SCHEMA,
        "status": "complete",
        "claim_boundary": (
            "Canonical weight-only ODT of the topology-specific homogeneous chi-MLP tree, "
            "including a frozen uniform simultaneous all-bond Algorithm-3 curve. "
            "This is a new positive control, not a reproduction of the paper's SVHN setup or numbers. "
            "This is not the symmetric polynomial quotient, attention ODT, a VLA decomposition, "
            "or evidence of semantic mechanisms."
        ),
        "seed": args.seed,
        "predecessor_failures": list(PREDECESSOR_FAILURES),
        "protocol": {
            "epochs": args.epochs,
            "batch_size": 256,
            "optimizer": "AdamW",
            "learning_rate": 2e-3,
            "weight_decay": 0.05,
            "betas": [0.9, 0.95],
            "gradient_clip": 1.0,
            "post_training_rbn_calibration_batches": 50,
            "post_training_rbn_calibration_batch_size": 256,
            "post_training_rbn_calibration_split": "first 12800 official training examples",
            "model_width": 32,
            "homogeneous_bond_width": 33,
            "layers": 3,
            "analysis_bond": bond,
            "analysis_bond_description": "homogeneous output of the first bilinear core",
            "rank_grid": ranks,
            "haar_draws": args.haar_draws,
            "gauge_trials": args.gauge_trials,
            "gauge_replay_sample_count": GAUGE_REPLAY_SAMPLE_COUNT,
            "gauge_replay_sample_indices_sha256": hashlib.sha256(
                np.asarray(
                    discovery_indices[:GAUGE_REPLAY_SAMPLE_COUNT], dtype=np.int64
                ).tobytes()
            ).hexdigest(),
            "minimum_full_model_accuracy": args.minimum_accuracy,
            "selection_accuracy_tolerance": SELECTION_ACCURACY_TOLERANCE,
            "all_bond_target_removal_basis_points": list(
                ALL_BOND_TARGET_REMOVAL_BASIS_POINTS
            ),
            "all_bond_expected_uniform_ranks": list(
                ALL_BOND_EXPECTED_UNIFORM_RANKS
            ),
            "all_bond_haar_draws": ALL_BOND_HAAR_DRAWS,
            "all_bond_haar_seed": ALL_BOND_HAAR_SEED,
            "all_bond_bound_relative_slack": ALL_BOND_BOUND_RELATIVE_SLACK,
            "all_bond_bound_absolute_slack_fraction": (
                ALL_BOND_BOUND_ABSOLUTE_SLACK_FRACTION
            ),
            "maximum_gauge_condition": 1000.0,
            "gauge_family": "independent-left-right-orthogonal-balanced-log-singular-v2",
            "gauge_transport": "recursive-canonical-prefix-overlap-polar-v1",
            "gauge_projector_roundoff_multiplier": 10_000.0,
            "gauge_required_projector_maximum_bound": GAUGE_CERTIFICATE_MAX_BOUND,
            "rank_eligibility_policy": "analytic-roundoff-only-prefilter-v1",
            "rank_eligibility_selection_gauge_trial_count": GAUGE_SELECTION_TRIALS,
            "rank_eligibility_validation_gauge_seed": GAUGE_VALIDATION_SEED,
            "rank_eligibility_validation_gauge_trial_count": args.gauge_trials,
            "rank_eligibility_validation_gauges_used_for_selection": False,
            "rank_eligibility_full_rank_fallback_allowed": True,
            "gauge_required_ranks_by_bond": [
                list(required) for required in required_ranks_by_bond
            ],
            "smoke_nontrivial_diagnostic_rank": smoke_diagnostic_rank,
            "split_seed": SPLIT_SEED,
            "discovery_count": len(discovery_indices),
            "confirmation_count": len(confirmation_indices),
            "training_split": "official SVHN train only",
            "deterministic_algorithms": True,
            "discovery_indices_sha256": hashlib.sha256(
                np.asarray(discovery_indices, dtype=np.int64).tobytes()
            ).hexdigest(),
            "confirmation_indices_sha256": hashlib.sha256(
                np.asarray(confirmation_indices, dtype=np.int64).tobytes()
            ).hexdigest(),
        },
        "training_discovery_accuracy_by_epoch": epoch_accuracy,
        "post_training_rbn_calibration": {
            "sites": [vars(site) for site in calibration_report.sites],
            "excluded_paths": list(calibration_report.excluded_paths),
        },
        "references": references,
        "canonicalization": {
            "isometry_errors": list(canonical.isometry_errors),
            "symmetry_errors": list(canonical.symmetry_errors),
            "factorization_errors": list(canonical.factorization_errors),
            "module_vs_homogeneous_discovery_abs": abs(
                module_discovery - homogeneous_discovery
            ),
            "module_vs_canonical_discovery_abs": abs(module_discovery - canonical_discovery),
            "module_vs_homogeneous_confirmation_abs": abs(
                module_confirmation - homogeneous_confirmation
            ),
            "module_vs_canonical_confirmation_abs": abs(
                module_confirmation - canonical_confirmation
            ),
            "sampled_logit_reconstruction": reconstruction,
            "sampled_logit_reconstruction_count": min(256, len(discovery[0])),
            "checkpoint_reload_max_absolute_logit_error": checkpoint_reload_max_absolute,
            "sampled_logit_gate_max_absolute": 1e-8,
            "sampled_logit_gate_pass": all(
                item["max_absolute_logit_error"] <= 1e-8
                for item in reconstruction.values()
            ),
        },
        "gauge_audit": {
            "base_spectra_by_bond": gauge_base_spectra,
            "resolved_ranks_by_bond": gauge_resolved_ranks_by_bond,
            "required_ranks_by_bond": [
                list(required) for required in required_ranks_by_bond
            ],
            "records": gauge_records,
            "max_raw_coordinate_replay_relative": max(
                record["raw_coordinate_replay_relative"] for record in gauge_records
            ),
            "max_canonicalized_function_relative": max(
                record["canonicalized_function_relative"] for record in gauge_records
            ),
            "max_canonicalized_coefficient_replay_relative": max(
                record["canonicalized_coefficient_replay_relative"]
                for record in gauge_records
            ),
            "max_sampled_single_bond_logit_replay_relative": max(
                record["maximum_sampled_single_bond_logit_replay_relative"]
                for record in gauge_records
            ),
            "max_singular_extrema_relative_error": max(
                singular_extrema_relative_errors
            ),
            "max_condition_relative_error": max(condition_relative_errors),
            "minimum_nonsymmetry_relative": min(
                record["minimum_realized_nonsymmetry_relative"]
                for record in gauge_records
            ),
            "max_prefix_overlap_span_defect_diagnostic": max(
                bond_record["prefix_overlap_span_defect"]
                for record in gauge_records
                for bond_record in record["bond_realizations"]
            ),
            "max_transport_orthogonality": max(
                bond_record["transport_orthogonality"]
                for record in gauge_records
                for bond_record in record["bond_realizations"]
            ),
            "max_raw_factor_transport_relative_diagnostic": max(
                bond_record["raw_factor_transport_relative_diagnostic"]
                for record in gauge_records
                for bond_record in record["bond_realizations"]
            ),
            "max_gram_covariance_relative": max(
                bond_record["gram_covariance_relative"]
                for record in gauge_records
                for bond_record in record["bond_realizations"]
            ),
            "max_spectrum_relative": max(
                bond_record["spectrum_relative"]
                for record in gauge_records
                for bond_record in record["bond_realizations"]
            ),
            "projector_certificate_count": len(projector_certificates),
            "expected_required_projector_certificate_count": (
                expected_required_projector_certificate_count
            ),
            "required_projector_certificate_count": len(
                required_projector_certificates
            ),
            "projector_certificate_coverage_complete": all(
                record["projector_certificate_pair_count"]
                == record["expected_projector_certificate_pair_count"]
                for record in gauge_records
            ) and all(
                [
                    item["rank"]
                    for item in bond_record["projector_certificates"]
                ]
                == gauge_resolved_ranks_by_bond[bond]
                for record in gauge_records
                for bond, bond_record in enumerate(record["bond_realizations"])
            ),
            "maximum_required_projector_overlap_support_defect": (
                max(
                    item["maximum_supported_overlap_defect"]
                    for item in required_projector_certificates
                )
                if required_projector_certificates
                else None
            ),
            "all_required_projectors_numerically_certifiable": (
                len(required_projector_certificates)
                == expected_required_projector_certificate_count
                and expected_required_projector_certificate_count > 0
                and all(
                    item["numerically_certifiable"] is True
                    for item in required_projector_certificates
                )
            ),
            "maximum_required_projector_bound_with_roundoff": (
                max(
                    item["bound_with_roundoff"]
                    for item in required_projector_certificates
                    if item["bound_with_roundoff"] is not None
                )
                if required_projector_certificates
                and all(
                    item["bound_with_roundoff"] is not None
                    for item in required_projector_certificates
                )
                else None
            ),
            "all_required_projectors_pass_bound": (
                len(required_projector_certificates)
                == expected_required_projector_certificate_count
                and expected_required_projector_certificate_count > 0
                and all(
                    item["passes_bound"] is True
                    for item in required_projector_certificates
                )
            ),
            "matrix_artifact": gauge_matrix_manifest,
        },
        "canonical_spectrum": spectrum,
        "canonical_eigengap_diagnostics": gap_diagnostics,
        "canonical_trace_tail_fraction": {
            str(rank): sum(nonnegative_spectrum[rank:]) / trace_total for rank in ranks
        },
        "curves": curves,
        "canonical_rank_selection": canonical_rank_selection,
        "haar_curves": haar,
        "haar_discovery_selected_confirmation": haar_selected,
        "discovery_selected_confirmation": selected,
        "simultaneous_uniform_all_bond_curve": all_bond_curve,
        "provenance": {
            "source_sha256": source_sha256,
            "frozen_source_manifest": (
                {
                    "path": str(manifest_path),
                    "sha256": digest(manifest_path),
                    "text": manifest_path.read_text(),
                }
                if manifest_path.exists()
                else None
            ),
            "dataset_root": str(args.data_root.resolve()),
            "dataset_files": dataset_files,
            "checkpoint": {
                "path": str(args.checkpoint_output),
                "bytes": args.checkpoint_output.stat().st_size,
                "sha256": digest(args.checkpoint_output),
            },
            "runtime": {
                **environment_provenance(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            },
        },
    }

    algebraic_gates = {
        "sampled_logit_reconstruction_le_1e-8": result["canonicalization"][
            "sampled_logit_gate_pass"
        ],
        "factorization_relative_le_1e-10": max(canonical.factorization_errors) <= 1e-10,
        "row_isometry_le_1e-10": max(canonical.isometry_errors) <= 1e-10,
        "gauge_sampled_single_bond_logit_replay_relative_le_1e-8": result["gauge_audit"][
            "max_sampled_single_bond_logit_replay_relative"
        ] <= 1e-8,
        "gauge_singular_extrema_relative_le_1e-12": result["gauge_audit"][
            "max_singular_extrema_relative_error"
        ] <= 1e-12,
        "gauge_condition_relative_le_1e-12": result["gauge_audit"][
            "max_condition_relative_error"
        ] <= 1e-12,
        "gauge_nonsymmetry_relative_gt_1e-6": result["gauge_audit"][
            "minimum_nonsymmetry_relative"
        ] > 1e-6,
        "gauge_coefficient_replay_relative_le_1e-8": result["gauge_audit"][
            "max_canonicalized_coefficient_replay_relative"
        ] <= 1e-8,
        "gauge_transport_orthogonality_le_1e-10": result["gauge_audit"][
            "max_transport_orthogonality"
        ] <= 1e-10,
        "gauge_gram_covariance_relative_le_1e-8": result["gauge_audit"][
            "max_gram_covariance_relative"
        ] <= 1e-8,
        "gauge_spectrum_relative_le_1e-8": result["gauge_audit"][
            "max_spectrum_relative"
        ] <= 1e-8,
        "gauge_projector_certificate_coverage_complete": result["gauge_audit"][
            "projector_certificate_coverage_complete"
        ],
        "gauge_required_projector_overlap_support_defect_le_1e-8": (
            result["gauge_audit"][
                "maximum_required_projector_overlap_support_defect"
            ] is not None
            and result["gauge_audit"][
                "maximum_required_projector_overlap_support_defect"
            ] <= 1e-8
        ),
        "gauge_required_projectors_numerically_certifiable": result[
            "gauge_audit"
        ]["all_required_projectors_numerically_certifiable"],
        "gauge_required_projectors_pass_nonvacuous_bound": (
            result["gauge_audit"]["all_required_projectors_pass_bound"] is True
            and result["gauge_audit"][
                "maximum_required_projector_bound_with_roundoff"
            ] is not None
            and result["gauge_audit"][
                "maximum_required_projector_bound_with_roundoff"
            ] <= GAUGE_CERTIFICATE_MAX_BOUND
        ),
        "selected_canonical_roundoff_only_prefilter_pass": result[
            "canonical_rank_selection"
        ]["prefilter_by_rank"][
            str(result["canonical_rank_selection"]["selected_rank"])
        ]["passes"]
        is True,
        "selected_canonical_eigengap_resolved": result[
            "discovery_selected_confirmation"
        ]["canonical_tree_odt"]["eigengap_resolved"] is True,
        "all_bond_uniform_schedule_exact": (
            result["simultaneous_uniform_all_bond_curve"]["bond_dimensions"]
            == list(ALL_BOND_EXPECTED_DIMENSIONS)
            and result["simultaneous_uniform_all_bond_curve"][
                "target_removal_basis_points"
            ]
            == list(ALL_BOND_TARGET_REMOVAL_BASIS_POINTS)
            and [
                point["rank_tuple"][0]
                for point in result["simultaneous_uniform_all_bond_curve"]["points"]
            ]
            == list(ALL_BOND_EXPECTED_UNIFORM_RANKS)
            and all(
                len(set(point["rank_tuple"])) == 1
                for point in result["simultaneous_uniform_all_bond_curve"]["points"]
            )
        ),
        "all_bond_trace_totals_relative_le_1e-8": result[
            "simultaneous_uniform_all_bond_curve"
        ]["maximum_trace_total_relative_disagreement"]
        <= 1e-8,
        "all_bond_full_rank_coefficient_reconstruction_le_1e-8": result[
            "simultaneous_uniform_all_bond_curve"
        ]["points"][0]["coefficient_certificate"][
            "actual_relative_coefficient_error"
        ]
        <= 1e-8,
        "all_bond_selected_canonical_eigengaps_resolved": result[
            "simultaneous_uniform_all_bond_curve"
        ]["points"][
            result["simultaneous_uniform_all_bond_curve"]["discovery_selection"][
                "selected_schedule_index"
            ]
        ]["canonical_all_boundaries_resolved"]
        is True,
        "all_bond_selected_canonical_roundoff_only_prefilter_pass": result[
            "simultaneous_uniform_all_bond_curve"
        ]["points"][
            result["simultaneous_uniform_all_bond_curve"]["discovery_selection"][
                "selected_schedule_index"
            ]
        ]["canonical_all_boundaries_roundoff_prefilter_pass"]
        is True,
        "all_bond_coefficient_errors_within_hsvd_bounds_plus_declared_numerical_slack": all(
            point["coefficient_certificate"][
                "actual_error_within_bound_plus_declared_numerical_slack"
            ] is True
            for point in result["simultaneous_uniform_all_bond_curve"]["points"]
        ),
    }
    if set(algebraic_gates) != EXPECTED_ALGEBRAIC_GATES:
        raise RuntimeError("internal algebraic gate inventory mismatch")
    capability_gates = {
        "module_discovery_accuracy_ge_minimum": (
            module_discovery >= args.minimum_accuracy
        ),
        "module_confirmation_accuracy_ge_minimum": (
            module_confirmation >= args.minimum_accuracy
        ),
    }
    if set(capability_gates) != EXPECTED_CAPABILITY_GATES:
        raise RuntimeError("internal capability gate inventory mismatch")
    result["algebraic_gates"] = algebraic_gates
    result["capability_gates"] = capability_gates
    result["status"] = (
        "complete"
        if all(value is True for value in algebraic_gates.values())
        and all(value is True for value in capability_gates.values())
        else "gate_failed"
    )

    final_entries = verify_source_freeze(project_root, manifest_path)
    if final_entries != frozen_entries or digest(manifest_path) != manifest_sha256_at_start:
        raise RuntimeError("source freeze changed during the run")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(args.output), "selected": selected}, indent=2), flush=True)
    if result["status"] != "complete":
        raise RuntimeError(
            f"canonical ODT gates failed: algebraic={algebraic_gates}, "
            f"capability={capability_gates}"
        )


if __name__ == "__main__":
    main()
