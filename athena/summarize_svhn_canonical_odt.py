"""Aggregate the frozen three-seed canonical ODT SVHN experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from xvla.models.chi_mlp import ChiMLP, ChiMLPConfig
from xvla.nn.normalization import RmsBatchNorm
from xvla.train.canonical_odt import (
    HomogeneousChiTN,
    apply_bond_gauge,
    canonical_environments,
    canonical_prefix_overlaps,
    canonicalize_homogeneous,
    coefficient_error_squared,
    coefficient_inner_product,
    export_homogeneous_network,
    homogeneous_forward,
    polar_orthogonal_transport,
    projector_perturbation_certificate,
    sorted_eigensystem,
)

try:
    from athena.launch_svhn_canonical_odt import (
        DAG_EDGES,
        DAG_NODE_ORDER,
        DAG_SCHEMA,
        LAUNCH_SCHEMA,
    )
    from athena.verify_canonical_odt_freeze import (
        RESULT_SOURCE_PATHS,
        SMOKE_REQUIRED_AUTHENTICATION_IDENTITIES,
        UNIT_CERTIFICATE_SCHEMA,
        reject_symlinks_in_directory,
        require_confined_path,
        require_physical_root,
        validate_required_authentication_decisions,
        verify as verify_source_freeze,
    )
except ModuleNotFoundError:
    from launch_svhn_canonical_odt import (
        DAG_EDGES,
        DAG_NODE_ORDER,
        DAG_SCHEMA,
        LAUNCH_SCHEMA,
    )
    from verify_canonical_odt_freeze import (
        RESULT_SOURCE_PATHS,
        SMOKE_REQUIRED_AUTHENTICATION_IDENTITIES,
        UNIT_CERTIFICATE_SCHEMA,
        reject_symlinks_in_directory,
        require_confined_path,
        require_physical_root,
        validate_required_authentication_decisions,
        verify as verify_source_freeze,
    )


SCHEMA = "xvla-canonical-tree-odt-svhn-extension-v3r5"
SUMMARY_SCHEMA = "xvla-canonical-tree-odt-svhn-summary-v3r5"
SMOKE_VALIDATION_SCHEMA = "xvla-canonical-tree-odt-smoke-validation-v1"
PREFETCH_SCHEMA = "xvla-canonical-tree-odt-svhn-prefetch-v3"
GAUGE_CERTIFICATE_MAX_BOUND = 1e-6
GAUGE_SEED = 2026090200
GAUGE_ROUNDOFF_MULTIPLIER = 10_000.0
GAUGE_SELECTION_TRIALS = 0
RAW_GAUGE_STREAM_RELATIVE_TOLERANCE = 1e-12
INDEPENDENT_MATRIX_RECONSTRUCTION_RELATIVE_TOLERANCE = 5e-8
NONREQUIRED_DIAGNOSTIC_PRIMITIVE_ABSOLUTE_TOLERANCE = 1e-8
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
EXPECTED_METHODS = {
    "canonical_tree_odt",
    "canonical_adjacent_local",
    "legacy_raw_downstream_zero_anchor",
    "activation_pca_centered",
}
EXPECTED_SUMMARY_GATES = {
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
EXPECTED_EMPIRICAL_GATES = {
    "all_bond_selected_confirmation_noninferior_in_all_seeds",
    "all_bond_selected_odt_gt_local_with_resolved_boundaries_in_all_seeds",
    "all_bond_selected_haar_randomization_p_le_0p05_in_all_seeds",
}
EXPECTED_DATASET_FILES = {
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
EXPECTED_PROTOCOL_KEYS = {
    "epochs", "batch_size", "optimizer", "learning_rate", "weight_decay",
    "betas", "gradient_clip", "post_training_rbn_calibration_batches",
    "post_training_rbn_calibration_batch_size", "post_training_rbn_calibration_split",
    "model_width", "homogeneous_bond_width",
    "layers", "analysis_bond", "analysis_bond_description", "rank_grid",
    "haar_draws", "gauge_trials", "minimum_full_model_accuracy",
    "maximum_gauge_condition", "gauge_family", "gauge_transport",
    "gauge_projector_roundoff_multiplier", "gauge_required_projector_maximum_bound",
    "gauge_required_ranks_by_bond", "smoke_nontrivial_diagnostic_rank", "split_seed",
    "rank_eligibility_policy", "rank_eligibility_selection_gauge_trial_count",
    "rank_eligibility_validation_gauge_seed",
    "rank_eligibility_validation_gauge_trial_count",
    "rank_eligibility_validation_gauges_used_for_selection",
    "rank_eligibility_full_rank_fallback_allowed",
    "discovery_count", "confirmation_count", "training_split",
    "deterministic_algorithms", "discovery_indices_sha256",
    "confirmation_indices_sha256", "selection_accuracy_tolerance",
    "gauge_replay_sample_count", "gauge_replay_sample_indices_sha256",
    "all_bond_target_removal_basis_points", "all_bond_expected_uniform_ranks",
    "all_bond_haar_draws", "all_bond_haar_seed",
    "all_bond_bound_relative_slack", "all_bond_bound_absolute_slack_fraction",
}
EXPECTED_PROTOCOL_COMMON = {
    "epochs": 20,
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
    "analysis_bond": 1,
    "analysis_bond_description": "homogeneous output of the first bilinear core",
    "gauge_trials": 50,
    "minimum_full_model_accuracy": 0.70,
    "selection_accuracy_tolerance": 0.01,
    "maximum_gauge_condition": 1000.0,
    "gauge_family": "independent-left-right-orthogonal-balanced-log-singular-v2",
    "gauge_transport": "recursive-canonical-prefix-overlap-polar-v1",
    "gauge_projector_roundoff_multiplier": 10_000.0,
    "gauge_required_projector_maximum_bound": GAUGE_CERTIFICATE_MAX_BOUND,
    "rank_eligibility_policy": "analytic-roundoff-only-prefilter-v1",
    "rank_eligibility_selection_gauge_trial_count": GAUGE_SELECTION_TRIALS,
    "rank_eligibility_validation_gauge_seed": GAUGE_SEED,
    "rank_eligibility_validation_gauge_trial_count": 50,
    "rank_eligibility_validation_gauges_used_for_selection": False,
    "rank_eligibility_full_rank_fallback_allowed": True,
    "split_seed": 20260902,
    "discovery_count": 4000,
    "confirmation_count": 22032,
    "training_split": "official SVHN train only",
    "deterministic_algorithms": True,
    "gauge_replay_sample_count": 16,
    "gauge_replay_sample_indices_sha256": "a7d474fd63473ac9800ee6078183888b6a2b1887ded5b9f8e789abf3928d8a27",
    "all_bond_target_removal_basis_points": [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9500, 9700],
    "all_bond_expected_uniform_ranks": [33, 30, 27, 24, 20, 17, 14, 10, 7, 4, 2, 1],
    "all_bond_haar_draws": 64,
    "all_bond_haar_seed": 700000,
    "all_bond_bound_relative_slack": 1e-8,
    "all_bond_bound_absolute_slack_fraction": 1e-10,
}
EXPECTED_DISCOVERY_SHA256 = "e9fb0b7eecabd6334bbcbf568816fdb1fbcd4cf7afa8869ea16502078b1d455d"
EXPECTED_CONFIRMATION_SHA256 = "9e85f62663f021376c408a0aff941b94399c7bb7d2cabe86ab77275a9589ffa4"
EXPECTED_V1_FAILURE = {
    "schema": "xvla-canonical-tree-odt-svhn-predecessor-failure-v1",
    "job_id": "834390",
    "result_schema": "xvla-canonical-tree-odt-svhn-v1",
    "immutable_root": "/work/joy/x-vla-odt-20260902-canonicalr1",
    "source_manifest_sha256": "6497030d8f9b5b4848c37f5a64980fe22c6702102b3a2ce0105435c6da6609a3",
    "result_path": "/work/joy/x-vla-odt-20260902-canonicalr1/results/smoke.json",
    "result_sha256": "652a63705522e63a33f1b257bb44b979c8bff3baccc9dafbe6a38cfe7d5ecc20",
    "failed_gate": "gauge_function_relative_le_1e-8",
    "observed_relative_error": 4.201135487728006e-8,
    "threshold": 1e-8,
}
EXPECTED_V2R3_FAILURE = {
    "schema": "xvla-canonical-tree-odt-svhn-predecessor-failure-v2r3",
    "job_id": "834408",
    "result_schema": "xvla-canonical-tree-odt-svhn-v2",
    "immutable_root": "/work/joy/x-vla-odt-20260902-canonicalv2r3",
    "source_manifest_sha256": "ef5d045356d3ac52f8455c69e66a663c93f52b50da68708bdadfcac4a185802a",
    "result_path": "/work/joy/x-vla-odt-20260902-canonicalv2r3/results/smoke.json",
    "result_sha256": "c8a249a13023422e0a197a0e417ac3c8b09dfa416afe3754586c9f15367a69e5",
    "failed_gate": "gauge_resolved_projector_relative_le_1e-8",
    "observed_relative_error": 1.1474492274034916e-7,
    "threshold": 1e-8,
    "empirical_positive_control_failure": (
        "the discovery-selected uniform all-bond point retained full rank, so it "
        "did not establish a nontrivial ODT-over-local compression result"
    ),
}
EXPECTED_V3R1_FAILURE = {
    "schema": "xvla-canonical-tree-odt-svhn-predecessor-failure-v3r1",
    "immutable_root": "/work/joy/x-vla-canonical-v3r1-production-r4",
    "source_manifest_sha256": "0e2ac659f7efd85fa8ce98b318aff6b0b364fe4644dafbade46001e5523b9a76",
    "launch_sha256": "d48faa50a705073e7ee9bb0e0bc3b43cc4db55e39f904ac2535d3494399c659b",
    "job_ids": {
        "tests": "834518", "prefetch": "834519", "smoke": "834520",
        "train_array": "834521", "summary": "834522",
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
    "summary_failure_sha256": "4ae60f3e84c0077a41a4cb906e02f9f532018aa7079f0dd3e7c707466ef4f348",
}
EXPECTED_V3R2_R5_FAILURE = {
    "schema": "xvla-canonical-tree-odt-svhn-predecessor-failure-v3r2-r5",
    "immutable_root": "/work/joy/x-vla-canonical-v3r2-production-r5",
    "source_manifest_sha256": "1a4cbce71b09a8550a52fdf11e1c0c224c694d7c2e7b09a6d58acc0dd3c0248b",
    "launch_sha256": "82065d24b8c71352723ece635d4753643d195ed82c669a3d4c2a9bae345fdb2c",
    "job_ids": {
        "tests": "834531", "prefetch": "834532", "smoke": "834533",
        "train_array": "834534", "summary": "834535",
    },
    "classification": "engineering_preflight_failure_no_science",
    "unit_test_result": "131 passed, 58 failed",
    "failure": "NumPy float64 escaped the finite-JSON test fixture",
    "unit_test_log_sha256": "72eb9a329411dc071367596dc022a758482670dcdbc496e88affddebdc72e90f",
    "summary_failure_sha256": "0e373e83fa1577c526e76fb575de4c05325c088572f1de5ae367704e53d49f55",
}
EXPECTED_V3R2_R6_FAILURE = {
    "schema": "xvla-canonical-tree-odt-svhn-predecessor-failure-v3r2-r6",
    "immutable_root": "/work/joy/x-vla-canonical-v3r2-production-r6",
    "source_manifest_sha256": "4bf4c30fab0926fd33b890ca189d299e18eb87346031036c5c58a2066d16baca",
    "launch_sha256": "e16e8aea18e24b90573f83b7cc794f4d5fd6f36e41cd72f303678c26040b4e24",
    "job_ids": {
        "tests": "834538", "prefetch": "834539", "smoke": "834540",
        "train_array": "834541", "summary": "834542",
    },
    "classification": "engineering_preflight_failure_no_science",
    "unit_test_result": "188 passed, 1 failed",
    "failure": "four fail-closed protocol cases ran inside a monkeypatch scope",
    "unit_test_log_sha256": "74619cb2151cafc2ed7ab6c07bef1e38fa0961cba74f4648d67a02e0501232f1",
    "summary_failure_sha256": "caaa3633eb240fb5625b85138fb4e62c9f2a7dc2cb3852b79df55474de7e16c2",
}
EXPECTED_V3R2_R7_FAILURE = {
    "schema": "xvla-canonical-tree-odt-svhn-predecessor-failure-v3r2-r7",
    "immutable_root": "/work/joy/x-vla-canonical-v3r2-production-r7",
    "source_manifest_sha256": "756427412047740f5411c0dfc373894fd071322a6995335ed4a62636b19e9f95",
    "launch_sha256": "e71b7fa16b3e2eb7d380fc437bd90f467ed5048d8b7ebb45f17a667475bcee63",
    "job_ids": {
        "tests": "834545", "prefetch": "834546", "smoke": "834547",
        "train_array": "834548", "summary": "834549",
    },
    "classification": "engineering_authentication_failure_no_seed_science",
    "unit_test_result": "189 passed, 0 failed",
    "smoke_status": "complete",
    "smoke_json_sha256": "8bb6c2d16b34b1993c5d65ced7047aeb3606c88a8c3a3d8d75817fad57d71d9f",
    "failure": (
        "CPU reconstruction rejected a non-required near-degenerate rank-31 "
        "Davis-Kahan diagnostic produced on GPU"
    ),
    "train_array_disposition": "cancelled_before_any_seed_result_completed",
    "summary_failure_sha256": "456c85524a0e16925583c008052f776f9900b4dcb308d6fbaa01899b85539e81",
}
EXPECTED_V3R3_R8_FAILURE = {
    "schema": "xvla-canonical-tree-odt-svhn-predecessor-failure-v3r3-r8",
    "immutable_root": "/work/joy/x-vla-canonical-v3r3-production-r8",
    "source_manifest_sha256": "47b994c6fe2661a007323f467835a892e43fd09cb144f2f3377844eabb7cdb20",
    "launch_sha256": "925398f97a451190d9292b80b76a2aefb13e4437cc8b5ccb5b72210c9b07343b",
    "job_ids": {
        "tests": "834553", "prefetch": "834554", "smoke": "834555",
        "train_array": "834556", "summary": "834557",
    },
    "classification": "engineering_authentication_failure_no_seed_science",
    "unit_test_result": "192 passed, 0 failed",
    "smoke_status": "producer_complete_independent_authentication_failed",
    "smoke_json_sha256": "478b5e543ce2a6e4b7d8b0541d88e7390472cb1a541d3789bceff526e9f8a885",
    "smoke_checkpoint_sha256": "77d3609683865e4299f2811cc8e6bb7b5b3e4f97f234263cf3e0b07ed9bf571a",
    "smoke_gauge_npz_sha256": "e8219293511951a9d539f216f1a20959286665f42cf0c448b9b3dd80a29f4b3c",
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
    "summary_failure_sha256": "21ea6a8a686bd707c198ce63553ed17e6627c33fd23e5007b138e92aef982cd0",
}
EXPECTED_V3R4_R9_FAILURE = {
    "schema": "xvla-canonical-tree-odt-svhn-predecessor-failure-v3r4-r9",
    "immutable_root": "/work/joy/x-vla-canonical-v3r4-production-r9",
    "source_manifest_sha256": "45c159d6559c01ad008931b0a4ad7d08c144937ce74eddf3670ebd8133075e15",
    "launch_sha256": "f7efc091a6f91dd36fee50b180db6ce8585726f5b72c97f941466323191b7798",
    "job_ids": {"tests": "834561", "prefetch": "834562", "smoke": "834563", "train_array": "834564", "summary": "834565"},
    "classification": "engineering_authentication_failure_no_seed_science",
    "unit_test_result": "194 passed, 0 failed",
    "smoke_status": "producer_complete_independent_authentication_failed",
    "smoke_json_sha256": "f1c9d9a0eb0af33343b1950726eecad1a5a420ea51e286c07395fcc2678a53c9",
    "smoke_checkpoint_sha256": "77d3609683865e4299f2811cc8e6bb7b5b3e4f97f234263cf3e0b07ed9bf571a",
    "smoke_gauge_npz_sha256": "e8219293511951a9d539f216f1a20959286665f42cf0c448b9b3dd80a29f4b3c",
    "failure": "cross-device scalar equality rejected two explicitly non-load-bearing simultaneous replay diagnostics although all 98,780 other comparisons and all 200 three-way required certificates passed",
    "login_cpu_mismatches": [
        {"trial": 49, "measurement": "raw_coordinate_replay_relative", "producer": 7.103200689715422e-8, "independent_cpu": 4.765571981717182e-8, "absolute_disagreement": 2.3376287079982402e-8},
        {"trial": 47, "measurement": "canonicalized_function_relative", "producer": 3.587349506016325e-8, "independent_cpu": 1.594611819870762e-8, "absolute_disagreement": 1.992737686145563e-8},
    ],
    "summary_compute_observation": "an independent summary-node CPU path first disagreed at trial 43 raw replay",
    "all_200_required_decisions_agree_and_pass": True,
    "maximum_required_bound_with_roundoff": 1.5973597226370406e-9,
    "maximum_required_support_defect": 2.7873892849173025e-14,
    "train_array_disposition": "cancelled_before_any_seed_result_completed",
    "summary_failure_sha256": "25dbf5a516cdfd7bdf68448b49ff076eb3ec911e04e9fac4053188007971cf83",
}
EXPECTED_PREDECESSOR_FAILURES = [
    EXPECTED_V1_FAILURE,
    EXPECTED_V2R3_FAILURE,
    EXPECTED_V3R1_FAILURE,
    EXPECTED_V3R2_R5_FAILURE,
    EXPECTED_V3R2_R6_FAILURE,
    EXPECTED_V3R2_R7_FAILURE,
    EXPECTED_V3R3_R8_FAILURE,
    EXPECTED_V3R4_R9_FAILURE,
]
ALL_BOND_DIMENSIONS = (33, 33, 33, 33)
ALL_BOND_MULTIPLICITIES = (8, 4, 2, 1)
EXPECTED_CLAIM_BOUNDARY = (
    "Canonical weight-only ODT of the topology-specific homogeneous chi-MLP tree, "
    "including a frozen uniform simultaneous all-bond Algorithm-3 curve. "
    "This is a new positive control, not a reproduction of the paper's SVHN setup or numbers. "
    "This is not the symmetric polynomial quotient, attention ODT, a VLA decomposition, "
    "or evidence of semantic mechanisms."
)
EXPECTED_ALL_BOND_CLAIM_BOUNDARY = (
    "Simultaneous Algorithm-3 truncation of all four bonds in the canonical "
    "topology-specific independent-clone coefficient tree. Uniform percentage "
    "removal is a frozen evaluation convention, not a globally optimal rank allocation, "
    "a symmetric-polynomial quotient, an attention decomposition, or a VLA decomposition."
)
EXPECTED_HAAR_CONSTRUCTION = (
    "one independently seeded square Gaussian QR basis per seed, draw, and bond; "
    "leading columns are nested across the complete frozen schedule"
)
FULL_RANK_GRID = [1, 2, 4, 6, 8, 12, 16, 24, 32, 33]
SMOKE_RANK_GRID = [1, 32, 33]


def validate_launch_binding(
    launch: dict,
    *,
    project_root: Path,
    manifest_sha256: str,
    current_summary_job_id: str | None,
    current_node: str = "summary",
) -> dict[str, str]:
    """Validate the exact launch DAG and bind this process to its declared node."""

    expected_top_keys = {
        "schema", "source_manifest_sha256", "run_root", "data_root",
        "submission_protocol", "dag",
    }
    if set(launch) != expected_top_keys or launch.get("schema") != LAUNCH_SCHEMA:
        raise RuntimeError("launch record schema or field inventory mismatch")
    project_root = require_physical_root(project_root)
    if launch.get("run_root") != str(project_root):
        raise RuntimeError("launch record run root does not match this physical run root")
    if launch.get("source_manifest_sha256") != manifest_sha256:
        raise RuntimeError("launch record does not match the frozen source manifest")
    if launch.get("submission_protocol") != {
        "first_node_submitted_held": True,
        "launch_record_written_before_release": True,
    }:
        raise RuntimeError("launch record does not certify transactional submission")
    raw_data_root = launch.get("data_root")
    if not isinstance(raw_data_root, str):
        raise RuntimeError("launch record data root is not an absolute physical path")
    data_root = Path(raw_data_root).resolve()
    if raw_data_root != str(data_root):
        raise RuntimeError("launch record data root is not an absolute physical path")
    if (
        data_root == project_root
        or project_root in data_root.parents
        or data_root in project_root.parents
    ):
        raise RuntimeError("launch record data root is inside the frozen run")

    dag = launch.get("dag")
    if not isinstance(dag, dict) or set(dag) != {
        "schema", "node_order", "nodes", "edges", "terminal_node",
        "terminal_dependency_expression",
    }:
        raise RuntimeError("launch DAG field inventory mismatch")
    if (
        dag.get("schema") != DAG_SCHEMA
        or dag.get("node_order") != list(DAG_NODE_ORDER)
        or dag.get("terminal_node") != "summary"
    ):
        raise RuntimeError("launch DAG schema, order, or terminal node mismatch")
    expected_edges = [
        {
            "upstream": upstream,
            "downstream": downstream,
            "slurm_dependency": dependency_mode,
        }
        for upstream, downstream, dependency_mode in DAG_EDGES
    ]
    if dag.get("edges") != expected_edges:
        raise RuntimeError("launch DAG dependency edges mismatch")

    expected_specifications = {
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
    nodes = dag.get("nodes")
    if not isinstance(nodes, dict) or tuple(nodes) != DAG_NODE_ORDER:
        raise RuntimeError("launch DAG node inventory or order mismatch")
    job_ids = {}
    for name in DAG_NODE_ORDER:
        node = nodes.get(name)
        if not isinstance(node, dict) or set(node) != {
            "job_id", "script", "arguments", "array"
        }:
            raise RuntimeError(f"launch DAG node {name} field inventory mismatch")
        script, arguments, array = expected_specifications[name]
        if (
            node.get("script") != script
            or node.get("arguments") != arguments
            or node.get("array") != array
        ):
            raise RuntimeError(f"launch DAG node {name} specification mismatch")
        job_id = node.get("job_id")
        if not isinstance(job_id, str) or not job_id.isdecimal():
            raise RuntimeError(f"launch DAG node {name} has an invalid Slurm job id")
        job_ids[name] = job_id
    if len(set(job_ids.values())) != len(job_ids):
        raise RuntimeError("launch DAG job ids are not unique")
    expected_terminal_expression = (
        f"afternotok:{job_ids['smoke_validate']}?afterany:{job_ids['train_array']}"
    )
    if dag.get("terminal_dependency_expression") != expected_terminal_expression:
        raise RuntimeError("launch DAG terminal OR dependency differs")
    if current_node not in job_ids:
        raise RuntimeError("requested launch DAG node is unknown")
    if current_summary_job_id is None or current_summary_job_id != job_ids[current_node]:
        raise RuntimeError(f"process Slurm job is not launch DAG node {current_node}")
    return job_ids


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_seed_result_paths(results: Path) -> list[Path]:
    """Require exactly the three prospectively named seed-result files."""

    expected = [results / f"seed_{seed}.json" for seed in range(3)]
    if sorted(results.glob("seed_*.json")) != expected:
        raise RuntimeError(
            "seed result filenames must be exactly seed_0.json, seed_1.json, and seed_2.json"
        )
    return expected


def require_seed_file_binding(records: list[dict], paths: list[Path]) -> None:
    """Bind each embedded seed to the prospectively named JSON that contains it."""

    if len(records) != 3 or len(paths) != 3:
        raise RuntimeError("exactly three seed records and paths are required")
    for expected_seed, (record, path) in enumerate(zip(records, paths)):
        if path.name != f"seed_{expected_seed}.json" or record.get("seed") != expected_seed:
            raise RuntimeError("seed JSON content does not match its exact filename")


def artifact_hash_manifest(
    *,
    launch_path: Path,
    certificate_path: Path,
    test_log: Path,
    prefetch_path: Path,
    smoke_validation_path: Path,
    smoke_path: Path,
    smoke_gauge_path: Path,
    records: list[dict],
    seed_paths: list[Path],
    seed_gauge_paths: list[Path],
) -> dict:
    """Hash every load-bearing input artifact consumed by the summary."""

    return {
        "launch_json": digest(launch_path),
        "unit_certificate_json": digest(certificate_path),
        "unit_test_log": digest(test_log),
        "prefetch_json": digest(prefetch_path),
        "smoke_validation_json": digest(smoke_validation_path),
        "smoke_json": digest(smoke_path),
        "smoke_checkpoint_pt": digest(smoke_path.with_suffix(".pt")),
        "smoke_gauge_matrices_npz": digest(smoke_gauge_path),
        "seed_json_by_seed": {
            str(record["seed"]): digest(path)
            for record, path in zip(records, seed_paths)
        },
        "gauge_matrices_npz_by_seed": {
            str(record["seed"]): digest(path)
            for record, path in zip(records, seed_gauge_paths)
        },
        "checkpoint_pt_by_seed": {
            str(record["seed"]): digest(path.with_suffix(".pt"))
            for record, path in zip(records, seed_paths)
        },
    }


def validate_pre_summary_result_inventory(results: Path, test_log: Path) -> list[str]:
    """Reject missing, extra, non-file, or symlinked production artifacts."""

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
    children = list(results.iterdir())
    actual_names = {path.name for path in children}
    if actual_names != expected_names or len(children) != len(expected_names):
        raise RuntimeError(
            "pre-summary results artifact inventory mismatch: "
            f"expected={sorted(expected_names)}, actual={sorted(actual_names)}"
        )
    for path in children:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"pre-summary result is not a physical regular file: {path}")
    return sorted(expected_names)


def validate_smoke_validation_receipt(
    receipt: dict,
    *,
    source_manifest_sha256: str,
    jobs: dict[str, str],
    launch_path: Path,
    certificate_path: Path,
    test_log: Path,
    prefetch_path: Path,
    smoke_path: Path,
    gauge_path: Path,
    dataset_files: dict,
    authentication: dict | None = None,
) -> None:
    """Authenticate the fresh-process CPU release-gate receipt."""

    expected_keys = {
        "schema", "status", "source_manifest_sha256", "artifact_sha256",
        "input_inventory", "dataset_files", "producer_slurm_job_id",
        "validator_slurm_job_id", "required_projector_three_way_authentication",
    }
    expected_input_inventory = sorted({
        "launch.json",
        "canonical_odt_unit_certificate.json",
        "prefetch.json",
        "smoke.json",
        "smoke.pt",
        "smoke_gauge_matrices.npz",
        test_log.name,
    })
    expected_hashes = {
        "launch_json": digest(launch_path),
        "unit_certificate_json": digest(certificate_path),
        "unit_test_log": digest(test_log),
        "prefetch_json": digest(prefetch_path),
        "smoke_json": digest(smoke_path),
        "smoke_checkpoint_pt": digest(smoke_path.with_suffix(".pt")),
        "smoke_gauge_matrices_npz": digest(gauge_path),
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected_keys
        or receipt.get("schema") != SMOKE_VALIDATION_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("source_manifest_sha256") != source_manifest_sha256
        or receipt.get("artifact_sha256") != expected_hashes
        or receipt.get("input_inventory") != expected_input_inventory
        or receipt.get("dataset_files") != dataset_files
        or receipt.get("producer_slurm_job_id") != jobs["smoke"]
        or receipt.get("validator_slurm_job_id") != jobs["smoke_validate"]
        or not isinstance(
            receipt.get("required_projector_three_way_authentication"), dict
        )
    ):
        raise RuntimeError("fresh-process smoke-validation receipt differs")
    receipt_identities = validate_required_authentication_decisions(
        receipt["required_projector_three_way_authentication"],
        expected_identities=SMOKE_REQUIRED_AUTHENTICATION_IDENTITIES,
    )
    if authentication is not None:
        current_identities = validate_required_authentication_decisions(
            authentication,
            expected_identities=SMOKE_REQUIRED_AUTHENTICATION_IDENTITIES,
        )
        if current_identities != receipt_identities:
            raise RuntimeError("fresh-process smoke authentication inventory differs")


def md5_digest(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def rehash_official_svhn(root: Path) -> dict[str, dict[str, int | str]]:
    result = {}
    for filename in EXPECTED_DATASET_FILES:
        path = root / filename
        if not path.is_file():
            raise RuntimeError(f"official SVHN file is missing during aggregation: {path}")
        result[filename] = {
            "bytes": path.stat().st_size,
            "md5": md5_digest(path),
            "sha256": digest(path),
        }
    if result != EXPECTED_DATASET_FILES:
        raise RuntimeError("official SVHN files changed before aggregation")
    return result


def regenerate_gauge_replay_sample(dataset_root: Path) -> torch.Tensor:
    """Regenerate the frozen first 16 discovery images without trusting the NPZ."""

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4377, 0.4438, 0.4728),
                (0.198, 0.201, 0.197),
            ),
        ]
    )
    test = datasets.SVHN(
        dataset_root,
        split="test",
        download=False,
        transform=transform,
    )
    generator = torch.Generator().manual_seed(EXPECTED_PROTOCOL_COMMON["split_seed"])
    indices = torch.randperm(len(test), generator=generator)[:16].tolist()
    index_hash = hashlib.sha256(
        np.asarray(indices, dtype=np.int64).tobytes()
    ).hexdigest()
    if index_hash != EXPECTED_PROTOCOL_COMMON["gauge_replay_sample_indices_sha256"]:
        raise RuntimeError("regenerated gauge replay sample indices changed")
    return torch.stack([test[index][0] for index in indices]).flatten(1).double()


def regenerate_trial_gauges(trial: int, dimensions: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    """Regenerate the exact frozen CPU gauge stream for one indexed trial."""

    if isinstance(trial, bool) or not isinstance(trial, int) or trial < 0:
        raise ValueError("gauge trial must be a nonnegative integer")
    generator = torch.Generator().manual_seed(GAUGE_SEED + trial)
    condition = 10.0 ** (3.0 * trial / 49.0)
    half_log_condition = 0.5 * math.log10(condition)
    gauges = []
    for dimension in dimensions:
        left, _ = torch.linalg.qr(
            torch.randn(
                dimension,
                dimension,
                generator=generator,
                dtype=torch.float64,
            )
        )
        right, _ = torch.linalg.qr(
            torch.randn(
                dimension,
                dimension,
                generator=generator,
                dtype=torch.float64,
            )
        )
        exponents = torch.linspace(
            -half_log_condition,
            half_log_condition,
            dimension,
            dtype=torch.float64,
        )
        gauges.append(left @ torch.diag(torch.pow(10.0, exponents)) @ right.T)
    return tuple(gauges)


def export_checkpoint_homogeneous(
    record: dict,
    *,
    project_root: Path,
) -> HomogeneousChiTN:
    """Load and independently export the exact checkpoint named by a result."""

    seed = record["seed"]
    checkpoint_name = "smoke.pt" if seed == 9173 else f"seed_{seed}.pt"
    checkpoint_path = require_confined_path(
        project_root,
        project_root / "results" / checkpoint_name,
        label=f"seed {seed} checkpoint for independent export",
        kind="file",
    )
    config = ChiMLPConfig(
        in_dim=3072,
        dim=32,
        n_layers=3,
        num_classes=10,
        norm="scalar_rbn",
    )
    model = ChiMLP(config).double().eval()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    for norm in model.norms:
        if not isinstance(norm, RmsBatchNorm) or not bool(norm.initialized):
            raise RuntimeError("checkpoint contains an uninitialized scalar RBN")
        norm.freeze()
    return export_homogeneous_network(model, require_frozen_rbn=True)


def require_exact_true_gates(record: dict, key: str, expected: set[str]) -> None:
    gates = record.get(key)
    if not isinstance(gates, dict) or set(gates) != expected:
        raise RuntimeError(
            f"seed {record.get('seed')} {key} inventory mismatch: "
            f"{sorted(gates) if isinstance(gates, dict) else type(gates).__name__}"
        )
    if any(value is not True for value in gates.values()):
        raise RuntimeError(f"seed {record.get('seed')} has a non-passing {key}: {gates}")


def expected_eigengap(spectrum: list[float], rank: int) -> dict:
    if not spectrum or not 1 <= rank <= len(spectrum):
        raise RuntimeError("eigengap rank is outside the recorded spectrum")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in spectrum):
        raise RuntimeError("canonical spectrum contains a non-finite value")
    retained = float(spectrum[rank - 1])
    if rank == len(spectrum):
        return {
            "rank": rank,
            "retained_value": retained,
            "discarded_value": 0.0,
            "absolute_gap": None,
            "relative_gap": None,
            "resolved": True,
        }
    discarded = float(spectrum[rank])
    absolute = retained - discarded
    scale = max(abs(float(spectrum[0])), np.finfo(np.float64).tiny)
    relative = absolute / scale
    tolerance = 1000 * np.finfo(np.float64).eps * len(spectrum) * scale
    return {
        "rank": rank,
        "retained_value": retained,
        "discarded_value": discarded,
        "absolute_gap": absolute,
        "relative_gap": relative,
        "resolved": absolute > tolerance and relative > 1e-8,
    }


def expected_roundoff_relative_gap_floor(
    dimension: int,
    *,
    epsilon: float = float(np.finfo(np.float64).eps),
    roundoff_multiplier: float = GAUGE_ROUNDOFF_MULTIPLIER,
    maximum_bound: float = GAUGE_CERTIFICATE_MAX_BOUND,
) -> float:
    """Independently recompute the draw-free rank-eligibility boundary."""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise RuntimeError("roundoff-prefilter dimension is invalid")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise RuntimeError("roundoff-prefilter epsilon is invalid")
    if not math.isfinite(roundoff_multiplier) or roundoff_multiplier < 0:
        raise RuntimeError("roundoff-prefilter multiplier is invalid")
    if not math.isfinite(maximum_bound) or maximum_bound <= 0:
        raise RuntimeError("roundoff-prefilter maximum bound is invalid")
    arithmetic_slack = roundoff_multiplier * epsilon * dimension
    if maximum_bound <= arithmetic_slack:
        return math.inf
    return float(
        arithmetic_slack
        * (2.0 + math.sqrt(2.0) / (maximum_bound - arithmetic_slack))
    )


def expected_roundoff_prefilter(spectrum: list[float], rank: int) -> dict:
    """Recompute the runner's analytic prefilter without importing it."""

    dimension = len(spectrum)
    if not spectrum or not 1 <= rank <= dimension:
        raise RuntimeError("roundoff-prefilter rank is outside the spectrum")
    scale = max((abs(float(value)) for value in spectrum), default=0.0)
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("roundoff-prefilter spectrum scale is invalid")
    floor = expected_roundoff_relative_gap_floor(dimension)
    if rank == dimension:
        return {
            "rank": rank,
            "kind": "full_rank_identity_fallback",
            "uses_random_gauge_draws": False,
            "relative_gap": None,
            "minimum_relative_gap_for_maximum_bound": floor,
            "roundoff_only_eta": None,
            "separation_margin": None,
            "bound_with_roundoff": None,
            "maximum_bound": GAUGE_CERTIFICATE_MAX_BOUND,
            "passes": True,
        }
    gap = float(spectrum[rank - 1]) - float(spectrum[rank])
    relative_gap = gap / scale
    arithmetic_slack = (
        GAUGE_ROUNDOFF_MULTIPLIER * float(np.finfo(np.float64).eps) * dimension
    )
    eta = arithmetic_slack * scale
    separation = gap - 2.0 * eta
    bound = (
        math.sqrt(2.0) * eta / separation + arithmetic_slack
        if separation > 0
        else None
    )
    return {
        "rank": rank,
        "kind": "nontrivial_roundoff_only_necessary_condition",
        "uses_random_gauge_draws": False,
        "relative_gap": relative_gap,
        "minimum_relative_gap_for_maximum_bound": floor,
        "roundoff_only_eta": eta,
        "separation_margin": separation,
        "bound_with_roundoff": bound,
        "maximum_bound": GAUGE_CERTIFICATE_MAX_BOUND,
        "passes": relative_gap >= floor and bound is not None and bound <= GAUGE_CERTIFICATE_MAX_BOUND,
    }


def expected_canonical_rank_selection(
    curve: dict, reference: float, spectrum: list[float]
) -> tuple[int, dict]:
    """Independently recover prospective selection without held-out gauges."""

    ranks = sorted(int(value) for value in curve["discovery"])
    prefilter = {str(rank): expected_roundoff_prefilter(spectrum, rank) for rank in ranks}
    allowed = {rank for rank in ranks if prefilter[str(rank)]["passes"] is True}
    selected = frozen_selected_rank(curve, reference, allowed_ranks=allowed)
    minimum_accuracy = reference - EXPECTED_PROTOCOL_COMMON["selection_accuracy_tolerance"]
    accuracy_eligible = [
        rank for rank in ranks if curve["discovery"][str(rank)] >= minimum_accuracy
    ]
    joint_nontrivial = [
        rank for rank in accuracy_eligible if rank < len(spectrum) and rank in allowed
    ]
    outcome = (
        "nontrivial_rank_selected"
        if selected < len(spectrum)
        else "full_rank_fallback_no_nontrivial_accuracy_and_prefilter_eligible_rank"
    )
    nontrivial_compression_selected = selected < len(spectrum)
    return selected, {
        "policy": "analytic-roundoff-only-prefilter-v1",
        "selection_gauge_trial_count": GAUGE_SELECTION_TRIALS,
        "held_out_validation_gauge_seed": GAUGE_SEED,
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


def frozen_selected_rank(
    curve: dict, reference: float, *, allowed_ranks: set[int] | None = None
) -> int:
    tolerance = EXPECTED_PROTOCOL_COMMON["selection_accuracy_tolerance"]
    available = sorted(
        int(rank)
        for rank in curve["discovery"]
        if allowed_ranks is None or int(rank) in allowed_ranks
    )
    if not available:
        raise RuntimeError("no allowed operating rank is present in the curve")
    eligible = [
        rank
        for rank in available
        if curve["discovery"][str(rank)] >= reference - tolerance
    ]
    return min(eligible) if eligible else max(available)


def _finite_number(value, *, label: str, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise RuntimeError(f"{label} must be a finite non-boolean number")
    if nonnegative and value < 0:
        raise RuntimeError(f"{label} must be nonnegative")
    return float(value)


def validate_finite_json_numbers(value, *, path: str = "result") -> None:
    """Reject every NaN or infinity recursively.

    Boolean rejection is handled by the typed validators for each numeric slot.
    """

    if isinstance(value, dict):
        for key, child in value.items():
            validate_finite_json_numbers(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_finite_json_numbers(child, path=f"{path}[{index}]")
    elif type(value) in (int, float):
        if not math.isfinite(float(value)):
            raise RuntimeError(f"{path} contains a non-finite numeric value")
    elif isinstance(value, bool) or value is None or isinstance(value, str):
        return
    else:
        raise RuntimeError(f"{path} contains a non-JSON scalar of type {type(value).__name__}")


def _accuracy(value, *, label: str) -> float:
    number = _finite_number(value, label=label)
    if not 0.0 <= number <= 1.0:
        raise RuntimeError(f"{label} must lie in [0, 1]")
    return number


def _typed_equal(actual, expected) -> bool:
    """JSON equality that does not treat booleans as numeric zero or one."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _typed_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _typed_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _close_number(actual, expected, *, label: str, relative_tolerance: float = 1e-12) -> None:
    """Compare independently reduced float64 quantities without demanding bit identity."""

    actual_number = _finite_number(actual, label=label)
    expected_number = _finite_number(expected, label=f"expected {label}")
    scale = max(abs(actual_number), abs(expected_number), np.finfo(np.float64).tiny)
    if abs(actual_number - expected_number) > relative_tolerance * scale:
        raise RuntimeError(
            f"{label} differs from its independent recomputation: "
            f"{actual_number} != {expected_number}"
        )


CERTIFICATE_FIELD_ORDER = (
    "rank", "gap", "covariance_residual_operator",
    "base_eigensolver_residual_operator", "candidate_eigensolver_residual_operator",
    "base_eigenvector_orthogonality", "candidate_eigenvector_orthogonality",
    "transport_orthogonality", "roundoff_allowance", "eta", "separation_margin",
    "numerically_certifiable", "normalized_projector_distance", "davis_kahan_bound",
    "bound_with_roundoff", "passes_bound", "base_supported_overlap_defect",
    "candidate_supported_overlap_defect", "polar_replacement_supported_defect",
    "maximum_supported_overlap_defect", "required_for_certificate",
)
CERTIFICATE_FIELDS = set(CERTIFICATE_FIELD_ORDER)


def recompute_gauge_audit(gauge: dict, trials: int) -> dict:
    """Fail closed over every serialized v3 gauge record and reduction."""

    base_spectra = gauge.get("base_spectra_by_bond")
    if (
        not isinstance(base_spectra, list)
        or len(base_spectra) != 4
        or any(not isinstance(values, list) or len(values) != 33 for values in base_spectra)
    ):
        raise RuntimeError("gauge base-spectrum inventory differs")
    for values in base_spectra:
        for value in values:
            _finite_number(value, label="gauge base spectrum")
        if any(values[index] < values[index + 1] for index in range(32)):
            raise RuntimeError("gauge base spectrum is not descending")
    expected_resolved = [
        [rank for rank in range(1, 33) if expected_eigengap(values, rank)["resolved"]]
        for values in base_spectra
    ]
    if not _typed_equal(gauge.get("resolved_ranks_by_bond"), expected_resolved):
        raise RuntimeError("gauge resolved ranks disagree with base spectra")
    required = gauge.get("required_ranks_by_bond")
    if (
        not isinstance(required, list)
        or len(required) != 4
        or any(
            not isinstance(ranks, list)
            or not ranks
            or any(type(rank) is not int for rank in ranks)
            or ranks != sorted(set(ranks))
            or not set(ranks) <= (set(expected_resolved[bond]) | {33})
            for bond, ranks in enumerate(required)
        )
    ):
        raise RuntimeError("gauge required-rank inventory is invalid or unresolved")

    records = gauge.get("records")
    if not isinstance(records, list) or len(records) != trials:
        raise RuntimeError("gauge record count differs from the frozen protocol")
    trial_fields = {
        "trial", "target_condition", "target_minimum_singular_value",
        "target_maximum_singular_value", "minimum_realized_singular_value",
        "maximum_realized_singular_value", "minimum_realized_condition",
        "maximum_realized_condition", "minimum_realized_nonsymmetry_relative",
        "bond_realizations", "raw_coordinate_replay_relative",
        "canonicalized_function_relative", "canonicalized_coefficient_replay_relative",
        "maximum_sampled_single_bond_logit_replay_relative",
        "max_prefix_overlap_span_defect", "max_transport_orthogonality",
        "max_raw_factor_transport_relative_diagnostic", "max_gram_covariance_relative",
        "max_spectrum_relative", "projector_certificate_pair_count",
        "expected_projector_certificate_pair_count",
    }
    bond_fields = {
        "bond", "minimum_singular_value", "maximum_singular_value", "condition",
        "nonsymmetry_relative", "sampled_single_bond_logit_replay_relative",
        "prefix_overlap_minimum_singular_value", "prefix_overlap_maximum_singular_value",
        "prefix_overlap_span_defect", "transport_orthogonality",
        "raw_factor_transport_relative_diagnostic", "gram_covariance_relative",
        "spectrum_relative", "projector_certificates",
    }
    numeric_trial_fields = trial_fields - {
        "trial", "bond_realizations", "projector_certificate_pair_count",
        "expected_projector_certificate_pair_count",
    }
    numeric_bond_fields = bond_fields - {"bond", "projector_certificates"}
    all_certificates = []
    for trial, item in enumerate(records):
        if set(item) != trial_fields or item.get("trial") != trial:
            raise RuntimeError("gauge trial inventory, field inventory, or order differs")
        condition = 10.0 ** (3.0 * trial / max(1, trials - 1))
        if (
            item["target_condition"] != condition
            or item["target_minimum_singular_value"] != condition ** -0.5
            or item["target_maximum_singular_value"] != condition ** 0.5
        ):
            raise RuntimeError("gauge trial targets differ from the frozen schedule")
        for key in numeric_trial_fields:
            _finite_number(
                item[key],
                label=f"gauge measurement trial {trial} {key}",
                nonnegative=True,
            )
        bonds = item["bond_realizations"]
        if (
            not isinstance(bonds, list)
            or len(bonds) != 4
            or [entry.get("bond") for entry in bonds] != list(range(4))
            or any(set(entry) != bond_fields for entry in bonds)
        ):
            raise RuntimeError("gauge bond-realization inventory is invalid")
        for bond, entry in enumerate(bonds):
            for key in numeric_bond_fields:
                _finite_number(
                    entry[key],
                    label=f"gauge measurement trial {trial} bond {bond} {key}",
                    nonnegative=True,
                )
            if (
                entry["minimum_singular_value"] <= 0
                or entry["maximum_singular_value"] < entry["minimum_singular_value"]
                or entry["condition"] < 1
                or not math.isclose(
                    entry["condition"],
                    entry["maximum_singular_value"] / entry["minimum_singular_value"],
                    rel_tol=1e-12,
                    abs_tol=0,
                )
                or not 0 < entry["nonsymmetry_relative"] <= 2
            ):
                raise RuntimeError("gauge generator realization is invalid")
            certificates = entry["projector_certificates"]
            if (
                not isinstance(certificates, list)
                or [certificate.get("rank") for certificate in certificates]
                != expected_resolved[bond]
                or any(set(certificate) != CERTIFICATE_FIELDS for certificate in certificates)
            ):
                raise RuntimeError("gauge projector-certificate rank inventory differs")
            for certificate in certificates:
                if any(
                    type(certificate[key]) is not bool
                    for key in (
                        "numerically_certifiable", "passes_bound",
                        "required_for_certificate",
                    )
                ):
                    raise RuntimeError("gauge projector certificate has a non-Boolean decision")
                for key in CERTIFICATE_FIELDS - {
                    "rank", "numerically_certifiable", "passes_bound",
                    "required_for_certificate", "davis_kahan_bound",
                    "bound_with_roundoff",
                }:
                    _finite_number(
                        certificate[key],
                        label=f"gauge projector certificate {key}",
                        nonnegative=key != "separation_margin",
                    )
                for key in ("davis_kahan_bound", "bound_with_roundoff"):
                    if certificate[key] is not None:
                        _finite_number(
                            certificate[key], label=f"gauge projector certificate {key}",
                            nonnegative=True,
                        )
                if (
                    certificate["required_for_certificate"]
                    is not (certificate["rank"] in required[bond])
                    or certificate["maximum_supported_overlap_defect"]
                    != max(
                        certificate["base_supported_overlap_defect"],
                        certificate["candidate_supported_overlap_defect"],
                        certificate["polar_replacement_supported_defect"],
                    )
                    or certificate["numerically_certifiable"]
                    is not (certificate["separation_margin"] > 0)
                    or (certificate["davis_kahan_bound"] is None)
                    is certificate["numerically_certifiable"]
                    or (certificate["bound_with_roundoff"] is None)
                    is certificate["numerically_certifiable"]
                ):
                    raise RuntimeError("gauge projector certificate fields disagree")
                all_certificates.append(certificate)
        recomputed_trial = {
            "minimum_realized_singular_value": min(e["minimum_singular_value"] for e in bonds),
            "maximum_realized_singular_value": max(e["maximum_singular_value"] for e in bonds),
            "minimum_realized_condition": min(e["condition"] for e in bonds),
            "maximum_realized_condition": max(e["condition"] for e in bonds),
            "minimum_realized_nonsymmetry_relative": min(e["nonsymmetry_relative"] for e in bonds),
            "maximum_sampled_single_bond_logit_replay_relative": max(
                e["sampled_single_bond_logit_replay_relative"] for e in bonds
            ),
            "max_prefix_overlap_span_defect": max(e["prefix_overlap_span_defect"] for e in bonds),
            "max_transport_orthogonality": max(e["transport_orthogonality"] for e in bonds),
            "max_raw_factor_transport_relative_diagnostic": max(
                e["raw_factor_transport_relative_diagnostic"] for e in bonds
            ),
            "max_gram_covariance_relative": max(e["gram_covariance_relative"] for e in bonds),
            "max_spectrum_relative": max(e["spectrum_relative"] for e in bonds),
            "projector_certificate_pair_count": sum(len(e["projector_certificates"]) for e in bonds),
            "expected_projector_certificate_pair_count": sum(map(len, expected_resolved)),
        }
        if any(item[key] != value for key, value in recomputed_trial.items()):
            raise RuntimeError("gauge trial aggregates disagree with bond realizations")

    required_certificates = [
        item for item in all_certificates
        if item["required_for_certificate"] is True
    ]
    expected_required_count = trials * sum(
        sum(rank < 33 for rank in ranks) for ranks in required
    )
    singular_errors = [
        abs(entry[observed] - item[target]) / item[target]
        for item in records for entry in item["bond_realizations"]
        for observed, target in (
            ("minimum_singular_value", "target_minimum_singular_value"),
            ("maximum_singular_value", "target_maximum_singular_value"),
        )
    ]
    condition_errors = [
        abs(entry["condition"] - item["target_condition"]) / item["target_condition"]
        for item in records for entry in item["bond_realizations"]
    ]
    return {
        "max_raw_coordinate_replay_relative": max(i["raw_coordinate_replay_relative"] for i in records),
        "max_canonicalized_function_relative": max(i["canonicalized_function_relative"] for i in records),
        "max_canonicalized_coefficient_replay_relative": max(
            i["canonicalized_coefficient_replay_relative"] for i in records
        ),
        "max_sampled_single_bond_logit_replay_relative": max(
            i["maximum_sampled_single_bond_logit_replay_relative"] for i in records
        ),
        "max_singular_extrema_relative_error": max(singular_errors),
        "max_condition_relative_error": max(condition_errors),
        "minimum_nonsymmetry_relative": min(i["minimum_realized_nonsymmetry_relative"] for i in records),
        "max_prefix_overlap_span_defect_diagnostic": max(
            e["prefix_overlap_span_defect"] for i in records for e in i["bond_realizations"]
        ),
        "max_transport_orthogonality": max(
            e["transport_orthogonality"] for i in records for e in i["bond_realizations"]
        ),
        "max_raw_factor_transport_relative_diagnostic": max(
            e["raw_factor_transport_relative_diagnostic"]
            for i in records for e in i["bond_realizations"]
        ),
        "max_gram_covariance_relative": max(
            e["gram_covariance_relative"] for i in records for e in i["bond_realizations"]
        ),
        "max_spectrum_relative": max(
            e["spectrum_relative"] for i in records for e in i["bond_realizations"]
        ),
        "projector_certificate_count": len(all_certificates),
        "expected_required_projector_certificate_count": expected_required_count,
        "required_projector_certificate_count": len(required_certificates),
        "projector_certificate_coverage_complete": len(all_certificates)
        == trials * sum(map(len, expected_resolved)),
        "maximum_required_projector_overlap_support_defect": max(
            (i["maximum_supported_overlap_defect"] for i in required_certificates),
            default=None,
        ),
        "all_required_projectors_numerically_certifiable": (
            len(required_certificates) == expected_required_count
            and expected_required_count > 0
            and all(i["numerically_certifiable"] is True for i in required_certificates)
        ),
        "maximum_required_projector_bound_with_roundoff": (
            max(i["bound_with_roundoff"] for i in required_certificates)
            if required_certificates
            and all(i["bound_with_roundoff"] is not None for i in required_certificates)
            else None
        ),
        "all_required_projectors_pass_bound": (
            len(required_certificates) == expected_required_count
            and expected_required_count > 0
            and all(i["passes_bound"] is True for i in required_certificates)
        ),
    }


def _close_matrix_measurement(
    actual,
    expected,
    *,
    label: str,
    absolute_tolerance: float = 1e-12,
) -> None:
    actual_number = _finite_number(actual, label=label)
    expected_number = _finite_number(expected, label=f"independently recomputed {label}")
    if not math.isclose(
        actual_number, expected_number, rel_tol=1e-9, abs_tol=absolute_tolerance
    ):
        raise RuntimeError(f"{label} disagrees with the gauge matrix artifact")


def _tensor_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return a scale-aware whole-tensor error, including near-zero entries."""

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return math.inf
    denominator = torch.maximum(
        torch.linalg.vector_norm(actual),
        torch.linalg.vector_norm(expected),
    ).clamp_min(torch.finfo(actual.dtype).tiny)
    return float((torch.linalg.vector_norm(actual - expected) / denominator).item())


def _require_tensor_relative_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    tolerance: float,
    label: str,
) -> None:
    error = _tensor_relative_error(actual, expected)
    if not math.isfinite(error) or error > tolerance:
        raise RuntimeError(
            f"{label} differs from its independent reconstruction: "
            f"relative error {error} exceeds {tolerance}"
        )


def _projector_bound_conditioning_tolerance(
    reported: dict,
    independent: dict,
) -> float:
    """Propagate primitive CPU/GPU discrepancies through the rational bound.

    This is used only to authenticate non-required diagnostic records. Required
    certificates remain on the strict scalar-comparison and threshold path.
    """

    separations = (
        float(reported["separation_margin"]),
        float(independent["separation_margin"]),
    )
    if min(separations) <= 0:
        return NONREQUIRED_DIAGNOSTIC_PRIMITIVE_ABSOLUTE_TOLERANCE
    eta_delta = abs(float(reported["eta"]) - float(independent["eta"]))
    gap_delta = abs(float(reported["gap"]) - float(independent["gap"]))
    maximum_gap = max(abs(float(reported["gap"])), abs(float(independent["gap"])))
    maximum_eta = max(abs(float(reported["eta"])), abs(float(independent["eta"])))
    minimum_separation = min(separations)
    denominator = minimum_separation * minimum_separation
    if denominator == 0 or not all(
        math.isfinite(value)
        for value in (
            eta_delta, gap_delta, maximum_gap, maximum_eta, denominator
        )
    ):
        raise RuntimeError(
            "non-required projector-bound conditioning tolerance is non-finite"
        )
    propagated = math.sqrt(2.0) * (
        maximum_gap * eta_delta + maximum_eta * gap_delta
    ) / denominator
    arithmetic_slack = 128.0 * float(np.finfo(np.float64).eps) * max(
        1.0,
        abs(float(reported["davis_kahan_bound"])),
        abs(float(independent["davis_kahan_bound"])),
    )
    tolerance = propagated + arithmetic_slack
    if not math.isfinite(tolerance):
        raise RuntimeError(
            "non-required projector-bound conditioning tolerance is non-finite"
        )
    return tolerance


def _nonrequired_projector_realization_tolerance(
    reported: dict,
    independent: dict,
) -> float:
    """Bound copy-to-copy drift of a non-load-bearing realized distance.

    ``normalized_projector_distance`` is an observed distance between two
    numerically constructed invariant subspaces.  It is not used to select a
    rank or establish an X-Net certificate.  For a non-required diagnostic,
    each copy is instead authenticated by reconstructing its matrices and by
    checking the same Davis--Kahan inequality internally.  Consequently the
    largest possible legitimate disagreement between two nonnegative realized
    distances is their largest certified upper envelope.  Required ranks never
    take this path: their stored scalar values remain strictly compared or are
    independently gate-replicated.
    """

    envelopes = (
        _finite_number(
            reported["bound_with_roundoff"],
            label="reported non-required projector-distance envelope",
            nonnegative=True,
        ),
        _finite_number(
            independent["bound_with_roundoff"],
            label="independent non-required projector-distance envelope",
            nonnegative=True,
        ),
    )
    arithmetic_slack = 128.0 * float(np.finfo(np.float64).eps) * max(
        1.0, *envelopes
    )
    tolerance = max(envelopes) + arithmetic_slack
    if not math.isfinite(tolerance):
        raise RuntimeError(
            "non-required projector-realization tolerance is non-finite"
        )
    return tolerance


def _validate_projector_certificate_self_consistency(
    certificate: dict,
    *,
    arithmetic_slack: float,
    label: str,
) -> None:
    """Authenticate derived diagnostic fields against their own primitives."""

    gap = _finite_number(certificate["gap"], label=f"{label} gap")
    eta = _finite_number(certificate["eta"], label=f"{label} eta", nonnegative=True)
    separation = _finite_number(
        certificate["separation_margin"], label=f"{label} separation margin"
    )
    scalar_slack = 128.0 * float(np.finfo(np.float64).eps) * max(
        1.0, abs(gap), abs(eta), abs(separation)
    )
    roundoff_allowance = _finite_number(
        certificate["roundoff_allowance"],
        label=f"{label} roundoff allowance",
        nonnegative=True,
    )
    if arithmetic_slack <= 0:
        raise RuntimeError(f"{label} arithmetic slack is invalid")
    scale = roundoff_allowance / arithmetic_slack
    expected_eta = (
        certificate["covariance_residual_operator"]
        + certificate["base_eigensolver_residual_operator"]
        + certificate["candidate_eigensolver_residual_operator"]
        + scale
        * (
            certificate["transport_orthogonality"]
            + certificate["base_eigenvector_orthogonality"]
            + certificate["candidate_eigenvector_orthogonality"]
        )
        + roundoff_allowance
    )
    if not math.isclose(eta, expected_eta, rel_tol=1e-12, abs_tol=scalar_slack):
        raise RuntimeError(f"{label} eta is internally inconsistent")
    if not math.isclose(separation, gap - 2.0 * eta, rel_tol=1e-12, abs_tol=scalar_slack):
        raise RuntimeError(f"{label} separation margin is internally inconsistent")
    expected_support_maximum = max(
        certificate["base_supported_overlap_defect"],
        certificate["candidate_supported_overlap_defect"],
        certificate["polar_replacement_supported_defect"],
    )
    if certificate["maximum_supported_overlap_defect"] != expected_support_maximum:
        raise RuntimeError(f"{label} support-defect maximum is internally inconsistent")
    certifiable = gap > 0 and separation > 0
    if certificate["numerically_certifiable"] is not certifiable:
        raise RuntimeError(f"{label} numerical-certifiability decision is inconsistent")
    if certifiable:
        expected_bound = math.sqrt(2.0) * eta / separation
        expected_with_roundoff = expected_bound + arithmetic_slack
        if not math.isclose(
            certificate["davis_kahan_bound"],
            expected_bound,
            rel_tol=1e-12,
            abs_tol=scalar_slack,
        ) or not math.isclose(
            certificate["bound_with_roundoff"],
            expected_with_roundoff,
            rel_tol=1e-12,
            abs_tol=scalar_slack,
        ):
            raise RuntimeError(f"{label} Davis-Kahan bound is internally inconsistent")
        expected_pass = (
            certificate["normalized_projector_distance"]
            <= certificate["bound_with_roundoff"]
        )
        if certificate["passes_bound"] is not expected_pass:
            raise RuntimeError(f"{label} projector-bound decision is internally inconsistent")
    elif (
        certificate["davis_kahan_bound"] is not None
        or certificate["bound_with_roundoff"] is not None
        or certificate["passes_bound"] is not False
    ):
        raise RuntimeError(f"{label} uncertifiable bound fields are inconsistent")


def _compare_projector_certificates(
    reported: dict,
    independent: dict,
    *,
    arithmetic_slack: float,
    label: str,
    required_scalar_policy: str,
) -> None:
    """Compare a serialized certificate to one recomputed from authenticated matrices."""

    if required_scalar_policy not in {
        "strict_stored_npz",
        "independent_gate_replication",
    }:
        raise ValueError("required scalar policy is invalid")
    if set(reported) != CERTIFICATE_FIELDS or set(independent) != CERTIFICATE_FIELDS:
        raise RuntimeError(f"{label} certificate field inventory differs")
    _validate_projector_certificate_self_consistency(
        reported, arithmetic_slack=arithmetic_slack, label=f"{label} reported"
    )
    _validate_projector_certificate_self_consistency(
        independent, arithmetic_slack=arithmetic_slack, label=f"{label} independent"
    )
    required = reported["required_for_certificate"]
    if required is not independent["required_for_certificate"] or type(required) is not bool:
        raise RuntimeError(f"{label} required-certificate classification differs")
    if required and required_scalar_policy == "independent_gate_replication":
        if reported["rank"] != independent["rank"] or type(reported["rank"]) is not int:
            raise RuntimeError(f"{label} required certificate rank differs")
        return
    conditioned_bound_tolerance = None
    if not required and reported["davis_kahan_bound"] is not None and independent[
        "davis_kahan_bound"
    ] is not None:
        conditioned_bound_tolerance = _projector_bound_conditioning_tolerance(
            reported, independent
        )
    for key in CERTIFICATE_FIELD_ORDER:
        expected = independent[key]
        actual = reported[key]
        if type(expected) is bool or expected is None or type(expected) is int:
            if actual != expected or type(actual) is not type(expected):
                raise RuntimeError(f"{label} {key} disagrees with reconstruction")
        elif required:
            _close_matrix_measurement(
                actual, expected, label=f"{label} required {key}", absolute_tolerance=1e-12
            )
        elif key in {"davis_kahan_bound", "bound_with_roundoff"}:
            if abs(float(actual) - float(expected)) > conditioned_bound_tolerance:
                raise RuntimeError(
                    f"{label} non-required {key} exceeds its gap-conditioned "
                    "authentication tolerance"
                )
        elif key == "normalized_projector_distance":
            if abs(float(actual) - float(expected)) > _nonrequired_projector_realization_tolerance(
                reported, independent
            ):
                raise RuntimeError(
                    f"{label} non-required {key} exceeds its independently "
                    "certified realization envelope"
                )
        else:
            _close_matrix_measurement(
                actual,
                expected,
                label=f"{label} non-required diagnostic {key}",
                absolute_tolerance=(
                    NONREQUIRED_DIAGNOSTIC_PRIMITIVE_ABSOLUTE_TOLERANCE
                ),
            )


def _required_certificate_decision(
    certificate: dict,
    *,
    maximum_bound: float,
) -> dict:
    """Return every load-bearing decision for one required projector copy."""

    bound = certificate["bound_with_roundoff"]
    support = certificate["maximum_supported_overlap_defect"]
    decision = {
        "numerically_certifiable": certificate["numerically_certifiable"],
        "passes_davis_kahan_inequality": certificate["passes_bound"],
        "bound_with_roundoff": bound,
        "bound_with_roundoff_le_maximum": (
            bound is not None and bound <= maximum_bound
        ),
        "maximum_supported_overlap_defect": support,
        "maximum_supported_overlap_defect_le_1e-8": support <= 1e-8,
    }
    decision["all_frozen_gates_pass"] = all(
        decision[key] is True
        for key in (
            "numerically_certifiable",
            "passes_davis_kahan_inequality",
            "bound_with_roundoff_le_maximum",
            "maximum_supported_overlap_defect_le_1e-8",
        )
    )
    return decision


def _validate_required_certificate_copies(
    decisions: dict,
    *,
    require_all_gates: bool,
) -> None:
    expected = {
        "producer",
        "stored_npz_cpu",
        "independent_checkpoint_raw_gauge_cpu",
    }
    if set(decisions) != expected or type(require_all_gates) is not bool:
        raise RuntimeError("required-projector copy inventory is invalid")
    if require_all_gates and any(
        item.get("all_frozen_gates_pass") is not True
        for item in decisions.values()
    ):
        raise RuntimeError(
            "one required-projector authentication copy failed a frozen gate"
        )


def validate_gauge_matrix_artifact(
    record: dict,
    *,
    project_root: Path,
    expected_path: Path,
    expected_raw_network: HomogeneousChiTN,
    expected_replay_sample: torch.Tensor,
    require_all_matrix_gates: bool = True,
    include_authentication_details: bool = False,
) -> Path | tuple[Path, dict]:
    """Load the v3 NPZ and independently recompute every matrix-derived claim.

    Gate-failed records still require authenticated agreement with every
    recomputed matrix measurement under the same frozen comparison policy.
    They differ only in allowing those authenticated measurements to remain on
    the failing side of a preregistered scientific threshold.
    """

    if type(require_all_matrix_gates) is not bool or type(include_authentication_details) is not bool:
        raise TypeError("matrix-gate and authentication-detail controls must be booleans")

    gauge = record["gauge_audit"]
    trials = record["protocol"]["gauge_trials"]
    manifest = gauge.get("matrix_artifact")
    if not isinstance(manifest, dict) or set(manifest) != {
        "path", "bytes", "sha256", "allow_pickle", "keys",
    }:
        raise RuntimeError("gauge matrix artifact manifest inventory differs")
    path = require_confined_path(
        project_root, Path(manifest.get("path", "")), label="gauge matrix artifact", kind="file"
    )
    if path != expected_path or manifest["allow_pickle"] is not False:
        raise RuntimeError("gauge matrix artifact path or pickle policy differs")
    if manifest["bytes"] != path.stat().st_size or manifest["sha256"] != digest(path):
        raise RuntimeError("gauge matrix artifact byte manifest differs")
    network_keys = {
        "replay_sample_flat",
        "raw_network_embedding", "raw_network_head",
        "base_canonical_embedding", "base_canonical_head",
        *(f"raw_network_core_{index}" for index in range(3)),
        *(f"base_canonical_core_{index}" for index in range(3)),
    }
    expected_keys = network_keys | {f"base_gram_bond_{bond}" for bond in range(4)} | {
        f"trial_{trial}_bond_{bond}_{suffix}"
        for trial in range(trials)
        for bond in range(4)
        for suffix in ("raw_gauge", "overlap", "transport", "candidate_gram")
    }
    expected_shapes = {
        "replay_sample_flat": [16, 3072],
        "raw_network_embedding": [33, 3073],
        "raw_network_head": [10, 33],
        "base_canonical_embedding": [33, 3073],
        "base_canonical_head": [10, 33],
        **{f"raw_network_core_{index}": [33, 33, 33] for index in range(3)},
        **{f"base_canonical_core_{index}": [33, 33, 33] for index in range(3)},
        **{f"base_gram_bond_{bond}": [33, 33] for bond in range(4)},
        **{
            f"trial_{trial}_bond_{bond}_{suffix}": [33, 33]
            for trial in range(trials)
            for bond in range(4)
            for suffix in ("raw_gauge", "overlap", "transport", "candidate_gram")
        },
    }
    expected_metadata = {
        key: {"dtype": "float64", "shape": expected_shapes[key]}
        for key in sorted(expected_keys)
    }
    if manifest["keys"] != expected_metadata:
        raise RuntimeError("gauge matrix artifact key metadata differs")

    with np.load(path, allow_pickle=False) as bundle:
        if set(bundle.files) != expected_keys:
            raise RuntimeError("gauge matrix artifact key inventory differs")
        matrices = {}
        for key in sorted(expected_keys):
            value = bundle[key]
            if (
                value.dtype != np.float64
                or list(value.shape) != expected_shapes[key]
                or not np.isfinite(value).all()
            ):
                raise RuntimeError(f"gauge matrix artifact entry {key} is invalid")
            matrices[key] = torch.from_numpy(value.copy())

    raw_network = HomogeneousChiTN(
        matrices["raw_network_embedding"],
        tuple(matrices[f"raw_network_core_{index}"] for index in range(3)),
        matrices["raw_network_head"],
    )
    base_network = HomogeneousChiTN(
        matrices["base_canonical_embedding"],
        tuple(matrices[f"base_canonical_core_{index}"] for index in range(3)),
        matrices["base_canonical_head"],
    )
    sample = matrices["replay_sample_flat"]
    expected_raw_parts = (
        expected_raw_network.embedding,
        *expected_raw_network.cores,
        expected_raw_network.head,
    )
    stored_raw_parts = (
        raw_network.embedding,
        *raw_network.cores,
        raw_network.head,
    )
    for index, (stored, expected) in enumerate(zip(stored_raw_parts, expected_raw_parts)):
        if not torch.allclose(stored, expected, rtol=1e-10, atol=1e-12):
            raise RuntimeError(
                f"gauge NPZ raw-network tensor {index} is not derived from the checkpoint"
            )
    if expected_replay_sample.shape != sample.shape or not torch.equal(
        sample, expected_replay_sample
    ):
        raise RuntimeError(
            "gauge NPZ replay sample is not the regenerated frozen discovery sample"
        )
    raw_base_logits = homogeneous_forward(raw_network, sample)
    canonical_base_logits = homogeneous_forward(base_network, sample)
    base_function_replay = float(
        (
            torch.linalg.vector_norm(raw_base_logits - canonical_base_logits)
            / torch.linalg.vector_norm(raw_base_logits).clamp_min(1e-30)
        ).item()
    )
    if base_function_replay > 1e-8:
        raise RuntimeError("raw/base-canonical NPZ functional replay exceeds 1e-8")
    base_canonicalized_raw = canonicalize_homogeneous(raw_network)
    base_reconstruction_error = math.sqrt(
        max(
            0.0,
            float(
                coefficient_error_squared(
                    base_network, base_canonicalized_raw.network
                ).item()
            ),
        )
        / float(coefficient_inner_product(base_network, base_network).item())
    )
    if base_reconstruction_error > 1e-8:
        raise RuntimeError("raw/base-canonical NPZ coefficient replay exceeds 1e-8")
    base_coefficient_norm_squared = float(
        coefficient_inner_product(base_network, base_network).item()
    )
    if not math.isfinite(base_coefficient_norm_squared) or base_coefficient_norm_squared <= 0:
        raise RuntimeError("gauge NPZ base coefficient norm is invalid")

    base_grams = [matrices[f"base_gram_bond_{bond}"] for bond in range(4)]
    independently_recomputed_base_grams = canonical_environments(base_network)
    for bond, gram in enumerate(base_grams):
        _require_tensor_relative_close(
            gram,
            independently_recomputed_base_grams[bond],
            tolerance=INDEPENDENT_MATRIX_RECONSTRUCTION_RELATIVE_TOLERANCE,
            label=f"stored base Gram bond {bond}",
        )
        values, _ = sorted_eigensystem(gram)
        reported = gauge["base_spectra_by_bond"][bond]
        for rank, (actual, expected) in enumerate(zip(reported, values.tolist())):
            _close_matrix_measurement(
                actual, expected, label=f"base spectrum bond {bond} coordinate {rank}"
            )

    records = gauge["records"]
    independent_required_decisions = {}
    required_decision_inventory = []
    conditioning_diagnostic_records = []
    for trial, item in enumerate(records):
        gauged_network = raw_network
        raw_gauges = [
            matrices[f"trial_{trial}_bond_{bond}_raw_gauge"] for bond in range(4)
        ]
        expected_raw_gauges = regenerate_trial_gauges(
            trial, raw_network.bond_dims
        )
        for bond, (stored_gauge, expected_gauge) in enumerate(
            zip(raw_gauges, expected_raw_gauges)
        ):
            _require_tensor_relative_close(
                stored_gauge,
                expected_gauge,
                tolerance=RAW_GAUGE_STREAM_RELATIVE_TOLERANCE,
                label=f"trial {trial} bond {bond} raw gauge from frozen stream",
            )
        for bond, raw_gauge in enumerate(raw_gauges):
            gauged_network = apply_bond_gauge(gauged_network, bond, raw_gauge)
        gauged_canonical = canonicalize_homogeneous(gauged_network)
        independent_base_network = base_canonicalized_raw.network
        independent_base_grams = canonical_environments(independent_base_network)
        independent_candidate_grams = canonical_environments(gauged_canonical.network)
        independent_overlaps = canonical_prefix_overlaps(
            gauged_canonical.network, independent_base_network
        )
        for bond, entry in enumerate(item["bond_realizations"]):
            prefix = f"trial_{trial}_bond_{bond}"
            raw_gauge = raw_gauges[bond]
            overlap = matrices[f"{prefix}_overlap"]
            transport = matrices[f"{prefix}_transport"]
            candidate_gram = matrices[f"{prefix}_candidate_gram"]
            base_gram = base_grams[bond]
            identity = torch.eye(33, dtype=torch.float64)

            gauge_singular = torch.linalg.svdvals(raw_gauge)
            gauge_minimum = float(gauge_singular[-1].item())
            gauge_maximum = float(gauge_singular[0].item())
            gauge_condition = gauge_maximum / gauge_minimum
            gauge_nonsymmetry = float(
                (
                    torch.linalg.matrix_norm(raw_gauge - raw_gauge.T)
                    / torch.linalg.matrix_norm(raw_gauge).clamp_min(1e-30)
                ).item()
            )
            for key, expected in (
                ("minimum_singular_value", gauge_minimum),
                ("maximum_singular_value", gauge_maximum),
                ("condition", gauge_condition),
                ("nonsymmetry_relative", gauge_nonsymmetry),
            ):
                _close_matrix_measurement(
                    entry[key], expected, label=f"trial {trial} bond {bond} {key}"
                )
            if require_all_matrix_gates and (
                abs(gauge_minimum - item["target_minimum_singular_value"])
                / item["target_minimum_singular_value"]
                > 1e-12
                or abs(gauge_maximum - item["target_maximum_singular_value"])
                / item["target_maximum_singular_value"]
                > 1e-12
                or abs(gauge_condition - item["target_condition"])
                / item["target_condition"]
                > 1e-12
                or gauge_nonsymmetry <= 1e-6
            ):
                raise RuntimeError("NPZ-recomputed raw gauge realization gate failed")

            single_bond_logits = homogeneous_forward(
                apply_bond_gauge(raw_network, bond, raw_gauge), sample
            )
            single_bond_replay = float(
                (
                    torch.linalg.vector_norm(single_bond_logits - raw_base_logits)
                    / torch.linalg.vector_norm(raw_base_logits).clamp_min(1e-30)
                ).item()
            )
            _close_matrix_measurement(
                entry["sampled_single_bond_logit_replay_relative"],
                single_bond_replay,
                label=f"trial {trial} bond {bond} single-bond functional replay",
                absolute_tolerance=1e-8,
            )
            if require_all_matrix_gates and single_bond_replay > 1e-8:
                raise RuntimeError("NPZ-recomputed single-bond functional replay exceeds 1e-8")
            overlap_singular = torch.linalg.svdvals(overlap)
            overlap_span_defect = float(
                torch.maximum(
                    torch.linalg.matrix_norm(overlap @ overlap.T - identity, ord=2),
                    torch.linalg.matrix_norm(overlap.T @ overlap - identity, ord=2),
                ).item()
            )
            left, _, right = torch.linalg.svd(overlap, full_matrices=False)
            recomputed_transport = left @ right
            _require_tensor_relative_close(
                transport,
                recomputed_transport,
                tolerance=INDEPENDENT_MATRIX_RECONSTRUCTION_RELATIVE_TOLERANCE,
                label=f"trial {trial} bond {bond} stored overlap polar factor",
            )
            transport_orthogonality = float(
                torch.linalg.matrix_norm(transport @ transport.T - identity, ord=2).item()
            )
            predicted_gram = transport @ base_gram @ transport.T
            gram_covariance = float(
                (
                    torch.linalg.matrix_norm(candidate_gram - predicted_gram, ord=2)
                    / torch.linalg.matrix_norm(candidate_gram, ord=2).clamp_min(1e-30)
                ).item()
            )
            base_values, base_vectors = sorted_eigensystem(base_gram)
            candidate_values, candidate_vectors = sorted_eigensystem(candidate_gram)
            spectrum_relative = float(
                (
                    torch.linalg.vector_norm(base_values - candidate_values)
                    / torch.linalg.vector_norm(base_values).clamp_min(1e-30)
                ).item()
            )
            for key, expected in (
                ("prefix_overlap_minimum_singular_value", float(overlap_singular[-1].item())),
                ("prefix_overlap_maximum_singular_value", float(overlap_singular[0].item())),
                ("prefix_overlap_span_defect", overlap_span_defect),
                ("transport_orthogonality", transport_orthogonality),
                ("gram_covariance_relative", gram_covariance),
                ("spectrum_relative", spectrum_relative),
            ):
                _close_matrix_measurement(
                    entry[key], expected, label=f"trial {trial} bond {bond} {key}"
                )
            if require_all_matrix_gates and (
                transport_orthogonality > 1e-10
                or gram_covariance > 1e-8
                or spectrum_relative > 1e-8
            ):
                raise RuntimeError("NPZ-recomputed stored structural gauge gate failed")

            independent_overlap = independent_overlaps[bond]
            independent_transport = polar_orthogonal_transport(independent_overlap)
            independent_identity = torch.eye(33, dtype=torch.float64)
            independent_base_gram = independent_base_grams[bond]
            independent_candidate_gram = independent_candidate_grams[bond]
            for stored, independent, label in (
                (overlap, independent_overlap, "prefix overlap"),
                (transport, independent_transport, "polar transport"),
                (candidate_gram, independent_candidate_gram, "candidate Gram"),
            ):
                _require_tensor_relative_close(
                    stored,
                    independent,
                    tolerance=INDEPENDENT_MATRIX_RECONSTRUCTION_RELATIVE_TOLERANCE,
                    label=(
                        f"stored trial {trial} bond {bond} {label} derived "
                        "from the independently reconstructed gauged network"
                    ),
                )
            independent_overlap_singular = torch.linalg.svdvals(independent_overlap)
            independent_overlap_span_defect = float(
                torch.maximum(
                    torch.linalg.matrix_norm(
                        independent_overlap @ independent_overlap.T
                        - independent_identity,
                        ord=2,
                    ),
                    torch.linalg.matrix_norm(
                        independent_overlap.T @ independent_overlap
                        - independent_identity,
                        ord=2,
                    ),
                ).item()
            )
            independent_covariance = float(
                (
                    torch.linalg.matrix_norm(
                        independent_candidate_gram
                        - independent_transport
                        @ independent_base_gram
                        @ independent_transport.T,
                        ord=2,
                    )
                    / torch.linalg.matrix_norm(
                        independent_candidate_gram, ord=2
                    ).clamp_min(1e-30)
                ).item()
            )
            independent_orthogonality = float(
                torch.linalg.matrix_norm(
                    independent_transport @ independent_transport.T
                    - independent_identity,
                    ord=2,
                ).item()
            )
            independent_base_values, independent_base_vectors = sorted_eigensystem(
                independent_base_gram
            )
            independent_candidate_values, independent_candidate_vectors = sorted_eigensystem(
                independent_candidate_gram
            )
            independent_spectrum = float(
                (
                    torch.linalg.vector_norm(
                        independent_base_values - independent_candidate_values
                    )
                    / torch.linalg.vector_norm(independent_base_values).clamp_min(1e-30)
                ).item()
            )
            independent_target_factor = (
                raw_gauge @ base_canonicalized_raw.raw_from_canonical[bond]
            )
            independent_candidate_factor = gauged_canonical.raw_from_canonical[bond]
            independent_left, _, independent_right = torch.linalg.svd(
                independent_candidate_factor.T @ independent_target_factor,
                full_matrices=True,
            )
            independent_raw_factor_transport = independent_left @ independent_right
            independent_raw_factor_residual = float(
                (
                    torch.linalg.matrix_norm(
                        independent_candidate_factor @ independent_raw_factor_transport
                        - independent_target_factor
                    )
                    / torch.linalg.matrix_norm(independent_target_factor).clamp_min(1e-30)
                ).item()
            )
            for key, expected in (
                (
                    "prefix_overlap_minimum_singular_value",
                    float(independent_overlap_singular[-1].item()),
                ),
                (
                    "prefix_overlap_maximum_singular_value",
                    float(independent_overlap_singular[0].item()),
                ),
                ("prefix_overlap_span_defect", independent_overlap_span_defect),
                ("transport_orthogonality", independent_orthogonality),
                ("raw_factor_transport_relative_diagnostic", independent_raw_factor_residual),
                ("gram_covariance_relative", independent_covariance),
                ("spectrum_relative", independent_spectrum),
            ):
                _close_matrix_measurement(
                    entry[key],
                    expected,
                    label=(
                        f"trial {trial} bond {bond} independently reconstructed {key}"
                    ),
                    absolute_tolerance=1e-8,
                )
            if require_all_matrix_gates and (
                independent_covariance > 1e-8
                or independent_orthogonality > 1e-10
                or independent_spectrum > 1e-8
            ):
                raise RuntimeError("independently reconstructed gauge matrix gate failed")
            for rank in gauge["required_ranks_by_bond"][bond]:
                if rank == 33:
                    continue
                independent_certificate = projector_perturbation_certificate(
                    independent_base_gram,
                    independent_candidate_gram,
                    independent_transport,
                    rank=rank,
                    roundoff_multiplier=record["protocol"][
                        "gauge_projector_roundoff_multiplier"
                    ],
                )
                independent_base_projector = (
                    independent_base_vectors[:, :rank]
                    @ independent_base_vectors[:, :rank].T
                )
                independent_candidate_projector = (
                    independent_candidate_vectors[:, :rank]
                    @ independent_candidate_vectors[:, :rank].T
                )
                independent_support_defect = max(
                    float(
                        torch.linalg.matrix_norm(
                            independent_base_projector
                            @ (independent_identity - independent_overlap.T @ independent_overlap)
                            @ independent_base_projector,
                            ord=2,
                        ).item()
                    ),
                    float(
                        torch.linalg.matrix_norm(
                            independent_candidate_projector
                            @ (independent_identity - independent_overlap @ independent_overlap.T)
                            @ independent_candidate_projector,
                            ord=2,
                        ).item()
                    ),
                    float(
                        torch.linalg.matrix_norm(
                            (independent_transport - independent_overlap)
                            @ independent_base_projector,
                            ord=2,
                        ).item()
                    ),
                )
                if require_all_matrix_gates and (
                    independent_support_defect > 1e-8
                    or independent_certificate.numerically_certifiable is not True
                    or independent_certificate.passes_bound is not True
                    or independent_certificate.bound_with_roundoff is None
                    or independent_certificate.bound_with_roundoff
                    > record["protocol"]["gauge_required_projector_maximum_bound"]
                ):
                    raise RuntimeError(
                        "independently reconstructed required-projector gate failed"
                    )

            for reported in entry["projector_certificates"]:
                rank = reported["rank"]
                independent_computed = vars(
                    projector_perturbation_certificate(
                        independent_base_gram,
                        independent_candidate_gram,
                        independent_transport,
                        rank=rank,
                        roundoff_multiplier=record["protocol"][
                            "gauge_projector_roundoff_multiplier"
                        ],
                    )
                )
                independent_base_projector = (
                    independent_base_vectors[:, :rank]
                    @ independent_base_vectors[:, :rank].T
                )
                independent_candidate_projector = (
                    independent_candidate_vectors[:, :rank]
                    @ independent_candidate_vectors[:, :rank].T
                )
                independent_computed.update(
                    {
                        "base_supported_overlap_defect": float(
                            torch.linalg.matrix_norm(
                                independent_base_projector
                                @ (
                                    independent_identity
                                    - independent_overlap.T @ independent_overlap
                                )
                                @ independent_base_projector,
                                ord=2,
                            ).item()
                        ),
                        "candidate_supported_overlap_defect": float(
                            torch.linalg.matrix_norm(
                                independent_candidate_projector
                                @ (
                                    independent_identity
                                    - independent_overlap @ independent_overlap.T
                                )
                                @ independent_candidate_projector,
                                ord=2,
                            ).item()
                        ),
                        "polar_replacement_supported_defect": float(
                            torch.linalg.matrix_norm(
                                (independent_transport - independent_overlap)
                                @ independent_base_projector,
                                ord=2,
                            ).item()
                        ),
                    }
                )
                independent_computed["maximum_supported_overlap_defect"] = max(
                    independent_computed["base_supported_overlap_defect"],
                    independent_computed["candidate_supported_overlap_defect"],
                    independent_computed["polar_replacement_supported_defect"],
                )
                independent_computed["required_for_certificate"] = (
                    rank in gauge["required_ranks_by_bond"][bond]
                )
                if set(independent_computed) != CERTIFICATE_FIELDS:
                    raise RuntimeError(
                        "independent projector-certificate field inventory differs"
                    )
                _compare_projector_certificates(
                    reported,
                    independent_computed,
                    arithmetic_slack=(
                        record["protocol"]["gauge_projector_roundoff_multiplier"]
                        * float(np.finfo(np.float64).eps)
                        * independent_base_gram.shape[0]
                    ),
                    label=(
                        f"trial {trial} bond {bond} projector {rank} "
                        "independently reconstructed"
                    ),
                    required_scalar_policy="independent_gate_replication",
                )
                if reported["required_for_certificate"] is True:
                    independent_required_decisions[(trial, bond, rank)] = (
                        _required_certificate_decision(
                            independent_computed,
                            maximum_bound=record["protocol"][
                                "gauge_required_projector_maximum_bound"
                            ],
                        )
                    )

            for reported in entry["projector_certificates"]:
                rank = reported["rank"]
                computed = vars(
                    projector_perturbation_certificate(
                        base_gram,
                        candidate_gram,
                        transport,
                        rank=rank,
                        roundoff_multiplier=record["protocol"][
                            "gauge_projector_roundoff_multiplier"
                        ],
                    )
                )
                base_projector = base_vectors[:, :rank] @ base_vectors[:, :rank].T
                candidate_projector = (
                    candidate_vectors[:, :rank] @ candidate_vectors[:, :rank].T
                )
                computed.update(
                    {
                        "base_supported_overlap_defect": float(
                            torch.linalg.matrix_norm(
                                base_projector @ (identity - overlap.T @ overlap) @ base_projector,
                                ord=2,
                            ).item()
                        ),
                        "candidate_supported_overlap_defect": float(
                            torch.linalg.matrix_norm(
                                candidate_projector
                                @ (identity - overlap @ overlap.T)
                                @ candidate_projector,
                                ord=2,
                            ).item()
                        ),
                        "polar_replacement_supported_defect": float(
                            torch.linalg.matrix_norm(
                                (transport - overlap) @ base_projector, ord=2
                            ).item()
                        ),
                    }
                )
                computed["maximum_supported_overlap_defect"] = max(
                    computed["base_supported_overlap_defect"],
                    computed["candidate_supported_overlap_defect"],
                    computed["polar_replacement_supported_defect"],
                )
                computed["required_for_certificate"] = (
                    rank in gauge["required_ranks_by_bond"][bond]
                )
                if set(computed) != CERTIFICATE_FIELDS:
                    raise RuntimeError("internal projector-certificate field inventory differs")
                _compare_projector_certificates(
                    reported,
                    computed,
                    arithmetic_slack=(
                        record["protocol"]["gauge_projector_roundoff_multiplier"]
                        * float(np.finfo(np.float64).eps)
                        * base_gram.shape[0]
                    ),
                    label=f"trial {trial} bond {bond} projector {rank} NPZ",
                    required_scalar_policy="strict_stored_npz",
                )
                if reported["required_for_certificate"] is True:
                    key = (trial, bond, rank)
                    independent_decision = independent_required_decisions.get(key)
                    if independent_decision is None:
                        raise RuntimeError(
                            "independent required-projector decision inventory is incomplete"
                        )
                    maximum_bound = record["protocol"][
                        "gauge_required_projector_maximum_bound"
                    ]
                    decisions = {
                        "producer": _required_certificate_decision(
                            reported, maximum_bound=maximum_bound
                        ),
                        "stored_npz_cpu": _required_certificate_decision(
                            computed, maximum_bound=maximum_bound
                        ),
                        "independent_checkpoint_raw_gauge_cpu": independent_decision,
                    }
                    _validate_required_certificate_copies(
                        decisions,
                        require_all_gates=require_all_matrix_gates,
                    )
                    required_decision_inventory.append(
                        {
                            "trial": trial,
                            "bond": bond,
                            "rank": rank,
                            "copies": decisions,
                        }
                    )

        raw_gauged_logits = homogeneous_forward(gauged_network, sample)
        raw_replay = float(
            (
                torch.linalg.vector_norm(raw_gauged_logits - raw_base_logits)
                / torch.linalg.vector_norm(raw_base_logits).clamp_min(1e-30)
            ).item()
        )
        gauged_canonical_logits = homogeneous_forward(gauged_canonical.network, sample)
        canonical_function_replay = float(
            (
                torch.linalg.vector_norm(gauged_canonical_logits - canonical_base_logits)
                / torch.linalg.vector_norm(canonical_base_logits).clamp_min(1e-30)
            ).item()
        )
        coefficient_replay = math.sqrt(
            max(
                0.0,
                float(coefficient_error_squared(base_network, gauged_canonical.network).item()),
            )
            / base_coefficient_norm_squared
        )
        _finite_number(raw_replay, label=f"trial {trial} independently recomputed raw replay")
        _finite_number(
            canonical_function_replay,
            label=f"trial {trial} independently recomputed canonical function replay",
        )
        producer_raw_replay = _finite_number(
            item["raw_coordinate_replay_relative"],
            label=f"trial {trial} producer raw-coordinate replay diagnostic",
        )
        producer_canonical_replay = _finite_number(
            item["canonicalized_function_relative"],
            label=f"trial {trial} producer canonicalized-function replay diagnostic",
        )
        conditioning_diagnostic_records.append({
            "trial": trial,
            "producer": {
                "raw_coordinate_replay_relative": producer_raw_replay,
                "canonicalized_function_relative": producer_canonical_replay,
            },
            "independent_checkpoint_npz_cpu": {
                "raw_coordinate_replay_relative": raw_replay,
                "canonicalized_function_relative": canonical_function_replay,
            },
            "absolute_copy_disagreement": {
                "raw_coordinate_replay_relative": abs(producer_raw_replay - raw_replay),
                "canonicalized_function_relative": abs(
                    producer_canonical_replay - canonical_function_replay
                ),
            },
        })
        _close_matrix_measurement(
            item["canonicalized_coefficient_replay_relative"],
            coefficient_replay,
            label=f"trial {trial} canonicalized_coefficient_replay_relative",
            absolute_tolerance=1e-8,
        )
        if require_all_matrix_gates and coefficient_replay > 1e-8:
            raise RuntimeError("NPZ-recomputed coefficient replay exceeds 1e-8")
        for bond, (entry, raw_gauge) in enumerate(
            zip(item["bond_realizations"], raw_gauges)
        ):
            base_factor = base_canonicalized_raw.raw_from_canonical[bond]
            gauged_factor = gauged_canonical.raw_from_canonical[bond]
            target_factor = raw_gauge @ base_factor
            left, _, right = torch.linalg.svd(
                gauged_factor.T @ target_factor, full_matrices=True
            )
            factor_transport = left @ right
            factor_error = float(
                (
                    torch.linalg.matrix_norm(
                        gauged_factor @ factor_transport - target_factor
                    )
                    / torch.linalg.matrix_norm(target_factor).clamp_min(1e-30)
                ).item()
            )
            _finite_number(
                factor_error,
                label=f"trial {trial} bond {bond} independently recomputed raw-factor diagnostic",
            )
    expected_required_count = trials * sum(
        sum(rank < 33 for rank in ranks)
        for ranks in gauge["required_ranks_by_bond"]
    )
    if len(required_decision_inventory) != expected_required_count:
        raise RuntimeError("required-projector decision inventory is incomplete")
    copy_names = (
        "producer",
        "stored_npz_cpu",
        "independent_checkpoint_raw_gauge_cpu",
    )
    aggregates = {}
    for copy_name in copy_names:
        decisions = [
            item["copies"][copy_name] for item in required_decision_inventory
        ]
        bounds = [
            item["bound_with_roundoff"]
            for item in decisions
            if item["bound_with_roundoff"] is not None
        ]
        nonvacuous = bool(decisions)
        aggregates[copy_name] = {
            "certificate_count": len(decisions),
            "all_numerically_certifiable": nonvacuous and all(
                item["numerically_certifiable"] is True for item in decisions
            ),
            "all_pass_davis_kahan_inequality": nonvacuous and all(
                item["passes_davis_kahan_inequality"] is True
                for item in decisions
            ),
            "all_bounds_le_maximum": nonvacuous and all(
                item["bound_with_roundoff_le_maximum"] is True
                for item in decisions
            ),
            "all_support_defects_le_1e-8": nonvacuous and all(
                item["maximum_supported_overlap_defect_le_1e-8"] is True
                for item in decisions
            ),
            "all_frozen_gates_pass": nonvacuous and all(
                item["all_frozen_gates_pass"] is True for item in decisions
            ),
            "maximum_bound_with_roundoff": max(bounds) if bounds else None,
            "maximum_supported_overlap_defect": max(
                (item["maximum_supported_overlap_defect"] for item in decisions),
                default=None,
            ),
        }
    aggregate_bounds = [
        item["maximum_bound_with_roundoff"]
        for item in aggregates.values()
        if item["maximum_bound_with_roundoff"] is not None
    ]
    aggregate_support_defects = [
        item["maximum_supported_overlap_defect"]
        for item in aggregates.values()
        if item["maximum_supported_overlap_defect"] is not None
    ]
    details = {
        "comparison_rule": (
            "strict-producer-to-stored-npz-plus-independent-required-gate-replication-v1"
        ),
        "frozen_thresholds": {
            "maximum_bound_with_roundoff": record["protocol"][
                "gauge_required_projector_maximum_bound"
            ],
            "maximum_supported_overlap_defect": 1e-8,
        },
        "required_certificate_count": len(required_decision_inventory),
        "expected_required_certificate_count": expected_required_count,
        "copies": aggregates,
        "worst_of_all_copies": {
            "maximum_bound_with_roundoff": (
                max(aggregate_bounds) if aggregate_bounds else None
            ),
            "maximum_supported_overlap_defect": (
                max(aggregate_support_defects)
                if aggregate_support_defects
                else None
            ),
            "all_frozen_gates_pass": all(
                item["all_frozen_gates_pass"] is True
                for item in aggregates.values()
            ),
        },
        "decision_inventory": required_decision_inventory,
        "non_load_bearing_conditioning_diagnostics": {
            "comparison_policy": (
                "report-both-cross-device-paths-without-a-pass-threshold-v1"
            ),
            "records": conditioning_diagnostic_records,
            "producer_maxima": {
                key: max(item["producer"][key] for item in conditioning_diagnostic_records)
                for key in (
                    "raw_coordinate_replay_relative",
                    "canonicalized_function_relative",
                )
            },
            "independent_checkpoint_npz_cpu_maxima": {
                key: max(
                    item["independent_checkpoint_npz_cpu"][key]
                    for item in conditioning_diagnostic_records
                )
                for key in (
                    "raw_coordinate_replay_relative",
                    "canonicalized_function_relative",
                )
            },
            "worst_of_both_maxima": {
                key: max(
                    max(
                        item["producer"][key],
                        item["independent_checkpoint_npz_cpu"][key],
                    )
                    for item in conditioning_diagnostic_records
                )
                for key in (
                    "raw_coordinate_replay_relative",
                    "canonicalized_function_relative",
                )
            },
        },
    }
    return (path, details) if include_authentication_details else path


def _gap_from_sparse_record(item: dict, *, rank: int, dimension: int, scale: float) -> dict:
    if not isinstance(item, dict) or set(item) != {
        "rank", "retained_value", "discarded_value", "absolute_gap",
        "relative_gap", "resolved",
    }:
        raise RuntimeError("all-bond eigengap field inventory differs")
    if item.get("rank") != rank or type(item.get("resolved")) is not bool:
        raise RuntimeError("all-bond eigengap rank or resolved status is invalid")
    retained = _finite_number(item.get("retained_value"), label="retained eigengap value")
    discarded = _finite_number(item.get("discarded_value"), label="discarded eigengap value")
    if rank == dimension:
        expected = {
            "rank": rank, "retained_value": retained, "discarded_value": 0.0,
            "absolute_gap": None, "relative_gap": None, "resolved": True,
        }
    else:
        absolute = retained - discarded
        relative = absolute / max(abs(scale), np.finfo(np.float64).tiny)
        tolerance = 1000 * np.finfo(np.float64).eps * dimension * max(
            abs(scale), np.finfo(np.float64).tiny
        )
        expected = {
            "rank": rank, "retained_value": retained, "discarded_value": discarded,
            "absolute_gap": absolute, "relative_gap": relative,
            "resolved": absolute > tolerance and relative > 1e-8,
        }
    if item != expected:
        raise RuntimeError("all-bond eigengap diagnostics are not self-consistent")
    return expected


def _all_bond_schedule() -> list[dict]:
    return [
        {
            "target_removal_basis_points": target,
            "rank_tuple": [rank] * 4,
        }
        for target, rank in zip(
            EXPECTED_PROTOCOL_COMMON["all_bond_target_removal_basis_points"],
            EXPECTED_PROTOCOL_COMMON["all_bond_expected_uniform_ranks"],
        )
    ]


def _dense_storage(ranks: list[int]) -> int:
    return ranks[0] * 3073 + sum(
        target * source * source for source, target in zip(ranks[:-1], ranks[1:])
    ) + 10 * ranks[-1]


def _all_bond_frontier(points: list[dict], reference: float) -> dict:
    threshold = reference - EXPECTED_PROTOCOL_COMMON["selection_accuracy_tolerance"]
    eligible = [
        index for index, point in enumerate(points)
        if point["canonical_all_boundaries_resolved"] is True
        and point["canonical_all_boundaries_roundoff_prefilter_pass"] is True
    ]
    if not eligible or eligible[0] != 0:
        raise RuntimeError("the all-bond full-rank point must be resolved")
    selected = eligible[0]
    first_failure = None
    for index in eligible:
        if points[index]["accuracy"]["canonical_tree_odt"]["discovery"] < threshold:
            first_failure = index
            break
        selected = index
    selected_rank_tuple = points[selected]["rank_tuple"]
    full_rank_tuple = points[0]["rank_tuple"]
    nontrivial_compression_selected = selected_rank_tuple != full_rank_tuple
    return {
        "selection_rule": (
            "most compressed member of the resolved analytic-roundoff-prefiltered "
            "discovery prefix before the first eligible point more than one percentage "
            "point below full rank; held-out gauge trials are validation-only"
        ),
        "selection_gauge_trial_count": GAUGE_SELECTION_TRIALS,
        "held_out_validation_gauge_seed": GAUGE_SEED,
        "held_out_validation_gauge_trial_count": 50,
        "held_out_validation_gauges_used_for_selection": False,
        "reference_discovery_accuracy": reference,
        "minimum_discovery_accuracy": threshold,
        "eligible_schedule_indices": eligible,
        "first_failing_eligible_schedule_index": first_failure,
        "selected_schedule_index": selected,
        "selected_rank_tuple": selected_rank_tuple,
        "nontrivial_compression_selected": nontrivial_compression_selected,
        "outcome": (
            "nontrivial_rank_tuple_selected"
            if nontrivial_compression_selected
            else "full_rank_selected_by_frozen_prefiltered_discovery_prefix_rule"
        ),
    }


def validate_all_bond_curve(record: dict) -> dict:
    curve = record.get("simultaneous_uniform_all_bond_curve")
    top_keys = {
        "claim_boundary", "target_removal_basis_points", "expected_uniform_ranks",
        "bond_dimensions", "bond_occurrence_multiplicities", "haar_draws",
        "haar_construction", "trace_totals", "local_spectra_by_bond",
        "maximum_trace_total_relative_disagreement", "points", "discovery_selection",
    }
    if not isinstance(curve, dict) or set(curve) != top_keys:
        raise RuntimeError("all-bond curve field inventory differs")
    expected_schedule = _all_bond_schedule()
    if (
        not _typed_equal(
            curve["target_removal_basis_points"],
            EXPECTED_PROTOCOL_COMMON["all_bond_target_removal_basis_points"],
        )
        or not _typed_equal(
            curve["expected_uniform_ranks"],
            EXPECTED_PROTOCOL_COMMON["all_bond_expected_uniform_ranks"],
        )
        or not _typed_equal(curve["bond_dimensions"], list(ALL_BOND_DIMENSIONS))
        or not _typed_equal(
            curve["bond_occurrence_multiplicities"], list(ALL_BOND_MULTIPLICITIES)
        )
        or curve["haar_draws"] != 64
        or curve["claim_boundary"] != EXPECTED_ALL_BOND_CLAIM_BOUNDARY
        or curve["haar_construction"] != EXPECTED_HAAR_CONSTRUCTION
    ):
        raise RuntimeError("all-bond frozen metadata differs")
    points = curve.get("points")
    if not isinstance(points, list) or len(points) != len(expected_schedule):
        raise RuntimeError("all-bond point inventory differs")

    base_spectra = record["gauge_audit"]["base_spectra_by_bond"]
    local_spectra = curve.get("local_spectra_by_bond")
    if (
        not isinstance(local_spectra, list)
        or len(local_spectra) != 4
        or any(
            not isinstance(values, list)
            or len(values) != 33
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
            or any(values[index] < values[index + 1] for index in range(32))
            for values in local_spectra
        )
    ):
        raise RuntimeError("all-bond local-spectrum inventory differs")
    coefficient_norm = None
    full_storage = _dense_storage(list(ALL_BOND_DIMENSIONS))
    unique_full = sum(ALL_BOND_DIMENSIONS)
    occurrence_full = sum(m * d for m, d in zip(ALL_BOND_MULTIPLICITIES, ALL_BOND_DIMENSIONS))
    for point_index, (point, scheduled) in enumerate(zip(points, expected_schedule)):
        expected_point_keys = {
            "schedule_index", "target_removal_basis_points", "uniform_rank", "rank_tuple",
            "canonical_boundaries", "canonical_all_boundaries_resolved",
            "canonical_roundoff_only_prefilters",
            "canonical_all_boundaries_roundoff_prefilter_pass",
            "local_boundaries", "local_all_boundaries_resolved", "compression",
            "coefficient_certificate", "accuracy",
        }
        if not isinstance(point, dict) or set(point) != expected_point_keys:
            raise RuntimeError("all-bond point field inventory differs")
        ranks = scheduled["rank_tuple"]
        rank = ranks[0]
        if (
            type(point["schedule_index"]) is not int
            or type(point["target_removal_basis_points"]) is not int
            or type(point["uniform_rank"]) is not int
            or point["schedule_index"] != point_index
            or point["target_removal_basis_points"] != scheduled["target_removal_basis_points"]
            or point["uniform_rank"] != rank
            or not _typed_equal(point["rank_tuple"], ranks)
        ):
            raise RuntimeError("all-bond rank schedule differs")
        for boundary_name in ("canonical_boundaries", "local_boundaries"):
            boundaries = point[boundary_name]
            if not isinstance(boundaries, list) or len(boundaries) != 4:
                raise RuntimeError("all-bond boundary inventory differs")
            expected_boundaries = []
            for bond, boundary in enumerate(boundaries):
                if boundary_name == "canonical_boundaries":
                    expected = expected_eigengap(base_spectra[bond], rank)
                    if boundary != expected:
                        raise RuntimeError("all-bond canonical boundary disagrees with base spectrum")
                else:
                    expected = expected_eigengap(local_spectra[bond], rank)
                    if boundary != expected:
                        raise RuntimeError(
                            "all-bond local boundary disagrees with serialized local spectrum"
                        )
                expected_boundaries.append(expected)
            expected_status = all(item["resolved"] is True for item in expected_boundaries)
            status_name = (
                "canonical_all_boundaries_resolved"
                if boundary_name == "canonical_boundaries" else "local_all_boundaries_resolved"
            )
            if point[status_name] is not expected_status:
                raise RuntimeError("all-bond resolved status disagrees with boundary diagnostics")

        expected_prefilters = [
            expected_roundoff_prefilter(spectrum, rank) for spectrum in base_spectra
        ]
        if (
            point["canonical_roundoff_only_prefilters"] != expected_prefilters
            or point["canonical_all_boundaries_roundoff_prefilter_pass"]
            is not all(item["passes"] is True for item in expected_prefilters)
        ):
            raise RuntimeError("all-bond analytic roundoff prefilter differs")

        compression = point["compression"]
        expected_compression = {
            "unique_bond_coordinate_removal_fraction": 1.0 - sum(ranks) / unique_full,
            "occurrence_weighted_bond_coordinate_removal_fraction": 1.0
            - sum(m * r for m, r in zip(ALL_BOND_MULTIPLICITIES, ranks)) / occurrence_full,
            "dense_homogeneous_tn_full_storage_scalars": full_storage,
            "dense_homogeneous_tn_retained_storage_scalars": _dense_storage(ranks),
            "dense_homogeneous_tn_storage_removal_fraction": 1.0 - _dense_storage(ranks) / full_storage,
        }
        if not _typed_equal(compression, expected_compression):
            raise RuntimeError("all-bond compression accounting differs")

        certificate = point["coefficient_certificate"]
        expected_certificate_keys = {
            "object", "error_evaluation", "full_coefficient_norm_squared", "tail_terms",
            "hsvd_upper_bound_squared", "hsvd_relative_error_upper_bound",
            "actual_coefficient_error_squared", "actual_relative_coefficient_error",
            "bound_numerical_slack",
            "actual_error_within_bound_plus_declared_numerical_slack",
        }
        if not isinstance(certificate, dict) or set(certificate) != expected_certificate_keys:
            raise RuntimeError("all-bond coefficient-certificate inventory differs")
        norm = _finite_number(
            certificate["full_coefficient_norm_squared"], label="full coefficient norm", nonnegative=True
        )
        if norm <= 0:
            raise RuntimeError("all-bond coefficient norm is nonpositive or changes across points")
        if coefficient_norm is not None:
            _close_number(
                norm,
                coefficient_norm,
                label="all-bond coefficient norm across points",
            )
        coefficient_norm = norm
        terms = certificate["tail_terms"]
        if not isinstance(terms, list) or len(terms) != 4:
            raise RuntimeError("all-bond tail-term inventory differs")
        expected_terms = []
        for bond, (spectrum, multiplicity) in enumerate(zip(base_spectra, ALL_BOND_MULTIPLICITIES)):
            tail = sum(max(0.0, value) for value in spectrum[rank:])
            expected_terms.append({
                "bond": bond, "rank": rank, "multiplicity": multiplicity,
                "discarded_trace": tail, "weighted_discarded_trace": multiplicity * tail,
            })
        for observed, expected in zip(terms, expected_terms):
            if (
                not isinstance(observed, dict)
                or set(observed) != set(expected)
                or any(
                    type(observed[key]) is not int or observed[key] != expected[key]
                    for key in ("bond", "rank", "multiplicity")
                )
            ):
                raise RuntimeError("all-bond coefficient tail inventory disagrees with base spectra")
            _close_number(
                observed["discarded_trace"],
                expected["discarded_trace"],
                label="all-bond discarded trace",
            )
            _close_number(
                observed["weighted_discarded_trace"],
                expected["weighted_discarded_trace"],
                label="all-bond weighted discarded trace",
            )
        bound = sum(item["weighted_discarded_trace"] for item in expected_terms)
        actual = _finite_number(
            certificate["actual_coefficient_error_squared"],
            label="actual coefficient error", nonnegative=True,
        )
        slack = 1e-8 * bound + 1e-10 * norm
        expected_derived = {
            "object": "topology_specific_independent_clone_tree",
            "error_evaluation": "canonicalized_block_sparse_difference_tree",
            "full_coefficient_norm_squared": norm,
            "tail_terms": expected_terms,
            "hsvd_upper_bound_squared": bound,
            "hsvd_relative_error_upper_bound": math.sqrt(bound / norm),
            "actual_coefficient_error_squared": actual,
            "actual_relative_coefficient_error": math.sqrt(actual / norm),
            "bound_numerical_slack": slack,
            "actual_error_within_bound_plus_declared_numerical_slack": (
                actual <= bound + slack
            ),
        }
        if (
            certificate["object"] != expected_derived["object"]
            or certificate["error_evaluation"] != expected_derived["error_evaluation"]
            or certificate["actual_error_within_bound_plus_declared_numerical_slack"]
            is not expected_derived[
                "actual_error_within_bound_plus_declared_numerical_slack"
            ]
        ):
            raise RuntimeError("all-bond coefficient certificate does not recompute")
        for key in (
            "full_coefficient_norm_squared",
            "hsvd_upper_bound_squared",
            "hsvd_relative_error_upper_bound",
            "actual_coefficient_error_squared",
            "actual_relative_coefficient_error",
            "bound_numerical_slack",
        ):
            _close_number(
                certificate[key],
                expected_derived[key],
                label=f"all-bond coefficient certificate {key}",
            )

        accuracy = point["accuracy"]
        if not isinstance(accuracy, dict) or set(accuracy) != {
            "canonical_tree_odt", "canonical_adjacent_local", "independent_nested_haar"
        }:
            raise RuntimeError("all-bond accuracy inventory differs")
        for method in ("canonical_tree_odt", "canonical_adjacent_local"):
            endpoint = accuracy[method]
            if not isinstance(endpoint, dict) or set(endpoint) != {"discovery", "confirmation"}:
                raise RuntimeError("all-bond deterministic accuracy inventory differs")
            for split in ("discovery", "confirmation"):
                _accuracy(endpoint[split], label=f"all-bond {method} {split} accuracy")
        haar = accuracy["independent_nested_haar"]
        if not isinstance(haar, dict) or set(haar) != {
            "draw_count", "discovery", "confirmation", "discovery_mean",
            "discovery_sample_standard_deviation", "confirmation_mean",
            "confirmation_sample_standard_deviation",
        } or haar["draw_count"] != 64:
            raise RuntimeError("all-bond Haar inventory differs")
        for split in ("discovery", "confirmation"):
            values = haar[split]
            if not isinstance(values, list) or len(values) != 64:
                raise RuntimeError("all-bond Haar draw count differs")
            for value in values:
                _accuracy(value, label=f"all-bond Haar {split} accuracy")
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            _close_number(
                haar[f"{split}_mean"],
                mean,
                label=f"all-bond Haar {split} mean",
            )
            _close_number(
                haar[f"{split}_sample_standard_deviation"],
                std,
                label=f"all-bond Haar {split} sample standard deviation",
            )

    trace_totals = curve["trace_totals"]
    expected_trace_totals = [sum(max(0.0, value) for value in spectrum) for spectrum in base_spectra]
    if not isinstance(trace_totals, list) or len(trace_totals) != 4 or coefficient_norm is None:
        raise RuntimeError("all-bond trace totals disagree with base spectra")
    for observed, expected in zip(trace_totals, expected_trace_totals):
        _close_number(observed, expected, label="all-bond trace total")
    trace_disagreement = max(abs(value - coefficient_norm) / coefficient_norm for value in trace_totals)
    _close_number(
        curve["maximum_trace_total_relative_disagreement"],
        trace_disagreement,
        label="all-bond trace disagreement aggregate",
    )

    reference = points[0]["accuracy"]["canonical_tree_odt"]["discovery"]
    expected_selection = _all_bond_frontier(points, reference)
    selected = points[expected_selection["selected_schedule_index"]]
    selected_confirmation = selected["accuracy"]["canonical_tree_odt"]["confirmation"]
    haar_confirmation = selected["accuracy"]["independent_nested_haar"]["confirmation"]
    exceedances = sum(value >= selected_confirmation for value in haar_confirmation)
    expected_selection.update({
        "selected_discovery_accuracy": selected["accuracy"]["canonical_tree_odt"]["discovery"],
        "selected_confirmation_accuracy": selected_confirmation,
        "selected_confirmation_drop_from_full": points[0]["accuracy"]["canonical_tree_odt"]["confirmation"] - selected_confirmation,
        "selected_local_all_boundaries_resolved": selected["local_all_boundaries_resolved"],
        "selected_confirmation_odt_minus_local": selected_confirmation - selected["accuracy"]["canonical_adjacent_local"]["confirmation"],
        "selected_confirmation_haar_at_least_canonical_count": exceedances,
        "selected_confirmation_haar_randomization_p": (1 + exceedances) / 65,
    })
    if curve["discovery_selection"] != expected_selection:
        raise RuntimeError("all-bond selection frontier or confirmation statistics differ")
    return {
        "uniform_schedule_exact": True,
        "trace_disagreement": trace_disagreement,
        "full_rank_relative_coefficient_error": points[0]["coefficient_certificate"]["actual_relative_coefficient_error"],
        "selected_all_canonical_boundaries_resolved": selected["canonical_all_boundaries_resolved"],
        "selected_all_canonical_roundoff_prefilter_pass": selected[
            "canonical_all_boundaries_roundoff_prefilter_pass"
        ],
        "all_coefficient_errors_within_bounds_plus_declared_numerical_slack": all(
            point["coefficient_certificate"][
                "actual_error_within_bound_plus_declared_numerical_slack"
            ] is True
            for point in points
        ),
        "selected_confirmation_noninferior": expected_selection["selected_confirmation_drop_from_full"] <= 0.01,
        "selected_odt_gt_local_with_resolved_boundaries": (
            expected_selection["selected_local_all_boundaries_resolved"] is True
            and type(expected_selection["selected_confirmation_odt_minus_local"]) in (int, float)
            and math.isfinite(expected_selection["selected_confirmation_odt_minus_local"])
            and expected_selection["selected_confirmation_odt_minus_local"] > 0.0
        ),
        "selected_haar_randomization_p_le_0p05": expected_selection["selected_confirmation_haar_randomization_p"] <= 0.05,
        "selected": expected_selection,
    }


def validate_protocol_and_measurements(
    record: dict,
    *,
    smoke: bool,
    expected_status: str = "complete",
    require_all_gates: bool = True,
) -> tuple[dict[str, bool], dict[str, bool]]:
    validate_finite_json_numbers(record)
    expected_top_keys = {
        "schema", "status", "claim_boundary", "seed", "predecessor_failures",
        "protocol", "training_discovery_accuracy_by_epoch", "references",
        "post_training_rbn_calibration",
        "canonicalization", "gauge_audit", "canonical_spectrum",
        "canonical_eigengap_diagnostics", "canonical_trace_tail_fraction", "curves",
        "canonical_rank_selection",
        "haar_curves", "haar_discovery_selected_confirmation",
        "discovery_selected_confirmation", "simultaneous_uniform_all_bond_curve",
        "provenance", "algebraic_gates", "capability_gates",
    }
    if set(record) != expected_top_keys:
        raise RuntimeError("result top-level field inventory differs")
    if (
        record.get("schema") != SCHEMA
        or record.get("status") != expected_status
        or record.get("claim_boundary") != EXPECTED_CLAIM_BOUNDARY
        or type(record.get("seed")) is not int
        or record.get("predecessor_failures") != EXPECTED_PREDECESSOR_FAILURES
    ):
        raise RuntimeError("result identity or structured predecessor-failure lineage differs")
    protocol = record.get("protocol")
    if not isinstance(protocol, dict) or set(protocol) != EXPECTED_PROTOCOL_KEYS:
        raise RuntimeError("result protocol field inventory differs")
    if any(
        not _typed_equal(protocol.get(key), value)
        for key, value in EXPECTED_PROTOCOL_COMMON.items()
    ):
        raise RuntimeError("result protocol differs from the frozen experiment")
    expected_ranks = SMOKE_RANK_GRID if smoke else FULL_RANK_GRID
    expected_haar = 1 if smoke else 16
    if (
        not _typed_equal(protocol.get("rank_grid"), expected_ranks)
        or type(protocol.get("haar_draws")) is not int
        or protocol.get("haar_draws") != expected_haar
    ):
        raise RuntimeError("result rank grid or Haar count differs from the frozen experiment")
    history = record.get("training_discovery_accuracy_by_epoch", [])
    if len(history) != protocol["epochs"] or any(
        type(value) not in (int, float)
        or not np.isfinite(value)
        or not 0.0 <= value <= 1.0
        for value in history
    ):
        raise RuntimeError("training history length differs from the frozen epoch count")
    calibration = record.get("post_training_rbn_calibration")
    expected_calibration_paths = [f"norms.{index}" for index in range(protocol["layers"])]
    if (
        not isinstance(calibration, dict)
        or set(calibration) != {"sites", "excluded_paths"}
        or calibration["excluded_paths"] != []
        or not isinstance(calibration["sites"], list)
        or [site.get("path") for site in calibration["sites"]] != expected_calibration_paths
        or any(
            not isinstance(site, dict)
            or set(site) != {"path", "scale", "batches"}
            or type(site["scale"]) not in (int, float)
            or not math.isfinite(float(site["scale"]))
            or site["scale"] <= 0
            or site["batches"] != protocol["post_training_rbn_calibration_batches"]
            for site in calibration["sites"]
        )
    ):
        raise RuntimeError("post-training scalar-RBN calibration inventory differs")
    if (
        protocol["discovery_indices_sha256"] != EXPECTED_DISCOVERY_SHA256
        or protocol["confirmation_indices_sha256"] != EXPECTED_CONFIRMATION_SHA256
    ):
        raise RuntimeError("result split hashes differ from the frozen panels")

    curves = record.get("curves", {})
    if set(curves) != EXPECTED_METHODS:
        raise RuntimeError("result curve method inventory differs")
    canonical_keys = {str(rank) for rank in expected_ranks}
    affine_keys = {str(rank - 1) for rank in expected_ranks}
    for method, curve in curves.items():
        expected_keys = (
            affine_keys
            if method in {"legacy_raw_downstream_zero_anchor", "activation_pca_centered"}
            else canonical_keys
        )
        if set(curve) != {"discovery", "confirmation"} or any(
            set(curve[split]) != expected_keys for split in ("discovery", "confirmation")
        ):
            raise RuntimeError("result curve rank inventory differs")
        for split in ("discovery", "confirmation"):
            for value in curve[split].values():
                _accuracy(value, label=f"{method} {split} accuracy")
    haar_curves = record.get("haar_curves")
    if (
        not isinstance(haar_curves, dict)
        or set(haar_curves) != {str(index) for index in range(expected_haar)}
    ):
        raise RuntimeError("result Haar inventory differs")
    if any(
        set(curve) != {"discovery", "confirmation"}
        or set(curve["discovery"]) != canonical_keys
        or set(curve["confirmation"]) != canonical_keys
        for curve in haar_curves.values()
    ):
        raise RuntimeError("result Haar rank inventory differs")
    for curve in haar_curves.values():
        for split in ("discovery", "confirmation"):
            for value in curve[split].values():
                _accuracy(value, label=f"single-bond Haar {split} accuracy")

    spectrum = record.get("canonical_spectrum")
    if (
        not isinstance(spectrum, list)
        or len(spectrum) != protocol["homogeneous_bond_width"]
        or any(spectrum[index] < spectrum[index + 1] for index in range(len(spectrum) - 1))
    ):
        raise RuntimeError("canonical spectrum is malformed or not descending")
    eigengaps = record.get("canonical_eigengap_diagnostics")
    expected_eigengaps = {
        str(rank): expected_eigengap(spectrum, rank) for rank in expected_ranks
    }
    if eigengaps != expected_eigengaps:
        raise RuntimeError("recorded eigengaps disagree with the canonical spectrum")
    resolved_global_ranks = {
        rank for rank in expected_ranks if expected_eigengaps[str(rank)]["resolved"]
    }
    nonnegative_spectrum = [max(0.0, float(value)) for value in spectrum]
    trace_total = sum(nonnegative_spectrum)
    if not math.isfinite(trace_total) or trace_total <= 0.0:
        raise RuntimeError("canonical spectrum has nonpositive total mass")
    expected_tails = {
        str(rank): sum(nonnegative_spectrum[rank:]) / trace_total
        for rank in expected_ranks
    }
    if record.get("canonical_trace_tail_fraction") != expected_tails:
        raise RuntimeError("canonical trace tails disagree with the spectrum")

    selected = record.get("discovery_selected_confirmation")
    if not isinstance(selected, dict) or set(selected) != EXPECTED_METHODS:
        raise RuntimeError("result selected-method inventory differs")
    references = record.get("references")
    if not isinstance(references, dict) or set(references) != {
        "module", "homogeneous_export", "canonical_full"
    } or any(
        not isinstance(endpoint, dict) or set(endpoint) != {"discovery", "confirmation"}
        for endpoint in references.values()
    ):
        raise RuntimeError("reference accuracy inventory differs")
    for name, endpoint in references.items():
        for split, value in endpoint.items():
            _accuracy(value, label=f"{name} {split} accuracy")
    module_confirmation = references["module"]["confirmation"]
    canonical_reference_rank = max(
        int(value) for value in curves["canonical_tree_odt"]["discovery"]
    )
    expected_canonical_rank, expected_rank_selection = expected_canonical_rank_selection(
        curves["canonical_tree_odt"],
        curves["canonical_tree_odt"]["discovery"][str(canonical_reference_rank)],
        spectrum,
    )
    if record.get("canonical_rank_selection") != expected_rank_selection:
        raise RuntimeError("canonical analytic rank-selection record differs")
    for method, endpoint in selected.items():
        curve = curves[method]
        rank = endpoint.get("rank")
        if not isinstance(rank, int) or str(rank) not in curve["discovery"]:
            raise RuntimeError("selected rank is absent from its curve")
        reference_rank = max(int(value) for value in curve["discovery"])
        expected_rank = (
            expected_canonical_rank
            if method == "canonical_tree_odt"
            else frozen_selected_rank(
                curve,
                curve["discovery"][str(reference_rank)],
            )
        )
        if rank != expected_rank:
            raise RuntimeError("selected rank differs from the frozen discovery rule")
        expected_endpoint = {
            "rank": rank,
            "effective_affine_rank": (
                rank + 1
                if method in {"legacy_raw_downstream_zero_anchor", "activation_pca_centered"}
                else rank
            ),
            "discovery_accuracy": curve["discovery"][str(rank)],
            "confirmation_accuracy": curve["confirmation"][str(rank)],
            "reference_rank": reference_rank,
            "reference_discovery_accuracy": curve["discovery"][str(reference_rank)],
            "confirmation_drop_from_module": (
                module_confirmation - curve["confirmation"][str(rank)]
            ),
            "eigengap_resolved": (
                eigengaps[str(rank)]["resolved"]
                if method == "canonical_tree_odt" else None
            ),
        }
        if endpoint != expected_endpoint:
            raise RuntimeError("selected endpoint disagrees with recorded curves")

    haar_selected = record.get("haar_discovery_selected_confirmation")
    if not isinstance(haar_selected, dict) or set(haar_selected) != set(haar_curves):
        raise RuntimeError("selected Haar inventory differs")
    for draw, endpoint in haar_selected.items():
        rank = endpoint.get("rank")
        curve = haar_curves[draw]
        if not isinstance(rank, int) or str(rank) not in curve["discovery"]:
            raise RuntimeError("selected Haar rank is absent from its curve")
        if rank != frozen_selected_rank(
            curve, record["references"]["canonical_full"]["discovery"]
        ):
            raise RuntimeError("selected Haar rank differs from the frozen discovery rule")
        if endpoint != {
            "rank": rank,
            "discovery_accuracy": curve["discovery"][str(rank)],
            "confirmation_accuracy": curve["confirmation"][str(rank)],
        }:
            raise RuntimeError("selected Haar endpoint disagrees with its curve")

    canonicalization = record["canonicalization"]
    canonicalization_keys = {
        "isometry_errors", "symmetry_errors", "factorization_errors",
        "module_vs_homogeneous_discovery_abs", "module_vs_canonical_discovery_abs",
        "module_vs_homogeneous_confirmation_abs", "module_vs_canonical_confirmation_abs",
        "sampled_logit_reconstruction", "sampled_logit_reconstruction_count",
        "checkpoint_reload_max_absolute_logit_error", "sampled_logit_gate_max_absolute",
        "sampled_logit_gate_pass",
    }
    if not isinstance(canonicalization, dict) or set(canonicalization) != canonicalization_keys:
        raise RuntimeError("canonicalization field inventory differs")
    for name, expected_length in (
        ("isometry_errors", 4), ("factorization_errors", 4), ("symmetry_errors", 3)
    ):
        values = canonicalization.get(name)
        if (
            not isinstance(values, list)
            or len(values) != expected_length
            or any(
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in values
            )
        ):
            raise RuntimeError("canonicalization error inventory or values differ")
    reconstruction_records = canonicalization.get("sampled_logit_reconstruction")
    if (
        not isinstance(reconstruction_records, dict)
        or set(reconstruction_records)
        != {"homogeneous_export", "canonical_full", "full_rank_diagonalized"}
        or any(
            not isinstance(item, dict)
            or set(item)
            != {"max_absolute_logit_error", "relative_frobenius_logit_error"}
            or any(
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in item.values()
            )
            for item in reconstruction_records.values()
        )
    ):
        raise RuntimeError("sampled reconstruction inventory or values differ")
    reconstruction = max(
        item["max_absolute_logit_error"]
        for item in reconstruction_records.values()
    )
    if (
        canonicalization.get("sampled_logit_reconstruction_count") != 256
        or canonicalization.get("sampled_logit_gate_max_absolute") != 1e-8
        or canonicalization.get("sampled_logit_gate_pass") is not (reconstruction <= 1e-8)
        or canonicalization.get("checkpoint_reload_max_absolute_logit_error") != 0.0
    ):
        raise RuntimeError("sampled reconstruction metadata disagrees with measurements")
    for split in ("discovery", "confirmation"):
        module = references["module"][split]
        homogeneous = references["homogeneous_export"][split]
        canonical = references["canonical_full"][split]
        if canonicalization[f"module_vs_homogeneous_{split}_abs"] != abs(module - homogeneous):
            raise RuntimeError("module/export accuracy delta is inconsistent")
        if canonicalization[f"module_vs_canonical_{split}_abs"] != abs(module - canonical):
            raise RuntimeError("module/canonical accuracy delta is inconsistent")
    all_bond = validate_all_bond_curve(record)
    selected_all_bond_ranks = all_bond["selected"]["selected_rank_tuple"]
    expected_required = [[rank] for rank in selected_all_bond_ranks]
    analysis_bond = protocol["analysis_bond"]
    expected_required[analysis_bond] = sorted(
        set(expected_required[analysis_bond]) | {selected["canonical_tree_odt"]["rank"]}
    )
    expected_smoke_diagnostic = 1 if smoke else None
    if expected_smoke_diagnostic is not None:
        expected_required = [
            sorted(set(ranks) | {expected_smoke_diagnostic})
            for ranks in expected_required
        ]
    if (
        protocol.get("gauge_required_ranks_by_bond") != expected_required
        or protocol.get("smoke_nontrivial_diagnostic_rank")
        != expected_smoke_diagnostic
        or record["gauge_audit"].get("required_ranks_by_bond") != expected_required
    ):
        raise RuntimeError("gauge required ranks do not match the frozen certificate inventory")
    gauge = record["gauge_audit"]
    computed_gauge = recompute_gauge_audit(gauge, protocol["gauge_trials"])
    if set(gauge) != {
        "base_spectra_by_bond", "resolved_ranks_by_bond", "required_ranks_by_bond",
        "records", "matrix_artifact", *computed_gauge,
    } or any(
        gauge[key] != value for key, value in computed_gauge.items()
    ):
        raise RuntimeError("reported gauge aggregates disagree with trial records")
    computed_algebraic = {
        "sampled_logit_reconstruction_le_1e-8": reconstruction <= 1e-8,
        "factorization_relative_le_1e-10": max(canonicalization["factorization_errors"]) <= 1e-10,
        "row_isometry_le_1e-10": max(canonicalization["isometry_errors"]) <= 1e-10,
        "gauge_sampled_single_bond_logit_replay_relative_le_1e-8": (
            gauge["max_sampled_single_bond_logit_replay_relative"] <= 1e-8
        ),
        "gauge_singular_extrema_relative_le_1e-12": (
            gauge["max_singular_extrema_relative_error"] <= 1e-12
        ),
        "gauge_condition_relative_le_1e-12": (
            gauge["max_condition_relative_error"] <= 1e-12
        ),
        "gauge_nonsymmetry_relative_gt_1e-6": (
            gauge["minimum_nonsymmetry_relative"] > 1e-6
        ),
        "gauge_coefficient_replay_relative_le_1e-8": (
            gauge["max_canonicalized_coefficient_replay_relative"] <= 1e-8
        ),
        "gauge_transport_orthogonality_le_1e-10": gauge["max_transport_orthogonality"] <= 1e-10,
        "gauge_gram_covariance_relative_le_1e-8": gauge["max_gram_covariance_relative"] <= 1e-8,
        "gauge_spectrum_relative_le_1e-8": gauge["max_spectrum_relative"] <= 1e-8,
        "gauge_projector_certificate_coverage_complete": (
            gauge["projector_certificate_coverage_complete"] is True
        ),
        "gauge_required_projector_overlap_support_defect_le_1e-8": (
            gauge["maximum_required_projector_overlap_support_defect"] is not None
            and gauge["maximum_required_projector_overlap_support_defect"] <= 1e-8
        ),
        "gauge_required_projectors_numerically_certifiable": (
            gauge["all_required_projectors_numerically_certifiable"] is True
        ),
        "gauge_required_projectors_pass_nonvacuous_bound": (
            gauge["all_required_projectors_pass_bound"] is True
            and gauge["maximum_required_projector_bound_with_roundoff"] is not None
            and gauge["maximum_required_projector_bound_with_roundoff"]
            <= protocol["gauge_required_projector_maximum_bound"]
        ),
        "selected_canonical_eigengap_resolved": selected["canonical_tree_odt"].get("eigengap_resolved") is True,
        "selected_canonical_roundoff_only_prefilter_pass": record[
            "canonical_rank_selection"
        ]["prefilter_by_rank"][str(record["canonical_rank_selection"]["selected_rank"])][
            "passes"
        ]
        is True,
        "all_bond_uniform_schedule_exact": all_bond["uniform_schedule_exact"],
        "all_bond_trace_totals_relative_le_1e-8": all_bond["trace_disagreement"] <= 1e-8,
        "all_bond_full_rank_coefficient_reconstruction_le_1e-8": (
            all_bond["full_rank_relative_coefficient_error"] <= 1e-8
        ),
        "all_bond_selected_canonical_eigengaps_resolved": (
            all_bond["selected_all_canonical_boundaries_resolved"] is True
        ),
        "all_bond_selected_canonical_roundoff_only_prefilter_pass": (
            all_bond["selected_all_canonical_roundoff_prefilter_pass"] is True
        ),
        "all_bond_coefficient_errors_within_hsvd_bounds_plus_declared_numerical_slack": (
            all_bond[
                "all_coefficient_errors_within_bounds_plus_declared_numerical_slack"
            ] is True
        ),
    }
    if (
        record.get("algebraic_gates") != computed_algebraic
        or require_all_gates and not all(computed_algebraic.values())
    ):
        raise RuntimeError("reported algebraic gates disagree with measurements")
    floor = protocol["minimum_full_model_accuracy"]
    computed_capability = {
        "module_discovery_accuracy_ge_minimum": record["references"]["module"]["discovery"] >= floor,
        "module_confirmation_accuracy_ge_minimum": record["references"]["module"]["confirmation"] >= floor,
    }
    if (
        record.get("capability_gates") != computed_capability
        or require_all_gates and not all(computed_capability.values())
    ):
        raise RuntimeError("reported capability gates disagree with measurements")
    if not require_all_gates and all((*computed_algebraic.values(), *computed_capability.values())):
        raise RuntimeError("gate-failed result contains no independently recomputed false gate")
    return computed_algebraic, computed_capability


def validate_seed_provenance(
    record: dict,
    *,
    project_root: Path,
    manifest_path: Path,
    manifest_text: str,
    manifest_sha256: str,
    frozen_entries: dict[str, str],
) -> None:
    provenance = record.get("provenance", {})
    if not isinstance(provenance, dict) or set(provenance) != {
        "source_sha256", "frozen_source_manifest", "dataset_root", "dataset_files",
        "checkpoint", "runtime",
    }:
        raise RuntimeError(f"seed {record.get('seed')} provenance field inventory differs")
    embedded = provenance.get("frozen_source_manifest")
    if not isinstance(embedded, dict):
        raise RuntimeError(f"seed {record.get('seed')} lacks its frozen manifest")
    embedded_manifest_path = require_confined_path(
        project_root,
        Path(embedded.get("path", "")),
        label=f"seed {record.get('seed')} embedded source manifest",
        kind="file",
    )
    if embedded_manifest_path != manifest_path:
        raise RuntimeError(f"seed {record.get('seed')} manifest path is outside this run root")
    if embedded.get("sha256") != manifest_sha256 or embedded.get("text") != manifest_text:
        raise RuntimeError(f"seed {record.get('seed')} was produced from a stale source manifest")
    if parse_manifest_text(embedded["text"]) != frozen_entries:
        raise RuntimeError(f"seed {record.get('seed')} embedded manifest entries differ")

    expected_source_hashes = {
        relative: frozen_entries[relative] for relative in RESULT_SOURCE_PATHS
    }
    if provenance.get("source_sha256") != expected_source_hashes:
        raise RuntimeError(f"seed {record.get('seed')} source hash map mismatch")
    if provenance.get("dataset_files") != EXPECTED_DATASET_FILES:
        raise RuntimeError(f"seed {record.get('seed')} official SVHN manifest mismatch")
    dataset_root = Path(provenance.get("dataset_root", "")).resolve()
    if dataset_root == project_root.resolve() or project_root.resolve() in dataset_root.parents:
        raise RuntimeError(f"seed {record.get('seed')} dataset root is inside the frozen run")

    seed = record.get("seed")
    checkpoint = provenance.get("checkpoint", {})
    checkpoint_name = "smoke.pt" if seed == 9173 else f"seed_{seed}.pt"
    expected_checkpoint = project_root / "results" / checkpoint_name
    checkpoint_path = require_confined_path(
        project_root,
        Path(checkpoint.get("path", "")),
        label=f"seed {seed} checkpoint",
        kind="file",
    )
    if checkpoint_path != expected_checkpoint:
        raise RuntimeError(f"seed {seed} checkpoint path is missing or outside this run")
    if checkpoint.get("bytes") != checkpoint_path.stat().st_size:
        raise RuntimeError(f"seed {seed} checkpoint size mismatch")
    if checkpoint.get("sha256") != digest(checkpoint_path):
        raise RuntimeError(f"seed {seed} checkpoint hash mismatch")

    runtime = provenance.get("runtime", {})
    expected_runtime_keys = {
        "python", "python_executable_invoked", "python_executable_resolved",
        "platform", "torch", "torchvision", "numpy",
        "cuda", "cudnn", "distributions", "fingerprint_sha256", "gpu", "scheduler",
        "slurm_job_id", "slurm_array_job_id", "slurm_array_task_id", "deterministic_algorithms",
        "cudnn_deterministic", "cudnn_benchmark", "float32_matmul_precision",
        "cublas_workspace_config",
    }
    if not isinstance(runtime, dict) or set(runtime) != expected_runtime_keys:
        raise RuntimeError(f"seed {seed} runtime field inventory differs")
    distributions = runtime["distributions"]
    if (
        not isinstance(distributions, list)
        or not distributions
        or any(
            not isinstance(item, list) or len(item) != 2
            or any(not isinstance(value, str) or not value for value in item)
            for item in distributions
        )
        or distributions != sorted(distributions, key=lambda item: (item[0].lower(), item[1]))
    ):
        raise RuntimeError(f"seed {seed} installed-distribution provenance differs")
    fingerprint_payload = {
        key: runtime[key] for key in (
            "python", "python_executable_invoked", "python_executable_resolved",
            "torch", "torchvision", "numpy",
            "cuda", "cudnn", "distributions",
        )
    }
    expected_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if runtime.get("fingerprint_sha256") != expected_fingerprint:
        raise RuntimeError(f"seed {seed} environment fingerprint does not authenticate its payload")
    if (
        runtime["python_executable_invoked"]
        != "/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python"
        or runtime["python_executable_resolved"]
        != "/work/joy/safesae-openvla/bin/python3.10"
        or runtime["deterministic_algorithms"] is not True
        or runtime["cudnn_deterministic"] is not True
        or runtime["cudnn_benchmark"] is not False
        or runtime["float32_matmul_precision"] != "highest"
        or runtime["cublas_workspace_config"] != ":4096:8"
    ):
        raise RuntimeError(f"seed {seed} deterministic runtime settings differ")
    gpu = runtime["gpu"]
    if (
        not isinstance(gpu, dict) or set(gpu) != {"name", "capability", "total_memory_bytes"}
        or not isinstance(gpu["name"], str) or not gpu["name"]
        or not isinstance(gpu["capability"], list) or len(gpu["capability"]) != 2
        or any(type(value) is not int or value < 0 for value in gpu["capability"])
        or type(gpu["total_memory_bytes"]) is not int or gpu["total_memory_bytes"] <= 0
    ):
        raise RuntimeError(f"seed {seed} GPU runtime inventory differs")
    scheduler = runtime["scheduler"]
    if not isinstance(scheduler, dict) or set(scheduler) != {
        "job_id", "array_job_id", "array_task_id", "partition", "constraint", "node_list"
    } or (
        scheduler["job_id"] != runtime["slurm_job_id"]
        or scheduler["array_job_id"] != runtime["slurm_array_job_id"]
        or scheduler["array_task_id"] != runtime["slurm_array_task_id"]
    ):
        raise RuntimeError(f"seed {seed} scheduler runtime inventory differs")
    if (
        not isinstance(runtime["slurm_job_id"], str) or not runtime["slurm_job_id"].isdecimal()
        or runtime["scheduler"]["partition"] != "gpu"
        or runtime["scheduler"]["constraint"] not in (None, "a6000")
        or "A6000" not in gpu["name"].upper()
        or not isinstance(runtime["scheduler"]["node_list"], str)
        or not runtime["scheduler"]["node_list"]
    ):
        raise RuntimeError(f"seed {seed} Slurm runtime binding differs")
    if seed == 9173:
        if runtime["slurm_array_job_id"] is not None or runtime["slurm_array_task_id"] is not None:
            raise RuntimeError("smoke runtime unexpectedly reports a Slurm array binding")
    elif (
        not isinstance(runtime["slurm_array_job_id"], str)
        or not runtime["slurm_array_job_id"].isdecimal()
        or runtime["slurm_array_task_id"] != str(seed)
    ):
        raise RuntimeError(f"seed {seed} Slurm array runtime binding differs")


def parse_manifest_text(text: str) -> dict[str, str]:
    """Parse embedded manifest text with the same strict rules as a file."""

    entries = {}
    for line in text.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise RuntimeError("malformed embedded source manifest")
        expected, relative = parts
        relative = relative.lstrip(" *")
        if relative in entries:
            raise RuntimeError(f"duplicate embedded manifest path: {relative}")
        entries[relative] = expected
    if not entries:
        raise RuntimeError("embedded source manifest is empty")
    return entries


def mean_std(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"count": 0, "mean": None, "sample_std": None, "minimum": None, "maximum": None}
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def bootstrap_mean_interval(
    values: list[float], *, seed: int = 20260902, draws: int = 100_000
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(draws, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "draws": draws,
        "seed": seed,
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_root = require_physical_root(Path(__file__).resolve().parents[1])
    expected_results = reject_symlinks_in_directory(
        project_root, project_root / "results", label="results directory"
    )
    logs_path = project_root / "logs"
    if logs_path.exists() or logs_path.is_symlink():
        reject_symlinks_in_directory(project_root, logs_path, label="logs directory")
    results_argument = require_confined_path(
        project_root, args.results, label="results directory", kind="directory"
    )
    if results_argument != expected_results:
        raise RuntimeError(f"results directory must be exactly {expected_results}")
    output = require_confined_path(
        project_root, args.output, label="summary output", allow_missing_leaf=True
    )
    if output != expected_results / "summary.json":
        raise RuntimeError(f"summary output must be exactly {expected_results / 'summary.json'}")
    manifest_path = project_root / "source_manifest.sha256"
    frozen_entries = verify_source_freeze(project_root, manifest_path)
    manifest_text = manifest_path.read_text()
    manifest_sha256 = digest(manifest_path)
    launch_path = require_confined_path(
        project_root,
        expected_results / "launch.json",
        label="launch record",
        kind="file",
    )
    launch = json.loads(launch_path.read_text())
    launch_jobs = validate_launch_binding(
        launch,
        project_root=project_root,
        manifest_sha256=manifest_sha256,
        current_summary_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    if (
        os.environ.get("SLURM_JOB_PARTITION") != "compute"
        or os.environ.get("SLURM_ARRAY_JOB_ID") is not None
        or os.environ.get("SLURM_ARRAY_TASK_ID") is not None
    ):
        raise RuntimeError("summary did not run as the launch DAG Athena compute job")
    certificate_path = require_confined_path(
        project_root,
        expected_results / "canonical_odt_unit_certificate.json",
        label="unit-test certificate",
        kind="file",
    )
    certificate = json.loads(
        certificate_path.read_text()
    )
    test_log = require_confined_path(
        project_root,
        Path(certificate.get("test_log_path", "")),
        label="unit-test log",
        kind="file",
    )
    if (
        certificate.get("schema") != UNIT_CERTIFICATE_SCHEMA
        or certificate.get("tests_passed") is not True
        or certificate.get("source_manifest_sha256") != manifest_sha256
        or certificate.get("source_entries") != frozen_entries
        or certificate.get("slurm_job_id") != launch_jobs["tests"]
        or test_log.parent != expected_results
        or certificate.get("test_log_sha256") != digest(test_log)
    ):
        raise RuntimeError("unit-test certificate does not match the frozen source")
    prefetch_path = require_confined_path(
        project_root,
        expected_results / "prefetch.json",
        label="prefetch receipt",
        kind="file",
    )
    prefetch = json.loads(prefetch_path.read_text())
    if prefetch != {
        "schema": PREFETCH_SCHEMA,
        "status": "complete",
        "source_manifest_sha256": manifest_sha256,
        "dataset_root": launch["data_root"],
        "dataset_files": EXPECTED_DATASET_FILES,
        "slurm_job_id": launch_jobs["prefetch"],
    }:
        raise RuntimeError("prefetch receipt is not bound to the launch DAG")
    smoke_validation_path = require_confined_path(
        project_root,
        expected_results / "smoke_validation.json",
        label="fresh-process smoke-validation receipt",
        kind="file",
    )
    smoke_validation = json.loads(smoke_validation_path.read_text())

    expected_seed_paths = exact_seed_result_paths(results_argument)
    paths = [
        require_confined_path(
            project_root, path, label="seed result", kind="file"
        )
        for path in expected_seed_paths
    ]
    records = [json.loads(path.read_text()) for path in paths]
    require_seed_file_binding(records, paths)
    if len(records) != 3:
        raise RuntimeError(f"expected exactly three seed JSON files, found {len(records)}")
    if any(record.get("schema") != SCHEMA for record in records):
        raise RuntimeError("result schema mismatch")
    if any(record.get("status") not in {"complete", "gate_failed"} for record in records):
        raise RuntimeError("seed result status is neither complete nor gate_failed")
    seeds = [record["seed"] for record in records]
    if seeds != [0, 1, 2]:
        raise RuntimeError(f"expected seeds 0, 1, 2 exactly, found {seeds}")
    protocol_comparison_keys = EXPECTED_PROTOCOL_KEYS - {
        "gauge_required_ranks_by_bond", "smoke_nontrivial_diagnostic_rank"
    }
    if any(
        any(
            not _typed_equal(record["protocol"][key], records[0]["protocol"][key])
            for key in protocol_comparison_keys
        )
        for record in records[1:]
    ):
        raise RuntimeError("frozen protocol mismatch across seeds")
    if any(record["claim_boundary"] != records[0]["claim_boundary"] for record in records[1:]):
        raise RuntimeError("claim boundary mismatch across seeds")
    if records[0]["protocol"]["discovery_count"] != 4000:
        raise RuntimeError("unexpected discovery count")
    if records[0]["protocol"]["confirmation_count"] != 22032:
        raise RuntimeError("unexpected confirmation count")
    if records[0]["protocol"].get("minimum_full_model_accuracy") != 0.70:
        raise RuntimeError("unexpected trained-model capability floor")
    if records[0]["protocol"].get("gauge_trials") != 50:
        raise RuntimeError("full runs must execute exactly 50 learned-checkpoint gauge trials")
    for record in records:
        status = record["status"]
        if status == "complete":
            require_exact_true_gates(record, "algebraic_gates", EXPECTED_ALGEBRAIC_GATES)
            require_exact_true_gates(record, "capability_gates", EXPECTED_CAPABILITY_GATES)
        validate_protocol_and_measurements(
            record,
            smoke=False,
            expected_status=status,
            require_all_gates=status == "complete",
        )
        validate_seed_provenance(
            record,
            project_root=project_root,
            manifest_path=manifest_path,
            manifest_text=manifest_text,
            manifest_sha256=manifest_sha256,
            frozen_entries=frozen_entries,
        )
    dataset_root = Path(launch["data_root"]).resolve()
    official_dataset_files = rehash_official_svhn(dataset_root)
    expected_replay_sample = regenerate_gauge_replay_sample(dataset_root)
    seed_gauge_validations = [
        validate_gauge_matrix_artifact(
            record,
            project_root=project_root,
            expected_path=expected_results / f"seed_{record['seed']}_gauge_matrices.npz",
            expected_raw_network=export_checkpoint_homogeneous(
                record, project_root=project_root
            ),
            expected_replay_sample=expected_replay_sample,
            require_all_matrix_gates=record["status"] == "complete",
            include_authentication_details=True,
        )
        for record in records
    ]
    seed_gauge_paths = [item[0] for item in seed_gauge_validations]
    seed_gauge_authentication = [item[1] for item in seed_gauge_validations]
    smoke_path = project_root / "results" / "smoke.json"
    smoke_path = require_confined_path(
        project_root,
        smoke_path,
        label="trained-checkpoint gauge preflight result",
        kind="file",
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
        raise RuntimeError("trained-checkpoint gauge preflight protocol did not pass")
    require_exact_true_gates(smoke, "algebraic_gates", EXPECTED_ALGEBRAIC_GATES)
    require_exact_true_gates(smoke, "capability_gates", EXPECTED_CAPABILITY_GATES)
    validate_protocol_and_measurements(smoke, smoke=True)
    validate_seed_provenance(
        smoke,
        project_root=project_root,
        manifest_path=manifest_path,
        manifest_text=manifest_text,
        manifest_sha256=manifest_sha256,
        frozen_entries=frozen_entries,
    )
    smoke_gauge_path, smoke_gauge_authentication = validate_gauge_matrix_artifact(
        smoke,
        project_root=project_root,
        expected_path=expected_results / "smoke_gauge_matrices.npz",
        expected_raw_network=export_checkpoint_homogeneous(
            smoke, project_root=project_root
        ),
        expected_replay_sample=expected_replay_sample,
        include_authentication_details=True,
    )
    validate_smoke_validation_receipt(
        smoke_validation,
        source_manifest_sha256=manifest_sha256,
        jobs=launch_jobs,
        launch_path=launch_path,
        certificate_path=certificate_path,
        test_log=test_log,
        prefetch_path=prefetch_path,
        smoke_path=smoke_path,
        gauge_path=smoke_gauge_path,
        dataset_files=official_dataset_files,
        authentication=smoke_gauge_authentication,
    )
    if smoke["provenance"]["runtime"].get("slurm_job_id") != launch_jobs["smoke"]:
        raise RuntimeError("smoke result is not bound to the launch DAG smoke job")
    if (
        smoke["provenance"]["runtime"].get("slurm_array_job_id") is not None
        or smoke["provenance"]["runtime"].get("slurm_array_task_id") is not None
    ):
        raise RuntimeError("smoke result unexpectedly has an array task id")
    pre_summary_inventory = validate_pre_summary_result_inventory(
        expected_results, test_log
    )
    for record in records:
        runtime = record["provenance"]["runtime"]
        if (
            runtime.get("slurm_array_job_id") != launch_jobs["train_array"]
            or runtime.get("slurm_array_task_id") != str(record["seed"])
            or not isinstance(runtime.get("slurm_job_id"), str)
            or not runtime["slurm_job_id"].isdecimal()
        ):
            raise RuntimeError(
                f"seed {record['seed']} is not bound to its launch DAG array task"
            )
    dataset_manifests = [record["provenance"]["dataset_files"] for record in records]
    if any(manifest != dataset_manifests[0] for manifest in dataset_manifests[1:]):
        raise RuntimeError("dataset manifest mismatch across seeds")
    dataset_roots = {record["provenance"]["dataset_root"] for record in records}
    if len(dataset_roots) != 1:
        raise RuntimeError(f"dataset root mismatch across seeds: {dataset_roots}")
    if Path(next(iter(dataset_roots))).resolve() != Path(launch["data_root"]):
        raise RuntimeError("seed dataset root is not bound to the launch DAG")
    if Path(next(iter(dataset_roots))).resolve() != dataset_root:
        raise RuntimeError("validated dataset root changed during aggregation")
    frozen_manifests = [record["provenance"]["frozen_source_manifest"] for record in records]
    split_keys = ("split_seed", "discovery_indices_sha256", "confirmation_indices_sha256")
    for key in split_keys:
        values = {record["protocol"][key] for record in records}
        if len(values) != 1:
            raise RuntimeError(f"test split mismatch across seeds for {key}: {values}")
    failed_seed_inventory = [
        {
            "seed": record["seed"],
            "false_algebraic_gates": sorted(
                key for key, value in record["algebraic_gates"].items() if value is False
            ),
            "false_capability_gates": sorted(
                key for key, value in record["capability_gates"].items() if value is False
            ),
        }
        for record in records
        if record["status"] == "gate_failed"
    ]
    if failed_seed_inventory:
        raise RuntimeError(
            "authenticated seed gate failures: "
            + json.dumps(failed_seed_inventory, sort_keys=True, separators=(",", ":"))
        )

    methods = sorted(records[0]["discovery_selected_confirmation"])
    if set(methods) != EXPECTED_METHODS:
        raise RuntimeError(f"unexpected method set: {methods}")
    for record in records:
        if set(record["discovery_selected_confirmation"]) != EXPECTED_METHODS:
            raise RuntimeError("method set mismatch across seeds")
        if set(record.get("curves", {})) != EXPECTED_METHODS:
            raise RuntimeError("curve method set mismatch across seeds")
        if len(record["haar_curves"]) != record["protocol"]["haar_draws"]:
            raise RuntimeError("incomplete Haar curves")
        if len(record["haar_discovery_selected_confirmation"]) != record["protocol"][
            "haar_draws"
        ]:
            raise RuntimeError("incomplete discovery-selected Haar endpoints")
    method_summary = {}
    for method in methods:
        ranks = [record["discovery_selected_confirmation"][method]["rank"] for record in records]
        effective_ranks = [
            record["discovery_selected_confirmation"][method]["effective_affine_rank"]
            for record in records
        ]
        confirmation = [
            record["discovery_selected_confirmation"][method]["confirmation_accuracy"]
            for record in records
        ]
        drops = [
            record["discovery_selected_confirmation"][method][
                "confirmation_drop_from_module"
            ]
            for record in records
        ]
        method_summary[method] = {
            "selected_rank": mean_std(ranks),
            "selected_effective_affine_rank": mean_std(effective_ranks),
            "confirmation_accuracy": mean_std(confirmation),
            "exploratory_three_seed_bootstrap_95": bootstrap_mean_interval(
                confirmation, seed=20260902 + methods.index(method)
            ),
            "confirmation_drop_from_module": mean_std(drops),
            "per_seed": [
                {
                    "seed": record["seed"],
                    **record["discovery_selected_confirmation"][method],
                }
                for record in records
            ],
        }

    reconstruction_max = max(
        item["max_absolute_logit_error"]
        for record in records
        for item in record["canonicalization"]["sampled_logit_reconstruction"].values()
    )
    factorization_max = max(
        max(record["canonicalization"]["factorization_errors"])
        for record in records
    )
    row_isometry_max = max(
        max(record["canonicalization"]["isometry_errors"])
        for record in records
    )
    gauge_raw_replay_max = max(
        record["gauge_audit"]["max_raw_coordinate_replay_relative"]
        for record in records
    )
    gauge_canonicalized_function_max = max(
        record["gauge_audit"]["max_canonicalized_function_relative"]
        for record in records
    )
    gauge_single_bond_function_max = max(
        record["gauge_audit"]["max_sampled_single_bond_logit_replay_relative"]
        for record in records
    )
    gauge_singular_extrema_max = max(
        record["gauge_audit"]["max_singular_extrema_relative_error"]
        for record in records
    )
    gauge_condition_max = max(
        record["gauge_audit"]["max_condition_relative_error"]
        for record in records
    )
    gauge_nonsymmetry_min = min(
        record["gauge_audit"]["minimum_nonsymmetry_relative"]
        for record in records
    )
    independent_conditioning_maxima = {
        key: max(
            authentication["non_load_bearing_conditioning_diagnostics"]
            ["independent_checkpoint_npz_cpu_maxima"][key]
            for authentication in seed_gauge_authentication
        )
        for key in (
            "raw_coordinate_replay_relative",
            "canonicalized_function_relative",
        )
    }
    gauge_spectrum_max = max(
        record["gauge_audit"]["max_spectrum_relative"] for record in records
    )
    gauge_covariance_max = max(
        record["gauge_audit"]["max_gram_covariance_relative"] for record in records
    )
    gauge_transport_max = max(
        record["gauge_audit"]["max_transport_orthogonality"] for record in records
    )
    gauge_coefficient_replay_max = max(
        record["gauge_audit"]["max_canonicalized_coefficient_replay_relative"]
        for record in records
    )
    gauge_projector_coverage_complete = all(
        record["gauge_audit"].get("projector_certificate_coverage_complete") is True
        for record in records
    ) and all(
        audit["required_certificate_count"]
        == audit["expected_required_certificate_count"]
        for audit in seed_gauge_authentication
    )
    gauge_projector_overlap_support_max = max(
        audit["worst_of_all_copies"]["maximum_supported_overlap_defect"]
        for audit in seed_gauge_authentication
    )
    gauge_projector_bound_max = max(
        audit["worst_of_all_copies"]["maximum_bound_with_roundoff"]
        for audit in seed_gauge_authentication
    )
    gauge_projectors_certifiable = all(
        copy["all_numerically_certifiable"] is True
        for audit in seed_gauge_authentication
        for copy in audit["copies"].values()
    )
    gauge_projectors_pass_bound = all(
        copy["all_pass_davis_kahan_inequality"] is True
        and copy["all_bounds_le_maximum"] is True
        for audit in seed_gauge_authentication
        for copy in audit["copies"].values()
    )
    source_manifest = {
        relative: frozen_entries[relative] for relative in RESULT_SOURCE_PATHS
    }
    all_bond_validations = [validate_all_bond_curve(record) for record in records]

    paired = {
        "canonical_minus_mean_haar_at_canonical_selected_rank": [],
        "canonical_minus_local_at_canonical_selected_rank": [],
        "canonical_minus_legacy_at_matched_affine_rank": [],
        "canonical_minus_pca_at_matched_affine_rank": [],
    }
    for record in records:
        canonical_endpoint = record["discovery_selected_confirmation"]["canonical_tree_odt"]
        rank = canonical_endpoint["rank"]
        canonical_accuracy = canonical_endpoint["confirmation_accuracy"]
        haar_accuracies = [
            curve["confirmation"][str(rank)] for curve in record["haar_curves"].values()
        ]
        paired["canonical_minus_mean_haar_at_canonical_selected_rank"].append(
            canonical_accuracy - float(np.mean(haar_accuracies))
        )
        paired["canonical_minus_local_at_canonical_selected_rank"].append(
            canonical_accuracy
            - record["curves"]["canonical_adjacent_local"]["confirmation"][str(rank)]
        )
        learned_rank = rank - 1
        for method, key in (
            ("legacy_raw_downstream_zero_anchor", "canonical_minus_legacy_at_matched_affine_rank"),
            ("activation_pca_centered", "canonical_minus_pca_at_matched_affine_rank"),
        ):
            curve = record["curves"][method]["confirmation"]
            if str(learned_rank) not in curve:
                raise RuntimeError(
                    f"seed {record['seed']} lacks the rank-{learned_rank} affine baseline "
                    f"needed to match canonical rank {rank}"
                )
            paired[key].append(canonical_accuracy - float(curve[str(learned_rank)]))
    paired_summary = {
        name: {
            "per_seed": values,
            "summary": mean_std(values),
        }
        for name, values in paired.items()
    }

    gates = {
        "three_complete_seeds": True,
        "identical_source_hashes": True,
        "identical_dataset_hashes": True,
        "sampled_logit_reconstruction_le_1e-8": reconstruction_max <= 1e-8,
        "factorization_relative_le_1e-10": factorization_max <= 1e-10,
        "row_isometry_le_1e-10": row_isometry_max <= 1e-10,
        "gauge_sampled_single_bond_logit_replay_relative_le_1e-8": (
            gauge_single_bond_function_max <= 1e-8
        ),
        "gauge_singular_extrema_relative_le_1e-12": (
            gauge_singular_extrema_max <= 1e-12
        ),
        "gauge_condition_relative_le_1e-12": gauge_condition_max <= 1e-12,
        "gauge_nonsymmetry_relative_gt_1e-6": gauge_nonsymmetry_min > 1e-6,
        "gauge_spectrum_relative_le_1e-8": gauge_spectrum_max <= 1e-8,
        "gauge_coefficient_replay_relative_le_1e-8": gauge_coefficient_replay_max <= 1e-8,
        "gauge_projector_certificate_coverage_complete": gauge_projector_coverage_complete,
        "gauge_gram_covariance_relative_le_1e-8": gauge_covariance_max <= 1e-8,
        "gauge_transport_orthogonality_le_1e-10": gauge_transport_max <= 1e-10,
        "gauge_required_projector_overlap_support_defect_le_1e-8": (
            gauge_projector_overlap_support_max <= 1e-8
        ),
        "gauge_required_projectors_numerically_certifiable": gauge_projectors_certifiable,
        "gauge_required_projectors_pass_nonvacuous_bound": (
            gauge_projectors_pass_bound
            and gauge_projector_bound_max <= GAUGE_CERTIFICATE_MAX_BOUND
        ),
        "selected_canonical_eigengap_resolved": all(
            record["discovery_selected_confirmation"]["canonical_tree_odt"][
                "eigengap_resolved"
            ] is True
            for record in records
        ),
        "selected_canonical_roundoff_only_prefilter_pass": all(
            record["canonical_rank_selection"]["prefilter_by_rank"][
                str(record["canonical_rank_selection"]["selected_rank"])
            ]["passes"]
            is True
            for record in records
        ),
        "all_seed_gate_inventories_exact_and_true": all(
            set(record["algebraic_gates"]) == EXPECTED_ALGEBRAIC_GATES
            and set(record["capability_gates"]) == EXPECTED_CAPABILITY_GATES
            and all(value is True for value in record["algebraic_gates"].values())
            and all(value is True for value in record["capability_gates"].values())
            for record in records + [smoke]
        ),
        "all_seed_manifests_match_current_freeze": all(
            record["provenance"]["frozen_source_manifest"]["sha256"] == manifest_sha256
            and record["provenance"]["source_sha256"] == {
                path: frozen_entries[path] for path in RESULT_SOURCE_PATHS
            }
            for record in records + [smoke]
        ),
        "official_svhn_hashes_match": all(
            record["provenance"]["dataset_files"] == EXPECTED_DATASET_FILES
            for record in records + [smoke]
        ),
        "checkpoint_files_rehashed": all(
            digest(project_root / "results" / ("smoke.pt" if record is smoke else f"seed_{record['seed']}.pt"))
            == record["provenance"]["checkpoint"]["sha256"]
            for record in records + [smoke]
        ),
        "gauge_npz_artifacts_independently_recomputed": (
            len(seed_gauge_paths) == 3 and smoke_gauge_path.is_file()
        ),
        "identical_environment_fingerprints": len(
            {
                record["provenance"]["runtime"]["fingerprint_sha256"]
                for record in records + [smoke]
            }
        )
        == 1,
        "module_accuracy_floor_passed": all(
            record["references"]["module"][split]
            >= record["protocol"]["minimum_full_model_accuracy"]
            for record in records + [smoke]
            for split in ("discovery", "confirmation")
        ),
        "trained_checkpoint_50_gauge_preflight_passed": (
            smoke["protocol"]["epochs"] == 20
            and smoke["protocol"]["gauge_trials"] == 50
            and smoke["status"] == "complete"
        ),
        "all_bond_uniform_schedule_exact": all(
            item["uniform_schedule_exact"] is True for item in all_bond_validations
        ),
        "all_bond_trace_totals_relative_le_1e-8": all(
            item["trace_disagreement"] <= 1e-8 for item in all_bond_validations
        ),
        "all_bond_full_rank_coefficient_reconstruction_le_1e-8": all(
            item["full_rank_relative_coefficient_error"] <= 1e-8
            for item in all_bond_validations
        ),
        "all_bond_selected_canonical_eigengaps_resolved": all(
            item["selected_all_canonical_boundaries_resolved"] is True
            for item in all_bond_validations
        ),
        "all_bond_selected_canonical_roundoff_only_prefilter_pass": all(
            item["selected_all_canonical_roundoff_prefilter_pass"] is True
            for item in all_bond_validations
        ),
        "all_bond_coefficient_errors_within_hsvd_bounds_plus_declared_numerical_slack": all(
            item[
                "all_coefficient_errors_within_bounds_plus_declared_numerical_slack"
            ] is True
            for item in all_bond_validations
        ),
    }
    empirical_gates = {
        "all_bond_selected_confirmation_noninferior_in_all_seeds": all(
            item["selected_confirmation_noninferior"] is True
            for item in all_bond_validations
        ),
        "all_bond_selected_odt_gt_local_with_resolved_boundaries_in_all_seeds": all(
            item["selected_odt_gt_local_with_resolved_boundaries"] is True
            for item in all_bond_validations
        ),
        "all_bond_selected_haar_randomization_p_le_0p05_in_all_seeds": all(
            item["selected_haar_randomization_p_le_0p05"] is True
            for item in all_bond_validations
        ),
    }
    if set(gates) != EXPECTED_SUMMARY_GATES:
        raise RuntimeError("internal summary gate inventory mismatch")
    if set(empirical_gates) != EXPECTED_EMPIRICAL_GATES:
        raise RuntimeError("internal empirical gate inventory mismatch")
    artifact_manifest = artifact_hash_manifest(
        launch_path=launch_path,
        certificate_path=certificate_path,
        test_log=test_log,
        prefetch_path=prefetch_path,
        smoke_validation_path=smoke_validation_path,
        smoke_path=smoke_path,
        smoke_gauge_path=smoke_gauge_path,
        records=records,
        seed_paths=paths,
        seed_gauge_paths=seed_gauge_paths,
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "algebraic_gate_status": (
            "pass" if all(value is True for value in gates.values()) else "fail"
        ),
        "empirical_comparison_status": "pass" if all(empirical_gates.values()) else "fail",
        "claim_boundary": records[0]["claim_boundary"],
        "seeds": sorted(seeds),
        "gates": gates,
        "empirical_gates": empirical_gates,
        "gate_maxima": {
            "sampled_logit_absolute": reconstruction_max,
            "factorization_relative": factorization_max,
            "row_isometry": row_isometry_max,
            "gauge_sampled_single_bond_logit_replay_relative": gauge_single_bond_function_max,
            "gauge_singular_extrema_relative": gauge_singular_extrema_max,
            "gauge_condition_relative": gauge_condition_max,
            "gauge_minimum_nonsymmetry_relative": gauge_nonsymmetry_min,
            "gauge_spectrum_relative": gauge_spectrum_max,
            "gauge_gram_covariance_relative": gauge_covariance_max,
            "gauge_transport_orthogonality": gauge_transport_max,
            "gauge_canonicalized_coefficient_replay_relative": gauge_coefficient_replay_max,
            "gauge_required_projector_overlap_support_defect": (
                gauge_projector_overlap_support_max
            ),
            "gauge_required_projector_bound_with_roundoff": gauge_projector_bound_max,
        },
        "numerical_conditioning_diagnostics": {
            "comparison_policy": "report-both-cross-device-paths-without-a-pass-threshold-v1",
            "producer_maxima": {
                "raw_coordinate_replay_relative": gauge_raw_replay_max,
                "canonicalized_function_relative": gauge_canonicalized_function_max,
            },
            "independent_checkpoint_npz_cpu_maxima": independent_conditioning_maxima,
            "worst_of_both_maxima": {
                "raw_coordinate_replay_relative": max(
                    gauge_raw_replay_max,
                    independent_conditioning_maxima["raw_coordinate_replay_relative"],
                ),
                "canonicalized_function_relative": max(
                    gauge_canonicalized_function_max,
                    independent_conditioning_maxima["canonicalized_function_relative"],
                ),
            },
        },
        "methods": method_summary,
        "paired_comparisons": paired_summary,
        "all_bond_selected_by_seed": {
            str(record["seed"]): validation["selected"]
            for record, validation in zip(records, all_bond_validations)
        },
        "canonical_rank_selection_by_seed": {
            str(record["seed"]): record["canonical_rank_selection"]
            for record in records
        },
        "nontrivial_single_bond_selected_seed_count": sum(
            record["canonical_rank_selection"]["outcome"]
            == "nontrivial_rank_selected"
            for record in records
        ),
        "nontrivial_all_bond_selected_seed_count": sum(
            record["simultaneous_uniform_all_bond_curve"]["discovery_selection"][
                "nontrivial_compression_selected"
            ]
            is True
            for record in records
        ),
        "predecessor_failures": EXPECTED_PREDECESSOR_FAILURES,
        "source_sha256": source_manifest,
        "frozen_source_manifest": frozen_manifests[0],
        "dataset_files": dataset_manifests[0],
        "checkpoint_sha256_by_seed": {
            str(record["seed"]): record["provenance"]["checkpoint"]["sha256"]
            for record in records
        },
        "gauge_npz_independent_recomputation": {
            "smoke": True,
            "seed_by_seed": {str(record["seed"]): True for record in records},
        },
        "required_projector_three_way_authentication": {
            "smoke": smoke_gauge_authentication,
            "seed_by_seed": {
                str(record["seed"]): authentication
                for record, authentication in zip(
                    records, seed_gauge_authentication
                )
            },
        },
        "gauge_artifact_authentication_policy": {
            "raw_gauge_stream_whole_tensor_relative_tolerance": (
                RAW_GAUGE_STREAM_RELATIVE_TOLERANCE
            ),
            "independent_matrix_whole_tensor_relative_tolerance": (
                INDEPENDENT_MATRIX_RECONSTRUCTION_RELATIVE_TOLERANCE
            ),
            "required_projector_certificates": (
                "strict-producer-to-stored-npz-plus-independent-gate-replication-v1"
            ),
            "required_path_dependent_residual_magnitude_equality": False,
            "nonrequired_resolved_rank_diagnostics": (
                "non-load-bearing-self-consistent-gap-conditioned-v1"
            ),
            "nonrequired_primitive_absolute_tolerance": (
                NONREQUIRED_DIAGNOSTIC_PRIMITIVE_ABSOLUTE_TOLERANCE
            ),
        },
        "artifact_sha256": artifact_manifest,
        "pre_summary_result_inventory": pre_summary_inventory,
        "gpu_by_seed": {
            str(record["seed"]): record["provenance"]["runtime"]["gpu"]
            for record in records
        },
        "environment_fingerprint_sha256": records[0]["provenance"]["runtime"][
            "fingerprint_sha256"
        ],
        "trained_checkpoint_preflight": {
            "seed": smoke["seed"],
            "checkpoint_sha256": smoke["provenance"]["checkpoint"]["sha256"],
            "gauge_trials": smoke["protocol"]["gauge_trials"],
            "discovery_accuracy": smoke["references"]["module"]["discovery"],
            "confirmation_accuracy": smoke["references"]["module"]["confirmation"],
            "fresh_process_validation_receipt_sha256": digest(smoke_validation_path),
            "fresh_process_required_authentication": smoke_validation[
                "required_projector_three_way_authentication"
            ],
            "terminal_process_required_authentication": smoke_gauge_authentication,
            "cross_process_conditioning_diagnostics": {
                "comparison_policy": (
                    "report-each-independent-cpu-copy-and-worst-without-scalar-equality-v1"
                ),
                "validator_cpu_maxima": smoke_validation[
                    "required_projector_three_way_authentication"
                ]["non_load_bearing_conditioning_diagnostics"][
                    "independent_checkpoint_npz_cpu_maxima"
                ],
                "terminal_summary_cpu_maxima": smoke_gauge_authentication[
                    "non_load_bearing_conditioning_diagnostics"
                ]["independent_checkpoint_npz_cpu_maxima"],
                "worst_across_producer_and_cpu_copies": {
                    key: max(
                        smoke_gauge_authentication[
                            "non_load_bearing_conditioning_diagnostics"
                        ]["producer_maxima"][key],
                        smoke_validation[
                            "required_projector_three_way_authentication"
                        ]["non_load_bearing_conditioning_diagnostics"][
                            "independent_checkpoint_npz_cpu_maxima"
                        ][key],
                        smoke_gauge_authentication[
                            "non_load_bearing_conditioning_diagnostics"
                        ]["independent_checkpoint_npz_cpu_maxima"][key],
                    )
                    for key in (
                        "raw_coordinate_replay_relative",
                        "canonicalized_function_relative",
                    )
                },
            },
        },
        "launch_binding": {
            "schema": LAUNCH_SCHEMA,
            "launch_record_sha256": digest(launch_path),
            "job_ids": launch_jobs,
            "terminal_summary_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    if verify_source_freeze(project_root, manifest_path) != frozen_entries:
        raise RuntimeError("source freeze changed during summary")
    reject_symlinks_in_directory(
        project_root, expected_results, label="results directory"
    )
    if validate_pre_summary_result_inventory(expected_results, test_log) != pre_summary_inventory:
        raise RuntimeError("pre-summary results artifact inventory changed during summary")
    if artifact_hash_manifest(
        launch_path=launch_path,
        certificate_path=certificate_path,
        test_log=test_log,
        prefetch_path=prefetch_path,
        smoke_validation_path=smoke_validation_path,
        smoke_path=smoke_path,
        smoke_gauge_path=smoke_gauge_path,
        records=records,
        seed_paths=paths,
        seed_gauge_paths=seed_gauge_paths,
    ) != artifact_manifest:
        raise RuntimeError("load-bearing result artifact bytes changed during summary")
    if logs_path.exists():
        reject_symlinks_in_directory(project_root, logs_path, label="logs directory")
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def authenticated_seed_failures(
    project_root: Path, manifest_sha256: str | None
) -> dict:
    """Authenticate every materialized seed before reporting exact failures."""

    results = project_root / "results"
    present = sorted(results.glob("seed_*.json")) if results.is_dir() else []
    if not present:
        return {
            "validated_seed_count": 0,
            "passing_seed_count": 0,
            "failed_seed_count": 0,
            "status_by_seed": {},
            "failures": [],
        }
    if manifest_sha256 is None:
        raise RuntimeError("seed results exist without a source-manifest identity")
    manifest_path = project_root / "source_manifest.sha256"
    manifest_text = manifest_path.read_text()
    frozen_entries = verify_source_freeze(project_root, manifest_path)
    if digest(manifest_path) != manifest_sha256:
        raise RuntimeError("source manifest changed while authenticating seed failures")
    launch_path = require_confined_path(
        project_root,
        results / "launch.json",
        label="launch record for seed failures",
        kind="file",
    )
    launch = json.loads(launch_path.read_text())
    jobs = validate_launch_binding(
        launch,
        project_root=project_root,
        manifest_sha256=manifest_sha256,
        current_summary_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    paths = exact_seed_result_paths(results)
    records = [json.loads(path.read_text()) for path in paths]
    require_seed_file_binding(records, paths)
    if [record.get("seed") for record in records] != [0, 1, 2]:
        raise RuntimeError("seed failure inventory is not exactly seeds 0, 1, 2")

    dataset_root = Path(launch["data_root"]).resolve()
    rehash_official_svhn(dataset_root)
    expected_replay_sample = regenerate_gauge_replay_sample(dataset_root)
    failures = []
    for record, path in zip(records, paths):
        status = record.get("status")
        if record.get("schema") != SCHEMA or status not in {"complete", "gate_failed"}:
            raise RuntimeError("seed failure schema or status differs")
        algebraic, capability = validate_protocol_and_measurements(
            record,
            smoke=False,
            expected_status=status,
            require_all_gates=status == "complete",
        )
        validate_seed_provenance(
            record,
            project_root=project_root,
            manifest_path=manifest_path,
            manifest_text=manifest_text,
            manifest_sha256=manifest_sha256,
            frozen_entries=frozen_entries,
        )
        provenance = record["provenance"]
        if provenance["dataset_root"] != str(dataset_root):
            raise RuntimeError(f"seed {record['seed']} dataset root differs from launch")
        runtime = provenance["runtime"]
        if runtime["slurm_array_job_id"] != jobs["train_array"]:
            raise RuntimeError(f"seed {record['seed']} array job differs from launch")
        checkpoint_network = export_checkpoint_homogeneous(record, project_root=project_root)
        gauge_path, gauge_authentication = validate_gauge_matrix_artifact(
            record,
            project_root=project_root,
            expected_path=results / f"seed_{record['seed']}_gauge_matrices.npz",
            expected_raw_network=checkpoint_network,
            expected_replay_sample=expected_replay_sample,
            require_all_matrix_gates=status == "complete",
            include_authentication_details=True,
        )
        if status == "gate_failed":
            false_algebraic = sorted(key for key, value in algebraic.items() if value is False)
            false_capability = sorted(key for key, value in capability.items() if value is False)
            if not false_algebraic and not false_capability:
                raise RuntimeError("authenticated gate-failed seed has no false gate")
            failures.append({
                "seed": record["seed"],
                "stage": "train_array",
                "status": status,
                "json": {"path": str(path), "sha256": digest(path)},
                "checkpoint": {
                    "path": provenance["checkpoint"]["path"],
                    "sha256": provenance["checkpoint"]["sha256"],
                },
                "gauge_npz": {"path": str(gauge_path), "sha256": digest(gauge_path)},
                "required_projector_three_way_authentication": gauge_authentication,
                "source_manifest_sha256": manifest_sha256,
                "slurm_array_job_id": runtime["slurm_array_job_id"],
                "slurm_array_task_id": runtime["slurm_array_task_id"],
                "slurm_job_id": runtime["slurm_job_id"],
                "canonical_rank_selection": record["canonical_rank_selection"],
                "all_bond_rank_selection": record[
                    "simultaneous_uniform_all_bond_curve"
                ]["discovery_selection"],
                "false_algebraic_gates": false_algebraic,
                "false_capability_gates": false_capability,
            })
    status_by_seed = {str(record["seed"]): record["status"] for record in records}
    return {
        "validated_seed_count": len(records),
        "passing_seed_count": sum(status == "complete" for status in status_by_seed.values()),
        "failed_seed_count": len(failures),
        "status_by_seed": status_by_seed,
        "failures": failures,
    }


def authenticated_smoke_failure(
    project_root: Path, manifest_sha256: str | None
) -> dict | None:
    """Return the bound smoke root cause without trusting unrelated JSON."""

    smoke_path = project_root / "results" / "smoke.json"
    if (
        manifest_sha256 is None
        or not smoke_path.is_file()
        or smoke_path.is_symlink()
    ):
        return None
    manifest_path = project_root / "source_manifest.sha256"
    manifest_text = manifest_path.read_text()
    frozen_entries = verify_source_freeze(project_root, manifest_path)
    if digest(manifest_path) != manifest_sha256:
        raise RuntimeError("source manifest changed while authenticating smoke failure")
    launch_path = require_confined_path(
        project_root,
        project_root / "results" / "launch.json",
        label="launch record for smoke failure",
        kind="file",
    )
    launch = json.loads(launch_path.read_text())
    jobs = validate_launch_binding(
        launch,
        project_root=project_root,
        manifest_sha256=manifest_sha256,
        current_summary_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    smoke = json.loads(smoke_path.read_text())
    smoke_manifest = smoke.get("provenance", {}).get(
        "frozen_source_manifest", {}
    )
    algebraic = smoke.get("algebraic_gates", {})
    capability = smoke.get("capability_gates", {})
    if (
        smoke.get("schema") != SCHEMA
        or smoke.get("status") not in {"complete", "gate_failed"}
        or smoke.get("seed") != 9173
        or smoke_manifest.get("sha256") != manifest_sha256
        or not isinstance(algebraic, dict)
        or not isinstance(capability, dict)
        or set(algebraic) != EXPECTED_ALGEBRAIC_GATES
        or set(capability) != EXPECTED_CAPABILITY_GATES
        or any(type(value) is not bool for value in (*algebraic.values(), *capability.values()))
        or smoke.get("protocol", {}).get("smoke_nontrivial_diagnostic_rank") != 1
        or smoke.get("provenance", {}).get("runtime", {}).get("slurm_job_id")
        != jobs["smoke"]
    ):
        raise RuntimeError("smoke failure schema, gates, seed, protocol, or job binding differs")
    validate_seed_provenance(
        smoke,
        project_root=project_root,
        manifest_path=manifest_path,
        manifest_text=manifest_text,
        manifest_sha256=manifest_sha256,
        frozen_entries=frozen_entries,
    )
    dataset_root = smoke.get("provenance", {}).get("dataset_root")
    if dataset_root != launch["data_root"]:
        raise RuntimeError("smoke dataset root differs from the authenticated launch")
    rehash_official_svhn(Path(dataset_root))
    if smoke["status"] == "complete":
        validate_protocol_and_measurements(smoke, smoke=True)
        validate_gauge_matrix_artifact(
            smoke,
            project_root=project_root,
            expected_path=project_root / "results" / "smoke_gauge_matrices.npz",
            expected_raw_network=export_checkpoint_homogeneous(
                smoke, project_root=project_root
            ),
            expected_replay_sample=regenerate_gauge_replay_sample(
                Path(dataset_root)
            ),
            include_authentication_details=True,
        )
        return None
    if all((*algebraic.values(), *capability.values())):
        raise RuntimeError("gate-failed smoke contains no reported false gate")
    computed_algebraic, computed_capability = validate_protocol_and_measurements(
        smoke,
        smoke=True,
        expected_status="gate_failed",
        require_all_gates=False,
    )
    _, gauge_authentication = validate_gauge_matrix_artifact(
        smoke,
        project_root=project_root,
        expected_path=project_root / "results" / "smoke_gauge_matrices.npz",
        expected_raw_network=export_checkpoint_homogeneous(
            smoke, project_root=project_root
        ),
        expected_replay_sample=regenerate_gauge_replay_sample(Path(dataset_root)),
        require_all_matrix_gates=False,
        include_authentication_details=True,
    )
    return {
        "stage": "smoke",
        "path": str(smoke_path),
        "sha256": digest(smoke_path),
        "status": smoke.get("status"),
        "false_algebraic_gates": sorted(
            key for key, passed in computed_algebraic.items() if passed is False
        ),
        "false_capability_gates": sorted(
            key for key, passed in computed_capability.items() if passed is False
        ),
        "required_projector_three_way_authentication": gauge_authentication,
    }


def write_summary_failure_receipt(
    project_root: Path, error: Exception, *, slurm_job_id: str | None
) -> None:
    """Write a terminal receipt even when upstream authentication itself fails."""

    failure_path = project_root / "results" / "summary_failure.json"
    if failure_path.exists() or failure_path.is_symlink():
        return
    manifest_path = project_root / "source_manifest.sha256"
    manifest_sha256 = digest(manifest_path) if manifest_path.is_file() else None
    try:
        if manifest_sha256 is None:
            raise RuntimeError("source manifest is absent")
        launch_path = require_confined_path(
            project_root, project_root / "results/launch.json",
            label="launch record for smoke validation", kind="file",
        )
        launch = json.loads(launch_path.read_text())
        jobs = validate_launch_binding(
            launch, project_root=project_root, manifest_sha256=manifest_sha256,
            current_summary_job_id=slurm_job_id,
        )
        validation_path = project_root / "results/smoke_validation.json"
        if not validation_path.exists():
            smoke_validation_outcome = {
                "stage": "smoke_validate", "status": "receipt_absent",
                "expected_slurm_job_id": jobs["smoke_validate"],
            }
        else:
            require_confined_path(
                project_root, validation_path,
                label="smoke-validation failure-path receipt", kind="file",
            )
            certificate_path = project_root / "results/canonical_odt_unit_certificate.json"
            certificate = json.loads(certificate_path.read_text())
            test_log = require_confined_path(
                project_root, Path(certificate["test_log_path"]),
                label="smoke-validation unit log", kind="file",
            )
            prefetch_path = require_confined_path(
                project_root, project_root / "results/prefetch.json",
                label="smoke-validation prefetch", kind="file",
            )
            smoke_path = require_confined_path(
                project_root, project_root / "results/smoke.json",
                label="smoke-validation smoke JSON", kind="file",
            )
            gauge_path = require_confined_path(
                project_root, project_root / "results/smoke_gauge_matrices.npz",
                label="smoke-validation gauge NPZ", kind="file",
            )
            dataset_files = rehash_official_svhn(Path(launch["data_root"]))
            validate_smoke_validation_receipt(
                json.loads(validation_path.read_text()),
                source_manifest_sha256=manifest_sha256,
                jobs=jobs,
                launch_path=launch_path,
                certificate_path=certificate_path,
                test_log=test_log,
                prefetch_path=prefetch_path,
                smoke_path=smoke_path,
                gauge_path=gauge_path,
                dataset_files=dataset_files,
            )
            smoke_validation_outcome = {
                "stage": "smoke_validate", "status": "authenticated_complete",
                "path": str(validation_path), "sha256": digest(validation_path),
                "slurm_job_id": jobs["smoke_validate"],
            }
        smoke_validation_authentication_error = None
    except Exception as authentication_error:
        smoke_validation_outcome = None
        smoke_validation_authentication_error = {
            "error_type": type(authentication_error).__name__,
            "error": str(authentication_error),
        }
    try:
        upstream_failure = authenticated_smoke_failure(
            project_root, manifest_sha256
        )
        upstream_authentication_error = None
    except Exception as authentication_error:
        upstream_failure = None
        upstream_authentication_error = {
            "error_type": type(authentication_error).__name__,
            "error": str(authentication_error),
        }
    try:
        seed_failures = authenticated_seed_failures(project_root, manifest_sha256)
        seed_authentication_error = None
    except Exception as authentication_error:
        seed_failures = None
        seed_authentication_error = {
            "error_type": type(authentication_error).__name__,
            "error": str(authentication_error),
        }
    failure_path.write_text(
        json.dumps(
            {
                "schema": SUMMARY_SCHEMA,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "source_manifest_sha256": manifest_sha256,
                "authenticated_upstream_failure": upstream_failure,
                "upstream_authentication_error": upstream_authentication_error,
                "smoke_validation_outcome": smoke_validation_outcome,
                "smoke_validation_authentication_error": (
                    smoke_validation_authentication_error
                ),
                "authenticated_seed_failures": seed_failures,
                "seed_authentication_error": seed_authentication_error,
                "slurm_job_id": slurm_job_id,
            },
            indent=2,
        )
    )


def main() -> None:
    """Aggregate, or leave an Athena terminal failure receipt under afterany."""

    try:
        _main()
    except Exception as error:
        project_root = Path(__file__).resolve().parents[1]
        try:
            write_summary_failure_receipt(
                project_root, error, slurm_job_id=os.environ.get("SLURM_JOB_ID")
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
