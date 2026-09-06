#!/usr/bin/env python3
"""Authenticate and run exact direct canonical ODT on the trained Product VLA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import secrets
import stat
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.dont_write_bytecode = True

_RUNNING_AS_ENTRYPOINT = __name__ == "__main__"
_MANIFEST_RELATIVE = "athena/product_rational_exact_odt_sources.json"
_MANIFEST_SCHEMA = "xvla_product_pade_rational_exact_odt_v1_source_manifest"
_EXPECTED_SOURCE_CLOSURE = (
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
_EXPECTED_AUTHENTICATED_INPUTS = (
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
_EXPECTED_LAUNCH_CLOSURE = (
    "athena/slurm_product_rational_odt_stage.sbatch",
    "athena/slurm_product_rational_odt_composite.sbatch",
    "athena/slurm_product_rational_odt_full.sbatch",
    "athena/slurm_product_rational_odt_validate.sbatch",
    "athena/submit_product_rational_exact_odt.sh",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_type_exact_equal(observed: Any, expected: Any) -> bool:
    """Compare authenticated JSON values without Python's bool/int aliases."""
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _json_type_exact_equal(observed[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _json_type_exact_equal(left, right)
            for left, right in zip(observed, expected)
        )
    if isinstance(expected, float):
        return math.isfinite(observed) and math.isfinite(expected) and observed == expected
    return observed == expected


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant is forbidden: {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Nonfinite JSON float is forbidden")
    return parsed


def _read_json_physical_once(path: Path, *, label: str) -> tuple[dict[str, Any], str, tuple[int, ...]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or nonphysical: {path}")
    before_path = os.lstat(path)
    if not stat.S_ISREG(before_path.st_mode):
        raise RuntimeError(f"{label} is not a regular file")
    with path.open("rb") as handle:
        before_fd = os.fstat(handle.fileno())
        raw = handle.read()
        after_fd = os.fstat(handle.fileno())
    identity = (
        before_fd.st_dev,
        before_fd.st_ino,
        before_fd.st_size,
        before_fd.st_mtime_ns,
        before_fd.st_ctime_ns,
    )
    if identity != (
        after_fd.st_dev,
        after_fd.st_ino,
        after_fd.st_size,
        after_fd.st_mtime_ns,
        after_fd.st_ctime_ns,
    ) or (before_path.st_dev, before_path.st_ino, before_path.st_size) != identity[:3]:
        raise RuntimeError(f"{label} changed while it was read")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not strict finite JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return payload, hashlib.sha256(raw).hexdigest(), identity


def _verify_json_snapshot(path: Path, *, label: str, digest: str, identity: tuple[int, ...]) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} disappeared or became nonphysical")
    observed = os.lstat(path)
    current_identity = (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )
    if current_identity != identity or file_sha256(path) != digest:
        raise RuntimeError(f"{label} changed after authenticated read")


def _safe_member(relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or not value.parts or any(part in {"", ".", ".."} for part in value.parts):
        raise RuntimeError(f"Unsafe manifest member {relative!r}")
    cursor = PROJECT_ROOT
    for part in value.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(f"Manifest member traverses a link: {relative}")
    resolved = cursor.resolve(strict=True)
    if resolved == PROJECT_ROOT or PROJECT_ROOT not in resolved.parents or not resolved.is_file():
        raise RuntimeError(f"Manifest member escapes or is missing: {relative}")
    return resolved


def _single_preimport_value(arguments: list[str], option: str) -> str:
    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument == option:
            if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
                raise RuntimeError(f"{option} requires one value")
            values.append(arguments[index + 1])
        elif argument.startswith(f"{option}="):
            value = argument.split("=", 1)[1]
            if not value:
                raise RuntimeError(f"{option} requires one value")
            values.append(value)
        elif argument.startswith(option[:-1]):
            raise RuntimeError(f"Abbreviated {option} options are forbidden")
    if len(values) != 1:
        raise RuntimeError(f"Exactly one {option} is required")
    return values[0]


def _verify_preimport_manifest(arguments: list[str]) -> dict[str, Any]:
    supplied = Path(_single_preimport_value(arguments, "--source-manifest"))
    raw = supplied if supplied.is_absolute() else PROJECT_ROOT / supplied
    expected = PROJECT_ROOT / _MANIFEST_RELATIVE
    if raw != expected or raw.is_symlink() or raw.resolve(strict=True) != expected:
        raise RuntimeError("Exact-ODT source manifest path differs from the frozen path")
    payload = json.loads(
        raw.read_text(),
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_json_constant,
        parse_float=_finite_json_float,
    )
    if not isinstance(payload, dict) or payload.get("schema") != _MANIFEST_SCHEMA:
        raise RuntimeError("Exact-ODT source manifest schema differs")
    sections = (
        ("source_closure", _EXPECTED_SOURCE_CLOSURE, "source_bundle_sha256"),
        (
            "authenticated_inputs",
            _EXPECTED_AUTHENTICATED_INPUTS,
            "authenticated_inputs_bundle_sha256",
        ),
        ("launch_closure", _EXPECTED_LAUNCH_CLOSURE, "launch_bundle_sha256"),
    )
    observed_sections: dict[str, dict[str, str]] = {}
    for section, paths, bundle_key in sections:
        stored = payload.get(section)
        if not isinstance(stored, dict) or tuple(sorted(stored)) != tuple(sorted(paths)):
            raise RuntimeError(f"Exact-ODT source manifest {section} differs")
        if payload.get(bundle_key) != _canonical_sha256(dict(sorted(stored.items()))):
            raise RuntimeError(f"Exact-ODT source manifest {bundle_key} differs")
        observed: dict[str, str] = {}
        for relative, digest in stored.items():
            if (
                not isinstance(relative, str)
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise RuntimeError(f"Exact-ODT source manifest {section} entry is malformed")
            path = _safe_member(relative)
            actual = _file_sha256(path)
            if actual != digest:
                raise RuntimeError(f"Authenticated Exact-ODT file changed: {relative}")
            observed[relative] = actual
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
    if payload.get("manifest_bundle_sha256") != _canonical_sha256(bound):
        raise RuntimeError("Exact-ODT aggregate manifest digest differs")
    return {
        "path": raw.resolve().as_posix(),
        "sha256": _file_sha256(raw),
        "source_sha256": observed_sections["source_closure"],
        "authenticated_inputs_sha256": observed_sections["authenticated_inputs"],
        "launch_sha256": observed_sections["launch_closure"],
        "manifest_bundle_sha256": payload["manifest_bundle_sha256"],
    }


_PREIMPORT_SOURCE_VERIFICATION = (
    _verify_preimport_manifest(sys.argv[1:]) if _RUNNING_AS_ENTRYPOINT else None
)

if _RUNNING_AS_ENTRYPOINT:
    from scripts.odt_direct_only_compliance import (
        DIRECT_RQ_RUNTIME_CALL,
        EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT,
        EXPECTED_RUNTIME_GUARD_ENTRYPOINTS,
        PRODUCTION_ALGORITHM3_RUNTIME_CALL,
        PRODUCTION_CLONE_QR_RUNTIME_CALL,
        PRODUCTION_SHARED_CLONE_ALGORITHM3_RUNTIME_CALL,
        STREAMED_CLONE_ORACLE_RUNTIME_CALLS,
        STREAMED_QR_RUNTIME_CALL,
        TRIANGULAR_RUNTIME_CALL,
        assert_direct_only_runtime_guard,
        audit_direct_only_launch,
        direct_only_runtime_report,
        install_direct_only_runtime_guard,
    )

    _TRANSITIVE_STATIC_AUDIT = audit_direct_only_launch(
        PROJECT_ROOT,
        tuple(PROJECT_ROOT / path for path in _EXPECTED_SOURCE_CLOSURE),
    )
    if _TRANSITIVE_STATIC_AUDIT["source_sha256"] != _PREIMPORT_SOURCE_VERIFICATION[
        "source_sha256"
    ]:
        raise RuntimeError("Authenticated source tuple differs from the exact static audit")
    for _field in (
        "prohibited_calls_found",
        "prohibited_self_overlap_sites",
        "guarded_dormant_spectral_norm_sites",
        "duplicate_top_level_definition_sites",
    ):
        if _TRANSITIVE_STATIC_AUDIT[_field] != []:
            raise RuntimeError(f"Exact-ODT static audit did not close {_field}")
    _RUNTIME_GUARD_AT_IMPORT = install_direct_only_runtime_guard()
else:
    from scripts.odt_direct_only_compliance import (
        DIRECT_RQ_RUNTIME_CALL,
        EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT,
        EXPECTED_RUNTIME_GUARD_ENTRYPOINTS,
        PRODUCTION_ALGORITHM3_RUNTIME_CALL,
        PRODUCTION_CLONE_QR_RUNTIME_CALL,
        PRODUCTION_SHARED_CLONE_ALGORITHM3_RUNTIME_CALL,
        STREAMED_CLONE_ORACLE_RUNTIME_CALLS,
        STREAMED_QR_RUNTIME_CALL,
        TRIANGULAR_RUNTIME_CALL,
        assert_direct_only_runtime_guard,
        audit_direct_only_launch,
        direct_only_runtime_report,
        install_direct_only_runtime_guard,
    )

    _TRANSITIVE_STATIC_AUDIT = None
    _RUNTIME_GUARD_AT_IMPORT = None

import numpy as np
import torch
import torch.nn as nn

from athena.aggregate_product_rational_results import (
    validate_capability_aggregate_evidence,
    verify_capability_evidence_snapshots,
)
from athena.product_rational_exact_odt_protocol import (
    AUTHENTICATED_INPUTS,
    CAPABILITY_RUN_ROOT,
    CAPABILITY_STAGE_ROOT,
    CLONE_RELATIVE_TOLERANCE,
    DOOMS_REFERENCE_SHA256,
    MANIFEST_PATH,
    OFFDIAGONAL_RELATIVE_TOLERANCE,
    ODT_SOURCE_OVERRIDES,
    PHYSICAL_MATERIALIZATION_TARGET,
    PRESTAGE_CLOSURE,
    PRESTAGE_SCHEMA,
    REPLAY_RELATIVE_TOLERANCE,
    SCHEMA,
    SOURCE_CLOSURE,
    LAUNCH_CLOSURE,
    SYNTHETIC_REPLAY_SEED,
    SYNTHETIC_REPLAY_TASKS,
    capability_aggregate_path,
    capability_gate_path,
    clone_oracle_junit_path,
    clone_oracle_result_path,
    composite_result_path,
    file_sha256,
    full_rank_progress_path,
    full_result_path,
    input_path,
    lane_test_junit_path,
    lane_test_result_path,
    preflight_gate_path,
    publish_json_exclusive,
    read_json_object,
    validate_gate_output,
    validate_progress_output,
    validate_result_output,
    verify_manifest,
)
from athena.product_rational_protocol import (
    AUTHENTICATED_INPUTS as CAPABILITY_AUTHENTICATED_INPUTS,
    CALIBRATION_INITIALIZER_LABEL,
    CALIBRATION_RECOVERY_SCOPE,
    CALIBRATION_RESUME_PROOF_SCHEMA,
    DEPLOYMENT_COMPLETION_SCHEMA,
    DEPLOYMENT_CLAIM_BOUNDARY,
    EXPECTED_PADE_SITE_COUNT,
    ACTION_DIM,
    FINAL_CALIBRATION_AUDIT_MAX_RELATIVE_CHANGE,
    FINAL_CALIBRATION_PASSES,
    OFFICIAL_TASK_LANGUAGES,
    OFFICIAL_TASK_LANGUAGES_SHA256,
    PRIMARY_CAPABILITY_FLOOR,
    RECIPE_VERSION,
    N_FACTORS,
    SCHEMA as CAPABILITY_SCHEMA,
    SMOKE_STEPS,
    SOURCE_CLOSURE as CAPABILITY_SOURCE_CLOSURE,
    LAUNCH_CLOSURE as CAPABILITY_LAUNCH_CLOSURE,
    VOCAB_SHA256,
    active_running_ms_values,
    build_vocab,
    config_record,
    load_training_only_precalibration_state,
    make_product_config,
    source_bundle_sha256,
    strict_load_deployment_state,
    training_recipe,
    validate_precalibration_to_deployment_transition,
    validate_preflight_certificate,
    validate_product_model,
    validate_same_allocation_calibration_resume_proof,
    validate_simultaneous_calibration_attestation,
)
from xvla.models.vla import ChiVLA
from xvla.train.direct_odt_truncation import (
    CompactRankBank,
    evaluate_diagonal_prefix_quotient,
    evaluate_diagonal_prefixes,
    evaluate_diagonal_suffix_quotient,
    implicit_storage_elements,
    projected_prefix_storage_elements,
    truncate_diagonal_prefixes,
)
from xvla.train.implicit_sparse_projective_odt import (
    COMPACT_RELATIVE_VALUE_FLOORS,
    COMPACT_TRACE_RETENTION_TARGETS,
    DIRECT_Q_PROVENANCE_TOLERANCE,
    DIRECT_RQ_METHOD,
    LOCAL_SCALE_CERTIFICATION_TOLERANCE,
    RECTANGULAR_RQ_METHOD,
    STREAMED_DIRECT_Q_PROVENANCE_METHOD,
    UNARY_RQ_METHOD,
    _projective_batch_relative_error,
    audit_algorithm1_factorization_calls,
    canonicalize_implicit_dag_direct_rq,
    diagonalize_implicit_dag_full_rank,
    evaluate_boundary_quotient,
    evaluate_projective_boundary,
    implicit_shape_statistics,
    predict_direct_rq_route_inventory,
    telemetry_dict,
    validate_canonical_exponent_normal_form,
)
from xvla.train.implicit_sparse_projective_odt_vla import (
    PRODUCT_COMPONENTS_OBSERVABLE,
    PRODUCT_COMPONENTS_FULL_VLA_CLAIM_BOUNDARY,
    assert_full_vla_norm_buffers_unchanged,
    compile_full_vla_projective_dag,
    evaluate_full_vla_observable,
    evaluate_full_vla_quotient,
    full_vla_structure_statistics,
    physical_batch_from_model_inputs,
    physical_source_mapping,
    source_full_vla_observable,
    source_full_vla_output,
)


EXPECTED_CLONE_TEST_NAMES = frozenset(
    {
        "test_reference_source_is_independent_and_stage_restricted",
        "test_unchanged_six_pade_attention_ffn_residual_block_matches_every_step",
        "test_heterogeneous_final_norm_pooling_and_product_head_matches_every_step",
        "test_rank_deficient_shared_child_matches_and_missing_occurrences_fail",
        "test_seeded_adversarial_shared_dag_family[0]",
        "test_seeded_adversarial_shared_dag_family[1]",
        "test_seeded_adversarial_shared_dag_family[2]",
        "test_seeded_adversarial_shared_dag_family[3]",
        "test_forced_streamed_shared_cp_matches_independent_clone_after_every_step",
        "test_bounded_rank_deficient_385x386_retains_direct_q_and_matches_clone_every_step",
        "test_rank_zero_direct_rq_reconstruction_and_homogeneous_replay",
        "test_over_bound_large_deficient_unfolding_fails_closed",
        "test_streamed_production_algorithms2_and3_match_separated_clone_reference",
        "test_shared_zero_route_environment_uses_only_nonzero_scale_center",
        "test_no_memo_expansion_fails_closed_at_occurrence_limit",
        "test_two_nonzero_shared_route_exponent_gap_fails_closed",
    }
)

EXPECTED_LANE_TEST_NAMES = frozenset(
    {
        "test_exact_protocol_and_runner_duplicate_the_same_complete_tuples",
        "test_exact_python_and_launch_closure_are_clean_and_complete",
        "test_reference_identity_and_orientation_record_are_fixed",
        "test_tiny_product_all_components_runs_direct_algorithms_and_compression",
        "test_projective_pair_gate_ignores_only_nonzero_rowwise_scale",
        "test_orientation_and_occurrence_mutations_fail_their_gates",
        "test_full_rank_consumer_reconstructs_every_algorithm_and_runtime_counter",
        "test_compression_consumer_reconstructs_curves_materialization_and_authorities",
        "test_zero_product_gate_decodes_to_negative_sign",
        "test_cli_authority_rejects_duplicates_and_abbreviations",
        "test_exact_output_publication_rejects_broken_link",
        "test_launch_shells_parse_and_stage_has_no_dynamic_global_rebinding",
        "test_delayed_stage_and_full_are_bound_to_prestage_and_lane_test_authorities",
        "test_strict_load_float64_conversion_revalidates_guard_before_real_forward",
        "test_exact_lane_modules_have_no_duplicate_top_level_definitions",
        "test_compact_callback_is_complete_and_cannot_retain_full_records",
        "test_compact_summary_keeps_a_tie_cluster_and_rejects_material_negativity",
        "test_every_prefix_step_matches_the_no_memo_clone_and_masked_replay",
        "test_omitting_one_shared_occurrence_is_rejected_before_replay",
        "test_plans_and_certificate_gates_fail_closed",
        "test_wide_dependent_regression_compresses_without_a_pivot_gate",
        "test_product_component_root_census_covers_center_factors_and_gates",
        "test_materialization_projective_comparison_ignores_rowwise_radial_scale",
        "test_patch_round_trip_matches_conv2d_order_exactly",
        "test_heterogeneous_source_round_trip_uses_unchanged_chivla_forward",
        "test_categorical_source_oracle_fails_closed_outside_one_hot_domain",
        "test_layout_rejects_a_changed_deployed_topology",
        "test_singleton_embodiment_is_a_checkpoint_constant_not_a_physical_leaf",
        "test_tiny_full_policy_is_one_exact_shared_dag_with_heterogeneous_leaves",
        "test_product_head_component_root_replays_every_expert_gate_and_public_decode",
        "test_product_head_fixed_external_sign_route_is_one_action_root",
        "test_product_head_fixed_route_rejects_non_sign_coordinates",
        "test_minimal_product_shared_dag_runs_direct_algorithms1_to3_with_step_replay",
        "test_product_head_subgraph_matches_no_memo_clone_at_every_canonical_step",
    }
)

DOOMS_ORIENTATION = {
    "reference_algorithm1_order": "child-before-parent postorder",
    "reference_algorithm1_core_update": "replace each local unfolding by direct reduced-RQ Q",
    "reference_algorithm1_factor_update": "push R into every connected parent-input occurrence or the root head",
    "reference_algorithm2_order": "root-first downstream environment contraction from the output head",
    "reference_algorithm3_output_update": "apply each ordered eigenbasis transpose on the bond output",
    "reference_algorithm3_input_update": "apply the same eigenbasis on every connected parent-input occurrence or the root head",
}

_STATIC_AUDIT_KEYS = {
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
_RUNTIME_GUARD_KEYS = {
    "installed",
    "patched_entrypoints",
    "patched_entrypoint_count",
    "allowed_call_count",
    "allowed_calls",
    "prohibited_attempt_count",
    "prohibited_attempts",
}
_SHAPE_KEYS = {
    "unique_nodes",
    "edge_occurrences",
    "cp_binary_nodes",
    "reduced_q_binary_nodes",
    "dense_clone_nodes",
    "unary_nodes",
    "heterogeneous_physical_sources",
    "maximum_local_bond_dimension",
    "maximum_cp_rank",
    "maximum_reduced_q_elements",
    "fused_token_feature_dimension",
}
_STRUCTURE_KEYS = {
    "unique_nodes",
    "edge_occurrences",
    "physical_source_count",
    "physical_leaf_count",
    "physical_source_widths",
    "maximum_physical_source_width",
    "root_projective_width",
    "observable_kind",
    "observable_euclidean_width",
    "product_signs",
    "active_pade_sites",
}
_ROUTE_KEYS = {
    "schema",
    "bounded_explicit_unfolding_element_limit",
    "bounded_explicit_q_element_limit",
    "bounded_retained_q_candidate_count",
    "bounded_retained_q_total_elements",
    "bounded_retained_q_replaced_cp_total_elements",
    "bounded_retained_q_signed_storage_delta_elements",
    "bounded_retained_q_positive_storage_delta_elements",
    "bounded_retained_q_maximum_unfolding_elements",
    "bounded_retained_q_maximum_elements",
    "streamed_cp_candidate_count",
    "streamed_tall_unrepresentable_count",
    "streamed_tall_unrepresentable_shape_counts",
    "bounded_retained_q_shape_counts",
    "nodes_with_propagated_input_shape_change",
    "predicted_root_output_dimension",
    "simulated_unique_node_count",
}
_NORMAL_FORM_KEYS = {
    "core_count",
    "all_core_exponents_zero",
    "head_binary_exponent",
    "head_exponent_matches_ledger",
    "cp_direct_q_certificates",
    "cp_direct_q_columns",
    "streamed_compact_cp_direct_q_certificates",
    "streamed_compact_cp_direct_q_columns",
    "all_streamed_compact_cp_q_has_direct_q_provenance",
}
_ALGORITHM1_KEYS = {
    "step_count",
    "factorization_methods",
    "factorization_method_counts",
    "canonical_postorder_uid_sha256",
    "algorithm1_step_uid_sha256",
    "child_parent_order_violation_count",
    "final_projective_replay_relative_error",
    "full_projective_replay_evaluations",
    "per_step_replays_performed",
    "maximum_local_scaled_reconstruction_relative_error",
    "maximum_absorption_scaled_relative_error",
    "maximum_local_reconstruction_exponent_delta",
    "maximum_absorption_exponent_delta",
    "push_ledger",
    "scale_sensitive_occurrence_ledger",
    "canonical_exponent_normal_form",
    "direct_q_provenance",
    "telemetry",
}
_DIRECT_Q_PROVENANCE_KEYS = {
    "verified_steps",
    "expected_steps",
    "streamed_certificates",
    "streamed_steps",
    "columns_compared",
    "expected_columns",
    "streamed_replay_calls",
    "expected_streamed_replay_calls",
    "maximum_compact_q_relative_error",
    "maximum_reconstruction_relative_error",
}
_TELEMETRY_KEYS = {
    "raw_order_three_core_materializations",
    "raw_width_cubic_core_materializations",
    "maximum_persistent_tensor_elements",
    "maximum_temporary_tensor_elements",
    "maximum_temporary_tensor_order",
    "local_direct_rq_factorizations",
    "householder_qr_kernel_calls",
    "svd_calls",
    "polar_calls",
    "normal_equation_factorizations",
    "fused_ffn_primitive_count",
    "ephemeral_ffn_bond_diagonalizations",
    "rectangular_direct_rq_factorizations",
    "maximum_rectangular_q_elements",
    "bounded_explicit_direct_rq_factorizations",
    "maximum_bounded_explicit_q_elements",
    "constant_unary_folds",
    "constant_cp_to_unary_folds",
    "constant_cp_to_constant_folds",
    "direct_q_provenance_certificates",
    "streamed_direct_q_provenance_certificates",
    "streamed_direct_q_replay_qr_factorizations",
    "direct_q_columns_compared",
    "maximum_direct_q_transition_elements",
    "maximum_direct_q_panel_elements",
    "maximum_direct_q_compact_relative_error",
    "maximum_direct_q_reconstruction_relative_error",
}
_SOURCE_GATE_KEYS = {
    "all_components_product_root",
    "compiled_component_replay",
    "compiled_public_action_replay",
    "source_component_decode_matches_public_forward",
    "all_physical_sources_present_once",
    "heterogeneous_ingress_unpadded",
    "all_pade_sites_compiled",
    "no_dense_clone_core",
    "route_inventory_exact_before_sweep",
}
_ALGORITHM1_GATE_KEYS = {
    "all_nodes_use_direct_reduced_rq",
    "final_projective_replay",
    "production_replay_schedule",
    "local_scale_certificates",
    "canonical_exponent_normal_form",
    "direct_q_provenance_complete",
    "streamed_direct_q_ledger",
    "direct_q_telemetry_complete",
    "every_factor_occurrence_pushed",
    "rank_deficiency_route_is_shape_only",
    "node_inventory_preserved",
}
_DOOMS_GATE_KEYS = {
    "algorithm1_child_before_parent",
    "algorithm1_q_replaces_every_local_core",
    "algorithm1_r_reaches_every_occurrence",
    "algorithm2_root_first_contraction_complete",
    "algorithm3_basis_applied_to_every_bond",
    "algorithm3_basis_reaches_every_occurrence",
}
_ALGORITHM23_GATE_KEYS = {
    "algorithm2_every_edge_message",
    "algorithm3_every_node",
    "algorithm3_every_occurrence",
    "algorithm3_replay",
    "algorithm3_recontracted_environments_diagonal",
    "zero_full_spectrum_retention",
    "compact_rank_bank_complete",
    "source_model_state_unchanged",
    "runtime_guard_exact",
}
_FULL_RANK_GATE_KEYS = (
    _SOURCE_GATE_KEYS | _ALGORITHM1_GATE_KEYS | _DOOMS_GATE_KEYS | _ALGORITHM23_GATE_KEYS
)
_FULL_RANK_TOP_KEYS = {
    "schema",
    "checkpoint_sha256",
    "metadata_sha256",
    "training_result_sha256",
    "smoke_calibration_resume_proof_sha256",
    "preflight",
    "bounded_clone_oracle",
    "exact_lane_tests",
    "dooms_reference",
    "replay_panel",
    "source_replay",
    "structure",
    "initial_shape",
    "canonical_shape",
    "route_inventory",
    "algorithm1",
    "algorithm2_and3",
    "numerical_execution",
    "timings_seconds",
    "peak_rss_mb",
    "full_rank_gates",
    "all_full_rank_gates_pass",
    "runtime_guard",
    "static_audit",
    "manifest_sha256",
    "manifest_bundle_sha256",
    "bounded_clone_oracle_hash_joined",
    "exact_lane_test_hash_joined",
    "canonical_direct_odt_algorithms_1_to_3_completed",
    "exact_global_decomposability_certified",
    "compression_claimed",
    "capability_claimed",
    "capable_global_decomposable_vla_claimed",
}
_FULL_RESULT_TOP_KEYS = {
    "schema",
    "checkpoint_sha256",
    "smoke_calibration_resume_proof_sha256",
    "full_rank_certificate",
    "bounded_clone_oracle",
    "exact_lane_tests",
    "preflight",
    "replay_panel",
    "full_rank_gates",
    "all_full_rank_gates_pass",
    "compact_census",
    "canonical_storage_elements",
    "compression_curves",
    "compression_curves_are_functional_not_task_degradation",
    "physical_materialization",
    "physical_materialization_gates",
    "all_compression_gates_pass",
    "timings_seconds",
    "peak_rss_mb",
    "runtime_guard_final",
    "static_audit",
    "manifest_sha256",
    "manifest_bundle_sha256",
    "canonical_direct_odt_algorithms_1_to_3_completed",
    "bounded_clone_oracle_hash_joined",
    "exact_lane_test_hash_joined",
    "exact_global_decomposability_certified",
    "compression_claimed_after_full_rank_certificate",
    "capability_claimed",
    "capable_global_decomposable_vla_claimed",
}
_ALGORITHM23_KEYS = {
    "explicit_downstream_contraction",
    "streamed_root_first",
    "child_messages",
    "eigenbasis_order",
    "diagonalized_nodes",
    "occurrence_push_ledger",
    "replay_relative_error",
    "maximum_recontracted_offdiagonal_ratio",
    "full_spectra_retained",
    "post_environment_records_retained",
}
_REPLAY_PANEL_KEYS = {
    "kind",
    "seed",
    "official_task_indices",
    "instruction_token_sha256",
    "image_sha256",
    "state_sha256",
    "embodiment",
    "batch_size",
    "panel_sha256",
}
_SOURCE_REPLAY_KEYS = {
    "component_relative_error",
    "public_action_relative_error",
    "source_decode_relative_error",
    "source_decode_bitwise_equal",
    "source_positive_sign_fraction",
    "source_zero_gate_count",
}
_COMPONENT_METRIC_KEYS = {
    "component_relative_error",
    "decoded_action_relative_error",
    "decoded_action_max_absolute_error",
    "decoded_action_coordinate_max_absolute_error",
    "gate_sign_flip_count",
    "gate_sign_flip_fraction",
}
_MATERIALIZATION_KEYS = {
    "target",
    "performed_after_all_masked_curves",
    "copy_network",
    "executor_kind",
    "materialized_tensor_network_executed",
    "source_model_forward_fallback_forbidden",
    "source_model_forward_calls_during_materialized_execution",
    "projective_pair_relative_error_against_masked",
    "projective_pair_gauge_invariant_relative_error_against_masked",
    "component_relative_error_against_masked",
    "decoded_action_relative_error_against_masked",
    "decoded_action_coordinate_max_absolute_error_against_masked",
    "gate_sign_mismatch_count_against_masked",
    "original_storage_elements",
    "projected_storage_elements",
    "materialized_storage_elements",
    "allocated_storage_elements",
    "reduced_bonds",
    "expected_occurrence_slices",
    "applied_occurrence_slices",
    "elapsed_seconds",
}
_MATERIALIZATION_GATE_KEYS = {
    "materialized_executor_has_no_source_model_forward_fallback",
    "projective_pair_matches_masked_up_to_rowwise_scale",
    "components_match_masked",
    "actions_match_masked",
    "signs_match_masked",
    "storage_origin_matches",
    "projected_materialized_allocated_equal",
    "strict_physical_storage_reduction",
    "bond_reduction_nonempty",
    "every_occurrence_sliced",
}


def _option_count(arguments: list[str], option: str) -> int:
    return sum(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--mode",
        choices=("preflight", "oracle", "lane-tests", "full", "composite"),
        required=True,
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    arguments = sys.argv[1:]
    parsed = parser.parse_args(arguments)
    for option in ("--mode", "--source-manifest"):
        if _option_count(arguments, option) != 1:
            parser.error(f"{option} must occur exactly once")
    expected_manifest = PROJECT_ROOT / MANIFEST_PATH
    supplied = (
        parsed.source_manifest
        if parsed.source_manifest.is_absolute()
        else PROJECT_ROOT / parsed.source_manifest
    )
    if supplied.resolve(strict=True) != expected_manifest.resolve(strict=True):
        parser.error("--source-manifest differs from the frozen exact-ODT manifest")
    parsed.source_manifest = supplied
    return parsed


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    first = actual.reshape(-1)
    second = expected.reshape(-1)
    numerator = torch.sqrt(torch.sum((first - second) * (first - second)))
    scale = torch.sqrt(torch.sum(second * second)).clamp_min(torch.finfo(second.dtype).tiny)
    return float((numerator / scale).item())


def _state_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _runtime_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "executable": Path(sys.executable).resolve().as_posix(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "device": "cpu",
        "dtype": "torch.float64",
    }


def _assert_runtime_versions() -> None:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Exact Product ODT requires frozen Python 3.10")
    if np.__version__ != "1.26.4":
        raise RuntimeError("Exact Product ODT requires frozen NumPy 1.26.4")
    if torch.__version__.split("+", 1)[0] != "2.7.1":
        raise RuntimeError("Exact Product ODT requires frozen Torch 2.7.1")


def _assert_sources_unchanged() -> dict[str, Any]:
    manifest = verify_manifest(PROJECT_ROOT / MANIFEST_PATH)
    current = audit_direct_only_launch(
        PROJECT_ROOT,
        tuple(PROJECT_ROOT / path for path in SOURCE_CLOSURE),
    )
    if current != _TRANSITIVE_STATIC_AUDIT:
        raise RuntimeError("Exact-ODT static source audit changed during execution")
    if current["source_sha256"] != manifest["source_closure"]:
        raise RuntimeError("Exact-ODT manifest and static source map differ")
    if file_sha256(PROJECT_ROOT / MANIFEST_PATH) != _PREIMPORT_SOURCE_VERIFICATION["sha256"]:
        raise RuntimeError("Exact-ODT manifest bytes changed during execution")
    return manifest


def _stored_static_closed(value: Any, sources: Mapping[str, str]) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == _STATIC_AUDIT_KEYS
        and value.get("scope") == "transitive_local_import_closure"
        and _json_type_exact_equal(value.get("entrypoints"), sorted(sources))
        and _json_type_exact_equal(value.get("source_sha256"), dict(sources))
        and type(value.get("source_count")) is int
        and value.get("source_count") == len(sources)
        and type(value.get("direct_qr_call_sites")) is int
        and value.get("direct_qr_call_sites") == 0
        and value.get("direct_qr_required") is False
        and _json_type_exact_equal(value.get("prohibited_calls_found"), [])
        and _json_type_exact_equal(value.get("prohibited_self_overlap_sites"), [])
        and _json_type_exact_equal(value.get("guarded_dormant_spectral_norm_sites"), [])
        and _json_type_exact_equal(value.get("duplicate_top_level_definition_sites"), [])
        and type(value.get("call_site_count")) is int
        and value.get("call_site_count") >= 0
    )


def _stored_runtime_closed(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == _RUNTIME_GUARD_KEYS
        and value.get("installed") is True
        and _json_type_exact_equal(
            value.get("patched_entrypoints"),
            sorted(EXPECTED_RUNTIME_GUARD_ENTRYPOINTS),
        )
        and type(value.get("patched_entrypoint_count")) is int
        and value.get("patched_entrypoint_count")
        == EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT
        and type(value.get("allowed_call_count")) is int
        and value.get("allowed_call_count") == 0
        and _json_type_exact_equal(value.get("allowed_calls"), [])
        and type(value.get("prohibited_attempt_count")) is int
        and value.get("prohibited_attempt_count") == 0
        and _json_type_exact_equal(value.get("prohibited_attempts"), [])
    )


def _stored_runtime_exact(
    value: Any,
    allowed_calls: set[str],
    *,
    exact_allowed_call_count: int | None = None,
) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == _RUNTIME_GUARD_KEYS
        and value.get("installed") is True
        and _json_type_exact_equal(
            value.get("patched_entrypoints"),
            sorted(EXPECTED_RUNTIME_GUARD_ENTRYPOINTS),
        )
        and type(value.get("patched_entrypoint_count")) is int
        and value.get("patched_entrypoint_count")
        == EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT
        and type(value.get("allowed_call_count")) is int
        and value.get("allowed_call_count") >= len(allowed_calls)
        and (
            exact_allowed_call_count is None
            or value.get("allowed_call_count") == exact_allowed_call_count
        )
        and _json_type_exact_equal(value.get("allowed_calls"), sorted(allowed_calls))
        and type(value.get("prohibited_attempt_count")) is int
        and value.get("prohibited_attempt_count") == 0
        and _json_type_exact_equal(value.get("prohibited_attempts"), [])
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _all_literal_true(value: Any, *, exact_keys: set[str] | None = None) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and (exact_keys is None or set(value) == exact_keys)
        and all(type(item) is bool and item for item in value.values())
    )


def _completion_bound(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
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


def _validate_copied_smoke_inputs(
    *,
    capability_manifest: Mapping[str, Any],
    capability_manifest_path: Path,
    preflight_path: Path,
    resume_proof: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_path = input_path("inputs/smoke_checkpoint.pt")
    completion_path = input_path("inputs/smoke_deployment_completion.json")
    metadata_path = input_path("inputs/smoke_metadata.json")
    precalibration_path = input_path("inputs/smoke_precalibration_ema_state.pt")
    training_path = input_path("inputs/smoke_training_result.json")
    metadata = read_json_object(metadata_path, label="Copied smoke metadata")
    training = read_json_object(training_path, label="Copied smoke training result")
    completion = read_json_object(
        completion_path, label="Copied smoke deployment completion"
    )
    checkpoint_sha = file_sha256(checkpoint_path)
    metadata_sha = file_sha256(metadata_path)
    training_sha = file_sha256(training_path)
    completion_sha = file_sha256(completion_path)
    precalibration_sha = file_sha256(precalibration_path)
    capability_manifest_sha = file_sha256(capability_manifest_path)
    capability_root = CAPABILITY_STAGE_ROOT / CAPABILITY_RUN_ROOT
    expected_paths = {
        "checkpoint": (
            capability_root / "checkpoints/product_pade_rational_smoke_s0.pt"
        ).as_posix(),
        "metadata": (
            capability_root / "metadata/product_pade_rational_smoke_s0.json"
        ).as_posix(),
        "training_result": (
            capability_root / "results/train_product_pade_rational_smoke_s0.json"
        ).as_posix(),
        "precalibration_ema_state": (
            capability_root
            / "training_only/precalibration_ema_product_pade_rational_smoke_s0.pt"
        ).as_posix(),
    }
    completion_keys = set(_completion_bound(completion)) | {"artifact_bundle_sha256"}
    if (
        set(completion) != completion_keys
        or completion.get("schema") != DEPLOYMENT_COMPLETION_SCHEMA
        or completion.get("complete") is not True
        or type(completion.get("seed")) is not int
        or completion.get("seed") != 0
        or completion.get("mode") != "smoke"
        or any(completion.get(key) != value for key, value in expected_paths.items())
        or completion.get("checkpoint_sha256") != checkpoint_sha
        or completion.get("metadata_sha256") != metadata_sha
        or completion.get("training_result_sha256") != training_sha
        or completion.get("precalibration_ema_state_sha256") != precalibration_sha
        or completion.get("source_manifest_sha256") != capability_manifest_sha
        or completion.get("manifest_bundle_sha256")
        != capability_manifest.get("manifest_bundle_sha256")
        or completion.get("artifact_bundle_sha256")
        != _canonical_sha256(_completion_bound(completion))
    ):
        raise RuntimeError("Copied smoke deployment completion differs")
    if (
        metadata.get("schema") != CAPABILITY_SCHEMA
        or training.get("schema") != f"{CAPABILITY_SCHEMA}_training_result"
        or type(metadata.get("seed")) is not int
        or type(training.get("seed")) is not int
        or metadata.get("seed") != training.get("seed")
        or metadata.get("seed") != 0
        or metadata.get("mode") != training.get("mode")
        or metadata.get("mode") != "smoke"
        or metadata.get("checkpoint_sha256") != checkpoint_sha
        or training.get("checkpoint_sha256") != checkpoint_sha
        or training.get("metadata_sha256") != metadata_sha
        or metadata.get("source_manifest", {}).get("sha256")
        != capability_manifest_sha
        or training.get("source_manifest_sha256") != capability_manifest_sha
        or metadata.get("source_manifest", {}).get("manifest_bundle_sha256")
        != capability_manifest.get("manifest_bundle_sha256")
        or training.get("manifest_bundle_sha256")
        != capability_manifest.get("manifest_bundle_sha256")
    ):
        raise RuntimeError("Copied smoke metadata or training result differs")
    loaded_precalibration = load_training_only_precalibration_state(
        precalibration_path,
        expected_file_sha256=precalibration_sha,
        expected_seed=0,
        expected_mode="smoke",
        expected_step_count=SMOKE_STEPS,
        expected_source_manifest_sha256=capability_manifest_sha,
        expected_manifest_bundle_sha256=capability_manifest[
            "manifest_bundle_sha256"
        ],
        expected_source_bundle_sha256=source_bundle_sha256(
            capability_manifest["source_closure"]
        ),
        expected_training_config_sha256=metadata["training_config"]["sha256"],
    )
    expected_static = training.get("transitive_direct_only_static_audit")
    if (
        not _stored_static_closed(expected_static, capability_manifest["source_closure"])
        or not _json_type_exact_equal(
            metadata.get("transitive_direct_only_static_audit"), expected_static
        )
        or not _json_type_exact_equal(
            metadata.get("end_transitive_direct_only_static_audit"), expected_static
        )
    ):
        raise RuntimeError("Copied smoke static audit differs")
    strict = validate_same_allocation_calibration_resume_proof(
        resume_proof,
        expected_source_manifest_sha256=capability_manifest_sha,
        expected_manifest_bundle_sha256=capability_manifest[
            "manifest_bundle_sha256"
        ],
        expected_preflight_sha256=file_sha256(preflight_path),
        expected_source_closure=capability_manifest["source_closure"],
        expected_static_audit=expected_static,
        expected_checkpoint_sha256=checkpoint_sha,
        expected_metadata_sha256=metadata_sha,
        expected_training_result_sha256=training_sha,
        expected_precalibration_sha256=precalibration_sha,
        expected_precalibration_training_state_sha256=loaded_precalibration[
            "training_state_sha256"
        ],
        expected_completion_sha256=completion_sha,
        expected_completion_bundle_sha256=completion["artifact_bundle_sha256"],
        expected_training_environment=metadata["training_environment"],
        expected_calibration_attestation=metadata[
            "post_ema_simultaneous_rational_calibration_attestation"
        ],
        expected_state_transition=metadata["calibration_state_transition"],
        expected_deployment_equivalence=metadata["deployment_export"][
            "inference_equivalence"
        ],
    )
    if not _all_literal_true(strict.get("conditions")):
        raise RuntimeError("Copied smoke resume-proof validator did not close")
    return {
        "checkpoint_sha256": checkpoint_sha,
        "metadata_sha256": metadata_sha,
        "training_result_sha256": training_sha,
        "precalibration_ema_state_sha256": precalibration_sha,
        "deployment_completion_sha256": completion_sha,
        "resume_proof_validation": strict,
    }


def _require_exact_mapping(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError(f"{label} fields differ")
    return value


def _finite_number(value: Any, *, label: str, minimum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"{label} is not numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise RuntimeError(f"{label} is nonfinite or outside its range")
    return parsed


def _finite_float(value: Any, *, label: str, minimum: float | None = None) -> float:
    if type(value) is not float:
        raise RuntimeError(f"{label} is not an exact JSON float")
    return _finite_number(value, label=label, minimum=minimum)


def _exact_integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeError(f"{label} is not an exact integer in range")
    return value


def _validate_replay_panel_record(value: Any) -> None:
    panel = _require_exact_mapping(value, _REPLAY_PANEL_KEYS, label="Replay panel")
    if (
        panel.get("kind") != "deterministic_functional_replay_not_task_degradation"
        or type(panel.get("seed")) is not int
        or panel.get("seed") != SYNTHETIC_REPLAY_SEED
        or not isinstance(panel.get("official_task_indices"), list)
        or any(type(item) is not int for item in panel["official_task_indices"])
        or panel.get("official_task_indices") != list(SYNTHETIC_REPLAY_TASKS)
        or type(panel.get("embodiment")) is not int
        or panel.get("embodiment") != 0
        or type(panel.get("batch_size")) is not int
        or panel.get("batch_size") != len(SYNTHETIC_REPLAY_TASKS)
        or not all(
            _is_sha256(panel.get(key))
            for key in ("instruction_token_sha256", "image_sha256", "state_sha256")
        )
    ):
        raise RuntimeError("Replay panel identity differs")
    bound = {key: panel[key] for key in _REPLAY_PANEL_KEYS - {"panel_sha256"}}
    if panel.get("panel_sha256") != _canonical_sha256(bound):
        raise RuntimeError("Replay panel digest differs")


def _validate_shape_record(value: Any, *, label: str) -> Mapping[str, Any]:
    shape = _require_exact_mapping(value, _SHAPE_KEYS, label=label)
    for key in _SHAPE_KEYS:
        _exact_integer(shape.get(key), label=f"{label}.{key}")
    if (
        shape["unique_nodes"] <= 0
        or shape["edge_occurrences"] <= 0
        or shape["cp_binary_nodes"]
        + shape["reduced_q_binary_nodes"]
        + shape["unary_nodes"]
        + shape["dense_clone_nodes"]
        != shape["unique_nodes"]
        or shape["heterogeneous_physical_sources"] != 97
        or shape["fused_token_feature_dimension"] <= 1
    ):
        raise RuntimeError(f"{label} topology arithmetic differs")
    return shape


def _validate_structure_record(value: Any) -> Mapping[str, Any]:
    structure = _require_exact_mapping(value, _STRUCTURE_KEYS, label="Structure")
    widths = structure.get("physical_source_widths")
    if (
        not isinstance(widths, Mapping)
        or len(widths) != 97
        or any(type(key) is not str or not key for key in widths)
        or any(type(width) is not int or width <= 0 for width in widths.values())
    ):
        raise RuntimeError("Structure physical-source inventory differs")
    for key in (
        "unique_nodes",
        "edge_occurrences",
        "physical_source_count",
        "physical_leaf_count",
        "maximum_physical_source_width",
        "root_projective_width",
        "observable_euclidean_width",
        "active_pade_sites",
    ):
        _exact_integer(structure.get(key), label=f"Structure.{key}")
    if (
        structure["unique_nodes"] <= 0
        or structure["edge_occurrences"] <= 0
        or structure["physical_source_count"] != structure["physical_leaf_count"]
        or structure["physical_source_count"] != len(widths)
        or structure["maximum_physical_source_width"] != max(widths.values())
        or structure["maximum_physical_source_width"] != 192
        or structure["root_projective_width"] != 285
        or structure["observable_kind"] != PRODUCT_COMPONENTS_OBSERVABLE
        or structure["observable_euclidean_width"] != 284
        or structure["product_signs"] is not None
        or structure["active_pade_sites"] != EXPECTED_PADE_SITE_COUNT
    ):
        raise RuntimeError("Structure all-components root differs")
    return structure


def _validate_route_record(
    value: Any, *, initial: Mapping[str, Any], structure: Mapping[str, Any]
) -> Mapping[str, Any]:
    route = _require_exact_mapping(value, _ROUTE_KEYS, label="Route inventory")
    for key in _ROUTE_KEYS - {
        "schema",
        "streamed_tall_unrepresentable_shape_counts",
        "bounded_retained_q_shape_counts",
    }:
        if key == "bounded_retained_q_signed_storage_delta_elements":
            if type(route.get(key)) is not int:
                raise RuntimeError("Route signed storage delta is not an integer")
        else:
            _exact_integer(route.get(key), label=f"Route inventory.{key}")
    for key in (
        "streamed_tall_unrepresentable_shape_counts",
        "bounded_retained_q_shape_counts",
    ):
        counts = route.get(key)
        if (
            not isinstance(counts, Mapping)
            or any(type(name) is not str or "x" not in name for name in counts)
            or any(type(count) is not int or count <= 0 for count in counts.values())
        ):
            raise RuntimeError(f"Route inventory {key} differs")
    if (
        route.get("schema") != "exact_postorder_structural_direct_rq_routes_v1"
        or route["bounded_retained_q_candidate_count"]
        + route["streamed_cp_candidate_count"]
        != initial["cp_binary_nodes"]
        or sum(route["bounded_retained_q_shape_counts"].values())
        != route["bounded_retained_q_candidate_count"]
        or sum(route["streamed_tall_unrepresentable_shape_counts"].values())
        != route["streamed_tall_unrepresentable_count"]
        or route["streamed_tall_unrepresentable_count"] != 0
        or route["simulated_unique_node_count"] != initial["unique_nodes"]
        or route["predicted_root_output_dimension"]
        != structure["root_projective_width"]
        or route["bounded_retained_q_signed_storage_delta_elements"]
        != route["bounded_retained_q_total_elements"]
        - route["bounded_retained_q_replaced_cp_total_elements"]
    ):
        raise RuntimeError("Route inventory arithmetic differs")
    return route


def _validate_algorithm1_record(
    value: Any,
    *,
    initial: Mapping[str, Any],
    canonical: Mapping[str, Any],
) -> set[str]:
    record = _require_exact_mapping(value, _ALGORITHM1_KEYS, label="Algorithm 1")
    normal = _require_exact_mapping(
        record.get("canonical_exponent_normal_form"),
        _NORMAL_FORM_KEYS,
        label="Algorithm 1 normal form",
    )
    provenance = _require_exact_mapping(
        record.get("direct_q_provenance"),
        _DIRECT_Q_PROVENANCE_KEYS,
        label="Algorithm 1 direct-Q provenance",
    )
    telemetry = _require_exact_mapping(
        record.get("telemetry"), _TELEMETRY_KEYS, label="Algorithm 1 telemetry"
    )
    for key in (
        "core_count",
        "cp_direct_q_certificates",
        "cp_direct_q_columns",
        "streamed_compact_cp_direct_q_certificates",
        "streamed_compact_cp_direct_q_columns",
    ):
        _exact_integer(normal.get(key), label=f"Algorithm 1 normal form.{key}")
    if type(normal.get("head_binary_exponent")) is not int:
        raise RuntimeError("Algorithm 1 normal-form head exponent is not an integer")
    methods = record.get("factorization_methods")
    method_counts = record.get("factorization_method_counts")
    if (
        not isinstance(methods, list)
        or methods != sorted(set(methods))
        or not methods
        or not set(methods)
        <= {DIRECT_RQ_METHOD, UNARY_RQ_METHOD, RECTANGULAR_RQ_METHOD}
        or not isinstance(method_counts, dict)
        or set(method_counts) != set(methods)
        or any(type(count) is not int or count <= 0 for count in method_counts.values())
        or sum(method_counts.values()) != record.get("step_count")
    ):
        raise RuntimeError("Algorithm 1 method inventory differs")
    for key in (
        "step_count",
        "full_projective_replay_evaluations",
        "per_step_replays_performed",
        "maximum_local_reconstruction_exponent_delta",
        "maximum_absorption_exponent_delta",
        "child_parent_order_violation_count",
    ):
        _exact_integer(record.get(key), label=f"Algorithm 1.{key}")
    for key in (
        "final_projective_replay_relative_error",
        "maximum_local_scaled_reconstruction_relative_error",
        "maximum_absorption_scaled_relative_error",
    ):
        _finite_float(record.get(key), label=f"Algorithm 1.{key}", minimum=0.0)
    for key in _DIRECT_Q_PROVENANCE_KEYS:
        if key.startswith("maximum_"):
            _finite_float(
                provenance.get(key),
                label=f"Algorithm 1 provenance.{key}",
                minimum=0.0,
            )
        else:
            _exact_integer(provenance.get(key), label=f"Algorithm 1 provenance.{key}")
    telemetry_float_keys = {
        "maximum_direct_q_compact_relative_error",
        "maximum_direct_q_reconstruction_relative_error",
    }
    for key, value_item in telemetry.items():
        if key in telemetry_float_keys:
            _finite_float(
                value_item, label=f"Algorithm 1 telemetry.{key}", minimum=0.0
            )
        else:
            _exact_integer(value_item, label=f"Algorithm 1 telemetry.{key}")
    push = record.get("push_ledger")
    scale_push = record.get("scale_sensitive_occurrence_ledger")
    expected_occurrences = canonical["edge_occurrences"] + 1
    for label, ledger in (("push", push), ("scale-sensitive push", scale_push)):
        if (
            type(ledger) is not list
            or len(ledger) != 2
            or any(type(item) is not int for item in ledger)
        ):
            raise RuntimeError(f"Algorithm 1 {label} ledger differs")
    if (
        record["step_count"] != initial["unique_nodes"]
        or record["step_count"] != canonical["unique_nodes"]
        or record["full_projective_replay_evaluations"] != 2
        or record["per_step_replays_performed"] != 0
        or not _is_sha256(record["canonical_postorder_uid_sha256"])
        or record["canonical_postorder_uid_sha256"]
        != record["algorithm1_step_uid_sha256"]
        or record["child_parent_order_violation_count"] != 0
        or record["final_projective_replay_relative_error"]
        >= REPLAY_RELATIVE_TOLERANCE
        or record["maximum_local_scaled_reconstruction_relative_error"]
        >= LOCAL_SCALE_CERTIFICATION_TOLERANCE
        or record["maximum_absorption_scaled_relative_error"]
        >= LOCAL_SCALE_CERTIFICATION_TOLERANCE
        or record["maximum_local_reconstruction_exponent_delta"] != 0
        or record["maximum_absorption_exponent_delta"] != 0
        or not _json_type_exact_equal(
            push, [expected_occurrences, expected_occurrences]
        )
        or not _json_type_exact_equal(
            scale_push, [expected_occurrences, expected_occurrences]
        )
        or set(normal) != _NORMAL_FORM_KEYS
        or normal.get("core_count") != record["step_count"]
        or normal.get("all_core_exponents_zero") is not True
        or normal.get("head_exponent_matches_ledger") is not True
        or normal.get("all_streamed_compact_cp_q_has_direct_q_provenance") is not True
        or normal.get("cp_direct_q_certificates") != canonical["cp_binary_nodes"]
        or normal.get("streamed_compact_cp_direct_q_certificates")
        != provenance["streamed_steps"]
        or provenance["verified_steps"] != record["step_count"]
        or provenance["expected_steps"] != record["step_count"]
        or provenance["streamed_certificates"] != provenance["streamed_steps"]
        or provenance["columns_compared"] != provenance["expected_columns"]
        or provenance["streamed_replay_calls"]
        != provenance["expected_streamed_replay_calls"]
        or provenance["maximum_compact_q_relative_error"]
        > DIRECT_Q_PROVENANCE_TOLERANCE
        or provenance["maximum_reconstruction_relative_error"]
        > DIRECT_Q_PROVENANCE_TOLERANCE
        or telemetry["raw_order_three_core_materializations"] != 0
        or telemetry["raw_width_cubic_core_materializations"] != 0
        or telemetry["svd_calls"] != 0
        or telemetry["polar_calls"] != 0
        or telemetry["normal_equation_factorizations"] != 0
        or telemetry["local_direct_rq_factorizations"] != record["step_count"]
        or telemetry["direct_q_provenance_certificates"] != record["step_count"]
        or telemetry["streamed_direct_q_provenance_certificates"]
        != provenance["streamed_steps"]
        or telemetry["streamed_direct_q_replay_qr_factorizations"]
        != provenance["expected_streamed_replay_calls"]
        or telemetry["householder_qr_kernel_calls"]
        != telemetry["local_direct_rq_factorizations"]
        - provenance["streamed_steps"]
        + 2 * telemetry["streamed_direct_q_replay_qr_factorizations"]
        or telemetry["direct_q_columns_compared"] != provenance["expected_columns"]
        or telemetry["maximum_direct_q_compact_relative_error"]
        > DIRECT_Q_PROVENANCE_TOLERANCE
        or telemetry["maximum_direct_q_reconstruction_relative_error"]
        > DIRECT_Q_PROVENANCE_TOLERANCE
    ):
        raise RuntimeError("Algorithm 1 reconstruction, occurrence, or provenance record differs")
    calls = {DIRECT_RQ_RUNTIME_CALL, PRODUCTION_ALGORITHM3_RUNTIME_CALL}
    if provenance["streamed_steps"] > 0:
        calls.update({STREAMED_QR_RUNTIME_CALL, TRIANGULAR_RUNTIME_CALL})
    return calls


def _validate_full_rank_certificate_payload(
    payload: Any,
    *,
    inputs: Mapping[str, Any],
    manifest: Mapping[str, Any],
    preflight: Mapping[str, Any],
    clone: Mapping[str, Any],
    lane_tests: Mapping[str, Any],
) -> dict[str, Any]:
    record = _require_exact_mapping(
        payload, _FULL_RANK_TOP_KEYS, label="Full-rank certificate"
    )
    _validate_replay_panel_record(record.get("replay_panel"))
    source_replay = _require_exact_mapping(
        record.get("source_replay"), _SOURCE_REPLAY_KEYS, label="Source replay"
    )
    for key in (
        "component_relative_error",
        "public_action_relative_error",
        "source_decode_relative_error",
        "source_positive_sign_fraction",
    ):
        _finite_float(source_replay.get(key), label=f"Source replay.{key}", minimum=0.0)
    _exact_integer(
        source_replay.get("source_zero_gate_count"),
        label="Source replay.source_zero_gate_count",
    )
    if (
        source_replay.get("component_relative_error") >= REPLAY_RELATIVE_TOLERANCE
        or source_replay.get("public_action_relative_error")
        >= REPLAY_RELATIVE_TOLERANCE
        or source_replay.get("source_decode_relative_error") >= CLONE_RELATIVE_TOLERANCE
        or source_replay.get("source_decode_bitwise_equal") is not True
        or source_replay.get("source_positive_sign_fraction") > 1.0
    ):
        raise RuntimeError("Source replay exactness differs")
    structure = _validate_structure_record(record.get("structure"))
    initial = _validate_shape_record(record.get("initial_shape"), label="Initial shape")
    canonical = _validate_shape_record(
        record.get("canonical_shape"), label="Canonical shape"
    )
    if (
        structure["unique_nodes"] != initial["unique_nodes"]
        or structure["edge_occurrences"] != initial["edge_occurrences"]
        or initial["dense_clone_nodes"] != 0
        or initial["reduced_q_binary_nodes"] != 0
        or canonical["unique_nodes"] != initial["unique_nodes"]
        or canonical["edge_occurrences"] != initial["edge_occurrences"]
        or canonical["dense_clone_nodes"] != 0
        or canonical["heterogeneous_physical_sources"]
        != initial["heterogeneous_physical_sources"]
        or canonical["fused_token_feature_dimension"]
        != initial["fused_token_feature_dimension"]
    ):
        raise RuntimeError("Initial, canonical, and structure inventories differ")
    route = _validate_route_record(
        record.get("route_inventory"), initial=initial, structure=structure
    )
    expected_runtime_calls = _validate_algorithm1_record(
        record.get("algorithm1"), initial=initial, canonical=canonical
    )
    algorithms = _require_exact_mapping(
        record.get("algorithm2_and3"), _ALGORITHM23_KEYS, label="Algorithms 2 and 3"
    )
    for key in ("child_messages", "diagonalized_nodes"):
        _exact_integer(algorithms.get(key), label=f"Algorithms 2 and 3.{key}")
    for key in ("replay_relative_error", "maximum_recontracted_offdiagonal_ratio"):
        _finite_float(
            algorithms.get(key), label=f"Algorithms 2 and 3.{key}", minimum=0.0
        )
    occurrence_ledger = algorithms.get("occurrence_push_ledger")
    if (
        type(occurrence_ledger) is not list
        or len(occurrence_ledger) != 2
        or any(type(item) is not int for item in occurrence_ledger)
    ):
        raise RuntimeError("Algorithm 3 occurrence ledger differs")
    if (
        algorithms.get("explicit_downstream_contraction") is not True
        or algorithms.get("streamed_root_first") is not True
        or algorithms.get("child_messages") != canonical["edge_occurrences"]
        or algorithms.get("eigenbasis_order")
        != "descending environment eigenvalue"
        or algorithms.get("diagonalized_nodes") != canonical["unique_nodes"]
        or not _json_type_exact_equal(
            algorithms.get("occurrence_push_ledger"),
            [canonical["edge_occurrences"] + 1, canonical["edge_occurrences"] + 1],
        )
        or algorithms.get("replay_relative_error") >= REPLAY_RELATIVE_TOLERANCE
        or algorithms.get("maximum_recontracted_offdiagonal_ratio")
        >= OFFDIAGONAL_RELATIVE_TOLERANCE
        or algorithms.get("full_spectra_retained") is not False
        or algorithms.get("post_environment_records_retained") is not False
    ):
        raise RuntimeError("Algorithms 2 and 3 contraction, gauge, or replay record differs")
    algorithm1_record = record["algorithm1"]
    algorithm1_telemetry = algorithm1_record["telemetry"]
    algorithm1_provenance = algorithm1_record["direct_q_provenance"]
    algorithm1_method_counts = algorithm1_record["factorization_method_counts"]
    if (
        algorithm1_provenance["streamed_steps"]
        != route["streamed_cp_candidate_count"]
        or algorithm1_method_counts.get(DIRECT_RQ_METHOD, 0)
        != route["streamed_cp_candidate_count"]
        or algorithm1_method_counts.get(RECTANGULAR_RQ_METHOD, 0)
        != route["bounded_retained_q_candidate_count"]
        or algorithm1_method_counts.get(UNARY_RQ_METHOD, 0)
        != initial["unary_nodes"]
    ):
        raise RuntimeError("Algorithm 1 methods do not reconcile with the predeclared routes")
    nonstreamed_direct_rq_calls = (
        algorithm1_telemetry["local_direct_rq_factorizations"]
        - algorithm1_provenance["streamed_steps"]
    )
    expected_householder_calls = (
        nonstreamed_direct_rq_calls
        + 2 * algorithm1_telemetry["streamed_direct_q_replay_qr_factorizations"]
    )
    expected_triangular_calls = algorithm1_provenance["streamed_steps"]
    expected_evd_calls = algorithms["diagonalized_nodes"]
    expected_runtime_call_count = (
        expected_householder_calls + expected_triangular_calls + expected_evd_calls
    )
    if (
        nonstreamed_direct_rq_calls <= 0
        or algorithm1_telemetry["householder_qr_kernel_calls"]
        != expected_householder_calls
        or expected_triangular_calls <= 0
        or expected_evd_calls != canonical["unique_nodes"]
    ):
        raise RuntimeError("Direct QR, triangular, or environment-EVD call arithmetic differs")
    numerical = _require_exact_mapping(
        record.get("numerical_execution"),
        {
            "runtime",
            "model_parameter_dtypes",
            "model_floating_buffer_dtypes",
            "physical_source_dtypes",
            "running_ms_guard_revalidated_after_float64_conversion",
            "controlled_operation_counts",
        },
        label="Numerical execution",
    )
    if (
        not _json_type_exact_equal(numerical.get("runtime"), _runtime_environment())
        or numerical.get("model_parameter_dtypes") != ["torch.float64"]
        or numerical.get("model_floating_buffer_dtypes") != ["torch.float64"]
        or numerical.get("physical_source_dtypes") != ["torch.float64"]
        or numerical.get("running_ms_guard_revalidated_after_float64_conversion")
        is not True
    ):
        raise RuntimeError("Full-rank numerical execution boundary differs")
    operation_counts = _require_exact_mapping(
        numerical.get("controlled_operation_counts"),
        {
            "direct_rq_householder_qr_calls",
            "streamed_tsqr_householder_qr_calls",
            "triangular_solve_calls",
            "environment_evd_calls",
            "total_controlled_calls",
        },
        label="Controlled numerical operation counts",
    )
    for key in operation_counts:
        _exact_integer(
            operation_counts.get(key), label=f"Controlled numerical operation counts.{key}"
        )
    expected_operation_counts = {
        "direct_rq_householder_qr_calls": nonstreamed_direct_rq_calls,
        "streamed_tsqr_householder_qr_calls": 2
        * algorithm1_telemetry["streamed_direct_q_replay_qr_factorizations"],
        "triangular_solve_calls": expected_triangular_calls,
        "environment_evd_calls": expected_evd_calls,
        "total_controlled_calls": expected_runtime_call_count,
    }
    if dict(operation_counts) != expected_operation_counts:
        raise RuntimeError("Controlled numerical operation ledger differs")
    timings = _require_exact_mapping(
        record.get("timings_seconds"),
        {"compile", "source_evaluation", "algorithm1", "algorithms2_and3", "through_full_rank"},
        label="Full-rank timings",
    )
    for key, value_item in timings.items():
        _finite_float(value_item, label=f"Full-rank timings.{key}", minimum=0.0)
    _finite_float(record.get("peak_rss_mb"), label="Full-rank peak RSS", minimum=0.0)
    gates = record.get("full_rank_gates")
    if not _all_literal_true(gates, exact_keys=_FULL_RANK_GATE_KEYS):
        raise RuntimeError("Full-rank gate inventory differs")
    if (
        record.get("schema")
        != f"{SCHEMA}_algorithm3_full_rank_complete_before_compression"
        or record.get("checkpoint_sha256") != inputs["checkpoint_sha256"]
        or record.get("metadata_sha256") != inputs["metadata_sha256"]
        or record.get("training_result_sha256") != inputs["training_result_sha256"]
        or record.get("smoke_calibration_resume_proof_sha256")
        != inputs["smoke_calibration_resume_proof_sha256"]
        or not _json_type_exact_equal(record.get("preflight"), preflight)
        or not _json_type_exact_equal(record.get("bounded_clone_oracle"), clone)
        or not _json_type_exact_equal(record.get("exact_lane_tests"), lane_tests)
        or not _json_type_exact_equal(
            record.get("dooms_reference"),
            {"sha256": DOOMS_REFERENCE_SHA256, "orientation": DOOMS_ORIENTATION},
        )
        or not _stored_runtime_exact(
            record.get("runtime_guard"),
            expected_runtime_calls,
            exact_allowed_call_count=expected_runtime_call_count,
        )
        or not _json_type_exact_equal(
            record.get("static_audit"), _TRANSITIVE_STATIC_AUDIT
        )
        or record.get("manifest_sha256") != file_sha256(PROJECT_ROOT / MANIFEST_PATH)
        or record.get("manifest_bundle_sha256") != manifest["manifest_bundle_sha256"]
        or record.get("all_full_rank_gates_pass") is not True
        or record.get("bounded_clone_oracle_hash_joined") is not True
        or record.get("exact_lane_test_hash_joined") is not True
        or record.get("canonical_direct_odt_algorithms_1_to_3_completed") is not True
        or record.get("exact_global_decomposability_certified") is not True
        or record.get("compression_claimed") is not False
        or record.get("capability_claimed") is not False
        or record.get("capable_global_decomposable_vla_claimed") is not False
    ):
        raise RuntimeError("Full-rank authority or claim boundary differs")
    return {
        "validated": True,
        "runtime_calls": sorted(expected_runtime_calls),
        "runtime_call_count": expected_runtime_call_count,
        "unique_nodes": canonical["unique_nodes"],
        "edge_occurrences": canonical["edge_occurrences"],
        "full_rank_gates": dict(gates),
    }


def _json_safe_compact_census(value: Mapping[str, Any]) -> dict[str, Any]:
    encoded = dict(value)
    groups = {
        name: dict(summary) for name, summary in value.get("groups", {}).items()
    }
    overall = dict(value.get("overall", {}))
    for summary in (overall, *groups.values()):
        tails = {
            target: dict(item)
            for target, item in summary.get("trace_tail_sums", {}).items()
        }
        for item in tails.values():
            if item.get("mantissa") == 0.0 and item.get("log2_value") == -math.inf:
                item["log2_value"] = None
        summary["trace_tail_sums"] = tails
    encoded["overall"] = overall
    encoded["groups"] = groups
    return encoded


def _validate_compact_group(value: Any, *, label: str) -> Mapping[str, Any]:
    group = _require_exact_mapping(
        value,
        {
            "node_count",
            "total_dimension",
            "trace_retained_dimensions",
            "relative_floor_dimensions",
            "trace_tail_sums",
            "zero_trace_nodes",
            "tie_extended_nodes",
        },
        label=label,
    )
    nodes = _exact_integer(group.get("node_count"), label=f"{label}.node_count")
    dimension = _exact_integer(
        group.get("total_dimension"), label=f"{label}.total_dimension"
    )
    zero_nodes = _exact_integer(
        group.get("zero_trace_nodes"), label=f"{label}.zero_trace_nodes"
    )
    target_keys = {str(value) for value in COMPACT_TRACE_RETENTION_TARGETS}
    floor_keys = {str(value) for value in COMPACT_RELATIVE_VALUE_FLOORS}
    retained = _require_exact_mapping(
        group.get("trace_retained_dimensions"),
        target_keys,
        label=f"{label}.trace_retained_dimensions",
    )
    floor = _require_exact_mapping(
        group.get("relative_floor_dimensions"),
        floor_keys,
        label=f"{label}.relative_floor_dimensions",
    )
    tails = _require_exact_mapping(
        group.get("trace_tail_sums"), target_keys, label=f"{label}.trace_tail_sums"
    )
    ties = _require_exact_mapping(
        group.get("tie_extended_nodes"), target_keys, label=f"{label}.tie_extended_nodes"
    )
    if zero_nodes > nodes:
        raise RuntimeError(f"{label} zero-trace count exceeds its node count")
    for name, rank in {**dict(retained), **dict(floor)}.items():
        _exact_integer(rank, label=f"{label}.rank[{name}]")
        if rank > dimension:
            raise RuntimeError(f"{label} retained dimension exceeds total")
    for name, count in ties.items():
        _exact_integer(count, label=f"{label}.ties[{name}]")
        if count > nodes:
            raise RuntimeError(f"{label} tie count exceeds nodes")
    for name, summary in tails.items():
        tail = _require_exact_mapping(
            summary,
            {"mantissa", "binary_exponent", "log2_value"},
            label=f"{label}.tail[{name}]",
        )
        mantissa = _finite_float(
            tail.get("mantissa"), label=f"{label}.tail[{name}].mantissa", minimum=0.0
        )
        _exact_integer(
            tail.get("binary_exponent"),
            label=f"{label}.tail[{name}].binary_exponent",
            minimum=-(1 << 62),
        )
        if mantissa == 0.0:
            if tail.get("log2_value") is not None:
                raise RuntimeError(f"{label} zero tail must use a null logarithm")
        else:
            observed_log = _finite_float(
                tail.get("log2_value"), label=f"{label}.tail[{name}].log2_value"
            )
            expected_log = math.log2(mantissa) + tail["binary_exponent"]
            if not math.isclose(observed_log, expected_log, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"{label} tail logarithm differs")
    return group


def _validate_compact_census(
    value: Any, *, expected_nodes: int
) -> Mapping[str, Any]:
    census = _require_exact_mapping(
        value,
        {
            "node_count",
            "full_spectra_retained",
            "tail_quantity",
            "simultaneous_multibond_error_bound_claimed",
            "trace_targets",
            "relative_value_floors",
            "overall",
            "groups",
        },
        label="Compact census",
    )
    groups = census.get("groups")
    if not isinstance(groups, Mapping) or not groups or any(
        type(name) is not str or not name for name in groups
    ):
        raise RuntimeError("Compact census group inventory differs")
    overall = _validate_compact_group(census.get("overall"), label="Compact census overall")
    validated_groups = {
        name: _validate_compact_group(summary, label=f"Compact census group {name}")
        for name, summary in groups.items()
    }
    _exact_integer(census.get("node_count"), label="Compact census.node_count")
    if (
        census.get("node_count") != expected_nodes
        or census.get("full_spectra_retained") is not False
        or census.get("tail_quantity")
        != (
            "sum of discarded Algorithm 2 environment eigenvalues, with "
            "the environment binary scale restored"
        )
        or census.get("simultaneous_multibond_error_bound_claimed") is not False
        or not _json_type_exact_equal(
            census.get("trace_targets"), list(COMPACT_TRACE_RETENTION_TARGETS)
        )
        or not _json_type_exact_equal(
            census.get("relative_value_floors"),
            list(COMPACT_RELATIVE_VALUE_FLOORS),
        )
        or overall["node_count"] != expected_nodes
        or sum(group["node_count"] for group in validated_groups.values())
        != expected_nodes
        or sum(group["total_dimension"] for group in validated_groups.values())
        != overall["total_dimension"]
        or any(
            validated_groups.get(name, {}).get("node_count", 0) <= 0
            for name in ("product_center", "product_factors", "product_gates")
        )
    ):
        raise RuntimeError("Compact census coverage or claim boundary differs")
    return census


def _validate_component_metric_record(value: Any, *, label: str) -> None:
    metrics = _require_exact_mapping(value, _COMPONENT_METRIC_KEYS, label=label)
    for key in (
        "component_relative_error",
        "decoded_action_relative_error",
        "decoded_action_max_absolute_error",
        "gate_sign_flip_fraction",
    ):
        _finite_float(metrics.get(key), label=f"{label}.{key}", minimum=0.0)
    coordinates = metrics.get("decoded_action_coordinate_max_absolute_error")
    if not isinstance(coordinates, list) or len(coordinates) != ACTION_DIM:
        raise RuntimeError(f"{label} action-coordinate inventory differs")
    for index, item in enumerate(coordinates):
        _finite_float(item, label=f"{label}.coordinate[{index}]", minimum=0.0)
    flips = _exact_integer(
        metrics.get("gate_sign_flip_count"), label=f"{label}.gate_sign_flip_count"
    )
    maximum_flips = len(SYNTHETIC_REPLAY_TASKS) * N_FACTORS
    if (
        flips > maximum_flips
        or metrics["gate_sign_flip_fraction"] > 1.0
        or not math.isclose(
            metrics["gate_sign_flip_fraction"],
            flips / maximum_flips,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise RuntimeError(f"{label} gate-sign arithmetic differs")


def _validate_compression_result_payload(
    payload: Any,
    *,
    full_rank: Mapping[str, Any],
    full_rank_sha256: str,
    full_rank_validation: Mapping[str, Any],
    inputs: Mapping[str, Any],
    manifest: Mapping[str, Any],
    preflight: Mapping[str, Any],
    clone: Mapping[str, Any],
    lane_tests: Mapping[str, Any],
) -> dict[str, Any]:
    record = _require_exact_mapping(
        payload, _FULL_RESULT_TOP_KEYS, label="Full compression result"
    )
    _validate_replay_panel_record(record.get("replay_panel"))
    if not _json_type_exact_equal(
        record.get("replay_panel"), full_rank.get("replay_panel")
    ):
        raise RuntimeError("Full result replay panel differs from full-rank certificate")
    reference = _require_exact_mapping(
        record.get("full_rank_certificate"),
        {"path", "sha256"},
        label="Full-rank certificate reference",
    )
    if (
        reference.get("path") != full_rank_progress_path().as_posix()
        or reference.get("sha256") != full_rank_sha256
    ):
        raise RuntimeError("Full-rank certificate physical hash join differs")
    canonical_storage = _exact_integer(
        record.get("canonical_storage_elements"),
        label="Canonical storage elements",
        minimum=1,
    )
    _validate_compact_census(
        record.get("compact_census"),
        expected_nodes=int(full_rank_validation["unique_nodes"]),
    )
    curves = _require_exact_mapping(
        record.get("compression_curves"),
        {str(value) for value in COMPACT_TRACE_RETENTION_TARGETS},
        label="Compression curves",
    )
    for target in COMPACT_TRACE_RETENTION_TARGETS:
        label = f"Compression curve {target}"
        curve = _require_exact_mapping(
            curves[str(target)],
            {
                "target",
                "retained_bond_dimensions",
                "original_bond_dimensions",
                "projected_storage_elements",
                "projected_storage_fraction",
                "prefix_metrics",
                "prefix_evaluation_seconds",
                "matched_width_trailing_control",
            },
            label=label,
        )
        retained = _exact_integer(
            curve.get("retained_bond_dimensions"),
            label=f"{label}.retained_bond_dimensions",
            minimum=1,
        )
        original = _exact_integer(
            curve.get("original_bond_dimensions"),
            label=f"{label}.original_bond_dimensions",
            minimum=1,
        )
        projected = _exact_integer(
            curve.get("projected_storage_elements"),
            label=f"{label}.projected_storage_elements",
            minimum=1,
        )
        fraction = _finite_float(
            curve.get("projected_storage_fraction"),
            label=f"{label}.projected_storage_fraction",
            minimum=0.0,
        )
        _finite_float(
            curve.get("prefix_evaluation_seconds"),
            label=f"{label}.prefix_evaluation_seconds",
            minimum=0.0,
        )
        _validate_component_metric_record(curve.get("prefix_metrics"), label=f"{label}.prefix_metrics")
        if (
            type(curve.get("target")) is not float
            or curve.get("target") != target
            or retained > original
            or projected > canonical_storage
            or fraction > 1.0
            or not math.isclose(
                fraction, projected / canonical_storage, rel_tol=0.0, abs_tol=1e-15
            )
        ):
            raise RuntimeError(f"{label} rank or storage arithmetic differs")
        control = curve.get("matched_width_trailing_control")
        if not isinstance(control, Mapping):
            raise RuntimeError(f"{label} trailing control is missing")
        valid = control.get("valid_projective_chart")
        if valid is True:
            _require_exact_mapping(
                control,
                {"valid_projective_chart", "metrics", "evaluation_seconds"},
                label=f"{label} trailing control",
            )
            _validate_component_metric_record(
                control.get("metrics"), label=f"{label}.trailing_control.metrics"
            )
        elif valid is False:
            _require_exact_mapping(
                control,
                {"valid_projective_chart", "rejection", "evaluation_seconds"},
                label=f"{label} trailing control",
            )
            if not isinstance(control.get("rejection"), str) or not control["rejection"]:
                raise RuntimeError(f"{label} trailing-control rejection is empty")
        else:
            raise RuntimeError(f"{label} trailing-control validity is not boolean")
        _finite_float(
            control.get("evaluation_seconds"),
            label=f"{label}.trailing_control.evaluation_seconds",
            minimum=0.0,
        )
    material = _require_exact_mapping(
        record.get("physical_materialization"),
        _MATERIALIZATION_KEYS,
        label="Physical materialization",
    )
    for key in (
        "projective_pair_relative_error_against_masked",
        "projective_pair_gauge_invariant_relative_error_against_masked",
        "component_relative_error_against_masked",
        "decoded_action_relative_error_against_masked",
        "elapsed_seconds",
    ):
        _finite_float(
            material.get(key),
            label=f"Physical materialization.{key}",
            minimum=0.0,
        )
    coordinates = material.get("decoded_action_coordinate_max_absolute_error_against_masked")
    if not isinstance(coordinates, list) or len(coordinates) != ACTION_DIM:
        raise RuntimeError("Physical materialization coordinate inventory differs")
    for index, item in enumerate(coordinates):
        _finite_float(
            item, label=f"Physical materialization.coordinate[{index}]", minimum=0.0
        )
    for key in (
        "gate_sign_mismatch_count_against_masked",
        "original_storage_elements",
        "projected_storage_elements",
        "materialized_storage_elements",
        "allocated_storage_elements",
        "reduced_bonds",
        "expected_occurrence_slices",
        "applied_occurrence_slices",
    ):
        _exact_integer(material.get(key), label=f"Physical materialization.{key}")
    material_gates = record.get("physical_materialization_gates")
    if not _all_literal_true(material_gates, exact_keys=_MATERIALIZATION_GATE_KEYS):
        raise RuntimeError("Physical materialization gate inventory differs")
    if (
        type(material.get("target")) is not float
        or material.get("target") != PHYSICAL_MATERIALIZATION_TARGET
        or material.get("performed_after_all_masked_curves") is not True
        or material.get("copy_network") is not False
        or material.get("executor_kind") != "materialized_implicit_projective_dag"
        or material.get("materialized_tensor_network_executed") is not True
        or material.get("source_model_forward_fallback_forbidden") is not True
        or type(
            material.get("source_model_forward_calls_during_materialized_execution")
        )
        is not int
        or material.get("source_model_forward_calls_during_materialized_execution") != 0
        or material["projective_pair_gauge_invariant_relative_error_against_masked"]
        >= CLONE_RELATIVE_TOLERANCE
        or material["component_relative_error_against_masked"] >= CLONE_RELATIVE_TOLERANCE
        or material["decoded_action_relative_error_against_masked"]
        >= CLONE_RELATIVE_TOLERANCE
        or material["gate_sign_mismatch_count_against_masked"] != 0
        or material["original_storage_elements"] != canonical_storage
        or material["projected_storage_elements"]
        != material["materialized_storage_elements"]
        or material["materialized_storage_elements"]
        != material["allocated_storage_elements"]
        or material["materialized_storage_elements"] >= canonical_storage
        or material["reduced_bonds"] <= 0
        or material["expected_occurrence_slices"]
        != material["applied_occurrence_slices"]
        or material["applied_occurrence_slices"] <= 0
    ):
        raise RuntimeError("Physical materialization equivalence or storage arithmetic differs")
    expected_runtime_calls = set(full_rank_validation["runtime_calls"])
    timings = record.get("timings_seconds")
    expected_timing_keys = set(full_rank["timings_seconds"]) | {"total"}
    _require_exact_mapping(timings, expected_timing_keys, label="Full-result timings")
    for key, value_item in timings.items():
        _finite_float(value_item, label=f"Full-result timings.{key}", minimum=0.0)
    if not _json_type_exact_equal(
        {key: timings[key] for key in full_rank["timings_seconds"]},
        full_rank["timings_seconds"],
    ):
        raise RuntimeError("Full-result timings do not preserve the full-rank certificate")
    _finite_float(record.get("peak_rss_mb"), label="Full-result peak RSS", minimum=0.0)
    if (
        record.get("schema") != f"{SCHEMA}_full_with_same_sweep_compression"
        or record.get("checkpoint_sha256") != inputs["checkpoint_sha256"]
        or record.get("smoke_calibration_resume_proof_sha256")
        != inputs["smoke_calibration_resume_proof_sha256"]
        or not _json_type_exact_equal(record.get("bounded_clone_oracle"), clone)
        or not _json_type_exact_equal(record.get("exact_lane_tests"), lane_tests)
        or not _json_type_exact_equal(record.get("preflight"), preflight)
        or not _json_type_exact_equal(
            record.get("full_rank_gates"), full_rank["full_rank_gates"]
        )
        or record.get("all_full_rank_gates_pass") is not True
        or record.get("compression_curves_are_functional_not_task_degradation") is not True
        or record.get("all_compression_gates_pass") is not True
        or not _stored_runtime_exact(
            record.get("runtime_guard_final"),
            expected_runtime_calls,
            exact_allowed_call_count=full_rank_validation["runtime_call_count"],
        )
        or not _json_type_exact_equal(
            record.get("static_audit"), _TRANSITIVE_STATIC_AUDIT
        )
        or record.get("manifest_sha256") != file_sha256(PROJECT_ROOT / MANIFEST_PATH)
        or record.get("manifest_bundle_sha256") != manifest["manifest_bundle_sha256"]
        or record.get("canonical_direct_odt_algorithms_1_to_3_completed") is not True
        or record.get("bounded_clone_oracle_hash_joined") is not True
        or record.get("exact_lane_test_hash_joined") is not True
        or record.get("exact_global_decomposability_certified") is not True
        or record.get("compression_claimed_after_full_rank_certificate") is not True
        or record.get("capability_claimed") is not False
        or record.get("capable_global_decomposable_vla_claimed") is not False
    ):
        raise RuntimeError("Full compression authority or claim boundary differs")
    return {
        "validated": True,
        "physical_materialization_gates": dict(material_gates),
        "canonical_storage_elements": canonical_storage,
    }


def _validate_prestage_inputs(exact_manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = input_path("inputs/odt_prestage_manifest.json")
    ledger_path = input_path("inputs/odt_prestage_ledger.json")
    prestage = read_json_object(manifest_path, label="Copied exact-ODT prestage manifest")
    ledger = read_json_object(ledger_path, label="Copied exact-ODT prestage ledger")
    closure = prestage.get("closure")
    if (
        prestage.get("schema") != PRESTAGE_SCHEMA
        or not isinstance(closure, dict)
        or tuple(sorted(closure)) != PRESTAGE_CLOSURE
        or prestage.get("bundle_sha256")
        != _canonical_sha256(dict(sorted(closure.items())))
    ):
        raise RuntimeError("Copied exact-ODT prestage manifest differs")
    expected = {
        **exact_manifest["source_closure"],
        **exact_manifest["launch_closure"],
        "reference/dooms_xnets_2504.02667.pdf": exact_manifest[
            "authenticated_inputs"
        ]["reference/dooms_xnets_2504.02667.pdf"],
    }
    if closure != dict(sorted(expected.items())):
        raise RuntimeError("Prestage authority bytes differ from the final exact manifest")
    capability_manifest_sha = file_sha256(
        input_path("inputs/capability_source_manifest.json")
    )
    if (
        prestage.get("capability_source_manifest_sha256")
        != capability_manifest_sha
        or prestage.get("dooms_reference_sha256") != DOOMS_REFERENCE_SHA256
        or ledger.get("schema") != f"{PRESTAGE_SCHEMA}_ledger"
        or ledger.get("prestage_manifest_sha256") != file_sha256(manifest_path)
        or ledger.get("prestage_bundle_sha256") != prestage.get("bundle_sha256")
        or ledger.get("capability_source_manifest_sha256")
        != capability_manifest_sha
        or ledger.get("source_sha256") != exact_manifest["source_closure"]
        or ledger.get("launch_sha256") != exact_manifest["launch_closure"]
        or ledger.get("no_job_submitted") is not True
    ):
        raise RuntimeError("Copied exact-ODT prestage ledger differs")
    static = ledger.get("static_audit")
    if not _stored_static_closed(static, exact_manifest["source_closure"]):
        raise RuntimeError("Prestage exact static audit did not close")
    return {
        "manifest_sha256": file_sha256(manifest_path),
        "ledger_sha256": file_sha256(ledger_path),
        "bundle_sha256": prestage["bundle_sha256"],
    }


def _capability_manifest() -> dict[str, Any]:
    path = input_path("inputs/capability_source_manifest.json")
    payload = read_json_object(path, label="Copied capability source manifest")
    if payload.get("schema") != f"{CAPABILITY_SCHEMA}_source_manifest":
        raise RuntimeError("Copied capability source manifest schema differs")
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
        if not isinstance(stored, dict) or tuple(sorted(stored)) != tuple(
            sorted(expected)
        ):
            raise RuntimeError(f"Copied capability manifest {section} differs")
        if any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in stored.values()
        ):
            raise RuntimeError(f"Copied capability manifest {section} digest is malformed")
        if payload.get(bundle_key) != _canonical_sha256(dict(sorted(stored.items()))):
            raise RuntimeError(f"Copied capability manifest {bundle_key} differs")
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
    if payload.get("manifest_bundle_sha256") != _canonical_sha256(bound):
        raise RuntimeError("Copied capability aggregate manifest digest differs")
    source = payload["source_closure"]
    exact_manifest = verify_manifest(PROJECT_ROOT / MANIFEST_PATH)
    for relative, digest in source.items():
        if exact_manifest["source_closure"].get(relative) != digest:
            if relative not in ODT_SOURCE_OVERRIDES:
                raise RuntimeError(f"Trained capability source byte changed in ODT stage: {relative}")
    if not set(ODT_SOURCE_OVERRIDES).issubset(source):
        raise RuntimeError("Exact-ODT source override is not in the trained source tuple")
    return payload


def _validate_capability_inputs() -> dict[str, Any]:
    exact_manifest = verify_manifest(PROJECT_ROOT / MANIFEST_PATH)
    prestage = _validate_prestage_inputs(exact_manifest)
    capability_manifest = _capability_manifest()
    capability_manifest_path = input_path("inputs/capability_source_manifest.json")
    preflight_path = input_path("inputs/capability_preflight.json")
    checkpoint_path = input_path("inputs/seed0_checkpoint.pt")
    completion_path = input_path("inputs/seed0_deployment_completion.json")
    metadata_path = input_path("inputs/seed0_metadata.json")
    precalibration_path = input_path("inputs/seed0_precalibration_ema_state.pt")
    resume_proof_path = input_path("inputs/smoke_calibration_resume_proof.json")
    training_path = input_path("inputs/seed0_training_result.json")
    metadata = read_json_object(metadata_path, label="Seed0 deployment metadata")
    training = read_json_object(training_path, label="Seed0 training result")
    completion = read_json_object(
        completion_path, label="Seed0 deployment completion"
    )
    checkpoint_sha = file_sha256(checkpoint_path)
    metadata_sha = file_sha256(metadata_path)
    training_sha = file_sha256(training_path)
    completion_sha = file_sha256(completion_path)
    precalibration_sha = file_sha256(precalibration_path)
    resume_proof_sha = file_sha256(resume_proof_path)
    resume_proof = read_json_object(
        resume_proof_path, label="Same-allocation smoke calibration resume proof"
    )
    smoke_validation = _validate_copied_smoke_inputs(
        capability_manifest=capability_manifest,
        capability_manifest_path=capability_manifest_path,
        preflight_path=preflight_path,
        resume_proof=resume_proof,
    )
    preflight = validate_preflight_certificate(
        preflight_path, capability_manifest_path, capability_manifest
    )
    export = metadata.get("deployment_export", {})
    inference = export.get("inference_equivalence", {})
    calibration = metadata.get("post_ema_simultaneous_rational_calibration", {})
    training_runtime_import = training.get("runtime_direct_only_guard_at_import")
    training_runtime_final = training.get("runtime_direct_only_guard_final")
    trained_sources = capability_manifest["source_closure"]
    metadata_manifest = metadata.get("source_manifest", {})
    completion_bound = {
        key: completion.get(key)
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
    capability_root = CAPABILITY_STAGE_ROOT / CAPABILITY_RUN_ROOT
    resume_static = resume_proof.get("transitive_direct_only_static_audit", {})
    resume_runtime = resume_proof.get("runtime_direct_only_guard", {})
    resume_bitwise = resume_proof.get("deployment_equivalence", {}).get(
        "bitwise_checks", {}
    )
    resume_allocation = resume_proof.get("allocation", {})
    resume_artifact_hash_fields = (
        "fresh_checkpoint_sha256",
        "fresh_metadata_sha256",
        "fresh_training_result_sha256",
        "precalibration_ema_state_sha256",
        "fresh_deployment_completion_sha256",
    )
    expected_official_tasks = {
        str(key): value for key, value in OFFICIAL_TASK_LANGUAGES.items()
    }
    conditions = {
        "training_schema": training.get("schema") == f"{CAPABILITY_SCHEMA}_training_result",
        "metadata_schema": metadata.get("schema") == CAPABILITY_SCHEMA,
        "recipe": training.get("recipe_version") == metadata.get("recipe_version") == RECIPE_VERSION,
        "seed0_full": type(training.get("seed")) is int
        and type(metadata.get("seed")) is int
        and training.get("seed") == metadata.get("seed") == 0
        and training.get("mode") == metadata.get("mode") == "full",
        "checkpoint_hash": training.get("checkpoint_sha256")
        == metadata.get("checkpoint_sha256")
        == checkpoint_sha,
        "metadata_hash": training.get("metadata_sha256") == metadata_sha,
        "completion_input_hash": completion_sha
        == exact_manifest["authenticated_inputs"][
            "inputs/seed0_deployment_completion.json"
        ],
        "completion_schema": completion.get("schema")
        == DEPLOYMENT_COMPLETION_SCHEMA
        and completion.get("complete") is True,
        "completion_identity": type(completion.get("seed")) is int
        and completion.get("seed") == 0
        and completion.get("mode") == "full"
        and completion.get("checkpoint")
        == (capability_root / "checkpoints/product_pade_rational_s0.pt").as_posix()
        and completion.get("metadata")
        == (capability_root / "metadata/product_pade_rational_s0.json").as_posix()
        and completion.get("training_result")
        == (capability_root / "results/train_product_pade_rational_s0.json").as_posix()
        and completion.get("precalibration_ema_state")
        == (
            capability_root
            / "training_only/precalibration_ema_product_pade_rational_s0.pt"
        ).as_posix(),
        "completion_hash_join": completion.get("checkpoint_sha256")
        == checkpoint_sha
        and completion.get("metadata_sha256") == metadata_sha
        and completion.get("training_result_sha256") == training_sha
        and completion.get("precalibration_ema_state_sha256")
        == precalibration_sha
        == metadata.get("training_only_precalibration_ema_state", {}).get("sha256")
        == training.get("training_only_precalibration_ema_state", {}).get("sha256"),
        "completion_source_join": completion.get("source_manifest_sha256")
        == file_sha256(capability_manifest_path)
        and completion.get("manifest_bundle_sha256")
        == capability_manifest.get("manifest_bundle_sha256"),
        "completion_bundle": completion.get("artifact_bundle_sha256")
        == _canonical_sha256(completion_bound),
        "resume_proof_schema": resume_proof.get("schema")
        == CALIBRATION_RESUME_PROOF_SCHEMA
        and type(resume_proof.get("seed")) is int
        and resume_proof.get("seed") == 0
        and resume_proof.get("mode") == "smoke",
        "resume_proof_zero_training_same_allocation": type(
            resume_proof.get("optimizer_steps_executed")
        )
        is int
        and resume_proof.get("optimizer_steps_executed") == 0
        and resume_proof.get("external_sha_authority_supplied") is True
        and resume_proof.get("same_slurm_gpu_allocation_as_fresh_smoke") is True
        and isinstance(resume_allocation, dict)
        and set(resume_allocation)
        == {
            "gpu",
            "compute_capability",
            "slurm_job_id",
            "slurm_job_gpus",
            "cuda_visible_devices",
        }
        and resume_allocation.get("gpu") == "NVIDIA RTX A6000"
        and _json_type_exact_equal(
            resume_allocation.get("compute_capability"), [8, 6]
        )
        and all(
            isinstance(resume_allocation.get(key), str)
            and bool(resume_allocation.get(key))
            for key in ("slurm_job_id", "slurm_job_gpus", "cuda_visible_devices")
        ),
        "resume_proof_exact_replay": resume_proof.get(
            "fresh_and_resume_calibration_report_bitwise_equal"
        )
        is True
        and resume_proof.get("fresh_and_resume_final_running_ms_bitwise_equal")
        is True
        and resume_proof.get("fresh_and_resume_deployment_state_bitwise_equal")
        is True
        and resume_proof.get(
            "fresh_and_resume_raw_components_and_actions_bitwise_equal"
        )
        is True
        and _json_type_exact_equal(
            resume_proof.get("removed_training_only_keys"),
            ["teacher_head.bias", "teacher_head.weight"],
        )
        and isinstance(resume_bitwise, dict)
        and bool(resume_bitwise)
        and all(value is True for value in resume_bitwise.values())
        and isinstance(resume_proof.get("fresh_output_tensor_sha256"), str)
        and len(resume_proof["fresh_output_tensor_sha256"]) == 64
        and all(
            character in "0123456789abcdef"
            for character in resume_proof["fresh_output_tensor_sha256"]
        )
        and resume_proof.get("fresh_output_tensor_sha256")
        == resume_proof.get("resume_output_tensor_sha256")
        == resume_proof.get("deployment_equivalence", {}).get(
            "output_tensor_sha256"
        ),
        "resume_proof_artifact_hashes": all(
            isinstance(resume_proof.get(field), str)
            and len(resume_proof[field]) == 64
            and all(character in "0123456789abcdef" for character in resume_proof[field])
            for field in resume_artifact_hash_fields
        ),
        "resume_proof_source": resume_proof.get("source_manifest_sha256")
        == file_sha256(capability_manifest_path)
        and resume_proof.get("manifest_bundle_sha256")
        == capability_manifest.get("manifest_bundle_sha256")
        and _json_type_exact_equal(
            resume_proof.get("source_snapshot_start"), trained_sources
        )
        and _json_type_exact_equal(
            resume_proof.get("source_snapshot_end"), trained_sources
        ),
        "resume_proof_preflight": resume_proof.get("preflight_certificate_sha256")
        == file_sha256(preflight_path),
        "resume_proof_static": _stored_static_closed(
            resume_static, trained_sources
        )
        and _json_type_exact_equal(
            resume_static, training.get("transitive_direct_only_static_audit")
        ),
        "resume_proof_runtime": _stored_runtime_closed(resume_runtime),
        "resume_proof_scope": resume_proof.get("recovery_scope")
        == CALIBRATION_RECOVERY_SCOPE
        and resume_proof.get("cross_version_recovery_supported") is False,
        "resume_proof_strict_independent_consumer": _all_literal_true(
            smoke_validation.get("resume_proof_validation", {}).get("conditions")
        ),
        "training_result_hash": training_sha
        == exact_manifest["authenticated_inputs"]["inputs/seed0_training_result.json"],
        "manifest_hash": training.get("source_manifest_sha256")
        == metadata_manifest.get("sha256")
        == file_sha256(capability_manifest_path),
        "manifest_bundles": training.get("manifest_bundle_sha256")
        == metadata_manifest.get("manifest_bundle_sha256")
        == capability_manifest.get("manifest_bundle_sha256")
        and metadata_manifest.get("bundle_sha256")
        == capability_manifest.get("source_bundle_sha256")
        and metadata_manifest.get("authenticated_inputs_bundle_sha256")
        == capability_manifest.get("authenticated_inputs_bundle_sha256")
        and metadata_manifest.get("launch_bundle_sha256")
        == capability_manifest.get("launch_bundle_sha256"),
        "preflight_hash": training.get("preflight_certificate", {}).get("sha256")
        == metadata.get("preflight_certificate", {}).get("sha256")
        == file_sha256(preflight_path),
        "deployment_only": metadata.get("checkpoint_contains_deployment_state_only") is True,
        "frozen_source_snapshots": _json_type_exact_equal(
            metadata.get("source_snapshot_start"), trained_sources
        )
        and _json_type_exact_equal(
            metadata.get("source_snapshot_end"), trained_sources
        ),
        "strict_export": export.get("strict_load") is True
        and export.get("serialized_strict_reload") is True
        and export.get("retained_state_bitwise_equal") is True,
        "only_teacher_removed": _json_type_exact_equal(
            export.get("removed_training_only_keys"),
            ["teacher_head.bias", "teacher_head.weight"],
        )
        and _json_type_exact_equal(export.get("removed_non_training_keys"), []),
        "inference_export_bitwise": isinstance(inference.get("bitwise_checks"), dict)
        and inference.get("bitwise_checks")
        and all(value is True for value in inference["bitwise_checks"].values())
        and isinstance(inference.get("output_tensor_sha256"), str)
        and len(inference["output_tensor_sha256"]) == 64
        and all(
            character in "0123456789abcdef"
            for character in inference["output_tensor_sha256"]
        )
        and type(inference.get("max_abs_action_error")) is float
        and inference.get("max_abs_action_error") == 0.0
        and type(inference.get("max_abs_gate_error")) is float
        and inference.get("max_abs_gate_error") == 0.0,
        "calibration_passes": type(calibration.get("update_passes")) is int
        and calibration.get("update_passes") == FINAL_CALIBRATION_PASSES,
        "calibration_final_audit": calibration.get("final_no_commit_audit_pass") is True
        and type(calibration.get("audit_observed_max_relative_change")) is float
        and calibration.get("audit_observed_max_relative_change", math.inf)
        <= FINAL_CALIBRATION_AUDIT_MAX_RELATIVE_CHANGE,
        "calibration_sites": type(calibration.get("live_site_count")) is int
        and calibration.get("live_site_count") == EXPECTED_PADE_SITE_COUNT
        and calibration.get("all_initialized_and_frozen") is True,
        "calibration_anchor_and_recovery": calibration.get("initializer")
        == CALIBRATION_INITIALIZER_LABEL
        and calibration.get("old_training_buffers_discarded") is False
        and calibration.get("observation_copy_dtype") == "float64"
        and calibration.get("observation_copy_before_square") is True
        and calibration.get("accumulator_dtype") == "float64"
        and calibration.get("recovery_scope") == CALIBRATION_RECOVERY_SCOPE
        and calibration.get("cross_version_recovery_supported") is False,
        "calibration_and_postema_artifact_joins": _json_type_exact_equal(
            training.get("post_ema_state_validation"),
            metadata.get("post_ema_state_validation"),
        )
        and _json_type_exact_equal(
            training.get("training_only_precalibration_ema_state"),
            metadata.get("training_only_precalibration_ema_state"),
        )
        and _json_type_exact_equal(
            training.get("post_ema_simultaneous_rational_calibration"), calibration
        )
        and _json_type_exact_equal(
            training.get("post_ema_simultaneous_rational_calibration_attestation"),
            metadata.get("post_ema_simultaneous_rational_calibration_attestation"),
        ),
        "completion_marker_required": training.get(
            "deployment_publication_requires_completion_marker"
        )
        is True
        and metadata.get("deployment_publication_requires_completion_marker")
        is True,
        "training_export_matches_metadata": _json_type_exact_equal(
            training.get("deployment_export"), export
        ),
        "training_calibration_matches_metadata": _json_type_exact_equal(
            training.get("post_ema_simultaneous_rational_calibration"), calibration
        ),
        "official_task_map": _json_type_exact_equal(
            metadata.get("official_tasks"), expected_official_tasks
        )
        and _canonical_sha256(OFFICIAL_TASK_LANGUAGES)
        == OFFICIAL_TASK_LANGUAGES_SHA256,
        "vocab_digest": metadata.get("vocab_sha256") == VOCAB_SHA256,
        "training_static": _stored_static_closed(
            training.get("transitive_direct_only_static_audit"), trained_sources
        ),
        "metadata_static_start": _stored_static_closed(
            metadata.get("transitive_direct_only_static_audit"), trained_sources
        ),
        "metadata_static_end": _stored_static_closed(
            metadata.get("end_transitive_direct_only_static_audit"), trained_sources
        ),
        "training_runtime_import": _stored_runtime_closed(training_runtime_import),
        "training_runtime_final": _stored_runtime_closed(training_runtime_final),
        "claim_boundary": training.get("claim_boundary")
        == metadata.get("claim_boundary")
        == DEPLOYMENT_CLAIM_BOUNDARY,
    }
    failed = sorted(name for name, passed in conditions.items() if not passed)
    if failed:
        raise RuntimeError(f"Seed0 capability input chain failed: {failed}")
    precalibration = load_training_only_precalibration_state(
        precalibration_path,
        expected_file_sha256=precalibration_sha,
        expected_seed=0,
        expected_mode="full",
        expected_step_count=training_recipe(False)["steps"],
        expected_source_manifest_sha256=file_sha256(capability_manifest_path),
        expected_manifest_bundle_sha256=capability_manifest[
            "manifest_bundle_sha256"
        ],
        expected_source_bundle_sha256=source_bundle_sha256(
            capability_manifest["source_closure"]
        ),
        expected_training_config_sha256=metadata["training_config"]["sha256"],
    )
    checkpoint_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_state, dict):
        raise RuntimeError("Seed0 checkpoint is not a state mapping")
    transition_view = validate_precalibration_to_deployment_transition(
        precalibration["training_state"],
        checkpoint_state,
        active_site_names=tuple(calibration.get("initializer_running_ms", {})),
        claimed_proof=metadata.get("calibration_state_transition", {}),
    )
    if training.get("calibration_state_transition") != metadata.get(
        "calibration_state_transition"
    ):
        raise RuntimeError("Seed0 calibration state-transition artifacts differ")
    del precalibration
    del checkpoint_state
    return {
        "checkpoint_sha256": checkpoint_sha,
        "metadata_sha256": metadata_sha,
        "training_result_sha256": training_sha,
        "deployment_completion_sha256": completion_sha,
        "smoke_calibration_resume_proof_sha256": resume_proof_sha,
        "precalibration_ema_state_sha256": precalibration_sha,
        "capability_source_manifest_sha256": file_sha256(capability_manifest_path),
        "capability_manifest_bundle_sha256": capability_manifest[
            "manifest_bundle_sha256"
        ],
        "capability_manifest": capability_manifest,
        "capability_preflight": preflight,
        "odt_prestage": prestage,
        "metadata": metadata,
        "training_result": training,
        "calibration_state_transition": transition_view,
        "smoke_calibration_resume_proof": resume_proof,
        "smoke_producer_artifacts": smoke_validation,
        "conditions": conditions,
    }


def _load_seed0_model() -> tuple[ChiVLA, dict[str, Any]]:
    validated = _validate_capability_inputs()
    metadata = validated["metadata"]
    vocab = metadata.get("vocab")
    if not isinstance(vocab, dict) or _canonical_sha256(vocab) != VOCAB_SHA256:
        raise RuntimeError("Seed0 deployment vocabulary differs")
    config = make_product_config(len(vocab), deployment=True)
    if metadata.get("deployment_config") != config_record(config):
        raise RuntimeError("Seed0 deployment config differs from the frozen Product recipe")
    state = torch.load(
        input_path("inputs/seed0_checkpoint.pt"), map_location="cpu", weights_only=True
    )
    if not isinstance(state, dict):
        raise RuntimeError("Seed0 deployment checkpoint is not a state mapping")
    if metadata.get("checkpoint_state_keys_sha256") != _canonical_sha256(
        sorted(state)
    ):
        raise RuntimeError("Seed0 deployment checkpoint key inventory differs")
    rejected = sorted(
        name for name in state if name.startswith("teacher_head.") or name.startswith("value_head.")
    )
    if rejected:
        raise RuntimeError(f"Seed0 deployment checkpoint contains training-only keys: {rejected}")
    nonfinite = sorted(
        name
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
        and value.is_floating_point()
        and not bool(torch.isfinite(value).all())
    )
    if nonfinite:
        raise RuntimeError(f"Seed0 deployment checkpoint contains nonfinite tensors: {nonfinite}")
    model = ChiVLA(config)
    strict_load_deployment_state(model, state)
    topology = validate_product_model(
        model, deployment=True, require_initialized_norms=True
    )
    if metadata.get("deployment_topology") != topology:
        raise RuntimeError("Loaded Product topology differs from training metadata")
    model = model.to(device="cpu", dtype=torch.float64).eval()
    converted_topology = validate_product_model(
        model, deployment=True, require_initialized_norms=True
    )
    if converted_topology != topology or converted_topology != metadata.get(
        "deployment_topology"
    ):
        raise RuntimeError("Float64 Product conversion changed or stale-guarded topology")
    calibration = metadata.get("post_ema_simultaneous_rational_calibration")
    if not isinstance(calibration, dict):
        raise RuntimeError("Seed0 calibration report is missing")
    initial_values = calibration.get("initializer_running_ms")
    if not isinstance(initial_values, dict):
        raise RuntimeError("Seed0 calibration initializer map is missing")
    calibration_validation = validate_simultaneous_calibration_attestation(
        calibration,
        initial_values=initial_values,
        final_values=active_running_ms_values(model),
        sample_count=int(metadata.get("cache_structure", {}).get("sample_count", -1)),
    )
    if calibration_validation != metadata.get(
        "post_ema_simultaneous_rational_calibration_attestation"
    ):
        raise RuntimeError("Seed0 calibration attestation differs after float64 load")
    return model, {
        **validated,
        "vocab": vocab,
        "topology": topology,
        "calibration_attestation": calibration_validation,
    }


def _replay_panel(model: ChiVLA, vocab: Mapping[str, int]):
    expected_vocab, encode = build_vocab(OFFICIAL_TASK_LANGUAGES)
    instructions = torch.tensor(
        [encode(OFFICIAL_TASK_LANGUAGES[index]) for index in SYNTHETIC_REPLAY_TASKS],
        dtype=torch.long,
    )
    if (
        dict(vocab) != expected_vocab
        or max(vocab.values()) >= model.cfg.vocab_size
        or len(vocab) != model.cfg.vocab_size
    ):
        raise RuntimeError("Replay vocabulary does not match the loaded model")
    generator = torch.Generator(device="cpu").manual_seed(SYNTHETIC_REPLAY_SEED)
    images = torch.rand(
        len(SYNTHETIC_REPLAY_TASKS),
        3,
        model.cfg.image_size,
        model.cfg.image_size,
        dtype=torch.float64,
        generator=generator,
    )
    state = torch.randn(
        len(SYNTHETIC_REPLAY_TASKS),
        model.cfg.state_dim,
        dtype=torch.float64,
        generator=generator,
    ) * 0.1
    embodiments = torch.zeros(len(SYNTHETIC_REPLAY_TASKS), dtype=torch.long)
    physical = physical_batch_from_model_inputs(
        model, images, instructions, state, embodiments
    )
    identity = {
        "kind": "deterministic_functional_replay_not_task_degradation",
        "seed": SYNTHETIC_REPLAY_SEED,
        "official_task_indices": list(SYNTHETIC_REPLAY_TASKS),
        "instruction_token_sha256": hashlib.sha256(
            instructions.contiguous().numpy().tobytes()
        ).hexdigest(),
        "image_sha256": hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest(),
        "state_sha256": hashlib.sha256(state.contiguous().numpy().tobytes()).hexdigest(),
        "embodiment": 0,
        "batch_size": len(SYNTHETIC_REPLAY_TASKS),
    }
    identity["panel_sha256"] = _canonical_sha256(identity)
    return images, instructions, state, embodiments, physical, identity


def _decode_components(
    model: ChiVLA, components: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    head = model.product_head
    center = components[:, : head.m]
    factor_end = head.m * (head.G + 1)
    factors = components[:, head.m : factor_end].reshape(
        components.shape[0], head.G, head.m
    )
    gates = components[:, factor_end:]
    if gates.shape != (components.shape[0], head.G):
        raise RuntimeError("Product all-components observable has the wrong gate width")
    signs = torch.where(gates > 0, torch.ones_like(gates), -torch.ones_like(gates))
    flat = head.action_for_signs(center, factors, signs)
    return flat.reshape(
        components.shape[0], model.cfg.action_horizon, model.cfg.action_dim
    ), signs


def _component_metrics(
    model: ChiVLA,
    components: torch.Tensor,
    source_components: torch.Tensor,
    source_actions: torch.Tensor,
    source_signs: torch.Tensor,
) -> dict[str, Any]:
    actions, signs = _decode_components(model, components)
    delta = (actions - source_actions).abs()
    return {
        "component_relative_error": _relative(components, source_components),
        "decoded_action_relative_error": _relative(actions, source_actions),
        "decoded_action_max_absolute_error": float(delta.max().item()),
        "decoded_action_coordinate_max_absolute_error": [
            float(value)
            for value in delta.amax(dim=(0, 1)).detach().cpu().tolist()
        ],
        "gate_sign_flip_count": int((signs != source_signs).sum().item()),
        "gate_sign_flip_fraction": float((signs != source_signs).double().mean().item()),
    }


def _walk_postorder(root: Any) -> tuple[Any, ...]:
    result: list[Any] = []
    seen: set[int] = set()

    def visit(node: Any) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        for child in node.children:
            visit(child)
        result.append(node)

    visit(root)
    return tuple(result)


def _dooms_order_gates(canonical: Any, diagonal: Any, shape: Mapping[str, int]) -> dict[str, bool]:
    postorder = _walk_postorder(canonical.network.root)
    position = {node.uid: index for index, node in enumerate(postorder)}
    return {
        "algorithm1_child_before_parent": [record.uid for record in canonical.steps]
        == [node.uid for node in postorder]
        and all(position[child.uid] < position[parent.uid] for parent in postorder for child in parent.children),
        "algorithm1_q_replaces_every_local_core": len(canonical.steps) == shape["unique_nodes"]
        and all(record.direct_q_provenance_verified for record in canonical.steps),
        "algorithm1_r_reaches_every_occurrence": all(
            record.parent_occurrences_pushed == record.expected_parent_occurrences
            for record in canonical.steps
        ),
        "algorithm2_root_first_contraction_complete": diagonal.algorithm2_child_messages
        == shape["edge_occurrences"],
        "algorithm3_basis_applied_to_every_bond": diagonal.diagonalized_node_count
        == shape["unique_nodes"],
        "algorithm3_basis_reaches_every_occurrence": diagonal.pushed_parent_occurrences
        == diagonal.expected_parent_occurrences
        == shape["edge_occurrences"] + 1,
    }


def _preflight_mode() -> None:
    output = validate_gate_output(preflight_gate_path())
    manifest = _assert_sources_unchanged()
    _assert_runtime_versions()
    reference = input_path("reference/dooms_xnets_2504.02667.pdf")
    if file_sha256(reference) != DOOMS_REFERENCE_SHA256:
        raise RuntimeError("Pinned Dooms reference differs")
    direct_call_audit = audit_algorithm1_factorization_calls()
    if direct_call_audit.get("prohibited_calls_found"):
        raise RuntimeError("Algorithm 1 direct-call audit did not close")
    model, inputs = _load_seed0_model()
    images, instructions, state, embodiments, _, panel = _replay_panel(
        model, inputs["vocab"]
    )
    state_before = _state_digest(model)
    with torch.inference_mode():
        public_actions, loss = model(images, instructions, state, embodiments)
        pooled = model.pooled_features(images, instructions, state, embodiments)
        center, factors, gates = model.product_head(pooled)
        components = torch.cat(
            (center, factors.reshape(images.shape[0], -1), gates), dim=1
        )
        decoded_actions, signs = _decode_components(model, components)
    if loss is not None:
        raise RuntimeError("Deployment Product model returned a training loss")
    public_decode_equal = torch.equal(public_actions, decoded_actions)
    topology = inputs["topology"]
    conditions = {
        "frozen_runtime": True,
        "exact_manifest": manifest["source_closure"] == _TRANSITIVE_STATIC_AUDIT["source_sha256"],
        "direct_call_audit": not direct_call_audit.get("prohibited_calls_found"),
        "dooms_reference": file_sha256(reference) == DOOMS_REFERENCE_SHA256,
        "seed0_artifact_chain": _all_literal_true(inputs["conditions"]),
        "strict_product_topology": topology.get("product_component_root_projective_width") == 285
        and topology.get("pade_site_count") == EXPECTED_PADE_SITE_COUNT
        and topology.get("linear_action_fallback_present") is False,
        "public_forward_equals_component_decode": public_decode_equal,
        "fixed_sign_boundary_exercised": signs.numel() > 0
        and bool(((signs == -1) | (signs == 1)).all()),
        "source_state_unchanged": state_before == _state_digest(model),
    }
    failed = sorted(name for name, passed in conditions.items() if not passed)
    if failed:
        raise RuntimeError(f"Product exact-ODT preflight failed: {failed}")
    runtime = assert_direct_only_runtime_guard(
        direct_only_runtime_report(), exact_allowed_calls=()
    )
    payload = {
        "schema": f"{SCHEMA}_seed0_checkpoint_integrity_gate",
        "ready_for_bounded_oracle": True,
        "ready_for_full_shared_dag": True,
        "checkpoint_sha256": inputs["checkpoint_sha256"],
        "metadata_sha256": inputs["metadata_sha256"],
        "training_result_sha256": inputs["training_result_sha256"],
        "smoke_calibration_resume_proof_sha256": inputs[
            "smoke_calibration_resume_proof_sha256"
        ],
        "topology": topology,
        "replay_panel": panel,
        "public_forward_component_decode": {
            "bitwise_equal": public_decode_equal,
            "maximum_absolute_error": float((public_actions - decoded_actions).abs().max().item()),
            "positive_sign_fraction": float((signs > 0).double().mean().item()),
            "zero_gate_count": int((gates == 0).sum().item()),
            "sign_rule": "+1 iff gate_logit > 0, else -1",
        },
        "dooms_reference": {
            "path": reference.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": DOOMS_REFERENCE_SHA256,
            "orientation": DOOMS_ORIENTATION,
        },
        "direct_call_audit": direct_call_audit,
        "conditions": conditions,
        "all_conditions_pass": True,
        "runtime": _runtime_environment(),
        "static_audit": _TRANSITIVE_STATIC_AUDIT,
        "runtime_guard_at_import": _RUNTIME_GUARD_AT_IMPORT,
        "runtime_guard_final": runtime,
        "exact_odt_claimed": False,
        "capability_claimed": False,
        "manifest_sha256": file_sha256(PROJECT_ROOT / MANIFEST_PATH),
        "manifest_bundle_sha256": manifest["manifest_bundle_sha256"],
    }
    _assert_sources_unchanged()
    digest = publish_json_exclusive(output, payload)
    print("PREFLIGHT", json.dumps({"path": output.as_posix(), "sha256": digest}), flush=True)


def _junit_summary(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag.endswith("testsuite") else list(root.findall("testsuite"))
    result: dict[str, Any] = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    result["test_names"] = sorted(
        element.attrib["name"]
        for element in root.iter()
        if element.tag.endswith("testcase") and "name" in element.attrib
    )
    return result


def _publish_existing_exclusive(temporary: Path, output: Path) -> str:
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing existing output {output}")
    source_stat = os.lstat(temporary)
    if not stat.S_ISREG(source_stat.st_mode):
        raise RuntimeError("Staged output is not a regular file")
    digest = file_sha256(temporary)
    linked = False
    try:
        os.link(temporary, output, follow_symlinks=False)
        linked = True
        if file_sha256(output) != digest:
            raise RuntimeError("Published output differs from staged bytes")
        os.chmod(output, 0o444, follow_symlinks=False)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return digest
    except BaseException:
        if linked and os.path.lexists(output):
            observed = os.lstat(output)
            if observed.st_dev == source_stat.st_dev and observed.st_ino == source_stat.st_ino:
                output.unlink()
        raise


def _oracle_mode() -> None:
    output = validate_result_output(clone_oracle_result_path())
    junit_output = validate_result_output(clone_oracle_junit_path())
    if output == junit_output:
        raise RuntimeError("Oracle JSON and JUnit outputs must differ")
    manifest = _assert_sources_unchanged()
    _assert_runtime_versions()
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    import pytest

    temporary = junit_output.with_name(
        f".{junit_output.name}.{secrets.token_hex(12)}.tmp"
    )
    if os.path.lexists(temporary):
        raise FileExistsError(f"Refusing stale oracle temporary {temporary}")
    test_path = PROJECT_ROOT / "tests/test_direct_odt_clone_reference.py"
    started = time.perf_counter()
    try:
        exit_code = int(
            pytest.main(
                [
                    "-q",
                    "-c",
                    os.devnull,
                    "-p",
                    "no:cacheprovider",
                    "--noconftest",
                    test_path.as_posix(),
                    f"--junitxml={temporary.as_posix()}",
                ]
            )
        )
        elapsed = time.perf_counter() - started
        junit = _junit_summary(temporary)
        runtime = assert_direct_only_runtime_guard(
            direct_only_runtime_report(),
            exact_allowed_calls=STREAMED_CLONE_ORACLE_RUNTIME_CALLS,
        )
        passed = bool(
            exit_code == 0
            and junit["tests"] == len(EXPECTED_CLONE_TEST_NAMES)
            and junit["failures"] == 0
            and junit["errors"] == 0
            and junit["skipped"] == 0
            and frozenset(junit["test_names"]) == EXPECTED_CLONE_TEST_NAMES
            and runtime["prohibited_attempt_count"] == 0
        )
        if not passed:
            raise RuntimeError("Bounded clone oracle did not pass its exact test inventory")
        _assert_sources_unchanged()
        junit_sha = _publish_existing_exclusive(temporary, junit_output)
        payload = {
            "schema": f"{SCHEMA}_bounded_clone_oracle",
            "claim_boundary": (
                "Bounded independent no-memo clone comparison after every canonicalization "
                "step. Whole-policy shared-DAG completion is required separately."
            ),
            "test_path": test_path.relative_to(PROJECT_ROOT).as_posix(),
            "pytest_exit_code": exit_code,
            "junit": {"path": junit_output.as_posix(), "sha256": junit_sha, **junit},
            "elapsed_seconds": elapsed,
            "peak_rss_mb": _peak_rss_mb(),
            "bounded_independent_clone_oracle_passed": True,
            "dooms_algorithms_1_to_3_stepwise_checked": True,
            "full_shared_dag_checked": False,
            "static_audit": _TRANSITIVE_STATIC_AUDIT,
            "runtime_guard_at_import": _RUNTIME_GUARD_AT_IMPORT,
            "runtime_guard_final": runtime,
            "manifest_sha256": file_sha256(PROJECT_ROOT / MANIFEST_PATH),
            "manifest_bundle_sha256": manifest["manifest_bundle_sha256"],
            "exact_global_decomposability_certified": False,
        }
        try:
            digest = publish_json_exclusive(output, payload)
        except BaseException:
            if os.path.lexists(junit_output):
                junit_output.chmod(0o644)
                junit_output.unlink()
            raise
        print("ORACLE", json.dumps({"path": output.as_posix(), "sha256": digest}), flush=True)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def _lane_test_mode() -> None:
    output = validate_result_output(lane_test_result_path())
    junit_output = validate_result_output(lane_test_junit_path())
    if output == junit_output:
        raise RuntimeError("Lane-test JSON and JUnit outputs must differ")
    manifest = _assert_sources_unchanged()
    _assert_runtime_versions()
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    import pytest

    temporary = junit_output.with_name(
        f".{junit_output.name}.{secrets.token_hex(12)}.tmp"
    )
    if os.path.lexists(temporary):
        raise FileExistsError(f"Refusing stale lane-test temporary {temporary}")
    test_paths = tuple(
        PROJECT_ROOT / relative
        for relative in (
            "tests/test_product_rational_exact_odt_lane.py",
            "tests/test_direct_odt_truncation.py",
            "tests/test_implicit_sparse_projective_odt_vla.py",
        )
    )
    started = time.perf_counter()
    try:
        exit_code = int(
            pytest.main(
                [
                    "-q",
                    "-c",
                    os.devnull,
                    "-p",
                    "no:cacheprovider",
                    "--noconftest",
                    *(path.as_posix() for path in test_paths),
                    f"--junitxml={temporary.as_posix()}",
                ]
            )
        )
        elapsed = time.perf_counter() - started
        junit = _junit_summary(temporary)
        runtime = assert_direct_only_runtime_guard(
            direct_only_runtime_report(),
            exact_allowed_calls={
                DIRECT_RQ_RUNTIME_CALL,
                PRODUCTION_ALGORITHM3_RUNTIME_CALL,
                PRODUCTION_CLONE_QR_RUNTIME_CALL,
                PRODUCTION_SHARED_CLONE_ALGORITHM3_RUNTIME_CALL,
            },
        )
        passed = bool(
            exit_code == 0
            and junit["tests"] == len(EXPECTED_LANE_TEST_NAMES)
            and junit["failures"] == 0
            and junit["errors"] == 0
            and junit["skipped"] == 0
            and frozenset(junit["test_names"]) == EXPECTED_LANE_TEST_NAMES
            and runtime["prohibited_attempt_count"] == 0
        )
        if not passed:
            raise RuntimeError("Exact Product lane tests did not pass their fixed inventory")
        _assert_sources_unchanged()
        junit_sha = _publish_existing_exclusive(temporary, junit_output)
        payload = {
            "schema": f"{SCHEMA}_lane_tests",
            "test_paths": [
                path.relative_to(PROJECT_ROOT).as_posix() for path in test_paths
            ],
            "pytest_exit_code": exit_code,
            "junit": {"path": junit_output.as_posix(), "sha256": junit_sha, **junit},
            "elapsed_seconds": elapsed,
            "peak_rss_mb": _peak_rss_mb(),
            "exact_inventory_passed_without_skips": True,
            "projective_rowwise_scale_regression_passed": True,
            "static_negative_mutations_passed": True,
            "static_audit": _TRANSITIVE_STATIC_AUDIT,
            "runtime_guard_at_import": _RUNTIME_GUARD_AT_IMPORT,
            "runtime_guard_final": runtime,
            "manifest_sha256": file_sha256(PROJECT_ROOT / MANIFEST_PATH),
            "manifest_bundle_sha256": manifest["manifest_bundle_sha256"],
            "exact_global_decomposability_certified": False,
        }
        try:
            digest = publish_json_exclusive(output, payload)
        except BaseException:
            if os.path.lexists(junit_output):
                junit_output.chmod(0o644)
                junit_output.unlink()
            raise
        print("LANE_TESTS", json.dumps({"path": output.as_posix(), "sha256": digest}), flush=True)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def _validated_preflight_gate(inputs: Mapping[str, Any]) -> dict[str, Any]:
    path = preflight_gate_path()
    payload = read_json_object(path, label="Product exact-ODT preflight gate")
    conditions = payload.get("conditions")
    if (
        set(payload)
        != {
            "schema",
            "ready_for_bounded_oracle",
            "ready_for_full_shared_dag",
            "checkpoint_sha256",
            "metadata_sha256",
            "training_result_sha256",
            "smoke_calibration_resume_proof_sha256",
            "topology",
            "replay_panel",
            "public_forward_component_decode",
            "dooms_reference",
            "direct_call_audit",
            "conditions",
            "all_conditions_pass",
            "runtime",
            "static_audit",
            "runtime_guard_at_import",
            "runtime_guard_final",
            "exact_odt_claimed",
            "capability_claimed",
            "manifest_sha256",
            "manifest_bundle_sha256",
        }
        or payload.get("schema") != f"{SCHEMA}_seed0_checkpoint_integrity_gate"
        or payload.get("ready_for_bounded_oracle") is not True
        or payload.get("ready_for_full_shared_dag") is not True
        or payload.get("all_conditions_pass") is not True
        or not _all_literal_true(
            conditions,
            exact_keys={
                "frozen_runtime",
                "exact_manifest",
                "direct_call_audit",
                "dooms_reference",
                "seed0_artifact_chain",
                "strict_product_topology",
                "public_forward_equals_component_decode",
                "fixed_sign_boundary_exercised",
                "source_state_unchanged",
            },
        )
        or payload.get("checkpoint_sha256") != inputs["checkpoint_sha256"]
        or payload.get("metadata_sha256") != inputs["metadata_sha256"]
        or payload.get("training_result_sha256") != inputs["training_result_sha256"]
        or payload.get("smoke_calibration_resume_proof_sha256")
        != inputs["smoke_calibration_resume_proof_sha256"]
        or not _json_type_exact_equal(payload.get("topology"), inputs["topology"])
        or not _json_type_exact_equal(
            payload.get("dooms_reference"),
            {
                "path": "reference/dooms_xnets_2504.02667.pdf",
                "sha256": DOOMS_REFERENCE_SHA256,
                "orientation": DOOMS_ORIENTATION,
            },
        )
        or not _json_type_exact_equal(payload.get("runtime"), _runtime_environment())
        or not _json_type_exact_equal(
            payload.get("static_audit"), _TRANSITIVE_STATIC_AUDIT
        )
        or not _stored_runtime_exact(payload.get("runtime_guard_at_import"), set())
        or not _stored_runtime_exact(payload.get("runtime_guard_final"), set())
        or payload.get("exact_odt_claimed") is not False
        or payload.get("capability_claimed") is not False
        or payload.get("manifest_sha256") != file_sha256(PROJECT_ROOT / MANIFEST_PATH)
        or payload.get("manifest_bundle_sha256")
        != verify_manifest(PROJECT_ROOT / MANIFEST_PATH)["manifest_bundle_sha256"]
    ):
        raise RuntimeError("Product exact-ODT preflight gate differs")
    return {"path": path.as_posix(), "sha256": file_sha256(path)}


def _validated_clone_oracle() -> dict[str, Any]:
    path = clone_oracle_result_path()
    junit_path = clone_oracle_junit_path()
    payload = read_json_object(path, label="Product bounded clone oracle")
    junit = payload.get("junit", {})
    if (
        set(payload)
        != {
            "schema",
            "claim_boundary",
            "test_path",
            "pytest_exit_code",
            "junit",
            "elapsed_seconds",
            "peak_rss_mb",
            "bounded_independent_clone_oracle_passed",
            "dooms_algorithms_1_to_3_stepwise_checked",
            "full_shared_dag_checked",
            "static_audit",
            "runtime_guard_at_import",
            "runtime_guard_final",
            "manifest_sha256",
            "manifest_bundle_sha256",
            "exact_global_decomposability_certified",
        }
        or set(junit)
        != {"path", "sha256", "tests", "failures", "errors", "skipped", "test_names"}
        or payload.get("schema") != f"{SCHEMA}_bounded_clone_oracle"
        or payload.get("claim_boundary")
        != (
            "Bounded independent no-memo clone comparison after every canonicalization "
            "step. Whole-policy shared-DAG completion is required separately."
        )
        or payload.get("test_path") != "tests/test_direct_odt_clone_reference.py"
        or type(payload.get("pytest_exit_code")) is not int
        or payload.get("pytest_exit_code") != 0
        or type(payload.get("elapsed_seconds")) is not float
        or not math.isfinite(payload.get("elapsed_seconds"))
        or payload.get("elapsed_seconds") < 0.0
        or type(payload.get("peak_rss_mb")) is not float
        or not math.isfinite(payload.get("peak_rss_mb"))
        or payload.get("peak_rss_mb") <= 0.0
        or payload.get("bounded_independent_clone_oracle_passed") is not True
        or payload.get("dooms_algorithms_1_to_3_stepwise_checked") is not True
        or payload.get("full_shared_dag_checked") is not False
        or payload.get("exact_global_decomposability_certified") is not False
        or junit.get("path") != junit_path.as_posix()
        or junit.get("sha256") != file_sha256(junit_path)
        or type(junit.get("tests")) is not int
        or junit.get("tests") != len(EXPECTED_CLONE_TEST_NAMES)
        or type(junit.get("failures")) is not int
        or junit.get("failures") != 0
        or type(junit.get("errors")) is not int
        or junit.get("errors") != 0
        or type(junit.get("skipped")) is not int
        or junit.get("skipped") != 0
        or not _json_type_exact_equal(
            junit.get("test_names"), sorted(EXPECTED_CLONE_TEST_NAMES)
        )
        or not _json_type_exact_equal(
            payload.get("static_audit"), _TRANSITIVE_STATIC_AUDIT
        )
        or not _stored_runtime_exact(payload.get("runtime_guard_at_import"), set())
        or not _stored_runtime_exact(
            payload.get("runtime_guard_final"), set(STREAMED_CLONE_ORACLE_RUNTIME_CALLS)
        )
        or payload.get("manifest_sha256") != file_sha256(PROJECT_ROOT / MANIFEST_PATH)
        or payload.get("manifest_bundle_sha256")
        != verify_manifest(PROJECT_ROOT / MANIFEST_PATH)["manifest_bundle_sha256"]
    ):
        raise RuntimeError("Product bounded clone oracle artifact differs")
    return {
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "junit_path": junit_path.as_posix(),
        "junit_sha256": file_sha256(junit_path),
    }


def _validated_lane_tests() -> dict[str, Any]:
    path = lane_test_result_path()
    junit_path = lane_test_junit_path()
    payload = read_json_object(path, label="Product exact-ODT lane tests")
    junit = payload.get("junit", {})
    if (
        set(payload)
        != {
            "schema",
            "test_paths",
            "pytest_exit_code",
            "junit",
            "elapsed_seconds",
            "peak_rss_mb",
            "exact_inventory_passed_without_skips",
            "projective_rowwise_scale_regression_passed",
            "static_negative_mutations_passed",
            "static_audit",
            "runtime_guard_at_import",
            "runtime_guard_final",
            "manifest_sha256",
            "manifest_bundle_sha256",
            "exact_global_decomposability_certified",
        }
        or set(junit)
        != {"path", "sha256", "tests", "failures", "errors", "skipped", "test_names"}
        or payload.get("schema") != f"{SCHEMA}_lane_tests"
        or not _json_type_exact_equal(
            payload.get("test_paths"),
            [
                "tests/test_product_rational_exact_odt_lane.py",
                "tests/test_direct_odt_truncation.py",
                "tests/test_implicit_sparse_projective_odt_vla.py",
            ],
        )
        or type(payload.get("pytest_exit_code")) is not int
        or payload.get("pytest_exit_code") != 0
        or type(payload.get("elapsed_seconds")) is not float
        or not math.isfinite(payload.get("elapsed_seconds"))
        or payload.get("elapsed_seconds") < 0.0
        or type(payload.get("peak_rss_mb")) is not float
        or not math.isfinite(payload.get("peak_rss_mb"))
        or payload.get("peak_rss_mb") <= 0.0
        or payload.get("exact_inventory_passed_without_skips") is not True
        or payload.get("projective_rowwise_scale_regression_passed") is not True
        or payload.get("static_negative_mutations_passed") is not True
        or payload.get("exact_global_decomposability_certified") is not False
        or junit.get("path") != junit_path.as_posix()
        or junit.get("sha256") != file_sha256(junit_path)
        or type(junit.get("tests")) is not int
        or junit.get("tests") != len(EXPECTED_LANE_TEST_NAMES)
        or type(junit.get("failures")) is not int
        or junit.get("failures") != 0
        or type(junit.get("errors")) is not int
        or junit.get("errors") != 0
        or type(junit.get("skipped")) is not int
        or junit.get("skipped") != 0
        or not _json_type_exact_equal(
            junit.get("test_names"), sorted(EXPECTED_LANE_TEST_NAMES)
        )
        or not _json_type_exact_equal(
            payload.get("static_audit"), _TRANSITIVE_STATIC_AUDIT
        )
        or not _stored_runtime_exact(payload.get("runtime_guard_at_import"), set())
        or not _stored_runtime_exact(
            payload.get("runtime_guard_final"),
            {
                DIRECT_RQ_RUNTIME_CALL,
                PRODUCTION_ALGORITHM3_RUNTIME_CALL,
                STREAMED_QR_RUNTIME_CALL,
                TRIANGULAR_RUNTIME_CALL,
            },
        )
        or payload.get("manifest_sha256") != file_sha256(PROJECT_ROOT / MANIFEST_PATH)
        or payload.get("manifest_bundle_sha256")
        != verify_manifest(PROJECT_ROOT / MANIFEST_PATH)["manifest_bundle_sha256"]
    ):
        raise RuntimeError("Product exact-ODT lane-test artifact differs")
    return {
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "junit_path": junit_path.as_posix(),
        "junit_sha256": file_sha256(junit_path),
    }


def _algorithm1_summary(canonical: Any) -> tuple[dict[str, Any], dict[str, bool], set[str]]:
    performed = tuple(
        step.per_step_function_replay_error
        for step in canonical.steps
        if step.per_step_function_replay_performed
    )
    maximum_local = max(
        (step.local_scaled_reconstruction_relative_error for step in canonical.steps),
        default=0.0,
    )
    maximum_absorption = max(
        (step.maximum_absorption_scaled_relative_error for step in canonical.steps),
        default=0.0,
    )
    normal_form = validate_canonical_exponent_normal_form(canonical.network)
    streamed = tuple(step for step in canonical.steps if step.method == DIRECT_RQ_METHOD)
    expected_columns = sum(int(step.unfolding_shape[1]) for step in canonical.steps)
    expected_streamed_replays = sum(
        step.direct_q_provenance_stage_count for step in streamed
    )
    postorder = _walk_postorder(canonical.network.root)
    postorder_uids = [int(node.uid) for node in postorder]
    step_uids = [int(step.uid) for step in canonical.steps]
    positions = {uid: index for index, uid in enumerate(postorder_uids)}
    order_violations = sum(
        positions[child.uid] >= positions[parent.uid]
        for parent in postorder
        for child in parent.children
    )
    methods = sorted({step.method for step in canonical.steps})
    summary = {
        "step_count": len(canonical.steps),
        "factorization_methods": methods,
        "factorization_method_counts": {
            method: sum(step.method == method for step in canonical.steps)
            for method in methods
        },
        "canonical_postorder_uid_sha256": _canonical_sha256(postorder_uids),
        "algorithm1_step_uid_sha256": _canonical_sha256(step_uids),
        "child_parent_order_violation_count": order_violations,
        "final_projective_replay_relative_error": canonical.final_projective_replay_error,
        "full_projective_replay_evaluations": canonical.full_projective_replay_evaluations,
        "per_step_replays_performed": len(performed),
        "maximum_local_scaled_reconstruction_relative_error": maximum_local,
        "maximum_absorption_scaled_relative_error": maximum_absorption,
        "maximum_local_reconstruction_exponent_delta": max(
            (abs(step.local_reconstruction_exponent_delta) for step in canonical.steps),
            default=0,
        ),
        "maximum_absorption_exponent_delta": max(
            (abs(step.maximum_absorption_exponent_delta) for step in canonical.steps),
            default=0,
        ),
        "push_ledger": [
            sum(step.parent_occurrences_pushed for step in canonical.steps),
            sum(step.expected_parent_occurrences for step in canonical.steps),
        ],
        "scale_sensitive_occurrence_ledger": [
            sum(step.scale_sensitive_occurrences_verified for step in canonical.steps),
            sum(step.parent_occurrences_pushed for step in canonical.steps),
        ],
        "canonical_exponent_normal_form": normal_form,
        "direct_q_provenance": {
            "verified_steps": sum(step.direct_q_provenance_verified for step in canonical.steps),
            "expected_steps": len(canonical.steps),
            "streamed_certificates": sum(
                step.direct_q_provenance_method == STREAMED_DIRECT_Q_PROVENANCE_METHOD
                for step in canonical.steps
            ),
            "streamed_steps": len(streamed),
            "columns_compared": sum(step.direct_q_columns_compared for step in canonical.steps),
            "expected_columns": expected_columns,
            "streamed_replay_calls": canonical.telemetry.streamed_direct_q_replay_qr_factorizations,
            "expected_streamed_replay_calls": expected_streamed_replays,
            "maximum_compact_q_relative_error": max(
                (step.direct_q_compact_relative_error for step in canonical.steps),
                default=0.0,
            ),
            "maximum_reconstruction_relative_error": max(
                (step.direct_q_reconstruction_relative_error for step in canonical.steps),
                default=0.0,
            ),
        },
        "telemetry": telemetry_dict(canonical.telemetry),
    }
    gates = {
        "all_nodes_use_direct_reduced_rq": all(
            step.method in {DIRECT_RQ_METHOD, UNARY_RQ_METHOD, RECTANGULAR_RQ_METHOD}
            for step in canonical.steps
        ),
        "final_projective_replay": canonical.final_projective_replay_error
        < REPLAY_RELATIVE_TOLERANCE,
        "production_replay_schedule": len(performed) == 0
        and canonical.full_projective_replay_evaluations == 2,
        "local_scale_certificates": maximum_local < LOCAL_SCALE_CERTIFICATION_TOLERANCE
        and maximum_absorption < LOCAL_SCALE_CERTIFICATION_TOLERANCE
        and all(
            step.local_reconstruction_exponent_delta == 0
            and step.maximum_absorption_exponent_delta == 0
            and step.scale_sensitive_occurrences_verified == step.parent_occurrences_pushed
            for step in canonical.steps
        ),
        "canonical_exponent_normal_form": normal_form["all_core_exponents_zero"]
        and normal_form["head_exponent_matches_ledger"],
        "direct_q_provenance_complete": all(
            step.direct_q_provenance_method
            and step.direct_q_provenance_verified
            and step.direct_q_columns_compared == int(step.unfolding_shape[1])
            and step.direct_q_compact_relative_error <= DIRECT_Q_PROVENANCE_TOLERANCE
            and step.direct_q_reconstruction_relative_error <= DIRECT_Q_PROVENANCE_TOLERANCE
            and step.direct_q_conditioning_accepted
            for step in canonical.steps
        ),
        "streamed_direct_q_ledger": sum(
            step.direct_q_provenance_method == STREAMED_DIRECT_Q_PROVENANCE_METHOD
            for step in canonical.steps
        )
        == len(streamed)
        and all(
            step.direct_q_provenance_stage_count
            == step.direct_q_provenance_column_blocks_compared
            == step.streamed_column_blocks
            and (step.direct_q_retained_transition_elements > 0)
            == (step.direct_q_provenance_stage_count > 1)
            for step in streamed
        ),
        "direct_q_telemetry_complete": canonical.telemetry.direct_q_provenance_certificates
        == len(canonical.steps)
        and canonical.telemetry.streamed_direct_q_provenance_certificates == len(streamed)
        and canonical.telemetry.direct_q_columns_compared == expected_columns
        and canonical.telemetry.streamed_direct_q_replay_qr_factorizations
        == expected_streamed_replays,
        "every_factor_occurrence_pushed": all(
            step.parent_occurrences_pushed == step.expected_parent_occurrences
            for step in canonical.steps
        ),
        "rank_deficiency_route_is_shape_only": all(
            step.direct_q_provenance_method == STREAMED_DIRECT_Q_PROVENANCE_METHOD
            or step.direct_q_conditioning_threshold == 0.0
            for step in canonical.steps
        ),
    }
    runtime_calls = {DIRECT_RQ_RUNTIME_CALL}
    if streamed:
        runtime_calls.update({STREAMED_QR_RUNTIME_CALL, TRIANGULAR_RUNTIME_CALL})
    return summary, gates, runtime_calls


def _publish_algorithm1_progress(
    completed: int,
    total: int,
    started: float,
    record: Any,
    base: Mapping[str, Any],
) -> None:
    output = validate_progress_output(
        PROJECT_ROOT
        / "athena/results/product_pade_rational_exact_odt_v1_progress"
        / f"algorithm1_step_{completed:07d}_of_{total:07d}.json"
    )
    manifest = _assert_sources_unchanged()
    payload = {
        "schema": f"{SCHEMA}_algorithm1_progress",
        **dict(base),
        "completed_steps": completed,
        "total_steps": total,
        "elapsed_seconds": time.perf_counter() - started,
        "last_step": {
            "uid": record.uid,
            "label": record.label,
            "method": record.method,
            "unfolding_shape": list(record.unfolding_shape),
            "direct_q_provenance_method": record.direct_q_provenance_method,
            "direct_q_columns_compared": record.direct_q_columns_compared,
            "parent_occurrence_ledger": [
                record.parent_occurrences_pushed,
                record.expected_parent_occurrences,
            ],
        },
        "peak_rss_mb": _peak_rss_mb(),
        "manifest_bundle_sha256": manifest["manifest_bundle_sha256"],
        "exact_global_decomposability_certified": False,
        "compression_claimed": False,
        "capability_claimed": False,
    }
    publish_json_exclusive(output, payload)


def _full_mode() -> None:
    output = validate_result_output(full_result_path())
    full_rank_output = validate_result_output(full_rank_progress_path())
    existing_progress = tuple(
        (PROJECT_ROOT / "athena/results/product_pade_rational_exact_odt_v1_progress").glob(
            "algorithm1_step_*.json"
        )
    )
    if existing_progress:
        raise FileExistsError("Refusing a full run with existing Algorithm 1 progress artifacts")
    manifest = _assert_sources_unchanged()
    _assert_runtime_versions()
    direct_call_audit = audit_algorithm1_factorization_calls()
    if direct_call_audit.get("prohibited_calls_found"):
        raise RuntimeError("Algorithm 1 direct-call audit did not close")
    model, inputs = _load_seed0_model()
    preflight = _validated_preflight_gate(inputs)
    clone_oracle = _validated_clone_oracle()
    lane_tests = _validated_lane_tests()
    _, _, _, _, physical, panel = _replay_panel(model, inputs["vocab"])
    model_state_before = _state_digest(model)
    started_total = time.perf_counter()
    started = time.perf_counter()
    oracle = compile_full_vla_projective_dag(model)
    compile_seconds = time.perf_counter() - started
    if (
        oracle.observable_kind != PRODUCT_COMPONENTS_OBSERVABLE
        or oracle.product_signs is not None
        or oracle.network.claim_boundary != PRODUCT_COMPONENTS_FULL_VLA_CLAIM_BOUNDARY
    ):
        raise RuntimeError("Full Product compiler did not preserve the all-components root")
    structure = full_vla_structure_statistics(oracle)
    initial_shape = implicit_shape_statistics(oracle.network)
    route_inventory = predict_direct_rq_route_inventory(oracle.network)
    if route_inventory["streamed_tall_unrepresentable_count"] != 0:
        raise RuntimeError("Pre-sweep route prediction found an unsupported unfolding shape")

    started = time.perf_counter()
    source_components = source_full_vla_observable(oracle, physical)
    source_actions = source_full_vla_output(model, physical)
    compiled_components = evaluate_full_vla_observable(oracle, physical)
    compiled_actions = evaluate_full_vla_quotient(oracle, physical)
    source_decoded_actions, source_signs = _decode_components(model, source_components)
    source_evaluation_seconds = time.perf_counter() - started
    source_gates = {
        "all_components_product_root": structure["observable_kind"]
        == PRODUCT_COMPONENTS_OBSERVABLE
        and structure["root_projective_width"] == 285
        and structure["observable_euclidean_width"] == 284
        and structure["product_signs"] is None,
        "compiled_component_replay": _relative(compiled_components, source_components)
        < REPLAY_RELATIVE_TOLERANCE,
        "compiled_public_action_replay": _relative(compiled_actions, source_actions)
        < REPLAY_RELATIVE_TOLERANCE,
        "source_component_decode_matches_public_forward": _relative(
            source_decoded_actions, source_actions
        )
        < CLONE_RELATIVE_TOLERANCE,
        "all_physical_sources_present_once": structure["physical_source_count"]
        == structure["physical_leaf_count"]
        == 97,
        "heterogeneous_ingress_unpadded": structure["maximum_physical_source_width"] == 192
        and all(value != model.cfg.dim for value in structure["physical_source_widths"].values()),
        "all_pade_sites_compiled": structure["active_pade_sites"]
        == EXPECTED_PADE_SITE_COUNT,
        "no_dense_clone_core": initial_shape["dense_clone_nodes"] == 0,
        "route_inventory_exact_before_sweep": route_inventory["schema"]
        == "exact_postorder_structural_direct_rq_routes_v1"
        and route_inventory["simulated_unique_node_count"] == initial_shape["unique_nodes"]
        and route_inventory["bounded_retained_q_candidate_count"]
        + route_inventory["streamed_cp_candidate_count"]
        == initial_shape["cp_binary_nodes"]
        and route_inventory["streamed_tall_unrepresentable_count"] == 0,
    }
    failed = sorted(name for name, passed in source_gates.items() if not passed)
    if failed:
        raise RuntimeError(f"Source/compiler Product replay failed before Algorithm 1: {failed}")
    source_replay = {
        "component_relative_error": _relative(compiled_components, source_components),
        "public_action_relative_error": _relative(compiled_actions, source_actions),
        "source_decode_relative_error": _relative(source_decoded_actions, source_actions),
        "source_decode_bitwise_equal": torch.equal(source_decoded_actions, source_actions),
        "source_positive_sign_fraction": float((source_signs > 0).double().mean().item()),
        "source_zero_gate_count": int(
            (source_components[:, model.product_head.m * (model.product_head.G + 1) :] == 0)
            .sum()
            .item()
        ),
    }
    raw = physical_source_mapping(model, physical)
    numerical_execution = {
        "runtime": _runtime_environment(),
        "model_parameter_dtypes": sorted({str(value.dtype) for value in model.parameters()}),
        "model_floating_buffer_dtypes": sorted(
            {str(value.dtype) for value in model.buffers() if value.is_floating_point()}
        ),
        "physical_source_dtypes": sorted({str(value.dtype) for value in raw.values()}),
        "running_ms_guard_revalidated_after_float64_conversion": inputs["topology"]
        .get("running_ms_fail_closed_guard", {})
        .get("guarded_module_count")
        == 74,
    }
    if (
        numerical_execution["model_parameter_dtypes"] != ["torch.float64"]
        or numerical_execution["model_floating_buffer_dtypes"] != ["torch.float64"]
        or numerical_execution["physical_source_dtypes"] != ["torch.float64"]
        or numerical_execution["running_ms_guard_revalidated_after_float64_conversion"]
        is not True
    ):
        raise RuntimeError("Product exact-ODT numerical execution boundary differs")
    progress_base = {
        "checkpoint_sha256": inputs["checkpoint_sha256"],
        "preflight_sha256": preflight["sha256"],
        "clone_oracle_sha256": clone_oracle["sha256"],
        "lane_tests_sha256": lane_tests["sha256"],
        "replay_panel_sha256": panel["panel_sha256"],
        "source_replay": source_replay,
        "initial_shape": initial_shape,
    }
    started = time.perf_counter()

    def algorithm1_progress(_work: Any, _node: Any, _factor: Any, record: Any) -> None:
        completed = record.step + 1
        total = initial_shape["unique_nodes"]
        if completed % 100_000 == 0 or completed == total:
            _publish_algorithm1_progress(completed, total, started, record, progress_base)

    canonical = canonicalize_implicit_dag_direct_rq(
        oracle.network,
        replay_inputs=raw,
        block_size=4096,
        replay_each_step=False,
        copy_network=False,
        step_callback=algorithm1_progress,
    )
    algorithm1_seconds = time.perf_counter() - started
    canonical_shape = implicit_shape_statistics(canonical.network)
    algorithm1, algorithm1_gates, runtime_calls = _algorithm1_summary(canonical)
    if canonical_shape["unique_nodes"] != initial_shape["unique_nodes"]:
        algorithm1_gates["node_inventory_preserved"] = False
    else:
        algorithm1_gates["node_inventory_preserved"] = True
    failed = sorted(name for name, passed in algorithm1_gates.items() if not passed)
    if failed:
        raise RuntimeError(f"Product Algorithm 1 failed before environment contraction: {failed}")

    bank = CompactRankBank.for_network(canonical.network)
    started = time.perf_counter()
    diagonal = diagonalize_implicit_dag_full_rank(
        canonical.network,
        replay_inputs=raw,
        copy_network=False,
        stream_pre_evd_environments=True,
        retain_eigenvalues=False,
        retain_post_evd_environments=False,
        compact_spectrum_callback=bank,
    )
    algorithms2_and3_seconds = time.perf_counter() - started
    runtime_calls.add(PRODUCTION_ALGORITHM3_RUNTIME_CALL)
    dooms_gates = _dooms_order_gates(canonical, diagonal, canonical_shape)
    full_rank_gates = {
        **source_gates,
        **algorithm1_gates,
        **dooms_gates,
        "algorithm2_every_edge_message": diagonal.algorithm2_child_messages
        == canonical_shape["edge_occurrences"],
        "algorithm3_every_node": diagonal.diagonalized_node_count
        == canonical_shape["unique_nodes"],
        "algorithm3_every_occurrence": diagonal.pushed_parent_occurrences
        == diagonal.expected_parent_occurrences
        == canonical_shape["edge_occurrences"] + 1,
        "algorithm3_replay": math.isfinite(diagonal.replay_relative_error)
        and diagonal.replay_relative_error < REPLAY_RELATIVE_TOLERANCE,
        "algorithm3_recontracted_environments_diagonal": math.isfinite(
            diagonal.maximum_recontracted_offdiagonal_ratio
        )
        and diagonal.maximum_recontracted_offdiagonal_ratio
        < OFFDIAGONAL_RELATIVE_TOLERANCE,
        "zero_full_spectrum_retention": diagonal.eigenvalues == ()
        and not diagonal.eigenvalue_spectra_retained,
        "compact_rank_bank_complete": bank.complete,
        "source_model_state_unchanged": model_state_before == _state_digest(model),
    }
    assert_full_vla_norm_buffers_unchanged(oracle)
    algorithm1_telemetry = algorithm1["telemetry"]
    streamed_steps = algorithm1["direct_q_provenance"]["streamed_steps"]
    nonstreamed_direct_rq_calls = (
        algorithm1_telemetry["local_direct_rq_factorizations"] - streamed_steps
    )
    numerical_execution["controlled_operation_counts"] = {
        "direct_rq_householder_qr_calls": nonstreamed_direct_rq_calls,
        "streamed_tsqr_householder_qr_calls": 2
        * algorithm1_telemetry["streamed_direct_q_replay_qr_factorizations"],
        "triangular_solve_calls": streamed_steps,
        "environment_evd_calls": diagonal.diagonalized_node_count,
        "total_controlled_calls": algorithm1_telemetry["householder_qr_kernel_calls"]
        + streamed_steps
        + diagonal.diagonalized_node_count,
    }
    runtime_full_rank = assert_direct_only_runtime_guard(
        direct_only_runtime_report(), exact_allowed_calls=runtime_calls
    )
    full_rank_gates["runtime_guard_exact"] = runtime_full_rank["prohibited_attempt_count"] == 0
    failed = sorted(name for name, passed in full_rank_gates.items() if not passed)
    if failed:
        raise RuntimeError(f"Product full-rank exact gates failed before compression: {failed}")

    full_rank_certificate = {
        "schema": f"{SCHEMA}_algorithm3_full_rank_complete_before_compression",
        "checkpoint_sha256": inputs["checkpoint_sha256"],
        "metadata_sha256": inputs["metadata_sha256"],
        "training_result_sha256": inputs["training_result_sha256"],
        "smoke_calibration_resume_proof_sha256": inputs[
            "smoke_calibration_resume_proof_sha256"
        ],
        "preflight": preflight,
        "bounded_clone_oracle": clone_oracle,
        "exact_lane_tests": lane_tests,
        "dooms_reference": {
            "sha256": DOOMS_REFERENCE_SHA256,
            "orientation": DOOMS_ORIENTATION,
        },
        "replay_panel": panel,
        "source_replay": source_replay,
        "structure": structure,
        "initial_shape": initial_shape,
        "canonical_shape": canonical_shape,
        "route_inventory": route_inventory,
        "algorithm1": algorithm1,
        "algorithm2_and3": {
            "explicit_downstream_contraction": True,
            "streamed_root_first": True,
            "child_messages": diagonal.algorithm2_child_messages,
            "eigenbasis_order": "descending environment eigenvalue",
            "diagonalized_nodes": diagonal.diagonalized_node_count,
            "occurrence_push_ledger": [
                diagonal.pushed_parent_occurrences,
                diagonal.expected_parent_occurrences,
            ],
            "replay_relative_error": diagonal.replay_relative_error,
            "maximum_recontracted_offdiagonal_ratio": diagonal.maximum_recontracted_offdiagonal_ratio,
            "full_spectra_retained": diagonal.eigenvalue_spectra_retained,
            "post_environment_records_retained": False,
        },
        "numerical_execution": numerical_execution,
        "timings_seconds": {
            "compile": compile_seconds,
            "source_evaluation": source_evaluation_seconds,
            "algorithm1": algorithm1_seconds,
            "algorithms2_and3": algorithms2_and3_seconds,
            "through_full_rank": time.perf_counter() - started_total,
        },
        "peak_rss_mb": _peak_rss_mb(),
        "full_rank_gates": full_rank_gates,
        "all_full_rank_gates_pass": True,
        "runtime_guard": runtime_full_rank,
        "static_audit": _TRANSITIVE_STATIC_AUDIT,
        "manifest_sha256": file_sha256(PROJECT_ROOT / MANIFEST_PATH),
        "manifest_bundle_sha256": manifest["manifest_bundle_sha256"],
        "bounded_clone_oracle_hash_joined": True,
        "exact_lane_test_hash_joined": True,
        "canonical_direct_odt_algorithms_1_to_3_completed": True,
        "exact_global_decomposability_certified": True,
        "compression_claimed": False,
        "capability_claimed": False,
        "capable_global_decomposable_vla_claimed": False,
    }
    _assert_sources_unchanged()
    full_rank_validation = _validate_full_rank_certificate_payload(
        full_rank_certificate,
        inputs=inputs,
        manifest=manifest,
        preflight=preflight,
        clone=clone_oracle,
        lane_tests=lane_tests,
    )
    full_rank_sha = publish_json_exclusive(full_rank_output, full_rank_certificate)

    compact_census = _json_safe_compact_census(bank.finish())
    groups = compact_census.get("groups", {})
    if not all(
        groups.get(name, {}).get("node_count", 0) > 0
        for name in ("product_center", "product_factors", "product_gates")
    ):
        raise RuntimeError("Compact rank census missed a Product component family")
    full_storage = implicit_storage_elements(diagonal.network)
    curves: dict[str, Any] = {}
    for target in COMPACT_TRACE_RETENTION_TARGETS:
        plan = bank.plan(target)
        projected_storage = projected_prefix_storage_elements(diagonal, plan)
        started = time.perf_counter()
        components = evaluate_diagonal_prefix_quotient(diagonal, raw, plan)
        curve = {
            "target": target,
            "retained_bond_dimensions": sum(plan[uid] for uid in plan),
            "original_bond_dimensions": sum(plan.expected_dimension(uid) for uid in plan),
            "projected_storage_elements": projected_storage,
            "projected_storage_fraction": projected_storage / full_storage,
            "prefix_metrics": _component_metrics(
                model,
                components,
                source_components,
                source_actions,
                source_signs,
            ),
            "prefix_evaluation_seconds": time.perf_counter() - started,
        }
        control_started = time.perf_counter()
        try:
            control_components = evaluate_diagonal_suffix_quotient(diagonal, raw, plan)
        except ValueError as error:
            message = str(error)
            if "denominator" not in message and "zero/nonfinite coordinates" not in message:
                raise
            curve["matched_width_trailing_control"] = {
                "valid_projective_chart": False,
                "rejection": message,
                "evaluation_seconds": time.perf_counter() - control_started,
            }
        else:
            curve["matched_width_trailing_control"] = {
                "valid_projective_chart": True,
                "metrics": _component_metrics(
                    model,
                    control_components,
                    source_components,
                    source_actions,
                    source_signs,
                ),
                "evaluation_seconds": time.perf_counter() - control_started,
            }
        curves[str(target)] = curve

    physical_plan = bank.plan(PHYSICAL_MATERIALIZATION_TARGET)
    physical_projected_storage = projected_prefix_storage_elements(
        diagonal, physical_plan
    )
    masked_pair = evaluate_diagonal_prefixes(diagonal, raw, physical_plan)
    masked_components = evaluate_diagonal_prefix_quotient(diagonal, raw, physical_plan)
    masked_actions, masked_signs = _decode_components(model, masked_components)
    materialization_started = time.perf_counter()
    source_model_forward_calls = 0

    def reject_source_model_forward(_module: Any, _arguments: Any) -> None:
        nonlocal source_model_forward_calls
        source_model_forward_calls += 1
        raise RuntimeError(
            "Source-model forward fallback is forbidden during materialized-DAG execution"
        )

    source_model_guard = model.register_forward_pre_hook(reject_source_model_forward)
    try:
        materialized = truncate_diagonal_prefixes(
            diagonal, physical_plan, copy_network=False
        )
        materialized_pair = evaluate_projective_boundary(materialized.network, raw)
        materialized_components = evaluate_boundary_quotient(materialized.network, raw)
        materialized_actions, materialized_signs = _decode_components(
            model, materialized_components
        )
    finally:
        source_model_guard.remove()
    materialization = {
        "target": PHYSICAL_MATERIALIZATION_TARGET,
        "performed_after_all_masked_curves": True,
        "copy_network": False,
        "executor_kind": "materialized_implicit_projective_dag",
        "materialized_tensor_network_executed": True,
        "source_model_forward_fallback_forbidden": True,
        "source_model_forward_calls_during_materialized_execution": (
            source_model_forward_calls
        ),
        "projective_pair_relative_error_against_masked": _relative(
            materialized_pair, masked_pair
        ),
        "projective_pair_gauge_invariant_relative_error_against_masked": (
            _projective_batch_relative_error(materialized_pair, masked_pair)
        ),
        "component_relative_error_against_masked": _relative(
            materialized_components, masked_components
        ),
        "decoded_action_relative_error_against_masked": _relative(
            materialized_actions, masked_actions
        ),
        "decoded_action_coordinate_max_absolute_error_against_masked": [
            float(value)
            for value in (materialized_actions - masked_actions)
            .abs()
            .amax(dim=(0, 1))
            .detach()
            .cpu()
            .tolist()
        ],
        "gate_sign_mismatch_count_against_masked": int(
            (materialized_signs != masked_signs).sum().item()
        ),
        "original_storage_elements": materialized.original_storage_elements,
        "projected_storage_elements": physical_projected_storage,
        "materialized_storage_elements": materialized.truncated_storage_elements,
        "allocated_storage_elements": materialized.allocated_storage_elements,
        "reduced_bonds": materialized.reduced_bonds,
        "expected_occurrence_slices": materialized.expected_occurrence_slices,
        "applied_occurrence_slices": materialized.applied_occurrence_slices,
        "elapsed_seconds": time.perf_counter() - materialization_started,
    }
    materialization_gates = {
        "materialized_executor_has_no_source_model_forward_fallback": materialization[
            "materialized_tensor_network_executed"
        ]
        and materialization["source_model_forward_fallback_forbidden"]
        and materialization["source_model_forward_calls_during_materialized_execution"]
        == 0,
        "projective_pair_matches_masked_up_to_rowwise_scale": materialization[
            "projective_pair_gauge_invariant_relative_error_against_masked"
        ]
        < CLONE_RELATIVE_TOLERANCE,
        "components_match_masked": materialization["component_relative_error_against_masked"]
        < CLONE_RELATIVE_TOLERANCE,
        "actions_match_masked": materialization["decoded_action_relative_error_against_masked"]
        < CLONE_RELATIVE_TOLERANCE,
        "signs_match_masked": materialization["gate_sign_mismatch_count_against_masked"] == 0,
        "storage_origin_matches": materialization["original_storage_elements"] == full_storage,
        "projected_materialized_allocated_equal": materialization["projected_storage_elements"]
        == materialization["materialized_storage_elements"]
        == materialization["allocated_storage_elements"],
        "strict_physical_storage_reduction": materialization["materialized_storage_elements"]
        < full_storage,
        "bond_reduction_nonempty": materialization["reduced_bonds"] > 0,
        "every_occurrence_sliced": materialization["expected_occurrence_slices"]
        == materialization["applied_occurrence_slices"]
        and materialization["applied_occurrence_slices"] > 0,
    }
    if not all(materialization_gates.values()):
        failed = sorted(name for name, passed in materialization_gates.items() if not passed)
        raise RuntimeError(f"Physical Product compression materialization failed: {failed}")
    assert_full_vla_norm_buffers_unchanged(oracle)
    if model_state_before != _state_digest(model):
        raise RuntimeError("Source Product model changed during exact ODT")
    runtime_final = assert_direct_only_runtime_guard(
        direct_only_runtime_report(), exact_allowed_calls=runtime_calls
    )
    final_manifest = _assert_sources_unchanged()
    result = {
        "schema": f"{SCHEMA}_full_with_same_sweep_compression",
        "checkpoint_sha256": inputs["checkpoint_sha256"],
        "smoke_calibration_resume_proof_sha256": inputs[
            "smoke_calibration_resume_proof_sha256"
        ],
        "full_rank_certificate": {
            "path": full_rank_output.as_posix(),
            "sha256": full_rank_sha,
        },
        "bounded_clone_oracle": clone_oracle,
        "exact_lane_tests": lane_tests,
        "preflight": preflight,
        "replay_panel": panel,
        "full_rank_gates": full_rank_gates,
        "all_full_rank_gates_pass": True,
        "compact_census": compact_census,
        "canonical_storage_elements": full_storage,
        "compression_curves": curves,
        "compression_curves_are_functional_not_task_degradation": True,
        "physical_materialization": materialization,
        "physical_materialization_gates": materialization_gates,
        "all_compression_gates_pass": True,
        "timings_seconds": {
            **full_rank_certificate["timings_seconds"],
            "total": time.perf_counter() - started_total,
        },
        "peak_rss_mb": _peak_rss_mb(),
        "runtime_guard_final": runtime_final,
        "static_audit": _TRANSITIVE_STATIC_AUDIT,
        "manifest_sha256": file_sha256(PROJECT_ROOT / MANIFEST_PATH),
        "manifest_bundle_sha256": final_manifest["manifest_bundle_sha256"],
        "canonical_direct_odt_algorithms_1_to_3_completed": True,
        "bounded_clone_oracle_hash_joined": True,
        "exact_lane_test_hash_joined": True,
        "exact_global_decomposability_certified": True,
        "compression_claimed_after_full_rank_certificate": True,
        "capability_claimed": False,
        "capable_global_decomposable_vla_claimed": False,
    }
    _validate_compression_result_payload(
        result,
        full_rank=full_rank_certificate,
        full_rank_sha256=full_rank_sha,
        full_rank_validation=full_rank_validation,
        inputs=inputs,
        manifest=final_manifest,
        preflight=preflight,
        clone=clone_oracle,
        lane_tests=lane_tests,
    )
    digest = publish_json_exclusive(output, result)
    print("FULL", json.dumps({"path": output.as_posix(), "sha256": digest}), flush=True)


def _composite_mode() -> None:
    output = validate_result_output(composite_result_path())
    manifest = _assert_sources_unchanged()
    _assert_runtime_versions()
    inputs = _validate_capability_inputs()
    aggregate_path = capability_aggregate_path()
    gate_path = capability_gate_path()
    capability_evidence = validate_capability_aggregate_evidence(
        CAPABILITY_STAGE_ROOT / CAPABILITY_RUN_ROOT,
        manifest=inputs["capability_manifest"],
        manifest_sha256=inputs["capability_source_manifest_sha256"],
    )
    aggregate = capability_evidence["aggregate"]
    aggregate_start_sha = capability_evidence["aggregate_sha256"]
    capability_gate = capability_evidence["gate"]
    gate_start_sha = capability_evidence["gate_sha256"]
    full_path = full_result_path()
    full, full_start_sha, full_identity = _read_json_physical_once(
        full_path, label="Product full exact-ODT result"
    )
    full_rank_path = full_rank_progress_path()
    full_rank, full_rank_start_sha, full_rank_identity = _read_json_physical_once(
        full_rank_path, label="Product full-rank certificate"
    )
    clone = _validated_clone_oracle()
    lane_tests = _validated_lane_tests()
    preflight = _validated_preflight_gate(inputs)
    full_rank_validation = _validate_full_rank_certificate_payload(
        full_rank,
        inputs=inputs,
        manifest=manifest,
        preflight=preflight,
        clone=clone,
        lane_tests=lane_tests,
    )
    compression_validation = _validate_compression_result_payload(
        full,
        full_rank=full_rank,
        full_rank_sha256=full_rank_start_sha,
        full_rank_validation=full_rank_validation,
        inputs=inputs,
        manifest=manifest,
        preflight=preflight,
        clone=clone,
        lane_tests=lane_tests,
    )
    seed0 = aggregate.get("seed_summaries", {}).get("0", {})
    capability_conditions = capability_gate.get("conditions")
    exact_conditions = full_rank_validation["full_rank_gates"]
    compression_gates = compression_validation["physical_materialization_gates"]
    expected_seed0_artifacts = {
        "checkpoint": inputs["checkpoint_sha256"],
        "metadata": inputs["metadata_sha256"],
        "training_result": inputs["training_result_sha256"],
        "precalibration_ema_state": inputs["precalibration_ema_state_sha256"],
        "deployment_completion": inputs["deployment_completion_sha256"],
    }
    conditions = {
        "capability_aggregate_schema": aggregate.get("schema")
        == f"{CAPABILITY_SCHEMA}_three_seed_aggregate",
        "capability_gate_schema": capability_gate.get("schema")
        == f"{CAPABILITY_SCHEMA}_seed0_capability_gate",
        "primary_seed_preregistered": aggregate.get("preregistered_primary_seed") == 0
        and capability_gate.get("primary_seed") == 0
        and capability_gate.get("preregistered_before_evaluation") is True,
        "primary_floor_preregistered": aggregate.get("preregistered_primary_floor")
        == capability_gate.get("required_overall")
        == PRIMARY_CAPABILITY_FLOOR,
        "primary_capability_pass": seed0.get("episode_count") == 500
        and seed0.get("overall", -math.inf) >= PRIMARY_CAPABILITY_FLOOR
        and aggregate.get("capability_gate_pass") is True
        and capability_gate.get("approved_for_capable_global_claim") is True,
        "all_capability_conditions": _all_literal_true(
            capability_conditions,
            exact_keys={
                "primary_seed_is_preregistered_seed0",
                "primary_overall_at_least_floor",
                "all_three_seed_reports_present",
                "all_four_shards_per_seed_present_and_unique",
                "exactly_500_episodes_per_seed",
                "canonical_task_and_initial_state_protocol",
                "no_nonfinite_or_decode_failures",
                "all_simulator_transitions_strictly_validated",
                "no_best_seed_post_selection",
            },
        ),
        "all_three_seed_reports_present": set(aggregate.get("seed_summaries", {}))
        == {"0", "1", "2"},
        "capability_aggregate_hash_joined": capability_gate.get("aggregate_sha256")
        == aggregate_start_sha,
        "capability_source_authority_joined": aggregate.get(
            "source_manifest_sha256"
        )
        == capability_gate.get("source_manifest_sha256")
        == inputs["capability_source_manifest_sha256"]
        and aggregate.get("manifest_bundle_sha256")
        == capability_gate.get("manifest_bundle_sha256")
        == inputs["capability_manifest_bundle_sha256"],
        "capability_seed0_full_artifact_inventory_joined": aggregate.get(
            "training_artifact_sha256", {}
        ).get("0")
        == expected_seed0_artifacts,
        "checkpoint_hash_joined": capability_gate.get("checkpoint_sha256")
        == full.get("checkpoint_sha256")
        == inputs["checkpoint_sha256"],
        "metadata_hash_joined": capability_gate.get("metadata_sha256")
        == inputs["metadata_sha256"],
        "training_result_hash_joined": capability_gate.get("training_result_sha256")
        == inputs["training_result_sha256"],
        "deployment_completion_hash_joined": capability_gate.get(
            "deployment_completion_sha256"
        )
        == inputs["deployment_completion_sha256"],
        "precalibration_hash_joined": capability_gate.get(
            "precalibration_ema_state_sha256"
        )
        == inputs["metadata"].get(
            "training_only_precalibration_ema_state", {}
        ).get("sha256"),
        "smoke_resume_proof_hash_joined": full.get(
            "smoke_calibration_resume_proof_sha256"
        )
        == full_rank.get("smoke_calibration_resume_proof_sha256")
        == inputs["smoke_calibration_resume_proof_sha256"],
        "full_result_schema": full.get("schema")
        == f"{SCHEMA}_full_with_same_sweep_compression",
        "full_rank_schema": full_rank.get("schema")
        == f"{SCHEMA}_algorithm3_full_rank_complete_before_compression",
        "full_rank_hash_joined": full.get("full_rank_certificate", {}).get("sha256")
        == full_rank_start_sha,
        "bounded_clone_hash_joined": full.get("bounded_clone_oracle", {}).get("sha256")
        == clone["sha256"],
        "exact_lane_test_hash_joined": full.get("exact_lane_tests", {}).get("sha256")
        == lane_tests["sha256"],
        "all_exact_gates": _all_literal_true(
            exact_conditions, exact_keys=_FULL_RANK_GATE_KEYS
        )
        and full_rank.get("all_full_rank_gates_pass") is True
        and full_rank_validation.get("validated") is True,
        "global_decomposability_certified": full_rank.get(
            "exact_global_decomposability_certified"
        )
        is True
        and full.get("exact_global_decomposability_certified") is True,
        "compression_after_exact_certificate": full.get(
            "compression_claimed_after_full_rank_certificate"
        )
        is True
        and _all_literal_true(
            compression_gates, exact_keys=_MATERIALIZATION_GATE_KEYS
        )
        and compression_validation.get("validated") is True,
        "capability_simulator_kept_outside_odt_numerics": aggregate.get(
            "claim_boundary"
        )
        == DEPLOYMENT_CLAIM_BOUNDARY,
    }
    failed = sorted(name for name, passed in conditions.items() if not passed)
    if failed:
        raise RuntimeError(f"Capable Product global-ODT composite failed: {failed}")
    runtime = assert_direct_only_runtime_guard(
        direct_only_runtime_report(), exact_allowed_calls=()
    )
    verify_capability_evidence_snapshots(capability_evidence["snapshots"])
    for path, label, digest, identity in (
        (full_path, "Full exact-ODT result", full_start_sha, full_identity),
        (
            full_rank_path,
            "Full-rank certificate",
            full_rank_start_sha,
            full_rank_identity,
        ),
    ):
        _verify_json_snapshot(path, label=label, digest=digest, identity=identity)
    final_manifest = _assert_sources_unchanged()
    payload = {
        "schema": f"{SCHEMA}_capable_global_attestation",
        "claim": (
            "The preregistered seed0 ProductRoutingHead VLA passed the fixed official "
            "capability floor and its unchanged deployment checkpoint completed canonical "
            "direct Algorithms 1 through 3 as one all-components shared DAG."
        ),
        "checkpoint_sha256": inputs["checkpoint_sha256"],
        "smoke_calibration_resume_proof_sha256": inputs[
            "smoke_calibration_resume_proof_sha256"
        ],
        "capability": {
            "aggregate_path": aggregate_path.as_posix(),
            "aggregate_sha256": aggregate_start_sha,
            "gate_path": gate_path.as_posix(),
            "gate_sha256": gate_start_sha,
            "primary_seed": 0,
            "episode_count": seed0["episode_count"],
            "overall": seed0["overall"],
            "required_floor": PRIMARY_CAPABILITY_FLOOR,
            "all_three_seed_summaries": aggregate["seed_summaries"],
            "simulator_external_to_odt": True,
            "odt_runtime_compliance_claimed_for_simulator": False,
        },
        "exact_odt": {
            "full_result_path": full_path.as_posix(),
            "full_result_sha256": full_start_sha,
            "full_rank_certificate_path": full_rank_path.as_posix(),
            "full_rank_certificate_sha256": full_rank_start_sha,
            "bounded_clone_oracle": clone,
            "exact_lane_tests": lane_tests,
            "all_components_product_root": True,
            "fixed_sign_decode_outside_tensor_network": True,
            "dooms_reference_sha256": DOOMS_REFERENCE_SHA256,
            "dooms_orientation": DOOMS_ORIENTATION,
        },
        "compression": {
            "evaluated_only_after_full_rank_certificate": True,
            "same_algorithm3_sweep_rank_bank": True,
            "physical_materialization_target": PHYSICAL_MATERIALIZATION_TARGET,
            "physical_materialization_gates": compression_gates,
            "functional_panel_only": True,
            "task_degradation_claimed": False,
        },
        "conditions": conditions,
        "all_conditions_pass": True,
        "capability_certified": True,
        "exact_global_decomposability_certified": True,
        "capable_global_decomposable_vla_certified": True,
        "runtime_guard": runtime,
        "static_audit": _TRANSITIVE_STATIC_AUDIT,
        "manifest_sha256": file_sha256(PROJECT_ROOT / MANIFEST_PATH),
        "manifest_bundle_sha256": final_manifest["manifest_bundle_sha256"],
    }
    digest = publish_json_exclusive(output, payload)
    print("COMPOSITE", json.dumps({"path": output.as_posix(), "sha256": digest}), flush=True)


def main() -> None:
    args = parse_args()
    if not _RUNNING_AS_ENTRYPOINT or _PREIMPORT_SOURCE_VERIFICATION is None:
        raise RuntimeError("Product exact ODT must use its authenticated entrypoint")
    if tuple(_EXPECTED_SOURCE_CLOSURE) != tuple(SOURCE_CLOSURE):
        raise RuntimeError("Runner source tuple differs from the protocol source tuple")
    if args.mode == "preflight":
        _preflight_mode()
    elif args.mode == "oracle":
        _oracle_mode()
    elif args.mode == "lane-tests":
        _lane_test_mode()
    elif args.mode == "full":
        _full_mode()
    elif args.mode == "composite":
        _composite_mode()
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
