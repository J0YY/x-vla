from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import attest_capable_linear_rtx6000_cross_stage as lane


_TEST_BOUNDED_SHA256 = "a" * 64
_TEST_ODT_SOURCES = {
    relative: f"{index + 1:064x}"
    for index, relative in enumerate(sorted(lane.EXPECTED_BOUNDED_SHARED_CORES))
}
_TEST_PATCHED_ENTRYPOINTS = [f"guarded.entrypoint.{index}" for index in range(87)]
_TEST_PATCHED_ENTRYPOINTS_SHA256 = lane._canonical_sha256(
    _TEST_PATCHED_ENTRYPOINTS
)


def _episode_records(start: int, stop: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for task in range(start, stop):
        for episode in range(50):
            success = (task + episode) % 2 == 0
            records.append(
                {
                    "task_index": task,
                    "episode": episode,
                    "initial_state_sha256": f"{task * 50 + episode:064x}",
                    "success": success,
                    "steps": 1 if success else lane.capability_lane.MAX_STEPS,
                    "terminated_without_success": False,
                    "elapsed_s": 0.25,
                }
            )
    return records


def _join_records() -> tuple[dict[str, object], dict[str, object]]:
    capability = {
        "checkpoint_sha256": lane.EXPECTED_CHECKPOINT_SHA256,
        "config_identity": lane.EXPECTED_CONFIG_IDENTITY,
        "trials": 500,
        "successes": 400,
        "gates": {"capability": True},
        "source_manifest_sha256": "1" * 64,
        "stage_ledger_sha256": "2" * 64,
    }
    full = {
        "checkpoint_sha256": lane.EXPECTED_CHECKPOINT_SHA256,
        "config_identity": lane.EXPECTED_CONFIG_IDENTITY,
        "gates": {"exact": True},
        "source_manifest_sha256": "3" * 64,
        "stage_ledger_sha256": "4" * 64,
        "b1c_historical_capability_evidence_used_as_capability_authority": False,
        "b1c_r7_control_used_as_exact_odt_authority": False,
    }
    return capability, full


def _full_scientific_record() -> dict[str, object]:
    node_count = lane.EXPECTED_B1C_STRUCTURE["unique_nodes"]
    edge_count = lane.EXPECTED_B1C_STRUCTURE["edge_occurrences"]
    streamed_count = lane.EXPECTED_B1C_ROUTE_INVENTORY[
        "streamed_cp_candidate_count"
    ]
    bounded_count = lane.EXPECTED_B1C_ROUTE_INVENTORY[
        "bounded_retained_q_candidate_count"
    ]
    pushes = edge_count + 1
    columns = 2_616_809_755
    replay_calls = 91_331
    maximum_compact_error = 1e-12
    maximum_reconstruction_error = 2e-15
    return {
        "schema": "xvla_capable_linear_b1c0_full_rank_direct_odt_v1",
        "claim_boundary": lane.EXPECTED_B1C_CLAIM_BOUNDARY,
        "deployment_interface_scope": copy.deepcopy(
            lane.EXPECTED_B1C_DEPLOYMENT_SCOPE
        ),
        "numerical_precision_boundary": copy.deepcopy(
            lane.EXPECTED_B1C_PRECISION_BOUNDARY
        ),
        "checkpoint": {
            "checkpoint_sha256": lane.EXPECTED_CHECKPOINT_SHA256,
            "checkpoint_bytes": lane.EXPECTED_B1C_CHECKPOINT_BYTES,
            "state_key_count": 540,
            "parameter_count": 20_137_352,
            "normalization_site_count": 74,
            "active_normalization_site_count": 73,
            "compiler_active_normalization_site_count": 73,
            "compiler_active_snapshot_site_count": 73,
            "compiler_active_set_equals_loaded_active_set": True,
            "normalization_buffer_inventory": copy.deepcopy(
                lane.EXPECTED_B1C_NORMALIZATION_BUFFER_INVENTORY
            ),
            "config_identity": copy.deepcopy(lane.EXPECTED_CONFIG_IDENTITY),
            "vocab_size": 26,
        },
        "inactive_vision_norm_sentinel": {
            "site": "vision.norm_out",
            "armed_through_full_rank": True,
            "call_count": 0,
            "total_site_count": 74,
            "inactive_sites": ["vision.norm_out"],
            "active_site_count": 73,
            "active_call_count": 146,
            "every_active_site_call_count": 2,
        },
        "capability_evidence": {},
        "current_combined_unit_attestation": {},
        "previous_exact_checkpoint_control": {},
        "dooms_reference_sha256": lane.EXPECTED_DOOMS_REFERENCE_SHA256,
        "dooms_algorithm_order": copy.deepcopy(lane.EXPECTED_B1C_ALGORITHM_ORDER),
        "replay_panel": copy.deepcopy(lane.EXPECTED_B1C_REPLAY_PANEL),
        "structure": copy.deepcopy(lane.EXPECTED_B1C_STRUCTURE),
        "initial_shape": copy.deepcopy(lane.EXPECTED_B1C_INITIAL_SHAPE),
        "canonical_shape": {
            "unique_nodes": node_count,
            "edge_occurrences": edge_count,
            "cp_binary_nodes": streamed_count,
            "reduced_q_binary_nodes": bounded_count,
            "dense_clone_nodes": 0,
            "unary_nodes": 711_975,
            "heterogeneous_physical_sources": 97,
            "maximum_local_bond_dimension": 385,
            "maximum_cp_rank": 1_153,
            "maximum_reduced_q_elements": 3_739_329,
            "fused_token_feature_dimension": 13_129,
        },
        "route_inventory": copy.deepcopy(lane.EXPECTED_B1C_ROUTE_INVENTORY),
        "source_replay": {
            "observable_relative_error": 1e-12,
            "public_action_relative_error": 1e-12,
            "observable_matches_public_action_relative_error": 1e-13,
        },
        "algorithm1": {
            "step_count": node_count,
            "factorization_methods": list(lane.EXPECTED_B1C_FACTORIZATION_METHODS),
            "final_projective_replay_relative_error": 1e-12,
            "full_projective_replay_evaluations": 2,
            "per_step_replays_performed": 0,
            "maximum_local_scaled_reconstruction_relative_error": 2e-15,
            "maximum_absorption_scaled_relative_error": 0.0,
            "push_ledger": [pushes, pushes],
            "scale_sensitive_occurrence_ledger": [pushes, pushes],
            "canonical_exponent_normal_form": {
                "core_count": node_count,
                "all_core_exponents_zero": True,
                "head_binary_exponent": lane.EXPECTED_B1C_HEAD_BINARY_EXPONENT,
                "head_exponent_matches_ledger": True,
                "cp_direct_q_certificates": streamed_count,
                "cp_direct_q_columns": lane.EXPECTED_B1C_CP_DIRECT_Q_COLUMNS,
                "streamed_compact_cp_direct_q_certificates": streamed_count,
                "streamed_compact_cp_direct_q_columns": (
                    lane.EXPECTED_B1C_CP_DIRECT_Q_COLUMNS
                ),
                "all_streamed_compact_cp_q_has_direct_q_provenance": True,
            },
            "direct_q_provenance": {
                "verified_steps": node_count,
                "expected_steps": node_count,
                "streamed_steps": streamed_count,
                "columns_compared": columns,
                "expected_columns": columns,
                "streamed_replay_calls": replay_calls,
                "expected_streamed_replay_calls": replay_calls,
                "maximum_compact_q_relative_error": maximum_compact_error,
                "maximum_reconstruction_relative_error": (
                    maximum_reconstruction_error
                ),
            },
            "telemetry": {
                **lane.EXPECTED_ALGORITHM1_STATIC_TELEMETRY,
                "maximum_direct_q_compact_relative_error": maximum_compact_error,
                "maximum_direct_q_reconstruction_relative_error": (
                    maximum_reconstruction_error
                ),
            },
        },
        "algorithms2_and3": {
            "algorithm2_child_messages": edge_count,
            "algorithm3_diagonalized_nodes": node_count,
            "algorithm3_occurrence_ledger": [pushes, pushes],
            "algorithm3_replay_relative_error": 1e-12,
            "maximum_recontracted_offdiagonal_ratio": 1e-12,
            "full_spectra_retained": False,
            "post_environment_records_retained": False,
        },
        "bounded_clone_and_regression_attestation": {
            "path": "inputs/direct_odt_truncation_unit_v2.json",
            "sha256": _TEST_BOUNDED_SHA256,
            "test_count": 52,
            "required_oracles": [
                "test_bounded_rank_deficient_385x386_retains_direct_q_and_matches_clone_every_step",
                "test_unchanged_six_pade_attention_ffn_residual_block_matches_every_step",
                "test_streamed_production_algorithms2_and3_match_separated_clone_reference",
                "test_every_prefix_step_matches_the_no_memo_clone_and_masked_replay",
                "test_materialization_projective_comparison_ignores_rowwise_radial_scale",
            ],
            "gates": {
                name: True for name in lane.EXPECTED_BOUNDED_ORACLE_GATES
            },
            "shared_core_sha256": dict(_TEST_ODT_SOURCES),
        },
        "bounded_clone_oracle_hash_joined": True,
        "full_rank_gates": {
            name: True for name in lane.EXPECTED_B1C_FULL_GATE_NAMES
        },
        "all_full_rank_gates_pass": True,
        "canonical_direct_odt_algorithms_1_to_3_completed": True,
        "exact_learned_neural_dag_global_decomposability_certified": True,
        "historical_capability_path_association_recorded": True,
        "historical_capability_sha256_bound_to_current_checkpoint": False,
        "compression_claimed": False,
        "runtime_guard": {
            "installed": True,
            "patched_entrypoint_count": 87,
            "patched_entrypoints": list(_TEST_PATCHED_ENTRYPOINTS),
            "allowed_call_count": lane.EXPECTED_B1C_RUNTIME_CALL_COUNT,
            "allowed_calls": sorted(lane.EXPECTED_B1C_RUNTIME_CALLS),
            "prohibited_attempt_count": 0,
            "prohibited_attempts": [],
        },
        "timings_seconds": {
            "compile": 1.0,
            "source_evaluation": 2.0,
            "algorithm1": 3.0,
            "algorithms2_and3": 4.0,
            "through_full_rank": 11.0,
        },
        "peak_rss_mb": 1024.0,
        "static_audit": {
            "scope": "transitive_local_import_closure",
            "entrypoints": list(lane.EXPECTED_B1C_STATIC_ENTRYPOINTS),
            "source_count": len(_TEST_ODT_SOURCES),
            "source_sha256": dict(_TEST_ODT_SOURCES),
            "call_site_count": 8_310,
            "direct_qr_required": True,
            "direct_qr_call_sites": 4,
            "prohibited_calls_found": [],
            "prohibited_self_overlap_sites": [],
            "guarded_dormant_spectral_norm_sites": [],
            "duplicate_top_level_definition_sites": [],
        },
        "source_manifest": {
            "path": lane.B1C_SOURCE_MANIFEST.as_posix(),
            "sha256": lane.EXPECTED_B1C_SOURCE_MANIFEST_SHA256,
            "source_count": len(_TEST_ODT_SOURCES),
            "source_sha256": dict(_TEST_ODT_SOURCES),
        },
    }


def test_episode_inventory_accepts_only_ordered_complete_records() -> None:
    records = _episode_records(0, 3)
    exact, pairs, successes, per_task = lane._episode_records_are_exact(
        records, 0, 3, max_steps=lane.capability_lane.MAX_STEPS
    )
    assert exact
    assert pairs[0] == (0, 0) and pairs[-1] == (2, 49)
    assert successes == 75
    assert per_task == {"0": 25, "1": 25, "2": 25}


def test_full_scientific_record_recomputes_every_gate() -> None:
    gates = lane._b1c_full_scientific_evidence_gates(
        _full_scientific_record(),
        bounded_unit_sha256=_TEST_BOUNDED_SHA256,
        odt_sources=_TEST_ODT_SOURCES,
        patched_entrypoints_sha256=_TEST_PATCHED_ENTRYPOINTS_SHA256,
    )
    assert gates
    assert all(gates.values()), gates


@pytest.mark.parametrize(
    ("mutation", "failed_gate"),
    (
        ("structure", "structure_exact"),
        ("top_level_extra", "top_level_schema_exact"),
        ("checkpoint_extra", "checkpoint_schema_and_identity_exact"),
        ("checkpoint_norm", "checkpoint_schema_and_identity_exact"),
        ("checkpoint_numeric_alias", "checkpoint_schema_and_identity_exact"),
        ("sentinel_extra", "inactive_sentinel_schema_and_evidence_exact"),
        ("sentinel_count", "inactive_sentinel_schema_and_evidence_exact"),
        ("sentinel_bool_alias", "inactive_sentinel_schema_and_evidence_exact"),
        ("source_manifest_extra", "source_manifest_schema_and_authority_exact"),
        ("source_manifest_count", "source_manifest_schema_and_authority_exact"),
        ("static_extra", "static_audit_schema_and_authority_exact"),
        ("static_call_count", "static_audit_schema_and_authority_exact"),
        ("precision", "numerical_precision_boundary_exact"),
        ("precision_bool_alias", "numerical_precision_boundary_exact"),
        ("replay_panel", "replay_panel_exact"),
        ("replay_panel_float_alias", "replay_panel_exact"),
        ("timing_missing", "timings_and_peak_valid"),
        ("timing_relation", "timings_and_peak_valid"),
        ("peak", "timings_and_peak_valid"),
        ("dooms_reference", "dooms_reference_exact"),
        ("canonical_shape", "canonical_shape_reconciled"),
        ("canonical_max_bond_low", "canonical_shape_reconciled"),
        ("canonical_max_bond_high", "canonical_shape_reconciled"),
        ("canonical_max_bond_type", "canonical_shape_reconciled"),
        ("canonical_max_cp_low", "canonical_shape_reconciled"),
        ("canonical_max_cp_high", "canonical_shape_reconciled"),
        ("canonical_max_cp_type", "canonical_shape_reconciled"),
        ("initial_bool_alias", "initial_shape_exact"),
        ("route", "route_inventory_exact"),
        ("route_bool_alias", "route_inventory_exact"),
        ("algorithm1_method", "algorithm1_methods_direct"),
        ("algorithm1_extra", "algorithm1_schema_exact"),
        ("algorithm1_push", "algorithm1_every_occurrence"),
        ("algorithm1_reconstruction", "algorithm1_reconstruction"),
        ("algorithm1_provenance", "algorithm1_provenance"),
        ("head_exponent_low", "algorithm1_provenance"),
        ("head_exponent_high", "algorithm1_provenance"),
        ("cp_columns_low", "algorithm1_provenance"),
        ("cp_columns_high", "algorithm1_provenance"),
        ("telemetry_missing", "algorithm1_telemetry"),
        ("telemetry_extra", "algorithm1_telemetry"),
        ("unused_counter_nonzero", "algorithm1_telemetry"),
        ("telemetry_bool_alias", "algorithm1_telemetry"),
        (
            "algorithm2_message_count",
            "algorithm2_explicit_contraction_and_eigendecomposition",
        ),
        (
            "algorithm2_runtime_call",
            "runtime_factorization_and_eigendecomposition_ledger",
        ),
        ("algorithms23_extra", "algorithm2_explicit_contraction_and_eigendecomposition"),
        (
            "runtime_patched_entrypoint",
            "runtime_factorization_and_eigendecomposition_ledger",
        ),
        ("runtime_count_low", "runtime_factorization_and_eigendecomposition_ledger"),
        ("runtime_count_high", "runtime_factorization_and_eigendecomposition_ledger"),
        ("runtime_bool_alias", "runtime_factorization_and_eigendecomposition_ledger"),
        ("algorithm3_node_count", "algorithm3_every_node_and_occurrence"),
        ("algorithm3_occurrence", "algorithm3_every_node_and_occurrence"),
        ("algorithm3_replay", "algorithm3_replay"),
        ("algorithm3_offdiagonal", "algorithm3_environment_diagonality"),
        ("scope", "deployment_scope_exact"),
        ("scope_bool_alias", "deployment_scope_exact"),
        ("claim_boundary", "claim_boundary_exact"),
        ("bounded_oracle_hash", "bounded_clone_oracle_joined"),
        ("bounded_oracle_inventory", "bounded_clone_oracle_joined"),
        ("bounded_oracle_extra", "bounded_clone_oracle_joined"),
        ("bounded_shared_core", "bounded_clone_oracle_joined"),
        ("gate_schema", "full_gate_schema_exact"),
    ),
)
def test_full_scientific_record_rejects_nested_mutation(
    mutation: str, failed_gate: str
) -> None:
    value = _full_scientific_record()
    if mutation == "structure":
        value["structure"]["unique_nodes"] += 1
    elif mutation == "top_level_extra":
        value["unexpected"] = None
    elif mutation == "checkpoint_extra":
        value["checkpoint"]["unexpected"] = None
    elif mutation == "checkpoint_norm":
        value["checkpoint"]["normalization_buffer_inventory"][
            "initialized_true_count"
        ] = 74
    elif mutation == "checkpoint_numeric_alias":
        value["checkpoint"]["vocab_size"] = 26.0
    elif mutation == "sentinel_extra":
        value["inactive_vision_norm_sentinel"]["unexpected"] = None
    elif mutation == "sentinel_count":
        value["inactive_vision_norm_sentinel"]["call_count"] = 1
    elif mutation == "sentinel_bool_alias":
        value["inactive_vision_norm_sentinel"]["call_count"] = False
    elif mutation == "source_manifest_extra":
        value["source_manifest"]["unexpected"] = None
    elif mutation == "source_manifest_count":
        value["source_manifest"]["source_count"] += 1
    elif mutation == "static_extra":
        value["static_audit"]["unexpected"] = None
    elif mutation == "static_call_count":
        value["static_audit"]["call_site_count"] -= 1
    elif mutation == "precision":
        value["numerical_precision_boundary"][
            "cross_precision_bitwise_equality_claimed"
        ] = True
    elif mutation == "precision_bool_alias":
        value["numerical_precision_boundary"][
            "cross_precision_bitwise_equality_claimed"
        ] = 0
    elif mutation == "replay_panel":
        value["replay_panel"]["sha256"] = "0" * 64
    elif mutation == "replay_panel_float_alias":
        value["replay_panel"]["samples"] = 2.0
    elif mutation == "timing_missing":
        value["timings_seconds"].pop("compile")
    elif mutation == "timing_relation":
        value["timings_seconds"]["through_full_rank"] = 1.0
    elif mutation == "peak":
        value["peak_rss_mb"] = float("nan")
    elif mutation == "dooms_reference":
        value["dooms_reference_sha256"] = "0" * 64
    elif mutation == "canonical_shape":
        value["canonical_shape"]["reduced_q_binary_nodes"] -= 1
    elif mutation == "canonical_max_bond_low":
        value["canonical_shape"]["maximum_local_bond_dimension"] -= 1
    elif mutation == "canonical_max_bond_high":
        value["canonical_shape"]["maximum_local_bond_dimension"] += 1
    elif mutation == "canonical_max_bond_type":
        value["canonical_shape"]["maximum_local_bond_dimension"] = 385.0
    elif mutation == "canonical_max_cp_low":
        value["canonical_shape"]["maximum_cp_rank"] -= 1
    elif mutation == "canonical_max_cp_high":
        value["canonical_shape"]["maximum_cp_rank"] += 1
    elif mutation == "canonical_max_cp_type":
        value["canonical_shape"]["maximum_cp_rank"] = 1_153.0
    elif mutation == "initial_bool_alias":
        value["initial_shape"]["dense_clone_nodes"] = False
    elif mutation == "route":
        value["route_inventory"]["streamed_cp_candidate_count"] += 1
    elif mutation == "route_bool_alias":
        value["route_inventory"]["streamed_tall_unrepresentable_count"] = False
    elif mutation == "algorithm1_method":
        value["algorithm1"]["factorization_methods"] = ["unregistered_factorization"]
    elif mutation == "algorithm1_extra":
        value["algorithm1"]["unexpected"] = 1
    elif mutation == "algorithm1_push":
        value["algorithm1"]["push_ledger"][0] -= 1
    elif mutation == "algorithm1_reconstruction":
        value["algorithm1"][
            "maximum_local_scaled_reconstruction_relative_error"
        ] = lane.LOCAL_CERTIFICATE_LIMIT
    elif mutation == "algorithm1_provenance":
        value["algorithm1"]["direct_q_provenance"]["verified_steps"] -= 1
    elif mutation == "head_exponent_low":
        value["algorithm1"]["canonical_exponent_normal_form"][
            "head_binary_exponent"
        ] -= 1
    elif mutation == "head_exponent_high":
        value["algorithm1"]["canonical_exponent_normal_form"][
            "head_binary_exponent"
        ] += 1
    elif mutation == "cp_columns_low":
        value["algorithm1"]["canonical_exponent_normal_form"][
            "cp_direct_q_columns"
        ] -= 1
    elif mutation == "cp_columns_high":
        value["algorithm1"]["canonical_exponent_normal_form"][
            "cp_direct_q_columns"
        ] += 1
    elif mutation == "telemetry_missing":
        value["algorithm1"]["telemetry"].pop("maximum_temporary_tensor_order")
    elif mutation == "telemetry_extra":
        value["algorithm1"]["telemetry"]["unexpected"] = 0
    elif mutation == "unused_counter_nonzero":
        value["algorithm1"]["telemetry"][
            lane._ZERO_UNUSED_FACTORIZATION_COUNTER_KEYS[0]
        ] = 1
    elif mutation == "telemetry_bool_alias":
        value["algorithm1"]["telemetry"][
            "raw_order_three_core_materializations"
        ] = False
    elif mutation == "algorithm2_message_count":
        value["algorithms2_and3"]["algorithm2_child_messages"] -= 1
    elif mutation == "algorithm2_runtime_call":
        value["runtime_guard"]["allowed_calls"].pop()
    elif mutation == "algorithms23_extra":
        value["algorithms2_and3"]["unexpected"] = 0
    elif mutation == "runtime_patched_entrypoint":
        value["runtime_guard"]["patched_entrypoints"][0] = "guarded.entrypoint.drift"
    elif mutation == "runtime_count_low":
        value["runtime_guard"]["allowed_call_count"] -= 1
    elif mutation == "runtime_count_high":
        value["runtime_guard"]["allowed_call_count"] += 1
    elif mutation == "runtime_bool_alias":
        value["runtime_guard"]["prohibited_attempt_count"] = False
    elif mutation == "algorithm3_node_count":
        value["algorithms2_and3"]["algorithm3_diagonalized_nodes"] -= 1
    elif mutation == "algorithm3_occurrence":
        value["algorithms2_and3"]["algorithm3_occurrence_ledger"][0] -= 1
    elif mutation == "algorithm3_replay":
        value["algorithms2_and3"]["algorithm3_replay_relative_error"] = float("nan")
    elif mutation == "algorithm3_offdiagonal":
        value["algorithms2_and3"][
            "maximum_recontracted_offdiagonal_ratio"
        ] = lane.ENVIRONMENT_DIAGONALITY_LIMIT
    elif mutation == "scope":
        value["deployment_interface_scope"][
            "gripper_sign_decode_contracted"
        ] = True
    elif mutation == "scope_bool_alias":
        value["deployment_interface_scope"][
            "gripper_sign_decode_contracted"
        ] = 0
    elif mutation == "claim_boundary":
        value["claim_boundary"] = "overbroad policy claim"
    elif mutation == "bounded_oracle_hash":
        value["bounded_clone_and_regression_attestation"]["sha256"] = "0" * 64
    elif mutation == "bounded_oracle_inventory":
        value["bounded_clone_and_regression_attestation"][
            "required_oracles"
        ].pop()
    elif mutation == "bounded_oracle_extra":
        value["bounded_clone_and_regression_attestation"]["unexpected"] = None
    elif mutation == "bounded_shared_core":
        value["bounded_clone_and_regression_attestation"][
            "shared_core_sha256"
        ]["xvla/train/__init__.py"] = "0" * 64
    else:
        value["full_rank_gates"]["unexpected"] = True
    gates = lane._b1c_full_scientific_evidence_gates(
        value,
        bounded_unit_sha256=_TEST_BOUNDED_SHA256,
        odt_sources=_TEST_ODT_SOURCES,
        patched_entrypoints_sha256=_TEST_PATCHED_ENTRYPOINTS_SHA256,
    )
    assert gates[failed_gate] is False
    assert not all(gates.values())


def test_b1c_checkpoint_schema_matches_exact_producer_fields() -> None:
    checkpoint = _full_scientific_record()["checkpoint"]
    assert lane._b1c_checkpoint_identity_is_exact(checkpoint)
    checkpoint["parameters"] = checkpoint.pop("parameter_count")
    assert not lane._b1c_checkpoint_identity_is_exact(checkpoint)


@pytest.mark.parametrize(
    "mutation", ("top_extra", "gate_extra", "count", "count_type_alias")
)
def test_b1c_current_unit_schema_rejects_mutation(mutation: str) -> None:
    unit_sha = "1" * 64
    direct_sha = "2" * 64
    value = {
        "path": "athena/results/capable_linear_b1c0/unit_suite.json",
        "sha256": unit_sha,
        "direct_result_path": (
            "athena/results/capable_linear_b1c0_core_unit.json"
        ),
        "direct_result_sha256": direct_sha,
        "test_counts": {"direct": 52, "capability": 34, "integrity": 7},
        "input_sha256": {
            "current_unit": unit_sha,
            "current_direct": direct_sha,
        },
        "gates": {
            name: True for name in lane.EXPECTED_B1C_CURRENT_UNIT_GATE_KEYS
        },
    }
    assert lane._b1c_current_unit_record_is_exact(value, unit_sha, direct_sha)
    if mutation == "top_extra":
        value["unexpected"] = None
    elif mutation == "gate_extra":
        value["gates"]["unexpected"] = True
    elif mutation == "count_type_alias":
        value["test_counts"]["direct"] = 52.0
    else:
        value["test_counts"]["capability"] -= 1
    assert not lane._b1c_current_unit_record_is_exact(
        value, unit_sha, direct_sha
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "duplicate",
        "wrong_bool",
        "too_many_steps",
        "zero_steps",
        "early_nonterminal_stop",
        "elapsed_int_alias",
    ),
)
def test_episode_inventory_rejects_adversarial_mutations(mutation: str) -> None:
    records = _episode_records(6, 8)
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records[-1] = dict(records[-2])
    elif mutation == "wrong_bool":
        records[0]["success"] = 1
    elif mutation == "too_many_steps":
        records[0]["steps"] = lane.capability_lane.MAX_STEPS + 1
    elif mutation == "zero_steps":
        records[0]["steps"] = 0
    elif mutation == "elapsed_int_alias":
        records[0]["elapsed_s"] = 0
    else:
        records[1]["success"] = False
        records[1]["terminated_without_success"] = False
        records[1]["steps"] = lane.capability_lane.MAX_STEPS - 1
    exact, *_ = lane._episode_records_are_exact(
        records, 6, 8, max_steps=lane.capability_lane.MAX_STEPS
    )
    assert not exact


def _install_test_row_bundle_authority(
    records: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    for task in range(10):
        row_hashes = [
            record["initial_state_sha256"]
            for record in records
            if record["task_index"] == task
        ]
        if row_hashes:
            monkeypatch.setitem(
                lane.EXPECTED_CAPABILITY_INITIAL_STATE_ROW_BUNDLE_SHA256,
                task,
                lane._canonical_sha256(row_hashes),
            )


def test_cross_consumer_rejects_row_hash_tamper_even_with_recomputed_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _episode_records(0, 3)
    _install_test_row_bundle_authority(records, monkeypatch)
    value = {
        "episodes": records,
        "episode_identity_sha256": lane._canonical_sha256(
            lane._episode_identity_rows(records)
        ),
    }
    assert lane._row_identity_evidence_is_exact(
        value, 0, 3, max_steps=280, episodes_per_task=50
    )
    records[0]["initial_state_sha256"] = "f" * 64
    value["episode_identity_sha256"] = lane._canonical_sha256(
        lane._episode_identity_rows(records)
    )
    assert not lane._row_identity_evidence_is_exact(
        value, 0, 3, max_steps=280, episodes_per_task=50
    )


def test_cross_consumer_rechecks_aggregate_copy_and_row_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_episodes = _episode_records(0, 10)
    _install_test_row_bundle_authority(shard_episodes, monkeypatch)
    aggregate = {
        "episodes": copy.deepcopy(shard_episodes),
        "episode_identity_sha256": lane._canonical_sha256(
            lane._episode_identity_rows(shard_episodes)
        ),
    }
    assert lane._aggregate_episode_copy_is_exact(aggregate, shard_episodes)
    aggregate["episodes"][0]["initial_state_sha256"] = "f" * 64
    aggregate["episode_identity_sha256"] = lane._canonical_sha256(
        lane._episode_identity_rows(aggregate["episodes"])
    )
    assert not lane._aggregate_episode_copy_is_exact(aggregate, shard_episodes)


def test_cross_stage_join_requires_distinct_authorities() -> None:
    capability, full = _join_records()
    assert all(lane._cross_stage_gates(capability, full).values())
    full["source_manifest_sha256"] = capability["source_manifest_sha256"]
    assert lane._cross_stage_gates(capability, full)["separate_source_authorities"] is False


@pytest.mark.parametrize(
    "mutation",
    ("checkpoint", "config", "trials", "floor", "full_gate", "capability_gate"),
)
def test_cross_stage_join_fails_each_scientific_gate(mutation: str) -> None:
    capability, full = _join_records()
    if mutation == "checkpoint":
        capability["checkpoint_sha256"] = "0" * 64
    elif mutation == "config":
        capability["config_identity"] = {**lane.EXPECTED_CONFIG_IDENTITY, "dim": 1}
    elif mutation == "trials":
        capability["trials"] = 499
    elif mutation == "floor":
        capability["successes"] = 399
    elif mutation == "full_gate":
        full["gates"] = {"exact": False}
    else:
        capability["gates"] = {"capability": False}
    assert not all(lane._cross_stage_gates(capability, full).values())


def test_publication_is_exclusive_and_read_only(tmp_path: Path) -> None:
    output = tmp_path / "composite.json"
    digest = lane._publish(output, {"ok": True})
    assert lane._sha256(output) == digest
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        lane._publish(output, {"ok": False})


def test_publication_rejects_broken_link(tmp_path: Path) -> None:
    output = tmp_path / "composite.json"
    output.symlink_to(tmp_path / "missing.json")
    with pytest.raises(FileExistsError):
        lane._publish(output, {"ok": True})


def test_entrypoint_inventory_is_unique_and_complete() -> None:
    assert len(lane._ENTRYPOINTS) == len({path.resolve() for path in lane._ENTRYPOINTS})
    assert {path.name for path in lane._ENTRYPOINTS} == {
        "attest_capable_linear_rtx6000_cross_stage.py",
        "run_capable_linear_rtx6000_capability.py",
        "run_capable_linear_rtx6000_tests.py",
        "test_capable_linear_rtx6000_cross_stage.py",
        "test_capable_linear_rtx6000_capability.py",
    }


def test_scientific_schemas_match_the_frozen_b1c_producer_ast() -> None:
    source_map = lane._read_line_manifest(
        lane.B1C_SOURCE_MANIFEST,
        lane.B1C_STAGE_ROOT,
        "frozen b1c producer manifest",
    )
    runner_relative = "scripts/run_capable_linear_direct_odt_spectrum.py"
    core_relative = "xvla/train/implicit_sparse_projective_odt.py"
    integrity_relative = "scripts/capable_linear_artifact_integrity.py"
    assert source_map[runner_relative] == lane._sha256(
        lane.B1C_STAGE_ROOT / runner_relative
    )
    assert source_map[core_relative] == lane._sha256(
        lane.B1C_STAGE_ROOT / core_relative
    )
    assert source_map[integrity_relative] == lane._sha256(
        lane.B1C_STAGE_ROOT / integrity_relative
    )

    runner_tree = ast.parse((lane.B1C_STAGE_ROOT / runner_relative).read_text())
    summary_function = next(
        node
        for node in runner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_algorithm1_summary"
    )
    summary_value = next(
        node.value
        for node in ast.walk(summary_function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "summary"
            for target in node.targets
        )
    )
    assert isinstance(summary_value, ast.Dict)
    assert {ast.literal_eval(key) for key in summary_value.keys} == (
        lane.EXPECTED_ALGORITHM1_KEYS
    )

    full_function = next(
        node
        for node in runner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_full_mode"
    )
    certificate_value = next(
        node.value
        for node in ast.walk(full_function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "full_rank_certificate"
            for target in node.targets
        )
    )
    assert isinstance(certificate_value, ast.Dict)
    assert {ast.literal_eval(key) for key in certificate_value.keys} == (
        lane.EXPECTED_B1C_FULL_TOP_LEVEL_KEYS
    )
    certificate_fields = {
        ast.literal_eval(key): value
        for key, value in zip(certificate_value.keys, certificate_value.values)
    }
    algorithms_value = certificate_fields["algorithms2_and3"]
    assert isinstance(algorithms_value, ast.Dict)
    assert {ast.literal_eval(key) for key in algorithms_value.keys} == (
        lane.EXPECTED_ALGORITHMS2_AND3_KEYS
    )
    for field_name, expected_keys in (
        ("inactive_vision_norm_sentinel", lane.EXPECTED_B1C_SENTINEL_KEYS),
        ("timings_seconds", lane.EXPECTED_B1C_TIMING_KEYS),
    ):
        nested = certificate_fields[field_name]
        assert isinstance(nested, ast.Dict)
        assert {ast.literal_eval(key) for key in nested.keys} == expected_keys
    assert ast.literal_eval(certificate_fields["numerical_precision_boundary"]) == (
        lane.EXPECTED_B1C_PRECISION_BOUNDARY
    )

    load_function = next(
        node
        for node in runner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_model"
    )
    load_return = next(
        node.value
        for node in ast.walk(load_function)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Tuple)
        and len(node.value.elts) == 2
        and isinstance(node.value.elts[1], ast.Dict)
    )
    checkpoint_value = load_return.elts[1]
    assert isinstance(checkpoint_value, ast.Dict)
    assert {ast.literal_eval(key) for key in checkpoint_value.keys} == (
        lane.EXPECTED_B1C_CHECKPOINT_KEYS
    )

    replay_function = next(
        node
        for node in runner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_replay_panel"
    )
    replay_return = next(
        node.value
        for node in ast.walk(replay_function)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Tuple)
        and len(node.value.elts) == 2
        and isinstance(node.value.elts[1], ast.Dict)
    )
    replay_value = replay_return.elts[1]
    assert isinstance(replay_value, ast.Dict)
    assert {ast.literal_eval(key) for key in replay_value.keys} == set(
        lane.EXPECTED_B1C_REPLAY_PANEL
    )

    core_tree = ast.parse((lane.B1C_STAGE_ROOT / core_relative).read_text())
    telemetry_function = next(
        node
        for node in core_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "telemetry_dict"
    )
    telemetry_value = next(
        node.value
        for node in ast.walk(telemetry_function)
        if isinstance(node, ast.Return)
    )
    assert isinstance(telemetry_value, ast.Dict)
    assert {ast.literal_eval(key) for key in telemetry_value.keys} == (
        lane.EXPECTED_ALGORITHM1_TELEMETRY_KEYS
    )

    integrity_tree = ast.parse(
        (lane.B1C_STAGE_ROOT / integrity_relative).read_text()
    )
    inventory_function = next(
        node
        for node in integrity_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "require_rational_norm_buffer_inventory"
    )
    inventory_return = next(
        node.value
        for node in ast.walk(inventory_function)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    )
    assert {ast.literal_eval(key) for key in inventory_return.keys} == set(
        lane.EXPECTED_B1C_NORMALIZATION_BUFFER_INVENTORY
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(lane.B1C_STAGE_ROOT / runner_relative),
            "--audit-only",
        ],
        cwd=lane.B1C_STAGE_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    static_audit = json.loads(completed.stdout)
    assert set(static_audit) == lane.EXPECTED_B1C_STATIC_AUDIT_KEYS
    assert static_audit["scope"] == "transitive_local_import_closure"
    assert static_audit["entrypoints"] == lane.EXPECTED_B1C_STATIC_ENTRYPOINTS
    assert static_audit["source_count"] == len(source_map) == 38
    assert static_audit["source_sha256"] == source_map
    assert static_audit["call_site_count"] == 8_310
    assert static_audit["direct_qr_required"] is True
    assert static_audit["direct_qr_call_sites"] == 4
    for field in (
        "prohibited_calls_found",
        "prohibited_self_overlap_sites",
        "guarded_dormant_spectral_norm_sites",
        "duplicate_top_level_definition_sites",
    ):
        assert static_audit[field] == []

    unit_tree = ast.parse(
        (lane.PROJECT_ROOT / "scripts/run_capable_linear_rtx6000_tests.py").read_text()
    )
    unit_assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in unit_tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id
        in {
            "EXPECTED_CAPABILITY_TEST_COUNT",
            "EXPECTED_CROSS_STAGE_TEST_COUNT",
            "EXPECTED_UNIT_KEYS",
            "EXPECTED_JUNIT_KEYS",
            "EXPECTED_GATE_KEYS",
            "EXPECTED_TEST_NAME_SHA256",
        }
    }
    assert unit_assignments == {
        "EXPECTED_CAPABILITY_TEST_COUNT": (
            lane.EXPECTED_RTX_CAPABILITY_TEST_COUNT
        ),
        "EXPECTED_CROSS_STAGE_TEST_COUNT": (
            lane.EXPECTED_RTX_CROSS_STAGE_TEST_COUNT
        ),
        "EXPECTED_UNIT_KEYS": lane.EXPECTED_RTX_UNIT_KEYS,
        "EXPECTED_JUNIT_KEYS": lane.EXPECTED_RTX_UNIT_JUNIT_KEYS,
        "EXPECTED_GATE_KEYS": lane.EXPECTED_RTX_UNIT_GATE_KEYS,
        "EXPECTED_TEST_NAME_SHA256": lane.EXPECTED_RTX_UNIT_TEST_NAME_SHA256,
    }


def test_cross_stage_source_orders_preimport_before_project_imports() -> None:
    path = Path(lane.__file__)
    tree = ast.parse(path.read_text())
    assignments = {
        node.targets[0].id: node.lineno
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    project_import_lines = [
        node.lineno
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("scripts")
    ]
    assert project_import_lines
    assert assignments["_PREIMPORT"] < min(project_import_lines)


def test_cross_stage_claims_remain_separate() -> None:
    source = Path(lane.__file__).read_text()
    assert '"simulator_inside_weight_only_odt_closure": False' in source
    assert '"compressed_policy_capability_measured": False' in source
    assert '"a30_hardware_reproduction_claimed": False' in source
    assert '"compressed_policy_capability_attested": False' in source
    assert "One fresh evaluation under the pinned Quadro RTX 6000" in (
        lane.capability_lane.DETERMINISM_CLAIM_SCOPE
    )
    assert "Repeated-run, external-simulator" in (
        lane.capability_lane.DETERMINISM_CLAIM_SCOPE
    )
    assert "were not tested and are not claimed" in (
        lane.capability_lane.DETERMINISM_CLAIM_SCOPE
    )
    assert "Repeated-run simulator determinism is not claimed" in (
        lane.capability_lane.CAPABILITY_CLAIM
    )


def test_fixed_b1c_authority_is_v2_stage() -> None:
    assert lane.B1C_STAGE_ROOT == Path("/work/joy/x-vla-capable-linear-b1c0-odt-v2")
    assert lane.EXPECTED_B1C_STAGE_LEDGER_SHA256 == (
        "e4716a413bc742943b623ddebd39783bb54c40558b59446c99998aa07892b8c3"
    )
    assert lane.EXPECTED_B1C_SOURCE_MANIFEST_SHA256 == (
        "ebd9dd4162734e15bfd93c3f8d0aded01fe04001ba0b102dce405879da620e6f"
    )


def test_no_output_resume_or_overwrite_branch() -> None:
    source = Path(lane.__file__).read_text()
    assert "os.path.lexists(CROSS_STAGE_OUTPUT)" in source
    assert "FileExistsError" in source
    assert "resume" not in source.lower()


def test_submit_dag_requires_green_unit_before_smoke_and_shards() -> None:
    source = (
        lane.PROJECT_ROOT / "athena/submit_capable_linear_rtx6000.sh"
    ).read_text()
    assert 'smoke_job=$(submit_job --dependency="afterok:${unit_job}"' in source
    assert source.count('--dependency="afterok:${unit_job}:${smoke_job}"') == 4
    assert '--dependency="afterok:${aggregate_job}:${b1c_full_job}"' in source
    wrappers = sorted(
        lane.PROJECT_ROOT.glob("athena/slurm_capable_linear_rtx6000_*.sbatch")
    )
    assert len(wrappers) == 5
    for wrapper in wrappers:
        assert "#SBATCH --no-requeue" in wrapper.read_text()


@pytest.mark.parametrize(
    "mutation",
    (
        "training_record",
        "seed",
        "seed_bool_alias",
        "checkpoint_twice",
        "ledger_twice",
    ),
)
def test_raw_capability_identity_rejects_each_mutation(mutation: str) -> None:
    value = {
        "training_record_sha256": lane.EXPECTED_TRAINING_RECORD_SHA256,
        "seed": 0,
        "exact_checkpoint_digest_verified_before_and_after": True,
        "stage_ledger_verified_before_and_after": True,
    }
    if mutation == "training_record":
        value["training_record_sha256"] = "0" * 64
    elif mutation == "seed":
        value["seed"] = 1
    elif mutation == "seed_bool_alias":
        value["seed"] = False
    elif mutation == "checkpoint_twice":
        value["exact_checkpoint_digest_verified_before_and_after"] = False
    else:
        value["stage_ledger_verified_before_and_after"] = False
    gates = lane._producer_identity_gates(value)
    assert gates["seed" if mutation == "seed_bool_alias" else mutation] is False
    assert sum(not passed for passed in gates.values()) == 1


@pytest.mark.parametrize("artifact_kind", ("smoke", "shard", "aggregate"))
def test_each_capability_artifact_kind_rejects_boolean_seed_alias(
    artifact_kind: str,
) -> None:
    value = {
        "training_record_sha256": lane.EXPECTED_TRAINING_RECORD_SHA256,
        "seed": False,
        "exact_checkpoint_digest_verified_before_and_after": True,
        "stage_ledger_verified_before_and_after": True,
        "artifact_kind": artifact_kind,
    }
    gates = lane._producer_identity_gates(value)
    assert gates["seed"] is False
    assert all(
        passed for name, passed in gates.items() if name != "seed"
    )


@pytest.mark.parametrize(
    ("record", "field", "alias"),
    (
        ("checkpoint", "state_count", 540.0),
        ("checkpoint", "active_normalization_site_count", 73.0),
        ("sentinel", "call_count", False),
        ("sentinel", "every_active_site_call_count", True),
        ("protocol", "settle_success_or_done_count", False),
        ("protocol", "episodes_per_task", True),
        ("protocol", "canonical_init_states", 1),
    ),
)
def test_cross_capability_authorities_reject_type_aliases(
    record: str, field: str, alias: object
) -> None:
    checkpoint = {
        "checkpoint_sha256": lane.EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_bytes": lane.EXPECTED_B1C_CHECKPOINT_BYTES,
        "state_count": 540,
        "state_key_sha256": (
            lane.capability_lane.EXPECTED_CHECKPOINT_STATE_KEY_SHA256
        ),
        "normalization_site_count": 74,
        "active_normalization_site_count": 73,
        "normalization_buffer_inventory": copy.deepcopy(
            lane.capability_lane.EXPECTED_NORMALIZATION_BUFFER_INVENTORY
        ),
        "parameters": 20_137_352,
        "config_identity": copy.deepcopy(lane.EXPECTED_CONFIG_IDENTITY),
    }
    sentinel = {
        "site": "vision.norm_out",
        "armed_during_sentinel_replay_and_all_rollout_policy_forwards": True,
        "call_count": 0,
        "total_site_count": 74,
        "inactive_sites": ["vision.norm_out"],
        "active_site_count": 73,
        "active_call_count": 73,
        "every_active_site_call_count": 1,
        "source_output_bitwise_equal_without_sentinels": True,
    }
    protocol = lane._expected_capability_rollout_protocol(1, 1)
    assert lane._capability_checkpoint_record_is_exact(checkpoint)
    assert lane._capability_sentinel_is_exact(sentinel)
    assert lane._rollout_protocol_is_exact(protocol, 1, 1)
    selected = {"checkpoint": checkpoint, "sentinel": sentinel, "protocol": protocol}[
        record
    ]
    selected[field] = alias
    validator = {
        "checkpoint": lane._capability_checkpoint_record_is_exact,
        "sentinel": lane._capability_sentinel_is_exact,
        "protocol": lambda value: lane._rollout_protocol_is_exact(value, 1, 1),
    }[record]
    assert not validator(selected)


@pytest.mark.parametrize(
    "mutation",
    (
        "top_extra",
        "junit_extra",
        "gate_extra",
        "name_inventory",
        "elapsed_nonfinite",
        "elapsed_type_alias",
        "count",
        "count_type_alias",
        "static_audit",
    ),
)
def test_rtx_current_unit_consumer_rejects_each_schema_mutation(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    capability_count = lane.EXPECTED_RTX_CAPABILITY_TEST_COUNT
    cross_count = lane.EXPECTED_RTX_CROSS_STAGE_TEST_COUNT
    total = capability_count + cross_count
    names = [f"test_case_{index:03d}" for index in range(total)]
    monkeypatch.setattr(
        lane, "EXPECTED_RTX_UNIT_TEST_NAME_SHA256", lane._canonical_sha256(names)
    )
    value = {
        "schema": "xvla_capable_linear_rtx6000_unit_suite_v1",
        "test_counts": {
            "capability": capability_count,
            "cross_stage": cross_count,
            "total": total,
        },
        "junit": {
            "tests": total,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "test_names": names,
        },
        "elapsed_seconds": 1.0,
        "capability_source_manifest_sha256": "1" * 64,
        "cross_stage_source_manifest_sha256": "2" * 64,
        "stage_ledger_sha256": "3" * 64,
        "static_audit": {"closed": True},
        "gates": {name: True for name in lane.EXPECTED_RTX_UNIT_GATE_KEYS},
        "all_gates_pass": True,
    }
    arguments = {
        "capability_manifest_sha256": "1" * 64,
        "cross_manifest_sha256": "2" * 64,
        "stage_ledger_sha256": "3" * 64,
        "static_audit": {"closed": True},
    }
    assert lane._rtx_current_unit_record_is_exact(value, **arguments)
    if mutation == "top_extra":
        value["unexpected"] = None
    elif mutation == "junit_extra":
        value["junit"]["unexpected"] = None
    elif mutation == "gate_extra":
        value["gates"]["unexpected"] = True
    elif mutation == "name_inventory":
        value["junit"]["test_names"][-1] = "wrong_test_name"
    elif mutation == "elapsed_nonfinite":
        value["elapsed_seconds"] = float("nan")
    elif mutation == "elapsed_type_alias":
        value["elapsed_seconds"] = 1
    elif mutation == "count":
        value["junit"]["tests"] -= 1
    elif mutation == "count_type_alias":
        value["junit"]["failures"] = False
    else:
        value["static_audit"] = {"closed": False}
    assert not lane._rtx_current_unit_record_is_exact(value, **arguments)


@pytest.mark.parametrize("field", ("settle", "policy"))
def test_cross_consumer_rejects_transition_count_mutation(field: str) -> None:
    records = _episode_records(0, 1)
    value = {
        "validated_transition_counts": {
            "settle": lane.capability_lane.SETTLE_STEPS * len(records),
            "policy": sum(record["steps"] for record in records),
        }
    }
    assert lane._transition_counts_are_exact(
        value,
        records,
        expected_settle=lane.capability_lane.SETTLE_STEPS * len(records),
    )
    value["validated_transition_counts"][field] += 1
    assert not lane._transition_counts_are_exact(
        value,
        records,
        expected_settle=lane.capability_lane.SETTLE_STEPS * len(records),
    )
    expected = (
        lane.capability_lane.SETTLE_STEPS * len(records)
        if field == "settle"
        else sum(record["steps"] for record in records)
    )
    value["validated_transition_counts"][field] = (
        False if expected == 0 else True
    )
    assert not lane._transition_counts_are_exact(
        value,
        records,
        expected_settle=lane.capability_lane.SETTLE_STEPS * len(records),
    )


@pytest.mark.parametrize(
    "mutation",
    ("top", "episode", "protocol", "checkpoint", "sentinel", "environment"),
)
def test_cross_capability_schema_rejects_unknown_nested_fields(
    mutation: str,
) -> None:
    value = {key: None for key in lane.EXPECTED_CAPABILITY_ROLLOUT_KEYS}
    value["episodes"] = [
        {key: None for key in lane.EXPECTED_CAPABILITY_EPISODE_KEYS}
    ]
    value["protocol"] = {
        key: None for key in lane.EXPECTED_CAPABILITY_ROLLOUT_PROTOCOL_KEYS
    }
    value["checkpoint"] = {
        key: None for key in lane.EXPECTED_CAPABILITY_CHECKPOINT_KEYS
    }
    value["inactive_vision_norm_sentinel"] = {
        key: None for key in lane.EXPECTED_CAPABILITY_SENTINEL_KEYS
    }
    value["environment"] = {
        **lane.capability_lane.EXPECTED_VERSIONS,
        "cuda": lane.capability_lane.EXPECTED_CUDA_VERSION,
        "gpu": lane.capability_lane.EXPECTED_GPU_NAME,
        "slurm_node": lane.capability_lane.EXPECTED_SLURM_NODE,
        "mujoco_gl": "egl",
        "matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "cublas_workspace_config": (
            lane.capability_lane.EXPECTED_CUBLAS_WORKSPACE_CONFIG
        ),
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }
    value["validated_transition_counts"] = {"settle": 10, "policy": 1}
    assert lane._capability_artifact_schema_is_exact(value, aggregate=False)
    target = value if mutation == "top" else value[
        {
            "episode": "episodes",
            "protocol": "protocol",
            "checkpoint": "checkpoint",
            "sentinel": "inactive_vision_norm_sentinel",
            "environment": "environment",
        }[mutation]
    ]
    if mutation == "episode":
        target = target[0]
    target["unexpected"] = None
    assert not lane._capability_artifact_schema_is_exact(value, aggregate=False)


@pytest.mark.parametrize(
    "mutation", ("top", "episode", "protocol", "shard", "sentinel", "environment")
)
def test_cross_aggregate_schema_rejects_unknown_nested_fields(
    mutation: str,
) -> None:
    value = {key: None for key in lane.EXPECTED_CAPABILITY_AGGREGATE_KEYS}
    value["episodes"] = [
        {key: None for key in lane.EXPECTED_CAPABILITY_EPISODE_KEYS}
    ]
    value["protocol"] = {
        key: None for key in lane.EXPECTED_CAPABILITY_AGGREGATE_PROTOCOL_KEYS
    }
    value["shards"] = [
        {key: None for key in lane.EXPECTED_CAPABILITY_SHARD_RECORD_KEYS}
        for _ in range(4)
    ]
    value["inactive_vision_norm_sentinel"] = {
        key: None for key in lane.EXPECTED_CAPABILITY_AGGREGATE_SENTINEL_KEYS
    }
    value["environment"] = {
        **lane.capability_lane.EXPECTED_VERSIONS,
        "cuda": lane.capability_lane.EXPECTED_CUDA_VERSION,
        "gpu": lane.capability_lane.EXPECTED_GPU_NAME,
        "slurm_node": lane.capability_lane.EXPECTED_SLURM_NODE,
        "mujoco_gl": "egl",
        "matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "cublas_workspace_config": (
            lane.capability_lane.EXPECTED_CUBLAS_WORKSPACE_CONFIG
        ),
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }
    value["validated_transition_counts"] = {"settle": 5_000, "policy": 500}
    assert lane._capability_artifact_schema_is_exact(value, aggregate=True)
    if mutation == "top":
        target = value
    elif mutation == "episode":
        target = value["episodes"][0]
    elif mutation == "shard":
        target = value["shards"][0]
    else:
        target = value[
            {
                "protocol": "protocol",
                "sentinel": "inactive_vision_norm_sentinel",
                "environment": "environment",
            }[mutation]
        ]
    target["unexpected"] = None
    assert not lane._capability_artifact_schema_is_exact(value, aggregate=True)
