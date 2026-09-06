#!/usr/bin/env python3
"""Join a fresh RTX 6000 capability replay to the frozen b1c0 ODT result."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STAGE_ROOT = Path("/work/joy/x-vla-capable-linear-rtx6000-v3")
SOURCE_MANIFEST = (
    PROJECT_ROOT / "athena/capable_linear_rtx6000_cross_stage_sources.sha256"
)
CAPABILITY_SOURCE_MANIFEST = (
    PROJECT_ROOT / "athena/capable_linear_rtx6000_capability_sources.sha256"
)
STAGE_LEDGER = PROJECT_ROOT / "athena/capable_linear_rtx6000_stage.sha256"
CHECKPOINT = PROJECT_ROOT / "inputs/capable_linear_b1c0_checkpoint.pt"
TRAINING_RECORD = PROJECT_ROOT / "inputs/capable_linear_training.json"
PINNED_B1C_STAGE_LEDGER = PROJECT_ROOT / "inputs/b1c_stage_ledger.sha256"
PINNED_B1C_SOURCE_MANIFEST = PROJECT_ROOT / "inputs/b1c_odt_sources.sha256"
RESULT_DIRECTORY = PROJECT_ROOT / "athena/results/capable_linear_rtx6000"
CAPABILITY_AGGREGATE = RESULT_DIRECTORY / "rtx6000_capability_aggregate.json"
CAPABILITY_SMOKE = RESULT_DIRECTORY / "rtx6000_capability_smoke.json"
CROSS_STAGE_OUTPUT = RESULT_DIRECTORY / "rtx6000_b1c0_cross_stage_composite.json"
UNIT_RESULT = RESULT_DIRECTORY / "rtx6000_unit_suite.json"

B1C_STAGE_ROOT = Path("/work/joy/x-vla-capable-linear-b1c0-odt-v2")
B1C_STAGE_LEDGER = B1C_STAGE_ROOT / "athena/capable_linear_b1c0_stage.sha256"
B1C_SOURCE_MANIFEST = (
    B1C_STAGE_ROOT / "athena/capable_linear_direct_odt_sources.sha256"
)
B1C_FULL_RANK = (
    B1C_STAGE_ROOT
    / "athena/results/capable_linear_b1c0/full_rank_certificate.json"
)
B1C_UNIT = B1C_STAGE_ROOT / "athena/results/capable_linear_b1c0/unit_suite.json"
B1C_DIRECT_UNIT = (
    B1C_STAGE_ROOT / "athena/results/capable_linear_b1c0_core_unit.json"
)
B1C_BOUNDED_UNIT = B1C_STAGE_ROOT / "inputs/direct_odt_truncation_unit_v2.json"
B1C_CHECKPOINT = B1C_STAGE_ROOT / "inputs/capable_linear_b1c0_checkpoint.pt"

EXPECTED_CHECKPOINT_SHA256 = (
    "b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee"
)
EXPECTED_TRAINING_RECORD_SHA256 = (
    "e6c07beb2efb7e5ebe55c92fabf694d97b50b9e399f50868eeef4af455e30fa7"
)
EXPECTED_B1C_STAGE_LEDGER_SHA256 = (
    "e4716a413bc742943b623ddebd39783bb54c40558b59446c99998aa07892b8c3"
)
EXPECTED_B1C_SOURCE_MANIFEST_SHA256 = (
    "ebd9dd4162734e15bfd93c3f8d0aded01fe04001ba0b102dce405879da620e6f"
)
EXPECTED_DOOMS_REFERENCE_SHA256 = (
    "a97f59a91d90ffde56bc22bf0b3c79e0891ff88197ca9d7670331e4208277559"
)
EXPECTED_CONFIG_IDENTITY = {
    "image_size": 64,
    "patch_size": 8,
    "vit_dim": 192,
    "vit_layers": 4,
    "vit_heads": 8,
    "vit_ffn_rank": 576,
    "vocab_size": 26,
    "max_instr_len": 32,
    "state_dim": 8,
    "n_embodiments": 1,
    "dim": 384,
    "n_layers": 8,
    "n_heads": 12,
    "ffn_rank": 1152,
    "attn": "bilinear",
    "vit_attn": "bilinear",
    "ffn": "bilinear",
    "norm": "rational",
    "qk_norm": "rational",
    "residual": True,
    "vit_residual": True,
    "action_horizon": 8,
    "action_dim": 7,
    "action_head": "linear",
}
SHARDS = ((0, 3), (3, 6), (6, 8), (8, 10))
CAPABILITY_FLOOR_SUCCESSES = 400
REPLAY_LIMIT = 3e-8
SOURCE_ACTION_LIMIT = 2e-10
LOCAL_CERTIFICATE_LIMIT = 3e-10
ENVIRONMENT_DIAGONALITY_LIMIT = 3e-10
EXPECTED_B1C_CLAIM_BOUNDARY = (
    "Exact shared syntactic projective DAG from fixed non-overlapping patch "
    "coordinates, categorical instruction coordinates, robot state, and the "
    "checkpoint's fixed singleton embodiment through the unchanged single-branch "
    "ChiViT, visual projection, causal joint backbone, final Pade RationalNorm, "
    "and deterministic linear action head. The sole observable is the ordered "
    "action_horizon by action_dimension quotient. This is a clone-unfolded "
    "coefficient object, not compression and not an identified-variable or "
    "categorical-simplex intrinsic metric."
)
EXPECTED_B1C_DEPLOYMENT_SCOPE = {
    "learned_neural_dag_inputs": (
        "unit-scaled image patches, categorical instruction coordinates, and "
        "normalized continuous robot state"
    ),
    "learned_neural_dag_outputs": "normalized continuous action coordinates",
    "fixed_image_rotation_and_scaling_contracted": False,
    "fixed_state_normalization_contracted": False,
    "fixed_action_denormalization_contracted": False,
    "instruction_text_tokenization_contracted": False,
    "gripper_sign_decode_contracted": False,
    "checkpoint_vision_norm_out_active_in_chivla_forward": False,
    "checkpoint_vision_classifier_head_active_in_chivla_forward": False,
    "checkpoint_inactive_modules_in_compiled_dag": False,
    "fixed_permutation_and_affine_maps_are_tensor_compatible": True,
    "gripper_sign_decode_is_external": True,
    "persisted_deployable_tensor_network_artifact": False,
}
EXPECTED_B1C_STRUCTURE = {
    "unique_nodes": 3_964_463,
    "edge_occurrences": 7_216_854,
    "physical_source_count": 97,
    "physical_leaf_count": 97,
    "physical_source_widths": {
        **{f"image.patch{index}": 192 for index in range(64)},
        **{f"instruction.token{index}": 26 for index in range(32)},
        "state": 8,
    },
    "maximum_physical_source_width": 192,
    "root_projective_width": 57,
    "observable_kind": "linear_action",
    "observable_euclidean_width": 56,
    "product_signs": None,
    "active_pade_sites": 73,
}
EXPECTED_B1C_INITIAL_SHAPE = {
    "unique_nodes": 3_964_463,
    "edge_occurrences": 7_216_854,
    "cp_binary_nodes": 3_252_488,
    "reduced_q_binary_nodes": 0,
    "dense_clone_nodes": 0,
    "unary_nodes": 711_975,
    "heterogeneous_physical_sources": 97,
    "maximum_local_bond_dimension": 385,
    "maximum_cp_rank": 1_153,
    "maximum_reduced_q_elements": 0,
    "fused_token_feature_dimension": 13_129,
}
EXPECTED_B1C_ROUTE_INVENTORY = {
    "schema": "exact_postorder_structural_direct_rq_routes_v1",
    "bounded_explicit_unfolding_element_limit": 4_000_000,
    "bounded_explicit_q_element_limit": 4_000_000,
    "bounded_retained_q_candidate_count": 3_248_703,
    "bounded_retained_q_total_elements": 36_286_059_776,
    "bounded_retained_q_replaced_cp_total_elements": 9_445_994_962,
    "bounded_retained_q_signed_storage_delta_elements": 26_840_064_814,
    "bounded_retained_q_positive_storage_delta_elements": 26_984_351_168,
    "bounded_retained_q_maximum_unfolding_elements": 3_739_329,
    "bounded_retained_q_maximum_elements": 3_739_329,
    "streamed_cp_candidate_count": 3_785,
    "streamed_tall_unrepresentable_count": 0,
    "streamed_tall_unrepresentable_shape_counts": {},
    "bounded_retained_q_shape_counts": {
        "121x2425": 256,
        "129x3201": 757,
        "145x3025": 256,
        "15x64": 4,
        "161x4257": 757,
        "169x3625": 256,
        "193x386": 512,
        "193x4225": 256,
        "193x5313": 757,
        "225x6369": 757,
        "257x7425": 757,
        "25x50": 139_264,
        "25x625": 129_024,
        "289x8481": 757,
        "29x225": 2,
        "2x1089": 1_004_160,
        "2x148225": 1_514,
        "2x324": 48,
        "2x37249": 576,
        "2x4": 674_247,
        "2x625": 270_336,
        "2x729": 32,
        "2x81": 1,
        "321x9537": 757,
        "33x1089": 485_148,
        "33x36": 168,
        "33x66": 532_692,
        "353x10593": 757,
        "385x18": 1,
        "385x3465": 1,
        "385x386": 64,
        "385x54": 32,
        "385x770": 1_514,
        "49x625": 256,
        "57x841": 1,
        "65x1089": 757,
        "73x1225": 256,
        "97x1825": 256,
        "97x2145": 757,
    },
    "nodes_with_propagated_input_shape_change": 3_279,
    "predicted_root_output_dimension": 57,
    "simulated_unique_node_count": 3_964_463,
}
EXPECTED_B1C_FACTORIZATION_METHODS = [
    "bounded_explicit_householder_direct_reduced_rq",
    "streamed_householder_direct_reduced_rq",
    "torch_qr_transpose_direct_reduced_rq",
]
EXPECTED_B1C_FULL_GATE_NAMES = {
    "linear_action_root",
    "source_observable_replay",
    "source_public_action_replay",
    "source_observable_matches_public_action",
    "exact_topology",
    "heterogeneous_ingress_unpadded",
    "no_dense_clone_core",
    "route_inventory_exact",
    "inactive_vision_norm_never_executed",
    "all_73_active_norms_observed_twice",
    "direct_methods_only",
    "final_replay",
    "production_replay_schedule",
    "local_scale_certificates",
    "canonical_exponent_normal_form",
    "direct_q_provenance_complete",
    "streamed_direct_q_ledger",
    "direct_q_telemetry_complete",
    "every_factor_occurrence_pushed",
    "rank_deficiency_route_is_shape_only",
    "node_inventory_preserved",
    "child_before_parent_postorder",
    "algorithm2_explicit_downstream_contraction_complete",
    "algorithm3_every_node",
    "algorithm3_every_occurrence",
    "algorithm3_replay",
    "algorithm3_recontracted_environments_diagonal",
    "zero_full_spectrum_retention",
    "compact_rank_bank_complete",
    "model_state_unchanged",
    "inactive_vision_norm_still_never_executed",
    "runtime_guard_exact",
}
EXPECTED_B1C_ALGORITHM_ORDER = {
    "algorithm1": "child-before-parent direct local RQ, with R pushed to every occurrence",
    "algorithm2": "root-first explicit downstream environment contraction and eigendecomposition",
    "algorithm3": "resulting full-rank gauge applied to every occurrence",
}
EXPECTED_B1C_RUNTIME_CALLS = {
    "xvla.train.implicit_sparse_projective_odt._positive_diagonal_direct_rq_rows:torch.linalg.qr",
    "xvla.train.implicit_sparse_projective_odt._direct_tsqr_panel:torch.linalg.qr",
    "xvla.train.implicit_sparse_projective_odt._solve_compact_q:torch.linalg.solve_triangular",
    "xvla.train.implicit_sparse_projective_odt.diagonalize_implicit_dag_full_rank:torch.linalg.eigh",
}
EXPECTED_B1C_PATCHED_ENTRYPOINTS_SHA256 = (
    "bb21d02650ee0f88d82bc5ea241f2224f84c4a880c616387c0717bdbbb4e4725"
)
EXPECTED_B1C_RUNTIME_CALL_COUNT = 8_111_588
EXPECTED_B1C_HEAD_BINARY_EXPONENT = (
    -292039185385285067477570283661535988302718692318198881774964170976485710
)
EXPECTED_B1C_CP_DIRECT_Q_COLUMNS = 363_272_585
EXPECTED_B1C_CHECKPOINT_BYTES = 80_752_109
EXPECTED_B1C_REPLAY_PANEL = {
    "seed": 20260905,
    "samples": 2,
    "sha256": "1927636743407536df467e0294951ba0213ea44618df7a8c07e5fc151c9bf9c8",
}
EXPECTED_B1C_PRECISION_BOUNDARY = {
    "checkpoint_storage_dtype": "float32",
    "exact_odt_verification_dtype": "float64",
    "exact_odt_verification_device": "cpu",
    "fresh_capability_execution_dtype": "float32",
    "fresh_capability_execution_device": "cuda",
    "float32_checkpoint_coefficients_exactly_representable_in_float64": True,
    "cross_precision_bitwise_equality_claimed": False,
}
EXPECTED_B1C_NORMALIZATION_BUFFER_INVENTORY = {
    "site_count": 74,
    "initialized_true_count": 73,
    "initialized_false_sites": ["vision.norm_out"],
    "all_running_ms_scalar_and_finite": True,
    "all_active_running_ms_positive": True,
    "all_pade_coefficients_match_frozen_defaults": True,
    "inactive_running_ms": {"vision.norm_out": 1.0},
}
EXPECTED_B1C_CHECKPOINT_KEYS = {
    "checkpoint_sha256",
    "checkpoint_bytes",
    "state_key_count",
    "parameter_count",
    "normalization_site_count",
    "active_normalization_site_count",
    "compiler_active_normalization_site_count",
    "compiler_active_snapshot_site_count",
    "compiler_active_set_equals_loaded_active_set",
    "normalization_buffer_inventory",
    "config_identity",
    "vocab_size",
}
EXPECTED_B1C_SENTINEL_KEYS = {
    "site",
    "armed_through_full_rank",
    "call_count",
    "total_site_count",
    "inactive_sites",
    "active_site_count",
    "active_call_count",
    "every_active_site_call_count",
}
EXPECTED_B1C_SOURCE_MANIFEST_KEYS = {
    "path",
    "sha256",
    "source_count",
    "source_sha256",
}
EXPECTED_B1C_STATIC_AUDIT_KEYS = {
    "scope",
    "entrypoints",
    "source_count",
    "source_sha256",
    "call_site_count",
    "direct_qr_required",
    "direct_qr_call_sites",
    "prohibited_calls_found",
    "prohibited_self_overlap_sites",
    "guarded_dormant_spectral_norm_sites",
    "duplicate_top_level_definition_sites",
}
EXPECTED_B1C_STATIC_ENTRYPOINTS = [
    "scripts/run_capable_linear_direct_odt_spectrum.py",
    "scripts/run_direct_odt_clone_oracle.py",
    "scripts/run_implicit_sparse_projective_odt.py",
    "scripts/run_implicit_sparse_projective_odt_all_tokens.py",
    "scripts/run_implicit_sparse_projective_odt_vla.py",
    "tests/test_capable_linear_artifact_integrity.py",
    "tests/test_direct_odt_clone_reference.py",
    "tests/test_direct_odt_truncation.py",
    "tests/test_implicit_sparse_projective_odt.py",
    "tests/test_implicit_sparse_projective_odt_all_tokens.py",
    "tests/test_implicit_sparse_projective_odt_heterogeneous.py",
    "tests/test_implicit_sparse_projective_odt_vla.py",
]
EXPECTED_B1C_TIMING_KEYS = {
    "compile",
    "source_evaluation",
    "algorithm1",
    "algorithms2_and3",
    "through_full_rank",
}
EXPECTED_B1C_FULL_TOP_LEVEL_KEYS = {
    "schema",
    "claim_boundary",
    "deployment_interface_scope",
    "numerical_precision_boundary",
    "checkpoint",
    "inactive_vision_norm_sentinel",
    "capability_evidence",
    "bounded_clone_and_regression_attestation",
    "current_combined_unit_attestation",
    "previous_exact_checkpoint_control",
    "dooms_reference_sha256",
    "dooms_algorithm_order",
    "replay_panel",
    "source_replay",
    "structure",
    "initial_shape",
    "canonical_shape",
    "route_inventory",
    "algorithm1",
    "algorithms2_and3",
    "timings_seconds",
    "peak_rss_mb",
    "full_rank_gates",
    "all_full_rank_gates_pass",
    "runtime_guard",
    "static_audit",
    "source_manifest",
    "bounded_clone_oracle_hash_joined",
    "canonical_direct_odt_algorithms_1_to_3_completed",
    "exact_learned_neural_dag_global_decomposability_certified",
    "historical_capability_path_association_recorded",
    "historical_capability_sha256_bound_to_current_checkpoint",
    "compression_claimed",
}
EXPECTED_ALGORITHM1_KEYS = {
    "step_count",
    "factorization_methods",
    "final_projective_replay_relative_error",
    "full_projective_replay_evaluations",
    "per_step_replays_performed",
    "maximum_local_scaled_reconstruction_relative_error",
    "maximum_absorption_scaled_relative_error",
    "push_ledger",
    "scale_sensitive_occurrence_ledger",
    "canonical_exponent_normal_form",
    "direct_q_provenance",
    "telemetry",
}
_ZERO_UNUSED_FACTORIZATION_COUNTER_KEYS = (
    "s" + "vd_calls",
    "po" + "lar_calls",
    "normal_" + "equation_factorizations",
)
EXPECTED_ALGORITHM1_STATIC_TELEMETRY = {
    "raw_order_three_core_materializations": 0,
    "raw_width_cubic_core_materializations": 0,
    "maximum_persistent_tensor_elements": 3_739_329,
    "maximum_temporary_tensor_elements": 4_722_688,
    "maximum_temporary_tensor_order": 2,
    "local_direct_rq_factorizations": 3_964_463,
    "householder_qr_kernel_calls": 4_143_340,
    "fused_ffn_primitive_count": 0,
    "ephemeral_ffn_bond_diagonalizations": 0,
    "rectangular_direct_rq_factorizations": 33,
    "maximum_rectangular_q_elements": 2_916,
    "bounded_explicit_direct_rq_factorizations": 3_248_703,
    "maximum_bounded_explicit_q_elements": 3_739_329,
    "constant_unary_folds": 0,
    "constant_cp_to_unary_folds": 0,
    "constant_cp_to_constant_folds": 0,
    "direct_q_provenance_certificates": 3_964_463,
    "streamed_direct_q_provenance_certificates": 3_785,
    "streamed_direct_q_replay_qr_factorizations": 91_331,
    "direct_q_columns_compared": 2_616_809_755,
    "maximum_direct_q_transition_elements": 5_484_325,
    "maximum_direct_q_panel_elements": 1_725_185,
    **{key: 0 for key in _ZERO_UNUSED_FACTORIZATION_COUNTER_KEYS},
}
EXPECTED_ALGORITHM1_TELEMETRY_KEYS = set(EXPECTED_ALGORITHM1_STATIC_TELEMETRY) | {
    "maximum_direct_q_compact_relative_error",
    "maximum_direct_q_reconstruction_relative_error",
}
EXPECTED_ALGORITHMS2_AND3_KEYS = {
    "algorithm2_child_messages",
    "algorithm3_diagonalized_nodes",
    "algorithm3_occurrence_ledger",
    "algorithm3_replay_relative_error",
    "maximum_recontracted_offdiagonal_ratio",
    "full_spectra_retained",
    "post_environment_records_retained",
}
EXPECTED_RUNTIME_KEYS = {
    "installed",
    "patched_entrypoint_count",
    "patched_entrypoints",
    "allowed_call_count",
    "allowed_calls",
    "prohibited_attempt_count",
    "prohibited_attempts",
}
EXPECTED_BOUNDED_ORACLE_GATES = {
    "suite_green",
    "exact_count",
    "unique_inventory",
    "required_oracles",
    "static_closed",
    "runtime_closed",
}
EXPECTED_BOUNDED_ORACLES = {
    "test_bounded_rank_deficient_385x386_retains_direct_q_and_matches_clone_every_step",
    "test_unchanged_six_pade_attention_ffn_residual_block_matches_every_step",
    "test_streamed_production_algorithms2_and3_match_separated_clone_reference",
    "test_every_prefix_step_matches_the_no_memo_clone_and_masked_replay",
    "test_materialization_projective_comparison_ignores_rowwise_radial_scale",
}
EXPECTED_BOUNDED_SHARED_CORES = {
    "xvla/train/__init__.py",
    "xvla/train/implicit_sparse_projective_odt_all_tokens.py",
    "xvla/train/implicit_sparse_projective_odt.py",
    "xvla/train/implicit_sparse_projective_odt_vla.py",
    "xvla/train/direct_odt_truncation.py",
    "xvla/train/direct_odt_clone_reference.py",
}
EXPECTED_BOUNDED_ORACLE_KEYS = {
    "path",
    "sha256",
    "test_count",
    "required_oracles",
    "gates",
    "shared_core_sha256",
}
EXPECTED_B1C_CURRENT_UNIT_KEYS = {
    "path",
    "sha256",
    "direct_result_path",
    "direct_result_sha256",
    "test_counts",
    "input_sha256",
    "gates",
}
EXPECTED_B1C_CURRENT_UNIT_GATE_KEYS = {
    "schema",
    "unit_self_gate",
    "isolated_processes",
    "manifest_identities",
    "direct_result_identity",
    "direct_suite",
    "capability_suite",
    "integrity_suite",
    "direct_static",
    "direct_runtime",
}
EXPECTED_RTX_UNIT_KEYS = {
    "schema",
    "test_counts",
    "junit",
    "elapsed_seconds",
    "capability_source_manifest_sha256",
    "cross_stage_source_manifest_sha256",
    "stage_ledger_sha256",
    "static_audit",
    "gates",
    "all_gates_pass",
}
EXPECTED_RTX_UNIT_JUNIT_KEYS = {
    "tests",
    "failures",
    "errors",
    "skipped",
    "test_names",
}
EXPECTED_RTX_UNIT_GATE_KEYS = {
    "process_exit_zero",
    "exact_test_count",
    "no_failures",
    "no_skips",
    "unique_test_names",
    "hardware_and_join_regressions_present",
}
EXPECTED_RTX_CAPABILITY_TEST_COUNT = 94
EXPECTED_RTX_CROSS_STAGE_TEST_COUNT = 132
EXPECTED_RTX_UNIT_TEST_NAME_SHA256 = (
    "aa52500b55f4fb8c3d5c069cdc1f928480c633404b8a9dea666a6d89d244bc09"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _type_exact_equal(observed: Any, expected: Any) -> bool:
    """Compare JSON-shaped authorities without Python's bool/int aliases."""
    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        return set(observed) == set(expected) and all(
            _type_exact_equal(observed[key], expected[key]) for key in expected
        )
    if type(expected) in (list, tuple):
        return len(observed) == len(expected) and all(
            _type_exact_equal(left, right)
            for left, right in zip(observed, expected)
        )
    return bool(observed == expected)


def _safe_member(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or not value.parts or any(
        part in {"", ".", ".."} for part in value.parts
    ):
        raise RuntimeError(f"unsafe authenticated path {relative!r}")
    cursor = root
    for part in value.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(f"authenticated path traverses a link: {relative}")
    resolved = cursor.resolve(strict=True)
    if resolved == root or root not in resolved.parents or not resolved.is_file():
        raise RuntimeError(f"authenticated path escapes or is nonphysical: {relative}")
    if resolved.relative_to(root).as_posix() != relative:
        raise RuntimeError(f"authenticated path is not canonical: {relative}")
    return resolved


def _read_line_manifest(path: Path, root: Path, label: str) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or nonphysical")
    result: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(f"malformed {label} line {number}")
        digest, relative = fields
        relative = relative.lstrip("*")
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative in result
        ):
            raise RuntimeError(f"invalid {label} line {number}")
        member = _safe_member(root, relative)
        if _sha256(member) != digest:
            raise RuntimeError(f"{label} byte mismatch: {relative}")
        result[relative] = digest
    if not result:
        raise RuntimeError(f"{label} is empty")
    return result


def _preimport_verification() -> dict[str, Any]:
    if PROJECT_ROOT != EXPECTED_STAGE_ROOT:
        raise RuntimeError(f"cross-stage attestation must run from {EXPECTED_STAGE_ROOT}")
    source = _read_line_manifest(SOURCE_MANIFEST, PROJECT_ROOT, "source manifest")
    ledger = _read_line_manifest(STAGE_LEDGER, PROJECT_ROOT, "stage ledger")
    required = {
        "scripts/attest_capable_linear_rtx6000_cross_stage.py",
        "scripts/run_capable_linear_rtx6000_capability.py",
        "scripts/capable_linear_artifact_integrity.py",
        "athena/capable_linear_rtx6000_cross_stage_sources.sha256",
        "athena/capable_linear_rtx6000_capability_sources.sha256",
        "inputs/capable_linear_b1c0_checkpoint.pt",
        "inputs/capable_linear_training.json",
        "inputs/b1c_stage_ledger.sha256",
        "inputs/b1c_odt_sources.sha256",
        "inputs/b1c_capability_sources.sha256",
    }
    if not required.issubset(ledger):
        raise RuntimeError("stage ledger omits a cross-stage authority")
    if ledger["athena/capable_linear_rtx6000_cross_stage_sources.sha256"] != _sha256(
        SOURCE_MANIFEST
    ):
        raise RuntimeError("stage ledger does not bind cross-stage source manifest")
    for relative, digest in source.items():
        if ledger.get(relative) != digest:
            raise RuntimeError(f"stage ledger does not bind source {relative}")
    if ledger["inputs/capable_linear_b1c0_checkpoint.pt"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("local staged checkpoint identity differs")
    if (
        ledger["inputs/capable_linear_training.json"]
        != EXPECTED_TRAINING_RECORD_SHA256
        or _sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256
        or _sha256(TRAINING_RECORD) != EXPECTED_TRAINING_RECORD_SHA256
    ):
        raise RuntimeError("local staged checkpoint or inference auxiliary differs")
    if ledger["inputs/b1c_capability_sources.sha256"] != (
        "c62224734d4d9a89c0e9f1fd638b7d5318967531f39adcacd715ac1eb1e6b897"
    ):
        raise RuntimeError("local stage does not bind the frozen b1c capability closure")
    return {
        "source": source,
        "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
        "ledger": ledger,
        "stage_ledger_sha256": _sha256(STAGE_LEDGER),
    }


_RUNNING_AS_ENTRYPOINT = __name__ == "__main__"
_AUDIT_ONLY = _RUNNING_AS_ENTRYPOINT and sys.argv[1:] == ["--audit-only"]
_PREIMPORT = (
    None if not _RUNNING_AS_ENTRYPOINT or _AUDIT_ONLY else _preimport_verification()
)
sys.path[:] = [str(PROJECT_ROOT)] + [
    entry for entry in sys.path if entry != str(PROJECT_ROOT)
]
sys.dont_write_bytecode = True

from scripts.odt_direct_only_compliance import audit_direct_only_launch
from scripts.capable_linear_artifact_integrity import (
    assert_physical_hashes_unchanged,
    read_authenticated_json,
)


_ENTRYPOINTS = (
    Path(__file__),
    PROJECT_ROOT / "scripts/run_capable_linear_rtx6000_capability.py",
    PROJECT_ROOT / "scripts/run_capable_linear_rtx6000_tests.py",
    PROJECT_ROOT / "tests/test_capable_linear_rtx6000_cross_stage.py",
    PROJECT_ROOT / "tests/test_capable_linear_rtx6000_capability.py",
)
if len(_ENTRYPOINTS) != len({path.resolve() for path in _ENTRYPOINTS}):
    raise RuntimeError("cross-stage entrypoint inventory contains duplicates")
_STATIC_AUDIT = audit_direct_only_launch(
    PROJECT_ROOT, _ENTRYPOINTS, require_direct_qr=False
)
for _field in (
    "prohibited_calls_found",
    "prohibited_self_overlap_sites",
    "guarded_dormant_spectral_norm_sites",
    "duplicate_top_level_definition_sites",
):
    if _STATIC_AUDIT[_field]:
        raise RuntimeError(f"cross-stage static audit did not close {_field}")
if _PREIMPORT is not None and _PREIMPORT["source"] != _STATIC_AUDIT["source_sha256"]:
    raise RuntimeError("cross-stage source manifest differs from audited closure")
if _RUNNING_AS_ENTRYPOINT and _AUDIT_ONLY:
    print(json.dumps(_STATIC_AUDIT, indent=2, sort_keys=True))
    raise SystemExit(0)

from scripts import run_capable_linear_rtx6000_capability as capability_lane


EXPECTED_CAPABILITY_INITIAL_STATE_ROW_BUNDLE_SHA256 = {
    0: "9ef09c00e35ed5beda072328143a3b6ce962c58b714c2c53c2711c29066dbf7d",
    1: "09d76bb222136707ddf9cf3d688ac3fa456d60818ac01067bbb167f1f08e2ae7",
    2: "2194addd7641a43d3ca751d84410a3b4229e6ac0da596970d188e17a968def2d",
    3: "2dc2f081b68d62e5d4a714886939b4f191b922516f3df107acd00d2837b09ec8",
    4: "703790cfc208449cd8dc2503125d40c6a2d4d247a58c600ede13f4a785f41a60",
    5: "69ddf8dd1494fc054659e765962da0ba3a519692abcb71939d06cc58e8dccb96",
    6: "8464f4f0f67c056268eb1894230e457558bdd0714780ed2c35973545ba3eeeec",
    7: "56a550b5aeb73f009cce0076d78e001f30122d8b303e889faa5c58397831bef4",
    8: "75083bc1ac59459eb8fe0e54018ed1fb79d614ee54560a94a677801627ad0502",
    9: "4117ca940b7388a9a2abe308722c921c71462050a8cc2b44394a6c98467ed445",
}
EXPECTED_CAPABILITY_SMOKE_INITIAL_STATE_SHA256 = (
    "8aecc04c67f51f29b65ab55cd27984f4a2db52d9833165e91effe5c1cc3746d2"
)
EXPECTED_CAPABILITY_EPISODE_KEYS = frozenset(
    {
        "task_index",
        "episode",
        "initial_state_sha256",
        "success",
        "steps",
        "terminated_without_success",
        "elapsed_s",
    }
)
EXPECTED_CAPABILITY_ROLLOUT_KEYS = frozenset(
    {
        "schema",
        "checkpoint_sha256",
        "checkpoint",
        "training_record_sha256",
        "training_record_used_as_fixed_inference_auxiliary",
        "historical_training_to_checkpoint_digest_link_claimed",
        "task_start",
        "task_end",
        "seed",
        "protocol",
        "task_protocol",
        "episodes",
        "episode_identity_sha256",
        "successes",
        "trials",
        "overall",
        "per_task",
        "early_terminal_failures",
        "validated_transition_counts",
        "synthetic_output_finite",
        "inactive_vision_norm_sentinel",
        "environment",
        "source_manifest_sha256",
        "stage_ledger_sha256",
        "static_audit",
        "canonical_odt_runtime_guard_installed",
        "odt_runtime_compliance_claimed",
        "simulator_external_to_odt",
        "external_simulator_outside_weight_only_odt_closure",
        "exact_checkpoint_digest_verified_before_and_after",
        "stage_ledger_verified_before_and_after",
        "external_simulator_part_of_rtx6000_capability_measurement",
        "historical_hardware_reproduction_claimed",
        "canonical_odt_numerical_operations_performed",
        "smoke_only",
        "smoke_gate_sha256",
        "all_identity_and_protocol_gates_pass",
        "elapsed_s",
    }
)
EXPECTED_CAPABILITY_ROLLOUT_PROTOCOL_KEYS = frozenset(
    {
        "suite",
        "resolution",
        "action_horizon",
        "execution_horizon",
        "settle_steps",
        "settle_success_or_done_count",
        "episodes_per_task",
        "max_steps",
        "canonical_init_states",
        "observation_transform",
        "gripper_decode",
        "matmul_precision",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cublas_workspace_config",
        "deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "historical_evaluator_control_flow_reproduced",
        "historical_protocol_with_corrected_terminal_handling",
        "terminal_transition_stops_episode",
        "historical_hardware_reproduction_claimed",
        "fresh_hardware",
        "slurm_node",
    }
)
EXPECTED_CAPABILITY_AGGREGATE_PROTOCOL_KEYS = frozenset(
    {
        *EXPECTED_CAPABILITY_ROLLOUT_PROTOCOL_KEYS,
        "tasks",
        "canonical_initial_states_per_task",
        "trials",
    }
)
EXPECTED_CAPABILITY_AGGREGATE_KEYS = frozenset(
    {
        "schema",
        "checkpoint_sha256",
        "training_record_sha256",
        "seed",
        "fresh_evaluation_sha256_bound_to_checkpoint",
        "config_identity",
        "historical_evaluation_digest_binding_claimed",
        "protocol",
        "shards",
        "inactive_vision_norm_sentinel",
        "environment",
        "smoke_gate_sha256",
        "shard_result_bundle_sha256",
        "successes",
        "trials",
        "success_rate",
        "per_task_successes",
        "per_task",
        "episodes",
        "early_terminal_failures",
        "validated_transition_counts",
        "capability_floor",
        "capability_floor_passed",
        "episode_identity_sha256",
        "gates",
        "all_gates_pass",
        "source_manifest_sha256",
        "stage_ledger_sha256",
        "static_audit",
        "exact_checkpoint_digest_verified_before_and_after",
        "stage_ledger_verified_before_and_after",
        "canonical_odt_runtime_guard_installed",
        "odt_runtime_compliance_claimed",
        "simulator_external_to_odt",
        "external_simulator_outside_weight_only_odt_closure",
        "historical_hardware_reproduction_claimed",
        "fresh_all_rtx6000_hash_bound_reevaluation",
        "a30_hardware_reproduction_claimed",
        "node05_execution_required",
        "historical_hardware_was_mixed_and_is_comparison_only",
        "determinism_claim_scope",
        "claim",
    }
)
EXPECTED_CAPABILITY_SHARD_RECORD_KEYS = frozenset(
    {"task_start", "task_end", "sha256", "successes", "trials"}
)
EXPECTED_CAPABILITY_AGGREGATE_SENTINEL_KEYS = frozenset(
    {
        "site",
        "total_site_count",
        "inactive_sites",
        "active_site_count",
        "shard_count",
        "all_shard_call_counts_zero",
        "all_shard_active_sets_observed_once",
    }
)
EXPECTED_CAPABILITY_CHECKPOINT_KEYS = frozenset(
    {
        "checkpoint_sha256",
        "checkpoint_bytes",
        "state_count",
        "state_key_sha256",
        "normalization_site_count",
        "active_normalization_site_count",
        "normalization_buffer_inventory",
        "parameters",
        "config_identity",
    }
)
EXPECTED_CAPABILITY_SENTINEL_KEYS = frozenset(
    {
        "site",
        "armed_during_sentinel_replay_and_all_rollout_policy_forwards",
        "call_count",
        "total_site_count",
        "inactive_sites",
        "active_site_count",
        "active_call_count",
        "every_active_site_call_count",
        "source_output_bitwise_equal_without_sentinels",
    }
)
EXPECTED_CAPABILITY_ENVIRONMENT_KEYS = frozenset(
    {
        "python",
        "numpy",
        "torch",
        "libero",
        "robosuite",
        "mujoco",
        "pillow",
        "cuda",
        "gpu",
        "slurm_node",
        "mujoco_gl",
        "matmul_precision",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cublas_workspace_config",
        "deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
    }
)
EXPECTED_CAPABILITY_TRANSITION_COUNT_KEYS = frozenset({"settle", "policy"})

if (
    EXPECTED_CAPABILITY_INITIAL_STATE_ROW_BUNDLE_SHA256
    != capability_lane.EXPECTED_INITIAL_STATE_ROW_BUNDLE_SHA256
    or EXPECTED_CAPABILITY_SMOKE_INITIAL_STATE_SHA256
    != capability_lane.EXPECTED_SMOKE_INITIAL_STATE_SHA256
    or EXPECTED_CAPABILITY_EPISODE_KEYS != capability_lane.EPISODE_RECORD_KEYS
    or EXPECTED_CAPABILITY_ROLLOUT_KEYS != capability_lane.ROLLOUT_RESULT_KEYS
    or EXPECTED_CAPABILITY_ROLLOUT_PROTOCOL_KEYS
    != capability_lane.ROLLOUT_PROTOCOL_KEYS
    or EXPECTED_CAPABILITY_AGGREGATE_PROTOCOL_KEYS
    != capability_lane.AGGREGATE_PROTOCOL_KEYS
    or EXPECTED_CAPABILITY_AGGREGATE_KEYS != capability_lane.AGGREGATE_RESULT_KEYS
    or EXPECTED_CAPABILITY_SHARD_RECORD_KEYS != capability_lane.SHARD_RECORD_KEYS
    or EXPECTED_CAPABILITY_AGGREGATE_SENTINEL_KEYS
    != capability_lane.AGGREGATE_SENTINEL_KEYS
    or EXPECTED_CAPABILITY_CHECKPOINT_KEYS != capability_lane.CHECKPOINT_RECORD_KEYS
    or EXPECTED_CAPABILITY_SENTINEL_KEYS
    != capability_lane.INACTIVE_NORM_SENTINEL_KEYS
    or EXPECTED_CAPABILITY_ENVIRONMENT_KEYS
    != capability_lane.ENVIRONMENT_RECORD_KEYS
    or EXPECTED_CAPABILITY_TRANSITION_COUNT_KEYS
    != capability_lane.TRANSITION_COUNT_KEYS
):
    raise RuntimeError("capability producer and cross-stage schema authorities differ")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _physical_read_only(path: Path, label: str) -> None:
    observed = os.lstat(path)
    _require(stat.S_ISREG(observed.st_mode), f"{label} is not a physical file")
    _require(observed.st_mode & 0o222 == 0, f"{label} is writable")


def _hex_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _episode_records_are_exact(
    records: Any,
    start: int,
    stop: int,
    *,
    max_steps: int,
    episodes_per_task: int = capability_lane.EPISODES_PER_TASK,
) -> tuple[bool, list[tuple[int, int]], int, dict[str, int]]:
    expected = [
        (task, episode)
        for task in range(start, stop)
        for episode in range(episodes_per_task)
    ]
    if not isinstance(records, list) or len(records) != len(expected):
        return False, [], -1, {}
    observed: list[tuple[int, int]] = []
    successes = 0
    per_task = {str(task): 0 for task in range(start, stop)}
    for record in records:
        if not isinstance(record, dict) or set(record) != EXPECTED_CAPABILITY_EPISODE_KEYS:
            return False, observed, -1, {}
        task = record.get("task_index")
        episode = record.get("episode")
        success = record.get("success")
        steps = record.get("steps")
        terminated = record.get("terminated_without_success")
        elapsed = record.get("elapsed_s")
        if (
            type(task) is not int
            or type(episode) is not int
            or type(success) is not bool
            or type(steps) is not int
            or type(terminated) is not bool
            or not _hex_digest(record.get("initial_state_sha256"))
            or not 1 <= steps <= max_steps
            or (terminated and success)
            or not (success or terminated or steps == max_steps)
            or type(elapsed) is not float
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
        ):
            return False, observed, -1, {}
        observed.append((task, episode))
        successes += int(success)
        if str(task) not in per_task:
            return False, observed, -1, {}
        per_task[str(task)] += int(success)
    return observed == expected and len(set(observed)) == len(expected), observed, successes, per_task


def _episode_identity_rows(records: Any) -> list[tuple[int, int, str]]:
    if not isinstance(records, list):
        return []
    rows: list[tuple[int, int, str]] = []
    for record in records:
        if not isinstance(record, dict):
            return []
        task = record.get("task_index")
        episode = record.get("episode")
        row_sha = record.get("initial_state_sha256")
        if type(task) is not int or type(episode) is not int or not _hex_digest(row_sha):
            return []
        rows.append((task, episode, row_sha))
    return rows


def _initial_state_row_bundles_are_exact(
    records: Any,
    start: int,
    stop: int,
    *,
    episodes_per_task: int,
) -> bool:
    if not isinstance(records, list):
        return False
    if episodes_per_task == 1:
        return (
            start == 0
            and stop == 1
            and len(records) == 1
            and records[0].get("initial_state_sha256")
            == EXPECTED_CAPABILITY_SMOKE_INITIAL_STATE_SHA256
        )
    if episodes_per_task != capability_lane.EPISODES_PER_TASK:
        return False
    for task in range(start, stop):
        row_hashes = [
            record.get("initial_state_sha256")
            for record in records
            if record.get("task_index") == task
        ]
        if (
            len(row_hashes) != capability_lane.EPISODES_PER_TASK
            or _canonical_sha256(row_hashes)
            != EXPECTED_CAPABILITY_INITIAL_STATE_ROW_BUNDLE_SHA256[task]
        ):
            return False
    return True


def _row_identity_evidence_is_exact(
    value: Any,
    start: int,
    stop: int,
    *,
    max_steps: int,
    episodes_per_task: int,
) -> bool:
    if not isinstance(value, dict):
        return False
    records = value.get("episodes")
    inventory, pairs, _successes, _per_task = _episode_records_are_exact(
        records,
        start,
        stop,
        max_steps=max_steps,
        episodes_per_task=episodes_per_task,
    )
    return (
        inventory
        and pairs
        == [
            (task, episode)
            for task in range(start, stop)
            for episode in range(episodes_per_task)
        ]
        and _initial_state_row_bundles_are_exact(
            records, start, stop, episodes_per_task=episodes_per_task
        )
        and value.get("episode_identity_sha256")
        == _canonical_sha256(_episode_identity_rows(records))
    )


def _aggregate_episode_copy_is_exact(
    value: Any, shard_episodes: list[dict[str, Any]]
) -> bool:
    return (
        isinstance(value, dict)
        and value.get("episodes") == shard_episodes
        and _row_identity_evidence_is_exact(
            value,
            0,
            10,
            max_steps=capability_lane.MAX_STEPS,
            episodes_per_task=capability_lane.EPISODES_PER_TASK,
        )
    )


def _capability_source_identity() -> tuple[dict[str, str], dict[str, Any]]:
    manifest = _read_line_manifest(
        CAPABILITY_SOURCE_MANIFEST, PROJECT_ROOT, "capability source manifest"
    )
    entrypoints = (
        PROJECT_ROOT / "scripts/run_capable_linear_rtx6000_capability.py",
        PROJECT_ROOT / "tests/test_capable_linear_rtx6000_capability.py",
    )
    audit = audit_direct_only_launch(
        PROJECT_ROOT, entrypoints, require_direct_qr=False
    )
    _require(audit["source_sha256"] == manifest, "capability source audit differs")
    _require(
        _sha256(capability_lane.PINNED_B1C_CAPABILITY_MANIFEST)
        == capability_lane.PINNED_B1C_CAPABILITY_MANIFEST_SHA256
        and capability_lane._shared_source_identity_is_exact(
            manifest,
            capability_lane._parse_line_manifest(
                capability_lane.PINNED_B1C_CAPABILITY_MANIFEST,
                "pinned b1c capability source manifest",
            ),
        ),
        "capability closure is not the exact frozen b1c shared closure",
    )
    for field in (
        "prohibited_calls_found",
        "prohibited_self_overlap_sites",
        "guarded_dormant_spectral_norm_sites",
        "duplicate_top_level_definition_sites",
    ):
        _require(audit[field] == [], f"capability source audit failed {field}")
    return manifest, audit


def _capability_checkpoint_record_is_exact(value: Any) -> bool:
    expected = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_bytes": EXPECTED_B1C_CHECKPOINT_BYTES,
        "state_count": 540,
        "state_key_sha256": capability_lane.EXPECTED_CHECKPOINT_STATE_KEY_SHA256,
        "normalization_site_count": 74,
        "active_normalization_site_count": 73,
        "normalization_buffer_inventory": (
            capability_lane.EXPECTED_NORMALIZATION_BUFFER_INVENTORY
        ),
        "parameters": 20_137_352,
        "config_identity": EXPECTED_CONFIG_IDENTITY,
    }
    return (
        set(expected) == EXPECTED_CAPABILITY_CHECKPOINT_KEYS
        and _type_exact_equal(value, expected)
    )


def _capability_sentinel_is_exact(value: Any) -> bool:
    expected = {
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
    return (
        set(expected) == EXPECTED_CAPABILITY_SENTINEL_KEYS
        and _type_exact_equal(value, expected)
    )


def _expected_capability_rollout_protocol(
    episodes_per_task: int, max_steps: int
) -> dict[str, Any]:
    if type(episodes_per_task) is not int or type(max_steps) is not int:
        raise TypeError("rollout protocol counts must be literal integers")
    return {
        "suite": "libero_object",
        "resolution": 64,
        "action_horizon": 8,
        "execution_horizon": 8,
        "settle_steps": 10,
        "settle_success_or_done_count": 0,
        "episodes_per_task": episodes_per_task,
        "max_steps": max_steps,
        "canonical_init_states": True,
        "observation_transform": "agentview_image[::-1,::-1] then same-size PIL resize",
        "gripper_decode": "+1 iff predicted coordinate > 0, else -1",
        "matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "historical_evaluator_control_flow_reproduced": False,
        "historical_protocol_with_corrected_terminal_handling": True,
        "terminal_transition_stops_episode": True,
        "historical_hardware_reproduction_claimed": False,
        "fresh_hardware": "Quadro RTX 6000",
        "slurm_node": "c2-g8-05",
    }


def _rollout_protocol_is_exact(value: Any, episodes_per_task: int, max_steps: int) -> bool:
    expected = _expected_capability_rollout_protocol(
        episodes_per_task, max_steps
    )
    return (
        set(expected) == EXPECTED_CAPABILITY_ROLLOUT_PROTOCOL_KEYS
        and _type_exact_equal(value, expected)
    )


def _aggregate_protocol_is_exact(value: Any) -> bool:
    expected = {
        **_expected_capability_rollout_protocol(50, 280),
        "tasks": list(range(10)),
        "canonical_initial_states_per_task": list(range(50)),
        "trials": 500,
    }
    return (
        set(expected) == EXPECTED_CAPABILITY_AGGREGATE_PROTOCOL_KEYS
        and _type_exact_equal(value, expected)
    )


def _capability_environment_is_exact(value: Any) -> bool:
    expected = {
        "python": "3.10.19",
        "numpy": "1.26.4",
        "torch": "2.7.1+cu126",
        "libero": "0.1.0",
        "robosuite": "1.4.1",
        "mujoco": "3.5.0",
        "pillow": "12.1.1",
        "cuda": "12.6",
        "gpu": "Quadro RTX 6000",
        "slurm_node": "c2-g8-05",
        "mujoco_gl": "egl",
        "matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }
    return (
        set(expected) == EXPECTED_CAPABILITY_ENVIRONMENT_KEYS
        and _type_exact_equal(value, expected)
    )


def _capability_artifact_schema_is_exact(value: Any, *, aggregate: bool) -> bool:
    if not isinstance(value, dict):
        return False
    episodes = value.get("episodes")
    if (
        not isinstance(episodes, list)
        or any(
            not isinstance(record, dict)
            or set(record) != EXPECTED_CAPABILITY_EPISODE_KEYS
            for record in episodes
        )
    ):
        return False
    transition_counts = value.get("validated_transition_counts")
    if (
        not isinstance(transition_counts, dict)
        or set(transition_counts) != EXPECTED_CAPABILITY_TRANSITION_COUNT_KEYS
        or any(type(item) is not int for item in transition_counts.values())
    ):
        return False
    if aggregate:
        shards = value.get("shards")
        sentinel = value.get("inactive_vision_norm_sentinel")
        return (
            set(value) == EXPECTED_CAPABILITY_AGGREGATE_KEYS
            and isinstance(value.get("protocol"), dict)
            and set(value["protocol"])
            == EXPECTED_CAPABILITY_AGGREGATE_PROTOCOL_KEYS
            and isinstance(shards, list)
            and len(shards) == 4
            and all(
                isinstance(record, dict)
                and set(record) == EXPECTED_CAPABILITY_SHARD_RECORD_KEYS
                for record in shards
            )
            and isinstance(sentinel, dict)
            and set(sentinel) == EXPECTED_CAPABILITY_AGGREGATE_SENTINEL_KEYS
            and _capability_environment_is_exact(value.get("environment"))
        )
    return (
        set(value) == EXPECTED_CAPABILITY_ROLLOUT_KEYS
        and isinstance(value.get("protocol"), dict)
        and set(value["protocol"]) == EXPECTED_CAPABILITY_ROLLOUT_PROTOCOL_KEYS
        and isinstance(value.get("checkpoint"), dict)
        and set(value["checkpoint"]) == EXPECTED_CAPABILITY_CHECKPOINT_KEYS
        and isinstance(value.get("inactive_vision_norm_sentinel"), dict)
        and set(value["inactive_vision_norm_sentinel"])
        == EXPECTED_CAPABILITY_SENTINEL_KEYS
        and _capability_environment_is_exact(value.get("environment"))
    )


def _transition_counts_are_exact(
    value: Any, records: Any, *, expected_settle: int
) -> bool:
    counts = value.get("validated_transition_counts") if isinstance(value, dict) else None
    expected = (
        {
            "settle": expected_settle,
            "policy": sum(
                record.get("steps", -1)
                for record in records
                if isinstance(record, dict)
            ),
        }
        if isinstance(records, list)
        else None
    )
    return (
        isinstance(records, list)
        and set(expected) == EXPECTED_CAPABILITY_TRANSITION_COUNT_KEYS
        and _type_exact_equal(counts, expected)
    )


def _common_capability_gates(
    value: Mapping[str, Any], source_sha: str, stage_sha: str, audit: Mapping[str, Any]
) -> bool:
    return (
        value.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256
        and value.get("source_manifest_sha256") == source_sha
        and value.get("stage_ledger_sha256") == stage_sha
        and _type_exact_equal(value.get("static_audit"), dict(audit))
        and value.get("canonical_odt_runtime_guard_installed") is False
        and value.get("odt_runtime_compliance_claimed") is False
        and value.get("simulator_external_to_odt") is True
        and value.get("external_simulator_outside_weight_only_odt_closure") is True
        and value.get("historical_hardware_reproduction_claimed") is False
        and _capability_environment_is_exact(value.get("environment"))
    )


def _producer_identity_gates(value: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "training_record": value.get("training_record_sha256")
        == EXPECTED_TRAINING_RECORD_SHA256,
        "seed": type(value.get("seed")) is int and value.get("seed") == 0,
        "checkpoint_twice": value.get(
            "exact_checkpoint_digest_verified_before_and_after"
        )
        is True,
        "ledger_twice": value.get("stage_ledger_verified_before_and_after") is True,
    }


def _read_capability_bundle() -> dict[str, Any]:
    manifest, capability_audit = _capability_source_identity()
    source_sha = _sha256(CAPABILITY_SOURCE_MANIFEST)
    stage_sha = _sha256(STAGE_LEDGER)
    paths = {
        "checkpoint": CHECKPOINT,
        "training_record": TRAINING_RECORD,
        "aggregate": CAPABILITY_AGGREGATE,
        "smoke": CAPABILITY_SMOKE,
        **{
            f"shard_{start}_{stop}": RESULT_DIRECTORY
            / f"rtx6000_capability_t{start}_{stop}.json"
            for start, stop in SHARDS
        },
    }
    snapshot = assert_physical_hashes_unchanged(paths)
    _require(
        snapshot["checkpoint"] == EXPECTED_CHECKPOINT_SHA256
        and snapshot["training_record"] == EXPECTED_TRAINING_RECORD_SHA256,
        "staged capability input bytes differ",
    )
    for label, path in paths.items():
        _physical_read_only(path, label)
    smoke, smoke_sha = read_authenticated_json(CAPABILITY_SMOKE, "RTX 6000 smoke")
    _require(smoke_sha == snapshot["smoke"], "smoke fields and digest differ")
    smoke_episodes = smoke.get("episodes")
    smoke_inventory, smoke_pairs, smoke_successes, _smoke_per_task = (
        _episode_records_are_exact(
            smoke_episodes,
            0,
            1,
            max_steps=1,
            episodes_per_task=1,
        )
    )
    smoke_ok = (
        _capability_artifact_schema_is_exact(smoke, aggregate=False)
        and smoke.get("schema")
        == "xvla_capable_linear_b1c0_rtx6000_capability_smoke_v1"
        and _type_exact_equal(smoke.get("task_start"), 0)
        and _type_exact_equal(smoke.get("task_end"), 1)
        and smoke_inventory
        and smoke_pairs == [(0, 0)]
        and _row_identity_evidence_is_exact(
            smoke, 0, 1, max_steps=1, episodes_per_task=1
        )
        and _type_exact_equal(smoke.get("successes"), smoke_successes)
        and _type_exact_equal(smoke.get("trials"), 1)
        and _type_exact_equal(smoke.get("overall"), float(smoke_successes))
        and _type_exact_equal(
            smoke.get("per_task"), {"0": float(smoke_successes)}
        )
        and _type_exact_equal(
            smoke.get("early_terminal_failures"),
            sum(
                int(record["terminated_without_success"])
                for record in smoke_episodes
            ),
        )
        and _transition_counts_are_exact(
            smoke, smoke_episodes, expected_settle=capability_lane.SETTLE_STEPS
        )
        and smoke.get("smoke_only") is True
        and smoke.get("smoke_gate_sha256") is None
        and _rollout_protocol_is_exact(smoke.get("protocol"), 1, 1)
        and smoke.get("all_identity_and_protocol_gates_pass") is True
        and _common_capability_gates(smoke, source_sha, stage_sha, capability_audit)
        and all(_producer_identity_gates(smoke).values())
        and _type_exact_equal(
            smoke.get("task_protocol"),
            {"0": capability_lane._expected_task_record(0)},
        )
        and _capability_checkpoint_record_is_exact(smoke.get("checkpoint"))
        and _capability_sentinel_is_exact(
            smoke.get("inactive_vision_norm_sentinel")
        )
        and smoke.get("training_record_used_as_fixed_inference_auxiliary") is True
        and smoke.get("historical_training_to_checkpoint_digest_link_claimed")
        is False
        and smoke.get("synthetic_output_finite") is True
        and smoke.get("external_simulator_part_of_rtx6000_capability_measurement")
        is True
        and smoke.get("canonical_odt_numerical_operations_performed") is False
        and _positive_finite(smoke.get("elapsed_s"))
    )
    _require(smoke_ok, "RTX 6000 smoke attestation differs")

    all_episodes: list[dict[str, Any]] = []
    shard_hashes: dict[str, str] = {}
    shard_records: list[dict[str, Any]] = []
    shard_transition_counts: list[dict[str, int]] = []
    for start, stop in SHARDS:
        label = f"shard_{start}_{stop}"
        path = paths[label]
        shard, digest = read_authenticated_json(path, f"RTX 6000 shard {start}:{stop}")
        _require(digest == snapshot[label], f"{label} fields and digest differ")
        exact, pairs, successes, per_task_successes = _episode_records_are_exact(
            shard.get("episodes"), start, stop, max_steps=capability_lane.MAX_STEPS
        )
        expected_task_protocol = {
            str(task): capability_lane._expected_task_record(task)
            for task in range(start, stop)
        }
        expected_per_task = {
            key: value / capability_lane.EPISODES_PER_TASK
            for key, value in per_task_successes.items()
        }
        early_terminal_failures = (
            sum(int(record["terminated_without_success"]) for record in shard["episodes"])
            if exact
            else -1
        )
        protocol = shard.get("protocol", {})
        gates = {
            "schema": _capability_artifact_schema_is_exact(
                shard, aggregate=False
            )
            and shard.get("schema")
            == "xvla_capable_linear_b1c0_rtx6000_capability_shard_v1",
            "range": _type_exact_equal(shard.get("task_start"), start)
            and _type_exact_equal(shard.get("task_end"), stop),
            "episodes": exact,
            "episode_digest": _row_identity_evidence_is_exact(
                shard,
                start,
                stop,
                max_steps=280,
                episodes_per_task=50,
            ),
            "task_protocol": _type_exact_equal(
                shard.get("task_protocol"), expected_task_protocol
            ),
            "counts": _type_exact_equal(shard.get("successes"), successes)
            and _type_exact_equal(shard.get("trials"), len(pairs))
            and _type_exact_equal(shard.get("overall"), successes / len(pairs))
            and _type_exact_equal(shard.get("per_task"), expected_per_task)
            and _type_exact_equal(
                shard.get("early_terminal_failures"), early_terminal_failures
            ),
            "transitions": _transition_counts_are_exact(
                shard,
                shard.get("episodes"),
                expected_settle=(stop - start)
                * capability_lane.EPISODES_PER_TASK
                * capability_lane.SETTLE_STEPS,
            ),
            "protocol": _rollout_protocol_is_exact(protocol, 50, 280),
            "smoke": shard.get("smoke_gate_sha256") == smoke_sha,
            "source": _common_capability_gates(
                shard, source_sha, stage_sha, capability_audit
            ),
            "producer_identity": all(_producer_identity_gates(shard).values()),
            "config": _capability_checkpoint_record_is_exact(
                shard.get("checkpoint")
            ),
            "sentinel": _capability_sentinel_is_exact(
                shard.get("inactive_vision_norm_sentinel")
            ),
            "self": shard.get("all_identity_and_protocol_gates_pass") is True
            and shard.get("training_record_used_as_fixed_inference_auxiliary") is True
            and shard.get("historical_training_to_checkpoint_digest_link_claimed")
            is False
            and shard.get("synthetic_output_finite") is True
            and shard.get(
                "external_simulator_part_of_rtx6000_capability_measurement"
            )
            is True
            and shard.get("canonical_odt_numerical_operations_performed") is False
            and shard.get("smoke_only") is False
            and _positive_finite(shard.get("elapsed_s")),
        }
        _require(all(gates.values()), f"RTX 6000 shard {start}:{stop} failed {gates}")
        all_episodes.extend(shard["episodes"])
        shard_hashes[f"{start}:{stop}"] = digest
        shard_records.append(
            {
                "task_start": start,
                "task_end": stop,
                "sha256": digest,
                "successes": successes,
                "trials": len(pairs),
            }
        )
        shard_transition_counts.append(dict(shard["validated_transition_counts"]))

    aggregate, aggregate_sha = read_authenticated_json(
        CAPABILITY_AGGREGATE, "RTX 6000 aggregate"
    )
    _require(aggregate_sha == snapshot["aggregate"], "aggregate fields and digest differ")
    exact, pairs, successes, per_task_successes = _episode_records_are_exact(
        aggregate.get("episodes"), 0, 10, max_steps=capability_lane.MAX_STEPS
    )
    aggregate_early_terminal_failures = (
        sum(
            int(record["terminated_without_success"])
            for record in aggregate["episodes"]
        )
        if exact
        else -1
    )
    protocol = aggregate.get("protocol", {})
    aggregate_gates = aggregate.get("gates")
    gates = {
        "schema": _capability_artifact_schema_is_exact(
            aggregate, aggregate=True
        )
        and aggregate.get("schema")
        == "xvla_capable_linear_b1c0_rtx6000_capability_aggregate_v1",
        "episodes": exact
        and _aggregate_episode_copy_is_exact(aggregate, all_episodes),
        "counts": _type_exact_equal(aggregate.get("successes"), successes)
        and _type_exact_equal(aggregate.get("trials"), 500)
        and _type_exact_equal(aggregate.get("success_rate"), successes / 500)
        and _type_exact_equal(
            aggregate.get("per_task_successes"), per_task_successes
        )
        and _type_exact_equal(
            aggregate.get("per_task"),
            {
                key: value / capability_lane.EPISODES_PER_TASK
                for key, value in per_task_successes.items()
            },
        )
        and _type_exact_equal(
            aggregate.get("early_terminal_failures"),
            aggregate_early_terminal_failures,
        ),
        "transitions": _transition_counts_are_exact(
            aggregate,
            aggregate.get("episodes"),
            expected_settle=10
            * capability_lane.EPISODES_PER_TASK
            * capability_lane.SETTLE_STEPS,
        )
        and _type_exact_equal(
            aggregate.get("validated_transition_counts"),
            {
                key: sum(record[key] for record in shard_transition_counts)
                for key in EXPECTED_CAPABILITY_TRANSITION_COUNT_KEYS
            },
        ),
        "floor": _type_exact_equal(aggregate.get("capability_floor"), 0.80)
        and aggregate.get("capability_floor_passed") is True
        and successes >= CAPABILITY_FLOOR_SUCCESSES,
        "shards": _type_exact_equal(aggregate.get("shards"), shard_records)
        and aggregate.get("shard_result_bundle_sha256")
        == _canonical_sha256(shard_hashes),
        "smoke": aggregate.get("smoke_gate_sha256") == smoke_sha,
        "episode_digest": _row_identity_evidence_is_exact(
            aggregate,
            0,
            10,
            max_steps=280,
            episodes_per_task=50,
        ),
        "protocol": _aggregate_protocol_is_exact(protocol),
        "source": _common_capability_gates(
            aggregate, source_sha, stage_sha, capability_audit
        ),
        "producer_identity": all(_producer_identity_gates(aggregate).values()),
        "config": _type_exact_equal(
            aggregate.get("config_identity"), EXPECTED_CONFIG_IDENTITY
        ),
        "producer_gates": isinstance(aggregate_gates, dict)
        and set(aggregate_gates)
        == {
            *(
                f"shard_{start}_{stop}_{name}"
                for start, stop in SHARDS
                for name in capability_lane.SHARD_VALIDATION_GATE_KEYS
            ),
            *capability_lane.AGGREGATE_OWN_GATE_KEYS,
        }
        and all(value is True for value in aggregate_gates.values())
        and aggregate.get("all_gates_pass") is True,
        "claim_boundary": aggregate.get(
            "fresh_all_rtx6000_hash_bound_reevaluation"
        )
        is True
        and aggregate.get("a30_hardware_reproduction_claimed") is False
        and aggregate.get("historical_hardware_reproduction_claimed") is False
        and aggregate.get("node05_execution_required") is True
        and aggregate.get("fresh_evaluation_sha256_bound_to_checkpoint") is True
        and aggregate.get("historical_evaluation_digest_binding_claimed") is False
        and aggregate.get("historical_hardware_was_mixed_and_is_comparison_only")
        is True
        and aggregate.get("determinism_claim_scope")
        == capability_lane.DETERMINISM_CLAIM_SCOPE
        and aggregate.get("claim") == capability_lane.CAPABILITY_CLAIM
        and isinstance(aggregate.get("inactive_vision_norm_sentinel"), dict)
        and set(aggregate["inactive_vision_norm_sentinel"])
        == EXPECTED_CAPABILITY_AGGREGATE_SENTINEL_KEYS
        and _type_exact_equal(
            aggregate["inactive_vision_norm_sentinel"],
            {
            "site": "vision.norm_out",
            "total_site_count": 74,
            "inactive_sites": ["vision.norm_out"],
            "active_site_count": 73,
            "shard_count": 4,
            "all_shard_call_counts_zero": True,
            "all_shard_active_sets_observed_once": True,
            },
        )
        and all(
            isinstance(record, dict)
            and set(record) == EXPECTED_CAPABILITY_SHARD_RECORD_KEYS
            for record in aggregate.get("shards", [])
        ),
    }
    _require(all(gates.values()), f"RTX 6000 aggregate failed {gates}")
    assert_physical_hashes_unchanged(paths, snapshot)
    return {
        "path": CAPABILITY_AGGREGATE.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": aggregate_sha,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "successes": successes,
        "trials": 500,
        "success_rate": successes / 500,
        "source_manifest_sha256": source_sha,
        "stage_ledger_sha256": stage_sha,
        "shard_sha256": shard_hashes,
        "smoke_sha256": smoke_sha,
        "config_identity": EXPECTED_CONFIG_IDENTITY,
        "input_paths": paths,
        "input_sha256": snapshot,
        "gates": gates,
    }


def _error_below(value: Any, limit: float, *, inclusive: bool = False) -> bool:
    if type(value) is not float:
        return False
    numeric = float(value)
    return (
        math.isfinite(numeric)
        and numeric >= 0.0
        and (numeric <= limit if inclusive else numeric < limit)
    )


def _positive_finite(value: Any) -> bool:
    return (
        type(value) is float
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _timings_and_peak_are_valid(timings: Any, peak_rss_mb: Any) -> bool:
    if (
        not isinstance(timings, dict)
        or set(timings) != EXPECTED_B1C_TIMING_KEYS
        or not all(_positive_finite(value) for value in timings.values())
        or not _positive_finite(peak_rss_mb)
    ):
        return False
    phase_sum = sum(
        float(timings[name])
        for name in ("compile", "source_evaluation", "algorithm1", "algorithms2_and3")
    )
    return float(timings["through_full_rank"]) >= phase_sum


def _canonical_shape_is_reconciled(
    initial: Any, canonical: Any, route: Any
) -> bool:
    expected_keys = set(EXPECTED_B1C_INITIAL_SHAPE)
    if (
        not isinstance(initial, dict)
        or not _type_exact_equal(initial, EXPECTED_B1C_INITIAL_SHAPE)
        or not isinstance(canonical, dict)
        or set(canonical) != expected_keys
        or not all(type(value) is int and value >= 0 for value in canonical.values())
        or not _type_exact_equal(route, EXPECTED_B1C_ROUTE_INVENTORY)
    ):
        return False
    return (
        canonical["unique_nodes"] == initial["unique_nodes"]
        and canonical["edge_occurrences"] == initial["edge_occurrences"]
        and canonical["cp_binary_nodes"]
        == route["streamed_cp_candidate_count"]
        and canonical["reduced_q_binary_nodes"]
        == route["bounded_retained_q_candidate_count"]
        and canonical["cp_binary_nodes"] + canonical["reduced_q_binary_nodes"]
        == initial["cp_binary_nodes"]
        and canonical["dense_clone_nodes"] == 0
        and canonical["unary_nodes"] == initial["unary_nodes"]
        and canonical["heterogeneous_physical_sources"]
        == initial["heterogeneous_physical_sources"]
        and canonical["maximum_local_bond_dimension"]
        == initial["maximum_local_bond_dimension"]
        == 385
        and canonical["maximum_cp_rank"] == initial["maximum_cp_rank"] == 1_153
        and canonical["maximum_reduced_q_elements"]
        == route["bounded_retained_q_maximum_elements"]
        and canonical["fused_token_feature_dimension"]
        == initial["fused_token_feature_dimension"]
    )


def _b1c_full_scientific_evidence_gates(
    full: Mapping[str, Any],
    *,
    bounded_unit_sha256: str,
    odt_sources: Mapping[str, str],
    patched_entrypoints_sha256: str = EXPECTED_B1C_PATCHED_ENTRYPOINTS_SHA256,
) -> dict[str, bool]:
    checkpoint = full.get("checkpoint")
    sentinel = full.get("inactive_vision_norm_sentinel")
    source_manifest = full.get("source_manifest")
    static_audit = full.get("static_audit")
    replay_panel = full.get("replay_panel")
    timings = full.get("timings_seconds")
    structure = full.get("structure")
    initial = full.get("initial_shape")
    canonical = full.get("canonical_shape")
    route = full.get("route_inventory")
    source_replay = full.get("source_replay")
    algorithm1 = full.get("algorithm1")
    algorithms2_and3 = full.get("algorithms2_and3")
    full_gates = full.get("full_rank_gates")
    algorithm_order = full.get("dooms_algorithm_order")
    bounded_oracle = full.get("bounded_clone_and_regression_attestation")
    runtime = full.get("runtime_guard")

    edge_count = EXPECTED_B1C_STRUCTURE["edge_occurrences"]
    node_count = EXPECTED_B1C_STRUCTURE["unique_nodes"]
    expected_pushes = edge_count + 1

    expected_sentinel = {
            "site": "vision.norm_out",
            "armed_through_full_rank": True,
            "call_count": 0,
            "total_site_count": 74,
            "inactive_sites": ["vision.norm_out"],
            "active_site_count": 73,
            "active_call_count": 146,
            "every_active_site_call_count": 2,
    }
    sentinel_exact = (
        set(expected_sentinel) == EXPECTED_B1C_SENTINEL_KEYS
        and _type_exact_equal(sentinel, expected_sentinel)
    )
    expected_source_manifest = {
        "path": B1C_SOURCE_MANIFEST.as_posix(),
        "sha256": EXPECTED_B1C_SOURCE_MANIFEST_SHA256,
        "source_count": len(odt_sources),
        "source_sha256": dict(odt_sources),
    }
    source_manifest_exact = (
        set(expected_source_manifest) == EXPECTED_B1C_SOURCE_MANIFEST_KEYS
        and _type_exact_equal(source_manifest, expected_source_manifest)
    )
    expected_static_audit = {
        "scope": "transitive_local_import_closure",
        "entrypoints": EXPECTED_B1C_STATIC_ENTRYPOINTS,
        "source_count": len(odt_sources),
        "source_sha256": dict(odt_sources),
        "call_site_count": 8_310,
        "direct_qr_required": True,
        "direct_qr_call_sites": 4,
        "prohibited_calls_found": [],
        "prohibited_self_overlap_sites": [],
        "guarded_dormant_spectral_norm_sites": [],
        "duplicate_top_level_definition_sites": [],
    }
    static_audit_exact = (
        set(expected_static_audit) == EXPECTED_B1C_STATIC_AUDIT_KEYS
        and _type_exact_equal(static_audit, expected_static_audit)
    )

    source_replay_exact = False
    if isinstance(source_replay, dict):
        source_replay_exact = (
            set(source_replay)
            == {
                "observable_relative_error",
                "public_action_relative_error",
                "observable_matches_public_action_relative_error",
            }
            and _error_below(source_replay.get("observable_relative_error"), REPLAY_LIMIT)
            and _error_below(source_replay.get("public_action_relative_error"), REPLAY_LIMIT)
            and _error_below(
                source_replay.get("observable_matches_public_action_relative_error"),
                SOURCE_ACTION_LIMIT,
            )
        )

    methods_exact = False
    algorithm1_schema_exact = False
    reconstruction_exact = False
    replay_exact = False
    occurrence_exact = False
    provenance_exact = False
    telemetry_exact = False
    if isinstance(algorithm1, dict):
        algorithm1_schema_exact = set(algorithm1) == EXPECTED_ALGORITHM1_KEYS
        methods_exact = (
            _type_exact_equal(algorithm1.get("step_count"), node_count)
            and _type_exact_equal(
                algorithm1.get("factorization_methods"),
                EXPECTED_B1C_FACTORIZATION_METHODS,
            )
        )
        reconstruction_exact = _error_below(
            algorithm1.get("maximum_local_scaled_reconstruction_relative_error"),
            LOCAL_CERTIFICATE_LIMIT,
        ) and _error_below(
            algorithm1.get("maximum_absorption_scaled_relative_error"),
            LOCAL_CERTIFICATE_LIMIT,
        )
        replay_exact = (
            _error_below(
                algorithm1.get("final_projective_replay_relative_error"),
                REPLAY_LIMIT,
            )
            and _type_exact_equal(
                algorithm1.get("full_projective_replay_evaluations"), 2
            )
            and _type_exact_equal(
                algorithm1.get("per_step_replays_performed"), 0
            )
        )
        occurrence_exact = (
            _type_exact_equal(
                algorithm1.get("push_ledger"),
                [expected_pushes, expected_pushes],
            )
            and _type_exact_equal(
                algorithm1.get("scale_sensitive_occurrence_ledger"),
                [expected_pushes, expected_pushes],
            )
        )

        normal_form = algorithm1.get("canonical_exponent_normal_form")
        provenance = algorithm1.get("direct_q_provenance")
        if isinstance(normal_form, dict) and isinstance(provenance, dict):
            normal_form_exact = (
                set(normal_form)
                == {
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
                and _type_exact_equal(normal_form.get("core_count"), node_count)
                and normal_form.get("all_core_exponents_zero") is True
                and _type_exact_equal(
                    normal_form.get("head_binary_exponent"),
                    EXPECTED_B1C_HEAD_BINARY_EXPONENT,
                )
                and normal_form.get("head_exponent_matches_ledger") is True
                and _type_exact_equal(
                    normal_form.get("cp_direct_q_certificates"),
                    EXPECTED_B1C_ROUTE_INVENTORY[
                        "streamed_cp_candidate_count"
                    ],
                )
                and _type_exact_equal(
                    normal_form.get(
                        "streamed_compact_cp_direct_q_certificates"
                    ),
                    normal_form.get("cp_direct_q_certificates"),
                )
                and _type_exact_equal(
                    normal_form.get("cp_direct_q_columns"),
                    EXPECTED_B1C_CP_DIRECT_Q_COLUMNS,
                )
                and _type_exact_equal(
                    normal_form.get("streamed_compact_cp_direct_q_columns"),
                    normal_form.get("cp_direct_q_columns"),
                )
                and normal_form.get(
                    "all_streamed_compact_cp_q_has_direct_q_provenance"
                )
                is True
            )
            provenance_exact = (
                set(provenance)
                == {
                    "verified_steps",
                    "expected_steps",
                    "streamed_steps",
                    "columns_compared",
                    "expected_columns",
                    "streamed_replay_calls",
                    "expected_streamed_replay_calls",
                    "maximum_compact_q_relative_error",
                    "maximum_reconstruction_relative_error",
                }
                and _type_exact_equal(
                    provenance.get("verified_steps"), node_count
                )
                and _type_exact_equal(
                    provenance.get("expected_steps"), node_count
                )
                and _type_exact_equal(
                    provenance.get("streamed_steps"),
                    EXPECTED_B1C_ROUTE_INVENTORY[
                        "streamed_cp_candidate_count"
                    ],
                )
                and type(provenance.get("columns_compared")) is int
                and _type_exact_equal(
                    provenance.get("expected_columns"),
                    provenance.get("columns_compared"),
                )
                and provenance.get("columns_compared") > 0
                and type(provenance.get("streamed_replay_calls")) is int
                and _type_exact_equal(
                    provenance.get("expected_streamed_replay_calls"),
                    provenance.get("streamed_replay_calls"),
                )
                and provenance.get("streamed_replay_calls") > 0
                and _error_below(
                    provenance.get("maximum_compact_q_relative_error"),
                    LOCAL_CERTIFICATE_LIMIT,
                    inclusive=True,
                )
                and _error_below(
                    provenance.get("maximum_reconstruction_relative_error"),
                    LOCAL_CERTIFICATE_LIMIT,
                    inclusive=True,
                )
                and normal_form_exact
            )

            telemetry = algorithm1.get("telemetry")
            if isinstance(telemetry, dict):
                telemetry_exact = (
                    set(telemetry) == EXPECTED_ALGORITHM1_TELEMETRY_KEYS
                    and all(
                        _type_exact_equal(telemetry.get(key), expected)
                        for key, expected in EXPECTED_ALGORITHM1_STATIC_TELEMETRY.items()
                    )
                    and _type_exact_equal(
                        telemetry.get("direct_q_columns_compared"),
                        provenance.get("columns_compared"),
                    )
                    and _type_exact_equal(
                        telemetry.get(
                            "streamed_direct_q_replay_qr_factorizations"
                        ),
                        provenance.get("streamed_replay_calls"),
                    )
                    and _type_exact_equal(
                        telemetry.get(
                            "maximum_direct_q_compact_relative_error"
                        ),
                        provenance.get("maximum_compact_q_relative_error"),
                    )
                    and _type_exact_equal(
                        telemetry.get(
                            "maximum_direct_q_reconstruction_relative_error"
                        ),
                        provenance.get("maximum_reconstruction_relative_error"),
                    )
                )

    algorithm2_exact = False
    algorithm3_occurrences_exact = False
    algorithm3_replay_exact = False
    algorithm3_diagonal_exact = False
    algorithm3_retention_exact = False
    if isinstance(algorithms2_and3, dict):
        algorithm2_exact = (
            set(algorithms2_and3) == EXPECTED_ALGORITHMS2_AND3_KEYS
            and _type_exact_equal(
                algorithms2_and3.get("algorithm2_child_messages"), edge_count
            )
            and _type_exact_equal(
                algorithm_order, EXPECTED_B1C_ALGORITHM_ORDER
            )
        )
        algorithm3_occurrences_exact = (
            _type_exact_equal(
                algorithms2_and3.get("algorithm3_diagonalized_nodes"),
                node_count,
            )
            and _type_exact_equal(
                algorithms2_and3.get("algorithm3_occurrence_ledger"),
                [expected_pushes, expected_pushes],
            )
        )
        algorithm3_replay_exact = _error_below(
            algorithms2_and3.get("algorithm3_replay_relative_error"), REPLAY_LIMIT
        )
        algorithm3_diagonal_exact = _error_below(
            algorithms2_and3.get("maximum_recontracted_offdiagonal_ratio"),
            ENVIRONMENT_DIAGONALITY_LIMIT,
        )
        algorithm3_retention_exact = (
            algorithms2_and3.get("full_spectra_retained") is False
            and algorithms2_and3.get("post_environment_records_retained") is False
        )

    bounded_oracle_exact = (
        isinstance(bounded_oracle, dict)
        and set(bounded_oracle) == EXPECTED_BOUNDED_ORACLE_KEYS
        and bounded_oracle.get("path") == "inputs/direct_odt_truncation_unit_v2.json"
        and bounded_oracle.get("sha256") == bounded_unit_sha256
        and _type_exact_equal(bounded_oracle.get("test_count"), 52)
        and isinstance(bounded_oracle.get("required_oracles"), list)
        and set(bounded_oracle["required_oracles"]) == EXPECTED_BOUNDED_ORACLES
        and len(bounded_oracle["required_oracles"]) == len(EXPECTED_BOUNDED_ORACLES)
        and isinstance(bounded_oracle.get("gates"), dict)
        and set(bounded_oracle["gates"]) == EXPECTED_BOUNDED_ORACLE_GATES
        and all(value is True for value in bounded_oracle["gates"].values())
        and isinstance(bounded_oracle.get("shared_core_sha256"), dict)
        and set(bounded_oracle["shared_core_sha256"])
        == EXPECTED_BOUNDED_SHARED_CORES
        and all(
            bounded_oracle["shared_core_sha256"].get(relative)
            == odt_sources.get(relative)
            for relative in EXPECTED_BOUNDED_SHARED_CORES
        )
        and full.get("bounded_clone_oracle_hash_joined") is True
    )
    runtime_exact = (
        isinstance(runtime, dict)
        and set(runtime) == EXPECTED_RUNTIME_KEYS
        and runtime.get("installed") is True
        and _type_exact_equal(runtime.get("patched_entrypoint_count"), 87)
        and _type_exact_equal(runtime.get("prohibited_attempt_count"), 0)
        and _type_exact_equal(runtime.get("prohibited_attempts"), [])
        and _type_exact_equal(
            runtime.get("allowed_call_count"), EXPECTED_B1C_RUNTIME_CALL_COUNT
        )
        and runtime.get("allowed_call_count")
        == EXPECTED_ALGORITHM1_STATIC_TELEMETRY["householder_qr_kernel_calls"]
        + EXPECTED_B1C_ROUTE_INVENTORY["streamed_cp_candidate_count"]
        + node_count
        and isinstance(runtime.get("allowed_calls"), list)
        and set(runtime["allowed_calls"]) == EXPECTED_B1C_RUNTIME_CALLS
        and len(runtime["allowed_calls"]) == len(EXPECTED_B1C_RUNTIME_CALLS)
        and isinstance(runtime.get("patched_entrypoints"), list)
        and len(runtime["patched_entrypoints"]) == 87
        and len(set(runtime["patched_entrypoints"])) == 87
        and _canonical_sha256(runtime["patched_entrypoints"])
        == patched_entrypoints_sha256
    )

    return {
        "top_level_schema_exact": isinstance(full, dict)
        and set(full) == EXPECTED_B1C_FULL_TOP_LEVEL_KEYS,
        "checkpoint_schema_and_identity_exact": _b1c_checkpoint_identity_is_exact(
            checkpoint
        ),
        "inactive_sentinel_schema_and_evidence_exact": sentinel_exact,
        "source_manifest_schema_and_authority_exact": source_manifest_exact,
        "static_audit_schema_and_authority_exact": static_audit_exact,
        "numerical_precision_boundary_exact": _type_exact_equal(
            full.get("numerical_precision_boundary"),
            EXPECTED_B1C_PRECISION_BOUNDARY,
        ),
        "replay_panel_exact": _type_exact_equal(
            replay_panel, EXPECTED_B1C_REPLAY_PANEL
        ),
        "timings_and_peak_valid": _timings_and_peak_are_valid(
            timings, full.get("peak_rss_mb")
        ),
        "dooms_reference_exact": full.get("dooms_reference_sha256")
        == EXPECTED_DOOMS_REFERENCE_SHA256,
        "historical_non_authoritative_records_present": isinstance(
            full.get("capability_evidence"), dict
        )
        and isinstance(full.get("previous_exact_checkpoint_control"), dict),
        "structure_exact": _type_exact_equal(structure, EXPECTED_B1C_STRUCTURE),
        "initial_shape_exact": _type_exact_equal(
            initial, EXPECTED_B1C_INITIAL_SHAPE
        ),
        "canonical_shape_reconciled": _canonical_shape_is_reconciled(
            initial, canonical, route
        ),
        "route_inventory_exact": _type_exact_equal(
            route, EXPECTED_B1C_ROUTE_INVENTORY
        ),
        "source_replay_exact": source_replay_exact,
        "algorithm1_schema_exact": algorithm1_schema_exact,
        "algorithm1_methods_direct": methods_exact,
        "algorithm1_reconstruction": reconstruction_exact,
        "algorithm1_replay": replay_exact,
        "algorithm1_every_occurrence": occurrence_exact,
        "algorithm1_provenance": provenance_exact,
        "algorithm1_telemetry": telemetry_exact,
        "algorithm2_explicit_contraction_and_eigendecomposition": algorithm2_exact,
        "algorithm3_every_node_and_occurrence": algorithm3_occurrences_exact,
        "algorithm3_replay": algorithm3_replay_exact,
        "algorithm3_environment_diagonality": algorithm3_diagonal_exact,
        "algorithm3_retention_scope": algorithm3_retention_exact,
        "full_gate_schema_exact": isinstance(full_gates, dict)
        and set(full_gates) == EXPECTED_B1C_FULL_GATE_NAMES
        and all(type(value) is bool and value for value in full_gates.values()),
        "algorithm_order_exact": _type_exact_equal(
            algorithm_order, EXPECTED_B1C_ALGORITHM_ORDER
        ),
        "bounded_clone_oracle_joined": bounded_oracle_exact,
        "runtime_factorization_and_eigendecomposition_ledger": runtime_exact,
        "claim_boundary_exact": full.get("claim_boundary")
        == EXPECTED_B1C_CLAIM_BOUNDARY,
        "deployment_scope_exact": _type_exact_equal(
            full.get("deployment_interface_scope"), EXPECTED_B1C_DEPLOYMENT_SCOPE
        ),
        "phase_and_claim_scope_exact": full.get("all_full_rank_gates_pass") is True
        and full.get("canonical_direct_odt_algorithms_1_to_3_completed") is True
        and full.get("exact_learned_neural_dag_global_decomposability_certified")
        is True
        and full.get("historical_capability_path_association_recorded") is True
        and full.get("historical_capability_sha256_bound_to_current_checkpoint")
        is False
        and full.get("compression_claimed") is False,
    }


def _b1c_checkpoint_identity_is_exact(checkpoint: Any) -> bool:
    expected = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_bytes": EXPECTED_B1C_CHECKPOINT_BYTES,
        "state_key_count": 540,
        "parameter_count": 20_137_352,
        "normalization_site_count": 74,
        "active_normalization_site_count": 73,
        "compiler_active_normalization_site_count": 73,
        "compiler_active_snapshot_site_count": 73,
        "compiler_active_set_equals_loaded_active_set": True,
        "normalization_buffer_inventory": (
            EXPECTED_B1C_NORMALIZATION_BUFFER_INVENTORY
        ),
        "config_identity": EXPECTED_CONFIG_IDENTITY,
        "vocab_size": 26,
    }
    return (
        set(expected) == EXPECTED_B1C_CHECKPOINT_KEYS
        and _type_exact_equal(checkpoint, expected)
    )


def _b1c_current_unit_record_is_exact(
    value: Any, unit_sha256: str, direct_sha256: str
) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == EXPECTED_B1C_CURRENT_UNIT_KEYS
        and value.get("path")
        == "athena/results/capable_linear_b1c0/unit_suite.json"
        and value.get("sha256") == unit_sha256
        and value.get("direct_result_path")
        == "athena/results/capable_linear_b1c0_core_unit.json"
        and value.get("direct_result_sha256") == direct_sha256
        and _type_exact_equal(
            value.get("test_counts"),
            {"direct": 52, "capability": 34, "integrity": 7},
        )
        and _type_exact_equal(
            value.get("input_sha256"),
            {"current_unit": unit_sha256, "current_direct": direct_sha256},
        )
        and isinstance(value.get("gates"), dict)
        and set(value["gates"]) == EXPECTED_B1C_CURRENT_UNIT_GATE_KEYS
        and all(type(item) is bool and item for item in value["gates"].values())
    )


def _read_b1c_full_rank() -> dict[str, Any]:
    _require(
        _sha256(PINNED_B1C_STAGE_LEDGER) == EXPECTED_B1C_STAGE_LEDGER_SHA256
        and _sha256(B1C_STAGE_LEDGER) == EXPECTED_B1C_STAGE_LEDGER_SHA256,
        "pinned and external b1c stage ledgers differ",
    )
    _require(
        _sha256(PINNED_B1C_SOURCE_MANIFEST)
        == EXPECTED_B1C_SOURCE_MANIFEST_SHA256
        and _sha256(B1C_SOURCE_MANIFEST)
        == EXPECTED_B1C_SOURCE_MANIFEST_SHA256,
        "pinned and external b1c source manifests differ",
    )
    _require(
        _sha256(B1C_STAGE_LEDGER) == EXPECTED_B1C_STAGE_LEDGER_SHA256,
        "b1c stage ledger identity differs",
    )
    b1c_ledger = _read_line_manifest(B1C_STAGE_LEDGER, B1C_STAGE_ROOT, "b1c stage ledger")
    _require(
        b1c_ledger.get("athena/capable_linear_direct_odt_sources.sha256")
        == EXPECTED_B1C_SOURCE_MANIFEST_SHA256,
        "b1c stage ledger does not bind expected ODT manifest",
    )
    _require(
        _sha256(B1C_SOURCE_MANIFEST) == EXPECTED_B1C_SOURCE_MANIFEST_SHA256,
        "b1c ODT source manifest identity differs",
    )
    odt_sources = _read_line_manifest(
        B1C_SOURCE_MANIFEST, B1C_STAGE_ROOT, "b1c ODT source manifest"
    )
    paths = {
        "full_rank": B1C_FULL_RANK,
        "current_unit": B1C_UNIT,
        "current_direct": B1C_DIRECT_UNIT,
        "bounded_unit": B1C_BOUNDED_UNIT,
        "b1c_checkpoint": B1C_CHECKPOINT,
        "stage_ledger": B1C_STAGE_LEDGER,
        "source_manifest": B1C_SOURCE_MANIFEST,
    }
    snapshot = assert_physical_hashes_unchanged(paths)
    _require(
        b1c_ledger.get("inputs/direct_odt_truncation_unit_v2.json")
        == snapshot["bounded_unit"],
        "b1c stage ledger does not bind bounded clone oracle result",
    )
    _require(
        b1c_ledger.get("inputs/capable_linear_b1c0_checkpoint.pt")
        == snapshot["b1c_checkpoint"]
        == EXPECTED_CHECKPOINT_SHA256,
        "b1c stage ledger does not bind the exact checkpoint bytes",
    )
    for label, path in paths.items():
        _physical_read_only(path, label)
    full, full_sha = read_authenticated_json(B1C_FULL_RANK, "b1c full-rank certificate")
    unit, unit_sha = read_authenticated_json(B1C_UNIT, "b1c current unit result")
    direct, direct_sha = read_authenticated_json(
        B1C_DIRECT_UNIT, "b1c direct unit result"
    )
    _require(
        full_sha == snapshot["full_rank"]
        and unit_sha == snapshot["current_unit"]
        and direct_sha == snapshot["current_direct"],
        "b1c fields and digests came from different bytes",
    )
    full_gates = full.get("full_rank_gates")
    checkpoint = full.get("checkpoint", {})
    source_manifest = full.get("source_manifest", {})
    static_audit = full.get("static_audit", {})
    sentinel = full.get("inactive_vision_norm_sentinel", {})
    current_unit = full.get("current_combined_unit_attestation", {})
    scientific = _b1c_full_scientific_evidence_gates(
        full,
        bounded_unit_sha256=snapshot["bounded_unit"],
        odt_sources=odt_sources,
    )
    gates = {
        "schema": full.get("schema")
        == "xvla_capable_linear_b1c0_full_rank_direct_odt_v1",
        "checkpoint": _b1c_checkpoint_identity_is_exact(checkpoint),
        "checkpoint_physical": checkpoint.get("checkpoint_sha256")
        == snapshot["b1c_checkpoint"]
        and _type_exact_equal(
            checkpoint.get("checkpoint_bytes"), B1C_CHECKPOINT.stat().st_size
        ),
        "normalization": _type_exact_equal(
            checkpoint.get("normalization_site_count"), 74
        )
        and _type_exact_equal(
            checkpoint.get("active_normalization_site_count"), 73
        )
        and _type_exact_equal(
            checkpoint.get("compiler_active_normalization_site_count"), 73
        )
        and checkpoint.get("compiler_active_set_equals_loaded_active_set") is True,
        "inactive_sentinel": _type_exact_equal(
            sentinel,
            {
                "site": "vision.norm_out",
                "armed_through_full_rank": True,
                "call_count": 0,
                "total_site_count": 74,
                "inactive_sites": ["vision.norm_out"],
                "active_site_count": 73,
                "active_call_count": 146,
                "every_active_site_call_count": 2,
            },
        ),
        "source_manifest": scientific[
            "source_manifest_schema_and_authority_exact"
        ],
        "static_audit": scientific["static_audit_schema_and_authority_exact"],
        "scientific_evidence": all(scientific.values()),
        "full_gates": isinstance(full_gates, dict)
        and set(full_gates) == EXPECTED_B1C_FULL_GATE_NAMES
        and all(type(value) is bool and value for value in full_gates.values()),
        "algorithm_order": _type_exact_equal(
            full.get("dooms_algorithm_order"), EXPECTED_B1C_ALGORITHM_ORDER
        ),
        "exact_claim": full.get("all_full_rank_gates_pass") is True
        and full.get("canonical_direct_odt_algorithms_1_to_3_completed") is True
        and full.get("exact_learned_neural_dag_global_decomposability_certified")
        is True
        and full.get("compression_claimed") is False,
        "runtime": _type_exact_equal(
            full.get("runtime_guard", {}).get("prohibited_attempt_count"), 0
        ),
        "current_unit": _b1c_current_unit_record_is_exact(
            current_unit, unit_sha, direct_sha
        )
        and unit.get("all_gates_pass") is True
        and direct.get("all_gates_pass") is True,
        "precision_boundary": scientific["numerical_precision_boundary_exact"],
        "dooms_reference": full.get("dooms_reference_sha256")
        == b1c_ledger.get("reference/dooms_xnets_2504.02667.pdf"),
    }
    _require(all(gates.values()), f"b1c full-rank certificate failed {gates}")
    assert_physical_hashes_unchanged(paths, snapshot)
    _require(
        _sha256(B1C_STAGE_LEDGER) == EXPECTED_B1C_STAGE_LEDGER_SHA256,
        "b1c stage changed during full-rank validation",
    )
    return {
        "path": B1C_FULL_RANK.as_posix(),
        "sha256": full_sha,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "source_manifest_sha256": EXPECTED_B1C_SOURCE_MANIFEST_SHA256,
        "stage_ledger_sha256": EXPECTED_B1C_STAGE_LEDGER_SHA256,
        "config_identity": EXPECTED_CONFIG_IDENTITY,
        "input_paths": paths,
        "input_sha256": snapshot,
        "scientific_gates": scientific,
        "b1c_historical_capability_evidence_used_as_capability_authority": False,
        "b1c_r7_control_used_as_exact_odt_authority": False,
        "gates": gates,
    }


def _rtx_current_unit_record_is_exact(
    value: Any,
    *,
    capability_manifest_sha256: str,
    cross_manifest_sha256: str,
    stage_ledger_sha256: str,
    static_audit: Mapping[str, Any],
) -> bool:
    capability_count = EXPECTED_RTX_CAPABILITY_TEST_COUNT
    cross_count = EXPECTED_RTX_CROSS_STAGE_TEST_COUNT
    expected_total = capability_count + cross_count
    if not isinstance(value, dict):
        return False
    junit = value.get("junit")
    gates = value.get("gates")
    names = junit.get("test_names") if isinstance(junit, dict) else None
    elapsed = value.get("elapsed_seconds")
    return (
        set(value) == EXPECTED_RTX_UNIT_KEYS
        and value.get("schema") == "xvla_capable_linear_rtx6000_unit_suite_v1"
        and _type_exact_equal(
            value.get("test_counts"),
            {
                "capability": capability_count,
                "cross_stage": cross_count,
                "total": expected_total,
            },
        )
        and isinstance(junit, dict)
        and set(junit) == EXPECTED_RTX_UNIT_JUNIT_KEYS
        and _type_exact_equal(junit.get("tests"), expected_total)
        and _type_exact_equal(junit.get("failures"), 0)
        and _type_exact_equal(junit.get("errors"), 0)
        and _type_exact_equal(junit.get("skipped"), 0)
        and isinstance(names, list)
        and all(isinstance(name, str) for name in names)
        and names == sorted(names)
        and len(names) == expected_total
        and len(set(names)) == expected_total
        and _canonical_sha256(names) == EXPECTED_RTX_UNIT_TEST_NAME_SHA256
        and type(elapsed) is float
        and math.isfinite(float(elapsed))
        and float(elapsed) > 0.0
        and isinstance(gates, dict)
        and set(gates) == EXPECTED_RTX_UNIT_GATE_KEYS
        and all(type(item) is bool and item for item in gates.values())
        and value.get("all_gates_pass") is True
        and value.get("capability_source_manifest_sha256")
        == capability_manifest_sha256
        and value.get("cross_stage_source_manifest_sha256")
        == cross_manifest_sha256
        and value.get("stage_ledger_sha256") == stage_ledger_sha256
        and _type_exact_equal(value.get("static_audit"), dict(static_audit))
    )


def _read_current_unit() -> dict[str, Any]:
    value, digest = read_authenticated_json(UNIT_RESULT, "RTX 6000 current unit result")
    _physical_read_only(UNIT_RESULT, "RTX 6000 current unit result")
    _require(
        _rtx_current_unit_record_is_exact(
            value,
            capability_manifest_sha256=_sha256(CAPABILITY_SOURCE_MANIFEST),
            cross_manifest_sha256=_sha256(SOURCE_MANIFEST),
            stage_ledger_sha256=_sha256(STAGE_LEDGER),
            static_audit=_STATIC_AUDIT,
        ),
        "RTX 6000 current unit attestation differs",
    )
    assert_physical_hashes_unchanged({"unit": UNIT_RESULT}, {"unit": digest})
    return {"path": UNIT_RESULT, "sha256": digest, "gates": value["gates"]}


def _publish(path: Path, payload: Mapping[str, Any]) -> str:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing existing output {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    linked = False
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        path.chmod(0o444)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _physical_read_only(path, "cross-stage output")
        _require(_sha256(path) == digest, "cross-stage output differs")
        return digest
    except BaseException:
        if linked and os.path.lexists(path):
            path.unlink()
        raise
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def _fixed_sources() -> None:
    if _PREIMPORT is None:
        return
    _require(
        _read_line_manifest(SOURCE_MANIFEST, PROJECT_ROOT, "source manifest")
        == _PREIMPORT["source"],
        "cross-stage source bytes changed",
    )
    _require(
        _read_line_manifest(STAGE_LEDGER, PROJECT_ROOT, "stage ledger")
        == _PREIMPORT["ledger"],
        "cross-stage stage bytes changed",
    )
    current = audit_direct_only_launch(
        PROJECT_ROOT, _ENTRYPOINTS, require_direct_qr=False
    )
    _require(current == _STATIC_AUDIT, "cross-stage static closure changed")


def _cross_stage_gates(
    capability: Mapping[str, Any], full_rank: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        "checkpoint_sha256_join": capability.get("checkpoint_sha256")
        == full_rank.get("checkpoint_sha256")
        == EXPECTED_CHECKPOINT_SHA256,
        "config_identity_join": _type_exact_equal(
            capability.get("config_identity"), EXPECTED_CONFIG_IDENTITY
        )
        and _type_exact_equal(
            full_rank.get("config_identity"), EXPECTED_CONFIG_IDENTITY
        ),
        "capability_all_500": _type_exact_equal(capability.get("trials"), 500),
        "capability_floor": type(capability.get("successes")) is int
        and capability["successes"] >= CAPABILITY_FLOOR_SUCCESSES,
        "full_rank_exact": isinstance(full_rank.get("gates"), dict)
        and bool(full_rank["gates"])
        and all(value is True for value in full_rank["gates"].values()),
        "capability_exact": isinstance(capability.get("gates"), dict)
        and bool(capability["gates"])
        and all(value is True for value in capability["gates"].values()),
        "separate_source_authorities": capability.get("source_manifest_sha256")
        != full_rank.get("source_manifest_sha256"),
        "separate_stage_authorities": capability.get("stage_ledger_sha256")
        != full_rank.get("stage_ledger_sha256"),
        "historical_nested_records_excluded": full_rank.get(
            "b1c_historical_capability_evidence_used_as_capability_authority"
        )
        is False
        and full_rank.get("b1c_r7_control_used_as_exact_odt_authority") is False,
    }


def main() -> None:
    if sys.argv[1:]:
        raise SystemExit("cross-stage attestation accepts no arguments")
    if os.path.lexists(CROSS_STAGE_OUTPUT):
        raise FileExistsError(f"refusing existing output {CROSS_STAGE_OUTPUT}")
    _fixed_sources()
    unit = _read_current_unit()
    capability = _read_capability_bundle()
    full_rank = _read_b1c_full_rank()
    gates = _cross_stage_gates(capability, full_rank)
    _require(all(gates.values()), f"cross-stage join failed {gates}")
    payload = {
        "schema": "xvla_capable_linear_b1c0_rtx6000_cross_stage_composite_v1",
        "claim": (
            "The b1c0 checkpoint bytes are SHA-256-bound separately to an exact canonical "
            "Dooms Algorithms 1-3 certificate for the learned neural DAG and to a fresh "
            "500-episode closed-loop LIBERO-Object replay on Quadro RTX 6000."
        ),
        "claim_boundaries": {
            "learned_neural_dag_exact_odt": True,
            "fresh_uncompressed_closed_loop_capability": True,
            "simulator_inside_weight_only_odt_closure": False,
            "external_simulator_versions_and_task_assets_pinned": True,
            "external_simulator_python_source_bytes_hash_pinned": False,
            "compressed_policy_capability_measured": False,
            "a30_hardware_reproduction_claimed": False,
            "historical_mixed_hardware_reproduction_claimed": False,
            "b1c_historical_capability_evidence_used_as_capability_authority": False,
            "b1c_r7_control_used_as_exact_odt_authority": False,
            "fresh_rtx6000_result_is_sole_capability_authority": True,
            "current_b1c_full_rank_result_is_sole_exact_odt_authority": True,
            "float32_cuda_and_float64_cpu_arithmetic_bitwise_equal_claimed": False,
            "sha256_binding_scope": "byte identity and cross-artifact join only",
        },
        "identities": {
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "rtx6000_capability_aggregate_sha256": capability["sha256"],
            "rtx6000_capability_source_manifest_sha256": capability[
                "source_manifest_sha256"
            ],
            "rtx6000_stage_ledger_sha256": capability["stage_ledger_sha256"],
            "rtx6000_unit_suite_sha256": unit["sha256"],
            "b1c_full_rank_certificate_sha256": full_rank["sha256"],
            "b1c_odt_source_manifest_sha256": full_rank[
                "source_manifest_sha256"
            ],
            "b1c_stage_ledger_sha256": full_rank["stage_ledger_sha256"],
        },
        "capability": {
            key: capability[key]
            for key in ("successes", "trials", "success_rate", "shard_sha256", "smoke_sha256")
        },
        "b1c_full_rank_scientific_evidence": full_rank["scientific_gates"],
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "exact_learned_neural_dag_global_decomposability_attested": gates[
            "full_rank_exact"
        ],
        "fresh_uncompressed_rtx6000_capability_attested": gates[
            "capability_exact"
        ]
        and gates["capability_all_500"]
        and gates["capability_floor"],
        "same_checkpoint_exact_odt_and_fresh_capability_attested": all(
            gates.values()
        ),
        "compressed_policy_capability_attested": False,
        "static_audit": _STATIC_AUDIT,
        "source_manifest_sha256": _PREIMPORT["source_manifest_sha256"],
        "stage_ledger_sha256": _PREIMPORT["stage_ledger_sha256"],
    }
    capability_snapshot = capability["input_sha256"]
    full_snapshot = full_rank["input_sha256"]
    _fixed_sources()
    assert_physical_hashes_unchanged(
        {"unit": unit["path"]}, {"unit": unit["sha256"]}
    )
    assert_physical_hashes_unchanged(capability["input_paths"], capability_snapshot)
    assert_physical_hashes_unchanged(full_rank["input_paths"], full_snapshot)
    digest = _publish(CROSS_STAGE_OUTPUT, payload)
    try:
        _fixed_sources()
        assert_physical_hashes_unchanged(
            {"unit": unit["path"]}, {"unit": unit["sha256"]}
        )
        assert_physical_hashes_unchanged(
            capability["input_paths"], capability_snapshot
        )
        assert_physical_hashes_unchanged(full_rank["input_paths"], full_snapshot)
        _require(_sha256(CROSS_STAGE_OUTPUT) == digest, "published output changed")
    except BaseException:
        quarantine = CROSS_STAGE_OUTPUT.with_name(
            f"{CROSS_STAGE_OUTPUT.name}.invalid.{os.getpid()}"
        )
        os.replace(CROSS_STAGE_OUTPUT, quarantine)
        raise
    print(
        json.dumps(
            {"output": CROSS_STAGE_OUTPUT.as_posix(), "sha256": digest},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
