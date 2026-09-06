#!/usr/bin/env python3
"""Hash-join the completed r7 full shared-DAG run and 16-test clone oracle.

This verifier uses only the Python standard library. It does not import the
model, Torch, NumPy, or either ODT implementation. Every accepted execution
artifact, source closure, test name, gate name, and controlled numerical call
is pinned to the independently harvested r7 evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


R7_FROZEN_ROOT = (
    "/work/joy/x-vla-global-odt-20260904-r7-rankdef-retained-q-c715f739"
)
R7_MANIFEST_PATH = (
    f"{R7_FROZEN_ROOT}/athena/implicit_sparse_projective_odt_vla_sources.sha256"
)
R7_ORACLE_JUNIT_PATH = (
    f"{R7_FROZEN_ROOT}/athena/results/"
    "direct_odt_clone_oracle_r7_rankdef_retained_q.junit.xml"
)

R7_MANIFEST_SHA256 = (
    "c715f739eb5d6d4a7d44bd12151ab9bab518a620b1ce55928919bd3385aa3884"
)
R7_FULL_RESULT_SHA256 = (
    "f90a2c8c4842206aface13801770ea07fd6a4b3f12c023b0faeb984177d8b374"
)
R7_ORACLE_JSON_SHA256 = (
    "98e49e57681ba5b79ff321e316a325ea7cf864d8752e422f99c6fc7b7f2ebe10"
)
R7_ORACLE_JUNIT_SHA256 = (
    "2ed69357e670ea2a2f506e29e4c22b310b70f749bd78503393b910af99a06277"
)
R7_FULL_STDOUT_SHA256 = (
    "7f8cb96c5aec84aaca041ca8f115e942d61e61b57be0445bc7f1df11f738da63"
)
R7_FULL_STDERR_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
R7_FULL_SACCT_SHA256 = (
    "3de30ed8be4157fa392d30a11ff3920e1bb24ee48ccb27d5ff3b24cc52ecbcab"
)
R7_ORACLE_STDOUT_SHA256 = (
    "e3426ce4b7b32f37f180c383d359d6821bde2f683470eb46ffb3e26a6e72f5e3"
)
R7_ORACLE_STDERR_SHA256 = R7_FULL_STDERR_SHA256
CHECKPOINT_SHA256 = (
    "96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9"
)

PRODUCTION_CORE_PATH = "xvla/train/implicit_sparse_projective_odt.py"
PRODUCTION_CORE_SHA256 = (
    "4b9551e00a4cedf6967e9d415c5b3b6a5e45a12eadfcdc16eeae494f7c0c865d"
)
INDEPENDENT_REFERENCE_PATH = "xvla/train/direct_odt_clone_reference.py"
INDEPENDENT_REFERENCE_SHA256 = (
    "e4c26ed4ea6ad274774d139543d5759675f0bfabb541ad68e10e80512742ecb9"
)
ORACLE_RUNNER_PATH = "scripts/run_direct_odt_clone_oracle.py"
ORACLE_RUNNER_SHA256 = (
    "c1ca813f863fbcc5edf507d48ec047e102f1f94c098230e591c69d3b8fcc4a8d"
)
ORACLE_TEST_PATH = "tests/test_direct_odt_clone_reference.py"
ORACLE_TEST_SHA256 = (
    "f8d111716c1d84dfaf7e4b17ae75281f065084db316ff77d0672eff5dd9518a1"
)
FULL_RUNNER_PATH = "scripts/run_implicit_sparse_projective_odt_vla.py"
FULL_RUNNER_SHA256 = (
    "1dac551f05a3affff315417e3f785c53fc4cc336a36a87ba10a0757cd7ccee18"
)
ATTESTER_TEST_PATH = "tests/test_attest_direct_odt_full_composite.py"
ATTESTER_TEST_SHA256 = (
    "c59dd9a25cb149cae3a49426984c27c8f1c65f653d7b0c423292ebdf7095d48a"
)

EXPECTED_SOURCE_COUNT = 34
EXPECTED_STATIC_CALL_SITE_COUNT = 6_333
EXPECTED_TRANSITIVE_DIRECT_QR_SITES = 4
EXPECTED_MODULE_DIRECT_QR_SITES = 5
EXPECTED_NODE_COUNT = 3_964_463
EXPECTED_EDGE_COUNT = 7_216_854
EXPECTED_POLICY_SOURCE_COUNT = 97
EXPECTED_PADE_SITES = 73
EXPECTED_ROOT_WIDTH = 57
EXPECTED_STREAMED_STEPS = 3_785
EXPECTED_DIRECT_Q_COLUMNS = 2_616_809_755
EXPECTED_STREAMED_REPLAY_QR_CALLS = 91_331
EXPECTED_RECTANGULAR_STEPS = 982
EXPECTED_FULL_ALLOWED_CALL_COUNT = 8_111_588
EXPECTED_ORACLE_ALLOWED_CALL_COUNT = 8_508
EXPECTED_JOB_ID = "835191"
EXPECTED_DIRECT_CALL_AUDIT_OBSERVED_CALLS_SHA256 = (
    "7eae52c24b8ba4f01ae6ef9fc94d6e9130e36d7a4a8cb90f2d9236d6039d2991"
)

GLOBAL_REPLAY_TOLERANCE = 3e-8
LOCAL_CERTIFICATE_TOLERANCE = 3e-10

R7_TEST_NAMES = frozenset(
    {
        "test_bounded_rank_deficient_385x386_retains_direct_q_and_matches_clone_every_step",
        "test_forced_streamed_shared_cp_matches_independent_clone_after_every_step",
        "test_heterogeneous_final_norm_pooling_and_product_head_matches_every_step",
        "test_no_memo_expansion_fails_closed_at_occurrence_limit",
        "test_over_bound_large_deficient_unfolding_fails_closed",
        "test_rank_deficient_shared_child_matches_and_missing_occurrences_fail",
        "test_rank_zero_direct_rq_reconstruction_and_homogeneous_replay",
        "test_reference_source_is_independent_and_stage_restricted",
        "test_seeded_adversarial_shared_dag_family[0]",
        "test_seeded_adversarial_shared_dag_family[1]",
        "test_seeded_adversarial_shared_dag_family[2]",
        "test_seeded_adversarial_shared_dag_family[3]",
        "test_shared_zero_route_environment_uses_only_nonzero_scale_center",
        "test_streamed_production_algorithms2_and3_match_separated_clone_reference",
        "test_two_nonzero_shared_route_exponent_gap_fails_closed",
        "test_unchanged_six_pade_attention_ffn_residual_block_matches_every_step",
    }
)

EXPECTED_FULL_GATE_NAMES = frozenset(
    {
        "algorithm1_canonical_exponent_normal_form",
        "algorithm1_direct_q_telemetry_ledger",
        "algorithm1_every_step_has_direct_q_provenance",
        "algorithm1_exact_route_prediction_is_internally_consistent",
        "algorithm1_final_projective_replay",
        "algorithm1_local_scale_certificates",
        "algorithm1_pushes_every_occurrence",
        "algorithm1_replay_schedule_is_lane_appropriate",
        "algorithm1_streamed_direct_q_provenance_ledger",
        "algorithm2_emits_every_edge_message",
        "algorithm3_memory_schedule_is_lane_appropriate",
        "algorithm3_projective_replay_and_diagonal",
        "algorithm3_pushes_every_occurrence",
        "all_active_pade_sites_present",
        "all_fixed_token_subgraphs_constant_folded",
        "all_nodes_use_direct_reduced_RQ",
        "direct_ODT_module_static_audit",
        "exact_source_tensor_observable_replay",
        "exact_unchanged_chivla_forward_replay",
        "heterogeneous_ingress_exercises_rectangular_direct_RQ",
        "immutable_source_manifest_matches_static_closure",
        "no_joint_width_input_padding",
        "one_exact_policy_observable_root",
        "one_leaf_per_heterogeneous_source",
        "pinned_capable_checkpoint",
        "production_graph_contains_no_dense_clone_core",
        "singleton_embodiment_is_fixed",
        "source_model_state_unchanged",
        "transitive_direct_only_runtime_audit",
        "transitive_direct_only_static_audit",
    }
)

EXPECTED_STATIC_ENTRYPOINTS = frozenset(
    {
        "scripts/run_direct_odt_clone_oracle.py",
        "scripts/run_implicit_sparse_projective_odt.py",
        "scripts/run_implicit_sparse_projective_odt_all_tokens.py",
        "scripts/run_implicit_sparse_projective_odt_vla.py",
        "tests/test_direct_odt_clone_reference.py",
        "tests/test_implicit_sparse_projective_odt.py",
        "tests/test_implicit_sparse_projective_odt_all_tokens.py",
        "tests/test_implicit_sparse_projective_odt_heterogeneous.py",
        "tests/test_implicit_sparse_projective_odt_vla.py",
    }
)

EXPECTED_GUARDED_DORMANT_SPECTRAL_NORM_SITES = (
    "xvla/nn/bilinear.py:76:spectral_clip:torch.linalg.matrix_norm",
    "xvla/nn/bilinear.py:77:spectral_clip:torch.linalg.matrix_norm",
    "xvla/nn/bilinear.py:78:spectral_clip:torch.linalg.matrix_norm",
)

EXPECTED_RUNTIME_PATCHED_ENTRYPOINTS = frozenset(
    {
        "numpy.linalg.cholesky",
        "numpy.linalg.cond",
        "numpy.linalg.eig",
        "numpy.linalg.eigh",
        "numpy.linalg.eigvals",
        "numpy.linalg.eigvalsh",
        "numpy.linalg.inv",
        "numpy.linalg.linalg.cholesky",
        "numpy.linalg.linalg.cond",
        "numpy.linalg.linalg.eig",
        "numpy.linalg.linalg.eigh",
        "numpy.linalg.linalg.eigvals",
        "numpy.linalg.linalg.eigvalsh",
        "numpy.linalg.linalg.inv",
        "numpy.linalg.linalg.lstsq",
        "numpy.linalg.linalg.matrix_power",
        "numpy.linalg.linalg.matrix_rank",
        "numpy.linalg.linalg.norm",
        "numpy.linalg.linalg.pinv",
        "numpy.linalg.linalg.solve",
        "numpy.linalg.linalg.svd",
        "numpy.linalg.linalg.tensorinv",
        "numpy.linalg.linalg.tensorsolve",
        "numpy.linalg.lstsq",
        "numpy.linalg.matrix_power",
        "numpy.linalg.matrix_rank",
        "numpy.linalg.norm",
        "numpy.linalg.pinv",
        "numpy.linalg.solve",
        "numpy.linalg.svd",
        "numpy.linalg.tensorinv",
        "numpy.linalg.tensorsolve",
        "torch.Tensor.cholesky",
        "torch.Tensor.cholesky_inverse",
        "torch.Tensor.cov",
        "torch.Tensor.inverse",
        "torch.Tensor.lstsq",
        "torch.Tensor.lu",
        "torch.Tensor.lu_solve",
        "torch.Tensor.matrix_power",
        "torch.Tensor.norm",
        "torch.Tensor.pinverse",
        "torch.Tensor.solve",
        "torch.Tensor.svd",
        "torch.cholesky",
        "torch.cholesky_inverse",
        "torch.cond",
        "torch.cov",
        "torch.functional.norm",
        "torch.inverse",
        "torch.linalg.cholesky",
        "torch.linalg.cholesky_ex",
        "torch.linalg.cond",
        "torch.linalg.eig",
        "torch.linalg.eigh",
        "torch.linalg.eigvals",
        "torch.linalg.eigvalsh",
        "torch.linalg.inv",
        "torch.linalg.lstsq",
        "torch.linalg.lu",
        "torch.linalg.lu_factor",
        "torch.linalg.lu_factor_ex",
        "torch.linalg.lu_solve",
        "torch.linalg.matrix_norm",
        "torch.linalg.matrix_power",
        "torch.linalg.matrix_rank",
        "torch.linalg.norm",
        "torch.linalg.pinv",
        "torch.linalg.qr",
        "torch.linalg.solve",
        "torch.linalg.solve_triangular",
        "torch.linalg.svd",
        "torch.linalg.svdvals",
        "torch.linalg.tensorinv",
        "torch.linalg.tensorsolve",
        "torch.lstsq",
        "torch.lu",
        "torch.lu_solve",
        "torch.matrix_power",
        "torch.matrix_rank",
        "torch.norm",
        "torch.pca_lowrank",
        "torch.pinverse",
        "torch.polar",
        "torch.solve",
        "torch.svd",
        "torch.svd_lowrank",
    }
)
EXPECTED_RUNTIME_PATCHED_ENTRYPOINT_COUNT = 87

FULL_REQUIRED_CALLS = frozenset(
    {
        "xvla.train.implicit_sparse_projective_odt._direct_tsqr_panel:torch.linalg.qr",
        "xvla.train.implicit_sparse_projective_odt._positive_diagonal_direct_rq_rows:torch.linalg.qr",
        "xvla.train.implicit_sparse_projective_odt._solve_compact_q:torch.linalg.solve_triangular",
        "xvla.train.implicit_sparse_projective_odt.diagonalize_implicit_dag_full_rank:torch.linalg.eigh",
    }
)
ORACLE_REQUIRED_CALLS = FULL_REQUIRED_CALLS | {
    "xvla.train.direct_odt_clone_reference._independent_dense_clone_rq:torch.linalg.qr",
    "xvla.train.direct_odt_clone_reference.diagonalize_shared_and_explicit_clone_independently:torch.linalg.eigh",
}
ALLOWED_RUNTIME_CALLS = FULL_REQUIRED_CALLS | ORACLE_REQUIRED_CALLS

EXPECTED_ROUTE_INVENTORY = {
    "bounded_explicit_q_element_limit": 4_000_000,
    "bounded_explicit_unfolding_element_limit": 4_000_000,
    "bounded_retained_q_candidate_count": 3_248_703,
    "bounded_retained_q_maximum_elements": 3_739_329,
    "bounded_retained_q_maximum_unfolding_elements": 3_739_329,
    "bounded_retained_q_positive_storage_delta_elements": 26_984_351_168,
    "bounded_retained_q_replaced_cp_total_elements": 9_445_994_962,
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
    "bounded_retained_q_signed_storage_delta_elements": 26_840_064_814,
    "bounded_retained_q_total_elements": 36_286_059_776,
    "nodes_with_propagated_input_shape_change": 3_279,
    "predicted_root_output_dimension": EXPECTED_ROOT_WIDTH,
    "schema": "exact_postorder_structural_direct_rq_routes_v1",
    "simulated_unique_node_count": EXPECTED_NODE_COUNT,
    "streamed_cp_candidate_count": EXPECTED_STREAMED_STEPS,
    "streamed_tall_unrepresentable_count": 0,
    "streamed_tall_unrepresentable_shape_counts": {},
}

FULL_CLAIM_BOUNDARY = (
    "Exact shared syntactic projective DAG from fixed non-overlapping patch "
    "coordinates, categorical instruction coordinates, robot state, and the "
    "checkpoint's fixed singleton embodiment through the unchanged single-branch "
    "ChiViT, visual projection, causal joint backbone, final Pade RationalNorm, "
    "and deterministic linear action head. The sole observable is the ordered "
    "action_horizon by action_dimension quotient. This is a clone-unfolded "
    "coefficient object, not compression and not an identified-variable or "
    "categorical-simplex intrinsic metric."
)
ORACLE_CLAIM_BOUNDARY = (
    "Bounded no-memo clone-unfolded Algorithms 1-3 validation. This is not a "
    "whole-policy clone expansion and must be hash-joined with a full shared-DAG "
    "result before any formal whole-policy claim."
)
COMPOSITE_CLAIM = (
    "For checkpoint 96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9 "
    "with the unchanged linear action head and singleton embodiment, the "
    "3,964,463-node shared policy DAG completed direct-RQ Algorithms 1-3 with "
    "all local, scale, occurrence, explicit-environment, gauge, replay, route, "
    "rank-deficiency, and direct-only gates. On the identical r7 source closure, "
    "the bounded independent no-memo clone oracle passed all 16 pinned tests, "
    "including direct-Q retention for the production 385x386 deficient shape, "
    "rank zero, over-bound fail-closed behavior, and streamed CP panelizations. "
    "This is a hash-joined implementation-and-execution attestation. It is not "
    "a literal clone expansion of the 4M-node policy, not an arbitrary-size or "
    "arbitrary-rank theorem, and not evidence that a trained ProductRoutingHead "
    "checkpoint was decomposed."
)
EXPECTED_CERTIFICATION_SCOPE = {
    "schema": "xvla-canonical-odt-certified-scope-v1",
    "certification_kind": "implementation_and_execution",
    "frozen_r7_root": R7_FROZEN_ROOT,
    "manifest_sha256": R7_MANIFEST_SHA256,
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "implementation": "exact_r7_shared_syntactic_projective_dag",
    "shared_dag_node_count": EXPECTED_NODE_COUNT,
    "full_shared_dag_algorithms_1_to_3_completed": True,
    "clone_oracle_scope": "bounded_no_memo_algorithms_1_to_3_test_fixtures",
    "bounded_no_memo_clone_oracle_replay": True,
    "bounded_clone_oracle_test_count": 16,
    "action_head": "linear",
    "singleton_embodiment_fixed": True,
    "literal_full_policy_clone_expansion_performed": False,
    "product_routing_head_decomposed": False,
    "arbitrary_size_or_rank_theorem_claimed": False,
    "compression_claimed": False,
}


class AttestationError(RuntimeError):
    """Raised before publication when any attestation condition is false."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AttestationError(message)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    _require(_is_sha256(value), f"{label} is not a lowercase SHA-256")
    return str(value)


def _stable_file_bytes(path: Path, label: str) -> tuple[bytes, str]:
    _require(path.exists(), f"{label} is missing: {path}")
    _require(path.is_file(), f"{label} is not a regular file: {path}")
    _require(not path.is_symlink(), f"{label} must not be a symlink: {path}")
    before = path.stat(follow_symlinks=False)
    with path.open("rb") as stream:
        payload = stream.read()
    after = path.stat(follow_symlinks=False)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    _require(identity_before == identity_after, f"{label} changed while it was read")
    return payload, hashlib.sha256(payload).hexdigest()


def _stable_file_sha256(path: Path, label: str) -> str:
    _require(path.exists(), f"{label} is missing: {path}")
    _require(path.is_file(), f"{label} is not a regular file: {path}")
    _require(not path.is_symlink(), f"{label} must not be a symlink: {path}")
    before = path.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat(follow_symlinks=False)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    _require(identity_before == identity_after, f"{label} changed while it was hashed")
    return digest.hexdigest()


def _artifact_bytes(path: Path, expected_sha256: str, label: str) -> tuple[bytes, str]:
    _require_sha256(expected_sha256, f"{label} expected SHA-256")
    payload, actual = _stable_file_bytes(path, label)
    _require(actual == expected_sha256, f"{label} SHA-256 differs from its r7 pin")
    return payload, actual


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AttestationError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise AttestationError(f"JSON contains non-finite constant {value}")


def _validate_finite_json(value: Any, label: str = "JSON") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"{label} contains a non-finite number")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{label}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json(item, f"{label}.{key}")


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AttestationError(f"{label} is not strict UTF-8 JSON") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    _validate_finite_json(value, label)
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value  # type: ignore[return-value]


def _sequence(value: object, label: str) -> list[Any]:
    _require(isinstance(value, list), f"{label} must be an array")
    return value  # type: ignore[return-value]


def _integer(value: object, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    return int(value)


def _number(value: object, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be a number",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _bounded_nonnegative(value: object, upper: float, label: str) -> float:
    result = _number(value, label)
    _require(result >= 0, f"{label} must be nonnegative")
    _require(result <= upper, f"{label} exceeds its tolerance")
    return result


def _exact_integer(value: object, expected: int, label: str) -> int:
    result = _integer(value, label)
    _require(result == expected, f"{label} differs")
    return result


def _exact_integer_list(value: object, expected: Sequence[int], label: str) -> None:
    sequence = _sequence(value, label)
    actual = [_integer(item, f"{label}[{index}]") for index, item in enumerate(sequence)]
    _require(actual == list(expected), f"{label} differs")


def _require_null(record: Mapping[str, Any], key: str, label: str) -> None:
    _require(key in record, f"{label}.{key} is missing")
    _require(record[key] is None, f"{label}.{key} must be null")


def _same_json_value_and_type(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _same_json_value_and_type(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_json_value_and_type(left, right)
            for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


def _require_exact_fields(
    record: Mapping[str, Any], expected: Iterable[str], label: str
) -> None:
    _require(
        set(record) == set(expected),
        f"{label} has unexpected or missing fields",
    )


def _tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _junit_summary(payload: bytes, label: str) -> dict[str, Any]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as error:
        raise AttestationError(f"{label} is not valid XML") from error
    _require(
        _tag_name(root) in {"testsuite", "testsuites"},
        f"{label} has an invalid root",
    )
    suites = (
        [root]
        if _tag_name(root) == "testsuite"
        else [child for child in root if _tag_name(child) == "testsuite"]
    )
    _require(bool(suites), f"{label} contains no test suite")
    cases = [item for item in root.iter() if _tag_name(item) == "testcase"]
    names = [item.attrib.get("name", "") for item in cases]
    _require(all(names), f"{label} contains an unnamed testcase")
    _require(len(names) == len(set(names)), f"{label} contains duplicate testcase names")
    failures = sum(
        any(_tag_name(child) == "failure" for child in item) for item in cases
    )
    errors = sum(any(_tag_name(child) == "error" for child in item) for item in cases)
    skipped = sum(any(_tag_name(child) == "skipped" for child in item) for item in cases)
    try:
        reported = {
            key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
    except ValueError as error:
        raise AttestationError(f"{label} contains a non-integer suite counter") from error
    actual = {
        "tests": len(cases),
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }
    _require(reported == actual, f"{label} suite counters disagree with its testcases")
    return {**actual, "test_names": sorted(names)}


def _manifest_root(path: Path) -> Path:
    resolved = path.resolve()
    _require(
        resolved.parent.name == "athena",
        "source manifest must live directly under athena/",
    )
    return resolved.parent.parent


def _parse_manifest(payload: bytes, label: str) -> dict[str, str]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise AttestationError(f"{label} is not UTF-8") from error
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        _require(len(fields) == 2, f"{label} line {line_number} is malformed")
        digest, relative = fields
        relative = relative.lstrip("*")
        _require_sha256(digest, f"{label} line {line_number} digest")
        candidate = Path(relative)
        _require(not candidate.is_absolute(), f"{label} contains an absolute path")
        _require(".." not in candidate.parts, f"{label} path escapes its source root")
        normalized = candidate.as_posix()
        _require(normalized not in {"", "."}, f"{label} contains an empty path")
        _require(normalized not in result, f"{label} contains duplicate path {normalized}")
        result[normalized] = digest
    _require(bool(result), f"{label} is empty")
    return result


def _validate_manifest_identities(manifest: Mapping[str, Any]) -> None:
    _require(manifest.get("sha256") == R7_MANIFEST_SHA256, "manifest SHA differs")
    _exact_integer(
        manifest.get("source_count"),
        EXPECTED_SOURCE_COUNT,
        "manifest source count",
    )
    source_map = _mapping(manifest.get("source_sha256"), "manifest source map")
    expected = {
        PRODUCTION_CORE_PATH: PRODUCTION_CORE_SHA256,
        INDEPENDENT_REFERENCE_PATH: INDEPENDENT_REFERENCE_SHA256,
        ORACLE_RUNNER_PATH: ORACLE_RUNNER_SHA256,
        ORACLE_TEST_PATH: ORACLE_TEST_SHA256,
        FULL_RUNNER_PATH: FULL_RUNNER_SHA256,
    }
    for path, digest in expected.items():
        _require(source_map.get(path) == digest, f"r7 source identity differs: {path}")


def _load_manifest(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    payload, digest = _stable_file_bytes(path, label)
    _require(digest == expected_sha256, f"{label} SHA-256 differs from its r7 pin")
    expected = _parse_manifest(payload, label)
    root = _manifest_root(path)
    actual: dict[str, str] = {}
    for relative, expected_digest in sorted(expected.items()):
        source = root / relative
        resolved = source.resolve()
        _require(
            resolved != root and root in resolved.parents,
            f"{label} source escapes its source root: {relative}",
        )
        source_digest = _stable_file_sha256(source, f"{label} source {relative}")
        _require(
            source_digest == expected_digest,
            f"{label} source hash changed: {relative}",
        )
        actual[relative] = source_digest
    manifest = {
        "path": path.resolve().as_posix(),
        "root": root.as_posix(),
        "sha256": digest,
        "source_count": len(actual),
        "source_sha256": actual,
    }
    _validate_manifest_identities(manifest)
    return manifest


def _validate_embedded_manifest(
    value: object, expected: Mapping[str, Any], label: str
) -> None:
    record = _mapping(value, label)
    _require(
        set(record) == {"path", "sha256", "source_count", "source_sha256"},
        f"{label} has unexpected or missing fields",
    )
    _require(
        record.get("path") == R7_MANIFEST_PATH,
        f"{label} is not from the frozen r7 root",
    )
    _require(
        record.get("sha256") == R7_MANIFEST_SHA256,
        f"{label} manifest SHA differs",
    )
    _exact_integer(
        record.get("source_count"),
        EXPECTED_SOURCE_COUNT,
        f"{label}.source_count",
    )
    _require(
        record.get("source_sha256") == expected["source_sha256"],
        f"{label} source map differs",
    )


def _validate_static_audit(
    value: object, manifest: Mapping[str, Any], label: str
) -> None:
    audit = _mapping(value, label)
    _validate_finite_json(audit, label)
    _require_exact_fields(
        audit,
        {
            "call_site_count",
            "direct_qr_call_sites",
            "entrypoints",
            "guarded_dormant_spectral_norm_sites",
            "prohibited_calls_found",
            "prohibited_self_overlap_sites",
            "scope",
            "source_count",
            "source_sha256",
        },
        label,
    )
    _require(
        audit.get("scope") == "transitive_local_import_closure",
        f"{label} scope differs",
    )
    _exact_integer(
        audit.get("source_count"),
        EXPECTED_SOURCE_COUNT,
        f"{label}.source_count",
    )
    _require(
        audit.get("source_sha256") == manifest["source_sha256"],
        f"{label} source map differs",
    )
    _exact_integer(
        audit.get("call_site_count"),
        EXPECTED_STATIC_CALL_SITE_COUNT,
        f"{label}.call_site_count",
    )
    _exact_integer(
        audit.get("direct_qr_call_sites"),
        EXPECTED_TRANSITIVE_DIRECT_QR_SITES,
        f"{label}.direct_qr_call_sites",
    )
    _require(
        frozenset(_sequence(audit.get("entrypoints"), f"{label}.entrypoints"))
        == EXPECTED_STATIC_ENTRYPOINTS,
        f"{label} entrypoint set differs",
    )
    _require(
        audit.get("prohibited_calls_found") == [],
        f"{label} found a prohibited call",
    )
    _require(
        audit.get("prohibited_self_overlap_sites") == [],
        f"{label} found a self-overlap",
    )
    _require(
        audit.get("guarded_dormant_spectral_norm_sites")
        == list(EXPECTED_GUARDED_DORMANT_SPECTRAL_NORM_SITES),
        f"{label} guarded dormant spectral-norm sites differ",
    )


def _validate_runtime_report(
    value: object,
    label: str,
    *,
    expected_calls: Iterable[str] = (),
    expected_call_count: int = 0,
) -> None:
    report = _mapping(value, label)
    _validate_finite_json(report, label)
    _require_exact_fields(
        report,
        {
            "allowed_call_count",
            "allowed_calls",
            "installed",
            "patched_entrypoint_count",
            "patched_entrypoints",
            "prohibited_attempt_count",
            "prohibited_attempts",
        },
        label,
    )
    _require(report.get("installed") is True, f"{label} guard was not installed")
    patched = _sequence(
        report.get("patched_entrypoints"), f"{label}.patched_entrypoints"
    )
    _require(
        len(EXPECTED_RUNTIME_PATCHED_ENTRYPOINTS)
        == EXPECTED_RUNTIME_PATCHED_ENTRYPOINT_COUNT,
        "attester guard-surface pin has the wrong cardinality",
    )
    _require(len(patched) == len(set(patched)), f"{label} patch ledger has duplicates")
    _require(
        frozenset(patched) == EXPECTED_RUNTIME_PATCHED_ENTRYPOINTS,
        f"{label} patched-target set differs from the exact r7 guard surface",
    )
    _exact_integer(
        report.get("patched_entrypoint_count"),
        EXPECTED_RUNTIME_PATCHED_ENTRYPOINT_COUNT,
        f"{label}.patched_entrypoint_count",
    )
    _require(
        _integer(
            report.get("prohibited_attempt_count"),
            f"{label}.prohibited_attempt_count",
        )
        == 0,
        f"{label} recorded a prohibited attempt",
    )
    _require(
        report.get("prohibited_attempts") == [],
        f"{label} prohibited ledger is nonempty",
    )
    allowed = _sequence(report.get("allowed_calls"), f"{label}.allowed_calls")
    _require(len(allowed) == len(set(allowed)), f"{label} allowed-call names repeat")
    expected_set = frozenset(expected_calls)
    _require(
        expected_set <= ALLOWED_RUNTIME_CALLS,
        f"{label} verifier expected an unknown caller",
    )
    _require(frozenset(allowed) == expected_set, f"{label} allowed-call set differs")
    _require(
        _integer(report.get("allowed_call_count"), f"{label}.allowed_call_count")
        == expected_call_count,
        f"{label} allowed-call count differs from the exact r7 execution",
    )


def _validate_junit_binding(
    artifact: Mapping[str, Any],
    junit_payload: bytes,
    junit_sha256: str,
    label: str,
) -> dict[str, Any]:
    summary = _junit_summary(junit_payload, f"{label} JUnit")
    _require(summary["tests"] == 16, f"{label} JUnit test count differs")
    _require(summary["failures"] == 0, f"{label} JUnit contains a failure")
    _require(summary["errors"] == 0, f"{label} JUnit contains an error")
    _require(summary["skipped"] == 0, f"{label} JUnit contains a skip")
    _require(
        frozenset(summary["test_names"]) == R7_TEST_NAMES,
        f"{label} JUnit names differ",
    )
    embedded = _mapping(artifact.get("junit"), f"{label}.junit")
    _require_exact_fields(
        embedded,
        {"errors", "failures", "path", "sha256", "skipped", "test_names", "tests"},
        f"{label}.junit",
    )
    _require(
        embedded.get("path") == R7_ORACLE_JUNIT_PATH,
        f"{label} embedded JUnit path differs",
    )
    _require(
        embedded.get("sha256") == junit_sha256 == R7_ORACLE_JUNIT_SHA256,
        f"{label} embedded JUnit hash differs",
    )
    for key in ("tests", "failures", "errors", "skipped"):
        _exact_integer(
            embedded.get(key), summary[key], f"{label}.junit.{key}"
        )
    _require(
        embedded.get("test_names") == summary["test_names"],
        f"{label} embedded JUnit test_names differs",
    )
    return summary


def _validate_oracle_artifact(
    artifact: Mapping[str, Any],
    *,
    junit_payload: bytes,
    junit_sha256: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    label = "r7 16-test clone oracle"
    _validate_finite_json(artifact, label)
    _require_exact_fields(
        artifact,
        {
            "bounded_independent_clone_oracle_passed",
            "canonical_odt_certified",
            "certification_status",
            "claim_boundary",
            "elapsed_seconds",
            "junit",
            "peak_rss_mb",
            "pytest_exit_code",
            "pytest_plugin_autoload_disabled",
            "runtime_direct_only_guard_at_import",
            "runtime_direct_only_guard_final",
            "schema",
            "source_manifest_before_result",
            "source_manifest_before_test",
            "static_audit",
            "test_path",
        },
        label,
    )
    _require(
        artifact.get("schema")
        == "xvla-bounded-independent-direct-odt-clone-oracle-v1",
        f"{label} schema differs",
    )
    _require(
        artifact.get("claim_boundary") == ORACLE_CLAIM_BOUNDARY,
        f"{label} claim boundary differs",
    )
    _require(artifact.get("test_path") == ORACLE_TEST_PATH, f"{label} test path differs")
    _require(
        _integer(artifact.get("pytest_exit_code"), f"{label}.pytest_exit_code") == 0,
        f"{label} pytest failed",
    )
    _require(
        artifact.get("pytest_plugin_autoload_disabled") is True,
        f"{label} plugin isolation is absent",
    )
    _require(
        artifact.get("bounded_independent_clone_oracle_passed") is True,
        f"{label} did not pass",
    )
    _require(
        artifact.get("canonical_odt_certified") is False,
        f"{label} overstates certification",
    )
    _require(
        artifact.get("certification_status")
        == "bounded_independent_clone_oracle_passed_awaiting_composite_attestation",
        f"{label} certification status differs",
    )
    _require(
        _number(artifact.get("elapsed_seconds"), f"{label}.elapsed_seconds") > 0,
        f"{label} elapsed time is invalid",
    )
    _require(
        _number(artifact.get("peak_rss_mb"), f"{label}.peak_rss_mb") > 0,
        f"{label} peak RSS is invalid",
    )
    junit = _validate_junit_binding(artifact, junit_payload, junit_sha256, label)
    _validate_embedded_manifest(
        artifact.get("source_manifest_before_test"),
        manifest,
        f"{label}.source_manifest_before_test",
    )
    _validate_embedded_manifest(
        artifact.get("source_manifest_before_result"),
        manifest,
        f"{label}.source_manifest_before_result",
    )
    _validate_static_audit(
        artifact.get("static_audit"), manifest, f"{label}.static_audit"
    )
    _validate_runtime_report(
        artifact.get("runtime_direct_only_guard_at_import"),
        f"{label}.runtime_guard_at_import",
    )
    _validate_runtime_report(
        artifact.get("runtime_direct_only_guard_final"),
        f"{label}.runtime_guard_final",
        expected_calls=ORACLE_REQUIRED_CALLS,
        expected_call_count=EXPECTED_ORACLE_ALLOWED_CALL_COUNT,
    )
    return junit


def _validate_zero_prohibited_counters(value: object, label: str) -> None:
    meter = _mapping(value, label)
    _validate_finite_json(meter, label)
    for key in (
        "svd_calls",
        "polar_calls",
        "normal_equation_factorizations",
        "raw_order_three_core_materializations",
        "raw_width_cubic_core_materializations",
    ):
        _require(
            _integer(meter.get(key), f"{label}.{key}") == 0,
            f"{label}.{key} is nonzero",
        )


def _validate_route_inventory(value: object, label: str) -> dict[str, Any]:
    inventory = _mapping(value, label)
    _validate_finite_json(inventory, label)
    _require(
        _same_json_value_and_type(inventory, EXPECTED_ROUTE_INVENTORY),
        f"{label} differs from the exact r7 route inventory",
    )
    _require(
        inventory["bounded_retained_q_signed_storage_delta_elements"]
        == inventory["bounded_retained_q_total_elements"]
        - inventory["bounded_retained_q_replaced_cp_total_elements"],
        f"{label} signed storage identity failed",
    )
    _require(
        inventory["bounded_retained_q_maximum_elements"]
        <= inventory["bounded_explicit_q_element_limit"],
        f"{label} retained Q exceeds its bound",
    )
    _require(
        inventory["bounded_retained_q_maximum_unfolding_elements"]
        <= inventory["bounded_explicit_unfolding_element_limit"],
        f"{label} bounded unfolding exceeds its bound",
    )
    return inventory


def _validate_full_result(
    result: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    label = "r7 full shared-DAG result"
    _validate_finite_json(result, label)
    _require_exact_fields(
        result,
        {
            "action_head",
            "algorithm1_exact_predicted_route_inventory",
            "algorithm1_live_progress",
            "all_completed_gates_pass",
            "all_gates_pass",
            "boundary_algorithms",
            "bounded_independent_clone_oracle_attached",
            "canonical_odt_certified",
            "certification_status",
            "checkpoint_sha256",
            "claim_boundary",
            "compile_seconds",
            "direct_call_audit",
            "evaluate_seconds",
            "full_shared_dag_algorithms_1_to_3_completed",
            "gates",
            "immutable_source_manifest",
            "implicit_shape",
            "lane",
            "peak_rss_mb",
            "runtime_direct_only_guard_at_import",
            "runtime_direct_only_guard_final",
            "schema",
            "source_manifest_before_result",
            "source_observable_relative_error",
            "source_replay_relative_error",
            "stage",
            "state_sha256_after",
            "state_sha256_before",
            "structure",
            "telemetry",
            "transitive_static_audit",
        },
        label,
    )
    _require(
        result.get("schema") == "xvla-full-heterogeneous-shared-dag-odt-v1",
        f"{label} schema differs",
    )
    _require(result.get("lane") == "production", f"{label} is not production")
    _require(result.get("stage") == "full", f"{label} did not run Algorithms 1-3")
    _require(result.get("action_head") == "linear", f"{label} action head differs")
    _require(
        result.get("checkpoint_sha256") == CHECKPOINT_SHA256,
        f"{label} checkpoint differs",
    )
    _require(
        result.get("claim_boundary") == FULL_CLAIM_BOUNDARY,
        f"{label} claim boundary differs",
    )
    _require(
        result.get("state_sha256_before") == result.get("state_sha256_after"),
        f"{label} mutated model state",
    )
    _require(
        _is_sha256(result.get("state_sha256_before")),
        f"{label} state digest is invalid",
    )
    _require(
        result.get("all_completed_gates_pass") is True,
        f"{label} completed gates did not pass",
    )
    _require(
        result.get("full_shared_dag_algorithms_1_to_3_completed") is True,
        f"{label} did not complete Algorithms 1-3",
    )
    _require(
        result.get("bounded_independent_clone_oracle_attached") is False,
        f"{label} unexpectedly embeds an oracle",
    )
    _require(
        result.get("canonical_odt_certified") is False,
        f"{label} source artifact overstates certification",
    )
    _require(
        result.get("all_gates_pass") is False,
        f"{label} legacy certification flag must remain false",
    )
    _require(
        result.get("certification_status")
        == "full_shared_dag_algorithms_1_to_3_completed_awaiting_composite_clone_attestation",
        f"{label} certification status differs",
    )

    gates = _mapping(result.get("gates"), f"{label}.gates")
    _require(
        frozenset(gates) == EXPECTED_FULL_GATE_NAMES,
        f"{label} gate set differs",
    )
    _require(
        all(type(value) is bool and value for value in gates.values()),
        f"{label} contains a false or non-boolean gate",
    )

    _validate_embedded_manifest(
        result.get("immutable_source_manifest"),
        manifest,
        f"{label}.immutable_source_manifest",
    )
    _validate_embedded_manifest(
        result.get("source_manifest_before_result"),
        manifest,
        f"{label}.source_manifest_before_result",
    )
    _validate_static_audit(
        result.get("transitive_static_audit"),
        manifest,
        f"{label}.transitive_static_audit",
    )
    direct_audit = _mapping(
        result.get("direct_call_audit"), f"{label}.direct_call_audit"
    )
    _require_exact_fields(
        direct_audit,
        {
            "direct_qr_required",
            "observed_calls",
            "prohibited_calls_found",
            "scope",
            "static_direct_qr_call_sites",
        },
        f"{label}.direct_call_audit",
    )
    _require(
        direct_audit.get("scope") == "entire_module",
        f"{label} direct-call scope differs",
    )
    _require(
        direct_audit.get("direct_qr_required") is True,
        f"{label} direct QR was not required",
    )
    _require(
        isinstance(direct_audit.get("observed_calls"), list),
        f"{label} direct-call observed ledger is missing",
    )
    observed_calls = _sequence(
        direct_audit.get("observed_calls"), f"{label}.direct_call_audit.observed_calls"
    )
    _require(
        len(observed_calls) == 204
        and len(set(observed_calls)) == 204
        and all(isinstance(call, str) and call for call in observed_calls),
        f"{label} direct-call observed ledger differs",
    )
    observed_calls_payload = json.dumps(
        observed_calls,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    _require(
        hashlib.sha256(observed_calls_payload).hexdigest()
        == EXPECTED_DIRECT_CALL_AUDIT_OBSERVED_CALLS_SHA256,
        f"{label} direct-call observed ledger digest differs",
    )
    _exact_integer(
        direct_audit.get("static_direct_qr_call_sites"),
        EXPECTED_MODULE_DIRECT_QR_SITES,
        f"{label}.direct_call_audit.static_direct_qr_call_sites",
    )
    _require(
        direct_audit.get("prohibited_calls_found") == [],
        f"{label} direct-call audit failed",
    )
    _validate_runtime_report(
        result.get("runtime_direct_only_guard_at_import"),
        f"{label}.runtime_guard_at_import",
    )
    _validate_runtime_report(
        result.get("runtime_direct_only_guard_final"),
        f"{label}.runtime_guard_final",
        expected_calls=FULL_REQUIRED_CALLS,
        expected_call_count=EXPECTED_FULL_ALLOWED_CALL_COUNT,
    )
    compile_meter = _mapping(result.get("telemetry"), f"{label}.compile_telemetry")
    _validate_zero_prohibited_counters(compile_meter, f"{label}.compile_telemetry")
    expected_compile_meter = {
        "bounded_explicit_direct_rq_factorizations": 0,
        "constant_cp_to_constant_folds": 4_110,
        "constant_cp_to_unary_folds": 24_346,
        "constant_unary_folds": 1_260,
        "direct_q_columns_compared": 0,
        "direct_q_provenance_certificates": 0,
        "ephemeral_ffn_bond_diagonalizations": 0,
        "fused_ffn_primitive_count": 1_013,
        "householder_qr_kernel_calls": 0,
        "local_direct_rq_factorizations": 0,
        "maximum_bounded_explicit_q_elements": 0,
        "maximum_direct_q_compact_relative_error": 0.0,
        "maximum_direct_q_panel_elements": 0,
        "maximum_direct_q_reconstruction_relative_error": 0.0,
        "maximum_direct_q_transition_elements": 0,
        "maximum_persistent_tensor_elements": 443_905,
        "maximum_rectangular_q_elements": 0,
        "maximum_temporary_tensor_elements": 0,
        "maximum_temporary_tensor_order": 0,
        "normal_equation_factorizations": 0,
        "polar_calls": 0,
        "raw_order_three_core_materializations": 0,
        "raw_width_cubic_core_materializations": 0,
        "rectangular_direct_rq_factorizations": 0,
        "streamed_direct_q_provenance_certificates": 0,
        "streamed_direct_q_replay_qr_factorizations": 0,
        "svd_calls": 0,
    }
    _require(
        _same_json_value_and_type(compile_meter, expected_compile_meter),
        f"{label} compile telemetry differs from r7",
    )

    structure = _mapping(result.get("structure"), f"{label}.structure")
    shape = _mapping(result.get("implicit_shape"), f"{label}.implicit_shape")
    _require_exact_fields(
        structure,
        {
            "active_pade_sites",
            "edge_occurrences",
            "maximum_physical_source_width",
            "observable_euclidean_width",
            "observable_kind",
            "physical_leaf_count",
            "physical_source_count",
            "physical_source_widths",
            "product_signs",
            "root_projective_width",
            "unique_nodes",
        },
        f"{label}.structure",
    )
    expected_shape = {
        "cp_binary_nodes": 3_252_488,
        "dense_clone_nodes": 0,
        "edge_occurrences": EXPECTED_EDGE_COUNT,
        "fused_token_feature_dimension": 13_129,
        "heterogeneous_physical_sources": EXPECTED_POLICY_SOURCE_COUNT,
        "maximum_cp_rank": 1_153,
        "maximum_local_bond_dimension": 385,
        "maximum_reduced_q_elements": 0,
        "reduced_q_binary_nodes": 0,
        "unary_nodes": 711_975,
        "unique_nodes": EXPECTED_NODE_COUNT,
    }
    _require(
        _same_json_value_and_type(shape, expected_shape),
        f"{label} implicit shape differs from r7",
    )
    for record, key, expected, record_label in (
        (structure, "unique_nodes", EXPECTED_NODE_COUNT, "structure"),
        (shape, "unique_nodes", EXPECTED_NODE_COUNT, "implicit_shape"),
        (structure, "edge_occurrences", EXPECTED_EDGE_COUNT, "structure"),
        (shape, "edge_occurrences", EXPECTED_EDGE_COUNT, "implicit_shape"),
        (
            structure,
            "physical_source_count",
            EXPECTED_POLICY_SOURCE_COUNT,
            "structure",
        ),
        (
            structure,
            "physical_leaf_count",
            EXPECTED_POLICY_SOURCE_COUNT,
            "structure",
        ),
        (
            shape,
            "heterogeneous_physical_sources",
            EXPECTED_POLICY_SOURCE_COUNT,
            "implicit_shape",
        ),
        (structure, "active_pade_sites", EXPECTED_PADE_SITES, "structure"),
        (structure, "root_projective_width", EXPECTED_ROOT_WIDTH, "structure"),
        (
            structure,
            "observable_euclidean_width",
            EXPECTED_ROOT_WIDTH - 1,
            "structure",
        ),
    ):
        _exact_integer(record.get(key), expected, f"{label}.{record_label}.{key}")
    _require(
        structure.get("observable_kind") == "linear_action",
        f"{label} observable kind differs",
    )
    _require_null(structure, "product_signs", label)
    _exact_integer(
        structure.get("maximum_physical_source_width"),
        192,
        f"{label}.structure.maximum_physical_source_width",
    )
    _require(
        _integer(shape.get("dense_clone_nodes"), f"{label}.dense_clone_nodes") == 0,
        f"{label} has a dense clone core",
    )

    widths = _mapping(
        structure.get("physical_source_widths"), f"{label}.physical_source_widths"
    )
    expected_widths = {
        **{f"image.patch{index}": 192 for index in range(64)},
        **{f"instruction.token{index}": 26 for index in range(32)},
        "state": 8,
    }
    _require(
        _same_json_value_and_type(widths, expected_widths),
        f"{label} heterogeneous source registry differs",
    )

    _bounded_nonnegative(
        result.get("source_replay_relative_error"),
        GLOBAL_REPLAY_TOLERANCE,
        f"{label}.source_replay",
    )
    _bounded_nonnegative(
        result.get("source_observable_relative_error"),
        GLOBAL_REPLAY_TOLERANCE,
        f"{label}.source_observable",
    )
    _require(
        _number(result.get("compile_seconds"), f"{label}.compile_seconds") > 0,
        f"{label} compile timing is invalid",
    )
    _require(
        _number(result.get("evaluate_seconds"), f"{label}.evaluate_seconds") > 0,
        f"{label} evaluation timing is invalid",
    )
    _require(
        _number(result.get("peak_rss_mb"), f"{label}.peak_rss_mb") > 0,
        f"{label} peak RSS is invalid",
    )

    route = _validate_route_inventory(
        result.get("algorithm1_exact_predicted_route_inventory"),
        f"{label}.algorithm1_exact_predicted_route_inventory",
    )
    live = _mapping(
        result.get("algorithm1_live_progress"),
        f"{label}.algorithm1_live_progress",
    )
    _validate_finite_json(live, f"{label}.algorithm1_live_progress")
    _require_exact_fields(
        live,
        {
            "completed_steps",
            "elapsed_seconds",
            "last_direct_q_columns_compared",
            "last_direct_q_provenance_method",
            "last_label",
            "last_method",
            "last_uid",
            "last_unfolding_shape",
            "peak_rss_mb",
            "total_steps",
        },
        f"{label}.algorithm1_live_progress",
    )
    _exact_integer(
        live.get("completed_steps"),
        EXPECTED_NODE_COUNT,
        f"{label}.algorithm1_live_progress.completed_steps",
    )
    _exact_integer(
        live.get("total_steps"),
        EXPECTED_NODE_COUNT,
        f"{label}.algorithm1_live_progress.total_steps",
    )
    _require(
        live.get("last_method") == "bounded_explicit_householder_direct_reduced_rq",
        f"{label} terminal method differs",
    )
    _require(
        live.get("last_direct_q_provenance_method")
        == f"retained_explicit_q::{live['last_method']}",
        f"{label} terminal direct-Q provenance differs",
    )
    _exact_integer(
        live.get("last_direct_q_columns_compared"),
        841,
        f"{label}.algorithm1_live_progress.last_direct_q_columns_compared",
    )
    _require(
        live.get("last_unfolding_shape") == [57, 841],
        f"{label} terminal unfolding shape differs",
    )
    _require(
        live.get("last_label") == "policy.action_concat.level2.pair0",
        f"{label} terminal node label differs",
    )
    _exact_integer(
        live.get("last_uid"),
        3_973_966,
        f"{label}.algorithm1_live_progress.last_uid",
    )
    _require(
        _number(live.get("elapsed_seconds"), f"{label}.live_elapsed") > 0,
        f"{label} live timing is invalid",
    )
    _require(
        _number(live.get("peak_rss_mb"), f"{label}.live_peak_rss") > 0,
        f"{label} live RSS is invalid",
    )

    algorithms = _mapping(
        result.get("boundary_algorithms"), f"{label}.boundary_algorithms"
    )
    _require_exact_fields(
        algorithms,
        {
            "algorithm1_absorption_maximum_exponent_delta",
            "algorithm1_canonical_exponent_normal_form",
            "algorithm1_direct_q_provenance",
            "algorithm1_final_projective_replay_relative_error",
            "algorithm1_full_projective_evaluations",
            "algorithm1_in_place",
            "algorithm1_local_reconstruction_maximum_exponent_delta",
            "algorithm1_maximum_absorption_scaled_relative_error",
            "algorithm1_maximum_local_scaled_reconstruction_relative_error",
            "algorithm1_maximum_step_replay_relative_error",
            "algorithm1_per_step_replays_performed",
            "algorithm1_push_ledger",
            "algorithm1_scale_sensitive_occurrence_ledger",
            "algorithm2_child_messages",
            "algorithm2_seconds",
            "algorithm2_streamed_inside_algorithm3",
            "algorithm3_diagonalized_node_count",
            "algorithm3_eigenvalue_spectra_retained",
            "algorithm3_offdiagonal_ratio",
            "algorithm3_push_ledger",
            "algorithm3_replay_relative_error",
            "algorithm3_seconds",
            "canonical_seconds",
            "canonical_telemetry",
            "factorization_methods",
            "rectangular_direct_rq_steps",
        },
        f"{label}.boundary_algorithms",
    )
    _require(
        algorithms.get("algorithm1_in_place") is True,
        f"{label} Algorithm 1 was not in place",
    )
    _require(
        _integer(
            algorithms.get("algorithm1_per_step_replays_performed"),
            f"{label}.algorithm1_per_step_replays_performed",
        )
        == 0,
        f"{label} performed quadratic full replays",
    )
    _exact_integer(
        algorithms.get("algorithm1_full_projective_evaluations"),
        2,
        f"{label}.algorithm1_full_projective_evaluations",
    )
    _require_null(
        algorithms,
        "algorithm1_maximum_step_replay_relative_error",
        f"{label}.boundary_algorithms",
    )
    _bounded_nonnegative(
        algorithms.get("algorithm1_final_projective_replay_relative_error"),
        GLOBAL_REPLAY_TOLERANCE,
        f"{label}.algorithm1_final_replay",
    )
    _bounded_nonnegative(
        algorithms.get(
            "algorithm1_maximum_local_scaled_reconstruction_relative_error"
        ),
        LOCAL_CERTIFICATE_TOLERANCE,
        f"{label}.local_reconstruction",
    )
    _bounded_nonnegative(
        algorithms.get("algorithm1_maximum_absorption_scaled_relative_error"),
        LOCAL_CERTIFICATE_TOLERANCE,
        f"{label}.absorption",
    )
    _require(
        _integer(
            algorithms.get(
                "algorithm1_local_reconstruction_maximum_exponent_delta"
            ),
            f"{label}.algorithm1_local_reconstruction_maximum_exponent_delta",
        )
        == 0,
        f"{label} local exponent delta is nonzero",
    )
    _require(
        _integer(
            algorithms.get("algorithm1_absorption_maximum_exponent_delta"),
            f"{label}.algorithm1_absorption_maximum_exponent_delta",
        )
        == 0,
        f"{label} absorption exponent delta is nonzero",
    )
    expected_pushes = EXPECTED_EDGE_COUNT + 1
    _exact_integer_list(
        algorithms.get("algorithm1_push_ledger"),
        [expected_pushes, expected_pushes],
        f"{label}.algorithm1_push_ledger",
    )
    _exact_integer_list(
        algorithms.get("algorithm1_scale_sensitive_occurrence_ledger"),
        [expected_pushes, expected_pushes],
        f"{label}.algorithm1_scale_sensitive_occurrence_ledger",
    )
    factorization_methods = _sequence(
        algorithms.get("factorization_methods"), f"{label}.factorization_methods"
    )
    _require(
        len(factorization_methods) == 3
        and len(set(factorization_methods)) == 3
        and all(isinstance(method, str) for method in factorization_methods)
        and frozenset(factorization_methods)
        == {
            "streamed_householder_direct_reduced_rq",
            "torch_qr_transpose_direct_reduced_rq",
            "bounded_explicit_householder_direct_reduced_rq",
        },
        f"{label} factorization method set differs",
    )
    _exact_integer(
        algorithms.get("rectangular_direct_rq_steps"),
        EXPECTED_RECTANGULAR_STEPS,
        f"{label}.rectangular_direct_rq_steps",
    )

    normal = _mapping(
        algorithms.get("algorithm1_canonical_exponent_normal_form"),
        f"{label}.canonical_normal_form",
    )
    _require_exact_fields(
        normal,
        {
            "all_core_exponents_zero",
            "all_streamed_compact_cp_q_has_direct_q_provenance",
            "core_count",
            "cp_direct_q_certificates",
            "cp_direct_q_columns",
            "head_binary_exponent",
            "head_exponent_matches_ledger",
            "streamed_compact_cp_direct_q_certificates",
            "streamed_compact_cp_direct_q_columns",
        },
        f"{label}.canonical_normal_form",
    )
    _exact_integer(
        normal.get("core_count"),
        EXPECTED_NODE_COUNT,
        f"{label}.canonical_normal_form.core_count",
    )
    _require(
        normal.get("all_core_exponents_zero") is True,
        f"{label} has a nonzero canonical core exponent",
    )
    _require(
        normal.get("head_exponent_matches_ledger") is True,
        f"{label} head exponent ledger differs",
    )
    _require(
        normal.get("all_streamed_compact_cp_q_has_direct_q_provenance") is True,
        f"{label} compact CP Q provenance is incomplete",
    )
    _exact_integer(
        normal.get("cp_direct_q_certificates"),
        EXPECTED_STREAMED_STEPS,
        f"{label}.canonical_normal_form.cp_direct_q_certificates",
    )
    _exact_integer(
        normal.get("streamed_compact_cp_direct_q_certificates"),
        EXPECTED_STREAMED_STEPS,
        f"{label}.canonical_normal_form.streamed_compact_cp_direct_q_certificates",
    )
    _exact_integer(
        normal.get("cp_direct_q_columns"),
        363_272_585,
        f"{label}.canonical_normal_form.cp_direct_q_columns",
    )
    _exact_integer(
        normal.get("streamed_compact_cp_direct_q_columns"),
        363_272_585,
        f"{label}.canonical_normal_form.streamed_compact_cp_direct_q_columns",
    )
    _integer(normal.get("head_binary_exponent"), f"{label}.head_binary_exponent")

    provenance = _mapping(
        algorithms.get("algorithm1_direct_q_provenance"),
        f"{label}.direct_q_provenance",
    )
    _require_exact_fields(
        provenance,
        {
            "columns_compared",
            "expected_columns",
            "expected_steps",
            "expected_streamed_replay_qr_calls",
            "maximum_compact_q_relative_error",
            "maximum_direct_q_reconstruction_relative_error",
            "streamed_certificates",
            "streamed_replay_qr_calls",
            "streamed_steps",
            "verified_steps",
        },
        f"{label}.direct_q_provenance",
    )
    exact_provenance_counts = {
        "verified_steps": EXPECTED_NODE_COUNT,
        "expected_steps": EXPECTED_NODE_COUNT,
        "streamed_steps": EXPECTED_STREAMED_STEPS,
        "streamed_certificates": EXPECTED_STREAMED_STEPS,
        "columns_compared": EXPECTED_DIRECT_Q_COLUMNS,
        "expected_columns": EXPECTED_DIRECT_Q_COLUMNS,
        "streamed_replay_qr_calls": EXPECTED_STREAMED_REPLAY_QR_CALLS,
        "expected_streamed_replay_qr_calls": EXPECTED_STREAMED_REPLAY_QR_CALLS,
    }
    for key, expected in exact_provenance_counts.items():
        _exact_integer(
            provenance.get(key),
            expected,
            f"{label}.direct_q_provenance.{key}",
        )
    _bounded_nonnegative(
        provenance.get("maximum_compact_q_relative_error"),
        LOCAL_CERTIFICATE_TOLERANCE,
        f"{label}.compact_q_error",
    )
    _bounded_nonnegative(
        provenance.get("maximum_direct_q_reconstruction_relative_error"),
        LOCAL_CERTIFICATE_TOLERANCE,
        f"{label}.direct_q_reconstruction",
    )

    meter = _mapping(
        algorithms.get("canonical_telemetry"), f"{label}.canonical_telemetry"
    )
    _require_exact_fields(
        meter,
        expected_compile_meter,
        f"{label}.canonical_telemetry",
    )
    _validate_zero_prohibited_counters(meter, f"{label}.canonical_telemetry")
    exact_meter_counts = {
        "local_direct_rq_factorizations": EXPECTED_NODE_COUNT,
        "direct_q_provenance_certificates": EXPECTED_NODE_COUNT,
        "direct_q_columns_compared": EXPECTED_DIRECT_Q_COLUMNS,
        "streamed_direct_q_provenance_certificates": EXPECTED_STREAMED_STEPS,
        "streamed_direct_q_replay_qr_factorizations": EXPECTED_STREAMED_REPLAY_QR_CALLS,
        "bounded_explicit_direct_rq_factorizations": route[
            "bounded_retained_q_candidate_count"
        ],
        "maximum_bounded_explicit_q_elements": route[
            "bounded_retained_q_maximum_elements"
        ],
        "householder_qr_kernel_calls": 4_143_340,
        "constant_cp_to_constant_folds": 0,
        "constant_cp_to_unary_folds": 0,
        "constant_unary_folds": 0,
        "ephemeral_ffn_bond_diagonalizations": 0,
        "fused_ffn_primitive_count": 0,
        "maximum_direct_q_panel_elements": 1_725_185,
        "maximum_direct_q_transition_elements": 5_484_325,
        "maximum_persistent_tensor_elements": 3_739_329,
        "maximum_rectangular_q_elements": 2_916,
        "maximum_temporary_tensor_elements": 4_722_688,
        "maximum_temporary_tensor_order": 2,
        "rectangular_direct_rq_factorizations": 33,
    }
    for key, expected in exact_meter_counts.items():
        _exact_integer(meter.get(key), expected, f"{label}.canonical_telemetry.{key}")
    _bounded_nonnegative(
        meter.get("maximum_direct_q_compact_relative_error"),
        LOCAL_CERTIFICATE_TOLERANCE,
        f"{label}.canonical_telemetry.maximum_direct_q_compact_relative_error",
    )
    _bounded_nonnegative(
        meter.get("maximum_direct_q_reconstruction_relative_error"),
        LOCAL_CERTIFICATE_TOLERANCE,
        f"{label}.canonical_telemetry.maximum_direct_q_reconstruction_relative_error",
    )

    _require(
        algorithms.get("algorithm2_streamed_inside_algorithm3") is True,
        f"{label} did not use streamed Algorithm 2",
    )
    _require_null(
        algorithms, "algorithm2_seconds", f"{label}.boundary_algorithms"
    )
    _exact_integer(
        algorithms.get("algorithm2_child_messages"),
        EXPECTED_EDGE_COUNT,
        f"{label}.algorithm2_child_messages",
    )
    _exact_integer(
        algorithms.get("algorithm3_diagonalized_node_count"),
        EXPECTED_NODE_COUNT,
        f"{label}.algorithm3_diagonalized_node_count",
    )
    _require(
        algorithms.get("algorithm3_eigenvalue_spectra_retained") is False,
        f"{label} retained production spectra",
    )
    _exact_integer_list(
        algorithms.get("algorithm3_push_ledger"),
        [expected_pushes, expected_pushes],
        f"{label}.algorithm3_push_ledger",
    )
    _bounded_nonnegative(
        algorithms.get("algorithm3_replay_relative_error"),
        GLOBAL_REPLAY_TOLERANCE,
        f"{label}.algorithm3_replay",
    )
    _bounded_nonnegative(
        algorithms.get("algorithm3_offdiagonal_ratio"),
        GLOBAL_REPLAY_TOLERANCE,
        f"{label}.algorithm3_offdiagonal",
    )
    _require(
        _number(algorithms.get("canonical_seconds"), f"{label}.canonical_seconds")
        > 0,
        f"{label} Algorithm 1 timing is invalid",
    )
    _require(
        _number(
            algorithms.get("algorithm3_seconds"), f"{label}.algorithm3_seconds"
        )
        > 0,
        f"{label} Algorithm 3 timing is invalid",
    )
    return {
        "unique_nodes": EXPECTED_NODE_COUNT,
        "edge_occurrences": EXPECTED_EDGE_COUNT,
        "physical_sources": EXPECTED_POLICY_SOURCE_COUNT,
        "root_projective_width": EXPECTED_ROOT_WIDTH,
        "retained_explicit_q_nodes": route["bounded_retained_q_candidate_count"],
        "streamed_cp_nodes": route["streamed_cp_candidate_count"],
        "source_replay_relative_error": result["source_replay_relative_error"],
        "algorithm1_replay_relative_error": algorithms[
            "algorithm1_final_projective_replay_relative_error"
        ],
        "algorithm3_replay_relative_error": algorithms[
            "algorithm3_replay_relative_error"
        ],
        "algorithm3_offdiagonal_ratio": algorithms["algorithm3_offdiagonal_ratio"],
    }


def _validate_full_execution_logs(
    *,
    stdout_payload: bytes,
    stderr_payload: bytes,
    sacct_payload: bytes,
    full_payload: bytes,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    label = "r7 full shared-DAG execution record"
    _require(stderr_payload == b"", f"{label} stderr is not empty")
    try:
        stdout = stdout_payload.decode("utf-8")
        sacct = sacct_payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AttestationError(f"{label} is not UTF-8") from error
    _require(stdout_payload.count(full_payload) == 1, f"{label} stdout result binding differs")
    before_payload, after_payload = stdout_payload.split(full_payload)
    try:
        before_lines = before_payload.decode("utf-8").splitlines()
        after_lines = after_payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise AttestationError(f"{label} wrapper checks are not UTF-8") from error
    _require("FAILED" not in stdout, f"{label} stdout contains a failed wrapper check")
    source_map = _mapping(manifest.get("source_sha256"), "r7 manifest source map")
    for relative in source_map:
        _require(
            before_lines.count(f"{relative}: OK") == 1
            and after_lines.count(f"{relative}: OK") == 1,
            f"{label} lacks both wrapper checks for {relative}",
        )
    _require(
        before_lines.count("artifacts/ckpt_linear_rat_vit_s0_v2.pt: OK") == 1
        and after_lines.count("artifacts/ckpt_linear_rat_vit_s0_v2.pt: OK") == 1,
        f"{label} lacks both checkpoint wrapper checks",
    )
    _require(
        after_lines.count(
            f"{R7_FULL_RESULT_SHA256}  "
            "athena/results/full_vla_production_full_r7_rankdef_retained_q.json"
        )
        == 1,
        f"{label} stdout final result digest differs",
    )
    _require(
        not any(
            line.startswith(f"{R7_FULL_RESULT_SHA256}  ") for line in before_lines
        ),
        f"{label} result digest precedes result publication",
    )

    header_line = (
        "JobID|JobName|Partition|State|ExitCode|Elapsed|TotalCPU|AveCPU|"
        "MaxRSS|AveRSS|ReqCPUS|ReqMem|AllocTRES|NodeList|Start|End"
    )
    lines = sacct.splitlines()
    _require(header_line in lines, f"{label} accounting header differs")
    header = header_line.split("|")
    records: dict[str, dict[str, str]] = {}
    for line in lines[lines.index(header_line) + 1 :]:
        if not line.startswith(EXPECTED_JOB_ID):
            continue
        fields = line.split("|")
        _require(len(fields) == len(header), f"{label} has a malformed accounting row")
        record = dict(zip(header, fields))
        job_id = record["JobID"]
        _require(job_id not in records, f"{label} repeats accounting row {job_id}")
        records[job_id] = record
    expected_job_ids = {
        EXPECTED_JOB_ID,
        f"{EXPECTED_JOB_ID}.batch",
        f"{EXPECTED_JOB_ID}.extern",
    }
    _require(set(records) == expected_job_ids, f"{label} accounting rows differ")
    for job_id, record in records.items():
        _require(record["State"] == "COMPLETED", f"{label} {job_id} did not complete")
        _require(record["ExitCode"] == "0:0", f"{label} {job_id} exit code differs")
        _require(record["Elapsed"] == "15:21:49", f"{label} {job_id} elapsed time differs")
    primary = records[EXPECTED_JOB_ID]
    _require(primary["JobName"] == "odt-r7-full-vla", f"{label} job name differs")
    _require(primary["ReqCPUS"] == "16", f"{label} requested CPU count differs")
    _require(primary["ReqMem"] == "600G", f"{label} requested memory differs")
    _require(primary["NodeList"] == "c2-g8-05", f"{label} node differs")
    batch_rss = records[f"{EXPECTED_JOB_ID}.batch"]["MaxRSS"]
    _require(
        batch_rss.endswith("K") and batch_rss[:-1].isdigit() and int(batch_rss[:-1]) > 0,
        f"{label} batch MaxRSS is invalid",
    )
    return {
        "job_id": EXPECTED_JOB_ID,
        "state": primary["State"],
        "exit_code": primary["ExitCode"],
        "elapsed": primary["Elapsed"],
        "batch_max_rss_kib": int(batch_rss[:-1]),
        "stderr_empty": True,
        "source_wrapper_checks_before_and_after": len(source_map),
        "checkpoint_wrapper_checks_before_and_after": True,
    }


def _validate_oracle_execution_logs(
    *,
    stdout_payload: bytes,
    stderr_payload: bytes,
    oracle_payload: bytes,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    label = "r7 bounded clone-oracle execution record"
    _require(stderr_payload == b"", f"{label} stderr is not empty")
    try:
        stdout = stdout_payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AttestationError(f"{label} stdout is not UTF-8") from error
    _require(
        stdout_payload.count(oracle_payload) == 1,
        f"{label} stdout oracle-result binding differs",
    )
    before_payload, after_payload = stdout_payload.split(oracle_payload)
    try:
        before_lines = before_payload.decode("utf-8").splitlines()
        after_lines = after_payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise AttestationError(f"{label} wrapper checks are not UTF-8") from error
    _require("FAILED" not in stdout, f"{label} stdout contains a failed wrapper check")
    _require(
        any("16 passed in 515.62s (0:08:35)" in line for line in before_lines),
        f"{label} lacks the pinned passing pytest summary",
    )
    source_map = _mapping(manifest.get("source_sha256"), "r7 manifest source map")
    for relative in source_map:
        _require(
            before_lines.count(f"{relative}: OK") == 1
            and after_lines.count(f"{relative}: OK") == 1,
            f"{label} lacks both wrapper checks for {relative}",
        )
    _require(
        after_lines.count(
            f"{R7_ORACLE_JSON_SHA256}  "
            "athena/results/direct_odt_clone_oracle_r7_rankdef_retained_q.json"
        )
        == 1,
        f"{label} stdout oracle JSON digest differs",
    )
    _require(
        after_lines.count(
            f"{R7_ORACLE_JUNIT_SHA256}  "
            "athena/results/direct_odt_clone_oracle_r7_rankdef_retained_q.junit.xml"
        )
        == 1,
        f"{label} stdout oracle JUnit digest differs",
    )
    _require(
        not any(
            line.startswith(f"{R7_ORACLE_JSON_SHA256}  ")
            or line.startswith(f"{R7_ORACLE_JUNIT_SHA256}  ")
            for line in before_lines
        ),
        f"{label} artifact digest precedes result publication",
    )
    return {
        "pytest_passed": 16,
        "pytest_failed": 0,
        "stderr_empty": True,
        "source_wrapper_checks_before_and_after": len(source_map),
    }


def _validate_certification_scope(value: object) -> dict[str, Any]:
    scope = _mapping(value, "composite certification scope")
    _require_exact_fields(
        scope,
        EXPECTED_CERTIFICATION_SCOPE,
        "composite certification scope",
    )
    _require(
        _same_json_value_and_type(scope, EXPECTED_CERTIFICATION_SCOPE),
        "composite certification scope differs or omits a required caveat",
    )
    return scope


def _reverify(
    *,
    paths_and_hashes: Sequence[tuple[Path, str, str]],
    manifest_path: Path,
    manifest: Mapping[str, Any],
    checkpoint: Path,
) -> None:
    for path, expected, label in paths_and_hashes:
        _require(
            _stable_file_sha256(path, label) == expected,
            f"{label} changed during attestation",
        )
    _require(
        _load_manifest(manifest_path, R7_MANIFEST_SHA256, "r7 manifest")
        == manifest,
        "r7 source closure changed during attestation",
    )
    _require(
        _stable_file_sha256(checkpoint, "checkpoint") == CHECKPOINT_SHA256,
        "checkpoint changed during attestation",
    )


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--full-json", type=Path, required=True)
    parser.add_argument("--oracle-json", type=Path, required=True)
    parser.add_argument("--oracle-junit", type=Path, required=True)
    parser.add_argument("--full-stdout", type=Path, required=True)
    parser.add_argument("--full-stderr", type=Path, required=True)
    parser.add_argument("--full-sacct", type=Path, required=True)
    parser.add_argument("--oracle-stdout", type=Path, required=True)
    parser.add_argument("--oracle-stderr", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_paths = (
        args.full_json,
        args.oracle_json,
        args.oracle_junit,
        args.full_stdout,
        args.full_stderr,
        args.full_sacct,
        args.oracle_stdout,
        args.oracle_stderr,
        args.manifest,
        args.checkpoint,
    )
    resolved_inputs = [path.resolve() for path in input_paths]
    _require(
        len(resolved_inputs) == len(set(resolved_inputs)),
        "attestation inputs must be distinct paths",
    )
    _require(
        args.output.resolve() not in set(resolved_inputs),
        "output collides with an input",
    )
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")

    manifest = _load_manifest(args.manifest, R7_MANIFEST_SHA256, "r7 manifest")
    full_payload, full_sha256 = _artifact_bytes(
        args.full_json, R7_FULL_RESULT_SHA256, "r7 full result"
    )
    oracle_payload, oracle_sha256 = _artifact_bytes(
        args.oracle_json, R7_ORACLE_JSON_SHA256, "r7 oracle JSON"
    )
    junit_payload, junit_sha256 = _artifact_bytes(
        args.oracle_junit, R7_ORACLE_JUNIT_SHA256, "r7 oracle JUnit"
    )
    stdout_payload, stdout_sha256 = _artifact_bytes(
        args.full_stdout, R7_FULL_STDOUT_SHA256, "r7 full stdout"
    )
    stderr_payload, stderr_sha256 = _artifact_bytes(
        args.full_stderr, R7_FULL_STDERR_SHA256, "r7 full stderr"
    )
    sacct_payload, sacct_sha256 = _artifact_bytes(
        args.full_sacct, R7_FULL_SACCT_SHA256, "r7 full Slurm accounting"
    )
    oracle_stdout_payload, oracle_stdout_sha256 = _artifact_bytes(
        args.oracle_stdout, R7_ORACLE_STDOUT_SHA256, "r7 oracle stdout"
    )
    oracle_stderr_payload, oracle_stderr_sha256 = _artifact_bytes(
        args.oracle_stderr, R7_ORACLE_STDERR_SHA256, "r7 oracle stderr"
    )

    checkpoint_sha256 = _stable_file_sha256(args.checkpoint, "checkpoint")
    _require(checkpoint_sha256 == CHECKPOINT_SHA256, "checkpoint SHA-256 differs")
    checkpoint_mode = args.checkpoint.stat(follow_symlinks=False).st_mode
    _require(
        checkpoint_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0,
        "checkpoint must be read-only",
    )
    _require(
        args.checkpoint.name == "ckpt_linear_rat_vit_s0_v2.pt",
        "checkpoint basename differs",
    )

    full = _strict_json(full_payload, "r7 full result")
    oracle = _strict_json(oracle_payload, "r7 oracle JSON")
    full_summary = _validate_full_result(full, manifest)
    execution_summary = _validate_full_execution_logs(
        stdout_payload=stdout_payload,
        stderr_payload=stderr_payload,
        sacct_payload=sacct_payload,
        full_payload=full_payload,
        manifest=manifest,
    )
    junit = _validate_oracle_artifact(
        oracle,
        junit_payload=junit_payload,
        junit_sha256=junit_sha256,
        manifest=manifest,
    )
    oracle_execution_summary = _validate_oracle_execution_logs(
        stdout_payload=oracle_stdout_payload,
        stderr_payload=oracle_stderr_payload,
        oracle_payload=oracle_payload,
        manifest=manifest,
    )

    attester_path = Path(__file__).resolve()
    attester_sha256 = _stable_file_sha256(attester_path, "attester")
    attester_test_path = attester_path.parent.parent / ATTESTER_TEST_PATH
    attester_test_sha256 = _stable_file_sha256(
        attester_test_path, "focused attester test source"
    )
    _require(
        attester_test_sha256 == ATTESTER_TEST_SHA256,
        "focused attester test source SHA-256 differs",
    )
    gates = {
        "r7_full_artifact_hash_pinned": True,
        "r7_full_execution_stdout_stderr_and_sacct_pinned": True,
        "r7_slurm_job_completed_with_zero_exit": True,
        "r7_wrapper_source_and_checkpoint_postchecks_passed": True,
        "r7_oracle_json_and_junit_hashes_pinned": True,
        "r7_oracle_stdout_stderr_and_wrapper_postchecks_pinned": True,
        "strict_json_and_junit_parsing": True,
        "r7_source_closure_verified": True,
        "production_core_identity_joined": True,
        "independent_reference_identity_joined": True,
        "checkpoint_execution_bound": True,
        "full_shared_dag_algorithms_1_to_3_verified": True,
        "rank_deficient_retained_q_route_inventory_verified": True,
        "bounded_independent_clone_oracle_16_passed": True,
        "direct_only_static_and_runtime_execution_verified": True,
        "claim_boundary_preserved": True,
        "certification_scope_machine_readable_and_caveated": True,
        "focused_attester_test_source_hash_pinned": True,
    }
    composite_passed = all(gates.values())
    certification_scope = dict(EXPECTED_CERTIFICATION_SCOPE)
    _validate_certification_scope(certification_scope)
    result = {
        "schema": "xvla-direct-odt-composite-attestation-v2-r7",
        "composite_attestation_passed": composite_passed,
        "composite_canonical_odt_evidence_attested": composite_passed,
        "canonical_odt_certified": composite_passed,
        "certification_scope": certification_scope,
        "claim": COMPOSITE_CLAIM,
        "literal_full_policy_clone_expansion_performed": False,
        "arbitrary_size_or_rank_theorem_claimed": False,
        "compression_claimed": False,
        "product_head_checkpoint_decomposed": False,
        "full_policy_action_head": "linear",
        "singleton_embodiment_fixed": True,
        "checkpoint_binding_scope": (
            "The checkpoint is loaded and bound by the full shared-DAG run. The "
            "bounded clone suite is a checkpoint-independent algorithm oracle "
            "bound to that run through the identical frozen r7 source closure."
        ),
        "gates": gates,
        "identities": {
            "manifest_sha256": R7_MANIFEST_SHA256,
            "production_core_sha256": PRODUCTION_CORE_SHA256,
            "independent_reference_sha256": INDEPENDENT_REFERENCE_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "full_result_sha256": full_sha256,
            "oracle_json_sha256": oracle_sha256,
            "oracle_junit_sha256": junit_sha256,
            "full_stdout_sha256": stdout_sha256,
            "full_stderr_sha256": stderr_sha256,
            "full_sacct_sha256": sacct_sha256,
            "oracle_stdout_sha256": oracle_stdout_sha256,
            "oracle_stderr_sha256": oracle_stderr_sha256,
            "attester_sha256": attester_sha256,
            "attester_test_sha256": attester_test_sha256,
        },
        "source_manifest": {
            "input_path": manifest["path"],
            "execution_path": R7_MANIFEST_PATH,
            "root": manifest["root"],
            "sha256": manifest["sha256"],
            "source_count": manifest["source_count"],
        },
        "verifier": {
            "attester_path": attester_path.as_posix(),
            "attester_sha256": attester_sha256,
            "focused_test_path": attester_test_path.as_posix(),
            "focused_test_sha256": attester_test_sha256,
            "identity_trust_boundary": (
                "These verifier digests are self-reported and require an external "
                "freeze or review record as the trust anchor."
            ),
        },
        "full_shared_dag": {
            "path": args.full_json.resolve().as_posix(),
            "sha256": full_sha256,
            **full_summary,
        },
        "full_execution": {
            "stdout_path": args.full_stdout.resolve().as_posix(),
            "stdout_sha256": stdout_sha256,
            "stderr_path": args.full_stderr.resolve().as_posix(),
            "stderr_sha256": stderr_sha256,
            "sacct_path": args.full_sacct.resolve().as_posix(),
            "sacct_sha256": sacct_sha256,
            **execution_summary,
        },
        "bounded_oracle_16_test": {
            "json_path": args.oracle_json.resolve().as_posix(),
            "json_sha256": oracle_sha256,
            "junit_path": args.oracle_junit.resolve().as_posix(),
            "execution_junit_path": R7_ORACLE_JUNIT_PATH,
            "junit_sha256": junit_sha256,
            "stdout_path": args.oracle_stdout.resolve().as_posix(),
            "stdout_sha256": oracle_stdout_sha256,
            "stderr_path": args.oracle_stderr.resolve().as_posix(),
            "stderr_sha256": oracle_stderr_sha256,
            "execution": oracle_execution_summary,
            **junit,
        },
    }
    _require(
        result["composite_attestation_passed"] is True,
        "composite gates did not pass",
    )
    _require(
        result["canonical_odt_certified"] is True,
        "canonical ODT was not certified inside the explicit composite scope",
    )
    scope = _validate_certification_scope(result["certification_scope"])
    _require(
        result["literal_full_policy_clone_expansion_performed"]
        is scope["literal_full_policy_clone_expansion_performed"]
        is False,
        "literal full-policy clone caveat differs",
    )
    _require(
        result["product_head_checkpoint_decomposed"]
        is scope["product_routing_head_decomposed"]
        is False,
        "ProductRoutingHead caveat differs",
    )
    _require(
        result["arbitrary_size_or_rank_theorem_claimed"]
        is scope["arbitrary_size_or_rank_theorem_claimed"]
        is False,
        "arbitrary-size theorem caveat differs",
    )
    _require(
        result["compression_claimed"] is scope["compression_claimed"] is False,
        "compression caveat differs",
    )

    paths_and_hashes = (
        (args.full_json, R7_FULL_RESULT_SHA256, "r7 full result"),
        (args.oracle_json, R7_ORACLE_JSON_SHA256, "r7 oracle JSON"),
        (args.oracle_junit, R7_ORACLE_JUNIT_SHA256, "r7 oracle JUnit"),
        (args.full_stdout, R7_FULL_STDOUT_SHA256, "r7 full stdout"),
        (args.full_stderr, R7_FULL_STDERR_SHA256, "r7 full stderr"),
        (args.full_sacct, R7_FULL_SACCT_SHA256, "r7 full Slurm accounting"),
        (args.oracle_stdout, R7_ORACLE_STDOUT_SHA256, "r7 oracle stdout"),
        (args.oracle_stderr, R7_ORACLE_STDERR_SHA256, "r7 oracle stderr"),
        (attester_path, attester_sha256, "attester"),
        (attester_test_path, ATTESTER_TEST_SHA256, "focused attester test source"),
    )
    _reverify(
        paths_and_hashes=paths_and_hashes,
        manifest_path=args.manifest,
        manifest=manifest,
        checkpoint=args.checkpoint,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    _require(
        not temporary.exists(),
        f"refusing to overwrite temporary output {temporary}",
    )
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    encoded_sha256 = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    published = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o444)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _require(
            _stable_file_sha256(temporary, "staged attestation") == encoded_sha256,
            "staged attestation bytes differ",
        )
        os.link(temporary, args.output, follow_symlinks=False)
        published = True
        temporary.unlink()
        _require(
            _stable_file_sha256(args.output, "published attestation")
            == encoded_sha256,
            "published attestation bytes differ",
        )
        _reverify(
            paths_and_hashes=paths_and_hashes,
            manifest_path=args.manifest,
            manifest=manifest,
            checkpoint=args.checkpoint,
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        if published and args.output.exists():
            rejected = args.output.with_name(
                f"{args.output.name}.rejected-artifact-integrity-{time.time_ns()}"
            )
            args.output.replace(rejected)
        raise
    print(encoded, end="")
    print(f"published_attestation_sha256={encoded_sha256}", file=sys.stderr)


if __name__ == "__main__":
    main()
