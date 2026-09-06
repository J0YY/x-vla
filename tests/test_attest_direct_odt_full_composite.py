from __future__ import annotations

import json
import math
import shutil
import stat
import sys
from pathlib import Path

import pytest

from scripts import attest_direct_odt_full_composite as composite


ROOT = Path(__file__).resolve().parents[1]
FULL_PATH = (
    ROOT
    / "athena/results/full_vla_production_full_r7_rankdef_retained_q.json"
)
ORACLE_PATH = (
    ROOT / "athena/results/direct_odt_clone_oracle_r7_rankdef_retained_q.json"
)
JUNIT_PATH = (
    ROOT
    / "athena/results/direct_odt_clone_oracle_r7_rankdef_retained_q.junit.xml"
)
STDOUT_PATH = ROOT / "athena/logs/odt-r7-full-vla-835191.out"
STDERR_PATH = ROOT / "athena/logs/odt-r7-full-vla-835191.err"
SACCT_PATH = ROOT / "athena/logs/odt-r7-full-vla-835191.sacct.txt"
ORACLE_STDOUT_PATH = ROOT / "athena/logs/odt-r7-oracle16-835189.out"
ORACLE_STDERR_PATH = ROOT / "athena/logs/odt-r7-oracle16-835189.err"
CHECKPOINT_PATH = ROOT / "tmp/athena_xvla_artifacts/ckpt_linear_rat_vit_s0_v2.pt"
LOCAL_FROZEN_ROOT = Path(
    "/private/tmp/x-vla-global-odt-20260904-r7-rankdef-retained-q-c715f739"
)
LOCAL_FROZEN_MANIFEST = (
    LOCAL_FROZEN_ROOT / "athena/implicit_sparse_projective_odt_vla_sources.sha256"
)


def _artifacts() -> tuple[dict, dict, bytes, dict]:
    full_payload, _ = composite._artifact_bytes(
        FULL_PATH, composite.R7_FULL_RESULT_SHA256, "test full result"
    )
    oracle_payload, _ = composite._artifact_bytes(
        ORACLE_PATH, composite.R7_ORACLE_JSON_SHA256, "test oracle JSON"
    )
    junit_payload, _ = composite._artifact_bytes(
        JUNIT_PATH, composite.R7_ORACLE_JUNIT_SHA256, "test oracle JUnit"
    )
    full = composite._strict_json(full_payload, "test full result")
    oracle = composite._strict_json(oracle_payload, "test oracle JSON")
    embedded = full["immutable_source_manifest"]
    manifest = {
        "path": embedded["path"],
        "root": composite.R7_FROZEN_ROOT,
        "sha256": embedded["sha256"],
        "source_count": embedded["source_count"],
        "source_sha256": embedded["source_sha256"],
    }
    composite._validate_manifest_identities(manifest)
    return full, oracle, junit_payload, manifest


def test_exact_r7_hash_gate_and_test_sets_are_pinned():
    assert composite.R7_MANIFEST_SHA256 == (
        "c715f739eb5d6d4a7d44bd12151ab9bab518a620b1ce55928919bd3385aa3884"
    )
    assert composite.R7_FULL_RESULT_SHA256 == (
        "f90a2c8c4842206aface13801770ea07fd6a4b3f12c023b0faeb984177d8b374"
    )
    assert composite.R7_ORACLE_JSON_SHA256 == (
        "98e49e57681ba5b79ff321e316a325ea7cf864d8752e422f99c6fc7b7f2ebe10"
    )
    assert composite.R7_ORACLE_JUNIT_SHA256 == (
        "2ed69357e670ea2a2f506e29e4c22b310b70f749bd78503393b910af99a06277"
    )
    assert composite.R7_FULL_STDOUT_SHA256 == (
        "7f8cb96c5aec84aaca041ca8f115e942d61e61b57be0445bc7f1df11f738da63"
    )
    assert composite.R7_FULL_STDERR_SHA256 == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert composite.R7_FULL_SACCT_SHA256 == (
        "3de30ed8be4157fa392d30a11ff3920e1bb24ee48ccb27d5ff3b24cc52ecbcab"
    )
    assert composite.R7_ORACLE_STDOUT_SHA256 == (
        "e3426ce4b7b32f37f180c383d359d6821bde2f683470eb46ffb3e26a6e72f5e3"
    )
    assert composite.R7_ORACLE_STDERR_SHA256 == composite.R7_FULL_STDERR_SHA256
    assert len(composite.EXPECTED_FULL_GATE_NAMES) == 30
    assert len(composite.R7_TEST_NAMES) == 16
    assert "algorithm1_exact_route_prediction_is_internally_consistent" in (
        composite.EXPECTED_FULL_GATE_NAMES
    )
    assert (
        "test_bounded_rank_deficient_385x386_retains_direct_q_and_matches_clone_every_step"
        in composite.R7_TEST_NAMES
    )
    assert "test_over_bound_large_deficient_unfolding_fails_closed" in (
        composite.R7_TEST_NAMES
    )
    assert "test_rank_zero_direct_rq_reconstruction_and_homogeneous_replay" in (
        composite.R7_TEST_NAMES
    )


def test_r7_runtime_guard_surface_and_exact_call_ledgers_are_pinned():
    assert (
        len(composite.EXPECTED_RUNTIME_PATCHED_ENTRYPOINTS)
        == composite.EXPECTED_RUNTIME_PATCHED_ENTRYPOINT_COUNT
        == 87
    )
    assert "numpy.linalg.linalg.svd" in (
        composite.EXPECTED_RUNTIME_PATCHED_ENTRYPOINTS
    )
    assert "torch.Tensor.norm" in composite.EXPECTED_RUNTIME_PATCHED_ENTRYPOINTS
    assert len(composite.FULL_REQUIRED_CALLS) == 4
    assert len(composite.ORACLE_REQUIRED_CALLS) == 6
    assert composite.EXPECTED_FULL_ALLOWED_CALL_COUNT == 8_111_588
    assert composite.EXPECTED_ORACLE_ALLOWED_CALL_COUNT == 8_508


def test_strict_json_rejects_duplicate_and_nonfinite_values():
    with pytest.raises(composite.AttestationError, match="duplicate key"):
        composite._strict_json(b'{"x": 1, "x": 2}', "fixture")
    with pytest.raises(composite.AttestationError, match="non-finite"):
        composite._strict_json(b'{"x": 1e999}', "fixture")
    with pytest.raises(composite.AttestationError, match="non-finite"):
        composite._strict_json(b'{"x": NaN}', "fixture")


def test_junit_parser_requires_consistent_case_level_counters():
    clean = (
        b'<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0">'
        b'<testcase name="a"/><testcase name="b"/></testsuite></testsuites>'
    )
    assert composite._junit_summary(clean, "fixture") == {
        "tests": 2,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "test_names": ["a", "b"],
    }
    dishonest = clean.replace(b'tests="2"', b'tests="3"')
    with pytest.raises(composite.AttestationError, match="counters disagree"):
        composite._junit_summary(dishonest, "fixture")


def test_completed_r7_full_and_oracle_artifacts_validate_exactly():
    full, oracle, junit_payload, manifest = _artifacts()
    full_summary = composite._validate_full_result(full, manifest)
    oracle_summary = composite._validate_oracle_artifact(
        oracle,
        junit_payload=junit_payload,
        junit_sha256=composite.R7_ORACLE_JUNIT_SHA256,
        manifest=manifest,
    )
    assert full_summary["unique_nodes"] == 3_964_463
    assert full_summary["edge_occurrences"] == 7_216_854
    assert full_summary["retained_explicit_q_nodes"] == 3_248_703
    assert full_summary["streamed_cp_nodes"] == 3_785
    assert oracle_summary["tests"] == 16
    assert oracle_summary["failures"] == 0
    assert oracle_summary["errors"] == 0
    assert oracle_summary["skipped"] == 0


def test_completed_r7_execution_logs_bind_wrapper_checks_and_slurm_completion():
    full, _, _, manifest = _artifacts()
    del full
    full_payload, _ = composite._artifact_bytes(
        FULL_PATH, composite.R7_FULL_RESULT_SHA256, "test full result"
    )
    summary = composite._validate_full_execution_logs(
        stdout_payload=STDOUT_PATH.read_bytes(),
        stderr_payload=STDERR_PATH.read_bytes(),
        sacct_payload=SACCT_PATH.read_bytes(),
        full_payload=full_payload,
        manifest=manifest,
    )
    assert summary["job_id"] == "835191"
    assert summary["state"] == "COMPLETED"
    assert summary["exit_code"] == "0:0"
    assert summary["source_wrapper_checks_before_and_after"] == 34


def test_completed_r7_oracle_logs_bind_tests_result_and_wrapper_postchecks():
    oracle_payload = ORACLE_PATH.read_bytes()
    _, _, _, manifest = _artifacts()
    summary = composite._validate_oracle_execution_logs(
        stdout_payload=ORACLE_STDOUT_PATH.read_bytes(),
        stderr_payload=ORACLE_STDERR_PATH.read_bytes(),
        oracle_payload=oracle_payload,
        manifest=manifest,
    )
    assert summary["pytest_passed"] == 16
    assert summary["pytest_failed"] == 0
    assert summary["source_wrapper_checks_before_and_after"] == 34


def test_full_log_validator_rejects_postcheck_relocated_before_result():
    full_payload = FULL_PATH.read_bytes()
    stdout_payload = STDOUT_PATH.read_bytes()
    before, after = stdout_payload.split(full_payload)
    check = b"scripts/__init__.py: OK\n"
    relocated = before + check + full_payload + after.replace(check, b"", 1)
    _, _, _, manifest = _artifacts()
    with pytest.raises(composite.AttestationError, match="lacks both wrapper checks"):
        composite._validate_full_execution_logs(
            stdout_payload=relocated,
            stderr_payload=STDERR_PATH.read_bytes(),
            sacct_payload=SACCT_PATH.read_bytes(),
            full_payload=full_payload,
            manifest=manifest,
        )


def test_oracle_log_validator_rejects_postcheck_relocated_before_result():
    oracle_payload = ORACLE_PATH.read_bytes()
    stdout_payload = ORACLE_STDOUT_PATH.read_bytes()
    before, after = stdout_payload.split(oracle_payload)
    check = b"scripts/__init__.py: OK\n"
    relocated = before + check + oracle_payload + after.replace(check, b"", 1)
    _, _, _, manifest = _artifacts()
    with pytest.raises(composite.AttestationError, match="lacks both wrapper checks"):
        composite._validate_oracle_execution_logs(
            stdout_payload=relocated,
            stderr_payload=ORACLE_STDERR_PATH.read_bytes(),
            oracle_payload=oracle_payload,
            manifest=manifest,
        )


def test_actual_frozen_r7_manifest_hashes_all_34_sources():
    manifest = composite._load_manifest(
        LOCAL_FROZEN_MANIFEST,
        composite.R7_MANIFEST_SHA256,
        "test frozen r7 manifest",
    )
    assert manifest["root"] == LOCAL_FROZEN_ROOT.as_posix()
    assert manifest["source_count"] == 34
    assert manifest["source_sha256"][composite.PRODUCTION_CORE_PATH] == (
        composite.PRODUCTION_CORE_SHA256
    )
    assert manifest["source_sha256"][composite.INDEPENDENT_REFERENCE_PATH] == (
        composite.INDEPENDENT_REFERENCE_SHA256
    )


def test_artifact_hash_pin_rejects_any_byte_drift(tmp_path: Path):
    payload = FULL_PATH.read_bytes() + b"\n"
    drifted = tmp_path / "drifted.json"
    drifted.write_bytes(payload)
    with pytest.raises(composite.AttestationError, match="differs from its r7 pin"):
        composite._artifact_bytes(
            drifted, composite.R7_FULL_RESULT_SHA256, "drifted full result"
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["gates"].pop(
                "algorithm1_exact_route_prediction_is_internally_consistent"
            ),
            "gate set differs",
        ),
        (
            lambda value: value["gates"].__setitem__(
                "algorithm3_projective_replay_and_diagonal", False
            ),
            "false or non-boolean gate",
        ),
        (
            lambda value: value.__setitem__("compile_seconds", math.inf),
            "non-finite",
        ),
        (
            lambda value: value.pop("evaluate_seconds"),
            "unexpected or missing fields",
        ),
        (
            lambda value: value["boundary_algorithms"].pop("algorithm2_seconds"),
            "unexpected or missing fields",
        ),
        (
            lambda value: value["structure"].pop("product_signs"),
            "unexpected or missing fields",
        ),
        (
            lambda value: value.__setitem__("source_replay_relative_error", -1.0),
            "must be nonnegative",
        ),
        (
            lambda value: value["boundary_algorithms"].__setitem__(
                "algorithm1_final_projective_replay_relative_error", -1.0
            ),
            "must be nonnegative",
        ),
        (
            lambda value: value["boundary_algorithms"][
                "algorithm1_direct_q_provenance"
            ].__setitem__("maximum_compact_q_relative_error", -1.0),
            "must be nonnegative",
        ),
        (
            lambda value: value["boundary_algorithms"].__setitem__(
                "algorithm3_offdiagonal_ratio", -1.0
            ),
            "must be nonnegative",
        ),
    ],
)
def test_full_result_fails_closed_on_missing_false_or_nonfinite_metrics(
    mutation, message: str
):
    full, _, _, manifest = _artifacts()
    mutation(full)
    with pytest.raises(composite.AttestationError, match=message):
        composite._validate_full_result(full, manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value[
                "algorithm1_exact_predicted_route_inventory"
            ].pop("streamed_tall_unrepresentable_count"),
            "exact r7 route inventory",
        ),
        (
            lambda value: value[
                "algorithm1_exact_predicted_route_inventory"
            ].__setitem__("bounded_retained_q_total_elements", math.nan),
            "non-finite",
        ),
        (
            lambda value: value["boundary_algorithms"]["canonical_telemetry"].pop(
                "svd_calls"
            ),
            "unexpected or missing fields",
        ),
        (
            lambda value: value["boundary_algorithms"]["canonical_telemetry"].__setitem__(
                "svd_calls", 1
            ),
            "svd_calls is nonzero",
        ),
        (
            lambda value: value["boundary_algorithms"]["canonical_telemetry"].__setitem__(
                "svd_calls", False
            ),
            "must be an integer",
        ),
        (
            lambda value: value[
                "algorithm1_exact_predicted_route_inventory"
            ]["bounded_retained_q_shape_counts"].__setitem__("2x81", True),
            "exact r7 route inventory",
        ),
    ],
)
def test_full_result_fails_closed_on_route_or_prohibited_metric_drift(
    mutation, message: str
):
    full, _, _, manifest = _artifacts()
    mutation(full)
    with pytest.raises(composite.AttestationError, match=message):
        composite._validate_full_result(full, manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["implicit_shape"].__setitem__("dense_clone_nodes", False),
        lambda value: value["boundary_algorithms"].__setitem__(
            "algorithm1_per_step_replays_performed", False
        ),
        lambda value: value["boundary_algorithms"].__setitem__(
            "algorithm1_local_reconstruction_maximum_exponent_delta", False
        ),
        lambda value: value["boundary_algorithms"]["canonical_telemetry"].__setitem__(
            "normal_equation_factorizations", False
        ),
    ],
)
def test_full_result_rejects_booleans_in_integer_ledgers(mutation):
    full, _, _, manifest = _artifacts()
    mutation(full)
    with pytest.raises(
        composite.AttestationError,
        match="must be an integer|implicit shape differs",
    ):
        composite._validate_full_result(full, manifest)


def test_full_result_requires_exact_terminal_provenance_method():
    full, _, _, manifest = _artifacts()
    full["algorithm1_live_progress"]["last_direct_q_provenance_method"] = (
        "retained_explicit_q::svd_fallback"
    )
    with pytest.raises(composite.AttestationError, match="provenance differs"):
        composite._validate_full_result(full, manifest)


def test_full_result_rejects_duplicate_factorization_method_ledger():
    full, _, _, manifest = _artifacts()
    methods = full["boundary_algorithms"]["factorization_methods"]
    methods.append(methods[0])
    with pytest.raises(composite.AttestationError, match="method set differs"):
        composite._validate_full_result(full, manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prohibited_attempt_count", 1, "prohibited attempt"),
        ("prohibited_attempts", ["blocked"], "prohibited ledger"),
        ("allowed_call_count", 8_507, "allowed-call count differs"),
    ],
)
def test_oracle_fails_closed_on_runtime_ledger_drift(
    field: str, value, message: str
):
    _, oracle, junit_payload, manifest = _artifacts()
    oracle["runtime_direct_only_guard_final"][field] = value
    with pytest.raises(composite.AttestationError, match=message):
        composite._validate_oracle_artifact(
            oracle,
            junit_payload=junit_payload,
            junit_sha256=composite.R7_ORACLE_JUNIT_SHA256,
            manifest=manifest,
        )


def test_oracle_fails_closed_on_nonfinite_or_missing_metrics():
    _, oracle, junit_payload, manifest = _artifacts()
    oracle["elapsed_seconds"] = math.nan
    with pytest.raises(composite.AttestationError, match="non-finite"):
        composite._validate_oracle_artifact(
            oracle,
            junit_payload=junit_payload,
            junit_sha256=composite.R7_ORACLE_JUNIT_SHA256,
            manifest=manifest,
        )

    _, oracle, junit_payload, manifest = _artifacts()
    oracle.pop("peak_rss_mb")
    with pytest.raises(composite.AttestationError, match="unexpected or missing fields"):
        composite._validate_oracle_artifact(
            oracle,
            junit_payload=junit_payload,
            junit_sha256=composite.R7_ORACLE_JUNIT_SHA256,
            manifest=manifest,
        )


def test_oracle_rejects_boolean_zero_junit_and_pytest_counters():
    _, oracle, junit_payload, manifest = _artifacts()
    oracle["junit"]["failures"] = False
    with pytest.raises(composite.AttestationError, match="must be an integer"):
        composite._validate_oracle_artifact(
            oracle,
            junit_payload=junit_payload,
            junit_sha256=composite.R7_ORACLE_JUNIT_SHA256,
            manifest=manifest,
        )

    _, oracle, junit_payload, manifest = _artifacts()
    oracle["pytest_exit_code"] = False
    with pytest.raises(composite.AttestationError, match="must be an integer"):
        composite._validate_oracle_artifact(
            oracle,
            junit_payload=junit_payload,
            junit_sha256=composite.R7_ORACLE_JUNIT_SHA256,
            manifest=manifest,
        )

def test_static_and_module_audits_fail_closed_on_prohibited_entries():
    full, _, _, manifest = _artifacts()
    full["transitive_static_audit"]["prohibited_calls_found"] = ["bad.call"]
    with pytest.raises(composite.AttestationError, match="prohibited call"):
        composite._validate_full_result(full, manifest)

    full, _, _, manifest = _artifacts()
    full["direct_call_audit"]["prohibited_calls_found"] = ["bad.call"]
    with pytest.raises(composite.AttestationError, match="direct-call audit failed"):
        composite._validate_full_result(full, manifest)

    full, _, _, manifest = _artifacts()
    full["direct_call_audit"]["observed_calls"][0] = "AlteredCall"
    with pytest.raises(composite.AttestationError, match="observed ledger digest"):
        composite._validate_full_result(full, manifest)


def test_main_attests_real_r7_closure_checkpoint_execution_and_atomic_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    checkpoint = tmp_path / "ckpt_linear_rat_vit_s0_v2.pt"
    shutil.copyfile(CHECKPOINT_PATH, checkpoint)
    checkpoint.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    output = tmp_path / "r7_composite_attestation.json"
    argv = [
        "attest_direct_odt_full_composite.py",
        "--full-json",
        str(FULL_PATH),
        "--oracle-json",
        str(ORACLE_PATH),
        "--oracle-junit",
        str(JUNIT_PATH),
        "--full-stdout",
        str(STDOUT_PATH),
        "--full-stderr",
        str(STDERR_PATH),
        "--full-sacct",
        str(SACCT_PATH),
        "--oracle-stdout",
        str(ORACLE_STDOUT_PATH),
        "--oracle-stderr",
        str(ORACLE_STDERR_PATH),
        "--manifest",
        str(LOCAL_FROZEN_MANIFEST),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    composite.main()
    capsys.readouterr()

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["composite_attestation_passed"] is True
    assert result["canonical_odt_certified"] is True
    assert result["full_execution"]["state"] == "COMPLETED"
    assert result["full_execution"]["exit_code"] == "0:0"
    scope = result["certification_scope"]
    assert scope["action_head"] == "linear"
    assert scope["singleton_embodiment_fixed"] is True
    assert scope["bounded_clone_oracle_test_count"] == 16
    assert scope["literal_full_policy_clone_expansion_performed"] is False
    assert scope["product_routing_head_decomposed"] is False
    assert scope["arbitrary_size_or_rank_theorem_claimed"] is False
    assert scope["compression_claimed"] is False
    assert output.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0

    with pytest.raises(composite.AttestationError, match="refusing to overwrite"):
        composite.main()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("literal_full_policy_clone_expansion_performed"),
        lambda value: value.__setitem__("product_routing_head_decomposed", True),
        lambda value: value.__setitem__("arbitrary_size_or_rank_theorem_claimed", True),
        lambda value: value.__setitem__("bounded_clone_oracle_test_count", 15),
        lambda value: value.__setitem__("action_head", "product"),
        lambda value: value.__setitem__("compression_claimed", True),
    ],
)
def test_certification_scope_fails_closed_on_any_caveat_drift(mutation):
    scope = dict(composite.EXPECTED_CERTIFICATION_SCOPE)
    mutation(scope)
    with pytest.raises(composite.AttestationError, match="certification scope"):
        composite._validate_certification_scope(scope)


def test_claim_preserves_the_bounded_composite_scope():
    assert "not a literal clone expansion of the 4M-node policy" in (
        composite.COMPOSITE_CLAIM
    )
    assert "not an arbitrary-size or arbitrary-rank theorem" in (
        composite.COMPOSITE_CLAIM
    )
    assert "not evidence that a trained ProductRoutingHead" in (
        composite.COMPOSITE_CLAIM
    )
