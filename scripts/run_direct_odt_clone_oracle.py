#!/usr/bin/env python3
"""Run the bounded independent direct-ODT oracle as an authenticated artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.dont_write_bytecode = True

EXPECTED_TEST_NAMES = frozenset(
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(path: Path) -> dict[str, object]:
    expected: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(f"malformed source manifest line {line_number}")
        digest, relative = fields
        relative = relative.lstrip("*")
        source = (PROJECT_ROOT / relative).resolve()
        if source == PROJECT_ROOT or PROJECT_ROOT not in source.parents:
            raise RuntimeError("source manifest path escapes the project root")
        normalized = source.relative_to(PROJECT_ROOT).as_posix()
        if normalized in expected:
            raise RuntimeError("source manifest contains a duplicate path")
        expected[normalized] = digest
    runner = Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix()
    if runner not in expected:
        raise RuntimeError("source manifest does not authenticate the oracle runner")
    actual: dict[str, str] = {}
    for relative, expected_digest in sorted(expected.items()):
        source = PROJECT_ROOT / relative
        if not source.is_file():
            raise RuntimeError(f"frozen source is missing: {relative}")
        digest = _sha256(source)
        if digest != expected_digest:
            raise RuntimeError(f"frozen source hash changed: {relative}")
        actual[relative] = digest
    return {
        "path": path.resolve().as_posix(),
        "sha256": _sha256(path),
        "source_count": len(actual),
        "source_sha256": actual,
    }


def _assert_manifest_snapshot(
    path: Path, expected: dict[str, object]
) -> dict[str, object]:
    actual = _verify_manifest(path)
    if actual != expected:
        raise RuntimeError("source manifest or authenticated source bytes changed")
    return actual


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _junit_summary(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag.endswith("testsuite") else list(root.findall("testsuite"))
    result: dict[str, object] = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    result["test_names"] = sorted(
        item.attrib["name"]
        for item in root.iter()
        if item.tag.endswith("testcase") and "name" in item.attrib
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--junit-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.junit_output.exists():
        raise FileExistsError("refusing to overwrite an oracle artifact")
    if args.output.resolve() == args.junit_output.resolve():
        raise ValueError("JSON and JUnit outputs must be distinct paths")

    manifest_before = _verify_manifest(args.source_manifest)
    from scripts.odt_direct_only_compliance import (
        STREAMED_CLONE_ORACLE_RUNTIME_CALLS,
        assert_direct_only_runtime_guard,
        audit_direct_only_launch,
        canonical_direct_only_entrypoints,
        direct_only_runtime_report,
        install_direct_only_runtime_guard,
    )

    static_audit = audit_direct_only_launch(
        PROJECT_ROOT,
        canonical_direct_only_entrypoints(PROJECT_ROOT, Path(__file__)),
    )
    if manifest_before["source_sha256"] != static_audit["source_sha256"]:
        raise RuntimeError("manifest differs from the authoritative source closure")
    runtime_at_import = install_direct_only_runtime_guard()

    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    import pytest

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.junit_output.parent.mkdir(parents=True, exist_ok=True)
    test_path = PROJECT_ROOT / "tests/test_direct_odt_clone_reference.py"
    temporary_junit = args.junit_output.with_name(
        f".{args.junit_output.name}.{os.getpid()}.tmp"
    )
    if temporary_junit.exists():
        raise FileExistsError("refusing to overwrite a temporary JUnit artifact")
    started = time.perf_counter()
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
                f"--junitxml={temporary_junit.as_posix()}",
            ]
        )
    )
    elapsed = time.perf_counter() - started
    junit = _junit_summary(temporary_junit)
    runtime_final = assert_direct_only_runtime_guard(
        direct_only_runtime_report(),
        exact_allowed_calls=STREAMED_CLONE_ORACLE_RUNTIME_CALLS,
    )
    passed = bool(
        exit_code == 0
        and junit["tests"] == len(EXPECTED_TEST_NAMES)
        and junit["failures"] == 0
        and junit["errors"] == 0
        and junit["skipped"] == 0
        and frozenset(junit["test_names"]) == EXPECTED_TEST_NAMES
        and runtime_final["prohibited_attempt_count"] == 0
    )
    manifest_before_result = _assert_manifest_snapshot(
        args.source_manifest, manifest_before
    )
    result = {
        "schema": "xvla-bounded-independent-direct-odt-clone-oracle-v1",
        "claim_boundary": (
            "Bounded no-memo clone-unfolded Algorithms 1-3 validation. This is "
            "not a whole-policy clone expansion and must be hash-joined with a "
            "full shared-DAG result before any formal whole-policy claim."
        ),
        "test_path": test_path.relative_to(PROJECT_ROOT).as_posix(),
        "pytest_exit_code": exit_code,
        "junit": {
            "path": args.junit_output.resolve().as_posix(),
            "sha256": _sha256(temporary_junit),
            **junit,
        },
        "elapsed_seconds": elapsed,
        "peak_rss_mb": _peak_rss_mb(),
        "bounded_independent_clone_oracle_passed": passed,
        "canonical_odt_certified": False,
        "certification_status": (
            "bounded_independent_clone_oracle_passed_awaiting_composite_attestation"
            if passed
            else "failed"
        ),
        "static_audit": static_audit,
        "runtime_direct_only_guard_at_import": runtime_at_import,
        "runtime_direct_only_guard_final": runtime_final,
        "pytest_plugin_autoload_disabled": True,
        "source_manifest_before_test": manifest_before,
        "source_manifest_before_result": manifest_before_result,
    }
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    temporary.write_text(encoded)
    expected_json_sha256 = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    expected_junit_sha256 = result["junit"]["sha256"]
    try:
        _assert_manifest_snapshot(args.source_manifest, manifest_before)
        temporary_junit.replace(args.junit_output)
        temporary.replace(args.output)
        if _sha256(args.output) != expected_json_sha256:
            raise RuntimeError("published oracle JSON differs from its staged bytes")
        if _sha256(args.junit_output) != expected_junit_sha256:
            raise RuntimeError("published JUnit differs from its staged bytes")
        _assert_manifest_snapshot(args.source_manifest, manifest_before)
    except Exception:
        suffix = f"rejected-artifact-integrity-{time.time_ns()}"
        if args.output.exists():
            args.output.replace(
                args.output.with_name(f"{args.output.name}.{suffix}")
            )
        if args.junit_output.exists():
            args.junit_output.replace(
                args.junit_output.with_name(f"{args.junit_output.name}.{suffix}")
            )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
