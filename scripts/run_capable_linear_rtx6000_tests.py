#!/usr/bin/env python3
"""Run the isolated RTX 6000 capability and cross-stage unit suite."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STAGE_ROOT = Path("/work/joy/x-vla-capable-linear-rtx6000-v3")
CAPABILITY_MANIFEST = (
    PROJECT_ROOT / "athena/capable_linear_rtx6000_capability_sources.sha256"
)
CROSS_STAGE_MANIFEST = (
    PROJECT_ROOT / "athena/capable_linear_rtx6000_cross_stage_sources.sha256"
)
STAGE_LEDGER = PROJECT_ROOT / "athena/capable_linear_rtx6000_stage.sha256"
OUTPUT = (
    PROJECT_ROOT
    / "athena/results/capable_linear_rtx6000/rtx6000_unit_suite.json"
)
EXPECTED_CAPABILITY_TEST_COUNT = 94
EXPECTED_CROSS_STAGE_TEST_COUNT = 132
EXPECTED_UNIT_KEYS = {
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
EXPECTED_JUNIT_KEYS = {"tests", "failures", "errors", "skipped", "test_names"}
EXPECTED_GATE_KEYS = {
    "process_exit_zero",
    "exact_test_count",
    "no_failures",
    "no_skips",
    "unique_test_names",
    "hardware_and_join_regressions_present",
}
EXPECTED_TEST_NAME_SHA256 = (
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


def _safe_member(relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or not value.parts or any(
        part in {"", ".", ".."} for part in value.parts
    ):
        raise RuntimeError(f"unsafe authenticated member {relative!r}")
    cursor = PROJECT_ROOT
    for part in value.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(f"authenticated member traverses a link: {relative}")
    resolved = cursor.resolve(strict=True)
    if resolved == PROJECT_ROOT or PROJECT_ROOT not in resolved.parents or not resolved.is_file():
        raise RuntimeError(f"authenticated member escapes or is nonphysical: {relative}")
    if resolved.relative_to(PROJECT_ROOT).as_posix() != relative:
        raise RuntimeError(f"authenticated member is not canonical: {relative}")
    return resolved


def _read_manifest(path: Path, label: str) -> dict[str, str]:
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
        if _sha256(_safe_member(relative)) != digest:
            raise RuntimeError(f"{label} byte mismatch: {relative}")
        result[relative] = digest
    return result


def _preimport() -> dict[str, Any]:
    if PROJECT_ROOT != EXPECTED_STAGE_ROOT:
        raise RuntimeError(f"unit suite must run from {EXPECTED_STAGE_ROOT}")
    capability = _read_manifest(CAPABILITY_MANIFEST, "capability manifest")
    cross = _read_manifest(CROSS_STAGE_MANIFEST, "cross-stage manifest")
    ledger = _read_manifest(STAGE_LEDGER, "stage ledger")
    required = {
        "scripts/run_capable_linear_rtx6000_tests.py",
        "tests/test_capable_linear_rtx6000_capability.py",
        "tests/test_capable_linear_rtx6000_cross_stage.py",
        "athena/capable_linear_rtx6000_capability_sources.sha256",
        "athena/capable_linear_rtx6000_cross_stage_sources.sha256",
    }
    if not required.issubset(ledger):
        raise RuntimeError("stage ledger omits a unit-suite authority")
    for source_map in (capability, cross):
        for relative, digest in source_map.items():
            if ledger.get(relative) != digest:
                raise RuntimeError(f"stage ledger does not bind source {relative}")
    return {"capability": capability, "cross": cross, "ledger": ledger}


_PREIMPORT = _preimport()
sys.path[:] = [str(PROJECT_ROOT)] + [
    entry for entry in sys.path if entry != str(PROJECT_ROOT)
]
sys.dont_write_bytecode = True

from scripts.odt_direct_only_compliance import audit_direct_only_launch


_ENTRYPOINTS = (
    PROJECT_ROOT / "scripts/attest_capable_linear_rtx6000_cross_stage.py",
    PROJECT_ROOT / "scripts/run_capable_linear_rtx6000_capability.py",
    Path(__file__),
    PROJECT_ROOT / "tests/test_capable_linear_rtx6000_cross_stage.py",
    PROJECT_ROOT / "tests/test_capable_linear_rtx6000_capability.py",
)
_STATIC_AUDIT = audit_direct_only_launch(
    PROJECT_ROOT, _ENTRYPOINTS, require_direct_qr=False
)
if _STATIC_AUDIT["source_sha256"] != _PREIMPORT["cross"]:
    raise RuntimeError("unit source closure differs from cross-stage manifest")
for _field in (
    "prohibited_calls_found",
    "prohibited_self_overlap_sites",
    "guarded_dormant_spectral_norm_sites",
    "duplicate_top_level_definition_sites",
):
    if _STATIC_AUDIT[_field]:
        raise RuntimeError(f"unit static audit did not close {_field}")


def _junit_summary(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag.endswith("testsuite") else list(root.findall("testsuite"))
    result: dict[str, Any] = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    result["test_names"] = sorted(
        item.attrib["name"]
        for item in root.iter()
        if item.tag.endswith("testcase") and "name" in item.attrib
    )
    return result


def _assert_sources() -> None:
    if _read_manifest(CAPABILITY_MANIFEST, "capability manifest") != _PREIMPORT["capability"]:
        raise RuntimeError("capability closure changed during unit suite")
    if _read_manifest(CROSS_STAGE_MANIFEST, "cross-stage manifest") != _PREIMPORT["cross"]:
        raise RuntimeError("cross-stage closure changed during unit suite")
    if _read_manifest(STAGE_LEDGER, "stage ledger") != _PREIMPORT["ledger"]:
        raise RuntimeError("stage closure changed during unit suite")


def _publish(payload: Mapping[str, Any]) -> str:
    if os.path.lexists(OUTPUT):
        raise FileExistsError(f"refusing existing unit output {OUTPUT}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.{secrets.token_hex(12)}.tmp")
    linked = False
    source_stat: os.stat_result | None = None
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        source_stat = os.lstat(temporary)
        if not stat.S_ISREG(source_stat.st_mode):
            raise RuntimeError("unit temporary is not a physical file")
        _assert_sources()
        os.link(temporary, OUTPUT, follow_symlinks=False)
        linked = True
        OUTPUT.chmod(0o444)
        descriptor = os.open(OUTPUT.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if _sha256(OUTPUT) != digest or OUTPUT.stat().st_mode & 0o222:
            raise RuntimeError("published unit output differs")
        _assert_sources()
        return digest
    except BaseException:
        if linked and source_stat is not None and os.path.lexists(OUTPUT):
            current = os.lstat(OUTPUT)
            if current.st_dev == source_stat.st_dev and current.st_ino == source_stat.st_ino:
                OUTPUT.unlink()
        raise
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit("RTX 6000 unit suite accepts no arguments")
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    junit_path = OUTPUT.with_name(f".{OUTPUT.name}.{secrets.token_hex(12)}.junit.tmp")
    started = time.perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-c",
            os.devnull,
            "-p",
            "no:cacheprovider",
            "--noconftest",
            "tests/test_capable_linear_rtx6000_capability.py",
            "tests/test_capable_linear_rtx6000_cross_stage.py",
            f"--junitxml={junit_path}",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    print(completed.stdout, end="", flush=True)
    print(completed.stderr, end="", file=sys.stderr, flush=True)
    try:
        junit = _junit_summary(junit_path)
    finally:
        if os.path.lexists(junit_path):
            junit_path.unlink()
    expected = EXPECTED_CAPABILITY_TEST_COUNT + EXPECTED_CROSS_STAGE_TEST_COUNT
    names = junit.get("test_names", [])
    gates = {
        "process_exit_zero": completed.returncode == 0,
        "exact_test_count": junit.get("tests") == expected,
        "no_failures": junit.get("failures") == 0 and junit.get("errors") == 0,
        "no_skips": junit.get("skipped") == 0,
        "unique_test_names": len(names) == expected and len(set(names)) == len(names),
        "hardware_and_join_regressions_present": {
            "test_environment_gate_rejects_gpu_drift",
            "test_environment_gate_rejects_node_drift",
            "test_torch_version_subclass_is_normalized_to_literal_string",
            "test_environment_gate_rejects_equal_string_subclasses[gpu]",
            "test_environment_record_rejects_non_json_and_scalar_type_aliases",
            "test_transition_normalizes_numpy_bool_to_literal_bool",
            "test_transition_rejects_non_boolean_done_aliases[integer-zero]",
            "test_cross_stage_join_fails_each_scientific_gate[checkpoint]",
            "test_cross_stage_join_requires_distinct_authorities",
            "test_recursive_authority_comparison_is_type_exact",
            (
                "test_full_scientific_record_rejects_nested_mutation"
                "[route_bool_alias-route_inventory_exact]"
            ),
            (
                "test_full_scientific_record_rejects_nested_mutation"
                "[head_exponent_low-algorithm1_provenance]"
            ),
            (
                "test_full_scientific_record_rejects_nested_mutation"
                "[canonical_max_bond_low-canonical_shape_reconciled]"
            ),
            (
                "test_each_capability_artifact_kind_rejects_boolean_seed_alias"
                "[aggregate]"
            ),
        }.issubset(set(names)),
    }
    payload = {
        "schema": "xvla_capable_linear_rtx6000_unit_suite_v1",
        "test_counts": {
            "capability": EXPECTED_CAPABILITY_TEST_COUNT,
            "cross_stage": EXPECTED_CROSS_STAGE_TEST_COUNT,
            "total": expected,
        },
        "junit": junit,
        "elapsed_seconds": time.perf_counter() - started,
        "capability_source_manifest_sha256": _sha256(CAPABILITY_MANIFEST),
        "cross_stage_source_manifest_sha256": _sha256(CROSS_STAGE_MANIFEST),
        "stage_ledger_sha256": _sha256(STAGE_LEDGER),
        "static_audit": _STATIC_AUDIT,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    if (
        set(payload) != EXPECTED_UNIT_KEYS
        or set(payload["junit"]) != EXPECTED_JUNIT_KEYS
        or set(payload["gates"]) != EXPECTED_GATE_KEYS
        or _canonical_sha256(payload["junit"]["test_names"])
        != EXPECTED_TEST_NAME_SHA256
    ):
        raise RuntimeError("RTX 6000 unit result schema or test inventory differs")
    digest = _publish(payload)
    print(json.dumps({"output": OUTPUT.as_posix(), "sha256": digest}, sort_keys=True))
    if not payload["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
