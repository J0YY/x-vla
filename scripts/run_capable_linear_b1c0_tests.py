#!/usr/bin/env python3
"""Run exact-ODT and simulator-lane tests in isolated Python processes."""

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
EXPECTED_STAGE_ROOT = Path("/work/joy/x-vla-capable-linear-b1c0-odt-v2")
ODT_MANIFEST = PROJECT_ROOT / "athena/capable_linear_direct_odt_sources.sha256"
CAPABILITY_MANIFEST = (
    PROJECT_ROOT / "athena/capable_linear_fresh_capability_sources.sha256"
)
DIRECT_TEST_MANIFEST = PROJECT_ROOT / "athena/direct_odt_truncation_tests_sources.sha256"
STAGE_LEDGER = PROJECT_ROOT / "athena/capable_linear_b1c0_stage.sha256"
DIRECT_OUTPUT = PROJECT_ROOT / "athena/results/capable_linear_b1c0_core_unit.json"
OUTPUT = PROJECT_ROOT / "athena/results/capable_linear_b1c0/unit_suite.json"
EXPECTED_DIRECT_TEST_COUNT = 52
EXPECTED_CAPABILITY_TEST_COUNT = 34
EXPECTED_ATTESTATION_TEST_COUNT = 7


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise RuntimeError(f"authenticated member is not normalized: {relative}")
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
            raise RuntimeError(f"invalid or duplicate {label} line {number}")
        if _sha256(_safe_member(relative)) != digest:
            raise RuntimeError(f"{label} byte mismatch: {relative}")
        result[relative] = digest
    return result


def _preimport() -> dict[str, Any]:
    if PROJECT_ROOT != EXPECTED_STAGE_ROOT:
        raise RuntimeError(f"unit suite must run from {EXPECTED_STAGE_ROOT}")
    odt = _read_manifest(ODT_MANIFEST, "ODT source manifest")
    capability = _read_manifest(CAPABILITY_MANIFEST, "capability source manifest")
    direct_tests = _read_manifest(DIRECT_TEST_MANIFEST, "direct test source manifest")
    ledger = _read_manifest(STAGE_LEDGER, "stage ledger")
    fixed_files = (
        Path(__file__).resolve(),
        ODT_MANIFEST,
        CAPABILITY_MANIFEST,
        DIRECT_TEST_MANIFEST,
        PROJECT_ROOT / "scripts/run_direct_odt_truncation_tests.py",
    )
    for path in fixed_files:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if ledger.get(relative) != _sha256(path):
            raise RuntimeError(f"stage ledger does not bind unit authority {relative}")
    for source_map in (odt, capability, direct_tests):
        for relative, digest in source_map.items():
            if ledger.get(relative) != digest:
                raise RuntimeError(f"stage ledger does not bind source {relative}")
    if _sha256(ODT_MANIFEST) == _sha256(CAPABILITY_MANIFEST):
        raise RuntimeError("ODT and external-simulator source authorities must remain distinct")
    return {
        "odt": odt,
        "capability": capability,
        "direct_tests": direct_tests,
        "ledger": ledger,
    }


_PREIMPORT = _preimport()
sys.dont_write_bytecode = True


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or nonphysical")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not an object")
    return value


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
    if _read_manifest(ODT_MANIFEST, "ODT source manifest") != _PREIMPORT["odt"]:
        raise RuntimeError("ODT closure changed during unit suite")
    if _read_manifest(CAPABILITY_MANIFEST, "capability source manifest") != _PREIMPORT["capability"]:
        raise RuntimeError("capability closure changed during unit suite")
    if _read_manifest(DIRECT_TEST_MANIFEST, "direct test source manifest") != _PREIMPORT["direct_tests"]:
        raise RuntimeError("direct test closure changed during unit suite")
    if _read_manifest(STAGE_LEDGER, "stage ledger") != _PREIMPORT["ledger"]:
        raise RuntimeError("stage closure changed during unit suite")


def _publish(payload: Mapping[str, Any]) -> str:
    if os.path.lexists(OUTPUT):
        raise FileExistsError(f"refusing existing unit output {OUTPUT}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.{secrets.token_hex(12)}.tmp")
    linked = False
    temporary_stat: os.stat_result | None = None
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_stat = os.lstat(temporary)
        if not stat.S_ISREG(temporary_stat.st_mode):
            raise RuntimeError("unit temporary is not a regular file")
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
        if linked and temporary_stat is not None and os.path.lexists(OUTPUT):
            current = os.lstat(OUTPUT)
            if current.st_dev == temporary_stat.st_dev and current.st_ino == temporary_stat.st_ino:
                OUTPUT.unlink()
        raise
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit("unit suite accepts no arguments")
    if os.path.lexists(DIRECT_OUTPUT):
        raise FileExistsError(f"refusing existing direct-test output {DIRECT_OUTPUT}")
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    started = time.perf_counter()
    direct = subprocess.run(
        [
            sys.executable,
            "-B",
            "scripts/run_direct_odt_truncation_tests.py",
            "--output",
            DIRECT_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    print(direct.stdout, end="", flush=True)
    print(direct.stderr, end="", file=sys.stderr, flush=True)
    direct_result = _read_object(DIRECT_OUTPUT, "direct test result")

    capability_junit_path = OUTPUT.with_name(
        f".{OUTPUT.name}.{secrets.token_hex(12)}.capability.junit.tmp"
    )
    capability_tests = "tests/test_capable_linear_fresh_capability.py"
    capability = subprocess.run(
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
            capability_tests,
            f"--junitxml={capability_junit_path}",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    print(capability.stdout, end="", flush=True)
    print(capability.stderr, end="", file=sys.stderr, flush=True)
    try:
        capability_junit = _junit_summary(capability_junit_path)
    finally:
        if os.path.lexists(capability_junit_path):
            capability_junit_path.unlink()
    attestation_junit_path = OUTPUT.with_name(
        f".{OUTPUT.name}.{secrets.token_hex(12)}.attestation.junit.tmp"
    )
    attestation_tests = "tests/test_capable_linear_artifact_integrity.py"
    attestation = subprocess.run(
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
            attestation_tests,
            f"--junitxml={attestation_junit_path}",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    print(attestation.stdout, end="", flush=True)
    print(attestation.stderr, end="", file=sys.stderr, flush=True)
    try:
        attestation_junit = _junit_summary(attestation_junit_path)
    finally:
        if os.path.lexists(attestation_junit_path):
            attestation_junit_path.unlink()
    direct_junit = direct_result.get("junit", {})
    all_names = list(direct_junit.get("test_names", [])) + list(
        capability_junit["test_names"]
    ) + list(attestation_junit["test_names"])
    gates = {
        "direct_process_exit_zero": direct.returncode == 0,
        "direct_self_gate": direct_result.get("all_gates_pass") is True,
        "direct_exact_count": direct_junit.get("tests") == EXPECTED_DIRECT_TEST_COUNT,
        "direct_no_skips": direct_junit.get("skipped") == 0,
        "capability_process_exit_zero": capability.returncode == 0,
        "capability_exact_count": capability_junit["tests"]
        == EXPECTED_CAPABILITY_TEST_COUNT,
        "capability_no_failures": capability_junit["failures"] == 0
        and capability_junit["errors"] == 0,
        "capability_no_skips": capability_junit["skipped"] == 0,
        "attestation_process_exit_zero": attestation.returncode == 0,
        "attestation_exact_count": attestation_junit["tests"]
        == EXPECTED_ATTESTATION_TEST_COUNT,
        "attestation_no_failures": attestation_junit["failures"] == 0
        and attestation_junit["errors"] == 0,
        "attestation_no_skips": attestation_junit["skipped"] == 0,
        "combined_unique_names": len(all_names)
        == EXPECTED_DIRECT_TEST_COUNT
        + EXPECTED_CAPABILITY_TEST_COUNT
        + EXPECTED_ATTESTATION_TEST_COUNT
        and len(set(all_names)) == len(all_names),
        "norm_inventory_and_dead_site_tests_present": {
            "test_checkpoint_norm_inventory_allows_only_the_exact_dead_vision_site",
            "test_inactive_norm_sentinel_raises_on_dead_site_execution",
        }.issubset(set(all_names)),
        "distinct_odt_and_capability_manifests": _sha256(ODT_MANIFEST)
        != _sha256(CAPABILITY_MANIFEST),
        "process_isolation": True,
    }
    payload = {
        "schema": "xvla_capable_linear_b1c0_combined_unit_suite_v1",
        "process_boundary": {
            "direct_odt_guarded_process": True,
            "fresh_capability_test_process": True,
            "attestation_integrity_test_process": True,
            "shared_process": False,
        },
        "direct_test_result": {
            "path": DIRECT_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(DIRECT_OUTPUT),
            "junit": direct_junit,
        },
        "capability_test_result": {"junit": capability_junit},
        "attestation_test_result": {"junit": attestation_junit},
        "elapsed_seconds": time.perf_counter() - started,
        "odt_source_manifest_sha256": _sha256(ODT_MANIFEST),
        "capability_source_manifest_sha256": _sha256(CAPABILITY_MANIFEST),
        "direct_test_manifest_sha256": _sha256(DIRECT_TEST_MANIFEST),
        "stage_ledger_sha256": _sha256(STAGE_LEDGER),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    digest = _publish(payload)
    print(json.dumps({"unit_suite_sha256": digest, "gates": gates}, sort_keys=True))
    if not payload["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
