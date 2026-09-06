#!/usr/bin/env python3
"""Run the authenticated direct-only truncation regression suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXPECTED_SOURCE_MANIFEST = (
    PROJECT_ROOT / "athena/direct_odt_truncation_tests_sources.sha256"
)
EXPECTED_TEST_COUNT = 52


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("the pinned test manifest must be a real file")
    if path.resolve().parent != (PROJECT_ROOT / "athena").resolve(strict=True):
        raise RuntimeError("the pinned test manifest is outside athena")
    expected: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(f"malformed test manifest line {line_number}")
        digest, relative = fields
        relative = relative.lstrip("*")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise RuntimeError(f"invalid test digest on line {line_number}")
        source = PROJECT_ROOT / relative
        resolved = source.resolve(strict=True)
        if (
            source.is_symlink()
            or resolved == PROJECT_ROOT
            or PROJECT_ROOT not in resolved.parents
        ):
            raise RuntimeError("test manifest contains an unsafe path")
        normalized = resolved.relative_to(PROJECT_ROOT).as_posix()
        if normalized in expected:
            raise RuntimeError("test manifest contains a duplicate path")
        expected[normalized] = digest
    runner = Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix()
    if runner not in expected:
        raise RuntimeError("test manifest does not authenticate this runner")
    actual: dict[str, str] = {}
    for relative, digest in sorted(expected.items()):
        actual_digest = _sha256(PROJECT_ROOT / relative)
        if actual_digest != digest:
            raise RuntimeError(f"authenticated test source changed: {relative}")
        actual[relative] = actual_digest
    return {
        "path": path.resolve().as_posix(),
        "sha256": _sha256(path),
        "source_count": len(actual),
        "source_sha256": actual,
    }


_PREIMPORT_MANIFEST = _verify_manifest(EXPECTED_SOURCE_MANIFEST)

from scripts.odt_direct_only_compliance import (
    STREAMED_CLONE_ORACLE_RUNTIME_CALLS,
    assert_direct_only_runtime_guard,
    audit_direct_only_launch,
    canonical_direct_only_entrypoints,
    direct_only_runtime_report,
    install_direct_only_runtime_guard,
)


_ENTRYPOINTS = (
    *canonical_direct_only_entrypoints(PROJECT_ROOT, Path(__file__)),
    PROJECT_ROOT / "tests/test_direct_odt_truncation.py",
)
_STATIC_AUDIT = audit_direct_only_launch(PROJECT_ROOT, _ENTRYPOINTS)
if _STATIC_AUDIT["guarded_dormant_spectral_norm_sites"]:
    raise RuntimeError("the test closure contains a dormant blocked norm site")
if _PREIMPORT_MANIFEST["source_sha256"] != _STATIC_AUDIT["source_sha256"]:
    raise RuntimeError("pinned test manifest differs from the audited closure")
_RUNTIME_AT_IMPORT = install_direct_only_runtime_guard()

import numpy as np
import pytest
import torch


if sys.version_info[:2] != (3, 10):
    raise RuntimeError("the test runner requires the frozen Python 3.10 environment")
if np.__version__ != "1.26.4":
    raise RuntimeError("the test runner requires the frozen NumPy 1.26.4 environment")
if torch.__version__.split("+", 1)[0] != "2.7.1":
    raise RuntimeError("the test runner requires the frozen Torch 2.7.1 environment")


def _assert_sources_unchanged() -> dict[str, object]:
    audit = audit_direct_only_launch(PROJECT_ROOT, _ENTRYPOINTS)
    if audit != _STATIC_AUDIT:
        raise RuntimeError("the audited test closure changed during execution")
    manifest = _verify_manifest(EXPECTED_SOURCE_MANIFEST)
    if manifest != _PREIMPORT_MANIFEST:
        raise RuntimeError("the authenticated test closure changed during execution")
    return manifest


def _validated_output(path: Path) -> Path:
    raw = path if path.is_absolute() else PROJECT_ROOT / path
    if os.path.lexists(raw):
        raise FileExistsError(f"refusing to replace existing result {raw}")
    directory_path = PROJECT_ROOT / "athena/results"
    if directory_path.is_symlink() or not directory_path.is_dir():
        raise RuntimeError("the result directory must be a real directory")
    directory = directory_path.resolve(strict=True)
    resolved = raw.resolve(strict=False)
    if resolved.parent != directory or resolved.name in {"", ".", ".."}:
        raise ValueError("output must be a direct child of athena/results")
    return resolved


def _junit_summary(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag.endswith("testsuite") else list(root.findall("testsuite"))
    summary: dict[str, object] = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    summary["test_names"] = sorted(
        item.attrib["name"]
        for item in root.iter()
        if item.tag.endswith("testcase") and "name" in item.attrib
    )
    return summary


def _publish(path: Path, payload: dict[str, object]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(12)}.tmp")
    published = False
    temporary_stat: os.stat_result | None = None
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_stat = os.lstat(temporary)
        if not stat.S_ISREG(temporary_stat.st_mode):
            raise RuntimeError("test-result temporary is not a regular file")
        _assert_sources_unchanged()
        os.link(temporary, path, follow_symlinks=False)
        published = True
        if _sha256(path) != digest:
            raise RuntimeError("published test result differs from staged bytes")
        _assert_sources_unchanged()
        os.chmod(path, 0o444, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return digest
    except BaseException:
        if published and temporary_stat is not None and os.path.lexists(path):
            current = os.lstat(path)
            if current.st_dev == temporary_stat.st_dev and current.st_ino == temporary_stat.st_ino:
                path.unlink()
        raise
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = _validated_output(args.output)

    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    temporary_junit = output.with_name(
        f".{output.name}.{secrets.token_hex(12)}.junit.tmp"
    )
    if os.path.lexists(temporary_junit):
        raise FileExistsError("refusing to replace a test temporary")
    test_paths = (
        "tests/test_direct_odt_truncation.py",
        "tests/test_direct_odt_clone_reference.py",
        "tests/test_implicit_sparse_projective_odt_heterogeneous.py",
        "tests/test_implicit_sparse_projective_odt_vla.py",
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
                    *test_paths,
                    f"--junitxml={temporary_junit}",
                ]
            )
        )
        junit = _junit_summary(temporary_junit)
    finally:
        if os.path.lexists(temporary_junit):
            temporary_junit.unlink()
    runtime = assert_direct_only_runtime_guard(direct_only_runtime_report())
    observed_calls = set(runtime["allowed_calls"])
    required_calls_present = STREAMED_CLONE_ORACLE_RUNTIME_CALLS <= observed_calls
    sources_before_result = _assert_sources_unchanged()
    gates = {
        "pytest_exit_zero": exit_code == 0,
        "exact_test_count": junit["tests"] == EXPECTED_TEST_COUNT,
        "no_failures": junit["failures"] == 0 and junit["errors"] == 0,
        "no_skips": junit["skipped"] == 0,
        "unique_test_names": (
            len(junit["test_names"]) == EXPECTED_TEST_COUNT
            and len(set(junit["test_names"])) == EXPECTED_TEST_COUNT
        ),
        "required_direct_runtime_calls_present": required_calls_present,
        "runtime_guard_clean": runtime["prohibited_attempt_count"] == 0,
        "source_closure_unchanged": True,
    }
    result = {
        "object_kind": "direct_odt_truncation_authenticated_unit_suite_v1",
        "test_paths": list(test_paths),
        "elapsed_seconds": time.perf_counter() - started,
        "junit": junit,
        "static_audit": _STATIC_AUDIT,
        "runtime_guard_at_import": _RUNTIME_AT_IMPORT,
        "runtime_guard_final": runtime,
        "source_manifest_before_imports": _PREIMPORT_MANIFEST,
        "source_manifest_before_result": sources_before_result,
        "frozen_runtime": {
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "executable": Path(sys.executable).resolve().as_posix(),
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    digest = _publish(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"published_sha256={digest}")
    if not result["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
