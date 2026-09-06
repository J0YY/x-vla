#!/usr/bin/env python3
"""Freeze the b1c0 dual-closure staging authority without submitting jobs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping


SOURCE_ROOT = Path(__file__).resolve().parents[1]
MUTABLE_ROOT = Path("/work/joy/x-vla-workshop")
PRESTAGE_ROOT = Path("/work/joy/x-vla-capable-linear-b1c0-prestage-v2")
PRESTAGE_SCHEMA = "xvla_capable_linear_b1c0_prestage_v2"
ODT_MANIFEST_RELATIVE = "athena/capable_linear_direct_odt_sources.sha256"
CAPABILITY_MANIFEST_RELATIVE = (
    "athena/capable_linear_fresh_capability_sources.sha256"
)
DIRECT_TEST_MANIFEST_RELATIVE = "athena/direct_odt_truncation_tests_sources.sha256"
LAUNCH_MANIFEST_RELATIVE = "athena/capable_linear_b1c0_launch.sha256"
EXPECTED_MANIFEST_SHA256 = {
    ODT_MANIFEST_RELATIVE: "ebd9dd4162734e15bfd93c3f8d0aded01fe04001ba0b102dce405879da620e6f",
    CAPABILITY_MANIFEST_RELATIVE: "c62224734d4d9a89c0e9f1fd638b7d5318967531f39adcacd715ac1eb1e6b897",
    DIRECT_TEST_MANIFEST_RELATIVE: "7e1cb9d5be89300b212b1d4dc958d8565fbb22f580a23b6d090e6864dfb73f74",
    LAUNCH_MANIFEST_RELATIVE: "5c2a6ed9ae2778e99789734cf80b943facef602fd5a498f07cb312b45eab6ee2",
}
INPUT_SHA256 = {
    "inputs/capable_linear_b1c0_checkpoint.pt": "b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee",
    "inputs/capable_linear_training.json": "e6c07beb2efb7e5ebe55c92fabf694d97b50b9e399f50868eeef4af455e30fa7",
    "inputs/capable_linear_capability_t0_3.json": "a89c586016ea87336128afefeaf1bdb922925de8abdac81eaa486f79de0bbe7f",
    "inputs/capable_linear_capability_t3_6.json": "f23ab6c31d308da4720c5e0ad8da1aa46729c9453fe4ec0b1cc24d1df9d2e599",
    "inputs/capable_linear_capability_t6_8.json": "40b17ca6a857c3999da8f981f9c3fa262bcbb23ae6ca0ca6cf0b9055556930c5",
    "inputs/capable_linear_capability_t8_10.json": "bd9d01e43c6a3fff455d0704f46351e3973675380174544bb1df53d39a62625d",
    "inputs/direct_odt_truncation_unit_v2.json": "0c41680e8603036de2577638e68b14bc397df7920d96c2fefe5026886cad54e7",
    "inputs/r7_full_exact_control.json": "f90a2c8c4842206aface13801770ea07fd6a4b3f12c023b0faeb984177d8b374",
    "inputs/r7_composite_control.json": "bad374a514e4d4503f5fe316f4a95a12bdbf2c897b964fd8ead601d1a20cc770",
    "reference/dooms_xnets_2504.02667.pdf": "a97f59a91d90ffde56bc22bf0b3c79e0891ff88197ca9d7670331e4208277559",
}


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


def _safe_member(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or not value.parts or any(
        part in {"", ".", ".."} for part in value.parts
    ):
        raise RuntimeError(f"unsafe authenticated member {relative!r}")
    cursor = root
    for part in value.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(f"authenticated member traverses a link: {relative}")
    resolved = cursor.resolve(strict=True)
    if resolved == root or root not in resolved.parents or not resolved.is_file():
        raise RuntimeError(f"authenticated member escapes or is nonphysical: {relative}")
    if resolved.relative_to(root).as_posix() != relative:
        raise RuntimeError(f"authenticated member is not normalized: {relative}")
    return resolved


def _read_manifest(relative: str) -> dict[str, str]:
    path = _safe_member(SOURCE_ROOT, relative)
    expected_manifest_sha = EXPECTED_MANIFEST_SHA256[relative]
    if _sha256(path) != expected_manifest_sha:
        raise RuntimeError(f"authenticated manifest digest differs: {relative}")
    result: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(f"malformed manifest line {number}: {relative}")
        digest, member = fields
        member = member.lstrip("*")
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or member in result
        ):
            raise RuntimeError(f"invalid manifest line {number}: {relative}")
        if _sha256(_safe_member(SOURCE_ROOT, member)) != digest:
            raise RuntimeError(f"manifest byte differs: {member}")
        result[member] = digest
    return result


def _input_sources() -> dict[str, Path]:
    return {
        "inputs/capable_linear_b1c0_checkpoint.pt": SOURCE_ROOT
        / "artifacts/ckpt_rational_norm_ablation_v1_rational_s0.pt",
        "inputs/capable_linear_training.json": SOURCE_ROOT
        / "athena/results/rational_norm_ablation_v1_train_rational_s0.json",
        "inputs/capable_linear_capability_t0_3.json": SOURCE_ROOT
        / "athena/results/rational_norm_ablation_v1_rational_s0_t0_3.json",
        "inputs/capable_linear_capability_t3_6.json": SOURCE_ROOT
        / "athena/results/rational_norm_ablation_v1_rational_s0_t3_6.json",
        "inputs/capable_linear_capability_t6_8.json": SOURCE_ROOT
        / "athena/results/rational_norm_ablation_v1_rational_s0_t6_8.json",
        "inputs/capable_linear_capability_t8_10.json": SOURCE_ROOT
        / "athena/results/rational_norm_ablation_v1_rational_s0_t8_10.json",
        "inputs/direct_odt_truncation_unit_v2.json": Path(
            "/work/joy/x-vla-direct-odt-compression-v2/athena/results/"
            "direct_odt_truncation_unit_v2.json"
        ),
        "inputs/r7_full_exact_control.json": SOURCE_ROOT
        / "athena/results/full_vla_production_full_r7_rankdef_retained_q.json",
        "inputs/r7_composite_control.json": SOURCE_ROOT
        / "athena/results/direct_odt_full_composite_attestation_r7.json",
        "reference/dooms_xnets_2504.02667.pdf": SOURCE_ROOT
        / "tmp/pdfs/dooms-xnets-2504.02667.pdf",
    }


def _copy(source: Path, destination: Path, digest: str) -> None:
    if source.is_symlink() or not source.is_file() or _sha256(source) != digest:
        raise RuntimeError(f"prestage source differs: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.is_symlink() or not destination.is_file() or _sha256(destination) != digest:
        raise RuntimeError(f"prestage copy differs: {destination}")
    destination.chmod(0o444)


def _publish(path: Path, payload: Mapping[str, Any]) -> str:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing existing prestage output {path}")
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
        if _sha256(path) != digest or path.stat().st_mode & 0o222:
            raise RuntimeError("published prestage authority differs")
        return digest
    except BaseException:
        if linked and os.path.lexists(path):
            path.unlink()
        raise
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit("prestage preparation accepts no arguments")
    if SOURCE_ROOT != MUTABLE_ROOT:
        raise RuntimeError(f"prestage preparation must run from {MUTABLE_ROOT}")
    if os.path.lexists(PRESTAGE_ROOT):
        raise FileExistsError(f"refusing existing prestage {PRESTAGE_ROOT}")
    odt_sources = _read_manifest(ODT_MANIFEST_RELATIVE)
    capability_sources = _read_manifest(CAPABILITY_MANIFEST_RELATIVE)
    direct_test_sources = _read_manifest(DIRECT_TEST_MANIFEST_RELATIVE)
    launch_sources = _read_manifest(LAUNCH_MANIFEST_RELATIVE)

    sys.path[:] = [str(SOURCE_ROOT)] + [
        entry for entry in sys.path if entry != str(SOURCE_ROOT)
    ]
    sys.dont_write_bytecode = True
    from scripts.odt_direct_only_compliance import (
        audit_direct_only_launch,
        canonical_direct_only_entrypoints,
    )

    if Path(sys.modules["scripts"].__file__).resolve() != (
        SOURCE_ROOT / "scripts/__init__.py"
    ).resolve() or Path(
        sys.modules["scripts.odt_direct_only_compliance"].__file__
    ).resolve() != (
        SOURCE_ROOT / "scripts/odt_direct_only_compliance.py"
    ).resolve():
        raise RuntimeError("preparer imported a shadowed compliance authority")

    odt_entrypoints = (
        *canonical_direct_only_entrypoints(
            SOURCE_ROOT,
            SOURCE_ROOT / "scripts/run_capable_linear_direct_odt_spectrum.py",
        ),
        SOURCE_ROOT / "tests/test_direct_odt_truncation.py",
        SOURCE_ROOT / "tests/test_capable_linear_artifact_integrity.py",
    )
    if len(odt_entrypoints) != len({path.resolve() for path in odt_entrypoints}):
        raise RuntimeError("ODT prestage entrypoint inventory contains duplicates")
    capability_entrypoints = (
        SOURCE_ROOT / "scripts/run_capable_linear_fresh_capability.py",
        SOURCE_ROOT / "tests/test_capable_linear_fresh_capability.py",
    )
    direct_test_entrypoints = (
        *canonical_direct_only_entrypoints(
            SOURCE_ROOT, SOURCE_ROOT / "scripts/run_direct_odt_truncation_tests.py"
        ),
        SOURCE_ROOT / "tests/test_direct_odt_truncation.py",
    )
    audits = {
        "odt": audit_direct_only_launch(SOURCE_ROOT, odt_entrypoints),
        "capability": audit_direct_only_launch(
            SOURCE_ROOT, capability_entrypoints, require_direct_qr=False
        ),
        "direct_tests": audit_direct_only_launch(
            SOURCE_ROOT, direct_test_entrypoints
        ),
    }
    for label, source_map in (
        ("odt", odt_sources),
        ("capability", capability_sources),
        ("direct_tests", direct_test_sources),
    ):
        audit = audits[label]
        if audit["source_sha256"] != source_map:
            raise RuntimeError(f"{label} manifest differs from its transitive audit")
        for field in (
            "prohibited_calls_found",
            "prohibited_self_overlap_sites",
            "guarded_dormant_spectral_norm_sites",
            "duplicate_top_level_definition_sites",
        ):
            if audit[field]:
                raise RuntimeError(f"{label} prestage audit failed {field}")

    input_sources = _input_sources()
    if set(input_sources) != set(INPUT_SHA256):
        raise RuntimeError("input source inventory differs")
    for relative, source in input_sources.items():
        if source.is_symlink() or not source.is_file() or _sha256(source) != INPUT_SHA256[relative]:
            raise RuntimeError(f"authenticated input differs: {relative}")

    authority = {
        **launch_sources,
        ODT_MANIFEST_RELATIVE: EXPECTED_MANIFEST_SHA256[ODT_MANIFEST_RELATIVE],
        CAPABILITY_MANIFEST_RELATIVE: EXPECTED_MANIFEST_SHA256[
            CAPABILITY_MANIFEST_RELATIVE
        ],
        DIRECT_TEST_MANIFEST_RELATIVE: EXPECTED_MANIFEST_SHA256[
            DIRECT_TEST_MANIFEST_RELATIVE
        ],
        LAUNCH_MANIFEST_RELATIVE: EXPECTED_MANIFEST_SHA256[
            LAUNCH_MANIFEST_RELATIVE
        ],
        "athena/prepare_capable_linear_b1c0_prestage.py": _sha256(
            Path(__file__).resolve()
        ),
    }
    sections = {
        "odt_source_sha256": odt_sources,
        "capability_source_sha256": capability_sources,
        "direct_test_source_sha256": direct_test_sources,
        "launch_sha256": authority,
        "input_sha256": INPUT_SHA256,
    }
    closure: dict[str, str] = {}
    for section in sections.values():
        for relative, digest in section.items():
            prior = closure.setdefault(relative, digest)
            if prior != digest:
                raise RuntimeError(f"prestage sections disagree on {relative}")

    PRESTAGE_ROOT.mkdir(mode=0o755)
    for relative, digest in sorted(closure.items()):
        source = (
            input_sources[relative]
            if relative in input_sources
            else _safe_member(SOURCE_ROOT, relative)
        )
        _copy(source, PRESTAGE_ROOT / relative, digest)
    manifest_payload = {
        "schema": PRESTAGE_SCHEMA,
        "sections": sections,
        "closure": closure,
        "bundle_sha256": _canonical_sha256(dict(sorted(closure.items()))),
    }
    manifest_path = PRESTAGE_ROOT / "athena/capable_linear_b1c0_prestage_manifest.json"
    manifest_sha = _publish(manifest_path, manifest_payload)
    ledger_payload = {
        "schema": f"{PRESTAGE_SCHEMA}_ledger",
        "prestage_root": PRESTAGE_ROOT.as_posix(),
        "prestage_manifest_sha256": manifest_sha,
        "prestage_bundle_sha256": manifest_payload["bundle_sha256"],
        "no_job_submitted": True,
    }
    ledger_path = PRESTAGE_ROOT / "prestage_ledger.json"
    ledger_sha = _publish(ledger_path, ledger_payload)
    for directory in sorted(
        (path for path in PRESTAGE_ROOT.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    PRESTAGE_ROOT.chmod(0o555)
    for relative, digest in closure.items():
        staged = _safe_member(PRESTAGE_ROOT, relative)
        if _sha256(staged) != digest or staged.stat().st_mode & 0o222:
            raise RuntimeError(f"frozen prestage member differs: {relative}")
    if _sha256(manifest_path) != manifest_sha or _sha256(ledger_path) != ledger_sha:
        raise RuntimeError("frozen prestage authority changed before handoff")
    print(
        json.dumps(
            {
                "prestage_root": PRESTAGE_ROOT.as_posix(),
                "manifest_sha256": manifest_sha,
                "ledger_sha256": ledger_sha,
                "bundle_sha256": manifest_payload["bundle_sha256"],
                "no_job_submitted": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
