#!/usr/bin/env python3
"""Create the b1c0 stage only from its immutable authenticated prestage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping


SOURCE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PRESTAGE_ROOT = Path("/work/joy/x-vla-capable-linear-b1c0-prestage-v2")
STAGE_ROOT = Path("/work/joy/x-vla-capable-linear-b1c0-odt-v2")
PRESTAGE_MANIFEST = SOURCE_ROOT / "athena/capable_linear_b1c0_prestage_manifest.json"
PRESTAGE_LEDGER = SOURCE_ROOT / "prestage_ledger.json"
PRESTAGE_SCHEMA = "xvla_capable_linear_b1c0_prestage_v2"
ODT_MANIFEST_RELATIVE = "athena/capable_linear_direct_odt_sources.sha256"
CAPABILITY_MANIFEST_RELATIVE = (
    "athena/capable_linear_fresh_capability_sources.sha256"
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


def _safe_member(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or not value.parts or any(
        part in {"", ".", ".."} for part in value.parts
    ):
        raise RuntimeError(f"unsafe prestage member {relative!r}")
    cursor = root
    for part in value.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(f"prestage member traverses a link: {relative}")
    resolved = cursor.resolve(strict=True)
    if resolved == root or root not in resolved.parents or not resolved.is_file():
        raise RuntimeError(f"prestage member escapes or is nonphysical: {relative}")
    if resolved.relative_to(root).as_posix() != relative:
        raise RuntimeError(f"prestage member is not normalized: {relative}")
    return resolved


def _verified_prestage() -> dict[str, Any]:
    if SOURCE_ROOT != EXPECTED_PRESTAGE_ROOT:
        raise RuntimeError(f"staging must run from {EXPECTED_PRESTAGE_ROOT}")
    manifest = _read_object(PRESTAGE_MANIFEST, "prestage manifest")
    ledger = _read_object(PRESTAGE_LEDGER, "prestage ledger")
    sections = manifest.get("sections")
    closure = manifest.get("closure")
    if (
        manifest.get("schema") != PRESTAGE_SCHEMA
        or not isinstance(sections, dict)
        or not isinstance(closure, dict)
    ):
        raise RuntimeError("prestage manifest schema or maps differ")
    expected_sections = {
        "odt_source_sha256",
        "capability_source_sha256",
        "direct_test_source_sha256",
        "launch_sha256",
        "input_sha256",
    }
    if set(sections) != expected_sections:
        raise RuntimeError("prestage section inventory differs")
    reconstructed: dict[str, str] = {}
    for section_name, section in sections.items():
        if not isinstance(section, dict):
            raise RuntimeError(f"prestage section is malformed: {section_name}")
        for relative, digest in section.items():
            if (
                not isinstance(relative, str)
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise RuntimeError(f"prestage section entry is malformed: {section_name}")
            prior = reconstructed.setdefault(relative, digest)
            if prior != digest:
                raise RuntimeError(f"prestage sections disagree on {relative}")
    if reconstructed != closure:
        raise RuntimeError("prestage closure differs from section union")
    if manifest.get("bundle_sha256") != _canonical_sha256(dict(sorted(closure.items()))):
        raise RuntimeError("prestage bundle differs")
    required = {
        "athena/stage_capable_linear_b1c0.py",
        "scripts/__init__.py",
        "scripts/odt_direct_only_compliance.py",
        ODT_MANIFEST_RELATIVE,
        CAPABILITY_MANIFEST_RELATIVE,
    }
    if not required.issubset(closure):
        raise RuntimeError("prestage omits a staging or source authority")
    for relative, digest in closure.items():
        if _sha256(_safe_member(SOURCE_ROOT, relative)) != digest:
            raise RuntimeError(f"prestage byte changed: {relative}")
    if (
        ledger.get("schema") != f"{PRESTAGE_SCHEMA}_ledger"
        or ledger.get("prestage_root") != SOURCE_ROOT.as_posix()
        or ledger.get("prestage_manifest_sha256") != _sha256(PRESTAGE_MANIFEST)
        or ledger.get("prestage_bundle_sha256") != manifest["bundle_sha256"]
        or ledger.get("no_job_submitted") is not True
    ):
        raise RuntimeError("prestage ledger differs")
    return {"manifest": manifest, "ledger": ledger}


_PREIMPORT = _verified_prestage()
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
    raise RuntimeError("stager imported a shadowed compliance authority")


def _audit_sections() -> dict[str, Any]:
    sections = _PREIMPORT["manifest"]["sections"]
    odt_entrypoints = (
        *canonical_direct_only_entrypoints(
            SOURCE_ROOT,
            SOURCE_ROOT / "scripts/run_capable_linear_direct_odt_spectrum.py",
        ),
        SOURCE_ROOT / "tests/test_direct_odt_truncation.py",
        SOURCE_ROOT / "tests/test_capable_linear_artifact_integrity.py",
    )
    if len(odt_entrypoints) != len({path.resolve() for path in odt_entrypoints}):
        raise RuntimeError("ODT staging entrypoint inventory contains duplicates")
    capability_entrypoints = (
        SOURCE_ROOT / "scripts/run_capable_linear_fresh_capability.py",
        SOURCE_ROOT / "tests/test_capable_linear_fresh_capability.py",
    )
    odt = audit_direct_only_launch(SOURCE_ROOT, odt_entrypoints)
    capability = audit_direct_only_launch(
        SOURCE_ROOT, capability_entrypoints, require_direct_qr=False
    )
    if odt["source_sha256"] != sections["odt_source_sha256"]:
        raise RuntimeError("prestage ODT section differs from transitive audit")
    if capability["source_sha256"] != sections["capability_source_sha256"]:
        raise RuntimeError("prestage capability section differs from transitive audit")
    for label, audit in (("ODT", odt), ("capability", capability)):
        for field in (
            "prohibited_calls_found",
            "prohibited_self_overlap_sites",
            "guarded_dormant_spectral_norm_sites",
            "duplicate_top_level_definition_sites",
        ):
            if audit[field]:
                raise RuntimeError(f"{label} staging audit failed {field}")
    return {"odt": odt, "capability": capability}


def _copy(source: Path, destination: Path, digest: str) -> None:
    if source.is_symlink() or not source.is_file() or _sha256(source) != digest:
        raise RuntimeError(f"staging source differs: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.is_symlink() or not destination.is_file() or _sha256(destination) != digest:
        raise RuntimeError(f"staged copy differs: {destination}")
    destination.chmod(0o444)


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit("stager accepts no arguments")
    audits = _audit_sections()
    if os.path.lexists(STAGE_ROOT):
        raise FileExistsError(f"refusing existing stage {STAGE_ROOT}")
    closure = dict(_PREIMPORT["manifest"]["closure"])
    closure["inputs/b1c0_prestage_manifest.json"] = _sha256(PRESTAGE_MANIFEST)
    closure["inputs/b1c0_prestage_ledger.json"] = _sha256(PRESTAGE_LEDGER)
    STAGE_ROOT.mkdir(mode=0o755)
    for relative, digest in sorted(closure.items()):
        source = (
            PRESTAGE_MANIFEST
            if relative == "inputs/b1c0_prestage_manifest.json"
            else PRESTAGE_LEDGER
            if relative == "inputs/b1c0_prestage_ledger.json"
            else _safe_member(SOURCE_ROOT, relative)
        )
        _copy(source, STAGE_ROOT / relative, digest)

    results_root = STAGE_ROOT / "athena/results"
    results = results_root / "capable_linear_b1c0"
    progress = results / "progress"
    logs = STAGE_ROOT / "athena/logs"
    results.mkdir(parents=True, mode=0o755)
    progress.mkdir(mode=0o755)
    logs.mkdir(parents=True, mode=0o755)
    ledger_path = STAGE_ROOT / "athena/capable_linear_b1c0_stage.sha256"
    with ledger_path.open("x", encoding="utf-8") as stream:
        for relative, digest in sorted(closure.items()):
            stream.write(f"{digest}  {relative}\n")
        stream.flush()
        os.fsync(stream.fileno())
    ledger_path.chmod(0o444)
    writable = {results_root, results, progress, logs}
    for directory in sorted(
        (path for path in STAGE_ROOT.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o755 if directory in writable else 0o555)
    STAGE_ROOT.chmod(0o555)
    for relative, digest in closure.items():
        staged = _safe_member(STAGE_ROOT, relative)
        if _sha256(staged) != digest or staged.stat().st_mode & 0o222:
            raise RuntimeError(f"staged member is not immutable: {relative}")
    if _verified_prestage() != _PREIMPORT:
        raise RuntimeError("prestage changed across final staging")
    payload = {
        "schema": "xvla_capable_linear_b1c0_stage_v2",
        "stage_root": STAGE_ROOT.as_posix(),
        "file_count": len(closure),
        "stage_ledger_sha256": _sha256(ledger_path),
        "prestage_manifest_sha256": _sha256(PRESTAGE_MANIFEST),
        "prestage_ledger_sha256": _sha256(PRESTAGE_LEDGER),
        "odt_static_audit": audits["odt"],
        "capability_static_audit": audits["capability"],
        "no_job_submitted": True,
        "next_command": f"bash {STAGE_ROOT}/athena/submit_capable_linear_b1c0.sh",
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
