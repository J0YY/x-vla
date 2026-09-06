#!/usr/bin/env python3
"""Freeze the RTX 6000 capability staging authority without submitting jobs."""

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
PRESTAGE_ROOT = Path("/work/joy/x-vla-capable-linear-rtx6000-prestage-v3")
FROZEN_B1C_ROOT = Path("/work/joy/x-vla-capable-linear-b1c0-odt-v2")
FROZEN_B1C_STAGE_LEDGER_SHA256 = (
    "e4716a413bc742943b623ddebd39783bb54c40558b59446c99998aa07892b8c3"
)
FROZEN_B1C_CAPABILITY_MANIFEST_SHA256 = (
    "c62224734d4d9a89c0e9f1fd638b7d5318967531f39adcacd715ac1eb1e6b897"
)
PRESTAGE_SCHEMA = "xvla_capable_linear_rtx6000_prestage_v3"
CROSS_STAGE_MANIFEST_RELATIVE = "athena/capable_linear_rtx6000_cross_stage_sources.sha256"
CAPABILITY_MANIFEST_RELATIVE = (
    "athena/capable_linear_rtx6000_capability_sources.sha256"
)
LAUNCH_MANIFEST_RELATIVE = "athena/capable_linear_rtx6000_launch.sha256"
EXPECTED_MANIFEST_SHA256 = {
    CROSS_STAGE_MANIFEST_RELATIVE: "60ecb8a69147758592e61dc52bd525776aa6112d3c085e6ea5d094d3ca62e83f",
    CAPABILITY_MANIFEST_RELATIVE: "285f2366ee780aaca1dd9ed3a7777d7cd9d9136d6466fabee42eb6e487148af8",
    LAUNCH_MANIFEST_RELATIVE: "1c4a7a678c0a28b4669e9511c7c43fba20de458e3871c46aee674cc17f7d35c6",
}
INPUT_SHA256 = {
    "inputs/capable_linear_b1c0_checkpoint.pt": "b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee",
    "inputs/capable_linear_training.json": "e6c07beb2efb7e5ebe55c92fabf694d97b50b9e399f50868eeef4af455e30fa7",
    "inputs/b1c_stage_ledger.sha256": "e4716a413bc742943b623ddebd39783bb54c40558b59446c99998aa07892b8c3",
    "inputs/b1c_odt_sources.sha256": "ebd9dd4162734e15bfd93c3f8d0aded01fe04001ba0b102dce405879da620e6f",
    "inputs/b1c_capability_sources.sha256": "c62224734d4d9a89c0e9f1fd638b7d5318967531f39adcacd715ac1eb1e6b897",
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
        result[member] = digest
    return result


def _read_frozen_manifest(path: Path, label: str) -> dict[str, str]:
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
        digest, member = fields
        member = member.lstrip("*")
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or member in result
        ):
            raise RuntimeError(f"invalid {label} line {number}")
        if _sha256(_safe_member(FROZEN_B1C_ROOT, member)) != digest:
            raise RuntimeError(f"{label} byte differs: {member}")
        result[member] = digest
    return result


def _input_sources() -> dict[str, Path]:
    return {
        "inputs/capable_linear_b1c0_checkpoint.pt": FROZEN_B1C_ROOT
        / "inputs/capable_linear_b1c0_checkpoint.pt",
        "inputs/capable_linear_training.json": FROZEN_B1C_ROOT
        / "inputs/capable_linear_training.json",
        "inputs/b1c_stage_ledger.sha256": FROZEN_B1C_ROOT
        / "athena/capable_linear_b1c0_stage.sha256",
        "inputs/b1c_odt_sources.sha256": FROZEN_B1C_ROOT
        / "athena/capable_linear_direct_odt_sources.sha256",
        "inputs/b1c_capability_sources.sha256": FROZEN_B1C_ROOT
        / "athena/capable_linear_fresh_capability_sources.sha256",
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
    cross_stage_sources = _read_manifest(CROSS_STAGE_MANIFEST_RELATIVE)
    capability_sources = _read_manifest(CAPABILITY_MANIFEST_RELATIVE)
    launch_sources = _read_manifest(LAUNCH_MANIFEST_RELATIVE)

    frozen_ledger_path = FROZEN_B1C_ROOT / "athena/capable_linear_b1c0_stage.sha256"
    if _sha256(frozen_ledger_path) != FROZEN_B1C_STAGE_LEDGER_SHA256:
        raise RuntimeError("frozen b1c stage ledger identity differs")
    frozen_ledger = _read_frozen_manifest(frozen_ledger_path, "frozen b1c stage ledger")
    frozen_capability_path = (
        FROZEN_B1C_ROOT / "athena/capable_linear_fresh_capability_sources.sha256"
    )
    if _sha256(frozen_capability_path) != FROZEN_B1C_CAPABILITY_MANIFEST_SHA256:
        raise RuntimeError("frozen b1c capability manifest identity differs")
    frozen_capability_sources = _read_frozen_manifest(
        frozen_capability_path, "frozen b1c capability manifest"
    )
    old_specific = {
        "scripts/run_capable_linear_fresh_capability.py",
        "tests/test_capable_linear_fresh_capability.py",
    }
    new_capability_specific = {
        "scripts/run_capable_linear_rtx6000_capability.py",
        "tests/test_capable_linear_rtx6000_capability.py",
    }
    new_cross_specific = {
        *new_capability_specific,
        "scripts/attest_capable_linear_rtx6000_cross_stage.py",
        "scripts/run_capable_linear_rtx6000_tests.py",
        "tests/test_capable_linear_rtx6000_cross_stage.py",
    }
    frozen_shared = {
        relative: digest
        for relative, digest in frozen_capability_sources.items()
        if relative not in old_specific
    }
    capability_shared = {
        relative: digest
        for relative, digest in capability_sources.items()
        if relative not in new_capability_specific
    }
    cross_shared = {
        relative: digest
        for relative, digest in cross_stage_sources.items()
        if relative not in new_cross_specific
    }
    if (
        capability_shared != frozen_shared
        or cross_shared != frozen_shared
        or set(capability_sources) - set(frozen_shared) != new_capability_specific
        or set(cross_stage_sources) - set(frozen_shared) != new_cross_specific
    ):
        raise RuntimeError("RTX closure shared-source intersection differs from frozen b1c")
    for relative, digest in frozen_shared.items():
        if frozen_ledger.get(relative) != digest:
            raise RuntimeError(f"frozen b1c ledger does not bind shared source {relative}")

    input_sources = _input_sources()
    if set(input_sources) != set(INPUT_SHA256):
        raise RuntimeError("input source inventory differs")
    for relative, source in input_sources.items():
        if source.is_symlink() or not source.is_file() or _sha256(source) != INPUT_SHA256[relative]:
            raise RuntimeError(f"authenticated input differs: {relative}")

    authority = {
        **launch_sources,
        CROSS_STAGE_MANIFEST_RELATIVE: EXPECTED_MANIFEST_SHA256[
            CROSS_STAGE_MANIFEST_RELATIVE
        ],
        CAPABILITY_MANIFEST_RELATIVE: EXPECTED_MANIFEST_SHA256[
            CAPABILITY_MANIFEST_RELATIVE
        ],
        LAUNCH_MANIFEST_RELATIVE: EXPECTED_MANIFEST_SHA256[
            LAUNCH_MANIFEST_RELATIVE
        ],
        "athena/prepare_capable_linear_rtx6000_prestage.py": _sha256(
            Path(__file__).resolve()
        ),
    }
    sections = {
        "cross_stage_source_sha256": cross_stage_sources,
        "capability_source_sha256": capability_sources,
        "launch_sha256": authority,
        "input_sha256": INPUT_SHA256,
    }
    closure: dict[str, str] = {}
    for section in sections.values():
        for relative, digest in section.items():
            prior = closure.setdefault(relative, digest)
            if prior != digest:
                raise RuntimeError(f"prestage sections disagree on {relative}")

    workshop_members = {
        *new_cross_specific,
        *launch_sources,
        CROSS_STAGE_MANIFEST_RELATIVE,
        CAPABILITY_MANIFEST_RELATIVE,
        LAUNCH_MANIFEST_RELATIVE,
        "athena/prepare_capable_linear_rtx6000_prestage.py",
    }
    source_origin: dict[str, str] = {}
    copy_sources: dict[str, Path] = {}
    for relative, digest in closure.items():
        if relative in input_sources:
            source = input_sources[relative]
            origin = "frozen_input"
        elif relative in workshop_members:
            source = _safe_member(SOURCE_ROOT, relative)
            origin = "new_workshop_authority"
        else:
            if frozen_ledger.get(relative) != digest:
                raise RuntimeError(f"shared source lacks frozen b1c identity: {relative}")
            source = _safe_member(FROZEN_B1C_ROOT, relative)
            origin = "frozen_b1c_v2"
        if source.is_symlink() or not source.is_file() or _sha256(source) != digest:
            raise RuntimeError(f"prestage source identity differs: {relative}")
        copy_sources[relative] = source
        source_origin[relative] = origin

    PRESTAGE_ROOT.mkdir(mode=0o755)
    for relative, digest in sorted(closure.items()):
        _copy(copy_sources[relative], PRESTAGE_ROOT / relative, digest)

    sys.path[:] = [str(PRESTAGE_ROOT)] + [
        entry
        for entry in sys.path
        if entry not in {str(PRESTAGE_ROOT), str(SOURCE_ROOT), str(FROZEN_B1C_ROOT)}
    ]
    sys.dont_write_bytecode = True
    from scripts.odt_direct_only_compliance import audit_direct_only_launch

    if Path(sys.modules["scripts"].__file__).resolve() != (
        PRESTAGE_ROOT / "scripts/__init__.py"
    ).resolve() or Path(
        sys.modules["scripts.odt_direct_only_compliance"].__file__
    ).resolve() != (
        PRESTAGE_ROOT / "scripts/odt_direct_only_compliance.py"
    ).resolve():
        raise RuntimeError("preparer imported a shadowed compliance authority")
    cross_stage_entrypoints = tuple(
        PRESTAGE_ROOT / relative for relative in sorted(new_cross_specific)
    )
    capability_entrypoints = tuple(
        PRESTAGE_ROOT / relative for relative in sorted(new_capability_specific)
    )
    audits = {
        "cross_stage": audit_direct_only_launch(
            PRESTAGE_ROOT, cross_stage_entrypoints, require_direct_qr=False
        ),
        "capability": audit_direct_only_launch(
            PRESTAGE_ROOT, capability_entrypoints, require_direct_qr=False
        ),
    }
    for label, source_map in (
        ("cross_stage", cross_stage_sources),
        ("capability", capability_sources),
    ):
        audit = audits[label]
        if audit["source_sha256"] != source_map:
            raise RuntimeError(f"{label} manifest differs from frozen mixed-origin audit")
        for field in (
            "prohibited_calls_found",
            "prohibited_self_overlap_sites",
            "guarded_dormant_spectral_norm_sites",
            "duplicate_top_level_definition_sites",
        ):
            if audit[field]:
                raise RuntimeError(f"{label} prestage audit failed {field}")
    manifest_payload = {
        "schema": PRESTAGE_SCHEMA,
        "sections": sections,
        "closure": closure,
        "source_origin": source_origin,
        "frozen_b1c_stage_ledger_sha256": FROZEN_B1C_STAGE_LEDGER_SHA256,
        "frozen_b1c_capability_manifest_sha256": (
            FROZEN_B1C_CAPABILITY_MANIFEST_SHA256
        ),
        "shared_source_sha256": frozen_shared,
        "bundle_sha256": _canonical_sha256(dict(sorted(closure.items()))),
    }
    manifest_path = PRESTAGE_ROOT / "athena/capable_linear_rtx6000_prestage_manifest.json"
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
