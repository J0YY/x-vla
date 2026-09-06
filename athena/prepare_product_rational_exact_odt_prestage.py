#!/usr/bin/env python3
"""Freeze the delayed Product exact-ODT staging authority before submission."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping


SOURCE_ROOT = Path(__file__).resolve().parents[1]
_ATHENA_INIT = SOURCE_ROOT / "athena/__init__.py"
_ATHENA_INIT_SHA256 = (
    "d496a05665474feac988bdd6f2c616ca82af4bc282c50917b54ae5bbea2d5426"
)
if (
    _ATHENA_INIT.is_symlink()
    or not _ATHENA_INIT.is_file()
    or hashlib.sha256(_ATHENA_INIT.read_bytes()).hexdigest() != _ATHENA_INIT_SHA256
):
    raise RuntimeError("Authenticated local experiment package initializer differs")
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from athena.product_rational_exact_odt_protocol import (
    CAPABILITY_RUN_ROOT,
    CAPABILITY_STAGE_ROOT,
    DOOMS_REFERENCE_SHA256,
    MUTABLE_SOURCE_ROOT,
    ODT_PRESTAGE_ROOT,
    ODT_SOURCE_OVERRIDES,
    PRESTAGE_CLOSURE,
    PRESTAGE_LEDGER_PATH,
    PRESTAGE_MANIFEST_PATH,
    PRESTAGE_SCHEMA,
    SOURCE_CLOSURE,
    canonical_sha256,
    file_sha256,
)
from athena.product_rational_protocol import (
    SOURCE_CLOSURE as CAPABILITY_SOURCE_CLOSURE,
    load_and_verify_source_manifest as verify_capability_manifest,
)
from scripts.odt_direct_only_compliance import audit_direct_only_launch


def _require_physical(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or nonphysical: {path}")


def _publish_json(path: Path, payload: Mapping[str, Any]) -> str:
    if os.path.lexists(path):
        raise FileExistsError(f"Refusing existing prestage output {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    linked = False
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        if file_sha256(path) != digest:
            raise RuntimeError("Published prestage bytes differ")
        path.chmod(0o444)
        return digest
    except BaseException:
        if linked and os.path.lexists(path):
            path.unlink()
        raise
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def _copy_physical(source: Path, destination: Path) -> str:
    _require_physical(source, "Prestage source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.is_symlink() or not destination.is_file():
        raise RuntimeError(f"Prestage member is nonphysical: {destination}")
    digest = file_sha256(source)
    if file_sha256(destination) != digest:
        raise RuntimeError(f"Prestage copy differs: {destination}")
    destination.chmod(0o444)
    return digest


def main() -> None:
    if SOURCE_ROOT != MUTABLE_SOURCE_ROOT:
        raise RuntimeError(f"Prestage preparation must run from {MUTABLE_SOURCE_ROOT}")
    if os.path.lexists(ODT_PRESTAGE_ROOT):
        raise FileExistsError(f"Refusing existing prestage root {ODT_PRESTAGE_ROOT}")
    capability_manifest_path = (
        CAPABILITY_STAGE_ROOT
        / CAPABILITY_RUN_ROOT
        / "manifest/product_pade_rational_source_manifest.json"
    )
    capability_manifest = verify_capability_manifest(
        capability_manifest_path, CAPABILITY_STAGE_ROOT
    )
    capability_sources = capability_manifest["source_closure"]
    for relative in CAPABILITY_SOURCE_CLOSURE:
        if (
            relative in SOURCE_CLOSURE
            and relative not in ODT_SOURCE_OVERRIDES
            and file_sha256(SOURCE_ROOT / relative) != capability_sources[relative]
        ):
            raise RuntimeError(
                f"Mutable source differs from the trained capability byte: {relative}"
            )

    local_audit = audit_direct_only_launch(
        SOURCE_ROOT,
        tuple(SOURCE_ROOT / relative for relative in SOURCE_CLOSURE),
    )
    if local_audit["source_sha256"] != {
        relative: file_sha256(SOURCE_ROOT / relative) for relative in SOURCE_CLOSURE
    }:
        raise RuntimeError("Local exact-ODT tuple differs from its static audit")
    for field in (
        "prohibited_calls_found",
        "prohibited_self_overlap_sites",
        "guarded_dormant_spectral_norm_sites",
    ):
        if local_audit[field] != []:
            raise RuntimeError(f"Local exact-ODT static gate failed: {field}")

    reference = SOURCE_ROOT / "tmp/pdfs/dooms-xnets-2504.02667.pdf"
    _require_physical(reference, "Dooms reference")
    if file_sha256(reference) != DOOMS_REFERENCE_SHA256:
        raise RuntimeError("Dooms reference digest differs")

    ODT_PRESTAGE_ROOT.mkdir(mode=0o755)
    staged: dict[str, str] = {}
    for relative in PRESTAGE_CLOSURE:
        if relative == "reference/dooms_xnets_2504.02667.pdf":
            source = reference
        elif relative in capability_sources and relative not in ODT_SOURCE_OVERRIDES:
            source = CAPABILITY_STAGE_ROOT / relative
        else:
            source = SOURCE_ROOT / relative
        staged[relative] = _copy_physical(source, ODT_PRESTAGE_ROOT / relative)

    frozen_audit = audit_direct_only_launch(
        ODT_PRESTAGE_ROOT,
        tuple(ODT_PRESTAGE_ROOT / relative for relative in SOURCE_CLOSURE),
    )
    expected_source = {
        relative: staged[relative] for relative in SOURCE_CLOSURE
    }
    if frozen_audit["source_sha256"] != expected_source:
        raise RuntimeError("Frozen prestage source map differs from its static audit")
    for field in (
        "prohibited_calls_found",
        "prohibited_self_overlap_sites",
        "guarded_dormant_spectral_norm_sites",
    ):
        if frozen_audit[field] != []:
            raise RuntimeError(f"Frozen prestage static gate failed: {field}")

    manifest = {
        "schema": PRESTAGE_SCHEMA,
        "closure": dict(sorted(staged.items())),
        "bundle_sha256": canonical_sha256(dict(sorted(staged.items()))),
        "capability_source_manifest_sha256": file_sha256(capability_manifest_path),
        "dooms_reference_sha256": DOOMS_REFERENCE_SHA256,
    }
    manifest_path = ODT_PRESTAGE_ROOT / PRESTAGE_MANIFEST_PATH
    manifest_sha = _publish_json(manifest_path, manifest)
    ledger = {
        "schema": f"{PRESTAGE_SCHEMA}_ledger",
        "prestage_root": ODT_PRESTAGE_ROOT.as_posix(),
        "prestage_manifest": PRESTAGE_MANIFEST_PATH.as_posix(),
        "prestage_manifest_sha256": manifest_sha,
        "prestage_bundle_sha256": manifest["bundle_sha256"],
        "capability_stage_root": CAPABILITY_STAGE_ROOT.as_posix(),
        "capability_source_manifest_sha256": file_sha256(capability_manifest_path),
        "source_sha256": expected_source,
        "launch_sha256": {
            relative: staged[relative]
            for relative in PRESTAGE_CLOSURE
            if relative not in SOURCE_CLOSURE
            and relative != "reference/dooms_xnets_2504.02667.pdf"
        },
        "static_audit": frozen_audit,
        "no_job_submitted": True,
    }
    _publish_json(ODT_PRESTAGE_ROOT / PRESTAGE_LEDGER_PATH, ledger)
    for directory in sorted(
        (item for item in ODT_PRESTAGE_ROOT.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    ODT_PRESTAGE_ROOT.chmod(0o555)
    print(
        "PRESTAGED",
        json.dumps(
            {
                "root": ODT_PRESTAGE_ROOT.as_posix(),
                "manifest_sha256": manifest_sha,
                "bundle_sha256": manifest["bundle_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
