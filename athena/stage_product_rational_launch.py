#!/usr/bin/env python3
"""Stage an exclusive byte-frozen ProductRoutingHead capability checkout on Athena."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from athena.product_rational_protocol import (
    AUTHENTICATED_INPUTS,
    CACHE_PATH,
    CACHE_SHA256,
    DEFAULT_RUN_ROOT,
    FROZEN_STAGE_ROOT,
    LAUNCH_CLOSURE,
    SOURCE_CLOSURE,
)

ATHENA_MUTABLE_SOURCE_ROOT = Path("/work/joy/x-vla-workshop")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_physical_source(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Staging source is missing or nonphysical: {path}")


def main() -> None:
    if SOURCE_ROOT != ATHENA_MUTABLE_SOURCE_ROOT:
        raise RuntimeError(
            f"Staging must run from {ATHENA_MUTABLE_SOURCE_ROOT}, got {SOURCE_ROOT}"
        )
    if os.path.lexists(FROZEN_STAGE_ROOT):
        raise FileExistsError(f"Refusing existing stage root {FROZEN_STAGE_ROOT}")
    paths = tuple(sorted(set(SOURCE_CLOSURE + AUTHENTICATED_INPUTS + LAUNCH_CLOSURE)))
    observed: dict[str, str] = {}
    FROZEN_STAGE_ROOT.mkdir(mode=0o755)
    for relative in paths:
        source = SOURCE_ROOT / relative
        require_physical_source(source)
        destination = FROZEN_STAGE_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if destination.is_symlink() or not destination.is_file():
            raise RuntimeError(f"Staged lane file is nonphysical: {relative}")
        source_sha = file_sha256(source)
        if file_sha256(destination) != source_sha:
            raise RuntimeError(f"Staged lane file differs: {relative}")
        observed[relative] = source_sha
        destination.chmod(0o555 if os.access(source, os.X_OK) else 0o444)
    cache_source = SOURCE_ROOT / CACHE_PATH
    require_physical_source(cache_source)
    if file_sha256(cache_source) != CACHE_SHA256:
        raise RuntimeError("Mutable-checkout cache SHA-256 differs before staging")
    cache_destination = FROZEN_STAGE_ROOT / CACHE_PATH
    cache_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_source, cache_destination)
    if cache_destination.is_symlink() or file_sha256(cache_destination) != CACHE_SHA256:
        raise RuntimeError("Copied staged cache identity differs")
    cache_independent_inode = cache_source.stat().st_ino != cache_destination.stat().st_ino
    if not cache_independent_inode:
        raise RuntimeError("Staged cache is not an independent physical copy")
    cache_destination.chmod(0o444)
    (FROZEN_STAGE_ROOT / "logs").mkdir(mode=0o755)
    run_root = FROZEN_STAGE_ROOT / DEFAULT_RUN_ROOT
    for relative in (
        "checkpoints",
        "gates",
        "manifest",
        "metadata",
        "results",
        "training_only",
    ):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    writable_directories = {
        FROZEN_STAGE_ROOT / "logs",
        *(
            run_root / relative
            for relative in (
                "checkpoints",
                "gates",
                "manifest",
                "metadata",
                "results",
                "training_only",
            )
        ),
    }
    locked_directories = sorted(
        path.relative_to(FROZEN_STAGE_ROOT).as_posix()
        for path in FROZEN_STAGE_ROOT.rglob("*")
        if path.is_dir() and path not in writable_directories
    )
    payload = {
        "schema": "xvla_product_pade_rational_vla_v2_stage",
        "source_root": SOURCE_ROOT.as_posix(),
        "stage_root": FROZEN_STAGE_ROOT.as_posix(),
        "run_root": run_root.as_posix(),
        "staged_file_sha256": observed,
        "cache_sha256": CACHE_SHA256,
        "cache_independent_inode": cache_independent_inode,
        "locked_source_directories": locked_directories,
        "writable_output_directories": sorted(
            path.relative_to(FROZEN_STAGE_ROOT).as_posix()
            for path in writable_directories
        ),
        "no_manifest_created": True,
        "no_job_submitted": True,
        "next_command": (
            "/users/joy/miniconda3/envs/safesae-openvla/bin/python -B "
            f"{FROZEN_STAGE_ROOT}/athena/prepare_product_rational_launch.py "
            f"--action create-manifest --run-root {DEFAULT_RUN_ROOT}"
        ),
    }
    ledger = FROZEN_STAGE_ROOT / "stage_ledger.json"
    with ledger.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    ledger.chmod(0o444)
    for path in sorted(
        (path for path in FROZEN_STAGE_ROOT.rglob("*") if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        path.chmod(0o755 if path in writable_directories else 0o555)
    FROZEN_STAGE_ROOT.chmod(0o555)
    print("STAGED", json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
