#!/usr/bin/env python3
"""Create the exclusive byte-frozen v2 stage for compression experiments."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
ATHENA_MUTABLE_SOURCE_ROOT = Path("/work/joy/x-vla-workshop")
FROZEN_STAGE_ROOT = Path("/work/joy/x-vla-direct-odt-compression-v2")
EXPECTED_MANIFEST_SHA256 = {
    "athena/direct_odt_spectrum_scaling_sources.sha256": (
        "ebdb779779dd99fcd3b214663a0cfac187fd9d06cbed579a44f0326db792cbb0"
    ),
    "athena/direct_odt_truncation_tests_sources.sha256": (
        "7e1cb9d5be89300b212b1d4dc958d8565fbb22f580a23b6d090e6864dfb73f74"
    ),
}
EXPECTED_LAUNCH_SHA256 = {
    "athena/slurm_direct_odt_truncation_unit.sbatch": (
        "62952ccef5409e7688df954e626016edb25db80e6009eebb856d9e309a867130"
    ),
    "athena/slurm_direct_odt_spectrum_tiny.sbatch": (
        "61f074c2cbf125bfafebbe9557a4abbb04dbe78e0876dcd01eeb59ce992b9b31"
    ),
    "athena/slurm_direct_odt_spectrum_small.sbatch": (
        "d0890761a6178370efc89132db81eb1d52270df54d68a7b8c68cad9c5eed445f"
    ),
    "athena/slurm_direct_odt_spectrum_medium.sbatch": (
        "26514679cb1bf1272e321854e7db11f843799bf6fb2d0740bdc0cc832f0c895b"
    ),
    "athena/submit_direct_odt_compression_v2.sh": (
        "5e4707eef3d213ee32f03e59b29da31fd3acc4e71cb33edde86b96b38522f45b"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_physical(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"staging input is missing or nonphysical: {path}")


def _manifest_sources(relative_manifest: str) -> dict[str, str]:
    path = SOURCE_ROOT / relative_manifest
    _require_physical(path)
    if _sha256(path) != EXPECTED_MANIFEST_SHA256[relative_manifest]:
        raise RuntimeError(f"source manifest digest drifted: {relative_manifest}")
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(
                f"malformed source manifest {relative_manifest}:{line_number}"
            )
        digest, relative = fields
        relative = relative.lstrip("*")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise RuntimeError(
                f"invalid source digest {relative_manifest}:{line_number}"
            )
        source = SOURCE_ROOT / relative
        resolved = source.resolve(strict=True)
        if (
            source.is_symlink()
            or resolved == SOURCE_ROOT
            or SOURCE_ROOT not in resolved.parents
        ):
            raise RuntimeError("source manifest contains an unsafe path")
        normalized = resolved.relative_to(SOURCE_ROOT).as_posix()
        if normalized in result:
            raise RuntimeError("source manifest contains a duplicate path")
        _require_physical(source)
        if _sha256(source) != digest:
            raise RuntimeError(f"manifested source drifted: {normalized}")
        result[normalized] = digest
    return result


def main() -> None:
    if SOURCE_ROOT != ATHENA_MUTABLE_SOURCE_ROOT:
        raise RuntimeError(
            f"staging must run from {ATHENA_MUTABLE_SOURCE_ROOT}, got {SOURCE_ROOT}"
        )
    if os.path.lexists(FROZEN_STAGE_ROOT):
        raise FileExistsError(f"refusing existing stage root {FROZEN_STAGE_ROOT}")

    expected: dict[str, str] = {}
    for relative, digest in EXPECTED_MANIFEST_SHA256.items():
        for source_relative, source_digest in _manifest_sources(relative).items():
            previous = expected.setdefault(source_relative, source_digest)
            if previous != source_digest:
                raise RuntimeError("source manifests disagree on shared bytes")
        expected[relative] = digest
    for relative, digest in EXPECTED_LAUNCH_SHA256.items():
        source = SOURCE_ROOT / relative
        _require_physical(source)
        if _sha256(source) != digest:
            raise RuntimeError(f"launch file digest drifted: {relative}")
        expected[relative] = digest

    FROZEN_STAGE_ROOT.mkdir(mode=0o755)
    observed: dict[str, str] = {}
    for relative, digest in sorted(expected.items()):
        source = SOURCE_ROOT / relative
        destination = FROZEN_STAGE_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        _require_physical(destination)
        if _sha256(destination) != digest:
            raise RuntimeError(f"staged file differs from source: {relative}")
        destination.chmod(0o444)
        observed[relative] = digest

    result_directory = FROZEN_STAGE_ROOT / "athena/results"
    log_directory = FROZEN_STAGE_ROOT / "athena/logs"
    result_directory.mkdir(mode=0o755)
    log_directory.mkdir(mode=0o755)
    ledger = FROZEN_STAGE_ROOT / "athena/direct_odt_compression_stage.sha256"
    with ledger.open("x", encoding="utf-8") as stream:
        for relative, digest in sorted(observed.items()):
            stream.write(f"{digest}  {relative}\n")
        stream.flush()
        os.fsync(stream.fileno())
    ledger.chmod(0o444)

    for directory in sorted(
        (
            path
            for path in FROZEN_STAGE_ROOT.rglob("*")
            if path.is_dir() and path not in {result_directory, log_directory}
        ),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    FROZEN_STAGE_ROOT.chmod(0o555)

    for relative, digest in observed.items():
        staged = FROZEN_STAGE_ROOT / relative
        _require_physical(staged)
        if _sha256(staged) != digest or staged.stat().st_mode & 0o222:
            raise RuntimeError(f"staged source is not byte-frozen: {relative}")
    payload = {
        "object_kind": "direct_odt_compression_exclusive_stage_v2",
        "source_root": SOURCE_ROOT.as_posix(),
        "stage_root": FROZEN_STAGE_ROOT.as_posix(),
        "staged_file_count": len(observed),
        "staged_file_sha256": observed,
        "stage_ledger_sha256": _sha256(ledger),
        "results_writable": os.access(result_directory, os.W_OK),
        "logs_writable": os.access(log_directory, os.W_OK),
        "no_job_submitted": True,
        "next_command": (
            f"bash {FROZEN_STAGE_ROOT}/athena/submit_direct_odt_compression_v2.sh"
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
