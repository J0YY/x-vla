#!/usr/bin/env python3
"""One unchanged canonical sweep, then six independently executable prefixes.

The hook terminates the old driver only after its full-rank certificate has
passed and been durably published. It does not intercept any factorization.
Unwinding that driver releases its model and multi-million-step Python ledger
before the physical exporter starts. No compression certificate is fabricated.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any

FROZEN_ROOT = Path("/work/joy/x-vla-capable-linear-b1c0-odt-v2")
LANE_ROOT = Path("/work/joy/x-vla-odt-dimension-curve-v2")
OUTPUT_ROOT = FROZEN_ROOT / "athena/results/odt_dimension_curve_v2"
EXPECTED_FROZEN_LEDGER = "e4716a413bc742943b623ddebd39783bb54c40558b59446c99998aa07892b8c3"
CHECKPOINT_SHA256 = "b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee"
EXPECTED_SERIALIZER = "e302131d021ed8fac5b8ec5bb1d321b4f3ca035487f4ba7626159d914d0f4bd9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_ledger(root: Path, ledger: Path, expected: str) -> dict[str, str]:
    if ledger.is_symlink() or sha256(ledger) != expected:
        raise RuntimeError(f"ledger identity differs: {ledger}")
    entries: dict[str, str] = {}
    for line in ledger.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        candidate = root / relative
        resolved = candidate.resolve(strict=True)
        if (relative in entries or candidate.is_symlink() or root not in resolved.parents
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or sha256(resolved) != digest):
            raise RuntimeError(f"invalid authenticated member: {relative}")
        entries[relative] = digest
    if not entries:
        raise RuntimeError("empty source ledger")
    return entries


def load_module(name: str, path: Path) -> Any:
    if name in sys.modules:
        raise RuntimeError(f"refusing to replace an already imported module: {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FullRankReady(Exception):
    """Controlled handoff, never a numerical fallback."""


def read_certificate(path: Path) -> tuple[dict[str, Any], str]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_mode & 0o222:
        raise RuntimeError("certificate must be an immutable unlinked physical file")
    encoded = path.read_bytes()
    after = path.lstat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
        raise RuntimeError("certificate changed during its authenticated read")
    return json.loads(encoded), hashlib.sha256(encoded).hexdigest()


def certified_handoff(base: Any, certificate_path: Path) -> dict[str, Any]:
    original_bank = base.CompactRankBank
    original_diagonalize = base.diagonalize_implicit_dag_full_rank
    captured: dict[str, Any] = {}

    def capture_diagonal(*args: Any, **kwargs: Any) -> Any:
        if captured:
            raise RuntimeError("more than one diagonalization in the handoff")
        result = original_diagonalize(*args, **kwargs)
        raw = kwargs["replay_inputs"]
        captured["diagonal"] = result
        captured["raw"] = ({key: value[:1].detach().clone() for key, value in raw.items()}
                           if isinstance(raw, dict) else raw[:1].detach().clone())
        return result

    class CaptureBank(original_bank):
        def finish(self) -> Any:
            census = super().finish()
            if "diagonal" not in captured or not certificate_path.is_file():
                raise RuntimeError("full-rank certificate must precede ladder handoff")
            certificate, certificate_sha256 = read_certificate(certificate_path)
            gates = certificate.get("full_rank_gates", {})
            if (certificate.get("all_full_rank_gates_pass") is not True or not gates
                    or any(value is not True for value in gates.values())
                    or certificate.get("schema") != "xvla_capable_linear_b1c0_full_rank_direct_odt_v1"
                    or certificate.get("canonical_direct_odt_algorithms_1_to_3_completed") is not True
                    or certificate.get("runtime_guard", {}).get("prohibited_attempt_count") != 0
                    or certificate.get("checkpoint", {}).get("checkpoint_sha256") != CHECKPOINT_SHA256):
                raise RuntimeError("full-rank certificate gates or checkpoint differ")
            captured["certificate_sha256"] = certificate_sha256
            captured["compact_census"] = census
            raise FullRankReady()

    base.CompactRankBank = CaptureBank
    base.diagonalize_implicit_dag_full_rank = capture_diagonal
    completed = False
    try:
        try:
            base._full_mode()
        except FullRankReady:
            completed = True
    finally:
        base.CompactRankBank = original_bank
        base.diagonalize_implicit_dag_full_rank = original_diagonalize
    if not completed or "certificate_sha256" not in captured:
        raise RuntimeError("canonical driver returned without the certified handoff")
    gc.collect()
    return captured


def publish(path: Path, payload: dict[str, Any]) -> str:
    encoded = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return hashlib.sha256(encoded).hexdigest()


def run(expected_lane_ledger: str) -> None:
    lane_ledger = LANE_ROOT / "stage.sha256"
    frozen_ledger = FROZEN_ROOT / "athena/capable_linear_b1c0_stage.sha256"
    frozen = verify_ledger(FROZEN_ROOT, frozen_ledger, EXPECTED_FROZEN_LEDGER)
    lane = verify_ledger(LANE_ROOT, lane_ledger, expected_lane_ledger)
    required = {
        "athena/run_direct_odt_dimension_curve_v1.py",
        "xvla/train/direct_odt_dimension_ladder.py",
        "xvla/train/implicit_projective_dag_artifact.py",
        "xvla/train/implicit_projective_prefix_artifact.py",
        "tests/test_direct_odt_dimension_ladder.py",
        "tests/test_direct_odt_dimension_curve_runner.py",
        "scripts/run_direct_odt_dimension_ladder_tests.py",
    }
    if not required.issubset(lane):
        raise RuntimeError("new lane is missing authenticated sources")
    if (frozen.get("inputs/capable_linear_b1c0_checkpoint.pt") != CHECKPOINT_SHA256
            or lane["xvla/train/implicit_projective_dag_artifact.py"] != EXPECTED_SERIALIZER):
        raise RuntimeError("checkpoint or frozen serializer changed")
    for relative in lane.keys() & frozen.keys():
        if relative.startswith(("xvla/", "scripts/odt_direct_only_compliance")) and lane[relative] != frozen[relative]:
            raise RuntimeError(f"test dependency differs from frozen production source: {relative}")
    tests = json.loads((LANE_ROOT / "results/unit.json").read_text())
    if tests.get("pytest_exit_code") != 0 or tests.get("stage_ledger_sha256") != expected_lane_ledger:
        raise RuntimeError("this exact stage has no passing unit receipt")
    sys.path.insert(0, str(FROZEN_ROOT))
    base = importlib.import_module("scripts.run_capable_linear_direct_odt_spectrum")
    load_module("xvla.train.implicit_projective_dag_artifact", LANE_ROOT / "xvla/train/implicit_projective_dag_artifact.py")
    load_module("xvla.train.implicit_projective_prefix_artifact", LANE_ROOT / "xvla/train/implicit_projective_prefix_artifact.py")
    ladder = load_module("xvla.train.direct_odt_dimension_ladder", LANE_ROOT / "xvla/train/direct_odt_dimension_ladder.py")
    new_audit = base.audit_direct_only_launch(LANE_ROOT, (
        LANE_ROOT / "athena/run_direct_odt_dimension_curve_v1.py",
        LANE_ROOT / "xvla/train/direct_odt_dimension_ladder.py",
        LANE_ROOT / "tests/test_direct_odt_dimension_ladder.py",
    ))
    for key in ("prohibited_calls_found", "prohibited_self_overlap_sites",
                "guarded_dormant_spectral_norm_sites", "duplicate_top_level_definition_sites"):
        if new_audit[key]:
            raise RuntimeError(f"new source audit rejected {key}")
    OUTPUT_ROOT.mkdir(parents=False, exist_ok=False)
    (OUTPUT_ROOT / "progress").mkdir()
    base.RESULT_DIRECTORY = OUTPUT_ROOT
    base.PROGRESS_DIRECTORY = OUTPUT_ROOT / "progress"
    base.FULL_RANK_OUTPUT = OUTPUT_ROOT / "full_rank_certificate.json"
    base.COMPRESSION_OUTPUT = OUTPUT_ROOT / "unused_legacy_compression.json"
    publish(OUTPUT_ROOT / "launch.json", {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "stage_ledger_sha256": expected_lane_ledger,
        "frozen_ledger_sha256": EXPECTED_FROZEN_LEDGER,
        "requested_dimensions_removed": list(ladder.REMOVALS),
        "dimension_denominator": ladder.DENOMINATOR,
        "static_audit": new_audit,
        "started_unix": time.time(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "unit_receipt_sha256": sha256(LANE_ROOT / "results/unit.json"),
    })
    captured = certified_handoff(base, base.FULL_RANK_OUTPUT)
    base._assert_sources()
    verify_ledger(LANE_ROOT, lane_ledger, expected_lane_ledger)
    print(json.dumps({"event": "full_rank_certified_handoff", "sha256": captured["certificate_sha256"]}), flush=True)

    def ready(record: dict[str, Any]) -> None:
        base.assert_direct_only_runtime_guard(base.direct_only_runtime_report())
        verify_ledger(LANE_ROOT, lane_ledger, expected_lane_ledger)
        print(json.dumps({"event": "physical_rung_ready", "record": record}, sort_keys=True), flush=True)

    result = ladder.export_dimension_ladder(
        captured["diagonal"], captured["raw"], OUTPUT_ROOT / "ladder",
        full_rank_certificate_sha256=captured["certificate_sha256"],
        checkpoint_sha256=CHECKPOINT_SHA256, on_rung=ready,
    )
    runtime = base.assert_direct_only_runtime_guard(base.direct_only_runtime_report())
    base._assert_sources()
    verify_ledger(LANE_ROOT, lane_ledger, expected_lane_ledger)
    publish(OUTPUT_ROOT / "export_complete.json", {
        "ladder_receipt_sha256": result["receipt_sha256"],
        "stage_ledger_sha256": expected_lane_ledger,
        "runtime_guard": runtime, "closed_loop_completed": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-ledger-sha256", required=True)
    arguments = parser.parse_args()
    run(arguments.stage_ledger_sha256)
