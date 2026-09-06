#!/usr/bin/env python3
"""Target-only 70-percent dimension-removal lane using the frozen full ODT code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


FROZEN_ROOT = Path("/work/joy/x-vla-capable-linear-b1c0-odt-v2")
SOURCE_LANE = Path("/work/joy/x-vla-capable-linear-b1c0-dooms-v1")
LANE_ROOT = Path("/work/joy/x-vla-capable-linear-b1c0-dooms-fast70-v1")
OUTPUT_ROOT = FROZEN_ROOT / "athena/results/capable_linear_b1c0_dooms_fast70_v1"
PROGRESS_ROOT = OUTPUT_ROOT / "progress"
FULL_RANK_OUTPUT = OUTPUT_ROOT / "full_rank_certificate.json"
IMPLEMENTATION_OUTPUT = OUTPUT_ROOT / "implementation_receipt.json"
FINAL_OUTPUT = OUTPUT_ROOT / "dooms_fast70_dimension_point.json"
ARTIFACT_ROOT = OUTPUT_ROOT / "uniform_actual70_tensor_network"
UNIT_OUTPUT = LANE_ROOT / "results/unit.json"
ROLLOUT_UNIT_OUTPUT = LANE_ROOT / "results/rollout_unit.json"
LANE_LEDGER = LANE_ROOT / "stage.sha256"
SOURCE_LEDGER = SOURCE_LANE / "stage.sha256"
FROZEN_LEDGER = FROZEN_ROOT / "athena/capable_linear_b1c0_stage.sha256"
ARTIFACT_MODULE = LANE_ROOT / "xvla/train/implicit_projective_dag_artifact.py"
EXPECTED_SOURCE_LEDGER_SHA256 = (
    "2a1171e79c7433da9d5f3e1329c663d30f0d74f85b21e8d9d7ad8cabcaf8a1d5"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee"
)
EXPECTED_ARTIFACT_MODULE_SHA256 = (
    "e302131d021ed8fac5b8ec5bb1d321b4f3ca035487f4ba7626159d914d0f4bd9"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ledger(path: Path, root: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing physical ledger: {path}")
    result: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(f"malformed ledger line {number}: {path}")
        digest, relative = fields
        relative = relative.lstrip("*")
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative in result
        ):
            raise RuntimeError(f"invalid ledger line {number}: {path}")
        candidate = root / relative
        resolved = candidate.resolve(strict=True)
        if candidate.is_symlink() or resolved == root or root not in resolved.parents:
            raise RuntimeError(f"unsafe ledger member: {relative}")
        if sha256(resolved) != digest:
            raise RuntimeError(f"ledger member changed: {relative}")
        result[relative] = digest
    return result


def verify_roots() -> dict[str, Any]:
    lane = read_ledger(LANE_LEDGER, LANE_ROOT)
    source = read_ledger(SOURCE_LEDGER, SOURCE_LANE)
    frozen = read_ledger(FROZEN_LEDGER, FROZEN_ROOT)
    required_lane = {
        Path(__file__).resolve().relative_to(LANE_ROOT).as_posix(),
        "xvla/train/implicit_projective_dag_artifact.py",
    }
    required_source = {
        "run_capable_linear_b1c0_dooms_sweep_v1.py",
        "run_capable_linear_b1c0_dooms_rollout_v1.py",
    }
    if not required_lane.issubset(lane) or not required_source.issubset(source):
        raise RuntimeError("fast lane omits a required authenticated source")
    if sha256(SOURCE_LEDGER) != EXPECTED_SOURCE_LEDGER_SHA256:
        raise RuntimeError("source ODT lane identity differs")
    if lane["xvla/train/implicit_projective_dag_artifact.py"] != (
        EXPECTED_ARTIFACT_MODULE_SHA256
    ):
        raise RuntimeError("serializer identity differs")
    if frozen.get("inputs/capable_linear_b1c0_checkpoint.pt") != (
        EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError("checkpoint identity differs")
    return {
        "lane_ledger_sha256": sha256(LANE_LEDGER),
        "source_lane_ledger_sha256": sha256(SOURCE_LEDGER),
        "frozen_stage_ledger_sha256": sha256(FROZEN_LEDGER),
        "lane_source_sha256": lane,
    }


def load_source(filename: str, name: str) -> Any:
    path = SOURCE_LANE / filename
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not load authenticated source: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def configure_sweep() -> Any:
    module = load_source(
        "run_capable_linear_b1c0_dooms_sweep_v1.py",
        "xvla_dooms_sweep_source_fast70_v1",
    )
    module.LANE_ROOT = LANE_ROOT
    module.OUTPUT_ROOT = OUTPUT_ROOT
    module.PROGRESS_ROOT = PROGRESS_ROOT
    module.FULL_RANK_OUTPUT = FULL_RANK_OUTPUT
    module.IMPLEMENTATION_OUTPUT = IMPLEMENTATION_OUTPUT
    module.FINAL_OUTPUT = FINAL_OUTPUT
    module.ARTIFACT_ROOT = ARTIFACT_ROOT
    module.LANE_LEDGER = LANE_LEDGER
    module.ARTIFACT_MODULE = ARTIFACT_MODULE
    module.UNIT_OUTPUT = UNIT_OUTPUT
    module.ROLLOUT_UNIT_OUTPUT = ROLLOUT_UNIT_OUTPUT
    module.REMOVAL_GRID = (0.0, 0.70)
    module.CONTROL_POINTS = frozenset((0.70,))
    module.verify_roots = verify_roots
    original_compute = module.compute_sweep

    def fast_compute(diagonal: Any, raw: Any, selected_plan: Any) -> Any:
        result = original_compute(diagonal, raw, selected_plan)
        if module.SWEEP_RECORD is None:
            raise RuntimeError("target-only sweep record is missing")
        module.SWEEP_RECORD["target_only_acceleration"] = True
        module.SWEEP_RECORD["omitted_nondecisive_uniform_points"] = [
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.80,
            0.90,
            0.95,
            0.975,
            0.99,
        ]
        return result

    module.compute_sweep = fast_compute
    return module


def configure_rollout() -> Any:
    module = load_source(
        "run_capable_linear_b1c0_dooms_rollout_v1.py",
        "xvla_dooms_rollout_source_fast70_v1",
    )
    module.LANE_ROOT = LANE_ROOT
    module.OUTPUT_ROOT = OUTPUT_ROOT
    module.SWEEP_RESULT = FINAL_OUTPUT
    module.ARTIFACT_MODULE = ARTIFACT_MODULE
    module.LANE_LEDGER = LANE_LEDGER
    module.UNIT_OUTPUT = ROLLOUT_UNIT_OUTPUT
    module.verify_roots = verify_roots
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("sweep-unit", "rollout-unit", "full", "smoke", "pilot"),
        required=True,
    )
    arguments = parser.parse_args()
    verify_roots()
    if arguments.mode in ("sweep-unit", "full"):
        module = configure_sweep()
        if arguments.mode == "sweep-unit":
            module.self_test()
        else:
            module.base.COMPACT_TRACE_RETENTION_TARGETS = ()
            module.run_full()
    else:
        module = configure_rollout()
        if arguments.mode == "rollout-unit":
            module.unit()
        else:
            module.evaluate(arguments.mode, 0, 0)


if __name__ == "__main__":
    main()
