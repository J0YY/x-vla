#!/usr/bin/env python3
"""Dooms-style uniform bond-dimension sweep on the frozen capable VLA lane."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import secrets
import stat
import sys
import time
from array import array
from pathlib import Path
from typing import Any, Mapping


FROZEN_ROOT = Path("/work/joy/x-vla-capable-linear-b1c0-odt-v2")
LANE_ROOT = Path("/work/joy/x-vla-capable-linear-b1c0-dooms-v1")
OUTPUT_ROOT = FROZEN_ROOT / "athena/results/capable_linear_b1c0_dooms_v1"
PROGRESS_ROOT = OUTPUT_ROOT / "progress"
FULL_RANK_OUTPUT = OUTPUT_ROOT / "full_rank_certificate.json"
IMPLEMENTATION_OUTPUT = OUTPUT_ROOT / "implementation_receipt.json"
FINAL_OUTPUT = OUTPUT_ROOT / "dooms_dimension_sweep.json"
ARTIFACT_ROOT = OUTPUT_ROOT / "uniform_actual70_tensor_network"
UNIT_OUTPUT = LANE_ROOT / "results/unit.json"
ROLLOUT_UNIT_OUTPUT = LANE_ROOT / "results/rollout_unit.json"
LANE_LEDGER = LANE_ROOT / "stage.sha256"
FROZEN_LEDGER = FROZEN_ROOT / "athena/capable_linear_b1c0_stage.sha256"
ARTIFACT_MODULE = LANE_ROOT / "xvla/train/implicit_projective_dag_artifact.py"

# Figure 2 axis used by the local paper reproduction. The physical point is
# selected from dimensions alone and never from replay or capability outcomes.
REMOVAL_GRID = (
    0.0,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.975,
    0.99,
)
CONTROL_POINTS = frozenset((0.50, 0.70, 0.90))
REQUESTED_PHYSICAL_REMOVAL = 0.70
MISSING_RANK = (1 << 32) - 1
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
    frozen = read_ledger(FROZEN_LEDGER, FROZEN_ROOT)
    runner_relative = Path(__file__).resolve().relative_to(LANE_ROOT).as_posix()
    required_lane = {
        runner_relative,
        "xvla/train/implicit_projective_dag_artifact.py",
    }
    if not required_lane.issubset(lane):
        raise RuntimeError("lane ledger omits the runner or serializer")
    if lane["xvla/train/implicit_projective_dag_artifact.py"] != (
        EXPECTED_ARTIFACT_MODULE_SHA256
    ):
        raise RuntimeError("serializer identity differs from its tested milestone")
    if frozen.get("inputs/capable_linear_b1c0_checkpoint.pt") != (
        EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError("frozen stage checkpoint identity differs")
    return {
        "lane_ledger_sha256": sha256(LANE_LEDGER),
        "frozen_stage_ledger_sha256": sha256(FROZEN_LEDGER),
        "lane_source_sha256": lane,
    }


def publish(path: Path, payload: Mapping[str, Any]) -> str:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing existing output: {path}")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise RuntimeError("output parent must be a physical directory")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(12)}.tmp")
    linked = False
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if not stat.S_ISREG(os.lstat(temporary).st_mode):
            raise RuntimeError("temporary result is not a regular file")
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        os.chmod(path, 0o444, follow_symlinks=False)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if sha256(path) != digest or os.lstat(path).st_mode & 0o222:
            raise RuntimeError("published result identity differs")
        return digest
    except BaseException:
        if linked and os.path.lexists(path):
            path.unlink()
        raise
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


if FROZEN_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, FROZEN_ROOT.as_posix())

import scripts.run_capable_linear_direct_odt_spectrum as base
from xvla.train.direct_odt_truncation import CompactRankPlan


ORIGINAL_BANK = base.CompactRankBank
ORIGINAL_PREFIX = base.evaluate_diagonal_prefix_quotient
ORIGINAL_SUFFIX = base.evaluate_diagonal_suffix_quotient
ORIGINAL_TRUNCATE = base.truncate_diagonal_prefixes

ACTIVE_BANK: "DoomsRankBank | None" = None
CAPTURED_MATERIALIZATION: Any = None
CAPTURED_RAW_SAMPLE: Any = None
CAPTURED_SELECTED_OUTPUT: Any = None
SWEEP_RECORD: dict[str, Any] | None = None


class DoomsRankBank(ORIGINAL_BANK):
    """Add uniform fractional ranks without changing Algorithm 2 or Algorithm 3."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        global ACTIVE_BANK
        if ACTIVE_BANK is not None:
            raise RuntimeError("more than one rank bank was constructed")
        ACTIVE_BANK = self

    def _rank_summary(
        self, nominal_fraction: float
    ) -> tuple[Any, Any, int, int, float]:
        if not self.complete:
            raise RuntimeError("uniform ranks are unavailable before Algorithm 3")
        if (
            type(nominal_fraction) is not float
            or not math.isfinite(nominal_fraction)
            or not 0.0 <= nominal_fraction <= 1.0
        ):
            raise ValueError("nominal removal fraction must be a finite float in [0,1]")
        expected = base.np.frombuffer(self._expected, dtype=base.np.uint8).astype(bool)
        dimensions = base.np.frombuffer(self._dimensions, dtype=base.np.uint32)
        selected_dimensions = dimensions[expected].astype(base.np.float64)
        if selected_dimensions.size != self._node_count or bool(
            (selected_dimensions < 1).any()
        ):
            raise RuntimeError("rank-bank dimensions are incomplete")
        retained = base.np.rint(
            (1.0 - nominal_fraction) * selected_dimensions
        ).astype(base.np.uint32)
        retained = base.np.maximum(retained, base.np.uint32(1))
        original = int(selected_dimensions.astype(base.np.uint64).sum())
        kept = int(retained.astype(base.np.uint64).sum())
        actual = 1.0 - kept / original
        return expected, retained, original, kept, actual

    def _rank_arrays(self, nominal_fraction: float) -> tuple[array, int, int, float]:
        expected, retained, original, kept, actual = self._rank_summary(
            nominal_fraction
        )
        dense = base.np.full(
            len(self._expected), MISSING_RANK, dtype=base.np.uint32
        )
        dense[expected] = retained
        ranks = array("I")
        ranks.frombytes(dense.tobytes(order="C"))
        if ranks.itemsize != 4 or len(ranks) != len(self._expected):
            raise RuntimeError("rank-array representation differs")
        return ranks, original, kept, actual

    def uniform_plan(
        self,
        nominal_fraction: float,
        *,
        requested_actual_fraction: float | None = None,
    ) -> CompactRankPlan:
        ranks, original, retained, actual = self._rank_arrays(nominal_fraction)
        plan = CompactRankPlan(
            ranks,
            self._dimensions,
            self._expected,
            float(
                actual
                if requested_actual_fraction is None
                else requested_actual_fraction
            ),
            self._node_count,
            self._network_identity,
        )
        plan.dooms_uniform_plan = True
        plan.dooms_nominal_fraction = nominal_fraction
        plan.dooms_actual_fraction = actual
        plan.dooms_original_dimensions = original
        plan.dooms_retained_dimensions = retained
        plan.dooms_requested_actual_fraction = requested_actual_fraction
        return plan

    def closest_actual_plan(self, requested: float) -> CompactRankPlan:
        if type(requested) is not float or not 0.0 <= requested < 1.0:
            raise ValueError("requested actual removal must be a float in [0,1)")
        low = 0.0
        high = 1.0
        candidates = {low, high, requested}
        for _ in range(48):
            midpoint = (low + high) / 2.0
            candidates.add(midpoint)
            _expected, _retained_values, _original, _retained, actual = (
                self._rank_summary(midpoint)
            )
            if actual < requested:
                low = midpoint
            else:
                high = midpoint
        scored = []
        for nominal in candidates:
            _expected, _retained_values, _original, _retained, actual = (
                self._rank_summary(nominal)
            )
            scored.append((abs(actual - requested), actual > requested, nominal, actual))
        _distance, _overshoots, nominal, _actual = min(scored)
        return self.uniform_plan(
            float(nominal), requested_actual_fraction=requested
        )

    def plan(self, target: float) -> CompactRankPlan:
        if float(target) == REQUESTED_PHYSICAL_REMOVAL:
            return self.closest_actual_plan(REQUESTED_PHYSICAL_REMOVAL)
        return super().plan(target)


def plan_record(plan: CompactRankPlan) -> dict[str, Any]:
    return {
        "nominal_fraction_dimensions_removed_per_bond": (
            plan.dooms_nominal_fraction
        ),
        "actual_fraction_unique_bond_dimensions_removed": (
            plan.dooms_actual_fraction
        ),
        "original_unique_bond_dimensions": plan.dooms_original_dimensions,
        "retained_unique_bond_dimensions": plan.dooms_retained_dimensions,
        "removed_unique_bond_dimensions": (
            plan.dooms_original_dimensions - plan.dooms_retained_dimensions
        ),
        "minimum_retained_rank_per_bond": 1,
        "rounding": "max(1, round((1-p)*d_i)); ties-to-even",
    }


def compute_sweep(diagonal: Any, raw: Any, selected_plan: CompactRankPlan) -> Any:
    global CAPTURED_RAW_SAMPLE, CAPTURED_SELECTED_OUTPUT, SWEEP_RECORD
    if SWEEP_RECORD is not None or ACTIVE_BANK is None:
        raise RuntimeError("uniform sweep interception state differs")
    bank = ACTIVE_BANK
    full_plan = bank.uniform_plan(0.0)
    baseline = ORIGINAL_PREFIX(diagonal, raw, full_plan)
    full_storage = base.implicit_storage_elements(diagonal.network)
    curves: list[dict[str, Any]] = []
    for fraction in REMOVAL_GRID:
        plan = bank.uniform_plan(float(fraction))
        started = time.perf_counter()
        candidate = baseline if fraction == 0.0 else ORIGINAL_PREFIX(diagonal, raw, plan)
        point = {
            **plan_record(plan),
            "projected_storage_elements": base.projected_prefix_storage_elements(
                diagonal, plan
            ),
            "prefix_action_relative_error": base._relative(candidate, baseline),
            "prefix_action_coordinate_max_absolute_error": [
                float(value)
                for value in (candidate - baseline)
                .reshape(-1, 8, 7)
                .abs()
                .amax(dim=(0, 1))
                .detach()
                .cpu()
                .tolist()
            ],
            "prefix_evaluation_seconds": time.perf_counter() - started,
        }
        point["projected_storage_fraction"] = (
            point["projected_storage_elements"] / full_storage
        )
        if fraction in CONTROL_POINTS:
            control_started = time.perf_counter()
            try:
                control = ORIGINAL_SUFFIX(diagonal, raw, plan)
            except ValueError as error:
                point["matched_width_trailing_control"] = {
                    "valid_projective_chart": False,
                    "rejection": str(error),
                    "evaluation_seconds": time.perf_counter() - control_started,
                }
            else:
                point["matched_width_trailing_control"] = {
                    "valid_projective_chart": True,
                    "action_relative_error": base._relative(control, baseline),
                    "evaluation_seconds": time.perf_counter() - control_started,
                }
        curves.append(point)
    selected_started = time.perf_counter()
    selected = ORIGINAL_PREFIX(diagonal, raw, selected_plan)
    selected_record = {
        **plan_record(selected_plan),
        "requested_actual_fraction_unique_bond_dimensions_removed": (
            REQUESTED_PHYSICAL_REMOVAL
        ),
        "selection_used_replay_or_capability_outcomes": False,
        "prefix_action_relative_error": base._relative(selected, baseline),
        "projected_storage_elements": base.projected_prefix_storage_elements(
            diagonal, selected_plan
        ),
        "evaluation_seconds": time.perf_counter() - selected_started,
    }
    selected_record["projected_storage_fraction"] = (
        selected_record["projected_storage_elements"] / full_storage
    )
    if abs(
        selected_record["actual_fraction_unique_bond_dimensions_removed"]
        - REQUESTED_PHYSICAL_REMOVAL
    ) > 0.005:
        selected_record["within_half_percentage_point_of_requested"] = False
    else:
        selected_record["within_half_percentage_point_of_requested"] = True
    CAPTURED_RAW_SAMPLE = {
        key: value[:1].detach().clone() for key, value in raw.items()
    }
    CAPTURED_SELECTED_OUTPUT = selected[:1].detach().clone()
    SWEEP_RECORD = {
        "schema": "xvla_capable_linear_b1c0_dooms_dimension_sweep_v1",
        "schedule": "uniform fractional leading-rank truncation at every unique ODT bond",
        "paper_axis": "Dimensions removed (%)",
        "paper_reference_sha256": (
            "a97f59a91d90ffde56bc22bf0b3c79e0891ff88197ca9d7670331e4208277559"
        ),
        "dimension_denominator": (
            "sum of output dimensions over unique shared-DAG tensor bonds; repeated consumer "
            "occurrences are sliced but are not counted again"
        ),
        "homogeneous_coordinate_convention": (
            "all projective bond coordinates are counted, matching the local Figure 2 "
            "reproduction convention"
        ),
        "full_storage_elements": full_storage,
        "grid": list(REMOVAL_GRID),
        "curves": curves,
        "physical_selection": selected_record,
        "replay_panel_samples": int(baseline.shape[0]),
        "claim_boundary": (
            "This sweep measures action fidelity on the frozen two-input exactness panel. "
            "It is not LIBERO accuracy. Closed-loop accuracy is a separate dependent run."
        ),
    }
    return selected


def prefix_interceptor(diagonal: Any, raw: Any, plan: Mapping[int, int]) -> Any:
    if getattr(plan, "dooms_requested_actual_fraction", None) is not None:
        return compute_sweep(diagonal, raw, plan)
    return ORIGINAL_PREFIX(diagonal, raw, plan)


def truncate_interceptor(
    diagonal: Any, plan: Mapping[int, int], *, copy_network: bool = True
) -> Any:
    global CAPTURED_MATERIALIZATION
    result = ORIGINAL_TRUNCATE(diagonal, plan, copy_network=copy_network)
    if getattr(plan, "dooms_requested_actual_fraction", None) is not None:
        if CAPTURED_MATERIALIZATION is not None:
            raise RuntimeError("physical materialization was captured twice")
        CAPTURED_MATERIALIZATION = result
    return result


def load_artifact_module() -> Any:
    if sha256(ARTIFACT_MODULE) != EXPECTED_ARTIFACT_MODULE_SHA256:
        raise RuntimeError("serializer changed before loading")
    name = "xvla_dooms_artifact_v1"
    specification = importlib.util.spec_from_file_location(name, ARTIFACT_MODULE)
    if specification is None or specification.loader is None:
        raise RuntimeError("serializer module could not be loaded")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def self_test() -> None:
    from xvla.train.implicit_sparse_projective_odt import CompactSpectrumRecord

    records = (
        CompactSpectrumRecord(
            uid=0,
            label="test.0",
            dimension=257,
            environment_binary_exponent=0,
            leading_value_mantissa=1.0,
            trace_mantissa=1.0,
            negative_roundoff_mass_mantissa=0.0,
            trace_retention_ranks=(257, 257, 257, 257, 257),
            relative_floor_ranks=(257, 257, 257, 257, 257, 257, 257),
            trace_tail_mantissas=(0.0, 0.0, 0.0, 0.0, 0.0),
            zero_trace=False,
            tie_extensions=(0, 0, 0, 0, 0),
        ),
        CompactSpectrumRecord(
            uid=1,
            label="test.1",
            dimension=257,
            environment_binary_exponent=0,
            leading_value_mantissa=1.0,
            trace_mantissa=1.0,
            negative_roundoff_mass_mantissa=0.0,
            trace_retention_ranks=(257, 257, 257, 257, 257),
            relative_floor_ranks=(257, 257, 257, 257, 257, 257, 257),
            trace_tail_mantissas=(0.0, 0.0, 0.0, 0.0, 0.0),
            zero_trace=False,
            tie_extensions=(0, 0, 0, 0, 0),
        ),
    )
    bank = DoomsRankBank((0, 1), network_identity=7)
    for record in records:
        bank(record)
    plan = bank.uniform_plan(0.70)
    if tuple(plan[uid] for uid in plan) != (77, 77):
        raise RuntimeError("paper-width uniform-rank regression failed")
    if plan.dooms_original_dimensions != 514 or plan.dooms_retained_dimensions != 154:
        raise RuntimeError("dimension-count regression failed")
    closest = bank.closest_actual_plan(0.70)
    if abs(closest.dooms_actual_fraction - 0.70) > 1.0 / 514.0:
        raise RuntimeError("closest-actual selection regression failed")
    payload = {
        "schema": "xvla_capable_linear_b1c0_dooms_dimension_unit_v1",
        "paper_width": 256,
        "projective_bond_dimension": 257,
        "nominal_removed_fraction": 0.70,
        "retained_rank": 77,
        "actual_removed_fraction": plan.dooms_actual_fraction,
        "closest_actual_removed_fraction": closest.dooms_actual_fraction,
        "roots": verify_roots(),
        "all_gates_pass": True,
    }
    digest = publish(UNIT_OUTPUT, payload)
    print(json.dumps({"unit_sha256": digest}, sort_keys=True), flush=True)


def run_full() -> None:
    global CAPTURED_MATERIALIZATION
    roots_before = verify_roots()
    if not UNIT_OUTPUT.is_file() or not ROLLOUT_UNIT_OUTPUT.is_file():
        raise RuntimeError("unit results are missing")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    PROGRESS_ROOT.mkdir()
    base.RESULT_DIRECTORY = OUTPUT_ROOT
    base.PROGRESS_DIRECTORY = PROGRESS_ROOT
    base.FULL_RANK_OUTPUT = FULL_RANK_OUTPUT
    base.COMPRESSION_OUTPUT = IMPLEMENTATION_OUTPUT
    base.PHYSICAL_TARGET = REQUESTED_PHYSICAL_REMOVAL
    base.CompactRankBank = DoomsRankBank
    base.evaluate_diagonal_prefix_quotient = prefix_interceptor
    base.truncate_diagonal_prefixes = truncate_interceptor
    started = time.perf_counter()
    base._full_mode()
    if (
        CAPTURED_MATERIALIZATION is None
        or CAPTURED_RAW_SAMPLE is None
        or CAPTURED_SELECTED_OUTPUT is None
        or SWEEP_RECORD is None
    ):
        raise RuntimeError("full run did not capture the selected materialization")
    implementation = json.loads(IMPLEMENTATION_OUTPUT.read_text())
    if (
        implementation.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
        or implementation.get("all_full_rank_gates_pass") is not True
        or implementation.get("all_compression_gates_pass") is not True
    ):
        raise RuntimeError("underlying full-rank or materialization gates failed")
    selected = SWEEP_RECORD["physical_selection"]
    materialized = CAPTURED_MATERIALIZATION
    if (
        materialized.original_bond_dimensions
        != selected["original_unique_bond_dimensions"]
        or materialized.retained_bond_dimensions
        != selected["retained_unique_bond_dimensions"]
        or materialized.expected_occurrence_slices
        != materialized.applied_occurrence_slices
    ):
        raise RuntimeError("physical truncation disagrees with the selected rank plan")
    materialized.network.claim_boundary = (
        "Physically sliced direct-ODT prefix at a dimensions-only selection closest to "
        "70 percent of unique bond directions. Task capability is not asserted by this artifact."
    )
    artifact_module = load_artifact_module()
    artifact_started = time.perf_counter()
    receipt = artifact_module.export_implicit_projective_dag_artifact(
        materialized.network, ARTIFACT_ROOT
    )
    artifact_seconds = time.perf_counter() - artifact_started
    artifact_record = {
        "path": ARTIFACT_ROOT.as_posix(),
        "manifest_sha256": receipt.manifest_sha256,
        "shard_sha256": dict(receipt.shard_sha256),
        "node_count": receipt.node_count,
        "tensor_count": receipt.tensor_count,
        "raw_byte_count": receipt.raw_byte_count,
        "export_seconds": artifact_seconds,
        "eager_reload_completed": False,
        "closed_loop_execution_completed": False,
    }
    final = {
        **SWEEP_RECORD,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "full_rank_certificate": {
            "path": FULL_RANK_OUTPUT.as_posix(),
            "sha256": sha256(FULL_RANK_OUTPUT),
        },
        "implementation_receipt": {
            "path": IMPLEMENTATION_OUTPUT.as_posix(),
            "sha256": sha256(IMPLEMENTATION_OUTPUT),
        },
        "physical_materialization": {
            "original_storage_elements": materialized.original_storage_elements,
            "truncated_storage_elements": materialized.truncated_storage_elements,
            "allocated_storage_elements": materialized.allocated_storage_elements,
            "reduced_bonds": materialized.reduced_bonds,
            "expected_occurrence_slices": materialized.expected_occurrence_slices,
            "applied_occurrence_slices": materialized.applied_occurrence_slices,
        },
        "serialized_tensor_network": artifact_record,
        "roots": roots_before,
        "roots_unchanged_after": verify_roots() == roots_before,
        "elapsed_seconds": time.perf_counter() - started,
        "all_full_rank_and_materialization_gates_pass": True,
        "accuracy_curve_completed": False,
    }
    if not final["roots_unchanged_after"]:
        raise RuntimeError("source roots changed during the full run")
    digest = publish(FINAL_OUTPUT, final)
    print(json.dumps({"dooms_dimension_sweep_sha256": digest}, sort_keys=True), flush=True)
    CAPTURED_MATERIALIZATION = None
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("self-test", "full"), required=True)
    arguments = parser.parse_args()
    if arguments.mode == "self-test":
        self_test()
    else:
        run_full()


if __name__ == "__main__":
    main()
