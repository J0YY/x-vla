"""Persist nested prefixes of one certified canonical ODT basis.

The reported denominator counts unique nonleaf, nonroot tensor output bonds.
Physical ingress leaves and the root-to-observable bond remain full width.  A
single full-rank certificate authorizes the first transition.  Later transitions
are authenticated descendants, never represented as new full-rank ODT results.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from xvla.train.direct_odt_truncation import (
    _certified_network,
    _compact_copy,
    implicit_allocated_storage_elements,
    implicit_storage_elements,
)
from xvla.train.implicit_projective_prefix_artifact import (
    export_implicit_projective_prefix_artifact,
)
from xvla.train.odt_engine_v2.graph import (
    _prepare_physical_input,
    _validate_network,
    _walk_unique,
    evaluate_projective_boundary,
)
from xvla.train.odt_engine_v2.ops import (
    _core_apply,
    _normalize_projective_batch,
)
from xvla.train.odt_engine_v2.types import (
    CPBinaryCore,
    DiagonalImplicitDAG,
    ImplicitNode,
    ImplicitProjectiveDAG,
    ReducedQRBinaryCore,
    UnaryCore,
)


REMOVALS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
SCHEMA = "xvla_direct_odt_internal_dimension_ladder_v1"
DENOMINATOR = "unique_nonleaf_nonroot_tensor_output_bonds"


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _publish(path: Path, value: Mapping[str, Any]) -> str:
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    temporary = path.parent / ("." + path.name + "." + secrets.token_hex(12))
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path, follow_symlinks=False)
    finally:
        if temporary.exists():
            temporary.unlink()
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(encoded).hexdigest()


class InternalDimensionPlans:
    """Compact original widths and monotone dimension-only prefix schedules."""

    def __init__(self, network: ImplicitProjectiveDAG) -> None:
        nodes = _walk_unique(network.root)
        uids = [node.uid for node in nodes]
        if not uids or min(uids) < 0 or len(set(uids)) != len(uids):
            raise ValueError("node identifiers must be unique nonnegative integers")
        if max(uids) >= 2 * len(nodes) + 1024:
            raise ValueError("node identifiers are too sparse for dense rank storage")
        self.widths = np.zeros(max(uids) + 1, dtype=np.uint32)
        self.eligible = np.zeros(max(uids) + 1, dtype=np.bool_)
        self.present = np.zeros(max(uids) + 1, dtype=np.bool_)
        self.root_uid = network.root.uid
        self.node_count = len(nodes)
        self.network_identity = id(network)
        self.leaf_count = 0
        self.leaf_dimensions = 0
        for node in nodes:
            width = node.output_dimension
            if not 1 <= width <= np.iinfo(np.uint32).max:
                raise ValueError("bond width does not fit the declared uint32 encoding")
            self.widths[node.uid] = width
            self.present[node.uid] = True
            self.eligible[node.uid] = bool(node.children) and node is not network.root
            if not node.children:
                self.leaf_count += 1
                self.leaf_dimensions += width
        self.internal_uids = np.flatnonzero(self.eligible)
        if not self.internal_uids.size:
            raise ValueError("the network has no eligible internal bonds")
        self.internal_widths = self.widths[self.internal_uids]
        self.unique_widths, self.width_inverse, self.width_counts = np.unique(
            self.internal_widths, return_inverse=True, return_counts=True
        )
        self.original_internal_dimensions = int(self.internal_widths.astype(np.uint64).sum())
        self.original_all_dimensions = int(self.widths.astype(np.uint64).sum())
        self.edge_occurrences = sum(len(node.children) for node in nodes)
        # All bonds of width d change their nearest-integer prefix at the same
        # rational thresholds (1/2d, 3/2d, ...).  Build the small width-class
        # event table once.  Partial tied events use ascending UID, giving exact
        # global budgets and nested prefixes without repeated full-array scans.
        event_count = sum(int(width) - 1 for width in self.unique_widths)
        if event_count > 2_000_000:
            raise ValueError("width-class allocation requires more than two million threshold events")
        grouped: dict[Fraction, list[tuple[int, int]]] = {}
        for index, width_value in enumerate(self.unique_widths):
            width = int(width_value)
            for removed in range(1, width):
                threshold = Fraction(2 * removed - 1, 2 * width)
                grouped.setdefault(threshold, []).append((index, width - removed))
        self.events = tuple(sorted(grouped.items()))

    def plan(self, requested: float) -> tuple[np.ndarray, dict[str, Any]]:
        if type(requested) is not float or not math.isfinite(requested) or not 0 <= requested < 1:
            raise ValueError("removal must be a finite builtin float in [0, 1)")

        exact_budget = Fraction(str(requested)) * self.original_internal_dimensions
        budget = round(exact_budget)
        maximum = self.original_internal_dimensions - int(self.internal_uids.size)
        budget = min(budget, maximum)
        actual = budget / self.original_internal_dimensions
        if abs(actual - requested) > 0.005:
            raise ValueError("requested removal is unattainable within half a percentage point with minimum rank one")
        retained_by_width = self.unique_widths.copy()
        remaining = budget
        partial_classes: list[int] = []
        partial_count = 0
        nominal = 0.0
        for threshold, events in self.events:
            if remaining == 0:
                break
            event_size = sum(int(self.width_counts[index]) for index, _rank in events)
            nominal = float(threshold)
            if remaining >= event_size:
                for index, rank in events:
                    retained_by_width[index] = rank
                remaining -= event_size
            else:
                partial_classes = [index for index, _rank in events]
                partial_count = remaining
                remaining = 0
                break
        if remaining:
            raise RuntimeError("dimension threshold events did not cover the integer budget")
        ranks = self.widths.copy()
        ranks[self.internal_uids] = retained_by_width[self.width_inverse]
        if partial_count:
            partial_uids = self.internal_uids[np.isin(self.width_inverse, partial_classes)]
            ranks[partial_uids[:partial_count]] -= 1
        kept = int(ranks[self.internal_uids].astype(np.uint64).sum())
        if self.original_internal_dimensions - kept != budget:
            raise RuntimeError("physical ranks do not match the exact dimension budget")
        record = {
            "requested_internal_dimension_removal": requested,
            "actual_internal_dimension_removal": actual,
            "nominal_per_internal_bond_removal": nominal,
            "allocation": "nearest_integer_width_thresholds_with_tied_events_split_by_ascending_uid",
            "partial_threshold_event_dimensions": partial_count,
            "original_internal_dimensions": self.original_internal_dimensions,
            "retained_internal_dimensions": kept,
            "removed_internal_dimensions": self.original_internal_dimensions - kept,
            "within_half_percentage_point": abs(actual - requested) <= 0.005,
            "eligible_unique_internal_bonds": int(self.internal_uids.size),
            "original_all_node_output_dimensions": self.original_all_dimensions,
            "retained_all_node_output_dimensions": self.original_all_dimensions - self.original_internal_dimensions + kept,
            "actual_all_node_output_dimension_removal": (self.original_internal_dimensions - kept) / self.original_all_dimensions,
            "excluded_leaf_count": self.leaf_count,
            "excluded_leaf_output_dimensions": self.leaf_dimensions,
            "excluded_root_uid": self.root_uid,
            "excluded_root_output_dimensions": int(self.widths[self.root_uid]),
            "dimension_denominator": DENOMINATOR,
            "minimum_rank": 1,
            "selection_used_actions_or_rollouts": False,
        }
        return ranks, record

    def write_original_widths(self, path: Path) -> dict[str, Any]:
        # UID is the index.  Zero means absent, and the bitmap identifies eligible
        # internal bonds.  Both streams are explicitly little endian.
        payload = self.widths.astype("<u4", copy=False).tobytes() + self.eligible.astype(np.uint8).tobytes()
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(0o444)
        return {"path": path.name, "sha256": _digest(path), "uid_slots": len(self.widths), "width_encoding": "uint32_little_endian_then_uint8_eligibility", "bytes": len(payload)}


def _validate_transition(network: ImplicitProjectiveDAG, plans: InternalDimensionPlans, current: np.ndarray, ranks: np.ndarray) -> tuple[ImplicitNode, ...]:
    if id(network) != plans.network_identity:
        raise ValueError("prefix lineage is bound to another network")
    if current.dtype != np.uint32 or ranks.dtype != np.uint32 or current.shape != plans.widths.shape or ranks.shape != current.shape:
        raise ValueError("rank array schema differs")
    if bool((ranks[plans.present] < 1).any()) or bool((ranks > current).any()):
        raise ValueError("descendant prefixes cannot restore discarded directions")
    if not np.array_equal(ranks[~plans.eligible], plans.widths[~plans.eligible]):
        raise ValueError("physical ingress and root widths must remain unchanged")
    nodes = _walk_unique(network.root)
    if len(nodes) != plans.node_count or network.root.uid != plans.root_uid:
        raise ValueError("prefix lineage topology changed")
    seen = np.zeros_like(plans.present)
    for node in nodes:
        if not 0 <= node.uid < len(current) or seen[node.uid] or not plans.present[node.uid]:
            raise ValueError("prefix lineage node inventory differs")
        seen[node.uid] = True
        if node.output_dimension != int(current[node.uid]):
            raise ValueError("prefix lineage current width differs")
    if not np.array_equal(seen, plans.present):
        raise ValueError("prefix lineage node inventory differs")
    _validate_network(network)
    return nodes


@torch.no_grad()
def zero_preserving_boundary(network: ImplicitProjectiveDAG, raw: Any, ranks: np.ndarray | None = None) -> torch.Tensor:
    """Evaluate the homogeneous boundary, retaining exactly zero rows.

    A zero intermediate is propagated algebraically, not replaced with a
    nonzero value.  A zero final row means the quotient is undefined in this
    float64 execution.  Nonfinite values remain hard execution failures.
    """
    batch, prepared = _prepare_physical_input(network, raw)
    memo: dict[int, torch.Tensor] = {}

    def normalize(value: torch.Tensor) -> torch.Tensor:
        maximum = value.abs().amax(dim=1)
        if not bool(torch.isfinite(maximum).all()):
            raise FloatingPointError("projective execution produced nonfinite coordinates")
        # frexp(0) returns exponent zero, so exact zero rows stay exactly zero.
        _mantissa, exponent = torch.frexp(maximum)
        return torch.ldexp(value, -exponent[:, None])

    for node in _walk_unique(network.root):
        if node.children:
            inputs = tuple(memo[child.uid] for child in node.children)
        else:
            if node.physical_source_key is not None:
                value = prepared[node.physical_source_key]
            else:
                value = prepared[:, node.physical_token]
            inputs = (torch.cat((value, value.new_ones(batch, 1)), dim=1),)
        value = _core_apply(node.core, inputs)
        rank = node.output_dimension if ranks is None else int(ranks[node.uid])
        if rank < value.shape[-1]:
            value[:, rank:] = 0.0
        memo[node.uid] = normalize(value)
    return normalize(memo[network.root.uid] @ network.head.T)


def masked_boundary(network: ImplicitProjectiveDAG, raw: Any, ranks: np.ndarray) -> torch.Tensor:
    """Independent zero-mask evaluator on the current unsliced prefix."""
    return zero_preserving_boundary(network, raw, ranks)


@torch.no_grad()
def slice_descendant_in_place(network: ImplicitProjectiveDAG, plans: InternalDimensionPlans, current: np.ndarray, ranks: np.ndarray) -> dict[str, int]:
    """Slice each producer and each parent's ordered input role exactly once."""
    nodes = _validate_transition(network, plans, current, ranks)
    expected = sum(sum(int(ranks[child.uid]) < int(current[child.uid]) for child in node.children) for node in nodes)
    applied = 0
    reduced = 0
    seen_storage: set[int] = set()
    copied_tensors = 0
    reused_tensors = 0

    def compact(value: torch.Tensor) -> torch.Tensor:
        nonlocal copied_tensors, reused_tensors
        storage = value.untyped_storage()
        pointer = int(storage.data_ptr())
        if (
            value.storage_offset() == 0
            and int(storage.nbytes()) == value.numel() * value.element_size()
            and value.is_contiguous()
            and pointer not in seen_storage
        ):
            result = value
            reused_tensors += 1
        else:
            result = _compact_copy(value)
            pointer = int(result.untyped_storage().data_ptr())
            copied_tensors += 1
        seen_storage.add(pointer)
        return result

    network.head = compact(network.head)
    for node in nodes:
        output = int(ranks[node.uid])
        children = tuple(int(ranks[child.uid]) for child in node.children)
        reduced += output < int(current[node.uid])
        applied += sum(int(ranks[child.uid]) < int(current[child.uid]) for child in node.children)
        core = node.core
        if isinstance(core, UnaryCore):
            incoming = children[0] if children else core.matrix.shape[1]
            source = core.matrix if (output, incoming) == tuple(core.matrix.shape) else core.matrix[:output, :incoming]
            matrix = compact(source)
            if matrix is not core.matrix:
                node.core = UnaryCore(matrix, core.kind, core.binary_exponent)
        elif isinstance(core, CPBinaryCore):
            output_factor = compact(core.output_factor if output == core.output_dimension else core.output_factor[:output])
            left_factor = compact(core.left_factor if children[0] == core.input_dimensions[0] else core.left_factor[:, :children[0]])
            right_factor = compact(core.right_factor if children[1] == core.input_dimensions[1] else core.right_factor[:, :children[1]])
            if core.direct_q_provenance is not None or output_factor is not core.output_factor or left_factor is not core.left_factor or right_factor is not core.right_factor:
                node.core = CPBinaryCore(output_factor, left_factor, right_factor, core.kind, core.binary_exponent, None)
        elif isinstance(core, ReducedQRBinaryCore):
            if (output, children[0], children[1]) == (core.output_dimension, core.left_dimension, core.right_dimension):
                rows = compact(core.q_rows)
            else:
                value = core.q_rows.reshape(core.output_dimension, core.left_dimension, core.right_dimension)
                rows = compact(value[:output, :children[0], :children[1]]).reshape(output, -1)
            if rows is not core.q_rows:
                node.core = ReducedQRBinaryCore(rows, children[0], children[1], core.kind, core.binary_exponent)
        else:
            raise TypeError("prefix export only supports production unary, CP, and retained-Q cores")
    if applied != expected:
        raise RuntimeError("a shared bond occurrence was not sliced")
    network.algorithm1_direct_rq_complete = False
    network.algorithm1_scale_ledger_complete = False
    network.algorithm1_certified_head_binary_exponent = None
    del seen_storage
    _validate_network(network)
    for node in nodes:
        if node.output_dimension != int(ranks[node.uid]):
            raise RuntimeError("physical output width differs from rank plan")
    logical = implicit_storage_elements(network)
    allocated = implicit_allocated_storage_elements(network)
    if allocated != logical:
        raise RuntimeError("discarded storage remains attached to the physical prefix")
    return {"reduced_bonds_in_transition": reduced, "expected_occurrence_slices": expected, "applied_occurrence_slices": applied, "physical_storage_elements": logical, "allocated_storage_elements": allocated, "copied_tensors": copied_tensors, "reused_compact_tensors": reused_tensors}


def _relative_pair(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    # Boundary rows are normalized up to a common projective sign.  Choose that
    # sign by the largest-magnitude reference coordinate, without self-overlaps.
    candidate_scale = candidate.abs().amax(dim=1, keepdim=True)
    reference_scale = reference.abs().amax(dim=1, keepdim=True)
    candidate = candidate / torch.where(candidate_scale == 0, torch.ones_like(candidate_scale), candidate_scale)
    reference = reference / torch.where(reference_scale == 0, torch.ones_like(reference_scale), reference_scale)
    pivot = reference.abs().argmax(dim=1, keepdim=True)
    sign = torch.sign(candidate.gather(1, pivot) * reference.gather(1, pivot))
    aligned = candidate * torch.where(sign == 0, torch.ones_like(sign), sign)
    return float((aligned - reference).abs().max().item())


def export_dimension_ladder(diagonal: DiagonalImplicitDAG, raw: Any, destination: str | Path, *, full_rank_certificate_sha256: str, checkpoint_sha256: str, removals: Sequence[float] = REMOVALS, on_rung: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Export six narrow policies without retaining six graph copies in RAM."""
    for digest in (full_rank_certificate_sha256, checkpoint_sha256):
        if type(digest) is not str or len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise ValueError("lineage requires canonical SHA-256 identities")
    requests = tuple(removals)
    if not requests or any(type(value) is not float for value in requests) or tuple(sorted(set(requests))) != requests:
        raise ValueError("removal requests must be unique ascending builtin floats")
    network = _certified_network(diagonal)
    plans = InternalDimensionPlans(network)
    root = Path(destination)
    root.mkdir(parents=False, exist_ok=False)
    width_receipt = plans.write_original_widths(root / "original_dimensions.bin")
    current = plans.widths.copy()
    baseline = evaluate_projective_boundary(network, raw)
    records = []
    previous_manifest: str | None = None
    for request in requests:
        ranks, selection = plans.plan(request)
        _validate_transition(network, plans, current, ranks)
        started = time.perf_counter()
        masked = masked_boundary(network, raw, ranks)
        slicing = slice_descendant_in_place(network, plans, current, ranks)
        replay = zero_preserving_boundary(network, raw)
        replay_error = _relative_pair(replay, masked)
        if not math.isfinite(replay_error) or replay_error > 2e-10:
            raise RuntimeError(f"masked/materialized boundary mismatch at {request}: {replay_error}")
        denominator = replay[:, -1]
        relative_denominator = denominator.abs() / replay.abs().amax(dim=1).clamp_min(torch.finfo(replay.dtype).tiny)
        valid_rows = relative_denominator > 100.0 * torch.finfo(replay.dtype).eps
        valid = bool(valid_rows.all())
        action_error: float | None = None
        if valid and bool((baseline[:, -1].abs() > 100.0 * torch.finfo(baseline.dtype).eps).all()):
            actions = replay[:, :-1] / denominator[:, None]
            baseline_actions = baseline[:, :-1] / baseline[:, -1, None]
            action_error = float((actions - baseline_actions).square().sum().sqrt().div(baseline_actions.square().sum().sqrt().clamp_min(torch.finfo(replay.dtype).tiny)).item())
        name = f"remove_{round(request * 100):02d}"
        network.claim_boundary = f"Physical descendant prefix of a certified direct ODT basis. Requested removal {request} uses {DENOMINATOR}. Capability is measured separately."
        artifact = export_implicit_projective_prefix_artifact(network, root / name)
        record = {
            "schema": SCHEMA,
            **selection,
            **slicing,
            "checkpoint_sha256": checkpoint_sha256,
            "full_rank_certificate_sha256": full_rank_certificate_sha256,
            "original_dimensions": width_receipt,
            "parent_prefix_manifest_sha256": previous_manifest,
            "artifact": {"path": name, "manifest_sha256": artifact.manifest_sha256, "node_count": artifact.node_count, "tensor_count": artifact.tensor_count, "raw_byte_count": artifact.raw_byte_count, "shard_sha256": dict(artifact.shard_sha256)},
            "masked_materialized_boundary_max_error": replay_error,
            "replay_samples": int(replay.shape[0]),
            "valid_projective_chart_on_replay": valid,
            "replay_chart_status": "valid" if valid else "undefined_in_float64_execution",
            "undefined_replay_row_indices": torch.nonzero(~valid_rows, as_tuple=False).flatten().tolist(),
            "action_relative_error_on_replay": action_error,
            "replay_input_sha256": _raw_digest(raw),
            "replay_boundary": replay.detach().cpu().tolist(),
            "elapsed_seconds": time.perf_counter() - started,
            "closed_loop_capability_completed": False,
            "new_full_rank_decomposition_claimed": False,
        }
        record["receipt_sha256"] = _publish(root / f"{name}.json", record)
        records.append(record)
        if on_rung is not None:
            on_rung(record)
        current = ranks
        previous_manifest = artifact.manifest_sha256
    result = {"schema": SCHEMA, "dimension_denominator": DENOMINATOR, "checkpoint_sha256": checkpoint_sha256, "full_rank_certificate_sha256": full_rank_certificate_sha256, "original_dimensions": width_receipt, "rungs": records, "all_requested_artifacts_exported": True, "closed_loop_capability_completed": False}
    result["receipt_sha256"] = _publish(root / "ladder.json", result)
    return result


def _raw_digest(raw: Any) -> str:
    digest = hashlib.sha256()
    items = sorted(raw.items()) if isinstance(raw, dict) else [("__homogeneous__", raw)]
    for name, value in items:
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
