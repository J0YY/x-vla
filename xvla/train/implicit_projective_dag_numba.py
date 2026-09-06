"""Experimental compiled interpreter for an authenticated mapped DAG.

This module is additive.  It does not change artifact parsing, tensor bytes,
the eager executor, or the production export lane.  The mapped loader remains
the authority for authentication and structural validation.  This interpreter
only replaces millions of Python-level node dispatches with one compiled loop.

Every core evaluates the same unary, CP-binary, or retained-Q contraction and
uses the same per-row power-of-two projective chart.  A precomputed arena plan
decrements every syntactic child occurrence, including repeated-child roles,
and makes a released span reusable only after its count reaches exactly zero.
The ordinary mapped executor is the required numerical oracle before use.
"""

from __future__ import annotations

import bisect
import gc
import hashlib
import math
import mmap
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from numba import njit, prange
from numba.typed import List as NumbaList
from torch import Tensor

from xvla.train.implicit_projective_dag_mapped import (
    AlignedContiguousTensorSpan,
    MappedImplicitProjectiveDAGArtifact,
    MappedExecutionReceipt,
    RawPhysicalInput,
    SegmentedTensorSpan,
    _clear_exception_graph_tracebacks,
    open_mapped_implicit_projective_dag_artifact,
)


_UNARY = 0
_CP_BINARY = 1
_RETAINED_Q_BINARY = 2
_MAXIMUM_PREBOUND_SEGMENTED_BYTES = 1 << 30
DEFAULT_MAXIMUM_ARENA_BYTES = 64 << 30


@dataclass(frozen=True)
class CompiledArenaReceipt:
    arena_width: int
    logical_output_width_sum: int
    released_nonroot_nodes: int
    peak_live_values: int
    edge_occurrences: int
    edge_ledger_sha256: str
    root_only_live: bool
    all_refcounts_zero: bool
    segmented_tensors_bound: int
    segmented_bytes_bound: int


@dataclass(frozen=True)
class CompiledMappedExecutionReceipt(MappedExecutionReceipt):
    """Mapped execution ledger plus the compiled arena allocation."""

    arena_width: int
    arena_bytes: int


def _as_i64(values: Sequence[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _arena_plan(
    output_dimensions: Sequence[int],
    child_offsets: Sequence[int],
    child_indices: Sequence[int],
    incoming_refcount: Sequence[int],
) -> tuple[np.ndarray, CompiledArenaReceipt]:
    """Assign reusable spans while independently replaying occurrence counts."""

    node_count = len(output_dimensions)
    if node_count < 1 or len(child_offsets) != node_count + 1:
        raise ValueError("compiled arena graph dimensions differ")
    remaining = [int(value) for value in incoming_refcount]
    slots = np.empty(node_count, dtype=np.int64)
    live: dict[int, tuple[int, int]] = {}
    free_by_width: dict[int, list[int]] = {}
    free_widths: list[int] = []
    arena_width = 0
    peak_live = 0
    released = 0
    edge_ordinal = 0
    edge_digest = hashlib.sha256()

    def release_span(offset: int, width: int) -> None:
        available = free_by_width.get(width)
        if available is None:
            free_by_width[width] = [offset]
            bisect.insort(free_widths, width)
        else:
            available.append(offset)

    def allocate_span(width: int) -> int:
        nonlocal arena_width
        position = bisect.bisect_left(free_widths, width)
        if position == len(free_widths):
            result = arena_width
            arena_width += width
            return result
        available_width = free_widths[position]
        offsets = free_by_width[available_width]
        result = offsets.pop()
        if not offsets:
            del free_by_width[available_width]
            del free_widths[position]
        remainder = available_width - width
        if remainder:
            release_span(result + width, remainder)
        return result

    for parent in range(node_count):
        width = int(output_dimensions[parent])
        if width < 1:
            raise ValueError("compiled arena encountered a nonpositive bond")
        offset = allocate_span(width)
        slots[parent] = offset
        live[parent] = (offset, width)
        peak_live = max(peak_live, len(live))
        start = int(child_offsets[parent])
        stop = int(child_offsets[parent + 1])
        if not 0 <= start <= stop <= len(child_indices):
            raise ValueError("compiled arena child offsets differ")
        for role, edge in enumerate(range(start, stop)):
            child = int(child_indices[edge])
            if child not in live or not 0 <= child < parent:
                raise RuntimeError("compiled arena observed a nonlive child")
            before = remaining[child]
            if before <= 0:
                raise RuntimeError("compiled arena occurrence ledger underflowed")
            after = before - 1
            remaining[child] = after
            edge_digest.update(
                struct.pack(
                    "<QQQQQQ",
                    edge_ordinal,
                    parent,
                    role,
                    child,
                    before,
                    after,
                )
            )
            edge_ordinal += 1
            if after == 0:
                child_offset, child_width = live.pop(child)
                release_span(child_offset, child_width)
                released += 1

    root = node_count - 1
    if any(remaining):
        raise RuntimeError("compiled arena occurrence ledger has leftovers")
    if set(live) != {root} or released != node_count - 1:
        raise RuntimeError("compiled arena did not finish root-only live")
    if edge_ordinal != len(child_indices):
        raise RuntimeError("compiled arena emitted-edge ledger is incomplete")
    return slots, CompiledArenaReceipt(
        arena_width=arena_width,
        logical_output_width_sum=sum(int(value) for value in output_dimensions),
        released_nonroot_nodes=released,
        peak_live_values=peak_live,
        edge_occurrences=edge_ordinal,
        edge_ledger_sha256=edge_digest.hexdigest(),
        root_only_live=True,
        all_refcounts_zero=True,
        segmented_tensors_bound=0,
        segmented_bytes_bound=0,
    )


@njit(inline="always")
def _normalize_span(
    arena: np.ndarray,
    batch_index: int,
    offset: int,
    width: int,
) -> bool:
    maximum = 0.0
    for coordinate in range(width):
        value = arena[batch_index, offset + coordinate]
        if not math.isfinite(value):
            return False
        magnitude = abs(value)
        if magnitude > maximum:
            maximum = magnitude
    if maximum == 0.0:
        return False
    _mantissa, exponent = math.frexp(maximum)
    for coordinate in range(width):
        arena[batch_index, offset + coordinate] = math.ldexp(
            arena[batch_index, offset + coordinate], -exponent
        )
    return True


@njit(parallel=True, fastmath=False)
def _execute_compiled(
    shards: Any,
    sources: Any,
    core_codes: np.ndarray,
    tensor_shards: np.ndarray,
    tensor_offsets: np.ndarray,
    tensor0: np.ndarray,
    tensor1: np.ndarray,
    tensor2: np.ndarray,
    output_dimensions: np.ndarray,
    input_left: np.ndarray,
    input_right: np.ndarray,
    cp_ranks: np.ndarray,
    child_offsets: np.ndarray,
    child_indices: np.ndarray,
    source_indices: np.ndarray,
    slots: np.ndarray,
    arena_width: int,
    head_tensor: int,
    head_output_dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = sources[0].shape[0]
    node_count = core_codes.shape[0]
    arena = np.empty((batch_size, arena_width), dtype=np.float64)
    output = np.empty((batch_size, head_output_dimension), dtype=np.float64)
    status = np.zeros(batch_size, dtype=np.uint8)

    for batch_index in prange(batch_size):
        failed = False
        for node in range(node_count):
            code = core_codes[node]
            output_width = output_dimensions[node]
            output_offset = slots[node]
            edge_start = child_offsets[node]
            edge_stop = child_offsets[node + 1]
            first_tensor = tensor0[node]
            first_weights = shards[tensor_shards[first_tensor]]
            first_weight_offset = tensor_offsets[first_tensor]

            if code == _UNARY:
                incoming = input_left[node]
                if edge_start == edge_stop:
                    source = sources[source_indices[node]]
                    for out_coordinate in range(output_width):
                        weight_row = first_weight_offset + out_coordinate * incoming
                        accumulator = first_weights[weight_row + incoming - 1]
                        for in_coordinate in range(incoming - 1):
                            accumulator += (
                                source[batch_index, in_coordinate]
                                * first_weights[weight_row + in_coordinate]
                            )
                        arena[batch_index, output_offset + out_coordinate] = accumulator
                else:
                    child = child_indices[edge_start]
                    child_offset = slots[child]
                    for out_coordinate in range(output_width):
                        weight_row = first_weight_offset + out_coordinate * incoming
                        accumulator = 0.0
                        for in_coordinate in range(incoming):
                            accumulator += (
                                arena[batch_index, child_offset + in_coordinate]
                                * first_weights[weight_row + in_coordinate]
                            )
                        arena[batch_index, output_offset + out_coordinate] = accumulator

            elif code == _CP_BINARY:
                left_child = child_indices[edge_start]
                right_child = child_indices[edge_start + 1]
                left_offset = slots[left_child]
                right_offset = slots[right_child]
                left_width = input_left[node]
                right_width = input_right[node]
                rank = cp_ranks[node]
                left_tensor = tensor1[node]
                right_tensor = tensor2[node]
                left_weights = shards[tensor_shards[left_tensor]]
                right_weights = shards[tensor_shards[right_tensor]]
                left_weight_offset = tensor_offsets[left_tensor]
                right_weight_offset = tensor_offsets[right_tensor]
                for out_coordinate in range(output_width):
                    arena[batch_index, output_offset + out_coordinate] = 0.0
                for rank_coordinate in range(rank):
                    left_value = 0.0
                    left_row = left_weight_offset + rank_coordinate * left_width
                    for coordinate in range(left_width):
                        left_value += (
                            arena[batch_index, left_offset + coordinate]
                            * left_weights[left_row + coordinate]
                        )
                    right_value = 0.0
                    right_row = right_weight_offset + rank_coordinate * right_width
                    for coordinate in range(right_width):
                        right_value += (
                            arena[batch_index, right_offset + coordinate]
                            * right_weights[right_row + coordinate]
                        )
                    product = left_value * right_value
                    for out_coordinate in range(output_width):
                        arena[batch_index, output_offset + out_coordinate] += (
                            product
                            * first_weights[
                                first_weight_offset
                                + out_coordinate * rank
                                + rank_coordinate
                            ]
                        )

            elif code == _RETAINED_Q_BINARY:
                left_child = child_indices[edge_start]
                right_child = child_indices[edge_start + 1]
                left_offset = slots[left_child]
                right_offset = slots[right_child]
                left_width = input_left[node]
                right_width = input_right[node]
                flattened = left_width * right_width
                for out_coordinate in range(output_width):
                    accumulator = 0.0
                    weight_row = first_weight_offset + out_coordinate * flattened
                    for left_coordinate in range(left_width):
                        left_value = arena[
                            batch_index, left_offset + left_coordinate
                        ]
                        base = weight_row + left_coordinate * right_width
                        for right_coordinate in range(right_width):
                            accumulator += (
                                first_weights[base + right_coordinate]
                                * left_value
                                * arena[
                                    batch_index,
                                    right_offset + right_coordinate,
                                ]
                            )
                    arena[batch_index, output_offset + out_coordinate] = accumulator
            else:
                failed = True
                break

            if not _normalize_span(
                arena,
                batch_index,
                output_offset,
                output_width,
            ):
                failed = True
                break

        if failed:
            status[batch_index] = 1
            continue

        root = node_count - 1
        root_offset = slots[root]
        root_width = output_dimensions[root]
        head_weights = shards[tensor_shards[head_tensor]]
        head_weight_offset = tensor_offsets[head_tensor]
        maximum = 0.0
        for out_coordinate in range(head_output_dimension):
            accumulator = 0.0
            weight_row = head_weight_offset + out_coordinate * root_width
            for in_coordinate in range(root_width):
                accumulator += (
                    arena[batch_index, root_offset + in_coordinate]
                    * head_weights[weight_row + in_coordinate]
                )
            output[batch_index, out_coordinate] = accumulator
            if not math.isfinite(accumulator):
                failed = True
            magnitude = abs(accumulator)
            if magnitude > maximum:
                maximum = magnitude
        if failed or maximum == 0.0:
            status[batch_index] = 2
            continue
        _mantissa, exponent = math.frexp(maximum)
        for out_coordinate in range(head_output_dimension):
            output[batch_index, out_coordinate] = math.ldexp(
                output[batch_index, out_coordinate], -exponent
            )
    return output, status


class CompiledMappedImplicitProjectiveDAG:
    """Own one mapped view and its read-only compiled execution bindings."""

    def __init__(
        self,
        mapped: MappedImplicitProjectiveDAGArtifact,
        *,
        maximum_arena_bytes: int = DEFAULT_MAXIMUM_ARENA_BYTES,
    ) -> None:
        if mapped.closed:
            raise RuntimeError("compiled mapped executor received a closed artifact")
        if not mapped.execution_layout_supported:
            raise RuntimeError("compiled mapped executor requires a supported layout")
        if (
            type(maximum_arena_bytes) is not int
            or maximum_arena_bytes < 8
            or maximum_arena_bytes > (1 << 63) - 1
        ):
            raise ValueError("compiled maximum arena bytes is invalid")
        self._mapped = mapped
        self._maximum_arena_bytes = maximum_arena_bytes
        self._closed = False
        self._close_pending = False
        self._active_evaluations = 0
        self._lock = threading.RLock()
        self._weight_arrays: tuple[np.ndarray, ...] = ()
        self._typed_weights: Any = None
        self._segmented_arrays: tuple[np.ndarray, ...] = ()
        self._bind()

    def _bind(self) -> None:
        mapped = self._mapped
        nodes = mapped._nodes
        tensors = mapped.tensors
        arrays: list[np.ndarray] = []
        for mapping in mapped._mappings:
            if len(mapping) % 8:
                raise RuntimeError("mapped shard byte count is not float64 aligned")
            value = np.frombuffer(mapping, dtype="<f8")
            value.flags.writeable = False
            arrays.append(value)
        original_shard_count = len(arrays)
        tensor_shards = np.full(len(tensors), -1, dtype=np.int64)
        tensor_offsets = np.full(len(tensors), -1, dtype=np.int64)
        segmented: list[np.ndarray] = []
        segmented_bytes = 0
        required_tensors = {mapped.network.head_tensor}
        for indices in nodes.tensor_indices:
            required_tensors.update(indices)
        for descriptor in tensors:
            if descriptor.index not in required_tensors:
                continue
            layout = descriptor.layout
            if isinstance(layout, AlignedContiguousTensorSpan):
                tensor_shards[descriptor.index] = layout.shard_index
                tensor_offsets[descriptor.index] = layout.offset_bytes // 8
                continue
            if not isinstance(layout, SegmentedTensorSpan) or layout.reason != "cross_shard":
                raise RuntimeError("compiled mapped tensor layout is unsupported")
            if descriptor.byte_count > mapped.maximum_segmented_gather_bytes:
                raise RuntimeError("compiled segmented tensor exceeds the bounded cap")
            gathered = np.empty(descriptor.byte_count // 8, dtype="<f8")
            destination = memoryview(gathered).cast("B")
            cursor = 0
            for segment in layout.segments:
                source = memoryview(mapped._mappings[segment.shard_index])[
                    segment.offset_bytes : segment.offset_bytes + segment.byte_count
                ]
                destination[cursor : cursor + segment.byte_count] = source
                source.release()
                cursor += segment.byte_count
            destination.release()
            if cursor != descriptor.byte_count:
                raise RuntimeError("compiled segmented tensor gather ledger differs")
            gathered.flags.writeable = False
            tensor_shards[descriptor.index] = len(arrays)
            tensor_offsets[descriptor.index] = 0
            arrays.append(gathered)
            segmented.append(gathered)
            segmented_bytes += descriptor.byte_count
            if segmented_bytes > _MAXIMUM_PREBOUND_SEGMENTED_BYTES:
                raise RuntimeError(
                    "compiled prebound segmented tensors exceed the 1-GiB cap"
                )
        required_indices = np.fromiter(
            sorted(required_tensors),
            dtype=np.int64,
            count=len(required_tensors),
        )
        if bool((tensor_shards[required_indices] < 0).any()) or bool(
            (tensor_offsets[required_indices] < 0).any()
        ):
            raise RuntimeError("compiled tensor binding ledger is incomplete")

        core_codes = np.empty(nodes.node_count, dtype=np.uint8)
        tensor0 = np.full(nodes.node_count, -1, dtype=np.int64)
        tensor1 = np.full(nodes.node_count, -1, dtype=np.int64)
        tensor2 = np.full(nodes.node_count, -1, dtype=np.int64)
        cp_ranks = np.zeros(nodes.node_count, dtype=np.int64)
        input_left = np.zeros(nodes.node_count, dtype=np.int64)
        input_right = np.zeros(nodes.node_count, dtype=np.int64)
        source_indices = np.full(nodes.node_count, -1, dtype=np.int64)
        source_keys = {
            specification.key: index
            for index, specification in enumerate(mapped.network.physical_sources)
        }
        for index in range(nodes.node_count):
            kind = nodes.core_type[index]
            indices = nodes.tensor_indices[index]
            dimensions = nodes.input_dimensions[index]
            if kind == "unary":
                core_codes[index] = _UNARY
                tensor0[index] = indices[0]
                input_left[index] = dimensions[0]
            elif kind == "cp_binary":
                core_codes[index] = _CP_BINARY
                tensor0[index], tensor1[index], tensor2[index] = indices
                input_left[index], input_right[index] = dimensions
                cp_ranks[index] = tensors[indices[0]].shape[1]
            elif kind == "reduced_q_binary":
                core_codes[index] = _RETAINED_Q_BINARY
                tensor0[index] = indices[0]
                input_left[index], input_right[index] = dimensions
            else:
                raise RuntimeError("compiled mapped executor encountered an unknown core")
            start = nodes.child_offsets[index]
            stop = nodes.child_offsets[index + 1]
            if start == stop:
                if source_keys:
                    key = nodes.physical_source_key[index]
                    if key not in source_keys:
                        raise RuntimeError("compiled heterogeneous leaf source differs")
                    source_indices[index] = source_keys[key]
                else:
                    token = nodes.physical_token[index]
                    if token is None:
                        raise RuntimeError("compiled homogeneous leaf token differs")
                    source_indices[index] = token

        slots, arena = _arena_plan(
            nodes.output_dimensions,
            nodes.child_offsets,
            nodes.child_indices,
            nodes.incoming_refcount,
        )
        arena = CompiledArenaReceipt(
            **{
                **arena.__dict__,
                "segmented_tensors_bound": len(segmented),
                "segmented_bytes_bound": segmented_bytes,
            }
        )
        typed = NumbaList()
        for value in arrays:
            typed.append(value)
        self._weight_arrays = tuple(arrays[:original_shard_count])
        self._segmented_arrays = tuple(segmented)
        self._typed_weights = typed
        self._tensor_shards = tensor_shards
        self._tensor_offsets = tensor_offsets
        self._core_codes = core_codes
        self._tensor0 = tensor0
        self._tensor1 = tensor1
        self._tensor2 = tensor2
        self._output_dimensions = _as_i64(nodes.output_dimensions)
        self._input_left = input_left
        self._input_right = input_right
        self._cp_ranks = cp_ranks
        self._child_offsets = _as_i64(nodes.child_offsets)
        self._child_indices = _as_i64(nodes.child_indices)
        self._source_indices = source_indices
        self._slots = slots
        self._arena = arena
        head = tensors[mapped.network.head_tensor]
        self._head_output_dimension = head.shape[0]

    @property
    def arena_receipt(self) -> CompiledArenaReceipt:
        return self._arena

    @property
    def manifest_sha256(self) -> str:
        return self._mapped.manifest_sha256

    @property
    def maximum_arena_bytes(self) -> int:
        return self._maximum_arena_bytes

    @property
    def execution_layout_supported(self) -> bool:
        return self._mapped.execution_layout_supported

    @property
    def segmented_bytes_per_evaluation(self) -> int:
        # Segmented tensors are gathered once at bind time, not once per replay.
        return 0

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> "CompiledMappedImplicitProjectiveDAG":
        self._require_open()
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("compiled mapped executor is closed")
        if self._close_pending:
            raise RuntimeError(
                "compiled mapped close is incomplete; release exported views "
                "and retry close"
            )

    def arena_bytes_for_batch(self, batch_size: int) -> int:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("compiled arena batch size must be a positive integer")
        return batch_size * self._arena.arena_width * 8

    def _prepared_sources(
        self,
        prepared: Tensor | Mapping[str, Tensor],
    ) -> Any:
        values: list[np.ndarray] = []
        specifications = self._mapped.network.physical_sources
        if specifications:
            if not isinstance(prepared, Mapping):
                raise RuntimeError("compiled heterogeneous inputs differ")
            for specification in specifications:
                values.append(
                    np.ascontiguousarray(prepared[specification.key].numpy())
                )
        else:
            if not isinstance(prepared, Tensor):
                raise RuntimeError("compiled homogeneous inputs differ")
            for token in range(self._mapped.network.token_count):
                values.append(np.ascontiguousarray(prepared[:, token].numpy()))
        typed = NumbaList()
        for value in values:
            typed.append(value)
        return typed

    def evaluate_projective_boundary(
        self,
        raw_input: RawPhysicalInput,
        *,
        return_receipt: bool = False,
    ) -> Tensor | tuple[Tensor, CompiledMappedExecutionReceipt]:
        if type(return_receipt) is not bool:
            raise TypeError("return_receipt must be boolean")
        with self._lock:
            self._require_open()
            self._active_evaluations += 1
        try:
            self._mapped.assert_unchanged()
            batch_size, prepared = self._mapped._prepare_input(raw_input)
            arena_bytes = self.arena_bytes_for_batch(batch_size)
            output_bytes = batch_size * self._head_output_dimension * 8
            if arena_bytes > self._maximum_arena_bytes - output_bytes:
                raise RuntimeError(
                    "compiled mapped arena exceeds its explicit byte cap before execution"
                )
            sources = self._prepared_sources(prepared)
            output, status = _execute_compiled(
                self._typed_weights,
                sources,
                self._core_codes,
                self._tensor_shards,
                self._tensor_offsets,
                self._tensor0,
                self._tensor1,
                self._tensor2,
                self._output_dimensions,
                self._input_left,
                self._input_right,
                self._cp_ranks,
                self._child_offsets,
                self._child_indices,
                self._source_indices,
                self._slots,
                self._arena.arena_width,
                self._mapped.network.head_tensor,
                self._head_output_dimension,
            )
            if output.shape[0] != batch_size or bool((status != 0).any()):
                raise ValueError(
                    "mapped projective evaluation produced zero/nonfinite coordinates"
                )
            self._mapped.assert_unchanged()
            result = torch.from_numpy(output)
            receipt = CompiledMappedExecutionReceipt(
                node_evaluations=self._core_codes.shape[0],
                edge_occurrences_emitted=self._arena.edge_occurrences,
                released_nonroot_nodes=self._arena.released_nonroot_nodes,
                peak_live_values=self._arena.peak_live_values,
                edge_ledger_sha256=self._arena.edge_ledger_sha256,
                root_only_live=self._arena.root_only_live,
                all_refcounts_zero=self._arena.all_refcounts_zero,
                segmented_tensor_gathers=0,
                segmented_bytes_gathered=0,
                arena_width=self._arena.arena_width,
                arena_bytes=arena_bytes,
            )
        finally:
            with self._lock:
                self._active_evaluations -= 1
        if return_receipt:
            return result, receipt
        return result

    def evaluate_boundary_quotient(
        self,
        raw_input: RawPhysicalInput,
        *,
        return_receipt: bool = False,
    ) -> Tensor | tuple[Tensor, CompiledMappedExecutionReceipt]:
        evaluated = self.evaluate_projective_boundary(
            raw_input,
            return_receipt=return_receipt,
        )
        if return_receipt:
            pair, receipt = evaluated
        else:
            pair = evaluated
            receipt = None
        denominator = pair[:, -1]
        relative = denominator.abs() / pair.abs().amax(dim=1).clamp_min(
            torch.finfo(pair.dtype).tiny
        )
        if float(relative.min().item()) <= 100.0 * torch.finfo(pair.dtype).eps:
            raise ValueError("mapped projective denominator is numerically zero")
        quotient = pair[:, :-1] / denominator[:, None]
        if return_receipt:
            assert receipt is not None
            return quotient, receipt
        return quotient

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._active_evaluations:
                raise RuntimeError("cannot close compiled mapped executor during evaluation")
            self._typed_weights = None
            self._weight_arrays = ()
            self._segmented_arrays = ()
            gc.collect()
            try:
                self._mapped.close()
            except BaseException:
                self._close_pending = True
                raise
            self._close_pending = False
            self._closed = True


def open_compiled_mapped_implicit_projective_dag_artifact(
    source: str | Path,
    *,
    expected_manifest_sha256: str,
    maximum_segmented_gather_bytes: int | None = None,
    maximum_arena_bytes: int = DEFAULT_MAXIMUM_ARENA_BYTES,
) -> CompiledMappedImplicitProjectiveDAG:
    """Authenticate with the ordinary loader, then bind the compiled plan."""

    arguments: dict[str, Any] = {
        "expected_manifest_sha256": expected_manifest_sha256,
    }
    if maximum_segmented_gather_bytes is not None:
        arguments["maximum_segmented_gather_bytes"] = maximum_segmented_gather_bytes
    mapped = open_mapped_implicit_projective_dag_artifact(source, **arguments)
    try:
        return CompiledMappedImplicitProjectiveDAG(
            mapped,
            maximum_arena_bytes=maximum_arena_bytes,
        )
    except BaseException as error:
        # A failed binder frame can itself retain NumPy exports of the mmap.
        # Clear completed frames before asking the underlying owner to close.
        _clear_exception_graph_tracebacks(error)
        gc.collect()
        mapped.close()
        raise


__all__ = [
    "CompiledArenaReceipt",
    "CompiledMappedImplicitProjectiveDAG",
    "DEFAULT_MAXIMUM_ARENA_BYTES",
    "open_compiled_mapped_implicit_projective_dag_artifact",
]
