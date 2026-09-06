"""Authenticated, read-only mapped execution for implicit DAG artifacts.

This module is intentionally additive to ``implicit_projective_dag_artifact``.
It consumes that module's byte-stable v1 format without reconstructing an
``ImplicitNode`` object graph and without copying tensor payloads.  Metadata is
parsed into columnar descriptors, while every tensor shard remains owned by a
read-only ``mmap`` for the lifetime of the context-managed view.

One contiguous float64-aligned span executes zero-copy.  A tensor split across
shards executes through an explicit, bounded per-node gather whose exact byte
cost is reported in the execution receipt.  The loader never clones the whole
artifact payload, and layouts outside those two cases fail before node one.

Opening cryptographically authenticates the manifest and every shard.  The
cheap checks around each evaluation subsequently validate the same no-follow,
single-link, read-only file cohort and its stat snapshots.  They are not a
second full-payload hash and therefore do not defend against a privileged
writer that can alter bytes while restoring indistinguishable metadata.
"""

from __future__ import annotations

import builtins
import hashlib
import math
import mmap
import os
import secrets
import stat
import struct
import sys
import threading
import traceback as traceback_module
import warnings
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from xvla.train import implicit_projective_dag_artifact as eager_artifact
from xvla.train.implicit_sparse_projective_odt import (
    MAXIMUM_RECTANGULAR_Q_ELEMENTS,
    PhysicalSourceSpec,
    RawPhysicalInput,
    _validate_direct_q_provenance,
)


_DTYPE_ALIGNMENT_BYTES = 8
_SHARD_ALIGNMENT_BYTES = 64
_READ_CHUNK_BYTES = 8 * 1024 * 1024
_INTEGER_MAXIMUM = (1 << 63) - 1
_BASE_EXCEPTION_GROUP_TYPE = getattr(builtins, "BaseExceptionGroup", None)
MAXIMUM_SEGMENTED_GATHER_BYTES = 64 * 1024 * 1024
PREFIX_ARTIFACT_SCHEMA = "xvla_implicit_projective_dag_prefix_artifact_v1"


class UnsupportedMappedTensorLayout(RuntimeError):
    """A valid archival tensor has no zero-copy kernel in this milestone."""


@dataclass(frozen=True)
class MappedSegment:
    shard_index: int
    offset_bytes: int
    byte_count: int


@dataclass(frozen=True)
class AlignedContiguousTensorSpan:
    shard_index: int
    offset_bytes: int
    byte_count: int


@dataclass(frozen=True)
class SegmentedTensorSpan:
    segments: tuple[MappedSegment, ...]
    reason: str


MappedTensorLayout = AlignedContiguousTensorSpan | SegmentedTensorSpan


@dataclass(frozen=True)
class MappedTensorDescriptor:
    index: int
    name: str
    shape: tuple[int, ...]
    byte_count: int
    layout: MappedTensorLayout

    @property
    def directly_executable(self) -> bool:
        return isinstance(self.layout, AlignedContiguousTensorSpan)


@dataclass(frozen=True)
class MappedNetworkDescriptor:
    artifact_schema: str
    head_tensor: int
    mask_tensor: int
    head_binary_exponent: int
    token_count: int
    feature_dimension: int
    selected_token: int
    claim_boundary: str
    algorithm1_direct_rq_complete: bool
    algorithm1_scale_ledger_complete: bool
    algorithm1_certified_head_binary_exponent: int | None
    physical_sources: tuple[PhysicalSourceSpec, ...]


@dataclass(frozen=True)
class MappedNodeDescriptor:
    index: int
    uid: int
    label: str
    origin_uid: int | None
    physical_token: int | None
    physical_source_key: str | None
    core_type: str
    core_kind: str
    children: tuple[int, ...]
    tensor_indices: tuple[int, ...]
    output_dimension: int
    input_dimensions: tuple[int, ...]


@dataclass(frozen=True)
class MappedExecutionReceipt:
    node_evaluations: int
    edge_occurrences_emitted: int
    released_nonroot_nodes: int
    peak_live_values: int
    edge_ledger_sha256: str
    root_only_live: bool
    all_refcounts_zero: bool
    segmented_tensor_gathers: int
    segmented_bytes_gathered: int


@dataclass(frozen=True)
class _PhysicalSnapshot:
    device: int
    inode: int
    mode: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int


def _clear_exception_graph_tracebacks(error: BaseException) -> None:
    """Release operands held by every completed frame in an exception graph."""

    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if current.__traceback__ is not None:
            traceback_module.clear_frames(current.__traceback__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if _BASE_EXCEPTION_GROUP_TYPE is not None and isinstance(
            current, _BASE_EXCEPTION_GROUP_TYPE
        ):
            pending.extend(current.exceptions)


def _snapshot(information: os.stat_result) -> _PhysicalSnapshot:
    return _PhysicalSnapshot(
        device=information.st_dev,
        inode=information.st_ino,
        mode=information.st_mode,
        links=information.st_nlink,
        size=information.st_size,
        modified_ns=information.st_mtime_ns,
        changed_ns=information.st_ctime_ns,
    )


def _require_directory(information: os.stat_result, label: str) -> None:
    if not stat.S_ISDIR(information.st_mode):
        raise ValueError(f"{label} must be a physical directory")
    if information.st_mode & 0o222:
        raise ValueError(f"{label} must be immutable")


def _require_member(
    information: os.stat_result,
    label: str,
    *,
    expected_byte_count: int | None = None,
) -> None:
    if not stat.S_ISREG(information.st_mode):
        raise ValueError(f"{label} must be a physical regular file")
    if information.st_nlink != 1:
        raise ValueError(f"{label} must not be hard-linked")
    if information.st_mode & 0o222:
        raise ValueError(f"{label} must be immutable")
    if expected_byte_count is not None and information.st_size != expected_byte_count:
        raise ValueError(f"{label} byte count differs from the manifest")


def _open_flags(*, directory: bool) -> int:
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if directory:
        flags |= os.O_DIRECTORY
    else:
        # A hostile FIFO must not block before fstat rejects it as non-regular.
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _require_secure_descriptor_platform() -> None:
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NONBLOCK")
    ):
        raise RuntimeError("mapped artifact loading requires no-follow directory opens")
    if os.open not in os.supports_dir_fd or os.listdir not in os.supports_fd:
        raise RuntimeError("mapped artifact loading requires directory-descriptor operations")
    if not hasattr(os, "pread"):
        raise RuntimeError("mapped artifact loading requires positioned descriptor reads")


def _open_root(source: str | os.PathLike[str]) -> tuple[int, _PhysicalSnapshot]:
    path = Path(source)
    before = os.lstat(path)
    _require_directory(before, "artifact root")
    descriptor = os.open(path, _open_flags(directory=True))
    try:
        opened = os.fstat(descriptor)
        _require_directory(opened, "artifact root")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("artifact root changed while opening")
        return descriptor, _snapshot(opened)
    except BaseException:
        os.close(descriptor)
        raise


def _open_member(
    root_descriptor: int,
    name: str,
    label: str,
    *,
    expected_byte_count: int | None = None,
) -> tuple[int, _PhysicalSnapshot]:
    if (
        type(name) is not str
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\x00" in name
    ):
        raise ValueError(f"{label} has an unsafe name")
    descriptor = os.open(
        name,
        _open_flags(directory=False),
        dir_fd=root_descriptor,
    )
    try:
        information = os.fstat(descriptor)
        _require_member(
            information,
            label,
            expected_byte_count=expected_byte_count,
        )
        return descriptor, _snapshot(information)
    except BaseException:
        os.close(descriptor)
        raise


def _read_descriptor(descriptor: int, byte_count: int, label: str) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < byte_count:
        chunk = os.pread(
            descriptor,
            min(_READ_CHUNK_BYTES, byte_count - offset),
            offset,
        )
        if not chunk:
            raise ValueError(f"{label} ended before its recorded byte count")
        chunks.append(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, byte_count):
        raise ValueError(f"{label} is longer than its recorded byte count")
    return b"".join(chunks)


def _sha256_descriptor(descriptor: int, byte_count: int, label: str) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < byte_count:
        chunk = os.pread(
            descriptor,
            min(_READ_CHUNK_BYTES, byte_count - offset),
            offset,
        )
        if not chunk:
            raise ValueError(f"{label} ended before its recorded byte count")
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, byte_count):
        raise ValueError(f"{label} is longer than its recorded byte count")
    return digest.hexdigest()


def _require_unchanged_descriptor(
    descriptor: int,
    expected: _PhysicalSnapshot,
    label: str,
    *,
    directory: bool,
) -> None:
    information = os.fstat(descriptor)
    if directory:
        _require_directory(information, label)
    else:
        _require_member(information, label, expected_byte_count=expected.size)
    if _snapshot(information) != expected:
        raise ValueError(f"{label} changed after authentication")


def _validate_network_descriptor(
    value: Any,
    physical_sources: tuple[PhysicalSourceSpec, ...],
    artifact_schema: str,
) -> MappedNetworkDescriptor:
    if not isinstance(value, dict):
        raise ValueError("network must be an object")
    eager_artifact._require_exact_keys(
        value,
        {
            "head_tensor",
            "mask_tensor",
            "head_binary_exponent",
            "token_count",
            "feature_dimension",
            "selected_token",
            "claim_boundary",
            "algorithm1_direct_rq_complete",
            "algorithm1_scale_ledger_complete",
            "algorithm1_certified_head_binary_exponent",
        },
        "network",
    )
    head_tensor = eager_artifact._require_integer(
        value["head_tensor"], "head tensor", minimum=0
    )
    mask_tensor = eager_artifact._require_integer(
        value["mask_tensor"], "mask tensor", minimum=0
    )
    if head_tensor != 0 or mask_tensor != 1:
        raise ValueError("network head and mask references are not canonical")
    token_count = eager_artifact._require_integer(
        value["token_count"], "token count", minimum=1
    )
    feature_dimension = eager_artifact._require_integer(
        value["feature_dimension"], "feature dimension", minimum=1
    )
    selected_token = eager_artifact._require_integer(
        value["selected_token"],
        "selected token",
        minimum=0,
        maximum=token_count - 1,
    )
    direct_complete = eager_artifact._require_bool(
        value["algorithm1_direct_rq_complete"], "Algorithm 1 completion"
    )
    scale_complete = eager_artifact._require_bool(
        value["algorithm1_scale_ledger_complete"],
        "Algorithm 1 scale ledger completion",
    )
    head_exponent = eager_artifact._require_integer(
        value["head_binary_exponent"], "head exponent"
    )
    certified = eager_artifact._require_optional_integer(
        value["algorithm1_certified_head_binary_exponent"],
        "certified head exponent",
    )
    if artifact_schema == eager_artifact.ARTIFACT_SCHEMA:
        if not direct_complete or not scale_complete:
            raise ValueError(
                "canonical mapped artifact requires completed Algorithm 1 ledgers"
            )
        if certified is None or certified != head_exponent:
            raise ValueError(
                "canonical boundary-head exponent differs from its scale ledger"
            )
    elif artifact_schema == PREFIX_ARTIFACT_SCHEMA:
        if direct_complete or scale_complete or certified is not None:
            raise ValueError(
                "prefix mapped artifact must clear every full-rank canonical flag"
            )
    else:
        raise ValueError("artifact schema differs")
    return MappedNetworkDescriptor(
        artifact_schema=artifact_schema,
        head_tensor=head_tensor,
        mask_tensor=mask_tensor,
        head_binary_exponent=head_exponent,
        token_count=token_count,
        feature_dimension=feature_dimension,
        selected_token=selected_token,
        claim_boundary=eager_artifact._require_nonempty_string(
            value["claim_boundary"], "claim boundary"
        ),
        algorithm1_direct_rq_complete=direct_complete,
        algorithm1_scale_ledger_complete=scale_complete,
        algorithm1_certified_head_binary_exponent=certified,
        physical_sources=physical_sources,
    )


def _validate_tensor_descriptors(
    value: Any,
    shard_records: Sequence[tuple[str, int, str]],
    maximum_bytes: int,
) -> tuple[MappedTensorDescriptor, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("tensors must be a nonempty list")
    metadata: list[tuple[int, str, tuple[int, ...], int, tuple[MappedSegment, ...]]] = []
    names: set[str] = set()
    total_declared = 0
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError("each tensor record must be an object")
        eager_artifact._require_exact_keys(
            item,
            {"index", "name", "shape", "byte_count", "segments"},
            "tensor",
        )
        if eager_artifact._require_integer(
            item["index"], "tensor index", minimum=0
        ) != index:
            raise ValueError("tensor indices are not canonical")
        name = eager_artifact._require_nonempty_string(item["name"], "tensor name")
        if name in names:
            raise ValueError("tensor names must be unique")
        names.add(name)
        shape_value = item["shape"]
        if not isinstance(shape_value, list) or not shape_value:
            raise ValueError("tensor shape must be a nonempty list")
        shape = tuple(
            eager_artifact._require_integer(
                dimension, "tensor dimension", minimum=1
            )
            for dimension in shape_value
        )
        byte_count = eager_artifact._require_integer(
            item["byte_count"], "tensor byte count", minimum=8
        )
        if byte_count != math.prod(shape) * 8:
            raise ValueError("tensor byte count differs from its float64 shape")
        segment_values = item["segments"]
        if not isinstance(segment_values, list) or not segment_values:
            raise ValueError("tensor segments must be a nonempty list")
        segments: list[MappedSegment] = []
        for segment_value in segment_values:
            if not isinstance(segment_value, dict):
                raise ValueError("tensor segment must be an object")
            eager_artifact._require_exact_keys(
                segment_value,
                {"shard_index", "offset_bytes", "byte_count"},
                "tensor segment",
            )
            segment = MappedSegment(
                shard_index=eager_artifact._require_integer(
                    segment_value["shard_index"],
                    "segment shard index",
                    minimum=0,
                    maximum=len(shard_records) - 1,
                ),
                offset_bytes=eager_artifact._require_integer(
                    segment_value["offset_bytes"],
                    "segment offset",
                    minimum=0,
                ),
                byte_count=eager_artifact._require_integer(
                    segment_value["byte_count"],
                    "segment byte count",
                    minimum=1,
                ),
            )
            if (
                segment.offset_bytes % 8
                or segment.byte_count % 8
                or segment.offset_bytes + segment.byte_count
                > shard_records[segment.shard_index][1]
            ):
                raise ValueError("tensor segment is not an in-range float64 span")
            segments.append(segment)
        metadata.append((index, name, shape, byte_count, tuple(segments)))
        total_declared += byte_count

    expected_shard_count = (total_declared + maximum_bytes - 1) // maximum_bytes
    if len(shard_records) != expected_shard_count:
        raise ValueError("shard count differs from canonical contiguous packing")
    expected_last_size = total_declared - maximum_bytes * (expected_shard_count - 1)
    if shard_records[-1][1] != expected_last_size:
        raise ValueError("final shard byte count differs from canonical packing")

    expected_shard_index = 0
    expected_offset = 0
    result: list[MappedTensorDescriptor] = []
    for index, name, shape, byte_count, segments in metadata:
        remaining = byte_count
        observed: list[MappedSegment] = []
        while remaining:
            if expected_offset == maximum_bytes:
                expected_shard_index += 1
                expected_offset = 0
            count = min(remaining, maximum_bytes - expected_offset)
            observed.append(
                MappedSegment(expected_shard_index, expected_offset, count)
            )
            expected_offset += count
            remaining -= count
        if segments != tuple(observed):
            raise ValueError(f"tensor {index} segments differ from canonical packing")
        if (
            len(segments) == 1
            and segments[0].offset_bytes % _DTYPE_ALIGNMENT_BYTES == 0
        ):
            only = segments[0]
            layout: MappedTensorLayout = AlignedContiguousTensorSpan(
                only.shard_index,
                only.offset_bytes,
                only.byte_count,
            )
        else:
            reason = "cross_shard" if len(segments) > 1 else "unaligned_start"
            layout = SegmentedTensorSpan(segments=segments, reason=reason)
        result.append(
            MappedTensorDescriptor(index, name, shape, byte_count, layout)
        )
    if len(result) < 2:
        raise ValueError("tensor table must contain head and mask")
    if result[0].name != "network.head" or result[1].name != "network.mask":
        raise ValueError("network tensor names differ")
    return tuple(result)


def _payload_tensor_view(
    mapping: mmap.mmap,
    *,
    byte_count: int,
    offset_bytes: int,
) -> Tensor:
    # PyTorch warns for every non-writable buffer even when the tensor remains
    # strictly internal.  No mapped tensor escapes this module and every use is
    # read-only, while ACCESS_READ provides the operating-system protection.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given buffer is not writable.*",
            category=UserWarning,
        )
        return torch.frombuffer(
            mapping,
            dtype=torch.float64,
            count=byte_count // 8,
            offset=offset_bytes,
        )


def _gather_segmented_tensor_view(
    descriptor: MappedTensorDescriptor,
    layout: SegmentedTensorSpan,
    mappings: Sequence[mmap.mmap],
) -> Tensor:
    """Gather exactly one authenticated cross-shard tensor into owned storage."""

    buffer = bytearray(descriptor.byte_count)
    destination = memoryview(buffer)
    cursor = 0
    try:
        for segment in layout.segments:
            whole_source = memoryview(mappings[segment.shard_index])
            source = whole_source[
                segment.offset_bytes : segment.offset_bytes + segment.byte_count
            ]
            try:
                destination[cursor : cursor + segment.byte_count] = source
            finally:
                source.release()
                whole_source.release()
            cursor += segment.byte_count
    finally:
        destination.release()
    if cursor != descriptor.byte_count:
        raise RuntimeError("segmented tensor gather byte ledger is incomplete")
    return torch.frombuffer(buffer, dtype=torch.float64).reshape(descriptor.shape)


def _validate_payloads(
    tensors: Sequence[MappedTensorDescriptor],
    mappings: Sequence[mmap.mmap],
    *,
    mask_index: int,
) -> None:
    for descriptor in tensors:
        layout = descriptor.layout
        segments = (
            (
                MappedSegment(
                    layout.shard_index,
                    layout.offset_bytes,
                    layout.byte_count,
                ),
            )
            if isinstance(layout, AlignedContiguousTensorSpan)
            else layout.segments
        )
        observed_elements = 0
        for segment in segments:
            chunk_offset = 0
            while chunk_offset < segment.byte_count:
                chunk_byte_count = min(
                    _READ_CHUNK_BYTES,
                    segment.byte_count - chunk_offset,
                )
                values = _payload_tensor_view(
                    mappings[segment.shard_index],
                    byte_count=chunk_byte_count,
                    offset_bytes=segment.offset_bytes + chunk_offset,
                )
                try:
                    if not bool(torch.isfinite(values).all()):
                        raise ValueError(
                            f"tensor {descriptor.name} contains a nonfinite value"
                        )
                    if descriptor.index == mask_index and not bool(
                        ((values == 0) | (values == 1)).all()
                    ):
                        raise ValueError("implicit fixed mask must be binary")
                    observed_elements += int(values.numel())
                finally:
                    del values
                chunk_offset += chunk_byte_count
        if observed_elements != math.prod(descriptor.shape):
            raise RuntimeError("mapped tensor span ledger is incomplete")


class _MappedNodeColumns:
    def __init__(
        self,
        *,
        uid: tuple[int, ...],
        label: tuple[str, ...],
        origin_uid: tuple[int | None, ...],
        physical_token: tuple[int | None, ...],
        physical_source_key: tuple[str | None, ...],
        core_type: tuple[str, ...],
        core_kind: tuple[str, ...],
        child_offsets: tuple[int, ...],
        child_indices: tuple[int, ...],
        incoming_refcount: tuple[int, ...],
        tensor_indices: tuple[tuple[int, ...], ...],
        output_dimensions: tuple[int, ...],
        input_dimensions: tuple[tuple[int, ...], ...],
    ) -> None:
        self.uid = uid
        self.label = label
        self.origin_uid = origin_uid
        self.physical_token = physical_token
        self.physical_source_key = physical_source_key
        self.core_type = core_type
        self.core_kind = core_kind
        self.child_offsets = child_offsets
        self.child_indices = child_indices
        self.incoming_refcount = incoming_refcount
        self.tensor_indices = tensor_indices
        self.output_dimensions = output_dimensions
        self.input_dimensions = input_dimensions

    @property
    def node_count(self) -> int:
        return len(self.uid)

    @property
    def root_index(self) -> int:
        return self.node_count - 1

    def children(self, index: int) -> tuple[int, ...]:
        start, stop = self.child_offsets[index : index + 2]
        return self.child_indices[start:stop]

    def descriptor(self, index: int) -> MappedNodeDescriptor:
        if type(index) is not int or not 0 <= index < self.node_count:
            raise IndexError("mapped node index is out of range")
        return MappedNodeDescriptor(
            index=index,
            uid=self.uid[index],
            label=self.label[index],
            origin_uid=self.origin_uid[index],
            physical_token=self.physical_token[index],
            physical_source_key=self.physical_source_key[index],
            core_type=self.core_type[index],
            core_kind=self.core_kind[index],
            children=self.children(index),
            tensor_indices=self.tensor_indices[index],
            output_dimension=self.output_dimensions[index],
            input_dimensions=self.input_dimensions[index],
        )


def _validate_depth_first_postorder(
    child_offsets: tuple[int, ...],
    child_indices: tuple[int, ...],
    root_index: int,
) -> None:
    node_count = root_index + 1
    state = bytearray(node_count)
    stack: list[tuple[int, int]] = [(root_index, child_offsets[root_index])]
    state[root_index] = 1
    expected_postorder = 0
    while stack:
        node_index, edge_cursor = stack[-1]
        edge_stop = child_offsets[node_index + 1]
        if edge_cursor < edge_stop:
            child_index = child_indices[edge_cursor]
            stack[-1] = (node_index, edge_cursor + 1)
            marker = state[child_index]
            if marker == 1:
                raise ValueError("graph contains a cycle")
            if marker == 0:
                state[child_index] = 1
                stack.append((child_index, child_offsets[child_index]))
        else:
            stack.pop()
            if state[node_index] != 2:
                state[node_index] = 2
                if node_index != expected_postorder:
                    raise ValueError(
                        "graph rows are not in canonical depth-first postorder"
                    )
                expected_postorder += 1
    if expected_postorder != node_count or any(marker != 2 for marker in state):
        raise ValueError("not every graph node is reachable from the root")


def _validate_graph_descriptor(
    graph: Any,
    tensors: Sequence[MappedTensorDescriptor],
    network: MappedNetworkDescriptor,
) -> _MappedNodeColumns:
    if not isinstance(graph, dict):
        raise ValueError("graph must be an object")
    column_names = {
        "uid",
        "label",
        "origin_uid",
        "physical_token",
        "physical_source_key",
        "core_type",
        "core_kind",
        "core_binary_exponent",
        "direct_q_provenance",
        "matrix_tensor",
        "output_factor_tensor",
        "left_factor_tensor",
        "right_factor_tensor",
        "q_rows_tensor",
        "left_dimension",
        "right_dimension",
    }
    eager_artifact._require_exact_keys(
        graph,
        {
            "order",
            "node_count",
            "root_index",
            "child_offsets",
            "child_indices",
            "incoming_refcount",
            *column_names,
        },
        "graph",
    )
    if graph["order"] != "depth_first_child_order_postorder":
        raise ValueError("graph ordering identifier differs")
    node_count = eager_artifact._require_integer(
        graph["node_count"], "node count", minimum=1
    )
    root_index = eager_artifact._require_integer(
        graph["root_index"],
        "root index",
        minimum=0,
        maximum=node_count - 1,
    )
    if root_index != node_count - 1:
        raise ValueError("canonical graph root must be the final postorder node")
    columns = {
        name: eager_artifact._column(graph, name, node_count)
        for name in column_names
    }
    offsets_value = graph["child_offsets"]
    children_value = graph["child_indices"]
    incoming_value = graph["incoming_refcount"]
    if (
        not isinstance(offsets_value, list)
        or len(offsets_value) != node_count + 1
        or not isinstance(children_value, list)
        or not isinstance(incoming_value, list)
        or len(incoming_value) != node_count
    ):
        raise ValueError("graph edge columns have inconsistent lengths")
    offsets = tuple(
        eager_artifact._require_integer(value, "child offset", minimum=0)
        for value in offsets_value
    )
    if (
        offsets[0] != 0
        or offsets[-1] != len(children_value)
        or any(left > right for left, right in zip(offsets, offsets[1:]))
    ):
        raise ValueError("child offsets are not a canonical offset column")
    children = tuple(
        eager_artifact._require_integer(
            value,
            "child index",
            minimum=0,
            maximum=node_count - 1,
        )
        for value in children_value
    )
    observed_incoming = array("q", [0]) * node_count
    for parent_index in range(node_count):
        for child_index in children[offsets[parent_index] : offsets[parent_index + 1]]:
            if child_index >= parent_index:
                raise ValueError("graph is not strict postorder")
            observed_incoming[child_index] += 1
            if observed_incoming[child_index] > _INTEGER_MAXIMUM:
                raise ValueError("graph occurrence refcount exceeds its integer range")
    incoming = tuple(
        eager_artifact._require_integer(
            value,
            "incoming refcount",
            minimum=0,
            maximum=_INTEGER_MAXIMUM,
        )
        for value in incoming_value
    )
    if incoming != tuple(observed_incoming):
        raise ValueError("stored incoming refcounts differ from graph edges")
    if incoming[root_index] != 0 or any(
        count < 1 for index, count in enumerate(incoming) if index != root_index
    ):
        raise ValueError("graph contains an unreachable node")
    _validate_depth_first_postorder(offsets, children, root_index)

    uids = tuple(
        eager_artifact._require_integer(value, "node uid", minimum=0)
        for value in columns["uid"]
    )
    if len(set(uids)) != node_count:
        raise ValueError("node uids must be unique")
    labels = tuple(
        eager_artifact._require_nonempty_string(value, "node label")
        for value in columns["label"]
    )
    origins = tuple(
        eager_artifact._require_optional_integer(
            value, "origin uid", minimum=0
        )
        for value in columns["origin_uid"]
    )
    physical_tokens = tuple(
        eager_artifact._require_optional_integer(
            value, "physical token", minimum=0
        )
        for value in columns["physical_token"]
    )
    source_keys = tuple(
        None
        if value is None
        else eager_artifact._require_nonempty_string(
            value, "physical source key"
        )
        for value in columns["physical_source_key"]
    )
    core_types = tuple(columns["core_type"])
    core_kinds = tuple(
        eager_artifact._require_nonempty_string(value, "core kind")
        for value in columns["core_kind"]
    )
    exponents = tuple(
        eager_artifact._require_integer(value, "core exponent")
        for value in columns["core_binary_exponent"]
    )
    if any(exponent != 0 for exponent in exponents):
        raise ValueError("canonical core exponent is not zero")

    next_tensor_index = 2
    tensor_indices: list[tuple[int, ...]] = []
    output_dimensions: list[int] = []
    input_dimensions: list[tuple[int, ...]] = []
    source_registry = {spec.key: spec for spec in network.physical_sources}
    reached_sources: set[str] = set()

    def reference(column_name: str, node_index: int) -> int | None:
        nonlocal next_tensor_index
        value = columns[column_name][node_index]
        if value is None:
            return None
        result = eager_artifact._require_integer(
            value,
            f"{column_name} reference",
            minimum=0,
            maximum=len(tensors) - 1,
        )
        if result != next_tensor_index:
            raise ValueError("tensor references are not in canonical order")
        next_tensor_index += 1
        suffix = column_name.removesuffix("_tensor")
        if tensors[result].name != f"node[{node_index}].{suffix}":
            raise ValueError("tensor name differs from its graph reference")
        return result

    for index in range(node_count):
        matrix = reference("matrix_tensor", index)
        output_factor = reference("output_factor_tensor", index)
        left_factor = reference("left_factor_tensor", index)
        right_factor = reference("right_factor_tensor", index)
        q_rows = reference("q_rows_tensor", index)
        provenance = eager_artifact._provenance_from_json(
            columns["direct_q_provenance"][index],
            f"node[{index}] provenance",
        )
        left_dimension = eager_artifact._require_optional_integer(
            columns["left_dimension"][index],
            "left dimension",
            minimum=1,
        )
        right_dimension = eager_artifact._require_optional_integer(
            columns["right_dimension"][index],
            "right dimension",
            minimum=1,
        )
        core_type = core_types[index]
        if core_type == "unary":
            if (
                matrix is None
                or any(
                    value is not None
                    for value in (output_factor, left_factor, right_factor, q_rows)
                )
                or provenance is not None
                or left_dimension is not None
                or right_dimension is not None
            ):
                raise ValueError("unary metadata differs")
            shape = tensors[matrix].shape
            if len(shape) != 2:
                raise ValueError("implicit unary core must be a matrix")
            node_tensors = (matrix,)
            output_dimension = shape[0]
            node_inputs = (shape[1],)
        elif core_type == "cp_binary":
            canonical = network.artifact_schema == eager_artifact.ARTIFACT_SCHEMA
            if (
                matrix is not None
                or q_rows is not None
                or any(
                    value is None
                    for value in (output_factor, left_factor, right_factor)
                )
                or left_dimension is not None
                or right_dimension is not None
                or (canonical and provenance is None)
                or (not canonical and provenance is not None)
            ):
                raise ValueError("CP metadata differs")
            assert output_factor is not None
            assert left_factor is not None
            assert right_factor is not None
            output_shape = tensors[output_factor].shape
            left_shape = tensors[left_factor].shape
            right_shape = tensors[right_factor].shape
            if any(len(shape) != 2 for shape in (output_shape, left_shape, right_shape)):
                raise ValueError("implicit CP factors must be matrices")
            rank = output_shape[1]
            if rank < 1 or left_shape[0] != rank or right_shape[0] != rank:
                raise ValueError("implicit CP factors must have one positive common rank")
            if canonical:
                assert provenance is not None
                _validate_direct_q_provenance(provenance, require_verified=True)
                if provenance.columns_compared != left_shape[1] * right_shape[1]:
                    raise ValueError(
                        "canonical compact CP Q provenance does not cover every column"
                    )
            node_tensors = (output_factor, left_factor, right_factor)
            output_dimension = output_shape[0]
            node_inputs = (left_shape[1], right_shape[1])
        elif core_type == "reduced_q_binary":
            if (
                q_rows is None
                or any(
                    value is not None
                    for value in (matrix, output_factor, left_factor, right_factor)
                )
                or provenance is not None
                or left_dimension is None
                or right_dimension is None
            ):
                raise ValueError("reduced-Q metadata differs")
            q_shape = tensors[q_rows].shape
            if len(q_shape) != 2:
                raise ValueError("reduced-Q binary rows must be a matrix")
            if q_shape[1] != left_dimension * right_dimension:
                raise ValueError("reduced-Q binary rows/input dimensions disagree")
            if math.prod(q_shape) > MAXIMUM_RECTANGULAR_Q_ELEMENTS:
                raise ValueError("reduced-Q binary core exceeds its bounded size")
            node_tensors = (q_rows,)
            output_dimension = q_shape[0]
            node_inputs = (left_dimension, right_dimension)
        else:
            raise ValueError("core type is unsupported")

        node_children = children[offsets[index] : offsets[index + 1]]
        if node_children:
            if physical_tokens[index] is not None or source_keys[index] is not None:
                raise ValueError("only physical leaves may select a raw source")
            if len(node_children) != len(node_inputs):
                raise ValueError("implicit node/core arity mismatch")
            if tuple(output_dimensions[child] for child in node_children) != node_inputs:
                raise ValueError("implicit child/core bond mismatch")
        else:
            if len(node_inputs) != 1:
                raise ValueError("implicit physical leaf must be unary")
            if source_registry:
                key = source_keys[index]
                if physical_tokens[index] is not None or key not in source_registry:
                    raise ValueError(
                        "heterogeneous leaf must select one declared physical source"
                    )
                assert key is not None
                reached_sources.add(key)
                expected_input = source_registry[key].width + 1
            else:
                token = physical_tokens[index]
                if (
                    source_keys[index] is not None
                    or token is None
                    or not 0 <= token < network.token_count
                ):
                    raise ValueError("implicit leaf must select one physical token")
                expected_input = network.feature_dimension + 1
            if node_inputs[0] != expected_input:
                raise ValueError("implicit physical leaf has the wrong homogeneous width")
        tensor_indices.append(node_tensors)
        output_dimensions.append(output_dimension)
        input_dimensions.append(node_inputs)

    if next_tensor_index != len(tensors):
        raise ValueError("tensor table contains an unreferenced tensor")
    if source_registry and reached_sources != set(source_registry):
        raise ValueError(
            "heterogeneous source registry must match reachable physical leaves"
        )
    head_shape = tensors[network.head_tensor].shape
    mask_shape = tensors[network.mask_tensor].shape
    if len(head_shape) != 2 or head_shape[1] != output_dimensions[root_index]:
        raise ValueError("implicit boundary head/root dimensions disagree")
    if mask_shape != (network.token_count, network.token_count):
        raise ValueError("implicit fixed mask has the wrong shape")

    return _MappedNodeColumns(
        uid=uids,
        label=labels,
        origin_uid=origins,
        physical_token=physical_tokens,
        physical_source_key=source_keys,
        core_type=tuple(
            eager_artifact._require_nonempty_string(value, "core type")
            for value in core_types
        ),
        core_kind=core_kinds,
        child_offsets=offsets,
        child_indices=children,
        incoming_refcount=incoming,
        tensor_indices=tuple(tensor_indices),
        output_dimensions=tuple(output_dimensions),
        input_dimensions=tuple(input_dimensions),
    )


def _execution_layout_plan(
    tensors: Sequence[MappedTensorDescriptor],
    network: MappedNetworkDescriptor,
    nodes: _MappedNodeColumns,
    maximum_segmented_gather_bytes: int,
) -> tuple[str | None, int, int]:
    gather_count = 0
    gathered_bytes = 0

    def account(label: str, indices: Sequence[int]) -> str | None:
        nonlocal gather_count, gathered_bytes
        group_bytes = 0
        for index in indices:
            descriptor = tensors[index]
            layout = descriptor.layout
            if isinstance(layout, AlignedContiguousTensorSpan):
                continue
            if layout.reason != "cross_shard":
                return (
                    f"tensor {descriptor.index} {descriptor.name!r} uses unsupported "
                    f"{layout.reason} mapped layout"
                )
            group_bytes += descriptor.byte_count
            gather_count += 1
            gathered_bytes += descriptor.byte_count
        if group_bytes > maximum_segmented_gather_bytes:
            return (
                f"{label} requires {group_bytes} segmented gather bytes, above "
                f"the {maximum_segmented_gather_bytes}-byte per-node cap"
            )
        return None

    error = account("boundary head", (network.head_tensor,))
    if error is not None:
        return error, gather_count, gathered_bytes
    for node_index, indices in enumerate(nodes.tensor_indices):
        error = account(f"node {node_index}", indices)
        if error is not None:
            return error, gather_count, gathered_bytes
    return None, gather_count, gathered_bytes


class MappedImplicitProjectiveDAGArtifact:
    """An authenticated descriptor graph whose tensor bytes stay mapped read-only."""

    def __init__(
        self,
        *,
        root_descriptor: int,
        root_snapshot: _PhysicalSnapshot,
        member_descriptors: dict[str, int],
        member_snapshots: dict[str, _PhysicalSnapshot],
        shard_names: tuple[str, ...],
        mappings: tuple[mmap.mmap, ...],
        manifest_sha256: str,
        maximum_segmented_gather_bytes: int,
        network: MappedNetworkDescriptor,
        tensors: tuple[MappedTensorDescriptor, ...],
        nodes: _MappedNodeColumns,
    ) -> None:
        self._root_descriptor = root_descriptor
        self._root_snapshot = root_snapshot
        self._member_descriptors = member_descriptors
        self._member_snapshots = member_snapshots
        self._shard_names = shard_names
        self._mappings = mappings
        self._network = network
        self._tensors = tensors
        self._nodes = nodes
        self._manifest_sha256 = manifest_sha256
        self._maximum_segmented_gather_bytes = maximum_segmented_gather_bytes
        (
            self._execution_layout_error,
            self._segmented_tensor_gathers,
            self._segmented_bytes_gathered,
        ) = _execution_layout_plan(
            tensors,
            network,
            nodes,
            maximum_segmented_gather_bytes,
        )
        self._closed = False
        self._close_pending = False
        self._closed_mapping_indices: set[int] = set()
        self._active_evaluations = 0
        self._lifecycle_lock = threading.RLock()

    def __enter__(self) -> "MappedImplicitProjectiveDAGArtifact":
        with self._lifecycle_lock:
            self._require_open()
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception: Any,
        traceback_object: Any,
    ) -> None:
        if isinstance(exception, BaseException):
            # A failing Python tensor wrapper can retain mmap-backed operands in
            # any traceback in the chained or grouped exception graph.  Those
            # references must be released before close() invalidates storage.
            _clear_exception_graph_tracebacks(exception)
        elif traceback_object is not None:
            traceback_module.clear_frames(traceback_object)
        self.close()

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    @property
    def network(self) -> MappedNetworkDescriptor:
        return self._network

    @property
    def tensors(self) -> tuple[MappedTensorDescriptor, ...]:
        return self._tensors

    @property
    def node_count(self) -> int:
        return self._nodes.node_count

    @property
    def edge_occurrence_count(self) -> int:
        return len(self._nodes.child_indices)

    @property
    def maximum_segmented_gather_bytes(self) -> int:
        return self._maximum_segmented_gather_bytes

    @property
    def execution_layout_supported(self) -> bool:
        return self._execution_layout_error is None

    @property
    def segmented_bytes_per_evaluation(self) -> int:
        return self._segmented_bytes_gathered

    @property
    def closed(self) -> bool:
        return self._closed

    def node(self, index: int) -> MappedNodeDescriptor:
        with self._lifecycle_lock:
            self._require_open()
            return self._nodes.descriptor(index)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("mapped artifact view is closed")
        if self._close_pending:
            raise RuntimeError(
                "mapped artifact close is incomplete; release exported views "
                "and retry close"
            )

    def assert_unchanged(self) -> None:
        """Validate the authenticated immutable cohort's inventory and stats.

        This intentionally does not rehash a potentially hundreds-of-GiB
        payload around every policy forward.  Byte authentication happens at
        open, while this gate detects changes observable through the validated
        descriptors and physical metadata.
        """
        with self._lifecycle_lock:
            self._require_open()
            _require_unchanged_descriptor(
                self._root_descriptor,
                self._root_snapshot,
                "artifact root",
                directory=True,
            )
            observed_names = set(os.listdir(self._root_descriptor))
            if observed_names != set(self._member_descriptors):
                raise ValueError("artifact inventory changed after authentication")
            for name, descriptor in self._member_descriptors.items():
                _require_unchanged_descriptor(
                    descriptor,
                    self._member_snapshots[name],
                    f"artifact member {name}",
                    directory=False,
                )

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            if self._active_evaluations:
                raise RuntimeError("cannot close a mapped artifact during evaluation")
            mapping_errors: list[BaseException] = []
            for index in reversed(range(len(self._mappings))):
                if index in self._closed_mapping_indices:
                    continue
                try:
                    self._mappings[index].close()
                except BaseException as error:
                    mapping_errors.append(error)
                else:
                    self._closed_mapping_indices.add(index)
            if mapping_errors:
                self._close_pending = True
                raise RuntimeError(
                    "mapped tensor storage is still exported; release all views "
                    "and retry close"
                ) from mapping_errors[0]

            descriptor_errors: list[BaseException] = []
            for name, descriptor in tuple(self._member_descriptors.items()):
                try:
                    os.close(descriptor)
                except BaseException as error:
                    descriptor_errors.append(error)
                else:
                    del self._member_descriptors[name]
            if self._root_descriptor >= 0:
                try:
                    os.close(self._root_descriptor)
                except BaseException as error:
                    descriptor_errors.append(error)
                else:
                    self._root_descriptor = -1
            if descriptor_errors:
                self._close_pending = True
                raise RuntimeError(
                    "mapped artifact descriptors did not close cleanly; retry close"
                ) from descriptor_errors[0]
            if (
                len(self._closed_mapping_indices) != len(self._mappings)
                or self._member_descriptors
                or self._root_descriptor >= 0
            ):
                raise RuntimeError("mapped artifact close ledger is incomplete")
            self._close_pending = False
            self._closed = True

    def _tensor_view(self, index: int) -> Tensor:
        self._require_open()
        descriptor = self._tensors[index]
        layout = descriptor.layout
        if isinstance(layout, AlignedContiguousTensorSpan):
            return _payload_tensor_view(
                self._mappings[layout.shard_index],
                byte_count=layout.byte_count,
                offset_bytes=layout.offset_bytes,
            ).reshape(descriptor.shape)
        if (
            layout.reason != "cross_shard"
            or descriptor.byte_count > self._maximum_segmented_gather_bytes
        ):
            raise UnsupportedMappedTensorLayout(
                f"tensor {index} {descriptor.name!r} uses unsupported "
                f"{layout.reason} mapped layout"
            )
        return _gather_segmented_tensor_view(descriptor, layout, self._mappings)

    @torch.no_grad()
    def evaluate_projective_boundary(
        self,
        raw_input: RawPhysicalInput,
        *,
        return_receipt: bool = False,
    ) -> Tensor | tuple[Tensor, MappedExecutionReceipt]:
        if type(return_receipt) is not bool:
            raise TypeError("return_receipt must be boolean")
        with self._lifecycle_lock:
            self._require_open()
            self._active_evaluations += 1
        try:
            self.assert_unchanged()
            if self._execution_layout_error is not None:
                raise UnsupportedMappedTensorLayout(self._execution_layout_error)
            batch_size, prepared = self._prepare_input(raw_input)
            output, receipt = self._execute(batch_size, prepared)
            self.assert_unchanged()
        except BaseException as error:
            # Evaluation failures can be caught and retained inside the mapped
            # context.  Clear completed tensor-wrapper frames here, rather than
            # relying only on __exit__, so their mmap-backed operands never
            # outlive the evaluation's ownership boundary.
            _clear_exception_graph_tracebacks(error)
            raise
        finally:
            with self._lifecycle_lock:
                self._active_evaluations -= 1
        if return_receipt:
            return output, receipt
        return output

    def _prepare_input(
        self, raw_input: RawPhysicalInput
    ) -> tuple[int, Tensor | dict[str, Tensor]]:
        specs = self._network.physical_sources
        if not specs:
            if not isinstance(raw_input, Tensor):
                raise TypeError("homogeneous implicit DAG input must be one tensor")
            if raw_input.ndim != 3 or tuple(raw_input.shape[1:]) != (
                self._network.token_count,
                self._network.feature_dimension,
            ):
                raise ValueError("raw input shape does not match implicit DAG")
            values: Tensor | dict[str, Tensor] = raw_input
            batch_size = int(raw_input.shape[0])
            candidates = (raw_input,)
        else:
            if isinstance(raw_input, Tensor):
                raise TypeError(
                    "heterogeneous implicit DAG input must be a mapping or tuple"
                )
            if isinstance(raw_input, Mapping):
                if set(raw_input) != {spec.key for spec in specs}:
                    raise ValueError(
                        "raw physical-source mapping keys do not match the registry"
                    )
                mapped = {spec.key: raw_input[spec.key] for spec in specs}
            else:
                sequence = tuple(raw_input)
                if len(sequence) != len(specs):
                    raise ValueError(
                        "raw physical-source tuple length does not match the registry"
                    )
                mapped = {
                    spec.key: value for spec, value in zip(specs, sequence)
                }
            batch_size: int | None = None
            for spec in specs:
                value = mapped[spec.key]
                if not isinstance(value, Tensor) or value.ndim != 2:
                    raise ValueError(
                        "each heterogeneous physical source must be a batch matrix"
                    )
                if value.shape[1] != spec.width:
                    raise ValueError(
                        f"physical source {spec.key!r} has the wrong width"
                    )
                if batch_size is None:
                    batch_size = int(value.shape[0])
                elif value.shape[0] != batch_size:
                    raise ValueError(
                        "heterogeneous physical sources have different batch sizes"
                    )
            assert batch_size is not None
            values = mapped
            candidates = tuple(mapped.values())
        if batch_size < 1:
            raise ValueError("mapped physical input must contain a nonempty batch")
        for value in candidates:
            if value.dtype != torch.float64 or value.device.type != "cpu":
                raise ValueError("mapped physical inputs must be CPU float64 tensors")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("mapped physical inputs must contain only finite values")
        return batch_size, values

    @torch.no_grad()
    def evaluate_boundary_quotient(
        self,
        raw_input: RawPhysicalInput,
        *,
        return_receipt: bool = False,
    ) -> Tensor | tuple[Tensor, MappedExecutionReceipt]:
        """Evaluate the projective pair and divide by its certified last entry."""

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

    def _apply_node(self, index: int, inputs: tuple[Tensor, ...]) -> Tensor:
        core_type = self._nodes.core_type[index]
        tensor_indices = self._nodes.tensor_indices[index]
        if core_type == "unary":
            if len(inputs) != 1:
                raise RuntimeError("mapped unary execution input ledger differs")
            matrix = self._tensor_view(tensor_indices[0])
            try:
                return inputs[0] @ matrix.T
            finally:
                del matrix
        if core_type == "cp_binary":
            if len(inputs) != 2:
                raise RuntimeError("mapped CP execution input ledger differs")
            output_factor = self._tensor_view(tensor_indices[0])
            left_factor = self._tensor_view(tensor_indices[1])
            right_factor = self._tensor_view(tensor_indices[2])
            try:
                left = inputs[0] @ left_factor.T
                right = inputs[1] @ right_factor.T
                return (left * right) @ output_factor.T
            finally:
                del output_factor, left_factor, right_factor
        if core_type == "reduced_q_binary":
            if len(inputs) != 2:
                raise RuntimeError("mapped reduced-Q execution input ledger differs")
            q_rows = self._tensor_view(tensor_indices[0])
            left_dimension, right_dimension = self._nodes.input_dimensions[index]
            try:
                tensor = q_rows.reshape(
                    self._nodes.output_dimensions[index],
                    left_dimension,
                    right_dimension,
                )
                try:
                    return torch.einsum(
                        "oij,bi,bj->bo", tensor, inputs[0], inputs[1]
                    )
                finally:
                    del tensor
            finally:
                del q_rows
        raise RuntimeError("mapped executor encountered an unsupported core")

    @staticmethod
    def _normalize(value: Tensor) -> Tensor:
        maximum = value.abs().amax(dim=1)
        if bool((maximum == 0).any()) or not bool(torch.isfinite(maximum).all()):
            raise ValueError(
                "mapped projective evaluation produced zero/nonfinite coordinates"
            )
        _, exponent = torch.frexp(maximum)
        return torch.ldexp(value, -exponent[:, None])

    def _execute(
        self,
        batch_size: int,
        prepared: Tensor | dict[str, Tensor],
    ) -> tuple[Tensor, MappedExecutionReceipt]:
        node_count = self._nodes.node_count
        root_index = self._nodes.root_index
        remaining = array("q", self._nodes.incoming_refcount)
        released = bytearray(node_count)
        live: dict[int, Tensor] = {}
        edge_digest = hashlib.sha256()
        edge_ordinal = 0
        peak_live = 0

        for index in range(node_count):
            children = self._nodes.children(index)
            if children:
                try:
                    child_values = tuple(live[child] for child in children)
                except KeyError as error:
                    raise RuntimeError(
                        "mapped executor observed a prematurely released child"
                    ) from error
                value = self._apply_node(index, child_values)
            else:
                source_key = self._nodes.physical_source_key[index]
                if source_key is not None:
                    if not isinstance(prepared, dict):
                        raise RuntimeError(
                            "heterogeneous mapped leaf received homogeneous input"
                        )
                    raw_value = prepared[source_key]
                else:
                    token = self._nodes.physical_token[index]
                    if token is None or not isinstance(prepared, Tensor):
                        raise RuntimeError("mapped leaf has no physical token")
                    raw_value = prepared[:, token]
                physical = torch.cat(
                    (raw_value, raw_value.new_ones(batch_size, 1)), dim=1
                )
                value = self._apply_node(index, (physical,))
                del physical
            live[index] = self._normalize(value)
            del value
            peak_live = max(peak_live, len(live))

            for role, child in enumerate(children):
                before = remaining[child]
                if before <= 0:
                    raise RuntimeError(
                        "mapped occurrence-refcount ledger underflowed"
                    )
                after = before - 1
                remaining[child] = after
                edge_digest.update(
                    struct.pack(
                        "<QQQQQQ",
                        edge_ordinal,
                        index,
                        role,
                        child,
                        before,
                        after,
                    )
                )
                edge_ordinal += 1
                if after == 0:
                    if released[child] or child not in live:
                        raise RuntimeError(
                            "mapped executor released a child more than once"
                        )
                    del live[child]
                    released[child] = 1
            if children:
                del child_values

        if edge_ordinal != len(self._nodes.child_indices):
            raise RuntimeError("mapped emitted-edge ledger is incomplete")
        if any(remaining):
            raise RuntimeError("mapped occurrence-refcount ledger has leftovers")
        if released[root_index] or any(
            not released[index] for index in range(node_count - 1)
        ):
            raise RuntimeError("mapped node-release ledger is incomplete")
        if set(live) != {root_index}:
            raise RuntimeError("mapped executor did not finish with root-only live")

        head = self._tensor_view(self._network.head_tensor)
        try:
            output = self._normalize(live[root_index] @ head.T)
        finally:
            del head
        receipt = MappedExecutionReceipt(
            node_evaluations=node_count,
            edge_occurrences_emitted=edge_ordinal,
            released_nonroot_nodes=int(sum(released)),
            peak_live_values=peak_live,
            edge_ledger_sha256=edge_digest.hexdigest(),
            root_only_live=True,
            all_refcounts_zero=True,
            segmented_tensor_gathers=self._segmented_tensor_gathers,
            segmented_bytes_gathered=self._segmented_bytes_gathered,
        )
        return output, receipt


def open_mapped_implicit_projective_dag_artifact(
    source: str | os.PathLike[str],
    *,
    expected_manifest_sha256: str,
    maximum_segmented_gather_bytes: int = MAXIMUM_SEGMENTED_GATHER_BYTES,
) -> MappedImplicitProjectiveDAGArtifact:
    """Authenticate and map one immutable v1 artifact without payload copies."""

    if sys.byteorder != "little":
        raise RuntimeError("mapped artifact loading requires a little-endian host")
    _require_secure_descriptor_platform()
    trusted_digest = eager_artifact._require_sha256(
        expected_manifest_sha256,
        "expected manifest sha256",
    )
    gather_cap = eager_artifact._require_integer(
        maximum_segmented_gather_bytes,
        "maximum segmented gather bytes",
        minimum=0,
        maximum=MAXIMUM_SEGMENTED_GATHER_BYTES,
    )
    root_descriptor = -1
    member_descriptors: dict[str, int] = {}
    mappings: list[mmap.mmap] = []
    try:
        root_descriptor, root_snapshot = _open_root(source)
        manifest_descriptor, manifest_snapshot = _open_member(
            root_descriptor,
            eager_artifact.MANIFEST_NAME,
            "artifact manifest",
        )
        member_descriptors[eager_artifact.MANIFEST_NAME] = manifest_descriptor
        manifest_bytes = _read_descriptor(
            manifest_descriptor,
            manifest_snapshot.size,
            "artifact manifest",
        )
        if not secrets.compare_digest(
            hashlib.sha256(manifest_bytes).hexdigest(), trusted_digest
        ):
            raise ValueError("artifact manifest digest differs from the trusted digest")
        manifest = eager_artifact._parse_canonical_json(manifest_bytes)
        eager_artifact._require_exact_keys(
            manifest,
            {
                "schema",
                "encoding",
                "network",
                "physical_sources",
                "graph",
                "tensors",
                "shards",
            },
            "manifest",
        )
        artifact_schema = manifest["schema"]
        if artifact_schema not in {
            eager_artifact.ARTIFACT_SCHEMA,
            PREFIX_ARTIFACT_SCHEMA,
        }:
            raise ValueError("artifact schema differs")
        encoding = manifest["encoding"]
        if not isinstance(encoding, dict):
            raise ValueError("encoding must be an object")
        eager_artifact._require_exact_keys(
            encoding,
            {"dtype", "byte_order", "layout", "maximum_shard_bytes"},
            "encoding",
        )
        if (
            encoding["dtype"] != "float64"
            or encoding["byte_order"] != "little"
            or encoding["layout"] != "c_contiguous"
        ):
            raise ValueError("tensor encoding differs")
        maximum_bytes = eager_artifact._require_integer(
            encoding["maximum_shard_bytes"],
            "maximum shard bytes",
            minimum=_SHARD_ALIGNMENT_BYTES,
            maximum=eager_artifact.MAXIMUM_SHARD_BYTES,
        )
        if maximum_bytes % _SHARD_ALIGNMENT_BYTES:
            raise ValueError("maximum shard bytes must be 64-byte aligned")
        shard_records = eager_artifact._validate_shard_table(
            manifest["shards"], maximum_bytes
        )
        expected_names = {
            eager_artifact.MANIFEST_NAME,
            *(record[0] for record in shard_records),
        }
        if set(os.listdir(root_descriptor)) != expected_names:
            raise ValueError("artifact contains a missing or extra file")

        member_snapshots = {
            eager_artifact.MANIFEST_NAME: manifest_snapshot
        }
        for name, byte_count, digest in shard_records:
            descriptor, snapshot = _open_member(
                root_descriptor,
                name,
                f"shard {name}",
                expected_byte_count=byte_count,
            )
            member_descriptors[name] = descriptor
            member_snapshots[name] = snapshot
            observed_digest = _sha256_descriptor(
                descriptor, byte_count, f"shard {name}"
            )
            if not secrets.compare_digest(observed_digest, digest):
                raise ValueError(f"shard {name} digest differs")
            mapping = mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ)
            mappings.append(mapping)
            _require_unchanged_descriptor(
                descriptor,
                snapshot,
                f"shard {name}",
                directory=False,
            )

        physical_sources = eager_artifact._parse_sources(
            manifest["physical_sources"]
        )
        network = _validate_network_descriptor(
            manifest["network"], physical_sources, artifact_schema
        )
        tensors = _validate_tensor_descriptors(
            manifest["tensors"], shard_records, maximum_bytes
        )
        _validate_payloads(
            tensors,
            mappings,
            mask_index=network.mask_tensor,
        )
        nodes = _validate_graph_descriptor(manifest["graph"], tensors, network)

        _require_unchanged_descriptor(
            root_descriptor,
            root_snapshot,
            "artifact root",
            directory=True,
        )
        if set(os.listdir(root_descriptor)) != expected_names:
            raise ValueError("artifact inventory changed during mapped loading")
        for name, descriptor in member_descriptors.items():
            _require_unchanged_descriptor(
                descriptor,
                member_snapshots[name],
                f"artifact member {name}",
                directory=False,
            )

        result = MappedImplicitProjectiveDAGArtifact(
            root_descriptor=root_descriptor,
            root_snapshot=root_snapshot,
            member_descriptors=member_descriptors,
            member_snapshots=member_snapshots,
            shard_names=tuple(record[0] for record in shard_records),
            mappings=tuple(mappings),
            manifest_sha256=trusted_digest,
            maximum_segmented_gather_bytes=gather_cap,
            network=network,
            tensors=tensors,
            nodes=nodes,
        )
        root_descriptor = -1
        member_descriptors = {}
        mappings = []
        return result
    except BaseException:
        for mapping in reversed(mappings):
            mapping.close()
        for descriptor in member_descriptors.values():
            os.close(descriptor)
        if root_descriptor >= 0:
            os.close(root_descriptor)
        raise


def evaluate_mapped_projective_boundary(
    artifact: MappedImplicitProjectiveDAGArtifact,
    raw_input: RawPhysicalInput,
    *,
    return_receipt: bool = False,
) -> Tensor | tuple[Tensor, MappedExecutionReceipt]:
    """Execute one already-open mapped artifact through the row-order engine."""

    if not isinstance(artifact, MappedImplicitProjectiveDAGArtifact):
        raise TypeError("artifact must be an open mapped DAG view")
    return artifact.evaluate_projective_boundary(
        raw_input,
        return_receipt=return_receipt,
    )


def evaluate_mapped_boundary_quotient(
    artifact: MappedImplicitProjectiveDAGArtifact,
    raw_input: RawPhysicalInput,
    *,
    return_receipt: bool = False,
) -> Tensor | tuple[Tensor, MappedExecutionReceipt]:
    """Evaluate an action-like quotient through an already-open mapped view."""

    if not isinstance(artifact, MappedImplicitProjectiveDAGArtifact):
        raise TypeError("artifact must be an open mapped DAG view")
    return artifact.evaluate_boundary_quotient(
        raw_input,
        return_receipt=return_receipt,
    )
