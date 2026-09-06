"""Deterministic, fail-closed persistence for canonical implicit DAGs.

The artifact is an eagerly loaded archival representation.  It deliberately
does not claim a bounded-memory execution path.  A directory becomes a valid
artifact only when its canonical manifest is linked into place as the final
commit record.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from xvla.train.odt_engine_v2.graph import (
    _validate_network,
    validate_canonical_exponent_normal_form,
)
from xvla.train.odt_engine_v2.types import (
    CPBinaryCore,
    DenseCloneCore,
    DirectQProvenance,
    ImplicitNode,
    ImplicitProjectiveDAG,
    PhysicalSourceSpec,
    ReducedQRBinaryCore,
    UnaryCore,
)


ARTIFACT_SCHEMA = "xvla_implicit_projective_dag_artifact_v1"
MANIFEST_NAME = "manifest.json"
MAXIMUM_SHARD_BYTES = 1 << 30
_SHARD_ALIGNMENT_BYTES = 64
_INTEGER_MINIMUM = -(1 << 63)
_INTEGER_MAXIMUM = (1 << 63) - 1
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class ArtifactReceipt:
    root: Path
    manifest_sha256: str
    shard_sha256: tuple[tuple[str, str], ...]
    node_count: int
    tensor_count: int
    raw_byte_count: int


def _sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"manifest contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"manifest contains nonfinite JSON constant {value!r}")


def _parse_canonical_json(encoded: bytes) -> dict[str, Any]:
    try:
        text = encoded.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("manifest root must be an object")
    if _canonical_json_bytes(value) != encoded:
        raise ValueError("manifest bytes are not canonical JSON")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields differ from the schema")


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _require_integer(
    value: Any,
    label: str,
    *,
    minimum: int = _INTEGER_MINIMUM,
    maximum: int = _INTEGER_MAXIMUM,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an in-range integer")
    return value


def _require_optional_integer(
    value: Any,
    label: str,
    *,
    minimum: int = _INTEGER_MINIMUM,
    maximum: int = _INTEGER_MAXIMUM,
) -> int | None:
    if value is None:
        return None
    return _require_integer(value, label, minimum=minimum, maximum=maximum)


def _require_nonempty_string(value: Any, label: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{label} must be a nonempty string without NUL")
    return value


def _require_finite_float(value: Any, label: str, *, nonnegative: bool) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite JSON real")
    if nonnegative and value < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_physical_file(path: Path, label: str) -> os.stat_result:
    information = os.lstat(path)
    if not stat.S_ISREG(information.st_mode):
        raise ValueError(f"{label} must be a physical regular file")
    if information.st_nlink != 1:
        raise ValueError(f"{label} must not be hard-linked")
    if information.st_mode & 0o222:
        raise ValueError(f"{label} must be immutable")
    return information


def _physical_file_snapshot(
    path: Path,
    label: str,
) -> tuple[int, int, int, int, int, int, int]:
    information = _require_physical_file(path, label)
    return (
        information.st_dev,
        information.st_ino,
        information.st_mode,
        information.st_nlink,
        information.st_size,
        information.st_mtime_ns,
        information.st_ctime_ns,
    )


def _read_stable_physical_bytes(
    path: Path, label: str, *, expected_byte_count: int | None = None
) -> bytes:
    before = _require_physical_file(path, label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_mode & 0o222
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError(f"{label} changed while opening")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if expected_byte_count is not None and observed > expected_byte_count:
                raise ValueError(f"{label} is longer than declared")
        after = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"{label} changed while reading")
        if expected_byte_count is not None and observed != expected_byte_count:
            raise ValueError(f"{label} byte count differs from the manifest")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _directory_snapshot(path: Path, label: str) -> tuple[int, int, int, int, int]:
    information = os.lstat(path)
    if not stat.S_ISDIR(information.st_mode):
        raise ValueError(f"{label} must be a physical directory")
    if information.st_mode & 0o222:
        raise ValueError(f"{label} must be immutable")
    return (
        information.st_dev,
        information.st_ino,
        information.st_mode,
        information.st_mtime_ns,
        information.st_ctime_ns,
    )


def _postorder_nodes(root: ImplicitNode) -> tuple[ImplicitNode, ...]:
    if not isinstance(root, ImplicitNode):
        raise TypeError("artifact root must be an ImplicitNode")
    state: dict[int, int] = {}
    nodes: list[ImplicitNode] = []
    stack: list[tuple[ImplicitNode, int]] = [(root, 0)]
    state[id(root)] = 1
    while stack:
        node, child_index = stack[-1]
        if child_index < len(node.children):
            child = node.children[child_index]
            if not isinstance(child, ImplicitNode):
                raise TypeError("artifact graph child must be an ImplicitNode")
            stack[-1] = (node, child_index + 1)
            marker = state.get(id(child), 0)
            if marker == 1:
                raise ValueError("artifact graph contains a cycle")
            if marker == 0:
                state[id(child)] = 1
                stack.append((child, 0))
        else:
            stack.pop()
            state[id(node)] = 2
            nodes.append(node)
    return tuple(nodes)


def _require_canonical_network(network: ImplicitProjectiveDAG) -> None:
    _validate_network(network)
    validate_canonical_exponent_normal_form(network)
    _require_integer(network.token_count, "token count", minimum=1)
    _require_integer(network.feature_dimension, "feature dimension", minimum=1)
    _require_integer(network.selected_token, "selected token", minimum=0)
    _require_integer(network.head_binary_exponent, "head exponent")
    _require_bool(network.algorithm1_direct_rq_complete, "Algorithm 1 completion")
    _require_bool(
        network.algorithm1_scale_ledger_complete,
        "Algorithm 1 scale ledger completion",
    )
    _require_optional_integer(
        network.algorithm1_certified_head_binary_exponent,
        "certified head exponent",
    )
    _require_nonempty_string(network.claim_boundary, "claim boundary")
    for source in network.physical_sources:
        if not isinstance(source, PhysicalSourceSpec):
            raise ValueError("physical source must be a PhysicalSourceSpec")
        _require_nonempty_string(source.key, "physical source key")
        _require_nonempty_string(source.label, "physical source label")
        _require_integer(source.width, "physical source width", minimum=1)
    if network.head.device.type != "cpu" or network.head.dtype != torch.float64:
        raise ValueError("artifact tensors must be CPU float64")
    if sys.byteorder != "little":
        raise RuntimeError("artifact persistence requires a little-endian host")
    for node in _postorder_nodes(network.root):
        if isinstance(node.core, DenseCloneCore):
            raise ValueError("explicit clone-oracle cores are not persistable")


def _provenance_to_json(value: DirectQProvenance | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, DirectQProvenance):
        raise ValueError("direct-Q provenance has the wrong type")
    return {
        "method": _require_nonempty_string(
            value.method, "direct-Q provenance method"
        ),
        "verified": _require_bool(
            value.verified, "direct-Q provenance verified"
        ),
        "stage_count": _require_integer(
            value.stage_count, "direct-Q provenance stage count", minimum=1
        ),
        "column_blocks_compared": _require_integer(
            value.column_blocks_compared,
            "direct-Q provenance column blocks",
            minimum=1,
        ),
        "columns_compared": _require_integer(
            value.columns_compared,
            "direct-Q provenance columns",
            minimum=1,
        ),
        "compact_q_relative_error": _require_finite_float(
            value.compact_q_relative_error,
            "direct-Q provenance compact-Q error",
            nonnegative=True,
        ),
        "direct_q_reconstruction_relative_error": _require_finite_float(
            value.direct_q_reconstruction_relative_error,
            "direct-Q provenance reconstruction error",
            nonnegative=True,
        ),
        "retained_transition_elements": _require_integer(
            value.retained_transition_elements,
            "direct-Q provenance retained transition elements",
            minimum=0,
        ),
        "minimum_pivot_to_maximum_entry": _require_finite_float(
            value.minimum_pivot_to_maximum_entry,
            "direct-Q provenance minimum pivot ratio",
            nonnegative=True,
        ),
        "conditioning_threshold": _require_finite_float(
            value.conditioning_threshold,
            "direct-Q provenance conditioning threshold",
            nonnegative=True,
        ),
        "conditioning_accepted": _require_bool(
            value.conditioning_accepted,
            "direct-Q provenance conditioning accepted",
        ),
    }


def _provenance_from_json(value: Any, label: str) -> DirectQProvenance | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object or null")
    fields = {
        "method",
        "verified",
        "stage_count",
        "column_blocks_compared",
        "columns_compared",
        "compact_q_relative_error",
        "direct_q_reconstruction_relative_error",
        "retained_transition_elements",
        "minimum_pivot_to_maximum_entry",
        "conditioning_threshold",
        "conditioning_accepted",
    }
    _require_exact_keys(value, fields, label)
    return DirectQProvenance(
        method=_require_nonempty_string(value["method"], f"{label}.method"),
        verified=_require_bool(value["verified"], f"{label}.verified"),
        stage_count=_require_integer(
            value["stage_count"], f"{label}.stage_count", minimum=1
        ),
        column_blocks_compared=_require_integer(
            value["column_blocks_compared"],
            f"{label}.column_blocks_compared",
            minimum=1,
        ),
        columns_compared=_require_integer(
            value["columns_compared"], f"{label}.columns_compared", minimum=1
        ),
        compact_q_relative_error=_require_finite_float(
            value["compact_q_relative_error"],
            f"{label}.compact_q_relative_error",
            nonnegative=True,
        ),
        direct_q_reconstruction_relative_error=_require_finite_float(
            value["direct_q_reconstruction_relative_error"],
            f"{label}.direct_q_reconstruction_relative_error",
            nonnegative=True,
        ),
        retained_transition_elements=_require_integer(
            value["retained_transition_elements"],
            f"{label}.retained_transition_elements",
            minimum=0,
        ),
        minimum_pivot_to_maximum_entry=_require_finite_float(
            value["minimum_pivot_to_maximum_entry"],
            f"{label}.minimum_pivot_to_maximum_entry",
            nonnegative=True,
        ),
        conditioning_threshold=_require_finite_float(
            value["conditioning_threshold"],
            f"{label}.conditioning_threshold",
            nonnegative=True,
        ),
        conditioning_accepted=_require_bool(
            value["conditioning_accepted"],
            f"{label}.conditioning_accepted",
        ),
    )


class _ShardWriter:
    def __init__(self, root: Path, maximum_bytes: int) -> None:
        self.root = root
        self.maximum_bytes = maximum_bytes
        self.index = -1
        self.stream: Any = None
        self.digest: hashlib._Hash | None = None
        self.byte_count = 0
        self.shards: list[dict[str, Any]] = []

    def _name(self, index: int) -> str:
        return f"tensors-{index:05d}.f64le"

    def _open(self) -> None:
        self.index += 1
        name = self._name(self.index)
        self.stream = (self.root / name).open("xb")
        self.digest = hashlib.sha256()
        self.byte_count = 0

    def _close(self) -> None:
        if self.stream is None or self.digest is None:
            return
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        name = self._name(self.index)
        path = self.root / name
        path.chmod(0o444)
        self.shards.append(
            {
                "name": name,
                "byte_count": self.byte_count,
                "sha256": self.digest.hexdigest(),
            }
        )
        self.stream = None
        self.digest = None

    def write(self, value: memoryview) -> list[dict[str, int]]:
        position = 0
        segments: list[dict[str, int]] = []
        while position < len(value):
            if self.stream is None:
                self._open()
            remaining_capacity = self.maximum_bytes - self.byte_count
            count = min(len(value) - position, remaining_capacity)
            chunk = value[position : position + count]
            offset = self.byte_count
            self.stream.write(chunk)
            assert self.digest is not None
            self.digest.update(chunk)
            self.byte_count += count
            position += count
            segments.append(
                {
                    "shard_index": self.index,
                    "offset_bytes": offset,
                    "byte_count": count,
                }
            )
            if self.byte_count == self.maximum_bytes:
                self._close()
        return segments

    def finish(self) -> tuple[dict[str, Any], ...]:
        self._close()
        if not self.shards:
            raise RuntimeError("artifact contains no tensor bytes")
        return tuple(self.shards)

    def abort(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None
            self.digest = None


def _tensor_bytes(value: torch.Tensor, label: str) -> tuple[torch.Tensor, memoryview]:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or value.is_complex()
        or value.numel() < 1
    ):
        raise ValueError(f"{label} must be a nonempty CPU float64 tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must contain only finite values")
    contiguous = value.detach().contiguous()
    raw = memoryview(contiguous.numpy()).cast("B")
    if len(raw) != contiguous.numel() * 8:
        raise RuntimeError(f"{label} raw byte count is inconsistent")
    return contiguous, raw


def export_implicit_projective_dag_artifact(
    network: ImplicitProjectiveDAG,
    destination: str | os.PathLike[str],
    *,
    shard_size_bytes: int = MAXIMUM_SHARD_BYTES,
) -> ArtifactReceipt:
    """Publish one canonical DAG artifact without replacing any path."""

    _require_canonical_network(network)
    if (
        type(shard_size_bytes) is not int
        or shard_size_bytes < _SHARD_ALIGNMENT_BYTES
        or shard_size_bytes > MAXIMUM_SHARD_BYTES
        or shard_size_bytes % _SHARD_ALIGNMENT_BYTES != 0
    ):
        raise ValueError(
            "shard_size_bytes must be a 64-byte aligned value at most 1 GiB"
        )

    requested_root = Path(destination)
    requested_parent = requested_root.parent
    parent_before = os.lstat(requested_parent)
    if not stat.S_ISDIR(parent_before.st_mode):
        raise ValueError("artifact destination parent must be a physical directory")
    parent = requested_parent.resolve(strict=True)
    parent_after = os.lstat(parent)
    if (parent_before.st_dev, parent_before.st_ino) != (
        parent_after.st_dev,
        parent_after.st_ino,
    ):
        raise ValueError("artifact destination parent changed during resolution")
    if requested_root.name in {"", ".", ".."}:
        raise ValueError("artifact destination has an unsafe name")
    root = parent / requested_root.name
    if os.path.lexists(root):
        raise FileExistsError(f"refusing existing artifact destination {root}")
    os.mkdir(root, mode=0o700)
    created = True
    committed = False
    writer = _ShardWriter(root, shard_size_bytes)
    try:
        nodes = _postorder_nodes(network.root)
        node_index = {id(node): index for index, node in enumerate(nodes)}
        tensor_values: list[tuple[str, torch.Tensor]] = [
            ("network.head", network.head),
            ("network.mask", network.mask),
        ]
        columns: dict[str, list[Any]] = {
            "uid": [],
            "label": [],
            "origin_uid": [],
            "physical_token": [],
            "physical_source_key": [],
            "core_type": [],
            "core_kind": [],
            "core_binary_exponent": [],
            "direct_q_provenance": [],
            "matrix_tensor": [],
            "output_factor_tensor": [],
            "left_factor_tensor": [],
            "right_factor_tensor": [],
            "q_rows_tensor": [],
            "left_dimension": [],
            "right_dimension": [],
        }
        child_offsets = [0]
        child_indices: list[int] = []
        incoming_refcount = [0 for _ in nodes]

        def add_tensor(name: str, value: torch.Tensor) -> int:
            tensor_values.append((name, value))
            return len(tensor_values) - 1

        for index, node in enumerate(nodes):
            columns["uid"].append(
                _require_integer(node.uid, "node uid", minimum=0)
            )
            columns["label"].append(
                _require_nonempty_string(node.label, "node label")
            )
            columns["origin_uid"].append(
                _require_optional_integer(node.origin_uid, "origin uid", minimum=0)
            )
            columns["physical_token"].append(
                _require_optional_integer(
                    node.physical_token, "physical token", minimum=0
                )
            )
            if node.physical_source_key is not None:
                _require_nonempty_string(
                    node.physical_source_key, "physical source key"
                )
            columns["physical_source_key"].append(node.physical_source_key)
            columns["core_kind"].append(
                _require_nonempty_string(node.core.kind, "core kind")
            )
            columns["core_binary_exponent"].append(
                _require_integer(node.core.binary_exponent, "core exponent")
            )
            for child in node.children:
                child_index = node_index[id(child)]
                if child_index >= index:
                    raise ValueError("node order is not postorder")
                child_indices.append(child_index)
                incoming_refcount[child_index] += 1
            child_offsets.append(len(child_indices))

            tensor_columns = {
                "matrix_tensor": None,
                "output_factor_tensor": None,
                "left_factor_tensor": None,
                "right_factor_tensor": None,
                "q_rows_tensor": None,
            }
            if isinstance(node.core, UnaryCore):
                columns["core_type"].append("unary")
                columns["direct_q_provenance"].append(None)
                tensor_columns["matrix_tensor"] = add_tensor(
                    f"node[{index}].matrix", node.core.matrix
                )
                columns["left_dimension"].append(None)
                columns["right_dimension"].append(None)
            elif isinstance(node.core, CPBinaryCore):
                columns["core_type"].append("cp_binary")
                columns["direct_q_provenance"].append(
                    _provenance_to_json(node.core.direct_q_provenance)
                )
                tensor_columns["output_factor_tensor"] = add_tensor(
                    f"node[{index}].output_factor", node.core.output_factor
                )
                tensor_columns["left_factor_tensor"] = add_tensor(
                    f"node[{index}].left_factor", node.core.left_factor
                )
                tensor_columns["right_factor_tensor"] = add_tensor(
                    f"node[{index}].right_factor", node.core.right_factor
                )
                columns["left_dimension"].append(None)
                columns["right_dimension"].append(None)
            elif isinstance(node.core, ReducedQRBinaryCore):
                columns["core_type"].append("reduced_q_binary")
                columns["direct_q_provenance"].append(None)
                tensor_columns["q_rows_tensor"] = add_tensor(
                    f"node[{index}].q_rows", node.core.q_rows
                )
                columns["left_dimension"].append(
                    _require_integer(
                        node.core.left_dimension,
                        "reduced-Q left dimension",
                        minimum=1,
                    )
                )
                columns["right_dimension"].append(
                    _require_integer(
                        node.core.right_dimension,
                        "reduced-Q right dimension",
                        minimum=1,
                    )
                )
            else:
                raise ValueError("artifact encountered an unsupported core")
            for name, value in tensor_columns.items():
                columns[name].append(value)

        tensor_records: list[dict[str, Any]] = []
        raw_byte_count = 0
        for tensor_index, (name, value) in enumerate(tensor_values):
            contiguous, raw = _tensor_bytes(value, name)
            segments = writer.write(raw)
            byte_count = len(raw)
            raw_byte_count += byte_count
            tensor_records.append(
                {
                    "index": tensor_index,
                    "name": name,
                    "shape": list(contiguous.shape),
                    "byte_count": byte_count,
                    "segments": segments,
                }
            )
        shards = list(writer.finish())
        physical_sources = [
            {"key": spec.key, "label": spec.label, "width": spec.width}
            for spec in network.physical_sources
        ]
        manifest: dict[str, Any] = {
            "schema": ARTIFACT_SCHEMA,
            "encoding": {
                "dtype": "float64",
                "byte_order": "little",
                "layout": "c_contiguous",
                "maximum_shard_bytes": shard_size_bytes,
            },
            "network": {
                "head_tensor": 0,
                "mask_tensor": 1,
                "head_binary_exponent": network.head_binary_exponent,
                "token_count": network.token_count,
                "feature_dimension": network.feature_dimension,
                "selected_token": network.selected_token,
                "claim_boundary": network.claim_boundary,
                "algorithm1_direct_rq_complete": network.algorithm1_direct_rq_complete,
                "algorithm1_scale_ledger_complete": network.algorithm1_scale_ledger_complete,
                "algorithm1_certified_head_binary_exponent": (
                    network.algorithm1_certified_head_binary_exponent
                ),
            },
            "physical_sources": physical_sources,
            "graph": {
                "order": "depth_first_child_order_postorder",
                "node_count": len(nodes),
                "root_index": len(nodes) - 1,
                "child_offsets": child_offsets,
                "child_indices": child_indices,
                "incoming_refcount": incoming_refcount,
                **columns,
            },
            "tensors": tensor_records,
            "shards": shards,
        }
        manifest_bytes = _canonical_json_bytes(manifest)
        temporary_manifest = root / f".{MANIFEST_NAME}.{secrets.token_hex(12)}.tmp"
        with temporary_manifest.open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_manifest.chmod(0o444)
        manifest_path = root / MANIFEST_NAME
        os.link(temporary_manifest, manifest_path, follow_symlinks=False)
        temporary_manifest.unlink()
        expected_names = {MANIFEST_NAME, *(item["name"] for item in shards)}
        actual_names = {entry.name for entry in os.scandir(root)}
        if actual_names != expected_names:
            raise RuntimeError("artifact publication contains unexpected files")
        if _sha256_path(manifest_path) != _sha256_bytes(manifest_bytes):
            raise RuntimeError("published manifest bytes differ")
        directory_descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        root.chmod(0o555)
        parent_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        committed = True
        return ArtifactReceipt(
            root=root.resolve(strict=True),
            manifest_sha256=_sha256_bytes(manifest_bytes),
            shard_sha256=tuple(
                (item["name"], item["sha256"]) for item in shards
            ),
            node_count=len(nodes),
            tensor_count=len(tensor_records),
            raw_byte_count=raw_byte_count,
        )
    finally:
        writer.abort()
        if created and not committed and os.path.lexists(root):
            root.chmod(0o700)
            shutil.rmtree(root)


def _validate_shard_table(
    value: Any, maximum_bytes: int
) -> tuple[tuple[str, int, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("shards must be a nonempty list")
    records: list[tuple[str, int, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError("each shard record must be an object")
        _require_exact_keys(item, {"name", "byte_count", "sha256"}, "shard")
        expected_name = f"tensors-{index:05d}.f64le"
        if item["name"] != expected_name:
            raise ValueError("shard names or ordering differ")
        byte_count = _require_integer(
            item["byte_count"], "shard byte_count", minimum=1, maximum=maximum_bytes
        )
        records.append(
            (
                expected_name,
                byte_count,
                _require_sha256(item["sha256"], "shard sha256"),
            )
        )
    if any(byte_count != maximum_bytes for _, byte_count, _ in records[:-1]):
        raise ValueError("all nonfinal shards must have the configured byte count")
    return tuple(records)


def _validate_tensor_table(
    value: Any,
    shard_records: Sequence[tuple[str, int, str]],
    maximum_bytes: int,
    root: Path,
) -> tuple[tuple[str, tuple[int, ...], bytearray], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("tensors must be a nonempty list")
    shard_bytes: list[bytes] = []
    tensor_metadata: list[tuple[str, tuple[int, ...], int, list[Any]]] = []
    names: set[str] = set()
    total_declared = 0
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError("each tensor record must be an object")
        _require_exact_keys(
            item,
            {"index", "name", "shape", "byte_count", "segments"},
            "tensor",
        )
        if _require_integer(item["index"], "tensor index", minimum=0) != index:
            raise ValueError("tensor indices are not canonical")
        name = _require_nonempty_string(item["name"], "tensor name")
        if name in names:
            raise ValueError("tensor names must be unique")
        names.add(name)
        shape_value = item["shape"]
        if not isinstance(shape_value, list) or not shape_value:
            raise ValueError("tensor shape must be a nonempty list")
        shape = tuple(
            _require_integer(dimension, "tensor dimension", minimum=1)
            for dimension in shape_value
        )
        element_count = math.prod(shape)
        byte_count = _require_integer(
            item["byte_count"], "tensor byte_count", minimum=8
        )
        if byte_count != element_count * 8:
            raise ValueError("tensor byte count differs from its float64 shape")
        segments = item["segments"]
        if not isinstance(segments, list) or not segments:
            raise ValueError("tensor segments must be a nonempty list")
        parsed_segments: list[dict[str, int]] = []
        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError("tensor segment must be an object")
            _require_exact_keys(
                segment,
                {"shard_index", "offset_bytes", "byte_count"},
                "tensor segment",
            )
            parsed_segments.append(
                {
                    "shard_index": _require_integer(
                        segment["shard_index"], "segment shard index", minimum=0
                    ),
                    "offset_bytes": _require_integer(
                        segment["offset_bytes"], "segment offset", minimum=0
                    ),
                    "byte_count": _require_integer(
                        segment["byte_count"], "segment byte count", minimum=1
                    ),
                }
            )
        tensor_metadata.append((name, shape, byte_count, parsed_segments))
        total_declared += byte_count

    expected_shard_count = (total_declared + maximum_bytes - 1) // maximum_bytes
    if len(shard_records) != expected_shard_count:
        raise ValueError("shard count differs from canonical contiguous packing")
    expected_last_size = total_declared - maximum_bytes * (expected_shard_count - 1)
    if shard_records[-1][1] != expected_last_size:
        raise ValueError("final shard byte count differs from canonical packing")

    for name, byte_count, digest in shard_records:
        raw = _read_stable_physical_bytes(
            root / name,
            f"shard {name}",
            expected_byte_count=byte_count,
        )
        if _sha256_bytes(raw) != digest:
            raise ValueError(f"shard {name} digest differs")
        shard_bytes.append(raw)

    expected_shard_index = 0
    expected_offset = 0
    result: list[tuple[str, tuple[int, ...], bytearray]] = []
    for tensor_index, (name, shape, byte_count, segments) in enumerate(tensor_metadata):
        remaining = byte_count
        raw_tensor = bytearray()
        observed_segments: list[dict[str, int]] = []
        while remaining:
            if expected_offset == maximum_bytes:
                expected_shard_index += 1
                expected_offset = 0
            count = min(remaining, maximum_bytes - expected_offset)
            observed_segments.append(
                {
                    "shard_index": expected_shard_index,
                    "offset_bytes": expected_offset,
                    "byte_count": count,
                }
            )
            raw_tensor.extend(
                shard_bytes[expected_shard_index][
                    expected_offset : expected_offset + count
                ]
            )
            expected_offset += count
            remaining -= count
        if segments != observed_segments:
            raise ValueError(
                f"tensor {tensor_index} segments differ from canonical packing"
            )
        result.append((name, shape, raw_tensor))
    return tuple(result)


def _decode_tensors(
    tensor_records: Sequence[tuple[str, tuple[int, ...], bytearray]]
) -> tuple[tuple[str, torch.Tensor], ...]:
    result: list[tuple[str, torch.Tensor]] = []
    for name, shape, raw in tensor_records:
        tensor = torch.frombuffer(raw, dtype=torch.float64).clone().reshape(shape)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"tensor {name} contains a nonfinite value")
        result.append((name, tensor))
    return tuple(result)


def _parse_sources(value: Any) -> tuple[PhysicalSourceSpec, ...]:
    if not isinstance(value, list):
        raise ValueError("physical_sources must be a list")
    result: list[PhysicalSourceSpec] = []
    keys: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("physical source record must be an object")
        _require_exact_keys(item, {"key", "label", "width"}, "physical source")
        key = _require_nonempty_string(item["key"], "physical source key")
        if key in keys:
            raise ValueError("physical source keys must be unique")
        keys.add(key)
        result.append(
            PhysicalSourceSpec(
                key=key,
                label=_require_nonempty_string(
                    item["label"], "physical source label"
                ),
                width=_require_integer(
                    item["width"], "physical source width", minimum=1
                ),
            )
        )
    return tuple(result)


def _column(value: Mapping[str, Any], name: str, count: int) -> list[Any]:
    result = value[name]
    if not isinstance(result, list) or len(result) != count:
        raise ValueError(f"graph column {name} has the wrong length")
    return result


def _parse_graph(
    graph: Any,
    tensors: Sequence[tuple[str, torch.Tensor]],
    head_index: int,
    mask_index: int,
) -> tuple[ImplicitNode, int]:
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
    _require_exact_keys(
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
    node_count = _require_integer(graph["node_count"], "node_count", minimum=1)
    root_index = _require_integer(
        graph["root_index"], "root_index", minimum=0, maximum=node_count - 1
    )
    if root_index != node_count - 1:
        raise ValueError("canonical graph root must be the final postorder node")
    columns = {name: _column(graph, name, node_count) for name in column_names}
    child_offsets = graph["child_offsets"]
    child_indices = graph["child_indices"]
    incoming = graph["incoming_refcount"]
    if (
        not isinstance(child_offsets, list)
        or len(child_offsets) != node_count + 1
        or not isinstance(child_indices, list)
        or not isinstance(incoming, list)
        or len(incoming) != node_count
    ):
        raise ValueError("graph edge columns have inconsistent lengths")
    parsed_offsets = [
        _require_integer(value, "child offset", minimum=0)
        for value in child_offsets
    ]
    if (
        parsed_offsets[0] != 0
        or parsed_offsets[-1] != len(child_indices)
        or any(left > right for left, right in zip(parsed_offsets, parsed_offsets[1:]))
    ):
        raise ValueError("child offsets are not a canonical offset column")
    parsed_children = [
        _require_integer(value, "child index", minimum=0, maximum=node_count - 1)
        for value in child_indices
    ]
    observed_incoming = [0 for _ in range(node_count)]
    for parent_index in range(node_count):
        for child_index in parsed_children[
            parsed_offsets[parent_index] : parsed_offsets[parent_index + 1]
        ]:
            if child_index >= parent_index:
                raise ValueError("graph is not strict postorder")
            observed_incoming[child_index] += 1
    parsed_incoming = [
        _require_integer(value, "incoming refcount", minimum=0)
        for value in incoming
    ]
    if parsed_incoming != observed_incoming:
        raise ValueError("stored incoming refcounts differ from graph edges")
    if parsed_incoming[root_index] != 0 or any(
        count < 1 for index, count in enumerate(parsed_incoming) if index != root_index
    ):
        raise ValueError("graph contains an unreachable node")

    reachable: set[int] = set()
    stack = [root_index]
    while stack:
        index = stack.pop()
        if index in reachable:
            continue
        reachable.add(index)
        stack.extend(
            parsed_children[parsed_offsets[index] : parsed_offsets[index + 1]]
        )
    if len(reachable) != node_count:
        raise ValueError("not every graph node is reachable from the root")

    uids = [
        _require_integer(value, "node uid", minimum=0) for value in columns["uid"]
    ]
    if len(set(uids)) != node_count:
        raise ValueError("node uids must be unique")
    tensor_count = len(tensors)
    if not 0 <= head_index < tensor_count or not 0 <= mask_index < tensor_count:
        raise ValueError("network tensor reference is out of range")
    if head_index == mask_index:
        raise ValueError("head and mask tensor references must differ")
    used_tensor_indices = {head_index, mask_index}
    next_tensor_index = 2
    nodes: list[ImplicitNode] = []

    def tensor_reference(column_name: str, index: int) -> torch.Tensor | None:
        nonlocal next_tensor_index
        value = columns[column_name][index]
        if value is None:
            return None
        reference = _require_integer(
            value, f"{column_name} reference", minimum=0, maximum=tensor_count - 1
        )
        if reference in used_tensor_indices:
            raise ValueError("tensor references must be unique")
        if reference != next_tensor_index:
            raise ValueError("tensor references are not in canonical order")
        used_tensor_indices.add(reference)
        next_tensor_index += 1
        expected_suffix = column_name.removesuffix("_tensor")
        if tensors[reference][0] != f"node[{index}].{expected_suffix}":
            raise ValueError("tensor name differs from its graph reference")
        return tensors[reference][1]

    for index in range(node_count):
        core_type = columns["core_type"][index]
        kind = _require_nonempty_string(columns["core_kind"][index], "core kind")
        exponent = _require_integer(
            columns["core_binary_exponent"][index], "core exponent"
        )
        matrix = tensor_reference("matrix_tensor", index)
        output_factor = tensor_reference("output_factor_tensor", index)
        left_factor = tensor_reference("left_factor_tensor", index)
        right_factor = tensor_reference("right_factor_tensor", index)
        q_rows = tensor_reference("q_rows_tensor", index)
        provenance = _provenance_from_json(
            columns["direct_q_provenance"][index], f"node[{index}] provenance"
        )
        left_dimension = _require_optional_integer(
            columns["left_dimension"][index],
            "left dimension",
            minimum=1,
        )
        right_dimension = _require_optional_integer(
            columns["right_dimension"][index],
            "right dimension",
            minimum=1,
        )
        values = (matrix, output_factor, left_factor, right_factor, q_rows)
        if core_type == "unary":
            if matrix is None or any(value is not None for value in values[1:]):
                raise ValueError("unary tensor references differ")
            if provenance is not None or left_dimension is not None or right_dimension is not None:
                raise ValueError("unary metadata differs")
            core = UnaryCore(matrix, kind, exponent)
        elif core_type == "cp_binary":
            if matrix is not None or q_rows is not None or any(
                value is None for value in (output_factor, left_factor, right_factor)
            ):
                raise ValueError("CP tensor references differ")
            if left_dimension is not None or right_dimension is not None:
                raise ValueError("CP dimension metadata differs")
            core = CPBinaryCore(
                output_factor,
                left_factor,
                right_factor,
                kind,
                exponent,
                provenance,
            )
        elif core_type == "reduced_q_binary":
            if q_rows is None or any(value is not None for value in values[:-1]):
                raise ValueError("reduced-Q tensor references differ")
            if provenance is not None or left_dimension is None or right_dimension is None:
                raise ValueError("reduced-Q metadata differs")
            core = ReducedQRBinaryCore(
                q_rows, left_dimension, right_dimension, kind, exponent
            )
        else:
            raise ValueError("core type is unsupported")
        start, stop = parsed_offsets[index], parsed_offsets[index + 1]
        children = tuple(nodes[child] for child in parsed_children[start:stop])
        physical_token = _require_optional_integer(
            columns["physical_token"][index], "physical token", minimum=0
        )
        source_key_value = columns["physical_source_key"][index]
        if source_key_value is not None:
            source_key_value = _require_nonempty_string(
                source_key_value, "physical source key"
            )
        nodes.append(
            ImplicitNode(
                uid=uids[index],
                label=_require_nonempty_string(columns["label"][index], "node label"),
                core=core,
                children=children,
                physical_token=physical_token,
                origin_uid=_require_optional_integer(
                    columns["origin_uid"][index], "origin uid", minimum=0
                ),
                physical_source_key=source_key_value,
            )
        )
    canonical_nodes = _postorder_nodes(nodes[root_index])
    if len(canonical_nodes) != node_count or any(
        observed is not expected
        for observed, expected in zip(nodes, canonical_nodes)
    ):
        raise ValueError("graph rows are not in canonical depth-first postorder")
    if used_tensor_indices != set(range(tensor_count)):
        raise ValueError("tensor table contains an unreferenced tensor")
    return nodes[root_index], node_count


def load_implicit_projective_dag_artifact(
    source: str | os.PathLike[str],
    *,
    expected_manifest_sha256: str,
) -> ImplicitProjectiveDAG:
    """Authenticate and eagerly reconstruct one canonical DAG artifact.

    ``expected_manifest_sha256`` is the external root of trust returned by
    publication.  The manifest in turn authenticates every tensor shard.
    """

    if sys.byteorder != "little":
        raise RuntimeError("artifact loading requires a little-endian host")
    trusted_manifest_sha256 = _require_sha256(
        expected_manifest_sha256,
        "expected manifest sha256",
    )
    root = Path(source)
    root_snapshot = _directory_snapshot(root, "artifact root")
    manifest_path = root / MANIFEST_NAME
    manifest_snapshot = _physical_file_snapshot(
        manifest_path,
        "artifact manifest",
    )
    manifest_bytes = _read_stable_physical_bytes(manifest_path, "artifact manifest")
    if not secrets.compare_digest(
        _sha256_bytes(manifest_bytes),
        trusted_manifest_sha256,
    ):
        raise ValueError("artifact manifest digest differs from the trusted digest")
    manifest = _parse_canonical_json(manifest_bytes)
    _require_exact_keys(
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
    if manifest["schema"] != ARTIFACT_SCHEMA:
        raise ValueError("artifact schema differs")
    encoding = manifest["encoding"]
    if not isinstance(encoding, dict):
        raise ValueError("encoding must be an object")
    _require_exact_keys(
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
    maximum_bytes = _require_integer(
        encoding["maximum_shard_bytes"],
        "maximum shard bytes",
        minimum=_SHARD_ALIGNMENT_BYTES,
        maximum=MAXIMUM_SHARD_BYTES,
    )
    if maximum_bytes % _SHARD_ALIGNMENT_BYTES:
        raise ValueError("maximum shard bytes must be 64-byte aligned")
    shard_records = _validate_shard_table(manifest["shards"], maximum_bytes)
    expected_names = {MANIFEST_NAME, *(record[0] for record in shard_records)}
    observed_names: set[str] = set()
    member_snapshots: dict[str, tuple[int, int, int, int, int, int, int]] = {}
    for entry in os.scandir(root):
        if entry.name in observed_names:
            raise ValueError("artifact directory contains duplicate names")
        observed_names.add(entry.name)
        member_snapshots[entry.name] = _physical_file_snapshot(
            root / entry.name,
            f"artifact member {entry.name}",
        )
    if observed_names != expected_names:
        raise ValueError("artifact contains a missing or extra file")
    if member_snapshots[MANIFEST_NAME] != manifest_snapshot:
        raise ValueError("artifact manifest changed during loading")

    tensor_records = _validate_tensor_table(
        manifest["tensors"], shard_records, maximum_bytes, root
    )
    tensors = _decode_tensors(tensor_records)
    if len(tensors) < 2:
        raise ValueError("tensor table must contain head and mask")
    if tensors[0][0] != "network.head" or tensors[1][0] != "network.mask":
        raise ValueError("network tensor names differ")
    network_value = manifest["network"]
    if not isinstance(network_value, dict):
        raise ValueError("network must be an object")
    _require_exact_keys(
        network_value,
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
    head_index = _require_integer(
        network_value["head_tensor"], "head tensor", minimum=0
    )
    mask_index = _require_integer(
        network_value["mask_tensor"], "mask tensor", minimum=0
    )
    if head_index != 0 or mask_index != 1:
        raise ValueError("network head and mask references are not canonical")
    root_node, _ = _parse_graph(
        manifest["graph"], tensors, head_index, mask_index
    )
    network = ImplicitProjectiveDAG(
        root=root_node,
        head=tensors[head_index][1],
        head_binary_exponent=_require_integer(
            network_value["head_binary_exponent"], "head exponent"
        ),
        token_count=_require_integer(
            network_value["token_count"], "token count", minimum=1
        ),
        feature_dimension=_require_integer(
            network_value["feature_dimension"], "feature dimension", minimum=1
        ),
        selected_token=_require_integer(
            network_value["selected_token"], "selected token", minimum=0
        ),
        mask=tensors[mask_index][1],
        claim_boundary=_require_nonempty_string(
            network_value["claim_boundary"], "claim boundary"
        ),
        algorithm1_direct_rq_complete=_require_bool(
            network_value["algorithm1_direct_rq_complete"],
            "Algorithm 1 completion",
        ),
        physical_sources=_parse_sources(manifest["physical_sources"]),
        algorithm1_scale_ledger_complete=_require_bool(
            network_value["algorithm1_scale_ledger_complete"],
            "Algorithm 1 scale ledger completion",
        ),
        algorithm1_certified_head_binary_exponent=_require_optional_integer(
            network_value["algorithm1_certified_head_binary_exponent"],
            "certified head exponent",
        ),
    )
    _require_canonical_network(network)
    if _directory_snapshot(root, "artifact root") != root_snapshot:
        raise ValueError("artifact directory changed during loading")
    final_names = {entry.name for entry in os.scandir(root)}
    if final_names != expected_names:
        raise ValueError("artifact inventory changed during loading")
    for name, snapshot in member_snapshots.items():
        if _physical_file_snapshot(root / name, f"artifact member {name}") != snapshot:
            raise ValueError(f"artifact member {name} changed during loading")
    return network
