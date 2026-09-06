from __future__ import annotations

import builtins
import gc
import hashlib
import json
import os
import random
import shutil
import struct
import threading
from pathlib import Path
from typing import Callable

import pytest
import torch

from xvla.train import implicit_projective_dag_artifact as eager_artifact
from xvla.train.direct_odt_dimension_ladder import (
    REMOVALS,
    InternalDimensionPlans,
    _relative_pair,
    export_dimension_ladder,
    masked_boundary,
)
from xvla.train.implicit_projective_dag_artifact import (
    MANIFEST_NAME,
    export_implicit_projective_dag_artifact,
    load_implicit_projective_dag_artifact,
)
from xvla.train.implicit_projective_dag_mapped import (
    AlignedContiguousTensorSpan,
    SegmentedTensorSpan,
    UnsupportedMappedTensorLayout,
    open_mapped_implicit_projective_dag_artifact,
)
from xvla.train.implicit_projective_prefix_artifact import (
    ARTIFACT_SCHEMA as PREFIX_ARTIFACT_SCHEMA,
    export_implicit_projective_prefix_artifact,
)
from xvla.train.implicit_sparse_projective_odt import (
    _Builder,
    CPBinaryCore,
    DirectQProvenance,
    ImplicitNode,
    ImplicitProjectiveDAG,
    MaterializationTelemetry,
    PhysicalSourceSpec,
    ReducedQRBinaryCore,
    UnaryCore,
    canonicalize_implicit_dag_direct_rq,
    diagonalize_implicit_dag_full_rank,
    evaluate_boundary_quotient,
    evaluate_projective_boundary,
)


DTYPE = torch.float64


def _matrix(offset: int = 0) -> torch.Tensor:
    values = torch.arange(offset, offset + 64, dtype=DTYPE).reshape(8, 8)
    return torch.eye(8, dtype=DTYPE) + values / 256.0


def _q_rows(offset: int = 0) -> torch.Tensor:
    values = torch.arange(offset, offset + 8 * 64, dtype=DTYPE).reshape(8, 64)
    return values / 2048.0 + torch.eye(8, 64, dtype=DTYPE)


def _provenance() -> DirectQProvenance:
    return DirectQProvenance(
        method="mapped_test_direct_q_replay",
        verified=True,
        stage_count=1,
        column_blocks_compared=1,
        columns_compared=64,
        compact_q_relative_error=0.0,
        direct_q_reconstruction_relative_error=0.0,
        retained_transition_elements=0,
        minimum_pivot_to_maximum_entry=1.0,
        conditioning_threshold=0.0,
        conditioning_accepted=True,
    )


def _leaf(uid: int, token: int, offset: int) -> ImplicitNode:
    return ImplicitNode(
        uid=uid,
        label=f"leaf.{uid}",
        core=UnaryCore(_matrix(offset), "physical"),
        physical_token=token,
    )


def _unary(uid: int, child: ImplicitNode, offset: int) -> ImplicitNode:
    return ImplicitNode(
        uid=uid,
        label=f"unary.{uid}",
        core=UnaryCore(_matrix(offset), "mapped_unary"),
        children=(child,),
    )


def _reduced(
    uid: int,
    left: ImplicitNode,
    right: ImplicitNode,
    offset: int,
) -> ImplicitNode:
    return ImplicitNode(
        uid=uid,
        label=f"reduced.{uid}",
        core=ReducedQRBinaryCore(
            _q_rows(offset),
            left_dimension=8,
            right_dimension=8,
            kind="bounded_direct_q",
        ),
        children=(left, right),
    )


def _network(pattern: str = "same_child") -> ImplicitProjectiveDAG:
    if pattern == "same_child":
        leaf_a = _leaf(10, 0, 0)
        leaf_b = _leaf(11, 1, 64)
        cp = ImplicitNode(
            uid=20,
            label="cp.20",
            core=CPBinaryCore(
                output_factor=_matrix(128),
                left_factor=_matrix(192),
                right_factor=_matrix(256),
                kind="canonical_cp",
                direct_q_provenance=_provenance(),
            ),
            children=(leaf_a, leaf_b),
        )
        shared = _unary(21, cp, 320)
        root = _reduced(30, shared, shared, 384)
    elif pattern == "two_parents":
        shared = _leaf(40, 0, 0)
        left = _unary(41, shared, 64)
        right = _unary(42, shared, 128)
        root = _reduced(43, left, right, 192)
    elif pattern == "deep_fanout":
        shared = _leaf(50, 0, 0)
        left = _unary(51, shared, 64)
        right = _unary(52, shared, 128)
        left_repeat = _reduced(53, left, left, 192)
        right_repeat = _reduced(54, right, right, 704)
        root = _reduced(55, left_repeat, right_repeat, 1216)
    else:
        raise ValueError("unknown fixture pattern")
    return ImplicitProjectiveDAG(
        root=root,
        head=_matrix(2048),
        head_binary_exponent=0,
        token_count=8,
        feature_dimension=7,
        selected_token=0,
        mask=torch.eye(8, dtype=DTYPE),
        claim_boundary="Mapped executor aligned unit fixture.",
        algorithm1_direct_rq_complete=True,
        algorithm1_scale_ledger_complete=True,
        algorithm1_certified_head_binary_exponent=0,
    )


def _unaligned_network() -> ImplicitProjectiveDAG:
    leaf = ImplicitNode(
        uid=1,
        label="unaligned.leaf",
        core=UnaryCore(torch.eye(2, dtype=DTYPE), "physical"),
        physical_token=0,
    )
    root = ImplicitNode(
        uid=2,
        label="unaligned.root",
        core=UnaryCore(torch.eye(2, dtype=DTYPE), "unary"),
        children=(leaf,),
    )
    return ImplicitProjectiveDAG(
        root=root,
        head=torch.eye(2, dtype=DTYPE),
        head_binary_exponent=0,
        token_count=2,
        feature_dimension=1,
        selected_token=0,
        mask=torch.eye(2, dtype=DTYPE),
        claim_boundary="Mapped executor unaligned unit fixture.",
        algorithm1_direct_rq_complete=True,
        algorithm1_scale_ledger_complete=True,
        algorithm1_certified_head_binary_exponent=0,
    )


def _heterogeneous_network() -> ImplicitProjectiveDAG:
    first = ImplicitNode(
        uid=70,
        label="heterogeneous.first",
        core=UnaryCore(
            torch.arange(24, dtype=DTYPE).reshape(8, 3) / 64.0 + 0.25,
            "physical_first",
        ),
        physical_source_key="first",
    )
    second = ImplicitNode(
        uid=71,
        label="heterogeneous.second",
        core=UnaryCore(
            torch.arange(32, dtype=DTYPE).reshape(8, 4) / 96.0 + 0.125,
            "physical_second",
        ),
        physical_source_key="second",
    )
    root = _reduced(72, first, second, 256)
    return ImplicitProjectiveDAG(
        root=root,
        head=_matrix(1024),
        head_binary_exponent=0,
        token_count=8,
        feature_dimension=7,
        selected_token=0,
        mask=torch.eye(8, dtype=DTYPE),
        claim_boundary="Mapped executor heterogeneous unit fixture.",
        algorithm1_direct_rq_complete=True,
        physical_sources=(
            PhysicalSourceSpec("first", "source.first", 2),
            PhysicalSourceSpec("second", "source.second", 3),
        ),
        algorithm1_scale_ledger_complete=True,
        algorithm1_certified_head_binary_exponent=0,
    )


def _random_shared_network(seed: int) -> ImplicitProjectiveDAG:
    generator = random.Random(seed)
    current = _leaf(1000 + seed * 100, 0, seed * 64)
    pool = [current]
    for step in range(1, 13):
        uid = 1000 + seed * 100 + step
        if step % 4 == 0:
            current = _unary(uid, current, (seed * 17 + step) * 64)
        else:
            other = pool[generator.randrange(len(pool))]
            if generator.randrange(2):
                current = _reduced(
                    uid,
                    current,
                    other,
                    (seed * 31 + step) * 512,
                )
            else:
                current = _reduced(
                    uid,
                    other,
                    current,
                    (seed * 31 + step) * 512,
                )
        pool.append(current)
    return ImplicitProjectiveDAG(
        root=current,
        head=_matrix(4096 + seed * 64),
        head_binary_exponent=0,
        token_count=8,
        feature_dimension=7,
        selected_token=0,
        mask=torch.eye(8, dtype=DTYPE),
        claim_boundary=f"Mapped randomized shared fixture {seed}.",
        algorithm1_direct_rq_complete=True,
        algorithm1_scale_ledger_complete=True,
        algorithm1_certified_head_binary_exponent=0,
    )


def _raw_inputs() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [0.25, -0.5, 0.75, -0.125, 0.375, 0.5, -0.25],
                [-0.75, 0.125, 0.25, 0.5, -0.375, 0.625, 0.875],
                [0.5, 0.25, -0.125, 0.75, -0.625, 0.375, 0.125],
                [0.125, -0.25, 0.5, 0.375, 0.625, -0.75, 0.25],
                [0.75, 0.5, 0.25, -0.5, -0.25, 0.125, 0.375],
                [-0.25, 0.625, 0.375, 0.125, 0.5, -0.375, 0.75],
                [0.375, -0.125, 0.625, -0.75, 0.25, 0.5, 0.125],
                [0.625, 0.375, -0.25, 0.5, 0.125, -0.5, 0.75],
            ],
            [
                [-0.125, 0.75, 0.5, 0.25, -0.5, 0.375, 0.625],
                [0.375, -0.625, 0.125, 0.75, 0.5, -0.25, 0.25],
                [0.625, 0.125, -0.5, 0.375, 0.25, 0.75, -0.125],
                [-0.5, 0.25, 0.75, 0.125, 0.375, 0.625, -0.25],
                [0.25, 0.5, -0.375, 0.625, -0.125, 0.75, 0.375],
                [0.75, -0.25, 0.375, 0.5, 0.625, 0.125, -0.5],
                [0.5, 0.375, 0.125, -0.25, 0.75, -0.625, 0.25],
                [-0.375, 0.625, 0.25, 0.5, -0.125, 0.375, 0.75],
            ],
        ],
        dtype=DTYPE,
    )


def _mapped_ladder_fixture() -> tuple[object, dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(917)
    width = 20
    builder = _Builder(torch.empty((), dtype=DTYPE), MaterializationTelemetry())
    source = builder.physical_pair(
        PhysicalSourceSpec("x", "physical_state", width)
    )
    shared = builder.unary(
        "shared.learned",
        torch.randn(width, width + 1, generator=generator, dtype=DTYPE),
        source,
        kind="learned_projection",
    )
    binary = builder.cp(
        "shared.square",
        torch.eye(width, dtype=DTYPE),
        torch.randn(width, width, generator=generator, dtype=DTYPE),
        torch.randn(width, width, generator=generator, dtype=DTYPE),
        (shared, shared),
        kind="repeated_bond",
    )
    root = builder.unary(
        "root.fixed",
        torch.eye(width, dtype=DTYPE),
        binary,
        kind="observable_boundary",
    )
    network = ImplicitProjectiveDAG(
        root=root,
        head=torch.randn(3, width, generator=generator, dtype=DTYPE),
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=width,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=DTYPE),
        physical_sources=builder.physical_source_specs,
    )
    raw = {"x": torch.randn(3, width, generator=generator, dtype=DTYPE)}
    canonical = canonicalize_implicit_dag_direct_rq(
        network,
        replay_inputs=raw,
        block_size=32,
        copy_network=False,
    )
    diagonal = diagonalize_implicit_dag_full_rank(
        canonical.network,
        replay_inputs=raw,
        copy_network=False,
        retain_eigenvalues=False,
        stream_pre_evd_environments=True,
        retain_post_evd_environments=False,
    )
    return diagonal, raw


def _canonical_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _manifest_digest(root: Path) -> str:
    return hashlib.sha256((root / MANIFEST_NAME).read_bytes()).hexdigest()


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
    }


def _mutable_copy(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    destination.chmod(0o755)
    for path in destination.iterdir():
        path.chmod(0o644)
    return destination


def _freeze(root: Path) -> None:
    for path in root.iterdir():
        path.chmod(0o444)
    root.chmod(0o555)


def _expected_edge_digest(root: Path) -> str:
    graph = json.loads((root / MANIFEST_NAME).read_text())["graph"]
    remaining = list(graph["incoming_refcount"])
    digest = hashlib.sha256()
    ordinal = 0
    for parent in range(graph["node_count"]):
        start = graph["child_offsets"][parent]
        stop = graph["child_offsets"][parent + 1]
        for role, child in enumerate(graph["child_indices"][start:stop]):
            before = remaining[child]
            after = before - 1
            digest.update(
                struct.pack("<QQQQQQ", ordinal, parent, role, child, before, after)
            )
            remaining[child] = after
            ordinal += 1
    assert not any(remaining)
    return digest.hexdigest()


def test_mapped_execution_is_bit_equal_deterministic_and_preserves_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    network = _network()
    receipt = export_implicit_projective_dag_artifact(network, root)
    before = _artifact_hashes(root)
    eager = load_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    )
    expected = evaluate_projective_boundary(eager, _raw_inputs())
    expected_quotient = evaluate_boundary_quotient(eager, _raw_inputs())

    with open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as mapped:
        assert mapped.manifest_sha256 == receipt.manifest_sha256
        assert mapped.node_count == 5
        assert mapped.edge_occurrence_count == 5
        assert all(tensor.directly_executable for tensor in mapped.tensors)
        assert mapped.node(mapped.node_count - 1).children == (3, 3)
        with pytest.raises(TypeError):
            mapped._mappings[0][0] = 1

        observed, first = mapped.evaluate_projective_boundary(
            _raw_inputs(), return_receipt=True
        )
        repeated, second = mapped.evaluate_projective_boundary(
            _raw_inputs(), return_receipt=True
        )
        assert torch.equal(observed, expected)
        assert torch.equal(repeated, expected)
        mapped_quotient, quotient_receipt = mapped.evaluate_boundary_quotient(
            _raw_inputs(), return_receipt=True
        )
        assert torch.equal(mapped_quotient, expected_quotient)
        assert quotient_receipt == first
        assert first == second
        assert first.node_evaluations == mapped.node_count
        assert first.edge_occurrences_emitted == mapped.edge_occurrence_count
        assert first.released_nonroot_nodes == mapped.node_count - 1
        assert first.root_only_live
        assert first.all_refcounts_zero
        assert first.edge_ledger_sha256 == _expected_edge_digest(root)
    assert mapped.closed
    with pytest.raises(RuntimeError, match="closed"):
        mapped.evaluate_projective_boundary(_raw_inputs())
    with pytest.raises(RuntimeError, match="closed"):
        mapped.node(0)
    assert _artifact_hashes(root) == before


@pytest.mark.parametrize(
    "pattern,expected_nodes,expected_edges",
    (
        ("two_parents", 4, 4),
        ("deep_fanout", 6, 8),
    ),
)
def test_occurrence_executor_handles_two_parents_and_deep_repeated_fanout(
    tmp_path: Path,
    pattern: str,
    expected_nodes: int,
    expected_edges: int,
) -> None:
    root = tmp_path / pattern
    network = _network(pattern)
    receipt = export_implicit_projective_dag_artifact(network, root)
    expected = evaluate_projective_boundary(network, _raw_inputs())
    with open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as mapped:
        observed, execution = mapped.evaluate_projective_boundary(
            _raw_inputs(), return_receipt=True
        )
        assert torch.equal(observed, expected)
        assert execution.node_evaluations == expected_nodes
        assert execution.edge_occurrences_emitted == expected_edges
        assert execution.released_nonroot_nodes == expected_nodes - 1
        assert execution.edge_ledger_sha256 == _expected_edge_digest(root)


@pytest.mark.parametrize("seed", range(8))
def test_randomized_shared_postorder_graphs_match_eager_bit_for_bit(
    tmp_path: Path,
    seed: int,
) -> None:
    root = tmp_path / f"random-{seed}"
    network = _random_shared_network(seed)
    receipt = export_implicit_projective_dag_artifact(network, root)
    expected = evaluate_projective_boundary(network, _raw_inputs())
    with open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as mapped:
        observed, execution = mapped.evaluate_projective_boundary(
            _raw_inputs(), return_receipt=True
        )
        assert torch.equal(observed, expected)
        assert execution.node_evaluations == mapped.node_count
        assert execution.edge_occurrences_emitted == mapped.edge_occurrence_count
        assert execution.released_nonroot_nodes == mapped.node_count - 1
        assert execution.edge_ledger_sha256 == _expected_edge_digest(root)


def test_heterogeneous_mapping_and_tuple_inputs_match_eager_bit_for_bit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "heterogeneous"
    network = _heterogeneous_network()
    receipt = export_implicit_projective_dag_artifact(network, root)
    first = torch.tensor(
        [[0.25, -0.5], [0.75, 0.125]], dtype=DTYPE
    )
    second = torch.tensor(
        [[-0.25, 0.5, 0.75], [0.625, -0.125, 0.375]], dtype=DTYPE
    )
    inputs = {"first": first, "second": second}
    expected = evaluate_projective_boundary(network, inputs)
    with open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as mapped:
        from_mapping = mapped.evaluate_projective_boundary(inputs)
        from_tuple = mapped.evaluate_projective_boundary((first, second))
        assert torch.equal(from_mapping, expected)
        assert torch.equal(from_tuple, expected)


def test_each_postorder_row_is_evaluated_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    receipt = export_implicit_projective_dag_artifact(_network("deep_fanout"), root)
    with open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as mapped:
        calls = [0] * mapped.node_count
        original = mapped._apply_node

        def counted(index: int, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
            calls[index] += 1
            return original(index, inputs)

        monkeypatch.setattr(mapped, "_apply_node", counted)
        mapped.evaluate_projective_boundary(_raw_inputs())
        assert calls == [1] * mapped.node_count


def test_dtype_aligned_and_bounded_segmented_tensors_execute_with_exact_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segmented_root = tmp_path / "segmented"
    segmented_receipt = export_implicit_projective_dag_artifact(
        _network(), segmented_root, shard_size_bytes=64
    )
    with open_mapped_implicit_projective_dag_artifact(
        segmented_root,
        expected_manifest_sha256=segmented_receipt.manifest_sha256,
    ) as mapped:
        executed_tensor_indices = {mapped.network.head_tensor}
        for node_index in range(mapped.node_count):
            executed_tensor_indices.update(mapped.node(node_index).tensor_indices)
        segmented = [
            descriptor
            for descriptor in mapped.tensors
            if descriptor.index in executed_tensor_indices
            if isinstance(descriptor.layout, SegmentedTensorSpan)
            and descriptor.layout.reason == "cross_shard"
        ]
        assert segmented
        assert mapped.execution_layout_supported
        assert mapped.segmented_bytes_per_evaluation == sum(
            descriptor.byte_count for descriptor in segmented
        )
        expected = evaluate_projective_boundary(_network(), _raw_inputs())
        observed, execution = mapped.evaluate_projective_boundary(
            _raw_inputs(), return_receipt=True
        )
        assert torch.equal(observed, expected)
        assert execution.segmented_tensor_gathers == len(segmented)
        assert execution.segmented_bytes_gathered == sum(
            descriptor.byte_count for descriptor in segmented
        )

    with open_mapped_implicit_projective_dag_artifact(
        segmented_root,
        expected_manifest_sha256=segmented_receipt.manifest_sha256,
        maximum_segmented_gather_bytes=64,
    ) as mapped:
        assert not mapped.execution_layout_supported
        calls = 0
        original = mapped._apply_node

        def counted(index: int, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
            nonlocal calls
            calls += 1
            return original(index, inputs)

        monkeypatch.setattr(mapped, "_apply_node", counted)
        with pytest.raises(UnsupportedMappedTensorLayout, match="above.*cap"):
            mapped.evaluate_projective_boundary(_raw_inputs())
        assert calls == 0

    unaligned_root = tmp_path / "unaligned"
    unaligned_receipt = export_implicit_projective_dag_artifact(
        _unaligned_network(), unaligned_root
    )
    with open_mapped_implicit_projective_dag_artifact(
        unaligned_root,
        expected_manifest_sha256=unaligned_receipt.manifest_sha256,
    ) as mapped:
        assert all(
            isinstance(descriptor.layout, AlignedContiguousTensorSpan)
            for descriptor in mapped.tensors
        )
        raw_input = torch.ones(1, 2, 1, dtype=DTYPE)
        expected = evaluate_projective_boundary(_unaligned_network(), raw_input)
        assert torch.equal(mapped.evaluate_projective_boundary(raw_input), expected)


@pytest.mark.parametrize("delta", (-1, 1))
def test_runtime_refcount_corruption_fails_closed(
    tmp_path: Path,
    delta: int,
) -> None:
    root = tmp_path / "artifact"
    receipt = export_implicit_projective_dag_artifact(_network(), root)
    with open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as mapped:
        root_children = mapped.node(mapped.node_count - 1).children
        assert root_children[0] == root_children[1]
        shared_index = root_children[0]
        incoming = list(mapped._nodes.incoming_refcount)
        incoming[shared_index] += delta
        mapped._nodes.incoming_refcount = tuple(incoming)
        match = "underflowed" if delta < 0 else "leftovers"
        with pytest.raises(RuntimeError, match=match):
            mapped.evaluate_projective_boundary(_raw_inputs())


def test_premature_release_is_detected_before_a_missing_child_is_reused(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    receipt = export_implicit_projective_dag_artifact(_network("two_parents"), root)
    with open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as mapped:
        incoming = list(mapped._nodes.incoming_refcount)
        shared = mapped.node(1).children[0]
        assert mapped.node(2).children[0] == shared
        incoming[shared] = 1
        mapped._nodes.incoming_refcount = tuple(incoming)
        with pytest.raises(RuntimeError, match="prematurely released"):
            mapped.evaluate_projective_boundary(_raw_inputs())


def test_open_and_execute_do_not_call_eager_loader_or_tensor_clone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    receipt = export_implicit_projective_dag_artifact(_network(), root)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("forbidden eager reconstruction or payload clone")

    monkeypatch.setattr(eager_artifact, "load_implicit_projective_dag_artifact", forbidden)
    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    with open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as mapped:
        observed = mapped.evaluate_projective_boundary(_raw_inputs())
        assert observed.shape == (2, 8)


def test_external_manifest_digest_is_mandatory_and_authoritative(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    receipt = export_implicit_projective_dag_artifact(_network(), root)
    with pytest.raises(TypeError):
        open_mapped_implicit_projective_dag_artifact(root)
    with pytest.raises(ValueError, match="trusted digest"):
        open_mapped_implicit_projective_dag_artifact(
            root,
            expected_manifest_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        open_mapped_implicit_projective_dag_artifact(
            root,
            expected_manifest_sha256=True,
        )
    with open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as mapped:
        assert not mapped.closed


def test_honest_physical_prefix_schema_executes_without_canonical_relabeling(
    tmp_path: Path,
) -> None:
    root = tmp_path / "prefix"
    network = _network()
    cp = network.root.children[0].children[0]
    assert isinstance(cp.core, CPBinaryCore)
    cp.core = CPBinaryCore(
        cp.core.output_factor,
        cp.core.left_factor,
        cp.core.right_factor,
        cp.core.kind,
        cp.core.binary_exponent,
        None,
    )
    network.algorithm1_direct_rq_complete = False
    network.algorithm1_scale_ledger_complete = False
    network.algorithm1_certified_head_binary_exponent = None
    receipt = export_implicit_projective_prefix_artifact(network, root)
    expected = evaluate_projective_boundary(network, _raw_inputs())

    with open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as mapped:
        assert mapped.network.artifact_schema == PREFIX_ARTIFACT_SCHEMA
        assert mapped.network.algorithm1_direct_rq_complete is False
        assert mapped.network.algorithm1_scale_ledger_complete is False
        assert mapped.network.algorithm1_certified_head_binary_exponent is None
        assert torch.equal(mapped.evaluate_projective_boundary(_raw_inputs()), expected)


def test_compiled_executor_matches_mapped_for_all_core_types_and_repeated_edges(
    tmp_path: Path,
) -> None:
    from xvla.train.implicit_projective_dag_numba import (
        open_compiled_mapped_implicit_projective_dag_artifact,
    )

    root = tmp_path / "compiled"
    receipt = export_implicit_projective_dag_artifact(
        _network(),
        root,
        shard_size_bytes=64,
    )
    with open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as mapped:
        expected, expected_receipt = mapped.evaluate_projective_boundary(
            _raw_inputs(),
            return_receipt=True,
        )
    with open_compiled_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as compiled:
        observed, observed_receipt = compiled.evaluate_projective_boundary(
            _raw_inputs(),
            return_receipt=True,
        )
        assert _relative_pair(observed, expected) <= 2e-10
        assert observed_receipt.node_evaluations == expected_receipt.node_evaluations
        assert (
            observed_receipt.edge_occurrences_emitted
            == expected_receipt.edge_occurrences_emitted
        )
        assert observed_receipt.edge_ledger_sha256 == expected_receipt.edge_ledger_sha256
        assert observed_receipt.released_nonroot_nodes == expected_receipt.released_nonroot_nodes
        assert observed_receipt.peak_live_values == expected_receipt.peak_live_values
        assert compiled.execution_layout_supported
        assert compiled.segmented_bytes_per_evaluation == 0
        assert observed_receipt.arena_width == compiled.arena_receipt.arena_width
        assert observed_receipt.arena_bytes == (
            len(_raw_inputs()) * compiled.arena_receipt.arena_width * 8
        )
        assert compiled.arena_receipt.arena_width < (
            compiled.arena_receipt.logical_output_width_sum
        )


def test_compiled_executor_matches_mapped_on_385_by_386_retained_q(
    tmp_path: Path,
) -> None:
    from xvla.train.implicit_projective_dag_numba import (
        open_compiled_mapped_implicit_projective_dag_artifact,
    )

    generator = torch.Generator().manual_seed(385386)
    left = ImplicitNode(
        uid=1,
        label="pade.left",
        core=UnaryCore(torch.eye(2, dtype=DTYPE), "physical"),
        physical_source_key="left",
    )
    right = ImplicitNode(
        uid=2,
        label="pade.right",
        core=UnaryCore(torch.eye(193, dtype=DTYPE), "physical"),
        physical_source_key="right",
    )
    root_node = ImplicitNode(
        uid=3,
        label="pade.retained_q",
        core=ReducedQRBinaryCore(
            torch.randn(385, 386, generator=generator, dtype=DTYPE) / 64.0,
            left_dimension=2,
            right_dimension=193,
            kind="bounded_direct_q",
        ),
        children=(left, right),
    )
    network = ImplicitProjectiveDAG(
        root=root_node,
        head=torch.randn(57, 385, generator=generator, dtype=DTYPE) / 32.0,
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=1,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=DTYPE),
        physical_sources=(
            PhysicalSourceSpec("left", "pade.left", 1),
            PhysicalSourceSpec("right", "pade.right", 192),
        ),
        algorithm1_direct_rq_complete=True,
        algorithm1_scale_ledger_complete=True,
        algorithm1_certified_head_binary_exponent=0,
    )
    raw = {
        "left": torch.randn(20, 1, generator=generator, dtype=DTYPE),
        "right": torch.randn(20, 192, generator=generator, dtype=DTYPE),
    }
    root = tmp_path / "retained-q-385x386"
    receipt = export_implicit_projective_dag_artifact(network, root)
    with open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as mapped:
        expected = mapped.evaluate_projective_boundary(raw)
    with open_compiled_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as compiled:
        observed = compiled.evaluate_projective_boundary(raw)
    assert _relative_pair(observed, expected) <= 2e-10


def test_compiled_executor_arena_cap_and_close_retry_fail_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xvla.train import implicit_projective_dag_numba as compiled_module

    root = tmp_path / "compiled-lifetime"
    receipt = export_implicit_projective_dag_artifact(_network(), root)
    compiled = compiled_module.open_compiled_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
        maximum_arena_bytes=8,
    )
    calls = 0

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("compiled kernel must not run above its arena cap")

    monkeypatch.setattr(compiled_module, "_execute_compiled", forbidden)
    with pytest.raises(RuntimeError, match="arena exceeds.*before execution"):
        compiled.evaluate_projective_boundary(_raw_inputs())
    assert calls == 0

    mappings = compiled._mapped._mappings
    exported = compiled._weight_arrays[0]
    with pytest.raises(RuntimeError, match="still exported.*retry close"):
        compiled.close()
    assert not compiled.closed
    with pytest.raises(RuntimeError, match="close is incomplete"):
        compiled.evaluate_projective_boundary(_raw_inputs())
    del exported
    gc.collect()
    compiled.close()
    assert compiled.closed
    assert all(mapping.closed for mapping in mappings)


def test_compiled_executor_close_cannot_invalidate_an_active_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    from xvla.train import implicit_projective_dag_numba as compiled_module

    root = tmp_path / "compiled-active"
    receipt = export_implicit_projective_dag_artifact(_network(), root)
    compiled = compiled_module.open_compiled_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    )
    started = threading.Event()
    proceed = threading.Event()

    def blocked(*_args: object, **_kwargs: object) -> tuple[object, object]:
        started.set()
        assert proceed.wait(timeout=10.0)
        return (
            np.ones((2, 8), dtype=np.float64),
            np.zeros(2, dtype=np.uint8),
        )

    monkeypatch.setattr(compiled_module, "_execute_compiled", blocked)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            compiled.evaluate_projective_boundary(_raw_inputs())
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    assert started.wait(timeout=10.0)
    with pytest.raises(RuntimeError, match="during evaluation"):
        compiled.close()
    assert not compiled.closed
    proceed.set()
    worker.join(timeout=10.0)
    assert not worker.is_alive()
    assert not errors
    compiled.close()
    assert compiled.closed


def test_all_six_physical_prefixes_match_their_original_mask_outputs_mapped(
    tmp_path: Path,
) -> None:
    from xvla.train.implicit_projective_dag_numba import (
        open_compiled_mapped_implicit_projective_dag_artifact,
    )

    diagonal, raw = _mapped_ladder_fixture()
    plans = InternalDimensionPlans(diagonal.network)
    expected = {}
    for removal in REMOVALS:
        ranks, _record = plans.plan(removal)
        expected[removal] = masked_boundary(diagonal.network, raw, ranks)

    destination = tmp_path / "ladder"
    result = export_dimension_ladder(
        diagonal,
        raw,
        destination,
        full_rank_certificate_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
    )
    assert tuple(
        record["requested_internal_dimension_removal"]
        for record in result["rungs"]
    ) == REMOVALS
    for record in result["rungs"]:
        removal = record["requested_internal_dimension_removal"]
        artifact = record["artifact"]
        with open_mapped_implicit_projective_dag_artifact(
            destination / artifact["path"],
            expected_manifest_sha256=artifact["manifest_sha256"],
        ) as mapped:
            assert mapped.network.artifact_schema == PREFIX_ARTIFACT_SCHEMA
            observed, execution = mapped.evaluate_projective_boundary(
                raw,
                return_receipt=True,
            )
            assert _relative_pair(observed, expected[removal]) <= 2e-10
            assert execution.node_evaluations == mapped.node_count
            assert execution.edge_occurrences_emitted == mapped.edge_occurrence_count
            assert execution.root_only_live
            assert execution.all_refcounts_zero
        with open_compiled_mapped_implicit_projective_dag_artifact(
            destination / artifact["path"],
            expected_manifest_sha256=artifact["manifest_sha256"],
        ) as compiled:
            observed, execution = compiled.evaluate_projective_boundary(
                raw,
                return_receipt=True,
            )
            assert _relative_pair(observed, expected[removal]) <= 2e-10
            assert execution.root_only_live
            assert execution.all_refcounts_zero


def test_symlink_root_symbolic_member_hardlink_and_writable_member_are_rejected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    receipt = export_implicit_projective_dag_artifact(_network(), source)

    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="physical directory"):
        open_mapped_implicit_projective_dag_artifact(
            linked_root,
            expected_manifest_sha256=receipt.manifest_sha256,
        )

    symbolic = _mutable_copy(source, tmp_path / "symbolic")
    shard = sorted(symbolic.glob("*.f64le"))[0]
    shard.unlink()
    shard.symlink_to(source / shard.name)
    _freeze(symbolic)
    with pytest.raises((OSError, ValueError)):
        open_mapped_implicit_projective_dag_artifact(
            symbolic,
            expected_manifest_sha256=_manifest_digest(symbolic),
        )

    hardlinked = _mutable_copy(source, tmp_path / "hardlinked")
    linked_shard = sorted(hardlinked.glob("*.f64le"))[0]
    linked_shard.chmod(0o444)
    os.link(linked_shard, tmp_path / "outside.f64le")
    _freeze(hardlinked)
    with pytest.raises(ValueError, match="hard-linked"):
        open_mapped_implicit_projective_dag_artifact(
            hardlinked,
            expected_manifest_sha256=_manifest_digest(hardlinked),
        )

    writable = _mutable_copy(source, tmp_path / "writable")
    _freeze(writable)
    writable_shard = sorted(writable.glob("*.f64le"))[0]
    writable_shard.chmod(0o644)
    with pytest.raises(ValueError, match="immutable"):
        open_mapped_implicit_projective_dag_artifact(
            writable,
            expected_manifest_sha256=_manifest_digest(writable),
        )


@pytest.mark.parametrize("member_kind", ("manifest", "shard"))
def test_fifo_member_is_rejected_without_a_blocking_open(
    tmp_path: Path,
    member_kind: str,
) -> None:
    source = tmp_path / "source"
    receipt = export_implicit_projective_dag_artifact(_network(), source)
    damaged = _mutable_copy(source, tmp_path / f"fifo-{member_kind}")
    if member_kind == "manifest":
        target = damaged / MANIFEST_NAME
    else:
        target = sorted(damaged.glob("*.f64le"))[0]
    target.unlink()
    os.mkfifo(target, mode=0o444)
    _freeze(damaged)

    with pytest.raises(ValueError, match="physical regular file"):
        open_mapped_implicit_projective_dag_artifact(
            damaged,
            expected_manifest_sha256=receipt.manifest_sha256,
        )


def test_tamper_after_open_and_inventory_change_are_rejected_before_replay(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    receipt = export_implicit_projective_dag_artifact(_network(), source)
    mapped = open_mapped_implicit_projective_dag_artifact(
        source,
        expected_manifest_sha256=receipt.manifest_sha256,
    )
    shard = sorted(source.glob("*.f64le"))[0]
    shard.chmod(0o644)
    raw = bytearray(shard.read_bytes())
    raw[0] ^= 1
    shard.write_bytes(raw)
    shard.chmod(0o444)
    with pytest.raises(ValueError, match="changed after authentication"):
        mapped.evaluate_projective_boundary(_raw_inputs())
    mapped.close()

    inventory_source = tmp_path / "inventory-source"
    inventory_receipt = export_implicit_projective_dag_artifact(
        _network(), inventory_source
    )
    mapped = open_mapped_implicit_projective_dag_artifact(
        inventory_source,
        expected_manifest_sha256=inventory_receipt.manifest_sha256,
    )
    inventory_source.chmod(0o755)
    (inventory_source / "extra").write_bytes(b"extra")
    inventory_source.chmod(0o555)
    with pytest.raises(ValueError, match="changed after authentication"):
        mapped.evaluate_projective_boundary(_raw_inputs())
    mapped.close()


def test_context_close_releases_every_mapping_and_descriptor(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    receipt = export_implicit_projective_dag_artifact(_network(), root)
    mapped = open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    )
    mappings = mapped._mappings
    root_descriptor = mapped._root_descriptor
    member_descriptors = tuple(mapped._member_descriptors.values())
    mapped.close()
    mapped.close()
    assert all(mapping.closed for mapping in mappings)
    for descriptor in (root_descriptor, *member_descriptors):
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_kernel_exception_does_not_pin_mapped_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    receipt = export_implicit_projective_dag_artifact(_network(), root)
    mappings: tuple[object, ...] = ()
    descriptors: tuple[int, ...] = ()

    def injected_failure(
        equation: str,
        tensor: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        assert equation == "oij,bi,bj->bo"
        raise RuntimeError("injected kernel failure")

    monkeypatch.setattr(torch, "einsum", injected_failure)
    with open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as mapped:
        mappings = mapped._mappings
        descriptors = (
            mapped._root_descriptor,
            *mapped._member_descriptors.values(),
        )
        with pytest.raises(RuntimeError, match="injected kernel failure") as captured:
            mapped.evaluate_projective_boundary(_raw_inputs())
        failure_frames = []
        current = captured.value.__traceback__
        while current is not None:
            if current.tb_frame.f_code is injected_failure.__code__:
                failure_frames.append(current.tb_frame)
            current = current.tb_next
        assert len(failure_frames) == 1
        assert not failure_frames[0].f_locals
        assert not mapped.closed
    assert mappings and all(mapping.closed for mapping in mappings)
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize("relation", ("cause", "context", "group", "cycle"))
def test_entire_kernel_exception_graph_releases_mapped_operands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relation: str,
) -> None:
    group_type = getattr(builtins, "ExceptionGroup", None)
    if relation == "group" and group_type is None:
        pytest.skip("ExceptionGroup requires Python 3.11 or newer")

    root = tmp_path / relation
    receipt = export_implicit_projective_dag_artifact(_network(), root)
    mappings: tuple[object, ...] = ()

    def retained_operand(tensor: torch.Tensor) -> None:
        raise RuntimeError("nested kernel failure")

    def injected_failure(
        equation: str,
        tensor: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        assert equation == "oij,bi,bj->bo"
        del left, right
        if relation == "cycle":
            cyclic = RuntimeError("cycle kernel failure")
            cyclic.__cause__ = cyclic
            raise cyclic
        try:
            retained_operand(tensor)
        except RuntimeError as nested:
            if relation == "context":
                raise RuntimeError("context wrapper failure")
            if relation == "cause":
                wrapped = RuntimeError("cause wrapper failure")
                wrapped.__cause__ = nested
            else:
                assert group_type is not None
                wrapped = group_type("group wrapper failure", [nested])
        raise wrapped

    monkeypatch.setattr(torch, "einsum", injected_failure)
    with open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as mapped:
        mappings = mapped._mappings
        with pytest.raises(BaseException, match=f"{relation}.*failure") as captured:
            mapped.evaluate_projective_boundary(_raw_inputs())
        pending = [captured.value]
        seen: set[int] = set()
        retained_frames = []
        while pending:
            error = pending.pop()
            if id(error) in seen:
                continue
            seen.add(id(error))
            current = error.__traceback__
            while current is not None:
                if current.tb_frame.f_code is retained_operand.__code__:
                    retained_frames.append(current.tb_frame)
                current = current.tb_next
            if error.__cause__ is not None:
                pending.append(error.__cause__)
            if error.__context__ is not None:
                pending.append(error.__context__)
            if group_type is not None and isinstance(error, group_type):
                pending.extend(error.exceptions)
        if relation == "cycle":
            assert captured.value.__cause__ is captured.value
        else:
            assert len(retained_frames) == 1
            assert not retained_frames[0].f_locals
    assert mappings and all(mapping.closed for mapping in mappings)


def test_close_after_buffer_error_is_retryable_and_never_partially_open(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    receipt = export_implicit_projective_dag_artifact(_network(), root)
    mapped = open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    )
    mappings = mapped._mappings
    root_descriptor = mapped._root_descriptor
    member_descriptors = tuple(mapped._member_descriptors.values())
    exported = memoryview(mappings[0])

    with pytest.raises(RuntimeError, match="still exported.*retry close"):
        mapped.close()
    assert not mapped.closed
    assert not mappings[0].closed
    for descriptor in (root_descriptor, *member_descriptors):
        os.fstat(descriptor)
    with pytest.raises(RuntimeError, match="close is incomplete"):
        mapped.node(0)

    exported.release()
    del exported
    gc.collect()
    mapped.close()
    assert mapped.closed
    assert all(mapping.closed for mapping in mappings)
    for descriptor in (root_descriptor, *member_descriptors):
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_concurrent_close_cannot_invalidate_an_active_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    receipt = export_implicit_projective_dag_artifact(_network(), root)
    mapped = open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    )
    started = threading.Event()
    proceed = threading.Event()
    original = mapped._execute

    def blocked(
        batch_size: int,
        prepared: torch.Tensor | dict[str, torch.Tensor],
    ) -> object:
        started.set()
        assert proceed.wait(timeout=10.0)
        return original(batch_size, prepared)

    monkeypatch.setattr(mapped, "_execute", blocked)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            mapped.evaluate_projective_boundary(_raw_inputs())
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    assert started.wait(timeout=10.0)
    with pytest.raises(RuntimeError, match="during evaluation"):
        mapped.close()
    assert not mapped.closed
    proceed.set()
    worker.join(timeout=10.0)
    assert not worker.is_alive()
    assert not errors
    mapped.close()
    assert mapped.closed


def test_mutation_during_evaluation_is_caught_by_the_post_execution_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    receipt = export_implicit_projective_dag_artifact(_network(), root)
    shard = sorted(root.glob("*.f64le"))[0]
    with open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as mapped:
        original = mapped._execute

        def execute_then_mutate(
            batch_size: int,
            prepared: torch.Tensor | dict[str, torch.Tensor],
        ) -> object:
            result = original(batch_size, prepared)
            shard.chmod(0o644)
            raw = bytearray(shard.read_bytes())
            raw[-1] ^= 1
            shard.write_bytes(raw)
            shard.chmod(0o444)
            # Some test filesystems can coalesce rapid writes/chmods into one
            # metadata timestamp tick.  Force the authenticated snapshot to
            # change so this test deterministically exercises the post-run
            # check instead of depending on filesystem clock granularity.
            information = shard.stat()
            os.utime(
                shard,
                ns=(information.st_atime_ns, information.st_mtime_ns + 1_000_000_000),
                follow_symlinks=False,
            )
            return result

        monkeypatch.setattr(mapped, "_execute", execute_then_mutate)
        with pytest.raises(ValueError, match="changed after authentication"):
            mapped.evaluate_projective_boundary(_raw_inputs())


def test_partial_mapping_failure_closes_every_opened_resource(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xvla.train import implicit_projective_dag_mapped as mapped_module

    root = tmp_path / "artifact"
    receipt = export_implicit_projective_dag_artifact(
        _network(), root, shard_size_bytes=64
    )
    original = mapped_module.mmap.mmap
    mappings: list[object] = []
    descriptors: list[int] = []

    def fail_second(descriptor: int, *args: object, **kwargs: object) -> object:
        descriptors.append(descriptor)
        if len(descriptors) == 2:
            raise RuntimeError("injected mapping failure")
        result = original(descriptor, *args, **kwargs)
        mappings.append(result)
        return result

    monkeypatch.setattr(mapped_module.mmap, "mmap", fail_second)
    with pytest.raises(RuntimeError, match="injected mapping failure"):
        open_mapped_implicit_projective_dag_artifact(
            root,
            expected_manifest_sha256=receipt.manifest_sha256,
        )
    assert mappings and all(mapping.closed for mapping in mappings)
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_authenticated_nonfinite_payload_is_rejected_without_reconstruction(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    export_implicit_projective_dag_artifact(_network(), source)
    damaged = _mutable_copy(source, tmp_path / "damaged")
    manifest = json.loads((damaged / MANIFEST_NAME).read_text())
    shard_path = damaged / manifest["shards"][0]["name"]
    raw = bytearray(shard_path.read_bytes())
    raw[:8] = struct.pack("<d", float("inf"))
    shard_path.write_bytes(raw)
    manifest["shards"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    (damaged / MANIFEST_NAME).write_bytes(_canonical_bytes(manifest))
    _freeze(damaged)
    with pytest.raises(ValueError, match="nonfinite"):
        open_mapped_implicit_projective_dag_artifact(
            damaged,
            expected_manifest_sha256=_manifest_digest(damaged),
        )


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda manifest: manifest["tensors"][0]["segments"][0].__setitem__(
                "offset_bytes", 64
            ),
            "canonical packing",
        ),
        (
            lambda manifest: manifest["tensors"][0]["segments"][0].__setitem__(
                "offset_bytes", 1
            ),
            "in-range float64 span",
        ),
        (
            lambda manifest: manifest["graph"]["incoming_refcount"].__setitem__(
                -2, 1
            ),
            "refcounts",
        ),
    ),
)
def test_authenticated_descriptor_corruption_is_rejected(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    source = tmp_path / "source"
    export_implicit_projective_dag_artifact(_network(), source)
    damaged = _mutable_copy(source, tmp_path / match.replace(" ", "-"))
    manifest = json.loads((damaged / MANIFEST_NAME).read_text())
    mutation(manifest)
    (damaged / MANIFEST_NAME).write_bytes(_canonical_bytes(manifest))
    _freeze(damaged)
    with pytest.raises(ValueError, match=match):
        open_mapped_implicit_projective_dag_artifact(
            damaged,
            expected_manifest_sha256=_manifest_digest(damaged),
        )


def test_existing_eager_serializer_sources_remain_byte_identical() -> None:
    repository = Path(__file__).resolve().parents[1]
    assert hashlib.sha256(
        (repository / "xvla/train/implicit_projective_dag_artifact.py").read_bytes()
    ).hexdigest() == "e302131d021ed8fac5b8ec5bb1d321b4f3ca035487f4ba7626159d914d0f4bd9"
    assert hashlib.sha256(
        (repository / "tests/test_implicit_projective_dag_artifact.py").read_bytes()
    ).hexdigest() == "10bbd014c10f279e50a31dc19ce57fd08cb595c864c2d251cfe86ede7823c2cd"
