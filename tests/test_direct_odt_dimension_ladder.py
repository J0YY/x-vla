from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from xvla.train.direct_odt_dimension_ladder import (
    DENOMINATOR,
    REMOVALS,
    InternalDimensionPlans,
    _relative_pair,
    export_dimension_ladder,
    masked_boundary,
    slice_descendant_in_place,
    zero_preserving_boundary,
)
from xvla.train.direct_odt_truncation import evaluate_diagonal_prefixes
from xvla.train.implicit_projective_dag_artifact import export_implicit_projective_dag_artifact, load_implicit_projective_dag_artifact
from xvla.train.implicit_projective_prefix_artifact import ARTIFACT_SCHEMA, export_implicit_projective_prefix_artifact, load_implicit_projective_prefix_artifact
from xvla.train.implicit_sparse_projective_odt import (
    ImplicitProjectiveDAG,
    ImplicitNode,
    MaterializationTelemetry,
    PhysicalSourceSpec,
    UnaryCore,
    _Builder,
    _walk_unique,
    canonicalize_implicit_dag_direct_rq,
    diagonalize_implicit_dag_full_rank,
    evaluate_projective_boundary,
)


def _fixture():
    generator = torch.Generator().manual_seed(284)
    dtype = torch.float64
    width = 20
    builder = _Builder(torch.empty((), dtype=dtype), MaterializationTelemetry())
    source = builder.physical_pair(PhysicalSourceSpec("x", "physical_state", width))
    shared = builder.unary(
        "shared.learned",
        torch.randn(width, width + 1, generator=generator, dtype=dtype),
        source,
        kind="learned_projection",
    )
    binary = builder.cp(
        "shared.square",
        torch.eye(width, dtype=dtype),
        torch.randn(width, width, generator=generator, dtype=dtype),
        torch.randn(width, width, generator=generator, dtype=dtype),
        (shared, shared),
        kind="repeated_bond",
    )
    root = builder.unary("root.fixed", torch.eye(width, dtype=dtype), binary, kind="observable_boundary")
    network = ImplicitProjectiveDAG(
        root=root,
        head=torch.randn(3, width, generator=generator, dtype=dtype),
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=width,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=dtype),
        physical_sources=builder.physical_source_specs,
    )
    raw = {"x": torch.randn(3, width, generator=generator, dtype=dtype)}
    canonical = canonicalize_implicit_dag_direct_rq(network, replay_inputs=raw, block_size=32, copy_network=False)
    diagonal = diagonalize_implicit_dag_full_rank(canonical.network, replay_inputs=raw, copy_network=False, retain_eigenvalues=False, stream_pre_evd_environments=True, retain_post_evd_environments=False)
    return diagonal, raw


def _assert_boundary_equal(actual, expected):
    expected = expected / expected.abs().amax(dim=1, keepdim=True)
    actual = actual / actual.abs().amax(dim=1, keepdim=True)
    pivot = expected.abs().argmax(dim=1, keepdim=True)
    sign = torch.sign(expected.gather(1, pivot) * actual.gather(1, pivot))
    torch.testing.assert_close(actual * sign, expected, atol=2e-10, rtol=2e-10)


def test_internal_counts_exclude_leaf_and_root_and_preserve_original_widths():
    diagonal, _raw = _fixture()
    plans = InternalDimensionPlans(diagonal.network)
    assert plans.internal_uids.size == 2
    for removal in REMOVALS:
        ranks, record = plans.plan(removal)
        assert record["dimension_denominator"] == DENOMINATOR
        assert record["original_internal_dimensions"] == 40
        assert ranks[diagonal.network.root.uid] == diagonal.network.root.output_dimension
        for node in _walk_unique(diagonal.network.root):
            if not node.children:
                assert ranks[node.uid] == node.output_dimension
        removed = sum(int(plans.widths[uid] - ranks[uid]) for uid in plans.internal_uids)
        assert record["actual_internal_dimension_removal"] == pytest.approx(removed / 40)
        assert record["actual_all_node_output_dimension_removal"] == pytest.approx(removed / plans.original_all_dimensions)


def test_zero_mask_evaluator_matches_original_canonical_prefix_oracle():
    diagonal, raw = _fixture()
    plans = InternalDimensionPlans(diagonal.network)
    for removal in REMOVALS:
        ranks, _record = plans.plan(removal)
        mapping = {node.uid: int(ranks[node.uid]) for node in _walk_unique(diagonal.network.root)}
        expected = evaluate_diagonal_prefixes(diagonal, raw, mapping)
        actual = masked_boundary(diagonal.network, raw, ranks)
        _assert_boundary_equal(actual, expected)


def test_ladder_reload_matches_each_original_mask_and_publishes_before_callback(tmp_path):
    diagonal, raw = _fixture()
    plans = InternalDimensionPlans(diagonal.network)
    expected = {}
    for removal in REMOVALS:
        ranks, _ = plans.plan(removal)
        mapping = {node.uid: int(ranks[node.uid]) for node in _walk_unique(diagonal.network.root)}
        expected[removal] = evaluate_diagonal_prefixes(diagonal, raw, mapping)
    destination = tmp_path / "ladder"
    callback_receipts = []

    def callback(record):
        path = destination / f"{record['artifact']['path']}.json"
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["receipt_sha256"]
        callback_receipts.append(record["receipt_sha256"])

    result = export_dimension_ladder(diagonal, raw, destination, full_rank_certificate_sha256="a" * 64, checkpoint_sha256="b" * 64, on_rung=callback)
    assert len(callback_receipts) == 6
    assert result["all_requested_artifacts_exported"] is True
    assert result["closed_loop_capability_completed"] is False
    previous = None
    for record in result["rungs"]:
        removal = record["requested_internal_dimension_removal"]
        network = load_implicit_projective_prefix_artifact(destination / record["artifact"]["path"], expected_manifest_sha256=record["artifact"]["manifest_sha256"])
        assert not network.algorithm1_direct_rq_complete
        assert record["parent_prefix_manifest_sha256"] == previous
        previous = record["artifact"]["manifest_sha256"]
        _assert_boundary_equal(evaluate_projective_boundary(network, raw), expected[removal])
        assert record["expected_occurrence_slices"] == record["applied_occurrence_slices"]
        assert record["physical_storage_elements"] == record["allocated_storage_elements"]
        assert record["masked_materialized_boundary_max_error"] < 2e-10
    assert result["rungs"][0]["expected_occurrence_slices"] == 3
    sidecar = destination / result["original_dimensions"]["path"]
    payload = sidecar.read_bytes()
    assert len(payload) == 5 * len(plans.widths)
    assert np.array_equal(np.frombuffer(payload[:4 * len(plans.widths)], dtype="<u4"), plans.widths)
    assert np.array_equal(np.frombuffer(payload[4 * len(plans.widths):], dtype=np.uint8), plans.eligible)
    assert hashlib.sha256(payload).hexdigest() == result["original_dimensions"]["sha256"]
    assert json.loads((destination / "ladder.json").read_text())["dimension_denominator"] == DENOMINATOR


def test_cannot_reenter_from_a_previously_truncated_network(tmp_path):
    diagonal, raw = _fixture()
    export_dimension_ladder(diagonal, raw, tmp_path / "one", full_rank_certificate_sha256="a" * 64, checkpoint_sha256="b" * 64, removals=(0.30,))
    with pytest.raises(ValueError, match="completed direct Algorithm 1"):
        export_dimension_ladder(diagonal, raw, tmp_path / "two", full_rank_certificate_sha256="a" * 64, checkpoint_sha256="b" * 64)


def test_descendant_cannot_restore_dimensions_or_alter_boundaries():
    diagonal, _raw = _fixture()
    network = diagonal.network
    plans = InternalDimensionPlans(network)
    ranks, _ = plans.plan(0.40)
    slice_descendant_in_place(network, plans, plans.widths, ranks)
    milder, _ = plans.plan(0.30)
    with pytest.raises(ValueError, match="restore"):
        slice_descendant_in_place(network, plans, ranks, milder)
    for uid in [plans.root_uid, *[node.uid for node in _walk_unique(network.root) if not node.children]]:
        invalid = ranks.copy()
        invalid[uid] -= 1
        with pytest.raises(ValueError, match="ingress and root"):
            slice_descendant_in_place(network, plans, ranks, invalid)


@pytest.mark.parametrize("value", [True, 1, float("nan"), float("inf"), -0.1, 1.0])
def test_invalid_removal_values_fail(value):
    diagonal, _raw = _fixture()
    with pytest.raises(ValueError):
        InternalDimensionPlans(diagonal.network).plan(value)


def test_existing_output_is_not_overwritten(tmp_path):
    diagonal, raw = _fixture()
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "sentinel"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        export_dimension_ladder(diagonal, raw, destination, full_rank_certificate_sha256="a" * 64, checkpoint_sha256="b" * 64)
    assert sentinel.read_text() == "preserve"


def _planning_chain(widths):
    previous = ImplicitNode(0, "physical", UnaryCore(torch.ones(2, 2, dtype=torch.float64), "physical"), physical_token=0)
    for uid, width in enumerate(widths, start=1):
        previous = ImplicitNode(uid, f"internal.{uid}", UnaryCore(torch.ones(width, previous.output_dimension, dtype=torch.float64), "learned"), children=(previous,))
    root = ImplicitNode(len(widths) + 1, "root", UnaryCore(torch.ones(2, previous.output_dimension, dtype=torch.float64), "root"), children=(previous,))
    return ImplicitProjectiveDAG(root=root, head=torch.eye(2, dtype=torch.float64), head_binary_exponent=0, token_count=1, feature_dimension=1, selected_token=0, mask=torch.ones(1, 1, dtype=torch.float64))


def test_repeated_width_ties_hit_exact_global_budgets_and_are_nested():
    plans = InternalDimensionPlans(_planning_chain([10] * 300))
    previous = plans.widths
    for requested in (0.301, 0.317, 0.333, 0.499, 0.601, 0.707, 0.801):
        ranks, record = plans.plan(requested)
        assert record["actual_internal_dimension_removal"] == pytest.approx(requested)
        assert np.all(ranks <= previous)
        assert np.all(np.diff(ranks[plans.internal_uids].astype(np.int64)) >= 0)
        previous = ranks


def test_mixed_width_rational_threshold_ties_remain_nested():
    plans = InternalDimensionPlans(_planning_chain([2, 3, 6, 10] * 80))
    previous = plans.widths
    for point in range(0, 81):
        requested = float(point / 100)
        ranks, record = plans.plan(requested)
        assert np.all(ranks <= previous)
        assert abs(record["actual_internal_dimension_removal"] - requested) <= 0.5 / plans.original_internal_dimensions
        previous = ranks


def test_minimum_rank_capacity_rejects_unattainable_actual_removal():
    plans = InternalDimensionPlans(_planning_chain([2] * 50))
    with pytest.raises(ValueError, match="unattainable"):
        plans.plan(0.80)


def test_prefix_codec_is_additive_and_does_not_weaken_canonical_codec(tmp_path):
    diagonal, raw = _fixture()
    with pytest.raises(ValueError, match="clear"):
        export_implicit_projective_prefix_artifact(diagonal.network, tmp_path / "false_prefix")
    result = export_dimension_ladder(diagonal, raw, tmp_path / "ladder", full_rank_certificate_sha256="a" * 64, checkpoint_sha256="b" * 64, removals=(0.30,))
    artifact = result["rungs"][0]["artifact"]
    path = tmp_path / "ladder" / artifact["path"]
    assert json.loads((path / "manifest.json").read_text())["schema"] == ARTIFACT_SCHEMA
    with pytest.raises(ValueError, match="schema"):
        load_implicit_projective_dag_artifact(path, expected_manifest_sha256=artifact["manifest_sha256"])
    with pytest.raises(ValueError, match="completed direct-RQ"):
        export_implicit_projective_dag_artifact(diagonal.network, tmp_path / "false_canonical")


def test_unchanged_compact_prefix_reuses_all_tensor_storage():
    diagonal, _raw = _fixture()
    network = diagonal.network
    plans = InternalDimensionPlans(network)
    ranks, _ = plans.plan(0.40)
    slice_descendant_in_place(network, plans, plans.widths, ranks)
    head_pointer = network.head.data_ptr()
    cores = [node.core for node in _walk_unique(network.root)]
    result = slice_descendant_in_place(network, plans, ranks, ranks)
    assert result["copied_tensors"] == 0
    assert result["reused_compact_tensors"] > 0
    assert result["reduced_bonds_in_transition"] == 0
    assert network.head.data_ptr() == head_pointer
    assert all(node.core is core for node, core in zip(_walk_unique(network.root), cores))
    assert result["physical_storage_elements"] == result["allocated_storage_elements"]


def test_shared_backing_storage_is_compacted_once_and_no_longer_aliased():
    network = _planning_chain([2, 2])
    nodes = _walk_unique(network.root)
    shared_matrix = torch.ones(2, 2, dtype=torch.float64)
    nodes[1].core = UnaryCore(shared_matrix, "shared_storage")
    nodes[2].core = UnaryCore(shared_matrix, "shared_storage")
    plans = InternalDimensionPlans(network)
    ranks, _ = plans.plan(0.0)
    raw = torch.tensor([[[0.2]]], dtype=torch.float64)
    expected = evaluate_projective_boundary(network, raw)
    result = slice_descendant_in_place(network, plans, plans.widths, ranks)
    assert result["copied_tensors"] >= 1
    assert nodes[1].core.matrix.data_ptr() != nodes[2].core.matrix.data_ptr()
    assert result["physical_storage_elements"] == result["allocated_storage_elements"]
    _assert_boundary_equal(evaluate_projective_boundary(network, raw), expected)


def test_exact_zero_boundary_comparison_never_accepts_zero_nonzero_mismatch():
    zero = torch.zeros(1, 2, dtype=torch.float64)
    assert _relative_pair(zero, zero) == 0.0
    nonzero = torch.tensor([[0.0, 0.5]], dtype=torch.float64)
    assert _relative_pair(zero, nonzero) == 1.0
    assert _relative_pair(nonzero, zero) == 1.0
    assert _relative_pair(-4.0 * nonzero, nonzero) == 0.0


def test_zero_preserving_validation_rejects_nonfinite_values():
    network = _planning_chain([2])
    raw = torch.tensor([[[float("inf")]]], dtype=torch.float64)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        zero_preserving_boundary(network, raw)


def test_undefined_prefix_chart_exports_without_inventing_actions(tmp_path):
    dtype = torch.float64
    builder = _Builder(torch.empty((), dtype=dtype), MaterializationTelemetry())
    source = builder.physical_pair(PhysicalSourceSpec("x", "state", 1))
    shared = builder.unary("internal", torch.eye(2, dtype=dtype), source)
    root = builder.unary("root", torch.eye(2, dtype=dtype), shared)
    network = ImplicitProjectiveDAG(root=root, head=torch.diag(torch.tensor([1.0, 0.1], dtype=dtype)), head_binary_exponent=0, token_count=1, feature_dimension=1, selected_token=0, mask=torch.ones(1, 1, dtype=dtype), physical_sources=builder.physical_source_specs)
    raw = {"x": torch.tensor([[0.0], [0.4]], dtype=dtype)}
    canonical = canonicalize_implicit_dag_direct_rq(network, replay_inputs=raw, block_size=32, copy_network=False)
    diagonal = diagonalize_implicit_dag_full_rank(canonical.network, replay_inputs=raw, copy_network=False, retain_eigenvalues=False, stream_pre_evd_environments=True, retain_post_evd_environments=False)
    result = export_dimension_ladder(diagonal, raw, tmp_path / "zero_chart", full_rank_certificate_sha256="a" * 64, checkpoint_sha256="b" * 64, removals=(0.0, 0.5))
    assert len(result["rungs"]) == 2
    assert result["rungs"][0]["valid_projective_chart_on_replay"] is True
    invalid = result["rungs"][1]
    assert invalid["replay_chart_status"] == "undefined_in_float64_execution"
    assert invalid["valid_projective_chart_on_replay"] is False
    assert invalid["action_relative_error_on_replay"] is None
    assert invalid["undefined_replay_row_indices"] == [0, 1]
    assert invalid["masked_materialized_boundary_max_error"] == 0.0
    assert invalid["replay_boundary"][0] == [0.0, 0.0]
