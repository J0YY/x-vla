from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import pytest
import torch

from scripts.odt_direct_only_compliance import (
    DirectOnlyComplianceError,
    audit_direct_only_launch,
)
from xvla.models.vla import ChiVLA, VLAConfig
from xvla.nn.normalization import RationalNorm
from xvla.train.direct_odt_truncation import (
    CompactRankBank,
    _slice_input,
    _slice_output,
    evaluate_diagonal_prefixes,
    evaluate_diagonal_suffixes,
    implicit_allocated_storage_elements,
    implicit_storage_elements,
    projected_prefix_storage_elements,
    truncate_diagonal_prefixes,
)
from xvla.train.implicit_sparse_projective_odt import (
    ImplicitNode,
    ImplicitProjectiveDAG,
    MaterializationTelemetry,
    PhysicalSourceSpec,
    UnaryCore,
    _Builder,
    _clone_network,
    _compact_spectrum_record,
    _parent_occurrences,
    _projective_batch_relative_error,
    _validate_network,
    _walk_unique,
    canonicalize_implicit_dag_direct_rq,
    diagonalize_implicit_dag_full_rank,
    diagonalize_shared_and_explicit_clone_independently,
    evaluate_projective_boundary,
)
from xvla.train.implicit_sparse_projective_odt_vla import (
    PRODUCT_COMPONENTS_OBSERVABLE,
    compile_full_vla_projective_dag,
    physical_batch_from_model_inputs,
    physical_source_mapping,
)


DTYPE = torch.float64


def _shared_network() -> tuple[ImplicitProjectiveDAG, dict[str, torch.Tensor]]:
    meter = MaterializationTelemetry()
    builder = _Builder(torch.empty((), dtype=DTYPE), meter)
    source = builder.physical_pair(PhysicalSourceSpec("x", "state.shared", 2))
    shared = builder.unary(
        "state.shared_chart",
        torch.tensor(
            ((1.0, 0.2, 0.1), (0.1, 0.9, -0.05), (0.0, 0.0, 1.0)),
            dtype=DTYPE,
        ),
        source,
        kind="shared_chart",
    )
    assert isinstance(shared, ImplicitNode)
    left = torch.zeros(4, 3, dtype=DTYPE)
    right = torch.zeros(4, 3, dtype=DTYPE)
    left[0, 0], right[0, 0] = 1.0, 1.0
    left[1, 1], right[1, 1] = 1.0, 1.0
    left[2, 0], right[2, 2] = 1.0, 1.0
    left[3, 2], right[3, 2] = 1.0, 1.0
    root = builder.cp(
        "joint.shared_square",
        torch.eye(4, dtype=DTYPE),
        left,
        right,
        (shared, shared),
        kind="shared_square",
    )
    assert isinstance(root, ImplicitNode)
    head = torch.tensor(
        ((1.0, 0.5, 0.2, 0.0), (0.0, 0.0, 0.0, 1.0)),
        dtype=DTYPE,
    )
    network = ImplicitProjectiveDAG(
        root=root,
        head=head,
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=1,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=DTYPE),
        physical_sources=builder.physical_source_specs,
    )
    raw = {
        "x": torch.tensor(
            ((-0.31, 0.17), (0.23, -0.09), (0.19, 0.27)), dtype=DTYPE
        )
    }
    return network, raw


def _diagonal_shared_network():
    network, raw = _shared_network()
    canonical = canonicalize_implicit_dag_direct_rq(
        network, replay_inputs=raw, block_size=32
    )
    bank = CompactRankBank.for_network(canonical.network)
    diagonal = diagonalize_implicit_dag_full_rank(
        canonical.network,
        replay_inputs=raw,
        retain_eigenvalues=False,
        compact_spectrum_callback=bank,
    )
    assert bank.complete
    return canonical, diagonal, bank, raw


def _truncate_all_clone_occurrences(
    network: ImplicitProjectiveDAG,
    plan: dict[int, int],
    *,
    omit: tuple[int, int] | None = None,
) -> tuple[ImplicitProjectiveDAG, int]:
    work = _clone_network(network, unfold=False)
    parents = _parent_occurrences(work.root)
    by_origin: dict[int, list[ImplicitNode]] = defaultdict(list)
    for node in _walk_unique(work.root):
        if node.origin_uid is None:
            raise AssertionError("clone occurrence lost its shared origin")
        by_origin[node.origin_uid].append(node)
    omitted = 0
    occurrence_index = 0
    for origin_uid, rank in plan.items():
        for node in by_origin[origin_uid]:
            node.core = _slice_output(node.core, rank)
            occurrences = parents.get(id(node), [])
            if occurrences:
                for parent, role in occurrences:
                    should_omit = omit == (origin_uid, occurrence_index)
                    occurrence_index += 1
                    if should_omit:
                        omitted += 1
                        continue
                    parent.core = _slice_input(parent.core, role, rank)
            else:
                work.head = work.head[:, :rank]
    _validate_network(work)
    return work, omitted


def test_compact_callback_is_complete_and_cannot_retain_full_records():
    canonical, diagonal, bank, _ = _diagonal_shared_network()
    summary = bank.finish()
    assert summary["full_spectra_retained"] is False
    assert summary["node_count"] == diagonal.diagonalized_node_count
    assert summary["overall"]["node_count"] == diagonal.diagonalized_node_count
    assert diagonal.eigenvalues == ()
    assert not diagonal.eigenvalue_spectra_retained

    second_bank = CompactRankBank.for_network(canonical.network)
    with pytest.raises(ValueError, match="retention"):
        diagonalize_implicit_dag_full_rank(
            canonical.network,
            compact_spectrum_callback=second_bank,
        )

    used_bank = CompactRankBank.for_network(canonical.network)
    diagonalize_implicit_dag_full_rank(
        canonical.network,
        retain_eigenvalues=False,
        compact_spectrum_callback=used_bank,
    )
    with pytest.raises(RuntimeError, match="rebound"):
        diagonalize_implicit_dag_full_rank(
            canonical.network,
            retain_eigenvalues=False,
            compact_spectrum_callback=used_bank,
        )


def test_compact_summary_keeps_a_tie_cluster_and_rejects_material_negativity():
    meter = MaterializationTelemetry()
    builder = _Builder(torch.empty((), dtype=DTYPE), meter)
    root = builder.physical_pair(PhysicalSourceSpec("x", "state.tie", 2))
    head = torch.diag(torch.sqrt(torch.tensor((0.85, 0.075, 0.075), dtype=DTYPE)))
    network = ImplicitProjectiveDAG(
        root=root,
        head=head,
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=1,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=DTYPE),
        physical_sources=builder.physical_source_specs,
    )
    raw = {"x": torch.tensor(((0.2, -0.1),), dtype=DTYPE)}
    canonical = canonicalize_implicit_dag_direct_rq(network, replay_inputs=raw)
    bank = CompactRankBank.for_network(canonical.network)
    diagonal = diagonalize_implicit_dag_full_rank(
        canonical.network,
        replay_inputs=raw,
        retain_eigenvalues=False,
        compact_spectrum_callback=bank,
    )
    assert bank.plan(0.9)[root.uid] == 3
    assert bank.finish()["overall"]["tie_extended_nodes"]["0.9"] == 1

    with pytest.raises(RuntimeError, match="materially negative"):
        _compact_spectrum_record(
            0,
            "bad",
            torch.tensor((1.0, -0.1), dtype=DTYPE),
            0,
        )


def test_every_prefix_step_matches_the_no_memo_clone_and_masked_replay():
    canonical, diagonal, bank, raw = _diagonal_shared_network()
    explicit = _clone_network(canonical.network, unfold=True)
    oracle = diagonalize_shared_and_explicit_clone_independently(
        canonical.network, explicit, raw
    )
    fixed = bank.plan(0.9)
    candidates = [
        node
        for node in _walk_unique(diagonal.network.root)
        if fixed[node.uid] < node.output_dimension
    ]
    if not candidates:
        candidates = [
            next(
                node
                for node in _walk_unique(diagonal.network.root)
                if node.output_dimension > 1
            )
        ]
    cumulative: dict[int, int] = {}
    for node in reversed(candidates):
        cumulative[node.uid] = min(fixed[node.uid], node.output_dimension - 1)
        materialized = truncate_diagonal_prefixes(diagonal, cumulative)
        clone, omitted = _truncate_all_clone_occurrences(
            oracle.explicit_clone_network, cumulative
        )
        assert omitted == 0
        shared_value = evaluate_projective_boundary(materialized.network, raw)
        clone_value = evaluate_projective_boundary(clone, raw)
        masked_value = evaluate_diagonal_prefixes(diagonal, raw, cumulative)
        assert _projective_batch_relative_error(shared_value, clone_value) < 2e-10
        assert _projective_batch_relative_error(shared_value, masked_value) < 2e-10
        assert (
            materialized.expected_occurrence_slices
            == materialized.applied_occurrence_slices
        )
        assert materialized.truncated_storage_elements == implicit_storage_elements(
            materialized.network
        )
        assert materialized.allocated_storage_elements == implicit_allocated_storage_elements(
            materialized.network
        )
        assert materialized.allocated_storage_elements == materialized.truncated_storage_elements
        assert materialized.truncated_storage_elements == projected_prefix_storage_elements(
            diagonal, cumulative
        )


def test_omitting_one_shared_occurrence_is_rejected_before_replay():
    canonical, diagonal, _, raw = _diagonal_shared_network()
    explicit = _clone_network(canonical.network, unfold=True)
    oracle = diagonalize_shared_and_explicit_clone_independently(
        canonical.network, explicit, raw
    )
    shared_node = next(
        node
        for node in _walk_unique(diagonal.network.root)
        if node.label == "state.shared_chart"
    )
    plan = {shared_node.uid: shared_node.output_dimension - 1}
    correct, omitted = _truncate_all_clone_occurrences(
        oracle.explicit_clone_network, plan
    )
    assert omitted == 0
    evaluate_projective_boundary(correct, raw)
    with pytest.raises(ValueError, match="bond mismatch"):
        _truncate_all_clone_occurrences(
            oracle.explicit_clone_network, plan, omit=(shared_node.uid, 0)
        )


def test_plans_and_certificate_gates_fail_closed(tmp_path):
    _, diagonal, bank, raw = _diagonal_shared_network()
    root = diagonal.network.root
    with pytest.raises(ValueError, match="unknown"):
        evaluate_diagonal_prefixes(diagonal, raw, {10_000_000: 1})
    with pytest.raises(ValueError, match="integers"):
        evaluate_diagonal_prefixes(diagonal, raw, {root.uid: True})
    with pytest.raises(ValueError, match="outside"):
        evaluate_diagonal_prefixes(diagonal, raw, {root.uid: 0})
    incomplete = replace(
        diagonal, pushed_parent_occurrences=diagonal.pushed_parent_occurrences - 1
    )
    with pytest.raises(ValueError, match="every Algorithm 3 occurrence"):
        evaluate_diagonal_prefixes(incomplete, raw, {})
    for offdiagonal in (float("nan"), 1e-4):
        invalid_diagonal = replace(
            diagonal,
            maximum_recontracted_offdiagonal_ratio=offdiagonal,
        )
        with pytest.raises(ValueError, match="diagonal Algorithm 3 environments"):
            evaluate_diagonal_prefixes(invalid_diagonal, raw, {})

    _, other_diagonal, _, other_raw = _diagonal_shared_network()
    with pytest.raises(ValueError, match="different network instance"):
        evaluate_diagonal_prefixes(other_diagonal, other_raw, bank.plan(0.9))

    if root.output_dimension > 1:
        plan = {root.uid: root.output_dimension - 1}
        leading = evaluate_diagonal_prefixes(diagonal, raw, plan)
        trailing = evaluate_diagonal_suffixes(diagonal, raw, plan)
        assert _projective_batch_relative_error(leading, trailing) > 1e-7

    hidden_network, _ = _shared_network()
    hidden_node = next(
        node
        for node in _walk_unique(hidden_network.root)
        if node.label == "state.shared_chart"
    )
    assert isinstance(hidden_node.core, UnaryCore)
    matrix = hidden_node.core.matrix
    backing = torch.empty(
        matrix.shape[0], matrix.shape[1] * 2, dtype=matrix.dtype
    )
    hidden_view = backing[:, : matrix.shape[1]]
    hidden_view.copy_(matrix)
    hidden_node.core = UnaryCore(
        hidden_view,
        hidden_node.core.kind,
        hidden_node.core.binary_exponent,
    )
    with pytest.raises(RuntimeError, match="hidden backing storage"):
        implicit_allocated_storage_elements(hidden_network)

    repeated_source = tmp_path / "repeated.py"
    repeated_source.write_text(
        "def repeated():\n    return 1\n\ndef repeated():\n    return 2\n"
    )
    with pytest.raises(
        DirectOnlyComplianceError,
        match="duplicate_top_level_definition:repeated",
    ):
        audit_direct_only_launch(
            tmp_path,
            (repeated_source,),
            require_direct_qr=False,
        )


def _wide_dependent_network():
    meter = MaterializationTelemetry()
    builder = _Builder(torch.empty((), dtype=DTYPE), meter)
    x = builder.physical_pair(PhysicalSourceSpec("x", "state.left", 1))
    y = builder.physical_pair(PhysicalSourceSpec("y", "state.right", 192))
    left = torch.zeros(386, 2, dtype=DTYPE)
    right = torch.zeros(386, 193, dtype=DTYPE)
    for atom in range(386):
        left[atom, atom // 193] = 1.0
        right[atom, atom % 193] = 1.0
    output = torch.zeros(385, 386, dtype=DTYPE)
    output[:383, :383] = torch.eye(383, dtype=DTYPE)
    output[383, 0] = 1.0
    output[384, 385] = 1.0
    root = builder.cp(
        "policy.pade.wide_dependent",
        output,
        left,
        right,
        (x, y),
        kind="wide_dependent",
    )
    assert isinstance(root, ImplicitNode)
    head = torch.zeros(2, 385, dtype=DTYPE)
    head[0, 0] = 1.0
    head[1, 384] = 1.0
    network = ImplicitProjectiveDAG(
        root=root,
        head=head,
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=1,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=DTYPE),
        physical_sources=builder.physical_source_specs,
    )
    raw = {
        "x": torch.tensor(((0.2,), (-0.3,)), dtype=DTYPE),
        "y": torch.linspace(-0.4, 0.5, 384, dtype=DTYPE).reshape(2, 192),
    }
    return network, raw


def test_wide_dependent_regression_compresses_without_a_pivot_gate():
    network, raw = _wide_dependent_network()
    canonical = canonicalize_implicit_dag_direct_rq(
        network, replay_inputs=raw, block_size=64
    )
    root_step = next(
        step for step in canonical.steps if step.label == "policy.pade.wide_dependent"
    )
    assert root_step.unfolding_shape == (385, 386)
    bank = CompactRankBank.for_network(canonical.network)
    diagonal = diagonalize_implicit_dag_full_rank(
        canonical.network,
        replay_inputs=raw,
        retain_eigenvalues=False,
        compact_spectrum_callback=bank,
    )
    plan = bank.plan(0.999)
    projected = projected_prefix_storage_elements(diagonal, plan)
    full = implicit_storage_elements(diagonal.network)
    materialized = truncate_diagonal_prefixes(diagonal, plan)
    masked = evaluate_diagonal_prefixes(diagonal, raw, plan)
    explicit_value = evaluate_projective_boundary(materialized.network, raw)
    assert projected < full
    assert materialized.truncated_storage_elements == projected
    assert materialized.allocated_storage_elements == projected
    assert _projective_batch_relative_error(masked, explicit_value) < 2e-10


def _minimal_product_model() -> ChiVLA:
    torch.manual_seed(31)
    model = ChiVLA(
        VLAConfig(
            image_size=1,
            patch_size=1,
            vit_dim=2,
            vit_layers=1,
            vit_heads=1,
            vit_ffn_rank=3,
            vocab_size=2,
            max_instr_len=1,
            state_dim=1,
            n_embodiments=1,
            dim=4,
            n_layers=1,
            n_heads=1,
            ffn_rank=5,
            attn="bilinear",
            vit_attn="bilinear",
            ffn="bilinear",
            norm="rational",
            qk_norm="rational",
            residual=True,
            vit_residual=True,
            action_horizon=2,
            action_dim=2,
            action_head="product",
            head_rank=3,
            n_factors=2,
        )
    ).to(dtype=DTYPE).eval()
    with torch.no_grad():
        for index, module in enumerate(
            item for item in model.modules() if isinstance(item, RationalNorm)
        ):
            module.running_ms.fill_(0.83 + 0.023 * index)
            module.initialized.fill_(True)
            module.frozen = True
    return model


def test_product_component_root_census_covers_center_factors_and_gates():
    model = _minimal_product_model()
    physical = physical_batch_from_model_inputs(
        model,
        torch.tensor([[[[0.13]], [[-0.17]], [[0.29]]]], dtype=DTYPE),
        torch.tensor(((1,),), dtype=torch.long),
        torch.tensor(((0.31,),), dtype=DTYPE),
        torch.zeros(1, dtype=torch.long),
    )
    oracle = compile_full_vla_projective_dag(model)
    assert oracle.observable_kind == PRODUCT_COMPONENTS_OBSERVABLE
    raw = physical_source_mapping(model, physical)
    canonical = canonicalize_implicit_dag_direct_rq(
        oracle.network, replay_inputs=raw, block_size=64
    )
    bank = CompactRankBank.for_network(canonical.network)
    diagonal = diagonalize_implicit_dag_full_rank(
        canonical.network,
        replay_inputs=raw,
        retain_eigenvalues=False,
        compact_spectrum_callback=bank,
    )
    groups = bank.finish()["groups"]
    assert groups["product_center"]["node_count"] > 0
    assert groups["product_factors"]["node_count"] > 0
    assert groups["product_gates"]["node_count"] > 0
    plan = bank.plan(0.999)
    masked = evaluate_diagonal_prefixes(diagonal, raw, plan)
    materialized = truncate_diagonal_prefixes(diagonal, plan)
    replay = evaluate_projective_boundary(materialized.network, raw)
    assert materialized.reduced_bonds > 0
    assert materialized.truncated_storage_elements < materialized.original_storage_elements
    assert materialized.allocated_storage_elements == materialized.truncated_storage_elements
    assert materialized.expected_occurrence_slices > 0
    assert materialized.expected_occurrence_slices == materialized.applied_occurrence_slices
    assert _projective_batch_relative_error(masked, replay) < 2e-10


def test_materialization_projective_comparison_ignores_rowwise_radial_scale():
    _, diagonal, bank, raw = _diagonal_shared_network()
    plan = bank.plan(0.999)
    masked = evaluate_diagonal_prefixes(diagonal, raw, plan)
    materialized = truncate_diagonal_prefixes(diagonal, plan)
    replay = evaluate_projective_boundary(materialized.network, raw)
    rowwise_scale = torch.tensor((0.5, -2.0, 8.0), dtype=DTYPE)[:, None]
    rescaled_replay = replay * rowwise_scale
    assert _projective_batch_relative_error(replay, masked) < 2e-10
    assert _projective_batch_relative_error(rescaled_replay, masked) < 2e-10
    assert _projective_batch_relative_error(rescaled_replay, replay) < 1e-15
    assert not torch.allclose(rescaled_replay, replay, rtol=1e-12, atol=1e-12)
