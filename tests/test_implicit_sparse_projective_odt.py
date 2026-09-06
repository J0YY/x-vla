import torch
import torch.nn as nn
import pytest

from xvla.nn.block import ChiTransformerBlock
from xvla.nn.normalization import RationalNorm
from xvla.train.implicit_sparse_projective_odt import (
    BOUNDED_EXPLICIT_RQ_METHOD,
    DIRECT_RQ_METHOD,
    UNARY_RQ_METHOD,
    ImplicitNode,
    ImplicitProjectiveDAG,
    MaterializationTelemetry,
    PhysicalSourceSpec,
    _Builder,
    audit_algorithm1_factorization_calls,
    canonicalize_implicit_dag_direct_rq,
    canonicalize_with_explicit_clone_step_trace,
    compare_shared_and_explicit_clone_environments,
    compile_implicit_projective_block_boundary,
    diagonalize_implicit_dag_full_rank,
    diagonalize_shared_and_explicit_clone_independently,
    drop_first_residual_add_cross_term,
    evaluate_boundary_quotient,
    implicit_shape_statistics,
    multiblock_tsqr_dense_equivalence,
    old_project_then_add_topology_is_rejected,
    reverse_implicit_environments,
    source_block_boundary,
    telemetry_dict,
    tiny_dense_direct_rq_equivalence,
)


DTYPE = torch.float64


def _oracle(seed: int = 0, *, tokens: int = 3, dimension: int = 2, heads: int = 1, rank: int = 3):
    torch.manual_seed(seed)
    block = ChiTransformerBlock(
        dim=dimension,
        n_heads=heads,
        ffn_rank=rank,
        n_layers=1,
        causal=True,
        norm="rational",
        qk_norm="rational",
        residual=True,
    ).double().eval()
    generator = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                module.weight.copy_(
                    0.07
                    + 0.11
                    * torch.rand(module.weight.shape, generator=generator, dtype=DTYPE)
                )
                if module.bias is not None:
                    module.bias.copy_(
                        -0.05
                        + 0.10
                        * torch.rand(module.bias.shape, generator=generator, dtype=DTYPE)
                    )
        for index, module in enumerate(
            item for item in block.modules() if isinstance(item, RationalNorm)
        ):
            module.running_ms.fill_(0.9 + 0.13 * index)
            module.initialized.fill_(True)
            module.frozen = True
    positions = torch.stack(
        [
            torch.linspace(-0.04 + 0.01 * token, 0.06 + 0.015 * token, dimension, dtype=DTYPE)
            for token in range(tokens)
        ]
    )
    return compile_implicit_projective_block_boundary(block, positions)


def _inputs(tokens: int, dimension: int) -> torch.Tensor:
    values = torch.linspace(-0.31, 0.37, 4 * tokens * dimension, dtype=DTYPE)
    result = values.reshape(4, tokens, dimension)
    result[1].copy_(result[1].flip(0))
    result[2].mul_(-0.8)
    result[3].zero_()
    return result


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        ((actual - expected).norm() / expected.norm().clamp_min(torch.finfo(DTYPE).tiny)).item()
    )


def _bounded_old_project_then_add_network():
    """A deficient 13x25 project-then-add core with a valid quotient."""

    builder = _Builder(torch.empty((), dtype=DTYPE), MaterializationTelemetry())
    left_value = builder.physical_pair(
        PhysicalSourceSpec("left", "old_project_then_add.left", 4)
    )
    right_value = builder.physical_pair(
        PhysicalSourceSpec("right", "old_project_then_add.right", 4)
    )
    output = torch.zeros(13, 9, dtype=DTYPE)
    left = torch.zeros(9, 5, dtype=DTYPE)
    right = torch.zeros(9, 5, dtype=DTYPE)
    for coordinate in range(4):
        output[coordinate, coordinate] = 1.0
        left[coordinate, coordinate] = 1.0
        right[coordinate, -1] = 1.0
        atom = 4 + coordinate
        output[4 + coordinate, atom] = 1.0
        left[atom, -1] = 1.0
        right[atom, coordinate] = 1.0
    output[-1, -1] = 1.0
    left[-1, -1] = 1.0
    right[-1, -1] = 1.0
    root = builder.cp(
        "old_project_then_add.bounded_deficient",
        output,
        left,
        right,
        (left_value, right_value),
        kind="old_project_then_add_bounded_deficient_control",
    )
    assert isinstance(root, ImplicitNode)
    network = ImplicitProjectiveDAG(
        root=root,
        head=torch.eye(13, dtype=DTYPE),
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=1,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=DTYPE),
        physical_sources=builder.physical_source_specs,
    )
    raw = {
        "left": torch.tensor(
            ((-0.31, 0.17, 0.23, -0.09), (0.19, -0.27, 0.05, 0.32)),
            dtype=DTYPE,
        ),
        "right": torch.tensor(
            ((0.11, -0.21, 0.07, 0.29), (-0.14, 0.09, 0.28, -0.18)),
            dtype=DTYPE,
        ),
    }
    return network, raw


def test_implicit_raw_block_replays_source_and_never_materializes_a_cubic_core():
    oracle = _oracle(2)
    raw = _inputs(3, 2)
    actual = evaluate_boundary_quotient(oracle.network, raw)
    expected = source_block_boundary(oracle, raw)
    assert _relative(actual, expected) < 2e-10
    telemetry = telemetry_dict(oracle.telemetry)
    assert telemetry["raw_order_three_core_materializations"] == 0
    assert telemetry["raw_width_cubic_core_materializations"] == 0
    assert telemetry["fused_ffn_primitive_count"] == 1
    assert telemetry["ephemeral_ffn_bond_diagonalizations"] == 0
    shape = implicit_shape_statistics(oracle.network)
    assert shape["maximum_local_bond_dimension"] < shape["fused_token_feature_dimension"]


def test_every_algorithm1_node_uses_only_direct_reduced_rq_and_replays_each_step():
    oracle = _oracle(3)
    raw = _inputs(3, 2)
    canonical = canonicalize_implicit_dag_direct_rq(
        oracle.network, replay_inputs=raw, block_size=64
    )
    assert canonical.steps
    assert {step.method for step in canonical.steps} <= {
        BOUNDED_EXPLICIT_RQ_METHOD,
        DIRECT_RQ_METHOD,
        UNARY_RQ_METHOD,
    }
    assert max(step.factorization_relative_error for step in canonical.steps) < 2e-10
    assert max(step.per_step_function_replay_error for step in canonical.steps) < 3e-8
    assert all(
        step.parent_occurrences_pushed == step.expected_parent_occurrences
        for step in canonical.steps
    )
    # V2 does not numerically test rank. It retains every shape-selected Q row.
    assert all(step.literal_q_chart_resolved for step in canonical.steps)
    assert all(not step.full_row_rank for step in canonical.steps)  # unassessed legacy field
    audit = audit_algorithm1_factorization_calls()
    assert audit["prohibited_calls_found"] == []
    telemetry = telemetry_dict(canonical.telemetry)
    assert telemetry["svd_calls"] == telemetry["polar_calls"] == 0
    assert telemetry["normal_equation_factorizations"] == 0


def test_streamed_shared_algorithm1_matches_independent_dense_no_memo_clone_every_step():
    oracle = _oracle(5, tokens=2, dimension=1, rank=1)
    raw = _inputs(2, 1)
    dense = tiny_dense_direct_rq_equivalence(oracle.network, raw, block_size=32)
    assert dense.steps > 20
    assert dense.maximum_factor_relative_error < 2e-9
    assert dense.maximum_q_core_relative_error < 2e-8
    assert dense.maximum_reconstruction_relative_error < 2e-10
    assert dense.maximum_partial_replay_relative_error < 3e-8

    trace = canonicalize_with_explicit_clone_step_trace(
        oracle.network, raw, block_size=32
    )
    assert len(trace.records) == len(trace.shared_steps)
    assert max(record.factor_relative_error for record in trace.records) < 2e-9
    assert max(record.q_core_relative_error for record in trace.records) < 2e-8
    assert max(record.transformed_parent_occurrence_relative_error for record in trace.records) < 2e-8
    assert max(record.supported_reconstruction_relative_error for record in trace.records) < 2e-10
    assert max(record.shared_function_replay_error for record in trace.records) < 3e-8
    assert max(record.clone_function_replay_error for record in trace.records) < 3e-8
    assert max(record.shared_clone_function_relative_error for record in trace.records) < 3e-8
    assert max(record.boundary_head_relative_error for record in trace.records) < 2e-8
    assert all(
        record.expected_shared_parent_occurrences == record.pushed_shared_parent_occurrences
        and record.expected_clone_parent_occurrences == record.pushed_clone_parent_occurrences
        for record in trace.records
    )
    assert all(method == "independent_symmetric_clone_qr_v2" for method in trace.clone_factorization_methods)
    assert all(record.literal_rq_occurrences_compared == record.clone_occurrences for record in trace.records)

    comparison = compare_shared_and_explicit_clone_environments(
        trace.shared_network, trace.explicit_clone_network
    )
    assert comparison["maximum_aggregate_relative_error"] < 3e-8
    assert comparison["explicit_clone_environment_count"] > comparison["shared_unique_environment_count"]


def test_extreme_binary_exponent_ledger_does_not_materialize_physical_scale_before_clone_qr():
    oracle = _oracle(7, tokens=2, dimension=1, rank=1)
    oracle.network.root.core.binary_exponent += 2200
    trace = canonicalize_with_explicit_clone_step_trace(
        oracle.network, _inputs(2, 1), block_size=32
    )
    assert max(record.shared_clone_function_relative_error for record in trace.records) < 3e-8
    assert max(record.supported_reconstruction_relative_error for record in trace.records) < 2e-10


def test_algorithm2_then_full_rank_algorithm3_replay_and_recontract_diagonal():
    oracle = _oracle(11)
    raw = _inputs(3, 2)
    canonical = canonicalize_implicit_dag_direct_rq(
        oracle.network, replay_inputs=raw, block_size=64
    )
    environments = reverse_implicit_environments(canonical.network)
    shape = implicit_shape_statistics(canonical.network)
    assert len(environments) == shape["unique_nodes"]
    assert sum(record.child_messages_emitted for record in environments) == shape[
        "edge_occurrences"
    ]
    diagonal = diagonalize_implicit_dag_full_rank(
        canonical.network, replay_inputs=raw
    )
    assert diagonal.replay_relative_error < 3e-8
    assert diagonal.maximum_recontracted_offdiagonal_ratio < 3e-8
    assert diagonal.pushed_parent_occurrences == diagonal.expected_parent_occurrences
    assert diagonal.pushed_parent_occurrences == shape["edge_occurrences"] + 1
    assert _relative(
        evaluate_boundary_quotient(diagonal.network, raw),
        source_block_boundary(oracle, raw),
    ) < 3e-8


def test_clone_algorithm2_and_independently_derived_algorithm3_match_shared_object():
    oracle = _oracle(19, tokens=2, dimension=1, rank=1)
    raw = _inputs(2, 1)
    trace = canonicalize_with_explicit_clone_step_trace(
        oracle.network, raw, block_size=32
    )
    result = diagonalize_shared_and_explicit_clone_independently(
        trace.shared_network, trace.explicit_clone_network, raw
    )
    assert result.pre_evd_aggregate_environment_relative_error < 3e-8
    assert result.eigenvalue_relative_error < 3e-8
    assert result.shared_replay_relative_error < 3e-8
    assert result.clone_replay_relative_error < 3e-8
    assert result.shared_clone_relative_error < 3e-8
    assert result.maximum_shared_aggregate_offdiagonal_ratio < 3e-8
    assert result.maximum_clone_aggregate_offdiagonal_ratio < 3e-8
    assert result.expected_shared_parent_occurrences == result.pushed_shared_parent_occurrences
    assert result.expected_clone_parent_occurrences == result.pushed_clone_parent_occurrences
    assert result.omitted_clone_occurrences == 0

    broken = diagonalize_shared_and_explicit_clone_independently(
        trace.shared_network,
        trace.explicit_clone_network,
        raw,
        omit_clone_parent=("pre_attention.token0.projective_output", 0),
    )
    assert broken.omitted_clone_occurrences == 1
    assert broken.targeted_clone_parent_occurrences > 1
    assert broken.pushed_clone_parent_occurrences == broken.expected_clone_parent_occurrences - 1
    assert broken.clone_replay_relative_error > 1e-6


def test_missing_R_and_residual_cross_term_are_load_bearing(monkeypatch):
    oracle = _oracle(13)
    raw = _inputs(3, 2)
    reference = evaluate_boundary_quotient(oracle.network, raw)
    from xvla.train.odt_engine_v2 import core as engine
    push = engine._push_factor_to_parents_with_scale_ledger
    injected = []
    def missing_occurrence(node, factor, parents, work):
        if not injected and len(parents.get(id(node), ())) > 1:
            injected.append(node.uid)
            parents = {**parents, id(node): parents[id(node)][1:]}
        return push(node, factor, parents, work)
    with monkeypatch.context() as patcher:
        patcher.setattr(engine, "_push_factor_to_parents_with_scale_ledger", missing_occurrence)
        with pytest.raises(RuntimeError, match="every occurrence"):
            canonicalize_implicit_dag_direct_rq(oracle.network, replay_inputs=raw, block_size=64)
    assert len(injected) == 1
    broken_residual = drop_first_residual_add_cross_term(oracle.network)
    assert _relative(evaluate_boundary_quotient(broken_residual, raw), reference) > 1e-6


def test_old_project_then_add_compact_chart_rejects_but_public_route_is_exact():
    with pytest.raises(ValueError, match="retired solver-dependent"):
        old_project_then_add_topology_is_rejected(12, 3, like=torch.empty((), dtype=DTYPE))
    network, raw = _bounded_old_project_then_add_network()
    reference = evaluate_boundary_quotient(network, raw)
    canonical = canonicalize_implicit_dag_direct_rq(
        network, replay_inputs=raw, block_size=64
    )
    target = next(
        step
        for step in canonical.steps
        if step.label == "old_project_then_add.bounded_deficient"
    )
    assert target.method == BOUNDED_EXPLICIT_RQ_METHOD
    assert not target.full_row_rank
    assert target.minimum_diagonal_to_maximum_entry == 0.0
    assert target.parent_occurrences_pushed == target.expected_parent_occurrences
    assert canonical.final_projective_replay_error < 3e-8
    assert _relative(
        evaluate_boundary_quotient(canonical.network, raw), reference
    ) < 3e-8


def test_algorithm2_rejects_a_noncanonical_raw_graph():
    oracle = _oracle(17)
    with pytest.raises(ValueError, match="completed direct-RQ Algorithm 1"):
        reverse_implicit_environments(oracle.network)


def test_algorithm2_direct_only_path_rejects_a_nonidentity_output_metric():
    oracle = _oracle(18)
    raw = _inputs(3, 2)
    canonical = canonicalize_implicit_dag_direct_rq(
        oracle.network, replay_inputs=raw, block_size=64
    )
    metric = torch.eye(canonical.network.head.shape[0], dtype=DTYPE)
    metric[0, 0] = 2.0
    with pytest.raises(TypeError, match="output_metric"):
        reverse_implicit_environments(canonical.network, output_metric=metric)


def test_native_multihead_concat_is_exact_with_shape_selected_q():
    oracle = _oracle(23, tokens=2, dimension=4, heads=2, rank=5)
    raw = _inputs(2, 4)
    source = source_block_boundary(oracle, raw)
    compiled = evaluate_boundary_quotient(oracle.network, raw)
    assert _relative(compiled, source) < 2e-10
    canonical = canonicalize_implicit_dag_direct_rq(
        oracle.network, replay_inputs=raw, block_size=64
    )
    assert all(step.literal_q_chart_resolved for step in canonical.steps)
    assert all(not step.full_row_rank for step in canonical.steps)
    assert max(step.per_step_function_replay_error for step in canonical.steps) < 3e-8


def test_solver_dependent_tsqr_control_is_retired():
    with pytest.raises(ValueError, match="retired TSQR solve route"):
        multiblock_tsqr_dense_equivalence()


def test_validator_rejects_nonbinary_mask_and_malformed_cp_factor():
    oracle = _oracle(29)
    raw = _inputs(3, 2)
    oracle.network.mask[0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        evaluate_boundary_quotient(oracle.network, raw)

    oracle = _oracle(30)
    oracle.network.root.core.left_factor = oracle.network.root.core.left_factor.unsqueeze(0)
    with pytest.raises(ValueError, match="CP factors must be matrices"):
        evaluate_boundary_quotient(oracle.network, raw)
