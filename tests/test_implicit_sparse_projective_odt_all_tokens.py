import torch
import torch.nn as nn

from xvla.nn.block import ChiTransformerBlock
from xvla.nn.normalization import RationalNorm
from xvla.train.implicit_sparse_projective_odt import (
    canonicalize_implicit_dag_direct_rq,
    canonicalize_with_explicit_clone_step_trace,
    compare_shared_and_explicit_clone_environments,
    diagonalize_shared_and_explicit_clone_independently,
    diagonalize_implicit_dag_full_rank,
    evaluate_boundary_quotient,
    implicit_shape_statistics,
    reverse_implicit_environments,
    telemetry_dict,
)
from xvla.train.implicit_sparse_projective_odt_all_tokens import (
    all_token_structure_statistics,
    assert_all_token_norm_buffers_unchanged,
    compile_all_token_projective_block,
    compile_all_token_projective_stack,
    evaluate_all_token_quotients,
    evaluate_observable_concat_quotient,
    source_all_token_output,
)


DTYPE = torch.float64


def _block(seed: int, dimension: int, heads: int, rank: int, causal: bool):
    torch.manual_seed(seed)
    block = ChiTransformerBlock(
        dim=dimension,
        n_heads=heads,
        ffn_rank=rank,
        n_layers=2,
        causal=causal,
        norm="rational",
        qk_norm="rational",
        residual=True,
    ).double().eval()
    generator = torch.Generator().manual_seed(seed + 7000)
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                module.weight.copy_(
                    0.09
                    + 0.13
                    * torch.rand(module.weight.shape, generator=generator, dtype=DTYPE)
                )
                if module.bias is not None:
                    module.bias.copy_(
                        -0.04
                        + 0.08
                        * torch.rand(module.bias.shape, generator=generator, dtype=DTYPE)
                    )
        for index, module in enumerate(
            item for item in block.modules() if isinstance(item, RationalNorm)
        ):
            module.running_ms.fill_(0.83 + 0.17 * index)
            module.initialized.fill_(True)
            module.frozen = True
    return block


def _positions(tokens: int, dimension: int) -> torch.Tensor:
    return torch.stack(
        [
            torch.linspace(
                -0.03 + token * 0.011,
                0.05 + token * 0.017,
                dimension,
                dtype=DTYPE,
            )
            for token in range(tokens)
        ]
    )


def _inputs(tokens: int, dimension: int) -> torch.Tensor:
    value = torch.linspace(-0.27, 0.35, 3 * tokens * dimension, dtype=DTYPE)
    result = value.reshape(3, tokens, dimension)
    result[1].copy_(result[1].flip(0))
    result[2].mul_(-0.7)
    return result


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = expected.norm().clamp_min(torch.finfo(expected.dtype).tiny)
    return float(((actual - expected).norm() / scale).item())


def test_all_token_custom_mask_replays_every_output_and_visibility_is_exact():
    block = _block(1, 2, 1, 3, causal=False)
    mask = torch.tensor(
        ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0)),
        dtype=DTYPE,
    )
    oracle = compile_all_token_projective_block(block, _positions(3, 2), mask=mask)
    raw = _inputs(3, 2)
    compiled = evaluate_all_token_quotients(oracle, raw)
    source = source_all_token_output(oracle, raw)
    assert compiled.shape == source.shape == (3, 3, 2)
    assert _relative(compiled, source) < 2e-10

    base = compiled[:1]
    for target in range(3):
        for source_token in range(3):
            if target == source_token:
                continue
            perturbed = raw[:1].clone()
            perturbed[:, source_token].add_(0.19)
            changed = evaluate_all_token_quotients(oracle, perturbed)
            delta = float((changed[:, target] - base[:, target]).norm().item())
            if mask[target, source_token] == 0:
                assert delta < 2e-12
            else:
                assert delta > 1e-8

    stats = all_token_structure_statistics(oracle)
    assert stats["unique_physical_leaves"] == 3
    assert stats["unfolded_token_root_node_sum"] > stats["unique_nodes"]
    counts = stats["label_parent_occurrences"]
    assert counts["block0.attention.head0.k1_norm.token0.projective_output"] == 2
    assert counts["block0.attention.head0.q1_norm.token2.projective_output"] == 2
    # Three Padé pre-norm roles plus the unchanged attention residual.
    assert counts["input.position.token0"] == 4
    assert_all_token_norm_buffers_unchanged(oracle)


def test_noncausal_multihead_all_token_tuple_and_observable_concat_are_exact():
    block = _block(3, 4, 2, 5, causal=False)
    oracle = compile_all_token_projective_block(
        block,
        _positions(4, 4),
        include_observable_boundary=True,
    )
    raw = _inputs(4, 4)
    token_value = evaluate_all_token_quotients(oracle, raw)
    observable_value = evaluate_observable_concat_quotient(oracle, raw)
    source = source_all_token_output(oracle, raw)
    assert _relative(token_value, source) < 2e-10
    assert _relative(observable_value, source) < 2e-10
    assert _relative(observable_value, token_value) < 2e-12
    stats = all_token_structure_statistics(oracle)
    counts = stats["label_parent_occurrences"]
    assert counts["block0.attention.head0.k1_norm.token0.projective_output"] == 4
    assert counts["block0.attention.head1.k2_norm.token3.projective_output"] == 4
    meter = telemetry_dict(oracle.telemetry)
    assert meter["raw_order_three_core_materializations"] == 0
    assert meter["raw_width_cubic_core_materializations"] == 0
    assert meter["fused_ffn_primitive_count"] == 4
    assert meter["ephemeral_ffn_bond_diagonalizations"] == 0


def test_tiny_all_token_observable_runs_algorithms1_to3_against_independent_clone():
    block = _block(5, 1, 1, 1, causal=True)
    oracle = compile_all_token_projective_block(
        block,
        _positions(2, 1),
        include_observable_boundary=True,
    )
    assert oracle.observable_network is not None
    raw = _inputs(2, 1)
    reference = evaluate_boundary_quotient(oracle.observable_network, raw)
    trace = canonicalize_with_explicit_clone_step_trace(
        oracle.observable_network,
        raw,
        block_size=32,
    )
    assert all(
        record.expected_shared_parent_occurrences
        == record.pushed_shared_parent_occurrences
        and record.expected_clone_parent_occurrences
        == record.pushed_clone_parent_occurrences
        for record in trace.records
    )
    assert max(record.shared_clone_function_relative_error for record in trace.records) < 3e-8
    comparison = compare_shared_and_explicit_clone_environments(
        trace.shared_network,
        trace.explicit_clone_network,
    )
    assert comparison["maximum_aggregate_relative_error"] < 3e-8
    diagonal = diagonalize_shared_and_explicit_clone_independently(
        trace.shared_network,
        trace.explicit_clone_network,
        raw,
    )
    assert diagonal.pre_evd_aggregate_environment_relative_error < 3e-8
    assert diagonal.eigenvalue_relative_error < 3e-8
    assert max(
        diagonal.shared_replay_relative_error,
        diagonal.clone_replay_relative_error,
        diagonal.shared_clone_relative_error,
        diagonal.maximum_shared_aggregate_offdiagonal_ratio,
        diagonal.maximum_clone_aggregate_offdiagonal_ratio,
    ) < 3e-8
    assert diagonal.pushed_shared_parent_occurrences == diagonal.expected_shared_parent_occurrences
    assert diagonal.pushed_clone_parent_occurrences == diagonal.expected_clone_parent_occurrences

    repeated_label = "block0.attention.head0.k1_norm.token0.projective_output"
    repeated_step = next(step for step in trace.shared_steps if step.label == repeated_label)
    assert repeated_step.expected_parent_occurrences > 1
    broken_r = canonicalize_implicit_dag_direct_rq(
        oracle.observable_network,
        replay_inputs=raw,
        block_size=32,
        omit_parent_push=(repeated_label, 0),
    )
    assert _relative(evaluate_boundary_quotient(broken_r.network, raw), reference) > 1e-6
    broken_evd = diagonalize_shared_and_explicit_clone_independently(
        trace.shared_network,
        trace.explicit_clone_network,
        raw,
        omit_clone_parent=(repeated_label, 0),
    )
    assert broken_evd.targeted_clone_parent_occurrences > 1
    assert broken_evd.omitted_clone_occurrences == 1
    assert broken_evd.clone_replay_relative_error > 1e-6


def test_two_block_token_tuple_adds_positions_once_and_replays_exactly():
    blocks = (
        _block(7, 1, 1, 1, causal=True),
        _block(8, 1, 1, 1, causal=True),
    )
    positions = _positions(2, 1)
    oracle = compile_all_token_projective_stack(blocks, positions)
    raw = _inputs(2, 1)
    compiled = evaluate_all_token_quotients(oracle, raw)
    source = source_all_token_output(oracle, raw)
    assert _relative(compiled, source) < 3e-9
    manual = raw + positions
    for block in blocks:
        manual = block(manual, mask=oracle.mask, method="explicit")
    assert _relative(compiled, manual) < 3e-9
    counts = all_token_structure_statistics(oracle)["label_parent_occurrences"]
    assert sum(label.startswith("input.position.token") for label in counts) == 2
    assert not any("block1.position" in label for label in counts)

    final_network = oracle.token_networks[-1]
    canonical = canonicalize_implicit_dag_direct_rq(
        final_network,
        replay_inputs=raw,
        block_size=32,
    )
    assert all(step.full_row_rank for step in canonical.steps)
    assert all(
        step.parent_occurrences_pushed == step.expected_parent_occurrences
        for step in canonical.steps
    )
    assert _relative(
        evaluate_boundary_quotient(canonical.network, raw), source[:, -1]
    ) < 3e-8
    environments = reverse_implicit_environments(canonical.network)
    shape = implicit_shape_statistics(canonical.network)
    assert len(environments) == shape["unique_nodes"]
    assert sum(record.child_messages_emitted for record in environments) == shape[
        "edge_occurrences"
    ]
    diagonal = diagonalize_implicit_dag_full_rank(
        canonical.network,
        replay_inputs=raw,
    )
    assert diagonal.replay_relative_error < 3e-8
    assert diagonal.maximum_recontracted_offdiagonal_ratio < 3e-8
    assert diagonal.pushed_parent_occurrences == diagonal.expected_parent_occurrences
    assert diagonal.pushed_parent_occurrences == shape["edge_occurrences"] + 1


def test_wide_observable_concat_fails_closed_but_local_token_tuple_compiles():
    block = _block(11, 4, 2, 5, causal=False)
    positions = _positions(4, 4)
    oracle = compile_all_token_projective_block(block, positions)
    assert oracle.observable_network is None
    try:
        compile_all_token_projective_block(
            block,
            positions,
            include_observable_boundary=True,
            maximum_materialized_observable_width=8,
        )
    except ValueError as error:
        assert "keep the token tuple local" in str(error)
    else:
        raise AssertionError("wide observable concat did not fail closed")


def test_selected_final_outputs_keep_all_source_tokens_and_requested_order():
    block = _block(13, 2, 1, 3, causal=False)
    oracle = compile_all_token_projective_block(
        block,
        _positions(4, 2),
        output_tokens=(3, 1),
    )
    raw = _inputs(4, 2)
    assert oracle.output_tokens == (3, 1)
    assert evaluate_all_token_quotients(oracle, raw).shape == (3, 2, 2)
    assert _relative(
        evaluate_all_token_quotients(oracle, raw),
        source_all_token_output(oracle, raw),
    ) < 2e-10
    stats = all_token_structure_statistics(oracle)
    assert stats["unique_physical_leaves"] == 4
