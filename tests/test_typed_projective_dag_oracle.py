"""Exact and adversarial gates for the typed local-bond projective DAG."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from xvla.nn.block import ChiTransformerBlock
from xvla.nn.normalization import RationalNorm
from xvla.train.typed_projective_dag_oracle import (
    CLAIM_BOUNDARY,
    FIXED_GAUGE,
    TypedProjectiveDAG,
    aggregate_clone_environments,
    assert_typed_norm_buffers_unchanged,
    brute_first_role_scaled_environment,
    canonicalize_typed_projective_dag,
    compile_typed_projective_block,
    diagonalize_typed_projective_dag_full_rank,
    evaluate_scaled_typed_pair,
    evaluate_typed_pair_unscaled_negative_control,
    evaluate_typed_quotient,
    explicit_clone_typed_environments,
    gauge_shared_typed_node,
    maximum_aggregate_environment_offdiagonal_ratio,
    omit_quotient_cross_terms_negative_control,
    quotient_pair_output_metric,
    rescale_typed_projective_output,
    reverse_typed_projective_environments,
    scaled_matrix_relative_error,
    scaled_projective_relative_error,
    scaled_projective_statistics,
    shared_parent_occurrence_count,
    source_typed_action,
    typed_dag_shape_statistics,
    typed_pair_coefficient_norm,
)


DTYPE = torch.float64


def _oracle(seed: int = 0, *, tokens: int = 3, dimension: int = 2, rank: int = 3):
    torch.manual_seed(seed)
    block = ChiTransformerBlock(
        dim=dimension,
        n_heads=1,
        ffn_rank=rank,
        n_layers=1,
        causal=True,
        norm="rational",
        qk_norm="rational",
        residual=True,
    ).double().eval()
    action_head = nn.Linear(dimension, 1, bias=True).double().eval()
    generator = torch.Generator().manual_seed(10_000 + seed)
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                module.weight.copy_(
                    0.04
                    + 0.12
                    * torch.rand(module.weight.shape, generator=generator, dtype=DTYPE)
                )
                if module.bias is not None:
                    module.bias.copy_(
                        -0.08
                        + 0.16
                        * torch.rand(module.bias.shape, generator=generator, dtype=DTYPE)
                    )
                    module.bias.add_(torch.where(module.bias >= 0, 0.025, -0.025))
        action_head.weight.copy_(
            torch.linspace(0.19, 0.31, dimension, dtype=DTYPE).reshape(1, dimension)
        )
        action_head.bias.fill_(-0.13)
        for index, norm in enumerate(
            module for module in block.modules() if isinstance(module, RationalNorm)
        ):
            norm.running_ms.fill_(0.95 + 0.09 * index)
            norm.initialized.fill_(True)
            norm.frozen = True
    positions = torch.stack(
        [
            torch.linspace(
                -0.045 + 0.017 * token,
                0.055 + 0.013 * token,
                dimension,
                dtype=DTYPE,
            )
            for token in range(tokens)
        ]
    )
    return compile_typed_projective_block(block, action_head, positions)


def _inputs(tokens: int = 3, dimension: int = 2) -> torch.Tensor:
    values = torch.linspace(-0.32, 0.41, 4 * tokens * dimension, dtype=DTYPE)
    raw = values.reshape(4, tokens, dimension)
    raw[1].copy_(raw[1].flip(0))
    raw[2].mul_(-0.7)
    raw[3].zero_()
    return raw


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)
    return float((torch.linalg.vector_norm(actual - expected) / scale).item())


def test_real_width_two_three_token_block_replays_with_only_local_bonds():
    oracle = _oracle(1)
    raw = _inputs()
    actual = evaluate_typed_quotient(oracle.network, raw)
    expected = source_typed_action(oracle, raw)
    torch.testing.assert_close(actual, expected, rtol=8e-10, atol=8e-11)
    shape = typed_dag_shape_statistics(oracle.network)
    assert shape["shared_node_count"] > 10
    assert shape["maximum_local_bond_dimension"] <= 4
    assert shape["maximum_local_bond_dimension"] < shape["fused_token_feature_dimension"]
    assert oracle.network.gauge == FIXED_GAUGE
    assert len(oracle.norm_buffer_snapshot) == 24
    assert "one-block construction" in CLAIM_BOUNDARY
    assert_typed_norm_buffers_unchanged(oracle)
    torch.testing.assert_close(
        typed_pair_coefficient_norm(oracle.network),
        raw.new_tensor(1.0),
        rtol=0,
        atol=2e-11,
    )


def test_scaled_typed_messages_repair_naive_global_pair_underflow():
    oracle = _oracle(2)
    zero = _inputs().new_zeros(1, 3, 2)
    scaled = evaluate_scaled_typed_pair(oracle.network, zero)
    assert torch.isfinite(scaled.mantissa).all()
    assert float(scaled.mantissa[0, 1].abs().item()) > 1e-8
    naive = evaluate_typed_pair_unscaled_negative_control(oracle.network, zero)
    assert float(naive[0, 1].item()) == 0.0
    torch.testing.assert_close(
        evaluate_typed_quotient(oracle.network, zero),
        source_typed_action(oracle, zero),
        rtol=8e-10,
        atol=8e-11,
    )
    stats = scaled_projective_statistics(scaled)
    assert stats["minimum_relative_denominator"] > 1e-10


def test_shared_algorithm1_reverse_algorithm2_and_full_rank_algorithm3_are_exact():
    oracle = _oracle(3)
    raw = _inputs()
    reference = evaluate_scaled_typed_pair(oracle.network, raw)
    canonical = canonicalize_typed_projective_dag(oracle.network)
    replay = evaluate_scaled_typed_pair(canonical.network, raw)
    assert scaled_projective_relative_error(replay, reference) < 2e-8
    assert max(canonical.isometry_errors) < 3e-10
    assert max(canonical.factorization_errors) < 3e-10
    assert canonical.factorization_methods
    assert set(canonical.factorization_methods) == {"direct_reduced_rq"}

    environments = reverse_typed_projective_environments(canonical.network)
    assert len(environments) == canonical.unique_node_count
    assert any(record.incoming_occurrences > 2 for record in environments)
    recursive, brute = brute_first_role_scaled_environment(canonical.network)
    assert scaled_matrix_relative_error(recursive, brute) < 2e-12

    diagonal = diagonalize_typed_projective_dag_full_rank(canonical.network)
    diagonal_replay = evaluate_scaled_typed_pair(diagonal.network, raw)
    assert scaled_projective_relative_error(diagonal_replay, reference) < 3e-8
    assert max(diagonal.diagonalization_errors) < 3e-10
    assert maximum_aggregate_environment_offdiagonal_ratio(diagonal.network) < 3e-10
    torch.testing.assert_close(
        evaluate_typed_quotient(diagonal.network, raw),
        source_typed_action(oracle, raw),
        rtol=3e-8,
        atol=3e-9,
    )


def test_shared_environment_rejects_invalid_output_metrics():
    canonical = canonicalize_typed_projective_dag(_oracle(33).network)
    with pytest.raises(ValueError, match="finite"):
        reverse_typed_projective_environments(
            canonical.network, torch.tensor(((1.0, float("nan")), (0.0, 1.0)), dtype=DTYPE)
        )
    with pytest.raises(ValueError, match="symmetric"):
        reverse_typed_projective_environments(
            canonical.network, torch.tensor(((1.0, 0.3), (0.0, 1.0)), dtype=DTYPE)
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        explicit_clone_typed_environments(
            canonical.network, torch.tensor(((1.0, 0.0), (0.0, -0.1)), dtype=DTYPE)
        )


def test_shared_hidden_gauge_requires_every_parent_and_recanonicalizes():
    oracle = _oracle(4)
    raw = _inputs()
    reference = evaluate_scaled_typed_pair(oracle.network, raw)
    label = "position.token2"
    assert shared_parent_occurrence_count(oracle.network, label) >= 4
    transform = torch.tensor(
        ((1.15, 0.12, -0.04), (-0.08, 0.91, 0.09), (0.06, -0.11, 1.07)),
        dtype=DTYPE,
    )
    gauged = gauge_shared_typed_node(oracle.network, label, transform)
    assert scaled_projective_relative_error(
        evaluate_scaled_typed_pair(gauged, raw), reference
    ) < 3e-10
    recanonical = canonicalize_typed_projective_dag(gauged)
    assert scaled_projective_relative_error(
        evaluate_scaled_typed_pair(recanonical.network, raw), reference
    ) < 3e-8

    one_parent_missing = gauge_shared_typed_node(
        oracle.network, label, transform, omit_parent_occurrence=0
    )
    wrong = evaluate_typed_quotient(one_parent_missing, raw)
    correct = evaluate_typed_quotient(oracle.network, raw)
    assert _relative_error(wrong, correct) > 1e-5


def test_absorbing_a_repeated_child_factor_into_only_one_leg_breaks_replay():
    oracle = _oracle(5)
    raw = _inputs()
    broken = canonicalize_typed_projective_dag(
        oracle.network,
        omit_factor_at=("attention.q2.norm.q2.pade_homogeneous", 0),
    )
    wrong = evaluate_typed_quotient(broken.network, raw)
    correct = evaluate_typed_quotient(oracle.network, raw)
    assert _relative_error(wrong, correct) > 1e-5


def test_fixed_gauge_forgery_and_missing_quotient_cross_terms_are_rejected():
    oracle = _oracle(6)
    raw = _inputs()
    rescaled = rescale_typed_projective_output(oracle.network, -6.5)
    with pytest.raises(ValueError, match="explicitly fixed"):
        canonicalize_typed_projective_dag(rescaled)
    forged = TypedProjectiveDAG(
        rescaled.root,
        rescaled.head,
        rescaled.token_count,
        rescaled.feature_dimension,
        FIXED_GAUGE,
    )
    with pytest.raises(ValueError, match="nonunit coefficient norm"):
        canonicalize_typed_projective_dag(forged)

    pair = evaluate_scaled_typed_pair(oracle.network, raw[1:2]).mantissa[0]
    metric = quotient_pair_output_metric(pair)
    scale = torch.linalg.vector_norm(metric) * torch.linalg.vector_norm(pair)
    assert float((torch.linalg.vector_norm(metric @ pair) / scale).item()) < 2e-14
    wrong_metric = omit_quotient_cross_terms_negative_control(metric)
    assert float((torch.linalg.vector_norm(wrong_metric @ pair) / scale).item()) > 1e-4


def test_reverse_shared_environments_equal_explicit_clone_sum_for_n2_d1():
    oracle = _oracle(7, tokens=2, dimension=1, rank=1)
    canonical = canonicalize_typed_projective_dag(oracle.network)
    reverse = reverse_typed_projective_environments(canonical.network)
    clones = explicit_clone_typed_environments(canonical.network)
    assert len(clones) > 20 * len(reverse)
    assert len({record.path for record in clones}) == len(clones)
    aggregate = aggregate_clone_environments(clones)
    reverse_by_identity = {
        record.node_identity: (record.gram, record.binary_exponent)
        for record in reverse
    }
    assert reverse_by_identity.keys() == aggregate.keys()
    maximum = max(
        scaled_matrix_relative_error(reverse_by_identity[identity], expected)
        for identity, expected in aggregate.items()
    )
    assert maximum < 2e-10
