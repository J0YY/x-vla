"""Exact and adversarial tests for the global Padé projective tree oracle."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from xvla.nn.block import ChiTransformerBlock
from xvla.nn.normalization import RationalNorm
from xvla.train.global_projective_odt_oracle import (
    CLAIM_BOUNDARY,
    FIXED_GAUGE,
    ProjectivePolynomialTree,
    UNFIXED_GAUGE,
    add_scaled_symmetric_messages,
    assert_norm_buffers_unchanged,
    brute_identity_role_environment,
    canonical_projective_environments,
    canonicalize_projective_tree,
    compile_global_projective_block,
    diagonalize_projective_tree_full_rank,
    drop_first_projective_add_cross_term,
    evaluate_projective_pair,
    evaluate_projective_pair_unscaled_negative_control,
    evaluate_quotient,
    evaluate_scaled_projective_pair,
    fix_projective_gauge,
    gauge_tree_occurrence,
    node_at_path,
    omit_quotient_cross_terms_negative_control,
    projective_pair_coefficient_norm,
    quotient_pair_output_metric,
    rescale_projective_output,
    scale_symmetric_message,
    scaled_projective_relative_error,
    scaled_projective_statistics,
    source_action,
)


DTYPE = torch.float64


def _oracle(seed: int = 0, *, rank: int = 2):
    torch.manual_seed(seed)
    block = ChiTransformerBlock(
        dim=1,
        n_heads=1,
        ffn_rank=rank,
        n_layers=1,
        causal=True,
        norm="rational",
        qk_norm="rational",
        residual=True,
    ).double().eval()
    action_head = nn.Linear(1, 1, bias=True).double().eval()
    generator = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                module.weight.copy_(
                    0.08
                    + 0.16
                    * torch.rand(
                        module.weight.shape,
                        generator=generator,
                        dtype=DTYPE,
                    )
                )
                if module.bias is not None:
                    module.bias.copy_(
                        -0.12
                        + 0.24
                        * torch.rand(
                            module.bias.shape,
                            generator=generator,
                            dtype=DTYPE,
                        )
                    )
                    module.bias.add_(
                        torch.where(module.bias >= 0, 0.04, -0.04)
                    )
        action_head.weight.fill_(0.31)
        action_head.bias.fill_(-0.17)
        for index, norm in enumerate(
            module for module in block.modules() if isinstance(module, RationalNorm)
        ):
            norm.running_ms.fill_(1.05 + 0.07 * index)
            norm.initialized.fill_(True)
            norm.frozen = True
    return compile_global_projective_block(block, action_head)


def _inputs() -> torch.Tensor:
    return torch.tensor(
        [
            [-0.35, 0.20],
            [0.10, -0.25],
            [0.30, 0.40],
            [-0.15, -0.10],
            [0.0, 0.0],
        ],
        dtype=DTYPE,
    )


def test_one_global_pair_tree_replays_unchanged_rational_residual_block():
    oracle = _oracle(1)
    raw = _inputs()
    actual = evaluate_quotient(oracle.network, raw)
    expected = source_action(oracle, raw)
    torch.testing.assert_close(actual, expected, rtol=3e-10, atol=3e-11)
    pair = evaluate_scaled_projective_pair(oracle.network, raw)
    stats = scaled_projective_statistics(pair)
    assert stats["minimum_relative_denominator"] > 1e-8
    assert float(
        evaluate_scaled_projective_pair(
            oracle.network, raw.new_zeros(1, 2)
        ).mantissa[0, 1]
    ) > 0
    torch.testing.assert_close(
        projective_pair_coefficient_norm(oracle.network),
        raw.new_tensor(1.0),
        rtol=0,
        atol=5e-12,
    )
    assert oracle.network.gauge == FIXED_GAUGE
    assert "no compression claim" in CLAIM_BOUNDARY
    assert len(oracle.norm_buffer_snapshot) == 24
    assert_norm_buffers_unchanged(oracle)


def test_binary_exponent_ledger_reproduces_and_repairs_r3_underflow():
    oracle = _oracle(1)
    zero = _inputs().new_zeros(1, 2)
    scaled = evaluate_scaled_projective_pair(oracle.network, zero)
    assert torch.isfinite(scaled.mantissa).all()
    assert float(scaled.mantissa[0, 1].abs().item()) > 1e-6
    naive = evaluate_projective_pair_unscaled_negative_control(oracle.network, zero)
    assert float(naive[0, 1].item()) == 0.0
    with pytest.raises(OverflowError, match="common scale is outside floating range"):
        evaluate_projective_pair(oracle.network, zero)
    torch.testing.assert_close(
        evaluate_quotient(oracle.network, zero),
        source_action(oracle, zero),
        rtol=3e-10,
        atol=3e-11,
    )


def test_global_bottom_up_rq_and_full_rank_evd_absorption_are_exact():
    oracle = _oracle(2, rank=2)
    raw = _inputs()
    reference_pair = evaluate_scaled_projective_pair(oracle.network, raw)
    canonical = canonicalize_projective_tree(oracle.network)
    canonical_pair = evaluate_scaled_projective_pair(canonical.network, raw)
    assert scaled_projective_relative_error(canonical_pair, reference_pair) < 3e-9
    assert max(canonical.isometry_errors) < 2e-10
    assert max(canonical.factorization_errors) < 2e-10
    assert canonical.factorization_methods
    assert set(canonical.factorization_methods) == {"direct_reduced_rq"}
    assert canonical.occurrence_count > canonical.unique_node_count

    environments = canonical_projective_environments(canonical.network)
    assert len(environments) == canonical.occurrence_count
    assert len({record.path for record in environments}) == len(environments)
    assert all(torch.isfinite(record.gram).all() for record in environments)
    root = node_at_path(canonical.network, ())
    root_record = next(record for record in environments if record.path == ())
    brute_unscaled = brute_identity_role_environment(root.core, root_record.gram, 0)
    expected_gram, expected_exponent = scale_symmetric_message(
        brute_unscaled, root_record.binary_exponent
    )
    actual_first_role = next(record for record in environments if record.path == (0,))
    assert actual_first_role.binary_exponent == expected_exponent
    torch.testing.assert_close(
        actual_first_role.gram, expected_gram, rtol=2e-12, atol=2e-12
    )
    diagonal = diagonalize_projective_tree_full_rank(canonical.network)
    diagonal_pair = evaluate_scaled_projective_pair(diagonal.network, raw)
    assert scaled_projective_relative_error(diagonal_pair, reference_pair) < 2e-8
    assert max(diagonal.diagonalization_errors) < 2e-10
    assert len(diagonal.eigenvalue_binary_exponents) == len(environments)
    torch.testing.assert_close(
        evaluate_quotient(diagonal.network, raw),
        source_action(oracle, raw),
        rtol=2e-8,
        atol=2e-9,
    )


def test_scaled_metric_messages_align_exponents_before_a_tied_occurrence_sum():
    first = torch.tensor(((0.75, 0.125), (0.125, 0.5)), dtype=DTYPE)
    second = torch.tensor(((0.25, -0.0625), (-0.0625, 0.375)), dtype=DTYPE)
    mantissa, exponent = add_scaled_symmetric_messages(((first, 31), (second, -17)))
    actual = torch.ldexp(mantissa, torch.tensor(exponent, dtype=torch.int64))
    expected = torch.ldexp(first, torch.tensor(31, dtype=torch.int64)) + torch.ldexp(
        second, torch.tensor(-17, dtype=torch.int64)
    )
    torch.testing.assert_close(actual, expected, rtol=2e-15, atol=0)


def test_fixed_syntactic_gauge_removes_constant_projective_rescaling():
    oracle = _oracle(3)
    raw = _inputs()
    scaled = rescale_projective_output(oracle.network, -7.25)
    assert scaled.gauge == UNFIXED_GAUGE
    with pytest.raises(ValueError, match="fixed projective gauge"):
        canonicalize_projective_tree(scaled)
    forged = ProjectivePolynomialTree(
        scaled.root,
        scaled.head,
        scaled.raw_input_dimension,
        FIXED_GAUGE,
    )
    with pytest.raises(ValueError, match="coefficient norm is not one"):
        canonicalize_projective_tree(forged)
    torch.testing.assert_close(
        evaluate_quotient(scaled, raw),
        evaluate_quotient(oracle.network, raw),
        rtol=2e-12,
        atol=2e-12,
    )
    refixed = fix_projective_gauge(scaled)
    assert scaled_projective_relative_error(
        evaluate_scaled_projective_pair(refixed, raw),
        evaluate_scaled_projective_pair(oracle.network, raw),
    ) < 3e-12


def test_hidden_nonorthogonal_occurrence_gauge_replays_and_recanonicalizes():
    oracle = _oracle(31, rank=1)
    raw = _inputs()
    reference = evaluate_scaled_projective_pair(oracle.network, raw)
    # This after-attention subtree is also reused inside the FFN branch.  Gauge
    # only its direct residual occurrence to test path-specific clone semantics.
    path = (0, 0)
    target = node_at_path(oracle.network, path)
    assert target.output_dimension == 2
    transform = torch.tensor(((1.3, 0.25), (-0.15, 0.8)), dtype=DTYPE)
    gauged = gauge_tree_occurrence(oracle.network, path, transform)
    assert scaled_projective_relative_error(
        evaluate_scaled_projective_pair(gauged, raw), reference
    ) < 2e-11
    gauged_canonical = canonicalize_projective_tree(gauged)
    assert scaled_projective_relative_error(
        evaluate_scaled_projective_pair(gauged_canonical.network, raw), reference
    ) < 5e-9


def test_quotient_cross_terms_are_required_by_the_projective_radial_null():
    oracle = _oracle(4)
    pair = evaluate_scaled_projective_pair(
        oracle.network, _inputs()[2:3]
    ).mantissa[0]
    metric = quotient_pair_output_metric(pair)
    radial_scale = torch.linalg.vector_norm(metric) * torch.linalg.vector_norm(pair)
    correct_residual = torch.linalg.vector_norm(metric @ pair) / radial_scale
    assert float(correct_residual.item()) < 2e-14

    wrong = omit_quotient_cross_terms_negative_control(metric)
    wrong_residual = torch.linalg.vector_norm(wrong @ pair) / radial_scale
    assert float(metric[0, 1].abs().item()) > 1e-8
    assert float(wrong_residual.item()) > 1e-3


def test_dropping_a_residual_numerator_denominator_cross_product_breaks_replay():
    oracle = _oracle(5)
    raw = _inputs()
    broken = drop_first_projective_add_cross_term(
        oracle.network, label_contains="ffn.residual_add"
    )
    correct = evaluate_quotient(oracle.network, raw)
    incorrect = evaluate_quotient(broken, raw)
    relative = torch.linalg.vector_norm(correct - incorrect) / torch.linalg.vector_norm(correct)
    assert float(relative.item()) > 1e-5
