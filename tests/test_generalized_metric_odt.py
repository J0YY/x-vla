"""Exact-oracle tests for generalized metric ODT and clone estimators."""

from __future__ import annotations

from itertools import product

import pytest
import torch

from xvla.train import generalized_metric_odt as generalized
from xvla.train.generalized_metric_odt import (
    balanced_metric_odt,
    estimate_forward_metric,
    estimate_shared_environment,
    forward_metric,
    role_environment,
    shared_environment,
)


DTYPE = torch.float64


def _spd(seed: int, dimension: int, *, rank: int | None = None) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    columns = dimension if rank is None else rank
    factor = torch.randn(dimension, columns, generator=generator, dtype=DTYPE)
    metric = factor @ factor.T / columns
    if rank is None:
        metric += 0.25 * torch.eye(dimension, dtype=DTYPE)
    return metric


def _core(seed: int, output: int, dimensions: tuple[int, ...]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((output,) + dimensions, generator=generator, dtype=DTYPE)


def _brute_forward(core: torch.Tensor, metrics: tuple[torch.Tensor, ...]) -> torch.Tensor:
    output = core.new_zeros(core.shape[0], core.shape[0])
    ranges = [range(dimension) for dimension in core.shape[1:]]
    for left in product(*ranges):
        for right in product(*ranges):
            weight = core.new_ones(())
            for metric, i, j in zip(metrics, left, right):
                weight *= metric[i, j]
            output += weight * torch.outer(core[(slice(None),) + left], core[(slice(None),) + right])
    return output


def _brute_role(
    core: torch.Tensor,
    downstream: torch.Tensor,
    metrics: tuple[torch.Tensor, ...],
    role: int,
) -> torch.Tensor:
    result = core.new_zeros(core.shape[role + 1], core.shape[role + 1])
    other_roles = [index for index in range(len(metrics)) if index != role]
    other_ranges = [range(core.shape[index + 1]) for index in other_roles]
    for i in range(core.shape[role + 1]):
        for j in range(core.shape[role + 1]):
            value = core.new_zeros(())
            for left_other in product(*other_ranges):
                for right_other in product(*other_ranges):
                    left = [0] * len(metrics)
                    right = [0] * len(metrics)
                    left[role] = i
                    right[role] = j
                    context_weight = core.new_ones(())
                    for offset, other_role in enumerate(other_roles):
                        left[other_role] = left_other[offset]
                        right[other_role] = right_other[offset]
                        context_weight *= metrics[other_role][left_other[offset], right_other[offset]]
                    left_vector = core[(slice(None),) + tuple(left)]
                    right_vector = core[(slice(None),) + tuple(right)]
                    value += context_weight * (left_vector @ downstream @ right_vector)
            result[i, j] = value
    return result


@pytest.mark.parametrize(
    "arity,dimensions",
    [(2, (2, 3)), (3, (2, 2, 3)), (5, (2, 2, 2, 2, 2))],
)
def test_exact_recursions_match_index_oracles(arity: int, dimensions: tuple[int, ...]):
    core = _core(10 + arity, 3, dimensions)
    metrics = tuple(_spd(100 + role, dimension) for role, dimension in enumerate(dimensions))
    downstream = _spd(200 + arity, 3)
    torch.testing.assert_close(
        forward_metric(core, metrics), _brute_forward(core, metrics), rtol=2e-12, atol=2e-12
    )
    for role in range(arity):
        torch.testing.assert_close(
            role_environment(core, downstream, metrics, role),
            _brute_role(core, downstream, metrics, role),
            rtol=2e-12,
            atol=2e-12,
        )


def test_identity_sibling_fast_path_avoids_kronecker_materialization(monkeypatch):
    core = _core(19, 3, (5, 5, 5, 5, 5))
    downstream = _spd(190, 3)
    identity = torch.eye(5, dtype=DTYPE)
    role = 2
    arranged = core.permute(0, role + 1, 1, 2, 4, 5)
    rows = arranged.reshape(3, 5, -1)
    expected = torch.einsum("oia,op,pja->ij", rows, downstream, rows)

    def forbidden(*args, **kwargs):
        raise AssertionError("identity fast path materialized a Kronecker context")

    monkeypatch.setattr(generalized, "_kronecker", forbidden)
    actual = role_environment(core, downstream, (identity,) * 5, role)
    torch.testing.assert_close(actual, 0.5 * (expected + expected.T), rtol=2e-12, atol=2e-12)


def test_identity_fast_path_preserves_unary_role_case(monkeypatch):
    core = _core(191, 3, (4,))
    downstream = _spd(192, 3)

    def forbidden(*args, **kwargs):
        raise AssertionError("unary identity path must not form a Kronecker product")

    monkeypatch.setattr(generalized, "_kronecker", forbidden)
    actual = role_environment(core, downstream, (torch.eye(4, dtype=DTYPE),), 0)
    expected = core.T @ downstream @ core
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)


def test_shared_environment_is_declared_sum_of_typed_roles():
    core = _core(20, 2, (3, 3, 3))
    metrics = (_spd(21, 3),) * 3
    downstream = _spd(22, 2)
    expected = sum(
        (role_environment(core, downstream, metrics, role) for role in (0, 2)),
        torch.zeros(3, 3, dtype=DTYPE),
    )
    torch.testing.assert_close(
        shared_environment(core, downstream, metrics, roles=(0, 2)), expected
    )


def test_shared_environment_compares_sanitized_metrics():
    core = _core(23, 2, (2, 2))
    almost_identity = torch.tensor([[1.0, 1e-15], [-1e-15, 1.0]], dtype=DTYPE)
    expected = shared_environment(
        core,
        torch.eye(2, dtype=DTYPE),
        (torch.eye(2, dtype=DTYPE),) * 2,
    )
    actual = shared_environment(
        core,
        torch.eye(2, dtype=DTYPE),
        (almost_identity, torch.eye(2, dtype=DTYPE)),
    )
    torch.testing.assert_close(actual, expected)


def test_diagonal_ternary_hand_oracle_and_energy_identity():
    core = torch.zeros(2, 2, 2, 2, dtype=DTYPE)
    core[0, 0, 0, 0] = 2.0
    core[1, 1, 1, 1] = 3.0
    upstream = torch.diag(torch.tensor([2.0, 5.0], dtype=DTYPE))
    downstream = torch.diag(torch.tensor([7.0, 11.0], dtype=DTYPE))
    metrics = (upstream,) * 3
    output = forward_metric(core, metrics)
    expected_output = torch.diag(torch.tensor([32.0, 1125.0], dtype=DTYPE))
    expected_role = torch.diag(torch.tensor([112.0, 2475.0], dtype=DTYPE))
    torch.testing.assert_close(output, expected_output)
    parent_energy = torch.trace(downstream @ output)
    for role in range(3):
        environment = role_environment(core, downstream, metrics, role)
        torch.testing.assert_close(environment, expected_role)
        torch.testing.assert_close(torch.trace(environment @ upstream), parent_energy)
    shared = shared_environment(core, downstream, metrics)
    torch.testing.assert_close(shared, 3.0 * expected_role)
    torch.testing.assert_close(torch.trace(shared @ upstream), 3.0 * parent_energy)
    torch.testing.assert_close(
        balanced_metric_odt(upstream, expected_role).eigenvalues,
        torch.tensor([12375.0, 224.0], dtype=DTYPE),
    )


def test_independent_input_and_output_gauges_preserve_recursions():
    core = _core(24, 2, (2, 3, 2))
    metrics = (_spd(25, 2), _spd(26, 3), _spd(27, 2))
    downstream = _spd(28, 2)
    transforms = (
        torch.tensor([[1.4, 0.2], [-0.1, 0.8]], dtype=DTYPE),
        torch.tensor([[1.1, 0.2, 0.0], [0.1, 0.9, -0.2], [0.0, 0.3, 1.3]], dtype=DTYPE),
        torch.tensor([[0.7, -0.2], [0.1, 1.5]], dtype=DTYPE),
    )
    output_transform = torch.tensor([[1.2, 0.3], [-0.2, 0.9]], dtype=DTYPE)
    gauged_core = torch.einsum("uv,vabc->uabc", output_transform, core)
    for role, transform in enumerate(transforms):
        inverse = torch.linalg.inv(transform)
        gauged_core = torch.tensordot(
            gauged_core, inverse, dims=([role + 1], [0])
        ).movedim(-1, role + 1)
    gauged_metrics = tuple(
        transform @ metric @ transform.T
        for transform, metric in zip(transforms, metrics)
    )
    output_inverse = torch.linalg.inv(output_transform)
    gauged_downstream = output_inverse.T @ downstream @ output_inverse
    torch.testing.assert_close(
        forward_metric(gauged_core, gauged_metrics),
        output_transform @ forward_metric(core, metrics) @ output_transform.T,
        rtol=2e-11,
        atol=2e-11,
    )
    for role, transform in enumerate(transforms):
        inverse = torch.linalg.inv(transform)
        expected = inverse.T @ role_environment(core, downstream, metrics, role) @ inverse
        torch.testing.assert_close(
            role_environment(gauged_core, gauged_downstream, gauged_metrics, role),
            expected,
            rtol=2e-11,
            atol=2e-11,
        )


def test_balanced_spectrum_is_invariant_to_invertible_state_gauge():
    dimension = 4
    upstream = _spd(30, dimension)
    downstream = _spd(31, dimension)
    transform = torch.tensor(
        [[1.4, 0.2, -0.1, 0.0], [0.1, 0.9, 0.3, 0.2], [0.0, -0.2, 1.2, 0.1], [0.2, 0.0, 0.1, 0.8]],
        dtype=DTYPE,
    )
    inverse = torch.linalg.inv(transform)
    gauged_upstream = transform @ upstream @ transform.T
    gauged_downstream = inverse.T @ downstream @ inverse
    original = balanced_metric_odt(upstream, downstream)
    gauged = balanced_metric_odt(gauged_upstream, gauged_downstream)
    torch.testing.assert_close(original.eigenvalues, gauged.eigenvalues, rtol=2e-11, atol=2e-11)
    original_basis, _ = torch.linalg.qr(original.state_directions[:, :2])
    mapped_basis, _ = torch.linalg.qr(inverse @ gauged.state_directions[:, :2])
    overlap = torch.linalg.matrix_norm(original_basis.T @ mapped_basis).square() / 2
    torch.testing.assert_close(overlap, torch.ones((), dtype=DTYPE), rtol=2e-11, atol=2e-11)


def test_support_detection_handles_scalar_rescaling_and_explicit_zero_support():
    upstream = torch.tensor([[1.0]], dtype=DTYPE)
    downstream = torch.tensor([[2.0]], dtype=DTYPE)
    original = balanced_metric_odt(upstream, downstream)
    scale = torch.tensor([[1e-8]], dtype=DTYPE)
    inverse = torch.linalg.inv(scale)
    gauged = balanced_metric_odt(
        scale @ upstream @ scale.T,
        inverse.T @ downstream @ inverse,
    )
    assert original.upstream_rank == gauged.upstream_rank == 1
    torch.testing.assert_close(original.eigenvalues, gauged.eigenvalues)
    zero = balanced_metric_odt(torch.zeros(3, 3, dtype=DTYPE), torch.eye(3, dtype=DTYPE))
    assert zero.whitened_vectors.shape == (0, 0)
    assert zero.state_directions.shape == (3, 0)


def test_balanced_odt_restricts_to_rank_deficient_upstream_support():
    upstream = _spd(40, 5, rank=3)
    downstream = _spd(41, 5)
    result = balanced_metric_odt(upstream, downstream)
    assert result.upstream_rank == 3
    assert result.eigenvalues.shape == (3,)
    assert result.state_directions.shape == (5, 3)
    assert bool((result.eigenvalues >= 0).all())


def test_monte_carlo_estimators_approach_exact_metrics():
    core = _core(50, 2, (2, 2, 2)) / 3
    upstream = _spd(51, 2)
    metrics = (upstream,) * 3
    downstream = _spd(55, 2)
    exact_forward = forward_metric(core, metrics)
    exact_backward = shared_environment(core, downstream, metrics)
    generator = torch.Generator().manual_seed(56)
    estimated_forward = estimate_forward_metric(
        core, metrics, samples=32768, generator=generator
    )
    estimated_backward = estimate_shared_environment(
        core, downstream, metrics, samples=32768, generator=generator
    )
    assert torch.linalg.vector_norm(estimated_forward - exact_forward) / torch.linalg.vector_norm(exact_forward) < 0.08
    assert torch.linalg.vector_norm(estimated_backward - exact_backward) / torch.linalg.vector_norm(exact_backward) < 0.08


def test_reusing_forward_clones_estimates_a_different_tied_moment():
    core = torch.eye(2, dtype=DTYPE).unsqueeze(0)
    metrics = (torch.eye(2, dtype=DTYPE),) * 2
    exact = forward_metric(core, metrics)
    independent = estimate_forward_metric(
        core,
        metrics,
        samples=65536,
        generator=torch.Generator().manual_seed(60),
    )
    reused = estimate_forward_metric(
        core,
        metrics,
        samples=65536,
        generator=torch.Generator().manual_seed(61),
        independent_clones=False,
    )
    assert abs(float(independent.item()) - 2.0) < 0.1
    assert abs(float(reused.item()) - 8.0) < 0.3
    assert float((reused - exact).abs().item()) > 5.0


def test_monte_carlo_preserves_accepted_tiny_positive_metric_support():
    core = torch.tensor([[0.0, 1.0]], dtype=DTYPE)
    metric = torch.diag(torch.tensor([1.0, 1e-15], dtype=DTYPE))
    exact = forward_metric(core, (metric,))
    estimate = estimate_forward_metric(
        core,
        (metric,),
        samples=65536,
        batch_size=4096,
        generator=torch.Generator().manual_seed(64),
    )
    assert float(exact.item()) == pytest.approx(1e-15)
    assert float(estimate.item()) > 0
    assert abs(float(estimate.item() / exact.item()) - 1.0) < 0.03


def test_scalar_ternary_backward_clone_reuse_targets_nine_not_three():
    core = torch.ones(1, 1, 1, 1, dtype=DTYPE)
    metrics = (torch.ones(1, 1, dtype=DTYPE),) * 3
    downstream = torch.ones(1, 1, dtype=DTYPE)
    exact = shared_environment(core, downstream, metrics)
    independent = estimate_shared_environment(
        core,
        downstream,
        metrics,
        samples=65536,
        batch_size=4096,
        generator=torch.Generator().manual_seed(68),
    )
    reused = estimate_shared_environment(
        core,
        downstream,
        metrics,
        samples=65536,
        batch_size=4096,
        generator=torch.Generator().manual_seed(69),
        independent_contexts=False,
    )
    assert float(exact.item()) == pytest.approx(3.0)
    assert abs(float(independent.item()) - 3.0) < 0.12
    assert abs(float(reused.item()) - 9.0) < 0.5


def test_chunked_and_unchunked_estimators_target_the_same_oracle():
    core = _core(65, 2, (2, 2, 2))
    metrics = (torch.eye(2, dtype=DTYPE),) * 3
    downstream = torch.eye(2, dtype=DTYPE)
    full_forward = estimate_forward_metric(
        core,
        metrics,
        samples=1024,
        generator=torch.Generator().manual_seed(66),
    )
    chunked_forward = estimate_forward_metric(
        core,
        metrics,
        samples=1024,
        batch_size=128,
        generator=torch.Generator().manual_seed(66),
    )
    # Clone draws are interleaved differently under batching, so the two
    # finite-sample matrices need not be bit-identical. Both target one oracle.
    exact_forward = forward_metric(core, metrics)
    assert torch.linalg.vector_norm(full_forward - exact_forward) / torch.linalg.vector_norm(exact_forward) < 0.25
    assert torch.linalg.vector_norm(chunked_forward - exact_forward) / torch.linalg.vector_norm(exact_forward) < 0.25
    chunked_backward = estimate_shared_environment(
        core,
        downstream,
        metrics,
        samples=4096,
        batch_size=256,
        generator=torch.Generator().manual_seed(67),
    )
    exact_backward = shared_environment(core, downstream, metrics)
    assert torch.linalg.vector_norm(chunked_backward - exact_backward) / torch.linalg.vector_norm(exact_backward) < 0.20


def test_invalid_shared_roles_and_non_psd_metrics_are_rejected():
    core = _core(70, 2, (2, 3))
    with pytest.raises(ValueError, match="same dimension"):
        shared_environment(core, torch.eye(2, dtype=DTYPE), (torch.eye(2, dtype=DTYPE), torch.eye(3, dtype=DTYPE)))
    bad = torch.diag(torch.tensor([1.0, -1.0], dtype=DTYPE))
    with pytest.raises(ValueError, match="positive semidefinite"):
        forward_metric(_core(71, 2, (2,)), (bad,))
    with pytest.raises(ValueError, match="roles must index"):
        shared_environment(_core(72, 2, (2, 2)), torch.eye(2, dtype=DTYPE), (torch.eye(2, dtype=DTYPE),) * 2, roles=(False,))
    for roles in ((-1,), (2,)):
        with pytest.raises(ValueError, match="roles must index"):
            shared_environment(_core(73, 2, (2, 2)), torch.eye(2, dtype=DTYPE), (torch.eye(2, dtype=DTYPE),) * 2, roles=roles)
    large_bad = torch.eye(384, dtype=torch.float32)
    large_bad[0, 0] = -1e-4
    with pytest.raises(ValueError, match="positive semidefinite"):
        forward_metric(torch.ones(1, 384, dtype=torch.float32), (large_bad,))


def test_maximum_supported_arity_avoids_reserved_einsum_labels():
    core = torch.ones((1,) + (1,) * 24, dtype=DTYPE)
    metrics = (torch.ones(1, 1, dtype=DTYPE),) * 24
    estimate = estimate_forward_metric(
        core,
        metrics,
        samples=8,
        generator=torch.Generator().manual_seed(80),
    )
    assert estimate.shape == (1, 1)
    assert bool(torch.isfinite(estimate).all())


@pytest.mark.parametrize("support_rtol", [True, float("nan"), float("inf"), -1.0, 1.0])
def test_invalid_support_tolerances_are_rejected(support_rtol: float):
    with pytest.raises(ValueError, match="support_rtol"):
        balanced_metric_odt(
            torch.eye(2, dtype=DTYPE),
            torch.eye(2, dtype=DTYPE),
            support_rtol=support_rtol,
        )
