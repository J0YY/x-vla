"""Exactness and typing tests for the first action-facing VLA ODT slice."""

from __future__ import annotations

import itertools

import pytest
import torch

from xvla.nn.bilinear import BilinearFFN
from xvla.nn.product_routing import ProductRoutingHead
from xvla.train.vla_bilinear_atlas import (
    action_metric,
    coefficient_mode_gram,
    evaluate_quadratics,
    product_routing_quadratic_atlas,
    scalar_quadratic,
    signed_eigensystem,
    symmetric_output_quadratics,
)


def _module(dim=4, rank=7, out_dim=3, down_bias=True):
    torch.manual_seed(12)
    module = BilinearFFN(dim, rank=rank, out_dim=out_dim, down_bias=down_bias).double()
    with torch.no_grad():
        module.left.bias.copy_(torch.linspace(-0.3, 0.2, rank, dtype=torch.float64))
        module.right.bias.copy_(torch.linspace(0.4, -0.1, rank, dtype=torch.float64))
        if module.down.bias is not None:
            module.down.bias.copy_(torch.linspace(-0.2, 0.25, out_dim, dtype=torch.float64))
    return module


def test_every_output_quadratic_reconstructs_module_with_down_bias():
    module = _module()
    x = torch.cat(
        (
            torch.zeros(1, module.dim, dtype=torch.float64),
            torch.eye(module.dim, dtype=torch.float64),
            torch.randn(11, module.dim, dtype=torch.float64),
        ),
        dim=0,
    )
    quadratics = symmetric_output_quadratics(module)
    reconstructed = evaluate_quadratics(quadratics, x)
    torch.testing.assert_close(reconstructed, module(x), rtol=1e-12, atol=1e-12)


def test_scalar_covector_reconstructs_signed_endpoint_and_eigendecomposition():
    module = _module()
    x = torch.randn(13, module.dim, dtype=torch.float64)
    covector = torch.tensor([0.3, -1.2, 0.7], dtype=torch.float64)
    quadratic = scalar_quadratic(module, covector)
    endpoint = evaluate_quadratics(quadratic, x)
    torch.testing.assert_close(endpoint, module(x) @ covector, rtol=1e-12, atol=1e-12)

    values, vectors = signed_eigensystem(quadratic)
    rebuilt = (vectors * values.unsqueeze(0)) @ vectors.T
    torch.testing.assert_close(rebuilt, quadratic, rtol=1e-12, atol=1e-12)
    assert torch.all(values[:-1].abs() >= values[1:].abs())


def test_scalar_quadratic_is_covariant_under_nonorthogonal_input_gauge():
    module = _module(down_bias=True)
    gauged = _module(down_bias=True)
    gauged.load_state_dict(module.state_dict())
    transform = torch.tensor(
        [[2.0, 0.3, 0.0, 0.0],
         [0.0, 0.7, 0.2, 0.0],
         [0.0, 0.0, 1.4, -0.1],
         [0.1, 0.0, 0.0, 0.9]],
        dtype=torch.float64,
    )
    inverse = torch.linalg.inv(transform)
    with torch.no_grad():
        gauged.left.weight.copy_(module.left.weight @ inverse)
        gauged.right.weight.copy_(module.right.weight @ inverse)

    covector = torch.tensor([0.2, 0.5, -0.4], dtype=torch.float64)
    original_q = scalar_quadratic(module, covector)
    gauged_q = scalar_quadratic(gauged, covector)
    homogeneous_transform = torch.eye(module.dim + 1, dtype=torch.float64)
    homogeneous_transform[1:, 1:] = transform
    expected = torch.linalg.inv(homogeneous_transform).T @ original_q @ torch.linalg.inv(homogeneous_transform)
    torch.testing.assert_close(gauged_q, expected, rtol=1e-12, atol=1e-12)

    x = torch.randn(17, module.dim, dtype=torch.float64)
    torch.testing.assert_close(
        evaluate_quadratics(original_q, x),
        evaluate_quadratics(gauged_q, x @ transform.T),
        rtol=1e-12,
        atol=1e-12,
    )


def test_metric_aware_gram_is_covariant_but_raw_gram_is_not():
    module = _module(down_bias=True)
    gauged = _module(down_bias=True)
    gauged.load_state_dict(module.state_dict())
    transform = torch.tensor(
        [[2.0, 0.3, 0.0, 0.0],
         [0.0, 0.7, 0.2, 0.0],
         [0.0, 0.0, 1.4, -0.1],
         [0.1, 0.0, 0.0, 0.9]],
        dtype=torch.float64,
    )
    inverse = torch.linalg.inv(transform)
    with torch.no_grad():
        gauged.left.weight.copy_(module.left.weight @ inverse)
        gauged.right.weight.copy_(module.right.weight @ inverse)

    homogeneous_transform = torch.eye(module.dim + 1, dtype=torch.float64)
    homogeneous_transform[1:, 1:] = transform
    homogeneous_inverse = torch.linalg.inv(homogeneous_transform)
    raw = torch.randn(module.out_dim, module.out_dim, dtype=torch.float64)
    output_metric = raw @ raw.T
    input_metric = torch.eye(module.dim + 1, dtype=torch.float64)
    transported_metric = homogeneous_transform @ input_metric @ homogeneous_transform.T

    original_gram = coefficient_mode_gram(
        module, output_metric, other_leg_metric=input_metric
    )
    gauged_gram = coefficient_mode_gram(
        gauged, output_metric, other_leg_metric=transported_metric
    )
    expected_gram = homogeneous_inverse.T @ original_gram @ homogeneous_inverse
    torch.testing.assert_close(gauged_gram, expected_gram, rtol=1e-11, atol=1e-11)

    original_balanced = input_metric @ original_gram
    gauged_balanced = transported_metric @ gauged_gram
    expected_balanced = homogeneous_transform @ original_balanced @ homogeneous_inverse
    torch.testing.assert_close(gauged_balanced, expected_balanced, rtol=1e-11, atol=1e-11)
    gauged_values = torch.linalg.eigvals(gauged_balanced)
    original_values = torch.linalg.eigvals(original_balanced)
    assert gauged_values.imag.abs().max() < 1e-10
    assert original_values.imag.abs().max() < 1e-10
    torch.testing.assert_close(
        gauged_values.real.sort().values,
        original_values.real.sort().values,
        rtol=1e-10,
        atol=1e-10,
    )

    untransported = coefficient_mode_gram(gauged, output_metric)
    assert not torch.allclose(untransported, expected_gram, rtol=1e-4, atol=1e-6)


def test_coefficient_mode_gram_matches_explicit_matricization_and_is_psd():
    module = _module(out_dim=4)
    raw = torch.randn(module.out_dim, module.out_dim, dtype=torch.float64)
    metric = raw @ raw.T
    gram = coefficient_mode_gram(module, metric)
    core = symmetric_output_quadratics(module)
    root = torch.linalg.cholesky(metric)
    weighted = torch.einsum("po,oij->pij", root.T, core)
    matrix = weighted.permute(1, 0, 2).reshape(module.dim + 1, -1)
    expected = matrix @ matrix.T
    torch.testing.assert_close(gram, expected, rtol=1e-11, atol=1e-11)
    assert torch.linalg.eigvalsh(gram).min() >= -1e-10


def test_coefficient_mode_gram_rejects_invalid_metrics():
    module = _module()
    with pytest.raises(ValueError, match="not symmetric"):
        coefficient_mode_gram(module, torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    with pytest.raises(ValueError, match="not positive semidefinite"):
        coefficient_mode_gram(module, torch.diag(torch.tensor([1.0, -1.0, 1.0])))
    with pytest.raises(ValueError, match="finite"):
        coefficient_mode_gram(module, torch.diag(torch.tensor([1.0, float("nan"), 1.0])))
    with pytest.raises(ValueError, match="finite"):
        coefficient_mode_gram(module, torch.diag(torch.tensor([1.0, float("inf"), 1.0])))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        coefficient_mode_gram(module, torch.eye(3), psd_tolerance=float("nan"))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        coefficient_mode_gram(module, torch.eye(3), psd_tolerance=-1.0)
    with pytest.raises(ValueError, match="other_leg_metric"):
        coefficient_mode_gram(module, torch.eye(3), other_leg_metric=-torch.eye(5))


def test_float32_large_output_metric_does_not_accept_material_negative_mode():
    module = _module(dim=2, rank=4, out_dim=384, down_bias=False).float()
    metric = torch.eye(384, dtype=torch.float32)
    metric[-1, -1] = -1.0e-4
    with pytest.raises(ValueError, match="not positive semidefinite"):
        coefficient_mode_gram(module, metric, dtype=torch.float32)


def test_signed_eigensystem_validates_inputs_and_supports_batches():
    batch = torch.stack(
        (torch.eye(3), torch.diag(torch.tensor([2.0, 2.0, -1.0])))
    ).double()
    values, vectors = signed_eigensystem(batch)
    rebuilt = torch.einsum("...ik,...k,...jk->...ij", vectors, values, vectors)
    torch.testing.assert_close(rebuilt, batch, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="not symmetric"):
        signed_eigensystem(torch.tensor([[1.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="finite"):
        signed_eigensystem(
            torch.tensor([[1.0, float("nan")], [float("nan"), 1.0]])
        )
    with pytest.raises(ValueError, match="finite"):
        signed_eigensystem(
            torch.tensor([[1.0, float("inf")], [float("inf"), 1.0]])
        )


def test_action_metric_preserves_horizon_action_indexing():
    metric = action_metric(
        3,
        7,
        action_indices=[0, 2, 6],
        horizon_indices=[0, 2],
        horizon_weights=[1.0, 2.0, 4.0],
    )
    diagonal = metric.diag().reshape(3, 7)
    expected = torch.zeros(3, 7, dtype=torch.float64)
    expected[0, [0, 2, 6]] = 1.0
    expected[2, [0, 2, 6]] = 4.0
    torch.testing.assert_close(diagonal, expected)
    with pytest.raises(TypeError, match="non-boolean integers"):
        action_metric(3, 7, action_indices=[True])
    with pytest.raises(TypeError, match="non-boolean integers"):
        action_metric(3, 7, horizon_indices=[1.0])


def test_product_routing_atlas_reconstructs_parts_routes_gates_and_flips():
    torch.manual_seed(21)
    head = ProductRoutingHead(5, action_dim=3, horizon=2, n_factors=2, rank=9).double()
    atlas = product_routing_quadratic_atlas(head)
    x = torch.randn(8, head.dim, dtype=torch.float64)
    c0, factors, gates = head(x)

    torch.testing.assert_close(
        evaluate_quadratics(atlas.center, x),
        c0.reshape(-1, head.horizon, head.action_dim),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        evaluate_quadratics(atlas.factors, x),
        factors.reshape(-1, head.G, head.horizon, head.action_dim),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        evaluate_quadratics(atlas.gates, x), gates, rtol=1e-12, atol=1e-12
    )

    for combination in itertools.product((-1.0, 1.0), repeat=head.G):
        signs = torch.tensor(combination, dtype=torch.float64)
        route = evaluate_quadratics(atlas.route(signs), x)
        expected_route = head.action_for_signs(
            c0, factors, signs.expand(x.shape[0], -1)
        ).reshape(-1, head.horizon, head.action_dim)
        torch.testing.assert_close(route, expected_route, rtol=1e-12, atol=1e-12)
        for factor in range(head.G):
            minus = signs.clone()
            plus = signs.clone()
            minus[factor] = -1.0
            plus[factor] = 1.0
            observed_flip = evaluate_quadratics(
                atlas.route(plus), x
            ) - evaluate_quadratics(atlas.route(minus), x)
            predicted_flip = evaluate_quadratics(atlas.factor_flip(factor), x)
            torch.testing.assert_close(
                observed_flip, predicted_flip, rtol=1e-12, atol=1e-12
            )

    greedy_signs = torch.where(gates[0] > 0, 1.0, -1.0)
    greedy_from_atlas = evaluate_quadratics(atlas.route(greedy_signs), x[:1])
    torch.testing.assert_close(
        greedy_from_atlas, head.decode(x[:1]), rtol=1e-12, atol=1e-12
    )


def test_product_routing_atlas_rejects_nonbinary_or_wrong_route():
    head = ProductRoutingHead(3, action_dim=2, horizon=2, n_factors=2, rank=4).double()
    atlas = product_routing_quadratic_atlas(head)
    with pytest.raises(ValueError, match="shape"):
        atlas.route(torch.ones(3))
    with pytest.raises(ValueError, match="exactly"):
        atlas.route(torch.tensor([0.0, 1.0]))


def test_analysis_dtype_must_be_real_floating_point():
    module = _module()
    with pytest.raises(TypeError, match="real floating-point"):
        symmetric_output_quadratics(module, dtype=torch.int64)
    with pytest.raises(TypeError, match="real floating-point"):
        symmetric_output_quadratics(module, dtype=torch.complex128)
