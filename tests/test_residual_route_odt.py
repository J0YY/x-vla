"""Exact tests for residual route metrics and covariances."""

from __future__ import annotations

import pytest
import torch

from xvla.nn.attention import BilinearAttention
from xvla.nn.bilinear import BilinearFFN
from xvla.train.complete_attention_odt import (
    attention_route_features,
    compile_bilinear_attention,
)
from xvla.train.residual_route_odt import (
    compile_bilinear_ffn_residual_core,
    compile_residual_route_assembly,
    joint_route_covariance,
    pullback_residual_feature_metric,
    pushforward_residual_covariance,
    residual_output_route_metric,
)
from xvla.train.generalized_metric_odt import forward_metric, role_environment


DTYPE = torch.float64


def test_residual_metric_has_exact_cross_block_form():
    raw = torch.tensor([[1.0, -0.3], [0.2, 0.8]], dtype=DTYPE)
    metric = raw @ raw.T
    gain = -0.7
    expected = torch.cat(
        (
            torch.cat((metric, gain * metric), dim=1),
            torch.cat((gain * metric, gain * gain * metric), dim=1),
        ),
        dim=0,
    )
    torch.testing.assert_close(residual_output_route_metric(metric, gain=gain), expected)


def test_identity_update_cancellation_requires_cross_terms():
    metric = torch.eye(3, dtype=DTYPE)
    gain = 0.5
    route_metric = residual_output_route_metric(metric, gain=gain)
    base = torch.tensor([1.0, -2.0, 0.5], dtype=DTYPE)
    update = -base / gain
    routes = torch.cat((base, update))
    full_energy = routes @ route_metric @ routes
    diagonal_energy = routes @ torch.diag(torch.diag(route_metric)) @ routes
    torch.testing.assert_close(full_energy, torch.zeros((), dtype=DTYPE), atol=1e-12, rtol=0)
    assert float(diagonal_energy.item()) > 0


def test_compiled_attention_residual_replays_affine_route_exactly():
    torch.manual_seed(1)
    module = BilinearAttention(4, 2, qk_norm="none", causal=True).double().eval()
    with torch.no_grad():
        for branch in (module.wq1, module.wk1, module.wq2, module.wk2, module.wv, module.wo):
            branch.bias.normal_()
    attention = compile_bilinear_attention(module, n_query=3)
    assert attention.output_assembly is not None
    gain = 0.3
    residual = compile_residual_route_assembly(attention.output_assembly, gain=gain)
    tokens = torch.randn(2, 3, 4, dtype=DTYPE)
    features = attention_route_features(attention, tokens)
    homogeneous_features = torch.cat((torch.ones(2, 1, dtype=DTYPE), features), dim=1)
    route_vector = torch.cat((tokens.flatten(1), homogeneous_features), dim=1)
    compiled_output = route_vector @ residual.assembly.T
    expected = tokens + gain * module(tokens)
    torch.testing.assert_close(
        compiled_output.reshape_as(expected), expected, rtol=5e-12, atol=5e-12
    )


def test_bilinear_ffn_residual_route_energy_identity():
    torch.manual_seed(2)
    ffn = BilinearFFN(3, rank=5, out_dim=3, down_bias=True).double().eval()
    states = torch.randn(17, 3, dtype=DTYPE)
    updates = ffn(states)
    gain = 0.8
    output = states + gain * updates
    raw = torch.randn(3, 3, dtype=DTYPE)
    metric = raw @ raw.T
    route_metric = residual_output_route_metric(metric, gain=gain)
    routes = torch.cat((states, updates), dim=1)
    route_energy = torch.einsum("bi,ij,bj->b", routes, route_metric, routes)
    output_energy = torch.einsum("bi,ij,bj->b", output, metric, output)
    torch.testing.assert_close(route_energy, output_energy, rtol=5e-12, atol=5e-12)


def test_nested_attention_then_compiled_bilinear_ffn_residual_replays_exactly():
    torch.manual_seed(21)
    attention = BilinearAttention(2, 1, qk_norm="none", causal=True).double().eval()
    ffn = BilinearFFN(2, rank=4, out_dim=2, down_bias=True).double().eval()
    with torch.no_grad():
        for branch in (attention.wq1, attention.wk1, attention.wq2, attention.wk2, attention.wv, attention.wo):
            branch.bias.normal_()
        ffn.left.bias.normal_()
        ffn.right.bias.normal_()
        ffn.down.bias.normal_()
    tokens = torch.randn(7, 2, 2, dtype=DTYPE)
    after_attention = tokens + 0.4 * attention(tokens)
    core = compile_bilinear_ffn_residual_core(ffn, gain=-0.3)
    homogeneous = torch.cat(
        (torch.ones(7, 2, 1, dtype=DTYPE), after_attention), dim=-1
    )
    compiled = torch.einsum("oij,bni,bnj->bno", core, homogeneous, homogeneous)
    expected = after_attention - 0.3 * ffn(after_attention)
    torch.testing.assert_close(compiled[..., 0], torch.ones_like(compiled[..., 0]))
    torch.testing.assert_close(compiled[..., 1:], expected, rtol=5e-12, atol=5e-12)

    # Check the actual compiled FFN core against the generalized forward and
    # backward energy recursion on independently cloned topology legs.
    flattened = homogeneous.reshape(-1, 3)
    covariance = flattened.T @ flattened / flattened.shape[0]
    output_metric = torch.diag(torch.tensor([0.0, 2.0, 1.0], dtype=DTYPE))
    output_covariance = forward_metric(core, (covariance, covariance))
    parent_energy = torch.trace(output_metric @ output_covariance)
    for role in range(2):
        environment = role_environment(
            core, output_metric, (covariance, covariance), role
        )
        torch.testing.assert_close(
            torch.trace(environment @ covariance), parent_energy, rtol=5e-12, atol=5e-12
        )


def test_residual_covariance_keeps_cross_covariance():
    torch.manual_seed(3)
    samples = torch.randn(1000, 4, dtype=DTYPE)
    transform = torch.tensor(
        [[0.4, 0.1, 0.0, 0.2], [0.0, -0.7, 0.3, 0.0], [0.2, 0.0, 0.5, -0.1], [0.0, 0.3, 0.0, 0.6]],
        dtype=DTYPE,
    )
    updates = samples @ transform.T
    base_covariance = samples.T @ samples / samples.shape[0]
    update_covariance = updates.T @ updates / samples.shape[0]
    cross = samples.T @ updates / samples.shape[0]
    gain = -0.4
    propagated = pushforward_residual_covariance(
        base_covariance, update_covariance, cross, gain=gain
    )
    outputs = samples + gain * updates
    expected = outputs.T @ outputs / samples.shape[0]
    torch.testing.assert_close(propagated, expected, rtol=5e-12, atol=5e-12)
    joint = joint_route_covariance(base_covariance, update_covariance, cross)
    explicit_joint = torch.cat((samples, updates), dim=1)
    torch.testing.assert_close(
        joint, explicit_joint.T @ explicit_joint / samples.shape[0], rtol=5e-12, atol=5e-12
    )


def test_impossible_cross_covariance_is_rejected_before_pushforward():
    identity = torch.eye(2, dtype=DTYPE)
    with pytest.raises(ValueError, match="joint route covariance"):
        pushforward_residual_covariance(identity, identity, 10.0 * identity, gain=-1.0)


def test_feature_pullback_metric_energy_identity_including_affine_constant():
    update_assembly = torch.tensor(
        [[0.2, 1.0, -0.3], [-0.5, 0.4, 0.7]], dtype=DTYPE
    )
    compiled = compile_residual_route_assembly(update_assembly, gain=1.2)
    raw = torch.tensor([[1.0, 0.2], [-0.1, 0.8]], dtype=DTYPE)
    metric = raw @ raw.T
    pulled = pullback_residual_feature_metric(compiled, metric)
    route = torch.tensor([0.4, -0.2, 1.0, 0.7, -1.1], dtype=DTYPE)
    output = compiled.assembly @ route
    torch.testing.assert_close(route @ pulled @ route, output @ metric @ output)


def test_residual_metric_is_covariant_under_nonsingular_output_gauge():
    metric = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=DTYPE)
    transform = torch.tensor([[1.4, 0.2], [-0.1, 0.7]], dtype=DTYPE)
    inverse = torch.linalg.inv(transform)
    gain = 0.6
    original = residual_output_route_metric(metric, gain=gain)
    gauged_metric = inverse.T @ metric @ inverse
    gauged = residual_output_route_metric(gauged_metric, gain=gain)
    route_gauge = torch.block_diag(transform, transform)
    torch.testing.assert_close(
        gauged,
        torch.linalg.inv(route_gauge).T @ original @ torch.linalg.inv(route_gauge),
        rtol=5e-12,
        atol=5e-12,
    )


@pytest.mark.parametrize("gain", [True, float("nan"), float("inf")])
def test_invalid_gains_are_rejected(gain):
    with pytest.raises((TypeError, ValueError), match="gain"):
        residual_output_route_metric(torch.eye(2, dtype=DTYPE), gain=gain)


@pytest.mark.parametrize("assembly", [torch.empty(0, 2, dtype=DTYPE), torch.empty(2, 0, dtype=DTYPE)])
def test_zero_dimensional_residual_assemblies_are_rejected(assembly):
    with pytest.raises(ValueError, match="nonzero dimensions"):
        compile_residual_route_assembly(assembly)


def test_scalar_joint_covariance_fails_with_validation_error():
    with pytest.raises(ValueError, match="matrix"):
        joint_route_covariance(
            torch.tensor(1.0, dtype=DTYPE),
            torch.eye(1, dtype=DTYPE),
            torch.eye(1, dtype=DTYPE),
        )


def test_nonfinite_or_overflowed_ffn_core_fails_closed():
    module = BilinearFFN(2, rank=3, out_dim=2, down_bias=True).double()
    with torch.no_grad():
        module.left.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        compile_bilinear_ffn_residual_core(module)
    finite = BilinearFFN(2, rank=3, out_dim=2, down_bias=True).double()
    with torch.no_grad():
        finite.down.weight.fill_(1e100)
    with pytest.raises(ValueError, match="nonfinite"):
        compile_bilinear_ffn_residual_core(finite, gain=1e308)
