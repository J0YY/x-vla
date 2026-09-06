"""Tests for the exact post-norm VLA action endpoint and balanced metrics."""

from __future__ import annotations

import pytest
import torch

from xvla.models.vla import ChiVLA, VLAConfig
from xvla.train.vla_terminal_balanced_odt import (
    PostNormLinearInterventionPolicy,
    balanced_haar_projector,
    balanced_projector,
    balanced_supported_from_factor,
    balanced_supported_from_observation_factor,
    balanced_supported_eigensystem,
    centered_flattened_reachable_factor,
    centered_reachable_metrics,
    compile_linear_action_endpoint,
    cross_covariance_projector,
    euclidean_principal_projector,
    intervene_post_norm_queries,
    match_removal_doses,
    per_horizon_observability_metrics,
    terminal_observability_metric,
    typed_action_metric,
)


DTYPE = torch.float64


def _model(**overrides) -> ChiVLA:
    values = dict(
        image_size=8,
        patch_size=4,
        vit_dim=4,
        vit_layers=1,
        vit_heads=2,
        vocab_size=16,
        max_instr_len=3,
        state_dim=2,
        n_embodiments=2,
        dim=4,
        n_layers=1,
        n_heads=2,
        ffn_rank=6,
        vit_ffn_rank=6,
        norm="none",
        qk_norm="none",
        action_horizon=3,
        action_dim=7,
        action_head="linear",
    )
    values.update(overrides)
    torch.manual_seed(0)
    return ChiVLA(VLAConfig(**values)).double().eval()


def _batch(seed: int = 1, batch_size: int = 2):
    generator = torch.Generator().manual_seed(seed)
    return {
        "img": torch.randn(batch_size, 3, 8, 8, dtype=DTYPE, generator=generator),
        "instr_ids": torch.randint(0, 16, (batch_size, 3), generator=generator),
        "state": torch.randn(batch_size, 2, dtype=DTYPE, generator=generator),
        "embodiment_id": torch.randint(0, 2, (batch_size,), generator=generator),
    }


def test_endpoint_exact_normalized_denormalized_and_gripper_boundary():
    model = _model()
    mean = torch.linspace(-0.3, 0.3, 7, dtype=DTYPE)
    std = torch.linspace(0.5, 1.1, 7, dtype=DTYPE)
    endpoint = compile_linear_action_endpoint(model, mean, std)
    states = torch.randn(5, 3, 4, dtype=DTYPE)
    normalized = model.action_head(states)
    torch.testing.assert_close(endpoint.normalized(states), normalized, rtol=0, atol=0)
    torch.testing.assert_close(endpoint.controls(states), normalized * std + mean)


def test_endpoint_can_compile_control_affine_map_directly_in_float64():
    model = _model().float()
    mean = torch.linspace(-0.2, 0.2, model.cfg.action_dim, dtype=torch.float32)
    std = torch.linspace(0.1, 0.7, model.cfg.action_dim, dtype=torch.float32)
    endpoint = compile_linear_action_endpoint(model, mean, std, dtype=torch.float64)
    states = torch.randn(5, model.cfg.action_horizon, model.cfg.dim, dtype=torch.float64)
    explicit = endpoint.normalized(states) * endpoint.action_std + endpoint.action_mean
    torch.testing.assert_close(endpoint.control_proposals(states), explicit, atol=1e-14, rtol=1e-14)
    assert endpoint.chunk_jacobian().shape == (21, 12)
    torch.testing.assert_close(
        endpoint.normalized_gripper_threshold,
        (-mean[6] / std[6]).double(),
    )


def test_time_major_metric_exec_h_and_exact_pullback():
    model = _model()
    endpoint = compile_linear_action_endpoint(
        model, torch.zeros(7, dtype=DTYPE), torch.arange(1, 8, dtype=DTYPE)
    )
    metric = typed_action_metric(
        horizon=3, action_dim=7, action_indices=(0, 2), horizon_indices=(0, 1), exec_h=2
    )
    assert metric.diag().nonzero().flatten().tolist() == [0, 2, 7, 9]
    jacobian = endpoint.chunk_jacobian()
    expected = jacobian.T @ metric.to(DTYPE) @ jacobian
    actual = terminal_observability_metric(endpoint, metric)
    torch.testing.assert_close(actual, expected)
    blocks = per_horizon_observability_metrics(actual, horizon=3, hidden_dim=4)
    assert blocks.shape == (3, 4, 4)
    assert float(blocks[2].abs().max()) == 0.0
    states = torch.arange(2 * 3 * 4, dtype=DTYPE).reshape(2, 3, 4) / 10
    delta = states - states.flip(0)
    direct = endpoint.control_proposals(states) - endpoint.control_proposals(states.flip(0))
    torch.testing.assert_close(
        direct.reshape(2, -1), delta.reshape(2, -1) @ endpoint.chunk_jacobian().T
    )
    cross_factor = torch.arange(21 * 3, dtype=DTYPE).reshape(21, 3) / 100
    cross_metric = cross_factor @ cross_factor.T
    torch.testing.assert_close(
        terminal_observability_metric(endpoint, cross_metric),
        endpoint.chunk_jacobian().T @ cross_metric @ endpoint.chunk_jacobian(),
    )


def test_centered_covariance_matches_direct_construction():
    states = torch.randn(17, 3, 4, dtype=DTYPE)
    means, covariance = centered_reachable_metrics(states)
    for horizon in range(3):
        centered = states[:, horizon] - states[:, horizon].mean(0)
        expected = centered.T @ centered / states.shape[0]
        torch.testing.assert_close(means[horizon], states[:, horizon].mean(0))
        torch.testing.assert_close(covariance[horizon], expected)
    factor = centered_flattened_reachable_factor(states)
    flattened = (states - factor.mean).reshape(states.shape[0], -1)
    expected_full = flattened.T @ flattened / states.shape[0]
    torch.testing.assert_close(factor.root @ factor.root.T, expected_full)
    observable_factor = torch.randn(6, expected_full.shape[0], dtype=DTYPE)
    observable = observable_factor.T @ observable_factor
    system = balanced_supported_from_factor(factor.root, observable)
    factor_system = balanced_supported_from_observation_factor(
        factor.root, observable_factor
    )
    torch.testing.assert_close(
        system.eigenvalues, factor_system.eigenvalues, rtol=1e-9, atol=1e-9
    )
    assert system.support_rank == factor.support_rank
    if system.observable_rank:
        identity = torch.eye(system.observable_rank, dtype=DTYPE)
        torch.testing.assert_close(
            system.analysis_directions @ system.state_directions,
            identity,
            rtol=1e-8,
            atol=1e-8,
        )


def test_rank_deficient_balance_projector_and_action_delta_identity():
    generator = torch.Generator().manual_seed(4)
    factor = torch.randn(6, 3, dtype=DTYPE, generator=generator)
    covariance = factor @ factor.T
    observable_factor = torch.randn(5, 6, dtype=DTYPE, generator=generator)
    observable = observable_factor.T @ observable_factor
    system = balanced_supported_eigensystem(covariance, observable, support_rank=3)
    projector = balanced_projector(system, 2)
    torch.testing.assert_close(projector @ projector, projector, rtol=1e-9, atol=1e-9)
    support_projector = balanced_projector(system, 3)
    torch.testing.assert_close(support_projector, system.projector)

    endpoint = compile_linear_action_endpoint(
        _model(dim=6), torch.linspace(-0.1, 0.1, 7), torch.linspace(0.5, 1.0, 7)
    )
    state = torch.randn(8, 6, dtype=DTYPE, generator=generator)
    center = torch.randn(6, dtype=DTYPE, generator=generator)
    edited = center + (state - center) @ projector.T
    actual_delta = endpoint.weight_control @ (edited - state).T
    expected_delta = endpoint.weight_control @ ((projector - torch.eye(6, dtype=DTYPE)) @ (state - center).T)
    torch.testing.assert_close(actual_delta, expected_delta)


def test_balanced_spectrum_and_projector_are_gauge_covariant():
    generator = torch.Generator().manual_seed(5)
    factor = torch.randn(6, 4, dtype=DTYPE, generator=generator)
    covariance = factor @ factor.T
    observable_factor = torch.randn(7, 6, dtype=DTYPE, generator=generator)
    observable = observable_factor.T @ observable_factor
    gauge = torch.randn(6, 6, dtype=DTYPE, generator=generator)
    gauge = gauge + 4 * torch.eye(6, dtype=DTYPE)
    inverse = torch.linalg.inv(gauge)
    first = balanced_supported_eigensystem(covariance, observable, support_rank=4)
    second = balanced_supported_eigensystem(
        gauge @ covariance @ gauge.T,
        inverse.T @ observable @ inverse,
        support_rank=4,
    )
    torch.testing.assert_close(first.eigenvalues, second.eigenvalues, rtol=1e-8, atol=1e-8)
    for rank in (1, 2, 4):
        expected = gauge @ balanced_projector(first, rank) @ inverse
        torch.testing.assert_close(balanced_projector(second, rank), expected, rtol=1e-7, atol=1e-7)


def test_intervention_identity_and_policy_wrapper_identity():
    model = _model()
    batch = _batch()
    _, states = model._encode(**batch)
    means = torch.randn(3, 4, dtype=DTYPE)
    identity = torch.eye(4, dtype=DTYPE)
    torch.testing.assert_close(
        intervene_post_norm_queries(states, means, identity, alpha=1), states
    )
    random_projector = torch.diag(torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=DTYPE))
    torch.testing.assert_close(
        intervene_post_norm_queries(states, means, random_projector, alpha=0), states
    )
    wrapped = PostNormLinearInterventionPolicy(model, means, identity, alpha=1)
    expected, _ = model(**batch)
    actual, loss = wrapped(**batch)
    assert loss is None
    torch.testing.assert_close(actual, expected)
    parameters_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    projector = torch.diag(torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=DTYPE))
    wrapped = PostNormLinearInterventionPolicy(model, means, projector, alpha=0.7)
    actual, _ = wrapped(**batch)
    _, queries = model._encode(**batch)
    manual_states = intervene_post_norm_queries(queries, means, projector, alpha=0.7)
    torch.testing.assert_close(actual, model.action_head(manual_states))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, parameters_before[name], rtol=0, atol=0)

    full = torch.eye(12, dtype=DTYPE)
    full[0, 0] = 0
    full[4, 0] = 0.25
    manual_flat = (queries - means).reshape(queries.shape[0], -1) @ full.T
    expected_full = means + manual_flat.reshape_as(queries)
    actual_full = intervene_post_norm_queries(queries, means, full)
    torch.testing.assert_close(actual_full, expected_full)
    with pytest.raises(ValueError, match="cross-horizon"):
        intervene_post_norm_queries(queries, means, full, horizon_indices=(0,))


def test_zero_observability_has_only_the_rank_zero_covariant_projector():
    covariance = torch.diag(torch.tensor([4.0, 1.0, 0.0], dtype=DTYPE))
    system = balanced_supported_eigensystem(
        covariance, torch.zeros(3, 3, dtype=DTYPE), support_rank=2
    )
    assert system.observable_rank == 0
    assert system.state_directions.shape == (3, 0)
    torch.testing.assert_close(balanced_projector(system, 0), torch.zeros(3, 3, dtype=DTYPE))
    with pytest.raises(ValueError, match="observable_rank"):
        balanced_projector(system, 1)


def test_endpoint_commits_gripper_outside_affine_proposal():
    model = _model()
    endpoint = compile_linear_action_endpoint(
        model, torch.zeros(7, dtype=DTYPE), torch.ones(7, dtype=DTYPE)
    )
    states = torch.randn(4, 3, 4, dtype=DTYPE)
    proposals = endpoint.control_proposals(states)
    committed = endpoint.committed_controls(states)
    torch.testing.assert_close(committed[..., :6], proposals[..., :6])
    expected_gripper = torch.where(
        proposals[..., 6] > 0,
        proposals.new_tensor(1.0),
        proposals.new_tensor(-1.0),
    )
    torch.testing.assert_close(committed[..., 6], expected_gripper)


def test_float32_default_metric_composition_promotes_without_downcast():
    model = _model().float()
    endpoint = compile_linear_action_endpoint(
        model, torch.zeros(7, dtype=torch.float32), torch.ones(7, dtype=torch.float32)
    )
    states = torch.randn(20, 3, 4, dtype=torch.float32)
    _, covariance = centered_reachable_metrics(states)
    metric = typed_action_metric(horizon=3, action_dim=7)
    chunk_observable = terminal_observability_metric(endpoint, metric)
    blocks = per_horizon_observability_metrics(chunk_observable, horizon=3, hidden_dim=4)
    system = balanced_supported_eigensystem(covariance[0], blocks[0])
    assert system.eigenvalues.dtype == torch.float64


def test_rank_matched_controls_and_dose_matching_are_exact():
    model = _model()
    endpoint = compile_linear_action_endpoint(
        model, torch.zeros(7, dtype=DTYPE), torch.ones(7, dtype=DTYPE)
    )
    generator = torch.Generator().manual_seed(31)
    states = torch.randn(80, 3, 4, dtype=DTYPE, generator=generator)
    means, covariances = centered_reachable_metrics(states)
    observation = endpoint.weight_normalized[:3]
    observable = observation.T @ observation
    systems = [balanced_supported_eigensystem(covariances[h], observable) for h in range(3)]
    projectors = {
        "balanced": torch.stack([balanced_projector(system, 2) for system in systems]),
        "pca": torch.stack([euclidean_principal_projector(covariances[h], 2) for h in range(3)]),
        "observable": torch.stack([euclidean_principal_projector(observable, 2) for _ in range(3)]),
        "cross_cov": torch.stack(
            [cross_covariance_projector(covariances[h], observation, 2) for h in range(3)]
        ),
        "haar": torch.stack(
            [balanced_haar_projector(system, 2, seed=100 + h) for h, system in enumerate(systems)]
        ),
    }
    for projector in projectors.values():
        torch.testing.assert_close(projector @ projector, projector, atol=1e-10, rtol=1e-10)
    matched = match_removal_doses(
        endpoint,
        states,
        means,
        projectors,
        primary="balanced",
        alpha_grid=(1.0, 0.5, 0.25, 0.1, 0.05),
        action_indices=(0, 1, 2),
    )
    assert all(0 < value <= 1 for value in matched.alphas.values())
    assert max(abs(value - matched.target_dose) for value in matched.realized_doses.values()) <= 1e-12


def test_thin_factor_system_is_covariant_under_nonorthogonal_gauge():
    generator = torch.Generator().manual_seed(41)
    root = torch.randn(7, 5, dtype=DTYPE, generator=generator)
    observation = torch.randn(3, 7, dtype=DTYPE, generator=generator)
    system = balanced_supported_from_observation_factor(root, observation)
    gauge = torch.randn(7, 7, dtype=DTYPE, generator=generator)
    gauge = gauge + 4 * torch.eye(7, dtype=DTYPE)
    inverse = torch.linalg.inv(gauge)
    gauged = balanced_supported_from_observation_factor(gauge @ root, observation @ inverse)
    torch.testing.assert_close(gauged.eigenvalues, system.eigenvalues, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        gauged.projector,
        gauge @ system.projector @ inverse,
        atol=1e-9,
        rtol=1e-9,
    )


@pytest.mark.parametrize("head", ["product", "flow", "quantile"])
def test_endpoint_rejects_nonlinear_or_decoded_heads(head):
    model = _model(action_head=head)
    with pytest.raises(ValueError, match="linear action head"):
        compile_linear_action_endpoint(model, torch.zeros(7), torch.ones(7))


def test_metric_and_intervention_validation_fail_closed():
    with pytest.raises(ValueError, match="outside exec_h"):
        typed_action_metric(horizon=3, action_dim=7, horizon_indices=(2,), exec_h=2)
    with pytest.raises(ValueError, match="unique"):
        typed_action_metric(horizon=3, action_dim=7, action_indices=(0, 0))
    with pytest.raises(ValueError, match="samples >= 2"):
        centered_reachable_metrics(torch.randn(1, 3, 4))
    covariance = torch.eye(3, dtype=DTYPE)
    nonsymmetric = covariance.clone()
    nonsymmetric[0, 1] = 0.1
    with pytest.raises(ValueError, match="symmetric"):
        balanced_supported_eigensystem(covariance, nonsymmetric)
    with pytest.raises(ValueError, match="finite and positive"):
        balanced_supported_eigensystem(covariance, covariance, relative_tolerance=float("nan"))
    with pytest.raises(ValueError, match="not gauge covariant"):
        balanced_supported_eigensystem(covariance, covariance, support_rank=2)
    with pytest.raises(ValueError, match="finite and positive"):
        balanced_supported_from_observation_factor(
            torch.eye(3, dtype=DTYPE), torch.eye(3, dtype=DTYPE), relative_tolerance=float("nan")
        )
    states = torch.randn(2, 3, 4, dtype=DTYPE)
    means = torch.zeros(3, 4, dtype=DTYPE)
    with pytest.raises(ValueError, match="alpha must be finite"):
        intervene_post_norm_queries(states, means, torch.eye(4, dtype=DTYPE), alpha=float("nan"))
    with pytest.raises(ValueError, match="alpha must be finite"):
        intervene_post_norm_queries(states, means, torch.eye(12, dtype=DTYPE), alpha=float("inf"))
