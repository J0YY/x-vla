from __future__ import annotations

import pytest
import torch

from xvla.train.generalized_metric_odt import forward_metric, role_environment
from xvla.train.planted_typed_vla import (
    CLAIM_BOUNDARY,
    GaugeTransform,
    SemanticCounterfactualPanel,
    TypedCPPolicy,
    TypedInputPanel,
    apply_hidden_gauge,
    analysis_covectors_from_state_directions,
    balanced_role_odt,
    component_zeroing_intervention,
    component_zeroing_projector,
    counterfactual_swap_error,
    dense_coefficient_core,
    dense_forward,
    dense_role_environment_oracle,
    empirical_input_metrics,
    factorized_forward,
    factorized_forward_metric,
    factorized_role_environment,
    gauge_panel,
    intervention_effects,
    make_discovery_panel,
    make_planted_policy,
    make_random_gauges,
    make_semantic_counterfactual_panel,
    map_state_directions_to_canonical,
    multimodal_query_branch_error,
    ordered_component_recovery,
    planted_input_metrics,
    subspace_recovery,
    transform_input_metrics,
    transform_inputs,
)


DTYPE = torch.float64


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)
    return float((torch.linalg.vector_norm(actual - expected) / denominator).item())


def random_spd(generator: torch.Generator, dimension: int) -> torch.Tensor:
    factor = torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
    return factor @ factor.T / dimension + 0.2 * torch.eye(dimension, dtype=DTYPE)


def tiny_policy(seed: int = 1) -> TypedCPPolicy:
    generator = torch.Generator().manual_seed(seed)
    factors = tuple(
        torch.linalg.qr(torch.randn(2, 2, generator=generator, dtype=DTYPE)).Q
        for _ in range(6)
    )
    output = torch.randn(3, 2, generator=generator, dtype=DTYPE)
    return TypedCPPolicy(
        factors,
        output,
        mechanism_names=("mechanism_0", "mechanism_1"),
    )


def test_factorized_forward_matches_independently_materialized_order_seven_core() -> None:
    policy = tiny_policy()
    generator = torch.Generator().manual_seed(2)
    inputs = tuple(
        torch.randn(9, dimension, generator=generator, dtype=DTYPE)
        for dimension in policy.role_dims
    )
    assert relative_error(factorized_forward(policy, inputs), dense_forward(policy, inputs)) < 2e-13


def test_dense_core_has_frozen_six_role_shape() -> None:
    policy = tiny_policy()
    assert dense_coefficient_core(policy).shape == (3, 2, 2, 2, 2, 2, 2)


def test_factorized_forward_metric_matches_generic_dense_core_recursion() -> None:
    policy = tiny_policy(3)
    generator = torch.Generator().manual_seed(4)
    metrics = tuple(random_spd(generator, dimension) for dimension in policy.role_dims)
    actual = factorized_forward_metric(policy, metrics)
    expected = forward_metric(dense_coefficient_core(policy), metrics)
    assert relative_error(actual, expected) < 4e-13


@pytest.mark.parametrize("role", range(6))
def test_every_factorized_role_environment_matches_two_dense_oracles(role: int) -> None:
    policy = tiny_policy(5)
    generator = torch.Generator().manual_seed(6)
    metrics = tuple(random_spd(generator, dimension) for dimension in policy.role_dims)
    downstream = random_spd(generator, policy.output_dim)
    actual = factorized_role_environment(policy, downstream, metrics, role)
    explicit = dense_role_environment_oracle(policy, downstream, metrics, role)
    generic = role_environment(dense_coefficient_core(policy), downstream, metrics, role)
    assert relative_error(actual, explicit) < 5e-13
    assert relative_error(actual, generic) < 5e-13


def test_roles_are_typed_and_role_specific_metrics_change_the_correct_environment() -> None:
    policy = tiny_policy(7)
    metrics = list(planted_input_metrics(policy))
    downstream = torch.eye(policy.output_dim, dtype=DTYPE)
    baseline = factorized_role_environment(policy, downstream, metrics, 0)
    metrics[1] = 2.0 * metrics[1]
    changed = factorized_role_environment(policy, downstream, metrics, 0)
    assert torch.allclose(changed, 2.0 * baseline, rtol=2e-12, atol=2e-12)


def test_nonorthogonal_hidden_gauge_preserves_policy_function() -> None:
    policy = make_planted_policy(8, dtype=DTYPE)
    gauges = make_random_gauges(9, policy.role_dims, dtype=DTYPE)
    transformed_policy = apply_hidden_gauge(policy, gauges)
    generator = torch.Generator().manual_seed(10)
    inputs = tuple(
        torch.randn(13, dimension, generator=generator, dtype=DTYPE)
        for dimension in policy.role_dims
    )
    transformed_inputs = transform_inputs(inputs, gauges)
    assert relative_error(transformed_policy(transformed_inputs), policy(inputs)) < 2e-13
    assert max(float(torch.linalg.cond(matrix).item()) for matrix in gauges.matrices) > 2


@pytest.mark.parametrize("role", range(6))
def test_role_environment_has_exact_covector_gauge_transport(role: int) -> None:
    policy = make_planted_policy(11, dtype=DTYPE)
    metrics = planted_input_metrics(policy)
    gauges = make_random_gauges(12, policy.role_dims, dtype=DTYPE)
    gauged_policy = apply_hidden_gauge(policy, gauges)
    gauged_metrics = transform_input_metrics(metrics, gauges)
    downstream = torch.eye(policy.output_dim, dtype=DTYPE)
    canonical = factorized_role_environment(policy, downstream, metrics, role)
    transformed = factorized_role_environment(
        gauged_policy, downstream, gauged_metrics, role
    )
    inverse = torch.linalg.inv(gauges.matrices[role])
    expected = inverse.T @ canonical @ inverse
    assert relative_error(transformed, expected) < 2e-12


@pytest.mark.parametrize("role", range(6))
def test_balanced_state_directions_are_gauge_covariant_and_recover_support(role: int) -> None:
    policy = make_planted_policy(13, dtype=DTYPE)
    metrics = planted_input_metrics(policy)
    gauges = make_random_gauges(14, policy.role_dims, dtype=DTYPE)
    gauged_policy = apply_hidden_gauge(policy, gauges)
    gauged_metrics = transform_input_metrics(metrics, gauges)
    downstream = torch.eye(policy.output_dim, dtype=DTYPE)
    system = balanced_role_odt(
        gauged_policy, downstream, gauged_metrics, role, support_rtol=1e-13
    )
    mapped = map_state_directions_to_canonical(
        system.state_directions[:, : policy.rank], gauges.matrices[role]
    )
    assert subspace_recovery(policy.role_factors[role], mapped) > 1 - 2e-12
    assert ordered_component_recovery(policy.role_factors[role], mapped) > 1 - 2e-12
    recovered_analysis = analysis_covectors_from_state_directions(
        gauged_metrics[role], system.state_directions[:, : policy.rank]
    )
    assert subspace_recovery(gauged_policy.role_factors[role], recovered_analysis) > 1 - 2e-12


def test_discovery_panel_exactly_realizes_requested_population_covariances() -> None:
    policy = make_planted_policy(15, dtype=DTYPE)
    metrics = planted_input_metrics(policy, signal_variance=1.0, nuisance_variance=9.0)
    panel = make_discovery_panel(policy, metrics, namespace="discovery")
    estimates = empirical_input_metrics(panel)
    assert all(relative_error(actual, expected) < 3e-15 for actual, expected in zip(estimates, metrics))
    assert panel.namespace == "discovery"
    assert len(panel.sample_ids) == 2 * policy.role_dims[0]


def test_discovery_and_evaluation_namespaces_and_ids_are_disjoint() -> None:
    policy = make_planted_policy(16, dtype=DTYPE)
    discovery = make_discovery_panel(
        policy, planted_input_metrics(policy), namespace="discovery-s16"
    )
    evaluation = make_semantic_counterfactual_panel(
        policy, 17, pairs=9, namespace="evaluation-s16"
    )
    assert discovery.namespace != evaluation.panel.namespace
    assert set(discovery.sample_ids).isdisjoint(evaluation.panel.sample_ids)


def test_semantic_panel_pairs_hold_every_noninstruction_role_fixed() -> None:
    policy = make_planted_policy(18, dtype=DTYPE)
    semantic = make_semantic_counterfactual_panel(
        policy, 19, pairs=12, namespace="evaluation"
    )
    for role in range(1, 6):
        assert torch.equal(
            semantic.panel.role_inputs[role][0::2],
            semantic.panel.role_inputs[role][1::2],
        )
    assert not torch.equal(
        semantic.panel.role_inputs[0][0::2],
        semantic.panel.role_inputs[0][1::2],
    )


def test_counterfactual_instruction_swap_exactly_reaches_paired_output() -> None:
    policy = make_planted_policy(20, dtype=DTYPE)
    semantic = make_semantic_counterfactual_panel(
        policy, 21, pairs=18, namespace="evaluation"
    )
    assert float(counterfactual_swap_error(policy, semantic).max().item()) < 2e-13
    outputs = policy(semantic.panel.role_inputs)
    assert bool((torch.linalg.vector_norm(outputs[0::2] - outputs[1::2], dim=1) > 0.1).all())


def test_action_query_contains_an_exact_known_opposite_mode_branch() -> None:
    policy = make_planted_policy(201, dtype=DTYPE)
    semantic = make_semantic_counterfactual_panel(
        policy, 202, pairs=12, namespace="evaluation"
    )
    assert float(multimodal_query_branch_error(policy, semantic).max().item()) < 2e-13
    outputs = policy(semantic.panel.role_inputs)
    assert bool((2 * torch.linalg.vector_norm(outputs, dim=1) > 0.1).all())


@pytest.mark.parametrize("role", range(6))
def test_targeted_component_intervention_is_selective_for_every_role(role: int) -> None:
    policy = make_planted_policy(22, dtype=DTYPE)
    semantic = make_semantic_counterfactual_panel(
        policy, 23, pairs=18, namespace="evaluation"
    )
    metric = planted_input_metrics(policy)[role]
    effects = intervention_effects(policy, semantic, role=role, upstream_metric=metric)
    assert float(effects["intervened_target_energy"].max().item()) < 2e-25
    assert float(effects["off_target_energy"].max().item()) < 2e-25
    assert bool((effects["target_energy"] > 1e-4).all())


def test_component_zeroing_preserves_all_nonselected_cp_scores() -> None:
    policy = make_planted_policy(24, dtype=DTYPE)
    generator = torch.Generator().manual_seed(25)
    values = torch.randn(11, policy.role_dims[2], generator=generator, dtype=DTYPE)
    before = values @ policy.role_factors[2]
    metric = planted_input_metrics(policy)[2]
    after_values = component_zeroing_intervention(values, policy.role_factors[2], 4, metric)
    after = after_values @ policy.role_factors[2]
    assert float(after[:, 4].abs().max().item()) < 3e-15
    assert torch.allclose(after[:, (0, 1, 2, 3, 5)], before[:, (0, 1, 2, 3, 5)], rtol=1e-13, atol=1e-13)


def test_gauged_semantic_counterfactual_and_intervention_remain_exact() -> None:
    policy = make_planted_policy(26, dtype=DTYPE)
    semantic = make_semantic_counterfactual_panel(
        policy, 27, pairs=9, namespace="canonical-eval"
    )
    gauges = make_random_gauges(28, policy.role_dims, dtype=DTYPE)
    gauged_policy = apply_hidden_gauge(policy, gauges)
    transported = gauge_panel(semantic.panel, gauges, namespace="gauged-eval")
    gauged_semantic = SemanticCounterfactualPanel(
        panel=transported,
        pair_ids=semantic.pair_ids,
        instruction_labels=semantic.instruction_labels,
        active_components=semantic.active_components,
    )
    canonical_outputs = policy(semantic.panel.role_inputs)
    gauged_outputs = gauged_policy(gauged_semantic.panel.role_inputs)
    assert relative_error(gauged_outputs, canonical_outputs) < 2e-13
    assert float(counterfactual_swap_error(gauged_policy, gauged_semantic).max().item()) < 2e-13
    gauged_metrics = transform_input_metrics(planted_input_metrics(policy), gauges)
    effects = intervention_effects(
        gauged_policy, gauged_semantic, role=0, upstream_metric=gauged_metrics[0]
    )
    assert float(effects["intervened_target_energy"].max().item()) < 2e-24
    assert float(effects["off_target_energy"].max().item()) < 2e-24


def test_component_zeroing_projector_transports_under_nonorthogonal_gauge() -> None:
    policy = make_planted_policy(126, dtype=DTYPE)
    metrics = planted_input_metrics(policy)
    gauges = make_random_gauges(127, policy.role_dims, dtype=DTYPE)
    gauged_policy = apply_hidden_gauge(policy, gauges)
    gauged_metrics = transform_input_metrics(metrics, gauges)
    for role, gauge in enumerate(gauges.matrices):
        canonical = component_zeroing_projector(
            policy.role_factors[role], 2, metrics[role]
        )
        transformed = component_zeroing_projector(
            gauged_policy.role_factors[role], 2, gauged_metrics[role]
        )
        torch.testing.assert_close(
            transformed,
            gauge @ canonical @ torch.linalg.inv(gauge),
            atol=1e-11,
            rtol=1e-11,
        )


def test_high_variance_activation_pca_is_a_real_negative_control() -> None:
    policy = make_planted_policy(29, dtype=DTYPE)
    metrics = planted_input_metrics(policy, signal_variance=1.0, nuisance_variance=9.0)
    values, vectors = torch.linalg.eigh(metrics[0])
    pca = vectors[:, torch.argsort(values, descending=True)[: policy.rank]]
    assert subspace_recovery(policy.role_factors[0], pca) < 1e-12
    system = balanced_role_odt(
        policy, torch.eye(policy.output_dim, dtype=DTYPE), metrics, 0
    )
    assert subspace_recovery(
        policy.role_factors[0], system.state_directions[:, : policy.rank]
    ) > 1 - 2e-12


def test_same_support_haar_preserves_span_but_destroys_frozen_component_order() -> None:
    policy = make_planted_policy(30, dtype=DTYPE)
    generator = torch.Generator().manual_seed(31)
    rotation = torch.linalg.qr(
        torch.randn(policy.rank, policy.rank, generator=generator, dtype=DTYPE)
    ).Q
    rotated = policy.role_factors[0] @ rotation
    assert subspace_recovery(policy.role_factors[0], rotated) > 1 - 2e-13
    assert ordered_component_recovery(policy.role_factors[0], rotated) < 0.6


def test_claim_boundary_refuses_real_vla_and_global_attention_claims() -> None:
    lowered = CLAIM_BOUNDARY.lower()
    assert "synthetic" in lowered
    assert "not establish global odt for attention" in lowered
    assert "not" in lowered and "trained vla" in lowered


@pytest.mark.parametrize(
    "builder",
    (
        lambda policy: factorized_forward(policy, policy.role_factors[:5]),
        lambda policy: factorized_role_environment(
            policy,
            torch.eye(policy.output_dim, dtype=DTYPE),
            planted_input_metrics(policy),
            True,
        ),
        lambda policy: make_semantic_counterfactual_panel(
            policy, 0, pairs=0, namespace="bad"
        ),
        lambda policy: component_zeroing_intervention(
            torch.zeros(1, policy.role_dims[0], dtype=DTYPE),
            policy.role_factors[0],
            9,
            planted_input_metrics(policy)[0],
        ),
    ),
)
def test_invalid_requests_fail_closed(builder) -> None:
    with pytest.raises((TypeError, ValueError)):
        builder(make_planted_policy(32, dtype=DTYPE))


def test_policy_panel_and_gauge_validation_fail_closed() -> None:
    policy = make_planted_policy(33, dtype=DTYPE)
    with pytest.raises(ValueError, match="six role"):
        TypedCPPolicy(policy.role_factors[:5], policy.output_factor)
    with pytest.raises(ValueError, match="invertible"):
        GaugeTransform(tuple(torch.zeros(12, 12, dtype=DTYPE) for _ in range(6)))
    with pytest.raises(ValueError, match="unique"):
        TypedInputPanel(
            "bad",
            ("same", "same"),
            tuple(torch.zeros(2, 12, dtype=DTYPE) for _ in range(6)),
        )
