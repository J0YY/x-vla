"""Pure tests for outcome-blind Stage 7 v2 control construction."""

from __future__ import annotations

import pytest
import torch

import xvla.train.post_norm_intervention_v2 as intervention_v2
from xvla.train.post_norm_intervention import HorizonLowRankProjector
from xvla.train.post_norm_intervention_v2 import (
    DESIGN_NAMESPACE, SAMPLE_NAMESPACE, SPLIT_NAMESPACE, NuisanceDiagnostics,
    first_passing_indices, namespaced_seed, nuisance_matches,
    select_conditional_haar_controls, select_largest_feasible_beta,
    task_stratified_split,
)
from xvla.train.vla_terminal_balanced_odt import LinearActionEndpoint


DTYPE = torch.float64


def endpoint(hidden: int = 3, actions: int = 2, horizon: int = 2) -> LinearActionEndpoint:
    weight = torch.zeros(actions, hidden, dtype=DTYPE)
    weight[0, 0] = 1
    weight[1, 1] = 1
    return LinearActionEndpoint(
        weight_normalized=weight, bias_normalized=torch.zeros(actions, dtype=DTYPE),
        weight_control=weight.clone(), bias_control=torch.zeros(actions, dtype=DTYPE),
        action_mean=torch.zeros(actions, dtype=DTYPE), action_std=torch.ones(actions, dtype=DTYPE),
        horizon=horizon,
    )


def primary(horizon: int = 2, hidden: int = 3) -> HorizonLowRankProjector:
    state = torch.zeros(horizon, hidden, 1, dtype=DTYPE)
    state[:, 0, 0] = 1
    return HorizonLowRankProjector(state, state.transpose(1, 2), "balanced_top")


def test_seed_namespaces_are_stable_and_disjoint() -> None:
    assert namespaced_seed(DESIGN_NAMESPACE, 1, 2, 3) == namespaced_seed(DESIGN_NAMESPACE, 1, 2, 3)
    assert namespaced_seed(DESIGN_NAMESPACE, 1, 2, 3) != namespaced_seed(SAMPLE_NAMESPACE, 1, 2, 3)
    assert namespaced_seed(SAMPLE_NAMESPACE, 1, 2, 3) != namespaced_seed(SPLIT_NAMESPACE, 1, 2, 3)
    assert namespaced_seed(DESIGN_NAMESPACE, 1, 2, 3) != namespaced_seed(DESIGN_NAMESPACE, 1, 2, 4)
    with pytest.raises(ValueError, match="namespace"):
        namespaced_seed("not-frozen", 1)


def test_task_stratified_split_is_disjoint_complete_balanced_and_order_stable() -> None:
    tasks = [0] * 6 + [1] * 6
    samples = [f"sample-{index}" for index in range(12)]
    selection, certification = task_stratified_split(tasks, samples, split_seed=17)
    assert not set(selection) & set(certification)
    assert set(selection) | set(certification) == set(range(12))
    assert [sum(tasks[index] == task for index in selection) for task in (0, 1)] == [3, 3]
    assert [sum(tasks[index] == task for index in certification) for task in (0, 1)] == [3, 3]
    assert (selection, certification) == task_stratified_split(tasks, samples, split_seed=17)


def test_task_stratified_split_rejects_odd_duplicate_or_bad_task_input() -> None:
    with pytest.raises(ValueError, match="even"):
        task_stratified_split([0, 0, 0], ["a", "b", "c"], split_seed=1)
    with pytest.raises(ValueError, match="unique"):
        task_stratified_split([0, 0], ["a", "a"], split_seed=1)
    with pytest.raises(ValueError, match="task ids"):
        task_stratified_split([False, False], ["a", "b"], split_seed=1)


def test_nuisance_match_checks_aggregate_allocation_and_incidence_independently() -> None:
    reference = NuisanceDiagnostics(1.0, 0.8, 0.6, 0.02, 0.03)
    assert nuisance_matches(reference, reference, aggregate_atol=1e-12, allocation_rtol=.1, incidence_atol=.01)
    fields = (
        NuisanceDiagnostics(1.1, .8, .6, .02, .03),
        NuisanceDiagnostics(1, .6, .6, .02, .03),
        NuisanceDiagnostics(1, .8, .6, .2, .03),
    )
    assert all(not nuisance_matches(value, reference, aggregate_atol=1e-12, allocation_rtol=.1, incidence_atol=.01) for value in fields)
    assert nuisance_matches(fields[0], reference, aggregate_atol=0, aggregate_rtol=.11, allocation_rtol=.1, incidence_atol=.01)


def test_conditional_haar_is_reproducible_and_returns_exact_inventory() -> None:
    generator = torch.Generator().manual_seed(44)
    queries = torch.randn(128, 2, 3, generator=generator, dtype=DTYPE)
    means = torch.zeros(2, 3, dtype=DTYPE)
    state = [torch.eye(3, dtype=DTYPE) for _ in range(2)]
    analysis = [torch.eye(3, dtype=DTYPE) for _ in range(2)]
    labels = ("matched_haar_d0", "matched_haar_d1")
    matched_endpoint = endpoint()
    matched_endpoint.weight_normalized[1].copy_(matched_endpoint.weight_normalized[0])
    matched_endpoint.weight_control[1].copy_(matched_endpoint.weight_control[0])
    kwargs = dict(
        checkpoint_seed=5, labels=labels, alpha_grid=(0.1,),
        feasibility_pool_size=5, minimum_feasibility_eligible=1, sampling_pool_size=100,
        aggregate_atol=1e-12, selection_allocation_rtol=100.0,
        selection_incidence_atol=1.0, certification_allocation_rtol=100.0,
        certification_incidence_atol=1.0,
    )
    left = select_conditional_haar_controls(
        matched_endpoint, queries, queries.clone(), means, primary(), state, analysis, **kwargs
    )
    right = select_conditional_haar_controls(
        matched_endpoint, queries, queries.clone(), means, primary(), state, analysis, **kwargs
    )
    assert tuple(left.projectors) == ("balanced_top", *labels)
    assert left.proposal_indices == right.proposal_indices
    assert left.proposal_seeds == right.proposal_seeds
    assert left.alphas == right.alphas
    for name in left.projectors:
        torch.testing.assert_close(left.projectors[name].dense(), right.projectors[name].dense(), rtol=0, atol=0)
        assert abs(left.selection[name].aggregate_rms - left.target_rms) <= 1e-12


def test_conditional_haar_has_fail_closed_budget_and_certification(monkeypatch: pytest.MonkeyPatch) -> None:
    queries = torch.arange(1, 1 + 64 * 2 * 3, dtype=DTYPE).reshape(64, 2, 3) / 100
    state = [torch.eye(3, dtype=DTYPE) for _ in range(2)]
    analysis = [torch.eye(3, dtype=DTYPE) for _ in range(2)]
    calls = 0
    def selection_only(*args, **kwargs):
        nonlocal calls
        calls += 1
        return calls <= 2
    monkeypatch.setattr(intervention_v2, "nuisance_matches", selection_only)
    with pytest.raises(RuntimeError, match="certification"):
        select_conditional_haar_controls(
            endpoint(), queries, queries, torch.zeros(2, 3, dtype=DTYPE), primary(), state, analysis,
            checkpoint_seed=2, labels=("matched_haar_d0",), alpha_grid=(1e-6,),
            feasibility_pool_size=1, minimum_feasibility_eligible=1, sampling_pool_size=1,
            aggregate_atol=1e-12, selection_allocation_rtol=100, selection_incidence_atol=1,
            certification_allocation_rtol=100, certification_incidence_atol=1,
        )


def test_two_stream_selection_oracles_are_exact() -> None:
    grid = (1.0, 0.75, 0.5, 0.25)
    counts = {1.0: 3, 0.75: 31, 0.5: 32, 0.25: 80}
    assert select_largest_feasible_beta(grid, counts, 32) == 0.5
    eligibility = (False, True, False, True, True, False, True)
    assert first_passing_indices(eligibility, 3) == (1, 3, 4)
    with pytest.raises(RuntimeError, match="few"):
        first_passing_indices(eligibility, 5)
