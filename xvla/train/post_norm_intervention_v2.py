"""Outcome-blind conditional-Haar controls for Stage 7 v2.

This module contains only tensor and deterministic-design utilities.  It never
loads LIBERO and never observes rollout rewards.  Candidate controls are Haar
draws inside the same balanced observable support as the primary projector.
They are accepted on a selection split only when precommitted action-space
nuisance diagnostics match the primary.  A disjoint certification split is
then used as a fail-closed check, never to choose another candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import torch

from xvla.train.post_norm_intervention import (
    HorizonLowRankProjector,
    normalized_action_delta,
    normalized_action_dose,
)
from xvla.train.vla_terminal_balanced_odt import LinearActionEndpoint


Tensor = torch.Tensor
DESIGN_NAMESPACE = "xvla-stage7-v2-design"
SAMPLE_NAMESPACE = "xvla-stage7-v2-sample"
SPLIT_NAMESPACE = "xvla-stage7-v2-split"


def namespaced_seed(namespace: str, *parts: object) -> int:
    """Return a stable positive 63-bit seed from an explicit namespace."""

    if namespace not in (DESIGN_NAMESPACE, SAMPLE_NAMESPACE, SPLIT_NAMESPACE):
        raise ValueError("seed namespace is not a frozen Stage 7 v2 namespace")
    payload = "|".join((namespace, *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def task_stratified_split(
    task_ids: Sequence[int],
    sample_ids: Sequence[str],
    *,
    split_seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split every task deterministically into equal selection/certification sets.

    The split uses only sample identity and the split namespace.  Outcomes,
    activations, actions, and design seeds cannot affect membership.
    """

    tasks = tuple(task_ids)
    samples = tuple(sample_ids)
    if not tasks or len(tasks) != len(samples) or len(set(samples)) != len(samples):
        raise ValueError("task_ids and unique sample_ids must have equal nonzero length")
    by_task: dict[int, list[int]] = {}
    for index, task in enumerate(tasks):
        if isinstance(task, bool) or not isinstance(task, int) or task < 0:
            raise ValueError("task ids must be nonnegative integers")
        by_task.setdefault(task, []).append(index)
    selection: list[int] = []
    certification: list[int] = []
    for task in sorted(by_task):
        ordered = sorted(
            by_task[task],
            key=lambda index: hashlib.sha256(
                f"{SPLIT_NAMESPACE}|{split_seed}|{task}|{samples[index]}".encode()
            ).hexdigest(),
        )
        if len(ordered) < 2 or len(ordered) % 2:
            raise ValueError("each task must have a positive even sample count")
        midpoint = len(ordered) // 2
        selection.extend(ordered[:midpoint])
        certification.extend(ordered[midpoint:])
    left = tuple(sorted(selection))
    right = tuple(sorted(certification))
    if set(left) & set(right) or set(left) | set(right) != set(range(len(tasks))):
        raise RuntimeError("stratified split is not a disjoint partition")
    for task in by_task:
        if sum(tasks[index] == task for index in left) != sum(tasks[index] == task for index in right):
            raise RuntimeError("stratified split is not balanced within task")
    return left, right


@dataclass(frozen=True)
class NuisanceDiagnostics:
    aggregate_rms: float
    continuous_rms: float
    gripper_rms: float
    gripper_flip_fraction: float
    gripper_margin_reachable_fraction: float


@dataclass(frozen=True)
class ConditionalHaarMatch:
    projectors: dict[str, HorizonLowRankProjector]
    alphas: dict[str, float]
    target_rms: float
    primary_alpha: float
    proposal_indices: dict[str, int]
    proposal_seeds: dict[str, tuple[int, ...]]
    selection: dict[str, NuisanceDiagnostics]
    certification: dict[str, NuisanceDiagnostics]
    proposals_examined: int
    feasibility_eligible_by_alpha: dict[float, int]
    feasibility_pool_size: int
    sampling_pool_size: int
    sampling_eligibility: tuple[bool, ...]


def select_largest_feasible_beta(
    alpha_grid: Sequence[float], counts: Mapping[float, int], minimum: int
) -> float:
    grid = tuple(float(value) for value in alpha_grid)
    if minimum <= 0 or not grid or any(left <= right for left, right in zip(grid, grid[1:])):
        raise ValueError("invalid beta selection design")
    if set(counts) != set(grid) or any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("feasibility count inventory differs")
    selected = next((alpha for alpha in grid if counts[alpha] >= minimum), None)
    if selected is None:
        raise RuntimeError("no beta has the minimum eligible design-stream feasibility count")
    return selected


def first_passing_indices(eligibility: Sequence[bool], count: int) -> tuple[int, ...]:
    if count <= 0 or any(type(value) is not bool for value in eligibility):
        raise ValueError("first-passing input differs")
    indices = tuple(index for index, passed in enumerate(eligibility) if passed)[:count]
    if len(indices) != count:
        raise RuntimeError("sample stream contains too few passing proposals")
    return indices


@torch.no_grad()
def nuisance_diagnostics(
    endpoint: LinearActionEndpoint,
    queries: Tensor,
    means: Tensor,
    projector: HorizonLowRankProjector,
    *,
    alpha: float,
) -> NuisanceDiagnostics:
    """Measure only pre-rollout normalized action-proposal nuisances."""

    from xvla.train.post_norm_intervention import attenuate_queries

    aggregate = normalized_action_dose(
        endpoint, queries, means, projector, alpha=alpha, action_indices=range(endpoint.action_dim)
    )
    continuous = normalized_action_dose(
        endpoint, queries, means, projector, alpha=alpha, action_indices=range(endpoint.action_dim - 1)
    )
    gripper = normalized_action_dose(
        endpoint, queries, means, projector, alpha=alpha, action_indices=(endpoint.action_dim - 1,)
    )
    baseline = endpoint.normalized(queries)
    edited = endpoint.normalized(attenuate_queries(queries, means, projector, alpha=alpha))
    threshold = endpoint.normalized_gripper_threshold
    base_margin = baseline[..., -1] - threshold
    edited_margin = edited[..., -1] - threshold
    delta = edited[..., -1] - baseline[..., -1]
    return NuisanceDiagnostics(
        aggregate_rms=aggregate,
        continuous_rms=continuous,
        gripper_rms=gripper,
        gripper_flip_fraction=float(((base_margin > 0) != (edited_margin > 0)).double().mean().item()),
        gripper_margin_reachable_fraction=float((base_margin.abs() <= delta.abs()).double().mean().item()),
    )


def _diagnostics_from_unit_delta(
    unit_delta: Tensor,
    baseline: Tensor,
    threshold: Tensor,
    *,
    alpha: float,
) -> NuisanceDiagnostics:
    delta = unit_delta * alpha
    base_margin = baseline[..., -1] - threshold
    edited_margin = base_margin + delta[..., -1]
    return NuisanceDiagnostics(
        aggregate_rms=float(delta.double().square().mean().sqrt().item()),
        continuous_rms=float(delta[..., :-1].double().square().mean().sqrt().item()),
        gripper_rms=float(delta[..., -1:].double().square().mean().sqrt().item()),
        gripper_flip_fraction=float(((base_margin > 0) != (edited_margin > 0)).double().mean().item()),
        gripper_margin_reachable_fraction=float((base_margin.abs() <= delta[..., -1].abs()).double().mean().item()),
    )


def nuisance_matches(
    candidate: NuisanceDiagnostics,
    primary: NuisanceDiagnostics,
    *,
    aggregate_atol: float,
    allocation_rtol: float,
    incidence_atol: float,
    aggregate_rtol: float = 0.0,
) -> bool:
    if any(value < 0 or not torch.isfinite(torch.tensor(value)) for value in candidate.__dict__.values()):
        return False
    if (
        abs(candidate.aggregate_rms - primary.aggregate_rms)
        > aggregate_atol + aggregate_rtol * max(primary.aggregate_rms, 1e-30)
    ):
        return False
    for name in ("continuous_rms", "gripper_rms"):
        reference = getattr(primary, name)
        if abs(getattr(candidate, name) - reference) / max(reference, 1e-30) > allocation_rtol:
            return False
    for name in ("gripper_flip_fraction", "gripper_margin_reachable_fraction"):
        if abs(getattr(candidate, name) - getattr(primary, name)) > incidence_atol:
            return False
    return True


def _haar_projector(
    state_directions: Sequence[Tensor],
    analysis_directions: Sequence[Tensor],
    *,
    checkpoint_seed: int,
    proposal_index: int,
    label: str,
    namespace: str,
) -> tuple[HorizonLowRankProjector, tuple[int, ...]]:
    if not state_directions or len(state_directions) != len(analysis_directions):
        raise ValueError("balanced system factors must have equal nonzero horizon")
    states: list[Tensor] = []
    analyses: list[Tensor] = []
    seeds: list[int] = []
    for horizon, (state, analysis) in enumerate(zip(state_directions, analysis_directions)):
        if state.ndim != 2 or analysis.shape != (state.shape[1], state.shape[0]):
            raise ValueError("balanced system factor shapes differ")
        if namespace not in (DESIGN_NAMESPACE, SAMPLE_NAMESPACE):
            raise ValueError("Haar proposal namespace must be design or sample")
        seed = namespaced_seed(namespace, checkpoint_seed, proposal_index, horizon)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        modal = torch.randn(state.shape[1], 1, generator=generator, dtype=torch.float64)
        modal = modal / torch.linalg.vector_norm(modal)
        states.append(state.double().cpu() @ modal)
        analyses.append(modal.T @ analysis.double().cpu())
        seeds.append(seed)
    return HorizonLowRankProjector(torch.stack(states), torch.stack(analyses), label), tuple(seeds)


@torch.no_grad()
def select_conditional_haar_controls(
    endpoint: LinearActionEndpoint,
    selection_queries: Tensor,
    certification_queries: Tensor,
    means: Tensor,
    primary: HorizonLowRankProjector,
    state_directions: Sequence[Tensor],
    analysis_directions: Sequence[Tensor],
    *,
    checkpoint_seed: int,
    labels: Sequence[str],
    alpha_grid: Sequence[float],
    feasibility_pool_size: int,
    minimum_feasibility_eligible: int,
    sampling_pool_size: int,
    aggregate_atol: float = 1e-12,
    selection_allocation_rtol: float = 0.08,
    selection_incidence_atol: float = 0.008,
    certification_allocation_rtol: float = 0.10,
    certification_incidence_atol: float = 0.01,
) -> ConditionalHaarMatch:
    """Choose beta on a feasibility stream, then controls on a sample stream.

    The complete design stream estimates feasibility for each beta.  The
    largest beta with the predeclared minimum eligible count is selected.
    Independent sample-stream proposals are then scanned exactly once and the
    first five eligible draws are frozen.  Certification failure is terminal
    and never resumes either stream.  No rollout outcome is an argument.
    """

    names = tuple(labels)
    grid = tuple(float(value) for value in alpha_grid)
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("labels must be unique nonempty strings")
    if (
        feasibility_pool_size <= 0 or minimum_feasibility_eligible <= 0
        or minimum_feasibility_eligible > feasibility_pool_size
        or sampling_pool_size < len(names) or not grid
        or any(not 0 < value <= 1 for value in grid)
    ):
        raise ValueError("invalid feasibility/sample budgets or alpha grid")
    if any(left <= right for left, right in zip(grid, grid[1:])):
        raise ValueError("alpha grid must be strictly descending")
    if primary.label != "balanced_top":
        raise ValueError("the v2 primary must be balanced_top")
    primary_full = normalized_action_dose(
        endpoint, selection_queries, means, primary, alpha=1.0, action_indices=range(endpoint.action_dim)
    )
    if primary_full <= 0:
        raise ValueError("primary selection dose must be positive")

    baseline_selection = endpoint.normalized(selection_queries)
    threshold = endpoint.normalized_gripper_threshold
    primary_by_alpha = {
        alpha: nuisance_diagnostics(endpoint, selection_queries, means, primary, alpha=alpha)
        for alpha in grid
    }
    feasibility_counts = {alpha: 0 for alpha in grid}
    for proposal_index in range(feasibility_pool_size):
        candidate, _ = _haar_projector(
            state_directions, analysis_directions,
            checkpoint_seed=checkpoint_seed, proposal_index=proposal_index,
            label="feasibility", namespace=DESIGN_NAMESPACE,
        )
        unit_delta = normalized_action_delta(
            endpoint, selection_queries, means, candidate, alpha=1.0
        )
        unscaled = float(unit_delta.double().square().mean().sqrt().item())
        if unscaled <= 0:
            continue
        for primary_alpha in grid:
            target = primary_full * primary_alpha
            if target > unscaled * (1 + 1e-12):
                continue
            primary_selection = primary_by_alpha[primary_alpha]
            diagnostics = _diagnostics_from_unit_delta(
                unit_delta, baseline_selection, threshold, alpha=target / unscaled
            )
            if nuisance_matches(
                diagnostics, primary_selection, aggregate_atol=aggregate_atol,
                allocation_rtol=selection_allocation_rtol,
                incidence_atol=selection_incidence_atol,
            ):
                feasibility_counts[primary_alpha] += 1
    chosen_primary_alpha = select_largest_feasible_beta(
        grid, feasibility_counts, minimum_feasibility_eligible
    )
    target = primary_full * chosen_primary_alpha
    primary_selection = nuisance_diagnostics(
        endpoint, selection_queries, means, primary, alpha=chosen_primary_alpha
    )

    accepted: dict[str, HorizonLowRankProjector] = {}
    accepted_alpha: dict[str, float] = {}
    proposal_indices: dict[str, int] = {}
    proposal_seeds: dict[str, tuple[int, ...]] = {}
    selection_records: dict[str, NuisanceDiagnostics] = {}
    examined = 0
    sampling_eligibility: list[bool] = []
    selection_records = {primary.label: primary_selection}
    for proposal_index in range(sampling_pool_size):
        if len(accepted) == len(names):
            break
        label = names[len(accepted)]
        candidate, seeds = _haar_projector(
            state_directions, analysis_directions,
            checkpoint_seed=checkpoint_seed, proposal_index=proposal_index,
            label=label, namespace=SAMPLE_NAMESPACE,
        )
        examined += 1
        unit_delta = normalized_action_delta(
            endpoint, selection_queries, means, candidate, alpha=1.0
        )
        unscaled = float(unit_delta.double().square().mean().sqrt().item())
        if unscaled <= 0 or target > unscaled * (1 + 1e-12):
            sampling_eligibility.append(False)
            continue
        alpha = target / unscaled
        diagnostics = _diagnostics_from_unit_delta(
            unit_delta, baseline_selection, threshold, alpha=alpha
        )
        eligible = nuisance_matches(
            diagnostics, primary_selection, aggregate_atol=aggregate_atol,
            allocation_rtol=selection_allocation_rtol,
            incidence_atol=selection_incidence_atol,
        )
        sampling_eligibility.append(eligible)
        if eligible:
            accepted[label] = candidate
            accepted_alpha[label] = alpha
            proposal_indices[label] = proposal_index
            proposal_seeds[label] = seeds
            selection_records[label] = diagnostics
    if len(accepted) != len(names):
        raise RuntimeError("conditional-Haar sample stream exhausted on the selection split")
    if tuple(proposal_indices.values()) != first_passing_indices(sampling_eligibility, len(names)):
        raise RuntimeError("sampled controls are not exactly the first five passing proposals")

    projectors = {primary.label: primary, **accepted}
    alphas = {primary.label: chosen_primary_alpha, **accepted_alpha}
    certification = {
        name: nuisance_diagnostics(
            endpoint, certification_queries, means, operator, alpha=alphas[name]
        )
        for name, operator in projectors.items()
    }
    primary_certification = certification[primary.label]
    if any(
        not nuisance_matches(
            certification[name],
            primary_certification,
            aggregate_atol=aggregate_atol,
            aggregate_rtol=0.10,
            allocation_rtol=certification_allocation_rtol,
            incidence_atol=certification_incidence_atol,
        )
        for name in names
    ):
        raise RuntimeError("first-passing controls failed the disjoint certification split")
    return ConditionalHaarMatch(
        projectors=projectors,
        alphas=alphas,
        target_rms=target,
        primary_alpha=chosen_primary_alpha,
        proposal_indices=proposal_indices,
        proposal_seeds=proposal_seeds,
        selection=selection_records,
        certification=certification,
        proposals_examined=examined,
        feasibility_eligible_by_alpha=feasibility_counts,
        feasibility_pool_size=feasibility_pool_size,
        sampling_pool_size=sampling_pool_size,
        sampling_eligibility=tuple(sampling_eligibility),
    )
