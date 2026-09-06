"""Outcome-blind tensor controls for the fresh Stage 7 v3 protocol.

The proposal API accepts development folds only.  Certification is a separate
terminal function which cannot draw, replace, or retune a proposal.  This is a
mechanical boundary, not merely a convention in a launcher.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np
import torch

from athena.post_norm_intervention_v3_common import (
    ACTION_GROUPS,
    ALPHA_GRID,
    CERTIFICATION_GATES,
    CHECKPOINTS,
    DESIGN_NAMESPACE,
    DEVELOPMENT_FOLDS,
    DEVELOPMENT_GATES,
    HAAR_CONDITIONS,
    NUMERICAL_GATES,
    PANEL_NAMESPACE,
    PANEL_SPLIT_SEED,
    PRIMARY_TERMINAL,
    SAMPLE_NAMESPACE,
    canonical_sha256,
    namespaced_seed,
    tensor_sha256,
    validate_certification_attempt_token,
)
from xvla.train.post_norm_intervention import HorizonLowRankProjector, normalized_action_delta
from xvla.train.vla_terminal_balanced_odt import LinearActionEndpoint


Tensor = torch.Tensor


class ScientificInfeasibility(RuntimeError):
    """A typed scientific stop which orchestration must serialize verbatim."""

    def __init__(self, failure_class: str, reason: str, details: Mapping[str, object]):
        super().__init__(reason)
        self.failure_class = failure_class
        self.reason = reason
        self.details = dict(details)


class DesignInfeasible(ScientificInfeasibility):
    pass


class SampleExhausted(ScientificInfeasibility):
    pass


class CertificationInfeasible(ScientificInfeasibility):
    pass


class FreshPanelInfeasible(ScientificInfeasibility):
    pass


class Float32NumericalFailure(ArithmeticError):
    """Only an observed non-finite float32 diagnostic, never an engineering fault."""


@dataclass(frozen=True)
class PanelPartition:
    development_folds: tuple[tuple[int, ...], ...]
    certification: tuple[int, ...]
    development_episode_keys: tuple[tuple[tuple[int, str], ...], ...]
    certification_episode_keys: tuple[tuple[int, str], ...]
    excluded_sample_count: int
    excluded_footprint_count: int
    excluded_episode_count: int


@dataclass(frozen=True)
class DevelopmentFold:
    name: str
    queries: Tensor
    task_ids: tuple[int, ...]


@dataclass(frozen=True)
class ActionProfile:
    aggregate_rms: float
    coordinate_rms: tuple[float, ...]
    group_rms: dict[str, float]
    horizon_coordinate_rms: tuple[tuple[float, ...], ...]
    coordinate_second_moment: tuple[tuple[float, ...], ...]
    gripper_flip_fraction: float
    gripper_margin_reachable_fraction: float
    task_profiles: dict[int, "ActionProfile"]


@dataclass(frozen=True)
class GateResult:
    name: str
    actual: float
    reference: float
    threshold: float
    signed_margin: float
    passed: bool
    location: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "actual": self.actual,
            "reference": self.reference,
            "threshold": self.threshold,
            "signed_margin": self.signed_margin,
            "passed": self.passed,
            "location": self.location,
        }


@dataclass(frozen=True)
class FrozenDevelopmentProposals:
    primary: HorizonLowRankProjector
    controls: dict[str, HorizonLowRankProjector]
    alphas: dict[str, float]
    chosen_beta: float
    proposal_indices: dict[str, int]
    proposal_seeds: dict[str, tuple[int, ...]]
    development_profiles: dict[str, dict[str, ActionProfile]]
    development_gate_results: dict[str, tuple[GateResult, ...]]
    design_counts_by_alpha: dict[float, int]
    design_quintets_by_alpha: dict[float, int]
    proposals_examined: int
    sampling_eligibility: tuple[bool, ...]
    pairwise_projector_cosines: dict[str, float]
    numerical_gate_results: dict[str, tuple[GateResult, ...]]


@dataclass(frozen=True)
class FrozenCertificationPayload:
    """The deliberately narrow payload visible to the certification job."""

    projectors: dict[str, HorizonLowRankProjector]
    alphas: dict[str, float]
    operator_sha256: dict[str, str]
    proposal_freeze_sha256: str
    sealed_certification_data_sha256: str
    discovery_queries_sha256: str
    discovery_means_sha256: str
    content_sha256: str


@dataclass(frozen=True)
class CertifiedControls:
    frozen: FrozenCertificationPayload
    certification_profiles: dict[str, ActionProfile]
    certification_gate_results: dict[str, tuple[GateResult, ...]]


def fresh_episode_partition(
    task_ids: Sequence[int],
    episode_ids: Sequence[str],
    sample_ids: Sequence[str],
    frame_footprints: Sequence[Sequence[str]],
    *,
    excluded_sample_ids: Sequence[str],
    excluded_frame_footprints: Sequence[str],
    excluded_episode_ids: Sequence[tuple[int, str]],
) -> PanelPartition:
    """Build a fresh episode-disjoint development and sealed-certification panel.

    Membership uses identities only.  Any exact-window or raw-frame-footprint
    collision with the previously opened panel is excluded before partitioning.
    There is no fallback that silently reintroduces an excluded item.
    """

    tasks = tuple(task_ids)
    episodes = tuple(str(value) for value in episode_ids)
    samples = tuple(sample_ids)
    footprints = tuple(tuple(values) for values in frame_footprints)
    if (
        not tasks
        or len(tasks) != len(episodes)
        or len(tasks) != len(samples)
        or len(tasks) != len(footprints)
        or len(set(samples)) != len(samples)
        or set(tasks) != set(range(10))
    ):
        raise ValueError("fresh panel identity arrays differ or are invalid")
    excluded_samples = set(excluded_sample_ids)
    excluded_frames = set(excluded_frame_footprints)
    excluded_episode_values = tuple(excluded_episode_ids)
    excluded_episodes = set(excluded_episode_values)
    if (
        len(excluded_episodes) != len(excluded_episode_values)
        or any(
            not isinstance(item, tuple) or len(item) != 2
            or type(item[0]) is not int or item[0] not in range(10)
            or not isinstance(item[1], str) or not item[1]
            for item in excluded_episode_values
        )
    ):
        raise ValueError("excluded episode inventory is malformed or contains duplicates")
    eligible: list[int] = []
    excluded_sample_count = 0
    excluded_footprint_count = 0
    excluded_episode_count = 0
    eligible_frames: set[str] = set()
    for index, (task, episode, sample, footprint) in enumerate(
        zip(tasks, episodes, samples, footprints)
    ):
        if isinstance(task, bool) or not isinstance(task, int) or task < 0 or not episode or not sample:
            raise ValueError("fresh panel contains malformed task, episode, or sample identity")
        if not footprint or len(set(footprint)) != len(footprint):
            raise ValueError("fresh panel raw-frame footprint is empty or contains duplicates")
        if sample in excluded_samples:
            excluded_sample_count += 1
            continue
        if (task, episode) in excluded_episodes:
            excluded_episode_count += 1
            continue
        if set(footprint) & excluded_frames:
            excluded_footprint_count += 1
            continue
        overlap = set(footprint) & eligible_frames
        if overlap:
            raise FreshPanelInfeasible(
                "fresh_panel_infeasible",
                "new panel raw-frame footprints overlap",
                {"eligible_samples": len(eligible), "overlapping_frame_footprints": sorted(overlap)},
            )
        eligible_frames.update(footprint)
        eligible.append(index)

    by_task_episode: dict[int, dict[str, list[int]]] = {}
    for index in eligible:
        by_task_episode.setdefault(tasks[index], {}).setdefault(episodes[index], []).append(index)
    if not by_task_episode:
        raise FreshPanelInfeasible(
            "fresh_panel_infeasible", "no fresh episode remains after v2 exclusion",
            {"eligible_samples": 0, "per_task_episode_counts": {}},
        )

    development_indices: list[list[int]] = [[] for _ in range(DEVELOPMENT_FOLDS)]
    development_keys: list[list[tuple[int, str]]] = [[] for _ in range(DEVELOPMENT_FOLDS)]
    certification: list[int] = []
    certification_keys: list[tuple[int, str]] = []
    counts = {task: len(by_task_episode.get(task, {})) for task in range(10)}
    if any(count < DEVELOPMENT_FOLDS + 1 for count in counts.values()):
        raise FreshPanelInfeasible(
            "fresh_panel_infeasible",
            "too few fresh episodes for episode-disjoint development folds and certification",
            {"eligible_samples": len(eligible), "per_task_episode_counts": counts},
        )
    for task in sorted(by_task_episode):
        episode_map = by_task_episode[task]
        ordered = sorted(
            episode_map,
            key=lambda episode: hashlib.sha256(
                f"{PANEL_NAMESPACE}|{PANEL_SPLIT_SEED}|{task}|{episode}".encode()
            ).hexdigest(),
        )
        certification_count = max(1, len(ordered) // 5)
        certification_episodes = ordered[:certification_count]
        development_episodes = ordered[certification_count:]
        for episode in certification_episodes:
            certification.extend(episode_map[episode])
            certification_keys.append((task, episode))
        for position, episode in enumerate(development_episodes):
            fold = position % DEVELOPMENT_FOLDS
            development_indices[fold].extend(episode_map[episode])
            development_keys[fold].append((task, episode))

    if any(not fold for fold in development_indices) or not certification:
        raise FreshPanelInfeasible(
            "fresh_panel_infeasible", "fresh panel produced an empty partition",
            {"eligible_samples": len(eligible), "per_task_episode_counts": counts},
        )
    development_episode_sets = [set(keys) for keys in development_keys]
    certification_episode_set = set(certification_keys)
    if any(values & certification_episode_set for values in development_episode_sets):
        raise RuntimeError("development and certification episodes overlap")
    if any(
        development_episode_sets[left] & development_episode_sets[right]
        for left in range(DEVELOPMENT_FOLDS)
        for right in range(left + 1, DEVELOPMENT_FOLDS)
    ):
        raise RuntimeError("development fold episodes overlap")
    return PanelPartition(
        development_folds=tuple(tuple(sorted(values)) for values in development_indices),
        certification=tuple(sorted(certification)),
        development_episode_keys=tuple(tuple(sorted(values)) for values in development_keys),
        certification_episode_keys=tuple(sorted(certification_keys)),
        excluded_sample_count=excluded_sample_count,
        excluded_footprint_count=excluded_footprint_count,
        excluded_episode_count=excluded_episode_count,
    )


def _core_profile(delta: Tensor, baseline: Tensor, threshold: Tensor) -> ActionProfile:
    delta = delta.double()
    baseline = baseline.double()
    if delta.ndim != 3 or baseline.shape != delta.shape or delta.shape[-1] != 7:
        raise ValueError("action delta and baseline must have shape [sample,horizon,7]")
    if (
        not torch.isfinite(delta).all().item() or not torch.isfinite(baseline).all().item()
        or not torch.isfinite(threshold).all().item()
    ):
        raise ValueError("action delta or baseline contains a non-finite value")
    coordinate_rms = delta.square().mean(dim=(0, 1)).sqrt()
    horizon_coordinate_rms = delta.square().mean(dim=0).sqrt()
    flattened = delta.reshape(-1, delta.shape[-1])
    second_moment = flattened.T @ flattened / flattened.shape[0]
    base_margin = baseline[..., -1] - threshold.double()
    edited_margin = base_margin + delta[..., -1]
    group_rms = {
        name: float(delta[..., list(indices)].square().mean().sqrt().item())
        for name, indices in ACTION_GROUPS.items()
    }
    return ActionProfile(
        aggregate_rms=float(delta.square().mean().sqrt().item()),
        coordinate_rms=tuple(float(value) for value in coordinate_rms.tolist()),
        group_rms=group_rms,
        horizon_coordinate_rms=tuple(
            tuple(float(value) for value in row) for row in horizon_coordinate_rms.tolist()
        ),
        coordinate_second_moment=tuple(
            tuple(float(value) for value in row) for row in second_moment.tolist()
        ),
        gripper_flip_fraction=float(((base_margin > 0) != (edited_margin > 0)).double().mean().item()),
        gripper_margin_reachable_fraction=float(
            (base_margin.abs() <= delta[..., -1].abs()).double().mean().item()
        ),
        task_profiles={},
    )


def _profile_from_delta(
    delta: Tensor,
    baseline: Tensor,
    threshold: Tensor,
    task_ids: Sequence[int],
) -> ActionProfile:
    overall = _core_profile(delta, baseline, threshold)
    tasks = torch.as_tensor(tuple(task_ids), dtype=torch.long, device=delta.device)
    per_task: dict[int, ActionProfile] = {}
    for task in sorted(set(task_ids)):
        mask = tasks == task
        per_task[int(task)] = _core_profile(delta[mask], baseline[mask], threshold)
    return ActionProfile(
        aggregate_rms=overall.aggregate_rms,
        coordinate_rms=overall.coordinate_rms,
        group_rms=overall.group_rms,
        horizon_coordinate_rms=overall.horizon_coordinate_rms,
        coordinate_second_moment=overall.coordinate_second_moment,
        gripper_flip_fraction=overall.gripper_flip_fraction,
        gripper_margin_reachable_fraction=overall.gripper_margin_reachable_fraction,
        task_profiles=per_task,
    )


@torch.no_grad()
def action_profile(
    endpoint: LinearActionEndpoint,
    queries: Tensor,
    means: Tensor,
    projector: HorizonLowRankProjector,
    *,
    alpha: float,
    task_ids: Sequence[int],
) -> ActionProfile:
    if queries.ndim != 3 or len(task_ids) != queries.shape[0]:
        raise ValueError("queries and task ids differ")
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("attenuation alpha is invalid")
    delta = normalized_action_delta(endpoint, queries, means, projector, alpha=alpha)
    baseline = endpoint.normalized(queries)
    threshold = endpoint.normalized_gripper_threshold
    return _profile_from_delta(delta, baseline, threshold, task_ids)


def _relative(actual: Tensor, reference: Tensor, floor: float) -> float:
    numerator = torch.linalg.vector_norm(actual.double() - reference.double())
    denominator = max(float(torch.linalg.vector_norm(reference.double()).item()), floor, 1e-30)
    return float(numerator.item()) / denominator


def _profile_gate_specs(
    candidate: ActionProfile,
    reference: ActionProfile,
    *,
    gates: Mapping[str, float],
    location: str,
) -> Sequence[tuple[str, float, float, float, str]]:
    """Build the canonical elementwise comparisons for stored diagnostics."""

    required = {
        "aggregate_relative", "group_rms_relative", "coordinate_rms_relative",
        "horizon_coordinate_profile_relative", "coordinate_second_moment_relative",
        "gripper_incidence_absolute", "task_group_rms_relative",
        "task_horizon_coordinate_profile_relative", "task_gripper_incidence_absolute",
    }
    if not required <= set(gates):
        raise ValueError("action-profile gate inventory is incomplete")
    results: list[tuple[str, float, float, float, str]] = []

    def add(name: str, actual: float, reference_value: float, threshold: float, where: str) -> None:
        results.append((name, actual, reference_value, threshold, where))

    aggregate_relative = abs(candidate.aggregate_rms - reference.aggregate_rms) / max(
        reference.aggregate_rms, 1e-30
    )
    add("aggregate_relative", aggregate_relative, 0.0, gates["aggregate_relative"], location)
    for group in ACTION_GROUPS:
        relative = abs(candidate.group_rms[group] - reference.group_rms[group]) / max(
            reference.group_rms[group], reference.aggregate_rms * 1e-6, 1e-30
        )
        add("group_rms_relative", relative, 0.0, gates["group_rms_relative"], f"{location}/group={group}")
    for coordinate, (actual_value, reference_value) in enumerate(
        zip(candidate.coordinate_rms, reference.coordinate_rms)
    ):
        relative = abs(actual_value - reference_value) / max(
            reference_value, reference.aggregate_rms * 1e-6, 1e-30
        )
        add(
            "coordinate_rms_relative", relative, 0.0, gates["coordinate_rms_relative"],
            f"{location}/coordinate={coordinate}",
        )
    for horizon, (candidate_row, reference_row) in enumerate(
        zip(candidate.horizon_coordinate_rms, reference.horizon_coordinate_rms)
    ):
        for coordinate, (actual_value, reference_value) in enumerate(zip(candidate_row, reference_row)):
            relative = abs(actual_value - reference_value) / max(
                reference_value, reference.aggregate_rms * 1e-6, 1e-30
            )
            add(
                "horizon_coordinate_profile_relative", relative, 0.0,
                gates["horizon_coordinate_profile_relative"],
                f"{location}/horizon={horizon}/coordinate={coordinate}",
            )
    for row, (candidate_row, reference_row) in enumerate(
        zip(candidate.coordinate_second_moment, reference.coordinate_second_moment)
    ):
        for column, (actual_value, reference_value) in enumerate(zip(candidate_row, reference_row)):
            relative = abs(actual_value - reference_value) / max(
                abs(reference_value), reference.aggregate_rms ** 2 * 1e-6, 1e-30
            )
            add(
                "coordinate_second_moment_relative", relative, 0.0,
                gates["coordinate_second_moment_relative"],
                f"{location}/second_moment={row},{column}",
            )
    for name in ("gripper_flip_fraction", "gripper_margin_reachable_fraction"):
        difference = abs(getattr(candidate, name) - getattr(reference, name))
        add("gripper_incidence_absolute", difference, 0.0, gates["gripper_incidence_absolute"], f"{location}/{name}")

    if set(candidate.task_profiles) != set(reference.task_profiles):
        raise ValueError("candidate and reference task profile inventories differ")
    for task in sorted(reference.task_profiles):
        candidate_task = candidate.task_profiles[task]
        reference_task = reference.task_profiles[task]
        for group in ACTION_GROUPS:
            relative = abs(candidate_task.group_rms[group] - reference_task.group_rms[group]) / max(
                reference_task.group_rms[group], reference_task.aggregate_rms * 1e-6, 1e-30
            )
            add(
                "task_group_rms_relative", relative, 0.0,
                gates["task_group_rms_relative"], f"{location}/task={task}/group={group}",
            )
        for horizon, (candidate_row, reference_row) in enumerate(
            zip(candidate_task.horizon_coordinate_rms, reference_task.horizon_coordinate_rms)
        ):
            for coordinate, (actual_value, reference_value) in enumerate(zip(candidate_row, reference_row)):
                relative = abs(actual_value - reference_value) / max(
                    reference_value, reference_task.aggregate_rms * 1e-6, 1e-30
                )
                add(
                    "task_horizon_coordinate_profile_relative", relative, 0.0,
                    gates["task_horizon_coordinate_profile_relative"],
                    f"{location}/task={task}/horizon={horizon}/coordinate={coordinate}",
                )
        for name in ("gripper_flip_fraction", "gripper_margin_reachable_fraction"):
            difference = abs(getattr(candidate_task, name) - getattr(reference_task, name))
            add(
                "task_gripper_incidence_absolute", difference, 0.0,
                gates["task_gripper_incidence_absolute"], f"{location}/task={task}/{name}",
            )
    return tuple(results)


def action_profiles_pass(
    candidate: ActionProfile,
    reference: ActionProfile,
    *,
    gates: Mapping[str, float],
    location: str,
) -> bool:
    """Vector-check the same 728 elements without allocating gate records."""

    del location
    required = {
        "aggregate_relative", "group_rms_relative", "coordinate_rms_relative",
        "horizon_coordinate_profile_relative", "coordinate_second_moment_relative",
        "gripper_incidence_absolute", "task_group_rms_relative",
        "task_horizon_coordinate_profile_relative", "task_gripper_incidence_absolute",
    }
    if not required <= set(gates):
        raise ValueError("action-profile gate inventory is incomplete")
    if set(candidate.task_profiles) != set(reference.task_profiles):
        raise ValueError("candidate and reference task profile inventories differ")

    def within(actual: np.ndarray | float, threshold: float) -> bool:
        values = np.asarray(actual, dtype=np.float64)
        return bool(np.isfinite(values).all() and np.less_equal(values, threshold).all())

    if not within(
        abs(candidate.aggregate_rms - reference.aggregate_rms)
        / max(reference.aggregate_rms, 1e-30),
        gates["aggregate_relative"],
    ):
        return False
    candidate_groups = np.asarray([candidate.group_rms[group] for group in ACTION_GROUPS])
    reference_groups = np.asarray([reference.group_rms[group] for group in ACTION_GROUPS])
    if not within(
        np.abs(candidate_groups - reference_groups)
        / np.maximum(reference_groups, max(reference.aggregate_rms * 1e-6, 1e-30)),
        gates["group_rms_relative"],
    ):
        return False
    candidate_coordinates = np.asarray(candidate.coordinate_rms)
    reference_coordinates = np.asarray(reference.coordinate_rms)
    if not within(
        np.abs(candidate_coordinates - reference_coordinates)
        / np.maximum(reference_coordinates, max(reference.aggregate_rms * 1e-6, 1e-30)),
        gates["coordinate_rms_relative"],
    ):
        return False
    candidate_horizon = np.asarray(candidate.horizon_coordinate_rms)
    reference_horizon = np.asarray(reference.horizon_coordinate_rms)
    if not within(
        np.abs(candidate_horizon - reference_horizon)
        / np.maximum(reference_horizon, max(reference.aggregate_rms * 1e-6, 1e-30)),
        gates["horizon_coordinate_profile_relative"],
    ):
        return False
    candidate_second = np.asarray(candidate.coordinate_second_moment)
    reference_second = np.asarray(reference.coordinate_second_moment)
    if not within(
        np.abs(candidate_second - reference_second)
        / np.maximum(np.abs(reference_second), max(reference.aggregate_rms ** 2 * 1e-6, 1e-30)),
        gates["coordinate_second_moment_relative"],
    ):
        return False
    if not within(np.asarray([
        abs(candidate.gripper_flip_fraction - reference.gripper_flip_fraction),
        abs(candidate.gripper_margin_reachable_fraction - reference.gripper_margin_reachable_fraction),
    ]), gates["gripper_incidence_absolute"]):
        return False

    tasks = sorted(reference.task_profiles)
    task_candidate_groups = np.asarray([
        [candidate.task_profiles[task].group_rms[group] for group in ACTION_GROUPS]
        for task in tasks
    ])
    task_reference_groups = np.asarray([
        [reference.task_profiles[task].group_rms[group] for group in ACTION_GROUPS]
        for task in tasks
    ])
    task_scales = np.asarray([
        max(reference.task_profiles[task].aggregate_rms * 1e-6, 1e-30) for task in tasks
    ])[:, None]
    if not within(
        np.abs(task_candidate_groups - task_reference_groups)
        / np.maximum(task_reference_groups, task_scales),
        gates["task_group_rms_relative"],
    ):
        return False
    task_candidate_horizon = np.asarray([
        candidate.task_profiles[task].horizon_coordinate_rms for task in tasks
    ])
    task_reference_horizon = np.asarray([
        reference.task_profiles[task].horizon_coordinate_rms for task in tasks
    ])
    if not within(
        np.abs(task_candidate_horizon - task_reference_horizon)
        / np.maximum(task_reference_horizon, task_scales[:, :, None]),
        gates["task_horizon_coordinate_profile_relative"],
    ):
        return False
    task_gripper = np.asarray([
        (
            abs(candidate.task_profiles[task].gripper_flip_fraction
                - reference.task_profiles[task].gripper_flip_fraction),
            abs(candidate.task_profiles[task].gripper_margin_reachable_fraction
                - reference.task_profiles[task].gripper_margin_reachable_fraction),
        )
        for task in tasks
    ])
    return within(task_gripper, gates["task_gripper_incidence_absolute"])


def compare_action_profiles(
    candidate: ActionProfile,
    reference: ActionProfile,
    *,
    gates: Mapping[str, float],
    location: str,
) -> tuple[GateResult, ...]:
    """Return every gate, including all failures, for quantitative receipts."""

    results: list[GateResult] = []
    for name, actual, reference_value, threshold, where in _profile_gate_specs(
        candidate, reference, gates=gates, location=location,
    ):
        finite = math.isfinite(actual) and math.isfinite(reference_value)
        margin = threshold - actual if finite else -math.inf
        results.append(
            GateResult(name, actual, reference_value, threshold, margin, finite and margin >= 0, where)
        )
    return tuple(results)


def projector_cosine(left: HorizonLowRankProjector, right: HorizonLowRankProjector) -> float:
    if left.rank != 1 or right.rank != 1:
        raise ValueError("v3 projector separation is defined for the frozen rank-one design")
    if left.horizon != right.horizon or left.hidden_dim != right.hidden_dim:
        raise ValueError("projector shapes differ")
    left_state = left.state_factors[..., 0].double()
    right_state = right.state_factors[..., 0].double()
    left_analysis = left.analysis_factors[:, 0, :].double()
    right_analysis = right.analysis_factors[:, 0, :].double()
    numerator = (
        (left_state * right_state).sum(dim=1)
        * (left_analysis * right_analysis).sum(dim=1)
    ).abs()
    denominator = (
        torch.linalg.vector_norm(left_state, dim=1)
        * torch.linalg.vector_norm(left_analysis, dim=1)
        * torch.linalg.vector_norm(right_state, dim=1)
        * torch.linalg.vector_norm(right_analysis, dim=1)
    )
    if bool((denominator <= 0).any()):
        raise ValueError("projector has zero Frobenius norm")
    return float((numerator / denominator).max().item())


def _haar_projector(
    state_directions: Sequence[Tensor],
    analysis_directions: Sequence[Tensor],
    *,
    checkpoint_seed: int,
    proposal_index: int,
    label: str,
    namespace: str,
) -> tuple[HorizonLowRankProjector, tuple[int, ...]]:
    if namespace not in (DESIGN_NAMESPACE, SAMPLE_NAMESPACE):
        raise ValueError("proposal namespace must be the fresh v3 design or sample stream")
    if not state_directions or len(state_directions) != len(analysis_directions):
        raise ValueError("balanced system factors must have equal nonzero horizon")
    states: list[Tensor] = []
    analyses: list[Tensor] = []
    seeds: list[int] = []
    for horizon, (state, analysis) in enumerate(zip(state_directions, analysis_directions)):
        if state.ndim != 2 or analysis.shape != (state.shape[1], state.shape[0]):
            raise ValueError("balanced system factor shapes differ")
        seed = namespaced_seed(namespace, checkpoint_seed, proposal_index, horizon)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        modal = torch.randn(state.shape[1], 1, generator=generator, dtype=torch.float64)
        modal = modal / torch.linalg.vector_norm(modal)
        states.append(state.double().cpu() @ modal)
        analyses.append(modal.T @ analysis.double().cpu())
        seeds.append(seed)
    return HorizonLowRankProjector(torch.stack(states), torch.stack(analyses), label), tuple(seeds)


def _validate_folds(folds: Sequence[DevelopmentFold], means: Tensor) -> tuple[DevelopmentFold, ...]:
    values = tuple(folds)
    expected_names = tuple(f"development_{index}" for index in range(DEVELOPMENT_FOLDS))
    if len(values) != DEVELOPMENT_FOLDS or tuple(fold.name for fold in values) != expected_names:
        raise ValueError("the exact four ordered development folds are required")
    for fold in values:
        if fold.queries.ndim != 3 or fold.queries.shape[1:] != means.shape:
            raise ValueError("development query or mean shape differs")
        if fold.queries.shape[0] != len(fold.task_ids) or not fold.task_ids:
            raise ValueError("development query and task inventory differs")
        if set(fold.task_ids) != set(range(10)):
            raise ValueError("every development fold must contain the exact ten-task inventory")
    return values


def projector_sha256(projector: HorizonLowRankProjector) -> str:
    return canonical_sha256({
        "label": projector.label,
        "state_factors": tensor_sha256(projector.state_factors),
        "analysis_factors": tensor_sha256(projector.analysis_factors),
    })


def _factor_dense_action_replay_error(
    endpoint: LinearActionEndpoint,
    queries: Tensor,
    means: Tensor,
    projector: HorizonLowRankProjector,
    alpha: float,
) -> float:
    factor_delta = normalized_action_delta(endpoint, queries, means, projector, alpha=alpha)
    centered = queries - means.to(queries)
    dense = projector.to(dtype=queries.dtype, device=queries.device).dense()
    removed = torch.einsum("hdk,bhk->bhd", dense, centered)
    dense_delta = torch.einsum(
        "ad,bhd->bha", endpoint.weight_normalized.to(queries), -alpha * removed
    )
    return float((factor_delta - dense_delta).abs().max().item())


def _raw_factor_delta(
    endpoint: LinearActionEndpoint, queries: Tensor, means: Tensor,
    state_factors: Tensor, analysis_factors: Tensor, alpha: float,
) -> Tensor:
    centered = queries - means.to(queries)
    coordinates = torch.einsum("hrd,bhd->bhr", analysis_factors, centered)
    removed = torch.einsum("hdr,bhr->bhd", state_factors, coordinates)
    return torch.einsum(
        "ad,bhd->bha", endpoint.weight_normalized.to(queries), -alpha * removed,
    )


def _raw_pooled_unit_rms(
    endpoint: LinearActionEndpoint, folds: Sequence[DevelopmentFold], means: Tensor,
    state_factors: Tensor, analysis_factors: Tensor,
) -> float:
    deltas = [
        _raw_factor_delta(
            endpoint, fold.queries, means, state_factors, analysis_factors, 1.0,
        )
        for fold in folds
    ]
    return float(torch.cat(deltas, dim=0).double().square().mean().sqrt().item())


@torch.no_grad()
def numerical_operator_gate_results(
    endpoint64: LinearActionEndpoint,
    endpoint32: LinearActionEndpoint,
    folds64: Sequence[DevelopmentFold],
    means64: Tensor,
    projectors: Mapping[str, HorizonLowRankProjector],
    alphas: Mapping[str, float],
) -> dict[str, tuple[GateResult, ...]]:
    """Gate float64/float32 duality, replay, idempotence, and realized dose."""

    folds = _validate_folds(folds64, means64)
    if set(projectors) != set(alphas) or PRIMARY_TERMINAL not in projectors:
        raise ValueError("numerical-gate operator inventory differs")
    primary_dose64 = _pooled_unit_rms(endpoint64, folds, means64, projectors[PRIMARY_TERMINAL]) * alphas[PRIMARY_TERMINAL]
    results: dict[str, tuple[GateResult, ...]] = {}
    for label, projector in projectors.items():
        alpha = alphas[label]
        operator64 = projector.to(dtype=torch.float64, device="cpu")
        dense64 = operator64.dense()
        dual64 = torch.einsum("hrd,hds->hrs", operator64.analysis_factors, operator64.state_factors)
        identity64 = torch.eye(operator64.rank, dtype=torch.float64).expand(operator64.horizon, -1, -1)
        dual64_error = float((dual64 - identity64).abs().max().item())
        idempotence64_error = float((dense64 @ dense64 - dense64).abs().max().item())
        replay64_error = max(
            _factor_dense_action_replay_error(
                endpoint64, fold.queries.double().cpu(), means64.double().cpu(), operator64, alpha
            )
            for fold in folds
        )
        dose64 = _pooled_unit_rms(endpoint64, folds, means64, operator64) * alpha

        float32_thresholds = (
            NUMERICAL_GATES["float32_dual_and_idempotence_max_abs"],
            NUMERICAL_GATES["float32_dual_and_idempotence_max_abs"],
            NUMERICAL_GATES["float32_dual_and_idempotence_max_abs"],
            NUMERICAL_GATES["float32_aggregate_relative_error"],
        )
        try:
            state32 = operator64.state_factors.float().cpu()
            analysis32 = operator64.analysis_factors.float().cpu()
            if not torch.isfinite(state32).all().item() or not torch.isfinite(analysis32).all().item():
                raise Float32NumericalFailure("float32 factor cast is non-finite")
            dense32 = torch.einsum("hdr,hrk->hdk", state32, analysis32)
            dual32 = torch.einsum("hrd,hds->hrs", analysis32, state32)
            identity32 = torch.eye(operator64.rank, dtype=torch.float32).expand(operator64.horizon, -1, -1)
            dual32_error = float((dual32 - identity32).abs().max().item())
            idempotence32_error = float((dense32 @ dense32 - dense32).abs().max().item())
            folds32 = tuple(
                DevelopmentFold(fold.name, fold.queries.float().cpu(), fold.task_ids) for fold in folds
            )
            replay32_error = 0.0
            for fold in folds32:
                factor_delta = _raw_factor_delta(
                    endpoint32, fold.queries, means64.float().cpu(), state32, analysis32, alpha,
                )
                centered = fold.queries - means64.float().cpu().to(fold.queries)
                removed = torch.einsum("hdk,bhk->bhd", dense32, centered)
                dense_delta = torch.einsum(
                    "ad,bhd->bha", endpoint32.weight_normalized.to(fold.queries), -alpha * removed,
                )
                replay32_error = max(
                    replay32_error, float((factor_delta - dense_delta).abs().max().item()),
                )
            dose32 = _raw_pooled_unit_rms(
                endpoint32, folds32, means64.float().cpu(), state32, analysis32,
            ) * alpha
            dose32_relative = abs(dose32 - dose64) / max(dose64, 1e-30)
            float32_actuals = (
                dual32_error, idempotence32_error, replay32_error, dose32_relative,
            )
            if any(not math.isfinite(actual) or actual < 0 for actual in float32_actuals):
                raise Float32NumericalFailure("float32 numerical diagnostic is invalid")
        except Float32NumericalFailure:
            # Receipts forbid NaN and infinity.  A deterministic finite value above
            # each fixed threshold records the typed numerical failure instead.
            float32_actuals = tuple(threshold + 1.0 for threshold in float32_thresholds)
        dual32_error, idempotence32_error, replay32_error, dose32_relative = float32_actuals

        values = (
            ("float64_dual", dual64_error, NUMERICAL_GATES["float64_dual_and_idempotence_max_abs"]),
            ("float64_idempotence", idempotence64_error, NUMERICAL_GATES["float64_dual_and_idempotence_max_abs"]),
            ("float64_factor_dense_action_replay", replay64_error, NUMERICAL_GATES["float64_dual_and_idempotence_max_abs"]),
            (
                "float64_pooled_development_aggregate",
                abs(dose64 - primary_dose64),
                NUMERICAL_GATES["float64_pooled_development_aggregate_max_abs"],
            ),
            ("float32_dual", dual32_error, NUMERICAL_GATES["float32_dual_and_idempotence_max_abs"]),
            ("float32_idempotence", idempotence32_error, NUMERICAL_GATES["float32_dual_and_idempotence_max_abs"]),
            ("float32_factor_dense_action_replay", replay32_error, NUMERICAL_GATES["float32_dual_and_idempotence_max_abs"]),
            ("float32_aggregate_relative_error", dose32_relative, NUMERICAL_GATES["float32_aggregate_relative_error"]),
        )
        if any(not math.isfinite(actual) or actual < 0 for _, actual, _ in values):
            raise ValueError("v3 numerical gate produced a non-finite or negative diagnostic")
        results[label] = tuple(
            GateResult(
                name=name, actual=actual, reference=0.0, threshold=threshold,
                signed_margin=threshold - actual, passed=actual <= threshold,
                location=f"development/{label}",
            )
            for name, actual, threshold in values
        )
    return results


def _candidate_development_evaluation(
    endpoint: LinearActionEndpoint,
    folds: Sequence[DevelopmentFold],
    means: Tensor,
    candidate: HorizonLowRankProjector,
    primary_profiles: Mapping[str, ActionProfile],
    *,
    alpha: float,
) -> tuple[bool, dict[str, ActionProfile], tuple[GateResult, ...]]:
    profiles: dict[str, ActionProfile] = {}
    gates: list[GateResult] = []
    for fold in folds:
        profile = action_profile(
            endpoint, fold.queries, means, candidate, alpha=alpha, task_ids=fold.task_ids
        )
        profiles[fold.name] = profile
        gates.extend(compare_action_profiles(
            profile, primary_profiles[fold.name], gates=DEVELOPMENT_GATES, location=fold.name
        ))
    return all(gate.passed for gate in gates), profiles, tuple(gates)


def _unit_deltas_by_fold(
    endpoint: LinearActionEndpoint,
    folds: Sequence[DevelopmentFold],
    means: Tensor,
    projector: HorizonLowRankProjector,
) -> dict[str, Tensor]:
    return {
        fold.name: normalized_action_delta(
            endpoint, fold.queries, means, projector, alpha=1.0
        ).double()
        for fold in folds
    }


def _pooled_rms_from_unit_deltas(
    folds: Sequence[DevelopmentFold], unit_deltas: Mapping[str, Tensor]
) -> float:
    if set(unit_deltas) != {fold.name for fold in folds}:
        raise ValueError("unit-delta fold inventory differs")
    return float(
        torch.cat([unit_deltas[fold.name] for fold in folds], dim=0)
        .square().mean().sqrt().item()
    )


def _cached_candidate_development_evaluation(
    endpoint: LinearActionEndpoint,
    folds: Sequence[DevelopmentFold],
    unit_deltas: Mapping[str, Tensor],
    primary_profiles: Mapping[str, ActionProfile],
    *,
    alpha: float,
) -> tuple[bool, dict[str, ActionProfile], tuple[GateResult, ...]]:
    profiles: dict[str, ActionProfile] = {}
    gates: list[GateResult] = []
    threshold = endpoint.normalized_gripper_threshold
    for fold in folds:
        profile = _profile_from_delta(
            unit_deltas[fold.name] * alpha,
            endpoint.normalized(fold.queries),
            threshold,
            fold.task_ids,
        )
        profiles[fold.name] = profile
        gates.extend(compare_action_profiles(
            profile, primary_profiles[fold.name], gates=DEVELOPMENT_GATES, location=fold.name
        ))
    return all(gate.passed for gate in gates), profiles, tuple(gates)


def _cached_candidate_development_passes(
    endpoint: LinearActionEndpoint,
    folds: Sequence[DevelopmentFold],
    unit_deltas: Mapping[str, Tensor],
    primary_profiles: Mapping[str, ActionProfile],
    *,
    alpha: float,
) -> bool:
    """Allocation-light but elementwise-identical design-screen predicate."""

    threshold = endpoint.normalized_gripper_threshold
    for fold in folds:
        profile = _profile_from_delta(
            unit_deltas[fold.name] * alpha,
            endpoint.normalized(fold.queries),
            threshold,
            fold.task_ids,
        )
        if not action_profiles_pass(
            profile, primary_profiles[fold.name], gates=DEVELOPMENT_GATES,
            location=fold.name,
        ):
            return False
    return True


def _pooled_unit_rms(
    endpoint: LinearActionEndpoint,
    folds: Sequence[DevelopmentFold],
    means: Tensor,
    projector: HorizonLowRankProjector,
) -> float:
    return _pooled_rms_from_unit_deltas(
        folds, _unit_deltas_by_fold(endpoint, folds, means, projector)
    )


@torch.no_grad()
def _freeze_development_proposals(
    endpoint: LinearActionEndpoint,
    deployed_float32_endpoint: LinearActionEndpoint,
    development_folds: Sequence[DevelopmentFold],
    means: Tensor,
    primary: HorizonLowRankProjector,
    state_directions: Sequence[Tensor],
    analysis_directions: Sequence[Tensor],
    *,
    checkpoint_seed: int,
    labels: Sequence[str] = HAAR_CONDITIONS,
    alpha_grid: Sequence[float] = ALPHA_GRID,
    feasibility_pool_size: int = 16_384,
    minimum_feasible_quintets: int = 32,
    sampling_pool_size: int = 65_536,
) -> FrozenDevelopmentProposals:
    """Freeze controls using development folds only.

    The design stream counts complete, pairwise-separated quintets which pass
    every aggregate, coordinate, horizon, covariance, incidence, and per-task
    development gate.  The sample stream freezes the first such quintet.  No
    certification tensor is accepted by this function.
    """

    folds = _validate_folds(development_folds, means)
    names = tuple(labels)
    grid = tuple(float(value) for value in alpha_grid)
    if names != HAAR_CONDITIONS:
        raise ValueError("v3 requires the exact five-control inventory")
    if primary.label != PRIMARY_TERMINAL:
        raise ValueError("v3 is hard-disabled for label-only or unavailable global ODT primaries")
    if (
        not grid or any(not 0 < value <= 1 for value in grid)
        or any(left <= right for left, right in zip(grid, grid[1:]))
        or feasibility_pool_size <= 0 or minimum_feasible_quintets <= 0
        or sampling_pool_size < len(names)
    ):
        raise ValueError("v3 design budgets or alpha grid are invalid")

    primary_unit_deltas = _unit_deltas_by_fold(endpoint, folds, means, primary)
    primary_unit_rms = _pooled_rms_from_unit_deltas(folds, primary_unit_deltas)
    if primary_unit_rms <= 0:
        raise ValueError("primary development dose must be positive")
    primary_profiles_by_alpha = {
        alpha: {
            fold.name: _profile_from_delta(
                primary_unit_deltas[fold.name] * alpha,
                endpoint.normalized(fold.queries),
                endpoint.normalized_gripper_threshold,
                fold.task_ids,
            )
            for fold in folds
        }
        for alpha in grid
    }
    design_counts = {alpha: 0 for alpha in grid}
    design_quintets = {alpha: 0 for alpha in grid}
    partial_quintets: dict[float, list[HorizonLowRankProjector]] = {alpha: [] for alpha in grid}
    for proposal_index in range(feasibility_pool_size):
        candidate, _ = _haar_projector(
            state_directions, analysis_directions, checkpoint_seed=checkpoint_seed,
            proposal_index=proposal_index, label="design", namespace=DESIGN_NAMESPACE,
        )
        if projector_cosine(candidate, primary) > DEVELOPMENT_GATES["candidate_primary_projector_cosine_max"]:
            continue
        unit_deltas = _unit_deltas_by_fold(endpoint, folds, means, candidate)
        unscaled = _pooled_rms_from_unit_deltas(folds, unit_deltas)
        if unscaled <= 0:
            continue
        for beta in grid:
            target = primary_unit_rms * beta
            if target > unscaled:
                continue
            alpha = target / unscaled
            eligible = _cached_candidate_development_passes(
                endpoint, folds, unit_deltas, primary_profiles_by_alpha[beta], alpha=alpha
            )
            if not eligible:
                continue
            design_counts[beta] += 1
            current = partial_quintets[beta]
            if any(
                projector_cosine(candidate, accepted)
                > DEVELOPMENT_GATES["candidate_pair_projector_cosine_max"]
                for accepted in current
            ):
                continue
            current.append(candidate)
            if len(current) == len(names):
                design_quintets[beta] += 1
                partial_quintets[beta] = []

    chosen_beta = next(
        (beta for beta in grid if design_quintets[beta] >= minimum_feasible_quintets), None
    )
    if chosen_beta is None:
        feasibility_gates = tuple(
            GateResult(
                name="ordered_greedy_design_quintet_shortfall",
                actual=float(max(0, minimum_feasible_quintets - design_quintets[beta])),
                reference=0.0,
                threshold=0.0,
                signed_margin=-float(max(0, minimum_feasible_quintets - design_quintets[beta])),
                passed=design_quintets[beta] >= minimum_feasible_quintets,
                location=f"development_design/beta={beta}",
            ).as_dict()
            for beta in grid
        )
        raise DesignInfeasible(
            "ordered_greedy_design_shortfall",
            "ordered greedy design stream formed fewer than the required disjoint quintets",
            {
                "design_counts_by_alpha": design_counts,
                "design_quintets_by_alpha": design_quintets,
                "minimum_feasible_quintets": minimum_feasible_quintets,
                "feasibility_pool_size": feasibility_pool_size,
                "gate_results": feasibility_gates,
            },
        )

    primary_profiles = primary_profiles_by_alpha[chosen_beta]
    target = primary_unit_rms * chosen_beta
    accepted: list[HorizonLowRankProjector] = []
    controls: dict[str, HorizonLowRankProjector] = {}
    alphas: dict[str, float] = {primary.label: chosen_beta}
    indices: dict[str, int] = {}
    seeds_by_label: dict[str, tuple[int, ...]] = {}
    profiles: dict[str, dict[str, ActionProfile]] = {primary.label: dict(primary_profiles)}
    gate_results: dict[str, tuple[GateResult, ...]] = {}
    sampling_eligibility: list[bool] = []
    pairwise_cosines: dict[str, float] = {}
    proposals_examined = 0
    for proposal_index in range(sampling_pool_size):
        if len(accepted) == len(names):
            break
        label = names[len(accepted)]
        candidate, seeds = _haar_projector(
            state_directions, analysis_directions, checkpoint_seed=checkpoint_seed,
            proposal_index=proposal_index, label=label, namespace=SAMPLE_NAMESPACE,
        )
        proposals_examined += 1
        primary_cosine = projector_cosine(candidate, primary)
        unit_deltas = _unit_deltas_by_fold(endpoint, folds, means, candidate)
        unscaled = _pooled_rms_from_unit_deltas(folds, unit_deltas)
        eligible = (
            primary_cosine <= DEVELOPMENT_GATES["candidate_primary_projector_cosine_max"]
            and unscaled > 0 and target <= unscaled
        )
        candidate_profiles: dict[str, ActionProfile] = {}
        candidate_gates: tuple[GateResult, ...] = ()
        alpha = target / unscaled if eligible else math.nan
        if eligible:
            eligible, candidate_profiles, candidate_gates = _cached_candidate_development_evaluation(
                endpoint, folds, unit_deltas, primary_profiles, alpha=alpha
            )
        pair_cosines = [projector_cosine(candidate, other) for other in accepted]
        if eligible and any(
            value > DEVELOPMENT_GATES["candidate_pair_projector_cosine_max"]
            for value in pair_cosines
        ):
            eligible = False
        sampling_eligibility.append(bool(eligible))
        if not eligible:
            continue
        accepted.append(candidate)
        controls[label] = candidate
        alphas[label] = alpha
        indices[label] = proposal_index
        seeds_by_label[label] = seeds
        profiles[label] = candidate_profiles
        gate_results[label] = candidate_gates
        pairwise_cosines[f"{label}|{primary.label}"] = primary_cosine
        for other_label, cosine in zip(tuple(controls)[:-1], pair_cosines):
            pairwise_cosines[f"{label}|{other_label}"] = cosine

    if len(accepted) != len(names):
        shortfall = len(names) - len(accepted)
        raise SampleExhausted(
            "ordered_greedy_sample_shortfall",
            "ordered greedy sample construction did not complete a development-passing quintet",
            {
                "chosen_beta": chosen_beta,
                "design_counts_by_alpha": design_counts,
                "design_quintets_by_alpha": design_quintets,
                "proposals_examined": proposals_examined,
                "sampling_pool_size": sampling_pool_size,
                "sampling_eligibility": tuple(sampling_eligibility),
                "selected_proposal_indices": dict(indices),
                "selected_proposal_seeds": dict(seeds_by_label),
                "selected_operator_sha256": {
                    label: projector_sha256(operator) for label, operator in controls.items()
                },
                "gate_results": (
                    GateResult(
                        name="ordered_greedy_sample_quintet_shortfall",
                        actual=float(shortfall), reference=0.0, threshold=0.0,
                        signed_margin=-float(shortfall), passed=False,
                        location="development_sample_stream",
                    ).as_dict(),
                ),
            },
        )
    all_projectors = {primary.label: primary, **controls}
    numerical_gates = numerical_operator_gate_results(
        endpoint, deployed_float32_endpoint, folds, means, all_projectors, alphas
    )
    numerical_failures = [
        gate for label_gates in numerical_gates.values() for gate in label_gates if not gate.passed
    ]
    if numerical_failures:
        offender = min(numerical_failures, key=lambda gate: gate.signed_margin)
        raise DesignInfeasible(
            "numerical_gate_infeasible", "frozen controls failed a deployed numerical gate",
            {
                "chosen_beta": chosen_beta,
                "design_counts_by_alpha": design_counts,
                "design_quintets_by_alpha": design_quintets,
                "proposals_examined": proposals_examined,
                "sampling_eligibility": tuple(sampling_eligibility),
                "selected_proposal_indices": dict(indices),
                "selected_proposal_seeds": dict(seeds_by_label),
                "selected_operator_sha256": {
                    label: projector_sha256(operator) for label, operator in controls.items()
                },
                "gate_results": tuple(
                    gate.as_dict() for label_gates in numerical_gates.values() for gate in label_gates
                ),
                "maximum_offender": offender.as_dict(),
            },
        )
    return FrozenDevelopmentProposals(
        primary=primary,
        controls=controls,
        alphas=alphas,
        chosen_beta=chosen_beta,
        proposal_indices=indices,
        proposal_seeds=seeds_by_label,
        development_profiles=profiles,
        development_gate_results=gate_results,
        design_counts_by_alpha=design_counts,
        design_quintets_by_alpha=design_quintets,
        proposals_examined=proposals_examined,
        sampling_eligibility=tuple(sampling_eligibility),
        pairwise_projector_cosines=pairwise_cosines,
        numerical_gate_results=numerical_gates,
    )


def freeze_development_proposals(
    endpoint: LinearActionEndpoint,
    deployed_float32_endpoint: LinearActionEndpoint,
    development_folds: Sequence[DevelopmentFold],
    means: Tensor,
    primary: HorizonLowRankProjector,
    state_directions: Sequence[Tensor],
    analysis_directions: Sequence[Tensor],
    *,
    checkpoint_seed: int,
) -> FrozenDevelopmentProposals:
    """Production entrypoint with every v3 design constant hard-coded."""

    return _freeze_development_proposals(
        endpoint,
        deployed_float32_endpoint,
        development_folds,
        means,
        primary,
        state_directions,
        analysis_directions,
        checkpoint_seed=checkpoint_seed,
        labels=HAAR_CONDITIONS,
        alpha_grid=ALPHA_GRID,
        feasibility_pool_size=16_384,
        minimum_feasible_quintets=32,
        sampling_pool_size=65_536,
    )


def make_certification_payload(
    frozen: FrozenDevelopmentProposals,
    *,
    proposal_freeze_sha256: str,
    sealed_certification_data_sha256: str,
    discovery_queries_sha256: str,
    discovery_means_sha256: str,
) -> FrozenCertificationPayload:
    """Strip proposal seeds, budgets, counts, and eligibility before lockbox access."""

    for label, digest in {
        "proposal-freeze": proposal_freeze_sha256,
        "sealed certification data": sealed_certification_data_sha256,
        "discovery queries": discovery_queries_sha256,
        "discovery means": discovery_means_sha256,
    }.items():
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{label} SHA256 is malformed")
    projectors = {frozen.primary.label: frozen.primary, **frozen.controls}
    if tuple(projectors) != (PRIMARY_TERMINAL, *HAAR_CONDITIONS):
        raise ValueError("certification payload operator inventory differs")
    operator_hashes = {label: projector_sha256(operator) for label, operator in projectors.items()}
    content = canonical_sha256({
        "alphas": frozen.alphas,
        "operator_sha256": operator_hashes,
        "proposal_freeze_sha256": proposal_freeze_sha256,
        "sealed_certification_data_sha256": sealed_certification_data_sha256,
        "discovery_queries_sha256": discovery_queries_sha256,
        "discovery_means_sha256": discovery_means_sha256,
    })
    return FrozenCertificationPayload(
        projectors=projectors,
        alphas=dict(frozen.alphas),
        operator_sha256=operator_hashes,
        proposal_freeze_sha256=proposal_freeze_sha256,
        sealed_certification_data_sha256=sealed_certification_data_sha256,
        discovery_queries_sha256=discovery_queries_sha256,
        discovery_means_sha256=discovery_means_sha256,
        content_sha256=content,
    )


def validate_certification_payload(payload: FrozenCertificationPayload) -> None:
    if tuple(payload.projectors) != (PRIMARY_TERMINAL, *HAAR_CONDITIONS):
        raise RuntimeError("certification payload operator inventory differs")
    if set(payload.alphas) != set(payload.projectors) or set(payload.operator_sha256) != set(payload.projectors):
        raise RuntimeError("certification payload alpha or hash inventory differs")
    for digest in (
        *payload.operator_sha256.values(), payload.proposal_freeze_sha256,
        payload.sealed_certification_data_sha256, payload.content_sha256,
        payload.discovery_queries_sha256, payload.discovery_means_sha256,
    ):
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise RuntimeError("certification payload contains a malformed SHA256")
    if any(
        type(alpha) not in (int, float) or not math.isfinite(float(alpha)) or not 0 < float(alpha) <= 1
        for alpha in payload.alphas.values()
    ):
        raise RuntimeError("certification payload contains an invalid alpha")
    live_hashes = {label: projector_sha256(operator) for label, operator in payload.projectors.items()}
    if payload.operator_sha256 != live_hashes:
        raise RuntimeError("certification payload live operator bytes differ")
    expected = canonical_sha256({
        "alphas": payload.alphas,
        "operator_sha256": live_hashes,
        "proposal_freeze_sha256": payload.proposal_freeze_sha256,
        "sealed_certification_data_sha256": payload.sealed_certification_data_sha256,
        "discovery_queries_sha256": payload.discovery_queries_sha256,
        "discovery_means_sha256": payload.discovery_means_sha256,
    })
    if payload.content_sha256 != expected:
        raise RuntimeError("certification payload content hash differs")


@torch.no_grad()
def certify_frozen_controls_once(
    endpoint: LinearActionEndpoint,
    certification_queries: Tensor,
    certification_task_ids: Sequence[int],
    means: Tensor,
    frozen: FrozenCertificationPayload,
    *,
    attempt_token: Mapping[str, object],
    checkpoint_seed: int,
    cohort_payload_sha256_by_seed: Mapping[str, str],
    cohort_manifest_sha256_by_seed: Mapping[str, str],
    certification_payload_content_sha256: str,
    certification_manifest_sha256: str,
    final_freeze_sha256: str,
) -> CertifiedControls:
    """Certify an already-sanitized payload after the exclusive attempt claim.

    The caller must durably create the attempt token before loading the sealed
    query tensor.  This API has no design/sample seeds, counts, eligibility,
    directions, proposal budgets, or beta grid and cannot redraw or retune.
    """

    validate_certification_payload(frozen)
    if checkpoint_seed not in CHECKPOINTS:
        raise ValueError("certification checkpoint seed differs")
    if (
        cohort_payload_sha256_by_seed.get(str(checkpoint_seed))
        != certification_payload_content_sha256
        or cohort_manifest_sha256_by_seed.get(str(checkpoint_seed))
        != certification_manifest_sha256
    ):
        raise RuntimeError("certification cohort does not bind the current checkpoint payload")
    validate_certification_attempt_token(
        dict(attempt_token),
        expected_proposal_freeze_sha256=frozen.proposal_freeze_sha256,
        expected_sealed_payload_sha256_by_seed=cohort_payload_sha256_by_seed,
        expected_sealed_certification_data_sha256=frozen.sealed_certification_data_sha256,
        expected_certification_manifest_sha256_by_seed=cohort_manifest_sha256_by_seed,
        expected_final_freeze_sha256=final_freeze_sha256,
    )
    if certification_queries.ndim != 3 or certification_queries.shape[1:] != means.shape:
        raise ValueError("sealed certification query shape differs")
    if len(certification_task_ids) != certification_queries.shape[0] or set(certification_task_ids) != set(range(10)):
        raise ValueError("sealed certification must contain the exact ten-task inventory")
    profiles = {
        label: action_profile(
            endpoint, certification_queries, means, projector,
            alpha=frozen.alphas[label], task_ids=certification_task_ids,
        )
        for label, projector in frozen.projectors.items()
    }
    reference = profiles[PRIMARY_TERMINAL]
    gates = {
        label: compare_action_profiles(
            profiles[label], reference, gates=CERTIFICATION_GATES,
            location=f"sealed_certification/condition={label}",
        )
        for label in HAAR_CONDITIONS
    }
    failures = [gate for values in gates.values() for gate in values if not gate.passed]
    if failures:
        maximum_offender = min(failures, key=lambda gate: gate.signed_margin)
        raise CertificationInfeasible(
            "certification_infeasible",
            "frozen quintet failed its one allowed sealed-certification attempt",
            {
                "certification_attempt": 1,
                "resumed_sampling": False,
                "sealed_payload_sha256": frozen.content_sha256,
                "proposal_freeze_sha256": frozen.proposal_freeze_sha256,
                "operator_sha256": frozen.operator_sha256,
                "gate_results": {
                    label: tuple(gate.as_dict() for gate in values)
                    for label, values in gates.items()
                },
                "certification_profiles": profiles,
                "maximum_offender": maximum_offender.as_dict(),
            },
        )
    return CertifiedControls(
        frozen=frozen,
        certification_profiles=profiles,
        certification_gate_results=gates,
    )
