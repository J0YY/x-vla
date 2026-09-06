"""Pure tests for the fresh-lockbox Stage 7 v3 tensor path."""

from __future__ import annotations

import inspect

import pytest
import torch

import xvla.train.post_norm_intervention_v3 as v3
from athena.build_post_norm_intervention_v3_panel import frame_sha256
from athena.post_norm_intervention_v3_common import (
    CERTIFICATION_ATTEMPT_KEYS,
    CERTIFICATION_GATES,
    DESIGN_NAMESPACE,
    HAAR_CONDITIONS,
    PANEL_NAMESPACE,
    SAMPLE_NAMESPACE,
    SEALED_CERTIFICATION_NAMESPACE,
    SCHEMA_CERTIFICATION_ATTEMPT,
    canonical_sha256,
    claim_certification_attempt,
    namespaced_seed,
)
from xvla.train.post_norm_intervention import HorizonLowRankProjector
from xvla.train.post_norm_intervention_v3 import (
    ActionProfile,
    CertificationInfeasible,
    DevelopmentFold,
    GateResult,
    certify_frozen_controls_once,
    compare_action_profiles,
    fresh_episode_partition,
    freeze_development_proposals,
    make_certification_payload,
    numerical_operator_gate_results,
)
from xvla.train.vla_terminal_balanced_odt import LinearActionEndpoint


DTYPE = torch.float64


def endpoint(hidden: int = 3, horizon: int = 2) -> LinearActionEndpoint:
    weight = torch.zeros(7, hidden, dtype=DTYPE)
    weight[0, 0] = 1
    weight[1, 1] = 1
    weight[6, 2] = 1
    return LinearActionEndpoint(
        weight_normalized=weight,
        bias_normalized=torch.zeros(7, dtype=DTYPE),
        weight_control=weight.clone(),
        bias_control=torch.zeros(7, dtype=DTYPE),
        action_mean=torch.zeros(7, dtype=DTYPE),
        action_std=torch.ones(7, dtype=DTYPE),
        horizon=horizon,
    )


def primary(horizon: int = 2, hidden: int = 3) -> HorizonLowRankProjector:
    state = torch.zeros(horizon, hidden, 1, dtype=DTYPE)
    state[:, 0, 0] = 1
    return HorizonLowRankProjector(state, state.transpose(1, 2), "balanced_top")


def folds() -> tuple[DevelopmentFold, ...]:
    generator = torch.Generator().manual_seed(4)
    return tuple(
        DevelopmentFold(
            name=f"development_{index}",
            queries=torch.randn(20, 2, 3, generator=generator, dtype=DTYPE),
            task_ids=tuple(task for task in range(10) for _ in range(2)),
        )
        for index in range(4)
    )


def profile(*, coordinate_zero: float = 1.0, include_tasks: bool = True) -> ActionProfile:
    coordinates = (coordinate_zero, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    base = ActionProfile(
        aggregate_rms=1.0,
        coordinate_rms=coordinates,
        group_rms={"translation": 1.0, "rotation": 1.0, "gripper": 1.0},
        horizon_coordinate_rms=(coordinates,) * 2,
        coordinate_second_moment=(coordinates,) * 7,
        gripper_flip_fraction=0.1,
        gripper_margin_reachable_fraction=0.1,
        task_profiles={},
    )
    if not include_tasks:
        return base
    return ActionProfile(
        aggregate_rms=base.aggregate_rms,
        coordinate_rms=base.coordinate_rms,
        group_rms=base.group_rms,
        horizon_coordinate_rms=base.horizon_coordinate_rms,
        coordinate_second_moment=base.coordinate_second_moment,
        gripper_flip_fraction=base.gripper_flip_fraction,
        gripper_margin_reachable_fraction=base.gripper_margin_reachable_fraction,
        task_profiles={task: profile(coordinate_zero=coordinate_zero, include_tasks=False) for task in range(10)},
    )


def test_profile_matching_is_elementwise_and_cannot_mask_one_bad_coordinate() -> None:
    gates = compare_action_profiles(
        profile(coordinate_zero=1.31), profile(), gates=CERTIFICATION_GATES,
        location="sealed_certification/condition=matched_haar_d0",
    )
    failures = [gate for gate in gates if not gate.passed]
    assert any(
        gate.name == "coordinate_rms_relative" and gate.location.endswith("coordinate=0")
        for gate in failures
    )
    assert any(
        gate.name == "horizon_coordinate_profile_relative"
        and gate.location.endswith("horizon=0/coordinate=0")
        for gate in failures
    )
    assert all("coordinate=" in gate.location or "second_moment=" in gate.location for gate in failures)


def test_seed_namespaces_are_fresh_stable_and_pairwise_disjoint() -> None:
    namespaces = (PANEL_NAMESPACE, DESIGN_NAMESPACE, SAMPLE_NAMESPACE, SEALED_CERTIFICATION_NAMESPACE)
    values = [namespaced_seed(namespace, 2, 9) for namespace in namespaces]
    assert len(set(values)) == len(values)
    assert namespaced_seed(DESIGN_NAMESPACE, 2, 9) == namespaced_seed(DESIGN_NAMESPACE, 2, 9)
    assert all("v3" in namespace and "v2" not in namespace for namespace in namespaces)
    with pytest.raises(ValueError, match="v3 namespace"):
        namespaced_seed("xvla-stage7-v2-sample", 2, 9)


def test_raw_frame_identity_includes_task_episode_and_frame() -> None:
    identities = {
        frame_sha256(task, episode, frame)
        for task in (0, 1) for episode in (0, 1) for frame in (0, 1)
    }
    assert len(identities) == 8


def test_fresh_panel_excludes_old_windows_and_frame_footprints_then_splits_by_episode() -> None:
    tasks: list[int] = []
    episodes: list[str] = []
    samples: list[str] = []
    footprints: list[tuple[str, ...]] = []
    for task in range(10):
        for episode in range(8):
            tasks.append(task)
            episodes.append(f"episode-{episode}")
            samples.append(f"task-{task}-window-{episode}")
            footprints.append((f"task-{task}-frame-{episode}",))
    partition = fresh_episode_partition(
        tasks,
        episodes,
        samples,
        footprints,
        excluded_sample_ids=tuple(f"task-{task}-window-0" for task in range(10)),
        excluded_frame_footprints=tuple(f"task-{task}-frame-1" for task in range(10)),
        excluded_episode_ids=tuple((task, "episode-2") for task in range(10)),
    )
    assert partition.excluded_sample_count == 10
    assert partition.excluded_footprint_count == 10
    assert partition.excluded_episode_count == 10
    development_keys = [set(values) for values in partition.development_episode_keys]
    certification_keys = set(partition.certification_episode_keys)
    assert all(values and not values & certification_keys for values in development_keys)
    assert all(
        not development_keys[left] & development_keys[right]
        for left in range(4) for right in range(left + 1, 4)
    )
    used = set(partition.certification)
    for values in partition.development_folds:
        assert not used & set(values)
        used.update(values)
    excluded_samples = {f"task-{task}-window-0" for task in range(10)}
    excluded_frames = {f"task-{task}-frame-1" for task in range(10)}
    assert all(samples[index] not in excluded_samples for index in used)
    assert all(not set(footprints[index]) & excluded_frames for index in used)


def test_fresh_panel_fails_closed_when_episode_disjoint_supply_is_too_small() -> None:
    with pytest.raises(v3.FreshPanelInfeasible) as error:
        fresh_episode_partition(
            [task for task in range(10) for _ in range(4)],
            [f"episode-{episode}" for _ in range(10) for episode in range(4)],
            [f"task-{task}-window-{episode}" for task in range(10) for episode in range(4)],
            [(f"task-{task}-frame-{episode}",) for task in range(10) for episode in range(4)],
            excluded_sample_ids=(), excluded_frame_footprints=(),
            excluded_episode_ids=(),
        )
    assert error.value.failure_class == "fresh_panel_infeasible"
    assert error.value.details["per_task_episode_counts"] == {task: 4 for task in range(10)}


def test_fresh_panel_rejects_internal_raw_frame_overlap() -> None:
    tasks = [task for task in range(10) for _ in range(5)]
    episodes = [f"episode-{episode}" for _ in range(10) for episode in range(5)]
    samples = [f"task-{task}-window-{episode}" for task in range(10) for episode in range(5)]
    footprints = [(f"task-{task}-frame-{episode}",) for task in range(10) for episode in range(5)]
    footprints[1] = footprints[0]
    with pytest.raises(v3.FreshPanelInfeasible, match="overlap"):
        fresh_episode_partition(
            tasks, episodes, samples, footprints,
            excluded_sample_ids=(), excluded_frame_footprints=(), excluded_episode_ids=(),
        )


def test_proposal_api_cannot_receive_certification_or_outcomes() -> None:
    parameters = set(inspect.signature(freeze_development_proposals).parameters)
    assert "certification_queries" not in parameters
    assert "certification_task_ids" not in parameters
    assert "outcomes" not in parameters
    certification_parameters = set(inspect.signature(certify_frozen_controls_once).parameters)
    assert not certification_parameters & {
        "state_directions", "analysis_directions", "alpha_grid",
        "feasibility_pool_size", "sampling_pool_size",
    }
    assert "checkpoint_seed" in certification_parameters
    panel_parameters = set(inspect.signature(fresh_episode_partition).parameters)
    assert "split_seed" not in panel_parameters
    assert "development_folds" not in panel_parameters


def test_attempt_token_is_exclusive_and_cannot_be_reclaimed(tmp_path) -> None:
    path = tmp_path / "attempt.json"
    kwargs = {
        "proposal_freeze_sha256": "f" * 64,
        "sealed_payload_sha256_by_seed": {str(seed): "e" * 64 for seed in range(3)},
        "sealed_certification_data_sha256": "d" * 64,
        "certification_manifest_sha256_by_seed": {str(seed): "c" * 64 for seed in range(3)},
        "final_freeze_sha256": "b" * 64,
        "frozen_source_hashes": {"xvla/train/post_norm_intervention_v3.py": "a" * 64},
    }
    token = claim_certification_attempt(path, **kwargs)
    assert token["attempt"] == 1
    with pytest.raises(FileExistsError):
        claim_certification_attempt(path, **kwargs)


def test_design_counts_complete_quintets_and_sample_freezes_first_quintet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v3, "projector_cosine", lambda left, right: 0.0)
    monkeypatch.setattr(v3, "_pooled_rms_from_unit_deltas", lambda *args, **kwargs: 1.0)

    def always_pass(endpoint_value, fold_values, unit_deltas, primary_profiles, *, alpha):
        profiles = {fold.name: primary_profiles[fold.name] for fold in fold_values}
        return True, profiles, ()

    monkeypatch.setattr(v3, "_cached_candidate_development_evaluation", always_pass)
    monkeypatch.setattr(v3, "_cached_candidate_development_passes", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        v3, "numerical_operator_gate_results",
        lambda *args, **kwargs: {label: () for label in ("balanced_top", *HAAR_CONDITIONS)},
    )
    system = [torch.eye(3, dtype=DTYPE) for _ in range(2)]
    frozen = v3._freeze_development_proposals(
        endpoint(), endpoint().to(dtype=torch.float32), folds(),
        torch.zeros(2, 3, dtype=DTYPE), primary(), system, system,
        checkpoint_seed=2, alpha_grid=(0.5,), feasibility_pool_size=10,
        minimum_feasible_quintets=2, sampling_pool_size=5,
    )
    assert frozen.chosen_beta == 0.5
    assert frozen.design_counts_by_alpha == {0.5: 10}
    assert frozen.design_quintets_by_alpha == {0.5: 2}
    assert frozen.proposal_indices == {label: index for index, label in enumerate(HAAR_CONDITIONS)}
    assert frozen.sampling_eligibility == (True,) * 5
    assert tuple(frozen.controls) == HAAR_CONDITIONS
    assert tuple(frozen.development_gate_results) == HAAR_CONDITIONS
    assert "balanced_top" not in frozen.development_gate_results


def test_design_rejects_alpha_above_one_even_inside_roundoff_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v3, "projector_cosine", lambda left, right: 0.0)
    calls = iter((1.0, *(1.0 - 5e-13 for _ in range(5))))
    monkeypatch.setattr(v3, "_pooled_rms_from_unit_deltas", lambda *args, **kwargs: next(calls))
    monkeypatch.setattr(
        v3, "_cached_candidate_development_passes",
        lambda *args, **kwargs: pytest.fail("an alpha above one reached the gate predicate"),
    )
    system = [torch.eye(3, dtype=DTYPE) for _ in range(2)]
    with pytest.raises(v3.DesignInfeasible) as error:
        v3._freeze_development_proposals(
            endpoint(), endpoint().to(dtype=torch.float32), folds(),
            torch.zeros(2, 3, dtype=DTYPE), primary(), system, system,
            checkpoint_seed=2, alpha_grid=(1.0,), feasibility_pool_size=5,
            minimum_feasible_quintets=1, sampling_pool_size=5,
        )
    assert error.value.failure_class == "ordered_greedy_design_shortfall"


def test_certification_failure_is_typed_quantitative_and_cannot_resume_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v3, "projector_cosine", lambda left, right: 0.0)
    monkeypatch.setattr(v3, "_pooled_rms_from_unit_deltas", lambda *args, **kwargs: 1.0)

    def always_pass(endpoint_value, fold_values, unit_deltas, primary_profiles, *, alpha):
        return True, {fold.name: primary_profiles[fold.name] for fold in fold_values}, ()

    monkeypatch.setattr(v3, "_cached_candidate_development_evaluation", always_pass)
    monkeypatch.setattr(v3, "_cached_candidate_development_passes", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        v3, "numerical_operator_gate_results",
        lambda *args, **kwargs: {label: () for label in ("balanced_top", *HAAR_CONDITIONS)},
    )
    system = [torch.eye(3, dtype=DTYPE) for _ in range(2)]
    frozen = v3._freeze_development_proposals(
        endpoint(), endpoint().to(dtype=torch.float32), folds(),
        torch.zeros(2, 3, dtype=DTYPE), primary(), system, system,
        checkpoint_seed=2, alpha_grid=(0.5,), feasibility_pool_size=5,
        minimum_feasible_quintets=1, sampling_pool_size=5,
    )
    failed_gate = GateResult(
        name="aggregate_relative", actual=0.11, reference=0.0, threshold=0.10,
        signed_margin=-0.01, passed=False, location="sealed_certification",
    )
    monkeypatch.setattr(v3, "compare_action_profiles", lambda *args, **kwargs: (failed_gate,))
    payload = make_certification_payload(
        frozen, proposal_freeze_sha256="f" * 64,
        sealed_certification_data_sha256="d" * 64,
        discovery_queries_sha256="9" * 64,
        discovery_means_sha256="8" * 64,
    )
    assert payload.content_sha256 != "e" * 64
    token = {key: None for key in CERTIFICATION_ATTEMPT_KEYS}
    token.update({
        "schema": SCHEMA_CERTIFICATION_ATTEMPT,
        "attempt": 1,
        "seeds": [0, 1, 2],
        "proposal_freeze_sha256": "f" * 64,
        "sealed_payload_sha256_by_seed": {
            "0": "e" * 64, "1": "e" * 64, "2": payload.content_sha256,
        },
        "sealed_certification_data_sha256": "d" * 64,
        "certification_manifest_sha256_by_seed": {
            "0": "a" * 64, "1": "a" * 64, "2": "c" * 64,
        },
        "final_freeze_sha256": "b" * 64,
        "source_hashes": {"xvla/train/post_norm_intervention_v3.py": "a" * 64},
    })
    token["content_sha256"] = canonical_sha256(
        {key: token[key] for key in token if key != "content_sha256"}
    )
    queries = torch.randn(20, 2, 3, generator=torch.Generator().manual_seed(9), dtype=DTYPE)
    with pytest.raises(CertificationInfeasible) as error:
        certify_frozen_controls_once(
            endpoint(), queries, tuple(task for task in range(10) for _ in range(2)),
            torch.zeros(2, 3, dtype=DTYPE), payload, attempt_token=token,
            checkpoint_seed=2,
            cohort_payload_sha256_by_seed=token["sealed_payload_sha256_by_seed"],
            cohort_manifest_sha256_by_seed=token["certification_manifest_sha256_by_seed"],
            certification_payload_content_sha256=payload.content_sha256,
            certification_manifest_sha256="c" * 64, final_freeze_sha256="b" * 64,
        )
    assert error.value.failure_class == "certification_infeasible"
    assert error.value.details["certification_attempt"] == 1
    assert error.value.details["resumed_sampling"] is False
    assert len(error.value.details["gate_results"]) == len(HAAR_CONDITIONS)
    assert set(error.value.details["certification_profiles"]) == {"balanced_top", *HAAR_CONDITIONS}
    assert error.value.details["maximum_offender"]["signed_margin"] == pytest.approx(-0.01)
    assert "selected_proposal_seeds" not in error.value.details
    assert "design_counts_by_alpha" not in error.value.details


def test_numerical_failure_preserves_exact_sample_stream_for_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v3, "projector_cosine", lambda left, right: 0.0)
    monkeypatch.setattr(v3, "_pooled_rms_from_unit_deltas", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(v3, "_cached_candidate_development_passes", lambda *args, **kwargs: True)

    def always_pass(endpoint_value, fold_values, unit_deltas, primary_profiles, *, alpha):
        return True, {fold.name: primary_profiles[fold.name] for fold in fold_values}, ()

    monkeypatch.setattr(v3, "_cached_candidate_development_evaluation", always_pass)
    failed = GateResult(
        name="float32_dual", actual=1.0, reference=0.0, threshold=5e-5,
        signed_margin=5e-5 - 1.0, passed=False, location="development/balanced_top",
    )
    monkeypatch.setattr(
        v3, "numerical_operator_gate_results", lambda *args, **kwargs: {"balanced_top": (failed,)},
    )
    system = [torch.eye(3, dtype=DTYPE) for _ in range(2)]
    with pytest.raises(v3.DesignInfeasible) as error:
        v3._freeze_development_proposals(
            endpoint(), endpoint().to(dtype=torch.float32), folds(),
            torch.zeros(2, 3, dtype=DTYPE), primary(), system, system,
            checkpoint_seed=2, alpha_grid=(0.5,), feasibility_pool_size=5,
            minimum_feasible_quintets=1, sampling_pool_size=5,
        )
    assert error.value.failure_class == "numerical_gate_infeasible"
    assert error.value.details["proposals_examined"] == 5
    assert error.value.details["sampling_eligibility"] == (True,) * 5


def test_unmodified_rank_one_operator_passes_float64_float32_replay_and_idempotence() -> None:
    model_endpoint = endpoint()
    operator = primary()
    results = numerical_operator_gate_results(
        model_endpoint,
        model_endpoint.to(dtype=torch.float32),
        folds(),
        torch.zeros(2, 3, dtype=DTYPE),
        {"balanced_top": operator},
        {"balanced_top": 0.5},
    )
    assert set(results) == {"balanced_top"}
    assert all(gate.passed for gate in results["balanced_top"])
    assert {gate.name for gate in results["balanced_top"]} == {
        "float64_dual", "float64_idempotence", "float64_factor_dense_action_replay",
        "float64_pooled_development_aggregate", "float32_dual", "float32_idempotence",
        "float32_factor_dense_action_replay", "float32_aggregate_relative_error",
    }


def test_float32_gate_measurement_does_not_construct_a_threshold_enforcing_projector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = HorizonLowRankProjector.to

    def guarded(self, *, dtype, device=None):
        if dtype == torch.float32:
            raise AssertionError("float32 constructor would preempt typed gate classification")
        return original(self, dtype=dtype, device=device)

    monkeypatch.setattr(HorizonLowRankProjector, "to", guarded)
    results = numerical_operator_gate_results(
        endpoint(), endpoint().to(dtype=torch.float32), folds(),
        torch.zeros(2, 3, dtype=DTYPE), {"balanced_top": primary()},
        {"balanced_top": 0.5},
    )
    assert len(results["balanced_top"]) == 8


def test_nonfinite_float32_factor_cast_becomes_finite_typed_gate_failures() -> None:
    state = torch.zeros(2, 3, 1, dtype=DTYPE)
    analysis = torch.zeros(2, 1, 3, dtype=DTYPE)
    state[:, 0, 0] = 1e40
    analysis[:, 0, 0] = 1e-40
    operator = HorizonLowRankProjector(state, analysis, "balanced_top")
    results = numerical_operator_gate_results(
        endpoint(), endpoint().to(dtype=torch.float32), folds(),
        torch.zeros(2, 3, dtype=DTYPE), {"balanced_top": operator},
        {"balanced_top": 0.5},
    )["balanced_top"]
    float32 = [gate for gate in results if gate.name.startswith("float32_")]
    assert len(float32) == 4
    assert all(not gate.passed and torch.isfinite(torch.tensor(gate.actual)) for gate in float32)


def test_engineering_runtime_error_during_float32_replay_is_not_scientific_infeasibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def engineering_fault(*args, **kwargs):
        raise RuntimeError("shape, allocator, or kernel fault")

    monkeypatch.setattr(v3, "_raw_factor_delta", engineering_fault)
    with pytest.raises(RuntimeError, match="allocator"):
        numerical_operator_gate_results(
            endpoint(), endpoint().to(dtype=torch.float32), folds(),
            torch.zeros(2, 3, dtype=DTYPE), {"balanced_top": primary()},
            {"balanced_top": 0.5},
        )
