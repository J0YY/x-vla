from __future__ import annotations

from dataclasses import replace

import torch
import pytest

import xvla.train.implicit_sparse_projective_odt as implicit_odt

from xvla.train.implicit_sparse_projective_odt import (
    RECTANGULAR_RQ_METHOD,
    ImplicitNode,
    ImplicitProjectiveDAG,
    MaterializationTelemetry,
    PhysicalSourceSpec,
    ProjectiveConstant,
    ReducedQRBinaryCore,
    ScaledMatrix,
    UnaryCore,
    _Builder,
    _add_pair_vectors,
    _affine_pair,
    _clone_network,
    _dot_pair_vectors,
    _projective_batch_relative_error,
    _scale_pair_vector_by_scalar,
    _walk_unique,
    canonicalize_implicit_dag_direct_rq,
    canonicalize_with_explicit_clone_step_trace,
    diagonalize_implicit_dag_full_rank,
    evaluate_boundary_quotient,
    evaluate_projective_boundary,
    implicit_shape_statistics,
    reverse_implicit_environments,
    telemetry_dict,
    validate_canonical_exponent_normal_form,
)


DTYPE = torch.float64


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = expected.norm().clamp_min(torch.finfo(expected.dtype).tiny)
    return float(((actual - expected).norm() / scale).item())


def _heterogeneous_rectangular_network():
    like = torch.empty((), dtype=DTYPE)
    telemetry = MaterializationTelemetry()
    builder = _Builder(like, telemetry)
    x_spec = PhysicalSourceSpec("x", "source.x", 2)
    y_spec = PhysicalSourceSpec("y", "source.y", 1)
    x = builder.physical_pair(x_spec)
    y = builder.physical_pair(y_spec)

    # Six Khatri-Rao atoms enumerate the exact 3-by-2 homogeneous input basis.
    # The output has seven rows, so this is the production rows>columns case.
    left = torch.zeros(6, 3, dtype=DTYPE)
    right = torch.zeros(6, 2, dtype=DTYPE)
    for atom in range(6):
        left[atom, atom // 2] = 1.0
        right[atom, atom % 2] = 1.0
    output = torch.zeros(7, 6, dtype=DTYPE)
    output[:6] = torch.eye(6, dtype=DTYPE)
    output[6, 5] = 1.0
    expanded = builder.cp(
        "rectangular.expand",
        output,
        left,
        right,
        (x, y),
        kind="heterogeneous_rectangular_adversary",
    )
    assert isinstance(expanded, ImplicitNode)
    head = torch.zeros(2, 7, dtype=DTYPE)
    head[0, 0] = 1.0
    head[0, 3] = 2.0
    head[1, 6] = 1.0
    root = builder.unary("observable", head, expanded, kind="projective_observable")
    assert isinstance(root, ImplicitNode)
    network = ImplicitProjectiveDAG(
        root=root,
        head=torch.eye(2, dtype=DTYPE),
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=1,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=DTYPE),
        physical_sources=builder.physical_source_specs,
    )
    raw = {
        "x": torch.tensor(((-0.3, 0.2), (0.5, -0.1), (0.0, 0.4)), dtype=DTYPE),
        "y": torch.tensor(((0.7,), (-0.2,), (0.9,)), dtype=DTYPE),
    }
    expected = raw["x"][:, :1] * raw["y"] + 2.0 * raw["x"][:, 1:2]
    return network, raw, expected


def _bounded_wide_deficient_network():
    like = torch.empty((), dtype=DTYPE)
    telemetry = MaterializationTelemetry()
    builder = _Builder(like, telemetry)
    x = builder.physical_pair(PhysicalSourceSpec("x", "source.x", 2))
    y = builder.physical_pair(PhysicalSourceSpec("y", "source.y", 1))
    left = torch.zeros(6, 3, dtype=DTYPE)
    right = torch.zeros(6, 2, dtype=DTYPE)
    for atom in range(6):
        left[atom, atom // 2] = 1.0
        right[atom, atom % 2] = 1.0
    output = torch.zeros(5, 6, dtype=DTYPE)
    output[0, 0] = 1.0
    output[1, 1] = 1.0
    output[2, 2] = 1.0
    output[3, 0] = 1.0  # An exact dependent output row.
    output[4, 5] = 1.0
    root = builder.cp(
        "bounded.wide.dependent",
        output,
        left,
        right,
        (x, y),
        kind="bounded_wide_dependent_adversary",
    )
    assert isinstance(root, ImplicitNode)
    head = torch.zeros(2, 5, dtype=DTYPE)
    head[0, 1] = 1.0
    head[1, 4] = 1.0
    network = ImplicitProjectiveDAG(
        root=root,
        head=head,
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=1,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=DTYPE),
        physical_sources=builder.physical_source_specs,
    )
    raw = {
        "x": torch.tensor(((-0.3, 0.2), (0.5, -0.1)), dtype=DTYPE),
        "y": torch.tensor(((0.7,), (-0.2,)), dtype=DTYPE),
    }
    return network, raw, raw["x"][:, :1]


def _production_width_streamed_cp(*, final_pivot: float = 1.0):
    rows = 96
    left_dimension = 65
    right_dimension = 65
    left = torch.zeros(rows, left_dimension, dtype=DTYPE)
    right = torch.zeros(rows, right_dimension, dtype=DTYPE)
    # A coprime stride spreads support across every streamed block, including
    # the short final block, while keeping every Khatri-Rao atom distinct.
    for atom in range(rows):
        flat = 65 * 65 - 1 if atom == rows - 1 else atom * 43
        left[atom, flat // right_dimension] = 1.0
        right[atom, flat % right_dimension] = 1.0
    output = torch.eye(rows, dtype=DTYPE)
    output[-1, -1] = final_pivot
    return implicit_odt.CPBinaryCore(
        output,
        left,
        right,
        "production_width_streamed_q_provenance_adversary",
    )


def test_heterogeneous_direct_rq_shrinks_only_the_structural_rectangular_bond():
    network, raw, expected = _heterogeneous_rectangular_network()
    assert _relative(evaluate_boundary_quotient(network, raw), expected) < 1e-14
    assert _relative(
        evaluate_boundary_quotient(network, (raw["x"], raw["y"])), expected
    ) < 1e-14

    canonical = canonicalize_implicit_dag_direct_rq(
        network, replay_inputs=raw, block_size=3
    )
    rectangular_steps = [
        step for step in canonical.steps if step.method == RECTANGULAR_RQ_METHOD
    ]
    assert len(rectangular_steps) == 1
    step = rectangular_steps[0]
    assert step.unfolding_shape == (7, 6)
    assert step.structurally_reduced
    assert step.literal_q_chart_resolved
    assert step.factorization_relative_error < 1e-14
    assert all(
        item.parent_occurrences_pushed == item.expected_parent_occurrences
        for item in canonical.steps
    )
    reduced = [
        node.core
        for node in _walk_unique(canonical.network.root)
        if isinstance(node.core, ReducedQRBinaryCore)
    ]
    assert len(reduced) == 1
    assert reduced[0].q_rows.shape == (6, 6)
    assert _relative(evaluate_boundary_quotient(canonical.network, raw), expected) < 1e-13
    shape = implicit_shape_statistics(canonical.network)
    assert shape["reduced_q_binary_nodes"] == 1
    assert shape["maximum_reduced_q_elements"] == 36
    meter = telemetry_dict(canonical.telemetry)
    assert meter["rectangular_direct_rq_factorizations"] == 1
    assert meter["maximum_rectangular_q_elements"] == 36
    assert meter["raw_order_three_core_materializations"] == 0
    assert meter["raw_width_cubic_core_materializations"] == 0


def test_heterogeneous_rectangular_shared_trace_matches_independent_clone_then_algorithms2_3():
    network, raw, expected = _heterogeneous_rectangular_network()
    trace = canonicalize_with_explicit_clone_step_trace(
        network, raw, block_size=3
    )
    rectangular = [
        record
        for record, step in zip(trace.records, trace.shared_steps)
        if step.method == RECTANGULAR_RQ_METHOD
    ]
    assert len(rectangular) == 1
    assert rectangular[0].literal_rq_occurrences_compared == 1
    assert rectangular[0].factor_relative_error < 1e-14
    assert rectangular[0].q_core_relative_error < 1e-14
    assert max(record.shared_function_replay_error for record in trace.records) < 1e-13
    assert max(record.clone_function_replay_error for record in trace.records) < 1e-13
    assert _relative(
        evaluate_boundary_quotient(trace.shared_network, raw), expected
    ) < 1e-13

    environments = reverse_implicit_environments(trace.shared_network)
    assert sum(record.child_messages_emitted for record in environments) == 3
    diagonal = diagonalize_implicit_dag_full_rank(
        trace.shared_network, replay_inputs=raw
    )
    assert diagonal.expected_parent_occurrences == diagonal.pushed_parent_occurrences
    assert diagonal.replay_relative_error < 1e-12
    assert diagonal.maximum_recontracted_offdiagonal_ratio < 1e-12
    assert _relative(evaluate_boundary_quotient(diagonal.network, raw), expected) < 1e-12


def test_bounded_wide_dependent_unfolding_retains_direct_q_without_a_solve():
    network, raw, expected = _bounded_wide_deficient_network()
    canonical = canonicalize_implicit_dag_direct_rq(
        network, replay_inputs=raw, block_size=3
    )
    step = next(item for item in canonical.steps if item.uid == network.root.uid)
    assert step.method == RECTANGULAR_RQ_METHOD
    assert step.unfolding_shape == (5, 6)
    assert not step.structurally_reduced
    assert not step.full_row_rank
    assert step.factorization_relative_error < 1e-14
    assert step.per_step_function_replay_error < 1e-14
    assert isinstance(canonical.network.root.core, ReducedQRBinaryCore)
    assert canonical.network.root.core.q_rows.shape == (5, 6)
    assert _relative(evaluate_boundary_quotient(canonical.network, raw), expected) < 1e-14
    environments = reverse_implicit_environments(canonical.network)
    assert sum(item.child_messages_emitted for item in environments) == 2
    diagonal = diagonalize_implicit_dag_full_rank(
        canonical.network, replay_inputs=raw
    )
    assert diagonal.replay_relative_error < 1e-13
    assert diagonal.expected_parent_occurrences == diagonal.pushed_parent_occurrences


def test_constants_are_partially_evaluated_without_fake_physical_leaves():
    like = torch.empty((), dtype=DTYPE)
    telemetry = MaterializationTelemetry()
    builder = _Builder(like, telemetry)
    spec = PhysicalSourceSpec("state", "source.state", 2)
    state = builder.physical_pair(spec)
    constant = builder.constant_pair(
        "constant.offset", torch.tensor((0.25, -0.4), dtype=DTYPE)
    )
    assert isinstance(constant, ProjectiveConstant)
    transformed = _affine_pair(
        builder,
        constant,
        torch.tensor(((1.2, -0.3), (0.4, 0.8)), dtype=DTYPE),
        torch.tensor((0.1, -0.2), dtype=DTYPE),
        "constant.affine",
    )
    assert isinstance(transformed, ProjectiveConstant)
    added = _add_pair_vectors(builder, state, transformed, "state.plus.constant")
    assert isinstance(added, ImplicitNode)
    assert isinstance(added.core, UnaryCore)
    constant_dot = _dot_pair_vectors(
        builder, transformed, transformed, "constant.dot.constant"
    )
    assert isinstance(constant_dot, ProjectiveConstant)
    scaled = _scale_pair_vector_by_scalar(
        builder, constant_dot, added, "constant.scalar.times.state"
    )
    assert isinstance(scaled, ImplicitNode)
    assert isinstance(scaled.core, UnaryCore)
    network = ImplicitProjectiveDAG(
        root=scaled,
        head=torch.eye(3, dtype=DTYPE),
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=2,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=DTYPE),
        physical_sources=builder.physical_source_specs,
    )
    raw = {"state": torch.tensor(((0.2, -0.1), (-0.4, 0.3)), dtype=DTYPE)}
    affine_constant = torch.tensor((0.52, -0.42), dtype=DTYPE)
    scale = float((affine_constant.square().sum()).item())
    expected = scale * (raw["state"] + affine_constant)
    assert _relative(evaluate_boundary_quotient(network, raw), expected) < 2e-14
    leaves = [node for node in _walk_unique(network.root) if not node.children]
    assert len(leaves) == 1
    assert leaves[0].physical_source_key == "state"
    meter = telemetry_dict(telemetry)
    assert meter["constant_unary_folds"] == 1
    assert meter["constant_cp_to_constant_folds"] >= 1
    assert meter["constant_cp_to_unary_folds"] >= 2


def test_heterogeneous_source_registry_fails_closed_without_padding():
    network, raw, _ = _heterogeneous_rectangular_network()
    with pytest.raises(ValueError, match="keys"):
        evaluate_boundary_quotient(network, {"x": raw["x"]})
    with pytest.raises(ValueError, match="wrong width"):
        evaluate_boundary_quotient(
            network,
            {"x": torch.zeros(3, 3, dtype=DTYPE), "y": raw["y"]},
        )
    with pytest.raises(ValueError, match="dtype and device"):
        evaluate_boundary_quotient(
            network,
            {"x": raw["x"].float(), "y": raw["y"].float()},
        )


def test_projective_replay_ignores_rowwise_gauge_but_detects_quotient_change():
    reference = torch.tensor(
        ((0.2, -0.7, 1.0), (1.1, 0.4, -0.8)), dtype=DTYPE
    )
    gauges = torch.tensor((1.0128, 1.1174), dtype=DTYPE)[:, None]
    equivalent = gauges * reference
    assert _projective_batch_relative_error(equivalent, reference) < 1e-15
    changed = equivalent.clone()
    changed[1, 0] += 0.1
    assert _projective_batch_relative_error(changed, reference) > 1e-3

    with pytest.raises(ValueError, match="width at least two"):
        _projective_batch_relative_error(
            torch.ones(2, 1, dtype=DTYPE), torch.ones(2, 1, dtype=DTYPE)
        )


def test_production_algorithm1_uses_local_scale_ledgers_then_one_final_replay():
    network, raw, expected = _heterogeneous_rectangular_network()
    canonical = canonicalize_implicit_dag_direct_rq(
        network,
        replay_inputs=raw,
        block_size=3,
        replay_each_step=False,
        copy_network=False,
    )
    assert canonical.network is network
    assert canonical.full_projective_replay_evaluations == 2
    assert canonical.final_projective_replay_error < 1e-13
    assert all(
        not record.per_step_function_replay_performed
        for record in canonical.steps
    )
    assert all(
        record.local_scaled_reconstruction_relative_error < 1e-13
        and record.local_reconstruction_exponent_delta == 0
        and record.maximum_absorption_scaled_relative_error < 1e-13
        and record.maximum_absorption_exponent_delta == 0
        and record.scale_sensitive_occurrences_verified
        == record.parent_occurrences_pushed
        for record in canonical.steps
    )
    normal_form = validate_canonical_exponent_normal_form(canonical.network)
    assert normal_form["all_core_exponents_zero"]
    assert normal_form["head_exponent_matches_ledger"]
    assert _relative(evaluate_boundary_quotient(network, raw), expected) < 1e-13


def test_algorithms2_3_reject_head_and_internal_exponent_corruption():
    network, raw, _ = _heterogeneous_rectangular_network()
    canonical = canonicalize_implicit_dag_direct_rq(
        network, replay_inputs=raw, block_size=3
    )

    bad_head = _clone_network(canonical.network, unfold=False)
    bad_head.head_binary_exponent += 1
    with pytest.raises(ValueError, match="boundary-head exponent"):
        reverse_implicit_environments(bad_head)
    with pytest.raises(ValueError, match="boundary-head exponent"):
        diagonalize_implicit_dag_full_rank(bad_head)

    bad_internal = _clone_network(canonical.network, unfold=False)
    _walk_unique(bad_internal.root)[0].core.binary_exponent += 1
    with pytest.raises(ValueError, match="core exponent is not zero"):
        reverse_implicit_environments(bad_internal)
    with pytest.raises(ValueError, match="core exponent is not zero"):
        diagonalize_implicit_dag_full_rank(bad_internal)


def test_streamed_algorithm2_immediate_gauges_match_separated_algorithms2_3():
    network, raw, expected = _heterogeneous_rectangular_network()
    canonical = canonicalize_implicit_dag_direct_rq(
        network, replay_inputs=raw, block_size=3
    )
    environments = reverse_implicit_environments(canonical.network)
    separated = diagonalize_implicit_dag_full_rank(
        canonical.network,
        replay_inputs=raw,
        precomputed_environments=environments,
    )
    streamed = diagonalize_implicit_dag_full_rank(
        canonical.network,
        replay_inputs=raw,
        stream_pre_evd_environments=True,
        retain_eigenvalues=False,
        retain_post_evd_environments=False,
    )
    assert streamed.algorithm2_child_messages == sum(
        record.child_messages_emitted for record in environments
    )
    assert streamed.diagonalized_node_count == len(_walk_unique(canonical.network.root))
    assert not streamed.eigenvalue_spectra_retained
    assert streamed.expected_parent_occurrences == streamed.pushed_parent_occurrences
    assert max(
        separated.replay_relative_error,
        streamed.replay_relative_error,
        separated.maximum_recontracted_offdiagonal_ratio,
        streamed.maximum_recontracted_offdiagonal_ratio,
    ) < 1e-12
    separated_value = evaluate_projective_boundary(separated.network, raw)
    streamed_value = evaluate_projective_boundary(streamed.network, raw)
    assert _projective_batch_relative_error(separated_value, streamed_value) < 1e-12
    assert _relative(evaluate_boundary_quotient(streamed.network, raw), expected) < 1e-12


def test_local_scaled_reconstruction_rejects_factor_exponent_corruption(
    monkeypatch: pytest.MonkeyPatch,
):
    network, raw, _ = _heterogeneous_rectangular_network()
    direct = implicit_odt._direct_rq_core
    corrupted_once = False

    def corrupt_factor_exponent(core, telemetry, *, block_size):
        nonlocal corrupted_once
        factor, canonical, diagnostics = direct(
            core, telemetry, block_size=block_size
        )
        if not corrupted_once:
            corrupted_once = True
            factor = ScaledMatrix(
                factor.mantissa, int(factor.binary_exponent) + 1
            )
        return factor, canonical, diagnostics

    monkeypatch.setattr(
        implicit_odt, "_direct_rq_core", corrupt_factor_exponent
    )
    with pytest.raises(RuntimeError, match="local scaled reconstruction"):
        canonicalize_implicit_dag_direct_rq(
            network, replay_inputs=raw, block_size=3
        )


def test_projective_evaluation_does_not_accumulate_fixed_width_scale_exponents():
    network, raw, expected = _heterogeneous_rectangular_network()
    stressed = _clone_network(network, unfold=False)
    enormous = 10**30
    for index, node in enumerate(_walk_unique(stressed.root), start=1):
        node.core.binary_exponent += index * enormous
    stressed.head_binary_exponent -= enormous**2
    coordinates = evaluate_projective_boundary(stressed, raw)
    assert bool(torch.isfinite(coordinates).all())
    assert _relative(evaluate_boundary_quotient(stressed, raw), expected) < 1e-14


def test_production_width_streamed_cp_certifies_nonzero_late_and_short_final_panels():
    core = _production_width_streamed_cp()
    telemetry = MaterializationTelemetry()
    _, canonical, diagnostics = implicit_odt._direct_rq_cp_streamed(
        core, telemetry, block_size=1024
    )
    provenance = diagnostics.direct_q_provenance
    assert provenance is not None
    assert provenance.method == implicit_odt.STREAMED_DIRECT_Q_PROVENANCE_METHOD
    assert provenance.verified
    assert provenance.stage_count == provenance.column_blocks_compared == 5
    assert provenance.columns_compared == 65 * 65
    assert provenance.compact_q_relative_error < 1e-13
    assert provenance.direct_q_reconstruction_relative_error < 1e-13
    assert provenance.conditioning_accepted
    assert provenance.minimum_pivot_to_maximum_entry > provenance.conditioning_threshold
    assert provenance.retained_transition_elements == 4 * 96 * 96
    assert canonical.direct_q_provenance == provenance
    assert telemetry.streamed_direct_q_provenance_certificates == 1
    assert telemetry.streamed_direct_q_replay_qr_factorizations == 5
    assert telemetry.direct_q_columns_compared == 65 * 65
    assert telemetry.maximum_direct_q_transition_elements == 5 * 96 * 96


def test_streamed_cp_rejects_near_null_direct_triangular_chart_without_fallback():
    core = _production_width_streamed_cp(final_pivot=1e-14)
    with pytest.raises(ValueError, match="conditioning gate rejected"):
        implicit_odt._direct_rq_cp_streamed(
            core, MaterializationTelemetry(), block_size=1024
        )


def test_streamed_cp_direct_q_provenance_rejects_a_corrupt_compact_solve(
    monkeypatch: pytest.MonkeyPatch,
):
    core = _production_width_streamed_cp(final_pivot=2e-12)
    solve_compact_q = implicit_odt._solve_compact_q

    def corrupt_compact_q(factor, output_factor):
        compact = solve_compact_q(factor, output_factor).clone()
        compact[-1, -1] += 1e-4
        return compact

    monkeypatch.setattr(implicit_odt, "_solve_compact_q", corrupt_compact_q)
    with pytest.raises(ValueError, match="direct TSQR Q provenance"):
        implicit_odt._direct_rq_cp_streamed(
            core, MaterializationTelemetry(), block_size=1024
        )


def test_streamed_cp_direct_q_provenance_rejects_a_corrupt_replay_panel(
    monkeypatch: pytest.MonkeyPatch,
):
    core = _production_width_streamed_cp()
    direct_panel = implicit_odt._direct_tsqr_panel
    calls = 0

    def corrupt_second_pass_panel(matrix):
        nonlocal calls
        calls += 1
        q_panel, upper = direct_panel(matrix)
        if calls == 10:
            q_panel = q_panel.clone()
            q_panel[-1, -1] += 1e-4
        return q_panel, upper

    monkeypatch.setattr(implicit_odt, "_direct_tsqr_panel", corrupt_second_pass_panel)
    with pytest.raises(ValueError, match="direct TSQR Q provenance"):
        implicit_odt._direct_rq_cp_streamed(
            core, MaterializationTelemetry(), block_size=1024
        )


def test_scaled_environment_aggregation_ignores_zero_message_exponents():
    identity = torch.eye(2, dtype=DTYPE)
    huge_zero = ScaledMatrix(torch.zeros_like(identity), 10_000)
    real = ScaledMatrix(identity, 0)
    combined = implicit_odt._add_scaled_matrices((huge_zero, real))
    assert combined.binary_exponent < 10_000
    assert implicit_odt._scaled_matrix_relative_error(combined, real) == 0.0
    all_zero = implicit_odt._add_scaled_matrices((huge_zero, huge_zero))
    assert all_zero.binary_exponent == 0
    assert torch.equal(all_zero.mantissa, torch.zeros_like(identity))
    scaled_zero = implicit_odt._scale_symmetric(torch.zeros_like(identity), -10_000)
    assert scaled_zero.binary_exponent == 0

    underflowing_nonzero = ScaledMatrix(identity, -5_000)
    with pytest.raises(ValueError, match="message underflowed"):
        implicit_odt._add_scaled_matrices((real, underflowing_nonzero))


def test_streamed_direct_q_metadata_validator_rejects_forged_ledgers():
    valid = implicit_odt.DirectQProvenance(
        method=implicit_odt.STREAMED_DIRECT_Q_PROVENANCE_METHOD,
        verified=True,
        stage_count=2,
        column_blocks_compared=2,
        columns_compared=17,
        compact_q_relative_error=1e-15,
        direct_q_reconstruction_relative_error=1e-15,
        retained_transition_elements=9,
        minimum_pivot_to_maximum_entry=2e-12,
        conditioning_threshold=1e-12,
        conditioning_accepted=True,
    )
    implicit_odt._validate_direct_q_provenance(valid, require_verified=True)
    invalid = (
        replace(valid, column_blocks_compared=1),
        replace(valid, minimum_pivot_to_maximum_entry=1e-12),
        replace(valid, retained_transition_elements=0),
        replace(
            valid,
            stage_count=1,
            column_blocks_compared=1,
            retained_transition_elements=9,
        ),
    )
    for provenance in invalid:
        with pytest.raises(ValueError, match="direct-Q"):
            implicit_odt._validate_direct_q_provenance(
                provenance, require_verified=True
            )
