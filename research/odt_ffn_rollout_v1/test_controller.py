from __future__ import annotations

import numpy as np
import pytest

from research.odt_ffn_rollout_v1.controller import _direct_qr_square_solve, _operational_coordinates, opspace_matrices_direct_qr
from research.odt_ffn_rollout_v1.guards import source_audit,install

source_audit()
install()


def test_direct_square_solve_reconstructs_known_solution():
    matrix = np.array([[3., 1., 0.], [0.5, 2., -0.2], [1., 0., 4.]], dtype=np.float64)
    expected = np.array([[0.2, -0.1], [1.2, 0.7], [-0.5, 0.3]], dtype=np.float64)
    actual = _direct_qr_square_solve(matrix, matrix @ expected)
    assert np.max(np.abs(actual - expected)) < 1e-13


def test_operational_coordinates_have_analytic_diagonal_answer():
    mass = np.diag(np.arange(1., 8.))
    jacobian = np.eye(6, 7, dtype=np.float64)
    full, position, orientation, nullspace = opspace_matrices_direct_qr(mass, jacobian, jacobian[:3], jacobian[3:])
    assert np.max(np.abs(full - np.diag(np.arange(1., 7.)))) < 1e-13
    assert np.max(np.abs(position - np.diag(np.arange(1., 4.)))) < 1e-13
    assert np.max(np.abs(orientation - np.diag(np.arange(4., 7.)))) < 1e-13
    expected = np.zeros((7, 7), dtype=np.float64)
    expected[-1, -1] = 1.
    assert np.max(np.abs(nullspace - expected)) < 1e-13


def test_zero_singular_system_fails_closed():
    with pytest.raises(RuntimeError, match="singular"):
        _direct_qr_square_solve(np.zeros((2, 2), dtype=np.float64), np.eye(2, dtype=np.float64))


def test_coupled_mass_and_jacobian_satisfy_original_constraint_equations():
    mass = np.diag(np.arange(2., 9.))
    for index in range(6):
        mass[index, index + 1] = mass[index + 1, index] = 0.125
    jacobian = np.eye(6, 7, dtype=np.float64)
    jacobian[:, -1] = np.arange(1., 7.) / 11.
    jacobian[0, 2] = 0.25
    jacobian[3, 1] = -0.375
    multiplier, acceleration = _operational_coordinates(mass, jacobian)
    nullspace = np.eye(7) - acceleration @ jacobian
    assert np.max(np.abs(mass @ acceleration - jacobian.T @ multiplier)) < 1e-12
    assert np.max(np.abs(jacobian @ acceleration - np.eye(6))) < 1e-12
    assert np.max(np.abs(jacobian @ nullspace)) < 1e-12


def test_direct_factors_handle_multiscale_known_solution_without_alternate_factorization():
    matrix = np.array([[1., 1., 1.], [1., 1.000001, 1.], [1., 1., 1.000002]], dtype=np.float64)
    expected = np.array([[2., -1.], [-3., 0.5], [1., 2.]], dtype=np.float64)
    actual = _direct_qr_square_solve(matrix, matrix @ expected)
    assert np.max(np.abs(matrix @ actual - matrix @ expected)) < 1e-12


def test_captured_real_system_matches_independent_decimal80_direct_qr():
    import json
    from pathlib import Path
    capture = json.loads((Path(__file__).parent / "fixtures/modal_odt_controller_real_gate_capture.json").read_text())
    matrix, rhs = np.asarray(capture["matrix"]), np.asarray(capture["rhs"])
    actual = _direct_qr_square_solve(matrix, rhs)
    reference = np.asarray(capture["reference_decimal80_solution"], dtype=np.longdouble)
    error = np.max(np.abs(actual.astype(np.longdouble) - reference)) / np.max(np.abs(reference))
    assert error < 2e-16
    # The formerly failing threshold rejects even the correctly rounded answer.
    rounded = reference.astype(np.float64).astype(np.longdouble)
    rhs_error = np.max(np.abs(matrix.astype(np.longdouble) @ rounded - rhs))
    assert rhs_error > 1e-9


def test_normwise_gate_rejects_a_deliberately_wrong_solution(monkeypatch):
    from research.odt_ffn_rollout_v1 import controller
    monkeypatch.setattr(controller, "_back_substitute", lambda _r, projected: np.zeros_like(projected))
    with pytest.raises(RuntimeError, match="backward error too large"):
        _direct_qr_square_solve(np.eye(2, dtype=np.float64), np.eye(2, dtype=np.float64))


def test_structural_zero_roundoff_matches_decimal80_despite_componentwise_warning():
    import json
    from pathlib import Path
    capture = json.loads((Path(__file__).parent / "fixtures/modal_odt_controller_zero_component_capture.json").read_text())
    matrix, rhs = np.asarray(capture["matrix"]), np.asarray(capture["rhs"])
    actual = _direct_qr_square_solve(matrix, rhs)
    reference = np.asarray(capture["reference_decimal80_solution"], dtype=np.longdouble)
    error = np.max(np.abs(actual.astype(np.longdouble) - reference)) / np.max(np.abs(reference))
    assert error < 2e-16
    assert capture["componentwise_backward_error"] > .5
    assert capture["worst_component_residual"] < 1e-35
    assert capture["normwise_backward_error"] < 1e-16


def test_guard_composition_preserves_live_legacy_and_scipy_owners():
    from research.odt_ffn_rollout_v1.guards import assert_intact
    receipt=assert_intact()
    assert receipt["native"]["prohibited_attempt_count"]==0
    assert receipt["controller"]["prohibited_calls"]==0
    assert "scipy.linalg.null_space" in receipt["controller"]["blocked_entrypoints"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_equation_rejected(bad):
    matrix = np.eye(2, dtype=np.float64)
    matrix[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        _direct_qr_square_solve(matrix, np.eye(2, dtype=np.float64))
