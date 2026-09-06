from __future__ import annotations

import numpy as np
import pytest

from modal_odt_dimension_curve_controller import _direct_qr_square_solve, _operational_coordinates, opspace_matrices_direct_qr


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


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_equation_rejected(bad):
    matrix = np.eye(2, dtype=np.float64)
    matrix[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        _direct_qr_square_solve(matrix, np.eye(2, dtype=np.float64))
