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


def test_direct_factors_handle_multiscale_known_solution_without_alternate_factorization():
    matrix = np.array([[1., 1., 1.], [1., 1.000001, 1.], [1., 1., 1.000002]], dtype=np.float64)
    expected = np.array([[2., -1.], [-3., 0.5], [1., 2.]], dtype=np.float64)
    actual = _direct_qr_square_solve(matrix, matrix @ expected)
    assert np.max(np.abs(matrix @ actual - matrix @ expected)) < 1e-12


def test_guard_blocks_all_known_rank_condition_and_basis_shortcuts(monkeypatch):
    import scipy.linalg
    import torch
    from modal_odt_dimension_curve_controller import install_prohibited_route_guards
    for namespace in (np.linalg, np.linalg.linalg, np, scipy.linalg, torch, torch.linalg, torch.Tensor):
        for name in ("svd", "svdvals", "svd_lowrank", "pca_lowrank", "pinv", "pinvh", "pinverse", "lstsq", "polar", "cov", "matrix_rank", "cond", "orth", "null_space", "norm", "matrix_norm"):
            if hasattr(namespace, name):
                monkeypatch.setattr(namespace, name, getattr(namespace, name))
    receipt = install_prohibited_route_guards()
    expected = {"numpy.linalg.matrix_rank", "numpy.linalg.cond", "torch.linalg.matrix_rank", "torch.linalg.cond", "scipy.linalg.orth", "scipy.linalg.null_space"}
    assert expected <= set(receipt["blocked_entrypoints"])
    # Inspect wrappers without invoking a prohibited entrypoint in this test.
    assert np.linalg.matrix_rank.__name__ == "reject"
    assert scipy.linalg.null_space.__name__ == "reject"


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_equation_rejected(bad):
    matrix = np.eye(2, dtype=np.float64)
    matrix[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        _direct_qr_square_solve(matrix, np.eye(2, dtype=np.float64))
