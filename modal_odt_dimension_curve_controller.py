"""Direct-QR operational-space controller adapter for the Modal ODT lane.

The standard controller uses pseudoinverses of weighted self-overlap matrices.
This adapter instead solves the equivalent constrained dynamics block system
directly.  It forms no self-overlap matrix, performs no spectral decomposition,
and rejects any singular or unreconstructed system.  Results must be compared
with a baseline evaluated using this same adapter.
"""

from __future__ import annotations

import numpy as np


COUNTS = {"calls": 0, "direct_qr_systems": 0, "largest_relative_residual": 0.0}


def install_prohibited_route_guards() -> dict:
    """Reject prohibited entrypoints throughout policy and simulator execution."""
    import scipy.linalg
    import torch
    blocked = []
    receipt = {"blocked_entrypoints": blocked, "prohibited_calls": 0, "matrix_spectral_norm_calls": 0}
    def reject(*_args, **_kwargs):
        receipt["prohibited_calls"] += 1
        raise RuntimeError("prohibited numerical route entered the Modal ODT process")
    names = ("svd", "svdvals", "svd_lowrank", "pca_lowrank", "pinv", "pinvh", "pinverse", "lstsq", "polar", "cov", "matrix_rank", "cond", "orth", "null_space")
    for label, namespace in (("numpy.linalg", np.linalg), ("numpy.linalg.linalg", np.linalg.linalg), ("numpy", np),
                             ("scipy.linalg", scipy.linalg), ("torch", torch),
                             ("torch.linalg", torch.linalg), ("torch.Tensor", torch.Tensor)):
        for name in names:
            if hasattr(namespace, name):
                setattr(namespace, name, reject)
                blocked.append(f"{label}.{name}")
    def protected_norm(original):
        def wrapper(value, *args, **kwargs):
            order = kwargs.get("ord", args[0] if args else None)
            axes = kwargs.get("axis", kwargs.get("dim", args[1] if len(args) > 1 else None))
            if getattr(value, "ndim", 0) >= 2 and order in (2, -2, "nuc") and (axes is None or isinstance(axes, tuple)):
                receipt["matrix_spectral_norm_calls"] += 1
                raise RuntimeError("matrix spectral norm entered the Modal ODT process")
            return original(value, *args, **kwargs)
        return wrapper
    for label, namespace in (("numpy.linalg", np.linalg), ("scipy.linalg", scipy.linalg), ("torch.linalg", torch.linalg)):
        namespace.norm = protected_norm(namespace.norm)
        blocked.append(f"{label}.norm_spectral_matrix_orders")
    torch.linalg.matrix_norm = reject
    blocked.append("torch.linalg.matrix_norm")
    blocked.sort()
    return receipt


def _back_substitute(r: np.ndarray, projected: np.ndarray) -> np.ndarray:
    solution = np.empty_like(projected)
    for row in range(r.shape[0] - 1, -1, -1):
        if r[row, row] == 0.0:
            raise RuntimeError("controller direct QR has a singular triangular equation")
        value = projected[row].copy()
        if row + 1 < r.shape[0]:
            value -= r[row, row + 1:] @ solution[row + 1:]
        solution[row] = value / r[row, row]
    return solution


def _direct_qr_square_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or rhs.shape[0] != matrix.shape[0]:
        raise ValueError("controller QR system has incompatible dimensions")
    if matrix.dtype != np.float64 or rhs.dtype != np.float64 or not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
        raise ValueError("controller QR system requires finite float64 coefficients")
    q, r = np.linalg.qr(matrix, mode="reduced")
    reconstruction = q @ r
    scale = max(float(np.max(np.abs(matrix))), np.finfo(np.float64).tiny)
    if float(np.max(np.abs(reconstruction - matrix))) > 1e-11 * scale:
        raise RuntimeError("controller direct QR reconstruction failed")
    projected = q.T @ rhs
    solution = _back_substitute(r, projected)
    # Two unconditional refinement steps reuse exactly the same direct factors.
    # Extended-precision residual arithmetic avoids cancellation in the tiny
    # constrained dynamics system.  There is no alternate factorization route.
    extended_matrix = matrix.astype(np.longdouble)
    extended_rhs = rhs.astype(np.longdouble)
    for _ in range(2):
        residual = extended_rhs - extended_matrix @ solution.astype(np.longdouble)
        correction = _back_substitute(r, q.T @ residual.astype(np.float64))
        solution += correction
    if not np.isfinite(solution).all():
        raise RuntimeError("controller direct QR solution is nonfinite")
    error = float(np.max(np.abs(extended_matrix @ solution.astype(np.longdouble) - extended_rhs)))
    relative = error / max(float(np.max(np.abs(rhs))), np.finfo(np.float64).tiny)
    if relative > 1e-9:
        raise RuntimeError(f"controller constrained dynamics residual too large: {relative}")
    COUNTS["direct_qr_systems"] += 1
    COUNTS["largest_relative_residual"] = max(COUNTS["largest_relative_residual"], relative)
    return solution


def _operational_coordinates(mass: np.ndarray, jacobian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joints, axes = mass.shape[0], jacobian.shape[0]
    if mass.shape != (joints, joints) or jacobian.shape[1] != joints:
        raise ValueError("constrained dynamics geometry differs")
    system = np.zeros((joints + axes, joints + axes), dtype=np.float64)
    system[:joints, :joints] = mass
    system[:joints, joints:] = -jacobian.T
    system[joints:, :joints] = jacobian
    rhs = np.zeros((joints + axes, axes), dtype=np.float64)
    rhs[joints:] = np.eye(axes, dtype=np.float64)
    solution = _direct_qr_square_solve(system, rhs)
    return solution[joints:], solution[:joints]


def opspace_matrices_direct_qr(mass_matrix, J_full, J_pos, J_ori):
    mass = np.asarray(mass_matrix, dtype=np.float64)
    lambda_full, dynamically_consistent_coordinates = _operational_coordinates(mass, np.asarray(J_full, dtype=np.float64))
    lambda_pos, _ = _operational_coordinates(mass, np.asarray(J_pos, dtype=np.float64))
    lambda_ori, _ = _operational_coordinates(mass, np.asarray(J_ori, dtype=np.float64))
    nullspace = np.eye(mass.shape[0], dtype=np.float64) - dynamically_consistent_coordinates @ J_full
    COUNTS["calls"] += 1
    return lambda_full, lambda_pos, lambda_ori, nullspace


def install_controller() -> dict:
    import robosuite.controllers.osc as controller
    import robosuite.utils.control_utils as utilities
    utilities.opspace_matrices = opspace_matrices_direct_qr
    controller.opspace_matrices = opspace_matrices_direct_qr
    if controller.opspace_matrices is not opspace_matrices_direct_qr:
        raise RuntimeError("controller binding installation failed")
    return {
        "schema": "xvla_modal_direct_qr_constrained_dynamics_controller_v1",
        "method": "direct_householder_qr_of_mass_jacobian_saddle_system",
        "unconditional_same_factor_refinement_steps": 2,
        "singular_policy": "fail_closed_no_fallback",
        "historical_controller_equivalence": "same_constrained_equations_on_nonsingular_systems_only",
    }
