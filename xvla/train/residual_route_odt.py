"""Exact linear route assemblies for residual tensor-network circuits.

Residual addition is not a sum of independent importance scores.  The route
metric contains identity-update cross blocks, and the upstream route
coefficient metric contains the corresponding cross covariances.  Dropping
either changes the declared coefficient-space objective.

The bilinear FFN helper below returns a separately identified symmetric
polynomial quotient for its two tied input legs. It agrees with the ordered
CP core on diagonal evaluation, but it is not the ordered coefficient tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from xvla.nn.bilinear import BilinearFFN


Tensor = torch.Tensor


@dataclass(frozen=True)
class ResidualRouteAssembly:
    """Dense tiny-oracle map from base plus update features to output."""

    assembly: Tensor
    output_dimension: int
    update_feature_dimension: int
    gain: float
    object_kind: str = "explicit_residual_route_assembly"


def _finite_gain(gain: float) -> float:
    if isinstance(gain, bool) or not isinstance(gain, (int, float)):
        raise TypeError("gain must be a finite real scalar")
    value = float(gain)
    if not math.isfinite(value):
        raise ValueError("gain must be finite")
    return value


def _validate_real_matrix(matrix: Tensor, name: str) -> None:
    if not isinstance(matrix, Tensor) or matrix.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not matrix.is_floating_point() or matrix.is_complex():
        raise TypeError(f"{name} must have a real floating-point dtype")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have nonzero dimensions")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_psd(metric: Tensor, dimension: int, name: str) -> Tensor:
    _validate_real_matrix(metric, name)
    if metric.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    symmetric = 0.5 * (metric + metric.T)
    scale = max(float(symmetric.abs().max().item()), torch.finfo(metric.dtype).tiny)
    tolerance = min(
        100 * torch.finfo(metric.dtype).eps * dimension * scale,
        1e-6 * scale,
    )
    if float((metric - metric.T).abs().max().item()) > tolerance:
        raise ValueError(f"{name} must be symmetric")
    values, vectors = torch.linalg.eigh(symmetric)
    if float(values[0].item()) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    if float(values[0].item()) < 0:
        symmetric = (vectors * values.clamp_min(0)[None, :]) @ vectors.T
    return 0.5 * (symmetric + symmetric.T)


@torch.no_grad()
def residual_output_route_metric(output_metric: Tensor, *, gain: float = 1.0) -> Tensor:
    """Pull back through ``y = identity + gain * update``.

    The returned block matrix is ``[[O, aO], [aO, a^2 O]]``.
    """

    value = _finite_gain(gain)
    if output_metric.ndim != 2 or output_metric.shape[0] != output_metric.shape[1]:
        raise ValueError("output_metric must be square")
    output = _validate_psd(output_metric, output_metric.shape[0], "output_metric")
    return torch.cat(
        (
            torch.cat((output, value * output), dim=1),
            torch.cat((value * output, value * value * output), dim=1),
        ),
        dim=0,
    )


@torch.no_grad()
def compile_residual_route_assembly(
    update_assembly: Tensor,
    *,
    gain: float = 1.0,
    maximum_elements: int = 10_000_000,
) -> ResidualRouteAssembly:
    """Compile ``y = base + gain * A_update @ update_features``."""

    _validate_real_matrix(update_assembly, "update_assembly")
    value = _finite_gain(gain)
    if isinstance(maximum_elements, bool) or not isinstance(maximum_elements, int) or maximum_elements <= 0:
        raise ValueError("maximum_elements must be a positive integer")
    output_dimension, update_features = update_assembly.shape
    element_count = output_dimension * (output_dimension + update_features)
    if element_count > maximum_elements:
        raise ValueError(
            f"dense residual assembly would require {element_count} elements, "
            f"above the tiny-oracle limit {maximum_elements}"
        )
    identity = torch.eye(
        output_dimension,
        dtype=update_assembly.dtype,
        device=update_assembly.device,
    )
    assembly = torch.cat((identity, value * update_assembly), dim=1)
    return ResidualRouteAssembly(
        assembly=assembly,
        output_dimension=output_dimension,
        update_feature_dimension=update_features,
        gain=value,
    )


@torch.no_grad()
def pullback_residual_feature_metric(
    compiled: ResidualRouteAssembly,
    output_metric: Tensor,
) -> Tensor:
    """Pull a metric through a compiled base-plus-update-feature assembly."""

    if output_metric.dtype != compiled.assembly.dtype or output_metric.device != compiled.assembly.device:
        raise ValueError("output_metric and assembly must share dtype and device")
    output = _validate_psd(
        output_metric, compiled.output_dimension, "output_metric"
    )
    result = compiled.assembly.T @ output @ compiled.assembly
    return 0.5 * (result + result.T)


@torch.no_grad()
def pushforward_residual_covariance(
    base_covariance: Tensor,
    update_covariance: Tensor,
    base_update_cross_covariance: Tensor,
    *,
    gain: float = 1.0,
) -> Tensor:
    """Propagate route covariance while retaining identity-update cross terms."""

    value = _finite_gain(gain)
    if base_covariance.ndim != 2 or base_covariance.shape[0] != base_covariance.shape[1]:
        raise ValueError("base_covariance must be square")
    dimension = base_covariance.shape[0]
    if update_covariance.dtype != base_covariance.dtype or update_covariance.device != base_covariance.device:
        raise ValueError("all covariance tensors must share dtype and device")
    joint = joint_route_covariance(
        base_covariance, update_covariance, base_update_cross_covariance
    )
    base = joint[:dimension, :dimension]
    update = joint[dimension:, dimension:]
    cross = joint[:dimension, dimension:]
    result = (
        base
        + value * cross
        + value * cross.T
        + value * value * update
    )
    return 0.5 * (result + result.T)


@torch.no_grad()
def joint_route_covariance(
    base_covariance: Tensor,
    update_covariance: Tensor,
    base_update_cross_covariance: Tensor,
) -> Tensor:
    """Return the exact joint covariance on ``[base, update]`` routes."""

    _validate_real_matrix(base_covariance, "base_covariance")
    if base_covariance.shape[0] != base_covariance.shape[1]:
        raise ValueError("base_covariance must be square")
    dimension = base_covariance.shape[0]
    base = _validate_psd(base_covariance, dimension, "base_covariance")
    update = _validate_psd(update_covariance, dimension, "update_covariance")
    _validate_real_matrix(base_update_cross_covariance, "base_update_cross_covariance")
    if base_update_cross_covariance.shape != (dimension, dimension):
        raise ValueError(
            f"base_update_cross_covariance must have shape ({dimension}, {dimension})"
        )
    if update.dtype != base.dtype or update.device != base.device or base_update_cross_covariance.dtype != base.dtype or base_update_cross_covariance.device != base.device:
        raise ValueError("all covariance tensors must share dtype and device")
    joint = torch.cat(
        (
            torch.cat((base, base_update_cross_covariance), dim=1),
            torch.cat((base_update_cross_covariance.T, update), dim=1),
        ),
        dim=0,
    )
    # Cross covariance is unconstrained alone, but the complete joint object
    # must be PSD to be a valid route covariance.
    return _validate_psd(joint, 2 * dimension, "joint route covariance")


@torch.no_grad()
def compile_bilinear_ffn_residual_core(
    module: BilinearFFN,
    *,
    gain: float = 1.0,
    maximum_elements: int = 10_000_000,
) -> Tensor:
    """Compile ``[1; x + gain * FFN(x)]`` as a symmetric quotient core.

    The identity is lifted into the same two homogeneous input legs as the
    bilinear update. Half of each linear identity term is placed in each leg,
    so tying the two inputs recovers one copy of ``x`` while preserving core
    symmetry for the Dooms tree recursion.

    This deliberately selects the tied-input symmetric polynomial quotient.
    It is not interchangeable with the ordered left/right topology tensor.
    """

    value = _finite_gain(gain)
    if module.out_dim != module.dim:
        raise ValueError("a residual BilinearFFN must have out_dim == dim")
    if isinstance(maximum_elements, bool) or not isinstance(maximum_elements, int) or maximum_elements <= 0:
        raise ValueError("maximum_elements must be a positive integer")
    dimension = module.dim + 1
    elements = dimension**3
    if elements > maximum_elements:
        raise ValueError(
            f"dense residual FFN core would require {elements} elements, "
            f"above the tiny-oracle limit {maximum_elements}"
        )
    left = torch.cat((module.left.bias.detach()[:, None], module.left.weight.detach()), dim=1)
    right = torch.cat((module.right.bias.detach()[:, None], module.right.weight.detach()), dim=1)
    update = torch.einsum("or,ri,rj->oij", module.down.weight.detach(), left, right)
    if module.down.bias is not None:
        update[:, 0, 0] += module.down.bias.detach()
    update = 0.5 * (update + update.transpose(1, 2))
    core = update.new_zeros(dimension, dimension, dimension)
    core[0, 0, 0] = 1.0
    identity = torch.eye(module.dim, dtype=core.dtype, device=core.device)
    core[1:, 0, 1:] += 0.5 * identity
    core[1:, 1:, 0] += 0.5 * identity
    core[1:] += value * update
    if not bool(torch.isfinite(core).all()):
        raise ValueError("residual FFN core is nonfinite after compilation")
    return core
