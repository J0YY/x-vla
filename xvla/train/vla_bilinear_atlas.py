"""Exact, typed quadratic operators for bilinear VLA endpoints.

This module is the first action-facing layer of the VLA ODT program.  It does
not claim to be a global Dooms ODT of the transformer.  Instead, it exposes the
exact function-level quadratic computed by one :class:`BilinearFFN`, preserves
the action-horizon and action-coordinate axes, and supplies a metric-aware
coefficient Gram for that single core.

For a bilinear module

    y = D[(L x_bar) * (R x_bar)] + b,

and an output covector ``u``, the scalar endpoint is

    u.T y = x_bar.T Q_u x_bar,

where ``Q_u`` is symmetric and includes ``u.T b`` in its homogeneous
constant-constant entry.  The signed eigensystem of ``Q_u`` is the Pearce et
al. local interaction object.  For a positive-semidefinite output metric
``M``, :func:`coefficient_mode_gram` is the exact coordinate Gram

    G[i,k] = sum[o,p,j] C[o,i,j] M[o,p] C[p,k,j]

of the symmetrized coefficient tensor ``C``.  It becomes a one-core ODT
environment only in an isometric input gauge, or when input-leg metrics are
transported explicitly.  It is deliberately not a surrogate for the missing
downstream environment of the full VLA.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Sequence

import torch

from xvla.nn.bilinear import BilinearFFN
from xvla.nn.homogeneous import to_homogeneous_matrix
from xvla.nn.product_routing import ProductRoutingHead


Tensor = torch.Tensor


def _analysis_dtype(module: BilinearFFN, dtype: torch.dtype | None) -> torch.dtype:
    requested = module.left.weight.dtype if dtype is None else dtype
    try:
        probe = torch.empty((), dtype=requested)
    except (RuntimeError, TypeError) as error:
        raise TypeError(f"unsupported analysis dtype {requested}") from error
    if not probe.is_floating_point() or probe.is_complex():
        raise TypeError("analysis dtype must be a real floating-point dtype")
    parameter_dtype = requested
    if parameter_dtype in (torch.float16, torch.bfloat16, torch.float32):
        return torch.float64
    return parameter_dtype


def _matrix_tolerance(matrix: Tensor, tolerance: float | None) -> float:
    if tolerance is not None:
        if not isfinite(tolerance) or tolerance < 0:
            raise ValueError("matrix tolerance must be finite and nonnegative")
        return tolerance
    scale = max(float(matrix.detach().double().abs().max().item()), 1.0)
    source_eps = torch.finfo(matrix.dtype).eps
    raw = 1000.0 * source_eps * matrix.shape[-1] * scale
    return min(raw, 1.0e-6 * scale)


def _validate_symmetric_matrix(
    matrix: Tensor,
    *,
    name: str,
    tolerance: float | None,
) -> tuple[Tensor, float]:
    if not matrix.is_floating_point() or matrix.is_complex():
        raise TypeError(f"{name} must have a real floating-point dtype")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} must contain only finite values")
    resolved_tolerance = _matrix_tolerance(matrix, tolerance)
    work = matrix.double()
    skew = float((work - work.transpose(-1, -2)).abs().max().item())
    if skew > resolved_tolerance:
        raise ValueError(f"{name} is not symmetric: max skew {skew:.3e}")
    return 0.5 * (work + work.transpose(-1, -2)), resolved_tolerance


def _validate_psd_matrix(
    matrix: Tensor,
    *,
    name: str,
    tolerance: float | None,
) -> Tensor:
    symmetric, resolved_tolerance = _validate_symmetric_matrix(
        matrix, name=name, tolerance=tolerance
    )
    minimum = float(torch.linalg.eigvalsh(symmetric).min().item())
    if minimum < -resolved_tolerance:
        raise ValueError(f"{name} is not positive semidefinite: min eigenvalue {minimum:.3e}")
    return symmetric


def _validate_bilinear_module(module: BilinearFFN) -> None:
    if not isinstance(module, BilinearFFN):
        raise TypeError(f"expected BilinearFFN, got {type(module).__name__}")


@torch.no_grad()
def symmetric_output_quadratics(
    module: BilinearFFN,
    *,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return every output coordinate as an exact symmetric quadratic.

    The result has shape ``(out_dim, dim + 1, dim + 1)`` and satisfies

    ``module(x)[b, o] == einsum('i,oij,j', x_bar[b], Q, x_bar[b])``

    up to the arithmetic precision used for the comparison.  Unlike
    :meth:`BilinearFFN.dense_core`, this routine also supports a down bias by
    placing it in the constant-constant coefficient.
    """

    _validate_bilinear_module(module)
    target_dtype = _analysis_dtype(module, dtype)
    device = module.left.weight.device
    left = to_homogeneous_matrix(module.left.weight, module.left.bias).to(
        device=device, dtype=target_dtype
    )
    right = to_homogeneous_matrix(module.right.weight, module.right.bias).to(
        device=device, dtype=target_dtype
    )
    down = module.down.weight.to(device=device, dtype=target_dtype)
    core = torch.einsum("or,ri,rj->oij", down, left, right)
    core = 0.5 * (core + core.transpose(-1, -2))
    if module.down.bias is not None:
        core[:, 0, 0] += module.down.bias.to(device=device, dtype=target_dtype)
    return core


@torch.no_grad()
def scalar_quadratic(
    module: BilinearFFN,
    output_covector: Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the exact signed quadratic for one scalar output endpoint."""

    core = symmetric_output_quadratics(module, dtype=dtype)
    u = torch.as_tensor(output_covector, device=core.device, dtype=core.dtype)
    if u.ndim != 1 or u.shape[0] != module.out_dim:
        raise ValueError(
            f"output_covector must have shape ({module.out_dim},), got {tuple(u.shape)}"
        )
    return torch.einsum("o,oij->ij", u, core)


@torch.no_grad()
def evaluate_quadratics(quadratics: Tensor, x: Tensor) -> Tensor:
    """Evaluate a batch of homogeneous quadratics while retaining typed axes.

    ``quadratics`` may have any leading axes and must end in ``(d+1, d+1)``.
    ``x`` has shape ``(batch, d)``.  The returned shape is
    ``(batch, *quadratics.shape[:-2])``.
    """

    if quadratics.ndim < 2 or quadratics.shape[-1] != quadratics.shape[-2]:
        raise ValueError("quadratics must end in two equal matrix dimensions")
    if x.ndim != 2 or x.shape[-1] + 1 != quadratics.shape[-1]:
        raise ValueError(
            "x must have shape (batch, quadratics.shape[-1] - 1)"
        )
    x_work = x.to(device=quadratics.device, dtype=quadratics.dtype)
    ones = torch.ones(
        x_work.shape[0], 1, device=x_work.device, dtype=x_work.dtype
    )
    homogeneous = torch.cat((ones, x_work), dim=-1)
    return torch.einsum("bi,...ij,bj->b...", homogeneous, quadratics, homogeneous)


@torch.no_grad()
def coefficient_mode_gram(
    module: BilinearFFN,
    output_metric: Tensor,
    *,
    other_leg_metric: Tensor | None = None,
    dtype: torch.dtype | None = None,
    psd_tolerance: float | None = None,
) -> Tensor:
    """Return an exact coordinate Gram for one input leg of a bilinear core.

    ``output_metric`` is a symmetric positive-semidefinite matrix on output
    coordinates.  ``other_leg_metric`` is the contravariant metric used when
    contracting the other quadratic input leg and defaults to the identity.
    The returned matrix is

    ``G[i,k] = sum[o,p,j,l] C[o,i,j] M[o,p] C[p,k,l] K[j,l]``.

    With ``K=I`` this is the ordinary Euclidean coefficient-matricization Gram
    of the symmetrized function-level core.  It is a Dooms-style ODT environment
    only when the input bond is already in an isometric canonical gauge.  Under
    a nonorthogonal input change ``x'=H x``, callers must transport
    ``K'=H K H.T`` and use a correspondingly transported retained-leg metric.
    Raw eigenvectors are otherwise coordinate dependent.
    """

    core = symmetric_output_quadratics(module, dtype=dtype)
    metric_input = torch.as_tensor(output_metric, device=core.device)
    expected = (module.out_dim, module.out_dim)
    if metric_input.ndim != 2 or tuple(metric_input.shape) != expected:
        raise ValueError(
            f"output_metric must have shape {expected}, got {tuple(metric_input.shape)}"
        )
    metric = _validate_psd_matrix(
        metric_input, name="output_metric", tolerance=psd_tolerance
    ).to(dtype=core.dtype)
    leg_size = module.dim + 1
    if other_leg_metric is None:
        leg_metric = torch.eye(leg_size, device=core.device, dtype=core.dtype)
    else:
        leg_input = torch.as_tensor(other_leg_metric, device=core.device)
        if leg_input.ndim != 2 or tuple(leg_input.shape) != (leg_size, leg_size):
            raise ValueError(
                "other_leg_metric must have shape "
                f"({leg_size}, {leg_size}), got {tuple(leg_input.shape)}"
            )
        leg_metric = _validate_psd_matrix(
            leg_input, name="other_leg_metric", tolerance=psd_tolerance
        ).to(dtype=core.dtype)
    gram = torch.einsum("oij,op,pkl,jl->ik", core, metric, core, leg_metric)
    return 0.5 * (gram + gram.T)


@torch.no_grad()
def signed_eigensystem(
    quadratic: Tensor,
    *,
    symmetry_tolerance: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Diagonalize symmetric quadratic matrices, sorting by decreasing ``abs(lambda)``.

    Batched leading axes are supported.  Eigenvectors are returned in columns.
    These are coordinate-basis signed interactions.  They are invariant only to
    orthogonal basis changes unless the bond has first been placed in an
    isometric gauge.  Individual eigenvectors are also non-identifiable inside
    repeated or unresolved eigenvalue clusters.
    """

    if quadratic.ndim < 2 or quadratic.shape[-1] != quadratic.shape[-2]:
        raise ValueError("quadratic must end in two equal matrix dimensions")
    sym, _ = _validate_symmetric_matrix(
        quadratic, name="quadratic", tolerance=symmetry_tolerance
    )
    sym = sym.to(dtype=quadratic.dtype)
    values, vectors = torch.linalg.eigh(sym)
    order = values.abs().argsort(dim=-1, descending=True)
    values = torch.gather(values, -1, order)
    gather_index = order.unsqueeze(-2).expand(*vectors.shape[:-2], vectors.shape[-2], order.shape[-1])
    vectors = torch.gather(vectors, -1, gather_index)
    return values, vectors


def action_metric(
    horizon: int,
    action_dim: int,
    *,
    action_indices: Sequence[int] | None = None,
    horizon_indices: Sequence[int] | None = None,
    horizon_weights: Sequence[float] | Tensor | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """Construct a diagonal PSD metric on a flattened ``horizon x action`` chunk.

    The helper preserves the physical indexing used by :class:`ProductRoutingHead`.
    It can select an action group, selected horizon steps, and optional nonnegative
    temporal weights.  Indices are validated rather than silently wrapped.
    """

    if horizon <= 0 or action_dim <= 0:
        raise ValueError("horizon and action_dim must be positive")
    actions = list(range(action_dim)) if action_indices is None else list(action_indices)
    times = list(range(horizon)) if horizon_indices is None else list(horizon_indices)
    if not actions or not times:
        raise ValueError("action and horizon selections must be nonempty")
    if any(isinstance(index, bool) or not isinstance(index, Integral) for index in actions):
        raise TypeError("action indices must be non-boolean integers")
    if any(isinstance(index, bool) or not isinstance(index, Integral) for index in times):
        raise TypeError("horizon indices must be non-boolean integers")
    actions = [int(index) for index in actions]
    times = [int(index) for index in times]
    if any(index < 0 or index >= action_dim for index in actions):
        raise ValueError("action index is outside the action dimension")
    if any(index < 0 or index >= horizon for index in times):
        raise ValueError("horizon index is outside the action horizon")
    if len(set(actions)) != len(actions) or len(set(times)) != len(times):
        raise ValueError("action and horizon indices must not contain duplicates")
    if horizon_weights is None:
        weights = torch.ones(horizon, dtype=dtype, device=device)
    else:
        weights = torch.as_tensor(horizon_weights, dtype=dtype, device=device)
        if weights.ndim != 1 or weights.shape[0] != horizon:
            raise ValueError(f"horizon_weights must have shape ({horizon},)")
        if not torch.isfinite(weights).all() or bool((weights < 0).any()):
            raise ValueError("horizon_weights must be finite and nonnegative")
    diagonal = torch.zeros(horizon * action_dim, dtype=dtype, device=device)
    for time in times:
        for action in actions:
            diagonal[time * action_dim + action] = weights[time]
    return torch.diag(diagonal)


@dataclass(frozen=True)
class ProductRoutingQuadraticAtlas:
    """Typed exact quadratics for a product-routing VLA action head."""

    center: Tensor
    factors: Tensor
    gates: Tensor
    horizon: int
    action_dim: int

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.action_dim <= 0:
            raise ValueError("horizon and action_dim must be positive")
        if self.center.ndim != 4:
            raise ValueError("center must have shape (horizon, action_dim, n, n)")
        if self.factors.ndim != 5:
            raise ValueError("factors must have shape (factor, horizon, action_dim, n, n)")
        if self.gates.ndim != 3:
            raise ValueError("gates must have shape (factor, n, n)")
        n = self.center.shape[-1]
        if self.center.shape != (self.horizon, self.action_dim, n, n):
            raise ValueError("center shape is inconsistent with the typed axes")
        if self.factors.shape[1:] != self.center.shape:
            raise ValueError("factor shapes are inconsistent with center")
        if self.gates.shape != (self.factors.shape[0], n, n):
            raise ValueError("gate shapes are inconsistent with factors")
        tensors = (self.center, self.factors, self.gates)
        if any(tensor.device != self.center.device for tensor in tensors[1:]):
            raise ValueError("atlas tensors must share one device")
        if any(tensor.dtype != self.center.dtype for tensor in tensors[1:]):
            raise ValueError("atlas tensors must share one dtype")
        if not self.center.is_floating_point() or self.center.is_complex():
            raise TypeError("atlas tensors must use a real floating-point dtype")

    @property
    def n_factors(self) -> int:
        return self.factors.shape[0]

    @torch.no_grad()
    def route(self, signs: Tensor) -> Tensor:
        """Return route-conditioned action quadratics without flattening time/action."""

        signs_work = torch.as_tensor(
            signs, device=self.center.device, dtype=self.center.dtype
        )
        if signs_work.ndim != 1 or signs_work.shape[0] != self.n_factors:
            raise ValueError(f"signs must have shape ({self.n_factors},)")
        if not bool(torch.all((signs_work == -1) | (signs_work == 1))):
            raise ValueError("route signs must be exactly -1 or +1")
        return self.center + torch.einsum("g,ghaij->haij", signs_work, self.factors)

    def factor_flip(self, factor: int) -> Tensor:
        """Return the exact action-quadratic change from flipping ``-1`` to ``+1``."""

        if factor < 0 or factor >= self.n_factors:
            raise IndexError("factor index is outside the product-routing head")
        return 2.0 * self.factors[factor]


@torch.no_grad()
def product_routing_quadratic_atlas(
    head: ProductRoutingHead,
    *,
    dtype: torch.dtype | None = None,
) -> ProductRoutingQuadraticAtlas:
    """Extract exact center, factor, and gate quadratics from a product head."""

    if not isinstance(head, ProductRoutingHead):
        raise TypeError(f"expected ProductRoutingHead, got {type(head).__name__}")
    center = symmetric_output_quadratics(head.center, dtype=dtype).reshape(
        head.horizon, head.action_dim, head.dim + 1, head.dim + 1
    )
    factors = torch.stack(
        [symmetric_output_quadratics(factor, dtype=dtype) for factor in head.factors],
        dim=0,
    ).reshape(head.G, head.horizon, head.action_dim, head.dim + 1, head.dim + 1)
    gates = torch.stack(
        [scalar_quadratic(gate, torch.ones(1, device=gate.left.weight.device), dtype=dtype)
         for gate in head.gates],
        dim=0,
    )
    return ProductRoutingQuadraticAtlas(
        center=center,
        factors=factors,
        gates=gates,
        horizon=head.horizon,
        action_dim=head.action_dim,
    )
