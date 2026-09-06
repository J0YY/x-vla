"""Projective Quotient Balanced Decomposition (PQBD).

PQBD is a separately named extension for rational maps.  It is not canonical
ODT and this module never relabels it as such.  A rational vector is represented
by homogeneous projective coordinates ``(N, D)`` with value ``N / D``.  The
quotient map removes the unobservable radial gauge ``(N, D) ~ (aN, aD)``.

The local input pullback for ``f(x) = N(x) / D(x)`` is

    J_f = (D J_N - N grad(D)^T) / D^2.

Its metric contains the two numerator-denominator cross terms.  Omitting those
terms breaks projective gauge invariance.  PQBD balances this exact quotient
pullback against a declared upstream metric on its positive support.  The
result is point-local whenever the supplied Jacobians are point-local.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from xvla.nn.attention import BilinearAttention
from xvla.nn.bilinear import BilinearFFN
from xvla.nn.normalization import RationalNorm


Tensor = torch.Tensor
PQBD_OBJECT_KIND = "projective_quotient_balanced_decomposition_noncanonical"
PQBD_CLAIM_BOUNDARY = (
    "PQBD is a gauge-invariant, support-balanced decomposition of an exact "
    "quotient Jacobian pullback. It is not canonical ODT, not rational ODT, "
    "and not a global decomposition of a deep VLA unless a separate exact "
    "global projective compilation has been supplied."
)


@dataclass(frozen=True)
class DenominatorLedgerEntry:
    """Observed denominator accounting for one exact projective operation.

    Bounds are evaluations on the supplied tensor, not global positivity
    certificates over an unstated domain.
    """

    label: str
    numerator_degree_upper_bound: int
    denominator_degree_upper_bound: int
    observed_minimum_absolute_denominator: float
    observed_maximum_absolute_denominator: float
    observed_all_positive: bool
    observation_count: int
    bound_kind: str = "observed_tensor_only"


@dataclass(frozen=True)
class ProjectivePair:
    """One vector numerator and one broadcastable scalar denominator."""

    numerator: Tensor
    denominator: Tensor
    numerator_degree_upper_bound: int
    denominator_degree_upper_bound: int
    denominator_ledger: tuple[DenominatorLedgerEntry, ...] = ()


@dataclass(frozen=True)
class QuotientPullback:
    """Exact expanded pullback, including both cross terms."""

    jacobian: Tensor
    metric: Tensor
    numerator_term: Tensor
    cross_term_left: Tensor
    cross_term_right: Tensor
    denominator_term: Tensor


@dataclass(frozen=True)
class ProjectiveOutputPullback:
    """Metric in the ambient ``[N; D]`` coordinates."""

    quotient_jacobian: Tensor
    metric: Tensor
    radial_residual: Tensor


@dataclass(frozen=True)
class ProjectiveQuotientBalancedDecomposition:
    """PQBD eigensystem on the positive support of an upstream metric."""

    eigenvalues: Tensor
    whitened_vectors: Tensor
    state_directions: Tensor
    quotient_pullback: Tensor
    upstream_rank: int
    upstream_tolerance: float
    object_kind: str = PQBD_OBJECT_KIND
    claim_boundary: str = PQBD_CLAIM_BOUNDARY


def _validate_real_finite(tensor: Tensor, *, name: str) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(f"{name} must have a real floating-point dtype")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_degree(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _validate_pair(pair: ProjectivePair) -> None:
    if not isinstance(pair, ProjectivePair):
        raise TypeError("pair must be a ProjectivePair")
    _validate_real_finite(pair.numerator, name="pair.numerator")
    _validate_real_finite(pair.denominator, name="pair.denominator")
    if pair.numerator.device != pair.denominator.device or pair.numerator.dtype != pair.denominator.dtype:
        raise ValueError("projective numerator and denominator must share dtype and device")
    if pair.numerator.ndim < 1:
        raise ValueError("projective numerator must have an output axis")
    expected = pair.numerator.shape[:-1] + (1,)
    if pair.denominator.shape not in (torch.Size([]), torch.Size(expected)):
        raise ValueError(
            "projective denominator must be scalar or have the numerator batch shape plus a singleton axis"
        )
    _validate_degree(pair.numerator_degree_upper_bound, name="numerator degree")
    _validate_degree(pair.denominator_degree_upper_bound, name="denominator degree")


def _denominator_threshold(tensor: Tensor, minimum_absolute_denominator: float) -> float:
    if (
        isinstance(minimum_absolute_denominator, bool)
        or not isinstance(minimum_absolute_denominator, (int, float))
        or not torch.isfinite(torch.tensor(float(minimum_absolute_denominator)))
        or minimum_absolute_denominator <= 0
    ):
        raise ValueError("minimum_absolute_denominator must be finite and positive")
    return max(
        float(minimum_absolute_denominator),
        100.0 * torch.finfo(tensor.dtype).eps,
    )


def denominator_ledger_entry(
    label: str,
    denominator: Tensor,
    *,
    numerator_degree_upper_bound: int,
    denominator_degree_upper_bound: int,
    minimum_absolute_denominator: float = 1e-12,
) -> DenominatorLedgerEntry:
    """Validate and record an observed denominator without claiming a domain proof."""

    if not isinstance(label, str) or not label:
        raise ValueError("label must be a nonempty string")
    _validate_real_finite(denominator, name="denominator")
    _validate_degree(numerator_degree_upper_bound, name="numerator degree")
    _validate_degree(denominator_degree_upper_bound, name="denominator degree")
    if denominator.numel() == 0:
        raise ValueError("denominator must contain at least one value")
    threshold = _denominator_threshold(denominator, minimum_absolute_denominator)
    minimum = float(denominator.abs().min().item())
    if minimum <= threshold:
        raise ValueError(
            f"observed denominator at {label!r} is at or below the safety threshold: "
            f"{minimum:.6g} <= {threshold:.6g}"
        )
    return DenominatorLedgerEntry(
        label=label,
        numerator_degree_upper_bound=numerator_degree_upper_bound,
        denominator_degree_upper_bound=denominator_degree_upper_bound,
        observed_minimum_absolute_denominator=minimum,
        observed_maximum_absolute_denominator=float(denominator.abs().max().item()),
        observed_all_positive=bool((denominator > 0).all()),
        observation_count=denominator.numel(),
    )


def projective_value(
    pair: ProjectivePair,
    *,
    minimum_absolute_denominator: float = 1e-12,
) -> Tensor:
    """Evaluate one projective pair after a fail-closed denominator check."""

    _validate_pair(pair)
    _ = denominator_ledger_entry(
        "projective_value",
        pair.denominator,
        numerator_degree_upper_bound=pair.numerator_degree_upper_bound,
        denominator_degree_upper_bound=pair.denominator_degree_upper_bound,
        minimum_absolute_denominator=minimum_absolute_denominator,
    )
    return pair.numerator / pair.denominator


def constant_projective(value: Tensor, *, input_degree: int = 0) -> ProjectivePair:
    """Embed a polynomial tensor as a denominator-one projective pair."""

    _validate_real_finite(value, name="value")
    _validate_degree(input_degree, name="input_degree")
    denominator = value.new_ones(value.shape[:-1] + (1,))
    return ProjectivePair(value, denominator, input_degree, 0)


def projective_rescale(
    pair: ProjectivePair,
    scale: Tensor | float,
    *,
    label: str = "projective_rescale",
    minimum_absolute_scale: float = 1e-12,
) -> ProjectivePair:
    """Apply a nonzero degree-zero projective gauge rescaling."""

    _validate_pair(pair)
    scale_tensor = torch.as_tensor(scale, dtype=pair.numerator.dtype, device=pair.numerator.device)
    _validate_real_finite(scale_tensor, name="scale")
    try:
        scaled_denominator = pair.denominator * scale_tensor
        scaled_numerator = pair.numerator * scale_tensor
    except RuntimeError as exc:
        raise ValueError("scale is not broadcastable to the projective pair") from exc
    if scaled_numerator.shape != pair.numerator.shape or scaled_denominator.shape != pair.denominator.shape:
        raise ValueError(
            "scale must preserve numerator and denominator shapes, use a scalar or one scalar per batch item"
        )
    entry = denominator_ledger_entry(
        label,
        scaled_denominator,
        numerator_degree_upper_bound=pair.numerator_degree_upper_bound,
        denominator_degree_upper_bound=pair.denominator_degree_upper_bound,
        minimum_absolute_denominator=minimum_absolute_scale,
    )
    return ProjectivePair(
        scaled_numerator,
        scaled_denominator,
        pair.numerator_degree_upper_bound,
        pair.denominator_degree_upper_bound,
        pair.denominator_ledger + (entry,),
    )


def projective_add(
    left: ProjectivePair,
    right: ProjectivePair,
    *,
    label: str = "projective_add",
    minimum_absolute_denominator: float = 1e-12,
) -> ProjectivePair:
    """Add exact rational vectors by cross multiplication."""

    _validate_pair(left)
    _validate_pair(right)
    if left.numerator.shape != right.numerator.shape:
        raise ValueError("projective addends must have identical numerator shapes")
    if left.numerator.dtype != right.numerator.dtype or left.numerator.device != right.numerator.device:
        raise ValueError("projective addends must share dtype and device")
    numerator = left.numerator * right.denominator + right.numerator * left.denominator
    denominator = left.denominator * right.denominator
    numerator_degree = max(
        left.numerator_degree_upper_bound + right.denominator_degree_upper_bound,
        right.numerator_degree_upper_bound + left.denominator_degree_upper_bound,
    )
    denominator_degree = left.denominator_degree_upper_bound + right.denominator_degree_upper_bound
    entry = denominator_ledger_entry(
        label,
        denominator,
        numerator_degree_upper_bound=numerator_degree,
        denominator_degree_upper_bound=denominator_degree,
        minimum_absolute_denominator=minimum_absolute_denominator,
    )
    return ProjectivePair(
        numerator,
        denominator,
        numerator_degree,
        denominator_degree,
        left.denominator_ledger + right.denominator_ledger + (entry,),
    )


def projective_product(
    left: ProjectivePair,
    right: ProjectivePair,
    *,
    label: str = "projective_product",
    minimum_absolute_denominator: float = 1e-12,
) -> ProjectivePair:
    """Multiply exact scalar or elementwise-compatible projective pairs."""

    _validate_pair(left)
    _validate_pair(right)
    if left.numerator.dtype != right.numerator.dtype or left.numerator.device != right.numerator.device:
        raise ValueError("projective factors must share dtype and device")
    try:
        numerator = left.numerator * right.numerator
        denominator = left.denominator * right.denominator
    except RuntimeError as exc:
        raise ValueError("projective factors are not broadcast-compatible") from exc
    numerator_degree = left.numerator_degree_upper_bound + right.numerator_degree_upper_bound
    denominator_degree = left.denominator_degree_upper_bound + right.denominator_degree_upper_bound
    entry = denominator_ledger_entry(
        label,
        denominator,
        numerator_degree_upper_bound=numerator_degree,
        denominator_degree_upper_bound=denominator_degree,
        minimum_absolute_denominator=minimum_absolute_denominator,
    )
    return ProjectivePair(
        numerator,
        denominator,
        numerator_degree,
        denominator_degree,
        left.denominator_ledger + right.denominator_ledger + (entry,),
    )


def affine_projective(pair: ProjectivePair, affine: nn.Linear) -> ProjectivePair:
    """Apply an affine map without dividing the projective pair."""

    _validate_pair(pair)
    if not isinstance(affine, nn.Linear):
        raise TypeError("affine must be torch.nn.Linear")
    if affine.in_features != pair.numerator.shape[-1]:
        raise ValueError("affine input width differs from the projective numerator")
    if affine.weight.dtype != pair.numerator.dtype or affine.weight.device != pair.numerator.device:
        raise ValueError("affine and projective pair must share dtype and device")
    numerator = torch.nn.functional.linear(pair.numerator, affine.weight, None)
    degree = pair.numerator_degree_upper_bound
    if affine.bias is not None:
        numerator = numerator + affine.bias * pair.denominator
        degree = max(degree, pair.denominator_degree_upper_bound)
    return ProjectivePair(
        numerator,
        pair.denominator,
        degree,
        pair.denominator_degree_upper_bound,
        pair.denominator_ledger,
    )


def rational_norm_projective(
    x: Tensor,
    module: RationalNorm,
    *,
    label: str = "rational_norm",
    minimum_absolute_denominator: float = 1e-8,
) -> ProjectivePair:
    """Compile the deployed Padé ``RationalNorm`` call to an exact pair.

    For configured Padé degree ``m``, the numerator degree is at most
    ``1 + 2m`` and the denominator degree is at most ``2m`` in the direct
    input coordinates. The arithmetic mirrors ``RationalNorm.forward``. It
    preserves float64 for certification and retains the deployed float32 path
    for float32 and lower-precision model inputs.
    """

    _validate_real_finite(x, name="x")
    if not isinstance(module, RationalNorm) or module.variant != "pade":
        raise ValueError("PQBD RationalNorm compilation requires variant='pade'")
    if module.training and not module.frozen:
        raise ValueError("RationalNorm must be eval-mode or frozen before projective compilation")
    if not bool(module.initialized):
        raise ValueError("RationalNorm running_ms must be initialized")
    if module.deg < 0 or module.pa.numel() != module.deg + 1 or module.pb.numel() != module.deg + 1:
        raise ValueError("RationalNorm coefficient lengths disagree with its degree")
    if not bool(torch.isfinite(module.pa).all() and torch.isfinite(module.pb).all()):
        raise ValueError("RationalNorm coefficients must be finite")
    if abs(float(module.pb[0].item()) - 1.0) > 10.0 * torch.finfo(module.pb.dtype).eps:
        raise ValueError("RationalNorm denominator must use the declared b0=1 convention")

    xf = x if x.dtype in (torch.float32, torch.float64) else x.float()
    mean_square = xf.pow(2).mean(dim=-1, keepdim=True) + module.eps
    scale = module.running_ms.clamp_min(1e-12)
    v = mean_square / scale
    numerator_polynomial = torch.zeros_like(v + module.pa[0])
    denominator_polynomial = torch.zeros_like(v + module.pb[0])
    power = torch.ones_like(v)
    for degree in range(module.deg + 1):
        numerator_polynomial = numerator_polynomial + module.pa[degree] * power
        denominator_polynomial = denominator_polynomial + module.pb[degree] * power
        power = power * v
    numerator = xf * numerator_polynomial * torch.rsqrt(scale)
    numerator = numerator.to(dtype=x.dtype)
    denominator = denominator_polynomial.to(dtype=x.dtype)
    numerator_degree = 1 + 2 * module.deg
    denominator_degree = 2 * module.deg
    entry = denominator_ledger_entry(
        label,
        denominator,
        numerator_degree_upper_bound=numerator_degree,
        denominator_degree_upper_bound=denominator_degree,
        minimum_absolute_denominator=minimum_absolute_denominator,
    )
    return ProjectivePair(
        numerator,
        denominator,
        numerator_degree,
        denominator_degree,
        (entry,),
    )


def quotient_jacobian(
    numerator: Tensor,
    denominator: Tensor | float,
    numerator_jacobian: Tensor,
    denominator_gradient: Tensor,
    *,
    minimum_absolute_denominator: float = 1e-12,
) -> Tensor:
    """Return ``d(N/D)/dx`` with the quotient cross contribution intact."""

    _validate_real_finite(numerator, name="numerator")
    _validate_real_finite(numerator_jacobian, name="numerator_jacobian")
    _validate_real_finite(denominator_gradient, name="denominator_gradient")
    if numerator.ndim != 1 or numerator_jacobian.ndim != 2 or denominator_gradient.ndim != 1:
        raise ValueError("quotient Jacobian inputs must be a vector, matrix, and vector")
    if numerator_jacobian.shape != (numerator.numel(), denominator_gradient.numel()):
        raise ValueError("numerator Jacobian shape is inconsistent")
    denominator_tensor = torch.as_tensor(
        denominator, dtype=numerator.dtype, device=numerator.device
    )
    _validate_real_finite(denominator_tensor, name="denominator")
    if denominator_tensor.numel() != 1:
        raise ValueError("denominator must be scalar")
    if any(
        tensor.dtype != numerator.dtype or tensor.device != numerator.device
        for tensor in (numerator_jacobian, denominator_gradient)
    ):
        raise ValueError("quotient Jacobian inputs must share dtype and device")
    threshold = _denominator_threshold(numerator, minimum_absolute_denominator)
    scalar = denominator_tensor.reshape(())
    if abs(float(scalar.item())) <= threshold:
        raise ValueError("quotient denominator is at or below the safety threshold")
    return (
        scalar * numerator_jacobian
        - numerator[:, None] * denominator_gradient[None, :]
    ) / scalar.square()


def _symmetric_psd(metric: Tensor, dimension: int, *, name: str) -> Tensor:
    _validate_real_finite(metric, name=name)
    if metric.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    symmetric = 0.5 * (metric + metric.T)
    scale = max(float(symmetric.abs().max().item()), torch.finfo(metric.dtype).tiny)
    tolerance = min(
        100.0 * torch.finfo(metric.dtype).eps * dimension * scale,
        1e-6 * scale,
    )
    if float((metric - metric.T).abs().max().item()) > tolerance:
        raise ValueError(f"{name} must be symmetric")
    values, vectors = torch.linalg.eigh(symmetric)
    if float(values[0].item()) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    if float(values[0].item()) < 0:
        symmetric = (vectors * values.clamp_min(0)[None, :]) @ vectors.T
        symmetric = 0.5 * (symmetric + symmetric.T)
    return symmetric


def quotient_metric_pullback(
    numerator: Tensor,
    denominator: Tensor | float,
    numerator_jacobian: Tensor,
    denominator_gradient: Tensor,
    output_metric: Tensor,
    *,
    minimum_absolute_denominator: float = 1e-12,
) -> QuotientPullback:
    """Expand ``J_f.T M J_f`` into all four quotient-rule terms."""

    jacobian = quotient_jacobian(
        numerator,
        denominator,
        numerator_jacobian,
        denominator_gradient,
        minimum_absolute_denominator=minimum_absolute_denominator,
    )
    if output_metric.dtype != numerator.dtype or output_metric.device != numerator.device:
        raise ValueError("output_metric must share numerator dtype and device")
    metric = _symmetric_psd(output_metric, numerator.numel(), name="output_metric")
    scalar = torch.as_tensor(denominator, dtype=numerator.dtype, device=numerator.device).reshape(())
    inverse_fourth = scalar.pow(-4)
    projected_numerator = numerator_jacobian.T @ metric @ numerator
    numerator_term = scalar.square() * (numerator_jacobian.T @ metric @ numerator_jacobian) * inverse_fourth
    cross_left = -scalar * projected_numerator[:, None] * denominator_gradient[None, :] * inverse_fourth
    cross_right = cross_left.T
    denominator_term = (
        (numerator @ metric @ numerator)
        * denominator_gradient[:, None]
        * denominator_gradient[None, :]
        * inverse_fourth
    )
    expanded = numerator_term + cross_left + cross_right + denominator_term
    direct = jacobian.T @ metric @ jacobian
    scale = max(float(direct.abs().max().item()), 1.0)
    tolerance = 500.0 * torch.finfo(direct.dtype).eps * max(direct.shape) * scale
    if float((expanded - direct).abs().max().item()) > tolerance:
        raise RuntimeError("expanded quotient pullback disagrees with direct Jacobian contraction")
    return QuotientPullback(
        jacobian=jacobian,
        metric=0.5 * (expanded + expanded.T),
        numerator_term=numerator_term,
        cross_term_left=cross_left,
        cross_term_right=cross_right,
        denominator_term=denominator_term,
    )


def projective_output_pullback(
    numerator: Tensor,
    denominator: Tensor | float,
    output_metric: Tensor,
    *,
    minimum_absolute_denominator: float = 1e-12,
) -> ProjectiveOutputPullback:
    """Pull an output metric into ambient projective coordinates ``[N; D]``."""

    _validate_real_finite(numerator, name="numerator")
    if numerator.ndim != 1:
        raise ValueError("numerator must be a vector")
    if output_metric.dtype != numerator.dtype or output_metric.device != numerator.device:
        raise ValueError("output_metric must share numerator dtype and device")
    metric = _symmetric_psd(output_metric, numerator.numel(), name="output_metric")
    scalar = torch.as_tensor(denominator, dtype=numerator.dtype, device=numerator.device)
    _validate_real_finite(scalar, name="denominator")
    if scalar.numel() != 1:
        raise ValueError("denominator must be scalar")
    threshold = _denominator_threshold(numerator, minimum_absolute_denominator)
    scalar = scalar.reshape(())
    if abs(float(scalar.item())) <= threshold:
        raise ValueError("projective denominator is at or below the safety threshold")
    quotient = torch.cat(
        (
            torch.eye(numerator.numel(), dtype=numerator.dtype, device=numerator.device) / scalar,
            (-numerator / scalar.square())[:, None],
        ),
        dim=1,
    )
    pullback = quotient.T @ metric @ quotient
    radial = torch.cat((numerator, scalar[None]))
    residual = pullback @ radial
    return ProjectiveOutputPullback(
        quotient_jacobian=quotient,
        metric=0.5 * (pullback + pullback.T),
        radial_residual=residual,
    )


def projective_quotient_balanced_decomposition(
    upstream_metric: Tensor,
    numerator: Tensor,
    denominator: Tensor | float,
    numerator_jacobian: Tensor,
    denominator_gradient: Tensor,
    output_metric: Tensor,
    *,
    support_rtol: float = 1e-12,
    support_rank: int | None = None,
    minimum_absolute_denominator: float = 1e-12,
) -> ProjectiveQuotientBalancedDecomposition:
    """Balance an exact local quotient pullback on ``support(upstream_metric)``."""

    if (
        isinstance(support_rtol, bool)
        or not isinstance(support_rtol, (int, float))
        or not torch.isfinite(torch.tensor(float(support_rtol)))
        or not 0 <= support_rtol < 1
    ):
        raise ValueError("support_rtol must be finite and in [0, 1)")
    dimension = denominator_gradient.numel()
    if upstream_metric.dtype != numerator.dtype or upstream_metric.device != numerator.device:
        raise ValueError("upstream_metric must share numerator dtype and device")
    upstream = _symmetric_psd(upstream_metric, dimension, name="upstream_metric")
    pullback = quotient_metric_pullback(
        numerator,
        denominator,
        numerator_jacobian,
        denominator_gradient,
        output_metric,
        minimum_absolute_denominator=minimum_absolute_denominator,
    ).metric
    if support_rank is not None and (
        isinstance(support_rank, bool)
        or not isinstance(support_rank, int)
        or not 0 <= support_rank <= dimension
    ):
        raise ValueError("support_rank must be an integer in [0, dimension]")
    values, vectors = torch.linalg.eigh(upstream)
    scale = max(float(values[-1].item()), 0.0)
    tolerance = max(
        float(support_rtol) * scale,
        100.0 * torch.finfo(upstream.dtype).eps * dimension * scale,
    ) if scale else 0.0
    keep = values > tolerance
    inferred_rank = int(keep.sum().item())
    if support_rank is not None and support_rank != inferred_rank:
        raise ValueError(
            "support_rank must equal the full positive numerical support; "
            "Euclidean support truncation is not gauge covariant"
        )
    if not bool(keep.any()):
        empty = upstream.new_empty(0)
        return ProjectiveQuotientBalancedDecomposition(
            eigenvalues=empty,
            whitened_vectors=upstream.new_empty(0, 0),
            state_directions=upstream.new_empty(dimension, 0),
            quotient_pullback=pullback,
            upstream_rank=0,
            upstream_tolerance=tolerance,
        )
    factor = vectors[:, keep] * values[keep].sqrt()[None, :]
    balanced = factor.T @ pullback @ factor
    balanced = 0.5 * (balanced + balanced.T)
    eigvals, eigvecs = torch.linalg.eigh(balanced)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(0)
    eigvecs = eigvecs[:, order]
    return ProjectiveQuotientBalancedDecomposition(
        eigenvalues=eigvals,
        whitened_vectors=eigvecs,
        state_directions=factor @ eigvecs,
        quotient_pullback=pullback,
        upstream_rank=int(keep.sum().item()),
        upstream_tolerance=tolerance,
    )


def rational_ffn_residual_projective(
    x: Tensor,
    normalization: RationalNorm,
    ffn: BilinearFFN,
    gain: Tensor | float,
    *,
    label: str = "rational_ffn_residual",
    minimum_absolute_denominator: float = 1e-8,
) -> ProjectivePair:
    """Exact replay of ``x + gain * ffn(RationalNorm(x))`` as one pair."""

    if not isinstance(ffn, BilinearFFN):
        raise TypeError("ffn must be BilinearFFN")
    normalized = rational_norm_projective(
        x,
        normalization,
        label=f"{label}.normalization",
        minimum_absolute_denominator=minimum_absolute_denominator,
    )
    left = affine_projective(normalized, ffn.left)
    right = affine_projective(normalized, ffn.right)
    product = projective_product(
        left,
        right,
        label=f"{label}.bilinear_product",
        minimum_absolute_denominator=minimum_absolute_denominator,
    )
    branch = affine_projective(product, ffn.down)
    gain_tensor = torch.as_tensor(gain, dtype=x.dtype, device=x.device)
    _validate_real_finite(gain_tensor, name="gain")
    if gain_tensor.numel() != 1:
        raise ValueError("gain must be a fixed scalar")
    gain_tensor = gain_tensor.reshape(())
    branch = ProjectivePair(
        branch.numerator * gain_tensor,
        branch.denominator,
        branch.numerator_degree_upper_bound,
        branch.denominator_degree_upper_bound,
        branch.denominator_ledger,
    )
    residual = projective_add(
        constant_projective(x, input_degree=1),
        branch,
        label=f"{label}.residual_add",
        minimum_absolute_denominator=minimum_absolute_denominator,
    )
    return residual


def _linear_split(linear: nn.Linear, tokens: Tensor, heads: int) -> Tensor:
    batch, positions, dimension = tokens.shape
    return linear(tokens).reshape(batch, positions, heads, dimension // heads).transpose(1, 2)


def _projective_dot(
    left: ProjectivePair,
    right: ProjectivePair,
    label: str,
    *,
    minimum_absolute_denominator: float,
) -> ProjectivePair:
    _validate_pair(left)
    _validate_pair(right)
    if left.numerator.shape != right.numerator.shape:
        raise ValueError("projective dot operands must have identical shapes")
    numerator = (left.numerator * right.numerator).sum(dim=-1, keepdim=True)
    denominator = left.denominator * right.denominator
    numerator_degree = left.numerator_degree_upper_bound + right.numerator_degree_upper_bound
    denominator_degree = left.denominator_degree_upper_bound + right.denominator_degree_upper_bound
    entry = denominator_ledger_entry(
        label,
        denominator,
        numerator_degree_upper_bound=numerator_degree,
        denominator_degree_upper_bound=denominator_degree,
        minimum_absolute_denominator=minimum_absolute_denominator,
    )
    return ProjectivePair(
        numerator,
        denominator,
        numerator_degree,
        denominator_degree,
        left.denominator_ledger + right.denominator_ledger + (entry,),
    )


@torch.no_grad()
def tiny_rational_attention_projective_oracle(
    module: BilinearAttention,
    tokens: Tensor,
    mask: Tensor | None = None,
    *,
    maximum_routes: int = 64,
    minimum_absolute_denominator: float = 1e-8,
) -> tuple[ProjectivePair, ...]:
    """Exact, deliberately tiny self-attention oracle for rational Q/K norms.

    It returns one projective pair per query row and retains every route
    denominator until exact cross multiplication.  This is an oracle, not a
    scalable global compiler.
    """

    if not isinstance(module, BilinearAttention) or module.qk_norm != "rational":
        raise ValueError("tiny oracle requires BilinearAttention(qk_norm='rational')")
    if module.training:
        raise ValueError("attention module must be in eval mode")
    _validate_real_finite(tokens, name="tokens")
    if tokens.ndim != 3 or tokens.shape[-1] != module.dim:
        raise ValueError("tokens must have shape (batch, positions, module.dim)")
    if isinstance(maximum_routes, bool) or not isinstance(maximum_routes, int) or maximum_routes <= 0:
        raise ValueError("maximum_routes must be a positive integer")
    batch, positions, _ = tokens.shape
    fixed_mask = module._resolve_mask(positions, tokens, mask)
    route_count = int(fixed_mask.sum().item()) * module.n_heads
    if route_count > maximum_routes:
        raise ValueError("tiny rational attention oracle route limit exceeded")

    q1_raw = _linear_split(module.wq1, tokens, module.n_heads)
    k1_raw = _linear_split(module.wk1, tokens, module.n_heads)
    q2_raw = _linear_split(module.wq2, tokens, module.n_heads)
    k2_raw = _linear_split(module.wk2, tokens, module.n_heads)
    value = _linear_split(module.wv, tokens, module.n_heads)
    branches = {
        "q1": rational_norm_projective(
            q1_raw,
            module.rn_q1,
            label="attention.q1",
            minimum_absolute_denominator=minimum_absolute_denominator,
        ),
        "k1": rational_norm_projective(
            k1_raw,
            module.rn_k1,
            label="attention.k1",
            minimum_absolute_denominator=minimum_absolute_denominator,
        ),
        "q2": rational_norm_projective(
            q2_raw,
            module.rn_q2,
            label="attention.q2",
            minimum_absolute_denominator=minimum_absolute_denominator,
        ),
        "k2": rational_norm_projective(
            k2_raw,
            module.rn_k2,
            label="attention.k2",
            minimum_absolute_denominator=minimum_absolute_denominator,
        ),
    }
    row_scales = module._row_scale_vec(fixed_mask)
    results = []
    for query in range(positions):
        bias = module.wo.bias.expand(batch, -1)
        accumulator = constant_projective(bias, input_degree=0)
        for source in range(positions):
            if float(fixed_mask[query, source].item()) == 0.0:
                continue
            for head in range(module.n_heads):
                def select(name: str, index: int) -> ProjectivePair:
                    pair = branches[name]
                    return ProjectivePair(
                        pair.numerator[:, head, index],
                        pair.denominator[:, head, index],
                        pair.numerator_degree_upper_bound,
                        pair.denominator_degree_upper_bound,
                        pair.denominator_ledger,
                    )

                score1 = _projective_dot(
                    select("q1", query),
                    select("k1", source),
                    "attention.score1",
                    minimum_absolute_denominator=minimum_absolute_denominator,
                )
                score2 = _projective_dot(
                    select("q2", query),
                    select("k2", source),
                    "attention.score2",
                    minimum_absolute_denominator=minimum_absolute_denominator,
                )
                score = projective_product(
                    score1,
                    score2,
                    label="attention.score_product",
                    minimum_absolute_denominator=minimum_absolute_denominator,
                )
                head_slice = slice(head * module.head_dim, (head + 1) * module.head_dim)
                projected_value = torch.nn.functional.linear(
                    value[:, head, source], module.wo.weight[:, head_slice], None
                )
                scale = row_scales[query] / module._score_denom
                contribution = ProjectivePair(
                    score.numerator * projected_value * scale,
                    score.denominator,
                    score.numerator_degree_upper_bound + 1,
                    score.denominator_degree_upper_bound,
                    score.denominator_ledger,
                )
                accumulator = projective_add(
                    accumulator,
                    contribution,
                    label=f"attention.query{query}.route_add",
                    minimum_absolute_denominator=minimum_absolute_denominator,
                )
        results.append(accumulator)
    return tuple(results)


def stack_projective_rows(
    rows: Sequence[ProjectivePair],
    *,
    minimum_absolute_denominator: float = 1e-12,
) -> Tensor:
    """Evaluate and stack query-row pairs from the tiny attention oracle."""

    if not rows:
        raise ValueError("rows must contain at least one projective pair")
    values = [
        projective_value(row, minimum_absolute_denominator=minimum_absolute_denominator)
        for row in rows
    ]
    reference = values[0].shape
    if any(value.shape != reference for value in values[1:]):
        raise ValueError("all projective rows must have the same shape")
    return torch.stack(values, dim=1)


__all__ = [
    "PQBD_CLAIM_BOUNDARY",
    "PQBD_OBJECT_KIND",
    "DenominatorLedgerEntry",
    "ProjectivePair",
    "ProjectiveOutputPullback",
    "ProjectiveQuotientBalancedDecomposition",
    "QuotientPullback",
    "affine_projective",
    "constant_projective",
    "denominator_ledger_entry",
    "projective_add",
    "projective_output_pullback",
    "projective_product",
    "projective_quotient_balanced_decomposition",
    "projective_rescale",
    "projective_value",
    "quotient_jacobian",
    "quotient_metric_pullback",
    "rational_ffn_residual_projective",
    "rational_norm_projective",
    "stack_projective_rows",
    "tiny_rational_attention_projective_oracle",
]
