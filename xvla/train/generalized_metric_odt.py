"""Metric ODT recursions and independent-clone Monte Carlo estimators.

This module works with an explicitly declared multilinear core

    y_o = F[o, i_1, ..., i_k] prod_a x_a[i_a].

The input occurrences are typed and independent unless a caller explicitly
ties them.  This distinction is load-bearing: replacing the independent
copies by one reused random vector estimates a higher-order tied moment, not
the coefficient-space metric used by the ODT recursion.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import ascii_lowercase
from typing import Sequence

import torch


Tensor = torch.Tensor
_ROLE_LABELS = "".join(label for label in ascii_lowercase if label not in {"n", "o"})


@dataclass(frozen=True)
class BalancedMetricODT:
    """Balanced eigensystem on the supported range of an upstream metric."""

    eigenvalues: Tensor
    whitened_vectors: Tensor
    state_directions: Tensor
    upstream_rank: int
    upstream_tolerance: float


def _validate_core(core: Tensor) -> None:
    if not isinstance(core, Tensor) or core.ndim < 2:
        raise ValueError("core must have one output axis and at least one input axis")
    if not core.is_floating_point() or core.is_complex():
        raise TypeError("core must have a real floating-point dtype")
    if not bool(torch.isfinite(core).all()):
        raise ValueError("core must contain only finite values")
    if core.ndim - 1 > len(_ROLE_LABELS):
        raise ValueError("core arity exceeds the supported einsum label budget")


def _validate_metric(metric: Tensor, dimension: int, *, name: str) -> Tensor:
    if not isinstance(metric, Tensor) or metric.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    if not metric.is_floating_point() or metric.is_complex():
        raise TypeError(f"{name} must have a real floating-point dtype")
    if not bool(torch.isfinite(metric).all()):
        raise ValueError(f"{name} must contain only finite values")
    symmetric = 0.5 * (metric + metric.T)
    scale = max(float(symmetric.abs().max().item()), torch.finfo(metric.dtype).tiny)
    tolerance = min(
        100.0 * torch.finfo(metric.dtype).eps * dimension * scale,
        1e-6 * scale,
    )
    if float((metric - metric.T).abs().max().item()) > tolerance:
        raise ValueError(f"{name} must be symmetric")
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    if float(eigenvalues[0].item()) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    if float(eigenvalues[0].item()) < 0:
        symmetric = (eigenvectors * eigenvalues.clamp_min(0)[None, :]) @ eigenvectors.T
        symmetric = 0.5 * (symmetric + symmetric.T)
    return symmetric


def _validated_inputs(core: Tensor, input_metrics: Sequence[Tensor]) -> tuple[Tensor, ...]:
    _validate_core(core)
    if len(input_metrics) != core.ndim - 1:
        raise ValueError("one input metric is required for every input occurrence")
    metrics = []
    for role, (dimension, metric) in enumerate(zip(core.shape[1:], input_metrics)):
        if metric.dtype != core.dtype or metric.device != core.device:
            raise ValueError("core and all metrics must share dtype and device")
        metrics.append(_validate_metric(metric, dimension, name=f"input_metrics[{role}]"))
    return tuple(metrics)


def _kronecker(matrices: Sequence[Tensor], *, like: Tensor) -> Tensor:
    result = like.new_ones(1, 1)
    for matrix in matrices:
        result = torch.kron(result, matrix)
    return result


@torch.no_grad()
def forward_metric(core: Tensor, input_metrics: Sequence[Tensor]) -> Tensor:
    """Propagate coefficient metrics forward through an arbitrary-arity core."""

    metrics = _validated_inputs(core, input_metrics)
    flat = core.reshape(core.shape[0], -1)
    product_metric = _kronecker(metrics, like=core)
    output = flat @ product_metric @ flat.T
    return 0.5 * (output + output.T)


@torch.no_grad()
def role_environment(
    core: Tensor,
    downstream_metric: Tensor,
    input_metrics: Sequence[Tensor],
    role: int,
) -> Tensor:
    """Pull an output metric back to one typed input occurrence."""

    metrics = _validated_inputs(core, input_metrics)
    if isinstance(role, bool) or not isinstance(role, int) or not 0 <= role < len(metrics):
        raise ValueError("role must index an input occurrence")
    if downstream_metric.dtype != core.dtype or downstream_metric.device != core.device:
        raise ValueError("core and downstream_metric must share dtype and device")
    downstream = _validate_metric(
        downstream_metric, core.shape[0], name="downstream_metric"
    )

    input_axis = role + 1
    other_axes = [axis for axis in range(1, core.ndim) if axis != input_axis]
    arranged = core.permute(0, input_axis, *other_axes)
    rows = arranged.reshape(core.shape[0], core.shape[input_axis], -1)
    other_metrics = [metrics[axis - 1] for axis in other_axes]
    # Canonical ODT calls this with row-isometric sibling prefixes, hence every
    # sibling metric is identity.  Materializing their Kronecker product would
    # allocate an identity with d**(2*(arity-1)) entries (d**8 for arity-five
    # attention).  Contracting the shared flattened context index is exactly
    # the same operation and needs only the core-sized ``rows`` view.
    identity_siblings = all(
        torch.equal(
            metric,
            torch.eye(metric.shape[0], dtype=metric.dtype, device=metric.device),
        )
        for metric in other_metrics
    )
    if identity_siblings:
        result = torch.einsum("oia,op,pja->ij", rows, downstream, rows)
    else:
        context_metric = _kronecker(other_metrics, like=core)
        result = torch.einsum("oia,op,pjb,ab->ij", rows, downstream, rows, context_metric)
    return 0.5 * (result + result.T)


@torch.no_grad()
def shared_environment(
    core: Tensor,
    downstream_metric: Tensor,
    input_metrics: Sequence[Tensor],
    roles: Sequence[int] | None = None,
) -> Tensor:
    """Sum typed role environments for one shared-coordinate objective.

    This is the exact objective for the sum of separate one-occurrence losses.
    It is also the telescoping upper-bound objective for applying one projector
    simultaneously at every selected occurrence after the common upstream
    metric has been whitened so the projector is orthogonal.  The raw-space
    matrix alone does not certify an arbitrary Euclidean projector.  It is not
    claimed to be the exact simultaneous-projection loss.
    """

    metrics = _validated_inputs(core, input_metrics)
    selected = tuple(range(core.ndim - 1)) if roles is None else tuple(roles)
    if not selected:
        raise ValueError("roles must contain at least one input occurrence")
    if len(set(selected)) != len(selected):
        raise ValueError("roles must not contain duplicates")
    if any(
        isinstance(role, bool)
        or not isinstance(role, int)
        or not 0 <= role < core.ndim - 1
        for role in selected
    ):
        raise ValueError("roles must index input occurrences")
    dimensions = [core.shape[role + 1] for role in selected]
    if len(set(dimensions)) != 1:
        raise ValueError("selected shared roles must have the same dimension")
    reference_metric = metrics[selected[0]]
    if any(
        not torch.equal(reference_metric, metrics[role]) for role in selected[1:]
    ):
        raise ValueError("selected shared roles must have the same upstream metric")
    return torch.stack(
        [role_environment(core, downstream_metric, metrics, role) for role in selected]
    ).sum(dim=0)


@torch.no_grad()
def balanced_metric_odt(
    upstream_metric: Tensor,
    downstream_environment: Tensor,
    *,
    support_rtol: float = 1e-12,
    support_rank: int | None = None,
) -> BalancedMetricODT:
    """Diagonalize ``R.T @ O @ R`` where ``C = R @ R.T`` on support(C)."""

    if isinstance(support_rtol, bool) or not isinstance(support_rtol, (int, float)) or not torch.isfinite(
        torch.tensor(float(support_rtol))
    ) or not 0 <= support_rtol < 1:
        raise ValueError("support_rtol must be finite and in [0, 1)")
    if upstream_metric.ndim != 2 or upstream_metric.shape[0] != upstream_metric.shape[1]:
        raise ValueError("upstream_metric must be square")
    dimension = upstream_metric.shape[0]
    upstream = _validate_metric(upstream_metric, dimension, name="upstream_metric")
    if downstream_environment.dtype != upstream.dtype or downstream_environment.device != upstream.device:
        raise ValueError("upstream_metric and downstream_environment must share dtype and device")
    downstream = _validate_metric(
        downstream_environment, dimension, name="downstream_environment"
    )
    if support_rank is not None and (
        isinstance(support_rank, bool)
        or not isinstance(support_rank, int)
        or not 0 <= support_rank <= dimension
    ):
        raise ValueError("support_rank must be an integer in [0, dimension]")

    values, vectors = torch.linalg.eigh(upstream)
    scale = max(float(values[-1].item()), 0.0)
    if scale == 0.0:
        if support_rank not in (None, 0):
            raise ValueError("support_rank includes a nonpositive upstream eigenvalue")
        return BalancedMetricODT(
            eigenvalues=upstream.new_empty(0),
            whitened_vectors=upstream.new_empty(0, 0),
            state_directions=upstream.new_empty(dimension, 0),
            upstream_rank=0,
            upstream_tolerance=0.0,
        )
    tolerance = max(
        float(support_rtol) * scale,
        100.0 * torch.finfo(upstream.dtype).eps * dimension * scale,
    )
    if support_rank is not None:
        keep = torch.zeros(dimension, dtype=torch.bool, device=upstream.device)
        if support_rank:
            keep[-support_rank:] = True
            if float(values[-support_rank].item()) <= 0:
                raise ValueError("support_rank includes a nonpositive upstream eigenvalue")
    else:
        keep = values > tolerance
    if not bool(keep.any()):
        empty = upstream.new_empty(0)
        return BalancedMetricODT(
            eigenvalues=empty,
            whitened_vectors=upstream.new_empty(0, 0),
            state_directions=upstream.new_empty(dimension, 0),
            upstream_rank=0,
            upstream_tolerance=tolerance,
        )

    factor = vectors[:, keep] * values[keep].sqrt()[None, :]
    balanced = factor.T @ downstream @ factor
    balanced = 0.5 * (balanced + balanced.T)
    balanced_values, balanced_vectors = torch.linalg.eigh(balanced)
    order = torch.argsort(balanced_values, descending=True)
    balanced_values = balanced_values[order].clamp_min(0)
    balanced_vectors = balanced_vectors[:, order]
    return BalancedMetricODT(
        eigenvalues=balanced_values,
        whitened_vectors=balanced_vectors,
        state_directions=factor @ balanced_vectors,
        upstream_rank=int(keep.sum().item()),
        upstream_tolerance=tolerance,
    )


def _psd_factor(metric: Tensor) -> Tensor:
    values, vectors = torch.linalg.eigh(metric)
    keep = values > 0
    return vectors[:, keep] * values[keep].clamp_min(0).sqrt()[None, :]


def _normal_samples(
    factor: Tensor,
    samples: int,
    generator: torch.Generator,
) -> Tensor:
    if factor.shape[1] == 0:
        return factor.new_zeros(samples, factor.shape[0])
    standard = torch.randn(
        samples,
        factor.shape[1],
        generator=generator,
        dtype=factor.dtype,
        device=factor.device,
    )
    return standard @ factor.T


def _validated_batch_size(samples: int, batch_size: int | None) -> int:
    if batch_size is None:
        return samples
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer or None")
    return min(samples, batch_size)


def _validate_generator_device(generator: torch.Generator, device: torch.device) -> None:
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    generator_device = torch.device(generator.device)
    if generator_device != device:
        raise ValueError(
            f"generator device {generator_device} is incompatible with tensor device {device}"
        )


def _multilinear_batch(core: Tensor, inputs: Sequence[Tensor]) -> Tensor:
    role_labels = _ROLE_LABELS[: len(inputs)]
    equation = "o" + role_labels
    operands: list[Tensor] = [core]
    for label, values in zip(role_labels, inputs):
        equation += f",n{label}"
        operands.append(values)
    equation += "->no"
    return torch.einsum(equation, *operands)


def _role_pullback_batch(
    core: Tensor,
    output_probes: Tensor,
    contexts: Sequence[Tensor],
    role: int,
) -> Tensor:
    role_labels = _ROLE_LABELS[: len(contexts)]
    equation = "o" + role_labels + ",no"
    operands: list[Tensor] = [core, output_probes]
    for index, (label, values) in enumerate(zip(role_labels, contexts)):
        if index != role:
            equation += f",n{label}"
            operands.append(values)
    equation += f"->n{role_labels[role]}"
    return torch.einsum(equation, *operands)


@torch.no_grad()
def estimate_forward_metric(
    core: Tensor,
    input_metrics: Sequence[Tensor],
    *,
    samples: int,
    generator: torch.Generator,
    independent_clones: bool = True,
    batch_size: int | None = None,
) -> Tensor:
    """Estimate a forward metric with zero-mean Gaussian coefficient probes.

    ``independent_clones=False`` is exposed only as an experimental negative
    control and requires identical input dimensions and metrics.
    """

    metrics = _validated_inputs(core, input_metrics)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    _validate_generator_device(generator, core.device)
    chunk_size = _validated_batch_size(samples, batch_size)
    factors = [_psd_factor(metric) for metric in metrics]
    if not independent_clones:
        if len(set(core.shape[1:])) != 1:
            raise ValueError("reused-clone control requires equal input dimensions")
        if any(not torch.equal(metrics[0], metric) for metric in metrics[1:]):
            raise ValueError("reused-clone control requires identical input metrics")
    estimate = core.new_zeros(core.shape[0], core.shape[0])
    completed = 0
    while completed < samples:
        current = min(chunk_size, samples - completed)
        if independent_clones:
            inputs = [_normal_samples(factor, current, generator) for factor in factors]
        else:
            shared = _normal_samples(factors[0], current, generator)
            inputs = [shared] * len(metrics)
        outputs = _multilinear_batch(core, inputs)
        estimate += outputs.T @ outputs
        completed += current
    estimate /= samples
    return 0.5 * (estimate + estimate.T)


@torch.no_grad()
def estimate_shared_environment(
    core: Tensor,
    downstream_metric: Tensor,
    input_metrics: Sequence[Tensor],
    *,
    samples: int,
    generator: torch.Generator,
    roles: Sequence[int] | None = None,
    independent_contexts: bool = True,
    batch_size: int | None = None,
) -> Tensor:
    """Estimate the sum of role environments using independent contexts."""

    metrics = _validated_inputs(core, input_metrics)
    downstream = _validate_metric(
        downstream_metric, core.shape[0], name="downstream_metric"
    )
    if downstream.dtype != core.dtype or downstream.device != core.device:
        raise ValueError("core and downstream_metric must share dtype and device")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    _validate_generator_device(generator, core.device)
    chunk_size = _validated_batch_size(samples, batch_size)
    selected = tuple(range(len(metrics))) if roles is None else tuple(roles)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("roles must be nonempty and unique")
    if any(isinstance(role, bool) or not isinstance(role, int) or not 0 <= role < len(metrics) for role in selected):
        raise ValueError("roles must index input occurrences")
    dimensions = [core.shape[role + 1] for role in selected]
    if len(set(dimensions)) != 1:
        raise ValueError("selected shared roles must have the same dimension")
    reference_metric = metrics[selected[0]]
    if any(not torch.equal(reference_metric, metrics[role]) for role in selected[1:]):
        raise ValueError("selected shared roles must have the same upstream metric")

    metric_factors = [_psd_factor(metric) for metric in metrics]
    output_factor = _psd_factor(downstream)
    estimate = core.new_zeros(dimensions[0], dimensions[0])
    if not independent_contexts:
        if len(set(core.shape[1:])) != 1:
            raise ValueError("reused-context control requires equal input dimensions")
        if any(not torch.equal(metrics[0], metric) for metric in metrics[1:]):
            raise ValueError("reused-context control requires identical input metrics")
    completed = 0
    while completed < samples:
        current = min(chunk_size, samples - completed)
        shared_contexts: list[Tensor] | None = None
        if not independent_contexts:
            shared = _normal_samples(metric_factors[0], current, generator)
            shared_contexts = [shared] * len(metrics)
        for role in selected:
            # A fresh output probe per role keeps the role estimates
            # independent. Reuse would remain unbiased for the sum, but would
            # introduce cross-role covariance that downstream uncertainty
            # estimates would have to retain explicitly.
            output_probes = _normal_samples(output_factor, current, generator)
            contexts = (
                [_normal_samples(factor, current, generator) for factor in metric_factors]
                if independent_contexts
                else shared_contexts
            )
            assert contexts is not None
            pullback = _role_pullback_batch(core, output_probes, contexts, role)
            estimate += pullback.T @ pullback
        completed += current
    estimate /= samples
    return 0.5 * (estimate + estimate.T)
