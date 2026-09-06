"""Balanced ODT at the exact post-normalization linear VLA endpoint.

This module deliberately begins at ``aq_out`` after the VLA's final
normalization. For a linear action head that cut is exactly affine. The
reachable metric is data-conditioned and the preceding rational or polynomial
backbone is outside this endpoint certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from xvla.models.vla import ChiVLA


Tensor = torch.Tensor


def _real_finite(tensor: Tensor, name: str) -> Tensor:
    value = torch.as_tensor(tensor)
    if not value.is_floating_point() or value.is_complex():
        raise TypeError(f"{name} must be a real floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validated_psd(tensor: Tensor, name: str, *, relative_tolerance: float) -> Tensor:
    value = _real_finite(tensor, name)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if (
        isinstance(relative_tolerance, bool)
        or not isinstance(relative_tolerance, (int, float))
        or not torch.isfinite(torch.tensor(float(relative_tolerance)))
        or relative_tolerance <= 0
    ):
        raise ValueError("relative_tolerance must be finite and positive")
    scale = max(float(value.abs().max().item()), torch.finfo(value.dtype).tiny)
    roundoff = 10 * torch.finfo(value.dtype).eps * value.shape[0] * scale
    if float((value - value.T).abs().max().item()) > roundoff:
        raise ValueError(f"{name} must be symmetric")
    symmetric = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    spectral_scale = max(float(eigenvalues[-1].item()), 0.0)
    tolerance = max(relative_tolerance * spectral_scale, roundoff)
    if float(eigenvalues[0].item()) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    if float(eigenvalues[0].item()) < 0:
        symmetric = (eigenvectors * eigenvalues.clamp_min(0).unsqueeze(0)) @ eigenvectors.T
    return symmetric


@dataclass(frozen=True)
class LinearActionEndpoint:
    """Exact affine maps from one post-norm query to normalized and control actions."""

    weight_normalized: Tensor
    bias_normalized: Tensor
    weight_control: Tensor
    bias_control: Tensor
    action_mean: Tensor
    action_std: Tensor
    horizon: int

    @property
    def action_dim(self) -> int:
        return self.weight_normalized.shape[0]

    @property
    def hidden_dim(self) -> int:
        return self.weight_normalized.shape[1]

    def normalized(self, states: Tensor) -> Tensor:
        return states @ self.weight_normalized.T + self.bias_normalized

    def to(self, *, dtype: torch.dtype, device: torch.device | str | None = None) -> "LinearActionEndpoint":
        target = {"dtype": dtype, "device": device}
        return LinearActionEndpoint(
            weight_normalized=self.weight_normalized.to(**target),
            bias_normalized=self.bias_normalized.to(**target),
            weight_control=self.weight_control.to(**target),
            bias_control=self.bias_control.to(**target),
            action_mean=self.action_mean.to(**target),
            action_std=self.action_std.to(**target),
            horizon=self.horizon,
        )

    def control_proposals(self, states: Tensor) -> Tensor:
        """Return pre-commit denormalized LIBERO controls, not SI displacement."""

        return states @ self.weight_control.T + self.bias_control

    def controls(self, states: Tensor) -> Tensor:
        """Backward-compatible alias for pre-commit denormalized proposals."""

        return self.control_proposals(states)

    def committed_controls(self, states: Tensor) -> Tensor:
        """Apply LIBERO's out-of-graph {-1,+1} gripper commitment."""

        if self.action_dim != 7:
            raise ValueError("LIBERO committed controls require seven action coordinates")
        result = self.control_proposals(states).clone()
        result[..., 6] = torch.where(
            result[..., 6] > 0,
            result.new_tensor(1.0),
            result.new_tensor(-1.0),
        )
        return result

    def chunk_jacobian(self, *, control_units: bool = True) -> Tensor:
        weight = self.weight_control if control_units else self.weight_normalized
        return torch.kron(
            torch.eye(self.horizon, dtype=weight.dtype, device=weight.device), weight
        )

    @property
    def normalized_gripper_threshold(self) -> Tensor:
        if self.action_dim < 1:
            raise ValueError("endpoint has no action coordinates")
        return -self.action_mean[-1] / self.action_std[-1]


@torch.no_grad()
def compile_linear_action_endpoint(
    model: ChiVLA,
    action_mean: Tensor,
    action_std: Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> LinearActionEndpoint:
    """Compile the exact shared linear action-query endpoint of ``model``."""

    if model.cfg.action_head != "linear" or not isinstance(model.action_head, nn.Linear):
        raise ValueError("terminal endpoint compiler requires a linear action head")
    target_dtype = model.action_head.weight.dtype if dtype is None else dtype
    if target_dtype not in (torch.float32, torch.float64):
        raise ValueError("endpoint dtype must be torch.float32 or torch.float64")
    mean = _real_finite(action_mean, "action_mean").to(
        dtype=target_dtype, device=model.action_head.weight.device
    )
    std = _real_finite(action_std, "action_std").to(
        dtype=target_dtype, device=model.action_head.weight.device
    )
    action_dim = model.cfg.action_dim
    if mean.shape != (action_dim,) or std.shape != (action_dim,):
        raise ValueError(f"action statistics must each have shape ({action_dim},)")
    if not bool((std > 0).all()):
        raise ValueError("action_std must be strictly positive")
    weight = model.action_head.weight.detach().to(dtype=target_dtype).clone()
    bias = model.action_head.bias.detach().to(dtype=target_dtype).clone()
    raw_weight = std[:, None] * weight
    raw_bias = std * bias + mean
    return LinearActionEndpoint(
        weight_normalized=weight,
        bias_normalized=bias,
        weight_control=raw_weight,
        bias_control=raw_bias,
        action_mean=mean.detach().clone(),
        action_std=std.detach().clone(),
        horizon=model.cfg.action_horizon,
    )


def typed_action_metric(
    *,
    horizon: int,
    action_dim: int,
    action_indices: Sequence[int] | None = None,
    horizon_indices: Sequence[int] | None = None,
    exec_h: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return a diagonal chunk metric in time-major ``t * action_dim + a`` order."""

    if horizon <= 0 or action_dim <= 0:
        raise ValueError("horizon and action_dim must be positive")
    if exec_h is not None and not 0 < exec_h <= horizon:
        raise ValueError("exec_h must lie in [1, horizon]")
    actions = tuple(range(action_dim) if action_indices is None else action_indices)
    times = tuple(range(exec_h or horizon) if horizon_indices is None else horizon_indices)
    if not actions or len(set(actions)) != len(actions):
        raise ValueError("action_indices must be nonempty and unique")
    if not times or len(set(times)) != len(times):
        raise ValueError("horizon_indices must be nonempty and unique")
    if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < action_dim for index in actions):
        raise ValueError("action index is out of range")
    maximum_time = exec_h if exec_h is not None else horizon
    if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < maximum_time for index in times):
        raise ValueError("horizon index is out of range or outside exec_h")
    diagonal = torch.zeros(horizon * action_dim, dtype=dtype, device=device)
    for time_index in times:
        for action_index in actions:
            diagonal[time_index * action_dim + action_index] = 1
    return torch.diag(diagonal)


@torch.no_grad()
def centered_reachable_metrics(states: Tensor) -> tuple[Tensor, Tensor]:
    """Return per-horizon means and population covariances of post-norm queries."""

    values = _real_finite(states, "states")
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("states must have shape (samples, horizon, hidden) with samples >= 2")
    means = values.mean(dim=0)
    centered = values - means
    covariances = torch.einsum("nhd,nhe->hde", centered, centered) / values.shape[0]
    return means, 0.5 * (covariances + covariances.transpose(-1, -2))


@torch.no_grad()
def terminal_observability_metric(
    endpoint: LinearActionEndpoint,
    action_metric: Tensor,
    *,
    control_units: bool = True,
) -> Tensor:
    """Pull a chunk action metric back to all post-norm action-query coordinates."""

    jacobian = endpoint.chunk_jacobian(control_units=control_units)
    metric = _real_finite(action_metric, "action_metric")
    outputs = endpoint.horizon * endpoint.action_dim
    if metric.shape != (outputs, outputs):
        raise ValueError(f"action_metric must have shape ({outputs}, {outputs})")
    if metric.device != jacobian.device:
        raise ValueError("action_metric and endpoint must share a device")
    common_dtype = torch.promote_types(metric.dtype, jacobian.dtype)
    metric = metric.to(dtype=common_dtype)
    jacobian = jacobian.to(dtype=common_dtype)
    metric = _validated_psd(metric, "action_metric", relative_tolerance=1e-10)
    pulled = jacobian.T @ metric @ jacobian
    return 0.5 * (pulled + pulled.T)


@torch.no_grad()
def per_horizon_observability_metrics(chunk_metric: Tensor, *, horizon: int, hidden_dim: int) -> Tensor:
    """Extract diagonal horizon blocks for explicitly per-query analyses."""

    metric = _real_finite(chunk_metric, "chunk_metric")
    expected = horizon * hidden_dim
    if metric.shape != (expected, expected):
        raise ValueError(f"chunk_metric must have shape ({expected}, {expected})")
    blocks = metric.reshape(horizon, hidden_dim, horizon, hidden_dim)
    return torch.stack([blocks[index, :, index, :] for index in range(horizon)])


@dataclass(frozen=True)
class BalancedSupportedEigensystem:
    eigenvalues: Tensor
    whitened_directions: Tensor
    state_directions: Tensor
    analysis_directions: Tensor
    projector: Tensor
    support_basis: Tensor
    support_eigenvalues: Tensor
    support_rank: int
    observable_rank: int


@dataclass(frozen=True)
class CenteredReachableFactor:
    """Thin factor ``R`` with ``C = R R^T`` for flattened action queries."""

    mean: Tensor
    root: Tensor
    singular_values: Tensor
    support_rank: int


@dataclass(frozen=True)
class RemovalDoseMatch:
    """Frozen linear-endpoint dose match for rank-matched removals."""

    primary: str
    target_dose: float
    unscaled_doses: dict[str, float]
    alphas: dict[str, float]
    realized_doses: dict[str, float]


@torch.no_grad()
def centered_flattened_reachable_factor(
    states: Tensor,
    *,
    relative_tolerance: float = 1e-10,
) -> CenteredReachableFactor:
    """Build a thin full-chunk C factor without materializing an HD square matrix."""

    values = _real_finite(states, "states")
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("states must have shape (samples, horizon, hidden) with samples >= 2")
    if (
        isinstance(relative_tolerance, bool)
        or not isinstance(relative_tolerance, (int, float))
        or not torch.isfinite(torch.tensor(float(relative_tolerance)))
        or relative_tolerance <= 0
    ):
        raise ValueError("relative_tolerance must be finite and positive")
    mean = values.mean(0)
    centered = (values - mean).reshape(values.shape[0], -1) / values.shape[0] ** 0.5
    _, singular_values, right = torch.linalg.svd(centered, full_matrices=False)
    maximum = float(singular_values[0].item()) if singular_values.numel() else 0.0
    if maximum <= 0:
        raise ValueError("states have empty centered support")
    covariance_maximum = maximum * maximum
    covariance_tolerance = max(
        relative_tolerance * covariance_maximum,
        10 * torch.finfo(values.dtype).eps * max(centered.shape) * covariance_maximum,
    )
    rank = int((singular_values.square() > covariance_tolerance).sum().item())
    if rank == 0:
        raise ValueError("states have empty numerical centered support")
    root = right[:rank].T * singular_values[:rank].unsqueeze(0)
    return CenteredReachableFactor(
        mean=mean,
        root=root,
        singular_values=singular_values,
        support_rank=rank,
    )


@torch.no_grad()
def balanced_supported_from_factor(
    reachable_root: Tensor,
    observability_metric: Tensor,
    *,
    relative_tolerance: float = 1e-10,
) -> BalancedSupportedEigensystem:
    """Balance a supplied full-column C factor against O without dense C."""

    root = _real_finite(reachable_root, "reachable_root")
    observable = _real_finite(observability_metric, "observability_metric")
    if root.ndim != 2 or root.shape[0] == 0 or root.shape[1] == 0 or root.shape[1] > root.shape[0]:
        raise ValueError("reachable_root must have shape (state, support) with 0 < support <= state")
    if observable.shape != (root.shape[0], root.shape[0]):
        raise ValueError("observability_metric dimension must match reachable_root")
    if observable.device != root.device:
        raise ValueError("reachable_root and observability_metric must share a device")
    common_dtype = torch.promote_types(root.dtype, observable.dtype)
    root = root.to(dtype=common_dtype)
    observable = observable.to(dtype=common_dtype)
    root_singular_values = torch.linalg.svdvals(root)
    gram_values = root_singular_values.square().flip(0)
    gram_scale = float(root_singular_values[0].square().item())
    gram_tolerance = max(
        relative_tolerance * gram_scale,
        10 * torch.finfo(root.dtype).eps * root.shape[1] * gram_scale,
    )
    if float(gram_values[0].item()) <= gram_tolerance:
        raise ValueError("reachable_root is not numerically full column rank")
    observable = _validated_psd(
        observable, "observability_metric", relative_tolerance=relative_tolerance
    )
    balanced = root.T @ observable @ root
    balanced = 0.5 * (balanced + balanced.T)
    values, vectors = torch.linalg.eigh(balanced)
    order = torch.argsort(values, descending=True)
    values = values[order].clamp_min(0)
    vectors = vectors[:, order]
    scale = max(float(values[0].item()), 0.0)
    tolerance = max(
        relative_tolerance * scale,
        10 * torch.finfo(root.dtype).eps * root.shape[1] * scale,
    )
    observable_rank = int((values > tolerance).sum().item())
    if observable_rank:
        positive_values = values[:observable_rank]
        state_directions = root @ vectors[:, :observable_rank]
        analysis_directions = (
            positive_values.reciprocal().unsqueeze(1)
            * state_directions.T
            @ observable
        )
        projector = state_directions @ analysis_directions
    else:
        state_directions = root.new_empty(root.shape[0], 0)
        analysis_directions = root.new_empty(0, root.shape[0])
        projector = root.new_zeros(root.shape[0], root.shape[0])
    return BalancedSupportedEigensystem(
        eigenvalues=values,
        whitened_directions=vectors,
        state_directions=state_directions,
        analysis_directions=analysis_directions,
        projector=projector,
        support_basis=root,
        support_eigenvalues=gram_values,
        support_rank=root.shape[1],
        observable_rank=observable_rank,
    )


@torch.no_grad()
def balanced_supported_from_observation_factor(
    reachable_root: Tensor,
    observation_map: Tensor,
    *,
    relative_tolerance: float = 1e-10,
) -> BalancedSupportedEigensystem:
    """Balance C=RR^T and O=A^T A without materializing either square metric."""

    root = _real_finite(reachable_root, "reachable_root")
    observation = _real_finite(observation_map, "observation_map")
    if (
        isinstance(relative_tolerance, bool)
        or not isinstance(relative_tolerance, (int, float))
        or not torch.isfinite(torch.tensor(float(relative_tolerance)))
        or relative_tolerance <= 0
    ):
        raise ValueError("relative_tolerance must be finite and positive")
    if root.ndim != 2 or root.shape[0] == 0 or root.shape[1] == 0 or root.shape[1] > root.shape[0]:
        raise ValueError("reachable_root must have shape (state, support) with 0 < support <= state")
    if observation.ndim != 2 or observation.shape[1] != root.shape[0]:
        raise ValueError("observation_map must have shape (outputs, state)")
    if observation.device != root.device:
        raise ValueError("reachable_root and observation_map must share a device")
    dtype = torch.promote_types(root.dtype, observation.dtype)
    root = root.to(dtype=dtype)
    observation = observation.to(dtype=dtype)
    root_singular_values = torch.linalg.svdvals(root)
    maximum = float(root_singular_values[0].square().item())
    tolerance = max(
        relative_tolerance * maximum,
        10 * torch.finfo(dtype).eps * root.shape[1] * maximum,
    )
    if float(root_singular_values[-1].square().item()) <= tolerance:
        raise ValueError("reachable_root is not numerically full column rank")
    observed_root = observation @ root
    balanced = observed_root.T @ observed_root
    balanced = 0.5 * (balanced + balanced.T)
    values, vectors = torch.linalg.eigh(balanced)
    order = torch.argsort(values, descending=True)
    values = values[order].clamp_min(0)
    vectors = vectors[:, order]
    scale = max(float(values[0].item()), 0.0)
    balance_tolerance = max(
        relative_tolerance * scale,
        10 * torch.finfo(dtype).eps * root.shape[1] * scale,
    )
    observable_rank = int((values > balance_tolerance).sum().item())
    if observable_rank:
        positive = values[:observable_rank]
        state_directions = root @ vectors[:, :observable_rank]
        analysis_directions = (
            positive.reciprocal().unsqueeze(1)
            * (observation @ state_directions).T
            @ observation
        )
        projector = state_directions @ analysis_directions
    else:
        state_directions = root.new_empty(root.shape[0], 0)
        analysis_directions = root.new_empty(0, root.shape[0])
        projector = root.new_zeros(root.shape[0], root.shape[0])
    return BalancedSupportedEigensystem(
        eigenvalues=values,
        whitened_directions=vectors,
        state_directions=state_directions,
        analysis_directions=analysis_directions,
        projector=projector,
        support_basis=root,
        support_eigenvalues=root_singular_values.square().flip(0),
        support_rank=root.shape[1],
        observable_rank=observable_rank,
    )


@torch.no_grad()
def balanced_supported_eigensystem(
    reachable_metric: Tensor,
    observability_metric: Tensor,
    *,
    support_rank: int | None = None,
    relative_tolerance: float = 1e-10,
) -> BalancedSupportedEigensystem:
    """Compute a data-conditioned action-balanced spectrum on full support.

    Canonical Dooms ODT is the weight-derived coefficient-space special case.

    An automatically inferred numerical rank needs a spectral gap to remain
    stable under finite-precision gauges. An explicit ``support_rank`` asserts
    the expected audit decision, but it may not truncate a genuinely positive C mode.
    Such a Euclidean C truncation would not be covariant under nonorthogonal
    state gauges.
    """

    covariance_input = _real_finite(reachable_metric, "reachable_metric")
    observable_input = _real_finite(observability_metric, "observability_metric")
    if observable_input.device != covariance_input.device:
        raise ValueError("reachable_metric and observability_metric must share a device")
    common_dtype = torch.promote_types(covariance_input.dtype, observable_input.dtype)
    covariance_input = covariance_input.to(dtype=common_dtype)
    observable_input = observable_input.to(dtype=common_dtype)
    if observable_input.shape != covariance_input.shape:
        raise ValueError("observability_metric must match reachable_metric")
    covariance = _validated_psd(
        covariance_input, "reachable_metric", relative_tolerance=relative_tolerance
    )
    observable = _validated_psd(
        observable_input, "observability_metric", relative_tolerance=relative_tolerance
    )
    c_values, c_vectors = torch.linalg.eigh(covariance)
    covariance_scale = float(c_values[-1].item())
    if covariance_scale <= 0:
        raise ValueError("reachable_metric has empty numerical support")
    numerical_floor = (
        10 * torch.finfo(covariance.dtype).eps * covariance.shape[0] * covariance_scale
    )
    tolerance = max(relative_tolerance * covariance_scale, numerical_floor)
    inferred_rank = int((c_values > tolerance).sum().item())
    if inferred_rank == 0 and support_rank is None:
        raise ValueError("reachable_metric has empty numerical support")
    rank = inferred_rank if support_rank is None else support_rank
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= covariance.shape[0]:
        raise ValueError("support_rank must lie in [1, dimension]")
    if support_rank is not None and rank != inferred_rank:
        raise ValueError(
            "support_rank must equal the full positive numerical rank; "
            "Euclidean support truncation is not gauge covariant"
        )
    selected_values = c_values[-rank:]
    if float(selected_values[0].item()) <= tolerance:
        raise ValueError("support_rank includes a numerical-null covariance direction")
    support = c_vectors[:, -rank:]
    root = support * selected_values.sqrt().unsqueeze(0)
    balanced = 0.5 * (root.T @ observable @ root + root.T @ observable.T @ root)
    values, vectors = torch.linalg.eigh(balanced)
    order = torch.argsort(values, descending=True)
    values = values[order].clamp_min(0)
    vectors = vectors[:, order]
    balance_scale = max(float(values[0].item()), 0.0)
    balance_tolerance = max(
        relative_tolerance * balance_scale,
        10 * torch.finfo(values.dtype).eps * rank * balance_scale,
    )
    observable_rank = int((values > balance_tolerance).sum().item())
    if observable_rank == 0:
        empty_state = covariance.new_empty(covariance.shape[0], 0)
        empty_analysis = covariance.new_empty(0, covariance.shape[0])
        return BalancedSupportedEigensystem(
            eigenvalues=values,
            whitened_directions=vectors,
            state_directions=empty_state,
            analysis_directions=empty_analysis,
            projector=covariance.new_zeros(covariance.shape),
            support_basis=support,
            support_eigenvalues=selected_values,
            support_rank=rank,
            observable_rank=0,
        )
    positive_values = values[:observable_rank]
    positive_vectors = vectors[:, :observable_rank]
    state_directions = root @ positive_vectors
    # This O-dual, unlike a Euclidean pseudoinverse of the gauged root, obeys
    # Y' = Y S^{-1}. Hence X_k Y_k is covariant under every invertible state
    # gauge. Zero-observability modes have no canonical covariant dual.
    analysis_directions = (
        positive_values.reciprocal().unsqueeze(1)
        * state_directions.T
        @ observable
    )
    projector = state_directions @ analysis_directions
    return BalancedSupportedEigensystem(
        eigenvalues=values,
        whitened_directions=vectors,
        state_directions=state_directions,
        analysis_directions=analysis_directions,
        projector=projector,
        support_basis=support,
        support_eigenvalues=selected_values,
        support_rank=rank,
        observable_rank=observable_rank,
    )


@torch.no_grad()
def balanced_projector(system: BalancedSupportedEigensystem, rank: int) -> Tensor:
    """Return the generally oblique rank-k balanced state projector."""

    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank <= system.observable_rank:
        raise ValueError("rank must lie in [0, observable_rank]")
    return system.state_directions[:, :rank] @ system.analysis_directions[:rank]


@torch.no_grad()
def euclidean_principal_projector(metric: Tensor, rank: int) -> Tensor:
    """Return the orthogonal projector onto the top PSD eigenspace."""

    value = _validated_psd(metric, "metric", relative_tolerance=1e-10)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 < rank <= value.shape[0]:
        raise ValueError("rank must lie in [1, dimension]")
    _, vectors = torch.linalg.eigh(value)
    basis = vectors[:, -rank:]
    return basis @ basis.T


@torch.no_grad()
def cross_covariance_projector(
    reachable_metric: Tensor,
    observation_map: Tensor,
    rank: int,
) -> Tensor:
    """Return the top left-singular projector of ``C A^T``."""

    covariance = _validated_psd(
        reachable_metric, "reachable_metric", relative_tolerance=1e-10
    )
    observation = _real_finite(observation_map, "observation_map")
    if observation.ndim != 2 or observation.shape[1] != covariance.shape[0]:
        raise ValueError("observation_map must have shape (outputs, state)")
    if observation.device != covariance.device:
        raise ValueError("reachable_metric and observation_map must share a device")
    dtype = torch.promote_types(covariance.dtype, observation.dtype)
    matrix = covariance.to(dtype=dtype) @ observation.to(dtype=dtype).T
    maximum_rank = min(matrix.shape)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 < rank <= maximum_rank:
        raise ValueError("rank must lie in [1, cross-covariance rank ceiling]")
    left, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    tolerance = max(
        1e-10 * float(singular_values[0].item()),
        10 * torch.finfo(dtype).eps * max(matrix.shape) * float(singular_values[0].item()),
    )
    if float(singular_values[rank - 1].item()) <= tolerance:
        raise ValueError("requested cross-covariance rank is numerically unsupported")
    basis = left[:, :rank]
    return basis @ basis.T


@torch.no_grad()
def balanced_haar_projector(
    system: BalancedSupportedEigensystem,
    rank: int,
    *,
    seed: int,
) -> Tensor:
    """Draw a fixed Haar subspace in observable balanced modal coordinates."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 < rank <= system.observable_rank:
        raise ValueError("rank must lie in [1, observable_rank]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draw = torch.randn(
        system.observable_rank,
        rank,
        generator=generator,
        dtype=system.state_directions.dtype,
        device="cpu",
    ).to(system.state_directions.device)
    modal, _ = torch.linalg.qr(draw, mode="reduced")
    state = system.state_directions @ modal
    analysis = modal.T @ system.analysis_directions
    return state @ analysis


@torch.no_grad()
def removal_action_delta(
    endpoint: LinearActionEndpoint,
    states: Tensor,
    means: Tensor,
    projectors: Tensor,
    *,
    action_indices: Sequence[int] | None = None,
    control_units: bool = False,
) -> Tensor:
    """Exact action delta caused by fully removing supplied query subspaces."""

    values = _real_finite(states, "states")
    center = _real_finite(means, "means").to(values)
    projector = _real_finite(projectors, "projectors").to(values)
    if values.ndim != 3 or center.shape != values.shape[1:]:
        raise ValueError("states/means must have shapes (batch, horizon, hidden)/(horizon, hidden)")
    if projector.ndim == 2:
        projector = projector.expand(values.shape[1], -1, -1)
    if projector.shape != (values.shape[1], values.shape[2], values.shape[2]):
        raise ValueError("projectors must have shape (hidden, hidden) or (horizon, hidden, hidden)")
    weight = endpoint.weight_control if control_units else endpoint.weight_normalized
    if weight.device != values.device or weight.dtype != values.dtype:
        weight = weight.to(values)
    actions = tuple(range(endpoint.action_dim) if action_indices is None else action_indices)
    if not actions or len(set(actions)) != len(actions) or any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < endpoint.action_dim
        for index in actions
    ):
        raise ValueError("action_indices must be unique valid coordinates")
    removed = torch.einsum("hde,bhe->bhd", projector, values - center)
    return -torch.einsum("ad,bhd->bha", weight[list(actions)], removed)


@torch.no_grad()
def match_removal_doses(
    endpoint: LinearActionEndpoint,
    states: Tensor,
    means: Tensor,
    projectors: dict[str, Tensor],
    *,
    primary: str,
    alpha_grid: Sequence[float],
    action_indices: Sequence[int] | None = None,
    control_units: bool = False,
) -> RemovalDoseMatch:
    """Dose-match removals without fitting to any downstream outcome."""

    if primary not in projectors or len(projectors) < 2:
        raise ValueError("primary must name one of at least two projectors")
    grid = tuple(float(value) for value in alpha_grid)
    if not grid or any(not 0 < value <= 1 or not torch.isfinite(torch.tensor(value)) for value in grid):
        raise ValueError("alpha_grid must contain finite values in (0, 1]")
    if any(left <= right for left, right in zip(grid, grid[1:])):
        raise ValueError("alpha_grid must be strictly descending")
    unscaled = {
        name: float(
            removal_action_delta(
                endpoint,
                states,
                means,
                projector,
                action_indices=action_indices,
                control_units=control_units,
            ).square().mean().sqrt().item()
        )
        for name, projector in projectors.items()
    }
    if any(not torch.isfinite(torch.tensor(value)) or value <= 0 for value in unscaled.values()):
        raise ValueError("every removal must have positive finite unscaled dose")
    primary_dose = unscaled[primary]
    selected = next(
        (alpha for alpha in grid if primary_dose * alpha <= min(unscaled.values()) * (1 + 1e-12)),
        None,
    )
    if selected is None:
        raise ValueError("no prespecified primary alpha can be matched by every control")
    target = primary_dose * selected
    alphas = {name: target / dose for name, dose in unscaled.items()}
    if any(not 0 < alpha <= 1 + 1e-12 for alpha in alphas.values()):
        raise RuntimeError("dose matching produced an invalid alpha")
    realized = {name: unscaled[name] * alphas[name] for name in unscaled}
    return RemovalDoseMatch(
        primary=primary,
        target_dose=target,
        unscaled_doses=unscaled,
        alphas=alphas,
        realized_doses=realized,
    )


@torch.no_grad()
def intervene_post_norm_queries(
    states: Tensor,
    means: Tensor,
    projectors: Tensor,
    *,
    alpha: float = 1.0,
    horizon_indices: Iterable[int] | None = None,
) -> Tensor:
    """Apply centered projector interpolation to selected action queries."""

    values = _real_finite(states, "states")
    center = _real_finite(means, "means").to(values)
    projector = _real_finite(projectors, "projectors").to(values)
    if values.ndim != 3:
        raise ValueError("states must have shape (batch, horizon, hidden)")
    if center.shape != values.shape[1:]:
        raise ValueError("means must have shape (horizon, hidden)")
    if not isinstance(alpha, (int, float)) or not torch.isfinite(torch.tensor(float(alpha))):
        raise ValueError("alpha must be finite")
    flattened_dimension = values.shape[1] * values.shape[2]
    if projector.ndim == 2 and projector.shape == (flattened_dimension, flattened_dimension):
        if horizon_indices is not None:
            raise ValueError("horizon_indices cannot be combined with a cross-horizon projector")
        flattened = (values - center).reshape(values.shape[0], flattened_dimension)
        projected = flattened @ projector.T
        edited = (1.0 - float(alpha)) * flattened + float(alpha) * projected
        return center + edited.reshape_as(values)
    if projector.ndim == 2:
        projector = projector.expand(values.shape[1], -1, -1)
    if projector.shape != (values.shape[1], values.shape[2], values.shape[2]):
        raise ValueError("projectors must have shape (hidden, hidden) or (horizon, hidden, hidden)")
    indices = tuple(range(values.shape[1]) if horizon_indices is None else horizon_indices)
    if len(set(indices)) != len(indices) or any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < values.shape[1]
        for index in indices
    ):
        raise ValueError("horizon_indices must be unique valid integers")
    result = values.clone()
    for index in indices:
        centered = values[:, index] - center[index]
        projected = centered @ projector[index].T
        result[:, index] = center[index] + (1.0 - float(alpha)) * centered + float(alpha) * projected
    return result


class PostNormLinearInterventionPolicy(nn.Module):
    """Inference-only post-norm editor returning normalized action proposals.

    The wrapper is non-mutating. It does not preserve the training loss,
    target-action, or phase-return portions of the full ChiVLA contract.
    """

    def __init__(self, model: ChiVLA, means: Tensor, projectors: Tensor, *, alpha: float = 1.0):
        super().__init__()
        if model.cfg.action_head != "linear" or not isinstance(model.action_head, nn.Linear):
            raise ValueError("intervention wrapper requires a linear action head")
        self.model = model
        self.register_buffer("means", _real_finite(means, "means").detach().clone())
        self.register_buffer("projectors", _real_finite(projectors, "projectors").detach().clone())
        self.alpha = float(alpha)

    def forward(self, img: Tensor, instr_ids: Tensor, state: Tensor, embodiment_id: Tensor, **kwargs):
        allowed = {"img2", "phase_id"}
        unexpected = set(kwargs) - allowed
        if unexpected:
            raise ValueError(f"unsupported intervention forward arguments: {sorted(unexpected)}")
        _, queries = self.model._encode(
            img,
            instr_ids,
            state,
            embodiment_id,
            img2=kwargs.get("img2"),
            phase_id=kwargs.get("phase_id"),
        )
        edited = intervene_post_norm_queries(
            queries, self.means, self.projectors, alpha=self.alpha
        )
        return self.model.action_head(edited), None
