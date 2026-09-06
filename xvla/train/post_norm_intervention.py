"""Low-rank causal edits at the post-normalization linear VLA endpoint.

The operators in this module act only on the final action-query states.  They
therefore test data-conditioned sensitivity of a deployed linear endpoint.
They are not a weight-only or whole-policy ODT decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from xvla.models.vla import ChiVLA
from xvla.train.vla_terminal_balanced_odt import LinearActionEndpoint


Tensor = torch.Tensor


def _finite_float(value: Tensor, name: str) -> Tensor:
    result = torch.as_tensor(value)
    if not result.is_floating_point() or result.is_complex():
        raise TypeError(f"{name} must be a real floating-point tensor")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _positive_finite_scalar(value: float, name: str, *, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not torch.isfinite(torch.tensor(result)) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if upper is not None and result > upper:
        raise ValueError(f"{name} must be at most {upper}")
    return result


def _unit_interval_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not torch.isfinite(torch.tensor(result)) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return result


@dataclass(frozen=True)
class HorizonLowRankProjector:
    """A fixed rank-r projector ``P_h = X_h Y_h`` at every horizon.

    ``state_factors`` has shape ``(H, D, r)`` and ``analysis_factors`` has
    shape ``(H, r, D)``.  Projectors may be oblique.  The dual condition
    ``Y_h X_h = I`` is checked explicitly, which makes ``P_h`` idempotent.
    """

    state_factors: Tensor
    analysis_factors: Tensor
    label: str

    def __post_init__(self) -> None:
        state = _finite_float(self.state_factors, "state_factors")
        analysis = _finite_float(self.analysis_factors, "analysis_factors")
        if state.ndim != 3 or analysis.ndim != 3:
            raise ValueError("projector factors must both be rank-three tensors")
        if state.shape[0] == 0 or state.shape[1] == 0 or state.shape[2] == 0:
            raise ValueError("projector factors must have nonzero dimensions")
        expected = (state.shape[0], state.shape[2], state.shape[1])
        if analysis.shape != expected:
            raise ValueError(f"analysis_factors must have shape {expected}")
        if state.device != analysis.device:
            raise ValueError("projector factors must share a device")
        if state.dtype != analysis.dtype:
            raise ValueError("projector factors must share a dtype")
        if state.dtype not in (torch.float32, torch.float64):
            raise ValueError("projector factors must use float32 or float64")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a nonempty string")
        identity = torch.eye(state.shape[2], dtype=state.dtype, device=state.device)
        error = (torch.einsum("hrd,hds->hrs", analysis, state) - identity).abs().max()
        tolerance = 5e-5 if state.dtype == torch.float32 else 1e-10
        if float(error.item()) > tolerance:
            raise ValueError("projector factors do not satisfy the YX identity")
        dense = torch.einsum("hdr,hrk->hdk", state, analysis)
        idempotence = (dense @ dense - dense).abs().max()
        if float(idempotence.item()) > tolerance:
            raise ValueError("post-cast projector idempotence exceeds the dtype threshold")

    @property
    def horizon(self) -> int:
        return self.state_factors.shape[0]

    @property
    def hidden_dim(self) -> int:
        return self.state_factors.shape[1]

    @property
    def rank(self) -> int:
        return self.state_factors.shape[2]

    def dense(self) -> Tensor:
        return torch.einsum("hdr,hrk->hdk", self.state_factors, self.analysis_factors)

    def to(self, *, dtype: torch.dtype, device: torch.device | str | None = None) -> "HorizonLowRankProjector":
        return HorizonLowRankProjector(
            self.state_factors.to(dtype=dtype, device=device),
            self.analysis_factors.to(dtype=dtype, device=device),
            self.label,
        )


@dataclass(frozen=True)
class AttenuationDoseMatch:
    primary: str
    alpha_grid: tuple[float, ...]
    target_rms: float
    unscaled_rms: dict[str, float]
    alphas: dict[str, float]
    realized_rms: dict[str, float]
    action_indices: tuple[int, ...]


def validate_action_indices(action_dim: int, indices: Sequence[int]) -> tuple[int, ...]:
    result = tuple(indices)
    if not result or len(result) != len(set(result)):
        raise ValueError("action_indices must be nonempty and unique")
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= action_dim for index in result):
        raise ValueError("action_indices contain an out-of-range coordinate")
    return result


@torch.no_grad()
def attenuate_queries(
    queries: Tensor,
    means: Tensor,
    projector: HorizonLowRankProjector,
    *,
    alpha: float,
) -> Tensor:
    """Attenuate the selected centered subspace as ``q' = q - alpha P(q-mu)``."""

    values = _finite_float(queries, "queries")
    center = _finite_float(means, "means")
    if values.dtype not in (torch.float32, torch.float64):
        raise ValueError("queries must use float32 or float64")
    strength = _unit_interval_scalar(alpha, "alpha")
    if values.ndim != 3:
        raise ValueError("queries must have shape (batch, horizon, hidden)")
    if center.shape != values.shape[1:]:
        raise ValueError("means must have shape (horizon, hidden)")
    if (projector.horizon, projector.hidden_dim) != tuple(values.shape[1:]):
        raise ValueError("projector dimensions do not match queries")
    cast_projector = projector.to(dtype=values.dtype, device=values.device)
    state = cast_projector.state_factors
    analysis = cast_projector.analysis_factors
    centered = values - center.to(values)
    coordinates = torch.einsum("hrd,bhd->bhr", analysis, centered)
    removed = torch.einsum("hdr,bhr->bhd", state, coordinates)
    return values - strength * removed


@torch.no_grad()
def normalized_action_delta(
    endpoint: LinearActionEndpoint,
    queries: Tensor,
    means: Tensor,
    projector: HorizonLowRankProjector,
    *,
    alpha: float,
) -> Tensor:
    """Exact precommit normalized-action delta of the low-rank attenuation."""

    values = _finite_float(queries, "queries")
    edited = attenuate_queries(values, means, projector, alpha=alpha)
    weight = endpoint.weight_normalized.to(values)
    return torch.einsum("ad,bhd->bha", weight, edited - values)


@torch.no_grad()
def normalized_action_dose(
    endpoint: LinearActionEndpoint,
    queries: Tensor,
    means: Tensor,
    projector: HorizonLowRankProjector,
    *,
    alpha: float,
    action_indices: Sequence[int],
) -> float:
    indices = validate_action_indices(endpoint.action_dim, action_indices)
    delta = normalized_action_delta(endpoint, queries, means, projector, alpha=alpha)
    return float(delta[..., list(indices)].double().square().mean().sqrt().item())


@torch.no_grad()
def match_normalized_action_doses(
    endpoint: LinearActionEndpoint,
    queries: Tensor,
    means: Tensor,
    projectors: Mapping[str, HorizonLowRankProjector],
    *,
    primary: str,
    alpha_grid: Sequence[float],
    action_indices: Sequence[int],
) -> AttenuationDoseMatch:
    """Match RMS normalized precommit action deltas on held-out calibration inputs.

    The grid chooses the largest predeclared primary strength that every
    comparison can match without extrapolating beyond full attenuation.
    No task outcome is used.
    """

    if primary not in projectors or len(projectors) < 2:
        raise ValueError("primary must name one of at least two projectors")
    if any(operator.label != name for name, operator in projectors.items()):
        raise ValueError("projector mapping keys must equal projector labels")
    grid = tuple(_positive_finite_scalar(value, "alpha_grid value", upper=1.0) for value in alpha_grid)
    if not grid or any(left <= right for left, right in zip(grid, grid[1:])):
        raise ValueError("alpha_grid must be strictly descending")
    indices = validate_action_indices(endpoint.action_dim, action_indices)
    unscaled = {
        name: normalized_action_dose(
            endpoint, queries, means, operator, alpha=1.0, action_indices=indices
        )
        for name, operator in projectors.items()
    }
    if any(not torch.isfinite(torch.tensor(value)) or value <= 0 for value in unscaled.values()):
        raise ValueError("every unscaled dose must be finite and positive")
    ceiling = min(unscaled.values())
    primary_full = unscaled[primary]
    selected = next((value for value in grid if primary_full * value <= ceiling * (1 + 1e-12)), None)
    if selected is None:
        raise ValueError("no alpha-grid value can be matched by all controls")
    target = primary_full * selected
    alphas = {name: target / value for name, value in unscaled.items()}
    if any(value <= 0 or value > 1 + 1e-12 for value in alphas.values()):
        raise RuntimeError("dose matching produced an invalid attenuation strength")
    realized = {
        name: normalized_action_dose(
            endpoint,
            queries,
            means,
            operator,
            alpha=alphas[name],
            action_indices=indices,
        )
        for name, operator in projectors.items()
    }
    return AttenuationDoseMatch(
        primary=primary,
        alpha_grid=grid,
        target_rms=target,
        unscaled_rms=unscaled,
        alphas=alphas,
        realized_rms=realized,
        action_indices=indices,
    )


class PostNormAttenuationPolicy(nn.Module):
    """Inference-only, non-mutating wrapper for one fixed endpoint edit."""

    def __init__(
        self,
        model: ChiVLA,
        means: Tensor,
        projector: HorizonLowRankProjector,
        *,
        alpha: float,
    ) -> None:
        super().__init__()
        if model.cfg.action_head != "linear" or not isinstance(model.action_head, nn.Linear):
            raise ValueError("post-norm intervention requires a linear action head")
        if model.training:
            raise ValueError("post-norm intervention policy is inference-only and requires eval mode")
        if tuple(means.shape) != (model.cfg.action_horizon, model.cfg.dim):
            raise ValueError("means do not match the model action-query shape")
        if (projector.horizon, projector.hidden_dim) != tuple(means.shape):
            raise ValueError("projector does not match the model action-query shape")
        self.model = model
        self.register_buffer("means", _finite_float(means, "means").detach().clone())
        self.register_buffer("state_factors", projector.state_factors.detach().clone())
        self.register_buffer("analysis_factors", projector.analysis_factors.detach().clone())
        self.label = projector.label
        self.alpha = _unit_interval_scalar(alpha, "alpha")
        self.eval()

    def operator(self) -> HorizonLowRankProjector:
        return HorizonLowRankProjector(self.state_factors, self.analysis_factors, self.label)

    @torch.inference_mode()
    def forward(self, img: Tensor, instr_ids: Tensor, state: Tensor, embodiment_id: Tensor, **kwargs):
        if self.training or self.model.training:
            raise RuntimeError("post-norm intervention policy must remain in eval mode")
        allowed = {"img2", "phase_id"}
        unexpected = set(kwargs) - allowed
        if unexpected:
            raise ValueError(f"unsupported intervention arguments: {sorted(unexpected)}")
        _, queries = self.model._encode(
            img,
            instr_ids,
            state,
            embodiment_id,
            img2=kwargs.get("img2"),
            phase_id=kwargs.get("phase_id"),
        )
        if (
            queries.dtype not in (torch.float32, torch.float64)
            or queries.dtype != self.state_factors.dtype
            or queries.device != self.state_factors.device
        ):
            raise RuntimeError("query and frozen intervention factor dtype/device must match")
        centered = queries - self.means.to(queries)
        coordinates = torch.einsum("hrd,bhd->bhr", self.analysis_factors, centered)
        removed = torch.einsum("hdr,bhr->bhd", self.state_factors, coordinates)
        edited = queries - self.alpha * removed
        return self.model.action_head(edited), None


CLAIM_BOUNDARY = (
    "A successful experiment supports causal, data-conditioned action sensitivity "
    "of fixed post-normalization action-query subspaces at the linear endpoint. "
    "It does not establish semantic concepts, a weight-only decomposition, or global ODT."
)
