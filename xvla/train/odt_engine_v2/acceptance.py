"""Fail-closed numerical acceptance, separate from the mathematical sweeps.

Passing these checks is bounded implementation evidence for the supplied graph
and replay inputs, not independent-clone validation or a capability certificate.
"""

from dataclasses import asdict, dataclass
import math

import torch

from xvla.train.odt_engine_v2.core import (
    canonicalize_implicit_dag_direct_rq, diagonalize_implicit_dag_full_rank,
)
from xvla.train.odt_engine_v2.diagnostics import predict_direct_rq_route_inventory
from xvla.train.odt_engine_v2.graph import _validate_symmetric_lift, evaluate_projective_boundary


@dataclass(frozen=True)
class AcceptanceTolerances:
    replay: float = 3e-8
    offdiagonal: float = 3e-8
    decoded_action: float = 3e-8
    denominator_margin: float = 1e-12

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} tolerance must be finite and in [0, 1)")


def _require_error(name, value, limit):
    if not math.isfinite(value) or value < 0 or value > limit:
        raise RuntimeError(f"ODT acceptance failed: {name}={value!r}, limit={limit!r}")


def _decode_with_margin(network, inputs, minimum):
    """Decode [numerators, denominator] and bound the scale-free chart margin."""
    value = evaluate_projective_boundary(network, inputs)
    if value.ndim != 2 or not value.shape[0] or value.shape[1] < 2 or not bool(torch.isfinite(value).all()):
        raise ValueError("acceptance requires finite nonempty projective output rows")
    size = value.abs().amax(dim=1)
    if bool((size == 0).any()):
        raise RuntimeError("ODT acceptance failed: zero projective row")
    margin = float((value[:, -1].abs() / size).min())
    if margin <= minimum:
        raise RuntimeError(f"ODT acceptance failed: denominator margin {margin!r} <= {minimum!r}")
    decoded = value[:, :-1] / value[:, -1:]
    if not bool(torch.isfinite(decoded).all()):
        raise RuntimeError("ODT acceptance failed: nonfinite decoded action")
    return decoded, margin


@torch.no_grad()
def accept_implicit_odt(network, replay_inputs, *, tolerances=AcceptanceTolerances(), block_size=4096):
    """Preflight ALL propagated shapes, then accept QR and full gauges or raise.

    The input graph is never mutated. Low-level reporting APIs stay available,
    but only this return value attests to the explicit acceptance thresholds.
    Fresh environments are contracted internally; no external records accepted.
    """
    if replay_inputs is None:
        raise ValueError("acceptance requires explicit replay inputs")
    if not isinstance(tolerances, AcceptanceTolerances):
        raise TypeError("explicit AcceptanceTolerances required")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("positive integer block_size required")
    _validate_symmetric_lift(network)
    routes = predict_direct_rq_route_inventory(network)
    reference, before_margin = _decode_with_margin(network, replay_inputs, tolerances.denominator_margin)

    def check_step(work, node, factor, record):
        if not record.per_step_function_replay_performed:
            raise RuntimeError("ODT acceptance failed: missing per-step replay")
        _require_error(f"QR step {record.step} replay", record.per_step_function_replay_error, tolerances.replay)

    canonical = canonicalize_implicit_dag_direct_rq(
        network, replay_inputs=replay_inputs, block_size=block_size,
        replay_each_step=True, copy_network=True, step_callback=check_step,
    )
    _require_error("final QR replay", canonical.final_projective_replay_error, tolerances.replay)
    diagonal = diagonalize_implicit_dag_full_rank(canonical.network, replay_inputs=replay_inputs)
    _require_error("gauge replay", diagonal.replay_relative_error, tolerances.replay)
    _require_error("aggregate offdiagonal ratio", diagonal.maximum_recontracted_offdiagonal_ratio, tolerances.offdiagonal)
    decoded, after_margin = _decode_with_margin(diagonal.network, replay_inputs, tolerances.denominator_margin)
    scale = max(1.0, float(reference.abs().max()))
    action_error = float((decoded / scale - reference / scale).abs().max())
    _require_error("scaled maximum decoded action error", action_error, tolerances.decoded_action)
    return diagonal, {
        "scope": "supplied graph and replay inputs only, not independent-clone or capability validation",
        "tolerances": asdict(tolerances), "routes": routes,
        "qr_replay": canonical.final_projective_replay_error,
        "gauge_replay": diagonal.replay_relative_error,
        "aggregate_offdiagonal_ratio": diagonal.maximum_recontracted_offdiagonal_ratio,
        "scaled_maximum_decoded_action_error": action_error,
        "minimum_relative_denominator_before": before_margin,
        "minimum_relative_denominator_after": after_margin,
    }
