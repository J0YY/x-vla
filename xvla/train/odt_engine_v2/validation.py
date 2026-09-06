"""Local reconstruction and scale-ledger checks. No alternate factorizations."""

from __future__ import annotations

import math

import torch

from xvla.train.odt_engine_v2.constants import (
    DIRECT_Q_PROVENANCE_TOLERANCE,
    STREAMED_DIRECT_Q_PROVENANCE_METHOD,
)

from xvla.train.odt_engine_v2.ops import (
    _core_column_blocks,
    _ldexp,
)

from xvla.train.odt_engine_v2.types import (
    CPBinaryCore,
    DirectQProvenance,
    ImplicitCore,
    MaterializationTelemetry,
    ReducedQRBinaryCore,
    ScaledMatrix,
    Tensor,
    UnaryCore,
    _validate_real_finite,
)


def _normalized_scaled_matrix_chart(value: ScaledMatrix) -> tuple[Tensor, int, bool]:
    """Return a power-of-two chart and an arbitrary-precision integer exponent."""

    maximum = value.mantissa.abs().max()
    if float(maximum.item()) == 0.0:
        return value.mantissa.clone(), 0, True
    _, local = torch.frexp(maximum)
    local_exponent = int(local.item())
    return (
        _ldexp(value.mantissa, -local_exponent),
        int(value.binary_exponent) + local_exponent,
        False,
    )


def _scaled_matrix_audit(
    actual: ScaledMatrix,
    expected: ScaledMatrix,
) -> tuple[float, int]:
    """Compare scaled matrices without narrowing their exponents to int64.

    The returned exponent delta is computed with Python integers.  A wildly
    different exponent therefore fails deterministically instead of wrapping
    inside a tensor exponent.  The mantissa comparison is only evaluated after
    both operands have been put in a bounded power-of-two chart.
    """

    if actual.mantissa.shape != expected.mantissa.shape:
        raise ValueError("scaled matrices have different shapes")
    left, left_exponent, left_zero = _normalized_scaled_matrix_chart(actual)
    right, right_exponent, right_zero = _normalized_scaled_matrix_chart(expected)
    if left_zero and right_zero:
        return 0.0, 0
    if left_zero:
        return 1.0, 0
    if right_zero:
        return math.inf, 0
    exponent_delta = int(left_exponent - right_exponent)
    if abs(exponent_delta) > 4096:
        return math.inf, exponent_delta
    center = max(left_exponent, right_exponent)
    left = _ldexp(left, left_exponent - center)
    right = _ldexp(right, right_exponent - center)
    scale = right.norm().clamp_min(torch.finfo(right.dtype).tiny)
    return float(((left - right).norm() / scale).item()), exponent_delta


def _scaled_matrix_relative_error(actual: ScaledMatrix, expected: ScaledMatrix) -> float:
    return _scaled_matrix_audit(actual, expected)[0]


def _local_direct_rq_scaled_reconstruction_audit(
    source_core: ImplicitCore,
    factor: ScaledMatrix,
    canonical_core: ImplicitCore,
    *,
    block_size: int,
    telemetry: MaterializationTelemetry,
    factor_normalization_binary_exponent: int,
) -> tuple[float, int]:
    """Certify ``source = factor @ canonical`` including binary scales.

    This is a local streamed reconstruction, not a quotient replay.  Every
    exponent sum is a Python integer, so corrupt scale bookkeeping cannot wrap
    even when the global rational degree is enormous.
    """

    if source_core.input_dimensions != canonical_core.input_dimensions:
        raise ValueError("direct-RQ reconstruction changed local input dimensions")
    if source_core.output_dimension != factor.mantissa.shape[0]:
        raise ValueError("direct-RQ factor/source output dimensions disagree")
    if canonical_core.output_dimension != factor.mantissa.shape[1]:
        raise ValueError("direct-RQ factor/canonical output dimensions disagree")

    source_exponent = int(source_core.binary_exponent)
    reconstructed_exponent = int(
        factor.binary_exponent + canonical_core.binary_exponent
    )
    raw_delta = reconstructed_exponent - source_exponent
    residual_squared = source_core.matrix.new_zeros(()) if isinstance(
        source_core, UnaryCore
    ) else factor.mantissa.new_zeros(())
    source_squared = residual_squared.clone()
    source_iterator = iter(_core_column_blocks(source_core, block_size, telemetry))
    canonical_iterator = iter(
        _core_column_blocks(canonical_core, block_size, telemetry)
    )
    block_count = 0
    source_nonzero = False
    reconstructed_nonzero = False
    reconstruction_finite = True
    while True:
        try:
            source = next(source_iterator)
        except StopIteration:
            source = None
        try:
            canonical = next(canonical_iterator)
        except StopIteration:
            canonical = None
        if source is None or canonical is None:
            if source is not None or canonical is not None:
                raise RuntimeError("direct-RQ reconstruction block counts disagree")
            break
        if source.shape[1] != canonical.shape[1]:
            raise RuntimeError("direct-RQ reconstruction block widths disagree")
        reconstructed = factor.mantissa @ canonical
        telemetry.observe_temporary(reconstructed)
        source_nonzero = source_nonzero or bool((source != 0).any())
        reconstructed_nonzero = reconstructed_nonzero or bool(
            (reconstructed != 0).any()
        )
        reconstruction_finite = reconstruction_finite and bool(
            torch.isfinite(reconstructed).all()
        )
        if abs(raw_delta) <= 4096:
            center = max(source_exponent, reconstructed_exponent)
            source_scaled = _ldexp(source, source_exponent - center)
            reconstructed_scaled = _ldexp(
                reconstructed, reconstructed_exponent - center
            )
            residual_squared += (reconstructed_scaled - source_scaled).square().sum()
            source_squared += source_scaled.square().sum()
        block_count += 1
    if block_count == 0:
        raise ValueError("direct-RQ reconstruction emitted no source blocks")
    exponent_delta = int(
        factor.binary_exponent
        - source_core.binary_exponent
        - factor_normalization_binary_exponent
        + canonical_core.binary_exponent
    )
    if not source_nonzero:
        if reconstructed_nonzero or not reconstruction_finite:
            raise ValueError(
                "zero direct-RQ source did not reconstruct to exact finite zero"
            )
        if exponent_delta != 0:
            raise ValueError(
                "zero direct-RQ reconstruction has an inconsistent exponent ledger"
            )
        return 0.0, 0
    if abs(raw_delta) > 4096:
        return math.inf, exponent_delta
    error = float(
        torch.sqrt(
            residual_squared
            / source_squared.clamp_min(torch.finfo(source_squared.dtype).tiny)
        ).item()
    )
    return error, exponent_delta


def _input_occurrence_matrix(core: ImplicitCore, role: int) -> Tensor:
    """Flatten one input axis while retaining every other local index."""

    if isinstance(core, UnaryCore):
        if role != 0:
            raise ValueError("unary input occurrence role must be zero")
        return core.matrix
    if isinstance(core, CPBinaryCore):
        if role == 0:
            return core.left_factor
        if role == 1:
            return core.right_factor
        raise ValueError("CP input occurrence role must be zero or one")
    if isinstance(core, ReducedQRBinaryCore):
        tensor = core.q_rows.reshape(
            core.output_dimension, core.left_dimension, core.right_dimension
        )
    else:
        tensor = core.tensor
    axis = role + 1
    if not 1 <= axis < tensor.ndim:
        raise ValueError("dense input occurrence role is out of range")
    return tensor.movedim(axis, -1).reshape(-1, tensor.shape[axis])


def _absorption_scale_audit(
    before: ImplicitCore,
    after: ImplicitCore,
    role: int,
    factor: ScaledMatrix,
) -> tuple[float, int]:
    before_matrix = _input_occurrence_matrix(before, role)
    after_matrix = _input_occurrence_matrix(after, role)
    expected_matrix = before_matrix @ factor.mantissa
    return _scaled_matrix_audit(
        ScaledMatrix(after_matrix, int(after.binary_exponent)),
        ScaledMatrix(
            expected_matrix,
            int(before.binary_exponent + factor.binary_exponent),
        ),
    )


def _projective_batch_relative_error(actual: Tensor, expected: Tensor) -> float:
    """Gauge-invariant error between batches of projective vector coordinates.

    For coordinates ``(N, D)``, equality of the represented quotient is
    ``N_a D_e = N_e D_a``.  The normalization below uses terms with the same
    bidegree, so it is invariant to an arbitrary nonzero common scale on either
    input, including a different scale for every batch element.  This is the
    relevant replay invariant for the rational network.  Comparing the radial
    magnitude of homogeneous coordinates is not.
    """

    if actual.shape != expected.shape or actual.ndim != 2 or actual.shape[1] < 2:
        raise ValueError(
            "projective batches must be equally shaped matrices with width at least two"
        )
    _validate_real_finite(actual, "actual projective batch")
    _validate_real_finite(expected, "expected projective batch")
    actual_denominator = actual[:, -1:]
    expected_denominator = expected[:, -1:]
    epsilon = 100.0 * torch.finfo(actual.dtype).eps
    actual_relative_denominator = actual_denominator.abs() / actual.abs().amax(
        dim=1, keepdim=True
    ).clamp_min(torch.finfo(actual.dtype).tiny)
    expected_relative_denominator = expected_denominator.abs() / expected.abs().amax(
        dim=1, keepdim=True
    ).clamp_min(torch.finfo(expected.dtype).tiny)
    if bool(
        (actual_relative_denominator <= epsilon).any()
        or (expected_relative_denominator <= epsilon).any()
    ):
        raise ValueError("projective replay encountered a numerically zero denominator")
    actual_term = actual[:, :-1] * expected_denominator
    expected_term = expected[:, :-1] * actual_denominator
    residual = (actual_term - expected_term).norm(dim=1)
    scale = (
        actual_term.norm(dim=1)
        + expected_term.norm(dim=1)
        + (actual_denominator[:, 0] * expected_denominator[:, 0]).abs()
    ).clamp_min(torch.finfo(actual.dtype).tiny)
    return float((residual / scale).max().item())


def _validate_direct_q_provenance(
    provenance: DirectQProvenance,
    *,
    require_verified: bool,
) -> None:
    if not isinstance(provenance.method, str) or not provenance.method:
        raise ValueError("direct-Q provenance method must be nonempty")
    integer_fields = (
        provenance.stage_count,
        provenance.column_blocks_compared,
        provenance.columns_compared,
        provenance.retained_transition_elements,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in integer_fields
    ):
        raise ValueError("direct-Q provenance counts must be nonnegative integers")
    if provenance.stage_count < 1 or provenance.column_blocks_compared < 1:
        raise ValueError("direct-Q provenance must contain at least one stage and block")
    if provenance.columns_compared < 1:
        raise ValueError("direct-Q provenance must cover at least one column")
    real_fields = (
        provenance.compact_q_relative_error,
        provenance.direct_q_reconstruction_relative_error,
        provenance.minimum_pivot_to_maximum_entry,
        provenance.conditioning_threshold,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in real_fields):
        raise ValueError("direct-Q provenance metrics must be finite and nonnegative")
    if require_verified and (
        not provenance.verified
        or not provenance.conditioning_accepted
        or provenance.compact_q_relative_error > DIRECT_Q_PROVENANCE_TOLERANCE
        or provenance.direct_q_reconstruction_relative_error
        > DIRECT_Q_PROVENANCE_TOLERANCE
    ):
        raise ValueError("direct-Q provenance certificate is not valid")
    if provenance.method == STREAMED_DIRECT_Q_PROVENANCE_METHOD:
        if provenance.stage_count != provenance.column_blocks_compared:
            raise ValueError(
                "streamed direct-Q provenance stage/block ledger is incomplete"
            )
        if (
            provenance.minimum_pivot_to_maximum_entry
            <= provenance.conditioning_threshold
            or not provenance.conditioning_accepted
        ):
            raise ValueError(
                "streamed direct-Q provenance conditioning gate is invalid"
            )
        is_multiblock = provenance.stage_count > 1
        if (provenance.retained_transition_elements > 0) != is_multiblock:
            raise ValueError(
                "streamed direct-Q transition ledger does not match its stage count"
            )
