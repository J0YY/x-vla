"""Tensor application, exact factor absorption, scaled arithmetic and downstream contractions."""

from __future__ import annotations

from typing import Iterator, Sequence

import torch

from xvla.train.odt_engine_v2.types import (
    CPBinaryCore,
    DenseCloneCore,
    ImplicitCore,
    MaterializationTelemetry,
    ReducedQRBinaryCore,
    ScaledBatch,
    ScaledMatrix,
    Tensor,
    UnaryCore,
    _validate_real_finite,
)


def _ldexp(value: Tensor, exponent: int) -> Tensor:
    return torch.ldexp(
        value,
        torch.tensor(exponent, dtype=torch.int64, device=value.device),
    )


def _strip_power_of_two(value: Tensor) -> tuple[Tensor, int]:
    _validate_real_finite(value, "scaled tensor")
    maximum = value.abs().max()
    if float(maximum.item()) == 0.0:
        return value.clone(), 0
    _, exponent = torch.frexp(maximum)
    integer = int(exponent.item())
    return _ldexp(value, -integer), integer


def _make_unary_core(
    matrix: Tensor,
    kind: str,
    telemetry: MaterializationTelemetry,
) -> UnaryCore:
    _validate_real_finite(matrix, kind)
    if matrix.ndim != 2:
        raise ValueError("unary primitive must be a matrix")
    mantissa, exponent = _strip_power_of_two(matrix)
    telemetry.observe_persistent(mantissa)
    return UnaryCore(mantissa, kind, exponent)


def _make_cp_core(
    output_factor: Tensor,
    left_factor: Tensor,
    right_factor: Tensor,
    kind: str,
    telemetry: MaterializationTelemetry,
) -> CPBinaryCore:
    for name, value in (
        ("output_factor", output_factor),
        ("left_factor", left_factor),
        ("right_factor", right_factor),
    ):
        _validate_real_finite(value, f"{kind}.{name}")
        if value.ndim != 2:
            raise ValueError("CP factors must be matrices")
    rank = output_factor.shape[1]
    if left_factor.shape[0] != rank or right_factor.shape[0] != rank:
        raise ValueError("CP factor ranks do not agree")
    output, output_exp = _strip_power_of_two(output_factor)
    left, left_exp = _strip_power_of_two(left_factor)
    right, right_exp = _strip_power_of_two(right_factor)
    telemetry.observe_persistent(output, left, right)
    return CPBinaryCore(output, left, right, kind, output_exp + left_exp + right_exp)


def _core_clone(core: ImplicitCore) -> ImplicitCore:
    return core.clone()


def _core_arity(core: ImplicitCore) -> int:
    return len(core.input_dimensions)


def _normalize_absorbed(value: Tensor) -> tuple[Tensor, int]:
    return _strip_power_of_two(value)


def _absorb_input_factor(
    core: ImplicitCore,
    role: int,
    factor: ScaledMatrix,
    *,
    normalize: bool = True,
) -> ImplicitCore:
    """C'[..., a, ...] = sum_i C[..., i, ...] factor[i, a] at slot role.

    The represented tensors include their powers of two: exponents add, plus
    any stripped normalization exponent. No inverse or decomposition occurs.
    """
    if isinstance(core, DenseCloneCore):
        if not 0 <= role < len(core.input_dimensions):
            raise ValueError("dense-clone role is out of range")
        axis = role + 1
        tensor = (core.tensor.movedim(axis, -1) @ factor.mantissa).movedim(-1, axis)
        if normalize:
            tensor, stripped = _normalize_absorbed(tensor)
        else:
            stripped = 0
        return DenseCloneCore(
            tensor,
            core.kind,
            core.binary_exponent + factor.binary_exponent + stripped,
        )
    if isinstance(core, ReducedQRBinaryCore):
        if role not in (0, 1):
            raise ValueError("reduced-QR binary role must be zero or one")
        tensor = core.q_rows.reshape(
            core.output_dimension, core.left_dimension, core.right_dimension
        )
        axis = role + 1
        if tensor.shape[axis] != factor.mantissa.shape[0]:
            raise ValueError("reduced-QR input-factor occurrence mismatch")
        tensor = (tensor.movedim(axis, -1) @ factor.mantissa).movedim(-1, axis)
        if normalize:
            tensor, stripped = _normalize_absorbed(tensor)
        else:
            stripped = 0
        return ReducedQRBinaryCore(
            tensor.reshape(tensor.shape[0], -1),
            int(tensor.shape[1]),
            int(tensor.shape[2]),
            core.kind,
            core.binary_exponent + factor.binary_exponent + stripped,
        )
    if isinstance(core, UnaryCore):
        if role != 0 or core.matrix.shape[1] != factor.mantissa.shape[0]:
            raise ValueError("unary input-factor occurrence mismatch")
        matrix = core.matrix @ factor.mantissa
        if normalize:
            matrix, stripped = _normalize_absorbed(matrix)
        else:
            stripped = 0
        return UnaryCore(
            matrix,
            core.kind,
            core.binary_exponent + factor.binary_exponent + stripped,
        )
    if role not in (0, 1):
        raise ValueError("binary role must be zero or one")
    selected = core.left_factor if role == 0 else core.right_factor
    if selected.shape[1] != factor.mantissa.shape[0]:
        raise ValueError("CP input-factor occurrence mismatch")
    selected = selected @ factor.mantissa
    if normalize:
        selected, stripped = _normalize_absorbed(selected)
    else:
        stripped = 0
    return CPBinaryCore(
        core.output_factor,
        selected if role == 0 else core.left_factor,
        selected if role == 1 else core.right_factor,
        core.kind,
        core.binary_exponent + factor.binary_exponent + stripped,
        core.direct_q_provenance,
    )


def _apply_output_basis(core: ImplicitCore, basis_transpose: Tensor) -> ImplicitCore:
    if basis_transpose.shape[1] != core.output_dimension:
        raise ValueError("output basis dimension mismatch")
    if isinstance(core, DenseCloneCore):
        return DenseCloneCore(
            torch.tensordot(basis_transpose, core.tensor, dims=([1], [0])),
            core.kind,
            core.binary_exponent,
        )
    if isinstance(core, UnaryCore):
        return UnaryCore(
            basis_transpose @ core.matrix,
            core.kind,
            core.binary_exponent,
        )
    if isinstance(core, ReducedQRBinaryCore):
        return ReducedQRBinaryCore(
            basis_transpose @ core.q_rows,
            core.left_dimension,
            core.right_dimension,
            core.kind,
            core.binary_exponent,
        )
    return CPBinaryCore(
        basis_transpose @ core.output_factor,
        core.left_factor,
        core.right_factor,
        core.kind,
        core.binary_exponent,
        core.direct_q_provenance,
    )


def _cp_column_blocks(
    core: CPBinaryCore,
    block_size: int,
    telemetry: MaterializationTelemetry,
) -> Iterator[Tensor]:
    left_dim, right_dim = core.input_dimensions
    total = left_dim * right_dim
    device = core.output_factor.device
    for start in range(0, total, block_size):
        stop = min(total, start + block_size)
        flat = torch.arange(start, stop, device=device)
        left_index = torch.div(flat, right_dim, rounding_mode="floor")
        right_index = flat.remainder(right_dim)
        atoms = core.left_factor[:, left_index] * core.right_factor[:, right_index]
        block = core.output_factor @ atoms
        telemetry.observe_temporary(flat, atoms, block)
        yield block


def _core_column_blocks(
    core: ImplicitCore,
    block_size: int,
    telemetry: MaterializationTelemetry,
) -> Iterator[Tensor]:
    if isinstance(core, DenseCloneCore):
        rows = core.tensor.reshape(core.output_dimension, -1)
        for start in range(0, rows.shape[1], block_size):
            block = rows[:, start : start + block_size]
            telemetry.observe_temporary(block)
            yield block
        return
    if isinstance(core, UnaryCore):
        for start in range(0, core.matrix.shape[1], block_size):
            block = core.matrix[:, start : start + block_size]
            telemetry.observe_temporary(block)
            yield block
        return
    if isinstance(core, ReducedQRBinaryCore):
        for start in range(0, core.q_rows.shape[1], block_size):
            block = core.q_rows[:, start : start + block_size]
            telemetry.observe_temporary(block)
            yield block
        return
    yield from _cp_column_blocks(core, block_size, telemetry)


def _normalize_batch(value: Tensor, exponent: Tensor) -> ScaledBatch:
    maximum = value.abs().amax(dim=1)
    if bool((maximum == 0).any()) or not bool(torch.isfinite(maximum).all()):
        raise ValueError("implicit projective evaluation produced zero/nonfinite coordinates")
    _, local = torch.frexp(maximum)
    return ScaledBatch(torch.ldexp(value, -local[:, None]), exponent + local)


def _normalize_projective_batch(value: Tensor) -> Tensor:
    """Choose a bounded binary projective chart without a scale ledger."""

    maximum = value.abs().amax(dim=1)
    if bool((maximum == 0).any()) or not bool(torch.isfinite(maximum).all()):
        raise ValueError("implicit projective evaluation produced zero/nonfinite coordinates")
    _, local = torch.frexp(maximum)
    return torch.ldexp(value, -local[:, None])


def _core_apply(core: ImplicitCore, inputs: Sequence[Tensor]) -> Tensor:
    if isinstance(core, DenseCloneCore):
        if len(inputs) != len(core.input_dimensions):
            raise ValueError("dense clone core input count mismatch")
        if len(inputs) == 1:
            return inputs[0] @ core.tensor.T
        if len(inputs) == 2:
            return torch.einsum("oij,bi,bj->bo", core.tensor, inputs[0], inputs[1])
        raise ValueError("tiny dense clone only supports unary/binary primitives")
    if isinstance(core, UnaryCore):
        if len(inputs) != 1:
            raise ValueError("unary core needs one input")
        return inputs[0] @ core.matrix.T
    if isinstance(core, ReducedQRBinaryCore):
        if len(inputs) != 2:
            raise ValueError("reduced-QR binary core needs two inputs")
        tensor = core.q_rows.reshape(
            core.output_dimension, core.left_dimension, core.right_dimension
        )
        return torch.einsum("oij,bi,bj->bo", tensor, inputs[0], inputs[1])
    if len(inputs) != 2:
        raise ValueError("CP core needs two inputs")
    left = inputs[0] @ core.left_factor.T
    right = inputs[1] @ core.right_factor.T
    return (left * right) @ core.output_factor.T


def _scale_symmetric(value: Tensor, exponent: int = 0) -> ScaledMatrix:
    symmetric = value / 2 + value.T / 2
    if not bool((symmetric != 0).any()):
        return ScaledMatrix(torch.zeros_like(symmetric), 0)
    mantissa, local = _strip_power_of_two(symmetric)
    return ScaledMatrix(mantissa, exponent + local)


def _add_scaled_matrices(values: Sequence[ScaledMatrix]) -> ScaledMatrix:
    if not values:
        raise ValueError("cannot add an empty scaled-message collection")
    total = values[0].mantissa.new_zeros(values[0].mantissa.shape)
    nonzero: list[ScaledMatrix] = []
    for value in values:
        if value.mantissa.shape != total.shape:
            raise ValueError("scaled-message shapes disagree")
        if bool((value.mantissa != 0).any()):
            nonzero.append(value)
    if not nonzero:
        return ScaledMatrix(total, 0)
    center = max(value.binary_exponent for value in nonzero)
    for value in nonzero:
        relative_exponent = int(value.binary_exponent - center)
        if relative_exponent < torch.iinfo(torch.int64).min:
            raise ValueError(
                "nonzero Algorithm 2 environment message exponent is outside "
                "the direct scaled-contraction ledger: "
                f"relative_exponent={relative_exponent}"
            )
        scaled = _ldexp(value.mantissa, relative_exponent)
        if bool(((value.mantissa != 0) & (scaled == 0)).any()):
            raise ValueError(
                "nonzero Algorithm 2 environment message underflowed during "
                f"scaled contraction: relative_exponent={relative_exponent}"
            )
        total += scaled
    return _scale_symmetric(total, center)


def _accumulate_scaled_message(
    pending: dict[int, ScaledMatrix],
    key: int,
    message: ScaledMatrix,
) -> None:
    message = _scale_symmetric(message.mantissa, message.binary_exponent)
    previous = pending.get(key)
    pending[key] = (
        message
        if previous is None
        else _add_scaled_matrices((previous, message))
    )


def _role_environment(
    core: ImplicitCore,
    downstream: Tensor,
    role: int,
    telemetry: MaterializationTelemetry | None = None,
) -> Tensor:
    """Exact identity-sibling environment contraction for a canonical core."""

    if isinstance(core, DenseCloneCore):
        if core.tensor.ndim == 2:
            if role != 0:
                raise ValueError("dense unary clone has only role zero")
            result = torch.einsum(
                "oi,op,pj->ij", core.tensor, downstream, core.tensor
            )
        elif core.tensor.ndim == 3:
            if role == 0:
                result = torch.einsum(
                    "oia,op,pja->ij", core.tensor, downstream, core.tensor
                )
            elif role == 1:
                result = torch.einsum(
                    "oai,op,paj->ij", core.tensor, downstream, core.tensor
                )
            else:
                raise ValueError("dense binary clone role must be zero or one")
        else:
            raise ValueError("tiny dense clone only supports unary/binary environments")
        if telemetry is not None:
            telemetry.observe_temporary(result)
        return 0.5 * (result + result.T)
    if isinstance(core, ReducedQRBinaryCore):
        tensor = core.q_rows.reshape(
            core.output_dimension, core.left_dimension, core.right_dimension
        )
        if role == 0:
            result = torch.einsum(
                "oia,op,pja->ij", tensor, downstream, tensor
            )
        elif role == 1:
            result = torch.einsum(
                "oai,op,paj->ij", tensor, downstream, tensor
            )
        else:
            raise ValueError("reduced-QR binary role must be zero or one")
        if telemetry is not None:
            telemetry.observe_temporary(tensor, result)
        return 0.5 * (result + result.T)
    if isinstance(core, UnaryCore):
        if role != 0:
            raise ValueError("unary core has only role zero")
        result = core.matrix.T @ downstream @ core.matrix
        if telemetry is not None:
            telemetry.observe_temporary(result)
        return 0.5 * (result + result.T)
    if role not in (0, 1):
        raise ValueError("CP core has only roles zero and one")
    output_metric = core.output_factor.T @ downstream @ core.output_factor
    if role == 0:
        sibling_contraction = torch.einsum(
            "ti,si->ts", core.right_factor, core.right_factor
        )
        selected = core.left_factor
    else:
        sibling_contraction = torch.einsum(
            "ti,si->ts", core.left_factor, core.left_factor
        )
        selected = core.right_factor
    weighted = output_metric * sibling_contraction
    result = selected.T @ weighted @ selected
    if telemetry is not None:
        telemetry.observe_temporary(
            output_metric, sibling_contraction, weighted, result
        )
    return 0.5 * (result + result.T)


def _offdiagonal_ratio(value: Tensor) -> float:
    diagonal = torch.diag(torch.diagonal(value))
    return float(
        ((value - diagonal).norm() / value.norm().clamp_min(torch.finfo(value.dtype).tiny)).item()
    )
