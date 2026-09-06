"""Algorithm 1: direct local QR, retaining every shape-selected Q direction.

Tied inputs use orthonormal symmetric coordinates, including deficient
completions. Unsupported storage shapes fail before factorization.
See research/odt_reference/SOURCES.md for the original pseudocode and extension.
"""

from __future__ import annotations

import math

import torch

from xvla.train.odt_engine_v2.constants import (
    BOUNDED_EXPLICIT_RQ_METHOD,
    DIRECT_Q_PROVENANCE_TOLERANCE,
    MAXIMUM_BOUNDED_EXPLICIT_Q_ELEMENTS,
    MAXIMUM_BOUNDED_EXPLICIT_UNFOLDING_ELEMENTS,
    UNARY_RQ_METHOD,
)
from xvla.train.odt_engine_v2.ops import _cp_column_blocks, _strip_power_of_two
from xvla.train.odt_engine_v2.types import (
    CPBinaryCore, DenseCloneCore, DirectQProvenance, DirectRQDiagnostics,
    ImplicitCore, MaterializationTelemetry, ReducedQRBinaryCore, ScaledMatrix,
    Tensor, UnaryCore,
)


def _positive_diagonal_direct_rq_rows(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """M=RQ by direct reduced QR(M.T), retaining zero pivots.

    R is lower triangular (the LQ convention), with deterministic column signs.
    """
    if matrix.is_complex() or matrix.ndim != 2 or 0 in matrix.shape or not bool(torch.isfinite(matrix).all()):
        raise ValueError("direct QR requires a real finite nonempty matrix")
    q_columns, upper = torch.linalg.qr(matrix.T, mode="reduced")
    diagonal = torch.diagonal(upper)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    factor, q_rows = (signs[:, None] * upper).T, (q_columns * signs[None, :]).T
    if not bool(torch.isfinite(factor).all() and torch.isfinite(q_rows).all()):
        raise ValueError("direct QR exceeded finite arithmetic")
    return factor, q_rows


def _retained_direct_q_provenance(method: str, columns: int,
                                reconstruction_error: float,
                                minimum_pivot_ratio: float = 0.0) -> DirectQProvenance:
    """Legacy pivot fields are unassessed compatibility sentinels, never tests."""
    return DirectQProvenance(
        method=f"retained_explicit_q::{method}", verified=True, stage_count=1,
        column_blocks_compared=1, columns_compared=int(columns),
        compact_q_relative_error=0.0,
        direct_q_reconstruction_relative_error=float(reconstruction_error),
        retained_transition_elements=0, minimum_pivot_to_maximum_entry=0.0,
        conditioning_threshold=0.0, conditioning_accepted=True,
    )


def _bounded_explicit_q_allowed_shape(rows: int, columns: int, *,
                                      storage_columns: int | None = None) -> bool:
    """Bound QR input and retained Q separately, before factorization."""
    stored = columns if storage_columns is None else storage_columns
    if min(rows, columns, stored) < 1:
        raise ValueError("direct-QR unfolding dimensions must be positive")
    if stored < columns:
        raise ValueError("retained-Q storage cannot have fewer columns than its QR coordinates")
    return (rows * columns <= MAXIMUM_BOUNDED_EXPLICIT_UNFOLDING_ELEMENTS
            and min(rows, columns) * stored <= MAXIMUM_BOUNDED_EXPLICIT_Q_ELEMENTS)


def _binary_columns(core: ImplicitCore, tied_inputs: bool) -> tuple[int, int]:
    left, right = core.input_dimensions
    if min(left, right) < 1 or (tied_inputs and left != right):
        raise ValueError("tied inputs require the same positive dimension")
    return (left * (left + 1) // 2 if tied_inputs else left * right), left * right


def _bounded_explicit_q_allowed_by_structure(core: CPBinaryCore, *,
                                             tied_inputs: bool = False) -> bool:
    columns, stored = _binary_columns(core, tied_inputs)
    return _bounded_explicit_q_allowed_shape(core.output_dimension, columns,
                                              storage_columns=stored)


def _bounded_block_size(core: ImplicitCore, block_size: int) -> int:
    width = max(core.output_dimension, core.cp_rank if isinstance(core, CPBinaryCore) else 1)
    if width > MAXIMUM_BOUNDED_EXPLICIT_UNFOLDING_ELEMENTS:
        raise ValueError("one coefficient column exceeds the declared temporary bound")
    return max(1, min(max(1, int(block_size)),
                      MAXIMUM_BOUNDED_EXPLICIT_UNFOLDING_ELEMENTS // width))


def _validate_tied_input_symmetry(core: ImplicitCore, *, block_size: int = 4096) -> None:
    """Compare actual C[o,i,j] and C[o,j,i] in bounded blocks, before reduction.

    Caller attests identical source children. No full unbounded tensor is formed.
    """
    if len(core.input_dimensions) != 2:
        raise ValueError("tied-input symmetry requires a binary core")
    _, columns = _binary_columns(core, True)
    dimension = core.input_dimensions[0]
    like = core.output_factor if isinstance(core, CPBinaryCore) else (
        core.q_rows if isinstance(core, ReducedQRBinaryCore) else core.tensor)
    maximum, skew = 0.0, 0.0
    block_size = _bounded_block_size(core, block_size)
    for start in range(0, columns, block_size):
        indices = torch.arange(start, min(columns, start + block_size), device=like.device)
        i, j = indices // dimension, indices % dimension
        if isinstance(core, CPBinaryCore):
            actual = core.output_factor @ (core.left_factor[:, i] * core.right_factor[:, j])
            swapped = core.output_factor @ (core.left_factor[:, j] * core.right_factor[:, i])
        else:
            rows = like.reshape(core.output_dimension, -1)
            actual, swapped = rows[:, indices], rows[:, j * dimension + i]
        if not bool(torch.isfinite(actual).all() and torch.isfinite(swapped).all()):
            raise ValueError("tied core contains nonfinite coefficient entries")
        maximum = max(maximum, float(actual.abs().max().item()))
        skew = max(skew, float((actual - swapped).abs().max().item()))
    if skew > 1e-12 * max(maximum, torch.finfo(like.dtype).tiny):
        raise ValueError("raw tied core is asymmetric; declare its symmetric lift before ODT")


def _packed_cp_unfolding(core: CPBinaryCore, telemetry: MaterializationTelemetry,
                         block_size: int) -> Tensor:
    """Upper-triangle row-major order, diagonals once and off-diagonals sqrt(2)."""
    dimension = core.input_dimensions[0]
    result = core.output_factor.new_empty(core.output_dimension, dimension * (dimension + 1) // 2)
    offset = 0
    for i in range(dimension):
        for start in range(i, dimension, block_size):
            j = torch.arange(start, min(dimension, start + block_size), device=result.device)
            # Raw symmetry is gated; averaging only removes accepted roundoff.
            atoms = (core.left_factor[:, i:i + 1] * core.right_factor[:, j] / 2
                     + core.left_factor[:, j] * core.right_factor[:, i:i + 1] / 2)
            block = core.output_factor @ atoms
            weights = torch.ones(j.shape, dtype=result.dtype, device=result.device)
            weights[j != i] = math.sqrt(2.0)
            result[:, offset:offset + j.numel()] = block * weights
            offset += j.numel()
            telemetry.observe_temporary(j, atoms, block)
    telemetry.observe_temporary(result)
    return result


def _unpack_symmetric_q(packed: Tensor, dimension: int) -> Tensor:
    tensor = packed.new_zeros(packed.shape[0], dimension, dimension)
    offset = 0
    for i in range(dimension):
        count = dimension - i
        values = packed[:, offset:offset + count].clone()
        values[:, 1:] /= math.sqrt(2.0)
        tensor[:, i, i:] = values
        tensor[:, i:, i] = values
        offset += count
    return tensor.reshape(packed.shape[0], -1)


def _finish_retained(core: ImplicitCore, canonical: ImplicitCore, factor: Tensor,
                     error: float, columns: int, method: str, blocks: int,
                     telemetry: MaterializationTelemetry, *,
                     compared_columns: int | None = None):
    if not math.isfinite(error) or error > DIRECT_Q_PROVENANCE_TOLERANCE:
        raise ValueError(f"direct QR failed local reconstruction: {error}")
    mantissa, exponent = _strip_power_of_two(factor)
    compared = columns if compared_columns is None else compared_columns
    provenance = _retained_direct_q_provenance(method, compared, error)
    telemetry.direct_q_provenance_certificates += 1
    telemetry.direct_q_columns_compared += compared
    telemetry.maximum_direct_q_reconstruction_relative_error = max(
        telemetry.maximum_direct_q_reconstruction_relative_error, error)
    # Rank/pivot fields are unassessed sentinels. Literal Q includes all completions.
    diagnostics = DirectRQDiagnostics(
        method, error, blocks, (core.output_dimension, columns),
        0.0, 0.0, False, core.output_dimension > columns, True, exponent, provenance)
    return ScaledMatrix(mantissa, core.binary_exponent + exponent), canonical, diagnostics


def _direct_rq_unary(core: UnaryCore, telemetry: MaterializationTelemetry):
    rows, columns = core.matrix.shape
    if not _bounded_explicit_q_allowed_shape(rows, columns):
        raise ValueError(f"unsupported retained direct-Q unary shape={rows}x{columns}")
    factor, q_rows = _positive_diagonal_direct_rq_rows(core.matrix)
    telemetry.local_direct_rq_factorizations += 1
    telemetry.householder_qr_kernel_calls += 1
    error = float(((factor @ q_rows - core.matrix).norm()
                   / core.matrix.norm().clamp_min(torch.finfo(core.matrix.dtype).tiny)).item())
    telemetry.observe_persistent(q_rows)
    return _finish_retained(core, UnaryCore(q_rows, core.kind, 0), factor,
                            error, columns, UNARY_RQ_METHOD, 1, telemetry)


def _direct_rq_bounded_cp(core: CPBinaryCore, telemetry: MaterializationTelemetry, *,
                           block_size: int, tied_inputs: bool = False):
    columns, stored = _binary_columns(core, tied_inputs)
    if not _bounded_explicit_q_allowed_by_structure(core, tied_inputs=tied_inputs):
        raise ValueError("bounded direct QR exceeds unfolding or unpacked retained-Q limits")
    block_size = _bounded_block_size(core, block_size)
    if tied_inputs:
        _validate_tied_input_symmetry(core, block_size=block_size)
        unfolding = _packed_cp_unfolding(core, telemetry, block_size)
    else:
        unfolding = core.output_factor.new_empty(core.output_dimension, columns)
        offset = 0
        for block in _cp_column_blocks(core, block_size, telemetry):
            unfolding[:, offset:offset + block.shape[1]] = block
            offset += block.shape[1]
        telemetry.observe_temporary(unfolding)
    factor, packed_q = _positive_diagonal_direct_rq_rows(unfolding)
    q_rows = _unpack_symmetric_q(packed_q, core.input_dimensions[0]) if tied_inputs else packed_q
    telemetry.local_direct_rq_factorizations += 1
    telemetry.householder_qr_kernel_calls += 1
    telemetry.bounded_explicit_direct_rq_factorizations += 1
    telemetry.maximum_bounded_explicit_q_elements = max(
        telemetry.maximum_bounded_explicit_q_elements, q_rows.numel())
    if core.output_dimension > columns:
        telemetry.rectangular_direct_rq_factorizations += 1
        telemetry.maximum_rectangular_q_elements = max(
            telemetry.maximum_rectangular_q_elements, q_rows.numel())
    source_squared, residual_squared = factor.new_zeros(()), factor.new_zeros(())
    offset, blocks = 0, 0
    # Reconstruct the ORIGINAL ordered core, not only its symmetric packing.
    for block in _cp_column_blocks(core, block_size, telemetry):
        reconstructed = factor @ q_rows[:, offset:offset + block.shape[1]]
        source_squared += block.square().sum()
        residual_squared += (block - reconstructed).square().sum()
        offset += block.shape[1]
        blocks += 1
    error = float(torch.sqrt(residual_squared / source_squared.clamp_min(torch.finfo(factor.dtype).tiny)).item())
    telemetry.observe_persistent(q_rows)
    canonical = ReducedQRBinaryCore(q_rows, *core.input_dimensions, core.kind, 0)
    return _finish_retained(core, canonical, factor, error, columns,
                            BOUNDED_EXPLICIT_RQ_METHOD, blocks, telemetry,
                            compared_columns=stored)


def _direct_rq_cp_streamed(core: CPBinaryCore, telemetry: MaterializationTelemetry, *,
                          block_size: int, tied_inputs: bool = False):
    """No supported retained streamed-Q representation exists. Fail before QR."""
    columns, stored = _binary_columns(core, tied_inputs)
    raise ValueError("unsupported streamed direct-Q representation: "
                     f"shape={core.output_dimension}x{columns}, stored_columns={stored}; "
                     "requires retained Householder Q, including singular completions")


def _direct_rq_cp(core: CPBinaryCore, telemetry: MaterializationTelemetry, *,
                 block_size: int, tied_inputs: bool = False):
    kernel = (_direct_rq_bounded_cp if _bounded_explicit_q_allowed_by_structure(
        core, tied_inputs=tied_inputs) else _direct_rq_cp_streamed)
    return kernel(core, telemetry, block_size=block_size, tied_inputs=tied_inputs)


def _direct_rq_core(core: ImplicitCore, telemetry: MaterializationTelemetry, *,
                   block_size: int, tied_inputs: bool = False):
    if isinstance(core, (DenseCloneCore, ReducedQRBinaryCore)):
        raise ValueError("fresh production QR requires an unfactored unary or CP core")
    if isinstance(core, UnaryCore):
        if tied_inputs:
            raise ValueError("unary core cannot declare tied binary inputs")
        return _direct_rq_unary(core, telemetry)
    return _direct_rq_cp(core, telemetry, block_size=block_size, tied_inputs=tied_inputs)
