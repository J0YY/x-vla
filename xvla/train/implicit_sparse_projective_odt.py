"""Direct-RQ canonical ODT for implicit sparse projective primitives.

This module is the production-width successor to the dense typed oracle.  A
binary primitive is stored in CP operator form

    y = U ((L x) * (R z)),

and a unary primitive is stored as a matrix.  No unbounded order-three add,
product, attention, or FFN core is materialized.  Algorithm 1 applies a
*direct* Householder QR to each local unfolding.  A bounded unfolding retains
the returned reduced Q, including its completion when R is singular.  A
scalable full-row-rank unfolding retains its directly certified compact CP
chart.  Algorithm 1 never forms a normal equation, takes a polar square root,
calls an eigensolver, or invokes an SVD.  Every R is pushed into every parent
occurrence of a shared child.

Algorithm 2 contracts downstream environments through the canonical unary and
CP operators.  Algorithm 3 rotates every shared bond with one eigenbasis of its
aggregate occurrence environment and compensates every incident occurrence.
The represented metric is the clone-unfolded syntactic coefficient metric.  It
is not the coefficient metric after identifying reused raw variables, is not a
quotient-intrinsic metric, and is not a compression certificate.
"""

from __future__ import annotations

import math
import ast
import inspect
import sys
from collections.abc import Mapping
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Sequence

import torch
import torch.nn as nn

from xvla.nn.attention import BilinearAttention, causal_mask
from xvla.nn.bilinear import BilinearFFN
from xvla.nn.block import ChiTransformerBlock
from xvla.nn.normalization import RationalNorm


Tensor = torch.Tensor
DIRECT_RQ_METHOD = "streamed_householder_direct_reduced_rq"
UNARY_RQ_METHOD = "torch_qr_transpose_direct_reduced_rq"
BOUNDED_EXPLICIT_RQ_METHOD = "bounded_explicit_householder_direct_reduced_rq"
# Backward-compatible name for the first structurally rectangular test lane.
RECTANGULAR_RQ_METHOD = BOUNDED_EXPLICIT_RQ_METHOD
MAXIMUM_BOUNDED_EXPLICIT_UNFOLDING_ELEMENTS = 4_000_000
MAXIMUM_BOUNDED_EXPLICIT_Q_ELEMENTS = 4_000_000
LOCAL_SCALE_CERTIFICATION_TOLERANCE = 3e-10
DIRECT_Q_PROVENANCE_TOLERANCE = 3e-10
STREAMED_RQ_MINIMUM_PIVOT_TO_MAXIMUM_ENTRY = 1e-12
STREAMED_DIRECT_Q_PROVENANCE_METHOD = (
    "two_pass_sequential_tsqr_explicit_q_panel_replay"
)
# Backward-compatible name used by the original tall-rectangle regression.
MAXIMUM_RECTANGULAR_Q_ELEMENTS = MAXIMUM_BOUNDED_EXPLICIT_Q_ELEMENTS
CLAIM_BOUNDARY = (
    "Exact direct-RQ ODT of one selected-token boundary of one "
    "unchanged Padé ChiTransformerBlock, in the clone-unfolded syntactic "
    "coefficient metric. Binary primitives remain implicit CP operators and "
    "the CP FFN rank is eliminated inside one fused primitive. This is not an "
    "identified-variable or quotient-intrinsic metric, not compression, and "
    "not yet a traversal of every block and final policy head."
)

# Fixed, versioned summaries for post-Algorithm-3 compression experiments.
# The callback added below receives only scalar summaries.  In particular, it
# can never retain the per-bond eigenvectors or full eigenvalue tensors.
COMPACT_TRACE_RETENTION_TARGETS = (
    0.9,
    0.99,
    0.999,
    0.999999,
    0.999999999,
)
COMPACT_RELATIVE_VALUE_FLOORS = (
    0.0,
    1e-15,
    1e-12,
    1e-9,
    1e-6,
    1e-3,
    1e-2,
)


@dataclass(frozen=True)
class CompactSpectrumRecord:
    """Scalar-only summary emitted at the existing Algorithm 3 EVD site."""

    uid: int
    label: str
    dimension: int
    environment_binary_exponent: int
    leading_value_mantissa: float
    trace_mantissa: float
    negative_roundoff_mass_mantissa: float
    trace_retention_ranks: tuple[int, ...]
    trace_tail_mantissas: tuple[float, ...]
    relative_floor_ranks: tuple[int, ...]
    tie_extensions: tuple[int, ...]
    zero_trace: bool


def _compact_spectrum_record(
    uid: int,
    label: str,
    values: Tensor,
    environment_binary_exponent: int,
) -> CompactSpectrumRecord:
    """Summarize one descending Algorithm 3 spectrum without retaining it."""

    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("Algorithm 3 spectrum must be a nonempty vector")
    dimension = int(values.numel())
    raw_values = tuple(float(value) for value in values.tolist())
    if any(not math.isfinite(value) for value in raw_values):
        raise ValueError("Algorithm 3 spectrum must contain only finite values")
    absolute_maximum = max(abs(value) for value in raw_values)
    roundoff = (
        4096.0
        * torch.finfo(values.dtype).eps
        * float(dimension)
        * absolute_maximum
    )
    if any(value < -roundoff for value in raw_values):
        raise RuntimeError(
            "Algorithm 3 environment has a materially negative eigenvalue"
        )
    nonnegative = tuple(max(0.0, value) for value in raw_values)
    total = math.fsum(nonnegative)
    zero_trace = total <= roundoff
    leading = nonnegative[0]
    negative_roundoff_mass = math.fsum(
        -value for value in raw_values if value < 0.0
    )

    retention_ranks: list[int] = []
    tail_mantissas: list[float] = []
    tie_extensions: list[int] = []
    if zero_trace:
        retention_ranks = [1 for _ in COMPACT_TRACE_RETENTION_TARGETS]
        tail_mantissas = [0.0 for _ in COMPACT_TRACE_RETENTION_TARGETS]
        tie_extensions = [0 for _ in COMPACT_TRACE_RETENTION_TARGETS]
    else:
        cumulative: list[float] = []
        partial = 0.0
        for value in nonnegative:
            partial += value
            cumulative.append(partial)
        for target in COMPACT_TRACE_RETENTION_TARGETS:
            threshold = total * float(target)
            raw_rank = next(
                (
                    index
                    for index, value in enumerate(cumulative, start=1)
                    if value >= threshold
                ),
                dimension,
            )
            boundary = nonnegative[raw_rank - 1]
            tie_tolerance = max(roundoff, absolute_maximum * 1e-12)
            tied_rank = sum(
                value >= boundary - tie_tolerance for value in nonnegative
            )
            tied_rank = min(dimension, max(raw_rank, tied_rank))
            retention_ranks.append(tied_rank)
            tie_extensions.append(tied_rank - raw_rank)
            tail_mantissas.append(max(0.0, total - cumulative[tied_rank - 1]))

    relative_floor_ranks = tuple(
        max(
            1,
            sum(
                value > (absolute_maximum * float(floor) + roundoff)
                for value in nonnegative
            ),
        )
        for floor in COMPACT_RELATIVE_VALUE_FLOORS
    )
    return CompactSpectrumRecord(
        uid=int(uid),
        label=label,
        dimension=dimension,
        environment_binary_exponent=int(environment_binary_exponent),
        leading_value_mantissa=leading,
        trace_mantissa=total,
        negative_roundoff_mass_mantissa=negative_roundoff_mass,
        trace_retention_ranks=tuple(retention_ranks),
        trace_tail_mantissas=tuple(tail_mantissas),
        relative_floor_ranks=relative_floor_ranks,
        tie_extensions=tuple(tie_extensions),
        zero_trace=zero_trace,
    )


def _validate_real_finite(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point() or value.is_complex():
        raise TypeError(f"{name} must be a real floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


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


@dataclass(frozen=True)
class ScaledMatrix:
    mantissa: Tensor
    binary_exponent: int = 0

    def __post_init__(self) -> None:
        _validate_real_finite(self.mantissa, "scaled matrix mantissa")
        if self.mantissa.ndim != 2:
            raise ValueError("scaled matrix mantissa must be a matrix")


@dataclass(frozen=True)
class ScaledBatch:
    mantissa: Tensor
    binary_exponent: Tensor


@dataclass(frozen=True)
class PhysicalSourceSpec:
    """One independent raw source bond in a heterogeneous projective DAG."""

    key: str
    label: str
    width: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("physical source key must be a nonempty string")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("physical source label must be a nonempty string")
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width < 1:
            raise ValueError("physical source width must be a positive integer")


@dataclass(frozen=True)
class ProjectiveConstant:
    """A compile-time projective value, never represented by a graph leaf."""

    label: str
    mantissa: Tensor
    binary_exponent: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("projective constant label must be nonempty")
        _validate_real_finite(self.mantissa, f"projective constant {self.label}")
        if self.mantissa.ndim != 1 or self.mantissa.numel() < 1:
            raise ValueError("projective constant must be a nonempty vector")
        if isinstance(self.binary_exponent, bool) or not isinstance(
            self.binary_exponent, int
        ):
            raise ValueError("projective constant exponent must be an integer")

    @property
    def output_dimension(self) -> int:
        return int(self.mantissa.numel())


@dataclass
class MaterializationTelemetry:
    raw_order_three_core_materializations: int = 0
    raw_width_cubic_core_materializations: int = 0
    maximum_persistent_tensor_elements: int = 0
    maximum_temporary_tensor_elements: int = 0
    maximum_temporary_tensor_order: int = 0
    local_direct_rq_factorizations: int = 0
    householder_qr_kernel_calls: int = 0
    svd_calls: int = 0
    polar_calls: int = 0
    normal_equation_factorizations: int = 0
    fused_ffn_primitive_count: int = 0
    ephemeral_ffn_bond_diagonalizations: int = 0
    rectangular_direct_rq_factorizations: int = 0
    maximum_rectangular_q_elements: int = 0
    bounded_explicit_direct_rq_factorizations: int = 0
    maximum_bounded_explicit_q_elements: int = 0
    constant_unary_folds: int = 0
    constant_cp_to_unary_folds: int = 0
    constant_cp_to_constant_folds: int = 0
    direct_q_provenance_certificates: int = 0
    streamed_direct_q_provenance_certificates: int = 0
    streamed_direct_q_replay_qr_factorizations: int = 0
    direct_q_columns_compared: int = 0
    maximum_direct_q_transition_elements: int = 0
    maximum_direct_q_panel_elements: int = 0
    maximum_direct_q_compact_relative_error: float = 0.0
    maximum_direct_q_reconstruction_relative_error: float = 0.0

    def observe_persistent(self, *values: Tensor) -> None:
        for value in values:
            self.maximum_persistent_tensor_elements = max(
                self.maximum_persistent_tensor_elements, value.numel()
            )

    def observe_temporary(self, *values: Tensor) -> None:
        for value in values:
            self.maximum_temporary_tensor_elements = max(
                self.maximum_temporary_tensor_elements, value.numel()
            )
            self.maximum_temporary_tensor_order = max(
                self.maximum_temporary_tensor_order, value.ndim
            )


@dataclass
class UnaryCore:
    matrix: Tensor
    kind: str
    binary_exponent: int = 0

    def clone(self) -> "UnaryCore":
        return UnaryCore(self.matrix.clone(), self.kind, self.binary_exponent)

    @property
    def output_dimension(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def input_dimensions(self) -> tuple[int, ...]:
        return (int(self.matrix.shape[1]),)


@dataclass(frozen=True)
class DirectQProvenance:
    """Non-Gram certificate binding a compact Q core to direct TSQR Q panels."""

    method: str
    verified: bool
    stage_count: int
    column_blocks_compared: int
    columns_compared: int
    compact_q_relative_error: float
    direct_q_reconstruction_relative_error: float
    retained_transition_elements: int
    minimum_pivot_to_maximum_entry: float
    conditioning_threshold: float
    conditioning_accepted: bool


@dataclass
class CPBinaryCore:
    output_factor: Tensor
    left_factor: Tensor
    right_factor: Tensor
    kind: str
    binary_exponent: int = 0
    direct_q_provenance: DirectQProvenance | None = None

    def clone(self) -> "CPBinaryCore":
        return CPBinaryCore(
            self.output_factor.clone(),
            self.left_factor.clone(),
            self.right_factor.clone(),
            self.kind,
            self.binary_exponent,
            self.direct_q_provenance,
        )

    @property
    def output_dimension(self) -> int:
        return int(self.output_factor.shape[0])

    @property
    def input_dimensions(self) -> tuple[int, ...]:
        return (int(self.left_factor.shape[1]), int(self.right_factor.shape[1]))

    @property
    def cp_rank(self) -> int:
        return int(self.output_factor.shape[1])


@dataclass
class ReducedQRBinaryCore:
    """Bounded binary reduced-Q retained directly after local Householder QR.

    ``q_rows`` is the reduced-Q transpose with shape
    ``(output_dimension, left_dimension * right_dimension)``.  This is a
    first-class production core, not the no-memo dense clone oracle.  For an
    ``m`` by ``n`` unfolding its output dimension is ``min(m, n)``.  Keeping
    it is the direct-QR representation for every bounded unfolding, including
    deficient square/wide cases where solving through the triangular factor
    would be invalid.
    """

    q_rows: Tensor
    left_dimension: int
    right_dimension: int
    kind: str
    binary_exponent: int = 0

    def clone(self) -> "ReducedQRBinaryCore":
        return ReducedQRBinaryCore(
            self.q_rows.clone(),
            self.left_dimension,
            self.right_dimension,
            self.kind,
            self.binary_exponent,
        )

    @property
    def output_dimension(self) -> int:
        return int(self.q_rows.shape[0])

    @property
    def input_dimensions(self) -> tuple[int, ...]:
        return (self.left_dimension, self.right_dimension)


@dataclass
class DenseCloneCore:
    """Tiny explicit-clone oracle core, forbidden in production networks."""

    tensor: Tensor
    kind: str
    binary_exponent: int = 0

    def clone(self) -> "DenseCloneCore":
        return DenseCloneCore(self.tensor.clone(), self.kind, self.binary_exponent)

    @property
    def output_dimension(self) -> int:
        return int(self.tensor.shape[0])

    @property
    def input_dimensions(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.tensor.shape[1:])


ImplicitCore = UnaryCore | CPBinaryCore | ReducedQRBinaryCore | DenseCloneCore


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


def _core_output_dimension(core: ImplicitCore) -> int:
    return core.output_dimension


def _core_input_dimensions(core: ImplicitCore) -> tuple[int, ...]:
    return core.input_dimensions


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


def _materialize_core(
    core: ImplicitCore,
    *,
    maximum_elements: int,
    telemetry: MaterializationTelemetry,
) -> Tensor:
    shape = (core.output_dimension, *_core_input_dimensions(core))
    elements = math.prod(shape)
    if elements > maximum_elements:
        raise ValueError("dense materialization is restricted to the tiny oracle")
    if isinstance(core, UnaryCore):
        return _ldexp(core.matrix, core.binary_exponent)
    if isinstance(core, DenseCloneCore):
        return _ldexp(core.tensor, core.binary_exponent)
    if isinstance(core, ReducedQRBinaryCore):
        return _ldexp(
            core.q_rows.reshape(
                core.output_dimension, core.left_dimension, core.right_dimension
            ),
            core.binary_exponent,
        )
    telemetry.raw_order_three_core_materializations += 1
    if elements >= core.output_dimension**3:
        telemetry.raw_width_cubic_core_materializations += 1
    dense = torch.einsum(
        "ot,ti,tj->oij",
        core.output_factor,
        core.left_factor,
        core.right_factor,
    )
    return _ldexp(dense, core.binary_exponent)


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


def _scaled_batch_relative_error(actual: ScaledBatch, expected: ScaledBatch) -> float:
    if actual.mantissa.shape != expected.mantissa.shape:
        raise ValueError("scaled batches have different shapes")
    errors = []
    for index in range(actual.mantissa.shape[0]):
        errors.append(
            _scaled_matrix_relative_error(
                ScaledMatrix(actual.mantissa[index : index + 1], int(actual.binary_exponent[index])),
                ScaledMatrix(expected.mantissa[index : index + 1], int(expected.binary_exponent[index])),
            )
        )
    return max(errors, default=0.0)


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


def _positive_diagonal_direct_rq_rows(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Return M=RQ via direct QR(M.T), with a deterministic sign chart."""

    q_columns, upper = torch.linalg.qr(matrix.T, mode="reduced")
    signs = torch.where(
        torch.diagonal(upper) < 0,
        -torch.ones((), dtype=matrix.dtype, device=matrix.device),
        torch.ones((), dtype=matrix.dtype, device=matrix.device),
    )
    q_columns = q_columns * signs[None, :]
    upper = signs[:, None] * upper
    return upper.T, q_columns.T


def _direct_tsqr_panel(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Return one explicit direct-QR panel for the streamed TSQR ledger."""

    return torch.linalg.qr(matrix, mode="reduced")


def _solve_compact_q(factor: Tensor, output_factor: Tensor) -> Tensor:
    """Recover the compact CP Q chart from the direct triangular RQ factor."""

    return torch.linalg.solve_triangular(
        factor,
        output_factor,
        upper=False,
    )


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


@dataclass(frozen=True)
class DirectRQDiagnostics:
    method: str
    factorization_relative_error: float
    streamed_column_blocks: int
    unfolding_shape: tuple[int, int]
    minimum_diagonal_to_maximum_entry: float
    diagonal_condition_proxy: float
    full_row_rank: bool
    structurally_reduced: bool = False
    literal_q_chart_resolved: bool = False
    factor_normalization_binary_exponent: int = 0
    direct_q_provenance: DirectQProvenance | None = None


@dataclass(frozen=True)
class AbsorptionScaleLedger:
    pushed_occurrences: int
    verified_occurrences: int
    maximum_scaled_relative_error: float
    maximum_exponent_delta: int


def _retained_direct_q_provenance(
    method: str,
    columns: int,
    reconstruction_error: float,
    minimum_pivot_ratio: float,
) -> DirectQProvenance:
    """Describe a bounded Q returned directly by the local QR call."""

    return DirectQProvenance(
        method=f"retained_explicit_q::{method}",
        verified=True,
        stage_count=1,
        column_blocks_compared=1,
        columns_compared=int(columns),
        compact_q_relative_error=0.0,
        direct_q_reconstruction_relative_error=float(reconstruction_error),
        retained_transition_elements=0,
        minimum_pivot_to_maximum_entry=float(minimum_pivot_ratio),
        conditioning_threshold=0.0,
        conditioning_accepted=True,
    )


def _bounded_explicit_q_allowed_shape(rows: int, columns: int) -> bool:
    """Return the pre-QR bounded direct-Q route for one unfolding shape."""

    if rows < 1 or columns < 1:
        raise ValueError("direct-RQ unfolding dimensions must be positive")
    unfolding_elements = rows * columns
    q_elements = min(rows, columns) * columns
    return (
        unfolding_elements <= MAXIMUM_BOUNDED_EXPLICIT_UNFOLDING_ELEMENTS
        and q_elements <= MAXIMUM_BOUNDED_EXPLICIT_Q_ELEMENTS
    )


def _bounded_explicit_q_allowed_by_structure(core: CPBinaryCore) -> bool:
    """Select retained direct Q before QR, using storage structure only.

    Every unfolding whose source and directly returned reduced Q both fit their
    declared bounds retains that Q, including any Householder completion.  This
    is not a numerical-rank branch and is decided before any factorization.
    """

    rows = core.output_dimension
    columns = math.prod(core.input_dimensions)
    return _bounded_explicit_q_allowed_shape(rows, columns)


def _direct_rq_unary(
    core: UnaryCore,
    telemetry: MaterializationTelemetry,
) -> tuple[ScaledMatrix, UnaryCore, DirectRQDiagnostics]:
    telemetry.local_direct_rq_factorizations += 1
    telemetry.householder_qr_kernel_calls += 1
    factor, rows = _positive_diagonal_direct_rq_rows(core.matrix)
    factor_mantissa, factor_exp = _strip_power_of_two(factor)
    reconstructed = factor @ rows
    scale = core.matrix.norm().clamp_min(torch.finfo(core.matrix.dtype).tiny)
    factor_error = float(((reconstructed - core.matrix).norm() / scale).item())
    diagonal = torch.diagonal(factor).abs()
    maximum = factor.abs().max().clamp_min(torch.finfo(factor.dtype).tiny)
    minimum_ratio = float((diagonal.min() / maximum).item()) if diagonal.numel() else 0.0
    full_row_rank = factor.shape[1] == core.matrix.shape[0] and minimum_ratio > (
        100.0 * torch.finfo(factor.dtype).eps
    )
    literal_q_chart_resolved = minimum_ratio > 100.0 * torch.finfo(factor.dtype).eps
    condition_proxy = (
        float((diagonal.max() / diagonal.min()).item())
        if diagonal.numel() and float(diagonal.min().item()) > 0.0
        else math.inf
    )
    if not bool(torch.isfinite(factor).all() and torch.isfinite(rows).all()):
        raise ValueError("direct unary RQ produced a nonfinite factor")
    provenance = _retained_direct_q_provenance(
        UNARY_RQ_METHOD,
        core.matrix.shape[1],
        factor_error,
        minimum_ratio,
    )
    telemetry.direct_q_provenance_certificates += 1
    telemetry.direct_q_columns_compared += provenance.columns_compared
    telemetry.maximum_direct_q_reconstruction_relative_error = max(
        telemetry.maximum_direct_q_reconstruction_relative_error,
        provenance.direct_q_reconstruction_relative_error,
    )
    return (
        ScaledMatrix(factor_mantissa, core.binary_exponent + factor_exp),
        UnaryCore(rows, core.kind, 0),
        DirectRQDiagnostics(
            UNARY_RQ_METHOD,
            factor_error,
            1,
            (core.matrix.shape[0], core.matrix.shape[1]),
            minimum_ratio,
            condition_proxy,
            full_row_rank,
            core.matrix.shape[0] > core.matrix.shape[1],
            literal_q_chart_resolved,
            factor_exp,
            provenance,
        ),
    )


def _direct_rq_bounded_cp(
    core: CPBinaryCore,
    telemetry: MaterializationTelemetry,
    *,
    block_size: int,
) -> tuple[ScaledMatrix, ReducedQRBinaryCore, DirectRQDiagnostics]:
    """Direct reduced QR retaining Q for one bounded CP unfolding.

    Routing is determined only by declared storage bounds, never by a numerical
    rank test.  Thus a deficient square/wide unfolding remains exact without a
    triangular solve, while a tall unfolding reduces its output bond to the
    number of columns.  In both cases the retained reduced-Q rows are bounded.
    """

    rows = core.output_dimension
    columns = math.prod(core.input_dimensions)
    unfolding_elements = rows * columns
    q_rows = min(rows, columns)
    q_elements = q_rows * columns
    if not _bounded_explicit_q_allowed_by_structure(core):
        raise ValueError(
            "bounded direct-RQ unfolding exceeds the explicit-core limits"
        )
    source_blocks = tuple(_cp_column_blocks(core, max(1, block_size), telemetry))
    if not source_blocks:
        raise RuntimeError("rectangular CP unfolding emitted no columns")
    unfolding = torch.cat(source_blocks, dim=1)
    if unfolding.shape != (rows, columns):
        raise RuntimeError("rectangular CP unfolding has the wrong shape")
    telemetry.observe_temporary(unfolding)
    telemetry.local_direct_rq_factorizations += 1
    telemetry.householder_qr_kernel_calls += 1
    telemetry.bounded_explicit_direct_rq_factorizations += 1
    if rows > columns:
        telemetry.rectangular_direct_rq_factorizations += 1
    factor, reduced_q_rows = _positive_diagonal_direct_rq_rows(unfolding)
    reconstructed = factor @ reduced_q_rows
    scale = unfolding.norm().clamp_min(torch.finfo(unfolding.dtype).tiny)
    factor_error = float(((reconstructed - unfolding).norm() / scale).item())
    factor_mantissa, factor_exp = _strip_power_of_two(factor)
    diagonal = torch.diagonal(factor).abs()
    maximum = factor.abs().max().clamp_min(torch.finfo(factor.dtype).tiny)
    minimum_ratio = float((diagonal.min() / maximum).item()) if diagonal.numel() else 0.0
    literal_q_chart_resolved = minimum_ratio > 100.0 * torch.finfo(factor.dtype).eps
    safe_minimum = diagonal.min().clamp_min(torch.finfo(factor.dtype).tiny)
    condition_proxy = (
        float((diagonal.max() / safe_minimum).item()) if diagonal.numel() else 1.0
    )
    if not bool(
        torch.isfinite(factor).all()
        and torch.isfinite(reduced_q_rows).all()
    ):
        raise ValueError("bounded direct CP RQ produced nonfinite factors")
    telemetry.maximum_bounded_explicit_q_elements = max(
        telemetry.maximum_bounded_explicit_q_elements, q_elements
    )
    if rows > columns:
        telemetry.maximum_rectangular_q_elements = max(
            telemetry.maximum_rectangular_q_elements, q_elements
        )
    telemetry.observe_persistent(reduced_q_rows)
    canonical = ReducedQRBinaryCore(
        reduced_q_rows,
        core.input_dimensions[0],
        core.input_dimensions[1],
        core.kind,
        0,
    )
    provenance = _retained_direct_q_provenance(
        BOUNDED_EXPLICIT_RQ_METHOD,
        columns,
        factor_error,
        minimum_ratio,
    )
    telemetry.direct_q_provenance_certificates += 1
    telemetry.direct_q_columns_compared += provenance.columns_compared
    telemetry.maximum_direct_q_reconstruction_relative_error = max(
        telemetry.maximum_direct_q_reconstruction_relative_error,
        provenance.direct_q_reconstruction_relative_error,
    )
    return (
        ScaledMatrix(factor_mantissa, core.binary_exponent + factor_exp),
        canonical,
        DirectRQDiagnostics(
            BOUNDED_EXPLICIT_RQ_METHOD,
            factor_error,
            len(source_blocks),
            (rows, columns),
            minimum_ratio,
            condition_proxy,
            bool(rows <= columns and literal_q_chart_resolved),
            bool(rows > columns),
            literal_q_chart_resolved,
            factor_exp,
            provenance,
        ),
    )


def _direct_rq_cp_streamed(
    core: CPBinaryCore,
    telemetry: MaterializationTelemetry,
    *,
    block_size: int,
) -> tuple[
    ScaledMatrix,
    CPBinaryCore | ReducedQRBinaryCore,
    DirectRQDiagnostics,
]:
    """Direct TSQR with a scalable explicit-Q provenance certificate.

    The first pass keeps only the square top panel from each sequential TSQR
    Q.  Those panels are converted in place to suffix actions.  A second pass
    recomputes one explicit Q panel at a time and thereby emits every block of
    the final direct Householder Q without ever retaining the dense global Q.
    Every emitted element is compared with the compact CP Q obtained from the
    triangular solve, and ``M = R Q_direct`` is reconstructed independently.
    """

    rows = core.output_dimension
    columns = math.prod(core.input_dimensions)
    if columns < rows:
        raise ValueError(
            "unbounded CP unfolding has fewer columns than rows; its exact "
            "reduced-Q core exceeds the declared storage bound: "
            f"shape={rows}x{columns}"
        )
    block_size = max(int(block_size), rows)
    upper: Tensor | None = None
    transition_suffixes: list[Tensor | None] = []
    # Count only the K-1 top transition panels.  Retained-list telemetry below
    # separately records the K suffix slots, including the final sign chart.
    retained_transition_elements = 0
    blocks = 0
    telemetry.local_direct_rq_factorizations += 1
    for block in _cp_column_blocks(core, block_size, telemetry):
        had_upper = upper is not None
        stacked = block.T if not had_upper else torch.cat((upper, block.T), dim=0)
        telemetry.observe_temporary(stacked)
        q_panel, upper = _direct_tsqr_panel(stacked)
        telemetry.observe_temporary(q_panel)
        telemetry.householder_qr_kernel_calls += 1
        telemetry.maximum_direct_q_panel_elements = max(
            telemetry.maximum_direct_q_panel_elements,
            q_panel.numel(),
        )
        if had_upper:
            transition = q_panel[:rows].clone()
            transition_suffixes.append(transition)
            retained_transition_elements += transition.numel()
        else:
            transition_suffixes.append(None)
        blocks += 1
    if upper is None or upper.shape != (rows, rows):
        raise RuntimeError("streamed direct QR did not produce a square R factor")
    signs = torch.where(
        torch.diagonal(upper) < 0,
        -torch.ones((), dtype=upper.dtype, device=upper.device),
        torch.ones((), dtype=upper.dtype, device=upper.device),
    )
    upper = signs[:, None] * upper
    factor = upper.T
    diagonal = torch.diagonal(factor).abs()
    maximum_entry = factor.abs().max().clamp_min(torch.finfo(factor.dtype).tiny)
    minimum_ratio = float((diagonal.min() / maximum_entry).item())
    conditioning_threshold = max(
        STREAMED_RQ_MINIMUM_PIVOT_TO_MAXIMUM_ENTRY,
        100.0 * torch.finfo(factor.dtype).eps,
    )
    conditioning_accepted = minimum_ratio > conditioning_threshold
    if not conditioning_accepted:
        raise ValueError(
            "streamed direct-RQ conditioning gate rejected a near-null triangular "
            "factor; no alternate factorization is permitted: "
            f"shape={rows}x{columns} blocks={blocks} "
            f"minimum_pivot_to_maximum_entry={minimum_ratio:.17g} "
            f"threshold={conditioning_threshold:.17g}"
        )
    canonical_output = _solve_compact_q(factor, core.output_factor)
    canonical = CPBinaryCore(
        canonical_output,
        core.left_factor,
        core.right_factor,
        core.kind,
        0,
    )

    # Convert the stored square transition panels into right suffix actions.
    # The final sign matrix transports the positive-diagonal chart back into
    # every direct Q block.  The list remains O(blocks * rows^2), and is
    # released before this local factorization returns.
    running_suffix = torch.diag(signs)
    for index in range(blocks - 1, -1, -1):
        transition = transition_suffixes[index]
        transition_suffixes[index] = running_suffix
        if transition is not None:
            running_suffix = transition @ running_suffix
            telemetry.observe_temporary(running_suffix)
    retained_suffix_list_elements = blocks * rows * rows
    telemetry.maximum_direct_q_transition_elements = max(
        telemetry.maximum_direct_q_transition_elements,
        retained_suffix_list_elements,
    )

    compact_residual_squared = factor.new_zeros(())
    direct_residual_squared = factor.new_zeros(())
    compact_q_residual_squared = factor.new_zeros(())
    direct_q_squared = factor.new_zeros(())
    source_squared = factor.new_zeros(())
    maximum_q_entry_residual = 0.0
    maximum_direct_q_entry = 0.0
    replay_upper: Tensor | None = None
    compact_iterator = iter(_cp_column_blocks(canonical, block_size, telemetry))
    columns_compared = 0
    compared_blocks = 0
    for index, source in enumerate(_cp_column_blocks(core, block_size, telemetry)):
        try:
            compact_q_block = next(compact_iterator)
        except StopIteration as error:
            raise RuntimeError(
                "compact CP Q emitted fewer blocks than the direct TSQR replay"
            ) from error
        had_upper = replay_upper is not None
        stacked = (
            source.T
            if not had_upper
            else torch.cat((replay_upper, source.T), dim=0)
        )
        direct_q_panel, replay_upper = _direct_tsqr_panel(stacked)
        telemetry.householder_qr_kernel_calls += 1
        telemetry.streamed_direct_q_replay_qr_factorizations += 1
        telemetry.observe_temporary(stacked, direct_q_panel)
        panel_rows = direct_q_panel if not had_upper else direct_q_panel[rows:]
        suffix = transition_suffixes[index]
        if suffix is None:
            raise RuntimeError("direct TSQR suffix action is missing")
        direct_q_block = (panel_rows @ suffix).T
        telemetry.observe_temporary(direct_q_block)
        if direct_q_block.shape != compact_q_block.shape:
            raise RuntimeError("direct and compact Q blocks have different shapes")
        q_residual = compact_q_block - direct_q_block
        maximum_q_entry_residual = max(
            maximum_q_entry_residual,
            float(q_residual.abs().max().item()),
        )
        maximum_direct_q_entry = max(
            maximum_direct_q_entry,
            float(direct_q_block.abs().max().item()),
        )
        compact_q_residual_squared += (
            q_residual.square().sum()
        )
        direct_q_squared += direct_q_block.square().sum()
        compact_residual_squared += (
            source - factor @ compact_q_block
        ).square().sum()
        direct_residual_squared += (
            source - factor @ direct_q_block
        ).square().sum()
        source_squared += source.square().sum()
        columns_compared += source.shape[1]
        compared_blocks += 1
    try:
        next(compact_iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError(
            "compact CP Q emitted more blocks than the direct TSQR replay"
        )
    if compared_blocks != blocks or columns_compared != columns:
        raise RuntimeError("direct TSQR provenance did not cover every Q element")
    factorization_error = float(
        torch.sqrt(
            compact_residual_squared
            / source_squared.clamp_min(torch.finfo(factor.dtype).tiny)
        ).item()
    )
    direct_reconstruction_error = float(
        torch.sqrt(
            direct_residual_squared
            / source_squared.clamp_min(torch.finfo(factor.dtype).tiny)
        ).item()
    )
    compact_q_error = float(
        torch.sqrt(
            compact_q_residual_squared
            / direct_q_squared.clamp_min(torch.finfo(factor.dtype).tiny)
        ).item()
    )
    elementwise_q_error = maximum_q_entry_residual / max(
        maximum_direct_q_entry,
        torch.finfo(factor.dtype).tiny,
    )
    provenance_tolerance = DIRECT_Q_PROVENANCE_TOLERANCE
    provenance_verified = bool(
        math.isfinite(compact_q_error)
        and math.isfinite(direct_reconstruction_error)
        and compact_q_error <= provenance_tolerance
        and elementwise_q_error <= provenance_tolerance
        and direct_reconstruction_error <= provenance_tolerance
        and conditioning_accepted
    )
    if not provenance_verified:
        raise ValueError(
            "streamed compact CP Q failed its direct TSQR Q provenance certificate: "
            f"shape={rows}x{columns} blocks={blocks} "
            f"compact_q_error={max(compact_q_error, elementwise_q_error):.17g} "
            f"direct_reconstruction_error={direct_reconstruction_error:.17g} "
            f"factorization_error={factorization_error:.17g} "
            f"minimum_pivot_to_maximum_entry={minimum_ratio:.17g}"
        )
    factor_mantissa, factor_exp = _strip_power_of_two(factor)
    condition_proxy = float((diagonal.max() / diagonal.min()).item())
    if not bool(
        torch.isfinite(factor).all()
        and torch.isfinite(canonical_output).all()
        and math.isfinite(condition_proxy)
    ):
        raise ValueError("direct CP RQ produced a nonfinite factor")
    provenance = DirectQProvenance(
        method=STREAMED_DIRECT_Q_PROVENANCE_METHOD,
        verified=provenance_verified,
        stage_count=blocks,
        column_blocks_compared=compared_blocks,
        columns_compared=columns_compared,
        compact_q_relative_error=max(compact_q_error, elementwise_q_error),
        direct_q_reconstruction_relative_error=direct_reconstruction_error,
        retained_transition_elements=retained_transition_elements,
        minimum_pivot_to_maximum_entry=minimum_ratio,
        conditioning_threshold=conditioning_threshold,
        conditioning_accepted=conditioning_accepted,
    )
    canonical.direct_q_provenance = provenance
    telemetry.direct_q_provenance_certificates += 1
    telemetry.streamed_direct_q_provenance_certificates += 1
    telemetry.direct_q_columns_compared += columns_compared
    telemetry.maximum_direct_q_compact_relative_error = max(
        telemetry.maximum_direct_q_compact_relative_error,
        provenance.compact_q_relative_error,
    )
    telemetry.maximum_direct_q_reconstruction_relative_error = max(
        telemetry.maximum_direct_q_reconstruction_relative_error,
        direct_reconstruction_error,
    )
    return (
        ScaledMatrix(factor_mantissa, core.binary_exponent + factor_exp),
        canonical,
        DirectRQDiagnostics(
            DIRECT_RQ_METHOD,
            factorization_error,
            blocks,
            (rows, columns),
            minimum_ratio,
            condition_proxy,
            True,
            False,
            True,
            factor_exp,
            provenance,
        ),
    )


def _direct_rq_cp(
    core: CPBinaryCore,
    telemetry: MaterializationTelemetry,
    *,
    block_size: int,
) -> tuple[
    ScaledMatrix,
    CPBinaryCore | ReducedQRBinaryCore,
    DirectRQDiagnostics,
]:
    """Select a direct-QR representation by storage shape, never by rank."""

    rows = core.output_dimension
    columns = math.prod(core.input_dimensions)
    if _bounded_explicit_q_allowed_by_structure(core):
        return _direct_rq_bounded_cp(core, telemetry, block_size=block_size)
    return _direct_rq_cp_streamed(core, telemetry, block_size=block_size)


def _direct_rq_core(
    core: ImplicitCore,
    telemetry: MaterializationTelemetry,
    *,
    block_size: int,
) -> tuple[ScaledMatrix, ImplicitCore, DirectRQDiagnostics]:
    if isinstance(core, DenseCloneCore):
        raise ValueError("production direct-RQ path cannot consume a dense clone core")
    if isinstance(core, ReducedQRBinaryCore):
        raise ValueError(
            "an already reduced-QR binary core cannot enter a fresh Algorithm 1 sweep"
        )
    if isinstance(core, UnaryCore):
        return _direct_rq_unary(core, telemetry)
    return _direct_rq_cp(core, telemetry, block_size=block_size)


def _independent_dense_clone_rq(
    core: ImplicitCore,
    telemetry: MaterializationTelemetry,
) -> tuple[ScaledMatrix, DenseCloneCore, DirectRQDiagnostics]:
    """Independent tiny oracle: explicit unfolding plus dense direct RQ."""

    elements = core.output_dimension * math.prod(core.input_dimensions)
    if elements > 100_000:
        raise ValueError("explicit clone RQ is restricted to the tiny oracle")
    if isinstance(core, UnaryCore):
        dense = core.matrix
    elif isinstance(core, DenseCloneCore):
        dense = core.tensor
    elif isinstance(core, ReducedQRBinaryCore):
        dense = core.q_rows.reshape(
            core.output_dimension, core.left_dimension, core.right_dimension
        )
    else:
        telemetry.raw_order_three_core_materializations += 1
        dense = torch.einsum(
            "ot,ti,tj->oij",
            core.output_factor,
            core.left_factor,
            core.right_factor,
        )
    telemetry.observe_temporary(dense)
    rows = dense.reshape(dense.shape[0], -1)
    telemetry.local_direct_rq_factorizations += 1
    telemetry.householder_qr_kernel_calls += 1
    q_columns, upper = torch.linalg.qr(rows.T, mode="reduced")
    signs = torch.where(
        torch.diagonal(upper) < 0,
        -torch.ones((), dtype=rows.dtype, device=rows.device),
        torch.ones((), dtype=rows.dtype, device=rows.device),
    )
    q_columns = q_columns * signs[None, :]
    upper = signs[:, None] * upper
    factor, q_rows = upper.T, q_columns.T
    reconstructed = factor @ q_rows
    scale = rows.norm().clamp_min(torch.finfo(rows.dtype).tiny)
    factor_error = float(((reconstructed - rows).norm() / scale).item())
    diagonal = torch.diagonal(factor).abs()
    maximum = factor.abs().max().clamp_min(torch.finfo(factor.dtype).tiny)
    minimum_ratio = float((diagonal.min() / maximum).item()) if diagonal.numel() else 0.0
    full_row_rank = factor.shape[1] == rows.shape[0] and minimum_ratio > (
        100.0 * torch.finfo(factor.dtype).eps
    )
    literal_q_chart_resolved = minimum_ratio > 100.0 * torch.finfo(factor.dtype).eps
    condition_proxy = (
        float((diagonal.max() / diagonal.min()).item())
        if diagonal.numel() and float(diagonal.min().item()) > 0.0
        else math.inf
    )
    factor_mantissa, factor_exponent = _strip_power_of_two(factor)
    provenance = _retained_direct_q_provenance(
        "independent_dense_torch_qr_transpose_direct_reduced_rq",
        rows.shape[1],
        factor_error,
        minimum_ratio,
    )
    telemetry.direct_q_provenance_certificates += 1
    telemetry.direct_q_columns_compared += provenance.columns_compared
    telemetry.maximum_direct_q_reconstruction_relative_error = max(
        telemetry.maximum_direct_q_reconstruction_relative_error,
        provenance.direct_q_reconstruction_relative_error,
    )
    return (
        ScaledMatrix(
            factor_mantissa, core.binary_exponent + factor_exponent
        ),
        DenseCloneCore(
            q_rows.reshape((q_rows.shape[0],) + tuple(dense.shape[1:])),
            f"explicit_clone::{core.kind}",
            0,
        ),
        DirectRQDiagnostics(
            "independent_dense_torch_qr_transpose_direct_reduced_rq",
            factor_error,
            1,
            (rows.shape[0], rows.shape[1]),
            minimum_ratio,
            condition_proxy,
            full_row_rank,
            rows.shape[0] > rows.shape[1],
            literal_q_chart_resolved,
            factor_exponent,
            provenance,
        ),
    )


@dataclass(eq=False)
class ImplicitNode:
    uid: int
    label: str
    core: ImplicitCore
    children: tuple["ImplicitNode", ...] = ()
    physical_token: int | None = None
    origin_uid: int | None = None
    physical_source_key: str | None = None

    @property
    def output_dimension(self) -> int:
        return _core_output_dimension(self.core)


ProjectiveValue = ImplicitNode | ProjectiveConstant
RawPhysicalInput = Tensor | Mapping[str, Tensor] | Sequence[Tensor]


@dataclass
class ImplicitProjectiveDAG:
    root: ImplicitNode
    head: Tensor
    head_binary_exponent: int
    token_count: int
    feature_dimension: int
    selected_token: int
    mask: Tensor
    claim_boundary: str = CLAIM_BOUNDARY
    algorithm1_direct_rq_complete: bool = False
    physical_sources: tuple[PhysicalSourceSpec, ...] = ()
    algorithm1_scale_ledger_complete: bool = False
    algorithm1_certified_head_binary_exponent: int | None = None


@dataclass
class ImplicitBlockOracle:
    network: ImplicitProjectiveDAG
    block: ChiTransformerBlock
    positions: Tensor
    telemetry: MaterializationTelemetry
    norm_buffer_snapshot: tuple[tuple[str, Tensor], ...]


@dataclass(frozen=True)
class CanonicalStepRecord:
    step: int
    uid: int
    label: str
    method: str
    factorization_relative_error: float
    streamed_column_blocks: int
    unfolding_shape: tuple[int, int]
    per_step_function_replay_error: float
    minimum_diagonal_to_maximum_entry: float
    diagonal_condition_proxy: float
    full_row_rank: bool
    expected_parent_occurrences: int
    parent_occurrences_pushed: int
    structurally_reduced: bool = False
    literal_q_chart_resolved: bool = False
    source_core_binary_exponent: int = 0
    factor_binary_exponent: int = 0
    canonical_core_binary_exponent: int = 0
    local_scaled_reconstruction_relative_error: float = 0.0
    local_reconstruction_exponent_delta: int = 0
    maximum_absorption_scaled_relative_error: float = 0.0
    maximum_absorption_exponent_delta: int = 0
    scale_sensitive_occurrences_verified: int = 0
    per_step_function_replay_performed: bool = False
    direct_q_provenance_method: str = ""
    direct_q_provenance_verified: bool = False
    direct_q_provenance_stage_count: int = 0
    direct_q_provenance_column_blocks_compared: int = 0
    direct_q_columns_compared: int = 0
    direct_q_compact_relative_error: float = math.inf
    direct_q_reconstruction_relative_error: float = math.inf
    direct_q_retained_transition_elements: int = 0
    direct_q_minimum_pivot_to_maximum_entry: float = 0.0
    direct_q_conditioning_threshold: float = 0.0
    direct_q_conditioning_accepted: bool = False


@dataclass
class CanonicalImplicitDAG:
    network: ImplicitProjectiveDAG
    steps: tuple[CanonicalStepRecord, ...]
    telemetry: MaterializationTelemetry
    final_projective_replay_error: float = 0.0
    full_projective_replay_evaluations: int = 0


@dataclass(frozen=True)
class CloneStepRecord:
    step: int
    uid: int
    label: str
    clone_occurrences: int
    factor_relative_error: float
    q_core_relative_error: float
    transformed_parent_occurrence_relative_error: float
    shared_function_replay_error: float
    clone_function_replay_error: float
    shared_clone_function_relative_error: float
    literal_rq_occurrences_compared: int
    null_gauge_occurrences_checked_by_reconstruction: int
    supported_reconstruction_relative_error: float
    expected_shared_parent_occurrences: int
    pushed_shared_parent_occurrences: int
    expected_clone_parent_occurrences: int
    pushed_clone_parent_occurrences: int
    boundary_head_relative_error: float


@dataclass
class CanonicalCloneTrace:
    shared_network: ImplicitProjectiveDAG
    explicit_clone_network: ImplicitProjectiveDAG
    records: tuple[CloneStepRecord, ...]
    shared_steps: tuple[CanonicalStepRecord, ...]
    clone_factorization_methods: tuple[str, ...]


@dataclass(frozen=True)
class TinyDenseEquivalence:
    steps: int
    full_row_rank_steps: int
    null_gauge_steps: int
    maximum_factor_relative_error: float
    maximum_q_core_relative_error: float
    maximum_reconstruction_relative_error: float
    maximum_partial_replay_relative_error: float


@dataclass(frozen=True)
class EnvironmentRecord:
    uid: int
    label: str
    mantissa: Tensor
    binary_exponent: int
    incoming_occurrences: int
    child_messages_emitted: int


@dataclass
class DiagonalImplicitDAG:
    network: ImplicitProjectiveDAG
    eigenvalues: tuple[ScaledMatrix, ...]
    replay_relative_error: float
    maximum_recontracted_offdiagonal_ratio: float
    expected_parent_occurrences: int
    pushed_parent_occurrences: int
    algorithm2_child_messages: int = 0
    diagonalized_node_count: int = 0
    eigenvalue_spectra_retained: bool = True


@dataclass
class CloneEVDTrace:
    shared_network: ImplicitProjectiveDAG
    explicit_clone_network: ImplicitProjectiveDAG
    shared_replay_relative_error: float
    clone_replay_relative_error: float
    shared_clone_relative_error: float
    aggregate_environment_relative_error: float
    maximum_shared_aggregate_offdiagonal_ratio: float
    maximum_clone_aggregate_offdiagonal_ratio: float
    expected_shared_parent_occurrences: int
    pushed_shared_parent_occurrences: int
    expected_clone_parent_occurrences: int
    pushed_clone_parent_occurrences: int
    omitted_clone_occurrences: int


@dataclass
class IndependentCloneEVDTrace:
    shared_network: ImplicitProjectiveDAG
    explicit_clone_network: ImplicitProjectiveDAG
    pre_evd_aggregate_environment_relative_error: float
    eigenvalue_relative_error: float
    shared_replay_relative_error: float
    clone_replay_relative_error: float
    shared_clone_relative_error: float
    maximum_shared_aggregate_offdiagonal_ratio: float
    maximum_clone_aggregate_offdiagonal_ratio: float
    expected_shared_parent_occurrences: int
    pushed_shared_parent_occurrences: int
    expected_clone_parent_occurrences: int
    pushed_clone_parent_occurrences: int
    omitted_clone_occurrences: int
    targeted_clone_parent_occurrences: int


def _walk_unique(root: ImplicitNode) -> tuple[ImplicitNode, ...]:
    result: list[ImplicitNode] = []
    state: dict[int, int] = {}

    def visit(node: ImplicitNode) -> None:
        marker = state.get(id(node), 0)
        if marker == 1:
            raise ValueError("implicit projective graph contains a cycle")
        if marker == 2:
            return
        state[id(node)] = 1
        for child in node.children:
            visit(child)
        state[id(node)] = 2
        result.append(node)

    visit(root)
    return tuple(result)


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


def _validate_network(network: ImplicitProjectiveDAG) -> None:
    if network.token_count < 1 or network.feature_dimension < 1:
        raise ValueError("implicit projective dimensions must be positive")
    if not 0 <= network.selected_token < network.token_count:
        raise ValueError("implicit selected token is invalid")
    _validate_real_finite(network.head, "implicit boundary head")
    if isinstance(network.head_binary_exponent, bool) or not isinstance(
        network.head_binary_exponent, int
    ):
        raise ValueError("implicit head exponent must be an integer")
    if not isinstance(network.algorithm1_scale_ledger_complete, bool):
        raise ValueError("Algorithm 1 scale-ledger state must be boolean")
    certified_head_exponent = network.algorithm1_certified_head_binary_exponent
    if certified_head_exponent is not None and (
        isinstance(certified_head_exponent, bool)
        or not isinstance(certified_head_exponent, int)
    ):
        raise ValueError("certified Algorithm 1 head exponent must be an integer")
    if network.head.ndim != 2 or network.head.shape[1] != network.root.output_dimension:
        raise ValueError("implicit boundary head/root dimensions disagree")
    if network.mask.shape != (network.token_count, network.token_count):
        raise ValueError("implicit fixed mask has the wrong shape")
    _validate_real_finite(network.mask, "implicit fixed mask")
    if network.mask.dtype != network.head.dtype or network.mask.device != network.head.device:
        raise ValueError("implicit mask and head must share dtype and device")
    if not bool(((network.mask == 0) | (network.mask == 1)).all()):
        raise ValueError("implicit fixed mask must be binary")
    source_specs: dict[str, PhysicalSourceSpec] = {}
    for spec in network.physical_sources:
        if spec.key in source_specs:
            raise ValueError("heterogeneous physical source keys must be unique")
        source_specs[spec.key] = spec
    nodes = _walk_unique(network.root)
    uid_owner: dict[int, int] = {}
    dtype = network.head.dtype
    device = network.head.device
    for node in nodes:
        owner = uid_owner.setdefault(node.uid, id(node))
        if owner != id(node):
            raise ValueError("distinct shared nodes must have unique uids")
        if isinstance(node.core.binary_exponent, bool) or not isinstance(
            node.core.binary_exponent, int
        ):
            raise ValueError("implicit core exponent must be an integer")
        if isinstance(node.core, UnaryCore):
            if node.core.matrix.ndim != 2:
                raise ValueError("implicit unary core must be a matrix")
            tensors = (node.core.matrix,)
        elif isinstance(node.core, DenseCloneCore):
            if node.core.tensor.ndim not in (2, 3):
                raise ValueError("dense clone core must be order two or three")
            tensors = (node.core.tensor,)
        elif isinstance(node.core, ReducedQRBinaryCore):
            if node.core.q_rows.ndim != 2:
                raise ValueError("reduced-QR binary rows must be a matrix")
            if node.core.left_dimension < 1 or node.core.right_dimension < 1:
                raise ValueError("reduced-QR binary input dimensions must be positive")
            if node.core.q_rows.shape[1] != (
                node.core.left_dimension * node.core.right_dimension
            ):
                raise ValueError("reduced-QR binary rows/input dimensions disagree")
            if node.core.q_rows.numel() > MAXIMUM_RECTANGULAR_Q_ELEMENTS:
                raise ValueError("reduced-QR binary core exceeds its bounded size")
            tensors = (node.core.q_rows,)
        else:
            if (
                node.core.output_factor.ndim != 2
                or node.core.left_factor.ndim != 2
                or node.core.right_factor.ndim != 2
            ):
                raise ValueError("implicit CP factors must be matrices")
            rank = node.core.output_factor.shape[1]
            if rank < 1 or node.core.left_factor.shape[0] != rank or node.core.right_factor.shape[0] != rank:
                raise ValueError("implicit CP factors must have one positive common rank")
            tensors = (
                node.core.output_factor,
                node.core.left_factor,
                node.core.right_factor,
            )
            if node.core.direct_q_provenance is not None:
                _validate_direct_q_provenance(
                    node.core.direct_q_provenance,
                    require_verified=False,
                )
        for value in tensors:
            _validate_real_finite(value, f"implicit core {node.label}")
            if value.dtype != dtype or value.device != device:
                raise ValueError("all implicit cores must share head dtype and device")
        if node.children:
            if node.physical_token is not None or node.physical_source_key is not None:
                raise ValueError("only physical leaves may select a raw source")
            if len(node.children) != _core_arity(node.core):
                raise ValueError("implicit node/core arity mismatch")
            if tuple(child.output_dimension for child in node.children) != node.core.input_dimensions:
                raise ValueError("implicit child/core bond mismatch")
        else:
            if source_specs:
                if node.physical_token is not None or node.physical_source_key not in source_specs:
                    raise ValueError("heterogeneous leaf must select one declared physical source")
                expected_physical_input = source_specs[node.physical_source_key].width + 1
            else:
                if (
                    node.physical_source_key is not None
                    or node.physical_token is None
                    or not 0 <= node.physical_token < network.token_count
                ):
                    raise ValueError("implicit leaf must select one physical token")
                expected_physical_input = network.feature_dimension + 1
            if isinstance(node.core, UnaryCore):
                physical_input = node.core.matrix.shape[1]
            elif isinstance(node.core, DenseCloneCore) and node.core.tensor.ndim == 2:
                physical_input = node.core.tensor.shape[1]
            else:
                physical_input = -1
            if physical_input != expected_physical_input:
                raise ValueError("implicit physical leaf has the wrong homogeneous width")
    if source_specs:
        reached = {node.physical_source_key for node in nodes if not node.children}
        if reached != set(source_specs):
            raise ValueError("heterogeneous source registry must match reachable physical leaves")


def validate_canonical_exponent_normal_form(
    network: ImplicitProjectiveDAG,
) -> dict[str, int | bool]:
    """Validate the scale normal form required by Algorithms 2 and 3.

    Algorithm 1 moves every local binary exponent upward.  Thus every
    canonical core must have exponent zero, while the sole remaining global
    exponent lives at the boundary head and must equal the value sealed by the
    local reconstruction/absorption ledger.
    """

    _validate_network(network)
    if not network.algorithm1_direct_rq_complete:
        raise ValueError(
            "canonical exponent normal form requires completed direct-RQ Algorithm 1"
        )
    if not network.algorithm1_scale_ledger_complete:
        raise ValueError(
            "canonical exponent normal form requires a completed scale ledger"
        )
    nodes = _walk_unique(network.root)
    invalid = tuple(
        (node.uid, node.label, int(node.core.binary_exponent))
        for node in nodes
        if node.core.binary_exponent != 0
    )
    if invalid:
        uid, label, exponent = invalid[0]
        raise ValueError(
            "canonical core exponent is not zero: "
            f"uid={uid} label={label!r} exponent={exponent}"
        )
    certified = network.algorithm1_certified_head_binary_exponent
    if certified is None or network.head_binary_exponent != certified:
        raise ValueError(
            "canonical boundary-head exponent differs from the Algorithm 1 scale ledger"
        )
    cp_certificates = 0
    cp_columns = 0
    streamed_certificates = 0
    streamed_columns = 0
    for node in nodes:
        if not isinstance(node.core, CPBinaryCore):
            continue
        provenance = node.core.direct_q_provenance
        if provenance is None:
            raise ValueError(
                "canonical compact CP Q lacks direct TSQR Q provenance: "
                f"uid={node.uid} label={node.label!r}"
            )
        _validate_direct_q_provenance(provenance, require_verified=True)
        expected_columns = math.prod(node.core.input_dimensions)
        if provenance.columns_compared != expected_columns:
            raise ValueError(
                "canonical compact CP Q provenance does not cover every column"
            )
        cp_certificates += 1
        cp_columns += provenance.columns_compared
        if provenance.method == STREAMED_DIRECT_Q_PROVENANCE_METHOD:
            streamed_certificates += 1
            streamed_columns += provenance.columns_compared
    return {
        "core_count": len(nodes),
        "all_core_exponents_zero": True,
        "head_binary_exponent": int(network.head_binary_exponent),
        "head_exponent_matches_ledger": True,
        "cp_direct_q_certificates": cp_certificates,
        "cp_direct_q_columns": cp_columns,
        "streamed_compact_cp_direct_q_certificates": streamed_certificates,
        "streamed_compact_cp_direct_q_columns": streamed_columns,
        "all_streamed_compact_cp_q_has_direct_q_provenance": True,
    }


def _topological_root_first(root: ImplicitNode) -> tuple[ImplicitNode, ...]:
    return tuple(reversed(_walk_unique(root)))


def _parent_occurrences(
    root: ImplicitNode,
) -> dict[int, list[tuple[ImplicitNode, int]]]:
    parents: dict[int, list[tuple[ImplicitNode, int]]] = defaultdict(list)
    for parent in _walk_unique(root):
        for role, child in enumerate(parent.children):
            parents[id(child)].append((parent, role))
    return parents


def _clone_network(network: ImplicitProjectiveDAG, *, unfold: bool) -> ImplicitProjectiveDAG:
    memo: dict[int, ImplicitNode] = {}
    next_clone_uid = 0

    def clone(node: ImplicitNode) -> ImplicitNode:
        nonlocal next_clone_uid
        if not unfold and id(node) in memo:
            return memo[id(node)]
        instance_uid = next_clone_uid if unfold else node.uid
        if unfold:
            next_clone_uid += 1
        result = ImplicitNode(
            instance_uid,
            node.label,
            _core_clone(node.core),
            tuple(clone(child) for child in node.children),
            node.physical_token,
            node.uid if node.origin_uid is None else node.origin_uid,
            node.physical_source_key,
        )
        if not unfold:
            memo[id(node)] = result
        return result

    return ImplicitProjectiveDAG(
        root=clone(network.root),
        head=network.head.clone(),
        head_binary_exponent=network.head_binary_exponent,
        token_count=network.token_count,
        feature_dimension=network.feature_dimension,
        selected_token=network.selected_token,
        mask=network.mask.clone(),
        claim_boundary=network.claim_boundary,
        algorithm1_direct_rq_complete=network.algorithm1_direct_rq_complete,
        physical_sources=network.physical_sources,
        algorithm1_scale_ledger_complete=network.algorithm1_scale_ledger_complete,
        algorithm1_certified_head_binary_exponent=(
            network.algorithm1_certified_head_binary_exponent
        ),
    )


@torch.no_grad()
def _prepare_physical_input(
    network: ImplicitProjectiveDAG,
    raw_input: RawPhysicalInput,
) -> tuple[int, Tensor | dict[str, Tensor]]:
    if not network.physical_sources:
        if not isinstance(raw_input, Tensor):
            raise TypeError("homogeneous implicit DAG input must be one tensor")
        if raw_input.ndim != 3 or tuple(raw_input.shape[1:]) != (
            network.token_count,
            network.feature_dimension,
        ):
            raise ValueError("raw input shape does not match implicit DAG")
        if raw_input.dtype != network.head.dtype or raw_input.device != network.head.device:
            raise ValueError("raw input must share the implicit DAG dtype and device")
        return int(raw_input.shape[0]), raw_input

    specs = network.physical_sources
    if isinstance(raw_input, Tensor):
        raise TypeError("heterogeneous implicit DAG input must be a mapping or tuple")
    if isinstance(raw_input, Mapping):
        if set(raw_input) != {spec.key for spec in specs}:
            raise ValueError("raw physical-source mapping keys do not match the registry")
        values = {spec.key: raw_input[spec.key] for spec in specs}
    else:
        sequence = tuple(raw_input)
        if len(sequence) != len(specs):
            raise ValueError("raw physical-source tuple length does not match the registry")
        values = {spec.key: value for spec, value in zip(specs, sequence)}
    batch_size: int | None = None
    for spec in specs:
        value = values[spec.key]
        if not isinstance(value, Tensor) or value.ndim != 2:
            raise ValueError("each heterogeneous physical source must be a batch matrix")
        if value.shape[1] != spec.width:
            raise ValueError(f"physical source {spec.key!r} has the wrong width")
        if value.dtype != network.head.dtype or value.device != network.head.device:
            raise ValueError("physical sources must share the implicit DAG dtype and device")
        if batch_size is None:
            batch_size = int(value.shape[0])
        elif value.shape[0] != batch_size:
            raise ValueError("heterogeneous physical sources have different batch sizes")
    if batch_size is None or batch_size < 1:
        raise ValueError("heterogeneous physical input must contain a nonempty batch")
    return batch_size, values


@torch.no_grad()
def evaluate_scaled_boundary(
    network: ImplicitProjectiveDAG,
    raw_input: RawPhysicalInput,
) -> ScaledBatch:
    _validate_network(network)
    batch_size, prepared = _prepare_physical_input(network, raw_input)
    memo: dict[int, ScaledBatch] = {}

    def evaluate(node: ImplicitNode) -> ScaledBatch:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        if node.children:
            children = tuple(evaluate(child) for child in node.children)
            value = _core_apply(node.core, tuple(child.mantissa for child in children))
            exponent = torch.stack(tuple(child.binary_exponent for child in children)).sum(0)
        else:
            if node.physical_source_key is not None:
                if not isinstance(prepared, dict):
                    raise ValueError("heterogeneous leaf received homogeneous input")
                raw_value = prepared[node.physical_source_key]
            else:
                if node.physical_token is None or not isinstance(prepared, Tensor):
                    raise ValueError("implicit leaf has no physical token")
                raw_value = prepared[:, node.physical_token]
            physical = torch.cat(
                (
                    raw_value,
                    raw_value.new_ones(batch_size, 1),
                ),
                dim=1,
            )
            value = _core_apply(node.core, (physical,))
            exponent = torch.zeros(
                batch_size, dtype=torch.int64, device=network.head.device
            )
        exponent = exponent + node.core.binary_exponent
        result = _normalize_batch(value, exponent)
        memo[id(node)] = result
        return result

    root = evaluate(network.root)
    output = root.mantissa @ network.head.T
    return _normalize_batch(output, root.binary_exponent + network.head_binary_exponent)


@torch.no_grad()
def evaluate_projective_boundary(
    network: ImplicitProjectiveDAG,
    raw_input: RawPhysicalInput,
) -> Tensor:
    """Evaluate bounded homogeneous coordinates, discarding common scales.

    Every primitive is homogeneous in each input.  Consequently each child,
    core, and head binary exponent contributes only one common scalar to its
    downstream projective value and cannot affect the quotient.  Omitting that
    scalar ledger prevents the exponentially growing polynomial degree of a
    deep policy from overflowing a fixed-width sample exponent while retaining
    the exact rational function.
    """

    _validate_network(network)
    batch_size, prepared = _prepare_physical_input(network, raw_input)
    memo: dict[int, Tensor] = {}

    def evaluate(node: ImplicitNode) -> Tensor:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        if node.children:
            children = tuple(evaluate(child) for child in node.children)
            value = _core_apply(node.core, children)
        else:
            if node.physical_source_key is not None:
                if not isinstance(prepared, dict):
                    raise ValueError("heterogeneous leaf received homogeneous input")
                raw_value = prepared[node.physical_source_key]
            else:
                if node.physical_token is None or not isinstance(prepared, Tensor):
                    raise ValueError("implicit leaf has no physical token")
                raw_value = prepared[:, node.physical_token]
            physical = torch.cat(
                (raw_value, raw_value.new_ones(batch_size, 1)), dim=1
            )
            value = _core_apply(node.core, (physical,))
        result = _normalize_projective_batch(value)
        memo[id(node)] = result
        return result

    root = evaluate(network.root)
    return _normalize_projective_batch(root @ network.head.T)


@torch.no_grad()
def evaluate_boundary_quotient(
    network: ImplicitProjectiveDAG,
    raw_input: RawPhysicalInput,
) -> Tensor:
    pair = evaluate_projective_boundary(network, raw_input)
    denominator = pair[:, -1]
    relative = denominator.abs() / pair.abs().amax(dim=1).clamp_min(
        torch.finfo(pair.dtype).tiny
    )
    if float(relative.min().item()) <= 100.0 * torch.finfo(pair.dtype).eps:
        raise ValueError("implicit projective denominator is numerically zero")
    return pair[:, :-1] / denominator[:, None]


def _push_factor_to_parents_with_scale_ledger(
    node: ImplicitNode,
    factor: ScaledMatrix,
    parents: dict[int, list[tuple[ImplicitNode, int]]],
    network: ImplicitProjectiveDAG,
    *,
    omit_occurrence: int | None = None,
) -> AbsorptionScaleLedger:
    occurrences = parents.get(id(node), [])
    if omit_occurrence is not None and not 0 <= omit_occurrence < len(occurrences):
        raise ValueError("omitted R occurrence index is out of range")
    pushed = 0
    verified = 0
    maximum_error = 0.0
    maximum_exponent_delta = 0
    for occurrence, (parent, role) in enumerate(occurrences):
        if occurrence == omit_occurrence:
            if factor.mantissa.shape[0] != factor.mantissa.shape[1]:
                raise ValueError("omitted R occurrence requires a square factor")
            continue
        before = parent.core
        after = _absorb_input_factor(before, role, factor)
        error, exponent_delta = _absorption_scale_audit(
            before, after, role, factor
        )
        if (
            not math.isfinite(error)
            or error > LOCAL_SCALE_CERTIFICATION_TOLERANCE
            or exponent_delta != 0
        ):
            raise RuntimeError(
                "direct-RQ parent absorption failed its scale-sensitive local ledger"
            )
        parent.core = after
        pushed += 1
        verified += 1
        maximum_error = max(maximum_error, error)
        maximum_exponent_delta = max(
            maximum_exponent_delta, abs(int(exponent_delta))
        )
    if not occurrences:
        if node is not network.root:
            raise ValueError("non-root implicit node has no parent occurrence")
        if network.head.shape[1] != factor.mantissa.shape[0]:
            raise ValueError("root factor does not match boundary head")
        before_head = ScaledMatrix(
            network.head, int(network.head_binary_exponent)
        )
        head = network.head @ factor.mantissa
        head, stripped = _strip_power_of_two(head)
        after_head = ScaledMatrix(
            head,
            int(
                network.head_binary_exponent
                + factor.binary_exponent
                + stripped
            ),
        )
        expected_head = ScaledMatrix(
            before_head.mantissa @ factor.mantissa,
            int(before_head.binary_exponent + factor.binary_exponent),
        )
        error, exponent_delta = _scaled_matrix_audit(after_head, expected_head)
        if (
            not math.isfinite(error)
            or error > LOCAL_SCALE_CERTIFICATION_TOLERANCE
            or exponent_delta != 0
        ):
            raise RuntimeError(
                "direct-RQ root absorption failed its scale-sensitive local ledger"
            )
        network.head = after_head.mantissa
        network.head_binary_exponent = int(after_head.binary_exponent)
        return AbsorptionScaleLedger(1, 1, error, abs(int(exponent_delta)))
    return AbsorptionScaleLedger(
        pushed,
        verified,
        maximum_error,
        maximum_exponent_delta,
    )


def _push_factor_to_parents(
    node: ImplicitNode,
    factor: ScaledMatrix,
    parents: dict[int, list[tuple[ImplicitNode, int]]],
    network: ImplicitProjectiveDAG,
    *,
    omit_occurrence: int | None = None,
) -> int:
    return _push_factor_to_parents_with_scale_ledger(
        node,
        factor,
        parents,
        network,
        omit_occurrence=omit_occurrence,
    ).pushed_occurrences


@torch.no_grad()
def canonicalize_implicit_dag_direct_rq(
    network: ImplicitProjectiveDAG,
    *,
    block_size: int = 4096,
    replay_inputs: RawPhysicalInput | None = None,
    telemetry: MaterializationTelemetry | None = None,
    omit_parent_push: tuple[str, int] | None = None,
    replay_each_step: bool = True,
    copy_network: bool = True,
    step_callback: Callable[
        [ImplicitProjectiveDAG, ImplicitNode, ScaledMatrix, CanonicalStepRecord],
        None,
    ]
    | None = None,
) -> CanonicalImplicitDAG:
    """Run Algorithm 1 with local scale certification on every step.

    ``replay_each_step`` remains the default for bounded tests and the explicit
    clone oracle.  Production sets it false, relying on the non-vacuous local
    reconstruction and absorption ledgers at every step followed by one full
    projective replay after the sweep.  ``copy_network=False`` avoids a second
    full set of core tensors; a failed in-place sweep leaves its input marked
    uncertified and must not be reused.
    """

    _validate_network(network)
    work = _clone_network(network, unfold=False) if copy_network else network
    work.algorithm1_direct_rq_complete = False
    work.algorithm1_scale_ledger_complete = False
    work.algorithm1_certified_head_binary_exponent = None
    meter = MaterializationTelemetry() if telemetry is None else telemetry
    parents = _parent_occurrences(work.root)
    reference = (
        None
        if replay_inputs is None
        else evaluate_projective_boundary(work, replay_inputs)
    )
    full_replay_evaluations = int(reference is not None)
    records: list[CanonicalStepRecord] = []
    omitted = False
    for step, node in enumerate(_walk_unique(work.root)):
        source_core = node.core
        try:
            factor, canonical_core, diagnostics = _direct_rq_core(
                source_core, meter, block_size=block_size
            )
        except Exception as error:
            rows = source_core.output_dimension
            columns = math.prod(source_core.input_dimensions)
            raise RuntimeError(
                "Algorithm 1 direct-RQ step failed: "
                f"step={step} uid={node.uid} label={node.label!r} "
                f"kind={source_core.kind!r} unfolding={rows}x{columns}"
            ) from error
        provenance = diagnostics.direct_q_provenance
        if provenance is None:
            raise RuntimeError("direct-RQ step did not return direct-Q provenance")
        _validate_direct_q_provenance(provenance, require_verified=True)
        expected_q_columns = math.prod(source_core.input_dimensions)
        if provenance.columns_compared != expected_q_columns:
            raise RuntimeError(
                "direct-Q provenance did not cover every local unfolding column"
            )
        if isinstance(canonical_core, CPBinaryCore) and (
            canonical_core.direct_q_provenance != provenance
        ):
            raise RuntimeError(
                "compact canonical CP core lost its direct-Q provenance"
            )
        local_reconstruction_error, local_exponent_delta = (
            _local_direct_rq_scaled_reconstruction_audit(
                source_core,
                factor,
                canonical_core,
                block_size=max(1, block_size),
                telemetry=meter,
                factor_normalization_binary_exponent=(
                    diagnostics.factor_normalization_binary_exponent
                ),
            )
        )
        if (
            not math.isfinite(local_reconstruction_error)
            or local_reconstruction_error > LOCAL_SCALE_CERTIFICATION_TOLERANCE
            or local_exponent_delta != 0
        ):
            raise RuntimeError(
                "direct-RQ local scaled reconstruction failed before parent absorption"
            )
        node.core = canonical_core
        omit_index = None
        if (
            omit_parent_push is not None
            and not omitted
            and node.label == omit_parent_push[0]
        ):
            omit_index = omit_parent_push[1]
            omitted = True
        absorption = _push_factor_to_parents_with_scale_ledger(
            node,
            factor,
            parents,
            work,
            omit_occurrence=omit_index,
        )
        pushed = absorption.pushed_occurrences
        if absorption.verified_occurrences != pushed:
            raise RuntimeError(
                "direct-RQ absorption ledger did not verify every pushed occurrence"
            )
        expected_pushes = len(parents.get(id(node), [])) or 1
        if omit_index is not None and pushed != expected_pushes - 1:
            raise RuntimeError("omitted R control did not remove exactly one occurrence")
        replay_error = 0.0
        replay_performed = reference is not None and replay_each_step
        if replay_performed:
            replay_error = _projective_batch_relative_error(
                evaluate_projective_boundary(work, replay_inputs), reference
            )
            full_replay_evaluations += 1
        record = CanonicalStepRecord(
                step=step,
                uid=node.uid,
                label=node.label,
                method=diagnostics.method,
                factorization_relative_error=(
                    diagnostics.factorization_relative_error
                ),
                streamed_column_blocks=diagnostics.streamed_column_blocks,
                unfolding_shape=diagnostics.unfolding_shape,
                per_step_function_replay_error=replay_error,
                minimum_diagonal_to_maximum_entry=(
                    diagnostics.minimum_diagonal_to_maximum_entry
                ),
                diagonal_condition_proxy=diagnostics.diagonal_condition_proxy,
                full_row_rank=diagnostics.full_row_rank,
                expected_parent_occurrences=expected_pushes,
                parent_occurrences_pushed=pushed,
                structurally_reduced=diagnostics.structurally_reduced,
                literal_q_chart_resolved=diagnostics.literal_q_chart_resolved,
                source_core_binary_exponent=int(source_core.binary_exponent),
                factor_binary_exponent=int(factor.binary_exponent),
                canonical_core_binary_exponent=int(
                    canonical_core.binary_exponent
                ),
                local_scaled_reconstruction_relative_error=(
                    local_reconstruction_error
                ),
                local_reconstruction_exponent_delta=int(local_exponent_delta),
                maximum_absorption_scaled_relative_error=(
                    absorption.maximum_scaled_relative_error
                ),
                maximum_absorption_exponent_delta=int(
                    absorption.maximum_exponent_delta
                ),
                scale_sensitive_occurrences_verified=(
                    absorption.verified_occurrences
                ),
                per_step_function_replay_performed=replay_performed,
                direct_q_provenance_method=provenance.method,
                direct_q_provenance_verified=provenance.verified,
                direct_q_provenance_stage_count=provenance.stage_count,
                direct_q_provenance_column_blocks_compared=(
                    provenance.column_blocks_compared
                ),
                direct_q_columns_compared=provenance.columns_compared,
                direct_q_compact_relative_error=(
                    provenance.compact_q_relative_error
                ),
                direct_q_reconstruction_relative_error=(
                    provenance.direct_q_reconstruction_relative_error
                ),
                direct_q_retained_transition_elements=(
                    provenance.retained_transition_elements
                ),
                direct_q_minimum_pivot_to_maximum_entry=(
                    provenance.minimum_pivot_to_maximum_entry
                ),
                direct_q_conditioning_threshold=(
                    provenance.conditioning_threshold
                ),
                direct_q_conditioning_accepted=(
                    provenance.conditioning_accepted
                ),
            )
        records.append(record)
        if step_callback is not None:
            step_callback(work, node, factor, record)
    if omit_parent_push is not None and not omitted:
        raise ValueError("requested omitted parent R occurrence was not found")
    complete = omit_parent_push is None
    work.algorithm1_direct_rq_complete = complete
    work.algorithm1_scale_ledger_complete = complete
    work.algorithm1_certified_head_binary_exponent = (
        int(work.head_binary_exponent) if complete else None
    )
    if complete:
        validate_canonical_exponent_normal_form(work)
    final_replay_error = 0.0
    if reference is not None:
        if replay_each_step:
            final_replay_error = records[-1].per_step_function_replay_error
        else:
            final_replay_error = _projective_batch_relative_error(
                evaluate_projective_boundary(work, replay_inputs), reference
            )
            full_replay_evaluations += 1
    return CanonicalImplicitDAG(
        work,
        tuple(records),
        meter,
        final_replay_error,
        full_replay_evaluations,
    )


def _tiny_core_mantissa(core: ImplicitCore, maximum_elements: int = 100_000) -> Tensor:
    elements = core.output_dimension * math.prod(core.input_dimensions)
    if elements > maximum_elements:
        raise ValueError("step-trace core comparison is restricted to the tiny oracle")
    if isinstance(core, UnaryCore):
        return core.matrix
    if isinstance(core, DenseCloneCore):
        return core.tensor
    if isinstance(core, ReducedQRBinaryCore):
        return core.q_rows.reshape(
            core.output_dimension, core.left_dimension, core.right_dimension
        )
    return torch.einsum(
        "ot,ti,tj->oij",
        core.output_factor,
        core.left_factor,
        core.right_factor,
    )


def _tiny_core_relative_error(actual: ImplicitCore, expected: ImplicitCore) -> float:
    left = _tiny_core_mantissa(actual).reshape(1, -1)
    right = _tiny_core_mantissa(expected).reshape(1, -1)
    return _scaled_matrix_relative_error(
        ScaledMatrix(left, actual.binary_exponent),
        ScaledMatrix(right, expected.binary_exponent),
    )


@torch.no_grad()
def canonicalize_with_explicit_clone_step_trace(
    network: ImplicitProjectiveDAG,
    replay_inputs: RawPhysicalInput,
    *,
    block_size: int = 512,
) -> CanonicalCloneTrace:
    """Compare shared Algorithm 1 with a no-memo clone after every RQ step.

    The explicit graph duplicates every syntactic occurrence.  For each shared
    node in bottom-up order, every corresponding clone occurrence is factored
    independently.  The gate compares positive-diagonal R factors, Q cores,
    every transformed parent occurrence, and full function replay before the
    next shared node is touched.
    """

    _validate_network(network)
    shared = _clone_network(network, unfold=False)
    clone = _clone_network(network, unfold=True)
    shared_parents = _parent_occurrences(shared.root)
    clone_parents = _parent_occurrences(clone.root)
    shared_schedule = _walk_unique(shared.root)
    clone_nodes = _walk_unique(clone.root)
    clones_by_origin: dict[int, list[ImplicitNode]] = defaultdict(list)
    for node in clone_nodes:
        if node.origin_uid is None:
            raise ValueError("explicit clone node lost its origin uid")
        clones_by_origin[node.origin_uid].append(node)
    reference = evaluate_projective_boundary(network, replay_inputs)
    shared_meter = MaterializationTelemetry()
    clone_meter = MaterializationTelemetry()
    records: list[CloneStepRecord] = []
    shared_steps: list[CanonicalStepRecord] = []
    clone_methods: list[str] = []

    for step, shared_node in enumerate(shared_schedule):
        shared_source_core = shared_node.core
        factor, q_core, diagnostics = _direct_rq_core(
            shared_source_core, shared_meter, block_size=block_size
        )
        provenance = diagnostics.direct_q_provenance
        if provenance is None:
            raise RuntimeError("shared clone trace lost direct-Q provenance")
        _validate_direct_q_provenance(provenance, require_verified=True)
        local_reconstruction_error, local_exponent_delta = (
            _local_direct_rq_scaled_reconstruction_audit(
                shared_source_core,
                factor,
                q_core,
                block_size=max(1, block_size),
                telemetry=shared_meter,
                factor_normalization_binary_exponent=(
                    diagnostics.factor_normalization_binary_exponent
                ),
            )
        )
        if (
            local_reconstruction_error > LOCAL_SCALE_CERTIFICATION_TOLERANCE
            or local_exponent_delta != 0
        ):
            raise RuntimeError("shared clone trace failed local scale reconstruction")
        shared_node.core = q_core
        shared_absorption = _push_factor_to_parents_with_scale_ledger(
            shared_node, factor, shared_parents, shared
        )
        shared_pushed = shared_absorption.pushed_occurrences
        candidates = clones_by_origin.get(shared_node.uid, [])
        if not candidates:
            raise ValueError("explicit clone trace missed a shared origin")
        factor_error = 0.0
        q_error = 0.0
        literal_compared = 0
        null_gauge_checked = 0
        supported_reconstruction_error = diagnostics.factorization_relative_error
        clone_pushed = 0
        for candidate in candidates:
            clone_source_core = candidate.core
            clone_factor, clone_q, clone_diagnostics = _independent_dense_clone_rq(
                clone_source_core, clone_meter
            )
            clone_local_error, clone_exponent_delta = (
                _local_direct_rq_scaled_reconstruction_audit(
                clone_source_core,
                clone_factor,
                clone_q,
                block_size=max(1, block_size),
                telemetry=clone_meter,
                factor_normalization_binary_exponent=(
                    clone_diagnostics.factor_normalization_binary_exponent
                ),
                )
            )
            if (
                clone_local_error > LOCAL_SCALE_CERTIFICATION_TOLERANCE
                or clone_exponent_delta != 0
            ):
                raise RuntimeError("explicit clone trace failed local scale reconstruction")
            candidate.core = clone_q
            clone_pushed += _push_factor_to_parents(
                candidate, clone_factor, clone_parents, clone
            )
            clone_methods.append(clone_diagnostics.method)
            supported_reconstruction_error = max(
                supported_reconstruction_error,
                clone_diagnostics.factorization_relative_error,
            )
            if (
                diagnostics.literal_q_chart_resolved
                and clone_diagnostics.literal_q_chart_resolved
            ):
                literal_compared += 1
                factor_error = max(
                    factor_error,
                    _scaled_matrix_relative_error(clone_factor, factor),
                )
                q_error = max(q_error, _tiny_core_relative_error(clone_q, q_core))
            else:
                # A null-row Q completion is not gauge-identifiable.  Its exact
                # M=RQ reconstruction, parent push, and full replay remain gated.
                null_gauge_checked += 1

        parent_error = 0.0
        shared_parent_nodes = {parent.uid: parent for parent, _ in shared_parents.get(id(shared_node), [])}
        for parent_uid, shared_parent in shared_parent_nodes.items():
            for clone_parent in clones_by_origin.get(parent_uid, []):
                parent_error = max(
                    parent_error,
                    _tiny_core_relative_error(clone_parent.core, shared_parent.core),
                )

        shared_value = evaluate_projective_boundary(shared, replay_inputs)
        clone_value = evaluate_projective_boundary(clone, replay_inputs)
        shared_replay = _projective_batch_relative_error(shared_value, reference)
        clone_replay = _projective_batch_relative_error(clone_value, reference)
        shared_clone = _projective_batch_relative_error(shared_value, clone_value)
        head_error = _scaled_matrix_relative_error(
            ScaledMatrix(shared.head, shared.head_binary_exponent),
            ScaledMatrix(clone.head, clone.head_binary_exponent),
        )
        expected_shared_pushes = len(shared_parents.get(id(shared_node), [])) or 1
        expected_clone_pushes = sum(
            len(clone_parents.get(id(candidate), [])) or 1 for candidate in candidates
        )
        records.append(
            CloneStepRecord(
                step,
                shared_node.uid,
                shared_node.label,
                len(candidates),
                factor_error,
                q_error,
                parent_error,
                shared_replay,
                clone_replay,
                shared_clone,
                literal_compared,
                null_gauge_checked,
                supported_reconstruction_error,
                expected_shared_pushes,
                shared_pushed,
                expected_clone_pushes,
                clone_pushed,
                head_error,
            )
        )
        shared_steps.append(
            CanonicalStepRecord(
                step=step,
                uid=shared_node.uid,
                label=shared_node.label,
                method=diagnostics.method,
                factorization_relative_error=(
                    diagnostics.factorization_relative_error
                ),
                streamed_column_blocks=diagnostics.streamed_column_blocks,
                unfolding_shape=diagnostics.unfolding_shape,
                per_step_function_replay_error=shared_replay,
                minimum_diagonal_to_maximum_entry=(
                    diagnostics.minimum_diagonal_to_maximum_entry
                ),
                diagonal_condition_proxy=diagnostics.diagonal_condition_proxy,
                full_row_rank=diagnostics.full_row_rank,
                expected_parent_occurrences=expected_shared_pushes,
                parent_occurrences_pushed=shared_pushed,
                structurally_reduced=diagnostics.structurally_reduced,
                literal_q_chart_resolved=diagnostics.literal_q_chart_resolved,
                source_core_binary_exponent=int(
                    shared_source_core.binary_exponent
                ),
                factor_binary_exponent=int(factor.binary_exponent),
                canonical_core_binary_exponent=int(q_core.binary_exponent),
                local_scaled_reconstruction_relative_error=(
                    local_reconstruction_error
                ),
                local_reconstruction_exponent_delta=int(local_exponent_delta),
                maximum_absorption_scaled_relative_error=(
                    shared_absorption.maximum_scaled_relative_error
                ),
                maximum_absorption_exponent_delta=int(
                    shared_absorption.maximum_exponent_delta
                ),
                scale_sensitive_occurrences_verified=(
                    shared_absorption.verified_occurrences
                ),
                per_step_function_replay_performed=True,
                direct_q_provenance_method=provenance.method,
                direct_q_provenance_verified=provenance.verified,
                direct_q_provenance_stage_count=provenance.stage_count,
                direct_q_provenance_column_blocks_compared=(
                    provenance.column_blocks_compared
                ),
                direct_q_columns_compared=provenance.columns_compared,
                direct_q_compact_relative_error=(
                    provenance.compact_q_relative_error
                ),
                direct_q_reconstruction_relative_error=(
                    provenance.direct_q_reconstruction_relative_error
                ),
                direct_q_retained_transition_elements=(
                    provenance.retained_transition_elements
                ),
                direct_q_minimum_pivot_to_maximum_entry=(
                    provenance.minimum_pivot_to_maximum_entry
                ),
                direct_q_conditioning_threshold=(
                    provenance.conditioning_threshold
                ),
                direct_q_conditioning_accepted=(
                    provenance.conditioning_accepted
                ),
            )
        )

    shared.algorithm1_direct_rq_complete = True
    shared.algorithm1_scale_ledger_complete = True
    shared.algorithm1_certified_head_binary_exponent = int(
        shared.head_binary_exponent
    )
    clone.algorithm1_direct_rq_complete = True
    clone.algorithm1_scale_ledger_complete = True
    clone.algorithm1_certified_head_binary_exponent = int(
        clone.head_binary_exponent
    )
    validate_canonical_exponent_normal_form(shared)
    validate_canonical_exponent_normal_form(clone)
    return CanonicalCloneTrace(
        shared,
        clone,
        tuple(records),
        tuple(shared_steps),
        tuple(clone_methods),
    )


@torch.no_grad()
def tiny_dense_direct_rq_equivalence(
    network: ImplicitProjectiveDAG,
    replay_inputs: RawPhysicalInput,
    *,
    block_size: int = 512,
) -> TinyDenseEquivalence:
    """Compare streamed Algorithm 1 to independent dense QR at each tiny step."""

    _validate_network(network)
    work = _clone_network(network, unfold=False)
    parents = _parent_occurrences(work.root)
    reference = evaluate_projective_boundary(work, replay_inputs)
    streamed_meter = MaterializationTelemetry()
    dense_meter = MaterializationTelemetry()
    factor_error = 0.0
    q_error = 0.0
    reconstruction_error = 0.0
    replay_error = 0.0
    full_steps = 0
    null_steps = 0
    for node in _walk_unique(work.root):
        original = _core_clone(node.core)
        factor, canonical, diagnostics = _direct_rq_core(
            original, streamed_meter, block_size=block_size
        )
        dense_factor, dense_q, dense_diagnostics = _independent_dense_clone_rq(
            original, dense_meter
        )
        reconstruction_error = max(
            reconstruction_error,
            diagnostics.factorization_relative_error,
            dense_diagnostics.factorization_relative_error,
        )
        if (
            diagnostics.literal_q_chart_resolved
            and dense_diagnostics.literal_q_chart_resolved
        ):
            full_steps += 1
            factor_error = max(
                factor_error,
                _scaled_matrix_relative_error(factor, dense_factor),
            )
            q_error = max(q_error, _tiny_core_relative_error(canonical, dense_q))
        else:
            null_steps += 1
        node.core = canonical
        _push_factor_to_parents(node, factor, parents, work)
        replay_error = max(
            replay_error,
            _projective_batch_relative_error(
                evaluate_projective_boundary(work, replay_inputs), reference
            ),
        )
    return TinyDenseEquivalence(
        len(_walk_unique(work.root)),
        full_steps,
        null_steps,
        factor_error,
        q_error,
        reconstruction_error,
        replay_error,
    )


def _scale_symmetric(value: Tensor, exponent: int = 0) -> ScaledMatrix:
    symmetric = 0.5 * (value + value.T)
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


@torch.no_grad()
def reverse_implicit_environments(
    network: ImplicitProjectiveDAG,
    *,
    output_metric: Tensor | None = None,
    telemetry: MaterializationTelemetry | None = None,
    retain_records: bool = True,
    record_callback: Callable[[EnvironmentRecord], None] | None = None,
) -> tuple[EnvironmentRecord, ...]:
    validate_canonical_exponent_normal_form(network)
    meter = MaterializationTelemetry() if telemetry is None else telemetry
    identity_output_metric = torch.eye(
        network.head.shape[0], dtype=network.head.dtype, device=network.head.device
    )
    if output_metric is None:
        output_metric = identity_output_metric
    _validate_real_finite(output_metric, "implicit output metric")
    if output_metric.shape != (network.head.shape[0], network.head.shape[0]):
        raise ValueError("implicit output metric has the wrong shape")
    # The decisive direct-only path fixes the observable coefficient metric to
    # identity.  This keeps Algorithm 2 entirely in explicit TN contractions;
    # the first eigendecomposition occurs only in Algorithm 3.
    if not torch.equal(output_metric, identity_output_metric):
        raise ValueError("implicit direct-only Algorithm 2 requires the identity output metric")

    root = network.head.T @ output_metric @ network.head
    pending: dict[int, ScaledMatrix] = {
        id(network.root): _scale_symmetric(
            root, 2 * network.head_binary_exponent
        )
    }
    incoming = {id(node): 0 for node in _walk_unique(network.root)}
    for parent in _walk_unique(network.root):
        for child in parent.children:
            incoming[id(child)] += 1
    records: list[EnvironmentRecord] = []
    emitted = 0
    for node in _topological_root_first(network.root):
        aggregate = pending.pop(id(node), None)
        if aggregate is None:
            raise ValueError("implicit reverse traversal found an unreachable node")
        child_messages_emitted = 0
        for role, child in enumerate(node.children):
            contracted = _role_environment(node.core, aggregate.mantissa, role, meter)
            _accumulate_scaled_message(
                pending,
                id(child),
                _scale_symmetric(
                    contracted,
                    aggregate.binary_exponent + 2 * node.core.binary_exponent,
                ),
            )
            child_messages_emitted += 1
        record = EnvironmentRecord(
                node.uid,
                node.label,
                aggregate.mantissa,
                aggregate.binary_exponent,
                incoming[id(node)],
                child_messages_emitted,
            )
        emitted += child_messages_emitted
        if record_callback is not None:
            record_callback(record)
        if retain_records:
            records.append(record)
    expected = sum(len(node.children) for node in _walk_unique(network.root))
    if emitted != expected:
        raise RuntimeError(
            f"Algorithm 2 emitted {emitted} child messages for {expected} edge occurrences"
        )
    return tuple(records)


@torch.no_grad()
def compare_shared_and_explicit_clone_environments(
    shared_network: ImplicitProjectiveDAG,
    explicit_clone_network: ImplicitProjectiveDAG,
    *,
    output_metric: Tensor | None = None,
) -> dict[str, float | int]:
    """Aggregate no-memo clone messages and compare Algorithm 2 exactly."""

    shared = reverse_implicit_environments(shared_network, output_metric=output_metric)
    clone = reverse_implicit_environments(explicit_clone_network, output_metric=output_metric)
    clone_origin = {
        node.uid: node.origin_uid for node in _walk_unique(explicit_clone_network.root)
    }
    grouped: dict[int, list[ScaledMatrix]] = defaultdict(list)
    for record in clone:
        origin = clone_origin.get(record.uid)
        if origin is None:
            raise ValueError("clone environment record has no origin")
        grouped[origin].append(ScaledMatrix(record.mantissa, record.binary_exponent))
    shared_by_uid = {
        record.uid: ScaledMatrix(record.mantissa, record.binary_exponent)
        for record in shared
    }
    if set(grouped) != set(shared_by_uid):
        raise ValueError("shared and clone environment origins disagree")
    error = 0.0
    for uid, messages in grouped.items():
        aggregate = messages[0] if len(messages) == 1 else _add_scaled_matrices(messages)
        error = max(error, _scaled_matrix_relative_error(aggregate, shared_by_uid[uid]))
    return {
        "maximum_aggregate_relative_error": error,
        "shared_unique_environment_count": len(shared),
        "explicit_clone_environment_count": len(clone),
        "origins_compared": len(grouped),
    }


def _offdiagonal_ratio(value: Tensor) -> float:
    diagonal = torch.diag(torch.diagonal(value))
    return float(
        ((value - diagonal).norm() / value.norm().clamp_min(torch.finfo(value.dtype).tiny)).item()
    )


@torch.no_grad()
def diagonalize_implicit_dag_full_rank(
    network: ImplicitProjectiveDAG,
    *,
    output_metric: Tensor | None = None,
    replay_inputs: RawPhysicalInput | None = None,
    telemetry: MaterializationTelemetry | None = None,
    copy_network: bool = True,
    precomputed_environments: Sequence[EnvironmentRecord] | None = None,
    stream_pre_evd_environments: bool = False,
    retain_eigenvalues: bool = True,
    retain_post_evd_environments: bool = True,
    compact_spectrum_callback: Callable[[CompactSpectrumRecord], None] | None = None,
) -> DiagonalImplicitDAG:
    """Algorithm 3 common-basis rotations on explicitly contracted environments.

    The production mode streams Algorithm 2 in root-first order.  It first
    emits every child message from the untouched canonical core, then
    decomposes that completed environment and immediately gauges the node and
    its already-processed parents.  Orthogonal gauges cancel from every child
    message already emitted, so no all-node basis table is retained.  Bounded
    tests retain the separated environment and spectrum records by default.
    """

    validate_canonical_exponent_normal_form(network)
    if compact_spectrum_callback is not None and retain_eigenvalues:
        raise ValueError(
            "compact spectrum streaming requires full-spectrum retention to be disabled"
        )
    if precomputed_environments is not None and stream_pre_evd_environments:
        raise ValueError(
            "precomputed and streamed pre-EVD environments are mutually exclusive"
        )
    work = _clone_network(network, unfold=False) if copy_network else network
    if compact_spectrum_callback is not None:
        bind_network = getattr(compact_spectrum_callback, "bind_network", None)
        if bind_network is not None:
            bind_network(work)
    meter = MaterializationTelemetry() if telemetry is None else telemetry
    reference = (
        None
        if replay_inputs is None
        else evaluate_projective_boundary(work, replay_inputs)
    )
    nodes_bottom_up = _walk_unique(work.root)
    parents = _parent_occurrences(work.root)
    bases: dict[int, Tensor] = {}
    eigenvalue_records: list[ScaledMatrix] = []
    algorithm2_child_messages = 0
    diagonalized_node_count = 0
    expected_parent_occurrences = 0
    pushed_parent_occurrences = 0

    if stream_pre_evd_environments:
        identity_output_metric = torch.eye(
            work.head.shape[0], dtype=work.head.dtype, device=work.head.device
        )
        metric = identity_output_metric if output_metric is None else output_metric
        _validate_real_finite(metric, "implicit output metric")
        if metric.shape != (work.head.shape[0], work.head.shape[0]):
            raise ValueError("implicit output metric has the wrong shape")
        if not torch.equal(metric, identity_output_metric):
            raise ValueError(
                "implicit direct-only Algorithm 2 requires the identity output metric"
            )
        root_environment = work.head.T @ metric @ work.head
        pending: dict[int, ScaledMatrix] = {
            id(work.root): _scale_symmetric(
                root_environment, 2 * int(work.head_binary_exponent)
            )
        }
        for node in _topological_root_first(work.root):
            aggregate = pending.pop(id(node), None)
            if aggregate is None:
                raise ValueError(
                    "streamed Algorithm 2 found an unreachable canonical node"
                )
            for role, child in enumerate(node.children):
                contracted = _role_environment(
                    node.core, aggregate.mantissa, role, meter
                )
                _accumulate_scaled_message(
                    pending,
                    id(child),
                    _scale_symmetric(
                        contracted,
                        aggregate.binary_exponent
                        + 2 * int(node.core.binary_exponent),
                    ),
                )
                algorithm2_child_messages += 1
            symmetric = 0.5 * (aggregate.mantissa + aggregate.mantissa.T)
            values, vectors = torch.linalg.eigh(symmetric)
            order = torch.argsort(values, descending=True)
            values = values[order]
            vectors = vectors[:, order]
            if compact_spectrum_callback is not None:
                compact_spectrum_callback(
                    _compact_spectrum_record(
                        node.uid,
                        node.label,
                        values,
                        aggregate.binary_exponent,
                    )
                )
            if retain_eigenvalues:
                eigenvalue_records.append(
                    ScaledMatrix(torch.diag(values), aggregate.binary_exponent)
                )
            node.core = _apply_output_basis(node.core, vectors.T)
            basis = ScaledMatrix(vectors, 0)
            occurrences = parents.get(id(node), [])
            expected_parent_occurrences += (
                len(occurrences) if occurrences else 1
            )
            for parent, role in occurrences:
                parent.core = _absorb_input_factor(
                    parent.core, role, basis, normalize=False
                )
                pushed_parent_occurrences += 1
            if not occurrences:
                if node is not work.root:
                    raise ValueError("non-root node has no EVD parent occurrence")
                work.head = work.head @ vectors
                pushed_parent_occurrences += 1
            diagonalized_node_count += 1
        expected_messages = sum(len(node.children) for node in nodes_bottom_up)
        if algorithm2_child_messages != expected_messages:
            raise RuntimeError(
                "streamed Algorithm 2 did not emit every child message"
            )
    else:
        environments = (
            tuple(precomputed_environments)
            if precomputed_environments is not None
            else reverse_implicit_environments(
                work, output_metric=output_metric, telemetry=meter
            )
        )
        by_uid = {record.uid: record for record in environments}
        if set(by_uid) != {node.uid for node in nodes_bottom_up}:
            raise ValueError(
                "precomputed Algorithm 2 environments do not match the network"
            )
        algorithm2_child_messages = sum(
            record.child_messages_emitted for record in environments
        )
        for node in nodes_bottom_up:
            record = by_uid[node.uid]
            symmetric = 0.5 * (record.mantissa + record.mantissa.T)
            values, vectors = torch.linalg.eigh(symmetric)
            order = torch.argsort(values, descending=True)
            values = values[order]
            bases[node.uid] = vectors[:, order]
            if compact_spectrum_callback is not None:
                compact_spectrum_callback(
                    _compact_spectrum_record(
                        node.uid,
                        node.label,
                        values,
                        record.binary_exponent,
                    )
                )
            if retain_eigenvalues:
                eigenvalue_records.append(
                    ScaledMatrix(torch.diag(values), record.binary_exponent)
                )
        del by_uid
        del environments

    if not stream_pre_evd_environments:
        diagonalized_node_count = len(bases)
        for node in nodes_bottom_up:
            vectors = bases.pop(node.uid)
            node.core = _apply_output_basis(node.core, vectors.T)
            basis = ScaledMatrix(vectors, 0)
            occurrences = parents.get(id(node), [])
            expected_parent_occurrences += len(occurrences) if occurrences else 1
            for parent, role in occurrences:
                parent.core = _absorb_input_factor(
                    parent.core, role, basis, normalize=False
                )
                pushed_parent_occurrences += 1
            if not occurrences:
                if node is not work.root:
                    raise ValueError("non-root node has no EVD parent occurrence")
                work.head = work.head @ vectors
                pushed_parent_occurrences += 1

    replay_error = 0.0
    if reference is not None:
        replay_error = _projective_batch_relative_error(
            evaluate_projective_boundary(work, replay_inputs), reference
        )
    if retain_post_evd_environments:
        recontracted = reverse_implicit_environments(
            work, output_metric=output_metric, telemetry=meter
        )
        maximum_offdiagonal = max(
            (_offdiagonal_ratio(record.mantissa) for record in recontracted),
            default=0.0,
        )
    else:
        maximum_offdiagonal = 0.0

        def observe_recontracted(record: EnvironmentRecord) -> None:
            nonlocal maximum_offdiagonal
            maximum_offdiagonal = max(
                maximum_offdiagonal, _offdiagonal_ratio(record.mantissa)
            )

        reverse_implicit_environments(
            work,
            output_metric=output_metric,
            telemetry=meter,
            retain_records=False,
            record_callback=observe_recontracted,
        )
    if pushed_parent_occurrences != expected_parent_occurrences:
        raise RuntimeError(
            "Algorithm 3 did not push its basis to every parent occurrence and root head"
        )
    return DiagonalImplicitDAG(
        work,
        tuple(eigenvalue_records),
        replay_error,
        maximum_offdiagonal,
        expected_parent_occurrences,
        pushed_parent_occurrences,
        algorithm2_child_messages,
        diagonalized_node_count,
        retain_eigenvalues,
    )


@torch.no_grad()
def apply_shared_eigenbases_to_explicit_clone_occurrences_control(
    shared_network: ImplicitProjectiveDAG,
    explicit_clone_network: ImplicitProjectiveDAG,
    replay_inputs: RawPhysicalInput,
    *,
    output_metric: Tensor | None = None,
    omit_clone_parent: tuple[str, int] | None = None,
) -> CloneEVDTrace:
    """Apply each shared aggregate EVD basis to every matching clone occurrence."""

    _validate_network(shared_network)
    _validate_network(explicit_clone_network)
    shared = _clone_network(shared_network, unfold=False)
    clone = _clone_network(explicit_clone_network, unfold=False)
    shared_reference = evaluate_projective_boundary(shared, replay_inputs)
    clone_reference = evaluate_projective_boundary(clone, replay_inputs)
    environments = reverse_implicit_environments(shared, output_metric=output_metric)
    bases: dict[int, Tensor] = {}
    for record in environments:
        symmetric = 0.5 * (record.mantissa + record.mantissa.T)
        values, vectors = torch.linalg.eigh(symmetric)
        bases[record.uid] = vectors[:, torch.argsort(values, descending=True)]

    shared_parents = _parent_occurrences(shared.root)
    clone_parents = _parent_occurrences(clone.root)
    clone_by_origin: dict[int, list[ImplicitNode]] = defaultdict(list)
    for node in _walk_unique(clone.root):
        if node.origin_uid is None:
            raise ValueError("explicit clone EVD occurrence lost its origin")
        clone_by_origin[node.origin_uid].append(node)

    expected_shared = pushed_shared = 0
    expected_clone = pushed_clone = 0
    omitted = 0
    targeted = 0
    target_occurrence = 0
    for shared_node in _walk_unique(shared.root):
        basis = bases[shared_node.uid]
        shared_node.core = _apply_output_basis(shared_node.core, basis.T)
        occurrences = shared_parents.get(id(shared_node), [])
        if occurrences:
            expected_shared += len(occurrences)
            for parent, role in occurrences:
                parent.core = _absorb_input_factor(
                    parent.core, role, ScaledMatrix(basis, 0), normalize=False
                )
                pushed_shared += 1
        else:
            expected_shared += 1
            shared.head = shared.head @ basis
            pushed_shared += 1

        candidates = clone_by_origin.get(shared_node.uid, [])
        if not candidates:
            raise ValueError("clone EVD traversal missed a shared origin")
        for candidate in candidates:
            candidate.core = _apply_output_basis(candidate.core, basis.T)
            clone_occurrences = clone_parents.get(id(candidate), [])
            if clone_occurrences:
                expected_clone += len(clone_occurrences)
                for parent, role in clone_occurrences:
                    should_omit = (
                        omit_clone_parent is not None
                        and shared_node.label == omit_clone_parent[0]
                        and target_occurrence == omit_clone_parent[1]
                    )
                    target_occurrence += int(
                        omit_clone_parent is not None
                        and shared_node.label == omit_clone_parent[0]
                    )
                    if should_omit:
                        omitted += 1
                        continue
                    parent.core = _absorb_input_factor(
                        parent.core,
                        role,
                        ScaledMatrix(basis, 0),
                        normalize=False,
                    )
                    pushed_clone += 1
            else:
                expected_clone += 1
                clone.head = clone.head @ basis
                pushed_clone += 1

    if omit_clone_parent is not None and omitted != 1:
        raise ValueError("requested one omitted clone EVD occurrence but did not find it")

    shared_value = evaluate_projective_boundary(shared, replay_inputs)
    clone_value = evaluate_projective_boundary(clone, replay_inputs)
    shared_replay = _projective_batch_relative_error(shared_value, shared_reference)
    clone_replay = _projective_batch_relative_error(clone_value, clone_reference)
    shared_clone = _projective_batch_relative_error(shared_value, clone_value)

    shared_after = reverse_implicit_environments(shared, output_metric=output_metric)
    clone_after = reverse_implicit_environments(clone, output_metric=output_metric)
    clone_origin = {node.uid: node.origin_uid for node in _walk_unique(clone.root)}
    grouped: dict[int, list[ScaledMatrix]] = defaultdict(list)
    for record in clone_after:
        origin = clone_origin.get(record.uid)
        if origin is None:
            raise ValueError("post-EVD clone environment lost its origin")
        grouped[origin].append(ScaledMatrix(record.mantissa, record.binary_exponent))
    clone_aggregates = {
        uid: messages[0] if len(messages) == 1 else _add_scaled_matrices(messages)
        for uid, messages in grouped.items()
    }
    shared_aggregates = {
        record.uid: ScaledMatrix(record.mantissa, record.binary_exponent)
        for record in shared_after
    }
    if set(clone_aggregates) != set(shared_aggregates):
        raise ValueError("post-EVD shared and clone environment origins disagree")
    environment_error = max(
        _scaled_matrix_relative_error(clone_aggregates[uid], shared_aggregates[uid])
        for uid in shared_aggregates
    )
    shared_offdiagonal = max(
        (_offdiagonal_ratio(value.mantissa) for value in shared_aggregates.values()),
        default=0.0,
    )
    clone_offdiagonal = max(
        (_offdiagonal_ratio(value.mantissa) for value in clone_aggregates.values()),
        default=0.0,
    )
    return CloneEVDTrace(
        shared,
        clone,
        shared_replay,
        clone_replay,
        shared_clone,
        environment_error,
        shared_offdiagonal,
        clone_offdiagonal,
        expected_shared,
        pushed_shared,
        expected_clone,
        pushed_clone,
        omitted,
    )


def _aggregate_clone_environment_records(
    network: ImplicitProjectiveDAG,
    records: Sequence[EnvironmentRecord],
) -> dict[int, ScaledMatrix]:
    origin_by_uid = {node.uid: node.origin_uid for node in _walk_unique(network.root)}
    grouped: dict[int, list[ScaledMatrix]] = defaultdict(list)
    for record in records:
        origin = origin_by_uid.get(record.uid)
        if origin is None:
            raise ValueError("explicit clone environment has no origin")
        grouped[origin].append(ScaledMatrix(record.mantissa, record.binary_exponent))
    return {
        uid: messages[0] if len(messages) == 1 else _add_scaled_matrices(messages)
        for uid, messages in grouped.items()
    }


def _eigenvalue_relative_error(
    first_values: ScaledMatrix,
    second_values: ScaledMatrix,
) -> float:
    center = max(first_values.binary_exponent, second_values.binary_exponent)
    first = _ldexp(
        torch.diagonal(first_values.mantissa),
        first_values.binary_exponent - center,
    )
    second = _ldexp(
        torch.diagonal(second_values.mantissa),
        second_values.binary_exponent - center,
    )
    scale = first.norm().clamp_min(torch.finfo(first.dtype).tiny)
    return float(((first - second).norm() / scale).item())


@torch.no_grad()
def diagonalize_shared_and_explicit_clone_independently(
    shared_network: ImplicitProjectiveDAG,
    explicit_clone_network: ImplicitProjectiveDAG,
    replay_inputs: RawPhysicalInput,
    *,
    output_metric: Tensor | None = None,
    omit_clone_parent: tuple[str, int] | None = None,
) -> IndependentCloneEVDTrace:
    """Independent Algorithm 3 on shared and no-memo dense-clone contractions."""

    _validate_network(shared_network)
    _validate_network(explicit_clone_network)
    shared = _clone_network(shared_network, unfold=False)
    clone = _clone_network(explicit_clone_network, unfold=False)
    shared_reference = evaluate_projective_boundary(shared, replay_inputs)
    clone_reference = evaluate_projective_boundary(clone, replay_inputs)

    shared_records = reverse_implicit_environments(shared, output_metric=output_metric)
    clone_records = reverse_implicit_environments(clone, output_metric=output_metric)
    shared_messages = {
        record.uid: ScaledMatrix(record.mantissa, record.binary_exponent)
        for record in shared_records
    }
    clone_messages = _aggregate_clone_environment_records(clone, clone_records)
    if set(shared_messages) != set(clone_messages):
        raise ValueError("independent clone EVD origins disagree with shared nodes")
    pre_environment_error = max(
        _scaled_matrix_relative_error(clone_messages[uid], shared_messages[uid])
        for uid in shared_messages
    )

    shared_bases: dict[int, Tensor] = {}
    clone_bases: dict[int, Tensor] = {}
    eigenvalue_error = 0.0
    for uid in shared_messages:
        shared_message = shared_messages[uid]
        clone_message = clone_messages[uid]
        shared_values, shared_vectors = torch.linalg.eigh(
            0.5 * (shared_message.mantissa + shared_message.mantissa.T)
        )
        clone_values, clone_vectors = torch.linalg.eigh(
            0.5 * (clone_message.mantissa + clone_message.mantissa.T)
        )
        shared_order = torch.argsort(shared_values, descending=True)
        clone_order = torch.argsort(clone_values, descending=True)
        shared_values = shared_values[shared_order]
        clone_values = clone_values[clone_order]
        shared_vectors = shared_vectors[:, shared_order]
        clone_vectors = clone_vectors[:, clone_order]
        shared_bases[uid] = shared_vectors
        clone_bases[uid] = clone_vectors
        value_error = _eigenvalue_relative_error(
            ScaledMatrix(torch.diag(shared_values), shared_message.binary_exponent),
            ScaledMatrix(torch.diag(clone_values), clone_message.binary_exponent),
        )
        eigenvalue_error = max(eigenvalue_error, value_error)

    shared_parents = _parent_occurrences(shared.root)
    clone_parents = _parent_occurrences(clone.root)
    clone_by_origin: dict[int, list[ImplicitNode]] = defaultdict(list)
    for node in _walk_unique(clone.root):
        if node.origin_uid is None:
            raise ValueError("independent clone EVD node lost its origin")
        clone_by_origin[node.origin_uid].append(node)

    expected_shared = pushed_shared = 0
    expected_clone = pushed_clone = 0
    omitted = 0
    targeted = 0
    target_occurrence = 0
    for shared_node in _walk_unique(shared.root):
        shared_basis = shared_bases[shared_node.uid]
        shared_node.core = _apply_output_basis(shared_node.core, shared_basis.T)
        occurrences = shared_parents.get(id(shared_node), [])
        if occurrences:
            expected_shared += len(occurrences)
            for parent, role in occurrences:
                parent.core = _absorb_input_factor(
                    parent.core,
                    role,
                    ScaledMatrix(shared_basis, 0),
                    normalize=False,
                )
                pushed_shared += 1
        else:
            expected_shared += 1
            shared.head = shared.head @ shared_basis
            pushed_shared += 1

        clone_basis = clone_bases[shared_node.uid]
        candidates = clone_by_origin.get(shared_node.uid, [])
        if not candidates:
            raise ValueError("independent clone EVD missed a shared origin")
        for candidate in candidates:
            candidate.core = _apply_output_basis(candidate.core, clone_basis.T)
            occurrences = clone_parents.get(id(candidate), [])
            if occurrences:
                expected_clone += len(occurrences)
                for parent, role in occurrences:
                    is_target = (
                        omit_clone_parent is not None
                        and shared_node.label == omit_clone_parent[0]
                    )
                    should_omit = is_target and target_occurrence == omit_clone_parent[1]
                    if is_target:
                        target_occurrence += 1
                        targeted += 1
                    if should_omit:
                        omitted += 1
                        continue
                    parent.core = _absorb_input_factor(
                        parent.core,
                        role,
                        ScaledMatrix(clone_basis, 0),
                        normalize=False,
                    )
                    pushed_clone += 1
            else:
                expected_clone += 1
                clone.head = clone.head @ clone_basis
                pushed_clone += 1
    if omit_clone_parent is not None and omitted != 1:
        raise ValueError("independent clone EVD omission did not resolve exactly once")

    shared_value = evaluate_projective_boundary(shared, replay_inputs)
    clone_value = evaluate_projective_boundary(clone, replay_inputs)
    shared_replay = _projective_batch_relative_error(shared_value, shared_reference)
    clone_replay = _projective_batch_relative_error(clone_value, clone_reference)
    shared_clone = _projective_batch_relative_error(shared_value, clone_value)
    shared_after = reverse_implicit_environments(shared, output_metric=output_metric)
    clone_after = reverse_implicit_environments(clone, output_metric=output_metric)
    shared_after_messages = {
        record.uid: ScaledMatrix(record.mantissa, record.binary_exponent)
        for record in shared_after
    }
    clone_after_messages = _aggregate_clone_environment_records(clone, clone_after)
    shared_offdiagonal = max(
        (_offdiagonal_ratio(value.mantissa) for value in shared_after_messages.values()),
        default=0.0,
    )
    clone_offdiagonal = max(
        (_offdiagonal_ratio(value.mantissa) for value in clone_after_messages.values()),
        default=0.0,
    )
    return IndependentCloneEVDTrace(
        shared,
        clone,
        pre_environment_error,
        eigenvalue_error,
        shared_replay,
        clone_replay,
        shared_clone,
        shared_offdiagonal,
        clone_offdiagonal,
        expected_shared,
        pushed_shared,
        expected_clone,
        pushed_clone,
        omitted,
        targeted,
    )


class _Builder:
    def __init__(self, like: Tensor, telemetry: MaterializationTelemetry):
        self.like = like
        self.telemetry = telemetry
        self.next_uid = 0
        self._physical_specs: dict[str, PhysicalSourceSpec] = {}
        self._physical_nodes: dict[str, ImplicitNode] = {}

    @property
    def physical_source_specs(self) -> tuple[PhysicalSourceSpec, ...]:
        return tuple(self._physical_specs.values())

    def _constant_coordinates(
        self,
        label: str,
        coordinates: Tensor,
        binary_exponent: int = 0,
    ) -> ProjectiveConstant:
        _validate_real_finite(coordinates, f"constant {label}")
        if coordinates.ndim != 1 or coordinates.numel() < 1:
            raise ValueError("constant projective coordinates must be a nonempty vector")
        mantissa, local = _strip_power_of_two(coordinates)
        self.telemetry.observe_persistent(mantissa)
        return ProjectiveConstant(label, mantissa, binary_exponent + local)

    def constant_pair(self, label: str, value: Tensor) -> ProjectiveConstant:
        _validate_real_finite(value, f"constant pair {label}")
        if value.ndim != 1:
            raise ValueError("constant pair value must be a vector")
        if value.dtype != self.like.dtype or value.device != self.like.device:
            raise ValueError("constant pair must share builder dtype and device")
        coordinates = torch.cat((value, value.new_ones(1)))
        return self._constant_coordinates(label, coordinates)

    def physical_pair(self, spec: PhysicalSourceSpec) -> ImplicitNode:
        existing = self._physical_nodes.get(spec.key)
        if existing is not None:
            if self._physical_specs[spec.key] != spec:
                raise ValueError("physical source key was reused with a different specification")
            return existing
        identity = torch.eye(
            spec.width + 1, dtype=self.like.dtype, device=self.like.device
        )
        result = self.unary(
            spec.label,
            identity,
            physical_source_key=spec.key,
            kind="heterogeneous_physical_homogeneous_embedding",
        )
        if not isinstance(result, ImplicitNode):
            raise RuntimeError("physical source construction unexpectedly folded")
        self._physical_specs[spec.key] = spec
        self._physical_nodes[spec.key] = result
        return result

    def node(
        self,
        label: str,
        core: ImplicitCore,
        children: tuple[ImplicitNode, ...] = (),
        *,
        physical_token: int | None = None,
        physical_source_key: str | None = None,
    ) -> ImplicitNode:
        if physical_token is not None and physical_source_key is not None:
            raise ValueError("a physical leaf cannot select two raw sources")
        physical_leaf = (
            physical_token is not None or physical_source_key is not None
        ) and not children
        if not physical_leaf and len(children) != _core_arity(core):
            raise ValueError(f"{label} child/core arity mismatch")
        if children and tuple(child.output_dimension for child in children) != core.input_dimensions:
            raise ValueError(f"{label} child/core dimensions mismatch")
        result = ImplicitNode(
            self.next_uid,
            label,
            core,
            children,
            physical_token,
            self.next_uid,
            physical_source_key,
        )
        self.next_uid += 1
        return result

    def unary(
        self,
        label: str,
        matrix: Tensor,
        child: ProjectiveValue | None = None,
        *,
        physical_token: int | None = None,
        physical_source_key: str | None = None,
        kind: str = "affine",
    ) -> ProjectiveValue:
        core = _make_unary_core(matrix, kind, self.telemetry)
        if isinstance(child, ProjectiveConstant):
            self.telemetry.constant_unary_folds += 1
            value = core.matrix @ child.mantissa
            return self._constant_coordinates(
                label,
                value,
                core.binary_exponent + child.binary_exponent,
            )
        children = () if child is None else (child,)
        return self.node(
            label,
            core,
            children,
            physical_token=physical_token,
            physical_source_key=physical_source_key,
        )

    def cp(
        self,
        label: str,
        output: Tensor,
        left: Tensor,
        right: Tensor,
        children: tuple[ProjectiveValue, ProjectiveValue],
        *,
        kind: str,
    ) -> ProjectiveValue:
        core = _make_cp_core(output, left, right, kind, self.telemetry)
        left_child, right_child = children
        left_constant = isinstance(left_child, ProjectiveConstant)
        right_constant = isinstance(right_child, ProjectiveConstant)
        if left_constant and right_constant:
            self.telemetry.constant_cp_to_constant_folds += 1
            left_value = core.left_factor @ left_child.mantissa
            right_value = core.right_factor @ right_child.mantissa
            value = core.output_factor @ (left_value * right_value)
            return self._constant_coordinates(
                label,
                value,
                core.binary_exponent
                + left_child.binary_exponent
                + right_child.binary_exponent,
            )
        if left_constant or right_constant:
            self.telemetry.constant_cp_to_unary_folds += 1
            constant = left_child if left_constant else right_child
            variable = right_child if left_constant else left_child
            if not isinstance(constant, ProjectiveConstant) or not isinstance(
                variable, ImplicitNode
            ):
                raise RuntimeError("constant CP partial evaluation lost its typed children")
            constant_factor = (
                core.left_factor if left_constant else core.right_factor
            )
            variable_factor = (
                core.right_factor if left_constant else core.left_factor
            )
            weights = constant_factor @ constant.mantissa
            matrix = (core.output_factor * weights[None, :]) @ variable_factor
            unary = _make_unary_core(
                matrix, f"constant_folded::{kind}", self.telemetry
            )
            unary.binary_exponent += core.binary_exponent + constant.binary_exponent
            return self.node(label, unary, (variable,))
        return self.node(
            label,
            core,
            (left_child, right_child),
        )


def _identity_pair_leaf(builder: _Builder, token: int, dimension: int) -> ImplicitNode:
    identity = torch.eye(dimension + 1, dtype=builder.like.dtype, device=builder.like.device)
    result = builder.unary(
        f"input.token{token}",
        identity,
        physical_token=token,
        kind="physical_homogeneous_embedding",
    )
    if not isinstance(result, ImplicitNode):
        raise RuntimeError("physical token construction unexpectedly folded")
    return result


def _affine_pair(
    builder: _Builder,
    value: ProjectiveValue,
    weight: Tensor,
    bias: Tensor | None,
    label: str,
) -> ProjectiveValue:
    output, input_dimension = weight.shape
    if value.output_dimension != input_dimension + 1:
        raise ValueError(f"{label} affine input dimension mismatch")
    matrix = weight.new_zeros(output + 1, input_dimension + 1)
    matrix[:-1, :-1] = weight
    if bias is not None:
        if bias.shape != (output,):
            raise ValueError(f"{label} affine bias dimension mismatch")
        matrix[:-1, -1] = bias
    matrix[-1, -1] = 1.0
    return builder.unary(label, matrix, value, kind="projective_affine")


def _scale_pair_vector(
    builder: _Builder,
    value: ProjectiveValue,
    scale: Tensor | float,
    label: str,
) -> ProjectiveValue:
    dimension = value.output_dimension - 1
    scalar = torch.as_tensor(
        scale, dtype=builder.like.dtype, device=builder.like.device
    ).reshape(())
    weight = torch.eye(dimension, dtype=builder.like.dtype, device=builder.like.device) * scalar
    return _affine_pair(builder, value, weight, None, label)


def _add_pair_vectors(
    builder: _Builder,
    left_value: ProjectiveValue,
    right_value: ProjectiveValue,
    label: str,
) -> ProjectiveValue:
    if left_value.output_dimension != right_value.output_dimension:
        raise ValueError("projective add requires equal vector dimensions")
    size = left_value.output_dimension
    dimension = size - 1
    rank = 2 * dimension + 1
    output = builder.like.new_zeros(size, rank)
    left = builder.like.new_zeros(rank, size)
    right = builder.like.new_zeros(rank, size)
    term = 0
    for index in range(dimension):
        output[index, term] = 1.0
        left[term, index] = 1.0
        right[term, -1] = 1.0
        term += 1
        output[index, term] = 1.0
        left[term, -1] = 1.0
        right[term, index] = 1.0
        term += 1
    output[-1, term] = 1.0
    left[term, -1] = 1.0
    right[term, -1] = 1.0
    return builder.cp(
        label,
        output,
        left,
        right,
        (left_value, right_value),
        kind="sparse_projective_vector_add",
    )


def _dot_pair_vectors(
    builder: _Builder,
    left_value: ProjectiveValue,
    right_value: ProjectiveValue,
    label: str,
) -> ProjectiveValue:
    if left_value.output_dimension != right_value.output_dimension:
        raise ValueError("projective dot requires equal vector dimensions")
    size = left_value.output_dimension
    dimension = size - 1
    rank = dimension + 1
    output = builder.like.new_zeros(2, rank)
    left = builder.like.new_zeros(rank, size)
    right = builder.like.new_zeros(rank, size)
    for index in range(dimension):
        output[0, index] = 1.0
        left[index, index] = 1.0
        right[index, index] = 1.0
    output[1, -1] = 1.0
    left[-1, -1] = 1.0
    right[-1, -1] = 1.0
    return builder.cp(
        label,
        output,
        left,
        right,
        (left_value, right_value),
        kind="sparse_projective_dot",
    )


def _multiply_pair_scalars(
    builder: _Builder,
    left_value: ProjectiveValue,
    right_value: ProjectiveValue,
    label: str,
) -> ProjectiveValue:
    if left_value.output_dimension != 2 or right_value.output_dimension != 2:
        raise ValueError("projective scalar multiply needs scalar pairs")
    output = torch.eye(2, dtype=builder.like.dtype, device=builder.like.device)
    factors = output.clone()
    return builder.cp(
        label,
        output,
        factors,
        factors,
        (left_value, right_value),
        kind="sparse_projective_scalar_product",
    )


def _scale_pair_vector_by_scalar(
    builder: _Builder,
    scalar: ProjectiveValue,
    vector: ProjectiveValue,
    label: str,
) -> ProjectiveValue:
    if scalar.output_dimension != 2:
        raise ValueError("first scale input must be a projective scalar")
    size = vector.output_dimension
    output = torch.eye(size, dtype=builder.like.dtype, device=builder.like.device)
    scalar_factor = builder.like.new_zeros(size, 2)
    vector_factor = torch.eye(size, dtype=builder.like.dtype, device=builder.like.device)
    scalar_factor[:-1, 0] = 1.0
    scalar_factor[-1, 1] = 1.0
    return builder.cp(
        label,
        output,
        scalar_factor,
        vector_factor,
        (scalar, vector),
        kind="sparse_projective_scalar_times_vector",
    )


def _concatenate_pair_vectors(
    builder: _Builder,
    first: ProjectiveValue,
    second: ProjectiveValue,
    label: str,
) -> ProjectiveValue:
    first_dimension = first.output_dimension - 1
    second_dimension = second.output_dimension - 1
    output_dimension = first_dimension + second_dimension + 1
    rank = output_dimension
    output = torch.eye(
        output_dimension, dtype=builder.like.dtype, device=builder.like.device
    )
    left = builder.like.new_zeros(rank, first_dimension + 1)
    right = builder.like.new_zeros(rank, second_dimension + 1)
    for index in range(first_dimension):
        left[index, index] = 1.0
        right[index, -1] = 1.0
    offset = first_dimension
    for index in range(second_dimension):
        left[offset + index, -1] = 1.0
        right[offset + index, index] = 1.0
    left[-1, -1] = 1.0
    right[-1, -1] = 1.0
    return builder.cp(
        label,
        output,
        left,
        right,
        (first, second),
        kind="sparse_projective_vector_concat",
    )


def _mean_square_pair(
    builder: _Builder,
    value: ProjectiveValue,
    module: RationalNorm,
    label: str,
) -> ProjectiveValue:
    dimension = value.output_dimension - 1
    size = dimension + 1
    rank = dimension + 1
    output = builder.like.new_zeros(2, rank)
    factors = builder.like.new_zeros(rank, size)
    for index in range(dimension):
        output[0, index] = 1.0 / dimension
        factors[index, index] = 1.0
    scale = module.running_ms.to(dtype=builder.like.dtype, device=builder.like.device).clamp_min(1e-12)
    output[0, -1] = float(module.eps)
    output[1, -1] = scale
    factors[-1, -1] = 1.0
    return builder.cp(
        label,
        output,
        factors,
        factors.clone(),
        (value, value),
        kind="sparse_projective_mean_square",
    )


def _pade_pair(
    builder: _Builder,
    square: ProjectiveValue,
    module: RationalNorm,
    label: str,
) -> ProjectiveValue:
    if square.output_dimension != 2 or int(module.deg) != 2:
        raise ValueError("implicit production kernel requires degree-two Padé")
    pa = module.pa.to(dtype=builder.like.dtype, device=builder.like.device)
    pb = module.pb.to(dtype=builder.like.dtype, device=builder.like.device)
    # Ordered atoms TT, ST, TS, SS.  Splitting the mixed coefficient equally
    # preserves the symmetric clone-unfolded convention of the dense oracle.
    output = torch.stack(
        (
            torch.stack((pa[0], pb[0])),
            torch.stack((pa[1] / 2.0, pb[1] / 2.0)),
            torch.stack((pa[1] / 2.0, pb[1] / 2.0)),
            torch.stack((pa[2], pb[2])),
        ),
        dim=1,
    )
    left = builder.like.new_tensor(((0.0, 1.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    right = builder.like.new_tensor(((0.0, 1.0), (0.0, 1.0), (1.0, 0.0), (1.0, 0.0)))
    return builder.cp(
        label,
        output,
        left,
        right,
        (square, square),
        kind="sparse_projective_pade_degree2",
    )


def _pade_norm_pair(
    builder: _Builder,
    value: ProjectiveValue,
    module: RationalNorm,
    label: str,
) -> ProjectiveValue:
    if not isinstance(module, RationalNorm) or module.variant != "pade":
        raise ValueError(f"{label} is not a Padé RationalNorm")
    if not bool(module.initialized):
        raise ValueError(f"{label} has an uninitialized running mean square")
    square = _mean_square_pair(builder, value, module, f"{label}.mean_square")
    pade = _pade_pair(builder, square, module, f"{label}.pade")
    dimension = value.output_dimension - 1
    size = dimension + 1
    output = torch.eye(size, dtype=builder.like.dtype, device=builder.like.device)
    vector_factor = output.clone()
    scalar_factor = builder.like.new_zeros(size, 2)
    scalar_factor[:-1, 0] = torch.rsqrt(
        module.running_ms.to(dtype=builder.like.dtype, device=builder.like.device).clamp_min(1e-12)
    )
    scalar_factor[-1, 1] = 1.0
    return builder.cp(
        f"{label}.projective_output",
        output,
        vector_factor,
        scalar_factor,
        (value, pade),
        kind="sparse_projective_pade_vector_output",
    )


def _fused_cp_ffn_pair(
    builder: _Builder,
    value: ProjectiveValue,
    module: BilinearFFN,
    label: str,
) -> ProjectiveValue:
    dimension = value.output_dimension - 1
    if module.dim != dimension or module.out_dim != dimension:
        raise ValueError("fused CP FFN dimensions do not match its pair bond")
    rank = module.rank + 1
    size = dimension + 1
    left = builder.like.new_zeros(rank, size)
    right = builder.like.new_zeros(rank, size)
    left[:-1, :-1] = module.left.weight
    left[:-1, -1] = module.left.bias
    right[:-1, :-1] = module.right.weight
    right[:-1, -1] = module.right.bias
    left[-1, -1] = 1.0
    right[-1, -1] = 1.0
    output = builder.like.new_zeros(size, rank)
    output[:-1, :-1] = module.down.weight
    if module.down.bias is not None:
        output[:-1, -1] = module.down.bias
    output[-1, -1] = 1.0
    builder.telemetry.fused_ffn_primitive_count += 1
    return builder.cp(
        label,
        output,
        left,
        right,
        (value, value),
        kind="fused_cp_ffn_projective_pair",
    )


def _norm_snapshot(block: ChiTransformerBlock) -> tuple[tuple[str, Tensor], ...]:
    result: list[tuple[str, Tensor]] = []
    for name, module in block.named_modules():
        if isinstance(module, RationalNorm):
            result.extend(
                (
                    (f"{name}.running_ms", module.running_ms.detach().clone()),
                    (f"{name}.initialized", module.initialized.detach().clone()),
                    (f"{name}.pa", module.pa.detach().clone()),
                    (f"{name}.pb", module.pb.detach().clone()),
                )
            )
    return tuple(result)


def _validate_block_source(
    block: ChiTransformerBlock,
    positions: Tensor,
) -> tuple[int, int, int]:
    if block.training or not block.residual:
        raise ValueError("implicit production kernel needs an eval-mode residual block")
    if not isinstance(block.attn, BilinearAttention) or block.attn.qk_norm != "rational":
        raise ValueError("implicit production kernel needs rational bilinear attention")
    if not isinstance(block.ffn, BilinearFFN):
        raise ValueError("implicit production kernel needs a BilinearFFN")
    if not isinstance(block.rbn_attn, RationalNorm) or not isinstance(block.rbn_ffn, RationalNorm):
        raise ValueError("both block pre-norms must be Padé RationalNorm modules")
    if positions.ndim != 2 or positions.shape[1] != block.attn.dim:
        raise ValueError("position tensor must have shape (tokens, width)")
    norms = tuple(module for module in block.modules() if isinstance(module, RationalNorm))
    if len(norms) != 6:
        raise ValueError("unchanged block must expose exactly six Padé module sites")
    for module in norms:
        if module.variant != "pade" or int(module.deg) != 2 or not bool(module.initialized):
            raise ValueError("every Padé site must be degree two and initialized")
    tensors = tuple(block.parameters()) + tuple(block.buffers()) + (positions,)
    floating = [tensor for tensor in tensors if tensor.is_floating_point()]
    if any(tensor.dtype != torch.float64 for tensor in floating):
        raise ValueError("implicit direct-RQ kernel requires float64 source tensors")
    devices = {tensor.device for tensor in floating}
    if len(devices) != 1:
        raise ValueError("implicit source tensors must share one device")
    return int(positions.shape[0]), block.attn.dim, block.attn.n_heads


@torch.no_grad()
def compile_implicit_projective_block_boundary(
    block: ChiTransformerBlock,
    positions: Tensor,
    *,
    selected_token: int | None = None,
    mask: Tensor | None = None,
) -> ImplicitBlockOracle:
    token_count, dimension, heads = _validate_block_source(block, positions)
    like = block.attn.wo.weight
    selected = token_count - 1 if selected_token is None else int(selected_token)
    if not 0 <= selected < token_count:
        raise ValueError("selected token is out of range")
    if mask is None:
        fixed_mask = (
            causal_mask(token_count, dtype=like.dtype, device=like.device)
            if block.attn.causal
            else like.new_ones(token_count, token_count)
        )
    else:
        fixed_mask = mask.to(dtype=like.dtype, device=like.device)
    if fixed_mask.shape != (token_count, token_count):
        raise ValueError("fixed attention mask has the wrong shape")
    if not bool(((fixed_mask == 0) | (fixed_mask == 1)).all()):
        raise ValueError("fixed attention mask must be binary")
    visible_sources = [
        source for source in range(token_count) if float(fixed_mask[selected, source]) == 1.0
    ]
    if not visible_sources:
        raise ValueError("selected attention row has no visible source")

    telemetry = MaterializationTelemetry()
    builder = _Builder(like, telemetry)
    raw = tuple(_identity_pair_leaf(builder, token, dimension) for token in range(token_count))
    identity = torch.eye(dimension, dtype=like.dtype, device=like.device)
    positioned = tuple(
        _affine_pair(builder, raw[token], identity, positions[token], f"position.token{token}")
        for token in range(token_count)
    )
    normalized = tuple(
        _pade_norm_pair(builder, value, block.rbn_attn, f"pre_attention.token{token}")
        for token, value in enumerate(positioned)
    )

    head_dimension = dimension // heads
    head_values: list[ImplicitNode] = []
    visible = like.new_tensor(float(len(visible_sources)))
    row_scale = visible.rsqrt() if block.attn.row_scale == "invsqrt" else visible.reciprocal()
    fixed_scale = row_scale / float(block.attn._score_denom)
    for head in range(heads):
        section = slice(head * head_dimension, (head + 1) * head_dimension)

        def project(linear: nn.Linear, token: int, name: str) -> ImplicitNode:
            return _affine_pair(
                builder,
                normalized[token],
                linear.weight[section],
                None if linear.bias is None else linear.bias[section],
                f"attention.head{head}.{name}.token{token}",
            )

        q1 = _pade_norm_pair(
            builder,
            project(block.attn.wq1, selected, "q1_affine"),
            block.attn.rn_q1,
            f"attention.head{head}.q1_norm.token{selected}",
        )
        q2 = _pade_norm_pair(
            builder,
            project(block.attn.wq2, selected, "q2_affine"),
            block.attn.rn_q2,
            f"attention.head{head}.q2_norm.token{selected}",
        )
        routes: list[ImplicitNode] = []
        for source in visible_sources:
            k1 = _pade_norm_pair(
                builder,
                project(block.attn.wk1, source, "k1_affine"),
                block.attn.rn_k1,
                f"attention.head{head}.k1_norm.token{source}",
            )
            k2 = _pade_norm_pair(
                builder,
                project(block.attn.wk2, source, "k2_affine"),
                block.attn.rn_k2,
                f"attention.head{head}.k2_norm.token{source}",
            )
            value = project(block.attn.wv, source, "value_affine")
            score1 = _dot_pair_vectors(
                builder, q1, k1, f"attention.head{head}.route{source}.score1"
            )
            score2 = _dot_pair_vectors(
                builder, q2, k2, f"attention.head{head}.route{source}.score2"
            )
            score = _multiply_pair_scalars(
                builder,
                score1,
                score2,
                f"attention.head{head}.route{source}.score_product",
            )
            route = _scale_pair_vector_by_scalar(
                builder,
                score,
                value,
                f"attention.head{head}.route{source}.score_times_value",
            )
            routes.append(
                _scale_pair_vector(
                    builder,
                    route,
                    fixed_scale,
                    f"attention.head{head}.route{source}.fixed_scale",
                )
            )
        head_value = routes[0]
        for route_index, route in enumerate(routes[1:], start=1):
            head_value = _add_pair_vectors(
                builder,
                head_value,
                route,
                f"attention.head{head}.route_add{route_index}",
            )
        head_values.append(head_value)

    attention = head_values[0]
    for head, head_value in enumerate(head_values[1:], start=1):
        attention = _concatenate_pair_vectors(
            builder, attention, head_value, f"attention.head_concat{head}"
        )
    attention = _affine_pair(
        builder,
        attention,
        block.attn.wo.weight,
        block.attn.wo.bias,
        "attention.output_affine",
    )
    attention = _scale_pair_vector(
        builder, attention, block.attn_gain, "attention.residual_gain"
    )
    after_attention = _add_pair_vectors(
        builder,
        positioned[selected],
        attention,
        "attention.residual_add",
    )
    ffn_input = _pade_norm_pair(builder, after_attention, block.rbn_ffn, "pre_ffn")
    ffn = _fused_cp_ffn_pair(builder, ffn_input, block.ffn, "ffn.fused_cp")
    ffn = _scale_pair_vector(builder, ffn, block.ffn_gain, "ffn.residual_gain")
    output = _add_pair_vectors(builder, after_attention, ffn, "ffn.residual_add")
    network = ImplicitProjectiveDAG(
        output,
        torch.eye(dimension + 1, dtype=like.dtype, device=like.device),
        0,
        token_count,
        dimension,
        selected,
        fixed_mask,
    )
    return ImplicitBlockOracle(
        network,
        block,
        positions.detach().clone(),
        telemetry,
        _norm_snapshot(block),
    )


@torch.no_grad()
def source_block_boundary(oracle: ImplicitBlockOracle, raw_input: Tensor) -> Tensor:
    value = raw_input + oracle.positions
    output = oracle.block(value, mask=oracle.network.mask, method="explicit")
    return output[:, oracle.network.selected_token]


@torch.no_grad()
def assert_norm_buffers_unchanged(oracle: ImplicitBlockOracle) -> None:
    current = dict(_norm_snapshot(oracle.block))
    for name, expected in oracle.norm_buffer_snapshot:
        if name not in current or not torch.equal(current[name], expected):
            raise RuntimeError(f"implicit compiler changed normalization buffer {name}")


def implicit_shape_statistics(network: ImplicitProjectiveDAG) -> dict[str, int]:
    nodes = _walk_unique(network.root)
    cp = [node.core for node in nodes if isinstance(node.core, CPBinaryCore)]
    unary = [node.core for node in nodes if isinstance(node.core, UnaryCore)]
    rectangular = [
        node.core for node in nodes if isinstance(node.core, ReducedQRBinaryCore)
    ]
    dense_clones = [node.core for node in nodes if isinstance(node.core, DenseCloneCore)]
    fused_raw_dimension = (
        sum(spec.width for spec in network.physical_sources) + 1
        if network.physical_sources
        else network.token_count * network.feature_dimension + 1
    )
    return {
        "unique_nodes": len(nodes),
        "edge_occurrences": sum(len(node.children) for node in nodes),
        "cp_binary_nodes": len(cp),
        "reduced_q_binary_nodes": len(rectangular),
        "dense_clone_nodes": len(dense_clones),
        "unary_nodes": len(unary),
        "heterogeneous_physical_sources": len(network.physical_sources),
        "maximum_local_bond_dimension": max(node.output_dimension for node in nodes),
        "maximum_cp_rank": max((core.cp_rank for core in cp), default=0),
        "maximum_reduced_q_elements": max(
            (core.q_rows.numel() for core in rectangular), default=0
        ),
        "fused_token_feature_dimension": fused_raw_dimension,
    }


def predict_direct_rq_route_inventory(
    network: ImplicitProjectiveDAG,
) -> dict[str, object]:
    """Exactly predict postorder structural Algorithm-1 routes.

    Algorithm 1 pushes each child factor before visiting its parent, so a raw
    graph inventory is not exact.  This dimension-only postorder simulation
    propagates every canonical child output width and then applies the same
    pre-QR bounded-shape predicate as the live sweep.  No tensor factorization
    or numerical-rank decision occurs here.

    Storage deltas are Python-integer syntactic tensor-element counts for each
    CP core as it exists at its predicted visit, after child-bond shrinkage.
    """

    nodes = _walk_unique(network.root)
    predicted_outputs: dict[int, int] = {}
    candidate_count = 0
    total_q_elements = 0
    total_replaced_cp_elements = 0
    positive_storage_delta = 0
    maximum_unfolding_elements = 0
    maximum_q_elements = 0
    streamed_count = 0
    streamed_tall_failure_count = 0
    changed_input_shape_count = 0
    bounded_shapes: dict[str, int] = defaultdict(int)
    streamed_tall_failure_shapes: dict[str, int] = defaultdict(int)

    for node in nodes:
        core = node.core
        if isinstance(core, (ReducedQRBinaryCore, DenseCloneCore)):
            raise ValueError(
                "route prediction requires a pre-Algorithm-1 production DAG"
            )
        if node.children:
            input_dimensions = tuple(
                predicted_outputs[id(child)] for child in node.children
            )
        else:
            input_dimensions = core.input_dimensions
        if input_dimensions != core.input_dimensions:
            changed_input_shape_count += 1
        rows = core.output_dimension
        columns = math.prod(input_dimensions)

        if isinstance(core, UnaryCore):
            if len(input_dimensions) != 1:
                raise ValueError("predicted unary route does not have one input")
            predicted_outputs[id(node)] = min(rows, columns)
            continue
        if not isinstance(core, CPBinaryCore) or len(input_dimensions) != 2:
            raise ValueError("predicted CP route does not have two inputs")

        q_elements = min(rows, columns) * columns
        shape = f"{rows}x{columns}"
        if not _bounded_explicit_q_allowed_shape(rows, columns):
            streamed_count += 1
            if columns < rows:
                streamed_tall_failure_count += 1
                streamed_tall_failure_shapes[shape] += 1
                predicted_outputs[id(node)] = min(rows, columns)
            else:
                predicted_outputs[id(node)] = rows
            continue

        rank = core.cp_rank
        cp_elements = rows * rank + rank * sum(input_dimensions)
        candidate_count += 1
        total_q_elements += q_elements
        total_replaced_cp_elements += cp_elements
        positive_storage_delta += max(0, q_elements - cp_elements)
        maximum_unfolding_elements = max(
            maximum_unfolding_elements, rows * columns
        )
        maximum_q_elements = max(maximum_q_elements, q_elements)
        bounded_shapes[shape] += 1
        predicted_outputs[id(node)] = min(rows, columns)

    return {
        "schema": "exact_postorder_structural_direct_rq_routes_v1",
        "bounded_explicit_unfolding_element_limit": (
            MAXIMUM_BOUNDED_EXPLICIT_UNFOLDING_ELEMENTS
        ),
        "bounded_explicit_q_element_limit": (
            MAXIMUM_BOUNDED_EXPLICIT_Q_ELEMENTS
        ),
        "bounded_retained_q_candidate_count": candidate_count,
        "bounded_retained_q_total_elements": total_q_elements,
        "bounded_retained_q_replaced_cp_total_elements": total_replaced_cp_elements,
        "bounded_retained_q_signed_storage_delta_elements": (
            total_q_elements - total_replaced_cp_elements
        ),
        "bounded_retained_q_positive_storage_delta_elements": positive_storage_delta,
        "bounded_retained_q_maximum_unfolding_elements": (
            maximum_unfolding_elements
        ),
        "bounded_retained_q_maximum_elements": maximum_q_elements,
        "streamed_cp_candidate_count": streamed_count,
        "streamed_tall_unrepresentable_count": streamed_tall_failure_count,
        "streamed_tall_unrepresentable_shape_counts": dict(
            sorted(streamed_tall_failure_shapes.items())
        ),
        "bounded_retained_q_shape_counts": dict(sorted(bounded_shapes.items())),
        "nodes_with_propagated_input_shape_change": changed_input_shape_count,
        "predicted_root_output_dimension": predicted_outputs[id(network.root)],
        "simulated_unique_node_count": len(nodes),
    }


@torch.no_grad()
def drop_first_residual_add_cross_term(
    network: ImplicitProjectiveDAG,
    *,
    label: str = "ffn.residual_add",
) -> ImplicitProjectiveDAG:
    work = _clone_network(network, unfold=False)
    matches = [node for node in _walk_unique(work.root) if node.label == label]
    if len(matches) != 1 or not isinstance(matches[0].core, CPBinaryCore):
        raise ValueError("residual-add negative control did not resolve one CP node")
    node = matches[0]
    if node.core.kind != "sparse_projective_vector_add":
        raise ValueError("negative-control target is not a projective add")
    output = node.core.output_factor.clone()
    output[:, 0] = 0.0
    node.core = CPBinaryCore(
        output,
        node.core.left_factor,
        node.core.right_factor,
        node.core.kind,
        node.core.binary_exponent,
        node.core.direct_q_provenance,
    )
    return work


@torch.no_grad()
def old_project_then_add_topology_is_rejected(
    dimension: int,
    heads: int,
    *,
    like: Tensor,
) -> bool:
    """Regression for the rank-deficient topology replaced by native concat."""

    if dimension % heads != 0 or heads < 3:
        raise ValueError("old-topology regression needs at least three equal heads")
    telemetry = MaterializationTelemetry()
    builder = _Builder(like, telemetry)
    size = dimension + 1
    dummy_left = builder.unary(
        "old.left",
        torch.eye(size, dtype=like.dtype, device=like.device),
        physical_token=0,
    )
    dummy_right = builder.unary(
        "old.right",
        torch.eye(size, dtype=like.dtype, device=like.device),
        physical_token=0,
    )
    add = _add_pair_vectors(builder, dummy_left, dummy_right, "old.project_then_add")
    head_dimension = dimension // heads
    first = like.new_zeros(size, head_dimension + 1)
    second = like.new_zeros(size, head_dimension + 1)
    first[:head_dimension, :head_dimension] = torch.eye(
        head_dimension, dtype=like.dtype, device=like.device
    )
    second[head_dimension : 2 * head_dimension, :head_dimension] = torch.eye(
        head_dimension, dtype=like.dtype, device=like.device
    )
    first[-1, -1] = 1.0
    second[-1, -1] = 1.0
    transformed = _absorb_input_factor(add.core, 0, ScaledMatrix(first, 0))
    transformed = _absorb_input_factor(transformed, 1, ScaledMatrix(second, 0))
    try:
        # The bounded tiny representation can retain an arbitrary null-row Q
        # completion exactly.  The production objection is specifically that
        # the compact streamed CP representation must solve through a singular
        # square R after projecting each head to the full model width.
        _direct_rq_cp_streamed(transformed, telemetry, block_size=512)
    except ValueError as error:
        message = str(error)
        return (
            "row-rank deficient" in message
            or "conditioning gate rejected" in message
        )
    return False


@torch.no_grad()
def multiblock_tsqr_dense_equivalence(
    *,
    seed: int = 91,
    block_sizes: Sequence[int] = (5, 7, 13, 41, 100),
    like: Tensor | None = None,
) -> dict[str, float | int]:
    """Non-divisible streamed-QR adversary against independent dense QR."""

    if like is None:
        like = torch.empty((), dtype=torch.float64)
    generator = torch.Generator(device=like.device).manual_seed(seed)
    output = torch.randn(5, 9, generator=generator, dtype=like.dtype, device=like.device)
    left = torch.randn(9, 6, generator=generator, dtype=like.dtype, device=like.device)
    right = torch.randn(9, 7, generator=generator, dtype=like.dtype, device=like.device)
    construction = MaterializationTelemetry()
    core = _make_cp_core(output, left, right, "multiblock_tsqr_adversary", construction)
    factor_error = q_error = reconstruction_error = 0.0
    minimum_blocks = math.inf
    maximum_blocks = 0
    for block_size in block_sizes:
        streamed_meter = MaterializationTelemetry()
        dense_meter = MaterializationTelemetry()
        factor, canonical, diagnostics = _direct_rq_cp_streamed(
            core.clone(), streamed_meter, block_size=block_size
        )
        dense_factor, dense_q, dense_diagnostics = _independent_dense_clone_rq(
            core.clone(), dense_meter
        )
        factor_error = max(
            factor_error, _scaled_matrix_relative_error(factor, dense_factor)
        )
        q_error = max(q_error, _tiny_core_relative_error(canonical, dense_q))
        reconstruction_error = max(
            reconstruction_error,
            diagnostics.factorization_relative_error,
            dense_diagnostics.factorization_relative_error,
        )
        minimum_blocks = min(minimum_blocks, diagnostics.streamed_column_blocks)
        maximum_blocks = max(maximum_blocks, diagnostics.streamed_column_blocks)
    return {
        "cases": len(tuple(block_sizes)),
        "minimum_streamed_column_blocks": int(minimum_blocks),
        "maximum_streamed_column_blocks": int(maximum_blocks),
        "maximum_factor_relative_error": factor_error,
        "maximum_q_core_relative_error": q_error,
        "maximum_reconstruction_relative_error": reconstruction_error,
    }


def telemetry_dict(
    telemetry: MaterializationTelemetry,
) -> dict[str, int | float]:
    return {
        "raw_order_three_core_materializations": telemetry.raw_order_three_core_materializations,
        "raw_width_cubic_core_materializations": telemetry.raw_width_cubic_core_materializations,
        "maximum_persistent_tensor_elements": telemetry.maximum_persistent_tensor_elements,
        "maximum_temporary_tensor_elements": telemetry.maximum_temporary_tensor_elements,
        "maximum_temporary_tensor_order": telemetry.maximum_temporary_tensor_order,
        "local_direct_rq_factorizations": telemetry.local_direct_rq_factorizations,
        "householder_qr_kernel_calls": telemetry.householder_qr_kernel_calls,
        "svd_calls": telemetry.svd_calls,
        "polar_calls": telemetry.polar_calls,
        "normal_equation_factorizations": telemetry.normal_equation_factorizations,
        "fused_ffn_primitive_count": telemetry.fused_ffn_primitive_count,
        "ephemeral_ffn_bond_diagonalizations": telemetry.ephemeral_ffn_bond_diagonalizations,
        "rectangular_direct_rq_factorizations": telemetry.rectangular_direct_rq_factorizations,
        "maximum_rectangular_q_elements": telemetry.maximum_rectangular_q_elements,
        "bounded_explicit_direct_rq_factorizations": (
            telemetry.bounded_explicit_direct_rq_factorizations
        ),
        "maximum_bounded_explicit_q_elements": (
            telemetry.maximum_bounded_explicit_q_elements
        ),
        "constant_unary_folds": telemetry.constant_unary_folds,
        "constant_cp_to_unary_folds": telemetry.constant_cp_to_unary_folds,
        "constant_cp_to_constant_folds": telemetry.constant_cp_to_constant_folds,
        "direct_q_provenance_certificates": (
            telemetry.direct_q_provenance_certificates
        ),
        "streamed_direct_q_provenance_certificates": (
            telemetry.streamed_direct_q_provenance_certificates
        ),
        "streamed_direct_q_replay_qr_factorizations": (
            telemetry.streamed_direct_q_replay_qr_factorizations
        ),
        "direct_q_columns_compared": telemetry.direct_q_columns_compared,
        "maximum_direct_q_transition_elements": (
            telemetry.maximum_direct_q_transition_elements
        ),
        "maximum_direct_q_panel_elements": telemetry.maximum_direct_q_panel_elements,
        "maximum_direct_q_compact_relative_error": (
            telemetry.maximum_direct_q_compact_relative_error
        ),
        "maximum_direct_q_reconstruction_relative_error": (
            telemetry.maximum_direct_q_reconstruction_relative_error
        ),
    }


def audit_algorithm1_factorization_calls() -> dict[str, object]:
    """Fail-closed module-wide AST audit of decomposition call sites."""

    prohibited_everywhere = {
        "svd",
        "svdvals",
        "pinv",
        "lstsq",
        "cholesky",
        "matrix_power",
        "polar",
        "inv",
        "inverse",
    }
    evd_calls = {"eig", "eigh", "eigvals", "eigvalsh"}
    allowed_evd_functions = {
        "diagonalize_implicit_dag_full_rank",
        "apply_shared_eigenbases_to_explicit_clone_occurrences_control",
        "diagonalize_shared_and_explicit_clone_independently",
    }
    tree = ast.parse(inspect.getsource(sys.modules[__name__]))
    observed: list[str] = []
    rejected: list[str] = []
    qr_calls = 0

    class CallAudit(ast.NodeVisitor):
        def __init__(self) -> None:
            self.functions: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        def visit_Call(self, node: ast.Call) -> None:
            nonlocal qr_calls
            target = node.func
            name = target.attr if isinstance(target, ast.Attribute) else (
                target.id if isinstance(target, ast.Name) else "<dynamic>"
            )
            observed.append(name)
            if name == "qr" or name == "_positive_diagonal_direct_rq_rows":
                qr_calls += 1
            current = self.functions[-1] if self.functions else "<module>"
            if name in prohibited_everywhere:
                rejected.append(f"{current}:{name}")
            if name in evd_calls and current not in allowed_evd_functions:
                rejected.append(f"{current}:{name}")
            self.generic_visit(node)

    CallAudit().visit(tree)
    if qr_calls == 0:
        rejected.append("module:no_direct_qr_reachable")
    if rejected:
        raise RuntimeError(
            "Algorithm 1 contains prohibited factorization calls: "
            + ", ".join(sorted(set(rejected)))
        )
    return {
        "scope": "entire_module",
        "observed_calls": sorted(set(observed)),
        "prohibited_calls_found": sorted(set(rejected)),
        "static_direct_qr_call_sites": qr_calls,
        "direct_qr_required": True,
    }


__all__ = [
    "BOUNDED_EXPLICIT_RQ_METHOD",
    "CLAIM_BOUNDARY",
    "COMPACT_RELATIVE_VALUE_FLOORS",
    "COMPACT_TRACE_RETENTION_TARGETS",
    "DIRECT_RQ_METHOD",
    "RECTANGULAR_RQ_METHOD",
    "UNARY_RQ_METHOD",
    "CanonicalCloneTrace",
    "CanonicalImplicitDAG",
    "CompactSpectrumRecord",
    "CloneEVDTrace",
    "DiagonalImplicitDAG",
    "ImplicitBlockOracle",
    "IndependentCloneEVDTrace",
    "ImplicitProjectiveDAG",
    "PhysicalSourceSpec",
    "ProjectiveConstant",
    "ProjectiveValue",
    "RawPhysicalInput",
    "ReducedQRBinaryCore",
    "MaterializationTelemetry",
    "TinyDenseEquivalence",
    "audit_algorithm1_factorization_calls",
    "assert_norm_buffers_unchanged",
    "apply_shared_eigenbases_to_explicit_clone_occurrences_control",
    "canonicalize_implicit_dag_direct_rq",
    "canonicalize_with_explicit_clone_step_trace",
    "compare_shared_and_explicit_clone_environments",
    "compile_implicit_projective_block_boundary",
    "diagonalize_implicit_dag_full_rank",
    "diagonalize_shared_and_explicit_clone_independently",
    "drop_first_residual_add_cross_term",
    "evaluate_boundary_quotient",
    "evaluate_projective_boundary",
    "evaluate_scaled_boundary",
    "implicit_shape_statistics",
    "multiblock_tsqr_dense_equivalence",
    "old_project_then_add_topology_is_rejected",
    "predict_direct_rq_route_inventory",
    "reverse_implicit_environments",
    "source_block_boundary",
    "telemetry_dict",
    "tiny_dense_direct_rq_equivalence",
]
