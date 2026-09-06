"""Stable tensor, graph and record schemas. No compiler or numerical sweeps."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Sequence

import torch

from xvla.train.odt_engine_v2.constants import (
    CLAIM_BOUNDARY,
)


Tensor = torch.Tensor


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


def _validate_real_finite(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point() or value.is_complex():
        raise TypeError(f"{name} must be a real floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


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
    ``m`` by ``n`` unfolding its output dimension is selected by shape. For
    tied symmetric inputs the available coordinate count is d*(d+1)//2,
    not d*d. Keeping
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


def _core_output_dimension(core: ImplicitCore) -> int:
    return core.output_dimension


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
