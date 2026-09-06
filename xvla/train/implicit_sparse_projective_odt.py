"""Compatibility exports for the former monolithic ODT engine.

The mathematical sweeps live in xvla.train.odt_engine_v2.core. Compilation, tensor
representations, scale validation, reports and independent oracles are separate.
New canonicalization uses the explicitly symmetric ordered lift v2. Historical
artifacts remain readable, but their old environments do not validate this lift.
Shared-basis ranking is an extension, not occurrence-specific tree ODT.
This refactor does not authorize production launches.
"""

from xvla.train.odt_engine_v2.compiler import (
    _Builder,
    _add_pair_vectors,
    _affine_pair,
    _concatenate_pair_vectors,
    _dot_pair_vectors,
    _fused_cp_ffn_pair,
    _identity_pair_leaf,
    _mean_square_pair,
    _multiply_pair_scalars,
    _norm_snapshot,
    _pade_norm_pair,
    _pade_pair,
    _scale_pair_vector,
    _scale_pair_vector_by_scalar,
    _validate_block_source,
    assert_norm_buffers_unchanged,
    compile_implicit_projective_block_boundary,
    source_block_boundary,
)

from xvla.train.odt_engine_v2.constants import (
    BOUNDED_EXPLICIT_RQ_METHOD,
    CLAIM_BOUNDARY,
    COMPACT_RELATIVE_VALUE_FLOORS,
    COMPACT_TRACE_RETENTION_TARGETS,
    DIRECT_Q_PROVENANCE_TOLERANCE,
    DIRECT_RQ_METHOD,
    LOCAL_SCALE_CERTIFICATION_TOLERANCE,
    MAXIMUM_BOUNDED_EXPLICIT_Q_ELEMENTS,
    MAXIMUM_BOUNDED_EXPLICIT_UNFOLDING_ELEMENTS,
    MAXIMUM_RECTANGULAR_Q_ELEMENTS,
    RECTANGULAR_RQ_METHOD,
    STREAMED_DIRECT_Q_PROVENANCE_METHOD,
    STREAMED_RQ_MINIMUM_PIVOT_TO_MAXIMUM_ENTRY,
    UNARY_RQ_METHOD,
)

from xvla.train.odt_engine_v2.core import (
    _push_factor_to_parents,
    _push_factor_to_parents_with_scale_ledger,
    canonicalize_implicit_dag_direct_rq,
    reverse_implicit_environments,
)

from xvla.train.odt_engine_v2.compat import diagonalize_implicit_dag_full_rank

from xvla.train.odt_engine_v2.diagnostics import (
    _compact_spectrum_record,
    audit_algorithm1_factorization_calls,
    implicit_shape_statistics,
    predict_direct_rq_route_inventory,
    telemetry_dict,
)

from xvla.train.odt_engine_v2.factorization import (
    _bounded_explicit_q_allowed_by_structure,
    _bounded_explicit_q_allowed_shape,
    _direct_rq_bounded_cp,
    _direct_rq_core,
    _direct_rq_cp,
    _direct_rq_cp_streamed,
    _direct_rq_unary,
    _positive_diagonal_direct_rq_rows,
    _retained_direct_q_provenance,
)

from xvla.train.odt_engine_v2.graph import (
    _clone_network,
    _parent_occurrences,
    _prepare_physical_input,
    _topological_root_first,
    _validate_network,
    _walk_unique,
    evaluate_boundary_quotient,
    evaluate_projective_boundary,
    evaluate_scaled_boundary,
    validate_canonical_exponent_normal_form,
)

from xvla.train.odt_engine_v2.ops import (
    _absorb_input_factor,
    _accumulate_scaled_message,
    _add_scaled_matrices,
    _apply_output_basis,
    _core_apply,
    _core_arity,
    _core_clone,
    _core_column_blocks,
    _cp_column_blocks,
    _ldexp,
    _make_cp_core,
    _make_unary_core,
    _normalize_absorbed,
    _normalize_batch,
    _normalize_projective_batch,
    _offdiagonal_ratio,
    _role_environment,
    _scale_symmetric,
    _strip_power_of_two,
)

from xvla.train.odt_engine_v2.oracles import (
    _aggregate_clone_environment_records,
    _eigenvalue_relative_error,
    _independent_dense_clone_rq,
    _tiny_core_mantissa,
    _tiny_core_relative_error,
    canonicalize_with_explicit_clone_step_trace,
    compare_shared_and_explicit_clone_environments,
    diagonalize_shared_and_explicit_clone_independently,
    drop_first_residual_add_cross_term,
    multiblock_tsqr_dense_equivalence,
    old_project_then_add_topology_is_rejected,
    tiny_dense_direct_rq_equivalence,
)

from xvla.train.odt_engine_v2.types import (
    AbsorptionScaleLedger,
    CPBinaryCore,
    CanonicalCloneTrace,
    CanonicalImplicitDAG,
    CanonicalStepRecord,
    CloneStepRecord,
    CompactSpectrumRecord,
    DenseCloneCore,
    DiagonalImplicitDAG,
    DirectQProvenance,
    DirectRQDiagnostics,
    EnvironmentRecord,
    ImplicitBlockOracle,
    ImplicitCore,
    ImplicitNode,
    ImplicitProjectiveDAG,
    IndependentCloneEVDTrace,
    MaterializationTelemetry,
    PhysicalSourceSpec,
    ProjectiveConstant,
    ProjectiveValue,
    RawPhysicalInput,
    ReducedQRBinaryCore,
    ScaledBatch,
    ScaledMatrix,
    Tensor,
    TinyDenseEquivalence,
    UnaryCore,
    _core_output_dimension,
    _validate_real_finite,
)

from xvla.train.odt_engine_v2.validation import (
    _absorption_scale_audit,
    _input_occurrence_matrix,
    _local_direct_rq_scaled_reconstruction_audit,
    _normalized_scaled_matrix_chart,
    _projective_batch_relative_error,
    _scaled_matrix_audit,
    _scaled_matrix_relative_error,
    _validate_direct_q_provenance,
)

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
