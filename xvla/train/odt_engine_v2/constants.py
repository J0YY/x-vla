"""Versioned numerical and representation contracts."""

from __future__ import annotations




DIRECT_RQ_METHOD = "streamed_householder_direct_reduced_rq"
UNARY_RQ_METHOD = "torch_qr_transpose_direct_reduced_rq"
BOUNDED_EXPLICIT_RQ_METHOD = "bounded_explicit_householder_direct_reduced_rq"
RECTANGULAR_RQ_METHOD = BOUNDED_EXPLICIT_RQ_METHOD
MAXIMUM_BOUNDED_EXPLICIT_UNFOLDING_ELEMENTS = 4_000_000
MAXIMUM_BOUNDED_EXPLICIT_Q_ELEMENTS = 4_000_000
LOCAL_SCALE_CERTIFICATION_TOLERANCE = 3e-10
DIRECT_Q_PROVENANCE_TOLERANCE = 3e-10
STREAMED_RQ_MINIMUM_PIVOT_TO_MAXIMUM_ENTRY = 1e-12
STREAMED_DIRECT_Q_PROVENANCE_METHOD = (
    "two_pass_sequential_tsqr_explicit_q_panel_replay"
)
MAXIMUM_RECTANGULAR_Q_ELEMENTS = MAXIMUM_BOUNDED_EXPLICIT_Q_ELEMENTS
LIFT_ID = "dooms_symmetric_ordered_lift_v2"
CLAIM_BOUNDARY = LIFT_ID + ": " + (
    "Direct-QR shared-basis ODT extension of one selected-token boundary of one "
    "unchanged Padé ChiTransformerBlock, in the clone-unfolded syntactic "
    "coefficient metric. Environments sum independent occurrence-cut losses, "
    "not simultaneous tied-truncation or action loss. Canonical cores retain "
    "explicit shape-selected Q. The CP FFN rank is eliminated inside one "
    "fused primitive. This is not an "
    "identified-variable or quotient-intrinsic metric, not compression, and "
    "not yet a traversal of every block and final policy head."
)
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
