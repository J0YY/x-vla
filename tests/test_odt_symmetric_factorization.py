"""Bounded direct-QR gates. Run only through the authenticated runtime guard."""

from unittest.mock import patch

import pytest
import torch

from xvla.train.odt_engine_v2 import factorization as kernel
from xvla.train.odt_engine_v2.types import CPBinaryCore, MaterializationTelemetry, UnaryCore


def _cp(tensor):
    """Literal coefficient atoms, with no decomposition constructor."""
    tensor = torch.as_tensor(tensor, dtype=torch.float64)
    output, left, right = tensor.shape
    left_atoms = torch.eye(left, dtype=torch.float64).repeat_interleave(right, dim=0)
    right_atoms = torch.eye(right, dtype=torch.float64).repeat(left, 1)
    return CPBinaryCore(tensor.reshape(output, -1), left_atoms, right_atoms, "test")


def _expanded_factor(factor):
    exponent = torch.tensor(factor.binary_exponent, dtype=torch.int64)
    return torch.ldexp(factor.mantissa, exponent)


def _factor(core, *, tied=True, block_size=2):
    telemetry = MaterializationTelemetry()
    result = kernel._direct_rq_core(core, telemetry, block_size=block_size, tied_inputs=tied)
    assert telemetry.householder_qr_kernel_calls == 1
    assert result[2].literal_q_chart_resolved
    assert result[2].minimum_diagonal_to_maximum_entry == 0.0
    return result, telemetry


def test_weighted_symmetric_coordinates_have_the_expected_literal_q():
    tensor = torch.zeros(3, 2, 2, dtype=torch.float64)
    tensor[0, 0, 0], tensor[1, 0, 1], tensor[1, 1, 0], tensor[2, 1, 1] = 2, 3, 3, 4
    (factor, q, diagnostics), _ = _factor(_cp(tensor))
    expected = torch.zeros_like(tensor)
    expected[0, 0, 0], expected[2, 1, 1] = 1, 1
    expected[1, 0, 1] = expected[1, 1, 0] = 2 ** -0.5
    torch.testing.assert_close(q.q_rows.reshape_as(expected), expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(_expanded_factor(factor) @ q.q_rows, tensor.reshape(3, -1))
    assert diagnostics.unfolding_shape == (3, 3)
    assert diagnostics.direct_q_provenance.columns_compared == 4


@pytest.mark.parametrize("zero", [False, True])
def test_deficient_and_zero_completions_remain_symmetric(zero):
    tensor = torch.zeros(5, 2, 2, dtype=torch.float64)
    if not zero:
        tensor[0, 0, 0], tensor[1, 0, 0] = 1, 2
    (factor, q, diagnostics), telemetry = _factor(_cp(tensor))
    assert q.output_dimension == 3  # Shape, never the numerical rank 0 or 1.
    assert diagnostics.structurally_reduced
    assert telemetry.maximum_bounded_explicit_q_elements == 12
    rows = q.q_rows.reshape(3, 2, 2)
    torch.testing.assert_close(rows, rows.transpose(1, 2), rtol=0, atol=0)
    torch.testing.assert_close(_expanded_factor(factor) @ q.q_rows, tensor.reshape(5, -1))
    kernel._validate_tied_input_symmetry(q, block_size=1)
    if zero:
        expected = torch.tensor([[1., 0., 0., 0.], [0., 2 ** -0.5, 2 ** -0.5, 0.],
                                 [0., 0., 0., 1.]], dtype=torch.float64)
        torch.testing.assert_close(q.q_rows, expected, rtol=1e-14, atol=1e-14)


def test_asymmetric_raw_tied_core_fails_before_qr():
    tensor = torch.zeros(2, 2, 2, dtype=torch.float64)
    tensor[0, 0, 1] = 1
    with patch.object(torch.linalg, "qr", side_effect=AssertionError("QR must not run")):
        with pytest.raises(ValueError, match="raw tied core is asymmetric"):
            kernel._direct_rq_core(_cp(tensor), MaterializationTelemetry(), block_size=1,
                                   tied_inputs=True)


def test_distinct_inputs_keep_the_ordered_asymmetric_tensor():
    tensor = torch.zeros(2, 2, 2, dtype=torch.float64)
    tensor[0, 0, 1], tensor[1, 1, 0] = 1, 2
    (factor, q, diagnostics), _ = _factor(_cp(tensor), tied=False)
    torch.testing.assert_close(_expanded_factor(factor) @ q.q_rows, tensor.reshape(2, -1))
    assert diagnostics.unfolding_shape == (2, 4)


def test_unpacked_q_storage_is_part_of_the_shape_gate():
    with patch.object(kernel, "MAXIMUM_BOUNDED_EXPLICIT_Q_ELEMENTS", 11):
        assert kernel._bounded_explicit_q_allowed_shape(4, 3)
        assert not kernel._bounded_explicit_q_allowed_shape(4, 3, storage_columns=4)
        with patch.object(torch.linalg, "qr", side_effect=AssertionError("QR must not run")):
            with pytest.raises(ValueError, match="unsupported streamed"):
                kernel._direct_rq_core(_cp(torch.zeros(4, 2, 2)), MaterializationTelemetry(),
                                       block_size=2, tied_inputs=True)


def test_unsupported_unary_and_streamed_shapes_fail_before_qr():
    with patch.object(kernel, "MAXIMUM_BOUNDED_EXPLICIT_UNFOLDING_ELEMENTS", 2):
        with patch.object(torch.linalg, "qr", side_effect=AssertionError("QR must not run")):
            with pytest.raises(ValueError, match="unsupported retained direct-Q unary"):
                kernel._direct_rq_core(UnaryCore(torch.eye(2, dtype=torch.float64), "test"),
                                       MaterializationTelemetry(), block_size=2)
            with pytest.raises(ValueError, match="unsupported streamed"):
                kernel._direct_rq_cp_streamed(_cp(torch.zeros(1, 2, 2)),
                                              MaterializationTelemetry(), block_size=2)


def test_raw_symmetry_validation_does_not_materialize_unbounded_tensor():
    vector = torch.arange(20, dtype=torch.float64).reshape(1, 20) / 20
    core = CPBinaryCore(torch.ones(2, 1, dtype=torch.float64), vector, vector.clone(), "large")
    with patch.object(kernel, "MAXIMUM_BOUNDED_EXPLICIT_UNFOLDING_ELEMENTS", 64):
        with patch.object(torch.linalg, "qr", side_effect=AssertionError("QR must not run")):
            kernel._validate_tied_input_symmetry(core, block_size=7)


def test_original_385_by_386_deficient_rectangular_regression():
    core = CPBinaryCore(torch.ones(385, 1, dtype=torch.float64),
                        torch.ones(1, 1, dtype=torch.float64),
                        torch.linspace(0., 1., 386, dtype=torch.float64).reshape(1, 386),
                        "pade_regression", binary_exponent=3)
    (factor, q, diagnostics), _ = _factor(core, tied=False, block_size=256)
    expected = 8 * core.output_factor @ core.right_factor
    torch.testing.assert_close(_expanded_factor(factor) @ q.q_rows, expected,
                               rtol=3e-12, atol=3e-12)
    assert q.q_rows.shape == (385, 386)
    assert diagnostics.literal_q_chart_resolved


def test_nonfinite_unfolding_is_rejected_before_qr():
    matrix = torch.eye(2, dtype=torch.float64)
    matrix[0, 0] = torch.nan
    with patch.object(torch.linalg, "qr", side_effect=AssertionError("QR must not run")):
        with pytest.raises(ValueError, match="finite nonempty"):
            kernel._positive_diagonal_direct_rq_rows(matrix)
