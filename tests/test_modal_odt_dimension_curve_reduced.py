from types import SimpleNamespace

import pytest
import torch

from modal_odt_dimension_curve_reduced import evaluate_valid_rows


def test_bisection_isolates_one_undefined_chart_and_preserves_valid_rows():
    class Mapped:
        def evaluate_boundary_quotient(self, raw, return_receipt):
            if bool((raw["state"][:, 0] == 2).any()):
                raise ValueError("mapped projective denominator is numerically zero")
            values = raw["state"][:, :1].repeat(1, 56)
            return values, SimpleNamespace(root_only_live=True, all_refcounts_zero=True,
                node_evaluations=5, edge_occurrences_emitted=6, edge_ledger_sha256="a" * 64, peak_live_values=3)
    raw = {"state": torch.arange(4, dtype=torch.float64).reshape(4, 1)}
    values, failed, receipts = evaluate_valid_rows(Mapped(), raw, [0, 1, 2, 3])
    assert failed == [2]
    assert set(values) == {0, 1, 3}
    assert values[3].shape == (8, 7)
    assert len(receipts) == 3


@pytest.mark.parametrize("error", [ValueError("artifact manifest digest differs from the trusted digest"),
    ValueError("mapped physical inputs must be CPU float64 tensors"), RuntimeError("kernel failed"), MemoryError("OOM")])
def test_integrity_shape_kernel_and_resource_errors_are_not_policy_failures(error):
    class Mapped:
        def evaluate_boundary_quotient(self, *_args, **_kwargs):
            raise error
    with pytest.raises(type(error), match=str(error)):
        evaluate_valid_rows(Mapped(), {"state": torch.ones(2, 1)}, [0, 1])
