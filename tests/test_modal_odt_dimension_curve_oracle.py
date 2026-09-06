from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from modal_odt_dimension_curve_oracle import compare_real_panel
from xvla.train.implicit_projective_dag_mapped import MappedExecutionReceipt


@dataclass(frozen=True)
class Receipt(MappedExecutionReceipt):
    arena_width: int = 12
    arena_bytes: int = 1920


class Executor:
    def __init__(self, *, multiplier=1., gripper=1., invalid=(), broken=(), exception=None, ledger="a"):
        self.multiplier, self.gripper = multiplier, gripper
        self.invalid, self.broken, self.exception, self.ledger = invalid, broken, exception, ledger
        self.arena_receipt = SimpleNamespace(arena_width=12, segmented_bytes_bound=0)

    def arena_bytes_for_batch(self, batch):
        return batch * 12 * 8

    def evaluate_projective_boundary(self, raw, *, return_receipt):
        if self.exception is not None:
            raise self.exception
        rows = raw["row"].reshape(-1).tolist()
        if any(row in self.broken for row in rows):
            raise ValueError("mapped projective evaluation produced zero/nonfinite coordinates")
        pair = torch.ones(len(rows), 57, dtype=torch.float64) * self.multiplier
        pair[:, 6:56:7] = self.gripper * self.multiplier
        for index, row in enumerate(rows):
            if row in self.invalid:
                pair[index, -1] = 0.
        return pair, Receipt(node_evaluations=8, edge_occurrences_emitted=10,
            released_nonroot_nodes=7, peak_live_values=4, edge_ledger_sha256=self.ledger,
            root_only_live=True, all_refcounts_zero=True, segmented_tensor_gathers=0, segmented_bytes_gathered=0)


def raw():
    return {"row": torch.arange(20).reshape(20, 1)}


def test_common_positive_chart_handles_power_of_two_boundary():
    receipt = compare_real_panel(Executor(multiplier=.9999999999999999), Executor(multiplier=.5), raw(), [0.] * 7, [1.] * 7)
    assert receipt["passed"] and receipt["projective_max_abs_error"] < 1e-15
    assert receipt["valid_rows"] == list(range(20))


def test_equal_numerical_chart_failures_remain_explicit_policy_outcomes():
    receipt = compare_real_panel(Executor(invalid=(2,), broken=(4,)), Executor(invalid=(2,), broken=(4,)), raw(), [0.] * 7, [1.] * 7)
    assert receipt["invalid_chart_rows"] == [2, 4]
    assert len(receipt["valid_rows"]) == 18


def test_different_chart_outcomes_fail_closed():
    with pytest.raises(RuntimeError, match="projective boundaries differ"):
        compare_real_panel(Executor(invalid=(2,)), Executor(), raw(), [0.] * 7, [1.] * 7)


def test_tiny_continuous_difference_cannot_flip_decoded_gripper():
    with pytest.raises(RuntimeError, match="gripper decisions differ"):
        compare_real_panel(Executor(gripper=0.), Executor(gripper=1e-12), raw(), [0.] * 7, [1.] * 7)


def test_integrity_and_kernel_errors_are_never_chart_failures():
    with pytest.raises(RuntimeError, match="integrity"):
        compare_real_panel(Executor(exception=RuntimeError("integrity")), Executor(), raw(), [0.] * 7, [1.] * 7)


def test_occurrence_ledger_mismatch_fails_closed():
    with pytest.raises(RuntimeError, match="ledgers differ"):
        compare_real_panel(Executor(), Executor(ledger="wrong"), raw(), [0.] * 7, [1.] * 7)


@pytest.mark.parametrize("deviation", [[1.] * 6, [0.] * 7, [float("inf")] * 7])
def test_invalid_action_normalization_rejected(deviation):
    with pytest.raises(RuntimeError, match="normalization differs"):
        compare_real_panel(Executor(), Executor(), raw(), [0.] * 7, deviation)


def test_failed_compiled_bind_releases_exported_mapping_and_preserves_error(monkeypatch, tmp_path):
    import mmap
    import numpy as np
    import modal_odt_dimension_curve_oracle as oracle
    target = tmp_path / "small.raw"
    target.write_bytes(bytes(8))
    with target.open("rb") as stream:
        mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    def broken_constructor(value):
        exported = np.frombuffer(value.mapping, dtype=np.float64)
        assert exported.shape == (1,)
        raise RuntimeError("intentional bind failure")
    monkeypatch.setattr(oracle, "CompiledMappedImplicitProjectiveDAG", broken_constructor)
    with pytest.raises(RuntimeError, match="intentional bind failure"):
        oracle.RealPanelCheckedCompiledExecutor(SimpleNamespace(mapping=mapped, manifest_sha256="0" * 64), [0.] * 7, [1.] * 7, lambda *_: None, "0" * 64)
    mapped.close()


def test_manifest_mismatch_rejected_before_compiled_binding(monkeypatch):
    import modal_odt_dimension_curve_oracle as oracle
    def must_not_bind(_):
        raise AssertionError("unauthenticated manifest reached compiled constructor")
    monkeypatch.setattr(oracle, "CompiledMappedImplicitProjectiveDAG", must_not_bind)
    with pytest.raises(RuntimeError, match="manifest binding differs"):
        oracle.RealPanelCheckedCompiledExecutor(SimpleNamespace(manifest_sha256="a" * 64), [0.] * 7, [1.] * 7, lambda *_: None, "b" * 64)
