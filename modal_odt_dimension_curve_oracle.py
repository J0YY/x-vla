"""Per-artifact, real-panel equivalence gate for the compiled arithmetic lane."""
from __future__ import annotations

import dataclasses
import gc
import time

import torch

from modal_odt_dimension_curve_reduced import CHART_FAILURE_MESSAGES
from xvla.train.implicit_projective_dag_mapped import _clear_exception_graph_tracebacks
from xvla.train.implicit_projective_dag_numba import CompiledMappedImplicitProjectiveDAG, DEFAULT_MAXIMUM_ARENA_BYTES


def compare_real_panel(reference, compiled, raw: dict, action_mean, action_std) -> dict:
    """Compare all initial rows, preserving equal chart failures as outcomes."""
    batch = next(iter(raw.values())).shape[0]
    if compiled.arena_bytes_for_batch(batch) + batch * 57 * 8 > DEFAULT_MAXIMUM_ARENA_BYTES:
        raise RuntimeError("real-panel compiled arena exceeds its byte cap")
    valid, invalid, attempts = [], [], []
    maxima = {"projective_max_abs_error": 0., "normalized_action_max_abs_error": 0., "action_max_abs_error": 0.}
    mean = torch.as_tensor(action_mean, dtype=torch.float64)
    standard_deviation = torch.as_tensor(action_std, dtype=torch.float64)
    if mean.shape != (7,) or standard_deviation.shape != (7,) or not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(standard_deviation).all()) or not bool((standard_deviation > 0).all()):
        raise RuntimeError("real-panel action normalization differs")
    mean, standard_deviation = mean.repeat(8), standard_deviation.repeat(8)
    def evaluate(executor, selected):
        before = time.monotonic()
        try:
            pair, receipt = executor.evaluate_projective_boundary(selected, return_receipt=True)
        except ValueError as error:
            if str(error) not in CHART_FAILURE_MESSAGES:
                raise
            return None, None, str(error), time.monotonic() - before
        return pair, receipt, None, time.monotonic() - before
    def compare(rows):
        selected = {key: value[rows] for key, value in raw.items()}
        expected, expected_receipt, expected_error, old_seconds = evaluate(reference, selected)
        actual, actual_receipt, actual_error, native_seconds = evaluate(compiled, selected)
        attempt = {"rows": rows, "mapped_seconds": old_seconds, "compiled_seconds": native_seconds,
            "mapped_chart_error": expected_error, "compiled_chart_error": actual_error}
        attempts.append(attempt)
        if expected_error != actual_error:
            raise RuntimeError("compiled and mapped real-panel chart outcomes differ")
        if expected_error is not None:
            if len(rows) == 1:
                invalid.extend(rows)
                return
            middle = len(rows) // 2
            compare(rows[:middle])
            compare(rows[middle:])
            return
        if expected.shape != actual.shape or tuple(expected.shape) != (len(rows), 57):
            raise RuntimeError("compiled and mapped real-panel output shapes differ")
        if not bool(torch.isfinite(expected).all()) or not bool(torch.isfinite(actual).all()):
            raise RuntimeError("real-panel projective outputs escaped the finite contract")
        ledger_fields = ("node_evaluations", "edge_occurrences_emitted", "released_nonroot_nodes",
            "peak_live_values", "edge_ledger_sha256", "root_only_live", "all_refcounts_zero")
        if any(getattr(expected_receipt, name) != getattr(actual_receipt, name) for name in ledger_fields):
            raise RuntimeError("compiled and mapped real-panel occurrence ledgers differ")
        if not actual_receipt.root_only_live or not actual_receipt.all_refcounts_zero:
            raise RuntimeError("real-panel occurrence ledger did not close")
        attempt["mapped_receipt"] = dataclasses.asdict(expected_receipt)
        attempt["compiled_receipt"] = dataclasses.asdict(actual_receipt)
        # Independent power-of-two charts can differ by exactly two at an
        # exponent boundary. Compare directions in a common positive max chart.
        expected_direction = expected / expected.abs().amax(dim=1, keepdim=True)
        actual_direction = actual / actual.abs().amax(dim=1, keepdim=True)
        error = float((actual_direction - expected_direction).abs().max())
        maxima["projective_max_abs_error"] = max(maxima["projective_max_abs_error"], error)
        if error > 2e-10:
            raise RuntimeError("compiled and mapped real-panel projective boundaries differ")
        threshold = 100. * torch.finfo(torch.float64).eps
        def bad_chart(pair):
            relative = pair[:, -1].abs() / pair.abs().amax(dim=1).clamp_min(torch.finfo(pair.dtype).tiny)
            return relative <= threshold
        old_bad, native_bad = bad_chart(expected), bad_chart(actual)
        if not bool(torch.equal(old_bad, native_bad)):
            raise RuntimeError("compiled and mapped real-panel denominator validity differs")
        for position, row in enumerate(rows):
            if bool(old_bad[position]):
                invalid.append(row)
                continue
            old_action = expected[position, :-1] / expected[position, -1]
            native_action = actual[position, :-1] / actual[position, -1]
            difference = (native_action - old_action).abs()
            old_raw = old_action * standard_deviation + mean
            native_raw = native_action * standard_deviation + mean
            if not bool(torch.equal(old_raw[6::7] > 0, native_raw[6::7] > 0)):
                raise RuntimeError("compiled and mapped real-panel gripper decisions differ")
            normalized_error = float(difference.max())
            action_error = float((native_raw - old_raw).abs().max())
            maxima["normalized_action_max_abs_error"] = max(maxima["normalized_action_max_abs_error"], normalized_error)
            maxima["action_max_abs_error"] = max(maxima["action_max_abs_error"], action_error)
            if normalized_error > 2e-10 or action_error > 2e-10:
                raise RuntimeError("compiled and mapped real-panel actions differ")
            valid.append(row)
    compare(list(range(batch)))
    if sorted(valid + invalid) != list(range(batch)):
        raise RuntimeError("real-panel oracle row ledger is incomplete")
    return {"schema": "xvla_real_panel_compiled_mapped_oracle_v1", "passed": True, "batch": batch,
        "valid_rows": sorted(valid), "invalid_chart_rows": sorted(invalid), "attempts": attempts, **maxima,
        "mapped_total_seconds": sum(value["mapped_seconds"] for value in attempts),
        "compiled_total_seconds": sum(value["compiled_seconds"] for value in attempts),
        "arena_width": compiled.arena_receipt.arena_width,
        "arena_bytes_batch": compiled.arena_bytes_for_batch(batch),
        "prebound_segmented_bytes": compiled.arena_receipt.segmented_bytes_bound}


class RealPanelCheckedCompiledExecutor:
    """Own one authenticated mapping, compare once, then use compiled only.

    The two interpreters share the same immutable mapped owner.  After the
    initial gate this wrapper has no runtime choice or source-policy fallback.
    """
    def __init__(self, mapped, action_mean, action_std, publish, manifest_sha256):
        if type(manifest_sha256) is not str or len(manifest_sha256) != 64 or manifest_sha256 != mapped.manifest_sha256:
            raise RuntimeError("compiled wrapper manifest binding differs from authenticated owner")
        self._reference = mapped
        try:
            self._compiled = CompiledMappedImplicitProjectiveDAG(mapped)
        except BaseException as error:
            _clear_exception_graph_tracebacks(error)
            gc.collect()
            raise
        self._action_std = action_std
        self._action_mean = action_mean
        self._publish = publish
        self._verified = False
        self._manifest_sha256 = mapped.manifest_sha256

    @property
    def execution_layout_supported(self):
        return self._compiled.execution_layout_supported

    @property
    def segmented_bytes_per_evaluation(self):
        return self._compiled.segmented_bytes_per_evaluation

    @property
    def verified(self):
        return self._verified

    def evaluate_boundary_quotient(self, raw, *, return_receipt=False):
        if not self._verified:
            if next(iter(raw.values())).shape[0] != 20:
                raise RuntimeError("first real-panel oracle must contain all 20 paired rows")
            receipt = compare_real_panel(self._reference, self._compiled, raw, self._action_mean, self._action_std)
            receipt["artifact_manifest_sha256"] = self._manifest_sha256
            self._publish("real_panel_oracle.json", receipt)
            self._reference = None
            self._verified = True
        return self._compiled.evaluate_boundary_quotient(raw, return_receipt=return_receipt)

    def close(self):
        self._reference = None
        self._compiled.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
