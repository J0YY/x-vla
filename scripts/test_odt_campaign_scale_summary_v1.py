"""Guarded scale-ledger, exact-zero, source and cutoff reporting regressions."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from research.odt_reference.run_tests import install_guards, COUNTS

# Discovery must install guards before constructing any numerical fixture.
install_guards()

from scripts.odt_campaign_scale_summary_v1 import (
    allocation_summary, assert_guards, chart_log_amplitude, distribution,
    log_difference, panel_summary, source_audit, validate_cross_and_deltas, validate_scales,
)
from scripts.odt_campaign_summary_v1 import authenticated_json, digest, validate_predictions


def fixture(projective, scales):
    p = np.asarray(projective, dtype=np.float64)
    scale = np.asarray(scales, dtype=np.float64)
    margin = np.abs(p[:, -1])
    zero = np.max(np.abs(p), axis=1) == 0
    code = np.where(zero, 2, np.where(margin <= 1e-12, 1, 0)).astype(np.int8)
    valid = code == 0
    decoded = np.full((len(p), p.shape[1]-1), np.nan)
    decoded[valid] = p[valid, :-1]/p[valid, -1:]
    with np.errstate(divide="ignore"):
        result = {"projective": p, "output_log2_scale": scale,
            "log2_abs_denominator": scale+np.log2(margin),
            "log2_numerator_max_abs": scale+np.log2(np.max(np.abs(p[:, :-1]), axis=1)),
            "denominator_sign": np.sign(p[:, -1]).astype(np.int8),
            "relative_denominator": margin, "valid": valid, "failure_code": code, "decoded": decoded}
    validate_predictions(result, shape=decoded.shape)
    return result


def with_deltas(current, baseline):
    for name in ("log2_abs_denominator", "log2_numerator_max_abs"):
        current["delta_"+name] = log_difference(current[name], baseline[name])
    a, b = current["projective"], baseline["projective"]
    cross = a[:, :-1]*b[:, -1:]-b[:, :-1]*a[:, -1:]
    current["cross_product_normalized_mantissa"] = cross
    with np.errstate(divide="ignore"):
        current["log2_cross_product_numerator_max_abs"] = np.log2(np.max(np.abs(cross), axis=1))+current["output_log2_scale"]+baseline["output_log2_scale"]
    return current


class ScaleSummaryTests(unittest.TestCase):
    def setUp(self):
        assert_guards()

    def test_huge_negative_ledger_and_common_rescaling_leave_chart_unchanged(self):
        baseline = fixture([[1., .25, .5]], [-1e9])
        current = with_deltas(fixture([[1., .25, .5]], [-1e9-100.]), baseline)
        validate_scales(baseline)
        validate_scales(current)
        validate_cross_and_deltas(current, baseline)
        result = panel_summary(current, baseline, np.array([True]))
        self.assertEqual(result["delta_log2_abs_denominator"]["finite_median"], -100.)
        self.assertEqual(result["delta_log2_numerator_max_abs"]["finite_median"], -100.)
        self.assertEqual(result["delta_log2_output_max_abs_from_chart"]["finite_median"], 0.)
        self.assertEqual(result["cross_product_fixed_lift_log2_max_abs"]["negative_infinity_count"], 1)

    def test_exact_zeros_are_explicit_undefined_or_infinite_not_false_finite_statistics(self):
        baseline = fixture([[1., 0., 1.], [0., 0., 1.], [0., 0., 0.]], [-20000., -20000., -np.inf])
        current = with_deltas(fixture([[1., 0., 0.], [0., 0., 1.], [0., 0., 0.]], [-20001., -20002., -np.inf]), baseline)
        validate_scales(baseline)
        validate_scales(current)
        validate_cross_and_deltas(current, baseline)
        result = panel_summary(current, baseline, np.ones(3, dtype=bool))
        d, n = result["delta_log2_abs_denominator"], result["delta_log2_numerator_max_abs"]
        self.assertEqual((d["finite_count"], d["negative_infinity_count"], d["undefined_count"]), (1, 1, 1))
        self.assertEqual((n["finite_count"], n["undefined_count"]), (1, 2))
        self.assertEqual(result["log2_output_max_abs_from_chart"]["positive_infinity_count"], 1)
        self.assertEqual(result["denominator_sign"]["source_nonzero_to_zero"], 1)
        self.assertEqual(result["denominator_sign"]["both_zero"], 1)
        json.dumps(result, allow_nan=False)
        reverse = distribution(log_difference(baseline["log2_abs_denominator"], current["log2_abs_denominator"]))
        self.assertEqual(reverse["positive_infinity_count"], 1)

    def test_opposite_denominator_sign_recorded_independently_of_amplitude(self):
        baseline = fixture([[1., 0., .5]], [-20000.])
        current = with_deltas(fixture([[-1., 0., -.5]], [-20000.]), baseline)
        result = panel_summary(current, baseline, np.array([True]))
        self.assertEqual(result["denominator_sign"]["opposite_nonzero"], 1)
        self.assertEqual(result["delta_log2_output_max_abs_from_chart"]["finite_median"], 0.)

    def test_nan_positive_infinity_and_mismatched_zero_ledgers_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            values = fixture([[1., 0., .5]], [-20000.])
            values["output_log2_scale"][0] = bad
            with self.assertRaisesRegex(ValueError, "ledger"):
                validate_scales(values)
        values = fixture([[1., 0., .5]], [-20000.])
        values["log2_abs_denominator"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_scales(values)
        values = fixture([[1., 0., .5]], [-20000.])
        values["log2_abs_denominator"][0] += 1.
        with self.assertRaisesRegex(ValueError, "replay"):
            validate_scales(values)

    def test_stored_delta_cross_sign_and_chart_replays_fail_closed(self):
        baseline = fixture([[1., 0., .5]], [-20000.])
        for key in ("delta_log2_abs_denominator", "cross_product_normalized_mantissa"):
            current = with_deltas(fixture([[1., .5, .5]], [-20001.]), baseline)
            current[key].flat[0] += 1.
            with self.assertRaises(ValueError):
                validate_cross_and_deltas(current, baseline)
        current = fixture([[1., 0., .5]], [-20000.])
        current["denominator_sign"][0] = -1
        with self.assertRaisesRegex(ValueError, "sign"):
            validate_scales(current)
        current["relative_denominator"][0] = .25
        with self.assertRaisesRegex(ValueError, "amplitude"):
            chart_log_amplitude(current)

    def test_cutoffs_match_spectrum_budget_and_bond_families(self):
        widths = [3, 2, 2, 4, 3, 3]
        ranks = [3, 1, 1, 2, 3, 3]
        nodes = [{"name": name, "children": children, "source": source}
                 for name, children, source in (("input0", [], "token0"),
                     ("rn:moment", [0], None), ("attention:score", [1], None),
                     ("hidden", [2], None), ("input1", [], "token1"),
                     ("root", [3, 4], None))]
        spectra = [np.arange(width, 0., -1., dtype=np.float64) for width in widths]
        spectra[1] = np.zeros(2)
        diagnostics = [{"eigen_equation_residual": 0.} for _ in widths]
        cutoffs = []
        for i in (1, 2, 3):
            rank = ranks[i]
            gap = float(spectra[i][rank-1]-spectra[i][rank])
            cutoffs.append({"node": i, "rank": rank, "width": widths[i], "eigengap": gap,
                "unresolved": gap == 0, "eigen_equation_residual": 0.,
                "tail_cost_stored_environment_units": float(np.sum(spectra[i][rank:]))})
        condition = {"ranks": ranks, "removed_dimensions": 4}
        result = allocation_summary(condition, cutoffs, nodes, widths, spectra, diagnostics)
        self.assertEqual(result["unresolved_cutoffs"], 1)
        self.assertEqual(result["by_bond_family"]["wide"]["removed_dimensions"], 2)
        self.assertEqual(result["by_bond_family"]["moment_pade"]["unresolved_cutoffs"], 1)
        self.assertEqual(result["by_bond_family"]["attention_scalar"]["cut_bonds"], 1)
        # Index 1 is a legitimate moment cut. The later second ingress remains protected.
        invalid_ranks = ranks.copy()
        invalid_ranks[4] = 2
        invalid_cutoffs = cutoffs + [{"node": 4, "rank": 2, "width": 3,
            "eigengap": 1., "unresolved": False, "eigen_equation_residual": 0.,
            "tail_cost_stored_environment_units": 1.}]
        with self.assertRaisesRegex(ValueError, "ingress or root"):
            allocation_summary({"ranks": invalid_ranks, "removed_dimensions": 5},
                invalid_cutoffs, nodes, widths, spectra, diagnostics)
        with self.assertRaisesRegex(ValueError, "inventory"):
            allocation_summary(condition, cutoffs[:2], nodes, widths, spectra, diagnostics)
        cutoffs[0]["unresolved"] = False
        with self.assertRaisesRegex(ValueError, "receipt"):
            allocation_summary(condition, cutoffs, nodes, widths, spectra, diagnostics)

    def test_hash_verification_rejects_tampering_before_json_parse(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"summary.json"
            path.write_text('{"complete": true}')
            expected = digest(path)
            path.write_text("malformed tampered input")
            with self.assertRaisesRegex(ValueError, "identity"):
                authenticated_json(path, expected)

    def test_fresh_discovery_and_cli_install_guards_before_arithmetic(self):
        for mode in ("discovery", "cli"):
            program = '''
import sys
import unittest
import numpy as np
before = np.linalg.qr
'''
            if mode == "discovery":
                program += '''
suite = unittest.defaultTestLoader.loadTestsFromName("scripts.test_odt_campaign_scale_summary_v1.ScaleSummaryTests.test_huge_negative_ledger_and_common_rescaling_leave_chart_unchanged")
assert np.linalg.qr is not before
assert unittest.TextTestRunner().run(suite).wasSuccessful()
'''
            else:
                program += '''
from scripts import odt_campaign_scale_summary_v1 as scale
def probe(*args):
    assert np.linalg.qr is not before
    scale.assert_guards()
    assert float(np.sum(np.arange(3.))) == 3.
    return {"complete": True, "condition_count": 25}
scale.summarize = probe
sys.argv = ["scale", "--campaign", "unused", "--results", "unused", "--paired-summary", "unused", "--paired-summary-sha256", "unused", "--output", "unused"]
scale.main()
'''
            program += '''
from research.odt_reference.run_tests import COUNTS
assert COUNTS == {"qr": 0, "eigh": 0, "prohibited_attempts": 0}
'''
            result = subprocess.run([sys.executable, "-c", program], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_audited_closure_and_no_factorization(self):
        self.assertEqual(len(source_audit()["local"]), 2)
        assert_guards()


if __name__ == "__main__":
    unittest.main()
