"""Guarded hand-computable, rank-deficient, and clone tests for local F."""
import json
import unittest

import numpy as np

from research.odt_reference.run_tests import install_guards, COUNTS
from research.odt_ffn_v1.run import source_audit
from research.odt_ffn_v1.core import compile_ffn, ffn_forward
from research.odt_reference.block import evaluate_block
from research.odt_reference.clone_passes import clone_node_count
from research.odt_reference.shared_dag import occurrence_counts
from research.odt_reference import run_curve as reference


def fixture(width=1):
    return {"ffn.left.weight": np.eye(width), "ffn.right.weight": np.eye(width),
        "ffn.down.weight": np.eye(width), "ffn.left.bias": np.ones(width),
        "ffn.right.bias": np.full(width, -2.), "ffn.down.bias": np.full(width, 4.),
        "ffn_gain": np.array(.25), "rbn_ffn.running_ms": np.array(1.),
        "rbn_ffn.initialized": np.array(True), "rbn_ffn.pa": np.array([1., 0., 0.]),
        "rbn_ffn.pb": np.array([1., 0., 0.])}


class FFNTests(unittest.TestCase):
    def test_hand_computable_bias_gain_residual(self):
        w = fixture()
        x = np.array([[[-3.]], [[0.]], [[2.]]])
        expected = x[:, 0] + .25 * ((x[:, 0] + 1.) * (x[:, 0] - 2.) + 4.)
        np.testing.assert_allclose(ffn_forward(x, w), expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(evaluate_block(compile_ffn(w), x), expected, rtol=1e-14, atol=1e-14)

    def test_rectangular_cp_nontrivial_pade_and_topology(self):
        # Width eight admits the reference gate's fixed 0–80% test schedule.
        # Smaller fixtures cannot remove 80% while retaining every scalar bond.
        w, rng = fixture(8), np.random.default_rng(1209)
        w["ffn.left.weight"] = rng.normal(size=(3, 8))
        w["ffn.right.weight"] = rng.normal(size=(3, 8))
        w["ffn.down.weight"] = rng.normal(size=(8, 3))
        w["ffn.left.bias"] = rng.normal(size=3)
        w["ffn.right.bias"] = rng.normal(size=3)
        w["rbn_ffn.pa"] = np.array([1.1, .2, .03])
        w["rbn_ffn.pb"] = np.array([1.3, .1, .01])
        w["rbn_ffn.running_ms"] = np.array(.7)
        x = rng.normal(size=(8, 1, 8))
        graph = compile_ffn(w)
        self.assertEqual(clone_node_count(graph), 21)
        self.assertEqual(tuple(occurrence_counts(graph)), (11, 4, 2, 2, 1, 1))
        np.testing.assert_allclose(evaluate_block(graph, x), ffn_forward(x, w), rtol=1e-12, atol=1e-12)
        canonical, bases, schedule, gates = reference.clone_gate(graph, x, ffn_forward(x, w), lambda _: None)
        self.assertEqual(schedule.canonical_internal_dimensions, 22)
        self.assertEqual([n.core.shape[0] for n in canonical.nodes], [9, 2, 2, 9, 9, 9])
        self.assertEqual(gates["maximum_errors"]["every_qr_step"], 0.)
        self.assertEqual(gates["maximum_errors"]["every_gauge_step"], 0.)

    def test_zero_gain_keeps_direct_q_completion_and_identity(self):
        w = fixture(8)
        w["ffn_gain"] = np.array(0.)
        x = np.arange(16, dtype=float).reshape(2, 1, 8) / 20.
        x[1] = 0.
        graph = compile_ffn(w)
        canonical, _, _, gates = reference.clone_gate(graph, x, x[:, 0], lambda _: None)
        self.assertEqual(canonical.nodes[4].core.shape[0], 9)
        np.testing.assert_allclose(evaluate_block(canonical, x), x[:, 0], atol=1e-12, rtol=1e-12)

    def test_fail_closed_shape_uninitialized_and_allocation(self):
        w = fixture(2)
        with self.assertRaises(ValueError):
            compile_ffn(w, maximum_elements=26)
        w["rbn_ffn.initialized"] = np.array(False)
        with self.assertRaises(ValueError):
            compile_ffn(w)
        w = fixture(2)
        w["ffn.left.bias"] = np.zeros(3)
        with self.assertRaises(ValueError):
            compile_ffn(w)


if __name__ == "__main__":
    sources = source_audit()
    install_guards()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FFNTests))
    passed = result.wasSuccessful() and COUNTS["prohibited_attempts"] == 0 and source_audit() == sources
    print(json.dumps({"passed": passed, "tests": result.testsRun, "kernels": COUNTS,
                      "numpy": np.__version__, "source_sha256": sources}, sort_keys=True))
    raise SystemExit(0 if passed else 1)
