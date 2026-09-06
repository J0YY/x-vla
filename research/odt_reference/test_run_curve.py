"""Synthetic campaign smoke tests, never evidence about a trained policy."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.odt_reference.run_curve import (
    SOURCES, CHECKPOINT_SHA, TOLERANCES, close, digest, write_json, save_graph,
    load_graph, homogeneous, checked_chart, clone_gate, evaluate,
)
from research.odt_reference.run_tests import audit
from research.odt_reference.shared_dag import Graph, Node


def fixture():
    width = 10
    core = np.zeros((width, width, width))
    for i in range(width):
        core[i, i, i] = 1. / (i + 1)
    return Graph([Node("input", np.eye(width), source="token0"),
                  Node("square", core, (0, 0)), Node("root", np.eye(width), (1,))], np.eye(width))


class CampaignTests(unittest.TestCase):
    def test_roundtrip_every_step_and_all_six_workers(self):
        graph = fixture()
        raw = np.arange(18, dtype=float).reshape(2, 1, 9) / 20 + .1
        expected = checked_chart(graph, homogeneous(raw)).decoded
        progress = []
        canonical, bases, schedule, gates = clone_gate(graph, raw, expected, progress.append)
        self.assertEqual(sum(p["phase"] == "every_qr_step" for p in progress), 3)
        self.assertEqual(sum(p["phase"] == "every_gauge_step" for p in progress), 3)
        self.assertLess(gates["maximum_errors"]["environment"], 1e-12)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            hashes = save_graph(output / "canonical", canonical, bases)
            reloaded, reloaded_bases = load_graph(output / "canonical", hashes)
            self.assertTrue(reloaded.canonical)
            np.testing.assert_array_equal(reloaded.nodes[1].core, canonical.nodes[1].core)
            np.testing.assert_array_equal(reloaded_bases[1], bases[1])
            np.savez(output / "panel.npz", raw=raw, expected=expected)
            sources = audit(SOURCES)
            write_json(output / "accepted.json", {"accepted": True, "source_sha256": sources,
                       "checkpoint_sha256": CHECKPOINT_SHA, "tolerances": TOLERANCES,
                       "numpy": np.__version__, "panel_sha256": digest(output / "panel.npz"),
                       "canonical_hashes": hashes, "raw_internal_dimensions": schedule.raw_internal_dimensions})
            for percent in (30, 40, 50, 60, 70, 80):
                evaluate(output, percent, sources)
                result = json.loads((output / f"p{percent}_leading_0" / "result.json").read_text())
                self.assertEqual(result["kept_internal_dimensions"], 10 - percent // 10)
                self.assertLess(result["rank_mask_equivalence"], 1e-12)
            self.assertTrue((output / "p70_random_2" / "result.json").is_file())
            with self.assertRaisesRegex(ValueError, "receipt mismatch"):
                evaluate(output, 30, {})
            with (output / "canonical" / "arrays.npz").open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_graph(output / "canonical", hashes)

    def test_finite_large_error_and_nonfinite_rejected(self):
        for actual in (np.array([2.]), np.array([np.nan]), np.array([np.inf])):
            with self.assertRaises(ValueError):
                close(actual, np.array([1.]), 1e-10, "corrupted")

    def test_exact_comparison_preserves_finite_gate_and_global_scaling(self):
        for value in (np.array([np.nan]), np.array([np.inf]), np.array([-np.inf])):
            with self.assertRaises(ValueError):
                close(value, value.copy(), 1e-10, "identical nonfinite")
        expected = np.zeros(131073)
        expected[0] = 1e8
        actual = expected.copy()
        self.assertEqual(close(actual, expected, 1e-10, "identical finite"), 0.)
        actual[-1] = 1e-4
        self.assertAlmostEqual(close(actual, expected, 1e-10, "global scaling"), 1e-12)


if __name__ == "__main__":
    unittest.main()
