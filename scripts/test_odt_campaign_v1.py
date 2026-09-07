"""Guarded behavioral tests for the separate immutable consumer campaign."""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

# Cover discovery/import based invocations before any numerical fixture setup.
from research.odt_reference.run_tests import COUNTS, install_guards

install_guards()

from research.odt_campaign_v1.campaign import (
    ENVIRONMENT_EXPONENT, allocation, anchored_bases, audit_campaign, chart_arrays,
    error_metrics, freeze_conditions, json_digest, load_campaign, native_parity,
    scaled_chart, spectra_from_environments, validate_native_panel, validate_record_partitions, zero_activations,
)
from research.odt_reference import run_curve as original
from research.odt_reference.curve import physical_variant, rank_schedule
from research.odt_reference.shared_dag import Graph, Node, apply_bases, canonicalize, common_bases, evaluate


def allocation_fixture():
    # This is a shape-only allocation fixture, not a canonicalization claim.
    nodes = [Node("input", np.eye(8), source="x"),
             Node("norm:moment", np.eye(2, 8), (0,)),
             Node("dot1", np.eye(2), (1,)),
             Node("wide", np.eye(8, 2), (2,)),
             Node("root", np.eye(8), (3,))]
    return Graph(nodes, np.eye(8), True)


def tied_fixture():
    core = np.zeros((2, 2, 2))
    core[0, 0, 0], core[1, 1, 1] = 3., 2.
    graph = Graph([Node("input", np.eye(2), source="x"),
                   Node("square", core, (0, 0)),
                   Node("root", np.eye(2), (1,))], np.eye(2))
    return canonicalize(graph)


def shared_fixture():
    core = np.zeros((3, 3, 3))
    core[0, 0, 0], core[0, 1, 1] = 1., 1.
    core[1, 2, 2] = 1.
    core[2, 1, 2], core[2, 2, 1] = .5, .5
    graph = Graph([Node("input", np.eye(3), source="x"),
                   Node("norm:moment", np.array([[1., 0., 0.], [0., 0., 1.]]), (0,)),
                   Node("wide", np.array([[1., 0.], [0., 1.], [1., 1.]]), (1,)),
                   Node("root", core, (2, 2))], np.eye(3))
    return canonicalize(graph)


def fixture_spectra(graph):
    return tuple(np.arange(node.core.shape[0], 0, -1, dtype=np.float64) for node in graph.nodes)


class CampaignTests(unittest.TestCase):
    def test_fresh_process_cli_installs_guards_before_numerical_work(self):
        program = '''
import sys
import numpy as np
from research.odt_campaign_v1 import campaign
from research.odt_reference.run_tests import COUNTS
before_qr = np.linalg.qr
before_disallowed = np.linalg.matrix_rank
def probe(*args):
    assert np.linalg.qr is not before_qr
    assert np.linalg.matrix_rank is not before_disallowed
    assert COUNTS == {"qr": 0, "eigh": 0, "prohibited_attempts": 0}
    np.linalg.qr(np.eye(2), mode="reduced")
    assert COUNTS == {"qr": 1, "eigh": 0, "prohibited_attempts": 0}
    print("cli_guard_before_work_verified")
campaign.prepare = probe
sys.argv = ["campaign", "prepare", "--producer", "unused", "--real-panel", "unused",
            "--real-metadata", "unused", "--output", "unused"]
campaign.main()
'''
        completed = subprocess.run([sys.executable, "-c", program], cwd=Path(__file__).resolve().parents[1],
                                   capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
        self.assertIn("cli_guard_before_work_verified", completed.stdout)

    def test_width_allocation_independent_exact_fraction_and_protection(self):
        graph = allocation_fixture()
        for policy in ("W", "WN"):
            previous = None
            for budget in range(8):
                plan = allocation(graph, (), policy, budget)
                protected = (1,) if policy == "WN" else ()
                events = sorted((Fraction(2*j-1, 2*plan.widths[i]), i)
                                for i in plan.eligible if i not in protected
                                for j in range(1, plan.widths[i]))
                expected = list(plan.widths)
                for _, index in events[:budget]:
                    expected[index] -= 1
                self.assertEqual(plan.ranks, tuple(expected))
                self.assertEqual(plan.original_dimensions, 12)
                self.assertEqual(plan.removed_dimensions, budget)
                self.assertEqual(plan.ranks[0], 8)
                self.assertEqual(plan.ranks[-1], 8)
                if previous is not None:
                    self.assertTrue(all(a <= b for a, b in zip(plan.ranks, previous)))
                previous = plan.ranks
        with self.assertRaisesRegex(ValueError, "unattainable"):
            allocation(graph, (), "WN", 9)
        for budget in (True, -1, 1.5):
            with self.assertRaises(ValueError):
                allocation(graph, (), "W", budget)

    def test_spectral_cost_global_optimum_against_exhaustive_feasible_ranks(self):
        graph = allocation_fixture()
        spectra = list(fixture_spectra(graph))
        spectra[1] = np.array([.4, .1])
        spectra[2] = np.array([2., .3])
        spectra[3] = np.array([80., 60., 20., 10., 5., 2., .2, .05])
        for policy in ("S", "SN"):
            for budget in range(8):
                plan = allocation(graph, spectra, policy, budget)
                observed = sum(float(np.sum(spectra[i][plan.ranks[i]:])) for i in plan.eligible)
                feasible = []
                for a in range(1, 3):
                    for b in range(1, 3):
                        for c in range(1, 9):
                            if (2-a)+(2-b)+(8-c) != budget or (policy == "SN" and a != 2):
                                continue
                            feasible.append(float(np.sum(spectra[1][a:])+np.sum(spectra[2][b:])+np.sum(spectra[3][c:])))
                self.assertAlmostEqual(observed, min(feasible), places=13)
                self.assertEqual(plan.original_dimensions, 12)
                self.assertTrue(all(rank >= 1 for rank in plan.ranks))

    def test_spectral_prefix_ties_and_invalid_spectra(self):
        graph = allocation_fixture()
        spectra = tuple(np.zeros(node.core.shape[0]) for node in graph.nodes)
        plan = allocation(graph, spectra, "S", 4)
        self.assertEqual(plan.ranks, (8, 1, 1, 6, 8))
        bad = list(spectra)
        bad[3] = np.arange(8.)
        with self.assertRaisesRegex(ValueError, "descending"):
            allocation(graph, bad, "S", 4)
        bad[3] = np.full(8, -.1)
        with self.assertRaises(ValueError):
            allocation(graph, bad, "S", 4)

    def test_condition_dedup_preserves_all_aliases_and_predeclared_contrasts(self):
        graph = allocation_fixture()
        spectra = fixture_spectra(graph)
        conditions, contrasts = freeze_conditions(graph, spectra, budgets=(1, 2, 4))
        expected_aliases = 1+4*3
        self.assertEqual(sum(len(c["aliases"]) for c in conditions), expected_aliases)
        self.assertLess(len(conditions), expected_aliases)
        self.assertEqual(len(contrasts), 12)
        self.assertEqual(len({c["condition_sha256"] for c in conditions}), len(conditions))
        for index, condition in enumerate(conditions):
            self.assertEqual(condition["task_index"], index)
            self.assertEqual(condition["rank_sha256"], json_digest(condition["ranks"]))
        for contrast in contrasts:
            self.assertTrue(0 <= contrast["a"] < len(conditions))
            self.assertTrue(0 <= contrast["b"] < len(conditions))

    def test_environment_spectra_equations_and_ledger(self):
        graph = tied_fixture()
        bases = common_bases(graph)
        spectra, diagnostics = spectra_from_environments(graph, bases, environment_exponent=ENVIRONMENT_EXPONENT)
        self.assertEqual(len(spectra), len(graph.nodes))
        self.assertTrue(all(np.all(np.diff(s) <= 0) for s in spectra))
        self.assertTrue(all(d["eigen_equation_residual"] < 1e-12 for d in diagnostics))
        with self.assertRaisesRegex(ValueError, "ledger"):
            spectra_from_environments(graph, bases, environment_exponent=0)
        wrong = list(bases)
        wrong[1] = wrong[1][:, ::-1]
        with self.assertRaisesRegex(ValueError, "unordered"):
            spectra_from_environments(graph, wrong, environment_exponent=ENVIRONMENT_EXPONENT)

    def test_anchor_contains_zero_and_direct_q_reconstruction_with_all_occurrences(self):
        graph = shared_fixture()
        spectral = common_bases(graph)
        anchors = zero_activations(graph)
        controls, diagnostic = anchored_bases(graph, spectral, 2)
        repeated, _ = anchored_bases(graph, spectral, 2)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(controls, repeated)))
        self.assertTrue(all(d["qr_reconstruction"] < 1e-12 for d in diagnostic))
        for i in rank_schedule(graph, percents=(0,)).plans[0].eligible:
            column = controls[i][:, 0]
            scalar = float(np.sum(column*anchors[i]))
            original.close(column*scalar, anchors[i], 1e-12, "anchor direct vector reconstruction")
        plan = allocation(graph, (), "W", 2)
        variant = physical_variant(graph, controls, plan)
        # Both repeated input roles MUST be physically narrowed.
        self.assertEqual(variant.graph.nodes[-1].core.shape[1:], (plan.ranks[2], plan.ranks[2]))
        inputs = {"x": np.array([[0., 0., 1.], [1., 2., 1.], [-.3, .2, 1.]])}
        full_zero = original.checked_chart(graph, {"x": inputs["x"][:1]})
        cut_zero = original.checked_chart(variant.graph, {"x": inputs["x"][:1]})
        original.close(cut_zero.decoded, full_zero.decoded, 1e-12, "anchored whole graph")
        masked = original.checked_chart(apply_bases(graph, controls), inputs, plan.ranks)
        physical, _ = scaled_chart(variant.graph, inputs)
        original.close(physical.projective, masked.projective, 1e-12, "all occurrences physical/mask")
        self.assertEqual(physical.valid_rows, masked.valid_rows)
        # An adversarial one-role-only contraction is caught by the shape gate.
        self.assertEqual(len([c for c in graph.nodes[-1].children if c == 2]), 2)

    def test_invalid_zero_anchor_fails_with_node_and_shape(self):
        graph = tied_fixture()
        graph.nodes[0].core[:, -1] = 0.
        with self.assertRaisesRegex(ValueError, "node 0, shape"):
            anchored_bases(graph, common_bases(graph), 0)

    def test_scale_ledger_matches_direct_values_and_repeated_child_multiplicity(self):
        graph = tied_fixture()
        inputs = {"x": np.array([[8., 1.], [.2, 1.], [0., 1.]])}
        chart, diagnostics = scaled_chart(graph, inputs, global_exponent=-7)
        for i, row in enumerate(inputs["x"]):
            direct = evaluate(graph, {"x": row}) * 2.**-7
            reconstructed = chart.projective[i] * 2.**diagnostics["output_log2_scale"][i]
            original.close(reconstructed, direct, 1e-12, "fixed-lift log scale")
            self.assertAlmostEqual(diagnostics["log2_abs_denominator"][i], math.log2(abs(direct[-1])), places=12)
        reference = original.checked_chart(graph, inputs)
        original.close(chart.projective, reference.projective, 1e-12, "scale evaluator chart replay")
        self.assertEqual(chart.valid_rows, reference.valid_rows)

    def test_scale_zero_rows_and_denominator_invalidity_preserved(self):
        graph = tied_fixture()
        chart, diagnostics = scaled_chart(graph, {"x": np.array([[1., 0.], [0., 0.], [1., 1.]])})
        self.assertEqual(chart.failure_reasons, ("denominator_margin", "zero_projective_row", None))
        arrays = chart_arrays(chart)
        self.assertTrue(np.isnan(arrays["decoded"][:2]).all())
        self.assertEqual(arrays["failure_code"].tolist(), [1, 2, 0])
        self.assertEqual(diagnostics["output_log2_scale"][1], -math.inf)

    def test_unconditional_invalidity_and_jointly_valid_metrics(self):
        output = np.array([[3., 4.], [np.nan, np.nan], [5., 6.]])
        valid = np.array([True, False, True])
        source = np.array([[1., 2.], [2., 3.], [np.nan, np.nan]])
        source_valid = np.array([True, True, False])
        report = error_metrics(output, valid, source, source_valid)
        self.assertEqual(report["input_count"], 3)
        self.assertEqual(report["invalid_count"], 1)
        self.assertEqual(report["reference_invalid_count"], 1)
        self.assertEqual(report["jointly_valid_count"], 1)
        self.assertEqual(report["rmse"], 2.)
        empty = error_metrics(output, np.zeros(3, dtype=bool), source, source_valid)
        self.assertEqual(empty["invalid_probability_unconditional"], 1.)
        self.assertEqual(empty["jointly_valid_count"], 0)
        self.assertIsNone(empty["rmse"])

    def test_native_gate_requires_absolute_and_scaled_limits(self):
        expected = np.ones((2, 3))
        native_parity(expected+1e-6, expected)
        with self.assertRaisesRegex(ValueError, "parity gate"):
            native_parity(expected+1e-4, expected)
        expected = np.full((2, 3), 1000.)
        with self.assertRaisesRegex(ValueError, "parity gate"):
            native_parity(expected+.001, expected)

    def test_panel_authentication_before_array_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path/"panel.npz").write_bytes(b"not an archive")
            original.write_json(path/"panel.json", {"panel_sha256": "wrong",
                                                     "checkpoint_sha256": original.CHECKPOINT_SHA})
            with self.assertRaisesRegex(ValueError, "authentication"):
                validate_native_panel(path/"panel.npz", path/"panel.json")

    def test_failed_native_producer_stops_before_array_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path/"panel.npz").write_bytes(b"not an archive")
            original.write_json(path/"panel.json", {"panel_sha256": original.digest(path/"panel.npz"),
                "checkpoint_sha256": original.CHECKPOINT_SHA, "schema": "odt-real-panel-v1", "passed": False})
            with self.assertRaisesRegex(ValueError, "producer did not pass"):
                validate_native_panel(path/"panel.npz", path/"panel.json")

    def test_task_episode_balance_pair_rule_and_development_disjointness(self):
        records = [{"task_id": task, "episode_id": episode, "frame_id": frame,
                    "pair_id": pair, "pair_positions": [27, 28] if pair == 0 else [0, 63]}
                   for task in range(10) for episode in range(10) for frame in (1, 10) for pair in (0, 1)]
        confirmation = [r for r in records if r["episode_id"] >= 2]
        development = [r for r in records if r["episode_id"] < 2]
        validate_record_partitions(confirmation, development)
        overlap = [dict(r, episode_id=r["episode_id"]+2) for r in development]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_record_partitions(confirmation, overlap)
        bad_pairs = [dict(r) for r in confirmation]
        bad_pairs[0]["pair_positions"] = [26, 27]
        with self.assertRaisesRegex(ValueError, "token pair"):
            validate_record_partitions(bad_pairs, development)
        duplicate = confirmation[:-1]+[confirmation[0]]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_record_partitions(duplicate, development)

    def test_changed_campaign_artifact_fails_before_loading_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            required = ("protocol.json", "inputs.npz", "records.json", "bases.npz", "spectral_diagnostics.json",
                        "anchor_diagnostics.json", "baseline.npz")
            for name in required:
                (path/name).write_text("original")
            hashes = {name: original.digest(path/name) for name in required}
            original.write_json(path/"manifest.json", {"ready": True, "version": "odt-campaign-v1", "artifacts": hashes})
            (path/"bases.npz").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_campaign(path)

    def test_audited_sources_and_runtime_counts(self):
        sources = audit_campaign()
        self.assertEqual(len(sources["campaign_sources"]), 3)
        self.assertIn("producer_sources", sources["accepted_consumer"])
        self.assertEqual(COUNTS["prohibited_attempts"], 0)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CampaignTests))
    print(json.dumps({"passed": result.wasSuccessful(), "tests": result.testsRun,
                      "kernels": COUNTS, "sources": audit_campaign()}, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() and COUNTS["prohibited_attempts"] == 0 else 1)
