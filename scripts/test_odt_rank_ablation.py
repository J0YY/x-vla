"""Independent budget and physical-mask tests for the separate consumer."""
import unittest
from fractions import Fraction
from pathlib import Path
import tempfile

import numpy as np

from research.odt_reference.run_tests import install_guards, COUNTS
from research.odt_reference import run_curve as original
from research.odt_reference.curve import rank_schedule, physical_variant
from research.odt_reference.shared_dag import Graph, Node, common_bases, apply_bases
from scripts.odt_rank_ablation import protected_plan, allocation, consumer_audit, load_accepted


def fixture():
    nodes = [Node("input", np.eye(8), source="x"),
             Node("norm:moment", np.eye(2, 8), (0,)),
             Node("dot1", np.eye(2), (1,)),
             Node("wide", np.eye(8, 2), (2,)),
             Node("root", np.eye(8), (3,))]
    return Graph(nodes, np.eye(8), True)


class RankAblationTests(unittest.TestCase):
    def test_independent_fraction_budget_and_nesting(self):
        graph = fixture()
        for protected in ((), (1,), (1, 2)):
            previous = None
            for p in (0, 1, 5, 10, 20, 23, 24, 25, 26, 30, 40, 50):
                plan = protected_plan(graph, p, protected)
                ranks = list(plan.widths)
                events = sorted((Fraction(2*j-1, 2*plan.widths[i]), i)
                                for i in plan.eligible if i not in protected
                                for j in range(1, plan.widths[i]))
                budget = round(Fraction(plan.original_dimensions*p, 100))
                for _, i in events[:budget]:
                    ranks[i] -= 1
                self.assertEqual(plan.ranks, tuple(ranks))
                self.assertEqual(plan.removed_dimensions, budget)
                self.assertEqual(plan.original_dimensions, 12)
                for i in protected:
                    self.assertEqual(plan.ranks[i], plan.widths[i])
                if previous is not None:
                    self.assertTrue(all(a <= b for a, b in zip(plan.ranks, previous)))
                previous = plan.ranks
                if not protected:
                    self.assertEqual(plan, rank_schedule(graph, percents=(p,)).plans[0])

    def test_half_even_and_invalid_protection(self):
        graph = fixture()
        for p, expected in ((25, 3), (50, 6)):
            self.assertEqual(protected_plan(graph, p).removed_dimensions, expected)
        # Total10 makes5%=0.5->0 and15%=1.5->2.
        graph.nodes[3].core = np.eye(6, 2)
        graph.nodes[4].core = np.eye(8, 6)
        self.assertEqual(protected_plan(graph, 5).removed_dimensions, 0)
        self.assertEqual(protected_plan(graph, 15).removed_dimensions, 2)
        for protected in ((0,), (4,), (1, 1), (True,), (-1,)):
            with self.assertRaises(ValueError):
                protected_plan(graph, 30, protected)
        with self.assertRaisesRegex(ValueError, "unattainable"):
            protected_plan(graph, 50, (1, 2, 3))

    def test_families_restoration_and_physical_replay(self):
        graph = fixture()
        bases = common_bases(graph)
        inputs = {"x": np.array([[1., 2., 3., 4., 5., 6., 7., 8.]])}
        for policy, p in (("original", 30), ("protect_all", 30), ("protect_norm", 30),
                          ("only_all", None), ("only_norm", None), ("only_score", None),
                          ("restore_all", 30), ("restore_norm", 30)):
            plan, selected = allocation(graph, policy, p)
            variant = physical_variant(graph, bases, plan)
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / "physical"
                hashes = original.save_graph(out, variant.graph, global_exponent=-1700)
                reloaded, _ = original.load_graph(out, hashes, global_exponent=-1700)
                a = original.checked_chart(reloaded, inputs)
                b = original.checked_chart(apply_bases(graph, variant.full_bases), inputs, plan.ranks)
                self.assertEqual(a.valid_rows, b.valid_rows)
                self.assertEqual(a.failure_reasons, b.failure_reasons)
                original.close(a.projective, b.projective, 3e-10, "physical/mask")
            if policy.startswith(("protect_", "restore_")):
                self.assertTrue(all(plan.ranks[i] == plan.widths[i] for i in selected))
            if policy.startswith("only_"):
                self.assertEqual(plan.removed_dimensions, len(selected))

    def test_source_audit_and_guard(self):
        self.assertIn("consumer_sha256", consumer_audit())
        self.assertEqual(COUNTS["prohibited_attempts"], 0)

    def test_forged_producer_rejected_before_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original.write_json(root / "accepted.json", {"accepted": True})
            with self.assertRaisesRegex(ValueError, "immutable accepted producer identity"):
                load_accepted(root, consumer_audit())


if __name__ == "__main__":
    install_guards()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RankAblationTests))
    print(COUNTS)
    raise SystemExit(0 if result.wasSuccessful() and COUNTS["prohibited_attempts"] == 0 else 1)
