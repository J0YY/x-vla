"""Small physical-curve tests, never trained evidence."""
import unittest

import numpy as np

from research.odt_reference.curve import rank_schedule, physical_variant, masked_evaluate
from research.odt_reference.shared_dag import Graph, Node, canonicalize, common_bases, apply_bases


def fixture(width=10):
    core = np.zeros((width, width, width))
    for i in range(width):
        core[i, i, i] = 1. / (i + 1)
    return Graph([Node("input", np.eye(width), source="x"),
                  Node("repeated", core, (0, 0)),
                  Node("head", np.eye(width), (1,))], np.eye(width))


class CurveTests(unittest.TestCase):
    def test_all_six_integer_budgets_are_exact_and_nested(self):
        original = fixture()
        canonical = canonicalize(original)
        schedule = rank_schedule(canonical, original)
        self.assertEqual(schedule.raw_internal_dimensions, 10)
        self.assertEqual(schedule.canonical_internal_dimensions, 10)
        previous = schedule.plans[0].ranks
        for plan in schedule.plans:
            self.assertEqual(plan.removed_dimensions, plan.percent // 10)
            self.assertEqual(sum(plan.widths[i] - plan.ranks[i] for i in plan.eligible), plan.removed_dimensions)
            self.assertTrue(all(a <= b for a, b in zip(plan.ranks, previous)))
            self.assertEqual(plan.ranks[0], plan.widths[0])
            self.assertEqual(plan.ranks[-1], plan.widths[-1])
            previous = plan.ranks

    def test_physical_leading_trailing_random_match_independent_masks(self):
        canonical = canonicalize(fixture())
        bases = common_bases(canonical)
        inputs = {"x": np.stack((np.arange(10) + .2, np.arange(10) - .3))}
        for mode in ("leading", "trailing", "random"):
            previous_bases = None
            for plan in rank_schedule(canonical).plans:
                variant = physical_variant(canonical, bases, plan, mode, seed=93)
                gauge_space = apply_bases(canonical, variant.full_bases)
                masked = masked_evaluate(gauge_space, inputs, plan.ranks)
                physical = masked_evaluate(variant.graph, inputs)
                self.assertEqual(masked.valid_rows, physical.valid_rows)
                self.assertEqual(masked.failure_reasons, physical.failure_reasons)
                np.testing.assert_allclose(masked.projective, physical.projective, rtol=2e-10, atol=2e-11)
                if masked.decoded is not None:
                    np.testing.assert_allclose(masked.decoded, physical.decoded, rtol=2e-10, atol=2e-11)
                self.assertEqual(variant.graph.nodes[1].children, (0, 0))
                self.assertEqual(variant.graph.nodes[2].core.shape[1], plan.ranks[1])
                if previous_bases is not None:
                    for before, after in zip(previous_bases, variant.full_bases):
                        np.testing.assert_array_equal(before, after)
                previous_bases = variant.full_bases

    def test_zero_and_pole_charts_are_reported_not_repaired(self):
        graph = Graph([Node("input", np.eye(2), source="x"),
                       Node("zero", np.zeros((2, 2)), (0,))], np.eye(2), True)
        result = masked_evaluate(graph, {"x": [1., 1.]})
        self.assertEqual(result.failure_reasons, ("zero_projective_row",))
        self.assertIsNone(result.decoded)
        np.testing.assert_array_equal(result.projective, [[0., 0.]])
        graph.nodes[1].core = np.diag([1., 0.])
        result = masked_evaluate(graph, {"x": [[1., 1.], [0., 1.]]})
        self.assertEqual(result.failure_reasons, ("denominator_margin", "zero_projective_row"))
        self.assertIsNone(result.decoded)

    def test_shape_only_reductions_are_not_spectral_removal(self):
        core = np.zeros((12, 4, 4))
        core[0] = np.eye(4)
        graph = Graph([Node("input", np.eye(4), source="x"),
                       Node("tall", core, (0, 0)),
                       Node("head", np.ones((2, 12)), (1,))], np.eye(2))
        canonical = canonicalize(graph)
        schedule = rank_schedule(canonical, graph)
        self.assertEqual((schedule.raw_internal_dimensions, schedule.canonical_internal_dimensions), (12, 10))
        self.assertEqual(schedule.plans[0].removed_dimensions, 0)

    def test_rejects_unattainable_or_invalid_requests(self):
        canonical = canonicalize(fixture(2))
        with self.assertRaisesRegex(ValueError, "unattainable"):
            rank_schedule(canonical, percents=(0, 80))
        for percents in ((True,), (40, 30), (30, 30), (100,)):
            with self.assertRaises(ValueError):
                rank_schedule(canonical, percents=percents)
        with self.assertRaises(ValueError):
            rank_schedule(fixture())


if __name__ == "__main__":
    unittest.main()
