"""Independent complete-tensor and dynamic-range tests for the binary ledger."""
import unittest

import numpy as np

from research.odt_reference.scaled import balanced, canonicalize_scaled_clones
from research.odt_reference.shared_dag import (
    Graph, Node, canonical_step, occurrence_counts, occurrence_environment_sums,
)
from research.odt_reference.clone_oracle import unfold_no_memo, walk_clones, clone_coefficients
from research.odt_reference.curve import masked_evaluate
from research.odt_reference.test_shared_dag import repeated, diamond, deficient_repeated


class BinaryLedgerTests(unittest.TestCase):
    def checked(self, original, *, complete=True):
        shared, exponent = original.copy(), 0
        counts = occurrence_counts(shared)
        if complete:
            expected, labels = clone_coefficients(unfold_no_memo(original))

        def compare(tree, origin, clone_exponent):
            nonlocal exponent
            shared.nodes[origin].core, shift = balanced(shared.nodes[origin].core)
            exponent += counts[origin] * shift
            canonical_step(shared, origin)
            self.assertIs(type(exponent), int)
            self.assertIs(type(clone_exponent), int)
            self.assertEqual(exponent, clone_exponent)
            for node in walk_clones(tree):
                np.testing.assert_array_equal(node.core, shared.nodes[node.origin].core)
            np.testing.assert_array_equal(tree.head, shared.head)
            if complete:
                actual, actual_labels = clone_coefficients(tree)
                self.assertEqual(actual_labels, labels)
                np.testing.assert_allclose(np.ldexp(actual, exponent), expected,
                                           rtol=3e-12, atol=3e-12)

        clones, clone_exponent = canonicalize_scaled_clones(original, after_step=compare)
        shared.head, shift = balanced(shared.head)
        exponent += shift
        shared.canonical = True
        self.assertEqual(exponent, clone_exponent)
        np.testing.assert_array_equal(shared.head, clones.tree.head)
        if complete:
            actual, _ = clone_coefficients(clones.tree)
            np.testing.assert_allclose(np.ldexp(actual, exponent), expected,
                                       rtol=3e-12, atol=3e-12)
        return shared, exponent

    def test_every_step_complete_coefficients_and_independent_occurrence_ledger(self):
        zero = Graph([Node("zero", np.zeros((2, 2)), source="x"),
                      Node("repeat", np.zeros((2, 2, 2)), (0, 0))], np.eye(2))
        for graph in (repeated(), diamond(), deficient_repeated(), zero):
            self.checked(graph)
        graph = repeated()
        graph.head = np.ldexp(graph.head, 9)
        self.checked(graph)

    def test_underflow_chain_preserves_chart_and_nonzero_environments(self):
        nodes = [Node("input", np.eye(2), source="x")]
        for index in range(4):
            nodes.append(Node(f"scale{index}", np.eye(2) * 1e-100, (index,)))
        graph, exponent = self.checked(Graph(nodes, np.eye(2)), complete=False)
        self.assertLess(exponent, -1074)
        chart = masked_evaluate(graph, {"x": [.2, 1.]})
        self.assertEqual(chart.valid_rows, (True,))
        np.testing.assert_allclose(chart.decoded, [[.2]], rtol=0., atol=1e-15)
        for environment in occurrence_environment_sums(graph):
            self.assertGreater(float(np.max(np.abs(environment))), .1)

    def test_repeated_occurrences_give_exact_known_exponent_delta(self):
        original = repeated()
        self.assertEqual(occurrence_counts(original), (4, 2, 1))
        baseline, baseline_exponent = self.checked(original)
        for exponents in ((-300, -200, -100), (300, 200, 100)):
            scaled = original.copy()
            for node, power in zip(scaled.nodes, exponents):
                node.core = np.ldexp(node.core, power)
            actual, exponent = self.checked(scaled, complete=False)
            expected = 4 * exponents[0] + 2 * exponents[1] + exponents[2]
            self.assertEqual(abs(expected), 1700)
            self.assertEqual(exponent - baseline_exponent, expected)
            for a, b in zip(actual.nodes, baseline.nodes):
                np.testing.assert_array_equal(a.core, b.core)
            np.testing.assert_array_equal(actual.head, baseline.head)

    def test_wrong_unique_node_multiplier_changes_complete_tensor(self):
        original = repeated()
        before, _ = clone_coefficients(unfold_no_memo(original))
        shared = original.copy()
        shared.nodes[0].core, shift = balanced(shared.nodes[0].core)
        self.assertNotEqual(shift, 0)
        canonical_step(shared, 0)
        current, _ = clone_coefficients(unfold_no_memo(shared))
        correct = occurrence_counts(shared)[0] * shift
        np.testing.assert_allclose(np.ldexp(current, correct), before, rtol=3e-12, atol=3e-12)
        wrong = np.ldexp(current, shift)
        self.assertGreater(float(np.max(np.abs(wrong - before))),
                           .1 * float(np.max(np.abs(before))))

    def test_exact_round_trip_and_unrepresentable_relative_range(self):
        for value in (np.zeros((2, 2)), np.array([np.nextafter(0., 1.), 0.]),
                      np.array([1e-300, -2e-300]), np.array([1e300, -2e300])):
            mantissa, exponent = balanced(value)
            self.assertIs(type(exponent), int)
            np.testing.assert_array_equal(np.ldexp(mantissa, exponent), value)
        impossible = np.array([[1e300, 1e-300], [0., 1.]])
        with self.assertRaisesRegex(ValueError, "preserve every"):
            balanced(impossible)
        graph = Graph([Node("input", impossible, source="x")], np.eye(2))
        with self.assertRaisesRegex(ValueError, "loses tensor entries"):
            canonicalize_scaled_clones(graph)
        for value in (np.array([np.nan]), np.array([np.inf]), np.array([], dtype=float)):
            with self.assertRaises(ValueError):
                balanced(value)


if __name__ == "__main__":
    unittest.main()
