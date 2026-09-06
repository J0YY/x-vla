"""Two-stage Algorithm 2 against literal, independently contracted networks."""
import unittest

import numpy as np

from research.odt_reference.shared_dag import (
    Graph, Node, canonicalize, occurrence_environment_sums,
)
from research.odt_reference.clone_passes import (
    canonicalize_clones, occurrence_environments, aggregate_environments,
)
from research.odt_reference.clone_oracle import clone_occurrence_environments


def fixtures():
    rng = np.random.default_rng(963)
    unary = Graph([
        Node("input", rng.normal(size=(3, 2)), source="x"),
        Node("unary", rng.normal(size=(2, 3)), (0,)),
    ], rng.normal(size=(2, 2)))
    binary = Graph([
        Node("left", rng.normal(size=(2, 2)), source="x"),
        Node("right", rng.normal(size=(3, 3)), source="y"),
        Node("binary", rng.normal(size=(4, 2, 3)), (0, 1)),
    ], rng.normal(size=(2, 4)))
    core = rng.normal(size=(5, 2, 2))
    tied = Graph([
        Node("input", rng.normal(size=(2, 2)), source="x"),
        Node("tied", core / 2 + core.swapaxes(1, 2) / 2, (0, 0)),
    ], rng.normal(size=(2, 5)))
    deficient = tied.copy()
    deficient.nodes[1].core[1:] = 0.
    zero = tied.copy()
    zero.nodes[0].core[:] = 0.
    diamond = Graph([
        Node("input", rng.normal(size=(2, 2)), source="x"),
        Node("left", rng.normal(size=(2, 2)), (0,)),
        Node("right", rng.normal(size=(3, 2)), (0,)),
        Node("merge", rng.normal(size=(2, 2, 3)), (1, 2)),
    ], rng.normal(size=(2, 2)))
    return (unary, binary, tied, deficient, zero, diamond)


def literal_recurrence(tree):
    """Original three-operand downstream contraction at each distinct cut."""
    cuts = {}
    def visit(node, environment):
        cuts[node.path] = (node.origin, environment)
        if len(node.children) == 1:
            visit(node.children[0], np.einsum("ai,ab,bj->ij", node.core, environment, node.core))
        elif len(node.children) == 2:
            visit(node.children[0], np.einsum("aik,ab,bjk->ij", node.core, environment, node.core))
            visit(node.children[1], np.einsum("aki,ab,bkj->ij", node.core, environment, node.core))
    visit(tree.root, np.einsum("oi,oj->ij", tree.head, tree.head))
    return cuts


class DownstreamContractionTests(unittest.TestCase):
    def assertEnvironmentEqual(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.isfinite(actual).all())
        scale = max(1., float(np.max(np.abs(expected))))
        self.assertLessEqual(float(np.max(np.abs(actual - expected))), 3e-12 * scale)

    def test_two_stage_matches_literal_contractions_for_every_occurrence(self):
        for graph in fixtures():
            with self.subTest(root=graph.nodes[-1].name, zero=not graph.nodes[0].core.any()):
                clones = canonicalize_clones(graph)
                actual = occurrence_environments(clones)
                for expected in (literal_recurrence(clones.tree), clone_occurrence_environments(clones.tree)):
                    self.assertEqual(set(actual), set(expected))
                    for path, (origin, environment) in expected.items():
                        self.assertEqual(actual[path][0], origin)
                        self.assertEnvironmentEqual(actual[path][1], environment)

    def test_shared_sums_match_independently_contracted_occurrences(self):
        for graph in fixtures():
            with self.subTest(root=graph.nodes[-1].name, zero=not graph.nodes[0].core.any()):
                clones = canonicalize_clones(graph)
                expected = [np.zeros((node.core.shape[0],) * 2) for node in canonicalize(graph).nodes]
                for origin, environment in clone_occurrence_environments(clones.tree).values():
                    expected[origin] += environment
                for actual in (occurrence_environment_sums(canonicalize(graph)), aggregate_environments(clones)):
                    for received, wanted in zip(actual, expected):
                        self.assertEnvironmentEqual(received, wanted)

    def test_nonfinite_clone_environment_still_fails_closed(self):
        clones = canonicalize_clones(fixtures()[0])
        clones.tree.head[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite downstream"):
            occurrence_environments(clones)
