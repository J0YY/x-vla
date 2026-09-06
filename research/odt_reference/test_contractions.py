"""Two-stage Algorithm 2 against literal, independently contracted networks."""
import unittest

import numpy as np

from research.odt_reference.shared_dag import (
    Graph, Node, canonicalize, occurrence_environment_sums,
)
from research.odt_reference.clone_passes import (
    canonicalize_clones, occurrence_environments, aggregate_environments,
)
from research.odt_reference.clone_oracle import (
    clone_occurrence_environments, clone_canonical_step, unfold_no_memo, walk_clones,
)


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

    def test_independent_vectorized_coordinates_match_literal_loops(self):
        rng = np.random.default_rng(27)
        for width, out in ((1, 2), (2, 5), (25, 2), (193, 2)):
            raw = rng.normal(size=(out, width, width))
            symmetric = raw / 2 + raw.swapaxes(1, 2) / 2
            for kind in ("full", "deficient", "zero", "large"):
                core = symmetric.copy()
                if kind == "deficient":
                    core[1:] = 0.
                elif kind == "zero":
                    core[:] = 0.
                elif kind == "large":
                    core *= 1e100
                # Frozen pre-vectorization coordinate loops: independent oracle
                # for both input packing and deficient/rectangular Q completion.
                columns = []
                for i in range(width):
                    for j in range(i, width):
                        average = core[:, i, j] / 2 + core[:, j, i] / 2
                        columns.append(average * (1 if i == j else np.sqrt(2.)))
                packed = np.stack(columns, axis=1)
                columns_q, upper = np.linalg.qr(packed.T, mode="reduced")
                rows = columns_q.T
                wanted_q = np.zeros((rows.shape[0], width, width))
                cursor = 0
                for i in range(width):
                    for j in range(i, width):
                        wanted_q[:, i, j] = wanted_q[:, j, i] = rows[:, cursor] / (1 if i == j else np.sqrt(2.))
                        cursor += 1
                parent = rng.normal(size=(2, out, out))
                graph = Graph([Node("input", np.eye(width), source="x"),
                               Node("tied", core, (0, 0)),
                               Node("parent", parent, (1, 1))], np.eye(2))
                tree = unfold_no_memo(graph)
                # Count locally while preserving the already-installed guard.
                # A -m runner's __main__ counters are not the imported module's.
                guarded_qr, calls = np.linalg.qr, []
                def observed_qr(*args, **kwargs):
                    calls.append(1)
                    return guarded_qr(*args, **kwargs)
                np.linalg.qr = observed_qr
                try:
                    clone_canonical_step(tree, 1)
                finally:
                    np.linalg.qr = guarded_qr
                self.assertEqual(len(calls), 2)
                wanted_parent = np.einsum("oij,ia,jb->oab", parent, upper.T, upper.T)
                for node in walk_clones(tree):
                    wanted = wanted_q if node.origin == 1 else (wanted_parent if node.origin == 2 else np.eye(width))
                    np.testing.assert_allclose(node.core, wanted, rtol=3e-12, atol=3e-12)
                np.testing.assert_array_equal(tree.head, graph.head)

    def test_vectorized_coordinates_still_reject_nonsymmetric_input(self):
        core = np.ones((2, 3, 3))
        core[0, 0, 1] = 2.
        graph = Graph([Node("input", np.eye(3), source="x"), Node("bad", core, (0, 0))], np.eye(2))
        with self.assertRaisesRegex(ValueError, "explicitly symmetric"):
            clone_canonical_step(unfold_no_memo(graph), 1)
