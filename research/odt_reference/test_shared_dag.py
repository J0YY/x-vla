"""Bounded adversarial tests, independent of all production ODT code."""
import unittest

import numpy as np

from research.odt_reference import dooms
from research.odt_reference.shared_dag import (
    Graph, Node, evaluate, symmetrize_tied, canonical_step, canonicalize,
    occurrence_counts, occurrence_environment_sums, common_bases, apply_bases,
)
from research.odt_reference.clone_oracle import (
    unfold_no_memo, walk_clones, clone_canonical_step, clone_coefficients,
    clone_occurrence_environments, clone_environment_sums, project_one_occurrence,
)


def symmetric(value):
    return value / 2 + value.swapaxes(1, 2) / 2


def repeated():
    rng = np.random.default_rng(100)
    return Graph([
        Node("embedding", rng.normal(size=(2, 3)), source="x"),
        Node("square", symmetric(rng.normal(size=(3, 2, 2))), (0, 0)),
        Node("second_square", symmetric(rng.normal(size=(2, 3, 3))), (1, 1)),
    ], rng.normal(size=(2, 2)))


def deficient_repeated():
    graph = repeated()
    graph.nodes[1].core[1] = 2 * graph.nodes[1].core[0]
    graph.nodes[1].core[2] = 0.
    return graph


def diamond(matrix=None):
    matrix = np.array([[2., 1.], [0., 1.]]) if matrix is None else np.asarray(matrix, dtype=float)
    return Graph([
        Node("embedding", np.eye(2), source="x"),
        Node("left_branch", np.eye(2), (0,)),
        Node("right_branch", np.eye(2), (0,)),
        Node("merge", matrix[None, :, :], (1, 2)),
    ], np.ones((1, 1)))


class SharedReferenceTests(unittest.TestCase):
    def assertTensorEqual(self, left, right, tolerance=2e-11):
        scale = max(float(np.max(np.abs(right))), 1.)
        self.assertLessEqual(float(np.max(np.abs(left - right))), tolerance * scale)

    def assertResolvedSubspacesEqual(self, actual, expected):
        actual_values, actual_basis = np.linalg.eigh(actual)
        values, basis = np.linalg.eigh(expected)
        self.assertTensorEqual(actual_values, values)
        values, basis, actual_basis = values[::-1], basis[:, ::-1], actual_basis[:, ::-1]
        for rank in range(1, len(values) + 1):
            if rank < len(values) and values[rank - 1] - values[rank] <= 1e-8 * max(1., abs(values[0])):
                continue
            # Direct QR coordinate reconstruction, no projector/self-overlap.
            target, candidate = basis[:, :rank], actual_basis[:, :rank]
            q, triangular = np.linalg.qr(target, mode="reduced")
            rhs = q.T @ candidate
            coordinates = np.zeros_like(rhs)
            for row in range(rank - 1, -1, -1):
                coordinates[row] = (rhs[row] - triangular[row, row + 1:] @ coordinates[row + 1:]) / triangular[row, row]
            self.assertTensorEqual(target @ coordinates, candidate)

    def test_every_step_matches_independent_clone_and_complete_tensor(self):
        for original in (repeated(), diamond(), deficient_repeated()):
            shared = original.copy()
            independent = unfold_no_memo(original)
            original_dense, labels = clone_coefficients(independent)
            for index in range(len(shared.nodes)):
                report = canonical_step(shared, index)
                clone_canonical_step(independent, index)
                current_tree = unfold_no_memo(shared)
                for actual_node, expected_node in zip(walk_clones(current_tree), walk_clones(independent)):
                    self.assertEqual(actual_node.path, expected_node.path)
                    self.assertTensorEqual(actual_node.core, expected_node.core)
                current_dense, current_labels = clone_coefficients(current_tree)
                oracle_dense, oracle_labels = clone_coefficients(independent)
                self.assertEqual(labels, current_labels)
                self.assertEqual(labels, oracle_labels)
                self.assertTensorEqual(current_dense, original_dense)
                self.assertTensorEqual(oracle_dense, original_dense)
                # Compare the independently expanded cut environments after
                # EACH step, not only at the end of canonicalization.
                cuts = clone_occurrence_environments(current_tree)
                expected = clone_occurrence_environments(independent)
                self.assertEqual(set(cuts), set(expected))
                for path in cuts:
                    self.assertEqual(cuts[path][0], expected[path][0])
                    self.assertTensorEqual(cuts[path][1], expected[path][1])
                    self.assertResolvedSubspacesEqual(cuts[path][1], expected[path][1])
                expected_edges = sum(child == index for node in shared.nodes for child in node.children)
                self.assertEqual(report["occurrences_updated"], expected_edges + int(index == len(shared.nodes) - 1))
                self.assertLess(report["reconstruction_max_error"], 1e-10)

    def test_shared_environment_sum_matches_fully_contracted_occurrences(self):
        for original in (repeated(), diamond(), deficient_repeated()):
            graph = canonicalize(original)
            explicit = unfold_no_memo(graph)
            sums = occurrence_environment_sums(graph)
            expected = clone_environment_sums(explicit, len(graph.nodes))
            for actual, oracle in zip(sums, expected):
                self.assertTensorEqual(actual, oracle)
                self.assertResolvedSubspacesEqual(actual, oracle)
            counts = occurrence_counts(graph)
            self.assertEqual(counts, tuple(sum(node.origin == index for node in walk_clones(explicit)) for index in range(len(graph.nodes))))

    def test_repeated_edge_is_counted_twice(self):
        graph = canonicalize(Graph([
            Node("e", np.eye(2), source="x"),
            Node("square", np.array([[[1., 0.], [0., 2.]]]), (0, 0)),
        ], np.ones((1, 1))))
        cuts = clone_occurrence_environments(unfold_no_memo(graph))
        single = cuts[(0,)][1]
        self.assertTensorEqual(single, cuts[(1,)][1])
        self.assertTensorEqual(occurrence_environment_sums(graph)[0], 2 * single)
        self.assertEqual(occurrence_counts(graph), (2, 1))

    def test_symmetric_chain_matches_paper_environment_up_to_multiplicity(self):
        original = repeated()
        e, cores, head = dooms.orthogonalize(original.nodes[0].core,
            tuple(node.core for node in original.nodes[1:]), original.head)
        paper = dooms.environments(cores, head)
        shared = canonicalize(original)
        self.assertTensorEqual(shared.nodes[0].core, e)
        for node, core in zip(shared.nodes[1:], cores):
            self.assertTensorEqual(node.core, core)
        self.assertTensorEqual(shared.head, head)
        for actual, single, count in zip(occurrence_environment_sums(shared), paper, occurrence_counts(shared)):
            self.assertTensorEqual(actual, count * single)

    def test_shape_only_symmetric_completion_retains_deficient_rows(self):
        core = np.zeros((5, 2, 2))
        core[0] = np.array([[1., 2.], [2., 4.]])
        core[1] = 2 * core[0]
        original = Graph([Node("e", np.eye(2), source="x"), Node("deficient", core, (0, 0))], np.ones((1, 5)))
        graph = canonicalize(original)
        self.assertEqual(graph.nodes[1].core.shape, (3, 2, 2))
        self.assertTensorEqual(graph.nodes[1].core, graph.nodes[1].core.swapaxes(1, 2))
        self.assertTensorEqual(clone_coefficients(unfold_no_memo(graph))[0], clone_coefficients(unfold_no_memo(original))[0])
        self.assertTensorEqual(occurrence_environment_sums(graph)[0], clone_environment_sums(unfold_no_memo(graph), 2)[0])

    def test_zero_rank_does_not_request_another_factorization(self):
        original = Graph([Node("e", np.zeros((3, 2)), source="x"),
            Node("square", np.zeros((5, 3, 3)), (0, 0))], np.ones((1, 5)))
        graph = canonicalize(original)
        self.assertEqual(graph.nodes[0].core.shape, (2, 2))
        self.assertEqual(graph.nodes[1].core.shape, (3, 2, 2))
        self.assertTensorEqual(evaluate(graph, {"x": [2., -3.]}), np.zeros(1))

    def test_original_skew_cannot_hide_behind_child_shape_reduction(self):
        skew = np.array([[[0., 3.], [-3., 0.]]])
        graph = Graph([Node("narrow_input", np.ones((2, 1)), source="x"),
            Node("skew", skew, (0, 0))], np.ones((1, 1)))
        before = [node.core.copy() for node in graph.nodes]
        with self.assertRaises(ValueError):
            canonicalize(graph)
        with self.assertRaises(ValueError):
            canonical_step(graph, 0)
        for node, value in zip(graph.nodes, before):
            self.assertTrue(np.array_equal(node.core, value))

    def test_cross_modal_legs_are_not_indiscriminately_symmetrized(self):
        original = Graph([Node("image", np.eye(2), source="image"),
            Node("language", np.eye(2), source="language"),
            Node("cross", np.array([[[0., 2.], [-1., 0.]]]), (0, 1))], np.ones((1, 1)))
        unchanged = symmetrize_tied(original)
        self.assertTrue(np.array_equal(unchanged.nodes[2].core, original.nodes[2].core))
        inputs = {"image": [1., 0.], "language": [0., 1.]}
        self.assertTensorEqual(evaluate(canonicalize(original), inputs), np.array([2.]))
        incorrectly_symmetric = original.copy()
        incorrectly_symmetric.nodes[2].core = symmetric(incorrectly_symmetric.nodes[2].core)
        self.assertTensorEqual(evaluate(incorrectly_symmetric, inputs), np.array([0.5]))

    def test_explicit_symmetrization_changes_lift_but_not_tied_polynomial(self):
        original = Graph([Node("e", np.eye(2), source="x"),
            Node("tied", np.array([[[0., 2.], [-1., 0.]]]), (0, 0))], np.ones((1, 1)))
        with self.assertRaises(ValueError):
            canonicalize(original)
        processed = symmetrize_tied(original)
        self.assertGreater(float(np.max(np.abs(clone_coefficients(unfold_no_memo(original))[0]
            - clone_coefficients(unfold_no_memo(processed))[0]))), 0.1)
        for x in ([1., 2.], [3., -4.], [0., 1.]):
            self.assertTensorEqual(evaluate(original, {"x": x}), evaluate(processed, {"x": x}))

    def test_full_rank_common_gauges_preserve_entire_lift(self):
        for original in (repeated(), diamond()):
            graph = canonicalize(original)
            before = clone_coefficients(unfold_no_memo(graph))[0]
            transformed = apply_bases(graph, common_bases(graph))
            self.assertTensorEqual(clone_coefficients(unfold_no_memo(transformed))[0], before)
            for environment in occurrence_environment_sums(transformed):
                off_diagonal = environment - np.diag(np.diag(environment))
                self.assertTensorEqual(off_diagonal, np.zeros_like(off_diagonal))

    def test_every_gauge_prefix_and_physical_cut_matches_independent_clone(self):
        for original in (repeated(), diamond(), deficient_repeated()):
            graph = canonicalize(original)
            full_bases = common_bases(graph)
            for reduced in (False, True):
                bases = tuple(b[:, :max(1, b.shape[1] - 1)] if reduced else b for b in full_bases)
                oracle = unfold_no_memo(graph)
                for index, basis in enumerate(bases):
                    paths = [node.path for node in walk_clones(oracle) if node.origin == index]
                    for path in paths:
                        project_one_occurrence(oracle, path, basis)
                    prefix = bases[:index + 1] + tuple(np.eye(b.shape[0]) for b in bases[index + 1:])
                    actual = unfold_no_memo(apply_bases(graph, prefix))
                    self.assertTensorEqual(clone_coefficients(actual)[0], clone_coefficients(oracle)[0])
                    for actual_node, oracle_node in zip(walk_clones(actual), walk_clones(oracle)):
                        self.assertEqual(actual_node.path, oracle_node.path)
                        self.assertTensorEqual(actual_node.core, oracle_node.core)

    def test_common_eigenbasis_need_not_diagonalize_each_occurrence(self):
        graph = canonicalize(diamond())
        basis = common_bases(graph)[0]
        environments = [env for origin, env in clone_occurrence_environments(unfold_no_memo(graph)).values() if origin == 0]
        self.assertEqual(len(environments), 2)
        rotated = [np.einsum("ia,ij,jb->ab", basis, env, basis) for env in environments]
        self.assertGreater(abs(rotated[0][0, 1]), 0.1)
        self.assertGreater(abs(rotated[1][0, 1]), 0.1)
        self.assertLess(abs((rotated[0] + rotated[1])[0, 1]), 1e-10)

    def test_sum_is_exact_independent_single_cut_loss(self):
        graph = canonicalize(diamond())
        basis = common_bases(graph, (1, 2, 2, 1))[0]
        original_tensor = clone_coefficients(unfold_no_memo(graph))[0]
        loss = 0.
        for node in walk_clones(unfold_no_memo(graph)):
            if node.origin == 0:
                one_cut = unfold_no_memo(graph)
                project_one_occurrence(one_cut, node.path, basis)
                difference = original_tensor - clone_coefficients(one_cut)[0]
                loss += float(np.sum(difference * difference))
        environment = occurrence_environment_sums(graph)[0]
        predicted = float(np.trace(environment) - np.einsum("ia,ij,ja->", basis, environment, basis))
        self.assertAlmostEqual(loss, predicted, places=10)

    def test_cross_occurrence_derivatives_cancel(self):
        antisymmetric = np.array([[0., 1.], [-1., 0.]])
        graph = diamond(antisymmetric)
        tree = unfold_no_memo(graph)
        diagonal_change = np.diag([1., -1.])
        leaf_paths = [node.path for node in walk_clones(tree) if node.origin == 0]
        derivatives = [clone_coefficients(tree, replacements={path: diagonal_change})[0] for path in leaf_paths]
        independent = sum(float(np.sum(value * value)) for value in derivatives)
        joint = derivatives[0] + derivatives[1]
        cross = 2 * float(np.sum(derivatives[0] * derivatives[1]))
        self.assertAlmostEqual(independent, 4.)
        self.assertAlmostEqual(cross, -4.)
        self.assertTensorEqual(joint, np.zeros_like(joint))
        for step in (-0.3, 0.2):
            changed = graph.copy()
            changed.nodes[0].core = np.eye(2) + step * diagonal_change
            actual = clone_coefficients(unfold_no_memo(changed))[0]
            self.assertTensorEqual(actual, (1 - step * step) * antisymmetric[None])

    def test_fixed_lift_environment_is_not_polynomial_intrinsic(self):
        original = diamond(np.array([[0., 1.], [-1., 0.]]))
        zero = diamond(np.zeros((2, 2)))
        for x in ([1., 2.], [-3., 5.], [2., 2.]):
            self.assertTensorEqual(evaluate(original, {"x": x}), evaluate(zero, {"x": x}))
        nonzero_environment = occurrence_environment_sums(canonicalize(original))[0]
        zero_environment = occurrence_environment_sums(canonicalize(zero))[0]
        self.assertTensorEqual(nonzero_environment, 2 * np.eye(2))
        self.assertTensorEqual(zero_environment, np.zeros((2, 2)))

    def test_finite_tied_projection_is_not_sum_of_single_cut_losses(self):
        graph = canonicalize(diamond(np.diag([2., 1.])))
        original = clone_coefficients(unfold_no_memo(graph))[0]
        basis = np.array([[1.], [0.]])
        individual = 0.
        paths = [node.path for node in walk_clones(unfold_no_memo(graph)) if node.origin == 0]
        for path in paths:
            tree = unfold_no_memo(graph)
            project_one_occurrence(tree, path, basis)
            error = original - clone_coefficients(tree)[0]
            individual += float(np.sum(error * error))
        bases = [np.eye(node.core.shape[0]) for node in graph.nodes]
        bases[0] = basis
        joint = original - clone_coefficients(unfold_no_memo(apply_bases(graph, bases)))[0]
        self.assertAlmostEqual(individual, 2.)
        self.assertAlmostEqual(float(np.sum(joint * joint)), 1.)

    def test_symmetric_common_top_direction_is_not_finite_tied_optimum(self):
        core = np.array([[[2., 0.], [0., 0.]], [[0., 2.], [2., 0.]]])
        graph = canonicalize(Graph([Node("e", np.eye(2), source="x"),
            Node("symmetric", core, (0, 0))], np.eye(2)))
        environment = occurrence_environment_sums(graph)[0]
        self.assertTensorEqual(environment, np.diag([16., 8.]))
        leading = common_bases(graph, (1, 2))[0]
        alternate = np.array([[np.sqrt(2 / 3)], [np.sqrt(1 / 3)]])
        before = clone_coefficients(unfold_no_memo(graph))[0]
        losses = []
        for basis in (leading, alternate):
            candidate = apply_bases(graph, (basis, np.eye(2)))
            error = before - clone_coefficients(unfold_no_memo(candidate))[0]
            losses.append(float(np.sum(error * error)))
        self.assertAlmostEqual(losses[0], 8.)
        self.assertAlmostEqual(losses[1], 20 / 3)
        self.assertGreater(losses[0], losses[1])

    def test_noncanonical_shortcut_and_oversized_oracle_fail_closed(self):
        with self.assertRaises(ValueError):
            occurrence_environment_sums(diamond())
        with self.assertRaises(ValueError):
            clone_coefficients(unfold_no_memo(repeated()), maximum_elements=2)
        one_leaf = unfold_no_memo(Graph([Node("e", np.eye(2), source="x")], np.ones((3, 2))))
        for limit, cut in ((2, None), (2, ()), (4, None)):
            with self.assertRaises(ValueError):
                clone_coefficients(one_leaf, cut=cut, maximum_elements=limit)


if __name__ == "__main__":
    unittest.main()
