"""Complete-block gates. Synthetic weights are never labeled trained results."""
import unittest
import numpy as np

from research.odt_reference import dooms
from research.odt_reference.block import BlockSpec, NORM_SITES, compile_block, evaluate_block
from research.odt_reference.block_oracle import block_forward
from research.odt_reference.clone_oracle import walk_clones, clone_occurrence_environments
from research.odt_reference.clone_passes import canonicalize_clones, aggregate_environments, occurrence_environments, apply_common_bases
from research.odt_reference.shared_dag import canonical_step, occurrence_environment_sums, apply_bases
from research.odt_reference.test_block_oracle import synthetic_identity_norm_weights
from research.odt_reference.test_shared_dag import repeated, diamond


def fixture(width=1):
    rng = np.random.default_rng(904)
    weights = synthetic_identity_norm_weights(width)
    for name in weights:
        if name.endswith(".weight"):
            weights[name] = rng.normal(size=weights[name].shape) * 0.3
        elif name.endswith(".bias"):
            weights[name] = rng.normal(size=weights[name].shape) * 0.05
    for index, name in enumerate(NORM_SITES):
        weights[name + ".running_ms"] = np.array(1. + 0.001 * index)
        weights[name + ".pa"] = np.array([0.9 + 0.005 * index, 0.2 + 0.003 * index, 0.01 + 0.001 * index])
        weights[name + ".pb"] = np.array([1., 0.25 + 0.004 * index, 0.02 + 0.002 * index])
    weights["attn_gain"], weights["ffn_gain"] = np.array(0.4), np.array(0.3)
    return weights


class CompleteBlockTests(unittest.TestCase):
    def assertClose(self, actual, expected, rtol=2e-10):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=2e-11)

    def assertRetainedSpans(self, environment, basis):
        values, independent = np.linalg.eigh(environment)
        values, independent = values[::-1], independent[:, ::-1]
        for rank in range(1, len(values) + 1):
            if rank < len(values) and values[rank-1] - values[rank] <= 1e-8 * max(1., abs(values[0])):
                continue
            target, candidate = independent[:, :rank], basis[:, :rank]
            q, r = np.linalg.qr(target, mode="reduced")
            rhs, coordinates = q.T @ candidate, np.zeros((rank, rank))
            for row in range(rank - 1, -1, -1):
                coordinates[row] = (rhs[row] - r[row, row+1:] @ coordinates[row+1:]) / r[row, row]
            self.assertClose(target @ coordinates, candidate)

    def test_all_tokens_heads_biases_residuals_and_empty_rows(self):
        raw = np.array([[[0.2, -0.1], [0.4, 0.3]], [[0.1, 0.5], [-0.2, 0.1]]])
        weights = fixture(2)
        for mask in (np.tril(np.ones((2, 2))), np.array([[0., 0.], [1., 1.]])):
            for heads in (1, 2):
                expected = block_forward(raw, weights, n_heads=heads, mask=mask)
                joint = compile_block(weights, spec=BlockSpec(heads), mask=mask)
                self.assertClose(evaluate_block(joint, raw), expected.reshape(len(raw), -1))
                for token in range(2):
                    graph = compile_block(weights, spec=BlockSpec(heads), mask=mask, selected_token=token)
                    self.assertClose(evaluate_block(graph, raw), expected[:, token])
                    for node in graph.nodes:
                        if len(node.children) == 2 and node.children[0] == node.children[1]:
                            self.assertClose(node.core, node.core.swapaxes(1, 2))

    def test_clone_environment_elimination_against_literal_contraction(self):
        for graph in (repeated(), diamond()):
            clones = canonicalize_clones(graph)
            actual = occurrence_environments(clones)
            literal = clone_occurrence_environments(clones.tree)
            self.assertEqual(actual.keys(), literal.keys())
            for path in actual:
                self.assertClose(actual[path][1], literal[path][1])

    def test_complete_two_token_block_every_qr_environment_and_gauge(self):
        weights, mask = fixture(), np.tril(np.ones((2, 2)))
        raw = np.array([[[0.2], [0.4]], [[-0.1], [0.1]]])
        for deficient in (False, True):
            if deficient:
                weights["attn.wq1.weight"] = np.zeros((1, 1))
            expected = block_forward(raw, weights, n_heads=1, mask=mask).reshape(len(raw), -1)
            original = compile_block(weights, spec=BlockSpec(1), mask=mask)
            shared = original.copy()
            def check_step(tree, index):
                record = canonical_step(shared, index)
                self.assertLess(record["reconstruction_max_error"], 2e-10)
                for node in walk_clones(tree):
                    self.assertClose(shared.nodes[node.origin].core, node.core)
                self.assertClose(shared.head, tree.head)
            clones = canonicalize_clones(original, after_step=check_step)
            shared.canonical = True
            self.assertClose(evaluate_block(shared, raw), expected)
            actual, independent = occurrence_environment_sums(shared), aggregate_environments(clones)
            for environment, oracle in zip(actual, independent):
                self.assertClose(environment, oracle)
            bases = dooms.eigenspaces(actual)
            for environment, basis in zip(independent, bases):
                self.assertRetainedSpans(environment, basis)
            gauged = apply_bases(shared, bases)
            oracle = apply_common_bases(clones, bases)
            for node in walk_clones(oracle):
                self.assertClose(gauged.nodes[node.origin].core, node.core)
            self.assertClose(gauged.head, oracle.head)
            self.assertClose(evaluate_block(gauged, raw), expected)

    def test_shape_gate_before_large_ffn_allocation(self):
        with self.assertRaises(ValueError):
            compile_block(fixture(2), spec=BlockSpec(1), mask=np.ones((1, 1)), selected_token=0, maximum_elements=10)

    def test_invalid_weights_and_hidden_sites_fail_closed(self):
        for key, value in (("ffn.down.weight", np.ones((3, 2))), ("ffn.left.weight", np.ones((2, 3))),
                           ("attn.wk1.weight", np.ones((3, 2))), ("attn.rn_k2.initialized", np.array(False)),
                           ("attn.rn_q1.pb", np.array([1., -0.1, 0.]))):
            weights = fixture(2)
            weights[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                compile_block(weights, spec=BlockSpec(1), mask=np.zeros((1, 1)), selected_token=0)
        with self.assertRaises(ValueError):
            BlockSpec(1, residual=False)
        with self.assertRaises(ValueError):
            compile_block(fixture(2), spec=BlockSpec(1), mask=np.empty((0, 0)))


if __name__ == "__main__":
    unittest.main()
