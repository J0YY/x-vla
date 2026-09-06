"""Small adversarial checks against independent indices, not policy probes.

Reference: Dooms et al., arXiv:2504.02667v1 §2 and Appendix G, Algorithms 1–3.
The only doubled tensor contraction here is the required downstream environment.
No production ODT/model code is imported.
"""

import itertools
import unittest

import numpy as np

from research.odt_reference import dooms
from research.odt_reference.weights import homogeneous_affine, symmetric_cp_ffn


def coefficients(embedding, cores, head):
    """Expand ordered physical leaves, with independent copies at every fork."""
    tensor = np.array(embedding, copy=True)
    for core in cores:
        tensor = np.einsum("oij,ia,jb->oab", core, tensor, tensor).reshape(core.shape[0], -1)
    return head @ tensor


def push_inputs(core, factor):
    """Independent literal sum for C' = C (R tensor R), in both occurrences."""
    result = np.zeros((core.shape[0], factor.shape[1], factor.shape[1]))
    for o, a, b, i, j in itertools.product(
            range(core.shape[0]), range(factor.shape[1]), range(factor.shape[1]),
            range(core.shape[1]), range(core.shape[2])):
        result[o, a, b] += core[o, i, j] * factor[i, a] * factor[j, b]
    return result


def paper_trace(embedding, cores, head):
    """Literal Algorithm 1, packing symmetric coordinates with explicit loops."""
    e, cs, h = embedding.copy(), [c.copy() for c in cores], head.copy()
    trace = []
    for stage in range(len(cs) + 1):
        current = e if stage == 0 else cs[stage - 1]
        if stage == 0:
            q, rt = np.linalg.qr(current.T, mode="reduced")
            replacement = q.T
        else:
            pairs = list(itertools.combinations_with_replacement(range(current.shape[1]), 2))
            packed = np.array([[current[o, i, j] * (1.0 if i == j else np.sqrt(2.0))
                                for i, j in pairs] for o in range(current.shape[0])])
            q, rt = np.linalg.qr(packed.T, mode="reduced")
            replacement = np.zeros((q.shape[1], current.shape[1], current.shape[2]))
            for a, (i, j) in itertools.product(range(q.shape[1]), pairs):
                replacement[a, i, j] = replacement[a, j, i] = q[pairs.index((i, j)), a] / (
                    1.0 if i == j else np.sqrt(2.0))
        if stage == 0:
            e = replacement
        else:
            cs[stage - 1] = replacement
        if stage == len(cs):
            h = h @ rt.T
        else:
            cs[stage] = push_inputs(cs[stage], rt.T)
        trace.append((e.copy(), tuple(c.copy() for c in cs), h.copy()))
    return trace


def explicit_environment(embedding, cores, head, bond, path=None):
    """Literal two-copy downstream contraction, with no opened-map array.

    Both copies independently evaluate every physical sibling subtree without
    memoization. Only corresponding external indices are identified. Thus no
    sibling-isometry cancellation or unfolding-overlap shortcut is assumed.
    """
    path = (0,)*(len(cores)-bond) if path is None else path
    physical_width = embedding.shape[1]
    def sibling(level, index):
        if level == 0:
            return embedding[:, index]
        stride = physical_width ** (2 ** (level - 1))
        left = sibling(level - 1, index // stride)
        right = sibling(level - 1, index % stride)
        return np.einsum("oij,i,j->o", cores[level - 1], left, right)
    width = embedding.shape[0] if bond == 0 else cores[bond - 1].shape[0]
    def downstream(output, coordinate, context):
        value = np.eye(width)[:, coordinate]
        for layer, index in zip(range(bond, len(cores)), context):
            other = sibling(layer, index)
            left, right = (value, other) if path[len(cores) - 1 - layer] == 0 else (other, value)
            value = np.einsum("oij,i,j->o", cores[layer], left, right)
        return head[output] @ value
    context_dimensions = [physical_width ** (2 ** layer) for layer in range(bond, len(cores))]
    environment = np.zeros((width, width))
    for a, b in itertools.product(range(width), repeat=2):
        for output in range(head.shape[0]):
            for context in itertools.product(*(range(size) for size in context_dimensions)):
                environment[a, b] += downstream(output, a, context) * downstream(output, b, context)
    return environment


class SymmetricWeightTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1247)

    def test_export_matches_raw_nested_weight_indices_and_biases(self):
        l, r, u = self.rng.normal(size=(3, 2)), self.rng.normal(size=(3, 2)), self.rng.normal(size=(2, 3))
        bl, br, bu, w = self.rng.normal(size=3), self.rng.normal(size=3), self.rng.normal(size=2), self.rng.normal(size=(2, 2))
        actual = symmetric_cp_ffn(l, r, u, bl, br, bu, w)
        expected = np.zeros((3, 3, 3))
        expected[0, 0, 0] = 1.0
        for o, i, j, a in itertools.product(range(2), range(3), range(3), range(3)):
            li, lj = (bl[a] if i == 0 else l[a, i-1]), (bl[a] if j == 0 else l[a, j-1])
            ri, rj = (br[a] if i == 0 else r[a, i-1]), (br[a] if j == 0 else r[a, j-1])
            expected[o+1, i, j] += u[o, a] * (li*rj + lj*ri) / 2
        for o in range(2):
            expected[o+1, 0, 0] += bu[o]
            for i in range(2):
                expected[o+1, i+1, 0] += w[o, i] / 2
                expected[o+1, 0, i+1] += w[o, i] / 2
        np.testing.assert_allclose(actual, expected, atol=2e-15)
        np.testing.assert_array_equal(actual, actual.swapaxes(1, 2))
        for x in self.rng.normal(size=(6, 2)):
            z = np.r_[1., x]
            np.testing.assert_allclose(np.einsum("oij,i,j->o", actual, z, z),
                                       np.r_[1., u @ ((l @ x + bl)*(r @ x + br)) + bu + w @ x], atol=1e-13)
        np.testing.assert_allclose(actual, symmetric_cp_ffn(r, l, u, br, bl, bu, w))

    def test_tied_weights_and_antisymmetric_null_component(self):
        l, u = self.rng.normal(size=(3, 2)), self.rng.normal(size=(2, 3))
        core = symmetric_cp_ffn(l, l, u)
        perturbation = self.rng.normal(size=core.shape)
        perturbation -= perturbation.swapaxes(1, 2)
        ordered = core + perturbation
        self.assertGreater(float(np.max(np.abs(ordered-core))), 0.1)
        np.testing.assert_allclose((ordered+ordered.swapaxes(1, 2))/2, core, atol=1e-15)
        for x in self.rng.normal(size=(4, 3)):
            np.testing.assert_allclose(np.einsum("oij,i,j->o", ordered, x, x),
                                       np.einsum("oij,i,j->o", core, x, x), atol=1e-13)

    def test_homogeneous_affine_and_invalid_weights(self):
        w, b = self.rng.normal(size=(3, 2)), self.rng.normal(size=3)
        e = homogeneous_affine(w, b)
        for x in self.rng.normal(size=(3, 2)):
            np.testing.assert_allclose(e @ np.r_[1., x], np.r_[1., w @ x+b])
        with self.assertRaises(ValueError):
            symmetric_cp_ffn(w, w, np.ones((2, 4)))
        with self.assertRaises(ValueError):
            homogeneous_affine(w, np.array([np.nan, 0., 0.]))
        with self.assertRaises(ValueError):
            homogeneous_affine(w.astype(complex))


class PaperReferenceTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(843)
        self.embedding = self.rng.normal(size=(3, 2))
        self.cores = tuple(self.symmetric(self.rng.normal(size=shape)) for shape in ((3, 3, 3), (2, 3, 3)))
        self.head = self.rng.normal(size=(2, 2))

    @staticmethod
    def symmetric(c):
        return (c+c.swapaxes(1, 2))/2

    def assert_same_subspace(self, expected, actual):
        # Solve for coordinates using direct QR and literal back-substitution.
        # No projection matrix or same-basis overlap is constructed.
        q, triangular = np.linalg.qr(expected, mode="reduced")
        rhs = q.T @ actual
        coordinates = np.zeros_like(rhs)
        for row in range(triangular.shape[0]-1, -1, -1):
            coordinates[row] = (rhs[row] - triangular[row, row+1:] @ coordinates[row+1:]) / triangular[row, row]
        np.testing.assert_allclose(expected @ coordinates, actual, atol=2e-11, rtol=2e-11)

    def test_symmetric_qr_completion_zero_deficient_and_structurally_tall(self):
        fixtures = [np.zeros((5, 2, 2)), self.symmetric(self.rng.normal(size=(5, 2, 2))),
                    self.symmetric(self.rng.normal(size=(4, 3, 3)))]
        fixtures[-1][1:] = fixtures[-1][0]
        for core in fixtures:
            with self.subTest(shape=core.shape, zero=not np.any(core)):
                r, q = dooms.symmetric_rq(core)
                self.assertEqual(q.shape[0], min(core.shape[0], core.shape[1]*(core.shape[1]+1)//2))
                np.testing.assert_array_equal(q, q.swapaxes(1, 2))
                np.testing.assert_allclose(np.einsum("ok,kij->oij", r, q), core, atol=2e-14)
                # A second direct QR must reconstruct an already-isometric row set
                # with a signed diagonal triangular factor, including completion.
                _, rt = np.linalg.qr(q.reshape(q.shape[0], -1).T, mode="reduced")
                np.testing.assert_allclose(rt, np.diag(np.diag(rt)), atol=2e-14)
                np.testing.assert_allclose(np.abs(np.diag(rt)), 1., atol=2e-14)

    def test_dense_qr_can_break_symmetry_in_a_null_completion(self):
        core = np.zeros((2, 2, 2))
        core[0, 0, 0] = 1.
        q, r = np.linalg.qr(core.reshape(2, 4).T, mode="reduced")
        dense_q = q.T.reshape(2, 2, 2)
        np.testing.assert_allclose(r.T @ q.T, core.reshape(2, 4))
        self.assertGreater(float(np.max(np.abs(dense_q-dense_q.swapaxes(1, 2)))), 0.5)
        _, symmetric_q = dooms.symmetric_rq(core)
        np.testing.assert_array_equal(symmetric_q, symmetric_q.swapaxes(1, 2))
        broken = core.copy()
        broken[0, 0, 1] = 1.
        with self.assertRaises(ValueError):
            dooms.symmetric_rq(broken)

    def test_reject_skew_before_shape_reducing_embedding_can_hide_it(self):
        embedding = np.ones((2, 1))
        skew = np.array([[[0., 1.], [-1., 0.]]])
        with self.assertRaises(ValueError):
            dooms.orthogonalize(embedding, (skew,), np.ones((1, 1)))

    def test_every_algorithm1_prefix_against_independent_interpreter(self):
        expected = coefficients(self.embedding, self.cores, self.head)
        trace = paper_trace(self.embedding, self.cores, self.head)
        for count, interpreted in enumerate(trace):
            # Identity head returns the accumulated factor at this exact prefix.
            old_width = self.embedding.shape[0] if count == 0 else self.cores[count-1].shape[0]
            e, done, factor = dooms.orthogonalize(self.embedding, self.cores[:count], np.eye(old_width))
            remaining = [c.copy() for c in self.cores[count:]]
            if remaining:
                remaining[0] = push_inputs(remaining[0], factor)
                h = self.head
            else:
                h = self.head @ factor
            actual = (e, tuple(done)+tuple(remaining), h)
            np.testing.assert_allclose(coefficients(*actual), expected, atol=2e-12, rtol=2e-12)
            for left, right in zip((actual[0], *actual[1], actual[2]), (interpreted[0], *interpreted[1], interpreted[2])):
                np.testing.assert_allclose(left, right, atol=2e-12, rtol=2e-12)
            for core in actual[1]:
                np.testing.assert_allclose(core, core.swapaxes(1, 2), atol=2e-14)

    def test_actual_environments_and_retained_subspaces(self):
        e, cs, h = dooms.orthogonalize(self.embedding, self.cores, self.head)
        actual = dooms.environments(cs, h)
        bases = dooms.eigenspaces(actual)
        for bond, (environment, basis) in enumerate(zip(actual, bases)):
            expected = explicit_environment(e, cs, h, bond)
            np.testing.assert_allclose(environment, expected, atol=2e-11, rtol=2e-12)
            for path in itertools.product((0, 1), repeat=len(cs)-bond):
                np.testing.assert_allclose(environment, explicit_environment(e, cs, h, bond, path),
                                           atol=2e-11, rtol=2e-12)
            values, vectors = np.linalg.eigh(expected)
            values, vectors = values[::-1], vectors[:, ::-1]
            np.testing.assert_allclose(environment @ basis, basis*values, atol=2e-11, rtol=2e-12)
            for rank in range(1, len(values)+1):
                if rank == len(values) or values[rank-1]-values[rank] > 1e-8*max(1., abs(values[0])):
                    self.assert_same_subspace(vectors[:, :rank], basis[:, :rank])

    def test_deficient_cores_through_all_three_passes(self):
        dependent = self.cores[0].copy()
        dependent[1], dependent[2] = 2 * dependent[0], 0.
        for first in (dependent, np.zeros_like(dependent)):
            with self.subTest(zero=not np.any(first)):
                cores = (first, self.cores[1])
                expected = coefficients(self.embedding, cores, self.head)
                for snapshot in paper_trace(self.embedding, cores, self.head):
                    np.testing.assert_allclose(coefficients(*snapshot), expected, atol=2e-11, rtol=2e-12)
                    for core in snapshot[1]:
                        np.testing.assert_allclose(core, core.swapaxes(1, 2), atol=2e-14)
                e, cs, h = dooms.orthogonalize(self.embedding, cores, self.head)
                envs = dooms.environments(cs, h)
                for bond, env in enumerate(envs):
                    np.testing.assert_allclose(env, explicit_environment(e, cs, h, bond), atol=2e-10, rtol=2e-12)
                bases = dooms.eigenspaces(envs)
                for count in range(len(bases) + 1):
                    gauges = bases[:count] + tuple(np.eye(b.shape[0]) for b in bases[count:])
                    changed = dooms.absorb(e, cs, h, gauges)
                    np.testing.assert_allclose(coefficients(*changed), expected, atol=2e-11, rtol=2e-12)
                    for core in changed[1]:
                        np.testing.assert_allclose(core, core.swapaxes(1, 2), atol=2e-14)

    def test_raw_weight_export_through_all_passes_and_actual_environments(self):
        embedding_weight, embedding_bias = self.rng.normal(size=(2, 1)), self.rng.normal(size=2)
        embedding = homogeneous_affine(embedding_weight, embedding_bias)
        head = self.rng.normal(size=(2, 3))
        for dependent in (False, True):
            left, right = self.rng.normal(size=(2, 2)), self.rng.normal(size=(2, 2))
            output = self.rng.normal(size=(2, 2))
            lb, rb, ob = (self.rng.normal(size=2) for _ in range(3))
            residual = self.rng.normal(size=(2, 2))
            if dependent:
                output[1] = 2 * output[0]
                ob, residual = np.zeros(2), np.zeros((2, 2))
            core = symmetric_cp_ffn(left, right, output, lb, rb, ob, residual)
            cores = (core, core.copy())
            expected = coefficients(embedding, cores, head)
            for snapshot in paper_trace(embedding, cores, head):
                np.testing.assert_allclose(coefficients(*snapshot), expected, atol=2e-10, rtol=2e-12)
                for current in snapshot[1]:
                    np.testing.assert_allclose(current, current.swapaxes(1, 2), atol=2e-14)
            e, cs, h = dooms.orthogonalize(embedding, cores, head)
            envs = dooms.environments(cs, h)
            bases = dooms.eigenspaces(envs)
            for bond, env in enumerate(envs):
                oracle = explicit_environment(e, cs, h, bond)
                np.testing.assert_allclose(env, oracle, atol=2e-9, rtol=2e-12)
                values, vectors = np.linalg.eigh(oracle)
                values, vectors = values[::-1], vectors[:, ::-1]
                for rank in range(1, len(values) + 1):
                    if rank == len(values) or values[rank-1] - values[rank] > 1e-8 * max(1., abs(values[0])):
                        self.assert_same_subspace(vectors[:, :rank], bases[bond][:, :rank])
            changed = dooms.absorb(e, cs, h, bases)
            np.testing.assert_allclose(coefficients(*changed), expected, atol=2e-10, rtol=2e-12)
            for x in (-0.2, 0.3):
                raw = embedding_weight[:, 0] * x + embedding_bias
                for _ in cores:
                    raw = output @ ((left @ raw + lb) * (right @ raw + rb)) + ob + residual @ raw
                raw = head @ np.r_[1., raw]
                value = changed[0] @ np.array([1., x])
                for current in changed[1]:
                    value = np.einsum("oij,i,j->o", current, value, value)
                np.testing.assert_allclose(changed[2] @ value, raw, atol=2e-10, rtol=2e-12)

    def test_algorithm3_every_gauge_prefix_full_coefficients_and_physical_masks(self):
        e, cs, h = dooms.orthogonalize(self.embedding, self.cores, self.head)
        environments = dooms.environments(cs, h)
        bases = dooms.eigenspaces(environments)
        expected = coefficients(e, cs, h)
        identities = [np.eye(b.shape[0]) for b in bases]
        for count in range(len(bases)+1):
            gauges = tuple(bases[:count])+tuple(identities[count:])
            changed = dooms.absorb(e, cs, h, gauges)
            np.testing.assert_allclose(coefficients(*changed), expected, atol=2e-11, rtol=2e-12)
            for core in changed[1]:
                np.testing.assert_allclose(core, core.swapaxes(1, 2), atol=2e-14)
        full_e, full_cs, full_h = dooms.absorb(e, cs, h, bases)
        ranks = [max(1, b.shape[1]-1) for b in bases]
        reduced = dooms.absorb(e, cs, h, dooms.eigenspaces(environments, ranks))
        sliced = (full_e[:ranks[0]], tuple(c[:ranks[i+1], :ranks[i], :ranks[i]] for i, c in enumerate(full_cs)), full_h[:, :ranks[-1]])
        masked_e, masked_cs = full_e.copy(), [c.copy() for c in full_cs]
        masked_e[ranks[0]:] = 0.
        for core, rank in zip(masked_cs, ranks[1:]):
            core[rank:] = 0.
        np.testing.assert_allclose(coefficients(*reduced), coefficients(*sliced), atol=2e-11, rtol=2e-12)
        np.testing.assert_allclose(coefficients(*reduced), coefficients(masked_e, masked_cs, full_h), atol=2e-11, rtol=2e-12)

    def test_degenerate_environment_requires_no_unique_eigenvector(self):
        environment = np.diag([4., 4., 0.])
        basis, = dooms.eigenspaces((environment,))
        np.testing.assert_allclose(environment @ basis, basis*np.array([4., 4., 0.]), atol=1e-14)
        self.assert_same_subspace(np.eye(3)[:, :2], basis[:, :2])
        # A cutoff within the tied pair is explicitly not compared as unique.
        e = np.eye(3)
        changed = dooms.absorb(e, (), np.diag([2., 2., 0.]), (basis,))
        np.testing.assert_allclose(coefficients(*changed), np.diag([2., 2., 0.]), atol=1e-14)

    def test_finite_environment_symmetrization_avoids_intermediate_overflow(self):
        environment = np.diag([1e308, 5e307])
        basis, = dooms.eigenspaces((environment,))
        np.testing.assert_array_equal(np.abs(basis), np.eye(2))


if __name__ == "__main__":
    unittest.main()
