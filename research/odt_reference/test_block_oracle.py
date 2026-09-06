"""Hand-computable synthetic checks for the independent raw block oracle."""

import unittest

import numpy as np

from research.odt_reference.block_oracle import block_forward


def synthetic_identity_norm_weights(width=1):
    """Test-only exact identity Padé buffers, not a trained checkpoint."""
    weights = {"attn_gain": np.array(1.), "ffn_gain": np.array(1.)}
    for name in ("rbn_attn", "attn.rn_q1", "attn.rn_k1", "attn.rn_q2", "attn.rn_k2", "rbn_ffn"):
        weights.update({name + ".running_ms": np.array(1.), name + ".initialized": np.array(True),
                        name + ".pa": np.array([1., 0., 0.]), name + ".pb": np.array([1., 0., 0.])})
    for name in ("attn.wq1", "attn.wk1", "attn.wq2", "attn.wk2", "attn.wv", "attn.wo",
                 "ffn.left", "ffn.right", "ffn.down"):
        weights[name + ".weight"] = np.eye(width)
        weights[name + ".bias"] = np.zeros(width)
    return weights


class BlockOracleTests(unittest.TestCase):
    def test_two_token_causal_scalar_block_by_hand(self):
        weights = synthetic_identity_norm_weights()
        raw = np.array([[[1.], [2.]]])
        actual, trace = block_forward(raw, weights, n_heads=1, mask=np.tril(np.ones((2, 2))), return_trace=True)
        # q=k=v=x. The two score systems yield x_i²*x_j², then multiply x_j.
        residual = np.array([[[2.], [2. + 36./np.sqrt(2.)]]])
        np.testing.assert_allclose(trace["residual1"], residual)
        np.testing.assert_allclose(actual, residual + residual**2)
        self.assertEqual(set(("q1", "k1", "q2", "k2")) - trace.keys(), set())

    def test_all_biases_residual_gains_and_empty_mask_row(self):
        weights = synthetic_identity_norm_weights()
        weights["attn_gain"], weights["ffn_gain"] = np.array(0.5), np.array(0.25)
        weights["attn.wo.bias"] = np.array([3.])
        weights["ffn.left.bias"], weights["ffn.right.bias"] = np.array([1.]), np.array([-2.])
        weights["ffn.down.bias"] = np.array([4.])
        raw = np.array([[[2.]]])
        actual = block_forward(raw, weights, n_heads=1, mask=np.zeros((1, 1)))
        residual = 2. + 0.5*3.
        self.assertAlmostEqual(float(actual[0, 0, 0]), residual + 0.25*((residual+1.)*(residual-2.)+4.))

    def test_head_score_scaling_and_head_order(self):
        weights = synthetic_identity_norm_weights(width=2)
        weights["ffn_gain"] = np.array(0.)
        raw = np.array([[[1., 2.]]])
        one_head = block_forward(raw, weights, n_heads=1, mask=np.ones((1, 1)))
        two_heads = block_forward(raw, weights, n_heads=2, mask=np.ones((1, 1)))
        np.testing.assert_allclose(one_head, raw + (25./4.)*raw)
        np.testing.assert_allclose(two_heads, raw + np.array([[[1., 32.]]]))

    def test_nontrivial_six_pade_sites_against_scalar_formula(self):
        weights = synthetic_identity_norm_weights()
        names = ("rbn_attn", "attn.rn_q1", "attn.rn_k1", "attn.rn_q2", "attn.rn_k2", "rbn_ffn")
        for index, name in enumerate(names):
            weights[name + ".running_ms"] = np.array(1. + index/10.)
            weights[name + ".pa"] = np.array([1., 0.2, 0.01])
            weights[name + ".pb"] = np.array([1., 0.3, 0.02])
        def scalar_norm(value, index):
            scale = 1. + index/10.
            v = (value*value + 1e-6)/scale
            return value*(1.+0.2*v+0.01*v*v)/(1.+0.3*v+0.02*v*v)/np.sqrt(scale)
        x = 0.2
        pre = scalar_norm(x, 0)
        q1, k1, q2, k2 = (scalar_norm(pre, index) for index in (1, 2, 3, 4))
        residual = x + q1*k1*q2*k2*pre
        expected = residual + scalar_norm(residual, 5)**2
        actual = block_forward(np.array([[[x]]]), weights, n_heads=1, mask=np.ones((1, 1)))
        self.assertAlmostEqual(float(actual[0, 0, 0]), expected, places=15)

    def test_invalid_source_configuration_and_pole_fail_closed(self):
        weights = synthetic_identity_norm_weights()
        raw, mask = np.ones((1, 1, 1)), np.ones((1, 1))
        with self.assertRaises(ValueError):
            block_forward(raw, weights, n_heads=2, mask=mask)
        with self.assertRaises(ValueError):
            block_forward(raw, weights, n_heads=1, mask=mask/2.)
        weights["rbn_attn.pb"] = np.array([-(1.+1e-6), 1., 0.])
        with self.assertRaises(FloatingPointError):
            block_forward(raw, weights, n_heads=1, mask=mask)


if __name__ == "__main__":
    unittest.main()
