"""Minimal prototype: HIERARCHICAL / PRODUCT tensor-pure mixture-of-experts head.

Standalone NumPy (the repo's Python is 3.14 -> no torch; this proves the math and
the training loop cheaply, exactly like ``born_mps_proto.py`` /
``energy_ebm_proto.py``). The production head is the identical construction in
torch: every gate and every leaf/factor expert is a ``BilinearFFN`` (CP core, see
``xvla/nn/bilinear.py``), so each folds to its own dense degree-2 tensor
(``dense_core`` -> the object ``xvla/train/odt.py`` diagonalizes). The only
non-polynomial ops (the routing argmax/threshold and the decode) live
OUT-OF-GRAPH in the controller -- the sanctioned category (spec Section 11), the
same reframe as OpenVLA's argmax and the gripper threshold.

WHY (DEVLOG 2026-07-17): the linear+MSE action head emits only E[a|obs]; on a
multimodal target it averages the modes (bimodal gripper -> ~0 -> 0% closed-loop;
the arm trajectory is mode-averaged too). A flat K-candidate head
(``xvla/nn/multihypothesis.py``, the sibling) fixes this but caps at K modes and
mode-collapses under MCL. This head generalizes it STRUCTURALLY:

  (A) DEPTH-D BINARY ROUTING TREE. At each internal node a bilinear gate emits a
      scalar routing score; its SIGN (out-of-graph) picks a child. A root->leaf
      path of D binary decisions selects one of 2^D leaves, each a bilinear
      unimodal expert. 2^D modes; O(D) core-evaluations along any path.
      Coarse-to-fine (hard-EM) assignment keeps every leaf populated -> beats the
      flat-K starvation.

  (B) PRODUCT / TENSOR-FACTORED ROUTING. G independent bilinear gates emit G bits
      b_1..b_G; the action is the TENSOR-PRODUCT (additive-factored) expert

          a(h; b) = c0(h) + sum_g (2 b_g - 1) * c_g(h)         (Eq. P)

      -> 2^G modes from O(G) experts (LINEAR params, exponential modes). This is
      literally a CP/rank structure over the action manifold: which-object x
      which-grasp x which-side factorizes, so the arm's combinatorial modes map
      onto independent routing factors. Per-factor assignment is its own balanced
      binary problem -> collapse-free by construction.

Run:  python3 xvla/train/tree_moe_proto.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# =========================================================================== #
# CP (bilinear) core -- the repo's degree-2 primitive (xvla/nn/bilinear.py).
#   y = D[ (L h_bar) (.) (R h_bar) ],   h_bar = [1; h]   (homogeneous coord)
# Folds to a dense tensor T[o,i,j] = sum_r D[o,r] L[r,i] R[r,j]  (dense_core),
# i.e. y_o = sum_ij T[o,i,j] h_bar_i h_bar_j -- exactly what odt.py consumes.
# =========================================================================== #
class CP:
    def __init__(self, d_in, d_out, rank=8, spread=0.0, scale=0.4):
        self.d_in, self.d_out, self.r = d_in, d_out, rank
        s = (d_in + 1) ** -0.5
        self.L = rng.standard_normal((rank, d_in + 1)) * s
        self.R = rng.standard_normal((rank, d_in + 1)) * s
        self.D = rng.standard_normal((d_out, rank)) * (rank ** -0.5) * scale
        # spread the homogeneous (bias) channel so sibling experts start apart
        # (breaks the assignment symmetry that causes mode collapse; stays
        # foldable -- the bias is part of h_bar, captured by dense_core).
        self.L[:, 0] += spread
        self._adam = {k: [np.zeros_like(getattr(self, k)), np.zeros_like(getattr(self, k))]
                      for k in ("L", "R", "D")}
        self._t = 0

    def _bar(self, h):
        return np.concatenate([np.ones((h.shape[0], 1)), h], 1)

    def forward(self, h):
        z = self._bar(h)                       # (B, d+1)
        u = z @ self.L.T                        # (B, r)
        v = z @ self.R.T                        # (B, r)
        p = u * v                               # (B, r)
        y = p @ self.D.T                        # (B, out)
        self._cache = (z, u, v, p)
        return y

    def backward(self, dy):
        z, u, v, p = self._cache
        gD = dy.T @ p                           # (out, r)
        dp = dy @ self.D                        # (B, r)
        du, dv = dp * v, dp * u
        gL = du.T @ z                           # (r, d+1)
        gR = dv.T @ z
        return {"L": gL, "R": gR, "D": gD}

    def step(self, grads, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self._t += 1
        for k, g in grads.items():
            w = getattr(self, k)
            if wd:
                g = g + wd * w
            m, v = self._adam[k]
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** self._t)
            vh = v / (1 - b2 ** self._t)
            w -= lr * mh / (np.sqrt(vh) + eps)

    def dense_core(self):
        """Fold to the symmetric dense tensor T (out, d+1, d+1) -- ODT-ready."""
        T = np.einsum("or,ri,rj->oij", self.D, self.L, self.R)
        return 0.5 * (T + T.transpose(0, 2, 1))

    def n_params(self):
        return self.L.size + self.R.size + self.D.size


# =========================================================================== #
# Data: a conditional, genuinely-multimodal target on the hypercube {+-1}^D.
#   G latent bits, P(bit_g=1 | obs)=clamp(sigmoid(u_g.obs)) in [0.15,0.85] so
#   residual multimodality remains even given obs (an MSE head must average ->
#   commits to nothing). MAP mode = the per-factor majority bit given obs.
# For d_a==D the target sits on hypercube corners (PRODUCT-structured). Pass a
# `codebook` to place the 2^D modes at ARBITRARY points (NON-product) instead.
# =========================================================================== #
def make_data(n, D, U, d_a=None, codebook=None, corner=1.0, noise=0.08):
    d_a = D if d_a is None else d_a
    obs = rng.standard_normal((n, U.shape[1]))
    p = 0.15 + 0.70 * sigmoid(obs @ U.T)          # (n, D) clamped bit-probs
    bits = (rng.random((n, D)) < p).astype(float)  # sampled latent mode
    if codebook is None:
        mean = (2 * bits - 1) * corner             # corners of {+-1}^D
    else:
        idx = (bits * (2 ** np.arange(D))).sum(1).astype(int)   # bit-tuple -> row
        mean = codebook[idx]
    a = mean + noise * rng.standard_normal((n, d_a))
    map_bits = (p > 0.5).astype(float)             # the MAP mode given obs
    return obs, a, bits, map_bits, p


def true_modes(D, codebook=None, corner=1.0):
    B = ((np.arange(2 ** D)[:, None] >> np.arange(D)) & 1).astype(float)
    return (2 * B - 1) * corner if codebook is None else codebook


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def coverage(dec, modes, tau=0.4):
    """How many of the true modes are hit by some decoded action (<tau)."""
    d = np.linalg.norm(dec[:, None, :] - modes[None, :, :], axis=-1)  # (N, M)
    return int((d.min(0) < tau).sum())


def commit_acc(dec, map_bits, D, codebook=None, corner=1.0, tau=0.4):
    """Fraction of samples whose decode lands on the correct (MAP) mode."""
    idx = (map_bits * (2 ** np.arange(D))).sum(1).astype(int)
    modes = true_modes(D, codebook, corner)
    tgt = modes[idx]
    return float((np.linalg.norm(dec - tgt, axis=-1) < tau).mean())


# =========================================================================== #
# Baseline 0 -- linear + MSE head (the current chi-VLA head). Emits the mean.
# =========================================================================== #
def train_mse(obs, a, d_a):
    Xb = np.concatenate([np.ones((obs.shape[0], 1)), obs], 1)
    W = np.linalg.lstsq(Xb, a, rcond=None)[0]
    return lambda o: np.concatenate([np.ones((o.shape[0], 1)), o], 1) @ W


# =========================================================================== #
# Baseline 1 -- FLAT K-candidate MCL head (the sibling, multihypothesis.py).
# K CP experts + K CP score gates; eps-relaxed MCL; decode = argmax score.
# =========================================================================== #
def train_flat_mcl(obs, a, K, d_a, iters=4000, eps=0.05, lr=3e-3):
    d = obs.shape[1]
    experts = [CP(d, d_a, spread=0.5 * (k - (K - 1) / 2)) for k in range(K)]
    scores = [CP(d, 1) for _ in range(K)]
    for _ in range(iters):
        idx = rng.integers(0, obs.shape[0], 512)
        h, y = obs[idx], a[idx]
        preds = np.stack([e.forward(h) for e in experts], 1)        # (B,K,d_a)
        errs = ((preds - y[:, None, :]) ** 2).mean(-1)              # (B,K)
        kstar = errs.argmin(1)
        w = np.full_like(errs, eps / max(K - 1, 1))
        w[np.arange(len(h)), kstar] = 1 - eps                      # eps-relaxed
        for k, e in enumerate(experts):                            # expert grads
            dy = (2.0 / d_a) * (preds[:, k] - y) * w[:, k:k + 1] / len(h)
            e.step(e.backward(dy), lr=lr)
        sc = np.concatenate([s.forward(h) for s in scores], 1)     # (B,K)
        tgt = -errs                                                # rank by -MSE
        for k, s in enumerate(scores):                            # score grads
            dy = (2.0 * (sc[:, k:k + 1] - tgt[:, k:k + 1])) / len(h)
            s.step(s.backward(dy), lr=lr)

    def decode(o):
        preds = np.stack([e.forward(o) for e in experts], 1)
        sc = np.concatenate([s.forward(o) for s in scores], 1)
        k = sc.argmax(1)                                           # OUT-OF-GRAPH
        return preds[np.arange(len(o)), k]
    n_par = sum(e.n_params() for e in experts) + sum(s.n_params() for s in scores)
    return decode, n_par


# =========================================================================== #
# Head A -- DEPTH-2 BINARY ROUTING TREE (3 gates + 4 leaf experts).
# Coarse-to-fine hard-EM assignment; gates trained by (loss-only) logistic on
# the assigned bit; balanced root split keeps both subtrees populated.
# Decode: b1 = 1[root>0]; b2 = 1[child_{b1}>0]  (both OUT-OF-GRAPH) -> leaf.
# =========================================================================== #
def train_tree_d2(obs, a, d_a, iters=4000, eps=0.05, lr=3e-3, balance=0.3):
    d = obs.shape[1]
    root = CP(d, 1, spread=0.0)
    child = [CP(d, 1, spread=0.0) for _ in range(2)]
    leaf = [CP(d, d_a, spread=0.6 * (k - 1.5)) for k in range(4)]  # 00,01,10,11
    for _ in range(iters):
        idx = rng.integers(0, obs.shape[0], 512)
        h, y = obs[idx], a[idx]
        Lp = np.stack([lf.forward(h) for lf in leaf], 1)          # (B,4,d_a)
        errs = ((Lp - y[:, None, :]) ** 2).mean(-1)               # (B,4)
        # coarse-to-fine assignment (E-step, loss-only, uses target):
        sub = errs.reshape(len(h), 2, 2)                          # [b1, b2]
        b1 = sub.min(2).argmin(1)                                 # coarse split
        b2 = sub[np.arange(len(h)), b1].argmin(1)                 # refine
        leaf_star = b1 * 2 + b2
        # leaf regression (eps-relaxed so idle leaves keep a trickle of grad):
        w = np.full((len(h), 4), eps / 3.0)
        w[np.arange(len(h)), leaf_star] = 1 - eps
        for k, lf in enumerate(leaf):
            dy = (2.0 / d_a) * (Lp[:, k] - y) * w[:, k:k + 1] / len(h)
            lf.step(lf.backward(dy), lr=lr)
        # gate logistic (M-step): softmax/BCE in the LOSS only; sign is deployed.
        rs = root.forward(h)[:, 0]                                # root score
        pr = sigmoid(rs)
        gr = (pr - b1)[:, None] / len(h)                         # d BCE / d score
        if balance:                                              # keep split ~50/50
            gr = gr + balance * (pr.mean() - 0.5) * np.ones_like(gr) / len(h)
        root.step(root.backward(gr), lr=lr)
        for c in (0, 1):                                          # train child on its samples
            m = b1 == c
            if m.any():
                cs = child[c].forward(h)[:, 0]
                gc = np.zeros((len(h), 1))
                gc[m, 0] = (sigmoid(cs[m]) - b2[m]) / m.sum()
                child[c].step(child[c].backward(gc), lr=lr)

    def decode(o):
        b1 = (root.forward(o)[:, 0] > 0).astype(int)
        cs = np.where(b1 == 0, child[0].forward(o)[:, 0], child[1].forward(o)[:, 0])
        b2 = (cs > 0).astype(int)
        star = b1 * 2 + b2
        Lp = np.stack([lf.forward(o) for lf in leaf], 1)
        return Lp[np.arange(len(o)), star]                        # OUT-OF-GRAPH path
    n_par = (root.n_params() + sum(c.n_params() for c in child)
             + sum(lf.n_params() for lf in leaf))
    return decode, n_par


# =========================================================================== #
# Head B -- PRODUCT / TENSOR-FACTORED ROUTING (G gates + (G+1) factor experts).
# a(h;b) = c0(h) + sum_g (2 b_g - 1) c_g(h)   -- 2^G modes from O(G) experts.
# Assignment enumerates 2^G combos (additivity also allows O(G) coord-descent
# for large G); each factor's bit is its own balanced binary problem -> no
# starvation. Decode: b_g = 1[gate_g>0] (OUT-OF-GRAPH), then Eq. P.
# =========================================================================== #
def train_product(obs, a, D, d_a, iters=4000, lr=3e-3):
    d = obs.shape[1]
    c = [CP(d, d_a, spread=0.0)] + [CP(d, d_a, spread=0.0) for _ in range(D)]
    gates = [CP(d, 1) for _ in range(D)]
    combos = ((np.arange(2 ** D)[:, None] >> np.arange(D)) & 1)   # (2^D, D) bits
    signs = (2 * combos - 1).astype(float)                        # (2^D, D)
    for _ in range(iters):
        idx = rng.integers(0, obs.shape[0], 512)
        h, y = obs[idx], a[idx]
        c0 = c[0].forward(h)                                      # (B,d_a)
        cg = np.stack([c[g + 1].forward(h) for g in range(D)], 1)  # (B,D,d_a)
        # a for every combo: (B, 2^D, d_a)
        alla = c0[:, None, :] + np.einsum("md,bdc->bmc", signs, cg)
        errs = ((alla - y[:, None, :]) ** 2).mean(-1)            # (B, 2^D)
        mstar = errs.argmin(1)                                    # best combo
        bstar = combos[mstar]                                     # (B, D) bits
        sstar = signs[mstar]                                      # (B, D) signs
        pred = c0 + np.einsum("bd,bdc->bc", sstar, cg)
        resid = (2.0 / d_a) * (pred - y) / len(h)                # (B,d_a)
        c[0].step(c[0].backward(resid), lr=lr)                    # d/dc0
        for g in range(D):                                        # d/dc_g = s_g * resid
            c[g + 1].step(c[g + 1].backward(sstar[:, g:g + 1] * resid), lr=lr)
        for g in range(D):                                        # gate BCE (loss-only)
            gs = gates[g].forward(h)[:, 0]
            gg = (sigmoid(gs) - bstar[:, g])[:, None] / len(h)
            gates[g].step(gates[g].backward(gg), lr=lr)

    def decode(o):
        b = np.stack([(gates[g].forward(o)[:, 0] > 0) for g in range(D)], 1)
        s = (2 * b - 1).astype(float)                            # OUT-OF-GRAPH
        c0 = c[0].forward(o)
        cg = np.stack([c[g + 1].forward(o) for g in range(D)], 1)
        return c0 + np.einsum("bd,bdc->bc", s, cg)
    n_par = sum(x.n_params() for x in c) + sum(g.n_params() for g in gates)
    return decode, n_par


# --------------------------------------------------------------------------- #
def _eval(name, decode, npar, obs, map_bits, D, modes, codebook=None):
    dec = decode(obs)
    cov = coverage(dec, modes)
    acc = commit_acc(dec, map_bits, D, codebook)
    print(f"    {name:22s} params={npar:5d}  coverage={cov:2d}/{len(modes)}  "
          f"commit_MAP={acc:.3f}  |dec|_mean={np.abs(dec).mean():.2f}")
    return cov, acc


def part1_minimal():
    print("Part 1: MINIMAL 4-mode 2-D target -- depth-2 tree & 2x2 product vs flat K=4")
    D, d, d_a = 2, 6, 2
    U = rng.standard_normal((D, d))
    obs, a, _, _, _ = make_data(20000, D, U)
    te_obs, _, _, te_map, _ = make_data(4000, D, U)
    modes = true_modes(D)

    mse = train_mse(obs, a, d_a)
    _eval("MSE (linear head)", mse, obs.shape[1] * d_a + d_a, te_obs, te_map, D, modes)
    dec, npar = train_flat_mcl(obs, a, 4, d_a)
    _eval("flat K=4 MCL (sibling)", dec, npar, te_obs, te_map, D, modes)
    dec, npar = train_tree_d2(obs, a, d_a)
    _eval("depth-2 tree (Head A)", dec, npar, te_obs, te_map, D, modes)
    dec, npar = train_product(obs, a, D, d_a)
    _eval("2x2 product (Head B)", dec, npar, te_obs, te_map, D, modes)


def part2_scaling():
    print("\nPart 2: SCALING -- 2^D modes. Product routing keeps O(D) params &")
    print("        collapse-free coverage where flat K=2^D starves.")
    d = 6
    for D in (2, 3, 4):
        U = rng.standard_normal((D, d))
        obs, a, _, _, _ = make_data(30000, D, U)
        te_obs, _, _, te_map, _ = make_data(6000, D, U)
        modes = true_modes(D)
        print(f"  D={D}  ({2**D} modes):")
        dec, npar = train_flat_mcl(obs, a, 2 ** D, D, iters=5000)
        _eval(f"flat K={2**D} MCL", dec, npar, te_obs, te_map, D, modes)
        dec, npar = train_product(obs, a, D, D, iters=5000)
        _eval("product routing", dec, npar, te_obs, te_map, D, modes)


def part3_nonproduct():
    print("\nPart 3: HONEST LIMIT -- 4 modes at ARBITRARY (non-product) points.")
    print("        Product/additive routing can only place a parallelogram; the")
    print("        tree (independent leaf experts) places all 4 freely.")
    D, d, d_a = 2, 6, 2
    codebook = np.array([[0.0, 2.0], [2.0, -0.2], [-1.8, -1.6], [1.2, 1.4]])
    U = rng.standard_normal((D, d))
    obs, a, _, _, _ = make_data(20000, D, U, d_a=d_a, codebook=codebook)
    te_obs, _, _, te_map, _ = make_data(4000, D, U, d_a=d_a, codebook=codebook)
    dec, npar = train_product(obs, a, D, d_a)
    _eval("2x2 product (Head B)", dec, npar, te_obs, te_map, D, codebook, codebook)
    dec, npar = train_tree_d2(obs, a, d_a)
    _eval("depth-2 tree (Head A)", dec, npar, te_obs, te_map, D, codebook, codebook)


def part4_fold_check():
    print("\nPart 4: tensor-purity check -- each gate/expert folds to a dense")
    print("        degree-2 core (odt.py-ready); CP forward == dense-core forward.")
    d, d_a = 6, 2
    e = CP(d, d_a, rank=8)
    h = rng.standard_normal((5, d))
    y_cp = e.forward(h)
    hb = np.concatenate([np.ones((5, 1)), h], 1)
    T = e.dense_core()
    y_dense = np.einsum("oij,bi,bj->bo", T, hb, hb)
    print(f"    max|CP - dense_core| = {np.abs(y_cp - y_dense).max():.2e}  "
          f"(exact fold; symmetric core is the ODT object)")


if __name__ == "__main__":
    part1_minimal()
    part2_scaling()
    part3_nonproduct()
    part4_fold_check()
