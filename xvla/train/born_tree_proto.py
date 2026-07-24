"""Minimal prototype: a 2-D-topology TREE tensor-network Born-machine action head.

Sibling to `born_mps_proto.py` (a 1-D MPS/tensor-train Born machine over a
*flattened* H*d_a order). The action chunk is not a 1-D string: it is a 2-D grid

        TIME  t = 0 .. H-1      (rows)
        DIM   d = 0 .. D-1      (cols, the d_a action coordinates)

and its multimodality is *spatiotemporal*: a whole reach-left vs reach-right
trajectory couples cells along BOTH axes. Imposing an arbitrary 1-D order on that
grid (what an MPS does) can turn a short-range 2-D correlation into a long-range
1-D one, forcing a large bond dimension.

This prototype builds an obs-free** Born machine  p(a) = |psi(a)|^2 / Z  whose
network TOPOLOGY matches the grid:

  * TREE tensor network (TTN) — a balanced binary tree whose leaf grouping is the
    2-D grid's column-major order (each action-dim's whole timeline is one
    contiguous subtree). A tree has NO loops, so EVERYTHING is exactly
    contractible in O(N * chi^k): Z = <psi|psi>, marginals, entanglement spectra,
    and exact ancestral sampling. This is the honest sweet spot the report argues
    for: PEPS matches the 2-D grid even better but its exact contraction is
    #P-hard; a tree stays exact.

We compare, at MATCHED bond dimension, three networks on a spatiotemporally
correlated target (each action-dim is a persistent "reach-left/right" trajectory):

  * TREE (grid-aware)               -> fits at chi = 2
  * MPS, column-major flatten       -> the *lucky* 1-D order, also fits at chi = 2
  * MPS, row-major flatten (t*D+d)  -> the *arbitrary* order the sibling warns
                                       about: interleaves the D timelines, so every
                                       time-cut straddles all D correlations at
                                       once -> needs chi up to 2^D.

The point is NOT "tree beats every MPS" (a *matched* 1-D order ties it). It is:
the grid-aware topology removes the ordering gamble — a naive flatten is
catastrophic, and there is in general no single 1-D order good for correlations
that run along BOTH axes (see the "both-axes" target). The report is honest that
for genuine 2-D area-law you want PEPS (not exactly contractible); a tree is the
exactly-contractible choice when the dominant structure is temporal-within-dim
(exactly the action-chunk case: each joint moves smoothly in time).

** obs-conditioning is the orthogonal, already-demonstrated piece: make every core
   AFFINE in a context feature phi=[1;h(obs)] (degree-1 = foldable/tensor-pure),
   exactly as `born_mps_proto.CondMPS` and `xvla/nn/bilinear.BilinearFFN` do. Here
   we hold obs fixed to isolate the topology/bond-dimension question.

Run:  python3 xvla/train/born_tree_proto.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


# ============================================================================
# The spatiotemporally-correlated target (a binary H x D action-chunk grid).
# 1 bit / cell = the SIGN of the action delta at (time t, dim d): the coarsest
# encoding of "this joint moves +/- at this step". Finer binning = more bits/cell
# (physical dim 2^b per leaf), identical machinery.
# ============================================================================
def sample_target(n, H, D, kind="temporal", persist=0.85):
    """Return bits of shape (n, H, D).

    kind="temporal": each dim d is an INDEPENDENT persistent chain in TIME
        (b[t]=b[t-1] w.p. `persist`, else flip; b[0]~Bernoulli(1/2)). So each
        column is mostly all-0 ("reach left") or all-1 ("reach right") with rare
        switches -> a per-dim bimodal WHOLE-TRAJECTORY mode, correlated along time,
        independent across dims. This is the dominant structure of real chunks.
    kind="both": additionally a global mode M~Bernoulli(1/2) biases dims 0 and 1
        to agree (a cross-DIM correlation on top of the temporal one), so
        correlations run along BOTH axes -> no single 1-D order is good.
    """
    g = np.zeros((n, H, D), dtype=int)
    for d in range(D):
        g[:, 0, d] = rng.random(n) < 0.5
        for t in range(1, H):
            flip = rng.random(n) > persist
            g[:, t, d] = np.where(flip, 1 - g[:, t - 1, d], g[:, t - 1, d])
    if kind == "both":
        M = (rng.random(n) < 0.5).astype(int)                 # global reach L/R
        for d in (0, 1):
            # re-bias this dim's whole trajectory toward the global mode
            bias = np.where(rng.random((n, H)) < 0.80, M[:, None], 1 - M[:, None])
            g[:, :, d] = bias
    return g


def to_order(bits_HD, order):
    """Flatten (n,H,D) grid bits into (n,N) leaf order.

    order="col": leaf index = d*H + t (each dim's timeline contiguous; the tree
                 and the "lucky" MPS use this).
    order="row": leaf index = t*D + d (interleaves the D timelines; the naive MPS).
    """
    n, H, D = bits_HD.shape
    if order == "col":
        return np.transpose(bits_HD, (0, 2, 1)).reshape(n, H * D)
    return bits_HD.reshape(n, H * D)                            # row-major


def exact_pmf_temporal(H, D, persist=0.85):
    """Exact p over all 2^(H*D) grids for the 'temporal' target, as a tensor of
    shape (2,)*N with axis (d*H + t) = the canonical (col-major) site index.

    p factorizes across dims; each dim is a 4-bit persistence Markov chain:
      p_d(b_0..b_{H-1}) = 1/2 * prod_t [persist if b_t==b_{t-1} else 1-persist].
    """
    N = H * D
    idx = np.arange(2 ** N)
    bits = ((idx[:, None] >> np.arange(N - 1, -1, -1)) & 1)      # (2^N, N) big-endian
    logp = np.zeros(2 ** N)
    for d in range(D):
        col = bits[:, d * H:(d + 1) * H]                         # sites d*H..d*H+H-1
        lp = np.full(2 ** N, np.log(0.5))
        for t in range(1, H):
            same = col[:, t] == col[:, t - 1]
            lp += np.where(same, np.log(persist), np.log(1 - persist))
        logp += lp
    p = np.exp(logp)
    p = p / p.sum()
    # tensor with axis a = canonical site a (bit a is the a-th big-endian bit)
    return p.reshape([2] * N)


def exact_entropy(p_tensor):
    p = p_tensor.ravel()
    p = p[p > 0]
    return -(p * np.log(p)).sum()


def schmidt_across(psi_tensor, site_order, cut):
    """Singular-value spectrum of psi matricized across a bipartition.

    `psi_tensor` has axis a = canonical site a. `site_order` is a permutation of
    site indices defining a 1-D layout; `cut` splits the first `cut` laid-out
    sites from the rest. Returns normalized singular values (sum of squares = 1).
    """
    N = psi_tensor.ndim
    M = np.transpose(psi_tensor, site_order).reshape(2 ** cut, 2 ** (N - cut))
    s = np.linalg.svd(M, compute_uv=False)
    return s / np.sqrt((s ** 2).sum())


def required_bond(psi_tensor, site_order, tol=1e-9):
    """Max Schmidt rank over all cuts of a 1-D layout = the MPS bond dim needed."""
    N = psi_tensor.ndim
    best_rank, worst = 1, None
    for cut in range(1, N):
        s = schmidt_across(psi_tensor, site_order, cut)
        r = int((s > tol * s[0]).sum())
        if r >= best_rank:
            best_rank, worst = r, (cut, s)
    return best_rank, worst


# ============================================================================
# TREE tensor-network Born machine (exactly contractible).
# ============================================================================
class _Node:
    __slots__ = ("is_leaf", "site", "out", "L", "R", "id")


class TreeBorn:
    """Balanced binary TTN amplitude over N leaves (physical dim 2).

    Internal-node tensor T[node] has shape (out, Lout, Rout): it contracts the two
    child bonds into a parent bond. Leaf out=2 (physical), internal out=chi, root
    out=1 (closes psi to a scalar). No loops => Z, gradients, entanglement spectra
    and sampling are all EXACT tree contractions.
    """

    def __init__(self, N, chi=2, scale=0.4):
        assert (N & (N - 1)) == 0, "this minimal build uses N = power of 2"
        self.N, self.chi = N, chi
        self.nodes = []
        leaves = []
        for i in range(N):
            nd = _Node(); nd.is_leaf = True; nd.site = i; nd.out = 2
            nd.id = len(self.nodes); self.nodes.append(nd); leaves.append(nd)
        level = leaves
        while len(level) > 1:
            nxt = []
            for i in range(0, len(level), 2):
                nd = _Node(); nd.is_leaf = False
                nd.L = level[i]; nd.R = level[i + 1]
                nd.out = chi
                nd.id = len(self.nodes); self.nodes.append(nd); nxt.append(nd)
            level = nxt
        self.root = level[0]
        self.root.out = 1                                       # scalar amplitude
        self.internal = [nd for nd in self.nodes if not nd.is_leaf]
        self.T = {}
        for nd in self.internal:
            self.T[nd.id] = scale * rng.standard_normal((nd.out, nd.L.out, nd.R.out))

    # -- upward amplitude messages for a batch of configs -------------------
    def _messages(self, bits):
        m = {}
        for nd in self.nodes:
            if nd.is_leaf:
                s = bits[:, nd.site]
                v = np.zeros((bits.shape[0], 2)); v[np.arange(len(s)), s] = 1.0
                m[nd.id] = v
            else:
                T = self.T[nd.id]
                m[nd.id] = np.einsum("plr,bl,br->bp", T, m[nd.L.id], m[nd.R.id])
        return m

    def amp(self, bits, m=None):
        m = m or self._messages(bits)
        return m[self.root.id][:, 0]

    # -- upward DOUBLED (rho) pass: rho[node] (out,out); Z = rho[root] ------
    def _rho(self):
        rho = {}
        for nd in self.nodes:
            if nd.is_leaf:
                rho[nd.id] = np.eye(2)                          # sum_s e_s e_s^T
            else:
                T = self.T[nd.id]
                rho[nd.id] = np.einsum("plr,PLR,lL,rR->pP", T, T,
                                       rho[nd.L.id], rho[nd.R.id])
        return rho

    def Z(self, rho=None):
        rho = rho or self._rho()
        return rho[self.root.id][0, 0], rho

    # -- downward DOUBLED environment Ed[node] (out,out) -------------------
    def _ed(self, rho):
        Ed = {self.root.id: np.ones((1, 1))}
        for nd in reversed(self.internal):                     # root -> leaves
            T = self.T[nd.id]; E = Ed[nd.id]
            # child L env: contract parent (2 copies) with parent env + sibling rho
            Ed[nd.L.id] = np.einsum("pP,plr,PLR,rR->lL", E, T, T, rho[nd.R.id])
            Ed[nd.R.id] = np.einsum("pP,plr,PLR,lL->rR", E, T, T, rho[nd.L.id])
        return Ed

    # -- downward amplitude co-vectors D[node] (B,out) for a batch ---------
    def _down(self, bits, m):
        B = bits.shape[0]
        D = {self.root.id: np.ones((B, 1))}
        for nd in reversed(self.internal):
            T = self.T[nd.id]; Dn = D[nd.id]
            D[nd.L.id] = np.einsum("bp,plr,br->bl", Dn, T, m[nd.R.id])
            D[nd.R.id] = np.einsum("bp,plr,bl->br", Dn, T, m[nd.L.id])
        return D

    # -- exact NLL + analytic Born gradient over a batch ------------------
    def nll_and_grad(self, bits):
        B = bits.shape[0]
        m = self._messages(bits)
        psi = self.amp(bits, m)
        Z, rho = self.Z()
        Ed = self._ed(rho)
        Dn = self._down(bits, m)
        nll = -np.mean(2 * np.log(np.abs(psi) + 1e-12)) + np.log(Z)
        g = {}
        for nd in self.internal:
            mL, mR = m[nd.L.id], m[nd.R.id]
            # data term: -(2/B) sum_b (1/psi) D[node] outer mL outer mR
            gdat = -(2.0 / B) * np.einsum("bp,bl,br->plr", Dn[nd.id] / psi[:, None], mL, mR)
            # logZ term: (1/Z) dZ/dT = (2/Z) Ed[node] . T . rhoL . rhoR
            gZ = (2.0 / Z) * np.einsum("pP,PLR,lL,rR->plr", Ed[nd.id], self.T[nd.id],
                                       rho[nd.L.id], rho[nd.R.id])
            g[nd.id] = gdat + gZ
        return nll, g

    def step(self, bits, lr):
        nll, g = self.nll_and_grad(bits)
        for nd in self.internal:
            self.T[nd.id] -= lr * g[nd.id]
        return nll

    # -- exact enumeration pmf (small N) ----------------------------------
    def enumerate_pmf(self):
        N = self.N
        idx = np.arange(2 ** N)
        bits = ((idx[:, None] >> np.arange(N - 1, -1, -1)) & 1)
        psi = self.amp(bits)
        Z, _ = self.Z()
        return bits, psi ** 2 / Z

    def sample(self, n):
        """Exact sampling. For this prototype size we sample from the enumerated
        pmf (itself exact). For large N the identical result comes from O(N)
        top-down ancestral sampling using the Ed environments (the tree analog of
        the MPS right-environment sweep) -- no enumeration needed."""
        bits, p = self.enumerate_pmf()
        sel = rng.choice(len(bits), size=n, p=p)
        return bits[sel]

    # -- entanglement (bond) spectrum across an edge (the ODT readout) -----
    def edge_spectrum(self, node):
        """Squared Schmidt spectrum across the edge ABOVE `node` (subtree `node`
        vs the rest). Gauge-invariant eigenvalues of rho_below @ Ed_above,
        normalized to sum 1. Entropy S = -sum lambda log lambda."""
        _, rho = self.Z()
        Ed = self._ed(rho)
        M = rho[node.id] @ Ed[node.id]
        ev = np.real(np.linalg.eigvals(M))
        ev = np.clip(ev, 0, None)
        ev = ev / ev.sum()
        S = -(ev[ev > 1e-12] * np.log(ev[ev > 1e-12])).sum()
        return np.sort(ev)[::-1], S

    def dim_subtree_root(self, d, H):
        """The internal node whose subtree is exactly action-dim d's whole
        timeline (leaves d*H .. d*H+H-1), in the col-major tree."""
        target = set(range(d * H, d * H + H))
        for nd in self.internal:
            leaves = set()
            stack = [nd]
            while stack:
                x = stack.pop()
                if x.is_leaf:
                    leaves.add(x.site)
                else:
                    stack += [x.L, x.R]
            if leaves == target:
                return nd
        return None


# ============================================================================
# 1-D MPS Born machine (unconditional) -- for the matched-bond comparison.
# Open-boundary tensor train; same |psi|^2/Z with exact transfer-matrix Z and the
# analytic Born gradient (Han et al. 2018), adapted from born_mps_proto.CondMPS.
# ============================================================================
class MPSBorn:
    def __init__(self, N, chi=2, scale=0.4):
        self.N = N
        rb = [1] + [chi] * (N - 1) + [1]
        self.rb = rb
        self.A = [scale * rng.standard_normal((rb[j], 2, rb[j + 1])) for j in range(N)]

    def amp(self, bits):
        v = np.ones((bits.shape[0], 1))
        for j in range(self.N):
            Aj = self.A[j][:, bits[:, j], :]                   # (rL,B,rR)
            v = np.einsum("br,rbc->bc", v, Aj)
        return v[:, 0]

    def _transfer(self):
        M = [np.einsum("rsb,RsB->rRbB", A, A).reshape(A.shape[0] ** 2, A.shape[2] ** 2)
             for A in self.A]
        z = np.ones(1)
        for Mj in M:
            z = z @ Mj
        return z[0], M

    def nll_and_grad(self, bits):
        B = bits.shape[0]; N = self.N
        Z, M = self._transfer()
        PL = [np.ones((B, 1))]
        for j in range(N):
            Aj = self.A[j][:, bits[:, j], :]
            PL.append(np.einsum("br,rbc->bc", PL[j], Aj))
        psi = PL[N][:, 0]
        PR = [None] * (N + 1); PR[N] = np.ones((B, 1))
        for j in range(N - 1, -1, -1):
            Aj = self.A[j][:, bits[:, j], :]
            PR[j] = np.einsum("rbc,bc->br", Aj, PR[j + 1])
        nll = -np.mean(2 * np.log(np.abs(psi) + 1e-12)) + np.log(Z)
        ZL = [np.ones(1)]
        for j in range(N):
            ZL.append(ZL[j] @ M[j])
        ZR = [None] * (N + 1); ZR[N] = np.ones(1)
        for j in range(N - 1, -1, -1):
            ZR[j] = M[j] @ ZR[j + 1]
        g = [np.zeros_like(self.A[j]) for j in range(N)]
        for j in range(N):
            rL, _, rR = self.A[j].shape
            gA = np.zeros((rL, 2, rR))
            for s in (0, 1):
                mk = bits[:, j] == s
                if mk.any():
                    gA[:, s, :] += -(2.0 / B) * np.einsum(
                        "br,bc,b->rc", PL[j][mk], PR[j + 1][mk], 1.0 / psi[mk])
            LL = ZL[j].reshape(rL, rL); RR = ZR[j + 1].reshape(rR, rR)
            for s in (0, 1):
                gA[:, s, :] += (2.0 / Z) * (LL @ self.A[j][:, s, :] @ RR)
            g[j] = gA
        return nll, g

    def step(self, bits, lr):
        nll, g = self.nll_and_grad(bits)
        for j in range(self.N):
            self.A[j] -= lr * g[j]
        return nll


# ============================================================================
# Driver
# ============================================================================
def held_out_nll_tree(model, bits):
    psi = model.amp(bits); Z, _ = model.Z()
    return (-np.mean(2 * np.log(np.abs(psi) + 1e-12)) + np.log(Z))


def held_out_nll_mps(model, bits):
    psi = model.amp(bits); Z, _ = model._transfer()
    return (-np.mean(2 * np.log(np.abs(psi) + 1e-12)) + np.log(Z))


def train(model, H, D, kind, steps, lr, bs=1024, is_tree=True, order="col"):
    for it in range(steps):
        g = sample_target(bs, H, D, kind)
        b = to_order(g, order)
        model.step(b, lr)
    ho = sample_target(4096, H, D, kind)
    b = to_order(ho, order)
    return held_out_nll_tree(model, b) if is_tree else held_out_nll_mps(model, b)


def part1_bonddim():
    H, D = 4, 4                                                # 16-cell chunk grid
    N = H * D
    print(f"Chunk grid: H={H} time x D={D} action-dims -> N={N} sites (1 bit/cell)")
    print(f"Target 'temporal': each action-dim is an independent persistent (reach-"
          f"left/right)\ntrajectory in TIME (correlations run along the time axis, "
          f"within each dim).\n")

    # ---- EXACT (optimization-free) required bond dimension -----------------
    # sqrt(p_true) is the ideal Born amplitude. The MPS bond a 1-D order needs =
    # the max Schmidt rank of sqrt(p) over that order's cuts. This is a property
    # of the TARGET + the LAYOUT, independent of any training.
    p = exact_pmf_temporal(H, D)
    psi = np.sqrt(p)
    Sfloor = exact_entropy(p)
    print(f"Entropy floor = {Sfloor:.3f} nats (exact; held-out NLL cannot go below).")
    print("\n  Exact required MPS bond (max Schmidt rank of sqrt(p) over the layout"
          "'s cuts):")
    col_order = [d * H + t for d in range(D) for t in range(H)]     # canonical
    row_order = [d * H + t for t in range(H) for d in range(D)]     # interleaved
    for name, order in (("col-major (each dim's timeline contiguous)", col_order),
                        ("row-major (naive flatten t*D+d)", row_order)):
        r, (cut, s) = required_bond(psi, order)
        print(f"    {name:44s} -> chi >= {r:2d}   (worst cut spectrum "
              f"{np.round(s[:6], 3)})")
    print(f"    tree (grid-aware): every edge is a within-dim OR cross-dim cut -> "
          f"chi = 2")
    print(f"\n  => the naive flatten needs chi ~ 2^D = {2**D} (every time-cut "
          f"straddles all D\n     timelines at once); the grid-aware layout / tree "
          f"needs only chi = 2.")

    # ---- training corroboration (fits at chi=2 for the good topologies) ----
    print("\n  Training corroboration at MATCHED chi = 2 (held-out NLL):")
    tree = TreeBorn(N, chi=2)
    nt = train(tree, H, D, "temporal", steps=1500, lr=0.1, is_tree=True, order="col")
    mc = MPSBorn(N, chi=2)
    nc = train(mc, H, D, "temporal", steps=1500, lr=0.1, is_tree=False, order="col")
    mr = MPSBorn(N, chi=2)
    nr = train(mr, H, D, "temporal", steps=1500, lr=0.1, is_tree=False, order="row")
    print(f"    TREE  (grid-aware)        chi=2   NLL = {nt:.3f}  (-> floor)")
    print(f"    MPS   col-major (matched) chi=2   NLL = {nc:.3f}  (-> floor)")
    print(f"    MPS   row-major (naive)   chi=2   NLL = {nr:.3f}  (STUCK: chi=2 << "
          f"required {2**D})")


def part2_sampling_and_spectra():
    H, D = 4, 4; N = H * D
    print("\n=== TREE: normalization, whole-trajectory sampling, ODT spectra ===")
    tree = TreeBorn(N, chi=4)
    train(tree, H, D, "both", steps=900, lr=0.05, is_tree=True, order="col")

    # (a) exact normalization: brute-force sum over all configs vs transfer Z
    bits, p = tree.enumerate_pmf()
    print(f"  sum_a p(a) over all 2^{N} configs = {p.sum():.6f}   (exact => 1)")

    # (b) whole-trajectory multimodality: does dim-0's trajectory commit L or R?
    S = tree.sample(20000)                                     # (n, N) col-major
    g = np.transpose(S.reshape(-1, D, H), (0, 2, 1))           # -> (n,H,D)
    d0_mean = g[:, :, 0].mean(1)                               # per-sample time-avg of dim0
    frac_left = np.mean(d0_mean < 0.5); frac_right = np.mean(d0_mean > 0.5)
    print(f"  dim-0 whole trajectory: frac mostly-LEFT={frac_left:.3f} "
          f"mostly-RIGHT={frac_right:.3f}  (bimodal => COMMITS, not the mean)")
    # temporal coherence within a trajectory: corr(bit_t0, bit_t3) for dim0
    c = np.corrcoef(g[:, 0, 0], g[:, 3, 0])[0, 1]
    print(f"  dim-0 temporal coherence corr(t=0, t=3) = {c:.3f}  (>0 => a mode "
          f"is a whole reach-left/right trajectory, not per-step coin flips)")
    # cross-dim coupling (the 'both' target ties dim-0 and dim-1)
    cx = np.corrcoef(g[:, 0, 0], g[:, 0, 1])[0, 1]
    print(f"  cross-dim coupling    corr(dim0, dim1) = {cx:.3f}  (>0 => dims 0,1 "
          f"share the global reach-L/R mode)")

    # (c) entanglement spectra: TIME-cut vs DIM-cut = distinct correlation readouts.
    #     time-cut: an edge INSIDE dim-0's subtree (early vs late timesteps).
    #     dim-cut : the edge ABOVE dim-0's subtree (dim-0 vs the other dims).
    d0 = tree.dim_subtree_root(0, H)                           # dim-0 whole timeline
    time_edge = d0.L                                           # early-t half of dim0
    dim_edge = d0                                              # dim-0 vs the rest
    ev_t, S_t = tree.edge_spectrum(time_edge)
    ev_d, S_d = tree.edge_spectrum(dim_edge)
    print(f"  entanglement entropy  TIME-cut (within dim-0) = {S_t:.3f} nats  "
          f"spectrum={np.round(ev_t[:4], 3)}")
    print(f"  entanglement entropy  DIM-cut  (dim-0 vs rest)= {S_d:.3f} nats  "
          f"spectrum={np.round(ev_d[:4], 3)}")
    # baseline: on 'temporal' (independent dims) the DIM-cut collapses to ~0
    tref = TreeBorn(N, chi=4)
    train(tref, H, D, "temporal", steps=900, lr=0.05, is_tree=True, order="col")
    _, S_d0 = tref.edge_spectrum(tref.dim_subtree_root(0, H))
    print(f"  (control) DIM-cut entropy on 'temporal' target = {S_d0:.3f} nats "
          f"(independent dims => ~0)")
    print("  -> the two cuts read out DIFFERENT correlations (temporal within a dim"
          " vs\n     cross-dim coupling); the DIM-cut rises only when dims couple. "
          "This 2-D\n     entanglement-spectrum readout is what a 1-D MPS over a "
          "flattened order\n     cannot expose (headline #5 ODT, extended to the "
          "chunk grid).")


if __name__ == "__main__":
    print("Part 1: matched-bond capacity -- 2-D tree topology vs 1-D MPS orderings")
    part1_bonddim()
    print("\nPart 2: exact contraction, sampling, and bond/entanglement spectra")
    part2_sampling_and_spectra()
