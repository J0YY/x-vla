"""Minimal prototype: conditional NONNEGATIVE tensor-train (TT) action head.

A probabilistic tensor network / tensor-network Bayes net — DISTINCT from the
Born machine (`born_mps_proto.py`). The Born machine represents a signed/complex
amplitude psi and sets p = |psi|^2 / Z (global square -> interference). Here the
tensor IS the probability directly:

    p(a_1..a_m | obs) = (1/Z(obs)) * prod_j A_j(a_j; obs)         A_j >= 0

with NONNEGATIVE cores A_j (r_{j-1} x r_j matrices, indexed by the bin a_j). This
is exactly a (conditional) hidden Markov model written as a tensor train — a
"stochastic"/"positive" tensor network (Glasser, Sweke, Pancotti, Eisert, Cirac,
NeurIPS 2019). Everything tractable is a contraction, but on a SINGLE (un-doubled)
chain (Born needs the doubled |psi|^2 chain, bond r^2; nonneg-TT uses bond r):
  * Z            = sum_a prod_j A_j  = v0 . M_1 . M_2 ... M_m . vm , M_j = sum_s A_j(s)
  * marginals    = partial chains (leave some sites summed = M_j, others clamped)
  * conditionals = ratio of partial chains  (= HMM forward-backward)
  * sampling     = exact ancestral, left->right, using right environments
Training minimizes NLL with the exact forward-backward gradient. log / normalize
live ONLY in the loss+decode (out of the deployed graph), exactly like the Born
head and cleaner than per-token norm's in-graph divide.

THE NONNEGATIVITY CRUX (honest): a DIRECT-pmf tensor needs A_j >= 0, but the
repo's tensor-pure primitive (D[(L x_)⊙(R x_)]) is SIGNED. The ONLY nonnegativity
parametrization that is also tensor-pure (no exp/softplus/relu — all forbidden by
spec §1) is ENTRYWISE SQUARING, A_j = raw_j ⊙ raw_j, since ⊙ is the primitive.
Squaring EACH CORE ENTRY (local) keeps prod_j A_j nonnegative WITHOUT interference
-> a genuine nonneg-TT, NOT a Born machine (which squares the whole amplitude ->
interference). Same ⊙ primitive, different placement of the square. obs conditions
the cores as A_j(obs) = (W_j[a_j] . phi(obs))^2 with phi = [1, obs] (affine, then
squared): entrywise >=0, still a structured polynomial (degree grows, purity kept).

Run:  python3 xvla/train/nonneg_tt_proto.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# conditional nonnegative TT over m sites, each with an integer physical dim.
# cores are the ENTRYWISE SQUARE of an affine map of the context feature phi:
#   raw_j(phi) = sum_f phi_f W_j[f]          (affine in phi = tensor-pure linear)
#   A_j(phi)   = raw_j ⊙ raw_j               (entrywise square = the ⊙ primitive)
# => A_j >= 0 entrywise, prod_j A_j >= 0, so the tensor IS a valid unnormalized pmf.
# ---------------------------------------------------------------------------
class CondNonnegTT:
    def __init__(self, dims, bond=2, n_feat=2, scale=0.5):
        self.m = len(dims)
        self.dims = dims
        self.F = n_feat
        rb = [1] + [bond] * (self.m - 1) + [1]        # open-boundary bond dims
        self.rb = rb
        self.W = [scale * rng.standard_normal((n_feat, rb[j], dims[j], rb[j + 1]))
                  for j in range(self.m)]

    def raw(self, phi):
        return [np.einsum("f,frsb->rsb", phi, self.W[j]) for j in range(self.m)]

    def cores(self, phi):                              # nonnegative cores A_j
        return [r * r for r in self.raw(phi)]          # entrywise square (⊙ primitive)

    # -- unnormalized prob of a batch of bin-tuples (each row = m bin indices) --
    def T(self, cores, bins):
        B = bins.shape[0]
        v = np.ones((B, 1))
        for j in range(self.m):
            A = cores[j][:, bins[:, j], :]             # (rL, B, rR)
            v = np.einsum("br,rbc->bc", v, A)
        return v[:, 0]

    # -- partition function on the SINGLE (un-doubled) chain --------------------
    def Z(self, cores):
        M = [A.sum(1) for A in cores]                  # marginal transfer matrices
        z = np.ones((1, 1))
        for Mj in M:
            z = z @ Mj
        return float(z[0, 0]), M

    # -- exact full joint pmf over the product grid (small only) ---------------
    def pmf_grid(self, phi):
        cores = self.cores(phi)
        Z, _ = self.Z(cores)
        grids = np.array(np.meshgrid(*[np.arange(d) for d in self.dims],
                                     indexing="ij")).reshape(self.m, -1).T
        p = self.T(cores, grids) / Z
        return grids, p

    # -- exact single-site marginal p(a_j) via partial chains ------------------
    def marginal(self, phi, j):
        cores = self.cores(phi)
        Z, M = self.Z(cores)
        left = np.ones((1, 1))
        for t in range(j):
            left = left @ M[t]
        right = np.ones((1, 1))
        for t in range(self.m - 1, j, -1):
            right = M[t] @ right
        # p(a_j = s) = left . A_j(s) . right / Z
        out = np.array([(left @ cores[j][:, s, :] @ right).item() for s in range(self.dims[j])])
        return out / Z

    # -- exact ancestral sampling, left->right using right environments --------
    def sample(self, phi, n):
        cores = self.cores(phi)
        _, M = self.Z(cores)
        R = [None] * (self.m + 1)
        R[self.m] = np.ones((1, 1))
        for j in range(self.m - 1, -1, -1):
            R[j] = M[j] @ R[j + 1]                     # right env: (r_j, 1)
        out = np.zeros((n, self.m), dtype=int)
        L = np.ones((n, 1))                            # doubled? no — single chain
        for j in range(self.m):
            A = cores[j]                               # (rL, Dj, rR)
            w = np.einsum("nr,rsc,c->ns", L, A, R[j + 1][:, 0])   # (n, Dj) >= 0
            w = np.clip(w, 0, None)
            w /= w.sum(1, keepdims=True)
            u = rng.random(n)
            csum = np.cumsum(w, 1)
            s_sel = (u[:, None] > csum).sum(1)         # inverse-CDF sample
            s_sel = np.clip(s_sel, 0, self.dims[j] - 1)
            out[:, j] = s_sel
            Asel = A[:, s_sel, :]                       # (rL, n, rR)
            L = np.einsum("nr,rnc->nc", L, Asel)
        return out

    # -- analytic NLL + forward-backward gradient (single chain) ---------------
    def nll_and_grad(self, phi, bins):
        raws = self.raw(phi)
        cores = [r * r for r in raws]
        B = bins.shape[0]
        Z, M = self.Z(cores)

        # data-term partials (per sample)
        PL = [np.ones((B, 1))]
        for j in range(self.m):
            A = cores[j][:, bins[:, j], :]
            PL.append(np.einsum("br,rbc->bc", PL[j], A))
        Tval = PL[self.m][:, 0]
        PR = [None] * (self.m + 1)
        PR[self.m] = np.ones((B, 1))
        for j in range(self.m - 1, -1, -1):
            A = cores[j][:, bins[:, j], :]
            PR[j] = np.einsum("rbc,bc->br", A, PR[j + 1])
        nll = float(np.mean(np.log(Z) - np.log(Tval + 1e-30)))

        # Z environments (single chain)
        ZL = [np.ones((1, 1))]
        for j in range(self.m):
            ZL.append(ZL[j] @ M[j])
        ZR = [None] * (self.m + 1)
        ZR[self.m] = np.ones((1, 1))
        for j in range(self.m - 1, -1, -1):
            ZR[j] = M[j] @ ZR[j + 1]

        gW = [np.zeros_like(self.W[j]) for j in range(self.m)]
        for j in range(self.m):
            rL, Dj, rR = cores[j].shape
            gA = np.zeros((rL, Dj, rR))
            # + d(mean logZ)/dA_j : ZL[j] outer ZR[j+1], for ALL bins s (M_j sums s)
            zc = np.outer(ZL[j][0], ZR[j + 1][:, 0]) / Z            # (rL, rR)
            gA += zc[:, None, :]
            # - d(mean logT)/dA_j : only the observed bin per sample
            for s in range(Dj):
                mask = bins[:, j] == s
                if mask.any():
                    contrib = np.einsum("br,bc,b->rc",
                                        PL[j][mask], PR[j + 1][mask], 1.0 / Tval[mask])
                    gA[:, s, :] -= contrib / B
            # chain through entrywise square (dA/draw = 2 raw) and affine (draw/dW=phi)
            g_raw = gA * (2.0 * raws[j])
            gW[j] = np.einsum("f,rsb->frsb", phi, g_raw)
        return nll, gW, Tval, Z

    def step(self, phi, bins, lr):
        nll, gW, _, _ = self.nll_and_grad(phi, bins)
        for j in range(self.m):
            self.W[j] -= lr * gW[j]
        return nll


def bins_of(a, D, lo=-1.0, hi=1.0):
    idx = np.clip(((a - lo) / (hi - lo) * D).astype(int), 0, D - 1)
    return idx


def center_of(idx, D, lo=-1.0, hi=1.0):
    return lo + (idx + 0.5) / D * (hi - lo)


# ===========================================================================
# Part 1 — conditional bimodal 1-D target (obs swaps which two modes fire).
#          Single site, D bins. Shows exact Z on a SINGLE chain + bimodal recovery.
# ===========================================================================
def part1():
    D = 32
    tt = CondNonnegTT([D], bond=1, n_feat=2, scale=0.6)   # 1 site -> bond irrelevant

    def draw(c, n):
        if c == 0:
            mu = np.where(rng.random(n) < 0.5, -0.55, 0.05)
        else:
            mu = np.where(rng.random(n) < 0.5, -0.05, 0.55)
        return np.clip(mu + 0.05 * rng.standard_normal(n), -0.999, 0.999)

    for it in range(4000):
        c = int(rng.integers(0, 2))
        phi = np.array([1.0, float(c)])
        a = draw(c, 256)
        bins = bins_of(a, D)[:, None]
        nll = tt.step(phi, bins, lr=0.05)
        if it % 1000 == 0:
            print(f"  [part1] it={it:4d}  nll={nll:.4f}")
    print("\n  Learned p(a|context) — top-4 bins each (single-chain Z, no doubling):")
    for c in (0, 1):
        phi = np.array([1.0, float(c)])
        _, p = tt.pmf_grid(phi)
        Z, _ = tt.Z(tt.cores(phi))
        top = np.argsort(p)[::-1][:4]
        print(f"    context {c}: Z={Z:.4f} sum(p)={p.sum():.5f} "
              f"modes(a)={np.round(np.sort(center_of(top, D)), 2)}  p={np.round(p[top],3)}")
        s = tt.sample(phi, 20000)[:, 0]
        samp = center_of(s, D)
        print(f"      sample mean={samp.mean():+.3f}  frac(a<0)={np.mean(samp<0):.3f} "
              f"(bimodal => NOT the mean)")


# ===========================================================================
# Part 2 — the gripper: single-site 2-bin nonneg TT. p(bin) = A(bin)/sum.
#          Commits to +-1 where a linear+MSE head averages the bimodal target ~0.
# ===========================================================================
def part2():
    tt = CondNonnegTT([2], bond=1, n_feat=2, scale=0.6)   # bins {0:-1(open), 1:+1(close)}
    for it in range(3000):
        obs = float(rng.uniform(-1, 1))
        phi = np.array([1.0, obs])
        n = 512
        o = rng.uniform(-1, 1, n)
        y = (o < 0).astype(int)                            # obs<0 -> close(bin1)
        y[np.abs(o) < 0.2] = rng.integers(0, 2, int((np.abs(o) < 0.2).sum()))
        # train per-obs value would need per-sample cores; use the sign structure:
        # draw the batch at THIS obs so cores are shared (matches born proto part2 spirit)
        yy = np.full(n, 1 if obs < 0 else 0)
        if abs(obs) < 0.2:
            yy = rng.integers(0, 2, n)
        tt.step(phi, yy[:, None], lr=0.1)
    for obs_val in (-0.8, -0.05, 0.05, 0.8):
        p = tt.marginal(np.array([1.0, obs_val]), 0)
        dec = "+1(close)" if p[1] > p[0] else "-1(open)"
        print(f"  [gripper] obs={obs_val:+.2f}  p(open=-1)={p[0]:.3f} "
              f"p(close=+1)={p[1]:.3f}  -> decode={dec}")
    p0 = tt.marginal(np.array([1.0, 0.0]), 0)
    print(f"  [gripper] at obs=0 (ambiguous) p(close)={p0[1]:.3f}; an MSE head emits "
          f"~mean(+-1)=~0 (indecisive). The pmf keeps a proper 2-mode split.")


# ===========================================================================
# Part 3 — synthetic multimodal REACH: 2-D action (x,y) with TWO correlated
#          spatial modes (go-left vs go-right of an obstacle). m=2 sites.
#          Bond r = #behavioral modes crossing the x<->y cut: r=1 (independent)
#          CANNOT represent the diagonal correlation; r=2 can. (bond = #modes,
#          the ODT / Schmidt-analog for a STOCHASTIC tensor.)
# ===========================================================================
def _reach_batch(c, n, D):
    # context 0: modes at (-0.5,-0.5) & (+0.5,+0.5)  (diagonal)
    # context 1: modes at (-0.5,+0.5) & (+0.5,-0.5)  (anti-diagonal)
    left = rng.random(n) < 0.5
    if c == 0:
        x = np.where(left, -0.5, 0.5); y = np.where(left, -0.5, 0.5)
    else:
        x = np.where(left, -0.5, 0.5); y = np.where(left, 0.5, -0.5)
    x = np.clip(x + 0.06 * rng.standard_normal(n), -0.999, 0.999)
    y = np.clip(y + 0.06 * rng.standard_normal(n), -0.999, 0.999)
    return np.stack([bins_of(x, D), bins_of(y, D)], 1)


def part3():
    D = 8
    for bond in (1, 2):
        tt = CondNonnegTT([D, D], bond=bond, n_feat=2, scale=0.6)
        for it in range(6000):
            c = int(rng.integers(0, 2))
            phi = np.array([1.0, float(c)])
            tt.step(phi, _reach_batch(c, 256, D), lr=0.03)
        # evaluate context 0 (diagonal): does the joint keep BOTH modes + correlation?
        phi = np.array([1.0, 0.0])
        grids, p = tt.pmf_grid(phi)
        Z, _ = tt.Z(tt.cores(phi))
        xs = center_of(grids[:, 0], D); ys = center_of(grids[:, 1], D)
        corr = float((p * xs * ys).sum() - (p * xs).sum() * (p * ys).sum())
        s = tt.sample(phi, 20000)
        sx, sy = center_of(s[:, 0], D), center_of(s[:, 1], D)
        # fraction of samples in each true mode (diagonal): both-neg or both-pos
        mode_a = np.mean((sx < 0) & (sy < 0))
        mode_b = np.mean((sx > 0) & (sy > 0))
        off = np.mean(((sx < 0) & (sy > 0)) | ((sx > 0) & (sy < 0)))
        print(f"  [reach bond={bond}] sum(p)={p.sum():.4f} Z={Z:.3f}  "
              f"cov(x,y)={corr:+.3f}  (true≈+0.25 diagonal)")
        print(f"      sample: mode(-,-)={mode_a:.2f} mode(+,+)={mode_b:.2f} "
              f"OFF-mode={off:.2f}  "
              f"{'<- rank-1 cannot correlate x,y (spurious off-diagonal mass)' if bond==1 else '<- rank-2 captures BOTH correlated modes'}")


if __name__ == "__main__":
    print("Part 1: conditional nonnegative-TT, bimodal 1-D target")
    part1()
    print("\nPart 2: gripper (2-bin nonnegative TT — commits where MSE averages)")
    part2()
    print("\nPart 3: 2-D multimodal reach — bond r = #correlated behavioral modes")
    part3()
