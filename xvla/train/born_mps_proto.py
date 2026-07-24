"""Minimal prototype: conditional MPS (tensor-train) Born-machine action head.

Standalone NumPy (the repo's Python is 3.14 → no torch; this proves the math and
the training loop cheaply). The production head is the identical construction in
torch, with the MPS cores emitted by a *linear/bilinear* (foldable) map of the
chi-VLA action-query token h(obs) — see report. Here `obs` is a small context so
the whole thing runs in seconds.

Represents  p(a | obs) = |psi(a; obs)|^2 / Z(obs),  psi an MPS over the bits of a
binned action dimension. Everything tractable is a contraction:
  * Z            = < psi | psi >          (doubled transfer-matrix chain)
  * marginals    = partial doubled chain
  * conditionals = ratio of partial doubled chains
  * sampling     = exact ancestral, left->right, using right environments
Training minimizes NLL with the exact analytic Born gradient (Han et al. 2018).

Run:  python3 xvla/train/born_mps_proto.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# binning: map a scalar action a in [-1, 1] to k bits (physical dim 2 per site)
# ----------------------------------------------------------------------------
def a_to_bits(a, k):
    idx = np.clip(((a + 1.0) * 0.5 * (2 ** k)).astype(int), 0, 2 ** k - 1)
    bits = ((idx[:, None] >> np.arange(k - 1, -1, -1)) & 1)
    return bits, idx


def idx_to_a(idx, k):
    return (idx + 0.5) / (2 ** k) * 2.0 - 1.0


# ----------------------------------------------------------------------------
# conditional MPS: cores are AFFINE in the context feature phi = [1, c]  (degree-1
# = foldable/tensor-pure). Wcore[j] has shape (F, rL, 2, rR); A_j(c)=sum_f phi_f W.
# ----------------------------------------------------------------------------
class CondMPS:
    def __init__(self, k, bond=4, n_feat=2, scale=0.3):
        self.k = k
        self.F = n_feat
        rb = [1] + [bond] * (k - 1) + [1]           # open boundary bond dims
        self.rb = rb
        self.W = [scale * rng.standard_normal((n_feat, rb[j], 2, rb[j + 1]))
                  for j in range(k)]

    def cores(self, phi):                            # phi: (F,) -> list of A_j
        return [np.einsum("f,frsb->rsb", phi, self.W[j]) for j in range(self.k)]

    # -- amplitude of a batch of bitstrings ---------------------------------
    def amp(self, cores, bits):
        B = bits.shape[0]
        v = np.ones((B, 1))                          # left boundary (rb0 = 1)
        for j in range(self.k):
            A = cores[j][:, bits[:, j], :]           # (rL, B, rR)
            v = np.einsum("br,rbc->bc", v, A)        # advance the bond
        return v[:, 0]                               # (B,)

    # -- partition function Z = <psi|psi> (doubled chain) -------------------
    def Z(self, cores):
        M = [np.einsum("rsb,RsB->rRbB", A, A).reshape(A.shape[0] ** 2,
                                                       A.shape[2] ** 2)
             for A in cores]                          # transfer matrices
        z = np.ones(1)
        for Mj in M:
            z = z @ Mj
        return z[0], M

    # -- exact marginal pmf over all 2^k bins (small k only) ---------------
    def pmf(self, phi):
        cores = self.cores(phi)
        Z, _ = self.Z(cores)
        idx = np.arange(2 ** self.k)
        bits = ((idx[:, None] >> np.arange(self.k - 1, -1, -1)) & 1)
        psi = self.amp(cores, bits)
        return psi ** 2 / Z

    # -- exact ancestral sampling, left->right using right environments -----
    def sample(self, phi, n):
        cores = self.cores(phi)
        _, M = self.Z(cores)
        # right doubled environments: R[j] length rb[j]^2, R[k]=[1]
        R = [None] * (self.k + 1)
        R[self.k] = np.ones(1)
        for j in range(self.k - 1, -1, -1):
            R[j] = M[j] @ R[j + 1]
        out_bits = np.zeros((n, self.k), dtype=int)
        # doubled left state per sample: L (n, rb[j]^2); start (n,1)=1
        L = np.ones((n, 1))
        for j in range(self.k):
            A = cores[j]
            rL, _, rR = A.shape
            probs = np.zeros((n, 2))
            for s in (0, 1):
                As = A[:, s, :]                       # (rL, rR)
                Es = np.kron(As, As)                  # (rL^2, rR^2)
                # unnormalized weight of choosing bit=s then summing the rest
                w = (L @ Es) @ R[j + 1]               # (n,)
                probs[:, s] = w
            probs = np.clip(probs, 0, None)
            probs /= probs.sum(1, keepdims=True)
            u = rng.random(n)
            s_sel = (u > probs[:, 0]).astype(int)
            out_bits[:, j] = s_sel
            # advance doubled left state with the chosen bit
            newL = np.zeros((n, self.rb[j + 1] ** 2))
            for s in (0, 1):
                As = A[:, s, :]
                Es = np.kron(As, As)
                m = s_sel == s
                if m.any():
                    newL[m] = L[m] @ Es
            L = newL
        idx = (out_bits << np.arange(self.k - 1, -1, -1)).sum(1)
        return idx

    # -- analytic NLL + gradient over a batch (bits, context features) ------
    def nll_and_grad(self, phi, bits):
        cores = self.cores(phi)
        B = bits.shape[0]
        Z, M = self.Z(cores)
        # data term: log|psi| via left/right partial (single) amplitudes
        # precompute left/right partials for every sample and site
        PL = [np.ones((B, 1))]                        # PL[j]: (B, rb[j])
        for j in range(self.k):
            A = cores[j][:, bits[:, j], :]            # (rL, B, rR)
            PL.append(np.einsum("br,rbc->bc", PL[j], A))
        psi = PL[self.k][:, 0]
        PR = [None] * (self.k + 1)
        PR[self.k] = np.ones((B, 1))
        for j in range(self.k - 1, -1, -1):
            A = cores[j][:, bits[:, j], :]            # (rL, B, rR)
            PR[j] = np.einsum("rbc,bc->br", A, PR[j + 1])
        nll = -np.mean(2 * np.log(np.abs(psi) + 1e-12) - np.log(Z))

        # Z environments (doubled)
        ZL = [np.ones(1)]
        for j in range(self.k):
            ZL.append(ZL[j] @ M[j])
        ZR = [None] * (self.k + 1)
        ZR[self.k] = np.ones(1)
        for j in range(self.k - 1, -1, -1):
            ZR[j] = M[j] @ ZR[j + 1]

        gW = [np.zeros_like(self.W[j]) for j in range(self.k)]
        for j in range(self.k):
            rL, _, rR = cores[j].shape
            # --- data gradient of -(2/B) sum log|psi| wrt A_j ---
            gA = np.zeros((rL, 2, rR))
            for s in (0, 1):
                m = bits[:, j] == s
                if m.any():
                    # outer(PL[j], PR[j+1]) / psi, summed over matching samples
                    contrib = np.einsum("br,bc,b->rc",
                                        PL[j][m], PR[j + 1][m], 1.0 / psi[m])
                    gA[:, s, :] += -(2.0 / B) * contrib
            # --- logZ gradient (+1/Z * dZ/dA_j), dZ/dA = 2 LL A RR ---
            LL = ZL[j].reshape(rL, rL)
            RR = ZR[j + 1].reshape(rR, rR)
            for s in (0, 1):
                As = cores[j][:, s, :]
                gA[:, s, :] += (2.0 / Z) * (LL @ As @ RR)
            # chain to W (affine in phi): dL/dW[f] = phi[f] * gA
            gW[j] = np.einsum("f,rsb->frsb", phi, gA)
        return nll, gW, psi, Z

    def step(self, phi, bits, lr):
        nll, gW, _, _ = self.nll_and_grad(phi, bits)
        for j in range(self.k):
            self.W[j] -= lr * gW[j]
        return nll


# ============================================================================
# Part 1 — conditional bimodal 1-D target (obs swaps which two modes are active)
# ============================================================================
def part1():
    k = 5                                             # 32 bins over [-1, 1]
    mps = CondMPS(k, bond=4, n_feat=2, scale=0.4)

    def draw(context, n):
        # context 0: modes at -0.55 / +0.05 ; context 1: modes at -0.05 / +0.55
        if context == 0:
            mu = np.where(rng.random(n) < 0.5, -0.55, 0.05)
        else:
            mu = np.where(rng.random(n) < 0.5, -0.05, 0.55)
        a = np.clip(mu + 0.05 * rng.standard_normal(n), -0.999, 0.999)
        return a

    for it in range(4000):
        c = rng.integers(0, 2)
        phi = np.array([1.0, float(c)])
        a = draw(c, 256)
        bits, _ = a_to_bits(a, k)
        nll = mps.step(phi, bits, lr=0.05)
        if it % 1000 == 0:
            print(f"  [part1] it={it:4d}  nll={nll:.4f}")

    print("\n  Learned p(a|context) — top-4 bins each:")
    for c in (0, 1):
        pmf = mps.pmf(np.array([1.0, float(c)]))
        Z, _ = mps.Z(mps.cores(np.array([1.0, float(c)])))
        top = np.argsort(pmf)[::-1][:4]
        centers = idx_to_a(top, k)
        print(f"    context {c}: Z={Z:.4f} sum(p)={pmf.sum():.5f} "
              f"modes(a)={np.round(np.sort(centers), 2)}  p={np.round(pmf[top], 3)}")
        idx = mps.sample(np.array([1.0, float(c)]), 20000)
        samp = idx_to_a(idx, k)
        print(f"      sample mean={samp.mean():+.3f}  frac(a<0)={np.mean(samp < 0):.3f} "
              f"(bimodal => NOT concentrated at the mean)")


# ============================================================================
# Part 2 — the gripper: single-site (2-bin) conditional Born machine.
# psi(obs) = W [1; obs] in R^2 ; p = psi^2/||psi||^2 ; bins = {-1, +1}.
# Shows it COMMITS (p -> ~1 on the correct bin) where an MSE head -> mean ~0.
# ============================================================================
def part2():
    W = 0.3 * rng.standard_normal((2, 2))             # (bin, feat=[1,obs])

    def probs(obs):
        phi = np.stack([np.ones_like(obs), obs], 1)   # (B,2)
        psi = phi @ W.T                               # (B,2)
        p = psi ** 2
        return p / p.sum(1, keepdims=True), psi

    # data: obs<0 -> close(+1, bin1); obs>0 -> open(-1, bin0); |obs|<0.2 bimodal
    for it in range(3000):
        obs = rng.uniform(-1, 1, 512)
        y = (obs < 0).astype(int)                     # bin1 = +1 = close
        amb = np.abs(obs) < 0.2
        y[amb] = rng.integers(0, 2, amb.sum())        # genuinely bimodal middle
        p, psi = probs(obs)
        # NLL grad: d(-log p_y)/dW.  p_i = psi_i^2/Z, Z=sum psi^2
        phi = np.stack([np.ones_like(obs), obs], 1)
        g = np.zeros_like(W)
        Z = (psi ** 2).sum(1, keepdims=True)
        # d(-log p_y)/dpsi_i = -2/psi_y [i=y] + 2 psi_i / Z
        dpsi = 2 * psi / Z
        dpsi[np.arange(len(y)), y] -= 2.0 / psi[np.arange(len(y)), y]
        g = dpsi.T @ phi / len(y)
        W -= 0.1 * g
    # report
    for obs_val in (-0.8, -0.05, 0.05, 0.8):
        p, _ = probs(np.array([obs_val]))
        print(f"  [gripper] obs={obs_val:+.2f}  p(open=-1)={p[0,0]:.3f} "
              f"p(close=+1)={p[0,1]:.3f}  -> decode={'+1(close)' if p[0,1]>p[0,0] else '-1(open)'}")
    # MSE-head comparison on the ambiguous region: regresses to the mean ~0
    obs = rng.uniform(-0.2, 0.2, 5000)
    y = np.where(rng.random(5000) < 0.5, -1.0, 1.0)
    print(f"  [gripper] MSE head on ambiguous region would emit ~mean={y.mean():+.3f} "
          f"(=> indecisive); Born p(close)|obs=0 = {probs(np.array([0.0]))[0][0,1]:.3f}")


if __name__ == "__main__":
    print("Part 1: conditional MPS Born machine, bimodal 1-D target")
    part1()
    print("\nPart 2: gripper (2-bin conditional Born machine)")
    part2()
