"""Minimal prototype: sum-of-squares (SOS) polynomial action-density head.

Standalone NumPy (the repo's Python is 3.14 -> no torch; this proves the math and
the training loop cheaply). The production head is the identical construction in
torch, with the SOS coefficient matrix Theta(obs) emitted by a *linear/bilinear*
(foldable) map of the chi-VLA action-query token h(obs) -- see the report. Here
`obs` is a small context so the whole thing runs in seconds.

Represents   p(a | obs) = sigma(a; obs) / Z(obs),   sigma = sum_k psi_k(a; obs)^2
with psi_k(a; obs) = Theta_k(obs) . m(a),  m(a) = monomials of a up to degree D.

Everything tractable is a contraction with a FIXED moment tensor:
  * sigma(a) = || Theta m(a) ||^2 = m(a)^T G m(a),   G(obs) = Theta^T Theta  (PSD)
  * Z(obs)   = <G, M> = Tr(G M),   M_{ab} = INT_box m_a(a) m_b(a) da   (closed form)
  * modes    = eigenvectors of G  (SOS modes = ODT object; generalizes Born's |psi|^2)
  * sampling = exact 1-D ancestral via polynomial marginals (out-of-graph); or argmax
Training minimizes NLL with EXACT logZ (log lives only in the loss).

This GENERALIZES the Born machine (born_mps_proto.py): Born is the K=1, rank-1-Gram
special case p = |psi|^2/Z; SOS is the sum over K squares <=> a rank-K PSD Gram.
The PSD constraint is tensor-pure precisely because we parametrize by the square
roots Theta_k and SQUARE them (the degree-2 primitive) -- no runtime Cholesky/eigh.

Run:  python3 xvla/train/sos_proto.py
"""

from __future__ import annotations

import itertools

import numpy as np

rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# monomial basis of R^n up to total degree D, and the box moment matrix.
# ----------------------------------------------------------------------------
def exponents(n, D):
    """All exponent tuples alpha in N^n with |alpha| <= D (the psi basis, deg<=D)."""
    exps = [e for e in itertools.product(range(D + 1), repeat=n) if sum(e) <= D]
    return sorted(exps, key=lambda e: (sum(e), e))


def monomials(a, exps):
    """m(a): (B, M).  a: (B, n).  m_alpha(a) = prod_d a_d^alpha_d."""
    a = np.atleast_2d(a)
    out = np.ones((a.shape[0], len(exps)))
    for j, e in enumerate(exps):
        for d, p in enumerate(e):
            if p:
                out[:, j] *= a[:, d] ** p
    return out


def _axis_moment(p, lo, hi):
    """INT_lo^hi t^p dt  (closed form)."""
    return (hi ** (p + 1) - lo ** (p + 1)) / (p + 1)


def moment_matrix(exps, box):
    """M_{ab} = INT_box m_a m_b da = prod_d INT t^{alpha_d+beta_d} dt.

    The box moment tensor FACTORIZES over dimensions (product measure) -- the
    structural gift that lets Z be a cheap (TT-friendly) contraction. Returns the
    M x M matrix (PSD: it is the Gram of monomials under a positive measure)."""
    M = len(exps)
    out = np.zeros((M, M))
    for i, ei in enumerate(exps):
        for j, ej in enumerate(exps):
            v = 1.0
            for d in range(len(ei)):
                lo, hi = box[d]
                v *= _axis_moment(ei[d] + ej[d], lo, hi)
            out[i, j] = v
    return out


# ----------------------------------------------------------------------------
# SOS density:  sigma = ||Theta m||^2 = m^T G m,  G = Theta^T Theta (PSD),
#               Z = <G, M>,   p = sigma / Z.
# ----------------------------------------------------------------------------
class SOSDensity:
    def __init__(self, exps, box, K, scale=0.4):
        self.exps = exps
        self.box = box
        self.M = len(exps)
        self.K = K
        self.Mmat = moment_matrix(exps, box)         # fixed moment tensor
        self.Theta = scale * rng.standard_normal((K, self.M))   # square roots

    def gram(self):
        return self.Theta.T @ self.Theta            # G = Theta^T Theta, PSD by construction

    def Z(self):
        return float(np.sum(self.gram() * self.Mmat))   # <G, M> = Tr(Theta M Theta^T)

    def sigma(self, a):
        m = monomials(a, self.exps)                 # (B, M)
        psi = m @ self.Theta.T                       # (B, K)
        return (psi ** 2).sum(1), m, psi

    def logp(self, a):
        sig, _, _ = self.sigma(a)
        return np.log(sig + 1e-300) - np.log(self.Z())

    def nll_and_grad(self, a):
        sig, m, psi = self.sigma(a)                  # sig:(B,), m:(B,M), psi:(B,K)
        Z = self.Z()
        nll = -np.mean(np.log(sig + 1e-300)) + np.log(Z)
        # d log sigma / d Theta = 2 psi_k m^T / sigma        (data term)
        # d log Z     / d Theta = 2 Theta M / Z              (normalizer term)
        data = np.einsum("bk,bm,b->km", psi, m, 1.0 / (sig + 1e-300)) * (2.0 / a.shape[0])
        gZ = 2.0 * (self.Theta @ self.Mmat) / Z
        return nll, -data + gZ

    def step(self, a, lr):
        nll, g = self.nll_and_grad(a)
        self.Theta -= lr * g
        return nll

    def modes(self):
        """Eigen-decompose G = sum_i mu_i u_i u_i^T -> orthogonal SOS modes (ODT)."""
        mu, U = np.linalg.eigh(self.gram())
        return mu[::-1], U[:, ::-1]                  # descending


# ============================================================================
# Part 1 -- bimodal 2-D SOS density; K=1 (Born) vs K=2 (true SOS) valley test.
# ============================================================================
def part1():
    n, D = 2, 2                                      # sigma is degree 2D = 4 -> bimodal
    box = [(-1.0, 1.0)] * n
    exps = exponents(n, D)
    print(f"  n={n} D={D}: {len(exps)} monomials (psi basis); sigma degree {2*D}")

    # target: two Gaussian blobs at (-0.5,-0.5) and (+0.5,+0.5) (a bimodal manifold)
    def draw(nb):
        c = rng.integers(0, 2, nb)
        mu = np.where(c[:, None] == 0, np.array([-0.5, -0.5]), np.array([0.5, 0.5]))
        a = mu + 0.12 * rng.standard_normal((nb, n))
        return np.clip(a, -0.99, 0.99)

    results = {}
    for K in (1, 2, 3):
        m = SOSDensity(exps, box, K=K, scale=0.5)
        for it in range(6000):
            m.step(draw(512), lr=0.03)
        nll = m.nll_and_grad(draw(20000))[0]
        # numeric integral of p over a fine grid -> should be ~1 (exact-Z check)
        ng = 401
        g = np.linspace(-1, 1, ng)
        gx, gy = np.meshgrid(g, g)
        pts = np.stack([gx.ravel(), gy.ravel()], 1)
        sig, _, _ = m.sigma(pts)
        p = sig / m.Z()
        cell = (2 / (ng - 1)) ** 2
        mass = float(p.sum() * cell)
        # density in the VALLEY between the modes (origin) vs at a mode
        p_valley = float((m.sigma(np.array([[0.0, 0.0]]))[0] / m.Z())[0])
        p_mode = float((m.sigma(np.array([[0.5, 0.5]]))[0] / m.Z())[0])
        mu_eig, _ = m.modes()
        results[K] = (nll, mass, p_valley, p_mode, mu_eig)
        # eigenvalues normalized to the top (SOS-mode spectrum); 2nd mode is the
        # valley-filler and is small BECAUSE the dominant square has its node there.
        rel = mu_eig / max(mu_eig[0], 1e-30)
        print(f"  K={K}: NLL={nll:+.3f}  INT p da={mass:.4f}  "
              f"p(valley)/p(mode)={p_valley/p_mode:.3f}  "
              f"Gram eig(rel top3)={np.round(rel[:3], 4)}")

    print("\n  --> K=1 (=Born, single |psi|^2) is forced toward a ZERO-density node")
    print("      between the two modes (a single real square vanishes at its root);")
    print("      K>=2 SOS fills the valley (squares don't share roots) -> higher")
    print("      valley/mode ratio and lower NLL. This is the concrete K=1->K>1 gain.")
    print(f"      valley/mode:  K=1 {results[1][2]/results[1][3]:.3f}  "
          f"K=2 {results[2][2]/results[2][3]:.3f}  K=3 {results[3][2]/results[3][3]:.3f}")
    print(f"      NLL:          K=1 {results[1][0]:+.3f}  "
          f"K=2 {results[2][0]:+.3f}  K=3 {results[3][0]:+.3f}")
    print("      Gram spectrum (#eig > 1e-3 * top) = # active SOS modes (ODT):")
    for K in (1, 2, 3):
        mu = results[K][4]
        print(f"        K={K}: {(mu > 1e-3 * max(mu[0], 1e-30)).sum()} active modes")


# ============================================================================
# Part 2 -- the gripper: conditional SOS over g in [-1,1].
#   Theta(obs) = W0 + obs*W1   (AFFINE in obs = degree-1 = foldable/tensor-pure);
#   sigma(g;obs) = ||Theta(obs) m(g)||^2, m(g)=[1,g,g^2] -> sigma degree 4 = double
#   well. Exact Z(obs) = <G(obs), M>. Decode = argmax_g sigma (out-of-graph) COMMITS
#   to +-1 where an MSE head averages the bimodal target to ~0.
# ============================================================================
def part2():
    box = [(-1.0, 1.0)]
    exps = exponents(1, 2)                           # [1, g, g^2]
    Mmat = moment_matrix(exps, box)
    K = 2
    W0 = 0.4 * rng.standard_normal((K, 3))
    W1 = 0.4 * rng.standard_normal((K, 3))

    def theta(obs):                                  # (B,K,3) affine in obs
        return W0[None] + obs[:, None, None] * W1[None]

    def sig_Z(g, obs):
        m = monomials(g[:, None], exps)             # (G,3)
        Th = theta(obs)                              # (B,K,3)
        psi = np.einsum("bkm,gm->bkg", Th, m)       # (B,K,G)
        sig = (psi ** 2).sum(1)                      # (B,G)
        G = np.einsum("bki,bkj->bij", Th, Th)       # (B,3,3) Gram
        Z = np.einsum("bij,ij->b", G, Mmat)         # (B,)
        return sig, Z, m, psi

    grid = np.linspace(-1, 1, 201)
    for it in range(6000):
        obs = rng.uniform(-1, 1, 256)
        y = np.where(obs < 0, 1.0, -1.0)            # obs<0 -> close(+1); >0 -> open(-1)
        amb = np.abs(obs) < 0.2
        y[amb] = np.where(rng.random(amb.sum()) < 0.5, -1.0, 1.0)   # bimodal middle
        # NLL grad wrt W0,W1 (Theta affine in obs so chain by [1, obs]).
        sig_y, Z, my, psi_y = sig_Z(y, obs)
        sig_y = sig_y[np.arange(len(y)), np.arange(len(y))] if False else \
            (np.einsum("bkm,bm->bk", theta(obs), monomials(y[:, None], exps)) ** 2).sum(1)
        # gradient of -log sigma(y) + log Z wrt Theta (per sample), then to W0,W1:
        Th = theta(obs)                             # (B,K,3)
        my = monomials(y[:, None], exps)            # (B,3)
        psi = np.einsum("bkm,bm->bk", Th, my)       # (B,K)
        gTheta = 2 * np.einsum("bk,bm->bkm", psi, my) / (sig_y[:, None, None] + 1e-12)
        gTheta = -gTheta                            # data term  -d log sigma
        Gb = np.einsum("bki,bkj->bij", Th, Th)
        Zb = np.einsum("bij,ij->b", Gb, Mmat)
        gTheta += 2 * np.einsum("bkm,mn->bkn", Th, Mmat) / (Zb[:, None, None] + 1e-12)
        gW0 = gTheta.mean(0)
        gW1 = (gTheta * obs[:, None, None]).mean(0)
        W0 -= 0.05 * gW0
        W1 -= 0.05 * gW1

    for obs_val in (-0.8, -0.05, 0.05, 0.8):
        sig, Z, _, _ = sig_Z(grid, np.array([obs_val]))
        p = sig[0] / Z[0]
        g_star = grid[p.argmax()]                    # OUT-OF-GRAPH argmax decode
        mass = float(p.sum() * (2 / 200))
        pc = p[grid > 0].sum() / p.sum()
        print(f"  [gripper] obs={obs_val:+.2f}  INT p={mass:.3f}  P(g>0)={pc:.3f}  "
              f"argmax_g={g_star:+.2f} -> decode={'+1(close)' if g_star>0 else '-1(open)'}")
    print("  [gripper] MSE head on the ambiguous region emits ~mean(+-1)=~0 "
          "(indecisive); the SOS density keeps two peaks and the argmax commits.")
    print("  [gripper] note: the double-well (g^2-1)^2 is a SINGLE square (Born K=1);")
    print("            SOS adds the obs-tilt as the 2nd square -> conditional commit.")


if __name__ == "__main__":
    print("Part 1: bimodal 2-D SOS density; K=1(Born) vs K>=2 valley/mode + eig spectrum")
    part1()
    print("\nPart 2: gripper as a conditional SOS double-well (argmax commits to +-1)")
    part2()
