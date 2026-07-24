"""Minimal prototype: monotone-polynomial AUTOREGRESSIVE QUANTILE action head.

Standalone NumPy (the repo's Python is 3.14 -> no torch; this proves the math and
the training loop cheaply). The production head is the identical construction in
torch, with the quantile-map coefficients emitted by a *linear/bilinear* (foldable)
map of the chi-VLA action-query token h(obs) -- see report. Here `obs` is a small
context so the whole thing runs in seconds.

REFRAME (distinct from the pushforward / Born / energy / K-candidate / latent
siblings): represent the policy by its conditional QUANTILE function.

    a_k = Q_k(tau_k | a_<k, obs),     tau_k ~ U[0,1]

Each Q_k is a MONOTONE polynomial in tau_k in [0,1]. Monotonicity is guaranteed as
a TENSOR-PURE algebraic identity via the SOS-integral construction:

    s(u)   = sum_{j=0..p} c_j u^j                          (a free polynomial)
    Q'(tau) = s(tau)^2 >= 0                                (squaring = degree-2 = pure)
    Q(tau)  = b + integral_0^tau s(u)^2 du                 (a polynomial of deg 2p+1)

Writing s(u)^2 = sum_m q_m u^m with q_m = sum_{j+l=m} c_j c_l and integrating:

    Q(tau) = b + sum_{m=0..2p} w_m tau^{m+1},   w_m = q_m/(m+1) = c^T A_m c

so every quantile-map coefficient w_m is a fixed SYMMETRIC BILINEAR (degree-2) form
in the coefficient vector c -- exactly the odt.py / odt_interp.py Q_c object. The map
obs -> (c, b) is itself linear/bilinear in the context feature (foldable). Q'=s^2>=0
holds for ANY weights, so monotonicity (=> a valid quantile => valid density)
survives folding/ODT with no clamp/relu/softmax anywhere in the graph.

Multimodality = plateaus of Q (mass concentrates where Q' is small -- the caustic /
quantile view of modes). Crisply: every real root of s in (0,1) makes Q'=0 there,
a density spike => a MODE. #modes = #roots of s in (0,1).

Sampling: draw tau ~ U[0,1]^m OUT-OF-GRAPH (a controller RNG op, like the gripper
threshold / OpenVLA's argmax outside its tensor boundary); then a_k=Q_k(tau_k|...) is
an IN-GRAPH polynomial forward pass and IS the sample. No argmin (energy), no
partition function (Born), no ancestral per-bit loop -- one forward eval per tau.

Contrast with the pushforward sibling a=G(obs,z): G is a general (non-monotone)
polynomial, modes = fold caustics G'=0, many-to-one => NO closed-form marginal/CDF.
The quantile map is the MONOTONE (Knothe-Rosenblatt / increasing-transport) case of
the same pushforward: invertible, so it gives EXACT 1-D marginals, median, and the
inverse-CDF in closed form -- the property the report contrasts.

Run:  python3 xvla/train/quantile_proto.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


# ===========================================================================
# The SOS-integral monotone quantile map  Q(tau) = b + integral_0^tau s(u)^2 du
# ===========================================================================
def sos_integral_matrices(p: int):
    """Return A_m (m=0..2p), each (p+1)x(p+1), with  w_m = c^T A_m c.

    A_m[j,l] = 1/(m+1) if j+l==m else 0  (symmetric).  Q(tau)=b+sum_m w_m tau^{m+1}.
    """
    A = []
    for m in range(2 * p + 1):
        Am = np.zeros((p + 1, p + 1))
        for j in range(p + 1):
            l = m - j
            if 0 <= l <= p:
                Am[j, l] = 1.0 / (m + 1)
        A.append(0.5 * (Am + Am.T))          # symmetrize (the ODT convention)
    return A


def q_coeffs(c, A):
    """w_m = c^T A_m c  (the folded bilinear map coeff-vector -> quantile coeffs)."""
    return np.array([c @ Am @ c for Am in A])            # (2p+1,)


def q_coeffs_batch(c, Astack):
    """Batched w = c^T A_m c.  c:(n,p+1), Astack:(2p+1,p+1,p+1) -> (n,2p+1)."""
    Ac = np.einsum("mij,nj->nmi", Astack, c)             # (n, 2p+1, p+1)
    return np.einsum("nmi,ni->nm", Ac, c)               # (n, 2p+1)


def dQ_dc_batch(c, Astack, tau):
    """Batched dQ/dc.  c:(n,p+1), tau:(n,) -> (n, p+1)."""
    Ac = np.einsum("mij,nj->nmi", Astack, c)             # (n, 2p+1, p+1)
    M = Astack.shape[0]
    powers = np.stack([tau ** (m + 1) for m in range(M)], -1)   # (n, 2p+1)
    return 2.0 * np.einsum("nm,nmi->ni", powers, Ac)    # (n, p+1)


def Q_eval(c, b, A, tau):
    """Q(tau) = b + sum_m w_m tau^{m+1}.  tau: (...,) -> (...,)."""
    w = q_coeffs(c, A)                                    # (2p+1,)
    powers = np.stack([tau ** (m + 1) for m in range(len(w))], -1)   # (..., 2p+1)
    return b + powers @ w


def Qprime_eval(c, tau):
    """Q'(tau) = s(tau)^2  (>= 0 by construction)."""
    p = len(c) - 1
    s = sum(c[j] * tau ** j for j in range(p + 1))
    return s * s


def s_roots_in_unit(c):
    """Real roots of s in (0,1) = the modes of the marginal."""
    r = np.roots(c[::-1]) if len(c) > 1 else np.array([])
    real = r[np.abs(r.imag) < 1e-6].real
    return np.sort(real[(real > 1e-3) & (real < 1 - 1e-3)])


# gradient of Q(tau) w.r.t. c  (analytic; needed because NumPy has no autodiff)
def dQ_dc(c, A, tau):
    """dQ/dc_j = sum_m (2 (A_m c)_j) tau^{m+1}.   Returns (..., p+1)."""
    Ac = np.stack([Am @ c for Am in A], 0)               # (2p+1, p+1)
    powers = np.stack([tau ** (m + 1) for m in range(len(A))], -1)   # (..., 2p+1)
    return 2.0 * powers @ Ac                             # (..., p+1)


# ===========================================================================
# A tiny Adam so the proto trains without torch.
# ===========================================================================
class Adam:
    def __init__(self, shapes, lr=5e-2):
        self.lr = lr
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (pr, g) in enumerate(zip(params, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            pr -= self.lr * mh / (np.sqrt(vh) + 1e-8)
        return params


# ===========================================================================
# Part 1 -- unconditional bimodal target, THREE ways.
#   (a) MSE mean-only head  -> collapses to the mean (the 0% gripper failure).
#   (b) pinball / IQN quantile regression (sample-based, no density).
#   (c) exact inverse-CDF matching (least squares to empirical quantiles).
# Target: 50/50 mixture of N(-1, .08) and N(+1, .08)  (the bimodal gripper).
# ===========================================================================
def target_bimodal(n):
    m = rng.random(n) < 0.5
    return np.where(m, 1.0, -1.0) + 0.08 * rng.standard_normal(n)


def pinball(u, tau):
    """rho_tau(u) = u (tau - 1{u<0}) ;  u = a* - Q(tau).  (loss-only indicator)."""
    return u * (tau - (u < 0).astype(float))


def part1():
    print("=" * 74)
    print("PART 1  unconditional bimodal target  (0.5*d(-1)+0.5*d(+1), sd .08)")
    print("=" * 74)
    p = 4                                                 # s degree 4 -> Q degree 9
    A = sos_integral_matrices(p)
    data = target_bimodal(20000)

    # (a) MSE mean-only head (a scalar) -- the current chi-VLA linear+MSE head.
    mean_pred = data.mean()
    print(f"(a) MSE head   -> constant a = {mean_pred:+.3f}   "
          f"(mean of a bimodal target; commits to NEITHER mode)")

    # (b) pinball / IQN quantile regression: sample tau, regress Q(tau) to data.
    # warm init: s == 1 (=> Q = b + tau, monotone linear), b at the median.
    c = np.zeros(p + 1); c[0] = 1.0
    b = float(np.median(data))
    opt = Adam([(p + 1,), ()], lr=5e-2)
    for it in range(12000):
        idx = rng.integers(0, len(data), size=256)
        a_star = data[idx]
        tau = rng.random(256)
        Q = Q_eval(c, b, A, tau)
        u = a_star - Q
        w_ind = tau - (u < 0).astype(float)              # dL/dQ = -w_ind
        dLdQ = -w_ind
        gc = (dLdQ[:, None] * dQ_dc(c, A, tau)).mean(0)
        gb = dLdQ.mean()
        c, b = opt.step([c, b], [gc, np.array(gb)])
    c_pin, b_pin = c.copy(), b
    roots = s_roots_in_unit(c_pin)
    samp = Q_eval(c_pin, b_pin, A, rng.random(20000))
    commit = np.mean(np.abs(samp) > 0.5)
    frac_plus = np.mean(samp > 0)
    print(f"(b) pinball/IQN quantile head (sample-based, NO density used):")
    print(f"      s has {len(roots)} root(s) in (0,1) -> {len(roots)} mode(s): "
          f"{np.round(roots,3).tolist()}")
    print(f"      Q(.1),Q(.5),Q(.9) = {Q_eval(c_pin,b_pin,A,np.array([.1,.5,.9])).round(3)}")
    print(f"      sampled: mean {samp.mean():+.3f}  frac(+)={frac_plus:.2f}  "
          f"commit |a|>0.5 = {commit:.3f}   (MSE head commit = 0.000)")

    # (c) exact inverse-CDF matching: fit Q to the empirical quantiles F^{-1}(tau).
    taus = (np.arange(1, 200) / 200.0)
    Finv = np.quantile(data, taus)
    c = np.zeros(p + 1); c[0] = 1.0
    b = float(np.median(data))
    opt = Adam([(p + 1,), ()], lr=5e-2)
    for it in range(12000):
        Q = Q_eval(c, b, A, taus)
        r = Q - Finv                                     # L = 1/2||Q-Finv||^2
        gc = (r[:, None] * dQ_dc(c, A, taus)).mean(0)
        gb = r.mean()
        c, b = opt.step([c, b], [gc, np.array(gb)])
    err = np.sqrt(np.mean((Q_eval(c, b, A, taus) - Finv) ** 2))
    print(f"(c) exact inverse-CDF matching: RMSE(Q, F^-1) = {err:.4f}  "
          f"(closed-form quantiles/median available)")

    # fold / ODT identity check: w_m = c^T A_m c reproduces direct integration.
    w = q_coeffs(c_pin, A)
    # direct: integrate s^2 on a grid
    grid = np.linspace(0, 1, 20001)
    s2 = Qprime_eval(c_pin, grid)
    Q_num = b_pin + np.cumsum(s2) * (grid[1] - grid[0])
    Q_alg = Q_eval(c_pin, b_pin, A, grid)
    print(f"    FOLD CHECK: max|Q_bilinear - Q_integrated| = "
          f"{np.max(np.abs(Q_alg - Q_num)):.2e}   (w_m = c^T A_m c is exact)")
    print(f"    MONOTONE CHECK: min Q'(tau) over grid = {s2.min():.2e}  (>=0 always)")


# ===========================================================================
# Part 2 -- CONDITIONAL (grounding): obs in {0,1}, mode mass depends on obs.
#   c(h)=C h + c0,  b(h)=beta.h + b0  (AFFINE map obs->coeffs; degree-2/CP in prod).
#   obs=0 -> P(+1)=0.2 ;  obs=1 -> P(+1)=0.8.  Bayes commit ceiling = 0.8.
# ===========================================================================
def part2():
    print()
    print("=" * 74)
    print("PART 2  conditional / grounding  (obs sets the mode mass; head must read it)")
    print("=" * 74)
    p = 4
    A = sos_integral_matrices(p)
    H = 3                                                 # feature dim of obs

    def batch(n):
        o = rng.integers(0, 2, n)                         # obs bit
        h = np.zeros((n, H)); h[:, 0] = 1.0; h[:, 1] = o  # [bias, obs, unused]
        pplus = np.where(o == 1, 0.8, 0.2)
        a = np.where(rng.random(n) < pplus, 1.0, -1.0) + 0.08 * rng.standard_normal(n)
        return h, a, pplus, o

    Astack = np.stack(A)                                  # (2p+1, p+1, p+1)
    # warm init: c(h) == [1,0,..] (s==1 => monotone), b(h) small.
    C = np.zeros((p + 1, H)); C[0, 0] = 1.0
    beta = np.zeros(H)
    opt = Adam([C.shape, beta.shape], lr=3e-2)
    for it in range(8000):
        h, a_star, _, _ = batch(256)
        c = h @ C.T                                       # (n, p+1)  affine coeffs
        b = h @ beta                                      # (n,)
        tau = rng.random(256)
        w = q_coeffs_batch(c, Astack)                     # (n, 2p+1)
        powers = np.stack([tau ** (m + 1) for m in range(2 * p + 1)], -1)
        Q = b + np.sum(powers * w, -1)
        u = a_star - Q
        dLdQ = -(tau - (u < 0).astype(float))
        dQ = dQ_dc_batch(c, Astack, tau)                  # (n, p+1)
        gC = (dLdQ[:, None, None] * (dQ[:, :, None] * h[:, None, :])).mean(0)
        gbeta = (dLdQ[:, None] * h).mean(0)
        [C, beta] = opt.step([C, beta], [gC, gbeta])

    # evaluate per obs
    for o in (0, 1):
        h = np.zeros((1, H)); h[0, 0] = 1.0; h[0, 1] = o
        c = (h @ C.T)[0]; b = (h @ beta)[0]
        samp = Q_eval(c, b, A, rng.random(20000))
        target_plus = 0.8 if o == 1 else 0.2
        print(f"  obs={o}: sampled frac(+)={np.mean(samp>0):.2f} "
              f"(target {target_plus:.2f})  commit |a|>.5={np.mean(np.abs(samp)>.5):.3f}  "
              f"#modes(s roots)={len(s_roots_in_unit(c))}")
    print("  => the SAME head commits to opposite modes as obs flips: language/obs is")
    print("     load-bearing (grounding), and each conditional is genuinely bimodal.")


# ===========================================================================
# Part 3 -- MULTIVARIATE via the AUTOREGRESSIVE (triangular / Knothe-Rosenblatt)
# quantile map. Target: two diagonal blobs (a1,a2) ~ (+/-1, +/-1) correlated.
#   A per-dim FACTORIZED model would also put mass on the anti-diagonal (wrong).
#   Q_1(tau1) bimodal;  Q_2(tau2 | a1) conditions on a1 (via affine c(a1)) and is
#   UNIMODAL given a1  -> joint mass lands only on the two real blobs.
# ===========================================================================
def part3():
    print()
    print("=" * 74)
    print("PART 3  multivariate: autoregressive triangular quantile map (chunk-ready)")
    print("=" * 74)
    p = 4
    A = sos_integral_matrices(p)

    def batch(n):
        s = np.where(rng.random(n) < 0.5, 1.0, -1.0)
        a1 = s + 0.10 * rng.standard_normal(n)
        a2 = s + 0.10 * rng.standard_normal(n)            # correlated with a1
        return a1, a2

    Astack = np.stack(A)
    # dim 1: unconditional bimodal quantile (pinball).  warm init s==1.
    c1 = np.zeros(p + 1); c1[0] = 1.0; b1 = 0.0
    opt1 = Adam([(p + 1,), ()], lr=5e-2)
    # dim 2: coeffs AFFINE in a1  -> c2(a1)=C2[:,0]+C2[:,1]*a1 ; b2(a1)=beta2.[1,a1]
    C2 = np.zeros((p + 1, 2)); C2[0, 0] = 1.0; beta2 = np.zeros(2)
    opt2 = Adam([C2.shape, beta2.shape], lr=3e-2)

    for it in range(8000):
        a1, a2 = batch(256)
        tau = rng.random(256)
        # dim1
        Q1 = Q_eval(c1, b1, A, tau); u1 = a1 - Q1
        dLdQ = -(tau - (u1 < 0).astype(float))
        gc = (dLdQ[:, None] * dQ_dc(c1, A, tau)).mean(0); gb = dLdQ.mean()
        c1, b1 = opt1.step([c1, b1], [gc, np.array(gb)])
        # dim2 | a1  (teacher-forced on the TRUE a1 -- standard autoregressive training)
        feat = np.stack([np.ones_like(a1), a1], -1)       # [1, a1]
        c2 = feat @ C2.T; b2 = feat @ beta2
        tau2 = rng.random(256)
        w2 = q_coeffs_batch(c2, Astack)
        powers = np.stack([tau2 ** (m + 1) for m in range(2 * p + 1)], -1)
        Q2 = b2 + np.sum(powers * w2, -1); u2 = a2 - Q2
        dLdQ2 = -(tau2 - (u2 < 0).astype(float))
        dQ = dQ_dc_batch(c2, Astack, tau2)
        gC2 = (dLdQ2[:, None, None] * (dQ[:, :, None] * feat[:, None, :])).mean(0)
        gbeta2 = (dLdQ2[:, None] * feat).mean(0)
        [C2, beta2] = opt2.step([C2, beta2], [gC2, gbeta2])

    # ancestral sample: tau1,tau2 ~ U ; a1=Q1(tau1) ; a2=Q2(tau2|a1)
    n = 20000
    a1 = Q_eval(c1, b1, A, rng.random(n))
    feat = np.stack([np.ones_like(a1), a1], -1)
    c2 = feat @ C2.T; b2 = feat @ beta2
    tau2 = rng.random(n)
    w2 = q_coeffs_batch(c2, Astack)
    powers = np.stack([tau2 ** (m + 1) for m in range(2 * p + 1)], -1)
    a2 = b2 + np.sum(powers * w2, -1)
    on_diag = np.mean(np.sign(a1) == np.sign(a2))
    comm = np.abs(a1) > 0.5
    on_diag_c = np.mean(np.sign(a1[comm]) == np.sign(a2[comm]))
    print(f"  sampled joint (a1,a2): corr(a1,a2) = {np.corrcoef(a1,a2)[0,1]:+.3f} "
          f"(target ~+1; a factorized/independent head gives ~0.0)")
    print(f"    P(sign match) = {on_diag:.3f} all,  {on_diag_c:.3f} on committed "
          f"|a1|>0.5  ({comm.mean():.2f} of samples)")
    print(f"    -> mass on the two REAL diagonal blobs; a factorized head would put")
    print(f"       ~half its mass on the spurious anti-diagonal (corr ~0).")
    print(f"    a MEAN/MSE head would predict (0,0): between BOTH blobs (collision).")


if __name__ == "__main__":
    part1()
    part2()
    part3()
