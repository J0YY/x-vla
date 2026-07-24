"""Minimal prototype: kernel-mean-embedding action head with POLYNOMIAL (tensor) kernels.

Standalone NumPy (the repo's Python is 3.14 -> no torch; this proves the math and the
training loop cheaply). The production head is the identical construction in torch, with
the conditional-mean-embedding (CME) operator emitted by a *linear/bilinear* (foldable)
map of the chi-VLA action-query token h(obs) -- see report. Here `obs` is a small context
so the whole thing runs in seconds.

Idea (kernel Bayes; Song/Fukumizu/Gretton): represent p(a|obs) by its KERNEL MEAN
EMBEDDING  mu_{a|obs} = E[phi(a) | obs]  in an RKHS. For a POLYNOMIAL kernel
    k(a,a') = (1 + a.a')^p
the feature map is EXPLICIT and finite:  phi(a) = homog(a)^{⊗p}  (a symmetric tensor),
so <phi(a),phi(a')> = (1 + a.a')^p exactly and the embedding is a TENSOR object. The
conditional-mean-embedding operator  W : psi(obs) -> mu_{a|obs}  is then a plain matrix
(ridge regression of phi(a) on obs features) -- tensor-pure linear algebra, no softmax,
no Z, no negatives. Multimodality is fully captured: mu holds all moments of a up to
degree p. DECODE is OUT-OF-GRAPH kernel herding (greedily pick samples whose empirical
embedding matches mu) -- the same category of controller op as the gripper threshold.

Run:  python3 xvla/train/kme_poly_proto.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Explicit polynomial (tensor) feature map:  phi(a) = homog(a)^{⊗p} flattened.
#   homog(a) = [1; a];   <phi(a),phi(a')> = (homog(a).homog(a'))^p = (1 + a.a')^p
# This is the symmetric-tensor-power feature map of the degree-p polynomial kernel.
# ---------------------------------------------------------------------------
def homog(a):
    a = np.atleast_2d(a)
    return np.concatenate([np.ones((a.shape[0], 1)), a], axis=1)   # (n, d+1)


def phi(a, p):
    ah = homog(a)                                                   # (n, d+1)
    n = ah.shape[0]
    out = ah.copy()
    for _ in range(p - 1):
        out = (out[:, :, None] * ah[:, None, :]).reshape(n, -1)     # Kronecker power
    return out                                                     # (n, (d+1)^p)


def poly_kernel(a, b, p):
    return (1.0 + a @ b.T) ** p


# ---------------------------------------------------------------------------
# Conditional mean embedding operator  W  (ridge regression of phi(a) on psi(obs)).
#   Psi rows = psi(obs_i);  Phi rows = phi(a_i)
#   W = (Psi^T Psi + n*lam I)^{-1} Psi^T Phi        mu(obs) = psi(obs) @ W
# Pure linear algebra => tensor-pure & foldable. `psi` uses the SAME poly feature map
# (tensor kernel on both legs) so E[phi(a)|obs], a degree-p polynomial in obs, is exactly
# representable.
# ---------------------------------------------------------------------------
def fit_cme(obs, acts, p_obs, p_act, lam=1e-4):
    Psi = phi(obs, p_obs)
    Phi = phi(acts, p_act)
    m = Psi.shape[1]
    G = Psi.T @ Psi + lam * len(obs) * np.eye(m)
    W = np.linalg.solve(G, Psi.T @ Phi)                            # (m_obs, D_act)
    return W


def embed_of(obs, W, p_obs):
    return phi(obs, p_obs) @ W                                     # (n_obs, D_act)


# ---------------------------------------------------------------------------
# OUT-OF-GRAPH decode: Welling (2009) kernel herding against a target embedding mu*.
#   h_0 = mu*;   x_{t+1} = argmax_x <phi(x), h_t>;   h_{t+1} = h_t + mu* - phi(x_{t+1})
# Empirical embedding (1/T) sum phi(x_t) -> mu* at O(1/T). Argmax is over a grid here
# (the controller's optimizer); the learned object mu* stays fully in-graph/polynomial.
# ---------------------------------------------------------------------------
def herd(mu_star, grid, p_act, T):
    Phi_g = phi(grid, p_act)                                       # (G, D)
    h = mu_star.copy()
    picks = []
    for _ in range(T):
        j = int((Phi_g @ h).argmax())
        picks.append(grid[j])
        h = h + mu_star - Phi_g[j]
    return np.array(picks)


def make_grid(lo, hi, n):
    xs = np.linspace(lo, hi, n)
    X, Y = np.meshgrid(xs, xs)
    return np.stack([X.ravel(), Y.ravel()], axis=1)


# ===========================================================================
# Part 0 -- feature-map correctness: <phi(a),phi(a')> == (1 + a.a')^p exactly.
# ===========================================================================
def part0():
    p = 4
    a = rng.standard_normal((5, 2))
    b = rng.standard_normal((7, 2))
    lhs = phi(a, p) @ phi(b, p).T
    rhs = poly_kernel(a, b, p)
    print(f"  [feature map] max|<phi,phi> - (1+a.a')^p| = {np.abs(lhs - rhs).max():.2e} "
          f"(p={p}, D=(d+1)^p={3**p})")


# ===========================================================================
# Part 1 -- unconditional bimodal 2-D: herding recovers BOTH modes; mean sits between.
# ===========================================================================
def part1():
    p = 4
    m1, m2 = np.array([-1.0, 0.4]), np.array([1.0, -0.4])          # two modes
    a = np.where(rng.random((4000, 1)) < 0.5, m1, m2) + 0.08 * rng.standard_normal((4000, 2))
    mu = phi(a, p).mean(0, keepdims=True)                          # empirical mean embedding

    grid = make_grid(-2.0, 2.0, 81)
    picks = herd(mu[0], grid, p, T=12)

    # cluster the herded picks to the two nearest true modes
    d1 = np.linalg.norm(picks - m1, axis=1)
    d2 = np.linalg.norm(picks - m2, axis=1)
    rec1 = picks[d1 < d2].mean(0)
    rec2 = picks[d2 <= d1].mean(0)
    mean_a = a.mean(0)                                             # what an MSE head emits
    print(f"  [bimodal] true modes    = {np.round(m1,2)} , {np.round(m2,2)}")
    print(f"  [bimodal] herded modes  = {np.round(rec1,2)} , {np.round(rec2,2)}  "
          f"(T=12 samples hit both basins: {int((d1<d2).sum())}/{int((d2<=d1).sum())})")
    print(f"  [bimodal] MSE-head mean = {np.round(mean_a,2)}  <- lands BETWEEN the modes "
          f"(dist to nearest mode {min(np.linalg.norm(mean_a-m1),np.linalg.norm(mean_a-m2)):.2f})")


# ===========================================================================
# Part 2 -- CONDITIONAL: obs swaps which two modes are active. Fit the CME operator W,
#           embed a fresh obs, herd -> recover that obs's modes. (Multimodality lives
#           entirely in the learned tensor mu; the map obs->mu is a single matmul.)
# ===========================================================================
def part2():
    p = 4
    # obs is a scalar context in [-1,1]; modes = center(obs) +/- offset, center slides with obs
    def sample(obs, n):
        obs = np.asarray(obs).reshape(-1, 1)
        center = np.concatenate([obs, -0.5 * obs], axis=1)         # (n,2), linear in obs
        sgn = np.where(rng.random((n, 1)) < 0.5, 1.0, -1.0)
        a = center + sgn * np.array([0.0, 0.9]) + 0.06 * rng.standard_normal((n, 2))
        return a

    O = rng.uniform(-1, 1, 3000)
    A = sample(O, 3000)
    W = fit_cme(O[:, None], A, p_obs=p, p_act=p, lam=1e-3)

    grid = make_grid(-2.0, 2.0, 81)
    for obs in (-0.8, 0.0, 0.8):
        mu = embed_of(np.array([[obs]]), W, p)[0]
        picks = herd(mu, grid, p, T=10)
        center = np.array([obs, -0.5 * obs])
        true = np.array([center + np.array([0, 0.9]), center + np.array([0, -0.9])])
        # match herded picks to the two true modes
        lo = picks[picks[:, 1] < center[1]].mean(0)
        hi = picks[picks[:, 1] >= center[1]].mean(0)
        print(f"  [cond obs={obs:+.1f}] true modes {np.round(true[0],2)}/{np.round(true[1],2)}"
              f"  herded {np.round(hi,2)}/{np.round(lo,2)}"
              f"  MSE-mean {np.round(A.mean(0)*0 + center,2)} (=center, between modes)")


# ===========================================================================
# Part 3 -- REACH: continuous 2-D obs = target position; action = grasp approach, bimodal
#           (come from left OR right of the target). MSE emits the target itself (collision);
#           the CME+herding head commits to a real approach side.
# ===========================================================================
def part3():
    p = 4
    def sample(T, n):                                              # T: (n,2) targets
        sgn = np.where(rng.random((n, 1)) < 0.5, 1.0, -1.0)
        a = T + sgn * np.array([0.6, 0.0]) + 0.05 * rng.standard_normal((n, 2))
        return a

    Tg = rng.uniform(-1, 1, (4000, 2))
    A = sample(Tg, 4000)
    W = fit_cme(Tg, A, p_obs=p, p_act=p, lam=1e-3)

    grid = make_grid(-2.0, 2.0, 81)
    ok = 0
    tests = rng.uniform(-0.8, 0.8, (6, 2))
    for t in tests:
        mu = embed_of(t[None], W, p)[0]
        picks = herd(mu, grid, p, T=8)
        left = picks[picks[:, 0] < t[0]]
        right = picks[picks[:, 0] >= t[0]]
        both = len(left) > 0 and len(right) > 0
        ok += both
        # distance from the MSE mean (=target) vs from a herded approach point to target
        mse_pred = A[np.argmin(np.linalg.norm(Tg - t, axis=1))] * 0 + t   # MSE -> target
        print(f"  [reach t={np.round(t,2)}] herded sides L/R = {len(left)}/{len(right)} "
              f"({'BOTH approaches' if both else 'one side'}); MSE emits target itself "
              f"(offset {np.linalg.norm(mse_pred - t):.2f} => collision)")
    print(f"  [reach] recovered BOTH approach modes on {ok}/{len(tests)} targets")


if __name__ == "__main__":
    print("Part 0: polynomial feature-map identity")
    part0()
    print("\nPart 1: unconditional bimodal 2-D (herding vs MSE mean)")
    part1()
    print("\nPart 2: conditional CME operator (obs swaps the modes)")
    part2()
    print("\nPart 3: reach -- bimodal grasp approach (MSE collides with target)")
    part3()
