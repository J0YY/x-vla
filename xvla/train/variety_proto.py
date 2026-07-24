"""Minimal prototype: algebraic-VARIETY (support-based) action head.

Standalone NumPy (the repo's Python is 3.14 -> no torch; this proves the math and
the fitting procedure cheaply). The production head is the identical construction
in torch, with the constraint polynomials' coefficients emitted by a *linear/
bilinear* (foldable) map of the chi-VLA action-query token h(obs) -- see report.

IDEA (distinct from density / energy / candidate / pushforward heads).
Do NOT model a density p(a|obs). Model the SUPPORT: the set of valid actions given
obs lies (near) a low-dimensional ALGEBRAIC VARIETY, the common zero-set

    V(obs) = { a :  f_j(obs, a) = 0  for all j }

of learned polynomials f_j that are tensor-pure (each f_j is a quadratic form
z^T Q_j z in the homogeneous action z = [1; a], i.e. exactly a folded symmetric
BilinearFFN core -- the SAME Q object odt_interp.py already diagonalizes).

Multimodality = the variety has MULTIPLE BRANCHES. A polynomial system naturally
has several solution branches: a quadratic in one action coordinate has 2 roots
=> 2 modes (two grasp poses); a conic a^2+b^2=r^2 is a 1-D manifold of directions.
We never AVERAGE (the MSE pathology): decode SNAPS a seed onto a branch by
out-of-graph Newton / root-finding. Branch selection is an out-of-graph choice.

KEY fitting fact (the tie to ODT): the polynomials that VANISH on a set of demos
are the null space of the second-moment matrix of the monomial (Veronese) features
  M = E_demo[ phi(a) phi(a)^T ],   phi(a) = degree<=2 monomials of [1;a].
So fitting the variety = an EIGENDECOMPOSITION of M and taking the SMALL-eigenvalue
eigenvectors as f_j.  Fitting is itself ODT; nondegeneracy = discard the ~0-noise
directions / normalize ||Q||=1 so f=0 is non-trivial.

Run:  python3 xvla/train/variety_proto.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


# ===========================================================================
# monomial (Veronese) lift of a 2-D action a=(a,b): degree<=2 in z=[1,a,b].
#   phi = [1, a, b, a^2, a*b, b^2]                         (6 monomials)
# Any quadratic constraint is theta.phi = z^T Q z with Q the symmetric 3x3
# core -- exactly BilinearFFN.dense_core's object.
# ===========================================================================
def phi2(A):                       # A: (n,2) -> (n,6)
    a, b = A[:, 0], A[:, 1]
    return np.stack([np.ones_like(a), a, b, a * a, a * b, b * b], -1)


def theta_to_Q(theta):             # 6-vector -> symmetric 3x3 (z=[1,a,b])
    c, ca, cb, caa, cab, cbb = theta
    return np.array([[c,      ca / 2, cb / 2],
                     [ca / 2, caa,    cab / 2],
                     [cb / 2, cab / 2, cbb]])


def eval_Q(Q, A):                  # z^T Q z for a batch, z=[1,a,b]
    z = np.concatenate([np.ones((A.shape[0], 1)), A], 1)
    return np.einsum("ni,ij,nj->n", z, Q, z)


def grad_Q(Q, A):                  # d/d(a,b) of z^T Q z = 2 (Q z)[1:]
    z = np.concatenate([np.ones((A.shape[0], 1)), A], 1)
    return 2.0 * (z @ Q.T)[:, 1:]  # (n,2) normal directions


# ===========================================================================
# Part A -- fit an unconditional conic variety from TWO ARCS, by ODT.
#   Demos lie on a^2 + b^2 = r^2 but ONLY on two opposite arcs (bimodal support).
#   Recover f = a^2 + b^2 - r^2 as the bottom eigenvector of the monomial moment
#   matrix M.  Then DECODE = Newton-project a seed onto f=0 (snaps to a branch);
#   show the MSE mean lands OFF the manifold, the projection lands ON it.
# ===========================================================================
def part_A():
    r = 0.8
    n = 4000
    # two arcs: theta in [20,70] deg (upper-right) and [200,250] deg (lower-left)
    t = np.where(rng.random(n) < 0.5,
                 np.deg2rad(rng.uniform(20, 70, n)),
                 np.deg2rad(rng.uniform(200, 250, n)))
    A = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    A += 0.01 * rng.standard_normal(A.shape)             # tiny demo noise

    # --- FIT = ODT: bottom eigenvector of M = E[phi phi^T] is the vanishing poly.
    P = phi2(A)
    M = P.T @ P / n
    evals, evecs = np.linalg.eigh(M)                     # ascending
    theta = evecs[:, 0]                                  # smallest eigenvalue dir
    theta = theta / np.linalg.norm(theta)               # nondegeneracy: ||.||=1
    # sign/scale for readability (make the a^2 coeff ~ +1)
    theta = theta / theta[3]
    Q = theta_to_Q(theta)
    print(f"  [A] fitted f(a,b) = {theta[3]:+.2f} a^2 {theta[5]:+.2f} b^2 "
          f"{theta[4]:+.2f} ab {theta[1]:+.2f} a {theta[2]:+.2f} b {theta[0]:+.2f}")
    print(f"      (true circle: a^2 + b^2 - r^2, r^2={r**2:.3f})   "
          f"smallest 2 eigs of M = {np.round(evals[:2], 5)}")
    resid = np.sqrt((eval_Q(Q, A) ** 2).mean())
    print(f"      RMS |f| on demos = {resid:.4e}  (=> demos lie on the zero-set)")

    # --- MSE baseline: the conditional mean of the (bimodal) demo set.
    mse_mean = A.mean(0)
    print(f"  [A] MSE mean action = {np.round(mse_mean, 3)}  "
          f"|f(mean)| = {abs(eval_Q(Q, mse_mean[None]))[0]:.3f}  "
          f"(OFF the manifold: the averaging pathology)")

    # --- DECODE = out-of-graph Newton projection onto f=0 from a seed.
    def newton_project(a0, iters=30):
        a = a0.astype(float).copy()
        for _ in range(iters):
            f = eval_Q(Q, a[None])[0]
            g = grad_Q(Q, a[None])[0]
            a = a - f * g / (g @ g + 1e-12)             # Gauss-Newton step
        return a

    # branch selection = seed side (out-of-graph choice). Same obs, two seeds.
    for seed, tag in [(np.array([0.6, 0.6]), "upper-right seed"),
                      (np.array([-0.6, -0.6]), "lower-left seed")]:
        a_star = newton_project(seed)
        print(f"  [A] {tag}: decode -> {np.round(a_star, 3)}  "
              f"|f|={abs(eval_Q(Q, a_star[None]))[0]:.2e}  "
              f"radius={np.linalg.norm(a_star):.3f} (on-manifold, r={r})")

    # --- ODT of the constraint JACOBIAN = tangent/normal geometry (readable).
    a_star = newton_project(np.array([0.6, 0.6]))
    J = grad_Q(Q, a_star[None])                          # (1,2) here: normal row
    U, S, Vt = np.linalg.svd(J)
    normal = Vt[0]                                        # top singular dir = normal
    tangent = Vt[-1]                                      # null dir = tangent
    # true tangent to the circle at a_star is perpendicular to the radius
    true_tan = np.array([-a_star[1], a_star[0]])
    true_tan /= np.linalg.norm(true_tan)
    align = abs(tangent @ true_tan)
    print(f"  [A] Jacobian ODT at {np.round(a_star,2)}: normal={np.round(normal,2)} "
          f"tangent={np.round(tangent,2)}  |<tangent,true_tangent>|={align:.4f}")


# ===========================================================================
# Part B -- CONDITIONAL 2-branch variety = the bimodal gripper (the 0% blocker).
#   f(g; obs) = g^2 + beta(obs) g + gamma(obs),  roots = the two demonstrated
#   modes.  Coeffs (beta,gamma) are LINEAR in obs features (bilinear obs(x)action
#   monomials => tensor-pure/foldable).  Decode = solve the quadratic (2 roots,
#   out-of-graph), pick the branch tilted by obs.  Commits to +-1 where MSE->0.
# ===========================================================================
def part_B():
    # data: gripper g in {-1,+1}; obs<0 -> close(+1), obs>0 -> open(-1),
    #       |obs|<0.2 genuinely bimodal.  Two modes always present at +-1.
    def draw(n):
        obs = rng.uniform(-1, 1, n)
        y = np.where(obs < 0, 1.0, -1.0)
        amb = np.abs(obs) < 0.2
        y[amb] = np.where(rng.random(amb.sum()) < 0.5, -1.0, 1.0)
        return obs, y

    # Fit beta(obs)=w_b.[1,obs], gamma(obs)=w_g.[1,obs] by regressing
    # f(y;obs) = y^2 + beta y + gamma  ->  0 on demos  (least squares; y^2 known).
    obs, y = draw(20000)
    ctx = np.stack([np.ones_like(obs), obs], 1)          # (n,2)
    # design: [beta-part: y*ctx | gamma-part: ctx],  target = -y^2
    X = np.concatenate([y[:, None] * ctx, ctx], 1)       # (n,4)
    coef, *_ = np.linalg.lstsq(X, -(y ** 2), rcond=None)
    w_b, w_g = coef[:2], coef[2:]
    print(f"  [B] fitted beta(obs)=w_b.[1,obs], w_b={np.round(w_b,3)}; "
          f"gamma(obs), w_g={np.round(w_g,3)}")

    def roots(obs_val):
        c = np.array([1.0, obs_val])
        beta, gamma = w_b @ c, w_g @ c
        disc = beta ** 2 - 4 * gamma
        if disc < 0:                                     # complex -> project real part
            return np.array([-beta / 2, -beta / 2]), disc
        s = np.sqrt(disc)
        return np.array([(-beta - s) / 2, (-beta + s) / 2]), disc

    for obs_val in (-0.8, -0.05, 0.05, 0.8):
        rts, disc = roots(obs_val)
        # branch select = obs-tilt: choose the root the demo distribution prefers.
        pick = rts[np.argmin(np.abs(rts - (1.0 if obs_val < 0 else -1.0)))]
        print(f"  [B] obs={obs_val:+.2f}  roots(2 branches)={np.round(rts,3)}  "
              f"decode={pick:+.2f} ({'close' if pick>0 else 'open'})")
    # MSE head on the ambiguous region -> the mean of +-1 ~ 0 (indecisive)
    o, yv = draw(5000)
    amb = np.abs(o) < 0.2
    print(f"  [B] MSE head on |obs|<0.2 emits ~mean={yv[amb].mean():+.3f} "
          f"(indecisive); variety keeps BOTH roots +-1 and snaps to one.")


# ===========================================================================
# Part C -- foldability check: f = z^T Q z is EXACTLY a CP/dense-core einsum
#   (the BilinearFFN.dense_core object), reconstructed to machine precision.
# ===========================================================================
def part_C():
    Q = rng.standard_normal((3, 3)); Q = 0.5 * (Q + Q.T)   # symmetric core
    A = rng.standard_normal((100, 2))
    z = np.concatenate([np.ones((100, 1)), A], 1)
    direct = np.einsum("ni,ij,nj->n", z, Q, z)
    # CP form: Q = sum_r d_r (l_r outer r_r) via eigdecomp (rank<=3), then
    # f = sum_r d_r (l_r.z)(r_r.z) -- the exact BilinearFFN forward.
    w, V = np.linalg.eigh(Q)
    cp = sum(w[r] * (z @ V[:, r]) * (z @ V[:, r]) for r in range(3))
    print(f"  [C] max|direct - CP/dense-core reconstruction| = "
          f"{np.abs(direct - cp).max():.2e}  (exact fold; degree-2 tensor-pure)")


if __name__ == "__main__":
    print("Part A: unconditional conic variety from 2 arcs (fit=ODT, Newton decode)")
    part_A()
    print("\nPart B: conditional 2-branch gripper variety (quadratic -> 2 roots)")
    part_B()
    print("\nPart C: foldability -- f=z^T Q z is exactly a CP/dense bilinear core")
    part_C()
