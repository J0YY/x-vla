"""Minimal prototype: TENSOR-PURE flow-matching / diffusion action head.

Standalone NumPy (the repo's Python is 3.14 -> no torch; this proves the math and
the training loop cheaply, exactly like ``energy_ebm_proto.py`` / ``born_mps_proto.py``).
The production head is the identical construction in torch, with the velocity's
coefficients realised by a *bilinear/CP* (foldable) core stack -- see the report.

THESIS (the purity reframe).  Only the LEARNED OBJECT must be in-graph & polynomial.
Here the learned object is the flow-matching VELOCITY field

    v_theta(a, t; obs)  ~=  E[a1 - a0 | a_t = a, t, obs]

parametrised as a POLYNOMIAL (a CP/bilinear tensor network) in the three legs
a (noisy action), t (diffusion time) and obs (context).  It is degree-p in a,
foldable, and ODT-analysable at every (t, obs): its Jacobian d v/d a is the
per-(t,obs) analogue of the energy Hessian / the Q_c object odt_interp.py already
diagonalises.

The DECODE is the reverse ODE integration  da/dt = v_theta(a, t; obs),  a0 ~ N(0,I),
t: 0 -> 1  -- an ITERATIVE, OUT-OF-GRAPH controller op, exactly the sanctioned
decode boundary (spec section 11 "no internal sampling"; the gripper threshold; the
phase argmax in vla.py; the multi-hypothesis decode argmax).  Each step calls the
pure tensor network v_theta once; the loop (the +, the schedule) lives in the
controller.

WHAT THIS PROTO SHOWS
  Part 1 -- unconditional bimodal 2-D target.
     * a degree-1-in-a (affine) velocity field can only produce ONE Gaussian blob
       (its flow is affine) -- it FAILS to be multimodal;
     * a degree-3-in-a polynomial velocity field, integrated, PUSHES the Gaussian
       noise onto BOTH modes -- multimodal, while its per-sample MEAN (what an MSE
       head emits) sits uselessly between the modes.
  Part 2 -- obs-conditioned "commit" (the gripper analogue): obs tilts which mode is
     favoured; the sampler COMMITS to the correct mode per obs, where an MSE head
     would average.

Training is pure MSE regression on a supervised target (a1 - a0) -- NO partition
function, NO negatives, NO in-graph normalisation (unlike EBM/InfoNCE or the Born
machine's Z = <psi|psi>).  The only non-polynomial ingredients -- the Gaussian
noise a0 and the time t ~ U[0,1] -- are out-of-loop DATA, never in the graph.

Run:  python3 xvla/train/diffusion_proto.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


# ===========================================================================
# Polynomial (CP-representable) velocity feature map.
#
#   phi(a, t, obs) = { a1^i a2^j * t^k * obs^m : i+j <= deg_a, k <= 2, m <= m_obs }
#
# v_theta = W @ phi  is LINEAR in the parameters W (so flow-matching is a convex
# least-squares fit here) and POLYNOMIAL of degree deg_a in the action a.  Each
# monomial a1^i a2^j is exactly what a stack of the repo's degree-2 BilinearFFN
# cores emits from the homogeneous input [1; a; t; obs] (two cores -> degree 4,
# etc.); the explicit monomial basis is used here only for a transparent proto.
# ===========================================================================
def make_feats(deg_a: int, m_obs: int):
    # enumerate monomial exponents (i, j) in the two action coords, i+j <= deg_a
    aexp = [(i, j) for i in range(deg_a + 1) for j in range(deg_a + 1 - i)]

    def phi(a, t, obs):
        a1, a2 = a[..., 0], a[..., 1]
        cols = []
        for i, j in aexp:
            base = (a1 ** i) * (a2 ** j)
            for k in range(3):                    # t^0, t^1, t^2
                for m in range(m_obs + 1):        # obs^0 .. obs^m_obs
                    cols.append(base * (t ** k) * (obs ** m))
        return np.stack(cols, -1)                 # (..., F)

    return phi, len(aexp) * 3 * (m_obs + 1)


def fit_flow_matching(target_sampler, phi, F, iters=6000, bs=4096):
    """Rectified-flow least squares:  min_W E || W phi(a_t,t,obs) - (a1 - a0) ||^2.

    a1 ~ data, a0 ~ N(0,I), a_t = (1-t) a0 + t a1, target u = a1 - a0.  Convex in W
    -> accumulate the normal equations  W = (E[u phi^T]) (E[phi phi^T])^{-1}.
    """
    d = 2
    Spp = np.zeros((F, F))
    Sup = np.zeros((d, F))
    for _ in range(iters // 8):
        a1, obs = target_sampler(bs)              # (bs,2), (bs,)
        a0 = rng.standard_normal((bs, d))
        t = rng.uniform(0.0, 1.0, bs)
        a_t = (1.0 - t)[:, None] * a0 + t[:, None] * a1
        u = a1 - a0                               # supervised velocity target
        P = phi(a_t, t, obs)                      # (bs, F)
        Spp += P.T @ P
        Sup += u.T @ P
    W = np.linalg.solve(Spp + 1e-4 * np.eye(F), Sup.T).T   # (d, F)
    return W


def sample_ode(W, phi, obs, n, steps=50):
    """OUT-OF-GRAPH decode: Euler-integrate da/dt = W phi(a,t,obs), a0 ~ N, t:0->1.

    The loop / the '+' / the schedule are the controller op; each step is ONE call
    to the pure polynomial field  a -> W phi(a, t, obs).
    """
    a = rng.standard_normal((n, 2))
    obs_v = np.full(n, obs, dtype=float)
    dt = 1.0 / steps
    for s in range(steps):
        t = np.full(n, s * dt)
        a = a + dt * (phi(a, t, obs_v) @ W.T)
    return a


# ===========================================================================
# Part 1 -- unconditional bimodal 2-D target; degree-1 (affine) vs degree-3 field.
# ===========================================================================
MODES = np.array([[-1.5, -1.5], [1.5, 1.5]])


def uncond_target(n):
    k = rng.integers(0, 2, n)
    a1 = MODES[k] + 0.20 * rng.standard_normal((n, 2))
    return a1, np.zeros(n)                         # obs unused (obs=0)


def _mode_coverage(samples):
    d = ((samples[:, None, :] - MODES[None]) ** 2).sum(-1)   # (n, 2)
    nearest = d.argmin(1)
    within = d.min(1) < 0.5 ** 2                              # within 0.5 of a mode
    frac0 = ((nearest == 0) & within).mean()
    frac1 = ((nearest == 1) & within).mean()
    return frac0, frac1, within.mean()


def part1():
    print("Part 1: unconditional bimodal 2-D  (modes at (-1.5,-1.5) and (+1.5,+1.5))")
    for deg_a, tag in [(1, "affine  (deg-1 in a)"), (3, "polynomial (deg-3 in a)")]:
        phi, F = make_feats(deg_a, m_obs=0)
        W = fit_flow_matching(uncond_target, phi, F)
        s = sample_ode(W, phi, obs=0.0, n=6000)
        f0, f1, cov = _mode_coverage(s)
        mean = s.mean(0)
        print(f"  [{tag}]  F={F:3d}  mode0={f0:.2f} mode1={f1:.2f} "
              f"(covered={cov:.2f})  sample-mean={np.round(mean,2)}  "
              f"(an MSE head emits this mean)")
    print("  => affine field cannot be bimodal (its flow is affine -> one blob);")
    print("     the deg-3 polynomial field integrates to BOTH modes; the mean is")
    print("     stuck between them (the exact MSE-collapse failure we escape).")


# ===========================================================================
# Part 2 -- obs-conditioned commit (the gripper analogue): obs tilts the mode mix.
# ===========================================================================
def cond_target(n):
    obs = rng.uniform(-1, 1, n)
    p_hi = np.where(obs > 0, 0.85, 0.15)           # obs>0 favours mode1, else mode0
    k = (rng.random(n) < p_hi).astype(int)
    a1 = MODES[k] + 0.20 * rng.standard_normal((n, 2))
    return a1, obs


def part2():
    print("\nPart 2: obs-conditioned commit  (obs>0 -> +mode, obs<0 -> -mode)")
    phi, F = make_feats(deg_a=3, m_obs=2)
    W = fit_flow_matching(cond_target, phi, F)
    for obs in (-0.8, +0.8):
        s = sample_ode(W, phi, obs=obs, n=6000)
        f0, f1, cov = _mode_coverage(s)
        favoured = "mode1(+)" if obs > 0 else "mode0(-)"
        print(f"  obs={obs:+.1f}: mode0={f0:.2f} mode1={f1:.2f} "
              f"-> commits to {favoured}   (sample-mean={np.round(s.mean(0),2)})")
    print("  => the sampler COMMITS to the obs-appropriate mode; an MSE head would")
    print("     emit the between-modes mean regardless of obs.")


if __name__ == "__main__":
    part1()
    part2()
