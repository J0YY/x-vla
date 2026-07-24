"""Minimal prototype: polynomial energy-based (implicit) action head.

Standalone NumPy (the repo's Python is 3.14 -> no torch; this proves the math and
the training loop cheaply). The production head is the identical construction in
torch, with the energy's coefficients emitted by a *linear/bilinear* (foldable)
map of the chi-VLA action-query token h(obs) -- see report. Here `obs` is a small
context so the whole thing runs in seconds.

Idea (implicit BC, cf. Florence et al. 2021): learn a SCALAR energy E(obs, a)
that is a LOW-DEGREE POLYNOMIAL in the action a (a is an input leg). The action
distribution is p(a|obs) ~ exp(-E), so

    * multimodality  = MULTIPLE MINIMA of the polynomial energy in a,
    * decode         = argmin_a E(obs, a)   (OUT-OF-GRAPH controller op).

The learned object (the polynomial energy) stays fully tensor-pure and
ODT-analyzable; only the argmin at inference leaves the graph -- exactly the
gripper-threshold reframe. A quartic (degree 4) in a scalar has 2 minima; degree
2K gives K minima. The bimodal gripper {-1,+1} is the minimal (quartic) case.

Two training objectives are demonstrated, both keeping E polynomial:
  * InfoNCE / contrastive: softmax over candidate actions (softmax in the LOSS
    ONLY, argmin at inference) -- the same trick as a BCE gripper head.
  * Exact score matching (Hyvarinen): J = E[ 1/2 E'(a)^2 + E''(a) ]. No negatives,
    no partition function; E' and E'' are themselves polynomials => fully in-graph.

Run:  python3 xvla/train/energy_ebm_proto.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


# ===========================================================================
# Part 1 -- fit a bimodal 1-D target with a free quartic energy E(a)=theta.f(a).
#           f(a) = [a, a^2, a^3, a^4]  (the degree-0 term cancels in exp(-E)/Z).
#           Two objectives: (A) InfoNCE over a grid, (B) exact score matching.
# ===========================================================================
def feats(a):                      # monomials    a, a^2, a^3, a^4
    return np.stack([a, a**2, a**3, a**4], -1)


def dfeats(a):                     # d/da         1, 2a, 3a^2, 4a^3
    return np.stack([np.ones_like(a), 2 * a, 3 * a**2, 4 * a**3], -1)


def ddfeats(a):                    # d2/da2       0, 2, 6a, 12a^2
    return np.stack([np.zeros_like(a), 2 * np.ones_like(a), 6 * a, 12 * a**2], -1)


def energy(theta, a):
    return feats(a) @ theta


def bimodal_samples(n):            # target: 1/2 N(-1,.15) + 1/2 N(+1,.15)
    mu = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    return mu + 0.15 * rng.standard_normal(n)


def part1():
    grid = np.linspace(-2.0, 2.0, 401)          # candidate actions for InfoNCE
    Fg = feats(grid)

    # --- (A) InfoNCE / contrastive: fit categorical over the grid ------------
    #     dNLL/dtheta = E_data[f] - E_model[f]   (moment matching)
    theta = np.zeros(4)
    for it in range(4000):
        a = bimodal_samples(512)
        logits = -(Fg @ theta)                  # -E over the grid
        p = np.exp(logits - logits.max()); p /= p.sum()
        g = feats(a).mean(0) - p @ Fg           # data moments - model moments
        theta -= 0.05 * g
        if it % 1000 == 0:
            E = Fg @ theta
            nll = (feats(a) @ theta).mean() + (np.log(np.exp(-(E - E.min())).sum())
                                               - E.min())
            print(f"  [part1/InfoNCE] it={it:4d}  nll={nll:.3f}")
    _report_quartic("InfoNCE", theta, grid)

    # --- (B) exact score matching: minimize 1/2 theta^T A theta + b^T theta ---
    #     A = E[f' f'^T], b = E[f''].  No negatives, no partition function; E', E''
    #     are polynomials => fully in-graph.
    a = bimodal_samples(20000)
    Fp, Fpp = dfeats(a), ddfeats(a)
    A = Fp.T @ Fp / len(a)                       # E[f' f'^T]
    b = Fpp.mean(0)                              # E[f'']
    # unconstrained closed form -- HONEST caveat: can pick a non-coercive (a^4<0)
    # inverted well (a known EBM/score-matching pathology), so decode is ill-posed.
    theta_free = -np.linalg.solve(A + 1e-6 * np.eye(4), b)
    _report_quartic("scoreMatch/free", theta_free, grid)
    # coercive-constrained: FIX the leading coeff theta_4 = t4 > 0, solve the rest.
    # This is the tensor-pure fix (a^4>0 guaranteed, e.g. via a squared readout).
    t4 = 4.0
    theta_r = -np.linalg.solve(A[:3, :3] + 1e-6 * np.eye(3),
                               A[:3, 3] * t4 + b[:3])
    theta_sm = np.concatenate([theta_r, [t4]])
    _report_quartic("scoreMatch/coercive", theta_sm, grid)


def _report_quartic(tag, theta, grid):
    E = energy(theta, grid)
    # local minima of E on the grid (out-of-graph decode = argmin / all minima)
    interior = np.arange(1, len(grid) - 1)
    is_min = (E[interior] < E[interior - 1]) & (E[interior] < E[interior + 1])
    minima = grid[interior][is_min]
    # the mean of p ~ exp(-E) (what an MSE head would emit) vs the modes
    p = np.exp(-(E - E.min())); p /= p.sum()
    mean_a = p @ grid
    coercive = theta[3] > 0                      # positive leading (a^4) coeff
    print(f"  [part1/{tag}] theta={np.round(theta, 3)}  coercive(a^4>0)={coercive}")
    print(f"  [part1/{tag}] energy minima (modes) = {np.round(np.sort(minima), 2)}"
          f"   E[a] under p~exp(-E) = {mean_a:+.3f}  (MSE head would emit this)")


# ===========================================================================
# Part 2 -- the gripper: conditional quartic double-well energy.
#   E(g; obs) = alpha (g^2 - 1)^2 + tau(obs) * g ,   alpha>0,  tau(obs)=w.[1;obs]
#   minima pinned at g=+-1 (barrier at 0); the obs-dependent tilt tau picks the
#   deeper well. Decode = argmin_g E over a grid => COMMITS to +-1, where an MSE
#   head averages the bimodal target to ~0.
# ===========================================================================
def part2():
    grid = np.linspace(-1.5, 1.5, 61)                # includes +-1
    log_alpha = np.log(2.0)                           # alpha = exp(log_alpha) > 0
    w_tau = np.zeros(2)                               # tau(obs) = w.[1; obs]

    def E_of(g, obs, la, w):
        alpha = np.exp(la)
        tau = np.stack([np.ones_like(obs), obs], -1) @ w   # (B,)
        return alpha * (g**2 - 1.0)**2 + tau[..., None] * g  # broadcast over grid

    for it in range(4000):
        obs = rng.uniform(-1, 1, 512)
        y = np.where(obs < 0, 1.0, -1.0)             # obs<0 -> close(+1); >0 -> open(-1)
        amb = np.abs(obs) < 0.2                       # genuinely bimodal middle
        y[amb] = np.where(rng.random(amb.sum()) < 0.5, -1.0, 1.0)

        alpha = np.exp(log_alpha)
        phi = np.stack([np.ones_like(obs), obs], -1)  # (B,2)
        Eg = E_of(grid[None, :], obs, log_alpha, w_tau)   # (B, G)
        p = np.exp(-(Eg - Eg.min(1, keepdims=True)))
        p /= p.sum(1, keepdims=True)                  # softmax over grid (LOSS only)

        # features whose (data - model) expectation is the gradient:
        #   d E/d log_alpha = alpha (g^2-1)^2 ;   d E/d w = g * phi
        fa_data = alpha * (y**2 - 1.0)**2
        fa_grid = alpha * (grid**2 - 1.0)**2
        g_la = (fa_data - (p * fa_grid[None, :]).sum(1)).mean()

        gw_data = y[:, None] * phi
        gw_grid = ((p * grid[None, :]).sum(1))[:, None] * phi   # E_model[g] * phi
        g_w = (gw_data - gw_grid).mean(0)

        log_alpha -= 0.02 * g_la
        w_tau -= 0.1 * g_w

    alpha = np.exp(log_alpha)
    print(f"  [gripper] learned alpha={alpha:.3f}  w_tau={np.round(w_tau, 3)} "
          f"(tau=w.[1;obs]; barrier at g=0 height=alpha)")
    for obs_val in (-0.8, -0.05, 0.05, 0.8):
        Eg = E_of(grid[None, :], np.array([obs_val]), log_alpha, w_tau)[0]
        g_star = grid[Eg.argmin()]                    # OUT-OF-GRAPH argmin decode
        print(f"  [gripper] obs={obs_val:+.2f}  argmin_g E = {g_star:+.2f} "
              f"-> decode={'+1(close)' if g_star > 0 else '-1(open)'}")
    print(f"  [gripper] MSE head on ambiguous region emits ~mean(+-1)=~0.0 "
          f"(indecisive); the energy commits to a well by the sign of tau.")


if __name__ == "__main__":
    print("Part 1: bimodal 1-D target as a quartic energy (InfoNCE + score matching)")
    part1()
    print("\nPart 2: gripper as a conditional double-well energy (argmin commits to +-1)")
    part2()
