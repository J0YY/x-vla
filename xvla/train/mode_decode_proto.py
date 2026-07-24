"""Minimal prototype: MODE / MAP decode for tensor-pure distributional heads.

Standalone NumPy (the repo's Python is 3.14 -> no torch; this proves the math and
the decode strategies cheaply, exactly like ``diffusion_proto.py`` /
``energy_ebm_proto.py`` / ``knothe_transport_proto.py``).

WHY THIS FILE EXISTS (DEVLOG 2026-07-18 cont.17).  After the observation 180-mirror
fix, the LINEAR+MSE head hits 25% closed-loop on LIBERO-Object; the tensor-pure
flow-matching head hits 0%.  Root cause suspected here: the flow head DECODES A
STOCHASTIC SINGLE ODE SAMPLE from Gaussian noise (``FlowMatchingActionHead.sample``
/ ``decode``: ``a0 ~ N(0,I)``, one Euler integration).  On a task whose conditional
p(a | obs) is NEAR-UNIMODAL in most states, a random draw from a slightly
miscalibrated learned distribution is WORSE for control than the conditional
mode/mean -- sampling injects the aleatoric jitter the task punishes.  The linear
head effectively returns the mean and wins.

THE FIX (all OUT-OF-GRAPH, learned object unchanged; spec section 11).  Decode the
MODE / most-likely action, never a random sample and never a soft average (a soft
average re-collapses to the between-modes point exactly as the linear head does):

  (a) flow MODE decode        : integrate the probability-flow ODE from the noise
                                MEAN a0 = 0 (deterministic).  For a near-Gaussian
                                conditional the rectified-flow map is affine and
                                Phi(0) = conditional mean = conditional mode ->
                                MATCHES the linear head exactly.  One-line change.
  (b) flow K-sample WTA       : draw K base points, integrate each, SCORE by the
                                exact CNF log-density (log N(a0) - integral tr(dv/da))
                                and return the ARGMAX candidate (winner-take-all,
                                NEVER average).  -> mode as K grows; commits on
                                multimodal conditionals.
  (c) energy MAP warm-start   : argmin_a E(a;obs) by a few Newton steps warm-started
                                from the linear mean (+ restarts along the top
                                Hessian eigvector = the Q_c object).  Same principle.

BIAS/VARIANCE (why sample-decode error > mode-decode error when near-unimodal).
Control cost ~ E||a_hat - a*||^2.  The conditional-mean estimator is the Bayes/MMSE
point: E||mean - a||^2 = tr(Sigma) (irreducible).  A single posterior SAMPLE has
    E_{a_hat ~ p} E_{a ~ p} ||a_hat - a||^2 = 2 tr(Sigma),
so on a (near-)unimodal conditional the sampler carries ~2x the MSE of the mean --
pure injected variance.  The linear head IS the mean -> tr(Sigma); the flow SAMPLE
decode -> ~2 tr(Sigma) -> strictly worse.  On a BIMODAL conditional the mean is an
INVALID between-modes action (lands in the empty valley), so mean-decode fails to
commit; MODE decode picks one mode exactly (no noise) -> dominates BOTH the sample
(noisy) and the mean (valley).  This proto measures all three numbers.

Purity argument (spec section 11): the LEARNED OBJECT stays the same polynomial
velocity field v_theta(a,t;obs) (a CP/bilinear tensor network, foldable, ODT-able).
Every new decode -- integrate from 0, the CNF divergence trace dv/da, the argmax,
the Newton argmin -- is an OUT-OF-GRAPH CONTROLLER op, the exact category as the
sanctioned gripper threshold / phase argmax / the existing Euler loop.  We change
ONLY the sampler's initial condition and its selection rule, never the graph.

Run:  python3 xvla/train/mode_decode_proto.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)

D = 2  # action dim (a small chunk stand-in)


# ===========================================================================
# Polynomial velocity feature map  phi(a, t, obs)  (CP-representable; a stack of
# the repo's degree-2 BilinearFFN cores emits exactly these monomials from the
# homogeneous input [1; a; t; obs]).  v_theta = W @ phi is linear in W, so
# flow-matching is a convex least-squares fit -- transparent for the proto.
# ===========================================================================
def make_feats(deg_a: int, deg_obs: int):
    aexp = [(i, j) for i in range(deg_a + 1) for j in range(deg_a + 1 - i)]

    def phi(a, t, obs):
        a1, a2 = a[..., 0], a[..., 1]
        cols = []
        for i, j in aexp:
            base = (a1 ** i) * (a2 ** j)
            for k in range(3):                      # t^0, t^1, t^2
                for m in range(deg_obs + 1):        # obs^0 .. obs^deg_obs
                    cols.append(base * (t ** k) * (obs ** m))
        return np.stack(cols, -1)

    return phi, len(aexp) * 3 * (deg_obs + 1)


def fit_flow_matching(sampler, phi, F, iters=8000, bs=8192):
    """Rectified-flow least squares: min_W E|| W phi(a_t,t,obs) - (a1 - a0) ||^2."""
    Spp = np.zeros((F, F))
    Sup = np.zeros((D, F))
    for _ in range(iters // 8):
        a1, obs = sampler(bs)
        a0 = rng.standard_normal((bs, D))
        t = rng.uniform(0.0, 1.0, bs)
        a_t = (1.0 - t)[:, None] * a0 + t[:, None] * a1
        u = a1 - a0
        P = phi(a_t, t, obs)
        Spp += P.T @ P
        Sup += u.T @ P
    return np.linalg.solve(Spp + 1e-4 * np.eye(F), Sup.T).T   # (D, F)


def fit_linear_mean(sampler, deg_obs=3, iters=8000, bs=8192):
    """The chi-VLA linear+MSE head analogue: a = M psi(obs), least squares -> E[a|obs]."""
    def psi(obs):
        return np.stack([obs ** m for m in range(deg_obs + 1)], -1)   # (n, deg_obs+1)
    Q = deg_obs + 1
    Spp = np.zeros((Q, Q)); Say = np.zeros((D, Q))
    for _ in range(iters // 8):
        a1, obs = sampler(bs)
        P = psi(obs)
        Spp += P.T @ P
        Say += a1.T @ P
    M = np.linalg.solve(Spp + 1e-6 * np.eye(Q), Say.T).T          # (D, Q)
    return lambda obs: psi(obs) @ M.T


# ===========================================================================
# Decoders.  All share the SAME learned field W; they differ ONLY in the
# out-of-graph initial condition + selection rule.
# ===========================================================================
CLIP = 6.0   # out-of-graph integration-domain clip (a controller op; spec section 11
             # lists environment-side clipping OUTSIDE the tensor boundary).  A fitted
             # polynomial field extrapolates wildly in the tails a random a0 can reach;
             # the deterministic a0=0 decode never approaches it -- itself a robustness
             # argument for MODE over SAMPLE.


def _step(a, W, phi, obs_v, t):
    return phi(a, np.full(a.shape[0], t), obs_v) @ W.T


def flow_sample(W, phi, obs, n, steps=50):
    """CURRENT head: a0 ~ N(0,I), single stochastic Euler integration (per-sample)."""
    a = rng.standard_normal((n, D))
    obs_v = np.full(n, obs); dt = 1.0 / steps
    for s in range(steps):
        a = np.clip(a + dt * _step(a, W, phi, obs_v, s * dt), -CLIP, CLIP)
    return a


def flow_mode(W, phi, obs, n, steps=50):
    """MODE decode: a0 = 0 (the noise MEAN), deterministic ODE.  One-line change from
    flow_sample.  For a near-Gaussian conditional Phi(0) = conditional mean = mode."""
    a = np.zeros((n, D))
    obs_v = np.full(n, obs); dt = 1.0 / steps
    for s in range(steps):
        a = np.clip(a + dt * _step(a, W, phi, obs_v, s * dt), -CLIP, CLIP)
    return a                                          # identical for all n (deterministic)


def flow_kwta(W, phi, obs, K=64, steps=50, eps=1e-3):
    """K-sample WINNER-TAKE-ALL: draw K base points, integrate each, score by the
    exact CNF log-density  log N(a0) - integral_0^1 tr(dv/da) dt,  return the ARGMAX.
    Never averages.  tr(dv/da) via central finite differences (the dv/da object the
    flow head's docstring flags as the t-indexed Q_c / energy-Hessian analogue)."""
    a = rng.standard_normal((K, D))
    a0 = a.copy()
    obs_v = np.full(K, obs); dt = 1.0 / steps
    logp = -0.5 * (a0 ** 2).sum(1) - D * 0.5 * np.log(2 * np.pi)   # log N(a0;0,I)
    for s in range(steps):
        t = s * dt
        # divergence tr(dv/da) by central differences (D extra pairs of field evals)
        div = np.zeros(K)
        for d in range(D):
            ap = a.copy(); ap[:, d] += eps
            am = a.copy(); am[:, d] -= eps
            div += (_step(ap, W, phi, obs_v, t)[:, d] - _step(am, W, phi, obs_v, t)[:, d]) / (2 * eps)
        logp = logp - dt * div                        # continuity eq. along the flow
        a = np.clip(a + dt * _step(a, W, phi, obs_v, t), -CLIP, CLIP)
    return a[logp.argmax()]                            # MAP candidate


# ===========================================================================
# Target: conditional is UNIMODAL in most obs, BIMODAL in a narrow band.
#   near mode   N(obs) = (obs, 0.6 obs)      obs-dependent -> precision matters
#   far  mode   Fm     = (-1.3, 1.3)         fixed alternate, present only if |obs|<=BAND
#   p_far(obs) = 0            if |obs| >  BAND   (genuinely UNIMODAL)
#              = 0.45         if |obs| <= BAND   (genuinely BIMODAL)
# Control target on the unimodal region = N(obs); the linear mean nails it there and
# lands in the empty valley on the bimodal band (exactly the LIBERO failure mode).
# ===========================================================================
BAND = 0.4
Fm = np.array([-1.3, 1.3])
SD = 0.08


def N_of(obs):
    return np.stack([obs, 0.6 * obs], -1)


def target(n):
    obs = rng.uniform(-1, 1, n)
    p_far = np.where(np.abs(obs) <= BAND, 0.45, 0.0)
    far = rng.random(n) < p_far
    a = N_of(obs) + SD * rng.standard_normal((n, D))
    a[far] = Fm + SD * rng.standard_normal((far.sum(), D))
    return a, obs


def part_precision(W, phi, mean_head):
    print("=" * 78)
    print("PART A  UNIMODAL region (|obs|>%.1f): control error to the correct action" % BAND)
    print("        N(obs)=(obs,0.6 obs).  Metric = mean ||a_hat - N(obs)||^2 (MSE).")
    print("=" * 78)
    obs_grid = np.concatenate([np.linspace(-1.0, -BAND - 0.05, 12),
                               np.linspace(BAND + 0.05, 1.0, 12)])
    err = {"linear-mean": [], "flow-SAMPLE": [], "flow-MODE(a0=0)": [], "flow-KWTA(K=64)": []}
    for obs in obs_grid:
        tgt = N_of(np.array([obs]))[0]
        lin = mean_head(np.array([obs]))[0]
        err["linear-mean"].append(((lin - tgt) ** 2).sum())
        s = flow_sample(W, phi, obs, n=400)            # 400 draws -> expected sample MSE
        err["flow-SAMPLE"].append((((s - tgt) ** 2).sum(1)).mean())
        md = flow_mode(W, phi, obs, n=1)[0]
        err["flow-MODE(a0=0)"].append(((md - tgt) ** 2).sum())
        wta = np.mean([((flow_kwta(W, phi, obs, K=64) - tgt) ** 2).sum() for _ in range(8)])
        err["flow-KWTA(K=64)"].append(wta)
    trSigma = D * SD ** 2                              # within-mode variance = tr(Sigma)
    mode_mse = np.mean(err["flow-MODE(a0=0)"])
    for k, v in err.items():
        m = np.mean(v)
        print(f"  {k:20s}  MSE = {m:.5f}   ({m / mode_mse:6.1f}x flow-MODE)")
    print(f"  reference tr(Sigma)=D*SD^2={trSigma:.5f} (irreducible within-mode var);"
          f"  2*tr(Sigma)={2*trSigma:.5f} = ideal MMSE sampling penalty")
    print("  => flow-SAMPLE (%.4f) >> flow-MODE (%.4f): TWO compounding penalties --"
          % (np.mean(err["flow-SAMPLE"]), mode_mse))
    print("     (i) the MMSE factor-2 variance (a posterior SAMPLE reproduces the")
    print("     demonstrator jitter the controller does NOT want), AND (ii) tail")
    print("     extrapolation -- a random a0 reaches regions where the fitted")
    print("     polynomial field is unreliable.  MODE decode from a0=0 avoids BOTH.")
    print("  => flow-MODE / flow-KWTA achieve <= linear-mean precision, SAME learned")
    print("     field, NO retrain -- ONLY the decode init/selection changed.  (flow-MODE")
    print("     beats the obs-only linear-mean head because it reads a jointly with obs.)")


def part_commit(W, phi, mean_head):
    print()
    print("=" * 78)
    print("PART B  BIMODAL band (obs=0): does the decode COMMIT to a mode, or average?")
    print("        modes: N(0)=(0,0) and Fm=(-1.3,1.3).  'commit' = within 0.5 of a mode.")
    print("=" * 78)
    modes = np.stack([N_of(np.array([0.0]))[0], Fm])   # (2, D)

    def commit_frac(pts):
        dmin = np.sqrt((((pts[:, None, :] - modes[None]) ** 2).sum(-1))).min(1)
        return float((dmin < 0.5).mean())

    lin = mean_head(np.array([0.0]))                    # (1, D)
    print(f"  linear-mean       -> a={np.round(lin[0],2)}  commit={commit_frac(lin):.2f}"
          f"   (lands in the valley between (0,0) and (-1.3,1.3): NEVER commits)")
    ssamp = flow_sample(W, phi, 0.0, n=2000)
    print(f"  flow-SAMPLE       -> commit={commit_frac(ssamp):.2f}  "
          f"frac->Fm={np.mean(np.linalg.norm(ssamp-Fm,axis=1)<np.linalg.norm(ssamp-modes[0],axis=1)):.2f}"
          f"   (commits, but each draw carries within-mode noise)")
    smode = flow_mode(W, phi, 0.0, n=1)
    print(f"  flow-MODE(a0=0)   -> a={np.round(smode[0],2)}  commit={commit_frac(smode):.2f}"
          f"   (!! on a SYMMETRIC bimodal obs the deterministic ODE from the symmetric")
    print("                        base point 0 maps to the between-modes valley -- the")
    print("                        one failure case of a0=0; use KWTA when multimodal)")
    wta = np.stack([flow_kwta(W, phi, 0.0, K=64) for _ in range(200)])
    print(f"  flow-KWTA(K=64)   -> commit={commit_frac(wta):.2f}  "
          f"frac->Fm={np.mean(np.linalg.norm(wta-Fm,axis=1)<np.linalg.norm(wta-modes[0],axis=1)):.2f}"
          f"   (MAP: off-axis draws + density argmax -> commits, no averaging)")
    print("  => a SOFT AVERAGE of samples would land back in the valley (= linear-mean")
    print("     failure).  flow-KWTA commits WITHOUT re-collapsing, on BOTH regions.")
    print("  RECOMMENDATION (two-tier): a0=0 is the minimal fix and is EXACT on")
    print("  unimodal conditionals (the LIBERO regime); KWTA is the robust general")
    print("  decode that also commits on genuinely (even symmetric) multimodal states.")


# ===========================================================================
# PART C -- the same principle for an ENERGY head: argmin_a E(a;obs) warm-started
# from the linear mean.  E = alpha (a^2-1)^2 + tau(obs) a  (double well; minima at
# +-1, obs-tilt tau picks the deeper).  Newton from the mean converges to the single
# well when unimodal; K restarts along +- the top Hessian eigvec commit when bimodal.
# ===========================================================================
def part_energy():
    print()
    print("=" * 78)
    print("PART C  ENERGY head MAP decode: argmin_a E(a;obs) warm-started from the mean")
    print("        E(a)=alpha (a^2-1)^2 + tau a.  Newton (+ Hessian-eigvec restarts).")
    print("=" * 78)
    alpha, w_tau = 2.0, np.array([0.0, 1.5])           # tau(obs) = w.[1, obs]

    def E(a, tau):   return alpha * (a * a - 1) ** 2 + tau * a
    def dE(a, tau):  return 4 * alpha * a * (a * a - 1) + tau      # dE/da
    def ddE(a):      return 4 * alpha * (3 * a * a - 1)            # d2E/da2 (the Q_c object)

    def newton(a0, tau, iters=40):
        a = float(a0)
        for _ in range(iters):
            h = ddE(a)
            a = a - dE(a, tau) / (h if abs(h) > 1e-6 else 1e-6)
        return a

    for obs, label in [(-0.7, "unambiguous (unimodal)"), (0.7, "unambiguous (unimodal)"),
                       (0.0, "AMBIGUOUS (bimodal)")]:
        tau = np.array([1.0, obs]) @ w_tau
        mean = np.tanh(-tau)                            # a cheap linear-mean stand-in warm start
        # restarts: the warm-started mean, plus +-1 along the (1-D) Hessian curvature
        cands = [newton(mean, tau), newton(mean + 1.0, tau), newton(mean - 1.0, tau)]
        cands = [c for c in cands if abs(c) < 3]
        star = min(cands, key=lambda c: E(c, tau))     # argmin = MAP (out-of-graph)
        commits = "one well" if abs(star) > 0.5 else "valley (barrier top)"
        print(f"  obs={obs:+.1f} ({label:23s}): warm-start mean={mean:+.2f} -> "
              f"argmin a*={star:+.2f} [{commits}]")
    print("  => Newton from the linear mean lands in the single deeper well when")
    print("     unimodal (= mean precision, sharpened); restarts + argmin commit to a")
    print("     well when bimodal, where the raw MSE mean would sit on the barrier top.")


if __name__ == "__main__":
    print("Fitting tensor-pure flow field + linear-mean head on the "
          "unimodal-mostly / bimodal-band target ...\n")
    phi, F = make_feats(deg_a=3, deg_obs=3)
    W = fit_flow_matching(target, phi, F)
    mean_head = fit_linear_mean(target, deg_obs=3)
    part_precision(W, phi, mean_head)
    part_commit(W, phi, mean_head)
    part_energy()
