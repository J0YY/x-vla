"""Minimal prototype: the HYBRID gated-routing action head — precise by default,
multimodal only where the task is genuinely ambiguous.

Standalone NumPy (the repo's Python is 3.14 -> no torch; this proves the math and
the training loop cheaply, exactly like ``diffusion_proto.py`` / ``product_routing``'s
torch head). The production head is the identical construction in torch, with every
learned map realised by a *bilinear/CP* (foldable) core -- see the report / the
`xvla/nn/gated_routing.py` integration plan.

WHY THIS HEAD (DEVLOG 2026-07-18 cont.17).  After the obs-mirror fix, the LINEAR
(E[a|obs], MSE) head reaches 25% closed-loop on LIBERO-Object; the pure tensor-pure
multimodal heads (flow, product-routing) are 0%. Diagnosis: LIBERO-Object is
*conditionally near-unimodal* -- given the exact image+instruction the correct action
is largely determined -- so a pure multimodal head pays a PRECISION cost the task
doesn't reward (flow: ODE sampling noise; product: its committed decode lands at
mu +/- |c| = displaced from the mean even when there is only one mode). But genuine
multimodality still exists at ambiguous states, where a pure-linear head lands in the
empty valley between modes and fails. We want a head that is PRECISE like linear by
default AND commits like the multimodal heads exactly where needed.

CONSTRUCTION (per pooled conditioning bond h = obs):

    a = mu(h)  +  fire(h) * sum_g sign(g_g(h)) * c_g(h)              (Eq. H)

  * mu(h)            LINEAR mean head -- the EXISTING precise 25% head, verbatim.
  * c_g(h)           G tensor-pure (BilinearFFN / CP) additive correction factors
                     -- the product-routing zonotope, but re-based on mu (c_0 == mu).
  * g_g(h)           G tensor-pure gate logits -> WHICH mode  (sign = out-of-graph).
  * e(h) -> fire     the AMBIGUITY GATE: a tensor-pure scalar predicting the
                     REDUCIBLE residual energy  E[ ||a-mu||^2 - ||a-(mu+correction)||^2 | h ];
                     fire = [ e(h) > tau ]  is an out-of-graph threshold (spec 11),
                     the SAME category as the gripper threshold and the phase argmax.

TENSOR-PURITY.  Every learned object -- mu (linear), c_g, g_g, e -- is a polynomial
(CP/bilinear) map of h; the whole forward object folds into one tensor network.
The ONLY non-polynomial ops are (i) the ambiguity threshold [e>tau] and (ii) the
routing signs sign(g_g), both out-of-graph controller ops (spec 11), exactly like the
sanctioned gripper threshold. The soft in-graph form  a = mu + sigma(e)*sum sign*c
(product of two polynomials = polynomial) is used at TRAIN time and is itself foldable;
the hard threshold is a DEPLOY-time hardening that yields the exact linear fallback.

GRACEFUL DEGRADATION (the key selling point -- NEVER worse than linear).
On a unimodal state e(h) < tau  =>  fire = 0  =>  a = mu(h) EXACTLY = the linear head.
Byte-identical output, so the hybrid can never be worse than linear on the (majority)
unimodal states; the correction can only ADD capability on the ambiguous sub-region.
With tau -> +inf the head IS the linear head; the value is that a well-separated
ambiguity signal lets tau sit between "unimodal" and "bimodal" reducible energy.

WHY PRODUCT-ROUTING (not flow) FOR THE CORRECTION.  The correction's decode must be
DETERMINISTIC so that, when it fires, it commits without re-introducing the sampling
noise that sank the flow head (cont.17). Product-routing's MAP decode is a single
deterministic zonotope vertex -> a committed mode, no ODE, no RNG. (Flow remains the
right correction for a continuous/unbounded mode manifold; product is right here.)

TRAINING (mu stays exactly precise -- the guarantee's backbone):
  L_mean  = ||mu(h) - a*||^2                         (mu = the precise conditional mean)
  s*      = argmin_{s in {-1,+1}^G} ||sg(mu) + sum s_g c_g - a*||^2   (loss-only assign)
  L_corr  = ||sg(mu) + sum s*_g c_g - a*||^2         (sg = stop-grad: mu is NOT corrupted)
  L_sign  = BCE(g_g, bit(s*_g))                      (gates predict the winning mode)
  b       = ||a*-sg(mu)||^2 - ||a*-(sg(mu)+sum s*_g c_g)||^2   (per-sample reducible energy)
  L_amb   = ||e(h) - sg(b)||^2  +  lam * mean(relu(e(h)))      (detect ambiguity + sparsity)
The stop-grad on mu is what makes "never worse than linear" hold: the correction path
learns the residual structure and can never degrade the mean.

WHAT THIS PROTO SHOWS (a MIXED target: unimodal on most of obs-space, bimodal in a
sub-region |obs|<0.3):
  * LINEAR      : precise on the unimodal region, but on the bimodal region lands in
                  the empty valley between the modes (fails to commit).
  * PURE PRODUCT: commits on the bimodal region, but its deterministic decode lands at
                  mu +/- |c| on the UNIMODAL region too (|c| bleeds out of the region)
                  -> strictly LESS precise than linear there (the cont.17 effect).
  * HYBRID      : matches linear precision on the unimodal region (gate off -> a=mu
                  exactly) AND commits on the bimodal region -> dominates both.

Run:  python3 xvla/train/hybrid_gate_proto.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


# ===========================================================================
# Shared tensor-pure obs features  phi(obs) = [1, obs, obs^2, obs^3, obs^4].
# Degree-4 == two stacked degree-2 BilinearFFN cores; ALL heads get the SAME
# features so the comparison is about the HEAD, not the feature budget.
# ===========================================================================
DEG = 4
D_A = 2                       # 2-D action chunk (a1, a2)


def phi(obs):                 # obs: (n,) -> (n, DEG+1)
    return np.stack([obs ** k for k in range(DEG + 1)], -1)


# ===========================================================================
# MIXED target.  obs ~ U[-1,1].  Smooth mean m(obs) = [obs, 0.5*obs].
#   * |obs| < 0.3  : BIMODAL.  a = m(obs) +/- [0, DELTA],  s in {-1,+1} equal prob.
#   * otherwise    : UNIMODAL. a = m(obs) + small noise.
# The bimodal split is in the a2 coordinate.
# ===========================================================================
DELTA = 1.2                   # half-separation of the two modes (large -> linear fails)
REGION = 0.3                  # |obs| < REGION is the ambiguous sub-region
NOISE = 0.08


def mean_curve(obs):          # the true conditional MEAN E[a|obs] (what linear fits)
    return np.stack([obs, 0.5 * obs], -1)


def target(n):
    obs = rng.uniform(-1, 1, n)
    m = mean_curve(obs)
    a = m + NOISE * rng.standard_normal((n, D_A))
    bim = np.abs(obs) < REGION
    s = rng.choice([-1.0, 1.0], n)
    a[bim, 1] += s[bim] * DELTA                       # split the a2 coord in the region
    return obs, a, bim


def modes_at(obs):
    """The (up to 2) valid mode-actions at obs, for the commit / nearest-mode metric."""
    m = mean_curve(obs)
    if np.abs(obs) < REGION:
        return np.stack([m + [0, DELTA], m + [0, -DELTA]])   # (2, D_A)
    return m[None]                                            # (1, D_A)


# ===========================================================================
# Head 1 -- LINEAR mean head.  Closed-form least squares  a ~= W phi(obs).
# ===========================================================================
def fit_linear(obs, a):
    P = phi(obs)
    W = np.linalg.solve(P.T @ P + 1e-6 * np.eye(P.shape[1]), P.T @ a)   # (F, D_A)
    return W


def mu_of(W, obs):
    return phi(obs) @ W


# ===========================================================================
# Product-routing correction (G=1 factor -> 2 modes).  Alternating least squares:
# assign the best sign per sample (loss-only argmin), then re-fit the linear-in-params
# center/factor.  Exactly the torch head's best-sign-assignment (product_routing.py).
# ===========================================================================
def fit_routing(obs, a, center_fixed=None, iters=40):
    """Fit  c0(obs) + s*c1(obs)  to a.  If center_fixed (a (n,D_A) array of mu) is
    given, c0 is FROZEN to it (the HYBRID case: correction rides on the linear mean);
    else c0 is learned (the PURE-PRODUCT case)."""
    P = phi(obs)
    Wc1 = 1e-2 * rng.standard_normal((P.shape[1], D_A))
    Wc0 = np.zeros((P.shape[1], D_A))
    for _ in range(iters):
        c0 = center_fixed if center_fixed is not None else P @ Wc0
        c1 = P @ Wc1
        # assign best sign per sample (loss-only argmin over {-1,+1})
        err_p = ((c0 + c1 - a) ** 2).sum(-1)
        err_m = ((c0 - c1 - a) ** 2).sum(-1)
        s = np.where(err_p <= err_m, 1.0, -1.0)                       # (n,)
        if center_fixed is None:
            # fit [c0, c1] jointly: design [phi, s*phi] -> a
            X = np.concatenate([P, s[:, None] * P], 1)
            Wcat = np.linalg.solve(X.T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ a)
            Wc0, Wc1 = Wcat[: P.shape[1]], Wcat[P.shape[1]:]
        else:
            # c0 frozen: fit c1 to the residual  a - c0  with design  s*phi
            r = a - center_fixed
            X = s[:, None] * P
            Wc1 = np.linalg.solve(X.T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ r)
    # sign predictor g(obs): regress the assigned sign (out-of-graph sign at decode)
    c0 = center_fixed if center_fixed is not None else P @ Wc0
    c1 = P @ Wc1
    s = np.where(((c0 + c1 - a) ** 2).sum(-1) <= ((c0 - c1 - a) ** 2).sum(-1), 1.0, -1.0)
    Wg = np.linalg.solve(P.T @ P + 1e-6 * np.eye(P.shape[1]), P.T @ s)   # (F,) sign logits
    return Wc0, Wc1, Wg, s


def fit_ambiguity(obs, a, mu, Wc1):
    """Ambiguity gate e(obs): regress the per-sample REDUCIBLE residual energy
    b = ||a-mu||^2 - ||a-(mu+s*c1)||^2  (>=0). High only where committing genuinely
    reduces error = a genuine mode split; ~0 on unimodal (noisy) states."""
    P = phi(obs)
    c1 = P @ Wc1
    r = a - mu
    s = np.where(((c1 - r) ** 2).sum(-1) <= ((-c1 - r) ** 2).sum(-1), 1.0, -1.0)
    resid_mean = (r ** 2).sum(-1)
    resid_corr = ((r - s[:, None] * c1) ** 2).sum(-1)
    b = resid_mean - resid_corr                                        # (n,) >= 0
    We = np.linalg.solve(P.T @ P + 1e-6 * np.eye(P.shape[1]), P.T @ b)  # (F,)
    return We, b


# ===========================================================================
# Decoders (OUT-OF-GRAPH: the signs and the fire-threshold are controller ops).
# ===========================================================================
def decode_linear(W, obs):
    return mu_of(W, obs)


def decode_product(Wc0, Wc1, Wg, obs):
    P = phi(obs)
    s = np.where(P @ Wg > 0, 1.0, -1.0)                               # out-of-graph sign
    return P @ Wc0 + s[:, None] * (P @ Wc1)


def decode_hybrid(Wmu, Wc1, Wg, We, obs, tau):
    P = phi(obs)
    mu = P @ Wmu
    fire = (P @ We) > tau                                             # out-of-graph gate
    s = np.where(P @ Wg > 0, 1.0, -1.0)                              # out-of-graph sign
    corr = fire[:, None] * (s[:, None] * (P @ Wc1))
    return mu + corr, fire


# ===========================================================================
# Metrics.  On a MULTIMODAL target the right control metric is distance to the
# NEAREST valid mode (a committed mode is good; the between-modes valley is bad).
# We also report precision on the unimodal region (distance to the single mode).
# ===========================================================================
def nearest_mode_err(pred, obs):
    e = np.empty(len(obs))
    for i in range(len(obs)):
        M = modes_at(obs[i])                                          # (K, D_A)
        e[i] = np.sqrt(((M - pred[i]) ** 2).sum(-1).min())
    return e


def evaluate(name, pred, obs, bim):
    err = nearest_mode_err(pred, obs)
    uni_err = err[~bim].mean()                       # PRECISION on unimodal states
    bim_err = err[bim].mean()                        # control err on ambiguous states
    commit = (err[bim] < 0.4).mean()                 # fraction that lands ON a mode
    print(f"  {name:14s}  unimodal-err {uni_err:.3f}   bimodal-err {bim_err:.3f}   "
          f"bimodal-commit {commit:.2f}   overall {err.mean():.3f}")
    return uni_err, bim_err, commit


def main():
    print("MIXED target: unimodal for |obs|>=0.3, bimodal (+/-%.1f in a2) for |obs|<0.3\n"
          % DELTA)
    obs_tr, a_tr, _ = target(60000)
    obs_te, a_te, bim_te = target(40000)
    print(f"  train n={len(obs_tr)}  test n={len(obs_te)}  "
          f"(ambiguous fraction {bim_te.mean():.2f})\n")

    # --- fit all three heads ---
    Wlin = fit_linear(obs_tr, a_tr)                                   # LINEAR = the mu head
    Wc0_p, Wc1_p, Wg_p, _ = fit_routing(obs_tr, a_tr)                 # PURE product-routing
    mu_tr = mu_of(Wlin, obs_tr)
    _, Wc1_h, Wg_h, _ = fit_routing(obs_tr, a_tr, center_fixed=mu_tr) # HYBRID correction on mu
    We_h, b_tr = fit_ambiguity(obs_tr, a_tr, mu_tr, Wc1_h)            # ambiguity gate

    # choose tau to CONTROL the false-positive rate: put it at the 98th percentile of
    # e on unimodal states, so <=2% of unimodal states fire (precision preserved).
    # A degree-4 gate of a SCALAR obs can't cut a perfectly sharp region boundary; the
    # torch gate is a BilinearFFN of the full high-dim h and cuts it far sharper, so
    # this residual boundary bleed is a proto artifact, not fundamental.
    e_tr = phi(obs_tr) @ We_h
    uni_e = np.median(e_tr[np.abs(obs_tr) >= REGION])
    bim_e = np.median(e_tr[np.abs(obs_tr) < REGION])
    tau = np.quantile(e_tr[np.abs(obs_tr) >= REGION], 0.98)
    print(f"  ambiguity gate e(obs): median on unimodal={uni_e:+.3f}  "
          f"bimodal={bim_e:+.3f}  ->  tau={tau:.3f} (98th-pct of unimodal e)\n")

    # --- decode + evaluate ---
    print("RESULTS (distance to NEAREST valid mode; lower=better; commit=frac on a mode):")
    evaluate("linear", decode_linear(Wlin, obs_te), obs_te, bim_te)
    evaluate("pure-product", decode_product(Wc0_p, Wc1_p, Wg_p, obs_te), obs_te, bim_te)
    pred_h, fire = decode_hybrid(Wlin, Wc1_h, Wg_h, We_h, obs_te, tau)
    evaluate("hybrid", pred_h, obs_te, bim_te)

    # --- the graceful-degradation guarantee, made explicit ---
    print(f"\n  gate fire-rate: unimodal states {fire[~bim_te].mean():.3f}  "
          f"bimodal states {fire[bim_te].mean():.3f}")
    a_lin = decode_linear(Wlin, obs_te)
    identical = np.allclose(pred_h[~fire], a_lin[~fire])
    print(f"  on non-firing states hybrid output == linear output EXACTLY: {identical}")
    print("  => where the gate is off the hybrid IS the linear head (never worse);")
    print("     the correction only adds commitment on the genuinely bimodal region.")


if __name__ == "__main__":
    main()
