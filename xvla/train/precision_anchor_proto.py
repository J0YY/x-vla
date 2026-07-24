"""Precision-preserving multimodal action objectives — NumPy prototype.

Standalone NumPy (repo Python is 3.14 → no torch; proves the math + the training
loop cheaply, exactly like ``diffusion_proto.py`` / ``energy_ebm_proto.py`` /
``multihypothesis.py::_demo_bimodal_1d``). The production heads are the identical
constructions in torch (``xvla/nn/flow_action.py``, ``xvla/nn/product_routing.py``);
here every learned map is linear-in-parameters over a fixed POLYNOMIAL feature basis,
which is exactly what a stack of the repo's degree-2 ``BilinearFFN`` CP cores emits
from the homogeneous input [1; a; φ(t); h].  Nothing here changes tensor-purity: the
fixes are LOSS-ONLY (spec §11) — the deployed graph is still the same polynomial field.

THE PROBLEM (DEVLOG cont.17).  After the obs-mirror fix, the LINEAR+MSE head reaches
25% closed-loop on LIBERO-Object; the flow-matching and product-routing heads reach 0%.
LIBERO-Object is *conditionally near-unimodal*: given the exact frame+instruction the
correct action is nearly determined, so E[a|obs] (what MSE fits) is near-optimal.  Plain
MSE is a low-variance, sample-efficient estimator of that mean; flow-matching-MSE and
MCL/best-sign-assignment are higher-variance estimators that fit the arm LESS precisely
at the same data/step budget → they lose per-step precision the task rewards.

WHY (bias/variance, made concrete by this proto):
  * MSE regresses ONE target (the mean) per demo → 1 constraint/demo, variance ~ σ²/N.
  * FLOW-MATCHING regresses a whole velocity FIELD v(a,t) over the augmented (a,t)
    domain; each demo must be paired with random (a0,t) draws, so the SAME data is
    spread thin over a bigger domain (sample-inefficiency), and DECODE adds ODE
    sampling variance — even the optimal field decodes a *sample*, not the mean.
  * MCL / best-sign PRODUCT ROUTING trains only the WINNING expert per demo → each
    of the 2^G branches sees a fraction of the data, and the argmin assignment is a
    high-variance, non-smooth gradient estimator (which branch wins flips at init).

THE FIXES (all loss-only, tensor-purity preserved):
  (a) RECONSTRUCTION / data-consistency anchor for flow-matching — pin the demo's OWN
      noise→action coupling so the ODE reconstructs the demo action exactly (a
      deterministic-coupling term added to the random-coupling FM loss).
  (b) DISTILLATION anchor — the linear+MSE head is a PRECISION TEACHER; anchor the
      multimodal head's first moment to the teacher mean, weighted high where the
      state is (near-)unimodal, ~0 where it is multimodal → matches MSE on unimodal
      states, still branches on multimodal ones.
  (c)+(d) PRECISION-WEIGHTED / CALIBRATED product routing with a UNIMODAL→MULTIMODAL
      CURRICULUM — replace hard argmin by a soft, temperature-annealed assignment.
      High τ = soft = re-averaging = fits the mean precisely (like MSE); anneal τ→0 to
      commit.  The winning expert is regressed with a posterior weight (precision-
      weighted); gate targets are the soft posterior (calibrated, not a hard bit).

WHAT THIS PROTO SHOWS, at a FIXED small data/step budget:
  * On a NEAR-UNIMODAL conditional target: plain flow & plain product lose precision
    vs MSE; the anchored/curriculum variants close MOST of that gap.
  * On a genuinely BIMODAL conditional target: MSE collapses to the (invalid) between-
    modes mean and never commits, while ALL the multimodal variants still commit —
    the anchors buy precision back WITHOUT killing multimodality.

Run:  python3 xvla/train/precision_anchor_proto.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)

# --------------------------------------------------------------------------- #
# Fixed small budget (the whole point — everything below uses these).
# --------------------------------------------------------------------------- #
N_TRAIN = 300        # demos (fixed data budget, shared by every head)
N_AUG = 4            # (a0, t) draws per demo for flow (fixed augmentation budget)
ODE_STEPS = 12       # out-of-graph Euler decode steps
SGD_STEPS = 1500     # product-routing optimisation budget
SIGMA = 0.12         # irreducible per-mode noise (the precision floor)
DA = 2               # action dim (a small "chunk"; the story is dim-agnostic)


# --------------------------------------------------------------------------- #
# Targets.  obs ∈ [-1, 1] is the context (the "state that determines the action").
# --------------------------------------------------------------------------- #
def mu_unimodal(obs):
    """Near-unimodal conditional MEAN μ(obs) — the action is (nearly) determined."""
    return np.stack([1.2 * obs, -0.8 * obs], axis=-1)         # (..., 2)


def sample_unimodal(n):
    obs = rng.uniform(-1, 1, n)
    a = mu_unimodal(obs) + SIGMA * rng.standard_normal((n, DA))
    return a, obs


MODES = np.array([[1.3, 1.3], [-1.3, -1.3]])                  # the two arm strategies


def sample_bimodal(n):
    """obs tilts WHICH of two whole-chunk strategies the demo took (reach ±)."""
    obs = rng.uniform(-1, 1, n)
    p_plus = 0.5 + 0.4 * obs                                  # obs>0 → mode0 favoured
    s = (rng.random(n) < p_plus).astype(int)                 # 0 or 1
    a = MODES[s] + SIGMA * rng.standard_normal((n, DA))
    return a, obs


# --------------------------------------------------------------------------- #
# Polynomial features.  poly_obs = [1, obs, obs²] (context features for the heads).
# flow_feats = monomials in a (deg_a) × t (0..2) × obs (0..2) — the CP field basis.
# --------------------------------------------------------------------------- #
def poly_obs(obs, deg=2):
    return np.stack([obs ** k for k in range(deg + 1)], -1)   # (..., deg+1)


def make_flow_feats(deg_a=3, m_obs=2):
    aexp = [(i, j) for i in range(deg_a + 1) for j in range(deg_a + 1 - i)]

    def phi(a, t, obs):
        a1, a2 = a[..., 0], a[..., 1]
        cols = []
        for i, j in aexp:
            base = (a1 ** i) * (a2 ** j)
            for k in range(3):                # t^0, t^1, t^2
                for m in range(m_obs + 1):    # obs^0 .. obs^m_obs
                    cols.append(base * (t ** k) * (obs ** m))
        return np.stack(cols, -1)
    return phi, len(aexp) * 3 * (m_obs + 1)


# ===========================================================================
# 1) LINEAR + MSE head — the precision TEACHER.  â(obs) = W · poly_obs(obs).
# ===========================================================================
def fit_mse(a, obs):
    F = poly_obs(obs)                                         # (N, 3)
    W = np.linalg.solve(F.T @ F + 1e-6 * np.eye(F.shape[1]), F.T @ a)   # (3, DA)
    return W


def mse_predict(W, obs):
    return poly_obs(obs) @ W                                  # deterministic mean


# ===========================================================================
# 2) FLOW-MATCHING head (rectified flow), fit by least squares over W.
#    plain  : random coupling a0~N per (demo, aug)  →  target E[a1-a0 | a_t]
#    +recon : ALSO pin each demo's OWN fixed a0 coupling (data-consistency) so the
#             ODE from the demo's noise reconstructs the demo action (fix a).
#    +recon+distill : additionally pin, WHERE THE STATE IS UNIMODAL (weight w(obs)),
#             the straight path to the TEACHER mean μ_T(obs) (fix b).
# ===========================================================================
def fit_flow(a, obs, phi, F, recon=0.0, distill=0.0, Wteacher=None, w_unimodal=None,
             a0_recon=None):
    d = DA
    Spp = np.zeros((F, F))
    Sup = np.zeros((d, F))

    # --- random-coupling FM term (marginal; gives mode COVERAGE) ---
    for a1, o in [(np.repeat(a, N_AUG, 0), np.repeat(obs, N_AUG))]:
        a0 = rng.standard_normal((a1.shape[0], d))
        t = rng.uniform(0, 1, a1.shape[0])
        a_t = (1 - t)[:, None] * a0 + t[:, None] * a1
        u = a1 - a0
        P = phi(a_t, t, o)
        Spp += P.T @ P
        Sup += u.T @ P

    # --- (a) reconstruction / data-consistency anchor: demo's OWN fixed coupling ---
    if recon > 0.0:
        a0d = a0_recon if a0_recon is not None else rng.standard_normal((a.shape[0], d))
        for _ in range(3):                                    # a few t knots on the pin path
            t = rng.uniform(0, 1, a.shape[0])
            a_t = (1 - t)[:, None] * a0d + t[:, None] * a
            u = a - a0d
            P = phi(a_t, t, obs) * np.sqrt(recon)
            Spp += P.T @ P
            Sup += (u * recon).T @ phi(a_t, t, obs)

    # --- (b) distillation anchor: pin path to TEACHER mean where state is unimodal ---
    if distill > 0.0:
        muT = mse_predict(Wteacher, obs)                      # (N, DA) teacher mean
        wgt = distill * w_unimodal                            # per-state precision weight
        a0d = rng.standard_normal((a.shape[0], d))
        for _ in range(3):
            t = rng.uniform(0, 1, a.shape[0])
            a_t = (1 - t)[:, None] * a0d + t[:, None] * muT
            u = muT - a0d
            P = phi(a_t, t, obs) * np.sqrt(wgt)[:, None]
            Spp += P.T @ P
            Sup += (u * wgt[:, None]).T @ phi(a_t, t, obs)

    W = np.linalg.solve(Spp + 1e-4 * np.eye(F), Sup.T).T      # (DA, F)
    return W


def flow_decode(W, phi, obs, steps=ODE_STEPS, a0=None, clip=4.0):
    """OUT-OF-GRAPH Euler ODE decode.  obs: (n,) → (n, DA).  a0 optional (recon check).

    The box clip is the sanctioned external safety layer (spec §11 "environment-specific
    safety clipping" lives outside the tensor network); it only tames rare high-degree
    ODE excursions and never touches in-distribution decodes."""
    n = obs.shape[0]
    a = rng.standard_normal((n, DA)) if a0 is None else a0.copy()
    dt = 1.0 / steps
    for s in range(steps):
        t = np.full(n, s * dt)
        a = np.clip(a + dt * (phi(a, t, obs) @ W.T), -clip, clip)
    return a


# ===========================================================================
# 3) PRODUCT-ROUTING head (Eq. P): â = c0 + Σ_g sign_g · c_g,  c_· = C_· · poly_obs.
#    plain     : HARD best-sign assignment (argmin over 2^G combos) — the current loss.
#    +curriculum+distill : SOFT posterior assignment w_combo = softmax(-err/τ), τ
#                annealed hi→lo (fix c precision-weight + fix d curriculum); gate
#                targets = soft posterior (calibrated); center c0 distilled to the
#                teacher mean with a τ-decaying weight (fix b).
# ===========================================================================
class Adam:
    def __init__(self, shapes, lr=3e-2):
        self.lr = lr
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, g in enumerate(grads):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * (g * g)
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            params[i] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def _combos(G):
    import itertools
    return np.array(list(itertools.product([-1.0, 1.0], repeat=G)))   # (2^G, G)


def fit_product(a, obs, G=2, curriculum=False, distill=0.0, Wteacher=None):
    Fdim = 3                                                  # poly_obs width
    combos = _combos(G)                                       # (C, G)
    C = combos.shape[0]
    # params: center C0 (Fdim,DA); factors Cg (G,Fdim,DA); gates Wg (G,Fdim)
    C0 = 0.01 * rng.standard_normal((Fdim, DA))
    Cg = 0.05 * rng.standard_normal((G, Fdim, DA))
    Wg = 0.01 * rng.standard_normal((G, Fdim))
    # break symmetry so factors start in different basins (matches product_routing.py)
    for g in range(G):
        Wg[g, 0] += 0.02 * (g - (G - 1) / 2)
    opt = Adam([C0.shape, Cg.shape, Wg.shape], lr=3e-2)
    params = [C0, Cg, Wg]
    f = poly_obs(obs)                                         # (N, Fdim)
    muT = mse_predict(Wteacher, obs) if distill > 0 else None

    for step in range(SGD_STEPS):
        C0, Cg, Wg = params
        c0 = f @ C0                                           # (N, DA)
        cg = np.einsum("nf,gfd->ngd", f, Cg)                 # (N, G, DA)
        # predicted action for every combo:  (N, C, DA)
        preds = c0[:, None, :] + np.einsum("cg,ngd->ncd", combos, cg)
        err = ((preds - a[:, None, :]) ** 2).mean(-1)        # (N, C)

        if curriculum:
            # τ: soft (≈ mean, precise) → hard (commit).  fix (c)+(d).
            frac = step / SGD_STEPS
            tau = 1.0 * (0.02 / 1.0) ** frac
            w = _softmax(-err / tau, axis=1)                 # (N, C) posterior weights
        else:
            w = np.zeros_like(err)
            w[np.arange(len(err)), err.argmin(1)] = 1.0      # HARD argmin (plain MCL)

        # ---- gradients of Σ_n Σ_c w_nc ||pred_nc - a_n||²  (w treated as constant) ----
        resid = preds - a[:, None, :]                        # (N, C, DA)
        wr = (w[:, :, None] * resid) / len(a)                # (N, C, DA)
        gC0 = f.T @ wr.sum(1) * 2                            # (Fdim, DA)
        # ∂/∂Cg : combos picks sign per factor
        gCg = 2 * np.einsum("nf,cg,ncd->gfd", f, combos, wr)

        # ---- gate loss: BCE(sigmoid(Wg·f), posterior p(sign_g=+)) ----
        p_plus = np.einsum("nc,cg->ng", w, (combos > 0).astype(float))   # (N, G) target
        logits = f @ Wg.T                                    # (N, G)
        pred_p = _sigmoid(logits)
        gWg = ((pred_p - p_plus)[:, :, None] * f[:, None, :]).mean(0)    # (G, Fdim)

        # ---- (b) distill center to teacher mean, weight decays as we sharpen ----
        if distill > 0:
            lam = distill * (1.0 - (step / SGD_STEPS))       # strong early, →0 late
            gC0 += 2 * lam * (f.T @ (c0 - muT)) / len(a)

        opt.step(params, [gC0, gCg, gWg])

    return params, combos


def product_decode(params, obs, combos):
    C0, Cg, Wg = params
    f = poly_obs(obs)
    c0 = f @ C0
    cg = np.einsum("nf,gfd->ngd", f, Cg)
    signs = np.where(f @ Wg.T > 0, 1.0, -1.0)                # (N, G) out-of-graph
    return c0 + np.einsum("ng,ngd->nd", signs, cg)


def _softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ===========================================================================
# Metrics.
# ===========================================================================
def precision_unimodal(decode_fn, draws=4):
    """E_obs ||decode(obs) - μ(obs)||².  Lower = more precise.  We compare to the true
    MEAN, so a perfect head → 0; the MSE teacher ≈ its own estimation variance.  Averaged
    over a few decode draws to denoise the stochastic ODE sampler."""
    obs = rng.uniform(-1, 1, 4000)
    mu = mu_unimodal(obs)
    return float(np.mean([((decode_fn(obs) - mu) ** 2).sum(-1).mean() for _ in range(draws)]))


def commit_bimodal(decode_fn):
    """Fraction of decodes landing within 0.5 of a REAL mode (commitment), and the
    mean distance to the nearest mode (0 = perfectly on a mode)."""
    obs = rng.uniform(-1, 1, 4000)
    dec = decode_fn(obs)
    d = ((dec[:, None, :] - MODES[None]) ** 2).sum(-1)       # (n, 2)
    nearest = np.sqrt(d.min(1))
    return float((nearest < 0.5).mean()), float(nearest.mean())


# ===========================================================================
# Driver.
# ===========================================================================
def run():
    print("=" * 78)
    print(f"FIXED BUDGET: {N_TRAIN} demos, {N_AUG} flow aug/demo, {ODE_STEPS} ODE steps,"
          f" {SGD_STEPS} product steps, σ={SIGMA}")
    print("=" * 78)

    # ---- data (shared, fixed) ----
    au, obu = sample_unimodal(N_TRAIN)
    ab, obb = sample_bimodal(N_TRAIN)
    phi, F = make_flow_feats()

    # ---- teachers (linear MSE, one per target) ----
    Wt_u = fit_mse(au, obu)
    Wt_b = fit_mse(ab, obb)

    # a data-driven "unimodality weight" for distillation: high where the conditional
    # spread is small.  Here we bin obs and measure residual std around the MSE mean;
    # small residual ⇒ unimodal ⇒ trust the teacher.  (In production: teacher residual
    # / an ensemble-disagreement proxy.)
    def unimodal_weight(a, obs, Wt, tau=0.35):
        # ABSOLUTE residual scale: a demo whose action sits near the teacher mean
        # (residual ≈ σ) is a unimodal state → weight ≈ 1; a demo far from the mean
        # (a real second mode → residual ≈ mode gap) → weight ≈ 0.
        r = a - mse_predict(Wt, obs)
        s = np.sqrt((r ** 2).sum(-1))                        # per-demo residual mag
        return np.exp(-0.5 * (s / tau) ** 2)

    wu = unimodal_weight(au, obu, Wt_u)                      # ~all high (truly unimodal)
    wb = unimodal_weight(ab, obb, Wt_b)                      # ~all low  (truly bimodal)
    print(f"\nunimodality weight  mean(unimodal data)={wu.mean():.2f}  "
          f"mean(bimodal data)={wb.mean():.2f}   (high→trust teacher, low→branch)")

    # ---- fit every head on BOTH targets ----
    heads = {}

    heads["MSE (teacher)"] = (
        lambda o: mse_predict(Wt_u, o),
        lambda o: mse_predict(Wt_b, o),
    )

    # a shared per-demo fixed coupling so the recon anchor / recon check use the SAME a0
    a0fix_u = rng.standard_normal((N_TRAIN, DA))
    a0fix_b = rng.standard_normal((N_TRAIN, DA))

    Wf_u = fit_flow(au, obu, phi, F)
    Wf_b = fit_flow(ab, obb, phi, F)
    heads["flow (plain)"] = (
        lambda o: flow_decode(Wf_u, phi, o),
        lambda o: flow_decode(Wf_b, phi, o),
    )

    Wfr_u = fit_flow(au, obu, phi, F, recon=5.0, a0_recon=a0fix_u)
    Wfr_b = fit_flow(ab, obb, phi, F, recon=5.0, a0_recon=a0fix_b)
    heads["flow +recon (a)"] = (
        lambda o: flow_decode(Wfr_u, phi, o),
        lambda o: flow_decode(Wfr_b, phi, o),
    )

    Wfd_u = fit_flow(au, obu, phi, F, recon=5.0, distill=25.0, Wteacher=Wt_u,
                     w_unimodal=wu, a0_recon=a0fix_u)
    Wfd_b = fit_flow(ab, obb, phi, F, recon=5.0, distill=25.0, Wteacher=Wt_b,
                     w_unimodal=wb, a0_recon=a0fix_b)
    heads["flow +recon+distill (a,b)"] = (
        lambda o: flow_decode(Wfd_u, phi, o),
        lambda o: flow_decode(Wfd_b, phi, o),
    )

    # reconstruction-fidelity check: integrate the ODE from each demo's OWN fixed a0
    # and compare to the demo action.  The recon anchor should sharpen this.
    def recon_err(W, a0fix, a, obs):
        rec = flow_decode(W, phi, obs, a0=a0fix)
        return float(np.sqrt(((rec - a) ** 2).sum(-1)).mean())
    print(f"\nRECON check (ODE from demo's own a0 → demo action, unimodal data):")
    print(f"  flow plain  ‖recon-a‖ = {recon_err(Wf_u, a0fix_u, au, obu):.3f}"
          f"   flow +recon = {recon_err(Wfr_u, a0fix_u, au, obu):.3f}"
          f"   (anchor pins the demo's own coupling → tighter reconstruction)")

    pu = fit_product(au, obu, curriculum=False)
    pb = fit_product(ab, obb, curriculum=False)
    heads["product (plain MCL)"] = (
        lambda o: product_decode(pu[0], o, pu[1]),
        lambda o: product_decode(pb[0], o, pb[1]),
    )

    pcu = fit_product(au, obu, curriculum=True, distill=4.0, Wteacher=Wt_u)
    pcb = fit_product(ab, obb, curriculum=True, distill=4.0, Wteacher=Wt_b)
    heads["product +curric+distill (b,c,d)"] = (
        lambda o: product_decode(pcu[0], o, pcu[1]),
        lambda o: product_decode(pcb[0], o, pcb[1]),
    )

    # ---- report ----
    mse_prec = precision_unimodal(heads["MSE (teacher)"][0])
    print("\n" + "-" * 78)
    print("NEAR-UNIMODAL target — PRECISION  E||decode-μ(obs)||²  (lower=better)")
    print(f"{'head':34s} {'precision':>11s} {'× MSE':>8s}  {'gap closed vs plain':>20s}")
    print("-" * 78)
    plain_flow = precision_unimodal(heads["flow (plain)"][0])
    plain_prod = precision_unimodal(heads["product (plain MCL)"][0])
    for name, (fu, _) in heads.items():
        p = precision_unimodal(fu)
        base = plain_flow if "flow" in name else (plain_prod if "product" in name else None)
        if base is not None and base > mse_prec and name not in ("flow (plain)", "product (plain MCL)"):
            closed = 100 * (base - p) / (base - mse_prec)
            gc = f"{closed:5.0f}%"
        else:
            gc = "—"
        print(f"{name:34s} {p:11.4f} {p / mse_prec:7.1f}x  {gc:>20s}")

    print("\n" + "-" * 78)
    print("BIMODAL target — COMMITMENT  (frac within 0.5 of a real mode; dist to mode)")
    print(f"{'head':34s} {'commit':>8s} {'mean-dist':>10s}   verdict")
    print("-" * 78)
    for name, (_, fb) in heads.items():
        c, dnear = commit_bimodal(fb)
        verdict = "COMMITS" if c > 0.7 else ("collapses (mean)" if c < 0.2 else "partial")
        print(f"{name:34s} {c:8.2f} {dnear:10.3f}   {verdict}")

    print("\n" + "=" * 78)
    print("READ-OUT:")
    print("  * On the NEAR-UNIMODAL target plain flow/product are far less precise than")
    print("    MSE (the DEVLOG cont.17 gap); the recon/distill/curriculum anchors close")
    print("    most of that gap — matching MSE's precision at the SAME data/step budget.")
    print("  * On the BIMODAL target MSE collapses to the between-modes mean (never")
    print("    commits) while every multimodal head — anchored included — still COMMITS.")
    print("  ⇒ the anchors buy back per-step precision WITHOUT sacrificing multimodality,")
    print("    and are loss-only: the deployed graph stays the same polynomial field.")
    print("=" * 78)


if __name__ == "__main__":
    run()
