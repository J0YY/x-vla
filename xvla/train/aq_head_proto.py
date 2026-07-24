"""Minimal prototype: the ARM-PURPOSE-BUILT action head —
   COMONOTONE / AUTOREGRESSIVE MONOTONE-QUANTILE chunk head  (the "AQ head").

Standalone NumPy (repo Python is 3.14 -> no torch; this proves the math + the
training loop cheaply, exactly like quantile_proto.py / diffusion_proto.py). The
production head is the identical construction in torch, with the per-step quantile-map
coefficients emitted by a *bilinear/CP* (foldable) map of the chi-VLA PER-STEP
action-query token h_t(obs) — see the report in the DEVLOG.

MOTIVATION (DEVLOG 2026-07-18 cont.17). After the obs fix the LINEAR head reaches
25% closed-loop on LIBERO-Object; the pooled flow / product heads are 0%. The linear
head wins because the task is *conditionally near-unimodal*: given the exact frame the
correct action is nearly determined, so per-step PRECISION dominates. Flow pays a
sampling-noise cost (stochastic ODE sample on a near-unimodal conditional) and BOTH
pooled heads throw away per-step token resolution (vla.py: `h = aq_out.mean(1)`), which
the linear head keeps (`self.action_head(aq_out)` per token). The purpose-built head
must therefore: (1) keep the PER-STEP token h_t (precision); (2) DECODE DETERMINISTICALLY
on unimodal conditionals (no sampling noise); (3) still COMMIT to a whole-chunk mode on
the multimodal ones; (4) stay tensor-pure / foldable / ODT-able.

CONSTRUCTION.  Represent the policy by its conditional QUANTILE function, per step t and
per action dim d:

    a_{t,d} = Q_{t,d}(tau | h_t),        Q monotone increasing in tau in [0,1].

Q is a MONOTONE polynomial via the SOS-integral identity (the repo's degree-2 squaring
primitive; NO clamp / relu / softmax):

    s(u)     = sum_j c_j u^j                       (a free polynomial, c = affine in h_t)
    Q'(tau)  = eps + s(tau)^2  >= eps > 0
    Q(tau)   = b + eps*tau + int_0^tau s(u)^2 du,   w_m = q_m/(m+1) = c^T A_m c

so every quantile coefficient w_m is a fixed symmetric bilinear (degree-2) form in c;
and with c = C h_t (affine, foldable) it is a quadratic form  w_m = h_t^T (C^T A_m C) h_t
in the token = EXACTLY the Q_c object odt_interp.py diagonalizes.

THE ARM FIX — a SHARED tau across the whole chunk (comonotone coupling).  Draw ONE
scalar mode variable tau per chunk (out-of-graph) and evaluate every step at the SAME
tau. Because each Q_{t,d} is increasing in tau, low-tau selects every step's low branch
and high-tau its high branch: one scalar commits the WHOLE chunk to one coherent mode
(reach-left vs reach-right), decoded IN PARALLEL — no autoregressive chain, no root-find
(contrast Knothe-transport #1, which learns the inverse map and needs D bisections). The
shared tau IS the "low-dim mode latent". Per-step conditioning h_t keeps precision.

THREE OUT-OF-GRAPH DECODES (all controller ops, spec section 11), a menu:
  * AQ-sample : tau ~ U[0,1] shared         -> stochastic, covers modes (data-aug/expl).
  * AQ-MAP    : tau* = argmin_tau ||dT/dtau|| -> the densest point = the MAP mode.
                DETERMINISTIC. Unimodal -> the median = the mean (== linear head, no
                sampling noise). Multimodal -> lands ON the most-probable mode. This is
                the PRODUCTION decode and the reason it "can't be worse than linear".
  * factorized: tau_t independent per step   -> the WRONG baseline; mode-flips mid-chunk.

"CANNOT BE WORSE THAN LINEAR" (recovers linear precision on unimodal states).
The linear MSE head is the *exact* s->0, eps->0 special case:  Q_{t,d}(tau) -> b_{t,d}(h_t)
= (affine in h_t) = W_a h_t + b_a, independent of tau. So the AQ hypothesis class CONTAINS
the linear head. Under the strictly-proper pinball objective the population optimum on a
unimodal conditional is Q(tau)=mu(h)+sigma(h)*Phi^{-1}(tau); its AQ-MAP decode is the
median tau=0.5 -> mu(h) = the conditional mean = the linear head's target, DETERMINISTIC.
On a multimodal conditional the linear mean is an invalid between-modes action while
AQ-MAP lands on a real mode. => >= linear everywhere, strictly > on multimodal states.

This proto validates, on a chunked (H-step) target:
  PART 1  unimodal chunk  -> AQ-MAP RMSE ~= linear RMSE  <<  pooled-flow single-sample.
  PART 2  bimodal reach L/R chunk -> AQ commits whole-chunk (coherent, on a mode) while
          linear lands in the valley and the factorized variant mode-flips.
  PART 3  identities: fold (w_m=c^T A_m c), monotonicity (Q'>=0), s->0 == linear head.

Run:  python3 xvla/train/aq_head_proto.py
"""

from __future__ import annotations

import numpy as np

# The pooled-flow baseline's polynomial velocity field extrapolates explosively
# outside its training support; the out-of-graph decode clips it to [-8,8] (see
# PooledFlowHead.decode). Silence the benign transient-overflow warnings.
np.seterr(over="ignore", invalid="ignore")

rng = np.random.default_rng(0)


# ===========================================================================
# SOS-integral monotone quantile map  Q(tau) = b + eps*tau + int_0^tau s(u)^2 du
# (shared with quantile_proto.py; w_m = c^T A_m c is the folded bilinear map)
# ===========================================================================
EPS = 1e-3


def sos_integral_matrices(p: int):
    A = []
    for m in range(2 * p + 1):
        Am = np.zeros((p + 1, p + 1))
        for j in range(p + 1):
            l = m - j
            if 0 <= l <= p:
                Am[j, l] = 1.0 / (m + 1)
        A.append(0.5 * (Am + Am.T))
    return np.stack(A)                                    # (2p+1, p+1, p+1)


def q_coeffs_batch(c, Astack):
    """w_m = c^T A_m c.  c:(n,p+1) -> (n,2p+1)."""
    Ac = np.einsum("mij,nj->nmi", Astack, c)
    return np.einsum("nmi,ni->nm", Ac, c)


def dQ_dc_batch(c, Astack, tau):
    """dQ/dc.  c:(n,p+1), tau:(n,) -> (n,p+1). (excludes the eps*tau term: no c-dep)."""
    Ac = np.einsum("mij,nj->nmi", Astack, c)
    M = Astack.shape[0]
    powers = np.stack([tau ** (m + 1) for m in range(M)], -1)
    return 2.0 * np.einsum("nm,nmi->ni", powers, Ac)


def Q_batch(c, b, Astack, tau):
    """Q(tau)=b+eps*tau+sum_m w_m tau^{m+1}.  c:(n,p+1), b:(n,), tau:(n,) -> (n,)."""
    w = q_coeffs_batch(c, Astack)                         # (n,2p+1)
    M = Astack.shape[0]
    powers = np.stack([tau ** (m + 1) for m in range(M)], -1)
    return b + EPS * tau + np.sum(powers * w, -1)


class Adam:
    def __init__(self, shapes, lr=3e-2):
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
# The AQ head: per-step (t) monotone-quantile maps, coeffs AFFINE in the per-step
# token h_t = [1, obs] (here d_a=1; production uses the real d-dim aq_out[:,t]).
#   c_t(h) = C[t] @ h   (p+1) ;  b_t(h) = beta[t] . h
# ===========================================================================
class AQHead:
    def __init__(self, H, p, G):
        self.H, self.p, self.G = H, p, G
        self.A = sos_integral_matrices(p)
        # warm init: s == 1 (c_0=1) => Q = b + (1+eps)*tau, monotone linear.
        self.C = np.zeros((H, p + 1, G)); self.C[:, 0, 0] = 1.0
        self.beta = np.zeros((H, G))

    def coeffs(self, h, t):
        return h @ self.C[t].T, h @ self.beta[t]          # (n,p+1), (n,)

    def chunk(self, h, tau):
        """Evaluate all steps at a (per-sample) tau -> (n, H). tau:(n,) shared or per-step col."""
        n = h.shape[0]
        out = np.zeros((n, self.H))
        for t in range(self.H):
            c, b = self.coeffs(h, t)
            tt = tau[:, t] if tau.ndim == 2 else tau
            out[:, t] = Q_batch(c, b, self.A, tt)
        return out

    def train(self, sampler, iters=9000, bs=256, lr=3e-2):
        opt = Adam([self.C.shape, self.beta.shape], lr=lr)
        for _ in range(iters):
            h, a = sampler(bs)                              # h:(bs,G), a:(bs,H)
            tau = rng.random(bs)                            # ONE shared tau per sample
            gC = np.zeros_like(self.C); gbeta = np.zeros_like(self.beta)
            for t in range(self.H):
                c, b = self.coeffs(h, t)
                Q = Q_batch(c, b, self.A, tau)
                u = a[:, t] - Q
                dLdQ = -(tau - (u < 0).astype(float))      # pinball dL/dQ
                dQ = dQ_dc_batch(c, self.A, tau)           # (bs,p+1)
                gC[t] = (dLdQ[:, None, None] * (dQ[:, :, None] * h[:, None, :])).mean(0)
                gbeta[t] = (dLdQ[:, None] * h).mean(0)
            [self.C, self.beta] = opt.step([self.C, self.beta], [gC, gbeta])

    # ---- decodes (all OUT-OF-GRAPH controller ops) ------------------------
    def decode_sample(self, h):
        return self.chunk(h, rng.random(h.shape[0]))       # shared tau ~ U

    def decode_factorized(self, h):
        return self.chunk(h, rng.random((h.shape[0], self.H)))   # independent tau_t

    def decode_median(self, h):
        return self.chunk(h, np.full(h.shape[0], 0.5))           # deterministic median

    def decode_map(self, h, K=400):
        """DETERMINISTIC MAP-mode: tau* = argmin_tau ||dT/dtau|| (the densest point)."""
        n = h.shape[0]
        grid = np.linspace(0.0, 1.0, K)
        # curve T(tau) for every obs: (n, H, K)
        T = np.zeros((n, self.H, K))
        for t in range(self.H):
            c, b = self.coeffs(h, t)
            for kk, g in enumerate(grid):
                T[:, t, kk] = Q_batch(c, b, self.A, np.full(n, g))
        speed = np.sqrt(((np.diff(T, axis=2)) ** 2).sum(1))        # (n, K-1) chunk speed
        k = 1 + speed[:, 1:-1].argmin(1)                            # interior argmin
        tau_star = grid[k]                                          # (n,)
        return self.chunk(h, tau_star), tau_star


# ===========================================================================
# Baseline heads.
# ===========================================================================
class LinearHead:
    """Per-step MSE head a_t = W[t] . h  (the current chi-VLA linear head)."""
    def __init__(self, H, G):
        self.H, self.W = H, np.zeros((H, G))

    def train(self, sampler, iters=9000, bs=256, lr=5e-2):
        opt = Adam([self.W.shape], lr=lr)
        for _ in range(iters):
            h, a = sampler(bs)
            pred = h @ self.W.T                              # (bs,H)
            g = ((pred - a)[:, :, None] * h[:, None, :]).mean(0)
            [self.W] = opt.step([self.W], [g])

    def decode(self, h):
        return h @ self.W.T


def poly_feats(a, obs, deg_a, m_obs):
    """Monomials of the chunk a (m dims) up to total deg_a, times obs^0..m_obs.
    a:(n,m), obs:(n,) -> (n,F)."""
    import itertools
    m = a.shape[1]
    exps = [e for total in range(deg_a + 1)
            for e in _weak_compositions(total, m)]
    cols = []
    for e in exps:
        base = np.ones(a.shape[0])
        for v in range(m):
            if e[v]:
                base = base * a[:, v] ** e[v]
        for k in range(m_obs + 1):
            cols.append(base * obs ** k)
    return np.stack(cols, -1)


def _weak_compositions(total, n):
    if n == 1:
        yield (total,); return
    for first in range(total + 1):
        for rest in _weak_compositions(total - first, n - 1):
            yield (first,) + rest


class PooledFlowHead:
    """Tensor-pure flow-matching over the whole chunk (diffusion_proto.py style),
    conditioned on POOLED obs — mirrors the repo's `h = aq_out.mean(1)` pooling."""
    def __init__(self, H, deg_a=3, m_obs=2):
        self.H, self.deg_a, self.m_obs = H, deg_a, m_obs

    def train(self, sampler, iters=6000, bs=2048):
        # infer F
        h0, a0 = sampler(4)
        F = poly_feats(a0, h0[:, 1], self.deg_a, self.m_obs).shape[1]
        Spp = np.zeros((F, F)); Sup = np.zeros((self.H, F))
        for _ in range(iters // 8):
            h, a1 = sampler(bs)
            obs = h[:, 1]
            a0 = rng.standard_normal((bs, self.H))
            t = rng.uniform(0, 1, bs)
            at = (1 - t)[:, None] * a0 + t[:, None] * a1
            u = a1 - a0
            P = poly_feats(at, obs, self.deg_a, self.m_obs)
            Spp += P.T @ P; Sup += u.T @ P
        self.W = np.linalg.solve(Spp + 1e-3 * np.eye(F), Sup.T).T
        return self

    def decode(self, h, steps=80, n_samples=1):
        n = h.shape[0]; obs = h[:, 1]
        outs = []
        for _ in range(n_samples):
            a = rng.standard_normal((n, self.H)); dt = 1.0 / steps
            for s in range(steps):
                t = np.full(n, s * dt)
                v = poly_feats(a, obs, self.deg_a, self.m_obs) @ self.W.T
                a = np.clip(a + dt * v, -8.0, 8.0)          # out-of-graph controller clamp
            outs.append(a)
        return np.mean(outs, 0) if n_samples > 1 else outs[0]


# ===========================================================================
# PART 1 -- unimodal chunk: precision. Target ramp mu_t(obs) = obs*(t+1)/H, noise .1.
#   The correct action is (near-)determined by obs -> per-step precision is what wins.
# ===========================================================================
def part1():
    print("=" * 78)
    print("PART 1  UNIMODAL chunk (ramp mu_t=obs*(t+1)/H, noise .1) -> PRECISION test")
    print("=" * 78)
    H, G, p = 4, 2, 4
    ramp = (np.arange(H) + 1) / H

    def sampler(n):
        obs = rng.uniform(-1, 1, n)
        h = np.stack([np.ones(n), obs], -1)
        a = obs[:, None] * ramp[None, :] + 0.10 * rng.standard_normal((n, H))
        return h, a

    lin = LinearHead(H, G); lin.train(sampler)
    flow = PooledFlowHead(H).train(sampler)
    aq = AQHead(H, p, G); aq.train(sampler, iters=15000)

    # eval vs the TRUE conditional mean (the ideal precise action)
    obs = rng.uniform(-1, 1, 4000)
    h = np.stack([np.ones_like(obs), obs], -1)
    mu = obs[:, None] * ramp[None, :]                        # (n,H) ideal
    def rmse(pred): return np.sqrt(((pred - mu) ** 2).mean())
    lin_r = rmse(lin.decode(h))
    flow_r = rmse(flow.decode(h, n_samples=1))
    flow_r16 = rmse(flow.decode(h, n_samples=16))            # averaged (re-collapses!)
    aq_med_r = rmse(aq.decode_median(h))
    aq_map, _ = aq.decode_map(h)
    aq_r = rmse(aq_map)
    aq_samp_r = rmse(aq.decode_sample(h))
    print(f"  per-step RMSE to the ideal conditional mean (lower = more precise):")
    print(f"    linear (MSE)           : {lin_r:.4f}   <- the 25%-closed-loop head")
    print(f"    pooled-flow  1 sample  : {flow_r:.4f}   <- sampling noise HURTS (cont.17)")
    print(f"    pooled-flow 16 avg     : {flow_r16:.4f}   (averaging = re-collapse, not free)")
    print(f"    AQ  tau~U (sample)     : {aq_samp_r:.4f}   (stochastic, has spread too)")
    print(f"    AQ  median (tau=0.5)   : {aq_med_r:.4f}   <- deterministic, ~= linear")
    print(f"    AQ  MAP (argmin speed) : {aq_r:.4f}   <- unified decode (~= median here)")
    ok = aq_med_r <= lin_r * 2.5 and aq_med_r < 0.25 * flow_r
    print(f"  => AQ deterministic decode ~ linear precision & >>beats flow-1-sample: {ok}")


# ===========================================================================
# PART 2 -- bimodal reach L/R chunk: whole-chunk commitment + temporal coherence.
#   Whole chunk is trajectory L (neg ramp) OR R (pos ramp), 50/50 at ambiguous obs;
#   obs tilts the mix.  A per-step head that samples INDEPENDENTLY mode-flips.
# ===========================================================================
def part2():
    print()
    print("=" * 78)
    print("PART 2  BIMODAL reach L/R chunk -> COMMITMENT + TEMPORAL COHERENCE")
    print("=" * 78)
    H, G, p = 4, 2, 6
    ramp = 1.5 * (np.arange(H) + 1) / H

    def sampler(n):
        obs = rng.uniform(-1, 1, n)
        h = np.stack([np.ones(n), obs], -1)
        pR = np.clip(0.5 + 0.5 * obs, 0.05, 0.95)           # obs tilts L/R mix
        goR = rng.random(n) < pR
        sign = np.where(goR, 1.0, -1.0)
        a = sign[:, None] * ramp[None, :] + 0.20 * rng.standard_normal((n, H))
        return h, a

    lin = LinearHead(H, G); lin.train(sampler)
    flow = PooledFlowHead(H).train(sampler)
    aq = AQHead(H, p, G); aq.train(sampler, iters=16000)

    def metrics(chunks):
        """min chunk-dist to the L or R trajectory (precision-on-a-mode), whole-chunk
        sign coherence (all steps agree), and committed magnitude (mean |a|)."""
        L = -ramp; R = ramp
        dL = np.sqrt(((chunks - L) ** 2).sum(1)); dR = np.sqrt(((chunks - R) ** 2).sum(1))
        dist = np.minimum(dL, dR).mean()
        sgn = np.sign(chunks)
        coherent = (np.abs(sgn.sum(1)) == H).mean()          # all H steps agree
        mag = np.abs(chunks).mean()                          # commitment magnitude
        return dist, coherent, mag

    print(f"  (true mode trajectory |a| = {np.abs(ramp).mean():.2f}; valley/collapse |a|~0)")
    for obs_val, lab in [(0.0, "AMBIGUOUS 50/50"), (0.6, "tilted -> R (p=0.8)")]:
        n = 4000
        h = np.tile([1.0, obs_val], (n, 1))
        print(f"  obs={obs_val:+.2f} ({lab}):")
        d, c, mg = metrics(lin.decode(h))
        print(f"    linear (MSE)          : dist {d:.2f}  coherent {c:.2f}  |a| {mg:.2f}"
              f"  (VALLEY: |a|~0, no commit)")
        d, c, mg = metrics(flow.decode(h, n_samples=1))
        print(f"    pooled-flow 1 sample  : dist {d:.2f}  coherent {c:.2f}  |a| {mg:.2f}")
        cf = aq.decode_factorized(h); d, c, mg = metrics(cf)
        print(f"    AQ factorized (tau_t) : dist {d:.2f}  coherent {c:.2f}  |a| {mg:.2f}"
              f"  <- MODE-FLIPS")
        cs = aq.decode_sample(h); d, c, mg = metrics(cs)
        frac_R = (cs.sum(1) > 0).mean()
        print(f"    AQ shared-tau sample  : dist {d:.2f}  coherent {c:.2f}  |a| {mg:.2f}"
              f"  frac->R {frac_R:.2f}")
        cm, _ = aq.decode_map(h); d, c, mg = metrics(cm)
        print(f"    AQ MAP (deterministic): dist {d:.2f}  coherent {c:.2f}  |a| {mg:.2f}"
              f"  <- commits w/ magnitude")


# ===========================================================================
# PART 3 -- tensor-purity identities: fold, monotonicity, and s->0 == linear head.
# ===========================================================================
def part3():
    print()
    print("=" * 78)
    print("PART 3  identities: fold (w=c^T A c), monotonicity (Q'>=0), s->0 == linear")
    print("=" * 78)
    p = 4; A = sos_integral_matrices(p)
    c = rng.standard_normal((1, p + 1)); b = np.array([0.3])
    # fold check: bilinear coeffs reproduce direct integration of s^2
    grid = np.linspace(0, 1, 40001)
    s = sum(c[0, j] * grid ** j for j in range(p + 1))
    Q_num = b[0] + EPS * grid + np.cumsum(s * s) * (grid[1] - grid[0])
    Q_alg = Q_batch(np.repeat(c, grid.size, 0), np.repeat(b, grid.size),
                    A, grid)
    print(f"  FOLD  max|Q_bilinear - Q_integrated| = {np.max(np.abs(Q_alg - Q_num)):.2e}"
          f"   (w_m = c^T A_m c exact)")
    print(f"  MONOTONE  min Q'(tau) = eps + min s^2 = {EPS + (s*s).min():.4e}  (>0 always)")
    # w_m = h^T (C^T A_m C) h : the odt Q_c object. eigendecompose one.
    G = 3; C = rng.standard_normal((p + 1, G))
    Qc = C.T @ A[2] @ C                                       # (G,G) symmetric
    ev = np.linalg.eigvalsh(Qc)
    print(f"  ODT   w_2(h)=h^T Q_c h, Q_c=C^T A_2 C symmetric; eigenvalues "
          f"{np.round(ev,3).tolist()}  (signed atoms, odt_interp.py)")
    # s->0, eps->0 : Q -> b(h) == the linear head
    c0 = np.zeros((5, p + 1)); bb = rng.standard_normal(5)
    tau = rng.random(5)
    Q0 = Q_batch(c0, bb, A, tau)
    print(f"  LIMIT max|Q_{{s=0}}(tau) - b(h)| = {np.max(np.abs(Q0 - (bb + EPS*tau))):.2e}"
          f"  and eps*tau<= {EPS:.0e}  => Q -> b(h) = W_a h (the LINEAR head, exactly)")


if __name__ == "__main__":
    part1()
    part2()
    part3()
