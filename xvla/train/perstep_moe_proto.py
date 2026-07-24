"""Minimal prototype: PER-TIMESTEP-CONDITIONED tensor-pure multimodal action head.

Standalone NumPy (the repo's Python is 3.14 -> no torch; this proves the math and
the training loop cheaply, exactly like ``tree_moe_proto.py`` / ``diffusion_proto.py``).
The production head is the identical construction in torch: every gate/expert is a
``BilinearFFN`` (CP core, ``xvla/nn/bilinear.py``) that folds to its own degree-2
tensor (``dense_core`` -> the object ``xvla/train/odt.py`` diagonalises). The only
non-polynomial op (the routing sign, the ODE decode) lives OUT-OF-GRAPH in the
controller -- the sanctioned category (spec Section 11).

WHY (DEVLOG 2026-07-18 cont.17).  The obs 180-mirror fix lifted the LINEAR head to
25% closed-loop on LIBERO-Object, but the two tensor-pure multimodal heads (flow,
product) stayed at 0% despite committing the gripper offline (0.98). PRIME SUSPECT:
both multimodal heads POOL the H action-query tokens ``h = aq_out.mean(1)`` to ONE
vector and regenerate the WHOLE flattened chunk (H*d_a) from it -- whereas the linear
head maps EACH token h_t to its own action a_t = W h_t, preserving per-step
conditioning. Pooling was chosen to keep the chunk mode-COHERENT (avoid per-step
mode-flipping), but it throws away the per-step precision closed-loop control needs.

THE FIX (this file).  Factor the chunk distribution as

    p(chunk | obs) = p(z | obs) * PROD_t p(a_t | h_t, z)

  * z  -- a SHARED chunk-level mode latent (discrete bits here; a shared continuous
          noise seed in the flow variant, Part 2). Chosen ONCE per chunk from POOLED
          obs -> temporal mode-coherence, no per-step mode-flipping.
  * a_t -- each per-step conditional reads its OWN token h_t (like the linear head)
          -> per-step precision.

So the head recovers the linear head's per-step precision AND keeps genuine
multimodality. The DISCRETE realisation is product-routing (Eq. P) made per-step:

    z_g   = 1[ Gate_g( mean_t h_t ) > 0 ]                 g = 1..G   (chunk-level)
    a_t   = c0(h_t) + SUM_g (2 z_g - 1) * c_g(h_t)        t = 1..H   (per-step)

vs the CURRENT (pooled) product head:  a = c0(hbar) + SUM_g (2z_g-1) c_g(hbar),
c* : R^d -> R^{H*d_a}, hbar = mean_t h_t.

WHAT THIS PROTO SHOWS (bimodal CHUNK target with per-step obs detail):
  Part 1 (discrete).  Three heads on the same data:
     - linear per-step   : gets per-step detail, COLLAPSES the mode (|amp|->0).
     - pooled product    : COMMITS the mode, BLURS per-step detail (only mean h).
     - per-step product  : COMMITS *and* keeps per-step precision -> wins both.
  Part 2 (continuous).  Same story with a flow field (gripper-like scalar action):
     a shared noise seed z picks the mode for all steps (coherent); the per-step
     field reads h_t -> per-step precision. Beats the pooled flow on per-step MSE.
  Part 3.  Fold check: each CP core == its dense degree-2 core (odt.py-ready).

Run:  python3 xvla/train/perstep_moe_proto.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


# =========================================================================== #
# CP (bilinear) core -- the repo's degree-2 primitive (xvla/nn/bilinear.py).
#   y = D[ (L h_bar) (.) (R h_bar) ],   h_bar = [1; h]   (homogeneous coord)
# Folds to a dense tensor T[o,i,j] = sum_r D[o,r] L[r,i] R[r,j]  (dense_core),
# i.e. y_o = sum_ij T[o,i,j] h_bar_i h_bar_j -- exactly what odt.py consumes.
# The SAME core is applied to every token h_t (weight-tied over t), exactly like
# the linear head is one nn.Linear applied to all H action-query tokens.
# =========================================================================== #
class CP:
    def __init__(self, d_in, d_out, rank=12, spread=0.0, scale=0.4):
        self.d_in, self.d_out, self.r = d_in, d_out, rank
        s = (d_in + 1) ** -0.5
        self.L = rng.standard_normal((rank, d_in + 1)) * s
        self.R = rng.standard_normal((rank, d_in + 1)) * s
        self.D = rng.standard_normal((d_out, rank)) * (rank ** -0.5) * scale
        self.L[:, 0] += spread                      # foldable bias-channel spread
        self._adam = {k: [np.zeros_like(getattr(self, k)), np.zeros_like(getattr(self, k))]
                      for k in ("L", "R", "D")}
        self._t = 0

    def _bar(self, h):
        return np.concatenate([np.ones((h.shape[0], 1)), h], 1)

    def forward(self, h):
        z = self._bar(h)
        u = z @ self.L.T
        v = z @ self.R.T
        p = u * v
        y = p @ self.D.T
        self._cache = (z, u, v, p)
        return y

    def backward(self, dy):
        z, u, v, p = self._cache
        gD = dy.T @ p
        dp = dy @ self.D
        du, dv = dp * v, dp * u
        gL = du.T @ z
        gR = dv.T @ z
        return {"L": gL, "R": gR, "D": gD}

    def step(self, grads, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k, g in grads.items():
            w = getattr(self, k)
            m, v = self._adam[k]
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** self._t)
            vh = v / (1 - b2 ** self._t)
            w -= lr * mh / (np.sqrt(vh) + eps)

    def dense_core(self):
        T = np.einsum("or,ri,rj->oij", self.D, self.L, self.R)
        return 0.5 * (T + T.transpose(0, 2, 1))

    def n_params(self):
        return self.L.size + self.R.size + self.D.size


# =========================================================================== #
# DATA -- a genuinely-multimodal CHUNK with per-step obs detail.
#
# obs is split into H per-step SLOTS o_t in R^c (obs = [o_1..o_H]).  A single
# SHARED binary mode s in {-1,+1} is drawn per chunk with obs-dependent (but
# residually stochastic) probability -> the chunk is bimodal but the mode is
# mostly predictable from POOLED obs (the DEVLOG "conditionally near-unimodal"
# regime).  The target trajectory is
#
#     a_t = s * amp_t   +   M @ o_t   +  noise            (t = 1..H)
#           \-------/       \------/
#         shared mode     per-step detail  (depends on slot o_t and on t)
#
# The ACTION-QUERY TOKEN for step t is  h_t = Wv @ o_t + pos_t : it carries its OWN
# slot o_t prominently (what a causal transformer produces).  Hence:
#   * mean_t h_t = Wv @ mean_t(o_t) + mean(pos)  -> LOSES the individual o_t, so a
#     whole-chunk head fed only the mean CANNOT reconstruct M @ o_t per step;
#   * the per-step head reads h_t ~ Wv @ o_t  -> recovers M @ o_t exactly.
# Both can recover the mode s from POOLED obs (it depends on the obs average).
# =========================================================================== #
class ChunkData:
    def __init__(self, H=8, c=3, d_a=2, d=16, mode_temp=9.0, noise=0.05):
        self.H, self.c, self.d_a, self.d = H, c, d_a, d
        self.noise = noise
        self.mode_temp = mode_temp
        tt = np.arange(H)
        # per-step mode shape amp_t (the SHARED mode's per-step signature)
        self.amp = np.stack([np.cos(2 * np.pi * tt / H),
                             np.sin(2 * np.pi * tt / H)], 1)[:, :d_a]      # (H,d_a)
        self.M = rng.standard_normal((d_a, c)) * 0.9                       # detail map
        self.Wv = rng.standard_normal((d, c)) * (c ** -0.5)               # slot->token
        self.pos = rng.standard_normal((H, d)) * 0.5                      # per-step pos
        # mode depends ONLY on the POOLED obs (mean over slots) so the chunk-level
        # gate can read it; the per-step DETAIL depends on each slot o_t (pooling
        # destroys that) -> commit is a pooled quantity, precision is a per-step one.
        self.w_s = rng.standard_normal(c)                                # mode logit dir (per-coord)

    def batch(self, n):
        o = rng.standard_normal((n, self.H, self.c))                      # (n,H,c) slots
        o_pool = o.mean(1)                                                # (n,c) pooled obs
        p_plus = np.clip(sigmoid(self.mode_temp * (o_pool @ self.w_s)
                                 / np.sqrt(self.c)), 0.08, 0.92)
        s = np.where(rng.random(n) < p_plus, 1.0, -1.0)                   # (n,) shared
        detail = np.einsum("ac,nhc->nha", self.M, o)                      # (n,H,d_a)
        y = s[:, None, None] * self.amp[None] + detail
        y = y + self.noise * rng.standard_normal(y.shape)
        # tokens h_t = Wv @ o_t + pos_t
        h = np.einsum("dc,nhc->nhd", self.Wv, o) + self.pos[None]         # (n,H,d)
        map_s = np.where(p_plus > 0.5, 1.0, -1.0)                         # MAP mode
        return h, y, s, map_s


# --------------------------------------------------------------------------- #
# Metrics.  We separate the two capabilities the DEVLOG conflated:
#   * COMMIT  -- does the head pick the correct shared mode (not average it)?
#   * PER-STEP PRECISION -- how well does it reconstruct the chunk given the mode?
# --------------------------------------------------------------------------- #
def mode_projection(chunk, amp):
    """Recover the scalar mode coefficient s_hat = <chunk-detail-free, amp>/||amp||^2.
    We use the raw chunk projected on amp (detail is ~orthogonal on average)."""
    a = amp.reshape(-1)
    return (chunk.reshape(chunk.shape[0], -1) @ a) / (a @ a)


def report(name, dec_greedy, dec_oracle, y, s, amp, npar):
    """dec_greedy : head's own decode (its mode choice).  Measures COMMIT.
       dec_oracle : head decoded with the TRUE mode forced.  Isolates PER-STEP MSE."""
    shat = mode_projection(dec_greedy, amp)
    commit = float((np.sign(shat) == s).mean())
    amp_mag = float(np.abs(shat).mean())                 # ~1 commits, ~0 collapses
    full_mse = float(((dec_greedy - y) ** 2).mean())
    step_mse = float(((dec_oracle - y) ** 2).mean())     # oracle mode -> pure per-step
    print(f"    {name:24s} params={npar:6d}  commit={commit:.3f}  "
          f"|amp|={amp_mag:.2f}  perstep_MSE(oracle)={step_mse:.4f}  "
          f"greedy_MSE={full_mse:.4f}")
    return commit, step_mse


# =========================================================================== #
# HEAD 0 -- LINEAR per-step head (the current chi-VLA head, spec section 11).
# a_t = W [1;h_t].  Per-step precision, but NO latent -> averages the mode.
# =========================================================================== #
def train_linear(data, n=40000):
    h, y, _, _ = data.batch(n)
    B = h.shape[0]
    X = np.concatenate([np.ones((B, data.H, 1)), h], -1).reshape(B * data.H, -1)
    Y = y.reshape(B * data.H, -1)
    W = np.linalg.lstsq(X, Y, rcond=None)[0]                              # (d+1,d_a)

    def decode(h_):
        b = h_.shape[0]
        Xt = np.concatenate([np.ones((b, data.H, 1)), h_], -1)
        return np.einsum("bhi,ia->bha", Xt, W)
    npar = W.size
    return decode, decode, npar   # no mode -> greedy == oracle


# =========================================================================== #
# HEAD 1 -- POOLED product routing (the CURRENT multimodal head, vla.py).
# hbar = mean_t h_t;  experts c0/c_g : R^d -> R^{H*d_a}; gate : R^d -> R^G.
# a = c0(hbar) + sum_g (2z_g-1) c_g(hbar).  Commits (shared z) but only sees the
# POOLED vector -> cannot reconstruct per-step detail M@o_t.
# =========================================================================== #
def train_pooled_product(data, G=1, iters=6000, lr=3e-3, n=40000):
    d, m = data.d, data.H * data.d_a
    c = [CP(d, m, spread=0.0)] + [CP(d, m, spread=0.0) for _ in range(G)]
    gates = [CP(d, 1) for _ in range(G)]
    combos = ((np.arange(2 ** G)[:, None] >> np.arange(G)) & 1)
    signs = (2 * combos - 1).astype(float)                               # (2^G,G)
    H, y_all, s_all, _ = data.batch(n)
    hbar_all = H.mean(1)                                                  # (n,d)
    for _ in range(iters):
        idx = rng.integers(0, n, 512)
        hb, y = hbar_all[idx], y_all[idx].reshape(len(idx), m)
        c0 = c[0].forward(hb)
        cg = np.stack([c[g + 1].forward(hb) for g in range(G)], 1)       # (B,G,m)
        alla = c0[:, None, :] + np.einsum("kg,bgm->bkm", signs, cg)      # (B,2^G,m)
        errs = ((alla - y[:, None, :]) ** 2).mean(-1)
        mstar = errs.argmin(1)
        sstar = signs[mstar]                                             # (B,G)
        bstar = combos[mstar]
        pred = c0 + np.einsum("bg,bgm->bm", sstar, cg)
        resid = (2.0 / m) * (pred - y) / len(idx)
        c[0].step(c[0].backward(resid), lr=lr)
        for g in range(G):
            c[g + 1].step(c[g + 1].backward(sstar[:, g:g + 1] * resid), lr=lr)
        for g in range(G):
            gs = gates[g].forward(hb)[:, 0]
            gg = (sigmoid(gs) - bstar[:, g])[:, None] / len(idx)
            gates[g].step(gates[g].backward(gg), lr=lr)

    def _decode(h_, force_s=None):
        hb = h_.mean(1)
        c0 = c[0].forward(hb)
        cg = np.stack([c[g + 1].forward(hb) for g in range(G)], 1)
        if force_s is None:
            z = np.stack([(gates[g].forward(hb)[:, 0] > 0) for g in range(G)], 1)
            sg = (2 * z - 1).astype(float)
        else:                                     # oracle: G=1 mode forced to true s
            sg = np.tile(force_s[:, None], (1, G))
        a = c0 + np.einsum("bg,bgm->bm", sg, cg)
        return a.reshape(h_.shape[0], data.H, data.d_a)
    npar = sum(x.n_params() for x in c) + sum(g.n_params() for g in gates)
    return (lambda h_: _decode(h_)), (lambda h_, s: _decode(h_, s)), npar


# =========================================================================== #
# HEAD 2 -- PER-STEP product routing with a SHARED chunk-level gate  (THE FIX).
# gate reads POOLED obs -> shared z (chunk-level, coherent); experts c0/c_g are
# applied PER TOKEN (weight-tied over t, like the linear head), so each step reads
# its OWN h_t -> per-step precision.
#     z_g = 1[Gate_g(mean_t h_t) > 0]
#     a_t = c0(h_t) + sum_g (2z_g-1) c_g(h_t)
# =========================================================================== #
def train_perstep_product(data, G=1, iters=6000, lr=3e-3, n=40000):
    d, d_a, H = data.d, data.d_a, data.H
    c = [CP(d, d_a, spread=0.0)] + [CP(d, d_a, spread=0.0) for _ in range(G)]
    gates = [CP(d, 1) for _ in range(G)]
    combos = ((np.arange(2 ** G)[:, None] >> np.arange(G)) & 1)
    signs = (2 * combos - 1).astype(float)
    Hb_all, y_all, s_all, _ = data.batch(n)
    for _ in range(iters):
        idx = rng.integers(0, n, 512)
        ht, y = Hb_all[idx], y_all[idx]                                  # (B,H,d), (B,H,d_a)
        B = len(idx)
        hbar = ht.mean(1)                                                # (B,d) pooled
        flat = ht.reshape(B * H, d)
        c0 = c[0].forward(flat).reshape(B, H, d_a)                       # per-step
        cg = np.stack([c[g + 1].forward(flat).reshape(B, H, d_a)
                      for g in range(G)], 1)                             # (B,G,H,d_a)
        # predicted chunk for every sign-combo: (B, 2^G, H, d_a)
        alla = c0[:, None] + np.einsum("kg,bghd->bkhd", signs, cg)
        errs = ((alla - y[:, None]) ** 2).mean((-1, -2))                # (B,2^G)
        mstar = errs.argmin(1)
        sstar = signs[mstar]                                            # (B,G)
        bstar = combos[mstar]
        pred = c0 + np.einsum("bg,bghd->bhd", sstar, cg)               # (B,H,d_a)
        resid = (2.0 / (H * d_a)) * (pred - y) / B                     # (B,H,d_a)
        c[0].forward(flat)                                             # refresh cache
        c[0].step(c[0].backward(resid.reshape(B * H, d_a)), lr=lr)
        for g in range(G):
            c[g + 1].forward(flat)
            dyg = (sstar[:, g][:, None, None] * resid).reshape(B * H, d_a)
            c[g + 1].step(c[g + 1].backward(dyg), lr=lr)
        for g in range(G):                                             # gate on POOLED
            gs = gates[g].forward(hbar)[:, 0]
            gg = (sigmoid(gs) - bstar[:, g])[:, None] / B
            gates[g].step(gates[g].backward(gg), lr=lr)

    def _decode(h_, force_s=None):
        B = h_.shape[0]
        hbar = h_.mean(1)
        flat = h_.reshape(B * H, d)
        c0 = c[0].forward(flat).reshape(B, H, d_a)
        cg = np.stack([c[g + 1].forward(flat).reshape(B, H, d_a) for g in range(G)], 1)
        if force_s is None:
            z = np.stack([(gates[g].forward(hbar)[:, 0] > 0) for g in range(G)], 1)
            sg = (2 * z - 1).astype(float)
        else:
            sg = np.tile(force_s[:, None], (1, G))
        return c0 + np.einsum("bg,bghd->bhd", sg, cg)
    npar = sum(x.n_params() for x in c) + sum(g.n_params() for g in gates)
    return (lambda h_: _decode(h_)), (lambda h_, s: _decode(h_, s)), npar


# =========================================================================== #
# PART 1 -- discrete: linear vs pooled-product vs per-step-product.
# =========================================================================== #
def part1():
    print("Part 1: bimodal CHUNK, per-step obs detail. commit vs per-step precision.")
    print("        (a_t = s*amp_t + M@o_t; token h_t carries its OWN slot o_t)")
    data = ChunkData(H=8, c=3, d_a=2, d=16)
    te_h, te_y, te_s, _ = data.batch(6000)

    gd, od, npar = train_linear(data)
    report("linear per-step (current)", gd(te_h), od(te_h), te_y, te_s, data.amp, npar)

    gd, od, npar = train_pooled_product(data, G=1)
    report("pooled product (current)", gd(te_h), od(te_h, te_s), te_y, te_s, data.amp, npar)

    gd, od, npar = train_perstep_product(data, G=1)
    report("per-step product (FIX)", gd(te_h), od(te_h, te_s), te_y, te_s, data.amp, npar)
    print("  => linear keeps detail but COLLAPSES the mode (|amp|~0, commit~.5);")
    print("     pooled product COMMITS but blurs detail (high perstep_MSE);")
    print("     per-step product COMMITS *and* has the low per-step MSE. Both win.")


# =========================================================================== #
# PART 2 -- continuous: per-step FLOW sharing ONE noise seed (mode-coherent) vs a
# whole-chunk pooled flow. Gripper-like scalar action (d_a=1). The velocity is
# bilinear -- polynomial in (a, tau) with coefficients LINEAR in the conditioning
# token -- so it folds like a BilinearFFN; only the Euler ODE loop is out-of-graph.
#   pooled flow : v(a in R^H, tau; hbar)   -- one field over the whole chunk.
#   per-step    : v(a_t, tau; h_t, z)      -- per-step field; z = shared noise seed
#                 drawn ONCE per chunk (broadcast to all t) -> temporal coherence.
# =========================================================================== #
def poly_feats_scalar(a, tau):
    """psi(a,tau) monomials for a scalar action a and time tau (up to deg 3 in a)."""
    return np.stack([np.ones_like(a), a, a ** 2, a ** 3,
                     tau, tau * a, tau * a ** 2, tau ** 2, tau ** 2 * a], -1)


def train_perstep_flow(data, iters=9000, bs=2048):
    """Per-step flow: v_t = W(h_t) . psi(a_t,tau).  W linear in [1;h_t] -> bilinear
    (foldable). z (shared noise seed) enters as one extra conditioning channel so the
    SAME seed picks the same mode for every step. Closed-form ridge (flow-matching)."""
    d, H = data.d, data.H
    Fp = 9                                            # poly feats
    # coefficient tensor: v = sum_f ([1;h_t;z] . theta_f) psi_f  ; params linear.
    ctx = 1 + d + 1                                   # [1 ; h_t ; z]
    Spp = np.zeros((ctx * Fp, ctx * Fp))
    Sup = np.zeros(ctx * Fp)
    for _ in range(iters):
        h, y, s, _ = data.batch(bs)                  # y:(bs,H,d_a); use d_a col 0
        yt = y[..., 0]                               # (bs,H) gripper-like scalar
        z = s.copy()                                  # shared mode seed (== noise sign)
        a0 = z[:, None] * np.abs(rng.standard_normal((bs, H))) * 0.0 + z[:, None]
        a0 = a0 + 0.15 * rng.standard_normal((bs, H))  # seeded noise (mode from z)
        tau = rng.uniform(0, 1, (bs, H))
        at = (1 - tau) * a0 + tau * yt
        u = yt - a0                                   # velocity target
        psi = poly_feats_scalar(at, tau)             # (bs,H,Fp)
        ctxv = np.concatenate([np.ones((bs, H, 1)), h,
                               np.broadcast_to(z[:, None, None], (bs, H, 1))], -1)
        feat = (ctxv[..., :, None] * psi[..., None, :]).reshape(bs, H, ctx * Fp)
        F = feat.reshape(bs * H, ctx * Fp)
        Spp += F.T @ F
        Sup += F.T @ u.reshape(-1)
    theta = np.linalg.solve(Spp + 1e-2 * np.eye(ctx * Fp), Sup)

    def decode(h_, force_s=None, steps=30, independent=False):
        B = h_.shape[0]
        # SHARED seed z per chunk (out-of-graph draw) -> same mode for every step
        # (temporal coherence). `independent`: a fresh seed per step (the naive
        # per-step-flow that FLIPS modes) -> shown incoherent for contrast.
        if force_s is not None:
            zt = np.tile(force_s[:, None], (1, H))
        elif independent:
            zt = np.sign(rng.standard_normal((B, H)))
        else:
            zt = np.tile(np.sign(rng.standard_normal((B, 1))), (1, H))
        a = zt + 0.15 * rng.standard_normal((B, H))
        ctxv = np.concatenate([np.ones((B, H, 1)), h_, zt[..., None]], -1)
        dt = 1.0 / steps
        for k in range(steps):
            tau = np.full((B, H), k * dt)
            psi = poly_feats_scalar(a, tau)
            feat = (ctxv[..., :, None] * psi[..., None, :]).reshape(B, H, ctx * Fp)
            v = feat @ theta
            a = a + dt * v
        return a
    npar = theta.size
    return decode, npar


def train_pooled_flow(data, iters=9000, bs=2048):
    """Pooled flow: ONE field over the whole H-vector conditioned on hbar=mean h."""
    d, H = data.d, data.H
    # simple per-coord poly (deg-3 in own coord) with coeffs linear in [1;hbar;z].
    Fp = 9
    ctx = 1 + d + 1
    # tie coords: same coefficient tensor applied per output coord but conditioned
    # only on the POOLED vector (no per-step token) -> the information bottleneck.
    Spp = np.zeros((ctx * Fp, ctx * Fp))
    Sup = np.zeros(ctx * Fp)
    for _ in range(iters):
        h, y, s, _ = data.batch(bs)
        yt = y[..., 0]
        hbar = h.mean(1)                              # (bs,d) pooled
        z = s.copy()
        a0 = z[:, None] + 0.15 * rng.standard_normal((bs, H))
        tau = rng.uniform(0, 1, (bs, H))
        at = (1 - tau) * a0 + tau * yt
        u = yt - a0
        psi = poly_feats_scalar(at, tau)
        ctxv = np.concatenate([np.ones((bs, 1)), hbar, z[:, None]], -1)  # (bs,ctx)
        ctxv = np.broadcast_to(ctxv[:, None, :], (bs, H, ctx))
        feat = (ctxv[..., :, None] * psi[..., None, :]).reshape(bs, H, ctx * Fp)
        F = feat.reshape(bs * H, ctx * Fp)
        Spp += F.T @ F
        Sup += F.T @ u.reshape(-1)
    theta = np.linalg.solve(Spp + 1e-2 * np.eye(ctx * Fp), Sup)

    def decode(h_, force_s=None, steps=30):
        B = h_.shape[0]
        hbar = h_.mean(1)
        z = force_s if force_s is not None else np.sign(rng.standard_normal(B))
        a = z[:, None] + 0.15 * rng.standard_normal((B, H))
        ctxv = np.concatenate([np.ones((B, 1)), hbar, z[:, None]], -1)
        ctxv = np.broadcast_to(ctxv[:, None, :], (B, H, ctx))
        dt = 1.0 / steps
        for k in range(steps):
            tau = np.full((B, H), k * dt)
            psi = poly_feats_scalar(a, tau)
            feat = (ctxv[..., :, None] * psi[..., None, :]).reshape(B, H, ctx * Fp)
            v = feat @ theta
            a = a + dt * v
        return a
    npar = theta.size
    return decode, npar


def _coherence_mse(dec_chunk, te_y, te_s, amp, **kw):
    """MSE of a natural (non-oracle) decode to the NEAREST valid single-mode chunk.
    The two valid chunks are the true one (mode s) and its opposite (mode -s):
    {te_y, te_y - 2 s amp}. A mode-COHERENT decode matches one of them (low MSE);
    an incoherent decode (modes flipped across steps) matches NEITHER (high MSE)."""
    a = amp[:, 0]
    chunk_s = te_y[..., 0]
    chunk_ns = chunk_s - 2 * te_s[:, None] * a[None, :]
    e_s = ((dec_chunk - chunk_s) ** 2).mean(1)
    e_ns = ((dec_chunk - chunk_ns) ** 2).mean(1)
    return float(np.minimum(e_s, e_ns).mean())


def part2():
    print("\nPart 2: continuous FLOW. A SHARED noise seed z picks the mode for every")
    print("        step (coherent); the per-step field reads its OWN h_t (precise).")
    print("        Gripper-like scalar action. per-step field vs pooled (mean h) field.")
    data = ChunkData(H=8, c=3, d_a=1, d=16, noise=0.05)
    te_h, te_y, te_s, _ = data.batch(4000)
    ytg = te_y[..., 0]

    dec, npar = train_pooled_flow(data, iters=2500, bs=1024)
    o = dec(te_h, force_s=te_s)
    coh = _coherence_mse(dec(te_h), te_y, te_s, data.amp)
    print(f"    pooled flow (current)    params={npar:5d}  "
          f"perstep_MSE(oracle)={((o - ytg) ** 2).mean():.4f}  "
          f"coherence_MSE(shared seed)={coh:.4f}")

    dec, npar = train_perstep_flow(data, iters=2500, bs=1024)
    o = dec(te_h, force_s=te_s)
    coh_shared = _coherence_mse(dec(te_h), te_y, te_s, data.amp)
    coh_indep = _coherence_mse(dec(te_h, independent=True), te_y, te_s, data.amp)
    print(f"    per-step flow (FIX)      params={npar:5d}  "
          f"perstep_MSE(oracle)={((o - ytg) ** 2).mean():.4f}  "
          f"coherence_MSE(shared seed)={coh_shared:.4f}")
    print(f"    per-step flow, INDEP seed per step (naive)                    "
          f"           coherence_MSE={coh_indep:.4f}  (flips modes)")
    print("  => per-step field >> pooled field on per-step MSE; the SHARED seed keeps")
    print("     the chunk on ONE mode (low coherence_MSE) where independent seeds flip")
    print("     modes step-to-step (high coherence_MSE = matches no single mode).")


# =========================================================================== #
# PART 3 -- tensor-purity fold check (odt.py-ready degree-2 core).
# =========================================================================== #
def part3():
    print("\nPart 3: tensor-purity -- each CP expert/gate folds to a dense degree-2")
    print("        core; CP forward == dense-core forward (odt.py object).")
    e = CP(16, 2, rank=12)
    h = rng.standard_normal((5, 16))
    y_cp = e.forward(h)
    hb = np.concatenate([np.ones((5, 1)), h], 1)
    T = e.dense_core()
    y_dense = np.einsum("oij,bi,bj->bo", T, hb, hb)
    print(f"    max|CP - dense_core| = {np.abs(y_cp - y_dense).max():.2e}  "
          f"(exact fold; the per-step experts stay in the tensor-network class)")


if __name__ == "__main__":
    part1()
    part2()
    part3()
