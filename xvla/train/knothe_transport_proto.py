"""Minimal prototype: monotone triangular transport (Knothe-Rosenblatt) action head.

Standalone NumPy/SciPy (the repo's Python is 3.14 -> no torch; this proves the math
and the training loop cheaply). The production head is the identical construction in
torch, with the transport-map coefficients emitted by a *linear/bilinear* (foldable)
map of the chi-VLA action-query token h(obs) -- see report. Here `obs` is a small
context so the whole thing runs in seconds.

Represents  p(a | obs)  as the pushforward of a standard Gaussian eta through an
INCREASING TRIANGULAR map. We learn the INVERSE map S : a -> z (data -> reference):

    z_k = S_k(a_1, ..., a_k ; obs),      d S_k / d a_k > 0   for all k.

Because S is lower-triangular, its Jacobian is triangular and

    p(a|obs) = eta(S(a;obs)) * |det grad S| = eta(S(a;obs)) * prod_k dS_k/da_k .

MONOTONICITY IS TENSOR-PURE BY CONSTRUCTION (the SOS-integral trick): we force the
diagonal derivative to be a squared polynomial + floor,

    dS_k/da_k = eps + h_k(a_{1:k-1}, a_k ; obs)^2  >=  eps > 0,

so S_k(a_{1:k-1}, a_k) = b_k(a_{1:k-1};obs) + eps*a_k + int_0^{a_k} h_k(.,t)^2 dt .

The integral of a squared polynomial is a polynomial -> S_k is polynomial in a_k of
odd degree 2p+1 with a positive leading term (monotone increasing), for ANY value of
the coefficients: monotonicity needs NO constraint, it rides on the degree-2 squaring
primitive the repo already has (BilinearFFN). obs (and the causal history a_{1:k-1})
enter ONLY as the coefficients c(.) , b(.) -- exactly how born_mps cores / energy tau
are conditioned.

Multimodality comes from the SHAPE of the map: a monotone 1-D map pushing N(0,1) to a
bimodal density exists + is unique (increasing rearrangement); it is flat over the
low-density valley and steep at the modes. Sharp separation costs polynomial degree
(honest limit, printed below).

  * training   = EXACT maximum likelihood; the log-det is the ONLY log and it lives
                 ONLY in the loss; training evaluates S (never inverts).
  * decode     = sample z~N(0,I), invert triangularly: D one-dimensional monotone
                 root-finds (bisection) -- an OUT-OF-GRAPH controller op.

Run:  python3 xvla/train/knothe_transport_proto.py
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(0)

EPS = 1e-3          # invertibility floor: dS/da >= EPS  (strict monotonicity)
P_DEG = 3          # integrand degree p  -> S_k has degree 2p+1 in a_k
Q_DEG = 2          # conditioning polynomial degree (in a_{1:k-1}, obs)


# ---------------------------------------------------------------------------
# polynomial features psi(u): all monomials of u up to total degree Q_DEG.
# ---------------------------------------------------------------------------
def _mono_exponents(n_vars, q):
    exps = []
    for total in range(q + 1):
        for combo in itertools.combinations_with_replacement(range(n_vars), total):
            e = np.zeros(n_vars, dtype=int)
            for i in combo:
                e[i] += 1
            exps.append(e)
    return np.array(exps) if n_vars > 0 else np.zeros((1, 0), dtype=int)


def psi(u, exps):                     # u: (B, n_vars) -> (B, F)
    if u.shape[1] == 0:
        return np.ones((u.shape[0], 1))
    # (B, F): prod_v u[:,v]^exps[f,v]
    return np.prod(u[:, None, :] ** exps[None, :, :], axis=2)


# ---------------------------------------------------------------------------
# one monotone triangular component  S_k(a_{1:k-1}, a_k ; obs)
#   coefficients c(u) = W @ psi(u)  (shape p+1) ; offset b(u) = v @ psi(u)
#   u = [a_{1:k-1}, obs]
# ---------------------------------------------------------------------------
_POW = np.arange(P_DEG + 1)
_EXP = _POW[:, None] + _POW[None, :] + 1          # (p+1,p+1): m+m'+1


def component(a_k, u, W, v, exps):
    """Return S_k (B,) and diagonal derivative dS_k/da_k (B,)."""
    F = psi(u, exps)                                # (B,F)
    c = F @ W.T                                     # (B, p+1)  integrand coeffs
    b = F @ v                                       # (B,)      offset
    # h(a_k) = sum_m c_m a_k^m
    apw = a_k[:, None] ** _POW[None, :]            # (B, p+1)
    h = (c * apw).sum(1)                            # (B,)
    diag = EPS + h * h                              # dS/da_k >= EPS
    # integral_0^{a_k} h^2 dt = sum_{m,m'} c_m c_m' a_k^{m+m'+1}/(m+m'+1)
    outer = c[:, :, None] * c[:, None, :]          # (B, p+1, p+1)
    apow = a_k[:, None, None] ** _EXP[None]        # (B, p+1, p+1)
    integ = (outer * apow / _EXP[None]).sum((1, 2))
    S_k = b + EPS * a_k + integ
    return S_k, diag


# ---------------------------------------------------------------------------
# the full triangular map: pack/unpack params, NLL, and inverse sampling.
# ---------------------------------------------------------------------------
class KRMap:
    def __init__(self, D, obs_dim):
        self.D = D
        self.obs_dim = obs_dim
        self.exps = []                             # per-component monomial table
        self.shapes = []
        for k in range(D):
            n_vars = k + obs_dim                    # a_{1:k-1} (k of them) + obs
            e = _mono_exponents(n_vars, Q_DEG)
            self.exps.append(e)
            F = e.shape[0]
            self.shapes.append(((P_DEG + 1, F), (F,)))   # W, v

    def init_params(self):
        parts = []
        for (Wsh, vsh) in self.shapes:
            W = 0.1 * rng.standard_normal(Wsh)
            W[0] += 1.0                              # c_0 ~ 1  -> dS/da ~ 1 (near identity)
            parts.append(W.ravel())
            parts.append(np.zeros(vsh))
        return np.concatenate(parts)

    def _unpack(self, theta):
        out, i = [], 0
        for (Wsh, vsh) in self.shapes:
            nW = int(np.prod(Wsh)); nv = int(np.prod(vsh))
            W = theta[i:i + nW].reshape(Wsh); i += nW
            v = theta[i:i + nv]; i += nv
            out.append((W, v))
        return out

    def _uk(self, a, obs, k):                       # conditioning vars for comp k
        cols = [a[:, :k]] if k > 0 else []
        if self.obs_dim > 0:
            cols.append(obs)
        return np.concatenate(cols, axis=1) if cols else np.zeros((a.shape[0], 0))

    def nll(self, theta, a, obs, l2=1e-4):
        params = self._unpack(theta)
        B = a.shape[0]
        quad = np.zeros(B); logdet = np.zeros(B)
        for k in range(self.D):
            W, v = params[k]
            S_k, diag = component(a[:, k], self._uk(a, obs, k), W, v, self.exps[k])
            quad += S_k ** 2
            logdet += np.log(diag)
        # NLL = 1/2 ||S||^2 - sum log dS_k/da_k   (+ const); log ONLY here (the loss)
        return (0.5 * quad - logdet).mean() + l2 * (theta @ theta)

    def fit(self, a, obs, iters=400):
        theta0 = self.init_params()
        res = minimize(self.nll, theta0, args=(a, obs), method="L-BFGS-B",
                       options={"maxiter": iters})
        self.theta = res.x
        return res

    def diag_min(self, a, obs):                      # monotonicity certificate
        params = self._unpack(self.theta)
        m = np.inf
        for k in range(self.D):
            W, v = params[k]
            _, diag = component(a[:, k], self._uk(a, obs, k), W, v, self.exps[k])
            m = min(m, diag.min())
        return m

    def sample(self, z, obs):                        # inverse: solve S(a)=z triangularly
        """z: (B, D) reference draws; obs: (B, obs_dim). OUT-OF-GRAPH bisection."""
        params = self._unpack(self.theta)
        B = z.shape[0]
        a = np.zeros((B, self.D))
        for k in range(self.D):
            W, v = params[k]
            uk = self._uk(a, obs, k)

            def S_of(ak):                            # monotone increasing in ak
                Sk, _ = component(ak, uk, W, v, self.exps[k])
                return Sk
            lo = np.full(B, -6.0); hi = np.full(B, 6.0)
            # widen bracket until it straddles z_k (map may need > |6|)
            for _ in range(20):
                need = S_of(lo) > z[:, k]; lo[need] -= 6.0
                need = S_of(hi) < z[:, k]; hi[need] += 6.0
            for _ in range(60):                       # bisection to ~1e-16 bracket
                mid = 0.5 * (lo + hi)
                go_hi = S_of(mid) < z[:, k]
                lo = np.where(go_hi, mid, lo)
                hi = np.where(go_hi, hi, mid)
            a[:, k] = 0.5 * (lo + hi)
        return a


# ===========================================================================
# Part 1 -- fit a BIMODAL 2-D target with a monotone triangular polynomial map.
#   target = 1/2 N(m_-, s) + 1/2 N(m_+, s),  m_+-=(+-1.5,+-1.5) (diagonal blobs).
#   Bimodal along the diagonal => the map must COUPLE a_1 and a_2 (triangular).
# ===========================================================================
def part1():
    n = 4000
    sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    mu = np.stack([1.5 * sign, 1.5 * sign], 1)
    a = mu + 0.35 * rng.standard_normal((n, 2))
    obs = np.zeros((n, 0))                            # unconditional

    m = KRMap(D=2, obs_dim=0)
    res = m.fit(a, obs, iters=600)
    print(f"  [part1] fit: final NLL = {res.fun:.3f}  (converged={res.success})")
    print(f"  [part1] monotonicity certificate: min_k min dS_k/da_k over data = "
          f"{m.diag_min(a, obs):.4f}  (>0 required; floor EPS={EPS})")

    # ancestral inverse sampling: z ~ N(0,I) -> a = S^{-1}(z)
    z = rng.standard_normal((n, 2))
    s = m.sample(z, np.zeros((n, 0)))

    # mode recovery: assign each sample to nearest true mode; both must be populated
    d_minus = np.linalg.norm(s - np.array([-1.5, -1.5]), axis=1)
    d_plus = np.linalg.norm(s - np.array([1.5, 1.5]), axis=1)
    to_minus = d_minus < d_plus
    frac_minus = to_minus.mean()
    err = np.minimum(d_minus, d_plus).mean()
    print(f"  [part1] pushforward samples: mode(-) share={frac_minus:.2f} "
          f"mode(+) share={1 - frac_minus:.2f}  (target 0.50/0.50) "
          f"mean dist-to-nearest-mode={err:.3f}")
    # the MSE head emits the single conditional MEAN -> the origin, between the modes
    mse_pred = a.mean(0)
    mse_miss = min(np.linalg.norm(mse_pred - np.array([-1.5, -1.5])),
                   np.linalg.norm(mse_pred - np.array([1.5, 1.5])))
    print(f"  [part1] MSE-head prediction = mean = {np.round(mse_pred, 2)} "
          f"-> dist to NEAREST true mode = {mse_miss:.2f} (lands in the empty valley)")


# ===========================================================================
# Part 2 -- synthetic multimodal REACH: obs selects a target; ambiguous obs is
#   genuinely bimodal p(a|obs). Action a in R^2 = 2-D reach endpoint.
#   obs<0 -> target A=(-1.2,+1.2);  obs>0 -> target B=(+1.2,-1.2);
#   |obs|<0.25 -> 50/50 (genuine multimodality). Continuous multimodal manifold.
# ===========================================================================
TA = np.array([-1.2, 1.2]); TB = np.array([1.2, -1.2])


def _reach_batch(n):
    obs = rng.uniform(-1, 1, (n, 1))
    o = obs[:, 0]
    pick_B = o > 0
    amb = np.abs(o) < 0.25
    pick_B[amb] = rng.random(amb.sum()) < 0.5         # bimodal middle
    tgt = np.where(pick_B[:, None], TB, TA)
    a = tgt + 0.2 * rng.standard_normal((n, 2))
    return a, obs


def part2():
    a, obs = _reach_batch(5000)
    m = KRMap(D=2, obs_dim=1)
    res = m.fit(a, obs, iters=600)
    print(f"  [part2] fit: final NLL = {res.fun:.3f}  (converged={res.success})")
    print(f"  [part2] monotonicity certificate: min dS_k/da_k = "
          f"{m.diag_min(a, obs):.4f}  (>0 required)")

    for obs_val, label in [(-0.7, "unambiguous A"), (0.7, "unambiguous B"),
                           (0.0, "AMBIGUOUS (bimodal)")]:
        nz = 2000
        z = rng.standard_normal((nz, 2))
        ov = np.full((nz, 1), obs_val)
        s = m.sample(z, ov)
        dA = np.linalg.norm(s - TA, axis=1); dB = np.linalg.norm(s - TB, axis=1)
        pA = (dA < dB).mean()
        err = np.minimum(dA, dB).mean()
        # a single committed decode: one z-sample commits to ONE mode (not the mean)
        one = m.sample(rng.standard_normal((1, 2)), np.array([[obs_val]]))[0]
        print(f"  [part2] obs={obs_val:+.2f} ({label}): share->A={pA:.2f} "
              f"share->B={1 - pA:.2f}  mean dist-to-nearest-target={err:.3f} | "
              f"one committed sample a={np.round(one, 2)}")
    # MSE baseline at the ambiguous obs: emits the mean of the two targets = origin
    mid = 0.5 * (TA + TB)
    print(f"  [part2] MSE head at ambiguous obs emits mean(A,B)={np.round(mid, 2)} "
          f"-> dist to nearest target = {min(np.linalg.norm(mid - TA), np.linalg.norm(mid - TB)):.2f} "
          f"(commits to NEITHER; the transport head commits to one).")


if __name__ == "__main__":
    print("Part 1: bimodal 2-D target as a monotone triangular polynomial map")
    part1()
    print("\nPart 2: multimodal reach; ambiguous obs is genuinely bimodal p(a|obs)")
    part2()
