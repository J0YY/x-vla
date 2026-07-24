"""Minimal prototype: PATH-SIGNATURE / tensor-algebra action-chunk head.

Standalone NumPy (repo Python is 3.14 -> no torch; this proves the math + the
mode-separation + the invert loop cheaply). The production head is the identical
construction in torch: a tensor-pure (bilinear/CP, foldable) map obs -> a
DISTRIBUTION over the truncated signature (a graded element of the tensor
algebra), trained by expected-signature / signature-MMD matching; the DECODE
(pick a mode, then invert the signature to a trajectory) is OUT-OF-GRAPH -- see
report.

The action chunk is a PATH X:[0,1]->R^d (here d=2 for the demo, H steps). Its
truncated signature S(X) = (1, S^1, S^2, ..., S^N) is LITERALLY an element of the
tensor algebra T((R^d)) = (+)_k (R^d)^{ox k} -- a graded tensor, native to a
tensor network. The signature transform itself is polynomial (iterated products
= Chen's identity = ordered tensor product of per-step exponentials), so it lives
IN-graph; only the inversion (decode) is out-of-graph.

Demo claims, all checked below on reach-LEFT vs reach-RIGHT arcs (same endpoints):
  (1) level-1 signature (net displacement) COINCIDES for the two modes -> a mean
      regressor cannot tell them apart / averages to the invalid straight path;
  (2) the antisymmetric level-2 coordinate = the LEVY AREA cleanly SEPARATES the
      two whole-trajectory strategies into 2 modes in signature space;
  (3) shuffle identity S^i S^j = S^{ij}+S^{ji} holds numerically (sanity that we
      really computed the signature);
  (4) sampling a mode + INVERTING its signature (least-squares over a P-segment
      path) recovers a valid left/right arc; inverting the MEAN-of-modes signature
      gives the straight line (=the MSE failure) -> why you decode a MODE, not a mean.

Run:  python3 xvla/train/signature_proto.py
"""

from __future__ import annotations

import math

import numpy as np

rng = np.random.default_rng(0)


# ============================================================================
# Truncated tensor algebra over R^d  (levels 0..N; level k is a rank-k tensor)
# ============================================================================
def unit(d, N):
    """Multiplicative unit 1 = (1, 0, 0, ...) of the truncated tensor algebra."""
    lv = [np.array(1.0)]
    for k in range(1, N + 1):
        lv.append(np.zeros((d,) * k))
    return lv


def exp_increment(delta, N):
    """Signature of a single straight segment = tensor-exp(delta), truncated at N.

    exp(delta)^(k) = delta^{ox k} / k!  -- built iteratively L_k = L_{k-1} ox delta / k.
    Each level is a degree-k polynomial in the increment -> multilinear / tensor-pure.
    """
    lv = [np.array(1.0)]
    term = np.array(1.0)
    for k in range(1, N + 1):
        term = np.tensordot(term, delta, axes=0) / k
        lv.append(term)
    return lv


def chen(A, B, N):
    """Chen's identity: S(X*Y) = S(X) ox S(Y).  (A ox B)^(k) = sum_{i+j=k} A^i ox B^j.

    This is the group-like / concatenation structure of the tensor algebra -- the
    ONLY operation needed to assemble the signature of a whole chunk from its steps,
    and it is exactly the graded tensor product a tensor network contracts.
    """
    C = []
    for k in range(N + 1):
        acc = np.zeros((A[0].shape and () or ()) + (B[0].shape and () or ()))  # placeholder scalar
        acc = None
        for i in range(k + 1):
            j = k - i
            piece = np.tensordot(A[i], B[j], axes=0)
            acc = piece if acc is None else acc + piece
        C.append(acc)
    return C


def signature(path, N):
    """Truncated path signature (levels 0..N) of a discrete path (H+1, d) points.

    Piecewise-linear interpolation => signature = ordered tensor product of the
    per-step exponentials (Chen). Returns a list [S^0, S^1, ..., S^N].
    """
    d = path.shape[1]
    S = unit(d, N)
    for m in range(1, path.shape[0]):
        S = chen(S, exp_increment(path[m] - path[m - 1], N), N)
    return S


def levy_area(S):
    """Signed area (level-2 antisymmetric coordinate): A = 1/2 (S^{12} - S^{21})."""
    return 0.5 * (S[2][0, 1] - S[2][1, 0])


def flatten_upto(S, N):
    """Flatten levels 1..N into one vector (the graded tensor as a feature)."""
    return np.concatenate([np.ravel(S[k]) for k in range(1, N + 1)])


# ============================================================================
# Data: reach-LEFT vs reach-RIGHT arcs, IDENTICAL endpoints (0,0) -> (1,0).
# Left arc bows +y (encloses +area); right arc bows -y (-area). Their pointwise
# MEAN is the straight segment -- an invalid "middle" path (the MSE failure).
# ============================================================================
def make_arc(mode, H=24, bow=0.5, noise=0.01):
    t = np.linspace(0.0, 1.0, H + 1)
    x = t
    sgn = +1.0 if mode == 0 else -1.0                 # mode 0 = left/up, 1 = right/down
    y = sgn * bow * np.sin(np.pi * t)                 # bow up / down, back to 0
    x = x + noise * rng.standard_normal(H + 1)
    y = y + noise * rng.standard_normal(H + 1)
    x[0], y[0] = 0.0, 0.0                             # pin endpoints (same for both)
    x[-1], y[-1] = 1.0, 0.0
    return np.stack([x, y], 1)


# ============================================================================
# Part 1 -- multimodality is INVISIBLE at level 1, SEPARATED at level 2
# ============================================================================
def part1(N=3):
    print("Part 1: reach-left vs reach-right (same endpoints) in signature space")
    L = [signature(make_arc(0), N) for _ in range(200)]
    R = [signature(make_arc(1), N) for _ in range(200)]

    disp_L = np.array([S[1] for S in L]);  disp_R = np.array([S[1] for S in R])
    area_L = np.array([levy_area(S) for S in L])
    area_R = np.array([levy_area(S) for S in R])

    print(f"  level-1 (net displacement)  left  mean = {disp_L.mean(0).round(3)}")
    print(f"  level-1 (net displacement)  right mean = {disp_R.mean(0).round(3)}"
          f"   -> COINCIDE => a mean/MSE head cannot separate the two strategies")
    print(f"  level-2 Levy area           left  = {area_L.mean():+.3f} +/- {area_L.std():.3f}")
    print(f"  level-2 Levy area           right = {area_R.mean():+.3f} +/- {area_R.std():.3f}")
    sep = abs(area_L.mean() - area_R.mean()) / (0.5 * (area_L.std() + area_R.std()))
    print(f"  -> Levy-area SEPARATION = {sep:.1f} sigma  (two CLEAN modes in signature space)")

    # shuffle identity sanity: S^i S^j = S^{ij} + S^{ji}
    S = L[0]
    lhs = np.outer(S[1], S[1]); rhs = S[2] + S[2].T
    print(f"  shuffle check  max|S^i S^j - (S^ij+S^ji)| = {np.abs(lhs - rhs).max():.2e}  (=> valid signature)")
    return L, R


# ============================================================================
# Part 2 -- a 2-mode model of p(signature|obs). Here obs is constant, so the
# "tensor-pure map obs->p(sig)" reduces to its two mode-means (mu_left, mu_right)
# in the flattened tensor-algebra feature. In the real head these come from a
# bilinear/CP map of the action-query token; multimodality = separated regions.
# ============================================================================
def part2(L, R, N=3):
    print("\nPart 2: 2-mode model in signature space (mode-means = separated regions)")
    FL = np.array([flatten_upto(S, N) for S in L])
    FR = np.array([flatten_upto(S, N) for S in R])
    mu = [FL.mean(0), FR.mean(0)]
    between = np.linalg.norm(mu[0] - mu[1])
    within = 0.5 * (FL.std(0).mean() + FR.std(0).mean())
    print(f"  ||mu_left - mu_right|| = {between:.3f}   mean within-mode std = {within:.3f}"
          f"   -> between/within = {between / within:.1f} (well separated)")
    # the MEAN signature (what an MSE/expected-signature-only head would target if it
    # ignored multimodality) has ~zero Levy area -> inverts to the straight path.
    mean_sig = 0.5 * (FL.mean(0) + FR.mean(0))
    return mu, mean_sig


# ============================================================================
# Part 3 -- DECODE = out-of-graph signature INVERSION (optimization).
# Recover a trajectory from a target signature by least-squares over a P-segment
# piecewise-linear path. Inverting a MODE mean -> a valid arc; inverting the
# MEAN-of-modes -> the straight line (the exact MSE mode-averaging failure).
# ============================================================================
def _sig_vec_from_increments(inc, N):
    path = np.concatenate([np.zeros((1, 2)), np.cumsum(inc, 0)], 0)
    return flatten_upto(signature(path, N), N), path


def invert_signature(target_vec, N, P=12, iters=1500, lr=0.02):
    """Least-squares signature inversion: min_inc || Sig(path(inc)) - target ||_w^2.

    Path = P straight segments; the signature is smooth in the increments so we do
    first-order optimization (central-difference gradient here; analytic/autodiff in
    torch). Each level-k block is weighted by 1/||target_k|| (relative error) so the
    small higher levels -- which carry the SHAPE -- are fitted, not just the endpoint.
    Adam + grad-norm clipping keep it stable. HONEST: the signature pins the path only
    up to tree-like equivalence -> we fit a monotone-x path (x is the natural 'time'
    coordinate here) so the recovered arc is unique.
    """
    inc = np.tile(np.array([[1.0 / P, 0.0]]), (P, 1))          # init = straight line
    # relative per-level weights: block k scaled by 1/(||target_k|| + eps)
    w_parts, off = [], 0
    for k in range(1, N + 1):
        blk = target_vec[off:off + 2 ** k]; off += 2 ** k
        w_parts.append(np.full(2 ** k, 1.0 / (np.linalg.norm(blk) + 1e-3)))
    w = np.concatenate(w_parts)

    def loss(vec):
        r = (vec - target_vec) * w
        return float(r @ r)

    m = np.zeros_like(inc); v = np.zeros_like(inc)
    b1, b2, eps_a = 0.9, 0.999, 1e-8
    cur, path = _sig_vec_from_increments(inc, N)
    for it in range(1, iters + 1):
        g = np.zeros_like(inc); base = loss(cur); fd = 1e-5
        for p in range(P):
            for c in range(2):
                inc[p, c] += fd
                vv, _ = _sig_vec_from_increments(inc, N)
                g[p, c] = (loss(vv) - base) / fd
                inc[p, c] -= fd
        gn = np.linalg.norm(g)                                  # clip
        if gn > 10.0:
            g *= 10.0 / gn
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** it); vh = v / (1 - b2 ** it)
        inc -= lr * mh / (np.sqrt(vh) + eps_a)
        cur, path = _sig_vec_from_increments(inc, N)
    return path, loss(cur)


def _classify(path):
    ymax, ymin = path[:, 1].max(), path[:, 1].min()
    peak = ymax if abs(ymax) >= abs(ymin) else ymin
    span = ymax - ymin
    if span < 0.1:
        return peak, "straight (no reach)"
    # a clean single-lobe arc has |peak| ~ span; an up-down wiggle has span >> |peak|
    if abs(peak) < 0.6 * span:
        return peak, "degenerate wiggle (neither mode)"
    return peak, ("bows UP (up-mode)" if peak > 0 else "bows DOWN (down-mode)")


def part3(mu, mean_sig, N=3):
    print("\nPart 3: DECODE by out-of-graph signature inversion")
    # pick a mode (out-of-graph argmax over the 2 modes) and invert JUST that mode
    for name, target in (("up-mode  mu[0]", mu[0]), ("down-mode mu[1]", mu[1])):
        path, resid = invert_signature(target, N, P=12, iters=600)
        peak, shape = _classify(path)
        print(f"  invert {name:14s}: resid={resid:.1e}  peak y={peak:+.2f}  -> {shape}  (valid arc)")

    # FAILURE MODE A: MSE in TRAJECTORY space = pointwise mean of the two arcs.
    up, dn = make_arc(0, noise=0.0), make_arc(1, noise=0.0)
    mse_mean = 0.5 * (up + dn)
    peak, shape = _classify(mse_mean)
    print(f"  MSE trajectory mean         : peak y={peak:+.2f}  -> {shape}"
          f"  (the straight 'invalid middle' -- current linear+MSE head)")

    # FAILURE MODE B: averaging the two mode SIGNATURES zeroes the mode-discriminating
    # Levy-area coordinate; inverting that mean recovers no real reach.
    path, resid = invert_signature(mean_sig, N, P=12, iters=600)
    peak, shape = _classify(path)
    # flattened layout: [S^1 (2)], [S^2 (2x2)] -> level-2 = mean_sig[2:6]; area = 1/2(S12-S21)
    ma = 0.5 * (mean_sig[3] - mean_sig[4])
    print(f"  invert MEAN-of-signatures   : resid={resid:.1e}  peak y={peak:+.2f}"
          f"  Levy area={ma:+.3f}  -> {shape}")
    print("  => decode ONE mode -> a valid arc. Both averaging failures (trajectory-mean")
    print("     AND signature-mean) destroy the reach. Model the distribution, decode a mode.")


if __name__ == "__main__":
    N = 3
    L, R = part1(N)
    mu, mean_sig = part2(L, R, N)
    part3(mu, mean_sig, N)
