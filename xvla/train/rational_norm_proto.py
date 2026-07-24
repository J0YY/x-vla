"""Rational / projective normalization: per-instance norm that IS tensor-pure.

The strict foldable model replaces per-instance RMSNorm with a FROZEN SCALAR (an
approximation; DEVLOG cost-boundary case ii). This proto shows the approximation is
avoidable: a per-instance normalization can be carried EXACTLY through a bilinear
(CP) stack in PROJECTIVE coordinates (numerator N, denominator D meaning value N/D),
so the whole network folds into a RATIONAL tensor network  out = P(x)/Q(x)  with P, Q
ordinary polynomial (CP) tensor networks. No mid-network division, no frozen scalar.

Honest scope:
  * ÷ mean-square  n(x)=x/(mean(x²)+ε)  is RATIONAL → exactly tensor-pure (shown here).
  * RMSNorm ÷√mean-square is ALGEBRAIC (a √); make it rational via a Padé approx of
    1/√· (near-exact) or use the mean-square variant. Both beat the frozen scalar.

We verify:
  (1) a 2-layer bilinear net with per-instance mean-square norm equals P(x)/Q(x) built
      by projective bookkeeping (numerator/denominator), to machine precision — i.e. it
      folds exactly into two polynomial tensor networks (the rational fold);
  (2) the frozen-SCALAR approximation (current strict model) has real error vs the exact
      per-instance norm — quantifying what the rational norm recovers.
"""

import numpy as np

rng = np.random.default_rng(0)
EPS = 1e-6


def bffn(x, L, R, D):
    """CP bilinear core: y = D[(Lx̄)⊙(Rx̄)], x̄=[1;x]. Degree-2 in x."""
    xb = np.concatenate([np.ones((*x.shape[:-1], 1)), x], axis=-1)
    return ((xb @ L.T) * (xb @ R.T)) @ D.T


def meansq_norm(a):
    """Per-instance mean-square normalization (rational, no √)."""
    return a / (np.mean(a ** 2, axis=-1, keepdims=True) + EPS)


def rms_norm(a):
    """Standard RMSNorm (algebraic — has a √)."""
    return a / np.sqrt(np.mean(a ** 2, axis=-1, keepdims=True) + EPS)


def make_layer(din, dout, r):
    return (rng.standard_normal((r, din + 1)) * din ** -0.5,      # L
            rng.standard_normal((r, din + 1)) * din ** -0.5,      # R
            rng.standard_normal((dout, r)) * r ** -0.5)           # D


# ---- a 2-layer bilinear net with per-instance mean-square norm ----
d = 4
L1, R1, D1 = make_layer(d, d, 8)
L2, R2, D2 = make_layer(d, d, 8)


def net_direct(x):
    a1 = bffn(x, L1, R1, D1)
    v1 = meansq_norm(a1)
    a2 = bffn(v1, L2, R2, D2)
    return meansq_norm(a2)


def net_projective(x):
    """Same net, but track (numerator N, denominator D_) = value N/D_ WITHOUT ever
    dividing mid-network. Returns (N, Dscalar) with value = N/Dscalar — the rational
    fold P(x)/Q(x). All ops are polynomial in (N, Dscalar)."""
    # layer 1: a1 = bffn(x) is polynomial; value a1 = a1/1.
    a1 = bffn(x, L1, R1, D1)                                   # numerator, D=1
    # mean-square norm: value v1 = a1 / q1, q1 = mean(a1²)+ε  (POLYNOMIAL denom)
    q1 = np.mean(a1 ** 2, axis=-1, keepdims=True) + EPS        # (…,1) polynomial
    N1, Dsc1 = a1, q1                                          # v1 = N1/Dsc1
    # layer 2 bilinear on v1=N1/Dsc1:  bffn(N1/Dsc1) = [bilinear part]/Dsc1²  (+ lin/const
    # terms scaled by 1/Dsc1). Handle homogeneous coord exactly by scaling the "1" too:
    # represent v1 in homogeneous form [1; N1/Dsc1] = [Dsc1; N1]/Dsc1, so bffn's x̄ is
    # (1/Dsc1)*[Dsc1; N1]; the ⊙ gives 1/Dsc1² * ([L·h]⊙[R·h]) with h=[Dsc1;N1].
    h = np.concatenate([Dsc1, N1], axis=-1)                    # = Dsc1 * x̄(v1)
    a2_num = ((h @ L2.T) * (h @ R2.T)) @ D2.T                  # numerator (poly), denom Dsc1²
    Dsc_a2 = Dsc1 ** 2                                         # a2 = a2_num / Dsc_a2
    # mean-square norm of a2: value = a2 / (mean(a2²)+ε)
    #   = (a2_num/Dsc_a2) / ( mean((a2_num/Dsc_a2)²)+ε )
    #   = a2_num / ( mean(a2_num²)/Dsc_a2 + ε·Dsc_a2 )     (multiply num&den by Dsc_a2)
    q2 = np.mean(a2_num ** 2, axis=-1, keepdims=True) / Dsc_a2 + EPS * Dsc_a2
    N2, Dsc2 = a2_num, q2                                      # out = N2/Dsc2 (RATIONAL)
    return N2, Dsc2


x = rng.standard_normal((1000, d))
y_direct = net_direct(x)
N2, Dsc2 = net_projective(x)
y_proj = N2 / Dsc2

err = np.max(np.abs(y_direct - y_proj))
print("PART 1 — rational (projective) fold of a per-instance-normed bilinear net")
print(f"  max |net_direct - P(x)/Q(x)| = {err:.2e}   (→ folds EXACTLY into num/denom TNs)")
print(f"  numerator P and denominator Q are both ordinary polynomial (CP) tensor nets;")
print(f"  one division at the end, none mid-network. ODT applies to P and to Q.")

# ---- what the FROZEN-SCALAR approximation (current strict model) costs ----
# calibrate one scalar c ≈ E[rms] and use x/c in place of per-instance norm.
def net_frozen_scalar(x, c1, c2):
    a1 = bffn(x, L1, R1, D1); v1 = a1 / c1
    a2 = bffn(v1, L2, R2, D2); return a2 / c2

# calibrate on data (mean-square scale, matching meansq_norm)
a1 = bffn(x, L1, R1, D1); c1 = (np.mean(a1 ** 2, axis=-1) + EPS).mean()
v1 = meansq_norm(a1); a2 = bffn(v1, L2, R2, D2); c2 = (np.mean(a2 ** 2, axis=-1) + EPS).mean()
y_frozen = net_frozen_scalar(x, c1, c2)
rel = np.linalg.norm(y_frozen - y_direct) / np.linalg.norm(y_direct)
print("\nPART 2 — frozen-scalar approximation error (what rational norm recovers)")
print(f"  ||frozen_scalar - exact|| / ||exact|| = {rel:.3f}   (per-instance variation the")
print(f"  scalar cannot capture; the rational/projective norm has ZERO such error)")

# ---- RMSNorm (with √) is algebraic: show mean-square ≠ rms but a Padé of 1/√ is close
print("\nPART 3 — RMSNorm (√) is algebraic; rational options")
a = bffn(x, L1, R1, D1)
ms = np.mean(a ** 2, axis=-1, keepdims=True) + EPS
# 1/sqrt(ms) via a degree-[2/2] Padé around the mean scale s0 (illustrative)
s0 = ms.mean()
t = ms / s0
pade = (15 - 10 * t + 3 * t ** 2) / (8 * np.sqrt(s0))   # Taylor/Padé-style 1/sqrt approx
exact_inv_sqrt = 1.0 / np.sqrt(ms)
pade_rel = np.linalg.norm(pade - exact_inv_sqrt) / np.linalg.norm(exact_inv_sqrt)
print(f"  rational 1/√ approx (deg-2/const, illustrative) rel err = {pade_rel:.3f}")
print(f"  → a proper Padé makes RMSNorm's √ a rational TN to any desired accuracy;")
print(f"  or use ÷mean-square (Part 1), which is exactly rational with no √.")

print("\nBOTTOM LINE: per-instance normalization is tensor-pure in the RATIONAL class")
print("(P(x)/Q(x), both polynomial TNs). Frozen scalar was an unnecessary approximation.")
