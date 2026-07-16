"""ODT *interpretation* — actually extracting and testing mechanisms (M5 → M7).

The M5 code (``odt.py``) proves the χ-MLP exports exactly to a tensor network and
that its hidden bonds are globally low-rank (truncation curves). It stops there:
it never extracts the mechanisms or asks whether they are *human-coherent* or
*causal*. This module closes that gap, following spec §16.2 Level A / §17 and
directly extending Dehérand (2026), who showed bilinear *convolutional* eigen-
spectra are spatially coherent but explicitly left two things open:

  (a) the interpretation was only run on a shallow single-bilinear layer;
  (b) visual coherence was never tied to a *causal* intervention.

Two things live here:

1. **Output-conditioned eigendecomposition** (Dehérand's u^T Q_u x form).
   For a single-bilinear χ-classifier the class logit is *exactly* a quadratic
   form in the homogeneous embedded input:

       logit_c(x) = z^T Q_c z + b_c ,   z = [1; E x] ,
       Q_c = Σ_o W_head[c,o] C[o]                    (head folded into the core)

   Eigendecomposing Q_c = Σ_i λ_i v_i v_i^T gives

       logit_c(x) = Σ_i λ_i (v_i·z)^2 + b_c ,

   signed eigen-"atoms": λ_i>0 supports class c, λ_i<0 suppresses it. Each v_i
   projects back through E to a pixel pattern (spec §17 "visual atoms").

2. **Faithfulness battery** (spec §17). Whether the ranking is causal, not just
   suggestive: keep top-r |λ| terms, project the top-r out, project a *random*
   r out (control), and amplify a +λ atom — measuring predicted vs observed
   accuracy / logit change. Random-vs-ranked is the key control Dehérand lacked.
"""

from __future__ import annotations

import torch

from xvla.train.odt import _homog, _fold_head_into_last


# --------------------------------------------------------------------------- #
# 1. Output-conditioned decomposition of a single-bilinear χ-classifier
# --------------------------------------------------------------------------- #

@torch.no_grad()
def class_quadratics(cores, head):
    """Return (Q, b) for a **single-layer** χ-MLP: logit_c(x) = z^T Q[c] z + b[c].

    Q has shape (num_classes, d+1, d+1) (symmetric), b shape (num_classes,).
    z = [1; E x] is the homogeneous embedded input. Only valid for len(cores)==1.
    """
    assert len(cores) == 1, "class_quadratics is the exact quadratic form of a 1-layer χ-MLP"
    Q = _fold_head_into_last(cores, head)             # (C, d+1, d+1), head folded in
    Q = 0.5 * (Q + Q.transpose(1, 2))                 # symmetrize (harmless; C already sym)
    _, bh = head
    return Q, bh.double()


@torch.no_grad()
def eig_atoms(Q_c):
    """Eigendecompose one class quadratic Q_c → (lambdas, vecs), |λ| descending.

    ``vecs[:, i]`` is the i-th eigen-atom in homogeneous z=[1;Ex] space.
    """
    evals, evecs = torch.linalg.eigh(Q_c)             # ascending, real symmetric
    order = evals.abs().argsort(descending=True)
    return evals[order], evecs[:, order]


@torch.no_grad()
def quad_logits(Q, b, x, embed):
    """Evaluate logit_c = z^T Q_c z + b_c over a batch x (exact, fp64)."""
    We, be = embed
    z = _homog(x.flatten(1).double() @ We.T.double() + be.double())   # (B, d+1)
    # z^T Q_c z for every class: einsum over the batch and the two z legs.
    return torch.einsum("bi,cij,bj->bc", z, Q, z) + b


@torch.no_grad()
def atom_to_pixels(v, embed, in_shape, pixel_std):
    """Project an eigen-atom v (d+1) to a pixel-space sensitivity map (spec §17).

    The factor is (v·z)^2 with z=[1; E x]; its linear sensitivity to a *pixel* is
    (E^T v[1:]) / std (chain rule through the input normalization). Reshaped to
    the image grid. Returned raw (not yet display-normalized).
    """
    We, _ = embed
    g = We.T.double() @ v[1:].double()                # (in_dim,) sensitivity in x-space
    g = g / pixel_std.flatten().double()              # de-normalize to pixel space
    return g.reshape(in_shape)


# --------------------------------------------------------------------------- #
# 2. Coherence metric — is an atom spatially localized? (vs random control)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def locality(pattern, frac=0.1):
    """Energy concentration: fraction of total L2 energy in the top-``frac`` pixels.

    1.0 → all energy in a few pixels (localized stroke); ~frac → spread out
    (diffuse). Averaged over channels. Compare an atom's value to a random
    direction's (~frac) to judge coherence quantitatively, not by eye.
    """
    e = (pattern ** 2).sum(0) if pattern.dim() == 3 else pattern ** 2   # per-pixel energy
    e = e.flatten()
    k = max(1, int(frac * e.numel()))
    top = torch.topk(e, k).values.sum()
    return (top / e.sum().clamp_min(1e-30)).item()


# --------------------------------------------------------------------------- #
# 3. Faithfulness battery — is the spectral ranking causal? (spec §17)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _acc(logits, y):
    return (logits.argmax(-1) == y).float().mean().item()


@torch.no_grad()
def faithfulness(Q, b, x, y, embed, ranks, rng=None):
    """Per-class quadratic interventions; returns accuracy under each.

    For each class its Q_c is eigendecomposed (|λ| desc). Then, at every r in
    ``ranks`` we rebuild every class's quadratic three ways and classify by
    argmax over the intervened logits:

      * ``keep``          : keep only the top-r |λ| eigen-terms of every Q_c.
      * ``drop_top``      : zero the top-r eigen-terms (project the ranked
                            mechanisms *out*).
      * ``drop_random``   : zero r *randomly chosen* eigen-terms (the control).

    If ``keep`` stays high at small r and ``drop_top`` collapses while
    ``drop_random`` barely moves, the ranking is causally load-bearing — the
    intervention Dehérand's visual eigenspectra never demonstrated.
    """
    C = Q.shape[0]
    We, be = embed
    z = _homog(x.flatten(1).double() @ We.T.double() + be.double())    # (B, d+1)
    n = Q.shape[1]

    # Pre-eigendecompose each class quadratic.
    lam, vec = [], []
    for c in range(C):
        l, v = eig_atoms(Q[c])
        lam.append(l); vec.append(v)

    def logits_from(masks):
        """masks[c] is a (n,) 0/1 weight over class c's eigen-terms."""
        out = z.new_zeros(z.shape[0], C)
        for c in range(C):
            proj = (z @ vec[c]) ** 2                    # (B, n) squared coords
            out[:, c] = (proj * (lam[c] * masks[c])).sum(-1) + b[c]
        return out

    res = {"keep": {}, "drop_top": {}, "drop_random": {}}
    ones = [torch.ones(n, device=z.device, dtype=torch.float64) for _ in range(C)]
    for r in ranks:
        keep = [torch.zeros(n, device=z.device, dtype=torch.float64) for _ in range(C)]
        for c in range(C):
            keep[c][:r] = 1.0
        res["keep"][r] = _acc(logits_from(keep), y)

        drop = [ones[c].clone() for c in range(C)]
        for c in range(C):
            drop[c][:r] = 0.0
        res["drop_top"][r] = _acc(logits_from(drop), y)

        randm = [ones[c].clone() for c in range(C)]
        for c in range(C):
            idx = torch.randperm(n, generator=rng, device=z.device)[:r]
            randm[c][idx] = 0.0
        res["drop_random"][r] = _acc(logits_from(randm), y)
    return res


@torch.no_grad()
def amplify_test(Q, b, x, y, embed, target_class, scale=3.0):
    """Amplify the top +λ atom of ``target_class`` and check the predicted effect.

    Doubling(-scaling) the leading positive eigenvalue of Q_c should raise class
    c's logit on inputs that already express that atom, i.e. shift predictions
    toward c. Returns the fraction of samples whose argmax moves to c and the
    mean change in the class-c logit — a directed, predicted-vs-observed check.
    """
    We, be = embed
    z = _homog(x.flatten(1).double() @ We.T.double() + be.double())
    base = quad_logits(Q, b, x, embed)
    l, v = eig_atoms(Q[target_class])
    pos = (l > 0).nonzero().flatten()
    if len(pos) == 0:
        return {"target": target_class, "note": "no positive eigenvalue"}
    top = v[:, pos[0]]
    delta = (scale - 1.0) * l[pos[0]] * (z @ top) ** 2         # added to logit_c
    amp = base.clone(); amp[:, target_class] += delta
    moved_to = ((base.argmax(-1) != target_class) &
                (amp.argmax(-1) == target_class)).float().mean().item()
    return {"target": target_class,
            "mean_logit_gain": delta.mean().item(),
            "frac_flipped_to_target": moved_to,
            "eig_value": l[pos[0]].item()}
