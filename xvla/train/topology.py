"""Topology sweep analysis (Experiment A, Q1/Q4).

For a :class:`ShallowBilinear` model the class logit is exactly quadratic in the
input pixels. We build the per-class quadratic ``Q_c`` (D×D, D = in_ch·hw·hw)
*exactly and cheaply* from the model's linear operators — no D×D Hessian (which
would OOM when vmapped through a conv) — then:

* verify the exact quadratic reconstruction of the module,
* extract eigen-atoms of ``Q_c`` and measure their **spatial locality** vs random
  directions (coherence),
* run the **faithfulness** battery (drop top-r vs random-r eigenvectors) in pixel
  space (causality).

Q_c construction. With ``u_{o,p} = linL functional`` and ``w_{o,p} = linR
functional`` (bias removed), feature_o = mean_p (u·x+bL)(w·x+bR), so the quadratic
part of logit_c is Σ_o (W_head[c,o]/P) Σ_p (u_{o,p}·x)(w_{o,p}·x). Stacking the
functionals into U,V ∈ R^{D×T} (T = width·P) with per-term coefficient
``c_t = W_head[c,o_t]/P``:

    Q_c = ½ ( U diag(c) Vᵀ + V diag(c) Uᵀ ).

U,V are obtained by applying linL/linR to the identity basis (chunked).
"""

from __future__ import annotations

import torch

from xvla.train.odt import random_projector


@torch.no_grad()
def linear_operators(model, device, chunk: int = 256):
    """Return U,V ∈ (D, width, P): the linL/linR functionals (bias removed).

    ``U[d,o,p] = ∂ linL[o,p] / ∂ x_d``. Built by pushing the identity basis
    through linL/linR in chunks (architecture-agnostic).
    """
    model = model.to(device).double().eval()
    D = model.in_ch * model.hw * model.hw
    zero = torch.zeros(1, model.in_ch, model.hw, model.hw, device=device, dtype=torch.float64)
    bL = model.linL(zero)[0]                      # (width, P)
    bR = model.linR(zero)[0]
    U = torch.empty(D, model.width, model.P, device=device, dtype=torch.float64)
    V = torch.empty(D, model.width, model.P, device=device, dtype=torch.float64)
    eye = torch.eye(D, device=device, dtype=torch.float64)
    for i in range(0, D, chunk):
        b = eye[i:i + chunk].reshape(-1, model.in_ch, model.hw, model.hw)
        U[i:i + chunk] = model.linL(b) - bL
        V[i:i + chunk] = model.linR(b) - bR
    return U, V, bL, bR


@torch.no_grad()
def class_quadratics(model, device):
    """Return Q (C,D,D), l (C,D), a (C,) with logit_c(x)=xᵀQ_c x + l_c·x + a_c.

    Uses the model's exact per-term coefficient map (``term_coeff``), so it is
    correct for any readout (global mean or spatial-grid). Each bilinear term
    (o,pos) contributes ``coeff[c,o,pos]·(u·x)(w·x)`` to logit_c.
    """
    model = model.to(device).double().eval()
    U, V, bL, bR = linear_operators(model, device)     # (D,width,P) linear parts + biases
    D = U.shape[0]
    Uf = U.reshape(D, -1)                              # (D, T=width*P)
    Vf = V.reshape(D, -1)
    coeff = model.term_coeff().reshape(model.head.weight.shape[0], -1).to(device)  # (C,T)
    bLf = bL.reshape(-1)                               # (T,)
    bRf = bR.reshape(-1)
    C = coeff.shape[0]
    Q = torch.empty(C, D, D, device=device, dtype=torch.float64)
    for c in range(C):
        M = (Uf * coeff[c]) @ Vf.T                     # (D,D)
        Q[c] = 0.5 * (M + M.T)
    # linear part: Σ_t coeff[c,t] (bR_t u_t + bL_t w_t)·x  → (C,D)
    l = (coeff * bRf) @ Uf.T + (coeff * bLf) @ Vf.T
    # constant part: Σ_t coeff[c,t] bL_t bR_t + b_head
    a = coeff @ (bLf * bRf) + model.head.bias.double().to(device)
    return Q, l, a


@torch.no_grad()
def quad_logits(Q, l, a, x):
    """logit_c = xᵀ Q_c x + l_c·x + a_c for a batch x (B, D)."""
    xq = torch.einsum("bi,cij,bj->bc", x, Q, x)
    return xq + x @ l.T + a


@torch.no_grad()
def locality(vec, in_ch, hw, frac=0.1):
    """Fraction of L2 energy in the top-``frac`` pixels (channels summed)."""
    e = (vec.reshape(in_ch, hw * hw) ** 2).sum(0)
    k = max(1, int(frac * e.numel()))
    return (torch.topk(e, k).values.sum() / e.sum().clamp_min(1e-30)).item()


@torch.no_grad()
def atom_report(Q, in_ch, hw, device, n_top=4, n_rand=300):
    """Mean locality of the top-|λ| eigen-atoms per class vs random-direction control."""
    D = Q.shape[1]
    rand = []
    g = torch.Generator(device=device).manual_seed(0)
    for _ in range(n_rand):
        v = torch.randn(D, generator=g, device=device, dtype=torch.float64)
        rand.append(locality(v, in_ch, hw))
    rand_loc = sum(rand) / len(rand)
    locs = []
    for c in range(Q.shape[0]):
        evals, evecs = torch.linalg.eigh(Q[c])
        order = evals.abs().argsort(descending=True)[:n_top]
        for i in order:
            locs.append(locality(evecs[:, i], in_ch, hw))
    atom_loc = sum(locs) / len(locs)
    return atom_loc, rand_loc


@torch.no_grad()
def faithfulness(Q, l, a, x, y, ranks, device):
    """Accuracy under keep-top-r / drop-top-r / drop-random-r eigen-atoms of each Q_c.

    Quadratic part is truncated per class; the linear+constant parts are kept, so
    the reported accuracy is real (isolates the quadratic mechanisms' contribution).
    """
    C, D, _ = Q.shape
    lam, vec = [], []
    for c in range(C):
        ev, V = torch.linalg.eigh(Q[c])
        o = ev.abs().argsort(descending=True)
        lam.append(ev[o]); vec.append(V[:, o])
    lin = x @ l.T + a                              # (B, C) linear+const, unchanged

    def acc(masks):
        out = lin.clone()
        for c in range(C):
            proj = (x @ vec[c]) ** 2               # (B, D)
            out[:, c] = out[:, c] + (proj * (lam[c] * masks[c])).sum(-1)
        return (out.argmax(-1) == y).float().mean().item()

    g = torch.Generator(device=device).manual_seed(0)
    res = {"keep": {}, "drop_top": {}, "drop_random": {}}
    for r in ranks:
        keep = [torch.zeros(D, device=device, dtype=torch.float64) for _ in range(C)]
        for c in range(C):
            keep[c][:r] = 1.0
        res["keep"][r] = acc(keep)
        drop = [torch.ones(D, device=device, dtype=torch.float64) for _ in range(C)]
        for c in range(C):
            drop[c][:r] = 0.0
        res["drop_top"][r] = acc(drop)
        rnd = [torch.ones(D, device=device, dtype=torch.float64) for _ in range(C)]
        for c in range(C):
            idx = torch.randperm(D, generator=g, device=device)[:r]
            rnd[c][idx] = 0.0
        res["drop_random"][r] = acc(rnd)
    return res


# --------------------------------------------------------------------------- #
# Data-driven input-space analysis (Experiment A-depth — works at any depth)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _acc_from(f, X, Y, P, shape, D):
    cor = tot = 0
    for i in range(0, X.shape[0], 500):
        xf = X[i:i + 500].reshape(-1, D).float()      # P is float; model wants float
        xp = (xf @ P).reshape(-1, *shape)
        lg = f(xp)
        cor += (lg.argmax(-1) == Y[i:i + 500]).sum().item(); tot += lg.shape[0]
    return cor / tot


def input_gram(f, X, Y, device, n_max=2000, chunk=200):
    """Data-driven input-space output-sensitivity Gram G = E[g gᵀ],
    g = ∂(true-class logit)/∂input. f(x)->logits; X (N,C,H,W). Returns (D,D)."""
    D = X[0].numel()
    G = torch.zeros(D, D, device=device, dtype=torch.float64)
    n = 0
    with torch.enable_grad():                          # may be called under no_grad
        for i in range(0, min(n_max, X.shape[0]), chunk):
            xb = X[i:i + chunk].clone().requires_grad_(True)
            sel = f(xb).gather(1, Y[i:i + chunk, None]).sum()
            g, = torch.autograd.grad(sel, xb)
            g = g.reshape(g.shape[0], -1).double()
            G += g.T @ g; n += g.shape[0]
    return G / n


@torch.no_grad()
def input_analysis(f, X, Y, device, ks=(1, 2, 4, 8, 16, 32, 64, 128, 256), n_top=6):
    """Locality of the top input-Gram eigenvectors (vs random) + faithfulness
    (truncate input onto top-k GLOBAL vs RANDOM subspace → accuracy)."""
    C, H, W = X.shape[1:]
    D = C * H * W
    G = input_gram(f, X, Y, device)
    evals, evecs = torch.linalg.eigh(G)
    evecs = evecs.flip(1)
    # coherence
    rl = []
    g = torch.Generator(device=device).manual_seed(0)
    for _ in range(200):
        rl.append(locality(torch.randn(D, generator=g, device=device, dtype=torch.float64), C, H))
    rand_loc = sum(rl) / len(rl)
    atom_loc = sum(locality(evecs[:, j], C, H) for j in range(n_top)) / n_top
    # faithfulness
    rng = torch.Generator(device=device).manual_seed(0)
    glob, rnd = {}, {}
    for k in ks:
        Vk = evecs[:, :k]
        glob[k] = _acc_from(f, X, Y, (Vk @ Vk.T).float(), (C, H, W), D)
        rnd[k] = _acc_from(f, X, Y, random_projector(D, k, device, rng).float(), (C, H, W), D)
    return {"atom_locality": round(atom_loc, 4), "random_locality": round(rand_loc, 4),
            "locality_ratio": round(atom_loc / rand_loc, 3),
            "global_curve": {int(k): round(v, 4) for k, v in glob.items()},
            "random_curve": {int(k): round(v, 4) for k, v in rnd.items()}}
