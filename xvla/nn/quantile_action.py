"""Autoregressive monotone-quantile (AQ) action head — a strict superset of the linear head.

The linear+MSE head emits E[a|obs] and collapses on multimodal targets; the pooled
flow/product heads fixed multimodality but LOST per-step precision (DEVLOG cont.17:
0% vs linear 25% on LIBERO) for two reasons — they pooled the H action-query tokens to
one vector, and flow's decode was a stochastic sample. The AQ head removes both:

  * PER-STEP conditioning: each chunk step t reads its OWN token aq_out[:,t] (like the
    linear head), never a pooled vector → recovers per-step precision.
  * DETERMINISTIC decode: a monotone quantile map decoded at its densest point (MAP) →
    no sampling noise on (near-)unimodal conditionals.

Construction (per step t, per action dim d), the SOS-integral monotone polynomial
(same primitive as quantile_proto.py / knothe_transport_proto.py):

    s(τ)  = Σ_{j=0}^{p} c_j τ^j           c = affine/bilinear map of the per-step token h_t
    Q'(τ) = ε + s(τ)²  ≥ ε > 0            (the repo's degree-2 ⊙ squaring primitive)
    Q(τ)  = b + ε·τ + Σ_{m=0}^{2p} w_m τ^{m+1},   w_m = q_m/(m+1) = cᵀ A_m c

Q is monotone-increasing in τ for ANY weights (no clamp/relu/softmax in the graph), so
monotonicity survives folding; every coefficient w_m = cᵀA_m c is a fixed symmetric
bilinear form in c, i.e. the exact Q_c object odt_interp diagonalizes. The coefficient
map c = C·h_t is a BilinearFFN, so h_t → Q folds via dense_core exactly like the linear
head.

MULTIMODALITY + temporal coherence: draw ONE scalar τ per chunk (out-of-graph) and
evaluate every (t,d) at the SAME τ. Since each Q is increasing in τ, low-τ selects every
step's low branch and high-τ its high branch → one scalar commits the WHOLE chunk to a
coherent mode (reach-L vs reach-R), decoded in parallel. Per-step h_t gives precision;
the shared τ is the low-dim mode latent giving coherence (independent per-step τ would
mode-flip mid-chunk).

Why it can't be worse than linear: as s→0, ε→0, Q(τ)→b(h_t) = a linear map of the token,
independent of τ — i.e. the current linear head. Under the strictly-proper pinball loss
the population optimum on a unimodal conditional is Q(τ)=μ(h)+σ(h)Φ⁻¹(τ), whose
deterministic median/MAP decode is μ(h) = the conditional mean = linear's target, with no
sampling noise. On a multimodal conditional linear's mean is an invalid between-modes
action while AQ-MAP lands on a real mode.

Decode (out-of-graph controller op, spec §11): AQ-MAP picks τ* = argmin_τ ‖dT/dτ‖ (the
densest point of the chunk curve) — a cheap 1-D grid scan of a low-degree polynomial, no
ODE loop, no root-find, no Z. AQ-sample draws a shared τ~U for coverage/exploration.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xvla.nn.bilinear import BilinearFFN


def sos_integral_matrices(p: int) -> torch.Tensor:
    """Constant matrices A_m (m=0..2p) s.t. w_m = cᵀ A_m c gives the integrated coeffs.

    s(τ)²=Σ_{j,l} c_j c_l τ^{j+l}; ∫₀^τ s² = Σ_m (Σ_{j+l=m} c_j c_l)/(m+1) · τ^{m+1}.
    So w_m = cᵀ A_m c with A_m[j,l] = 1/(m+1) if j+l==m else 0. Returns (2p+1, p+1, p+1).
    """
    A = torch.zeros(2 * p + 1, p + 1, p + 1)
    for m in range(2 * p + 1):
        for j in range(p + 1):
            l = m - j
            if 0 <= l <= p:
                A[m, j, l] = 1.0 / (m + 1)
    return A


class AutoregQuantileHead(nn.Module):
    """Per-step monotone-quantile chunk head with a shared mode latent τ.

    Args:
        dim: per-step token width (= VLAConfig.dim; reads aq_out[:,t], NOT pooled).
        action_dim: per-step action width d_a.
        horizon: chunk length H.
        degree: polynomial degree p of s(τ) (#modes ≤ p; higher p sharpens modes).
        rank: CP rank of the coefficient BilinearFFN.
        eps: strict-monotonicity floor (Q' ≥ eps).
        grid: τ-grid resolution for the out-of-graph MAP decode.
    """

    def __init__(self, dim: int, action_dim: int, horizon: int, degree: int = 4,
                 rank: int | None = None, eps: float = 1e-3, grid: int = 128):
        super().__init__()
        self.dim = dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.p = degree
        self.eps = eps
        self.grid = grid
        # per-step, per-dim: c ∈ R^{p+1} (the s-poly coeffs) and b ∈ R (the offset).
        self.c_head = BilinearFFN(dim, rank=rank, out_dim=action_dim * (degree + 1), down_bias=False)
        self.b_head = BilinearFFN(dim, rank=rank, out_dim=action_dim, down_bias=False)
        self.register_buffer("A", sos_integral_matrices(degree))   # (2p+1, p+1, p+1)

    def _coeffs(self, aq_out: torch.Tensor):
        """aq_out: (B, H, dim) → c: (B,H,d_a,p+1), b: (B,H,d_a), w: (B,H,d_a,2p+1)."""
        B, H, _ = aq_out.shape
        c = self.c_head(aq_out).reshape(B, H, self.action_dim, self.p + 1)
        b = self.b_head(aq_out)                                     # (B,H,d_a)
        w = torch.einsum("mjl,bhdj,bhdl->bhdm", self.A.to(c.dtype), c, c)  # (B,H,d_a,2p+1)
        return c, b, w

    def _Q(self, b: torch.Tensor, w: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Evaluate Q(τ). b:(B,H,d_a), w:(B,H,d_a,2p+1), tau: broadcastable to (B,H,d_a).
        Returns Q with the shape of the broadcast."""
        # τ^{m+1} for m=0..2p
        powers = torch.arange(1, 2 * self.p + 2, device=w.device, dtype=w.dtype)
        tau_pow = tau.unsqueeze(-1) ** powers                      # (...,2p+1)
        poly = (w * tau_pow).sum(-1)                               # (...)
        return b + self.eps * tau + poly

    def forward(self, aq_out: torch.Tensor):
        return self._coeffs(aq_out)

    def loss(self, aq_out: torch.Tensor, target_chunk: torch.Tensor) -> torch.Tensor:
        """Strictly-proper pinball (quantile-regression) loss with a SHARED per-sample τ
        (enforces comonotone whole-chunk coupling). aq_out:(B,H,dim); target:(B,H,d_a).
        No Z, no negatives, no coercivity — same training-cleanliness class as flow."""
        B, H, _ = aq_out.shape
        _, b, w = self._coeffs(aq_out)
        tau = torch.rand(B, 1, 1, device=aq_out.device, dtype=aq_out.dtype)   # shared per sample
        q = self._Q(b, w, tau.expand(B, H, self.action_dim))
        u = target_chunk - q
        # pinball ρ_τ(u) = u·(τ - 1[u<0])
        rho = u * (tau - (u < 0).to(u.dtype))
        return rho.mean()

    @torch.no_grad()
    def decode(self, aq_out: torch.Tensor) -> torch.Tensor:
        """OUT-OF-GRAPH AQ-MAP: τ* = argmin_τ ‖dT/dτ‖ (densest point of the chunk curve),
        shared across all (t,d) → coherent whole-chunk commitment. (B,H,dim)→(B,H,d_a)."""
        B, H, _ = aq_out.shape
        _, b, w = self._coeffs(aq_out)
        taus = torch.linspace(0.0, 1.0, self.grid, device=aq_out.device, dtype=aq_out.dtype)
        # Q over the grid: (B,H,d_a,grid)
        powers = torch.arange(1, 2 * self.p + 2, device=w.device, dtype=w.dtype)
        tau_pow = taus[:, None] ** powers                          # (grid, 2p+1)
        poly = torch.einsum("bhdm,gm->bhdg", w, tau_pow)           # (B,H,d_a,grid)
        Q = b[..., None] + self.eps * taus + poly                 # (B,H,d_a,grid)
        # chunk-curve speed ‖dT/dτ‖ over (H·d_a), per grid step
        dQ = (Q[..., 1:] - Q[..., :-1]).reshape(B, H * self.action_dim, self.grid - 1)
        speed = dQ.norm(dim=1)                                     # (B, grid-1)
        star = speed.argmin(dim=1)                                 # (B,) densest τ index
        idx = star[:, None, None, None].expand(B, H, self.action_dim, 1)
        return Q.gather(-1, idx).squeeze(-1)                       # (B,H,d_a)

    @torch.no_grad()
    def sample(self, aq_out: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Shared-τ stochastic decode (coverage). (B,H,dim)→(B,n_samples,H,d_a)."""
        B, H, _ = aq_out.shape
        _, b, w = self._coeffs(aq_out)
        outs = []
        for _ in range(n_samples):
            tau = torch.rand(B, 1, 1, device=aq_out.device, dtype=aq_out.dtype)
            outs.append(self._Q(b, w, tau.expand(B, H, self.action_dim)))
        return torch.stack(outs, dim=1)
