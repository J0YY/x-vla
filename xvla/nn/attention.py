"""Softmax-free bilinear attention — spec §6.

Each head learns *two* query–key systems and one value system::

    Q1,K1,Q2,K2,V = X W_{Q1,K1,Q2,K2,V}
    A1 = Q1 K1ᵀ ,  A2 = Q2 K2ᵀ
    Y  = C [ M ⊙ (A1 ⊙ A2)/d_h ] V ,   out = Y W_O

with fixed mask ``M`` and fixed row-scaling ``C_ii = 1/√n_i`` where ``n_i`` is
the number of keys visible to query ``i``. There is **no softmax**.

Three mathematically-identical execution paths are provided (spec §6.5):

* ``forward`` / ``_explicit`` — materialize the N×N pattern (best for short N).
* ``_khatri_rao`` — row-wise Kronecker features q̃=q1⊗q2, so A = Q̃ K̃ᵀ, letting
  ``Y = Q̃ (K̃ᵀ V)`` avoid the N×N matrix (best when N > d_h²).
* ``_causal_scan`` — prefix-scan accumulator ``S_i = Σ_{j≤i} k̃_j v_jᵀ`` giving
  ``y_i = q̃_iᵀ S_i / (d_h √i)`` for the causal decoder.

All three agree to FP32 ~1e-5 (validated in ``tests/test_operators.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xvla.nn.normalization import RmsBatchNorm, HomotopyNorm


def causal_mask(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Lower-triangular {0,1} mask of shape (n, n): query i sees keys j ≤ i."""
    return torch.tril(torch.ones(n, n, device=device, dtype=dtype))


def _row_kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise Kronecker product along the last dim.

    For ``a,b`` of shape ``(..., d)`` returns ``(..., d*d)`` with element
    ``[..., i*d + j] = a[..., i] * b[..., j]`` — i.e. ``(a ⊗ b)`` per row.
    """
    *lead, d = a.shape
    return (a.unsqueeze(-1) * b.unsqueeze(-2)).reshape(*lead, d * d)


class BilinearAttention(nn.Module):
    """Multi-head bilinear (softmax-free) attention (spec §6).

    Args:
        dim: model width d.
        n_heads: number of heads H (head dim d_h = d / H).
        causal: if True the default mask is lower-triangular.
        row_scale: ``"invsqrt"`` for C_ii = 1/√n_i (default) or ``"inv"`` for 1/n_i.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        causal: bool = False,
        row_scale: str = "invsqrt",
        score_scale: str = "d_h2",
        qk_norm: str = "per_token",
        qk_rbn: bool | None = None,
        rbn_momentum: float = 0.99,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.causal = causal
        if row_scale not in ("invsqrt", "inv"):
            raise ValueError("row_scale must be 'invsqrt' or 'inv'")
        self.row_scale = row_scale
        self.norm_eps = norm_eps
        # Score denominator applied to the combined pattern A1⊙A2. "d_h" matches
        # spec §6.1 literally; "d_h2" divides each score by d_h (Riggs' working
        # reference), shrinking the heavy-tailed degree-4 pattern by d_h× and
        # keeping the residual stream stable through the softmax-free stack.
        if score_scale not in ("d_h", "d_h2"):
            raise ValueError("score_scale must be 'd_h' or 'd_h2'")
        self.score_scale = score_scale
        self._score_denom = self.head_dim ** (2 if score_scale == "d_h2" else 1)

        # Six dense maps vs four in ordinary attention (spec §6.6): +50% params.
        self.wq1 = nn.Linear(dim, dim, bias=True)
        self.wk1 = nn.Linear(dim, dim, bias=True)
        self.wq2 = nn.Linear(dim, dim, bias=True)
        self.wk2 = nn.Linear(dim, dim, bias=True)
        self.wv = nn.Linear(dim, dim, bias=True)
        self.wo = nn.Linear(dim, dim, bias=True)

        # Q/K/V branch normalization. Two modes:
        #   "per_token"  – RMS-norm each q_i,k_i over head_dim (Riggs' recipe).
        #                  Makes ‖q_i‖²=d_h so |q_i·k_j| ≤ d_h and, with
        #                  score_scale="d_h2", the pattern is bounded to ~[-1,1]:
        #                  both STABLE and expressive. Input-dependent → NOT
        #                  foldable (spec §18 "per-instance L2 normalization",
        #                  the sanctioned non-strict stability control).
        #   "scalar_rbn" – scalar RmsBatchNorm per branch (spec §7.4): foldable
        #                  and tensor-pure, the strict export target. Weaker
        #                  (doesn't bound per-token scores); pair with a
        #                  per-token-trained teacher / Stage-6 calibration.
        #   "none"       – raw projections.
        if qk_rbn is not None:  # back-compat: qk_rbn True/False overrides qk_norm
            qk_norm = "scalar_rbn" if qk_rbn else "none"
        if qk_norm not in ("per_token", "scalar_rbn", "homotopy", "none"):
            raise ValueError("qk_norm must be 'per_token', 'scalar_rbn', 'homotopy', or 'none'")
        self.qk_norm = qk_norm
        if qk_norm == "scalar_rbn":
            self.rbn_q1 = RmsBatchNorm(momentum=rbn_momentum)
            self.rbn_k1 = RmsBatchNorm(momentum=rbn_momentum)
            self.rbn_q2 = RmsBatchNorm(momentum=rbn_momentum)
            self.rbn_k2 = RmsBatchNorm(momentum=rbn_momentum)
            self.rbn_v = RmsBatchNorm(momentum=rbn_momentum)
        elif qk_norm == "homotopy":
            # Per-branch per-token↔scalar knob on Q/K (matches per_token at κ=0,
            # scalar-foldable at κ=1). V is left raw, as in per_token mode.
            self.rbn_q1 = HomotopyNorm(momentum=rbn_momentum)
            self.rbn_k1 = HomotopyNorm(momentum=rbn_momentum)
            self.rbn_q2 = HomotopyNorm(momentum=rbn_momentum)
            self.rbn_k2 = HomotopyNorm(momentum=rbn_momentum)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Q/K scaled so each score factor is O(1) after the 1/d_h division; the
        # two score systems together carry the 1/d_h, so scale each by d_h^-1/4.
        qk_std = self.head_dim ** -0.25
        for lin in (self.wq1, self.wk1, self.wq2, self.wk2):
            nn.init.normal_(lin.weight, std=qk_std)
            nn.init.zeros_(lin.bias)
        nn.init.normal_(self.wv.weight, std=self.dim ** -0.5)
        nn.init.zeros_(self.wv.bias)
        # Output projection at reduced residual-branch scale (spec §13.1).
        nn.init.normal_(self.wo.weight, std=(self.dim ** -0.5) * 0.5)
        nn.init.zeros_(self.wo.bias)

    def _project(self, x: torch.Tensor):
        """Return q1,k1,q2,k2,v each shaped (B, H, N, d_h)."""
        B, N, _ = x.shape
        def split(lin):
            return lin(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        q1, k1 = split(self.wq1), split(self.wk1)
        q2, k2 = split(self.wq2), split(self.wk2)
        v = split(self.wv)
        if self.qk_norm == "per_token":
            def rms(t):  # RMS-norm over head_dim: ‖t_i‖² = d_h
                return t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
            q1, k1, q2, k2 = rms(q1), rms(k1), rms(q2), rms(k2)
        elif self.qk_norm == "scalar_rbn":
            q1, k1 = self.rbn_q1(q1), self.rbn_k1(k1)
            q2, k2 = self.rbn_q2(q2), self.rbn_k2(k2)
            v = self.rbn_v(v)
        elif self.qk_norm == "homotopy":
            q1, k1 = self.rbn_q1(q1), self.rbn_k1(k1)
            q2, k2 = self.rbn_q2(q2), self.rbn_k2(k2)
        return q1, k1, q2, k2, v

    def _resolve_mask(self, n: int, x: torch.Tensor, mask):
        if mask is not None:
            return mask.to(device=x.device, dtype=x.dtype)
        if self.causal:
            return causal_mask(n, device=x.device, dtype=x.dtype)
        return x.new_ones(n, n)

    def _row_scale_vec(self, mask: torch.Tensor) -> torch.Tensor:
        """C diagonal from a {0,1} mask: 1/√n_i or 1/n_i (n_i = keys per query)."""
        n_i = mask.sum(dim=-1).clamp_min(1.0)  # (N,)
        if self.row_scale == "invsqrt":
            return n_i.rsqrt()
        return n_i.reciprocal()

    # -- execution paths ------------------------------------------------------
    def _explicit(self, x, mask=None):
        B, N, _ = x.shape
        q1, k1, q2, k2, v = self._project(x)
        m = self._resolve_mask(N, x, mask)                    # (N, N)
        a1 = q1 @ k1.transpose(-1, -2)                        # (B,H,N,N)
        a2 = q2 @ k2.transpose(-1, -2)
        a = (a1 * a2) / self._score_denom
        a = a * m                                             # fixed mask
        c = self._row_scale_vec(m).view(1, 1, N, 1)           # C row scaling
        y = c * (a @ v)                                       # (B,H,N,d_h)
        y = y.transpose(1, 2).reshape(B, N, self.dim)
        return self.wo(y)

    def _khatri_rao(self, x, mask=None):
        B, N, _ = x.shape
        q1, k1, q2, k2, v = self._project(x)
        m = self._resolve_mask(N, x, mask)
        qt = _row_kron(q1, q2)                                # (B,H,N,d_h²)
        kt = _row_kron(k1, k2)
        a = (qt @ kt.transpose(-1, -2)) / self._score_denom  # == (a1⊙a2) scaled
        a = a * m
        c = self._row_scale_vec(m).view(1, 1, N, 1)
        y = c * (a @ v)
        y = y.transpose(1, 2).reshape(B, N, self.dim)
        return self.wo(y)

    def _causal_scan(self, x):
        """Prefix-scan causal form (spec §6.4). Requires causal masking."""
        B, N, _ = x.shape
        q1, k1, q2, k2, v = self._project(x)
        qt = _row_kron(q1, q2)                                # (B,H,N,d_h²)
        kt = _row_kron(k1, k2)
        # S_i = Σ_{j≤i} k̃_j v_jᵀ  (B,H,N,d_h²,d_h)
        outer = kt.unsqueeze(-1) * v.unsqueeze(-2)
        S = torch.cumsum(outer, dim=2)
        # y_i = q̃_iᵀ S_i / (d_h √i)
        yi = torch.einsum("bhnf,bhnfd->bhnd", qt, S) / self._score_denom
        i = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
        if self.row_scale == "invsqrt":
            scale = i.rsqrt()
        else:
            scale = i.reciprocal()
        y = yi * scale.view(1, 1, N, 1)
        y = y.transpose(1, 2).reshape(B, N, self.dim)
        return self.wo(y)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                method: str = "explicit") -> torch.Tensor:
        """Compute bilinear attention.

        Args:
            x: (B, N, d) tokens.
            mask: optional fixed (N, N) {0,1} mask overriding the causal default.
            method: ``"explicit"`` | ``"khatri_rao"`` | ``"causal_scan"`` — all
                compute the same function (``causal_scan`` requires causal use).
        """
        if method == "explicit":
            return self._explicit(x, mask)
        if method == "khatri_rao":
            return self._khatri_rao(x, mask)
        if method == "causal_scan":
            if mask is not None:
                raise ValueError("causal_scan uses the implicit lower-triangular mask")
            return self._causal_scan(x)
        raise ValueError(f"unknown method {method!r}")
