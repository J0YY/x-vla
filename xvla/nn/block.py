"""χ-transformer block and stack — spec §4.4 (vision) and §10 (joint).

Each block is::

    u = RBN_attn(x)
    x = x + α · BiAttn(u)
    v = RBN_ffn(x)
    x = x + β · BFFN(v)

with foldable RmsBatchNorm before each multiplicative branch, softmax-free
bilinear attention, CP-factorized bilinear FFN, and residual gains
α = β = 1/√(2L) by default (spec §10). No LayerNorm/softmax/GELU remains after
export; the RBN scalars fold into the following projections.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xvla.nn.attention import BilinearAttention
from xvla.nn.bilinear import BilinearFFN
from xvla.nn.normalization import make_norm


class ChiTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_rank: int | None = None,
        n_layers: int = 1,
        causal: bool = False,
        norm: str = "per_token",
        qk_norm: str = "per_token",
        rbn_momentum: float = 0.99,
        learned_gain: bool = False,
    ):
        super().__init__()
        self.rbn_attn = make_norm(norm, momentum=rbn_momentum)
        self.attn = BilinearAttention(dim, n_heads, causal=causal,
                                      qk_norm=qk_norm, rbn_momentum=rbn_momentum)
        self.rbn_ffn = make_norm(norm, momentum=rbn_momentum)
        self.ffn = BilinearFFN(dim, rank=ffn_rank)

        gain = (2.0 * n_layers) ** -0.5
        if learned_gain:
            # Alternative: learned scalar init 0.01, no nonlinear bounding (spec §10).
            self.attn_gain = nn.Parameter(torch.tensor(0.01))
            self.ffn_gain = nn.Parameter(torch.tensor(0.01))
        else:
            self.register_buffer("attn_gain", torch.tensor(gain))
            self.register_buffer("ffn_gain", torch.tensor(gain))

    def forward(self, x, mask=None, method: str = "explicit"):
        u = self.rbn_attn(x)
        x = x + self.attn_gain * self.attn(u, mask=mask, method=method)
        v = self.rbn_ffn(x)
        x = x + self.ffn_gain * self.ffn(v)
        return x


class ChiTransformer(nn.Module):
    """Stack of χ-transformer blocks (spec §4 / §10)."""

    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
        ffn_rank: int | None = None,
        causal: bool = False,
        norm: str = "per_token",
        qk_norm: str = "per_token",
        rbn_momentum: float = 0.99,
        learned_gain: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.causal = causal
        self.blocks = nn.ModuleList([
            ChiTransformerBlock(
                dim, n_heads, ffn_rank=ffn_rank, n_layers=n_layers,
                causal=causal, norm=norm, qk_norm=qk_norm,
                rbn_momentum=rbn_momentum, learned_gain=learned_gain,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x, mask=None, method: str = "explicit"):
        for block in self.blocks:
            x = block(x, mask=mask, method=method)
        return x
