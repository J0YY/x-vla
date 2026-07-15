"""Ordinary (non-tensor) baselines for the four-way ablation — spec §14 Stage 1.

These are used ONLY to compare against the tensor primitives during
Stage-1 language pretraining (spec §18 ablation table). They are never part of
the deployed χ-VLA forward graph.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from xvla.nn.attention import causal_mask


class SoftmaxAttention(nn.Module):
    """Ordinary causal multi-head softmax attention (baseline)."""

    def __init__(self, dim: int, n_heads: int, causal: bool = True):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.causal = causal
        self.wqkv = nn.Linear(dim, 3 * dim, bias=True)
        self.wo = nn.Linear(dim, dim, bias=True)
        nn.init.normal_(self.wo.weight, std=(dim ** -0.5) * 0.5)
        nn.init.zeros_(self.wo.bias)

    def forward(self, x, mask=None, method: str = "explicit"):
        B, N, D = x.shape
        q, k, v = self.wqkv(x).split(D, dim=-1)
        def sh(t):
            return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal and mask is None)
        y = y.transpose(1, 2).reshape(B, N, D)
        return self.wo(y)


class SwiGLU(nn.Module):
    """Ordinary SwiGLU FFN (baseline): D(swish(Lx) ⊙ Rx)."""

    def __init__(self, dim: int, rank: int | None = None):
        super().__init__()
        rank = rank if rank is not None else 3 * dim
        self.left = nn.Linear(dim, rank, bias=True)
        self.right = nn.Linear(dim, rank, bias=True)
        self.down = nn.Linear(rank, dim, bias=True)

    def forward(self, x):
        return self.down(F.silu(self.left(x)) * self.right(x))
