"""χ-ViT — a fully tensor-decomposable vision transformer (spec §4, Milestone 2).

Pixels → fixed patch extraction → linear patch embedding (+ learned 2D position
embeddings, and an optional camera embedding) → bidirectional χ-transformer blocks
(softmax-free bilinear attention + bilinear FFN + foldable norm) → pooled features
→ linear head. Everything before the (training-only) softmax classification loss is
a pure tensor network: the patch embedding is affine (fixed RGB standardization can
be folded into it), position/camera embeddings are constants, and the blocks are the
same primitives validated in Milestone 1.

Two branches (spatial + semantic, spec §4.1/4.2) use the same architecture with
different weights/objectives; that is expressed by instantiating two χ-ViTs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from xvla.nn.block import ChiTransformer
from xvla.nn.normalization import make_norm


@dataclass
class ViTConfig:
    image_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ffn_rank: int | None = None       # default 3*dim
    num_classes: int = 10
    norm: str = "per_token"           # "per_token" | "scalar_rbn" | "homotopy"
    qk_norm: str = "per_token"
    attn: str = "bilinear"            # "bilinear" (χ) | "softmax" (matched baseline)
    pool: str = "mean"                # "mean" | "cls"
    rbn_momentum: float = 0.99

    @property
    def num_patches(self) -> int:
        g = self.image_size // self.patch_size
        return g * g


class ChiViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        # Fixed patch extraction + affine embedding (a strided conv is exactly a
        # per-patch linear map on the fixed reshape — tensor-compatible).
        self.patch = nn.Conv2d(cfg.in_chans, cfg.dim, kernel_size=cfg.patch_size,
                               stride=cfg.patch_size)
        n = cfg.num_patches
        self.pos_emb = nn.Parameter(torch.zeros(1, n, cfg.dim))
        nn.init.normal_(self.pos_emb, std=cfg.dim ** -0.5)
        self.use_cls = cfg.pool == "cls"
        if self.use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, cfg.dim))
            nn.init.normal_(self.cls, std=cfg.dim ** -0.5)

        self.blocks = ChiTransformer(
            cfg.dim, cfg.n_layers, cfg.n_heads, ffn_rank=cfg.ffn_rank,
            causal=False, norm=cfg.norm, qk_norm=cfg.qk_norm, attn=cfg.attn,
            rbn_momentum=cfg.rbn_momentum)
        self.norm_out = make_norm(cfg.norm, momentum=cfg.rbn_momentum)
        self.head = nn.Linear(cfg.dim, cfg.num_classes)

    def features(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return per-token features (B, N[, +1 cls], dim) — the pure tensor part."""
        x = self.patch(imgs)                       # (B, dim, g, g)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)           # (B, N, dim)
        x = x + self.pos_emb
        if self.use_cls:
            x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        return self.blocks(x)

    def forward(self, imgs: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.features(imgs)
        x = self.norm_out(x)
        pooled = x[:, 0] if self.use_cls else x.mean(dim=1)
        logits = self.head(pooled)
        loss = None if targets is None else F.cross_entropy(logits, targets)
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ChiConvEncoder(nn.Module):
    """Fully tensor-decomposable CONVOLUTIONAL vision encoder (drop-in for ChiViT).

    The from-scratch ChiViT (a single strided-conv patch embed + bilinear transformer) is the
    weakest component on a pixel task: ViTs need ~1M+ images or distillation to match a CNN in
    the small-data regime, whereas convolution gives the right locality/weight-sharing inductive
    bias for free (and the repo's own topology study found conv >> dense for coherent features).
    This is a ResNet-style FOLDABLE stem: a stack of strided BILINEAR conv layers
    ``h = convL(n(h)) * convR(n(h))`` (each degree-2, tensor-convertible) with a foldable norm
    between them to tame the degree-2^depth magnitude growth, downsampling to a ``grid x grid``
    token map fed to the joint transformer. Same ``.features(img) -> (B, N, dim)`` contract as
    ChiViT, so it swaps in via ``VLAConfig.vision_encoder='conv'`` with zero downstream changes.
    With ``norm='rational'`` the whole encoder folds (bilinear conv = CP core, rational norm =
    P(x)/Q(x)); with ``norm='per_token'`` it is the non-strict capability-first variant.
    """

    def __init__(self, cfg: ViTConfig, kernel: int = 3, grid: int = 8, norm: str | None = None):
        super().__init__()
        self.cfg = cfg
        w = cfg.dim
        self.grid = grid
        norm = norm or cfg.norm
        n_down = max(1, int(round(math.log2(max(cfg.image_size, grid) / grid))))  # stride-2 stages
        p = kernel // 2
        self.convL = nn.ModuleList()
        self.convR = nn.ModuleList()
        self.norms = nn.ModuleList()
        c_in = cfg.in_chans
        for _ in range(n_down):
            self.convL.append(nn.Conv2d(c_in, w, kernel, stride=2, padding=p))
            self.convR.append(nn.Conv2d(c_in, w, kernel, stride=2, padding=p))
            self.norms.append(make_norm(norm, momentum=cfg.rbn_momentum))
            c_in = w
        self.norm_out = make_norm(norm, momentum=cfg.rbn_momentum)
        self.pos_emb = nn.Parameter(torch.zeros(1, grid * grid, w))
        nn.init.normal_(self.pos_emb, std=w ** -0.5)

    @staticmethod
    def _norm_c(n: nn.Module, h: torch.Tensor) -> torch.Tensor:
        # apply a (last-dim) norm over the CHANNEL axis of a (B,C,H,W) feature map
        return n(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

    def features(self, imgs: torch.Tensor) -> torch.Tensor:
        """Pixels (B,3,H,W) -> spatial token features (B, grid*grid, dim)."""
        h = imgs
        for lL, lR, n in zip(self.convL, self.convR, self.norms):
            h = lL(h) * lR(h)                       # bilinear conv (downsamples by 2)
            h = self._norm_c(n, h)                  # foldable norm over channels
        if h.shape[-1] != self.grid:
            h = F.adaptive_avg_pool2d(h, self.grid)
        h = self._norm_c(self.norm_out, h)
        tok = h.flatten(2).transpose(1, 2)          # (B, grid*grid, dim)
        return tok + self.pos_emb

    def forward(self, imgs: torch.Tensor, targets: torch.Tensor | None = None):
        # classifier path (for standalone SVHN/CIFAR sanity), mean-pool + head-less caller uses features()
        return self.features(imgs)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
