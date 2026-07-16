"""χ-MLP — a feedforward bilinear (CP) network for exact global ODT (Milestone 5).

This is deliberately NOT the transformer: a clean chain of bilinear CP layers with
no residual and no attention, so the whole map (after the linear input embed) is a
tree tensor network whose exact orthogonalize→diagonalize→truncate (ODT) analysis
is tractable — matching the setting Dooms et al. analyze. The transformer's exact
global ODT is Level-C / open (spec §16.1) and out of scope here.

    x → E (linear embed, kept out of the core) → [ scalarRBN → bilinear CP ]×L → linear head

Each bilinear layer is a CP factorization of a symmetric 3rd-order core
T ∈ R^{d×(d+1)×(d+1)} (via ``BilinearFFN.dense_core``). Foldable scalar norms fold
into the layer inputs, leaving a pure chain of cores + linear embed + linear head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from xvla.nn.bilinear import BilinearFFN
from xvla.nn.normalization import make_norm


@dataclass
class ChiMLPConfig:
    in_dim: int = 3072            # 32*32*3 flattened SVHN/CIFAR
    dim: int = 32                 # bond width d (kept small so exact ODT is cheap)
    n_layers: int = 3
    num_classes: int = 10
    norm: str = "scalar_rbn"      # foldable; folded into layer inputs before ODT


class ChiMLP(nn.Module):
    def __init__(self, cfg: ChiMLPConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(cfg.in_dim, cfg.dim)     # linear front-end (foldable)
        self.norms = nn.ModuleList([make_norm(cfg.norm) for _ in range(cfg.n_layers)])
        self.layers = nn.ModuleList([
            BilinearFFN(cfg.dim, rank=3 * cfg.dim, out_dim=cfg.dim, down_bias=False)
            for _ in range(cfg.n_layers)
        ])
        self.head = nn.Linear(cfg.dim, cfg.num_classes)

    def forward(self, x, targets=None):
        h = self.embed(x.flatten(1))
        for norm, layer in zip(self.norms, self.layers):
            h = layer(norm(h))                          # NO residual: clean chain
        logits = self.head(h)
        loss = None if targets is None else F.cross_entropy(logits, targets)
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
