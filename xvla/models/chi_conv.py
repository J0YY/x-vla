"""Shallow bilinear classifiers of varying *topology* (Experiment A, Q1/Q4).

One architecture-agnostic form so dense / convolutional / locally-connected models
can be compared on a single axis. Each is a *single* bilinear layer with NO
nonlinearity, so the class logit is exactly a quadratic form in the input pixels:

    feature = readout( linL(x) ⊙ linR(x) )        # readout pools the spatial product
    logit_c(x) = Σ_k W_head[c,k] feature_k + b_head[c] = xᵀ Q_c x + l_c·x + a_c

Two axes vary:

* ``mode`` — the *feature* topology of linL/linR:
    ``dense`` (full linear, no locality/sharing) · ``conv`` (local + weight-shared)
    · ``local`` (local but weights UNshared per position — isolates sharing).
* ``readout`` — how the (width, P) spatial product becomes the head's input:
    ``global`` (mean over all P positions → translation-invariant, width features)
    · ``spatial`` (average-pool to a G×G grid then flatten → keeps position,
    width·G² features). ``global`` conv is translation-invariant (and capacity-
    starved); ``spatial`` breaks that and restores capacity — the fair conv test.

``term_coeff()`` exposes, per class, the exact coefficient of each (o,pos) bilinear
term in ``logit_c`` — this is all the analysis needs to build ``Q_c`` exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from xvla.nn.normalization import make_norm


class ShallowBilinear(nn.Module):
    def __init__(self, mode: str = "dense", readout: str = "global", grid: int = 8,
                 width: int = 48, kernel: int = 5, in_ch: int = 3, hw: int = 32,
                 num_classes: int = 10):
        super().__init__()
        self.mode, self.readout, self.grid = mode, readout, grid
        self.width, self.kernel = width, kernel
        self.in_ch, self.hw = in_ch, hw
        self.P = 1 if mode == "dense" else hw * hw
        if mode == "dense":
            self.readout = "global"                      # no spatial axis to keep
        if mode == "dense":
            D = in_ch * hw * hw
            self.L = nn.Linear(D, width)
            self.R = nn.Linear(D, width)
        elif mode == "conv":
            p = kernel // 2
            self.L = nn.Conv2d(in_ch, width, kernel, padding=p)
            self.R = nn.Conv2d(in_ch, width, kernel, padding=p)
        elif mode == "local":
            self.pad = kernel // 2
            fan = in_ch * kernel * kernel
            self.WL = nn.Parameter(torch.randn(width, fan, hw * hw) * 0.02)
            self.WR = nn.Parameter(torch.randn(width, fan, hw * hw) * 0.02)
            self.bL = nn.Parameter(torch.zeros(width, hw * hw))
            self.bR = nn.Parameter(torch.zeros(width, hw * hw))
        else:
            raise ValueError(f"unknown mode {mode!r}")

        if self.readout == "spatial":
            assert hw % grid == 0, "grid must divide hw"
            out = width * grid * grid
            # pos -> grid-cell index (row-major), as a buffer for term_coeff
            step = hw // grid
            rows = torch.arange(hw).view(hw, 1).expand(hw, hw)
            cols = torch.arange(hw).view(1, hw).expand(hw, hw)
            cell = (rows // step) * grid + (cols // step)
            self.register_buffer("cell", cell.reshape(-1))     # (P,)
            self.cell_size = step * step
        else:
            out = width
        self.head = nn.Linear(out, num_classes)

    def _lin(self, x, which):
        if self.mode == "dense":
            m = (self.L if which == "L" else self.R)
            return m(x.flatten(1)).unsqueeze(-1)                     # (B, width, 1)
        if self.mode == "conv":
            m = (self.L if which == "L" else self.R)
            return m(x).flatten(2)                                   # (B, width, P)
        W = self.WL if which == "L" else self.WR
        b = self.bL if which == "L" else self.bR
        patches = F.unfold(x, self.kernel, padding=self.pad)         # (B, fan, P)
        return torch.einsum("bil,oil->bol", patches, W) + b.unsqueeze(0)

    def linL(self, x):
        return self._lin(x, "L")

    def linR(self, x):
        return self._lin(x, "R")

    def features(self, x):
        prod = self.linL(x) * self.linR(x)                          # (B, width, P)
        if self.readout == "global":
            return prod.mean(dim=-1)                                 # (B, width)
        pm = prod.reshape(prod.shape[0], self.width, self.hw, self.hw)
        return F.adaptive_avg_pool2d(pm, self.grid).flatten(1)       # (B, width*G²)

    def logits(self, x):
        return self.head(self.features(x))

    def forward(self, x, targets=None):
        lg = self.logits(x)
        loss = None if targets is None else F.cross_entropy(lg, targets)
        return lg, loss

    @torch.no_grad()
    def term_coeff(self):
        """Coefficient of each (o,pos) bilinear term in logit_c → (C, width, P)."""
        Wh = self.head.weight.double()                              # (C, out)
        C = Wh.shape[0]
        if self.readout == "global":
            return (Wh / self.P).unsqueeze(-1).expand(C, self.width, self.P).contiguous()
        Whr = Wh.reshape(C, self.width, self.grid * self.grid)      # (C, width, G²)
        return Whr[:, :, self.cell] / self.cell_size                # (C, width, P)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class ConvBilinearDeep(nn.Module):
    """Deep bilinear CONV classifier (Experiment A-depth). Stacks bilinear conv
    layers ``h ← convL(n(h)) ⊙ convR(n(h))`` with a foldable scalar norm between
    layers to tame the degree-2^depth magnitude blow-up, then spatial-pool + head.
    Pure-bilinear (tensor-convertible); analyzed via the data-driven input-space
    Gram (topology.py) since it is not a single quadratic at depth>1."""

    def __init__(self, depth: int = 3, width: int = 48, kernel: int = 5, grid: int = 8,
                 in_ch: int = 3, hw: int = 32, num_classes: int = 10, norm: str = "scalar_rbn"):
        super().__init__()
        p = kernel // 2
        self.grid, self.hw, self.in_ch, self.width = grid, hw, in_ch, width
        self.stemL = nn.Conv2d(in_ch, width, kernel, padding=p)
        self.stemR = nn.Conv2d(in_ch, width, kernel, padding=p)
        self.norms = nn.ModuleList([make_norm(norm) for _ in range(depth - 1)])
        self.midL = nn.ModuleList([nn.Conv2d(width, width, kernel, padding=p) for _ in range(depth - 1)])
        self.midR = nn.ModuleList([nn.Conv2d(width, width, kernel, padding=p) for _ in range(depth - 1)])
        self.norm_out = make_norm(norm)
        self.head = nn.Linear(width * grid * grid, num_classes)

    def logits(self, x):
        h = self.stemL(x) * self.stemR(x)
        for n, lL, lR in zip(self.norms, self.midL, self.midR):
            hn = n(h)
            h = lL(hn) * lR(hn)
        h = self.norm_out(h)
        h = F.adaptive_avg_pool2d(h, self.grid).flatten(1)
        return self.head(h)

    def forward(self, x, targets=None):
        lg = self.logits(x)
        loss = None if targets is None else F.cross_entropy(lg, targets)
        return lg, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
