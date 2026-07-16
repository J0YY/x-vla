"""Cross-bilinear CP projector — dual-encoder fusion (spec §8, Milestone 3).

Fuses aligned spatial (s) and semantic (m) patch features so that spatial×semantic
interactions are exposed explicitly rather than left for the backbone to recover:

    p_i = W_s s_i + W_m m_i + D_p[ (L_s s̄_i) ⊙ (R_m m̄_i) ]

with s̄=[1;s], m̄=[1;m]. The two linear paths keep spatial-only and semantic-only
information; the CP interaction term (rank r_p, default 2·d_out) is a factorization
of the cross-tensor whose rank components read as "this spatial direction × this
semantic direction". All three paths are tensor-decomposable. ``ConcatLinear`` is
the matched, interaction-free baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossBilinearProjector(nn.Module):
    def __init__(self, d_spatial: int, d_semantic: int, d_out: int, rank: int | None = None):
        super().__init__()
        self.rank = rank if rank is not None else 2 * d_out
        self.w_s = nn.Linear(d_spatial, d_out, bias=False)
        self.w_m = nn.Linear(d_semantic, d_out, bias=True)     # one bias for the sum
        self.l_s = nn.Linear(d_spatial, self.rank, bias=True)  # bias = homogeneous s̄
        self.r_m = nn.Linear(d_semantic, self.rank, bias=True) # bias = homogeneous m̄
        self.d_p = nn.Linear(self.rank, d_out, bias=False)
        self.use_interaction = True
        nn.init.normal_(self.l_s.weight, std=d_spatial ** -0.5)
        nn.init.normal_(self.r_m.weight, std=d_semantic ** -0.5)
        nn.init.normal_(self.d_p.weight, std=(self.rank ** -0.5) * 0.5)

    def forward(self, s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        p = self.w_s(s) + self.w_m(m)
        if self.use_interaction:
            p = p + self.d_p(self.l_s(s) * self.r_m(m))
        return p


class ConcatLinear(nn.Module):
    """Matched baseline: concat(s, m) → Linear (depth=1) or 2-layer MLP (depth=2)."""

    def __init__(self, d_spatial: int, d_semantic: int, d_out: int, depth: int = 1,
                 hidden: int | None = None):
        super().__init__()
        d_in = d_spatial + d_semantic
        if depth == 1:
            self.net = nn.Linear(d_in, d_out)
        else:
            h = hidden or d_out
            self.net = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, d_out))

    def forward(self, s, m):
        return self.net(torch.cat([s, m], dim=-1))
