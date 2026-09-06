"""χ-net architecture matching the constraints reported by Dooms et al.

This module is intentionally separate from :mod:`xvla.models.chi_mlp`. The
VLA-oriented model uses a three-factor CP bilinear block
``D[(Lx) * (Rx)]``. Equation 2 of *Compositionality Unlocks Deep
Interpretable Models* instead uses the direct transposed Khatri-Rao map
``(Ax) * (Bx)`` with no learned output projection.

The paper leaves initialization and several training details unspecified, so
this class supports a paper-reported-configuration reproduction rather than a
claim of checkpoint identity with the authors' unpublished implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from xvla.nn.homogeneous import to_homogeneous_matrix
from xvla.nn.normalization import make_norm


PAPER_REPORTED_ARCHITECTURE_SCHEMA = "dooms-xnet-direct-ab-equation2-v1"


@dataclass(frozen=True)
class DoomsReportedChiNetConfig:
    """Architecture values explicitly reported for the three-layer SVHN model."""

    in_dim: int = 1024
    dim: int = 256
    n_layers: int = 3
    num_classes: int = 10
    norm: str = "scalar_rbn"


class DirectBilinearLayer(nn.Module):
    """Direct Equation-2 bilinear map ``(A x) * (B x)``."""

    def __init__(self, dim: int):
        super().__init__()
        if dim < 1:
            raise ValueError("dimension must be positive")
        self.dim = dim
        self.left = nn.Linear(dim, dim, bias=True)
        self.right = nn.Linear(dim, dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # The paper does not report initialization. This explicit assumption is
        # serialized by the reproduction runner.
        nn.init.normal_(self.left.weight, std=self.dim**-0.5)
        nn.init.normal_(self.right.weight, std=self.dim**-0.5)
        nn.init.zeros_(self.left.bias)
        nn.init.zeros_(self.right.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.left(x) * self.right(x)

    @torch.no_grad()
    def dense_core(self) -> torch.Tensor:
        """Return the post-hoc symmetrized homogeneous Equation-2 core."""

        left = to_homogeneous_matrix(self.left.weight, self.left.bias)
        right = to_homogeneous_matrix(self.right.weight, self.right.bias)
        core = torch.einsum("oi,oj->oij", left, right)
        return 0.5 * (core + core.transpose(1, 2))


class DoomsReportedChiNet(nn.Module):
    """Three-layer direct-bilinear classifier from the paper's reported setup."""

    architecture_schema = PAPER_REPORTED_ARCHITECTURE_SCHEMA

    def __init__(self, cfg: DoomsReportedChiNetConfig | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else DoomsReportedChiNetConfig()
        self.embed = nn.Linear(self.cfg.in_dim, self.cfg.dim)
        self.norms = nn.ModuleList(
            make_norm(self.cfg.norm) for _ in range(self.cfg.n_layers)
        )
        self.layers = nn.ModuleList(
            DirectBilinearLayer(self.cfg.dim) for _ in range(self.cfg.n_layers)
        )
        self.head = nn.Linear(self.cfg.dim, self.cfg.num_classes)

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self.embed(x.flatten(1))
        for norm, layer in zip(self.norms, self.layers):
            hidden = layer(norm(hidden))
        logits = self.head(hidden)
        loss = None if targets is None else F.cross_entropy(logits, targets)
        return logits, loss

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
