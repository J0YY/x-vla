"""Bilinear (CP-factorized) FFN — spec §5.

Replaces SwiGLU ``D(swish(Lx) ⊙ Rx)`` with the activation-free bilinear form::

    BFFN(x) = D[ (L x̄) ⊙ (R x̄) ]

where ``x̄ = [1; x]`` (implemented as ordinary Linear biases on L, R). Writing
it out per output coordinate::

    y_o = Σ_r D_or (L_r·x̄)(R_r·x̄) = Σ_ij ( Σ_r D_or L_ri R_rj ) x̄_i x̄_j

so BFFN is exactly a CP factorization of a dense third-order tensor
``T ∈ R^{d_out × (d+1) × (d+1)}`` with parameter cost ≈ 3dr instead of ≈ d³.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xvla.nn.homogeneous import to_homogeneous_matrix


class BilinearFFN(nn.Module):
    """CP-factorized bilinear FFN (spec §5).

    Args:
        dim: input/output width d.
        rank: CP rank r (spec default r = 3d for both vision and joint towers).
        out_dim: output width (defaults to ``dim``).
        down_bias: whether the down projection carries a bias. Defaults False:
            the constant×constant term already present in ``(Lx̄)⊙(Rx̄)`` makes a
            separate down bias redundant, and omitting it keeps the module a
            pure CP core (spec §5 math has ``D`` without bias).
    """

    def __init__(
        self,
        dim: int,
        rank: int | None = None,
        out_dim: int | None = None,
        down_bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.rank = rank if rank is not None else 3 * dim
        # Biases on L, R realise the homogeneous coordinate x̄ = [1; x].
        self.left = nn.Linear(dim, self.rank, bias=True)
        self.right = nn.Linear(dim, self.rank, bias=True)
        self.down = nn.Linear(self.rank, self.out_dim, bias=down_bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Input projections ~ 1/sqrt(d_in); down at reduced scale (spec §13.1).
        nn.init.normal_(self.left.weight, std=self.dim ** -0.5)
        nn.init.normal_(self.right.weight, std=self.dim ** -0.5)
        nn.init.zeros_(self.left.bias)
        nn.init.zeros_(self.right.bias)
        nn.init.normal_(self.down.weight, std=(self.rank ** -0.5) * 0.5)
        if self.down.bias is not None:
            nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.left(x) * self.right(x))

    @torch.no_grad()
    def spectral_clip(self, budget: float) -> float:
        """Cap the degree-2 gain by bounding ‖D‖₂·‖L_lin‖₂·‖R_lin‖₂ ≤ budget.

        Operates only on the *weight* matrices (the parts that multiply x), not on
        the biases — the affine/constant channel is left free (the linear vs.
        quadratic decoupling). Rescaling weights is foldable: at inference the
        clipped weights are just matrices. Returns the pre-clip product.
        """
        sL = torch.linalg.matrix_norm(self.left.weight, ord=2)
        sR = torch.linalg.matrix_norm(self.right.weight, ord=2)
        sD = torch.linalg.matrix_norm(self.down.weight, ord=2)
        prod = (sL * sR * sD).item()
        if prod > budget:
            s = (budget / prod) ** (1.0 / 3.0)
            self.left.weight.mul_(s)
            self.right.weight.mul_(s)
            self.down.weight.mul_(s)
        return prod

    # -- tensor-network export ------------------------------------------------
    @torch.no_grad()
    def dense_core(self) -> torch.Tensor:
        """Materialize the dense third-order tensor ``T`` (spec §5).

        Returns ``T`` of shape ``(out_dim, dim+1, dim+1)`` such that
        ``y = einsum('oij,i,j->o', T, x̄, x̄)`` reproduces ``forward`` **when the
        down projection has no bias**. Intended for small-dimension operator
        validation only (cost is O(out·(d+1)²·r)).
        """
        if self.down.bias is not None:
            raise ValueError("dense_core requires down_bias=False for exact equivalence")
        L = to_homogeneous_matrix(self.left.weight, self.left.bias)   # (r, d+1)
        R = to_homogeneous_matrix(self.right.weight, self.right.bias)  # (r, d+1)
        D = self.down.weight  # (out, r)
        # T_oij = Σ_r D_or L_ri R_rj
        return torch.einsum("or,ri,rj->oij", D, L, R)
