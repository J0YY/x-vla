"""Homogeneous-coordinate convention (spec §3).

Dooms et al. append a constant coordinate so a stack of ``L`` self-bilinear
layers can express *every* polynomial degree up to ``2**L`` (biases, linear
terms, pairwise and higher-order interactions) rather than being forced onto
the top-degree term only::

    x̄ = [1; x]

In the *implementation* we simply use ordinary ``nn.Linear`` biases. At tensor-
network *export* those biases are converted into the constant row of a
homogeneous weight matrix. These helpers implement that conversion so the
operator-validation tests can check the affine ↔ homogeneous equivalence.
"""

from __future__ import annotations

import torch


def homogenize(x: torch.Tensor) -> torch.Tensor:
    """Prepend a constant-1 coordinate along the last dimension: ``x -> [1; x]``."""
    ones = x.new_ones(*x.shape[:-1], 1)
    return torch.cat([ones, x], dim=-1)


def to_homogeneous_matrix(weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Fold an affine map ``y = W x + b`` into a single homogeneous matrix ``W̄``.

    Given ``W`` of shape ``(out, in)`` and ``b`` of shape ``(out,)``, returns
    ``W̄`` of shape ``(out, in + 1)`` such that ``W̄ @ [1; x] == W x + b``. The
    constant coordinate is the leading column.
    """
    out_dim = weight.shape[0]
    if bias is None:
        bias = weight.new_zeros(out_dim)
    return torch.cat([bias.unsqueeze(1), weight], dim=1)
