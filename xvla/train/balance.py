"""CP factor balancing (spec §13.2).

CP factorizations have a scale-gauge freedom::

    l_r → a·l_r ,  r_r → b·r_r ,  d_r → (1/ab)·d_r

leaving the represented tensor unchanged while letting individual factor norms
drift and become ill-conditioned. Periodically (every ``N`` optimizer steps) we
redistribute scale across each rank component ``r`` toward its geometric mean::

    g_r = ( ‖l_r‖ · ‖r_r‖ · ‖d_r‖ )^{1/3}

For :class:`~xvla.nn.bilinear.BilinearFFN` the factors are, per rank ``r``:
row ``r`` of ``left`` (weight+bias, homogeneous), row ``r`` of ``right``, and
column ``r`` of ``down``. The rescale preserves the function exactly because the
product of the three per-rank scalings is 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xvla.nn.bilinear import BilinearFFN


@torch.no_grad()
def balance_bilinear_ffn(ffn: BilinearFFN, eps: float = 1e-12) -> float:
    """Rebalance one BilinearFFN in place. Returns the max pre-balance imbalance
    (max/min of per-rank factor norms, a condition-style diagnostic)."""
    # l_r norm includes the homogeneous bias coordinate.
    l_norm = torch.sqrt(ffn.left.weight.pow(2).sum(dim=1) + ffn.left.bias.pow(2) + eps)
    r_norm = torch.sqrt(ffn.right.weight.pow(2).sum(dim=1) + ffn.right.bias.pow(2) + eps)
    d_norm = torch.sqrt(ffn.down.weight.pow(2).sum(dim=0) + eps)  # column r

    g = (l_norm * r_norm * d_norm).clamp_min(eps).pow(1.0 / 3.0)
    sl = (g / l_norm).unsqueeze(1)
    sr = (g / r_norm).unsqueeze(1)
    sd = (g / d_norm).unsqueeze(0)

    ffn.left.weight.mul_(sl); ffn.left.bias.mul_(sl.squeeze(1))
    ffn.right.weight.mul_(sr); ffn.right.bias.mul_(sr.squeeze(1))
    ffn.down.weight.mul_(sd)

    all_norms = torch.cat([l_norm, r_norm, d_norm])
    return float(all_norms.max() / all_norms.min().clamp_min(eps))


@torch.no_grad()
def balance_model(model: nn.Module) -> float:
    """Balance every BilinearFFN in a model; return the worst imbalance seen."""
    worst = 0.0
    for m in model.modules():
        if isinstance(m, BilinearFFN):
            worst = max(worst, balance_bilinear_ffn(m))
    return worst


@torch.no_grad()
def spectral_clip_model(model: nn.Module, budget: float) -> float:
    """Spectral-clip every BilinearFFN's degree-2 gain to ``budget`` (foldable).

    Returns the largest pre-clip ‖D‖·‖L‖·‖R‖ product seen (a diagnostic of how
    hard the degree-2 amplification is pushing). Run after ``balance_model`` so
    the spectral norms are gauge-fair."""
    worst = 0.0
    for m in model.modules():
        if isinstance(m, BilinearFFN):
            worst = max(worst, m.spectral_clip(budget))
    return worst
