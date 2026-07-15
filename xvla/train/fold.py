"""Fold frozen scalar RmsBatchNorm into adjacent weights (spec §7.2–7.5, §19).

A frozen RmsBatchNorm is a fixed scalar division ``x → x/c``. Folding absorbs
``c`` into the neighbouring projection so the module can be deleted, leaving an
inference graph with **no normalization division at all** (spec §19). Because
``u = x/c`` feeds ``W u + b = (W/c) x + b``, we divide only the *weight* by the
block-norm constant (the bias is added after and is untouched); a per-branch
Q/K/V scalar divides both that branch's weight and bias.

``fold_lm`` returns a folded copy and the max |pre − post| output discrepancy,
which must be < 1e-5 (the §19 acceptance gate).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from xvla.models.lm import ChiLanguageModel
from xvla.nn.normalization import RmsBatchNorm, HomotopyNorm


def _c(module: nn.Module) -> torch.Tensor:
    if isinstance(module, HomotopyNorm):
        if abs(float(module.kappa) - 1.0) > 1e-6:
            raise ValueError(f"HomotopyNorm not fully annealed (κ={float(module.kappa):.3f}); "
                             "only κ=1 sites are foldable.")
        return module.scale.detach()
    if not isinstance(module, RmsBatchNorm):
        raise TypeError(f"expected RmsBatchNorm/HomotopyNorm to fold, got {type(module).__name__}. "
                        "Folding only applies to the strict scalar-norm model.")
    return module.scale.detach()


@torch.no_grad()
def fold_lm(model: ChiLanguageModel) -> ChiLanguageModel:
    """Return a folded, normalization-free copy of a scalar-norm ChiLanguageModel."""
    m = copy.deepcopy(model)
    for block in m.blocks:
        # Block pre-attention norm folds into all six attention input weights.
        c_attn = _c(block.rbn_attn)
        attn = block.attn
        for lin in (attn.wq1, attn.wk1, attn.wq2, attn.wk2, attn.wv):
            lin.weight.div_(c_attn)
        block.rbn_attn = nn.Identity()

        # Per-branch Q/K/V norms fold into their own weight AND bias.
        if attn.qk_norm in ("scalar_rbn", "homotopy"):
            pairs = [(attn.wq1, "rbn_q1"), (attn.wk1, "rbn_k1"),
                     (attn.wq2, "rbn_q2"), (attn.wk2, "rbn_k2"), (attn.wv, "rbn_v")]
            for lin, name in pairs:
                rbn = getattr(attn, name, None)
                if rbn is None:  # homotopy leaves V un-normalized
                    continue
                cb = _c(rbn)
                lin.weight.div_(cb)
                lin.bias.div_(cb)
                delattr(attn, name)  # exported graph is normalization-free (spec §19)
            attn.qk_norm = "none"

        # Block pre-FFN norm folds into the FFN's two input projections.
        c_ffn = _c(block.rbn_ffn)
        block.ffn.left.weight.div_(c_ffn)
        block.ffn.right.weight.div_(c_ffn)
        block.rbn_ffn = nn.Identity()

    # Final norm folds into the head. The head is tied to the embedding, so we
    # untie: fold into a fresh, independent head weight (embedding untouched).
    c_out = _c(m.rbn_out)
    folded_head = nn.Linear(m.head.in_features, m.head.out_features, bias=False).to(
        device=m.head.weight.device, dtype=m.head.weight.dtype)
    folded_head.weight.copy_(m.head.weight / c_out)
    m.head = folded_head
    m.cfg.tie_embeddings = False
    m.rbn_out = nn.Identity()
    return m


def _is_norm(m: nn.Module) -> bool:
    from xvla.nn.normalization import PerTokenRmsNorm
    return isinstance(m, (RmsBatchNorm, HomotopyNorm, PerTokenRmsNorm))


@torch.no_grad()
def verify_fold(model: ChiLanguageModel, x: torch.Tensor):
    """Fold and return (folded_model, max_abs_diff, max_rel_diff) on input ``x``.

    Folding is an exact algebraic identity; any discrepancy is floating-point
    accumulation. Absolute error grows with logit magnitude (a trained, confident
    model has large logits), so the relative error is the fairer precision metric
    — and a FP64 check (see tests/test_fold.py) folds to ~1e-12.
    """
    model = model.eval()
    folded = fold_lm(model).eval()
    before, _ = model(x)
    after, _ = folded(x)
    abs_diff = (before.float() - after.float()).abs()
    max_abs = abs_diff.max().item()
    max_rel = (abs_diff / before.float().abs().clamp_min(1e-6)).max().item()
    return folded, max_abs, max_rel
