"""End-to-end fold equivalence for the strict scalar-norm LM (spec §19).

Builds a small strict (scalar-norm) χ-LM, calibrates its RmsBatchNorm constants,
folds them all into adjacent weights, and asserts the folded (normalization-free)
graph matches the pre-fold model — the §19 "pre-fold and post-fold numerical
outputs match" acceptance gate — and that no normalization modules remain.
"""

import torch

from xvla.models.lm import ChiLanguageModel, LMConfig
from xvla.nn.normalization import RmsBatchNorm, PerTokenRmsNorm
from xvla.train.calibrate import calibrate_rbn
from xvla.train.fold import fold_lm, verify_fold


def _strict_lm():
    cfg = LMConfig(vocab_size=256, dim=64, n_layers=4, n_heads=4, max_seq_len=32,
                   attn="bilinear", ffn="bilinear",
                   norm="scalar_rbn", qk_norm="scalar_rbn",
                   learned_gain=True, tie_embeddings=True)
    return ChiLanguageModel(cfg).double()


def test_strict_lm_folds_to_pure_graph():
    torch.manual_seed(0)
    model = _strict_lm()

    def batch():
        idx = torch.randint(0, 256, (4, 32))
        return idx, idx
    # Warm up + calibrate the scalar RBN constants, then freeze.
    model.train()
    for _ in range(20):
        x, _ = batch()
        model(x)
    calibrate_rbn(model, batch, iters=30)
    for m in model.modules():
        if isinstance(m, RmsBatchNorm):
            m.frozen = True

    x, _ = batch()
    folded, abs_diff, rel_diff = verify_fold(model, x)
    # FP64 fold is an exact algebraic identity — expect near machine precision.
    assert abs_diff < 1e-5, f"fold mismatch {abs_diff:.2e} exceeds §19 gate 1e-5"

    # The folded inference graph must contain NO normalization modules.
    leftover = [n for n, mm in folded.named_modules()
                if isinstance(mm, (RmsBatchNorm, PerTokenRmsNorm))]
    assert leftover == [], f"normalization modules remain after fold: {leftover}"


def test_fold_requires_scalar_norm():
    # Folding a per-token model must raise (per-token norm is not foldable).
    cfg = LMConfig(vocab_size=256, dim=64, n_layers=2, n_heads=4, max_seq_len=32,
                   norm="per_token", qk_norm="per_token")
    model = ChiLanguageModel(cfg).double()
    import pytest
    with pytest.raises(TypeError):
        fold_lm(model)
