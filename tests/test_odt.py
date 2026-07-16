"""Exact-reconstruction tests for the χ-MLP ODT pipeline (spec §14/§19, M5)."""

import torch

from xvla.models.chi_mlp import ChiMLP, ChiMLPConfig
from xvla.nn.normalization import RmsBatchNorm
from xvla.train.odt import (export_cores, unroll_forward, downstream_gram,
                            local_gram, top_projector)


def _trained_stub():
    torch.manual_seed(0)
    cfg = ChiMLPConfig(in_dim=48, dim=8, n_layers=3, num_classes=5, norm="scalar_rbn")
    model = ChiMLP(cfg).double()
    # warm up + freeze the scalar norms so they fold to fixed constants.
    model.train()
    for _ in range(30):
        model(torch.randn(16, 48, dtype=torch.float64))
    for m in model.modules():
        if isinstance(m, RmsBatchNorm):
            m.eval(); m.freeze()
    model.eval()
    return model, cfg


def test_unroll_matches_module():
    model, _ = _trained_stub()
    x = torch.randn(20, 48, dtype=torch.float64)
    embed, cores, head = export_cores(model)
    ref = model(x)[0]
    got = unroll_forward(embed, cores, head, x)
    assert torch.allclose(ref, got, atol=1e-9), (ref - got).abs().max()


def test_full_rank_truncation_is_identity():
    model, cfg = _trained_stub()
    x = torch.randn(20, 48, dtype=torch.float64)
    embed, cores, head = export_cores(model)
    full = unroll_forward(embed, cores, head, x)
    G = downstream_gram(cores, head, bond=1)
    P, _ = top_projector(G, cfg.dim)              # k = d → projector is identity
    trunc = unroll_forward(embed, cores, head, x, proj=P, proj_bond=1)
    assert torch.allclose(full, trunc, atol=1e-9), (full - trunc).abs().max()


def test_gram_shapes_and_psd():
    model, cfg = _trained_stub()
    embed, cores, head = export_cores(model)
    for G in (downstream_gram(cores, head, 1), local_gram(cores, 1),
              downstream_gram(cores, head, 2), local_gram(cores, 2)):
        assert G.shape == (cfg.dim, cfg.dim)
        # Gram matrices are symmetric PSD.
        assert torch.allclose(G, G.T, atol=1e-9)
        assert torch.linalg.eigvalsh(G).min() > -1e-9
