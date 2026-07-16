"""Exactness tests for ODT *interpretation* (output-conditioned quadratics, M5→M7).

These lock the math that the coherence/faithfulness story rests on: for a
single-bilinear χ-classifier the class logit really is the exact quadratic form
z^T Q_c z + b_c, and the eigen-atom / faithfulness reconstructions are exact.
"""

import torch

from xvla.models.chi_mlp import ChiMLP, ChiMLPConfig
from xvla.nn.normalization import RmsBatchNorm
from xvla.train.odt import export_cores
from xvla.train.odt_interp import (class_quadratics, eig_atoms, quad_logits,
                                   faithfulness, atom_to_pixels, locality)


def _shallow_stub():
    torch.manual_seed(0)
    cfg = ChiMLPConfig(in_dim=48, dim=16, n_layers=1, num_classes=5, norm="scalar_rbn")
    model = ChiMLP(cfg).double()
    model.train()
    for _ in range(30):
        model(torch.randn(16, 48, dtype=torch.float64))
    for m in model.modules():
        if isinstance(m, RmsBatchNorm):
            m.eval(); m.freeze()
    model.eval()
    return model, cfg


def test_quadratic_form_is_exact():
    """logit_c(x) == z^T Q_c z + b_c to fp64 (the whole interpretation rests here)."""
    model, _ = _shallow_stub()
    x = torch.randn(20, 48, dtype=torch.float64)
    embed, cores, head = export_cores(model)
    Q, b = class_quadratics(cores, head)
    ref = model(x)[0]
    got = quad_logits(Q, b, x, embed)
    assert torch.allclose(ref, got, atol=1e-9), (ref - got).abs().max()


def test_eig_reconstruction_and_ordering():
    """Σ_i λ_i (v_i·z)^2 reconstructs z^T Q_c z; |λ| is descending."""
    model, _ = _shallow_stub()
    embed, cores, head = export_cores(model)
    Q, _ = class_quadratics(cores, head)
    z = torch.randn(8, Q.shape[1], dtype=torch.float64)
    for c in range(Q.shape[0]):
        l, v = eig_atoms(Q[c])
        assert torch.all(l.abs()[:-1] >= l.abs()[1:] - 1e-12)     # |λ| descending
        recon = ((z @ v) ** 2 * l).sum(-1)
        direct = torch.einsum("bi,ij,bj->b", z, Q[c], z)
        assert torch.allclose(recon, direct, atol=1e-9)


def test_faithfulness_full_keep_equals_base():
    """keep-all == base accuracy, and drop-all collapses to constant (b only)."""
    model, cfg = _shallow_stub()
    x = torch.randn(64, 48, dtype=torch.float64)
    embed, cores, head = export_cores(model)
    Q, b = class_quadratics(cores, head)
    y = quad_logits(Q, b, x, embed).argmax(-1)
    n = Q.shape[1]
    res = faithfulness(Q, b, x, y, embed, ranks=[n, 0])
    assert abs(res["keep"][n] - 1.0) < 1e-9            # keep every term == labels
    assert abs(res["drop_top"][n] - res["drop_random"][n]) < 1e-9   # both drop all


def test_atom_locality_range():
    """Locality is a fraction in (0,1]; a one-hot pattern is maximally local."""
    model, _ = _shallow_stub()
    embed, cores, head = export_cores(model)
    Q, _ = class_quadratics(cores, head)
    std = torch.ones(3, 4, 4).double()                  # in_dim 48 = 3*4*4
    _, v = eig_atoms(Q[0])
    p = atom_to_pixels(v[:, 0], embed, (3, 4, 4), std)
    loc = locality(p, frac=0.25)
    assert 0.0 < loc <= 1.0
    spike = torch.zeros(3, 4, 4); spike[0, 0, 0] = 1.0
    assert locality(spike, frac=1.0 / 16) > 0.99        # all energy in one pixel
