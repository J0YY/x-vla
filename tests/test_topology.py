"""Exactness tests for the topology-sweep quadratic decomposition (Experiment A)."""

import torch

from xvla.models.chi_conv import ShallowBilinear
from xvla.train.topology import class_quadratics, quad_logits


def _check(mode, kernel=3, readout="global", grid=3):
    torch.manual_seed(0)
    m = ShallowBilinear(mode=mode, width=5, kernel=kernel, in_ch=2, hw=6, num_classes=4,
                        readout=readout, grid=grid)
    # randomize biases so linear/const parts are non-trivial
    for p in m.parameters():
        p.data = p.data + 0.1 * torch.randn_like(p)
    dev = "cpu"
    Q, l, a = class_quadratics(m, dev)
    D = m.in_ch * m.hw * m.hw
    x = torch.randn(7, D, dtype=torch.float64)
    ref = m.to(dev).double().logits(x.reshape(-1, m.in_ch, m.hw, m.hw))
    got = quad_logits(Q, l, a, x)
    assert torch.allclose(ref, got, atol=1e-9), (mode, (ref - got).abs().max().item())


def test_dense_quadratic_exact():
    _check("dense")


def test_conv_quadratic_exact():
    _check("conv", kernel=3)


def test_local_quadratic_exact():
    _check("local", kernel=3)


def test_conv_spatial_readout_exact():
    _check("conv", kernel=3, readout="spatial", grid=3)


def test_local_spatial_readout_exact():
    _check("local", kernel=3, readout="spatial", grid=2)


def test_input_analysis_runs_deep():
    """input_analysis (data-driven, any-depth) runs end-to-end on a deep conv model:
    guards the enable_grad + dtype paths that broke the A100 runs."""
    from xvla.models.chi_conv import ConvBilinearDeep
    from xvla.train.topology import input_analysis
    torch.manual_seed(0)
    m = ConvBilinearDeep(depth=2, width=4, kernel=3, grid=2, in_ch=2, hw=8, num_classes=3).eval()
    X = torch.randn(24, 2, 8, 8); Y = torch.randint(0, 3, (24,))
    res = input_analysis(lambda z: m.logits(z), X, Y, "cpu", ks=(1, 4, 16))
    assert res["locality_ratio"] > 0 and res["locality_ratio"] == res["locality_ratio"]  # finite
    assert set(res["global_curve"]) == {1, 4, 16}


def test_Q_is_symmetric():
    torch.manual_seed(1)
    m = ShallowBilinear(mode="conv", width=4, kernel=3, in_ch=1, hw=6, num_classes=3)
    Q, _, _ = class_quadratics(m, "cpu")
    assert torch.allclose(Q, Q.transpose(1, 2), atol=1e-10)
