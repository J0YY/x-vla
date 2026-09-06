"""Exact structural tests for the paper-reported direct-AB chi-net."""

from __future__ import annotations

import pytest
import torch

from xvla.models.dooms_xnet import (
    PAPER_REPORTED_ARCHITECTURE_SCHEMA,
    DirectBilinearLayer,
    DoomsReportedChiNet,
    DoomsReportedChiNetConfig,
)
from xvla.nn.normalization import RmsBatchNorm
from xvla.train.canonical_odt import export_homogeneous_network, homogeneous_forward


DTYPE = torch.float64


def test_reported_config_matches_explicit_paper_architecture_constraints():
    config = DoomsReportedChiNetConfig()
    assert config.in_dim == 1024
    assert config.dim == 256
    assert config.n_layers == 3
    assert config.num_classes == 10
    assert config.norm == "scalar_rbn"
    model = DoomsReportedChiNet(config)
    assert model.architecture_schema == PAPER_REPORTED_ARCHITECTURE_SCHEMA
    assert len(model.layers) == 3
    assert all(type(layer) is DirectBilinearLayer for layer in model.layers)
    assert all(not hasattr(layer, "down") for layer in model.layers)
    assert all(layer.left.out_features == 256 for layer in model.layers)
    assert all(layer.right.out_features == 256 for layer in model.layers)


def test_direct_bilinear_dense_core_is_exact_symmetric_equation2_map():
    torch.manual_seed(901)
    layer = DirectBilinearLayer(5).double().eval()
    with torch.no_grad():
        layer.left.bias.normal_(std=0.2)
        layer.right.bias.normal_(std=0.2)
    inputs = torch.randn(17, 5, dtype=DTYPE)
    ones = torch.ones(len(inputs), 1, dtype=DTYPE)
    homogeneous = torch.cat([ones, inputs], dim=1)
    core = layer.dense_core()
    expected = layer(inputs)
    actual = torch.einsum("oij,bi,bj->bo", core, homogeneous, homogeneous)
    assert core.shape == (5, 6, 6)
    assert torch.equal(core, core.transpose(1, 2))
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_reported_chinet_homogeneous_export_replays_all_biases_and_rbn():
    torch.manual_seed(902)
    config = DoomsReportedChiNetConfig(
        in_dim=7,
        dim=4,
        n_layers=3,
        num_classes=3,
    )
    model = DoomsReportedChiNet(config).double().eval()
    with torch.no_grad():
        model.embed.bias.normal_(std=0.2)
        model.head.bias.normal_(std=0.2)
        for index, (norm, layer) in enumerate(zip(model.norms, model.layers)):
            assert isinstance(norm, RmsBatchNorm)
            norm.running_rms.fill_(1.1 + 0.2 * index)
            norm.initialized.fill_(True)
            norm.freeze()
            layer.left.bias.normal_(std=0.2)
            layer.right.bias.normal_(std=0.2)
    inputs = torch.randn(19, 7, dtype=DTYPE)
    network = export_homogeneous_network(model, require_frozen_rbn=True)
    expected = model(inputs)[0]
    actual = homogeneous_forward(network, inputs)
    assert network.bond_dims == (5, 5, 5, 5)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_production_export_rejects_every_nonfrozen_rbn_state():
    model = DoomsReportedChiNet(
        DoomsReportedChiNetConfig(in_dim=7, dim=4, n_layers=3, num_classes=3)
    ).double().eval()
    with pytest.raises(ValueError, match="not initialized and frozen"):
        export_homogeneous_network(model, require_frozen_rbn=True)

    for norm in model.norms:
        assert isinstance(norm, RmsBatchNorm)
        norm.initialized.fill_(True)
    with pytest.raises(ValueError, match="not initialized and frozen"):
        export_homogeneous_network(model, require_frozen_rbn=True)

    for norm in model.norms:
        norm.freeze()
    model.norms[1].calibrating = True
    with pytest.raises(ValueError, match="not initialized and frozen"):
        export_homogeneous_network(model, require_frozen_rbn=True)


def test_production_export_rejects_nonfinite_or_nonpositive_fold_scale():
    model = DoomsReportedChiNet(
        DoomsReportedChiNetConfig(in_dim=7, dim=4, n_layers=3, num_classes=3)
    ).double().eval()
    for norm in model.norms:
        assert isinstance(norm, RmsBatchNorm)
        norm.initialized.fill_(True)
        norm.freeze()
    model.norms[0].running_rms.fill_(float("nan"))
    with pytest.raises(ValueError, match="invalid fold scale"):
        export_homogeneous_network(model, require_frozen_rbn=True)

    for value in (-model.norms[0].eps, -1.0):
        model.norms[0].running_rms.fill_(value)
        with pytest.raises(ValueError, match="invalid fold scale"):
            export_homogeneous_network(model, require_frozen_rbn=True)
