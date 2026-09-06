"""Exact tests for strict scalar χ-VLA folding."""

from __future__ import annotations

import copy

import pytest
import torch

from xvla.models.vla import ChiVLA, VLAConfig
from xvla.nn.normalization import RmsBatchNorm
from xvla.train.calibrate import calibrate_rbn_sequential
from xvla.train.fold_vla import (
    _metadata_sha256,
    fold_vla,
    load_strict_vla_export_package,
    make_strict_vla_export_package,
    remaining_rbn_paths,
    validate_strict_vla_inputs,
    verify_vla_fold,
)


DTYPE = torch.float64


def _config(**overrides) -> VLAConfig:
    values = dict(
        image_size=8,
        patch_size=4,
        vit_dim=4,
        vit_layers=1,
        vit_heads=2,
        vocab_size=16,
        max_instr_len=3,
        state_dim=2,
        n_embodiments=2,
        dim=4,
        n_layers=1,
        n_heads=2,
        ffn_rank=7,
        vit_ffn_rank=7,
        norm="scalar_rbn",
        qk_norm="scalar_rbn",
        action_horizon=2,
        action_dim=3,
        action_head="linear",
    )
    values.update(overrides)
    return VLAConfig(**values)


def _batch(seed: int = 0, batch_size: int = 3) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "img": torch.randn(batch_size, 3, 8, 8, generator=generator, dtype=DTYPE),
        "instr_ids": torch.randint(0, 16, (batch_size, 3), generator=generator),
        "state": torch.randn(batch_size, 2, generator=generator, dtype=DTYPE),
        "embodiment_id": torch.randint(0, 2, (batch_size,), generator=generator),
    }


def _calibrated_model(seed: int = 0, **config_overrides) -> ChiVLA:
    torch.manual_seed(seed)
    model = ChiVLA(_config(**config_overrides)).double().eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)) and module.bias is not None:
                module.bias.normal_()
    report = calibrate_rbn_sequential(
        model,
        lambda index: _batch(seed + 100 + index),
        lambda calibrated, data: calibrated(**data),
        iters=8,
        excluded_paths=("vision.norm_out",),
    )
    assert report.sites
    assert all(site.batches == 8 for site in report.sites)
    return model


def test_strict_vla_fold_matches_deployed_actions_and_removes_norms():
    model = _calibrated_model(1)
    result = verify_vla_fold(model, _batch(2))
    assert result.maximum_absolute_error < 1e-10
    assert result.maximum_relative_error < 1e-9
    assert result.remaining_norm_paths == ()
    assert result.removed_dead_paths == ("vision.norm_out",)
    assert remaining_rbn_paths(result.model) == ()
    assert result.model.cfg.norm == "none"
    assert result.model.cfg.qk_norm == "none"


def test_multilayer_vision_and_joint_stacks_calibrate_and_fold_exactly():
    model = _calibrated_model(29, vit_layers=2, n_layers=2)
    result = verify_vla_fold(model, _batch(30))
    assert result.maximum_absolute_error < 1e-10
    assert result.maximum_relative_error < 1e-9
    assert len(result.model.vision.blocks.blocks) == 2
    assert len(result.model.backbone.blocks) == 2


def test_each_scalar_is_absorbed_only_into_its_declared_consumers():
    torch.manual_seed(31)
    model = ChiVLA(_config()).double().eval()
    live_norms = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, RmsBatchNorm) and name != "vision.norm_out"
    ]
    for index, (_, norm) in enumerate(live_norms):
        norm.running_rms.fill_(2.0 + 0.1 * index - norm.eps)
        norm.initialized.fill_(True)
        norm.frozen = True
    block = model.backbone.blocks[0]
    snapshots = {
        name: tensor.detach().clone()
        for name, tensor in block.state_dict().items()
        if ".running_rms" not in name and "initialized" not in name
    }
    action_weight = model.action_head.weight.detach().clone()
    action_bias = model.action_head.bias.detach().clone()
    pre_attention = float(block.rbn_attn.scale)
    branch_scales = {
        name: float(getattr(block.attn, name).scale)
        for name in ("rbn_q1", "rbn_k1", "rbn_q2", "rbn_k2", "rbn_v")
    }
    pre_ffn = float(block.rbn_ffn.scale)
    output_scale = float(model.norm_out.scale)
    folded = fold_vla(model)
    folded_block = folded.backbone.blocks[0]
    for projection_name, norm_name in (
        ("wq1", "rbn_q1"),
        ("wk1", "rbn_k1"),
        ("wq2", "rbn_q2"),
        ("wk2", "rbn_k2"),
        ("wv", "rbn_v"),
    ):
        projection = getattr(folded_block.attn, projection_name)
        torch.testing.assert_close(
            projection.weight,
            snapshots[f"attn.{projection_name}.weight"] / pre_attention / branch_scales[norm_name],
        )
        torch.testing.assert_close(
            projection.bias,
            snapshots[f"attn.{projection_name}.bias"] / branch_scales[norm_name],
        )
    torch.testing.assert_close(folded_block.attn.wo.weight, snapshots["attn.wo.weight"])
    torch.testing.assert_close(folded_block.attn.wo.bias, snapshots["attn.wo.bias"])
    for name in ("left", "right"):
        torch.testing.assert_close(
            getattr(folded_block.ffn, name).weight,
            snapshots[f"ffn.{name}.weight"] / pre_ffn,
        )
        torch.testing.assert_close(
            getattr(folded_block.ffn, name).bias, snapshots[f"ffn.{name}.bias"]
        )
    torch.testing.assert_close(folded_block.ffn.down.weight, snapshots["ffn.down.weight"])
    torch.testing.assert_close(folded.action_head.weight, action_weight / output_scale)
    torch.testing.assert_close(folded.action_head.bias, action_bias)


def test_uninitialized_or_unfrozen_deployed_norm_fails_closed():
    model = _calibrated_model(3)
    model.backbone.blocks[0].rbn_attn.frozen = False
    with pytest.raises(ValueError, match="calibrated, initialized, and frozen"):
        fold_vla(model)


def test_sequential_calibration_is_replayable_and_leaves_live_sites_frozen():
    torch.manual_seed(10)
    first = ChiVLA(_config()).double().eval()
    second = copy.deepcopy(first)
    kwargs = dict(
        batch_at=lambda index: _batch(500 + index),
        forward_batch=lambda calibrated, data: calibrated(**data),
        iters=4,
        excluded_paths=("vision.norm_out",),
    )
    report_first = calibrate_rbn_sequential(first, **kwargs)
    report_second = calibrate_rbn_sequential(second, **kwargs)
    assert report_first == report_second
    for path, module in first.named_modules():
        if isinstance(module, RmsBatchNorm) and path != "vision.norm_out":
            assert module.frozen and not module.calibrating and bool(module.initialized)


def test_sequential_calibration_rejects_bad_iteration_or_dead_path_contract():
    model = ChiVLA(_config()).double().eval()
    forward = lambda calibrated, data: calibrated(**data)
    with pytest.raises(ValueError, match="positive"):
        calibrate_rbn_sequential(model, lambda _: _batch(), forward, iters=0)
    with pytest.raises(ValueError, match="do not exist"):
        calibrate_rbn_sequential(
            model,
            lambda _: _batch(),
            forward,
            iters=1,
            excluded_paths=("not.a.norm",),
        )
    with pytest.raises(ValueError, match="excluded RBN executed"):
        calibrate_rbn_sequential(
            model,
            lambda _: _batch(),
            forward,
            iters=1,
            excluded_paths=("backbone.blocks.0.rbn_attn",),
        )


def test_training_mode_and_invalid_scale_fail_closed():
    model = _calibrated_model(11)
    model.train()
    with pytest.raises(ValueError, match=r"model\.eval"):
        fold_vla(model)
    with pytest.raises(ValueError, match=r"model\.eval"):
        verify_vla_fold(model, _batch(99))
    assert model.training
    model.eval()
    model.norm_out.running_rms.fill_(float("nan"))
    with pytest.raises(ValueError, match="contains nonfinite values"):
        fold_vla(model)
    model = _calibrated_model(39).half()
    with pytest.raises(ValueError, match="only float32 and float64"):
        fold_vla(model)


def test_nonfinite_live_buffer_and_attention_runtime_drift_fail_closed():
    model = _calibrated_model(15)
    model.backbone.blocks[0].attn_gain.fill_(float("nan"))
    with pytest.raises(ValueError, match="live buffer"):
        fold_vla(model)
    model = _calibrated_model(16)
    model.backbone.blocks[0].attn._score_denom = 0
    with pytest.raises(ValueError, match="score denominator"):
        fold_vla(model)
    model = _calibrated_model(17)
    model.vision.blocks.blocks[0].attn.causal = True
    with pytest.raises(ValueError, match="deployed stack"):
        fold_vla(model)


def test_wrong_action_head_module_fails_closed():
    model = _calibrated_model(18)
    model.action_head = torch.nn.Sequential(model.action_head)
    with pytest.raises((TypeError, ValueError), match="shape signature|exactly Linear"):
        fold_vla(model)


def test_injected_nonlinear_live_wrapper_fails_closed():
    model = _calibrated_model(24)
    model.vis_proj = torch.nn.Sequential(model.vis_proj, torch.nn.ReLU())
    with pytest.raises((TypeError, ValueError), match="shape signature|vis_proj"):
        fold_vla(model)
    model = _calibrated_model(25)
    model.backbone.blocks[0].attn.wo = torch.nn.Sequential(
        model.backbone.blocks[0].attn.wo, torch.nn.ReLU()
    )
    with pytest.raises((TypeError, ValueError), match="shape signature|child-module inventory"):
        fold_vla(model)


def test_config_or_internal_shape_drift_fails_closed():
    model = _calibrated_model(32)
    model.cfg.action_horizon += 1
    with pytest.raises(ValueError, match="shape signature"):
        fold_vla(model)
    model = _calibrated_model(36)
    model.vision.use_cls = True
    with pytest.raises(ValueError, match="vision runtime metadata"):
        fold_vla(model)
    model = _calibrated_model(33)
    model.backbone.blocks[0].attn.wo = torch.nn.Linear(4, 3)
    with pytest.raises(ValueError, match="shape signature"):
        fold_vla(model)
    model = _calibrated_model(26)
    model.backbone.blocks[0].ffn.down = torch.nn.Sequential(
        model.backbone.blocks[0].ffn.down, torch.nn.ReLU()
    )
    with pytest.raises((TypeError, ValueError), match="shape signature|child-module inventory"):
        fold_vla(model)


def test_strict_input_schema_rejects_short_instruction_and_bad_ids():
    model = _calibrated_model(27)
    valid = _batch(28)
    validate_strict_vla_inputs(model, valid)
    short = dict(valid)
    short["instr_ids"] = short["instr_ids"][:, :-1]
    with pytest.raises(ValueError, match="instr_ids.*shape"):
        validate_strict_vla_inputs(model, short)
    bad = dict(valid)
    bad["embodiment_id"] = torch.full_like(
        bad["embodiment_id"], model.cfg.n_embodiments
    )
    with pytest.raises(ValueError, match="embedding vocabulary"):
        validate_strict_vla_inputs(model, bad)


def test_uninitialized_source_norm_still_fails_closed():
    model = ChiVLA(_config()).double().eval()
    for norm in model.modules():
        if isinstance(norm, RmsBatchNorm):
            norm.frozen = True
    with pytest.raises(ValueError, match="calibrated, initialized, and frozen"):
        fold_vla(model)


@pytest.mark.parametrize(
    "override",
    [
        {"norm": "per_token", "qk_norm": "per_token"},
        {"qk_norm": "per_token"},
        {"action_head": "product"},
        {"dual_vision": True},
        {"vision_encoder": "conv"},
        {"n_phases": 2},
    ],
)
def test_unsupported_vla_configs_fail_closed(override):
    with pytest.raises(ValueError, match="strict VLA export requires"):
        fold_vla(ChiVLA(_config(**override)).double().eval())


def test_fold_is_a_snapshot_and_does_not_mutate_source_model():
    model = _calibrated_model(4)
    source_weight = model.action_head.weight.detach().clone()
    folded = fold_vla(model)
    torch.testing.assert_close(model.action_head.weight, source_weight)
    assert remaining_rbn_paths(model)
    with torch.no_grad():
        model.action_head.weight.add_(10.0)
    assert not torch.equal(folded.action_head.weight, model.action_head.weight)


def test_folded_state_dict_roundtrips_into_normalization_free_config():
    folded = fold_vla(_calibrated_model(5))
    restored = ChiVLA(copy.deepcopy(folded.cfg)).double().eval()
    restored.load_state_dict(folded.state_dict(), strict=True)
    batch = _batch(6)
    expected, _ = folded(**batch)
    actual, _ = restored(**batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_structured_export_restores_frozen_flags_and_folded_snapshot():
    source = _calibrated_model(7)
    package = make_strict_vla_export_package(
        source, source_checkpoint_sha256="a" * 64
    )
    restored = load_strict_vla_export_package(package)
    batch = _batch(8)
    expected, _ = fold_vla(source)(**batch)
    actual, _ = restored(**batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert package["decomposition_class"] == "normalization_free_strict_polynomial_program"
    assert package["source_checkpoint_sha256"] == "a" * 64


def test_structured_export_rejects_state_tampering():
    package = make_strict_vla_export_package(_calibrated_model(9))
    first = next(iter(package["source_state"]))
    package["source_state"][first].view(-1)[0] += 1.0
    with pytest.raises(ValueError, match="source state hash mismatch"):
        load_strict_vla_export_package(package)

    package = make_strict_vla_export_package(_calibrated_model(19))
    first = next(iter(package["folded_state"]))
    package["folded_state"][first].view(-1)[0] += 1.0
    with pytest.raises(ValueError, match="folded state hash mismatch"):
        load_strict_vla_export_package(package)


def test_structured_export_rejects_runtime_or_scale_metadata_tampering():
    package = make_strict_vla_export_package(_calibrated_model(12))
    runtime_path = next(iter(package["attention_runtime"]))
    package["attention_runtime"][runtime_path]["row_scale"] = "inv"
    with pytest.raises(ValueError, match="metadata hash mismatch"):
        load_strict_vla_export_package(package)

    package = make_strict_vla_export_package(_calibrated_model(12))
    runtime_path = next(iter(package["attention_runtime"]))
    package["attention_runtime"][runtime_path]["row_scale"] = "bogus"
    package["metadata_sha256"] = _metadata_sha256(package)
    with pytest.raises(ValueError, match="row_scale"):
        load_strict_vla_export_package(package)

    package = make_strict_vla_export_package(_calibrated_model(13))
    scale_path = next(iter(package["folded_scales"]))
    package["folded_scales"][scale_path] += 1.0
    package["metadata_sha256"] = _metadata_sha256(package)
    with pytest.raises(ValueError, match="folded scale metadata mismatch"):
        load_strict_vla_export_package(package)


def test_structured_export_rejects_malformed_checkpoint_hash():
    with pytest.raises(ValueError, match="64 hexadecimal"):
        make_strict_vla_export_package(
            _calibrated_model(14), source_checkpoint_sha256="not-a-hash"
        )


def test_structured_export_rejects_folded_config_or_implementation_tampering():
    package = make_strict_vla_export_package(_calibrated_model(20))
    package["folded_config"]["flow_steps"] += 1
    package["metadata_sha256"] = _metadata_sha256(package)
    with pytest.raises(ValueError, match="folded config metadata mismatch"):
        load_strict_vla_export_package(package)
    package = make_strict_vla_export_package(_calibrated_model(21))
    package["implementation_sha256"] = "0" * 64
    package["metadata_sha256"] = _metadata_sha256(package)
    with pytest.raises(ValueError, match="implementation hash mismatch"):
        load_strict_vla_export_package(package)


def test_structured_export_rejects_input_schema_removal_or_tampering():
    package = make_strict_vla_export_package(_calibrated_model(34))
    del package["input_schema"]
    with pytest.raises(ValueError, match="missing keys"):
        load_strict_vla_export_package(package)
    package = make_strict_vla_export_package(_calibrated_model(35))
    package["input_schema"]["token_count"] += 1
    with pytest.raises(ValueError, match="metadata hash mismatch"):
        load_strict_vla_export_package(package)


def test_structured_export_roundtrips_through_disk(tmp_path):
    source = _calibrated_model(22)
    package_path = tmp_path / "strict-vla.pt"
    torch.save(make_strict_vla_export_package(source), package_path)
    loaded = torch.load(package_path, weights_only=False)
    restored = load_strict_vla_export_package(loaded)
    batch = _batch(23)
    expected, _ = fold_vla(source)(**batch)
    actual, _ = restored(**batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_structured_export_roundtrips_supported_nondefault_runtime():
    source = _calibrated_model(37)
    source.backbone.blocks[0].attn.row_scale = "inv"
    source.norm_out.eps = 2e-5
    package = make_strict_vla_export_package(source)
    restored = load_strict_vla_export_package(package)
    batch = _batch(38)
    expected, _ = fold_vla(source)(**batch)
    actual, _ = restored(**batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
