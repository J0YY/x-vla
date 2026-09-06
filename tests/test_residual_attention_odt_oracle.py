"""Regression tests for the whole-state residual-attention ODT oracle."""

from __future__ import annotations

import pytest
import torch

from xvla.models.vla import ChiVLA, VLAConfig
from xvla.nn.attention import causal_mask
from xvla.nn.block import ChiTransformerBlock
from xvla.nn.normalization import RationalNorm, RmsBatchNorm
from xvla.train.residual_attention_odt_oracle import (
    POLYNOMIAL_CLAIM_BOUNDARY,
    RATIONAL_INTERFACE_GAP,
    canonical_environments_mixed_arity,
    canonicalize_mixed_arity,
    compile_tiny_vla_residual_oracle,
    compile_tiny_vla_whole_workspace_oracle,
    gauge_bond,
    mixed_arity_forward,
    observation_to_polynomial_input,
    oracle_actions,
    one_leg_gauge_negative_control,
    replay_rational_residual_block,
)


DTYPE = torch.float64


def _model(seed: int = 0, **overrides) -> ChiVLA:
    torch.manual_seed(seed)
    values = dict(
        image_size=1,
        patch_size=1,
        vit_dim=1,
        vit_layers=0,
        vit_heads=1,
        vocab_size=2,
        max_instr_len=1,
        state_dim=1,
        n_embodiments=2,
        dim=1,
        n_layers=1,
        n_heads=1,
        ffn_rank=3,
        attn="bilinear",
        ffn="bilinear",
        norm="none",
        qk_norm="none",
        residual=True,
        action_horizon=1,
        action_dim=1,
        action_head="linear",
    )
    values.update(overrides)
    model = ChiVLA(VLAConfig(**values)).double().eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.normal_(std=0.2)
        model.vision.patch.bias.normal_(std=0.2)
    return model


def _batch(model: ChiVLA, seed: int = 0, batch: int = 4):
    generator = torch.Generator().manual_seed(seed)
    img = torch.randn(
        batch,
        3,
        model.cfg.image_size,
        model.cfg.image_size,
        generator=generator,
        dtype=DTYPE,
    )
    instr = torch.randint(0, model.cfg.vocab_size, (batch, 1), generator=generator)
    state = torch.randn(batch, 1, generator=generator, dtype=DTYPE)
    embodiment = torch.randint(0, model.cfg.n_embodiments, (batch,), generator=generator)
    return img, instr, state, embodiment


@pytest.mark.parametrize("seed", range(4))
def test_full_observation_to_action_replay_preserves_vision_and_both_residuals(seed: int):
    model = _model(seed)
    oracle = compile_tiny_vla_residual_oracle(model)
    batch = _batch(model, seed + 10)
    raw = observation_to_polynomial_input(model, *batch)
    direct = model(*batch)[0]
    torch.testing.assert_close(oracle_actions(oracle, raw), direct, rtol=2e-11, atol=2e-11)
    assert oracle.network.arities == (5, 2)
    image_columns = oracle.network.embedding[1:, 1:4]
    assert float(image_columns.norm().item()) > 0.0


@pytest.mark.parametrize("seed", range(4))
def test_canonical_replay_isometric_rows_and_top_down_grams(seed: int):
    model = _model(seed)
    oracle = compile_tiny_vla_residual_oracle(model)
    raw = observation_to_polynomial_input(model, *_batch(model, seed + 20))
    canonical = canonicalize_mixed_arity(oracle.network)
    torch.testing.assert_close(
        mixed_arity_forward(canonical.network, raw),
        mixed_arity_forward(oracle.network, raw),
        rtol=2e-10,
        atol=2e-10,
    )
    assert max(canonical.isometry_errors) < 2e-10
    assert max(canonical.factorization_errors) < 2e-10
    assert max(canonical.symmetry_errors) < 2e-12
    grams = canonical_environments_mixed_arity(canonical.network)
    assert len(grams) == 3
    assert all(torch.isfinite(gram).all() for gram in grams)


@pytest.mark.parametrize("bond", [0, 1, 2])
def test_nonorthogonal_gauge_preserves_function_and_canonical_gram_spectra(bond: int):
    model = _model(31 + bond)
    oracle = compile_tiny_vla_residual_oracle(model)
    raw = observation_to_polynomial_input(model, *_batch(model, 41 + bond))
    dimension = oracle.network.bond_dims[bond]
    generator = torch.Generator().manual_seed(51 + bond)
    random = torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
    orthogonal, _ = torch.linalg.qr(random)
    transform = orthogonal @ torch.diag(torch.linspace(0.6, 1.8, dimension, dtype=DTYPE)) @ orthogonal.T
    gauged = gauge_bond(oracle.network, bond, transform)
    torch.testing.assert_close(
        mixed_arity_forward(gauged, raw),
        mixed_arity_forward(oracle.network, raw),
        rtol=2e-10,
        atol=2e-10,
    )
    base = canonicalize_mixed_arity(oracle.network).network
    candidate = canonicalize_mixed_arity(gauged).network
    torch.testing.assert_close(
        mixed_arity_forward(candidate, raw),
        mixed_arity_forward(base, raw),
        rtol=2e-10,
        atol=2e-10,
    )
    base_grams = canonical_environments_mixed_arity(base)
    candidate_grams = canonical_environments_mixed_arity(candidate)
    for left, right in zip(base_grams, candidate_grams):
        torch.testing.assert_close(
            torch.linalg.eigvalsh(left),
            torch.linalg.eigvalsh(right),
            rtol=2e-8,
            atol=2e-10,
        )


def test_two_block_composition_replay_canonicalizes_and_survives_every_bond_gauge():
    model = _model(60, n_layers=2)
    oracle = compile_tiny_vla_residual_oracle(model)
    batch = _batch(model, 61, batch=3)
    raw = observation_to_polynomial_input(model, *batch)
    direct = model(*batch)[0]
    assert oracle.network.arities == (5, 2, 5, 2)
    torch.testing.assert_close(oracle_actions(oracle, raw), direct, rtol=3e-10, atol=3e-10)
    canonical = canonicalize_mixed_arity(oracle.network)
    canonical_output = mixed_arity_forward(canonical.network, raw).reshape_as(direct)
    torch.testing.assert_close(canonical_output, direct, rtol=3e-9, atol=3e-10)
    reference_grams = canonical_environments_mixed_arity(canonical.network)
    assert len(reference_grams) == 5

    for bond, dimension in enumerate(oracle.network.bond_dims):
        generator = torch.Generator().manual_seed(700 + bond)
        random = torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
        orthogonal, _ = torch.linalg.qr(random)
        transform = orthogonal @ torch.diag(
            torch.linspace(0.65, 1.75, dimension, dtype=DTYPE)
        ) @ orthogonal.T
        gauged = gauge_bond(oracle.network, bond, transform)
        torch.testing.assert_close(
            mixed_arity_forward(gauged, raw).reshape_as(direct),
            direct,
            rtol=3e-9,
            atol=3e-10,
        )
        gauged_canonical = canonicalize_mixed_arity(gauged).network
        candidate_grams = canonical_environments_mixed_arity(gauged_canonical)
        for reference, candidate in zip(reference_grams, candidate_grams):
            reference_spectrum = torch.linalg.eigvalsh(reference)
            candidate_spectrum = torch.linalg.eigvalsh(candidate)
            spectrum_relative = (reference_spectrum - candidate_spectrum).norm() / (
                reference_spectrum.norm().clamp_min(torch.finfo(DTYPE).tiny)
            )
            assert float(spectrum_relative.item()) < 3e-7


def test_last_binary_cut_gram_matches_explicit_contraction():
    canonical = canonicalize_mixed_arity(
        compile_tiny_vla_residual_oracle(_model(62, n_layers=2)).network
    ).network
    grams = canonical_environments_mixed_arity(canonical)
    core = canonical.cores[-1]
    downstream = canonical.head.T @ canonical.head
    transformed = torch.einsum("op,pAB->oAB", downstream, core)
    left = torch.einsum("oab,oAb->aA", core, transformed)
    right = torch.einsum("oab,oaB->bB", core, transformed)
    explicit = 0.25 * (left + right + left.T + right.T)
    relative = (grams[-2] - explicit).norm() / explicit.norm().clamp_min(torch.finfo(DTYPE).tiny)
    assert float(relative.item()) < 1e-8


def test_one_leg_factor_push_is_a_detectable_negative_control():
    model = _model(63, n_layers=2)
    oracle = compile_tiny_vla_residual_oracle(model)
    raw = observation_to_polynomial_input(model, *_batch(model, 64, batch=6))
    dimension = oracle.network.bond_dims[0]
    transform = torch.diag(torch.linspace(0.5, 1.9, dimension, dtype=DTYPE))
    incorrect = one_leg_gauge_negative_control(oracle.network, 0, transform, role=0)
    relative = (
        mixed_arity_forward(incorrect, raw) - mixed_arity_forward(oracle.network, raw)
    ).norm() / mixed_arity_forward(oracle.network, raw).norm().clamp_min(torch.finfo(DTYPE).tiny)
    assert float(relative.item()) > 1e-8
    with pytest.raises(ValueError, match="not permutation-symmetric"):
        canonicalize_mixed_arity(incorrect)


def test_whole_workspace_one_vit_one_joint_block_replays_full_model():
    model = _model(65, vit_layers=1, n_layers=1)
    oracle = compile_tiny_vla_whole_workspace_oracle(model)
    batch = _batch(model, 66, batch=5)
    raw = observation_to_polynomial_input(model, *batch)
    direct = model(*batch)[0]
    assert oracle.network.arities == (5, 2, 1, 5, 2)
    torch.testing.assert_close(oracle_actions(oracle, raw), direct, rtol=3e-10, atol=3e-10)
    assert float(oracle.network.embedding[1:, 1:4].norm().item()) > 0.0
    assert oracle.polynomial_model.vision.blocks.blocks[0].residual
    assert oracle.polynomial_model.backbone.blocks[0].residual


def test_whole_workspace_multitoken_vision_attention_replay_is_not_degenerate():
    model = _model(651, image_size=2, patch_size=1, vit_layers=1, n_layers=1)
    oracle = compile_tiny_vla_whole_workspace_oracle(model)
    batch = _batch(model, 652, batch=2)
    raw = observation_to_polynomial_input(model, *batch)
    direct = model(*batch)[0]
    assert model.cfg.vit_config().num_patches == 4
    assert oracle.token_count == 9
    torch.testing.assert_close(oracle_actions(oracle, raw), direct, rtol=2e-9, atol=2e-9)
    canonical = canonicalize_mixed_arity(oracle.network)
    torch.testing.assert_close(
        mixed_arity_forward(canonical.network, raw).reshape_as(direct),
        direct,
        rtol=2e-8,
        atol=2e-8,
    )


def test_whole_workspace_canonical_replay_and_every_bond_gauge():
    model = _model(67, vit_layers=1, n_layers=1)
    oracle = compile_tiny_vla_whole_workspace_oracle(model)
    raw = observation_to_polynomial_input(model, *_batch(model, 68, batch=3))
    reference = mixed_arity_forward(oracle.network, raw)
    canonical = canonicalize_mixed_arity(oracle.network)
    torch.testing.assert_close(
        mixed_arity_forward(canonical.network, raw), reference, rtol=2e-8, atol=2e-9
    )
    reference_grams = canonical_environments_mixed_arity(canonical.network)
    assert len(reference_grams) == 6
    for bond, dimension in enumerate(oracle.network.bond_dims):
        generator = torch.Generator().manual_seed(900 + bond)
        random = torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
        orthogonal, _ = torch.linalg.qr(random)
        transform = orthogonal @ torch.diag(
            torch.linspace(0.7, 1.6, dimension, dtype=DTYPE)
        ) @ orthogonal.T
        gauged = gauge_bond(oracle.network, bond, transform)
        torch.testing.assert_close(
            mixed_arity_forward(gauged, raw), reference, rtol=2e-8, atol=2e-9
        )
        candidate = canonicalize_mixed_arity(gauged).network
        candidate_grams = canonical_environments_mixed_arity(candidate)
        for left, right in zip(reference_grams, candidate_grams):
            left_spectrum = torch.linalg.eigvalsh(left)
            right_spectrum = torch.linalg.eigvalsh(right)
            relative = (left_spectrum - right_spectrum).norm() / left_spectrum.norm().clamp_min(
                torch.finfo(DTYPE).tiny
            )
            assert float(relative.item()) < 2e-6


def test_whole_workspace_frozen_scalar_norm_fold_is_exact_and_nonmutating():
    model = _model(
        69,
        vit_layers=1,
        n_layers=1,
        norm="scalar_rbn",
        qk_norm="scalar_rbn",
    )
    with torch.no_grad():
        for index, norm in enumerate(
            module for module in model.modules() if isinstance(module, RmsBatchNorm)
        ):
            norm.running_rms.fill_(0.9 + 0.04 * index)
            norm.initialized.fill_(True)
            norm.freeze()
    source_norm_count = sum(isinstance(module, RmsBatchNorm) for module in model.modules())
    oracle = compile_tiny_vla_whole_workspace_oracle(model)
    batch = _batch(model, 70)
    raw = observation_to_polynomial_input(model, *batch)
    torch.testing.assert_close(oracle_actions(oracle, raw), model(*batch)[0], rtol=2e-9, atol=2e-9)
    assert sum(isinstance(module, RmsBatchNorm) for module in model.modules()) == source_norm_count
    assert not any(isinstance(module, RmsBatchNorm) for module in oracle.polynomial_model.modules())


def test_polynomial_export_fails_closed_on_rational_checkpoint_without_changing_it():
    model = _model(70, norm="rational", qk_norm="rational")
    original_types = tuple(type(module) for module in model.modules() if isinstance(module, RationalNorm))
    with pytest.raises(ValueError, match="pair-valued multilinear cores"):
        compile_tiny_vla_residual_oracle(model)
    assert tuple(type(module) for module in model.modules() if isinstance(module, RationalNorm)) == original_types
    assert "not a canonical decomposition" in POLYNOMIAL_CLAIM_BOUNDARY
    assert "pair-valued multilinear cores" in RATIONAL_INTERFACE_GAP


def test_frozen_scalar_rbn_is_folded_exactly_without_mutating_source_model():
    model = _model(75, norm="scalar_rbn", qk_norm="scalar_rbn")
    with torch.no_grad():
        for index, norm in enumerate(
            module for module in model.modules() if isinstance(module, RmsBatchNorm)
        ):
            norm.running_rms.fill_(0.8 + 0.07 * index)
            norm.initialized.fill_(True)
            norm.freeze()
    original_paths = tuple(name for name, module in model.named_modules() if isinstance(module, RmsBatchNorm))
    oracle = compile_tiny_vla_residual_oracle(model)
    batch = _batch(model, 76)
    raw = observation_to_polynomial_input(model, *batch)
    direct = model(*batch)[0]
    torch.testing.assert_close(oracle_actions(oracle, raw), direct, rtol=5e-11, atol=5e-11)
    assert oracle.source_normalization == "scalar_rbn"
    assert tuple(name for name, module in model.named_modules() if isinstance(module, RmsBatchNorm)) == original_paths
    assert not any(isinstance(module, RmsBatchNorm) for module in oracle.polynomial_model.modules())


def test_staged_projective_replay_keeps_rational_norms_and_residuals_exact():
    torch.manual_seed(80)
    block = ChiTransformerBlock(
        dim=2,
        n_heads=1,
        n_layers=1,
        causal=True,
        norm="rational",
        qk_norm="rational",
        residual=True,
    ).double().eval()
    with torch.no_grad():
        for norm in (module for module in block.modules() if isinstance(module, RationalNorm)):
            norm.running_ms.fill_(1.3)
            norm.initialized.fill_(True)
    tokens = torch.randn(2, 2, 2, dtype=DTYPE)
    mask = causal_mask(2, dtype=DTYPE)
    direct = block(tokens, mask=mask, method="explicit")
    replay = replay_rational_residual_block(block, tokens, mask, maximum_routes=8)
    torch.testing.assert_close(replay.value, direct, rtol=5e-10, atol=5e-10)
    assert replay.denominator_ledger
    assert replay.object_kind.endswith("noncanonical")
    assert isinstance(block.rbn_attn, RationalNorm)
    assert isinstance(block.rbn_ffn, RationalNorm)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"vit_layers": 1}, "vit_layers"),
        ({"n_layers": 0}, "n_layers"),
        ({"residual": False}, "residual"),
        ({"norm": "per_token", "qk_norm": "per_token"}, "identity or frozen scalar"),
    ],
)
def test_tiny_scope_rejections_are_explicit(override, message):
    with pytest.raises(ValueError, match=message):
        compile_tiny_vla_residual_oracle(_model(90, **override))
