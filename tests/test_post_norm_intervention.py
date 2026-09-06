"""Independent algebra and contract tests for Stage 7 endpoint interventions."""

from __future__ import annotations

import pytest
import torch

from xvla.models.vla import ChiVLA, VLAConfig
from xvla.train.post_norm_intervention import (
    CLAIM_BOUNDARY,
    HorizonLowRankProjector,
    PostNormAttenuationPolicy,
    attenuate_queries,
    match_normalized_action_doses,
    normalized_action_delta,
    normalized_action_dose,
)
from xvla.train.vla_terminal_balanced_odt import compile_linear_action_endpoint


DTYPE = torch.float64


def _model() -> ChiVLA:
    torch.manual_seed(13)
    return ChiVLA(
        VLAConfig(
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
            ffn_rank=6,
            vit_ffn_rank=6,
            norm="none",
            qk_norm="none",
            action_horizon=3,
            action_dim=7,
            action_head="linear",
        )
    ).double().eval()


def _orthogonal_operators() -> dict[str, HorizonLowRankProjector]:
    bases = {
        "balanced_top": torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=DTYPE),
        "reachable_only": torch.tensor([[0.0], [1.0], [0.0], [0.0]], dtype=DTYPE),
        "observable_only": torch.tensor([[0.0], [0.0], [1.0], [0.0]], dtype=DTYPE),
    }
    return {
        name: HorizonLowRankProjector(
            vector.expand(3, -1, -1).clone(),
            vector.T.expand(3, -1, -1).clone(),
            name,
        )
        for name, vector in bases.items()
    }


def _batch(seed: int = 17, count: int = 32) -> torch.Tensor:
    return torch.randn(count, 3, 4, generator=torch.Generator().manual_seed(seed), dtype=DTYPE)


def test_low_rank_and_independent_dense_oracles_match_exactly() -> None:
    states = _batch()
    means = states.mean(0)
    for operator in _orthogonal_operators().values():
        alpha = 0.37
        actual = attenuate_queries(states, means, operator, alpha=alpha)
        centered = states - means
        dense = operator.dense()
        expected = states - alpha * torch.einsum("hde,bhe->bhd", dense, centered)
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(
            dense @ dense,
            dense,
            atol=1e-14,
            rtol=1e-14,
        )


def test_oblique_rank_one_operator_and_dual_are_supported() -> None:
    state = torch.tensor([[[2.0], [1.0], [0.0], [0.0]]], dtype=DTYPE).expand(3, -1, -1).clone()
    analysis = torch.tensor([[[0.5, 0.0, 0.3, 0.0]]], dtype=DTYPE).expand(3, -1, -1).clone()
    operator = HorizonLowRankProjector(state, analysis, "oblique")
    dense = operator.dense()
    torch.testing.assert_close(dense @ dense, dense, atol=1e-14, rtol=1e-14)
    assert not torch.allclose(dense, dense.transpose(-1, -2))


def test_action_delta_matches_direct_affine_endpoint_and_zero_is_identity() -> None:
    model = _model()
    endpoint = compile_linear_action_endpoint(model, torch.zeros(7, dtype=DTYPE), torch.ones(7, dtype=DTYPE))
    states = _batch()
    means = states.mean(0)
    operator = _orthogonal_operators()["balanced_top"]
    edited = attenuate_queries(states, means, operator, alpha=0.61)
    actual = endpoint.normalized(edited) - endpoint.normalized(states)
    predicted = normalized_action_delta(endpoint, states, means, operator, alpha=0.61)
    torch.testing.assert_close(predicted, actual, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(
        attenuate_queries(states, means, operator, alpha=0.0), states, atol=0, rtol=0
    )


def test_dose_matching_is_outcome_free_and_exact_for_prespecified_group() -> None:
    model = _model()
    endpoint = compile_linear_action_endpoint(model, torch.zeros(7, dtype=DTYPE), torch.ones(7, dtype=DTYPE))
    states = _batch(count=80)
    means = states.mean(0)
    operators = _orthogonal_operators()
    matched = match_normalized_action_doses(
        endpoint,
        states,
        means,
        operators,
        primary="balanced_top",
        alpha_grid=(1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01),
        action_indices=(0, 1, 2, 3, 4, 5, 6),
    )
    assert all(0 < value <= 1 for value in matched.alphas.values())
    assert max(abs(value - matched.target_rms) for value in matched.realized_rms.values()) <= 1e-12
    for name, operator in operators.items():
        expected = normalized_action_dose(
            endpoint,
            states,
            means,
            operator,
            alpha=matched.alphas[name],
            action_indices=matched.action_indices,
        )
        assert expected == matched.realized_rms[name]


def test_policy_wrapper_matches_manual_edit_and_preserves_model_state() -> None:
    model = _model()
    batch = {
        "img": torch.randn(2, 3, 8, 8, dtype=DTYPE),
        "instr_ids": torch.randint(0, 16, (2, 3)),
        "state": torch.randn(2, 2, dtype=DTYPE),
        "embodiment_id": torch.zeros(2, dtype=torch.long),
    }
    with torch.no_grad():
        _, queries = model._encode(**batch)
    means = _batch().mean(0)
    operator = _orthogonal_operators()["balanced_top"]
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    wrapper = PostNormAttenuationPolicy(model, means, operator, alpha=0.4)
    actual, loss = wrapper(**batch)
    expected = model.action_head(attenuate_queries(queries, means, operator, alpha=0.4))
    assert loss is None
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    wrapper.train()
    with pytest.raises(RuntimeError, match="remain in eval mode"):
        wrapper(**batch)


def test_validation_fails_closed() -> None:
    state = torch.eye(4, dtype=DTYPE)[:, :1].expand(3, -1, -1).clone()
    with pytest.raises(ValueError, match="YX identity"):
        HorizonLowRankProjector(state, 2 * state.transpose(-1, -2), "bad")
    operator = HorizonLowRankProjector(state, state.transpose(-1, -2), "good")
    states = _batch()
    means = states.mean(0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        attenuate_queries(states, means, operator, alpha=1.1)
    with pytest.raises(ValueError, match="mapping keys"):
        model = _model()
        endpoint = compile_linear_action_endpoint(model, torch.zeros(7, dtype=DTYPE), torch.ones(7, dtype=DTYPE))
        match_normalized_action_doses(
            endpoint,
            states,
            means,
            {"different": operator, "good": operator},
            primary="good",
            alpha_grid=(1.0,),
            action_indices=(0,),
        )
    training_model = _model().train()
    with pytest.raises(ValueError, match="inference-only"):
        PostNormAttenuationPolicy(training_model, means, operator, alpha=0.5)
    assert "not establish semantic concepts" in CLAIM_BOUNDARY
