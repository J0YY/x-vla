"""Exact tests for complete typed bilinear-attention compilation."""

from __future__ import annotations

import copy

import pytest
import torch

from xvla.nn.attention import BilinearAttention
from xvla.train.complete_attention_odt import (
    ATTENTION_ROLES,
    attention_route_features,
    assemble_attention_routes,
    compile_attention_head_cores,
    compile_bilinear_attention,
    evaluate_compiled_attention,
    pullback_route_metric,
)


DTYPE = torch.float64


def _module(
    seed: int,
    *,
    dimension: int = 4,
    heads: int = 2,
    causal: bool = False,
    row_scale: str = "invsqrt",
    score_scale: str = "d_h2",
) -> BilinearAttention:
    torch.manual_seed(seed)
    module = BilinearAttention(
        dimension,
        heads,
        causal=causal,
        row_scale=row_scale,
        score_scale=score_scale,
        qk_norm="none",
    ).double().eval()
    with torch.no_grad():
        for branch in (module.wq1, module.wk1, module.wq2, module.wk2, module.wv, module.wo):
            branch.bias.normal_()
    return module


def _split(linear: torch.nn.Linear, tokens: torch.Tensor, heads: int) -> torch.Tensor:
    batch, positions, dimension = tokens.shape
    head_dim = dimension // heads
    return linear(tokens).reshape(batch, positions, heads, head_dim).transpose(1, 2)


def _direct_cross(
    module: BilinearAttention,
    query: torch.Tensor,
    source: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    q1 = _split(module.wq1, query, module.n_heads)
    q2 = _split(module.wq2, query, module.n_heads)
    k1 = _split(module.wk1, source, module.n_heads)
    k2 = _split(module.wk2, source, module.n_heads)
    value = _split(module.wv, source, module.n_heads)
    pattern = (q1 @ k1.transpose(-1, -2)) * (q2 @ k2.transpose(-1, -2))
    pattern = pattern * mask / module._score_denom
    counts = mask.sum(-1).clamp_min(1)
    row = counts.rsqrt() if module.row_scale == "invsqrt" else counts.reciprocal()
    output = row[None, None, :, None] * (pattern @ value)
    output = output.transpose(1, 2).reshape(query.shape[0], query.shape[1], module.dim)
    return module.wo(output)


@pytest.mark.parametrize("row_scale", ["invsqrt", "inv"])
@pytest.mark.parametrize("score_scale", ["d_h", "d_h2"])
@pytest.mark.parametrize("causal", [False, True])
def test_complete_compiler_matches_self_attention(row_scale: str, score_scale: str, causal: bool):
    module = _module(
        1,
        dimension=6,
        heads=3,
        causal=causal,
        row_scale=row_scale,
        score_scale=score_scale,
    )
    tokens = torch.randn(3, 4, 6, dtype=DTYPE)
    compiled = compile_bilinear_attention(
        module,
        n_query=4,
        query_types=("vision", "language", "state", "action"),
        source_types=("vision", "language", "state", "action"),
    )
    expected = module(tokens, method="explicit")
    actual = evaluate_compiled_attention(compiled, tokens)
    torch.testing.assert_close(actual, expected, rtol=5e-12, atol=5e-12)


def test_complete_compiler_matches_typed_cross_attention_and_zero_row():
    module = _module(2, dimension=4, heads=2, row_scale="inv")
    query = torch.randn(2, 3, 4, dtype=DTYPE)
    source = torch.randn(2, 5, 4, dtype=DTYPE)
    mask = torch.tensor(
        [[1, 0, 1, 0, 1], [0, 0, 0, 0, 0], [0, 1, 1, 0, 0]], dtype=DTYPE
    )
    compiled = compile_bilinear_attention(
        module,
        n_query=3,
        n_source=5,
        mask=mask,
        query_types=("action",) * 3,
        source_types=("vision", "vision", "language", "state", "state"),
    )
    actual = evaluate_compiled_attention(compiled, query, source)
    expected = _direct_cross(module, query, source, mask)
    torch.testing.assert_close(actual, expected, rtol=5e-12, atol=5e-12)
    torch.testing.assert_close(actual[:, 1], module.wo.bias.expand_as(actual[:, 1]))


def test_head_core_matches_each_affine_branch_formula():
    module = _module(3, dimension=4, heads=2)
    cores = compile_attention_head_cores(module)
    query = torch.randn(7, 4, dtype=DTYPE)
    source = torch.randn(7, 4, dtype=DTYPE)
    qbar = torch.cat((torch.ones(7, 1, dtype=DTYPE), query), dim=1)
    sbar = torch.cat((torch.ones(7, 1, dtype=DTYPE), source), dim=1)
    q1 = module.wq1(query).reshape(7, 2, 2)
    q2 = module.wq2(query).reshape(7, 2, 2)
    k1 = module.wk1(source).reshape(7, 2, 2)
    k2 = module.wk2(source).reshape(7, 2, 2)
    value = module.wv(source).reshape(7, 2, 2)
    for head in range(2):
        actual = torch.einsum(
            "cpqrst,bp,bq,br,bs,bt->bc",
            cores[head], qbar, qbar, sbar, sbar, sbar,
        )
        expected = (
            (q1[:, head] * k1[:, head]).sum(-1)
            * (q2[:, head] * k2[:, head]).sum(-1)
        )[:, None] * value[:, head]
        torch.testing.assert_close(actual, expected, rtol=5e-12, atol=5e-12)


def test_factor_interface_order_matches_named_dense_axes():
    module = _module(31, dimension=4, heads=2)
    compiled = compile_bilinear_attention(module, n_query=1)
    assert ATTENTION_ROLES == ("query_1", "query_2", "key_1", "key_2", "value")
    branches = (module.wq1, module.wq2, module.wk1, module.wk2, module.wv)
    token = torch.randn(5, 4, dtype=DTYPE)
    homogeneous = torch.cat((torch.ones(5, 1, dtype=DTYPE), token), dim=1)
    for factor, branch in zip(compiled.head_factors, branches):
        reconstructed = torch.einsum("bd,hpd->bhp", homogeneous, factor).reshape(5, 4)
        torch.testing.assert_close(reconstructed, branch(token), rtol=5e-12, atol=5e-12)


def test_dense_route_metric_preserves_cross_head_cancellation():
    module = _module(4, dimension=2, heads=2)
    with torch.no_grad():
        for branch in (module.wq1, module.wk1, module.wq2, module.wk2, module.wv):
            branch.weight[1].copy_(branch.weight[0])
            branch.bias[1].copy_(branch.bias[0])
        module.wo.weight.zero_()
        module.wo.weight[0] = torch.tensor([1.0, -1.0], dtype=DTYPE)
        module.wo.bias.zero_()
    compiled = compile_bilinear_attention(module, n_query=1)
    output_metric = torch.eye(2, dtype=DTYPE)
    route_metric = pullback_route_metric(compiled, output_metric)
    features = torch.tensor([1.0, 1.0], dtype=DTYPE)
    full_energy = features @ route_metric[1:, 1:] @ features
    diagonal_surrogate = features @ torch.diag(torch.diag(route_metric[1:, 1:])) @ features
    torch.testing.assert_close(full_energy, torch.zeros((), dtype=DTYPE), atol=1e-12, rtol=0)
    assert float(diagonal_surrogate.item()) > 1.0
    assert float(route_metric[1, 2].item()) < 0


def test_route_metric_energy_identity_with_coupled_queries():
    module = _module(5, dimension=4, heads=2)
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=DTYPE)
    compiled = compile_bilinear_attention(module, n_query=2, n_source=3, mask=mask)
    raw = torch.randn(8, 8, dtype=DTYPE)
    output_metric = raw @ raw.T
    route_metric = pullback_route_metric(compiled, output_metric)
    route_vector = torch.randn(compiled.output_assembly.shape[1], dtype=DTYPE)
    output = compiled.output_assembly @ route_vector
    torch.testing.assert_close(
        route_vector @ route_metric @ route_vector,
        output @ output_metric @ output,
        rtol=5e-12,
        atol=5e-12,
    )


def test_scalar_rbn_requires_frozen_initialized_branches_and_then_compiles():
    module = BilinearAttention(4, 2, qk_norm="scalar_rbn").double()
    with pytest.raises(ValueError, match="calibrated"):
        compile_bilinear_attention(module, n_query=2)
    module.train()
    module(torch.randn(3, 2, 4, dtype=DTYPE))
    with torch.no_grad():
        for branch in (module.wq1, module.wk1, module.wq2, module.wk2, module.wv, module.wo):
            branch.bias.normal_()
    module.eval()
    for norm in (module.rbn_q1, module.rbn_k1, module.rbn_q2, module.rbn_k2, module.rbn_v):
        norm.freeze()
    tokens = torch.randn(2, 2, 4, dtype=DTYPE)
    compiled = compile_bilinear_attention(module, n_query=2)
    torch.testing.assert_close(
        evaluate_compiled_attention(compiled, tokens),
        module(tokens),
        rtol=5e-12,
        atol=5e-12,
    )


def test_factorized_real_width_compile_avoids_dense_petabyte_core_and_assembly():
    module = _module(8, dimension=384, heads=12)
    compiled = compile_bilinear_attention(
        module,
        n_query=1,
        dense_oracle_maximum_elements=1000,
    )
    assert compiled.head_cores is None
    assert compiled.output_assembly is None
    assert compiled.head_factors[0].shape == (12, 32, 385)
    tokens = torch.randn(2, 1, 384, dtype=DTYPE)
    torch.testing.assert_close(
        evaluate_compiled_attention(compiled, tokens),
        module(tokens),
        rtol=5e-11,
        atol=5e-11,
    )
    with pytest.raises(ValueError, match="dense attention core"):
        compile_attention_head_cores(module)


def test_masked_khatri_rao_updates_stateful_norm_statistics_once():
    explicit = BilinearAttention(4, 2, qk_norm="scalar_rbn").double().train()
    khatri = copy.deepcopy(explicit)
    warmup = torch.randn(3, 3, 4, dtype=DTYPE) * 4.0
    explicit(warmup, method="explicit")
    khatri(warmup, method="explicit")
    tokens = torch.randn(3, 3, 4, dtype=DTYPE)
    mask = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=DTYPE)
    expected = explicit(tokens, mask=mask, method="explicit")
    actual = khatri(tokens, mask=mask, method="khatri_rao")
    torch.testing.assert_close(actual, expected, rtol=5e-12, atol=5e-12)
    for name in ("rbn_q1", "rbn_k1", "rbn_q2", "rbn_k2", "rbn_v"):
        torch.testing.assert_close(
            getattr(khatri, name).running_rms,
            getattr(explicit, name).running_rms,
        )


def test_compiled_parameters_and_mask_are_immutable_snapshots():
    module = _module(9, dimension=4, heads=2)
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=DTYPE)
    tokens = torch.randn(2, 2, 4, dtype=DTYPE)
    compiled = compile_bilinear_attention(module, n_query=2, mask=mask)
    expected = evaluate_compiled_attention(compiled, tokens)
    with torch.no_grad():
        module.wo.weight.add_(10.0)
        module.wo.bias.sub_(7.0)
        mask.zero_()
    actual = evaluate_compiled_attention(compiled, tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        compiled.mask,
        torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=DTYPE),
    )


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
def test_invalid_dense_oracle_limits_are_rejected(limit):
    with pytest.raises(ValueError, match="dense_oracle_maximum_elements"):
        compile_bilinear_attention(_module(10), n_query=1, dense_oracle_maximum_elements=limit)


def test_actual_cross_source_features_require_off_diagonal_route_terms():
    module = BilinearAttention(1, 1, qk_norm="none", row_scale="inv").double().eval()
    with torch.no_grad():
        for branch in (module.wq1, module.wk1, module.wq2, module.wk2):
            branch.weight.zero_()
            branch.bias.fill_(1.0)
        module.wv.weight.fill_(1.0)
        module.wv.bias.zero_()
        module.wo.weight.fill_(1.0)
        module.wo.bias.zero_()
    compiled = compile_bilinear_attention(
        module,
        n_query=1,
        n_source=2,
        mask=torch.ones(1, 2, dtype=DTYPE),
    )
    query = torch.zeros(1, 1, 1, dtype=DTYPE)
    source = torch.tensor([[[1.0], [-1.0]]], dtype=DTYPE)
    features = attention_route_features(compiled, query, source)
    output = assemble_attention_routes(compiled, features)
    torch.testing.assert_close(output, torch.zeros_like(output), atol=1e-12, rtol=0)
    metric = pullback_route_metric(compiled, torch.ones(1, 1, dtype=DTYPE))[1:, 1:]
    feature_vector = features[0]
    full_energy = feature_vector @ metric @ feature_vector
    diagonal_energy = feature_vector @ torch.diag(torch.diag(metric)) @ feature_vector
    torch.testing.assert_close(full_energy, torch.zeros((), dtype=DTYPE), atol=1e-12, rtol=0)
    assert float(diagonal_energy.item()) > 0


@pytest.mark.parametrize(
    "mask",
    [torch.ones(2, 3), torch.tensor([[1.0, float("nan")], [0.0, 1.0]]), torch.tensor([[1.0, 0.5], [0.0, 1.0]])],
)
def test_malformed_masks_are_rejected(mask: torch.Tensor):
    module = _module(6, dimension=4, heads=2)
    tokens = torch.randn(1, 2, 4, dtype=DTYPE)
    with pytest.raises(ValueError, match="mask"):
        module(tokens, mask=mask, method="explicit")
    with pytest.raises(ValueError, match="mask"):
        compile_bilinear_attention(module, n_query=2, mask=mask)


def test_causal_scan_rejects_noncausal_module():
    module = _module(7, dimension=4, heads=2, causal=False)
    with pytest.raises(ValueError, match="causal=True"):
        module(torch.randn(1, 3, 4, dtype=DTYPE), method="causal_scan")


@pytest.mark.parametrize("qk_norm", ["per_token", "homotopy", "rational"])
def test_nonpolynomial_or_unfolded_branch_norms_fail_closed(qk_norm: str):
    module = BilinearAttention(4, 2, qk_norm=qk_norm).double().eval()
    with pytest.raises(ValueError, match="not a strict polynomial"):
        compile_bilinear_attention(module, n_query=2)
