"""Exact and adversarial tests for separately named PQBD."""

from __future__ import annotations

import inspect

import pytest
import torch

from xvla.nn.attention import BilinearAttention
from xvla.nn.bilinear import BilinearFFN
from xvla.nn.normalization import RationalNorm
from xvla.train import projective_quotient as pq


DTYPE = torch.float64


def _norm(*, degree: int = 2, seed: int = 0) -> RationalNorm:
    torch.manual_seed(seed)
    coefficients = {}
    if degree != 2:
        coefficients = {
            "pade_numerator": tuple(1.0 / (index + 1) for index in range(degree + 1)),
            "pade_denominator": (1.0,) + (0.125,) * degree,
        }
    norm = RationalNorm(variant="pade", deg=degree, **coefficients).double().eval()
    with torch.no_grad():
        norm.running_ms.fill_(1.7)
        norm.initialized.fill_(True)
    return norm


def _rational_attention(seed: int = 0, *, causal: bool = False) -> BilinearAttention:
    torch.manual_seed(seed)
    module = BilinearAttention(
        dim=4,
        n_heads=2,
        causal=causal,
        qk_norm="rational",
        row_scale="invsqrt",
        score_scale="d_h2",
    ).double().eval()
    with torch.no_grad():
        for norm in (module.rn_q1, module.rn_k1, module.rn_q2, module.rn_k2):
            norm.running_ms.fill_(1.3)
            norm.initialized.fill_(True)
        for branch in (module.wq1, module.wk1, module.wq2, module.wk2, module.wv, module.wo):
            branch.bias.normal_()
    return module


def _quotient_fixture(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    numerator = torch.randn(3, generator=generator, dtype=DTYPE)
    denominator = torch.tensor(1.4, dtype=DTYPE)
    numerator_jacobian = torch.randn(3, 4, generator=generator, dtype=DTYPE)
    denominator_gradient = torch.randn(4, generator=generator, dtype=DTYPE)
    raw = torch.randn(3, 3, generator=generator, dtype=DTYPE)
    output_metric = raw @ raw.T
    return numerator, denominator, numerator_jacobian, denominator_gradient, output_metric


def test_name_and_claim_boundary_are_explicitly_noncanonical():
    assert pq.PQBD_OBJECT_KIND == "projective_quotient_balanced_decomposition_noncanonical"
    assert "not canonical ODT" in pq.PQBD_CLAIM_BOUNDARY
    assert "not rational ODT" in pq.PQBD_CLAIM_BOUNDARY
    source = inspect.getsource(pq)
    assert "from xvla.train.canonical_odt" not in source
    assert "from xvla.train.generalized_metric_odt" not in source


def test_projective_primitives_replay_exact_values_and_conservative_degrees():
    x = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=DTYPE)
    left = pq.ProjectivePair(2 * x, torch.full((2, 1), 3.0, dtype=DTYPE), 3, 2)
    right = pq.ProjectivePair(-x, torch.full((2, 1), 5.0, dtype=DTYPE), 4, 1)
    added = pq.projective_add(left, right)
    product = pq.projective_product(left, right)
    torch.testing.assert_close(pq.projective_value(added), 2 * x / 3 - x / 5)
    torch.testing.assert_close(pq.projective_value(product), (2 * x / 3) * (-x / 5))
    assert added.numerator_degree_upper_bound == 6
    assert added.denominator_degree_upper_bound == 3
    assert product.numerator_degree_upper_bound == 7
    assert product.denominator_degree_upper_bound == 3


def test_projective_constant_rescaling_is_value_invariant_and_logged():
    pair = pq.ProjectivePair(
        torch.tensor([[2.0, -1.0]], dtype=DTYPE),
        torch.tensor([[4.0]], dtype=DTYPE),
        5,
        4,
    )
    scaled = pq.projective_rescale(pair, -7.0, label="negative_constant_gauge")
    torch.testing.assert_close(pq.projective_value(pair), pq.projective_value(scaled))
    assert scaled.denominator_ledger[-1].label == "negative_constant_gauge"
    assert not scaled.denominator_ledger[-1].observed_all_positive
    with pytest.raises(ValueError, match="preserve numerator and denominator shapes"):
        pq.projective_rescale(pair, torch.tensor([2.0, 3.0], dtype=DTYPE))


def test_denominator_ledger_is_observed_only_and_fail_closed():
    denominator = torch.tensor([[-2.0], [3.0]], dtype=DTYPE)
    entry = pq.denominator_ledger_entry(
        "mixed_sign", denominator,
        numerator_degree_upper_bound=5,
        denominator_degree_upper_bound=4,
    )
    assert entry.bound_kind == "observed_tensor_only"
    assert entry.observation_count == 2
    assert not entry.observed_all_positive
    with pytest.raises(ValueError, match="safety threshold"):
        pq.denominator_ledger_entry(
            "zero", torch.tensor([0.0], dtype=DTYPE),
            numerator_degree_upper_bound=1,
            denominator_degree_upper_bound=1,
        )


def test_projective_pair_rejects_nonfinite_and_dtype_mismatch():
    bad_dtype = pq.ProjectivePair(
        torch.ones(2, 3, dtype=torch.float64),
        torch.ones(2, 1, dtype=torch.float32),
        1,
        0,
    )
    with pytest.raises(ValueError, match="share dtype"):
        pq.projective_value(bad_dtype)
    nonfinite = pq.ProjectivePair(
        torch.tensor([[1.0, float("nan")]], dtype=DTYPE),
        torch.ones(1, 1, dtype=DTYPE),
        1,
        0,
    )
    with pytest.raises(ValueError, match="finite"):
        pq.projective_value(nonfinite)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_pade_rational_norm_exact_pair_and_degree_ledger(degree: int):
    norm = _norm(degree=degree)
    x = torch.randn(4, 5, dtype=DTYPE)
    pair = pq.rational_norm_projective(x, norm, label=f"pade{degree}")
    torch.testing.assert_close(pq.projective_value(pair), norm(x), rtol=2e-12, atol=2e-12)
    assert pair.numerator_degree_upper_bound == 1 + 2 * degree
    assert pair.denominator_degree_upper_bound == 2 * degree
    assert pair.denominator_ledger[0].label == f"pade{degree}"
    assert pair.denominator_ledger[0].observation_count == 4


def test_pade_compiler_rejects_nonfixed_uninitialized_or_wrong_variant():
    x = torch.randn(2, 3, dtype=DTYPE)
    training = RationalNorm(variant="pade").double().train()
    training.initialized.fill_(True)
    with pytest.raises(ValueError, match="eval-mode or frozen"):
        pq.rational_norm_projective(x, training)
    uninitialized = RationalNorm(variant="pade").double().eval()
    with pytest.raises(ValueError, match="initialized"):
        pq.rational_norm_projective(x, uninitialized)
    wrong = RationalNorm(variant="meansq").double().eval()
    wrong.initialized.fill_(True)
    with pytest.raises(ValueError, match="variant='pade'"):
        pq.rational_norm_projective(x, wrong)


def test_rational_norm_float32_deployed_path_is_unchanged():
    norm = _norm().float()
    x = torch.randn(5, 7, dtype=torch.float32)
    xf = x.float()
    mean_square = xf.square().mean(dim=-1, keepdim=True) + norm.eps
    scale = norm.running_ms.clamp_min(1e-12)
    v = mean_square / scale
    numerator = sum(norm.pa[k] * v ** k for k in range(norm.deg + 1))
    denominator = sum(norm.pb[k] * v ** k for k in range(norm.deg + 1))
    inv_sqrt = (numerator / denominator) * torch.rsqrt(scale)
    historical = xf * inv_sqrt
    torch.testing.assert_close(norm(x), historical, rtol=0, atol=0)


def test_rational_ffn_residual_pair_is_exact_and_degree_accounted():
    torch.manual_seed(11)
    norm = _norm()
    ffn = BilinearFFN(5, rank=9, down_bias=True).double().eval()
    with torch.no_grad():
        ffn.left.bias.normal_()
        ffn.right.bias.normal_()
        ffn.down.bias.normal_()
    x = torch.randn(3, 7, 5, dtype=DTYPE)
    gain = torch.tensor(0.37, dtype=DTYPE)
    pair = pq.rational_ffn_residual_projective(x, norm, ffn, gain)
    expected = x + gain * ffn(norm(x))
    torch.testing.assert_close(
        pq.projective_value(pair, minimum_absolute_denominator=1e-8),
        expected,
        rtol=2e-11,
        atol=2e-11,
    )
    assert pair.numerator_degree_upper_bound == 10
    assert pair.denominator_degree_upper_bound == 8
    assert pair.denominator_ledger[-1].label.endswith("residual_add")


def test_rational_ffn_residual_rejects_nonscalar_gain():
    norm = _norm()
    ffn = BilinearFFN(5, rank=9, down_bias=True).double().eval()
    x = torch.randn(3, 7, 5, dtype=DTYPE)
    with pytest.raises(ValueError, match="fixed scalar"):
        pq.rational_ffn_residual_projective(
            x, norm, ffn, torch.ones(7, 1, dtype=DTYPE)
        )


def test_quotient_jacobian_matches_autograd_for_nonlinear_pair():
    x = torch.tensor([0.2, -0.4, 0.7], dtype=DTYPE, requires_grad=True)

    def numerator_fn(value):
        return torch.stack((value[0] ** 2 + value[1], value[1] * value[2] - value[0]))

    def denominator_fn(value):
        return 1.8 + value.square().sum()

    numerator = numerator_fn(x)
    denominator = denominator_fn(x)
    numerator_jacobian = torch.autograd.functional.jacobian(numerator_fn, x)
    denominator_gradient = torch.autograd.grad(denominator, x)[0]
    actual = pq.quotient_jacobian(
        numerator.detach(), denominator.detach(), numerator_jacobian, denominator_gradient
    )
    expected = torch.autograd.functional.jacobian(
        lambda value: numerator_fn(value) / denominator_fn(value), x
    )
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("seed", range(8))
def test_expanded_pullback_includes_cross_terms_and_matches_direct(seed: int):
    args = _quotient_fixture(seed)
    result = pq.quotient_metric_pullback(*args)
    direct = result.jacobian.T @ args[-1] @ result.jacobian
    torch.testing.assert_close(result.metric, direct, rtol=5e-12, atol=5e-12)
    without_cross = result.numerator_term + result.denominator_term
    assert float((without_cross - direct).abs().max().item()) > 1e-6
    torch.testing.assert_close(result.cross_term_left.T, result.cross_term_right)


def test_projective_output_metric_has_the_radial_gauge_in_its_nullspace():
    numerator, denominator, _, _, metric = _quotient_fixture(21)
    result = pq.projective_output_pullback(numerator, denominator, metric)
    radial = torch.cat((numerator, denominator.reshape(1)))
    torch.testing.assert_close(result.quotient_jacobian @ radial, torch.zeros(3, dtype=DTYPE), atol=2e-12, rtol=0)
    torch.testing.assert_close(result.metric @ radial, torch.zeros(4, dtype=DTYPE), atol=2e-11, rtol=0)
    torch.testing.assert_close(result.radial_residual, torch.zeros(4, dtype=DTYPE), atol=2e-11, rtol=0)


def test_nonconstant_projective_gauge_preserves_quotient_jacobian_and_pullback():
    numerator, denominator, jn, gd, metric = _quotient_fixture(22)
    scale = torch.tensor(-1.7, dtype=DTYPE)
    gauge_gradient = torch.tensor([0.2, -0.1, 0.4, 0.7], dtype=DTYPE)
    scaled_numerator = scale * numerator
    scaled_denominator = scale * denominator
    scaled_jn = scale * jn + numerator[:, None] * gauge_gradient[None, :]
    scaled_gd = scale * gd + denominator * gauge_gradient
    base = pq.quotient_metric_pullback(numerator, denominator, jn, gd, metric)
    transformed = pq.quotient_metric_pullback(
        scaled_numerator, scaled_denominator, scaled_jn, scaled_gd, metric
    )
    torch.testing.assert_close(transformed.jacobian, base.jacobian, rtol=5e-12, atol=5e-12)
    torch.testing.assert_close(transformed.metric, base.metric, rtol=5e-12, atol=5e-12)


def test_projective_chain_pullback_matches_expanded_input_pullback():
    numerator, denominator, jn, gd, metric = _quotient_fixture(23)
    projective = pq.projective_output_pullback(numerator, denominator, metric)
    chain = torch.cat((jn, gd[None, :]), dim=0)
    chained_metric = chain.T @ projective.metric @ chain
    direct = pq.quotient_metric_pullback(numerator, denominator, jn, gd, metric).metric
    torch.testing.assert_close(chained_metric, direct, rtol=5e-12, atol=5e-12)


def test_pqbd_matches_manual_supported_balance_and_is_sorted():
    numerator, denominator, jn, gd, metric = _quotient_fixture(24)
    raw = torch.tensor(
        [[1.2, 0.2, 0.0, 0.0], [0.2, 0.7, 0.0, 0.0], [0.0, 0.0, 0.3, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=DTYPE,
    )
    result = pq.projective_quotient_balanced_decomposition(
        raw, numerator, denominator, jn, gd, metric
    )
    values, vectors = torch.linalg.eigh(raw)
    keep = values > result.upstream_tolerance
    factor = vectors[:, keep] * values[keep].sqrt()[None, :]
    manual = torch.linalg.eigvalsh(factor.T @ result.quotient_pullback @ factor).flip(0).clamp_min(0)
    torch.testing.assert_close(result.eigenvalues, manual, rtol=5e-12, atol=5e-12)
    assert result.upstream_rank == 3
    assert bool((result.eigenvalues[:-1] >= result.eigenvalues[1:]).all())
    assert result.object_kind == pq.PQBD_OBJECT_KIND


def test_pqbd_input_coordinate_change_preserves_spectrum():
    numerator, denominator, jn, gd, metric = _quotient_fixture(25)
    generator = torch.Generator().manual_seed(25)
    base_matrix = torch.randn(4, 4, generator=generator, dtype=DTYPE)
    upstream = base_matrix @ base_matrix.T + 0.4 * torch.eye(4, dtype=DTYPE)
    gauge = torch.randn(4, 4, generator=generator, dtype=DTYPE) + 2.0 * torch.eye(4, dtype=DTYPE)
    inverse = torch.linalg.inv(gauge)
    transformed_upstream = gauge @ upstream @ gauge.T
    base = pq.projective_quotient_balanced_decomposition(
        upstream, numerator, denominator, jn, gd, metric
    )
    transformed = pq.projective_quotient_balanced_decomposition(
        transformed_upstream,
        numerator,
        denominator,
        jn @ inverse,
        torch.linalg.solve(gauge.T, gd),
        metric,
    )
    torch.testing.assert_close(transformed.eigenvalues, base.eigenvalues, rtol=2e-10, atol=2e-10)
    assert float((base.eigenvalues[:-1] - base.eigenvalues[1:]).abs().min().item()) > 1e-6
    mapped = inverse @ transformed.state_directions[:, 0]
    reference = base.state_directions[:, 0]
    cosine = torch.dot(mapped, reference).abs() / (
        torch.linalg.vector_norm(mapped) * torch.linalg.vector_norm(reference)
    )
    assert float(cosine.item()) > 1 - 1e-10


def test_pqbd_zero_upstream_support_is_well_defined():
    numerator, denominator, jn, gd, metric = _quotient_fixture(26)
    result = pq.projective_quotient_balanced_decomposition(
        torch.zeros(4, 4, dtype=DTYPE), numerator, denominator, jn, gd, metric
    )
    assert result.upstream_rank == 0
    assert result.eigenvalues.numel() == 0
    assert result.state_directions.shape == (4, 0)


def test_pqbd_support_rank_is_an_assertion_not_euclidean_truncation():
    numerator, denominator, jn, gd, metric = _quotient_fixture(126)
    with pytest.raises(ValueError, match="full positive numerical support"):
        pq.projective_quotient_balanced_decomposition(
            torch.eye(4, dtype=DTYPE),
            numerator,
            denominator,
            jn,
            gd,
            metric,
            support_rank=3,
        )
    rank_three = torch.diag(torch.tensor([3.0, 2.0, 1.0, 0.0], dtype=DTYPE))
    result = pq.projective_quotient_balanced_decomposition(
        rank_three,
        numerator,
        denominator,
        jn,
        gd,
        metric,
        support_rank=3,
    )
    assert result.upstream_rank == 3


def test_tiny_rational_attention_oracle_matches_explicit_masked_forward():
    module = _rational_attention(31)
    tokens = torch.randn(2, 3, 4, dtype=DTYPE)
    mask = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 0, 0]], dtype=DTYPE)
    rows = pq.tiny_rational_attention_projective_oracle(module, tokens, mask, maximum_routes=20)
    actual = pq.stack_projective_rows(rows, minimum_absolute_denominator=1e-8)
    expected = module(tokens, mask=mask, method="explicit")
    torch.testing.assert_close(actual, expected, rtol=5e-10, atol=5e-10)
    assert len(rows) == 3
    assert rows[2].denominator_degree_upper_bound == 0


def test_tiny_rational_attention_oracle_matches_causal_forward_and_gauge():
    module = _rational_attention(32, causal=True)
    tokens = torch.randn(1, 3, 4, dtype=DTYPE)
    rows = pq.tiny_rational_attention_projective_oracle(module, tokens, maximum_routes=20)
    actual = pq.stack_projective_rows(rows, minimum_absolute_denominator=1e-8)
    torch.testing.assert_close(actual, module(tokens), rtol=5e-10, atol=5e-10)
    gauged = tuple(pq.projective_rescale(row, index + 2.0) for index, row in enumerate(rows))
    torch.testing.assert_close(
        pq.stack_projective_rows(gauged, minimum_absolute_denominator=1e-8), actual
    )


def test_tiny_attention_oracle_rejects_wrong_mode_and_route_explosion():
    raw = BilinearAttention(4, 2, qk_norm="none").double().eval()
    tokens = torch.randn(1, 3, 4, dtype=DTYPE)
    with pytest.raises(ValueError, match="qk_norm='rational'"):
        pq.tiny_rational_attention_projective_oracle(raw, tokens)
    rational = _rational_attention(33)
    with pytest.raises(ValueError, match="route limit"):
        pq.tiny_rational_attention_projective_oracle(rational, tokens, maximum_routes=1)


def test_projective_dot_threads_nondefault_denominator_threshold():
    left = pq.ProjectivePair(
        torch.tensor([[1.0, 2.0]], dtype=DTYPE),
        torch.tensor([[0.2]], dtype=DTYPE),
        1,
        1,
    )
    right = pq.ProjectivePair(
        torch.tensor([[3.0, 4.0]], dtype=DTYPE),
        torch.tensor([[0.2]], dtype=DTYPE),
        1,
        1,
    )
    with pytest.raises(ValueError, match="attention.score1"):
        pq._projective_dot(
            left,
            right,
            "attention.score1",
            minimum_absolute_denominator=0.1,
        )
    accepted = pq._projective_dot(
        left,
        right,
        "attention.score1",
        minimum_absolute_denominator=0.01,
    )
    assert accepted.denominator_ledger[-1].observed_minimum_absolute_denominator == pytest.approx(0.04)


def test_tiny_attention_oracle_passes_threshold_to_both_score_paths(monkeypatch):
    original = pq._projective_dot
    observed: list[tuple[str, float]] = []

    def spy(left, right, label, *, minimum_absolute_denominator):
        observed.append((label, minimum_absolute_denominator))
        return original(
            left,
            right,
            label,
            minimum_absolute_denominator=minimum_absolute_denominator,
        )

    monkeypatch.setattr(pq, "_projective_dot", spy)
    threshold = 3e-9
    module = _rational_attention(35)
    tokens = torch.randn(1, 2, 4, dtype=DTYPE)
    pq.tiny_rational_attention_projective_oracle(
        module,
        tokens,
        maximum_routes=20,
        minimum_absolute_denominator=threshold,
    )
    assert {label for label, _ in observed} == {"attention.score1", "attention.score2"}
    assert observed and all(value == threshold for _, value in observed)


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("zero_denominator", "safety threshold"),
        ("nonsymmetric_metric", "symmetric"),
        ("indefinite_metric", "positive semidefinite"),
        ("bad_support_rank", "support_rank"),
    ],
)
def test_negative_controls_fail_closed(mutation: str, match: str):
    numerator, denominator, jn, gd, metric = _quotient_fixture(40)
    upstream = torch.eye(4, dtype=DTYPE)
    kwargs = {}
    if mutation == "zero_denominator":
        denominator = torch.tensor(0.0, dtype=DTYPE)
    elif mutation == "nonsymmetric_metric":
        metric = metric.clone()
        metric[0, 1] += 1.0
    elif mutation == "indefinite_metric":
        metric = -torch.eye(3, dtype=DTYPE)
    else:
        kwargs["support_rank"] = 5
    with pytest.raises(ValueError, match=match):
        pq.projective_quotient_balanced_decomposition(
            upstream, numerator, denominator, jn, gd, metric, **kwargs
        )
