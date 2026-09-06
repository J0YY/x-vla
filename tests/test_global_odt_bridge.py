"""Exact tests for the tiny attention-to-action global ODT bridge."""

from __future__ import annotations

import torch

from xvla.train.canonical_odt import (
    HomogeneousChiTN,
    apply_bond_gauge,
    canonical_environments,
    canonicalize_homogeneous,
    explicit_bond_gram,
    explicit_downstream_tensor,
    materialize_tree_coefficient,
    sorted_eigensystem,
)
from xvla.train.global_odt_bridge import (
    ATTENTION_LEGS,
    attention_coefficient_embedding,
    attention_occurrence_gram,
    build_bilinear_attention_core,
    coefficient_occurrence_gram,
    evaluate_attention_core,
    make_attention_action_tree,
    residual_ffn_action_core,
    shared_attention_gram,
    symmetrize_tied_attention_core,
    trace_before_square_source_gram,
    verify_one_occurrence_tail,
)
from xvla.nn.bilinear import BilinearFFN


DTYPE = torch.float64


def _weights(seed: int, dimension: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
        for name in ("Wq1", "Wk1", "Wq2", "Wk2", "Wv", "Wo")
    }


def _project_occurrence(coefficient: torch.Tensor, projector: torch.Tensor, axis: int) -> torch.Tensor:
    projected = torch.tensordot(projector, coefficient, dims=([1], [axis]))
    return projected.movedim(0, axis)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)
    return float((torch.linalg.vector_norm(actual - expected) / denominator).item())


def _pure_attention_action_tree(
    seed: int = 0, components: int = 1
) -> tuple[torch.Tensor, HomogeneousChiTN]:
    core = torch.stack(
        [
            build_bilinear_attention_core(_weights(seed * components + component))
            for component in range(components)
        ]
    ).sum(dim=0)
    embedding = attention_coefficient_embedding(core, scale=0.5, include_tensor_unit=False)
    generator = torch.Generator().manual_seed(1000 + seed)
    action_core = torch.randn(2, core.shape[0], core.shape[0], generator=generator, dtype=DTYPE)
    action_core = 0.5 * (action_core + action_core.transpose(1, 2))
    return core, make_attention_action_tree(embedding, action_core)


def test_bilinear_attention_core_matches_direct_typed_formula():
    weights = _weights(1, dimension=3)
    core = build_bilinear_attention_core(weights)
    generator = torch.Generator().manual_seed(2)
    for _ in range(12):
        query = torch.randn(3, generator=generator, dtype=DTYPE)
        source = torch.randn(3, generator=generator, dtype=DTYPE)
        q1 = weights["Wq1"] @ query
        k1 = weights["Wk1"] @ source
        q2 = weights["Wq2"] @ query
        k2 = weights["Wk2"] @ source
        value = weights["Wv"] @ source
        expected = (q1 @ k1) * (q2 @ k2) * (weights["Wo"] @ value)
        torch.testing.assert_close(
            evaluate_attention_core(core, query, source), expected, rtol=1e-12, atol=1e-12
        )


def test_typed_and_tied_attention_objects_compute_same_diagonal_function_but_different_grams():
    typed = build_bilinear_attention_core(_weights(22, dimension=3))
    tied = symmetrize_tied_attention_core(typed)
    generator = torch.Generator().manual_seed(23)
    for _ in range(10):
        query = torch.randn(3, generator=generator, dtype=DTYPE)
        source = torch.randn(3, generator=generator, dtype=DTYPE)
        torch.testing.assert_close(
            evaluate_attention_core(tied, query, source),
            evaluate_attention_core(typed, query, source),
            rtol=1e-12,
            atol=1e-12,
        )
    typed_gram = attention_occurrence_gram(typed, "key_1")
    tied_gram = attention_occurrence_gram(tied, "key_1")
    assert not torch.allclose(typed_gram, tied_gram, rtol=1e-5, atol=1e-8)


def test_typed_attention_grams_match_explicit_matricizations_and_tails():
    core = build_bilinear_attention_core(_weights(3))
    raw = torch.tensor([[1.0, -0.2], [0.3, 0.7]], dtype=DTYPE)
    metric = raw @ raw.T
    for leg_index, leg_name in enumerate(ATTENTION_LEGS):
        gram = attention_occurrence_gram(core, leg_name, output_metric=metric)
        moved = core.movedim(leg_index + 1, 0).reshape(core.shape[1], core.shape[0], -1)
        expected = torch.einsum("aok,op,bpk->ab", moved, metric, moved)
        torch.testing.assert_close(gram, expected, rtol=1e-12, atol=1e-12)

        values, vectors = sorted_eigensystem(gram)
        projector = vectors[:, :1] @ vectors[:, :1].T
        projected = _project_occurrence(core, projector, leg_index + 1)
        weighted_error = torch.einsum(
            "o...,op,p...->", core - projected, metric, core - projected
        )
        torch.testing.assert_close(weighted_error, values[1:].sum(), rtol=1e-11, atol=1e-11)


def test_trace_before_square_can_erase_nonzero_query_energy():
    core = torch.zeros(1, 2, 2, 2, 2, 2, dtype=DTYPE)
    core[0, 0, 0, 0, 0, 0] = 1.0
    core[0, 1, 1, 0, 0, 0] = -1.0
    historical = trace_before_square_source_gram(core)
    ordinary = attention_occurrence_gram(core, "key_1")
    assert torch.count_nonzero(historical) == 0
    assert float(torch.trace(ordinary).item()) == 2.0


def test_shared_gram_tail_is_sum_of_separate_occurrence_errors():
    core = build_bilinear_attention_core(_weights(4))
    legs = ("key_1", "key_2", "value")
    gram = shared_attention_gram(core, legs)
    values, vectors = sorted_eigensystem(gram)
    projector = vectors[:, :1] @ vectors[:, :1].T
    explicit = core.new_zeros(())
    for leg in legs:
        axis = 1 + ATTENTION_LEGS.index(leg)
        explicit += torch.linalg.vector_norm(core - _project_occurrence(core, projector, axis)).square()
    torch.testing.assert_close(explicit, values[1:].sum(), rtol=1e-11, atol=1e-11)

    simultaneous = core
    for leg in legs:
        axis = 1 + ATTENTION_LEGS.index(leg)
        simultaneous = _project_occurrence(simultaneous, projector, axis)
    simultaneous_error = torch.linalg.vector_norm(core - simultaneous).square()
    assert float(simultaneous_error.item()) <= float(explicit.item()) * (1.0 + 1e-11)


def test_tensor_unit_embedding_and_residual_ffn_action_core_reconstruct():
    core = build_bilinear_attention_core(_weights(24))
    embedding = attention_coefficient_embedding(core, scale=0.25, include_tensor_unit=True)
    assert embedding.shape == (3, 1 + 3**5)
    assert embedding[0, 0] == 1
    assert torch.count_nonzero(embedding[0, 1:]) == 0
    torch.testing.assert_close(embedding[1:, 1:], 0.25 * core.reshape(2, -1))

    torch.manual_seed(25)
    module = BilinearFFN(2, rank=5, out_dim=2, down_bias=True).double()
    action_weight = torch.tensor([[0.4, -0.7], [1.1, 0.2]], dtype=DTYPE)
    action_bias = torch.tensor([0.3, -0.1], dtype=DTYPE)
    action_core = residual_ffn_action_core(module, action_weight, action_bias)
    hidden = torch.randn(11, 2, dtype=DTYPE)
    homogeneous = torch.cat((torch.ones(11, 1, dtype=DTYPE), hidden), dim=1)
    reconstructed = torch.einsum("bi,aij,bj->ba", homogeneous, action_core, homogeneous)
    expected = (hidden + module(hidden)) @ action_weight.T + action_bias
    torch.testing.assert_close(reconstructed, expected, rtol=1e-12, atol=1e-12)


def test_attention_to_action_tree_is_exact_global_odt_at_hidden_cut():
    _, network = _pure_attention_action_tree(seed=5)
    before = materialize_tree_coefficient(network)
    canonical = canonicalize_homogeneous(network)
    after = materialize_tree_coefficient(canonical.network)
    assert _relative_error(after, before) < 1e-11
    assert max(canonical.isometry_errors) < 1e-11
    assert max(canonical.factorization_errors) < 1e-11

    recursive = canonical_environments(canonical.network)[0]
    explicit = explicit_bond_gram(canonical.network, 0)
    torch.testing.assert_close(recursive, explicit, rtol=1e-11, atol=1e-11)

    _, basis = sorted_eigensystem(recursive)
    cut_tensor = explicit_downstream_tensor(canonical.network, 0)
    tail_check = verify_one_occurrence_tail(cut_tensor, recursive, basis[:, :1])
    assert tail_check.relative_difference < 1e-10


def test_global_bridge_spectrum_is_gauge_invariant_and_raw_local_metric_is_not():
    core, network = _pure_attention_action_tree(seed=6)
    transform = torch.tensor([[1.8, 0.4], [-0.2, 0.7]], dtype=DTYPE)
    gauged = apply_bond_gauge(network, 0, transform)
    original_coefficient = materialize_tree_coefficient(network)
    gauged_coefficient = materialize_tree_coefficient(gauged)
    torch.testing.assert_close(gauged_coefficient, original_coefficient, rtol=1e-11, atol=1e-11)

    original_canonical = canonicalize_homogeneous(network).network
    gauged_canonical = canonicalize_homogeneous(gauged).network
    values_a, _ = sorted_eigensystem(canonical_environments(original_canonical)[0])
    values_b, _ = sorted_eigensystem(canonical_environments(gauged_canonical)[0])
    torch.testing.assert_close(values_a, values_b, rtol=1e-10, atol=1e-10)

    gauged_core = torch.einsum("uv,vpqrst->upqrst", transform, core)
    local_a = attention_occurrence_gram(core, "key_1")
    local_b = attention_occurrence_gram(gauged_core, "key_1")
    assert not torch.allclose(local_a, local_b, rtol=1e-4, atol=1e-6)
    inverse = torch.linalg.inv(transform)
    transported_output_metric = inverse.T @ inverse
    transported_local_b = attention_occurrence_gram(
        gauged_core, "key_1", output_metric=transported_output_metric
    )
    torch.testing.assert_close(transported_local_b, local_a, rtol=1e-11, atol=1e-11)


def test_global_same_source_cut_is_optimal_for_its_declared_typed_object():
    core, network = _pure_attention_action_tree(seed=7, components=2)
    full = materialize_tree_coefficient(network)
    input_dimension = core.shape[1]
    full_typed = full.reshape((full.shape[0],) + (input_dimension,) * 10)
    left_key_1_axis = 3
    global_gram = coefficient_occurrence_gram(full_typed, left_key_1_axis)
    local_gram = attention_occurrence_gram(core, "key_1")
    trace_gram = trace_before_square_source_gram(core)

    _, global_vectors = sorted_eigensystem(global_gram)
    _, local_vectors = sorted_eigensystem(local_gram)
    _, trace_vectors = sorted_eigensystem(trace_gram)
    errors = []
    for vectors in (global_vectors, local_vectors, trace_vectors):
        projector = vectors[:, :1] @ vectors[:, :1].T
        approximation = _project_occurrence(full_typed, projector, left_key_1_axis)
        errors.append(torch.linalg.vector_norm(full_typed - approximation).square())
    assert float(errors[0].item()) <= float(errors[1].item()) + 1e-10
    assert float(errors[0].item()) <= float(errors[2].item()) + 1e-10
    assert float(errors[1].item() - errors[0].item()) > 1e-8
    assert not torch.allclose(
        global_vectors[:, :1] @ global_vectors[:, :1].T,
        local_vectors[:, :1] @ local_vectors[:, :1].T,
        rtol=1e-6,
        atol=1e-8,
    )

    transform = torch.tensor([[1.4, -0.3], [0.2, 0.8]], dtype=DTYPE)
    inverse = torch.linalg.inv(transform)
    gauged_core = torch.einsum("uv,vpqrst->upqrst", transform, core)
    gauged_embedding = attention_coefficient_embedding(
        gauged_core, scale=0.5, include_tensor_unit=False
    )
    gauged_action_core = torch.einsum(
        "oij,ia,jb->oab", network.cores[0], inverse, inverse
    )
    gauged_network = make_attention_action_tree(gauged_embedding, gauged_action_core)
    gauged_full = materialize_tree_coefficient(gauged_network)
    gauged_typed = gauged_full.reshape_as(full_typed)
    torch.testing.assert_close(gauged_typed, full_typed, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(
        coefficient_occurrence_gram(gauged_typed, left_key_1_axis),
        global_gram,
        rtol=1e-11,
        atol=1e-11,
    )
