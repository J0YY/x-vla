"""Load-bearing tests for canonical homogeneous tree ODT."""

from __future__ import annotations

import itertools
import math

import pytest

pytest.skip(
    "retired historical canonical_odt.py path; canonical certification is direct-RQ only",
    allow_module_level=True,
)

import torch

from xvla.models.chi_mlp import ChiMLP, ChiMLPConfig
from xvla.nn.normalization import RmsBatchNorm
from xvla.train.canonical_odt import (
    HomogeneousChiTN,
    apply_bond_gauge,
    canonical_environments,
    canonical_prefix_overlaps,
    canonicalize_homogeneous,
    coefficient_error_squared,
    coefficient_inner_product,
    compress_homogeneous,
    discarded_trace,
    eigengap_diagnostics,
    explicit_bond_gram,
    export_homogeneous_network,
    hierarchical_tail_bound_squared,
    homogeneous_forward,
    local_environments,
    materialize_tree_coefficient,
    odt_eigensystems,
    polar_orthogonal_transport,
    prefix_coefficients,
    projector_perturbation_certificate,
    reduced_rq_rows,
    sorted_eigensystem,
    top_bases,
)


DTYPE = torch.float64


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(actual - expected)
    denominator = torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)
    return float((numerator / denominator).item())


def _biased_model() -> ChiMLP:
    torch.manual_seed(41)
    config = ChiMLPConfig(in_dim=6, dim=3, n_layers=2, num_classes=2, norm="scalar_rbn")
    model = ChiMLP(config).double()
    with torch.no_grad():
        model.embed.bias.normal_(std=0.2)
        model.head.bias.normal_(std=0.2)
        for layer in model.layers:
            layer.left.bias.normal_(std=0.2)
            layer.right.bias.normal_(std=0.2)
        for index, norm in enumerate(model.norms):
            assert isinstance(norm, RmsBatchNorm)
            norm.running_rms.fill_(1.2 + 0.1 * index)
            norm.initialized.fill_(True)
            norm.freeze()
    return model.eval()


def _random_tree(seed: int = 0, layers: int = 2) -> HomogeneousChiTN:
    generator = torch.Generator().manual_seed(seed)
    input_homogeneous = 3
    hidden = 3
    embedding = torch.randn(hidden, input_homogeneous, generator=generator, dtype=DTYPE)
    embedding = embedding + 1.5 * torch.eye(hidden, input_homogeneous, dtype=DTYPE)
    cores = []
    for _ in range(layers):
        core = torch.randn(hidden, hidden, hidden, generator=generator, dtype=DTYPE)
        core = 0.5 * (core + core.transpose(1, 2))
        for output in range(hidden):
            core[output, output, output] += 1.0
        cores.append(core)
    head = torch.randn(2, hidden, generator=generator, dtype=DTYPE)
    return HomogeneousChiTN(embedding, tuple(cores), head)


def _heterogeneous_tree(seed: int = 0) -> HomogeneousChiTN:
    """Non-square dimensions expose otherwise shape-valid axis mistakes."""

    generator = torch.Generator().manual_seed(seed)
    dimensions = (3, 2, 4)
    embedding = torch.randn(3, 4, generator=generator, dtype=DTYPE)
    cores = []
    previous = embedding.shape[0]
    for output in dimensions[1:]:
        core = torch.randn(output, previous, previous, generator=generator, dtype=DTYPE)
        cores.append(0.5 * (core + core.transpose(1, 2)))
        previous = output
    head = torch.randn(2, previous, generator=generator, dtype=DTYPE)
    return HomogeneousChiTN(embedding, tuple(cores), head)


def _basis_enumerated_coefficient(network: HomogeneousChiTN) -> torch.Tensor:
    """Independent oracle that never contracts two coefficient tensors directly."""

    leaf_count = 2 ** network.n_layers
    input_dimension = network.embedding.shape[1]
    shape = (network.head.shape[0],) + (input_dimension,) * leaf_count
    coefficient = torch.zeros(shape, dtype=network.embedding.dtype)
    eye = torch.eye(input_dimension, dtype=network.embedding.dtype)

    for leaf_indices in itertools.product(range(input_dimension), repeat=leaf_count):
        nodes = [network.embedding @ eye[index] for index in leaf_indices]
        for core in network.cores:
            nodes = [
                torch.einsum("oab,a,b->o", core, nodes[index], nodes[index + 1])
                for index in range(0, len(nodes), 2)
            ]
        coefficient[(slice(None),) + leaf_indices] = network.head @ nodes[0]
    return coefficient


def _diagonal_tensor_forward(coefficient: torch.Tensor, homogeneous_x: torch.Tensor) -> torch.Tensor:
    value = coefficient
    for _ in range(coefficient.ndim - 1):
        value = torch.tensordot(value, homogeneous_x, dims=([1], [0]))
    return value


def _well_conditioned_gauge(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    left, _ = torch.linalg.qr(
        torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
    )
    right, _ = torch.linalg.qr(
        torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
    )
    scales = torch.linspace(0.55, 1.85, dimension, dtype=DTYPE)
    gauge = left @ torch.diag(scales) @ right.T
    if dimension > 1:
        shear = torch.eye(dimension, dtype=DTYPE)
        shear[0, 1] = 0.3
        gauge = shear @ gauge
    return gauge


def _row_space_projector(rows: torch.Tensor) -> torch.Tensor:
    basis, _ = torch.linalg.qr(rows.T, mode="reduced")
    return basis @ basis.T


def _basis_enumerated_cut_tensor(
    network: HomogeneousChiTN, bond: int, occurrence: int
) -> torch.Tensor:
    """Independent cut oracle using basis insertion and scalar tree evaluation."""

    layer_count = network.n_layers
    if not 0 <= bond <= layer_count:
        raise ValueError("invalid bond")
    occurrences = 2 ** (layer_count - bond)
    if not 0 <= occurrence < occurrences:
        raise ValueError("invalid occurrence")
    leaf_count = 2**layer_count
    first_inside = occurrence * (2**bond)
    inside = set(range(first_inside, first_inside + 2**bond))
    outside = [leaf for leaf in range(leaf_count) if leaf not in inside]
    input_dimension = network.embedding.shape[1]
    cut_dimension = network.bond_dims[bond]
    shape = (cut_dimension, network.head.shape[0]) + (input_dimension,) * len(outside)
    result = torch.zeros(shape, dtype=network.embedding.dtype)
    input_eye = torch.eye(input_dimension, dtype=network.embedding.dtype)
    cut_eye = torch.eye(cut_dimension, dtype=network.embedding.dtype)

    def evaluate_node(
        level: int,
        position: int,
        assignment: dict[int, int],
        cut_vector: torch.Tensor,
    ) -> torch.Tensor:
        if level == bond and position == occurrence:
            return cut_vector
        if level == 0:
            return network.embedding @ input_eye[assignment[position]]
        left = evaluate_node(level - 1, 2 * position, assignment, cut_vector)
        right = evaluate_node(level - 1, 2 * position + 1, assignment, cut_vector)
        return torch.einsum("oab,a,b->o", network.cores[level - 1], left, right)

    for outside_indices in itertools.product(range(input_dimension), repeat=len(outside)):
        assignment = dict(zip(outside, outside_indices))
        for cut_index in range(cut_dimension):
            root = evaluate_node(layer_count, 0, assignment, cut_eye[cut_index])
            result[(cut_index, slice(None)) + outside_indices] = network.head @ root
    return result


def _cut_gram(cut: torch.Tensor, output_metric: torch.Tensor | None = None) -> torch.Tensor:
    rows = cut.reshape(cut.shape[0], cut.shape[1], -1)
    if output_metric is None:
        output_metric = torch.eye(cut.shape[1], dtype=cut.dtype)
    return torch.einsum("aok,op,bpk->ab", rows, output_metric, rows)


def test_homogeneous_export_matches_module_with_nonzero_biases():
    model = _biased_model()
    network = export_homogeneous_network(model)
    x = torch.randn(13, 6, dtype=DTYPE)
    expected = model(x)[0]
    actual = homogeneous_forward(network, x)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert network.object_kind == "tree_topology"


def test_zero_core_linear_tree_canonicalization_pushes_embedding_factor_once():
    generator = torch.Generator().manual_seed(20260903)
    network = HomogeneousChiTN(
        embedding=torch.randn(3, 5, generator=generator, dtype=DTYPE),
        cores=(),
        head=torch.randn(2, 3, generator=generator, dtype=DTYPE),
    )
    canonical = canonicalize_homogeneous(network)
    x = torch.randn(11, 4, generator=generator, dtype=DTYPE)

    assert canonical.network.n_layers == 0
    assert len(canonical.raw_from_canonical) == 1
    assert torch.allclose(
        homogeneous_forward(canonical.network, x),
        homogeneous_forward(network, x),
        atol=1e-12,
        rtol=1e-12,
    )


def test_row_rq_orientation_sign_and_row_isometry():
    matrix = torch.tensor(
        [[2.0, -1.0, 0.5, 0.2], [0.3, 1.7, -0.4, 0.9], [1.1, 0.2, 1.9, -0.6]],
        dtype=DTYPE,
    )
    factor, isometry = reduced_rq_rows(matrix)
    assert torch.allclose(factor @ isometry, matrix, atol=1e-12, rtol=1e-12)
    assert torch.allclose(isometry @ isometry.T, torch.eye(3, dtype=DTYPE), atol=1e-12)
    assert torch.all(torch.diagonal(factor) >= 0)


def test_rank_deficient_rq_remains_exact_without_numerical_shrinkage():
    matrix = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=DTYPE)
    factor, isometry = reduced_rq_rows(matrix)
    assert factor.shape == (2, 2)
    assert isometry.shape == (2, 3)
    assert torch.allclose(factor @ isometry, matrix, atol=1e-12, rtol=1e-12)
    assert torch.allclose(isometry @ isometry.T, torch.eye(2, dtype=DTYPE), atol=1e-12)


@pytest.mark.parametrize(
    "matrix",
    [
        torch.tensor([[1.0, 0.0], [0.0, 1e-30]], dtype=DTYPE),
        torch.zeros(2, 3, dtype=DTYPE),
    ],
)
def test_near_singular_and_zero_rq_are_exact(matrix):
    factor, isometry = reduced_rq_rows(matrix)
    assert factor.shape[1] == matrix.shape[0]
    assert torch.equal(factor @ isometry, matrix)
    assert torch.allclose(
        isometry @ isometry.T, torch.eye(matrix.shape[0], dtype=DTYPE), atol=1e-12
    )


def test_reduced_rq_shrinks_expanding_rank_deficient_bond_exactly():
    matrix = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=DTYPE
    )
    factor, isometry = reduced_rq_rows(matrix)
    assert factor.shape == (4, 2)
    assert isometry.shape == (2, 2)
    assert torch.allclose(factor @ isometry, matrix, atol=1e-12, rtol=1e-12)
    assert torch.allclose(isometry @ isometry.T, torch.eye(2, dtype=DTYPE), atol=1e-12)


def test_materialized_coefficient_matches_basis_enumeration():
    network = _random_tree(seed=1)
    direct = materialize_tree_coefficient(network)
    oracle = _basis_enumerated_coefficient(network)
    assert _relative_error(direct, oracle) < 1e-12


def test_canonicalization_preserves_tensor_and_makes_row_isometries():
    network = _random_tree(seed=2)
    before = _basis_enumerated_coefficient(network)
    canonical = canonicalize_homogeneous(network)
    after = _basis_enumerated_coefficient(canonical.network)
    assert _relative_error(after, before) < 1e-11
    assert max(canonical.isometry_errors) < 1e-11
    assert max(canonical.symmetry_errors) < 1e-11
    assert max(canonical.factorization_errors) < 1e-12

    x = torch.randn(8, network.embedding.shape[1] - 1, dtype=DTYPE)
    assert torch.allclose(
        homogeneous_forward(canonical.network, x),
        homogeneous_forward(network, x),
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize("zero", [False, True])
def test_rank_deficient_complete_tree_is_preserved_and_gauge_invariant(zero):
    network = _random_tree(seed=21)
    embedding = network.embedding.clone()
    if zero:
        embedding.zero_()
    else:
        embedding[2] = embedding[1]
    network = HomogeneousChiTN(embedding, network.cores, network.head)
    canonical = canonicalize_homogeneous(network)
    assert max(canonical.factorization_errors) < 1e-12
    assert _relative_error(
        _basis_enumerated_coefficient(canonical.network),
        _basis_enumerated_coefficient(network),
    ) < 1e-11

    gauged = network
    for bond, dimension in enumerate(network.bond_dims):
        gauged = apply_bond_gauge(gauged, bond, _well_conditioned_gauge(dimension, 210 + bond))
    gauged_canonical = canonicalize_homogeneous(gauged).network
    spectra_a = [system[0] for system in odt_eigensystems(canonical.network)]
    spectra_b = [system[0] for system in odt_eigensystems(gauged_canonical)]
    for values_a, values_b in zip(spectra_a, spectra_b):
        assert torch.allclose(values_a, values_b, atol=1e-8, rtol=1e-8)


def test_recursive_environments_match_independent_basis_cut_oracle():
    canonical = canonicalize_homogeneous(_random_tree(seed=3)).network
    recursive = canonical_environments(canonical)
    for bond, gram in enumerate(recursive):
        for occurrence in {0, 2 ** (canonical.n_layers - bond) - 1}:
            cut = _basis_enumerated_cut_tensor(canonical, bond, occurrence)
            explicit = _cut_gram(cut)
            assert _relative_error(gram, explicit) < 1e-11, (bond, occurrence)


def test_environment_apis_reject_raw_or_nonsymmetric_tied_leg_networks():
    raw = _random_tree(seed=31)
    with pytest.raises(ValueError, match="not row-isometric"):
        canonical_environments(raw)
    with pytest.raises(ValueError, match="not row-isometric"):
        local_environments(raw)

    generator = torch.Generator().manual_seed(311)
    rows, _ = torch.linalg.qr(
        torch.randn(9, 2, generator=generator, dtype=DTYPE), mode="reduced"
    )
    unequal = HomogeneousChiTN(
        torch.eye(3, dtype=DTYPE),
        (rows.T.reshape(2, 3, 3),),
        torch.eye(2, dtype=DTYPE),
    )
    with pytest.raises(ValueError, match="not input-leg symmetric"):
        canonical_environments(unequal)
    with pytest.raises(ValueError, match="not input-leg symmetric"):
        local_environments(unequal)


def test_odt_rejects_nonsymmetric_core_even_when_leg_grams_are_equal():
    inverse_sqrt_two = 2.0**-0.5
    antisymmetric = torch.tensor(
        [[[0.0, inverse_sqrt_two], [-inverse_sqrt_two, 0.0]]], dtype=DTYPE
    )
    network = HomogeneousChiTN(
        torch.eye(2, dtype=DTYPE),
        (antisymmetric,),
        torch.ones(1, 1, dtype=DTYPE),
    )

    left = torch.einsum("oab,oAb->aA", antisymmetric, antisymmetric)
    right = torch.einsum("oab,oaB->bB", antisymmetric, antisymmetric)
    assert torch.allclose(left, right, atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match="not input-leg symmetric"):
        canonicalize_homogeneous(network)
    with pytest.raises(ValueError, match="not input-leg symmetric"):
        canonical_environments(network)


def test_rank_deficient_symmetric_core_gets_symmetric_isometric_completion():
    network = HomogeneousChiTN(
        torch.eye(2, dtype=DTYPE),
        (torch.zeros(2, 2, 2, dtype=DTYPE),),
        torch.tensor([[0.5, -1.0], [1.5, 0.25]], dtype=DTYPE),
    )
    canonical = canonicalize_homogeneous(network)
    assert canonical.symmetry_errors[0] < 1e-12
    assert canonical.isometry_errors[1] < 1e-12
    assert canonical.factorization_errors[1] == 0.0
    assert torch.equal(
        materialize_tree_coefficient(canonical.network),
        materialize_tree_coefficient(network),
    )
    canonical_environments(canonical.network)


def test_nonzero_rank_deficient_symmetric_core_is_preserved_exactly():
    core = torch.tensor(
        [
            [[1.0, 2.0], [2.0, -1.0]],
            [[1.0, 2.0], [2.0, -1.0]],
        ],
        dtype=DTYPE,
    )
    network = HomogeneousChiTN(
        torch.eye(2, dtype=DTYPE),
        (core,),
        torch.tensor([[0.75, -0.25]], dtype=DTYPE),
    )
    canonical = canonicalize_homogeneous(network)
    assert canonical.symmetry_errors[0] < 1e-12
    assert canonical.isometry_errors[1] < 1e-12
    assert canonical.factorization_errors[1] < 1e-12
    assert torch.allclose(
        materialize_tree_coefficient(canonical.network),
        materialize_tree_coefficient(network),
        atol=1e-12,
        rtol=1e-12,
    )
    canonical_environments(canonical.network)


def test_output_metric_is_validated_and_matches_rank_one_cut_oracle():
    canonical = canonicalize_homogeneous(_random_tree(seed=32)).network
    output_vector = torch.tensor([0.4, -1.2], dtype=DTYPE)
    metric = torch.outer(output_vector, output_vector)
    recursive = canonical_environments(canonical, output_metric=metric)
    for bond, gram in enumerate(recursive):
        cut = _basis_enumerated_cut_tensor(canonical, bond, 0)
        assert _relative_error(gram, _cut_gram(cut, metric)) < 1e-11

    nonsymmetric = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=DTYPE)
    indefinite = torch.diag(torch.tensor([1.0, -0.1], dtype=DTYPE))
    with pytest.raises(ValueError, match="symmetric"):
        canonical_environments(canonical, output_metric=nonsymmetric)
    with pytest.raises(ValueError, match="positive semidefinite"):
        canonical_environments(canonical, output_metric=indefinite)


def test_nonorthogonal_gauges_preserve_tensor_spectra_and_physical_subspaces():
    network = _random_tree(seed=4)
    gauged = network
    for bond, dimension in enumerate(network.bond_dims):
        gauged = apply_bond_gauge(gauged, bond, _well_conditioned_gauge(dimension, 100 + bond))

    original_tensor = _basis_enumerated_coefficient(network)
    gauged_tensor = _basis_enumerated_coefficient(gauged)
    assert _relative_error(gauged_tensor, original_tensor) < 1e-10

    canonical_a = canonicalize_homogeneous(network).network
    canonical_b = canonicalize_homogeneous(gauged).network
    systems_a = odt_eigensystems(canonical_a)
    systems_b = odt_eigensystems(canonical_b)
    prefixes_a = prefix_coefficients(canonical_a)
    prefixes_b = prefix_coefficients(canonical_b)

    for bond, ((values_a, vectors_a), (values_b, vectors_b)) in enumerate(
        zip(systems_a, systems_b)
    ):
        assert torch.allclose(values_a, values_b, atol=1e-9, rtol=1e-9), bond
        rank = 2
        features_a = vectors_a[:, :rank].T @ prefixes_a[bond].reshape(values_a.numel(), -1)
        features_b = vectors_b[:, :rank].T @ prefixes_b[bond].reshape(values_b.numel(), -1)
        projector_a = _row_space_projector(features_a)
        projector_b = _row_space_projector(features_b)
        assert _relative_error(projector_a, projector_b) < 1e-8, bond


@pytest.mark.parametrize("factory", [lambda: _random_tree(seed=5), lambda: _heterogeneous_tree(5)])
def test_full_rank_diagonalized_network_preserves_tensor(factory):
    canonical = canonicalize_homogeneous(factory()).network
    systems = odt_eigensystems(canonical)
    ranks = list(canonical.bond_dims)
    rotated = compress_homogeneous(canonical, top_bases(systems, ranks))
    expected = _basis_enumerated_coefficient(canonical)
    actual = _basis_enumerated_coefficient(rotated)
    assert _relative_error(actual, expected) < 1e-10
    assert rotated.bond_dims == canonical.bond_dims


def test_compression_rejects_nonorthonormal_columns():
    canonical = canonicalize_homogeneous(_random_tree(seed=51)).network
    bad = torch.eye(canonical.bond_dims[1], dtype=DTYPE)
    bad[:, 0] *= 2.0
    with pytest.raises(ValueError, match="not orthonormal"):
        compress_homogeneous(canonical, {1: bad})


def test_root_trace_tail_equals_actual_coefficient_error():
    canonical = canonicalize_homogeneous(_random_tree(seed=6)).network
    systems = odt_eigensystems(canonical)
    root = canonical.n_layers
    rank = 1
    basis = systems[root][1][:, :rank]
    compressed = compress_homogeneous(canonical, {root: basis})
    full_tensor = materialize_tree_coefficient(canonical)
    compressed_tensor = materialize_tree_coefficient(compressed)
    actual_error_squared = torch.linalg.vector_norm(full_tensor - compressed_tensor).square()
    predicted_error_squared = discarded_trace(systems[root][0], rank)
    assert torch.allclose(
        actual_error_squared, predicted_error_squared, atol=1e-9, rtol=1e-9
    )


def test_top_projector_actual_cut_error_is_trace_tail_and_optimal():
    canonical = canonicalize_homogeneous(_random_tree(seed=7)).network
    generator = torch.Generator().manual_seed(700)
    for bond in range(canonical.n_layers):
        cut = _basis_enumerated_cut_tensor(canonical, bond, 0)
        rows = cut.reshape(cut.shape[0], -1)
        gram = rows @ rows.T
        values, vectors = sorted_eigensystem(gram)
        rank = 1
        identity = torch.eye(gram.shape[0], dtype=DTYPE)
        projector = vectors[:, :rank] @ vectors[:, :rank].T
        top_loss = torch.linalg.vector_norm((identity - projector) @ rows).square()
        assert torch.allclose(
            top_loss, discarded_trace(values, rank), atol=1e-10, rtol=1e-10
        )

        for _ in range(100):
            random_matrix = torch.randn(gram.shape[0], rank, generator=generator, dtype=DTYPE)
            random_basis, _ = torch.linalg.qr(random_matrix, mode="reduced")
            random_projector = random_basis @ random_basis.T
            random_loss = torch.linalg.vector_norm(
                (identity - random_projector) @ rows
            ).square()
            assert top_loss <= random_loss + 1e-10


@pytest.mark.parametrize("seed", [8, 18, 28])
def test_simultaneous_truncation_obeys_bound_for_every_rank_vector(seed):
    canonical = canonicalize_homogeneous(_random_tree(seed=seed)).network
    systems = odt_eigensystems(canonical)
    full_tensor = _basis_enumerated_coefficient(canonical)
    rank_ranges = [range(1, dimension + 1) for dimension in canonical.bond_dims]
    for ranks in itertools.product(*rank_ranges):
        compressed = compress_homogeneous(canonical, top_bases(systems, ranks))
        compressed_tensor = _basis_enumerated_coefficient(compressed)
        actual_error_squared = torch.linalg.vector_norm(
            full_tensor - compressed_tensor
        ).square()
        bound = hierarchical_tail_bound_squared(systems, ranks)
        assert actual_error_squared <= bound * (1.0 + 1e-9) + 1e-9, ranks


@pytest.mark.parametrize("seed", [9, 19, 29])
def test_mixed_tree_inner_product_and_error_match_explicit_coefficients(seed):
    canonical = canonicalize_homogeneous(_random_tree(seed=seed)).network
    systems = odt_eigensystems(canonical)
    ranks = tuple(max(1, dimension - 1) for dimension in canonical.bond_dims)
    compressed = compress_homogeneous(canonical, top_bases(systems, ranks))

    full_tensor = _basis_enumerated_coefficient(canonical)
    compressed_tensor = _basis_enumerated_coefficient(compressed)
    expected_inner = torch.sum(full_tensor * compressed_tensor)
    expected_error_squared = torch.linalg.vector_norm(
        full_tensor - compressed_tensor
    ).square()

    assert torch.allclose(
        coefficient_inner_product(canonical, compressed),
        expected_inner,
        atol=1e-9,
        rtol=1e-9,
    )
    assert torch.allclose(
        coefficient_error_squared(canonical, compressed),
        expected_error_squared,
        atol=1e-9,
        rtol=1e-9,
    )


def test_stable_coefficient_error_resolves_full_rank_rotation_and_near_equal_tree():
    canonical = canonicalize_homogeneous(_random_tree(seed=109, layers=3)).network
    systems = odt_eigensystems(canonical)
    rotated = compress_homogeneous(canonical, top_bases(systems, canonical.bond_dims))
    reference_tensor = _basis_enumerated_coefficient(canonical)
    reference_norm = torch.linalg.vector_norm(reference_tensor).square()
    full_rank_error = coefficient_error_squared(canonical, rotated)
    assert torch.sqrt(full_rank_error / reference_norm) < 1e-10

    perturbed_head = rotated.head.clone()
    perturbation = 1e-3
    perturbed_head[0, 0] += perturbation
    perturbed = HomogeneousChiTN(rotated.embedding, rotated.cores, perturbed_head)
    rotated_tensor = _basis_enumerated_coefficient(rotated)
    perturbed_tensor = _basis_enumerated_coefficient(perturbed)
    expected = torch.linalg.vector_norm(rotated_tensor - perturbed_tensor).square()
    actual = coefficient_error_squared(rotated, perturbed)
    assert expected > 0.0
    assert torch.allclose(
        expected,
        torch.tensor(perturbation**2, dtype=DTYPE),
        atol=0.0,
        rtol=1e-4,
    )
    assert torch.allclose(actual, expected, atol=1e-18, rtol=1e-3)


def test_depth_three_all_bond_mixed_error_and_hsvd_bound_match_explicit_tensor():
    canonical = canonicalize_homogeneous(_random_tree(seed=110, layers=3)).network
    systems = odt_eigensystems(canonical)
    full_tensor = _basis_enumerated_coefficient(canonical)
    for ranks in ((3, 3, 3, 3), (2, 3, 2, 1), (1, 1, 1, 1)):
        compressed = compress_homogeneous(canonical, top_bases(systems, ranks))
        compressed_tensor = _basis_enumerated_coefficient(compressed)
        expected_error = torch.linalg.vector_norm(full_tensor - compressed_tensor).square()
        actual_error = coefficient_error_squared(canonical, compressed)
        bound = hierarchical_tail_bound_squared(systems, ranks)
        assert torch.allclose(actual_error, expected_error, atol=1e-8, rtol=1e-8)
        assert actual_error <= bound * (1.0 + 1e-9) + 1e-9


def test_mixed_tree_inner_product_supports_output_metrics_and_heterogeneous_bonds():
    canonical = canonicalize_homogeneous(_heterogeneous_tree(seed=93)).network
    systems = odt_eigensystems(canonical)
    ranks = (2, 1, 3)
    compressed = compress_homogeneous(canonical, top_bases(systems, ranks))
    metric_factor = torch.tensor([[1.0, 0.2], [-0.4, 0.7]], dtype=DTYPE)
    metric = metric_factor.T @ metric_factor

    full_tensor = _basis_enumerated_coefficient(canonical)
    compressed_tensor = _basis_enumerated_coefficient(compressed)
    expected = torch.einsum(
        "u...,uv,v...->",
        full_tensor,
        metric,
        compressed_tensor,
    )
    assert compressed.bond_dims != canonical.bond_dims
    assert torch.allclose(
        coefficient_inner_product(canonical, compressed, metric),
        expected,
        atol=1e-9,
        rtol=1e-9,
    )
    expected_error = torch.einsum(
        "u...,uv,v...->",
        full_tensor - compressed_tensor,
        metric,
        full_tensor - compressed_tensor,
    )
    assert torch.allclose(
        coefficient_error_squared(canonical, compressed, metric),
        expected_error,
        atol=1e-9,
        rtol=1e-9,
    )


@pytest.mark.parametrize(
    ("right", "message"),
    [
        (HomogeneousChiTN(torch.eye(2, dtype=DTYPE), (), torch.ones(2, 2, dtype=DTYPE)), "depth"),
        (_random_tree(seed=95, layers=2), "leaf dimension"),
    ],
)
def test_mixed_tree_inner_product_rejects_incompatible_trees(right, message):
    left = _heterogeneous_tree(seed=94)
    with pytest.raises(ValueError, match=message):
        coefficient_inner_product(left, right)


def test_heterogeneous_tree_coefficient_cut_compression_and_gauge_oracles():
    network = _heterogeneous_tree(seed=81)
    assert _relative_error(
        materialize_tree_coefficient(network), _basis_enumerated_coefficient(network)
    ) < 1e-12
    gauged = network
    for bond, dimension in enumerate(network.bond_dims):
        gauged = apply_bond_gauge(gauged, bond, _well_conditioned_gauge(dimension, 810 + bond))
    assert _relative_error(
        _basis_enumerated_coefficient(gauged), _basis_enumerated_coefficient(network)
    ) < 1e-10

    canonical = canonicalize_homogeneous(network).network
    gauged_canonical = canonicalize_homogeneous(gauged).network
    assert canonical.bond_dims == (3, 2, 3)
    assert gauged_canonical.bond_dims == canonical.bond_dims
    assert _relative_error(
        _basis_enumerated_coefficient(canonical),
        _basis_enumerated_coefficient(network),
    ) < 1e-10
    assert _relative_error(
        _basis_enumerated_coefficient(gauged_canonical),
        _basis_enumerated_coefficient(network),
    ) < 1e-10

    systems = odt_eigensystems(canonical)
    gauged_systems = odt_eigensystems(gauged_canonical)
    prefixes = prefix_coefficients(canonical)
    gauged_prefixes = prefix_coefficients(gauged_canonical)
    for bond, ((values, vectors), (gauged_values, gauged_vectors)) in enumerate(
        zip(systems, gauged_systems)
    ):
        assert torch.allclose(values, gauged_values, atol=1e-8, rtol=1e-8), bond
        rank = min(2, values.numel())
        features = vectors[:, :rank].T @ prefixes[bond].reshape(values.numel(), -1)
        gauged_features = (
            gauged_vectors[:, :rank].T
            @ gauged_prefixes[bond].reshape(gauged_values.numel(), -1)
        )
        assert _relative_error(
            _row_space_projector(features), _row_space_projector(gauged_features)
        ) < 1e-8, bond

    for bond, gram in enumerate(canonical_environments(canonical)):
        cut = _basis_enumerated_cut_tensor(canonical, bond, 0)
        assert _relative_error(gram, _cut_gram(cut)) < 1e-11
    full_rank = compress_homogeneous(canonical, top_bases(systems, canonical.bond_dims))
    assert _relative_error(
        _basis_enumerated_coefficient(full_rank), _basis_enumerated_coefficient(canonical)
    ) < 1e-10


def test_explicit_cut_helper_matches_independent_oracle_on_heterogeneous_tree():
    canonical = canonicalize_homogeneous(_heterogeneous_tree(seed=82)).network
    for bond in range(canonical.n_layers + 1):
        expected = _cut_gram(_basis_enumerated_cut_tensor(canonical, bond, 0))
        actual = explicit_bond_gram(canonical, bond)
        assert _relative_error(actual, expected) < 1e-11, bond


def test_recursive_canonical_prefix_overlaps_match_explicit_prefix_rows():
    raw = _heterogeneous_tree(seed=820)
    base = canonicalize_homogeneous(raw).network
    gauged_raw = raw
    for bond, dimension in enumerate(raw.bond_dims):
        gauged_raw = apply_bond_gauge(
            gauged_raw,
            bond,
            _well_conditioned_gauge(dimension, 821 + bond),
        )
    gauged = canonicalize_homogeneous(gauged_raw).network

    recursive = canonical_prefix_overlaps(gauged, base)
    gauged_prefixes = prefix_coefficients(gauged)
    base_prefixes = prefix_coefficients(base)
    assert len(recursive) == len(base.bond_dims)
    for bond, (actual, left_prefix, right_prefix) in enumerate(
        zip(recursive, gauged_prefixes, base_prefixes)
    ):
        expected = left_prefix.reshape(left_prefix.shape[0], -1) @ right_prefix.reshape(
            right_prefix.shape[0], -1
        ).T
        assert torch.allclose(actual, expected, atol=1e-11, rtol=1e-11), bond

    with pytest.raises(ValueError, match="not row-isometric"):
        canonical_prefix_overlaps(raw, base)


def test_prefix_transport_maps_gauge_equivalent_grams_and_projectors():
    raw = _random_tree(seed=830, layers=3)
    base = canonicalize_homogeneous(raw).network
    gauged_raw = raw
    for bond, dimension in enumerate(raw.bond_dims):
        gauged_raw = apply_bond_gauge(
            gauged_raw,
            bond,
            _well_conditioned_gauge(dimension, 831 + bond),
        )
    gauged = canonicalize_homogeneous(gauged_raw).network

    overlaps = canonical_prefix_overlaps(gauged, base)
    base_grams = canonical_environments(base)
    gauged_grams = canonical_environments(gauged)
    for bond, (overlap, base_gram, gauged_gram) in enumerate(
        zip(overlaps, base_grams, gauged_grams)
    ):
        transport = polar_orthogonal_transport(overlap)
        identity = torch.eye(transport.shape[0], dtype=DTYPE)
        assert torch.allclose(
            transport @ transport.T,
            identity,
            atol=1e-11,
            rtol=1e-11,
        )
        assert _relative_error(
            gauged_gram,
            transport @ base_gram @ transport.T,
        ) < 1e-9, bond
        certificate = projector_perturbation_certificate(
            base_gram,
            gauged_gram,
            transport,
            rank=1,
        )
        assert certificate.numerically_certifiable, bond
        assert certificate.passes_bound, bond
        assert certificate.normalized_projector_distance < 1e-8, bond


def test_prefix_transport_survives_exact_condition_1000_gauges_at_every_bond():
    raw = _random_tree(seed=830, layers=3)
    base = canonicalize_homogeneous(raw).network
    gauged_raw = raw
    generator = torch.Generator().manual_seed(2026090203)
    for bond, dimension in enumerate(raw.bond_dims):
        left, _ = torch.linalg.qr(
            torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
        )
        right, _ = torch.linalg.qr(
            torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
        )
        singular = torch.pow(
            10.0, torch.linspace(-1.5, 1.5, dimension, dtype=DTYPE)
        )
        gauged_raw = apply_bond_gauge(
            gauged_raw, bond, left @ torch.diag(singular) @ right.T
        )
    gauged = canonicalize_homogeneous(gauged_raw).network

    overlaps = canonical_prefix_overlaps(gauged, base)
    base_grams = canonical_environments(base)
    gauged_grams = canonical_environments(gauged)
    for bond, (overlap, base_gram, gauged_gram) in enumerate(
        zip(overlaps, base_grams, gauged_grams)
    ):
        transport = polar_orthogonal_transport(overlap)
        singular = torch.linalg.svdvals(overlap)
        assert float((singular - 1.0).abs().max()) <= 1e-8, bond
        assert torch.linalg.matrix_norm(
            transport.T @ transport - torch.eye(transport.shape[0], dtype=DTYPE),
            ord=2,
        ) <= 1e-11, bond
        assert (
            torch.linalg.matrix_norm(
                gauged_gram - transport @ base_gram @ transport.T, ord=2
            )
            / torch.linalg.matrix_norm(gauged_gram, ord=2)
        ) <= 1e-8, bond
        values, _ = sorted_eigensystem(base_gram)
        for rank in range(1, values.numel()):
            certificate = projector_perturbation_certificate(
                base_gram, gauged_gram, transport, rank
            )
            assert certificate.numerically_certifiable == (
                certificate.gap > 2.0 * certificate.eta
            ), (bond, rank)
            if certificate.numerically_certifiable:
                assert certificate.bound_with_roundoff is not None
                assert (
                    certificate.normalized_projector_distance
                    <= certificate.bound_with_roundoff
                ), (bond, rank)


@pytest.mark.parametrize(
    ("sine", "certifiable"),
    [(0.10, True), (0.49, True), (0.51, False)],
)
def test_projector_certificate_has_analytic_two_by_two_threshold(sine, certifiable):
    theta = torch.asin(torch.tensor(sine, dtype=DTYPE))
    rotation = torch.stack(
        (
            torch.stack((torch.cos(theta), -torch.sin(theta))),
            torch.stack((torch.sin(theta), torch.cos(theta))),
        )
    )
    base = torch.diag(torch.tensor([1.0, 0.0], dtype=DTYPE))
    candidate = rotation @ base @ rotation.T
    certificate = projector_perturbation_certificate(
        base,
        candidate,
        torch.eye(2, dtype=DTYPE),
        rank=1,
        roundoff_multiplier=0.0,
    )
    assert certificate.covariance_residual_operator == pytest.approx(sine, abs=1e-12)
    assert certificate.eta == pytest.approx(sine, abs=1e-12)
    assert certificate.normalized_projector_distance == pytest.approx(
        math.sqrt(2.0) * sine, abs=1e-12
    )
    assert certificate.numerically_certifiable is certifiable
    if certifiable:
        expected = math.sqrt(2.0) * sine / (1.0 - 2.0 * sine)
        assert certificate.davis_kahan_bound == pytest.approx(expected, abs=1e-11)
        assert certificate.passes_bound
        if sine == 0.49:
            assert certificate.davis_kahan_bound > math.sqrt(2.0)
    else:
        assert certificate.davis_kahan_bound is None
        assert certificate.bound_with_roundoff is None
        assert not certificate.passes_bound


def test_projector_certificate_is_scale_equivariant():
    sine = 0.10
    theta = torch.asin(torch.tensor(sine, dtype=DTYPE))
    rotation = torch.stack(
        (
            torch.stack((torch.cos(theta), -torch.sin(theta))),
            torch.stack((torch.sin(theta), torch.cos(theta))),
        )
    )
    base_unit = torch.diag(torch.tensor([1.0, 0.0], dtype=DTYPE))
    candidate_unit = rotation @ base_unit @ rotation.T
    certificates = [
        projector_perturbation_certificate(
            scale * base_unit,
            scale * candidate_unit,
            torch.eye(2, dtype=DTYPE),
            rank=1,
            roundoff_multiplier=0.0,
        )
        for scale in (1e-12, 1.0, 1e12)
    ]
    reference = certificates[1]
    for scale, certificate in zip((1e-12, 1.0, 1e12), certificates):
        assert certificate.numerically_certifiable == reference.numerically_certifiable
        assert certificate.passes_bound == reference.passes_bound
        assert certificate.eta / certificate.gap == pytest.approx(
            reference.eta / reference.gap, rel=1e-10, abs=1e-12
        )
        assert certificate.normalized_projector_distance == pytest.approx(
            reference.normalized_projector_distance, rel=1e-10, abs=1e-12
        )
        assert certificate.davis_kahan_bound == pytest.approx(
            reference.davis_kahan_bound, rel=1e-10, abs=1e-12
        )
        assert certificate.eta == pytest.approx(
            scale * reference.eta, rel=1e-10, abs=1e-25
        )
        assert certificate.gap == pytest.approx(
            scale * reference.gap, rel=1e-10, abs=1e-25
        )


def test_projector_certificate_rank_two_matches_analytic_rotation():
    theta = torch.tensor(0.01, dtype=DTYPE)
    crossing = torch.eye(4, dtype=DTYPE)
    crossing[1, 1] = torch.cos(theta)
    crossing[1, 2] = -torch.sin(theta)
    crossing[2, 1] = torch.sin(theta)
    crossing[2, 2] = torch.cos(theta)
    base = torch.diag(torch.tensor([9.0, 7.0, 3.0, 1.0], dtype=DTYPE))
    candidate = crossing @ base @ crossing.T
    certificate = projector_perturbation_certificate(
        base,
        candidate,
        torch.eye(4, dtype=DTYPE),
        rank=2,
        roundoff_multiplier=0.0,
    )
    sine = float(torch.sin(theta))
    assert certificate.covariance_residual_operator == pytest.approx(
        4.0 * sine, abs=1e-11
    )
    assert certificate.normalized_projector_distance == pytest.approx(sine, abs=1e-11)
    assert certificate.numerically_certifiable
    assert certificate.passes_bound
    assert certificate.davis_kahan_bound == pytest.approx(
        math.sqrt(2.0) * sine / (1.0 - 2.0 * sine), abs=1e-11
    )


def test_projector_certificate_alone_does_not_hide_prefix_span_mismatch():
    cosine = 0.5
    sine = math.sqrt(1.0 - cosine**2)
    base = HomogeneousChiTN(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE),
        (),
        torch.diag(torch.tensor([math.sqrt(3.0), 1.0], dtype=DTYPE)),
    )
    candidate = HomogeneousChiTN(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, cosine, sine]], dtype=DTYPE),
        (),
        torch.diag(torch.tensor([math.sqrt(3.0), 1.0], dtype=DTYPE)),
    )
    overlap = canonical_prefix_overlaps(candidate, base)[0]
    assert torch.allclose(overlap, torch.diag(torch.tensor([1.0, 0.5], dtype=DTYPE)))
    span_defect = torch.linalg.matrix_norm(
        torch.eye(2, dtype=DTYPE) - overlap.T @ overlap, ord=2
    )
    assert float(span_defect) == pytest.approx(0.75, abs=1e-12)
    transport = polar_orthogonal_transport(overlap)
    base_gram = canonical_environments(base)[0]
    candidate_gram = canonical_environments(candidate)[0]
    certificate = projector_perturbation_certificate(
        base_gram, candidate_gram, transport, rank=1, roundoff_multiplier=0.0
    )
    assert certificate.passes_bound
    assert certificate.normalized_projector_distance <= 1e-12
    assert float(coefficient_error_squared(base, candidate)) > 0.0


def test_supported_transport_accepts_rank_deficient_canonical_completion():
    base = HomogeneousChiTN(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE),
        (),
        torch.tensor([[1.0, 0.0]], dtype=DTYPE),
    )
    candidate = HomogeneousChiTN(
        torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=DTYPE),
        (),
        torch.tensor([[0.0, 1.0]], dtype=DTYPE),
    )
    overlap = canonical_prefix_overlaps(candidate, base)[0]
    transport = polar_orthogonal_transport(overlap)
    assert torch.allclose(
        transport @ torch.tensor([1.0, 0.0], dtype=DTYPE),
        torch.tensor([0.0, 1.0], dtype=DTYPE),
        atol=1e-12,
    )
    base_gram = canonical_environments(base)[0]
    candidate_gram = canonical_environments(candidate)[0]
    certificate = projector_perturbation_certificate(
        base_gram, candidate_gram, transport, rank=1, roundoff_multiplier=0.0
    )
    base_projector = torch.diag(torch.tensor([1.0, 0.0], dtype=DTYPE))
    candidate_projector = torch.diag(torch.tensor([0.0, 1.0], dtype=DTYPE))
    identity = torch.eye(2, dtype=DTYPE)
    defects = (
        torch.linalg.matrix_norm(
            base_projector @ (identity - overlap.T @ overlap) @ base_projector,
            ord=2,
        ),
        torch.linalg.matrix_norm(
            candidate_projector
            @ (identity - overlap @ overlap.T)
            @ candidate_projector,
            ord=2,
        ),
        torch.linalg.matrix_norm((transport - overlap) @ base_projector, ord=2),
    )
    assert max(float(defect) for defect in defects) <= 1e-12
    assert certificate.numerically_certifiable
    assert certificate.passes_bound
    assert certificate.normalized_projector_distance <= 1e-12
    assert float(coefficient_error_squared(base, candidate)) <= 1e-12


def test_projector_certificate_handles_resolved_and_uncertifiable_boundaries():
    angle = torch.tensor(0.37, dtype=DTYPE)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=DTYPE,
    )
    base = torch.diag(torch.tensor([5.0, 3.0, 1.0], dtype=DTYPE))
    perturbation = torch.tensor(
        [
            [2e-10, -1e-10, 0.0],
            [-1e-10, 0.0, 1e-10],
            [0.0, 1e-10, -1e-10],
        ],
        dtype=DTYPE,
    )
    candidate = rotation @ base @ rotation.T + perturbation
    resolved = projector_perturbation_certificate(
        base,
        candidate,
        rotation,
        rank=1,
    )
    assert resolved.numerically_certifiable
    assert resolved.passes_bound
    assert resolved.bound_with_roundoff is not None
    assert resolved.normalized_projector_distance <= resolved.bound_with_roundoff
    assert resolved.bound_with_roundoff < 1e-6

    clustered = torch.diag(torch.tensor([5.0, 3.0, 3.0 - 1e-12], dtype=DTYPE))
    noisy = clustered + torch.diag(
        torch.tensor([0.0, 2e-12, -2e-12], dtype=DTYPE)
    )
    uncertifiable = projector_perturbation_certificate(
        clustered,
        noisy,
        torch.eye(3, dtype=DTYPE),
        rank=2,
    )
    assert not uncertifiable.numerically_certifiable
    assert uncertifiable.davis_kahan_bound is None
    assert uncertifiable.bound_with_roundoff is None
    assert not uncertifiable.passes_bound


def test_projector_certificate_rejects_invalid_numeric_inputs_and_transport():
    gram = torch.diag(torch.tensor([3.0, 2.0, 1.0], dtype=DTYPE))
    nonorthogonal = torch.diag(torch.tensor([1.0, 1.0, 1.01], dtype=DTYPE))
    with pytest.raises(ValueError, match="transport must be orthogonal"):
        projector_perturbation_certificate(gram, gram, nonorthogonal, rank=1)
    nonsymmetric = gram.clone()
    nonsymmetric[0, 1] = 0.1
    with pytest.raises(ValueError, match="materially nonsymmetric"):
        projector_perturbation_certificate(
            nonsymmetric,
            gram,
            torch.eye(3, dtype=DTYPE),
            rank=1,
        )
    nonfinite = gram.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        projector_perturbation_certificate(
            nonfinite,
            gram,
            torch.eye(3, dtype=DTYPE),
            rank=1,
        )
    with pytest.raises(TypeError, match="rank must be an integer"):
        projector_perturbation_certificate(
            gram,
            gram,
            torch.eye(3, dtype=DTYPE),
            rank=True,
        )


def test_chimlp_export_canonicalize_environment_and_rotation_end_to_end():
    model = _biased_model()
    raw = export_homogeneous_network(model)
    canonical = canonicalize_homogeneous(raw)
    systems = odt_eigensystems(canonical.network)
    rotated = compress_homogeneous(
        canonical.network, top_bases(systems, canonical.network.bond_dims)
    )
    x = torch.randn(17, 6, dtype=DTYPE)
    expected = model(x)[0]
    assert torch.allclose(homogeneous_forward(raw, x), expected, atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        homogeneous_forward(canonical.network, x), expected, atol=1e-10, rtol=1e-10
    )
    assert torch.allclose(
        homogeneous_forward(rotated, x), expected, atol=1e-10, rtol=1e-10
    )
    assert _relative_error(
        materialize_tree_coefficient(rotated), materialize_tree_coefficient(raw)
    ) < 1e-10


def test_eigengap_diagnostics_block_unresolved_cluster_boundaries():
    values = torch.tensor([3.0, 2.0, 2.0, 0.5], dtype=DTYPE)
    vectors = torch.eye(4, dtype=DTYPE)
    diagnostic = eigengap_diagnostics(values, rank=2)
    assert not diagnostic.resolved
    with pytest.raises(ValueError, match="unresolved eigenvalue cluster"):
        top_bases(((values, vectors),), (2,), require_resolved_boundary=True)
    assert eigengap_diagnostics(values, rank=3).resolved


def test_tree_and_symmetric_polynomial_agree_on_diagonal_but_not_on_mode_gram():
    network = _random_tree(seed=9, layers=2)
    tree = materialize_tree_coefficient(network)
    leaf_axes = tuple(range(1, tree.ndim))
    symmetric = torch.zeros_like(tree)
    permutations = list(itertools.permutations(leaf_axes))
    for permutation in permutations:
        symmetric = symmetric + tree.permute((0,) + permutation)
    symmetric = symmetric / len(permutations)

    generator = torch.Generator().manual_seed(901)
    for _ in range(10):
        x = torch.randn(tree.shape[1], generator=generator, dtype=DTYPE)
        assert torch.allclose(
            _diagonal_tensor_forward(tree, x),
            _diagonal_tensor_forward(symmetric, x),
            atol=1e-10,
            rtol=1e-10,
        )

    tree_rows = tree.movedim(1, 0).reshape(tree.shape[1], -1)
    symmetric_rows = symmetric.movedim(1, 0).reshape(symmetric.shape[1], -1)
    tree_gram = tree_rows @ tree_rows.T
    symmetric_gram = symmetric_rows @ symmetric_rows.T
    assert _relative_error(tree_gram, symmetric_gram) > 1e-4
