"""Canonical homogeneous ODT utilities for tree-structured chi MLPs.

This module implements the orthogonalize and diagonalize stages described for
tree chi-nets by Dooms et al.  It deliberately does not cover attention,
residual connections, or data-conditioned policy importance.

The represented object is the topology-specific unfolded tree coefficient
tensor.  Since every cloned leaf receives the same input during inference,
this tensor is not the unique fully symmetric polynomial quotient.  Callers
must not silently transfer conclusions between those two objects.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from xvla.models.chi_mlp import ChiMLP
from xvla.nn.normalization import RmsBatchNorm


Tensor = torch.Tensor


@dataclass(frozen=True)
class HomogeneousChiTN:
    """A complete affine chi-MLP as a homogeneous tree tensor network.

    ``embedding`` maps ``[1; x]`` to the first homogeneous hidden bond.
    Each core maps two copies of one homogeneous bond to the next bond.
    ``head`` maps the final homogeneous bond directly to logits.
    """

    embedding: Tensor
    cores: tuple[Tensor, ...]
    head: Tensor
    object_kind: str = "tree_topology"

    @property
    def n_layers(self) -> int:
        return len(self.cores)

    @property
    def bond_dims(self) -> tuple[int, ...]:
        return (self.embedding.shape[0],) + tuple(core.shape[0] for core in self.cores)


@dataclass(frozen=True)
class CanonicalODT:
    """Bottom-up isometric gauge, canonical up to orthogonal bond transforms.

    ``raw_from_canonical[i]`` maps the isometric coordinates at bond ``i``
    into that bond's coordinates before the ODT sweep. Individual coordinates
    remain ambiguous under signs and rotations within degenerate subspaces.
    """

    network: HomogeneousChiTN
    raw_from_canonical: tuple[Tensor, ...]
    isometry_errors: tuple[float, ...]
    symmetry_errors: tuple[float, ...]
    factorization_errors: tuple[float, ...]


@dataclass(frozen=True)
class EigengapDiagnostics:
    """Diagnostics for whether a retained eigenspace has a resolved boundary."""

    rank: int
    retained_value: float
    discarded_value: float
    absolute_gap: float
    relative_gap: float
    resolved: bool


@dataclass(frozen=True)
class ProjectorPerturbationCertificate:
    """A posteriori Davis-Kahan audit for a transported leading eigenspace.

    The recorded roundoff term is a declared numerical allowance.  It is not an
    interval-arithmetic proof of the floating-point reductions themselves.
    """

    rank: int
    gap: float
    covariance_residual_operator: float
    base_eigensolver_residual_operator: float
    candidate_eigensolver_residual_operator: float
    base_eigenvector_orthogonality: float
    candidate_eigenvector_orthogonality: float
    transport_orthogonality: float
    roundoff_allowance: float
    eta: float
    separation_margin: float
    numerically_certifiable: bool
    normalized_projector_distance: float
    davis_kahan_bound: float | None
    bound_with_roundoff: float | None
    passes_bound: bool


def _validate_network(network: HomogeneousChiTN) -> None:
    if network.object_kind != "tree_topology":
        raise ValueError(f"unsupported decomposition object {network.object_kind!r}")
    if network.embedding.ndim != 2:
        raise ValueError("embedding must be a matrix")
    previous = network.embedding.shape[0]
    for index, core in enumerate(network.cores):
        if core.ndim != 3:
            raise ValueError(f"core {index} must be order three")
        if core.shape[1:] != (previous, previous):
            raise ValueError(
                f"core {index} consumes {core.shape[1:]}, expected {(previous, previous)}"
            )
        previous = core.shape[0]
    if network.head.ndim != 2 or network.head.shape[1] != previous:
        raise ValueError("head input dimension does not match the final bond")


@torch.no_grad()
def export_homogeneous_network(
    model: ChiMLP, *, require_frozen_rbn: bool = False
) -> HomogeneousChiTN:
    """Export a chi-MLP, including every affine constant, as a homogeneous TN.

    ``require_frozen_rbn`` is the fail-closed production mode.  It rejects a
    running statistic that is uninitialized, still calibrating, or allowed to
    update.  The default remains available for historical diagnostic callers
    whose eval-mode running scalar is intentionally treated as fixed.
    """

    exported = copy.deepcopy(model).double().eval()
    dense_cores: list[Tensor] = []
    for index, (norm, layer) in enumerate(zip(exported.norms, exported.layers)):
        if isinstance(norm, RmsBatchNorm):
            if require_frozen_rbn and (
                not bool(norm.initialized)
                or not norm.frozen
                or norm.calibrating
            ):
                raise ValueError(
                    f"normalization at layer {index} is not initialized and frozen"
                )
            scale = norm.scale
            if not torch.isfinite(scale) or float(scale.item()) <= 0.0:
                raise ValueError(
                    f"normalization at layer {index} has an invalid fold scale"
                )
            layer.left.weight.div_(scale)
            layer.right.weight.div_(scale)
        elif not isinstance(norm, nn.Identity):
            raise ValueError(
                f"normalization at layer {index} is {type(norm).__name__}, which is not foldable"
            )

        core = layer.dense_core()
        core = 0.5 * (core + core.transpose(1, 2))
        input_dim = core.shape[1]
        homogeneous = core.new_zeros(core.shape[0] + 1, input_dim, input_dim)
        homogeneous[0, 0, 0] = 1.0
        homogeneous[1:] = core
        dense_cores.append(homogeneous)

    weight = exported.embed.weight.detach().clone()
    bias = exported.embed.bias.detach().clone()
    embedding = weight.new_zeros(weight.shape[0] + 1, weight.shape[1] + 1)
    embedding[0, 0] = 1.0
    embedding[1:, 0] = bias
    embedding[1:, 1:] = weight

    head = torch.cat(
        [exported.head.bias.detach().clone()[:, None], exported.head.weight.detach().clone()],
        dim=1,
    )
    network = HomogeneousChiTN(embedding, tuple(dense_cores), head)
    _validate_network(network)
    return network


def homogeneous_forward(network: HomogeneousChiTN, x: Tensor) -> Tensor:
    """Evaluate a homogeneous network, including physically compressed networks."""

    _validate_network(network)
    flat = x.flatten(1).to(dtype=network.embedding.dtype, device=network.embedding.device)
    one = torch.ones(flat.shape[0], 1, dtype=flat.dtype, device=flat.device)
    state = torch.cat([one, flat], dim=1) @ network.embedding.T
    for core in network.cores:
        state = torch.einsum("oij,bi,bj->bo", core, state, state)
    return state @ network.head.T


def reduced_rq_rows(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``R, Q`` with ``matrix = R @ Q`` and orthonormal rows in ``Q``.

    This is a reduced QR of the transpose even when ``R`` is singular. If the
    row count exceeds the column count, the intermediate bond shrinks to the
    column count. Numerical-rank truncation is deliberately not part of
    canonicalization.
    """

    if matrix.ndim != 2:
        raise ValueError("RQ input must be a matrix")

    q_columns, r_upper = torch.linalg.qr(matrix.T, mode="reduced")
    signs = torch.where(
        torch.diagonal(r_upper) < 0,
        -torch.ones((), dtype=matrix.dtype, device=matrix.device),
        torch.ones((), dtype=matrix.dtype, device=matrix.device),
    )
    q_columns = q_columns * signs[None, :]
    r_upper = signs[:, None] * r_upper
    return r_upper.T, q_columns.T


def _row_isometry_error(matrix: Tensor) -> float:
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    return float(torch.linalg.matrix_norm(matrix @ matrix.T - identity).item())


def _relative_symmetry_error(core: Tensor) -> float:
    numerator = torch.linalg.vector_norm(core - core.transpose(1, 2))
    denominator = torch.linalg.vector_norm(core).clamp_min(torch.finfo(core.dtype).tiny)
    return float((numerator / denominator).item())


def _require_symmetric_cores(network: HomogeneousChiTN) -> None:
    """Enforce the input-leg symmetry assumed throughout the chi-net ODT paper."""

    for index, core in enumerate(network.cores):
        error = _relative_symmetry_error(core)
        tolerance = min(
            1e-6,
            1000 * torch.finfo(core.dtype).eps * max(1, core.shape[1]),
        )
        if error > tolerance:
            raise ValueError(
                f"core {index} is not input-leg symmetric: relative error {error:.3e}"
            )


def _symmetric_core_coordinates(core: Tensor) -> Tensor:
    """Vectorize symmetric matrices isometrically using their upper triangles."""

    input_dim = core.shape[1]
    row, column = torch.triu_indices(input_dim, input_dim, device=core.device)
    coordinates = core[:, row, column].clone()
    off_diagonal = row != column
    coordinates[:, off_diagonal] *= math.sqrt(2.0)
    return coordinates


def _symmetric_core_from_coordinates(coordinates: Tensor, input_dim: int) -> Tensor:
    """Invert ``_symmetric_core_coordinates`` while preserving row inner products."""

    row, column = torch.triu_indices(input_dim, input_dim, device=coordinates.device)
    core = coordinates.new_zeros(coordinates.shape[0], input_dim, input_dim)
    values = coordinates.clone()
    off_diagonal = row != column
    values[:, off_diagonal] /= math.sqrt(2.0)
    core[:, row, column] = values
    core[:, column, row] = values
    return core


def _reduced_rq_symmetric_core(core: Tensor) -> tuple[Tensor, Tensor]:
    """RQ a symmetric core without completing a singular row space asymmetrically."""

    # Averaging removes roundoff-scale skew accepted by the validation tolerance.
    # It leaves C(x, x) unchanged because the antisymmetric part cancels exactly.
    symmetric = 0.5 * (core + core.transpose(1, 2))
    coordinates = _symmetric_core_coordinates(symmetric)
    factor, isometric_coordinates = reduced_rq_rows(coordinates)
    return factor, _symmetric_core_from_coordinates(
        isometric_coordinates, input_dim=core.shape[1]
    )


def _relative_factorization_error(actual: Tensor, factor: Tensor, isometry: Tensor) -> float:
    denominator = torch.linalg.vector_norm(actual).clamp_min(torch.finfo(actual.dtype).tiny)
    return float(
        (torch.linalg.vector_norm(actual - factor @ isometry) / denominator).item()
    )


def _require_canonical_network(
    network: HomogeneousChiTN, *, atol: float = 1e-9, rtol: float = 1e-8
) -> None:
    """Reject raw tensors for APIs whose recursion assumes an isometric gauge."""

    _validate_network(network)
    _require_symmetric_cores(network)
    factors = (network.embedding,) + tuple(core.reshape(core.shape[0], -1) for core in network.cores)
    for index, factor in enumerate(factors):
        error = _row_isometry_error(factor)
        scale = max(1.0, float(factor.shape[0]) ** 0.5)
        dtype_tolerance = min(
            1e-6,
            1000 * torch.finfo(factor.dtype).eps * max(1, factor.shape[0]),
        )
        if error > max(atol + rtol * scale, dtype_tolerance):
            raise ValueError(
                f"bond factor {index} is not row-isometric: error {error:.3e}"
            )


def _validate_output_metric(metric: Tensor, outputs: int) -> None:
    if metric.shape != (outputs, outputs):
        raise ValueError("output metric shape does not match the number of outputs")
    if not torch.isfinite(metric).all():
        raise ValueError("output metric must be finite")
    tolerance = 100 * torch.finfo(metric.dtype).eps * max(1, outputs)
    symmetry_error = torch.linalg.matrix_norm(metric - metric.T)
    scale = torch.linalg.matrix_norm(metric).clamp_min(torch.finfo(metric.dtype).tiny)
    if float((symmetry_error / scale).item()) > tolerance:
        raise ValueError("output metric must be symmetric")
    smallest = torch.linalg.eigvalsh(0.5 * (metric + metric.T))[0]
    if float(smallest.item()) < -tolerance * max(1.0, float(scale.item())):
        raise ValueError("output metric must be positive semidefinite")


def _tied_leg_gram(left: Tensor, right: Tensor, *, context: str) -> Tensor:
    """Validate that the two occurrences of a tied bond have one environment."""

    scale = torch.linalg.matrix_norm(left).clamp_min(torch.finfo(left.dtype).tiny)
    relative = torch.linalg.matrix_norm(left - right) / scale
    tolerance = 1000 * torch.finfo(left.dtype).eps * max(1, left.shape[0])
    if float(relative.item()) > tolerance:
        raise ValueError(
            f"{context} has unequal left and right tied-leg environments: "
            f"relative error {float(relative.item()):.3e}"
        )
    gram = 0.5 * (left + right)
    return 0.5 * (gram + gram.T)


@torch.no_grad()
def canonicalize_homogeneous(network: HomogeneousChiTN) -> CanonicalODT:
    """Perform bottom-up reduced RQ and push each ``R`` through both cloned legs."""

    _validate_network(network)
    _require_symmetric_cores(network)
    embedding = network.embedding.detach().clone()
    cores = [core.detach().clone() for core in network.cores]
    head = network.head.detach().clone()

    raw_from_canonical: list[Tensor] = []
    isometry_errors: list[float] = []
    symmetry_errors: list[float] = []
    factorization_errors: list[float] = []

    factor, embedding_iso = reduced_rq_rows(embedding)
    raw_from_canonical.append(factor)
    isometry_errors.append(_row_isometry_error(embedding_iso))
    factorization_errors.append(_relative_factorization_error(embedding, factor, embedding_iso))
    if cores:
        transformed = torch.einsum("oij,ia->oaj", cores[0], factor)
        cores[0] = torch.einsum("oaj,jb->oab", transformed, factor)
    else:
        head = head @ factor

    canonical_cores: list[Tensor] = []
    for index, core in enumerate(cores):
        symmetric_core = 0.5 * (core + core.transpose(1, 2))
        symmetric_rows = symmetric_core.reshape(symmetric_core.shape[0], -1)
        factor, isometric_core = _reduced_rq_symmetric_core(core)
        isometric_rows = isometric_core.reshape(isometric_core.shape[0], -1)
        canonical_cores.append(isometric_core)
        raw_from_canonical.append(factor)
        isometry_errors.append(_row_isometry_error(isometric_rows))
        symmetry_errors.append(_relative_symmetry_error(core))
        factorization_errors.append(
            _relative_factorization_error(symmetric_rows, factor, isometric_rows)
        )

        if index + 1 < len(cores):
            transformed = torch.einsum("oij,ia->oaj", cores[index + 1], factor)
            cores[index + 1] = torch.einsum("oaj,jb->oab", transformed, factor)
        else:
            head = head @ factor

    canonical = HomogeneousChiTN(embedding_iso, tuple(canonical_cores), head)
    _validate_network(canonical)
    _require_symmetric_cores(canonical)
    return CanonicalODT(
        network=canonical,
        raw_from_canonical=tuple(raw_from_canonical),
        isometry_errors=tuple(isometry_errors),
        symmetry_errors=tuple(symmetry_errors),
        factorization_errors=tuple(factorization_errors),
    )


def canonical_environments(
    network: HomogeneousChiTN, output_metric: Tensor | None = None
) -> tuple[Tensor, ...]:
    """Return the full downstream Gram for every homogeneous hidden bond.

    The input network must already be bottom-up row-isometric. Bond zero is
    the embedding output. Bond ``L`` is the final core output. A nonidentity
    PSD ``output_metric`` is an extension of the paper's Euclidean construction
    and induces a seminorm in which metric-null output directions are ignored.
    """

    _require_canonical_network(network)
    layer_count = network.n_layers
    grams: list[Tensor | None] = [None] * (layer_count + 1)
    if output_metric is None:
        output_metric = torch.eye(
            network.head.shape[0], dtype=network.head.dtype, device=network.head.device
        )
    _validate_output_metric(output_metric, network.head.shape[0])
    grams[layer_count] = network.head.T @ output_metric @ network.head

    for bond in range(layer_count - 1, -1, -1):
        core = network.cores[bond]
        downstream = grams[bond + 1]
        assert downstream is not None
        # Stage the contraction explicitly.  Each step is O(h^4) and uses an
        # order-three temporary.  A single three-operand einsum can otherwise
        # select an O(h^5) path at the width-257 paper configuration.
        transformed = torch.einsum("op,pAB->oAB", downstream, core)
        left = torch.einsum("oab,oAb->aA", core, transformed)
        right = torch.einsum("oab,oaB->bB", core, transformed)
        grams[bond] = _tied_leg_gram(left, right, context=f"bond {bond}")

    return tuple(gram for gram in grams if gram is not None)


def canonical_prefix_overlaps(
    left: HomogeneousChiTN, right: HomogeneousChiTN
) -> tuple[Tensor, ...]:
    """Contract row-function overlaps for every pair of canonical prefixes.

    The bond-``i`` matrix has entries ``<left_i[a], right_i[A]>`` in the
    independent-clone coefficient Hilbert space. For equivalent canonical
    trees with equal bond dimensions it is the orthogonal coordinate transport
    from the right prefix basis to the left prefix basis, up to roundoff.

    This contraction is functional. It does not recover the transport through
    the potentially ill-conditioned raw RQ factors.
    """

    _validate_network(left)
    _validate_network(right)
    _require_canonical_network(left)
    _require_canonical_network(right)
    if left.n_layers != right.n_layers:
        raise ValueError("coefficient trees must have the same depth")
    if left.embedding.shape[1] != right.embedding.shape[1]:
        raise ValueError("coefficient trees must have the same leaf dimension")
    if left.embedding.dtype != right.embedding.dtype:
        raise ValueError("coefficient trees must have the same dtype")
    if left.embedding.device != right.embedding.device:
        raise ValueError("coefficient trees must be on the same device")

    overlap = left.embedding @ right.embedding.T
    overlaps = [overlap]
    for left_core, right_core in zip(left.cores, right.cores):
        # Written as three contractions so the largest intermediate is order
        # three and each step has the O(h^4) scaling of the ODT recursion.
        transformed = torch.einsum("oij,iI->oIj", left_core, overlap)
        transformed = torch.einsum("oIj,jJ->oIJ", transformed, overlap)
        overlap = torch.einsum("oIJ,OIJ->oO", transformed, right_core)
        overlaps.append(overlap)
    return tuple(overlaps)


def polar_orthogonal_transport(overlap: Tensor) -> Tensor:
    """Return the closest orthogonal transport to a square prefix overlap."""

    if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError("prefix overlap must be square")
    left, _, right_adjoint = torch.linalg.svd(overlap, full_matrices=False)
    return left @ right_adjoint


def projector_perturbation_certificate(
    base_gram: Tensor,
    candidate_gram: Tensor,
    transport: Tensor,
    rank: int,
    *,
    roundoff_multiplier: float = 10_000.0,
) -> ProjectorPerturbationCertificate:
    """Audit transported eigenspaces with a gap-aware Davis-Kahan envelope.

    A fixed projector tolerance is not meaningful near an eigengap of the same
    scale. This routine separates exact structural identifiability from the
    finite-precision question. It includes covariance, eigensolver, and a
    declared dimension-scaled roundoff allowance in ``eta`` and requires
    ``gap > 2*eta`` before emitting a perturbation bound.  The allowance is an
    explicit reproducible robustness convention, not a formal forward-error
    bound for the floating-point eigensolver and norm kernels.
    """

    if base_gram.ndim != 2 or base_gram.shape[0] != base_gram.shape[1]:
        raise ValueError("base Gram must be square")
    if candidate_gram.shape != base_gram.shape:
        raise ValueError("candidate Gram shape must match base Gram")
    if transport.shape != base_gram.shape:
        raise ValueError("transport shape must match the Grams")
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("rank must be an integer")
    if not 1 <= rank < base_gram.shape[0]:
        raise ValueError("rank must be a nontrivial projector boundary")
    if not math.isfinite(roundoff_multiplier) or roundoff_multiplier < 0:
        raise ValueError("roundoff multiplier must be finite and nonnegative")
    if base_gram.dtype != candidate_gram.dtype or base_gram.dtype != transport.dtype:
        raise ValueError("Grams and transport must have the same dtype")
    if base_gram.device != candidate_gram.device or base_gram.device != transport.device:
        raise ValueError("Grams and transport must be on the same device")
    if not base_gram.is_floating_point():
        raise TypeError("Grams and transport must be floating point")
    if not (
        torch.isfinite(base_gram).all()
        and torch.isfinite(candidate_gram).all()
        and torch.isfinite(transport).all()
    ):
        raise ValueError("Grams and transport must be finite")

    dimension = base_gram.shape[0]
    epsilon = torch.finfo(base_gram.dtype).eps
    for name, gram in (("base", base_gram), ("candidate", candidate_gram)):
        scale = torch.linalg.matrix_norm(gram, ord=2).clamp_min(
            torch.finfo(gram.dtype).tiny
        )
        skew = torch.linalg.matrix_norm(gram - gram.T, ord=2) / scale
        if float(skew.item()) > 1000 * epsilon * max(1, dimension):
            raise ValueError(f"{name} Gram is materially nonsymmetric")

    identity = torch.eye(
        dimension,
        dtype=transport.dtype,
        device=transport.device,
    )
    transport_orthogonality = torch.linalg.matrix_norm(
        transport.T @ transport - identity,
        ord=2,
    )
    orthogonality_tolerance = 1000 * epsilon * max(1, dimension)
    if float(transport_orthogonality.item()) > orthogonality_tolerance:
        raise ValueError("transport must be orthogonal within the declared tolerance")

    base_symmetric = 0.5 * (base_gram + base_gram.T)
    candidate_symmetric = 0.5 * (candidate_gram + candidate_gram.T)
    base_values, base_vectors = sorted_eigensystem(base_symmetric)
    candidate_values, candidate_vectors = sorted_eigensystem(candidate_symmetric)
    predicted = transport @ base_symmetric @ transport.T
    covariance_residual = torch.linalg.matrix_norm(
        candidate_symmetric - predicted, ord=2
    )
    base_identity = torch.eye(
        base_vectors.shape[1], dtype=base_vectors.dtype, device=base_vectors.device
    )
    candidate_identity = torch.eye(
        candidate_vectors.shape[1],
        dtype=candidate_vectors.dtype,
        device=candidate_vectors.device,
    )
    base_vector_orthogonality = torch.linalg.matrix_norm(
        base_vectors.T @ base_vectors - base_identity,
        ord=2,
    )
    candidate_vector_orthogonality = torch.linalg.matrix_norm(
        candidate_vectors.T @ candidate_vectors - candidate_identity,
        ord=2,
    )
    base_reconstructed = (base_vectors * base_values.unsqueeze(0)) @ base_vectors.T
    candidate_reconstructed = (
        candidate_vectors * candidate_values.unsqueeze(0)
    ) @ candidate_vectors.T
    base_residual = torch.linalg.matrix_norm(
        base_symmetric - base_reconstructed,
        ord=2,
    )
    candidate_residual = torch.linalg.matrix_norm(
        candidate_symmetric - candidate_reconstructed,
        ord=2,
    )
    scale = torch.stack(
        (
            torch.linalg.matrix_norm(base_symmetric, ord=2),
            torch.linalg.matrix_norm(candidate_symmetric, ord=2),
            torch.linalg.matrix_norm(predicted, ord=2),
        )
    ).max().clamp_min(
        torch.finfo(predicted.dtype).tiny
    )
    roundoff = (
        roundoff_multiplier
        * torch.finfo(predicted.dtype).eps
        * dimension
        * scale
    )
    eta_tensor = (
        covariance_residual
        + base_residual
        + candidate_residual
        + scale
        * (
            transport_orthogonality
            + base_vector_orthogonality
            + candidate_vector_orthogonality
        )
        + roundoff
    )
    gap_tensor = base_values[rank - 1] - base_values[rank]
    separation_tensor = gap_tensor - 2.0 * eta_tensor
    certifiable = bool((gap_tensor > 0) & (separation_tensor > 0))

    base_projector = base_vectors[:, :rank] @ base_vectors[:, :rank].T
    candidate_projector = (
        candidate_vectors[:, :rank] @ candidate_vectors[:, :rank].T
    )
    predicted_projector = transport @ base_projector @ transport.T
    distance = torch.linalg.matrix_norm(
        candidate_projector - predicted_projector
    ) / math.sqrt(rank)

    bound = None
    bound_with_roundoff = None
    passes = False
    if certifiable:
        # Keep the conservative two-sided envelope uncapped.  Although the
        # geometric distance between equal-rank projectors is at most sqrt(2),
        # clipping to that value would hide a numerically vacuous certificate.
        bound_tensor = math.sqrt(2.0) * eta_tensor / separation_tensor
        arithmetic_slack = (
            roundoff_multiplier
            * torch.finfo(predicted.dtype).eps
            * dimension
        )
        bound = float(bound_tensor.item())
        bound_with_roundoff = float((bound_tensor + arithmetic_slack).item())
        passes = bool(distance <= bound_tensor + arithmetic_slack)

    return ProjectorPerturbationCertificate(
        rank=rank,
        gap=float(gap_tensor.item()),
        covariance_residual_operator=float(covariance_residual.item()),
        base_eigensolver_residual_operator=float(base_residual.item()),
        candidate_eigensolver_residual_operator=float(candidate_residual.item()),
        base_eigenvector_orthogonality=float(base_vector_orthogonality.item()),
        candidate_eigenvector_orthogonality=float(
            candidate_vector_orthogonality.item()
        ),
        transport_orthogonality=float(transport_orthogonality.item()),
        roundoff_allowance=float(roundoff.item()),
        eta=float(eta_tensor.item()),
        separation_margin=float(separation_tensor.item()),
        numerically_certifiable=certifiable,
        normalized_projector_distance=float(distance.item()),
        davis_kahan_bound=bound,
        bound_with_roundoff=bound_with_roundoff,
        passes_bound=passes,
    )


def local_environments(network: HomogeneousChiTN) -> tuple[Tensor, ...]:
    """Return adjacent-operator mode Grams in the same canonical bond gauges."""

    _require_canonical_network(network)
    layer_count = network.n_layers
    grams: list[Tensor] = []
    for bond in range(layer_count):
        core = network.cores[bond]
        left = torch.einsum("oab,oAb->aA", core, core)
        right = torch.einsum("oab,oaB->bB", core, core)
        grams.append(_tied_leg_gram(left, right, context=f"local bond {bond}"))
    root = network.head.T @ network.head
    grams.append(0.5 * (root + root.T))
    return tuple(grams)


def sorted_eigensystem(gram: Tensor) -> tuple[Tensor, Tensor]:
    """Return Gram eigenvalues and column eigenvectors in decreasing order."""

    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("Gram must be a square matrix")
    symmetric = 0.5 * (gram + gram.T)
    values, vectors = torch.linalg.eigh(symmetric)
    scale = values.abs().max().clamp_min(torch.finfo(values.dtype).tiny)
    tolerance = 1000 * torch.finfo(values.dtype).eps * max(1, values.numel()) * scale
    if float(values[0].item()) < -float(tolerance.item()):
        raise ValueError(
            f"Gram is materially indefinite: smallest eigenvalue {float(values[0]):.3e}"
        )
    values = values.clamp_min(0)
    order = torch.argsort(values, descending=True)
    return values[order], vectors[:, order]


def eigengap_diagnostics(
    eigenvalues: Tensor,
    rank: int,
    *,
    relative_tolerance: float = 1e-8,
) -> EigengapDiagnostics:
    """Describe the spectral boundary at ``rank``.

    A full-rank basis has no discarded neighbor and is therefore resolved.
    Individual eigenvectors inside an unresolved cluster are not identifiable.
    """

    if eigenvalues.ndim != 1:
        raise ValueError("eigenvalues must be a vector")
    if not 1 <= rank <= eigenvalues.numel():
        raise ValueError("rank is outside the eigenspectrum")
    retained = float(eigenvalues[rank - 1].item())
    if rank == eigenvalues.numel():
        discarded = 0.0
        absolute = math.inf
        relative = math.inf
        resolved = True
    else:
        discarded = float(eigenvalues[rank].item())
        absolute = retained - discarded
        global_scale = max(
            abs(float(eigenvalues[0].item())), torch.finfo(eigenvalues.dtype).tiny
        )
        numerical_tolerance = (
            1000 * torch.finfo(eigenvalues.dtype).eps * eigenvalues.numel() * global_scale
        )
        relative = absolute / global_scale
        resolved = absolute > numerical_tolerance and relative > relative_tolerance
    return EigengapDiagnostics(
        rank=rank,
        retained_value=retained,
        discarded_value=discarded,
        absolute_gap=absolute,
        relative_gap=relative,
        resolved=resolved,
    )


def odt_eigensystems(
    network: HomogeneousChiTN, output_metric: Tensor | None = None
) -> tuple[tuple[Tensor, Tensor], ...]:
    return tuple(
        sorted_eigensystem(gram)
        for gram in canonical_environments(network, output_metric=output_metric)
    )


def top_bases(
    eigensystems: Sequence[tuple[Tensor, Tensor]],
    ranks: Sequence[int],
    *,
    require_resolved_boundary: bool = False,
    relative_gap_tolerance: float = 1e-8,
) -> tuple[Tensor, ...]:
    if len(eigensystems) != len(ranks):
        raise ValueError("one rank is required for every bond")
    bases: list[Tensor] = []
    for bond, ((values, vectors), rank) in enumerate(zip(eigensystems, ranks)):
        if not 1 <= rank <= vectors.shape[1]:
            raise ValueError(f"rank {rank} is invalid for bond {bond}")
        diagnostic = eigengap_diagnostics(
            values, rank, relative_tolerance=relative_gap_tolerance
        )
        if require_resolved_boundary and not diagnostic.resolved:
            raise ValueError(
                f"rank {rank} cuts through an unresolved eigenvalue cluster at bond {bond}"
            )
        bases.append(vectors[:, :rank])
    return tuple(bases)


def compress_homogeneous(
    network: HomogeneousChiTN,
    bases: Sequence[Tensor | None] | Mapping[int, Tensor],
) -> HomogeneousChiTN:
    """Absorb orthonormal retained bond bases into adjacent tensors."""

    _validate_network(network)
    _require_symmetric_cores(network)
    dimensions = network.bond_dims
    if isinstance(bases, Mapping):
        selected = [bases.get(index) for index in range(len(dimensions))]
    else:
        if len(bases) != len(dimensions):
            raise ValueError("one basis or None is required for every bond")
        selected = list(bases)

    complete: list[Tensor] = []
    for bond, (dimension, basis) in enumerate(zip(dimensions, selected)):
        if basis is None:
            basis = torch.eye(
                dimension, dtype=network.embedding.dtype, device=network.embedding.device
            )
        if basis.ndim != 2 or basis.shape[0] != dimension:
            raise ValueError(f"basis {bond} has shape {tuple(basis.shape)}, expected ({dimension}, r)")
        identity = torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)
        error = torch.linalg.matrix_norm(basis.T @ basis - identity)
        tolerance = 1000 * torch.finfo(basis.dtype).eps * max(1, basis.shape[1])
        if float(error.item()) > tolerance:
            raise ValueError(
                f"basis {bond} columns are not orthonormal: error {float(error.item()):.3e}"
            )
        complete.append(basis)

    embedding = complete[0].T @ network.embedding
    cores: list[Tensor] = []
    for index, core in enumerate(network.cores):
        source = complete[index]
        target = complete[index + 1]
        transformed = torch.einsum("or,oab->rab", target, core)
        transformed = torch.einsum("rab,ai->rib", transformed, source)
        compressed = torch.einsum("rib,bj->rij", transformed, source)
        cores.append(compressed)
    head = network.head @ complete[-1]
    compressed_network = HomogeneousChiTN(embedding, tuple(cores), head)
    _validate_network(compressed_network)
    return compressed_network


def materialize_tree_coefficient(network: HomogeneousChiTN) -> Tensor:
    """Materialize the unfolded tree coefficient tensor for tiny test networks."""

    _validate_network(network)
    coefficient = network.embedding
    for core in network.cores:
        left = torch.tensordot(core, coefficient, dims=([1], [0]))
        coefficient = torch.tensordot(left, coefficient, dims=([1], [0]))
    return torch.tensordot(network.head, coefficient, dims=([1], [0]))


def prefix_coefficients(network: HomogeneousChiTN) -> tuple[Tensor, ...]:
    prefixes = [network.embedding]
    for core in network.cores:
        left = torch.tensordot(core, prefixes[-1], dims=([1], [0]))
        prefixes.append(torch.tensordot(left, prefixes[-1], dims=([1], [0])))
    return tuple(prefixes)


def explicit_downstream_tensor(network: HomogeneousChiTN, bond: int) -> Tensor:
    """Materialize one bond-to-output environment, including every sibling subtree."""

    _validate_network(network)
    if not 0 <= bond <= network.n_layers:
        raise ValueError(f"bond must be in [0, {network.n_layers}]")
    prefixes = prefix_coefficients(network)
    downstream = network.head

    for consumer in range(network.n_layers - 1, bond - 1, -1):
        environment_axes = downstream.ndim - 2
        expanded = torch.tensordot(downstream, network.cores[consumer], dims=([1], [0]))
        expanded = torch.tensordot(expanded, prefixes[consumer], dims=([-1], [0]))
        exposed_axis = 1 + environment_axes
        permutation = (
            [0, exposed_axis]
            + list(range(1, exposed_axis))
            + list(range(exposed_axis + 1, expanded.ndim))
        )
        downstream = expanded.permute(permutation)
    return downstream


def explicit_bond_gram(network: HomogeneousChiTN, bond: int) -> Tensor:
    """Brute-force a bond Gram from the explicitly materialized environment."""

    downstream = explicit_downstream_tensor(network, bond)
    rows = downstream.movedim(1, 0).reshape(downstream.shape[1], -1)
    return rows @ rows.T


def apply_bond_gauge(network: HomogeneousChiTN, bond: int, gauge: Tensor) -> HomogeneousChiTN:
    """Apply an exact hidden-coordinate gauge at one homogeneous bond."""

    _validate_network(network)
    dimensions = network.bond_dims
    if not 0 <= bond < len(dimensions):
        raise ValueError(f"bond must be in [0, {len(dimensions) - 1}]")
    if gauge.shape != (dimensions[bond], dimensions[bond]):
        raise ValueError("gauge shape does not match the selected bond")

    inverse = torch.linalg.inv(gauge)
    embedding = network.embedding.detach().clone()
    cores = [core.detach().clone() for core in network.cores]
    head = network.head.detach().clone()

    if bond == 0:
        embedding = gauge @ embedding
    else:
        cores[bond - 1] = torch.einsum("ro,oab->rab", gauge, cores[bond - 1])

    if bond == network.n_layers:
        head = head @ inverse
    else:
        transformed = torch.einsum("oij,ia->oaj", cores[bond], inverse)
        cores[bond] = torch.einsum("oaj,jb->oab", transformed, inverse)

    gauged = HomogeneousChiTN(embedding, tuple(cores), head)
    _validate_network(gauged)
    return gauged


def orthonormal_span(matrix: Tensor) -> Tensor:
    """Return an orthonormal basis for the columns of a full-column-rank matrix."""

    if matrix.ndim != 2:
        raise ValueError("span input must be a matrix")
    basis, _ = torch.linalg.qr(matrix, mode="reduced")
    return basis


def subspace_projector(matrix: Tensor) -> Tensor:
    basis = orthonormal_span(matrix)
    return basis @ basis.T


def discarded_trace(eigenvalues: Tensor, rank: int) -> Tensor:
    """Squared coefficient error for one unfolded bond occurrence."""

    if not 0 <= rank <= eigenvalues.numel():
        raise ValueError("rank is outside the eigenspectrum")
    return eigenvalues[rank:].clamp_min(0).sum()


def hierarchical_tail_bound_squared(
    eigensystems: Sequence[tuple[Tensor, Tensor]], ranks: Sequence[int]
) -> Tensor:
    """HSVD-style squared-error upper bound for tied, simultaneous truncation.

    Bond ``i`` occurs ``2 ** (L - i)`` times in the unfolded binary tree.
    The function returns the unnormalized sum of all discarded trace tails in
    the same output (semi)norm used to construct ``eigensystems``. This is an
    HSVD-style upper bound, not a globally optimal joint rank-tuple error.
    """

    if len(eigensystems) != len(ranks):
        raise ValueError("one rank is required for every bond")
    layer_count = len(eigensystems) - 1
    bound = eigensystems[0][0].new_zeros(())
    for bond, ((values, _), rank) in enumerate(zip(eigensystems, ranks)):
        multiplicity = 2 ** (layer_count - bond)
        bound = bound + multiplicity * discarded_trace(values, rank)
    return bound


@torch.no_grad()
def coefficient_inner_product(
    left: HomogeneousChiTN,
    right: HomogeneousChiTN,
    output_metric: Tensor | None = None,
) -> Tensor:
    """Contract the inner product of two topology-specific coefficient trees.

    The two networks may have different internal bond dimensions, as happens
    after physical truncation, but must have the same tree depth, leaf space,
    output space, dtype, and device.  The contraction keeps only one mixed
    overlap matrix per level.  It therefore does not materialize the
    exponentially large unfolded coefficient tensor.

    This is an inner product of the independent-clone tree tensors.  It is not
    an inner product of their fully symmetric polynomial quotients or of their
    outputs on a data distribution.
    """

    _validate_network(left)
    _validate_network(right)
    for name, network in (("left", left), ("right", right)):
        tensors = (network.embedding,) + network.cores + (network.head,)
        if any(tensor.dtype != network.embedding.dtype for tensor in tensors):
            raise ValueError(f"{name} coefficient tree mixes tensor dtypes")
        if any(tensor.device != network.embedding.device for tensor in tensors):
            raise ValueError(f"{name} coefficient tree mixes tensor devices")
    if left.n_layers != right.n_layers:
        raise ValueError("coefficient trees must have the same depth")
    if left.embedding.shape[1] != right.embedding.shape[1]:
        raise ValueError("coefficient trees must have the same leaf dimension")
    if left.head.shape[0] != right.head.shape[0]:
        raise ValueError("coefficient trees must have the same output dimension")
    if left.embedding.dtype != right.embedding.dtype:
        raise ValueError("coefficient trees must have the same dtype")
    if left.embedding.device != right.embedding.device:
        raise ValueError("coefficient trees must be on the same device")

    outputs = left.head.shape[0]
    if output_metric is None:
        output_metric = torch.eye(
            outputs, dtype=left.embedding.dtype, device=left.embedding.device
        )
    elif (
        output_metric.dtype != left.embedding.dtype
        or output_metric.device != left.embedding.device
    ):
        raise ValueError("output metric must match the coefficient-tree dtype and device")
    _validate_output_metric(output_metric, outputs)

    overlap = left.embedding @ right.embedding.T
    for left_core, right_core in zip(left.cores, right.cores):
        # Contract one child edge at a time.  A direct four-tensor einsum can
        # select an O(h^6) path, whereas these contractions require O(h^4)
        # arithmetic and O(h^3) temporary storage for equal bond widths.
        transformed = torch.einsum("oij,iI->oIj", left_core, overlap)
        transformed = torch.einsum("oIj,jJ->oIJ", transformed, overlap)
        overlap = torch.einsum("oIJ,OIJ->oO", transformed, right_core)

    return torch.einsum(
        "ui,uv,vj,ij->", left.head, output_metric, right.head, overlap
    )


@torch.no_grad()
def coefficient_error_squared(
    reference: HomogeneousChiTN,
    candidate: HomogeneousChiTN,
    output_metric: Tensor | None = None,
) -> Tensor:
    """Return ``||reference-candidate||^2`` without subtracting large norms.

    A direct ``<A,A> + <B,B> - 2<A,B>`` loses roughly half the available
    digits when ``A`` and ``B`` are nearly equal.  That is unacceptable for
    full-rank reconstruction certificates.  Instead, form one block-sparse
    tree for ``A-B``, bottom-up orthogonalize it, and measure the remaining
    canonical head directly.  Rank deficiency caused by identical branches is
    exposed during the RQ sweep before the residual is squared.
    """

    _validate_network(reference)
    _validate_network(candidate)
    _require_symmetric_cores(reference)
    _require_symmetric_cores(candidate)
    for name, network in (("reference", reference), ("candidate", candidate)):
        tensors = (network.embedding,) + network.cores + (network.head,)
        if any(tensor.dtype != network.embedding.dtype for tensor in tensors):
            raise ValueError(f"{name} coefficient tree mixes tensor dtypes")
        if any(tensor.device != network.embedding.device for tensor in tensors):
            raise ValueError(f"{name} coefficient tree mixes tensor devices")
    if reference.n_layers != candidate.n_layers:
        raise ValueError("coefficient trees must have the same depth")
    if reference.embedding.shape[1] != candidate.embedding.shape[1]:
        raise ValueError("coefficient trees must have the same leaf dimension")
    if reference.head.shape[0] != candidate.head.shape[0]:
        raise ValueError("coefficient trees must have the same output dimension")
    if reference.embedding.dtype != candidate.embedding.dtype:
        raise ValueError("coefficient trees must have the same dtype")
    if reference.embedding.device != candidate.embedding.device:
        raise ValueError("coefficient trees must be on the same device")

    outputs = reference.head.shape[0]
    if output_metric is None:
        output_metric = torch.eye(
            outputs,
            dtype=reference.embedding.dtype,
            device=reference.embedding.device,
        )
    elif (
        output_metric.dtype != reference.embedding.dtype
        or output_metric.device != reference.embedding.device
    ):
        raise ValueError("output metric must match the coefficient-tree dtype and device")
    _validate_output_metric(output_metric, outputs)

    embedding = torch.cat((reference.embedding, candidate.embedding), dim=0)
    cores: list[Tensor] = []
    reference_source = reference.embedding.shape[0]
    candidate_source = candidate.embedding.shape[0]
    for reference_core, candidate_core in zip(reference.cores, candidate.cores):
        reference_target = reference_core.shape[0]
        candidate_target = candidate_core.shape[0]
        core = reference_core.new_zeros(
            reference_target + candidate_target,
            reference_source + candidate_source,
            reference_source + candidate_source,
        )
        core[:reference_target, :reference_source, :reference_source] = reference_core
        core[
            reference_target:,
            reference_source:,
            reference_source:,
        ] = candidate_core
        cores.append(core)
        reference_source = reference_target
        candidate_source = candidate_target
    head = torch.cat((reference.head, -candidate.head), dim=1)
    difference = HomogeneousChiTN(embedding, tuple(cores), head)
    canonical_difference = canonicalize_homogeneous(difference).network

    metric_values, metric_vectors = torch.linalg.eigh(
        0.5 * (output_metric + output_metric.T)
    )
    weighted_head = metric_values.clamp_min(0).sqrt()[:, None] * (
        metric_vectors.T @ canonical_difference.head
    )
    return weighted_head.square().sum()
