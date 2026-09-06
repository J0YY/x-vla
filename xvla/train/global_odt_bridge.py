"""Exact tiny bridges from bilinear attention coefficients to global tree ODT.

The routines in this module deliberately separate three different objects:

* an ordinary mode Gram of one typed occurrence in an attention coefficient
  tensor,
* a sum of such Grams for a shared projector objective, and
* a canonical Dooms-style environment after the complete attention coefficient
  map is used as the embedding of a strict one-core self-bilinear tree.

Only the third object is global ODT, and only for the declared tiny tree.  Its
attention embedding represents one bias-free query-source contribution, not a
complete attention layer with a source sum, mask, normalization, or residual.
It clones that contribution on the two input legs of the action core, exactly
as the unfolded tree coefficient object does.  It is not an exact decomposition
of a residual transformer with shared activations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import prod
from typing import Mapping, Sequence

import torch

from xvla.nn.bilinear import BilinearFFN
from xvla.train.canonical_odt import HomogeneousChiTN
from xvla.train.vla_bilinear_atlas import symmetric_output_quadratics


Tensor = torch.Tensor
ATTENTION_LEGS = ("query_1", "query_2", "key_1", "key_2", "value")


def _validate_attention_core(core: Tensor) -> None:
    if core.ndim != 6:
        raise ValueError("attention core must have axes (output, query_1, query_2, key_1, key_2, value)")
    if not core.is_floating_point() or core.is_complex():
        raise TypeError("attention core must have a real floating-point dtype")
    if not bool(torch.isfinite(core).all()):
        raise ValueError("attention core must contain only finite values")
    if len(set(core.shape[1:])) != 1:
        raise ValueError("all five typed input legs must have the same dimension")


def _validate_output_metric(core: Tensor, output_metric: Tensor | None) -> Tensor:
    outputs = core.shape[0]
    if output_metric is None:
        return torch.eye(outputs, dtype=core.dtype, device=core.device)
    metric = torch.as_tensor(output_metric, dtype=core.dtype, device=core.device)
    if metric.shape != (outputs, outputs):
        raise ValueError(f"output_metric must have shape ({outputs}, {outputs})")
    if not bool(torch.isfinite(metric).all()):
        raise ValueError("output_metric must contain only finite values")
    symmetric = 0.5 * (metric + metric.T)
    scale = max(float(metric.abs().max().item()), 1.0)
    tolerance = 1000 * torch.finfo(metric.dtype).eps * outputs * scale
    if float((metric - metric.T).abs().max().item()) > tolerance:
        raise ValueError("output_metric must be symmetric")
    if float(torch.linalg.eigvalsh(symmetric)[0].item()) < -tolerance:
        raise ValueError("output_metric must be positive semidefinite")
    return symmetric


@torch.no_grad()
def coefficient_occurrence_gram(
    coefficient: Tensor,
    axis: int,
    *,
    output_metric: Tensor | None = None,
) -> Tensor:
    """Return an ordinary mode Gram for any coefficient input occurrence.

    Axis zero is reserved for the output.  Every other axis is an independent
    occurrence in the declared coefficient tensor.
    """

    if coefficient.ndim < 2:
        raise ValueError("coefficient must have an output axis and at least one input axis")
    if not coefficient.is_floating_point() or coefficient.is_complex():
        raise TypeError("coefficient must have a real floating-point dtype")
    if not bool(torch.isfinite(coefficient).all()):
        raise ValueError("coefficient must contain only finite values")
    if isinstance(axis, bool) or not isinstance(axis, int) or not 1 <= axis < coefficient.ndim:
        raise ValueError("axis must name a non-output coefficient occurrence")
    metric = _validate_output_metric(coefficient, output_metric)
    moved = coefficient.movedim(axis, 0)
    rows = moved.reshape(moved.shape[0], coefficient.shape[0], -1)
    gram = torch.einsum("aok,op,bpk->ab", rows, metric, rows)
    return 0.5 * (gram + gram.T)


@torch.no_grad()
def attention_occurrence_gram(
    core: Tensor,
    leg: str | int,
    *,
    output_metric: Tensor | None = None,
) -> Tensor:
    """Return the ordinary coefficient-matricization Gram of one typed leg.

    No input occurrence is traced or identified before squaring.  Therefore
    the trace tail of this Gram is exactly the squared coefficient error for
    projecting this occurrence alone while retaining every other occurrence.
    """

    _validate_attention_core(core)
    if isinstance(leg, str):
        if leg not in ATTENTION_LEGS:
            raise ValueError(f"unknown attention leg {leg!r}")
        leg_axis = 1 + ATTENTION_LEGS.index(leg)
    elif isinstance(leg, int) and not isinstance(leg, bool) and 0 <= leg < 5:
        leg_axis = 1 + leg
    else:
        raise ValueError("leg must be a typed leg name or an integer in [0, 4]")

    return coefficient_occurrence_gram(core, leg_axis, output_metric=output_metric)


@torch.no_grad()
def shared_attention_gram(
    core: Tensor,
    legs: Sequence[str | int],
    *,
    output_metric: Tensor | None = None,
) -> Tensor:
    """Sum occurrence Grams for a shared-projector, separate-error objective.

    This sum exactly ranks the sum of errors obtained by projecting each named
    occurrence separately.  It does not equal the generally nonlinear error
    from applying one projector simultaneously to all tied occurrences.
    """

    if not legs:
        raise ValueError("at least one attention leg is required")
    grams = [attention_occurrence_gram(core, leg, output_metric=output_metric) for leg in legs]
    return torch.stack(grams).sum(dim=0)


@torch.no_grad()
def trace_before_square_source_gram(core: Tensor) -> Tensor:
    """Reproduce the historical query-trace-first source Gram.

    This is a chosen reduced weight metric, not an ordinary coefficient mode
    Gram and not a Dooms ODT environment.  Query components can cancel in the
    trace before the square is taken.
    """

    _validate_attention_core(core)
    traced = torch.einsum("uppxyz->uxyz", core)
    gram = torch.einsum("uxyz,uvyz->xv", traced, traced)
    return 0.5 * (gram + gram.T)


@torch.no_grad()
def build_bilinear_attention_core(
    weights: Mapping[str, Tensor],
) -> Tensor:
    """Build the exact typed order-six core for one bias-free attention head.

    Required matrices are ``Wq1``, ``Wk1``, ``Wq2``, ``Wk2``, ``Wv``, and
    ``Wo``.  Every matrix has shape ``(d, d)``.  A leading homogeneous column
    is inserted, but no affine bias is included.  The result has shape
    ``(d, d+1, d+1, d+1, d+1, d+1)``.
    """

    required = ("Wq1", "Wk1", "Wq2", "Wk2", "Wv", "Wo")
    missing = [name for name in required if name not in weights]
    if missing:
        raise ValueError(f"missing attention weights: {', '.join(missing)}")
    matrices = [torch.as_tensor(weights[name]) for name in required]
    first = matrices[0]
    if not first.is_floating_point() or first.is_complex():
        raise TypeError("attention weights must have a real floating-point dtype")
    if first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("attention weights must be square matrices")
    dimension = first.shape[0]
    for name, matrix in zip(required, matrices):
        if matrix.shape != (dimension, dimension):
            raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
        if matrix.dtype != first.dtype or matrix.device != first.device:
            raise ValueError("all attention weights must share dtype and device")
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("attention weights must contain only finite values")

    def homogeneous(matrix: Tensor) -> Tensor:
        return torch.cat((matrix.new_zeros(dimension, 1), matrix), dim=1)

    hq1, hk1, hq2, hk2, hv, wo = [homogeneous(matrix) if index < 5 else matrix for index, matrix in enumerate(matrices)]
    m1 = hq1.T @ hk1
    m2 = hq2.T @ hk2
    typed = torch.einsum("pr,qs->pqrs", m1, m2)
    value_core = torch.einsum("pqrs,ut->upqrst", typed, hv)
    return torch.einsum("vu,upqrst->vpqrst", wo, value_core)


@torch.no_grad()
def symmetrize_tied_attention_core(core: Tensor) -> Tensor:
    """Collect typed roles into the tied query/source polynomial quotient.

    The two query occurrences are symmetrized with each other and the three
    source occurrences are symmetrized with each other.  Evaluation is
    unchanged when all query roles receive one vector and all source roles
    receive one vector, but occurrence Grams generally change because the
    coefficient object has changed.
    """

    _validate_attention_core(core)
    terms = [
        core.permute((0,) + query_order + source_order)
        for query_order in permutations((1, 2))
        for source_order in permutations((3, 4, 5))
    ]
    return torch.stack(terms).mean(dim=0)


@torch.no_grad()
def evaluate_attention_core(core: Tensor, query: Tensor, source: Tensor) -> Tensor:
    """Evaluate one query-source term of a typed attention core."""

    _validate_attention_core(core)
    dimension = core.shape[1] - 1
    if query.shape != (dimension,) or source.shape != (dimension,):
        raise ValueError(f"query and source must each have shape ({dimension},)")
    one = core.new_ones(1)
    q = torch.cat((one, query.to(dtype=core.dtype, device=core.device)))
    k = torch.cat((one, source.to(dtype=core.dtype, device=core.device)))
    return torch.einsum("upqrst,p,q,r,s,t->u", core, q, q, k, k, k)


@torch.no_grad()
def attention_coefficient_embedding(
    core: Tensor,
    *,
    scale: float = 1.0,
    include_tensor_unit: bool = True,
) -> Tensor:
    """Make the declared coefficient map from typed attention features to ``h``.

    With ``include_tensor_unit=True``, the first feature and first output are
    the tensor unit, giving ``1 plus attention features -> [1; h]``.  With it
    false, the map is the pure attention coefficient matrix.  In both cases
    there are no residual or duplicated route blocks in this v1 object.
    """

    _validate_attention_core(core)
    if not isinstance(scale, (int, float)) or not torch.isfinite(torch.tensor(float(scale))):
        raise ValueError("scale must be finite")
    outputs = core.shape[0]
    features = prod(core.shape[1:])
    if include_tensor_unit:
        embedding = core.new_zeros(outputs + 1, features + 1)
        embedding[0, 0] = 1.0
        embedding[1:, 1:] = float(scale) * core.reshape(outputs, features)
        return embedding
    return float(scale) * core.reshape(outputs, features)


@torch.no_grad()
def residual_ffn_action_core(
    module: BilinearFFN,
    action_weight: Tensor,
    action_bias: Tensor | None = None,
) -> Tensor:
    """Encode ``action_weight @ (h + module(h)) + action_bias`` as one core."""

    if module.out_dim != module.dim:
        raise ValueError("the residual FFN must preserve its input dimension")
    quadratics = symmetric_output_quadratics(module)
    dimension = module.dim
    residual = quadratics.new_zeros(dimension, dimension + 1, dimension + 1)
    coordinates = torch.arange(dimension, device=quadratics.device)
    residual[coordinates, 0, coordinates + 1] = 0.5
    residual[coordinates, coordinates + 1, 0] = 0.5

    weight = torch.as_tensor(action_weight, dtype=quadratics.dtype, device=quadratics.device)
    if weight.ndim != 2 or weight.shape[1] != dimension:
        raise ValueError(f"action_weight must have shape (actions, {dimension})")
    core = torch.einsum("ao,oij->aij", weight, quadratics + residual)
    if action_bias is not None:
        bias = torch.as_tensor(action_bias, dtype=core.dtype, device=core.device)
        if bias.shape != (weight.shape[0],):
            raise ValueError(f"action_bias must have shape ({weight.shape[0]},)")
        core[:, 0, 0] += bias
    return 0.5 * (core + core.transpose(1, 2))


@torch.no_grad()
def make_attention_action_tree(embedding: Tensor, action_core: Tensor) -> HomogeneousChiTN:
    """Create the exact one-core tree used by the global bridge oracle."""

    if embedding.ndim != 2:
        raise ValueError("embedding must be a matrix")
    if action_core.ndim != 3 or action_core.shape[1:] != (embedding.shape[0], embedding.shape[0]):
        raise ValueError("action_core input dimensions must match the embedding output")
    if not torch.allclose(action_core, action_core.transpose(1, 2)):
        raise ValueError("action_core must be symmetric in its tied input legs")
    head = torch.eye(action_core.shape[0], dtype=action_core.dtype, device=action_core.device)
    return HomogeneousChiTN(embedding, (action_core,), head)


@dataclass(frozen=True)
class OneOccurrenceProjection:
    squared_error: float
    gram_predicted_squared_error: float
    relative_difference: float


@torch.no_grad()
def verify_one_occurrence_tail(
    coefficient: Tensor,
    gram: Tensor,
    basis: Tensor,
    *,
    output_metric: Tensor | None = None,
) -> OneOccurrenceProjection:
    """Compare a Gram prediction with explicit projection of one coefficient leg.

    ``gram`` must have been constructed with the same ``output_metric``.  The
    basis may be any orthonormal subspace.  For a leading eigenspace, the Gram
    prediction is the usual discarded trace tail.
    """

    if coefficient.ndim != 3:
        raise ValueError("coefficient must have axes (output, left, right)")
    if gram.shape != (coefficient.shape[1], coefficient.shape[1]):
        raise ValueError("gram dimension must match the projected coefficient leg")
    if basis.ndim != 2 or basis.shape[0] != coefficient.shape[1]:
        raise ValueError("basis row dimension must match the projected coefficient leg")
    identity_basis = torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)
    basis_error = torch.linalg.matrix_norm(basis.T @ basis - identity_basis)
    tolerance = 1000 * torch.finfo(basis.dtype).eps * max(1, basis.shape[1])
    if float(basis_error.item()) > tolerance:
        raise ValueError("basis columns must be orthonormal")
    metric = _validate_output_metric(coefficient, output_metric)
    projector = basis @ basis.T
    projected = torch.einsum("oab,aA->oAb", coefficient, projector)
    difference_tensor = coefficient - projected
    squared_error = torch.einsum(
        "oab,op,pab->", difference_tensor, metric, difference_tensor
    )
    gram_symmetric = 0.5 * (gram + gram.T)
    predicted = torch.trace(gram_symmetric) - torch.trace(basis.T @ gram_symmetric @ basis)
    denominator = max(float(predicted.abs().item()), torch.finfo(coefficient.dtype).tiny)
    difference = abs(float(squared_error.item() - predicted.item())) / denominator
    return OneOccurrenceProjection(
        squared_error=float(squared_error.item()),
        gram_predicted_squared_error=float(predicted.item()),
        relative_difference=difference,
    )
