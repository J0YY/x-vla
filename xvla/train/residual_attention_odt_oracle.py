"""Dense canonical ODT oracle for one residual VLA attention block.

This module is deliberately a tiny, exact oracle.  It compiles the real
``ChiVLA`` patch embedding, token embeddings, state projection, one causal
joint bilinear-attention residual plus bilinear-FFN residual per backbone
block, and the linear action head into a mixed-arity homogeneous tensor chain.
Each attention core has arity five and each FFN core has arity two.  Residuals
are represented in the cores rather than removed.

The canonical sweep is the Dooms bottom-up reduced-RQ construction generalized
to a symmetric core of arbitrary arity.  Top-down environments are computed
with ``generalized_metric_odt.role_environment``.  Dense materialization is
only intended for very small dimensions.

Claim boundary: this is canonical polynomial ODT for identity-normalized or
exactly folded, frozen scalar-RBN models.  Padé ``RationalNorm`` checkpoints
remain exact rational tensor networks, but are not silently converted into a
polynomial chain.  ``replay_rational_residual_block`` exercises the existing
projective-pair arithmetic exactly at each deployed normalization site.  A
global rational canonical sweep still needs a compositional pair-valued core
interface and a quotient-aware bottom-up gauge.
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass
from string import ascii_lowercase

import torch
import torch.nn as nn
import torch.nn.functional as F

from xvla.models.vla import ChiVLA
from xvla.nn.attention import BilinearAttention, causal_mask
from xvla.nn.bilinear import BilinearFFN
from xvla.nn.normalization import RationalNorm, RmsBatchNorm
from xvla.train.canonical_odt import reduced_rq_rows
from xvla.train.complete_attention_odt import compile_bilinear_attention
from xvla.train.fold_vla import fold_vla
from xvla.train.generalized_metric_odt import role_environment
from xvla.train.projective_quotient import (
    DenominatorLedgerEntry,
    projective_value,
    rational_ffn_residual_projective,
    rational_norm_projective,
    stack_projective_rows,
    tiny_rational_attention_projective_oracle,
)
from xvla.train.residual_route_odt import compile_bilinear_ffn_residual_core


Tensor = torch.Tensor
POLYNOMIAL_OBJECT_KIND = "symmetric_mixed_arity_whole_state_polynomial_tree"
RATIONAL_REPLAY_OBJECT_KIND = "staged_projective_pair_residual_block_replay_noncanonical"
POLYNOMIAL_CLAIM_BOUNDARY = (
    "Canonical weight-only ODT for the tiny identity/scalar-folded polynomial "
    "VLA subclass. It is not a canonical decomposition of a Padé RationalNorm checkpoint."
)
RATIONAL_INTERFACE_GAP = (
    "Global Padé ODT needs pair-valued multilinear cores that accept ProjectivePair inputs, "
    "retain denominator products across block boundaries, and expose quotient-aware RQ and "
    "top-down environments modulo the projective radial gauge."
)


@dataclass(frozen=True)
class MixedArityHomogeneousTN:
    """Homogeneous chain whose core input occurrences are tied at inference."""

    embedding: Tensor
    cores: tuple[Tensor, ...]
    head: Tensor
    object_kind: str = POLYNOMIAL_OBJECT_KIND

    @property
    def arities(self) -> tuple[int, ...]:
        return tuple(core.ndim - 1 for core in self.cores)

    @property
    def bond_dims(self) -> tuple[int, ...]:
        return (self.embedding.shape[0],) + tuple(core.shape[0] for core in self.cores)


@dataclass(frozen=True)
class CanonicalMixedArityODT:
    network: MixedArityHomogeneousTN
    raw_from_canonical: tuple[Tensor, ...]
    isometry_errors: tuple[float, ...]
    symmetry_errors: tuple[float, ...]
    factorization_errors: tuple[float, ...]


@dataclass(frozen=True)
class TinyVLAResidualOracle:
    """Compiled network plus the exact polynomial model copy it represents."""

    network: MixedArityHomogeneousTN
    polynomial_model: ChiVLA
    raw_input_dimension: int
    token_count: int
    source_normalization: str
    claim_boundary: str = POLYNOMIAL_CLAIM_BOUNDARY


@dataclass(frozen=True)
class RationalResidualBlockReplay:
    value: Tensor
    denominator_ledger: tuple[DenominatorLedgerEntry, ...]
    object_kind: str = RATIONAL_REPLAY_OBJECT_KIND
    interface_gap: str = RATIONAL_INTERFACE_GAP


def _validate_network(network: MixedArityHomogeneousTN) -> None:
    if network.object_kind != POLYNOMIAL_OBJECT_KIND:
        raise ValueError(f"unsupported decomposition object {network.object_kind!r}")
    if network.embedding.ndim != 2:
        raise ValueError("embedding must be a matrix")
    if not network.embedding.is_floating_point() or network.embedding.is_complex():
        raise TypeError("embedding must have a real floating dtype")
    previous = network.embedding.shape[0]
    for index, core in enumerate(network.cores):
        if core.ndim < 2:
            raise ValueError(f"core {index} must have arity at least one")
        if any(dimension != previous for dimension in core.shape[1:]):
            raise ValueError(f"core {index} must consume tied copies of bond {previous}")
        if core.dtype != network.embedding.dtype or core.device != network.embedding.device:
            raise ValueError("all network tensors must share dtype and device")
        previous = core.shape[0]
    if network.head.ndim != 2 or network.head.shape[1] != previous:
        raise ValueError("head input dimension does not match the final bond")
    if network.head.dtype != network.embedding.dtype or network.head.device != network.embedding.device:
        raise ValueError("all network tensors must share dtype and device")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in (network.embedding, *network.cores, network.head)):
        raise ValueError("network tensors must be finite")


def _symmetry_error(core: Tensor) -> float:
    denominator = torch.linalg.vector_norm(core).clamp_min(torch.finfo(core.dtype).tiny)
    errors = []
    for role in range(core.ndim - 2):
        axes = list(range(core.ndim))
        axes[role + 1], axes[role + 2] = axes[role + 2], axes[role + 1]
        errors.append(torch.linalg.vector_norm(core - core.permute(*axes)) / denominator)
    return max((float(error.item()) for error in errors), default=0.0)


def _require_symmetric(network: MixedArityHomogeneousTN) -> None:
    for index, core in enumerate(network.cores):
        error = _symmetry_error(core)
        tolerance = min(1e-6, 5000 * torch.finfo(core.dtype).eps * max(core.shape))
        if error > tolerance:
            raise ValueError(f"core {index} is not permutation-symmetric: {error:.3e}")


def _unique_permutations(values: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(set(itertools.permutations(values)))


def _symmetric_coordinates(core: Tensor) -> tuple[Tensor, tuple[tuple[int, ...], ...]]:
    """Isometric coordinates for a fully symmetric arbitrary-arity core."""

    arity = core.ndim - 1
    representatives = tuple(itertools.combinations_with_replacement(range(core.shape[1]), arity))
    columns = []
    for representative in representatives:
        multiplicity = len(_unique_permutations(representative))
        columns.append(core[(slice(None), *representative)] * math.sqrt(multiplicity))
    return torch.stack(columns, dim=1), representatives


def _core_from_symmetric_coordinates(
    coordinates: Tensor,
    representatives: tuple[tuple[int, ...], ...],
    input_dimension: int,
) -> Tensor:
    arity = len(representatives[0])
    core = coordinates.new_zeros((coordinates.shape[0],) + (input_dimension,) * arity)
    for column, representative in enumerate(representatives):
        permutations = _unique_permutations(representative)
        value = coordinates[:, column] / math.sqrt(len(permutations))
        for permutation in permutations:
            core[(slice(None), *permutation)] = value
    return core


def _row_isometry_error(matrix: Tensor) -> float:
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    return float(torch.linalg.matrix_norm(matrix @ matrix.T - identity).item())


def _relative_error(actual: Tensor, expected: Tensor) -> float:
    denominator = torch.linalg.vector_norm(actual).clamp_min(torch.finfo(actual.dtype).tiny)
    return float((torch.linalg.vector_norm(actual - expected) / denominator).item())


def _supported_reduced_rq_rows(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Use Dooms reduced RQ, shrinking only an algebraically unsupported row space.

    Whole-state VLA assembly has structural dependencies because BOS and action
    query coordinates are constants.  A full-width QR would invent arbitrary
    completion vectors for those zero singular directions and would destroy
    nonorthogonal-gauge invariance.  Full-row-rank factors use the repository's
    canonical reduced RQ unchanged.  Rank-deficient factors use the supported
    compact SVD, which is an equally valid row-isometric gauge on the exact
    coefficient support.
    """

    singular_values = torch.linalg.svdvals(matrix)
    scale = float(singular_values[0].item()) if singular_values.numel() else 0.0
    tolerance = torch.finfo(matrix.dtype).eps * max(matrix.shape) * scale
    rank = int((singular_values > tolerance).sum().item())
    if rank == matrix.shape[0]:
        return reduced_rq_rows(matrix)
    if rank == 0:
        raise ValueError("canonical factor has zero supported row rank")
    left, values, right = torch.linalg.svd(matrix, full_matrices=False)
    return left[:, :rank] * values[:rank][None, :], right[:rank]


def _apply_input_factor(core: Tensor, factor: Tensor) -> Tensor:
    """Substitute ``old_state = factor @ new_state`` in every input role."""

    result = core
    for axis in range(1, core.ndim):
        moved = result.movedim(axis, -1)
        moved = moved @ factor
        result = moved.movedim(-1, axis)
    return result


def _apply_output_factor(core: Tensor, factor: Tensor) -> Tensor:
    return torch.tensordot(factor, core, dims=([1], [0]))


@torch.no_grad()
def mixed_arity_forward(network: MixedArityHomogeneousTN, raw_input: Tensor) -> Tensor:
    _validate_network(network)
    if raw_input.ndim != 2 or raw_input.shape[1] + 1 != network.embedding.shape[1]:
        raise ValueError("raw_input width does not match the homogeneous embedding")
    state = torch.cat((raw_input.new_ones(raw_input.shape[0], 1), raw_input), dim=1)
    state = state @ network.embedding.T
    role_labels = "".join(label for label in ascii_lowercase if label not in {"b", "o"})
    for core in network.cores:
        arity = core.ndim - 1
        labels = role_labels[:arity]
        equation = f"o{labels}," + ",".join(f"b{label}" for label in labels) + "->bo"
        state = torch.einsum(equation, core, *([state] * arity))
    return state @ network.head.T


@torch.no_grad()
def canonicalize_mixed_arity(network: MixedArityHomogeneousTN) -> CanonicalMixedArityODT:
    """Dooms reduced-RQ sweep with each factor pushed through every tied leg."""

    _validate_network(network)
    _require_symmetric(network)
    embedding = network.embedding.detach().clone()
    cores = [core.detach().clone() for core in network.cores]
    head = network.head.detach().clone()
    raw_from_canonical = []
    isometry_errors = []
    symmetry_errors = []
    factorization_errors = []

    factor, embedding_iso = _supported_reduced_rq_rows(embedding)
    raw_from_canonical.append(factor)
    isometry_errors.append(_row_isometry_error(embedding_iso))
    factorization_errors.append(_relative_error(embedding, factor @ embedding_iso))
    if cores:
        cores[0] = _apply_input_factor(cores[0], factor)
    else:
        head = head @ factor

    canonical_cores = []
    for index, core in enumerate(cores):
        symmetry_errors.append(_symmetry_error(core))
        coordinates, representatives = _symmetric_coordinates(core)
        factor, iso_coordinates = _supported_reduced_rq_rows(coordinates)
        iso_core = _core_from_symmetric_coordinates(
            iso_coordinates, representatives, core.shape[1]
        )
        canonical_cores.append(iso_core)
        raw_from_canonical.append(factor)
        isometry_errors.append(_row_isometry_error(iso_core.reshape(iso_core.shape[0], -1)))
        factorization_errors.append(
            _relative_error(core, _apply_output_factor(iso_core, factor))
        )
        if index + 1 < len(cores):
            cores[index + 1] = _apply_input_factor(cores[index + 1], factor)
        else:
            head = head @ factor

    canonical_network = MixedArityHomogeneousTN(embedding_iso, tuple(canonical_cores), head)
    _validate_network(canonical_network)
    _require_symmetric(canonical_network)
    return CanonicalMixedArityODT(
        canonical_network,
        tuple(raw_from_canonical),
        tuple(isometry_errors),
        tuple(symmetry_errors),
        tuple(factorization_errors),
    )


@torch.no_grad()
def canonical_environments_mixed_arity(
    network: MixedArityHomogeneousTN,
    output_metric: Tensor | None = None,
) -> tuple[Tensor, ...]:
    """Return one full top-down Gram per hidden bond in canonical gauge."""

    _validate_network(network)
    _require_symmetric(network)
    factors = (network.embedding,) + tuple(core.reshape(core.shape[0], -1) for core in network.cores)
    for index, factor in enumerate(factors):
        if _row_isometry_error(factor) > 2e-8 * max(1.0, math.sqrt(factor.shape[0])):
            raise ValueError(f"bond factor {index} is not row-isometric")
    if output_metric is None:
        output_metric = torch.eye(network.head.shape[0], dtype=network.head.dtype, device=network.head.device)
    if output_metric.shape != (network.head.shape[0], network.head.shape[0]):
        raise ValueError("output metric shape mismatch")
    grams: list[Tensor | None] = [None] * (len(network.cores) + 1)
    grams[-1] = network.head.T @ output_metric @ network.head
    for bond in range(len(network.cores) - 1, -1, -1):
        core = network.cores[bond]
        downstream = grams[bond + 1]
        assert downstream is not None
        identity = torch.eye(core.shape[1], dtype=core.dtype, device=core.device)
        roles = tuple(
            role_environment(core, downstream, (identity,) * (core.ndim - 1), role)
            for role in range(core.ndim - 1)
        )
        scale = torch.linalg.matrix_norm(roles[0]).clamp_min(torch.finfo(core.dtype).tiny)
        disagreement = max(
            float((torch.linalg.matrix_norm(role - roles[0]) / scale).item())
            for role in roles
        )
        if disagreement > min(1e-6, 5000 * torch.finfo(core.dtype).eps * core.shape[1]):
            raise ValueError(f"bond {bond} tied-role environments disagree: {disagreement:.3e}")
        grams[bond] = 0.5 * (sum(roles) / len(roles) + (sum(roles) / len(roles)).T)
    return tuple(gram for gram in grams if gram is not None)


@torch.no_grad()
def gauge_bond(
    network: MixedArityHomogeneousTN,
    bond: int,
    transform: Tensor,
) -> MixedArityHomogeneousTN:
    """Apply a nonsingular hidden-coordinate gauge and its exact inverse downstream."""

    _validate_network(network)
    if not 0 <= bond <= len(network.cores):
        raise ValueError("bond index out of range")
    dimension = network.bond_dims[bond]
    if transform.shape != (dimension, dimension):
        raise ValueError("gauge transform shape mismatch")
    if transform.dtype != network.embedding.dtype or transform.device != network.embedding.device:
        raise ValueError("gauge transform dtype/device mismatch")
    inverse = torch.linalg.inv(transform)
    embedding = network.embedding.detach().clone()
    cores = [core.detach().clone() for core in network.cores]
    head = network.head.detach().clone()
    if bond == 0:
        embedding = transform @ embedding
    else:
        cores[bond - 1] = _apply_output_factor(cores[bond - 1], transform)
    if bond < len(cores):
        cores[bond] = _apply_input_factor(cores[bond], inverse)
    else:
        head = head @ inverse
    result = MixedArityHomogeneousTN(embedding, tuple(cores), head)
    _validate_network(result)
    _require_symmetric(result)
    return result


@torch.no_grad()
def one_leg_gauge_negative_control(
    network: MixedArityHomogeneousTN,
    bond: int,
    transform: Tensor,
    *,
    role: int = 0,
) -> MixedArityHomogeneousTN:
    """Incorrectly push an inverse gauge into only one cloned downstream leg.

    This is an adversarial control for the load-bearing Dooms rule: a factor
    emitted at a shared bond must be pushed through every occurrence of that
    bond.  The returned network is intentionally asymmetric and should not
    replay the original function for a nontrivial transform.
    """

    _validate_network(network)
    if not 0 <= bond < len(network.cores):
        raise ValueError("negative control needs a bond followed by a core")
    dimension = network.bond_dims[bond]
    if transform.shape != (dimension, dimension):
        raise ValueError("gauge transform shape mismatch")
    if transform.dtype != network.embedding.dtype or transform.device != network.embedding.device:
        raise ValueError("gauge transform dtype/device mismatch")
    next_core = network.cores[bond]
    if not 0 <= role < next_core.ndim - 1:
        raise ValueError("role index out of range")
    inverse = torch.linalg.inv(transform)
    embedding = network.embedding.detach().clone()
    cores = [core.detach().clone() for core in network.cores]
    head = network.head.detach().clone()
    if bond == 0:
        embedding = transform @ embedding
    else:
        cores[bond - 1] = _apply_output_factor(cores[bond - 1], transform)
    axis = role + 1
    moved = cores[bond].movedim(axis, -1) @ inverse
    cores[bond] = moved.movedim(-1, axis)
    result = MixedArityHomogeneousTN(embedding, tuple(cores), head)
    _validate_network(result)
    return result


def _global_index(token: int, local_index: int, dimension: int) -> int:
    return 0 if local_index == 0 else 1 + token * dimension + local_index - 1


def _symmetrize_inputs(core: Tensor) -> Tensor:
    permutations = tuple(itertools.permutations(range(core.ndim - 1)))
    result = torch.zeros_like(core)
    for permutation in permutations:
        result += core.permute(0, *(axis + 1 for axis in permutation))
    return result / len(permutations)


@torch.no_grad()
def _compile_whole_state_attention_residual(
    block,
    token_count: int,
    maximum_elements: int,
    *,
    active_tokens: tuple[int, ...] | None = None,
) -> Tensor:
    if not isinstance(block.attn, BilinearAttention) or block.attn.qk_norm != "none":
        raise ValueError("polynomial oracle requires a folded BilinearAttention with qk_norm='none'")
    if not isinstance(block.rbn_attn, nn.Identity):
        raise ValueError("attention pre-norm must be identity after exact scalar folding")
    dimension = block.attn.dim
    active_tokens = tuple(range(token_count)) if active_tokens is None else tuple(active_tokens)
    if not active_tokens or len(set(active_tokens)) != len(active_tokens):
        raise ValueError("active_tokens must contain distinct workspace token indices")
    if any(token < 0 or token >= token_count for token in active_tokens):
        raise ValueError("active token index is outside the workspace")
    homogeneous_dimension = 1 + token_count * dimension
    elements = homogeneous_dimension ** 6
    if elements > maximum_elements:
        raise ValueError(
            f"whole-state attention core needs {elements} elements, above tiny-oracle limit {maximum_elements}"
        )
    active_count = len(active_tokens)
    mask = (
        causal_mask(active_count, dtype=block.attn.wo.weight.dtype, device=block.attn.wo.weight.device)
        if block.attn.causal
        else block.attn.wo.weight.new_ones(active_count, active_count)
    )
    compiled = compile_bilinear_attention(
        block.attn,
        n_query=active_count,
        mask=mask,
        dense_oracle_maximum_elements=maximum_elements,
    )
    if compiled.head_cores is None:
        raise ValueError("dense per-head attention cores exceeded the tiny-oracle limit")
    shape = (homogeneous_dimension,) + (homogeneous_dimension,) * 5
    ordered = block.attn.wo.weight.new_zeros(shape)
    zeros = (0,) * 5
    ordered[(0, *zeros)] = 1.0
    for coordinate in range(token_count * dimension):
        ordered[(1 + coordinate, 1 + coordinate, 0, 0, 0, 0)] = 1.0
    gain = block.attn_gain.detach().reshape(())
    for query in range(active_count):
        workspace_query = active_tokens[query]
        for output in range(dimension):
            ordered[(1 + workspace_query * dimension + output, *zeros)] += gain * compiled.output_bias[output]
    local_dimension = dimension + 1
    for route in compiled.routes:
        head_slice = slice(
            route.head_index * compiled.head_dim,
            (route.head_index + 1) * compiled.head_dim,
        )
        output_weights = compiled.output_weight[:, head_slice]
        workspace_query = active_tokens[route.query_index]
        workspace_source = active_tokens[route.source_index]
        role_tokens = (
            workspace_query,
            workspace_query,
            workspace_source,
            workspace_source,
            workspace_source,
        )
        for value_coordinate in range(compiled.head_dim):
            local_core = compiled.head_cores[route.head_index, value_coordinate]
            for local_indices in itertools.product(range(local_dimension), repeat=5):
                coefficient = local_core[local_indices]
                if float(coefficient.item()) == 0.0:
                    continue
                global_indices = tuple(
                    _global_index(token, local, dimension)
                    for token, local in zip(role_tokens, local_indices)
                )
                for output in range(dimension):
                    row = 1 + workspace_query * dimension + output
                    ordered[(row, *global_indices)] += (
                        gain * route.scale * output_weights[output, value_coordinate] * coefficient
                    )
    return _symmetrize_inputs(ordered)


@torch.no_grad()
def _compile_whole_state_ffn_residual(
    block,
    token_count: int,
    maximum_elements: int,
    *,
    active_tokens: tuple[int, ...] | None = None,
) -> Tensor:
    if not isinstance(block.ffn, BilinearFFN):
        raise ValueError("polynomial oracle requires BilinearFFN")
    if not isinstance(block.rbn_ffn, nn.Identity):
        raise ValueError("FFN pre-norm must be identity after exact scalar folding")
    local = compile_bilinear_ffn_residual_core(
        block.ffn,
        gain=float(block.ffn_gain.detach().item()),
        maximum_elements=maximum_elements,
    )
    dimension = block.ffn.dim
    active_tokens = tuple(range(token_count)) if active_tokens is None else tuple(active_tokens)
    if not active_tokens or len(set(active_tokens)) != len(active_tokens):
        raise ValueError("active_tokens must contain distinct workspace token indices")
    if any(token < 0 or token >= token_count for token in active_tokens):
        raise ValueError("active token index is outside the workspace")
    homogeneous_dimension = 1 + token_count * dimension
    if homogeneous_dimension ** 3 > maximum_elements:
        raise ValueError("whole-state FFN core exceeds the tiny-oracle limit")
    core = local.new_zeros(homogeneous_dimension, homogeneous_dimension, homogeneous_dimension)
    core[0, 0, 0] = 1.0
    identity = torch.eye(
        homogeneous_dimension - 1, dtype=core.dtype, device=core.device
    )
    core[1:, 0, 1:] += 0.5 * identity
    core[1:, 1:, 0] += 0.5 * identity
    for token in active_tokens:
        for output in range(1, dimension + 1):
            row = 1 + token * dimension + output - 1
            for left in range(dimension + 1):
                for right in range(dimension + 1):
                    # ``local`` contains the identity residual.  The workspace
                    # identity was inserted once above, so add only the branch.
                    identity_coefficient = 0.0
                    if left == 0 and right == output:
                        identity_coefficient += 0.5
                    if right == 0 and left == output:
                        identity_coefficient += 0.5
                    core[
                        row,
                        _global_index(token, left, dimension),
                        _global_index(token, right, dimension),
                    ] += local[output, left, right] - identity_coefficient
    return core


def raw_input_dimension(model: ChiVLA) -> int:
    cfg = model.cfg
    return (
        3 * cfg.image_size * cfg.image_size
        + cfg.max_instr_len * cfg.vocab_size
        + cfg.state_dim
        + cfg.n_embodiments
    )


@torch.no_grad()
def observation_to_polynomial_input(
    model: ChiVLA,
    img: Tensor,
    instr_ids: Tensor,
    state: Tensor,
    embodiment_id: Tensor,
) -> Tensor:
    cfg = model.cfg
    if img.ndim != 4 or img.shape[1:] != (3, cfg.image_size, cfg.image_size):
        raise ValueError("img shape mismatch")
    batch = img.shape[0]
    if instr_ids.shape != (batch, cfg.max_instr_len) or state.shape != (batch, cfg.state_dim):
        raise ValueError("instruction or state shape mismatch")
    if embodiment_id.shape != (batch,):
        raise ValueError("embodiment shape mismatch")
    instruction = F.one_hot(instr_ids, num_classes=cfg.vocab_size).to(dtype=img.dtype)
    embodiment = F.one_hot(embodiment_id, num_classes=cfg.n_embodiments).to(dtype=img.dtype)
    return torch.cat((img.flatten(1), instruction.flatten(1), state, embodiment), dim=1)


@torch.no_grad()
def _joint_tokens_from_polynomial_input(model: ChiVLA, raw: Tensor) -> Tensor:
    cfg = model.cfg
    cursor = 0
    image_count = 3 * cfg.image_size * cfg.image_size
    img = raw[:, cursor : cursor + image_count].reshape(-1, 3, cfg.image_size, cfg.image_size)
    cursor += image_count
    instruction_count = cfg.max_instr_len * cfg.vocab_size
    instruction = raw[:, cursor : cursor + instruction_count].reshape(
        -1, cfg.max_instr_len, cfg.vocab_size
    )
    cursor += instruction_count
    state = raw[:, cursor : cursor + cfg.state_dim]
    cursor += cfg.state_dim
    embodiment = raw[:, cursor : cursor + cfg.n_embodiments]
    cursor += cfg.n_embodiments
    if cursor != raw.shape[1]:
        raise ValueError("raw polynomial input width mismatch")
    batch = raw.shape[0]
    visual = model._visual_tokens(img)
    instruction_tokens = instruction @ model.tok_emb.weight
    state_token = F.linear(state, model.state_proj.weight, model.state_proj.bias)[:, None]
    embodiment_token = (embodiment @ model.embodiment_emb.weight)[:, None]
    tokens = torch.cat(
        (
            visual,
            model.bos.expand(batch, -1, -1),
            instruction_tokens,
            state_token,
            embodiment_token,
            model.action_queries.expand(batch, -1, -1),
        ),
        dim=1,
    )
    return tokens + model.pos_emb[:, : tokens.shape[1]]


@torch.no_grad()
def _prevision_workspace_from_polynomial_input(model: ChiVLA, raw: Tensor) -> Tensor:
    """Assemble the eventual joint workspace before the ChiViT block stack."""

    cfg = model.cfg
    if cfg.vit_dim != cfg.dim:
        raise ValueError("whole-workspace oracle requires vit_dim == dim")
    cursor = 0
    image_count = 3 * cfg.image_size * cfg.image_size
    img = raw[:, cursor : cursor + image_count].reshape(-1, 3, cfg.image_size, cfg.image_size)
    cursor += image_count
    instruction_count = cfg.max_instr_len * cfg.vocab_size
    instruction = raw[:, cursor : cursor + instruction_count].reshape(
        -1, cfg.max_instr_len, cfg.vocab_size
    )
    cursor += instruction_count
    state = raw[:, cursor : cursor + cfg.state_dim]
    cursor += cfg.state_dim
    embodiment = raw[:, cursor : cursor + cfg.n_embodiments]
    cursor += cfg.n_embodiments
    if cursor != raw.shape[1]:
        raise ValueError("raw polynomial input width mismatch")
    batch = raw.shape[0]
    visual = model.vision.patch(img).flatten(2).transpose(1, 2)
    visual = visual + model.vision.pos_emb
    instruction_tokens = instruction @ model.tok_emb.weight
    state_token = F.linear(state, model.state_proj.weight, model.state_proj.bias)[:, None]
    embodiment_token = (embodiment @ model.embodiment_emb.weight)[:, None]
    return torch.cat(
        (
            visual,
            model.bos.expand(batch, -1, -1),
            instruction_tokens,
            state_token,
            embodiment_token,
            model.action_queries.expand(batch, -1, -1),
        ),
        dim=1,
    )


@torch.no_grad()
def _compile_input_embedding(model: ChiVLA) -> Tensor:
    input_dimension = raw_input_dimension(model)
    basis = torch.cat(
        (
            torch.zeros(1, input_dimension, dtype=model.pos_emb.dtype, device=model.pos_emb.device),
            torch.eye(input_dimension, dtype=model.pos_emb.dtype, device=model.pos_emb.device),
        ),
        dim=0,
    )
    outputs = _joint_tokens_from_polynomial_input(model, basis).flatten(1)
    base = outputs[0]
    linear = (outputs[1:] - base).T
    embedding = outputs.new_zeros(outputs.shape[1] + 1, input_dimension + 1)
    embedding[0, 0] = 1.0
    embedding[1:, 0] = base
    embedding[1:, 1:] = linear
    return embedding


@torch.no_grad()
def _compile_prevision_workspace_embedding(model: ChiVLA) -> Tensor:
    input_dimension = raw_input_dimension(model)
    basis = torch.cat(
        (
            torch.zeros(1, input_dimension, dtype=model.pos_emb.dtype, device=model.pos_emb.device),
            torch.eye(input_dimension, dtype=model.pos_emb.dtype, device=model.pos_emb.device),
        ),
        dim=0,
    )
    outputs = _prevision_workspace_from_polynomial_input(model, basis).flatten(1)
    base = outputs[0]
    linear = (outputs[1:] - base).T
    embedding = outputs.new_zeros(outputs.shape[1] + 1, input_dimension + 1)
    embedding[0, 0] = 1.0
    embedding[1:, 0] = base
    embedding[1:, 1:] = linear
    return embedding


@torch.no_grad()
def _compile_visual_projection_bridge(model: ChiVLA) -> Tensor:
    """Map prevision workspace to the exact pre-backbone joint token state."""

    cfg = model.cfg
    if cfg.vit_dim != cfg.dim:
        raise ValueError("visual projection bridge requires vit_dim == dim")
    homogeneous_dimension = 1 + model.seq_len * cfg.dim
    bridge = model.pos_emb.new_zeros(homogeneous_dimension, homogeneous_dimension)
    bridge[0, 0] = 1.0
    visual_count = cfg.vit_config().num_patches
    for token in range(model.seq_len):
        rows = slice(1 + token * cfg.dim, 1 + (token + 1) * cfg.dim)
        columns = slice(1 + token * cfg.dim, 1 + (token + 1) * cfg.dim)
        bridge[rows, 0] = model.pos_emb[0, token]
        if token < visual_count:
            bridge[rows, 0] += model.vis_proj.bias
            bridge[rows, columns] = model.vis_proj.weight
        else:
            bridge[rows, columns] = torch.eye(
                cfg.dim, dtype=bridge.dtype, device=bridge.device
            )
    return bridge


@torch.no_grad()
def _compile_action_head(model: ChiVLA) -> Tensor:
    cfg = model.cfg
    homogeneous_dimension = 1 + model.seq_len * cfg.dim
    head = model.action_head.weight.new_zeros(
        cfg.action_horizon * cfg.action_dim, homogeneous_dimension
    )
    first_action_token = model.seq_len - cfg.action_horizon
    for horizon in range(cfg.action_horizon):
        rows = slice(horizon * cfg.action_dim, (horizon + 1) * cfg.action_dim)
        columns = slice(
            1 + (first_action_token + horizon) * cfg.dim,
            1 + (first_action_token + horizon + 1) * cfg.dim,
        )
        head[rows, 0] = model.action_head.bias
        head[rows, columns] = model.action_head.weight
    return head


def _validate_tiny_vla(model: ChiVLA) -> None:
    if type(model) is not ChiVLA:
        raise TypeError("oracle requires exactly ChiVLA")
    cfg = model.cfg
    required = {
        "vision_encoder": "vit",
        "dual_vision": False,
        "vit_layers": 0,
        "attn": "bilinear",
        "ffn": "bilinear",
        "residual": True,
        "action_head": "linear",
        "n_phases": 0,
    }
    for name, expected in required.items():
        if getattr(cfg, name) != expected:
            raise ValueError(f"tiny oracle requires {name}={expected!r}")
    if isinstance(cfg.n_layers, bool) or not isinstance(cfg.n_layers, int) or cfg.n_layers < 1:
        raise ValueError("tiny oracle requires n_layers>=1")
    if model.training:
        raise ValueError("oracle requires eval mode")
    dtypes = {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()}
    if dtypes != {torch.float64}:
        raise ValueError("exact oracle requires one float64 parameter dtype")
    if (cfg.norm, cfg.qk_norm) not in (("none", "none"), ("scalar_rbn", "scalar_rbn")):
        if cfg.norm in ("rational", "pade") or cfg.qk_norm == "rational":
            raise ValueError(
                "Padé checkpoints require the rational projective path; " + RATIONAL_INTERFACE_GAP
            )
        raise ValueError("polynomial oracle supports identity or frozen scalar-RBN normalization")


def _validate_whole_workspace_vla(model: ChiVLA) -> None:
    if type(model) is not ChiVLA:
        raise TypeError("whole-workspace oracle requires exactly ChiVLA")
    cfg = model.cfg
    required = {
        "vision_encoder": "vit",
        "dual_vision": False,
        "vit_layers": 1,
        "n_layers": 1,
        "attn": "bilinear",
        "ffn": "bilinear",
        "residual": True,
        "action_head": "linear",
        "n_phases": 0,
    }
    for name, expected in required.items():
        if getattr(cfg, name) != expected:
            raise ValueError(f"whole-workspace oracle requires {name}={expected!r}")
    if cfg.vit_dim != cfg.dim:
        raise ValueError("whole-workspace oracle requires vit_dim == dim")
    if model.vision.use_cls:
        raise ValueError("whole-workspace oracle does not support a vision CLS token")
    if model.training:
        raise ValueError("whole-workspace oracle requires eval mode")
    dtypes = {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()}
    if dtypes != {torch.float64}:
        raise ValueError("exact whole-workspace oracle requires one float64 parameter dtype")
    if (cfg.norm, cfg.qk_norm) not in (("none", "none"), ("scalar_rbn", "scalar_rbn")):
        if cfg.norm in ("rational", "pade") or cfg.qk_norm == "rational":
            raise ValueError(
                "Padé checkpoints require the rational projective path; " + RATIONAL_INTERFACE_GAP
            )
        raise ValueError("whole-workspace oracle supports identity or frozen scalar-RBN normalization")


@torch.no_grad()
def compile_tiny_vla_residual_oracle(
    model: ChiVLA,
    *,
    maximum_elements: int = 2_000_000,
) -> TinyVLAResidualOracle:
    """Compile a real one-block tiny VLA without deleting vision or residuals."""

    _validate_tiny_vla(model)
    source_normalization = model.cfg.norm
    if model.cfg.norm == "scalar_rbn":
        polynomial_model = fold_vla(model)
    else:
        polynomial_model = copy.deepcopy(model).eval()
    if len(polynomial_model.vision.blocks.blocks) != 0:
        raise AssertionError("tiny oracle vision-depth invariant changed")
    embedding = _compile_input_embedding(polynomial_model)
    cores = []
    for index, block in enumerate(polynomial_model.backbone.blocks):
        if not block.residual:
            raise ValueError(f"residual route must remain enabled at backbone block {index}")
        cores.append(
            _compile_whole_state_attention_residual(
                block, polynomial_model.seq_len, maximum_elements
            )
        )
        cores.append(
            _compile_whole_state_ffn_residual(
                block, polynomial_model.seq_len, maximum_elements
            )
        )
    head = _compile_action_head(polynomial_model)
    network = MixedArityHomogeneousTN(embedding, tuple(cores), head)
    _validate_network(network)
    _require_symmetric(network)
    return TinyVLAResidualOracle(
        network,
        polynomial_model,
        raw_input_dimension(polynomial_model),
        polynomial_model.seq_len,
        source_normalization,
    )


@torch.no_grad()
def compile_tiny_vla_whole_workspace_oracle(
    model: ChiVLA,
    *,
    maximum_elements: int = 2_000_000,
) -> TinyVLAResidualOracle:
    """Compile one real ChiViT block and one real joint block end to end.

    The common homogeneous workspace carries nonvisual tokens unchanged through
    the vision block.  This is only a tensor-network representation strategy:
    the represented function is exactly the original model, in which the
    ChiViT block receives only visual tokens.  The arity-one bridge applies the
    learned visual projection and then adds every learned joint position.
    """

    _validate_whole_workspace_vla(model)
    source_normalization = model.cfg.norm
    if model.cfg.norm == "scalar_rbn":
        polynomial_model = fold_vla(model)
    else:
        polynomial_model = copy.deepcopy(model).eval()
    visual_count = polynomial_model.cfg.vit_config().num_patches
    active_visual_tokens = tuple(range(visual_count))
    embedding = _compile_prevision_workspace_embedding(polynomial_model)
    vision_block = polynomial_model.vision.blocks.blocks[0]
    if vision_block.attn.causal:
        raise ValueError("ChiViT attention must remain bidirectional")
    vision_attention = _compile_whole_state_attention_residual(
        vision_block,
        polynomial_model.seq_len,
        maximum_elements,
        active_tokens=active_visual_tokens,
    )
    vision_ffn = _compile_whole_state_ffn_residual(
        vision_block,
        polynomial_model.seq_len,
        maximum_elements,
        active_tokens=active_visual_tokens,
    )
    bridge = _compile_visual_projection_bridge(polynomial_model)
    joint_block = polynomial_model.backbone.blocks[0]
    if not joint_block.attn.causal:
        raise ValueError("joint attention must remain causal")
    joint_attention = _compile_whole_state_attention_residual(
        joint_block, polynomial_model.seq_len, maximum_elements
    )
    joint_ffn = _compile_whole_state_ffn_residual(
        joint_block, polynomial_model.seq_len, maximum_elements
    )
    head = _compile_action_head(polynomial_model)
    network = MixedArityHomogeneousTN(
        embedding,
        (vision_attention, vision_ffn, bridge, joint_attention, joint_ffn),
        head,
    )
    _validate_network(network)
    _require_symmetric(network)
    return TinyVLAResidualOracle(
        network,
        polynomial_model,
        raw_input_dimension(polynomial_model),
        polynomial_model.seq_len,
        source_normalization,
        claim_boundary=(
            "Canonical weight-only ODT for a tiny one-ChiViT-block plus one-joint-block "
            "identity/scalar-folded polynomial VLA. It is not a Padé rational canonical ODT."
        ),
    )


@torch.no_grad()
def oracle_actions(oracle: TinyVLAResidualOracle, raw_input: Tensor) -> Tensor:
    flat = mixed_arity_forward(oracle.network, raw_input)
    cfg = oracle.polynomial_model.cfg
    return flat.reshape(raw_input.shape[0], cfg.action_horizon, cfg.action_dim)


@torch.no_grad()
def replay_rational_residual_block(
    block,
    tokens: Tensor,
    mask: Tensor | None = None,
    *,
    maximum_routes: int = 64,
    minimum_absolute_denominator: float = 1e-8,
) -> RationalResidualBlockReplay:
    """Exact staged projective replay of an unchanged Padé residual block.

    This function does not claim global rational canonicalization.  It keeps
    the deployed norms and residuals and records every denominator ledger entry
    exposed by the current projective primitives.
    """

    if not block.residual:
        raise ValueError("rational replay requires the deployed residual route")
    if not isinstance(block.rbn_attn, RationalNorm) or not isinstance(block.rbn_ffn, RationalNorm):
        raise ValueError("block pre-norms must be Padé RationalNorm")
    if not isinstance(block.attn, BilinearAttention) or block.attn.qk_norm != "rational":
        raise ValueError("attention must use rational Q/K normalization")
    normalized_attention = rational_norm_projective(
        tokens,
        block.rbn_attn,
        label="block.attention_pre_norm",
        minimum_absolute_denominator=minimum_absolute_denominator,
    )
    normalized_tokens = projective_value(
        normalized_attention,
        minimum_absolute_denominator=minimum_absolute_denominator,
    )
    attention_rows = tiny_rational_attention_projective_oracle(
        block.attn,
        normalized_tokens,
        mask,
        maximum_routes=maximum_routes,
        minimum_absolute_denominator=minimum_absolute_denominator,
    )
    attention_value = stack_projective_rows(
        attention_rows,
        minimum_absolute_denominator=minimum_absolute_denominator,
    )
    after_attention = tokens + block.attn_gain * attention_value
    ffn_pair = rational_ffn_residual_projective(
        after_attention,
        block.rbn_ffn,
        block.ffn,
        block.ffn_gain,
        label="block.ffn",
        minimum_absolute_denominator=minimum_absolute_denominator,
    )
    value = projective_value(ffn_pair, minimum_absolute_denominator=minimum_absolute_denominator)
    ledger = normalized_attention.denominator_ledger
    for row in attention_rows:
        ledger += row.denominator_ledger
    ledger += ffn_pair.denominator_ledger
    return RationalResidualBlockReplay(value, ledger)


__all__ = [
    "CanonicalMixedArityODT",
    "MixedArityHomogeneousTN",
    "POLYNOMIAL_CLAIM_BOUNDARY",
    "POLYNOMIAL_OBJECT_KIND",
    "RATIONAL_INTERFACE_GAP",
    "RATIONAL_REPLAY_OBJECT_KIND",
    "RationalResidualBlockReplay",
    "TinyVLAResidualOracle",
    "canonical_environments_mixed_arity",
    "canonicalize_mixed_arity",
    "compile_tiny_vla_residual_oracle",
    "compile_tiny_vla_whole_workspace_oracle",
    "gauge_bond",
    "mixed_arity_forward",
    "observation_to_polynomial_input",
    "oracle_actions",
    "one_leg_gauge_negative_control",
    "raw_input_dimension",
    "replay_rational_residual_block",
]
