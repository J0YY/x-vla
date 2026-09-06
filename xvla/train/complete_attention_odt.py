"""Exact typed compiler for complete polynomial bilinear attention.

The compiler keeps every query, source, and head route distinct until one
linear output assembly has mixed them.  This is essential because downstream
metrics generally contain cross-head and cross-source terms.  The represented
object is the ordered, clone-unfolded topology tensor, not the symmetrized
tied-input polynomial quotient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from xvla.nn.attention import BilinearAttention


Tensor = torch.Tensor
ATTENTION_ROLES = ("query_1", "query_2", "key_1", "key_2", "value")


@dataclass(frozen=True)
class AttentionRoute:
    query_index: int
    source_index: int
    head_index: int
    scale: float
    query_type: str
    source_type: str


@dataclass(frozen=True)
class CompiledBilinearAttention:
    """Complete typed route representation of one attention operation."""

    head_factors: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    head_cores: Tensor | None
    routes: tuple[AttentionRoute, ...]
    output_weight: Tensor
    output_bias: Tensor
    output_assembly: Tensor | None
    mask: Tensor
    row_scales: Tensor
    score_denominator: float
    n_query: int
    n_source: int
    model_dim: int
    head_dim: int
    object_kind: str = "typed_clone_unfolded_attention_topology"

    @property
    def output_shape(self) -> tuple[int, int]:
        return self.n_query, self.model_dim

    @property
    def route_feature_dimension(self) -> int:
        return len(self.routes) * self.head_dim


def _homogeneous(weight: Tensor, bias: Tensor, scale: Tensor | float = 1.0) -> Tensor:
    return torch.cat((bias[:, None], weight), dim=1) / scale


def _branch_matrices(module: BilinearAttention) -> tuple[Tensor, ...]:
    # Order exactly matches ATTENTION_ROLES. Do not use module declaration
    # order here because generalized role environments consume this interface.
    branches = (module.wq1, module.wq2, module.wk1, module.wk2, module.wv)
    if module.qk_norm == "none":
        scales: tuple[Tensor | float, ...] = (1.0,) * 5
    elif module.qk_norm == "scalar_rbn":
        norms = (
            module.rbn_q1,
            module.rbn_q2,
            module.rbn_k1,
            module.rbn_k2,
            module.rbn_v,
        )
        if any(not norm.frozen or not bool(norm.initialized) for norm in norms):
            raise ValueError("scalar_rbn branches must be calibrated, initialized, and frozen")
        scales = tuple(norm.scale for norm in norms)
    else:
        raise ValueError(
            f"qk_norm={module.qk_norm!r} is not a strict polynomial attention branch"
        )
    return tuple(
        _homogeneous(branch.weight.detach(), branch.bias.detach(), scale)
        for branch, scale in zip(branches, scales)
    )


@torch.no_grad()
def compile_attention_head_factors(
    module: BilinearAttention,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return factored projections in ``ATTENTION_ROLES`` order.

    These factors have shape ``(H, p, D)``. They form a multi-index tensor
    network, not a conventional CP decomposition with one shared rank index.
    """

    matrices = _branch_matrices(module)
    return tuple(
        matrix.reshape(module.n_heads, module.head_dim, module.dim + 1)
        for matrix in matrices
    )


@torch.no_grad()
def compile_attention_head_cores(
    module: BilinearAttention,
    *,
    maximum_elements: int = 10_000_000,
) -> Tensor:
    """Compile all heads to shape ``(H, p, D, D, D, D, D)``."""

    if isinstance(maximum_elements, bool) or not isinstance(maximum_elements, int) or maximum_elements <= 0:
        raise ValueError("maximum_elements must be a positive integer")
    dense_elements = module.n_heads * module.head_dim * (module.dim + 1) ** 5
    if dense_elements > maximum_elements:
        raise ValueError(
            f"dense attention core would require {dense_elements} elements, "
            f"above the tiny-oracle limit {maximum_elements}; use factorized head_factors"
        )
    q1, q2, k1, k2, value = compile_attention_head_factors(module)
    head_cores = []
    for head in range(module.n_heads):
        head_cores.append(
            torch.einsum(
                "ap,bq,ar,bs,ct->cpqrst",
                q1[head],
                q2[head],
                k1[head],
                k2[head],
                value[head],
            )
        )
    return torch.stack(head_cores)


def _validated_mask(
    mask: Tensor | None,
    *,
    n_query: int,
    n_source: int,
    causal: bool,
    like: Tensor,
) -> Tensor:
    if mask is None:
        if causal:
            if n_query != n_source:
                raise ValueError("implicit causal attention requires n_query == n_source")
            return torch.tril(like.new_ones(n_query, n_source))
        return like.new_ones(n_query, n_source)
    candidate = torch.as_tensor(mask, dtype=like.dtype, device=like.device)
    if candidate.shape != (n_query, n_source):
        raise ValueError(f"mask must have shape ({n_query}, {n_source})")
    if not bool(torch.isfinite(candidate).all()):
        raise ValueError("mask must contain only finite values")
    if not bool(((candidate == 0) | (candidate == 1)).all()):
        raise ValueError("mask must be binary with values in {0, 1}")
    return candidate.clone()


def _types(values: Sequence[str] | None, length: int, default: str, name: str) -> tuple[str, ...]:
    result = (default,) * length if values is None else tuple(values)
    if len(result) != length or any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain {length} nonempty strings")
    return result


@torch.no_grad()
def compile_bilinear_attention(
    module: BilinearAttention,
    *,
    n_query: int,
    n_source: int | None = None,
    mask: Tensor | None = None,
    query_types: Sequence[str] | None = None,
    source_types: Sequence[str] | None = None,
    dense_oracle_maximum_elements: int = 10_000_000,
) -> CompiledBilinearAttention:
    """Compile fixed-mask multihead attention into typed route cores."""

    if isinstance(n_query, bool) or not isinstance(n_query, int) or n_query <= 0:
        raise ValueError("n_query must be a positive integer")
    n_source = n_query if n_source is None else n_source
    if isinstance(n_source, bool) or not isinstance(n_source, int) or n_source <= 0:
        raise ValueError("n_source must be a positive integer")
    if (
        isinstance(dense_oracle_maximum_elements, bool)
        or not isinstance(dense_oracle_maximum_elements, int)
        or dense_oracle_maximum_elements <= 0
    ):
        raise ValueError("dense_oracle_maximum_elements must be a positive integer")
    head_factors = compile_attention_head_factors(module)
    factor_like = head_factors[0]
    dense_core_elements = module.n_heads * module.head_dim * (module.dim + 1) ** 5
    head_cores = (
        compile_attention_head_cores(
            module, maximum_elements=dense_oracle_maximum_elements
        )
        if dense_core_elements <= dense_oracle_maximum_elements
        else None
    )
    fixed_mask = _validated_mask(
        mask,
        n_query=n_query,
        n_source=n_source,
        causal=module.causal,
        like=factor_like,
    )
    counts = fixed_mask.sum(dim=-1).clamp_min(1.0)
    row_scales = counts.rsqrt() if module.row_scale == "invsqrt" else counts.reciprocal()
    q_types = _types(query_types, n_query, "query", "query_types")
    s_types = _types(source_types, n_source, "source", "source_types")

    routes = []
    for query_index in range(n_query):
        for source_index in range(n_source):
            if float(fixed_mask[query_index, source_index].item()) == 0.0:
                continue
            for head_index in range(module.n_heads):
                routes.append(
                    AttentionRoute(
                        query_index=query_index,
                        source_index=source_index,
                        head_index=head_index,
                        scale=float(
                            (
                                fixed_mask[query_index, source_index]
                                * row_scales[query_index]
                                / module._score_denom
                            ).item()
                        ),
                        query_type=q_types[query_index],
                        source_type=s_types[source_index],
                    )
                )

    output_dimension = n_query * module.dim
    output_bias = module.wo.bias.detach().to(
        dtype=factor_like.dtype, device=factor_like.device
    ).clone()
    output_weight = module.wo.weight.detach().to(
        dtype=factor_like.dtype, device=factor_like.device
    ).clone()
    dense_assembly_elements = output_dimension * (1 + len(routes) * module.head_dim)
    assembly = None
    if dense_assembly_elements <= dense_oracle_maximum_elements:
        assembly = factor_like.new_zeros(
            output_dimension, 1 + len(routes) * module.head_dim
        )
        for query_index in range(n_query):
            row = slice(query_index * module.dim, (query_index + 1) * module.dim)
            assembly[row, 0] = output_bias
        for route_index, route in enumerate(routes):
            row = slice(route.query_index * module.dim, (route.query_index + 1) * module.dim)
            column = slice(
                1 + route_index * module.head_dim,
                1 + (route_index + 1) * module.head_dim,
            )
            head_column = slice(
                route.head_index * module.head_dim,
                (route.head_index + 1) * module.head_dim,
            )
            assembly[row, column] = route.scale * output_weight[:, head_column]

    return CompiledBilinearAttention(
        head_factors=head_factors,
        head_cores=head_cores,
        routes=tuple(routes),
        output_weight=output_weight,
        output_bias=output_bias,
        output_assembly=assembly,
        mask=fixed_mask,
        row_scales=row_scales,
        score_denominator=float(module._score_denom),
        n_query=n_query,
        n_source=n_source,
        model_dim=module.dim,
        head_dim=module.head_dim,
    )


def _homogenize_tokens(tokens: Tensor, dimension: int, name: str) -> Tensor:
    if tokens.ndim != 3 or tokens.shape[-1] != dimension:
        raise ValueError(f"{name} must have shape (batch, positions, {dimension})")
    one = tokens.new_ones(*tokens.shape[:-1], 1)
    return torch.cat((one, tokens), dim=-1)


@torch.no_grad()
def attention_route_features(
    compiled: CompiledBilinearAttention,
    query_tokens: Tensor,
    source_tokens: Tensor | None = None,
) -> Tensor:
    """Evaluate all route features in declared route-major ordering."""

    source_tokens = query_tokens if source_tokens is None else source_tokens
    factor_like = compiled.head_factors[0]
    if query_tokens.dtype != factor_like.dtype or query_tokens.device != factor_like.device:
        raise ValueError("compiled factors and query_tokens must share dtype and device")
    if source_tokens.dtype != factor_like.dtype or source_tokens.device != factor_like.device:
        raise ValueError("compiled factors and source_tokens must share dtype and device")
    queries = _homogenize_tokens(query_tokens, compiled.model_dim, "query_tokens")
    sources = _homogenize_tokens(source_tokens, compiled.model_dim, "source_tokens")
    if queries.shape[1] != compiled.n_query or sources.shape[1] != compiled.n_source:
        raise ValueError("token position counts do not match the compiled object")
    if queries.shape[0] != sources.shape[0]:
        raise ValueError("query_tokens and source_tokens must have the same batch size")
    q1, q2, k1, k2, value = compiled.head_factors
    if not compiled.routes:
        return query_tokens.new_empty(query_tokens.shape[0], 0)
    projected_q1 = torch.einsum("bnd,hpd->bhnp", queries, q1)
    projected_q2 = torch.einsum("bnd,hpd->bhnp", queries, q2)
    projected_k1 = torch.einsum("bnd,hpd->bhnp", sources, k1)
    projected_k2 = torch.einsum("bnd,hpd->bhnp", sources, k2)
    projected_value = torch.einsum("bnd,hpd->bhnp", sources, value)
    heads = torch.tensor(
        [route.head_index for route in compiled.routes],
        dtype=torch.long,
        device=query_tokens.device,
    )
    query_indices = torch.tensor(
        [route.query_index for route in compiled.routes],
        dtype=torch.long,
        device=query_tokens.device,
    )
    source_indices = torch.tensor(
        [route.source_index for route in compiled.routes],
        dtype=torch.long,
        device=query_tokens.device,
    )
    selected_q1 = projected_q1[:, heads, query_indices]
    selected_q2 = projected_q2[:, heads, query_indices]
    selected_k1 = projected_k1[:, heads, source_indices]
    selected_k2 = projected_k2[:, heads, source_indices]
    selected_value = projected_value[:, heads, source_indices]
    score = (selected_q1 * selected_k1).sum(-1) * (selected_q2 * selected_k2).sum(-1)
    return (score[..., None] * selected_value).reshape(query_tokens.shape[0], -1)


@torch.no_grad()
def assemble_attention_routes(
    compiled: CompiledBilinearAttention,
    route_features: Tensor,
) -> Tensor:
    """Apply the implicit sparse/block output assembly without densifying it."""

    if route_features.ndim != 2 or route_features.shape[1] != compiled.route_feature_dimension:
        raise ValueError(
            f"route_features must have shape (batch, {compiled.route_feature_dimension})"
        )
    output = compiled.output_bias.expand(
        route_features.shape[0], compiled.n_query, compiled.model_dim
    ).clone()
    if compiled.routes:
        features = route_features.reshape(
            route_features.shape[0], len(compiled.routes), compiled.head_dim
        )
        heads = torch.tensor(
            [route.head_index for route in compiled.routes],
            dtype=torch.long,
            device=route_features.device,
        )
        queries = torch.tensor(
            [route.query_index for route in compiled.routes],
            dtype=torch.long,
            device=route_features.device,
        )
        scales = route_features.new_tensor([route.scale for route in compiled.routes])
        head_weights = compiled.output_weight.reshape(
            compiled.model_dim, -1, compiled.head_dim
        ).permute(1, 0, 2)[heads]
        contributions = torch.einsum("brp,rdp,r->brd", features, head_weights, scales)
        output.index_add_(1, queries, contributions)
    return output


@torch.no_grad()
def evaluate_compiled_attention(
    compiled: CompiledBilinearAttention,
    query_tokens: Tensor,
    source_tokens: Tensor | None = None,
) -> Tensor:
    """Evaluate the exact compiled route object on typed token inputs."""

    features = attention_route_features(compiled, query_tokens, source_tokens)
    return assemble_attention_routes(compiled, features)


@torch.no_grad()
def pullback_route_metric(
    compiled: CompiledBilinearAttention,
    output_metric: Tensor,
) -> Tensor:
    """Return the dense exact metric on tensor-unit plus route features."""

    expected = compiled.n_query * compiled.model_dim
    if output_metric.shape != (expected, expected):
        raise ValueError(f"output_metric must have shape ({expected}, {expected})")
    if compiled.output_assembly is None:
        raise ValueError(
            "dense route metric exceeds the tiny-oracle materialization limit; "
            "a factorized joint pullback is not implemented by this function"
        )
    if output_metric.dtype != compiled.output_assembly.dtype or output_metric.device != compiled.output_assembly.device:
        raise ValueError("output_metric and compiled assembly must share dtype and device")
    if not bool(torch.isfinite(output_metric).all()):
        raise ValueError("output_metric must contain only finite values")
    tolerance = 100 * torch.finfo(output_metric.dtype).eps * expected * max(
        float(output_metric.abs().max().item()), torch.finfo(output_metric.dtype).tiny
    )
    if float((output_metric - output_metric.T).abs().max().item()) > tolerance:
        raise ValueError("output_metric must be symmetric")
    eigenvalues = torch.linalg.eigvalsh(0.5 * (output_metric + output_metric.T))
    if float(eigenvalues[0].item()) < -min(tolerance, 1e-6 * max(float(eigenvalues[-1].item()), 1e-300)):
        raise ValueError("output_metric must be positive semidefinite")
    result = compiled.output_assembly.T @ output_metric @ compiled.output_assembly
    return 0.5 * (result + result.T)
