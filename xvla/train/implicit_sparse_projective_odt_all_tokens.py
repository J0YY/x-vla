"""Exact all-token projective compilation for unchanged Padé χ blocks.

The compositional object is a tuple of local ``(numerator_vector, denominator)``
bonds.  Tokens remain separate between blocks.  A balanced concatenation is
available only as an optional, small observable boundary for the existing
single-root direct-RQ oracle.  It is never inserted between transformer blocks.
"""

from __future__ import annotations

import ast
import inspect
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from xvla.nn.attention import causal_mask
from xvla.nn.block import ChiTransformerBlock
from xvla.train.implicit_sparse_projective_odt import (
    DenseCloneCore,
    ImplicitNode,
    ImplicitProjectiveDAG,
    MaterializationTelemetry,
    ScaledBatch,
    _Builder,
    _add_pair_vectors,
    _affine_pair,
    _concatenate_pair_vectors,
    _core_apply,
    _dot_pair_vectors,
    _fused_cp_ffn_pair,
    _identity_pair_leaf,
    _multiply_pair_scalars,
    _norm_snapshot,
    _normalize_batch,
    _pade_norm_pair,
    _scale_pair_vector,
    _scale_pair_vector_by_scalar,
    _validate_block_source,
    _validate_network,
    _walk_unique,
    evaluate_boundary_quotient,
)


ALL_TOKEN_CLAIM_BOUNDARY = (
    "Exact projective token-tuple graph for every requested final output token, with "
    "all final tokens requested by default, of unchanged Padé ChiTransformerBlock "
    "instances. Raw token leaves, positioned inputs, pre-norms, "
    "and per-head Q/K/V nodes are shared by object identity. Tokens stay on local "
    "feature bonds between blocks. An optional balanced concatenation exists only at "
    "the final observable boundary. This is a clone-unfolded syntactic object, not "
    "compression and not an identified-variable or quotient-intrinsic metric."
)


@dataclass
class AllTokenImplicitOracle:
    token_networks: tuple[ImplicitProjectiveDAG, ...]
    token_outputs: tuple[ImplicitNode, ...]
    output_tokens: tuple[int, ...]
    raw_leaves: tuple[ImplicitNode, ...]
    blocks: tuple[ChiTransformerBlock, ...]
    positions: Tensor
    mask: Tensor
    telemetry: MaterializationTelemetry
    norm_buffer_snapshot: tuple[tuple[str, Tensor], ...]
    observable_network: ImplicitProjectiveDAG | None

    @property
    def token_count(self) -> int:
        return self.positions.shape[0]

    @property
    def feature_dimension(self) -> int:
        return self.positions.shape[1]


def _resolve_fixed_mask(
    block: ChiTransformerBlock,
    token_count: int,
    like: Tensor,
    mask: Tensor | None,
) -> Tensor:
    if mask is None:
        result = (
            causal_mask(token_count, dtype=like.dtype, device=like.device)
            if block.attn.causal
            else like.new_ones(token_count, token_count)
        )
    else:
        result = mask.to(dtype=like.dtype, device=like.device)
    if result.shape != (token_count, token_count):
        raise ValueError("all-token attention mask has the wrong shape")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("all-token attention mask must be finite")
    if not bool(((result == 0) | (result == 1)).all()):
        raise ValueError("all-token attention mask must be binary")
    if not bool((result.sum(dim=1) > 0).all()):
        raise ValueError("every all-token attention row needs one visible source")
    return result


def _validate_token_tuple(
    builder: _Builder,
    block: ChiTransformerBlock,
    token_nodes: tuple[ImplicitNode, ...],
) -> tuple[int, int, int]:
    if not token_nodes:
        raise ValueError("compile_block_tokens needs at least one token")
    dimensions = {node.output_dimension for node in token_nodes}
    if len(dimensions) != 1:
        raise ValueError("all projective token bonds must have one width")
    feature_dimension = next(iter(dimensions)) - 1
    if feature_dimension < 1:
        raise ValueError("projective token feature dimension must be positive")
    token_count, checked_dimension, heads = _validate_block_source(
        block,
        builder.like.new_zeros(len(token_nodes), feature_dimension),
    )
    if token_count != len(token_nodes) or checked_dimension != feature_dimension:
        raise ValueError("block and token tuple dimensions disagree")
    if (
        block.attn.wo.weight.dtype != builder.like.dtype
        or block.attn.wo.weight.device != builder.like.device
    ):
        raise ValueError("block and graph builder must share dtype and device")
    return token_count, feature_dimension, heads


def compile_block_tokens(
    builder: _Builder,
    block: ChiTransformerBlock,
    token_nodes: tuple[ImplicitNode, ...],
    mask: Tensor | None,
    label: str,
    output_tokens: Sequence[int] | None = None,
) -> tuple[ImplicitNode, ...]:
    """Compile one complete block while retaining a local bond per output token."""

    token_count, dimension, heads = _validate_token_tuple(builder, block, token_nodes)
    fixed_mask = _resolve_fixed_mask(block, token_count, builder.like, mask)
    targets = (
        tuple(range(token_count))
        if output_tokens is None
        else tuple(int(token) for token in output_tokens)
    )
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("output token selection must be nonempty and unique")
    if any(not 0 <= token < token_count for token in targets):
        raise ValueError("output token selection is out of range")
    normalized = tuple(
        _pade_norm_pair(builder, value, block.rbn_attn, f"{label}.pre_attention.token{token}")
        for token, value in enumerate(token_nodes)
    )

    head_dimension = dimension // heads
    branch_nodes: dict[str, tuple[tuple[ImplicitNode, ...], ...]] = {}
    for branch_name, linear, norm in (
        ("q1", block.attn.wq1, block.attn.rn_q1),
        ("k1", block.attn.wk1, block.attn.rn_k1),
        ("q2", block.attn.wq2, block.attn.rn_q2),
        ("k2", block.attn.wk2, block.attn.rn_k2),
    ):
        per_head: list[tuple[ImplicitNode, ...]] = []
        for head in range(heads):
            section = slice(head * head_dimension, (head + 1) * head_dimension)
            projected = tuple(
                _affine_pair(
                    builder,
                    normalized[token],
                    linear.weight[section],
                    None if linear.bias is None else linear.bias[section],
                    f"{label}.attention.head{head}.{branch_name}_affine.token{token}",
                )
                for token in range(token_count)
            )
            per_head.append(
                tuple(
                    _pade_norm_pair(
                        builder,
                        value,
                        norm,
                        f"{label}.attention.head{head}.{branch_name}_norm.token{token}",
                    )
                    for token, value in enumerate(projected)
                )
            )
        branch_nodes[branch_name] = tuple(per_head)

    value_nodes: list[tuple[ImplicitNode, ...]] = []
    for head in range(heads):
        section = slice(head * head_dimension, (head + 1) * head_dimension)
        value_nodes.append(
            tuple(
                _affine_pair(
                    builder,
                    normalized[token],
                    block.attn.wv.weight[section],
                    None if block.attn.wv.bias is None else block.attn.wv.bias[section],
                    f"{label}.attention.head{head}.value_affine.token{token}",
                )
                for token in range(token_count)
            )
        )

    outputs: list[ImplicitNode] = []
    for target in targets:
        visible_sources = [
            source
            for source in range(token_count)
            if float(fixed_mask[target, source].item()) == 1.0
        ]
        visible = builder.like.new_tensor(float(len(visible_sources)))
        row_scale = (
            visible.rsqrt()
            if block.attn.row_scale == "invsqrt"
            else visible.reciprocal()
        )
        fixed_scale = row_scale / float(block.attn._score_denom)
        head_outputs: list[ImplicitNode] = []
        for head in range(heads):
            routes: list[ImplicitNode] = []
            for source in visible_sources:
                score1 = _dot_pair_vectors(
                    builder,
                    branch_nodes["q1"][head][target],
                    branch_nodes["k1"][head][source],
                    f"{label}.attention.target{target}.head{head}.source{source}.score1",
                )
                score2 = _dot_pair_vectors(
                    builder,
                    branch_nodes["q2"][head][target],
                    branch_nodes["k2"][head][source],
                    f"{label}.attention.target{target}.head{head}.source{source}.score2",
                )
                score = _multiply_pair_scalars(
                    builder,
                    score1,
                    score2,
                    f"{label}.attention.target{target}.head{head}.source{source}.score_product",
                )
                route = _scale_pair_vector_by_scalar(
                    builder,
                    score,
                    value_nodes[head][source],
                    f"{label}.attention.target{target}.head{head}.source{source}.score_times_value",
                )
                routes.append(
                    _scale_pair_vector(
                        builder,
                        route,
                        fixed_scale,
                        f"{label}.attention.target{target}.head{head}.source{source}.fixed_scale",
                    )
                )
            head_output = routes[0]
            for route_index, route in enumerate(routes[1:], start=1):
                head_output = _add_pair_vectors(
                    builder,
                    head_output,
                    route,
                    f"{label}.attention.target{target}.head{head}.route_add{route_index}",
                )
            head_outputs.append(head_output)

        attention = head_outputs[0]
        for head, head_output in enumerate(head_outputs[1:], start=1):
            attention = _concatenate_pair_vectors(
                builder,
                attention,
                head_output,
                f"{label}.attention.target{target}.head_concat{head}",
            )
        attention = _affine_pair(
            builder,
            attention,
            block.attn.wo.weight,
            block.attn.wo.bias,
            f"{label}.attention.target{target}.output_affine",
        )
        attention = _scale_pair_vector(
            builder,
            attention,
            block.attn_gain,
            f"{label}.attention.target{target}.residual_gain",
        )
        after_attention = _add_pair_vectors(
            builder,
            token_nodes[target],
            attention,
            f"{label}.attention.target{target}.residual_add",
        )
        ffn_input = _pade_norm_pair(
            builder,
            after_attention,
            block.rbn_ffn,
            f"{label}.pre_ffn.token{target}",
        )
        ffn = _fused_cp_ffn_pair(
            builder,
            ffn_input,
            block.ffn,
            f"{label}.ffn.token{target}.fused_cp",
        )
        ffn = _scale_pair_vector(
            builder,
            ffn,
            block.ffn_gain,
            f"{label}.ffn.token{target}.residual_gain",
        )
        outputs.append(
            _add_pair_vectors(
                builder,
                after_attention,
                ffn,
                f"{label}.ffn.token{target}.residual_add",
            )
        )
    return tuple(outputs)


def _balanced_observable_concat(
    builder: _Builder,
    token_outputs: tuple[ImplicitNode, ...],
    label: str,
) -> ImplicitNode:
    level = list(token_outputs)
    depth = 0
    while len(level) > 1:
        following: list[ImplicitNode] = []
        for pair in range(0, len(level), 2):
            if pair + 1 == len(level):
                following.append(level[pair])
            else:
                following.append(
                    _concatenate_pair_vectors(
                        builder,
                        level[pair],
                        level[pair + 1],
                        f"{label}.level{depth}.pair{pair // 2}",
                    )
                )
        level = following
        depth += 1
    return level[0]


def _snapshot_blocks(
    blocks: tuple[ChiTransformerBlock, ...],
) -> tuple[tuple[str, Tensor], ...]:
    result: list[tuple[str, Tensor]] = []
    for index, block in enumerate(blocks):
        result.extend(
            (f"block{index}.{name}", value)
            for name, value in _norm_snapshot(block)
        )
    return tuple(result)


@torch.no_grad()
def compile_all_token_projective_stack(
    blocks: Sequence[ChiTransformerBlock],
    positions: Tensor,
    *,
    mask: Tensor | None = None,
    include_observable_boundary: bool = False,
    maximum_materialized_observable_width: int = 512,
    output_tokens: Sequence[int] | None = None,
) -> AllTokenImplicitOracle:
    """Compile one or more unchanged blocks, adding positions exactly once."""

    block_tuple = tuple(blocks)
    if not block_tuple:
        raise ValueError("all-token compilation needs at least one block")
    if positions.ndim != 2:
        raise ValueError("all-token positions must be a token-by-feature matrix")
    like = block_tuple[0].attn.wo.weight
    token_count, dimension = positions.shape
    if positions.dtype != like.dtype or positions.device != like.device:
        raise ValueError("positions and blocks must share dtype and device")
    telemetry = MaterializationTelemetry()
    builder = _Builder(like, telemetry)
    raw_leaves = tuple(
        _identity_pair_leaf(builder, token, dimension) for token in range(token_count)
    )
    identity = torch.eye(dimension, dtype=like.dtype, device=like.device)
    token_outputs = tuple(
        _affine_pair(
            builder,
            raw_leaves[token],
            identity,
            positions[token],
            f"input.position.token{token}",
        )
        for token in range(token_count)
    )
    fixed_mask = _resolve_fixed_mask(block_tuple[0], token_count, like, mask)
    for block_index, block in enumerate(block_tuple):
        candidate_mask = _resolve_fixed_mask(block, token_count, like, mask)
        if not torch.equal(candidate_mask, fixed_mask):
            raise ValueError("all blocks in a token stack must use the same fixed mask")
        token_outputs = compile_block_tokens(
            builder,
            block,
            token_outputs,
            fixed_mask,
            f"block{block_index}",
            output_tokens=(
                output_tokens if block_index == len(block_tuple) - 1 else None
            ),
        )

    selected_outputs = (
        tuple(range(token_count))
        if output_tokens is None
        else tuple(int(token) for token in output_tokens)
    )

    token_networks = tuple(
        ImplicitProjectiveDAG(
            root,
            torch.eye(dimension + 1, dtype=like.dtype, device=like.device),
            0,
            token_count,
            dimension,
            token,
            fixed_mask,
            ALL_TOKEN_CLAIM_BOUNDARY,
        )
        for token, root in zip(selected_outputs, token_outputs)
    )
    observable_network = None
    if include_observable_boundary:
        observable_width = len(token_outputs) * dimension
        if observable_width > maximum_materialized_observable_width:
            raise ValueError(
                "refusing to materialize a wide observable concat; keep the token tuple local"
            )
        observable_root = _balanced_observable_concat(
            builder, token_outputs, "observable.concat"
        )
        observable_network = ImplicitProjectiveDAG(
            observable_root,
            torch.eye(observable_width + 1, dtype=like.dtype, device=like.device),
            0,
            token_count,
            dimension,
            0,
            fixed_mask,
            ALL_TOKEN_CLAIM_BOUNDARY,
        )
        _validate_network(observable_network)
    oracle = AllTokenImplicitOracle(
        token_networks,
        token_outputs,
        selected_outputs,
        raw_leaves,
        block_tuple,
        positions.detach().clone(),
        fixed_mask,
        telemetry,
        _snapshot_blocks(block_tuple),
        observable_network,
    )
    _validate_all_token_oracle(oracle)
    return oracle


@torch.no_grad()
def compile_all_token_projective_block(
    block: ChiTransformerBlock,
    positions: Tensor,
    *,
    mask: Tensor | None = None,
    include_observable_boundary: bool = False,
    maximum_materialized_observable_width: int = 512,
    output_tokens: Sequence[int] | None = None,
) -> AllTokenImplicitOracle:
    return compile_all_token_projective_stack(
        (block,),
        positions,
        mask=mask,
        include_observable_boundary=include_observable_boundary,
        maximum_materialized_observable_width=maximum_materialized_observable_width,
        output_tokens=output_tokens,
    )


def _union_nodes(roots: tuple[ImplicitNode, ...]) -> tuple[ImplicitNode, ...]:
    by_identity: dict[int, ImplicitNode] = {}
    for root in roots:
        for node in _walk_unique(root):
            by_identity.setdefault(id(node), node)
    return tuple(by_identity.values())


def _validate_all_token_oracle(oracle: AllTokenImplicitOracle) -> None:
    if len(oracle.token_networks) != len(oracle.token_outputs):
        raise ValueError("all-token network count does not match token outputs")
    if len(oracle.output_tokens) != len(oracle.token_outputs):
        raise ValueError("all-token output index count does not match token outputs")
    if not oracle.output_tokens or len(set(oracle.output_tokens)) != len(oracle.output_tokens):
        raise ValueError("all-token output indices must be nonempty and unique")
    for output_index, (token, network) in enumerate(
        zip(oracle.output_tokens, oracle.token_networks)
    ):
        _validate_network(network)
        if (
            network.root is not oracle.token_outputs[output_index]
            or network.selected_token != token
        ):
            raise ValueError("all-token root ordering is inconsistent")
        if not torch.equal(network.mask, oracle.mask):
            raise ValueError("all-token networks disagree on their fixed mask")
    nodes = _union_nodes(oracle.token_outputs)
    uid_owners: dict[int, int] = {}
    leaves: dict[int, set[int]] = defaultdict(set)
    for node in nodes:
        owner = uid_owners.setdefault(node.uid, id(node))
        if owner != id(node):
            raise ValueError("all-token union has duplicate uids on distinct nodes")
        if node.physical_token is not None:
            leaves[node.physical_token].add(id(node))
    if set(leaves) != set(range(oracle.token_count)):
        raise ValueError("all-token graph does not reach every raw token")
    if any(len(identities) != 1 for identities in leaves.values()):
        raise ValueError("raw token leaves were cloned instead of shared")
    if any(id(oracle.raw_leaves[token]) not in leaves[token] for token in leaves):
        raise ValueError("all-token raw leaf registry is inconsistent")


@torch.no_grad()
def evaluate_all_token_quotients(
    oracle: AllTokenImplicitOracle,
    raw_input: Tensor,
) -> Tensor:
    """Evaluate every token with one shared-node memo and no intermediate division."""

    _validate_all_token_oracle(oracle)
    if raw_input.ndim != 3 or tuple(raw_input.shape[1:]) != (
        oracle.token_count,
        oracle.feature_dimension,
    ):
        raise ValueError("raw input shape does not match all-token graph")
    memo: dict[int, ScaledBatch] = {}

    def evaluate(node: ImplicitNode) -> ScaledBatch:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        if node.children:
            children = tuple(evaluate(child) for child in node.children)
            value = _core_apply(node.core, tuple(child.mantissa for child in children))
            exponent = torch.stack(
                tuple(child.binary_exponent for child in children)
            ).sum(0)
        else:
            if node.physical_token is None:
                raise ValueError("all-token leaf has no physical token")
            physical = torch.cat(
                (
                    raw_input[:, node.physical_token],
                    raw_input.new_ones(raw_input.shape[0], 1),
                ),
                dim=1,
            )
            value = _core_apply(node.core, (physical,))
            exponent = torch.zeros(
                raw_input.shape[0], dtype=torch.int64, device=raw_input.device
            )
        result = _normalize_batch(
            value,
            exponent + node.core.binary_exponent,
        )
        memo[id(node)] = result
        return result

    quotients: list[Tensor] = []
    for root in oracle.token_outputs:
        pair = evaluate(root).mantissa
        denominator = pair[:, -1]
        relative = denominator.abs() / pair.abs().amax(dim=1).clamp_min(
            torch.finfo(pair.dtype).tiny
        )
        if float(relative.min().item()) <= 100.0 * torch.finfo(pair.dtype).eps:
            raise ValueError("all-token projective denominator is numerically zero")
        quotients.append(pair[:, :-1] / denominator[:, None])
    return torch.stack(quotients, dim=1)


@torch.no_grad()
def evaluate_observable_concat_quotient(
    oracle: AllTokenImplicitOracle,
    raw_input: Tensor,
) -> Tensor:
    if oracle.observable_network is None:
        raise ValueError("all-token oracle has no materialized observable boundary")
    flat = evaluate_boundary_quotient(oracle.observable_network, raw_input)
    return flat.reshape(
        raw_input.shape[0], len(oracle.output_tokens), oracle.feature_dimension
    )


@torch.no_grad()
def source_all_token_output(
    oracle: AllTokenImplicitOracle,
    raw_input: Tensor,
) -> Tensor:
    value = raw_input + oracle.positions
    for block in oracle.blocks:
        value = block(value, mask=oracle.mask, method="explicit")
    return value[:, oracle.output_tokens]


def assert_all_token_norm_buffers_unchanged(oracle: AllTokenImplicitOracle) -> None:
    current = dict(_snapshot_blocks(oracle.blocks))
    for name, expected in oracle.norm_buffer_snapshot:
        if name not in current or not torch.equal(current[name], expected):
            raise RuntimeError(f"all-token compiler changed normalization buffer {name}")


def all_token_structure_statistics(
    oracle: AllTokenImplicitOracle,
) -> dict[str, int | dict[str, int]]:
    nodes = _union_nodes(oracle.token_outputs)
    parents: dict[int, int] = defaultdict(int)
    for parent in nodes:
        for child in parent.children:
            parents[id(child)] += 1
    label_parent_occurrences = {
        node.label: parents.get(id(node), 0)
        for node in nodes
    }
    return {
        "token_outputs": len(oracle.token_outputs),
        "unique_nodes": len(nodes),
        "unfolded_token_root_node_sum": sum(
            len(_walk_unique(root)) for root in oracle.token_outputs
        ),
        "edge_occurrences": sum(len(node.children) for node in nodes),
        "unique_physical_leaves": len(
            {id(node) for node in nodes if node.physical_token is not None}
        ),
        "maximum_local_token_bond_dimension": max(
            node.output_dimension for node in nodes
        ),
        "dense_clone_nodes": sum(
            isinstance(node.core, DenseCloneCore) for node in nodes
        ),
        "label_parent_occurrences": label_parent_occurrences,
    }


def audit_all_token_compiler_calls() -> dict[str, object]:
    """Fail closed if the tuple compiler grows a matrix-factorization path."""

    prohibited = {
        "svd",
        "svdvals",
        "pinv",
        "lstsq",
        "cholesky",
        "matrix_power",
        "polar",
        "inv",
        "inverse",
        "eig",
        "eigh",
        "eigvals",
        "eigvalsh",
    }
    tree = ast.parse(inspect.getsource(sys.modules[__name__]))
    observed: list[str] = []
    rejected: list[str] = []

    class CallAudit(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            function = node.func
            if isinstance(function, ast.Attribute):
                name = function.attr
            elif isinstance(function, ast.Name):
                name = function.id
            else:
                name = "<dynamic>"
            observed.append(name)
            if name in prohibited:
                rejected.append(name)
            self.generic_visit(node)

    CallAudit().visit(tree)
    if rejected:
        raise RuntimeError(
            "all-token compiler contains prohibited factorization calls: "
            + ", ".join(sorted(set(rejected)))
        )
    return {
        "scope": "entire_all_token_module",
        "observed_calls": sorted(set(observed)),
        "prohibited_calls_found": [],
    }


__all__ = [
    "ALL_TOKEN_CLAIM_BOUNDARY",
    "AllTokenImplicitOracle",
    "all_token_structure_statistics",
    "assert_all_token_norm_buffers_unchanged",
    "audit_all_token_compiler_calls",
    "compile_all_token_projective_block",
    "compile_all_token_projective_stack",
    "compile_block_tokens",
    "evaluate_all_token_quotients",
    "evaluate_observable_concat_quotient",
    "source_all_token_output",
]
