"""Adversarial Kronecker-closure audit for actual ODT bottom-up factors.

The experiment uses exact per-token homogeneous coordinates, so each hidden
bond is ``R^N tensor R^(d+1)`` and its token/feature Kronecker rank is
well-defined.  It compiles the repository's real bilinear attention and FFN
weights, performs the actual reduced-RQ sweep, pushes every emitted factor into
every occurrence of the next core, and measures numerical Kronecker ranks of
the raw cores, transformed cores, R factors, and row-isometric quotients.

Both ordered typed-role attention and its tied-input symmetric quotient are
run.  The latter preserves the diagonal forward map but can have substantially
higher Kronecker rank.  No rank-one closure is assumed by the gates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from xvla.nn.attention import BilinearAttention, causal_mask
from xvla.nn.bilinear import BilinearFFN
from xvla.nn.block import ChiTransformerBlock
from xvla.train.canonical_odt import reduced_rq_rows
from xvla.train.complete_attention_odt import compile_bilinear_attention
from xvla.train.residual_route_odt import compile_bilinear_ffn_residual_core


DTYPE = torch.float64


@dataclass(frozen=True)
class Case:
    name: str
    residual: bool
    symmetrize_attention: bool
    affine_biases: bool
    position_embedding: bool


CASES = (
    Case("ordered_nores_nobias_nopos", False, False, False, False),
    Case("symmetric_nores_nobias_nopos", False, True, False, False),
    Case("ordered_residual_nobias_nopos", True, False, False, False),
    Case("symmetric_residual_nobias_nopos", True, True, False, False),
    Case("symmetric_residual_bias_nopos", True, True, True, False),
    Case("symmetric_residual_nobias_pos", True, True, False, True),
    Case("ordered_residual_bias_pos", True, False, True, True),
    Case("symmetric_residual_bias_pos", True, True, True, True),
)


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = expected.norm().clamp_min(torch.finfo(expected.dtype).tiny)
    return float(((actual - expected).norm() / scale).item())


def numerical_rank(singular_values: torch.Tensor, tolerance: float) -> int:
    if singular_values.numel() == 0 or float(singular_values[0].item()) == 0.0:
        return 0
    return int((singular_values / singular_values[0] > tolerance).sum().item())


def matrix_kron_spectrum(matrix: torch.Tensor, tokens: int, feature: int) -> torch.Tensor:
    if matrix.shape != (tokens * feature, tokens * feature):
        raise ValueError("matrix does not have a square token-by-feature bond shape")
    rearranged = matrix.reshape(tokens, feature, tokens, feature)
    rearranged = rearranged.permute(0, 2, 1, 3).reshape(tokens**2, feature**2)
    return torch.linalg.svdvals(rearranged)


def core_kron_spectrum(core: torch.Tensor, tokens: int, feature: int) -> torch.Tensor:
    arity = core.ndim - 1
    expected = tokens * feature
    if any(dimension != expected for dimension in core.shape):
        raise ValueError("core does not have uniform token-by-feature bond axes")
    split_shape = tuple(value for _ in range(arity + 1) for value in (tokens, feature))
    split = core.reshape(split_shape)
    token_axes = tuple(2 * occurrence for occurrence in range(arity + 1))
    feature_axes = tuple(axis + 1 for axis in token_axes)
    rearranged = split.permute(*(token_axes + feature_axes)).reshape(
        tokens ** (arity + 1), feature ** (arity + 1)
    )
    return torch.linalg.svdvals(rearranged)


def spectrum_record(values: torch.Tensor, tolerance: float) -> dict[str, object]:
    scale = values[0].clamp_min(torch.finfo(values.dtype).tiny) if values.numel() else values.new_tensor(1.0)
    normalized = values / scale
    return {
        "rank": numerical_rank(values, tolerance),
        "normalized_singular_values": [float(value.item()) for value in normalized[:16]],
    }


def apply_input_factor(core: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    result = core
    for axis in range(1, core.ndim):
        result = (result.movedim(axis, -1) @ factor).movedim(-1, axis)
    return result


def evaluate_core(core: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    labels = "ijklmnpqrstuvwxyzabcdefgh"
    arity = core.ndim - 1
    roles = labels[:arity]
    equation = f"o{roles}," + ",".join(f"b{role}" for role in roles) + "->bo"
    return torch.einsum(equation, core, *([state] * arity))


def symmetrize_inputs(core: torch.Tensor) -> torch.Tensor:
    permutations = tuple(itertools.permutations(range(core.ndim - 1)))
    result = torch.zeros_like(core)
    for permutation in permutations:
        result += core.permute(0, *(role + 1 for role in permutation))
    return result / len(permutations)


def global_index(token: int, local: int, feature: int) -> int:
    return token * feature + local


@torch.no_grad()
def compile_attention_core(
    block: ChiTransformerBlock,
    tokens: int,
    *,
    symmetrize: bool,
    maximum_elements: int,
) -> torch.Tensor:
    dimension = block.attn.dim
    feature = dimension + 1
    hidden = tokens * feature
    elements = hidden ** 6
    if elements > maximum_elements:
        raise ValueError(f"attention core needs {elements} elements above cap {maximum_elements}")
    mask = causal_mask(tokens, dtype=DTYPE)
    compiled = compile_bilinear_attention(
        block.attn,
        n_query=tokens,
        mask=mask,
        dense_oracle_maximum_elements=maximum_elements,
    )
    if compiled.head_cores is None:
        raise ValueError("dense head core unavailable")
    core = block.attn.wo.weight.new_zeros((hidden,) + (hidden,) * 5)
    gain = block.attn_gain.detach().reshape(())
    for token in range(tokens):
        constant = global_index(token, 0, feature)
        core[(constant, constant, constant, constant, constant, constant)] = 1.0
        if block.residual:
            for output in range(dimension):
                row = global_index(token, output + 1, feature)
                core[(row, row, constant, constant, constant, constant)] = 1.0
        for output in range(dimension):
            row = global_index(token, output + 1, feature)
            core[(row, constant, constant, constant, constant, constant)] += (
                gain * compiled.output_bias[output]
            )
    for route in compiled.routes:
        head_slice = slice(
            route.head_index * compiled.head_dim,
            (route.head_index + 1) * compiled.head_dim,
        )
        output_weight = compiled.output_weight[:, head_slice]
        role_tokens = (
            route.query_index,
            route.query_index,
            route.source_index,
            route.source_index,
            route.source_index,
        )
        for value_coordinate in range(compiled.head_dim):
            local_core = compiled.head_cores[route.head_index, value_coordinate]
            for local_indices in itertools.product(range(feature), repeat=5):
                coefficient = local_core[local_indices]
                if float(coefficient.item()) == 0.0:
                    continue
                indices = tuple(
                    global_index(token, local, feature)
                    for token, local in zip(role_tokens, local_indices)
                )
                for output in range(dimension):
                    row = global_index(route.query_index, output + 1, feature)
                    core[(row, *indices)] += (
                        gain * route.scale * output_weight[output, value_coordinate] * coefficient
                    )
    return symmetrize_inputs(core) if symmetrize else core


@torch.no_grad()
def compile_ffn_core(
    block: ChiTransformerBlock,
    tokens: int,
    *,
    maximum_elements: int,
) -> torch.Tensor:
    dimension = block.ffn.dim
    feature = dimension + 1
    hidden = tokens * feature
    if hidden**3 > maximum_elements:
        raise ValueError("FFN core exceeds cap")
    if block.residual:
        local = compile_bilinear_ffn_residual_core(
            block.ffn,
            gain=float(block.ffn_gain.item()),
            maximum_elements=maximum_elements,
        )
    else:
        left = torch.cat((block.ffn.left.bias[:, None], block.ffn.left.weight), dim=1)
        right = torch.cat((block.ffn.right.bias[:, None], block.ffn.right.weight), dim=1)
        update = torch.einsum("or,ri,rj->oij", block.ffn.down.weight, left, right)
        if block.ffn.down.bias is not None:
            update[:, 0, 0] += block.ffn.down.bias
        update = 0.5 * (update + update.transpose(1, 2))
        local = update.new_zeros(feature, feature, feature)
        local[0, 0, 0] = 1.0
        local[1:] = block.ffn_gain * update
    core = local.new_zeros(hidden, hidden, hidden)
    for token in range(tokens):
        for output in range(feature):
            for left in range(feature):
                for right in range(feature):
                    core[
                        global_index(token, output, feature),
                        global_index(token, left, feature),
                        global_index(token, right, feature),
                    ] = local[output, left, right]
    return core


def position_embedding_matrix(position: torch.Tensor) -> torch.Tensor:
    tokens, dimension = position.shape
    feature = dimension + 1
    matrix = position.new_zeros(tokens * feature, tokens * feature)
    for token in range(tokens):
        offset = token * feature
        matrix[offset, offset] = 1.0
        matrix[offset + 1 : offset + feature, offset] = position[token]
        matrix[offset + 1 : offset + feature, offset + 1 : offset + feature] = torch.eye(
            dimension, dtype=position.dtype, device=position.device
        )
    return matrix


def make_blocks(
    seed: int,
    dimension: int,
    depth: int,
    case: Case,
) -> tuple[ChiTransformerBlock, ...]:
    torch.manual_seed(seed)
    blocks = tuple(
        ChiTransformerBlock(
            dim=dimension,
            n_heads=1,
            ffn_rank=max(3, 3 * dimension),
            n_layers=depth,
            causal=True,
            norm="none",
            qk_norm="none",
            residual=case.residual,
        ).double().eval()
        for _ in range(depth)
    )
    generator = torch.Generator().manual_seed(10_000 + seed)
    with torch.no_grad():
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear) and module.bias is not None:
                    if case.affine_biases:
                        module.bias.normal_(std=0.2, generator=generator)
                    else:
                        module.bias.zero_()
    return blocks


@torch.no_grad()
def run_case(
    *,
    seed: int,
    tokens: int,
    dimension: int,
    depth: int,
    case: Case,
    tolerance: float,
    maximum_elements: int,
) -> dict[str, object]:
    blocks = make_blocks(seed, dimension, depth, case)
    generator = torch.Generator().manual_seed(20_000 + seed)
    position = (
        torch.randn(tokens, dimension, generator=generator, dtype=DTYPE) * 0.3
        if case.position_embedding
        else torch.zeros(tokens, dimension, dtype=DTYPE)
    )
    embedding = position_embedding_matrix(position)
    cores = []
    for block in blocks:
        cores.append(
            compile_attention_core(
                block,
                tokens,
                symmetrize=case.symmetrize_attention,
                maximum_elements=maximum_elements,
            )
        )
        cores.append(compile_ffn_core(block, tokens, maximum_elements=maximum_elements))

    raw_x = torch.randn(3, tokens, dimension, generator=generator, dtype=DTYPE)
    direct = raw_x + position
    for block in blocks:
        direct = block(direct, mask=causal_mask(tokens, dtype=DTYPE), method="explicit")
    homogeneous = torch.cat((torch.ones(3, tokens, 1, dtype=DTYPE), raw_x), dim=-1).reshape(3, -1)
    state = homogeneous @ embedding.T
    for core in cores:
        state = evaluate_core(core, state)
    compiled = state.reshape(3, tokens, dimension + 1)[..., 1:]
    replay_error = relative_error(compiled, direct)

    feature = dimension + 1
    embedding_spectrum = matrix_kron_spectrum(embedding, tokens, feature)
    factor, quotient = reduced_rq_rows(embedding)
    rows = [
        {
            "stage": "embedding",
            "raw_kron": spectrum_record(embedding_spectrum, tolerance),
            "emitted_r_kron": spectrum_record(
                matrix_kron_spectrum(factor, tokens, feature), tolerance
            ),
            "row_isometry_error": relative_error(
                quotient @ quotient.T,
                torch.eye(quotient.shape[0], dtype=DTYPE),
            ),
            "factorization_error": relative_error(factor @ quotient, embedding),
        }
    ]
    canonical_state = homogeneous @ quotient.T
    transformed_cores = [core.detach().clone() for core in cores]
    transformed_cores[0] = apply_input_factor(transformed_cores[0], factor)
    emitted_r_ranks = [numerical_rank(matrix_kron_spectrum(factor, tokens, feature), tolerance)]
    transformed_core_ranks = []
    raw_core_ranks = []
    for index, transformed in enumerate(transformed_cores):
        raw_spectrum = core_kron_spectrum(cores[index], tokens, feature)
        transformed_spectrum = core_kron_spectrum(transformed, tokens, feature)
        raw_core_ranks.append(numerical_rank(raw_spectrum, tolerance))
        transformed_core_ranks.append(numerical_rank(transformed_spectrum, tolerance))
        matrix = transformed.reshape(transformed.shape[0], -1)
        factor, quotient = reduced_rq_rows(matrix)
        quotient_core = quotient.reshape_as(transformed)
        canonical_state = evaluate_core(quotient_core, canonical_state)
        r_spectrum = matrix_kron_spectrum(factor, tokens, feature)
        emitted_r_ranks.append(numerical_rank(r_spectrum, tolerance))
        rows.append(
            {
                "stage": f"block{index // 2}.{'attention' if index % 2 == 0 else 'ffn'}",
                "arity": transformed.ndim - 1,
                "raw_core_kron": spectrum_record(raw_spectrum, tolerance),
                "transformed_core_kron": spectrum_record(transformed_spectrum, tolerance),
                "emitted_r_kron": spectrum_record(r_spectrum, tolerance),
                "quotient_core_kron": spectrum_record(
                    core_kron_spectrum(quotient_core, tokens, feature), tolerance
                ),
                "row_isometry_error": relative_error(
                    quotient @ quotient.T,
                    torch.eye(quotient.shape[0], dtype=DTYPE),
                ),
                "factorization_error": relative_error(factor @ quotient, matrix),
            }
        )
        if index + 1 < len(transformed_cores):
            transformed_cores[index + 1] = apply_input_factor(
                transformed_cores[index + 1], factor
            )

    canonical_replay = canonical_state @ factor.T
    canonical_replay_error = relative_error(canonical_replay, state)
    closure_after_embedding = all(rank == 1 for rank in emitted_r_ranks)
    return {
        "case": case.name,
        "tokens": tokens,
        "dimension": dimension,
        "homogeneous_feature_dimension": feature,
        "homogeneous_constants": True,
        "depth": depth,
        "residual": case.residual,
        "symmetrize_attention": case.symmetrize_attention,
        "affine_biases": case.affine_biases,
        "position_embedding": case.position_embedding,
        "tied_forward_relative_error": replay_error,
        "canonical_replay_relative_error": canonical_replay_error,
        "emitted_r_kron_ranks": emitted_r_ranks,
        "raw_core_kron_ranks": raw_core_ranks,
        "transformed_core_kron_ranks": transformed_core_ranks,
        "rank_one_r_closure": closure_after_embedding,
        "maximum_r_kron_rank": max(emitted_r_ranks),
        "maximum_transformed_core_kron_rank": max(transformed_core_ranks),
        "stages": rows,
    }


def parse_grid(value: str) -> tuple[tuple[int, int, int], ...]:
    result = []
    for item in value.split(","):
        fields = item.split("x")
        if len(fields) != 3:
            raise ValueError("grid entries must have NxDxL form")
        triple = tuple(int(field) for field in fields)
        if any(field <= 0 for field in triple):
            raise ValueError("grid values must be positive")
        result.append(triple)
    return tuple(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="2x1x1,2x1x2,2x2x1,2x2x2,3x1x1,3x1x2")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rank-tolerance", type=float, default=1e-10)
    parser.add_argument("--maximum-elements", type=int, default=5_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    grid = parse_grid(args.grid)
    if args.seeds <= 0 or not 0 < args.rank_tolerance < 1 or args.maximum_elements <= 0:
        raise ValueError("invalid seeds, tolerance, or element cap")
    started = time.perf_counter()
    rows = [
        run_case(
            seed=seed,
            tokens=tokens,
            dimension=dimension,
            depth=depth,
            case=case,
            tolerance=args.rank_tolerance,
            maximum_elements=args.maximum_elements,
        )
        for tokens, dimension, depth in grid
        for seed in range(args.seeds)
        for case in CASES
    ]
    full_cases = [row for row in rows if row["case"] == "symmetric_residual_bias_pos"]
    ideal_cases = [row for row in rows if row["case"] == "ordered_nores_nobias_nopos"]
    gates = {
        "all_tied_forwards_exact": max(float(row["tied_forward_relative_error"]) for row in rows) < 1e-9,
        "all_canonical_replays_exact": max(
            float(row["canonical_replay_relative_error"]) for row in rows
        ) < 1e-9,
        "all_factorizations_exact": max(
            float(stage["factorization_error"])
            for row in rows
            for stage in row["stages"]
        ) < 1e-9,
        "all_rows_isometric": max(
            float(stage["row_isometry_error"])
            for row in rows
            for stage in row["stages"]
        ) < 1e-9,
        "full_variant_was_adversarially_measured": bool(full_cases),
        "ideal_variant_was_measured": bool(ideal_cases),
    }
    result = {
        "schema": "xvla-kron-closure-audit-v1",
        "object": (
            "actual reduced-RQ factors on per-token homogeneous residual attention and FFN cores"
        ),
        "rank_definition": "Van Loan-Pitsianis token-versus-feature rearrangement SVD",
        "rank_tolerance_relative": args.rank_tolerance,
        "grid": grid,
        "seeds": args.seeds,
        "case_names": [case.name for case in CASES],
        "elapsed_seconds": time.perf_counter() - started,
        "gates": gates,
        "all_correctness_gates_pass": all(gates.values()),
        "finding": {
            "full_variant_rank_one_closure_fraction": sum(
                bool(row["rank_one_r_closure"]) for row in full_cases
            ) / len(full_cases),
            "full_variant_maximum_r_kron_rank": max(
                int(row["maximum_r_kron_rank"]) for row in full_cases
            ),
            "full_variant_maximum_transformed_core_kron_rank": max(
                int(row["maximum_transformed_core_kron_rank"]) for row in full_cases
            ),
            "ideal_variant_rank_one_closure_fraction": sum(
                bool(row["rank_one_r_closure"]) for row in ideal_cases
            ) / len(ideal_cases),
        },
        "rows": rows,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    if not result["all_correctness_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
