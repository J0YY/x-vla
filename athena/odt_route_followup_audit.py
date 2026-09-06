#!/usr/bin/env python3
"""Gauge-invariant and rank-resolved follow-up to the ODT route audit."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from athena.extended_abstract_weight_audit_v1 import (
    canonicalize_columns,
    linear_arrays,
    numeric_summary,
    optimal_assignment,
    upper_tail_p,
)
from athena.odt_route_criterion_audit import (
    EXPECTED_CHECKPOINTS,
    load_model,
    score_matrix,
    sha256,
)
from xvla.train.exact_odt_attention_proto import build_gram_reduced_head, hat


SCHEMA = "xvla-odt-route-followup-audit-v1"
BLOCKS = tuple(range(8))
RANK_CONFIGS = (
    (8, 1),
    (16, 1),
    (32, 3),
    (64, 5),
    (96, 8),
    (128, 11),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--haar-draws", type=int, default=128)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def left_singular_basis(matrix: np.ndarray) -> np.ndarray:
    vectors, _, _ = np.linalg.svd(matrix, full_matrices=False)
    return canonicalize_columns(vectors)


def extract_systems(model) -> dict[int, dict[int, dict[str, Any]]]:
    systems: dict[int, dict[int, dict[str, Any]]] = {}
    for block_index in BLOCKS:
        attention = model.backbone.blocks[block_index].attn
        q1w, q1b = linear_arrays(attention.wq1)
        k1w, k1b = linear_arrays(attention.wk1)
        q2w, q2b = linear_arrays(attention.wq2)
        k2w, k2b = linear_arrays(attention.wk2)
        vw, vb = linear_arrays(attention.wv)
        ow, _ = linear_arrays(attention.wo)
        systems[block_index] = {}
        for head in range(attention.n_heads):
            start, stop = head * attention.head_dim, (head + 1) * attention.head_dim
            sl = slice(start, stop)
            gram = build_gram_reduced_head(
                q1w[sl], q1b[sl], k1w[sl], k1b[sl], q2w[sl], q2b[sl],
                k2w[sl], k2b[sl], vw[sl], vb[sl], ow[:, sl],
            )
            values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
            order = np.argsort(values)[::-1]
            values = values[order]
            vectors = canonicalize_columns(vectors[:, order])
            wo = ow[:, sl]
            value_transport = wo @ hat(vw[sl], vb[sl])
            systems[block_index][head] = {
                "input_vectors": vectors,
                "input_values": values,
                "wo_output_vectors": left_singular_basis(wo),
                "value_transport_output_vectors": left_singular_basis(value_transport),
            }
    return systems


def bases(
    systems: dict[int, dict[int, dict[str, Any]]],
    block: int,
    definition: str,
    input_rank: int,
    output_rank: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    inputs = [systems[block][head]["input_vectors"][:, :input_rank] for head in range(12)]
    key = f"{definition}_output_vectors"
    outputs = [systems[block][head][key][:, :output_rank] for head in range(12)]
    return inputs, outputs


def observed(scores: np.ndarray) -> dict[str, Any]:
    optimal, assignment = optimal_assignment(scores)
    return {
        "all_pair_mean": float(scores.mean()),
        "matched_head_mean": float(np.diag(scores).mean()),
        "optimal_assignment_mean": float(optimal),
        "optimal_source_to_target_head": assignment,
        "minimum_cell": float(scores.min()),
        "maximum_cell": float(scores.max()),
    }


def null_calibration(
    target_inputs: list[np.ndarray],
    output_rank: int,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    all_pair = []
    matched = []
    optimal = []
    for _ in range(draws):
        outputs = [
            np.linalg.qr(rng.normal(size=(384, output_rank)), mode="reduced")[0]
            for _ in range(12)
        ]
        scores = score_matrix(outputs, target_inputs)
        all_pair.append(float(scores.mean()))
        matched.append(float(np.diag(scores).mean()))
        optimal.append(float(optimal_assignment(scores)[0]))
    return {
        "draws": draws,
        "all_pair": np.asarray(all_pair),
        "matched": np.asarray(matched),
        "optimal": np.asarray(optimal),
    }


def calibrated_record(observation: dict[str, Any], null: dict[str, Any]) -> dict[str, Any]:
    return {
        **observation,
        "all_pair_haar_upper_tail_p": upper_tail_p(
            observation["all_pair_mean"], null["all_pair"]
        ),
        "matched_head_haar_upper_tail_p": upper_tail_p(
            observation["matched_head_mean"], null["matched"]
        ),
        "optimal_assignment_haar_upper_tail_p": upper_tail_p(
            observation["optimal_assignment_mean"], null["optimal"]
        ),
    }


def transitions() -> list[tuple[int, int]]:
    ordered = []
    for source in range(7):
        ordered.extend(((source, source + 1), (source + 1, source)))
    ordered.extend(((0, 2), (2, 0), (2, 4), (4, 2), (0, 4), (4, 0)))
    return ordered


def rank_spectral_mass(systems: dict[int, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    result = {}
    for block in BLOCKS:
        result[str(block)] = {}
        for input_rank, _ in RANK_CONFIGS:
            values = []
            for head in range(12):
                eigenvalues = np.clip(systems[block][head]["input_values"], 0.0, None)
                values.append(
                    float(eigenvalues[:input_rank].sum() / max(eigenvalues.sum(), 1e-300))
                )
            result[str(block)][str(input_rank)] = numeric_summary(np.asarray(values))
    return result


def main() -> None:
    args = parse_args()
    digest = sha256(args.checkpoint)
    if digest != EXPECTED_CHECKPOINTS[args.seed]:
        raise RuntimeError(f"Checkpoint digest mismatch: {digest}")
    if args.haar_draws < 32:
        raise ValueError("haar-draws must be at least 32")

    model = load_model(args.checkpoint)
    systems = extract_systems(model)
    pairs = transitions()
    rank_sweep: dict[str, Any] = {}
    for definition in ("wo", "value_transport"):
        by_definition = {}
        for input_rank, output_rank in RANK_CONFIGS:
            rows = {}
            for source, target in pairs:
                target_inputs, _ = bases(
                    systems, target, definition, input_rank, output_rank
                )
                _, source_outputs = bases(
                    systems, source, definition, input_rank, output_rank
                )
                row = observed(score_matrix(source_outputs, target_inputs))
                row["haar_expectation"] = input_rank / 384
                row["all_pair_excess_over_haar_expectation"] = (
                    row["all_pair_mean"] - input_rank / 384
                )
                rows[f"{source}_to_{target}"] = row
            by_definition[f"input_{input_rank}_output_{output_rank}"] = rows
        rank_sweep[definition] = by_definition

    formal: dict[str, Any] = {}
    for output_rank in (11, 32):
        nulls = {}
        for target in BLOCKS:
            target_inputs, _ = bases(systems, target, "wo", 128, output_rank)
            nulls[target] = null_calibration(
                target_inputs, output_rank, args.haar_draws,
                20260830 + args.seed * 10000 + target * 100 + output_rank,
            )
        for definition in ("wo", "value_transport"):
            rows = {}
            for source, target in pairs:
                target_inputs, _ = bases(systems, target, definition, 128, output_rank)
                _, source_outputs = bases(systems, source, definition, 128, output_rank)
                rows[f"{source}_to_{target}"] = calibrated_record(
                    observed(score_matrix(source_outputs, target_inputs)), nulls[target]
                )
            formal[f"{definition}_input_128_output_{output_rank}"] = rows
        formal[f"haar_input_128_output_{output_rank}"] = {
            str(target): {
                "all_pair": numeric_summary(nulls[target]["all_pair"]),
                "matched": numeric_summary(nulls[target]["matched"]),
                "optimal": numeric_summary(nulls[target]["optimal"]),
            }
            for target in BLOCKS
        }

    full_rank_definition_difference = {}
    for source, target in pairs:
        target_wo, _ = bases(systems, target, "wo", 128, 32)
        _, source_wo = bases(systems, source, "wo", 128, 32)
        target_hvo, _ = bases(systems, target, "value_transport", 128, 32)
        _, source_hvo = bases(systems, source, "value_transport", 128, 32)
        full_rank_definition_difference[f"{source}_to_{target}"] = {
            "score_matrix_max_abs": float(
                np.max(np.abs(
                    score_matrix(source_wo, target_wo)
                    - score_matrix(source_hvo, target_hvo)
                ))
            )
        }

    payload = {
        "schema": SCHEMA,
        "checkpoint": {"path": str(args.checkpoint), "seed": args.seed, "sha256": digest},
        "definitions": {
            "wo": "Leading left singular vectors of the per-head W_o slice. Rank truncation is gauge-dependent.",
            "value_transport": (
                "Leading left singular vectors of W_o [b_v | W_v]. This product is invariant "
                "under exact value-coordinate changes, but it is not the full rational head tensor."
            ),
            "full_rank_32": (
                "The complete column space of W_o. This is gauge-invariant and should agree with "
                "the full-rank value-transport column space when [b_v | W_v] has full row rank."
            ),
        },
        "rank_configs": [
            {"input_rank": input_rank, "output_rank": output_rank}
            for input_rank, output_rank in RANK_CONFIGS
        ],
        "transitions": [list(pair) for pair in pairs],
        "input_rank_spectral_mass": rank_spectral_mass(systems),
        "rank_sweep": rank_sweep,
        "formal_haar_calibration": formal,
        "full_rank_definition_difference": full_rank_definition_difference,
        "scope": (
            "This audit tests whether the negative route result survives gauge-invariant local "
            "output definitions, rank choices, adjacent-block comparisons, reverse directions, "
            "and conditional Haar controls. It still does not compose the intervening nonlinear "
            "blocks or include rational numerator and denominator factors in one global ODT object."
        ),
    }
    atomic_json(args.output, payload)
    print(json.dumps({
        "output": str(args.output),
        "seed": args.seed,
        "selected": {
            key: formal["value_transport_input_128_output_11"][key]
            for key in ("0_to_2", "2_to_4", "2_to_0", "4_to_2")
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
