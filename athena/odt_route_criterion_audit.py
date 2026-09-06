#!/usr/bin/env python3
"""Adversarial audit of the local-attention route criterion.

This script separates three questions that were previously conflated:

1. Does the reduced attention Gram reproduce the stated tensor contraction?
2. Can the extraction recover a planted cross-block route?
3. Is the cross-block score invariant under an exact value-space change of basis?

The third check is load-bearing.  For one attention head, replacing

    V by B V, and W_o by W_o B^{-1}

leaves the deployed head function unchanged.  A circuit statistic should therefore
not change.  The current input Gram is invariant because it depends on W_o V, but
the rank-truncated output SVD need not be invariant.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from athena.extended_abstract_weight_audit_v1 import (
    BLOCKS,
    INPUT_RANK,
    OUTPUT_RANK,
    attention_bases,
    canonicalize_columns,
    linear_arrays,
    orthonormalize,
    subspace_record,
)
from athena.run_global_bond_rank_audit import DATASET_TASKS
from athena.run_xvla_experiment import build_vocab, make_config
from xvla.models.vla import ChiVLA
from xvla.train.exact_odt_attention_proto import (
    build_gram_reduced_head,
    verify_reduction_vs_bruteforce,
)


SCHEMA = "xvla-odt-route-criterion-audit-v1"
EXPECTED_CHECKPOINTS = {
    0: "96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9",
    1: "cc0780b989165a80a449c03cbb3574e2f56b2e5747a64cf32d1fde418df8ec3c",
    2: "413a770071bd8f16b924c5604f7c7a58567d2eb6c40c66aab202bcf046fec910",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gauge-scale", type=float, default=4.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def random_basis(rng: np.random.Generator, dim: int, rank: int) -> np.ndarray:
    basis, _ = np.linalg.qr(rng.normal(size=(dim, rank)), mode="reduced")
    return canonicalize_columns(basis)


def extract_input_basis(head: dict[str, np.ndarray], rank: int) -> tuple[np.ndarray, np.ndarray]:
    gram = build_gram_reduced_head(
        head["wq1"], head["bq1"], head["wk1"], head["bk1"],
        head["wq2"], head["bq2"], head["wk2"], head["bk2"],
        head["wv"], head["bv"], head["wo"],
    )
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(values)[::-1]
    return canonicalize_columns(vectors[:, order[:rank]]), values[order]


def extract_output_basis(wo: np.ndarray, rank: int) -> np.ndarray:
    values, vectors = np.linalg.eigh(wo.T @ wo)
    order = np.argsort(values)[::-1]
    local = canonicalize_columns(vectors[:, order[:rank]])
    return orthonormalize(wo @ local, rank)


def make_planted_head(
    rng: np.random.Generator,
    dim: int,
    head_dim: int,
    input_basis: np.ndarray,
    output_basis: np.ndarray,
) -> dict[str, np.ndarray]:
    input_rank = input_basis.shape[1]
    output_rank = output_basis.shape[1]

    def projected_rows() -> np.ndarray:
        return rng.normal(size=(head_dim, input_rank)) @ input_basis.T

    return {
        "wq1": projected_rows(),
        "bq1": np.zeros(head_dim),
        "wk1": projected_rows(),
        "bk1": np.zeros(head_dim),
        "wq2": projected_rows(),
        "bq2": np.zeros(head_dim),
        "wk2": projected_rows(),
        "bk2": np.zeros(head_dim),
        "wv": projected_rows(),
        "bv": np.zeros(head_dim),
        "wo": output_basis @ rng.normal(size=(output_rank, head_dim)),
    }


def score_matrix(outputs: list[np.ndarray], inputs: list[np.ndarray]) -> np.ndarray:
    scores = np.empty((len(outputs), len(inputs)), dtype=np.float64)
    for source, output in enumerate(outputs):
        for target, input_basis in enumerate(inputs):
            scores[source, target] = subspace_record(output, input_basis)[
                "mean_squared_canonical_correlation"
            ]
    return scores


def planted_route_audit() -> dict[str, Any]:
    rng = np.random.default_rng(20260830)
    dim, heads, head_dim, rank = 24, 3, 8, 4
    block_inputs: dict[int, list[np.ndarray]] = {block: [] for block in BLOCKS}
    block_outputs: dict[int, list[np.ndarray]] = {block: [] for block in BLOCKS}
    systems: dict[int, list[dict[str, np.ndarray]]] = {block: [] for block in BLOCKS}

    input_zero = [random_basis(rng, dim, rank) for _ in range(heads)]
    output_zero = [random_basis(rng, dim, rank) for _ in range(heads)]
    output_two = [random_basis(rng, dim, rank) for _ in range(heads)]
    output_four = [random_basis(rng, dim, rank) for _ in range(heads)]
    planted_inputs = {0: input_zero, 2: output_zero, 4: output_two}
    planted_outputs = {0: output_zero, 2: output_two, 4: output_four}

    minimum_positive_mass = 1.0
    for block in BLOCKS:
        for head_index in range(heads):
            head = make_planted_head(
                rng, dim, head_dim,
                planted_inputs[block][head_index],
                planted_outputs[block][head_index],
            )
            systems[block].append(head)
            input_basis, values = extract_input_basis(head, rank)
            output_basis = extract_output_basis(head["wo"], rank)
            block_inputs[block].append(input_basis)
            block_outputs[block].append(output_basis)
            positive = np.clip(values, 0.0, None)
            minimum_positive_mass = min(
                minimum_positive_mass,
                float(positive[:rank].sum() / max(positive.sum(), 1e-300)),
            )

    direct_02 = score_matrix(block_outputs[0], block_inputs[2])
    direct_24 = score_matrix(block_outputs[2], block_inputs[4])

    rotation, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    rotated_inputs = [rotation @ basis for basis in block_inputs[2]]
    naive_rotated = score_matrix(block_outputs[0], rotated_inputs)
    transported = score_matrix(
        [rotation @ basis for basis in block_outputs[0]], rotated_inputs
    )

    return {
        "geometry": {"dim": dim, "heads": heads, "head_dim": head_dim, "rank": rank},
        "minimum_input_rank_spectral_mass": minimum_positive_mass,
        "planted_0_to_2": {
            "score_matrix": direct_02.tolist(),
            "matched_mean": float(np.diag(direct_02).mean()),
        },
        "planted_2_to_4": {
            "score_matrix": direct_24.tolist(),
            "matched_mean": float(np.diag(direct_24).mean()),
        },
        "intervening_rotation_counterexample": {
            "construction": (
                "An exact orthogonal transport is inserted between the source output and target "
                "input. The identity-coordinate test ignores that transport."
            ),
            "naive_matched_mean": float(np.diag(naive_rotated).mean()),
            "transport_aware_matched_mean": float(np.diag(transported).mean()),
            "haar_expectation": rank / dim,
        },
    }


def load_model(checkpoint: Path) -> ChiVLA:
    vocab, _ = build_vocab(DATASET_TASKS)
    config = make_config(
        "chi", len(vocab), state_dim=8, action_dim=7, res=64, horizon=8,
        vision_encoder="vit",
    )
    model = ChiVLA(config).cpu().double()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    return model


def mean_projector(bases: list[np.ndarray]) -> np.ndarray:
    return sum(basis @ basis.T for basis in bases) / len(bases)


def gauge_head(
    wo: np.ndarray,
    wv: np.ndarray,
    bv: np.ndarray,
    target_projector: np.ndarray,
    rank: int,
    scale: float,
    maximize: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    column_basis, triangular = np.linalg.qr(wo, mode="reduced")
    restricted = column_basis.T @ target_projector @ column_basis
    values, vectors = np.linalg.eigh(0.5 * (restricted + restricted.T))
    order = np.argsort(values)[::-1] if maximize else np.argsort(values)
    vectors = canonicalize_columns(vectors[:, order])
    singular_scales = np.ones(wo.shape[1])
    singular_scales[:rank] = scale
    desired = column_basis @ vectors @ np.diag(singular_scales)
    inverse_gauge = np.linalg.solve(triangular, vectors @ np.diag(singular_scales))
    gauge = np.linalg.inv(inverse_gauge)
    transformed_wo = desired
    transformed_wv = gauge @ wv
    transformed_bv = gauge @ bv

    product_error = float(np.max(np.abs(transformed_wo @ transformed_wv - wo @ wv)))
    bias_error = float(np.max(np.abs(transformed_wo @ transformed_bv - wo @ bv)))
    inputs = rng.normal(size=(17, wv.shape[1]))
    original = (inputs @ wv.T + bv) @ wo.T
    transformed = (inputs @ transformed_wv.T + transformed_bv) @ transformed_wo.T
    functional_error = float(np.max(np.abs(original - transformed)))
    return transformed_wo, transformed_wv, transformed_bv, {
        "gauge_condition_number": float(np.linalg.cond(gauge)),
        "wo_wv_product_max_abs_error": product_error,
        "wo_bv_product_max_abs_error": bias_error,
        "random_value_transport_max_abs_error": functional_error,
    }


def actual_checkpoint_gauge_audit(
    model: ChiVLA, systems: dict[int, dict[int, dict[str, Any]]], scale: float
) -> dict[str, Any]:
    rng = np.random.default_rng(20260831)
    result: dict[str, Any] = {}
    for source_block, target_block in ((0, 2), (2, 4)):
        target_inputs = [systems[target_block][head]["input"] for head in range(12)]
        target_mean = mean_projector(target_inputs)
        base_outputs = [systems[source_block][head]["output_lifted"] for head in range(12)]
        base_scores = score_matrix(base_outputs, target_inputs)
        source_attention = model.backbone.blocks[source_block].attn
        gauged_modules = {
            "maximize": copy.deepcopy(source_attention).cpu().double().eval(),
            "minimize": copy.deepcopy(source_attention).cpu().double().eval(),
        }
        full_outputs = []
        maximum_outputs = []
        minimum_outputs = []
        diagnostics = {"maximize": [], "minimize": []}
        gram_errors = {"maximize": [], "minimize": []}

        for head in range(12):
            start, stop = head * source_attention.head_dim, (head + 1) * source_attention.head_dim
            sl = slice(start, stop)
            q1w, q1b = linear_arrays(source_attention.wq1)
            k1w, k1b = linear_arrays(source_attention.wk1)
            q2w, q2b = linear_arrays(source_attention.wq2)
            k2w, k2b = linear_arrays(source_attention.wk2)
            v_w, v_b = linear_arrays(source_attention.wv)
            o_w, _ = linear_arrays(source_attention.wo)
            wo, wv, bv = o_w[:, sl], v_w[sl], v_b[sl]
            full_outputs.append(extract_output_basis(wo, source_attention.head_dim))
            original_gram = build_gram_reduced_head(
                q1w[sl], q1b[sl], k1w[sl], k1b[sl], q2w[sl], q2b[sl],
                k2w[sl], k2b[sl], wv, bv, wo,
            )

            for label, maximize, collection in (
                ("maximize", True, maximum_outputs),
                ("minimize", False, minimum_outputs),
            ):
                new_wo, new_wv, new_bv, audit = gauge_head(
                    wo, wv, bv, target_mean, OUTPUT_RANK, scale, maximize, rng
                )
                collection.append(extract_output_basis(new_wo, OUTPUT_RANK))
                transformed_gram = build_gram_reduced_head(
                    q1w[sl], q1b[sl], k1w[sl], k1b[sl], q2w[sl], q2b[sl],
                    k2w[sl], k2b[sl], new_wv, new_bv, new_wo,
                )
                denominator = max(float(np.linalg.norm(original_gram)), 1e-300)
                gram_errors[label].append(
                    float(np.linalg.norm(transformed_gram - original_gram) / denominator)
                )
                diagnostics[label].append(audit)
                with torch.no_grad():
                    module = gauged_modules[label]
                    module.wv.weight[sl].copy_(torch.from_numpy(new_wv))
                    module.wv.bias[sl].copy_(torch.from_numpy(new_bv))
                    module.wo.weight[:, sl].copy_(torch.from_numpy(new_wo))

        generator = torch.Generator(device="cpu").manual_seed(20260831 + source_block)
        replay_input = torch.randn(
            3, 9, source_attention.dim, dtype=torch.float64, generator=generator
        )
        replay_mask = torch.tril(torch.ones(9, 9, dtype=torch.float64))
        with torch.inference_mode():
            reference_output = source_attention(
                replay_input, mask=replay_mask, method="explicit"
            )
            module_errors = {
                label: float(
                    (
                        module(replay_input, mask=replay_mask, method="explicit")
                        - reference_output
                    ).abs().max()
                )
                for label, module in gauged_modules.items()
            }

        maximum_scores = score_matrix(maximum_outputs, target_inputs)
        minimum_scores = score_matrix(minimum_outputs, target_inputs)
        full_scores = score_matrix(full_outputs, target_inputs)
        result[f"{source_block}_to_{target_block}"] = {
            "base_rank_11_all_pair_mean": float(base_scores.mean()),
            "same_function_maximized_rank_11_all_pair_mean": float(maximum_scores.mean()),
            "same_function_minimized_rank_11_all_pair_mean": float(minimum_scores.mean()),
            "gauge_invariant_full_rank_32_all_pair_mean": float(full_scores.mean()),
            "rank_11_score_range_under_exact_value_gauge": float(
                maximum_scores.mean() - minimum_scores.mean()
            ),
            "maximum_input_gram_relative_error": float(
                max(max(gram_errors["maximize"]), max(gram_errors["minimize"]))
            ),
            "maximum_functional_error": float(
                max(
                    row["random_value_transport_max_abs_error"]
                    for rows in diagnostics.values() for row in rows
                )
            ),
            "torch_attention_module_max_abs_error": module_errors,
            "maximum_product_error": float(
                max(
                    max(row["wo_wv_product_max_abs_error"], row["wo_bv_product_max_abs_error"])
                    for rows in diagnostics.values() for row in rows
                )
            ),
            "maximum_gauge_condition_number": float(
                max(row["gauge_condition_number"] for rows in diagnostics.values() for row in rows)
            ),
            "per_head_diagnostics": diagnostics,
        }
    return result


def main() -> None:
    args = parse_args()
    checkpoint_digest = sha256(args.checkpoint)
    if checkpoint_digest != EXPECTED_CHECKPOINTS[args.seed]:
        raise RuntimeError(
            f"Checkpoint digest mismatch: expected {EXPECTED_CHECKPOINTS[args.seed]}, got {checkpoint_digest}"
        )
    if args.gauge_scale <= 1.0:
        raise ValueError("gauge-scale must be greater than one")

    model = load_model(args.checkpoint)
    systems, _ = attention_bases(model)
    payload = {
        "schema": SCHEMA,
        "checkpoint": {
            "path": str(args.checkpoint),
            "seed": args.seed,
            "sha256": checkpoint_digest,
        },
        "reduced_gram_bruteforce_max_abs_error": verify_reduction_vs_bruteforce(
            dim_full=8, n_heads=2, seed=0
        ),
        "planted_route": planted_route_audit(),
        "actual_checkpoint_value_gauge": actual_checkpoint_gauge_audit(
            model, systems, args.gauge_scale
        ),
        "interpretation": {
            "coding_check": (
                "The reduced Gram must match brute force and recover the identity-coordinate planted route."
            ),
            "criterion_check": (
                "The rank-11 route score should not move under V -> B V and W_o -> W_o B^{-1}, "
                "because this leaves the deployed attention function unchanged."
            ),
            "skip_block_check": (
                "A direct principal-angle comparison is not necessary for a route when an intervening "
                "block transports the representation."
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "hostname": platform.node(),
            "gauge_scale": args.gauge_scale,
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
