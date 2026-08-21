#!/usr/bin/env python3
"""Validate and score the frozen three-checkpoint local instruction experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from athena.run_local_instruction_specificity import (
    PROTOCOL,
    SCHEMA,
    file_sha256,
    hash_array,
    specificity_metrics,
    target_phrase,
)
from athena.run_xvla_experiment import (
    build_encoder,
    build_vocab,
    load_suite,
    readable_object_name,
    task_languages,
)


EXPECTED_CACHE_SHA256 = "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662"
EXPECTED_PROVENANCE_JOB_ID = "830988"
EXPECTED_CHECKPOINTS = {
    0: {
        "basename": "ckpt_linear_rat_vit_s0_v2.pt",
        "sha256": "96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9",
    },
    1: {
        "basename": "ckpt_linear_rat_vit_s1.pt",
        "sha256": "cc0780b989165a80a449c03cbb3574e2f56b2e5747a64cf32d1fde418df8ec3c",
    },
    2: {
        "basename": "ckpt_linear_rat_vit_s2.pt",
        "sha256": "413a770071bd8f16b924c5604f7c7a58567d2eb6c40c66aab202bcf046fec910",
    },
}
FROZEN_GATES = {
    "completed_trials_per_checkpoint": 500,
    "minimum_positive_task_means": 8,
    "semantic_mean_bootstrap_lower_strictly_above_m": 0.0,
    "minimum_mean_specificity_rank": 0.75,
    "specificity_rank_bootstrap_lower_strictly_above": 0.50,
    "bootstrap_draws": 20000,
    "all_three_checkpoints_must_pass": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        action="append",
        required=True,
        help="One full result file. Exactly three are required.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def assert_close(label: str, actual: Any, expected: Any, atol: float = 1e-12) -> None:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual_array.shape != expected_array.shape or not np.allclose(
        actual_array,
        expected_array,
        rtol=0.0,
        atol=atol,
    ):
        raise RuntimeError(f"{label} does not reproduce")


def expected_prompt_catalog() -> list[dict[str, Any]]:
    tasks = task_languages(load_suite("libero_object"))
    vocab, _ = build_vocab(tasks)
    encode = build_encoder(vocab)
    catalog = [
        {
            "prompt_id": prompt_id,
            "text": tasks[prompt_id],
            "target_phrase": target_phrase(tasks[prompt_id]),
            "instruction_ids": encode(tasks[prompt_id]),
        }
        for prompt_id in range(10)
    ]
    catalog.append(
        {
            "prompt_id": "empty",
            "text": "",
            "target_phrase": None,
            "instruction_ids": encode(""),
        }
    )
    return catalog


def validate_row(
    row: dict[str, Any],
    catalog: list[dict[str, Any]],
) -> tuple[int, int, float, float, float]:
    identity = row["episode_identity"]
    task_index = int(identity["task_index"])
    episode = int(identity["episode"])
    if identity["suite"] != "libero_object":
        raise RuntimeError("Row suite identity is invalid")
    if int(identity["init_state_index"]) != episode:
        raise RuntimeError("Canonical init-state index is not the frozen episode index")
    if int(identity["reset_seed"]) != task_index * 100 + episode:
        raise RuntimeError("Reset seed is not the frozen task/episode seed")
    if len(str(identity["init_state_sha256"])) != 64:
        raise RuntimeError("Initial-state hash is malformed")

    scene = row["scene"]
    target_id = int(scene["target_prompt_id"])
    distractor_id = int(scene["distractor_prompt_id"])
    if target_id != task_index or distractor_id == target_id:
        raise RuntimeError("Target/distractor prompt identity is invalid")
    absent_ids = [int(value) for value in scene["absent_prompt_ids"]]
    if absent_ids != [
        prompt_id for prompt_id in range(10) if prompt_id not in (target_id, distractor_id)
    ]:
        raise RuntimeError("Absent-prompt identities are not the frozen complement")
    if len(scene["eligible_bodies"]) != 2:
        raise RuntimeError("Scene does not contain exactly two eligible objects")
    if readable_object_name(scene["target_body"]) != catalog[target_id]["target_phrase"]:
        raise RuntimeError("Target body does not match its official prompt")
    if (
        readable_object_name(scene["distractor_body"])
        != catalog[distractor_id]["target_phrase"]
    ):
        raise RuntimeError("Distractor body does not match its official prompt")

    model_input = row["model_input"]
    if model_input.get("all_clone_hashes_match") is not True:
        raise RuntimeError("A cloned model input did not match")
    if int(model_input.get("clone_count", -1)) != 11:
        raise RuntimeError("Row does not contain eleven cloned conditions")
    input_hash = str(model_input["combined_sha256"])
    if len(input_hash) != 64:
        raise RuntimeError("Combined input hash is malformed")
    component_hashes = model_input.get("component_sha256", {})
    expected_component_keys = {
        "model_image",
        "robot_state",
        "eef",
        f"body:{scene['target_body']}",
        f"body:{scene['distractor_body']}",
    }
    if set(component_hashes) != expected_component_keys or any(
        len(str(value)) != 64 for value in component_hashes.values()
    ):
        raise RuntimeError("Model-input component hashes are incomplete")

    conditions = row["conditions"]
    expected_keys = {str(prompt_id) for prompt_id in range(10)} | {"empty"}
    if set(conditions) != expected_keys:
        raise RuntimeError("Language condition set is incomplete")
    for catalog_row in catalog:
        prompt_key = str(catalog_row["prompt_id"])
        condition = conditions[prompt_key]
        if condition["prompt_id"] != catalog_row["prompt_id"]:
            raise RuntimeError("Condition prompt ID differs from the frozen catalog")
        if condition["prompt_text"] != catalog_row["text"]:
            raise RuntimeError("Condition prompt text differs from the frozen catalog")
        if condition["instruction_ids"] != catalog_row["instruction_ids"]:
            raise RuntimeError("Condition instruction IDs differ from the frozen catalog")
        if condition["input_hash"] != input_hash:
            raise RuntimeError("Condition input hash differs from its paired input")
        chunk = np.asarray(condition["action_chunk"], dtype=np.float32)
        if chunk.shape != (8, 7) or not np.isfinite(chunk).all():
            raise RuntimeError("Action chunk shape or values are invalid")
        if condition["action_chunk_sha256"] != hash_array(chunk):
            raise RuntimeError("Action-chunk hash does not reproduce")
        assert_close(
            "predicted cumulative xyz",
            condition["predicted_cumulative_xyz"],
            chunk[:, :3].sum(axis=0),
            atol=1e-7,
        )
        execution = condition["execution"]
        if int(execution["steps"]) != 8:
            raise RuntimeError("A condition did not execute exactly eight actions")
        start = np.asarray(execution["start_eef"], dtype=np.float64)
        end = np.asarray(execution["end_eef"], dtype=np.float64)
        displacement = np.asarray(execution["displacement"], dtype=np.float64)
        if start.shape != (3,) or end.shape != (3,) or displacement.shape != (3,):
            raise RuntimeError("Execution geometry has the wrong shape")
        assert_close("eef displacement", displacement, end - start)
        assert_close("paired start eef", start, scene["start_eef"])

    unit_target = np.asarray(scene["unit_target"], dtype=np.float64)
    unit_distractor = np.asarray(scene["unit_distractor"], dtype=np.float64)
    start_eef = np.asarray(scene["start_eef"], dtype=np.float64)
    target_position = np.asarray(scene["target_position"], dtype=np.float64)
    distractor_position = np.asarray(scene["distractor_position"], dtype=np.float64)
    reproduced_target_unit = (target_position - start_eef) / np.linalg.norm(
        target_position - start_eef
    )
    reproduced_distractor_unit = (
        distractor_position - start_eef
    ) / np.linalg.norm(distractor_position - start_eef)
    assert_close("target unit norm", np.linalg.norm(unit_target), 1.0)
    assert_close("distractor unit norm", np.linalg.norm(unit_distractor), 1.0)
    assert_close("target unit direction", unit_target, reproduced_target_unit)
    assert_close("distractor unit direction", unit_distractor, reproduced_distractor_unit)
    reproduced = specificity_metrics(
        conditions,
        target_id,
        distractor_id,
        absent_ids,
        unit_target,
        unit_distractor,
    )
    metrics = row["metrics"]
    assert_close("semantic score", metrics["semantic_score_m"], reproduced["semantic_score_m"])
    if bool(metrics["semantic_score_positive"]) != bool(
        reproduced["semantic_score_positive"]
    ):
        raise RuntimeError("Semantic score sign does not reproduce")
    assert_close("specificity rank", metrics["specificity_rank"], reproduced["specificity_rank"])
    assert_close(
        "direction contrast",
        metrics["direction_contrast"],
        reproduced["direction_contrast"],
    )
    assert_close(
        "secondary command score",
        metrics["secondary_predicted_command_score"],
        reproduced["secondary_predicted_command_score"],
        atol=1e-7,
    )
    observed_controls = metrics["ordered_absent_controls"]
    expected_controls = reproduced["ordered_absent_controls"]
    if len(observed_controls) != 56 or len(expected_controls) != 56:
        raise RuntimeError("Ordered control count is not 56")
    for observed, expected in zip(observed_controls, expected_controls):
        if (
            int(observed["left_prompt_id"]) != int(expected["left_prompt_id"])
            or int(observed["right_prompt_id"]) != int(expected["right_prompt_id"])
        ):
            raise RuntimeError("Ordered control identity does not reproduce")
        assert_close("ordered control score", observed["score_m"], expected["score_m"])

    observed_no_language = metrics["no_language_diagnostic"]
    reproduced_no_language = reproduced["no_language_diagnostic"]
    if set(observed_no_language) != set(reproduced_no_language):
        raise RuntimeError("No-language diagnostic fields are incomplete")
    for key in observed_no_language:
        assert_close(
            f"no-language diagnostic {key}",
            observed_no_language[key],
            reproduced_no_language[key],
        )

    empty_norm = float(observed_no_language["empty_displacement_norm_m"])
    if not np.isfinite(empty_norm):
        raise RuntimeError("No-language diagnostic is non-finite")
    return (
        task_index,
        episode,
        float(metrics["semantic_score_m"]),
        float(metrics["specificity_rank"]),
        empty_norm,
    )


def hierarchical_bootstrap_ci(
    by_task: dict[int, list[float]],
    seed: int,
    draws: int,
) -> list[float]:
    if set(by_task) != set(range(10)) or any(len(values) != 50 for values in by_task.values()):
        raise RuntimeError("Bootstrap input must contain ten tasks by 50 states")
    matrix = np.asarray([by_task[task] for task in range(10)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled_tasks = rng.integers(0, 10, size=10)
        total = 0.0
        for task in sampled_tasks:
            sampled_episodes = rng.integers(0, 50, size=50)
            total += float(matrix[task, sampled_episodes].mean())
        means[draw] = total / 10.0
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def checkpoint_stratified_bootstrap_ci(
    values: dict[int, np.ndarray],
    seed: int,
    draws: int,
) -> list[float]:
    if set(values) != {0, 1, 2} or any(array.shape != (500,) for array in values.values()):
        raise RuntimeError("Pooled bootstrap requires three 500-state checkpoints")
    matrices = {
        checkpoint: array.reshape(10, 50)
        for checkpoint, array in values.items()
    }
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        checkpoint_means = []
        for checkpoint in range(3):
            sampled_tasks = rng.integers(0, 10, size=10)
            task_means = []
            for task in sampled_tasks:
                sampled_episodes = rng.integers(0, 50, size=50)
                task_means.append(
                    float(matrices[checkpoint][task, sampled_episodes].mean())
                )
            checkpoint_means.append(float(np.mean(task_means)))
        means[draw] = float(np.mean(checkpoint_means))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if len(args.result) != 3 or len(set(args.result)) != 3:
        raise ValueError("Exactly three distinct full-result paths are required")
    catalog = expected_prompt_catalog()
    validated: dict[int, dict[str, Any]] = {}
    provenance_hash = None
    runner_hash = file_sha256(Path(__file__).with_name("run_local_instruction_specificity.py"))

    for path in args.result:
        result = load_json(path)
        if result.get("schema") != SCHEMA or result.get("mode") != "full":
            raise RuntimeError(f"{path} is not a full {SCHEMA} result")
        if result.get("protocol") != PROTOCOL:
            raise RuntimeError(f"{path} protocol differs from the frozen protocol")
        if result.get("prompt_catalog") != catalog:
            raise RuntimeError(f"{path} prompt catalog differs from the official catalog")
        identity = result["identity"]
        seed = int(identity["checkpoint_seed"])
        if seed not in EXPECTED_CHECKPOINTS or seed in validated:
            raise RuntimeError(f"Unexpected or duplicate checkpoint seed {seed}")
        expected_checkpoint = EXPECTED_CHECKPOINTS[seed]
        if Path(identity["checkpoint"]).name != expected_checkpoint["basename"]:
            raise RuntimeError(f"Seed {seed} checkpoint basename is invalid")
        if identity["checkpoint_sha256"] != expected_checkpoint["sha256"]:
            raise RuntimeError(f"Seed {seed} checkpoint SHA is invalid")
        if identity["cache_sha256"] != EXPECTED_CACHE_SHA256:
            raise RuntimeError(f"Seed {seed} cache SHA is invalid")
        if identity["provenance_verified"] is not True:
            raise RuntimeError(f"Seed {seed} cache provenance did not pass")
        if str(identity["provenance_job_id"]) != EXPECTED_PROVENANCE_JOB_ID:
            raise RuntimeError(f"Seed {seed} provenance job identity is invalid")
        if identity["runner_source_sha256"] != runner_hash:
            raise RuntimeError(f"Seed {seed} runner source SHA is stale")
        if provenance_hash is None:
            provenance_hash = identity["provenance_result_sha256"]
        elif identity["provenance_result_sha256"] != provenance_hash:
            raise RuntimeError("Checkpoint results used different provenance JSON files")
        evaluation = result["evaluation"]
        if (
            int(evaluation["task_start"]),
            int(evaluation["task_end"]),
            int(evaluation["eps_per_task"]),
            int(evaluation["expected_trials"]),
            int(evaluation["completed_trials"]),
            evaluation["matmul_precision"],
        ) != (0, 10, 50, 500, 500, "highest"):
            raise RuntimeError(f"Seed {seed} evaluation identity is invalid")

        by_task_scores = {task: [] for task in range(10)}
        by_task_ranks = {task: [] for task in range(10)}
        empty_norms = []
        episode_keys = set()
        for row in result["rows"]:
            task, episode, score, rank, empty_norm = validate_row(row, catalog)
            key = (task, episode)
            if key in episode_keys:
                raise RuntimeError(f"Seed {seed} has duplicate episode {key}")
            episode_keys.add(key)
            by_task_scores[task].append(score)
            by_task_ranks[task].append(rank)
            empty_norms.append(empty_norm)
        expected_keys = {(task, episode) for task in range(10) for episode in range(50)}
        if episode_keys != expected_keys:
            raise RuntimeError(f"Seed {seed} episode identities are incomplete")

        draws = int(FROZEN_GATES["bootstrap_draws"])
        score_ci = hierarchical_bootstrap_ci(by_task_scores, 2026082100 + seed, draws)
        rank_ci = hierarchical_bootstrap_ci(by_task_ranks, 2026082200 + seed, draws)
        task_mean_scores = {
            str(task): float(np.mean(values))
            for task, values in by_task_scores.items()
        }
        task_mean_ranks = {
            str(task): float(np.mean(values))
            for task, values in by_task_ranks.items()
        }
        scores = np.asarray(
            [value for values in by_task_scores.values() for value in values],
            dtype=np.float64,
        )
        ranks = np.asarray(
            [value for values in by_task_ranks.values() for value in values],
            dtype=np.float64,
        )
        positive_task_means = sum(value > 0 for value in task_mean_scores.values())
        checks = {
            "semantic_score_bootstrap_lower_above_zero": score_ci[0] > 0.0,
            "at_least_eight_positive_task_means": positive_task_means >= 8,
            "mean_specificity_rank_at_least_0p75": float(ranks.mean()) >= 0.75,
            "specificity_rank_bootstrap_lower_above_0p50": rank_ci[0] > 0.50,
        }
        validated[seed] = {
            "result": str(path),
            "result_sha256": file_sha256(path),
            "checkpoint": identity["checkpoint"],
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "trials": len(scores),
            "mean_semantic_score_m": float(scores.mean()),
            "semantic_score_95pct_hierarchical_bootstrap_ci_m": score_ci,
            "fraction_semantic_score_positive": float((scores > 0).mean()),
            "positive_task_means": positive_task_means,
            "task_mean_semantic_score_m": task_mean_scores,
            "mean_specificity_rank": float(ranks.mean()),
            "specificity_rank_95pct_hierarchical_bootstrap_ci": rank_ci,
            "task_mean_specificity_rank": task_mean_ranks,
            "mean_empty_prompt_displacement_norm_m": float(np.mean(empty_norms)),
            "gate_checks": checks,
            "passes_frozen_gate": all(checks.values()),
            "_scores": scores,
            "_ranks": ranks,
        }

    if set(validated) != set(EXPECTED_CHECKPOINTS):
        raise RuntimeError("The three frozen checkpoint seeds are not all represented")
    pooled_scores = np.concatenate([validated[seed]["_scores"] for seed in range(3)])
    pooled_ranks = np.concatenate([validated[seed]["_ranks"] for seed in range(3)])
    draws = int(FROZEN_GATES["bootstrap_draws"])
    pooled_score_ci = checkpoint_stratified_bootstrap_ci(
        {seed: validated[seed]["_scores"] for seed in range(3)},
        2026082300,
        draws,
    )
    pooled_rank_ci = checkpoint_stratified_bootstrap_ci(
        {seed: validated[seed]["_ranks"] for seed in range(3)},
        2026082400,
        draws,
    )
    checkpoint_rows = {}
    for seed, row in validated.items():
        checkpoint_rows[str(seed)] = {
            key: value
            for key, value in row.items()
            if key not in {"_scores", "_ranks"}
        }
    overall_pass = all(row["passes_frozen_gate"] for row in checkpoint_rows.values())
    output = {
        "schema": "xvla-local-instruction-specificity-summary-v1",
        "protocol": PROTOCOL,
        "frozen_gates": FROZEN_GATES,
        "identity_validated": True,
        "overall_pass": overall_pass,
        "claim_eligible": overall_pass,
        "cache_sha256": EXPECTED_CACHE_SHA256,
        "provenance_job_id": EXPECTED_PROVENANCE_JOB_ID,
        "provenance_result_sha256": provenance_hash,
        "summary_source_sha256": file_sha256(Path(__file__).resolve()),
        "checkpoint_results": checkpoint_rows,
        "pooled_descriptive_only": {
            "trials": int(len(pooled_scores)),
            "mean_semantic_score_m": float(pooled_scores.mean()),
            "semantic_score_95pct_checkpoint_stratified_bootstrap_ci_m": pooled_score_ci,
            "fraction_semantic_score_positive": float((pooled_scores > 0).mean()),
            "mean_specificity_rank": float(pooled_ranks.mean()),
            "specificity_rank_95pct_checkpoint_stratified_bootstrap_ci": pooled_rank_ci,
        },
        "claim_if_passed": (
            "At fixed canonical LIBERO-Object observations, changing the object noun causes "
            "a replicated, target-aligned eight-action local motor response beyond matched "
            "absent-object language prompts."
        ),
        "claim_boundaries": (
            "This does not establish BDDL or counterfactual task success, robust instruction "
            "grounding, compositional or zero-shot language understanding, broad-suite "
            "generalization, or a benefit unique to tensor decomposability."
        ),
    }
    write_json(args.output, output)
    print("RESULT", json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
