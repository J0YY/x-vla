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
    official_init_states,
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
    "completed_trials_per_checkpoint": 400,
    "primary_canonical_episode_start": 10,
    "primary_canonical_episode_end_exclusive": 50,
    "minimum_positive_task_means": 8,
    "semantic_mean_hierarchical_resampling_lower_strictly_above_m": 0.0,
    "minimum_mean_specificity_rank": 0.75,
    "specificity_rank_hierarchical_resampling_lower_strictly_above": 0.50,
    "hierarchical_resampling_draws": 20000,
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


def expected_prompt_catalog(suite: Any) -> list[dict[str, Any]]:
    tasks = task_languages(suite)
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
    expected_init_state_sha256: str,
) -> tuple[int, int, float, float, float, str, dict[str, str]]:
    identity = row["episode_identity"]
    task_index = int(identity["task_index"])
    episode = int(identity["episode"])
    if task_index not in range(10) or episode not in range(10, 50):
        raise RuntimeError("Row task or episode is outside the frozen primary set")
    if identity["suite"] != "libero_object":
        raise RuntimeError("Row suite identity is invalid")
    if identity["task_language"] != catalog[task_index]["text"]:
        raise RuntimeError("Row task language differs from the official catalog")
    if int(identity["init_state_index"]) != episode:
        raise RuntimeError("Canonical init-state index is not the frozen episode index")
    if int(identity["reset_seed"]) != task_index * 100 + episode:
        raise RuntimeError("Reset seed is not the frozen task/episode seed")
    if identity["init_state_sha256"] != expected_init_state_sha256:
        raise RuntimeError("Initial-state hash differs from the official canonical state")

    scene = row["scene"]
    target_id = int(scene["target_prompt_id"])
    if target_id != task_index:
        raise RuntimeError("Target prompt identity is invalid")
    present_ids = [int(value) for value in scene["present_prompt_ids"]]
    distractor_ids = [int(value) for value in scene["distractor_prompt_ids"]]
    absent_ids = [int(value) for value in scene["absent_prompt_ids"]]
    if present_ids != sorted(present_ids) or len(set(present_ids)) != 6:
        raise RuntimeError("Present-prompt identities are invalid")
    if target_id not in present_ids:
        raise RuntimeError("Target prompt is absent from the scene")
    if distractor_ids != [value for value in present_ids if value != target_id]:
        raise RuntimeError("Distractor-prompt identities are not the frozen complement")
    if absent_ids != [value for value in range(10) if value not in present_ids]:
        raise RuntimeError("Absent-prompt identities are not the frozen complement")
    if len(distractor_ids) != 5 or len(absent_ids) != 4:
        raise RuntimeError("Scene must contain five distractors and four absent prompts")
    eligible_bodies = list(scene["eligible_bodies"])
    if len(eligible_bodies) != 6 or len(set(eligible_bodies)) != 6:
        raise RuntimeError("Scene does not contain exactly six eligible objects")
    body_to_prompt = {
        str(body): int(prompt_id)
        for body, prompt_id in scene["body_to_prompt_id"].items()
    }
    if set(body_to_prompt) != set(eligible_bodies) or sorted(body_to_prompt.values()) != present_ids:
        raise RuntimeError("Body-to-prompt mapping is invalid")
    if body_to_prompt.get(str(scene["target_body"])) != target_id:
        raise RuntimeError("Target body-to-prompt mapping is invalid")
    if readable_object_name(scene["target_body"]) != catalog[target_id]["target_phrase"]:
        raise RuntimeError("Target body does not match its official prompt")
    distractor_bodies = list(scene["distractor_bodies"])
    if len(distractor_bodies) != 5:
        raise RuntimeError("Distractor-body catalog is incomplete")
    for body, prompt_id in zip(distractor_bodies, distractor_ids):
        if body_to_prompt.get(body) != prompt_id:
            raise RuntimeError("Distractor body order differs from prompt order")
        if readable_object_name(body) != catalog[prompt_id]["target_phrase"]:
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
        "simulator_state",
        "model_image",
        "robot_state",
        "eef",
    } | {f"body:{body}" for body in eligible_bodies}
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

    start_eef = np.asarray(scene["start_eef"], dtype=np.float64)
    prompt_positions = {
        int(prompt_id): np.asarray(position, dtype=np.float64)
        for prompt_id, position in scene["prompt_positions"].items()
    }
    unit_directions = {
        int(prompt_id): np.asarray(direction, dtype=np.float64)
        for prompt_id, direction in scene["unit_directions"].items()
    }
    if set(prompt_positions) != set(present_ids) or set(unit_directions) != set(present_ids):
        raise RuntimeError("Present-object geometry is incomplete")
    assert_close("target position", scene["target_position"], prompt_positions[target_id])
    for prompt_id in present_ids:
        position = prompt_positions[prompt_id]
        reproduced_unit = (position - start_eef) / np.linalg.norm(position - start_eef)
        assert_close(
            f"unit norm {prompt_id}",
            np.linalg.norm(unit_directions[prompt_id]),
            1.0,
        )
        assert_close(
            f"unit direction {prompt_id}",
            unit_directions[prompt_id],
            reproduced_unit,
        )
    reproduced = specificity_metrics(
        conditions,
        target_id,
        distractor_ids,
        absent_ids,
        unit_directions,
    )
    metrics = row["metrics"]
    assert_close("semantic score", metrics["semantic_score_m"], reproduced["semantic_score_m"])
    if bool(metrics["semantic_score_positive"]) != bool(
        reproduced["semantic_score_positive"]
    ):
        raise RuntimeError("Semantic score sign does not reproduce")
    assert_close("specificity rank", metrics["specificity_rank"], reproduced["specificity_rank"])
    assert_close(
        "secondary command score",
        metrics["secondary_predicted_command_score"],
        reproduced["secondary_predicted_command_score"],
        atol=1e-7,
    )
    observed_comparisons = metrics["distractor_comparisons"]
    expected_comparisons = reproduced["distractor_comparisons"]
    if len(observed_comparisons) != 5 or len(expected_comparisons) != 5:
        raise RuntimeError("Distractor comparison count is not five")
    for observed, expected in zip(observed_comparisons, expected_comparisons):
        if int(observed["distractor_prompt_id"]) != int(
            expected["distractor_prompt_id"]
        ):
            raise RuntimeError("Distractor comparison identity does not reproduce")
        assert_close(
            "distractor semantic score",
            observed["semantic_score_m"],
            expected["semantic_score_m"],
        )
        assert_close(
            "distractor specificity rank",
            observed["specificity_rank"],
            expected["specificity_rank"],
        )
        observed_controls = observed["ordered_absent_controls"]
        expected_controls = expected["ordered_absent_controls"]
        if len(observed_controls) != 12 or len(expected_controls) != 12:
            raise RuntimeError("Per-distractor ordered control count is not 12")
        for observed_control, expected_control in zip(
            observed_controls, expected_controls
        ):
            if (
                int(observed_control["left_prompt_id"])
                != int(expected_control["left_prompt_id"])
                or int(observed_control["right_prompt_id"])
                != int(expected_control["right_prompt_id"])
            ):
                raise RuntimeError("Ordered control identity does not reproduce")
            assert_close(
                "ordered control score",
                observed_control["score_m"],
                expected_control["score_m"],
            )
        assert_close(
            "distractor command score",
            observed["secondary_predicted_command_score"],
            expected["secondary_predicted_command_score"],
            atol=1e-7,
        )
        assert_close(
            "distractor direction contrast",
            observed["direction_contrast"],
            expected["direction_contrast"],
        )

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
        input_hash,
        dict(component_hashes),
    )


def hierarchical_bootstrap_ci(
    by_task: dict[int, list[float]],
    seed: int,
    draws: int,
) -> list[float]:
    if set(by_task) != set(range(10)) or any(len(values) != 40 for values in by_task.values()):
        raise RuntimeError("Resampling input must contain ten tasks by 40 states")
    matrix = np.asarray([by_task[task] for task in range(10)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled_tasks = rng.integers(0, 10, size=10)
        total = 0.0
        for task in sampled_tasks:
            sampled_episodes = rng.integers(0, 40, size=40)
            total += float(matrix[task, sampled_episodes].mean())
        means[draw] = total / 10.0
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def checkpoint_stratified_bootstrap_ci(
    values: dict[int, np.ndarray],
    seed: int,
    draws: int,
) -> list[float]:
    if set(values) != {0, 1, 2} or any(array.shape != (400,) for array in values.values()):
        raise RuntimeError("Pooled resampling requires three 400-state checkpoints")
    matrices = {
        checkpoint: array.reshape(10, 40)
        for checkpoint, array in values.items()
    }
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled_tasks = rng.integers(0, 10, size=10)
        sampled_episodes_by_position = [
            rng.integers(0, 40, size=40) for _ in sampled_tasks
        ]
        checkpoint_means = []
        for checkpoint in range(3):
            task_means = []
            for task, sampled_episodes in zip(
                sampled_tasks, sampled_episodes_by_position
            ):
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
    summary_source_sha256 = file_sha256(Path(__file__).resolve())
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if len(args.result) != 3 or len(set(args.result)) != 3:
        raise ValueError("Exactly three distinct full-result paths are required")
    suite = load_suite("libero_object")
    catalog = expected_prompt_catalog(suite)
    canonical_states = {
        task: official_init_states(suite, task) for task in range(10)
    }
    expected_init_state_sha256 = {
        (task, episode): hash_array(np.asarray(canonical_states[task][episode]))
        for task in range(10)
        for episode in range(10, 50)
    }
    validated: dict[int, dict[str, Any]] = {}
    paired_state_identities: dict[tuple[int, int], tuple[str, dict[str, str]]] = {}
    provenance_hash = None
    runner_hash = file_sha256(Path(__file__).with_name("run_local_instruction_specificity.py"))
    repository_root = Path(__file__).resolve().parents[1]
    expected_imported_source_sha256 = {
        relative: file_sha256(repository_root / relative)
        for relative in (
            "athena/run_xvla_experiment.py",
            "xvla/models/vla.py",
            "xvla/models/vit.py",
            "xvla/nn/attention.py",
            "xvla/nn/normalization.py",
        )
    }

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
        if identity.get("imported_source_sha256") != expected_imported_source_sha256:
            raise RuntimeError(f"Seed {seed} imported source identities are stale")
        if provenance_hash is None:
            provenance_hash = identity["provenance_result_sha256"]
        elif identity["provenance_result_sha256"] != provenance_hash:
            raise RuntimeError("Checkpoint results used different provenance JSON files")
        evaluation = result["evaluation"]
        if (
            int(evaluation["task_start"]),
            int(evaluation["task_end"]),
            int(evaluation["episode_start"]),
            int(evaluation["eps_per_task"]),
            int(evaluation["expected_trials"]),
            int(evaluation["completed_trials"]),
            evaluation["matmul_precision"],
        ) != (0, 10, 10, 40, 400, 400, "highest"):
            raise RuntimeError(f"Seed {seed} evaluation identity is invalid")

        score_by_key: dict[tuple[int, int], float] = {}
        rank_by_key: dict[tuple[int, int], float] = {}
        empty_norms = []
        episode_keys = set()
        for row in result["rows"]:
            task, episode, score, rank, empty_norm, input_hash, component_hashes = (
                validate_row(
                    row,
                    catalog,
                    expected_init_state_sha256[(
                        int(row["episode_identity"]["task_index"]),
                        int(row["episode_identity"]["episode"]),
                    )],
                )
            )
            key = (task, episode)
            if key in episode_keys:
                raise RuntimeError(f"Seed {seed} has duplicate episode {key}")
            episode_keys.add(key)
            score_by_key[key] = score
            rank_by_key[key] = rank
            empty_norms.append(empty_norm)
            paired_identity = (input_hash, component_hashes)
            if key not in paired_state_identities:
                paired_state_identities[key] = paired_identity
            elif paired_state_identities[key] != paired_identity:
                raise RuntimeError(
                    f"Settled physical input differs across checkpoints for {key}"
                )
        expected_keys = {
            (task, episode)
            for task in range(10)
            for episode in range(10, 50)
        }
        if episode_keys != expected_keys:
            raise RuntimeError(f"Seed {seed} episode identities are incomplete")

        by_task_scores = {
            task: [score_by_key[(task, episode)] for episode in range(10, 50)]
            for task in range(10)
        }
        by_task_ranks = {
            task: [rank_by_key[(task, episode)] for episode in range(10, 50)]
            for task in range(10)
        }

        draws = int(FROZEN_GATES["hierarchical_resampling_draws"])
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
            "semantic_score_resampling_lower_above_zero": score_ci[0]
            > float(
                FROZEN_GATES[
                    "semantic_mean_hierarchical_resampling_lower_strictly_above_m"
                ]
            ),
            "at_least_eight_positive_task_means": positive_task_means
            >= int(FROZEN_GATES["minimum_positive_task_means"]),
            "mean_specificity_rank_at_least_0p75": float(ranks.mean())
            >= float(FROZEN_GATES["minimum_mean_specificity_rank"]),
            "specificity_rank_resampling_lower_above_0p50": rank_ci[0]
            > float(
                FROZEN_GATES[
                    "specificity_rank_hierarchical_resampling_lower_strictly_above"
                ]
            ),
        }
        validated[seed] = {
            "result": str(path),
            "result_sha256": file_sha256(path),
            "checkpoint": identity["checkpoint"],
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "trials": len(scores),
            "mean_semantic_score_m": float(scores.mean()),
            "semantic_score_95pct_hierarchical_resampling_interval_m": score_ci,
            "fraction_semantic_score_positive": float((scores > 0).mean()),
            "positive_task_means": positive_task_means,
            "task_mean_semantic_score_m": task_mean_scores,
            "mean_specificity_rank": float(ranks.mean()),
            "specificity_rank_95pct_hierarchical_resampling_interval": rank_ci,
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
    draws = int(FROZEN_GATES["hierarchical_resampling_draws"])
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
        "schema": "xvla-local-instruction-specificity-summary-v2",
        "protocol": PROTOCOL,
        "frozen_gates": FROZEN_GATES,
        "identity_validated": True,
        "overall_pass": overall_pass,
        "claim_eligible": overall_pass,
        "cache_sha256": EXPECTED_CACHE_SHA256,
        "provenance_job_id": EXPECTED_PROVENANCE_JOB_ID,
        "provenance_result_sha256": provenance_hash,
        "summary_source_sha256": summary_source_sha256,
        "checkpoint_results": checkpoint_rows,
        "pooled_descriptive_only": {
            "trials": int(len(pooled_scores)),
            "mean_semantic_score_m": float(pooled_scores.mean()),
            "semantic_score_95pct_paired_checkpoint_resampling_interval_m": pooled_score_ci,
            "fraction_semantic_score_positive": float((pooled_scores > 0).mean()),
            "mean_specificity_rank": float(pooled_ranks.mean()),
            "specificity_rank_95pct_paired_checkpoint_resampling_interval": pooled_rank_ci,
        },
        "interval_scope": (
            "Intervals are empirical hierarchical stability intervals over the finite "
            "canonical task-state set, not population confidence intervals."
        ),
        "claim_if_passed": (
            "Across three specified legacy chi-ViT checkpoints and canonical LIBERO-Object "
            "states held out from intervention development, replacing the task target phrase "
            "with five prespecified co-present object phrases produces, on average, a "
            "differential eight-action end-effector displacement along the corresponding "
            "target-versus-distractor directions. The state-level effect is larger than "
            "typical contrasts among four truly absent object phrases."
        ),
        "claim_boundaries": (
            "This does not establish absolute motion toward either named object, bidirectional "
            "noun selection, BDDL or counterfactual task success, robust instruction grounding, "
            "compositional or zero-shot language understanding, broad-suite generalization, "
            "a practically meaningful effect size, or a benefit unique to decomposability."
        ),
    }
    write_json(args.output, output)
    print("RESULT", json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
