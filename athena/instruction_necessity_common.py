#!/usr/bin/env python3
"""Frozen identities and scoring utilities for instruction necessity v1.

This is a new closed-loop endpoint designed after partial results made the
different local displacement-rank gate impossible.  It never reuses those
scores or changes that failed gate.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

from athena.counterfactual_target_swap_common import (
    CACHE,
    CHECKPOINTS,
    LIBERO_RUNTIME,
    ORIGINAL_BDDL,
    PROMPTS,
    PROVENANCE,
    RECEPTACLE_SYMBOL,
    TARGETS,
    SOURCE_PATHS as TARGET_SOURCE_PATHS,
    decode_array,
    encode_array,
    file_sha256,
    hash_array,
    validate_libero_runtime,
    write_json,
)


SCHEMA_MANIFEST = "xvla-instruction-necessity-manifest-v1"
SCHEMA_RUN = "xvla-instruction-necessity-run-v1"
SCHEMA_SUMMARY = "xvla-instruction-necessity-summary-v1"

MANIFEST_PATH = "artifacts/instruction_necessity_v1_manifest.json"
SMOKE_RESULT_PATH = "results/instruction_necessity_v1_smoke.json"
SUMMARY_RESULT_PATH = "results/instruction_necessity_v1_summary.json"
LOCAL_SPECIFICITY_SUMMARY_PATH = "results/local_instruction_specificity_v2_summary.json"
LOCAL_SPECIFICITY_SUMMARY_JOB = "831016"

CONDITIONS = ("correct_prompt", "copresent_distractor_prompt", "empty_instruction")
TASK_SHARDS = ((0, 2), (2, 4), (4, 6), (6, 8), (8, 10))

PROTOCOL = {
    "suite": "libero_object",
    "checkpoint_seeds": [0, 1, 2],
    "task_indices": list(range(10)),
    "canonical_episode_start": 40,
    "canonical_episode_end_exclusive": 50,
    "conditions": list(CONDITIONS),
    "copresent_distractor_selector": (
        "minimum SHA-256 of instruction-necessity-v1|task|episode|prompt_id "
        "over the five observation-verified co-present non-target prompt IDs"
    ),
    "copresent_semantics": (
        "co-presence is verified from simulator observation fields and does not assert "
        "camera visibility or unoccluded pixels"
    ),
    "empty_instruction_semantics": (
        "the empty string encodes tokenizer BOS plus 31 unmasked PAD tokens; the model "
        "also prepends its common learned BOS token"
    ),
    "condition_order": (
        "one of six permutations selected by SHA-256 of "
        "instruction-necessity-v1-order|checkpoint|task|episode"
    ),
    "settle_steps": 10,
    "action_horizon": 8,
    "execution_horizon": 8,
    "max_steps": 280,
    "resolution": 64,
    "matmul_precision": "highest",
    "deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cublas_workspace_config": ":4096:8",
    "gpu_family": "a6000",
    "gpu_name": "NVIDIA RTX A6000",
    "gpu_compute_capability": [8, 6],
    "runtime_versions": {
        "python": "3.10.19",
        "torch": "2.7.1",
        "torch_build": "2.7.1+cu126",
        "cuda": "12.6",
        "numpy": "1.26.4",
        "pillow": "12.1.1",
        "mujoco": "3.5.0",
        "robosuite": "1.4.1",
    },
    "task_shards": [list(bounds) for bounds in TASK_SHARDS],
    "checkpoint_state_pairs_per_checkpoint": 100,
    "rollouts_per_checkpoint": 300,
    "unique_paired_states": 100,
    "total_checkpoint_state_pairs": 300,
    "total_rollouts": 900,
}

FROZEN_GATES = {
    "minimum_correct_success_pooled": 0.70,
    "minimum_correct_success_every_checkpoint": 0.60,
    "minimum_correct_minus_copresent_distractor_pooled": 0.20,
    "minimum_correct_minus_empty_instruction_pooled": 0.20,
    "each_control_margin_strictly_positive_every_checkpoint": True,
    "one_sided_paired_exact_p_at_most": 0.01,
    "minimum_tasks_with_strictly_positive_gap_per_control": 8,
    "negative_task_gaps_allowed_per_control": 0,
    "task_stratified_state_cluster_bootstrap_draws": 20000,
    "task_stratified_state_cluster_bootstrap_lower_strictly_above": 0.0,
    "state_cluster_retains_all_three_checkpoint_outcomes": True,
    "all_three_checkpoints_must_pass": True,
}

DESIGN_PROVENANCE = {
    "designed_after_partial_local_specificity_failure_became_mathematically_forced": True,
    "observation_time_local": "2026-08-21T08:04:00-07:00",
    "different_endpoint": (
        "280-step paired closed-loop original-goal success under correct, one "
        "deterministic observation-verified co-present-distractor, and empty-string "
        "instruction encodings"
    ),
    "failed_endpoint_not_reanalyzed": (
        "eight-action displacement specificity ranks against absent-prompt contrasts"
    ),
    "partial_rows_observed": {"0": 345, "1": 339, "2": 362},
    "maximum_possible_final_mean_rank": {
        "0": 0.6363333333333333,
        "1": 0.658916666666667,
        "2": 0.6454583333333335,
    },
    "failed_gate_threshold": 0.75,
    "declaration": (
        "This is a new exploratory-confirmatory endpoint designed after a different "
        "frozen endpoint became impossible. It is not a rescue reanalysis. No local "
        "specificity score, rank, threshold, or outcome selects any condition, state, "
        "distractor, task, checkpoint, or gate in this protocol."
    ),
    "pre_outcome_semantic_and_inference_hardening": {
        "commit_time_local": "2026-08-21T08:40:15-07:00",
        "first_submission_time_local": "2026-08-21T08:41:26-07:00",
        "commit_preceded_submission": True,
        "reason": "independent code review before any preflight or rollout began",
        "canceled_unstarted_graphs": [
            list(range(831332, 831350)),
            list(range(831352, 831370)),
        ],
        "changes": (
            "renamed controls to copresent_distractor_prompt and empty_instruction, "
            "documented their exact semantics, and added a task-stratified state-cluster "
            "bootstrap retaining all three checkpoint outcomes together"
        ),
        "outcomes_observed_before_change": False,
    },
}

SOURCE_PATHS = (
    "athena/instruction_necessity_common.py",
    "athena/preflight_instruction_necessity.py",
    "athena/run_instruction_necessity.py",
    "athena/summarize_instruction_necessity.py",
    "athena/preflight_counterfactual_target_swap.py",
    "athena/run_counterfactual_target_swap.py",
    *TARGET_SOURCE_PATHS,
)


def source_hashes(repository_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source is absent: {path}")
        result[relative] = file_sha256(path)
    return result


def prompt_target_to_id() -> dict[str, int]:
    return {target: prompt_id for prompt_id, target in TARGETS.items()}


def selector_digest(task: int, episode: int, prompt_id: int) -> str:
    payload = f"instruction-necessity-v1|{task}|{episode}|{prompt_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def select_copresent_distractor(
    task: int, episode: int, present_prompt_ids: list[int]
) -> tuple[int, dict[str, str]]:
    candidates = sorted(set(int(value) for value in present_prompt_ids) - {task})
    if len(candidates) != 5:
        raise RuntimeError("Co-present-distractor selection requires exactly five candidates")
    digests = {str(prompt_id): selector_digest(task, episode, prompt_id) for prompt_id in candidates}
    selected = min(candidates, key=lambda prompt_id: (digests[str(prompt_id)], prompt_id))
    return selected, digests


def condition_order(seed: int, task: int, episode: int) -> list[str]:
    permutations = list(itertools.permutations(CONDITIONS))
    payload = f"instruction-necessity-v1-order|{seed}|{task}|{episode}".encode()
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(permutations)
    return list(permutations[index])


def one_sided_paired_exact_p(correct_only: int, control_only: int) -> float:
    """One-sided exact McNemar p-value for correct success exceeding control."""
    if correct_only < 0 or control_only < 0:
        raise ValueError("Discordant counts must be nonnegative")
    discordant = correct_only + control_only
    if discordant == 0:
        return 1.0
    numerator = sum(math.comb(discordant, k) for k in range(correct_only, discordant + 1))
    return numerator / (2**discordant)


def validate_failed_local_summary(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    errors = []
    if result.get("schema") != "xvla-local-instruction-specificity-summary-v2":
        errors.append("wrong local-specificity summary schema")
    if result.get("identity_validated") is not True:
        errors.append("local-specificity identities were not validated")
    if result.get("overall_pass") is not False or result.get("claim_eligible") is not False:
        errors.append("the previous endpoint did not formally fail")
    checkpoints = result.get("checkpoint_results", {})
    if set(checkpoints) != {"0", "1", "2"}:
        errors.append("local-specificity checkpoint set is incomplete")
    elif all(bool(row.get("passes_frozen_gate")) for row in checkpoints.values()):
        errors.append("local-specificity component gates unexpectedly all pass")
    if errors:
        raise RuntimeError("; ".join(errors))
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def validate_manifest(path: Path, root: Path) -> dict[str, Any]:
    result = load_json(path)
    errors = []
    if result.get("schema") != SCHEMA_MANIFEST:
        errors.append("wrong manifest schema")
    if result.get("protocol") != PROTOCOL:
        errors.append("manifest protocol differs")
    if result.get("frozen_gates") != FROZEN_GATES:
        errors.append("manifest gates differ")
    if result.get("design_provenance") != DESIGN_PROVENANCE:
        errors.append("design provenance differs")
    if result.get("mapping_count") != 100 or len(result.get("mappings", [])) != 100:
        errors.append("manifest mapping matrix is incomplete")
    if result.get("checkpoints") != {str(seed): row for seed, row in CHECKPOINTS.items()}:
        errors.append("manifest checkpoint identities differ")
    cache = result.get("cache", {})
    if cache.get("live_sha256") != CACHE["sha256"] or any(
        cache.get(key) != value for key, value in CACHE.items()
    ):
        errors.append("manifest cache identity differs")
    provenance = result.get("provenance", {})
    if provenance.get("verified") is not True or any(
        provenance.get(key) != value for key, value in PROVENANCE.items()
    ):
        errors.append("manifest provenance identity differs")
    if result.get("source_sha256_start") != result.get("source_sha256_end"):
        errors.append("manifest sources changed during preflight")
    if result.get("source_sha256_end") != source_hashes(root):
        errors.append("live source identities differ from manifest")
    if result.get("libero_runtime_start") != LIBERO_RUNTIME or result.get("libero_runtime_end") != LIBERO_RUNTIME:
        errors.append("manifest LIBERO runtime identity differs")
    mappings = result.get("mappings", [])
    mapping_keys = {
        (int(row.get("task_index", -1)), int(row.get("episode", -1)))
        for row in mappings
    }
    expected_keys = {(task, episode) for task in range(10) for episode in range(40, 50)}
    if mapping_keys != expected_keys:
        errors.append("manifest task-episode keys differ")
    else:
        for row in mappings:
            task = int(row["task_index"])
            episode = int(row["episode"])
            present = [int(value) for value in row.get("present_prompt_ids", [])]
            try:
                selected, digests = select_copresent_distractor(task, episode, present)
            except RuntimeError as error:
                errors.append(f"manifest selector input invalid: {error}")
                break
            if (
                int(row.get("selected_distractor_prompt_id", -1)) != selected
                or row.get("selector_digests") != digests
                or row.get("task_language") != PROMPTS[task]
                or row.get("init_state_index") != episode
                or row.get("reset_seed") != task * 100 + episode
                or len(str(row.get("settled_physical_input_sha256", ""))) != 64
            ):
                errors.append("manifest mapping does not reproduce")
                break
    local_path = root / LOCAL_SPECIFICITY_SUMMARY_PATH
    if not local_path.is_file() or result.get("failed_local_summary_sha256") != file_sha256(local_path):
        errors.append("failed local-summary identity differs")
    if errors:
        raise RuntimeError("; ".join(errors))
    return result


def synthetic_known_answer() -> dict[str, Any]:
    selected, digests = select_copresent_distractor(0, 40, [0, 1, 2, 3, 4, 5])
    order = condition_order(2, 9, 49)
    p_value = one_sided_paired_exact_p(10, 0)
    if selected != 1 or order != [
        "empty_instruction",
        "copresent_distractor_prompt",
        "correct_prompt",
    ]:
        raise RuntimeError("Frozen selector known answer changed")
    if abs(p_value - 0.0009765625) > 1e-15:
        raise RuntimeError("Paired exact-test known answer changed")
    return {"selected": selected, "selector_digests": digests, "order": order, "p": p_value}


if __name__ == "__main__":
    print(json.dumps(synthetic_known_answer(), indent=2, sort_keys=True))
