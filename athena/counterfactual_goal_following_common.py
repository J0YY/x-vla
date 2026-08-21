#!/usr/bin/env python3
"""Frozen identities and exact tests for counterfactual goal following v1."""

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
    decode_array,
    encode_array,
    file_sha256,
    hash_array,
    inspect_original_bddl,
    rewrite_bddl,
    validate_libero_runtime,
    write_json,
)
from athena.instruction_necessity_common import (
    DESIGN_PROVENANCE as INSTRUCTION_DESIGN_PROVENANCE,
    FROZEN_GATES as INSTRUCTION_GATES,
    PROTOCOL as INSTRUCTION_PROTOCOL,
    SCHEMA_RUN as INSTRUCTION_RUN_SCHEMA,
    SCHEMA_SUMMARY as INSTRUCTION_SUMMARY_SCHEMA,
)


SCHEMA_MANIFEST = "xvla-counterfactual-goal-following-manifest-v1"
SCHEMA_RUN = "xvla-counterfactual-goal-following-run-v1"
SCHEMA_SUMMARY = "xvla-counterfactual-goal-following-summary-v1"

INSTRUCTION_SUMMARY_JOB = "831389"
INSTRUCTION_SUMMARY_PATH = "results/instruction_necessity_v1_summary.json"
MANIFEST_PATH = "artifacts/counterfactual_goal_following_v1_manifest.json"
BDDL_DIRECTORY = "artifacts/counterfactual_goal_following_v1_bddl"
SMOKE_RESULT_PATH = "results/counterfactual_goal_following_v1_smoke.json"
SUMMARY_RESULT_PATH = "results/counterfactual_goal_following_v1_summary.json"

CONDITIONS = (
    "original_goal_original_prompt",
    "original_goal_counterfactual_prompt",
    "counterfactual_goal_counterfactual_prompt",
    "counterfactual_goal_original_prompt",
)
TASK_SHARDS = ((0, 2), (2, 4), (4, 6), (6, 8), (8, 10))

PROTOCOL = {
    "suite": "libero_object",
    "checkpoint_seeds": [0, 1, 2],
    "task_indices": list(range(10)),
    "canonical_episode_start": 20,
    "canonical_episode_end_exclusive": 30,
    "co_present_counterfactual_targets_per_task": 5,
    "conditions": list(CONDITIONS),
    "condition_order": (
        "one of 24 permutations selected by SHA-256 of "
        "counterfactual-goal-following-v1-order|checkpoint|task|episode|target"
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
    "four_arm_units_per_checkpoint": 500,
    "rollouts_per_checkpoint": 2000,
    "total_four_arm_units": 1500,
    "total_rollouts": 6000,
    "exact_cluster_unit": (
        "task sum retaining all three checkpoints, 10 episodes, and five targets"
    ),
}

FROZEN_GATES = {
    "minimum_original_goal_matching_success_every_checkpoint": 0.60,
    "minimum_counterfactual_goal_matching_task_macro_success": 0.25,
    "minimum_counterfactual_goal_matching_success_every_checkpoint": 0.20,
    "minimum_original_goal_directional_gap_pooled": 0.20,
    "minimum_counterfactual_goal_directional_gap_pooled": 0.20,
    "minimum_symmetric_aligned_prompt_gap_pooled": 0.20,
    "all_three_gaps_strictly_positive_every_checkpoint": True,
    "one_sided_exact_mcnemar_p_at_most_each_direction": 0.01,
    "one_sided_exact_task_signflip_p_at_most_each_metric": 0.01,
    "minimum_tasks_with_strictly_positive_gap_each_metric": 8,
    "negative_task_gaps_allowed_each_metric": 0,
    "all_three_checkpoints_must_pass": True,
}

DESIGN_PROVENANCE = {
    "frozen_time_local": "2026-08-21T08:38:00-07:00",
    "designed_after_local_rank_failure": True,
    "local_rank_summary_job": "831016",
    "local_rank_overall_pass": False,
    "old_target_swap_preflight_job": "831233",
    "old_target_swap_preflight_failed_closed": True,
    "old_target_swap_descendants_canceled": list(range(831234, 831251)),
    "designed_before_instruction_necessity_outcomes": True,
    "instruction_necessity_chronology": {
        "initial_graph_831332_through_831349": (
            "canceled entirely unstarted for pre-outcome semantic and inference hardening"
        ),
        "second_graph_831352_through_831369": (
            "canceled entirely unstarted for pre-outcome chronology and unit hardening"
        ),
        "final_preflight_831372": "PENDING_PRIORITY",
        "final_smoke_831373": "HELD_BEFORE_ANY_SCIENTIFIC_OUTCOME",
        "final_full_831374_through_831388": "PENDING_DEPENDENCY",
        "final_summary_831389": "PENDING_DEPENDENCY",
        "outcomes_observed": False,
    },
    "final_inference_hardening_time_local": "2026-08-21T08:52:00-07:00",
    "final_inference_hardening": (
        "clustered exact sensitivity changed from 30 checkpoint-task sums to ten task "
        "sums retaining all checkpoints, episodes, and target swaps; pooled McNemar was "
        "explicitly labeled non-cluster-robust"
    ),
    "fresh_namespace": "counterfactual_goal_following_v1",
    "fresh_state_set": "official canonical episodes 20 through 29",
    "declaration": (
        "This fresh 2x2 goal-by-prompt endpoint was fixed after the local rank failure "
        "and before any instruction-necessity outcome. It is conditional on the new "
        "instruction-necessity summary passing and does not revive or reinterpret the "
        "failed 831233 graph. No prior score or outcome selects an episode, target, arm, "
        "threshold, checkpoint, task, or statistic."
    ),
}

SOURCE_PATHS = (
    "athena/COUNTERFACTUAL_GOAL_FOLLOWING_PROTOCOL.md",
    "athena/counterfactual_goal_following_common.py",
    "athena/preflight_counterfactual_goal_following.py",
    "athena/run_counterfactual_goal_following.py",
    "athena/summarize_counterfactual_goal_following.py",
    "athena/counterfactual_target_swap_common.py",
    "athena/preflight_counterfactual_target_swap.py",
    "athena/run_counterfactual_target_swap.py",
    "athena/summarize_counterfactual_target_swap.py",
    "athena/instruction_necessity_common.py",
    "athena/preflight_instruction_necessity.py",
    "athena/run_instruction_necessity.py",
    "athena/summarize_instruction_necessity.py",
    "athena/libero_dataset_metadata.py",
    "athena/run_local_instruction_specificity.py",
    "athena/run_xvla_experiment.py",
    "xvla/__init__.py",
    "xvla/models/__init__.py",
    "xvla/models/lm.py",
    "xvla/models/vla.py",
    "xvla/models/vit.py",
    "xvla/nn/__init__.py",
    "xvla/nn/attention.py",
    "xvla/nn/baselines.py",
    "xvla/nn/bilinear.py",
    "xvla/nn/block.py",
    "xvla/nn/flow_action.py",
    "xvla/nn/homogeneous.py",
    "xvla/nn/normalization.py",
    "xvla/nn/product_routing.py",
    "xvla/nn/projector.py",
    "xvla/nn/quantile_action.py",
)


def source_hashes(root: Path) -> dict[str, str]:
    result = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source is absent: {path}")
        result[relative] = file_sha256(path)
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def condition_order(seed: int, task: int, episode: int, target: int) -> list[str]:
    permutations = list(itertools.permutations(CONDITIONS))
    payload = (
        f"counterfactual-goal-following-v1-order|{seed}|{task}|{episode}|{target}"
    ).encode()
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(permutations)
    return list(permutations[index])


def one_sided_exact_mcnemar_p(matching_only: int, mismatching_only: int) -> float:
    if matching_only < 0 or mismatching_only < 0:
        raise ValueError("Discordant counts must be nonnegative")
    discordant = matching_only + mismatching_only
    if discordant == 0:
        return 1.0
    return sum(math.comb(discordant, k) for k in range(matching_only, discordant + 1)) / (
        2**discordant
    )


def one_sided_exact_signflip_p(cluster_sums: list[int]) -> float:
    """Exact sign-flip p-value conditional on nonzero cluster magnitudes."""
    values = [int(value) for value in cluster_sums if int(value) != 0]
    if not values:
        return 1.0
    observed = sum(values)
    counts = {0: 1}
    for value in values:
        magnitude = abs(value)
        updated: dict[int, int] = {}
        for total, count in counts.items():
            updated[total + magnitude] = updated.get(total + magnitude, 0) + count
            updated[total - magnitude] = updated.get(total - magnitude, 0) + count
        counts = updated
    numerator = sum(count for total, count in counts.items() if total >= observed)
    return numerator / (2 ** len(values))


def validate_instruction_summary(path: Path, root: Path) -> dict[str, Any]:
    result = load_json(path)
    errors = []
    if result.get("schema") != INSTRUCTION_SUMMARY_SCHEMA:
        errors.append("wrong instruction-necessity summary schema")
    if result.get("protocol") != INSTRUCTION_PROTOCOL:
        errors.append("instruction-necessity protocol differs")
    if result.get("frozen_gates") != INSTRUCTION_GATES:
        errors.append("instruction-necessity gates differ")
    if result.get("design_provenance") != INSTRUCTION_DESIGN_PROVENANCE:
        errors.append("instruction-necessity design provenance differs")
    if result.get("identity_validated") is not True:
        errors.append("instruction-necessity identities were not validated")
    if result.get("overall_pass") is not True or result.get("claim_eligible") is not True:
        errors.append("instruction-necessity prerequisite did not pass")
    gates = result.get("gates")
    if not isinstance(gates, dict) or not gates or not all(value is True for value in gates.values()):
        errors.append("an instruction-necessity component gate did not pass")
    if set(result.get("correct_success_by_checkpoint", {})) != {"0", "1", "2"}:
        errors.append("instruction-necessity checkpoint matrix differs")
    if result.get("summary_source_sha256") != file_sha256(
        root / "athena/summarize_instruction_necessity.py"
    ):
        errors.append("instruction-necessity summary source is stale")
    inputs = result.get("input_results")
    if not isinstance(inputs, dict) or len(inputs) != 15:
        errors.append("instruction-necessity full-result set differs")
    else:
        for path_text, identity in inputs.items():
            raw_path = Path(path_text)
            if not raw_path.is_absolute():
                raw_path = root / raw_path
            if not raw_path.is_file() or identity.get("sha256") != file_sha256(raw_path):
                errors.append("an instruction-necessity raw result identity is stale")
                break
            raw = load_json(raw_path)
            if (
                raw.get("schema") != INSTRUCTION_RUN_SCHEMA
                or raw.get("mode") != "full"
                or raw.get("protocol") != INSTRUCTION_PROTOCOL
                or raw.get("frozen_gates") != INSTRUCTION_GATES
                or raw.get("design_provenance") != INSTRUCTION_DESIGN_PROVENANCE
            ):
                errors.append("an instruction-necessity raw result differs")
                break
    if errors:
        raise RuntimeError("Instruction-necessity prerequisite failed: " + "; ".join(errors))
    return result


def validate_manifest(path: Path, root: Path) -> dict[str, Any]:
    result = load_json(path)
    errors = []
    if result.get("schema") != SCHEMA_MANIFEST:
        errors.append("wrong goal-following manifest schema")
    if result.get("protocol") != PROTOCOL or result.get("frozen_gates") != FROZEN_GATES:
        errors.append("goal-following protocol or gates differ")
    if result.get("design_provenance") != DESIGN_PROVENANCE:
        errors.append("goal-following design provenance differs")
    if result.get("mapping_count") != 50 or len(result.get("mappings", [])) != 50:
        errors.append("goal-following mapping count differs")
    if result.get("checkpoints") != {str(seed): row for seed, row in CHECKPOINTS.items()}:
        errors.append("checkpoint identities differ")
    if result.get("source_sha256_start") != result.get("source_sha256_end"):
        errors.append("sources changed during preflight")
    if result.get("source_sha256_end") != source_hashes(root):
        errors.append("live source closure differs")
    if result.get("libero_runtime_start") != LIBERO_RUNTIME or result.get("libero_runtime_end") != LIBERO_RUNTIME:
        errors.append("LIBERO runtime identity differs")
    summary_path = root / INSTRUCTION_SUMMARY_PATH
    if not summary_path.is_file() or result.get("instruction_summary_sha256") != file_sha256(summary_path):
        errors.append("instruction-necessity summary identity differs")
    mapping_keys = {
        (int(row.get("task_index", -1)), int(row.get("counterfactual_prompt_id", -1)))
        for row in result.get("mappings", [])
    }
    if len(mapping_keys) != 50:
        errors.append("mapping identities are incomplete")
    for row in result.get("mappings", []):
        task = int(row.get("task_index", -1))
        target = int(row.get("counterfactual_prompt_id", -1))
        try:
            original_name, original_sha = ORIGINAL_BDDL[task]
        except KeyError:
            errors.append("mapping task differs")
            break
        if (
            target == task
            or target not in row.get("present_prompt_ids", [])
            or row.get("original_bddl_file") != original_name
            or row.get("original_bddl_sha256") != original_sha
            or row.get("original_prompt") != PROMPTS[task]
            or row.get("counterfactual_prompt") != PROMPTS[target]
            or len(str(row.get("rewritten_bddl_sha256", ""))) != 64
        ):
            errors.append("mapping identity differs")
            break
        rewritten = Path(row.get("rewritten_bddl_path", ""))
        if not rewritten.is_file() or file_sha256(rewritten) != row["rewritten_bddl_sha256"]:
            errors.append("rewritten BDDL identity differs")
            break
    if errors:
        raise RuntimeError("Invalid goal-following manifest: " + "; ".join(errors))
    return result


def synthetic_known_answer() -> dict[str, Any]:
    order = condition_order(2, 9, 29, 3)
    mcnemar = one_sided_exact_mcnemar_p(10, 0)
    signflip = one_sided_exact_signflip_p([2, 1, -1, 3])
    expected_order = [
        "original_goal_original_prompt",
        "counterfactual_goal_original_prompt",
        "counterfactual_goal_counterfactual_prompt",
        "original_goal_counterfactual_prompt",
    ]
    if order != expected_order or mcnemar != 0.0009765625 or signflip != 0.1875:
        raise RuntimeError("Counterfactual goal-following known answer changed")
    return {"order": order, "mcnemar_p": mcnemar, "signflip_p": signflip}


if __name__ == "__main__":
    print(json.dumps(synthetic_known_answer(), indent=2, sort_keys=True))
