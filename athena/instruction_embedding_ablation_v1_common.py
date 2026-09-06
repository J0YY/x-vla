#!/usr/bin/env python3
"""Frozen identities and utilities for instruction-embedding ablation v1."""

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
    SOURCE_PATHS as IMPORTED_SOURCE_PATHS,
    decode_array,
    encode_array,
    file_sha256,
    hash_array,
    source_hashes as imported_source_hashes,
    validate_libero_runtime,
    write_json,
)


SCHEMA_MANIFEST = "xvla-instruction-embedding-ablation-manifest-v1"
SCHEMA_RUN = "xvla-instruction-embedding-ablation-run-v1"
SCHEMA_SUMMARY = "xvla-instruction-embedding-ablation-summary-v1"

MANIFEST_PATH = "artifacts/instruction_embedding_ablation_v1_manifest.json"
SMOKE_RESULT_PATH = "results/instruction_embedding_ablation_v1_smoke.json"
SUMMARY_RESULT_PATH = "results/instruction_embedding_ablation_v1_summary.json"

CONDITIONS = ("full_lexical_embeddings", "zeroed_lexical_embeddings")
TASK_SHARDS = ((0, 2), (2, 4), (4, 6), (6, 8), (8, 10))

PROTOCOL = {
    "suite": "libero_object",
    "checkpoint_seeds": [0, 1, 2],
    "task_indices": list(range(10)),
    "canonical_episode_start": 30,
    "canonical_episode_end_exclusive": 40,
    "conditions": list(CONDITIONS),
    "intervention": (
        "A scoped PyTorch forward hook replaces the complete (B,32,d) output of "
        "model.tok_emb with torch.zeros_like(output) after the embedding lookup. "
        "The correct instruction IDs, sequence length, token positions, vision input, "
        "robot-state input, embodiment token, learned model BOS, joint positions, and "
        "action queries are otherwise unchanged."
    ),
    "learned_common_bos_semantics": (
        "model.bos is a separate learned token prepended after visual tokens and before "
        "the 32 instruction positions. It is outside model.tok_emb and is not ablated."
    ),
    "condition_order": (
        "one of two permutations selected by SHA-256 of "
        "instruction-embedding-ablation-v1-order|checkpoint|task|episode"
    ),
    "settle_steps": 10,
    "action_horizon": 8,
    "execution_horizon": 8,
    "max_steps": 280,
    "resolution": 64,
    "instruction_token_count": 32,
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
    "summary_python_executable": (
        "/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python"
    ),
    "summary_cuda_visible_devices": "",
    "task_shards": [list(bounds) for bounds in TASK_SHARDS],
    "checkpoint_state_pairs_per_checkpoint": 100,
    "rollouts_per_checkpoint": 200,
    "unique_paired_states": 100,
    "total_checkpoint_state_pairs": 300,
    "total_rollouts": 600,
}

FROZEN_GATES = {
    "minimum_full_lexical_success_pooled": 0.70,
    "minimum_full_lexical_success_every_checkpoint": 0.60,
    "minimum_pooled_success_gap": 0.20,
    "success_gap_strictly_positive_every_checkpoint": True,
    "one_sided_paired_exact_p_at_most": 0.01,
    "minimum_tasks_with_strictly_positive_gap": 8,
    "negative_task_gaps_allowed": 0,
    "task_stratified_state_cluster_bootstrap_draws": 20000,
    "task_stratified_state_cluster_bootstrap_seed": 2026082701,
    "task_stratified_state_cluster_bootstrap_lower_strictly_above": 0.0,
    "state_cluster_retains_all_three_checkpoint_outcomes": True,
    "all_three_checkpoints_must_pass": True,
}

DESIGN_PROVENANCE = {
    "frozen_before_any_endpoint_outcome": True,
    "design_time_local": "2026-08-21T12:45:00-07:00",
    "designed_after_instruction_necessity_v1_outcomes": True,
    "outcome_informed_rationale": (
        "The prompt-level instruction-necessity result motivated a more precise mechanism "
        "test. No episode-30-through-39 lexical-ablation outcome was inspected when this "
        "protocol, matrix, intervention, or gate was chosen."
    ),
    "fresh_state_partition": {
        "this_study": "canonical episodes 30 through 39",
        "counterfactual_goal_following_v1": "canonical episodes 20 through 29",
        "instruction_necessity_v1": "canonical episodes 40 through 49",
    },
    "gate_rationale": (
        "The 20-point paired gap, 70-percent pooled baseline, 60-percent per-checkpoint "
        "baseline, task consistency, exact paired test, and clustered bootstrap were "
        "declared prospectively for the fresh state partition."
    ),
    "declaration": (
        "This is a prospective causal ablation of one learned input path on held-out "
        "canonical states. It is not a reanalysis of the earlier prompt-control trials."
    ),
}

NEW_SOURCE_PATHS = (
    "athena/INSTRUCTION_EMBEDDING_ABLATION_V1_PROTOCOL.md",
    "athena/instruction_embedding_ablation_v1_common.py",
    "athena/preflight_instruction_embedding_ablation_v1.py",
    "athena/run_instruction_embedding_ablation_v1.py",
    "athena/summarize_instruction_embedding_ablation_v1.py",
    "athena/slurm_preflight_instruction_embedding_ablation_v1.sbatch",
    "athena/slurm_instruction_embedding_ablation_v1.sbatch",
    "athena/slurm_summarize_instruction_embedding_ablation_v1.sbatch",
)
RUNTIME_HELPER_SOURCE_PATHS = (
    "athena/preflight_counterfactual_target_swap.py",
    "athena/run_counterfactual_target_swap.py",
)
SOURCE_PATHS = tuple(
    dict.fromkeys(
        (*IMPORTED_SOURCE_PATHS, *RUNTIME_HELPER_SOURCE_PATHS, *NEW_SOURCE_PATHS)
    )
)


def source_hashes(root: Path) -> dict[str, str]:
    """Validate the frozen imported stack and hash this extension's complete closure."""
    return imported_source_hashes(
        root, extra=(*RUNTIME_HELPER_SOURCE_PATHS, *NEW_SOURCE_PATHS)
    )


def condition_order(seed: int, task: int, episode: int) -> list[str]:
    permutations = list(itertools.permutations(CONDITIONS))
    payload = (
        f"instruction-embedding-ablation-v1-order|{seed}|{task}|{episode}".encode()
    )
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2
    return list(permutations[index])


def one_sided_paired_exact_p(full_only: int, ablated_only: int) -> float:
    """One-sided exact McNemar p-value for full-path success exceeding ablated."""
    if full_only < 0 or ablated_only < 0:
        raise ValueError("Discordant counts must be nonnegative")
    discordant = full_only + ablated_only
    if discordant == 0:
        return 1.0
    numerator = sum(
        math.comb(discordant, k) for k in range(full_only, discordant + 1)
    )
    return numerator / (2**discordant)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def validate_manifest(path: Path, root: Path) -> dict[str, Any]:
    result = load_json(path)
    errors: list[str] = []
    if result.get("schema") != SCHEMA_MANIFEST:
        errors.append("wrong manifest schema")
    if result.get("protocol") != PROTOCOL:
        errors.append("manifest protocol differs")
    if result.get("frozen_gates") != FROZEN_GATES:
        errors.append("manifest gates differ")
    if result.get("design_provenance") != DESIGN_PROVENANCE:
        errors.append("manifest design provenance differs")
    if result.get("checkpoints") != {
        str(seed): row for seed, row in CHECKPOINTS.items()
    }:
        errors.append("manifest checkpoint identities differ")
    if result.get("mapping_count") != 100 or len(result.get("mappings", [])) != 100:
        errors.append("manifest mapping matrix is incomplete")
    expected_keys = {
        (task, episode) for task in range(10) for episode in range(30, 40)
    }
    observed_keys = {
        (int(row["task_index"]), int(row["episode"]))
        for row in result.get("mappings", [])
    }
    if observed_keys != expected_keys:
        errors.append("manifest state key matrix differs")
    cache = result.get("cache", {})
    if cache.get("path") != CACHE["path"] or cache.get("live_sha256") != CACHE["sha256"]:
        errors.append("manifest cache identity differs")
    provenance = result.get("provenance", {})
    if (
        provenance.get("path") != PROVENANCE["path"]
        or str(provenance.get("job_id")) != PROVENANCE["job_id"]
        or provenance.get("verified") is not True
    ):
        errors.append("manifest provenance identity differs")
    live_sources = source_hashes(root)
    if result.get("source_sha256_start") != live_sources:
        errors.append("manifest start-source closure differs")
    if result.get("source_sha256_end") != live_sources:
        errors.append("manifest end-source closure differs")
    if result.get("libero_runtime_start") != LIBERO_RUNTIME:
        errors.append("manifest LIBERO start identity differs")
    if result.get("libero_runtime_end") != LIBERO_RUNTIME:
        errors.append("manifest LIBERO end identity differs")
    if errors:
        raise RuntimeError("; ".join(errors))
    return result
