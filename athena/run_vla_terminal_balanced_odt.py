#!/usr/bin/env python3
"""Audited balanced ODT at the exact post-norm linear VLA endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import socket
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from athena.libero_dataset_metadata import load_dataset_task_languages
from athena.run_global_bond_rank_audit import (
    DATASET_TASKS,
    DATASET_TO_OFFICIAL_TASK,
    TASK_METADATA,
)
from athena.run_xvla_experiment import (
    array_sha256,
    build_vocab,
    load_suite,
    load_cache_statistics,
    make_config,
    task_languages,
)
from xvla.models.vla import ChiVLA
from xvla.nn.normalization import RationalNorm
from xvla.train.vla_terminal_balanced_odt import (
    BalancedSupportedEigensystem,
    balanced_haar_projector,
    balanced_projector,
    balanced_supported_eigensystem,
    balanced_supported_from_observation_factor,
    centered_flattened_reachable_factor,
    centered_reachable_metrics,
    compile_linear_action_endpoint,
    cross_covariance_projector,
    euclidean_principal_projector,
    intervene_post_norm_queries,
    match_removal_doses,
)


SCHEMA = "xvla-vla-terminal-balanced-odt-v1"
PANEL_NAMESPACE = "xvla-terminal-balanced-odt-v1"
HORIZON = 8
ACTION_DIM = 7
STATE_DIM = 8
PANEL_WINDOWS_PER_TASK = 64
MAX_WINDOWS_PER_EPISODE = 4
EXPECTED_CACHE_SHA256 = "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662"
EXPECTED_CHECKPOINTS = {
    0: ("ckpt_linear_rat_vit_s0_v2.pt", "96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9"),
    1: ("ckpt_linear_rat_vit_s1.pt", "cc0780b989165a80a449c03cbb3574e2f56b2e5747a64cf32d1fde418df8ec3c"),
    2: ("ckpt_linear_rat_vit_s2.pt", "413a770071bd8f16b924c5604f7c7a58567d2eb6c40c66aab202bcf046fec910"),
}
EXPECTED_LEGACY_ARRAY_HASHES = {
    "action_mean": "8bf27cf3b90806945e70e6c79c65ac9c46eb992458f81bcd0086779085a8bf1c",
    "action_std": "0308b1e948fb96c8f55c2339147d559b0ae2f1a9830882de891c6d139b29263e",
    "state_mean": "26f8b764ad80b3cce8aafb690b348ca5e5cced7760e9c1ea9c4e3fd155452715",
    "state_std": "f1a98fde137cc8eb2787a8b7d181341faf5752f0872ea492ffcf0f58104a9d7a",
}
ACTION_GROUPS = {
    "all_controls": tuple(range(7)),
    "translation": (0, 1, 2),
    "rotation": (3, 4, 5),
    "gripper_margin": (6,),
}
ALPHA_GRID = (1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01)
HAAR_DRAWS = 5
TESTED_RANK_CAP = 4
RANK_TOLERANCES = (1e-8, 1e-10, 1e-12)
EXPECTED_RUNTIME = {
    "python": "3.10.19",
    "torch": "2.7.1+cu126",
    "cuda": "12.6",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_sha256(record: Any) -> str:
    digest = hashlib.sha256()
    for value in (int(record[0]), int(record[1]), int(record[5])):
        digest.update(value.to_bytes(8, "little", signed=True))
    for value in record[2:5]:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def window_sha256(window: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(PANEL_NAMESPACE.encode())
    for key in ("task", "episode", "frame"):
        digest.update(int(window[key]).to_bytes(8, "little", signed=True))
    digest.update(bytes.fromhex(window["row_sha256"]))
    digest.update(np.ascontiguousarray(window["actions"]).tobytes())
    return digest.hexdigest()


def load_validated_windows(cache: Path) -> tuple[list[Any], dict[str, list[dict[str, Any]]], dict[str, dict[str, np.ndarray]]]:
    with cache.open("rb") as handle:
        frames = pickle.load(handle)
    if not isinstance(frames, (list, tuple)) or len(frames) != 66_984:
        raise RuntimeError("Object cache must contain exactly 66,984 records")
    episodes: dict[int, list[Any]] = defaultdict(list)
    seen_rows: set[tuple[int, int]] = set()
    for record in frames:
        if not isinstance(record, (list, tuple)) or len(record) != 6:
            raise RuntimeError("cache contains a noncanonical record")
        episode, frame, image, state, action, task = record
        identity = (int(episode), int(frame))
        if identity in seen_rows:
            raise RuntimeError(f"duplicate cache row {identity}")
        seen_rows.add(identity)
        image_array = np.asarray(image)
        state_array = np.asarray(state)
        action_array = np.asarray(action)
        if image_array.dtype != np.uint8 or image_array.shape != (64, 64, 3):
            raise RuntimeError("cache image contract failed")
        if state_array.shape != (STATE_DIM,) or not np.isfinite(state_array).all():
            raise RuntimeError("cache state contract failed")
        if action_array.shape != (ACTION_DIM,) or not np.isfinite(action_array).all():
            raise RuntimeError("cache action contract failed")
        if not 0 <= int(task) < 10:
            raise RuntimeError("cache task is out of range")
        episodes[int(episode)].append(record)
    if len(episodes) != 454:
        raise RuntimeError(f"expected 454 episodes, got {len(episodes)}")

    universes: dict[str, list[dict[str, Any]]] = {"legacy_exclusive": [], "inclusive": []}
    for episode_id, rows in episodes.items():
        rows.sort(key=lambda item: int(item[1]))
        tasks = {int(row[5]) for row in rows}
        indices = [int(row[1]) for row in rows]
        if len(tasks) != 1:
            raise RuntimeError(f"episode {episode_id} mixes tasks")
        if any(right != left + 1 for left, right in zip(indices, indices[1:])):
            raise RuntimeError(f"episode {episode_id} frame indices are not contiguous")
        for policy, stop in (
            ("legacy_exclusive", len(rows) - HORIZON),
            ("inclusive", len(rows) - HORIZON + 1),
        ):
            for index in range(max(0, stop)):
                row = rows[index]
                window = {
                    "episode": episode_id,
                    "frame": int(row[1]),
                    "task": int(row[5]),
                    "image": np.asarray(row[2], dtype=np.uint8),
                    "state": np.asarray(row[3], dtype=np.float32),
                    "actions": np.stack(
                        [rows[index + offset][4] for offset in range(HORIZON)]
                    ).astype(np.float32),
                    "row_sha256": row_sha256(row),
                }
                window["window_sha256"] = window_sha256(window)
                universes[policy].append(window)
    if len(universes["legacy_exclusive"]) != 63_352:
        raise RuntimeError("legacy-exclusive window count differs from 63,352")
    if len(universes["inclusive"]) != 63_806:
        raise RuntimeError("inclusive window count differs from 63,806")

    statistics = {}
    for policy, windows in universes.items():
        actions = np.stack([window["actions"] for window in windows])
        states = np.stack([window["state"] for window in windows])
        statistics[policy] = {
            "action_mean": actions.mean((0, 1)).astype(np.float32),
            "action_std": (actions.std((0, 1)) + 1e-6).astype(np.float32),
            "state_mean": states.mean(0).astype(np.float32),
            "state_std": (states.std(0) + 1e-6).astype(np.float32),
        }
    return list(frames), universes, statistics


def choose_panel(windows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_task_episode: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for window in windows:
        by_task_episode[window["task"]][window["episode"]].append(window)
    panels = {"discovery": [], "evaluation": []}
    episode_records: dict[str, dict[str, list[int]]] = {"discovery": {}, "evaluation": {}}
    for task in range(10):
        ranked_episodes = sorted(
            by_task_episode[task],
            key=lambda episode: hashlib.sha256(
                f"{PANEL_NAMESPACE}|episode|{task}|{episode}".encode()
            ).hexdigest(),
        )
        split_episodes = {
            "discovery": ranked_episodes[::2],
            "evaluation": ranked_episodes[1::2],
        }
        for split, episode_ids in split_episodes.items():
            episode_records[split][str(task)] = episode_ids
            candidates = []
            for episode in episode_ids:
                ordered = sorted(
                    by_task_episode[task][episode], key=lambda window: window["window_sha256"]
                )[:MAX_WINDOWS_PER_EPISODE]
                candidates.extend(ordered)
            selected = sorted(candidates, key=lambda window: window["window_sha256"])[
                :PANEL_WINDOWS_PER_TASK
            ]
            if len(selected) != PANEL_WINDOWS_PER_TASK:
                raise RuntimeError(f"insufficient {split} windows for task {task}")
            panels[split].extend(selected)
    discovery_episodes = {window["episode"] for window in panels["discovery"]}
    evaluation_episodes = {window["episode"] for window in panels["evaluation"]}
    if discovery_episodes & evaluation_episodes:
        raise RuntimeError("discovery and evaluation panels share episodes")
    if {window["window_sha256"] for window in panels["discovery"]} & {
        window["window_sha256"] for window in panels["evaluation"]
    }:
        raise RuntimeError("discovery and evaluation panels share windows")
    records = {
        split: [
            {
                "dataset_task": window["task"],
                "official_task": DATASET_TO_OFFICIAL_TASK[window["task"]],
                "episode": window["episode"],
                "frame": window["frame"],
                "horizon": HORIZON,
                "row_sha256": window["row_sha256"],
                "window_sha256": window["window_sha256"],
            }
            for window in panel
        ]
        for split, panel in panels.items()
    }
    panel_sha = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return panels, {"records": records, "episodes": episode_records, "sha256": panel_sha}


def tensorize(windows: list[dict[str, Any]], encode, stats: dict[str, np.ndarray]) -> tuple[torch.Tensor, ...]:
    images = torch.from_numpy(np.stack([window["image"] for window in windows])).permute(0, 3, 1, 2)
    instructions = torch.tensor(
        [encode(DATASET_TASKS[window["task"]]) for window in windows], dtype=torch.int64
    )
    states = np.stack([window["state"] for window in windows])
    states = (states - stats["state_mean"]) / stats["state_std"]
    embodiments = torch.zeros(len(windows), dtype=torch.int64)
    targets = torch.from_numpy(np.stack([window["actions"] for window in windows]))
    return images.float().div(255), instructions, torch.from_numpy(states).float(), embodiments, targets


@torch.no_grad()
def collect_queries(model: ChiVLA, panel: tuple[torch.Tensor, ...], batch_size: int) -> tuple[torch.Tensor, dict[str, float]]:
    query_batches = []
    max_forward_error = 0.0
    relative_numerator = 0.0
    relative_denominator = 0.0
    for start in range(0, panel[0].shape[0], batch_size):
        inputs = [value[start : start + batch_size].cuda(non_blocking=True) for value in panel[:4]]
        _, queries = model._encode(*inputs)
        direct = model.action_head(queries)
        forward, _ = model(*inputs)
        difference = forward - direct
        max_forward_error = max(max_forward_error, float(difference.abs().max().item()))
        relative_numerator += float(difference.double().square().sum().item())
        relative_denominator += float(forward.double().square().sum().item())
        query_batches.append(queries.double().cpu())
    return torch.cat(query_batches), {
        "maximum_absolute_forward_vs_endpoint_error": max_forward_error,
        "relative_l2_forward_vs_endpoint_error": (relative_numerator / max(relative_denominator, 1e-300)) ** 0.5,
    }


def energy_rank(values: torch.Tensor, fraction: float) -> int:
    positive = values.clamp_min(0)
    if float(positive.sum()) == 0:
        return 0
    return int(torch.searchsorted(torch.cumsum(positive, 0) / positive.sum(), positive.new_tensor(fraction)).item() + 1)


def system_diagnostics(system: BalancedSupportedEigensystem, observable: torch.Tensor) -> dict[str, Any]:
    rank = system.observable_rank
    if rank:
        state = system.state_directions
        analysis = system.analysis_directions
        dual_error = float((analysis @ state - torch.eye(rank, dtype=state.dtype)).abs().max().item())
        observed = state.T @ observable @ state
        diagonal_error = float((observed - torch.diag(system.eigenvalues[:rank])).abs().max().item())
        projector_error = float((system.projector @ system.projector - system.projector).abs().max().item())
    else:
        dual_error = diagonal_error = projector_error = 0.0
    return {
        "support_rank": system.support_rank,
        "observable_rank": rank,
        "balanced_eigenvalues": system.eigenvalues.tolist(),
        "energy_rank_90": energy_rank(system.eigenvalues, 0.90),
        "energy_rank_95": energy_rank(system.eigenvalues, 0.95),
        "energy_rank_99": energy_rank(system.eigenvalues, 0.99),
        "maximum_dual_error": dual_error,
        "maximum_state_observability_diagonal_error": diagonal_error,
        "maximum_projector_idempotence_error": projector_error,
    }


def system_diagnostics_from_map(
    system: BalancedSupportedEigensystem, observation_map: torch.Tensor
) -> dict[str, Any]:
    rank = system.observable_rank
    if rank:
        state = system.state_directions
        analysis = system.analysis_directions
        dual_error = float(
            (analysis @ state - torch.eye(rank, dtype=state.dtype)).abs().max().item()
        )
        observed_state = observation_map @ state
        observed = observed_state.T @ observed_state
        diagonal_error = float(
            (observed - torch.diag(system.eigenvalues[:rank])).abs().max().item()
        )
        dual_residual = analysis @ state - torch.eye(rank, dtype=state.dtype)
        projector_error = float((state @ dual_residual @ analysis).abs().max().item())
    else:
        dual_error = diagonal_error = projector_error = 0.0
    return {
        "support_rank": system.support_rank,
        "observable_rank": rank,
        "balanced_eigenvalues": system.eigenvalues.tolist(),
        "energy_rank_90": energy_rank(system.eigenvalues, 0.90),
        "energy_rank_95": energy_rank(system.eigenvalues, 0.95),
        "energy_rank_99": energy_rank(system.eigenvalues, 0.99),
        "maximum_dual_error": dual_error,
        "maximum_state_observability_diagonal_error": diagonal_error,
        "maximum_projector_idempotence_error": projector_error,
    }


def rank_stability_from_map(
    reachable_root: torch.Tensor,
    observation_map: torch.Tensor,
) -> dict[str, Any]:
    rows = []
    for tolerance in RANK_TOLERANCES:
        system = balanced_supported_from_observation_factor(
            reachable_root, observation_map, relative_tolerance=tolerance
        )
        rows.append(
            {
                "relative_tolerance": tolerance,
                "support_rank": system.support_rank,
                "observable_rank": system.observable_rank,
            }
        )
    return {"rows": rows, "stable": len({(row["support_rank"], row["observable_rank"]) for row in rows}) == 1}


def rank_stability_dense(
    reachable_metric: torch.Tensor,
    observability_metric: torch.Tensor,
) -> dict[str, Any]:
    rows = []
    for tolerance in RANK_TOLERANCES:
        system = balanced_supported_eigensystem(
            reachable_metric, observability_metric, relative_tolerance=tolerance
        )
        rows.append(
            {
                "relative_tolerance": tolerance,
                "support_rank": system.support_rank,
                "observable_rank": system.observable_rank,
            }
        )
    return {"rows": rows, "stable": len({(row["support_rank"], row["observable_rank"]) for row in rows}) == 1}


def intervention_delta_error(
    endpoint,
    states: torch.Tensor,
    means: torch.Tensor,
    projector: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    edited = intervene_post_norm_queries(states, means, projector)
    actual = endpoint.control_proposals(edited) - endpoint.control_proposals(states)
    if projector.shape[0] == states.shape[-1]:
        predicted = torch.einsum(
            "ad,bhd->bha",
            endpoint.weight_control,
            torch.einsum("de,bhe->bhd", projector - torch.eye(projector.shape[0]), states - means),
        )
    else:
        jacobian = endpoint.chunk_jacobian()
        delta_state = (states - means).reshape(states.shape[0], -1)
        predicted = (delta_state @ (projector - torch.eye(projector.shape[0])).T @ jacobian.T).reshape_as(actual)
    error = float((actual - predicted).abs().max().item())
    rms = {
        name: float(actual[..., indices].square().mean().sqrt().item())
        for name, indices in ACTION_GROUPS.items()
    }
    return error, rms


def intervention_delta_error_factors(
    endpoint,
    states: torch.Tensor,
    means: torch.Tensor,
    state_factor: torch.Tensor,
    analysis_factor: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    """Check a full-chunk keep edit without constructing its dense projector."""

    centered = (states - means).reshape(states.shape[0], -1)
    projected = (centered @ analysis_factor.T) @ state_factor.T
    edited = means + projected.reshape_as(states)
    actual = endpoint.control_proposals(edited) - endpoint.control_proposals(states)
    jacobian = endpoint.chunk_jacobian()
    predicted = ((projected - centered) @ jacobian.T).reshape_as(actual)
    error = float((actual - predicted).abs().max().item())
    rms = {
        name: float(actual[..., indices].square().mean().sqrt().item())
        for name, indices in ACTION_GROUPS.items()
    }
    return error, rms


def save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def analyze_checkpoint(
    seed: int,
    checkpoint: Path,
    panels: dict[str, tuple[torch.Tensor, ...]],
    stats: dict[str, np.ndarray],
    artifact_dir: Path,
    batch_size: int,
) -> tuple[dict[str, Any], bool]:
    expected_name, expected_sha = EXPECTED_CHECKPOINTS[seed]
    if checkpoint.name != expected_name or file_sha256(checkpoint) != expected_sha:
        raise RuntimeError(f"checkpoint seed {seed} identity mismatch")
    vocab, _ = build_vocab(DATASET_TASKS)
    config = make_config("chi", len(vocab), STATE_DIM, ACTION_DIM, 64, HORIZON, "vit")
    expected_config = {
        "vision_encoder": "vit", "dual_vision": False, "attn": "bilinear",
        "ffn": "bilinear", "norm": "rational", "qk_norm": "rational",
        "action_head": "linear", "n_phases": 0, "distill_teacher": False,
        "value_head": False, "vit_dim": 192, "vit_layers": 4, "vit_heads": 8,
        "dim": 384, "n_layers": 8, "n_heads": 12, "action_horizon": 8,
        "action_dim": 7, "max_instr_len": 32,
    }
    if any(getattr(config, key) != value for key, value in expected_config.items()):
        raise RuntimeError("rational checkpoint config differs from frozen endpoint contract")
    model = ChiVLA(config).cuda().eval()
    if model.seq_len != 107 or type(model.action_head) is not nn.Linear:
        raise RuntimeError("model endpoint structure mismatch")
    state = torch.load(checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(state, strict=True)
    if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise RuntimeError("checkpoint contains nonfinite parameters")
    rational_sites = {
        name: module for name, module in model.named_modules() if isinstance(module, RationalNorm)
    }
    executed_rational_paths: list[str] = []
    handles = [
        module.register_forward_pre_hook(
            lambda _module, _inputs, path=name: executed_rational_paths.append(path)
        )
        for name, module in rational_sites.items()
    ]
    try:
        probe = [value[:1].cuda() for value in panels["discovery"][:4]]
        with torch.no_grad():
            model._encode(*probe)
    finally:
        for handle in handles:
            handle.remove()
    if len(executed_rational_paths) != len(set(executed_rational_paths)):
        raise RuntimeError("a RationalNorm site executes more than once per endpoint forward")
    executed_set = set(executed_rational_paths)
    unexecuted_set = set(rational_sites) - executed_set
    if unexecuted_set != {"vision.norm_out"}:
        raise RuntimeError(f"unexpected dead RationalNorm paths: {sorted(unexecuted_set)}")
    if not executed_set or any(not bool(rational_sites[name].initialized) for name in executed_set):
        raise RuntimeError("an active RationalNorm buffer is uninitialized")
    rational_before = {
        f"{name}.{buffer_name}": buffer.detach().cpu().clone()
        for name, module in rational_sites.items() if name in executed_set
        for buffer_name, buffer in module.named_buffers(recurse=False)
    }
    discovery, discovery_replay = collect_queries(model, panels["discovery"], batch_size)
    evaluation, evaluation_replay = collect_queries(model, panels["evaluation"], batch_size)
    rational_after = {
        f"{name}.{buffer_name}": buffer.detach().cpu()
        for name, module in rational_sites.items() if name in executed_set
        for buffer_name, buffer in module.named_buffers(recurse=False)
    }
    rational_unchanged = all(torch.equal(value, rational_after[name]) for name, value in rational_before.items())

    endpoint32 = compile_linear_action_endpoint(
        model,
        torch.from_numpy(stats["action_mean"]).cuda(),
        torch.from_numpy(stats["action_std"]).cuda(),
    )
    endpoint = compile_linear_action_endpoint(
        model,
        torch.from_numpy(stats["action_mean"]).cuda(),
        torch.from_numpy(stats["action_std"]).cuda(),
        dtype=torch.float64,
    ).to(dtype=torch.float64, device="cpu")
    normalized = endpoint.normalized(evaluation)
    controls = endpoint.control_proposals(evaluation)
    explicit_controls = normalized * endpoint.action_std + endpoint.action_mean
    endpoint_control_error = float((controls - explicit_controls).abs().max().item())
    evaluation32 = evaluation.float().cuda()
    deployed_control_error = float(
        (
            endpoint32.control_proposals(evaluation32)
            - (endpoint32.normalized(evaluation32) * endpoint32.action_std + endpoint32.action_mean)
        ).abs().max().item()
    )
    means, covariances = centered_reachable_metrics(discovery)
    flattened = centered_flattened_reachable_factor(discovery)
    jacobian = endpoint.chunk_jacobian()
    artifact_arrays: dict[str, np.ndarray] = {
        "discovery_mean": means.numpy(),
        "discovery_per_horizon_covariance": covariances.numpy(),
        "full_reachable_root": flattened.root.numpy(),
        "full_reachable_singular_values": flattened.singular_values.numpy(),
    }
    metrics: dict[str, Any] = {}
    maximum_algebra_error = 0.0
    maximum_dual_error = 0.0
    maximum_projector_error = 0.0
    maximum_observability_error = 0.0
    all_rank_gates = True
    all_dose_gates = True
    for metric_index, (metric_name, action_indices) in enumerate(ACTION_GROUPS.items()):
        selected_rows = [time * ACTION_DIM + action for time in range(HORIZON) for action in action_indices]
        observation_map = jacobian[selected_rows]
        full_system = balanced_supported_from_observation_factor(flattened.root, observation_map)
        full_diagnostics = system_diagnostics_from_map(full_system, observation_map)
        full_stability = rank_stability_from_map(flattened.root, observation_map)
        expected_full_observable_rank = HORIZON * len(action_indices)
        full_rank = min(TESTED_RANK_CAP, full_system.observable_rank)
        full_delta_error, full_delta_rms = intervention_delta_error_factors(
            endpoint,
            evaluation,
            flattened.mean,
            full_system.state_directions[:, :full_rank],
            full_system.analysis_directions[:full_rank],
        )
        maximum_algebra_error = max(maximum_algebra_error, full_delta_error)
        maximum_dual_error = max(maximum_dual_error, full_diagnostics["maximum_dual_error"])
        maximum_projector_error = max(
            maximum_projector_error, full_diagnostics["maximum_projector_idempotence_error"]
        )
        maximum_observability_error = max(
            maximum_observability_error,
            full_diagnostics["maximum_state_observability_diagonal_error"],
        )
        all_rank_gates = all_rank_gates and (
            full_system.support_rank == 639
            and full_system.observable_rank == expected_full_observable_rank
            and full_stability["stable"]
        )
        artifact_arrays[f"{metric_name}_full_state_directions"] = full_system.state_directions.numpy()
        artifact_arrays[f"{metric_name}_full_analysis_directions"] = full_system.analysis_directions.numpy()
        per_horizon = []
        local_observable = endpoint.weight_control[list(action_indices)].T @ endpoint.weight_control[list(action_indices)]
        observation_local = endpoint.weight_control[list(action_indices)]
        control_projectors: dict[str, list[torch.Tensor]] = {
            "balanced_top": [],
            "reachable_only": [],
            "observable_only": [],
            "cross_cov_svd": [],
            **{f"balanced_haar_d{draw}": [] for draw in range(HAAR_DRAWS)},
        }
        svd_oracle_error = 0.0
        for time in range(HORIZON):
            system = balanced_supported_eigensystem(covariances[time], local_observable)
            diagnostics = system_diagnostics(system, local_observable)
            stability = rank_stability_dense(covariances[time], local_observable)
            rank = min(TESTED_RANK_CAP, system.observable_rank)
            projector = balanced_projector(system, rank)
            delta_error, delta_rms = intervention_delta_error(
                endpoint, evaluation[:, time : time + 1], means[time : time + 1], projector
            )
            maximum_algebra_error = max(maximum_algebra_error, delta_error)
            maximum_dual_error = max(maximum_dual_error, diagnostics["maximum_dual_error"])
            maximum_projector_error = max(
                maximum_projector_error, diagnostics["maximum_projector_idempotence_error"]
            )
            maximum_observability_error = max(
                maximum_observability_error,
                diagnostics["maximum_state_observability_diagonal_error"],
            )
            all_rank_gates = all_rank_gates and (
                system.support_rank == config.dim
                and system.observable_rank == len(action_indices)
                and stability["stable"]
            )
            artifact_arrays[f"{metric_name}_h{time}_state_directions"] = system.state_directions.numpy()
            artifact_arrays[f"{metric_name}_h{time}_analysis_directions"] = system.analysis_directions.numpy()
            control_projectors["balanced_top"].append(projector)
            control_projectors["reachable_only"].append(
                euclidean_principal_projector(covariances[time], rank)
            )
            observable_projector = euclidean_principal_projector(local_observable, rank)
            _, _, right = torch.linalg.svd(observation_local, full_matrices=False)
            svd_projector = right[:rank].T @ right[:rank]
            svd_oracle_error = max(
                svd_oracle_error,
                float((observable_projector - svd_projector).abs().max().item()),
            )
            control_projectors["observable_only"].append(observable_projector)
            control_projectors["cross_cov_svd"].append(
                cross_covariance_projector(covariances[time], observation_local, rank)
            )
            for draw in range(HAAR_DRAWS):
                control_projectors[f"balanced_haar_d{draw}"].append(
                    balanced_haar_projector(
                        system,
                        rank,
                        seed=seed * 100_000 + metric_index * 1_000 + draw * 100 + time,
                    )
                )
            per_horizon.append(
                {**diagnostics, "time_index": time, "tested_rank": rank,
                 "rank_stability": stability,
                 "maximum_action_delta_identity_error": delta_error,
                 "heldout_action_delta_rms": delta_rms}
            )
        stacked_projectors = {
            name: torch.stack(projectors) for name, projectors in control_projectors.items()
        }
        dose_match = match_removal_doses(
            endpoint,
            evaluation,
            means,
            stacked_projectors,
            primary="balanced_top",
            alpha_grid=ALPHA_GRID,
            action_indices=action_indices,
            control_units=False,
        )
        dose_error = max(
            abs(value - dose_match.target_dose) for value in dose_match.realized_doses.values()
        )
        all_dose_gates = all_dose_gates and dose_error <= 1e-12 and svd_oracle_error <= 1e-10
        control_record = {
            "dose_units": "normalized_precommit_action_proposals",
            "primary": dose_match.primary,
            "alpha_grid": list(ALPHA_GRID),
            "target_dose": dose_match.target_dose,
            "unscaled_doses": dose_match.unscaled_doses,
            "alphas": dose_match.alphas,
            "realized_doses": dose_match.realized_doses,
            "maximum_realized_dose_error": dose_error,
            "observable_only_vs_direct_right_svd_projector_error": svd_oracle_error,
            "projector_sha256": {
                name: array_sha256(projector.numpy())
                for name, projector in stacked_projectors.items()
            },
            "haar_seeds": {
                f"balanced_haar_d{draw}": [
                    seed * 100_000 + metric_index * 1_000 + draw * 100 + time
                    for time in range(HORIZON)
                ]
                for draw in range(HAAR_DRAWS)
            },
        }
        metrics[metric_name] = {
            "meaning": (
                "Exact affine gripper-margin metric before out-of-graph sign commitment."
                if metric_name == "gripper_margin"
                else "Exact denormalized LIBERO control-proposal metric."
            ),
            "action_indices": list(action_indices),
            "full_chunk": {
                **full_diagnostics,
                "tested_rank": full_rank,
                "rank_stability": full_stability,
                "maximum_action_delta_identity_error": full_delta_error,
                "heldout_action_delta_rms": full_delta_rms,
            },
            "per_horizon": per_horizon,
            "rank_matched_controls": control_record,
        }
    identity_edit = intervene_post_norm_queries(
        evaluation, means, torch.eye(config.dim, dtype=torch.float64), alpha=1.0
    )
    alpha_zero_edit = intervene_post_norm_queries(
        evaluation, means, torch.zeros(config.dim, config.dim, dtype=torch.float64), alpha=0.0
    )
    identity_error = max(
        float((identity_edit - evaluation).abs().max().item()),
        float((alpha_zero_edit - evaluation).abs().max().item()),
    )
    artifact_path = artifact_dir / f"terminal_balanced_odt_seed{seed}.npz"
    save_npz_atomic(artifact_path, artifact_arrays)
    checks = {
        "forward_endpoint_replay_below_2e-5": max(
            discovery_replay["maximum_absolute_forward_vs_endpoint_error"],
            evaluation_replay["maximum_absolute_forward_vs_endpoint_error"],
        ) <= 2e-5,
        # The equality above is evaluated from a directly compiled float64
        # endpoint and should be exact up to float64 arithmetic.  Deployed
        # float32 reassociation is gated separately below.
        "float64_denormalized_endpoint_algebra_below_1e-12": endpoint_control_error <= 1e-12,
        "deployed_float32_compiled_endpoint_below_2e-6": deployed_control_error <= 2e-6,
        "rational_buffers_initialized_and_unchanged": rational_unchanged,
        "balanced_dual_error_below_1e-8": maximum_dual_error <= 1e-8,
        "projector_idempotence_error_below_1e-8": maximum_projector_error <= 1e-8,
        "observability_diagonal_error_below_1e-8": maximum_observability_error <= 1e-8,
        "support_and_observable_ranks_stable_and_expected": all_rank_gates,
        "control_doses_and_svd_oracle_below_1e-10": all_dose_gates,
        "intervention_action_delta_error_below_1e-10": maximum_algebra_error <= 1e-10,
        "identity_interventions_below_1e-12": identity_error <= 1e-12,
        "artifact_is_finite": all(np.isfinite(value).all() for value in artifact_arrays.values()),
    }
    return {
        "seed": seed,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": expected_sha,
        "configuration": expected_config,
        "rational_site_count": len(rational_sites),
        "executed_rational_paths": executed_rational_paths,
        "proven_dead_rational_paths": sorted(unexecuted_set),
        "endpoint_replay": {"discovery": discovery_replay, "evaluation": evaluation_replay},
        "denormalized_proposal_reconstruction_maximum_absolute_error": endpoint_control_error,
        "deployed_float32_compiled_endpoint_maximum_absolute_error": deployed_control_error,
        "normalized_gripper_threshold": float(endpoint.normalized_gripper_threshold.item()),
        "metrics": metrics,
        "artifact": {
            "path": artifact_path.as_posix(),
            "sha256": file_sha256(artifact_path),
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": array_sha256(value)}
                for name, value in artifact_arrays.items()
            },
        },
        "checks": checks,
    }, all(checks.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if len(args.checkpoint) != 3 or args.batch_size <= 0:
        raise ValueError("exactly three checkpoints and a positive batch size are required")
    if args.output.exists() or args.artifact_dir.exists():
        raise FileExistsError("refusing to overwrite terminal ODT outputs")
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if runtime != EXPECTED_RUNTIME:
        raise RuntimeError(f"runtime identity mismatch: {runtime}")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be frozen to :4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if file_sha256(args.cache) != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache identity mismatch")
    official_tasks = task_languages(load_suite("libero_object"))
    dataset_tasks, task_metadata = load_dataset_task_languages(
        "libero_object", official_tasks
    )
    if dataset_tasks != DATASET_TASKS:
        raise RuntimeError("live pinned dataset languages differ from frozen cache languages")
    if task_metadata != TASK_METADATA:
        raise RuntimeError("live task metadata provenance differs from frozen identity")
    _, universes, statistics = load_validated_windows(args.cache)
    legacy = statistics["legacy_exclusive"]
    legacy_hashes = {name: array_sha256(value) for name, value in legacy.items()}
    if legacy_hashes != EXPECTED_LEGACY_ARRAY_HASHES:
        raise RuntimeError(f"legacy normalization hashes mismatch: {legacy_hashes}")
    imported = load_cache_statistics(args.cache, HORIZON)
    if imported["sample_count"] != 63_352 or any(
        not np.array_equal(imported[name], legacy[name]) for name in EXPECTED_LEGACY_ARRAY_HASHES
    ):
        raise RuntimeError("independent legacy stats do not match training helper bitwise")
    raw_panels, panel_record = choose_panel(universes["legacy_exclusive"])
    vocab, encode = build_vocab(DATASET_TASKS)
    if len(vocab) <= 2:
        raise RuntimeError("vocabulary construction failed")
    panels = {
        split: tensorize(windows, encode, legacy) for split, windows in raw_panels.items()
    }
    cache_roundtrip_errors = {}
    action_mean = torch.from_numpy(legacy["action_mean"])
    action_std = torch.from_numpy(legacy["action_std"])
    for split, panel in panels.items():
        targets = panel[4]
        reconstructed = ((targets - action_mean) / action_std) * action_std + action_mean
        cache_roundtrip_errors[split] = float((reconstructed - targets).abs().max().item())
    cache_roundtrip_passed = max(cache_roundtrip_errors.values()) <= 2e-6
    args.artifact_dir.mkdir(parents=True)
    checkpoint_rows = []
    passed = cache_roundtrip_passed
    checkpoint_by_name = {path.name: path for path in args.checkpoint}
    for seed, (name, _) in EXPECTED_CHECKPOINTS.items():
        if name not in checkpoint_by_name:
            raise RuntimeError(f"missing checkpoint {name}")
        row, row_passed = analyze_checkpoint(
            seed, checkpoint_by_name[name], panels, legacy, args.artifact_dir, args.batch_size
        )
        checkpoint_rows.append(row)
        passed = passed and row_passed
        torch.cuda.empty_cache()
    inclusive = statistics["inclusive"]
    stats_audit = {
        "primary_policy": "legacy_exclusive_range_len_minus_horizon",
        "primary_sample_count": len(universes["legacy_exclusive"]),
        "primary_array_sha256": legacy_hashes,
        "sensitivity_policy": "inclusive_range_len_minus_horizon_plus_one",
        "sensitivity_sample_count": len(universes["inclusive"]),
        "sensitivity_array_sha256": {name: array_sha256(value) for name, value in inclusive.items()},
        "sensitivity_maximum_absolute_delta": {
            name: float(np.max(np.abs(inclusive[name].astype(np.float64) - legacy[name].astype(np.float64))))
            for name in EXPECTED_LEGACY_ARRAY_HASHES
        },
    }
    manifest = Path("athena/vla_terminal_balanced_odt_sources.sha256")
    payload = {
        "schema": SCHEMA,
        "frozen_acceptance_gates": {"passed": passed},
        "provenance": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cache": args.cache.as_posix(),
            "cache_sha256": EXPECTED_CACHE_SHA256,
            "source_manifest": manifest.as_posix(),
            "source_manifest_sha256": file_sha256(manifest),
            "source_tree_sha256": os.environ.get("XVLA_SOURCE_TREE_SHA256"),
            "pip_freeze_sha256": os.environ.get("XVLA_PIP_FREEZE_SHA256"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "task_metadata": task_metadata,
            "live_official_task_languages": {str(key): value for key, value in official_tasks.items()},
        },
        "window_statistics_audit": stats_audit,
        "cache_float32_normalize_denormalize_roundtrip": {
            "maximum_absolute_error_by_panel": cache_roundtrip_errors,
            "below_2e-6": cache_roundtrip_passed,
            "scope": "representation-boundary audit, independent of endpoint predictions",
        },
        "panel": panel_record,
        "checkpoints": checkpoint_rows,
        "claim_boundary": (
            "The endpoint after the final normalizer is exactly affine. O is exact and weight-derived "
            "in denormalized LIBERO control-proposal units, while C is data-conditioned on the frozen "
            "discovery panel. The rational backbone, gripper sign commitment, and simulator dynamics "
            "are outside this certificate. The action-head rank ceiling is structural, not discovered "
            "deep-policy compression. Evaluation episodes are held out from metric discovery, but not "
            "from the checkpoints' original training cache. This is not global VLA ODT or sufficient "
            "interpretability."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"passed": passed, "panel_sha256": panel_record["sha256"]}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
