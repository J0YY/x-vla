"""Reduced-policy rollout with isolated, explicitly counted chart failures.

Shared preprocessing and controller helpers are imported from the exact source
worker used for the paired baseline.  Only the reduced execution branch differs.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from modal_odt_dimension_curve_worker import (
    HORIZON, MAX_STEPS, PANEL, array_sha256, episode_records, model_inputs,
    physical_inputs, sha256, transition,
)


CHART_FAILURE_MESSAGES = frozenset({
    "mapped projective denominator is numerically zero",
    "mapped projective evaluation produced zero/nonfinite coordinates",
})


@torch.inference_mode()
def evaluate_valid_rows(mapped, raw: dict, rows: list[int]) -> tuple[dict, list[int], list[dict]]:
    selected = {key: value[rows] for key, value in raw.items()}
    started = time.monotonic()
    try:
        result, receipt = mapped.evaluate_boundary_quotient(selected, return_receipt=True)
    except ValueError as error:
        if str(error) not in CHART_FAILURE_MESSAGES:
            raise
        if len(rows) == 1:
            return {}, rows, [{"rows": rows, "numerical_chart_failure": str(error), "seconds": time.monotonic() - started}]
        middle = len(rows) // 2
        left, left_failed, left_receipts = evaluate_valid_rows(mapped, raw, rows[:middle])
        right, right_failed, right_receipts = evaluate_valid_rows(mapped, raw, rows[middle:])
        return {**left, **right}, left_failed + right_failed, left_receipts + right_receipts
    if not receipt.root_only_live or not receipt.all_refcounts_zero:
        raise RuntimeError("mapped occurrence lifetime ledger did not close")
    if tuple(result.shape) != (len(rows), HORIZON * 7) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("reduced tensor-network output contract differs")
    return {row: result[index].reshape(HORIZON, 7) for index, row in enumerate(rows)}, [], [{
        "rows": rows, "seconds": time.monotonic() - started,
        "nodes": receipt.node_evaluations, "edges": receipt.edge_occurrences_emitted,
        "edge_sha256": receipt.edge_ledger_sha256, "peak_live_values": receipt.peak_live_values,
    }]


def run_reduced_panel(mapped, training: dict, protocol: dict, progress, *, smoke: bool = False) -> list[dict]:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    # A reduced smoke measures the actual batch-20 panel, since a batch-one
    # timing does not establish the wide-kernel rollout throughput.
    pairs = PANEL
    max_steps = 1 if smoke else MAX_STEPS
    entries = []
    try:
        for task_index, episode in pairs:
            task = suite.get_task(task_index)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            original_load = torch.load
            def packaged_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return original_load(*args, **kwargs)
            torch.load = packaged_load
            try:
                states = suite.get_task_init_states(task_index)
            finally:
                torch.load = original_load
            expected = protocol["tasks"][str(task_index)]
            if [str(task.language), str(task.bddl_file), sha256(bddl), array_sha256(states)] != expected:
                raise RuntimeError(f"task {task_index} identity differs")
            environment = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
            entry = {"task": task_index, "episode": episode, "environment": environment,
                "instruction": str(task.language), "initial_state_sha256": array_sha256(states[episode]),
                "steps": 0, "success": False, "terminated": False, "invalid_chart": False}
            entries.append(entry)
            environment.seed(task_index * 100 + episode)
            environment.reset()
            observation = environment.set_init_state(states[episode])
            for _ in range(10):
                observation, reward, done, _info = transition(environment, [0., 0., 0., 0., 0., 0., -1.])
                if done or reward > 0:
                    raise RuntimeError("episode terminated during settling")
            entry["observation"] = observation
        inference_index = 0
        while True:
            active = [entry for entry in entries if not entry["success"] and not entry["terminated"] and not entry["invalid_chart"] and entry["steps"] < max_steps]
            if not active:
                break
            images, tokens, states = model_inputs([entry["observation"] for entry in active], [entry["instruction"] for entry in active], training)
            outputs, failed, receipts = evaluate_valid_rows(mapped, physical_inputs(images, tokens, states), list(range(len(active))))
            for row in failed:
                active[row]["invalid_chart"] = True
            normalization = training["normalization"]
            action_mean = torch.tensor(normalization["action_mean"], dtype=torch.float64)
            action_std = torch.tensor(normalization["action_std"], dtype=torch.float64)
            actions = {row: (value * action_std + action_mean).numpy() for row, value in outputs.items()}
            if any(not np.isfinite(value).all() for value in actions.values()):
                raise RuntimeError("reduced action affine produced nonfinite output")
            for offset in range(HORIZON):
                for row, chunk in actions.items():
                    entry = active[row]
                    if entry["success"] or entry["terminated"] or entry["steps"] >= max_steps:
                        continue
                    action = chunk[offset].copy()
                    action[-1] = 1. if action[-1] > 0 else -1.
                    observation, reward, done, _info = transition(entry["environment"], action.tolist())
                    entry.update(observation=observation, steps=entry["steps"] + 1, success=reward > 0, terminated=done)
            inference_index += 1
            records = episode_records(entries)
            for record, entry in zip(records, entries):
                record["invalid_chart"] = entry["invalid_chart"]
            progress({"inference_index": inference_index, "metrics": {"mapped_attempts": receipts}, "episodes": records})
        records = episode_records(entries)
        for record, entry in zip(records, entries):
            record["invalid_chart"] = entry["invalid_chart"]
        return records
    finally:
        for entry in entries:
            entry["environment"].close()
