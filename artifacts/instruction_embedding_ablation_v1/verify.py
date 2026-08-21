#!/usr/bin/env python3
"""Verify the frozen instruction-embedding-path ablation with the standard library."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import struct
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any


ARTIFACT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = ARTIFACT_DIR.parents[1]
CONDITIONS = ("full_lexical_embeddings", "zeroed_lexical_embeddings")
SHARDS = ((0, 2), (2, 4), (4, 6), (6, 8), (8, 10))
RAW_RE = re.compile(r"instruction_embedding_ablation_v1_s([0-2])_t(0|2|4|6|8)_(2|4|6|8|10)\.json")


class VerificationError(RuntimeError):
    """Raised when committed evidence differs from the frozen artifact."""


def require(value: bool, message: str) -> None:
    if not value:
        raise VerificationError(message)


def reject_constant(value: str) -> None:
    raise VerificationError(f"non-finite JSON constant {value}")


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path}: top-level value must be an object")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise VerificationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def product(values: list[int]) -> int:
    total = 1
    for value in values:
        require(type(value) is int and value >= 0, "array shape is malformed")
        total *= value
    return total


def decode_array(value: dict[str, Any]) -> tuple[list[float | bool], list[int]]:
    require(value.get("codec") == "zlib+base64", "array codec differs")
    shape = value.get("shape")
    require(isinstance(shape, list), "array shape is absent")
    try:
        raw = zlib.decompress(base64.b64decode(value["data"], validate=True))
    except Exception as exc:
        raise VerificationError(f"cannot decode array: {exc}") from exc
    dtype = value.get("dtype")
    formats = {"<f8": ("<d", 8), "<f4": ("<f", 4), "|b1": ("<?", 1)}
    require(dtype in formats, f"unsupported array dtype {dtype!r}")
    fmt, width = formats[dtype]
    count = product(shape)
    require(len(raw) == count * width, "array byte count differs from shape")
    digest = hashlib.sha256()
    digest.update(dtype.encode("ascii"))
    digest.update(struct.pack(f"<{len(shape)}q", *shape))
    digest.update(raw)
    require(digest.hexdigest() == value.get("sha256"), "array digest differs")
    return [item[0] for item in struct.iter_unpack(fmt, raw)], shape


def validate_rollout(rollout: dict[str, Any]) -> bool:
    steps = rollout.get("steps")
    require(type(steps) is int and 1 <= steps <= 280, "rollout step count differs")
    actions, action_shape = decode_array(rollout["executed_actions"])
    rewards, reward_shape = decode_array(rollout["rewards"])
    dones, done_shape = decode_array(rollout["dones"])
    require(action_shape == [steps, 7], "executed-action shape differs")
    require(reward_shape == [steps] and done_shape == [steps], "reward or done shape differs")
    require(all(math.isfinite(float(x)) for x in actions + rewards), "non-finite trajectory value")
    require(all(type(x) is bool for x in dones), "done array is not boolean")
    success = any(float(reward) > 0.0 for reward in rewards)
    require(rollout.get("success") is success, "success does not reproduce from rewards")
    success_step = next((index + 1 for index, reward in enumerate(rewards) if float(reward) > 0.0), None)
    require(rollout.get("success_step") == success_step, "success step does not reproduce")

    rebuilt: list[float] = []
    expected_start = 0
    for chunk in rollout.get("chunks", []):
        require(chunk.get("start_step") == expected_start, "action chunks are not contiguous")
        prediction, shape = decode_array(chunk["prediction"])
        require(shape == [8, 7], "action-chunk shape differs")
        rows = chunk.get("executed_rows")
        require(type(rows) is int and 1 <= rows <= 8, "executed chunk-row count differs")
        for row in range(rows):
            values = [float(x) for x in prediction[row * 7 : (row + 1) * 7]]
            values[-1] = 1.0 if values[-1] > 0.0 else -1.0
            rebuilt.extend(values)
        expected_start += rows
    require(expected_start == steps, "chunks do not cover the trajectory")
    require(rebuilt == [float(x) for x in actions], "executed actions do not reproduce from chunks")
    return success


def exact_one_sided_p(full_only: int, zeroed_only: int) -> float:
    discordant = full_only + zeroed_only
    return sum(math.comb(discordant, k) for k in range(full_only, discordant + 1)) / (2**discordant)


def counts(records: list[dict[str, int | bool]]) -> dict[str, float | int]:
    full = sum(int(row["full"]) for row in records)
    zeroed = sum(int(row["zeroed"]) for row in records)
    full_only = sum(int(bool(row["full"]) and not bool(row["zeroed"])) for row in records)
    zeroed_only = sum(int(bool(row["zeroed"]) and not bool(row["full"])) for row in records)
    n = len(records)
    return {
        "pairs": n,
        "full_successes": full,
        "zeroed_successes": zeroed,
        "full_rate": full / n,
        "zeroed_rate": zeroed / n,
        "gap": (full - zeroed) / n,
        "full_only": full_only,
        "zeroed_only": zeroed_only,
        "one_sided_exact_p": exact_one_sided_p(full_only, zeroed_only),
    }


def close(actual: Any, expected: Any) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-15)


def verify(root: Path) -> dict[str, Any]:
    package = load_json(ARTIFACT_DIR / "manifest.json")
    require(package.get("schema") == "anonymous-instruction-embedding-ablation-artifact-v1", "artifact schema differs")
    files = package.get("files")
    require(isinstance(files, dict) and len(files) == 18, "artifact file manifest differs")
    for relative, digest in files.items():
        require(".." not in Path(relative).parts and not Path(relative).is_absolute(), "unsafe artifact path")
        require(file_sha256(root / relative) == digest, f"file digest differs: {relative}")

    preflight = load_json(root / "athena/results/instruction_embedding_ablation_v1_manifest.json")
    smoke = load_json(root / "athena/results/instruction_embedding_ablation_v1_smoke.json")
    summary = load_json(root / "athena/results/instruction_embedding_ablation_v1_summary.json")
    require(preflight.get("schema") == "xvla-instruction-embedding-ablation-manifest-v1", "preflight schema differs")
    require(preflight.get("mapping_count") == 100 and len(preflight.get("mappings", [])) == 100, "preflight matrix differs")
    mapping_keys = {(row.get("task_index"), row.get("episode")) for row in preflight["mappings"]}
    require(mapping_keys == {(task, episode) for task in range(10) for episode in range(30, 40)}, "preflight state keys differ")
    require(smoke.get("schema") == "xvla-instruction-embedding-ablation-run-v1" and smoke.get("mode") == "strict_smoke", "smoke identity differs")
    require(len(smoke.get("smoke_rows", [])) == 10, "smoke task coverage differs")
    for row in smoke["smoke_rows"]:
        require(set(row.get("conditions", {})) == set(CONDITIONS), "smoke condition set differs")
        for condition, value in row["conditions"].items():
            audit = value.get("ablation_audit", {})
            if condition == CONDITIONS[0]:
                require(audit.get("enabled") is False and audit.get("hook_calls") == 0, "smoke full-path hook audit differs")
            else:
                require(audit.get("enabled") is True and audit.get("hook_calls") == 1, "smoke ablation hook count differs")
                require(audit.get("all_returned_outputs_exactly_zero") is True, "smoke ablation was not exactly zero")

    raw_paths = [root / path for path in files if RAW_RE.fullmatch(Path(path).name)]
    require(len(raw_paths) == 15, "raw shard count differs")
    records: list[dict[str, int | bool]] = []
    observed: set[tuple[int, int, int]] = set()
    protocol = preflight.get("protocol")
    gates_spec = preflight.get("frozen_gates")
    for path in sorted(raw_paths):
        match = RAW_RE.fullmatch(path.name)
        require(match is not None, f"raw filename differs: {path.name}")
        seed, start, end = map(int, match.groups())
        require((start, end) in SHARDS, f"raw shard bounds differ: {path.name}")
        result = load_json(path)
        require(result.get("schema") == "xvla-instruction-embedding-ablation-run-v1" and result.get("mode") == "full", "raw identity differs")
        require(result.get("protocol") == protocol and result.get("frozen_gates") == gates_spec, "raw protocol or gates differ")
        identity = result.get("identity", {})
        require(identity.get("checkpoint_seed") == seed, "raw checkpoint seed differs")
        require(identity.get("source_sha256_start") == identity.get("source_sha256_end"), "source closure changed")
        require(identity.get("model_state_sha256_start") == identity.get("model_state_sha256_end"), "model state changed")
        require(identity.get("protected_component_sha256_start") == identity.get("protected_component_sha256_end"), "protected model component changed")
        evaluation = result.get("evaluation", {})
        require((evaluation.get("task_start"), evaluation.get("task_end")) == (start, end), "raw evaluation bounds differ")
        require(evaluation.get("episode_start") == 30 and evaluation.get("eps_per_task") == 10, "raw episode partition differs")
        rows = result.get("checkpoint_state_pairs", [])
        require(len(rows) == 20, "raw shard row count differs")
        for row in rows:
            task, episode = row.get("task_index"), row.get("episode")
            key = (seed, task, episode)
            require(key not in observed and start <= task < end and 30 <= episode < 40, "duplicate or out-of-range row")
            observed.add(key)
            require(set(row.get("conditions", {})) == set(CONDITIONS), "raw condition set differs")
            starts: list[Any] = []
            contracts: list[Any] = []
            prompts: list[Any] = []
            outcome: dict[str, int | bool] = {"seed": seed, "task": task, "episode": episode}
            for condition, value in row["conditions"].items():
                rollout = value.get("rollout", {})
                audit = value.get("ablation_audit", {})
                starts.append(rollout.get("start_physical"))
                contracts.append(value.get("input_contract"))
                prompts.append((value.get("prompt_id"), value.get("prompt_text"), value.get("instruction_ids")))
                if condition == CONDITIONS[0]:
                    require(audit.get("enabled") is False and audit.get("hook_calls") == 0, "full-path hook audit differs")
                else:
                    require(audit.get("enabled") is True, "ablation hook was not enabled")
                    require(audit.get("hook_calls") == len(rollout.get("chunks", [])), "ablation hook count differs")
                    require(audit.get("all_returned_outputs_exactly_zero") is True, "ablation hook output was not exactly zero")
                    require(all(call.get("returned_nonzero_elements") == 0 and call.get("zeroed_token_positions") == 32 for call in audit.get("calls", [])), "ablation hook call record differs")
                outcome["full" if condition == CONDITIONS[0] else "zeroed"] = validate_rollout(rollout)
            require(starts[0] == starts[1] and contracts[0] == contracts[1] and prompts[0] == prompts[1], "paired inputs differ")
            records.append(outcome)

    expected_keys = {(seed, task, episode) for seed in range(3) for task in range(10) for episode in range(30, 40)}
    require(observed == expected_keys and len(records) == 300, "complete paired matrix is absent")
    pooled = counts(records)
    by_checkpoint = [counts([row for row in records if row["seed"] == seed]) for seed in range(3)]
    by_task = [counts([row for row in records if row["task"] == task]) for task in range(10)]
    expected = package["expected"]
    require(pooled["full_successes"] == expected["full_successes"], "full success total differs")
    require(pooled["zeroed_successes"] == expected["zeroed_successes"], "zeroed success total differs")
    require(pooled["full_only"] == expected["full_only"] and pooled["zeroed_only"] == expected["zeroed_only"], "discordance totals differ")
    require(close(pooled["gap"], expected["paired_gap"]), "paired gap differs")
    require(close(pooled["one_sided_exact_p"], expected["one_sided_exact_p"]), "exact paired p-value differs")
    require([row["full_successes"] for row in by_checkpoint] == expected["checkpoint_full_successes"], "checkpoint full totals differ")
    require([row["zeroed_successes"] for row in by_checkpoint] == expected["checkpoint_zeroed_successes"], "checkpoint zeroed totals differ")
    require(all(close(row["gap"], value) for row, value in zip(by_task, expected["task_gaps"])), "task gaps differ")

    by_state: dict[tuple[int, int], list[dict[str, int | bool]]] = defaultdict(list)
    for row in records:
        by_state[(int(row["task"]), int(row["episode"]))].append(row)
    task_support_minima = []
    for task in range(10):
        state_sums = [sum(int(row["full"]) - int(row["zeroed"]) for row in by_state[(task, episode)]) for episode in range(30, 40)]
        task_support_minima.append(min(state_sums))
    support_lower = sum(10 * value for value in task_support_minima) / 300
    require(close(support_lower, expected["bootstrap_support_lower_bound"]), "bootstrap support lower bound differs")
    require(support_lower > 0.0, "bootstrap support permits a nonpositive draw")

    require(summary.get("schema") == "xvla-instruction-embedding-ablation-summary-v1", "summary schema differs")
    require(summary.get("identity_validated") is True and summary.get("overall_pass") is True and summary.get("claim_eligible") is True, "strict summary did not pass")
    require(all(summary.get("gates", {}).values()) and len(summary.get("gates", {})) == 8, "summary gate set differs")
    require(all(close(a, b) for a, b in zip(summary.get("task_stratified_state_cluster_bootstrap_interval", []), expected["bootstrap_interval"])), "bootstrap interval differs")
    require(close(summary["pooled"]["paired_success_gap"], pooled["gap"]), "summary pooled gap differs")
    require(summary["pooled"]["full_only"] == pooled["full_only"] and summary["pooled"]["zeroed_only"] == pooled["zeroed_only"], "summary discordances differ")

    gates = {
        "full_pooled_at_least_70_percent": float(pooled["full_rate"]) >= 0.70,
        "full_every_checkpoint_at_least_60_percent": all(float(row["full_rate"]) >= 0.60 for row in by_checkpoint),
        "pooled_gap_at_least_20_points": float(pooled["gap"]) >= 0.20,
        "gap_positive_every_checkpoint": all(float(row["gap"]) > 0.0 for row in by_checkpoint),
        "one_sided_exact_p_at_most_0p01": float(pooled["one_sided_exact_p"]) <= 0.01,
        "at_least_eight_positive_tasks": sum(float(row["gap"]) > 0.0 for row in by_task) >= 8,
        "no_negative_task": all(float(row["gap"]) >= 0.0 for row in by_task),
        "bootstrap_lower_above_zero": expected["bootstrap_interval"][0] > 0.0 and support_lower > 0.0,
    }
    require(all(gates.values()), "a recomputed frozen gate failed")
    return {
        "verified": True,
        "pooled": pooled,
        "by_checkpoint": by_checkpoint,
        "by_task": by_task,
        "bootstrap_interval": expected["bootstrap_interval"],
        "bootstrap_support_lower_bound": support_lower,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(args.root.resolve())
    except VerificationError as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        pooled = result["pooled"]
        print("Instruction-embedding-path ablation artifact: VERIFIED")
        print(f"  full path: {pooled['full_successes']}/300 ({100 * pooled['full_rate']:.1f}%)")
        print(f"  zeroed path: {pooled['zeroed_successes']}/300 ({100 * pooled['zeroed_rate']:.1f}%)")
        print(f"  paired gap: {100 * pooled['gap']:.1f} percentage points")
        print(f"  one-sided exact p: {pooled['one_sided_exact_p']:.3e}")
        low, high = result["bootstrap_interval"]
        print(f"  state-cluster bootstrap: [{100 * low:.1f}, {100 * high:.1f}] points")
        print("  frozen gates: 8/8 passed")


if __name__ == "__main__":
    main()
