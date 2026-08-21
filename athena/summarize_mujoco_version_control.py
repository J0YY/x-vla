#!/usr/bin/env python3
"""Validate the paired MuJoCo version-isolation control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED = {
    ("base350", "vit_s1_task5"): {
        "mujoco": "3.5.0",
        "seed": 1,
        "task": 5,
        "checkpoint": "artifacts/ckpt_linear_rat_vit_s1.pt",
    },
    ("base350", "vit_s0_task3"): {
        "mujoco": "3.5.0",
        "seed": 0,
        "task": 3,
        "checkpoint": "artifacts/ckpt_linear_rat_vit_s0_v2.pt",
    },
    ("clone316", "vit_s1_task5"): {
        "mujoco": "3.1.6",
        "seed": 1,
        "task": 5,
        "checkpoint": "artifacts/ckpt_linear_rat_vit_s1.pt",
    },
    ("clone316", "vit_s0_task3"): {
        "mujoco": "3.1.6",
        "seed": 0,
        "task": 3,
        "checkpoint": "artifacts/ckpt_linear_rat_vit_s0_v2.pt",
    },
}
MATERIAL_RECOVERY_MIN_SUCCESS_GAIN = 8
EXPECTED_CACHE_SHA256 = "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662"
EXPECTED_CHECKPOINT_SHA256 = {
    "artifacts/ckpt_linear_rat_vit_s1.pt": "cc0780b989165a80a449c03cbb3574e2f56b2e5747a64cf32d1fde418df8ec3c",
    "artifacts/ckpt_linear_rat_vit_s0_v2.pt": "96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", type=Path, required=True)
    parser.add_argument("--historical", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def result_key(path: Path) -> tuple[str, str]:
    for version in ("base350", "clone316"):
        prefix = f"{version}_"
        if path.stem.startswith(prefix):
            return version, path.stem[len(prefix) :]
    raise RuntimeError(f"Unexpected result name: {path}")


def validate(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    key = result_key(path)
    expected = EXPECTED[key]
    capability = result["capability"]
    forensics = result["version_isolation_forensics"]
    environment = forensics["environment"]
    frozen_inputs = forensics["frozen_inputs"]
    checks = {
        "mode": result["mode"] == "capability",
        "architecture": result["architecture"] == "chi",
        "vision_encoder": result["vision_encoder"] == "vit",
        "checkpoint": result["checkpoint"] == expected["checkpoint"],
        "seed": result["seed"] == expected["seed"],
        "precision": result["matmul_precision"] == "highest",
        "tasks": capability["task_indices"] == [expected["task"]],
        "trials": capability["trials"] == 20,
        "episodes": [row["episode"] for row in capability["episodes"]]
        == list(range(20)),
        "canonical": capability["canonical_init_states"] is True,
        "finite": result["prediction_finite"] is True,
        "mujoco": environment["critical_packages"]["mujoco"]
        == expected["mujoco"],
        "forensic_count": len(result["version_isolation_forensics"]["episodes"])
        == 20,
        "cache_sha256": frozen_inputs["cache_sha256"] == EXPECTED_CACHE_SHA256,
        "checkpoint_sha256": frozen_inputs["checkpoint_sha256"]
        == EXPECTED_CHECKPOINT_SHA256[expected["checkpoint"]],
        "records_actions": forensics["records_all_executed_policy_actions"] is True,
        "records_chunks": forensics["records_all_raw_action_chunks"] is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Identity failure in {path}: {failed}")
    for episode in forensics["episodes"]:
        for action in episode["policy_actions"]:
            values = action["values"]
            if len(values) != 7 or values[-1] not in (-1.0, 1.0):
                raise RuntimeError(f"Invalid executed action in {path}")
            if not all(math.isfinite(float(value)) for value in values):
                raise RuntimeError(f"Non-finite executed action in {path}")
        for query in episode["policy_queries"]:
            chunk = query["raw_action_chunk"]
            if len(chunk) != 8 or any(len(action) != 7 for action in chunk):
                raise RuntimeError(f"Invalid raw action chunk in {path}")
            if not all(
                math.isfinite(float(value))
                for action in chunk
                for value in action
            ):
                raise RuntimeError(f"Non-finite raw action chunk in {path}")

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "successes": capability["successes"],
        "trials": capability["trials"],
        "environment_identity_sha256": environment["environment_identity_sha256"],
        "pip_freeze_sha256": environment["pip_freeze_sha256"],
        "pip_freeze_without_mujoco_sha256": environment[
            "pip_freeze_without_mujoco_sha256"
        ],
        "mujoco_install_record_sha256": environment[
            "mujoco_install_record_sha256"
        ],
        "source_sha256": frozen_inputs["source_sha256"],
        "torch_flags": environment["torch_flags"],
    }


def paired_forensics(
    base: dict[str, Any], clone: dict[str, Any]
) -> dict[str, Any]:
    base_rows = base["version_isolation_forensics"]["episodes"]
    clone_rows = clone["version_isolation_forensics"]["episodes"]
    if [(row["task_index"], row["episode"]) for row in base_rows] != [
        (row["task_index"], row["episode"]) for row in clone_rows
    ]:
        raise RuntimeError("Paired episode identities differ across MuJoCo versions")

    fields = (
        "canonical_init_state_sha256",
        "post_set_sim_state_sha256",
        "post_wait_sim_state_sha256",
        "initial_raw_image_sha256",
        "initial_model_image_sha256",
        "initial_robot_state_sha256",
    )
    agreements = {
        field: sum(a[field] == b[field] for a, b in zip(base_rows, clone_rows, strict=True))
        for field in fields
    }
    exact_action_episodes = sum(
        a["policy_actions"] == b["policy_actions"]
        for a, b in zip(base_rows, clone_rows, strict=True)
    )
    first_action_equal = sum(
        bool(a["policy_actions"])
        and bool(b["policy_actions"])
        and a["policy_actions"][0] == b["policy_actions"][0]
        for a, b in zip(base_rows, clone_rows, strict=True)
    )
    identical_input_chunk_mismatches = []
    for base_row, clone_row in zip(base_rows, clone_rows, strict=True):
        base_queries = {
            (
                query["model_image_sha256"],
                query["robot_state_sha256"],
                query["instruction_token_sha256"],
            ): query["raw_action_chunk_sha256"]
            for query in base_row["policy_queries"]
        }
        for query in clone_row["policy_queries"]:
            key = (
                query["model_image_sha256"],
                query["robot_state_sha256"],
                query["instruction_token_sha256"],
            )
            if key in base_queries and base_queries[key] != query["raw_action_chunk_sha256"]:
                identical_input_chunk_mismatches.append(
                    {"task_index": base_row["task_index"], "episode": base_row["episode"]}
                )
    if identical_input_chunk_mismatches:
        raise RuntimeError(
            "Identical policy inputs produced different action chunks across versions"
        )
    return {
        "episode_count": len(base_rows),
        "hash_agreements": agreements,
        "exact_full_action_trace_agreements": exact_action_episodes,
        "exact_first_action_agreements": first_action_equal,
        "identical_input_action_chunk_mismatches": identical_input_chunk_mismatches,
    }


def main() -> None:
    args = parse_args()
    if len(args.result) != 4 or len(args.historical) != 2:
        raise ValueError("Expected four forensic results and two historical controls")
    results = {result_key(path): load(path) for path in args.result}
    if set(results) != set(EXPECTED):
        raise RuntimeError("Result set does not match the frozen four-condition design")
    validated = {
        "/".join(key): validate(path, results[key])
        for key, path in ((result_key(path), path) for path in args.result)
    }

    freeze_without = {
        results[key]["version_isolation_forensics"]["environment"][
            "pip_freeze_without_mujoco_sha256"
        ]
        for key in results
    }
    if len(freeze_without) != 1:
        raise RuntimeError("Package freezes differ beyond the MuJoCo distribution")

    source_manifests = {
        json.dumps(
            result["version_isolation_forensics"]["frozen_inputs"]["source_sha256"],
            sort_keys=True,
        )
        for result in results.values()
    }
    if len(source_manifests) != 1:
        raise RuntimeError("Source hashes differ across version-isolation arms")

    torch_flags = {
        json.dumps(
            result["version_isolation_forensics"]["environment"]["torch_flags"],
            sort_keys=True,
        )
        for result in results.values()
    }
    if len(torch_flags) != 1:
        raise RuntimeError("Torch, GPU, CUDA, or numerical flags differ across arms")

    python_hashes = {
        result["version_isolation_forensics"]["environment"][
            "python_executable_sha256"
        ]
        for result in results.values()
    }
    if len(python_hashes) != 1:
        raise RuntimeError("Python executable bytes differ across cloned environments")

    slices: dict[str, Any] = {}
    base_total = 0
    clone_total = 0
    no_slice_decrease = True
    for task_label in ("vit_s1_task5", "vit_s0_task3"):
        base = results[("base350", task_label)]
        clone = results[("clone316", task_label)]
        base_successes = base["capability"]["successes"]
        clone_successes = clone["capability"]["successes"]
        base_total += base_successes
        clone_total += clone_successes
        no_slice_decrease = no_slice_decrease and clone_successes >= base_successes
        slices[task_label] = {
            "base350_successes": base_successes,
            "clone316_successes": clone_successes,
            "success_gain": clone_successes - base_successes,
            "paired_forensics": paired_forensics(base, clone),
        }
        if (
            slices[task_label]["paired_forensics"]["hash_agreements"][
                "canonical_init_state_sha256"
            ]
            != 20
        ):
            raise RuntimeError("Canonical initial-state inputs differ across versions")

    historical = [load(path) for path in args.historical]
    historical_successes = sorted(
        (item["seed"], item["capability"]["task_indices"][0], item["capability"]["successes"])
        for item in historical
    )
    expected_historical = [(0, 3, 9), (1, 5, 6)]
    if historical_successes != expected_historical:
        raise RuntimeError(
            f"Historical 830843 identities or outcomes changed: {historical_successes}"
        )

    gain = clone_total - base_total
    material_recovery = gain >= MATERIAL_RECOVERY_MIN_SUCCESS_GAIN and no_slice_decrease
    summary = {
        "schema_version": "xvla-mujoco-version-control-summary-v1",
        "validated_results": validated,
        "only_package_change_is_mujoco": True,
        "historical_830843_successes": {"vit_s1_task5": 6, "vit_s0_task3": 9},
        "slices": slices,
        "aggregate": {
            "base350_successes": base_total,
            "clone316_successes": clone_total,
            "trials_per_version": 40,
            "success_gain": gain,
            "success_rate_gain": gain / 40,
        },
        "material_recovery_gate": {
            "minimum_success_gain": MATERIAL_RECOVERY_MIN_SUCCESS_GAIN,
            "minimum_success_rate_gain": MATERIAL_RECOVERY_MIN_SUCCESS_GAIN / 40,
            "no_slice_decrease": True,
            "pass": material_recovery,
        },
        "full_locked_replay_recommendation": (
            "queue_mujoco_3_1_6_full_locked_replay"
            if material_recovery
            else "do_not_queue_from_this_control"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
