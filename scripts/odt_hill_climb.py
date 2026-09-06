#!/usr/bin/env python3
"""Validate and report the χ-ODT closed-loop metric used for iteration.

Examples:
    python scripts/odt_hill_climb.py score result.json
    python scripts/odt_hill_climb.py compare baseline.json candidate.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "xvla_modal_odt_fixed20_closed_loop_v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: result must be a JSON object")
    return value


def _validated_score(path: Path) -> dict[str, Any]:
    result = _load(path)
    if result.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"{path}: unexpected schema {result.get('schema')!r}")
    if result.get("completed") is not True:
        raise ValueError(f"{path}: evaluation is not complete")
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{path}: episodes must be a nonempty list")
    successes = sum(row.get("success") is True for row in episodes if isinstance(row, dict))
    if successes != result.get("successes") or len(episodes) != result.get("episode_count"):
        raise ValueError(f"{path}: aggregate does not match episode rows")
    rate = successes / len(episodes)
    if abs(rate - float(result.get("success_rate", -1.0))) > 1e-12:
        raise ValueError(f"{path}: reported success rate is inconsistent")
    guards = result.get("numerical_guards")
    if not isinstance(guards, dict):
        raise ValueError(f"{path}: missing numerical guard receipt")
    for key in ("prohibited_calls", "matrix_spectral_norm_calls"):
        value = guards.get(key)
        if value not in (0, [], {}, None):
            raise ValueError(f"{path}: direct-only gate failed for {key}: {value!r}")
    for key in ("checkpoint_sha256", "source_bundle_sha256", "protocol_sha256"):
        value = result.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{path}: missing valid {key}")
    return {
        "path": str(path),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "requested_removal_percent": result["requested_removal_percent"],
        "successes": successes,
        "episode_count": len(episodes),
        "success_rate": rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score_parser = subparsers.add_parser("score", help="validate one result and print its metric")
    score_parser.add_argument("result", type=Path)
    compare_parser = subparsers.add_parser("compare", help="compare a candidate with its paired baseline")
    compare_parser.add_argument("baseline", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    arguments = parser.parse_args()

    if arguments.command == "score":
        payload = {"metric": "closed_loop_success_rate", **_validated_score(arguments.result)}
    else:
        baseline = _validated_score(arguments.baseline)
        candidate = _validated_score(arguments.candidate)
        if baseline["checkpoint_sha256"] != candidate["checkpoint_sha256"]:
            raise ValueError("candidate and baseline use different checkpoints")
        payload = {
            "metric": "closed_loop_success_rate",
            "baseline": baseline,
            "candidate": candidate,
            "delta": candidate["success_rate"] - baseline["success_rate"],
        }
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
