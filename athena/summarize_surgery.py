#!/usr/bin/env python3
"""Rank the preregistered attention-surgery discovery sweep without hiding failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=float, default=0.15)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.results_dir.glob("surgery_discovery_b*_k*.json")):
        result = json.loads(path.read_text())["surgery"]
        comparisons = result["comparisons"]
        keep_advantage = float(comparisons["keep_top_minus_random_normmatched"])
        removal_advantage = float(comparisons["remove_random_normmatched_minus_top"])
        row = {
            "file": str(path),
            "block_index": result["block_index"],
            "rank": result["rank"],
            "overall_by_condition": result["overall_by_condition"],
            "keep_advantage": keep_advantage,
            "removal_advantage": removal_advantage,
            "minimum_preregistered_advantage": min(keep_advantage, removal_advantage),
            "passes_both_15_point_gates": (
                keep_advantage >= args.gate and removal_advantage >= args.gate
            ),
        }
        rows.append(row)
    rows.sort(key=lambda row: row["minimum_preregistered_advantage"], reverse=True)
    eligible = [row for row in rows if row["passes_both_15_point_gates"]]
    result = {
        "gate": args.gate,
        "configurations_evaluated": len(rows),
        "eligible_configurations": len(eligible),
        "best_by_preregistered_minimum_advantage": rows[0] if rows else None,
        "confirmation_candidate": eligible[0] if eligible else None,
        "rows": rows,
        "selection_rule": (
            "A configuration advances only when both keep-top beats Frobenius-matched random "
            "and remove-top is more damaging than Frobenius-matched random by at least the "
            "prespecified gate. The selected configuration must then be confirmed on held-out "
            "tasks before any positive coefficient-surgery claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
