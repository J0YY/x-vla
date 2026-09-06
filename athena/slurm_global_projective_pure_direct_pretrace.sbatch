#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

readonly run_root="$1"
readonly output_dir="$2"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python

cd "$run_root"
export PYTHONPATH="$run_root"
export PYTHONUNBUFFERED=1

echo "run_root=$run_root"
echo "output_dir=$output_dir"
echo "python=$($python_executable --version 2>&1)"
sha256sum xvla/train/global_projective_odt_oracle.py
sha256sum scripts/run_global_projective_odt_oracle.py
sha256sum tests/test_global_projective_odt_oracle.py

"$python_executable" -m pytest -q \
  tests/test_global_projective_odt_oracle.py \
  tests/test_projective_quotient.py \
  tests/test_residual_route_odt.py

rank2="$output_dir/global_projective_odt_oracle_pure_direct_pretrace_rank2_seeds8.json"
rank1="$output_dir/global_projective_odt_oracle_pure_direct_pretrace_rank1_seeds4.json"
rank3="$output_dir/global_projective_odt_oracle_pure_direct_pretrace_rank3_seeds4.json"
summary="$output_dir/global_projective_odt_oracle_pure_direct_pretrace_summary.json"
test ! -e "$rank2"
test ! -e "$rank1"
test ! -e "$rank3"
test ! -e "$summary"

"$python_executable" scripts/run_global_projective_odt_oracle.py \
  --seeds 8 --rank 2 --output "$rank2"
sha256sum "$rank2"
"$python_executable" scripts/run_global_projective_odt_oracle.py \
  --seeds 4 --rank 1 --output "$rank1"
sha256sum "$rank1"
"$python_executable" scripts/run_global_projective_odt_oracle.py \
  --seeds 4 --rank 3 --output "$rank3"
sha256sum "$rank3"

"$python_executable" - "$summary" "$rank2" "$rank1" "$rank3" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
rows = []
for name in sys.argv[2:]:
    path = Path(name)
    payload = json.loads(path.read_text())
    rows.append(
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rank": payload["object"]["ffn_rank"],
            "seeds": payload["seeds"],
            "all_gates_pass": payload.get("all_gates_pass") is True,
            "all_nodes_direct_reduced_rq": payload.get("gates", {}).get(
                "all_nodes_direct_reduced_rq"
            ) is True,
        }
    )
summary = {
    "schema": "xvla-global-projective-pure-direct-rq-pretrace-summary-v1",
    "claim_boundary": (
        "Pure direct reduced-RQ clone-oracle pretrace baseline. This predates the "
        "decisive per-step shared-DAG versus clone trace."
    ),
    "all_gates_pass": all(row["all_gates_pass"] for row in rows),
    "all_nodes_direct_reduced_rq": all(
        row["all_nodes_direct_reduced_rq"] for row in rows
    ),
    "rows": rows,
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
if not summary["all_gates_pass"] or not summary["all_nodes_direct_reduced_rq"]:
    raise SystemExit(1)
PY

sha256sum "$summary"
