#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

readonly run_root="$1"
readonly output_dir="$2"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python
readonly summary="$output_dir/typed_projective_dag_oracle_v1_r2_replication_summary.json"

cd "$run_root"
export PYTHONPATH="$run_root"
export PYTHONUNBUFFERED=1

echo "run_root=$run_root"
echo "output_dir=$output_dir"
echo "python=$($python_executable --version 2>&1)"
sha256sum xvla/train/typed_projective_dag_oracle.py
sha256sum tests/test_typed_projective_dag_oracle.py
sha256sum scripts/run_typed_projective_dag_oracle.py

"$python_executable" -m pytest -q \
  tests/test_typed_projective_dag_oracle.py \
  tests/test_global_projective_odt_oracle.py \
  tests/test_projective_quotient.py \
  tests/test_residual_route_odt.py

for rank in 1 2 3; do
  for seed in 1 2 3 5; do
    output="$output_dir/typed_projective_dag_oracle_v1_r2_s${seed}_rank${rank}.json"
    test ! -e "$output"
    "$python_executable" scripts/run_typed_projective_dag_oracle.py \
      --seed "$seed" \
      --rank "$rank" \
      --output "$output"
    sha256sum "$output"
  done
done

test ! -e "$summary"
"$python_executable" - "$output_dir" "$summary" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
expected = [(seed, rank) for rank in (1, 2, 3) for seed in (1, 2, 3, 5)]
rows = []
for seed, rank in expected:
    path = output_dir / f"typed_projective_dag_oracle_v1_r2_s{seed}_rank{rank}.json"
    payload = json.loads(path.read_text())
    rows.append(
        {
            "seed": seed,
            "rank": rank,
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "schema": payload.get("schema"),
            "gate_count": len(payload.get("gates", {})),
            "all_gates_pass": payload.get("all_gates_pass") is True,
        }
    )

summary = {
    "schema": "xvla-typed-projective-dag-replication-summary-v1",
    "expected_runs": len(expected),
    "completed_runs": len(rows),
    "all_gates_pass": all(row["all_gates_pass"] for row in rows),
    "rows": rows,
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
if len(rows) != len(expected) or not summary["all_gates_pass"]:
    raise SystemExit(1)
PY

sha256sum "$summary"
