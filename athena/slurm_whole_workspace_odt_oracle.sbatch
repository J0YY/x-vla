#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

readonly run_root="$1"
readonly seeds="$2"
readonly output="$3"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python

cd "$run_root"
export PYTHONPATH="$run_root"
export PYTHONUNBUFFERED=1

echo "run_root=$run_root"
echo "seeds=$seeds"
echo "output=$output"
echo "python=$($python_executable --version 2>&1)"
echo "source_tree_sha256=$(find xvla -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
sha256sum scripts/run_whole_workspace_odt_oracle.py

"$python_executable" -m pytest -q tests/test_residual_attention_odt_oracle.py
"$python_executable" scripts/run_whole_workspace_odt_oracle.py \
  --seeds "$seeds" \
  --output "$output"

sha256sum "$output"
