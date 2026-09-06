#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

readonly run_root="$1"
readonly seeds="$2"
readonly rank="$3"
readonly output="$4"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python

cd "$run_root"
export PYTHONPATH="$run_root"
export PYTHONUNBUFFERED=1

echo "run_root=$run_root"
echo "seeds=$seeds"
echo "rank=$rank"
echo "output=$output"
echo "python=$($python_executable --version 2>&1)"
sha256sum xvla/train/global_projective_odt_oracle.py
sha256sum scripts/run_global_projective_odt_oracle.py

"$python_executable" -m pytest -q \
  tests/test_global_projective_odt_oracle.py \
  tests/test_projective_quotient.py \
  tests/test_residual_route_odt.py

"$python_executable" scripts/run_global_projective_odt_oracle.py \
  --seeds "$seeds" \
  --rank "$rank" \
  --output "$output"

sha256sum "$output"
