#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

readonly run_root="$1"
readonly checkpoint="$2"
readonly output="$3"
readonly expected_sha256="$4"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python

cd "$run_root"
export PYTHONPATH="$run_root"
export PYTHONUNBUFFERED=1

echo "run_root=$run_root"
echo "checkpoint=$checkpoint"
echo "output=$output"
echo "python=$($python_executable --version 2>&1)"
echo "source_tree_sha256=$(find xvla -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
sha256sum scripts/run_production_factorized_odt_stress.py
sha256sum "$checkpoint"

"$python_executable" scripts/run_production_factorized_odt_stress.py \
  --checkpoint "$checkpoint" \
  --expected-checkpoint-sha256 "$expected_sha256" \
  --output "$output"

sha256sum "$output"
