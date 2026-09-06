#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

readonly run_root="$1"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python

cd "$run_root"
export PYTHONPATH="$run_root"
export CUDA_VISIBLE_DEVICES=""

"$python_executable" -u athena/validate_svhn_canonical_odt_smoke.py \
  --root "$run_root" \
  --output "$run_root/results/smoke_validation.json"
