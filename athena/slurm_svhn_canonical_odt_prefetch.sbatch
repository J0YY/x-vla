#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

readonly run_root="$1"
readonly data_root="$2"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python

cd "$run_root"
export PYTHONPATH="$run_root"

"$python_executable" -u athena/run_svhn_canonical_odt.py \
  --download-only \
  --data-root "$data_root"
