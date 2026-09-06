#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail
readonly run_root="$1"
readonly data_root="$2"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python
cd "$run_root"
export PYTHONPATH="$run_root"
"$python_executable" -u athena/dooms_svhn_reproduction.py prefetch --run-root "$run_root" --data-root "$data_root"
