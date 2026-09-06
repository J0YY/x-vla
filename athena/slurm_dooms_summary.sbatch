#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --constraint=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
readonly run_root="$1"
readonly data_root="$2"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python
cd "$run_root"
export PYTHONPATH="$run_root"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$python_executable" -u athena/dooms_svhn_reproduction.py summary --run-root "$run_root" --data-root "$data_root"
