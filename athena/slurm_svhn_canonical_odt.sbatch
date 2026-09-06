#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --constraint=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

readonly run_root="$1"
readonly data_root="$2"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python
readonly seed="${SLURM_ARRAY_TASK_ID:?array task id is required}"

cd "$run_root"
export PYTHONPATH="$run_root"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$python_executable" -u athena/run_svhn_canonical_odt.py \
  --data-root "$data_root" \
  --seed "$seed" \
  --epochs 20 \
  --rank-grid 1,2,4,6,8,12,16,24,32 \
  --haar-draws 16 \
  --gauge-trials 50 \
  --minimum-accuracy 0.70 \
  --output "$run_root/results/seed_${seed}.json" \
  --checkpoint-output "$run_root/results/seed_${seed}.pt" \
  --gauge-matrix-output "$run_root/results/seed_${seed}_gauge_matrices.npz"
