#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --constraint=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

readonly run_root="$1"
readonly data_root="$2"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python

cd "$run_root"
export PYTHONPATH="$run_root"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$python_executable" -u athena/run_svhn_canonical_odt.py \
  --data-root "$data_root" \
  --seed 9173 \
  --epochs 20 \
  --rank-grid 1,32 \
  --haar-draws 1 \
  --gauge-trials 50 \
  --smoke-nontrivial-diagnostic-rank 1 \
  --minimum-accuracy 0.70 \
  --output "$run_root/results/smoke.json" \
  --checkpoint-output "$run_root/results/smoke.pt" \
  --gauge-matrix-output "$run_root/results/smoke_gauge_matrices.npz"
