#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

readonly run_root="$1"
readonly checkpoint="$2"
readonly stack="$3"
readonly output="$4"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python

cd "$run_root"
export PYTHONPATH="$run_root"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "run_root=$run_root"
echo "checkpoint=$checkpoint"
echo "stack=$stack"
echo "output=$output"
echo "python=$($python_executable --version 2>&1)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
sha256sum scripts/run_production_ffn_direct_rq_feasibility.py
sha256sum "$checkpoint"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"$python_executable" scripts/run_production_ffn_direct_rq_feasibility.py \
  --checkpoint "$checkpoint" \
  --stack "$stack" \
  --block-index 0 \
  --batch-size 8 \
  --seed 0 \
  --device cuda \
  --output "$output"

sha256sum "$output"
