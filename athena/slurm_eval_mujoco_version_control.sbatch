#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail

cd /work/joy/x-vla-workshop

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/work/joy/hf_cache
export PYTHONPATH="/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO:${PYTHONPATH:-}"

run_slice() {
  local env_prefix=$1
  local version_label=$2
  local checkpoint=$3
  local seed=$4
  local task=$5
  local task_label=$6
  export CONDA_PREFIX="$env_prefix"
  "$env_prefix/bin/python" -u athena/run_mujoco_version_forensics.py \
    --mode capability \
    --architecture chi \
    --checkpoint "$checkpoint" \
    --cache artifacts/libero_frames_100000_64.pkl \
    --output "results/mujoco_version_control/${version_label}_${task_label}.json" \
    --seed "$seed" \
    --matmul-precision highest \
    --task-start "$task" \
    --task-end "$((task + 1))" \
    --eps-per-task 20 \
    --max-steps 280 \
    --profile-iters 20
}

for version in base350 clone316; do
  if [[ "$version" == base350 ]]; then
    env_prefix=/work/joy/safesae-openvla
  else
    env_prefix=/work/joy/safesae-openvla-mujoco-3.1.6
  fi
  run_slice "$env_prefix" "$version" artifacts/ckpt_linear_rat_vit_s1.pt 1 5 vit_s1_task5
  run_slice "$env_prefix" "$version" artifacts/ckpt_linear_rat_vit_s0_v2.pt 0 3 vit_s0_task3
done
