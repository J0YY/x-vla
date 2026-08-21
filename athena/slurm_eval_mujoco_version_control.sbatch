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
export PYTHONPATH="/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO"
unset LD_LIBRARY_PATH

run_slice() {
  local version_label=$1
  local overlay_path=$2
  local checkpoint=$3
  local seed=$4
  local task=$5
  local task_label=$6
  export CONDA_PREFIX=/work/joy/safesae-openvla
  if [[ -n "$overlay_path" ]]; then
    export XVLA_MUJOCO_OVERLAY="$overlay_path"
    export PYTHONPATH="$overlay_path:/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO"
  else
    unset XVLA_MUJOCO_OVERLAY
    export PYTHONPATH="/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO"
  fi
  /work/joy/safesae-openvla/bin/python -u athena/run_mujoco_version_forensics.py \
    --mode capability \
    --architecture chi \
    --checkpoint "$checkpoint" \
    --cache artifacts/libero_frames_100000_64.pkl \
    --output "results/mujoco_version_overlay_control/${version_label}_${task_label}.json" \
    --seed "$seed" \
    --matmul-precision highest \
    --task-start "$task" \
    --task-end "$((task + 1))" \
    --eps-per-task 20 \
    --max-steps 280 \
    --profile-iters 20
}

# Fixed-order paired diagnostic. This does not include an AB/BA order-drift arm, so
# the summary limits attribution accordingly even when the package-recovery gate passes.
for version in base350 overlay316; do
  if [[ "$version" == base350 ]]; then
    overlay_path=
  else
    overlay_path=/work/joy/xvla-mujoco-3.1.6-overlay
  fi
  run_slice "$version" "$overlay_path" artifacts/ckpt_linear_rat_vit_s1.pt 1 5 vit_s1_task5
  run_slice "$version" "$overlay_path" artifacts/ckpt_linear_rat_vit_s0_v2.pt 0 3 vit_s0_task3
done
