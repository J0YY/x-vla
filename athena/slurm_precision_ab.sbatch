#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail

source /athenahomes/joy/miniconda3/etc/profile.d/conda.sh
conda activate safesae-openvla

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/work/joy/hf_cache
export PYTHONPATH="/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO:${PYTHONPATH:-}"

cd /work/joy/x-vla-workshop

run_condition() {
  local checkpoint=$1
  local seed=$2
  local task=$3
  local precision=$4
  local label=$5
  python -u athena/run_xvla_experiment.py \
    --mode capability \
    --architecture chi \
    --checkpoint "$checkpoint" \
    --cache artifacts/libero_frames_100000_64.pkl \
    --output "results/precision_ab_${label}_${precision}.json" \
    --seed "$seed" \
    --matmul-precision "$precision" \
    --task-start "$task" \
    --task-end "$((task + 1))" \
    --eps-per-task 20 \
    --max-steps 280 \
    --profile-iters 20
}

# These are the two largest preserved historical-to-Athena regressions. Running
# both precision settings sequentially in one allocation controls GPU model,
# driver, simulator, cache, checkpoint, and canonical initial states.
for precision in high highest; do
  run_condition artifacts/ckpt_linear_rat_vit_s1.pt 1 5 "$precision" vit_s1_task5
  run_condition artifacts/ckpt_linear_rat_vit_s0_v2.pt 0 3 "$precision" vit_s0_task3
done
