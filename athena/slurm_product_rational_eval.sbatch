#!/usr/bin/env bash
#SBATCH --job-name=product-rational-eval
#SBATCH --partition=low-prio-gpu
#SBATCH --account=vgg
#SBATCH --qos=normal
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/joy/x-vla-product-pade-rational-v2/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-product-pade-rational-v2/logs/%x_%j.err

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 MODE SEED TASK_START TASK_END" >&2
  exit 2
fi
readonly mode=$1
readonly seed=$2
readonly task_start=$3
readonly task_end=$4
case "$mode" in
  smoke|full) ;;
  *) echo "mode must be smoke or full" >&2; exit 2 ;;
esac
case "$seed" in
  0|1|2) ;;
  *) echo "seed must be 0, 1, or 2" >&2; exit 2 ;;
esac
if [[ "$mode" == smoke ]]; then
  if [[ "$seed" != 0 || "$task_start" != 0 || "$task_end" != 1 ]]; then
    echo "smoke evaluation is frozen to seed 0 and task range [0,1)" >&2
    exit 2
  fi
else
  case "$task_start:$task_end" in
    0:3|3:6|6:8|8:10) ;;
    *) echo "full evaluation task range is not a frozen shard" >&2; exit 2 ;;
  esac
fi

readonly project_root=/work/joy/x-vla-product-pade-rational-v2
readonly run_root="$project_root/athena/results/product_pade_rational_v2"
readonly manifest="$run_root/manifest/product_pade_rational_source_manifest.json"
readonly python_executable=/users/joy/miniconda3/envs/safesae-openvla/bin/python

cd "$project_root"
export PYTHONPATH="$project_root"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export HF_HOME=/work/joy/hf_cache
export HF_HUB_CACHE=/work/joy/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

"$python_executable" -B -u athena/eval_product_rational_checkpoint.py \
  --run-root "$run_root" \
  --source-manifest "$manifest" \
  --mode "$mode" \
  --seed "$seed" \
  --task-start "$task_start" \
  --task-end "$task_end"
