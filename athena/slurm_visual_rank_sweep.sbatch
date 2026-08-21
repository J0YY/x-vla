#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 CHECKPOINT SEED VISION_ENCODER OUTPUT_PREFIX" >&2
  exit 2
fi

checkpoint=$1
seed=$2
vision_encoder=$3
output_prefix=$4

source /athenahomes/joy/miniconda3/etc/profile.d/conda.sh
conda activate safesae-openvla

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/work/joy/hf_cache
export PYTHONPATH="/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO:${PYTHONPATH:-}"

cd /work/joy/x-vla-workshop

for rank in 64 96 128 192; do
  python -u athena/run_xvla_experiment.py \
    --mode visual_subspace \
    --architecture chi \
    --vision-encoder "$vision_encoder" \
    --checkpoint "$checkpoint" \
    --cache artifacts/libero_frames_100000_64.pkl \
    --output "results/${output_prefix}_r${rank}.json" \
    --seed "$seed" \
    --rank "$rank" \
    --gram-action-group all_balanced \
    --gram-probes 4 \
    --gram-samples 1024 \
    --offline-eval-samples 1024 \
    --gram-task-start 0 \
    --gram-task-end 4 \
    --random-controls 3 \
    --task-start 4 \
    --task-end 8 \
    --eps-per-task 10 \
    --max-steps 400 \
    --profile-iters 20
done
