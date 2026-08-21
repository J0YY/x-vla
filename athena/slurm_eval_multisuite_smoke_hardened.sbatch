#!/usr/bin/env bash
#SBATCH --partition=low-prio-gpu
#SBATCH --qos=normal
#SBATCH --constraint=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 ARCHITECTURE CHECKPOINT METADATA" >&2
  exit 2
fi

architecture=$1
checkpoint=$2
metadata=$3

source /athenahomes/joy/miniconda3/etc/profile.d/conda.sh
conda activate safesae-openvla

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/work/joy/hf_cache
export PYTHONPATH="/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO:${PYTHONPATH:-}"
export XVLA_FROZEN_EVALUATOR_SHA256=91ae342892e32b0aa019a43ff06b5809dca2a49d4be20ce9a55fbad1b15cdbb3
export XVLA_FROZEN_EVALUATION_GPU_FAMILY=a6000

cd /work/joy/x-vla-workshop
observed_sha256=$(sha256sum athena/run_xvla_experiment.py | awk '{print $1}')
if [[ "$observed_sha256" != "$XVLA_FROZEN_EVALUATOR_SHA256" ]]; then
  echo "Hardened evaluator SHA-256 mismatch" >&2
  exit 3
fi
python -u athena/run_xvla_experiment.py \
  --mode smoke \
  --architecture "$architecture" \
  --vision-encoder vit \
  --suite libero_object \
  --training-suite libero_object \
  --checkpoint "$checkpoint" \
  --model-metadata "$metadata" \
  --cache artifacts/libero_frames_100000_64.pkl \
  --output "results/generalist_hardened_${architecture}_eval_smoke.json" \
  --seed 0 \
  --matmul-precision highest \
  --task-start 0 \
  --task-end 1 \
  --eps-per-task 50 \
  --max-steps 280 \
  --profile-iters 2
end_sha256=$(sha256sum athena/run_xvla_experiment.py | awk '{print $1}')
if [[ "$end_sha256" != "$XVLA_FROZEN_EVALUATOR_SHA256" ]]; then
  echo "Hardened evaluator changed during the smoke" >&2
  exit 3
fi
