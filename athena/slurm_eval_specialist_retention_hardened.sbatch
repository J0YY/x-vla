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

source /athenahomes/joy/miniconda3/etc/profile.d/conda.sh
conda activate safesae-openvla

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO:${PYTHONPATH:-}"
export HF_HOME=/work/joy/hf_cache
export HF_HUB_CACHE=/work/joy/hf_cache/hub
export HF_DATASETS_CACHE=/work/joy/hf_cache/datasets
export TRANSFORMERS_CACHE=/work/joy/hf_cache/transformers
export XDG_CACHE_HOME=/work/joy/cache
export TORCH_HOME=/work/joy/torch_cache
export MPLCONFIGDIR=/work/joy/cache/matplotlib
export XVLA_FROZEN_SPECIALIST_EVALUATOR_SHA256=fd28d56acb9e198c1dbcd207c8614d32a81d9ac50f3de3016fe9a2817bd19939
export XVLA_FROZEN_SPECIALIST_GPU_FAMILY=a6000

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" \
  "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"
cd /work/joy/x-vla-workshop
observed_sha256=$(sha256sum athena/eval_specialist_retention_hardened.py | awk '{print $1}')
if [[ "$observed_sha256" != "$XVLA_FROZEN_SPECIALIST_EVALUATOR_SHA256" ]]; then
  echo "Hardened specialist evaluator SHA-256 mismatch" >&2
  exit 3
fi
python -u athena/eval_specialist_retention_hardened.py "$@"
end_sha256=$(sha256sum athena/eval_specialist_retention_hardened.py | awk '{print $1}')
if [[ "$end_sha256" != "$XVLA_FROZEN_SPECIALIST_EVALUATOR_SHA256" ]]; then
  echo "Hardened specialist evaluator changed during the job" >&2
  exit 3
fi
