#!/usr/bin/env bash
#SBATCH --job-name=product-rational-preflight
#SBATCH --partition=compute
#SBATCH --account=vgg
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/joy/x-vla-product-pade-rational-v2/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-product-pade-rational-v2/logs/%x_%j.err

set -euo pipefail

readonly project_root=/work/joy/x-vla-product-pade-rational-v2
readonly run_root="$project_root/athena/results/product_pade_rational_v2"
readonly manifest="$run_root/manifest/product_pade_rational_source_manifest.json"
readonly output="$run_root/gates/product_pade_rational_preflight.json"
readonly python_executable=/users/joy/miniconda3/envs/safesae-openvla/bin/python

cd "$project_root"
export PYTHONPATH="$project_root"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONHASHSEED=0
export HF_HOME=/work/joy/hf_cache
export HF_HUB_CACHE=/work/joy/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$python_executable" -B -u athena/train_product_rational_checkpoint.py \
  --run-root "$run_root" \
  --source-manifest "$manifest" \
  --mode smoke \
  --preflight-only \
  --preflight-output "$output"
