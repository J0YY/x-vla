#!/usr/bin/env bash
#SBATCH --job-name=product-rational-aggregate
#SBATCH --partition=compute
#SBATCH --account=vgg
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/joy/x-vla-product-pade-rational-v2/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-product-pade-rational-v2/logs/%x_%j.err

set -euo pipefail

readonly project_root=/work/joy/x-vla-product-pade-rational-v2
readonly run_root="$project_root/athena/results/product_pade_rational_v2"
readonly manifest="$run_root/manifest/product_pade_rational_source_manifest.json"
readonly python_executable=/users/joy/miniconda3/envs/safesae-openvla/bin/python

cd "$project_root"
export PYTHONPATH="$project_root"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0

"$python_executable" -B -u athena/aggregate_product_rational_results.py \
  --run-root "$run_root" \
  --source-manifest "$manifest"
