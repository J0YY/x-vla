#!/bin/bash
#SBATCH --job-name=implicit-all-token
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/implicit-all-token-%j.out
#SBATCH --error=logs/implicit-all-token-%j.err

set -euo pipefail

LANE="${LANE:-tiny}"
STAGE="${STAGE:-replay}"
OUTPUT="${OUTPUT:?set OUTPUT to a new result path}"
CHECKPOINT="${CHECKPOINT:-/work/joy/x-vla-workshop/artifacts/ckpt_linear_rat_vit_s0_v2.pt}"
DEVICE="${DEVICE:-cuda}"
SAMPLES="${SAMPLES:-1}"

mkdir -p logs "$(dirname "$OUTPUT")"
if [[ "$LANE" == "tiny" ]]; then
  python scripts/run_implicit_sparse_projective_odt_all_tokens.py \
    --lane tiny \
    --stage "$STAGE" \
    --device cpu \
    --block-size 32 \
    --output "$OUTPUT"
else
  python scripts/run_implicit_sparse_projective_odt_all_tokens.py \
    --lane "$LANE" \
    --stage "$STAGE" \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE" \
    --samples "$SAMPLES" \
    --output "$OUTPUT"
fi
