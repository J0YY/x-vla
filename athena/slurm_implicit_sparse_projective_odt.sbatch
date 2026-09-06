#!/bin/bash
#SBATCH --job-name=implicit-direct-rq
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/implicit-direct-rq-%j.out
#SBATCH --error=logs/implicit-direct-rq-%j.err

set -euo pipefail

LANE="${LANE:-tiny}"
STAGE="${STAGE:-algorithm1}"
OUTPUT="${OUTPUT:?set OUTPUT to a new result path}"
CHECKPOINT="${CHECKPOINT:-/work/joy/x-vla-workshop/artifacts/ckpt_linear_rat_vit_s0_v2.pt}"
DEVICE="${DEVICE:-cuda}"
BLOCK_SIZE="${BLOCK_SIZE:-4096}"

mkdir -p logs "$(dirname "$OUTPUT")"
if [[ "$LANE" == "tiny" ]]; then
  python scripts/run_implicit_sparse_projective_odt.py \
    --lane tiny \
    --stage "$STAGE" \
    --device cpu \
    --block-size 64 \
    --output "$OUTPUT"
else
  python scripts/run_implicit_sparse_projective_odt.py \
    --lane "$LANE" \
    --stage "$STAGE" \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE" \
    --block-size "$BLOCK_SIZE" \
    --output "$OUTPUT"
fi
