#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

checkpoints=(
  artifacts/ckpt_linear_rat_vit_s0_v2.pt
  artifacts/ckpt_linear_rat_vit_s1.pt
  artifacts/ckpt_linear_rat_vit_s2.pt
)

for seed in 0 1 2; do
  sbatch --parsable \
    --job-name="xvla-vit-rank-sweep-s${seed}" \
    athena/slurm_visual_rank_sweep.sbatch \
    "${checkpoints[$seed]}" \
    "$seed" \
    vit \
    "visual_balanced_pairedrank_vit_s${seed}"
done
