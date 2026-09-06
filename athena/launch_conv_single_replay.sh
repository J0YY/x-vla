#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

cache=artifacts/libero_frames_100000_64.pkl
checkpoints=(
  artifacts/ckpt_linear_rat_conv_s0.pt
  artifacts/ckpt_linear_rat_conv_s1_matched.pt
  artifacts/ckpt_linear_rat_conv_s2_matched.pt
)

# Lock the paper's three-seed convolutional capability result in the same
# Athena environment and the historical evaluator's default precision mode.
for seed in 0 1 2; do
  for range in "0 3" "3 6" "6 8" "8 10"; do
    read -r start end <<<"$range"
    sbatch --parsable \
      --job-name="xvla-locked-conv-s${seed}-${start}${end}" \
      athena/slurm_xvla.sbatch \
      --mode capability \
      --architecture chi \
      --vision-encoder conv \
      --checkpoint "${checkpoints[$seed]}" \
      --cache "$cache" \
      --output "results/locked_conv_s${seed}_t${start}_${end}.json" \
      --seed "$seed" \
      --matmul-precision highest \
      --task-start "$start" \
      --task-end "$end" \
      --eps-per-task 50 \
      --max-steps 280 \
      --profile-iters 20
  done
done
