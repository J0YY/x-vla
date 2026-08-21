#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

cache=artifacts/libero_frames_100000_64.pkl
checkpoint=artifacts/ckpt_linear_rat_conv_s1_matched.pt

smoke=$(sbatch --parsable \
  --partition=low-prio-gpu \
  --qos=normal \
  --cpus-per-task=4 \
  --mem=32G \
  --time=01:00:00 \
  --job-name=xvla-conv-cp-smoke \
  athena/slurm_xvla.sbatch \
  --mode cp_pruning \
  --architecture chi \
  --vision-encoder conv \
  --checkpoint "$checkpoint" \
  --cache "$cache" \
  --output results/conv_cp_pruning_smoke.json \
  --prune-fraction 0.25 \
  --prune-strategy magnitude \
  --task-start 0 \
  --task-end 1 \
  --eps-per-task 1 \
  --max-steps 8 \
  --profile-iters 20)

sbatch --parsable \
  --partition=low-prio-gpu \
  --qos=normal \
  --cpus-per-task=4 \
  --mem=32G \
  --time=06:00:00 \
  --dependency="afterok:${smoke}" \
  --job-name=xvla-conv-baseline-5 \
  athena/slurm_xvla.sbatch \
  --mode capability \
  --architecture chi \
  --vision-encoder conv \
  --checkpoint "$checkpoint" \
  --cache "$cache" \
  --output results/conv_s1_baseline_first5.json \
  --task-start 0 \
  --task-end 10 \
  --eps-per-task 5 \
  --max-steps 280 \
  --profile-iters 20

for fraction in 0.25 0.30 0.35 0.40 0.45 0.50; do
  label=${fraction/./p}
  sbatch --parsable \
    --partition=low-prio-gpu \
    --qos=normal \
    --cpus-per-task=4 \
    --mem=32G \
    --time=06:00:00 \
    --dependency="afterok:${smoke}" \
    --job-name="xvla-conv-cp-mag-${label}" \
    athena/slurm_xvla.sbatch \
    --mode cp_pruning \
    --architecture chi \
    --vision-encoder conv \
    --checkpoint "$checkpoint" \
    --cache "$cache" \
    --output "results/conv_cp_pruning_magnitude_f${label}.json" \
    --prune-fraction "$fraction" \
    --prune-strategy magnitude \
    --seed 1 \
    --task-start 0 \
    --task-end 10 \
    --eps-per-task 5 \
    --max-steps 280 \
    --profile-iters 20
done

echo "Convolutional CP-pruning smoke job: ${smoke}"

