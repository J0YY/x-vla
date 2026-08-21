#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

cache=artifacts/libero_frames_100000_64.pkl
checkpoint=artifacts/ckpt_linear_rat_vit_s2.pt

smoke=$(sbatch --parsable \
  --partition=low-prio-gpu \
  --cpus-per-task=4 \
  --mem=32G \
  --time=01:00:00 \
  --job-name=xvla-cp-prune-smoke \
  athena/slurm_xvla.sbatch \
  --mode cp_pruning \
  --architecture chi \
  --checkpoint "$checkpoint" \
  --cache "$cache" \
  --output results/cp_pruning_smoke.json \
  --prune-fraction 0.5 \
  --prune-strategy magnitude \
  --task-start 0 \
  --task-end 1 \
  --eps-per-task 1 \
  --max-steps 8 \
  --profile-iters 20)

for fraction in 0.25 0.30 0.35 0.40 0.45 0.50 0.60 0.67 0.75; do
  label=${fraction/./p}
  sbatch --parsable \
    --partition=low-prio-gpu \
    --cpus-per-task=4 \
    --mem=32G \
    --time=06:00:00 \
    --dependency="afterok:${smoke}" \
    --job-name="xvla-cp-mag-${label}" \
    athena/slurm_xvla.sbatch \
    --mode cp_pruning \
    --architecture chi \
    --checkpoint "$checkpoint" \
    --cache "$cache" \
    --output "results/cp_pruning_magnitude_f${label}.json" \
    --prune-fraction "$fraction" \
    --prune-strategy magnitude \
    --seed 2 \
    --task-start 0 \
    --task-end 10 \
    --eps-per-task 5 \
    --max-steps 280 \
    --profile-iters 20

  for control_seed in 100 101 102; do
    sbatch --parsable \
      --partition=low-prio-gpu \
      --cpus-per-task=4 \
      --mem=32G \
      --time=06:00:00 \
      --dependency="afterok:${smoke}" \
      --job-name="xvla-cp-rand-${label}-${control_seed}" \
      athena/slurm_xvla.sbatch \
      --mode cp_pruning \
      --architecture chi \
      --checkpoint "$checkpoint" \
      --cache "$cache" \
      --output "results/cp_pruning_random_f${label}_s${control_seed}.json" \
      --prune-fraction "$fraction" \
      --prune-strategy random \
      --seed "$control_seed" \
      --task-start 0 \
      --task-end 10 \
      --eps-per-task 5 \
      --max-steps 280 \
      --profile-iters 20
  done
done

echo "CP-pruning smoke job: ${smoke}"
