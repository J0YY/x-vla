#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

cache=artifacts/libero_frames_100000_64.pkl
vit_s0=artifacts/ckpt_linear_rat_vit_s0_v2.pt
vit_s1=artifacts/ckpt_linear_rat_vit_s1.pt
vit_s2=artifacts/ckpt_linear_rat_vit_s2.pt
conv_s0=artifacts/ckpt_linear_rat_conv_s0.pt
conv_s1=artifacts/ckpt_linear_rat_conv_s1_matched.pt
conv_s2=artifacts/ckpt_linear_rat_conv_s2_matched.pt

# Rank 96 is frozen from the task-4-to-7 sweep. Tasks 8 and 9 were not used to
# estimate the Gram or select the rank, so these jobs are the blind behavioral
# confirmation. Each task is a separate shard to maximize scheduler parallelism.
vit_dependencies=(830700 830701 830702)
for seed in 0 1 2; do
  case "$seed" in
    0) checkpoint=$vit_s0 ;;
    1) checkpoint=$vit_s1 ;;
    2) checkpoint=$vit_s2 ;;
  esac
  dependency=${vit_dependencies[$seed]}
  for task in 8 9; do
    end=$((task + 1))
    sbatch --parsable \
      --dependency="afterok:${dependency}" \
      --job-name="xvla-vit-r96-blind-s${seed}-t${task}" \
      athena/slurm_xvla.sbatch \
      --mode visual_subspace \
      --architecture chi \
      --vision-encoder vit \
      --checkpoint "$checkpoint" \
      --cache "$cache" \
      --output "results/visual_balanced_vit_s${seed}_r96_blind_t${task}.json" \
      --seed "$seed" \
      --rank 96 \
      --gram-action-group all_balanced \
      --gram-probes 4 \
      --gram-samples 1024 \
      --offline-eval-samples 1024 \
      --gram-task-start 0 \
      --gram-task-end 4 \
      --random-controls 3 \
      --task-start "$task" \
      --task-end "$end" \
      --eps-per-task 50 \
      --max-steps 400 \
      --profile-iters 20
  done

  # A larger random-control bank is cheap offline and makes the selectivity
  # estimate much less sensitive to any one random projector.
  sbatch --parsable \
    --dependency="afterok:${dependency}" \
    --job-name="xvla-vit-r96-random20-s${seed}" \
    athena/slurm_xvla.sbatch \
    --mode visual_subspace \
    --architecture chi \
    --vision-encoder vit \
    --checkpoint "$checkpoint" \
    --cache "$cache" \
    --output "results/visual_balanced_vit_s${seed}_r96_random20.json" \
    --seed "$seed" \
    --rank 96 \
    --gram-action-group all_balanced \
    --gram-probes 4 \
    --gram-samples 1024 \
    --offline-eval-samples 4096 \
    --gram-task-start 0 \
    --gram-task-end 4 \
    --random-controls 20 \
    --subspace-offline-only \
    --profile-iters 20
done

# Replicate the frozen rank and protocol with an independently trained
# convolutional visual encoder. The smoke gates all six blind task shards.
conv_smoke=$(sbatch --parsable \
  --job-name=xvla-conv-r96-subspace-smoke \
  athena/slurm_xvla.sbatch \
  --mode visual_subspace \
  --architecture chi \
  --vision-encoder conv \
  --checkpoint "$conv_s0" \
  --cache "$cache" \
  --output results/visual_balanced_conv_r96_smoke.json \
  --seed 0 \
  --rank 96 \
  --gram-action-group all_balanced \
  --gram-probes 1 \
  --gram-samples 32 \
  --offline-eval-samples 32 \
  --gram-task-start 0 \
  --gram-task-end 4 \
  --random-controls 2 \
  --subspace-offline-only \
  --profile-iters 20)

for seed in 0 1 2; do
  case "$seed" in
    0) checkpoint=$conv_s0 ;;
    1) checkpoint=$conv_s1 ;;
    2) checkpoint=$conv_s2 ;;
  esac
  for task in 8 9; do
    end=$((task + 1))
    sbatch --parsable \
      --dependency="afterok:${conv_smoke}" \
      --job-name="xvla-conv-r96-blind-s${seed}-t${task}" \
      athena/slurm_xvla.sbatch \
      --mode visual_subspace \
      --architecture chi \
      --vision-encoder conv \
      --checkpoint "$checkpoint" \
      --cache "$cache" \
      --output "results/visual_balanced_conv_s${seed}_r96_blind_t${task}.json" \
      --seed "$seed" \
      --rank 96 \
      --gram-action-group all_balanced \
      --gram-probes 4 \
      --gram-samples 1024 \
      --offline-eval-samples 1024 \
      --gram-task-start 0 \
      --gram-task-end 4 \
      --random-controls 3 \
      --task-start "$task" \
      --task-end "$end" \
      --eps-per-task 50 \
      --max-steps 400 \
      --profile-iters 20
  done

  sbatch --parsable \
    --dependency="afterok:${conv_smoke}" \
    --job-name="xvla-conv-r96-random20-s${seed}" \
    athena/slurm_xvla.sbatch \
    --mode visual_subspace \
    --architecture chi \
    --vision-encoder conv \
    --checkpoint "$checkpoint" \
    --cache "$cache" \
    --output "results/visual_balanced_conv_s${seed}_r96_random20.json" \
    --seed "$seed" \
    --rank 96 \
    --gram-action-group all_balanced \
    --gram-probes 4 \
    --gram-samples 1024 \
    --offline-eval-samples 4096 \
    --gram-task-start 0 \
    --gram-task-end 4 \
    --random-controls 20 \
    --subspace-offline-only \
    --profile-iters 20
done

echo "Convolutional smoke job: ${conv_smoke}"
