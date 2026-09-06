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

smoke=$(sbatch --parsable \
  --job-name=xvla-highest-precision-smoke \
  athena/slurm_xvla.sbatch \
  --mode capability \
  --architecture chi \
  --checkpoint "$vit_s0" \
  --cache "$cache" \
  --output results/highest_precision_smoke.json \
  --matmul-precision highest \
  --task-start 0 \
  --task-end 1 \
  --eps-per-task 1 \
  --max-steps 8 \
  --profile-iters 20)

for seed in 0 1 2; do
  case "$seed" in
    0) checkpoint=$vit_s0 ;;
    1) checkpoint=$vit_s1 ;;
    2) checkpoint=$vit_s2 ;;
  esac
  for range in "0 3" "3 6" "6 8" "8 10"; do
    read -r start end <<<"$range"
    sbatch --parsable \
      --dependency="afterok:${smoke}" \
      --job-name="xvla-highest-vit-s${seed}-${start}${end}" \
      athena/slurm_xvla.sbatch \
      --mode capability \
      --architecture chi \
      --checkpoint "$checkpoint" \
      --cache "$cache" \
      --output "results/highest_precision_vit_s${seed}_t${start}_${end}.json" \
      --seed "$seed" \
      --matmul-precision highest \
      --task-start "$start" \
      --task-end "$end" \
      --eps-per-task 50 \
      --max-steps 280 \
      --profile-iters 20
  done
done

for family in vit conv; do
  if [[ "$family" == vit ]]; then
    checkpoint0=$vit_s0
    checkpoint1=$vit_s1
    checkpoint2=$vit_s2
  else
    checkpoint0=$conv_s0
    checkpoint1=$conv_s1
    checkpoint2=$conv_s2
  fi
  for range in "0 3" "3 6" "6 8" "8 10"; do
    read -r start end <<<"$range"
    sbatch --parsable \
      --dependency="afterok:${smoke}" \
      --job-name="xvla-highest-${family}-ens-${start}${end}" \
      athena/slurm_xvla.sbatch \
      --mode ensemble_capability \
      --architecture chi \
      --vision-encoder "$family" \
      --checkpoint "$checkpoint0" \
      --ensemble-checkpoint "$checkpoint1" \
      --ensemble-checkpoint "$checkpoint2" \
      --ensemble-reduction mean \
      --cache "$cache" \
      --output "results/highest_precision_${family}_ensemble_t${start}_${end}.json" \
      --matmul-precision highest \
      --task-start "$start" \
      --task-end "$end" \
      --eps-per-task 50 \
      --max-steps 280 \
      --profile-iters 20
  done
done

echo "Highest-precision smoke job: ${smoke}"
