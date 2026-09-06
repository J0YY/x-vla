#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

cache=artifacts/libero_frames_100000_64.pkl
conv_s0=artifacts/ckpt_linear_rat_conv_s0.pt
conv_s1=artifacts/ckpt_linear_rat_conv_s1_matched.pt
conv_s2=artifacts/ckpt_linear_rat_conv_s2_matched.pt

smoke=$(sbatch --parsable \
  --partition=low-prio-gpu \
  --qos=normal \
  --cpus-per-task=4 \
  --mem=48G \
  --time=01:00:00 \
  --job-name=xvla-conv-ensemble-smoke \
  athena/slurm_xvla.sbatch \
  --mode ensemble_capability \
  --architecture chi \
  --vision-encoder conv \
  --checkpoint "$conv_s0" \
  --ensemble-checkpoint "$conv_s1" \
  --ensemble-checkpoint "$conv_s2" \
  --ensemble-reduction mean \
  --cache "$cache" \
  --output results/conv_ensemble_mean_smoke.json \
  --task-start 0 \
  --task-end 1 \
  --eps-per-task 1 \
  --max-steps 8 \
  --profile-iters 20)

for reduction in mean median; do
  for range in "0 3" "3 6" "6 8" "8 10"; do
    read -r start end <<<"$range"
    sbatch --parsable \
      --partition=low-prio-gpu \
      --qos=normal \
      --cpus-per-task=4 \
      --mem=48G \
      --time=12:00:00 \
      --dependency="afterok:${smoke}" \
      --job-name="xvla-conv-ens-${reduction}-${start}${end}" \
      athena/slurm_xvla.sbatch \
      --mode ensemble_capability \
      --architecture chi \
      --vision-encoder conv \
      --checkpoint "$conv_s0" \
      --ensemble-checkpoint "$conv_s1" \
      --ensemble-checkpoint "$conv_s2" \
      --ensemble-reduction "$reduction" \
      --cache "$cache" \
      --output "results/conv_ensemble_${reduction}_t${start}_${end}.json" \
      --task-start "$start" \
      --task-end "$end" \
      --eps-per-task 50 \
      --max-steps 280 \
      --profile-iters 20
  done
done

echo "Convolutional ensemble smoke job: ${smoke}"

