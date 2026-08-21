#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

cache=artifacts/libero_frames_100000_64.pkl
corrected_s0=artifacts/ckpt_ensemble_dagger_s0.pt
corrected_s1=artifacts/ckpt_ensemble_dagger_s1.pt
corrected_s2=artifacts/ckpt_ensemble_dagger_s2.pt

smoke=$(sbatch --parsable \
  --partition=low-prio-gpu \
  --qos=normal \
  --cpus-per-task=4 \
  --mem=48G \
  --time=01:00:00 \
  --job-name=xvla-corrected-ens-smoke \
  athena/slurm_xvla.sbatch \
  --mode ensemble_capability \
  --architecture chi \
  --vision-encoder conv \
  --checkpoint "$corrected_s0" \
  --ensemble-checkpoint "$corrected_s1" \
  --ensemble-checkpoint "$corrected_s2" \
  --ensemble-reduction mean \
  --cache "$cache" \
  --output results/corrected_conv_ensemble_smoke.json \
  --task-start 0 \
  --task-end 1 \
  --eps-per-task 1 \
  --max-steps 8 \
  --profile-iters 20)

for range in "0 3" "3 6" "6 8" "8 10"; do
  read -r start end <<<"$range"
  sbatch --parsable \
    --partition=low-prio-gpu \
    --qos=normal \
    --cpus-per-task=4 \
    --mem=48G \
    --time=12:00:00 \
    --dependency="afterok:${smoke}" \
    --job-name="xvla-corrected-ens-${start}${end}" \
    athena/slurm_xvla.sbatch \
    --mode ensemble_capability \
    --architecture chi \
    --vision-encoder conv \
    --checkpoint "$corrected_s0" \
    --ensemble-checkpoint "$corrected_s1" \
    --ensemble-checkpoint "$corrected_s2" \
    --ensemble-reduction mean \
    --cache "$cache" \
    --output "results/corrected_conv_ensemble_mean_t${start}_${end}.json" \
    --task-start "$start" \
    --task-end "$end" \
    --eps-per-task 50 \
    --max-steps 280 \
    --profile-iters 20
done

echo "Corrected convolutional ensemble smoke job: ${smoke}"

