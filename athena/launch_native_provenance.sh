#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

cache=artifacts/libero_frames_100000_64.pkl

submit_training() {
  local architecture=$1
  local seed=$2
  local checkpoint=$3
  local result=$4
  sbatch --parsable \
    --job-name="xvla-native-${architecture}-s${seed}" \
    athena/slurm_train_checkpoint.sbatch \
    --architecture "$architecture" \
    --cache "$cache" \
    --checkpoint-output "$checkpoint" \
    --result-output "$result" \
    --seed "$seed" \
    --steps 40000 \
    --batch-size 256 \
    --lr 8e-4 \
    --ema-decay 0.999
}

submit_evaluation() {
  local architecture=$1
  local seed=$2
  local checkpoint=$3
  local dependency=$4
  local range start end
  for range in "0 3" "3 6" "6 8" "8 10"; do
    read -r start end <<<"$range"
    sbatch --parsable \
      --dependency="afterok:${dependency}" \
      --job-name="xvla-native-${architecture}-s${seed}-${start}${end}" \
      athena/slurm_xvla.sbatch \
      --mode capability \
      --architecture "$architecture" \
      --checkpoint "$checkpoint" \
      --cache "$cache" \
      --output "results/native_${architecture}_s${seed}_t${start}_${end}.json" \
      --seed "$seed" \
      --task-start "$start" \
      --task-end "$end" \
      --eps-per-task 50 \
      --max-steps 280
  done
}

conventional_checkpoint=artifacts/ckpt_native_conventional_s0.pt
conventional_job=$(submit_training \
  conventional 0 "$conventional_checkpoint" results/train_native_conventional_s0.json)

declare -A chi_jobs
for seed in 0 1 2; do
  checkpoint="artifacts/ckpt_native_chi_s${seed}.pt"
  chi_jobs[$seed]=$(submit_training \
    chi "$seed" "$checkpoint" "results/train_native_chi_s${seed}.json")
done

submit_evaluation conventional 0 "$conventional_checkpoint" "$conventional_job"
for seed in 0 1 2; do
  submit_evaluation \
    chi "$seed" "artifacts/ckpt_native_chi_s${seed}.pt" "${chi_jobs[$seed]}"
done

echo "Native training jobs: conventional=${conventional_job}, chi0=${chi_jobs[0]}, chi1=${chi_jobs[1]}, chi2=${chi_jobs[2]}"
