#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

declare -A cache_jobs
declare -A caches
for suite in libero_spatial libero_goal libero_10; do
  cache="artifacts/${suite}_frames_100000_64.pkl"
  caches[$suite]=$cache
  cache_jobs[$suite]=$(sbatch --parsable \
    --job-name="xvla-cache-${suite}" \
    athena/slurm_build_suite_cache.sbatch \
    --suite "$suite" \
    --output "$cache" \
    --result-output "results/cache_${suite}.json" \
    --n-frames 100000 \
    --res 64)
done

submit_training() {
  local suite=$1
  local architecture=$2
  local cache_job=$3
  local cache=$4
  local checkpoint="artifacts/ckpt_indomain_${suite}_${architecture}_s0.pt"
  local result="results/train_indomain_${suite}_${architecture}_s0.json"
  sbatch --parsable \
    --dependency="afterok:${cache_job}" \
    --job-name="xvla-train-${suite}-${architecture}" \
    athena/slurm_train_checkpoint.sbatch \
    --architecture "$architecture" \
    --suite "$suite" \
    --cache "$cache" \
    --checkpoint-output "$checkpoint" \
    --result-output "$result" \
    --seed 0 \
    --steps 40000 \
    --batch-size 256 \
    --lr 8e-4 \
    --ema-decay 0.999
}

submit_evaluations() {
  local suite=$1
  local architecture=$2
  local training_job=$3
  local cache=$4
  local checkpoint="artifacts/ckpt_indomain_${suite}_${architecture}_s0.pt"
  local range start end
  for range in "0 3" "3 6" "6 8" "8 10"; do
    read -r start end <<<"$range"
    sbatch --parsable \
      --dependency="afterok:${training_job}" \
      --job-name="xvla-${suite}-${architecture}-${start}${end}" \
      athena/slurm_xvla.sbatch \
      --mode capability \
      --architecture "$architecture" \
      --suite "$suite" \
      --training-suite "$suite" \
      --checkpoint "$checkpoint" \
      --cache "$cache" \
      --output "results/indomain_${suite}_${architecture}_s0_t${start}_${end}.json" \
      --seed 0 \
      --task-start "$start" \
      --task-end "$end" \
      --eps-per-task 50 \
      --max-steps 280
  done
}

for suite in libero_spatial libero_goal libero_10; do
  for architecture in chi conventional; do
    training_job=$(submit_training \
      "$suite" "$architecture" "${cache_jobs[$suite]}" "${caches[$suite]}")
    submit_evaluations \
      "$suite" "$architecture" "$training_job" "${caches[$suite]}"
    echo "${suite} ${architecture}: cache=${cache_jobs[$suite]}, training=${training_job}"
  done
done
