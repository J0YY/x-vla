#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

declare -A train_jobs=(
  [libero_spatial]=830781
  [libero_goal]=830783
  [libero_10]=830667
)
declare -A caches=(
  [libero_spatial]=artifacts/libero_spatial_frames_100000_64.pkl
  [libero_goal]=artifacts/libero_goal_frames_100000_64.pkl
  [libero_10]=artifacts/libero_10_frames_100000_64.pkl
)
declare -a evaluation_jobs=()

for suite in libero_spatial libero_goal libero_10; do
  checkpoint="artifacts/ckpt_indomain_${suite}_chi_s0.pt"
  for range in "0 3" "3 6" "6 8" "8 10"; do
    read -r start end <<<"$range"
    job=$(sbatch --parsable \
      --partition=low-prio-gpu \
      --qos=normal \
      --dependency="afterok:${train_jobs[$suite]}" \
      --job-name="xvla-retention-${suite}-${start}${end}" \
      athena/slurm_xvla.sbatch \
      --mode capability \
      --architecture chi \
      --vision-encoder vit \
      --suite "$suite" \
      --training-suite "$suite" \
      --checkpoint "$checkpoint" \
      --cache "${caches[$suite]}" \
      --output "results/corrected_specialist_${suite}_chi_s0_t${start}_${end}.json" \
      --seed 0 \
      --matmul-precision highest \
      --task-start "$start" \
      --task-end "$end" \
      --eps-per-task 50 \
      --max-steps 280 \
      --profile-iters 20)
    evaluation_jobs+=("$job")
  done
done

all_dependencies=(
  830894 830895 830896 830897
  830929
  "${evaluation_jobs[@]}"
)
dependency=$(IFS=:; echo "${all_dependencies[*]}")
summary_job=$(sbatch --parsable \
  --dependency="afterok:${dependency}" \
  --job-name=xvla-specialist-retention-summary \
  athena/slurm_specialist_retention_summary.sbatch \
  --results-dir results \
  --output results/generalist_specialist_retention_summary.json)

echo "Specialist evaluation jobs: ${evaluation_jobs[*]}"
echo "Retention summary job: $summary_job"
