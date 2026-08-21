#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

libero10_cache_job=${LIBERO10_CACHE_JOB:-830646}
manifest=artifacts/libero_all_manifest.json

manifest_job=$(sbatch --parsable \
  --dependency="afterok:${libero10_cache_job}" \
  --job-name=xvla-generalist-manifest \
  athena/slurm_build_multisuite_manifest.sbatch \
  --suite-cache libero_object=artifacts/libero_frames_100000_64.pkl \
  --suite-cache libero_spatial=artifacts/libero_spatial_frames_100000_64.pkl \
  --suite-cache libero_goal=artifacts/libero_goal_frames_100000_64.pkl \
  --suite-cache libero_10=artifacts/libero_10_frames_100000_64.pkl \
  --output "$manifest" \
  --horizon 8 \
  --res 64)

declare -A smoke_jobs
declare -A evaluator_smoke_jobs
declare -a evaluation_jobs=()

for architecture in chi conventional; do
  smoke_jobs[$architecture]=$(sbatch --parsable \
    --dependency="afterok:${manifest_job}" \
    --job-name="xvla-generalist-${architecture}-smoke" \
    athena/slurm_train_multisuite.sbatch \
    --architecture "$architecture" \
    --vision-encoder vit \
    --manifest "$manifest" \
    --checkpoint-output "artifacts/ckpt_generalist_${architecture}_smoke.pt" \
    --metadata-output "artifacts/ckpt_generalist_${architecture}_smoke.json" \
    --result-output "results/train_generalist_${architecture}_smoke.json" \
    --seed 0 \
    --steps 10 \
    --batch-size 256 \
    --lr 8e-4 \
    --ema-decay 0.999)

  evaluator_smoke_jobs[$architecture]=$(sbatch --parsable \
    --dependency="afterok:${smoke_jobs[$architecture]}" \
    --job-name="xvla-generalist-${architecture}-eval-smoke" \
    athena/slurm_xvla.sbatch \
    --mode smoke \
    --architecture "$architecture" \
    --vision-encoder vit \
    --suite libero_object \
    --training-suite libero_object \
    --checkpoint "artifacts/ckpt_generalist_${architecture}_smoke.pt" \
    --model-metadata "artifacts/ckpt_generalist_${architecture}_smoke.json" \
    --cache artifacts/libero_frames_100000_64.pkl \
    --output "results/generalist_${architecture}_eval_smoke.json" \
    --seed 0 \
    --matmul-precision highest \
    --task-start 0 \
    --task-end 1 \
    --eps-per-task 1 \
    --max-steps 8 \
    --profile-iters 2)
done

cache_for_suite() {
  case "$1" in
    libero_object) echo artifacts/libero_frames_100000_64.pkl ;;
    libero_spatial) echo artifacts/libero_spatial_frames_100000_64.pkl ;;
    libero_goal) echo artifacts/libero_goal_frames_100000_64.pkl ;;
    libero_10) echo artifacts/libero_10_frames_100000_64.pkl ;;
    *) return 1 ;;
  esac
}

for architecture in chi conventional; do
  for seed in 0 1 2; do
    checkpoint="artifacts/ckpt_generalist_${architecture}_s${seed}.pt"
    metadata="artifacts/ckpt_generalist_${architecture}_s${seed}.json"
    train_job=$(sbatch --parsable \
      --dependency="afterok:${evaluator_smoke_jobs[$architecture]}" \
      --job-name="xvla-generalist-${architecture}-s${seed}" \
      athena/slurm_train_multisuite.sbatch \
      --architecture "$architecture" \
      --vision-encoder vit \
      --manifest "$manifest" \
      --checkpoint-output "$checkpoint" \
      --metadata-output "$metadata" \
      --result-output "results/train_generalist_${architecture}_s${seed}.json" \
      --seed "$seed" \
      --steps 160000 \
      --batch-size 256 \
      --lr 8e-4 \
      --ema-decay 0.999 \
      --recovery-output "artifacts/recovery_generalist_${architecture}_s${seed}.pt" \
      --recovery-interval 10000)

    for suite in libero_object libero_spatial libero_goal libero_10; do
      cache=$(cache_for_suite "$suite")
      for range in "0 3" "3 6" "6 8" "8 10"; do
        read -r start end <<<"$range"
        evaluation_job=$(sbatch --parsable \
          --partition=low-prio-gpu \
          --qos=normal \
          --dependency="afterok:${train_job}" \
          --job-name="xvla-gen-${architecture}-s${seed}-${suite}-${start}${end}" \
          athena/slurm_xvla.sbatch \
          --mode capability \
          --architecture "$architecture" \
          --vision-encoder vit \
          --suite "$suite" \
          --training-suite "$suite" \
          --checkpoint "$checkpoint" \
          --model-metadata "$metadata" \
          --cache "$cache" \
          --output "results/generalist_${architecture}_s${seed}_${suite}_t${start}_${end}.json" \
          --seed "$seed" \
          --matmul-precision highest \
          --task-start "$start" \
          --task-end "$end" \
          --eps-per-task 50 \
          --max-steps 280 \
          --profile-iters 20)
        evaluation_jobs+=("$evaluation_job")
      done
    done
    echo "$architecture seed $seed training job: $train_job"
  done
done

evaluation_dependency=$(IFS=:; echo "${evaluation_jobs[*]}")
summary_job=$(sbatch --parsable \
  --dependency="afterok:${evaluation_dependency}" \
  --job-name=xvla-generalist-summary \
  athena/slurm_multisuite_summary.sbatch \
  --results-dir results \
  --output results/generalist_summary.json)

echo "Manifest job: $manifest_job"
echo "Chi smoke job: ${smoke_jobs[chi]}"
echo "Conventional smoke job: ${smoke_jobs[conventional]}"
echo "Chi evaluator smoke job: ${evaluator_smoke_jobs[chi]}"
echo "Conventional evaluator smoke job: ${evaluator_smoke_jobs[conventional]}"
echo "Evaluation jobs: ${evaluation_jobs[*]}"
echo "Summary job: $summary_job"
