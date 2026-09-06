#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

provenance_job=830988
provenance_result=results/cache_provenance_libero_object.json
cache=artifacts/libero_frames_100000_64.pkl
cache_sha=053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662

declare -A checkpoints=(
  [0]=artifacts/ckpt_linear_rat_vit_s0_v2.pt
  [1]=artifacts/ckpt_linear_rat_vit_s1.pt
  [2]=artifacts/ckpt_linear_rat_vit_s2.pt
)
declare -A checkpoint_shas=(
  [0]=96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9
  [1]=cc0780b989165a80a449c03cbb3574e2f56b2e5747a64cf32d1fde418df8ec3c
  [2]=413a770071bd8f16b924c5604f7c7a58567d2eb6c40c66aab202bcf046fec910
)

provenance_state=$(sacct -j "$provenance_job" --starttime 2026-08-20 \
  --noheader --parsable2 --allocations --format=State | head -1 | cut -d'|' -f1)
case "$provenance_state" in
  FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*)
    echo "Refusing to launch: Object provenance job $provenance_job is $provenance_state" >&2
    exit 4
    ;;
  COMPLETED)
    if [[ ! -s "$provenance_result" ]]; then
      echo "Refusing to launch: completed provenance job has no result" >&2
      exit 4
    fi
    ;;
  RUNNING|PENDING|CONFIGURING|COMPLETING)
    ;;
  *)
    echo "Refusing to launch: unexpected provenance state '$provenance_state'" >&2
    exit 4
    ;;
esac

smoke_submit=(sbatch --parsable)
if [[ "$provenance_state" != "COMPLETED" ]]; then
  smoke_submit+=(--dependency="afterok:${provenance_job}")
fi

for path in "$cache" "${checkpoints[0]}" "${checkpoints[1]}" "${checkpoints[2]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing frozen input: $path" >&2
    exit 3
  fi
done
for output in \
  results/local_instruction_specificity_v2_smoke.json \
  results/local_instruction_specificity_v2_s0.json \
  results/local_instruction_specificity_v2_s1.json \
  results/local_instruction_specificity_v2_s2.json \
  results/local_instruction_specificity_v2_summary.json; do
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing result: $output" >&2
    exit 3
  fi
done

smoke_job=$("${smoke_submit[@]}" \
  --job-name=xvla-local-language-v2-smoke \
  athena/slurm_local_instruction_specificity.sbatch \
  --checkpoint "${checkpoints[0]}" \
  --expected-checkpoint-sha256 "${checkpoint_shas[0]}" \
  --cache "$cache" \
  --expected-cache-sha256 "$cache_sha" \
  --provenance-result "$provenance_result" \
  --expected-provenance-job-id "$provenance_job" \
  --output results/local_instruction_specificity_v2_smoke.json \
  --checkpoint-seed 0 \
  --mode strict_smoke \
  --task-start 0 \
  --task-end 10 \
  --episode-start 0 \
  --eps-per-task 1 \
  --expected-trials 10 \
  --res 64 \
  --horizon 8 \
  --num-steps-wait 10 \
  --matmul-precision highest)

declare -a full_jobs=()
for seed in 0 1 2; do
  full_jobs[$seed]=$(sbatch --parsable \
    --dependency="afterok:${smoke_job}" \
    --job-name="xvla-local-language-v2-s${seed}" \
    athena/slurm_local_instruction_specificity.sbatch \
    --checkpoint "${checkpoints[$seed]}" \
    --expected-checkpoint-sha256 "${checkpoint_shas[$seed]}" \
    --cache "$cache" \
    --expected-cache-sha256 "$cache_sha" \
    --provenance-result "$provenance_result" \
    --expected-provenance-job-id "$provenance_job" \
    --output "results/local_instruction_specificity_v2_s${seed}.json" \
    --checkpoint-seed "$seed" \
    --mode full \
    --task-start 0 \
    --task-end 10 \
    --episode-start 10 \
    --eps-per-task 40 \
    --expected-trials 400 \
    --res 64 \
    --horizon 8 \
    --num-steps-wait 10 \
    --matmul-precision highest)
done

dependency="afterok:${full_jobs[0]}:${full_jobs[1]}:${full_jobs[2]}"
summary_job=$(sbatch --parsable \
  --dependency="$dependency" \
  --job-name=xvla-local-language-v2-summary \
  athena/slurm_summarize_local_instruction_specificity.sbatch \
  --result results/local_instruction_specificity_v2_s0.json \
  --result results/local_instruction_specificity_v2_s1.json \
  --result results/local_instruction_specificity_v2_s2.json \
  --output results/local_instruction_specificity_v2_summary.json)

echo "provenance $provenance_job $provenance_state"
echo "smoke $smoke_job"
echo "full_s0 ${full_jobs[0]}"
echo "full_s1 ${full_jobs[1]}"
echo "full_s2 ${full_jobs[2]}"
echo "summary $summary_job"
