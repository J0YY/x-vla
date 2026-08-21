#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

readonly specificity_job=831016
readonly specificity_result=results/local_instruction_specificity_v2_summary.json
readonly provenance_result=results/cache_provenance_libero_object.json
readonly cache=artifacts/libero_frames_100000_64.pkl
readonly manifest=artifacts/counterfactual_target_swap_v1_manifest.json
readonly bddl_dir=artifacts/counterfactual_target_swap_v1_bddl
readonly smoke_result=results/counterfactual_target_swap_v1_smoke.json
readonly summary_result=results/counterfactual_target_swap_v1_summary.json

declare -Ar checkpoints=(
  [0]=artifacts/ckpt_linear_rat_vit_s0_v2.pt
  [1]=artifacts/ckpt_linear_rat_vit_s1.pt
  [2]=artifacts/ckpt_linear_rat_vit_s2.pt
)
declare -Ar source_shas=(
  [athena/counterfactual_target_swap_common.py]=26d185f15f787ac292a3457cb66b9773cd4281f3ee4050fc9971a6dd4d817b81
  [athena/preflight_counterfactual_target_swap.py]=06bc40b323e741d229bf54c19a288fd71537b9fd756140b9408a394d49e7c19e
  [athena/run_counterfactual_target_swap.py]=298a4b2bef1501c10e00d4818fc73b69cbc1f9e6aaa26e580ef8f290938f233a
  [athena/summarize_counterfactual_target_swap.py]=c15b3310ef937663ec7b57ad7d1115ade98e896aa69f27a7e6cf86815fa49759
  [athena/slurm_preflight_counterfactual_target_swap.sbatch]=874aebf5363e61c3aa69f176507f4e7d5ef44705c1e06a9a233d131f1299fe24
  [athena/slurm_counterfactual_target_swap.sbatch]=51420807c7dec82166b42f660d3bbe102e40609e8a0b0b9a57a85c6b519f427f
  [athena/slurm_summarize_counterfactual_target_swap.sbatch]=de16dc4d1b945bda5aa9600881588741db4ef5f13006468fcd73424dedd3b041
)

for path in "${!source_shas[@]}"; do
  [[ -f "$path" ]] || { echo "Missing frozen source: $path" >&2; exit 3; }
  observed=$(sha256sum "$path" | cut -d' ' -f1)
  [[ "$observed" == "${source_shas[$path]}" ]] || { echo "Stale frozen source: $path" >&2; exit 3; }
done
for path in "$cache" "$provenance_result" "${checkpoints[0]}" "${checkpoints[1]}" "${checkpoints[2]}"; do
  [[ -f "$path" ]] || { echo "Missing frozen input: $path" >&2; exit 3; }
done

for path in "$manifest" "$bddl_dir" "$smoke_result" "$summary_result"; do
  [[ ! -e "$path" ]] || { echo "Refusing to overwrite: $path" >&2; exit 3; }
done
for seed in 0 1 2; do
  for shard in 0_2 2_4 4_6 6_8 8_10; do
    path="results/counterfactual_target_swap_v1_s${seed}_t${shard}.json"
    [[ ! -e "$path" ]] || { echo "Refusing to overwrite: $path" >&2; exit 3; }
  done
done

specificity_state=$(sacct -j "$specificity_job" --starttime 2026-08-20 --noheader --parsable2 --allocations --format=State | head -1 | cut -d'|' -f1)
case "$specificity_state" in
  COMPLETED)
    [[ -s "$specificity_result" ]] || { echo "Completed specificity job has no summary" >&2; exit 4; }
    ;;
  RUNNING|PENDING|CONFIGURING|COMPLETING)
    ;;
  FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*)
    echo "Refusing launch because specificity job $specificity_job is $specificity_state" >&2
    exit 4
    ;;
  *)
    echo "Unexpected specificity job state: $specificity_state" >&2
    exit 4
    ;;
esac

preflight_submit=(sbatch --parsable)
if [[ "$specificity_state" != COMPLETED ]]; then
  preflight_submit+=(--dependency="afterok:${specificity_job}")
fi
preflight_job=$("${preflight_submit[@]}" \
  --job-name=xvla-target-swap-v1-preflight \
  athena/slurm_preflight_counterfactual_target_swap.sbatch \
  --specificity-summary "$specificity_result" \
  --provenance-result "$provenance_result" \
  --cache "$cache" \
  --output "$manifest" \
  --bddl-dir "$bddl_dir")

smoke_job=$(sbatch --parsable \
  --dependency="afterok:${preflight_job}" \
  --time=06:00:00 \
  --job-name=xvla-target-swap-v1-smoke \
  athena/slurm_counterfactual_target_swap.sbatch \
  --mode strict_smoke \
  --checkpoint-seed 0 \
  --checkpoint "${checkpoints[0]}" \
  --cache "$cache" \
  --provenance-result "$provenance_result" \
  --specificity-summary "$specificity_result" \
  --manifest "$manifest" \
  --task-start 0 --task-end 10 \
  --episode-start 40 --eps-per-task 1 \
  --expected-rollouts 0 \
  --res 64 --horizon 8 --exec-h 8 --max-steps 280 --num-steps-wait 10 \
  --matmul-precision highest \
  --output "$smoke_result")

declare -a full_jobs=()
declare -a result_args=()
job_index=0
for seed in 0 1 2; do
  for bounds in "0 2" "2 4" "4 6" "6 8" "8 10"; do
    read -r task_start task_end <<< "$bounds"
    result="results/counterfactual_target_swap_v1_s${seed}_t${task_start}_${task_end}.json"
    full_jobs[$job_index]=$(sbatch --parsable \
      --dependency="afterok:${smoke_job}" \
      --job-name="xvla-target-swap-v1-s${seed}-t${task_start}-${task_end}" \
      athena/slurm_counterfactual_target_swap.sbatch \
      --mode full \
      --checkpoint-seed "$seed" \
      --checkpoint "${checkpoints[$seed]}" \
      --cache "$cache" \
      --provenance-result "$provenance_result" \
      --specificity-summary "$specificity_result" \
      --manifest "$manifest" \
      --task-start "$task_start" --task-end "$task_end" \
      --episode-start 40 --eps-per-task 10 \
      --expected-rollouts 220 \
      --res 64 --horizon 8 --exec-h 8 --max-steps 280 --num-steps-wait 10 \
      --matmul-precision highest \
      --output "$result")
    result_args+=(--result "$result")
    ((job_index += 1))
  done
done

dependency="afterok:$(IFS=:; echo "${full_jobs[*]}")"
summary_job=$(sbatch --parsable \
  --dependency="$dependency" \
  --job-name=xvla-target-swap-v1-summary \
  athena/slurm_summarize_counterfactual_target_swap.sbatch \
  --smoke "$smoke_result" \
  "${result_args[@]}" \
  --manifest "$manifest" \
  --specificity-summary "$specificity_result" \
  --output "$summary_result")

echo "specificity $specificity_job $specificity_state"
echo "preflight $preflight_job"
echo "smoke $smoke_job"
for index in "${!full_jobs[@]}"; do echo "full_${index} ${full_jobs[$index]}"; done
echo "summary $summary_job"
