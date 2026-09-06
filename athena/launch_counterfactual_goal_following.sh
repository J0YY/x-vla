#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

readonly instruction_job=831389
readonly instruction_summary=results/instruction_necessity_v1_summary.json
readonly provenance_result=results/cache_provenance_libero_object.json
readonly cache=artifacts/libero_frames_100000_64.pkl
readonly manifest=artifacts/counterfactual_goal_following_v1_manifest.json
readonly bddl_dir=artifacts/counterfactual_goal_following_v1_bddl
readonly smoke_result=results/counterfactual_goal_following_v1_smoke.json
readonly summary_result=results/counterfactual_goal_following_v1_summary.json

declare -Ar checkpoints=(
  [0]=artifacts/ckpt_linear_rat_vit_s0_v2.pt
  [1]=artifacts/ckpt_linear_rat_vit_s1.pt
  [2]=artifacts/ckpt_linear_rat_vit_s2.pt
)
declare -Ar source_shas=(
  [athena/COUNTERFACTUAL_GOAL_FOLLOWING_PROTOCOL.md]=d468e6f05bd7b38b2387755379f17655b88206bd1e8f2409a3126f6ed25b5d03
  [athena/counterfactual_goal_following_common.py]=616201783cc07a42124f2ff3de3bb5f2b35cfd5a0cc855ffcbda102b046560f7
  [athena/preflight_counterfactual_goal_following.py]=83b68914ca33f99d143cc3068ab6e13ad3039d76246ecbe7333e790243f78b83
  [athena/run_counterfactual_goal_following.py]=81987bddc951dbff0f5ec5408cb581a6e1ed7642d48eaa8dac0e05bd7413a73c
  [athena/summarize_counterfactual_goal_following.py]=0e216eb07b6224d76af5202c57515466eebfaf6b5405682d9ebf84b3be309031
  [athena/slurm_preflight_counterfactual_goal_following.sbatch]=9c92afacc5850630c700e384ad9b561b76c5e3298d265d829247f89267e4361d
  [athena/slurm_counterfactual_goal_following.sbatch]=2d6ba6c07832cf84c76ed68872eafc5868356770bbb184457e60fe301839f924
  [athena/slurm_summarize_counterfactual_goal_following.sbatch]=18b311962e15d34f0bde6a0769a496b7407c8bdcd1a264854e2247466af404c8
)

for path in "${!source_shas[@]}"; do
  [[ -f "$path" ]] || { echo "Missing frozen source: $path" >&2; exit 3; }
  observed=$(sha256sum "$path" | cut -d' ' -f1)
  [[ "$observed" == "${source_shas[$path]}" ]] || {
    echo "Stale frozen source: $path" >&2
    exit 3
  }
done
for path in "$cache" "$provenance_result" "${checkpoints[0]}" "${checkpoints[1]}" "${checkpoints[2]}"; do
  [[ -f "$path" ]] || { echo "Missing frozen input: $path" >&2; exit 3; }
done
for path in "$manifest" "$bddl_dir" "$smoke_result" "$summary_result"; do
  [[ ! -e "$path" ]] || { echo "Refusing to overwrite: $path" >&2; exit 3; }
done
for seed in 0 1 2; do
  for shard in 0_2 2_4 4_6 6_8 8_10; do
    path="results/counterfactual_goal_following_v1_s${seed}_t${shard}.json"
    [[ ! -e "$path" ]] || { echo "Refusing to overwrite: $path" >&2; exit 3; }
  done
done

instruction_state=$(sacct -j "$instruction_job" --starttime 2026-08-21 --noheader --parsable2 --allocations --format=State | head -1 | cut -d'|' -f1)
case "$instruction_state" in
  COMPLETED)
    [[ -s "$instruction_summary" ]] || {
      echo "Completed instruction job has no strict summary" >&2
      exit 4
    }
    ;;
  RUNNING|PENDING|CONFIGURING|COMPLETING)
    ;;
  FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*)
    echo "Refusing launch because instruction summary job $instruction_job is $instruction_state" >&2
    exit 4
    ;;
  *)
    echo "Unexpected instruction summary job state: $instruction_state" >&2
    exit 4
    ;;
esac

preflight_job=$(sbatch --parsable \
  --dependency="afterok:${instruction_job}" \
  --job-name=xvla-goal-following-v1-preflight \
  athena/slurm_preflight_counterfactual_goal_following.sbatch \
  --instruction-summary "$instruction_summary" \
  --provenance-result "$provenance_result" \
  --cache "$cache" \
  --output "$manifest" \
  --bddl-dir "$bddl_dir")

smoke_job=$(sbatch --parsable \
  --dependency="afterok:${preflight_job}" \
  --time=12:00:00 \
  --job-name=xvla-goal-following-v1-smoke \
  athena/slurm_counterfactual_goal_following.sbatch \
  --mode strict_smoke \
  --checkpoint-seed 0 \
  --checkpoint "${checkpoints[0]}" \
  --cache "$cache" \
  --provenance-result "$provenance_result" \
  --instruction-summary "$instruction_summary" \
  --manifest "$manifest" \
  --task-start 0 --task-end 10 \
  --episode-start 20 --eps-per-task 1 \
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
    result="results/counterfactual_goal_following_v1_s${seed}_t${task_start}_${task_end}.json"
    full_jobs[$job_index]=$(sbatch --parsable \
      --dependency="afterok:${smoke_job}" \
      --job-name="xvla-goal-following-v1-s${seed}-t${task_start}-${task_end}" \
      athena/slurm_counterfactual_goal_following.sbatch \
      --mode full \
      --checkpoint-seed "$seed" \
      --checkpoint "${checkpoints[$seed]}" \
      --cache "$cache" \
      --provenance-result "$provenance_result" \
      --instruction-summary "$instruction_summary" \
      --manifest "$manifest" \
      --task-start "$task_start" --task-end "$task_end" \
      --episode-start 20 --eps-per-task 10 \
      --expected-rollouts 400 \
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
  --job-name=xvla-goal-following-v1-summary \
  athena/slurm_summarize_counterfactual_goal_following.sbatch \
  --smoke "$smoke_result" \
  "${result_args[@]}" \
  --manifest "$manifest" \
  --instruction-summary "$instruction_summary" \
  --output "$summary_result")

echo "instruction_summary $instruction_job $instruction_state"
echo "preflight $preflight_job"
echo "smoke $smoke_job"
for index in "${!full_jobs[@]}"; do echo "full_${index} ${full_jobs[$index]}"; done
echo "summary $summary_job"
