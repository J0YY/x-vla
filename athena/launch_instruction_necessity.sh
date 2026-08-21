#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

readonly failed_local_job=831016
readonly failed_local_summary=results/local_instruction_specificity_v2_summary.json
readonly cache=artifacts/libero_frames_100000_64.pkl
readonly provenance=results/cache_provenance_libero_object.json
readonly manifest=artifacts/instruction_necessity_v1_manifest.json
readonly smoke_result=results/instruction_necessity_v1_smoke.json
readonly summary_result=results/instruction_necessity_v1_summary.json
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python

declare -Ar checkpoints=(
  [0]=artifacts/ckpt_linear_rat_vit_s0_v2.pt
  [1]=artifacts/ckpt_linear_rat_vit_s1.pt
  [2]=artifacts/ckpt_linear_rat_vit_s2.pt
)
declare -Ar expected_shas=(
  [athena/instruction_necessity_common.py]=8ab8839241adbc93ddc486c4b27a96f3bd78e1afd4a9f543603ee2206efc7551
  [athena/preflight_instruction_necessity.py]=ed689e9ec2678ea28ca594a4778ddf6ee69244b3a4bd03c2067e37f3232d2c32
  [athena/run_instruction_necessity.py]=90e06ec5d29d2c2f0b466dc259984190240270ce3110f787352de78689496a7e
  [athena/summarize_instruction_necessity.py]=7bea6aee78d1594709524d110f91d973bd96e720ad04d618fde61ad405820263
  [athena/slurm_preflight_instruction_necessity.sbatch]=eb62bc78e88d7a1331742334a5a369c6d22274ce8c0cfad2fdb9de226ee279df
  [athena/slurm_instruction_necessity.sbatch]=168b40dd7047317cb459372335332ac42e55f6dfe6e2f7426b5082f89ec5b516
  [athena/slurm_summarize_instruction_necessity.sbatch]=9998587f4f799dc6078bb93b5151a1558f8d31f70e929ed8b5c0c4b92d2edce3
)

for path in "${!expected_shas[@]}"; do
  [[ -f "$path" ]] || { echo "Missing frozen source: $path" >&2; exit 3; }
  observed=$(sha256sum "$path" | cut -d' ' -f1)
  [[ "$observed" == "${expected_shas[$path]}" ]] || {
    echo "Stale frozen source: $path" >&2
    exit 3
  }
done

state=$(sacct -j "$failed_local_job" --starttime 2026-08-20 --noheader --parsable2 --allocations --format=State | head -1 | cut -d'|' -f1)
[[ "$state" == COMPLETED ]] || {
  echo "Refusing launch until failed local endpoint has a completed formal summary: $state" >&2
  exit 4
}
[[ -s "$failed_local_summary" ]] || { echo "Failed local summary is absent" >&2; exit 4; }
[[ -x "$python_executable" ]] || { echo "Frozen Python executable is absent" >&2; exit 3; }
"$python_executable" - "$failed_local_summary" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
assert result["schema"] == "xvla-local-instruction-specificity-summary-v2"
assert result["identity_validated"] is True
assert result["overall_pass"] is False
assert result["claim_eligible"] is False
assert set(result["checkpoint_results"]) == {"0", "1", "2"}
assert not all(row["passes_frozen_gate"] for row in result["checkpoint_results"].values())
PY

for path in "$cache" "$provenance" "$failed_local_summary" "${checkpoints[0]}" "${checkpoints[1]}" "${checkpoints[2]}"; do
  [[ -f "$path" ]] || { echo "Missing frozen input: $path" >&2; exit 3; }
done
for path in "$manifest" "$smoke_result" "$summary_result"; do
  [[ ! -e "$path" ]] || { echo "Refusing to overwrite: $path" >&2; exit 3; }
done
for seed in 0 1 2; do
  for shard in 0_2 2_4 4_6 6_8 8_10; do
    path="results/instruction_necessity_v1_s${seed}_t${shard}.json"
    [[ ! -e "$path" ]] || { echo "Refusing to overwrite: $path" >&2; exit 3; }
  done
done

preflight_job=$(sbatch --parsable \
  --job-name=xvla-instruction-necessity-v1-preflight \
  athena/slurm_preflight_instruction_necessity.sbatch \
  --cache "$cache" \
  --provenance-result "$provenance" \
  --failed-local-summary "$failed_local_summary" \
  --output "$manifest")

smoke_job=$(sbatch --parsable \
  --dependency="afterok:${preflight_job}" \
  --job-name=xvla-instruction-necessity-v1-smoke \
  athena/slurm_instruction_necessity.sbatch \
  --mode strict_smoke \
  --checkpoint-seed 0 \
  --checkpoint "${checkpoints[0]}" \
  --cache "$cache" \
  --provenance-result "$provenance" \
  --manifest "$manifest" \
  --task-start 0 --task-end 10 \
  --episode-start 40 --eps-per-task 1 \
  --expected-rollouts 0 \
  --res 64 --horizon 8 --exec-h 8 --max-steps 280 --num-steps-wait 10 \
  --matmul-precision highest \
  --output "$smoke_result")

declare -a full_jobs=()
declare -a result_args=()
index=0
for seed in 0 1 2; do
  for bounds in "0 2" "2 4" "4 6" "6 8" "8 10"; do
    read -r task_start task_end <<< "$bounds"
    result="results/instruction_necessity_v1_s${seed}_t${task_start}_${task_end}.json"
    full_jobs[$index]=$(sbatch --parsable \
      --dependency="afterok:${smoke_job}" \
      --job-name="xvla-instruction-necessity-v1-s${seed}-t${task_start}-${task_end}" \
      athena/slurm_instruction_necessity.sbatch \
      --mode full \
      --checkpoint-seed "$seed" \
      --checkpoint "${checkpoints[$seed]}" \
      --cache "$cache" \
      --provenance-result "$provenance" \
      --manifest "$manifest" \
      --task-start "$task_start" --task-end "$task_end" \
      --episode-start 40 --eps-per-task 10 \
      --expected-rollouts 60 \
      --res 64 --horizon 8 --exec-h 8 --max-steps 280 --num-steps-wait 10 \
      --matmul-precision highest \
      --output "$result")
    result_args+=(--result "$result")
    ((index += 1))
  done
done

dependency="afterok:$(IFS=:; echo "${full_jobs[*]}")"
summary_job=$(sbatch --parsable \
  --dependency="$dependency" \
  --job-name=xvla-instruction-necessity-v1-summary \
  athena/slurm_summarize_instruction_necessity.sbatch \
  --smoke "$smoke_result" \
  "${result_args[@]}" \
  --manifest "$manifest" \
  --output "$summary_result")

echo "failed_local_summary $failed_local_job $state"
echo "preflight $preflight_job"
echo "smoke $smoke_job"
for job_index in "${!full_jobs[@]}"; do echo "full_${job_index} ${full_jobs[$job_index]}"; done
echo "summary $summary_job"
