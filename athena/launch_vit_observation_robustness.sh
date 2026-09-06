#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

readonly cache=artifacts/libero_frames_100000_64.pkl
readonly cache_sha=053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662
readonly provenance=results/cache_provenance_libero_object.json
readonly provenance_sha=1e3ed7eaeef317a221bb6649ea75ef68ff924b57797682e853b4bd90a138f4c4
readonly manifest=artifacts/vit_capability/manifest.json
readonly manifest_sha=c23a9ba8bf791d55b0a731743fdac7810395d0030fdd6225ee98b7ec6da42453

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

verify_sha256() {
  local path=$1
  local expected=$2
  [[ -f "$path" ]] || {
    echo "Missing frozen input: $path" >&2
    exit 3
  }
  local observed
  observed=$(sha256sum "$path" | cut -d' ' -f1)
  [[ "$observed" == "$expected" ]] || {
    echo "Frozen input SHA-256 mismatch: $path" >&2
    exit 3
  }
}

verify_sha256 "$cache" "$cache_sha"
verify_sha256 "$provenance" "$provenance_sha"
verify_sha256 "$manifest" "$manifest_sha"
for seed in 0 1 2; do
  verify_sha256 "${checkpoints[$seed]}" "${checkpoint_shas[$seed]}"
done

declare -a outputs=(
  results/vit_observation_robustness_v1_smoke.json
  results/vit_observation_robustness_v1_summary.json
)
for seed in 0 1 2; do
  for range in "0 3" "3 6" "6 8" "8 10"; do
    read -r start end <<<"$range"
    outputs+=("results/vit_observation_robustness_v1_s${seed}_t${start}_${end}.json")
  done
done
for output in "${outputs[@]}"; do
  if [[ -e "$output" || -e "${output}.tmp" ]]; then
    echo "Refusing to overwrite robustness output: $output" >&2
    exit 3
  fi
done

gpu_submit=(
  sbatch --parsable
  --partition=ddp-2way
  --qos=combined12gpus
  --constraint=a6000
)

smoke_job=$("${gpu_submit[@]}" \
  --job-name=xvla-vit-obs-robust-v1-smoke \
  athena/slurm_vit_observation_robustness.sbatch \
  --checkpoint "${checkpoints[0]}" \
  --expected-checkpoint-sha256 "${checkpoint_shas[0]}" \
  --checkpoint-seed 0 \
  --cache "$cache" \
  --provenance-result "$provenance" \
  --capability-manifest "$manifest" \
  --task-start 0 \
  --task-end 3 \
  --mode strict_smoke \
  --output results/vit_observation_robustness_v1_smoke.json)

declare -a full_jobs=()
declare -a result_args=()
for seed in 0 1 2; do
  for range in "0 3" "3 6" "6 8" "8 10"; do
    read -r start end <<<"$range"
    output="results/vit_observation_robustness_v1_s${seed}_t${start}_${end}.json"
    job=$("${gpu_submit[@]}" \
      --dependency="afterok:${smoke_job}" \
      --job-name="xvla-vit-obs-robust-v1-s${seed}-${start}${end}" \
      athena/slurm_vit_observation_robustness.sbatch \
      --checkpoint "${checkpoints[$seed]}" \
      --expected-checkpoint-sha256 "${checkpoint_shas[$seed]}" \
      --checkpoint-seed "$seed" \
      --cache "$cache" \
      --provenance-result "$provenance" \
      --capability-manifest "$manifest" \
      --task-start "$start" \
      --task-end "$end" \
      --mode full \
      --output "$output")
    full_jobs+=("$job")
    result_args+=(--result "$output")
  done
done

dependency=$(IFS=:; echo "${full_jobs[*]}")
summary_job=$(sbatch --parsable \
  --dependency="afterok:${dependency}" \
  --job-name=xvla-vit-obs-robust-v1-summary \
  athena/slurm_summarize_vit_observation_robustness.sbatch \
  "${result_args[@]}" \
  --checkpoint "${checkpoints[0]}" \
  --checkpoint "${checkpoints[1]}" \
  --checkpoint "${checkpoints[2]}" \
  --cache "$cache" \
  --provenance-result "$provenance" \
  --capability-manifest "$manifest" \
  --output results/vit_observation_robustness_v1_summary.json)

echo "smoke $smoke_job"
echo "full ${full_jobs[*]}"
echo "summary $summary_job"
