#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

readonly provenance_job=830988
readonly provenance_result=results/cache_provenance_libero_object.json
readonly cache=artifacts/libero_frames_100000_64.pkl
readonly cache_sha=053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662
readonly capability_manifest=artifacts/vit_capability/manifest.json
readonly capability_manifest_sha=c23a9ba8bf791d55b0a731743fdac7810395d0030fdd6225ee98b7ec6da42453

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
  if [[ ! -f "$path" ]]; then
    echo "Missing frozen input: $path" >&2
    exit 3
  fi
  local actual
  actual=$(sha256sum "$path" | cut -d' ' -f1)
  if [[ "$actual" != "$expected" ]]; then
    echo "Frozen input SHA mismatch: $path" >&2
    exit 3
  fi
}

provenance_state=$(sacct -j "$provenance_job" --starttime 2026-08-20 \
  --noheader --parsable2 --allocations --format=State | head -1 | cut -d'|' -f1)
case "$provenance_state" in
  COMPLETED)
    [[ -s "$provenance_result" ]] || {
      echo "Completed provenance job has no result" >&2
      exit 4
    }
    ;;
  RUNNING|PENDING|CONFIGURING|COMPLETING)
    ;;
  *)
    echo "Refusing to launch: Object provenance job $provenance_job is '$provenance_state'" >&2
    exit 4
    ;;
esac

verify_sha256 "$cache" "$cache_sha"
verify_sha256 "$capability_manifest" "$capability_manifest_sha"
for seed in 0 1 2; do
  verify_sha256 "${checkpoints[$seed]}" "${checkpoint_shas[$seed]}"
done

for output in \
  results/vit_modality_contribution_v1_smoke_ddp2_a6000.json \
  results/vit_modality_contribution_v1_s0.json \
  results/vit_modality_contribution_v1_s1.json \
  results/vit_modality_contribution_v1_s2.json \
  results/vit_modality_contribution_v1_summary.json; do
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing result: $output" >&2
    exit 3
  fi
done

# This is the tested nonpreemptive A6000 path with the shortest observed start latency.
gpu_submit=(
  sbatch --parsable
  --partition=ddp-2way
  --qos=combined12gpus
  --constraint=a6000
)

smoke_submit=("${gpu_submit[@]}")
if [[ "$provenance_state" != "COMPLETED" ]]; then
  smoke_submit+=(--dependency="afterok:${provenance_job}")
fi

smoke_job=$("${smoke_submit[@]}" \
  --job-name=xvla-vit-modality-v1-smoke-a6000 \
  athena/slurm_vit_modality_contribution_audit.sbatch \
  --checkpoint "${checkpoints[0]}" \
  --expected-checkpoint-sha256 "${checkpoint_shas[0]}" \
  --checkpoint-seed 0 \
  --cache "$cache" \
  --expected-cache-sha256 "$cache_sha" \
  --provenance-result "$provenance_result" \
  --expected-provenance-job-id "$provenance_job" \
  --capability-manifest "$capability_manifest" \
  --output results/vit_modality_contribution_v1_smoke_ddp2_a6000.json \
  --mode strict_smoke)

declare -a full_jobs=()
for seed in 0 1 2; do
  full_jobs[$seed]=$("${gpu_submit[@]}" \
    --dependency="afterok:${smoke_job}" \
    --job-name="xvla-vit-modality-v1-s${seed}-a6000" \
    athena/slurm_vit_modality_contribution_audit.sbatch \
    --checkpoint "${checkpoints[$seed]}" \
    --expected-checkpoint-sha256 "${checkpoint_shas[$seed]}" \
    --checkpoint-seed "$seed" \
    --cache "$cache" \
    --expected-cache-sha256 "$cache_sha" \
    --provenance-result "$provenance_result" \
    --expected-provenance-job-id "$provenance_job" \
    --capability-manifest "$capability_manifest" \
    --output "results/vit_modality_contribution_v1_s${seed}.json" \
    --mode full)
done

dependency="afterok:${full_jobs[0]}:${full_jobs[1]}:${full_jobs[2]}"
summary_job=$(sbatch --parsable \
  --dependency="$dependency" \
  --job-name=xvla-vit-modality-v1-summary \
  athena/slurm_summarize_vit_modality_contribution_audit.sbatch \
  --result results/vit_modality_contribution_v1_s0.json \
  --result results/vit_modality_contribution_v1_s1.json \
  --result results/vit_modality_contribution_v1_s2.json \
  --checkpoint "${checkpoints[0]}" \
  --checkpoint "${checkpoints[1]}" \
  --checkpoint "${checkpoints[2]}" \
  --cache "$cache" \
  --provenance-result "$provenance_result" \
  --capability-manifest "$capability_manifest" \
  --output results/vit_modality_contribution_v1_summary.json)

echo "provenance $provenance_job $provenance_state"
echo "smoke $smoke_job"
echo "full_s0 ${full_jobs[0]}"
echo "full_s1 ${full_jobs[1]}"
echo "full_s2 ${full_jobs[2]}"
echo "summary $summary_job"
