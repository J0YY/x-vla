#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

provenance_job=830988
provenance_result=results/cache_provenance_libero_object.json
cache=artifacts/libero_frames_100000_64.pkl
cache_sha=053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662

declare -A checkpoints=(
  [0]=artifacts/ckpt_linear_rat_conv_s0.pt
  [1]=artifacts/ckpt_linear_rat_conv_s1_matched.pt
  [2]=artifacts/ckpt_linear_rat_conv_s2_matched.pt
)
declare -A checkpoint_shas=(
  [0]=4f9f3eef4bd661934b7c66af117995f02bdfc777368f52464f03d229ccb3e72d
  [1]=9d8df0c30583222490536b21aac040b47c5f89556aa23c08265605f7c2bc8cb6
  [2]=fc2e0bfa1a2ea2737ed0af13b168cd2562e4b0e36b159d0f98f3c5038131ace5
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

verify_sha256 "$cache" "$cache_sha"
for seed in 0 1 2; do
  verify_sha256 "${checkpoints[$seed]}" "${checkpoint_shas[$seed]}"
done

for output in \
  results/conv_joint_attention_certificate_v1_smoke.json \
  results/conv_joint_attention_certificate_v1_s0.json \
  results/conv_joint_attention_certificate_v1_s1.json \
  results/conv_joint_attention_certificate_v1_s2.json \
  results/conv_joint_attention_certificate_v1_summary.json; do
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing result: $output" >&2
    exit 3
  fi
done

smoke_submit=(sbatch --parsable)
if [[ "$provenance_state" != "COMPLETED" ]]; then
  smoke_submit+=(--dependency="afterok:${provenance_job}")
fi

smoke_job=$("${smoke_submit[@]}" \
  --job-name=xvla-conv-joint-exact-v1-smoke \
  athena/slurm_conv_joint_attention_certificate.sbatch \
  --checkpoint "${checkpoints[0]}" \
  --expected-checkpoint-sha256 "${checkpoint_shas[0]}" \
  --checkpoint-seed 0 \
  --cache "$cache" \
  --expected-cache-sha256 "$cache_sha" \
  --provenance-result "$provenance_result" \
  --expected-provenance-job-id "$provenance_job" \
  --output results/conv_joint_attention_certificate_v1_smoke.json \
  --mode strict_smoke)

declare -a full_jobs=()
for seed in 0 1 2; do
  full_jobs[$seed]=$(sbatch --parsable \
    --dependency="afterok:${smoke_job}" \
    --job-name="xvla-conv-joint-exact-v1-s${seed}" \
    athena/slurm_conv_joint_attention_certificate.sbatch \
    --checkpoint "${checkpoints[$seed]}" \
    --expected-checkpoint-sha256 "${checkpoint_shas[$seed]}" \
    --checkpoint-seed "$seed" \
    --cache "$cache" \
    --expected-cache-sha256 "$cache_sha" \
    --provenance-result "$provenance_result" \
    --expected-provenance-job-id "$provenance_job" \
    --output "results/conv_joint_attention_certificate_v1_s${seed}.json" \
    --mode full)
done

dependency="afterok:${full_jobs[0]}:${full_jobs[1]}:${full_jobs[2]}"
summary_job=$(sbatch --parsable \
  --dependency="$dependency" \
  --job-name=xvla-conv-joint-exact-v1-summary \
  athena/slurm_summarize_conv_joint_attention_certificate.sbatch \
  --result results/conv_joint_attention_certificate_v1_s0.json \
  --result results/conv_joint_attention_certificate_v1_s1.json \
  --result results/conv_joint_attention_certificate_v1_s2.json \
  --cache "$cache" \
  --provenance-result "$provenance_result" \
  --output results/conv_joint_attention_certificate_v1_summary.json)

echo "provenance $provenance_job $provenance_state"
echo "smoke $smoke_job"
echo "full_s0 ${full_jobs[0]}"
echo "full_s1 ${full_jobs[1]}"
echo "full_s2 ${full_jobs[2]}"
echo "summary $summary_job"
