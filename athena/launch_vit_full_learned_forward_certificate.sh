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

declare -A frozen_sources=(
  [athena/run_vit_full_learned_forward_certificate.py]=7bdf83530accb3a705f412fee7b89a9b0430ffc693d1a6264f528220edadc012
  [athena/summarize_vit_full_learned_forward_certificate.py]=d069f8ea2272d3659bf96837136653ec84ee30616623df9a544a9893d7c071b0
  [athena/slurm_vit_full_learned_forward_certificate.sbatch]=e25f8001335545dab917268c44937e4d9450bbcd58aea2d3bab067e0fc3cc4ce
  [athena/slurm_summarize_vit_full_learned_forward_certificate.sbatch]=ffcd875af8a6c97ab798458dd330e63a4afa2fd171f3947279216198e6762f4a
  [athena/VIT_FULL_LEARNED_FORWARD_PROTOCOL.md]=a391df9d1cf4f8faaea142a8d8d79f1447a44b359c8e94b18ae1f10e80edaec9
  [athena/VIT_FULL_LEARNED_FORWARD_SOURCE_BUNDLE.sha256]=a02628e818eb2afe17f0bd751e6fed6e3ba9e5479d77ca980badd52e5fe206c1
  [athena/run_vit_modality_contribution_audit.py]=81d46fe4a4f7ea23d5cfb159e41e001d352468982ec3f698aeee8aa036d11be4
  [athena/run_conv_joint_block_certificate.py]=0ac1550a97a6667485211bd5de52237b3a877087e8014ab79c8b900891096699
  [athena/run_conv_joint_ffn_certificate.py]=d7e78469e87146e5edaf56b6e91857cd6ea8e1dfcfdde736cc10ce0774be5df4
  [athena/run_conv_joint_attention_certificate.py]=1e838bfb61cb16a946a48f9c75ed0291008d9c1a87eb953444f949998a688325
  [athena/libero_dataset_metadata.py]=3e14b117ee72b010939c0fdd2c20417778b89a8691156553a0bfb2ad2d0b4edb
  [athena/run_xvla_experiment.py]=91ae342892e32b0aa019a43ff06b5809dca2a49d4be20ce9a55fbad1b15cdbb3
  [xvla/models/vla.py]=bc276b0328b53cf1f54b42bcd75137df5785cb8625f51d2b09671081d7c4275f
  [xvla/models/vit.py]=111049ad4c24179e294da8cccb732e3a416223f77b42a08ed56291febe4ddf87
  [xvla/nn/attention.py]=4f8e49d9dc25ee292b34daf60687f80f38a8f85548ef2905b3217e66dc8c27ec
  [xvla/nn/baselines.py]=eb4d6ba5f3aa66589991ee3221017dc59092ec9fd35255f9124812420451a8d4
  [xvla/nn/bilinear.py]=162929529750fe719c6b4199ae316ddb45f72f52de7bbbd07d329dfb5f8287c8
  [xvla/nn/block.py]=7555c0b7b11c2592c97bf90ec7eeca6ecd0b76f58960df78eefa984386eea969
  [xvla/nn/flow_action.py]=1108fd31c0091ce51afa342e798ab625514bc5a8064e45b8f0ce38790d71c1aa
  [xvla/nn/homogeneous.py]=314ba488c638da18d74de213218e310ea75aafa4fbbc92d9cb838aaa5a31701b
  [xvla/nn/normalization.py]=67406ca22083c28223023ce1efcfc448470654da8477a535bf4a5284cf7338a9
  [xvla/nn/product_routing.py]=20da357c875e426b2341e1950736ac6cb88700cb1654e46a4b8a2943c666c9b2
  [xvla/nn/projector.py]=6212e35fbab973de2e79d74bd5064635db12a2c5e48c8c7df7428cbf1e31aaed
  [xvla/nn/quantile_action.py]=6bd1ecbe5491457fb9ae0154ea566c0ce24e43f102516699f0b3941d4b497c7e
  [xvla/train/exact_odt_attention_proto.py]=048e094f384ef6064cbd02c6d5668024a49ed8a218d4c318f7846e66d47e3238
  [athena/results/exact_attention_frozen_sources/xvla_models_vla.py.b64]=b1f1309182bee516882962415d0bf358adb63a8f50e5a9fe2757f448b82527b6
  [athena/results/exact_attention_frozen_sources/xvla_models_vit.py.b64]=1c3a6bc32d8580a13770ed1bcb273a20c03b747b58d60d05bd83696e5e1729c5
  [athena/results/exact_attention_frozen_sources/xvla_nn_product_routing.py.b64]=e2cc8b13001a5f2cdcd79f6aa135cb50453d55da13024cd1dd0a22f0e820c6d3
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
    echo "Frozen SHA mismatch: $path, got $actual" >&2
    exit 3
  fi
}

provenance_state=$(sacct -j "$provenance_job" --starttime 2026-08-20 \
  --noheader --parsable2 --allocations --format=State | head -1 | cut -d'|' -f1)
if [[ "$provenance_state" != "COMPLETED" ]]; then
  echo "Refusing to launch: Object provenance job $provenance_job is '$provenance_state'" >&2
  exit 4
fi
verify_sha256 "$provenance_result" 1e3ed7eaeef317a221bb6649ea75ef68ff924b57797682e853b4bd90a138f4c4
verify_sha256 "$cache" "$cache_sha"
verify_sha256 "$capability_manifest" "$capability_manifest_sha"
for seed in 0 1 2; do
  verify_sha256 "${checkpoints[$seed]}" "${checkpoint_shas[$seed]}"
done
for path in "${!frozen_sources[@]}"; do
  verify_sha256 "$path" "${frozen_sources[$path]}"
done

for output in \
  results/vit_full_learned_forward_v1_smoke_ddp2_a6000.json \
  results/vit_full_learned_forward_v1_s0.json \
  results/vit_full_learned_forward_v1_s1.json \
  results/vit_full_learned_forward_v1_s2.json \
  results/vit_full_learned_forward_v1_summary.json; do
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing result: $output" >&2
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
  --job-name=xvla-vit-full-forward-v1-smoke-a6000 \
  athena/slurm_vit_full_learned_forward_certificate.sbatch \
  --checkpoint "${checkpoints[0]}" \
  --expected-checkpoint-sha256 "${checkpoint_shas[0]}" \
  --checkpoint-seed 0 \
  --cache "$cache" \
  --expected-cache-sha256 "$cache_sha" \
  --provenance-result "$provenance_result" \
  --expected-provenance-job-id "$provenance_job" \
  --capability-manifest "$capability_manifest" \
  --output results/vit_full_learned_forward_v1_smoke_ddp2_a6000.json \
  --mode strict_smoke)

declare -a full_jobs=()
for seed in 0 1 2; do
  full_jobs[$seed]=$("${gpu_submit[@]}" \
    --dependency="afterok:${smoke_job}" \
    --job-name="xvla-vit-full-forward-v1-s${seed}-a6000" \
    athena/slurm_vit_full_learned_forward_certificate.sbatch \
    --checkpoint "${checkpoints[$seed]}" \
    --expected-checkpoint-sha256 "${checkpoint_shas[$seed]}" \
    --checkpoint-seed "$seed" \
    --cache "$cache" \
    --expected-cache-sha256 "$cache_sha" \
    --provenance-result "$provenance_result" \
    --expected-provenance-job-id "$provenance_job" \
    --capability-manifest "$capability_manifest" \
    --output "results/vit_full_learned_forward_v1_s${seed}.json" \
    --mode full)
done

dependency="afterok:${full_jobs[0]}:${full_jobs[1]}:${full_jobs[2]}"
summary_job=$(sbatch --parsable \
  --dependency="$dependency" \
  --job-name=xvla-vit-full-forward-v1-summary \
  athena/slurm_summarize_vit_full_learned_forward_certificate.sbatch \
  --result results/vit_full_learned_forward_v1_s0.json \
  --result results/vit_full_learned_forward_v1_s1.json \
  --result results/vit_full_learned_forward_v1_s2.json \
  --checkpoint "${checkpoints[0]}" \
  --checkpoint "${checkpoints[1]}" \
  --checkpoint "${checkpoints[2]}" \
  --cache "$cache" \
  --provenance-result "$provenance_result" \
  --capability-manifest "$capability_manifest" \
  --output results/vit_full_learned_forward_v1_summary.json)

echo "provenance $provenance_job $provenance_state"
echo "smoke $smoke_job"
echo "full_s0 ${full_jobs[0]}"
echo "full_s1 ${full_jobs[1]}"
echo "full_s2 ${full_jobs[2]}"
echo "summary $summary_job"
