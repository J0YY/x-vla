#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

manifest=artifacts/libero_all_manifest.json
manifest_sha256=d0183b465c4d687a4b31f1fc2ca8c75a35786e164cd07f638ca47e347e21f4fd
trainer_sha256=9b63cce90d6c99a3e4f9ecef1db2914352e0221f57ed0bbd3e159eb6c1534f98
evaluator_sha256=91ae342892e32b0aa019a43ff06b5809dca2a49d4be20ce9a55fbad1b15cdbb3
summary_sha256=cd87e66f267472d9b5c934804958f802e176b17618fca7f3be463db4a441a865
train_wrapper_sha256=e4f32dea9944ae8a4e4339602fd5cc12120b00d1c5d68991a8f059ed9757e659
eval_wrapper_sha256=70c4afc8b501cd1b7e314a31ce972ea2991c3cb67c20a237759b85a5ac4dfce2
eval_smoke_wrapper_sha256=fa1e0f02393520827258d4ef7e47044b0658a70888c50290975fa3cc6542d42e
summary_wrapper_sha256=1acca9b74a2fb81a8f59f8db9baf6a587f55c6f1770c3a5280c98d251c8b4a6a

if [[ ! -f "$manifest" ]]; then
  echo "Missing frozen generalist manifest: $manifest" >&2
  exit 3
fi
observed_manifest_sha256=$(sha256sum "$manifest" | awk '{print $1}')
if [[ "$observed_manifest_sha256" != "$manifest_sha256" ]]; then
  echo "Frozen generalist manifest SHA-256 mismatch" >&2
  exit 3
fi

declare -A frozen_sources=(
  [athena/train_multisuite_checkpoint.py]="$trainer_sha256"
  [athena/run_xvla_experiment.py]="$evaluator_sha256"
  [athena/summarize_multisuite_generalist.py]="$summary_sha256"
  [athena/slurm_train_multisuite_hardened.sbatch]="$train_wrapper_sha256"
  [athena/slurm_eval_multisuite_checkpoint_hardened.sbatch]="$eval_wrapper_sha256"
  [athena/slurm_eval_multisuite_smoke_hardened.sbatch]="$eval_smoke_wrapper_sha256"
  [athena/slurm_multisuite_summary_hardened.sbatch]="$summary_wrapper_sha256"
)
for source in "${!frozen_sources[@]}"; do
  observed=$(sha256sum "$source" | awk '{print $1}')
  if [[ "$observed" != "${frozen_sources[$source]}" ]]; then
    echo "Frozen source SHA-256 mismatch: $source" >&2
    exit 3
  fi
done

python3 -c \
  'from pathlib import Path; from athena.summarize_multisuite_generalist import validate_cache_provenance; validate_cache_provenance(Path("results"))'

declare -a protected_outputs=(results/generalist_hardened_summary.json)
for architecture in chi conventional; do
  protected_outputs+=(
    "artifacts/ckpt_generalist_hardened_${architecture}_smoke.pt"
    "artifacts/ckpt_generalist_hardened_${architecture}_smoke.json"
    "results/train_generalist_hardened_${architecture}_smoke.json"
    "results/generalist_hardened_${architecture}_eval_smoke.json"
  )
  for seed in 0 1 2; do
    protected_outputs+=(
      "artifacts/ckpt_generalist_hardened_${architecture}_s${seed}.pt"
      "artifacts/ckpt_generalist_hardened_${architecture}_s${seed}.json"
      "artifacts/recovery_generalist_hardened_${architecture}_s${seed}.pt"
      "results/train_generalist_hardened_${architecture}_s${seed}.json"
    )
    for suite in libero_object libero_spatial libero_goal libero_10; do
      for range in "0 3" "3 6" "6 8" "8 10"; do
        read -r start end <<<"$range"
        protected_outputs+=(
          "results/generalist_hardened_${architecture}_s${seed}_${suite}_t${start}_${end}.json"
        )
      done
    done
  done
done
for output in "${protected_outputs[@]}"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to reuse hardened output: $output" >&2
    exit 3
  fi
done

declare -A evaluator_smoke_jobs
declare -a evaluation_jobs=()
for architecture in chi conventional; do
  smoke_checkpoint="artifacts/ckpt_generalist_hardened_${architecture}_smoke.pt"
  smoke_metadata="artifacts/ckpt_generalist_hardened_${architecture}_smoke.json"
  smoke_train_job=$(sbatch --parsable \
    --job-name="xvla-hardened-generalist-${architecture}-smoke" \
    athena/slurm_train_multisuite_hardened.sbatch \
    --architecture "$architecture" \
    --vision-encoder vit \
    --manifest "$manifest" \
    --checkpoint-output "$smoke_checkpoint" \
    --metadata-output "$smoke_metadata" \
    --result-output "results/train_generalist_hardened_${architecture}_smoke.json" \
    --seed 0 \
    --steps 10 \
    --batch-size 256 \
    --lr 8e-4 \
    --ema-decay 0.999)
  evaluator_smoke_jobs[$architecture]=$(sbatch --parsable \
    --dependency="afterok:${smoke_train_job}" \
    --constraint=a6000 \
    --job-name="xvla-hardened-generalist-${architecture}-eval-smoke" \
    athena/slurm_eval_multisuite_smoke_hardened.sbatch \
    "$architecture" "$smoke_checkpoint" "$smoke_metadata")
  echo "$architecture hardened smoke jobs: $smoke_train_job ${evaluator_smoke_jobs[$architecture]}"
done

for architecture in chi conventional; do
  for seed in 0 1 2; do
    checkpoint="artifacts/ckpt_generalist_hardened_${architecture}_s${seed}.pt"
    metadata="artifacts/ckpt_generalist_hardened_${architecture}_s${seed}.json"
    train_job=$(sbatch --parsable \
      --dependency="afterok:${evaluator_smoke_jobs[$architecture]}" \
      --job-name="xvla-hardened-generalist-${architecture}-s${seed}" \
      athena/slurm_train_multisuite_hardened.sbatch \
      --architecture "$architecture" \
      --vision-encoder vit \
      --manifest "$manifest" \
      --checkpoint-output "$checkpoint" \
      --metadata-output "$metadata" \
      --result-output "results/train_generalist_hardened_${architecture}_s${seed}.json" \
      --seed "$seed" \
      --steps 160000 \
      --batch-size 256 \
      --lr 8e-4 \
      --ema-decay 0.999 \
      --recovery-output "artifacts/recovery_generalist_hardened_${architecture}_s${seed}.pt" \
      --recovery-interval 10000)
    evaluation_job=$(sbatch --parsable \
      --dependency="afterok:${train_job}" \
      --constraint=a6000 \
      --job-name="xvla-hardened-generalist-${architecture}-s${seed}-all" \
      athena/slurm_eval_multisuite_checkpoint_hardened.sbatch \
      "$architecture" "$seed" "$checkpoint" "$metadata")
    evaluation_jobs+=("$evaluation_job")
    echo "$architecture seed $seed hardened jobs: $train_job $evaluation_job"
  done
done

evaluation_dependency=$(IFS=:; echo "${evaluation_jobs[*]}")
summary_job=$(sbatch --parsable \
  --dependency="afterok:${evaluation_dependency}" \
  --job-name=xvla-hardened-generalist-summary \
  athena/slurm_multisuite_summary_hardened.sbatch \
  --results-dir results \
  --artifacts-dir artifacts \
  --output results/generalist_hardened_summary.json)

echo "Hardened evaluation jobs: ${evaluation_jobs[*]}"
echo "Hardened summary job: $summary_job"
