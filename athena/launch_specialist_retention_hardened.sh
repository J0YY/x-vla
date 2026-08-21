#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

protocol_sha256=0bee62f9296496c4ee0943a0b74860c927c69d436bac468c7dd947500b7700eb
trainer_sha256=95e086b739359ba0cc653e3de13199c031ac22414bd217ff08113ea3c3129c27
evaluator_sha256=fd28d56acb9e198c1dbcd207c8614d32a81d9ac50f3de3016fe9a2817bd19939
summary_sha256=f5414d7a3348103c7fc27b5c58abcc4413b7f6df3a238982ebcddac63a3bdc3a
train_wrapper_sha256=41f7a1d700d60184c8640a064e10dacd67951dc7b85cd2bcae164986369408f1
eval_wrapper_sha256=61e613f4194d4d5b80eaa01f1434646e15140f49b36bcf0e11025facdaf16743
summary_wrapper_sha256=58b609f20af54c852f4881394dc93872c7d39349dd3e6a7472e057d33998d5b1

declare -A frozen_sources=(
  [athena/specialist_retention_protocol.py]="$protocol_sha256"
  [athena/train_specialist_retention_hardened.py]="$trainer_sha256"
  [athena/eval_specialist_retention_hardened.py]="$evaluator_sha256"
  [athena/summarize_specialist_retention_hardened.py]="$summary_sha256"
  [athena/slurm_train_specialist_retention_hardened.sbatch]="$train_wrapper_sha256"
  [athena/slurm_eval_specialist_retention_hardened.sbatch]="$eval_wrapper_sha256"
  [athena/slurm_summarize_specialist_retention_hardened.sbatch]="$summary_wrapper_sha256"
)
for source in "${!frozen_sources[@]}"; do
  observed=$(sha256sum "$source" | awk '{print $1}')
  if [[ "$observed" != "${frozen_sources[$source]}" ]]; then
    echo "Frozen specialist source SHA-256 mismatch: $source" >&2
    exit 3
  fi
done

python3 - <<'PY'
from athena.specialist_retention_protocol import (
    MODEL_SOURCE_HASHES,
    SUITES,
    source_snapshot,
    validate_cache_and_provenance,
    validate_source_snapshot,
)
validate_source_snapshot(source_snapshot(MODEL_SOURCE_HASHES), MODEL_SOURCE_HASHES)
for suite in SUITES:
    validate_cache_and_provenance(suite)
PY

generalist_job=${GENERALIST_CHI_S0_EVAL_JOB:-}
if [[ -n "$generalist_job" ]]; then
  if [[ ! "$generalist_job" =~ ^[0-9]+$ ]]; then
    echo "GENERALIST_CHI_S0_EVAL_JOB must be one numeric Slurm job ID" >&2
    exit 3
  fi
  if ! scontrol show job "$generalist_job" >/dev/null 2>&1; then
    echo "GENERALIST_CHI_S0_EVAL_JOB is not visible to Slurm: $generalist_job" >&2
    exit 3
  fi
else
  generalist_required=(
    artifacts/ckpt_generalist_hardened_chi_s0.pt
    artifacts/ckpt_generalist_hardened_chi_s0.json
  )
  for suite in libero_object libero_spatial libero_goal libero_10; do
    for range in "0 3" "3 6" "6 8" "8 10"; do
      read -r start end <<<"$range"
      generalist_required+=(
        "results/generalist_hardened_chi_s0_${suite}_t${start}_${end}.json"
      )
    done
  done
  for path in "${generalist_required[@]}"; do
    if [[ ! -f "$path" ]]; then
      echo "Generalist seed-0 is incomplete. Set GENERALIST_CHI_S0_EVAL_JOB." >&2
      exit 3
    fi
  done
fi

declare -a protected_outputs=(results/specialist_retention_hardened_summary.json)
for suite in libero_object libero_spatial libero_goal libero_10; do
  protected_outputs+=(
    "artifacts/ckpt_specialist_retention_hardened_${suite}_chi_s0_smoke.pt"
    "artifacts/ckpt_specialist_retention_hardened_${suite}_chi_s0_smoke.json"
    "results/train_specialist_retention_hardened_${suite}_chi_s0_smoke.json"
    "results/specialist_retention_hardened_${suite}_chi_s0_eval_smoke.json"
    "artifacts/ckpt_specialist_retention_hardened_${suite}_chi_s0.pt"
    "artifacts/ckpt_specialist_retention_hardened_${suite}_chi_s0.json"
    "results/train_specialist_retention_hardened_${suite}_chi_s0.json"
  )
  for range in "0 3" "3 6" "6 8" "8 10"; do
    read -r start end <<<"$range"
    protected_outputs+=(
      "results/specialist_retention_hardened_${suite}_chi_s0_t${start}_${end}.json"
    )
  done
done
for output in "${protected_outputs[@]}"; do
  if [[ -e "$output" || -e "${output}.tmp" ]]; then
    echo "Refusing to reuse hardened specialist output: $output" >&2
    exit 3
  fi
done

# Replace only the unpinned seed-0 Object lineage and the dead retention graph.
# Corrected Object seeds 1--2 and conventional seeds 0--2 remain useful controls.
legacy_jobs=()
for job in $(seq 830893 830897) $(seq 830971 830983); do
  legacy_jobs+=("$job")
done
legacy_csv=$(IFS=,; echo "${legacy_jobs[*]}")
active_legacy=$(squeue -h -j "$legacy_csv" -o '%A' | sort -u || true)
if [[ -n "$active_legacy" ]]; then
  while read -r job; do
    scancel "$job"
    echo "Canceled superseded specialist job: $job"
  done <<<"$active_legacy"
fi

declare -a smoke_eval_jobs=()
for suite in libero_object libero_spatial libero_goal libero_10; do
  train_smoke=$(sbatch --parsable \
    --job-name="xvla-spec-hard-${suite}-train-smoke" \
    athena/slurm_train_specialist_retention_hardened.sbatch \
    --suite "$suite" \
    --mode smoke)
  eval_smoke=$(sbatch --parsable \
    --dependency="afterok:${train_smoke}" \
    --constraint=a6000 \
    --job-name="xvla-spec-hard-${suite}-eval-smoke" \
    athena/slurm_eval_specialist_retention_hardened.sbatch \
    --suite "$suite" \
    --mode smoke)
  smoke_eval_jobs+=("$eval_smoke")
  echo "$suite hardened smoke jobs: $train_smoke $eval_smoke"
done

all_smokes=$(IFS=:; echo "${smoke_eval_jobs[*]}")
declare -a full_eval_jobs=()
for suite in libero_object libero_spatial libero_goal libero_10; do
  train_job=$(sbatch --parsable \
    --dependency="afterok:${all_smokes}" \
    --job-name="xvla-spec-hard-${suite}-train" \
    athena/slurm_train_specialist_retention_hardened.sbatch \
    --suite "$suite" \
    --mode full)
  echo "$suite hardened full trainer: $train_job"
  for range in "0 3" "3 6" "6 8" "8 10"; do
    read -r start end <<<"$range"
    eval_job=$(sbatch --parsable \
      --dependency="afterok:${train_job}" \
      --constraint=a6000 \
      --job-name="xvla-spec-hard-${suite}-${start}${end}" \
      athena/slurm_eval_specialist_retention_hardened.sbatch \
      --suite "$suite" \
      --mode full \
      --task-start "$start" \
      --task-end "$end")
    full_eval_jobs+=("$eval_job")
  done
done

summary_dependencies=("${full_eval_jobs[@]}")
if [[ -n "$generalist_job" ]]; then
  summary_dependencies+=("$generalist_job")
fi
summary_dependency=$(IFS=:; echo "${summary_dependencies[*]}")
summary_job=$(sbatch --parsable \
  --dependency="afterok:${summary_dependency}" \
  --job-name=xvla-specialist-retention-hardened-summary \
  athena/slurm_summarize_specialist_retention_hardened.sbatch \
  --results-dir results \
  --artifacts-dir artifacts \
  --output results/specialist_retention_hardened_summary.json)

echo "Hardened specialist evaluation jobs: ${full_eval_jobs[*]}"
echo "Hardened specialist-retention summary job: $summary_job"
