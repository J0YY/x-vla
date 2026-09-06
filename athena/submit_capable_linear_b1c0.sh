#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "b1c0 submission accepts no arguments" >&2
  exit 2
fi
readonly stage_root=/work/joy/x-vla-capable-linear-b1c0-odt-v2
readonly stage_ledger="$stage_root/athena/capable_linear_b1c0_stage.sha256"
cd "$stage_root"
verify_stage() {
  test -f "$stage_ledger"
  test ! -L "$stage_ledger"
  sha256sum --check --strict --quiet "$stage_ledger"
}
trap verify_stage EXIT
verify_stage
mkdir athena/results/capable_linear_b1c0/submission.lock

submit_job() {
  local value
  value=$(sbatch --parsable --kill-on-invalid-dep=yes "$@")
  value=${value%%;*}
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "sbatch returned a nonnumeric job id: $value" >&2
    exit 1
  fi
  printf '%s\n' "$value"
}

preflight_job=$(submit_job athena/slurm_capable_linear_b1c0_preflight.sbatch)
unit_job=$(submit_job athena/slurm_capable_linear_b1c0_unit.sbatch)
smoke_job=$(submit_job athena/slurm_capable_linear_b1c0_eval_smoke.sbatch)
full_job=$(submit_job \
  --dependency="afterok:${preflight_job}:${unit_job}" \
  athena/slurm_capable_linear_b1c0_full.sbatch)
eval_0_3=$(submit_job --job-name=xvla-b1c0-eval-0-3 \
  --dependency="afterok:${unit_job}:${smoke_job}" \
  athena/slurm_capable_linear_b1c0_eval.sbatch 0 3)
eval_3_6=$(submit_job --job-name=xvla-b1c0-eval-3-6 \
  --dependency="afterok:${unit_job}:${smoke_job}" \
  athena/slurm_capable_linear_b1c0_eval.sbatch 3 6)
eval_6_8=$(submit_job --job-name=xvla-b1c0-eval-6-8 \
  --dependency="afterok:${unit_job}:${smoke_job}" \
  athena/slurm_capable_linear_b1c0_eval.sbatch 6 8)
eval_8_10=$(submit_job --job-name=xvla-b1c0-eval-8-10 \
  --dependency="afterok:${unit_job}:${smoke_job}" \
  athena/slurm_capable_linear_b1c0_eval.sbatch 8 10)
aggregate_job=$(submit_job \
  --dependency="afterok:${eval_0_3}:${eval_3_6}:${eval_6_8}:${eval_8_10}" \
  athena/slurm_capable_linear_b1c0_aggregate.sbatch)
composite_job=$(submit_job --dependency="afterok:${full_job}:${aggregate_job}" \
  athena/slurm_capable_linear_b1c0_composite.sbatch)
sha256sum --check --strict --quiet "$stage_ledger"
printf '%s\n' \
  "full_job=$full_job" \
  "preflight_job=$preflight_job" \
  "unit_job=$unit_job" \
  "smoke_job=$smoke_job" \
  "eval_0_3=$eval_0_3" \
  "eval_3_6=$eval_3_6" \
  "eval_6_8=$eval_6_8" \
  "eval_8_10=$eval_8_10" \
  "aggregate_job=$aggregate_job" \
  "composite_job=$composite_job"
