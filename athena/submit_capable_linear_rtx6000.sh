#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "RTX 6000 submission accepts no arguments" >&2
  exit 2
fi
readonly stage_root=/work/joy/x-vla-capable-linear-rtx6000-v3
readonly stage_ledger="$stage_root/athena/capable_linear_rtx6000_stage.sha256"
readonly b1c_full_job=835410
cd "$stage_root"
verify_stage() {
  test -f "$stage_ledger"
  test ! -L "$stage_ledger"
  sha256sum --check --strict --quiet "$stage_ledger"
}
trap verify_stage EXIT
verify_stage
job_record=$(scontrol show job --oneliner "$b1c_full_job")
if [[ "$job_record" != *"JobName=xvla-b1c0-odt-full"* || \
      "$job_record" != *"WorkDir=/work/joy/x-vla-capable-linear-b1c0-odt-v2"* ]]; then
  echo "job 835410 is not the frozen b1c full run" >&2
  exit 1
fi
mkdir athena/results/capable_linear_rtx6000/submission.lock

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

unit_job=$(submit_job athena/slurm_capable_linear_rtx6000_unit.sbatch)
smoke_job=$(submit_job --dependency="afterok:${unit_job}" \
  athena/slurm_capable_linear_rtx6000_smoke.sbatch)
eval_0_3=$(submit_job --job-name=xvla-rtx6k-eval-0-3 \
  --dependency="afterok:${unit_job}:${smoke_job}" \
  athena/slurm_capable_linear_rtx6000_eval.sbatch 0 3)
eval_3_6=$(submit_job --job-name=xvla-rtx6k-eval-3-6 \
  --dependency="afterok:${unit_job}:${smoke_job}" \
  athena/slurm_capable_linear_rtx6000_eval.sbatch 3 6)
eval_6_8=$(submit_job --job-name=xvla-rtx6k-eval-6-8 \
  --dependency="afterok:${unit_job}:${smoke_job}" \
  athena/slurm_capable_linear_rtx6000_eval.sbatch 6 8)
eval_8_10=$(submit_job --job-name=xvla-rtx6k-eval-8-10 \
  --dependency="afterok:${unit_job}:${smoke_job}" \
  athena/slurm_capable_linear_rtx6000_eval.sbatch 8 10)
aggregate_job=$(submit_job \
  --dependency="afterok:${eval_0_3}:${eval_3_6}:${eval_6_8}:${eval_8_10}" \
  athena/slurm_capable_linear_rtx6000_aggregate.sbatch)
composite_job=$(submit_job \
  --dependency="afterok:${aggregate_job}:${b1c_full_job}" \
  athena/slurm_capable_linear_rtx6000_composite.sbatch)
verify_stage
printf '%s\n' \
  "b1c_full_job=$b1c_full_job" \
  "unit_job=$unit_job" \
  "smoke_job=$smoke_job" \
  "eval_0_3=$eval_0_3" \
  "eval_3_6=$eval_3_6" \
  "eval_6_8=$eval_6_8" \
  "eval_8_10=$eval_8_10" \
  "aggregate_job=$aggregate_job" \
  "composite_job=$composite_job"
