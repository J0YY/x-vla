#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "Dooms submission accepts no arguments" >&2
  exit 2
fi
readonly lane_root=/work/joy/x-vla-capable-linear-b1c0-dooms-v1
readonly frozen_root=/work/joy/x-vla-capable-linear-b1c0-odt-v2
readonly lane_ledger="$lane_root/stage.sha256"
readonly frozen_ledger="$frozen_root/athena/capable_linear_b1c0_stage.sha256"
verify_fixed() {
  (cd "$lane_root" && sha256sum --check --strict --quiet "$lane_ledger")
  (cd "$frozen_root" && sha256sum --check --strict --quiet "$frozen_ledger")
}
trap verify_fixed EXIT
verify_fixed
mkdir "$lane_root/results/submission.lock"

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

unit_job=$(submit_job "$lane_root/slurm_capable_linear_b1c0_dooms_unit_v1.sbatch")
full_job=$(submit_job --dependency="afterok:${unit_job}" \
  "$lane_root/slurm_capable_linear_b1c0_dooms_full_v1.sbatch")
smoke_job=$(submit_job --job-name=xvla-dooms-smoke \
  --dependency="afterok:${full_job}" \
  "$lane_root/slurm_capable_linear_b1c0_dooms_rollout_v1.sbatch" smoke 0 0)
pilot_job=$(submit_job --job-name=xvla-dooms-pilot \
  --dependency="afterok:${smoke_job}" \
  "$lane_root/slurm_capable_linear_b1c0_dooms_rollout_v1.sbatch" pilot 0 0)
shard_0_3=$(submit_job --job-name=xvla-dooms-t0-3 \
  --dependency="afterok:${pilot_job}" \
  "$lane_root/slurm_capable_linear_b1c0_dooms_rollout_v1.sbatch" shard 0 3)
shard_3_6=$(submit_job --job-name=xvla-dooms-t3-6 \
  --dependency="afterok:${pilot_job}" \
  "$lane_root/slurm_capable_linear_b1c0_dooms_rollout_v1.sbatch" shard 3 6)
shard_6_8=$(submit_job --job-name=xvla-dooms-t6-8 \
  --dependency="afterok:${pilot_job}" \
  "$lane_root/slurm_capable_linear_b1c0_dooms_rollout_v1.sbatch" shard 6 8)
shard_8_10=$(submit_job --job-name=xvla-dooms-t8-10 \
  --dependency="afterok:${pilot_job}" \
  "$lane_root/slurm_capable_linear_b1c0_dooms_rollout_v1.sbatch" shard 8 10)
aggregate_job=$(submit_job \
  --dependency="afterok:${shard_0_3}:${shard_3_6}:${shard_6_8}:${shard_8_10}" \
  "$lane_root/slurm_capable_linear_b1c0_dooms_aggregate_v1.sbatch")
verify_fixed
printf '%s\n' \
  "unit_job=$unit_job" \
  "full_job=$full_job" \
  "smoke_job=$smoke_job" \
  "pilot_job=$pilot_job" \
  "shard_0_3=$shard_0_3" \
  "shard_3_6=$shard_3_6" \
  "shard_6_8=$shard_6_8" \
  "shard_8_10=$shard_8_10" \
  "aggregate_job=$aggregate_job"
