#!/usr/bin/env bash
set -euo pipefail
readonly lane=/work/joy/x-vla-capable-linear-b1c0-dooms-fast70-v1
readonly source_lane=/work/joy/x-vla-capable-linear-b1c0-dooms-v1
readonly frozen=/work/joy/x-vla-capable-linear-b1c0-odt-v2
verify_fixed() {
  (cd "$lane" && sha256sum --check --strict --quiet stage.sha256)
  (cd "$source_lane" && sha256sum --check --strict --quiet stage.sha256)
  (cd "$frozen" && sha256sum --check --strict --quiet athena/capable_linear_b1c0_stage.sha256)
}
trap verify_fixed EXIT
verify_fixed
mkdir "$lane/results/submission.lock"
submit() {
  local value
  value=$(sbatch --parsable --kill-on-invalid-dep=yes "$@")
  value=${value%%;*}
  [[ "$value" =~ ^[0-9]+$ ]] || exit 1
  printf '%s\n' "$value"
}
unit=$(submit "$lane/slurm_capable_linear_b1c0_dooms_fast70_unit_v1.sbatch")
full=$(submit --dependency="afterok:$unit" "$lane/slurm_capable_linear_b1c0_dooms_fast70_full_v1.sbatch")
smoke=$(submit --job-name=xvla-fast70-smoke --dependency="afterok:$full" \
  "$lane/slurm_capable_linear_b1c0_dooms_fast70_rollout_v1.sbatch" smoke)
pilot=$(submit --job-name=xvla-fast70-pilot --dependency="afterok:$smoke" \
  "$lane/slurm_capable_linear_b1c0_dooms_fast70_rollout_v1.sbatch" pilot)
verify_fixed
printf '%s\n' "unit_job=$unit" "full_job=$full" "smoke_job=$smoke" "pilot_job=$pilot"
