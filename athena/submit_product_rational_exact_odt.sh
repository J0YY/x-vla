#!/bin/bash
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 CAPABILITY_AGGREGATE_JOB_ID" >&2
  exit 2
fi
capability_aggregate_job_id=$1
stage_root=/work/joy/x-vla-product-pade-rational-odt-v3
if [[ "$PWD" != "$stage_root" ]]; then
  cd "$stage_root"
fi
lock="$stage_root/athena/results/product_pade_rational_exact_odt_v1_gates/submission.lock"
mkdir "$lock"

submit_job() {
  local result
  result=$(sbatch --parsable --kill-on-invalid-dep=yes "$@")
  result=${result%%;*}
  if [[ ! "$result" =~ ^[0-9]+$ ]]; then
    echo "sbatch returned a nonnumeric job id: $result" >&2
    exit 1
  fi
  printf '%s\n' "$result"
}

preflight_job=$(submit_job athena/slurm_product_rational_odt_validate.sbatch preflight)
oracle_job=$(submit_job athena/slurm_product_rational_odt_validate.sbatch oracle)
lane_tests_job=$(submit_job athena/slurm_product_rational_odt_validate.sbatch lane-tests)
full_job=$(submit_job \
  --dependency="afterok:${preflight_job}:${oracle_job}:${lane_tests_job}" \
  athena/slurm_product_rational_odt_full.sbatch)
composite_job=$(submit_job \
  --dependency="afterok:${full_job}:${capability_aggregate_job_id}" \
  athena/slurm_product_rational_odt_composite.sbatch)

printf '%s\n' \
  "preflight_job=$preflight_job" \
  "oracle_job=$oracle_job" \
  "lane_tests_job=$lane_tests_job" \
  "full_job=$full_job" \
  "capability_aggregate_job=$capability_aggregate_job_id" \
  "composite_job=$composite_job"
