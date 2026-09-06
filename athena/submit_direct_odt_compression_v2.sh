#!/usr/bin/env bash
# Submit four independent authenticated direct-ODT compression jobs.

set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "This frozen submission script accepts no arguments" >&2
  exit 2
fi

readonly run_root=/work/joy/x-vla-direct-odt-compression-v2
readonly stage_ledger="$run_root/athena/direct_odt_compression_stage.sha256"

cd "$run_root"
export PYTHONNOUSERSITE=1
test -f "$stage_ledger"
test ! -L "$stage_ledger"
sha256sum --check --strict --quiet "$stage_ledger"
mkdir athena/results/submission.lock

unit_raw=$(sbatch --parsable athena/slurm_direct_odt_truncation_unit.sbatch \
  "$run_root" "$run_root/athena/results/direct_odt_truncation_unit_v2.json")
unit_job=${unit_raw%%;*}
echo "unit=$unit_job"

tiny_raw=$(sbatch --parsable athena/slurm_direct_odt_spectrum_tiny.sbatch \
  "$run_root" "$run_root/athena/results/direct_odt_spectrum_tiny_v2.json")
tiny_job=${tiny_raw%%;*}
echo "tiny=$tiny_job"

small_raw=$(sbatch --parsable --dependency="afterok:$unit_job:$tiny_job" \
  athena/slurm_direct_odt_spectrum_small.sbatch \
  "$run_root" "$run_root/athena/results/direct_odt_spectrum_small_v2.json")
small_job=${small_raw%%;*}
echo "small=$small_job"

medium_raw=$(sbatch --parsable --dependency="afterok:$unit_job:$tiny_job" \
  athena/slurm_direct_odt_spectrum_medium.sbatch \
  "$run_root" "$run_root/athena/results/direct_odt_spectrum_medium_v2.json")
medium_job=${medium_raw%%;*}
echo "medium=$medium_job"

sha256sum --check --strict --quiet "$stage_ledger"
