#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

echo "RETIRED: legacy canonical_odt.py tests use prohibited historical machinery." >&2
echo "Use scripts/run_implicit_sparse_projective_odt_vla.py from a frozen direct-only manifest." >&2
exit 64

readonly run_root="$1"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python

cd "$run_root"
export PYTHONPATH="$run_root"

"$python_executable" athena/verify_canonical_odt_freeze.py \
  --root "$run_root" \
  --manifest "$run_root/source_manifest.sha256"

readonly test_log="$run_root/results/canonical_odt_unit_${SLURM_JOB_ID}.log"
"$python_executable" -m pytest -q \
  tests/test_canonical_odt.py \
  tests/test_canonical_odt_pipeline.py \
  tests/test_odt.py \
  tests/test_odt_interp.py | tee "$test_log"

"$python_executable" athena/verify_canonical_odt_freeze.py \
  --root "$run_root" \
  --manifest "$run_root/source_manifest.sha256" \
  --test-log "$test_log" \
  --certificate "$run_root/results/canonical_odt_unit_certificate.json"
