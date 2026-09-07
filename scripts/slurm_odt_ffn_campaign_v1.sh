#!/bin/bash
set -euo pipefail
campaign_root="${ODT_FFN_ROOT:?isolated source packet required}"
cd "$campaign_root"
export PYTHONPATH="$campaign_root" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS" NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
sha256sum --check sources.sha256
runtime=/work/joy/safesae/bin/python
"$runtime" -c 'import numpy; assert numpy.__version__ == "2.2.6", numpy.__version__'
"$runtime" -u -m research.odt_reference.run_tests --suite all > reference_tests.log 2>&1
"$runtime" -u -m research.odt_ffn_v1.test_ffn > ffn_tests.log 2>&1
exec "$runtime" -u -m research.odt_ffn_v1.run \
  --checkpoint /work/joy/x-vla-capable-linear-b1c0-odt-v2/inputs/capable_linear_b1c0_checkpoint.pt \
  --output "$campaign_root/results"
