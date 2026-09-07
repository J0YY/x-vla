#!/bin/bash
set -euo pipefail
campaign_root="${ODT_ROLLOUT_SUMMARY_ROOT:?isolated source packet required}"
cd "$campaign_root"
export PYTHONPATH="$campaign_root" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
sha256sum --check sources.sha256
runtime=/work/joy/safesae/bin/python
"$runtime" -u -m scripts.test_odt_rollout_summary_v1 -v > summary_tests.log 2>&1
exec "$runtime" -u -m scripts.odt_rollout_summary_v1 \
  --rollout /work/joy/x-vla-odt-campaign-20260907-v1/rollout_v1 \
  --training /work/joy/x-vla-capable-linear-b1c0-odt-v2/inputs/capable_linear_training.json \
  --output "$campaign_root/results"
