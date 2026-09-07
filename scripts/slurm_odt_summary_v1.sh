#!/bin/bash
set -euo pipefail
campaign_root="${ODT_SUMMARY_ROOT:?isolated source packet required}"
cd "$campaign_root"
export PYTHONPATH="$campaign_root" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
sha256sum --check sources.sha256
runtime=/work/joy/safesae/bin/python
"$runtime" -u -m scripts.test_odt_campaign_summary_v1 > summary_tests.log 2>&1
exec "$runtime" -u -m scripts.odt_campaign_summary_v1 \
  --campaign /work/joy/x-vla-odt-campaign-20260907-v1/consumer_v1/campaign \
  --results /work/joy/x-vla-odt-campaign-20260907-v1/consumer_v1/results \
  --output "$campaign_root/results"
