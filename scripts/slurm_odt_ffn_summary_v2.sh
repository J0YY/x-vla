#!/bin/bash
set -euo pipefail
campaign_root="${ODT_FFN_SUMMARY_ROOT:?isolated source packet required}"
cd "$campaign_root"
export PYTHONPATH="$campaign_root" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
sha256sum --check sources.sha256
runtime=/work/joy/safesae/bin/python
"$runtime" -u -m scripts.test_odt_ffn_summary_v2 -v > summary_tests.log 2>&1
exec "$runtime" -u -m scripts.odt_ffn_summary_v2 \
  --actions /work/joy/x-vla-odt-campaign-20260907-v1/ffn_native_v2/actions \
  --panel /work/joy/x-vla-odt-campaign-20260907-v1/native_v1/results/panel.npz \
  --panel-metadata /work/joy/x-vla-odt-campaign-20260907-v1/native_v1/results/panel_manifest.json \
  --variants /work/joy/x-vla-odt-campaign-20260907-v1/ffn_bridge_v1/variants \
  --output "$campaign_root/results"
