#!/bin/bash
set -euo pipefail
campaign_root="${ODT_POST_ROOT:?isolated source packet required}"
mode="${ODT_POST_MODE:?ffn or scale required}"
cd "$campaign_root"
export PYTHONPATH="$campaign_root" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
sha256sum --check sources.sha256
runtime=/work/joy/safesae/bin/python
if [[ "$mode" == ffn ]]; then
  "$runtime" -u -m scripts.test_odt_ffn_summary_v1 -v > ffn_summary_tests.log 2>&1
  exec "$runtime" -u -m scripts.odt_ffn_summary_v1 \
    --actions /work/joy/x-vla-odt-campaign-20260907-v1/ffn_native_v2/actions \
    --panel /work/joy/x-vla-odt-campaign-20260907-v1/native_v1/results/panel.npz \
    --panel-metadata /work/joy/x-vla-odt-campaign-20260907-v1/native_v1/results/panel_manifest.json \
    --variants /work/joy/x-vla-odt-campaign-20260907-v1/ffn_bridge_v1/variants \
    --output "$campaign_root/ffn_results"
elif [[ "$mode" == scale ]]; then
  "$runtime" -u -m scripts.test_odt_campaign_scale_summary_v1 -v > scale_summary_tests.log 2>&1
  exec "$runtime" -u -m scripts.odt_campaign_scale_summary_v1 \
    --campaign /work/joy/x-vla-odt-campaign-20260907-v1/consumer_v1/campaign \
    --results /work/joy/x-vla-odt-campaign-20260907-v1/consumer_v1/results \
    --paired-summary /work/joy/x-vla-odt-campaign-20260907-v1/summary_v1/results/summary.json \
    --paired-summary-sha256 3d7887e8985e016fbe15116c81b9fbeca17031abb10cd7e0b803802e80f36353 \
    --output "$campaign_root/scale_results"
else
  exit 2
fi
