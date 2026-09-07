#!/bin/bash
set -euo pipefail
campaign_root="${ODT_FFN_V2_ROOT:?isolated source packet required}"
cd "$campaign_root"
export PYTHONPATH="$campaign_root" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
sha256sum --check sources.sha256
runtime=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python
"$runtime" -u -m research.odt_ffn_native_v2.test_native -v > native_v2_tests.log 2>&1
exec "$runtime" -u -m research.odt_ffn_native_v2.native \
  --variants /work/joy/x-vla-odt-campaign-20260907-v1/ffn_bridge_v1/variants \
  --panel /work/joy/x-vla-odt-campaign-20260907-v1/native_v1/results/panel.npz \
  --panel-metadata /work/joy/x-vla-odt-campaign-20260907-v1/native_v1/results/panel_manifest.json \
  --checkpoint /work/joy/x-vla-capable-linear-b1c0-odt-v2/inputs/capable_linear_b1c0_checkpoint.pt \
  --source-manifest native_sources.json --output "$campaign_root/actions" \
  --device cuda --token-batch 16 --image-batch 2
