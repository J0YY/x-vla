#!/bin/bash
set -euo pipefail
campaign_root="${ODT_FFN_BRIDGE_ROOT:?isolated source packet required}"
mode="${ODT_FFN_BRIDGE_MODE:?reference or native required}"
cd "$campaign_root"
export PYTHONPATH="$campaign_root" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
sha256sum --check sources.sha256
if [[ "$mode" == reference ]]; then
  runtime=/work/joy/safesae/bin/python
  "$runtime" -u -m research.odt_ffn_native_v1.test_reference > bridge_reference_tests.log 2>&1
  exec "$runtime" -u -m research.odt_ffn_native_v1.reference \
    --producer /work/joy/x-vla-odt-campaign-20260907-v1/ffn_v1/results \
    --receipt-sha256 d95301a94a3708bdb5f77bd7c9ae2d0896b4795670f1c80c5f6cc0948e1dc790 \
    --panel /work/joy/x-vla-odt-campaign-20260907-v1/native_v1/results/panel.npz \
    --panel-metadata /work/joy/x-vla-odt-campaign-20260907-v1/native_v1/results/panel_manifest.json \
    --source-manifest reference_sources.json --output "$campaign_root/variants"
elif [[ "$mode" == native ]]; then
  runtime=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python
  "$runtime" -u -m research.odt_ffn_native_v1.test_native -v > bridge_native_tests.log 2>&1
  exec "$runtime" -u -m research.odt_ffn_native_v1.native \
    --variants "$campaign_root/variants" \
    --panel /work/joy/x-vla-odt-campaign-20260907-v1/native_v1/results/panel.npz \
    --panel-metadata /work/joy/x-vla-odt-campaign-20260907-v1/native_v1/results/panel_manifest.json \
    --checkpoint /work/joy/x-vla-capable-linear-b1c0-odt-v2/inputs/capable_linear_b1c0_checkpoint.pt \
    --source-manifest native_sources.json --output "$campaign_root/actions" \
    --device cuda --token-batch 16 --image-batch 2
else
  exit 2
fi
