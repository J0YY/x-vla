#!/bin/bash
set -euo pipefail
campaign_root="${ODT_NATIVE_ROOT:?isolated source packet required}"
cd "$campaign_root"
export PYTHONPATH="$campaign_root" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
sha256sum --check sources.sha256
runtime=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python
"$runtime" -u -m unittest scripts.test_odt_real_panel_v1 -v > panel_tests.log 2>&1
exec "$runtime" -u -m scripts.odt_real_panel_v1 \
  --source-manifest native_sources.sha256 \
  --checkpoint /work/joy/x-vla-capable-linear-b1c0-odt-v2/inputs/capable_linear_b1c0_checkpoint.pt \
  --training /work/joy/x-vla-capable-linear-b1c0-odt-v2/inputs/capable_linear_training.json \
  --cache /work/joy/x-vla-workshop/artifacts/libero_frames_100000_64.pkl \
  --provenance athena/results/cache_provenance_libero_object.json \
  --output "$campaign_root/results" --device cuda --batch-size 8
