#!/bin/bash
set -euo pipefail
campaign_root="${ODT_CAMPAIGN_ROOT:?isolated source packet required}"
mode="${ODT_CAMPAIGN_MODE:?prepare or task required}"
cd "$campaign_root"
export PYTHONPATH="$campaign_root" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS" NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
sha256sum --check sources.sha256
runtime=/work/joy/safesae/bin/python
if [[ "$mode" == prepare ]]; then
  "$runtime" -u -m scripts.test_odt_campaign_v1 > campaign_tests.log 2>&1
  exec "$runtime" -u -m research.odt_campaign_v1.campaign prepare \
    --producer /work/joy/x-vla-odt-deadline-curve-v5/results \
    --real-panel /work/joy/x-vla-odt-campaign-20260907-v1/native_v1/results/panel.npz \
    --real-metadata /work/joy/x-vla-odt-campaign-20260907-v1/native_v1/results/panel_manifest.json \
    --output "$campaign_root/campaign"
elif [[ "$mode" == task ]]; then
  index="${SLURM_ARRAY_TASK_ID:?explicit array task index required}"
  [[ "$index" =~ ^[0-9]+$ ]] || exit 2
  exec "$runtime" -u -m research.odt_campaign_v1.campaign task \
    --producer /work/joy/x-vla-odt-deadline-curve-v5/results \
    --campaign "$campaign_root/campaign" --output "$campaign_root/results/task_$index" \
    --task-index "$index"
else
  exit 2
fi
