#!/bin/bash
set -euo pipefail
campaign_root="${ODT_FINE_ROOT:?isolated source packet required}"
mode="${ODT_FINE_MODE:?prepare, task, or summary required}"
cd "$campaign_root"
export PYTHONPATH="$campaign_root" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4
sha256sum --check sources.sha256
runtime=/work/joy/safesae/bin/python
if [[ "$mode" == prepare ]]; then
  "$runtime" -u -m scripts.test_odt_campaign_fine_v1 -v > fine_tests.log 2>&1
  exec "$runtime" -u -m scripts.odt_campaign_fine_v1 \
    --parent /work/joy/x-vla-odt-campaign-20260907-v1/consumer_v1/campaign \
    --producer /work/joy/x-vla-odt-deadline-curve-v5/results --output "$campaign_root/campaign"
elif [[ "$mode" == task ]]; then
  index="${SLURM_ARRAY_TASK_ID:?explicit array index required}"
  [[ "$index" =~ ^[0-9]+$ ]] || exit 2
  exec "$runtime" -u -m research.odt_campaign_v1.campaign task \
    --producer /work/joy/x-vla-odt-deadline-curve-v5/results \
    --campaign "$campaign_root/campaign" --output "$campaign_root/results/task_$index" --task-index "$index"
elif [[ "$mode" == summary ]]; then
  exec "$runtime" -u -m scripts.odt_campaign_summary_v1 \
    --campaign "$campaign_root/campaign" --results "$campaign_root/results" --output "$campaign_root/analysis"
else
  exit 2
fi
