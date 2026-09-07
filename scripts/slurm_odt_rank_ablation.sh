#!/bin/bash
set -euo pipefail
campaign_root="${ODT_ABLATION_ROOT:?isolated source packet required}"
cd "$campaign_root"
export PYTHONPATH="$campaign_root" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS" NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
sha256sum --check sources.sha256
index="${SLURM_ARRAY_TASK_ID:?explicit ablation task index required}"
[[ "$index" =~ ^([0-9]|1[0-9]|2[0-6])$ ]] || exit 2
exec /work/joy/safesae/bin/python -u -m scripts.odt_rank_ablation \
  --producer /work/joy/x-vla-odt-deadline-curve-v5/results \
  --output "$campaign_root/results/task_$index" --task-index "$index"
