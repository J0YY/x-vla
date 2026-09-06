#!/bin/bash
# One independent-clone producer, then a six-task array (external %3 cap).
set -euo pipefail
campaign_root=/work/joy/x-vla-odt-deadline-curve-v3
cd "$campaign_root"
export PYTHONPATH="$campaign_root"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS" VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"
sha256sum --check sources.sha256
if [[ "${1:-}" == prepare ]]; then
    /work/joy/safesae/bin/python -u -m research.odt_reference.run_tests --suite all
    exec /work/joy/safesae/bin/python -u -m research.odt_reference.run_curve \
        --phase prepare --output "$campaign_root/results" \
        --checkpoint /work/joy/x-vla-capable-linear-b1c0-odt-v2/inputs/capable_linear_b1c0_checkpoint.pt
elif [[ "${1:-}" == evaluate ]]; then
    percentages=(30 40 50 60 70 80)
    index="${SLURM_ARRAY_TASK_ID:?array index required}"
    [[ "$index" =~ ^[0-5]$ ]] || exit 2
    exec /work/joy/safesae/bin/python -u -m research.odt_reference.run_curve \
        --phase evaluate --output "$campaign_root/results" --percent "${percentages[$index]}"
else
    exit 2
fi
