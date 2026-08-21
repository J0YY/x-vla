#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail

cd /work/joy/x-vla-workshop
python3 athena/summarize_blind_subspace.py \
  --results-dir results \
  --output results/visual_blind_rank96_summary.json \
  --bootstrap-samples 20000 \
  --seed 20260821
