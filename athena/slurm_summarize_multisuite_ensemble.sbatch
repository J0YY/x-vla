#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MANIFEST MANIFEST_SHA256" >&2
  exit 2
fi

cd /work/joy/x-vla-workshop
export PYTHONPATH="/work/joy/x-vla-workshop:${PYTHONPATH:-}"

python3 -u athena/summarize_multisuite_ensemble.py \
  --manifest "$1" \
  --manifest-sha256 "$2" \
  --results-dir results \
  --output results/multisuite_mean_ensemble_v1_summary.json
