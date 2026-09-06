#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail

cd /work/joy/x-vla-workshop
python3 athena/summarize_surgery.py results --output results/surgery_discovery_summary.json
