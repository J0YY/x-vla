#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail

source /athenahomes/joy/miniconda3/etc/profile.d/conda.sh
conda activate safesae-openvla
export PYTHONPATH="/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO:${PYTHONPATH:-}"

cd /work/joy/x-vla-workshop
python -u athena/build_multisuite_manifest.py "$@"
