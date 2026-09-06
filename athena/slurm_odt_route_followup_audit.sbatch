#!/usr/bin/env bash
#SBATCH --job-name=xvla-odt-route-followup
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /work/joy/x-vla-workshop
export PYTHONPATH=/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python
"$python_executable" -u athena/odt_route_followup_audit.py "$@"
