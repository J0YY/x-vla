#!/usr/bin/env bash
#SBATCH --job-name=xvla-physical-curve
#SBATCH --partition=low-prio-gpu
#SBATCH --account=vgg
#SBATCH --qos=normal
#SBATCH --nodelist=c2-g8-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=545G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/joy/x-vla-odt-dimension-curve-v2/logs/full_%j.out
#SBATCH --error=/work/joy/x-vla-odt-dimension-curve-v2/logs/full_%j.err

set -euo pipefail
if [[ "$#" -ne 1 || ! "$1" =~ ^[0-9a-f]{64}$ ]]; then
  echo "one exact stage ledger SHA-256 is required" >&2
  exit 2
fi
readonly curve_lane=/work/joy/x-vla-odt-dimension-curve-v2
readonly curve_frozen=/work/joy/x-vla-capable-linear-b1c0-odt-v2
readonly curve_python=/users/joy/miniconda3/envs/safesae-openvla/bin/python
verify_curve() {
  (cd "$curve_lane" && sha256sum --check --strict --quiet stage.sha256)
  (cd "$curve_frozen" && sha256sum --check --strict --quiet athena/capable_linear_b1c0_stage.sha256)
}
trap verify_curve EXIT
verify_curve
cd "$curve_lane"
export PYTHONPATH="$curve_frozen"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
unset PYTEST_ADDOPTS PYTEST_PLUGINS
"$curve_python" -B athena/run_direct_odt_dimension_curve_v1.py --stage-ledger-sha256 "$1"
verify_curve
