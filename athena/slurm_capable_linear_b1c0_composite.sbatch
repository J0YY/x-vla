#!/usr/bin/env bash
#SBATCH --job-name=xvla-b1c0-odt-composite
#SBATCH --partition=compute
#SBATCH --account=vgg
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work/joy/x-vla-capable-linear-b1c0-odt-v2/athena/logs/b1c0_odt_composite_%j.out
#SBATCH --error=/work/joy/x-vla-capable-linear-b1c0-odt-v2/athena/logs/b1c0_odt_composite_%j.err

set -euo pipefail
if [[ "$#" -ne 0 ]]; then
  echo "b1c0 composite job accepts no arguments" >&2
  exit 2
fi
readonly stage_root=/work/joy/x-vla-capable-linear-b1c0-odt-v2
readonly stage_ledger="$stage_root/athena/capable_linear_b1c0_stage.sha256"
readonly python_bin=/users/joy/miniconda3/envs/safesae-openvla/bin/python
cd "$stage_root"
export PYTHONPATH="$stage_root"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
unset PYTEST_ADDOPTS PYTEST_PLUGINS
verify_stage() {
  test -f "$stage_ledger"
  test ! -L "$stage_ledger"
  sha256sum --check --strict --quiet "$stage_ledger"
}
trap verify_stage EXIT
verify_stage
"$python_bin" -B scripts/run_capable_linear_direct_odt_spectrum.py --mode composite
verify_stage
