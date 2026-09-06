#!/usr/bin/env bash
#SBATCH --job-name=xvla-dooms-unit
#SBATCH --partition=compute
#SBATCH --account=vgg
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/joy/x-vla-capable-linear-b1c0-dooms-v1/logs/dooms_unit_%j.out
#SBATCH --error=/work/joy/x-vla-capable-linear-b1c0-dooms-v1/logs/dooms_unit_%j.err

set -euo pipefail
if [[ "$#" -ne 0 ]]; then
  echo "Dooms unit job accepts no arguments" >&2
  exit 2
fi
readonly lane_root=/work/joy/x-vla-capable-linear-b1c0-dooms-v1
readonly frozen_root=/work/joy/x-vla-capable-linear-b1c0-odt-v2
readonly lane_ledger="$lane_root/stage.sha256"
readonly frozen_ledger="$frozen_root/athena/capable_linear_b1c0_stage.sha256"
readonly python_bin=/users/joy/miniconda3/envs/safesae-openvla/bin/python
verify_fixed() {
  test -f "$lane_ledger"
  test ! -L "$lane_ledger"
  test -f "$frozen_ledger"
  test ! -L "$frozen_ledger"
  (cd "$lane_root" && sha256sum --check --strict --quiet "$lane_ledger")
  (cd "$frozen_root" && sha256sum --check --strict --quiet "$frozen_ledger")
}
trap verify_fixed EXIT
verify_fixed
cd "$lane_root"
export PYTHONPATH="$frozen_root"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
unset PYTEST_ADDOPTS PYTEST_PLUGINS
"$python_bin" -B run_capable_linear_b1c0_dooms_sweep_v1.py --mode self-test
"$python_bin" -B run_capable_linear_b1c0_dooms_rollout_v1.py --mode unit
verify_fixed
