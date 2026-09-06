#!/usr/bin/env bash
#SBATCH --job-name=xvla-rtx6k-composite
#SBATCH --partition=compute
#SBATCH --account=vgg
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work/joy/x-vla-capable-linear-rtx6000-v3/athena/logs/rtx6000_composite_%j.out
#SBATCH --error=/work/joy/x-vla-capable-linear-rtx6000-v3/athena/logs/rtx6000_composite_%j.err

set -euo pipefail
if [[ "$#" -ne 0 ]]; then
  echo "RTX 6000 cross-stage composite accepts no arguments" >&2
  exit 2
fi
readonly stage_root=/work/joy/x-vla-capable-linear-rtx6000-v3
readonly stage_ledger="$stage_root/athena/capable_linear_rtx6000_stage.sha256"
readonly b1c_root=/work/joy/x-vla-capable-linear-b1c0-odt-v2
readonly b1c_ledger="$b1c_root/athena/capable_linear_b1c0_stage.sha256"
readonly b1c_ledger_sha=e4716a413bc742943b623ddebd39783bb54c40558b59446c99998aa07892b8c3
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
verify_fixed() {
  test -f "$stage_ledger"
  test ! -L "$stage_ledger"
  sha256sum --check --strict --quiet "$stage_ledger"
  printf '%s  %s\n' "$b1c_ledger_sha" "$b1c_ledger" | sha256sum --check --strict --quiet
  (cd "$b1c_root" && sha256sum --check --strict --quiet "$b1c_ledger")
}
trap verify_fixed EXIT
verify_fixed
"$python_bin" -B scripts/attest_capable_linear_rtx6000_cross_stage.py
verify_fixed
