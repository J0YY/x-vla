#!/usr/bin/env bash
#SBATCH --job-name=xvla-rtx6k-aggregate
#SBATCH --partition=compute
#SBATCH --account=vgg
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work/joy/x-vla-capable-linear-rtx6000-v3/athena/logs/rtx6000_aggregate_%j.out
#SBATCH --error=/work/joy/x-vla-capable-linear-rtx6000-v3/athena/logs/rtx6000_aggregate_%j.err

set -euo pipefail
if [[ "$#" -ne 0 ]]; then
  echo "RTX 6000 aggregate accepts no arguments" >&2
  exit 2
fi
readonly stage_root=/work/joy/x-vla-capable-linear-rtx6000-v3
readonly stage_ledger="$stage_root/athena/capable_linear_rtx6000_stage.sha256"
readonly checkpoint="$stage_root/inputs/capable_linear_b1c0_checkpoint.pt"
readonly checkpoint_sha=b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee
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
  printf '%s  %s\n' "$checkpoint_sha" "$checkpoint" | sha256sum --check --strict --quiet
}
trap verify_fixed EXIT
verify_fixed
"$python_bin" -B scripts/run_capable_linear_rtx6000_capability.py --mode aggregate
verify_fixed
