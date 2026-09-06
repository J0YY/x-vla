#!/usr/bin/env bash
#SBATCH --job-name=xvla-rtx6k-smoke
#SBATCH --partition=low-prio-gpu
#SBATCH --account=vgg
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --nodelist=c2-g8-05
#SBATCH --constraint=quadro_rtx_6000
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/work/joy/x-vla-capable-linear-rtx6000-v3/athena/logs/rtx6000_smoke_%j.out
#SBATCH --error=/work/joy/x-vla-capable-linear-rtx6000-v3/athena/logs/rtx6000_smoke_%j.err

set -euo pipefail
if [[ "$#" -ne 0 ]]; then
  echo "RTX 6000 smoke accepts no arguments" >&2
  exit 2
fi
readonly stage_root=/work/joy/x-vla-capable-linear-rtx6000-v3
readonly stage_ledger="$stage_root/athena/capable_linear_rtx6000_stage.sha256"
readonly checkpoint="$stage_root/inputs/capable_linear_b1c0_checkpoint.pt"
readonly checkpoint_sha=b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee
readonly python_bin=/users/joy/miniconda3/envs/safesae-openvla/bin/python
cd "$stage_root"
export PYTHONPATH="$stage_root:/users/joy/LIBERO"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
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
"$python_bin" -B scripts/run_capable_linear_rtx6000_capability.py --mode smoke
verify_fixed
