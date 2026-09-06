#!/usr/bin/env bash
#SBATCH --job-name=xvla-dooms-rollout
#SBATCH --partition=low-prio-gpu
#SBATCH --account=vgg
#SBATCH --qos=normal
#SBATCH --nodelist=c2-g8-08
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=700G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/joy/x-vla-capable-linear-b1c0-dooms-v1/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-capable-linear-b1c0-dooms-v1/logs/%x_%j.err

set -euo pipefail
if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 MODE TASK_START TASK_END" >&2
  exit 2
fi
case "$1:$2:$3" in
  smoke:0:0|pilot:0:0|shard:0:3|shard:3:6|shard:6:8|shard:8:10) ;;
  *) echo "rollout mode or task range is not preregistered" >&2; exit 2 ;;
esac
readonly lane_root=/work/joy/x-vla-capable-linear-b1c0-dooms-v1
readonly frozen_root=/work/joy/x-vla-capable-linear-b1c0-odt-v2
readonly lane_ledger="$lane_root/stage.sha256"
readonly frozen_ledger="$frozen_root/athena/capable_linear_b1c0_stage.sha256"
readonly checkpoint="$frozen_root/inputs/capable_linear_b1c0_checkpoint.pt"
readonly checkpoint_sha=b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee
readonly python_bin=/users/joy/miniconda3/envs/safesae-openvla/bin/python
verify_fixed() {
  (cd "$lane_root" && sha256sum --check --strict --quiet "$lane_ledger")
  (cd "$frozen_root" && sha256sum --check --strict --quiet "$frozen_ledger")
  printf '%s  %s\n' "$checkpoint_sha" "$checkpoint" | sha256sum --check --strict --quiet
}
trap verify_fixed EXIT
verify_fixed
cd "$lane_root"
export PYTHONPATH="$frozen_root:/users/joy/LIBERO"
export PYTHONUNBUFFERED=1
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
"$python_bin" -B run_capable_linear_b1c0_dooms_rollout_v1.py \
  --mode "$1" --task-start "$2" --task-end "$3"
verify_fixed
