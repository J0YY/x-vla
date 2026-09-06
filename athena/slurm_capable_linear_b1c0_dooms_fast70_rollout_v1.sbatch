#!/usr/bin/env bash
#SBATCH --job-name=xvla-fast70-rollout
#SBATCH --partition=low-prio-gpu
#SBATCH --account=vgg
#SBATCH --qos=normal
#SBATCH --nodelist=c2-g8-07
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/joy/x-vla-capable-linear-b1c0-dooms-fast70-v1/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-capable-linear-b1c0-dooms-fast70-v1/logs/%x_%j.err

set -euo pipefail
if [[ "$#" -ne 1 || ( "$1" != smoke && "$1" != pilot ) ]]; then
  echo "usage: $0 smoke|pilot" >&2
  exit 2
fi
readonly lane=/work/joy/x-vla-capable-linear-b1c0-dooms-fast70-v1
readonly source_lane=/work/joy/x-vla-capable-linear-b1c0-dooms-v1
readonly frozen=/work/joy/x-vla-capable-linear-b1c0-odt-v2
readonly checkpoint="$frozen/inputs/capable_linear_b1c0_checkpoint.pt"
readonly checkpoint_sha=b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee
readonly python=/users/joy/miniconda3/envs/safesae-openvla/bin/python
verify_fixed() {
  (cd "$lane" && sha256sum --check --strict --quiet stage.sha256)
  (cd "$source_lane" && sha256sum --check --strict --quiet stage.sha256)
  (cd "$frozen" && sha256sum --check --strict --quiet athena/capable_linear_b1c0_stage.sha256)
  printf '%s  %s\n' "$checkpoint_sha" "$checkpoint" | sha256sum --check --strict --quiet
}
trap verify_fixed EXIT
verify_fixed
cd "$lane"
export PYTHONPATH="$frozen:/users/joy/LIBERO"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"$python" -B run_capable_linear_b1c0_dooms_fast70_v1.py --mode "$1"
verify_fixed
