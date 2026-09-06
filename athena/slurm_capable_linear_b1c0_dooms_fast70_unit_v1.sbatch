#!/usr/bin/env bash
#SBATCH --job-name=xvla-fast70-unit
#SBATCH --partition=compute
#SBATCH --account=vgg
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/joy/x-vla-capable-linear-b1c0-dooms-fast70-v1/logs/fast70_unit_%j.out
#SBATCH --error=/work/joy/x-vla-capable-linear-b1c0-dooms-fast70-v1/logs/fast70_unit_%j.err

set -euo pipefail
readonly lane=/work/joy/x-vla-capable-linear-b1c0-dooms-fast70-v1
readonly source_lane=/work/joy/x-vla-capable-linear-b1c0-dooms-v1
readonly frozen=/work/joy/x-vla-capable-linear-b1c0-odt-v2
readonly python=/users/joy/miniconda3/envs/safesae-openvla/bin/python
verify_fixed() {
  (cd "$lane" && sha256sum --check --strict --quiet stage.sha256)
  (cd "$source_lane" && sha256sum --check --strict --quiet stage.sha256)
  (cd "$frozen" && sha256sum --check --strict --quiet athena/capable_linear_b1c0_stage.sha256)
}
trap verify_fixed EXIT
verify_fixed
cd "$lane"
export PYTHONPATH="$frozen"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"$python" -B run_capable_linear_b1c0_dooms_fast70_v1.py --mode sweep-unit
"$python" -B run_capable_linear_b1c0_dooms_fast70_v1.py --mode rollout-unit
verify_fixed
