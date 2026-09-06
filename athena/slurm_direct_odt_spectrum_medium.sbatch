#!/usr/bin/env bash
#SBATCH --job-name=odt-spectrum-medium
#SBATCH --partition=compute
#SBATCH --account=vgg
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=athena/logs/%x-%j.out
#SBATCH --error=athena/logs/%x-%j.err

set -euo pipefail

readonly run_root="${1:?immutable run root required}"
readonly output="${2:?new result output required}"
readonly python_executable=/users/joy/miniconda3/envs/safesae-openvla/bin/python
readonly source_manifest="$run_root/athena/direct_odt_spectrum_scaling_sources.sha256"
readonly expected_manifest_sha256=ebdb779779dd99fcd3b214663a0cfac187fd9d06cbed579a44f0326db792cbb0

cd "$run_root"
export PYTHONPATH="$run_root"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

test -f "$source_manifest"
test ! -L "$source_manifest"
printf '%s  %s\n' "$expected_manifest_sha256" "$source_manifest" | sha256sum --check --strict --quiet -
sha256sum --check --strict --quiet "$source_manifest"
test ! -e "$output"
test ! -L "$output"
"$python_executable" -B scripts/run_direct_odt_spectrum_scaling.py \
  --case medium \
  --samples 2 \
  --block-size 512 \
  --output "$output"
printf '%s  %s\n' "$expected_manifest_sha256" "$source_manifest" | sha256sum --check --strict --quiet -
sha256sum --check --strict --quiet "$source_manifest"
sha256sum "$output"
