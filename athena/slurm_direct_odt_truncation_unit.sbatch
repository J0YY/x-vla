#!/usr/bin/env bash
#SBATCH --job-name=odt-trunc-unit
#SBATCH --partition=compute
#SBATCH --account=vgg
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=athena/logs/%x-%j.out
#SBATCH --error=athena/logs/%x-%j.err

set -euo pipefail

readonly run_root="${1:?immutable run root required}"
readonly result_output="${2:?new result output required}"
readonly python_executable=/users/joy/miniconda3/envs/safesae-openvla/bin/python
readonly source_manifest="$run_root/athena/direct_odt_truncation_tests_sources.sha256"
readonly expected_manifest_sha256=7e1cb9d5be89300b212b1d4dc958d8565fbb22f580a23b6d090e6864dfb73f74

cd "$run_root"
export PYTHONPATH="$run_root"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
unset PYTEST_PLUGINS

test -f "$source_manifest"
test ! -L "$source_manifest"
printf '%s  %s\n' "$expected_manifest_sha256" "$source_manifest" | sha256sum --check --strict --quiet -
sha256sum --check --strict --quiet "$source_manifest"
test ! -e "$result_output"
test ! -L "$result_output"
"$python_executable" -B scripts/run_direct_odt_truncation_tests.py \
  --output "$result_output"
printf '%s  %s\n' "$expected_manifest_sha256" "$source_manifest" | sha256sum --check --strict --quiet -
sha256sum --check --strict --quiet "$source_manifest"
sha256sum "$result_output"
