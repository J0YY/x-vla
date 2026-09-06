#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

readonly run_root="$1"
readonly image_size="$2"
readonly dimension="$3"
readonly layers="$4"
readonly maximum_core_elements="$5"
readonly maximum_environment_elements="$6"
readonly output="$7"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python

cd "$run_root"
export PYTHONPATH="$run_root"
export PYTHONUNBUFFERED=1

"$python_executable" scripts/run_residual_attention_odt_scaling.py \
  --image-size "$image_size" \
  --dim "$dimension" \
  --layers "$layers" \
  --maximum-core-elements "$maximum_core_elements" \
  --maximum-environment-elements "$maximum_environment_elements" \
  --output "$output"

sha256sum "$output"
