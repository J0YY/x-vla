#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

readonly run_root="$1"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python
readonly source_manifest="$run_root/athena/planted_typed_vla_sources.sha256"

cd "$run_root"
export PYTHONPATH="$run_root"

sha256sum --check "$source_manifest"
export XVLA_SOURCE_TREE_SHA256="$(find xvla -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)"
export XVLA_PIP_FREEZE_SHA256="$("$python_executable" -m pip freeze | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"
echo "xvla_python_tree_sha256=$XVLA_SOURCE_TREE_SHA256"
echo "pip_freeze_sha256=$XVLA_PIP_FREEZE_SHA256"
"$python_executable" -c 'import platform, torch; print(f"python={platform.python_version()} torch={torch.__version__}")'

"$python_executable" -m pytest -q \
  tests/test_generalized_metric_odt.py \
  tests/test_planted_typed_vla.py

"$python_executable" athena/run_planted_typed_vla.py \
  --output athena/results/planted_typed_vla_v1.json \
  --artifact-dir athena/results/planted_typed_vla_v1_artifacts

sha256sum athena/results/planted_typed_vla_v1.json
sha256sum athena/results/planted_typed_vla_v1_artifacts/*.npz
sha256sum --check "$source_manifest"
