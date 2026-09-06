#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

readonly run_root="$1"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python
readonly source_manifest="$run_root/athena/global_odt_bridge_sources.sha256"
readonly expected_xvla_tree_sha256=d622c89e3c3fc6f729f4171da5d303478510e38e847d57ff19aa2f4c3c0592d2

cd "$run_root"
export PYTHONPATH="$run_root"

sha256sum --check "$source_manifest"
actual_xvla_tree_sha256="$(find xvla -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)"
test "$actual_xvla_tree_sha256" = "$expected_xvla_tree_sha256"
export XVLA_SOURCE_TREE_SHA256="$actual_xvla_tree_sha256"
export XVLA_PIP_FREEZE_SHA256="$("$python_executable" -m pip freeze | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"
echo "xvla_python_tree_sha256=$XVLA_SOURCE_TREE_SHA256"
echo "pip_freeze_sha256=$XVLA_PIP_FREEZE_SHA256"
"$python_executable" -c \
  'import platform, torch; print(f"python={platform.python_version()} torch={torch.__version__}")'

"$python_executable" -m pytest -q \
  tests/test_global_odt_bridge.py \
  tests/test_vla_bilinear_atlas.py \
  tests/test_canonical_odt.py

"$python_executable" athena/run_global_odt_bridge.py \
  --seeds 32 \
  --dimension 2 \
  --actions 3 \
  --cp-rank 6 \
  --attention-components 2 \
  --output athena/results/global_odt_bridge_v1.json

sha256sum athena/results/global_odt_bridge_v1.json
sha256sum --check "$source_manifest"
