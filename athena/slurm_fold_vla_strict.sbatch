#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
readonly run_root="$1"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python
readonly source_manifest="$run_root/athena/fold_vla_strict_sources.sha256"
cd "$run_root"
export PYTHONPATH="$run_root"
sha256sum --check "$source_manifest"
export XVLA_SOURCE_TREE_SHA256="$(find xvla -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)"
export XVLA_PIP_FREEZE_SHA256="$("$python_executable" -m pip freeze | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"
"$python_executable" -m pytest -q \
  tests/test_fold_vla.py \
  tests/test_vla_terminal_balanced_odt.py \
  tests/test_residual_route_odt.py \
  tests/test_complete_attention_odt.py \
  tests/test_generalized_metric_odt.py \
  tests/test_global_odt_bridge.py \
  tests/test_canonical_odt.py
"$python_executable" athena/run_fold_vla_strict.py \
  --seeds 32 \
  --output athena/results/fold_vla_strict_v1.json
sha256sum athena/results/fold_vla_strict_v1.json
sha256sum --check "$source_manifest"
