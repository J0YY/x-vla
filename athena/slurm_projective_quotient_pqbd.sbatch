#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
readonly run_root="$1"
readonly artifact_root=/work/joy/x-vla-workshop/artifacts
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python
cd "$run_root"
export PYTHONPATH="$run_root"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
sha256sum --check athena/projective_quotient_pqbd_sources.sha256
export XVLA_SOURCE_TREE_SHA256="$(find xvla -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)"
export XVLA_PIP_FREEZE_SHA256="$("$python_executable" -m pip freeze | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"
"$python_executable" -m pytest -q \
  tests/test_projective_quotient.py \
  tests/test_generalized_metric_odt.py \
  tests/test_complete_attention_odt.py \
  tests/test_residual_route_odt.py \
  tests/test_vla_terminal_balanced_odt.py \
  tests/test_multimodal_heads.py \
  tests/test_operators.py
"$python_executable" athena/run_projective_quotient_pqbd.py \
  --checkpoint "$artifact_root/ckpt_linear_rat_vit_s0_v2.pt" \
  --quotient-trials 64 \
  --pade-trials 32 \
  --attention-trials 16 \
  --output athena/results/projective_quotient_pqbd_v1.json
sha256sum athena/results/projective_quotient_pqbd_v1.json
sha256sum --check athena/projective_quotient_pqbd_sources.sha256
