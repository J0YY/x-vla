#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
readonly run_root="$1"
readonly artifact_root=/work/joy/x-vla-workshop/artifacts
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python
cd "$run_root"
export PYTHONPATH="$run_root:/athenahomes/joy/LIBERO"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME=/work/joy/x-vla-hf-cache
sha256sum --check athena/vla_terminal_balanced_odt_sources.sha256
export XVLA_SOURCE_TREE_SHA256="$(find xvla -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)"
export XVLA_PIP_FREEZE_SHA256="$("$python_executable" -m pip freeze | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"
"$python_executable" -m pytest -q tests/test_vla_terminal_balanced_odt.py
"$python_executable" athena/run_vla_terminal_balanced_odt.py \
  --cache "$artifact_root/libero_frames_100000_64.pkl" \
  --checkpoint "$artifact_root/ckpt_linear_rat_vit_s0_v2.pt" \
  --checkpoint "$artifact_root/ckpt_linear_rat_vit_s1.pt" \
  --checkpoint "$artifact_root/ckpt_linear_rat_vit_s2.pt" \
  --output athena/results/vla_terminal_balanced_odt_v1.json \
  --artifact-dir athena/results/vla_terminal_balanced_odt_v1_artifacts \
  --batch-size 32
sha256sum athena/results/vla_terminal_balanced_odt_v1.json
sha256sum athena/results/vla_terminal_balanced_odt_v1_artifacts/*.npz
sha256sum --check athena/vla_terminal_balanced_odt_sources.sha256
