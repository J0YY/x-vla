#!/usr/bin/env bash
#SBATCH --partition=ddp-2way
#SBATCH --qos=combined12gpus
#SBATCH --constraint=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail
source /athenahomes/joy/miniconda3/etc/profile.d/conda.sh
conda activate safesae-openvla
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/work/joy/hf_cache
export PYTHONPATH="/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO:${PYTHONPATH:-}"
export XVLA_FROZEN_EVALUATION_GPU_FAMILY=a6000
export XVLA_INSTRUCTION_NECESSITY_COMMON_SHA256="f90e6088889fbf30805e25ef30880d903dbda2fc8b8e335270fa683c2440c27f"
export XVLA_INSTRUCTION_NECESSITY_RUNNER_SHA256="9e5a1d3892ce88caab6b6a934a8f70639dd09ea1b05c2ad345f31e0f0b9157ea"

cd /work/joy/x-vla-workshop
[[ "$(sha256sum athena/instruction_necessity_common.py | cut -d' ' -f1)" == "$XVLA_INSTRUCTION_NECESSITY_COMMON_SHA256" ]]
[[ "$(sha256sum athena/run_instruction_necessity.py | cut -d' ' -f1)" == "$XVLA_INSTRUCTION_NECESSITY_RUNNER_SHA256" ]]
python -u athena/run_instruction_necessity.py "$@"
