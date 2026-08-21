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
export XVLA_INSTRUCTION_NECESSITY_COMMON_SHA256="8ab8839241adbc93ddc486c4b27a96f3bd78e1afd4a9f543603ee2206efc7551"
export XVLA_INSTRUCTION_NECESSITY_RUNNER_SHA256="90e06ec5d29d2c2f0b466dc259984190240270ce3110f787352de78689496a7e"

cd /work/joy/x-vla-workshop
[[ "$(sha256sum athena/instruction_necessity_common.py | cut -d' ' -f1)" == "$XVLA_INSTRUCTION_NECESSITY_COMMON_SHA256" ]]
[[ "$(sha256sum athena/run_instruction_necessity.py | cut -d' ' -f1)" == "$XVLA_INSTRUCTION_NECESSITY_RUNNER_SHA256" ]]
python -u athena/run_instruction_necessity.py "$@"
