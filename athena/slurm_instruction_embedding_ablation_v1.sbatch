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
export XVLA_INSTRUCTION_EMBEDDING_COMMON_SHA256="8a8992f3fc5cdbbc92db2bfbf69c09e03f2d23d313a4155ab209d4a067c62865"
export XVLA_INSTRUCTION_EMBEDDING_RUNNER_SHA256="ef645a3175e6659d199ae9716ee2e98f56568371bea1781a04193e7822a76ff7"

cd /work/joy/x-vla-workshop
[[ "$(sha256sum athena/instruction_embedding_ablation_v1_common.py | cut -d' ' -f1)" == "$XVLA_INSTRUCTION_EMBEDDING_COMMON_SHA256" ]]
[[ "$(sha256sum athena/run_instruction_embedding_ablation_v1.py | cut -d' ' -f1)" == "$XVLA_INSTRUCTION_EMBEDDING_RUNNER_SHA256" ]]
python -u athena/run_instruction_embedding_ablation_v1.py "$@"
