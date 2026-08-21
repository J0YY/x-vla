#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --constraint=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
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
export XVLA_TARGET_SWAP_COMMON_SHA256="26d185f15f787ac292a3457cb66b9773cd4281f3ee4050fc9971a6dd4d817b81"
export XVLA_TARGET_SWAP_RUNNER_SHA256="298a4b2bef1501c10e00d4818fc73b69cbc1f9e6aaa26e580ef8f290938f233a"

cd /work/joy/x-vla-workshop
[[ "$(sha256sum athena/counterfactual_target_swap_common.py | cut -d' ' -f1)" == "$XVLA_TARGET_SWAP_COMMON_SHA256" ]]
[[ "$(sha256sum athena/run_counterfactual_target_swap.py | cut -d' ' -f1)" == "$XVLA_TARGET_SWAP_RUNNER_SHA256" ]]
python -u athena/run_counterfactual_target_swap.py "$@"
