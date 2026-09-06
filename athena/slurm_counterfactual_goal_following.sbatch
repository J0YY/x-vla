#!/usr/bin/env bash
#SBATCH --partition=ddp-2way
#SBATCH --qos=combined12gpus
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
export XVLA_GOAL_FOLLOWING_COMMON_SHA256="616201783cc07a42124f2ff3de3bb5f2b35cfd5a0cc855ffcbda102b046560f7"
export XVLA_GOAL_FOLLOWING_RUNNER_SHA256="81987bddc951dbff0f5ec5408cb581a6e1ed7642d48eaa8dac0e05bd7413a73c"

cd /work/joy/x-vla-workshop
[[ "$(sha256sum athena/counterfactual_goal_following_common.py | cut -d' ' -f1)" == "$XVLA_GOAL_FOLLOWING_COMMON_SHA256" ]]
[[ "$(sha256sum athena/run_counterfactual_goal_following.py | cut -d' ' -f1)" == "$XVLA_GOAL_FOLLOWING_RUNNER_SHA256" ]]
python -u athena/run_counterfactual_goal_following.py "$@"
