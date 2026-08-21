#!/usr/bin/env bash
#SBATCH --partition=ddp-2way
#SBATCH --qos=combined12gpus
#SBATCH --constraint=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail
source /athenahomes/joy/miniconda3/etc/profile.d/conda.sh
conda activate safesae-openvla
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export PYTHONPATH="/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO:${PYTHONPATH:-}"

cd /work/joy/x-vla-workshop
readonly expected_common_sha="8a8992f3fc5cdbbc92db2bfbf69c09e03f2d23d313a4155ab209d4a067c62865"
readonly expected_preflight_sha="6f40ae57097bb5287092f2f605d85085f2fb9762d6f1dd83fc81dd8b8eb9f85c"
[[ "$(sha256sum athena/instruction_embedding_ablation_v1_common.py | cut -d' ' -f1)" == "$expected_common_sha" ]]
[[ "$(sha256sum athena/preflight_instruction_embedding_ablation_v1.py | cut -d' ' -f1)" == "$expected_preflight_sha" ]]
python -u athena/preflight_instruction_embedding_ablation_v1.py "$@"
