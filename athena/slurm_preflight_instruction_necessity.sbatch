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
readonly expected_common_sha="8ab8839241adbc93ddc486c4b27a96f3bd78e1afd4a9f543603ee2206efc7551"
readonly expected_preflight_sha="ed689e9ec2678ea28ca594a4778ddf6ee69244b3a4bd03c2067e37f3232d2c32"
[[ "$(sha256sum athena/instruction_necessity_common.py | cut -d' ' -f1)" == "$expected_common_sha" ]]
[[ "$(sha256sum athena/preflight_instruction_necessity.py | cut -d' ' -f1)" == "$expected_preflight_sha" ]]
python -u athena/preflight_instruction_necessity.py "$@"
