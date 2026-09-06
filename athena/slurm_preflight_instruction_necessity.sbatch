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
readonly expected_common_sha="701323551c7fc4a8bcf9a27a3858bbef844e4486aacd8af6515922ce87cda188"
readonly expected_preflight_sha="10e2e001426b42680c867dc1fc91c30278cfe99cd4a2f3feb73189b2dac6573f"
[[ "$(sha256sum athena/instruction_necessity_common.py | cut -d' ' -f1)" == "$expected_common_sha" ]]
[[ "$(sha256sum athena/preflight_instruction_necessity.py | cut -d' ' -f1)" == "$expected_preflight_sha" ]]
python -u athena/preflight_instruction_necessity.py "$@"
