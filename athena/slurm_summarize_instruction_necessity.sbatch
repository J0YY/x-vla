#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail
source /athenahomes/joy/miniconda3/etc/profile.d/conda.sh
conda activate safesae-openvla
export PYTHONUNBUFFERED=1
export PYTHONPATH="/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO:${PYTHONPATH:-}"

cd /work/joy/x-vla-workshop
readonly expected_common_sha="f90e6088889fbf30805e25ef30880d903dbda2fc8b8e335270fa683c2440c27f"
readonly expected_summary_sha="23220f36afc3383eee604cc2de263c6c5403aa8f2ef5c43b7b77741b5214c78b"
[[ "$(sha256sum athena/instruction_necessity_common.py | cut -d' ' -f1)" == "$expected_common_sha" ]]
[[ "$(sha256sum athena/summarize_instruction_necessity.py | cut -d' ' -f1)" == "$expected_summary_sha" ]]
python -u athena/summarize_instruction_necessity.py "$@"
