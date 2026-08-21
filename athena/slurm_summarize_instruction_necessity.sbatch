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
readonly expected_common_sha="701323551c7fc4a8bcf9a27a3858bbef844e4486aacd8af6515922ce87cda188"
readonly expected_summary_sha="07ba645e823fa2954fef76b6fa12c383ed8eee8d8806f4ef316294b6e6bc3014"
[[ "$(sha256sum athena/instruction_necessity_common.py | cut -d' ' -f1)" == "$expected_common_sha" ]]
[[ "$(sha256sum athena/summarize_instruction_necessity.py | cut -d' ' -f1)" == "$expected_summary_sha" ]]
python -u athena/summarize_instruction_necessity.py "$@"
