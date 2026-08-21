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
readonly expected_common_sha="8ab8839241adbc93ddc486c4b27a96f3bd78e1afd4a9f543603ee2206efc7551"
readonly expected_summary_sha="7bea6aee78d1594709524d110f91d973bd96e720ad04d618fde61ad405820263"
[[ "$(sha256sum athena/instruction_necessity_common.py | cut -d' ' -f1)" == "$expected_common_sha" ]]
[[ "$(sha256sum athena/summarize_instruction_necessity.py | cut -d' ' -f1)" == "$expected_summary_sha" ]]
python -u athena/summarize_instruction_necessity.py "$@"
