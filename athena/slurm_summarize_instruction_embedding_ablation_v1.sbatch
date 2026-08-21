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
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export PYTHONPATH="/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO:${PYTHONPATH:-}"
export XVLA_SUMMARY_PYTHON_EXECUTABLE=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python

cd /work/joy/x-vla-workshop
readonly expected_common_sha="8a8992f3fc5cdbbc92db2bfbf69c09e03f2d23d313a4155ab209d4a067c62865"
readonly expected_summary_sha="f633b8d54bc94b4ce7f593d216245195445d38e5e6ba945bf60ac657f24f8d78"
[[ "$(sha256sum athena/instruction_embedding_ablation_v1_common.py | cut -d' ' -f1)" == "$expected_common_sha" ]]
[[ "$(sha256sum athena/summarize_instruction_embedding_ablation_v1.py | cut -d' ' -f1)" == "$expected_summary_sha" ]]
[[ -x "$XVLA_SUMMARY_PYTHON_EXECUTABLE" ]]
"$XVLA_SUMMARY_PYTHON_EXECUTABLE" -u athena/summarize_instruction_embedding_ablation_v1.py "$@"
