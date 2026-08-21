#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail

source /athenahomes/joy/miniconda3/etc/profile.d/conda.sh
conda activate safesae-openvla
export PYTHONUNBUFFERED=1
export HF_HOME=/work/joy/hf_cache
export PYTHONPATH="/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO:${PYTHONPATH:-}"

cd /work/joy/x-vla-workshop
readonly expected_common_sha="26d185f15f787ac292a3457cb66b9773cd4281f3ee4050fc9971a6dd4d817b81"
readonly expected_preflight_sha="06bc40b323e741d229bf54c19a288fd71537b9fd756140b9408a394d49e7c19e"
[[ "$(sha256sum athena/counterfactual_target_swap_common.py | cut -d' ' -f1)" == "$expected_common_sha" ]]
[[ "$(sha256sum athena/preflight_counterfactual_target_swap.py | cut -d' ' -f1)" == "$expected_preflight_sha" ]]
python -u athena/preflight_counterfactual_target_swap.py "$@"
