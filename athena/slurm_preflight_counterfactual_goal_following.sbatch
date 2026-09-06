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
readonly expected_common_sha="616201783cc07a42124f2ff3de3bb5f2b35cfd5a0cc855ffcbda102b046560f7"
readonly expected_preflight_sha="83b68914ca33f99d143cc3068ab6e13ad3039d76246ecbe7333e790243f78b83"
[[ "$(sha256sum athena/counterfactual_goal_following_common.py | cut -d' ' -f1)" == "$expected_common_sha" ]]
[[ "$(sha256sum athena/preflight_counterfactual_goal_following.py | cut -d' ' -f1)" == "$expected_preflight_sha" ]]
python -u athena/preflight_counterfactual_goal_following.py "$@"
