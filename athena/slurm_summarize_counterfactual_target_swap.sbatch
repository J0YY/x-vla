#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
readonly expected_runner_sha="298a4b2bef1501c10e00d4818fc73b69cbc1f9e6aaa26e580ef8f290938f233a"
readonly expected_summary_sha="c15b3310ef937663ec7b57ad7d1115ade98e896aa69f27a7e6cf86815fa49759"
[[ "$(sha256sum athena/counterfactual_target_swap_common.py | cut -d' ' -f1)" == "$expected_common_sha" ]]
[[ "$(sha256sum athena/preflight_counterfactual_target_swap.py | cut -d' ' -f1)" == "$expected_preflight_sha" ]]
[[ "$(sha256sum athena/run_counterfactual_target_swap.py | cut -d' ' -f1)" == "$expected_runner_sha" ]]
[[ "$(sha256sum athena/summarize_counterfactual_target_swap.py | cut -d' ' -f1)" == "$expected_summary_sha" ]]
python -u athena/summarize_counterfactual_target_swap.py "$@"
