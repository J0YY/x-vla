#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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

readonly runner=athena/run_vit_full_learned_forward_certificate.py
readonly summary=athena/summarize_vit_full_learned_forward_certificate.py
readonly expected_runner_sha=7bdf83530accb3a705f412fee7b89a9b0430ffc693d1a6264f528220edadc012
readonly expected_summary_sha=d069f8ea2272d3659bf96837136653ec84ee30616623df9a544a9893d7c071b0
readonly source_bundle=athena/VIT_FULL_LEARNED_FORWARD_SOURCE_BUNDLE.sha256
readonly expected_source_bundle_sha=a02628e818eb2afe17f0bd751e6fed6e3ba9e5479d77ca980badd52e5fe206c1
actual_runner_sha=$(sha256sum "$runner" | cut -d' ' -f1)
actual_summary_sha=$(sha256sum "$summary" | cut -d' ' -f1)
actual_source_bundle_sha=$(sha256sum "$source_bundle" | cut -d' ' -f1)
if [[ "$actual_runner_sha" != "$expected_runner_sha" ]]; then
  echo "Refusing stale full-forward runner: $actual_runner_sha" >&2
  exit 3
fi
if [[ "$actual_summary_sha" != "$expected_summary_sha" ]]; then
  echo "Refusing stale full-forward summary: $actual_summary_sha" >&2
  exit 3
fi
if [[ "$actual_source_bundle_sha" != "$expected_source_bundle_sha" ]]; then
  echo "Refusing stale full-forward source bundle: $actual_source_bundle_sha" >&2
  exit 3
fi
sha256sum --check --strict "$source_bundle"

python -u "$summary" "$@"
