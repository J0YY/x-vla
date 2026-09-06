#!/usr/bin/env bash
#SBATCH --job-name=product-rational-train
#SBATCH --partition=gpu
#SBATCH --account=vgg
#SBATCH --qos=combined12gpus
#SBATCH --constraint=a6000
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/joy/x-vla-product-pade-rational-v2/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-product-pade-rational-v2/logs/%x_%j.err

set -euo pipefail

if [[ $# -ne 2 && $# -ne 4 ]]; then
  echo "usage: $0 MODE SEED [resume-precalibration SHA256]" >&2
  exit 2
fi
readonly mode=$1
readonly seed=$2
case "$mode" in
  smoke|full) ;;
  *) echo "mode must be smoke or full" >&2; exit 2 ;;
esac
case "$seed" in
  0|1|2) ;;
  *) echo "seed must be 0, 1, or 2" >&2; exit 2 ;;
esac
if [[ "$mode" == smoke && "$seed" != 0 ]]; then
  echo "smoke is frozen to seed 0" >&2
  exit 2
fi
declare -a resume_arguments=()
if [[ $# -eq 4 ]]; then
  if [[ "$3" != resume-precalibration ]]; then
    echo "the third argument must be the literal resume-precalibration" >&2
    exit 2
  fi
  if [[ ! "$4" =~ ^[0-9a-f]{64}$ ]]; then
    echo "the resume artifact SHA-256 must be 64 lowercase hexadecimal characters" >&2
    exit 2
  fi
  resume_arguments=(--resume-precalibration --precalibration-sha256 "$4")
fi

readonly project_root=/work/joy/x-vla-product-pade-rational-v2
readonly run_root="$project_root/athena/results/product_pade_rational_v2"
readonly manifest="$run_root/manifest/product_pade_rational_source_manifest.json"
readonly preflight="$run_root/gates/product_pade_rational_preflight.json"
readonly python_executable=/users/joy/miniconda3/envs/safesae-openvla/bin/python

cd "$project_root"
export PYTHONPATH="$project_root"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export HF_HOME=/work/joy/hf_cache
export HF_HUB_CACHE=/work/joy/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$python_executable" -B -u athena/train_product_rational_checkpoint.py \
  --run-root "$run_root" \
  --source-manifest "$manifest" \
  --mode "$mode" \
  --seed "$seed" \
  --preflight-result "$preflight" \
  "${resume_arguments[@]}"

if [[ "$mode" == smoke ]]; then
  readonly precalibration_artifact="$run_root/training_only/precalibration_ema_product_pade_rational_s0_smoke.pt"
  read -r precalibration_sha256 _ < <(sha256sum "$precalibration_artifact")
  if [[ ! "$precalibration_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "could not derive the physical smoke pre-calibration SHA-256" >&2
    exit 1
  fi
  "$python_executable" -B -u athena/train_product_rational_checkpoint.py \
    --run-root "$run_root" \
    --source-manifest "$manifest" \
    --mode smoke \
    --seed 0 \
    --preflight-result "$preflight" \
    --resume-precalibration \
    --precalibration-sha256 "$precalibration_sha256" \
    --resume-proof-only
fi
