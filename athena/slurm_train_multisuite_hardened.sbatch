#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail

source /athenahomes/joy/miniconda3/etc/profile.d/conda.sh
conda activate safesae-openvla

export PYTHONUNBUFFERED=1
export PYTHONPATH="/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO:${PYTHONPATH:-}"
export XVLA_FROZEN_TRAINER_SHA256=9b63cce90d6c99a3e4f9ecef1db2914352e0221f57ed0bbd3e159eb6c1534f98

cd /work/joy/x-vla-workshop
observed_sha256=$(sha256sum athena/train_multisuite_checkpoint.py | awk '{print $1}')
if [[ "$observed_sha256" != "$XVLA_FROZEN_TRAINER_SHA256" ]]; then
  echo "Hardened trainer SHA-256 mismatch" >&2
  exit 3
fi
python -u athena/train_multisuite_checkpoint.py "$@"
end_sha256=$(sha256sum athena/train_multisuite_checkpoint.py | awk '{print $1}')
if [[ "$end_sha256" != "$XVLA_FROZEN_TRAINER_SHA256" ]]; then
  echo "Hardened trainer changed during the job" >&2
  exit 3
fi
