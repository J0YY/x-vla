#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail

env_prefix=/work/joy/safesae-openvla-mujoco-3.1.6
cd /work/joy/x-vla-workshop

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/work/joy/hf_cache
export PYTHONPATH="/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO:${PYTHONPATH:-}"
export CONDA_PREFIX="$env_prefix"

"$env_prefix/bin/python" -u athena/run_mujoco_version_forensics.py \
  --mode smoke \
  --architecture chi \
  --checkpoint artifacts/ckpt_linear_rat_vit_s1.pt \
  --cache artifacts/libero_frames_100000_64.pkl \
  --output results/mujoco_version_control/mujoco316_smoke.json \
  --seed 1 \
  --matmul-precision highest \
  --task-start 5 \
  --task-end 6 \
  --eps-per-task 1 \
  --max-steps 16 \
  --profile-iters 2

"$env_prefix/bin/python" - results/mujoco_version_control/mujoco316_smoke.json <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
forensics = result["version_isolation_forensics"]
assert forensics["environment"]["critical_packages"]["mujoco"] == "3.1.6"
assert result["matmul_precision"] == "highest"
assert result["prediction_finite"] is True
assert result["capability"]["trials"] == 1
assert len(forensics["episodes"]) == 1
assert len(forensics["episodes"][0]["policy_actions"]) == result["capability"]["episodes"][0]["steps"]
print("MUJOCO_316_SMOKE_PASS")
PY
