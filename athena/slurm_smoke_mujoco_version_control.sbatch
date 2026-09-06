#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail

env_prefix=/work/joy/safesae-openvla
overlay=/work/joy/xvla-mujoco-3.1.6-overlay
cd /work/joy/x-vla-workshop

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/work/joy/hf_cache
export PYTHONPATH="$overlay:/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO"
unset LD_LIBRARY_PATH
export CONDA_PREFIX="$env_prefix"
export XVLA_MUJOCO_OVERLAY="$overlay"

"$env_prefix/bin/python" -u athena/run_mujoco_version_forensics.py \
  --mode smoke \
  --architecture chi \
  --checkpoint artifacts/ckpt_linear_rat_vit_s1.pt \
  --cache artifacts/libero_frames_100000_64.pkl \
  --output results/mujoco_version_overlay_control/mujoco316_smoke.json \
  --seed 1 \
  --matmul-precision highest \
  --task-start 5 \
  --task-end 6 \
  --eps-per-task 1 \
  --max-steps 16 \
  --profile-iters 2

"$env_prefix/bin/python" - \
  results/mujoco_version_overlay_control/mujoco316_smoke.json \
  results/mujoco_version_overlay_control/environment_setup.json \
  "$overlay" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
setup = json.loads(Path(sys.argv[2]).read_text())
overlay = sys.argv[3]
forensics = result["version_isolation_forensics"]
assert forensics["environment"]["critical_packages"]["mujoco"] == "3.1.6"
assert forensics["environment"]["mujoco_runtime_version"] == "3.1.6"
assert forensics["environment"]["mujoco_overlay"] == overlay
assert forensics["environment"]["mujoco_module"].startswith(f"{overlay}/")
assert (
    forensics["environment"]["mujoco_install_record_sha256"]
    == setup["overlay_runtime_identity"]["mujoco_distribution"]["record_sha256"]
)
assert (
    forensics["environment"]["mujoco_distribution_tree_sha256"]
    == setup["overlay_runtime_identity"]["mujoco_distribution"][
        "distribution_tree_sha256"
    ]
)
assert forensics["frozen_inputs"]["source_snapshot_stable"] is True
assert forensics["protocol"]["num_steps_wait"] == 10
assert forensics["protocol"]["exec_h"] == 8
assert forensics["protocol"]["res"] == 64
assert forensics["protocol"]["horizon"] == 8
assert result["matmul_precision"] == "highest"
assert result["prediction_finite"] is True
assert result["capability"]["trials"] == 1
assert len(forensics["episodes"]) == 1
assert len(forensics["episodes"][0]["policy_actions"]) == result["capability"]["episodes"][0]["steps"]
print("MUJOCO_316_SMOKE_PASS")
PY
