#!/usr/bin/env bash

set -euo pipefail

cd /work/joy/x-vla-workshop

base=/work/joy/safesae-openvla
expected_base_freeze=40eaa7969d7e11e40a3835cb500c919fc5f470080598152614c548790b302599
actual_base_freeze=$(env -u PYTHONPATH -u XVLA_MUJOCO_OVERLAY PYTHONNOUSERSITE=1 \
  "$base/bin/python" -m pip freeze --all | LC_ALL=C sort | sha256sum | cut -d' ' -f1)
if [[ "$actual_base_freeze" != "$expected_base_freeze" ]]; then
  echo "Base environment package freeze changed" >&2
  exit 2
fi
base_mujoco=$(env -u PYTHONPATH -u XVLA_MUJOCO_OVERLAY PYTHONNOUSERSITE=1 \
  "$base/bin/python" -c 'import importlib.metadata; print(importlib.metadata.version("mujoco"))')
if [[ "$base_mujoco" != "3.5.0" ]]; then
  echo "Base MuJoCo version changed: $base_mujoco" >&2
  exit 2
fi
base_runtime_tag=$(env -u PYTHONPATH -u XVLA_MUJOCO_OVERLAY PYTHONNOUSERSITE=1 \
  "$base/bin/python" -c 'import platform, sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}-{platform.system().lower()}-{platform.machine().lower()}")')
if [[ "$base_runtime_tag" != "cp310-linux-x86_64" ]]; then
  echo "Frozen MuJoCo wheel requires cp310-linux-x86_64, found: $base_runtime_tag" >&2
  exit 2
fi

verify_sha256() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "$path" | cut -d' ' -f1)
  if [[ "$actual" != "$expected" ]]; then
    echo "Frozen input SHA mismatch: $path" >&2
    exit 2
  fi
}

verify_sha256 artifacts/libero_frames_100000_64.pkl \
  053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662
verify_sha256 artifacts/ckpt_linear_rat_vit_s1.pt \
  cc0780b989165a80a449c03cbb3574e2f56b2e5747a64cf32d1fde418df8ec3c
verify_sha256 artifacts/ckpt_linear_rat_vit_s0_v2.pt \
  96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9

for path in \
  /work/joy/xvla-mujoco-3.1.6-overlay \
  results/mujoco_version_overlay_control; do
  if [[ -e "$path" ]]; then
    echo "Refusing to overwrite existing path: $path" >&2
    exit 1
  fi
done

setup_job=$(sbatch --parsable \
  --job-name=xvla-mujoco316-overlay-setup \
  athena/slurm_setup_mujoco_version_control.sbatch)
smoke_job=$(sbatch --parsable \
  --partition=low-prio-gpu \
  --dependency="afterok:${setup_job}" \
  --job-name=xvla-mujoco316-overlay-smoke \
  athena/slurm_smoke_mujoco_version_control.sbatch)
eval_job=$(sbatch --parsable \
  --partition=low-prio-gpu \
  --dependency="afterok:${smoke_job}" \
  --job-name=xvla-mujoco-overlay-paired-20 \
  athena/slurm_eval_mujoco_version_control.sbatch)
summary_job=$(sbatch --parsable \
  --dependency="afterok:${eval_job}" \
  --job-name=xvla-mujoco-overlay-summary \
  athena/slurm_summarize_mujoco_version_control.sbatch)

echo "setup $setup_job"
echo "smoke $smoke_job"
echo "eval $eval_job"
echo "summary $summary_job"
