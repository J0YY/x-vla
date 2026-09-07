#!/bin/bash
set -euo pipefail
campaign_root="${ODT_ROLLOUT_ROOT:?isolated source packet required}"
mode="${ODT_ROLLOUT_MODE:?smoke or worker required}"
cd "$campaign_root"
export PYTHONPATH="$campaign_root" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
sha256sum --check sources.sha256
runtime=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python
common=(--source-manifest rollout_sources.json
  --offline /work/joy/x-vla-odt-campaign-20260907-v1/ffn_native_v2/actions
  --offline-sha256 083d5f3d9e9ec1262cd4d9cbe62ec8a759e4768971bf7b5fc801bec991ffc73b
  --variants /work/joy/x-vla-odt-campaign-20260907-v1/ffn_bridge_v1/variants
  --checkpoint /work/joy/x-vla-capable-linear-b1c0-odt-v2/inputs/capable_linear_b1c0_checkpoint.pt
  --training /work/joy/x-vla-capable-linear-b1c0-odt-v2/inputs/capable_linear_training.json
  --device cuda)
if [[ "$mode" == smoke ]]; then
  "$runtime" -u -m pytest research/odt_ffn_rollout_v1/test_controller.py research/odt_ffn_rollout_v1/test_worker.py -q > rollout_tests.log 2>&1
  exec "$runtime" -u -m research.odt_ffn_rollout_v1.worker --mode smoke "${common[@]}" --output "$campaign_root/smoke"
elif [[ "$mode" == worker ]]; then
  index="${SLURM_ARRAY_TASK_ID:?explicit array index required}"
  [[ "$index" =~ ^[0-9]+$ ]] && (( index < 40 )) || exit 2
  arms=(native fullrank leading_k174 anchored_k174_s0)
  arm="${arms[$((index / 10))]}"
  task="$((index % 10))"
  exec "$runtime" -u -m research.odt_ffn_rollout_v1.worker --mode worker "${common[@]}" \
    --arm "$arm" --task-index "$task" --output "$campaign_root/results/task_$index" \
    --smoke "$campaign_root/smoke/receipt.json" --smoke-sha256 "${ODT_SMOKE_SHA256:?authenticated timing gate required}"
else
  exit 2
fi
