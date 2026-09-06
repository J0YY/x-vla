#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail

source /athenahomes/joy/miniconda3/etc/profile.d/conda.sh
conda activate safesae-openvla
cd /work/joy/x-vla-workshop

python -u athena/summarize_mujoco_version_control.py \
  --environment-setup results/mujoco_version_overlay_control/environment_setup.json \
  --result results/mujoco_version_overlay_control/base350_vit_s1_task5.json \
  --result results/mujoco_version_overlay_control/base350_vit_s0_task3.json \
  --result results/mujoco_version_overlay_control/overlay316_vit_s1_task5.json \
  --result results/mujoco_version_overlay_control/overlay316_vit_s0_task3.json \
  --historical results/precision_ab_vit_s1_task5_highest.json \
  --historical results/precision_ab_vit_s0_task3_highest.json \
  --output results/mujoco_version_overlay_control/summary.json
