#!/bin/bash
#SBATCH --job-name=xvla-product-odt-stage
#SBATCH --partition=compute
#SBATCH --account=vgg
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/joy/x-vla-workshop/athena/logs/product_odt_stage_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/athena/logs/product_odt_stage_%j.err

set -euo pipefail
if [[ "$#" -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: sbatch --dependency=afterok:SEED0_JOB $0 CAPABILITY_AGGREGATE_JOB" >&2
  exit 2
fi
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTEST_ADDOPTS PYTEST_PLUGINS
prestage_root=/work/joy/x-vla-product-pade-rational-odt-prestage-v3
stage_root=/work/joy/x-vla-product-pade-rational-odt-v3
python_bin=/users/joy/miniconda3/envs/safesae-openvla/bin/python
cd "$prestage_root"
"$python_bin" -B athena/stage_product_rational_exact_odt.py
exec /bin/bash "$stage_root/athena/submit_product_rational_exact_odt.sh" "$1"
