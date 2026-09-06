#!/bin/bash
#SBATCH --job-name=xvla-product-odt-composite
#SBATCH --partition=compute
#SBATCH --account=vgg
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/joy/x-vla-product-pade-rational-odt-v3/athena/logs/product_odt_composite_%j.out
#SBATCH --error=/work/joy/x-vla-product-pade-rational-odt-v3/athena/logs/product_odt_composite_%j.err

set -euo pipefail
if [[ "$#" -ne 0 ]]; then
  echo "Product composite job accepts no arguments" >&2
  exit 2
fi
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
unset PYTEST_ADDOPTS PYTEST_PLUGINS
stage_root=/work/joy/x-vla-product-pade-rational-odt-v3
python_bin=/users/joy/miniconda3/envs/safesae-openvla/bin/python
cd "$stage_root"
exec "$python_bin" -B athena/run_product_rational_exact_odt.py \
  --mode composite \
  --source-manifest athena/product_rational_exact_odt_sources.json
