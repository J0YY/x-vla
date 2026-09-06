#!/bin/bash
#SBATCH --job-name=xvla-product-odt-validate
#SBATCH --partition=compute
#SBATCH --account=vgg
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/joy/x-vla-product-pade-rational-odt-v3/athena/logs/product_odt_validate_%j.out
#SBATCH --error=/work/joy/x-vla-product-pade-rational-odt-v3/athena/logs/product_odt_validate_%j.err

set -euo pipefail
if [[ "$#" -ne 1 || ( "$1" != "preflight" && "$1" != "oracle" && "$1" != "lane-tests" ) ]]; then
  echo "usage: $0 preflight|oracle|lane-tests" >&2
  exit 2
fi
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
unset PYTEST_ADDOPTS PYTEST_PLUGINS
stage_root=/work/joy/x-vla-product-pade-rational-odt-v3
python_bin=/users/joy/miniconda3/envs/safesae-openvla/bin/python
cd "$stage_root"
exec "$python_bin" -B athena/run_product_rational_exact_odt.py \
  --mode "$1" \
  --source-manifest athena/product_rational_exact_odt_sources.json
