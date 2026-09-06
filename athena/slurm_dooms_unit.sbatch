#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail
readonly run_root="$1"
readonly data_root="$2"
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python
cd "$run_root"
export PYTHONPATH="$run_root"
"$python_executable" -m pytest -q \
  tests/test_canonical_odt.py \
  tests/test_dooms_xnet.py \
  tests/test_dooms_svhn_reproduction.py \
  | tee "$run_root/results/unit_tests.log"
"$python_executable" -u athena/dooms_svhn_reproduction.py unit \
  --run-root "$run_root" --data-root "$data_root" \
  --test-log "$run_root/results/unit_tests.log"
