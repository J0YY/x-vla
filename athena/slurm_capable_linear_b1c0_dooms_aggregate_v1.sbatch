#!/usr/bin/env bash
#SBATCH --job-name=xvla-dooms-aggregate
#SBATCH --partition=compute
#SBATCH --account=vgg
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/joy/x-vla-capable-linear-b1c0-dooms-v1/logs/dooms_aggregate_%j.out
#SBATCH --error=/work/joy/x-vla-capable-linear-b1c0-dooms-v1/logs/dooms_aggregate_%j.err

set -euo pipefail
if [[ "$#" -ne 0 ]]; then
  echo "Dooms aggregate accepts no arguments" >&2
  exit 2
fi
readonly lane_root=/work/joy/x-vla-capable-linear-b1c0-dooms-v1
readonly frozen_root=/work/joy/x-vla-capable-linear-b1c0-odt-v2
readonly lane_ledger="$lane_root/stage.sha256"
readonly frozen_ledger="$frozen_root/athena/capable_linear_b1c0_stage.sha256"
readonly python_bin=/users/joy/miniconda3/envs/safesae-openvla/bin/python
verify_fixed() {
  (cd "$lane_root" && sha256sum --check --strict --quiet "$lane_ledger")
  (cd "$frozen_root" && sha256sum --check --strict --quiet "$frozen_ledger")
}
trap verify_fixed EXIT
verify_fixed
cd "$lane_root"
export PYTHONPATH="$frozen_root"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
"$python_bin" -B run_capable_linear_b1c0_dooms_rollout_v1.py --mode aggregate
verify_fixed
