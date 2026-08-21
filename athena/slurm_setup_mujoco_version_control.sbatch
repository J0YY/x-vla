#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail

base=/work/joy/safesae-openvla
clone=/work/joy/safesae-openvla-mujoco-3.1.6
result_dir=/work/joy/x-vla-workshop/results/mujoco_version_control

if [[ ! -x "$base/bin/python" ]]; then
  echo "Base environment is missing: $base" >&2
  exit 1
fi
if [[ -e "$clone" ]]; then
  echo "Refusing to overwrite existing clone: $clone" >&2
  exit 1
fi

mkdir -p "$result_dir" /work/joy/conda_pkgs_mujoco_control /work/joy/pip_cache_mujoco_control
export CONDA_PKGS_DIRS=/work/joy/conda_pkgs_mujoco_control
export PIP_CACHE_DIR=/work/joy/pip_cache_mujoco_control
export XDG_CACHE_HOME=/work/joy/cache_mujoco_control

source /athenahomes/joy/miniconda3/etc/profile.d/conda.sh
conda create --yes --copy --prefix "$clone" --clone "$base"

"$base/bin/python" -m pip freeze --all | LC_ALL=C sort > "$result_dir/base_pip_freeze.txt"
"$clone/bin/python" -m pip freeze --all | LC_ALL=C sort > "$result_dir/clone_before_pip_freeze.txt"
cmp "$result_dir/base_pip_freeze.txt" "$result_dir/clone_before_pip_freeze.txt"
echo '40eaa7969d7e11e40a3835cb500c919fc5f470080598152614c548790b302599  base_pip_freeze.txt' \
  | (cd "$result_dir" && sha256sum --check -)

mkdir -p "$result_dir/wheels"
"$base/bin/python" -m pip download \
  --disable-pip-version-check \
  --no-deps \
  --only-binary=:all: \
  --dest "$result_dir/wheels" \
  "mujoco==3.1.6"
mapfile -t mujoco_wheels < <(find "$result_dir/wheels" -maxdepth 1 -type f -name 'mujoco-3.1.6-*.whl')
if [[ "${#mujoco_wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one MuJoCo 3.1.6 wheel" >&2
  exit 1
fi
sha256sum "${mujoco_wheels[0]}" > "$result_dir/mujoco_3_1_6_wheel.sha256"
"$clone/bin/python" -m pip install \
  --disable-pip-version-check \
  --no-deps \
  --no-index \
  "${mujoco_wheels[0]}"
"$clone/bin/python" -m pip freeze --all | LC_ALL=C sort > "$result_dir/clone_after_pip_freeze.txt"

grep -vi '^mujoco==' "$result_dir/base_pip_freeze.txt" > "$result_dir/base_without_mujoco.txt"
grep -vi '^mujoco==' "$result_dir/clone_after_pip_freeze.txt" > "$result_dir/clone_without_mujoco.txt"
cmp "$result_dir/base_without_mujoco.txt" "$result_dir/clone_without_mujoco.txt"
grep -ix 'mujoco==3.5.0' "$result_dir/base_pip_freeze.txt"
grep -ix 'mujoco==3.1.6' "$result_dir/clone_after_pip_freeze.txt"

set +e
"$base/bin/python" -m pip check > "$result_dir/base_pip_check.txt" 2>&1
base_check_status=$?
"$clone/bin/python" -m pip check > "$result_dir/clone_pip_check.txt" 2>&1
clone_check_status=$?
set -e
cmp "$result_dir/base_pip_check.txt" "$result_dir/clone_pip_check.txt"
if [[ "$base_check_status" -ne "$clone_check_status" ]]; then
  echo "pip check status differs across environments" >&2
  exit 1
fi

conda list --explicit --prefix "$base" > "$result_dir/base_conda_explicit.txt"
conda list --explicit --prefix "$clone" > "$result_dir/clone_conda_explicit.txt"

"$clone/bin/python" - "$base" "$clone" "$result_dir" "${SLURM_JOB_ID:-unknown}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
clone = Path(sys.argv[2])
result_dir = Path(sys.argv[3])
job_id = sys.argv[4]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

payload = {
    "schema_version": "xvla-mujoco-version-env-setup-v1",
    "job_id": job_id,
    "base_prefix": str(base),
    "clone_prefix": str(clone),
    "base_pip_freeze_sha256": digest(result_dir / "base_pip_freeze.txt"),
    "clone_before_pip_freeze_sha256": digest(result_dir / "clone_before_pip_freeze.txt"),
    "clone_after_pip_freeze_sha256": digest(result_dir / "clone_after_pip_freeze.txt"),
    "base_without_mujoco_sha256": digest(result_dir / "base_without_mujoco.txt"),
    "clone_without_mujoco_sha256": digest(result_dir / "clone_without_mujoco.txt"),
    "only_pip_freeze_change_is_mujoco": True,
    "base_mujoco_version": "3.5.0",
    "clone_mujoco_version": "3.1.6",
    "base_pip_check_sha256": digest(result_dir / "base_pip_check.txt"),
    "clone_pip_check_sha256": digest(result_dir / "clone_pip_check.txt"),
    "mujoco_3_1_6_wheel_sha256": digest(next((result_dir / "wheels").glob("mujoco-3.1.6-*.whl"))),
}
target = result_dir / "environment_setup.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(target)
print(json.dumps(payload, sort_keys=True))
PY
