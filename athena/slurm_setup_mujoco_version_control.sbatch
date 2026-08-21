#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail

cd /work/joy/x-vla-workshop

base=/work/joy/safesae-openvla
overlay=/work/joy/xvla-mujoco-3.1.6-overlay
result_dir=/work/joy/x-vla-workshop/results/mujoco_version_overlay_control
expected_base_freeze=c7e80877c4d26ec8c1db88d68e3d610c0b5174a802192b6f5d195ba4582952a8
expected_wheel=mujoco-3.1.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
expected_wheel_sha256=f4c2bbc5025bd0a06e939397188be9dc4a6b7d98aa39176d43a302a5811c0b12

if [[ ! -x "$base/bin/python" ]]; then
  echo "Base environment is missing: $base" >&2
  exit 1
fi
if [[ -e "$overlay" ]]; then
  echo "Refusing to overwrite existing package overlay: $overlay" >&2
  exit 1
fi
if [[ -e "$result_dir" ]]; then
  echo "Refusing to overwrite existing result directory: $result_dir" >&2
  exit 1
fi

mkdir -p "$result_dir/wheels" /work/joy/pip_cache_mujoco_control
export PIP_CACHE_DIR=/work/joy/pip_cache_mujoco_control
export XDG_CACHE_HOME=/work/joy/cache_mujoco_control
export PYTHONNOUSERSITE=1

env -u PYTHONPATH -u XVLA_MUJOCO_OVERLAY \
  "$base/bin/python" -m pip freeze --all | LC_ALL=C sort \
  > "$result_dir/base_pip_freeze.txt"
echo "$expected_base_freeze  base_pip_freeze.txt" \
  | (cd "$result_dir" && sha256sum --check -)

env -u PYTHONPATH -u XVLA_MUJOCO_OVERLAY \
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
if [[ "$(basename "${mujoco_wheels[0]}")" != "$expected_wheel" ]]; then
  echo "Unexpected MuJoCo wheel: ${mujoco_wheels[0]}" >&2
  exit 1
fi
actual_wheel_sha256=$(sha256sum "${mujoco_wheels[0]}" | cut -d' ' -f1)
if [[ "$actual_wheel_sha256" != "$expected_wheel_sha256" ]]; then
  echo "Frozen MuJoCo wheel SHA mismatch: $actual_wheel_sha256" >&2
  exit 1
fi
sha256sum "${mujoco_wheels[0]}" > "$result_dir/mujoco_3_1_6_wheel.sha256"

env -u PYTHONPATH -u XVLA_MUJOCO_OVERLAY \
  "$base/bin/python" -m pip install \
  --disable-pip-version-check \
  --no-deps \
  --no-index \
  --target "$overlay" \
  "${mujoco_wheels[0]}"

env -u PYTHONPATH -u XVLA_MUJOCO_OVERLAY \
  "$base/bin/python" -m pip freeze --all | LC_ALL=C sort \
  > "$result_dir/base_pip_freeze_after_overlay.txt"
cmp "$result_dir/base_pip_freeze.txt" "$result_dir/base_pip_freeze_after_overlay.txt"

env PYTHONPATH="$overlay" XVLA_MUJOCO_OVERLAY="$overlay" \
  "$base/bin/python" -m pip freeze --all | LC_ALL=C sort \
  > "$result_dir/overlay_pip_freeze.txt"
grep -vi '^mujoco==' "$result_dir/base_pip_freeze.txt" \
  > "$result_dir/base_without_mujoco.txt"
grep -vi '^mujoco==' "$result_dir/overlay_pip_freeze.txt" \
  > "$result_dir/overlay_without_mujoco.txt"
cmp "$result_dir/base_without_mujoco.txt" "$result_dir/overlay_without_mujoco.txt"
grep -ix 'mujoco==3.5.0' "$result_dir/base_pip_freeze.txt"
grep -ix 'mujoco==3.1.6' "$result_dir/overlay_pip_freeze.txt"

set +e
env -u PYTHONPATH -u XVLA_MUJOCO_OVERLAY \
  "$base/bin/python" -m pip check > "$result_dir/base_pip_check.txt" 2>&1
base_check_status=$?
env PYTHONPATH="$overlay" XVLA_MUJOCO_OVERLAY="$overlay" \
  "$base/bin/python" -m pip check > "$result_dir/overlay_pip_check.txt" 2>&1
overlay_check_status=$?
set -e
cmp "$result_dir/base_pip_check.txt" "$result_dir/overlay_pip_check.txt"
if [[ "$base_check_status" -ne "$overlay_check_status" ]]; then
  echo "pip check status differs between base and overlay environments" >&2
  exit 1
fi

write_runtime_identity() {
  local output=$1
  local overlay_path=$2
  local -a environment_command
  if [[ -n "$overlay_path" ]]; then
    environment_command=(env PYTHONPATH="$overlay_path" XVLA_MUJOCO_OVERLAY="$overlay_path")
  else
    environment_command=(env -u PYTHONPATH -u XVLA_MUJOCO_OVERLAY)
  fi
  "${environment_command[@]}" "$base/bin/python" - "$base" "$overlay_path" "$output" <<'PY'
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

base = Path(sys.argv[1]).resolve()
overlay_value = sys.argv[2]
overlay = Path(overlay_value).resolve() if overlay_value else None
output = Path(sys.argv[3])

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def distribution_identity(name):
    distribution = importlib.metadata.distribution(name)
    files = []
    native = {}
    record_sha256 = None
    for relative in sorted(distribution.files or [], key=str):
        path = Path(distribution.locate_file(relative)).resolve()
        if path.suffix in (".pyc", ".pyo") or "__pycache__" in path.parts:
            continue
        if not path.is_file():
            continue
        file_sha256 = digest(path)
        files.append((str(relative), file_sha256))
        if str(relative).endswith(".dist-info/RECORD"):
            record_sha256 = file_sha256
        if path.suffix in (".so", ".dylib", ".dll", ".pyd") or ".so." in path.name:
            native[str(path)] = file_sha256
    tree_material = "\n".join(f"{relative}\0{value}" for relative, value in files)
    return {
        "version": distribution.version,
        "record_sha256": record_sha256,
        "distribution_tree_sha256": hashlib.sha256(tree_material.encode("utf-8")).hexdigest(),
        "native_library_sha256": native,
    }

import mujoco

payload = {
    "base_prefix": str(base),
    "overlay_prefix": str(overlay) if overlay is not None else None,
    "python_executable": str(Path(sys.executable).resolve()),
    "python_executable_sha256": digest(sys.executable),
    "python_version": sys.version,
    "mujoco_module": str(Path(mujoco.__file__).resolve()),
    "mujoco_runtime_version": str(mujoco.__version__),
    "mujoco_distribution": distribution_identity("mujoco"),
}
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(output)
PY
}

write_runtime_identity "$result_dir/base_runtime_identity.json" ""
write_runtime_identity "$result_dir/overlay_runtime_identity.json" "$overlay"

"$base/bin/python" - \
  "$base" \
  "$overlay" \
  "$result_dir" \
  "${SLURM_JOB_ID:-unknown}" \
  "$base_check_status" \
  "$overlay_check_status" \
  "$expected_base_freeze" \
  "$expected_wheel" \
  "$expected_wheel_sha256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
overlay = Path(sys.argv[2])
result_dir = Path(sys.argv[3])
job_id = sys.argv[4]
base_check_status = int(sys.argv[5])
overlay_check_status = int(sys.argv[6])
expected_base_freeze = sys.argv[7]
expected_wheel = sys.argv[8]
expected_wheel_sha256 = sys.argv[9]

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def directory_identity(root):
    manifest = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix in (".pyc", ".pyo") or "__pycache__" in path.parts:
            continue
        manifest[str(path.relative_to(root))] = digest(path)
    material = "\n".join(f"{relative}\0{value}" for relative, value in manifest.items())
    return {
        "file_count": len(manifest),
        "tree_sha256": hashlib.sha256(material.encode("utf-8")).hexdigest(),
        "file_sha256": manifest,
    }

payload = {
    "schema_version": "xvla-mujoco-version-env-setup-v3",
    "job_id": job_id,
    "setup_source_sha256": digest(Path("athena/slurm_setup_mujoco_version_control.sbatch")),
    "base_prefix": str(base),
    "overlay_prefix": str(overlay),
    "base_pip_freeze_sha256": digest(result_dir / "base_pip_freeze.txt"),
    "base_pip_freeze_after_overlay_sha256": digest(result_dir / "base_pip_freeze_after_overlay.txt"),
    "overlay_pip_freeze_sha256": digest(result_dir / "overlay_pip_freeze.txt"),
    "base_without_mujoco_sha256": digest(result_dir / "base_without_mujoco.txt"),
    "overlay_without_mujoco_sha256": digest(result_dir / "overlay_without_mujoco.txt"),
    "only_pip_freeze_change_is_mujoco": True,
    "expected_base_pip_freeze_sha256": expected_base_freeze,
    "base_mujoco_version": "3.5.0",
    "overlay_mujoco_version": "3.1.6",
    "base_pip_check_status": base_check_status,
    "overlay_pip_check_status": overlay_check_status,
    "pip_check_status_and_output_equal": True,
    "base_pip_check_sha256": digest(result_dir / "base_pip_check.txt"),
    "overlay_pip_check_sha256": digest(result_dir / "overlay_pip_check.txt"),
    "mujoco_3_1_6_wheel_filename": expected_wheel,
    "expected_mujoco_3_1_6_wheel_sha256": expected_wheel_sha256,
    "mujoco_3_1_6_wheel_sha256": digest(result_dir / "wheels" / expected_wheel),
    "base_runtime_identity": json.loads((result_dir / "base_runtime_identity.json").read_text()),
    "overlay_runtime_identity": json.loads((result_dir / "overlay_runtime_identity.json").read_text()),
    "overlay_directory_identity": directory_identity(overlay),
    "base_environment_mutated": False,
    "isolation_mechanism": "base_python_with_package_only_overlay_first_on_pythonpath",
}
target = result_dir / "environment_setup.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(target)
print(json.dumps(payload, sort_keys=True))
PY
