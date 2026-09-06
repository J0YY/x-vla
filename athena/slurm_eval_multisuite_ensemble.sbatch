#!/usr/bin/env bash
#SBATCH --partition=low-prio-gpu
#SBATCH --qos=normal
#SBATCH --constraint=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 MANIFEST MANIFEST_SHA256 SUITE TASK_START TASK_END" >&2
  exit 2
fi

ensemble_manifest=$1
ensemble_manifest_sha256=$2
suite=$3
task_start=$4
task_end=$5

source /athenahomes/joy/miniconda3/etc/profile.d/conda.sh
conda activate safesae-openvla

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/work/joy/hf_cache
export PYTHONPATH="/work/joy/x-vla-workshop:/athenahomes/joy/LIBERO:${PYTHONPATH:-}"
export XVLA_FROZEN_EVALUATOR_SHA256=91ae342892e32b0aa019a43ff06b5809dca2a49d4be20ce9a55fbad1b15cdbb3
export XVLA_FROZEN_EVALUATION_GPU_FAMILY=a6000

cd /work/joy/x-vla-workshop

cache_for_suite() {
  case "$1" in
    libero_object) echo artifacts/libero_frames_100000_64.pkl ;;
    libero_spatial) echo artifacts/libero_spatial_frames_100000_64.pkl ;;
    libero_goal) echo artifacts/libero_goal_frames_100000_64.pkl ;;
    libero_10) echo artifacts/libero_10_frames_100000_64.pkl ;;
    *) return 1 ;;
  esac
}

checkpoint0=artifacts/ckpt_generalist_hardened_chi_s0.pt
checkpoint1=artifacts/ckpt_generalist_hardened_chi_s1.pt
checkpoint2=artifacts/ckpt_generalist_hardened_chi_s2.pt
metadata0=artifacts/ckpt_generalist_hardened_chi_s0.json
cache=$(cache_for_suite "$suite")
output="results/multisuite_mean_ensemble_v1_${suite}_t${task_start}_${task_end}.json"

python3 - "$ensemble_manifest" "$ensemble_manifest_sha256" <<'PY'
import hashlib, json, pathlib, sys
manifest_path = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if digest != expected:
    raise RuntimeError(f"ensemble manifest mismatch: {digest} != {expected}")
manifest = json.loads(manifest_path.read_text())
for path, frozen in manifest["source_files"].items():
    observed = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    if observed != frozen:
        raise RuntimeError(f"frozen ensemble source mismatch: {path}")
for artifact in manifest["artifacts"].values():
    for key, digest_key in (("checkpoint_path", "checkpoint_sha256"), ("metadata_path", "metadata_sha256")):
        observed = hashlib.sha256(pathlib.Path(artifact[key]).read_bytes()).hexdigest()
        if observed != artifact[digest_key]:
            raise RuntimeError(f"frozen ensemble artifact mismatch: {artifact[key]}")
PY

if [[ -e "$output" || -e "${output}.tmp" ]]; then
  echo "Refusing to overwrite ensemble output: $output" >&2
  exit 3
fi

python -u athena/run_xvla_experiment.py \
  --mode ensemble_capability \
  --architecture chi \
  --vision-encoder vit \
  --suite "$suite" \
  --training-suite "$suite" \
  --checkpoint "$checkpoint0" \
  --ensemble-checkpoint "$checkpoint1" \
  --ensemble-checkpoint "$checkpoint2" \
  --ensemble-reduction mean \
  --model-metadata "$metadata0" \
  --cache "$cache" \
  --output "$output" \
  --seed 0 \
  --matmul-precision highest \
  --task-start "$task_start" \
  --task-end "$task_end" \
  --eps-per-task 50 \
  --max-steps 280 \
  --profile-iters 20

python3 - "$ensemble_manifest" "$ensemble_manifest_sha256" "$output" <<'PY'
import hashlib, json, pathlib, sys
manifest_path = pathlib.Path(sys.argv[1])
manifest_sha256 = sys.argv[2]
output_path = pathlib.Path(sys.argv[3])
manifest = json.loads(manifest_path.read_text())

def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if sha(manifest_path) != manifest_sha256:
    raise RuntimeError("ensemble manifest changed during evaluation")
sources_end = {path: sha(path) for path in manifest["source_files"]}
if sources_end != manifest["source_files"]:
    raise RuntimeError("ensemble sources changed during evaluation")
members_end = {
    seed: sha(artifact["checkpoint_path"])
    for seed, artifact in manifest["artifacts"].items()
}
members_expected = {
    seed: artifact["checkpoint_sha256"]
    for seed, artifact in manifest["artifacts"].items()
}
if members_end != members_expected:
    raise RuntimeError("ensemble checkpoints changed during evaluation")
payload = json.loads(output_path.read_text())
payload["ensemble_run_identity"] = {
    "manifest_sha256": manifest_sha256,
    "source_files_start": manifest["source_files"],
    "source_files_end": sources_end,
    "member_checkpoint_sha256_start": members_expected,
    "member_checkpoint_sha256_end": members_end,
}
temporary = output_path.with_suffix(output_path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(output_path)
PY
