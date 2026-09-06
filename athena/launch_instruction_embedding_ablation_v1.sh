#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

readonly phase=${1:-}
shift || true
if [[ "$phase" != "preflight-smoke" && "$phase" != "full" ]]; then
  echo "Usage: $0 preflight-smoke | full PRECHECK_JOB SMOKE_JOB" >&2
  exit 2
fi

readonly cache=artifacts/libero_frames_100000_64.pkl
readonly provenance=results/cache_provenance_libero_object.json
readonly manifest=artifacts/instruction_embedding_ablation_v1_manifest.json
readonly smoke_result=results/instruction_embedding_ablation_v1_smoke.json
readonly summary_result=results/instruction_embedding_ablation_v1_summary.json
readonly python_executable=/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python

declare -Ar checkpoints=(
  [0]=artifacts/ckpt_linear_rat_vit_s0_v2.pt
  [1]=artifacts/ckpt_linear_rat_vit_s1.pt
  [2]=artifacts/ckpt_linear_rat_vit_s2.pt
)
declare -Ar checkpoint_shas=(
  [0]=96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9
  [1]=cc0780b989165a80a449c03cbb3574e2f56b2e5747a64cf32d1fde418df8ec3c
  [2]=413a770071bd8f16b924c5604f7c7a58567d2eb6c40c66aab202bcf046fec910
)
declare -Ar expected_shas=(
  [athena/counterfactual_target_swap_common.py]=26d185f15f787ac292a3457cb66b9773cd4281f3ee4050fc9971a6dd4d817b81
  [athena/preflight_counterfactual_target_swap.py]=06bc40b323e741d229bf54c19a288fd71537b9fd756140b9408a394d49e7c19e
  [athena/run_counterfactual_target_swap.py]=298a4b2bef1501c10e00d4818fc73b69cbc1f9e6aaa26e580ef8f290938f233a
  [athena/INSTRUCTION_EMBEDDING_ABLATION_V1_PROTOCOL.md]=f4c7ae93561bfbb56bb42e4b648d88f95eef76b0931a13cb998547cbe2009669
  [athena/instruction_embedding_ablation_v1_common.py]=8a8992f3fc5cdbbc92db2bfbf69c09e03f2d23d313a4155ab209d4a067c62865
  [athena/preflight_instruction_embedding_ablation_v1.py]=6f40ae57097bb5287092f2f605d85085f2fb9762d6f1dd83fc81dd8b8eb9f85c
  [athena/run_instruction_embedding_ablation_v1.py]=ef645a3175e6659d199ae9716ee2e98f56568371bea1781a04193e7822a76ff7
  [athena/summarize_instruction_embedding_ablation_v1.py]=f633b8d54bc94b4ce7f593d216245195445d38e5e6ba945bf60ac657f24f8d78
  [athena/slurm_preflight_instruction_embedding_ablation_v1.sbatch]=4cc800f31b43763cd5eaa2294f1648a6a597ddc04747268b95054f0b528b4f3d
  [athena/slurm_instruction_embedding_ablation_v1.sbatch]=18c359fc67bc958f56418c3d2c141f6abe80550a375220d7867b7363c4c4eb97
  [athena/slurm_summarize_instruction_embedding_ablation_v1.sbatch]=b3438c337a902f3f77b061650f415aad8304e6ac2e523a0307e3a223564c9e5c
)

[[ -x "$python_executable" ]] || { echo "Frozen Python executable is absent" >&2; exit 3; }
for path in "${!expected_shas[@]}"; do
  [[ -f "$path" ]] || { echo "Missing frozen source: $path" >&2; exit 3; }
  observed=$(sha256sum "$path" | cut -d' ' -f1)
  [[ "$observed" == "${expected_shas[$path]}" ]] || {
    echo "Stale frozen source: $path" >&2
    exit 3
  }
done
for seed in 0 1 2; do
  path=${checkpoints[$seed]}
  [[ -f "$path" ]] || { echo "Missing checkpoint: $path" >&2; exit 3; }
  [[ "$(sha256sum "$path" | cut -d' ' -f1)" == "${checkpoint_shas[$seed]}" ]] || {
    echo "Checkpoint identity differs for seed $seed" >&2
    exit 3
  }
done
for path in "$cache" "$provenance"; do
  [[ -f "$path" ]] || { echo "Missing frozen input: $path" >&2; exit 3; }
done
if [[ "$phase" == "preflight-smoke" ]]; then
  [[ "$#" -eq 0 ]] || { echo "preflight-smoke takes no job arguments" >&2; exit 2; }
  for path in "$manifest" "$smoke_result" "$summary_result"; do
    [[ ! -e "$path" ]] || { echo "Refusing to overwrite: $path" >&2; exit 3; }
  done
  for seed in 0 1 2; do
    for shard in 0_2 2_4 4_6 6_8 8_10; do
      path="results/instruction_embedding_ablation_v1_s${seed}_t${shard}.json"
      [[ ! -e "$path" ]] || { echo "Refusing to overwrite: $path" >&2; exit 3; }
    done
  done

  preflight_job=$(sbatch --parsable \
    --job-name=xvla-instruction-embedding-ablation-v1-preflight \
    athena/slurm_preflight_instruction_embedding_ablation_v1.sbatch \
    --cache "$cache" \
    --provenance-result "$provenance" \
    --output "$manifest")

  smoke_job=$(sbatch --parsable \
    --dependency="afterok:${preflight_job}" \
    --job-name=xvla-instruction-embedding-ablation-v1-smoke \
    athena/slurm_instruction_embedding_ablation_v1.sbatch \
    --mode strict_smoke \
    --checkpoint-seed 0 \
    --checkpoint "${checkpoints[0]}" \
    --cache "$cache" \
    --provenance-result "$provenance" \
    --manifest "$manifest" \
    --task-start 0 --task-end 10 \
    --episode-start 30 --eps-per-task 1 \
    --expected-rollouts 0 \
    --res 64 --horizon 8 --exec-h 8 --max-steps 280 --num-steps-wait 10 \
    --matmul-precision highest \
    --output "$smoke_result")

  echo "preflight $preflight_job"
  echo "smoke $smoke_job afterok:$preflight_job"
  exit 0
fi

[[ "$#" -eq 2 ]] || { echo "full requires PRECHECK_JOB SMOKE_JOB" >&2; exit 2; }
readonly preflight_job=$1
readonly smoke_job=$2
[[ "$preflight_job" =~ ^[0-9]+$ && "$smoke_job" =~ ^[0-9]+$ ]] || {
  echo "Preflight and smoke job IDs must be numeric" >&2
  exit 2
}
preflight_state=$(sacct -j "$preflight_job" --noheader --parsable2 --allocations --format=State | head -1 | cut -d'|' -f1)
smoke_state=$(sacct -j "$smoke_job" --noheader --parsable2 --allocations --format=State | head -1 | cut -d'|' -f1)
[[ "$preflight_state" == COMPLETED && "$smoke_state" == COMPLETED ]] || {
  echo "Preflight/smoke not both complete: $preflight_state $smoke_state" >&2
  exit 4
}
[[ -s "$manifest" && -s "$smoke_result" ]] || {
  echo "Completed preflight/smoke outputs are absent" >&2
  exit 4
}
[[ ! -e "$summary_result" ]] || { echo "Refusing to overwrite: $summary_result" >&2; exit 3; }
for seed in 0 1 2; do
  for shard in 0_2 2_4 4_6 6_8 8_10; do
    path="results/instruction_embedding_ablation_v1_s${seed}_t${shard}.json"
    [[ ! -e "$path" ]] || { echo "Refusing to overwrite: $path" >&2; exit 3; }
  done
done

"$python_executable" - "$manifest" "$smoke_result" <<'PY'
import json
import sys
from pathlib import Path

from athena.instruction_embedding_ablation_v1_common import (
    CONDITIONS,
    DESIGN_PROVENANCE,
    FROZEN_GATES,
    PROTOCOL,
    SCHEMA_RUN,
    condition_order,
    source_hashes,
    validate_manifest,
)

root = Path.cwd()
manifest = validate_manifest(Path(sys.argv[1]), root)
smoke = json.loads(Path(sys.argv[2]).read_text())
assert smoke["schema"] == SCHEMA_RUN and smoke["mode"] == "strict_smoke"
assert smoke["protocol"] == PROTOCOL and smoke["frozen_gates"] == FROZEN_GATES
assert smoke["design_provenance"] == DESIGN_PROVENANCE
assert smoke["identity"]["source_sha256_start"] == source_hashes(root)
assert smoke["identity"]["source_sha256_end"] == source_hashes(root)
assert smoke["identity"]["manifest_sha256"]
assert smoke["evaluation"] == {
    "task_start": 0,
    "task_end": 10,
    "episode_start": 30,
    "eps_per_task": 1,
    "expected_rollouts": 0,
    "completed_rollouts": 0,
    "completed_checkpoint_state_pairs": 0,
    "smoke_states": 10,
    "matmul_precision": "highest",
}
assert len(smoke["smoke_rows"]) == 10 and not smoke["checkpoint_state_pairs"]
for row in smoke["smoke_rows"]:
    task = int(row["task_index"])
    episode = int(row["episode"])
    assert episode == 30
    assert row["condition_order"] == condition_order(0, task, episode)
    assert set(row["conditions"]) == set(CONDITIONS)
    assert row["conditions"]["full_lexical_embeddings"]["ablation_audit"]["hook_calls"] == 0
    ablated = row["conditions"]["zeroed_lexical_embeddings"]["ablation_audit"]
    assert ablated["hook_calls"] == 1
    assert ablated["all_returned_outputs_exactly_zero"] is True
    assert all(call["returned_nonzero_elements"] == 0 for call in ablated["calls"])
print("strict_smoke_gate_ok", len(smoke["smoke_rows"]), manifest["mapping_count"])
PY

declare -a full_jobs=()
declare -a result_args=()
index=0
for seed in 0 1 2; do
  for bounds in "0 2" "2 4" "4 6" "6 8" "8 10"; do
    read -r task_start task_end <<< "$bounds"
    result="results/instruction_embedding_ablation_v1_s${seed}_t${task_start}_${task_end}.json"
    full_jobs[$index]=$(sbatch --parsable \
      --dependency="afterok:${smoke_job}" \
      --job-name="xvla-instruction-embedding-ablation-v1-s${seed}-t${task_start}-${task_end}" \
      athena/slurm_instruction_embedding_ablation_v1.sbatch \
      --mode full \
      --checkpoint-seed "$seed" \
      --checkpoint "${checkpoints[$seed]}" \
      --cache "$cache" \
      --provenance-result "$provenance" \
      --manifest "$manifest" \
      --task-start "$task_start" --task-end "$task_end" \
      --episode-start 30 --eps-per-task 10 \
      --expected-rollouts 40 \
      --res 64 --horizon 8 --exec-h 8 --max-steps 280 --num-steps-wait 10 \
      --matmul-precision highest \
      --output "$result")
    result_args+=(--result "$result")
    ((index += 1))
  done
done

dependency="afterok:$(IFS=:; echo "${full_jobs[*]}")"
summary_job=$(sbatch --parsable \
  --dependency="$dependency" \
  --job-name=xvla-instruction-embedding-ablation-v1-summary \
  athena/slurm_summarize_instruction_embedding_ablation_v1.sbatch \
  --smoke "$smoke_result" \
  "${result_args[@]}" \
  --manifest "$manifest" \
  --output "$summary_result")

echo "validated_preflight $preflight_job $preflight_state"
echo "validated_smoke $smoke_job $smoke_state"
for job_index in "${!full_jobs[@]}"; do echo "full_${job_index} ${full_jobs[$job_index]}"; done
echo "summary $summary_job"
