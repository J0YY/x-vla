#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop
export PYTHONPATH="/work/joy/x-vla-workshop:${PYTHONPATH:-}"

manifest=artifacts/multisuite_mean_ensemble_v1_manifest.json
summary=results/multisuite_mean_ensemble_v1_summary.json

declare -a protected_outputs=("$manifest" "$summary")
for suite in libero_object libero_spatial libero_goal libero_10; do
  for range in "0 3" "3 6" "6 8" "8 10"; do
    read -r start end <<<"$range"
    protected_outputs+=("results/multisuite_mean_ensemble_v1_${suite}_t${start}_${end}.json")
  done
done
for output in "${protected_outputs[@]}"; do
  if [[ -e "$output" || -e "${output}.tmp" ]]; then
    echo "Refusing to reuse ensemble output: $output" >&2
    exit 3
  fi
done

python3 -u athena/build_multisuite_ensemble_manifest.py \
  --artifacts-dir artifacts \
  --results-dir results \
  --output "$manifest"
manifest_sha256=$(sha256sum "$manifest" | awk '{print $1}')

declare -a jobs=()
for suite in libero_object libero_spatial libero_goal libero_10; do
  for range in "0 3" "3 6" "6 8" "8 10"; do
    read -r start end <<<"$range"
    job=$(sbatch --parsable \
      --job-name="xvla-multisuite-ensemble-${suite}-${start}${end}" \
      athena/slurm_eval_multisuite_ensemble.sbatch \
      "$manifest" "$manifest_sha256" "$suite" "$start" "$end")
    jobs+=("$job")
    echo "ensemble $suite tasks $start:$end job: $job"
  done
done

dependency=$(IFS=:; echo "${jobs[*]}")
summary_job=$(sbatch --parsable \
  --dependency="afterok:${dependency}" \
  --job-name=xvla-multisuite-ensemble-summary \
  athena/slurm_summarize_multisuite_ensemble.sbatch \
  "$manifest" "$manifest_sha256")

echo "ensemble manifest SHA-256: $manifest_sha256"
echo "ensemble evaluation jobs: ${jobs[*]}"
echo "ensemble summary job: $summary_job"
