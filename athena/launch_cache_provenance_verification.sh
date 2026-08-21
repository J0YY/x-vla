#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

declare -A caches=(
  [libero_object]=artifacts/libero_frames_100000_64.pkl
  [libero_spatial]=artifacts/libero_spatial_frames_100000_64.pkl
  [libero_goal]=artifacts/libero_goal_frames_100000_64.pkl
  [libero_10]=artifacts/libero_10_frames_100000_64.pkl
)
declare -A expected_frames=(
  [libero_object]=66984
  [libero_spatial]=52970
  [libero_goal]=52042
  [libero_10]=100000
)

for suite in libero_object libero_spatial libero_goal libero_10; do
  cache=${caches[$suite]}
  output="results/cache_provenance_${suite}.json"
  if [[ ! -f "$cache" ]]; then
    echo "Missing cache: $cache" >&2
    exit 3
  fi
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing audit: $output" >&2
    exit 3
  fi
done

for suite in libero_object libero_spatial libero_goal libero_10; do
  cache=${caches[$suite]}
  output="results/cache_provenance_${suite}.json"
  job=$(sbatch --parsable \
    --job-name="xvla-cache-audit-${suite}" \
    athena/slurm_verify_suite_cache_provenance.sbatch \
    --suite "$suite" \
    --cache "$cache" \
    --output "$output" \
    --res 64 \
    --expected-cache-frames "${expected_frames[$suite]}" \
    --max-cache-frames 100000 \
    --mismatch-example-limit 20)
  echo "$suite $job $output"
done
