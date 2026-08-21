#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail

cd /work/joy/x-vla-workshop
expected_sha256=cd87e66f267472d9b5c934804958f802e176b17618fca7f3be463db4a441a865
observed_sha256=$(sha256sum athena/summarize_multisuite_generalist.py | awk '{print $1}')
if [[ "$observed_sha256" != "$expected_sha256" ]]; then
  echo "Hardened summary SHA-256 mismatch" >&2
  exit 3
fi
python3 -u athena/summarize_multisuite_generalist.py "$@"
end_sha256=$(sha256sum athena/summarize_multisuite_generalist.py | awk '{print $1}')
if [[ "$end_sha256" != "$expected_sha256" ]]; then
  echo "Hardened summary changed during the job" >&2
  exit 3
fi
