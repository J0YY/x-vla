#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/work/joy/x-vla-workshop/logs/%x_%j.out
#SBATCH --error=/work/joy/x-vla-workshop/logs/%x_%j.err

set -euo pipefail

cd /work/joy/x-vla-workshop
expected_sha256=f5414d7a3348103c7fc27b5c58abcc4413b7f6df3a238982ebcddac63a3bdc3a
observed_sha256=$(sha256sum athena/summarize_specialist_retention_hardened.py | awk '{print $1}')
if [[ "$observed_sha256" != "$expected_sha256" ]]; then
  echo "Hardened specialist-retention summary SHA-256 mismatch" >&2
  exit 3
fi
python3 -u athena/summarize_specialist_retention_hardened.py "$@"
end_sha256=$(sha256sum athena/summarize_specialist_retention_hardened.py | awk '{print $1}')
if [[ "$end_sha256" != "$expected_sha256" ]]; then
  echo "Hardened specialist-retention summary changed during the job" >&2
  exit 3
fi
