#!/usr/bin/env bash
# Submit the authenticated ProductRoutingHead capability DAG after CPU preflight.

set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "This frozen submission script accepts no arguments" >&2
  exit 2
fi

readonly project_root=/work/joy/x-vla-product-pade-rational-v2
readonly run_root="$project_root/athena/results/product_pade_rational_v2"
readonly python_executable=/users/joy/miniconda3/envs/safesae-openvla/bin/python
cd "$project_root"
export PYTHONNOUSERSITE=1

"$python_executable" -B athena/prepare_product_rational_launch.py \
  --action verify-preflight \
  --run-root athena/results/product_pade_rational_v2

mkdir "$run_root/gates/submission.lock"

smoke_train_raw=$(sbatch --parsable --time=02:00:00 \
  athena/slurm_product_rational_train.sbatch smoke 0)
smoke_train=${smoke_train_raw%%;*}
smoke_eval_raw=$(sbatch --parsable --dependency="afterok:$smoke_train" \
  --time=01:00:00 athena/slurm_product_rational_eval.sbatch smoke 0 0 1)
smoke_eval=${smoke_eval_raw%%;*}

declare -a train_jobs=()
declare -a eval_jobs=()
for seed in 0 1 2; do
  train_raw=$(sbatch --parsable --dependency="afterok:$smoke_eval" \
    athena/slurm_product_rational_train.sbatch full "$seed")
  train_job=${train_raw%%;*}
  train_jobs+=("$train_job")
  for shard in "0 3" "3 6" "6 8" "8 10"; do
    read -r task_start task_end <<<"$shard"
    eval_raw=$(sbatch --parsable --dependency="afterok:$train_job" \
      athena/slurm_product_rational_eval.sbatch full "$seed" "$task_start" "$task_end")
    eval_job=${eval_raw%%;*}
    eval_jobs+=("$eval_job")
  done
done

dependency=afterok
for job in "${eval_jobs[@]}"; do
  dependency="$dependency:$job"
done
aggregate_raw=$(sbatch --parsable --dependency="$dependency" \
  athena/slurm_product_rational_aggregate.sbatch)
aggregate_job=${aggregate_raw%%;*}

echo "smoke_train=$smoke_train"
echo "smoke_eval=$smoke_eval"
echo "full_train=${train_jobs[*]}"
echo "full_eval=${eval_jobs[*]}"
echo "aggregate=$aggregate_job"
