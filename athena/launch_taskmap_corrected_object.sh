#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

cache=artifacts/libero_frames_100000_64.pkl

submit_smoke() {
  local architecture=$1
  sbatch --parsable \
    --job-name="xvla-taskmap-${architecture}-smoke" \
    athena/slurm_train_checkpoint.sbatch \
    --architecture "$architecture" \
    --vision-encoder vit \
    --suite libero_object \
    --cache "$cache" \
    --checkpoint-output "artifacts/ckpt_taskmap_${architecture}_smoke.pt" \
    --result-output "results/train_taskmap_${architecture}_smoke.json" \
    --seed 0 \
    --steps 1 \
    --batch-size 8 \
    --lr 8e-4 \
    --ema-decay 0.999
}

submit_train() {
  local architecture=$1
  local seed=$2
  local smoke_job=$3
  sbatch --parsable \
    --dependency="afterok:${smoke_job}" \
    --job-name="xvla-taskmap-${architecture}-s${seed}" \
    athena/slurm_train_checkpoint.sbatch \
    --architecture "$architecture" \
    --vision-encoder vit \
    --suite libero_object \
    --cache "$cache" \
    --checkpoint-output "artifacts/ckpt_taskmap_${architecture}_s${seed}.pt" \
    --result-output "results/train_taskmap_${architecture}_s${seed}.json" \
    --seed "$seed" \
    --steps 40000 \
    --batch-size 256 \
    --lr 8e-4 \
    --ema-decay 0.999
}

submit_evaluations() {
  local architecture=$1
  local seed=$2
  local train_job=$3
  local checkpoint="artifacts/ckpt_taskmap_${architecture}_s${seed}.pt"
  local range start end
  for range in "0 3" "3 6" "6 8" "8 10"; do
    read -r start end <<<"$range"
    sbatch --parsable \
      --partition=low-prio-gpu \
      --qos=normal \
      --dependency="afterok:${train_job}" \
      --job-name="xvla-taskmap-${architecture}-s${seed}-${start}${end}" \
      athena/slurm_xvla.sbatch \
      --mode capability \
      --architecture "$architecture" \
      --vision-encoder vit \
      --suite libero_object \
      --training-suite libero_object \
      --checkpoint "$checkpoint" \
      --cache "$cache" \
      --output "results/taskmap_${architecture}_s${seed}_t${start}_${end}.json" \
      --seed "$seed" \
      --matmul-precision highest \
      --task-start "$start" \
      --task-end "$end" \
      --eps-per-task 50 \
      --max-steps 280 \
      --profile-iters 20
  done
}

chi_smoke=$(submit_smoke chi)
conventional_smoke=$(submit_smoke conventional)

for architecture in chi conventional; do
  if [[ "$architecture" == chi ]]; then
    smoke_job=$chi_smoke
  else
    smoke_job=$conventional_smoke
  fi
  for seed in 0 1 2; do
    train_job=$(submit_train "$architecture" "$seed" "$smoke_job")
    submit_evaluations "$architecture" "$seed" "$train_job"
    echo "$architecture seed $seed training job: $train_job"
  done
done

echo "Chi mapping smoke job: $chi_smoke"
echo "Conventional mapping smoke job: $conventional_smoke"
