#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

cache=artifacts/libero_frames_100000_64.pkl
checkpoints=(
  artifacts/ckpt_linear_rat_vit_s0_v2.pt
  artifacts/ckpt_linear_rat_vit_s1.pt
  artifacts/ckpt_linear_rat_vit_s2.pt
)

# This harder control is frozen before evaluation. It projects onto the top
# uncentered visual-activation energy directions using the same discovery
# samples, rank, and linear intervention as the downstream-sensitivity basis.
# Supplying a smoke job ID reuses an already queued technical gate.
if [[ $# -gt 0 ]]; then
  smoke=$1
else
  smoke=$(sbatch --parsable \
    --job-name=xvla-activation-energy-smoke \
    athena/slurm_xvla.sbatch \
    --mode visual_subspace \
    --architecture chi \
    --vision-encoder vit \
    --checkpoint "${checkpoints[0]}" \
    --cache "$cache" \
    --output results/visual_activation_energy_smoke.json \
    --seed 0 \
    --rank 96 \
    --gram-action-group all_balanced \
    --gram-probes 1 \
    --gram-samples 32 \
    --offline-eval-samples 32 \
    --gram-task-start 0 \
    --gram-task-end 4 \
    --random-controls 1 \
    --activation-energy-control \
    --subspace-offline-only \
    --profile-iters 20)
fi

for seed in 0 1 2; do
  for task in 8 9; do
    end=$((task + 1))
    sbatch --parsable \
      --dependency="afterok:${smoke}" \
      --job-name="xvla-activation-energy-s${seed}-t${task}" \
      athena/slurm_xvla.sbatch \
      --mode visual_subspace \
      --architecture chi \
      --vision-encoder vit \
      --checkpoint "${checkpoints[$seed]}" \
      --cache "$cache" \
      --output "results/visual_activation_energy_s${seed}_r96_t${task}.json" \
      --seed "$seed" \
      --rank 96 \
      --gram-action-group all_balanced \
      --gram-probes 4 \
      --gram-samples 1024 \
      --offline-eval-samples 1024 \
      --gram-task-start 0 \
      --gram-task-end 4 \
      --random-controls 1 \
      --activation-energy-control \
      --subspace-rollout-conditions full,causal_topk,activation_energy_topk \
      --task-start "$task" \
      --task-end "$end" \
      --eps-per-task 50 \
      --max-steps 400 \
      --profile-iters 20
  done
done

echo "Activation-energy smoke job: ${smoke}"
