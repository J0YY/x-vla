#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

cache=artifacts/libero_frames_100000_64.pkl
vit_checkpoints=(
  artifacts/ckpt_linear_rat_vit_s0_v2.pt
  artifacts/ckpt_linear_rat_vit_s1.pt
  artifacts/ckpt_linear_rat_vit_s2.pt
)
conv_checkpoints=(
  artifacts/ckpt_linear_rat_conv_s0.pt
  artifacts/ckpt_linear_rat_conv_s1_matched.pt
  artifacts/ckpt_linear_rat_conv_s2_matched.pt
)

# Rank 96 is fixed before any corrected-basis outcome. The prior numerical
# screen is excluded because its cache task join was invalid.
vit_smoke=$(sbatch --parsable \
  --job-name=xvla-taskmap-visual-vit-smoke \
  athena/slurm_xvla.sbatch \
  --mode visual_subspace \
  --architecture chi \
  --vision-encoder vit \
  --checkpoint "${vit_checkpoints[0]}" \
  --cache "$cache" \
  --output results/visual_taskmap_vit_smoke.json \
  --seed 0 \
  --rank 96 \
  --gram-action-group all_balanced \
  --gram-probes 1 \
  --gram-samples 32 \
  --offline-eval-samples 32 \
  --gram-task-start 0 \
  --gram-task-end 4 \
  --random-controls 1 \
  --subspace-offline-only \
  --profile-iters 2)

activation_smoke=$(sbatch --parsable \
  --dependency="afterok:${vit_smoke}" \
  --job-name=xvla-taskmap-activation-smoke \
  athena/slurm_xvla.sbatch \
  --mode visual_subspace \
  --architecture chi \
  --vision-encoder vit \
  --checkpoint "${vit_checkpoints[0]}" \
  --cache "$cache" \
  --output results/visual_taskmap_activation_smoke.json \
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
  --profile-iters 2)

conv_smoke=$(sbatch --parsable \
  --job-name=xvla-taskmap-visual-conv-smoke \
  athena/slurm_xvla.sbatch \
  --mode visual_subspace \
  --architecture chi \
  --vision-encoder conv \
  --checkpoint "${conv_checkpoints[0]}" \
  --cache "$cache" \
  --output results/visual_taskmap_conv_smoke.json \
  --seed 0 \
  --rank 96 \
  --gram-action-group all_balanced \
  --gram-probes 1 \
  --gram-samples 32 \
  --offline-eval-samples 32 \
  --gram-task-start 0 \
  --gram-task-end 4 \
  --random-controls 1 \
  --subspace-offline-only \
  --profile-iters 2)

declare -a vit_blind_jobs=()
declare -a activation_jobs=()
declare -a conv_jobs=()
declare -a exact_jobs=()

for seed in 0 1 2; do
  exact_job=$(sbatch --parsable \
    --dependency="afterok:${vit_smoke}" \
    --job-name="xvla-taskmap-exact-s${seed}" \
    athena/slurm_xvla.sbatch \
    --mode exact_attention \
    --architecture chi \
    --vision-encoder vit \
    --checkpoint "${vit_checkpoints[$seed]}" \
    --cache "$cache" \
    --output "results/exact_attention_taskmap_s${seed}.json" \
    --seed "$seed" \
    --profile-iters 20)
  exact_jobs+=("$exact_job")

  for task in 8 9; do
    end=$((task + 1))
    vit_job=$(sbatch --parsable \
      --dependency="afterok:${vit_smoke}" \
      --job-name="xvla-taskmap-vit-s${seed}-t${task}" \
      athena/slurm_xvla.sbatch \
      --mode visual_subspace \
      --architecture chi \
      --vision-encoder vit \
      --checkpoint "${vit_checkpoints[$seed]}" \
      --cache "$cache" \
      --output "results/visual_taskmap_vit_s${seed}_r96_blind_t${task}.json" \
      --seed "$seed" \
      --rank 96 \
      --gram-action-group all_balanced \
      --gram-probes 4 \
      --gram-samples 1024 \
      --offline-eval-samples 1024 \
      --gram-task-start 0 \
      --gram-task-end 4 \
      --random-controls 3 \
      --task-start "$task" \
      --task-end "$end" \
      --eps-per-task 50 \
      --max-steps 400 \
      --profile-iters 20)
    vit_blind_jobs+=("$vit_job")

    activation_job=$(sbatch --parsable \
      --dependency="afterok:${activation_smoke}" \
      --job-name="xvla-taskmap-activation-s${seed}-t${task}" \
      athena/slurm_xvla.sbatch \
      --mode visual_subspace \
      --architecture chi \
      --vision-encoder vit \
      --checkpoint "${vit_checkpoints[$seed]}" \
      --cache "$cache" \
      --output "results/visual_taskmap_activation_s${seed}_r96_t${task}.json" \
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
      --profile-iters 20)
    activation_jobs+=("$activation_job")

    conv_job=$(sbatch --parsable \
      --dependency="afterok:${conv_smoke}" \
      --job-name="xvla-taskmap-conv-s${seed}-t${task}" \
      athena/slurm_xvla.sbatch \
      --mode visual_subspace \
      --architecture chi \
      --vision-encoder conv \
      --checkpoint "${conv_checkpoints[$seed]}" \
      --cache "$cache" \
      --output "results/visual_taskmap_conv_s${seed}_r96_blind_t${task}.json" \
      --seed "$seed" \
      --rank 96 \
      --gram-action-group all_balanced \
      --gram-probes 4 \
      --gram-samples 1024 \
      --offline-eval-samples 1024 \
      --gram-task-start 0 \
      --gram-task-end 4 \
      --random-controls 3 \
      --task-start "$task" \
      --task-end "$end" \
      --eps-per-task 50 \
      --max-steps 400 \
      --profile-iters 20)
    conv_jobs+=("$conv_job")
  done
done

summary_dependencies=("${vit_blind_jobs[@]}" "${activation_jobs[@]}")
summary_dependency=$(IFS=:; echo "${summary_dependencies[*]}")
summary_job=$(sbatch --parsable \
  --dependency="afterok:${summary_dependency}" \
  --job-name=xvla-taskmap-blind-summary \
  athena/slurm_blind_subspace_summary.sbatch \
  --results-dir results \
  --output results/visual_taskmap_blind_rank96_summary.json \
  --bootstrap-samples 20000 \
  --seed 20260821)

echo "ViT smoke job: $vit_smoke"
echo "Activation-energy smoke job: $activation_smoke"
echo "Conv smoke job: $conv_smoke"
echo "ViT blind jobs: ${vit_blind_jobs[*]}"
echo "Activation-energy jobs: ${activation_jobs[*]}"
echo "Conv blind jobs: ${conv_jobs[*]}"
echo "Corrected-input exact-attention jobs: ${exact_jobs[*]}"
echo "Summary job: $summary_job"
