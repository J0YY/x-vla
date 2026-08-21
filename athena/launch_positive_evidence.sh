#!/usr/bin/env bash
set -euo pipefail

cd /work/joy/x-vla-workshop

cache=artifacts/libero_frames_100000_64.pkl
chi_s0=artifacts/ckpt_linear_rat_vit_s0_v2.pt
chi_s1=artifacts/ckpt_linear_rat_vit_s1.pt
chi_s2=artifacts/ckpt_linear_rat_vit_s2.pt

ensemble_smoke=$(sbatch --parsable \
  --job-name=xvla-ensemble-smoke \
  athena/slurm_xvla.sbatch \
  --mode ensemble_capability \
  --architecture chi \
  --checkpoint "$chi_s0" \
  --ensemble-checkpoint "$chi_s1" \
  --ensemble-checkpoint "$chi_s2" \
  --ensemble-reduction mean \
  --cache "$cache" \
  --output results/ensemble_mean_smoke.json \
  --task-start 0 \
  --task-end 1 \
  --eps-per-task 1 \
  --max-steps 8 \
  --profile-iters 20)

balanced_smoke=$(sbatch --parsable \
  --job-name=xvla-balanced-subspace-smoke \
  athena/slurm_xvla.sbatch \
  --mode visual_subspace \
  --architecture chi \
  --checkpoint "$chi_s0" \
  --cache "$cache" \
  --output results/visual_balanced_smoke.json \
  --rank 64 \
  --gram-action-group all_balanced \
  --gram-probes 1 \
  --gram-samples 32 \
  --offline-eval-samples 32 \
  --gram-task-start 0 \
  --gram-task-end 4 \
  --random-controls 2 \
  --subspace-offline-only \
  --profile-iters 20)

for reduction in mean median; do
  for range in "0 3" "3 6" "6 8" "8 10"; do
    read -r start end <<<"$range"
    sbatch --parsable \
      --dependency="afterok:${ensemble_smoke}" \
      --job-name="xvla-ensemble-${reduction}-${start}${end}" \
      athena/slurm_xvla.sbatch \
      --mode ensemble_capability \
      --architecture chi \
      --checkpoint "$chi_s0" \
      --ensemble-checkpoint "$chi_s1" \
      --ensemble-checkpoint "$chi_s2" \
      --ensemble-reduction "$reduction" \
      --cache "$cache" \
      --output "results/ensemble_${reduction}_t${start}_${end}.json" \
      --task-start "$start" \
      --task-end "$end" \
      --eps-per-task 50 \
      --max-steps 280 \
      --profile-iters 20
  done
done

for seed in 0 1 2; do
  case "$seed" in
    0) checkpoint=$chi_s0 ;;
    1) checkpoint=$chi_s1 ;;
    2) checkpoint=$chi_s2 ;;
  esac
  sbatch --parsable \
    --dependency="afterok:${balanced_smoke}" \
    --job-name="xvla-disjoint-gripper-s${seed}" \
    athena/slurm_xvla.sbatch \
    --mode visual_subspace \
    --architecture chi \
    --checkpoint "$checkpoint" \
    --cache "$cache" \
    --output "results/visual_disjoint_gripper_s${seed}.json" \
    --seed "$seed" \
    --rank 64 \
    --gram-action-group gripper \
    --gram-samples 1024 \
    --offline-eval-samples 1024 \
    --gram-task-start 0 \
    --gram-task-end 10 \
    --random-controls 5 \
    --subspace-offline-only \
    --profile-iters 20
done

for rank in 64 96 128 192; do
  for seed in 0 1 2; do
    case "$seed" in
      0) checkpoint=$chi_s0 ;;
      1) checkpoint=$chi_s1 ;;
      2) checkpoint=$chi_s2 ;;
    esac
    sbatch --parsable \
      --dependency="afterok:${balanced_smoke}" \
      --job-name="xvla-balanced-s${seed}-r${rank}" \
      athena/slurm_xvla.sbatch \
      --mode visual_subspace \
      --architecture chi \
      --checkpoint "$checkpoint" \
      --cache "$cache" \
      --output "results/visual_balanced_s${seed}_r${rank}_confirm.json" \
      --seed "$seed" \
      --rank "$rank" \
      --gram-action-group all_balanced \
      --gram-probes 4 \
      --gram-samples 1024 \
      --offline-eval-samples 1024 \
      --gram-task-start 0 \
      --gram-task-end 4 \
      --random-controls 3 \
      --task-start 4 \
      --task-end 8 \
      --eps-per-task 5 \
      --max-steps 400 \
      --profile-iters 20
  done
done

echo "Smoke jobs: ensemble=${ensemble_smoke}, balanced_subspace=${balanced_smoke}"
