# Athena experiments

This directory runs the χ-VLA LIBERO evaluation natively on Athena through Slurm. It does not
depend on Modal. The runner preserves the paper protocol:

- official packaged LIBERO initial states, with no random-reset fallback
- all ten LIBERO-Object tasks
- 50 canonical episodes per task for the matched evaluation
- 280 simulator steps, 10 settle steps, and receding-horizon execution
- the corrected 180-degree camera transform, `[::-1, ::-1]`
- the vocabulary and normalization statistics reconstructed from the exact training cache

The conventional control uses the same χ-VLA skeleton, dimensions, training data, and linear
action head. Only the internal block family changes to softmax attention, SwiGLU, and RMSNorm.
The SwiGLU ranks are chosen to parameter-match the χ model at each block width.

Submit a smoke test from Athena with:

```bash
mkdir -p /work/joy/x-vla-workshop/logs /work/joy/x-vla-workshop/results
sbatch --job-name=xvla-smoke athena/slurm_xvla.sbatch \
  --mode smoke \
  --architecture conventional \
  --checkpoint artifacts/ckpt_linear_conventional_vit_s0.pt \
  --cache artifacts/libero_frames_100000_64.pkl \
  --output results/conventional_s0_smoke.json
```

After the smoke test passes, change `--mode smoke` to `--mode capability` for the full evaluation.
Use `--mode profile` when only latency, memory, and parameter counts are needed.

Additional modes cover the main review risks:

- `causal` runs paired true-instruction and counterfactual-instruction trajectories from matched
  canonical states and reports a trajectory-level proximity shift.
- `exact_attention` reconstructs every bilinear-attention module in both transformer stacks from
  its learned coefficients.
- `surgery` edits the learned Q, K, and V coefficient columns using exact weight-derived subspaces.
  Discovery sweeps must pass both prespecified directional gates before held-out confirmation.

`profile_training_pair.py` measures χ and conventional training-step cost on the same fixed batch
and physical GPU. `train_checkpoint.py` reproduces the full 40k-step EMA training recipe for new
matched seeds. `summarize_xvla_results.py` aggregates sharded canonical evaluations with per-task
counts and Wilson intervals. `summarize_surgery.py` applies the prespecified discovery gate without
hiding failed configurations.

## First completed matched result

The first mixed-GPU Athena pass gives 449/500 (89.8%) for the seed-0 conventional control and
426/500 (85.2%) for the rational χ-ViT. It also reveals a large architecture-by-GPU interaction,
so all-task A6000 and A30 replication matrices are required before interpreting that gap. The
models contain 20,138,632 and 20,137,352 parameters, respectively. Raw shards, smoke outputs,
all-layer audits, and paired causal-rollout records live in `athena/results/`. The dated job
manifest in `RUNS_2026-08-20.md` records Slurm identifiers, artifact-transfer failures,
resubmissions, hardware assignments, and the prespecified surgery gate.
