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
- `visual_subspace` rebuilds a checkpoint-specific, gripper-sensitive visual-bond Gram from one
  training-cache sample, measures offline reconstruction on a disjoint cache sample, then compares
  full, top-rank, and matched random projections on independent canonical simulator states.
- `exact_attention` reconstructs every bilinear-attention module in both transformer stacks from
  its learned coefficients.
- `surgery` edits the learned Q, K, and V coefficient columns using exact weight-derived subspaces.
  Discovery sweeps must pass both prespecified directional gates before held-out confirmation.

`profile_training_pair.py` measures χ and conventional training-step cost on the same fixed batch
and physical GPU. `train_checkpoint.py` reproduces the full 40k-step EMA training recipe for new
matched seeds. `summarize_xvla_results.py` aggregates sharded canonical evaluations with per-task
counts and Wilson intervals. `summarize_surgery.py` applies the prespecified discovery gate without
hiding failed configurations.

The `--suite` flag also supports `libero_spatial`, `libero_goal`, and `libero_10`.
`--training-suite` records the vocabulary and normalization provenance of the checkpoint. When the
two suite flags differ, the result is explicitly zero-shot and unseen instruction words map to the
padding identifier. When they match, `build_suite_cache.py` and the generalized trainer support an
in-domain control. `launch_indomain_multisuite.sh` builds three caches in parallel, trains matched
χ and conventional seed-0 policies, and dependency-queues their canonical evaluations.

`launch_native_provenance.sh` trains conventional seed 0 and χ seeds 0 to 2 concurrently with the
same Athena-native trainer, then queues four canonical shards per checkpoint. This matrix separates
training provenance from the severe closed-loop seed sensitivity observed in the legacy comparison.

`launch_positive_evidence.sh` submits two preregistered tests. The first evaluates mean and
coordinate-median ensembles of the three fixed χ checkpoints over the full canonical 500-trial
protocol. The second replaces the gripper-only visual Gram with a balanced action-Jacobian Gram,
uses disjoint cache samples, compares three random projectors per checkpoint, and reports the full
rank 64, 96, 128, and 192 curve on confirmation tasks 4 to 7. Smoke jobs gate every dependent run.

`launch_cp_pruning.sh` tests a direct benefit of tensor decomposability. It removes complete
rank-one terms from every bilinear FFN using a weight-only, CP-gauge-invariant norm product. The
pilot evaluates a complete 25 to 75 percent removal curve on the strongest fixed χ checkpoint and
compares each point with three matched random component masks. The checkpoint is functionally
pruned for evaluation. Parameter counts describe the equivalent physically compacted model.

`launch_conv_cp_pruning.sh` independently measures the same magnitude-pruning curve on the
strongest convolutional χ checkpoint. It includes an unpruned 50-trial control under the Athena
runner. A ViT discovery point is eligible for promotion only after a 500-trial confirmation on
other checkpoints. The convolutional sweep is an architecture-replication check.

## First completed matched result

The first mixed-GPU Athena pass gives 449/500 (89.8%) for the seed-0 conventional control and
426/500 (85.2%) for the rational χ-ViT. It also reveals a large architecture-by-GPU interaction,
so all-task A6000 and A30 replication matrices are required before interpreting that gap. The
models contain 20,138,632 and 20,137,352 parameters, respectively. Raw shards, smoke outputs,
all-layer audits, and paired causal-rollout records live in `athena/results/`. The dated job
manifest in `RUNS_2026-08-20.md` records Slurm identifiers, artifact-transfer failures,
resubmissions, hardware assignments, and the prespecified surgery gate.
