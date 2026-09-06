# Instruction-embedding ablation v1, prospectively frozen protocol

## Question and claim boundary

This study asks whether the deployed policy's learned lexical-embedding path is
necessary for closed-loop success on familiar LIBERO-Object tasks. It does not
test grounding, paraphrase robustness, unseen objects, novel compositions,
cross-suite transfer, physical transfer, or an advantage caused by tensor
decomposability.

The protocol was designed after the separate instruction-necessity result made
a more precise mechanism test useful. No result from the endpoint or its fresh
canonical states was inspected before the intervention, matrix, and gates were
fixed.

## Fixed intervention

Both conditions receive the same correct 32-token instruction-ID tensor. The
full-path condition uses the deployed model unchanged. In the ablated condition,
a scoped forward hook on `model.tok_emb` replaces the complete output tensor with
`torch.zeros_like(output)` after the embedding lookup. This zeros every learned
embedding vector at all 32 instruction positions, including tokenizer BOS and
PAD positions.

The intervention does not alter token IDs, sequence length, sequence positions,
visual tokens, robot-state projection, embodiment token, the separate learned
`model.bos`, joint positional embeddings, action queries, transformer weights,
normalization, or action head. The hook is installed only around an ablated
prediction or rollout, records every invocation, and is removed afterward.
Model state is hashed before and after evaluation. The common learned BOS,
action queries, positions, and token-embedding weights receive separate hashes.

## Fixed evaluation matrix

- Checkpoints: capable chi-ViT seeds 0, 1, and 2 with the exact hashes in the
  manifest.
- Suite: all ten LIBERO-Object tasks.
- States: canonical episodes 30 through 39, ten per task. These states are fresh
  relative to counterfactual goal following, which uses episodes 20 through 29,
  and instruction necessity, which uses episodes 40 through 49.
- Conditions: correct prompt with full lexical embeddings and the identical
  correct prompt with all 32 token-embedding outputs zeroed.
- Pairing: every condition begins from the exact same official initial state,
  reset seed, ten-step settle, and validated physical snapshot.
- Rollout: resolution 64, action horizon 8, execution horizon 8, and at most 280
  environment steps.
- Numerics: PyTorch deterministic algorithms, highest float32 matmul precision,
  cuBLAS workspace `:4096:8`, cuDNN benchmark disabled, one RTX A6000.
- Scale: 100 unique states, 300 checkpoint-state pairs, and 600 rollouts.
- Parallelization: five two-task shards for each of three checkpoints, followed
  by one CPU-only strict summary.

Condition order is deterministically balanced by a frozen SHA-256 rule over
checkpoint, task, and episode. The strict smoke evaluates one input from every
task at seed 0 and executes both conditions on the same tensors without running
the simulator trajectory.

## Frozen gates

The endpoint passes only if all of the following hold:

- full-path success is at least 70 percent pooled and at least 60 percent at
  every checkpoint
- the pooled paired success gap is at least 20 percentage points
- the paired gap is strictly positive at every checkpoint
- the one-sided exact paired McNemar p-value is at most 0.01
- at least eight of ten task aggregates have a strictly positive gap and no
  task aggregate has a negative gap
- the lower endpoint of a 20,000-draw task-stratified state-cluster bootstrap is
  strictly above zero

Each bootstrap draw samples ten episode states with replacement within every
task and keeps the three checkpoint outcomes belonging to a sampled state
together. The pooled exact McNemar calculation is a paired descriptive test over
300 checkpoint-state pairs and is not cluster-robust inference.

## Validation and reporting

The preflight freezes checkpoint, cache, provenance, BDDL, official state,
settled physical input, imported source, extension source, runtime, and LIBERO
tree identities. Raw outputs contain every action chunk, executed action, reward,
done flag, physical start, final state, intervention audit, and source identity.
The CPU summary reconstructs every executed trajectory from its saved chunks,
recomputes success from rewards, checks exact paired starts and instruction IDs,
requires a hook call for every ablated chunk and none for the full condition,
and refuses any incomplete or identity-mismatched matrix. It is pinned to the
`safesae-openvla` Python executable, Python 3.10.19, NumPy 1.26.4, and an empty
`CUDA_VISIBLE_DEVICES` value before scoring begins.

If all gates pass, the permitted claim is limited to learned lexical-embedding
path necessity for familiar LIBERO-Object tasks and these three fixed policies.
