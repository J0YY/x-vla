# Frozen ViT full learned-forward numerical certificate, v1

This protocol was fixed before any smoke or full outcome was observed.

## Scientific question

For the same three ViT checkpoints whose frozen closed-loop member success rates are
85.2%, 80.8%, and 90.0%, can an implementation independent of learned-module forward
methods reproduce the complete deterministic learned map on fixed corrected inputs?

The reconstruction starts from the exact prepared model input tensors and evaluates saved
weights and buffers with NumPy float64. It includes the learned patch projection and vision
position embedding, all four vision blocks, the visual projection, language and embodiment
embeddings, state projection, learned BOS and action-query tokens, joint position embedding,
all eight joint blocks, final RationalNorm, and the linear action head. The reference is one
unmodified PyTorch float32 model call on the same tensors.

## Frozen identities

- Suite and cache: LIBERO-Object, `libero_frames_100000_64.pkl`, SHA-256
  `053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662`.
- Provenance: completed job `830988` and its exact cache/source comparison artifact.
- Capability manifest: SHA-256
  `c23a9ba8bf791d55b0a731743fdac7810395d0030fdd6225ee98b7ec6da42453`.
- Checkpoints:
  - seed 0, `ckpt_linear_rat_vit_s0_v2.pt`, SHA-256
    `96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9`.
  - seed 1, `ckpt_linear_rat_vit_s1.pt`, SHA-256
    `cc0780b989165a80a449c03cbb3574e2f56b2e5747a64cf32d1fde418df8ec3c`.
  - seed 2, `ckpt_linear_rat_vit_s2.pt`, SHA-256
    `413a770071bd8f16b924c5604f7c7a58567d2eb6c40c66aab202bcf046fec910`.

All runner, wrapper, launcher, imported model, operator, cache, checkpoint, provenance,
dataset-metadata, capability-manifest, and runtime identities are checked before execution and
again before result publication. Every queued wrapper revalidates the same SHA-256 source-bundle
manifest at job start, so dependent jobs cannot silently adopt post-smoke source changes. Exact
snapshots bind the three dirty imported sources to a later clean checkout without committing the
user's working copies. The GPU runtime and the CPU summary's Python, NumPy, and PyTorch versions
are frozen. Outputs use a fresh protected namespace.

## Frozen input selection

Within each official Object task, rank every cache row by
`SHA256(protocol_id || checkpoint_sha256 || canonical_cache_record_sha256)`. Select two rows
from each official task 0 through 5 and one row from each task 6 through 9, for exactly 16
inputs per checkpoint. The smoke uses the first selected seed-0 input and traverses every
learned component in the full protocol. Input selection never depends on observed residuals.
The ten exact instruction strings come from the SHA-pinned capability manifest. The external
LIBERO installation must reproduce them exactly before tokenization.

## Frozen gate

The certificate passes only if all of the following hold:

1. The manual PyTorch component traversal is bitwise identical to the unmodified deployed
   model output on every input.
2. Every stored stage and action value is finite.
3. Exactly 48 action chunks of shape 8 by 7 are evaluated.
4. The maximum per-input relative L2 error between the deployed float32 action chunk and the
   independently reconstructed NumPy float64 action chunk is at most `1e-3`.
5. A CPU summary independently recomputes the action errors, hashes, counts, and gate from the
   stored action arrays and revalidates every frozen identity.

The threshold allows accumulated float32 versus float64 operation-order error across twelve
sequential transformer blocks while remaining a strict numerical-fidelity test.

## Claim boundary

A pass supports an input-conditioned full learned-forward numerical certificate for these
three capable checkpoints. It is not one compact symbolic tensor contraction and is not an
input-general formal identity proof. It excludes image acquisition and resizing, cache and
state preprocessing, fixed physical-action unnormalization, routing or decoding outside the
linear head, chunk execution, simulator dynamics, and success evaluation. It therefore does
not by itself certify the complete closed-loop policy or a vision-stack/action-head ODT.
