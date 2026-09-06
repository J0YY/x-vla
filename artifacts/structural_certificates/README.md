# Structural-certificate artifact

This anonymous, CPU-only artifact verifies the six structural certificates reported in the
paper. It uses only the Python standard library and the committed JSON evidence. No checkpoint,
dataset, GPU, network access, or package installation is needed.

From the repository root, run:

```bash
python3 artifacts/structural_certificates/verify.py
```

The command performs seven checks:

1. It checks the SHA-256 digest of all eighteen raw results, six summaries, and three frozen source
   snapshots against `manifest.json`. Every new raw source map must match its summary and either
   the clean checkout or an exact decoded snapshot.
2. It independently recomputes every Conv joint-attention checkpoint maximum and the frozen
   `1e-6` aggregate gate from the recorded module and head rows.
3. It independently recomputes every Conv joint-FFN checkpoint maximum and the frozen `1e-10`
   aggregate gate from 384 module-input rows.
4. It independently recomputes every complete Conv joint-block checkpoint maximum and the frozen
   `1e-4` aggregate gate from 384 block-input rows.
5. It independently recomputes the ViT modality partition, algebraic errors, coherent-energy
   distributions, and descriptive prompt-permutation statistics from all raw rows.
6. It independently recomputes the fixed-input ViT learned-forward action arrays, digests, errors,
   per-checkpoint aggregates, and frozen `1e-3` gate across all 48 replays and 816 recorded stages.
7. It independently recomputes the RationalNorm denominator, action-NRMSE, range, and local-error
   gates from the 159 recorded site rows and action sum-of-squares statistics. It then reproduces
   the three-checkpoint summary.

A successful run reports that Conv joint attention, joint FFNs, complete joint blocks, the ViT
modality partition, the sequential ViT learned-forward replay, and the RationalNorm primary
denominator and floating-point-fidelity gate pass.
It also reports the preregistered RationalNorm secondary approximation-coverage gate as false.
That secondary result is intentionally preserved in the immutable manifest and is not part of the
primary claim. Modality magnitudes and prompt-permutation changes are descriptive, with no favorable
threshold or grounding claim.

The learned-forward result is input-conditioned and numerical. It starts at exact prepared model
tensors and reaches the learned linear action output, but it is not one compact symbolic
contraction or an input-general proof. It excludes preprocessing, fixed action unnormalization,
chunk execution, simulator dynamics, and success. The verifier checks recorded evidence. It does
not rerun model inference or widen the scopes stated inside the source summaries.
