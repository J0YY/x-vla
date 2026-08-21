# ViT capability artifact

This anonymous, CPU-only artifact verifies the closed-loop LIBERO-Object capability result for
three fixed χ-VLA ViT checkpoints and their preregistered arithmetic prediction-mean ensemble. It
uses only the Python standard library and the committed JSON evidence. No checkpoint, dataset,
GPU, network access, or package installation is needed.

From the repository root, run:

```bash
python3 artifacts/vit_capability/verify.py
```

The command checks all 16 raw-result SHA-256 digests, the frozen Object cache provenance, the
complete seed-by-shard matrix, and every task/episode key. It independently recomputes shard and
per-task counts, the three 500-trial member totals, the ensemble total, the member mean and sample
standard deviation, and the ensemble gains over the member mean and best member.

The frozen rollout protocol is ten tasks, 50 canonical episodes per task, a 280-step cap, 10
settle steps, and an execution horizon of eight. The verifier rejects missing, duplicate, or
out-of-range episode rows and any result labeled with a different protocol.

The expected verified result is:

- members: 426/500, 404/500, and 450/500
- member mean: 85.3% with a 4.6% sample standard deviation across checkpoints
- arithmetic prediction-mean ensemble: 467/500, or 93.4%
- ensemble gain: 8.1 percentage points over the member mean and 3.4 points over the best member

The member result recorder predates embedded checkpoint and cache digests. This package therefore
binds each immutable raw-result digest and recorded artifact basename to the checkpoint SHA-256
identities later verified live and recorded in `athena/RUNS_2026-08-20.md`. The cache identity is
also checked against the committed provenance result. This verifies the recorded evidence. It does
not rerun model inference or convert the single deterministic ensemble evaluation into an
independent training replicate.

The paper-ready two-panel figure is generated from the verifier's recomputed summary, never from
hand-entered plot values:

```bash
python3 scripts/plot_verified_vit_capability.py
```

This writes `paper_figures/vit_capability_verified.pdf` under a new filename.
