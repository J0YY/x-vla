# Four-suite breadth artifact

This anonymous, CPU-only artifact verifies the recorded χ-VLA breadth results over the 40 familiar
tasks in LIBERO-Object, LIBERO-Spatial, LIBERO-Goal, and LIBERO-10. It uses only the Python standard
library, sixteen manifest records, and 96 raw evaluation shards. No checkpoint, dataset, GPU,
network access, or package installation is needed.

From the repository root, run:

```bash
python3 artifacts/four_suite_breadth/verify.py
```

The verifier checks the immutable SHA-256 identity of the frozen ensemble manifest, the strict
ensemble summary, and the separate seed-0 specialist-retention summary. It then independently
verifies the SHA-256 identity and frozen provenance of every raw shard, then recomputes all 12,000
matched-comparison episode outcomes. From those outcomes it derives the three member macro scores,
their mean and sample standard deviation, the matched conventional comparison, suite means,
task-threshold counts, ensemble uplift and gates, and the generalist-to-specialist retention ratio.
Six training records and their exact checkpoint metadata verify that both architecture families
used the same recipe and differed in parameter count by only 1,280 parameters, or 0.006%.

The expected verified results are:

- three individual joint checkpoints: 70.85%, 71.35%, and 68.85%, or 70.35% with a 1.32% sample
  standard deviation across checkpoints
- matched conventional checkpoints: 71.80%, 67.55%, and 68.00%, or 69.12% with a 2.33% sample
  standard deviation; χ-VLA is 1.23 percentage points higher under the frozen comparison
- fixed three-checkpoint mean-prediction ensemble: 1,495/2,000, or 74.75%, with every suite at or
  above 50%, 34/40 tasks at or above 50%, and a 4.40 percentage-point gain over the member mean
- equal-update seed-0 retention comparison: one joint checkpoint retains 95.16% of four separately
  trained suite specialists' pooled task-macro success, above the frozen 85% gate

All 40 tasks appeared during training. The ensemble performs three policy forward passes per query,
and its protocol was frozen after the member checkpoints had been selected but before ensemble
outcomes were observed. This package verifies the recorded raw evaluation outcomes and derives the
matched comparison from their episode rows. It does not rerun model inference or establish
unseen-task generalization.
