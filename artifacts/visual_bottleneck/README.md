# Verified visual-bottleneck artifact

This CPU-only artifact verifies the corrected rank-96 visual-bottleneck result from the committed
LIBERO-Object JSON files. It requires only the Python standard library.

From the repository root, run:

```bash
python3 artifacts/visual_bottleneck/verify.py
```

The verifier checks the SHA-256 identity of the hardened summary and all twelve raw result files.
It then independently rebuilds the complete three-checkpoint by two-task by 50-episode matrix,
requires identical full-policy outcomes across the two matched allocations, and recomputes the
conditional retention of the action-Jacobian, activation-energy, and three fixed random rank-96
projectors.

The expected result is:

- action-Jacobian projector: 259 of 272 baseline successes retained, or 95.2 percent
- activation-energy projector: 267 of 272 retained, or 98.2 percent
- fixed random projectors: 37, 9, and 22 of 272 retained
- all three action-Jacobian-versus-random checkpoint comparisons have the same positive sign
- the frozen fixed-rank selectivity gate passes
- the stronger action-Jacobian-specificity gate does not pass

The projector discovery cache uses official Object tasks 0 through 3 after applying the pinned
LeRobot-to-LIBERO task mapping. Rank 96 was fixed before any corrected-basis outcome was observed.
Closed-loop evaluation uses tasks 8 and 9, which are absent from corrected projector construction.
The policies were trained on all ten Object tasks. This is not unseen-task, zero-shot, or clean
rank-selection-holdout evidence.

The verifier also reproduces deterministic 20,000-sample paired checkpoint-task-episode cluster
resampling intervals. These are empirical stability intervals over the finite canonical-state
matrix, not population confidence intervals.
