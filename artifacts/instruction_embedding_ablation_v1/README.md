# Instruction-embedding-path ablation artifact

This anonymous CPU-only artifact verifies a prospectively frozen closed-loop ablation on three
fixed chi-VLA ViT policies. Both paired conditions receive the identical correct 32-token ID
tensor. The ablated condition replaces only the learned token-embedding output with exact zeros
after lookup. Vision, robot state, embodiment, sequence length, positions, the separate learned
common BOS, action queries, and all weights remain unchanged.

From the repository root, run:

```bash
python3 artifacts/instruction_embedding_ablation_v1/verify.py
```

The verifier uses only the Python standard library. It checks the SHA-256 digest of the preflight
manifest, smoke result, strict summary, and all 15 raw shards. It rejects duplicate JSON keys,
rebuilds the complete 3-checkpoint by 10-task by 10-episode matrix, verifies paired inputs and hook
audits, decodes every saved trajectory, reproduces success from rewards, and reconstructs executed
actions from the recorded action chunks. It then recomputes checkpoint, task, discordance, exact
paired-test, and frozen-gate statistics.

The expected result is 263/300 successes with the intact learned lexical path and 63/300 after
zeroing the post-lookup embeddings, a paired gap of 66.7 percentage points. Full-path success is
84%, 87%, and 92% by checkpoint. All ten task gaps are positive. The one-sided paired exact
p-value is 1.42e-56, and the frozen task-stratified state-cluster bootstrap interval is 61.7 to
71.3 points. The verifier also derives a 20-point lower bound over the entire support of that
bootstrap resampling scheme, independently establishing that its lower endpoint must be positive.

The paper figure is generated from verifier output rather than hand-entered results:

```bash
python3 scripts/plot_instruction_embedding_ablation.py
```

This evidence establishes necessity of the learned lexical-embedding path only for familiar
LIBERO-Object tasks and the three fixed policies. It does not establish semantic grounding,
paraphrase robustness, unseen-object or compositional generalization, cross-suite or physical
transfer, or a benefit unique to tensor decomposability.
