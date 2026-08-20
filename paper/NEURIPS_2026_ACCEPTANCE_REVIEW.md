# NeurIPS 2026 workshop acceptance review

Assessment date: August 20, 2026.

These are subjective probability bands based on topical fit, paper completeness, and the usual
standard of workshop review. They are not official acceptance-rate estimates.

## Overall judgment

The current paper is credible workshop material. Its strongest combination is unusually concrete:
a mechanically certified rational VLA, competitive protocol-aligned closed-loop results, exact
layerwise weight structure, a causal intervention, and an openly reported grounding failure. It is
not yet a convincing NeurIPS main-track paper because the central causal comparison and the promised
coefficient-level payoff remain incomplete.

Estimated acceptance bands for the tailored versions:

- VLM4RWD: 60 to 75 percent. Excellent fit across VLA architecture, grounding, causal analysis,
  and failure diagnosis.
- Neural Network Artifacts extended abstract: 50 to 65 percent. Strong weight-artifact novelty,
  bounded by layerwise scope and a failed edit screen.
- Robot Learning Workshop: 35 to 50 percent. Strong robotics result, but no zero-shot or
  cross-task generalization.

The VLM4RWD version is the best single submission today. Neural Network Artifacts is the strongest
alternative if the author wants the paper judged primarily as a model-weights contribution.

## Evidence likely to help in review

1. The $93.7\%\pm0.8\%$ three-seed result uses the published $280$-step, $50$-trial-per-task
   protocol and is supported by 1,500 closed-loop rollouts.
2. The operator claim is not based only on an architecture diagram. The trained checkpoint is
   mechanically audited, and representative rational and bilinear branches are reconstructed.
3. The exact attention calculation is mathematically specific and scales through a reduced Gram.
   The full appendix screen includes all 96 block-head pairs and informative weak blocks.
4. The selected visual subspace has a closed-loop causal consequence, not only an offline probe.
5. The paper reports its counterfactual language shortcut and failed direct-edit gate. This makes
   the scope more trustworthy and fits workshops that welcome failure analysis or negative results.

## Main rejection risks

1. **No matched conventional twin.** The capability result shows that the rational policy works,
   but not what decomposability costs. This prevents a strong near-free-conversion claim.
2. **Weak language grounding.** The policy often follows scene identity rather than the edited
   object name. This is a central concern for a VLA or grounded-model audience.
3. **Layerwise exactness only.** The architecture is rational by construction, but the tractable
   exact decomposition is per attention layer. The whole-policy ranking uses activations and
   downstream sensitivity.
4. **Limited causal sample.** The decisive visual-subspace intervention uses one checkpoint and
   20 rollouts over four tasks.
5. **No successful coefficient-level repair.** The first direct query-key-value edit screen is a
   preregistered no-go. The paper has an analysis surface, not yet a selective editing interface.
6. **External baselines were not rerun.** Published references are protocol-aligned, but training
   and implementation differences remain.
7. **Narrow benchmark.** LIBERO-Object is a specialist setting and can reward scene-to-action
   memorization. There is no LIBERO-Goal, LIBERO-Long, or cross-environment evaluation.
8. **Artifact availability is promised, not delivered.** Anonymous code, checkpoint hashes,
   episode logs, and exact-analysis outputs would materially improve reviewer confidence.
9. **The closest literature is now direct.** Activation-based VLA interpretation and steering
   already exists, and 2026 work reports counterfactual grounding failures plus train-free
   mitigation. The paper must distinguish exact weight-derived structure from activation steering
   and cannot present the visual shortcut itself as the novelty.

## Highest-value work before submission

### Tier 1

1. Train and evaluate a same-skeleton conventional twin with identical data, updates, seeds,
   token layout, action decoder, and evaluation states. Predeclare a noninferiority margin.
2. Repeat the causal subspace intervention across all three trained checkpoints and more tasks,
   with paired canonical trials and confidence intervals.
3. Release an anonymous reproducibility package containing code, exact configs, checkpoint hashes,
   per-episode logs, and the mechanical audit output.

### Tier 2

4. Diagnose the failed coefficient edit before another large rollout. Test redundant paths,
   reparameterization or gauge sensitivity, and the mapping from Gram eigenspaces to independent
   query-key-value edits.
5. Add a focused grounding evaluation or repair that changes object-name binding without losing
   ordinary success. If no repair is ready, retain the failure analysis and avoid any claim of
   faithful grounding.
6. Report parameter-matched FLOPs, throughput, memory, and latency for rational and conventional
   models. This would also make AXIOM a credible fourth target.

### Tier 3

7. Expand to LIBERO-Goal or LIBERO-Long and include a counterfactual benchmark.
8. Strengthen related work and baseline coverage with current VLA robustness, weight-space
   interpretability, and structured-policy papers.

## Submission recommendation

If no additional training result arrives, submit the current VLM4RWD version after artifact release
and a final citation audit. If the matched twin or a selective edit succeeds, update all versions and
reassess because either result could materially raise the ceiling. Do not submit the same empirical
paper concurrently to multiple NeurIPS workshops without written permission from every affected
workshop chair.
