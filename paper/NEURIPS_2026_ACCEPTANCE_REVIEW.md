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

Estimated acceptance bands for the tailored versions after the matched seed-0 control and the
three-checkpoint all-layer Athena audit:

- VLM4RWD: 50 to 65 percent. Excellent fit across VLA architecture, grounded deployment,
  causal analysis, and failure diagnosis. The nonzero matched capability cost and incomplete
  task-level counterfactual grounding are material risks.
- Neural Network Artifacts extended abstract: 55 to 70 percent. The exact all-layer reconstruction
  and trained-checkpoint audit are unusually direct weight-artifact evidence, bounded by the lack
  of a successful selective coefficient edit.
- Robot Learning Workshop: 30 to 45 percent. Strong robotics evidence and an honest specialist
  Physical-AI result, but no zero-shot, cross-task, or cross-environment generalization.

The VLM4RWD version is the best single submission today. Neural Network Artifacts is the strongest
alternative if the author wants the paper judged primarily as a model-weights contribution.

## Evidence likely to help in review

1. The $93.7\%\pm0.8\%$ three-seed result uses the published $280$-step, $50$-trial-per-task
   protocol and is supported by 1,500 closed-loop rollouts.
2. The same-skeleton models differ by only 1,280 parameters, or 0.006 percent. Complete all-task
   controls reach 448 of 500 conventional versus 426 rational on A30, and 451 versus 422 on A6000.
   The seed-0 rational deficit is 4.4 to 5.8 points and misses the predeclared three-point
   noninferiority margin on both platforms. Task 3 accounts for 21 lost successes in each matrix.
3. The operator claim is not based only on an architecture diagram. The trained checkpoint is
   mechanically audited, and representative rational and bilinear branches are reconstructed.
4. The exact attention calculation is mathematically specific and scales through a reduced Gram.
   The native audit reconstructs all 12 attention modules and 128 heads in each of three
   checkpoints, with maximum relative error below $2.6\times10^{-7}$. Two checkpoints meet the
   original $2\times10^{-5}$ absolute indicator, while the third reaches $2.23\times10^{-5}$.
5. The selected visual subspace has a closed-loop causal consequence, not only an offline probe.
6. A 300-pair, three-checkpoint instruction intervention shifts the relative end-effector
   preference toward the renamed object in 84 to 88 percent of trials. Only 28 to 33 percent end
   closer to that object. A seed-0 conventional control responds even more strongly but has the
   same 33 percent endpoint rate, so responsiveness is not misattributed to decomposability.
7. The paper reports its incomplete grounding and failed direct-edit gate. This makes
   the scope more trustworthy and fits workshops that welcome failure analysis or negative results.

## Main rejection risks

1. **The matched capability cost is nonzero at seed 0.** The rational policy trails by 4.4 points
   on A30 and 5.8 on A6000. Conventional seeds 1 and 2 are running and are required before treating
   this as an across-training-seed estimate. Partial-conversion controls must localize the task-3 gap.
2. **Incomplete task-level language grounding.** The longer paired rollout shows a replicated
   directional response to the renamed object, but only 28 to 33 percent of counterfactual runs end
   closer to it and the original one-step screen remains weak. The conventional control is stronger,
   so this experiment supports a failure diagnosis rather than an architectural advantage.
3. **Layerwise exactness only.** All deployed attention modules have tiny scale-aware
   reconstruction error, but the tractable
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
8. **Artifact availability is not yet submission-ready.** Native scripts and raw episode-level
   results are versioned, but an anonymous public package with checkpoint hashes and one-command
   reproduction would materially improve reviewer confidence.
9. **The closest literature is now direct.** Activation-based VLA interpretation and steering
   already exists, and 2026 work reports counterfactual grounding failures plus train-free
   mitigation. The paper must distinguish exact weight-derived structure from activation steering
   and cannot present the visual shortcut itself as the novelty.

## Highest-value work before submission

### Tier 1

1. Finish conventional seeds 1 and 2 and their 1,000 canonical rollouts. The same-hardware
   seed-0 matrices are complete and reject the predeclared three-point noninferiority claim at
   this seed. Use partial-conversion controls to localize the task-3 deficit.
2. Repeat the causal visual-subspace intervention across checkpoints with paired canonical trials
   and confidence intervals. The separate three-checkpoint instruction intervention and
   conventional control are complete.
3. Release an anonymous reproducibility package containing code, exact configs, checkpoint hashes,
   per-episode logs, and the mechanical audit output.

### Tier 2

4. Diagnose the failed coefficient edit before another large rollout. Test redundant paths,
   reparameterization or gauge sensitivity, and the mapping from Gram eigenspaces to independent
   query-key-value edits.
5. Add a focused grounding evaluation or repair that changes object-name binding without losing
   ordinary success. If no repair is ready, retain the failure analysis and avoid any claim of
   faithful grounding.
6. Report parameter-matched FLOPs, throughput, memory, eager latency, and compiled latency for
   rational and conventional models. This would also make AXIOM a credible fourth target.

### Tier 3

7. Expand to LIBERO-Goal or LIBERO-Long and include a counterfactual benchmark.
8. Strengthen related work and baseline coverage with current VLA robustness, weight-space
   interpretability, and structured-policy papers.

## Submission recommendation

Neural Network Artifacts is the safest target because its central exact-audit result is unaffected
by the nonzero capability cost. VLM4RWD remains a strong second choice because the grounding
failure analysis fits its call, while Robot Learning Workshop has the largest theme mismatch.
Finish the live multi-seed and surgery jobs before the submission freeze if time permits, then
publish the anonymous artifact and run a final citation audit. Do not submit
the same empirical paper concurrently to multiple NeurIPS workshops without written permission from
every affected workshop chair.
