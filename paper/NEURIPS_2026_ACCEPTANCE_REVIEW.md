# NeurIPS 2026 workshop acceptance review

Assessment date: August 21, 2026.

These are subjective probability bands based on topical fit, paper completeness, and the usual
standard of workshop review. They are not official acceptance-rate estimates.

## Overall judgment

The current paper is credible workshop material. Its strongest combination is unusually concrete:
a mechanically certified rational VLA, competitive protocol-aligned closed-loop results, exact
layerwise weight structure, a causal intervention, and an openly reported grounding failure. It is
not yet a convincing NeurIPS main-track paper because the causal bottleneck is not near-lossless
and the promised coefficient-level payoff remains incomplete.

Estimated acceptance bands for the tailored versions after the matched seed-0 control, the
three-checkpoint all-layer Athena audit, the completed conventional-seed runs, and the completed
coefficient-surgery sweep:

- VLM4RWD: 45 to 60 percent. Excellent fit across VLA architecture, grounded deployment,
  causal analysis, and failure diagnosis. The nonzero matched capability cost and incomplete
  task-level counterfactual grounding are material risks, and the newly observed training variance
  weakens the controlled-comparison story.
- Neural Network Artifacts extended abstract: 50 to 65 percent. The exact all-layer reconstruction
  and trained-checkpoint audit are unusually direct weight-artifact evidence, bounded by the lack
  of a successful selective coefficient edit.
- Robot Learning Workshop: 25 to 40 percent. Strong robotics evidence and an honest specialist
  Physical-AI result, but no zero-shot, cross-task, or cross-environment generalization.

The VLM4RWD version is the best single submission today. Neural Network Artifacts is the strongest
alternative if the author wants the paper judged primarily as a model-weights contribution.

## Evidence likely to help in review

1. The $93.7\%\pm0.8\%$ three-seed result uses the published $280$-step, $50$-trial-per-task
   protocol and is supported by 1,500 closed-loop rollouts.
2. The same-skeleton models differ by only 1,280 parameters, or 0.006 percent. Complete all-task
   controls reach 448 of 500 conventional versus 426 rational on A30, and 451 versus 422 on A6000.
   The seed-0 rational deficit is 4.4 to 5.8 points and misses the predeclared three-point
   noninferiority margin on both platforms. Task 3 accounts for 21 lost successes in each matrix,
   but χ seeds 1 and 2 each reach 41 of 50 on that task, compared with 37 and 33 for their
   conventional counterparts. The task-specific deficit is not consistent across checkpoints.
   The χ checkpoint reaches 84.6 percent on A40, between its 85.2 percent A30 and 84.4 percent
   A6000 controls. The conventional A40 matrix is still running.
3. The operator claim is not based only on an architecture diagram. The trained checkpoint is
   mechanically audited, and representative rational and bilinear branches are reconstructed.
4. The exact attention calculation is mathematically specific and scales through a reduced Gram.
   The native audit reconstructs all 12 attention modules and 128 heads in each of three
   checkpoints, with maximum relative error below $2.6\times10^{-7}$. Two checkpoints meet the
   original $2\times10^{-5}$ absolute indicator, while the third reaches $2.23\times10^{-5}$.
5. The selected visual subspace has a replicated closed-loop causal consequence on independent
   canonical simulator states, not only an offline probe. Across three checkpoints, full,
   selected, and random conditions reach 47 of 60, 29 of 60, and 0 of 60 successes. Selected
   beats random with the same sign at every checkpoint, but it remains 30 points below full.
   The earlier near-lossless result does not replicate. Its ranking uses training-cache
   demonstrations, and the completed offline fidelity values reuse discovery samples.
6. A 300-pair, three-checkpoint instruction intervention shifts the relative end-effector
   preference toward the renamed object in 84 to 88 percent of trials. Only 28 to 33 percent end
   closer to that object. A seed-0 conventional control responds even more strongly but has the
   same 33 percent endpoint rate, so responsiveness is not misattributed to decomposability.
7. The paper reports its incomplete grounding and failed direct-edit gate. This makes
   the scope more trustworthy and fits workshops that welcome failure analysis or negative results.

## Main rejection risks

1. **The controlled training story is unstable.** The rational policy trails by 4.4 points on A30
   and 5.8 on A6000 at seed 0. The completed conventional seeds 1 and 2 reach only 9.0 and 8.2
   percent despite low final-minibatch losses, compared with 89.8 percent at seed 0. This is
   evidence of severe closed-loop seed sensitivity, not an across-seed estimate of conversion
   cost. Checkpoint provenance is also confounded because the strong conventional seed 0 is legacy,
   while seeds 1 and 2 were trained natively on Athena. The full χ seed-1 checkpoint reaches
   80.8 percent, but its trainer provenance is not matched to the collapsed conventional seed.
   The collapsed conventional checkpoints have training-cache normalized MSE of
   $9.51\times10^{-4}$ and $1.19\times10^{-3}$. This supports checkpoint integrity but cannot
   distinguish covariate shift from held-out generalization failure. Targeted task-3 runs reverse
   the seed-0 gap, but the native provenance matrix and partial-conversion controls are still needed.
2. **Incomplete task-level language grounding.** The longer paired rollout shows a replicated
   directional response to the renamed object, but only 28 to 33 percent of counterfactual runs end
   closer to it and the original one-step screen remains weak. The conventional control is stronger,
   so this experiment supports a failure diagnosis rather than an architectural advantage.
3. **Layerwise exactness only.** All deployed attention modules have tiny scale-aware
   reconstruction error, but the tractable
   exact decomposition is per attention layer. The whole-policy ranking uses activations and
   downstream sensitivity.
4. **Causal bottleneck is lossy.** The visual-subspace result now covers three checkpoints and
   60 paired trials over four tasks. The selected subspace beats random consistently, but loses
   18 of the 47 full-policy successes. Completed offline values reuse the discovery sample, so
   only the independent closed-loop canonical states support the replicated causal claim.
5. **No successful coefficient-level repair.** The full 15-configuration discovery sweep has no
   configuration that passes both preregistered 15-point gates. The best candidate has a 10-point
   keep advantage and a 40-point removal advantage. Its affected-weight relative norms are 56.4
   and 82.6 percent, so this is a broad functional ablation rather than a small semantic edit. The
   paper has an analysis surface, not yet a selective editing interface.
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

1. Complete the running Athena-native provenance matrix: conventional seed 0 and χ seeds 0 to 2,
   followed by 2,000 canonical trials. This tests whether conventional seeds 1 and 2 collapse because of
   seed sensitivity or legacy-versus-native checkpoint provenance. The same-hardware
   seed-0 matrices reject the predeclared three-point noninferiority claim at this seed. Use
   partial-conversion controls to localize the task-3 deficit.
2. Finish the three queued disjoint-sample offline diagnostics. The three-checkpoint closed-loop
   visual-subspace replication is complete and should be reported as a causal but lossy
   bottleneck. Do not promote it as near-lossless. The separate instruction intervention and
   conventional control are also complete.
3. Release an anonymous reproducibility package containing code, exact configs, checkpoint hashes,
   per-episode logs, and the mechanical audit output.

### Tier 2

4. Diagnose the failed coefficient edit before any further surgery rollout. Test redundant paths,
   reparameterization or gauge sensitivity, and the mapping from Gram eigenspaces to independent
   query-key-value edits. Do not extend the current sweep without a new method and independent
   discovery and confirmation split.
5. Add a focused grounding evaluation or repair that changes object-name binding without losing
   ordinary success. If no repair is ready, retain the failure analysis and avoid any claim of
   faithful grounding.
6. Report parameter-matched FLOPs, throughput, memory, eager latency, and compiled latency for
   rational and conventional models. This would also make AXIOM a credible fourth target.

### Tier 3

7. Complete the queued in-domain seed-0 χ and conventional controls on LIBERO-Spatial, LIBERO-Goal,
   and LIBERO-10. Restricted zero-shot runs with unseen words mapped to padding do not close this gap.
8. Strengthen related work and baseline coverage with current VLA robustness, weight-space
   interpretability, and structured-policy papers.

## Submission recommendation

Neural Network Artifacts is the safest target because its central exact-audit result is unaffected
by the nonzero capability cost. VLM4RWD remains a strong second choice because the grounding
failure analysis fits its call, while Robot Learning Workshop has the largest theme mismatch.
Finish the live native-provenance, in-domain cross-suite, partial-conversion, A40, and disjoint
offline jobs before the submission freeze, then publish the anonymous artifact and run a final
citation audit. Do not submit
the same empirical paper concurrently to multiple NeurIPS workshops without written permission from
every affected workshop chair.
