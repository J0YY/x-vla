# NeurIPS 2026 workshop acceptance review

Assessment date: August 21, 2026.

These are subjective probability bands based on topical fit, paper completeness, and typical
workshop review standards. They are not official acceptance rates and no experiment can guarantee
acceptance.

## Current judgment

The workshop paper now has a focused positive thesis: a capable rational VLA can expose exact
layerwise tensor structure and a compact causal visual bottleneck without requiring a separate
post-hoc model. The strongest new result is the prespecified rank-96 confirmation. A selected
96-dimensional subspace, 25 percent of the visual width, retains 49 of 52 successes of the full
policy on independent canonical rollouts. Three matched random controls retain only 7 of 180.

The tailored versions deliberately do not pursue coefficient editing, counterfactual language
repair, or an across-seed conversion-cost estimate. Omitting those experiments is defensible because
the corresponding claims were removed as well. The retained scope is capability, exact layerwise
structure, and causal compression.

Estimated acceptance bands for the current versions:

- Neural Network Artifacts: 75 to 85 percent. This is the best fit. The paper studies the trained
  weights as an artifact, mechanically verifies every deployed attention module in three
  checkpoints, and connects that structure to a compact causal subspace.
- VLM4RWD: 65 to 78 percent. The capable VLA and causal policy analysis fit well, but the evaluation
  is still limited to one LIBERO suite and the causal ranking is data driven at policy scale.
- Robot Learning Workshop: 50 to 65 percent. The closed-loop evidence is substantial, but broader
  robot-learning generalization would help more here than at the artifact workshop.

Conditional bands if the blind task-8/9 confirmation, independent convolutional-encoder
replication, and ensemble replay all pass their frozen gates:

- Neural Network Artifacts: 85 to 92 percent.
- VLM4RWD: 78 to 88 percent.
- Robot Learning Workshop: 65 to 78 percent. A positive in-domain second-suite result would be
  needed to push this target higher.

## Evidence that now carries the paper

1. The convolutional rational policy reaches $93.7\%\pm0.8\%$ over three seeds on the published
   280-step, 50-trial-per-task protocol, supported by 1,500 closed-loop rollouts. A three-checkpoint
   mean ensemble reaches 96.2 percent.
2. The trained-checkpoint audit reconstructs all 12 attention modules and 128 heads in each of
   three checkpoints. Maximum relative error is below $2.6\times10^{-7}$.
3. On discovery tasks 0 to 3 and evaluation tasks 4 to 7, the frozen rank-96 selected subspace
   reaches 50 of 60 successes, compared with 52 of 60 for the full policy. It retains 49 of 52
   baseline successes, or 94.2 percent conditional retention.
4. The same selected-over-random direction holds at every checkpoint. The three random controls per
   checkpoint pool to 7 of 180 successes, or 3.9 percent.
5. The subspace uses 96 of 384 visual dimensions and captures 59.5 percent of the balanced
   downstream-sensitivity spectrum. Its advantage is therefore not explained by choosing nearly
   the full representation.
6. Disjoint-sample offline checks show median random-to-selected error ratios from 33.5 to 110.6
   across output groups and checkpoints. These checks support fidelity, while the independent
   canonical rollouts carry the causal claim.

## Remaining rejection risks

1. **One specialist suite.** The main evidence is on LIBERO-Object. The paper must describe this as
   a controlled specialist-policy study, not broad VLA generalization.
2. **Layerwise exactness.** The weight-only reconstruction is exact per attention layer. The
   policy-scale ranking uses cached activations and downstream gradients.
3. **Basis discovery data.** Basis discovery uses training-cache demonstrations. The causal
   evaluation is independent closed-loop rollout, but a second task holdout and an independent
   encoder family would make the separation much harder to dismiss.
4. **Ensemble cost.** The strongest capability number uses three inference models. Single-model
   results and the threefold inference cost must remain visible.
5. **Artifact readiness.** An anonymous package still needs checkpoint hashes, episode-level files,
   exact environment pins, and a one-command audit.
6. **Numerical provenance.** Historical and Athena evaluations differ in matrix-precision and
   simulator versions. Only results from a single locked environment should be compared in one
   table.

## Frozen high-value confirmation plan

The following jobs were chosen before seeing their outcomes:

1. **Blind task holdout:** evaluate the frozen rank-96 basis on tasks 8 and 9, which were used in
   neither basis discovery nor rank selection. Run 50 canonical states per task, three checkpoints,
   and three matched random controls.
2. **Independent encoder family:** repeat the fixed rank-96 protocol with the three verified
   convolutional checkpoints. This tests whether the causal bottleneck is a transformer-specific
   artifact.
3. **Same-GPU rank curve:** run ranks 64, 96, 128, and 192 sequentially in one allocation per
   checkpoint. This removes GPU assignment as a rank-sweep confound.
4. **Capability replication:** rerun the verified convolutional ensemble and replay the ViT models
   under the historical `highest` matrix-precision setting. Do not combine numbers across precision
   settings.
5. **In-domain second suites:** finish native training and canonical evaluation on LIBERO-Spatial,
   LIBERO-Goal, and LIBERO-10. Zero-shot controls with out-of-vocabulary instructions are not a
   substitute.

The blind causal result should be promoted only if selected retention remains at least 85 percent of
full-policy successes, selected beats each matched random control with the same sign across
checkpoints, and ordinary full-policy success remains high enough to make retention meaningful. The
independent-encoder result should be reported separately rather than pooled with the ViT result.

## Submission recommendation

Submit the Neural Network Artifacts version first if workshop policies prohibit overlapping
submissions. It already has the clearest reviewer contract. VLM4RWD is the best broader VLA target.
Robot Learning Workshop becomes substantially stronger only with a positive second-suite result.
Before submission, lock the evaluation environment, publish the anonymous artifact, and run a final
claim-to-artifact and citation audit.
