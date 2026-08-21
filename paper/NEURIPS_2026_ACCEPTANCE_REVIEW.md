# NeurIPS 2026 workshop acceptance review

Assessment date: August 21, 2026.

These are subjective probability bands based on topical fit, paper completeness, and typical
workshop review standards. They are not official acceptance rates, and no experiment can ensure
acceptance.

## Current judgment

The submission-safe drafts now make one focused claim: a capable rational VLA exposes exact
layerwise tensor structure directly from its trained coefficients. The historical Modal capability
record and exact weight reconstruction are unaffected by the Athena task-index issue described
below. The old Athena rank-96 causal result is excluded from all four manuscripts while its
corrected replacement runs. Two new frozen certificates now bind exact joint-attention
reconstruction and rational-normalizer floating-point fidelity to the three released
convolutional checkpoint files associated with the historical capability aggregates.

Estimated acceptance bands for the current drafts, before the corrected confirmations finish:

- Neural Network Artifacts: 81 to 89 percent. This is the strongest present fit because the paper
  treats trained weights as an inspectable artifact, verifies every joint-attention head on the
  capability checkpoint family, and provides large-scale denominator and floating-point checks.
- VLM4RWD: 58 to 73 percent. Capability and exact access are relevant, but the current evidence is
  a single-suite specialist study without a clean policy-level instruction intervention result.
- Robot Learning Workshop: 45 to 61 percent. Exact structure is interesting, but this venue needs
  broader control evidence more than the artifact venue does.

If the corrected blind visual-bottleneck test, action-specificity control, convolutional-encoder
replication, and local instruction-specificity test all pass their frozen gates, the estimated
bands become:

- Neural Network Artifacts: 88 to 94 percent.
- VLM4RWD: 80 to 89 percent.
- Robot Learning Workshop: 68 to 80 percent.

A passing gated counterfactual target-swap test could move VLM4RWD into roughly the 85 to 92
percent range. A strong four-suite generalist that passes both the capability and 85 percent
specialist-retention gates could move Robot Learning Workshop into roughly the 84 to 91 percent
range. With all venue-specific gates passing, Neural Network Artifacts is estimated at 89 to 95
percent. These outcomes would materially strengthen the papers, but they still cannot guarantee a
review decision.

## Evidence that currently carries the paper

1. The historical convolutional rational policy record is $93.7\%\pm0.8\%$ over three seeds on
   LIBERO-Object, with a recorded three-checkpoint ensemble result of 96.2 percent. The papers label
   these as bounded historical aggregate records rather than new portability results.
2. On the three released convolutional checkpoint files associated with the historical capability
   aggregates, an independent certificate reconstructs all 4,608 joint-attention head-input cases
   with worst relative error $6.36\times10^{-16}$ against a frozen $10^{-6}$ gate.
3. The separate ViT trained-checkpoint audit reconstructs all 12 attention modules and 128 heads in
   each of three checkpoints. Maximum relative error is below $2.6\times10^{-7}$.
4. All 36 primary block-head tests favor the rank-128 exact weight-derived attention subspace over
   equal-rank random controls. The median fidelity ratio is 2.44.
5. Across 544,542,720 normalization rows, every deployed rational denominator is finite and
   positive, and action NRMSE from recomputing only the RationalNorm scales in float64 is at most
   $2.414\times10^{-6}$ against a frozen $10^{-3}$ gate.
6. The real deployed graph passes a mechanical operator audit. Learned operations are bilinear,
   linear, or rational, rather than hidden incompatible nonlinear blocks.
7. A standard-library one-command artifact checks eight immutable raw and summary files and
   independently recomputes both structural certificate decisions.

## Integrity correction

LeRobot stores suite-specific dataset task identifiers in an order that differs from the official
LIBERO benchmark order. Historical Modal training read the dataset metadata and is unaffected.
Athena closed-loop evaluations of frozen historical checkpoints that use only official simulator
tasks are also unaffected.

Pre-correction Athena-native training and cache-backed visual-basis analysis used raw dataset task
identifiers as official identifiers. Those runs paired most cached images with the wrong
instruction and compromised the intended discovery and holdout split. Their numerical results are
excluded from every workshop paper. The corrected code now reads pinned LeRobot metadata, maps
dataset identifiers to official identifiers before splitting, and pairs every cached frame with
its dataset-indexed instruction.

This correction lowers the temporary acceptance estimates because the earlier policy-level causal
evidence no longer counts. It is an infrastructure audit correction, not a negative scientific
outcome. Corrected replacement experiments are already queued under new result names.

## Remaining rejection risks

1. **One validated specialist suite.** The current main capability evidence is LIBERO-Object. The
   papers must not claim broad robot generalization until corrected non-Object results finish.
2. **Layerwise exactness.** The exact weight reconstruction is per attention layer. It is not yet
   one compact symbolic contraction of the entire eight-block policy.
3. **Capability provenance.** The historical aggregate records survive, but the original 500-trial
   JSON files were overwritten. The papers therefore qualify them as bounded historical evidence.
4. **No current policy-level intervention claim.** The old visual-basis result has been removed.
   Only the corrected, frozen protocol can restore it.
5. **Instruction grounding.** Fixed LIBERO instructions do not establish robust language use.
6. **Ensemble cost.** The strongest recorded capability number uses three forward passes.
7. **Artifact readiness.** The anonymous package now verifies the structural certificates and
   immutable hashes in one command. Full figure regeneration and unavailable historical
   episode-level capability logs remain outside that package.

## Frozen acceptance-critical experiments

The following protocols were fixed before their corrected outcomes were observed:

1. **Corrected blind visual bottleneck.** Build one balanced action-Jacobian basis from official
   Object tasks 0 to 3, keep rank 96 fixed, and evaluate official tasks 8 and 9 with 50 canonical
   states per task, three ViT checkpoints, and three fixed random controls.
2. **Action-specificity control.** On the same discovery examples and blind rollouts, compare the
   Jacobian basis with an equal-rank uncentered activation-energy basis. Promote specificity only
   if the Jacobian basis retains at least 90 percent of full-policy successes, wins by at least 10
   points pooled and in every checkpoint, and has a task-and-checkpoint-stratified bootstrap lower
   bound above zero while the energy basis preserves raw activations at least as well.
3. **Independent encoder family.** Repeat the fixed rank-96 blind protocol with three verified
   convolutional checkpoints. Report it separately from the ViT result.
4. **Local instruction specificity.** On held-out canonical Object states, compare eight-action
   displacement under the target prompt, five co-present-object prompts, and four absent-object
   prompts across three fixed ViT checkpoints. A pass is required before the target-swap graph can
   start.
5. **Gated counterfactual target swap.** If local specificity passes, execute all 50 prespecified
   target-only BDDL rewrites on episodes 40 to 49 across the same three checkpoints. This is a new
   closed-loop intervention but not an independent-state replication of the prerequisite.
6. **Corrected four-suite generalist.** Train chi and conventional policies jointly on Object,
   Spatial, Goal, and LIBERO-10 for 160,000 updates across three seeds. The primary chi gate is at
   least 70 percent suite-macro success, every suite at least 50 percent, at least 32 of 40 tasks at
   50 percent or better, and every chi seed at least 60 percent.
7. **Specialist-retention control.** Train corrected seed-0 chi specialists for all four suites and
   require generalist seed 0 to retain at least 85 percent of their four-suite task-macro success.
8. **Pinned-cache provenance.** Compare cached task IDs, language joins, states, actions, and resized
   images against the pinned LeRobot revisions and record both cache and metadata hashes. This
   audit is complete: all 271,996 frames across the four suites match exactly with zero field
   mismatches.

The blind bottleneck is positive only if selected conditional retention is at least 85 percent,
selected beats every matched random control with the same sign in every checkpoint, and the full
policy supplies a meaningful success denominator. Corrected results are reported in full. A failed
gate removes the corresponding claim instead of triggering another rank or task search.

## Submission recommendation

Submit Neural Network Artifacts first if workshop policies prohibit overlapping submissions. It
has the clearest reviewer contract even before the new runs finish. VLM4RWD becomes competitive
with a clean blind bottleneck and action-specificity result. Robot Learning Workshop needs the
four-suite generalist and specialist-retention result to approach the requested 85 to 90 percent
band. Before any submission, lock one evaluation environment, publish the anonymous artifact, and
run a final claim-to-artifact and citation audit.
