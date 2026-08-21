# NeurIPS 2026 workshop acceptance review

Assessment date: August 21, 2026.

These are subjective probability bands based on topical fit, paper completeness, and typical
workshop review standards. They are not official acceptance rates, and no experiment can ensure
acceptance.

## Current judgment

The submission-safe drafts now make one focused claim: a capable rational VLA exposes exact
layerwise tensor structure directly from its trained coefficients. The historical Modal capability
record and exact weight reconstruction are unaffected by the Athena task-index issue described
below. The old Athena rank-96 causal result is excluded from all four manuscripts while its clean
replacement runs.

Estimated acceptance bands for the current drafts, before the corrected confirmations finish:

- Neural Network Artifacts: 65 to 78 percent. This is the strongest present fit because the paper
  treats trained weights as an inspectable artifact and verifies every deployed attention module
  in three checkpoints.
- VLM4RWD: 45 to 60 percent. Capability and exact access are relevant, but the current evidence is
  a single-suite specialist study without a clean policy-level intervention result.
- Robot Learning Workshop: 35 to 50 percent. Exact structure is interesting, but this venue needs
  broader control evidence more than the artifact venue does.

If the corrected blind visual-bottleneck test, action-specificity control, convolutional-encoder
replication, and locked capability replay all pass their frozen gates, the estimated bands become:

- Neural Network Artifacts: 88 to 94 percent.
- VLM4RWD: 82 to 90 percent.
- Robot Learning Workshop: 72 to 84 percent.

A strong four-suite generalist that passes both the capability and 85 percent specialist-retention
gates could move Robot Learning Workshop into roughly the 82 to 90 percent range. These outcomes
would materially strengthen the papers, but they still cannot guarantee a review decision.

## Evidence that currently carries the paper

1. The historical convolutional rational policy record is $93.7\%\pm0.8\%$ over three seeds on
   LIBERO-Object, with a recorded three-checkpoint ensemble result of 96.2 percent. A locked replay
   with episode-level JSON is queued to restore complete numerical provenance.
2. The trained-checkpoint audit reconstructs all 12 attention modules and 128 heads in each of
   three checkpoints. Maximum relative error is below $2.6\times10^{-7}$.
3. All 36 primary block-head tests favor the rank-128 exact weight-derived attention subspace over
   equal-rank random controls. The median fidelity ratio is 2.44.
4. The real deployed graph passes a mechanical operator audit. Learned operations are bilinear,
   linear, or rational, rather than hidden incompatible nonlinear blocks.

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
outcome. Clean replacement experiments are already queued under new result names.

## Remaining rejection risks

1. **One validated specialist suite.** The current main capability evidence is LIBERO-Object. The
   papers must not claim broad robot generalization until corrected non-Object results finish.
2. **Layerwise exactness.** The exact weight reconstruction is per attention layer. It is not yet
   one compact symbolic contraction of the entire eight-block policy.
3. **Capability provenance.** The historical aggregate records survive, but the original 500-trial
   JSON files were overwritten. The locked replay must produce immutable episode-level artifacts.
4. **No current policy-level intervention claim.** The old visual-basis result has been removed.
   Only the corrected, frozen protocol can restore it.
5. **Instruction grounding.** Fixed LIBERO instructions do not establish robust language use.
6. **Ensemble cost.** The strongest recorded capability number uses three forward passes.
7. **Artifact readiness.** The anonymous package still needs hashes, environment pins, exact
   certificates, episode logs, and a one-command audit.

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
4. **Locked capability replay.** Evaluate the three frozen convolutional checkpoints and their
   arithmetic-mean ensemble at `highest` matrix precision, with checkpoint hashes and 500
   episode-level outcomes per model condition.
5. **Corrected four-suite generalist.** Train chi and conventional policies jointly on Object,
   Spatial, Goal, and LIBERO-10 for 160,000 updates across three seeds. The primary chi gate is at
   least 70 percent suite-macro success, every suite at least 50 percent, at least 32 of 40 tasks at
   50 percent or better, and every chi seed at least 60 percent.
6. **Specialist-retention control.** Train corrected seed-0 chi specialists for all four suites and
   require generalist seed 0 to retain at least 85 percent of their four-suite task-macro success.
7. **Pinned-cache provenance.** Compare cached task IDs, language joins, states, actions, and resized
   images against the pinned LeRobot revisions and record both cache and metadata hashes.

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
