# NeurIPS 2026 workshop acceptance review

Assessment date: August 21, 2026.

These are subjective probability bands based on topical fit, paper completeness, and typical
workshop review standards. They are not official acceptance rates, and no experiment can ensure
acceptance.

## Current judgment

The submission-safe drafts make one focused claim: a capable rational VLA exposes exact layerwise
tensor structure directly from its trained coefficients. The capability anchor is three ViT
checkpoint artifacts also named by the exact-attention records. All $2{,}000$ member and ensemble
episode rows, protocol fields, and SHA-256 identities are preserved and independently verified.
The corrected rank-96 visual-bottleneck program has now passed its frozen fixed-rank selectivity gate.
Across three ViT checkpoints, the structured projector retains $259/272$ full-policy successes on
tasks 8 and 9, which are absent from corrected projector construction. Three fixed equal-rank random projectors retain only $37/272$,
$9/272$, and $22/272$. These tasks were used during policy training, so the result is not unseen-task
generalization. Separate convolutional checkpoints test whether structural certificates transfer
across visual encoder families.

Estimated acceptance bands after integrating the corrected fixed-rank result:

- Neural Network Artifacts: 84 to 91 percent. This is the strongest present fit because the paper
  treats trained weights as an inspectable artifact, unifies capability and exact attention on the
  same checkpoint family, and now includes a policy-level closed-loop intervention.
- VLM4RWD: 66 to 79 percent. Capability, exact access, and a compact visual bottleneck are relevant,
  but the current evidence does not establish robust instruction grounding.
- Robot Learning Workshop: 52 to 67 percent. The bottleneck result adds intervention evidence, but
  this venue still needs broader control evidence more than the artifact venue does.

The stronger action-specificity gate did not pass. Pooled across all three checkpoints, the
activation-energy basis retains $267/272$ baseline successes and the action-Jacobian basis retains
$259/272$. The action-Jacobian minus activation-energy difference is $-2.9$ points, with a frozen
paired resampling interval from $-6.6$ to $+1.4$ points. The supported result is therefore a
structured low-dimensional visual bottleneck relative to random projectors, not a uniquely
action-Jacobian-specific mechanism.

If the convolutional-encoder replication and local instruction-specificity test pass their frozen
gates, the estimated bands become:

- Neural Network Artifacts: 87 to 93 percent.
- VLM4RWD: 75 to 85 percent.
- Robot Learning Workshop: 61 to 74 percent.

A passing gated counterfactual target-swap test could move VLM4RWD into roughly the 83 to 90
percent range. A strong four-suite generalist that passes both the capability and 85 percent
specialist-retention gates could move Robot Learning Workshop into roughly the 82 to 89 percent
range. With all venue-specific gates passing, Neural Network Artifacts is estimated at 88 to 94
percent. These outcomes would materially strengthen the papers, but they still cannot guarantee a
review decision.

## Evidence that currently carries the paper

1. Three independently trained 20.1M ViT policies reach 426/500, 404/500, and 450/500 on
   LIBERO-Object, for $85.3\%\pm4.6\%$. A fixed elementwise prediction-mean ensemble reaches
   467/500, or 93.4 percent, with three forward passes. All $2{,}000$ episode rows are immutable.
2. On three separate convolutional checkpoints, an independent certificate reconstructs all
   4,608 joint-attention head-input cases
   with worst relative error $6.36\times10^{-16}$ against a frozen $10^{-6}$ gate.
3. The corrected-input ViT exact-attention records name the same checkpoint artifacts and
   numerically replay all 12 modules on one fixed cached input per checkpoint, covering 36
   module-input cases and 384 architectural heads. Maximum relative error is
   $2.016119\times10^{-7}$. A standard-library verifier binds all raw result, cache, provenance,
   runtime, and source identities and recomputes the certificate.
4. All 36 primary block-head tests favor the rank-128 exact weight-derived attention subspace over
   equal-rank random controls. The median fidelity ratio is 2.44.
5. On corrected-projector holdout tasks 8 and 9, learned rank-96 visual projectors retain $259/272$ and
   $267/272$ baseline successes. Three equal-rank random projectors retain $37/272$, $9/272$, and
   $22/272$. A standard-library verifier recomputes all 12 raw files and paired intervals.
6. Across 544,542,720 normalization rows, every deployed rational denominator is finite and
   positive, and action NRMSE from recomputing only the RationalNorm scales in float64 is at most
   $2.414\times10^{-6}$ against a frozen $10^{-3}$ gate.
7. The real deployed graph passes a mechanical operator audit. Learned operations are bilinear,
   linear, or rational, rather than hidden incompatible nonlinear blocks.
8. Standard-library artifact verifiers check the immutable capability and structural files,
   recompute all success totals, and independently recompute both structural certificate decisions.

## Integrity correction

LeRobot stores suite-specific dataset task identifiers in an order that differs from the official
LIBERO benchmark order. Historical Modal training read the dataset metadata and is unaffected.
The preserved ViT capability evaluations use official simulator tasks and are also unaffected.

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
3. **Single-suite capability.** The capability artifact is now complete and immutable, but all
   current closed-loop results still come from LIBERO-Object.
4. **No unique ranking mechanism.** The corrected intervention supports a structured low-dimensional
   visual bottleneck relative to random projectors. It does not show that action-Jacobian ranking is
   superior to a matched activation-energy basis.
5. **Instruction grounding.** Fixed LIBERO instructions do not establish robust language use.
6. **Ensemble cost.** The strongest recorded capability number uses three forward passes.
7. **Artifact readiness.** The anonymous package now verifies the structural certificates,
   capability protocol, $2{,}000$ capability episode rows, corrected visual-bottleneck episode
   matrix, totals, paired intervals, and immutable hashes. The remaining artifact work is the
   venue-specific evidence as those frozen runs finish.

## Frozen acceptance-critical experiments

The following protocols were fixed before their corrected outcomes were observed:

1. **Corrected fixed-rank visual bottleneck, passed.** Build one balanced action-Jacobian basis from
   official Object tasks 0 to 3, keep rank 96 fixed before corrected outcomes, and evaluate official tasks 8 and 9 with 50
   canonical states per task, three ViT checkpoints, and three fixed random controls. The projector
   retains $259/272$ baseline successes, versus $37/272$, $9/272$, and $22/272$ for the controls.
2. **Action-specificity control, not passed.** On the same discovery examples and fixed-rank rollouts, compare the
   Jacobian basis with an equal-rank uncentered activation-energy basis. Promote specificity only
   if the Jacobian basis retains at least 90 percent of full-policy successes, wins by at least 10
   points pooled and in every checkpoint, and has a task-and-checkpoint-stratified bootstrap lower
   bound above zero while the energy basis preserves raw activations at least as well.
3. **Independent encoder family.** Repeat the corrected fixed rank-96 protocol with three verified
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
9. **Joint-FFN certificate.** Replay all eight bilinear FFNs on the same deterministic inputs used
   by the convolutional attention certificate. The frozen gate requires every output to be finite
   and the worst module relative L2 error to be at most $10^{-10}$ across three checkpoints.
10. **Complete joint-block certificate.** Independently rebuild both RationalNorms, attention,
    residual gains and ordering, bilinear FFN, and both residual paths for all eight joint blocks.
    The frozen gate is finite output and maximum relative L2 error at most $10^{-4}$ across $384$
    block-input cases. This tests composition, not an end-to-end policy contraction.
11. **Exact modality-contribution audit.** Decompose each ViT joint-attention head output into
    exhaustive vision, instruction, state, and action-query source groups on $128$ balanced inputs
    per checkpoint. Exact reconstruction is gated at relative L2 error at most $10^{-6}$. Group
    magnitudes are descriptive and do not establish grounding or causal necessity.

The fixed-rank bottleneck is positive only if selected conditional retention is at least 85 percent,
selected beats every matched random control with the same sign in every checkpoint, and the full
policy supplies a meaningful success denominator. Corrected results are reported in full. A failed
gate removes the corresponding claim instead of triggering another rank or task search.

Rank 96 was fixed before corrected-basis outcomes, but the earlier rank-selection screen was
invalidated by the task-map audit. Tasks 8 and 9 are absent from the corrected projector discovery,
not a clean rank-selection holdout. They also appeared during policy training.

## Submission recommendation

Submit Neural Network Artifacts first if workshop policies prohibit overlapping submissions. It
has the clearest reviewer contract and is already in the requested high-probability band. VLM4RWD
needs a clean instruction-specificity result to reach the same band. Robot Learning Workshop needs the
four-suite generalist and specialist-retention result to approach the requested 85 to 90 percent
band. Before any submission, lock one evaluation environment, publish the anonymous artifact, and
run a final claim-to-artifact and citation audit.
