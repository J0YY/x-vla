# NeurIPS 2026 workshop acceptance review

Assessment date: August 26, 2026.

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
across visual encoder families. On the same sixteen deterministic inputs per convolutional
checkpoint, independent certificates now cover every joint-attention head, every joint FFN, and
every complete joint block. A separate ViT audit exactly partitions each action-query attention
update into exhaustive vision, instruction, robot-state, and action-query source groups on 128
task-balanced inputs per checkpoint. The partition magnitudes are descriptive and do not establish
causal importance or grounding.

A separate prospectively frozen closed-loop test now shows bounded instruction necessity. Across
100 unique canonical Object states and three frozen ViT checkpoints, the correct familiar
instruction succeeds on 247 of 300 checkpoint-state pairs. An observation-verified co-present
control instruction succeeds on 72 of 300, and an empty instruction succeeds on 78 of 300. The
paired gaps are 58.3 and 56.3 points. They are positive for every checkpoint aggregate and every
task aggregate. One-sided exact McNemar tests over the 300 checkpoint-state pairs give
$p=2.02\times10^{-45}$ and $p=1.52\times10^{-44}$. These pooled tests are not cluster-robust
inference. A frozen task-stratified state-cluster bootstrap over 100 unique states gives 95 percent
intervals of 51.0 to 65.7 and 50.0 to 62.3 points. This is a behavioral evidence package, not an
additional structural certificate.

A second prospectively frozen study isolates the learned lexical-embedding path on disjoint
episodes 30 to 39. Both arms receive the identical correct 32-token ID tensor. The ablated arm
zeros only the post-lookup token vectors while preserving sequence length, positions, vision,
robot state, embodiment, the separate learned common BOS, action queries, and all weights. The
intact path succeeds on 263 of 300 checkpoint-state pairs, compared with 63 of 300 after zeroing,
for a 66.7-point gap. Full-path success is 84, 87, and 92 percent by checkpoint. All ten task
aggregates are positive, the one-sided paired exact p-value is $1.42\times10^{-56}$, and the frozen
state-cluster bootstrap interval is 61.7 to 71.3 points. This strengthens the evidence that these
specialists use learned instruction vectors. It does not establish semantic grounding or
generalization.

The breadth package now closes the earlier single-suite objection. Three jointly trained
$20.1$M checkpoints spanning all $40$ familiar LIBERO tasks reach $70.35\%\pm1.32\%$
task-macro success, compared with $69.12\%\pm2.33\%$ for same-recipe conventional controls
whose parameter counts differ by $0.006\%$. The fixed mean-prediction ensemble reaches $74.75\%$
with every suite at or above $50\%$, and the equal-update seed-$0$ joint checkpoint retains
$95.16\%$ of four suite specialists' pooled success. The anonymous artifact independently
recomputes these quantities from $12{,}000$ canonical outcomes across $96$ immutable raw shards.
All tasks appeared in training, so this establishes multi-suite capacity rather than unseen-task
transfer.

Estimated acceptance bands after the four-suite matched-control result, submission-format audit,
and final claim-to-artifact review:

- Neural Network Artifacts: 91 to 96 percent. The checkpoint is a behaviorally capable artifact,
  its structural claims have executable verifiers, and the matched four-suite result reduces the
  risk that the artifact is only a single-suite curiosity. The remaining limitation is that the
  learned-forward replay is input-conditioned rather than one compact symbolic contraction.
- VLM4RWD: 86 to 93 percent. Two disjoint paired studies show both prompt-level instruction
  necessity and necessity of the learned post-lookup lexical path, while matched four-suite
  capability strengthens the deployment relevance. The evidence still does not show
  counterfactual goal completion, unseen language, or conflicting-cue resolution.
- Robot Learning Workshop: 77 to 88 percent. The four-suite joint policies and specialist-retention
  comparison close the former breadth blocker. Fit remains limited because every task and
  instruction appeared in training and there is no physical-robot or zero-shot evaluation.

The largest remaining reviewer objection differs by venue:

- Neural Network Artifacts: the sequential capable-ViT replay remains input-conditioned. It is not
  one compact symbolic contraction and does not provide an input-general numerical certificate.
- VLM4RWD: the paired controls establish familiar-task instruction necessity for the original goal,
  not counterfactual goal selection. All tested task instructions appeared during policy training,
  and co-presence in simulator fields does not guarantee unoccluded camera visibility.
- Robot Learning Workshop: the validated breadth is multi-suite but remains familiar-task
  simulation. The paper does not demonstrate zero-shot, unseen-task, or physical-robot behavior.

The stronger action-specificity gate did not pass. Pooled across all three checkpoints, the
activation-energy basis retains $267/272$ baseline successes and the action-Jacobian basis retains
$259/272$. The action-Jacobian minus activation-energy difference is $-2.9$ points, with a frozen
paired resampling interval from $-6.6$ to $+1.4$ points. The supported result is therefore a
structured low-dimensional visual bottleneck relative to random projectors, not a uniquely
action-Jacobian-specific mechanism.

The counterfactual goal-following study did not pass its frozen gate. Checkpoint 0 reached only
49/500 counterfactual-goal matching successes, or 9.8 percent, below the fixed 20 percent minimum,
so the remaining checkpoints were stopped and the result is not presented as positive evidence.
The separate lexical-path ablation passes and can be reported within its narrower claim. The
four-suite generalist and the 85 percent specialist-retention gate have now passed. Grounding or
paraphrase evidence would be the most direct remaining gain for VLM4RWD, but neither is a bounded
acceptance-critical follow-up at this stage. No combination of outcomes can guarantee a review
decision.

## Evidence that currently carries the paper

1. Three independently trained 20.1M ViT policies reach 426/500, 404/500, and 450/500 on
   LIBERO-Object, for $85.3\%\pm4.6\%$. A fixed elementwise prediction-mean ensemble reaches
   467/500, or 93.4 percent, with three forward passes. All $2{,}000$ episode rows are immutable.
2. Three jointly trained checkpoints span $40$ familiar tasks and reach $70.35\%\pm1.32\%$
   task-macro success, compared with $69.12\%\pm2.33\%$ for the near-equal-parameter
   conventional controls. Their fixed ensemble reaches $74.75\%$, every suite is at or above
   $50\%$, and the equal-update seed-$0$ joint checkpoint retains $95.16\%$ of the four
   specialists' pooled success. A standard-library verifier recomputes the result from $12{,}000$
   raw outcomes.
3. The prospectively frozen instruction study contains 100 unique canonical states, 300
   checkpoint-state pairs, and 900 full 280-step rollouts. Correct, co-present control, and empty
   instructions succeed on 247/300, 72/300, and 78/300 pairs. Both paired gaps exceed 56 points,
   pass the exact paired and state-cluster bootstrap gates, and remain positive for every checkpoint
   aggregate and task aggregate. Fifteen immutable raw shards, the manifest, smoke, strict summary,
   and verifier are released.
   A disjoint lexical-path ablation adds another 300 paired checkpoint-state evaluations. Exact
   post-lookup zeroing changes success from 263/300 to 63/300, a 66.7-point gap with all ten task
   aggregates positive and a 61.7 to 71.3-point state-cluster bootstrap interval.
4. On three separate convolutional checkpoints and the same sixteen deterministic inputs per
   checkpoint, independent certificates reconstruct all 4,608 joint-attention head-input cases,
   all 384 joint-FFN module-input cases, and all 384 complete joint-block input cases. Worst
   relative errors are $6.36\times10^{-16}$, $1.3871\times10^{-15}$, and
   $1.5024\times10^{-7}$ against frozen $10^{-6}$, $10^{-10}$, and $10^{-4}$ gates.
5. The corrected-input ViT exact-attention records name the same checkpoint artifacts and
   numerically replay all 12 modules on one fixed cached input per checkpoint, covering 36
   module-input cases and 384 architectural heads. Maximum relative error is
   $2.016119\times10^{-7}$. A standard-library verifier binds all raw result, cache, provenance,
   runtime, and source identities and recomputes the certificate.
   A separate sequential replay rebuilds four vision and eight joint learned blocks through the
   linear action output on 48 fixed inputs. The worst final-action relative error is
   $3.9822\times10^{-7}$ against a frozen $10^{-3}$ gate, and the manual deployed traversal equals
   the unmodified model call bitwise.
6. On 128 deterministic, official-task-balanced inputs for each of the three ViT capability
   checkpoints, exhaustive vision, instruction, robot-state, and action-query groups reconstruct
   all 36,864 action-query head-input updates and 3,072 projected module-input cases. Maximum
   relative error is $5.0741\times10^{-7}$ against a frozen $10^{-6}$ gate. Mean coherent-energy
   shares are 67.7 percent vision, 19.2 percent robot state, 7.9 percent action query, and 5.1
   percent instruction. These shares are descriptive rather than causal.
7. All 36 primary block-head tests favor the rank-128 exact weight-derived attention subspace over
   equal-rank random controls. The median fidelity ratio is 2.44.
8. On corrected-projector holdout tasks 8 and 9, learned rank-96 visual projectors retain $259/272$ and
   $267/272$ baseline successes. Three equal-rank random projectors retain $37/272$, $9/272$, and
   $22/272$. A standard-library verifier recomputes all 12 raw files and paired intervals.
9. Across 544,542,720 normalization rows, every deployed rational denominator is finite and
   positive, and action NRMSE from recomputing only the RationalNorm scales in float64 is at most
   $2.414\times10^{-6}$ against a frozen $10^{-3}$ gate.
10. The real deployed graph passes a mechanical operator audit. Learned operations are bilinear,
   linear, or rational, rather than hidden incompatible nonlinear blocks.
11. Standard-library artifact verifiers check the immutable capability and structural files,
   recompute all success totals, and independently recompute all five structural certificate
   decisions. The structural package binds fifteen raw JSONs, five summaries, and two exact source
   snapshots through an immutable SHA-256 manifest. The separate instruction package binds fifteen
   rollout shards, its canonical-state manifest, smoke, and strict passing summary.

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

This correction removed the earlier policy-level causal evidence. It was an infrastructure audit
correction, not a negative scientific outcome. All evidence retained in the workshop papers now
uses corrected mappings or simulator-native official task identifiers.

## Remaining rejection risks

1. **Familiar-task scope.** The validated breadth now spans all four LIBERO suites, but every task
   appeared in training. The papers must not claim unseen-task, zero-shot, or physical-robot
   generalization.
2. **Input-conditioned rather than symbolic whole-policy exactness.** Attention, FFN, complete
   blocks, and a sequential learned-forward replay now pass. The $48$-input replay is still not one
   compact input-general tensor contraction of the capable ViT policy.
3. **Small matched-control difference.** The three-seed joint-policy mean is $1.23$ points above
   the conventional mean. This supports near-zero observed cost under the fixed recipe, not a
   statistical superiority claim.
4. **No unique ranking mechanism.** The corrected intervention supports a structured low-dimensional
   visual bottleneck relative to random projectors. It does not show that action-Jacobian ranking is
   superior to a matched activation-energy basis.
5. **Bounded instruction evidence.** The paired study establishes necessity for familiar Object-suite
   instructions and the original task goal. It does not establish counterfactual goal following,
   paraphrase robustness, unseen-object composition, or zero-shot language use.
6. **Ensemble cost.** The strongest recorded capability number uses three forward passes.
7. **Artifact usability.** The anonymous package verifies the structural certificates, capability
   protocol, $2{,}000$ Object episode rows, $12{,}000$ breadth outcomes, corrected visual-bottleneck
   matrix, and instruction-control rollouts. The remaining risk is reviewer friction from the
   package's breadth, not a missing claimed endpoint.

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
4. **Local instruction specificity, not passed.** On canonical Object states, compare eight-action
   displacement under the target prompt, five co-present-object prompts, and four absent-object
   prompts across three fixed ViT checkpoints. Mean specificity ranks are 0.570, 0.586, and 0.592,
   below the frozen 0.75 requirement in every checkpoint. The pooled descriptive rank is 0.583.
   The positive semantic displacement does not override this failed comparator gate.
5. **Gated counterfactual target swap, not run.** The prerequisite in item 4 did not pass, so the
   frozen target-swap graph failed closed before any rollout. It supplies no evidence and is not
   revived by the later instruction-necessity study.
6. **Full-horizon instruction necessity, passed.** This distinct endpoint was designed after the
   local displacement-rank gate became impossible and before its own outcomes. On episodes 40 to
   49, all three checkpoints receive correct, deterministic co-present control, and empty
   instructions from identical physical starts. Correct success is 247/300, compared with 72/300
   and 78/300. All nine frozen gates pass. The endpoint shares canonical states with the failed
   local study but uses a new full-horizon original-goal outcome, so it is not a rescue reanalysis.
7. **Conditional counterfactual goal following, not passed.** This protocol was designed after the
   local rank failure and before the instruction-necessity outcomes. It uses independent episodes
   20 to 29, all prespecified co-present target swaps, three frozen ViT checkpoints, paired original
   and counterfactual goals, and full closed-loop success. Checkpoint $0$ reaches $49/500=9.8\%$,
   below the frozen $20\%$ gate. The remaining checkpoints were stopped, and the endpoint supplies
   no positive claim.
8. **Corrected four-suite generalist, passed.** Train chi and conventional policies jointly on Object,
   Spatial, Goal, and LIBERO-10 for 160,000 updates across three seeds. The primary chi gate is at
   least 70 percent suite-macro success, every suite at least 50 percent, at least 32 of 40 tasks at
   50 percent or better, and every chi seed at least 60 percent. The three seeds reach
   $70.85\%$, $71.35\%$, and $68.85\%$. The mean is $70.35\%\pm1.32\%$, and the frozen
   ensemble reaches $74.75\%$ with every suite above $50\%$.
9. **Specialist-retention control, passed.** Train corrected seed-0 chi specialists for all four
   suites and require generalist seed 0 to retain at least 85 percent of their four-suite
   task-macro success. The retained fraction is $70.85/74.45=95.16\%$.
10. **Pinned-cache provenance.** Compare cached task IDs, language joins, states, actions, and resized
   images against the pinned LeRobot revisions and record both cache and metadata hashes. This
   audit is complete: all 271,996 frames across the four suites match exactly with zero field
   mismatches.
11. **Joint-FFN certificate, passed.** All $384$ module-input cases are finite. The worst relative L2
   error is $1.3871\times10^{-15}$ against the frozen $10^{-10}$ gate.
12. **Complete joint-block certificate, passed.** Independent RationalNorm, attention, residual,
    and bilinear-FFN equations reconstruct all $384$ block-input cases with worst relative L2 error
    $1.5024\times10^{-7}$ against the frozen $10^{-4}$ gate. This tests input-conditioned,
    per-block composition, not a sequential end-to-end policy contraction.
13. **Exact modality-contribution audit, passed.** Exhaustive source groups reconstruct all
    $36{,}864$ action-query head-input cases and $3{,}072$ projected module-input cases with maximum
    relative L2 error $5.0741\times10^{-7}$ against the frozen $10^{-6}$ gate. Group magnitudes and
    familiar-prompt permutations remain descriptive and do not establish grounding or causal
    necessity.

The fixed-rank bottleneck is positive only if selected conditional retention is at least 85 percent,
selected beats every matched random control with the same sign in every checkpoint, and the full
policy supplies a meaningful success denominator. Corrected results are reported in full. A failed
gate removes the corresponding claim instead of triggering another rank or task search.

Rank 96 was fixed before corrected-basis outcomes, but the earlier rank-selection screen was
invalidated by the task-map audit. Tasks 8 and 9 are absent from the corrected projector discovery,
not a clean rank-selection holdout. They also appeared during policy training.

## Submission recommendation

Submit Neural Network Artifacts first if workshop policies prohibit overlapping submissions. It
has the clearest reviewer contract and is the only individual venue whose present subjective band
is entirely above 90 percent. VLM4RWD is also credible, with matched multi-suite capability and a
strong familiar-task instruction-necessity result, but its remaining grounding risk is real. The
Robot Learning Workshop version now has the breadth and retention evidence it previously lacked,
while its zero-shot theme remains a fit limitation. The final claim-to-artifact, citation, page-limit,
and visual audits pass. No additional outcome-seeking experiment is justified before submission.
