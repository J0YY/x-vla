**Athena ODT campaign, 7 September 2026**

**The campaign establishes a useful small-cut regime and a native-policy FFN intervention. It does not establish that ODT preserves closed-loop capability better than the matched random control.** Spectral allocation substantially improves the original small-removal block curve. At large removal it can make decoded error worse. The native action experiment provides a stronger connection to policy outputs, but offline fidelity and successful control remain different endpoints.

This publication records the accepted September 7 campaign on checkpoint `b1c0dfce…`. It certifies the stated local representations and measured panels. Corrected full-policy ODT remains open. Two independently initialized agents implemented and reviewed the work without inherited conversation history.

**What ran and what was held fixed.** The primary study contains 25 distinct conditions after exact rank-vector/basis/seed deduplication. Each condition evaluates 320 real activation pairs and 256 newly drawn Gaussian inputs. The real pairs come from 160 frames in 80 episodes across all ten LIBERO-Object tasks, with two fixed token pairs per frame. Another 20 episodes supplied development checks. Episode identities and conditions were frozen before reduced-model evaluation. The real frames come from the checkpoint's training-data source, so this is not unseen-training-data generalization.

Every run uses checkpoint `b1c0dfce…`, the accepted symmetric ordered lift, unchanged direct QR/RQ rules, and the declared scale ledger. No SVD, pseudoinverse, Gram, polar or alternate factorization was used. A separate six-node residual-FFN graph was accepted with every-step clone replay before creating 17 physical variants. Native inference then evaluated those variants and the original policy on all 160 confirmation frames. The closed-loop pilot uses four arms and 20 paired initial states, for 80 accepted episodes after the homogeneous-hardware audit.

The two-token operator itself differs from the corresponding native 64-token-context outputs at real-panel RMSE 0.2753093, or 18.165% of native-context output RMS. Block truncation errors below compare with the two-token operator. Native two-token parity does not remove this context gap. The native FFN intervention is a separate experiment that retains all 64 attention tokens.

**1. Spectral allocation repairs the small-removal result.** W is the original width-based allocation. S allocates removals using the explicit occurrence-environment spectral tails. Both retain the leading ODT directions at the resulting per-bond ranks. The original dimension denominator remains 9,274.

| Approximate bond-dimension removal | Exact removed dimensions | W real RMSE | S real RMSE | W Gaussian RMSE | S Gaussian RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1% | 93 | 0.042030 | **0.000159** | 0.058148 | **0.000164** |
| 5% | 464 | 0.086894 | **0.031470** | 0.106534 | **0.030965** |
| 10% | 927 | 0.109973 | **0.096475** | 0.142223 | **0.127134** |
| 30% | 2,782 | **0.374901** | 0.569509 | 0.467858 | **0.333199** |

At 1%, S has **265.17 times lower real RMSE** than W. Its error is 0.0115% of the source two-token output RMS. The paired real-input MSE difference, S minus W, is approximately −0.00176647, with a task-stratified episode-bootstrap 95% interval of [−0.00178370, −0.00175067]. These are pointwise descriptive intervals, without multiple-comparison adjustment.

The 1% spectral allocation removes 93 directions from eight wide bonds. It cuts no moment/Padé or attention-scalar bond. At 30%, S cuts only wide bonds yet has worse real decoded error than W. Protecting normalization under W improves its 30% real RMSE to 0.327004. Therefore the allocation fix is useful at small cuts, and protection alone is not a general solution. Minimizing the independent occurrence-cut coefficient objective does not minimize simultaneous decoded error.

A separately frozen supplementary small-cut extension reused the exact primary inputs, baseline and bases in a separate protocol. It is not an independent replication:

| Removed dimensions | W real RMSE | S real RMSE |
| ---: | ---: | ---: |
| 1 | 0.010050 | 8.68e−8 |
| 9 | 0.018190 | 4.60e−6 |
| 46 | 0.029897 | 5.36e−5 |

All 25 primary conditions and seven supplementary conditions have valid outputs on every frozen input. Independent analysis rehashed artifacts, recomputed errors from saved predictions and checked the published metric receipts. [Primary results and paired intervals](../../output/odt_campaign_20260907/main/summary.md), [supplementary results](../../output/odt_campaign_20260907/fine/summary.md).

**2. Stronger matched controls support an ordering effect.** The controls retain exactly the same ranks and preserve the graph's zero-input output through direct-QR anchored bases. Three fixed basis seeds are evaluated at 10% and 30%, including normalization-protected allocations. These seeds describe retained-basis variation, not policy-training-seed uncertainty.

At 10% under W, leading real RMSE is 0.109973 versus 0.952759–1.064549 for the anchored controls. Under S, it is 0.096475 versus 1.038374–1.813259. At protected 30%, leading W gives 0.327004 versus 1.24476–1.48559. The ordering advantage survives controls that preserve a common zero-input behavior, rather than depending solely on arbitrary destruction of constant structure. It still does not identify semantic features or prove optimal simultaneous truncation.

**3. A corrected local ODT now reaches native policy actions.** The new graph represents the complete residual FFN function `F(r) = r + ffn_gain * FFN(Padé(r))`, where `r` is the native post-attention residual. Original 64-token attention remains in place. The same transformed F replaces the first vision block's FFN computation at all 64 tokens, exactly once. Its environment is local to that FFN, not the downstream full policy.

The graph has six shared nodes, 21 explicit clones and 390 eligible internal dimensions. The trained producer passed direct-QR, explicit-environment, every-step clone, gauge and actual-cutoff checks. Gauged source replay error is 1.74e−14. Full-rank replacement agrees with native denormalized actions at RMSE 7.34e−8 and maximum absolute error 5.96e−7.

Only the FFN-output bond is narrowed. The other local bonds remain full. The primary action endpoint is denormalized MSE across all seven action components and eight chunk positions, compared with the full-rank replacement. It is not a success metric or a physical-unit optimality criterion.

| Retained width out of 193 | Local graph dimensions removed | Leading action RMSE | Matched anchored action RMSE, three seeds |
| ---: | ---: | ---: | --- |
| 192 | 1/390 | 0.00002435 | 0.001004–0.001300 |
| 184 | 9/390 | 0.00009184 | 0.003653–0.004426 |
| 174 | 19/390 | 0.00016337 | 0.005913–0.006681 |
| 154 | 39/390 | 0.00032687 | 0.010107–0.013384 |

Every arm completes all 160 frames, with no invalid action output or gripper-sign disagreement. All 12 paired leading-versus-control action-MSE intervals favor leading ODT on these frames. For width 174 and seed zero, the MSE difference is −3.49e−5 with 95% interval [−4.01e−5, −3.03e−5]. These paired intervals resample 80 episodes within task and are pointwise 95% intervals without multiplicity adjustment. This is strong offline action-fidelity evidence conditional on one checkpoint and this data source. [Independently validated FFN results](../../output/odt_campaign_20260907/ffn/summary.json).

Width 174 removes 9.84% of that one bond and 4.87% of eligible local graph dimensions. Neither number is a policy-parameter compression ratio. The transformed evaluator materializes dense float64 cores and is not a deployment-efficiency improvement over the original factorized FFN.

**4. Closed-loop evidence remains a pilot.** The rank, random seed, initial states, horizon and controller were fixed before rollout outcomes. All arms use the same direct-QR constrained-dynamics controller, exactly two same-factor refinements, and the unchanged 1e−12 normwise backward-error gate. No stock pseudoinverse controller is admitted.

| Arm | Successes | Success rate | Descriptive Wilson 95% interval |
| --- | ---: | ---: | --- |
| Native policy | 17/20 | 85% | 64.0–94.8% |
| Full-rank FFN replacement | 17/20 | 85% | 64.0–94.8% |
| Leading ODT, width 174 | 16/20 | 80% | 58.4–91.9% |
| Anchored random, width 174, seed zero | 19/20 | 95% | 76.4–99.1% |

Native and full-rank have identical success outcomes on all 20 paired initial states. Their trajectories are not claimed to be identical. Leading ODT loses one success relative to full-rank and gains none, giving a −5 percentage-point difference and exact two-sided McNemar p = 1.0. Random succeeds on three states where leading fails, with no reverse discordance, giving a +15-point difference and exact p = 0.25. These data do not establish superiority or noninferiority.

The exploratory task-stratified bootstrap gives a random-minus-leading interval of [+5, +25] percentage points. With only two observed initial states per task, resampling cannot generate unseen opposite-direction outcomes. That interval is fragile and does not establish random superiority. Likewise, the degenerate [0, 0] bootstrap interval for full-rank minus native is not proof of equivalence. Wilson intervals are descriptive binomial summaries of this small, fixed-task panel. All intervals and tests are unadjusted for multiple comparisons.

The unchanged independent analyzer accepted all 80 complete trajectories, zero experiment failures, matching initial-state hashes and first observations across all four arms, replayed chunk denormalization and gripper execution, and authenticated task packages and source hashes. Every admitted worker used the smoke-validated RTX 6000 runtime. The largest recorded controller normwise backward error was 1.47e−17, below the unchanged 1e−12 gate. The two failed GPU-initialization attempts produced no episode outcomes and remain in the evidence archive. [Accepted pilot results and paired data](../../output/odt_campaign_20260907/rollout/summary.json).

**Additional exploratory check, 10:19 UTC.** After the campaign, we compared the first saved action chunk on every one of the 20 common post-settling rollout observations. All four arms have identical first image, robot state and simulator initial-state hashes. Relative to full-rank actions, leading width 174 has RMSE 0.000903471 and the anchored random control has RMSE 0.053819340, a 59.57-fold difference. Leading has lower per-start squared error on 20/20 starts. The translation and rotation RMSE ratios are 48.80 and 41.87. Leading has zero gripper-sign differences across 160 chunk positions and random has one. Native versus full-rank RMSE is 1.56e−7. An independent agent rehashed all 80 trajectory files and reproduced the statistics using scalar summation. This extends the action-fidelity observation to simulator-rendered starting inputs. It does not establish unseen-training-state generalization, later on-policy fidelity or a causal explanation of the success outcomes. [Matched-start analysis](../../output/odt_campaign_20260907/rollout/entry_descriptive_v1.json).

**5. The denominator interpretation is now testable and narrower.** The saved projective chart margin is an output-amplitude statistic, `1/max(1,max_abs(decoded))`. It is not evidence of a pole. The new diagnostics preserve fixed-lift logarithmic numerator and denominator scales without exponentiating the huge common exponent.

At spectral 1%, numerator and denominator scales barely change. At spectral 30%, the real denominator changes by a median of only +0.0254 bits, with a full observed range of −0.02718 to +0.05979 bits and zero sign changes across all 320 real inputs, while decoded RMSE reaches 0.5695. This damage cannot honestly be attributed to denominator collapse. The large errors of unprotected random 30% cuts accompany enormous changes in both numerator and denominator scales, unequal rescaling and sign differences. They do not establish poles or conditioning. No input perturbation test measured a crossing.

All primary cutoff gaps pass the declared resolution criterion, including 4,779 cutoff records counted across conditions with repetition. This does not turn those records into 4,779 distinct independently clone-certified boundaries. The new main-consumer cutoffs remain explicitly outside the original per-cut clone-certification set. The new FFN producer separately checked its four actual leading cutoffs. [Fixed-lift diagnostics](../../output/odt_campaign_20260907/scale/summary.md).

**Validation and corrections.** The workflow kept numerical experiment sources and failed attempts immutable. Independent analysis reads saved arrays and recomputes endpoints, rather than accepting a success flag or summary scalar. Development and confirmation identities, exact source/weight hashes, physical-versus-masked replay, native parity, every-step clone checks and runtime guards are distinct gates.

Review or validation caught and corrected these issues before accepting the relevant final result:

- Failed FFN image batches could incorrectly label a valid neighboring frame as denominator-invalid and overcount unattempted frames. A separate v2 distinguishes denominator failures, execution failures and batch-aborted neighbors. The initial experiment never triggered that branch, and all 18 arms were repeated under v2.
- The proposed rollout recorder needed strict observation, finite scalar reward and boolean termination checks. Those checks and regressions were added before the first simulator smoke.
- A scale-analysis helper assumed input nodes occupied fixed positions. Graph metadata now identifies ingress nodes, with a regression covering the actual scalar node at index one.
- The first independent FFN analyzer hashed scalar-array headers differently from the frozen native producer. It stopped without publishing accepted statistics. A separate v2 matches the producer's contiguous representation, with a scalar regression, and retains all original artifact hashes.
- Generic GPU allocation allowed four pilot workers onto other hardware. The strict independent audit rejected the mixed-hardware dataset. Exactly those workers were selected for replacement by GPU metadata, using an explicit RTX 6000 resource constraint. Two replacement attempts then failed during initialization because the allocated GPU was unavailable. Their protocol files and logs were preserved. A bounded retry selected healthy devices only from its own Slurm allocation, before any rollout outcome. Original results were retained and the hardware gate was not relaxed.

The final rollout audit also required an explicit Bash launcher after Slurm's default `/bin/sh` rejected `pipefail` before Python started. This changed no experiment or analysis code. The final primary, supplementary, FFN-action, scale and rollout analyses all passed. All accepted measurements have zero prohibited numerical attempts. The final rollout audit completed at approximately 10:06 UTC, before the 10:33:57 UTC cutoff. Finite tests and independent reviews reduce risk, but cannot prove universal bug-freedom.

**Claim-to-evidence map.** This is the current boundary for writing about the results:

| Reviewer concern | What this campaign resolves | What remains |
| --- | --- | --- |
| Synthetic-only evaluation | Real image-derived activations, fresh Gaussian panel and native Torch parity | Real data still comes from the training-data source |
| Width allocation confound | Explicit spectral-tail allocation at identical budgets, including small cuts | No decoded-error or simultaneous-cut optimality guarantee |
| Destructive random controls | Identical ranks, zero-input-preserving QR controls and three seeds | No semantic or across-checkpoint feature claim |
| Missing policy-output bridge | Accepted local FFN inserted into the unchanged native 64-token policy | Full-policy environments and corrected whole-policy ODT remain absent |
| Missing closed-loop evidence | Paired four-arm pilot with source and full-rank controls | Pilot cannot establish noninferiority or a success advantage |
| Denominator mechanism | Saved scale ledgers, sign changes and explicit amplitude identity | No pole or conditioning certificate |
| Missing uncertainty/raw outputs | Per-input predictions, episode-cluster analysis and paired trajectories | One training seed and only two rollout initial states per task |
| Numerical and provenance bugs | Immutable snapshots, independent analysis and failed-gate corrections | Finite checks do not prove universal stability |
| Compression and interpretability claims | Precise distinction between graph dimensions and policy parameters | No practical compression, speedup or human-readable mechanism result |
| Manuscript status | The current draft includes the local results and the accepted rollout pilot | No anonymous complete checkpoint/raw-prediction release is supplied |

**Defensible manuscript claim.** “For an independently checked trained two-token vision block, spectral-tail allocation reduces real-activation truncation error substantially at small removal budgets. An independently accepted local residual-FFN decomposition can also be inserted into a native 64-token VLA policy, where leading ODT directions preserve offline action outputs better than matched zero-input-preserving random subspaces. A small paired closed-loop pilot does not establish a capability advantage. These results concern fixed local tensor representations and do not constitute corrected full-policy ODT, a general truncation guarantee, practical model compression or semantic interpretation.”

The remaining high-priority scientific issue is the relationship between local coefficient fidelity, on-policy action changes and closed-loop success. More synthetic high-removal rungs would not resolve it. A later study should freeze an on-policy state panel and evaluate a paired local rank ladder on more initial states, while keeping source, full-rank and structural controls in the same runtime.

**Reproducibility.** Experiment and analysis source packets are saved in [the local packet index](../../output/odt_campaign_20260907/source_packets/index.json). The [metadata evidence archive](../../output/odt_campaign_20260907/metadata_evidence.tar.gz) preserves receipts, launch records, failed attempts, test logs and Slurm accounting. Full physical graph arrays and rollout trajectories remain under `/work/joy/x-vla-odt-campaign-20260907-v1` on Athena. Each summary binds its prediction, graph, panel, checkpoint and source hashes. The [campaign status](../../output/odt_campaign_20260907/status.json) records accepted jobs and the final cutoff. Large raw arrays and the full checkpoint are not included in this Git snapshot. The summaries preserve their identities, but independent end-to-end rerunning requires those artifacts.


**Published entry points.** The accepted native-action runner is `research.odt_ffn_native_v2.native`. The accepted action analyzer is `scripts.odt_ffn_summary_v2`. The paired rollout analyzer is `scripts.odt_rollout_summary_v1`, applied to the corrected `rollout_rtx6000_v1` dataset recorded in the status file. The v1 native action runner and v1 action analyzer are retained for provenance and source dependencies. Their superseded measurements are not the accepted endpoints. Frozen launch scripts preserve the original deployment paths. In particular, the original rollout-summary launcher targets the rejected mixed-hardware dataset. The final explicit-Bash launch command is preserved in the metadata archive as `slurm_rollout_final_audit_bash.sh`. Use the accepted roots and versions above when reconstructing the campaign.

The [paper source](../../paper/odt.tex) and [compiled paper](../../paper/odt.pdf) accompany these records. The reference/analysis runtime was NumPy 2.2.6. Native inference used Python 3.10.19, NumPy 1.26.4 and PyTorch 2.7.1+cu126. Saved source packets and runtime receipts provide the exact source closures. All canonical numerical paths retain the direct QR/RQ invariant.
