# χ-ODT development log

This log intentionally starts with the current direct-ODT implementation. Earlier χ-VLA,
workshop, intervention, and exploratory histories were moved under `prev/` and are not part of the
current source closure.

## 2026-09-04, canonical reset

- Reset the project around exact, full-policy ODT on the unchanged capable χ-VLA.
- Made direct QR/RQ an execution invariant. SVD, pseudoinverse, least-squares, Gram, polar,
  covariance, normal-equation, and self-overlap diagnostics are prohibited in the canonical path.
- Added explicit polynomial and projective shared-DAG representations for attention, residual
  routes, rational normalization, and the action head.
- Added the decisive clone-unfolded, no-memo oracle. Every shared-DAG canonicalization step must
  replay against it and fail closed on disagreement.
- Pinned the capable checkpoint and verified exact factorized replay of the full policy descriptor.

## 2026-09-04 to 2026-09-05, direct ODT implementation

- Implemented Algorithm 1 as direct reduced RQ with factors pushed into every syntactic occurrence
  of each connected shared bond.
- Implemented Algorithms 2 and 3 using explicit downstream tensor-network contraction,
  eigendecomposition of that environment, and full-rank gauge application at every occurrence.
- Replaced invalid dense and self-overlap diagnostics with reconstruction and shared-versus-clone
  replay checks.
- Added scale ledgers, occurrence ledgers, source hashes, runtime guards, immutable progress records,
  and fail-closed attestation.
- Added bounded rank-deficient support that retains the Q produced directly by QR, including its
  orthogonal completion. Unbounded cases fail closed with their shape.
- Completed the r7 full-policy run over 3,964,463 tensor nodes. All three ODT stages completed in
  15 h 22 min. Relative source replay error was `4.805e-14`, and the maximum environment
  off-diagonal residual was `1.389e-14`.
- Kept compression claims separate from exact decomposition. The current result establishes
  scalable global decomposability, not useful compression of the capable checkpoint.

## 2026-09-05, capability and controlled truncation lanes

- Froze the Product-plus-Padé capability and exact-ODT launch path.
- Added controlled physical truncation artifacts and matched leading-versus-trailing controls.
- Added a capable linear b1c checkpoint lane to separate policy capability from ODT mechanics.
- Verified a fresh closed-loop capability campaign at 419/500 successes, or 83.8%, on checkpoint
  `b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee`.
- Preserved the earlier independently verified 426/500 result for the checkpoint used by the
  completed r7 decomposition. The two checkpoint identities must not be conflated.

## 2026-09-06, current status

- Added a mapped artifact executor and a dimension-ladder pipeline for paired physical truncation
  experiments at 0, 30, 40, 50, 60, 70, and 80 percent requested removal.
- Added pinned Modal controller and worker images, source-bundle verification, direct-only runtime
  guards, deterministic LIBERO initialization, and per-episode result publication.
- Added `scripts/odt_hill_climb.py` as the reviewer-facing entry point. It validates a result,
  reports the single hill-climb metric (`success_rate`), and compares a candidate against its paired
  baseline. Exact replay and structural invariants remain mandatory gates, not optimization metrics.
- The b1c full direct-ODT job completed all 3,964,463 local RQ steps but later exhausted a 480 GiB
  allocation. It produced no full-rank certificate or compression result. The failure does not
  invalidate the completed r7 decomposition, but b1c compression remains incomplete.
- Next experiment: run paired baseline and reduced-policy panels through the dimension curve, then
  optimize closed-loop success subject to all exactness, provenance, and direct-only gates passing.

## Reviewer contract

The PR-facing contract is deliberately small:

1. Run the direct-only unit and source-closure checks.
2. Run a pinned paired closed-loop evaluation.
3. Feed each resulting `result.json` to `scripts/odt_hill_climb.py score`.
4. Use `scripts/odt_hill_climb.py compare` to report candidate success, baseline success, and the
   paired delta.

The number to hill climb is closed-loop success rate. A higher number is accepted only when the
candidate result is complete, internally consistent, checkpoint-bound, source-bound, and all
direct-only numerical guards remain clean.

## September 6, 2026, 06:15 UTC: physical six-rung curve launched

Latest user request explicitly authorizes concurrent Modal and Athena work for
30/40/50/60/70/80 percent dimensions removed versus real LIBERO success, aiming
for a first curve by approximately 17:50 UTC. This is new action authority,
not the preceding read-only heartbeat. Full protocol and exact source identities
are in `athena/ODT_DIMENSION_CURVE_V1_PROTOCOL.md`, local planning material.

Three agents built and cross-reviewed the physical ladder, mapped executor and
Modal rollout path. The review found a genuine legacy export bug: a truncated
network correctly clears full-rank flags, but the old serializer rejects those
flags. The new distinct prefix schema fixes that without re-certification or
changing the canonical serializer. The ladder uses one graph, exact aggregate
dimension budgets, every producer/consumer occurrence, and immediate per-rung
publication. The denominator is fixed original nonleaf/nonroot bond dimensions.

Combined test835571 passed96 tests with one Python-version skip, including all
21 ladder tests and all-six physical/mapped-versus-original-mask comparisons.
All runtime guards remained intact, zero forbidden attempts. Independent review
approved launch. Benchmark835572 supported eight threads over sixteen on direct
RQ and genuine contracted-environment/EVD motifs, not a guaranteed full-job ETA.

Full export job835574 is RUNNING, eight CPUs and545GiB on c2-g8-07, started around
06:10 UTC. Source root /work/joy/x-vla-odt-dimension-curve-v2 has ledger
b778a620444f72258b1170f6ee34d9e46771de2e54f3f693c83904e16a9dd666.
It imports the unchanged frozen b1c canonical source/checkpoint. Launch receipt
is published under the frozen base's athena/results/odt_dimension_curve_v2,
stderr empty. No full-rank or prefix artifact exists from this job yet.

Modal admitted336GiB plusL4, insufficient for full ODT but usable for reduced
execution. The initial paired panel is20 episodes at every point. The stock
simulator controller used pinv, so an isolated direct-QR constrained-dynamics
adapter was tested. Initial tests and real smoke passed, but baseline20 app
ap-IruXa7zkS0zqxH0EWqaVpS subsequently failed closed at controller residual
1.0665e-9 against1e-9. That baseline is incomplete. Same-factor fixed-count
iterative refinement is being tested, with no numerical fallback. All points
must share the final controller, and historical outcomes remain a separate
protocol. Synthetic mapped dispatch extrapolates to about5.35h per full panel
before real matrix costs, motivating independently validated native dispatch.

Preserved advanced835546 for full-rank/action-replay evidence. Its old physical
export is known-broken. Canceled duplicate835565 and835566–835567, obsolete
rollouts835547–835553, and superseded A30 baseline835409 and835411–835415.
No results or frozen sources were removed. A direct-upload cloud broker and
Athena resident watcher are being built to avoid dependence on an awake laptop.
No compressed success, no-accuracy-drop result, or guaranteed12h completion is
claimed.

## September 6, 06:35 UTC: compiled evaluator passed, controller and transfer gates remain

835578 completed the isolated compiled mapped-executor test lane:5 selected
tests passed,39 deselected,zero prohibited numerical attempts. All-six-prefix
oracle comparison, repeated-edge/lifecycle cases, and385x386 retained-Q are
included.10k-node batch20 old1.803s versus compiled0.03335s (54.1x). Wide-core
batch20 compiled0.001780 versus old0.001672s, so no wide-kernel speedup claim.
Maximum projective difference2.11e-15. Source59ca78a035a88731fcc4bd0dc79c50bec1b3f3d248738c095570fe5ecbf7e2e5.
Full-policy real-rung replay is still a required gate, and no backend switching
based on rollout success is permitted. Producer835574 remains running.

baseline20_v2 ap-13p6gr97ZiTnecg1a0lQcM failed closed at RHSrelative residual
2.3853e-9 despite two unconditional same-factor refinements. No completed
baseline success observation exists. Isolated actual-state capture
ap-nhe6Cte8T3ttij4md5y6pH compares the failure against80-digit direct Householder
QR and retains the failing gate. A different residual criterion is not yet
approved or deployed.

Transport13 local adversarial tests pass. Exact tiny-prefix plus fresh random
one-GiB fixtures are ready. Broker deployment was explicitly denied by the
permission checker as new private source egress. Asked the user for specific
approval using the asynchronous approval UI. Do not bypass or retry the denied
deployment without authorization. No broker deployment or resident watcher is
claimed yet. Tokens are outsideGit and private; no account token/SSHkey egress.

## September 6, 06:53 UTC: final worker regression gate and remaining deployment hold

Canonical producer835574 remains running42minutes, about124.6GiB sampledRSS,
stderr empty, no full-rank/prefix output yet. Advanced700GiB835546 remains
running as a canonical-phase memory canary as well as exactness evidence.
545GiB is not certified sufficient: prior480GiB failed afterAlgorithm1, and
the newdriver releases extras only afterfullcert. Do notdiscardthecanary's
headstart withoutnewmemory evidence. Its legacyphysicalexport remainsbroken.

Transport packet nowpasses16stdlibtests. Sourceledger
athena/odt_transport_v1_stage.sha256 is
5521c419413d2a6dbc3faa4c5b29b5ff87f4ec90d2c10dc4ee7e40efa1ba8ecc.
Runbook athena/ODT_TRANSPORT_V1_RUNBOOK.md recordsexactpostapprovalsteps.
835579 validatedoriginaltinyfixtureaterror0.0, notcloudpublication. Broker
deployment remainsDENIEDpendingexplicitusersourceuploadapproval. No bypass.

Six336GiB/16CPU/L4 reservationswereadmitted intwogroupsofthree. Three-way
overlapwasmeasured, notsix-way. Alltestreservationswereterminated.

Thefirstcapturedcontrollercase showedcorrectlyroundedfloat64failsRHSscaling.
Thesecondcase showedcomponentwise0.623 fromresidual5.85e-38 andscale9.39e-38,
despite normwise3.91e-18 andDecimal80forwardrelativeerror2.76e-17. Secondcapture
SHA1d00f4e60c8713d0dcf5edefa18c330c36e66cf59cf49faa2483adc30c85b5a1.
ApprovedstandardnormwiseQRbackwardgate1e-12, botholdmetricsretainedaswarnings,
nochangetosolve or2fixedrefinements. Source_v6bundle
9005fe5cd4a93cd0b459093b42ff1cf20a1b29ac8a02d39d7fa76921f7b921c1,
controller e89c2b9efe953086859a5c9f37fcbca0fef1fcf9dd9f8c301fe94ce61aaf2687.
Finalcontroller12tests ap-zKo0AlmhTMLSuYsIBNdmrv andworker20tests
ap-NrnWuBl8AR8nZQtclzgN2j pass. FP64real-smoke ap-SrYLNvSUkcF3FvdtuAq2c0
is pendingcompletion. Earlier source_v5 smoke failed andwasnotdeployed.
Nativeper-rungoraclecoldreviewGO aftergrippersign,power-of-twochart,manifest
binding,andfailed-bindcleanup regressions. Allrungscapabilityresultsremainpending.

FinalFP64real-smoke subsequentlyPASSED in16.00s,825directQRsystems,
maxnormwisebackwarderror1.11e-17,zero prohibitedcalls. Both20-episode source
panelsareactuallyRUNNINGconcurrently, notjustconfigured:
primaryfloat64 baseline20_v3_fp64, app ap-hkkweC994hGA9mX9I7l7FA,
andfloat32precisionbridge baseline20_v3_fp32, app ap-N6iinJA6OXqPwDbv9Jw9Dw.
Bothshare source_v6/controllerandtheexactsamepreprocessingandpairedinitial
states. No baseline result orreducedclosed-loopresultiscompleteyet.

Scientific v2worker isdeployedas xvla-odt-dimension-curve-v2. Pinnedfunctions:
baseline fu-bqdBtNCKYcaGErOiiJkHkB, reduced fu-Xo1JVl6eHY2hQPfjtb5Eoe.
Finalrunner695bcf87a5c92340fb52470d17045b13b803003663bae0ab16c07d9d64d4178d.
The localpublicbrokerconfiguration nowpins thesefinalsource/function/controller
identities andbranch-specificFP64/native-oraclerequirements. Thiswasalocal
configurationeditonly. ThedeniedbrokerhasNOTbeendeployed, nooutboundupload
wasperformed,andtheprivatebearerconfigurationwasnottouched.

## September 6, 07:15 UTC: paired baselines complete, Git checkpoint and revised ETA

Read the actual Modal result.json files. Primary baseline20_v3_fp64 completed
14/20 in 364.429 seconds. The separate baseline20_v3_fp32 precision bridge
completed 14/20 in 361.256 seconds, with identical paired success indicators.
Both match the final source_v6/runner/protocol pins and report zero prohibited
calls. These are source-policy observations, not physically truncated outcomes.
No six-rung reduced-policy success result exists yet.

835574 remains running, with no physical prefix yet. The advanced 835546 canary
reached 3,100,000/3,964,463 Algorithm 1 steps at 07:00 UTC, reporting 14,861
seconds in that sweep. Do not mistake its head start for producer835574 progress.
The older completed r7 artifact records 26,397 seconds for canonicalization and
24,269 seconds for Algorithms 2/3 (about 6.74 hours), plus compilation/replays.
Planning estimate at 07:15 UTC is therefore roughly 14–20 more hours for the
complete six-point twenty-pair pilot, with low confidence until a real physical
rung is timed. This assumes prompt transport approval and no memory/oracle
failure. The original 17:50 UTC target is at risk, not promised. Broker deployment
is still denied pending explicit approval. No bypass or new deployment occurred.

User explicitly requested periodic GitHub main checkpoints. Pushed verified
source/test milestone ae566bd2fde37a3590b0993f7cbf3d01f2a239e8 to public origin
J0YY/x-vla main and verified the remote hash. Thirteen explicitly selected files,
no DEVLOG, private configs, raw artifacts, transport prototype, or unrelated
moves/deletions. Fresh syntax/direct-only source-closure checks and tested-byte
hashes passed. Prior release gates were 12 controller, 20 worker/oracle, and five
native executor cases plus real FP64 smoke. Updated the existing 30-minute
heartbeat to commit and non-force push meaningful verified ODT milestones only,
preserving unrelated/pre-staged work and stopping on upstream divergence. The
GitHub authorization does not authorize the separately denied Modal deployment.

## Corrected deadline block curve launched, 2026-09-06 21:22 UTC

Latest user explicitly approved the corrected curve launch using Athena or at
most $45 remaining Modal credits. Canceled only the verified superseded Athena
jobs 835546 and 835574. Their source packets, logs and artifacts are preserved.
No Modal jobs, broker deployment or cross-provider transfer was performed.

Reviewed and pushed two task-owned commits to public origin/main: 1113b88 and
42ca0e0. The latter is the frozen launch source. Independent review round 2 was
clean after replacing a retained-space solve diagnostic with the independent
environment eigen-equation residual/separation check. Mathematical environment
contractions were reassociated into two explicit binary contractions, independently
in the shared and clone implementations. No change of factorization or objective.
All 58 guarded tests passed locally and on Athena, with zero prohibited attempts.
Athena tests took 54.731 seconds, 16,224 direct QR calls and 492 environment EVDs.

Fresh isolated stage: /work/joy/x-vla-odt-deadline-curve-v3. Frozen archive SHA256:
a8ffd958faa51b35d0668a10f6e2b7c7c5ad44f55c60b48cd75f288e69e11f45.
Source manifest sources.sha256 covers every executed reference/test source and
the Slurm script. Runtime /work/joy/safesae/bin/python is Python 3.10.19 with
NumPy 2.2.6. Checkpoint remains the immutable b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee
file under /work/joy/x-vla-capable-linear-b1c0-odt-v2/inputs.

Producer 835665 is running on c2-g8-05, 16 CPUs, 256 GiB, four-hour wall limit.
Array 835666 contains six afterok-dependent workers for 30/40/50/60/70/80 percent
post-QR unique nonleaf/nonroot bond directions removed, capped at three workers
concurrently. Each worker requests 8 CPUs, 64 GiB and two hours. Producer supplies
the 0-percent baseline. The 50/70 workers also run matched trailing-direction and
three frozen direct-QR random-basis controls. All saved physical graphs are reloaded
and compared with independently executed gauge-space masks.

Scope is unchanged trained width192/head8/rank576 TWO-TOKEN vision block with all
six Padé norms, biases, residuals and attention. This is NOT a full64token policy,
NOT LIBERO accuracy, and NOT storage reduction. The 64 frozen synthetic input
panel passed independently reconstructed raw-weight export replay at 3.729988859457336e-15
global-max-scaled error. Explicit clone count is 81,377. Export replay is not
completed ODT. Every-origin clone QR/gauge equality, environment and retained-space
checks, finite-error thresholds, and the accepted.json receipt remain required.
Progress is results/progress.json, logs/producer-835665.log. Accepted canonical
source is results/canonical and each worker writes results/p{percent}_{mode}_{seed}.
No accepted trained decomposition or curve result existed at this launch entry.

Reactivated the existing 30-minute watch-canonical-odt-runs heartbeat with the new
job IDs, source pins, scope and $45 budget. It must stay quiet on unchanged healthy
state, report meaningful results/failures/deadline changes, never revive the stopped
legacy jobs, and never bypass the separately denied Modal broker deployment.
Submission planning window was 5-8 hours from about20:55 UTC. A wall-time limit is
not a completion ETA. Derive timings from actual phase progress, retain 45 minutes
for reporting, and do not turn synthetic block errors into robot-success claims.

## Deadline performance repair and v4 replacement, 2026-09-06 22:09 UTC

At the 21:56 UTC heartbeat, v3 producer835665 had run36minutes and completed
only26/453 QR origins. The checked shared/clone cores matched exactly, local QR
reconstruction was at most8.820354658116359e-16 scaled error, and RSS was about
26GiB. No trained ODT acceptance receipt existed and the curve was still gated.
The submission window was at risk from runtime, not a discovered numerical failure.

A guarded weight/topology inspection identified277,900,086 literal Python
symmetric-coordinate iterations across the clone tree. The two attention-input
moment nodes alone are2x193x193 with7392occurrences EACH. Replaced only the clone
oracle's independent pack/unpack coordinate loops by row-major np.nonzero indexing.
Preserved a/2+b/2, sqrt(2) weights, every independent occurrence's direct QR, both
unpacked symmetric entries and every consumer absorption. No factors are reused.
Added an exact-equality fast branch to close(): all entries must compare equal
AND finite before returning zero, otherwise the original scaled comparison runs.
All full clone scans, every-step comparisons, environments and thresholds remain.

Independent agent checked35fixtures including actual2x193x193, zero/singular/
rectangular completions and1e+/-200 magnitudes. Packed arrays,Q and triangular
factors were bitwise identical,70independent QR calls,zero prohibited calls.
Three local repeats of actual-shape pack+QR+unpack took0.30360seconds for literal
loops versus0.006717seconds for indexed operations,45.2x ONLY for that operation.
This is not an Athena or whole-producer speedup/ETA. A separate finite equality
microbenchmark was2.4-3.3x faster locally. BLAS threading remains unchanged.

Added literal-original-loop regression tests for complete Q/parent/head equality,
every-occurrence QR counts, zero/deficient/rectangular/large cases, nonsymmetry,
nonfinite identical arrays and preserved global comparison scaling. An initial
test-counter assertion read the imported runner counter rather than __main__'s
installed guard, failed honestly, and was corrected with a local observer that
still calls the installed QR guard. Final61tests passed locally in41.264seconds
and passed again on Athena,16,272 QR,492 EVD,zero prohibited attempts. Independent
read-only review was clean. Source commits e1a5a57 and041d41d are pushed to main.
Frozen final source041d41d65bd4d6ccd317c15b823c738f310ca320.

NEW immutable stage /work/joy/x-vla-odt-deadline-curve-v4, archive SHA256
cf55a8ad15eb17d6c16a95a49e3fba793059ca6b0b0f66c73ef9abf3b38ea09a,
sources.sha256 verified before launch. Script requires explicit ODT_CAMPAIGN_ROOT.
Producer835669 runs16CPUs/256GiB on c2-g4-17, four-hour limit. Array835670 tasks0..5
retain the same six percentages, afterok receipt gate, maximum three8CPU/64GiB
workers and same leading/trailing/random controls. Runtime and checkpoint are
unchanged. The independent64input export replay again measured3.729988859457336e-15.
No decomposition acceptance or physical curve result exists yet.

After the v4 remote61-test pass and export replay, canceled superseded835665 and
835666, preserving all source packets/logs/artifacts. Updated existing30minute
heartbeat to v4 job IDs and source pins. No Modal spending or broker deployment.
Deadline and scientific scope remain unchanged: trained TWO-TOKEN block output
error on synthetic inputs, not full-policy LIBERO success. Do not promise the
submission curve until actual full-phase timings and acceptance support it.

First matched Athena timing check: v3 reached8origins in1086.14113726496s of
sweep time, v4 in143.9142514653504s, about7.55x faster for this prefix on different
Athena nodes. Both recorded33,481 independent QR calls, exactly zero shared/clone
comparison error, and the same7.320516381796115e-16 maximum local reconstruction
error. v4 moment origin2 finished by80.564s versus938.896s in v3. Do not extrapolate
this prefix speedup to all QR origins, the environments, gauges or curve workers.
v4 RSS was27,352,812KiB. Slurm confirms835665 and all835666tasks canceled, with
logs intact. Athena lacks rg, so use grep for subsequent remote text checks.

## v4 halfway QR timing check, 2026-09-06 22:42 UTC

Producer835669 remains healthy at36:08 Slurm elapsed, with230/453 QR origins
completed in2064.610103256069seconds of sweep time. It has made75,349 direct QR
calls, zero prohibited attempts, exactly zero shared/clone core discrepancy,
and maximum local reconstruction error1.190508995754726e-15. RSS27,360,236KiB.
The latest ordinary origins take approximately8.2-8.5seconds each. A pace-only
estimate is about30-40minutes MORE for Algorithm1, subject to remaining large
cores. Environment/gauge timings remain unmeasured, so this is not a full-curve
ETA. There is no accepted.json and array835670 remains correctly dependency-gated.
No new launches, source edits, threshold changes, Modal spend or commits were
needed for this healthy progress check. All six percentages remain queued.

## Confirmed scale underflow, exact-ledger repair and v5, 2026-09-06 23:38 UTC

v4 producer835669 failed at01:06:55 elapsed AFTER all453QR origins. Final
81,830direct QR calls, zero prohibited attempts, exactly zero shared/clone core
discrepancy and1.2181952514699522e-15 maximum local reconstruction error. It then
rejected the full-rank chart before any environments/gauges. Array835670 canceled
automatically. No accepted.json or physical curve result was produced.

Two independent agents found no masking/source-order/rescaling semantic error.
Guarded immutable-weight shared-only reproduction confirmed actual cause:
all64original inputs valid, minimum denominator margin.3359675287509632 and
maximum expected output2.976478124888233. All64canonical rows were exactly zero,
head maximum/nonzero count bothzero. First zero pre-QR core297 rbn_ffn:moment,
after residual1core maximum2.6730555465649116e-177. The same collapse recurs in
the secondtoken. This is accumulated scalar underflow, not ODT impossibility.
Independent clone agreement had reproduced the same underflow, illustrating why
that check alone cannot establish finite-arithmetic preservation.

Added compact exact global scale ledger outside the unmodified QR primitive.
For fixed ordered lift, unique core v occurs m_v times. Replacing C_v by2^-s C_v
and G by G+m_v*s preserves T_original=2^G*T_stored. Do this immediately before
each direct QR. Every triangular factor remains absorbed into every occurrence.
Clone independently balances each physical occurrence using its own frexp/ldexp
calculation and sums eachshift, never sharedmultiplicities or reusedfactors.
Integer ledgers and allcores/head compare everyorigin, plus headbalance once.
Every scaling requires exactfinite elementwise power2 roundtrip, so lost small
entries failclosed. Q cores remainactualQ. All original environments share the
same2^(2G) factor, preserving the individual and summed ranking. G and2G are
authenticated in canonical/physical graph metadata and acceptance receipts.
This does not admit arbitrary within-core dynamic range, cancellation, or finite
threshold relaxation. Oldzero-headtensor is not reused, restart from rawweights.

Independent repeated/diamond/singular/zero tests compare COMPLETE coefficients
after everyorigin, including the ledger. Wrongmultiplicity is a negative control.
Both positive/negative1700 shifts give the exactpredicted ledger. A1e-100 identity
chain retains validreplay and nonzeroenvironments. Roundtrip tests reject lost
coefficients. All66guarded tests passed locally in48.402s,16,360QR,495EVD,zero
prohibited calls, and passed again onAthena. Independent read-only review clean.
Source+derivation commit32fb1e7b74c10b807de2696840473622b6e44a89 pushed to main.
Newfile scaled.py is79lines, keeps bookkeeping outside the direct-QR core.

Cheap shared-only trained preflight now runs all QR/environment/EVD/gauge/replay/
offdiagonal gates BEFORE the expensive independentclone run, and is explicitly
NOT acceptance. Local result: G=-12910, canonical1.3321596930475942e-14,
gauged2.47671260267967e-14, offdiag2.2509717349885758e-15, all64valid.
Athena v5 preflight: canonical5.846757537199374e-15, gauged2.3349730260202924e-14,
offdiag1.5315165481036623e-15, sameG=-12910,54.04seconds includingcompile/replays.

NEW isolated stage /work/joy/x-vla-odt-deadline-curve-v5. ArchiveSHA256
f3b5167daf4ec95d130574951c221d754537379fba31b6fd9b9241fee461019e.
Producer835679 runs16CPU/256GiB on c2-g8-05, capped2h45. Array835680 has allsix
unchanged percentages, max3 concurrent8CPU/64GiB workers,40minute caps. Same0%
baseline,50/70trailing and3randomcontrols, sameinputs andrankbudget. All remain
dependency/receipt gated. Remote66-test suite and trainedsharedpreflight passed,
and fresh independentclone QR is running. No accepteddecomposition orcurve yet.

No Modal spend: user's$45 untouched. Updated existing30minute heartbeat to v5.
Five-hour target is nowat risk. The extendedeight-hour window endsabout04:55UTC
Sep7, preserve45minutes forreporting and freezeabout04:10. PreviousQR took65.2min,
but scaledclone/environment/gauge durationsneednewmeasurements. No wholecurve ETA
is promised. Scope remains trained TWO-TOKEN block error, NOT LIBERO/fullpolicy.

## v5 healthy midpoint check, 2026-09-07 00:14 UTC

Producer835679 running at39:24 Slurm elapsed,253/453 independent QR origins
complete after2231.03s sweep time. Shared/clone core comparison remains exactly
zero, maximum local reconstruction error1.2181952514699522e-15,76,573QR calls
including the shared-only preflight,453preflight EVD calls,zero prohibited
attempts. Current compared global exponent is-10582, not yet its final value.
RSS27,455,768KiB. Latest ordinary origins takeabout8.2s each. Pace-only QR
remaining estimateabout28-35minutes, not a full acceptance or curve ETA.
Environment/gauge acceptance timing remains unmeasured. No accepted.json;
835680's six tasks remain correctly dependency-gated. No source/job changes,
additional spending, new launches or commits required for this healthy check.

## v5 QR and independent environments passed, 2026-09-07 00:46 UTC

All453QR origins completed in3877.343952s of clone sweep time (64.62minutes).
Every-origin shared/clone core error exactlyzero, final global exponent-12910,
maximum local QR reconstruction1.871078855804712e-15. Canonical full-rank replay
passed5.846757537199374e-15 on the64frozen synthetic inputs. The former all-zero
head/chart failure did not recur.

Independent occurrence-environment contraction, aggregation, EVD and the
nondegenerate retained-space checks completed before the first gauge comparison.
Maximum shared/clone environment difference2.074384070927484e-14. Environment
phase entry to first gauge receipt94.74s, including retained-space checks and
the first gauge, not a pure contraction benchmark. Resolved/degenerate cutoff
counts will only be available in the final acceptance receipt.

At latestcheck,19/453 every-origin gauges compare exactlyzero.82,283QR and
1,359EVD calls including shared preflight,zero prohibited attempts. Slurm
elapsed1:10:36,RSS54,259,764KiB. Ordinary gauge origins currently7.4-9s each,
so about55-70minutes of gauge checking may remain, subject to later shapes,
then final replay/offdiagonal gates and physical workers. No accepted.json
and array835680 remains dependency-gated. This milestone supports the bounded
trained two-token block, not full-policy ODT or LIBERO. No source changes,
new launches, threshold changes or Modal spending. Earlier five-hour target
remains at risk, extended04:10UTC reporting freeze unchanged.

## v5 healthy gauge midpoint, 2026-09-07 01:17 UTC

Producer835679 running at1:42:04 Slurm elapsed.252/453 gauge-origin comparisons
complete, every gauge and QR shared/clone core error still exactlyzero. Final
global exponent remains-12910, environment error2.074384070927484e-14,
canonical replay5.846757537199374e-15.82,283QR and1,359EVD calls including
preflight,zero prohibited attempts,RSS54,259,764KiB. Recent steps7.5-9.2s,
so approximately27-35minutes of gauge checking remain at observed pace, then
final replay/offdiagonal/serialization gates and dependent physical workers.
No accepted.json or rung results. Array835680 remains correctly dependency
gated. No failure, new launch, source change, threshold change, Modal spending
or commit. Continue existing heartbeat, no unchanged-state notification.

## v5 complete, curve audited, 2026-09-07 01:58 UTC

Producer835679 COMPLETED in02:09:28, exit0. Receipt accepted.json exists and
SHA256bc961a3d700e94a15341aa2700eb85ca638cda00d80a2c0e7f6d803b14753a3d.
All453QR and453gauge comparisons exactlyzero. Canonicalreplay5.846757537199374e-15,
gaugedreplay2.409572803209439e-14,environment2.074384070927484e-14,
localQR1.871078855804712e-15,offdiagonal1.5315165481036623e-15.
1540resolvedsubspacechecks,0degeneratecutoffs.82,283QR/1,359EVD includingpreflight,
0prohibitedcalls. G=-12910,2G=-25820,81,377clones,453nodes,450eligiblebonds,
raw/postQRboth9274dimensions. Source remains immutable32fb1e7.

Allsix835680arraytasks COMPLETED exit0,14-80sperworker. All14savedvariants
leading30/40/50/60/70/80, trailing50/70 andrandomseeds0/1/2at50/70 arepresent.
Everyvariant64/64valid, worstphysicalmaskerror4.599098879509711e-13.
Leading keptcounts6492,5564,4637,3710,2782,1855. CorrespondingRMSE
.45667105737445346,.46600741937179246,.48048035006082634,
.49768610002109576,.5165410719581165,.5627084694582132.
Do NOT call this near-lossless: originaloutputRMS/zeropredictor=.6148044305029251,
identitypredictionRMSE=.5818154894832621,in-panelmeanRMSE=.5988388523335124.
These posthocscalediagnostics usedguardedNumPy with0QR/0EVD/0prohibitedcalls.
NormalizedleadingRMSE74.28-91.53%. Leadingbeatscontrols, but random/trailing
near-polemargins accountformuchof theirhugeerrors. Allmetricsreportedscopebound.

DownloadedJSON/panelto /private/tmp/odt-v5-results.4RNcHR. Independently checked
allsourcehashes, and30actualgraph/arraybytehashes across15artifacts onAthena,
allmatch. Twoindependentagentscheckedallreceiptlinkages, exactFractionrankbudget,
nesting/equalcontrols,820consumeroccurrences pergraph,exponents,metricstables,
scopeandinterpretation. No blockingfinding. Incorporatedscientificwordingfix:
futureequalbudgetnormprotectedablationcouldtestcontributionofearlycuts,not
uniquelyseparateallocationfromODTordering. Current30%alreadyreducesall232width2
bonds,including136moment/Padébonds,torank1. Thisisdeclaredallocation,notabug.

CuratedresultandREADME committed as7d5085e, onlytwofiles. Fullsummaryis
research/odt_reference/TRAINED_BLOCK_CURVE.md. No math/source/thresholdchanges,
rawresults/checkpoints/planningdocs notcommitted. Existinguntrackedpaper/odt.tex
leftuntouched, its historicalfullpolicyclaims stillneedreconciliation. No
remainingcampaignjobs ornewlaunches. Modalspending$0,$45creditspreserved.
Requestedcurvecomplete, notfullpolicytruncatability. Pausecompleted-jobmonitor.

## Authorized rank-allocation rerun staged, 2026-09-07 04:15 UTC

User explicitly requested rerun and fanned audits after discussing narrow-bond
collapse. Currentclock was04:02UTC at taskstart, not the earlier02:00status.
Use shortexisting-artifact evaluations, no repeated2hcloneproducer and noModal.
Independent strategy audits confirmed width-based event filtering preserves
the9274denominator, exactglobalintegerbudgets and nestedranks. Half-even25%
budget is2318, yielding170narrowcuts, not171. Scalar projectivevariation can
be lost at rank1, but wholeblockneednotbeconstant and widercutsremainconfounders.

Added onlyscripts/odt_rank_ablation.py, test_odt_rank_ablation.py and
slurm_odt_rank_ablation.sh. Allresearch/odt_reference executed sources remain
byte-identical to acceptedv5. Newconsumerpins exactacceptedreceiptSHA, checks
oldsourceclosure/checkpoint/panel/canonicalhashes/NumPy/globalexponent, records
separateconsumerhash. Additionalcutoffs getexplicitcontractedenvironment
eigen-equationchecks andunresolvedgapflags, NOT retroactiveclonecertification.
Physicalgraphsare saved/reloaded andcomparedagainstindependentmasks; denominator
failures/all64rowsare recorded. Fresh0%replay hasunchanged1e-10 gate.

TASKS27: indices0-9 original0/1/5/10/20/23/24/25/26/30%;10-15protectallwidth2
30/40/50/60/70/80%;16-21protectonly136moment/Padé atsamepercentages;
22-24cutonlyall232/norm136/attention96narrowbonds;25-26restoreall/normnarrow
bondsintheoriginal30%rankswithoutreallocating. Causalonly/restore variants
explicitlynotbudgetmatched, actualremovedcountisreported. Original0baseline
willgateother26tasks. Proposedresources4CPU/16GiB,fiveminutecaps,max6concurrent.

Original66tests passedagain44.38s,16,360QR/495EVD/0prohibited. New5tests pass
locallyandAthena,0QR/5EVD/0prohibited. Independently inspectedall27plansagainst
authenticatedtrainedmetadatawithoutnumericalfactorization. Committeda01248b,
thenfullrank/forgedreceiptgateeebbdf5f81f9754175b802cc9f0760dda56a1654.
Finalisolatedstage/work/joy/x-vla-odt-rank-ablation-v2, archivesha
6fd1a6a4d300c73685745ee7519c50b4505ff7a75f0233eb3b1cb22ecafd6c1c.
Olderablation-v1stagingpacket preserved, nojobslaunchedfromit.

SeparateCLIreview --base includedunrelateddirtyworktreeandflaggedexisting
paper/chi-vla.tex deletion, notourcommits. Itfoundnodefiniteablationdefect.
Do NOT restore/touchthatuser-ownedpaperchange. A secondstrictlycommittedthree-
pathread-onlyreview isrunning, output/private/tmp/odt-ablation-packet.DHtHuq/
scoped-review.txt, session01a07a11-a4be-7632-b7fc-c095c231d10e.
Nocurrentablationjobsuntilreviewandfreshbaselinegatepass. NoModalspend.

## Rank-allocation reruns complete, 2026-09-07 04:34 UTC

Scoped cumulative review found a P1: test discovery did not install guards and
the consumer did not statically audit the test file. Fixed both in ffff39b and
re-ran direct-module and unittest-discovery test invocations. Final independent
review clean, /private/tmp/odt-ablation-packet.DHtHuq/scoped-review-final.txt.
Original 66 tests and new five tests passed before launch. No production math
changes. Deleted temporary _review-loop-baseline tag after completion.

Broad push was auto-review rejected. Did not bypass. Read-only verified exact
remote and prior main commit 7d5085e, and that the cumulative three commits
modify only scripts/odt_rank_ablation.py, scripts/test_odt_rank_ablation.py and
scripts/slurm_odt_rank_ablation.sh. Explicit nonforce SHA-to-main push to
https://github.com/J0YY/x-vla.git was approved and succeeded at ffff39b272e2d91d36ff359d0532e8e423b5f02a.

Frozen launch stage /work/joy/x-vla-odt-rank-ablation-v3, packet SHA
61c18260b66a1a8a91182b1e2eb7edf3e64004d0cfac0c3259fa5d0adc612a3d.
Consumer SHA b84260e03413c5259593c0e26096ccf029d811675f3d98d21d0eb1dea122bf69,
test SHA 1fd0ba69de2a00f893fc5bfa986d9af61c5e55d96b4b7ba1fedfa9ecb236b2ba.
Older v1/v2 stage directories preserved, no jobs launched from them.

Fresh baseline job 835699 passed: scaled replay 2.3424330037392073e-14,
absolute RMSE 1.0573373916890912e-14, all 64 valid. Array 835700 then ran
26 workers, maximum six concurrent, 4 CPUs/16 GiB each, five-minute caps.
All 27 jobs now COMPLETED exit 0. Actual individual elapsed times 18–33s.
No queued/running campaign jobs remain. No Modal use, remaining $45 preserved.

All 27 results downloaded as JSON/graph metadata to
/private/tmp/odt-rank-results.5Ij6RV. Remote immutable results and arrays remain
at /work/joy/x-vla-odt-rank-ablation-v3/results/task_0 through task_26.
Read-only Athena rehash verified all 54 graph/array file hashes against receipts.
Local independent Fraction event reconstruction verified all 27 rank plans,
fixed 9274 denominator, nesting, and all 820 consumer-input occurrences per graph.
Every result 64/64 valid, no unresolved new cutoff gaps. Max mask 7.649436639667329e-14,
basis eigen-equation residual 2.048385152284135e-15. All workers 0 QR/0 EVD/
0 prohibited calls, reusing accepted full bases. New cutoffs are explicitly
NOT retroactively part of old independent clone retained-space certification.

Results: original 1/5/10/20/23/24/25/26/30 percent RMSE
.058803035,.106198757,.140580860,.243236116,.269399493,.277149564,
.383640303,.451470933,.456671057. Original 30 exactly matches v5 receipt.
Only norm: 136 cuts/1.466465%, RMSE .421296318. Only attention: 96 cuts/
1.035152%, .073429105. Only all narrow: 232 cuts/2.501617%, .421317449.
Restore norm in original 30: 28.531378% removal, .334606393. Restore all:
27.496226%, .333923123. Same-budget protect_norm30 .334608597 and protect_all30
.349396085. Protection at higher removal WORSENS: protect_norm70 1.104769545
vs original .516541072. Full tables in RANK_ALLOCATION_ABLATIONS.md.

Independent agent audited result interpretation. Supports a real damaging
normalization-cut intervention and partial allocation benefit at 30%, not a
general fix. Wide cuts already cause .2694 RMSE at 23%. Improved 30 remains
54.4% of original-output RMS. Similar norm-only/all-narrow scalar RMSE does not
prove identical outputs or zero attention contribution. Old matched controls
cannot transfer to new rank allocations. No full-policy or LIBERO evidence.
Local plan updated. Paper files untouched, historical full-policy claims still
require reconciliation. No new full sweep or speculative alternate allocator.

Final evidence review of result report and README updates found no numeric or
scientific overclaim issues. Clarified that 18–33s is Slurm elapsed, not the
receipt-internal evaluation timer. Scoped result docs committed as
ea8053ef7562b6c3f470a479b12190a9a0259220. This commit contains only the new
RANK_ALLOCATION_ABLATIONS.md and links/updates in README and TRAINED_BLOCK_CURVE.

## Reviewer audit of current ODT evidence, 2026-09-07

User requested an in-depth scientific reviewer assessment, so the review traced
the corrected reference, production v2, frozen r7 source, receipts, ablations and
manuscript rather than launching another experiment. Full report is
ODT_VLA_REVIEW_2026-09-07.md. Numerical sources, paper and historical receipts
were not edited. No remote compute, commit or push was performed by this review.

Fresh guarded checks: 66 reference tests passed in 46.065s, 16,360 direct QR,
495 permitted environment EVD, zero prohibited attempts. Five allocation tests
passed, zero QR, five environment EVD, zero prohibited attempts. Accepted v5
receipt and panel hashes, all 12 numerical source hashes, and graph JSON hashes
for all 41 original/ablation physical results match. All 41 report 64/64 valid
inputs. Large arrays are absent from these local copies, so prior remote
array-byte rehash was not repeated. Evidence and test logs are saved under
output/odt_review_2026-09-07/.

New reporting findings: the stored chart margin equals
1/max(1,max(abs(decoded_output))), exactly by definition. It cannot independently
identify a pole or establish denominator stability. Checked against original
panel outputs and full-rank receipt to 1.71e-14. Thus attributing huge control
errors to near-pole behavior needs additional evidence. Also the paper's stated
relative-L2 metric differs from frozen r7's projective residual, which decodes to
norm(ya-ye)/(norm(ya)+norm(ye)+1). The reported 4.805e-14 compares Algorithm 3
against its canonical input, not a fresh final-versus-raw-source comparison.

Scientific verdict: accepted corrected trained two-token block ODT is credible,
but corrected full-policy acceptance, useful capability retention and semantic
mechanisms remain unestablished. Old full-policy receipts certify the old lift,
and current streamed-Q production route explicitly fails closed. The sum of
independent occurrence-cut losses is not simultaneous tied truncation loss.
Normalization cuts explain a damaging intervention, not all distortion or a
general repair. Priority is correcting claim/metric labels, then held-out real
activation and matched structural-control evidence before expensive scale-up.

## Three-agent 3–4 hour experiment planning, 2026-09-07

User explicitly requested fan-out to think deeply about experiments for the
next 3–4 hours on their cluster. Three agents reviewed experimental science,
real/native activation capture and local policy insertion, and scaling/resource
feasibility. Proposed protocol is ODT_4H_EXPERIMENT_PLAN_2026-09-07.md. This was
planning and read-only feasibility inspection. No cluster jobs, training,
rollout, commit or push were launched during planning.

Read-only Athena checks found no current user jobs and confirmed accepted v5
arrays, rank-ablation results, b1c inputs, LIBERO frame cache and both native and
reference runtimes. At inspection, a compute node had 56 unallocated CPU slots
and about 220 GB scheduler-unallocated memory. A Quadro RTX 6000 node had three
unallocated GPUs and about 301 GB scheduler-unallocated memory. These are
availability snapshots, not reservations or guarantees of immediate start.

Recommendation: reuse accepted block for native/runtime parity and frozen real
activations, four width/spectral allocation x norm-protection definitions, and
three zero-input-preserving direct-QR random seeds at matched ranks. Deduplicate
to about 35 primary conditions on 320 real pairs from 80 episode clusters plus
256 new synthetic inputs. Keep training-source caveat, separate two-token
context error from truncation error, save all predictions and cluster uncertainty
by episode. Add fixed-lift numerator/denominator scale tracking if replay gates
pass. Existing 18–33s/64-input consumers suggest 30–45min primary compute with
six 4CPU/16GiB workers, reserving 60min and measuring a pilot first.

Higher-upside optional lane: new six-node, 21-clone residual-FFN ODT with ~111MiB
raw shared cores, inserted tokenwise after unchanged native 64-token attention.
Cut only the FFN-output bond (one local clone occurrence), keep norm bonds full,
predeclare widths and matched QR controls. Require fresh clone acceptance and
full-rank native action parity. Offline downstream action evidence is primary.
Only admit an 80-episode paired four-arm rollout pilot after a measured runtime
gate by T+150min. This remains local FFN ODT, not corrected full-policy ODT.

Integer topology accounting reproduced the accepted two-token graph. Current
dense joint-64-token compilation projects ~83.3m clones and ~27.5TiB raw clone
cores before working memory. This is a current-compiler limitation, not a
universal bound. A large new producer is not the recommended use of this window.
The plan freezes primary code/protocol by T+90 and reserves final aggregation
time. It includes explicit failure gates and three-hour fallback priorities.

Final independent plan review found no material topology/rank-budget errors.
Clarified invalidity as an unconditional reported outcome and decoded paired
error as conditional on joint validity, with no silent dropping/post-hoc penalty.
Added explicit same-controller gate for ALL proposed rollout arms: authenticated
direct-QR constrained dynamics, two fixed same-factor refinements and established
1e-12 normwise backward-error criterion, plus captured-case regressions and
transitive simulator guards. Native model-loader reuse alone does not cover the
stock simulator's previously observed prohibited pseudoinverse route.


## 2026-09-07 Athena ODT campaign execution, first accepted stages

User explicitly authorized executing the 3–4h plan on Athena, parallel work,
and fresh-context independent agents. Campaign began 08:39:34 UTC, deadline
12:39:34 UTC. All new agents were spawned with no inherited turn context.
Canonical direct QR/RQ invariants remain unchanged. No commits or pushes.

Immutable remote root: /work/joy/x-vla-odt-campaign-20260907-v1. New source packets
are separately hash-frozen, accepted producer/reference files are untouched.
FFN producer job835742 completed: 66 reference and4 FFN tests, every-step
clone acceptance, 34.61s production, gauged replay1.74e-14, all four actual
cutoffs independently checked, 54QR19EVD0prohibited. Receipt SHA
d95301a94a3708bdb5f77bd7c9ae2d0896b4795670f1c80c5f6cc0948e1dc790.
Native panel job835744 completed: 10 tests, all four real/native gates passed.
80 confirmation episodes across10 tasks,160 frames,320 fixed token pairs,
plus20 disjoint development episodes. These remain training-source episodes.
Native2-token max error4.69e-6. Two-token context-removal RMSE0.2753, which is
reported separately and is not an acceptance failure or an ODT truncation error.

Consumer prepare835745 accepted, fullrank replay4.60e-15. Exact rank-vector
deduplication leaves25 conditions. Pilots835748_0/1 both passed in~200s,
physical/masked equivalence <=1.08e-14, no invalid outputs/prohibited calls.
Main array835750, tasks2–24, six concurrent4CPU16G workers, submitted09:19UTC.
Independent analysis queued after completion, recomputes metrics from hashed
arrays and bootstraps task-stratified episode clusters (4000 replicates).

FFN physical-reference835746 prepared17 variants, native action835747 completed
all18 arms with160/160 valid frames. Fullrank actionRMSE7.34e-8. Leadingk174
actionRMSE0.0001634 vs anchored controls0.00591–0.00668. Preliminary only until
independent aggregation. Independent review found a failed-batch accounting bug:
v1 could label a valid neighbor denominator-invalid and overcount remaining
frames. No arm triggered this branch. New v2 corrects the status categories,
preserves token masks, and adds mixed/final-batch regressions. Full18-arm rerun
queued with reviewed v2, v1 artifacts preserved. Numerical contraction unchanged.
Optional rollout implementation is under two independent reviews and will only
run after v2 action acceptance, controller tests, and measured timing admission.


## 2026-09-07 GitHub publication snapshot

The user explicitly requested pushing completed work and updating the GitHub
DEVLOG, with no other Markdown changes. This request overrides the older
local-only DEVLOG convention for this publication. Existing history is retained.
The snapshot includes the frozen real-input campaign, local FFN producer,
native-panel capture, independent summary analyzer, their targeted tests and
Slurm entrypoints, and the two saved acceptance receipts. In-progress native
FFN action and rollout packages are outside this publication snapshot.

Pre-publication checks: 17 campaign tests, 14 independent-summary tests,
10 native-panel tests, and 4 local-FFN tests passed (45 total). These are fresh
lightweight local regressions, not a replacement for Athena's pinned-runtime
acceptance. The local FFN check used NumPy 2.4.2. Its accepted Athena producer
receipt used NumPy 2.2.6. All guarded tests recorded zero prohibited attempts.
All 37 unique source hashes named by the two saved receipts match local files.
The current consumer and analyzer also match the frozen campaign source hashes.

Accepted evidence remains scoped: direct-QR/RQ decomposition is certified for
the corrected trained two-token block and the local residual FFN. Native forward
agreement on real images is established. Two-token context removal itself has
RMSE 0.2753, so two-token compression results cannot be presented as full-context
policy retention. The real-image panel comes from the policy training cache,
with development/confirmation episode separation, not unseen training data.

The preceding execution entry is the latest locally recorded Athena status.
This publication check did not launch, stop, or inspect live cluster jobs.
Final 25-condition aggregation, corrected FFN action acceptance, closed-loop
retention, and corrected whole-policy ODT are not established by this snapshot.
The promising preliminary FFN action comparison must retain that qualification.
No claim of universal bug-freedom or human-nameable semantic features is made.
