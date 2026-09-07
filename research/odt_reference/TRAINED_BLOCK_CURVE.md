# Completed trained two-token block curve

September 7, 2026. This is a bounded trained-block result, **not full-policy
ODT, LIBERO success, robot-action accuracy, or storage compression**.

## Outcome

An unchanged-weight, two-token vision block completed independent clone-checked
shared-DAG ODT. All six requested physical truncation levels and eight matched
controls completed. Leading directions preserve valid denominator charts and
outperform the controls on the frozen synthetic panel, but output distortion
is substantial. Near-lossless or capability-preserving truncation is not shown.

The block retains trained width 192, eight attention heads, CP rank 576, biases,
attention, residuals and all six Padé normalization modules. It has 453 unique
graph nodes and 81,377 clone occurrences. The deployed policy has 64 vision
tokens, whereas this experiment has two. No retraining was performed.

## Independent acceptance

| Check | Maximum reported error |
| --- | ---: |
| Every QR-origin shared/clone core and head comparison | 0 |
| Local direct-QR reconstruction | 1.87108e-15 |
| Full-rank canonical replay | 5.84676e-15 |
| Independently contracted occurrence-environment sums | 2.07438e-14 |
| Every gauge-origin shared/clone core and head comparison | 0 |
| Full-rank gauged replay | 2.40957e-14 |
| Post-gauge environment off-diagonal check | 1.53152e-15 |

The gate also passed 1,540 resolved-cutoff retained-space checks, with zero
cutoffs reported as degenerate. Every-origin comparisons check all literal
clone cores and the head. They do not expand the trained block's enormous
complete coefficient tensor. Literal coefficient expansion remains a tiny-fixture
test. Independent export replay from the original weights measured 3.72999e-15.

The producer recorded 82,283 direct QR calls, 1,359 environment EVD calls and
zero prohibited attempts, including its shared-only preflight. The 66-test
startup suite passed separately. Its global binary exponent is -12,910 and the
environment exponent is -25,820. Recorded replay errors are global-max-scaled
errors, **not RMSE**, and must not be plotted as the RMSE baseline below.

## Requested physical curve

Inputs were frozen before truncation: 64 independent two-token arrays with
entries drawn from N(0, 0.2²), seed 20260906. These are synthetic inputs, not
held-out deployed-policy activations. Errors compare all 384 decoded block-output
coordinates against an independent original-weight forward implementation.

The x-axis counts unique nonleaf/nonroot bond dimensions. There are 450 eligible
bonds and 9,274 dimensions both before and after shape-only QR in this experiment.
The allocator uses a nested width-proportional integer budget with minimum rank
one. ODT orders directions within each bond. It does not globally optimize the
allocation of dimensions between bonds.

| Requested removal | Actual removal | Kept dimensions | Absolute output RMSE | RMSE / original output RMS | Minimum denominator margin |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30% | 29.9978% | 6,492 | 0.456671 | 74.28% | 0.814397 |
| 40% | 40.0043% | 5,564 | 0.466007 | 75.80% | 0.874262 |
| 50% | 50.0000% | 4,637 | 0.480480 | 78.15% | 0.909468 |
| 60% | 59.9957% | 3,710 | 0.497686 | 80.95% | 0.958201 |
| 70% | 70.0022% | 2,782 | 0.516541 | 84.02% | 1.000000 |
| 80% | 79.9978% | 1,855 | 0.562708 | 91.53% | 1.000000 |

Every row has 64/64 valid inputs. Each reduced graph was physically constructed,
saved, reloaded and compared with the independently applied full-width gauge
mask. Across all 14 variants the worst projective mask discrepancy was
4.59910e-13, below the unchanged 3e-10 gate. Producer widths and all 820
consumer-input occurrences in each reduced graph agree with its saved ranks.

Post-hoc scale diagnostics on the same frozen original-output panel are:

- Original output RMS, equivalently zero-output prediction RMSE: 0.614804.
- Identity/raw-input prediction RMSE: 0.581815.
- Per-coordinate in-panel mean prediction RMSE: 0.598839. This is an optimistic
  fitted-on-panel reference, not an independently evaluated baseline.

All leading rungs beat these simple references on this panel. Nevertheless,
74–92% normalized RMSE is substantial distortion. Neither near-lossless
preservation nor complete collapse follows from the absolute RMSE curve alone.

## Matched controls

Each control uses exactly the same per-bond ranks as its corresponding leading
model. Random bases use direct QR with frozen seeds 0, 1 and 2.

| Removal | Selection | Seed | Absolute RMSE | Minimum denominator margin |
| ---: | --- | ---: | ---: | ---: |
| 50% | Leading | 0 | 0.480480 | 0.909468 |
| 50% | Trailing | 0 | 567,139.630 | 1.45203e-7 |
| 50% | Random | 0 | 863.884 | 3.91807e-5 |
| 50% | Random | 1 | 41.913 | 1.46874e-3 |
| 50% | Random | 2 | 549.400 | 7.04525e-5 |
| 70% | Leading | 0 | 0.516541 | 1.000000 |
| 70% | Trailing | 0 | 35,509,287.121 | 7.89070e-10 |
| 70% | Random | 0 | 5,486.236 | 5.03513e-6 |
| 70% | Random | 1 | 62.317 | 7.10955e-4 |
| 70% | Random | 2 | 164.929 | 1.99378e-4 |

All controls also decode all 64 inputs at the fixed margin threshold 1e-12.
Passing that threshold is not evidence of robust denominator stability. Their
near-zero denominators amplify errors, so the enormous error ratios must not
be presented as proportional gains in semantic information retention.

## Interpretation and next discriminating experiment

This establishes a trained-block decomposition, physical truncation mechanics
and an empirical preference for leading ODT directions over matched controls.
It does not establish full-policy feasibility, closed-loop capability retention,
human-interpretable directions, or a useful low-error truncation regime.

Even the 30% rung reduces all 232 width-two internal bonds to rank one, including
136 named moment/Padé bonds. Thus every requested rung changes normalization
coordinates. This follows the declared allocator and is not a newly discovered
implementation bug. A separately declared smaller-removal ladder and an
equal-budget ablation protecting narrow moment/Padé bonds would test whether
those early cuts contribute to distortion. Protecting these bonds does not
preserve every normalization input or establish optimality of the ODT ordering.
Both experiments have since completed as separately labeled post-hoc
[rank-allocation ablations](RANK_ALLOCATION_ABLATIONS.md). Protecting moment/Padé
bonds improves the equal-budget 30% result, but worsens 40–80% results. The
original curve and its acceptance evidence above are unchanged.

The shared ranking objective is summed independent occurrence single-cut
coefficient loss. It does not certify simultaneous all-bond truncation or
decoded rational-output error. Historical full-policy receipts do not validate
this corrected representation or ranking.

Suggested bounded abstract sentence:

> On an unchanged-weight, two-token vision block, independently validated
> shared-DAG ODT supports physical removal of 30–80% of internal bond directions.
> Leading directions outperform matched random and trailing controls on 64
> synthetic inputs and preserve valid denominator charts, although output
> distortion remains substantial. Capability-preserving full-policy truncation
> remains unestablished.

## Reproducibility and provenance

- Immutable experiment source: `32fb1e7b74c10b807de2696840473622b6e44a89`.
- Checkpoint SHA-256: `b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee`.
- Acceptance receipt SHA-256: `bc961a3d700e94a15341aa2700eb85ca638cda00d80a2c0e7f6d803b14753a3d`.
- Frozen panel SHA-256: `7f8da7fc7fc878a4f6e4e81144c1c8696a4fa95bc1712cf21fbfe96e54e27f43`.
- Canonical graph SHA-256: `fbc3abd7d24d129afad76fe2d1edc9963f06bd4abb92821303fa8877082452e8`.
- Canonical arrays SHA-256: `0d53b2b33609d39cec2945c71c703d7787b993c1112411bdffcea14a4d21f294`.
- Runtime: Python 3.10.19, NumPy 2.2.6 on Athena. Direct QR only for factorization.
- Producer job 835679 completed in 2h09m28s. All six 835680 array tasks completed
  in 14–80 seconds each, at most three concurrently. All exited successfully.
- Campaign Modal spending: $0.

The authenticated receipts contain exact tolerances, source hashes, per-bond
ranks, all sample denominator margins and per-artifact hashes. They remain with
the immutable campaign artifacts rather than being replaced by rounded tables.
