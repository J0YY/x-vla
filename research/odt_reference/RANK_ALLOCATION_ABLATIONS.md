# Rank-allocation ablations on the trained two-token block

September 7, 2026. All 27 evaluations completed on Athena. These are post-hoc
diagnostics on the same frozen synthetic panel as [the original curve](TRAINED_BLOCK_CURVE.md),
not a new test set, full-policy decomposition, LIBERO success, or storage results.

## Answer

Normalization-bond cuts are a damaging intervention on this block. Cutting only
136 moment/Padé bonds removes 1.4665% of all eligible directions yet causes
0.421296 output RMSE. Avoiding those cuts at the **same 30% total-removal budget**
reduces RMSE from 0.456671 to 0.334609.

This is a partial allocation improvement, not a general rescue. Protection is
slightly worse at 40% and substantially worse at 50–80%. Wider-bond cuts already
cause 0.269399 RMSE at 23%, before any width-two cuts. Even improved 30% removal
has error equal to 54.4% of the original output RMS. Useful near-lossless
truncation and full-policy capability retention remain unestablished.

## What changed, and what did not

The unchanged trained block has width 192, eight attention heads, CP rank 576,
all six Padé normalization modules, attention, biases and residuals. It uses
two tokens, not the deployed policy's 64-token vision input. The accepted ODT
representation, direct-QR factors, full environment eigenbases, checkpoint,
global scale and 64 synthetic inputs are unchanged.

Only per-bond rank allocation changed. The original allocator orders removal
events by width threshold `(2*j-1)/(2*d)`, with node-index tie breaking, while
ODT determines the retained directions within each bond. This width-based
allocation is not a global spectral optimization over bonds.

Of 450 eligible bonds, 232 have width two. These comprise 136 moment/Padé bonds
and 96 attention scalar bonds named `dot1`, `dot2` and `score`. The original
allocator removes one direction from every width-two bond between 23.17% and
25.66% total removal. A rank-one projective pair has constant or undefined
decoded ratio wherever that pair is decoded. This does not imply the entire
block becomes constant.

- `protect_norm` keeps all 136 moment/Padé bonds at width two.
- `protect_all` keeps all 232 width-two bonds at width two.
- Both reallocate the exact same number of removals to unprotected bonds using
  the original event ordering. All 9,274 original eligible directions remain
  in the x-axis denominator. Protected bonds are not excluded from it.
- `only_*` cuts just the named family, leaving all other bonds full rank.
- `restore_*` starts from the original 30% ranks and restores the named family
  without reallocating cuts. These are not matched-budget comparisons.

Protection preserves these bonds' widths, not all their upstream inputs or
the original normalization function after other bonds are truncated.

## Smaller-removal and transition curve

Errors are absolute RMSE over all 384 decoded block-output coordinates on all
64 inputs. Original output RMS, equivalently zero-output prediction RMSE, is
0.614804. All models below decode all 64 inputs at the fixed margin threshold.

| Requested removal | Removed directions / 9,274 | Actual removal | Output RMSE | Width-two cuts | Moment/Padé cuts |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 0 | 0% | 1.05734e-14 | 0 | 0 |
| 1% | 93 | 1.0028% | 0.058803 | 0 | 0 |
| 5% | 464 | 5.0032% | 0.106199 | 0 | 0 |
| 10% | 927 | 9.9957% | 0.140581 | 0 | 0 |
| 20% | 1,855 | 20.0022% | 0.243236 | 0 | 0 |
| 23% | 2,133 | 22.9998% | 0.269399 | 0 | 0 |
| 24% | 2,226 | 24.0026% | 0.277150 | 78 | 54 |
| 25% | 2,318 | 24.9946% | 0.383640 | 170 | 110 |
| 26% | 2,411 | 25.9974% | 0.451471 | 232 | 136 |
| 30% | 2,782 | 29.9978% | 0.456671 | 232 | 136 |

Half-even integer rounding makes the 25% budget 2,318, not 2,319. The 30% rerun
RMSE exactly matches the original receipt. At 1%, 5% and 10%, normalized RMSE
is approximately 9.6%, 17.3% and 22.9%, respectively.

## Equal-budget protected curves

All three columns use the same removed-dimension count at each row. Original
40–80% values are from the immutable original campaign, not rerun duplicates.

| Requested removal | Actual removal | Removed directions | Original allocation RMSE | Protect moment/Padé RMSE | Protect all width-two RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30% | 29.9978% | 2,782 | 0.456671 | 0.334609 | 0.349396 |
| 40% | 40.0043% | 3,710 | 0.466007 | 0.472710 | 0.474569 |
| 50% | 50.0000% | 4,637 | 0.480480 | 0.612691 | 0.653552 |
| 60% | 59.9957% | 5,564 | 0.497686 | 0.836543 | 0.834494 |
| 70% | 70.0022% | 6,492 | 0.516541 | 1.104770 | 1.106210 |
| 80% | 79.9978% | 7,419 | 0.562708 | 1.211833 | 1.131807 |

The 30% normalization-protected RMSE is 26.7% lower than the original
allocation's RMSE. This is not a percentage of error causally explained.
The interventions interact nonlinearly. Protected curves cannot inherit the
old random/trailing comparisons because their per-bond ranks differ.

## Family-only and restoration interventions

| Intervention | Removed directions | Actual removal | Output RMSE |
| --- | ---: | ---: | ---: |
| Cut only 136 moment/Padé bonds | 136 | 1.4665% | 0.421296 |
| Cut only 96 attention scalar bonds | 96 | 1.0352% | 0.073429 |
| Cut all 232 width-two bonds, no wider bonds | 232 | 2.5016% | 0.421317 |
| Original 30% ranks, restore moment/Padé | 2,646 | 28.5314% | 0.334606 |
| Original 30% ranks, restore all width-two | 2,550 | 27.4962% | 0.333923 |

Normalization-only cuts cause substantial error even without wider cuts.
Restoration helps the original 30% model, and reallocating cuts elsewhere to
restore the exact 30% budget retains nearly the same normalization-protection
benefit. Together these support a damaging allocation choice at 30%.

Similar scalar RMSE for normalization-only and all-narrow interventions does
not establish identical predictions or no effect of attention cuts. Paired
predictions are not saved in these receipts. None of these interventions
establishes an optimal allocation or a guarantee at other input distributions.

## Verification and limits

- The original 66 guarded tests and five new allocation/boundary tests passed
  before launch. Independent review caught a test-discovery guard gap. Guards
  now install at test-module import, and both consumer and test sources are
  statically audited and hashed. The scoped cumulative re-review was clean.
- A fresh 0% worker passed the unchanged full-rank replay gate before the other
  26 jobs could start. Scaled replay error was 2.34243e-14, distinct from its
  absolute RMSE of 1.05734e-14.
- Every physical graph was saved, reloaded and compared with an independently
  applied full-width rank mask. Maximum projective discrepancy was 7.64944e-14,
  below the unchanged 3e-10 gate.
- All 27 rank plans were independently reconstructed with exact rational event
  ordering. Every saved producer width and all 820 consumer-input occurrences
  per graph match the declared ranks. Budget denominators and nesting passed.
- All 54 saved graph/array file hashes were independently recomputed on Athena
  and match the 27 result receipts.
- All 27 models have 64/64 valid inputs. Minimum denominator margins across
  these runs remain above 0.129. Validity on this panel is not a global
  denominator-stability certificate.
- Reused bases passed explicit contracted-environment eigen-equation checks,
  maximum residual 2.04839e-15. No new cutoff was marked unresolved. These
  additional cutoffs were **not independently clone-checked** and must not be
  retroactively included in the producer's 1,540 retained-space checks.
- No new factorization or eigendecomposition was needed. All 27 workers record
  zero QR, zero EVD and zero prohibited calls. The accepted producer used direct
  QR and explicit environment contraction/EVD, with no prohibited alternatives.

The scope of the original [Dooms pipeline](https://arxiv.org/html/2504.02667v1#A7)
and our shared-occurrence extension remains important. Summed independent
occurrence single-cut coefficient loss does not certify simultaneous all-bond
truncation or decoded rational-output error. These results do not falsify the
accepted full-rank block decomposition, and they do not establish useful
full-policy compression or human-interpretable directions.

## Reproducibility

- Rerun source commit: `ffff39b272e2d91d36ff359d0532e8e423b5f02a`.
- Immutable producer source: `32fb1e7b74c10b807de2696840473622b6e44a89`.
- Producer receipt SHA-256: `bc961a3d700e94a15341aa2700eb85ca638cda00d80a2c0e7f6d803b14753a3d`.
- Consumer SHA-256: `b84260e03413c5259593c0e26096ccf029d811675f3d98d21d0eb1dea122bf69`.
- Consumer tests SHA-256: `1fd0ba69de2a00f893fc5bfa986d9af61c5e55d96b4b7ba1fedfa9ecb236b2ba`.
- Frozen packet SHA-256: `61c18260b66a1a8a91182b1e2eb7edf3e64004d0cfac0c3259fa5d0adc612a3d`.
- Checkpoint, panel and canonical hashes are unchanged from the original report.
- Athena artifact directory: `/work/joy/x-vla-odt-rank-ablation-v3/results`.
  Each `task_0` through `task_26` contains `result.json`, `graph.json` and
  `arrays.npz`. Receipts retain exact ranks, all denominator margins, errors,
  source identities and artifact SHA-256 hashes.
- Job 835699 is the full-rank dependency gate. Array 835700 contains the other
  26 evaluations, at most six concurrently, four CPUs and 16 GiB per worker.
  All exited successfully, with individual Slurm elapsed times of 18–33 seconds.
- Runtime: Python 3.10.19 and NumPy 2.2.6. Campaign Modal spending remains $0.

The original numerical source closure was not modified. Reproduce the guarded
consumer tests from this source commit with
`python -m scripts.test_odt_rank_ablation`. The separate
`scripts/odt_rank_ablation.py` consumer authenticates the original producer
before accepting its artifacts. These experiments were designed after seeing
the original curve and must be labeled as post-hoc diagnostics.
