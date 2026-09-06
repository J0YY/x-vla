# χ-ODT

This repository tests whether Orthogonalisation, Diagonalisation, and Truncation (ODT) can run on a
capable vision-language-action policy, not just a small classifier.

The current implementation applies ODT to the complete shared tensor graph. It uses direct QR/RQ,
updates every use of a shared bond, builds downstream environments by explicit contraction, and
checks every stage against a separately unfolded reference implementation.

## Start here

Read these files in order:

1. `paper/odt.tex` for the claim and experiment.
2. `DEVLOG.md` for the short history of the current implementation.
3. `xvla/train/implicit_sparse_projective_odt.py` for the production ODT path.
4. `xvla/train/direct_odt_clone_reference.py` for the independent correctness oracle.
5. `scripts/odt_direct_only_compliance.py` for the prohibited-route checks.
6. `modal_odt_dimension_curve.py` for the pinned closed-loop evaluation.

Older papers, experiments, plots, and result bundles are under `prev/`. They are not part of the
current ODT source closure.

## Review the implementation

Run the focused local gate:

```bash
python3 scripts/run_direct_odt_dimension_ladder_tests.py
```

This checks the direct-only source closure, runtime guards, mapped artifacts, the clone oracle, and
the dimension-ladder machinery. It requires the project test dependencies, including PyTorch and
pytest.

The canonical path must never call SVD, pseudoinverse, least-squares, Gram, polar, covariance, or
normal-equation routines. Rank deficiency does not relax this rule.

## Produce the number to improve

The optimization metric is closed-loop success rate. Exact replay, source identity, and numerical
compliance are required gates, not metrics that can be traded away.

Run a paired baseline and candidate in the pinned Modal environment:

```bash
modal run modal_odt_dimension_curve.py::evaluate \
  --removal 0 \
  --run-name baseline_review

modal run modal_odt_dimension_curve.py::evaluate \
  --removal 30 \
  --run-name candidate_review \
  --artifact-manifest-sha256 <manifest-sha256> \
  --receipt-sha256 <receipt-sha256>
```

The reduced-policy hashes come from the authenticated dimension-ladder receipt. The evaluator
rejects unsupported removal levels, mismatched checkpoints, changed source bundles, incomplete
episodes, and prohibited numerical calls.

After copying the two emitted `result.json` files locally, validate and compare them:

```bash
python3 scripts/odt_hill_climb.py score baseline.json
python3 scripts/odt_hill_climb.py compare baseline.json candidate.json
```

The comparison prints baseline success, candidate success, and their difference. That is the small,
repeatable loop to use when changing truncation or rank allocation.

## Where to go next

The full-rank decomposition is complete. The open problem is useful physical truncation of the
capable checkpoint.

A good next change should:

- modify one clearly stated truncation or rank-allocation choice,
- keep the checkpoint, episode panel, source bundle, and protocol fixed,
- pass every direct-only and exact-replay gate,
- improve paired closed-loop success, and
- include the two result files or their immutable receipts in the PR description.

Do not treat an unpaired historical run as the baseline. Do not claim compression from exact
full-rank replay alone.
