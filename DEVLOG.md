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
