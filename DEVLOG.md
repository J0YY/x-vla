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
