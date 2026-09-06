# Bounded reference validation, 2026-09-06

Historical initial-reference checkpoint. For subsequent source changes and the
complete-block gate, see `BLOCK_VALIDATION.md`. Hashes here remain those of the
initial run, not an attestation of the later source tree.

The combined guarded suite passes **31 tests** on NumPy 2.4.2 with float64
arrays. Its runtime counters report 312 direct QR calls, 203 eigendecompositions
of contracted environments, and zero prohibited numerical-route attempts.
The executable source hashes and run record are in `validation.json`.

Run from the repository root:

```sh
python3 -m research.odt_reference.run_tests
python3 -m research.odt_reference.run_tests --suite tree
python3 -m research.odt_reference.run_tests --suite dag
```

The separately selectable suites contain 13 tree/weight tests and 18 DAG tests.
They use NumPy and the standard library, without importing the production
compiler, policy model, old ODT implementation, SciPy, or cluster tooling.

## What passed

- Independent symmetric CP export from raw affine weights, biases, and residuals,
  checked with literal index sums and raw forward evaluation. Exported cores also
  pass all three ODT stages, including a deliberately dependent-weight case.
- Full ordered coefficient preservation after each canonicalization step.
  The clone interpreter independently packs coordinates, runs direct QR and
  absorbs factors at every occurrence, without memoization.
- Symmetry under shape-selected QR completions, including dependent, zero and
  rectangular unfoldings. Original skew is rejected before a narrow child factor
  could conceal it. Cross-modal legs are not indiscriminately symmetrized.
- Actual occurrence environments from literal two-copy downstream contractions,
  not output probes or opened-unfolding self-products. Shared and cloned results
  agree after each canonicalization step, including deficient fixtures.
- Retained subspaces at resolved eigengaps agree through direct QR and coordinate
  reconstruction. Eigenvector signs and rotations within tied clusters are not
  incorrectly treated as failures or unique interpretations.
- Full-rank common gauges and physical cuts agree with independently applied
  clone transformations after every prefix. Physical tree slicing also agrees
  with the corresponding coordinate masks.
- Repeated-edge multiplicities, heterogeneous diamonds, symmetric-chain
  equivalence to the paper environment up to occurrence count, and the exact
  sum-of-independent-single-cut-loss identity.
- Counterexamples to stronger claims: a common basis need not diagonalize each
  occurrence, tied derivative cross terms can cancel, and the best single-cut
  ordering need not give the best finite simultaneous tied truncation.

The symmetric two-output counterexample is especially explicit: the spectral
rank-one choice gives finite tied squared error 8, while another direction gives
20/3. This limits the claimed optimization objective, not exact decomposability.

## Independent review

Three agents separately handled source provenance, independent weight tests,
and the DAG derivation. A source auditor also cold-reviewed the other work.
An additional read-only Codex review ran two rounds, with fixes made by the
main session. The final cumulative review reported no actionable P0–P3 findings
and independently reproduced the 31-test result and kernel counters.

The reviewed files were `dooms.py`, `weights.py`, `shared_dag.py`,
`clone_oracle.py`, `test_reference.py`, `test_shared_dag.py`, `run_tests.py`,
`README.md`, `SHARED_DAG.md`, and `SOURCES.md`. This validation record itself was
written afterward from the completed run. Review and tests are evidence, not
a guarantee that every untested case is bug-free.

## Limits and next acceptance gate

This is a small dense mathematical reference. The 129-line tree implementation
and 215-line DAG implementation include validation and documentation, but no
compiler, streaming, serialization or experiment-launch code. Tests and the
explicit clone oracle live separately.

The reference fixes a locally symmetric ordered-leaf tensor representation.
It does not claim a unique raw-variable polynomial or rational-quotient metric,
nor a globally optimal simultaneous truncation or a closed-loop success bound.
Symmetric-coordinate completion is an explicit extension derived here, not
authenticated as Thomas Dooms's original 25-line implementation.

Before production use, independently specify and export the complete bounded
VLA block from its actual weights, including attention, residual sharing, and
the fixed Padé representation. Compare cores, full coefficients, occurrence
environments and retained subspaces against this reference after every step.
Only after that gate should the streaming implementation and full-policy
truncation curve be reconsidered.

No production model, paper, cluster job, Modal deployment, Git commit, or Git
push was changed by this reference work. The prior campaign hold remains.
