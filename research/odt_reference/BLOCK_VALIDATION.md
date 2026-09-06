# Complete-block gate and implementation cleanup

This supersedes the initial 31-test snapshot for the current source tree.
The mathematical kernel remains separate from export, independent oracles,
checkpoint I/O, tests, and any production execution machinery.

## Results, 2026-09-06

- **Small complete joint-output block:** all three ODT passes agree with the
  independent explicit-clone implementation. Two tokens, all six distinct
  nontrivial Padé norm records, affine biases, attention, FFN, gains, both
  residuals, and a common final output denominator are included. A deliberately
  deficient projection passes as well. These are synthetic weights.
- **Unchanged trained block export:** first vision block of checkpoint
  `b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee`,
  width192, 8 heads, CP rank576, two fully visible tokens, joint output. No
  trained dimensions or weights were removed. On two synthetic input sequences,
  exported replay matches the independently transcribed raw NumPy forward with
  maximum absolute error **6.217248937900877e-15** and max-entry-scaled relative
  error **2.3685313406884594e-15**. This is not a PyTorch-runtime or LIBERO result.
- **Trained-block ODT:** not completed. The 453-node shared export's explicit
  clone tensors require **26,655,005,520 bytes** before additional QR workspace.
  This exceeds the local machine's 25,769,803,776 bytes (24GiB) of total RAM.
  The bounded 256MiB clone gate rejects before allocation. This is a limit of
  the deliberately materialized oracle, not a theorem against ODT or sharing.

The combined guarded suite passes 48 tests in 46.004 seconds, with 16,122 direct
QR calls, 483 permitted environment eigendecompositions, and zero prohibited
attempts. A separate read-only reviewer completed two review rounds and repeated
all 48 tests, reporting no actionable P0–P3 issues in the final round. This is
bounded validation, not a guarantee of bug-free behavior at production scale.
Source hashes and the test record are in `block_validation.json`.
The unchanged checkpoint export record and its source hashes are in
`trained_block_export_validation.json`.
The runner audits itself and rejects undeclared local numerical dependencies.
Original model source is read for specification, never imported by
this path. A separate restricted checkpoint reader loads only permitted raw
tensor storage, without importing torch or executing arbitrary pickle globals.

## Read the implementation in this order

1. `dooms.py`: direct QR, shape-only symmetric completions, explicit contracted
   environments, their EVD, and adjacent basis absorption. 129 lines including
   documentation and validation.
2. `shared_dag.py`: the common-bond extension, 215 lines. Its objective remains
   the sum of independent occurrence-cut losses described in `SHARED_DAG.md`.
3. `block.py`: a 244-line typed weight exporter and decoded replay helper. It
   contains no ODT pass, streaming code, serialization, or launch machinery.

`block_oracle.py` independently transcribes the raw source forward. The separate
`clone_oracle.py` and `clone_passes.py` implement the explicit-copy oracle.
`checkpoint.py` is strictly a file boundary. None belong inside the ODT kernel.
Line counts include comments, whitespace, and validation, not only mathematics.

## Exact source contract and measured object

`BlockSpec` declares all six norms as fixed degree-two Padé, a common epsilon,
both residuals enabled, head-dimension-squared score scaling, inverse-square-root
visible-key scaling, and the head count. These attributes are not inferred from
state-dict arrays. The actual checkpoint runner uses the configuration in
`scripts/run_capable_linear_fresh_capability.py::_config` and the constructor
defaults in `xvla/nn/block.py`, `attention.py`, and `normalization.py`.

Inputs are already at the block boundary. Positional addition is not inserted
again. All six norm records are checked before compilation, even for empty mask
rows. Supported denominators have positive constant and nonnegative remaining
coefficients, so omitted masked routes cannot conceal a real-input Padé pole.
All projection and FFN shapes must match exactly before any head selection or
homogeneous-coordinate permutation.

For coordinates `(n,d)` with denominator last, the primitive formulas are:

```text
affine:      (Wn + bd, d)
bilinear:    (B(n,m), de)
sum:         (ne + md, de)
concatenate: (ne, md, de)
Padé:        S = mean(n²) + eps*d², T = s*d²
             A = a0*T² + a1*S*T + a2*S²
             B = b0*T² + b1*S*T + b2*S²
             output = (n*A/sqrt(s), d*B), s = max(running_ms, 1e-12)
```

Identical-child cores are explicitly symmetrized before Algorithm 1, following
[Dooms §2 and Algorithms 1–3](https://arxiv.org/html/2504.02667v1#A7).
The joint output uses a declared common-denominator concatenation. Optional
selected-token compilation is a row diagnostic with a different output metric,
not evidence for joint-block ODT. No quotient-intrinsic metric is claimed.

## What the block oracle checks

Every Algorithm 1 step compares every explicit clone core and the head against
the shared graph, alongside direct local reconstruction. This certifies the
same tensor-network representation structurally. The astronomical coefficient
array is not materialized. Algorithm 2 contracts each clone occurrence
independently using the sibling-isometry consequence of its own direct QR pass,
then groups by origin. Small fixtures cross-check this exact contraction rule
against literal two-copy physical-index enumeration.

Actual environments and independently derived retained spans are compared at
resolved eigengaps. Algorithm 3 independently absorbs the supplied common bases
into all clone occurrences and matches shared-core coordinates and decoded
outputs. Sample-specific scalar normalization is used only for decoded replay,
never for weights, QR, environments, or EVD.

## Historical cleanup snapshot, before the production module split

The paragraphs in this section record the preceding cleanup, not the current
production layout. The subsequent symmetric-lift refactor and its distinct
validation boundary are documented in
[`odt_engine_v2/README.md`](../../xvla/train/odt_engine_v2/README.md).
Neither change retroactively validates historical production ranking.

Removed 201 lines of unused definitions from the old engine, plus their export
and EVD-allowlist entries. The retired symbols are `_core_input_dimensions`,
`_materialize_core`, `_scaled_batch_relative_error`, `CloneEVDTrace`, and
`apply_shared_eigenbases_to_explicit_clone_occurrences_control`. The tracked
cleanup totals 209 deleted lines and a six-line legacy/scope clarification.
These edits are recoverable from Git. Frozen experiment source snapshots and
results were not changed.

The old module is explicitly labeled legacy. Its remaining artifact schemas,
streaming representations, scale tracking, and callers are retained for
compatibility, not silently replaced by a tiny dense implementation. Moving
5,000 lines behind an import would not constitute a simplification. Nor can its
old certificates be relabeled as validating the new symmetric lift.

Production launches, deployments, commits, and pushes remain on hold. A full
trained-block ODT claim still needs a memory-bounded independent clone strategy
or an appropriately authorized larger validation run, followed by numerical
scale and replay checks. No compression curve or full-VLA result follows yet.
