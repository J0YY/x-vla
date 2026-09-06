# A small, auditable ODT reference

Current complete-block results and cleanup: `BLOCK_VALIDATION.md`. The combined
suite now has 48 tests. `VALIDATION.md` and `validation.json` preserve the initial
31-test reference checkpoint, not an attestation of subsequent source changes.

This directory is deliberately independent of `xvla`, the production compiler,
streaming kernels, saved artifacts, and cluster tooling. It is a bounded dense
mathematical reference, not a new full-VLA implementation. Production remains
paused pending the semantic and scaling gates below.

Read `dooms.py` first. It implements the three passes in Appendix G of
[Dooms et al., 2025](https://arxiv.org/html/2504.02667v1#A7).
`weights.py` constructs its inputs independently from raw parameter arrays.
`shared_dag.py` and `SHARED_DAG.md` state and test the extension separately.
`SOURCES.md` distinguishes the published pseudocode from related author code.

Run all checks from the repository root:

```sh
python3 -m research.odt_reference.run_tests
```

Use `--suite tree` or `--suite dag` for separately audited source closures.
The runner rejects prohibited numerical calls, counts direct QR and environment
eigendecompositions, and prints source hashes. It is a bounded source audit and
runtime guard, not a proof about arbitrary Python dependencies.
The initial run and independent review are recorded in `VALIDATION.md` and
`validation.json`. The current block gate is documented in `BLOCK_VALIDATION.md`.

## The measured object comes first

Write a tied chain as `z = E x`, followed by `z = C_i(z,z)`, and output `U z`.
Each core has axes `[output,left,right]`. Before decomposition, its two tied
input legs are symmetrized. Untying the copies of each input yields an ordered
multilinear coefficient tensor. This is the object measured here.

It is not a unique representation of the tied-input polynomial. Local sibling
symmetry is not full symmetry of all repeated raw inputs. A Padé numerator /
denominator representation introduces further choices. None of these choices
are erased by accurate full-rank replay, or by two interpreters accepting the
same compiled object.

## Derivation of the compact core

### 0. Symmetric coordinates, including deficient cores

For a symmetric `h` by `h` input slice, store its diagonal entries and
`sqrt(2)` times its upper off-diagonal entries. This is a fixed orthonormal
coordinate system on the symmetric subspace. Apply direct reduced QR to the
transpose of that packed unfolding: `M.T = Q R`, hence `M = R.T Q.T`.
Unpack `Q.T` by dividing the off-diagonal entries by `sqrt(2)` and reflecting.
This uses the LQ triangular orientation: the left factor is lower triangular.
It is a direct-QR right-isometric factorization, not a claim to reproduce the
author's upper-triangular RQ gauge or unpublished implementation byte-for-byte.

All retained Householder columns, including numerical-null-space completions,
then represent symmetric tensors. Their count is selected solely by shape,
`min(output_width, h*(h+1)//2)`. A singular triangular factor is valid and never
inverted. This completion convention is an explicit extension derived here,
not an implementation detail specified in the paper.

### 1. Orthogonalize

Factor the embedding first. At each subsequent core, absorb the previous
triangular factor into both input legs, then factor that core. Absorb the final
triangular factor into the output map. Reconstruction `C = R Q` and the two
incident absorptions give exact equality of the ordered coefficient tensor.
Changing both input coordinates together preserves sibling symmetry.

### 2. Contract environments, then eigendecompose

For canonical cores, the unembedding gives

`E_last[a,b] = sum_o U[o,a] U[o,b]`.

One occurrence of the preceding bond has environment

`E_i[a,b] = sum_(u,v,c) Q_i[u,a,c] E_next[u,v] Q_i[v,b,c]`.

This is an explicit downstream tensor-network contraction, including the
sibling-leg trace. No unfolding self-overlap shortcut is used. Sibling
symmetry makes the other input-role environment equal in this tied-chain
case. Eigendecompose the resulting environment and order its eigenvalues
descending. A cutoff through an eigenvalue tie does not identify a unique
retained subspace.

### 3. Absorb retained bases

Let `B_i` contain the retained environment eigenvectors at each bond. Set

`E_new = B_0.T E`,

`C_i_new = B_(i+1).T C_i (B_i tensor_product B_i)`,

`U_new = U B_last`.

Keeping every shape-supported coordinate reconstructs the coefficient tensor.
Keeping fewer columns physically narrows the adjacent tensors. The code
contracts these factors directly, without constructing projector matrices.
No printed global truncation theorem or behavioral-accuracy bound is claimed.

## Validation boundaries

- Raw weights are exported using independently checked index formulas, not the
  production FFN compiler. Symmetrization must preserve its biased forward map.
- Direct QR reconstruction and core symmetry are checked, including zero,
  dependent-row, and rectangular cases. There are no orthogonality self-overlap
  diagnostics, numerical rank tests, or alternate factorizations.
- Full ordered coefficients and actual downstream environments are compared
  against bounded independent constructions. Output replay is additional
  evidence, not a substitute.
- Retained subspaces are compared through direct-QR coordinate reconstruction
  at resolved gaps, not by demanding matching eigenvector signs or constructing
  projector self-overlaps. Degenerate clusters are treated separately.
- Shared-DAG occurrence aggregation is checked against an explicitly unfolded
  no-memo tree after each canonicalization step. Its objective and exclusions
  are derived in `SHARED_DAG.md`.

## Before returning to the VLA

The typed symmetric block export now includes attention/residual sharing and
the fixed Padé representation. Small joint-output blocks pass the independent
ODT gate. The unchanged trained vision block passes export replay, but its
explicit-clone ODT run exceeds the oracle's memory bound. A streaming
implementation is not yet accepted. Symmetrizing an FFN can be represented
by pairing CP terms, without changing policy weights or retraining. That does
not by itself validate every other compiler primitive or a full-policy ranking.

This directory does not resume jobs, deployment, Git pushes, or the previous
truncation-curve deadline.
