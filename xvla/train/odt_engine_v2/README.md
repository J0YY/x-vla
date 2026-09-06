# Shared-basis ODT: contract, proof, code, tests

This engine preserves a **fixed, symmetrized, ordered tensor representation**
of a Padé policy. Preservation and ranking are different obligations. The shared
basis ranks directions by a sum of independent occurrence-cut losses, not by
simultaneous truncation error or robot success. This refactor does **not**
establish corrected trained-block ODT or a completed full-policy result.

The factorization, downstream contraction and full-gauge structure follow
[Dooms et al., Algorithms 1–3](https://arxiv.org/html/2504.02667v1#A7).
The symmetric completion convention and shared-occurrence objective below are
explicit extensions. The independent tree transcription is
[dooms.py](../../../research/odt_reference/dooms.py), with no production imports.
[SOURCES.md](../../../research/odt_reference/SOURCES.md) records provenance.
Authenticated author code has not been located. This is not a claim to reproduce
Thomas's original 25-line implementation verbatim.

## 1. What is fixed

Take a finite real acyclic graph of unary and binary tensor cores with ordered
input slots and one external linear output head. Freeze its weights, biases,
normalization buffers, Padé coefficients, input conventions, masks, topology,
output selection and Euclidean output metric. Retain the deployed numerator and
denominator representation. Any external mode selection or action decoding is a
separate observable, not another bilinear core.

Unfold the graph recursively **without memoization**. Every copied physical
input is a separate ordered tensor axis. Evaluating these copies on their common
physical input recovers the nonlinear graph. The ordered coefficient tensor
contains more information than this diagonal evaluation.

Apply the local symmetrization in Step 1 below, then declare the resulting full
ordered coefficient tensor to be **T**. Subsequent exactness statements refer
to this T. Local symmetry does not make T, its ranking, or its coefficient norm
intrinsic to every equivalent polynomial or rational quotient. Distinct nodes
that happen to evaluate identically are not silently identified.
The lift identifier is `dooms_symmetric_ordered_lift_v2`.

## 2. Four short arguments

### Step 1: genuinely repeated inputs may be symmetrized

For a binary core whose two child objects are identical, set

\[
C^{\mathrm{sym}}_{oij}=\tfrac12(C_{oij}+C_{oji}).
\]

For every common input z, the antisymmetric part cancels:

\[
\sum_{ij}(C-C^{\mathrm{sym}})_{oij}z_i z_j=0.
\]

Thus the **evaluated function** is unchanged. The pre-symmetrization ordered
tensor need not be unchanged. This is why T is fixed only after this step.
Applying the same operation to distinct image and language inputs is not valid.

Factor tied-input cores in symmetric coordinates: store diagonal entries once
and upper-triangle off-diagonals multiplied by sqrt(2). For a symmetric slice,

\[
\|C\|_F^2=\sum_i C_{ii}^2+2\sum_{i<j}C_{ij}^2.
\]

Packing is therefore an isometry, not a new interaction weighting. Direct QR in
this space retains min(out, d(d+1)/2) shape-selected directions. Unpacking keeps
**every** Q slice symmetric, including rank-deficient completions. No numerical
rank is inferred. Raw tied-input symmetry is checked before any child factor
can hide an asymmetric component through a dimension reduction.

### Step 2: direct QR and every-occurrence substitution preserve T

For each unique node, children first, flatten its input indices into I and use
direct reduced QR of the unfolding's transpose:

\[
C_v^\top=\widehat Q_v R_v,\qquad
C_v=L_v Q_v,\quad L_v=R_v^\top,\quad Q_v=\widehat Q_v^\top.
\]

This is the **LQ convention**, despite historical rq API names. The left factor
is lower triangular. In tied coordinates Q is unpacked as above. Retain Q as
the producer and absorb L into every connected parent-input slot:

\[
A'[\ldots,a,\ldots]=\sum_o A[\ldots,o,\ldots]L_v[o,a].
\]

At the root, absorb L into the external head. In particular,

\[
z=Lq,\qquad y=A(z,z)=A(Lq,Lq).
\]

Both inputs receive L. Each clone occurrence is an exact substitution, so the
complete ordered T is preserved after each step. **Singular L is fine**. There
is no inverse, pivot test or alternative factorization. Direct Q, including its
orthogonal completion, makes each cloned subtree canonical by induction from
its independently ordered leaf axes.

Scale is also part of the tensor: a stored mantissa represents its value times
2 raised to log2_scale. The helper `ops.py::_absorb_input_factor` implements
exactly the indexed contraction above and adds the input exponents plus any
explicitly stripped normalization exponent. Dropping a scale changes the tensor.

### Step 3: message aggregation equals the sum of clone environments

For clone occurrence p of bond v, cut that single occurrence. Contract two copies
of its downstream network over outputs and all other ordered physical indices,
leaving the two cut coordinates open. Call this literal environment E_p. Define

\[
H_v=\sum_{p\mapsto v}E_p.
\]

All occurrences use the same coordinates inherited from their shared producer.
H is **not** the environment of a hypothetical bond formed by merging all cuts.
For the root, with linear head B,

\[
H_{\mathrm{root}}[a,b]=\sum_\eta B[\eta,a]B[\eta,b].
\]

For a canonical binary parent Q, the explicitly contracted child messages are

\[
\Phi_L(E)_{ab}=\sum_{opj}Q_{oaj}E_{op}Q_{pbj},\qquad
\Phi_R(E)_{ab}=\sum_{opi}Q_{oia}E_{op}Q_{pib}.
\]

For a unary parent, omit the sibling index. Canonical sibling subtrees contract
to matching sibling indices, which justifies this local recurrence. Each clone
child has a unique parent occurrence and input role. Since each Phi is linear,

\[
H_v=\sum_{(w,r):\,\mathrm{child}_r(w)=v}\Phi_{w,r}(H_w)
    =\sum_{p\mapsto v}E_p.
\]

This proves the parents-first recurrence by induction. A repeated edge receives
both role messages. A diamond receives all incoming parent messages. Aggregating
before propagation is exact for **this sum**, not a proof that independently
chosen occurrence-specific eigenbases have the same objective.
The `incoming_occurrences` metadata counts immediate DAG slots, not expanded
clone multiplicity. Multiplicity is already represented by the accumulated H.

### Step 4: full eigenbasis gauges preserve T

Eigendecompose contracted H_v and let U contain a full orthonormal eigenbasis.
Apply the same coordinate change to the producer and every consumer:

\[
Q'_{\alpha I}=\sum_a U_{a\alpha}Q_{aI},\qquad
A'_{\ldots\alpha\ldots}=\sum_a A_{\ldots a\ldots}U_{a\alpha}.
\]

Summing over the full new bond gives

\[
\sum_\alpha U_{a\alpha}U_{b\alpha}=\delta_{ab}.
\]

Each bond substitution therefore preserves the entire ordered T, including
repeated occurrences. This identity is the proof, **not** a self-overlap
diagnostic executed by the tests. Keeping only k columns instead inserts a
projection and changes T.

## 3. Exactly what the ranking optimizes

Let U_k be one orthonormal k-dimensional basis used for a shared origin v.
Let T_(p,U_k) restrict **only occurrence p**, leaving every other cut unchanged.
Canonicality gives

\[
\sum_{p\mapsto v}\|T-T_{(p,U_k)}\|_F^2
=\operatorname{tr}(H_v)-\sum_{\alpha=1}^k u_\alpha^\top H_vu_\alpha.
\]

Consequently the leading eigenspace minimizes the sum of independent single-cut
coefficient losses. This is the shared-subspace extension's optimization proof.
It does **not** imply that the common basis diagonalizes each E_p, minimizes
simultaneous tied or multibond truncation error, minimizes decoded action error,
or preserves robot capability. A tested symmetric fixture has simultaneous loss
8 for the leading direction and 20/3 for another rank-one direction.
[SHARED_DAG.md](../../../research/odt_reference/SHARED_DAG.md) gives the example
and explicit cross-occurrence terms.

In Dooms's symmetric cloned chain, H is the common occurrence environment times
the occurrence count. Eigenvectors agree, while absolute eigenvalues include
that multiplicity. A tied eigenvalue at the retained boundary makes the retained
subspace nonunique. Tests use resolved eigenspaces or whole degenerate clusters,
not arbitrary eigenvector signs or a prohibited overlap factorization.

For Padé, coefficient preservation implies quotient preservation wherever the
denominator is nonzero. Approximate coefficient preservation alone does not
bound action error near a small denominator. Truncation must be evaluated on
decoded actions, denominator behavior and closed-loop success separately.

## 4. Proof-to-code-to-test correspondence

| Obligation | Production implementation | Independent check |
| --- | --- | --- |
| Symmetrization preserves tied evaluation | compiler.py::_Builder.cp, factorization.py::_packed_cp_unfolding | Weight-derived exporter in research/odt_reference/weights.py, test_full_block_export_uses_weight_derived_symmetric_ffn_and_all_six_norms |
| Direct QR plus all input occurrences preserve ordered T | core.py::canonicalize_implicit_dag_direct_rq, factorization.py, ops.py::_absorb_input_factor | test_every_deficient_qr_and_gauge_prefix_matches_literal_clone compares complete coefficients and transported coordinates after every step |
| Contracted H equals the literal occurrence sum | core.py::_environment_records, ops.py::_role_environment | _literal_environments independently sums external indices in two clone copies, compared at each QR prefix |
| Full common gauges preserve ordered T | core.py::_apply_gauge, diagonalize_implicit_dag_full_rank | The same prefix test compares complete coefficients after every gauge, plus independent clone EVD tests |
| Leading coordinates minimize independent single-cut loss only | Contracted-environment EVD in core.py | Independent shared-DAG tests check the loss identity and simultaneous-truncation counterexamples |
| Numerical acceptance rejects finite errors and unsupported shapes | acceptance.py::accept_implicit_odt | tests/test_odt_acceptance.py injects finite errors, later NaNs, final-callback corruption and a late unsupported shape |

The production prefix and weight-export tests are in
[test_odt_refactor_integration.py](../../../tests/test_odt_refactor_integration.py).
The independent reference does not import the production compiler or sweeps.
Tiny complete coefficient comparisons and synthetic complete blocks establish
different bounded checks. They do not materialize the complete trained-policy
coefficient tensor. Shared-versus-clone aggregation verifies H, not equality to
an alternative procedure selecting a different basis at every clone occurrence.

## 5. Why sharing is compact, and what still costs memory

For V unique nodes and E ordered parent-input slots, the graph algorithm uses
one QR and one EVD per node and one message and transport per slot. Graph
traversal and structural metadata are O(V+E), rather than proportional to the
potentially exponential number of clones. This does not make local tensors free.

For a core with m outputs, n=a*b stored input coordinates and s QR coordinates,
let k=min(m,s). Untied s=n. For repeated width d, s=d(d+1)/2 but stored n=d*d.
QR costs O(m*s*k). The unfolding has m*s entries and the retained, unpacked Q has
k*n entries, plus the m*k factor and numerical-library workspace. For CP rank r,
materialization and reconstruction add O(m*r*n + m*k*n) arithmetic. A CP input
transport costs O(r*m*k) at that slot. Dense transport costs depend on the
corresponding local tensor shape.

A binary environment contraction can cost O(k*k*a*b + k*a*b*(a+b)), with
O(k*a*b + a*a + b*b) temporary storage. Each d-dimensional environment stores
d*d entries and its EVD costs O(d*d*d). Persistent Q storage sums k*n over nodes.
The live environment frontier, graph, factors and heads add storage. Exact scale
exponents use integers whose bit lengths can grow with depth.

`predict_direct_rq_route_inventory` propagates shape-only widths across the
whole DAG before factorization. Its all_node_* counters include unary nodes as
well as binary ones, unlike the retained legacy binary counters. Current explicit
unfolding and retained-Q bounds are each 4,000,000 elements per node. These are
route bounds, not a whole-device peak-memory certificate. Unsupported shapes
fail closed. A new scalable streamed representation has **not** been proved
by deleting the old numerical-pivot route.

Shape-only width reductions must be counted separately from subsequent spectral
removal. Freeze both original and post-QR dimension totals before reporting a
dimensions-removed curve. Storage bytes are not the x-axis for that curve.

## 6. Numerical acceptance and reproducibility

`core.py` contains low-level sweeps and diagnostics. It rejects nonfinite replay
and off-diagonal values, but its reporting APIs do not reject every large finite
error. A successful low-level return is not numerical acceptance.

Use the separate `accept_implicit_odt` harness for a supplied graph and explicit
replay inputs. It validates the declared lift and **all** propagated routes
before the first QR, copies the input graph, checks every QR replay, computes a
fresh final replay after callbacks, contracts fresh environments internally,
and checks full-gauge replay and recontracted aggregate diagonalization. It
accepts no caller-supplied environment records. Defaults are:

| Check | Acceptance threshold |
| --- | --- |
| Per-step and final projective replay | <= 3e-8, using the recorded projective replay metric |
| Recontracted aggregate off-diagonal ratio | <= 3e-8 |
| Maximum decoded difference divided by max(1, maximum reference magnitude) | <= 3e-8 |
| Minimum relative projective denominator, abs(last coordinate)/max(abs(row)) | > 1e-12, before and after |

All inputs to these decisions must be finite. A zero projective row is rejected.
The denominator margin is a scale-free **chart** check, not an absolute polynomial
denominator bound or quotient-intrinsic metric. These tolerances are explicit,
versioned defaults for bounded float64 validation, not a claim that all hardware
or checkpoints meet them. Replay checks cover only the supplied inputs and do
not replace independent coefficient or occurrence-environment comparisons.

Install [requirements-validation.txt](requirements-validation.txt) in a dedicated
Python 3.12 environment, then run this one command from the repository root:

```sh
OMP_NUM_THREADS=1 python -m scripts.validate_odt_proof
```

It runs independent NumPy mathematics, source/boundary checks and guarded Torch
production fixtures in separate processes, with NumPy 2.2.6, Torch 2.8.0 and
pytest 8.4.2 pinned. The guards audit local numerical import closures before
execution and restrict QR/EVD calls to declared sources. They prohibit SVD,
Gram-factorization shortcuts, polar, rank-test, inverse, least-squares and
fallback routes, including tests and diagnostics. Algorithm 2's prescribed
explicit downstream contraction remains required, even though Dooms calls its
result a Gram matrix. No self-overlap validation diagnostic is permitted.
This is a scientific source audit and runtime guard, not a hostile-Python sandbox.

The command saves commands, versions, source hashes and test results in
[proof_validation.json](proof_validation.json). Tolerances are pinned in the
hashed acceptance source. It loads no trained checkpoint and launches no cluster
job. The saved record excludes five optional Numba compiled-executor tests because
that dependency was unavailable. [validation.json](validation.json) is the
earlier refactor snapshot with its own source hashes, not certification of later
source changes. The reference's historical trained-export report records checkpoint
provenance in [BLOCK_VALIDATION.md](../../../research/odt_reference/BLOCK_VALIDATION.md).
That checkpoint is not silently treated as exercised by this local command.

## 7. Ownership, compatibility and remaining scientific gates

| File | Narrow responsibility |
| --- | --- |
| core.py | Three sweeps, one environment recurrence, one EVD/gauge path |
| factorization.py | Direct QR, symmetric coordinates, shape-only bounds |
| compiler.py | Weights and Padé primitives to the declared lift |
| types.py, graph.py, ops.py | Representation, traversal, indexed contractions |
| validation.py, diagnostics.py, acceptance.py | Provenance, reports, explicit acceptance thresholds |
| oracles.py | Explicit clones and controls, outside primary imports |
| compat.py | Historical rank-bank callback binding, outside the sweeps |

`implicit_sparse_projective_odt.py` is a 214-line explicit compatibility export
module with no algorithm bodies, down from the 4,959-line Git baseline. This is
not a claim that all implementation machinery became 25 lines. Compiler, graph
and exact scale handling remain separate. The old pivot/compact-solve route and
duplicate EVD paths were removed. Compilation, serialization, streaming and
cluster machinery are not hidden inside the mathematical reference.

Old serialized class tags remain readable, but old lift certificates cannot
enter the new environment pass. Old source attestations do not authorize this
engine. Removed fault-injection and output-metric options are intentional API
breaks. Retired oracle exports `CloneEVDTrace` and
`apply_shared_eigenbases_to_explicit_clone_occurrences_control` migrate to
`diagonalize_shared_and_explicit_clone_independently` and
`IndependentCloneEVDTrace`, not a drop-in alias. No active repository caller
imports the retired symbols. The compatibility entry point binds historical
rank-bank callbacks to copied networks. Direct low-level callers must bind any
identity-sensitive callback to the actual target and use `copy_network=False`.

Remaining go/no-go work, before claiming corrected full-policy ranking:

1. Complete the unchanged **trained-block** reference comparison. Compare
   occurrence-specific leading bases, the common shared basis and matched
   trailing/random controls. Measure actual simultaneous truncation and decoded
   action error with denominator behavior. Match retained ranks per occurrence
   and report independent-basis duplication separately. Synthetic replacements
   or trained export replay alone do not meet this gate. The existing trained
   clone attempt exceeded its memory bound and did not complete.
2. Demonstrate supported routes and memory feasibility for **every production
   shape**, then pass corrected production-versus-reference checks. Bounded
   local tests do not establish the full graph's feasibility.
3. Run a fresh full-policy campaign with frozen source/checkpoint hashes,
   representation, dimension denominator and paired rollout protocol. Save and
   replay each physically reduced policy, then measure action error, denominator
   behavior and closed-loop success. Reuse a baseline only if its protocol is
   identical. Historical full-policy artifacts are not recertified here.

The proofs establish exact-arithmetic properties of the declared algorithm.
Independent tests provide bounded implementation evidence. A corrected
full-scale run must establish production feasibility. None substitutes for the
other two.
