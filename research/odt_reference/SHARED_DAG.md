# A shared-DAG extension, and its exact boundary

## Relation to the source

Dooms et al. define a nonlinear network through cloning, then analyze its
ordered-leaf multilinear tree. Section 2 imposes symmetry on the two inputs of
each cloned layer. Algorithm 1 pushes the R factor into both next-layer legs.
Algorithms 2 and 3 contract downstream environments and absorb their eigenbases.
Appendix F explains why symmetry allows one environment per layer.
[Source: Sections 2–3 and Appendices F–G](https://arxiv.org/html/2504.02667v1).

The derivation below is an explicitly scoped extension, not a claim made by that
paper. It concerns one chosen tensor representation. It does not identify a
function-intrinsic polynomial object, rational quotient, or intervention metric.

## Assumptions and objects

Let a finite, acyclic, real graph contain embedding leaves, unary cores, and
binary cores. Each core output is one unique bond. Every ordered parent-input
slot is a separate syntactic occurrence, including two slots with the same child.
An external linear head supplies the Euclidean output metric.

Unfold the graph recursively **without memoization**. Copies of the same physical
source become distinct, ordered tensor axes. Evaluating all copies on the same
physical input recovers the nonlinear graph, but the ordered coefficient tensor
contains more information than that diagonal evaluation.

An allowed local preprocessing step averages the two input orders only when
the two syntactic child IDs are identical. This preserves the evaluated graph,
but can change its ordered coefficient tensor. Fix the lift **after** that step.
The reference does not automatically symmetrize arbitrary image/language legs
or distinct child IDs. A proof that distinct children compute identical functions
could justify a separately declared lift change, but numerical agreement on a
few inputs is insufficient. All raw tied-child
cores must pass the symmetry check before any child dimension changes. Otherwise
a narrow child factor could conceal an originally asymmetric component.

The collapsed-sibling environment recurrence below assumes canonical cores.
This precondition follows from direct RQ, not a numerical overlap diagnostic.

## Algorithm 1 on the graph

For each unique node in child-before-parent order, matricize its core as
`out × product(inputs)` and compute direct QR of its transpose. Write the result
as `C = R Q`. Replace the core by Q. Contract R into **every** parent-input slot
connected to this unique output, or into the external head at the root.

Each substitution is an algebraic equality at every clone occurrence. Therefore
the complete ordered coefficient tensor is preserved, not only a sample of the
diagonal input evaluation. R need not be invertible. No numerical rank decision
or triangular solve enters this argument.

For genuinely tied binary inputs, use orthonormal symmetric coordinates:
diagonal entries once and upper-triangle off-diagonals weighted by sqrt(2).
Direct reduced QR there retains `min(out, d(d+1)/2)` coordinates selected by shape,
including completion directions when R is singular. Expanding the coordinates
back keeps every retained Q slice symmetric. This completion rule is our explicit
shape-only extension, not a numerical-rank interpretation of the paper.

## What the occurrence sum means

For an explicit clone occurrence p of unique bond v, cut that one bond. Contract
two independent copies of the downstream network, matching their output and all
other physical leaf indices, and leave the two cut coordinates a,b open. Denote
the resulting environment by `E_p[a,b]`.

Define `H_v = sum_{p with origin v} E_p`. This sum is well-defined in the common
coordinate system inherited from the tied core. It is not the environment of a
single hypothetical bond obtained by merging all occurrences.

At the root, `H_root[a,b] = sum_o head[o,a] head[o,b]`. For a unary parent with
aggregate H, its child message is

```text
M[a,b] = sum_{o,p} C[o,a] H[o,p] C[p,b].
```

For a binary parent the left and right messages are

```text
M_left[a,b]  = sum_{o,p,j} C[o,a,j] H[o,p] C[p,b,j]
M_right[a,b] = sum_{o,p,i} C[o,i,a] H[o,p] C[p,i,b].
```

Canonical sibling subtrees eliminate to matching sibling indices in these
double-layer contractions. The messages are linear in the parent's environment.
Therefore summing parent occurrences before propagation gives the same result
as propagating every clone occurrence separately, then grouping by origin.
Induction in parent-before-child order proves the recurrence. A repeated child
receives both messages. Diamond sharing receives messages from both parents.

Let S be a k-dimensional subspace and let its orthonormal basis columns be v_i.
For each occurrence p separately, insert S only at that cut, leaving all other
cuts unchanged. Canonicality gives

```text
sum_p ||T - T_(only cut p restricted to S)||_F²
    = trace(H_v) - sum_{i=1}^k v_iᵀ H_v v_i.
```

Consequently the leading eigenspace of H_v minimizes this **sum of independent
single-cut losses**. This is the precise common-subspace objective. The same
statement applies to a fixed representation with independent physical leaves,
not automatically to any distribution of diagonal polynomial inputs.

In a symmetric cloned chain, every occurrence at one layer has the same
environment. Hence `H_layer = occurrence_count × E_paper_layer`. Eigenvectors
agree, but eigenvalues and absolute tail energies contain the multiplicity.
At a repeated eigenvalue, a selected direction is nonunique. Comparisons should
use separated eigenvalues or whole tied subspaces, not arbitrary column signs.

## What does not follow

### A common basis need not diagonalize each occurrence

Use identity embedding, two separate identity branches sharing that embedding,
and a scalar binary merge `A = [[2,1],[0,1]]`. Explicit occurrence contractions
give `E_left = [[5,1],[1,1]]` and `E_right = [[4,2],[2,2]]`.
Their sum has a well-defined leading direction, but its eigenbasis leaves
nonzero, opposite off-diagonal entries in the two individual environments.
Only their sum is diagonalized.

### Tied interventions have cross-occurrence terms

For occurrence derivatives J_p, the simultaneous tied derivative is `sum_p J_p`.
Its squared norm contains `2 sum_{p<q} <J_p,J_q>`, absent from the independent
objective. These terms can cancel exactly.

In the same diamond, let `A = [[0,1],[-1,0]]` and perturb the shared embedding by
`D = diag(1,-1)`. The two ordered-coefficient derivatives are `Dᵀ A` and `A D`.
Each has squared norm 2, but they sum to zero. The independent total is 4, the
cross contribution is -4, and the tied first-order effect is zero. The finite
ordered tensor is `(1-t²) A`, so a first-order argument does not determine the
finite effect either.

### Finite tied truncation need not be minimized by this ranking

This failure also occurs with **symmetric, genuinely repeated children**. Take
identity embedding and two root-output slices

```text
T1 = [[2,0],[0,0]],     T2 = [[0,2],[2,0]].
```

The aggregate environment is `diag(16,8)`, whose leading direction is e1. A
common rank-one vector with `u = v1²` retains ordered coefficient energy
`4u² + 16u(1-u)`. The spectral choice u=1 retains 4, whereas u=2/3 retains 16/3.
Corresponding finite tied squared errors are 8 and 20/3. Thus the aggregate
single-cut optimum is not even the finite tied-loss optimum in this small case.
This is not a failure of exact ODT. Nor does Dooms's tied-tree procedure promise
the globally optimal simultaneous rank-constrained approximation. It is a
counterexample to attributing that stronger objective to occurrence aggregation.

### The object is not polynomial-intrinsic

The antisymmetric diamond above evaluates to zero for every tied physical x.
A zero merge core computes the identical zero polynomial. Nevertheless their
fixed-lift occurrence sums at the embedding are `2 I` and zero, respectively.
Local symmetry of genuinely identical child IDs does not remove this example,
because the merge has two distinct identity-branch nodes.

For a fixed Padé numerator/denominator lift, multiplying their joint output head
by a nonzero scalar c leaves the decoded ratio unchanged but multiplies every
environment by c². A common nonconstant polynomial factor can change the tensor
representation more substantially, while preserving the ratio where that factor
is nonzero. This reference fixes the representation. It claims neither quotient
invariance nor representation-independent absolute eigenvalues.

No simultaneous multibond error certificate, finite tied-intervention metric,
distributional action guarantee, or quotient-intrinsic interpretation is claimed
from H_v alone.

## Full-rank gauges and implementation boundary

Contract each full eigenbasis adjoint into its producer and the basis into every
parent slot, using both slots for repeated edges. Adjacent contractions cancel,
so the entire fixed ordered tensor is preserved. Recomputed aggregate
environments transform into that common basis. Physical lower-rank contraction
uses the same every-occurrence mechanics, but changes the represented tensor.

The kernel is `shared_dag.py`. The separately owned `clone_oracle.py` copies every
occurrence, independently performs direct QR with its own symmetric-coordinate
packing and input-factor updates, and recursively contracts complete coefficient
tensors. Its environment oracle directly evaluates two downstream copies inside
explicit external-index sums. It does not build an opened-context array and
multiply that array by itself, and assumes no sibling isometry cancellation.

`python3 -m research.odt_reference.run_tests --suite dag` passes 18 tests.
Repeated-edge and diamond fixtures compare complete tensors and independently
contracted occurrence environments and resolved retained subspaces after **each**
canonical step, including a deficient fixture. Full gauges and physical cuts
also match independently applied clone transformations after every prefix. Further tests
cover singular/zero completions, symmetry-before-reduction, cross-modal rejection,
chain multiplicity, full gauges, the exact independent-loss identity, and all
counterexamples above. Everything is bounded NumPy/unittest, with no production
imports, streaming, serialization, cluster runs, or changes to the paper.
