# Compact ODT reference: source provenance

Audit date: 2026-09-06. Public sources were read as text, never imported or executed.

## Authoritative algorithm source

[Dooms, Gauderis, Wiggins and Oramas, *Compositionality Unlocks Deep Interpretable Models*, arXiv:2504.02667v1](https://arxiv.org/html/2504.02667v1).

- [Section 2](https://arxiv.org/html/2504.02667v1#S2) imposes exchange symmetry of each quadratic core's two input legs. The trained input is cloned, while analysis uses the unfolded multilinear tree.
- [Section 3](https://arxiv.org/html/2504.02667v1#S3) specifies reduced RQ, downstream contraction and environment eigendecomposition, followed by bond truncation.
- [Appendix F](https://arxiv.org/html/2504.02667v1#A6) explicitly relies on core symmetry to calculate one projector per layer. Its stated scope is the weight-tied binary tree, not an arbitrary shared DAG.
- [Appendix G, Algorithms 1–3](https://arxiv.org/html/2504.02667v1#A7) is the executable reference's source. Algorithm 1 sends each factor into both inputs of the next quadratic core. Algorithm 2 is a doubled downstream-network contraction with the selected bond open. Its compact composition notation includes contraction of the sibling leg. Algorithm 3 changes both producer and consumer coordinates.

The reference here is a new transcription of published pseudocode. It is not authenticated as the author's original executable implementation.

## Original-code search

The [author's X-net page](https://tdooms.github.io/research/xnets) links a paper and video, not source code. The [authors' publication page](https://compinterp.github.io/publications/) links the ODT paper, poster and slides, but no executable repository.

The complete public repository lists for [Thomas Dooms](https://github.com/tdooms?tab=repositories) and [Ward Gauderis](https://github.com/WardGauderis?tab=repositories) were checked. No repository was authenticated as the original ODT implementation. This is a search result, not a claim that no such code exists.

Related code that must not be mislabeled as original ODT:

- [tdooms/bilinear-decomposition](https://github.com/tdooms/bilinear-decomposition) identifies itself as the official repository for the different paper *Bilinear MLPs enable weight-based mechanistic interpretability*.
- [tdooms/bilinear-interp, `image/model.py`, commit `f34176c32659667ed56105c87210d5797d81adcb`](https://github.com/tdooms/bilinear-interp/blob/f34176c32659667ed56105c87210d5797d81adcb/image/model.py): `Model.decompose` constructs a symmetric bilinear tensor, eigendecomposes it and projects directions through the embedding. Its docstring explicitly scopes it to one layer. It does not implement the three ODT sweeps.
- The public tree of [tdooms/tensor-similarity, commit `bc914a4babc943541d414b4239b1df7f032d115b`](https://github.com/tdooms/tensor-similarity/tree/bc914a4babc943541d414b4239b1df7f032d115b) was searched for ODT, orthogonalization, decomposition and X-net paths. No original ODT source was identified. This repository describes tensor-network similarity experiments.

No original 25-line executable, source path or commit has been verified. The author can resolve that provenance gap with the exact snippet or repository link.

## Explicitly new derivations, not claims about author code

These points follow from the algebra and are implemented or tested here, rather than attributed to unpublished author code:

1. Use the orthonormal symmetric-coordinate basis with diagonal entries unchanged and off-diagonal entries weighted by `sqrt(2)`. Apply direct QR to the transpose of that local unfolding. Unpacking its retained Q keeps every completion vector symmetric, including deficient cases. The retained dimension is chosen from shape, `min(output_width, input_width * (input_width + 1) // 2)`, before factorization. There is no numerical rank decision or alternate factorization.
2. Symmetry of siblings is not full permutation symmetry of all bottom-level leaf occurrences. It does not select a unique representation for every equivalent polynomial or rational function.
3. The eigenvalues of the doubled environment are squared cut singular values, by direct expansion of the contraction. Keep that distinction even where the paper's prose calls them singular values. No singular-value routine is required.
4. Summing environments of heterogeneous occurrences defines a shared-subspace objective that must be stated and derived separately. It is not automatically one actual cut environment, the exact simultaneous-truncation loss, or an intrinsic metric on the input-output function.

The production compiler, streaming implementation, clone implementation and old successful output replays are not sources for this mathematical transcription.
