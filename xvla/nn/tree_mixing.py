"""Separator-friendly token mixers — the alternatives to bilinear attention.

Bilinear attention admits no narrow ODT separator, for two independent reasons
(see ``docs`` in :mod:`xvla.train.canonical_odt` for the Gram machinery):

1. **Environment inequivalence.** In ``y_i = c_i Σ_j m_ij (q1_i·k1_j)(q2_i·k2_j) v_j``
   the token ``x_i`` occurs on the query side *and* inside the shared source
   accumulator ``S = Σ_j k̃_j v_jᵀ``. Those two occurrences see different
   environments, so no single projector is correct for both, and the Gram at a
   per-token bond is not well defined. ODT tolerates *cloning* (Dooms' χ-MLP is a
   chain with tied legs) but not asymmetric multi-occurrence.
2. **Width.** The minimum valid separator is then the whole layer state, ``N·d``.

Both modules here fix (1) by construction: every occurrence of a token bond has
one environment. Neither fixes (2) on its own, because a *residual stream* routes
the full ``N·d`` state around any bottleneck. Narrowness is a property of the
surrounding architecture (a strict funnel), not of the mixer, so these are the
mixing primitives for a funnel/tree model rather than drop-in ODT enablers.

They are written to the :class:`~xvla.nn.attention.BilinearAttention` call
signature so they slot into :class:`~xvla.nn.block.ChiTransformerBlock`
unchanged, which is what makes an apples-to-apples capability comparison
possible.

``FixedTokenMix``
    Learned, *input-independent* token mixing (the MLP-Mixer token-mixing stage).
    Isolates the contribution of input-dependence: it keeps all-to-all reach and
    removes content-based routing entirely. Note the separator consequence, which
    is the point of the arm: a rank-``r`` mixer has separator width ``r·d``, so a
    *generic dense* mixer is exact but intractable, while a low-rank or
    hierarchically structured one reassociates into narrow bonds.

``ButterflyBilinearMix``
    Fixed-routing hierarchical bilinear merging, i.e. the tree design's merge
    primitive in drop-in shape. ``ceil(log2 N)`` stages; at stage ``s`` token
    ``p`` merges with token ``p - 2**s`` (causal parallel-prefix) or ``p ^ 2**s``
    (bidirectional butterfly). Each merge is one arity-2 core of bond width
    ``width`` in the cross-bilinear form of
    :class:`~xvla.nn.projector.CrossBilinearProjector`,

        ``y_p = D_s (L x_p) + D_m (R x_q) + D [(L x_p) ⊙ (R x_q)]``

    which is exactly one typed homogeneous core (the two linear paths are the
    ``i=0`` and ``j=0`` slices of that core, and are *why* no separate residual
    connection is needed inside the merge). Reaches every token in ``log2 N``
    depth. Routing is fixed by position, so interaction is multiplicative but
    not content-addressed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _shift_prefix(x: torch.Tensor, offset: int) -> torch.Tensor:
    """Zero-padded causal shift: ``out[:, p] = x[:, p - offset]``, 0 for ``p < offset``.

    Zero padding (rather than clamping to index 0) keeps the merge honest: a
    token with no partner at this stage contributes only through the block's own
    residual path, instead of being silently squared against itself.
    """
    if offset <= 0:
        return x
    n = x.shape[1]
    if offset >= n:
        return torch.zeros_like(x)
    pad = x.new_zeros(x.shape[0], offset, x.shape[2])
    return torch.cat([pad, x[:, : n - offset]], dim=1)


def _shift_butterfly(x: torch.Tensor, offset: int) -> torch.Tensor:
    """Bidirectional butterfly partner: ``out[:, p] = x[:, p ^ offset]``.

    Indices whose XOR partner falls outside ``[0, n)`` get a ZERO partner, not
    themselves. Keeping themselves would make the merge compute
    ``up((L x_p) * (R x_p))``, a squared self-term, which is exactly the artifact
    :func:`_shift_prefix` is written to avoid. At n=87 the self-partner version
    would hit 1, 1, 1, 7, 9, 23 and 41 tokens at stages 0 to 6.
    """
    n = x.shape[1]
    idx = torch.arange(n, device=x.device)
    partner = idx ^ offset
    valid = partner < n
    gathered = x[:, torch.where(valid, partner, idx)]
    return gathered * valid.to(gathered.dtype).view(1, n, 1)


class FixedTokenMix(nn.Module):
    """Input-independent learned token mixing (MLP-Mixer token-mixing stage).

    ``y[b, p, :] = W_O Σ_q M[p, q] x[b, q, :]`` with ``M`` a learned constant.
    One input leg, no cloning, degree unchanged. All-to-all reach with zero
    content-based routing.

    Args:
        dim: model width ``d``.
        max_tokens: token budget ``N`` the mixing matrix is allocated for.
        causal: if True, ``M`` is lower-triangular so token ``p`` reads only ``q <= p``.
    """

    def __init__(self, dim: int, max_tokens: int, causal: bool = False):
        super().__init__()
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        self.dim = dim
        self.max_tokens = max_tokens
        self.causal = causal
        self.mix = nn.Parameter(torch.empty(max_tokens, max_tokens))
        nn.init.normal_(self.mix, std=max_tokens ** -0.5)
        self.wo = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.wo.weight, std=dim ** -0.5)
        # Row-count normalisation, the analogue of BilinearAttention's C_ii = 1/sqrt(n_i).
        # Without it a causal tril row-sum grows with position and token 86 has a ~27x
        # hotter branch than token 0, which is a position-dependent scale artifact rather
        # than anything about mixing.
        self.register_buffer("row_norm", torch.tensor(True), persistent=False)
        # Post-hoc branch scale, set by `match_branch_scale` so arms are comparable.
        self.register_buffer("out_scale", torch.ones(()))

    def forward(self, x, mask=None, method: str = "explicit"):
        """``method`` is accepted and ignored (there is only one execution path)."""
        n = x.shape[1]
        if n > self.max_tokens:
            raise ValueError(f"got {n} tokens, mixer allocated for {self.max_tokens}")
        m = self.mix[:n, :n]
        if self.causal:
            m = torch.tril(m)
        elif mask is not None:
            m = m * mask[:n, :n].to(m.dtype)
        if bool(self.row_norm):
            counts = m.ne(0).sum(dim=1, keepdim=True).clamp_min(1).to(m.dtype)
            m = m / counts.sqrt()
        y = self.wo(torch.einsum("pq,bqd->bpd", m, x))
        return self.out_scale.to(y.dtype) * y


class ButterflyBilinearMix(nn.Module):
    """Fixed-routing hierarchical bilinear token merging (the tree merge primitive).

    ``ceil(log2 N)`` stages of arity-2 cross-bilinear merges.

    **THIS IS NOT A TREE AND IT IS NOT ODT-SEPARABLE. Both claims previously made
    here were false and are retracted.**

    Measured on this module: every slot has fan-out exactly 2 (``h[q]`` feeds
    ``h'[q]`` through ``left``/``up_self``/``up`` and ``h'[q ^ 2**s]`` through
    ``right``/``up_partner``), and those two occurrences have **different maps and
    different environments**. That is precisely the asymmetric multi-occurrence
    condition that makes attention non-separable, and it is what
    ``canonical_odt._tied_leg_gram`` raises on. It is an FFT butterfly, not a merge
    tree: a merge tree REDUCES the token count, this preserves it.

    Consequently the separator is the whole token state ``N*d`` (41,089 at N=107,
    d=384, so ODT cost about 2.85e18), not ``width``. Measured per-stage Jacobian
    ranks confirm ``width`` is not a bond: the SELF path has rank ``d`` (a full-rank
    channel per slot, via the ``h + ...`` identity plus ``up_self``) and only the
    PARTNER path has rank ``width``. A genuine merge node emits ``width`` dimensions
    total, with BOTH children compressed through it.

    This module is therefore a capability probe for fixed-routing bilinear mixing.
    It is not a decomposable object and no ODT claim may be made with it.

    Args:
        dim: model width ``d``.
        max_tokens: token budget ``N``, sets the stage count ``ceil(log2 N)``.
        width: rank of the PARTNER path only. NOT a bond width, and NOT the quantity
            any ODT rank claim is about (see the class docstring). The self path stays
            full rank ``d`` regardless.
        causal: if True use the parallel-prefix pattern (partner ``p - 2**s``),
            else the XOR butterfly (partner ``p ^ 2**s``).
    """

    def __init__(self, dim: int, max_tokens: int, width: int = 64, causal: bool = False):
        super().__init__()
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        self.dim = dim
        self.max_tokens = max_tokens
        self.width = width
        self.causal = causal
        self.n_stages = max(1, math.ceil(math.log2(max_tokens)))
        # Per stage: two input maps into the bond (self leg, partner leg) and three
        # readouts (bilinear term, and the two linear slices of the same core).
        self.left = nn.ModuleList(nn.Linear(dim, width, bias=False) for _ in range(self.n_stages))
        self.right = nn.ModuleList(nn.Linear(dim, width, bias=False) for _ in range(self.n_stages))
        self.up = nn.ModuleList(nn.Linear(width, dim, bias=False) for _ in range(self.n_stages))
        self.up_self = nn.ModuleList(nn.Linear(width, dim, bias=False) for _ in range(self.n_stages))
        self.up_partner = nn.ModuleList(nn.Linear(width, dim, bias=False) for _ in range(self.n_stages))
        # 1/sqrt(2*stages) matches the block residual-gain convention (spec §10) so
        # depth does not change the initial scale of the mixing branch.
        gain = (2.0 * self.n_stages) ** -0.5
        self.register_buffer("stage_gain", torch.tensor(gain))
        # Init so one stage has branch RMS ~1 on unit-RMS input: `left`/`right` at
        # dim^-0.5 give unit-RMS `ell`,`r`; `up*` at width^-0.5 give unit-RMS readouts;
        # 1/sqrt(3) undoes the three-term sum. Leaving `left`/`right` at torch's default
        # kaiming-uniform made this branch ~41x hotter than bilinear attention's, which
        # would confound "the tree cannot bind" with "the tree was hotter at init".
        for mod in list(self.left) + list(self.right):
            nn.init.normal_(mod.weight, std=dim ** -0.5)
        for mod in list(self.up) + list(self.up_self) + list(self.up_partner):
            nn.init.normal_(mod.weight, std=width ** -0.5)
        self.register_buffer("term_norm", torch.tensor(3.0 ** -0.5))
        # Post-hoc branch scale, set by `match_branch_scale` so arms are comparable.
        self.register_buffer("out_scale", torch.ones(()))

    def forward(self, x, mask=None, method: str = "explicit"):
        """``method`` is accepted and ignored (one execution path).

        The prefix pattern (``causal=True``) is causal by construction, so a
        causal mask is redundant and is ignored. The butterfly pattern is
        inherently bidirectional and *cannot* honour a mask, so supplying one
        there raises rather than being silently violated.
        """
        n = x.shape[1]
        if n > self.max_tokens:
            raise ValueError(f"got {n} tokens, mixer allocated for {self.max_tokens}")
        if mask is not None and not self.causal:
            raise ValueError(
                "ButterflyBilinearMix(causal=False) mixes across the whole sequence and "
                "cannot honour a mask. Use causal=True for the prefix pattern, or "
                "FixedTokenMix for arbitrary masked mixing."
            )
        shift = _shift_prefix if self.causal else _shift_butterfly
        h = x
        out = torch.zeros_like(x)
        for s in range(self.n_stages):
            partner = shift(h, 1 << s)
            ell = self.left[s](h)
            r = self.right[s](partner)
            merged = self.term_norm.to(x.dtype) * (
                self.up[s](ell * r) + self.up_self[s](ell) + self.up_partner[s](r)
            )
            h = h + self.stage_gain * merged
            out = out + self.stage_gain * merged
        return self.out_scale.to(out.dtype) * out


@torch.no_grad()
def measure_branch_scales(model, batch_args, mixer_types=None, prefix: str = "") -> dict[str, float]:
    """RMS of each token-mixing branch's output, per block, on one batch.

    Mixers initialised by different conventions can differ in branch magnitude by
    more than an order of magnitude, which at a shared learning rate makes each arm
    a different optimisation problem. Comparing arms without checking this confounds
    "this mixer cannot learn the task" with "this mixer's branch was hotter at
    init", so any arm comparison must report these numbers.
    """
    from xvla.nn.attention import BilinearAttention
    from xvla.nn.baselines import SoftmaxAttention

    if mixer_types is None:
        mixer_types = (BilinearAttention, SoftmaxAttention, FixedTokenMix, ButterflyBilinearMix)
    stats: dict[str, list[float]] = {}
    handles = []

    def make_hook(name):
        def hook(_m, _args, out):
            stats.setdefault(name, []).append(float(out.float().pow(2).mean().sqrt()))
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, mixer_types) and name.startswith(prefix):
            handles.append(mod.register_forward_hook(make_hook(name)))
    was_training = model.training
    model.eval()
    try:
        model(*batch_args)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()
    return {k: sum(v) / len(v) for k, v in stats.items() if v}


@torch.no_grad()
def match_branch_scale(model, batch_args, target_rms: float, only_new: bool = True,
                       prefix: str = "") -> dict:
    """Rescale every settable mixer branch so its output RMS matches ``target_rms``.

    ``target_rms`` should come from :func:`measure_branch_scales` on the reference
    arm, computed over the SAME ``prefix``. Averaging over a mixed population is a
    trap: if the vision encoder is pinned to one mixer while only the backbone varies,
    a whole-model mean cannot be driven to the target because the pinned modules hold
    it away from it. Only :class:`FixedTokenMix` and :class:`ButterflyBilinearMix` carry an
    ``out_scale``, so with ``only_new=True`` (the default) the reference
    architecture is left untouched and the comparison arms are brought to it.

    Returns the applied scales and the achieved RMS, both of which belong in the
    experiment's output so the matching is auditable rather than assumed.
    """
    settable = [(n, m) for n, m in model.named_modules()
                if isinstance(m, (FixedTokenMix, ButterflyBilinearMix)) and n.startswith(prefix)]
    if not settable:
        return dict(applied={}, achieved={}, target_rms=target_rms, note="no settable mixers")
    before = measure_branch_scales(model, batch_args, prefix=prefix)
    # Iterate: the mixers sit in a residual stack, so rescaling block 0 changes block 1's
    # input and one pass does not converge (measured about 1.25x off after a single pass).
    # A few fixed-point steps bring every site to the target.
    applied = {}
    for _ in range(6):
        current = measure_branch_scales(model, batch_args, prefix=prefix)
        worst = 0.0
        for name, mod in settable:
            rms = current.get(name)
            if rms is None or rms <= 0:
                continue
            factor = target_rms / rms
            mod.out_scale.mul_(factor)
            applied[name] = float(mod.out_scale)
            worst = max(worst, abs(factor - 1.0))
        if worst < 0.02:
            break
    after = measure_branch_scales(model, batch_args, prefix=prefix)
    achieved_ratio = {k: (v / target_rms) for k, v in after.items() if target_rms > 0}
    return dict(applied=applied, achieved=after, achieved_ratio=achieved_ratio,
                before=before, target_rms=target_rms, only_new=only_new)


def mixer_param_count(model) -> int:
    """Parameters in the token-mixing branches only.

    Arms whose mixers differ several-fold in parameter count cannot support a
    clean "mechanism X is load-bearing" conclusion, so this number has to be
    reported next to any such claim.
    """
    from xvla.nn.attention import BilinearAttention
    from xvla.nn.baselines import SoftmaxAttention

    types = (BilinearAttention, SoftmaxAttention, FixedTokenMix, ButterflyBilinearMix)
    return sum(sum(p.numel() for p in m.parameters())
               for m in model.modules() if isinstance(m, types))
