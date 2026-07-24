"""K-candidate multi-hypothesis action head — a tensor-pure mixture of Diracs.

The χ-VLA linear+MSE action head emits only the conditional MEAN E[a | context].
For a bimodal target (LIBERO gripper ∈ {-1,+1}) the mean is ≈0 → the gripper never
commits → 0% closed-loop. This head fixes that WITHOUT leaving the tensor-pure regime.

Construction (per action bond feature ``h`` = post-attention action-query rep):

    a_k = D_k[(L_k h̄) ⊙ (R_k h̄)]          k = 1..K   candidate action  ∈ R^{d_a}
    s_k = w_k[(l_k h̄) ⊙ (r_k h̄)]          k = 1..K   candidate score    ∈ R
    h̄ = [1; h]

Each ``a_k`` / ``s_k`` is a BilinearFFN, i.e. a CP factorization of a symmetric
degree-2 core (``BilinearFFN.dense_core`` → T^{(k)} ∈ R^{d_a×(d+1)×(d+1)}). So the
forward map context → {a_k, s_k} is polynomial (degree 2 over the bond) and folds
into the tensor network exactly like the existing linear head — K+K parallel cores
instead of one. NOTHING in the deployed forward graph is a softmax/argmax/min.

The implied conditional is a K-atom mixture of Diracs:
    p(a | h) = Σ_k π_k(h) δ(a − a_k(h)).
Multimodality lives in the K candidates. The DECODE picks a winner by an
OUT-OF-GRAPH argmax over the emitted scores (``decode`` below) — the same category
of controller op as the existing action/safety boundary (spec §11). Soft weighting
(π = normalize(s); â = Σ π_k a_k) would re-average the modes back toward the mean —
that is exactly the bug we are escaping — so decode is winner-take-all, not a
soft expectation.

Training uses Multiple-Choice Learning (Guzman-Rivera 2012; ε-relaxed / Confident
MCL, Lee 2016/2017): the min-over-K assignment and the decode argmax are BOTH
loss-/controller-only ops that never enter the deployed graph. See ``mcl_loss``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xvla.nn.bilinear import BilinearFFN


class MultiHypothesisHead(nn.Module):
    """K independent bilinear (CP) candidate heads + K bilinear score heads.

    Each candidate/score is its OWN degree-2 core (not a shared-feature readout), so
    each folds to its own tensor factor T^{(k)} / S^{(k)} and ODT can analyze one
    MODE at a time (per-mode mechanisms). ``candidate_cores()`` / ``score_cores()``
    export the folded cores for the odt.py / odt_interp.py machinery.

    Args:
        dim: bond width d (e.g. VLAConfig.dim, the post-attention feature).
        action_dim: candidate output width d_a (e.g. 1 for a gripper-only head,
            7 for a full action, or H*d_a for a whole chunk — see note in the report
            on per-chunk vs per-step candidates: per-chunk avoids mode flipping).
        K: number of hypotheses (2 for the bimodal gripper; 4–8 for the arm).
        rank: CP rank of each core (defaults to BilinearFFN's 3*dim).
        spread_init: tiny per-candidate perturbation of the homogeneous (bias)
            channel so the K candidates start in different basins — breaks the MCL
            symmetry that causes mode collapse. Stays exactly foldable (biases are
            part of x̄, so dense_core includes them).
    """

    def __init__(
        self,
        dim: int,
        action_dim: int,
        K: int,
        rank: int | None = None,
        spread_init: float = 0.02,
    ):
        super().__init__()
        self.K = K
        self.action_dim = action_dim
        self.cands = nn.ModuleList(
            [BilinearFFN(dim, rank=rank, out_dim=action_dim, down_bias=False) for _ in range(K)]
        )
        self.scores = nn.ModuleList(
            [BilinearFFN(dim, rank=rank, out_dim=1, down_bias=False) for _ in range(K)]
        )
        if spread_init:
            with torch.no_grad():
                for k, c in enumerate(self.cands):
                    # distinct constant channel per candidate → distinct init output
                    c.left.bias.add_(spread_init * (k - (K - 1) / 2))

    def forward(self, h: torch.Tensor):
        """h: (..., dim) → (candidates (..., K, d_a), scores (..., K))."""
        a = torch.stack([c(h) for c in self.cands], dim=-2)          # (..., K, d_a)
        s = torch.cat([w(h) for w in self.scores], dim=-1)           # (..., K)
        return a, s

    @torch.no_grad()
    def candidate_cores(self):
        """Folded symmetric cores T^{(k)} (d_a, d+1, d+1) — one tensor factor / mode."""
        return [c.dense_core() for c in self.cands]

    @torch.no_grad()
    def score_cores(self):
        """Folded symmetric score cores S^{(k)} (1, d+1, d+1) — the per-mode gate."""
        return [w.dense_core() for w in self.scores]


def mcl_loss(
    cands: torch.Tensor,
    scores: torch.Tensor,
    target: torch.Tensor,
    eps: float = 0.05,
    score_weight: float = 1.0,
    repulsion: float = 0.0,
    margin: float = 1.0,
):
    """ε-relaxed Multiple-Choice Learning loss + softmax-free score regression.

    All selection here (argmin winner) is LOSS-ONLY: it uses ``target`` which exists
    only at training time. The deployed forward map never contains it.

    Args:
        cands:  (B, K, d_a) candidate actions.
        scores: (B, K) candidate scores.
        target: (B, d_a) ground-truth action.
        eps:    ε-MCL floor — the winner gets weight (1-ε), each loser ε/(K-1), so
                losers keep receiving gradient and do not starve (mode-collapse
                mitigation, Lee 2016). eps=0 → hard winner-take-all (Guzman-Rivera).
        score_weight: weight on the score-regression term.
        repulsion: coefficient of an exp-free pairwise repulsion that pushes
                candidates apart (coverage; mode-collapse mitigation). 0 = off.
        margin: repulsion margin (target min pairwise sq-distance).

    Returns: (loss, kstar) where kstar (B,) is the winning index (for logging).
    """
    B, K, d_a = cands.shape
    d = ((cands - target[:, None, :]) ** 2).mean(-1)                 # (B, K) per-cand MSE
    kstar = d.argmin(1)                                              # winner (loss-only)

    # ε-relaxed assignment weights (all in the loss, never in the graph).
    w = torch.full_like(d, eps / max(K - 1, 1))
    w.scatter_(1, kstar[:, None], 1.0 - eps)
    regress = (w * d).sum(1).mean()

    # Score = negative distance (softmax-free confidence). Detach candidates so the
    # score head only learns to RANK; candidates specialize via the MCL term above.
    # argmax_k s_k ≈ argmin_k ||a_k - a*||  → decode picks the winning mode.
    with torch.no_grad():
        t = -d.detach()
    score_loss = ((scores - t) ** 2).mean()

    loss = regress + score_weight * score_loss
    if repulsion > 0.0 and K > 1:
        # mean pairwise squared distance between candidates, hinged at `margin`.
        pd = ((cands[:, :, None, :] - cands[:, None, :, :]) ** 2).mean(-1)  # (B,K,K)
        iu = torch.triu_indices(K, K, offset=1)
        pair = pd[:, iu[0], iu[1]]                                  # (B, #pairs)
        loss = loss + repulsion * torch.clamp(margin - pair, min=0.0).mean()
    return loss, kstar


@torch.no_grad()
def decode(cands: torch.Tensor, scores: torch.Tensor):
    """OUT-OF-GRAPH winner-take-all decode: â = a_{argmax_k s_k}.

    cands: (..., K, d_a), scores: (..., K) → (..., d_a). This argmax is the ONLY
    non-polynomial op and it lives in the controller, not the tensor network.
    """
    k = scores.argmax(-1)                                           # (...,)
    idx = k[..., None, None].expand(*k.shape, 1, cands.shape[-1])
    return cands.gather(-2, idx).squeeze(-2)


# --------------------------------------------------------------------------- #
# Stage-A prototype: K=2 head on a bimodal 1-D target.
# Reproduces the MSE→mean-collapse failure, then fixes it with the MCL head.
# (Local Python here is 3.14 w/o torch — run on Modal, or any torch env.)
# --------------------------------------------------------------------------- #
def _demo_bimodal_1d(steps: int = 2000, device: str = "cpu"):
    import torch.nn.functional as F

    torch.manual_seed(0)
    dim = 16

    def batch(bs=512):
        # context c ∈ {0,1}; c=0 → target mostly -1, c=1 → target mostly +1, but
        # each mode is bimodal so a plain regressor is forced toward the mean ≈ 0.
        c = torch.randint(0, 2, (bs,), device=device)
        h = F.one_hot(c, 2).float() @ torch.randn(2, dim, device=device)  # feature
        p_plus = torch.where(c == 1, 0.7, 0.3)                            # P(+1)
        y = torch.where(torch.rand(bs, device=device) < p_plus, 1.0, -1.0)
        return h, y[:, None], p_plus

    # --- baseline: linear + MSE (current χ-VLA head) → collapses to 2p-1 ≈ ±0.4
    lin = nn.Linear(dim, 1).to(device)
    opt = torch.optim.Adam(lin.parameters(), 1e-2)
    for _ in range(steps):
        h, y, _ = batch()
        loss = F.mse_loss(lin(h), y)
        opt.zero_grad(); loss.backward(); opt.step()
    h, y, pplus = batch(4096)
    base_pred = lin(h)
    base_commit = (base_pred.sign() == y.sign()).float().mean().item()

    # --- MCL K=2 multi-hypothesis head → candidates → {+1, -1}, argmax commits
    head = MultiHypothesisHead(dim, action_dim=1, K=2).to(device)
    opt = torch.optim.Adam(head.parameters(), 1e-2)
    for _ in range(steps):
        h, y, _ = batch()
        a, s = head(h)
        loss, _ = mcl_loss(a, s, y, eps=0.05, repulsion=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
    h, y, pplus = batch(4096)
    a, s = head(h)
    dec = decode(a, s)
    mcl_commit = (dec.sign() == y.sign()).float().mean().item()
    cand_means = a.mean(0).squeeze(-1).tolist()

    print(f"baseline |pred| mean = {base_pred.abs().mean():.3f} "
          f"(collapses to ~|2p-1|); commit acc = {base_commit:.3f}")
    print(f"MCL candidates ≈ {['%.2f' % v for v in cand_means]} "
          f"(should straddle ±1); commit acc = {mcl_commit:.3f} "
          f"(≈ max(p,1-p) Bayes ceiling)")


if __name__ == "__main__":
    _demo_bimodal_1d()
