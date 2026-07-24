"""Product-routing mixture-of-experts action head — combinatorial modes, linear params.

Generalizes the flat K-candidate head (multihypothesis.py) structurally. Instead of K
flat siblings that cap at K modes and mode-collapse under MCL, use G INDEPENDENT binary
routing factors over tensor-pure experts:

    b_g = 1[ g_g(h) > 0 ]                            g = 1..G   (out-of-graph sign)
    â(h) = c_0(h) + Σ_{g=1}^{G} (2 b_g - 1) · c_g(h)            (Eq. P)

The 2^G reachable chunk means are the vertices of a G-dim zonotope centered at c_0(h)
with edge vectors ±c_g(h) — literally a rank-G CP structure over the action manifold.
Each gate g_g and factor expert c_g is a BilinearFFN (a CP degree-2 core, bilinear.py),
so the forward map h → {gates, experts} is polynomial and folds exactly. The routing
signs are the ONLY non-polynomial op and they live in the controller (spec §11), same
category as the gripper threshold and phase argmax.

Why this fits the ARM: its multimodality is combinatorial and factored —
which-object × which-grasp × which-side — so 2^G behaviors emerge from O(G) experts,
and each factor is a *balanced binary* problem → no factor can starve (collapse-free by
construction, unlike flat-K / MCL). Whole-chunk routing (m = H·d_a) avoids per-step
mode-flipping. The gripper bimodality is the degenerate G=1 case.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from xvla.nn.bilinear import BilinearFFN


class ProductRoutingHead(nn.Module):
    """G binary gate cores + (G+1) additive factor-expert cores (Eq. P).

    Args:
        dim: bond width d (= VLAConfig.dim, pooled post-attention feature).
        action_dim: per-step action width d_a.
        horizon: chunk length H. Models the whole chunk a ∈ R^{H·d_a}.
        n_factors: G routing factors → 2^G modes.
        rank: CP rank of each core.
    """

    def __init__(self, dim: int, action_dim: int, horizon: int, n_factors: int = 3,
                 rank: int | None = None):
        super().__init__()
        self.dim = dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.m = horizon * action_dim
        self.G = n_factors
        self.center = BilinearFFN(dim, rank=rank, out_dim=self.m, down_bias=False)
        self.factors = nn.ModuleList(
            [BilinearFFN(dim, rank=rank, out_dim=self.m, down_bias=False) for _ in range(n_factors)]
        )
        self.gates = nn.ModuleList(
            [BilinearFFN(dim, rank=rank, out_dim=1, down_bias=False) for _ in range(n_factors)]
        )
        # break gate symmetry so factors start in different basins
        with torch.no_grad():
            for g, gate in enumerate(self.gates):
                gate.left.bias.add_(0.02 * (g - (n_factors - 1) / 2))

    def _parts(self, h: torch.Tensor):
        c0 = self.center(h)                                        # (B, m)
        cs = torch.stack([f(h) for f in self.factors], dim=1)     # (B, G, m)
        gs = torch.cat([g(h) for g in self.gates], dim=-1)        # (B, G)
        return c0, cs, gs

    def forward(self, h: torch.Tensor):
        """h: (B, dim) → (center (B,m), factors (B,G,m), gate logits (B,G))."""
        return self._parts(h)

    def action_for_signs(self, c0: torch.Tensor, cs: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        """c0:(B,m), cs:(B,G,m), signs:(B,G) in {-1,+1} → (B,m)."""
        return c0 + (signs[..., None] * cs).sum(1)

    def loss(self, h: torch.Tensor, target_chunk: torch.Tensor, gate_weight: float = 1.0,
             tau: float | None = None, teacher_mean: torch.Tensor | None = None,
             lambda_distill: float = 0.0, straight_through: bool = False,
             st_temp: float = 1.0, st_samples: int = 4):
        """Assign sign-tuple per sample (loss-only), regress experts + BCE gates.

        Precision-preserving variants (DEVLOG cont.17 fix), both LOSS-ONLY:

        (c)+(d) CURRICULUM / PRECISION-WEIGHTED assignment (``tau``): the hard argmin
            (``tau=None``) trains only the winning expert per demo → each of the 2^G
            branches sees a fraction of the data → high variance / imprecise on a
            near-unimodal target.  With ``tau`` set, use a SOFT posterior
            w = softmax(-err/tau): high τ ⇒ soft ⇒ re-averaging ⇒ fits the mean
            precisely (like MSE); anneal τ→0 over training to commit.  The winner is
            regressed with its posterior weight (precision-weighted) and the gate
            targets are the calibrated soft posterior, not a hard bit.

        (b) DISTILLATION anchor (``lambda_distill``): pull the mixture CENTER c0(h)
            toward the linear teacher mean — so on a unimodal state the mixture
            collapses onto the precise mean and the factors carry only true deviation.

        Collapse-free: each gate is a balanced binary target. h:(B,dim); (B,H,d_a).
        """
        B = target_chunk.shape[0]
        a_star = target_chunk.reshape(B, self.m)
        c0, cs, gs = self._parts(h)
        if straight_through:
            # ---- Method 5 (STC-MDN): straight-through Gumbel routing, HYBRID ----
            # Diagnosed bug: the old path assigns signs by ORACLE argmin and trains the gates
            # only by a decoupled BCE to a soft posterior, so the DEPLOYED decision sign(gs)
            # never got gradient through the actual action reconstruction — on scarce/near-
            # unimodal data p_plus≈0.5 leaves gates at a coin-flip (0.64 plateau + seed-variance
            # off-mean-vertex collapse). Fix WITHOUT collapsing the mixture: keep the oracle
            # argmin regression to PLACE the experts (c0, cs) — training experts under random
            # routing instead makes cs→0 / everything→mean (posterior collapse) — but train the
            # GATES by straight-through reconstruction through their OWN routing sign(gate+noise),
            # with the experts DETACHED so gate routing cannot pull the experts to the mean.
            # Gumbel noise + tanh relaxation + detach live ONLY here; decode() is unchanged, so
            # the deployed map c0 + Σ sign(g)·c_g still folds exactly (4e-16).
            combos = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=self.G)),
                                  device=h.device, dtype=h.dtype)                 # (2^G, G)
            preds = c0[:, None, :] + torch.einsum("cg,bgm->bcm", combos, cs)      # (B, 2^G, m)
            err = ((preds - a_star[:, None, :]) ** 2).mean(-1)                    # (B, 2^G)
            best = err.argmin(1)
            regress = err.gather(1, best[:, None]).mean()                         # trains c0, cs
            c0d, csd = c0.detach(), cs.detach()                                   # gates see fixed experts
            S = st_samples if self.training else 1
            gate_recon = 0.0
            for _ in range(S):
                gl = gs
                if self.training:
                    u = torch.rand_like(gs).clamp_(1e-6, 1 - 1e-6)
                    gl = gs + torch.log(u) - torch.log1p(-u)        # logistic (binary-concrete) noise
                b_soft = torch.tanh(gl / st_temp)                    # (B,G) relaxed in (-1,1)
                b_hard = torch.where(gl > 0, 1.0, -1.0).to(h.dtype)  # sampled hard routing
                b_st = b_hard + (b_soft - b_soft.detach())           # value=hard, grad=soft (ST)
                a_gate = c0d + (b_st[..., None] * csd).sum(1)        # (B, m), grad → gates only
                gate_recon = gate_recon + ((a_gate - a_star) ** 2).mean()
            loss = regress + gate_weight * (gate_recon / S)
            if lambda_distill > 0.0 and teacher_mean is not None:
                loss = loss + lambda_distill * ((c0 - teacher_mean.reshape(B, self.m)) ** 2).mean()
            return loss, best
        combos = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=self.G)),
                              device=h.device, dtype=h.dtype)              # (2^G, G)
        # predicted action for every combo: (B, 2^G, m)
        preds = c0[:, None, :] + torch.einsum("cg,bgm->bcm", combos, cs)
        err = ((preds - a_star[:, None, :]) ** 2).mean(-1)                # (B, 2^G)
        best = err.argmin(1)                                              # (B,) loss-only
        if tau is None:
            regress = err.gather(1, best[:, None]).mean()
            w = F.one_hot(best, combos.shape[0]).to(h.dtype)             # (B, 2^G)
        else:
            w = F.softmax(-err / tau, dim=1)                            # (B, 2^G) posterior
            regress = (w * err).sum(1).mean()
        # gate targets = posterior prob each factor is +.  (Hard tau → the winning bit.)
        sign_pos = (combos > 0).to(h.dtype)                             # (2^G, G)
        p_plus = w @ sign_pos                                            # (B, G) in [0,1]
        gate_loss = F.binary_cross_entropy_with_logits(gs, p_plus)
        loss = regress + gate_weight * gate_loss
        if lambda_distill > 0.0 and teacher_mean is not None:
            loss = loss + lambda_distill * ((c0 - teacher_mean.reshape(B, self.m)) ** 2).mean()
        return loss, best

    @torch.no_grad()
    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """OUT-OF-GRAPH greedy routing: b_g = sign(g_g(h)), then Eq. P. (B,dim)→(B,H,d_a)."""
        c0, cs, gs = self._parts(h)
        signs = torch.where(gs > 0, 1.0, -1.0).to(h.dtype)              # (B, G)
        a = self.action_for_signs(c0, cs, signs)                        # (B, m)
        return a.reshape(-1, self.horizon, self.action_dim)

    @torch.no_grad()
    def enumerate_modes(self, h: torch.Tensor) -> torch.Tensor:
        """All 2^G mode chunks (for coverage analysis). (B,dim)→(B, 2^G, H, d_a)."""
        c0, cs, _ = self._parts(h)
        combos = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=self.G)),
                              device=h.device, dtype=h.dtype)
        preds = c0[:, None, :] + torch.einsum("cg,bgm->bcm", combos, cs)
        return preds.reshape(h.shape[0], -1, self.horizon, self.action_dim)
