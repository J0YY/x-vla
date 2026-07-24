"""Tensor-pure flow-matching (rectified-flow) action head — the multimodal fix.

The χ-VLA linear+MSE action head emits only E[a | obs]. On a multimodal target
(the LIBERO gripper ∈ {-1,+1}, and the arm chunk when two whole strategies exist —
reach-left vs reach-right) the conditional mean is an invalid *between-modes* action,
so the robot never commits → 0% closed-loop (DEVLOG cont. 12/13).

This head keeps the learned object tensor-pure and moves only the decode out-of-graph
(spec §11), exactly like the sanctioned gripper threshold / phase argmax:

  learned object  :  a POLYNOMIAL velocity field  v_θ(a, t; h)
  decode          :  an OUT-OF-GRAPH reverse ODE  a_{k+1} = a_k + Δt·v_θ(a_k, t_k; h)

Construction (per whole action chunk a ∈ R^m, m = H·d_a; conditioning bond h = pooled
post-attention action-query rep, the bond DEVLOG Finding C-v2 identified as where the
bound quantity becomes linearly decodable):

  z = [ a ; φ(t) ; h ]                       homogeneous joint vector
  x_0 = a ;   x_{l+1} = Core_l([ x_l ; φ(t) ; h ])         (l = 0..depth-1)
  v_θ = x_depth                              ∈ R^m

Each Core_l is a BilinearFFN — a CP factorization of a symmetric degree-2 core
(bilinear.py). Re-injecting a (and the polynomial time features φ(t)=[t,t²,…]) at each
layer raises the polynomial degree in a multiplicatively (≈ 2^depth), so the field can
resolve several modes. The ⊙ product natively couples the three legs: a⊗h (obs
modulates the field), a⊗φ(t) (time modulates it), a⊗a (the nonlinearity that makes the
flow multimodal). NOTHING non-polynomial is in the graph — φ(t) is a fixed polynomial
feature of the sampler's t (an input, like the homogeneous constant channel).

Training = flow-matching MSE (rectified flow):  v_θ((1-t)a0 + t·a1, t; h) ≈ a1 - a0,
a0~N(0,I), a1~data, t~U[0,1]. This is Z-free, negative-free, and has no coercivity
pathology — the target a1-a0 is *given*, so it is ordinary well-posed regression, the
key advantage over Born (needs Z) and energy models (need negatives / hit a⁴<0).

Foldable / ODT: v_θ is a chain of CP cores at any frozen (t, h); it folds via
BilinearFFN.dense_core exactly like the existing head, and its score-Jacobian ∂v/∂a
is the t-indexed analogue of the odt_interp Q_c object (a new ODT read).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xvla.nn.bilinear import BilinearFFN


def time_features(t: torch.Tensor, degree: int) -> torch.Tensor:
    """Fixed POLYNOMIAL time features φ(t) = [t, t², …, t^degree].

    Polynomial (not sinusoidal) so the map (a, t, h) → v stays tensor-pure; t is a
    sampler input, never a learned nonlinearity. t: (..., 1) → (..., degree).
    """
    return torch.cat([t ** k for k in range(1, degree + 1)], dim=-1)


class FlowMatchingActionHead(nn.Module):
    """Polynomial velocity field over the whole action chunk (tensor-pure).

    Args:
        dim: conditioning bond width d (= VLAConfig.dim; h is pooled aq_out).
        action_dim: per-step action width d_a.
        horizon: chunk length H. The head models the joint chunk a ∈ R^{H·d_a}.
        rank: CP rank of each core (defaults to BilinearFFN's 3*width).
        depth: number of stacked bilinear cores. degree in a ≈ 2^depth; depth≥2
            gives a genuinely multimodal field (depth=1 is degree-2, already
            bimodal-capable; the proto used degree-3).
        time_degree: number of polynomial time features.
    """

    def __init__(
        self,
        dim: int,
        action_dim: int,
        horizon: int,
        rank: int | None = None,
        depth: int = 2,
        time_degree: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.m = horizon * action_dim
        self.time_degree = time_degree
        ctx = time_degree + dim                 # [φ(t) ; h]
        in_dim = self.m + ctx                    # [a ; φ(t) ; h]
        cores = []
        for _ in range(depth):
            cores.append(BilinearFFN(in_dim, rank=rank, out_dim=self.m, down_bias=False))
            in_dim = self.m + ctx                # re-inject a-slot (=prev output) + ctx
        self.cores = nn.ModuleList(cores)

    def velocity(self, a: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """v_θ(a, t; h).  a: (B, m), t: (B, 1), h: (B, dim) → (B, m)."""
        phi = time_features(t, self.time_degree)          # (B, time_degree)
        ctx = torch.cat([phi, h], dim=-1)                 # (B, ctx)
        x = a
        for core in self.cores:
            x = core(torch.cat([x, ctx], dim=-1))         # (B, m)
        return x

    def forward(self, a: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.velocity(a, t, h)

    def fm_loss(
        self,
        h: torch.Tensor,
        target_chunk: torch.Tensor,
        teacher_mean: torch.Tensor | None = None,
        w_unimodal: torch.Tensor | None = None,
        lambda_recon: float = 0.0,
        lambda_distill: float = 0.0,
    ) -> torch.Tensor:
        """Precision-preserving rectified-flow loss (DEVLOG cont.17 fix).

        Base term = rectified-flow matching v_θ((1-t)a0+t·a1, t; h) ≈ a1 - a0.

        (a) RECONSTRUCTION / data-consistency anchor (``lambda_recon``): the implied
            endpoint  x1_pred = a_t + (1-t)·v  must reconstruct the demo action a1.
            This is the x1-prediction reweighting of flow-matching — it up-weights
            the field's *endpoint fidelity* (the quantity the decode actually needs)
            instead of only the instantaneous velocity, pinning per-step precision.

        (b) DISTILLATION anchor (``lambda_distill``): where the state is (near-)unimodal
            (``w_unimodal``≈1, from the linear teacher's residual), pull that same
            endpoint toward the teacher mean ``teacher_mean`` — so the flow matches the
            precise MSE mean on unimodal states and only branches where w_unimodal→0.

        Both are LOSS-ONLY: the deployed velocity field / ODE decode is unchanged, so
        the graph stays polynomial/foldable (spec §11). h:(B,dim); chunks (B,H,d_a).
        """
        B = target_chunk.shape[0]
        a1 = target_chunk.reshape(B, self.m)
        a0 = torch.randn_like(a1)
        t = torch.rand(B, 1, device=a1.device, dtype=a1.dtype)
        at = (1.0 - t) * a0 + t * a1
        v = self.velocity(at, t, h)
        loss = ((v - (a1 - a0)) ** 2).mean()

        if lambda_recon > 0.0 or (lambda_distill > 0.0 and teacher_mean is not None):
            x1_pred = at + (1.0 - t) * v                  # endpoint implied by this v
        if lambda_recon > 0.0:
            loss = loss + lambda_recon * ((x1_pred - a1) ** 2).mean()
        if lambda_distill > 0.0 and teacher_mean is not None:
            mt = teacher_mean.reshape(B, self.m)
            w = 1.0 if w_unimodal is None else w_unimodal.reshape(B, 1)
            loss = loss + lambda_distill * (w * (x1_pred - mt) ** 2).mean()
        return loss

    def _integrate(self, a: torch.Tensor, hexp: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Shared OUT-OF-GRAPH probability-flow Euler ODE (the controller loop; spec §11).
        a: (N, m) initial condition, hexp: (N, dim) → (N, m) endpoint."""
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t = torch.full((a.shape[0], 1), k * dt, device=a.device, dtype=a.dtype)
            a = a + dt * self.velocity(a, t, hexp)
        return a

    @torch.no_grad()
    def sample(self, h: torch.Tensor, n_steps: int = 10, n_samples: int = 1) -> torch.Tensor:
        """STOCHASTIC decode: a0 ~ N(0,I) per sample. (B,dim) → (B, n_samples, H, d_a).

        NOTE: a single stochastic sample injects the conditional variance tr(Σ) as
        excess control error (≈2× the mean's MSE on a near-unimodal conditional). Use
        `decode` (mode / KWTA) for committed control — see mode_decode_proto.py.
        """
        B = h.shape[0]
        hexp = h[:, None].expand(B, n_samples, self.dim).reshape(B * n_samples, self.dim)
        a = torch.randn(B * n_samples, self.m, device=h.device, dtype=h.dtype)
        a = self._integrate(a, hexp, n_steps)
        return a.reshape(B, n_samples, self.horizon, self.action_dim)

    @torch.no_grad()
    def decode(self, h: torch.Tensor, n_steps: int = 10, strategy: str = "mode",
               n_samples: int = 16) -> torch.Tensor:
        """MODE / MAP decode — the committed, non-stochastic controller op. (B,dim)→(B,H,d_a).

        strategy:
          "mode"   : integrate the ODE from the noise MEAN a0=0 (deterministic). For a
                     near-Gaussian conditional the rectified-flow map is affine and
                     Φ(0)=conditional mean=mode → matches the linear head's precision on
                     unimodal states with ZERO added variance. The minimal fix (a0=0 vs
                     randn). Caveat: on a SYMMETRIC multimodal conditional Φ(0) maps to the
                     between-modes point — use "kwta" there.
          "kwta"   : K-sample winner-take-all. Draw n_samples base points, integrate each,
                     score by the CNF log-density log N(a0) − ∫tr(∂v/∂a)dt (Hutchinson
                     divergence — the ∂v/∂a object is the t-indexed Q_c), return the ARGMAX
                     candidate. Robust on BOTH unimodal (→mode) and multimodal (commits,
                     never averages) states. Learned object unchanged; the divergence trace
                     + argmax are out-of-graph.
          "sample" : legacy single stochastic draw (a0~N). Kept for ablation only.
        """
        B = h.shape[0]
        if strategy == "sample":
            return self.sample(h, n_steps=n_steps, n_samples=1)[:, 0]
        if strategy == "mode":
            a = torch.zeros(B, self.m, device=h.device, dtype=h.dtype)
            a = self._integrate(a, h, n_steps)
            return a.reshape(B, self.horizon, self.action_dim)
        if strategy == "kwta":
            K = n_samples
            hexp = h[:, None].expand(B, K, self.dim).reshape(B * K, self.dim)
            a = torch.randn(B * K, self.m, device=h.device, dtype=h.dtype)
            logp = -0.5 * (a ** 2).sum(-1)                       # log N(a0;0,I) (+const)
            dt = 1.0 / n_steps
            for k in range(n_steps):
                t = torch.full((B * K, 1), k * dt, device=h.device, dtype=h.dtype)
                with torch.enable_grad():                        # divergence via Hutchinson
                    a_ = a.detach().requires_grad_(True)
                    v = self.velocity(a_, t, hexp)
                    eps = torch.randint(0, 2, a_.shape, device=a.device,
                                        dtype=a.dtype).mul_(2).sub_(1)   # Rademacher
                    vjp = torch.autograd.grad((v * eps).sum(), a_)[0]
                    div = (vjp * eps).sum(-1)                    # ≈ tr(∂v/∂a)
                logp = logp - dt * div.detach()
                a = a + dt * v.detach()
            logp = logp.reshape(B, K)
            best = logp.argmax(1)                                # MAP candidate per obs
            a = a.reshape(B, K, self.m)[torch.arange(B, device=h.device), best]
            return a.reshape(B, self.horizon, self.action_dim)
        raise ValueError(f"unknown flow decode strategy {strategy!r}")
