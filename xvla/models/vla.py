"""χ-VLA — the full pixels+instruction+state → continuous action model (M4, spec §10).

Sequence (spec §10):
  [visual patch tokens | BOS | instruction tokens | robot-state | embodiment | action queries]
run through a causal joint χ-transformer; the action-query output states are mapped
by a linear head to a continuous action chunk (H × d_a). Every learned block is a
tensor primitive (bilinear attention + bilinear FFN + foldable norm); the vision
front-end is a χ-ViT; the action head and all embeddings are linear/affine. No
softmax/argmax/discretization in the forward graph (spec §11) — the action decode is
a direct linear regression.

Optionally fuses a second (semantic) vision branch through the cross-bilinear CP
projector (spec §8) instead of a single encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from xvla.models.vit import ChiViT, ChiConvEncoder, ViTConfig
from xvla.nn.attention import causal_mask
from xvla.nn.block import ChiTransformer
from xvla.nn.flow_action import FlowMatchingActionHead
from xvla.nn.normalization import make_norm
from xvla.nn.product_routing import ProductRoutingHead
from xvla.nn.quantile_action import AutoregQuantileHead
from xvla.nn.projector import CrossBilinearProjector


@dataclass
class VLAConfig:
    # vision
    image_size: int = 32
    patch_size: int = 4
    vit_dim: int = 192
    vit_layers: int = 4
    vit_heads: int = 6
    dual_vision: bool = False          # two branches + cross-bilinear projector
    projector: str = "crossbilinear"   # "crossbilinear" | "concat"
    vision_encoder: str = "vit"        # "vit" (ChiViT) | "conv" (ChiConvEncoder, ResNet-style
                                       #   foldable bilinear-conv stem — stronger small-data vision)
    conv_kernel: int = 3
    conv_grid: int = 8                 # conv encoder output token grid (grid*grid tokens)
    # language / state
    vocab_size: int = 128
    max_instr_len: int = 16
    state_dim: int = 8
    n_embodiments: int = 4
    # joint backbone
    dim: int = 384
    n_layers: int = 8
    n_heads: int = 12
    ffn_rank: int | None = None
    vit_ffn_rank: int | None = None
    attn: str = "bilinear"             # "bilinear" | "softmax"
    ffn: str = "bilinear"              # "bilinear" | "swiglu"
    norm: str = "per_token"
    qk_norm: str = "per_token"
    # action
    action_horizon: int = 4
    action_dim: int = 7
    # action-head type. "linear" = the original E[a|obs] MSE head (unimodal,
    # collapses on multimodal targets). "flow" = tensor-pure flow-matching velocity
    # field (rectified flow, out-of-graph ODE decode). "product" = product-routing
    # mixture-of-experts (2^G modes from O(G) tensor-pure cores). Both multimodal
    # heads keep the learned object polynomial/foldable and put only the decode
    # (ODE integration / routing signs) out-of-graph (spec §11).
    action_head: str = "linear"
    head_rank: int | None = None
    flow_depth: int = 2
    flow_time_degree: int = 3
    flow_steps: int = 10
    # flow decode strategy (out-of-graph controller op; DEVLOG cont.17). "mode" =
    # deterministic ODE from a0=0 (conditional mode/mean; matches linear precision on
    # unimodal states, no injected sampling variance — the fix). "kwta" = K-sample
    # winner-take-all by CNF log-density (robust on symmetric multimodal states).
    # "sample" = legacy single stochastic draw (the 0%-closed-loop behaviour).
    flow_decode: str = "mode"
    flow_kwta_samples: int = 16
    n_factors: int = 3
    # AQ head (autoregressive monotone-quantile): per-step conditioned (no pooling) +
    # deterministic MAP decode; a STRICT SUPERSET of the linear head (s→0 recovers it),
    # so it matches linear precision on unimodal states and commits on multimodal ones.
    quantile_degree: int = 4
    quantile_grid: int = 128
    # ---- precision-preserving anchors (DEVLOG cont.17 fix). All LOSS-ONLY; the
    # deployed graph is unchanged. Defaults 0/off → identical to prior behaviour. ----
    distill_teacher: bool = False      # train an auxiliary linear MSE head as a precision
                                       #   teacher and distill the multimodal head to it.
    teacher_weight: float = 1.0        # weight on the teacher's own MSE loss.
    lambda_recon: float = 0.0          # (a) flow endpoint-reconstruction anchor.
    lambda_distill: float = 0.0        # (b) distillation anchor (flow: unimodal-gated
                                       #   endpoint→teacher; product: center→teacher).
    unimodal_tau: float = 0.5          # residual scale for the per-sample unimodality
                                       #   weight w=exp(-½(‖a-teacher‖/τ)²) (flow only).
    curriculum: bool = False           # (c)+(d) product soft→hard assignment schedule.
    distill_tau0: float = 1.0          # curriculum start temperature (soft ≈ mean).
    distill_tau1: float = 0.02         # curriculum end temperature (hard commit).
    product_st: bool = False           # Method 5 (STC-MDN): straight-through Gumbel routing
                                       #   — gates get end-to-end recon gradient (kills the
                                       #   train=argmin/deploy=sign mismatch + seed variance).
    st_temp0: float = 1.5              # STC start temperature (soft/exploratory).
    st_temp1: float = 0.3              # STC end temperature (sharp, ≈ deployed sign).
    st_samples: int = 4                # routing samples averaged per step (grad-variance).
    # discrete latent / phase conditioning (0 = disabled).
    # A discrete z (e.g. reach/grasp/lift phase, or a VQ code) enters as ONE extra
    # token embedding — exactly the mechanism language uses — so p(a|obs,z) stays a
    # linear+MSE (tensor-pure/foldable) head. The choice of z (argmax/argmin) is an
    # OUT-OF-GRAPH controller op (spec §11): the forward graph only emits linear
    # phase LOGITS; no softmax/argmax lives in the exported tensor network.
    n_phases: int = 0

    def vit_config(self):
        return ViTConfig(image_size=self.image_size, patch_size=self.patch_size,
                         dim=self.vit_dim, n_layers=self.vit_layers, n_heads=self.vit_heads,
                         ffn_rank=self.vit_ffn_rank, norm=self.norm, qk_norm=self.qk_norm,
                         attn=self.attn, ffn=self.ffn, num_classes=1)


class ChiVLA(nn.Module):
    def __init__(self, cfg: VLAConfig):
        super().__init__()
        self.cfg = cfg

        def _make_vision():
            if cfg.vision_encoder == "conv":
                return ChiConvEncoder(cfg.vit_config(), kernel=cfg.conv_kernel, grid=cfg.conv_grid)
            if cfg.vision_encoder == "vit":
                return ChiViT(cfg.vit_config())
            raise ValueError(f"unknown vision encoder {cfg.vision_encoder!r}")

        self._make_vision = _make_vision
        self.vision = _make_vision()
        if cfg.dual_vision:
            self.vision2 = _make_vision()
            if cfg.projector == "crossbilinear":
                self.projector = CrossBilinearProjector(cfg.vit_dim, cfg.vit_dim, cfg.dim)
            else:
                self.projector = None
                self.vis_proj = nn.Linear(2 * cfg.vit_dim, cfg.dim)
        else:
            self.vis_proj = nn.Linear(cfg.vit_dim, cfg.dim)

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.bos = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.state_proj = nn.Linear(cfg.state_dim, cfg.dim)
        self.embodiment_emb = nn.Embedding(cfg.n_embodiments, cfg.dim)
        self.action_queries = nn.Parameter(torch.zeros(1, cfg.action_horizon, cfg.dim))

        # Discrete latent / phase: a query token that reads obs → linear phase logits
        # p(z|obs), and an embedding table that turns the (out-of-graph) chosen z into
        # one conditioning token appended before the action queries.
        if cfg.n_phases > 0:
            self.phase_query = nn.Parameter(torch.zeros(1, 1, cfg.dim))
            self.phase_emb = nn.Embedding(cfg.n_phases, cfg.dim)
            self.phase_head = nn.Linear(cfg.dim, cfg.n_phases)   # LINEAR logits; argmax is out-of-graph
            n_phase_tok = 2
        else:
            n_phase_tok = 0

        n_vis = cfg.conv_grid ** 2 if cfg.vision_encoder == "conv" else cfg.vit_config().num_patches
        seq_len = n_vis + 1 + cfg.max_instr_len + 1 + 1 + n_phase_tok + cfg.action_horizon
        self.seq_len = seq_len
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, cfg.dim))
        for p in (self.bos, self.action_queries, self.pos_emb):
            nn.init.normal_(p, std=cfg.dim ** -0.5)
        nn.init.normal_(self.tok_emb.weight, std=cfg.dim ** -0.5)
        nn.init.normal_(self.embodiment_emb.weight, std=cfg.dim ** -0.5)
        if cfg.n_phases > 0:
            nn.init.normal_(self.phase_query, std=cfg.dim ** -0.5)
            nn.init.normal_(self.phase_emb.weight, std=cfg.dim ** -0.5)

        self.backbone = ChiTransformer(cfg.dim, cfg.n_layers, cfg.n_heads,
                                       ffn_rank=cfg.ffn_rank, causal=True,
                                       norm=cfg.norm, qk_norm=cfg.qk_norm,
                                       attn=cfg.attn, ffn=cfg.ffn)
        self.norm_out = make_norm(cfg.norm)
        if cfg.action_head == "flow":
            self.flow_head = FlowMatchingActionHead(
                cfg.dim, cfg.action_dim, cfg.action_horizon, rank=cfg.head_rank,
                depth=cfg.flow_depth, time_degree=cfg.flow_time_degree)
        elif cfg.action_head == "product":
            self.product_head = ProductRoutingHead(
                cfg.dim, cfg.action_dim, cfg.action_horizon, n_factors=cfg.n_factors,
                rank=cfg.head_rank)
        elif cfg.action_head == "quantile":
            self.quantile_head = AutoregQuantileHead(
                cfg.dim, cfg.action_dim, cfg.action_horizon, degree=cfg.quantile_degree,
                rank=cfg.head_rank, grid=cfg.quantile_grid)
        else:
            self.action_head = nn.Linear(cfg.dim, cfg.action_dim)

        # Auxiliary linear MSE teacher (foldable) for the distillation anchor. It is a
        # precision prior for the multimodal head on (near-)unimodal states; only used
        # in the loss (its output is not the deployed action unless action_head=linear).
        if cfg.distill_teacher and cfg.action_head in ("flow", "product"):
            self.teacher_head = nn.Linear(cfg.dim, cfg.action_dim)

    def _visual_tokens(self, img, img2=None):
        s = self.vision.features(img)                     # (B, Nv, vit_dim)
        if not self.cfg.dual_vision:
            return self.vis_proj(s)
        m = self.vision2.features(img2 if img2 is not None else img)
        if self.projector is not None:
            return self.projector(s, m)
        return self.vis_proj(torch.cat([s, m], dim=-1))

    def forward(self, img, instr_ids, state, embodiment_id, img2=None,
                target_actions=None, phase_id=None, phase_labels=None,
                phase_weight=1.0, return_phase=False, progress=0.0):
        """Latent-variable action head:  p(a|obs) = Σ_z p(z|obs) p(a|obs,z).

        - `phase_head` emits LINEAR logits for p(z|obs) (read from a causal query
          token that sees only obs). The choice ẑ = argmax_z p(z|obs) is done
          OUT-OF-GRAPH by the caller (a controller op, spec §11), then passed back
          in as `phase_id` — nothing nonlinear enters the exported graph.
        - `phase_id` (the chosen/true z) is embedded into ONE conditioning token
          the action queries attend to, so p(a|obs,z) is a linear+MSE head that only
          has to fit the (near-)unimodal conditional mean.
        Training: teacher-force `phase_id`=`phase_labels` (from the demo gripper
        signal), add cross-entropy on the phase logits (softmax lives in the LOSS
        only, never the forward graph)."""
        B = img.shape[0]
        vis = self._visual_tokens(img, img2)              # (B, Nv, d)
        bos = self.bos.expand(B, -1, -1)
        instr = self.tok_emb(instr_ids)                   # (B, T, d)
        st = self.state_proj(state)[:, None]              # (B, 1, d)
        emb = self.embodiment_emb(embodiment_id)[:, None] # (B, 1, d)
        aq = self.action_queries.expand(B, -1, -1)        # (B, H, d)

        toks = [vis, bos, instr, st, emb]
        if self.cfg.n_phases > 0:
            toks.append(self.phase_query.expand(B, -1, -1))     # reads obs → p(z|obs)
            if phase_id is None:
                phase_id = torch.zeros(B, dtype=torch.long, device=img.device)
            toks.append(self.phase_emb(phase_id)[:, None])      # (B, 1, d) conditioning z
        toks.append(aq)
        x = torch.cat(toks, dim=1)
        x = x + self.pos_emb[:, : x.shape[1]]
        mask = causal_mask(x.shape[1], device=x.device, dtype=x.dtype)
        x = self.backbone(x, mask=mask)
        x = self.norm_out(x)
        aq_out = x[:, -self.cfg.action_horizon:]          # (B, H, d)
        # Multimodal heads model the WHOLE chunk from one pooled conditioning bond h
        # (whole-chunk decode avoids per-step mode-flipping). The linear head keeps
        # the original per-token map.
        loss_action = None
        teacher_loss = None
        use_teacher = (self.cfg.distill_teacher
                       and self.cfg.action_head in ("flow", "product"))
        if self.cfg.action_head in ("flow", "product"):
            h = aq_out.mean(1)                            # (B, d)
            # Precision teacher: linear MSE head + per-sample unimodality weight.
            teacher_mean = w_unimodal = None
            if use_teacher and target_actions is not None:
                teacher_mean = self.teacher_head(aq_out)             # (B, H, d_a)
                teacher_loss = F.mse_loss(teacher_mean, target_actions)
                resid = ((target_actions - teacher_mean.detach()) ** 2
                         ).mean(dim=(1, 2)).sqrt()                   # (B,)
                w_unimodal = torch.exp(-0.5 * (resid / self.cfg.unimodal_tau) ** 2)
                teacher_mean = teacher_mean.detach()                 # anchor is a target
        if self.cfg.action_head == "flow":
            if target_actions is not None:
                loss_action = self.flow_head.fm_loss(
                    h, target_actions, teacher_mean=teacher_mean, w_unimodal=w_unimodal,
                    lambda_recon=self.cfg.lambda_recon, lambda_distill=self.cfg.lambda_distill)
            actions = self.flow_head.decode(
                h, n_steps=self.cfg.flow_steps, strategy=self.cfg.flow_decode,
                n_samples=self.cfg.flow_kwta_samples)
        elif self.cfg.action_head == "product":
            if target_actions is not None:
                # distill strong early, decays as the assignment sharpens
                lam_d = self.cfg.lambda_distill * (1.0 - progress)
                if self.cfg.product_st:
                    # Method 5: straight-through Gumbel routing, temperature annealed soft→sharp
                    st_temp = self.cfg.st_temp0 * (
                        self.cfg.st_temp1 / self.cfg.st_temp0) ** progress
                    loss_action, _ = self.product_head.loss(
                        h, target_actions, teacher_mean=teacher_mean, lambda_distill=lam_d,
                        straight_through=True, st_temp=st_temp, st_samples=self.cfg.st_samples)
                else:
                    tau = None
                    if self.cfg.curriculum:
                        tau = self.cfg.distill_tau0 * (
                            self.cfg.distill_tau1 / self.cfg.distill_tau0) ** progress
                    loss_action, _ = self.product_head.loss(
                        h, target_actions, tau=tau, teacher_mean=teacher_mean, lambda_distill=lam_d)
            actions = self.product_head.decode(h)
        elif self.cfg.action_head == "quantile":
            # PER-STEP conditioning (no pooling) — reads the full aq_out (B,H,d).
            if target_actions is not None:
                loss_action = self.quantile_head.loss(aq_out, target_actions)
            actions = self.quantile_head.decode(aq_out)   # deterministic AQ-MAP
        else:
            actions = self.action_head(aq_out)            # (B, H, d_a)
            if target_actions is not None:
                loss_action = F.mse_loss(actions, target_actions)

        phase_logits = None
        if self.cfg.n_phases > 0:
            # phase-query token sits just before (phase_cond, action_queries)
            pq_idx = -(1 + 1 + self.cfg.action_horizon)
            phase_logits = self.phase_head(x[:, pq_idx])  # (B, n_phases) LINEAR

        loss = None
        if target_actions is not None:
            loss = loss_action
            if teacher_loss is not None:
                loss = loss + self.cfg.teacher_weight * teacher_loss
            if phase_labels is not None and phase_logits is not None:
                loss = loss + phase_weight * F.cross_entropy(phase_logits, phase_labels)
        if return_phase:
            return actions, loss, phase_logits
        return actions, loss          # backward-compatible 2-tuple

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
