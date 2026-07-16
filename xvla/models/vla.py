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

from xvla.models.vit import ChiViT, ViTConfig
from xvla.nn.attention import causal_mask
from xvla.nn.block import ChiTransformer
from xvla.nn.normalization import make_norm
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
    norm: str = "per_token"
    qk_norm: str = "per_token"
    # action
    action_horizon: int = 4
    action_dim: int = 7

    def vit_config(self):
        return ViTConfig(image_size=self.image_size, patch_size=self.patch_size,
                         dim=self.vit_dim, n_layers=self.vit_layers, n_heads=self.vit_heads,
                         norm=self.norm, qk_norm=self.qk_norm, num_classes=1)


class ChiVLA(nn.Module):
    def __init__(self, cfg: VLAConfig):
        super().__init__()
        self.cfg = cfg
        self.vision = ChiViT(cfg.vit_config())
        if cfg.dual_vision:
            self.vision2 = ChiViT(cfg.vit_config())
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

        n_vis = cfg.vit_config().num_patches
        seq_len = n_vis + 1 + cfg.max_instr_len + 1 + 1 + cfg.action_horizon
        self.seq_len = seq_len
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, cfg.dim))
        for p in (self.bos, self.action_queries, self.pos_emb):
            nn.init.normal_(p, std=cfg.dim ** -0.5)
        nn.init.normal_(self.tok_emb.weight, std=cfg.dim ** -0.5)
        nn.init.normal_(self.embodiment_emb.weight, std=cfg.dim ** -0.5)

        self.backbone = ChiTransformer(cfg.dim, cfg.n_layers, cfg.n_heads,
                                       ffn_rank=cfg.ffn_rank, causal=True,
                                       norm=cfg.norm, qk_norm=cfg.qk_norm)
        self.norm_out = make_norm(cfg.norm)
        self.action_head = nn.Linear(cfg.dim, cfg.action_dim)

    def _visual_tokens(self, img, img2=None):
        s = self.vision.features(img)                     # (B, Nv, vit_dim)
        if not self.cfg.dual_vision:
            return self.vis_proj(s)
        m = self.vision2.features(img2 if img2 is not None else img)
        if self.projector is not None:
            return self.projector(s, m)
        return self.vis_proj(torch.cat([s, m], dim=-1))

    def forward(self, img, instr_ids, state, embodiment_id, img2=None,
                target_actions=None):
        B = img.shape[0]
        vis = self._visual_tokens(img, img2)              # (B, Nv, d)
        bos = self.bos.expand(B, -1, -1)
        instr = self.tok_emb(instr_ids)                   # (B, T, d)
        st = self.state_proj(state)[:, None]              # (B, 1, d)
        emb = self.embodiment_emb(embodiment_id)[:, None] # (B, 1, d)
        aq = self.action_queries.expand(B, -1, -1)        # (B, H, d)
        x = torch.cat([vis, bos, instr, st, emb, aq], dim=1)
        x = x + self.pos_emb[:, : x.shape[1]]
        mask = causal_mask(x.shape[1], device=x.device, dtype=x.dtype)
        x = self.backbone(x, mask=mask)
        x = self.norm_out(x)
        aq_out = x[:, -self.cfg.action_horizon:]          # (B, H, d)
        actions = self.action_head(aq_out)                # (B, H, d_a)
        loss = None
        if target_actions is not None:
            loss = F.mse_loss(actions, target_actions)
        return actions, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
