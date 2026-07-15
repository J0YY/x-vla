"""χ-language model for Stage-1 pretraining and the four-way ablation (spec §14).

The joint backbone is trained as a causal LM *before* vision is introduced. The
four matched configurations (spec §14 Stage 1 / §18) are selected by
``LMConfig.attn`` × ``LMConfig.ffn``:

    1. softmax  + swiglu     (ordinary matched baseline)
    2. softmax  + bilinear   (isolate the FFN replacement)
    3. bilinear + swiglu     (isolate the attention replacement)
    4. bilinear + bilinear   (the tensor-transformer core)

Softmax and cross-entropy live only in the *training loss*; configuration (4)'s
forward graph contains no softmax/activation and is fully tensor-decomposable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from xvla.nn.attention import BilinearAttention
from xvla.nn.baselines import SoftmaxAttention, SwiGLU
from xvla.nn.bilinear import BilinearFFN
from xvla.nn.normalization import make_norm


@dataclass
class LMConfig:
    vocab_size: int = 50304
    dim: int = 512
    n_layers: int = 12
    n_heads: int = 8
    ffn_rank: int | None = None       # default 3*dim
    max_seq_len: int = 512
    attn: str = "bilinear"            # "bilinear" | "softmax"
    ffn: str = "bilinear"             # "bilinear" | "swiglu"
    qk_norm: str = "per_token"        # "per_token" | "scalar_rbn" | "none"
    norm: str = "per_token"           # block/out norm: "per_token"|"scalar_rbn"|"none"
    rbn_momentum: float = 0.99
    learned_gain: bool = False
    tie_embeddings: bool = True

    @property
    def tag(self) -> str:
        return f"{self.attn}-{self.ffn}"


class _LMBlock(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        gain = (2.0 * cfg.n_layers) ** -0.5
        self.rbn_attn = make_norm(cfg.norm, momentum=cfg.rbn_momentum)
        self.rbn_ffn = make_norm(cfg.norm, momentum=cfg.rbn_momentum)

        if cfg.attn == "bilinear":
            self.attn = BilinearAttention(cfg.dim, cfg.n_heads, causal=True,
                                          qk_norm=cfg.qk_norm, rbn_momentum=cfg.rbn_momentum)
        elif cfg.attn == "softmax":
            self.attn = SoftmaxAttention(cfg.dim, cfg.n_heads, causal=True)
        else:
            raise ValueError(f"unknown attn {cfg.attn!r}")

        if cfg.ffn == "bilinear":
            self.ffn = BilinearFFN(cfg.dim, rank=cfg.ffn_rank)
        elif cfg.ffn == "swiglu":
            self.ffn = SwiGLU(cfg.dim, rank=cfg.ffn_rank)
        else:
            raise ValueError(f"unknown ffn {cfg.ffn!r}")

        if cfg.learned_gain:
            self.attn_gain = nn.Parameter(torch.tensor(0.01))
            self.ffn_gain = nn.Parameter(torch.tensor(0.01))
        else:
            self.register_buffer("attn_gain", torch.tensor(gain))
            self.register_buffer("ffn_gain", torch.tensor(gain))

    def forward(self, x, method: str = "explicit"):
        u = self.rbn_attn(x)
        x = x + self.attn_gain * self.attn(u, method=method)
        v = self.rbn_ffn(x)
        x = x + self.ffn_gain * self.ffn(v)
        return x


class ChiLanguageModel(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.dim)
        self.blocks = nn.ModuleList([_LMBlock(cfg) for _ in range(cfg.n_layers)])
        # Final normalization before the linear vocabulary head.
        self.rbn_out = make_norm(cfg.norm, momentum=cfg.rbn_momentum)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight

        nn.init.normal_(self.tok_emb.weight, std=cfg.dim ** -0.5)
        nn.init.normal_(self.pos_emb.weight, std=cfg.dim ** -0.5)

    def forward(self, idx, targets=None, method: str = "explicit", return_hidden: bool = False,
                input_scale: float = 1.0):
        B, N = idx.shape
        pos = torch.arange(N, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        if input_scale != 1.0:  # magnitude stress test (spec §19 stress inputs)
            x = x * input_scale
        hiddens = []
        for block in self.blocks:
            x = block(x, method=method)
            if return_hidden:
                hiddens.append(x)
        x = self.rbn_out(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        if return_hidden:
            return logits, loss, hiddens
        return logits, loss

    def num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        if self.cfg.tie_embeddings:
            n -= self.head.weight.numel()  # shared with tok_emb
        return n
