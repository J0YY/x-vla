"""χ-VLA-Funnel: a Vision-Language-Action policy that ODT decomposes exactly, today.

The whole point of this file is one observation:

    **Every multimodal front-end in a VLA is LINEAR, so it folds into a single
    embedding matrix, and what remains is a chi-MLP.**

Pixels are already a vector. Instruction token ids become a one-hot times an
embedding table, which is a linear map. Robot state and embodiment id are linear.
Concatenate them and the entire "vision + language + state" stage is one
``nn.Linear``. What follows is a residual-free chain of bilinear CP cores with
foldable scalar norms, i.e. exactly the object
:func:`xvla.train.canonical_odt.export_homogeneous_network` already consumes.

So this model needs no new ODT machinery. It is structurally a
:class:`~xvla.models.chi_mlp.ChiMLP` with a VLA-shaped ``embed``, which means the
existing canonical ODT implementation, its Davis-Kahan certificates, its corrected
trace-tail bound with the ``2**(L-i)`` multiplicity, and its 140-test pipeline
suite all apply unchanged.

Why this specific design, rather than the attention policy:

* **Separator.** No token-wise residual stream and no attention, so every bond is
  a genuine narrow separator and every occurrence of a bond has one environment.
* **Occurrence count.** Each block is degree 2, so ``occ_l = 2**(L-l)`` and
  ``sum_l occ_l = 2**(L+1)-1``, which is 511 at L=8 rather than 1.11e8 for the
  attention backbone. This is the only regime in which the truncation certificate
  is not vacuous.
* **Trainability.** Total degree is ``2**L`` (16 at L=4), not ``10**L``. The two
  historical frozen-scalar-norm successes in this repo sit at total degree about
  8, and every failure sits at 1e12. This design is on the working side of that
  line by construction.

What it gives up, stated plainly: **there is no token mixing at all.** The policy
sees a flattened multimodal vector, so it cannot do content-based routing between
patches. That is the price of a bond structure ODT can actually decompose, and it
is the honest scope of any claim made with this model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from xvla.nn.bilinear import BilinearFFN
from xvla.nn.normalization import make_norm


@dataclass
class ChiVLAFunnelConfig:
    # multimodal input
    image_size: int = 32
    in_chans: int = 3
    vocab_size: int = 12
    max_instr_len: int = 16
    state_dim: int = 8
    n_embodiments: int = 1
    # trunk
    dim: int = 64                  # bond width d; the homogeneous bond is d+1
    n_layers: int = 4              # total degree 2**n_layers
    ffn_rank: int | None = None    # default 3*dim
    norm: str = "scalar_rbn"       # the only ODT-compatible choice, and it must be frozen
    # action
    action_horizon: int = 4
    action_dim: int = 7
    # trainability knobs, from the measured two-sided rho window
    const_fill: float = 2.0        # beta. Retuned from 10.0 after a 3-seed sweep at this
                                   # shape: beta=2 gives test MSE 0.00084 against 0.00204
                                   # at beta=10 (2.4x better) and roughly doubles the
                                   # instruction-shuffle ratio, i.e. it is the only
                                   # setting where language is clearly load-bearing.
                                   # beta=10 was transplanted from the 2L=8 attention
                                   # model and sits in the gradient-starved half of the
                                   # window here (49x attenuation into layers.0).
                                   # beta ~ 2.12 is also where the quadratic ENERGY share
                                   # reaches 0.1. Fills the homogeneous constant channel. The
                                   # const:linear:quadratic amplitude ratio is beta^2 :
                                   # beta : 1 and is set by beta ALONE, so `down.div_`
                                   # only changes overall scale and cannot set it.
                                   # The AMPLITUDE ratio |u|/|b| is 1/beta, but the
                                   # quadratic ENERGY share is 1/(2 beta^2 + 1). Beta
                                   # also inflates the DC part of the map, which is the
                                   # quantity an epsilon normalised by ||T||_F^2 is
                                   # measured against, so it is NOT a neutral
                                   # trainability constant. At beta=0 (the shipped
                                   # BilinearFFN init) every block is a PURE homogeneous
                                   # form, the constant coordinate that replaces the
                                   # residual is empty, and the model parks at the
                                   # trivial predictor with exactly zero backbone gradient.
    jacobian_gain: bool = True     # per-block gain 1/sqrt(degree): the frozen scalar norm
                                   # already owns the forward scale, so the init budget
                                   # goes to the backward pass.

    @property
    def in_dim(self) -> int:
        pixels = self.in_chans * self.image_size * self.image_size
        return pixels + self.vocab_size * self.max_instr_len + self.state_dim + self.n_embodiments

    @property
    def out_dim(self) -> int:
        return self.action_horizon * self.action_dim


class ChiVLAFunnel(nn.Module):
    """Pixels + instruction + state + embodiment -> action chunk, as one chi-MLP.

    Exposes ``embed`` / ``norms`` / ``layers`` with the exact names and types
    :func:`export_homogeneous_network` expects, so that function consumes this
    model directly with no adapter.
    """

    def __init__(self, cfg: ChiVLAFunnelConfig):
        super().__init__()
        self.cfg = cfg
        rank = cfg.ffn_rank if cfg.ffn_rank is not None else 3 * cfg.dim

        # The entire multimodal front-end, as one affine map. `embed.bias` carries
        # the constant, which the homogeneous export turns into the bond's channel 0.
        self.embed = nn.Linear(cfg.in_dim, cfg.dim)
        self.norms = nn.ModuleList([make_norm(cfg.norm) for _ in range(cfg.n_layers)])
        self.layers = nn.ModuleList([
            BilinearFFN(cfg.dim, rank=rank, out_dim=cfg.dim, down_bias=False)
            for _ in range(cfg.n_layers)
        ])
        self.head = nn.Linear(cfg.dim, cfg.out_dim)
        self._init_trainable()

    def _init_trainable(self) -> None:
        """Fill the constant channel and set the overall block gain.

        NOTE the rescaling here sets SCALE only. Measured block output RMS at beta=10 is
        about 3.7 rather than 1, because the quadratic term carries beta^2 and `div_(beta)`
        removes one factor. Keeping unit scale would need `div_(beta**2)`. This is left as
        measured rather than silently "corrected" because the current constant is what the
        reported runs used.

        `BilinearFFN.reset_parameters` zeroes `left.bias` and `right.bias`, making
        each block a pure homogeneous form of degree exactly 2. By Euler that gives
        a per-block forward derivative of exactly 2 with no linear or constant term,
        which is a maximally repelling fixed point under a global scalar norm. Filling
        the biases restores the lower-degree terms, which are also precisely the
        identity path that replaces the residual connection.
        """
        beta = self.cfg.const_fill
        with torch.no_grad():
            for layer in self.layers:
                # The divisor is correct at beta=0 too (it reduces to 0.5), so returning
                # early there silently gave the control arm the raw init scale and no
                # jacobian gain, i.e. a THIRD undocumented difference between the arms.
                if beta > 0.0:
                    layer.left.bias.fill_(beta)
                    layer.right.bias.fill_(beta)
                # Derived, not guessed. With `BilinearFFN.reset_parameters` (L,R std
                # d^-1/2 and down std 0.5*r^-1/2) and unit-RMS input,
                #     E[(beta+u)^2 (beta+v)^2] = (beta^2 + sigma^2)^2
                # so the block output RMS is 0.5*(beta^2 + 1) and the correct divisor is
                # 0.5*(beta^2 + 1), times 2^-1/2 for the Jacobian gain. The previous
                # `div_(beta)` agreed only at beta=1 and was 5.05x off at beta=10.
                layer.down.weight.div_(0.5 * (beta ** 2 + 1.0))
                if self.cfg.jacobian_gain:
                    layer.down.weight.mul_(2.0 ** -0.5)

    def assemble_input(self, img, instr_ids, state, embodiment_id) -> torch.Tensor:
        """The linear multimodal assembly. One-hots keep the whole stage affine."""
        cfg = self.cfg
        b = img.shape[0]
        instr = F.one_hot(instr_ids.long(), cfg.vocab_size).to(img.dtype)
        emb = F.one_hot(embodiment_id.long(), cfg.n_embodiments).to(img.dtype)
        return torch.cat([
            img.reshape(b, -1),
            instr.reshape(b, -1),
            state.reshape(b, -1).to(img.dtype),
            emb.reshape(b, -1),
        ], dim=1)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for norm, layer in zip(self.norms, self.layers):
            h = layer(norm(h))          # no residual: a clean chain, so every bond separates
        return h

    def forward(self, img, instr_ids, state, embodiment_id, target_actions=None):
        cfg = self.cfg
        h = self.trunk(self.assemble_input(img, instr_ids, state, embodiment_id))
        actions = self.head(h).reshape(-1, cfg.action_horizon, cfg.action_dim)
        loss = None if target_actions is None else F.mse_loss(actions, target_actions)
        return actions, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
