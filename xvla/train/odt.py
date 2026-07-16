"""Exact orthogonalize / diagonalize / truncate (ODT) for the χ-MLP chain (M5).

The χ-MLP (after folding its scalar norms) is a chain of symmetric bilinear cores:

    z0 = [1; E x]
    a_i = C_i(z_{i-1}, z_{i-1})      C_i ∈ R^{d×(d+1)×(d+1)} symmetric
    z_i = [1; a_i]
    logits = W_head a_L + b_head

This module: (1) folds + exports the cores and checks the unrolled tensor network
reproduces the module forward (exact reconstruction, spec §14/§19); (2) builds the
weight-based GLOBAL downstream Gram at a bond — contracting the *entire* downstream
sub-network (all layers below the bond) with itself over the output and environment
legs — vs the LOCAL Gram (adjacent core only); (3) truncates the bond onto the top
directions of each and measures accuracy, so global ODT can be compared to local SVD
at matched rank (spec §16.2 / §18).

Only feedforward chains are handled — exact global ODT on the attention/residual
transformer is Level-C / open (spec §16.1).
"""

from __future__ import annotations

import copy

import torch

from xvla.models.chi_mlp import ChiMLP
from xvla.nn.normalization import RmsBatchNorm


@torch.no_grad()
def export_cores(model: ChiMLP):
    """Fold scalar norms into each bilinear layer and return (embed, cores, head).

    ``cores[i]`` is the symmetric dense core ``C_{i+1}`` (fp64), shape
    (d, d+1, d+1), consuming homogeneous ``z_i`` and producing ``a_{i+1}``.
    """
    m = copy.deepcopy(model).double().eval()
    cores = []
    for norm, layer in zip(m.norms, m.layers):
        if isinstance(norm, RmsBatchNorm):           # fold c into L,R (spec §7.3)
            c = norm.scale
            layer.left.weight.div_(c); layer.right.weight.div_(c)
        T = layer.dense_core()                       # (d, d+1, d+1)
        T = 0.5 * (T + T.transpose(1, 2))            # symmetrize (spec §5)
        cores.append(T)
    embed = (m.embed.weight.detach().clone(), m.embed.bias.detach().clone())
    head = (m.head.weight.detach().clone(), m.head.bias.detach().clone())
    return embed, cores, head


def _homog(a):
    return torch.cat([a.new_ones(*a.shape[:-1], 1), a], dim=-1)


@torch.no_grad()
def unroll_forward(embed, cores, head, x, proj=None, proj_bond=None):
    """Exact tensor-network forward from the cores.

    ``proj`` (d×d) optionally projects ``a_{proj_bond}`` (1-indexed layer output)
    onto a subspace — used for truncation evaluation.
    """
    We, be = embed
    a = x.flatten(1).double() @ We.T.double() + be.double()
    for i, C in enumerate(cores, start=1):
        z = _homog(a)
        a = torch.einsum("oij,bi,bj->bo", C, z, z)
        if proj is not None and proj_bond == i:
            a = a @ proj.T
    Wh, bh = head
    return a @ Wh.T.double() + bh.double()


@torch.no_grad()
def _fold_head_into_last(cores, head):
    """C_L' [c,i,j] = Σ_o W_head[c,o] C_L[o,i,j]  (drops the head bias / constant)."""
    Wh, _ = head
    return torch.einsum("co,oij->cij", Wh.double(), cores[-1])


@torch.no_grad()
def downstream_gram(cores, head, bond: int):
    """Weight-based GLOBAL Gram of ``a_bond`` directions over the full downstream.

    ``bond`` is 1-indexed: bond=i analyzes ``a_i`` = output of layer i, which feeds
    layers i+1..L. Returns the (d×d) Gram over the non-homogeneous (a) subspace.
    """
    L = len(cores)
    C_last = _fold_head_into_last(cores, head)        # (num_classes, d+1, d+1)
    if bond == L - 1:
        # downstream is just the last layer: G[p,p'] = Σ_{c,q} C[c,p,q] C[c,p',q]
        G = torch.einsum("cpq,cPq->pP", C_last, C_last)
        return G[1:, 1:]
    if bond == L - 2:
        # downstream = layers (L-1, L). Build degree-4 tensor D over z_bond.
        C_mid = cores[bond]                           # (d, d+1, d+1): a_{bond+1}
        d1 = C_mid.shape[1]
        # Z2[p,i,j]: z_{bond+1}[p] as a quadratic form in z_bond.
        Z2 = C_mid.new_zeros(d1, d1, d1)
        Z2[0, 0, 0] = 1.0                             # homogeneous coord: z[0]=1
        Z2[1:, :, :] = C_mid
        # out[c] = Σ_{p,q} C_last[c,p,q] Z2[p,i,j] Z2[q,k,l] z_i z_j z_k z_l
        D = torch.einsum("cpq,pij,qkl->cijkl", C_last, Z2, Z2)
        G = torch.einsum("cijkl,cIjkl->iI", D, D)
        return G[1:, 1:]
    raise NotImplementedError("downstream_gram supports the two deepest bonds (L=3)")


@torch.no_grad()
def local_gram(cores, bond: int):
    """LOCAL Gram: only the adjacent core that consumes ``a_bond`` (layer bond+1)."""
    C = cores[bond]                                   # (d, d+1, d+1)
    G = torch.einsum("opq,oPq->pP", C, C)
    return G[1:, 1:]


@torch.no_grad()
def top_projector(G_aa, k: int):
    """Rank-k projector onto the top-k eigenvectors of a (d×d) Gram, + spectrum."""
    evals, evecs = torch.linalg.eigh(G_aa)            # ascending
    evals = evals.flip(0); evecs = evecs.flip(1)
    Vk = evecs[:, :k]
    return Vk @ Vk.T, evals


@torch.no_grad()
def random_projector(d: int, k: int, device, rng=None):
    """Rank-k projector onto a uniformly random k-subspace of R^d (control).

    Truncating onto this instead of the top-k global directions is the causal
    control: if global >> random at matched k, the spectral ranking is doing
    real work, not just any k dimensions.
    """
    A = torch.randn(d, k, generator=rng, device=device, dtype=torch.float64)
    Vk, _ = torch.linalg.qr(A)
    return Vk @ Vk.T


@torch.no_grad()
def accuracy(logits, labels):
    return (logits.argmax(-1) == labels).float().mean().item()


@torch.no_grad()
def truncation_curve(embed, cores, head, bond, G_aa, x, labels, ks):
    """Accuracy vs retained rank k at ``bond`` using directions from ``G_aa``."""
    out = {}
    for k in ks:
        P, _ = top_projector(G_aa, k)
        logits = unroll_forward(embed, cores, head, x, proj=P, proj_bond=bond)
        out[k] = accuracy(logits, labels)
    return out
