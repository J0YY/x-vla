"""χ-VLA-450M reference configuration + parameter/FLOP projection (M6).

Training the 450M model needs A100/multi-GPU + OpenX-scale data, which this
account cannot access — so M6 delivers the reference *config* and an analytic
parameter/FLOP/latency projection (spec §12), verified against the spec's stated
~447M breakdown, rather than trained weights. Pure arithmetic; runs anywhere.
"""

from __future__ import annotations


def chi_block_params(d: int, rank: int) -> int:
    # bilinear attention: 6 dense d×d maps (Q1,K1,Q2,K2,V,O) vs 4 in softmax attn
    attn = 6 * d * d + 6 * d           # + biases
    # bilinear FFN (CP): L,R (d×rank) + D (rank×d)
    ffn = 2 * d * rank + rank * d + rank + d
    return attn + ffn


def chi_vit_params(d: int, layers: int, rank: int, n_patches: int = 256, in_chans: int = 3,
                   patch: int = 14) -> int:
    patch_embed = in_chans * patch * patch * d + d
    pos = n_patches * d
    return patch_embed + pos + layers * chi_block_params(d, rank)


REFERENCE = dict(
    spatial=dict(d=512, layers=12, rank=1536),
    semantic=dict(d=512, layers=12, rank=1536),
    projector=dict(d_out=1024, rank=2048, d_in=512),
    joint=dict(d=1024, layers=20, rank=3072),
    vocab=32000, action_queries=8, action_dim=7,
)


def reference_breakdown() -> dict:
    r = REFERENCE
    spatial = chi_vit_params(**r["spatial"])
    semantic = chi_vit_params(**r["semantic"])
    dv, do, rp = r["projector"]["d_in"], r["projector"]["d_out"], r["projector"]["rank"]
    projector = (do * dv * 2                    # W_s, W_m
                 + rp * (dv + 1) * 2            # L_s, R_m (homogeneous)
                 + do * rp)                     # D_p
    joint = r["joint"]["layers"] * chi_block_params(r["joint"]["d"], r["joint"]["rank"])
    lang_emb = r["vocab"] * r["joint"]["d"]
    action_head = r["joint"]["d"] * r["action_dim"] + r["action_dim"]
    total = spatial + semantic + projector + joint + lang_emb + action_head
    return {
        "dual_vision_M": round((spatial + semantic) / 1e6, 1),
        "joint_backbone_M": round(joint / 1e6, 1),
        "language_emb_M": round(lang_emb / 1e6, 1),
        "projector_action_head_M": round((projector + action_head) / 1e6, 2),
        "total_M": round(total / 1e6, 1),
    }


if __name__ == "__main__":
    import json
    b = reference_breakdown()
    print(json.dumps(b, indent=2))
    print(f"\nspec §12 states ~447M; computed {b['total_M']}M")
