# χ‑VLA v0.1: full tensor-decomposable VLA specification

Yes. A ground-up tensor VLA can be specified cleanly by combining:

1. **OpenVLA’s overall division of labor:** dual visual encoders → visual projector → language/policy backbone → robot action.
2. **Riggs’ tensor-transformer replacements:** bilinear FFNs and softmax-free bilinear attention.
3. **Dooms et al.’s χ-net machinery:** homogeneous constant dimensions, CP-factorized bilinear cores, foldable RmsBatchNorm, and post-training orthogonalization/diagonalization/truncation.

OpenVLA currently uses fused DINOv2 and SigLIP features, a small two-layer MLP projector, and a Llama 2 backbone that predicts discretized robot actions. Image patches are processed separately by both vision encoders and concatenated channel-wise; each action dimension is discretized into 256 bins for token prediction. ([arXiv][1])

The proposed model replaces **every learned non-compositional component** in that pipeline.

---

## 1. Exact scope of “fully decomposable”

The final deployed policy core will implement:

[
F_\theta:
(\text{pixels},\text{token identities},\text{robot state})
\longrightarrow
\text{continuous pre-clipped action chunk}
]

using only:

* linear and affine maps;
* tensor contractions;
* element-wise multiplication;
* fixed addition and residual paths;
* fixed masks, permutations, and reshapes;
* learned constants;
* fixed scalar or diagonal rescalings.

After token IDs are represented as one-hot vectors, the entire learned map is a structured polynomial and can be unfolded into one tensor network.

The following remain outside the tensor-network boundary:

* string tokenization;
* camera decoding;
* environment-specific safety clipping;
* gripper thresholding;
* actuator communication.

Those are discrete or operational interfaces rather than learned neural components.

### Non-negotiable architecture rules

| Ordinary component                        | χ‑VLA replacement                                |
| ----------------------------------------- | ------------------------------------------------ |
| ReLU, GELU, SiLU, sigmoid                 | Element-wise products between linear projections |
| SwiGLU FFN                                | (D((Lx)\odot(Rx)))                               |
| Softmax attention                         | Product of two learned attention score matrices  |
| LayerNorm/RMSNorm at inference            | Training-time RmsBatchNorm, then freeze and fold |
| MLP visual projector                      | Cross-bilinear CP-factorized projector           |
| DINOv2/SigLIP ordinary ViTs               | Two independently pretrained χ‑ViTs              |
| Llama 2 ordinary transformer              | χ-transformer                                    |
| Discrete action tokenizer and LM decoding | Direct continuous linear action-chunk head       |
| Bias terms                                | Homogeneous constant coordinate                  |
| Dynamic token pruning/MoE/top-(k)         | Prohibited in the initial strict architecture    |

A terminology distinction matters:

* **CP decomposition** parameterizes bilinear and multilinear weight tensors.
* **RmsBatchNorm** provides foldable normalization.

CP is not itself a normalization method.

---

# 2. End-to-end architecture

```text
RGB OBSERVATION
      │
      ├─────────────────────────────────────────────┐
      │                                             │
fixed patch extraction                       fixed patch extraction
      │                                             │
linear patch embedding                       linear patch embedding
      │                                             │
SPATIAL χ-ViT                               SEMANTIC χ-ViT
DINO-like training role                    SigLIP-like training role
      │                                             │
spatial patch features                       semantic patch features
      └──────────────────────┬──────────────────────┘
                             │
                    CROSS-BILINEAR PROJECTOR
          linear paths + spatial × semantic interactions
                             │
                    projected visual tokens
                             │
                             ├─────────────────────────────────┐
                             │                                 │
LANGUAGE INSTRUCTION         │                         ROBOT STATE
      │                      │                                 │
external tokenizer           │                         fixed affine scaling
      │                      │                                 │
one-hot token IDs            │                         linear embedding
      │                      │                                 │
linear embedding             │                         embodiment token
      │                      │                                 │
      └──────────────────────┴─────────────────────────────────┘
                             │
             [visual prefix | language prefix |
              proprioception | learned action queries]
                             │
                    JOINT χ-TRANSFORMER
        ┌────────────────────────────────────────────────┐
        │ foldable RmsBatchNorm                         │
        │ bilinear causal attention                     │
        │ residual                                      │
        │ foldable RmsBatchNorm                         │
        │ bilinear FFN                                  │
        │ residual                                      │
        │ repeat L times                                │
        └────────────────────────────────────────────────┘
                             │
                     action-query states
                             │
                  linear continuous-action head
                             │
                   fixed affine denormalization
                             │
             continuous pre-clipped action chunk
                             │
                  external robot safety layer
```

This preserves OpenVLA’s high-level visual–language–action design while replacing its learned internals.

---

# 3. Universal homogeneous-coordinate convention

Dooms et al. append a constant input so a deep χ-net can express lower-order as well as highest-order polynomial terms:

[
\bar x=
\begin{bmatrix}
1\x
\end{bmatrix}.
]

Without this constant, a stack of (L) self-bilinear layers naturally emphasizes degree (2^L) terms. With the constant dimension, it can represent every degree up to (2^L), including:

* biases;
* constants;
* linear terms;
* pairwise interactions;
* higher-order interactions.

This is essential for practical generalization. Their χ-net cores are CP/Khatri–Rao-factorized bilinear maps, and the constant dimension allows parts of the network to remain effectively linear when that is what the problem requires. 

For clarity, the mathematical specification explicitly uses (\bar x=[1;x]). The implementation can use ordinary linear-layer biases and convert them to homogeneous matrices during tensor-network export.

A residual update is represented as:

[
\begin{bmatrix}
1\x
\end{bmatrix}
+
\begin{bmatrix}
0\\Delta x
\end{bmatrix}
=============

\begin{bmatrix}
1\x+\Delta x
\end{bmatrix}.
]

The constant coordinate is therefore preserved exactly.

---

# 4. Spatial and semantic χ‑vision encoders

OpenVLA gets complementary visual information by combining DINOv2 and SigLIP. The tensor model should preserve that useful division of labor, but it cannot retain the original nonlinear encoders if pixel-to-action decomposability is required. ([arXiv][1])

We therefore train two tensor vision transformers.

## 4.1 Spatial χ‑ViT

Purpose:

* local geometry;
* edges and contours;
* object positions;
* spatial relationships;
* pose and orientation;
* fine manipulation structure.

Its training role is analogous to DINOv2, but every transformer block is tensor-decomposable.

## 4.2 Semantic χ‑ViT

Purpose:

* object identity;
* image-language alignment;
* attributes and categories;
* instruction-relevant semantics;
* visual concepts learned from image-text data.

Its training role is analogous to SigLIP, again using a fully tensorized forward architecture.

The two branches may use identical architectures but different weights and training objectives.

## 4.3 Patch input

For a reference 224 × 224 image with 14 × 14 patches:

[
N_v=16\times16=256
]

patch tokens.

For each patch (p_i):

[
x_i=W_{\text{patch}}p_i+b_{\text{patch}}.
]

Patch extraction is a fixed reshape and the projection is affine, so it is tensor-compatible.

Fixed RGB standardization can be folded directly into (W_{\text{patch}}) and (b_{\text{patch}}).

Each branch receives:

[
X^{(0)}=
\left[
x_1+e^{2D}*1+e*{\text{camera}},
\ldots,
x_{N_v}+e^{2D}*{N_v}+e*{\text{camera}}
\right].
]

Learned positional and camera embeddings are constants, so their addition is affine.

## 4.4 χ‑ViT block

Every vision block uses:

[
U_\ell=\operatorname{RBN}^{A}*\ell(X*\ell)
]

[
X'_\ell
=======

X_\ell+
\alpha_\ell
\operatorname{BiAttn}*\ell(U*\ell)
]

[
V_\ell=\operatorname{RBN}^{F}*\ell(X'*\ell)
]

[
X_{\ell+1}
==========

X'*\ell+
\beta*\ell
\operatorname{BFFN}*\ell(V*\ell).
]

Here:

* `RBN` is foldable RmsBatchNorm;
* `BiAttn` is bilinear attention;
* `BFFN` is a CP-factorized bilinear FFN;
* (\alpha_\ell,\beta_\ell) are fixed or learned scalar residual gains.

No GELU, softmax, or LayerNorm remains after export.

---

# 5. Bilinear FFN specification

For token (x\in\mathbb R^d), define:

[
\bar x=[1;x]\in\mathbb R^{d+1}.
]

The bilinear FFN is:

[
\operatorname{BFFN}(x)
======================

D
\left[
(L\bar x)\odot(R\bar x)
\right],
]

with:

[
L,R\in\mathbb R^{r\times(d+1)},
\qquad
D\in\mathbb R^{d\times r}.
]

Each hidden rank component computes:

[
(L_r^\top\bar x)(R_r^\top\bar x).
]

The output is:

[
y_o
===

\sum_{r=1}^{R}
D_{or}
(L_r^\top\bar x)
(R_r^\top\bar x).
]

This is a CP factorization of a dense third-order tensor:

[
T\in\mathbb R^{d_{\text{out}}\times(d+1)\times(d+1)}.
]

Its parameter count is approximately:

[
3dr,
]

rather than the approximately (d^3) cost of a dense bilinear tensor.

Riggs’ experiments replace SwiGLU,

[
D\left(\operatorname{swish}(Lx)\odot Rx\right),
]

with exactly this activation-free bilinear form. In the reported approximately 500M-parameter language models, bilinear FFNs and SwiGLU FFNs were nearly equivalent under the selected training comparison. 

### Recommended implementation

```python
class BilinearFFN:
    left:  Linear(d, rank, bias=True)
    right: Linear(d, rank, bias=True)
    down:  Linear(rank, d, bias=True)

    def forward(self, x):
        return self.down(self.left(x) * self.right(x))
```

The bias terms are converted into homogeneous-coordinate tensor entries during export.

### Rank choice

Initial recommendation:

[
r_{\text{vision}}=3d_{\text{vision}},
\qquad
r_{\text{joint}}=3d_{\text{joint}}.
]

This keeps the parameter count similar to a gated transformer FFN.

### Post-training symmetrization

Because both branches receive the same (x), the effective bilinear core can be symmetrized after training:

[
T_{oij}
\leftarrow
\frac12(T_{oij}+T_{oji}).
]

Dooms et al. enforce this symmetry before ODT analysis. A factorized implementation may require refactorization or rank expansion to express the symmetrized core exactly. 

---

# 6. Bilinear attention specification

## 6.1 Basic head

Given token matrix:

[
X\in\mathbb R^{N\times d},
]

each head computes two query–key systems:

[
Q_1=XW_{Q_1},\qquad K_1=XW_{K_1},
]

[
Q_2=XW_{Q_2},\qquad K_2=XW_{K_2},
]

and one value system:

[
V=XW_V.
]

The two score matrices are:

[
A_1=Q_1K_1^\top,
\qquad
A_2=Q_2K_2^\top.
]

The tensor attention pattern is:

[
A=A_1\odot A_2.
]

The head output is:

[
Y=
C
\left[
M\odot
\frac{A_1\odot A_2}{d_h}
\right]
V.
]

Where:

* (M) is a fixed attention mask;
* (d_h) is head dimension;
* (C) is a fixed row-scaling matrix.

A suitable default row scale is:

[
C_{ii}=\frac{1}{\sqrt{n_i}},
]

where (n_i) is the fixed number of keys visible to query (i). An (1/n_i) scale should also be tested.

There is no softmax.

Riggs’ reported variant uses two independently learned (Q,K) systems and multiplies their attention patterns element-wise. The approximately 500M-parameter experiments showed a modest degradation relative to softmax attention, though the comparison had parameter, kernel, and hyperparameter caveats. 

## 6.2 Polynomial degree

Each attention score is quadratic in the token inputs:

[
(Q_1K_1^\top)_{ij}.
]

Multiplying two score systems makes the attention pattern fourth-order. Multiplying by (V) produces an overall degree-five attention update.

That high degree is why normalization and residual scaling must be treated as first-class architectural components.

## 6.3 Khatri–Rao feature form

Define row-wise Kronecker features:

[
\widetilde q_i=q_{1,i}\otimes q_{2,i},
\qquad
\widetilde k_j=k_{1,j}\otimes k_{2,j}.
]

Then:

[
(q_{1,i}^{\top}k_{1,j})
(q_{2,i}^{\top}k_{2,j})
=======================

\widetilde q_i^\top\widetilde k_j.
]

Therefore:

[
Y
=

\widetilde Q
\left(
\widetilde K^\top V
\right).
]

This is exactly equivalent to:

[
(Q_1K_1^\top)\odot(Q_2K_2^\top)
]

but does not require materializing the full attention matrix.

Riggs derives complexity of approximately:

[
O(Nd_h^3)
]

for the factorized contraction, compared with:

[
O(N^2d_h)
]

for ordinary attention. The factorized contraction becomes attractive when (N>d_h^2), while the attention matrix rank is bounded by (d_h^2). 

## 6.4 Causal scan form

For the joint decoder, define:

[
S_i
===

\sum_{j\leq i}
\widetilde k_jv_j^\top.
]

Then:

[
y_i
===

\frac{1}{d_h\sqrt{i}}
\widetilde q_i^\top S_i.
]

This can be implemented as an associative prefix scan.

For a bidirectional vision encoder:

[
S
=

\sum_j\widetilde k_jv_j^\top,
\qquad
y_i
===

\frac{1}{d_h\sqrt N}
\widetilde q_i^\top S.
]

## 6.5 Execution choice

For the proposed VLA, (N) will initially be only a few hundred tokens. Since this is usually below (d_h^2), the explicit (N\times N) implementation may be faster than the Khatri–Rao implementation.

The implementation should support both:

```text
short sequence:
explicit tiled attention

long sequence/video/multi-camera:
Khatri–Rao contraction or causal prefix scan
```

Both compute the same tensor function.

## 6.6 Parameterization

A standard attention layer has four dense maps:

[
Q,K,V,O.
]

Bilinear attention has six:

[
Q_1,K_1,Q_2,K_2,V,O.
]

Thus the attention module uses approximately:

[
6d^2
]

parameters instead of (4d^2).

This is a 50% increase in attention parameters, though the increase in whole-model parameters is much smaller because FFNs dominate many transformer parameter budgets.

---

# 7. Foldable RmsBatchNorm

## 7.1 Training-time operation

At each normalization site, maintain a running scalar RMS:

[
c_t
===

\rho c_{t-1}
+
(1-\rho)
\sqrt{
\mathbb E_{\text{batch,tokens}}
\left[
\frac{|x|_2^2}{d}
\right]
}.
]

During training:

[
\operatorname{RBN}(x)
=====================

\frac{x}{c_t+\epsilon}.
]

Recommended initial settings:

[
\rho\in{0.99,0.999},
\qquad
\epsilon=10^{-6}.
]

Dooms et al. use a variant of BatchNorm that only divides by an average L2/RMS magnitude. After training, that average is contracted into a neighboring matrix. Their reported χ-net therefore has no remaining input-dependent normalization in its exported form. 

## 7.2 Pre-attention folding

Suppose:

[
u=x/c.
]

Then:

[
Q_1=uW_{Q_1}=x(W_{Q_1}/c).
]

At export:

[
W'*{Q_1}=W*{Q_1}/c,
]

and likewise for:

[
W_{K_1},W_{Q_2},W_{K_2},W_V.
]

The normalization module is deleted.

## 7.3 Pre-FFN folding

For:

[
B(x/c)
======

D
\left[
\left(Lx/c\right)
\odot
\left(Rx/c\right)
\right],
]

fold with:

[
L'=L/c,
\qquad
R'=R/c.
]

Equivalently:

[
D'=D/c^2.
]

The first form is preferable because it preserves output-projection scale.

## 7.4 Optional head-level RBN

Riggs notes that the experimental tensor transformer normalized (Q) and (K) using a procedure that was not itself fully tensor-network-compatible. 

Replace that with scalar running RMS values for every head branch:

[
q'*1=q_1/c*{q_1},
\qquad
k'*1=k_1/c*{k_1},
]

and similarly for (q_2,k_2,v).

Each constant folds directly into its projection matrix.

## 7.5 Calibration and export

After training:

1. Disable gradient updates.
2. Re-estimate all running RMS constants on a representative calibration set.
3. Freeze the constants.
4. Fold each constant into its neighboring weights.
5. Delete all RBN modules.
6. Compare pre-fold and post-fold outputs.

Required equivalence:

[
\max |F_{\text{before}}(x)-F_{\text{after}}(x)|
<10^{-5}
]

in FP32 on the calibration suite.

The final inference graph contains no normalization division.

---

# 8. Dual-encoder fusion and projector

Let the aligned spatial and semantic patch representations be:

[
s_i\in\mathbb R^{d_s},
\qquad
m_i\in\mathbb R^{d_m}.
]

The strict cross-bilinear projector is:

[
p_i
===

D_p
\left[
(L_s\bar s_i)
\odot
(R_m\bar m_i)
\right].
]

With homogeneous coordinates, the same layer can encode:

* spatial-only terms;
* semantic-only terms;
* constant bias terms;
* spatial × semantic interactions.

For optimization, use an explicit residual linear path:

[
p_i
===

W_s s_i
+
W_m m_i
+
D_p
\left[
(L_s\bar s_i)
\odot
(R_m\bar m_i)
\right].
]

All three paths remain tensor-decomposable.

```text
spatial patch sᵢ ── Ws ──────────────────────────┐
                                                 │
semantic patch mᵢ ─ Wm ──────────────────────────┤
                                                 ├── sum → visual token pᵢ
sᵢ ── Ls ──┐                                    │
             ⊙ ── Dp ────────────────────────────┘
mᵢ ── Rm ──┘
```

This is preferable to simple channel concatenation followed by an MLP because it exposes the cross-encoder interactions explicitly.

Recommended projector rank:

[
r_p=2d_{\text{joint}}.
]

---

# 9. Language and robot-state inputs

## 9.1 Language

The tokenizer remains external:

[
\text{string}\rightarrow(t_1,\ldots,t_T).
]

Each token ID is interpreted as a basis vector:

[
e_t=E,\mathrm{onehot}(t).
]

The embedding matrix is therefore linear.

Use a fixed maximum instruction length, initially:

[
T=64.
]

Shorter instructions are filled with an ordinary PAD token. Avoid a sample-dependent padding mask in the first strict implementation; PAD is simply another token the model learns to ignore.

## 9.2 Robot state

Normalize proprioception using fixed dataset statistics:

[
p'=\operatorname{diag}(\sigma^{-1})(p-\mu).
]

This is affine and can be folded into its embedding:

[
e_p=W_pp'+b_p.
]

Useful inputs include:

* joint positions;
* end-effector pose;
* gripper state;
* previous action;
* control frequency;
* camera calibration metadata;
* robot embodiment ID.

An embodiment ID is one-hot encoded and linearly embedded.

Use a common normalized action space across embodiments whenever possible. If robots have different action dimensions, use a fixed maximum action vector and an external execution mask rather than dynamic internal heads.

---

# 10. Joint χ-transformer sequence

Recommended sequence layout:

```text
[visual patch tokens]
[BOS]
[instruction tokens, padded to fixed length]
[robot-state token]
[embodiment token]
[action query 1]
[action query 2]
...
[action query H]
```

For the initial model:

* visual tokens: 256;
* language tokens: 64;
* robot/embodiment tokens: 2;
* action queries: 8;
* total: approximately 330 tokens.

Use a fixed causal mask.

This retains the decoder-only organization of an OpenVLA-like system:

* visual tokens form the earliest prefix;
* instruction tokens can attend to visual information;
* robot-state tokens can attend to image and language;
* each action query can attend to the complete prior context;
* later action queries can attend to earlier action queries.

Because the mask is fixed, it is a constant sparse tensor.

## Joint block pseudocode

```python
def chi_transformer_block(x):
    u = rbn_attention(x)
    x = x + attention_gain * bilinear_attention(u)

    v = rbn_ffn(x)
    x = x + ffn_gain * bilinear_ffn(v)

    return x
```

Recommended initial residual gains:

[
\alpha_\ell=\beta_\ell=\frac{1}{\sqrt{2L}}.
]

An alternative is a learned scalar initialized to (0.01), with regularization but no nonlinear bounding function.

---

# 11. Continuous action decoder

Original OpenVLA uses tokenized actions and a deterministic action detokenizer. ([arXiv][1])

For a genuinely decomposable pixel-to-action core, replace that with learned action-query tokens.

For action query state (h_k):

[
\hat a_k=W_ah_k+b_a.
]

For an (H)-step chunk:

[
\hat A
======

\begin{bmatrix}
\hat a_1\
\vdots\
\hat a_H
\end{bmatrix}
\in\mathbb R^{H\times d_a}.
]

A typical (d_a=7) output can represent:

[
[\Delta x,\Delta y,\Delta z,
\Delta r_1,\Delta r_2,\Delta r_3,
g].
]

Use a fixed affine map to restore environment units:

[
a^{\text{physical}}
===================

\operatorname{diag}(\sigma_a)\hat a+\mu_a.
]

This can be folded into (W_a,b_a).

The gripper output remains a real score. A hardware-side threshold occurs outside the network.

No internal:

* tanh;
* sigmoid;
* softmax;
* sampling;
* clipping.

A discrete action-token compatibility version can also be trained, but only the map to action logits would be tensor-decomposable; argmax and token lookup would remain external.

---

# 12. Reference χ‑VLA-450M configuration

This is a research-scale starting point aligned with the scale at which the LessWrong language experiment became encouraging. It is not claimed to be optimal.

| Component                  | Configuration                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------ |
| Input image                | 224 × 224 RGB                                                                        |
| Patch size                 | 14 × 14                                                                              |
| Patches per camera         | 256                                                                                  |
| Cameras                    | 1 initially                                                                          |
| Spatial χ‑ViT              | 12 layers, (d=512), 16 heads, (d_h=32), FFN rank 1536                                |
| Semantic χ‑ViT             | Same architecture, separate weights                                                  |
| Fusion projector           | output (d=1024), rank 2048                                                           |
| Joint χ-transformer        | 20 layers, (d=1024), 32 heads, (d_h=32), FFN rank 3072                               |
| Text vocabulary            | approximately 32K                                                                    |
| Maximum instruction length | 64                                                                                   |
| Action queries             | 8                                                                                    |
| Action dimensions          | 7                                                                                    |
| Normalization              | scalar pre-attention and pre-FFN RmsBatchNorm; optional scalar per Q/K/V head branch |
| Position encoding          | learned fixed 2D vision embeddings and learned 1D sequence embeddings                |
| Output                     | 8 × 7 continuous action chunk                                                        |
| Approximate parameters     | 447M                                                                                 |

Approximate breakdown:

* dual vision encoders: 94M;
* joint backbone: 315M;
* language embeddings: 33M;
* projector, patch embeddings, and action head: approximately 5M.

The bilinear FFNs have approximately the same parameter structure as SwiGLU. Most of the increase relative to an ordinary transformer comes from the second (Q,K) pair.

---

# 13. Initialization and optimization

## 13.1 Initialization

For every linear input projection:

[
\operatorname{std}(W)\approx\frac{1}{\sqrt{d_{\text{in}}}}.
]

Use orthogonal or semi-orthogonal initialization where dimensions allow.

For bilinear FFNs:

* initialize (L) and (R) independently;
* initialize (D) at reduced scale;
* apply residual scaling separately.

For attention:

* normalize Q/K projection scales so each score factor has order-one variance after division by (\sqrt{d_h});
* initialize (W_O) at reduced residual-branch scale.

## 13.2 CP factor balancing

CP parameterizations have scale-gauge freedom:

[
l_r\rightarrow a l_r,
\qquad
r_r\rightarrow b r_r,
\qquad
d_r\rightarrow\frac{1}{ab}d_r.
]

The represented function is unchanged, but individual factor norms can become badly conditioned.

Every fixed number of optimizer steps:

1. Compute the norms of each CP rank component.
2. Redistribute the scale across (L_r,R_r,D_r).
3. Preserve the represented tensor exactly.
4. Log the maximum factor imbalance.

A geometric-mean balancing rule is appropriate:

[
g_r=
\left(
|l_r||r_r||d_r|
\right)^{1/3}.
]

Rescale all three factors toward norm (g_r).

## 13.3 Recommended optimizer setup

Initial sweep:

* optimizer: AdamW;
* precision: bfloat16;
* gradient clipping: 1.0;
* warmup: 2–5%;
* cosine decay;
* weight decay: sweep 0.05–0.5;
* no dropout in the first strict model;
* EMA momentum for RmsBatchNorm: 0.99 and 0.999;
* monitor activation RMS at every block;
* monitor CP factor condition numbers;
* monitor attention accumulator norms.

Dooms et al. found χ-nets can become less competitive in low-data settings, while the LessWrong results also varied substantially with model scale and hyperparameters. Large-scale pretraining or distillation should therefore precede robot-only training.

---

# 14. Training curriculum

## Stage 0: operator validation

Before training a VLA, validate every tensor primitive.

Required tests:

* bilinear FFN versus explicit dense third-order tensor;
* explicit attention versus Khatri–Rao attention;
* causal score matrix versus prefix-scan implementation;
* RmsBatchNorm fold equivalence;
* bias-to-homogeneous-coordinate conversion;
* residual direct-sum tensor export;
* FP32 and BF16 numerical stability.

## Stage 1: χ-language pretraining

Train the joint backbone as a causal language model before introducing vision.

Architecture:

```text
token embedding
      │
χ-transformer
      │
linear vocabulary logits
```

Softmax and cross-entropy may be used in the **training loss**. They are not part of the deployed VLA forward graph.

Compare four matched baselines:

1. softmax attention + SwiGLU;
2. softmax attention + bilinear FFN;
3. bilinear attention + SwiGLU;
4. bilinear attention + bilinear FFN.

This reproduces the key LessWrong ablation in the intended model implementation before adding multimodality.

## Stage 2: spatial χ‑vision pretraining

Train the spatial branch with one or more of:

* masked patch prediction;
* teacher feature regression;
* image reconstruction in latent space;
* self-distillation between image crops;
* dense spatial correspondence.

Teacher models and non-tensor training objectives are acceptable because they are not part of inference.

## Stage 3: semantic χ‑vision pretraining

Train the semantic branch with:

* image-text contrastive learning;
* caption-conditioned feature regression;
* SigLIP teacher distillation;
* object/category supervision where available.

The final branch remains a pure χ‑ViT regardless of the training objective.

## Stage 4: χ‑VLM alignment

Join:

* both vision encoders;
* cross-bilinear projector;
* pretrained χ-language backbone.

Train on image-text and visual-instruction data.

Possible objectives:

* next-token language modeling;
* image-caption contrastive loss;
* teacher-logit distillation;
* patch-to-word alignment;
* visual question-answering.

## Stage 5: robot behavior cloning

Use robot trajectories containing:

[
(o_t,\ell,p_t,a_{t:t+H-1}),
]

where:

* (o_t) is image observation;
* (\ell) is language instruction;
* (p_t) is robot state;
* (a_{t:t+H-1}) is action chunk.

Primary loss:

[
\mathcal L_{\text{action}}
==========================

\sum_{k=1}^{H}
\sum_{j=1}^{d_a}
w_j
\left(
\hat a_{k,j}-a_{k,j}
\right)^2.
]

Optional training-only losses:

[
\mathcal L
==========

\mathcal L_{\text{action}}
+
\lambda_{\text{distill}}\mathcal L_{\text{teacher}}
+
\lambda_{\text{align}}\mathcal L_{\text{vision-language}}
+
\lambda_{\text{ortho}}\mathcal L_{\text{orthogonality}}
+
\lambda_{\text{factor}}\mathcal L_{\text{factor-balance}}.
]

OpenVLA’s original training uses a large cross-embodiment collection and action-token cross-entropy; the proposed model can use the same observations and instructions but directly regress normalized continuous action chunks. ([arXiv][1])

### Unfreezing schedule

A reasonable sequence is:

1. Train projector and action head with both vision branches and joint backbone frozen.
2. Unfreeze the upper half of the joint backbone.
3. Unfreeze the full joint backbone.
4. Unfreeze the semantic vision branch.
5. Unfreeze the spatial branch last.
6. Finish with low-learning-rate end-to-end training.

## Stage 6: normalization calibration and folding

After behavioral training:

1. Run a broad calibration set across all embodiments.
2. Freeze each RmsBatchNorm statistic.
3. Fold all statistics into projection matrices.
4. Delete RmsBatchNorm modules.
5. Export the pure tensor graph.
6. Re-evaluate robot behavior before decomposition.

---

# 15. Tensor-network export format

Each component should export to a common intermediate representation.

## Node types

* matrix;
* vector;
* third-order CP core;
* copy/cloning tensor;
* element-wise product spider;
* fixed mask tensor;
* addition/direct-sum tensor;
* reshape/permutation;
* homogeneous-coordinate injector;
* constant positional tensor.

## Required metadata

Every open or internal wire should record:

* dimension;
* semantic role;
* layer;
* token position or token class;
* modality;
* camera;
* attention head;
* CP rank index;
* action horizon;
* action dimension.

This metadata will be essential when interpreting globally ranked directions.

## Exported blocks

### Bilinear FFN

```text
            ┌── L ──┐
input ─copy─┤        spider ── D ── output
            └── R ──┘
```

### Bilinear attention

```text
input copies
  │
  ├── Q1 ─────┐
  ├── K1 ─────┤── score contraction 1 ──┐
  ├── Q2 ─────┤                          spider ── value contraction
  ├── K2 ─────┤── score contraction 2 ──┘
  └── V ────────────────────────────────────────────┘
```

### Residual

Represent the identity path and tensor branch as a sum of compatible tensor networks.

---

# 16. ODT analysis plan

Dooms et al.’s published ODT procedure performs:

1. **Orthogonalization:** reduced RQ decomposition of each matricized core, replacing it by an isometry and pushing the non-orthogonal factor upward.
2. **Diagonalization:** construction and eigendecomposition of a global Gram matrix at each hidden bond.
3. **Truncation:** projection onto the globally most important singular directions and absorption of those projectors into adjacent cores.

For their tree χ-net, the stated complexity is (O(Lh^4)), with (O(h^3)) memory as the major width bottleneck. Their small SVHN experiment found substantial globally low-rank structure and removed roughly 70% of hidden dimensions without reducing classification accuracy. That result is encouraging but should not be assumed to transfer directly to a VLA. 

## 16.1 Critical limitation

The published ODT guarantees apply to a tree-structured χ-net.

A transformer with:

* attention across token positions;
* residual sums;
* shared projection matrices;
* multiple output action queries;

is a more general tensor network.

Therefore two different claims must remain separate:

### Guaranteed by architecture

The complete χ‑VLA inference function is exactly representable as a tensor network.

### Not yet guaranteed at scale

The published tree-ODT algorithm can be applied unchanged to the full attention-containing VLA with the same complexity and error bound.

That second claim requires new work.

## 16.2 Decomposition levels

### Level A: exact module decomposition

Apply immediately to:

* every bilinear FFN;
* visual projector;
* action head;
* individual attention-head score tensors.

This yields local and input/output-conditioned spectra.

### Level B: exact stage-global decomposition

Define major analysis bonds:

* spatial vision output;
* semantic vision output;
* fused visual output;
* each joint-backbone block output;
* action-query representation;
* action output.

For each bond (H_i):

1. Split the tensor graph into upstream pre-network (P_i) and downstream post-network (Q_i).
2. Orthogonalize (P_i).
3. Construct:

[
G_i=Q_i^\ast Q_i.
]

4. Eigendecompose:

[
G_i=V_i\Lambda_iV_i^\ast.
]

5. Order hidden directions using (\Lambda_i).
6. Insert and optionally truncate (V_iV_i^\ast).

This is the direct conceptual generalization of ODT.

### Level C: full token- and head-level global decomposition

Unroll the model at a fixed:

* image resolution;
* text length;
* action horizon;
* causal mask.

Represent bilinear attention through its Khatri–Rao accumulator rather than a dense score matrix. Cache repeated environment contractions and exploit shared head structure.

Start with a small model where exact contraction is feasible. Only then attempt approximate low-rank environments for the 450M model.

## 16.3 ODT-native fallback

If applying the published ODT algorithm with minimal theoretical extension is a hard requirement, replace attention with a hierarchical tree mixer:

```text
patch/token leaves
      │
pairwise bilinear merges
      │
higher-level bilinear merges
      │
multimodal root
      │
action output
```

That model is much closer to the tree χ-net analyzed by Dooms et al., but it gives up the stronger transformer capability evidence supplied by the LessWrong experiments.

The recommended research program is therefore:

* tensor-attention χ‑VLA as the primary capability model;
* smaller tree-structured χ‑VLA as the exact-ODT reference model.

---

# 17. Interpretability outputs

After global diagonalization, the system should expose:

## Visual atoms

Project important spatial and semantic directions back through the patch embeddings to pixels.

Expected objects:

* edge or contour detectors;
* object-part templates;
* spatial relation patterns;
* semantic object directions;
* camera-specific features.

## Cross-encoder interactions

For projector component (r):

[
(L_s\bar s)_r(R_m\bar m)_r.
]

This directly indicates which spatial direction interacts with which semantic direction.

## Language directions

Project globally important joint-backbone directions into the token embedding matrix:

[
E^\top v.
]

This produces vocabulary tokens most aligned with each direction.

## Action directions

Project a hidden singular direction through the action head:

[
W_av.
]

This reveals whether the direction primarily controls:

* translation;
* rotation;
* gripper state;
* a particular action horizon;
* combinations of control dimensions.

## Circuits

Trace high-weight paths such as:

```text
image patch direction
      ×
instruction-token direction
      →
fused visual-language factor
      →
action-query direction
      →
end-effector translation
```

## Faithfulness tests

For each extracted direction:

1. Truncate it.
2. Project it out.
3. Amplify it.
4. Replace it with another direction.
5. Measure predicted action change.
6. Run robot evaluation when safe.

A valid interpretation should predict the effect of these interventions.

---

# 18. Required ablations

| Ablation                          | Purpose                               |
| --------------------------------- | ------------------------------------- |
| Softmax + SwiGLU                  | Ordinary matched baseline             |
| Softmax + bilinear FFN            | Isolate FFN replacement               |
| Bilinear attention + SwiGLU       | Isolate attention replacement         |
| Bilinear attention + bilinear FFN | Tensor-transformer core               |
| One χ-vision branch               | Test dual-encoder necessity           |
| Concatenation projector           | Compare against cross-bilinear fusion |
| Tied squared attention            | Compare lower-parameter attention     |
| Untied bilinear attention         | Main model                            |
| Per-instance L2 normalization     | Non-strict stability control          |
| Foldable RmsBatchNorm             | Strict final model                    |
| Discrete action head              | OpenVLA-comparable output             |
| Continuous action head            | Strict end-to-end tensor core         |
| Local SVD                         | Compare with global ODT               |
| ODT stage decomposition           | Test global low-rank structure        |
| ODT truncation                    | Test compression and faithfulness     |

---

# 19. Acceptance criteria

## Tensor purity

* No ReLU, GELU, SiLU, sigmoid, softmax, LayerNorm, RMSNorm, top-(k), or input-dependent division in the exported inference graph.
* Every affine map exported through a homogeneous coordinate.
* Fixed masks represented as tensors.
* Pre-fold and post-fold numerical outputs match.

## Operator correctness

* Explicit and Khatri–Rao attention agree within (10^{-5}) FP32 relative error.
* Causal dense attention and prefix-scan attention agree within (10^{-5}).
* Bilinear FFNs match explicit dense-core contraction on small dimensions.
* Tensor-network export matches the PyTorch forward pass.

## Capability

A reasonable initial gate is:

* at least 90% of matched baseline task success on a small controlled benchmark;
* no catastrophic loss of language grounding;
* no large increase in unstable or out-of-range actions;
* competitive action MSE under equal training data.

The final target should be parity rather than merely surviving.

## Decomposition

* Exact reconstruction before truncation.
* Singular directions ordered consistently across repeated calibrations.
* Truncation error predicted by tensor norm on small exact models.
* Extracted directions show intervention faithfulness.
* Global ODT outperforms local SVD at preserving task behavior for a matched rank budget.

## Deployment

* Post-fold inference uses only the tensor core.
* Safety clipping remains external and auditable.
* Output norms remain bounded on in-distribution and stress-test inputs.
* Inference latency is measured both with explicit and factorized attention kernels.

---

# 20. Principal risks and mitigations

## Norm instability

**Risk:** repeated degree-two and degree-five operations amplify outliers.

**Mitigation:**

* pre-branch RmsBatchNorm;
* optional Q/K/V branch RmsBatchNorm;
* fixed attention scaling;
* conservative residual gains;
* CP factor balancing;
* gradient clipping;
* orthogonal initialization;
* periodic activation-norm audits.

## Insufficient attention expressivity

**Risk:** bilinear attention has rank bounded by (d_h^2) and lacks softmax’s sharply selective behavior. 

**Mitigation:**

* more heads with small (d_h);
* untied (Q_1,K_1,Q_2,K_2);
* local and global heads;
* increase (d_h) where memory permits;
* retain an identity/linear token-mixing path;
* distill attention outputs from an ordinary teacher;
* compare tied squared attention against untied bilinear attention.

## Low-data sample efficiency

**Risk:** robotics datasets may not compensate for the inductive-bias change.

**Mitigation:**

* language and vision pretraining first;
* teacher distillation;
* Open-X-scale behavior data;
* freeze/unfreeze curriculum;
* regularize toward pretrained feature spaces.

## Attention-kernel speed

**Risk:** optimized softmax kernels may initially be faster.

**Mitigation:**

* explicit fused Triton kernel for short sequences;
* Khatri–Rao scan for long sequences;
* choose implementation dynamically from (N) and (d_h^2);
* benchmark test-time latency separately from theoretical FLOPs.

## ODT scalability

**Risk:** the full attention network is not the same tree topology analyzed in the Dooms paper.

**Mitigation:**

* exact ODT on a small reference model;
* stage-global decomposition first;
* use accumulator-form attention;
* low-rank environment approximations;
* maintain an ODT-native hierarchical baseline;
* do not claim the tree-network error bound for the full transformer until derived.

---

# 21. Recommended build sequence

## Milestone 1: tensor-transformer core

Build:

* foldable RmsBatchNorm;
* bilinear FFN;
* explicit bilinear attention;
* Khatri–Rao attention;
* causal prefix scan;
* tensor export.

Train a small language model and reproduce the four-way FFN/attention ablation.

## Milestone 2: single χ‑ViT

Train one tensor vision encoder on a small image task. Confirm:

* stable optimization;
* normalization folding;
* pixel-space feature projection;
* local ODT.

## Milestone 3: dual χ‑vision system

Train separate spatial and semantic branches. Add the cross-bilinear projector and test whether its rank components show coherent cross-branch interactions.

## Milestone 4: small χ‑VLA

Use:

* one camera;
* short instructions;
* 4–8 action queries;
* 50–100M total parameters;
* a controlled simulation benchmark.

This is the first pixel-language-to-action tensor network.

## Milestone 5: exact decomposition reference

Train a small ODT-native or fixed-token tensor model. Perform:

* full network contraction;
* orthogonalization;
* global Gram computation;
* diagonalization;
* truncation;
* robot-policy intervention tests.

## Milestone 6: χ‑VLA-450M

Scale to the reference configuration, pretrain vision and language, then train on a broad robot mixture.

## Milestone 7: generalized ODT

Develop stage-global and eventually end-to-end environment contractions for the attention-containing tensor network.

---

# Final reference specification

The proposed strict model is:

[
\boxed{
\begin{aligned}
&\text{pixels}
\xrightarrow{\text{linear patch maps}}
\text{spatial and semantic χ-ViTs}\
&\xrightarrow{\text{cross-bilinear CP projector}}
\text{visual tokens}\
&\quad+\text{linear token embeddings}
+\text{linear robot-state embeddings}\
&\xrightarrow{\text{bilinear attention + bilinear FFN blocks}}
\text{action-query states}\
&\xrightarrow{\text{linear head}}
\text{continuous action chunk}.
\end{aligned}
}
]

During training, every multiplicative branch uses Dooms-style running-RMS normalization. After calibration, all normalization constants are folded into adjacent weights and removed. Every FFN and visual projector is stored as a CP-factorized tensor. Every attention block is represented as a product of two query–key score tensors and can be contracted either explicitly or through Khatri–Rao features. Residuals are sums of compatible tensor networks, biases are encoded through a constant coordinate, and action decoding is linear.

That gives a **genuinely tensor-decomposable learned VLA core from pixels and token identities to continuous robot actions**. The architecture is concrete with the existing methods. The remaining major research challenge is not constructing the forward model; it is extending Dooms’ efficient global ODT guarantees from tree χ-nets to the complete attention-containing multimodal tensor network.

[1]: https://arxiv.org/html/2406.09246v3 "OpenVLA: An Open-Source Vision-Language-Action Model"
