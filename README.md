# χ-VLA — a fully tensor-decomposable Vision-Language-Action model

Implementation of a VLA whose entire
*learned* pixel-to-action map is a structured polynomial (linear/affine maps,
tensor contractions, element-wise products, residual sums, fixed masks, learned
constants, foldable scalar rescalings) — **no softmax, GELU/ReLU/SiLU,
LayerNorm/RMSNorm, or input-dependent division** survives in the exported
inference graph. The whole thing unfolds into one tensor network for ODT-style
interpretability (Dooms et al.).

Design references (in repo): `chi nets.pdf` (Dooms et al., foldable norm + ODT)
and the LessWrong tensor-transformer PDF / Logan Riggs'
[`modded-nanogpt`](https://github.com/loganriggs/modded-nanogpt) (bilinear FFN +
softmax-free bilinear attention).

## Status — Milestone 1: tensor-transformer core (spec §21)

Implemented and validated:

| Primitive | File | Spec |
|---|---|---|
| Homogeneous coordinate (bias→constant col) | `xvla/nn/homogeneous.py` | §3 |
| Foldable RmsBatchNorm (scalar, foldable) | `xvla/nn/normalization.py` | §7 |
| Bilinear FFN (CP-factorized `D[(Lx̄)⊙(Rx̄)]`) | `xvla/nn/bilinear.py` | §5 |
| Bilinear attention — explicit / Khatri-Rao / causal-scan | `xvla/nn/attention.py` | §6 |
| Optional scalar per-Q/K/V RBN (foldable QK-norm) | `xvla/nn/attention.py` | §7.4 |
| χ-transformer block + stack | `xvla/nn/block.py` | §4.4, §10 |
| χ-language model + 4-way ablation | `xvla/models/lm.py` | §14 |
| CP factor balancing | `xvla/train/balance.py` | §13.2 |
| Stage-1 training loop | `xvla/train/train_lm.py` | §13.3, §14 |

Stage-0 operator validation (`tests/`, spec §14 Stage 0 / §19): BFFN vs dense CP
core, explicit vs Khatri-Rao vs causal-scan attention, RBN fold equivalence
(linear, bilinear FFN, QK/V branches, and **end-to-end LM**), bias→homogeneous,
residual constant preservation, BF16/FP32 stability. **14/14 pass.**

## Strict, fully-foldable model — the central claim (verified ✅)

The working per-token model uses input-dependent norm (not tensor-pure). The
*strict* model uses scalar `RmsBatchNorm` everywhere; once calibrated + frozen,
every norm is a fixed scalar rescaling (spec §1) — no input-dependent division
(§19). It's distilled from the per-token teacher (`xvla/train/distill.py`),
calibrated (`xvla/train/calibrate.py`), and folded to a normalization-free graph
(`xvla/train/fold.py`).

Result (tiny-shakespeare): the strict scalar-norm LM trains stably to **val 5.68**
and **folds to a pure tensor network with max |pre−post| = 7.6e-6 < 1e-5**. The
key enabler is **near-identity learned residual gains** (init 0.01, spec §10):
they keep the stack near-linear so the degree-2 FFN's per-instance magnitude
amplification — which diverges with the default `1/√2L` gains — never compounds.
See `DEVLOG.md` for the full arc (the fixed-gain strict model diverges; this is
spec risk §20 #1 characterized and resolved).

## Running (compute on Modal)

Local Python is 3.14 (no torch wheels); all execution is on Modal GPUs.

```bash
modal run modal_app.py::validate               # operator + fold tests (GPU)
modal run modal_app.py::prepare_data           # tokenize tiny-shakespeare into the volume
modal run modal_app.py::smoke_ablation         # fast 4-way ablation
modal run modal_app.py::run_ablation           # larger 4-way ablation
modal run modal_app.py::distill_strict_model   # strict scalar-norm model: distill→calibrate→fold→§19 gate
```

## Next milestones

M2 single χ-ViT · M3 dual χ-vision + cross-bilinear projector · M4 small χ-VLA ·
M5 exact-ODT reference · M6 χ-VLA-450M · M7 generalized ODT.
