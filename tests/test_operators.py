"""Stage 0: operator validation (spec §14 Stage 0, §19 operator correctness).

Every tensor primitive is validated before any training:

* bilinear FFN vs explicit dense third-order tensor,
* explicit attention vs Khatri–Rao attention,
* causal score matrix vs prefix-scan implementation,
* RmsBatchNorm fold equivalence,
* bias → homogeneous-coordinate conversion,
* residual direct-sum preservation of the constant coordinate,
* FP32 / BF16 numerical stability.

All exact-equivalence checks must hold to ~1e-5 in FP32.
"""

import torch
import pytest

from xvla.nn.attention import BilinearAttention, causal_mask
from xvla.nn.bilinear import BilinearFFN
from xvla.nn.homogeneous import homogenize, to_homogeneous_matrix
from xvla.nn.normalization import RmsBatchNorm

TOL = 1e-5


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- bilinear FFN vs dense third-order CP core -------------------------------
def test_bffn_matches_dense_core():
    d, r, out = 6, 10, 5
    ffn = BilinearFFN(d, rank=r, out_dim=out, down_bias=False).double()
    x = torch.randn(4, d, dtype=torch.float64)
    T = ffn.dense_core()  # (out, d+1, d+1)
    xbar = homogenize(x)
    expected = torch.einsum("oij,bi,bj->bo", T, xbar, xbar)
    got = ffn(x)
    assert torch.allclose(got, expected, atol=TOL), (got - expected).abs().max()


# --- explicit vs Khatri–Rao attention ---------------------------------------
@pytest.mark.parametrize("causal", [False, True])
def test_explicit_vs_khatri_rao(causal):
    dim, heads, N, B = 16, 4, 12, 3
    attn = BilinearAttention(dim, heads, causal=causal).double().eval()
    x = torch.randn(B, N, dim, dtype=torch.float64)
    a = attn(x, method="explicit")
    b = attn(x, method="khatri_rao")
    assert torch.allclose(a, b, atol=TOL), (a - b).abs().max()


def test_explicit_vs_khatri_rao_custom_mask():
    dim, heads, N, B = 16, 4, 10, 2
    attn = BilinearAttention(dim, heads).double().eval()
    x = torch.randn(B, N, dim, dtype=torch.float64)
    mask = (torch.rand(N, N) > 0.3).double()
    mask[range(N), range(N)] = 1.0  # every query sees at least itself
    a = attn(x, mask=mask, method="explicit")
    b = attn(x, mask=mask, method="khatri_rao")
    assert torch.allclose(a, b, atol=TOL), (a - b).abs().max()


# --- causal explicit vs prefix scan -----------------------------------------
@pytest.mark.parametrize("row_scale", ["invsqrt", "inv"])
def test_causal_explicit_vs_scan(row_scale):
    dim, heads, N, B = 16, 4, 14, 3
    attn = BilinearAttention(dim, heads, causal=True, row_scale=row_scale).double().eval()
    x = torch.randn(B, N, dim, dtype=torch.float64)
    explicit = attn(x, method="explicit")
    scan = attn(x, method="causal_scan")
    assert torch.allclose(explicit, scan, atol=TOL), (explicit - scan).abs().max()


# --- RmsBatchNorm fold equivalence ------------------------------------------
def test_rbn_fold_into_linear():
    d, out = 8, 5
    rbn = RmsBatchNorm().double()
    lin = torch.nn.Linear(d, out).double()

    # Warm up the running statistic, then freeze (post-calibration state).
    rbn.train()
    for _ in range(50):
        rbn(torch.randn(16, 32, d, dtype=torch.float64) * 2.3)
    rbn.eval()
    rbn.freeze()

    x = torch.randn(4, d, dtype=torch.float64)
    ref = lin(rbn(x))

    c = rbn.scale
    folded_w = lin.weight / c        # W' = W / c  (spec §7.2)
    folded = x @ folded_w.T + lin.bias
    assert torch.allclose(ref, folded, atol=TOL), (ref - folded).abs().max()


def test_rbn_fold_into_bilinear_ffn():
    d = 8
    rbn = RmsBatchNorm().double()
    ffn = BilinearFFN(d, rank=16, down_bias=False).double()
    rbn.train()
    for _ in range(50):
        rbn(torch.randn(16, 32, d, dtype=torch.float64) * 1.7)
    rbn.eval(); rbn.freeze()

    x = torch.randn(4, d, dtype=torch.float64)
    ref = ffn(rbn(x))

    # Fold c into both input projections: L'=L/c, R'=R/c (spec §7.3).
    c = rbn.scale
    left = (x @ (ffn.left.weight / c).T) + ffn.left.bias
    right = (x @ (ffn.right.weight / c).T) + ffn.right.bias
    folded = (left * right) @ ffn.down.weight.T
    assert torch.allclose(ref, folded, atol=TOL), (ref - folded).abs().max()


# --- QK/V branch RBN folds into projections (spec §7.4) ---------------------
def test_qk_rbn_folds_into_projections():
    dim, heads, N, B = 16, 4, 12, 2
    a = BilinearAttention(dim, heads, causal=True, qk_rbn=True).double()
    # Warm up the per-branch running RMS, then freeze (post-calibration).
    a.train()
    for _ in range(40):
        a(torch.randn(B, N, dim, dtype=torch.float64) * 1.9)
    a.eval()
    for m in (a.rbn_q1, a.rbn_k1, a.rbn_q2, a.rbn_k2, a.rbn_v):
        m.freeze()

    # Build an equivalent module with qk_rbn removed, folding each scalar
    # c into its projection: W' = W/c, b' = b/c.
    b = BilinearAttention(dim, heads, causal=True, qk_rbn=False).double()
    b.load_state_dict(
        {k: v for k, v in a.state_dict().items() if not k.startswith("rbn_")},
        strict=False,
    )
    for lin, rbn in ((b.wq1, a.rbn_q1), (b.wk1, a.rbn_k1),
                     (b.wq2, a.rbn_q2), (b.wk2, a.rbn_k2), (b.wv, a.rbn_v)):
        lin.weight.data.copy_(lin.weight.data / rbn.scale)
        lin.bias.data.copy_(lin.bias.data / rbn.scale)
    b.eval()

    x = torch.randn(B, N, dim, dtype=torch.float64)
    assert torch.allclose(a(x), b(x), atol=TOL), (a(x) - b(x)).abs().max()


# --- bias → homogeneous coordinate ------------------------------------------
def test_bias_to_homogeneous():
    d, out = 7, 4
    lin = torch.nn.Linear(d, out).double()
    x = torch.randn(5, d, dtype=torch.float64)
    ref = lin(x)
    Wbar = to_homogeneous_matrix(lin.weight, lin.bias)  # (out, d+1)
    got = homogenize(x) @ Wbar.T
    assert torch.allclose(ref, got, atol=TOL)


# --- residual preserves the constant coordinate -----------------------------
def test_residual_preserves_constant():
    # x̄ + [0; Δx] = [1; x+Δx]: the leading constant coordinate is unchanged.
    x = torch.randn(3, 6, dtype=torch.float64)
    dx = torch.randn(3, 6, dtype=torch.float64)
    xbar = homogenize(x)
    update = torch.cat([torch.zeros(3, 1, dtype=torch.float64), dx], dim=-1)
    out = xbar + update
    assert torch.allclose(out[:, 0], torch.ones(3, dtype=torch.float64), atol=TOL)
    assert torch.allclose(out[:, 1:], x + dx, atol=TOL)


# --- BF16 / FP32 stability ---------------------------------------------------
def test_bf16_close_to_fp32():
    dim, heads, N, B = 32, 8, 20, 2
    attn = BilinearAttention(dim, heads, causal=True).eval()
    x = torch.randn(B, N, dim)
    fp32 = attn(x, method="explicit")
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        bf16 = attn(x, method="explicit")
    # BF16 has ~1e-2 relative precision; just require no blow-up / NaNs.
    assert torch.isfinite(bf16).all()
    rel = (fp32 - bf16.float()).abs().max() / fp32.abs().max().clamp_min(1e-6)
    assert rel < 5e-2, rel
