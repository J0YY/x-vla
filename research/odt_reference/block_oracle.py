"""Independent NumPy forward for an unchanged rational ChiTransformerBlock.

Read-only source specification: xvla/nn/block.py:91–97, attention.py:149–210,
bilinear.py:64–65, and normalization.py:307–329. No model, exporter, or ODT
implementation is imported. Inputs are already at the block boundary, so any
positional addition belongs outside this function. This is not a whole VLA.
"""

import numpy as np


def _array(value):
    if np.iscomplexobj(value):
        raise ValueError("real arrays required")
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("finite arrays required")
    return value


def _scalar(value):
    value = _array(value)
    if value.ndim != 0:
        raise ValueError("scalar buffer required")
    return float(value)


def _affine(value, weights, name, *, optional_bias=False):
    matrix = _array(weights[name + ".weight"])
    if matrix.ndim != 2 or matrix.shape[1] != value.shape[-1]:
        raise ValueError(f"{name} input dimensions disagree")
    bias = weights.get(name + ".bias", np.zeros(matrix.shape[0])) if optional_bias else weights[name + ".bias"]
    bias = _array(bias)
    if bias.shape != (matrix.shape[0],):
        raise ValueError(f"{name} bias dimensions disagree")
    return value @ matrix.T + bias


def _pade(value, weights, name, eps):
    if _scalar(weights[name + ".initialized"]) != 1.0:
        raise ValueError(f"{name} must be initialized before export")
    scale = max(_scalar(weights[name + ".running_ms"]), 1e-12)
    pa, pb = _array(weights[name + ".pa"]), _array(weights[name + ".pb"])
    if pa.shape != (3,) or pb.shape != (3,) or pb[0] == 0.0:
        raise ValueError("fixed degree-two numerator/denominator buffers required")
    v = (np.mean(value * value, axis=-1, keepdims=True) + eps) / scale
    numerator = pa[0] + pa[1]*v + pa[2]*v**2
    denominator = pb[0] + pb[1]*v + pb[2]*v**2
    return value * ((numerator / denominator) / np.sqrt(scale))


def block_forward(raw, weights, *, n_heads, mask, eps=1e-6, return_trace=False):
    """Evaluate all tokens using flat block state-dict keys, in float64.

    Includes six Padé norms, all projection biases, bilinear attention with
    head-dimension-squared score scaling and inverse-sqrt visible-key scaling,
    the CP FFN, and both gain-weighted residuals. Weights are never changed or
    fitted. Trace entries ``attn/pre_residual`` and ``ffn`` include their gains.
    The optional FFN down bias is supported. Numerical poles fail closed.
    """
    raw, mask = _array(raw), _array(mask)
    if raw.ndim != 3 or 0 in raw.shape:
        raise ValueError("raw must have nonempty shape (batch, tokens, width)")
    batch, tokens, width = raw.shape
    if isinstance(n_heads, bool) or not isinstance(n_heads, (int, np.integer)) or n_heads <= 0 or width % n_heads:
        raise ValueError("n_heads must be a positive integer dividing width")
    if mask.shape != (tokens, tokens) or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must be a fixed binary token-by-token matrix")
    eps = _scalar(eps)
    if eps < 0:
        raise ValueError("eps must be nonnegative")
    head_width = width // n_heads
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        pre = _pade(raw, weights, "rbn_attn", eps)
        projected = {}
        for branch in ("q1", "k1", "q2", "k2", "v"):
            value = _affine(pre, weights, "attn.w" + branch)
            if value.shape[-1] != width:
                raise ValueError("each attention branch must retain model width")
            value = value.reshape(batch, tokens, n_heads, head_width).transpose(0, 2, 1, 3)
            projected[branch] = value if branch == "v" else _pade(value, weights, "attn.rn_" + branch, eps)
        score1 = np.einsum("bhid,bhjd->bhij", projected["q1"], projected["k1"])
        score2 = np.einsum("bhid,bhjd->bhij", projected["q2"], projected["k2"])
        pattern = (score1 * score2) * mask / float(head_width**2)
        attended = np.einsum("bhij,bhjd->bhid", pattern, projected["v"])
        attended /= np.sqrt(np.maximum(mask.sum(axis=-1), 1.0))[None, None, :, None]
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, tokens, width)
        attention = _scalar(weights["attn_gain"]) * _affine(attended, weights, "attn.wo")
        if attention.shape != raw.shape:
            raise ValueError("attention output must retain model width")
        residual1 = raw + attention
        ffn_input = _pade(residual1, weights, "rbn_ffn", eps)
        left, right = (_affine(ffn_input, weights, "ffn." + side) for side in ("left", "right"))
        if left.shape != right.shape:
            raise ValueError("FFN factors must have the same CP width")
        ffn = _scalar(weights["ffn_gain"]) * _affine(left * right, weights, "ffn.down", optional_bias=True)
        if ffn.shape != raw.shape:
            raise ValueError("FFN output must retain model width")
        output = _array(residual1 + ffn)
    if not return_trace:
        return output
    trace = {"rbn_attn": pre, **projected, "attn/pattern": pattern,
             "attn/pre_residual": attention, "residual1": residual1,
             "rbn_ffn": ffn_input, "ffn": ffn, "output": output}
    return output, trace
