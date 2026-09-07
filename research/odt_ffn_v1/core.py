"""One fixed ordered lift of the trained tokenwise residual FFN.

Input and output use numerator coordinates followed by one denominator.
This module does not construct a native model or a whole-policy environment.
"""
import numpy as np

from research.odt_reference.block import Builder
from research.odt_reference.shared_dag import Graph, real
from research.odt_reference.weights import symmetric_cp_ffn


def checked_weights(weights):
    weights = {key: real(value) for key, value in weights.items()}
    left = weights["ffn.left.weight"]
    right = weights["ffn.right.weight"]
    down = weights["ffn.down.weight"]
    if left.ndim != 2 or not all(left.shape):
        raise ValueError("nonempty two-dimensional FFN left factor required")
    rank, width = left.shape
    if (right.shape != (rank, width) or down.shape != (width, rank)
            or weights["ffn.left.bias"].shape != (rank,)
            or weights["ffn.right.bias"].shape != (rank,)
            or weights.get("ffn.down.bias", np.zeros(width)).shape != (width,)
            or weights["ffn_gain"].shape != ()):
        raise ValueError("FFN dimensions or gain disagree")
    name = "rbn_ffn"
    if (weights[name + ".running_ms"].shape != ()
            or weights[name + ".initialized"].shape != ()
            or float(weights[name + ".initialized"]) != 1.
            or weights[name + ".pa"].shape != (3,)
            or weights[name + ".pb"].shape != (3,)
            or weights[name + ".pb"][0] <= 0.
            or np.any(weights[name + ".pb"][1:] < 0.)):
        raise ValueError("initialized fixed pole-free quadratic Pade buffers required")
    return weights, width


def compile_ffn(weights, *, eps=1e-6, maximum_elements=16_000_000):
    weights, width = checked_weights(weights)
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("finite nonnegative epsilon required")
    builder = Builder(maximum_elements)
    builder.check_shape((width + 1, width + 1, width + 1))
    source = builder.put("input", np.eye(width + 1), source="token0")
    normalized = builder.pade(source, weights, "rbn_ffn", eps)
    gain = float(weights["ffn_gain"])
    core = symmetric_cp_ffn(
        weights["ffn.left.weight"], weights["ffn.right.weight"],
        gain * weights["ffn.down.weight"], weights["ffn.left.bias"],
        weights["ffn.right.bias"], gain * weights.get("ffn.down.bias", np.zeros(width)))
    order = list(range(1, width + 1)) + [0]
    core = core[np.ix_(order, order, order)]
    ffn = builder.put("ffn", core, (normalized, normalized))
    output = builder.combine(source, ffn, "residual_ffn")
    if (len(builder.nodes) != 6 or output != 5 or ffn != 4
            or tuple(node.children for node in builder.nodes) !=
            ((), (0, 0), (1, 1), (0, 2), (3, 3), (0, 4))):
        raise ValueError("residual FFN topology changed")
    return Graph(builder.nodes, np.eye(width + 1))


def ffn_forward(raw, weights, *, eps=1e-6):
    """Independent direct factor evaluation, with no builder or graph calls."""
    raw = real(raw, 3)
    weights, width = checked_weights(weights)
    if raw.shape[1:] != (1, width) or not len(raw):
        raise ValueError("FFN oracle needs [batch,1,width] residual inputs")
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("finite nonnegative epsilon required")
    scale = max(float(weights["rbn_ffn.running_ms"]), 1e-12)
    pa, pb = weights["rbn_ffn.pa"], weights["rbn_ffn.pb"]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        v = (np.mean(raw * raw, axis=-1, keepdims=True) + eps) / scale
        numerator = pa[0] + pa[1] * v + pa[2] * v * v
        denominator = pb[0] + pb[1] * v + pb[2] * v * v
        normalized = raw * (numerator / denominator) / np.sqrt(scale)
        left = normalized @ weights["ffn.left.weight"].T + weights["ffn.left.bias"]
        right = normalized @ weights["ffn.right.weight"].T + weights["ffn.right.bias"]
        value = (left * right) @ weights["ffn.down.weight"].T
        value += weights.get("ffn.down.bias", np.zeros(width))
        result = raw + float(weights["ffn_gain"]) * value
    if not np.isfinite(result).all():
        raise ValueError("nonfinite native-factor FFN output")
    return result[:, 0]
