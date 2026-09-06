"""Bounded weight-only export of a complete rational ChiTransformerBlock.

Coordinates are (numerator vector, denominator LAST). Every primitive below
declares its ordered tensor lift. Identical-child cores are symmetrized before
ODT as required by Dooms §2. No model, legacy compiler, or numerical fit imports.
The block boundary is already positioned. Masks and Padé buffers are fixed.
"""
import math
from dataclasses import dataclass
import numpy as np

from research.odt_reference.shared_dag import Graph, Node, real
from research.odt_reference.weights import symmetric_cp_ffn


NORM_SITES = ("rbn_attn", "attn.rn_q1", "attn.rn_k1", "attn.rn_q2", "attn.rn_k2", "rbn_ffn")


@dataclass(frozen=True)
class BlockSpec:
    """Explicit source semantics, not information inferred from a state_dict."""
    n_heads: int
    eps: float = 1e-6
    normalization: str = "pade[2/2]"
    residual: bool = True
    score_scale: str = "d_h2"
    row_scale: str = "invsqrt"

    def __post_init__(self):
        if (type(self.n_heads) is not int or self.n_heads < 1 or not np.isfinite(self.eps) or self.eps < 0
                or self.normalization != "pade[2/2]" or self.residual is not True
                or self.score_scale != "d_h2" or self.row_scale != "invsqrt"):
            raise ValueError("unsupported block source semantics")


def validate_weights(weights, width):
    for name in ("attn.wq1", "attn.wk1", "attn.wq2", "attn.wk2", "attn.wv", "attn.wo"):
        if weights[name + ".weight"].shape != (width, width) or weights[name + ".bias"].shape != (width,):
            raise ValueError("attention maps must exactly retain model width")
    left, right, down = (weights["ffn." + side + ".weight"] for side in ("left", "right", "down"))
    if (left.ndim != 2 or left.shape[1] != width or right.shape != left.shape
            or down.shape != (width, left.shape[0])
            or weights["ffn.left.bias"].shape != (left.shape[0],)
            or weights["ffn.right.bias"].shape != (left.shape[0],)
            or weights.get("ffn.down.bias", np.zeros(width)).shape != (width,)):
        raise ValueError("FFN input, output and CP dimensions must match exactly")
    if any(weights[name].shape != () for name in ("attn_gain", "ffn_gain")):
        raise ValueError("scalar residual gains required")
    for name in NORM_SITES:
        pa, pb = weights[name + ".pa"], weights[name + ".pb"]
        if (pa.shape != (3,) or pb.shape != (3,) or pb[0] <= 0 or np.any(pb[1:] < 0)
                or weights[name + ".running_ms"].shape != ()
                or weights[name + ".initialized"].shape != () or float(weights[name + ".initialized"]) != 1.):
            raise ValueError("all six sites need fixed quadratic, pole-free Padé buffers")


class Builder:
    """Only finite unary/binary tensors, with shape gates before allocation."""
    def __init__(self, maximum_elements=8_000_000):
        if type(maximum_elements) is not int or maximum_elements < 1:
            raise ValueError("positive local tensor element limit required")
        self.nodes, self.maximum_elements = [], maximum_elements

    def check_shape(self, shape):
        if math.prod(shape) > self.maximum_elements:
            raise ValueError(f"bounded export rejects tensor shape {shape}")

    def zeros(self, shape):
        self.check_shape(shape)
        return np.zeros(shape)

    def put(self, label, core, children=(), source=None):
        if core.size > self.maximum_elements:
            raise ValueError(f"bounded export rejects tensor shape {core.shape}")
        if len(children) == 2 and children[0] == children[1]:
            core = core / 2 + core.swapaxes(1, 2) / 2
        self.nodes.append(Node(f"{len(self.nodes)}:{label}", real(core), children, source))
        return len(self.nodes) - 1

    def width(self, node):
        return self.nodes[node].core.shape[0] - 1

    def affine(self, node, weight, bias, label):
        weight, bias = real(weight, 2), real(bias, 1)
        if weight.shape[1] != self.width(node) or bias.shape != (weight.shape[0],):
            raise ValueError("affine weight dimensions disagree")
        core = self.zeros((weight.shape[0] + 1, weight.shape[1] + 1))
        core[:-1, :-1], core[:-1, -1], core[-1, -1] = weight, bias, 1.
        return self.put(label, core, (node,))

    def product(self, left, right, coefficients, label):
        coefficients = real(coefficients, 3)
        if coefficients.shape[1:] != (self.width(left), self.width(right)):
            raise ValueError("bilinear input dimensions disagree")
        core = self.zeros(tuple(size + 1 for size in coefficients.shape))
        core[:-1, :-1, :-1], core[-1, -1, -1] = coefficients, 1.
        return self.put(label, core, (left, right))

    def combine(self, left, right, label, *, concatenate=False):
        a, b = self.width(left), self.width(right)
        if not concatenate and a != b:
            raise ValueError("addition needs equal widths")
        core = self.zeros(((a + b if concatenate else a) + 1, a + 1, b + 1))
        for i in range(a):
            core[i, i, -1] += 1.
        for j in range(b):
            core[a + j if concatenate else j, -1, j] += 1.
        core[-1, -1, -1] = 1.
        return self.put(label, core, (left, right))

    def pade(self, node, weights, name, eps):
        if float(weights[name + ".initialized"]) != 1.:
            raise ValueError("Padé buffers must be initialized and fixed")
        scale = max(float(weights[name + ".running_ms"]), 1e-12)
        pa, pb = real(weights[name + ".pa"], 1), real(weights[name + ".pb"], 1)
        if pa.shape != (3,) or pb.shape != (3,) or not np.isfinite(scale):
            raise ValueError("finite quadratic Padé buffers required")
        width = self.width(node)
        moment = self.zeros((2, width + 1, width + 1))
        moment[0, :-1, :-1] = np.eye(width) / width
        moment[0, -1, -1], moment[1, -1, -1] = eps, scale
        ratio = self.put(name + ":moment", moment, (node, node))
        polynomial = np.array([[[c[2], c[1] / 2], [c[1] / 2, c[0]]] for c in (pa, pb)])
        quotient = self.put(name + ":pade", polynomial, (ratio, ratio))
        return self.product(node, quotient, (np.eye(width) / np.sqrt(scale))[:, :, None], name)


def compile_block(weights, *, spec, mask, selected_token=None, maximum_elements=8_000_000):
    """Return the complete joint-output block as one fixed-lift DAG.

    All visible tokens and heads contribute. No trained dimensions are sliced.
    Default output order is token-major, with one common final denominator.
    Optional selected_token is a ROW-ONLY diagnostic, not the joint ODT metric.
    Empty mask rows retain output bias and
    both residuals exactly. Only the source's d_h²/invsqrt conventions are used.
    """
    if not isinstance(spec, BlockSpec):
        raise ValueError("explicit supported source BlockSpec required")
    n_heads, eps = spec.n_heads, spec.eps
    weights = {name: real(value) for name, value in weights.items()}
    mask = real(mask, 2)
    if weights["attn.wq1.weight"].ndim != 2:
        raise ValueError("attention weight must be a matrix")
    width = weights["attn.wq1.weight"].shape[1]
    validate_weights(weights, width)
    if type(n_heads) is not int or n_heads < 1 or width % n_heads:
        raise ValueError("positive head count dividing model width required")
    tokens = len(mask)
    if mask.shape != (tokens, tokens) or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("fixed square binary mask required")
    if selected_token is not None and (type(selected_token) is not int or not 0 <= selected_token < tokens):
        raise ValueError("selected output token is out of range")
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("finite nonnegative epsilon required")
    builder, cache = Builder(maximum_elements), {}
    builder.check_shape((width + 1, width + 1, width + 1))
    def source(token):
        key = ("input", token)
        if key not in cache:
            core = builder.zeros((width + 1, width + 1))
            np.fill_diagonal(core, 1.)
            cache[key] = builder.put("input", core, source=f"token{token}")
        return cache[key]
    def pre(token):
        if ("pre", token) not in cache:
            cache["pre", token] = builder.pade(source(token), weights, "rbn_attn", eps)
        return cache["pre", token]
    head_width = width // n_heads
    def branch(kind, token, head):
        key = (kind, token, head)
        if key not in cache:
            section = slice(head * head_width, (head + 1) * head_width)
            name = "attn.w" + kind
            node = builder.affine(pre(token), weights[name + ".weight"][section], weights[name + ".bias"][section], name)
            cache[key] = node if kind == "v" else builder.pade(node, weights, "attn.rn_" + kind, eps)
        return cache[key]
    # The independent weight exporter uses constant FIRST. Permute exactly once.
    builder.check_shape((width + 1, width + 1, width + 1))
    core = symmetric_cp_ffn(weights["ffn.left.weight"], weights["ffn.right.weight"],
                            float(weights["ffn_gain"]) * weights["ffn.down.weight"],
                            weights["ffn.left.bias"], weights["ffn.right.bias"],
                            float(weights["ffn_gain"]) * weights.get("ffn.down.bias", np.zeros(width)))
    order = list(range(1, width + 1)) + [0]
    core = core[np.ix_(order, order, order)]
    outputs = []
    for query in (range(tokens) if selected_token is None else (selected_token,)):
        visible = np.flatnonzero(mask[query]).tolist()
        heads = []
        for head in range(n_heads):
            routes = []
            for token in visible:
                scores = [builder.product(branch("q" + side, query, head), branch("k" + side, token, head),
                                          np.eye(head_width)[None], "dot" + side) for side in ("1", "2")]
                score = builder.product(*scores, np.ones((1, 1, 1)) / (head_width**2 * np.sqrt(len(visible))), "score")
                routes.append(builder.product(score, branch("v", token, head), np.eye(head_width)[:, None, :], "weighted_value"))
            if routes:
                value = routes[0]
                for other in routes[1:]:
                    value = builder.combine(value, other, "route_sum")
            else:
                value = builder.affine(source(query), np.zeros((head_width, width)), np.zeros(head_width), "empty_attention")
            heads.append(value)
        attended = heads[0]
        for other in heads[1:]:
            attended = builder.combine(attended, other, "heads", concatenate=True)
        gain = float(weights["attn_gain"])
        attention = builder.affine(attended, gain * weights["attn.wo.weight"], gain * weights["attn.wo.bias"], "attention")
        residual = builder.combine(source(query), attention, "residual1")
        normalized = builder.pade(residual, weights, "rbn_ffn", eps)
        ffn = builder.put("ffn", core, (normalized, normalized))
        outputs.append(builder.combine(residual, ffn, "residual2"))
    output = outputs[0]
    for other in outputs[1:]:
        output = builder.combine(output, other, "joint_output", concatenate=True)
    return Graph(builder.nodes, np.eye(builder.width(output) + 1))


def evaluate_block(graph, raw):
    """Decoded replay with per-node scalar rescaling, not an ODT metric.

    Homogeneity makes these sample-specific rescalings cancel on final decode.
    No weights or graph tensors change. ODT itself never uses this replay chart.
    """
    raw = real(raw, 3)
    result = []
    for sample in raw:
        values = []
        for node in graph.nodes:
            if not node.children:
                value = node.core @ np.r_[sample[int(node.source[5:])], 1.]
            elif len(node.children) == 1:
                value = node.core @ values[node.children[0]]
            else:
                value = np.einsum("oij,i,j->o", node.core, values[node.children[0]], values[node.children[1]])
            magnitude = float(np.max(np.abs(value)))
            if not np.isfinite(magnitude) or magnitude == 0.:
                raise ValueError("nonfinite or zero homogeneous replay chart")
            values.append(value / magnitude)
        output = graph.head @ values[-1]
        if output[-1] == 0. or not np.isfinite(output).all():
            raise ValueError("undefined projective output")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result.append(output[:-1] / output[-1])
    return np.array(result)
