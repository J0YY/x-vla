"""Independent dense weight export for the symmetric cores assumed in Dooms §2.

This is deliberately not a VLA compiler. All arrays are real, and coordinate zero
is the homogeneous constant. No production model or decomposition code is used.
"""

import numpy as np


def _real(value):
    if np.iscomplexobj(value):
        raise ValueError("real weights required")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("weights and biases must be finite")
    return result


def homogeneous_affine(weight, bias=None):
    """Return E with E @ [1, x] = [1, weight @ x + bias]."""
    weight = _real(weight)
    if weight.ndim != 2:
        raise ValueError("weight must have shape (output, input)")
    bias = np.zeros(weight.shape[0]) if bias is None else _real(bias)
    if bias.shape != (weight.shape[0],):
        raise ValueError("bias must have shape (output,)")
    result = np.zeros((weight.shape[0] + 1, weight.shape[1] + 1))
    result[0, 0] = 1.0
    result[1:, 0], result[1:, 1:] = bias, weight
    if not np.isfinite(result).all():
        raise ValueError("weights and biases must be finite")
    return result


def symmetric_cp_ffn(left, right, output, left_bias=None, right_bias=None,
                     output_bias=None, residual=None):
    """Export C[o,i,j] for [1, U((Lx+bL)*(Rx+bR)) + bU + Wx].

    L,R have shape (CP width, input), U has shape (output, CP width).
    The optional residual W is any (output, input) linear map. C is symmetric
    in its two input axes, and contraction with two copies of [1,x] exactly
    reproduces the affine-biased FFN. This symmetry is local to a cloned input.
    """
    left, right, output = (_real(a) for a in (left, right, output))
    if left.ndim != 2 or right.shape != left.shape or output.ndim != 2:
        raise ValueError("left/right must match and output must be a matrix")
    width, inputs = left.shape
    outputs = output.shape[0]
    if output.shape[1] != width:
        raise ValueError("output CP width must match left/right")
    lb = np.zeros(width) if left_bias is None else _real(left_bias)
    rb = np.zeros(width) if right_bias is None else _real(right_bias)
    ob = np.zeros(outputs) if output_bias is None else _real(output_bias)
    residual = np.zeros((outputs, inputs)) if residual is None else _real(residual)
    if lb.shape != (width,) or rb.shape != (width,) or ob.shape != (outputs,):
        raise ValueError("bias shape does not match its output")
    if residual.shape != (outputs, inputs):
        raise ValueError("residual must have shape (output, input)")
    l = np.column_stack((lb, left))
    r = np.column_stack((rb, right))
    ordered = np.einsum("or,ri,rj->oij", output, l, r)
    core = np.zeros((outputs + 1, inputs + 1, inputs + 1))
    core[0, 0, 0] = 1.0
    core[1:] = (ordered + ordered.swapaxes(1, 2)) / 2.0
    core[1:, 0, 0] += ob
    core[1:, 1:, 0] += residual / 2.0
    core[1:, 0, 1:] += residual / 2.0
    if not np.isfinite(core).all():
        raise ValueError("weights and biases must be finite")
    return core
