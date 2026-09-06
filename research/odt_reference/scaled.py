"""Binary scale bookkeeping outside the direct-QR mathematical core.

The represented ordered tensor is ``2**G * contraction(stored_network)``.
G is a Python integer, never materialized during replay or environment EVD.
This protects common scales, not arbitrary within-tensor dynamic range.
"""
import math

import numpy as np

from research.odt_reference.clone_oracle import (
    unfold_no_memo, walk_clones, clone_canonical_step,
)
from research.odt_reference.clone_passes import CanonicalClones, clone_node_count, _finite_tree


def balanced(tensor):
    """Return a finite float64 mantissa and exact binary exponent.

    Nonzero mantissas have maximum absolute entry in [0.5, 1). A round trip
    must preserve EVERY entry exactly, including tiny nonzero coefficients.
    """
    value = np.asarray(tensor)
    if value.dtype != np.float64 or not value.size or not np.isfinite(value).all():
        raise ValueError("binary balancing needs a finite nonempty float64 tensor")
    maximum = float(np.max(np.abs(value)))
    shift = int(np.frexp(maximum)[1]) if maximum else 0
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        mantissa = np.ldexp(value, -shift)
        restored = np.ldexp(mantissa, shift)
    if not np.isfinite(mantissa).all() or not np.array_equal(restored, value):
        raise ValueError("binary balancing cannot preserve every tensor entry")
    return mantissa, shift


def canonicalize_scaled_clones(graph, *, maximum_nodes=100_000,
                               maximum_bytes=256 * 1024**2, after_step=None):
    """Independent clone QR with one separately measured scale per occurrence.

    Callback(tree, origin, G) runs after EVERY origin, before final head
    balancing. No shared occurrence counts or shared balancing helper are used.
    Factor absorption and explicit clone QR remain in clone_canonical_step.
    """
    count = clone_node_count(graph, maximum_nodes, maximum_bytes)
    for node in graph.nodes:
        if len(node.children) == 2 and node.children[0] == node.children[1]:
            scale = max(float(np.max(np.abs(node.core))), np.finfo(float).tiny)
            if np.max(np.abs(node.core - node.core.swapaxes(1, 2))) > 1e-12 * scale:
                raise ValueError("symmetrize raw tied cores before any child QR")
    tree, exponent = unfold_no_memo(graph), 0
    _finite_tree(tree)

    def independently_balance(value):
        if not value.size or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError("nonfinite or invalid explicit-clone tensor")
        maximum = float(np.max(np.abs(value)))
        shift = int(math.frexp(maximum)[1]) if maximum != 0. else 0
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            scaled = np.ldexp(value, -shift)
            reconstruction = np.ldexp(scaled, shift)
        if not np.isfinite(scaled).all() or not np.array_equal(reconstruction, value):
            raise ValueError("explicit-clone binary scale loses tensor entries")
        return scaled, shift

    origins = tuple(range(len(graph.nodes)))
    for origin in origins:
        for node in walk_clones(tree):
            if node.origin == origin:
                node.core, shift = independently_balance(node.core)
                exponent += shift
        clone_canonical_step(tree, origin)
        _finite_tree(tree)
        if after_step is not None:
            after_step(tree, origin, exponent)
    tree.head, shift = independently_balance(tree.head)
    exponent += shift
    _finite_tree(tree)
    return CanonicalClones(tree, count, origins), exponent
