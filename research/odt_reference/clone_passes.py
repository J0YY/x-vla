"""Bounded Dooms passes on explicit clones, without physical tensor expansion.

Algorithm 2 eliminates sibling subtrees by the isometries produced by this
module's own direct QR pass. Tiny tests compare it to literal two-copy networks.
This is not an unfolded-map overlap diagnostic or a production implementation.
"""
import copy
from dataclasses import dataclass

import numpy as np

from research.odt_reference.clone_oracle import (
    clone_canonical_step, unfold_no_memo, walk_clones,
)


@dataclass(frozen=True)
class CanonicalClones:
    """Own-QR provenance, not an attestation of arbitrary mutable user arrays."""
    tree: object
    node_count: int
    origins: tuple


def _finite_tree(tree):
    if any(not np.isfinite(node.core).all() for node in walk_clones(tree)) or not np.isfinite(tree.head).all():
        raise ValueError("nonfinite explicit-clone state")


def clone_node_count(graph, maximum_nodes=100_000, maximum_bytes=256 * 1024**2):
    """Count occurrences from topology and reject before any clone allocation."""
    if type(maximum_nodes) is not int or maximum_nodes < 1:
        raise ValueError("positive integer explicit-clone node bound required")
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise ValueError("positive explicit-clone byte bound required")
    sizes, byte_sizes = [], []
    for index, node in enumerate(graph.nodes):
        if any(type(child) is not int or not 0 <= child < index for child in node.children):
            raise ValueError("child-before-parent topology required")
        size = 1 + sum(sizes[child] for child in node.children)
        byte_size = node.core.nbytes + sum(byte_sizes[child] for child in node.children)
        if size > maximum_nodes:
            raise ValueError(f"explicit clone has more than {maximum_nodes} nodes")
        if byte_size + graph.head.nbytes > maximum_bytes:
            raise ValueError(f"explicit clone requires at least {byte_size + graph.head.nbytes} tensor bytes, limit {maximum_bytes}")
        sizes.append(size)
        byte_sizes.append(byte_size)
    if not sizes:
        raise ValueError("nonempty graph required")
    return sizes[-1]


def canonicalize_clones(graph, *, maximum_nodes=100_000, maximum_bytes=256 * 1024**2, after_step=None):
    """Algorithm 1 independently factors EVERY clone, grouped by origin.

    Callback(tree, origin) is read-only and runs after all copies of that origin
    are updated. It can compare against a shared step without copying the tree.
    """
    count = clone_node_count(graph, maximum_nodes, maximum_bytes)
    for node in graph.nodes:
        if len(node.children) == 2 and node.children[0] == node.children[1]:
            scale = max(float(np.max(np.abs(node.core))), np.finfo(float).tiny)
            if np.max(np.abs(node.core - node.core.swapaxes(1, 2))) > 1e-12 * scale:
                raise ValueError("symmetrize raw tied cores before any child QR")
    tree = unfold_no_memo(graph)
    _finite_tree(tree)
    origins = tuple(range(len(graph.nodes)))
    for origin in origins:
        clone_canonical_step(tree, origin)
        _finite_tree(tree)
        if after_step is not None:
            after_step(tree, origin)
    return CanonicalClones(tree, count, origins)


def occurrence_environments(canonical):
    """Algorithm 2: independent one-occurrence downstream contractions.

    Every path gets its own environment. No shared-origin accumulation occurs
    during traversal. QR-certified sibling elimination is the Appendix G rule.
    """
    if not isinstance(canonical, CanonicalClones):
        raise ValueError("own direct-QR clone pass required first")
    result = {}
    head = canonical.tree.head
    def descend(node, environment):
        if not np.isfinite(environment).all():
            raise ValueError("nonfinite downstream clone environment")
        result[node.path] = (node.origin, environment)
        if not node.children:
            return
        # Independent two-stage contraction: attach this occurrence's actual
        # downstream environment before closing the QR-certified sibling leg.
        shape = node.core.shape
        attached = (environment @ node.core.reshape(shape[0], -1)).reshape(shape)
        if len(node.children) == 1:
            child = np.einsum("ai,aj->ij", node.core, attached, optimize=True)
            descend(node.children[0], child)
        elif len(node.children) == 2:
            left = np.einsum("aik,ajk->ij", node.core, attached, optimize=True)
            right = np.einsum("aki,akj->ij", node.core, attached, optimize=True)
            descend(node.children[0], left)
            descend(node.children[1], right)
    descend(canonical.tree.root, np.einsum("oi,oj->ij", head, head))
    return result


def aggregate_environments(canonical):
    """Group explicit cut environments only after their independent traversal."""
    sums = [None] * len(canonical.origins)
    for origin, environment in occurrence_environments(canonical).values():
        sums[origin] = environment.copy() if sums[origin] is None else sums[origin] + environment
    return tuple(sums)


def apply_common_bases(canonical, bases_by_origin, *, after_step=None):
    """Algorithm 3's common-origin gauge, independently updating every clone.

    Bases must be supplied by the environment EVD. Orthogonality is a caller
    precondition, NOT checked here. Only shape/type/finiteness are checked.
    This mathematical primitive is not a validator for arbitrary input bases
    and does not claim finite tied-loss optimality.
    Callback(tree, origin) is read-only. Full or reduced widths are supported.
    """
    if not isinstance(canonical, CanonicalClones):
        raise ValueError("own direct-QR clone pass required first")
    bases = tuple(np.asarray(value) for value in bases_by_origin)
    if len(bases) != len(canonical.origins):
        raise ValueError("one basis required for each origin")
    for node in walk_clones(canonical.tree):
        basis = bases[node.origin]
        if (np.iscomplexobj(basis) or basis.ndim != 2 or basis.shape[0] != node.core.shape[0]
                or not 1 <= basis.shape[1] <= basis.shape[0] or not np.isfinite(basis).all()):
            raise ValueError("finite real basis with compatible retained width required")
    tree = copy.deepcopy(canonical.tree)
    def absorb(node, origin, parent=None, role=None):
        if node.origin == origin:
            basis = bases[origin]
            shape = node.core.shape
            node.core = (basis.T @ node.core.reshape(shape[0], -1)).reshape((basis.shape[1],) + shape[1:])
            if parent is None:
                tree.head = tree.head @ basis
            else:
                moved = np.moveaxis(parent.core, role + 1, -1)
                updated = moved.reshape(-1, moved.shape[-1]) @ basis
                parent.core = np.moveaxis(updated.reshape(moved.shape[:-1] + (basis.shape[1],)), -1, role + 1)
        for child_role, child in enumerate(node.children):
            absorb(child, origin, node, child_role)
    for origin in canonical.origins:
        absorb(tree.root, origin)
        _finite_tree(tree)
        if after_step is not None:
            after_step(tree, origin)
    return tree
