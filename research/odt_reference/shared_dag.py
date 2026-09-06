"""Bounded shared-DAG reference for a FIXED ordered-leaf tensor lift.

The aggregated environment is a sum over clone occurrences, not a metric for
simultaneous tied interventions. See SHARED_DAG.md for the exact assumptions.
Only direct QR/RQ and eigendecomposition of contracted environments are used.
"""
from dataclasses import dataclass

import numpy as np

from research.odt_reference.dooms import rq, symmetric_rq, eigenspaces


def real(value, ndim=None):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("real tensors required")
    result = np.array(raw, dtype=np.float64, copy=True)
    if (ndim is not None and result.ndim != ndim) or not result.size or not np.isfinite(result).all():
        raise ValueError("finite nonempty tensor required")
    return result


@dataclass
class Node:
    name: str
    core: np.ndarray
    children: tuple = ()
    source: str | None = None


@dataclass
class Graph:
    nodes: list
    head: np.ndarray
    canonical: bool = False

    def __post_init__(self):
        if not self.nodes:
            raise ValueError("nonempty graph required")
        self.head = real(self.head, 2)
        names, source_dimensions = set(), {}
        for index, node in enumerate(self.nodes):
            if not isinstance(node.name, str) or node.name in names:
                raise ValueError("unique node names required")
            names.add(node.name)
            node.children = tuple(node.children)
            if len(node.children) > 2 or any(type(child) is not int or not 0 <= child < index for child in node.children):
                raise ValueError("nodes must be a unary/binary child-before-parent DAG")
            node.core = real(node.core, 3 if len(node.children) == 2 else 2)
            if 0 in node.core.shape:
                raise ValueError("nonempty dimensions required")
            if not node.children:
                if not isinstance(node.source, str):
                    raise ValueError("each leaf needs an explicit physical source")
                previous = source_dimensions.setdefault(node.source, node.core.shape[1])
                if previous != node.core.shape[1]:
                    raise ValueError("one physical source has incompatible dimensions")
            elif node.source is not None:
                raise ValueError("internal node cannot declare a physical source")
            for role, child in enumerate(node.children):
                if node.core.shape[role + 1] != self.nodes[child].core.shape[0]:
                    raise ValueError("connected dimensions differ")
        if self.head.shape[1] != self.nodes[-1].core.shape[0]:
            raise ValueError("head dimension differs")
        seen = set()
        def visit(index):
            seen.add(index)
            for child in self.nodes[index].children:
                if child not in seen:
                    visit(child)
        visit(len(self.nodes) - 1)
        if len(seen) != len(self.nodes):
            raise ValueError("unreachable nodes are outside this reference")

    def copy(self):
        return Graph([Node(n.name, n.core.copy(), n.children, n.source) for n in self.nodes],
                     self.head.copy(), self.canonical)


def evaluate(graph, inputs):
    values = []
    for node in graph.nodes:
        if not node.children:
            source = real(inputs[node.source], 1)
            if source.shape != (node.core.shape[1],):
                raise ValueError("physical input dimension differs")
            value = node.core @ source
        elif len(node.children) == 1:
            value = node.core @ values[node.children[0]]
        else:
            value = np.einsum("oij,i,j->o", node.core, values[node.children[0]], values[node.children[1]])
        values.append(value)
    return graph.head @ values[-1]


def symmetrize_tied(graph):
    """Explicit preprocessing: preserves tied-input evaluation, NOT the lift."""
    result = graph.copy()
    result.canonical = False
    for node in result.nodes:
        if len(node.children) == 2 and node.children[0] == node.children[1]:
            node.core = node.core / 2 + node.core.swapaxes(1, 2) / 2
    return result


def absorb_axis(core, role, factor):
    """Contract old-index x new-index into one syntactic input occurrence."""
    axis = role + 1
    return np.moveaxis(np.tensordot(core, factor, axes=(axis, 0)), -1, axis)


def validate_raw_tied_symmetry(graph):
    """Reject original skew before any shape-reducing factor can erase it."""
    for node in graph.nodes:
        if len(node.children) == 2 and node.children[0] == node.children[1]:
            scale = max(float(np.max(np.abs(node.core))), np.finfo(float).tiny)
            if np.max(np.abs(node.core - node.core.swapaxes(1, 2))) > 1e-12 * scale:
                raise ValueError("explicitly symmetrize every raw tied-child core before ODT")


def canonical_step(graph, index):
    """One direct RQ, with its R pushed into EVERY parent-role occurrence."""
    if index == 0:
        validate_raw_tied_symmetry(graph)
    node = graph.nodes[index]
    old_shape = node.core.shape
    if len(node.children) == 2 and node.children[0] == node.children[1]:
        factor, core = symmetric_rq(node.core)
    else:
        factor, rows = rq(node.core.reshape(old_shape[0], -1))
        core = rows.reshape((rows.shape[0],) + old_shape[1:])
    error = float(np.max(np.abs(np.tensordot(factor, core, axes=(1, 0)) - node.core)))
    node.core = core
    occurrences = 0
    for parent in graph.nodes:
        for role, child in enumerate(parent.children):
            if child == index:
                parent.core = absorb_axis(parent.core, role, factor)
                occurrences += 1
    if index == len(graph.nodes) - 1:
        graph.head = graph.head @ factor
        occurrences += 1
    graph.canonical = False
    return {"node": index, "old_shape": old_shape, "new_shape": core.shape,
            "occurrences_updated": occurrences, "reconstruction_max_error": error}


def canonicalize(graph, callback=None):
    validate_raw_tied_symmetry(graph)
    result = graph.copy()
    for index in range(len(result.nodes)):
        record = canonical_step(result, index)
        if callback is not None:
            callback(result, index, record)
    result.canonical = True
    return result


def occurrence_counts(graph):
    counts = [0] * len(graph.nodes)
    counts[-1] = 1
    for index in reversed(range(len(graph.nodes))):
        for child in graph.nodes[index].children:
            counts[child] += counts[index]
    return tuple(counts)


def occurrence_environment_sums(graph):
    """Explicit downstream double-layer recurrence for canonical sibling trees.

    Each entry sums all independently cut clone occurrences of this unique bond.
    In particular a repeated child contributes BOTH role contractions.
    """
    if not graph.canonical:
        raise ValueError("the collapsed-sibling recurrence requires canonical cores")
    result = [np.zeros((node.core.shape[0], node.core.shape[0])) for node in graph.nodes]
    result[-1] = np.einsum("oa,ob->ab", graph.head, graph.head)
    for index in reversed(range(len(graph.nodes))):
        node, environment = graph.nodes[index], result[index]
        if len(node.children) == 1:
            message = np.einsum("oa,op,pb->ab", node.core, environment, node.core)
            result[node.children[0]] += message
        elif len(node.children) == 2:
            left = np.einsum("oaj,op,pbj->ab", node.core, environment, node.core)
            right = np.einsum("oia,op,pib->ab", node.core, environment, node.core)
            result[node.children[0]] += left
            result[node.children[1]] += right
    return tuple(result)


def common_bases(graph, ranks=None):
    return eigenspaces(occurrence_environment_sums(graph), ranks)


def apply_bases(graph, bases):
    """Every-occurrence full-rank gauge or a physical common-subspace truncation.

    Bases are orthonormal columns produced by common_bases. Full-rank gauges
    preserve the entire ordered coefficient tensor. Truncations need not.
    """
    bases = tuple(real(value, 2) for value in bases)
    if len(bases) != len(graph.nodes):
        raise ValueError("one common basis required per unique output bond")
    result = graph.copy()
    for index, original in enumerate(graph.nodes):
        if bases[index].shape[0] != original.core.shape[0]:
            raise ValueError("basis input dimension differs")
        core = np.tensordot(bases[index].T, original.core, axes=(1, 0))
        for role, child in enumerate(original.children):
            core = absorb_axis(core, role, bases[child])
        result.nodes[index].core = core
    result.head = graph.head @ bases[-1]
    result.canonical = graph.canonical and all(value.shape[0] == value.shape[1] for value in bases)
    return result
