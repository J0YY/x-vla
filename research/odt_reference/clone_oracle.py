"""Independent bounded clone oracle, kept separate from the shared kernel."""
from dataclasses import dataclass
import itertools
import math
import numpy as np
from research.odt_reference.shared_dag import real

@dataclass
class Clone:
    origin: int
    path: tuple
    core: np.ndarray
    children: tuple
    source: str | None
    tied_children: bool


@dataclass
class CloneTree:
    root: Clone
    head: np.ndarray


def unfold_no_memo(graph):
    """Independent arrays and nodes at EVERY tree occurrence, never memoized."""
    def clone(index, path):
        node = graph.nodes[index]
        return Clone(index, path, node.core.copy(), tuple(clone(child, path + (role,))
            for role, child in enumerate(node.children)), node.source,
            len(node.children) == 2 and node.children[0] == node.children[1])
    return CloneTree(clone(len(graph.nodes) - 1, ()), graph.head.copy())


def walk_clones(tree):
    def walk(node):
        yield node
        for child in node.children:
            yield from walk(child)
    return tuple(walk(tree.root))


def clone_canonical_step(tree, origin):
    """Oracle RQ duplicated independently, including symmetric coordinates.

    Each matching occurrence is factorized separately. It does not call the
    shared canonical_step, rq, symmetric_rq, or absorb_axis helpers.
    """
    def step(node, parent=None, role=None):
        if node.origin == origin:
            original = node.core
            if node.tied_children:
                width = original.shape[1]
                scale = max(float(np.max(np.abs(original))), np.finfo(float).tiny)
                if width != original.shape[2] or np.max(np.abs(original - original.swapaxes(1, 2))) > 1e-12 * scale:
                    raise ValueError("oracle needs the same explicitly symmetric lift")
                columns = []
                for i in range(width):
                    for j in range(i, width):
                        average = original[:, i, j] / 2 + original[:, j, i] / 2
                        columns.append(average * (1 if i == j else np.sqrt(2.)))
                packed = np.stack(columns, axis=1)
                columns_q, upper = np.linalg.qr(packed.T, mode="reduced")
                rows = columns_q.T
                q = np.zeros((rows.shape[0], width, width))
                cursor = 0
                for i in range(width):
                    for j in range(i, width):
                        q[:, i, j] = q[:, j, i] = rows[:, cursor] / (1 if i == j else np.sqrt(2.))
                        cursor += 1
            else:
                columns_q, upper = np.linalg.qr(original.reshape(original.shape[0], -1).T, mode="reduced")
                q = columns_q.T.reshape((columns_q.shape[1],) + original.shape[1:])
            node.core = q
            if parent is None:
                tree.head = tree.head @ upper.T
            else:
                axis = role + 1
                # Independent axis contraction from the shared implementation.
                moved = np.moveaxis(parent.core, axis, -1)
                updated = moved.reshape(-1, moved.shape[-1]) @ upper.T
                parent.core = np.moveaxis(updated.reshape(moved.shape[:-1] + (upper.shape[0],)), -1, axis)
        for child_role, child in enumerate(node.children):
            step(child, node, child_role)
    step(tree.root)


def clone_coefficients(tree, *, cut=None, replacements=None, maximum_elements=1_000_000):
    """Materialize a bounded ordered-leaf tensor, recursively without memoization.

    `cut` replaces one subtree by an open coordinate leg. `replacements` changes
    explicit occurrence cores and is used for intervention counterexamples.
    """
    if type(maximum_elements) is not int or maximum_elements < 1:
        raise ValueError("positive integer coefficient-array size limit required")
    def check_size(size):
        if size > maximum_elements:
            raise ValueError("bounded coefficient oracle would exceed its explicit size limit")
    replacements = {} if replacements is None else replacements
    def dense(node):
        if node.path == cut:
            dimension = node.core.shape[0]
            check_size(dimension * dimension)
            return np.eye(dimension), (("cut", node.path),)
        core = replacements.get(node.path, node.core)
        if not node.children:
            check_size(core.size)
            return core.copy(), (("physical", node.path),)
        left, left_labels = dense(node.children[0])
        check_size(core.shape[0] * math.prod(core.shape[2:]) * math.prod(left.shape[1:]))
        if len(node.children) == 1:
            result = np.tensordot(core, left, axes=(1, 0))
            labels = left_labels
        else:
            right, right_labels = dense(node.children[1])
            check_size(core.shape[0] * math.prod(left.shape[1:]) * math.prod(right.shape[1:]))
            result = np.tensordot(np.tensordot(core, left, axes=(1, 0)), right, axes=(1, 0))
            labels = left_labels + right_labels
        return result, labels
    result, labels = dense(tree.root)
    check_size(tree.head.shape[0] * math.prod(result.shape[1:]))
    return np.tensordot(tree.head, result, axes=(1, 0)), labels


def clone_occurrence_environments(tree):
    """Literal two-copy downstream contraction, no opened context-map array.

    Both copies recursively evaluate each sibling subtree independently, without
    memoization or isometry cancellation. Only external indices are matched.
    """
    result = {}
    leaves = [node for node in walk_clones(tree) if not node.children]
    for node in walk_clones(tree):
        width = node.core.shape[0]
        outside = [leaf for leaf in leaves if leaf.path[:len(node.path)] != node.path]
        def downstream(current, coordinate, physical_indices):
            if current.path == node.path:
                value = np.zeros(width)
                value[coordinate] = 1.
                return value
            if not current.children:
                return current.core[:, physical_indices[current.path]]
            left = downstream(current.children[0], coordinate, physical_indices)
            if len(current.children) == 1:
                return current.core @ left
            right = downstream(current.children[1], coordinate, physical_indices)
            return np.einsum("oij,i,j->o", current.core, left, right)
        environment = np.zeros((width, width))
        for i, j in itertools.product(range(width), repeat=2):
            for output in range(tree.head.shape[0]):
                for indices in itertools.product(*(range(leaf.core.shape[1]) for leaf in outside)):
                    physical = {leaf.path: index for leaf, index in zip(outside, indices)}
                    ket = tree.head[output] @ downstream(tree.root, i, physical)
                    bra = tree.head[output] @ downstream(tree.root, j, physical)
                    environment[i, j] += ket * bra
        result[node.path] = (node.origin, environment)
    return result


def clone_environment_sums(tree, node_count):
    result = [None] * node_count
    for origin, environment in clone_occurrence_environments(tree).values():
        result[origin] = environment.copy() if result[origin] is None else result[origin] + environment
    if any(value is None for value in result):
        raise ValueError("clone origin inventory differs")
    return tuple(result)


def project_one_occurrence(tree, path, basis):
    """Insert a subspace at one explicit cut through adjacent contractions."""
    basis = real(basis, 2)
    for node in walk_clones(tree):
        if node.path == path:
            node.core = np.tensordot(basis.T, node.core, axes=(1, 0))
            if path == ():
                tree.head = tree.head @ basis
            else:
                parent = next(item for item in walk_clones(tree) if item.path == path[:-1])
                axis = path[-1] + 1
                moved = np.moveaxis(parent.core, axis, -1)
                updated = moved.reshape(-1, moved.shape[-1]) @ basis
                parent.core = np.moveaxis(updated.reshape(moved.shape[:-1] + (basis.shape[1],)), -1, axis)
            return
    raise ValueError("cut occurrence does not exist")
