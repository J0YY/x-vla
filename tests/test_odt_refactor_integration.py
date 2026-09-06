"""Tiny independent coefficient/environment gates for the split production core.

The literal oracle below contracts both downstream copies inside external-index
loops. It never constructs an opened-context map or assumes canonical siblings.
Run only after the source audit and the split Torch/NumPy runtime guard.
"""

import itertools
import math
from unittest.mock import patch

import pytest
import torch

from xvla.nn.block import ChiTransformerBlock
from xvla.nn.normalization import RationalNorm
from xvla.train.odt_engine_v2 import core as algorithm
from xvla.train.odt_engine_v2.compiler import compile_implicit_projective_block_boundary
from xvla.train.odt_engine_v2.graph import _walk_unique, evaluate_boundary_quotient
from xvla.train.odt_engine_v2.oracles import _independent_dense_clone_rq
from xvla.train.odt_engine_v2.types import (
    CPBinaryCore, DenseCloneCore, EnvironmentRecord, ImplicitNode,
    ImplicitProjectiveDAG, MaterializationTelemetry, ReducedQRBinaryCore, UnaryCore,
)


DTYPE = torch.float64


@pytest.mark.parametrize("entries,message", [
    ([[1., 1.], [0., 1.]], "not symmetric"),
    ([[1., 0.], [0., -1.]], "indefinite"),
    ([[float("inf"), 0.], [0., 1.]], "finite"),
    ([[1., 0., 0.]], "square"),
])
def test_invalid_environment_is_rejected(entries, message):
    record = EnvironmentRecord(0, "invalid", torch.tensor(entries, dtype=DTYPE), 0, 0, 0)
    with pytest.raises(ValueError, match=message):
        algorithm._environment_eigensystem(record)


def test_large_finite_environment_symmetrization_does_not_overflow():
    matrix = torch.diag(torch.tensor([1e308, 8e307], dtype=DTYPE))
    record = EnvironmentRecord(0, "large", matrix, 0, 0, 0)
    values, vectors = algorithm._environment_eigensystem(record)
    assert bool(torch.isfinite(values).all() and torch.isfinite(vectors).all())
    assert torch.allclose(values / 1e308, torch.tensor([1., .8], dtype=DTYPE))


def _dense(core):
    """Literal local coefficients, independently including their recorded scale."""
    assert core.output_dimension * math.prod(core.input_dimensions) <= 5000
    if isinstance(core, CPBinaryCore):
        value = torch.einsum("or,ri,rj->oij", core.output_factor, core.left_factor, core.right_factor)
    elif isinstance(core, UnaryCore):
        value = core.matrix
    elif isinstance(core, ReducedQRBinaryCore):
        value = core.q_rows.reshape(core.output_dimension, *core.input_dimensions)
    else:
        value = core.tensor
    return value * math.ldexp(1.0, core.binary_exponent)


def _cp(tensor):
    tensor = torch.tensor(tensor, dtype=DTYPE)
    out, left, right = tensor.shape
    return CPBinaryCore(tensor.reshape(out, -1),
                        torch.eye(left, dtype=DTYPE).repeat_interleave(right, dim=0),
                        torch.eye(right, dtype=DTYPE).repeat(left, 1), "literal_fixture")


def _graph():
    # A deficient embedding, a genuinely repeated symmetric core, then diamond sharing.
    leaf = ImplicitNode(0, "leaf", UnaryCore(torch.tensor([[1., 0.], [2., 0.]], dtype=DTYPE),
                                             "embedding", 1), physical_token=0, origin_uid=0)
    repeated = ImplicitNode(1, "repeated", _cp([[[1., .25], [.25, 0.]],
                                               [[2., .5], [.5, 0.]],
                                               [[0., 0.], [0., 0.]]]), (leaf, leaf), origin_uid=1)
    left = ImplicitNode(2, "left", UnaryCore(torch.tensor([[1., .2, -.1], [.3, 1., .2]], dtype=DTYPE),
                                             "left"), (repeated,), origin_uid=2)
    right = ImplicitNode(3, "right", UnaryCore(torch.tensor([[.5, -.2, .3], [.4, .8, -.1]], dtype=DTYPE),
                                               "right"), (repeated,), origin_uid=3)
    root = ImplicitNode(4, "diamond", _cp([[[1., .2], [.3, .7]], [[.2, .1], [.4, 1.]]]),
                        (left, right), origin_uid=4)
    return ImplicitProjectiveDAG(root, torch.eye(2, dtype=DTYPE), 0, 1, 1, 0,
                                 torch.ones(1, 1, dtype=DTYPE))


def _occurrences(node, path=()):
    yield node, path
    for role, child in enumerate(node.children):
        yield from _occurrences(child, path + (role,))


def _clone(network):
    counter = itertools.count()

    def visit(node):
        return ImplicitNode(next(counter), node.label, node.core.clone(),
                            tuple(visit(child) for child in node.children),
                            node.physical_token, node.uid, node.physical_source_key)

    return ImplicitProjectiveDAG(visit(network.root), network.head.clone(),
                                 network.head_binary_exponent, 1, 1, 0, network.mask.clone())


def _head(network):
    return network.head * math.ldexp(1.0, network.head_binary_exponent)


def _coefficient(network, coordinates, weights, *, cut=None, direction=0):
    def visit(node, path):
        if path == cut:
            result = torch.zeros(node.output_dimension, dtype=DTYPE)
            result[direction] = 1
            return result
        tensor = weights[id(node)]
        if not node.children:
            return tensor[:, coordinates[path]]
        children = tuple(visit(child, path + (role,)) for role, child in enumerate(node.children))
        if len(children) == 1:
            return torch.einsum("oi,i->o", tensor, children[0])
        return torch.einsum("oij,i,j->o", tensor, children[0], children[1])

    return torch.einsum("oa,a->o", _head(network), visit(network.root, ()))


def _ordered_tensor(network):
    occurrences = tuple(_occurrences(network.root))
    weights = {id(node): _dense(node.core) for node, _ in occurrences}
    leaves = tuple((path, node.core.input_dimensions[0]) for node, path in occurrences if not node.children)
    assert math.prod(size for _, size in leaves) <= 32
    return torch.stack([_coefficient(network, dict(zip((path for path, _ in leaves), indices)), weights)
                        for indices in itertools.product(*(range(size) for _, size in leaves))], dim=1)


def _literal_environments(network):
    occurrences = tuple(_occurrences(network.root))
    weights = {id(node): _dense(node.core) for node, _ in occurrences}
    result = {}
    for node, cut in occurrences:
        leaves = tuple((path, other.core.input_dimensions[0]) for other, path in occurrences
                       if not other.children and path[:len(cut)] != cut)
        environment = torch.zeros(node.output_dimension, node.output_dimension, dtype=DTYPE)
        for a, b in itertools.product(range(node.output_dimension), repeat=2):
            for indices in itertools.product(*(range(size) for _, size in leaves)):
                coordinates = dict(zip((path for path, _ in leaves), indices))
                # Two fresh contractions, with only output and external indices matched.
                first = _coefficient(network, coordinates, weights, cut=cut, direction=a)
                second = _coefficient(network, coordinates, weights, cut=cut, direction=b)
                for output in range(network.head.shape[0]):
                    environment[a, b] += first[output] * second[output]
        origin = node.uid if node.origin_uid is None else node.origin_uid
        result[origin] = result.get(origin, torch.zeros_like(environment)) + environment
    return result


def _absorb_clone(network, child, factor):
    parents = [(node, role) for node, _ in _occurrences(network.root)
               for role, candidate in enumerate(node.children) if candidate is child]
    for parent, role in parents:
        tensor = _dense(parent.core)
        if tensor.ndim == 2:
            transformed = torch.einsum("oi,ia->oa", tensor, factor)
        elif role == 0:
            transformed = torch.einsum("oij,ia->oaj", tensor, factor)
        else:
            transformed = torch.einsum("oij,ja->oia", tensor, factor)
        parent.core = DenseCloneCore(transformed, parent.core.kind, 0)
    if not parents:
        assert child is network.root
        network.head = _head(network) @ factor
        network.head_binary_exponent = 0


def _assert_same_lift_and_coordinates(shared, clone):
    torch.testing.assert_close(_ordered_tensor(shared), _ordered_tensor(clone), rtol=3e-10, atol=3e-10)
    unique = {node.uid: node for node in _walk_unique(shared.root)}
    for node, _ in _occurrences(clone.root):
        torch.testing.assert_close(_dense(unique[node.origin_uid].core), _dense(node.core),
                                   rtol=3e-10, atol=3e-10)
    torch.testing.assert_close(_head(shared), _head(clone), rtol=3e-10, atol=3e-10)
    shared_envs, clone_envs = _literal_environments(shared), _literal_environments(clone)
    assert shared_envs.keys() == clone_envs.keys()
    for uid in shared_envs:
        torch.testing.assert_close(shared_envs[uid], clone_envs[uid], rtol=3e-10, atol=3e-10)


def test_every_deficient_qr_and_gauge_prefix_matches_literal_clone():
    source = _graph()
    original = _ordered_tensor(source)
    clone = _clone(source)
    tied = {node.uid: len(node.children) == 2 and node.children[0] is node.children[1]
            for node in _walk_unique(source.root)}
    checked = []

    def after_step(shared, node, factor, record):
        for candidate, _ in tuple(_occurrences(clone.root)):
            if candidate.origin_uid != node.uid:
                continue
            r, q, diagnostics = _independent_dense_clone_rq(
                candidate.core, MaterializationTelemetry(), tied_inputs=tied[node.uid])
            assert diagnostics.literal_q_chart_resolved
            candidate.core = q
            _absorb_clone(clone, candidate, r.mantissa * math.ldexp(1., r.binary_exponent))
        assert record.parent_occurrences_pushed == record.expected_parent_occurrences
        assert record.scale_sensitive_occurrences_verified == record.expected_parent_occurrences
        assert record.literal_q_chart_resolved
        _assert_same_lift_and_coordinates(shared, clone)
        torch.testing.assert_close(_ordered_tensor(shared), original, rtol=3e-10, atol=3e-10)
        checked.append(node.uid)

    canonical = algorithm.canonicalize_implicit_dag_direct_rq(source, block_size=3, step_callback=after_step)
    assert checked == [node.uid for node in _walk_unique(source.root)]
    expected = _literal_environments(canonical.network)
    resolved = 0
    for record in algorithm.reverse_implicit_environments(canonical.network):
        actual = record.mantissa * math.ldexp(1., record.binary_exponent)
        torch.testing.assert_close(actual, expected[record.uid], rtol=3e-10, atol=3e-10)
        literal_record = EnvironmentRecord(record.uid, record.label, expected[record.uid], 0,
                                           record.incoming_occurrences, record.child_messages_emitted)
        _, vectors = algorithm._environment_eigensystem(record)
        values, literal_vectors = algorithm._environment_eigensystem(literal_record)
        if len(values) == 1 or float(values[0] - values[1]) > 1e-9 * max(1., float(values[0])):
            residual = vectors[:, 0] - literal_vectors[:, 0] * torch.sum(literal_vectors[:, 0] * vectors[:, 0])
            assert float(residual.abs().max()) < 3e-9
            resolved += 1
    assert resolved > 0
    original_apply = algorithm._apply_gauge
    gauges = []

    def checked_gauge(node, vectors, parents, shared):
        count = original_apply(node, vectors, parents, shared)
        for candidate, _ in tuple(_occurrences(clone.root)):
            if candidate.origin_uid == node.uid:
                transformed = torch.tensordot(vectors.T, _dense(candidate.core), dims=([1], [0]))
                candidate.core = DenseCloneCore(transformed, candidate.core.kind, 0)
                _absorb_clone(clone, candidate, vectors)
        _assert_same_lift_and_coordinates(shared, clone)
        gauges.append(node.uid)
        return count

    with patch.object(algorithm, "_apply_gauge", checked_gauge):
        diagonal = algorithm.diagonalize_implicit_dag_full_rank(canonical.network)
    assert len(gauges) == len(checked)
    torch.testing.assert_close(_ordered_tensor(diagonal.network), original, rtol=3e-10, atol=3e-10)


def test_full_block_export_uses_weight_derived_symmetric_ffn_and_all_six_norms():
    block = ChiTransformerBlock(2, 1, ffn_rank=2, norm="rational", qk_norm="rational").double().eval()
    block.ffn.down = torch.nn.Linear(2, 2, bias=True).double()
    with torch.no_grad():
        for index, module in enumerate(item for item in block.modules() if isinstance(item, torch.nn.Linear)):
            module.weight.copy_(torch.arange(module.weight.numel(), dtype=DTYPE).reshape_as(module.weight) * .013 + .07 + index * .009)
            if module.bias is not None:
                module.bias.copy_(torch.linspace(-.03, .04, module.bias.numel(), dtype=DTYPE) + index * .003)
        norms = tuple(item for item in block.modules() if isinstance(item, RationalNorm))
        assert len(norms) == 6
        for index, norm in enumerate(norms):
            norm.running_ms.fill_(.8 + index * .13)
            norm.initialized.fill_(True)
            norm.pa[0] *= 1 + index * .01
            norm.pb[1] *= 1 + index * .007
    positions = torch.tensor([[.02, -.03]], dtype=DTYPE)
    oracle = compile_implicit_projective_block_boundary(block, positions)
    fused = next(node for node in _walk_unique(oracle.network.root) if node.label == "ffn.fused_cp")
    left = torch.cat((block.ffn.left.weight, block.ffn.left.bias[:, None]), dim=1)
    right = torch.cat((block.ffn.right.weight, block.ffn.right.bias[:, None]), dim=1)
    expected = torch.zeros(3, 3, 3, dtype=DTYPE)
    for o, i, j, r in itertools.product(range(2), range(3), range(3), range(2)):
        expected[o, i, j] += block.ffn.down.weight[o, r] * (left[r, i] * right[r, j] + left[r, j] * right[r, i]) / 2
    expected[:2, -1, -1] += block.ffn.down.bias
    expected[-1, -1, -1] = 1
    torch.testing.assert_close(_dense(fused.core), expected, rtol=1e-13, atol=1e-13)
    for node in _walk_unique(oracle.network.root):
        if len(node.children) == 2 and node.children[0] is node.children[1]:
            tensor = _dense(node.core)
            torch.testing.assert_close(tensor, tensor.transpose(1, 2), rtol=1e-13, atol=1e-13)
    inputs = torch.tensor([[[-.1, .2]], [[.3, -.2]], [[0., 0.]]], dtype=DTYPE)
    torch.testing.assert_close(evaluate_boundary_quotient(oracle.network, inputs),
                               block(inputs + positions)[:, 0], rtol=3e-11, atol=3e-11)


def test_raw_skew_is_rejected_before_a_child_factor_can_hide_it():
    leaf = ImplicitNode(0, "narrow", UnaryCore(torch.tensor([[1., 0.], [0., 1.], [0., 0.]], dtype=DTYPE),
                                               "embedding"), physical_token=0)
    tensor = torch.zeros(2, 3, 3, dtype=DTYPE)
    tensor[0, 2, 0] = 1
    root = ImplicitNode(1, "hidden_skew", _cp(tensor.tolist()), (leaf, leaf))
    network = ImplicitProjectiveDAG(root, torch.eye(2, dtype=DTYPE), 0, 1, 1, 0,
                                    torch.ones(1, 1, dtype=DTYPE))
    with patch.object(torch.linalg, "qr", side_effect=AssertionError("QR must not run")):
        with pytest.raises(ValueError, match="asymmetric"):
            algorithm.canonicalize_implicit_dag_direct_rq(network)


@pytest.mark.parametrize("corruption", ["omitted_transport", "wrong_scale"])
def test_corrupted_occurrence_transport_fails_closed(corruption):
    source = _graph()
    original = algorithm._absorb_input_factor

    def corrupt(core, role, factor, **kwargs):
        if corruption == "omitted_transport":
            return core
        result = original(core, role, factor, **kwargs)
        result.binary_exponent += 1
        return result

    with patch.object(algorithm, "_absorb_input_factor", corrupt):
        with pytest.raises(RuntimeError, match="absorption failed"):
            algorithm.canonicalize_implicit_dag_direct_rq(source, copy_network=False)
    assert not source.algorithm1_direct_rq_complete
    assert not source.algorithm1_scale_ledger_complete


def test_v1_artifact_round_trip_preserves_classes_sharing_and_bytes(tmp_path):
    from xvla.train.implicit_sparse_projective_odt import ImplicitNode as FacadeNode
    from xvla.train.implicit_projective_dag_artifact import (
        ARTIFACT_SCHEMA, export_implicit_projective_dag_artifact, load_implicit_projective_dag_artifact,
    )

    assert FacadeNode is ImplicitNode
    assert ARTIFACT_SCHEMA == "xvla_implicit_projective_dag_artifact_v1"
    canonical = algorithm.canonicalize_implicit_dag_direct_rq(_graph()).network
    first = export_implicit_projective_dag_artifact(canonical, tmp_path / "first")
    loaded = load_implicit_projective_dag_artifact(first.root, expected_manifest_sha256=first.manifest_sha256)
    second = export_implicit_projective_dag_artifact(loaded, tmp_path / "second")
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.shard_sha256 == second.shard_sha256
    assert {path.name: path.read_bytes() for path in first.root.iterdir()} == {
        path.name: path.read_bytes() for path in second.root.iterdir()}
    assert isinstance(loaded.root, ImplicitNode)
    assert loaded.root.children[0].children[0] is loaded.root.children[1].children[0]
    torch.testing.assert_close(_ordered_tensor(loaded), _ordered_tensor(canonical), rtol=0, atol=0)
