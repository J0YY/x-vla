"""Explicit acceptance gates, including large finite error and preflight faults."""

from dataclasses import replace
from collections import Counter
from unittest.mock import patch

import pytest
import torch

from tests.test_odt_refactor_integration import _graph, _ordered_tensor
from xvla.train.odt_engine_v2 import acceptance, core, factorization


INPUTS = torch.tensor([[[-.2]], [[.3]]], dtype=torch.float64)


def test_acceptance_preserves_input_and_accepts_bounded_fixture():
    network = _graph()
    before = _ordered_tensor(network)
    diagonal, report = acceptance.accept_implicit_odt(network, INPUTS)
    assert report["routes"]["simulated_unique_node_count"] == 5
    assert report["scaled_maximum_decoded_action_error"] < 3e-8
    assert report["aggregate_offdiagonal_ratio"] < 3e-8
    assert not network.algorithm1_direct_rq_complete
    torch.testing.assert_close(_ordered_tensor(network), before, rtol=0, atol=0)
    torch.testing.assert_close(_ordered_tensor(diagonal.network), before, rtol=3e-10, atol=3e-10)


@pytest.mark.parametrize("error", [.01, float("inf"), float("nan")])
def test_bad_qr_replay_is_rejected_before_any_gauge(error):
    with patch.object(core, "_projective_batch_relative_error", return_value=error), \
            patch.object(acceptance, "diagonalize_implicit_dag_full_rank", side_effect=AssertionError("must not gauge")):
        with pytest.raises(RuntimeError):
            acceptance.accept_implicit_odt(_graph(), INPUTS)


@pytest.mark.parametrize("field", ["replay_relative_error", "maximum_recontracted_offdiagonal_ratio"])
def test_large_finite_post_gauge_diagnostic_is_rejected(field):
    original = acceptance.diagonalize_implicit_dag_full_rank

    def corrupt(*args, **kwargs):
        return replace(original(*args, **kwargs), **{field: .01})

    with patch.object(acceptance, "diagonalize_implicit_dag_full_rank", corrupt):
        with pytest.raises(RuntimeError, match="acceptance failed"):
            acceptance.accept_implicit_odt(_graph(), INPUTS)


def test_unsupported_later_shape_rejects_before_first_qr():
    # Leaf Q needs 4 entries, but the subsequent tied core needs 12, not 9.
    network = _graph()
    original = _ordered_tensor(network)
    with patch.object(factorization, "MAXIMUM_BOUNDED_EXPLICIT_Q_ELEMENTS", 11), \
            patch.object(torch.linalg, "qr", side_effect=AssertionError("QR must not run")):
        with pytest.raises(ValueError, match="unsupported retained direct-Q route"):
            acceptance.accept_implicit_odt(network, INPUTS)
    torch.testing.assert_close(_ordered_tensor(network), original, rtol=0, atol=0)


def test_predicted_shapes_and_storage_include_unary_and_deficient_nodes():
    network = _graph()
    inventory = acceptance.predict_direct_rq_route_inventory(network)
    canonical = core.canonicalize_implicit_dag_direct_rq(network)
    actual = Counter(f"{step.unfolding_shape[0]}x{step.unfolding_shape[1]}" for step in canonical.steps)
    assert inventory["all_node_unfolding_shape_counts"] == dict(actual)
    assert inventory["unary_node_count"] == 3
    assert inventory["all_node_retained_q_total_elements"] == 36


def test_final_replay_measures_state_after_last_callback():
    def change_final_head(network, node, factor, record):
        if node is network.root:
            network.head[0].mul_(2.)

    result = core.canonicalize_implicit_dag_direct_rq(_graph(), replay_inputs=INPUTS, step_callback=change_final_head)
    assert result.steps[-1].per_step_function_replay_error < 3e-8
    assert result.final_projective_replay_error > .01
    with pytest.raises(RuntimeError, match="final replay"):
        acceptance._require_error("final replay", result.final_projective_replay_error, 3e-8)


def test_later_nan_offdiagonal_cannot_disappear_in_maximum():
    canonical = core.canonicalize_implicit_dag_direct_rq(_graph()).network
    with patch.object(core, "_offdiagonal_ratio", side_effect=[0., float("nan"), 0., 0., 0.]):
        with pytest.raises(RuntimeError, match="nonfinite"):
            core.diagonalize_implicit_dag_full_rank(canonical, copy_network=False)
    assert not canonical.algorithm1_direct_rq_complete


@pytest.mark.parametrize("entries", [torch.zeros(2, 2), torch.tensor([[1., 0.], [1., 1.]]),
                                     torch.tensor([[1., 1e-14], [1., 1.]]), torch.full((2, 2), float("nan"))])
def test_zero_near_pole_and_nonfinite_charts_are_rejected(entries):
    with patch.object(acceptance, "evaluate_projective_boundary", return_value=entries):
        with pytest.raises((ValueError, RuntimeError)):
            acceptance.accept_implicit_odt(_graph(), INPUTS)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -.1, True, 1.])
def test_invalid_tolerances_fail_before_sweeps(value):
    with pytest.raises(ValueError):
        acceptance.AcceptanceTolerances(replay=value)


def test_missing_replay_is_not_a_zero_error_certificate():
    with pytest.raises(ValueError, match="explicit replay"):
        acceptance.accept_implicit_odt(_graph(), None)


def test_large_finite_decoded_action_error_is_rejected_separately():
    original = acceptance._decode_with_margin
    calls = []

    def corrupt_final(*args):
        decoded, margin = original(*args)
        calls.append(True)
        return (decoded + 1 if len(calls) == 2 else decoded), margin

    with patch.object(acceptance, "_decode_with_margin", corrupt_final):
        with pytest.raises(RuntimeError, match="decoded action error"):
            acceptance.accept_implicit_odt(_graph(), INPUTS)
