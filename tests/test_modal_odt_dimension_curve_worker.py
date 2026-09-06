from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from modal_odt_dimension_curve_worker import PANEL, array_sha256, physical_inputs, predict


def test_fixed_balanced_panel_has_same_two_states_for_all_ten_tasks():
    assert PANEL == tuple((task, episode) for task in range(10) for episode in (0, 1))
    assert len(PANEL) == len(set(PANEL)) == 20


def test_array_identity_matches_capability_protocol_encoding():
    value = np.array([[1., 2.], [-3., 4.]], dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(b"float64\0")
    digest.update(json.dumps([2, 2]).encode())
    digest.update(b"\0")
    digest.update(value.tobytes())
    assert array_sha256(value) == digest.hexdigest()


def test_physical_patch_sources_preserve_spatial_channel_order():
    images = torch.arange(2 * 3 * 64 * 64, dtype=torch.float64).reshape(2, 3, 64, 64)
    tokens = torch.arange(64).reshape(2, 32).remainder(26)
    states = torch.arange(16, dtype=torch.float64).reshape(2, 8)
    actual = physical_inputs(images, tokens, states)
    assert len(actual) == 97
    for row in range(8):
        for column in range(8):
            expected = images[:, :, row * 8:(row + 1) * 8, column * 8:(column + 1) * 8].reshape(2, 192)
            assert torch.equal(actual[f"image.patch{row * 8 + column}"], expected)
    for index in range(32):
        assert torch.equal(actual[f"instruction.token{index}"], torch.nn.functional.one_hot(tokens[:, index], 26).double())
    assert actual["state"] is states


def test_reduced_predict_rejects_a_source_model_before_execution(monkeypatch):
    import modal_odt_dimension_curve_worker as worker
    monkeypatch.setattr(worker, "model_inputs", lambda *_args: (torch.zeros(1, 3, 64, 64), torch.zeros(1, 32, dtype=torch.long), torch.zeros(1, 8)))
    class MustNotExecute:
        def __getattr__(self, _name):
            raise AssertionError("source or mapped model was executed")
    import pytest
    with pytest.raises(RuntimeError, match="must not own"):
        predict(MustNotExecute(), MustNotExecute(), [{}], [""], {})
