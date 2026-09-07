"""Regressions for frozen sample identity, metadata joins and bridge gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.odt_real_panel_v1 import (
    DATASET_TO_OFFICIAL, PAIR_POSITIONS, NATIVE_TOLERANCES,
    array_sha256, comparison, encode_instruction, expand_pair_records,
    metadata_panel, panel_audit, publish_json, require_hash, sha256,
)
from research.odt_reference.block_oracle import block_forward


def fixture_cache():
    return [(episode, frame * 10, np.zeros((1,), dtype=np.uint8), np.zeros(8), np.zeros(7), task)
            for task in range(10) for episode in range(12) for frame in range(9)]


class PanelTests(unittest.TestCase):
    def test_balanced_disjoint_episode_partitions(self):
        panel = metadata_panel(fixture_cache())
        dev, confirmation = (panel["partitions"][key] for key in ("development", "confirmation"))
        self.assertEqual(len(dev), 40)
        self.assertEqual(len(confirmation), 160)
        dev_ids = {(r["task_id"], r["episode_id"]) for r in dev}
        confirmation_ids = {(r["task_id"], r["episode_id"]) for r in confirmation}
        self.assertFalse(dev_ids & confirmation_ids)
        self.assertEqual(len(dev_ids), 20)
        self.assertEqual(len(confirmation_ids), 80)
        for task in range(10):
            self.assertEqual(sum(r["task_id"] == task for r in confirmation), 16)
        for row in confirmation:
            self.assertEqual(row["task_id"], DATASET_TO_OFFICIAL[row["dataset_task_id"]])
            self.assertIn(row["frame_id"], (20, 60))

    def test_pixels_actions_and_input_order_cannot_select_episodes(self):
        cache = fixture_cache()
        first = metadata_panel(cache)
        changed = [(a, b, np.ones(5), np.ones(8)*90, np.ones(7)*-200, f) for a,b,c,d,e,f in reversed(cache)]
        second = metadata_panel(changed)
        def identities(panel):
            return {name: [(r["task_id"],r["episode_id"],r["frame_id"])
                           for r in records] for name, records in panel["partitions"].items()}
        self.assertEqual(identities(first), identities(second))

    def test_duplicate_and_missing_task_fail_closed(self):
        cache = fixture_cache()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            metadata_panel(cache + [cache[0]])
        with self.assertRaisesRegex(ValueError, "coverage"):
            metadata_panel([row for row in cache if row[5] != 0])
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            metadata_panel([(True, *cache[0][1:])])

    def test_pair_alignment_and_spatial_separation(self):
        records = metadata_panel(fixture_cache())["partitions"]["confirmation"]
        rows = expand_pair_records(records)
        self.assertEqual(len(rows), 320)
        for index, row in enumerate(rows):
            self.assertEqual(row["image_index"], index // 2)
            self.assertEqual(row["pair_id"], index % 2)
            self.assertEqual(tuple(row["pair_positions"]), PAIR_POSITIONS[index % 2])
        a, b = PAIR_POSITIONS[0]
        self.assertEqual(a // 8, b // 8)
        self.assertEqual(abs(a - b), 1)
        self.assertEqual(PAIR_POSITIONS[1], (0, 63))

    def test_hash_binds_dtype_shape_and_bytes_and_rejects_object(self):
        a = np.arange(8, dtype=np.float32)
        self.assertNotEqual(array_sha256(a), array_sha256(a.reshape(2, 4)))
        self.assertNotEqual(array_sha256(a), array_sha256(a.astype(np.float64)))
        with self.assertRaisesRegex(ValueError, "object"):
            array_sha256(np.asarray([{}], dtype=object))

    def test_frozen_artifact_refuses_overwrite_and_hash_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "freeze.json"
            digest = publish_json(path, {"x": [1, 2]})
            self.assertEqual(digest, publish_json(path, {"x": [1, 2]}))
            require_hash(path, digest)
            with self.assertRaisesRegex(RuntimeError, "replace"):
                publish_json(path, {"x": [1, 3]})
            path.write_text(json.dumps({"x": [9]}))
            with self.assertRaisesRegex(RuntimeError, "identity"):
                require_hash(path, digest)

    def test_parity_requires_both_frozen_absolute_and_scaled_gates(self):
        reference = np.ones((2, 2, 3))
        self.assertTrue(comparison(reference, reference)["passed"])
        self.assertFalse(comparison(reference + 3e-5, reference)["passed"])
        large = reference * 1e5
        self.assertFalse(comparison(large + 3e-4, large)["passed"])
        invalid = reference.copy()
        invalid[0, 0, 0] = np.nan
        self.assertFalse(comparison(invalid, reference)["passed"])
        with self.assertRaisesRegex(ValueError, "equal nonempty"):
            comparison(reference, reference[:, :1])
        self.assertEqual(NATIVE_TOLERANCES["absolute"], 2e-4)
        self.assertEqual(NATIVE_TOLERANCES["scaled"], 2e-5)

    def test_instruction_encoding_matches_fixed_bos_pad_protocol(self):
        result = encode_instruction("Pick up unknown.", {"pick": 17, "up": 25})
        self.assertEqual(len(result), 32)
        self.assertEqual(result[:5], [1, 17, 25, 0, 0])

    def test_independent_oracle_retains_both_residual_gains_and_optional_down_bias(self):
        width = 4
        weights = {"attn_gain": np.array(.5), "ffn_gain": np.array(.25)}
        for name in ("rbn_attn", "rbn_ffn", "attn.rn_q1", "attn.rn_k1", "attn.rn_q2", "attn.rn_k2"):
            weights.update({name+".initialized": np.array(True), name+".running_ms": np.array(1.),
                            name+".pa": np.array([1., 0., 0.]), name+".pb": np.array([1., 0., 0.])})
        for name in ("attn.wq1", "attn.wk1", "attn.wq2", "attn.wk2", "attn.wv", "attn.wo"):
            weights[name+".weight"] = np.zeros((width, width))
            weights[name+".bias"] = np.zeros(width)
        weights["attn.wo.bias"] = np.ones(width) * 2
        for name in ("ffn.left", "ffn.right"):
            weights[name+".weight"] = np.eye(width)
            weights[name+".bias"] = np.zeros(width)
        weights["ffn.down.weight"] = np.eye(width)
        weights["ffn.down.bias"] = np.ones(width) * 3
        raw = np.arange(8, dtype=np.float64).reshape(1, 2, 4) / 8
        output, trace = block_forward(raw, weights, n_heads=2, mask=np.ones((2, 2)), return_trace=True)
        residual = raw + 1
        expected = residual + .25 * (residual * residual + 3)
        np.testing.assert_allclose(output, expected, atol=1e-14, rtol=0)
        np.testing.assert_allclose(trace["residual1"], residual, atol=1e-14, rtol=0)

    def test_source_closure_excludes_historical_training_and_simulator(self):
        audit = panel_audit()
        sources = set(audit["source_sha256"])
        self.assertIn("xvla/models/vla.py", sources)
        self.assertIn("research/odt_reference/block_oracle.py", sources)
        self.assertFalse(any("run_xvla" in name or "modal_app" in name or "run_capable" in name for name in sources))
        self.assertEqual(audit["prohibited_calls_found"], [])
        self.assertEqual(audit["prohibited_self_overlap_sites"], [])


if __name__ == "__main__":
    unittest.main()
