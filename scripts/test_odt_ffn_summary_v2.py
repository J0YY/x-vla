"""Independent endpoint, authentication, clustering and guard regressions."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import odt_ffn_summary_v2 as analysis

analysis.audit_analysis()
analysis.install_guards()


def identity_fixture(count=160, episode_offset=0):
    rows = [(task, episode + episode_offset, frame) for task in range(10)
            for episode in range(count // 20) for frame in (10, 20)]
    records = [{"image_index": index, "task_id": task, "episode_id": episode,
                "frame_id": frame, "pair_id": pair, "pair_positions": [[27, 28], [0, 63]][pair]}
               for index, (task, episode, frame) in enumerate(rows) for pair in (0, 1)]
    panel = {f"image_{field}_ids": np.array([row[index] for row in rows], dtype=np.int64)
             for index, field in enumerate(("task", "episode", "frame"))}
    return rows, records, panel


def prediction_fixture(count=160, offset=0.):
    _, _, panel = identity_fixture(count)
    panel.update({"normalization__action_std": np.arange(1, 8, dtype=np.float32),
                  "normalization__action_mean": np.zeros(7, dtype=np.float32)})
    normalized = np.full((count, 8, 7), offset, dtype=np.float32)
    result = {"actions_normalized": normalized,
        "actions": normalized * panel["normalization__action_std"],
        "stage_output": np.zeros((count, 64, 192), dtype=np.float32),
        "valid": np.ones(count, dtype=bool), "attempted": np.ones(count, dtype=bool),
        "status_code": np.ones(count, dtype=np.int8), "stage_token_valid": np.ones((count, 64), dtype=bool),
        "latency_s_per_frame": np.ones(count, dtype=np.float64),
        "action_delta": np.zeros((count, 8, 7), dtype=np.float64),
        "gripper_sign_disagreement": np.zeros((count, 8), dtype=bool)}
    result.update({field + "_ids": panel["image_" + field + "_ids"] for field in ("task", "episode", "frame")})
    return result, panel


class FFNSummaryTests(unittest.TestCase):
    def test_native_scalar_hash_uses_contiguous_one_element_header(self):
        scalar = np.array(2., dtype="<f8")
        expected = "650dcb2c51ce477a302fd8d2294fe91628f8dc330f624d5dbd563564edbd15e8"
        self.assertEqual(scalar.shape, ())
        self.assertEqual(analysis.array_hash(scalar), expected)
        self.assertEqual(analysis.array_hash(np.array([2.], dtype="<f8")), expected)

    def test_receipts_recomputed_with_all_seven_components_and_eight_positions(self):
        baseline, _ = prediction_fixture()
        result, _ = prediction_fixture()
        result["actions"][0, 0, 0] = 2
        metrics, delta, sign = analysis.recompute(result, baseline)
        self.assertAlmostEqual(metrics["all"]["rmse"], (4 / (160 * 8 * 7)) ** .5)
        self.assertAlmostEqual(metrics["translation"]["rmse"], (4 / (160 * 8 * 3)) ** .5)
        self.assertEqual(metrics["rotation"]["rmse"], 0.)
        self.assertEqual(delta[0, 0, 0], 2.)
        self.assertFalse(np.any(sign))
        analysis.verify(metrics, metrics)
        tampered = dict(metrics, frames=159)
        with self.assertRaisesRegex(ValueError, "exact value"):
            analysis.verify(tampered, metrics)
        tampered = dict(metrics, invalid_probability_if_fully_attempted=1e-5)
        with self.assertRaisesRegex(ValueError, "value differs"):
            analysis.verify(tampered, metrics)

    def test_primary_fullrank_reference_is_separate_from_native(self):
        native, _ = prediction_fixture()
        fullrank, _ = prediction_fixture(offset=.25)
        reduced, _ = prediction_fixture(offset=.5)
        primary, frame = analysis.descriptive(reduced, fullrank)
        descriptive, _ = analysis.descriptive(reduced, native)
        self.assertAlmostEqual(descriptive["action_mse_all7"], 4 * primary["action_mse_all7"])
        self.assertTrue(np.all(frame == primary["action_mse_all7"]))

    def test_sign_zero_policy_convention_is_reported_separately(self):
        zero, _ = prediction_fixture()
        negative, _ = prediction_fixture()
        negative["actions"][:, :, -1] = -1
        measures, _ = analysis.descriptive(negative, zero)
        self.assertEqual(measures["gripper_sign_disagreement_probability"], 1.)
        self.assertEqual(measures["executed_gripper_sign_disagreement_probability"], 0.)

    def test_invalid_aborted_unattempted_and_nonfinite_never_silently_drop(self):
        for field, value in (("valid", False), ("attempted", False), ("status_code", 2),
                             ("status_code", 4), ("stage_token_valid", False)):
            result, panel = prediction_fixture()
            result[field][0] = value
            with self.assertRaisesRegex(ValueError, "zero-invalid"):
                analysis.validate_prediction(result, panel)
        for field in ("actions", "actions_normalized", "stage_output", "latency_s_per_frame"):
            result, panel = prediction_fixture()
            result[field].flat[0] = np.nan
            with self.assertRaisesRegex(ValueError, "schema differs"):
                analysis.validate_prediction(result, panel)

    def test_prediction_dtype_identity_and_denormalization_contracts(self):
        result, panel = prediction_fixture(offset=1.)
        analysis.validate_prediction(result, panel)
        result["actions"][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "denormalization"):
            analysis.validate_prediction(result, panel)
        result, panel = prediction_fixture()
        result["status_code"] = result["status_code"].astype(np.int64)
        with self.assertRaisesRegex(ValueError, "zero-invalid"):
            analysis.validate_prediction(result, panel)
        result, panel = prediction_fixture()
        result["frame_ids"] = result["frame_ids"].copy()
        result["frame_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "identities"):
            analysis.validate_prediction(result, panel)

    def test_metadata_frame_pairs_and_episode_cluster_balance(self):
        rows, records, panel = identity_fixture()
        measured, keys = analysis.validate_ids(panel, records)
        self.assertEqual(measured, rows)
        self.assertEqual(len(keys), 80)
        records[0]["episode_id"] += 1
        with self.assertRaisesRegex(ValueError, "disagree"):
            analysis.validate_ids(panel, records)
        _, records, panel = identity_fixture()
        records[1] = records[0].copy()
        with self.assertRaisesRegex(ValueError, "pair index"):
            analysis.validate_ids(panel, records)

    def test_bootstrap_resamples_episodes_within_task_not_frames(self):
        rows, _, _ = identity_fixture()
        design = analysis.bootstrap_design(rows)
        cluster, weights = design
        self.assertEqual(weights.shape, (4000, 80))
        self.assertTrue(np.all(np.sum(weights.reshape(4000, 10, 8), axis=2) == 8))
        self.assertTrue(np.all(cluster[::2] == cluster[1::2]))
        zero = np.zeros(160)
        opposite_frames = np.tile(np.array([1., -1.]), 80)
        paired = analysis.paired_bootstrap(opposite_frames, zero, design)
        self.assertEqual(paired["interval95"], {"lower": 0., "upper": 0.})
        constant = analysis.paired_bootstrap(np.full(160, 2.), zero, design)
        self.assertEqual(constant["action_mse_difference_leading_minus_anchored"], 2.)
        self.assertEqual(constant["interval95"], {"lower": 2., "upper": 2.})
        again = analysis.bootstrap_design(rows)
        self.assertTrue(np.array_equal(weights, again[1]))
        bad_rows = rows.copy(); bad_rows[0] = bad_rows[2]
        with self.assertRaisesRegex(ValueError, "eight episodes"):
            analysis.bootstrap_design(bad_rows)

    def test_double_receipt_gate_is_rechecked_not_boolean_trusted(self):
        gate = {"passed": True, "finite": True, "maximum_absolute_error": 1e-12,
                "scaled_error": 1e-12, "scale": 1., "tolerance": 1e-10}
        analysis.validate_double_receipt(gate)
        for key, value in (("scaled_error", 1e-8), ("finite", False), ("scale", 0.), ("maximum_absolute_error", 1.)):
            with self.assertRaises(ValueError):
                analysis.validate_double_receipt(dict(gate, **{key: value}))

    def test_authentication_rejects_tampered_array_and_pinned_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "values.npz"
            np.savez(path, a=np.zeros(2))
            expected = analysis.digest(path)
            self.assertTrue(np.array_equal(analysis.read_arrays(path, expected)["a"], np.zeros(2)))
            np.savez(path, a=np.ones(2))
            with self.assertRaisesRegex(ValueError, "identity"):
                analysis.read_arrays(path, expected)
            (root / "manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "identity"):
                analysis.summarize(root, root / "panel", root / "metadata", root / "variants", root / "out")
            self.assertFalse((root / "out").exists())

    def test_missing_or_old_v2_failure_contract_rejected_before_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "manifest.json"
            for value in ({}, {"schema": "odt-ffn-native-actions-v1"},
                          {"schema": "odt-ffn-native-actions-v2", "failure_accounting_version": 1}):
                path.write_text(json.dumps(value))
                with patch.object(analysis, "ACTIONS_SHA", analysis.digest(path)):
                    with self.assertRaisesRegex(ValueError, "manifest admission"):
                        analysis.summarize(root, root / "panel", root / "metadata", root / "variants", root / "out")

    def test_source_audit_and_18_conditions(self):
        self.assertEqual(len(analysis.audit_analysis()), 3)
        self.assertEqual(len(analysis.conditions()), 17)
        self.assertEqual({row["seed"] for row in analysis.conditions() if row["seed"] is not None}, {0, 1, 2})
        self.assertEqual(analysis.COUNTS, {"qr": 0, "eigh": 0, "prohibited_attempts": 0})

    def test_fresh_main_and_discovery_install_guard_before_fixtures(self):
        scripts = [
            "from scripts import odt_ffn_summary_v2 as m\nimport sys\n"
            "def stub(*args):\n assert m.np.linalg.svd.__name__ == 'reject'\n assert m.COUNTS == {'qr':0,'eigh':0,'prohibited_attempts':0}\n return {'complete':True,'arms':[],'kernels':dict(m.COUNTS)}\n"
            "m.summarize=stub\nsys.argv=['summary','--actions','a','--panel','p','--panel-metadata','m','--variants','v','--output','o']\nm.main()\n",
            "import scripts.test_odt_ffn_summary_v2 as t\n"
            "assert t.np.linalg.svd.__name__ == 'reject'\nt.prediction_fixture()\n"
            "assert t.analysis.COUNTS == {'qr':0,'eigh':0,'prohibited_attempts':0}\n",
        ]
        for code in scripts:
            result = subprocess.run([sys.executable, "-c", code], cwd=analysis.ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
