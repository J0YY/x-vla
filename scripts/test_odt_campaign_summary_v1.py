"""Check paired conditioning and episode resampling, independent of consumers."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.odt_reference.run_tests import install_guards, COUNTS

# unittest discovery executes imports before setUp, so install here too.
install_guards()

from scripts.odt_campaign_summary_v1 import (
    ARTIFACTS, CHECKPOINT_SHA, PRODUCER_RECEIPT_SHA, audit_analysis, cluster_design,
    descriptive_seed_summaries, digest, json_digest, paired_comparison, recompute_all_metrics,
    recompute_metrics, summarize, validate_predictions, validate_records, verify_metric_receipt,
)


def records_fixture():
    records = [{"panel": "real", "task_id": task, "episode_id": episode,
                "frame_id": frame, "pair_id": pair}
               for task in range(10) for episode in range(8) for frame in (0, 10) for pair in (0, 1)]
    records += [{"panel": "gaussian", "task_id": "gaussian", "episode_id": i,
                 "frame_id": i, "pair_id": i} for i in range(256)]
    return records


def predictions_fixture(*, rows=576, width=384, value=1., invalid=()):
    valid = np.ones(rows, dtype=bool)
    valid[list(invalid)] = False
    decoded = np.full((rows, width), value, dtype=np.float64)
    decoded[~valid] = np.nan
    projective = np.ones((rows, width+1), dtype=np.float64)
    projective[:, :-1] *= value
    projective /= np.max(np.abs(projective), axis=1, keepdims=True)
    projective[~valid] = 0.
    return {"decoded": decoded, "valid": valid, "projective": projective,
            "failure_code": np.where(valid, 0, 2).astype(np.int8),
            "relative_denominator": np.abs(projective[:, -1])}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def campaign_fixture(root):
    campaign, results = root/"campaign", root/"results"
    campaign.mkdir()
    results.mkdir()
    records = records_fixture()
    baseline = predictions_fixture()
    baseline["expected"] = baseline["decoded"].copy()
    np.savez(campaign/"baseline.npz", **baseline)
    write_json(campaign/"records.json", records)
    for name in ARTIFACTS-{"baseline.npz", "records.json", "protocol.json"}:
        (campaign/name).write_bytes(b"authenticated fixture artifact")
    widths = [193, 193]+[21]*274+[20]*176+[385]
    identity = {"ranks": widths, "construction": "leading", "seed": None}
    condition = {"task_index": 0, **identity, "condition_sha256": json_digest(identity),
                 "rank_sha256": json_digest(widths), "aliases": [{"policy": "full", "budget": 0}],
                 "removed_dimensions": 0, "original_dimensions": 9274}
    protocol = {"version": "odt-campaign-v1", "conditions": [condition], "contrasts": [],
                "sources": {"fixture": "analysis boundary fixture, no numerical graph load"},
                "producer_receipt_sha256": PRODUCER_RECEIPT_SHA, "checkpoint_sha256": CHECKPOINT_SHA,
                "global_binary_exponent": -12910, "environment_binary_exponent": -25820,
                "original_dimensions": 9274, "minimum_internal_rank": 1, "ingress_and_root_full_rank": True,
                "inputs_sha256": digest(campaign/"inputs.npz"), "records_sha256": digest(campaign/"records.json"),
                "scope": "authenticated boundary fixture"}
    write_json(campaign/"protocol.json", protocol)
    manifest = {"ready": True, "version": "odt-campaign-v1", "condition_count": 1,
                "artifacts": {name: digest(campaign/name) for name in ARTIFACTS},
                "context_removal_error": {}, "new_full_rank_replay_error": 0., "native_two_token_parity": {}}
    write_json(campaign/"manifest.json", manifest)
    directory = results/"task_0"
    directory.mkdir()
    predictions = predictions_fixture()
    predictions["squared_error"] = np.zeros(576)
    np.savez(directory/"predictions.npz", **predictions)
    (directory/"arrays.npz").write_bytes(b"physical fixture hash boundary")
    write_json(directory/"graph.json", {"global_binary_exponent": -12910})
    metrics, task_metrics = recompute_all_metrics(predictions, baseline, records)
    receipt = {"condition": condition, "completed": True, "version": "odt-campaign-v1",
        "campaign_manifest_sha256": digest(campaign/"manifest.json"),
        "protocol_sha256": manifest["artifacts"]["protocol.json"],
        "basis_artifact_sha256": manifest["artifacts"]["bases.npz"], "source_sha256": protocol["sources"],
        "producer_receipt_sha256": PRODUCER_RECEIPT_SHA, "global_binary_exponent": -12910,
        "environment_binary_exponent": -25820, "kernels": {"prohibited_attempts": 0},
        "predictions_sha256": digest(directory/"predictions.npz"),
        "physical_artifact": {"graph_sha256": digest(directory/"graph.json"), "arrays_sha256": digest(directory/"arrays.npz")},
        "metrics": metrics, "task_metrics": task_metrics}
    write_json(directory/"result.json", receipt)
    return campaign, results, receipt, predictions


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.assertEqual(np.linalg.qr.__name__, "counted")
        self.records = [{"panel": "real", "task_id": t, "episode_id": e}
                        for t in (0, 1) for e in (0, 1) for _ in range(2)]
        self.design = cluster_design(self.records, repetitions=100)
        self.baseline = {"valid": np.ones(8, dtype=bool), "decoded": np.zeros((8, 2))}

    def test_resampling_preserves_tasks_and_episode_cluster(self):
        indices, row_clusters, keys, weights = self.design
        self.assertEqual(len(keys), 4)
        self.assertTrue(np.all(row_clusters[::2] == row_clusters[1::2]))
        np.testing.assert_array_equal(weights[:, :2].sum(axis=1), np.full(100, 2))
        np.testing.assert_array_equal(weights[:, 2:].sum(axis=1), np.full(100, 2))

    def test_identical_conditions_exact_zero_interval(self):
        result = paired_comparison(self.baseline, self.baseline, self.baseline, self.design)
        self.assertEqual(result["mse_difference_a_minus_b_conditional_on_joint_validity"], 0.)
        self.assertEqual(result["mse_difference_interval95"]["lower"], 0.)
        self.assertEqual(result["mse_difference_interval95"]["upper"], 0.)

    def test_invalid_rows_not_imputed_as_zero_error(self):
        a = {"valid": np.array([False, True, True, True, True, True, True, True]),
             "decoded": np.ones((8, 2))}
        a["decoded"][0] = np.nan
        result = paired_comparison(a, self.baseline, self.baseline, self.design)
        self.assertEqual(result["jointly_valid_count"], 7)
        self.assertEqual(result["mse_difference_a_minus_b_conditional_on_joint_validity"], 1.)
        self.assertEqual(result["invalid_probability_difference_a_minus_b"], .125)
        self.assertEqual(result["mse_difference_interval95"]["lower"], 1.)

    def test_no_joint_validity_has_no_decoded_endpoint(self):
        a = {"valid": np.zeros(8, dtype=bool), "decoded": np.full((8, 2), np.nan)}
        result = paired_comparison(a, self.baseline, self.baseline, self.design)
        self.assertIsNone(result["mse_difference_a_minus_b_conditional_on_joint_validity"])
        self.assertIsNone(result["mse_difference_interval95"]["lower"])
        self.assertEqual(result["invalid_probability_difference_a_minus_b"], 1.)

    def test_independent_metric_counts_conditioning_and_nulls(self):
        baseline = predictions_fixture(rows=3, width=2)
        value = predictions_fixture(rows=3, width=2, value=3., invalid=(0,))
        metric = recompute_metrics(value, baseline, np.ones(3, dtype=bool))
        self.assertEqual(metric["invalid_count"], 1)
        self.assertEqual(metric["jointly_valid_count"], 2)
        self.assertEqual(metric["rmse"], 2.)
        self.assertEqual(metric["rmse_over_source_rms"], 2.)
        self.assertEqual(metric["maximum_absolute_error"], 2.)
        changed = dict(metric, invalid_count=0)
        with self.assertRaisesRegex(ValueError, "exact value"):
            verify_metric_receipt(changed, metric)
        changed = dict(metric, rmse=2.0001)
        with self.assertRaisesRegex(ValueError, "metric value"):
            verify_metric_receipt(changed, metric)
        changed = dict(metric, rmse=2.+1e-12)
        verify_metric_receipt(changed, metric)
        invalid = predictions_fixture(rows=3, width=2, invalid=(0, 1, 2))
        empty = recompute_metrics(invalid, baseline, np.ones(3, dtype=bool))
        self.assertIsNone(empty["rmse"])
        with self.assertRaisesRegex(ValueError, "exact value"):
            verify_metric_receipt(dict(empty, rmse=0.), empty)

    def test_strict_prediction_schema_finiteness_and_quotient(self):
        values = predictions_fixture(rows=3, width=2, invalid=(0,))
        validate_predictions(values, shape=(3, 2))
        values["decoded"][1, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_predictions(values, shape=(3, 2))
        values = predictions_fixture(rows=3, width=2)
        values["valid"] = values["valid"].astype(np.int8)
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_predictions(values, shape=(3, 2))
        values = predictions_fixture(rows=3, width=2)
        values["decoded"][1, 0] += 1.
        with self.assertRaisesRegex(ValueError, "quotient"):
            validate_predictions(values, shape=(3, 2))
        values = predictions_fixture(rows=3, width=2, invalid=(0,))
        values["expected"] = np.ones((3, 2))
        with self.assertRaisesRegex(ValueError, "baseline contains invalid"):
            validate_predictions(values, shape=(3, 2), baseline=True)

    def test_record_task_balance_and_duplicates(self):
        records = records_fixture()
        validate_records(records)
        records[0] = records[1]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_records(records)

    def test_full_hash_bound_summary_recomputes_metrics_without_contrasts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign, results, _, _ = campaign_fixture(root)
            summarize(campaign, results, root/"out")
            summary = json.loads((root/"out/summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["condition_count"], 1)
            self.assertTrue(summary["conditions"][0]["receipt_metrics_independently_verified"])
            self.assertEqual(summary["conditions"][0]["metrics"]["real"]["rmse"], 0.)

    def test_stale_receipt_contract_and_tampered_predictions_rejected(self):
        for attack in ("contract", "predictions", "metrics", "task_metrics", "physical"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                campaign, results, receipt, _ = campaign_fixture(root)
                directory = results/"task_0"
                if attack == "contract":
                    receipt["global_binary_exponent"] = 0
                elif attack == "predictions":
                    (directory/"predictions.npz").write_bytes(b"tampered before array load")
                elif attack == "metrics":
                    receipt["metrics"]["real"]["rmse"] = .25
                elif attack == "task_metrics":
                    receipt["task_metrics"][0]["invalid_count"] = 1
                elif attack == "physical":
                    (directory/"arrays.npz").write_bytes(b"tampered physical core archive")
                write_json(directory/"result.json", receipt)
                with self.assertRaises(ValueError):
                    summarize(campaign, results, root/"out")

    def test_hash_valid_but_nonfinite_or_wrong_squared_errors_rejected(self):
        for attack in ("nonfinite", "squared_error"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                campaign, results, receipt, predictions = campaign_fixture(root)
                if attack == "nonfinite":
                    predictions["decoded"][0, 0] = np.inf
                else:
                    predictions["squared_error"][0] = 1.
                np.savez(results/"task_0/predictions.npz", **predictions)
                receipt["predictions_sha256"] = digest(results/"task_0/predictions.npz")
                write_json(results/"task_0/result.json", receipt)
                with self.assertRaises(ValueError):
                    summarize(campaign, results, root/"out", allow_partial=True)

    def test_three_basis_seeds_descriptive_only_and_missing_seed_undefined(self):
        conditions = []
        for seed in range(3):
            metrics = {"rmse": float(seed+1), "rmse_over_source_rms": float(seed+1),
                       "invalid_probability_unconditional": 0., "jointly_valid_count": 320}
            conditions.append({"condition": {"construction": "zero_input_anchored_direct_qr", "rank_sha256": "same",
                "aliases": [{"policy": "W", "budget": 927}], "seed": seed},
                "metrics": {"real": metrics, "gaussian": metrics}})
        summary = descriptive_seed_summaries(conditions)[0]
        self.assertEqual(summary["panels"]["real"]["rmse"]["arithmetic_mean"], 2.)
        self.assertEqual(summary["panels"]["real"]["rmse"]["minimum"], 1.)
        self.assertEqual(summary["panels"]["real"]["rmse"]["maximum"], 3.)
        self.assertIn("not policy-training", summary["scope"])
        partial = descriptive_seed_summaries(conditions[:2])[0]
        self.assertFalse(partial["complete_three_basis_seeds"])
        self.assertIsNone(partial["panels"]["real"]["rmse"]["arithmetic_mean"])

    def test_fresh_discovery_installs_guard_before_fixture(self):
        program = '''
import unittest
import numpy as np
before = np.linalg.qr
suite = unittest.defaultTestLoader.loadTestsFromName(
    "scripts.test_odt_campaign_summary_v1.SummaryTests.test_identical_conditions_exact_zero_interval")
assert np.linalg.qr is not before
result = unittest.TextTestRunner().run(suite)
assert result.wasSuccessful()
from research.odt_reference.run_tests import COUNTS
assert COUNTS == {"qr": 0, "eigh": 0, "prohibited_attempts": 0}
'''
        result = subprocess.run([sys.executable, "-c", program], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_fresh_cli_guards_before_arithmetic_without_factorizations(self):
        program = '''
import sys
import numpy as np
from scripts import odt_campaign_summary_v1 as summary
from research.odt_reference.run_tests import COUNTS
before = np.linalg.qr
def probe(*args, **kwargs):
    assert np.linalg.qr is not before
    assert float(np.sum(np.arange(3.))) == 3.
    assert COUNTS == {"qr": 0, "eigh": 0, "prohibited_attempts": 0}
summary.summarize = probe
sys.argv = ["summary", "--campaign", "unused", "--results", "unused", "--output", "unused"]
summary.main()
'''
        result = subprocess.run([sys.executable, "-c", program], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_source_audit_and_no_numerical_factorizations(self):
        self.assertEqual(len(audit_analysis()), 3)
        self.assertEqual(COUNTS, {"qr": 0, "eigh": 0, "prohibited_attempts": 0})


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SummaryTests))
    raise SystemExit(0 if result.wasSuccessful() and COUNTS["prohibited_attempts"] == 0 else 1)
