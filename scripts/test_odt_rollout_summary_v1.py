"""Authentication, trajectory replay and paired inference regression tests."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts import odt_rollout_summary_v1 as analysis

analysis.audit_analysis()
analysis.install_guards()


def fixture(steps=9, success=True, terminal=False):
    authority = analysis.read_json(analysis.ROOT / "research/odt_ffn_rollout_v1/tasks.json")
    normalization = {"action_std": np.arange(1, 8, dtype=np.float32), "action_mean": np.zeros(7, dtype=np.float32)}
    chunks = (steps + 7) // 8
    normalized = (np.arange(chunks * 8 * 7, dtype=np.float32).reshape(chunks, 8, 7) - 15) / np.float32(100)
    denormalized = normalized * normalization["action_std"] + normalization["action_mean"]
    executed = denormalized.reshape(-1, 7)[:steps].copy()
    executed[:, -1] = np.where(executed[:, -1] > 0, 1., -1.)
    arrays = {"rgb": np.zeros((steps, 64, 64, 3), dtype=np.uint8), "states_raw": np.zeros((steps, 8), dtype=np.float32),
        "actions_executed": executed, "actions_normalized": normalized.reshape(-1, 7)[:steps].copy(),
        "rewards": np.zeros(steps, dtype=np.float64), "terminals": np.zeros(steps, dtype=bool),
        "prediction_steps": np.arange(0, steps, 8, dtype=np.int64), "action_chunks_normalized": normalized,
        "action_chunks_denormalized": denormalized, "executed_step_action_valid": np.ones(steps, dtype=bool),
        "initial_state": np.arange(12, dtype=np.float64)}
    arrays["rewards"][-1] = int(success)
    arrays["terminals"][-1] = terminal
    task = authority["EXPECTED_TASK_PROTOCOL"]["0"]
    record = {"arm": "native", "task_index": 0, "episode": 0, "seed": 0, "steps": steps,
        "measurement_complete": True, "failure": None, "success": success, "terminated_without_success": terminal and not success,
        "elapsed_s": 2., "setup_elapsed_s": 1., "initial_state_sha256": analysis.array_digest(arrays["initial_state"]),
        "task_protocol": {"task_index": 0, "language": task[0], "bddl_sha256": task[2], "init_states_array_sha256": task[3],
            "packaged_file": {"path": "/trusted/initialstates", "sha256": "a" * 64}},
        "controller_counts": {"calls": 10, "direct_qr_systems": 30, "largest_relative_residual": 1e-10,
            "largest_componentwise_backward_error": 1e-9, "largest_normwise_backward_error": 1e-15,
            "rhs_residual_warning_systems": 0, "componentwise_residual_warning_systems": 1}}
    return arrays, record, normalization, authority


def validate(arrays, record, normalization, authority):
    return analysis.validate_episode(arrays, record, normalization, authority, arm="native", task=0, episode=0)


class RolloutSummaryTests(unittest.TestCase):
    def test_action_chunk_replay_including_partial_chunk_and_gripper(self):
        values = fixture()
        result = validate(*values)
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 9)
        arrays, record, normalization, authority = values
        arrays["actions_executed"][0, -1] *= -1
        with self.assertRaisesRegex(ValueError, "saved policy chunk"):
            validate(arrays, record, normalization, authority)

    def test_normalized_to_denormalized_and_schedule_cannot_drift(self):
        arrays, record, normalization, authority = fixture()
        arrays["action_chunks_denormalized"][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "denormalization"):
            validate(arrays, record, normalization, authority)
        arrays, record, normalization, authority = fixture()
        arrays["prediction_steps"][1] = 7
        with self.assertRaisesRegex(ValueError, "schedule"):
            validate(arrays, record, normalization, authority)
        arrays, record, normalization, authority = fixture()
        arrays["actions_normalized"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "saved policy chunk"):
            validate(arrays, record, normalization, authority)

    def test_horizon_failure_and_early_terminal_are_distinct_complete_outcomes(self):
        result = validate(*fixture(280, success=False))
        self.assertFalse(result["success"])
        self.assertEqual(result["steps"], 280)
        self.assertFalse(validate(*fixture(17, success=False, terminal=True))["success"])
        with self.assertRaisesRegex(ValueError, "horizon"):
            validate(*fixture(17, success=False))

    def test_reward_and_terminal_evidence_override_claimed_success(self):
        arrays, record, normalization, authority = fixture()
        record["success"] = False
        with self.assertRaisesRegex(ValueError, "success/termination"):
            validate(arrays, record, normalization, authority)
        for key, value in (("rewards", 1.), ("terminals", True)):
            arrays, record, normalization, authority = fixture()
            arrays[key][0] = value
            with self.assertRaisesRegex(ValueError, "success/termination"):
                validate(arrays, record, normalization, authority)

    def test_nonfinite_wrong_shape_dtype_invalid_or_failed_cannot_pass(self):
        for key in ("states_raw", "actions_executed", "rewards", "action_chunks_normalized"):
            arrays, record, normalization, authority = fixture()
            arrays[key].flat[0] = np.nan
            with self.assertRaisesRegex(ValueError, "schema"):
                validate(arrays, record, normalization, authority)
        for mutation in ("dtype", "length", "invalid", "failed"):
            arrays, record, normalization, authority = fixture()
            if mutation == "dtype": arrays["terminals"] = arrays["terminals"].astype(np.int8)
            if mutation == "length": arrays["rgb"] = arrays["rgb"][:-1]
            if mutation == "invalid": arrays["executed_step_action_valid"][0] = False
            if mutation == "failed": record["measurement_complete"] = False
            with self.assertRaises(ValueError):
                validate(arrays, record, normalization, authority)

    def test_exact_initial_state_and_official_task_protocol(self):
        arrays, record, normalization, authority = fixture()
        arrays["initial_state"][0] += 1
        with self.assertRaisesRegex(ValueError, "initial-state"):
            validate(arrays, record, normalization, authority)
        arrays, record, normalization, authority = fixture()
        record["task_protocol"]["init_states_array_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "official task"):
            validate(arrays, record, normalization, authority)

    def test_paired_first_rgb_state_and_initial_state_all_must_match(self):
        observed = validate(*fixture())
        identity = analysis.matched_identity(observed)
        self.assertEqual(analysis.matched_identity(observed, identity), identity)
        for key in identity:
            changed = dict(observed, **{key: "different"})
            with self.assertRaisesRegex(ValueError, "first policy observations"):
                analysis.matched_identity(changed, identity)

    def test_controller_normwise_gate_and_three_qr_systems(self):
        counts = fixture()[1]["controller_counts"]
        analysis.validate_counts(counts)
        # Componentwise warnings are descriptive; only normwise is a gate.
        for key, value in (("largest_normwise_backward_error", 1.1e-12), ("direct_qr_systems", 29),
                           ("largest_relative_residual", np.nan), ("calls", 0)):
            with self.assertRaises(ValueError):
                analysis.validate_counts(dict(counts, **{key: value}))

    def test_wilson_and_exact_mcnemar_known_cases(self):
        zero = analysis.wilson(0, 20)
        full = analysis.wilson(20, 20)
        self.assertAlmostEqual(zero["upper"], .16112515805281938)
        self.assertAlmostEqual(full["lower"], 1 - zero["upper"])
        weights = analysis.bootstrap_design()
        a, b = np.zeros(20, dtype=bool), np.zeros(20, dtype=bool)
        self.assertEqual(analysis.paired(a, b, weights)["exact_mcnemar_two_sided_p"], 1.)
        a[:5] = True
        paired = analysis.paired(a, b, weights)
        self.assertEqual(paired["exact_mcnemar_two_sided_p"], .0625)
        self.assertEqual(paired["success_difference_a_minus_b"], .25)
        self.assertEqual(paired["a_only_success"], 5)
        reverse = analysis.paired(b, a, weights)
        self.assertEqual(reverse["exact_mcnemar_two_sided_p"], .0625)
        self.assertEqual(reverse["success_difference_a_minus_b"], -.25)

    def test_bootstrap_preserves_two_initial_states_per_task_and_pairs(self):
        weights = analysis.bootstrap_design()
        self.assertEqual(weights.shape, (4000, 20))
        self.assertTrue(np.all(np.sum(weights.reshape(4000, 10, 2), axis=2) == 2))
        self.assertTrue(np.array_equal(weights, analysis.bootstrap_design()))
        a, b = np.ones(20, dtype=bool), np.zeros(20, dtype=bool)
        self.assertEqual(analysis.paired(a, b, weights)["paired_task_stratified_interval95"], {"lower": 1., "upper": 1.})
        self.assertEqual(analysis.paired(a, a, weights)["paired_task_stratified_interval95"], {"lower": 0., "upper": 0.})

    def test_tampered_pinned_smoke_and_array_hash_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "smoke").mkdir()
            (root / "smoke/receipt.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "identity"):
                analysis.summarize(root, root / "training.json", root / "output")
            self.assertFalse((root / "output").exists())
            path = root / "array.npz"
            np.savez(path, a=np.zeros(2))
            original = analysis.digest(path)
            np.savez(path, a=np.ones(2))
            with self.assertRaisesRegex(ValueError, "identity"):
                analysis.require_hash(path, original)

    def test_fresh_cli_and_test_discovery_guards_precede_work(self):
        codes = [
            "from scripts import odt_rollout_summary_v1 as m\nimport sys\n"
            "def stub(*args):\n assert m.np.linalg.svd.__name__ == 'reject'\n assert not any(m.COUNTS.values())\n return {'complete':True,'episodes':80,'analysis_kernels':dict(m.COUNTS)}\n"
            "m.summarize=stub\nsys.argv=['summary','--rollout','r','--training','t','--output','o']\nm.main()\n",
            "import scripts.test_odt_rollout_summary_v1 as t\nassert t.np.linalg.svd.__name__ == 'reject'\nt.fixture()\nassert not any(t.analysis.COUNTS.values())\n",
        ]
        for code in codes:
            result = subprocess.run([sys.executable, "-c", code], cwd=analysis.ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(analysis.audit_analysis()), 3)
        self.assertEqual(analysis.COUNTS, {"qr": 0, "eigh": 0, "prohibited_attempts": 0})


if __name__ == "__main__":
    unittest.main()
