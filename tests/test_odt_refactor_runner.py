"""Import-free numerical-safety checks for the explicit test-target boundary."""

from pathlib import Path
from contextlib import redirect_stderr
from io import StringIO
import unittest
from unittest.mock import patch
import subprocess

from scripts.run_odt_refactor_tests import _pytest_arguments
from scripts.validate_odt_proof import _reports_match_sources, _run


ROOT = Path(__file__).resolve().parents[1]
TEST = "tests/test_odt_symmetric_factorization.py"


class RunnerArgumentsTests(unittest.TestCase):
    def test_valid_targets_are_canonical_and_disable_implicit_collection(self):
        paths, args = _pytest_arguments(ROOT, [TEST, "-k", "symmetry", "--tb=line", "-x"])
        self.assertEqual(paths, ((ROOT / TEST).resolve(),))
        self.assertIn(str(paths[0]), args)
        self.assertIn("--noconftest", args)
        self.assertIn("-c", args)
        self.assertEqual(args[-1], "-x")

    def test_extra_unaudited_collection_targets_are_rejected(self):
        for target in ("tests", "xvla", "scripts/run_odt_refactor_tests.py", TEST + "::case"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                _pytest_arguments(ROOT, [TEST, target])

    def test_collection_and_plugin_options_are_rejected(self):
        for options in (("--pyargs", "xvla"), ("-p", "plugin"), ("-c", "config"),
                        ("--override-ini", "addopts=tests"), ("--import-mode=importlib",)):
            with self.subTest(options=options), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                _pytest_arguments(ROOT, [TEST, *options])

    def test_proof_run_requires_numerical_report(self):
        result = subprocess.CompletedProcess([], 0, "3 passed", "")
        with patch("scripts.validate_odt_proof.subprocess.run", return_value=result):
            self.assertFalse(_run("production", [], {})["passed"])

    def test_proof_run_preserves_failed_guard_result(self):
        result = subprocess.CompletedProcess([], 0, '{"passed": false}', "")
        with patch("scripts.validate_odt_proof.subprocess.run", return_value=result):
            self.assertFalse(_run("production", [], {})["passed"])

    def test_proof_run_records_timeout(self):
        error = subprocess.TimeoutExpired([], 1800, output=b"partial output")
        with patch("scripts.validate_odt_proof.subprocess.run", side_effect=error):
            record = _run("production", [], {})
        self.assertFalse(record["passed"])
        self.assertEqual(record["exit_code"], 124)
        self.assertEqual(record["stdout"], "partial output")

    def test_reference_report_accepts_sorted_json_keys(self):
        result = subprocess.CompletedProcess([], 0, '{"kernels": {}, "passed": true, "tests": 48}', "")
        with patch("scripts.validate_odt_proof.subprocess.run", return_value=result):
            record = _run("independent_reference", [], {})
        self.assertTrue(record["passed"])
        self.assertEqual(record["tests_passed"], 48)

    def test_unittest_stdout_is_not_a_numerical_report(self):
        result = subprocess.CompletedProcess([], 0, '{"passed": false}\n3 passed', "Ran 28 tests\nOK")
        with patch("scripts.validate_odt_proof.subprocess.run", return_value=result):
            record = _run("source_and_boundary", [], {})
        self.assertTrue(record["passed"])
        self.assertEqual(record["tests_passed"], 28)

    def test_report_hashes_must_match_frozen_sources(self):
        records = [
            {"name": "independent_reference", "report": {"source_sha256": {"dooms.py": "a"}}},
            {"name": "production", "report": {"sources": {"source_sha256": {"core.py": "b"}}}},
        ]
        expected = {"independent_reference": {"dooms.py": "a"}, "production": {"core.py": "b"}}
        self.assertTrue(_reports_match_sources(records, expected))
        for bad in ({}, {"core.py": "changed"}, {"core.py": "b", "extra.py": "c"}):
            records[1]["report"]["sources"]["source_sha256"] = bad
            self.assertFalse(_reports_match_sources(records, expected))
        self.assertFalse(_reports_match_sources(records[:1], expected))
        records[1]["report"]["sources"]["source_sha256"] = {"core.py": "b"}
        records[0]["report"]["source_sha256"] = {}
        self.assertFalse(_reports_match_sources(records, expected))


if __name__ == "__main__":
    unittest.main()
