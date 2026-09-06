"""Static-closure regressions, without executing forbidden numerical routes."""
import unittest

from research.odt_reference.run_tests import audit


class GuardTests(unittest.TestCase):
    def test_runner_is_audited_even_with_no_numerical_modules(self):
        self.assertEqual(set(audit(())), {"run_tests.py"})

    def test_unlisted_local_import_fails_before_loading_it(self):
        with self.assertRaisesRegex(RuntimeError, "undeclared local dependency"):
            audit(("block.py",))
