"""Check that a secondary protocol cannot alter its inherited data contract."""
import unittest
from research.odt_reference.run_tests import install_guards, COUNTS
install_guards()
from scripts.odt_campaign_fine_v1 import BUDGETS, audit_extension, extended_protocol


class FineTests(unittest.TestCase):
    def test_only_conditions_and_extension_change(self):
        parent = {"conditions": [{"old": True}], "contrasts": [1], "inputs_sha256": "a",
                  "records_sha256": "b", "sources": {"unchanged": True}, "version": "odt-campaign-v1"}
        result = extended_protocol(parent, [{"new": True}], [], {"source": "hash"}, "parent")
        self.assertEqual(parent["conditions"], [{"old": True}])
        for key in set(parent)-{"conditions", "contrasts"}:
            self.assertEqual(parent[key], result[key])
        self.assertEqual(result["extension"]["budgets"], [1, 9, 46])
        self.assertTrue(result["extension"]["confirmation_reused"])
        self.assertEqual(result["extension"]["parent_protocol_sha256"], "parent")

    def test_source_audit_and_predeclared_budgets(self):
        self.assertEqual(BUDGETS, (1, 9, 46))
        self.assertEqual(len(audit_extension()["extension_sources"]), 2)
        self.assertEqual(COUNTS["prohibited_attempts"], 0)


if __name__ == "__main__":
    unittest.main()
