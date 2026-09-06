"""New-lane compliance tests, separate from historical attestation fixtures."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import odt_direct_only_compliance as guard


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "xvla/train/odt_engine_v2"


class SplitSourceComplianceTests(unittest.TestCase):
    def setUp(self):
        self.prior_audited_sources = dict(guard._AUDITED_RUNTIME_SOURCES)

    def tearDown(self):
        guard._AUDITED_RUNTIME_SOURCES.clear()
        guard._AUDITED_RUNTIME_SOURCES.update(self.prior_audited_sources)

    def audit_fixture(self, module, text):
        with tempfile.TemporaryDirectory(prefix="odt-split-source-") as directory:
            root = Path(directory)
            path = root.joinpath(*module.split(".")).with_suffix(".py")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            return guard.audit_direct_only_launch(root, (path,), require_direct_qr=False)

    def test_moved_qr_and_single_evd_have_exact_source_owners(self):
        self.audit_fixture("xvla.train.odt_engine_v2.factorization",
                           "from torch.linalg import qr as factor\ndef _positive_diagonal_direct_rq_rows(x):\n    return factor(x.T, mode='reduced')\n")
        self.audit_fixture("xvla.train.odt_engine_v2.core",
                           "from torch.linalg import eigh as diagonalize\ndef _environment_eigensystem(record):\n    return diagonalize(record.mantissa)\n")
        self.assertIn(guard.SPLIT_DIRECT_RQ_RUNTIME_CALL, guard.ALLOWED_RUNTIME_CALLS)
        self.assertIn(guard.SPLIT_ALGORITHM3_RUNTIME_CALL, guard.ALLOWED_RUNTIME_CALLS)
        self.assertFalse(any("solve_triangular" in item or "implicit_sparse_projective_odt." in item
                             for item in guard.ALLOWED_RUNTIME_CALLS))
        self.assertEqual(guard.DIRECT_RQ_RUNTIME_CALL,
                         "xvla.train.implicit_sparse_projective_odt._positive_diagonal_direct_rq_rows:torch.linalg.qr")

    def test_retired_wrong_owner_and_unexecuted_prohibited_bindings_fail(self):
        fixtures = (
            ("xvla.train.implicit_sparse_projective_odt", "def _positive_diagonal_direct_rq_rows(x):\n    return torch.linalg.qr(x)"),
            ("xvla.train.odt_engine_v2.factorization", "def _solve_compact_q(r,x):\n    return torch.linalg.solve_triangular(r,x,upper=True)"),
            ("xvla.train.odt_engine_v2.factorization", "def _environment_eigensystem(x):\n    return torch.linalg.eigh(x)"),
            ("xvla.train.odt_engine_v2.core", "def wrong_owner(x):\n    return torch.linalg.eigh(x)"),
            ("xvla.train.odt_engine_v2.factorization", "from torch.linalg import svd as unused_forbidden"),
            ("xvla.train.odt_engine_v2.factorization", "unused_forbidden = torch.linalg.pinv"),
            ("xvla.train.odt_engine_v2.factorization", "def bad_overlap(q):\n    return q @ q.T"),
        )
        for module, source in fixtures:
            with self.subTest(module=module, source=source), self.assertRaises(guard.DirectOnlyComplianceError):
                self.audit_fixture(module, "import torch\n" + source + "\n")

    def test_primary_closure_is_transitive_hashed_and_excludes_oracles(self):
        entries = guard.canonical_direct_only_entrypoints(ROOT, SPLIT / "core.py")
        report = guard.audit_direct_only_launch(ROOT, entries)
        sources = report["source_sha256"]
        for name in ("core", "compiler", "ops", "factorization", "graph", "types", "validation", "diagnostics", "constants", "__init__"):
            key = f"xvla/train/odt_engine_v2/{name}.py"
            self.assertIn(key, sources)
            self.assertEqual(sources[key], hashlib.sha256((ROOT / key).read_bytes()).hexdigest())
        self.assertIn("xvla/nn/normalization.py", sources)
        self.assertIn("xvla/nn/tree_mixing.py", sources)
        self.assertNotIn("xvla/train/odt_engine_v2/oracles.py", sources)
        self.assertNotIn("xvla/train/direct_odt_clone_reference.py", sources)
        self.assertFalse(any(key.startswith("tests/") for key in sources))
        self.assertEqual(report["direct_qr_call_sites"], 1)
        self.assertEqual(report["prohibited_calls_found"], [])

    def test_dynamic_numerical_dispatch_is_rejected_without_execution(self):
        for source in (
            "name = '_linalg_svd'\ngetattr(torch.ops.aten, name).default(x)",
            "namespace = torch.ops.aten\nname = 'linalg_' + 'svd'\ngetattr(namespace, name).default(x)",
            "from builtins import getattr as lookup\nname = 'svd'\nlookup(torch.linalg, name)(x)",
        ):
            with self.subTest(source=source), self.assertRaises(guard.DirectOnlyComplianceError):
                self.audit_fixture("candidate", "import torch\n" + source)

    def test_failed_audit_revokes_previous_runtime_source_authority(self):
        guard.audit_direct_only_launch(ROOT, (SPLIT / "core.py",))
        self.assertTrue(guard._AUDITED_RUNTIME_SOURCES)
        with self.assertRaises(guard.DirectOnlyComplianceError):
            self.audit_fixture("xvla.train.odt_engine_v2.core", "import torch\ndef bad(x):\n    return torch.linalg.svd(x)\n")
        self.assertEqual(guard._AUDITED_RUNTIME_SOURCES, {})

    def test_explicit_oracle_import_adds_only_its_own_audited_authority(self):
        report = guard.audit_direct_only_launch(ROOT, (SPLIT / "core.py", SPLIT / "oracles.py"))
        self.assertIn("xvla/train/odt_engine_v2/oracles.py", report["source_sha256"])
        self.assertIn("xvla.train.odt_engine_v2.oracles", guard._AUDITED_RUNTIME_SOURCES)
        self.assertNotIn("xvla.train.direct_odt_clone_reference", guard._AUDITED_RUNTIME_SOURCES)

    def test_wrapper_blocks_unaudited_allowlisted_identity_before_any_original_call(self):
        called = []
        wrapped = guard._controlled_wrapper(lambda: called.append(True), "torch.linalg.qr", guard._DIRECT_QR_CALLERS)
        with patch.dict(guard._AUDITED_RUNTIME_SOURCES, {}, clear=True), \
                patch.object(guard, "_caller_identity", return_value=("xvla.train.odt_engine_v2.factorization", "_positive_diagonal_direct_rq_rows")), \
                patch.dict(guard._RUNTIME_STATE, {"prohibited_attempts": [], "allowed_calls": []}):
            with self.assertRaises(guard.DirectOnlyComplianceError):
                wrapped()
        self.assertEqual(called, [])

    def test_module_package_collision_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="odt-module-collision-") as directory:
            root = Path(directory)
            (root / "collision").mkdir()
            (root / "collision.py").write_text("# legacy source\n")
            (root / "collision/__init__.py").write_text("# new package\n")
            child = root / "collision/child.py"
            child.write_text("# directly named submodule\n")
            with self.assertRaisesRegex(guard.DirectOnlyComplianceError, "collision"):
                guard._resolve_local_module(root, "collision")
            with self.assertRaisesRegex(guard.DirectOnlyComplianceError, "parent.*collision"):
                guard.transitive_local_sources(root, (child,))

    def test_runtime_profile_is_explicit_and_preserves_historical_surface(self):
        self.assertEqual(len(guard.EXPECTED_RUNTIME_GUARD_ENTRYPOINTS), 87)
        self.assertEqual(len(guard.SPLIT_EXPECTED_RUNTIME_GUARD_ENTRYPOINTS), 89)
        self.assertEqual(guard.SPLIT_EXPECTED_RUNTIME_GUARD_ENTRYPOINTS - guard.EXPECTED_RUNTIME_GUARD_ENTRYPOINTS,
                         {"numpy.linalg.svdvals", "numpy.linalg.linalg.svdvals"})
        with self.assertRaises(guard.DirectOnlyComplianceError):
            guard.install_direct_only_runtime_guard(profile="auto")


_RUNTIME_PROBE = r'''
import json
from pathlib import Path
from scripts import odt_direct_only_compliance as guard
root = Path.cwd()
guard.audit_direct_only_launch(root, guard.canonical_direct_only_entrypoints(root, root / 'xvla/train/odt_engine_v2/core.py'))
guard.install_direct_only_runtime_guard(profile=guard.SPLIT_RUNTIME_PROFILE)
import torch
from xvla.train.odt_engine_v2.factorization import _positive_diagonal_direct_rq_rows
from xvla.train.odt_engine_v2.core import _environment_eigensystem
from xvla.train.odt_engine_v2.types import EnvironmentRecord
matrix = torch.tensor([[1., 2., 0.], [0., 3., 4.]], dtype=torch.float64)
r, q = _positive_diagonal_direct_rq_rows(matrix)
assert float((r @ q - matrix).abs().max()) < 1e-12
environment = EnvironmentRecord(0, 'manual', torch.diag(torch.tensor([1.,3.],dtype=torch.float64)), 0, 0, 0)
values, vectors = _environment_eigensystem(environment)
assert float((environment.mantissa @ vectors - vectors * values).abs().max()) < 1e-12
before = guard.direct_only_runtime_report()
guard.assert_direct_only_runtime_guard(before, exact_allowed_calls=(guard.SPLIT_DIRECT_RQ_RUNTIME_CALL, guard.SPLIT_ALGORITHM3_RUNTIME_CALL))
blocked = []
for name in ('svd', 'pinv', 'lstsq', 'solve_triangular', 'qr', 'eigh'):
    try:
        getattr(torch.linalg, name)(matrix)
    except guard.DirectOnlyComplianceError:
        blocked.append(name)
assert len(blocked) == 6
guard._AUDITED_RUNTIME_SOURCES.clear()
try:
    _positive_diagonal_direct_rq_rows(matrix)
except guard.DirectOnlyComplianceError:
    blocked.append('unaudited_moved_qr')
assert len(blocked) == 7
print(json.dumps({'allowed_calls': before['allowed_calls'], 'blocked_before_execution': blocked,
                  'patched_entrypoint_count': before['patched_entrypoint_count']}))
'''


class SplitRuntimeComplianceTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "guarded torch runtime is not installed in this interpreter")
    def test_real_moved_qr_evd_and_prohibited_routes_under_fresh_guard(self):
        result = subprocess.run([sys.executable, "-c", _RUNTIME_PROBE], cwd=ROOT,
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        evidence = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(len(evidence["blocked_before_execution"]), 7)
        self.assertEqual(evidence["patched_entrypoint_count"], guard.SPLIT_EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT)


if __name__ == "__main__":
    unittest.main()
