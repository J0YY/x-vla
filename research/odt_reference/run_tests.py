"""Guarded, local-only runner. Execute from the repository root with -m."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
PROHIBITED = {
    "svd", "svdvals", "pinv", "pinvh", "lstsq", "matrix_rank", "cond",
    "polar", "cov", "corrcoef", "cholesky", "inv", "solve", "det", "slogdet",
    "eig", "eigvals", "eigvalsh", "matrix_power", "tensorsolve", "tensorinv",
    "norm", "matrix_norm", "vector_norm",
}
COUNTS = {"qr": 0, "eigh": 0, "prohibited_attempts": 0}
SUITES = {
    "tree": ("dooms.py", "weights.py", "test_reference.py"),
    "dag": ("dooms.py", "shared_dag.py", "clone_oracle.py", "test_shared_dag.py"),
    "block": ("dooms.py", "weights.py", "shared_dag.py", "clone_oracle.py", "clone_passes.py",
              "block.py", "block_oracle.py", "test_block_oracle.py", "test_block.py", "test_shared_dag.py"),
    "checkpoint": ("checkpoint.py", "test_checkpoint.py"),
    "guard": ("test_guard.py",),
}
BOUNDARY_FILES = {"checkpoint.py", "test_checkpoint.py", "run_checkpoint_gate.py", "run_tests.py"}


def audit(files):
    """Bounded source audit, not a general Python security/soundness proof."""
    digests = {}
    declared = set(files) | {"run_tests.py"}
    for name in sorted(declared):
        path = HERE / name
        tree = ast.parse(path.read_text())
        digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for node in ast.walk(tree):
            local = []
            if isinstance(node, ast.Import):
                local = [x.name.rsplit(".", 1)[-1] + ".py" for x in node.names
                         if x.name.startswith("research.odt_reference.")]
            if isinstance(node, ast.ImportFrom):
                if node.module == "research.odt_reference":
                    local = [x.name + ".py" for x in node.names]
                elif (node.module or "").startswith("research.odt_reference."):
                    local = [node.module.rsplit(".", 1)[-1] + ".py"]
            if any(name not in declared for name in local):
                raise RuntimeError(f"undeclared local dependency: {path.name}:{node.lineno}: {local}")
            if isinstance(node, ast.Import):
                allowed = {"numpy", "itertools", "unittest", "dataclasses", "copy", "math"}
                if path.name in BOUNDARY_FILES:
                    allowed |= {"io", "pickle", "pickletools", "zipfile", "argparse", "ast", "hashlib", "json"}
                if any(x.name.split(".")[0] not in allowed for x in node.names):
                    raise RuntimeError(f"unreviewed dependency: {path.name}:{node.lineno}")
            if isinstance(node, ast.ImportFrom) and not (node.module or "").startswith(
                    ("research.odt_reference", "dataclasses", "typing", "__future__") +
                    (("collections", "unittest.mock", "pathlib") if path.name in BOUNDARY_FILES else ())):
                raise RuntimeError(f"unreviewed dependency: {path.name}:{node.lineno}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in PROHIBITED:
                raise RuntimeError(f"prohibited call site: {path.name}:{node.lineno}")
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                for plain, transposed in ((node.left, node.right), (node.right, node.left)):
                    if isinstance(transposed, ast.Attribute) and transposed.attr == "T" and ast.dump(plain) == ast.dump(transposed.value):
                        raise RuntimeError(f"self-overlap diagnostic: {path.name}:{node.lineno}")
    return digests


def install_guards():
    def reject(*args, **kwargs):
        COUNTS["prohibited_attempts"] += 1
        raise RuntimeError("prohibited numerical route")
    for namespace in (np, np.linalg):
        for name in PROHIBITED:
            if hasattr(namespace, name):
                setattr(namespace, name, reject)
    for name in ("qr", "eigh"):
        original = getattr(np.linalg, name)
        def counted(*args, _name=name, _original=original, **kwargs):
            COUNTS[_name] += 1
            return _original(*args, **kwargs)
        setattr(np.linalg, name, counted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=(*SUITES, "all"), default="all")
    selected = parser.parse_args().suite
    files = set(sum(SUITES.values(), ())) if selected == "all" else set(SUITES[selected])
    sources = audit(files)
    install_guards()
    modules = ["research.odt_reference." + name[:-3] for name in sorted(files) if name.startswith("test_")]
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.wasSuccessful() and COUNTS["prohibited_attempts"] == 0
    print(json.dumps({"passed": passed, "suite": selected, "tests": result.testsRun, "numpy": np.__version__,
                      "kernels": COUNTS, "source_sha256": sources}, sort_keys=True))
    raise SystemExit(0 if passed else 1)
