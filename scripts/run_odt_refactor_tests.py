"""Local, audited CPU validation of the symmetric ODT engine refactor.

No checkpoint loading, cluster launch or production sweep occurs here. Use the
explicit pinned Torch 2.8.0 / NumPy 2.2.6 runtime. Arguments are pytest targets.
"""

from pathlib import Path
import argparse
import json
import os
import sys

from scripts.odt_direct_only_compliance import (
    SPLIT_RUNTIME_PROFILE, audit_direct_only_launch,
    direct_only_runtime_report, install_direct_only_runtime_guard,
)


def _pytest_arguments(root, arguments):
    """Whitelist selection options, never forward arbitrary pytest arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+")
    parser.add_argument("-k", default="")
    parser.add_argument("--tb", choices=("short", "long", "line", "no"), default="short")
    parser.add_argument("-x", action="store_true")
    parsed = parser.parse_args(arguments or ["tests/test_odt_symmetric_factorization.py"])
    paths = tuple((root / target).resolve() for target in parsed.targets)
    if any(path.suffix != ".py" or not path.is_file() or not path.is_relative_to(root / "tests") for path in paths):
        raise ValueError("only explicit existing repository test .py files are allowed")
    # No implicit conftest, configuration, or environment-specified plugins.
    args = ["-q", "--noconftest", "-c", os.devnull, "--rootdir", str(root),
            "--tb=" + parsed.tb, "-k", parsed.k, *map(str, paths)]
    return paths, args + (["-x"] if parsed.x else [])


def main():
    root = Path(__file__).resolve().parents[1]
    paths, targets = _pytest_arguments(root, sys.argv[1:])
    entries = (
        Path(__file__).resolve(),
        root / "xvla/train/odt_engine_v2/core.py",
        root / "xvla/train/odt_engine_v2/compiler.py",
        root / "xvla/train/odt_engine_v2/oracles.py",
        *paths,
    )
    sources = audit_direct_only_launch(root, entries)
    install_direct_only_runtime_guard(profile=SPLIT_RUNTIME_PROFILE)
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    import pytest
    class AuditIsolation:
        def pytest_runtest_setup(self, item):
            # Some tests intentionally run narrower or rejected source audits.
            # Restore this test launch's complete authority before each case.
            current = audit_direct_only_launch(root, entries)
            if current["source_sha256"] != sources["source_sha256"]:
                raise RuntimeError("source changed during the guarded test run")
    code = pytest.main(targets, plugins=[AuditIsolation()])
    runtime = direct_only_runtime_report()
    passed = code == 0 and not runtime["prohibited_attempts"]
    print(json.dumps({"passed": passed, "sources": sources, "runtime": runtime}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
