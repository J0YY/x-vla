"""One local command for independent mathematics and production acceptance tests.

Run with the pinned requirements in xvla/train/odt_engine_v2. No checkpoints,
cluster jobs, or deployment actions are performed. Results are saved even on
test failure, never promoted to a full-policy certificate.
"""

from datetime import datetime, timezone
from importlib.metadata import version
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NUMERICAL_TESTS = (
    "test_odt_symmetric_factorization", "test_odt_refactor_integration",
    "test_odt_acceptance", "test_implicit_sparse_projective_odt",
    "test_direct_odt_truncation", "test_direct_odt_dimension_ladder",
    "test_implicit_projective_dag_artifact", "test_implicit_projective_dag_mapped",
    "test_implicit_sparse_projective_odt_all_tokens", "test_implicit_sparse_projective_odt_vla",
)
STATIC_TESTS = ("test_odt_refactor_structure", "test_odt_split_compliance", "test_odt_refactor_runner")


def _source_snapshot():
    """Hash the numerical import closure and independent reference before/after."""
    from scripts.odt_direct_only_compliance import audit_direct_only_launch
    entries = [ROOT / "scripts/run_odt_refactor_tests.py",
               *(ROOT / "xvla/train/odt_engine_v2" / (name + ".py") for name in ("core", "compiler", "oracles")),
               *(ROOT / "tests" / (name + ".py") for name in NUMERICAL_TESTS)]
    sources = audit_direct_only_launch(ROOT, entries)["source_sha256"]
    production_sources = dict(sources)
    files = {Path(__file__).resolve(), *(ROOT / "tests" / (name + ".py") for name in STATIC_TESTS),
             *(ROOT / "research/odt_reference").glob("*.py"),
             ROOT / "xvla/train/odt_engine_v2/README.md", ROOT / "xvla/train/odt_engine_v2/requirements-validation.txt"}
    sources.update({str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    # Read the independent runner's literal suite declaration without importing
    # its NumPy implementation into this process.
    tree = ast.parse((ROOT / "research/odt_reference/run_tests.py").read_text())
    declaration = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "SUITES" for target in node.targets))
    reference_names = set(sum(ast.literal_eval(declaration).values(), ())) | {"run_tests.py"}
    reference_sources = {name: sources["research/odt_reference/" + name] for name in reference_names}
    return sources, {"production": production_sources, "independent_reference": reference_sources}


def _reports_match_sources(records, expected):
    seen = set()
    for record in records:
        name = record["name"]
        if name == "source_and_boundary":
            continue
        if name not in expected or name in seen:
            return False
        seen.add(name)
        payload = record.get("report", {})
        hashes = payload.get("source_sha256") if name == "independent_reference" else payload.get("sources", {}).get("source_sha256")
        if not expected[name] or hashes != expected[name]:
            return False
    return seen == set(expected)


def _run(name, arguments, environment):
    command = [sys.executable, "-m", *arguments]
    try:
        result = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, timeout=1800)
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        result = subprocess.CompletedProcess(command, 124, output, "Validation subprocess exceeded 1800 seconds")
    record = {"name": name, "command": command, "exit_code": result.returncode,
              "stdout": result.stdout, "stderr": result.stderr}
    # Numerical runners emit one JSON object. Key order is not part of the
    # protocol, and unittest stdout may contain deliberately failing fixtures.
    if name != "source_and_boundary":
        for line in result.stdout.splitlines():
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and isinstance(payload.get("passed"), bool):
                    record["report"] = payload
    count = re.search(r"Ran (\d+) tests", result.stderr) or re.search(r"(\d+) passed", result.stdout)
    record["tests_passed"] = int(count[1]) if count and result.returncode == 0 else 0
    if "report" in record and name == "independent_reference":
        record["tests_passed"] = record["report"]["tests"] if record["report"]["passed"] else 0
    has_required_report = name == "source_and_boundary" or "report" in record
    record["passed"] = result.returncode == 0 and has_required_report and record.get("report", {}).get("passed", True)
    print(f"{name}: {'PASS' if record['passed'] else 'FAIL'} ({record['tests_passed']} tests)", flush=True)
    if not record["passed"]:
        print(result.stdout + result.stderr, flush=True)
    return record


def main():
    expected = {"numpy": "2.2.6", "torch": "2.8.0", "pytest": "8.4.2"}
    actual = {name: version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"pinned validation dependencies required: {expected}, found {actual}")
    environment = dict(os.environ, OMP_NUM_THREADS="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_PLUGINS", None)
    source_hashes, expected_reports = _source_snapshot()
    # Separate processes prevent the independent NumPy and production Torch
    # guards from sharing implementation state or replacing each other's hooks.
    jobs = (
        ("independent_reference", ["research.odt_reference.run_tests", "--suite", "all"]),
        ("source_and_boundary", ["unittest", *("tests." + name for name in STATIC_TESTS)]),
        ("production", ["scripts.run_odt_refactor_tests", *("tests/" + name + ".py" for name in NUMERICAL_TESTS),
                        "-k", "not compiled and not numba and not all_six_physical_prefixes", "--tb=short"]),
    )
    records = [_run(name, arguments, environment) for name, arguments in jobs]
    source_error = None
    try:
        sources_unchanged = _source_snapshot() == (source_hashes, expected_reports) and _reports_match_sources(records, expected_reports)
    except (OSError, RuntimeError, ValueError) as error:
        sources_unchanged, source_error = False, str(error)
    report = {"schema": "odt_proof_acceptance_v1", "completed_at": datetime.now(timezone.utc).isoformat(),
              "passed": sources_unchanged and all(record["passed"] for record in records),
              "tests_passed": sum(record["tests_passed"] for record in records),
              "versions": actual, "python": sys.version,
              "scope": "bounded reference and production fixtures only, no trained-checkpoint or full-policy acceptance",
              "excluded": "five optional Numba compiled-executor tests, dependency unavailable",
              "sources_unchanged": sources_unchanged, "source_error": source_error, "source_sha256": source_hashes,
              "runs": records}
    destination = ROOT / "xvla/train/odt_engine_v2/proof_validation.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Saved {destination}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
