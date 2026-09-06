#!/usr/bin/env python3
"""Run the new dimension-ladder tests with the direct-only guard installed."""

from __future__ import annotations

import json
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

from scripts.odt_direct_only_compliance import (
    assert_direct_only_runtime_guard,
    audit_direct_only_launch,
    direct_only_runtime_report,
    install_direct_only_runtime_guard,
)

ENTRYPOINTS = tuple(ROOT / relative for relative in (
    "tests/test_direct_odt_dimension_ladder.py",
    "tests/test_direct_odt_dimension_curve_runner.py",
    "tests/test_implicit_projective_dag_artifact.py",
    "tests/test_implicit_projective_dag_mapped.py",
))
audit = audit_direct_only_launch(ROOT, ENTRYPOINTS)
for field in ("prohibited_calls_found", "prohibited_self_overlap_sites", "guarded_dormant_spectral_norm_sites", "duplicate_top_level_definition_sites"):
    if audit[field]:
        raise RuntimeError(f"direct-only audit rejected {field}: {audit[field]}")
install_direct_only_runtime_guard()

import pytest
import torch

torch.set_num_threads(1)
code = pytest.main([*(str(path) for path in ENTRYPOINTS), "-q", "-p", "no:cacheprovider", "--junitxml=" + str(ROOT / "results/ladder_unit.xml")])
runtime = assert_direct_only_runtime_guard(direct_only_runtime_report())
if audit_direct_only_launch(ROOT, ENTRYPOINTS) != audit:
    raise RuntimeError("sources changed during the test run")
ledger = ROOT / "stage.sha256"
payload = {"pytest_exit_code": code, "static_audit": audit, "runtime_guard": runtime,
           "stage_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest()}
with (ROOT / "results/unit.json").open("x") as stream:
    json.dump(payload, stream, sort_keys=True)
print(json.dumps(payload, sort_keys=True), flush=True)
raise SystemExit(code)
