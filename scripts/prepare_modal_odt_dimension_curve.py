"""Freeze only the audited local closure needed by the Modal ODT worker."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.odt_direct_only_compliance import audit_direct_only_launch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    options = parser.parse_args()
    destination = options.destination.resolve()
    if destination.exists():
        raise FileExistsError("freeze destination already exists")
    worker = ROOT / "modal_odt_dimension_curve_worker.py"
    audit = audit_direct_only_launch(ROOT, [worker])
    source_map = dict(audit["source_sha256"])
    destination.mkdir(parents=True)
    for relative, digest in source_map.items():
        original = ROOT / relative
        if hashlib.sha256(original.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"source changed during freeze: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    for relative in ("modal_odt_dimension_curve_controller.py", "scripts/odt_direct_only_compliance.py"):
        original = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        source_map[relative] = hashlib.sha256(original.read_bytes()).hexdigest()
    authority = ROOT / "scripts/run_capable_linear_fresh_capability.py"
    constants = {}
    for node in ast.parse(authority.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"EXPECTED_TASK_PROTOCOL", "EXPECTED_CONFIG_IDENTITY", "EXPECTED_VERSIONS"}:
                constants[name] = ast.literal_eval(node.value)
    if len(constants) != 3:
        raise RuntimeError("protocol authority omitted required literal fields")
    protocol = {"schema": "xvla_modal_fixed20_panel_v1", "tasks": constants["EXPECTED_TASK_PROTOCOL"],
        "configuration": constants["EXPECTED_CONFIG_IDENTITY"], "versions": constants["EXPECTED_VERSIONS"],
        "authority_sha256": hashlib.sha256(authority.read_bytes()).hexdigest(),
        "pairs": [[task, episode] for task in range(10) for episode in range(2)],
        "max_steps": 280, "settle_steps": 10, "action_horizon": 8,
        "nominal_internal_dimension_removal_percent": [30, 40, 50, 60, 70, 80]}
    protocol_bytes = (json.dumps(protocol, sort_keys=True, indent=2) + "\n").encode()
    (destination / "protocol.json").write_bytes(protocol_bytes)
    source_map["protocol.json"] = hashlib.sha256(protocol_bytes).hexdigest()
    manifest = {"schema": "xvla_modal_odt_source_bundle_v1", "files": source_map,
        "static_audit": audit, "controller_review_required": True}
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    (destination / "bundle.json").write_bytes(manifest_bytes)
    print(json.dumps({"destination": str(destination), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "file_count": len(source_map), "protocol_sha256": source_map["protocol.json"]}, sort_keys=True))


if __name__ == "__main__":
    main()
