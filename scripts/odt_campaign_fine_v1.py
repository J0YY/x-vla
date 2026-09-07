"""Freeze predeclared sub-one-percent cuts on the identical authenticated panel."""
import argparse
import ast
import json
from pathlib import Path
import shutil

import numpy as np

from research.odt_campaign_v1.campaign import (
    audit_campaign, check_runtime, freeze_conditions, load_campaign,
)
from research.odt_reference.run_tests import COUNTS, PROHIBITED, install_guards
from scripts.odt_rank_ablation import load_accepted
from research.odt_reference import run_curve as original


BUDGETS = (1, 9, 46)


def audit_extension():
    here = Path(__file__).resolve()
    files = (here, here.with_name("test_odt_campaign_fine_v1.py"))
    allowed = {"argparse", "ast", "json", "shutil", "numpy", "unittest"}
    dependencies = {"pathlib", "research.odt_campaign_v1.campaign",
                    "research.odt_reference.run_tests", "scripts.odt_rank_ablation",
                    "research.odt_reference", "scripts.odt_campaign_fine_v1"}
    hashes = {}
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import) and any(item.name not in allowed for item in node.names):
                raise ValueError("unreviewed extension import")
            if isinstance(node, ast.ImportFrom) and node.module not in dependencies:
                raise ValueError("unreviewed extension dependency")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in PROHIBITED:
                raise ValueError("prohibited extension numerical route")
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                raise ValueError("extension must not perform contractions")
        hashes[str(path.relative_to(here.parents[1]))] = original.digest(path)
    return {"extension_sources": hashes, "consumer_sources": audit_campaign()}


def extended_protocol(parent, conditions, contrasts, sources, parent_sha):
    """Reuse data and mathematics exactly, changing only frozen conditions."""
    result = dict(parent)
    result["conditions"], result["contrasts"] = conditions, contrasts
    result["extension"] = {
        "schema": "odt-fine-cut-extension-v1", "budgets": list(BUDGETS),
        "parent_protocol_sha256": parent_sha, "sources": sources,
        "selection": "K=1,9,46 prespecified in ODT_4H_EXPERIMENT_PLAN_2026-09-07.md",
        "confirmation_reused": True, "interpretation": "descriptive secondary analysis",
        "same_panel_baseline_and_bases": True,
    }
    return result


def prepare(parent, producer, output):
    sources = audit_extension()
    check_runtime()
    manifest, protocol = load_campaign(parent)
    receipt, canonical, _, _, _ = load_accepted(producer, protocol["sources"]["accepted_consumer"])
    if original.digest(producer/"accepted.json") != protocol["producer_receipt_sha256"]:
        raise ValueError("parent producer identity differs")
    with np.load(parent/"bases.npz", allow_pickle=False) as data:
        spectra = tuple(data[f"spectrum_{i}"] for i in range(len(canonical.nodes)))
    conditions, contrasts = freeze_conditions(canonical, spectra, budgets=BUDGETS, random_seeds=())
    revised = extended_protocol(protocol, conditions, contrasts, sources, original.digest(parent/"protocol.json"))
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in manifest["artifacts"].items():
        if name == "protocol.json":
            continue
        shutil.copyfile(parent/name, output/name)
        if original.digest(output/name) != expected:
            raise ValueError("copied immutable artifact identity differs")
    original.write_json(output/"protocol.json", revised)
    revised_manifest = dict(manifest)
    revised_manifest["artifacts"] = dict(manifest["artifacts"])
    revised_manifest["artifacts"]["protocol.json"] = original.digest(output/"protocol.json")
    revised_manifest["condition_count"] = len(conditions)
    revised_manifest["extension"] = revised["extension"]
    revised_manifest["kernels"] = dict(COUNTS)
    if audit_extension() != sources or COUNTS["prohibited_attempts"]:
        raise ValueError("extension source or runtime changed")
    original.write_json(output/"manifest.json", revised_manifest)
    load_campaign(output)
    print(json.dumps({"conditions": len(conditions), "protocol_sha256": original.digest(output/"protocol.json"),
                      "parent_manifest_sha256": original.digest(parent/"manifest.json"), "kernels": dict(COUNTS)}))


def main():
    audit_extension()
    install_guards()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "producer", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.parent, args.producer, args.output)


if __name__ == "__main__":
    main()
