#!/usr/bin/env python3
"""Create the immutable ProductRoutingHead launch manifest, without starting work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from athena.product_rational_protocol import (
    AUTHENTICATED_INPUTS,
    DEFAULT_RUN_ROOT,
    LAUNCH_CLOSURE,
    SOURCE_CLOSURE,
    assert_cache_identity,
    assert_outputs_absent,
    closure_snapshot,
    file_sha256,
    load_and_verify_source_manifest,
    preflight_result_path,
    source_manifest_path,
    source_snapshot,
    validate_preflight_certificate,
    write_source_manifest_exclusive,
)
from scripts.odt_direct_only_compliance import audit_direct_only_launch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--action", choices=("create-manifest", "verify-preflight"), required=True
    )
    parser.add_argument("--run-root", type=Path, required=True)
    arguments = sys.argv[1:]
    parsed = parser.parse_args(arguments)
    for option in ("--action", "--run-root"):
        count = sum(
            argument == option or argument.startswith(f"{option}=")
            for argument in arguments
        )
        if count != 1:
            parser.error(f"{option} must occur exactly once")
    return parsed


def validate_frozen_run_root(path: Path) -> Path:
    if path != DEFAULT_RUN_ROOT:
        raise RuntimeError(
            f"Run root must be the frozen path {DEFAULT_RUN_ROOT}, got {path}"
        )
    candidate = PROJECT_ROOT / path
    if candidate.is_symlink():
        raise RuntimeError("Frozen run root may not be a link")
    candidate.mkdir(parents=True, exist_ok=True)
    resolved = candidate.resolve()
    if PROJECT_ROOT not in resolved.parents:
        raise RuntimeError("Frozen run root escapes the project")
    return resolved


def audit_text_launch_closure() -> dict[str, object]:
    blocked = (
        "s" + "vd",
        "g" + "ram",
        "p" + "olar",
        "c" + "ovariance",
        "normal" + "-equation",
        "normal" + " equation",
        "p" + "inv",
        "l" + "stsq",
    )
    observed = closure_snapshot(LAUNCH_CLOSURE)
    findings: list[str] = []
    for relative in LAUNCH_CLOSURE:
        text = (PROJECT_ROOT / relative).read_text().lower()
        for token in blocked:
            if token in text:
                findings.append(f"{relative}:{token}")
    if findings:
        raise RuntimeError(f"Launch text closure contains blocked routes: {findings}")
    return {
        "paths": list(LAUNCH_CLOSURE),
        "sha256": observed,
        "blocked_route_occurrences": findings,
    }


def main() -> None:
    args = parse_args()
    run_root = validate_frozen_run_root(args.run_root)
    manifest_path = source_manifest_path(run_root)
    assert_outputs_absent(run_root, include_smoke=True)
    cache = assert_cache_identity()
    static_audit = audit_direct_only_launch(
        PROJECT_ROOT,
        tuple(PROJECT_ROOT / relative for relative in SOURCE_CLOSURE),
        require_direct_qr=False,
    )
    if static_audit["source_sha256"] != source_snapshot():
        raise RuntimeError("Manual source snapshot differs from transitive static audit")
    for field in (
        "prohibited_calls_found",
        "prohibited_self_overlap_sites",
        "guarded_dormant_spectral_norm_sites",
    ):
        if static_audit[field] != []:
            raise RuntimeError(f"Product lane static audit did not close {field}")
    launch_text_audit = audit_text_launch_closure()
    inputs = closure_snapshot(AUTHENTICATED_INPUTS)
    if args.action == "create-manifest":
        if manifest_path.exists() or manifest_path.is_symlink():
            raise FileExistsError(f"Refusing existing source manifest {manifest_path}")
        manifest = write_source_manifest_exclusive(manifest_path)
        if load_and_verify_source_manifest(manifest_path) != manifest:
            raise RuntimeError("Written source manifest did not verify exactly")
        os.chmod(manifest_path.parent, 0o555)
        preflight = None
    else:
        manifest = load_and_verify_source_manifest(manifest_path)
        preflight = validate_preflight_certificate(
            preflight_result_path(run_root), manifest_path, manifest
        )
    payload = {
        "ready_for_cpu_preflight_only": True,
        "run_root": run_root.as_posix(),
        "source_manifest": manifest_path.as_posix(),
        "source_manifest_sha256": file_sha256(manifest_path),
        "manifest_bundle_sha256": manifest["manifest_bundle_sha256"],
        "cache": cache,
        "authenticated_inputs": inputs,
        "static_audit": static_audit,
        "launch_text_audit": launch_text_audit,
        "no_job_submitted": True,
        "action": args.action,
        "preflight": preflight,
    }
    print("MANIFEST", json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
