"""Verify a frozen source manifest and issue a unit-test certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


UNIT_CERTIFICATE_SCHEMA = "xvla-canonical-odt-unit-certificate-v4"
REQUIRED_AUTHENTICATION_COPIES = (
    "producer", "stored_npz_cpu", "independent_checkpoint_raw_gauge_cpu",
)
SMOKE_REQUIRED_AUTHENTICATION_IDENTITIES = tuple(
    (trial, bond, 1) for trial in range(50) for bond in range(4)
)


def validate_required_authentication_decisions(
    value: dict,
    *,
    expected_identities: tuple[tuple[int, int, int], ...] | None = None,
) -> list[tuple[int, int, int]]:
    """Validate every load-bearing decision without equating residual magnitudes."""

    expected_top = {
        "comparison_rule", "frozen_thresholds", "required_certificate_count",
        "expected_required_certificate_count", "copies", "worst_of_all_copies",
        "decision_inventory", "non_load_bearing_conditioning_diagnostics",
    }
    if not isinstance(value, dict) or set(value) != expected_top:
        raise RuntimeError("required authentication field inventory differs")
    thresholds = value.get("frozen_thresholds")
    if thresholds != {
        "maximum_bound_with_roundoff": 1e-6,
        "maximum_supported_overlap_defect": 1e-8,
    }:
        raise RuntimeError("required authentication thresholds differ")
    inventory = value.get("decision_inventory")
    expected_count = value.get("expected_required_certificate_count")
    if (
        value.get("comparison_rule")
        != "strict-producer-to-stored-npz-plus-independent-required-gate-replication-v1"
        or type(expected_count) is not int
        or expected_count <= 0
        or value.get("required_certificate_count") != expected_count
        or not isinstance(inventory, list)
        or len(inventory) != expected_count
    ):
        raise RuntimeError("required authentication count or policy differs")
    identities: list[tuple[int, int, int]] = []
    decisions_by_copy: dict[str, list[dict]] = {
        name: [] for name in REQUIRED_AUTHENTICATION_COPIES
    }
    decision_keys = {
        "numerically_certifiable", "passes_davis_kahan_inequality",
        "bound_with_roundoff", "bound_with_roundoff_le_maximum",
        "maximum_supported_overlap_defect",
        "maximum_supported_overlap_defect_le_1e-8", "all_frozen_gates_pass",
    }
    for item in inventory:
        if not isinstance(item, dict) or set(item) != {"trial", "bond", "rank", "copies"}:
            raise RuntimeError("required authentication decision entry differs")
        identity = (item["trial"], item["bond"], item["rank"])
        if any(type(part) is not int or part < 0 for part in identity):
            raise RuntimeError("required authentication decision identity differs")
        copies = item["copies"]
        if not isinstance(copies, dict) or tuple(copies) != REQUIRED_AUTHENTICATION_COPIES:
            raise RuntimeError("required authentication copy inventory differs")
        identities.append(identity)
        for name in REQUIRED_AUTHENTICATION_COPIES:
            decision = copies[name]
            if not isinstance(decision, dict) or set(decision) != decision_keys:
                raise RuntimeError("required authentication decision fields differ")
            bound = decision["bound_with_roundoff"]
            support = decision["maximum_supported_overlap_defect"]
            if (
                any(
                    decision[key] is not True
                    for key in (
                        "numerically_certifiable", "passes_davis_kahan_inequality",
                        "bound_with_roundoff_le_maximum",
                        "maximum_supported_overlap_defect_le_1e-8",
                        "all_frozen_gates_pass",
                    )
                )
                or type(bound) not in (int, float)
                or not math.isfinite(bound)
                or not 0.0 <= bound <= 1e-6
                or type(support) not in (int, float)
                or not math.isfinite(support)
                or not 0.0 <= support <= 1e-8
            ):
                raise RuntimeError("required authentication decision failed")
            decisions_by_copy[name].append(decision)
    if len(set(identities)) != len(identities) or identities != sorted(identities):
        raise RuntimeError("required authentication decision identities differ")
    if expected_identities is not None and tuple(identities) != expected_identities:
        raise RuntimeError("required authentication decision identities differ from frozen scope")
    aggregates = value.get("copies")
    if not isinstance(aggregates, dict) or tuple(aggregates) != REQUIRED_AUTHENTICATION_COPIES:
        raise RuntimeError("required authentication aggregate copies differ")
    aggregate_keys = {
        "certificate_count", "all_numerically_certifiable",
        "all_pass_davis_kahan_inequality", "all_bounds_le_maximum",
        "all_support_defects_le_1e-8", "all_frozen_gates_pass",
        "maximum_bound_with_roundoff", "maximum_supported_overlap_defect",
    }
    for name in REQUIRED_AUTHENTICATION_COPIES:
        aggregate = aggregates[name]
        decisions = decisions_by_copy[name]
        if (
            not isinstance(aggregate, dict)
            or set(aggregate) != aggregate_keys
            or aggregate["certificate_count"] != expected_count
            or any(
                aggregate[key] is not True
                for key in (
                    "all_numerically_certifiable", "all_pass_davis_kahan_inequality",
                    "all_bounds_le_maximum", "all_support_defects_le_1e-8",
                    "all_frozen_gates_pass",
                )
            )
            or aggregate["maximum_bound_with_roundoff"]
            != max(item["bound_with_roundoff"] for item in decisions)
            or aggregate["maximum_supported_overlap_defect"]
            != max(item["maximum_supported_overlap_defect"] for item in decisions)
        ):
            raise RuntimeError("required authentication aggregate differs")
    worst = value.get("worst_of_all_copies")
    if worst != {
        "maximum_bound_with_roundoff": max(
            aggregates[name]["maximum_bound_with_roundoff"]
            for name in REQUIRED_AUTHENTICATION_COPIES
        ),
        "maximum_supported_overlap_defect": max(
            aggregates[name]["maximum_supported_overlap_defect"]
            for name in REQUIRED_AUTHENTICATION_COPIES
        ),
        "all_frozen_gates_pass": True,
    }:
        raise RuntimeError("required authentication worst-of-copies aggregate differs")
    diagnostics = value.get("non_load_bearing_conditioning_diagnostics")
    if (
        not isinstance(diagnostics, dict)
        or set(diagnostics) != {
            "comparison_policy", "records", "producer_maxima",
            "independent_checkpoint_npz_cpu_maxima", "worst_of_both_maxima",
        }
        or diagnostics.get("comparison_policy")
        != "report-both-cross-device-paths-without-a-pass-threshold-v1"
        or not isinstance(diagnostics.get("records"), list)
        or not diagnostics["records"]
    ):
        raise RuntimeError("non-load-bearing conditioning diagnostics differ")
    for key in ("producer_maxima", "independent_checkpoint_npz_cpu_maxima", "worst_of_both_maxima"):
        values = diagnostics.get(key)
        if not isinstance(values, dict) or set(values) != {
            "raw_coordinate_replay_relative", "canonicalized_function_relative"
        } or any(type(number) not in (int, float) or not math.isfinite(number) for number in values.values()):
            raise RuntimeError("conditioning diagnostic maxima differ")
    return identities


RESULT_SOURCE_PATHS = (
    "athena/run_svhn_canonical_odt.py",
    "athena/CANONICAL_ODT_PHASE1_PROTOCOL.md",
    "athena/slurm_canonical_odt_tests.sbatch",
    "athena/slurm_svhn_canonical_odt_prefetch.sbatch",
    "athena/slurm_svhn_canonical_odt_smoke.sbatch",
    "athena/slurm_svhn_canonical_odt_smoke_validate.sbatch",
    "athena/slurm_svhn_canonical_odt.sbatch",
    "athena/slurm_svhn_canonical_odt_summary.sbatch",
    "athena/summarize_svhn_canonical_odt.py",
    "athena/validate_svhn_canonical_odt_smoke.py",
    "athena/verify_canonical_odt_freeze.py",
    "athena/launch_svhn_canonical_odt.py",
    "tests/test_canonical_odt.py",
    "tests/test_canonical_odt_pipeline.py",
    "xvla/models/chi_mlp.py",
    "xvla/nn/bilinear.py",
    "xvla/nn/homogeneous.py",
    "xvla/nn/normalization.py",
    "xvla/train/canonical_odt.py",
    "xvla/train/odt.py",
)


FIXED_SOURCE_PATHS = {
    "ODT_VLA_AUDIT_2026-09-02.md",
    "pyproject.toml",
    "athena/CANONICAL_ODT_PHASE1_PROTOCOL.md",
    "athena/run_svhn_canonical_odt.py",
    "athena/summarize_svhn_canonical_odt.py",
    "athena/validate_svhn_canonical_odt_smoke.py",
    "athena/verify_canonical_odt_freeze.py",
    "athena/launch_svhn_canonical_odt.py",
    "athena/slurm_canonical_odt_tests.sbatch",
    "athena/slurm_svhn_canonical_odt_prefetch.sbatch",
    "athena/slurm_svhn_canonical_odt_smoke.sbatch",
    "athena/slurm_svhn_canonical_odt_smoke_validate.sbatch",
    "athena/slurm_svhn_canonical_odt.sbatch",
    "athena/slurm_svhn_canonical_odt_summary.sbatch",
    "tests/test_canonical_odt.py",
    "tests/test_canonical_odt_pipeline.py",
    "tests/test_odt.py",
    "tests/test_odt_interp.py",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require_physical_root(root: Path) -> Path:
    """Return the physical run root, rejecting an alias through a symlink."""

    root = Path(root)
    if root.is_symlink():
        raise RuntimeError(f"run root must not be a symlink: {root}")
    lexical = Path(os.path.abspath(root))
    try:
        resolved = root.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"run root is missing: {root}") from error
    if not resolved.is_dir():
        raise RuntimeError(f"run root is not a directory: {resolved}")
    if lexical != resolved:
        raise RuntimeError(f"run root must not traverse a symlink: {root}")
    return resolved


def require_confined_path(
    root: Path,
    path: Path,
    *,
    label: str,
    kind: str | None = None,
    allow_missing_leaf: bool = False,
) -> Path:
    """Reject symlink traversal and require a path physically below ``root``."""

    physical_root = require_physical_root(root)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = physical_root / candidate
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(physical_root)
    except ValueError as error:
        raise RuntimeError(f"{label} is outside this run root: {path}") from error

    cursor = physical_root
    for index, part in enumerate(relative.parts):
        cursor = cursor / part
        is_leaf = index == len(relative.parts) - 1
        if cursor.is_symlink():
            raise RuntimeError(f"{label} must not traverse a symlink: {cursor}")
        if not cursor.exists() and not (allow_missing_leaf and is_leaf):
            raise RuntimeError(f"{label} is missing: {cursor}")

    try:
        resolved = lexical.resolve(strict=not allow_missing_leaf)
    except FileNotFoundError as error:
        raise RuntimeError(f"{label} is missing: {lexical}") from error
    if resolved != physical_root and physical_root not in resolved.parents:
        raise RuntimeError(f"{label} resolves outside this run root: {path}")
    if kind == "file" and (not lexical.exists() or not lexical.is_file()):
        raise RuntimeError(f"{label} is not a regular file: {lexical}")
    if kind == "directory" and (not lexical.exists() or not lexical.is_dir()):
        raise RuntimeError(f"{label} is not a directory: {lexical}")
    return lexical


def reject_symlinks_in_directory(root: Path, directory: Path, *, label: str) -> Path:
    """Require a confined directory and reject every symlink below it."""

    directory = require_confined_path(root, directory, label=label, kind="directory")
    for descendant in directory.rglob("*"):
        if descendant.is_symlink():
            raise RuntimeError(f"{label} contains a symlink: {descendant}")
        require_confined_path(root, descendant, label=label)
    return directory


def parse_manifest(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip(" *")
        if relative in entries:
            raise RuntimeError(f"duplicate manifest path: {relative}")
        entries[relative] = expected
    if not entries:
        raise RuntimeError("source manifest is empty")
    return entries


def verify(root: Path, manifest: Path) -> dict[str, str]:
    root = require_physical_root(root)
    expected_manifest = root / "source_manifest.sha256"
    manifest = require_confined_path(
        root, manifest, label="source manifest", kind="file"
    )
    if manifest != expected_manifest:
        raise RuntimeError(f"source manifest must be exactly {expected_manifest}")
    for directory_name in ("logs", "results"):
        directory = root / directory_name
        if directory.exists() or directory.is_symlink():
            reject_symlinks_in_directory(
                root, directory, label=f"{directory_name} directory"
            )
    entries = parse_manifest(manifest)
    xvla_root = require_confined_path(
        root, root / "xvla", label="xvla source directory", kind="directory"
    )
    expected_paths = FIXED_SOURCE_PATHS | {
        str(path.relative_to(root)) for path in xvla_root.rglob("*.py")
    }
    if set(entries) != expected_paths:
        missing = sorted(expected_paths - set(entries))
        unexpected = sorted(set(entries) - expected_paths)
        raise RuntimeError(
            f"manifest path set mismatch, missing={missing}, unexpected={unexpected}"
        )
    for relative, expected in entries.items():
        source = require_confined_path(
            root, root / relative, label=f"manifest source {relative}", kind="file"
        )
        actual = digest(source)
        if actual != expected:
            raise RuntimeError(f"source hash mismatch for {relative}: {actual} != {expected}")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--test-log", type=Path)
    parser.add_argument("--certificate", type=Path)
    args = parser.parse_args()

    root = require_physical_root(args.root)
    entries = verify(root, args.manifest)
    result = {
        "schema": UNIT_CERTIFICATE_SCHEMA,
        "source_manifest_sha256": digest(args.manifest),
        "source_entries": entries,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if args.test_log is not None:
        if args.certificate is None:
            parser.error("--certificate is required with --test-log")
        results = reject_symlinks_in_directory(
            root, root / "results", label="results directory"
        )
        test_log = require_confined_path(
            root, args.test_log, label="unit-test log", kind="file"
        )
        if test_log.parent != results or not test_log.name.startswith("canonical_odt_unit_"):
            raise RuntimeError("unit-test log must be a canonical log directly inside results")
        certificate = require_confined_path(
            root,
            args.certificate,
            label="unit-test certificate",
            allow_missing_leaf=True,
        )
        expected_certificate = results / "canonical_odt_unit_certificate.json"
        if certificate != expected_certificate:
            raise RuntimeError(f"unit-test certificate must be exactly {expected_certificate}")
        text = test_log.read_text()
        if " passed" not in text or " failed" in text:
            raise RuntimeError("test log does not certify a passing pytest run")
        result.update(
            {
                "tests_passed": True,
                "test_log_path": str(test_log),
                "test_log_sha256": digest(test_log),
            }
        )
        certificate.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
