"""Freeze and submit the fully dependent Athena Phase 1 pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from athena.verify_canonical_odt_freeze import (
        FIXED_SOURCE_PATHS,
        reject_symlinks_in_directory,
        require_confined_path,
        require_physical_root,
        verify,
    )
except ModuleNotFoundError:
    from verify_canonical_odt_freeze import (
        FIXED_SOURCE_PATHS,
        reject_symlinks_in_directory,
        require_confined_path,
        require_physical_root,
        verify,
    )


LAUNCH_SCHEMA = "xvla-canonical-odt-athena-launch-v4"
DAG_SCHEMA = "xvla-canonical-odt-athena-dag-v4"
DAG_NODE_ORDER = (
    "tests", "prefetch", "smoke", "smoke_validate", "train_array", "summary",
)
DAG_EDGES = (
    ("tests", "prefetch", "afterany"),
    ("prefetch", "smoke", "afterany"),
    ("smoke", "smoke_validate", "afterany"),
    ("smoke_validate", "train_array", "afterok"),
    ("smoke_validate", "summary", "afternotok"),
    ("train_array", "summary", "afterany"),
)
ATHENA_PYTHON = "/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def verify_import_closure(root: Path) -> None:
    """Fail before freezing unless every production entrypoint imports."""

    if os.path.abspath(sys.executable) != ATHENA_PYTHON:
        raise RuntimeError(
            f"launcher must use the frozen Athena Python {ATHENA_PYTHON}, got {sys.executable}"
        )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.check_call(
        [
            ATHENA_PYTHON,
            "-B",
            "-c",
            (
                "import xvla; "
                "import athena.run_svhn_canonical_odt; "
                "import athena.validate_svhn_canonical_odt_smoke; "
                "import athena.summarize_svhn_canonical_odt; "
                "import athena.verify_canonical_odt_freeze"
            ),
        ],
        cwd=root,
        env=environment,
    )


def submit(
    root: Path,
    name: str,
    script: str,
    arguments: list[str],
    dependency=None,
    dependency_mode="afterany",
    dependency_expression=None,
    array=None,
    *,
    hold: bool = False,
):
    command = [
        "sbatch",
        "--parsable",
        "--kill-on-invalid-dep=yes",
        f"--job-name={name}",
        f"--output={root}/logs/{name}-%A_%a.out" if array else f"--output={root}/logs/{name}-%j.out",
        f"--error={root}/logs/{name}-%A_%a.err" if array else f"--error={root}/logs/{name}-%j.err",
    ]
    if dependency is not None and dependency_expression is not None:
        raise ValueError("dependency and dependency expression are mutually exclusive")
    if dependency_expression is not None:
        if not isinstance(dependency_expression, str) or "?" not in dependency_expression:
            raise ValueError("Slurm dependency expression differs")
        command.append(f"--dependency={dependency_expression}")
    elif dependency is not None:
        if dependency_mode not in {"afterany", "afterok"}:
            raise ValueError("Slurm dependency mode differs")
        command.append(f"--dependency={dependency_mode}:{dependency}")
    if array is not None:
        command.append(f"--array={array}")
    if hold:
        command.append("--hold")
    command.extend([str(root / script), *arguments])
    response = subprocess.check_output(command, text=True).strip()
    job_id = response.split(";", maxsplit=1)[0]
    if not job_id.isdecimal():
        raise RuntimeError(f"sbatch returned an invalid Slurm job id: {response!r}")
    return job_id


def make_launch_record(
    *,
    root: Path,
    data_root: Path,
    manifest_sha256: str,
    job_ids: dict[str, str],
) -> dict:
    """Build the exact scheduler DAG consumed by the final aggregator."""

    if tuple(job_ids) != DAG_NODE_ORDER:
        raise RuntimeError(f"job inventory or order differs from {DAG_NODE_ORDER}")
    if any(not isinstance(job_id, str) or not job_id.isdecimal() for job_id in job_ids.values()):
        raise RuntimeError("every submitted Slurm job id must be a decimal string")
    specifications = {
        "tests": {
            "script": "athena/slurm_canonical_odt_tests.sbatch",
            "arguments": [str(root)],
            "array": None,
        },
        "prefetch": {
            "script": "athena/slurm_svhn_canonical_odt_prefetch.sbatch",
            "arguments": [str(root), str(data_root)],
            "array": None,
        },
        "smoke": {
            "script": "athena/slurm_svhn_canonical_odt_smoke.sbatch",
            "arguments": [str(root), str(data_root)],
            "array": None,
        },
        "smoke_validate": {
            "script": "athena/slurm_svhn_canonical_odt_smoke_validate.sbatch",
            "arguments": [str(root)],
            "array": None,
        },
        "train_array": {
            "script": "athena/slurm_svhn_canonical_odt.sbatch",
            "arguments": [str(root), str(data_root)],
            "array": "0-2",
        },
        "summary": {
            "script": "athena/slurm_svhn_canonical_odt_summary.sbatch",
            "arguments": [str(root)],
            "array": None,
        },
    }
    nodes = {
        name: {"job_id": job_ids[name], **specifications[name]}
        for name in DAG_NODE_ORDER
    }
    terminal_expression = (
        f"afternotok:{job_ids['smoke_validate']}?afterany:{job_ids['train_array']}"
    )
    return {
        "schema": LAUNCH_SCHEMA,
        "source_manifest_sha256": manifest_sha256,
        "run_root": str(root),
        "data_root": str(data_root),
        "submission_protocol": {
            "first_node_submitted_held": True,
            "launch_record_written_before_release": True,
        },
        "dag": {
            "schema": DAG_SCHEMA,
            "node_order": list(DAG_NODE_ORDER),
            "nodes": nodes,
            "edges": [
                {
                    "upstream": upstream,
                    "downstream": downstream,
                    "slurm_dependency": dependency_mode,
                }
                for upstream, downstream, dependency_mode in DAG_EDGES
            ],
            "terminal_node": "summary",
            "terminal_dependency_expression": terminal_expression,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    root = require_physical_root(args.run_root)
    failure_path = root.parent / f"{root.name}.launch_failure.json"
    if failure_path.exists() or failure_path.is_symlink():
        raise RuntimeError(
            f"run root has a prior launch failure receipt and cannot be reused: {failure_path}"
        )
    if args.data_root.is_symlink():
        raise RuntimeError("data root must not be a symlink")
    data_root = args.data_root.resolve(strict=True)
    if Path(os.path.abspath(args.data_root)) != data_root or not data_root.is_dir():
        raise RuntimeError("data root must be an existing physical directory")
    if data_root == root or root in data_root.parents or data_root in root.parents:
        raise RuntimeError("data root must be outside the immutable run root")
    manifest = root / "source_manifest.sha256"
    if manifest.exists() or manifest.is_symlink():
        raise RuntimeError("run root has already been frozen")
    for directory_name in ("logs", "results"):
        directory = root / directory_name
        if directory.is_symlink():
            raise RuntimeError(f"run root contains symlinked {directory_name} directory")
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f"run root contains stale {directory_name} artifacts")
        directory.mkdir(exist_ok=True)
        reject_symlinks_in_directory(root, directory, label=f"{directory_name} directory")

    xvla_root = require_confined_path(
        root, root / "xvla", label="xvla source directory", kind="directory"
    )
    expected = FIXED_SOURCE_PATHS | {
        str(path.relative_to(root)) for path in xvla_root.rglob("*.py")
    }
    for relative in sorted(expected):
        require_confined_path(
            root, root / relative, label=f"source file {relative}", kind="file"
        )
    submitted: list[str] = []
    launch_path = root / "results" / "launch.json"
    try:
        verify_import_closure(root)
        for directory_name in ("logs", "results"):
            directory = root / directory_name
            if any(directory.iterdir()):
                raise RuntimeError(
                    f"import closure contaminated the empty {directory_name} directory"
                )
        manifest.write_text(
            "".join(
                f"{digest(root / relative)}  {relative}\n"
                for relative in sorted(expected)
            )
        )
        verify(root, manifest)
        os.chmod(manifest, 0o444)
        test_job = submit(
            root,
            "xvla-odt-p1-tests",
            "athena/slurm_canonical_odt_tests.sbatch",
            [str(root)],
            hold=True,
        )
        submitted.append(test_job)
        prefetch_job = submit(
            root,
            "xvla-odt-p1-prefetch",
            "athena/slurm_svhn_canonical_odt_prefetch.sbatch",
            [str(root), str(data_root)],
            dependency=test_job,
        )
        submitted.append(prefetch_job)
        smoke_job = submit(
            root,
            "xvla-odt-p1-smoke",
            "athena/slurm_svhn_canonical_odt_smoke.sbatch",
            [str(root), str(data_root)],
            dependency=prefetch_job,
        )
        submitted.append(smoke_job)
        smoke_validate_job = submit(
            root,
            "xvla-odt-p1-smoke-validate",
            "athena/slurm_svhn_canonical_odt_smoke_validate.sbatch",
            [str(root)],
            dependency=smoke_job,
        )
        submitted.append(smoke_validate_job)
        train_job = submit(
            root,
            "xvla-odt-p1-svhn3",
            "athena/slurm_svhn_canonical_odt.sbatch",
            [str(root), str(data_root)],
            dependency=smoke_validate_job,
            dependency_mode="afterok",
            array="0-2",
        )
        submitted.append(train_job)
        summary_job = submit(
            root,
            "xvla-odt-p1-summary",
            "athena/slurm_svhn_canonical_odt_summary.sbatch",
            [str(root)],
            dependency_expression=(
                f"afternotok:{smoke_validate_job}?afterany:{train_job}"
            ),
        )
        submitted.append(summary_job)
        launch = make_launch_record(
            root=root,
            data_root=data_root,
            manifest_sha256=digest(manifest),
            job_ids={
                "tests": test_job,
                "prefetch": prefetch_job,
                "smoke": smoke_job,
                "smoke_validate": smoke_validate_job,
                "train_array": train_job,
                "summary": summary_job,
            },
        )
        require_confined_path(
            root, launch_path, label="launch record", allow_missing_leaf=True
        )
        atomic_json(launch_path, launch)
        os.chmod(launch_path, 0o444)
        subprocess.check_call(["scontrol", "release", test_job])
    except BaseException as error:
        if submitted:
            subprocess.run(["scancel", *submitted], check=False)
        if launch_path.is_file():
            launch_path.unlink()
        if not submitted and manifest.is_file() and not manifest.is_symlink():
            manifest.unlink()
        if not failure_path.exists() and not failure_path.is_symlink():
            atomic_json(
                failure_path,
                {
                    "run_root": str(root),
                    "submitted_jobs": submitted,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise
    print(json.dumps(launch, indent=2))


if __name__ == "__main__":
    main()
