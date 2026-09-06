"""Freeze and submit the staged Athena Dooms SVHN reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from athena.dooms_svhn_reproduction import (
    DAG_ORDER,
    DAG_SCHEMA,
    SCHEMA,
    SOURCE_PATHS,
    atomic_json,
    physical_file,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink")
    lexical = Path(os.path.abspath(path))
    resolved = path.resolve(strict=True)
    if lexical != resolved or not resolved.is_dir():
        raise RuntimeError(f"{label} must be an existing physical directory")
    return resolved


def submit(
    root: Path, name: str, script: str, arguments: list[str], dependency=None,
    array=None, hold: bool = False,
) -> str:
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={name}",
        f"--output={root}/logs/{name}-%A_%a.out" if array else f"--output={root}/logs/{name}-%j.out",
        f"--error={root}/logs/{name}-%A_%a.err" if array else f"--error={root}/logs/{name}-%j.err",
    ]
    if dependency:
        command.append(f"--dependency={dependency}")
    if array:
        command.append(f"--array={array}")
    if hold:
        command.append("--hold")
    command.extend([str(root / script), *arguments])
    job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
    if not job_id.isdecimal():
        raise RuntimeError(f"sbatch returned malformed job id {job_id!r}")
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    root = physical_directory(args.run_root, "run root")
    data_root = physical_directory(args.data_root, "data root")
    if root == data_root or root in data_root.parents or data_root in root.parents:
        raise RuntimeError("dataset root must be outside the frozen source root")
    manifest = root / "source_manifest.sha256"
    if manifest.exists() or manifest.is_symlink():
        raise RuntimeError("run root is already frozen")
    for name in ("logs", "results"):
        directory = root / name
        if directory.is_symlink() or (directory.exists() and any(directory.iterdir())):
            raise RuntimeError(f"{name} directory is symlinked or nonempty")
        directory.mkdir(exist_ok=True)
    jobs = {}
    frozen_paths = []
    try:
        entries = {}
        for relative in SOURCE_PATHS:
            path = root / relative
            physical_file(path, root, f"required source {relative}")
            entries[relative] = digest(path)
        manifest.write_text("".join(f"{entries[key]}  {key}\n" for key in sorted(entries)))
        for relative in SOURCE_PATHS:
            path = root / relative
            os.chmod(path, 0o444)
            frozen_paths.append(path)
        os.chmod(manifest, 0o444)
        jobs["unit"] = submit(
            root, "dooms-unit", "athena/slurm_dooms_unit.sbatch",
            [str(root), str(data_root)], hold=True,
        )
        jobs["prefetch"] = submit(
            root, "dooms-prefetch", "athena/slurm_dooms_prefetch.sbatch",
            [str(root), str(data_root)], dependency=f"afterany:{jobs['unit']}",
        )
        jobs["feasibility"] = submit(
            root, "dooms-feasibility", "athena/slurm_dooms_feasibility.sbatch",
            [str(root), str(data_root)], dependency=f"afterany:{jobs['prefetch']}",
        )
        jobs["train"] = submit(
            root, "dooms-train", "athena/slurm_dooms_train.sbatch",
            [str(root), str(data_root)], dependency=f"afterany:{jobs['feasibility']}", array="0-4",
        )
        jobs["calibrate"] = submit(
            root, "dooms-calibrate", "athena/slurm_dooms_calibrate.sbatch",
            [str(root), str(data_root)], dependency=f"afterany:{jobs['train']}", array="0-4",
        )
        jobs["odt"] = submit(
            root, "dooms-odt", "athena/slurm_dooms_odt.sbatch",
            [str(root), str(data_root)], dependency=f"afterany:{jobs['calibrate']}", array="0-4",
        )
        jobs["eval"] = submit(
            root, "dooms-eval", "athena/slurm_dooms_eval.sbatch",
            [str(root), str(data_root)], dependency=f"afterany:{jobs['odt']}", array="0-4",
        )
        jobs["verify_eval"] = submit(
            root, "dooms-verify-eval", "athena/slurm_dooms_verify_eval.sbatch",
            [str(root), str(data_root)], dependency=f"afterany:{jobs['eval']}", array="0-4",
        )
        jobs["summary"] = submit(
            root, "dooms-summary", "athena/slurm_dooms_summary.sbatch",
            [str(root), str(data_root)], dependency=f"afterany:{jobs['verify_eval']}",
        )
        if tuple(jobs) != DAG_ORDER or len(set(jobs.values())) != len(jobs):
            raise RuntimeError("launch DAG job order or uniqueness changed")
        launch = {
        "schema": SCHEMA,
        "dag_schema": DAG_SCHEMA,
        "run_root": str(root),
        "data_root": str(data_root),
        "source_manifest_sha256": digest(manifest),
        "node_order": list(DAG_ORDER),
        "jobs": jobs,
        "dependencies": {
            "prefetch": f"afterany:{jobs['unit']}",
            "feasibility": f"afterany:{jobs['prefetch']}",
            "train": f"afterany:{jobs['feasibility']}",
            "calibrate": f"afterany:{jobs['train']}",
            "odt": f"afterany:{jobs['calibrate']}",
            "eval": f"afterany:{jobs['odt']}",
            "verify_eval": f"afterany:{jobs['eval']}",
            "summary": f"afterany:{jobs['verify_eval']}",
        },
        "array": {
            "train": "0-4", "calibrate": "0-4", "odt": "0-4", "eval": "0-4",
            "verify_eval": "0-4",
        },
        "unit_submitted_held": True,
        }
        atomic_json(root / "results" / "launch.json", launch)
        subprocess.run(["scontrol", "release", jobs["unit"]], check=True)
    except Exception as error:
        if jobs:
            subprocess.run(["scancel", *jobs.values()], check=False)
        else:
            for path in frozen_paths:
                try:
                    os.chmod(path, 0o644)
                except OSError:
                    pass
            try:
                if manifest.is_file() and not manifest.is_symlink():
                    try:
                        os.chmod(manifest, 0o644)
                    except OSError:
                        pass
                    manifest.unlink()
            except OSError:
                pass
        receipt = root.parent / f"{root.name}.launch_failure.json"
        if not receipt.exists():
            atomic_json(
                receipt,
                {"run_root": str(root), "submitted_jobs": jobs, "error": repr(error)},
            )
        raise
    print(json.dumps(launch, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
