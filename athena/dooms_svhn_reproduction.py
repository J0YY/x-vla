"""Staged Athena reproduction of the reported three-layer SVHN chi-net.

This lane is deliberately separate from the width-32 CP-tree extension.  It
implements the architecture and hyperparameters stated in Dooms et al., while
serializing every implementation choice that the paper leaves unspecified.
No model-bearing stage before ``eval`` consumes official SVHN test examples or
labels. The model-free prefetch stage authenticates the downloaded test split.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv

from xvla.models.dooms_xnet import (
    PAPER_REPORTED_ARCHITECTURE_SCHEMA,
    DirectBilinearLayer,
    DoomsReportedChiNet,
    DoomsReportedChiNetConfig,
)
from xvla.nn.normalization import RmsBatchNorm
from xvla.train.calibrate import calibrate_rbn_sequential
from xvla.train.canonical_odt import (
    HomogeneousChiTN,
    canonical_environments,
    canonicalize_homogeneous,
    coefficient_error_squared,
    coefficient_inner_product,
    compress_homogeneous,
    discarded_trace,
    eigengap_diagnostics,
    export_homogeneous_network,
    hierarchical_tail_bound_squared,
    homogeneous_forward,
    local_environments,
    sorted_eigensystem,
    top_bases,
)


SCHEMA = "xvla-dooms-chi-net-svhn-reported-spec-reproduction-v1"
PAPER_ARXIV_ID = "2504.02667v1"
PAPER_PDF_SHA256 = "a97f59a91d90ffde56bc22bf0b3c79e0891ff88197ca9d7670331e4208277559"
SEEDS = (0, 1, 2, 3, 4)
TRAIN_BATCH_SIZE = 2048
CALIBRATION_BATCH_SIZE = 2048
CALIBRATION_BATCHES = 64
EPSILON_GRID = (0.0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
REMOVAL_GRID = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99)
EXPECTED_PARAMETER_COUNT = 659_722
EXPECTED_SPLIT_LENGTHS = {"train": 73_257, "extra": 531_131, "test": 26_032}
SVHN_FILES = {
    "train_32x32.mat": {
        "bytes": 182040794,
        "md5": "e26dedcc434d2e4c54c9b2d4a06d8373",
        "sha256": "435e94d69a87fde4fd4d7f3dd208dfc32cb6ae8af2240d066de1df7508d083b8",
    },
    "extra_32x32.mat": {
        "bytes": 1329278602,
        "md5": "a93ce644f1a588dc4d68dda5feec44a7",
        "sha256": "a133a4beb38a00fcdda90c9489e0c04f900b660ce8a316a5e854838379a71eb3",
    },
    "test_32x32.mat": {
        "bytes": 64275384,
        "md5": "eb5a983be6a315427106f1b164d9cef3",
        "sha256": "cdce80dfb2a2c4c6160906d0bd7c68ec5a99d7ca4831afa54f09182025b6a75b",
    },
}
ASSUMPTIONS = {
    "grayscale": "float RGB in [0,1], coefficients [0.2989,0.5870,0.1140]",
    "input_noise_norm_0p3": "interpreted as per-pixel Gaussian standard deviation 0.3",
    "noise_scope": "training only, after grayscale, no clipping",
    "noise_rng": "dedicated CUDA torch.Generator stream with fixed seed and fixed loader order",
    "bilinear_initialization": "A and B iid normal std=width^-0.5, zero biases",
    "embedding_head_initialization": "PyTorch nn.Linear default initialization",
    "biases": "embedding, every A/B projection, and head all include bias",
    "normalization_placement": "one scalar RmsBatchNorm immediately before each bilinear layer",
    "normalization_definition": (
        "training divides by current scalar batch RMS over batch and features; EMA momentum 0.99; "
        "epsilon 1e-6 is added after RMS; no affine parameters"
    ),
    "calibration": "64 equal train-only batches, no noise, sequential execution-order calibration",
    "adamw": "all parameters, betas [0.9,0.999], epsilon 1e-8, fused false",
    "cosine": "CosineAnnealingLR per optimizer step, no warmup, eta_min 0",
    "remainder": "drop_last false, so the final 228-example batch is used",
    "precision": "float32 training without autocast, float64 ODT",
    "softmax": "cross entropy consumes logits; ODT analyzes the pre-softmax logit coefficient tensor",
}
_FIXED_SOURCE_PATHS = (
    "athena/DOOMS_SVHN_REPRODUCTION_PROTOCOL.md",
    "athena/dooms_svhn_reproduction.py",
    "athena/launch_dooms_svhn_reproduction.py",
    "athena/slurm_dooms_calibrate.sbatch",
    "athena/slurm_dooms_eval.sbatch",
    "athena/slurm_dooms_feasibility.sbatch",
    "athena/slurm_dooms_odt.sbatch",
    "athena/slurm_dooms_prefetch.sbatch",
    "athena/slurm_dooms_summary.sbatch",
    "athena/slurm_dooms_train.sbatch",
    "athena/slurm_dooms_unit.sbatch",
    "athena/slurm_dooms_verify_eval.sbatch",
    "pyproject.toml",
    "tests/test_canonical_odt.py",
    "tests/test_dooms_xnet.py",
    "tests/test_dooms_svhn_reproduction.py",
    "tmp/pdfs/dooms-xnets-2504.02667.clean.txt",
    "tmp/pdfs/dooms-xnets-2504.02667.pdf",
)
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = _FIXED_SOURCE_PATHS + tuple(
    sorted(
        str(path.relative_to(_SOURCE_ROOT))
        for path in (_SOURCE_ROOT / "xvla").glob("**/*.py")
    )
)
MATRIX_LAYOUT = {
    "canonical_embedding": ((257, 1025), np.dtype("float64")),
    "canonical_core_0": ((257, 257, 257), np.dtype("float64")),
    "canonical_core_1": ((257, 257, 257), np.dtype("float64")),
    "canonical_core_2": ((257, 257, 257), np.dtype("float64")),
    "canonical_head": ((10, 257), np.dtype("float64")),
    "global_grams": ((4, 257, 257), np.dtype("float64")),
    "local_grams": ((4, 257, 257), np.dtype("float64")),
    "global_eigenvalues": ((4, 257), np.dtype("float64")),
    "global_eigenvectors": ((4, 257, 257), np.dtype("float64")),
}
REPLAY_ATOL = 1e-9
REPLAY_RTOL = 1e-10
DAG_SCHEMA = "xvla-dooms-svhn-athena-terminal-dag-v3"
DAG_ORDER = (
    "unit", "prefetch", "feasibility", "train", "calibrate", "odt", "eval",
    "verify_eval", "summary",
)
ARRAY_STAGES = ("train", "calibrate", "odt", "eval", "verify_eval")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite result {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"temporary result already exists {temporary}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite result {path}")
    temporary = path.with_name(path.name + ".tmp.npz")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"temporary result already exists {temporary}")
    np.savez(temporary, **arrays)
    with np.load(temporary, allow_pickle=False) as bundle:
        if set(bundle.files) != set(arrays):
            raise RuntimeError("NPZ key inventory changed on round trip")
        for key, expected in arrays.items():
            actual = bundle[key]
            if actual.dtype != expected.dtype or actual.shape != expected.shape:
                raise RuntimeError(f"NPZ metadata changed for {key}")
            if not np.array_equal(actual, expected):
                raise RuntimeError(f"NPZ values changed for {key}")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite result {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"temporary result already exists {temporary}")
    torch.save(value, temporary)
    if not temporary.is_file() or temporary.is_symlink() or temporary.stat().st_size <= 0:
        raise RuntimeError("checkpoint temporary file was not written safely")
    os.replace(temporary, path)


def strict_json(path: Path) -> dict:
    def reject_constant(value: str):
        raise ValueError(f"nonstandard JSON numeric constant {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(), parse_constant=reject_constant, object_pairs_hook=unique_object
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def finite_real(value, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise RuntimeError(f"{label} must be a finite real number")
    return float(value)


def replay_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    if reference.shape != candidate.shape or not bool(torch.isfinite(reference).all()) or not bool(
        torch.isfinite(candidate).all()
    ):
        raise RuntimeError("replay tensors differ in shape or contain nonfinite values")
    maximum_absolute = float((reference - candidate).abs().max())
    output_scale = max(1.0, float(reference.abs().max()))
    tolerance = REPLAY_ATOL + REPLAY_RTOL * output_scale
    return {
        "maximum_absolute": maximum_absolute,
        "reference_maximum_absolute": output_scale,
        "absolute_tolerance": REPLAY_ATOL,
        "relative_tolerance": REPLAY_RTOL,
        "combined_tolerance": tolerance,
        "passed": maximum_absolute <= tolerance,
    }


def validate_replay_inventory(record: dict, expected_names: set[str], label: str) -> None:
    if not isinstance(record, dict) or set(record) != expected_names:
        raise RuntimeError(f"{label} replay inventory mismatch")
    expected_metric_keys = {
        "maximum_absolute", "reference_maximum_absolute", "absolute_tolerance",
        "relative_tolerance", "combined_tolerance", "passed",
    }
    for name, metrics in record.items():
        if not isinstance(metrics, dict) or set(metrics) != expected_metric_keys:
            raise RuntimeError(f"{label} replay metric inventory mismatch for {name}")
        if metrics["passed"] is not True:
            raise RuntimeError(f"{label} replay {name} did not pass")
        maximum = finite_real(metrics["maximum_absolute"], f"{label} {name} maximum")
        combined = finite_real(metrics["combined_tolerance"], f"{label} {name} tolerance")
        if maximum < 0.0 or combined < 0.0 or maximum > combined:
            raise RuntimeError(f"{label} replay {name} exceeds its tolerance")
        require_close(metrics["absolute_tolerance"], REPLAY_ATOL, f"{label} absolute tolerance", relative=0.0)
        require_close(metrics["relative_tolerance"], REPLAY_RTOL, f"{label} relative tolerance", relative=0.0)
        expected_combined = REPLAY_ATOL + REPLAY_RTOL * finite_real(
            metrics["reference_maximum_absolute"], f"{label} reference scale"
        )
        require_close(combined, expected_combined, f"{label} combined tolerance")


def physical_file(path: Path, root: Path, label: str) -> Path:
    root_absolute = Path(os.path.abspath(root))
    if root_absolute.is_symlink() or root_absolute.resolve(strict=True) != root_absolute:
        raise RuntimeError(f"{label} root must be a physical directory")
    path_absolute = Path(os.path.abspath(path))
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes its root") from error
    current = root_absolute
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"{label} contains a symlink component")
    if not path_absolute.is_file() or path_absolute.resolve(strict=True) != path_absolute:
        raise RuntimeError(f"{label} is not a physical file")
    return path_absolute


def coefficient_cancellation_audit(
    reference: HomogeneousChiTN, candidate: HomogeneousChiTN
) -> dict[str, float | bool]:
    """Record cancellation consistency without claiming a stable error norm.

    The exact block-difference contraction is prohibitive at homogeneous width
    257.  This audit evaluates all three mixed inner products and declares the
    cancellation allowance explicitly.  It complements, rather than replaces,
    the RQ residual and independent logit replay gates.
    """

    reference_squared = float(coefficient_inner_product(reference, reference))
    candidate_squared = float(coefficient_inner_product(candidate, candidate))
    cross = float(coefficient_inner_product(reference, candidate))
    raw_error = reference_squared + candidate_squared - 2.0 * cross
    scale = abs(reference_squared) + abs(candidate_squared) + 2.0 * abs(cross)
    dimension = max(reference.bond_dims + candidate.bond_dims)
    allowance = (
        10_000.0
        * torch.finfo(reference.embedding.dtype).eps
        * dimension
        * (reference.n_layers + 1)
        * scale
    )
    if not all(
        math.isfinite(value)
        for value in (reference_squared, candidate_squared, cross, raw_error, allowance)
    ):
        raise RuntimeError("coefficient identity audit produced a nonfinite value")
    if reference_squared <= 0.0 or candidate_squared <= 0.0:
        raise RuntimeError("coefficient norm is nonpositive")
    conservative_error_squared = max(raw_error, 0.0) + allowance
    return {
        "reference_norm_squared": reference_squared,
        "candidate_norm_squared": candidate_squared,
        "cross_inner_product": cross,
        "raw_subtractive_error_squared": raw_error,
        "declared_cancellation_allowance": allowance,
        "cancellation_compatible": abs(raw_error) <= allowance,
        "heuristic_relative_error_indicator": math.sqrt(
            conservative_error_squared / reference_squared
        ),
        "interpretation": (
            "Numerical cancellation consistency only. The local RQ identities, square-basis "
            "orthogonality, and independent logit replay are the load-bearing algebraic gates."
        ),
        "relative_norm_squared_disagreement": abs(
            candidate_squared - reference_squared
        )
        / reference_squared,
    }


def validate_cancellation_diagnostic(record: dict, label: str) -> None:
    expected_keys = {
        "reference_norm_squared", "candidate_norm_squared", "cross_inner_product",
        "raw_subtractive_error_squared", "declared_cancellation_allowance",
        "cancellation_compatible", "heuristic_relative_error_indicator",
        "interpretation", "relative_norm_squared_disagreement",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise RuntimeError(f"{label} cancellation diagnostic inventory mismatch")
    numeric_keys = expected_keys - {"cancellation_compatible", "interpretation"}
    numbers = {key: finite_real(record[key], f"{label} {key}") for key in numeric_keys}
    if numbers["reference_norm_squared"] <= 0.0 or numbers["candidate_norm_squared"] <= 0.0:
        raise RuntimeError(f"{label} cancellation diagnostic has nonpositive norms")
    if (
        numbers["declared_cancellation_allowance"] < 0.0
        or numbers["heuristic_relative_error_indicator"] < 0.0
        or numbers["relative_norm_squared_disagreement"] < 0.0
    ):
        raise RuntimeError(f"{label} cancellation diagnostic has a negative magnitude")
    expected_interpretation = (
        "Numerical cancellation consistency only. The local RQ identities, square-basis "
        "orthogonality, and independent logit replay are the load-bearing algebraic gates."
    )
    if record["interpretation"] != expected_interpretation:
        raise RuntimeError(f"{label} cancellation interpretation mismatch")
    if record["cancellation_compatible"] is not (
        abs(numbers["raw_subtractive_error_squared"])
        <= numbers["declared_cancellation_allowance"]
    ):
        raise RuntimeError(f"{label} cancellation Boolean is inconsistent")
    expected_indicator = math.sqrt(
        (max(numbers["raw_subtractive_error_squared"], 0.0)
         + numbers["declared_cancellation_allowance"])
        / numbers["reference_norm_squared"]
    )
    require_close(
        numbers["heuristic_relative_error_indicator"], expected_indicator,
        f"{label} heuristic cancellation indicator",
    )


def artifact(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def capture_artifact(path: Path, run_root: Path, label: str) -> dict:
    """Capture the exact physical bytes a stage is about to consume."""

    physical_file(path, run_root, label)
    return artifact(path)


def require_artifact_unchanged(path: Path, captured: dict, run_root: Path, label: str) -> None:
    """Close the consumer-side TOCTOU window before committing a result."""

    physical_file(path, run_root, label)
    if artifact(path) != captured:
        raise RuntimeError(f"{label} changed while the stage was running")


def validate_source_manifest(run_root: Path) -> dict[str, str]:
    manifest = run_root / "source_manifest.sha256"
    physical_file(manifest, run_root, "frozen source manifest")
    entries: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip(" *")
        if relative in entries:
            raise RuntimeError(f"duplicate source manifest entry {relative}")
        entries[relative] = expected
    expected_paths = set(SOURCE_PATHS)
    if set(entries) != expected_paths:
        raise RuntimeError(
            f"source manifest inventory mismatch: expected={sorted(expected_paths)}, "
            f"actual={sorted(entries)}"
        )
    for relative, expected in entries.items():
        path = run_root / relative
        physical_file(path, run_root, f"source file {relative}")
        if sha256(path) != expected:
            raise RuntimeError(f"source manifest verification failed for {relative}")
    if sha256(run_root / "tmp/pdfs/dooms-xnets-2504.02667.pdf") != PAPER_PDF_SHA256:
        raise RuntimeError("frozen Dooms paper PDF hash mismatch")
    return entries


def validate_unit_certificate(run_root: Path) -> dict:
    path = run_root / "results" / "unit_certificate.json"
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("source-bound unit certificate is missing or symlinked")
    physical_file(path, run_root, "unit certificate")
    record = strict_json(path)
    expected_keys = {
        "schema",
        "stage",
        "tests_passed",
        "source_manifest_sha256",
        "test_log",
        "runtime",
    }
    if set(record) != expected_keys or record.get("schema") != SCHEMA:
        raise RuntimeError("unit certificate schema or inventory mismatch")
    if record.get("stage") != "unit" or record.get("tests_passed") is not True:
        raise RuntimeError("unit certificate does not attest a passing run")
    if record.get("source_manifest_sha256") != sha256(run_root / "source_manifest.sha256"):
        raise RuntimeError("unit certificate source binding mismatch")
    test_log = run_root / "results" / "unit_tests.log"
    if record.get("test_log") != artifact(test_log):
        raise RuntimeError("unit certificate test-log binding mismatch")
    return record


def stage_unit(run_root: Path, test_log: Path) -> None:
    validate_source_manifest(run_root)
    expected = run_root / "results" / "unit_tests.log"
    if test_log.resolve(strict=True) != expected.resolve(strict=True):
        raise RuntimeError("unit test log path is not the frozen exact filename")
    text = test_log.read_text()
    if " passed" not in text or " failed" in text or " error" in text.lower():
        raise RuntimeError("pytest log does not certify a clean unit run")
    atomic_json(
        run_root / "results" / "unit_certificate.json",
        {
            "schema": SCHEMA,
            "stage": "unit",
            "tests_passed": True,
            "source_manifest_sha256": sha256(run_root / "source_manifest.sha256"),
            "test_log": artifact(test_log),
            "runtime": runtime_provenance(),
        },
    )


def validate_dataset_files(data_root: Path) -> dict[str, dict[str, int | str]]:
    records = {}
    for name, expected in SVHN_FILES.items():
        path = data_root / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"official dataset file is missing or symlinked: {path}")
        actual = {"bytes": path.stat().st_size, "md5": md5(path), "sha256": sha256(path)}
        if actual != expected:
            raise RuntimeError(f"dataset checksum mismatch for {name}: {actual}")
        records[name] = actual
    return records


def runtime_provenance() -> dict:
    packages = {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "numpy", "scipy")
    }
    return {
        "python": platform.python_version(),
        "python_executable_invoked": os.path.abspath(sys.executable),
        "python_executable_resolved": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


def validate_array_seed(seed: int) -> None:
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is not None and task != str(seed):
        raise RuntimeError("Slurm array task id does not match the requested seed")


def validate_runtime_binding(runtime: dict, launch: dict, stage: str, seed: int | None) -> None:
    expected_keys = {
        "python", "python_executable_invoked", "python_executable_resolved", "platform",
        "packages", "cuda", "cudnn", "gpu", "slurm_job_id", "slurm_array_job_id",
        "slurm_array_task_id",
    }
    if not isinstance(runtime, dict) or set(runtime) != expected_keys:
        raise RuntimeError(f"{stage} runtime record is malformed")
    if runtime["python_executable_invoked"] != "/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python":
        raise RuntimeError(f"{stage} Python executable mismatch")
    if not isinstance(runtime["python_executable_resolved"], str) or not runtime["python_executable_resolved"]:
        raise RuntimeError(f"{stage} resolved Python executable is malformed")
    if not isinstance(runtime["python"], str) or not isinstance(runtime["platform"], str):
        raise RuntimeError(f"{stage} Python or platform provenance is malformed")
    if not isinstance(runtime["packages"], dict) or set(runtime["packages"]) != {
        "torch", "torchvision", "numpy", "scipy"
    } or any(not isinstance(value, str) for value in runtime["packages"].values()):
        raise RuntimeError(f"{stage} package provenance is malformed")
    gpu_stages = {
        "feasibility", "train", "calibrate", "odt", "eval", "verify_eval", "summary"
    }
    if stage in gpu_stages:
        if (
            not isinstance(runtime["gpu"], str)
            or "A6000" not in runtime["gpu"].upper()
            or not isinstance(runtime["cuda"], str)
            or type(runtime["cudnn"]) is not int
        ):
            raise RuntimeError(f"{stage} did not record the required A6000 CUDA runtime")
    elif runtime["gpu"] is not None:
        raise RuntimeError(f"{stage} unexpectedly recorded a GPU")
    job_id = runtime.get("slurm_job_id")
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError(f"{stage} did not record a Slurm job id")
    expected_job = launch["jobs"][stage]
    if stage in ARRAY_STAGES:
        if runtime.get("slurm_array_job_id") != expected_job:
            raise RuntimeError(f"{stage} Slurm array-job binding mismatch")
        if type(seed) is not int or runtime.get("slurm_array_task_id") != str(seed):
            raise RuntimeError(f"{stage} Slurm array-task binding mismatch")
        if not job_id.isdecimal():
            raise RuntimeError(f"{stage} Slurm element job id must be decimal")
    else:
        if job_id != expected_job:
            raise RuntimeError(f"{stage} Slurm job binding mismatch")
        if runtime.get("slurm_array_job_id") is not None or runtime.get("slurm_array_task_id") is not None:
            raise RuntimeError(f"{stage} unexpectedly ran as an array task")


def runtime_software_fingerprint(runtime: dict) -> dict:
    return {
        key: runtime[key]
        for key in (
            "python", "python_executable_invoked", "python_executable_resolved",
            "packages", "cuda", "cudnn",
        )
    }


def validate_software_match(current: dict, upstream: dict, label: str) -> None:
    """Require identical executable and library stacks across a dependency edge.

    The host platform string remains recorded but is intentionally not compared because
    Athena's compute and GPU partitions can report different kernels or node classes.
    """

    if runtime_software_fingerprint(current) != runtime_software_fingerprint(upstream):
        raise RuntimeError(f"software environment differs across {label}")


def validate_launch(run_root: Path, data_root: Path) -> dict:
    path = run_root / "results" / "launch.json"
    physical_file(path, run_root, "launch record")
    launch = strict_json(path)
    expected_keys = {
        "schema",
        "dag_schema",
        "run_root",
        "data_root",
        "source_manifest_sha256",
        "node_order",
        "jobs",
        "dependencies",
        "array",
        "unit_submitted_held",
    }
    if set(launch) != expected_keys:
        raise RuntimeError("launch record field inventory mismatch")
    if launch["schema"] != SCHEMA or launch["dag_schema"] != DAG_SCHEMA:
        raise RuntimeError("launch schema mismatch")
    if launch["run_root"] != str(run_root) or launch["data_root"] != str(data_root):
        raise RuntimeError("launch root binding mismatch")
    if launch["source_manifest_sha256"] != sha256(run_root / "source_manifest.sha256"):
        raise RuntimeError("launch source-manifest binding mismatch")
    if launch["node_order"] != list(DAG_ORDER):
        raise RuntimeError("launch node order mismatch")
    if launch["unit_submitted_held"] is not True:
        raise RuntimeError("launch did not record the transactional unit hold")
    jobs = launch["jobs"]
    if not isinstance(jobs, dict) or set(jobs) != set(DAG_ORDER):
        raise RuntimeError("launch job inventory mismatch")
    if any(not isinstance(value, str) or not value.isdecimal() for value in jobs.values()):
        raise RuntimeError("launch job ids must be decimal strings")
    if len(set(jobs.values())) != len(jobs):
        raise RuntimeError("launch job ids are not unique")
    expected_dependencies = {
        "prefetch": f"afterany:{jobs['unit']}",
        "feasibility": f"afterany:{jobs['prefetch']}",
        "train": f"afterany:{jobs['feasibility']}",
        "calibrate": f"afterany:{jobs['train']}",
        "odt": f"afterany:{jobs['calibrate']}",
        "eval": f"afterany:{jobs['odt']}",
        "verify_eval": f"afterany:{jobs['eval']}",
        "summary": f"afterany:{jobs['verify_eval']}",
    }
    if launch["dependencies"] != expected_dependencies:
        raise RuntimeError("launch dependency graph mismatch")
    if launch["array"] != {stage: "0-4" for stage in ARRAY_STAGES}:
        raise RuntimeError("launch array map mismatch")
    return launch


def validate_current_stage_launch(run_root: Path, data_root: Path, stage: str, seed: int | None) -> dict:
    launch = validate_launch(run_root, data_root)
    validate_runtime_binding(runtime_provenance(), launch, stage, seed)
    return launch


def validate_record_identity(record: dict, run_root: Path, launch: dict, stage: str, seed: int) -> None:
    if (
        not isinstance(record, dict)
        or record.get("schema") != SCHEMA
        or record.get("stage") != stage
        or type(record.get("seed")) is not int
        or record.get("seed") != seed
    ):
        raise RuntimeError(f"{stage} record identity mismatch")
    if record.get("source_manifest_sha256") != sha256(run_root / "source_manifest.sha256"):
        raise RuntimeError(f"{stage} record source binding mismatch")
    validate_runtime_binding(record.get("runtime"), launch, stage, seed)


def grayscale(batch: torch.Tensor) -> torch.Tensor:
    if batch.ndim != 4 or batch.shape[1:] != (3, 32, 32):
        raise ValueError("SVHN RGB batch must have shape [N,3,32,32]")
    scaled = batch.to(torch.float32) / 255.0
    coefficients = scaled.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    return (scaled * coefficients).sum(dim=1, keepdim=True)


class RawSVHN(torch.utils.data.Dataset):
    """Expose official uint8 images, labels, split id, and within-split index."""

    def __init__(self, dataset, split_id: int):
        self.data = dataset.data
        self.labels = dataset.labels
        self.split_id = split_id

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.data[index]),
            int(self.labels[index]),
            self.split_id,
            index,
        )


def load_training_datasets(data_root: Path):
    train = tv.datasets.SVHN(data_root, split="train", download=False)
    extra = tv.datasets.SVHN(data_root, split="extra", download=False)
    if len(train) != EXPECTED_SPLIT_LENGTHS["train"] or len(extra) != EXPECTED_SPLIT_LENGTHS["extra"]:
        raise RuntimeError("official SVHN training split length mismatch")
    return train, extra, torch.utils.data.ConcatDataset((RawSVHN(train, 0), RawSVHN(extra, 1)))


def exact_model() -> DoomsReportedChiNet:
    config = DoomsReportedChiNetConfig()
    model = DoomsReportedChiNet(config)
    if config != DoomsReportedChiNetConfig():
        raise RuntimeError("paper architecture was overridden")
    if model.architecture_schema != PAPER_REPORTED_ARCHITECTURE_SCHEMA:
        raise RuntimeError("paper architecture schema mismatch")
    if model.num_params() != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"parameter count changed: {model.num_params()}")
    if any(type(layer) is not DirectBilinearLayer for layer in model.layers):
        raise RuntimeError("paper model contains a non-direct bilinear layer")
    if any(not isinstance(norm, RmsBatchNorm) for norm in model.norms):
        raise RuntimeError("paper model contains an unexpected normalization")
    return model


def checkpoint_payload(model: DoomsReportedChiNet, seed: int, stage: str) -> dict:
    return {
        "schema": SCHEMA,
        "architecture_schema": PAPER_REPORTED_ARCHITECTURE_SCHEMA,
        "config": asdict(DoomsReportedChiNetConfig()),
        "parameter_count": model.num_params(),
        "seed": seed,
        "stage": stage,
        "state_dict": model.state_dict(),
        "rbn_runtime_state": [
            {
                "frozen": norm.frozen,
                "calibrating": norm.calibrating,
            }
            for norm in model.norms
        ],
    }


def load_checkpoint(path: Path, seed: int, stage: str, device: str = "cuda") -> DoomsReportedChiNet:
    payload = torch.load(path, map_location=device, weights_only=True)
    expected_payload_keys = {
        "schema",
        "architecture_schema",
        "config",
        "parameter_count",
        "seed",
        "stage",
        "state_dict",
        "rbn_runtime_state",
    }
    if not isinstance(payload, dict) or set(payload) != expected_payload_keys:
        raise RuntimeError("checkpoint payload inventory mismatch")
    if type(payload["seed"]) is not int or type(payload["parameter_count"]) is not int:
        raise RuntimeError("checkpoint integer identity fields are malformed")
    expected = {
        "schema": SCHEMA,
        "architecture_schema": PAPER_REPORTED_ARCHITECTURE_SCHEMA,
        "config": asdict(DoomsReportedChiNetConfig()),
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "seed": seed,
        "stage": stage,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"checkpoint {key} mismatch")
    model = exact_model().to(device)
    if not isinstance(payload["state_dict"], dict) or any(
        not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
        for value in payload["state_dict"].values()
    ):
        raise RuntimeError("checkpoint state dictionary is malformed or nonfinite")
    model.load_state_dict(payload["state_dict"], strict=True)
    runtime_state = payload.get("rbn_runtime_state")
    if not isinstance(runtime_state, list) or len(runtime_state) != len(model.norms):
        raise RuntimeError("checkpoint RBN runtime-state inventory mismatch")
    for norm, state in zip(model.norms, runtime_state):
        if not isinstance(state, dict) or set(state) != {"frozen", "calibrating"}:
            raise RuntimeError("checkpoint RBN runtime state is malformed")
        if type(state["frozen"]) is not bool or type(state["calibrating"]) is not bool:
            raise RuntimeError("checkpoint RBN runtime flags must be booleans")
        norm.frozen = state["frozen"]
        norm.calibrating = state["calibrating"]
    required_frozen = stage == "calibrated_final"
    if stage not in {"raw_final_epoch", "calibrated_final"}:
        raise RuntimeError("unknown checkpoint stage")
    for norm in model.norms:
        if (
            not bool(norm.initialized)
            or norm.frozen is not required_frozen
            or norm.calibrating
            or not bool(torch.isfinite(norm.scale))
            or float(norm.scale) <= 0.0
        ):
            raise RuntimeError("checkpoint RBN state does not match its stage")
    return model


def validate_calibrated_transition(
    raw_model: DoomsReportedChiNet, calibrated_model: DoomsReportedChiNet
) -> None:
    raw_state = raw_model.state_dict()
    calibrated_state = calibrated_model.state_dict()
    if set(raw_state) != set(calibrated_state):
        raise RuntimeError("raw and calibrated state inventories differ")
    allowed_changes = {
        f"norms.{index}.{name}"
        for index in range(3)
        for name in ("running_rms", "initialized", "_calib_sum", "_calib_count")
    }
    unexpected = [
        key for key in raw_state
        if key not in allowed_changes and not torch.equal(raw_state[key], calibrated_state[key])
    ]
    if unexpected:
        raise RuntimeError(f"calibration changed learned or non-calibration state: {unexpected}")
    raw_parameters = dict(raw_model.named_parameters())
    calibrated_parameters = dict(calibrated_model.named_parameters())
    if set(raw_parameters) != set(calibrated_parameters) or any(
        not torch.equal(raw_parameters[key], calibrated_parameters[key]) for key in raw_parameters
    ):
        raise RuntimeError("calibration changed learned parameters")


def stage_prefetch(run_root: Path, data_root: Path) -> None:
    entries = validate_source_manifest(run_root)
    unit = validate_unit_certificate(run_root)
    unit_path = run_root / "results" / "unit_certificate.json"
    unit_identity = capture_artifact(unit_path, run_root, "unit certificate")
    launch = validate_current_stage_launch(run_root, data_root, "prefetch", None)
    current_runtime = runtime_provenance()
    validate_runtime_binding(current_runtime, launch, "prefetch", None)
    validate_runtime_binding(unit["runtime"], launch, "unit", None)
    validate_software_match(current_runtime, unit["runtime"], "unit to prefetch")
    lengths = {}
    for split in ("train", "extra", "test"):
        dataset = tv.datasets.SVHN(data_root, split=split, download=True)
        lengths[split] = len(dataset)
    if lengths != EXPECTED_SPLIT_LENGTHS:
        raise RuntimeError(f"SVHN split lengths differ: {lengths}")
    files = validate_dataset_files(data_root)
    labels = {}
    for split in ("train", "extra", "test"):
        dataset = tv.datasets.SVHN(data_root, split=split, download=False)
        labels[split] = hashlib.sha256(np.asarray(dataset.labels, dtype=np.int64).tobytes()).hexdigest()
    validate_source_manifest(run_root)
    require_artifact_unchanged(unit_path, unit_identity, run_root, "unit certificate")
    atomic_json(
        run_root / "results" / "dataset_certificate.json",
        {
            "schema": SCHEMA,
            "stage": "prefetch",
            "unit_certificate": unit_identity,
            "source_manifest_sha256": sha256(run_root / "source_manifest.sha256"),
            "source_entries": entries,
            "dataset_files": files,
            "split_lengths": lengths,
            "labels_sha256": labels,
            "ordered_training_concatenation": ["train", "extra"],
            "runtime": current_runtime,
        },
    )


def validate_dataset_certificate(run_root: Path, data_root: Path, launch: dict | None = None) -> dict:
    path = run_root / "results" / "dataset_certificate.json"
    physical_file(path, run_root, "dataset certificate")
    record = strict_json(path)
    expected_keys = {
        "schema", "stage", "unit_certificate", "source_manifest_sha256", "source_entries", "dataset_files",
        "split_lengths", "labels_sha256", "ordered_training_concatenation", "runtime",
    }
    if set(record) != expected_keys or record["schema"] != SCHEMA or record["stage"] != "prefetch":
        raise RuntimeError("dataset certificate identity or inventory mismatch")
    if record["source_manifest_sha256"] != sha256(run_root / "source_manifest.sha256"):
        raise RuntimeError("dataset certificate source binding mismatch")
    unit = validate_unit_certificate(run_root)
    if record["unit_certificate"] != artifact(run_root / "results" / "unit_certificate.json"):
        raise RuntimeError("dataset certificate unit-chain mismatch")
    if record["source_entries"] != validate_source_manifest(run_root):
        raise RuntimeError("dataset certificate source-entry inventory mismatch")
    if record["dataset_files"] != validate_dataset_files(data_root):
        raise RuntimeError("dataset certificate file binding mismatch")
    if record["split_lengths"] != EXPECTED_SPLIT_LENGTHS:
        raise RuntimeError("dataset certificate split lengths mismatch")
    labels = record["labels_sha256"]
    if (
        not isinstance(labels, dict)
        or set(labels) != set(EXPECTED_SPLIT_LENGTHS)
        or any(not isinstance(value, str) or len(value) != 64 for value in labels.values())
    ):
        raise RuntimeError("dataset certificate label hashes are malformed")
    if record["ordered_training_concatenation"] != ["train", "extra"]:
        raise RuntimeError("dataset certificate training order mismatch")
    if launch is None:
        launch = validate_launch(run_root, data_root)
    validate_runtime_binding(record["runtime"], launch, "prefetch", None)
    validate_runtime_binding(unit["runtime"], launch, "unit", None)
    validate_software_match(record["runtime"], unit["runtime"], "unit to prefetch record")
    return record


def train_loader(data_root: Path, seed: int):
    _, _, dataset = load_training_datasets(data_root)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    return dataset, loader


def expected_cosine_learning_rate_trace(total_steps: int) -> np.ndarray:
    parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=0.0
    )
    values = []
    for _ in range(total_steps):
        values.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        scheduler.step()
    return np.asarray(values, dtype=np.float64)


def stage_train(run_root: Path, data_root: Path, seed: int) -> None:
    validate_source_manifest(run_root)
    validate_unit_certificate(run_root)
    validate_dataset_files(data_root)
    validate_array_seed(seed)
    launch = validate_current_stage_launch(run_root, data_root, "train", seed)
    current_runtime = runtime_provenance()
    validate_runtime_binding(current_runtime, launch, "train", seed)
    dataset_certificate = validate_dataset_certificate(run_root, data_root, launch)
    feasibility_record = validate_feasibility_record(run_root, launch)
    validate_software_match(
        current_runtime, feasibility_record["runtime"], "feasibility to train"
    )
    dataset_path = run_root / "results" / "dataset_certificate.json"
    feasibility_path = run_root / "results" / "feasibility.json"
    feasibility_matrix_path = run_root / "results" / "feasibility_matrices.npz"
    dataset_identity = capture_artifact(dataset_path, run_root, "dataset certificate")
    feasibility_identity = capture_artifact(feasibility_path, run_root, "feasibility record")
    feasibility_matrix_identity = capture_artifact(
        feasibility_matrix_path, run_root, "feasibility matrix artifact"
    )
    configure_determinism(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    dataset, loader = train_loader(data_root, seed)
    model = exact_model().cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1.0,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=False,
    )
    total_steps = 20 * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=0.0
    )
    noise_seed = 1_000_003 + seed
    noise_generator = torch.Generator(device="cuda").manual_seed(noise_seed)
    epochs = []
    learning_rates = []
    global_step = 0
    started = time.monotonic()
    for epoch in range(20):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for rgb, labels, _split_ids, _indices in loader:
            images = grayscale(rgb.cuda(non_blocking=True))
            noise = torch.randn(
                images.shape,
                dtype=images.dtype,
                device=images.device,
                generator=noise_generator,
            )
            images = images + 0.3 * noise
            labels = labels.cuda(non_blocking=True)
            learning_rates.append(float(optimizer.param_groups[0]["lr"]))
            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(images, labels)
            assert loss is not None
            loss.backward()
            optimizer.step()
            scheduler.step()
            batch = len(labels)
            loss_sum += float(loss.detach()) * batch
            correct += int((logits.detach().argmax(-1) == labels).sum())
            seen += batch
            global_step += 1
        if seen != len(dataset):
            raise RuntimeError("training epoch did not consume train+extra exactly once")
        record = {
            "epoch": epoch,
            "loss": loss_sum / seen,
            "accuracy": correct / seen,
            "examples": seen,
            "steps": len(loader),
            "last_learning_rate": learning_rates[-1],
        }
        epochs.append(record)
        print(json.dumps(record), flush=True)
    if global_step != total_steps or len(learning_rates) != total_steps:
        raise RuntimeError("optimizer or scheduler step count mismatch")
    expected_learning_rates = expected_cosine_learning_rate_trace(total_steps)
    if not np.array_equal(np.asarray(learning_rates, dtype=np.float64), expected_learning_rates):
        raise RuntimeError("training learning-rate trace differs from the frozen cosine schedule")
    validate_source_manifest(run_root)
    checkpoint = run_root / "results" / f"paper_seed_{seed}_raw.pt"
    atomic_torch_save(checkpoint, checkpoint_payload(model, seed, "raw_final_epoch"))
    verified_model = load_checkpoint(checkpoint, seed, "raw_final_epoch")
    del verified_model
    require_artifact_unchanged(dataset_path, dataset_identity, run_root, "dataset certificate")
    require_artifact_unchanged(feasibility_path, feasibility_identity, run_root, "feasibility record")
    require_artifact_unchanged(
        feasibility_matrix_path, feasibility_matrix_identity, run_root, "feasibility matrix artifact"
    )
    result = {
        "schema": SCHEMA,
        "stage": "train",
        "seed": seed,
        "paper": {
            "arxiv_id": PAPER_ARXIV_ID,
            "pdf_sha256": PAPER_PDF_SHA256,
            "reported_test_accuracy": 0.854,
        },
        "reported_configuration": {
            "training_splits": ["train", "extra"],
            "grayscale": True,
            "input_noise_norm": 0.3,
            "width": 256,
            "bilinear_layers": 3,
            "batch_size": TRAIN_BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 1.0,
            "schedule": "cosine",
            "epochs": 20,
        },
        "implementation_assumptions": ASSUMPTIONS,
        "architecture_schema": PAPER_REPORTED_ARCHITECTURE_SCHEMA,
        "parameter_count": model.num_params(),
        "dataset_examples": len(dataset),
        "steps_per_epoch": len(loader),
        "total_steps": total_steps,
        "noise_seed": noise_seed,
        "learning_rate_trace_sha256": hashlib.sha256(
            np.asarray(learning_rates, dtype=np.float64).tobytes()
        ).hexdigest(),
        "learning_rate_first": learning_rates[0],
        "learning_rate_last_before_step": learning_rates[-1],
        "learning_rate_after_final_step": optimizer.param_groups[0]["lr"],
        "epochs": epochs,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": artifact(checkpoint),
        "dataset_certificate": dataset_identity,
        "consumed_feasibility_json": feasibility_identity,
        "consumed_feasibility_matrix": feasibility_matrix_identity,
        "dataset_files": validate_dataset_files(data_root),
        "source_manifest_sha256": sha256(run_root / "source_manifest.sha256"),
        "runtime": current_runtime,
    }
    atomic_json(run_root / "results" / f"paper_seed_{seed}_train.json", result)


def calibration_batch(train, extra, batch_index: int) -> torch.Tensor:
    start = batch_index * CALIBRATION_BATCH_SIZE
    stop = start + CALIBRATION_BATCH_SIZE
    train_length = len(train)
    pieces = []
    if start < train_length:
        pieces.append(train.data[start : min(stop, train_length)])
    if stop > train_length:
        pieces.append(extra.data[max(0, start - train_length) : stop - train_length])
    array = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)
    if len(array) != CALIBRATION_BATCH_SIZE:
        raise RuntimeError("calibration batch is not full-sized")
    return torch.from_numpy(array)


def stage_calibrate(run_root: Path, data_root: Path, seed: int) -> None:
    validate_source_manifest(run_root)
    validate_unit_certificate(run_root)
    validate_dataset_files(data_root)
    validate_array_seed(seed)
    launch = validate_current_stage_launch(run_root, data_root, "calibrate", seed)
    current_runtime = runtime_provenance()
    validate_runtime_binding(current_runtime, launch, "calibrate", seed)
    dataset_certificate = validate_dataset_certificate(run_root, data_root, launch)
    configure_determinism(seed)
    raw_path = run_root / "results" / f"paper_seed_{seed}_raw.pt"
    train_json_path = run_root / "results" / f"paper_seed_{seed}_train.json"
    train_record = validate_training_chain(
        run_root, data_root, launch, seed, dataset_certificate, current_runtime
    )
    raw_identity = capture_artifact(raw_path, run_root, "raw checkpoint")
    train_identity = capture_artifact(train_json_path, run_root, "train result")
    dataset_path = run_root / "results" / "dataset_certificate.json"
    dataset_identity = capture_artifact(dataset_path, run_root, "dataset certificate")
    model = load_checkpoint(raw_path, seed, "raw_final_epoch").eval()
    train, extra, _ = load_training_datasets(data_root)
    calibration_count = CALIBRATION_BATCH_SIZE * CALIBRATION_BATCHES
    calibration_positions = np.arange(calibration_count, dtype=np.int64)
    calibration_identities = np.stack(
        (
            (calibration_positions >= len(train)).astype(np.int64),
            np.where(
                calibration_positions < len(train),
                calibration_positions,
                calibration_positions - len(train),
            ),
        ),
        axis=1,
    )

    def batch_at(index: int) -> torch.Tensor:
        return grayscale(calibration_batch(train, extra, index).cuda(non_blocking=True))

    report = calibrate_rbn_sequential(
        model,
        batch_at,
        lambda module, batch: module(batch)[0],
        iters=CALIBRATION_BATCHES,
    )
    for norm in model.norms:
        if (
            not bool(norm.initialized)
            or not norm.frozen
            or norm.calibrating
            or not bool(torch.isfinite(norm.scale))
            or float(norm.scale) <= 0
        ):
            raise RuntimeError("calibration did not leave every RBN export-ready")
    sample64 = batch_at(0)[:16].double()
    model64 = copy.deepcopy(model).double().eval()
    with torch.no_grad():
        module_logits = model64(sample64)[0]
    network = export_homogeneous_network(
        model64, require_frozen_rbn=True
    )
    with torch.no_grad():
        folded_logits = homogeneous_forward(network, sample64)
    replay = replay_metrics(module_logits, folded_logits)
    if replay["passed"] is not True:
        raise RuntimeError(f"strict folded replay failed: {replay}")
    calibrated = run_root / "results" / f"paper_seed_{seed}_calibrated.pt"
    atomic_torch_save(calibrated, checkpoint_payload(model, seed, "calibrated_final"))
    reloaded = load_checkpoint(calibrated, seed, "calibrated_final").double().eval()
    validate_calibrated_transition(model, reloaded.float())
    reloaded = reloaded.double()
    reloaded_network = export_homogeneous_network(reloaded, require_frozen_rbn=True)
    with torch.no_grad():
        reloaded_logits = reloaded(sample64)[0]
        reloaded_folded_logits = homogeneous_forward(reloaded_network, sample64)
    reloaded_replay = replay_metrics(reloaded_logits, reloaded_folded_logits)
    if reloaded_replay["passed"] is not True:
        raise RuntimeError(f"post-save strict folded replay failed: {reloaded_replay}")
    validate_source_manifest(run_root)
    require_artifact_unchanged(raw_path, raw_identity, run_root, "raw checkpoint")
    require_artifact_unchanged(train_json_path, train_identity, run_root, "train result")
    require_artifact_unchanged(dataset_path, dataset_identity, run_root, "dataset certificate")
    atomic_json(
        run_root / "results" / f"paper_seed_{seed}_calibrate.json",
        {
            "schema": SCHEMA,
            "stage": "calibrate",
            "seed": seed,
            "raw_checkpoint": raw_identity,
            "consumed_train_json": train_identity,
            "calibrated_checkpoint": artifact(calibrated),
            "dataset_certificate": dataset_identity,
            "dataset_labels_sha256": dataset_certificate["labels_sha256"],
            "calibration_example_identity": "ordered (split_id, within_split_index) int64 pairs",
            "calibration_example_identities_sha256": hashlib.sha256(
                calibration_identities.tobytes()
            ).hexdigest(),
            "calibration_examples": int(calibration_identities.shape[0]),
            "batch_size": CALIBRATION_BATCH_SIZE,
            "batches": CALIBRATION_BATCHES,
            "noise": False,
            "sites": [asdict(site) for site in report.sites],
            "strict_folded_replay": replay,
            "post_save_reload_strict_folded_replay": reloaded_replay,
            "implementation_assumptions": ASSUMPTIONS,
            "source_manifest_sha256": sha256(run_root / "source_manifest.sha256"),
            "runtime": current_runtime,
        },
    )


def cluster_complete_rank(values: torch.Tensor, rank: int) -> int:
    """Move a proposed truncation upward until it no longer cuts a cluster."""

    dimension = values.numel()
    if not 1 <= rank <= dimension:
        raise ValueError("rank must lie in the spectrum")
    while rank < dimension and not eigengap_diagnostics(values, rank).resolved:
        rank += 1
    return rank


def epsilon_rank_schedule(
    eigensystems, full_norm_squared: float, epsilon_grid=EPSILON_GRID
) -> list[dict]:
    """Allocate ranks independently using the corrected trace-tail HSVD budget."""

    if not math.isfinite(full_norm_squared) or full_norm_squared <= 0.0:
        raise ValueError("full coefficient norm must be finite and positive")
    projection_count = 2 ** len(eigensystems) - 1
    schedules = []
    previous = None
    for epsilon in epsilon_grid:
        budget = epsilon**2 * full_norm_squared / projection_count
        ranks = []
        tails = []
        padded_tails = []
        uncertainties = []
        for values, _vectors in eigensystems:
            dimension = values.numel()
            uncertainty = (
                1000.0
                * torch.finfo(values.dtype).eps
                * dimension
                * max(float(values.abs().max()), torch.finfo(values.dtype).tiny)
            )
            proposed = dimension
            if epsilon > 0.0:
                for rank in range(1, dimension + 1):
                    nominal_tail = float(discarded_trace(values, rank))
                    padded_tail = nominal_tail + (dimension - rank) * uncertainty
                    if padded_tail <= budget:
                        proposed = rank
                        break
            rank = cluster_complete_rank(values, proposed)
            ranks.append(rank)
            nominal_tail = float(discarded_trace(values, rank))
            tails.append(nominal_tail)
            padded_tails.append(nominal_tail + (dimension - rank) * uncertainty)
            uncertainties.append(uncertainty)
        if previous is not None and any(rank > old for rank, old in zip(ranks, previous)):
            raise RuntimeError("epsilon-derived ranks are not nested")
        previous = tuple(ranks)
        schedules.append(
            {
                "epsilon": epsilon,
                "per_projection_tail_budget": budget,
                "rank_tuple": ranks,
                "discarded_trace_by_bond": tails,
                "heuristic_fp_pad_per_eigenvalue_by_bond": uncertainties,
                "heuristically_padded_discarded_trace_by_bond": padded_tails,
            }
        )
    return schedules


def corrected_rank_allocation(eigensystems, full_norm_squared: float) -> dict:
    """Build the corrected trace-tail allocation with a non-certifying FP pad."""
    schedules = epsilon_rank_schedule(eigensystems, full_norm_squared)
    records = []
    for schedule in schedules:
        ranks = tuple(schedule["rank_tuple"])
        nominal_bound_squared = float(hierarchical_tail_bound_squared(eigensystems, ranks))
        layer_count = len(eigensystems) - 1
        bound_squared = sum(
            2 ** (layer_count - bond) * tail
            for bond, tail in enumerate(schedule["heuristically_padded_discarded_trace_by_bond"])
        )
        target_squared = schedule["epsilon"] ** 2 * full_norm_squared
        bound_slack = 1e-8 * target_squared + 1e-10 * full_norm_squared
        tail_slack = 1e-12 * full_norm_squared
        if any(
            tail > schedule["per_projection_tail_budget"] + tail_slack
            for tail in schedule["heuristically_padded_discarded_trace_by_bond"]
        ):
            raise RuntimeError("epsilon schedule violates a per-projection tail budget")
        if bound_squared > target_squared + bound_slack:
            raise RuntimeError("epsilon schedule violates its corrected HSVD budget")
        records.append(
            {
                **schedule,
                "nominal_weighted_hsvd_bound_squared": nominal_bound_squared,
                "heuristically_padded_weighted_hsvd_estimate_squared": bound_squared,
                "heuristically_padded_relative_hsvd_estimate": math.sqrt(
                    bound_squared / full_norm_squared
                ),
                "target_relative_error": schedule["epsilon"],
                "padded_estimate_numerical_acceptance_slack": bound_slack,
                "per_projection_tail_numerical_slack": tail_slack,
                "padded_estimate_within_target_plus_declared_slack": True,
                "all_boundaries_resolved": all(
                    rank == values.numel() or eigengap_diagnostics(values, rank).resolved
                    for (values, _), rank in zip(eigensystems, ranks)
                ),
            }
        )
    return {
        "name": "hsvd_trace_equal_projection_budget_with_heuristic_fp_pad_v3",
        "projection_count": 2 ** len(eigensystems) - 1,
        "epsilon_grid": list(EPSILON_GRID),
        "schedules": records,
        "paper_printed_gram_frobenius_rule": (
            "Computed and evaluated separately. The corrected trace-tail family gives an "
            "exact-real-arithmetic HSVD estimate. Its float64 pad is heuristic and non-certifying."
        ),
    }


def paper_printed_rank_allocation(eigensystems) -> dict:
    """Implement the printed threshold and expose the cluster-safe evaluated rank."""

    projection_count = 2 ** len(eigensystems) - 1
    records = []
    previous = None
    for epsilon in EPSILON_GRID:
        minimal_ranks = []
        evaluated_ranks = []
        full_gram_norms = []
        minimal_discarded_gram_norms = []
        evaluated_discarded_gram_norms = []
        budgets = []
        for values, _vectors in eigensystems:
            squares = values.clamp_min(0).square()
            full = float(squares.sum())
            budget = epsilon**2 * full / projection_count
            proposed = values.numel()
            for rank in range(1, values.numel() + 1):
                if float(squares[rank:].sum()) <= budget:
                    proposed = rank
                    break
            rank = values.numel() if epsilon == 0.0 else cluster_complete_rank(values, proposed)
            minimal_ranks.append(proposed)
            evaluated_ranks.append(rank)
            full_gram_norms.append(full)
            minimal_discarded_gram_norms.append(float(squares[proposed:].sum()))
            evaluated_discarded_gram_norms.append(float(squares[rank:].sum()))
            budgets.append(budget)
        if previous is not None and any(rank > old for rank, old in zip(evaluated_ranks, previous)):
            raise RuntimeError("paper-printed epsilon-derived ranks are not nested")
        previous = tuple(evaluated_ranks)
        records.append(
            {
                "epsilon": epsilon,
                "minimal_printed_rule_rank_tuple": minimal_ranks,
                "evaluated_cluster_complete_rank_tuple": evaluated_ranks,
                "rank_tuple": evaluated_ranks,
                "full_gram_frobenius_squared_by_bond": full_gram_norms,
                "discarded_gram_frobenius_squared_at_minimal_printed_rank_by_bond": (
                    minimal_discarded_gram_norms
                ),
                "discarded_gram_frobenius_squared_at_evaluated_rank_by_bond": (
                    evaluated_discarded_gram_norms
                ),
                "per_bond_printed_budget": budgets,
                "mean_fraction_dimensions_removed": 1.0 - sum(evaluated_ranks) / (257.0 * 4.0),
                "all_boundaries_resolved": all(
                    rank == values.numel() or eigengap_diagnostics(values, rank).resolved
                    for (values, _), rank in zip(eigensystems, evaluated_ranks)
                ),
            }
        )
    return {
        "name": "dooms_et_al_printed_rule_with_cluster_safe_evaluation_v2",
        "projection_count": projection_count,
        "epsilon_grid": list(EPSILON_GRID),
        "schedules": records,
        "claim_boundary": (
            "The minimal rank literally implements the inequality printed in arXiv:2504.02667v1. "
            "The evaluated rank is forced to full dimension at epsilon zero and otherwise moved "
            "upward only to avoid an unresolved numerical eigenvalue cluster. Neither is presented "
            "as a corrected coefficient-error bound."
        ),
    }


def uniform_dimension_rank_sweep(eigensystems) -> dict:
    """Prospective uniform-rank sweep matching the x-axis used in paper Figure 2."""

    dimensions = [values.numel() for values, _vectors in eigensystems]
    if not dimensions or len(set(dimensions)) != 1:
        raise ValueError("uniform dimension sweep requires equal nonempty bond dimensions")
    dimension = dimensions[0]
    schedules = []
    for fraction in REMOVAL_GRID:
        proposed = max(1, int(round((1.0 - fraction) * dimension)))
        ranks = [cluster_complete_rank(values, proposed) for values, _ in eigensystems]
        schedules.append(
            {
                "target_fraction_dimensions_removed": fraction,
                "nominal_rank_before_cluster_completion": proposed,
                "rank_tuple": ranks,
                "actual_mean_fraction_dimensions_removed": 1.0 - sum(ranks) / (
                    float(dimension) * len(eigensystems)
                ),
                "all_boundaries_resolved": all(
                    rank == values.numel() or eigengap_diagnostics(values, rank).resolved
                    for (values, _), rank in zip(eigensystems, ranks)
                ),
            }
        )
    return {
        "name": "uniform_dimension_removal_sweep_figure2_axis_v1",
        "target_removal_grid": list(REMOVAL_GRID),
        "homogeneous_coordinate_convention": (
            "Fractions count all 257 homogeneous bond coordinates. The paper reports width 256 "
            "and does not specify how its constant coordinate is counted."
        ),
        "schedules": schedules,
    }


def network_to_arrays(network: HomogeneousChiTN, prefix: str) -> dict[str, np.ndarray]:
    arrays = {
        f"{prefix}_embedding": network.embedding.detach().cpu().numpy(),
        f"{prefix}_head": network.head.detach().cpu().numpy(),
    }
    arrays.update(
        {
            f"{prefix}_core_{index}": core.detach().cpu().numpy()
            for index, core in enumerate(network.cores)
        }
    )
    return arrays


def arrays_to_network(bundle, prefix: str, device: str = "cuda") -> HomogeneousChiTN:
    embedding = torch.from_numpy(bundle[f"{prefix}_embedding"]).to(device)
    cores = tuple(
        torch.from_numpy(bundle[f"{prefix}_core_{index}"]).to(device)
        for index in range(3)
    )
    head = torch.from_numpy(bundle[f"{prefix}_head"]).to(device)
    return HomogeneousChiTN(embedding, cores, head)


def require_close(actual, expected, label: str, *, relative: float = 1e-9, absolute: float = 1e-12) -> None:
    actual_value = finite_real(actual, label)
    expected_value = finite_real(expected, f"expected {label}")
    if not math.isclose(actual_value, expected_value, rel_tol=relative, abs_tol=absolute):
        raise RuntimeError(f"{label} mismatch: actual={actual_value}, expected={expected_value}")


def expected_matrix_metadata(path: Path, arrays: dict[str, np.ndarray]) -> dict:
    return {
        **artifact(path),
        "allow_pickle": False,
        "keys": {
            key: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for key, value in sorted(arrays.items())
        },
    }


def validate_matrix_bundle(
    run_root: Path, matrix_path: Path, declared: dict, *, device: str = "cuda"
) -> tuple[HomogeneousChiTN, tuple[torch.Tensor, ...], dict[str, np.ndarray]]:
    physical_file(matrix_path, run_root, "ODT matrix artifact")
    expected_declared_keys = {"path", "bytes", "sha256", "allow_pickle", "keys"}
    if not isinstance(declared, dict) or set(declared) != expected_declared_keys:
        raise RuntimeError("ODT matrix metadata inventory mismatch")
    if declared.get("allow_pickle") is not False:
        raise RuntimeError("ODT matrix artifact must forbid pickle")
    arrays: dict[str, np.ndarray] = {}
    with np.load(matrix_path, allow_pickle=False) as bundle:
        if set(bundle.files) != set(MATRIX_LAYOUT):
            raise RuntimeError("ODT matrix key inventory mismatch")
        for key, (shape, dtype) in MATRIX_LAYOUT.items():
            value = bundle[key]
            if value.shape != shape or value.dtype != dtype or not value.dtype.isnative:
                raise RuntimeError(f"ODT matrix metadata mismatch for {key}")
            if not np.isfinite(value).all():
                raise RuntimeError(f"ODT matrix contains nonfinite values in {key}")
            arrays[key] = value.copy()
    if declared != expected_matrix_metadata(matrix_path, arrays):
        raise RuntimeError("ODT JSON does not bind the exact matrix artifact")
    canonical = arrays_to_network(arrays, "canonical", device=device)
    recomputed_global = canonical_environments(canonical)
    recomputed_local = local_environments(canonical)
    for name, recomputed in (("global_grams", recomputed_global), ("local_grams", recomputed_local)):
        stored = torch.from_numpy(arrays[name]).to(device)
        for bond in range(4):
            scale = max(1.0, float(torch.linalg.matrix_norm(recomputed[bond])))
            disagreement = float(torch.linalg.matrix_norm(stored[bond] - recomputed[bond]))
            if not math.isfinite(disagreement) or disagreement > 1e-9 * scale:
                raise RuntimeError(f"serialized {name} bond {bond} disagrees with the canonical tree")
    grams = tuple(torch.from_numpy(arrays["global_grams"][index]).to(device) for index in range(4))
    values = torch.from_numpy(arrays["global_eigenvalues"]).to(device)
    vectors = torch.from_numpy(arrays["global_eigenvectors"]).to(device)
    identity = torch.eye(257, dtype=torch.float64, device=device)
    for bond, gram in enumerate(grams):
        gram_scale = max(1.0, float(torch.linalg.matrix_norm(gram)))
        symmetry = float(torch.linalg.matrix_norm(gram - gram.T))
        if not math.isfinite(symmetry) or symmetry > 1e-10 * gram_scale:
            raise RuntimeError(f"global Gram {bond} is materially nonsymmetric")
        bond_values = values[bond]
        bond_vectors = vectors[bond]
        if bool((bond_values[:-1] < bond_values[1:]).any()):
            raise RuntimeError(f"global eigenvalues {bond} are not descending")
        negative_tolerance = 1e-10 * max(1.0, float(bond_values.abs().max()))
        if float(bond_values.min()) < -negative_tolerance:
            raise RuntimeError(f"global Gram {bond} is materially indefinite")
        orthogonality = float(torch.linalg.matrix_norm(bond_vectors.T @ bond_vectors - identity))
        residual = float(
            torch.linalg.matrix_norm(gram @ bond_vectors - bond_vectors * bond_values[None, :])
        )
        if orthogonality > 1e-10 * 257 or residual > 1e-9 * gram_scale:
            raise RuntimeError(f"global eigensystem {bond} failed its residual gate")
    systems = tuple((values[index], vectors[index]) for index in range(4))
    return canonical, systems, arrays


def validate_rank_allocation(rank_allocation: dict, eigensystems, full_norm_squared: float) -> tuple[dict, ...]:
    expected_allocation_keys = {
        "name",
        "projection_count",
        "epsilon_grid",
        "schedules",
        "paper_printed_gram_frobenius_rule",
    }
    if not isinstance(rank_allocation, dict) or set(rank_allocation) != expected_allocation_keys:
        raise RuntimeError("rank-allocation inventory mismatch")
    if rank_allocation["name"] != "hsvd_trace_equal_projection_budget_with_heuristic_fp_pad_v3":
        raise RuntimeError("rank-allocation algorithm mismatch")
    if type(rank_allocation["projection_count"]) is not int or rank_allocation["projection_count"] != 15:
        raise RuntimeError("rank-allocation projection count mismatch")
    if rank_allocation["epsilon_grid"] != list(EPSILON_GRID):
        raise RuntimeError("rank-allocation epsilon grid mismatch")
    if rank_allocation["paper_printed_gram_frobenius_rule"] != (
        "Computed and evaluated separately. The corrected trace-tail family gives an "
        "exact-real-arithmetic HSVD estimate. Its float64 pad is heuristic and non-certifying."
    ):
        raise RuntimeError("rank-allocation paper-rule annotation mismatch")
    schedules = rank_allocation["schedules"]
    recomputed = epsilon_rank_schedule(eigensystems, full_norm_squared)
    if not isinstance(schedules, list) or len(schedules) != len(EPSILON_GRID):
        raise RuntimeError("rank-allocation schedule count mismatch")
    expected_schedule_keys = {
        "epsilon",
        "per_projection_tail_budget",
        "rank_tuple",
        "discarded_trace_by_bond",
        "heuristic_fp_pad_per_eigenvalue_by_bond",
        "heuristically_padded_discarded_trace_by_bond",
        "nominal_weighted_hsvd_bound_squared",
        "heuristically_padded_weighted_hsvd_estimate_squared",
        "heuristically_padded_relative_hsvd_estimate",
        "target_relative_error",
        "padded_estimate_numerical_acceptance_slack",
        "per_projection_tail_numerical_slack",
        "padded_estimate_within_target_plus_declared_slack",
        "all_boundaries_resolved",
    }
    previous_ranks = None
    previous_tails = None
    validated = []
    for index, (stored, base) in enumerate(zip(schedules, recomputed)):
        if not isinstance(stored, dict) or set(stored) != expected_schedule_keys:
            raise RuntimeError(f"schedule {index} field inventory mismatch")
        epsilon = EPSILON_GRID[index]
        require_close(stored["epsilon"], epsilon, f"schedule {index} epsilon", relative=0.0)
        require_close(stored["target_relative_error"], epsilon, f"schedule {index} target", relative=0.0)
        ranks = stored["rank_tuple"]
        if (
            not isinstance(ranks, list)
            or len(ranks) != 4
            or any(type(rank) is not int or not 1 <= rank <= 257 for rank in ranks)
        ):
            raise RuntimeError(f"schedule {index} rank tuple is malformed")
        if ranks != base["rank_tuple"]:
            raise RuntimeError(f"schedule {index} ranks are not the minimal cluster-complete allocation")
        tails = stored["discarded_trace_by_bond"]
        padded_tails = stored["heuristically_padded_discarded_trace_by_bond"]
        uncertainties = stored["heuristic_fp_pad_per_eigenvalue_by_bond"]
        if (
            not isinstance(tails, list) or len(tails) != 4
            or not isinstance(padded_tails, list) or len(padded_tails) != 4
            or not isinstance(uncertainties, list) or len(uncertainties) != 4
        ):
            raise RuntimeError(f"schedule {index} tail inventory mismatch")
        for bond, (actual, expected) in enumerate(zip(tails, base["discarded_trace_by_bond"])):
            require_close(actual, expected, f"schedule {index} bond {bond} tail")
            require_close(
                padded_tails[bond], base["heuristically_padded_discarded_trace_by_bond"][bond],
                f"schedule {index} bond {bond} heuristically padded tail",
            )
            require_close(
                uncertainties[bond], base["heuristic_fp_pad_per_eigenvalue_by_bond"][bond],
                f"schedule {index} bond {bond} heuristic FP pad",
            )
        budget = epsilon**2 * full_norm_squared / 15
        require_close(stored["per_projection_tail_budget"], budget, f"schedule {index} budget")
        nominal_bound = float(hierarchical_tail_bound_squared(eigensystems, tuple(ranks)))
        layer_count = len(eigensystems) - 1
        bound = sum(
            2 ** (layer_count - bond) * tail
            for bond, tail in enumerate(padded_tails)
        )
        relative_bound = math.sqrt(max(0.0, bound) / full_norm_squared)
        bound_slack = 1e-8 * epsilon**2 * full_norm_squared + 1e-10 * full_norm_squared
        tail_slack = 1e-12 * full_norm_squared
        require_close(
            stored["nominal_weighted_hsvd_bound_squared"], nominal_bound,
            f"schedule {index} nominal bound",
        )
        require_close(
            stored["heuristically_padded_weighted_hsvd_estimate_squared"], bound,
            f"schedule {index} padded estimate",
        )
        require_close(
            stored["heuristically_padded_relative_hsvd_estimate"], relative_bound,
            f"schedule {index} relative padded estimate",
        )
        require_close(
            stored["padded_estimate_numerical_acceptance_slack"], bound_slack,
            f"schedule {index} padded-estimate slack",
        )
        require_close(
            stored["per_projection_tail_numerical_slack"], tail_slack,
            f"schedule {index} tail slack",
        )
        within = bound <= epsilon**2 * full_norm_squared + bound_slack
        resolved = all(
            rank == values.numel() or eigengap_diagnostics(values, rank).resolved
            for (values, _vectors), rank in zip(eigensystems, ranks)
        )
        if stored["padded_estimate_within_target_plus_declared_slack"] is not within or not within:
            raise RuntimeError(f"schedule {index} padded-estimate gate failed")
        if stored["all_boundaries_resolved"] is not resolved or not resolved:
            raise RuntimeError(f"schedule {index} cuts an unresolved eigenvalue cluster")
        if any(float(tail) > budget + tail_slack for tail in padded_tails):
            raise RuntimeError(f"schedule {index} exceeds a per-projection tail budget")
        if previous_ranks is not None and any(new > old for new, old in zip(ranks, previous_ranks)):
            raise RuntimeError("rank schedules are not nested")
        if previous_tails is not None and any(new + tail_slack < old for new, old in zip(tails, previous_tails)):
            raise RuntimeError("discarded traces are not monotone")
        previous_ranks = ranks
        previous_tails = tails
        validated.append(stored)
    return tuple(validated)


def validate_paper_printed_rank_allocation(record: dict, eigensystems) -> tuple[dict, ...]:
    expected = paper_printed_rank_allocation(eigensystems)
    if not isinstance(record, dict) or set(record) != set(expected):
        raise RuntimeError("paper-printed rank-allocation inventory mismatch")
    for key in ("name", "projection_count", "epsilon_grid", "claim_boundary"):
        if record[key] != expected[key]:
            raise RuntimeError(f"paper-printed rank-allocation mismatch for {key}")
    schedules = record["schedules"]
    expected_schedules = expected["schedules"]
    if not isinstance(schedules, list) or len(schedules) != len(expected_schedules):
        raise RuntimeError("paper-printed schedule count mismatch")
    keys = {
        "epsilon", "rank_tuple", "minimal_printed_rule_rank_tuple",
        "evaluated_cluster_complete_rank_tuple", "full_gram_frobenius_squared_by_bond",
        "discarded_gram_frobenius_squared_at_minimal_printed_rank_by_bond",
        "discarded_gram_frobenius_squared_at_evaluated_rank_by_bond",
        "per_bond_printed_budget",
        "mean_fraction_dimensions_removed", "all_boundaries_resolved",
    }
    previous = None
    for index, (actual, target) in enumerate(zip(schedules, expected_schedules)):
        if not isinstance(actual, dict) or set(actual) != keys:
            raise RuntimeError(f"paper-printed schedule {index} inventory mismatch")
        if actual["epsilon"] != target["epsilon"]:
            raise RuntimeError(f"paper-printed schedule {index} epsilon identity mismatch")
        if actual["rank_tuple"] != target["rank_tuple"]:
            raise RuntimeError(f"paper-printed schedule {index} evaluated rank mismatch")
        if actual["minimal_printed_rule_rank_tuple"] != target["minimal_printed_rule_rank_tuple"]:
            raise RuntimeError(f"paper-printed schedule {index} minimal rank mismatch")
        if actual["evaluated_cluster_complete_rank_tuple"] != target["evaluated_cluster_complete_rank_tuple"]:
            raise RuntimeError(f"paper-printed schedule {index} cluster-complete rank mismatch")
        if actual["rank_tuple"] != actual["evaluated_cluster_complete_rank_tuple"]:
            raise RuntimeError(f"paper-printed schedule {index} evaluated rank alias mismatch")
        for rank_field in (
            "rank_tuple", "minimal_printed_rule_rank_tuple",
            "evaluated_cluster_complete_rank_tuple",
        ):
            if any(
                type(rank) is not int or not 1 <= rank <= 257
                for rank in actual[rank_field]
            ):
                raise RuntimeError(
                    f"paper-printed schedule {index} {rank_field} type mismatch"
                )
        for field in (
            "full_gram_frobenius_squared_by_bond",
            "discarded_gram_frobenius_squared_at_minimal_printed_rank_by_bond",
            "discarded_gram_frobenius_squared_at_evaluated_rank_by_bond",
            "per_bond_printed_budget",
        ):
            if not isinstance(actual[field], list) or len(actual[field]) != 4:
                raise RuntimeError(f"paper-printed schedule {index} {field} inventory mismatch")
            for left, right in zip(actual[field], target[field]):
                require_close(left, right, f"paper-printed schedule {index} {field}")
        require_close(
            actual["mean_fraction_dimensions_removed"],
            target["mean_fraction_dimensions_removed"],
            f"paper-printed schedule {index} removed fraction",
        )
        if actual["all_boundaries_resolved"] is not True or target["all_boundaries_resolved"] is not True:
            raise RuntimeError(f"paper-printed schedule {index} cuts an unresolved cluster")
        if previous is not None and any(new > old for new, old in zip(actual["rank_tuple"], previous)):
            raise RuntimeError("paper-printed schedules are not nested")
        previous = actual["rank_tuple"]
    return tuple(schedules)


def validate_uniform_dimension_sweep(record: dict, eigensystems) -> tuple[dict, ...]:
    expected = uniform_dimension_rank_sweep(eigensystems)
    if not isinstance(record, dict) or set(record) != set(expected):
        raise RuntimeError("uniform dimension sweep inventory mismatch")
    for key in ("name", "target_removal_grid", "homogeneous_coordinate_convention"):
        if record[key] != expected[key]:
            raise RuntimeError(f"uniform dimension sweep mismatch for {key}")
    schedules = record["schedules"]
    if not isinstance(schedules, list) or len(schedules) != len(REMOVAL_GRID):
        raise RuntimeError("uniform dimension sweep schedule count mismatch")
    expected_keys = {
        "target_fraction_dimensions_removed", "nominal_rank_before_cluster_completion",
        "rank_tuple", "actual_mean_fraction_dimensions_removed", "all_boundaries_resolved",
    }
    for index, (actual, target) in enumerate(zip(schedules, expected["schedules"])):
        if not isinstance(actual, dict) or set(actual) != expected_keys:
            raise RuntimeError(f"uniform dimension schedule {index} inventory mismatch")
        if actual["rank_tuple"] != target["rank_tuple"]:
            raise RuntimeError(f"uniform dimension schedule {index} rank mismatch")
        if actual["nominal_rank_before_cluster_completion"] != target["nominal_rank_before_cluster_completion"]:
            raise RuntimeError(f"uniform dimension schedule {index} nominal rank mismatch")
        if actual["all_boundaries_resolved"] is not True:
            raise RuntimeError(f"uniform dimension schedule {index} cuts an unresolved cluster")
        for key in ("target_fraction_dimensions_removed", "actual_mean_fraction_dimensions_removed"):
            require_close(actual[key], target[key], f"uniform dimension schedule {index} {key}")
    return tuple(schedules)


def validate_odt_record(
    run_root: Path, seed: int, *, device: str = "cuda"
) -> tuple[
    dict,
    HomogeneousChiTN,
    tuple,
    tuple[dict, ...],
    tuple[dict, ...],
    tuple[dict, ...],
    dict[str, np.ndarray],
]:
    odt_path = run_root / "results" / f"paper_seed_{seed}_odt.json"
    physical_file(odt_path, run_root, "ODT JSON")
    odt = strict_json(odt_path)
    expected_top_keys = {
        "schema", "stage", "seed", "claim_boundary", "calibrated_checkpoint",
        "consumed_calibration_json",
        "matrix_artifact", "bond_dimensions", "isometry_errors", "symmetry_errors",
        "factorization_errors", "full_coefficient_norm_squared", "trace_totals",
        "maximum_trace_total_relative_disagreement",
        "canonicalization_coefficient_cancellation_diagnostic",
        "full_rank_coefficient_cancellation_diagnostic",
        "full_rank_basis_orthogonality_errors", "synthetic_logit_replay",
        "rank_allocation", "paper_printed_rank_allocation", "uniform_dimension_sweep",
        "elapsed_seconds", "peak_cuda_memory_bytes",
        "implementation_assumptions", "source_manifest_sha256", "runtime",
    }
    if set(odt) != expected_top_keys:
        raise RuntimeError("ODT JSON top-level inventory mismatch")
    if odt["schema"] != SCHEMA or odt["stage"] != "odt" or type(odt["seed"]) is not int or odt["seed"] != seed:
        raise RuntimeError("ODT JSON identity mismatch")
    if odt["source_manifest_sha256"] != sha256(run_root / "source_manifest.sha256"):
        raise RuntimeError("ODT JSON source binding mismatch")
    if odt["implementation_assumptions"] != ASSUMPTIONS or odt["bond_dimensions"] != [257] * 4:
        raise RuntimeError("ODT JSON architecture assumptions mismatch")
    calibrated_path = run_root / "results" / f"paper_seed_{seed}_calibrated.pt"
    calibration_json_path = run_root / "results" / f"paper_seed_{seed}_calibrate.json"
    physical_file(calibrated_path, run_root, "calibrated checkpoint")
    physical_file(calibration_json_path, run_root, "calibration JSON")
    if odt["calibrated_checkpoint"] != artifact(calibrated_path):
        raise RuntimeError("ODT JSON calibrated-checkpoint binding mismatch")
    if odt["consumed_calibration_json"] != artifact(calibration_json_path):
        raise RuntimeError("ODT JSON calibration-result binding mismatch")
    matrix_path = run_root / "results" / f"paper_seed_{seed}_odt_matrices.npz"
    canonical, systems, arrays = validate_matrix_bundle(
        run_root, matrix_path, odt["matrix_artifact"], device=device
    )
    full_norm_squared = finite_real(odt["full_coefficient_norm_squared"], "full coefficient norm")
    if full_norm_squared <= 0.0:
        raise RuntimeError("full coefficient norm must be positive")
    traces = [float(values.clamp_min(0).sum()) for values, _vectors in systems]
    if not isinstance(odt["trace_totals"], list) or len(odt["trace_totals"]) != 4:
        raise RuntimeError("ODT trace inventory mismatch")
    for bond, (actual, expected) in enumerate(zip(odt["trace_totals"], traces)):
        require_close(actual, expected, f"bond {bond} trace total")
        require_close(expected, full_norm_squared, f"bond {bond} trace versus norm", relative=1e-8)
    maximum_disagreement = max(abs(value - full_norm_squared) / full_norm_squared for value in traces)
    require_close(
        odt["maximum_trace_total_relative_disagreement"], maximum_disagreement,
        "maximum trace disagreement",
    )
    for label in ("isometry_errors", "symmetry_errors", "factorization_errors", "full_rank_basis_orthogonality_errors"):
        values_record = odt[label]
        expected_length = 3 if label == "symmetry_errors" else 4
        if not isinstance(values_record, list) or len(values_record) != expected_length:
            raise RuntimeError(f"{label} inventory mismatch")
        threshold = 1e-10
        if any(
            not 0.0 <= finite_real(value, label) <= threshold for value in values_record
        ):
            raise RuntimeError(f"{label} gate failed")
    validate_cancellation_diagnostic(
        odt["canonicalization_coefficient_cancellation_diagnostic"], "canonicalization"
    )
    validate_cancellation_diagnostic(
        odt["full_rank_coefficient_cancellation_diagnostic"], "full-rank"
    )
    validate_replay_inventory(
        odt["synthetic_logit_replay"],
        {"module_vs_raw", "raw_vs_canonical", "raw_vs_full_rank"},
        "ODT synthetic",
    )
    schedules = validate_rank_allocation(odt["rank_allocation"], systems, full_norm_squared)
    printed_schedules = validate_paper_printed_rank_allocation(
        odt["paper_printed_rank_allocation"], systems
    )
    uniform_schedules = validate_uniform_dimension_sweep(
        odt["uniform_dimension_sweep"], systems
    )
    return odt, canonical, systems, schedules, printed_schedules, uniform_schedules, arrays


def validate_odt_verification(run_root: Path, launch: dict, seed: int) -> dict:
    path = run_root / "results" / f"paper_seed_{seed}_odt_verification.json"
    record = strict_json(path)
    expected_keys = {
        "schema", "stage", "seed", "odt_json", "odt_matrix_artifact",
        "calibrated_checkpoint", "calibration_json", "validated_bonds",
        "validated_schedule_counts", "source_manifest_sha256", "runtime",
    }
    if (
        not isinstance(record, dict)
        or set(record) != expected_keys
        or record["schema"] != SCHEMA
        or record["stage"] != "odt_verification"
        or record["seed"] != seed
    ):
        raise RuntimeError("ODT verification identity or inventory mismatch")
    expected_artifacts = {
        "odt_json": artifact(run_root / "results" / f"paper_seed_{seed}_odt.json"),
        "odt_matrix_artifact": artifact(
            run_root / "results" / f"paper_seed_{seed}_odt_matrices.npz"
        ),
        "calibrated_checkpoint": artifact(
            run_root / "results" / f"paper_seed_{seed}_calibrated.pt"
        ),
        "calibration_json": artifact(
            run_root / "results" / f"paper_seed_{seed}_calibrate.json"
        ),
    }
    if any(record[key] != value for key, value in expected_artifacts.items()):
        raise RuntimeError("ODT verification artifact chain mismatch")
    if record["validated_bonds"] != 4 or record["validated_schedule_counts"] != [15, 15, 13]:
        raise RuntimeError("ODT verification coverage mismatch")
    if record["source_manifest_sha256"] != sha256(run_root / "source_manifest.sha256"):
        raise RuntimeError("ODT verification source binding mismatch")
    validate_runtime_binding(record["runtime"], launch, "odt", seed)
    return record


def build_evaluation_plan(
    schedules: tuple[dict, ...], printed_schedules: tuple[dict, ...],
    uniform_schedules: tuple[dict, ...],
) -> tuple[dict, ...]:
    if (
        len(schedules) != len(EPSILON_GRID)
        or len(printed_schedules) != len(EPSILON_GRID)
        or len(uniform_schedules) != len(REMOVAL_GRID)
    ):
        raise RuntimeError("evaluation plan requires every schedule in all families")
    rows = [{
        "row_index": 0,
        "kind": "uncompressed_full",
        "family_code": 0,
        "schedule_index": -1,
        "rank_tuple": [257] * 4,
    }]
    for index, schedule in enumerate(printed_schedules):
        rows.append({
            "row_index": index + 1,
            "kind": "paper_printed_gram_frobenius",
            "family_code": 1,
            "schedule_index": index,
            "epsilon": schedule["epsilon"],
            "rank_tuple": list(schedule["rank_tuple"]),
        })
    for index, schedule in enumerate(schedules):
        rows.append({
            "row_index": len(EPSILON_GRID) + index + 1,
            "kind": "corrected_trace_hsvd",
            "family_code": 2,
            "schedule_index": index,
            "epsilon": schedule["epsilon"],
            "rank_tuple": list(schedule["rank_tuple"]),
        })
    for index, schedule in enumerate(uniform_schedules):
        rows.append({
            "row_index": 2 * len(EPSILON_GRID) + index + 1,
            "kind": "uniform_dimension_sweep",
            "family_code": 3,
            "schedule_index": index,
            "target_fraction_dimensions_removed": schedule[
                "target_fraction_dimensions_removed"
            ],
            "rank_tuple": list(schedule["rank_tuple"]),
        })
    return tuple(rows)


def validate_evaluation_record(
    run_root: Path,
    seed: int,
    evaluate: dict,
    schedules: tuple[dict, ...],
    printed_schedules: tuple[dict, ...],
    uniform_schedules: tuple[dict, ...],
    official_labels: np.ndarray,
) -> tuple[list[dict], dict, np.ndarray]:
    expected_top_keys = {
        "schema", "stage", "seed", "official_test_access", "evaluation_rows",
        "uncompressed_full", "paper_printed_curve", "corrected_epsilon_curve",
        "uniform_dimension_curve",
        "prediction_artifact", "odt_json",
        "odt_matrix_artifact", "consumed_odt_verification",
        "paper_reported_accuracy_comparison_only",
        "source_manifest_sha256", "runtime",
    }
    if set(evaluate) != expected_top_keys or evaluate["official_test_access"] is not True:
        raise RuntimeError("evaluation record inventory or test-access flag mismatch")
    if evaluate["paper_reported_accuracy_comparison_only"] != 0.854:
        raise RuntimeError("paper-reported comparison constant mismatch")
    if evaluate["consumed_odt_verification"] != artifact(
        run_root / "results" / f"paper_seed_{seed}_odt_verification.json"
    ):
        raise RuntimeError("evaluation ODT-verification chain mismatch")
    plan = build_evaluation_plan(schedules, printed_schedules, uniform_schedules)
    rows = evaluate["evaluation_rows"]
    printed_curve = evaluate["paper_printed_curve"]
    corrected_curve = evaluate["corrected_epsilon_curve"]
    uniform_curve = evaluate["uniform_dimension_curve"]
    if (
        not isinstance(rows, list) or len(rows) != 44
        or not isinstance(printed_curve, list) or len(printed_curve) != 15
        or not isinstance(corrected_curve, list) or len(corrected_curve) != 15
        or not isinstance(uniform_curve, list) or len(uniform_curve) != len(REMOVAL_GRID)
    ):
        raise RuntimeError("evaluation row count mismatch")
    metric_keys = {"loss_sum", "correct", "count", "accuracy"}
    for index, (row, descriptor) in enumerate(zip(rows, plan)):
        if not isinstance(row, dict) or set(row) != set(descriptor) | metric_keys:
            raise RuntimeError(f"evaluation row {index} inventory mismatch")
        for key, value in descriptor.items():
            if row.get(key) != value:
                raise RuntimeError(f"evaluation row {index} descriptor mismatch for {key}")
    if evaluate["uncompressed_full"] != rows[0]:
        raise RuntimeError("uncompressed evaluation row alias mismatch")
    for family_name, curve, descriptors, family_schedules, family_rows in (
        ("paper-printed", printed_curve, plan[1:16], printed_schedules, rows[1:16]),
        ("corrected", corrected_curve, plan[16:31], schedules, rows[16:31]),
        ("uniform", uniform_curve, plan[31:44], uniform_schedules, rows[31:44]),
    ):
        for index, (curve_row, descriptor, schedule, row) in enumerate(
            zip(curve, descriptors, family_schedules, family_rows)
        ):
            expected_keys = set(descriptor) | set(schedule) | metric_keys
            if not isinstance(curve_row, dict) or set(curve_row) != expected_keys:
                raise RuntimeError(f"{family_name} curve row {index} inventory mismatch")
            expected = {**descriptor, **schedule, **{key: row[key] for key in metric_keys}}
            for key, value in expected.items():
                if curve_row.get(key) != value:
                    raise RuntimeError(f"{family_name} curve row {index} mismatch for {key}")
    prediction_path = run_root / "results" / f"paper_seed_{seed}_test_predictions.npz"
    physical_file(prediction_path, run_root, "prediction artifact")
    declared = evaluate["prediction_artifact"]
    expected_declared_keys = {
        "path", "bytes", "sha256", "allow_pickle", "keys",
        "predictions_sha256", "labels_sha256",
    }
    if not isinstance(declared, dict) or set(declared) != expected_declared_keys or declared["allow_pickle"] is not False:
        raise RuntimeError("prediction artifact metadata inventory mismatch")
    with np.load(prediction_path, allow_pickle=False) as bundle:
        if set(bundle.files) != {
            "predictions", "labels", "family_codes", "schedule_indices", "rank_tuples"
        }:
            raise RuntimeError("prediction NPZ key inventory mismatch")
        predictions = bundle["predictions"].copy()
        labels = bundle["labels"].copy()
        family_codes = bundle["family_codes"].copy()
        schedule_indices = bundle["schedule_indices"].copy()
        rank_tuples = bundle["rank_tuples"].copy()
    expected_shapes = {
        "predictions": (44, EXPECTED_SPLIT_LENGTHS["test"]),
        "labels": (EXPECTED_SPLIT_LENGTHS["test"],),
        "family_codes": (44,),
        "schedule_indices": (44,),
        "rank_tuples": (44, 4),
    }
    arrays = {
        "predictions": predictions,
        "labels": labels,
        "family_codes": family_codes,
        "schedule_indices": schedule_indices,
        "rank_tuples": rank_tuples,
    }
    for key, value in arrays.items():
        if value.shape != expected_shapes[key] or value.dtype != np.dtype("int64") or not value.dtype.isnative:
            raise RuntimeError(f"prediction NPZ metadata mismatch for {key}")
    expected_metadata = {
        key: {"dtype": "int64", "shape": list(expected_shapes[key])}
        for key in sorted(expected_shapes)
    }
    if declared["keys"] != expected_metadata or {
        key: declared[key] for key in ("path", "bytes", "sha256")
    } != artifact(prediction_path):
        raise RuntimeError("evaluation JSON does not bind the exact prediction artifact")
    if np.any(predictions < 0) or np.any(predictions > 9) or np.any(labels < 0) or np.any(labels > 9):
        raise RuntimeError("prediction artifact contains labels outside [0,9]")
    if not np.array_equal(labels, official_labels.astype(np.int64, copy=False)):
        raise RuntimeError("prediction labels do not equal the official SVHN test labels")
    expected_family_codes = np.asarray(
        [0] + [1] * 15 + [2] * 15 + [3] * len(REMOVAL_GRID), dtype=np.int64
    )
    expected_indices = np.asarray(
        [-1] + list(range(15)) + list(range(15)) + list(range(len(REMOVAL_GRID))),
        dtype=np.int64,
    )
    expected_ranks = np.asarray([row["rank_tuple"] for row in plan], dtype=np.int64)
    if (
        not np.array_equal(family_codes, expected_family_codes)
        or not np.array_equal(schedule_indices, expected_indices)
        or not np.array_equal(rank_tuples, expected_ranks)
    ):
        raise RuntimeError("prediction row-to-schedule mapping mismatch")
    if declared["predictions_sha256"] != hashlib.sha256(predictions.tobytes()).hexdigest():
        raise RuntimeError("prediction payload digest mismatch")
    if declared["labels_sha256"] != hashlib.sha256(labels.tobytes()).hexdigest():
        raise RuntimeError("prediction label digest mismatch")
    correct = (predictions == labels[None, :]).sum(axis=1)
    derived = []
    for index, (row, count) in enumerate(zip(rows, correct)):
        if type(row["correct"]) is not int or row["correct"] != int(count):
            raise RuntimeError(f"evaluation row {index} correct-count mismatch")
        if type(row["count"]) is not int or row["count"] != EXPECTED_SPLIT_LENGTHS["test"]:
            raise RuntimeError(f"evaluation row {index} example-count mismatch")
        accuracy = float(count / EXPECTED_SPLIT_LENGTHS["test"])
        if type(row["accuracy"]) is not float or row["accuracy"] != accuracy:
            raise RuntimeError(f"evaluation row {index} accuracy mismatch")
        loss_sum = finite_real(row["loss_sum"], f"evaluation row {index} loss")
        if loss_sum < 0.0:
            raise RuntimeError(f"evaluation row {index} loss is negative")
        derived.append({**plan[index], "correct": int(count), "count": len(labels), "accuracy": accuracy})
    return derived, {"path": str(prediction_path), "sha256": sha256(prediction_path)}, predictions


def stage_odt(run_root: Path, data_root: Path, seed: int) -> None:
    validate_source_manifest(run_root)
    validate_unit_certificate(run_root)
    validate_dataset_files(data_root)
    validate_array_seed(seed)
    launch = validate_current_stage_launch(run_root, data_root, "odt", seed)
    current_runtime = runtime_provenance()
    validate_runtime_binding(current_runtime, launch, "odt", seed)
    dataset_certificate = validate_dataset_certificate(run_root, data_root, launch)
    configure_determinism(seed)
    calibrated = run_root / "results" / f"paper_seed_{seed}_calibrated.pt"
    calibration_json = run_root / "results" / f"paper_seed_{seed}_calibrate.json"
    _train_record, calibration_record = validate_calibration_chain(
        run_root, data_root, launch, seed, dataset_certificate, current_runtime
    )
    calibrated_identity = capture_artifact(calibrated, run_root, "calibrated checkpoint")
    calibration_identity = capture_artifact(calibration_json, run_root, "calibration result")
    model = load_checkpoint(calibrated, seed, "calibrated_final").double().eval()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    raw = export_homogeneous_network(model, require_frozen_rbn=True)
    canonical = canonicalize_homogeneous(raw)
    global_grams = canonical_environments(canonical.network)
    local_grams = local_environments(canonical.network)
    systems = tuple(sorted_eigensystem(gram) for gram in global_grams)
    canonicalization_cancellation = coefficient_cancellation_audit(raw, canonical.network)
    if max(canonical.isometry_errors) > 1e-10:
        raise RuntimeError("canonical row-isometry gate failed")
    if max(canonical.factorization_errors) > 1e-10:
        raise RuntimeError("canonical RQ factorization gate failed")
    if max(canonical.symmetry_errors, default=0.0) > 1e-10:
        raise RuntimeError("canonical input-leg symmetry gate failed")
    full_norm_squared = float(coefficient_inner_product(canonical.network, canonical.network))
    if not math.isfinite(full_norm_squared) or full_norm_squared <= 0.0:
        raise RuntimeError("canonical coefficient norm is nonfinite or nonpositive")
    trace_totals = [float(values.clamp_min(0).sum()) for values, _ in systems]
    maximum_trace_disagreement = max(
        abs(value - full_norm_squared) / full_norm_squared for value in trace_totals
    )
    if maximum_trace_disagreement > 1e-8:
        raise RuntimeError("Gram traces disagree with the coefficient norm")
    rank_allocation = corrected_rank_allocation(systems, full_norm_squared)
    printed_rank_allocation = paper_printed_rank_allocation(systems)
    dimension_sweep = uniform_dimension_rank_sweep(systems)
    schedules = rank_allocation["schedules"]
    full = compress_homogeneous(
        canonical.network, top_bases(systems, canonical.network.bond_dims)
    )
    full_cancellation = coefficient_cancellation_audit(canonical.network, full)
    full_basis_orthogonality = [
        float(torch.linalg.matrix_norm(vectors.T @ vectors - torch.eye(
            vectors.shape[1], dtype=vectors.dtype, device=vectors.device
        )))
        for _values, vectors in systems
    ]
    if max(full_basis_orthogonality) > 1e-10:
        raise RuntimeError("full-rank Algorithm-3 eigenbasis is not orthogonal")
    replay_sample = torch.linspace(
        0.0, 1.0, 16 * 1024, dtype=torch.float64, device="cuda"
    ).reshape(16, 1, 32, 32)
    with torch.no_grad():
        module_logits = model(replay_sample)[0]
        raw_logits = homogeneous_forward(raw, replay_sample)
        canonical_logits = homogeneous_forward(canonical.network, replay_sample)
        full_logits = homogeneous_forward(full, replay_sample)
    replay_errors = {
        "module_vs_raw": replay_metrics(module_logits, raw_logits),
        "raw_vs_canonical": replay_metrics(raw_logits, canonical_logits),
        "raw_vs_full_rank": replay_metrics(raw_logits, full_logits),
    }
    if any(item["passed"] is not True for item in replay_errors.values()):
        raise RuntimeError(f"ODT logit replay failed: {replay_errors}")
    arrays = network_to_arrays(canonical.network, "canonical")
    arrays["global_grams"] = torch.stack(global_grams).cpu().numpy()
    arrays["local_grams"] = torch.stack(local_grams).cpu().numpy()
    arrays["global_eigenvalues"] = torch.stack([item[0] for item in systems]).cpu().numpy()
    arrays["global_eigenvectors"] = torch.stack([item[1] for item in systems]).cpu().numpy()
    matrix_path = run_root / "results" / f"paper_seed_{seed}_odt_matrices.npz"
    validate_source_manifest(run_root)
    atomic_npz(matrix_path, arrays)
    with np.load(matrix_path, allow_pickle=False) as bundle:
        if set(bundle.files) != set(arrays):
            raise RuntimeError("ODT matrix bundle changed on round trip")
        for key in bundle.files:
            if bundle[key].dtype != np.float64 or not np.isfinite(bundle[key]).all():
                raise RuntimeError(f"invalid ODT matrix entry {key}")
    require_artifact_unchanged(
        calibrated, calibrated_identity, run_root, "calibrated checkpoint"
    )
    require_artifact_unchanged(
        calibration_json, calibration_identity, run_root, "calibration result"
    )
    odt_path = run_root / "results" / f"paper_seed_{seed}_odt.json"
    atomic_json(
        odt_path,
        {
            "schema": SCHEMA,
            "stage": "odt",
            "seed": seed,
            "claim_boundary": (
                "Export, symmetrization, canonical orthogonalization, eigendecomposition, and "
                "full-rank basis absorption are exact in real arithmetic for the post-calibration, "
                "topology-specific independent-clone logit coefficient tree and are numerically "
                "checked at declared float64 tolerances. Truncated networks are approximations. "
                "This is not the training-time "
                "batch-normalized map, symmetric polynomial quotient, post-softmax map, "
                "attention, or a full VLA decomposition."
            ),
            "calibrated_checkpoint": calibrated_identity,
            "consumed_calibration_json": calibration_identity,
            "matrix_artifact": {
                **artifact(matrix_path),
                "allow_pickle": False,
                "keys": {
                    key: {"dtype": str(value.dtype), "shape": list(value.shape)}
                    for key, value in sorted(arrays.items())
                },
            },
            "bond_dimensions": list(canonical.network.bond_dims),
            "isometry_errors": list(canonical.isometry_errors),
            "symmetry_errors": list(canonical.symmetry_errors),
            "factorization_errors": list(canonical.factorization_errors),
            "full_coefficient_norm_squared": full_norm_squared,
            "trace_totals": trace_totals,
            "maximum_trace_total_relative_disagreement": maximum_trace_disagreement,
            "canonicalization_coefficient_cancellation_diagnostic": canonicalization_cancellation,
            "full_rank_coefficient_cancellation_diagnostic": full_cancellation,
            "full_rank_basis_orthogonality_errors": full_basis_orthogonality,
            "synthetic_logit_replay": replay_errors,
            "rank_allocation": rank_allocation,
            "paper_printed_rank_allocation": printed_rank_allocation,
            "uniform_dimension_sweep": dimension_sweep,
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "implementation_assumptions": ASSUMPTIONS,
            "source_manifest_sha256": sha256(run_root / "source_manifest.sha256"),
            "runtime": current_runtime,
        },
    )
    validated_odt, _network, validated_systems, corrected, printed, uniform, _bundle = (
        validate_odt_record(run_root, seed)
    )
    if (
        validated_odt["matrix_artifact"] != expected_matrix_metadata(matrix_path, arrays)
        or len(validated_systems) != 4
        or (len(corrected), len(printed), len(uniform)) != (15, 15, len(REMOVAL_GRID))
    ):
        raise RuntimeError("post-serialization ODT validation disagrees with the computed result")
    require_artifact_unchanged(
        calibrated, calibrated_identity, run_root, "calibrated checkpoint"
    )
    require_artifact_unchanged(
        calibration_json, calibration_identity, run_root, "calibration result"
    )
    atomic_json(
        run_root / "results" / f"paper_seed_{seed}_odt_verification.json",
        {
            "schema": SCHEMA,
            "stage": "odt_verification",
            "seed": seed,
            "odt_json": artifact(odt_path),
            "odt_matrix_artifact": artifact(matrix_path),
            "calibrated_checkpoint": calibrated_identity,
            "calibration_json": calibration_identity,
            "validated_bonds": len(validated_systems),
            "validated_schedule_counts": [len(printed), len(corrected), len(uniform)],
            "source_manifest_sha256": sha256(run_root / "source_manifest.sha256"),
            "runtime": current_runtime,
        },
    )


@torch.no_grad()
def evaluate_networks(networks: list[HomogeneousChiTN], dataset) -> tuple[list[dict], np.ndarray]:
    loader = torch.utils.data.DataLoader(
        RawSVHN(dataset, 2), batch_size=512, shuffle=False, num_workers=4, pin_memory=True
    )
    losses = np.zeros(len(networks), dtype=np.float64)
    correct = np.zeros(len(networks), dtype=np.int64)
    predictions = np.empty((len(networks), len(dataset)), dtype=np.int64)
    offset = 0
    for rgb, labels, _split, _indices in loader:
        images = grayscale(rgb.cuda(non_blocking=True)).double()
        labels_gpu = labels.cuda(non_blocking=True)
        for index, network in enumerate(networks):
            logits = homogeneous_forward(network, images)
            losses[index] += float(F.cross_entropy(logits, labels_gpu, reduction="sum"))
            predicted = logits.argmax(-1)
            correct[index] += int((predicted == labels_gpu).sum())
            predictions[index, offset : offset + len(labels)] = predicted.cpu().numpy()
        offset += len(labels)
    if offset != len(dataset):
        raise RuntimeError("test evaluation did not cover the official test split")
    if not np.isfinite(losses).all() or np.any(predictions < 0) or np.any(predictions > 9):
        raise RuntimeError("test evaluation produced invalid losses or predictions")
    records = [
        {
            "loss_sum": float(losses[index]),
            "correct": int(correct[index]),
            "count": len(dataset),
            "accuracy": float(correct[index] / len(dataset)),
        }
        for index in range(len(networks))
    ]
    return records, predictions


def stage_eval(run_root: Path, data_root: Path, seed: int) -> None:
    validate_source_manifest(run_root)
    validate_unit_certificate(run_root)
    validate_dataset_files(data_root)
    validate_array_seed(seed)
    launch = validate_current_stage_launch(run_root, data_root, "eval", seed)
    current_runtime = runtime_provenance()
    validate_runtime_binding(current_runtime, launch, "eval", seed)
    dataset_certificate = validate_dataset_certificate(run_root, data_root, launch)
    configure_determinism(seed)
    odt_path = run_root / "results" / f"paper_seed_{seed}_odt.json"
    matrix_path = run_root / "results" / f"paper_seed_{seed}_odt_matrices.npz"
    odt_verification_path = (
        run_root / "results" / f"paper_seed_{seed}_odt_verification.json"
    )
    validate_calibration_chain(
        run_root, data_root, launch, seed, dataset_certificate, current_runtime
    )
    odt_identity = capture_artifact(odt_path, run_root, "ODT JSON")
    matrix_identity = capture_artifact(matrix_path, run_root, "ODT matrix artifact")
    (
        odt, canonical, systems, schedules, printed_schedules, uniform_schedules, _arrays
    ) = validate_odt_record(run_root, seed)
    validate_runtime_binding(odt["runtime"], launch, "odt", seed)
    validate_software_match(current_runtime, odt["runtime"], "ODT to evaluation")
    odt_verification = validate_odt_verification(run_root, launch, seed)
    odt_verification_identity = capture_artifact(
        odt_verification_path, run_root, "ODT verification"
    )
    validate_software_match(
        current_runtime, odt_verification["runtime"], "ODT verification to evaluation"
    )
    vectors = torch.stack([vectors for _values, vectors in systems])
    evaluation_plan = build_evaluation_plan(schedules, printed_schedules, uniform_schedules)
    networks = [canonical]
    for schedule in printed_schedules + schedules + uniform_schedules:
        bases = tuple(
            vectors[bond, :, :rank]
            for bond, rank in enumerate(schedule["rank_tuple"])
        )
        networks.append(compress_homogeneous(canonical, bases))
    if len(networks) != len(evaluation_plan):
        raise RuntimeError("evaluation network and row-plan counts differ")
    test = tv.datasets.SVHN(data_root, split="test", download=False)
    if len(test) != EXPECTED_SPLIT_LENGTHS["test"]:
        raise RuntimeError("official test split length mismatch")
    labels = np.asarray(test.labels, dtype=np.int64)
    if hashlib.sha256(labels.tobytes()).hexdigest() != dataset_certificate["labels_sha256"]["test"]:
        raise RuntimeError("official test labels differ from the prefetched dataset certificate")
    records, predictions = evaluate_networks(networks, test)
    prediction_path = run_root / "results" / f"paper_seed_{seed}_test_predictions.npz"
    atomic_npz(
        prediction_path,
        {
            "predictions": predictions,
            "labels": labels,
            "family_codes": np.asarray(
                [row["family_code"] for row in evaluation_plan], dtype=np.int64
            ),
            "schedule_indices": np.asarray(
                [row["schedule_index"] for row in evaluation_plan], dtype=np.int64
            ),
            "rank_tuples": np.asarray(
                [row["rank_tuple"] for row in evaluation_plan], dtype=np.int64
            ),
        },
    )
    validate_source_manifest(run_root)
    with np.load(prediction_path, allow_pickle=False) as bundle:
        recomputed = (bundle["predictions"] == bundle["labels"][None, :]).sum(axis=1)
        if not np.array_equal(recomputed, np.asarray([item["correct"] for item in records])):
            raise RuntimeError("prediction artifact does not reproduce correct counts")
        if bundle["predictions"].shape != (44, EXPECTED_SPLIT_LENGTHS["test"]):
            raise RuntimeError("prediction artifact row count mismatch")
        if not np.array_equal(
            bundle["schedule_indices"],
            np.asarray(
                [-1] + list(range(15)) + list(range(15))
                + list(range(len(REMOVAL_GRID))),
                dtype=np.int64,
            ),
        ):
            raise RuntimeError("prediction artifact schedule row mapping mismatch")
    require_artifact_unchanged(odt_path, odt_identity, run_root, "ODT JSON")
    require_artifact_unchanged(matrix_path, matrix_identity, run_root, "ODT matrix artifact")
    require_artifact_unchanged(
        odt_verification_path, odt_verification_identity, run_root, "ODT verification"
    )
    eval_path = run_root / "results" / f"paper_seed_{seed}_eval.json"
    eval_record = {
            "schema": SCHEMA,
            "stage": "eval",
            "seed": seed,
            "official_test_access": True,
            "evaluation_rows": [
                {**row, **metrics}
                for row, metrics in zip(evaluation_plan, records)
            ],
            "uncompressed_full": {**evaluation_plan[0], **records[0]},
            "paper_printed_curve": [
                {**row, **schedule, **metrics}
                for row, schedule, metrics in zip(
                    evaluation_plan[1:16], printed_schedules, records[1:16]
                )
            ],
            "corrected_epsilon_curve": [
                {**row, **schedule, **metrics}
                for row, schedule, metrics in zip(
                    evaluation_plan[16:31], schedules, records[16:31]
                )
            ],
            "uniform_dimension_curve": [
                {**row, **schedule, **metrics}
                for row, schedule, metrics in zip(
                    evaluation_plan[31:44], uniform_schedules, records[31:44]
                )
            ],
            "prediction_artifact": {
                **artifact(prediction_path),
                "allow_pickle": False,
                "keys": {
                    "labels": {"dtype": "int64", "shape": [EXPECTED_SPLIT_LENGTHS["test"]]},
                    "family_codes": {"dtype": "int64", "shape": [44]},
                    "predictions": {"dtype": "int64", "shape": [44, EXPECTED_SPLIT_LENGTHS["test"]]},
                    "rank_tuples": {"dtype": "int64", "shape": [44, 4]},
                    "schedule_indices": {"dtype": "int64", "shape": [44]},
                },
                "predictions_sha256": hashlib.sha256(predictions.tobytes()).hexdigest(),
                "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            },
            "odt_json": odt_identity,
            "odt_matrix_artifact": matrix_identity,
            "consumed_odt_verification": odt_verification_identity,
            "paper_reported_accuracy_comparison_only": 0.854,
            "source_manifest_sha256": sha256(run_root / "source_manifest.sha256"),
            "runtime": current_runtime,
        }
    atomic_json(eval_path, eval_record)
    validated_rows, prediction_identity, validated_predictions = validate_evaluation_record(
        run_root, seed, strict_json(eval_path), schedules, printed_schedules,
        uniform_schedules, labels,
    )
    if len(validated_rows) != 44 or not np.array_equal(validated_predictions, predictions):
        raise RuntimeError("post-serialization evaluation validation disagrees with inference")


def stage_verify_eval(run_root: Path, data_root: Path, seed: int) -> None:
    """Independently reconstruct and re-evaluate every stored truncation row."""

    validate_source_manifest(run_root)
    validate_unit_certificate(run_root)
    dataset_files_before = validate_dataset_files(data_root)
    validate_array_seed(seed)
    launch = validate_current_stage_launch(run_root, data_root, "verify_eval", seed)
    current_runtime = runtime_provenance()
    validate_runtime_binding(current_runtime, launch, "verify_eval", seed)
    dataset_certificate = validate_dataset_certificate(run_root, data_root, launch)
    configure_determinism(seed)
    paths = {
        "evaluation_json": run_root / "results" / f"paper_seed_{seed}_eval.json",
        "prediction_artifact": (
            run_root / "results" / f"paper_seed_{seed}_test_predictions.npz"
        ),
        "odt_json": run_root / "results" / f"paper_seed_{seed}_odt.json",
        "odt_matrix_artifact": (
            run_root / "results" / f"paper_seed_{seed}_odt_matrices.npz"
        ),
        "consumed_odt_verification": (
            run_root / "results" / f"paper_seed_{seed}_odt_verification.json"
        ),
    }
    identities = {
        key: capture_artifact(path, run_root, key.replace("_", " "))
        for key, path in paths.items()
    }
    validate_calibration_chain(
        run_root, data_root, launch, seed, dataset_certificate, current_runtime
    )
    odt, canonical, systems, schedules, printed, uniform, _arrays = validate_odt_record(
        run_root, seed
    )
    odt_verification = validate_odt_verification(run_root, launch, seed)
    evaluate = strict_json(paths["evaluation_json"])
    validate_record_identity(evaluate, run_root, launch, "eval", seed)
    validate_software_match(current_runtime, odt["runtime"], "ODT to evaluation verification")
    validate_software_match(
        current_runtime, odt_verification["runtime"], "ODT verifier to evaluation verification"
    )
    validate_software_match(
        current_runtime, evaluate["runtime"], "evaluation to independent verification"
    )
    test = tv.datasets.SVHN(data_root, split="test", download=False)
    labels = np.asarray(test.labels, dtype=np.int64)
    if hashlib.sha256(labels.tobytes()).hexdigest() != dataset_certificate["labels_sha256"]["test"]:
        raise RuntimeError("verification labels differ from the dataset certificate")
    derived_rows, _prediction_identity, stored_predictions = validate_evaluation_record(
        run_root, seed, evaluate, schedules, printed, uniform, labels
    )
    vectors = torch.stack([bond_vectors for _values, bond_vectors in systems])
    networks = [canonical]
    for schedule in printed + schedules + uniform:
        bases = tuple(
            vectors[bond, :, :rank]
            for bond, rank in enumerate(schedule["rank_tuple"])
        )
        networks.append(compress_homogeneous(canonical, bases))
    fresh_metrics, fresh_predictions = evaluate_networks(networks, test)
    if not np.array_equal(fresh_predictions, stored_predictions):
        raise RuntimeError("independent evaluation predictions differ from the stored artifact")
    for index, (fresh, stored) in enumerate(zip(fresh_metrics, evaluate["evaluation_rows"])):
        require_close(fresh["loss_sum"], stored["loss_sum"], f"verification row {index} loss")
        if any(fresh[key] != stored[key] for key in ("correct", "count", "accuracy")):
            raise RuntimeError(f"independent evaluation metric mismatch at row {index}")
    if len(derived_rows) != 44 or len(fresh_metrics) != 44:
        raise RuntimeError("independent evaluation did not cover all 44 rows")
    for key, path in paths.items():
        require_artifact_unchanged(path, identities[key], run_root, key.replace("_", " "))
    validate_source_manifest(run_root)
    if validate_dataset_files(data_root) != dataset_files_before:
        raise RuntimeError("dataset files changed during independent evaluation verification")
    atomic_json(
        run_root / "results" / f"paper_seed_{seed}_eval_verification.json",
        {
            "schema": SCHEMA,
            "stage": "eval_verification",
            "seed": seed,
            **identities,
            "validated_rows": 44,
            "fresh_prediction_exact_match": True,
            "fresh_discrete_metric_exact_match": True,
            "fresh_loss_match_tolerance": {"relative": 1e-9, "absolute": 1e-12},
            "source_manifest_sha256": sha256(run_root / "source_manifest.sha256"),
            "runtime": current_runtime,
        },
    )


def stage_feasibility(run_root: Path, data_root: Path) -> None:
    validate_source_manifest(run_root)
    validate_unit_certificate(run_root)
    validate_dataset_files(data_root)
    launch = validate_current_stage_launch(run_root, data_root, "feasibility", None)
    current_runtime = runtime_provenance()
    validate_runtime_binding(current_runtime, launch, "feasibility", None)
    dataset_certificate = validate_dataset_certificate(run_root, data_root, launch)
    dataset_path = run_root / "results" / "dataset_certificate.json"
    dataset_identity = capture_artifact(dataset_path, run_root, "dataset certificate")
    validate_software_match(
        current_runtime, dataset_certificate["runtime"], "prefetch to feasibility"
    )
    configure_determinism(9173)
    model = exact_model().double().cuda().eval()
    for norm in model.norms:
        norm.initialized.fill_(True)
        norm.freeze()
    sample = torch.linspace(0.0, 1.0, 16 * 1024, dtype=torch.float64, device="cuda").reshape(16, 1, 32, 32)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    raw = export_homogeneous_network(model, require_frozen_rbn=True)
    canonical = canonicalize_homogeneous(raw)
    grams = canonical_environments(canonical.network)
    local_grams = local_environments(canonical.network)
    systems = tuple(sorted_eigensystem(gram) for gram in grams)
    full = compress_homogeneous(canonical.network, top_bases(systems, canonical.network.bond_dims))
    reduced_ranks = tuple(dimension - 1 for dimension in canonical.network.bond_dims)
    reduced = compress_homogeneous(canonical.network, top_bases(systems, reduced_ranks))
    with torch.no_grad():
        raw_logits = homogeneous_forward(raw, sample)
        canonical_logits = homogeneous_forward(canonical.network, sample)
        full_logits = homogeneous_forward(full, sample)
        reduced_logits = homogeneous_forward(reduced, sample)
    replay = {
        "raw_vs_canonical": replay_metrics(raw_logits, canonical_logits),
        "raw_vs_full_rank": replay_metrics(raw_logits, full_logits),
    }
    if any(item["passed"] is not True for item in replay.values()):
        raise RuntimeError(f"width-257 feasibility replay failed: {replay}")
    reduced_logit_norm = float(torch.linalg.vector_norm(reduced_logits))
    if not math.isfinite(reduced_logit_norm):
        raise RuntimeError("reduced feasibility logits are nonfinite")
    if (
        max(canonical.isometry_errors) > 1e-10
        or max(canonical.factorization_errors) > 1e-10
        or max(canonical.symmetry_errors, default=0.0) > 1e-10
    ):
        raise RuntimeError("width-257 feasibility canonicalization gate failed")
    cancellation_diagnostics = {
        "raw_vs_canonical": coefficient_cancellation_audit(raw, canonical.network),
        "canonical_vs_full_rank": coefficient_cancellation_audit(canonical.network, full),
    }
    full_norm_squared = float(coefficient_inner_product(canonical.network, canonical.network))
    if not math.isfinite(full_norm_squared) or full_norm_squared <= 0.0:
        raise RuntimeError("feasibility coefficient norm is invalid")
    allocation = corrected_rank_allocation(systems, full_norm_squared)
    validate_rank_allocation(allocation, systems, full_norm_squared)
    path = run_root / "results" / "feasibility_matrices.npz"
    arrays = network_to_arrays(canonical.network, "canonical")
    arrays["global_grams"] = torch.stack(grams).cpu().numpy()
    arrays["local_grams"] = torch.stack(local_grams).cpu().numpy()
    arrays["global_eigenvalues"] = torch.stack([item[0] for item in systems]).cpu().numpy()
    arrays["global_eigenvectors"] = torch.stack([item[1] for item in systems]).cpu().numpy()
    atomic_npz(path, arrays)
    matrix_metadata = expected_matrix_metadata(path, arrays)
    validate_matrix_bundle(run_root, path, matrix_metadata)
    validate_source_manifest(run_root)
    require_artifact_unchanged(
        dataset_path, dataset_identity, run_root, "dataset certificate"
    )
    atomic_json(
        run_root / "results" / "feasibility.json",
        {
            "schema": SCHEMA,
            "stage": "feasibility_only",
            "dataset_certificate": dataset_identity,
            "uses_test_data": False,
            "parameter_count": model.num_params(),
            "bond_dimensions": list(canonical.network.bond_dims),
            "full_replay": replay,
            "coefficient_cancellation_diagnostics": cancellation_diagnostics,
            "full_coefficient_norm_squared": full_norm_squared,
            "rank_allocation_validated": True,
            "reduced_rank_tuple": list(reduced_ranks),
            "reduced_logit_norm_diagnostic": reduced_logit_norm,
            "isometry_errors": list(canonical.isometry_errors),
            "symmetry_errors": list(canonical.symmetry_errors),
            "factorization_errors": list(canonical.factorization_errors),
            "matrix_artifact": matrix_metadata,
            "production_width_cancellation_and_serialization_paths_exercised": True,
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "source_manifest_sha256": sha256(run_root / "source_manifest.sha256"),
            "runtime": current_runtime,
        },
    )


def validate_feasibility_record(run_root: Path, launch: dict) -> dict:
    record_path = run_root / "results" / "feasibility.json"
    matrix_path = run_root / "results" / "feasibility_matrices.npz"
    physical_file(record_path, run_root, "feasibility record")
    physical_file(matrix_path, run_root, "feasibility matrix artifact")
    record = strict_json(record_path)
    expected_keys = {
        "schema", "stage", "dataset_certificate", "uses_test_data", "parameter_count", "bond_dimensions",
        "full_replay", "coefficient_cancellation_diagnostics", "full_coefficient_norm_squared",
        "rank_allocation_validated", "reduced_rank_tuple", "reduced_logit_norm_diagnostic",
        "isometry_errors", "symmetry_errors", "factorization_errors", "matrix_artifact",
        "production_width_cancellation_and_serialization_paths_exercised",
        "elapsed_seconds", "peak_cuda_memory_bytes",
        "source_manifest_sha256", "runtime",
    }
    if set(record) != expected_keys or record["schema"] != SCHEMA or record["stage"] != "feasibility_only":
        raise RuntimeError("feasibility record identity or inventory mismatch")
    if record["uses_test_data"] is not False or record["parameter_count"] != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("feasibility record scope mismatch")
    if (
        record["rank_allocation_validated"] is not True
        or record["production_width_cancellation_and_serialization_paths_exercised"] is not True
    ):
        raise RuntimeError("feasibility did not exercise the production algebraic path")
    if record["bond_dimensions"] != [257] * 4 or record["reduced_rank_tuple"] != [256] * 4:
        raise RuntimeError("feasibility bond dimensions mismatch")
    if record["source_manifest_sha256"] != sha256(run_root / "source_manifest.sha256"):
        raise RuntimeError("feasibility source binding mismatch")
    dataset_path = run_root / "results" / "dataset_certificate.json"
    if record["dataset_certificate"] != artifact(dataset_path):
        raise RuntimeError("feasibility dataset-certificate chain mismatch")
    validate_runtime_binding(record["runtime"], launch, "feasibility", None)
    dataset_certificate = strict_json(dataset_path)
    validate_software_match(
        record["runtime"], dataset_certificate["runtime"], "prefetch to feasibility record"
    )
    validate_replay_inventory(
        record["full_replay"], {"raw_vs_canonical", "raw_vs_full_rank"}, "feasibility"
    )
    for label, length in (("isometry_errors", 4), ("symmetry_errors", 3), ("factorization_errors", 4)):
        values = record[label]
        if not isinstance(values, list) or len(values) != length or any(
            not 0.0 <= finite_real(value, label) <= 1e-10 for value in values
        ):
            raise RuntimeError(f"feasibility {label} gate failed")
    if finite_real(record["reduced_logit_norm_diagnostic"], "reduced logit norm") < 0.0:
        raise RuntimeError("feasibility reduced logit norm is negative")
    validate_matrix_bundle(run_root, matrix_path, record["matrix_artifact"])
    if finite_real(record["full_coefficient_norm_squared"], "feasibility coefficient norm") <= 0.0:
        raise RuntimeError("feasibility coefficient norm is nonpositive")
    diagnostics = record["coefficient_cancellation_diagnostics"]
    if not isinstance(diagnostics, dict) or set(diagnostics) != {
        "raw_vs_canonical", "canonical_vs_full_rank"
    }:
        raise RuntimeError("feasibility cancellation diagnostic inventory mismatch")
    for name, diagnostic in diagnostics.items():
        validate_cancellation_diagnostic(diagnostic, f"feasibility {name}")
    return record


def expected_result_inventory() -> set[str]:
    names = {
        "launch.json", "unit_tests.log", "unit_certificate.json", "dataset_certificate.json",
        "feasibility.json", "feasibility_matrices.npz",
    }
    for seed in SEEDS:
        names.update(
            {
                f"paper_seed_{seed}_raw.pt",
                f"paper_seed_{seed}_train.json",
                f"paper_seed_{seed}_calibrated.pt",
                f"paper_seed_{seed}_calibrate.json",
                f"paper_seed_{seed}_odt.json",
                f"paper_seed_{seed}_odt_matrices.npz",
                f"paper_seed_{seed}_odt_verification.json",
                f"paper_seed_{seed}_eval.json",
                f"paper_seed_{seed}_test_predictions.npz",
                f"paper_seed_{seed}_eval_verification.json",
            }
        )
    return names


def validate_training_protocol_record(train: dict, seed: int, dataset_files: dict) -> None:
    expected_keys = {
        "schema", "stage", "seed", "paper", "reported_configuration",
        "implementation_assumptions", "architecture_schema", "parameter_count",
        "dataset_examples", "steps_per_epoch", "total_steps", "noise_seed",
        "learning_rate_trace_sha256", "learning_rate_first",
        "learning_rate_last_before_step", "learning_rate_after_final_step", "epochs",
        "elapsed_seconds", "checkpoint", "dataset_certificate", "consumed_feasibility_json",
        "consumed_feasibility_matrix", "dataset_files", "source_manifest_sha256", "runtime",
    }
    if not isinstance(train, dict) or set(train) != expected_keys:
        raise RuntimeError("training record field inventory mismatch")
    expected_paper = {
        "arxiv_id": PAPER_ARXIV_ID,
        "pdf_sha256": PAPER_PDF_SHA256,
        "reported_test_accuracy": 0.854,
    }
    expected_configuration = {
        "training_splits": ["train", "extra"],
        "grayscale": True,
        "input_noise_norm": 0.3,
        "width": 256,
        "bilinear_layers": 3,
        "batch_size": TRAIN_BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 1.0,
        "schedule": "cosine",
        "epochs": 20,
    }
    dataset_examples = EXPECTED_SPLIT_LENGTHS["train"] + EXPECTED_SPLIT_LENGTHS["extra"]
    steps_per_epoch = math.ceil(dataset_examples / TRAIN_BATCH_SIZE)
    total_steps = 20 * steps_per_epoch
    if (
        train["paper"] != expected_paper
        or train["reported_configuration"] != expected_configuration
        or train["implementation_assumptions"] != ASSUMPTIONS
        or train["architecture_schema"] != PAPER_REPORTED_ARCHITECTURE_SCHEMA
        or train["parameter_count"] != EXPECTED_PARAMETER_COUNT
        or train["dataset_examples"] != dataset_examples
        or train["steps_per_epoch"] != steps_per_epoch
        or train["total_steps"] != total_steps
        or train["noise_seed"] != 1_000_003 + seed
        or train["dataset_files"] != dataset_files
    ):
        raise RuntimeError(f"seed {seed} frozen training protocol mismatch")
    digest = train["learning_rate_trace_sha256"]
    expected_trace = expected_cosine_learning_rate_trace(total_steps)
    expected_digest = hashlib.sha256(expected_trace.tobytes()).hexdigest()
    if digest != expected_digest:
        raise RuntimeError(f"seed {seed} learning-rate trace digest mismatch")
    require_close(train["learning_rate_first"], 0.001, "first learning rate", relative=0.0)
    expected_last = 0.001 * (1.0 + math.cos(math.pi * (total_steps - 1) / total_steps)) / 2.0
    require_close(train["learning_rate_last_before_step"], expected_last, "last learning rate", relative=1e-7)
    require_close(train["learning_rate_after_final_step"], 0.0, "post-schedule learning rate", relative=0.0)
    epochs = train["epochs"]
    if not isinstance(epochs, list) or len(epochs) != 20:
        raise RuntimeError(f"seed {seed} epoch inventory mismatch")
    epoch_keys = {"epoch", "loss", "accuracy", "examples", "steps", "last_learning_rate"}
    for epoch_index, epoch in enumerate(epochs):
        if not isinstance(epoch, dict) or set(epoch) != epoch_keys:
            raise RuntimeError(f"seed {seed} epoch {epoch_index} inventory mismatch")
        if (
            type(epoch["epoch"]) is not int
            or epoch["epoch"] != epoch_index
            or type(epoch["examples"]) is not int
            or epoch["examples"] != dataset_examples
            or type(epoch["steps"]) is not int
            or epoch["steps"] != steps_per_epoch
        ):
            raise RuntimeError(f"seed {seed} epoch {epoch_index} count mismatch")
        loss = finite_real(epoch["loss"], "training loss")
        accuracy = finite_real(epoch["accuracy"], "training accuracy")
        learning_rate = finite_real(epoch["last_learning_rate"], "epoch learning rate")
        if loss < 0.0 or not 0.0 <= accuracy <= 1.0 or not 0.0 <= learning_rate <= 0.001:
            raise RuntimeError(f"seed {seed} epoch {epoch_index} metric out of range")
        epoch_last_step = (epoch_index + 1) * steps_per_epoch - 1
        expected_epoch_lr = 0.001 * (
            1.0 + math.cos(math.pi * epoch_last_step / total_steps)
        ) / 2.0
        require_close(learning_rate, expected_epoch_lr, "epoch terminal learning rate", relative=1e-7)


def validate_training_chain(
    run_root: Path,
    data_root: Path,
    launch: dict,
    seed: int,
    dataset_certificate: dict,
    consumer_runtime: dict,
) -> dict:
    """Validate the full immutable ancestry of a raw training checkpoint."""

    train_path = run_root / "results" / f"paper_seed_{seed}_train.json"
    raw_path = run_root / "results" / f"paper_seed_{seed}_raw.pt"
    train = strict_json(train_path)
    validate_record_identity(train, run_root, launch, "train", seed)
    validate_training_protocol_record(train, seed, validate_dataset_files(data_root))
    if train["checkpoint"] != artifact(raw_path):
        raise RuntimeError("training record raw-checkpoint chain mismatch")
    if train["dataset_certificate"] != artifact(
        run_root / "results" / "dataset_certificate.json"
    ):
        raise RuntimeError("training record dataset-certificate chain mismatch")
    if train["consumed_feasibility_json"] != artifact(
        run_root / "results" / "feasibility.json"
    ):
        raise RuntimeError("training record feasibility-JSON chain mismatch")
    if train["consumed_feasibility_matrix"] != artifact(
        run_root / "results" / "feasibility_matrices.npz"
    ):
        raise RuntimeError("training record feasibility-matrix chain mismatch")
    if dataset_certificate["dataset_files"] != train["dataset_files"]:
        raise RuntimeError("training record dataset bytes differ from the dataset certificate")
    validate_software_match(consumer_runtime, train["runtime"], "train to consumer")
    return train


def validate_calibration_protocol_record(calibrate: dict, dataset_certificate: dict) -> None:
    expected_keys = {
        "schema", "stage", "seed", "raw_checkpoint", "calibrated_checkpoint",
        "consumed_train_json", "dataset_certificate", "dataset_labels_sha256",
        "calibration_example_identity", "calibration_example_identities_sha256",
        "calibration_examples", "batch_size", "batches", "noise", "sites",
        "strict_folded_replay", "post_save_reload_strict_folded_replay",
        "implementation_assumptions", "source_manifest_sha256", "runtime",
    }
    if not isinstance(calibrate, dict) or set(calibrate) != expected_keys:
        raise RuntimeError("calibration record field inventory mismatch")
    positions = np.arange(CALIBRATION_BATCH_SIZE * CALIBRATION_BATCHES, dtype=np.int64)
    train_length = EXPECTED_SPLIT_LENGTHS["train"]
    identities = np.stack(
        (
            (positions >= train_length).astype(np.int64),
            np.where(positions < train_length, positions, positions - train_length),
        ),
        axis=1,
    )
    if (
        calibrate["dataset_labels_sha256"] != dataset_certificate["labels_sha256"]
        or calibrate["calibration_example_identity"]
        != "ordered (split_id, within_split_index) int64 pairs"
        or calibrate["calibration_example_identities_sha256"]
        != hashlib.sha256(identities.tobytes()).hexdigest()
        or calibrate["calibration_examples"] != len(identities)
        or calibrate["batch_size"] != CALIBRATION_BATCH_SIZE
        or calibrate["batches"] != CALIBRATION_BATCHES
        or calibrate["noise"] is not False
        or calibrate["implementation_assumptions"] != ASSUMPTIONS
    ):
        raise RuntimeError("calibration protocol record mismatch")
    sites = calibrate["sites"]
    if not isinstance(sites, list) or len(sites) != 3:
        raise RuntimeError("calibration site inventory mismatch")
    for index, site in enumerate(sites):
        if not isinstance(site, dict) or set(site) != {"path", "scale", "batches"}:
            raise RuntimeError("calibration site record is malformed")
        if site["path"] != f"norms.{index}" or site["batches"] != CALIBRATION_BATCHES:
            raise RuntimeError("calibration site order or batch count mismatch")
        if finite_real(site["scale"], "calibration scale") <= 0.0:
            raise RuntimeError("calibration scale is nonpositive")
    validate_replay_inventory(
        {"fold": calibrate["strict_folded_replay"]}, {"fold"}, "calibration before save"
    )
    validate_replay_inventory(
        {"fold": calibrate["post_save_reload_strict_folded_replay"]},
        {"fold"},
        "calibration after save",
    )


def validate_calibration_chain(
    run_root: Path,
    data_root: Path,
    launch: dict,
    seed: int,
    dataset_certificate: dict,
    consumer_runtime: dict,
) -> tuple[dict, dict]:
    """Validate train, raw, dataset, calibrated, and software ancestry before ODT."""

    train = validate_training_chain(
        run_root, data_root, launch, seed, dataset_certificate, consumer_runtime
    )
    calibration_path = run_root / "results" / f"paper_seed_{seed}_calibrate.json"
    calibration = strict_json(calibration_path)
    validate_record_identity(calibration, run_root, launch, "calibrate", seed)
    validate_calibration_protocol_record(calibration, dataset_certificate)
    raw_path = run_root / "results" / f"paper_seed_{seed}_raw.pt"
    calibrated_path = run_root / "results" / f"paper_seed_{seed}_calibrated.pt"
    train_path = run_root / "results" / f"paper_seed_{seed}_train.json"
    dataset_path = run_root / "results" / "dataset_certificate.json"
    if calibration["raw_checkpoint"] != artifact(raw_path):
        raise RuntimeError("calibration raw-checkpoint chain mismatch")
    if calibration["consumed_train_json"] != artifact(train_path):
        raise RuntimeError("calibration train-JSON chain mismatch")
    if calibration["calibrated_checkpoint"] != artifact(calibrated_path):
        raise RuntimeError("calibration calibrated-checkpoint chain mismatch")
    if calibration["dataset_certificate"] != artifact(dataset_path):
        raise RuntimeError("calibration dataset-certificate chain mismatch")
    raw_model = load_checkpoint(raw_path, seed, "raw_final_epoch")
    calibrated_model = load_checkpoint(calibrated_path, seed, "calibrated_final")
    validate_calibrated_transition(raw_model, calibrated_model)
    validate_software_match(consumer_runtime, calibration["runtime"], "calibration to consumer")
    return train, calibration


def validate_eval_verification(run_root: Path, launch: dict, seed: int) -> dict:
    path = run_root / "results" / f"paper_seed_{seed}_eval_verification.json"
    record = strict_json(path)
    expected_keys = {
        "schema", "stage", "seed", "evaluation_json", "prediction_artifact",
        "odt_json", "odt_matrix_artifact", "consumed_odt_verification",
        "validated_rows", "fresh_prediction_exact_match",
        "fresh_discrete_metric_exact_match", "fresh_loss_match_tolerance",
        "source_manifest_sha256", "runtime",
    }
    if (
        not isinstance(record, dict)
        or set(record) != expected_keys
        or record["schema"] != SCHEMA
        or record["stage"] != "eval_verification"
        or record["seed"] != seed
    ):
        raise RuntimeError("evaluation verification identity or inventory mismatch")
    expected = {
        "evaluation_json": artifact(run_root / "results" / f"paper_seed_{seed}_eval.json"),
        "prediction_artifact": artifact(
            run_root / "results" / f"paper_seed_{seed}_test_predictions.npz"
        ),
        "odt_json": artifact(run_root / "results" / f"paper_seed_{seed}_odt.json"),
        "odt_matrix_artifact": artifact(
            run_root / "results" / f"paper_seed_{seed}_odt_matrices.npz"
        ),
        "consumed_odt_verification": artifact(
            run_root / "results" / f"paper_seed_{seed}_odt_verification.json"
        ),
    }
    if any(record[key] != value for key, value in expected.items()):
        raise RuntimeError("evaluation verification artifact chain mismatch")
    if (
        record["validated_rows"] != 44
        or record["fresh_prediction_exact_match"] is not True
        or record["fresh_discrete_metric_exact_match"] is not True
        or record["fresh_loss_match_tolerance"] != {"relative": 1e-9, "absolute": 1e-12}
    ):
        raise RuntimeError("evaluation verification coverage mismatch")
    if record["source_manifest_sha256"] != sha256(run_root / "source_manifest.sha256"):
        raise RuntimeError("evaluation verification source binding mismatch")
    validate_runtime_binding(record["runtime"], launch, "verify_eval", seed)
    return record


def stage_summary(run_root: Path, data_root: Path) -> None:
    validate_source_manifest(run_root)
    unit = validate_unit_certificate(run_root)
    dataset_files = validate_dataset_files(data_root)
    launch = validate_current_stage_launch(run_root, data_root, "summary", None)
    current_runtime = runtime_provenance()
    validate_runtime_binding(current_runtime, launch, "summary", None)
    validate_runtime_binding(unit["runtime"], launch, "unit", None)
    dataset_certificate = validate_dataset_certificate(run_root, data_root, launch)
    feasibility_record = validate_feasibility_record(run_root, launch)
    software_fingerprint = runtime_software_fingerprint(unit["runtime"])
    for label, runtime in (
        ("prefetch", dataset_certificate["runtime"]),
        ("feasibility", feasibility_record["runtime"]),
    ):
        if runtime_software_fingerprint(runtime) != software_fingerprint:
            raise RuntimeError(f"{label} software environment differs from the unit environment")
    result_entries = list((run_root / "results").iterdir())
    if any(path.is_symlink() or not path.is_file() for path in result_entries):
        raise RuntimeError("results directory contains a symlink or non-file entry")
    actual_inventory = {path.name for path in result_entries}
    expected_inventory = expected_result_inventory()
    if actual_inventory != expected_inventory:
        raise RuntimeError(
            f"result artifact inventory mismatch: expected={sorted(expected_inventory)}, "
            f"actual={sorted(actual_inventory)}"
        )
    captured_result_inputs = {
        name: capture_artifact(
            run_root / "results" / name, run_root, f"summary input {name}"
        )
        for name in sorted(expected_inventory)
    }
    test = tv.datasets.SVHN(data_root, split="test", download=False)
    official_labels = np.asarray(test.labels, dtype=np.int64)
    if (
        len(test) != EXPECTED_SPLIT_LENGTHS["test"]
        or hashlib.sha256(official_labels.tobytes()).hexdigest()
        != dataset_certificate["labels_sha256"]["test"]
    ):
        raise RuntimeError("summary test labels do not match the dataset certificate")
    records = []
    checkpoint_hashes = set()
    for seed in SEEDS:
        configure_determinism(seed)
        train_path = run_root / "results" / f"paper_seed_{seed}_train.json"
        calibrate_path = run_root / "results" / f"paper_seed_{seed}_calibrate.json"
        odt_path = run_root / "results" / f"paper_seed_{seed}_odt.json"
        eval_path = run_root / "results" / f"paper_seed_{seed}_eval.json"
        train = strict_json(train_path)
        calibrate = strict_json(calibrate_path)
        evaluate = strict_json(eval_path)
        validate_record_identity(train, run_root, launch, "train", seed)
        validate_record_identity(calibrate, run_root, launch, "calibrate", seed)
        validate_record_identity(evaluate, run_root, launch, "eval", seed)
        for stage_name, runtime in (
            ("train", train["runtime"]),
            ("calibrate", calibrate["runtime"]),
            ("eval", evaluate["runtime"]),
        ):
            if runtime_software_fingerprint(runtime) != software_fingerprint:
                raise RuntimeError(f"seed {seed} {stage_name} software environment drifted")
        expected_train_keys = {
            "schema", "stage", "seed", "paper", "reported_configuration",
            "implementation_assumptions", "architecture_schema", "parameter_count",
            "dataset_examples", "steps_per_epoch", "total_steps", "noise_seed",
            "learning_rate_trace_sha256", "learning_rate_first",
            "learning_rate_last_before_step", "learning_rate_after_final_step", "epochs",
            "elapsed_seconds", "checkpoint", "dataset_certificate", "consumed_feasibility_json",
            "consumed_feasibility_matrix", "dataset_files", "source_manifest_sha256", "runtime",
        }
        expected_calibrate_keys = {
            "schema", "stage", "seed", "raw_checkpoint", "calibrated_checkpoint",
            "consumed_train_json",
            "dataset_certificate", "dataset_labels_sha256", "calibration_example_identity",
            "calibration_example_identities_sha256", "calibration_examples", "batch_size",
            "batches", "noise", "sites", "strict_folded_replay",
            "post_save_reload_strict_folded_replay", "implementation_assumptions",
            "source_manifest_sha256", "runtime",
        }
        if set(train) != expected_train_keys or set(calibrate) != expected_calibrate_keys:
            raise RuntimeError(f"seed {seed} train or calibration inventory mismatch")
        validate_training_protocol_record(train, seed, dataset_files)
        validate_calibration_protocol_record(calibrate, dataset_certificate)
        validate_training_chain(
            run_root, data_root, launch, seed, dataset_certificate, current_runtime
        )
        validate_calibration_chain(
            run_root, data_root, launch, seed, dataset_certificate, current_runtime
        )
        raw_path = run_root / "results" / f"paper_seed_{seed}_raw.pt"
        calibrated_path = run_root / "results" / f"paper_seed_{seed}_calibrated.pt"
        matrix_path = run_root / "results" / f"paper_seed_{seed}_odt_matrices.npz"
        prediction_path = run_root / "results" / f"paper_seed_{seed}_test_predictions.npz"
        for path in (raw_path, calibrated_path, matrix_path, prediction_path):
            physical_file(path, run_root, f"seed {seed} artifact")
        if train["checkpoint"] != artifact(raw_path) or calibrate["raw_checkpoint"] != artifact(raw_path):
            raise RuntimeError("raw checkpoint chain mismatch")
        if calibrate["consumed_train_json"] != artifact(train_path):
            raise RuntimeError("calibration train-JSON chain mismatch")
        if calibrate["calibrated_checkpoint"] != artifact(calibrated_path):
            raise RuntimeError("calibrated checkpoint chain mismatch")
        if calibrate["dataset_certificate"] != artifact(run_root / "results" / "dataset_certificate.json"):
            raise RuntimeError("calibration dataset-certificate chain mismatch")
        if calibrate["dataset_labels_sha256"] != dataset_certificate["labels_sha256"]:
            raise RuntimeError("calibration dataset-label binding mismatch")
        raw_model = load_checkpoint(raw_path, seed, "raw_final_epoch")
        checkpoint_hashes.update((sha256(raw_path), sha256(calibrated_path)))
        (
            odt, canonical, systems, schedules, printed_schedules, uniform_schedules, arrays
        ) = validate_odt_record(run_root, seed)
        validate_runtime_binding(odt["runtime"], launch, "odt", seed)
        if runtime_software_fingerprint(odt["runtime"]) != software_fingerprint:
            raise RuntimeError(f"seed {seed} ODT software environment drifted")
        if odt["calibrated_checkpoint"] != calibrate["calibrated_checkpoint"]:
            raise RuntimeError("ODT calibration chain mismatch")
        if odt["consumed_calibration_json"] != artifact(calibrate_path):
            raise RuntimeError("ODT calibration-JSON chain mismatch")
        if evaluate["odt_json"] != artifact(odt_path):
            raise RuntimeError("evaluation ODT JSON chain mismatch")
        if evaluate["odt_matrix_artifact"] != artifact(matrix_path):
            raise RuntimeError("evaluation ODT matrix chain mismatch")
        derived_rows, prediction_identity, stored_predictions = validate_evaluation_record(
            run_root, seed, evaluate, schedules, printed_schedules, uniform_schedules,
            official_labels,
        )
        if evaluate["prediction_artifact"]["sha256"] != prediction_identity["sha256"]:
            raise RuntimeError("evaluation prediction chain mismatch")
        odt_verification = validate_odt_verification(run_root, launch, seed)
        eval_verification = validate_eval_verification(run_root, launch, seed)
        validate_software_match(
            current_runtime, odt_verification["runtime"], "ODT verification to summary"
        )
        validate_software_match(
            current_runtime, eval_verification["runtime"], "evaluation verification to summary"
        )
        recomputed_norm = float(coefficient_inner_product(canonical, canonical))
        require_close(
            recomputed_norm, odt["full_coefficient_norm_squared"],
            f"seed {seed} recomputed coefficient norm", relative=1e-8,
        )
        calibrated_model = load_checkpoint(calibrated_path, seed, "calibrated_final").double().eval()
        validate_calibrated_transition(raw_model, calibrated_model.float())
        calibrated_model = calibrated_model.double()
        for index, (site, norm) in enumerate(zip(calibrate["sites"], calibrated_model.norms)):
            require_close(
                site["scale"], float(norm.scale),
                f"seed {seed} calibrated RBN scale {index}", relative=1e-7,
            )
        raw = export_homogeneous_network(calibrated_model, require_frozen_rbn=True)
        recomputed_canonical = canonicalize_homogeneous(raw)
        recomputed_diagnostics = {
            "isometry_errors": list(recomputed_canonical.isometry_errors),
            "symmetry_errors": list(recomputed_canonical.symmetry_errors),
            "factorization_errors": list(recomputed_canonical.factorization_errors),
        }
        for label, values in recomputed_diagnostics.items():
            if max(values, default=0.0) > 1e-10:
                raise RuntimeError(f"seed {seed} recomputed {label} gate failed")
            if len(values) != len(odt[label]):
                raise RuntimeError(f"seed {seed} recomputed {label} inventory mismatch")
            for actual, expected in zip(values, odt[label]):
                require_close(actual, expected, f"seed {seed} recomputed {label}")
        recomputed_arrays = network_to_arrays(recomputed_canonical.network, "canonical")
        checkpoint_to_matrix_relative_errors = {}
        for key, expected_array in recomputed_arrays.items():
            stored_tensor = torch.from_numpy(arrays[key]).cuda()
            expected_tensor = torch.from_numpy(expected_array).cuda()
            denominator = max(1.0, float(torch.linalg.vector_norm(expected_tensor)))
            disagreement = float(torch.linalg.vector_norm(stored_tensor - expected_tensor)) / denominator
            if not math.isfinite(disagreement) or disagreement > 1e-10:
                raise RuntimeError(f"seed {seed} checkpoint-to-matrix replay failed for {key}")
            checkpoint_to_matrix_relative_errors[key] = disagreement
        sample = torch.linspace(0.0, 1.0, 16 * 1024, dtype=torch.float64, device="cuda").reshape(16, 1, 32, 32)
        full = compress_homogeneous(canonical, top_bases(systems, canonical.bond_dims))
        with torch.no_grad():
            model_logits = calibrated_model(sample)[0]
            canonical_logits = homogeneous_forward(canonical, sample)
            full_logits = homogeneous_forward(full, sample)
        summary_replay = {
            "checkpoint_vs_serialized_canonical": replay_metrics(model_logits, canonical_logits),
            "serialized_canonical_vs_full_rank": replay_metrics(canonical_logits, full_logits),
        }
        if any(item["passed"] is not True for item in summary_replay.values()):
            raise RuntimeError(f"seed {seed} independent summary replay failed")
        records.append(
            {
                "seed": seed,
                "uncompressed_full_accuracy": derived_rows[0]["accuracy"],
                "evaluation_rows_from_authenticated_predictions": derived_rows,
                "checkpoint_to_matrix_relative_errors": checkpoint_to_matrix_relative_errors,
                "recomputed_canonicalization_diagnostics": recomputed_diagnostics,
                "independent_summary_replay": summary_replay,
                "recomputed_coefficient_norm_squared": recomputed_norm,
                "raw_checkpoint_sha256": sha256(raw_path),
                "calibrated_checkpoint_sha256": sha256(calibrated_path),
                "odt_matrix_sha256": sha256(matrix_path),
                "prediction_sha256": sha256(prediction_path),
                "odt_verification_sha256": sha256(
                    run_root / "results" / f"paper_seed_{seed}_odt_verification.json"
                ),
                "evaluation_verification_sha256": sha256(
                    run_root / "results" / f"paper_seed_{seed}_eval_verification.json"
                ),
            }
        )
        del (
            raw_model, calibrated_model, raw, recomputed_canonical, canonical, full, systems, arrays,
            stored_predictions,
        )
        torch.cuda.empty_cache()
    if len(checkpoint_hashes) != 2 * len(SEEDS):
        raise RuntimeError("checkpoint reuse detected across seeds or stages")
    accuracies = np.asarray(
        [item["uncompressed_full_accuracy"] for item in records], dtype=np.float64
    )
    validate_source_manifest(run_root)
    for name, identity in captured_result_inputs.items():
        require_artifact_unchanged(
            run_root / "results" / name, identity, run_root, f"summary input {name}"
        )
    final_entries = list((run_root / "results").iterdir())
    if (
        any(path.is_symlink() or not path.is_file() for path in final_entries)
        or {path.name for path in final_entries} != expected_inventory
    ):
        raise RuntimeError("result artifact inventory changed while summary was running")
    summary = {
        "schema": SCHEMA,
        "stage": "summary",
        "claim_boundary": (
            "A five-seed reproduction attempt of the configuration reported by Dooms et al. "
            "Unspecified details are frozen assumptions, so numerical correspondence is not "
            "checkpoint identity with unpublished author code. Export, symmetrization, canonical "
            "orthogonalization, eigendecomposition, and full-rank basis absorption are exact in "
            "real arithmetic only for the post-calibration topology-specific pre-softmax coefficient "
            "tree and are numerically checked at declared tolerances. Truncations are approximations."
        ),
        "reported_spec_fidelity": {
            "status": "reported_configuration_implemented_with_frozen_assumptions",
            "paper_pdf_sha256_verified": True,
            "unpublished_checkpoint_or_author_code_identity_claimed": False,
        },
        "algebraic_validity": {
            "status": (
                "full_rank_transformations_algebraically_exact_in_real_arithmetic_and_"
                "numerically_checked_at_declared_tolerances_truncations_approximate"
            ),
            "canonical_network_reconstructed_from_float64_artifacts": True,
            "global_and_local_environments_recomputed": True,
            "eigensystem_residuals_and_orthogonality_recomputed": True,
            "trace_tail_rank_schedules_and_hsvd_bounds_recomputed": True,
            "checkpoint_to_canonical_matrices_recomputed": True,
            "full_rank_function_replay_recomputed": True,
            "compressed_schedule_coefficient_errors": (
                "not_directly_measured_nominal_exact_arithmetic_HSVD_estimates_recomputed_from_"
                "float64_spectra_with_a_non_certifying_heuristic_float64_pad"
            ),
            "cancellation_diagnostics": (
                "non_load_bearing_consistency_indicators_not_error_upper_bounds"
            ),
            "truncation_families": [
                "paper_printed_gram_frobenius_without_coefficient_error_guarantee",
                "uniform_dimensions_removed_figure2_axis",
                "trace_hsvd_with_non_certifying_heuristic_float64_pad",
            ],
        },
        "empirical_result": {
            "paper_reported_accuracy": 0.854,
            "accuracy_source": "recomputed_only_from_authenticated_prediction_arrays_and_official_labels",
            "seed_count": len(records),
            "mean_full_test_accuracy": float(accuracies.mean()),
            "sample_standard_deviation": float(accuracies.std(ddof=1)),
            "per_seed": records,
        },
        "implementation_assumptions": ASSUMPTIONS,
        "dataset_files": dataset_files,
        "consumed_result_artifacts": captured_result_inputs,
        "launch": artifact(run_root / "results" / "launch.json"),
        "source_manifest_sha256": sha256(run_root / "source_manifest.sha256"),
        "runtime": current_runtime,
    }
    atomic_json(run_root / "results" / "summary.json", summary)


def stage_summary_with_failure_receipt(run_root: Path, data_root: Path) -> None:
    """Run aggregation and leave a terminal, immutable receipt on any failure."""

    try:
        stage_summary(run_root, data_root)
    except Exception as error:
        failure_path = run_root / "results" / "summary_failure.json"
        try:
            try:
                manifest_digest = sha256(run_root / "source_manifest.sha256")
            except Exception:
                manifest_digest = None
            try:
                failure_runtime = runtime_provenance()
            except Exception:
                failure_runtime = None
            if not failure_path.exists() and not failure_path.is_symlink():
                atomic_json(
                    failure_path,
                    {
                        "schema": SCHEMA,
                        "stage": "summary_failure",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "source_manifest_sha256": manifest_digest,
                        "runtime": failure_runtime,
                    },
                )
        except Exception:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "unit", "prefetch", "feasibility", "train", "calibrate", "odt", "eval",
            "verify_eval", "summary",
        ),
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--test-log", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve(strict=True)
    data_root = args.data_root.resolve(strict=True)
    if args.stage in {"train", "calibrate", "odt", "eval", "verify_eval"}:
        if args.seed not in SEEDS:
            parser.error(f"--seed must be one of {SEEDS} for stage {args.stage}")
    elif args.seed is not None:
        parser.error(f"--seed is not valid for stage {args.stage}")
    if args.stage == "unit":
        if args.test_log is None:
            parser.error("--test-log is required for the unit stage")
        stage_unit(run_root, args.test_log)
    elif args.test_log is not None:
        parser.error("--test-log is only valid for the unit stage")
    elif args.stage == "prefetch":
        stage_prefetch(run_root, data_root)
    elif args.stage == "feasibility":
        stage_feasibility(run_root, data_root)
    elif args.stage == "train":
        stage_train(run_root, data_root, args.seed)
    elif args.stage == "calibrate":
        stage_calibrate(run_root, data_root, args.seed)
    elif args.stage == "odt":
        stage_odt(run_root, data_root, args.seed)
    elif args.stage == "eval":
        stage_eval(run_root, data_root, args.seed)
    elif args.stage == "verify_eval":
        stage_verify_eval(run_root, data_root, args.seed)
    else:
        stage_summary_with_failure_receipt(run_root, data_root)


if __name__ == "__main__":
    main()
