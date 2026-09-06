#!/usr/bin/env python3
"""Certify and compress the official-capable linear ChiVLA with direct ODT."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import secrets
import stat
import sys
import time
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STAGE_ROOT = Path("/work/joy/x-vla-capable-linear-b1c0-odt-v2")
sys.path[:] = [str(PROJECT_ROOT)] + [
    entry for entry in sys.path if entry != str(PROJECT_ROOT)
]
sys.dont_write_bytecode = True

SOURCE_MANIFEST = PROJECT_ROOT / "athena/capable_linear_direct_odt_sources.sha256"
FRESH_CAPABILITY_SOURCE_MANIFEST = (
    PROJECT_ROOT / "athena/capable_linear_fresh_capability_sources.sha256"
)
STAGE_LEDGER = PROJECT_ROOT / "athena/capable_linear_b1c0_stage.sha256"
CHECKPOINT = PROJECT_ROOT / "inputs/capable_linear_b1c0_checkpoint.pt"
TRAINING_RESULT = PROJECT_ROOT / "inputs/capable_linear_training.json"
CAPABILITY_SHARDS = (
    PROJECT_ROOT / "inputs/capable_linear_capability_t0_3.json",
    PROJECT_ROOT / "inputs/capable_linear_capability_t3_6.json",
    PROJECT_ROOT / "inputs/capable_linear_capability_t6_8.json",
    PROJECT_ROOT / "inputs/capable_linear_capability_t8_10.json",
)
UNIT_ATTESTATION = PROJECT_ROOT / "inputs/direct_odt_truncation_unit_v2.json"
DIRECT_TEST_SOURCE_MANIFEST = (
    PROJECT_ROOT / "athena/direct_odt_truncation_tests_sources.sha256"
)
R7_FULL_CONTROL = PROJECT_ROOT / "inputs/r7_full_exact_control.json"
R7_COMPOSITE_CONTROL = PROJECT_ROOT / "inputs/r7_composite_control.json"
DOOMS_REFERENCE = PROJECT_ROOT / "reference/dooms_xnets_2504.02667.pdf"
RESULT_DIRECTORY = PROJECT_ROOT / "athena/results/capable_linear_b1c0"
PROGRESS_DIRECTORY = RESULT_DIRECTORY / "progress"
FULL_RANK_OUTPUT = RESULT_DIRECTORY / "full_rank_certificate.json"
COMPRESSION_OUTPUT = RESULT_DIRECTORY / "compression_result.json"
COMPOSITE_OUTPUT = RESULT_DIRECTORY / "composite_attestation.json"
FRESH_CAPABILITY_OUTPUT = RESULT_DIRECTORY / "fresh_capability_aggregate.json"
FRESH_CAPABILITY_SMOKE_OUTPUT = RESULT_DIRECTORY / "fresh_capability_smoke.json"
CURRENT_UNIT_ATTESTATION = RESULT_DIRECTORY / "unit_suite.json"
CURRENT_DIRECT_TEST_RESULT = (
    PROJECT_ROOT / "athena/results/capable_linear_b1c0_core_unit.json"
)

EXPECTED_INPUT_SHA256 = {
    "inputs/capable_linear_b1c0_checkpoint.pt": "b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee",
    "inputs/capable_linear_training.json": "e6c07beb2efb7e5ebe55c92fabf694d97b50b9e399f50868eeef4af455e30fa7",
    "inputs/capable_linear_capability_t0_3.json": "a89c586016ea87336128afefeaf1bdb922925de8abdac81eaa486f79de0bbe7f",
    "inputs/capable_linear_capability_t3_6.json": "f23ab6c31d308da4720c5e0ad8da1aa46729c9453fe4ec0b1cc24d1df9d2e599",
    "inputs/capable_linear_capability_t6_8.json": "40b17ca6a857c3999da8f981f9c3fa262bcbb23ae6ca0ca6cf0b9055556930c5",
    "inputs/capable_linear_capability_t8_10.json": "bd9d01e43c6a3fff455d0704f46351e3973675380174544bb1df53d39a62625d",
    "inputs/direct_odt_truncation_unit_v2.json": "0c41680e8603036de2577638e68b14bc397df7920d96c2fefe5026886cad54e7",
    "inputs/r7_full_exact_control.json": "f90a2c8c4842206aface13801770ea07fd6a4b3f12c023b0faeb984177d8b374",
    "inputs/r7_composite_control.json": "bad374a514e4d4503f5fe316f4a95a12bdbf2c897b964fd8ead601d1a20cc770",
    "reference/dooms_xnets_2504.02667.pdf": "a97f59a91d90ffde56bc22bf0b3c79e0891ff88197ca9d7670331e4208277559",
}
EXPECTED_CHECKPOINT_BYTES = 80_752_109
EXPECTED_PARAMETER_COUNT = 20_137_352
EXPECTED_NORMALIZATION_SITE_COUNT = 74
EXPECTED_INACTIVE_NORMALIZATION_SITES = ("vision.norm_out",)
EXPECTED_PADE_NUMERATOR = (
    5.3299665451049805,
    7.535519123077393,
    0.32727372646331787,
)
EXPECTED_PADE_DENOMINATOR = (
    1.0,
    9.648269653320312,
    2.606637477874756,
)
EXPECTED_CHECKPOINT_TENSOR_SHAPES = {
    "tok_emb.weight": (26, 384),
    "action_head.weight": (7, 384),
    "action_head.bias": (7,),
}
EXPECTED_OLD_CHECKPOINT_SHA256 = (
    "96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9"
)
EXPECTED_TEST_COUNT = 52
EXPECTED_CAPABILITY_TEST_COUNT = 34
EXPECTED_ATTESTATION_TEST_COUNT = 7
REPLAY_LIMIT = 3e-8
STRICT_COMPARISON_LIMIT = 2e-10
OFFDIAGONAL_LIMIT = 3e-10
PHYSICAL_TARGET = 0.999999999
CAPABILITY_FLOOR = 0.80
EXPECTED_CONFIG_IDENTITY = {
    "image_size": 64,
    "patch_size": 8,
    "vit_dim": 192,
    "vit_layers": 4,
    "vit_heads": 8,
    "vit_ffn_rank": 576,
    "vocab_size": 26,
    "max_instr_len": 32,
    "state_dim": 8,
    "n_embodiments": 1,
    "dim": 384,
    "n_layers": 8,
    "n_heads": 12,
    "ffn_rank": 1152,
    "attn": "bilinear",
    "vit_attn": "bilinear",
    "ffn": "bilinear",
    "norm": "rational",
    "qk_norm": "rational",
    "residual": True,
    "vit_residual": True,
    "action_horizon": 8,
    "action_dim": 7,
    "action_head": "linear",
}
EXPECTED_FRESH_CAPABILITY_ENVIRONMENT = {
    "python": "3.10.19",
    "numpy": "1.26.4",
    "torch": "2.7.1+cu126",
    "libero": "0.1.0",
    "robosuite": "1.4.1",
    "mujoco": "3.5.0",
    "pillow": "12.1.1",
    "cuda": "12.6",
    "gpu": "NVIDIA A30",
    "mujoco_gl": "egl",
    "matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": True,
    "cublas_workspace_config": ":4096:8",
    "deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
}
EXPECTED_CAPABILITY_TASK_PROTOCOL = {
    0: ("pick up the alphabet soup and place it in the basket", "pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl", "df088984da13131f8332ee0f13a7896c6a97afd02ee5007a42e8fc5e0893571e", "91eed35ff60c2b791ba1d80996e359177a9202d81358e9fddb42ccafffa3f34e"),
    1: ("pick up the cream cheese and place it in the basket", "pick_up_the_cream_cheese_and_place_it_in_the_basket.bddl", "7019f37ee158d67a21338a6df0c441dd8f84979b946b5e20bb49463ef0508ea2", "6443945d386a843b9998caeee65f3093f50e4ab65e913e7102aa39eaee7edd59"),
    2: ("pick up the salad dressing and place it in the basket", "pick_up_the_salad_dressing_and_place_it_in_the_basket.bddl", "024978e9e8f43a49b49d0965f4f426d364ce056307f0fcf5733dc4682fac437a", "7f365994bb15f09bed7e4611603d8b5e4a3576f7fccb827241f6ad60cb8b3e3f"),
    3: ("pick up the bbq sauce and place it in the basket", "pick_up_the_bbq_sauce_and_place_it_in_the_basket.bddl", "8a7e37b76fce5621e649e260dbcfcec7ce2f9a055ca09a4f1ebaf5ca74a739c2", "b186084cc72f9cf8c07ca98c18542f4f0269b23f53bf98ac8474d1cc5b642702"),
    4: ("pick up the ketchup and place it in the basket", "pick_up_the_ketchup_and_place_it_in_the_basket.bddl", "4a4e545483a3fe30cf0ef3dc05bfe5e6ad5e37c94b4376dfa73fe1b60e75b319", "8a035ce260650b525cbdec044e777f2625e689aca72dd737983c4a61539d5ab8"),
    5: ("pick up the tomato sauce and place it in the basket", "pick_up_the_tomato_sauce_and_place_it_in_the_basket.bddl", "b1f4bb69d256a05f693838de46182bb0d3de0de68eec350df1b5b1e76f6c136d", "2015e882bad1d77efd5b973a070fd6bd4cf524f35c36497f5060b1acc7b59dcb"),
    6: ("pick up the butter and place it in the basket", "pick_up_the_butter_and_place_it_in_the_basket.bddl", "5d8053b40ff33246fdbcc38548db84032c6bb647b4b88b6646d9cf6b73a5006c", "981f9d28ca49331ad5faf46ff765233fe7d4fa39a3f35dd332104f900c8cc766"),
    7: ("pick up the milk and place it in the basket", "pick_up_the_milk_and_place_it_in_the_basket.bddl", "9910aabf6717e8ba3e24a3f8500d9bce9ee075346ea6961e8fd766f1257d2996", "f78700be5769be90421173e1689e458b65f4d791855aedc3d2dedc63e3b7a0aa"),
    8: ("pick up the chocolate pudding and place it in the basket", "pick_up_the_chocolate_pudding_and_place_it_in_the_basket.bddl", "674ceabc400b7b16d46e8eaf678709d4a633235c7e0a80233775044fe38b19b0", "54593bb836dc12d321d9e7971e54034bbe01c02929cf0c09cb93b75038bedf5a"),
    9: ("pick up the orange juice and place it in the basket", "pick_up_the_orange_juice_and_place_it_in_the_basket.bddl", "6298533e7bcfb83e40779a77fda39216e8cd53f22bf1af6388585dad0abb0b50", "c198880f92818df55d0a8e53ab739bbfd36acfab236a276c7ac72a27a27baf0d"),
}

_AUDIT_ONLY = sys.argv[1:] == ["--audit-only"]
if not _AUDIT_ONLY and PROJECT_ROOT != EXPECTED_STAGE_ROOT:
    raise RuntimeError(f"capable ODT must run from {EXPECTED_STAGE_ROOT}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or nonphysical")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _verify_line_manifest(
    path: Path, label: str, required_member: str | None = None
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a physical file")
    expected: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(f"malformed {label} line {line_number}")
        digest, relative = fields
        relative = relative.lstrip("*")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise RuntimeError(f"invalid {label} digest on line {line_number}")
        candidate = PROJECT_ROOT / relative
        resolved = candidate.resolve(strict=True)
        if candidate.is_symlink() or resolved == PROJECT_ROOT or PROJECT_ROOT not in resolved.parents:
            raise RuntimeError(f"{label} contains an unsafe path")
        normalized = resolved.relative_to(PROJECT_ROOT).as_posix()
        if normalized in expected:
            raise RuntimeError(f"{label} contains a duplicate path")
        expected[normalized] = digest
    if required_member is not None and required_member not in expected:
        raise RuntimeError(f"{label} does not authenticate {required_member}")
    actual = {relative: _sha256(PROJECT_ROOT / relative) for relative in sorted(expected)}
    if actual != expected:
        changed = sorted(key for key in expected if actual.get(key) != expected[key])
        raise RuntimeError(f"authenticated {label} bytes changed: {changed}")
    return {
        "path": path.resolve().as_posix(),
        "sha256": _sha256(path),
        "source_count": len(actual),
        "source_sha256": actual,
    }


def _verify_source_manifest() -> dict[str, Any]:
    runner = Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix()
    return _verify_line_manifest(SOURCE_MANIFEST, "source manifest", runner)


_PREIMPORT_MANIFEST = None if _AUDIT_ONLY else _verify_source_manifest()

from scripts.odt_direct_only_compliance import (
    DIRECT_RQ_RUNTIME_CALL,
    PRODUCTION_ALGORITHM3_RUNTIME_CALL,
    STREAMED_QR_RUNTIME_CALL,
    TRIANGULAR_RUNTIME_CALL,
    assert_direct_only_runtime_guard,
    audit_direct_only_launch,
    canonical_direct_only_entrypoints,
    direct_only_runtime_report,
    install_direct_only_runtime_guard,
)
from scripts.capable_linear_artifact_integrity import (
    assert_physical_hashes_unchanged,
    read_authenticated_json,
    require_rational_norm_buffer_inventory,
)

if Path(sys.modules["scripts"].__file__).resolve() != (
    PROJECT_ROOT / "scripts/__init__.py"
).resolve() or Path(
    sys.modules["scripts.odt_direct_only_compliance"].__file__
).resolve() != (
    PROJECT_ROOT / "scripts/odt_direct_only_compliance.py"
).resolve():
    raise RuntimeError("capable ODT imported a shadowed compliance authority")


_ENTRYPOINTS = (
    *canonical_direct_only_entrypoints(PROJECT_ROOT, Path(__file__)),
    PROJECT_ROOT / "tests/test_direct_odt_truncation.py",
    PROJECT_ROOT / "tests/test_capable_linear_artifact_integrity.py",
)
if len(_ENTRYPOINTS) != len({path.resolve() for path in _ENTRYPOINTS}):
    raise RuntimeError("direct-only audit entrypoint inventory contains duplicates")
_STATIC_AUDIT = audit_direct_only_launch(PROJECT_ROOT, _ENTRYPOINTS)
for _field in (
    "prohibited_calls_found",
    "prohibited_self_overlap_sites",
    "guarded_dormant_spectral_norm_sites",
    "duplicate_top_level_definition_sites",
):
    if _STATIC_AUDIT[_field]:
        raise RuntimeError(f"direct-only static audit did not close {_field}")
if _PREIMPORT_MANIFEST is not None and (
    _PREIMPORT_MANIFEST["source_sha256"] != _STATIC_AUDIT["source_sha256"]
):
    raise RuntimeError("source manifest differs from the audited closure")
if "--audit-only" in sys.argv[1:]:
    if not _AUDIT_ONLY:
        raise RuntimeError("--audit-only cannot be combined with run arguments")
    print(json.dumps(_STATIC_AUDIT, indent=2, sort_keys=True))
    raise SystemExit(0)

_RUNTIME_AT_IMPORT = install_direct_only_runtime_guard()

import numpy as np
import torch
import torch.nn as nn

from scripts.run_production_factorized_odt_stress import production_config
from xvla.models.vla import ChiVLA
from xvla.nn.normalization import RationalNorm
from xvla.train.direct_odt_truncation import (
    CompactRankBank,
    evaluate_diagonal_prefix_quotient,
    evaluate_diagonal_prefixes,
    evaluate_diagonal_suffix_quotient,
    implicit_storage_elements,
    projected_prefix_storage_elements,
    truncate_diagonal_prefixes,
)
from xvla.train.implicit_sparse_projective_odt import (
    COMPACT_TRACE_RETENTION_TARGETS,
    DIRECT_Q_PROVENANCE_TOLERANCE,
    DIRECT_RQ_METHOD,
    LOCAL_SCALE_CERTIFICATION_TOLERANCE,
    RECTANGULAR_RQ_METHOD,
    STREAMED_DIRECT_Q_PROVENANCE_METHOD,
    UNARY_RQ_METHOD,
    _projective_batch_relative_error,
    audit_algorithm1_factorization_calls,
    canonicalize_implicit_dag_direct_rq,
    diagonalize_implicit_dag_full_rank,
    evaluate_boundary_quotient,
    evaluate_projective_boundary,
    implicit_shape_statistics,
    predict_direct_rq_route_inventory,
    telemetry_dict,
    validate_canonical_exponent_normal_form,
)
from xvla.train.implicit_sparse_projective_odt_vla import (
    LINEAR_ACTION_OBSERVABLE,
    _active_norm_snapshot,
    assert_full_vla_norm_buffers_unchanged,
    compile_full_vla_projective_dag,
    evaluate_full_vla_observable,
    evaluate_full_vla_quotient,
    full_vla_structure_statistics,
    physical_batch_from_model_inputs,
    physical_source_mapping,
    source_full_vla_observable,
    source_full_vla_output,
)


DTYPE = torch.float64


def _assert_runtime() -> None:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("capable linear ODT requires frozen Python 3.10")
    if np.__version__ != "1.26.4":
        raise RuntimeError("capable linear ODT requires frozen NumPy 1.26.4")
    if torch.__version__.split("+", 1)[0] != "2.7.1":
        raise RuntimeError("capable linear ODT requires frozen Torch 2.7.1")


def _assert_sources() -> dict[str, Any]:
    current_manifest = _verify_source_manifest()
    current_audit = audit_direct_only_launch(PROJECT_ROOT, _ENTRYPOINTS)
    if current_audit != _STATIC_AUDIT:
        raise RuntimeError("audited source closure changed during execution")
    if current_manifest != _PREIMPORT_MANIFEST:
        raise RuntimeError("authenticated source closure changed during execution")
    return current_manifest


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    first = actual.reshape(-1)
    second = expected.reshape(-1)
    numerator = torch.sqrt(torch.sum((first - second) * (first - second)))
    denominator = torch.sqrt(torch.sum(second * second)).clamp_min(
        torch.finfo(second.dtype).tiny
    )
    return float((numerator / denominator).item())


def _state_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _validate_output(path: Path, directory: Path) -> Path:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing existing output {path}")
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError("output directory must be physical")
    resolved = path.resolve(strict=False)
    if resolved.parent != directory.resolve(strict=True):
        raise RuntimeError("output must be a direct child of its fixed directory")
    return resolved


def _publish(path: Path, payload: Mapping[str, Any]) -> str:
    _validate_output(path, path.parent)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(12)}.tmp")
    linked = False
    source_stat: os.stat_result | None = None
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        source_stat = os.lstat(temporary)
        if not stat.S_ISREG(source_stat.st_mode):
            raise RuntimeError("result temporary is not a regular file")
        _assert_sources()
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        if _sha256(path) != digest:
            raise RuntimeError("published result differs from encoded bytes")
        _assert_sources()
        os.chmod(path, 0o444, follow_symlinks=False)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if _sha256(path) != digest or os.lstat(path).st_mode & 0o222:
            raise RuntimeError("published result is not immutable")
        return digest
    except BaseException:
        if linked and source_stat is not None and os.path.lexists(path):
            observed = os.lstat(path)
            if observed.st_dev == source_stat.st_dev and observed.st_ino == source_stat.st_ino:
                path.unlink()
        raise
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def _verify_fixed_inputs() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative, expected in EXPECTED_INPUT_SHA256.items():
        path = PROJECT_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"authenticated input is missing or nonphysical: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"authenticated input changed: {relative}")
        result[relative] = actual
    if CHECKPOINT.stat().st_size != EXPECTED_CHECKPOINT_BYTES:
        raise RuntimeError("checkpoint byte count differs")
    return result


def _expected_capability_task_record(task_index: int) -> dict[str, Any]:
    language, bddl_file, bddl_sha, states_sha = EXPECTED_CAPABILITY_TASK_PROTOCOL[
        task_index
    ]
    return {
        "language": language,
        "problem_folder": "libero_object",
        "bddl_file": bddl_file,
        "bddl_sha256": bddl_sha,
        "init_state_count": 50,
        "init_states_sha256": states_sha,
    }


def _fresh_environment_is_exact(value: Any) -> bool:
    return isinstance(value, dict) and value == EXPECTED_FRESH_CAPABILITY_ENVIRONMENT


def _capability_attestation() -> dict[str, Any]:
    training = _read_json(TRAINING_RESULT, "training result")
    training_gates = {
        "architecture": training.get("architecture") == "chi",
        "vision": training.get("vision_encoder") == "vit",
        "suite": training.get("suite") == "libero_object",
        "seed": training.get("seed") == 0,
        "checkpoint_name": training.get("checkpoint")
        == "artifacts/ckpt_rational_norm_ablation_v1_rational_s0.pt",
        "recipe": training.get("steps") == 40_000
        and training.get("batch_size") == 256
        and training.get("lr") == 8e-4
        and training.get("res") == 64
        and training.get("horizon") == 8,
        "sample_and_parameter_counts": training.get("samples") == 63_352
        and training.get("parameters") == 20_137_352,
        "vocabulary": len(training.get("vocab", {})) == 26
        and training.get("vocab_sha256")
        == "1e353b073134c6a1e62ca0c2ee0e9ad7204a007fd343b1bc6e09075ec0a8f2fe",
    }
    failures = sorted(name for name, value in training_gates.items() if not value)
    if failures:
        raise RuntimeError(f"training identity gates failed: {failures}")

    expected_ranges = ((0, 3), (3, 6), (6, 8), (8, 10))
    total_successes = 0
    total_trials = 0
    tasks: set[int] = set()
    shards: list[dict[str, Any]] = []
    exact_protocol = {
        "res": 64,
        "horizon": 8,
        "num_steps_wait": 10,
        "exec_h": 8,
        "eps_per_task": 50,
        "max_steps": 280,
    }
    for path, (start, stop) in zip(CAPABILITY_SHARDS, expected_ranges):
        value = _read_json(path, f"capability shard {start}:{stop}")
        capability = value.get("capability")
        if not isinstance(capability, dict):
            raise RuntimeError("capability shard lacks a capability object")
        indices = list(range(start, stop))
        episodes = capability.get("episodes")
        task_protocol = capability.get("task_protocol")
        expected_pairs = [
            (task_index, episode)
            for task_index in indices
            for episode in range(50)
        ]
        observed_pairs: list[tuple[int, int]] = []
        episode_fields_valid = isinstance(episodes, list)
        if episode_fields_valid:
            for item in episodes:
                if not isinstance(item, dict):
                    episode_fields_valid = False
                    break
                task_index = item.get("task_index")
                episode = item.get("episode")
                success = item.get("success")
                steps = item.get("steps")
                elapsed = item.get("elapsed_s")
                if (
                    type(task_index) is not int
                    or type(episode) is not int
                    or type(success) is not bool
                    or type(steps) is not int
                    or not 0 <= steps <= 280
                    or not isinstance(elapsed, (int, float))
                    or isinstance(elapsed, bool)
                    or not math.isfinite(float(elapsed))
                    or float(elapsed) < 0.0
                ):
                    episode_fields_valid = False
                    break
                observed_pairs.append((task_index, episode))
        expected_task_protocol = {
            str(index): _expected_capability_task_record(index) for index in indices
        }
        gates = {
            "identity": value.get("architecture") == "chi"
            and value.get("vision_encoder") == "vit"
            and value.get("suite") == value.get("training_suite") == "libero_object"
            and value.get("seed") == 0
            and value.get("checkpoint")
            == "artifacts/ckpt_rational_norm_ablation_v1_rational_s0.pt",
            "mode": value.get("mode") == "capability"
            and value.get("cp_pruning") is None
            and value.get("ensemble_checkpoints") is None
            and value.get("ensemble_reduction") is None,
            "protocol": value.get("evaluation_protocol") == exact_protocol,
            "tasks": capability.get("task_indices") == indices,
            "canonical_states": capability.get("canonical_init_states") is True,
            "counts": capability.get("eps_per_task") == 50
            and capability.get("trials") == 50 * len(indices),
            "episodes": isinstance(episodes, list)
            and len(episodes) == 50 * len(indices)
            and episode_fields_valid
            and observed_pairs == expected_pairs,
            "task_protocol": task_protocol == expected_task_protocol,
            "prediction": value.get("prediction_finite") is True
            and value.get("prediction_shape") == [1, 8, 7]
            and value.get("vocab_size") == 26,
        }
        failed = sorted(name for name, passed in gates.items() if not passed)
        if failed:
            raise RuntimeError(f"capability shard {start}:{stop} failed: {failed}")
        observed_successes = sum(item.get("success") is True for item in episodes)
        observed_tasks = {int(item.get("task_index")) for item in episodes}
        observed_counts = {
            index: sum(int(item.get("task_index")) == index for item in episodes)
            for index in indices
        }
        expected_per_task = {
            EXPECTED_CAPABILITY_TASK_PROTOCOL[index][0]: (
                sum(
                    int(item["success"])
                    for item in episodes
                    if item["task_index"] == index
                )
                / 50
            )
            for index in indices
        }
        if (
            observed_successes != capability.get("successes")
            or observed_tasks != set(indices)
            or set(observed_counts.values()) != {50}
            or capability.get("per_task") != expected_per_task
            or not math.isclose(
                capability.get("overall"),
                observed_successes / len(episodes),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise RuntimeError(f"capability shard {start}:{stop} count mismatch")
        total_successes += observed_successes
        total_trials += len(episodes)
        tasks.update(observed_tasks)
        shards.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _sha256(path),
                "tasks": indices,
                "successes": observed_successes,
                "trials": len(episodes),
            }
        )
    if total_successes != 419 or total_trials != 500 or tasks != set(range(10)):
        raise RuntimeError("official capability aggregate is not exactly 419/500")
    return {
        "claim_boundary": (
            "Separate closed-loop evidence associated with the same checkpoint pathname "
            "under the official LIBERO-Object 280-step, 50-canonical-initial-state-per-task "
            "protocol. The historical records did not store a checkpoint digest, so this "
            "is not an at-evaluation SHA-256 binding to the current checkpoint bytes."
        ),
        "training_sha256": _sha256(TRAINING_RESULT),
        "training_gates": training_gates,
        "shards": shards,
        "task_indices": list(range(10)),
        "successes": total_successes,
        "trials": total_trials,
        "success_rate": total_successes / total_trials,
        "official_protocol": exact_protocol,
        "recorded_checkpoint_path": (
            "artifacts/ckpt_rational_norm_ablation_v1_rational_s0.pt"
        ),
        "historical_checkpoint_digest_recorded": False,
        "sha256_bound_to_current_checkpoint": False,
    }


def _unit_attestation() -> dict[str, Any]:
    payload = _read_json(UNIT_ATTESTATION, "direct ODT unit attestation")
    junit = payload.get("junit")
    static = payload.get("static_audit")
    runtime = payload.get("runtime_guard_final")
    required = {
        "test_bounded_rank_deficient_385x386_retains_direct_q_and_matches_clone_every_step",
        "test_unchanged_six_pade_attention_ffn_residual_block_matches_every_step",
        "test_streamed_production_algorithms2_and3_match_separated_clone_reference",
        "test_every_prefix_step_matches_the_no_memo_clone_and_masked_replay",
        "test_materialization_projective_comparison_ignores_rowwise_radial_scale",
    }
    names = junit.get("test_names") if isinstance(junit, dict) else None
    gates = {
        "suite_green": payload.get("all_gates_pass") is True,
        "exact_count": isinstance(junit, dict)
        and junit.get("tests") == EXPECTED_TEST_COUNT
        and junit.get("failures") == 0
        and junit.get("errors") == 0
        and junit.get("skipped") == 0,
        "unique_inventory": isinstance(names, list)
        and len(names) == EXPECTED_TEST_COUNT
        and len(set(names)) == EXPECTED_TEST_COUNT,
        "required_oracles": isinstance(names, list) and required.issubset(names),
        "static_closed": isinstance(static, dict)
        and static.get("prohibited_calls_found") == []
        and static.get("prohibited_self_overlap_sites") == []
        and static.get("guarded_dormant_spectral_norm_sites") == []
        and static.get("duplicate_top_level_definition_sites") == [],
        "runtime_closed": isinstance(runtime, dict)
        and runtime.get("prohibited_attempt_count") == 0
        and runtime.get("prohibited_attempts") in ([], ()),
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    if failed:
        raise RuntimeError(f"direct ODT unit attestation failed: {failed}")
    unit_sources = static.get("source_sha256", {})
    for relative in (
        "xvla/train/implicit_sparse_projective_odt.py",
        "xvla/train/implicit_sparse_projective_odt_vla.py",
        "xvla/train/direct_odt_truncation.py",
        "xvla/train/direct_odt_clone_reference.py",
    ):
        if unit_sources.get(relative) != _STATIC_AUDIT["source_sha256"].get(relative):
            raise RuntimeError(f"unit attestation core identity differs: {relative}")
    return {
        "path": UNIT_ATTESTATION.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": _sha256(UNIT_ATTESTATION),
        "test_count": EXPECTED_TEST_COUNT,
        "required_oracles": sorted(required),
        "gates": gates,
        "shared_core_sha256": {
            relative: unit_sources[relative]
            for relative in sorted(unit_sources)
            if relative.startswith("xvla/train/")
        },
    }


def _junit_is_exact(value: Any, expected_count: int) -> bool:
    if not isinstance(value, dict):
        return False
    names = value.get("test_names")
    return (
        value.get("tests") == expected_count
        and value.get("failures") == 0
        and value.get("errors") == 0
        and value.get("skipped") == 0
        and isinstance(names, list)
        and len(names) == expected_count
        and len(set(names)) == expected_count
    )


def _current_unit_attestation(manifest: dict[str, Any]) -> dict[str, Any]:
    fresh_manifest = _verify_line_manifest(
        FRESH_CAPABILITY_SOURCE_MANIFEST,
        "fresh capability source manifest",
        "scripts/run_capable_linear_fresh_capability.py",
    )
    direct_manifest = _verify_line_manifest(
        DIRECT_TEST_SOURCE_MANIFEST,
        "direct test source manifest",
        "scripts/run_direct_odt_truncation_tests.py",
    )
    unit, unit_sha256 = read_authenticated_json(
        CURRENT_UNIT_ATTESTATION, "current combined unit attestation"
    )
    direct, direct_sha256 = read_authenticated_json(
        CURRENT_DIRECT_TEST_RESULT, "current direct ODT test result"
    )
    unit_gates = unit.get("gates")
    expected_unit_gate_names = {
        "direct_process_exit_zero",
        "direct_self_gate",
        "direct_exact_count",
        "direct_no_skips",
        "capability_process_exit_zero",
        "capability_exact_count",
        "capability_no_failures",
        "capability_no_skips",
        "attestation_process_exit_zero",
        "attestation_exact_count",
        "attestation_no_failures",
        "attestation_no_skips",
        "combined_unique_names",
        "norm_inventory_and_dead_site_tests_present",
        "distinct_odt_and_capability_manifests",
        "process_isolation",
    }
    direct_junit = direct.get("junit")
    capability_junit = unit.get("capability_test_result", {}).get("junit")
    attestation_junit = unit.get("attestation_test_result", {}).get("junit")
    direct_static = direct.get("static_audit")
    direct_runtime = direct.get("runtime_guard_final")
    direct_names = direct_junit.get("test_names") if isinstance(direct_junit, dict) else []
    required_direct_tests = {
        "test_bounded_rank_deficient_385x386_retains_direct_q_and_matches_clone_every_step",
        "test_unchanged_six_pade_attention_ffn_residual_block_matches_every_step",
        "test_streamed_production_algorithms2_and3_match_separated_clone_reference",
        "test_every_prefix_step_matches_the_no_memo_clone_and_masked_replay",
        "test_materialization_projective_comparison_ignores_rowwise_radial_scale",
    }
    gates = {
        "schema": unit.get("schema")
        == "xvla_capable_linear_b1c0_combined_unit_suite_v1",
        "unit_self_gate": unit.get("all_gates_pass") is True
        and isinstance(unit_gates, dict)
        and set(unit_gates) == expected_unit_gate_names
        and all(value is True for value in unit_gates.values()),
        "isolated_processes": unit.get("process_boundary")
        == {
            "direct_odt_guarded_process": True,
            "fresh_capability_test_process": True,
            "attestation_integrity_test_process": True,
            "shared_process": False,
        },
        "manifest_identities": unit.get("odt_source_manifest_sha256")
        == manifest["sha256"]
        and unit.get("capability_source_manifest_sha256")
        == fresh_manifest["sha256"]
        and unit.get("direct_test_manifest_sha256") == direct_manifest["sha256"]
        and unit.get("stage_ledger_sha256") == _sha256(STAGE_LEDGER),
        "direct_result_identity": unit.get("direct_test_result", {}).get("path")
        == CURRENT_DIRECT_TEST_RESULT.relative_to(PROJECT_ROOT).as_posix()
        and unit.get("direct_test_result", {}).get("sha256") == direct_sha256
        and unit.get("direct_test_result", {}).get("junit") == direct_junit,
        "direct_suite": direct.get("all_gates_pass") is True
        and _junit_is_exact(direct_junit, EXPECTED_TEST_COUNT)
        and required_direct_tests.issubset(direct_names),
        "capability_suite": _junit_is_exact(
            capability_junit, EXPECTED_CAPABILITY_TEST_COUNT
        ),
        "integrity_suite": _junit_is_exact(
            attestation_junit, EXPECTED_ATTESTATION_TEST_COUNT
        ),
        "direct_static": isinstance(direct_static, dict)
        and direct_static.get("source_sha256") == direct_manifest["source_sha256"]
        and direct_static.get("direct_qr_call_sites") == 4
        and direct_static.get("prohibited_calls_found") == []
        and direct_static.get("prohibited_self_overlap_sites") == []
        and direct_static.get("guarded_dormant_spectral_norm_sites") == []
        and direct_static.get("duplicate_top_level_definition_sites") == [],
        "direct_runtime": isinstance(direct_runtime, dict)
        and direct_runtime.get("prohibited_attempt_count") == 0
        and direct_runtime.get("prohibited_attempts") in ([], ()),
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    if failed:
        raise RuntimeError(f"current combined unit attestation failed: {failed}")
    input_sha256 = {
        "current_unit": unit_sha256,
        "current_direct": direct_sha256,
    }
    assert_physical_hashes_unchanged(
        {
            "current_unit": CURRENT_UNIT_ATTESTATION,
            "current_direct": CURRENT_DIRECT_TEST_RESULT,
        },
        input_sha256,
    )
    return {
        "path": CURRENT_UNIT_ATTESTATION.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": unit_sha256,
        "direct_result_path": CURRENT_DIRECT_TEST_RESULT.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "direct_result_sha256": direct_sha256,
        "test_counts": {
            "direct": EXPECTED_TEST_COUNT,
            "capability": EXPECTED_CAPABILITY_TEST_COUNT,
            "integrity": EXPECTED_ATTESTATION_TEST_COUNT,
        },
        "input_sha256": input_sha256,
        "gates": gates,
    }


def _assert_current_unit_unchanged(value: Mapping[str, Any]) -> None:
    expected = value.get("input_sha256")
    if not isinstance(expected, dict):
        raise RuntimeError("current unit attestation omits its input identities")
    assert_physical_hashes_unchanged(
        {
            "current_unit": CURRENT_UNIT_ATTESTATION,
            "current_direct": CURRENT_DIRECT_TEST_RESULT,
        },
        expected,
    )


def _r7_control_attestation() -> dict[str, Any]:
    full = _read_json(R7_FULL_CONTROL, "r7 full exact control")
    composite = _read_json(R7_COMPOSITE_CONTROL, "r7 composite control")
    identities = composite.get("identities")
    full_shared = composite.get("full_shared_dag")
    if (
        full.get("checkpoint_sha256") != EXPECTED_OLD_CHECKPOINT_SHA256
        or full.get("full_shared_dag_algorithms_1_to_3_completed") is not True
        or full.get("structure", {}).get("unique_nodes") != 3_964_463
        or full.get("structure", {}).get("edge_occurrences") != 7_216_854
        or composite.get("canonical_odt_certified") is not True
        or composite.get("composite_attestation_passed") is not True
        or not isinstance(identities, dict)
        or identities.get("checkpoint_sha256") != EXPECTED_OLD_CHECKPOINT_SHA256
        or identities.get("full_result_sha256") != _sha256(R7_FULL_CONTROL)
        or not isinstance(full_shared, dict)
        or full_shared.get("unique_nodes") != 3_964_463
    ):
        raise RuntimeError("r7 exact regression control did not validate")
    return {
        "claim_boundary": (
            "Previously certified exact linear-policy execution on another checkpoint, "
            "included only as a topology and implementation regression control."
        ),
        "checkpoint_sha256": EXPECTED_OLD_CHECKPOINT_SHA256,
        "full_result_sha256": _sha256(R7_FULL_CONTROL),
        "composite_sha256": _sha256(R7_COMPOSITE_CONTROL),
        "unique_nodes": 3_964_463,
        "edge_occurrences": 7_216_854,
        "canonical_odt_certified": True,
    }


def _arm_inactive_vision_norm_sentinel(
    model: ChiVLA,
) -> tuple[dict[str, int], Any]:
    counter = {"calls": 0}

    def reject_inactive_site(_module: Any, _inputs: Any) -> None:
        counter["calls"] += 1
        raise RuntimeError("inactive vision normalization site was executed")

    handle = model.vision.norm_out.register_forward_pre_hook(reject_inactive_site)
    return counter, handle


def _arm_active_norm_call_counters(
    model: ChiVLA,
) -> tuple[dict[str, int], list[Any]]:
    counters: dict[str, int] = {}
    handles: list[Any] = []
    for name, module in model.named_modules():
        if not isinstance(module, RationalNorm) or name in EXPECTED_INACTIVE_NORMALIZATION_SITES:
            continue
        counters[name] = 0

        def observe(_module: Any, _inputs: Any, *, site: str = name) -> None:
            counters[site] += 1

        handles.append(module.register_forward_pre_hook(observe))
    if len(counters) != 73:
        raise RuntimeError("active normalization sentinel inventory differs")
    return counters, handles


def _remove_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def _load_model() -> tuple[ChiVLA, dict[str, Any]]:
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or len(state) != 540:
        raise RuntimeError("checkpoint state inventory differs")
    if not all(isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in state.items()):
        raise RuntimeError("checkpoint is not a string-to-tensor mapping")
    if any(
        name not in state or tuple(state[name].shape) != expected
        for name, expected in EXPECTED_CHECKPOINT_TENSOR_SHAPES.items()
    ):
        raise RuntimeError("checkpoint tensor-shape identity differs")
    normalization_inventory = require_rational_norm_buffer_inventory(
        state,
        expected_count=EXPECTED_NORMALIZATION_SITE_COUNT,
        expected_inactive_sites=EXPECTED_INACTIVE_NORMALIZATION_SITES,
        expected_inactive_running_ms={"vision.norm_out": 1.0},
        expected_pade_numerator=EXPECTED_PADE_NUMERATOR,
        expected_pade_denominator=EXPECTED_PADE_DENOMINATOR,
    )
    config = production_config(26)
    config_identity = {
        key: getattr(config, key) for key in EXPECTED_CONFIG_IDENTITY
    }
    if config_identity != EXPECTED_CONFIG_IDENTITY:
        raise RuntimeError("exact compiler config identity differs")
    model = ChiVLA(config)
    model.load_state_dict(state, strict=True)
    model = model.to(dtype=DTYPE, device=torch.device("cpu")).eval()
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("loaded model parameter count differs")
    if model.cfg.action_head != "linear" or model.cfg.norm != "rational":
        raise RuntimeError("loaded checkpoint topology is not the linear rational VLA")
    named_norms = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, RationalNorm)
    }
    if set(named_norms) != {
        name[: -len(".initialized")]
        for name in state
        if name.endswith(".initialized")
    }:
        raise RuntimeError("loaded normalization module inventory differs")
    active_norms = {
        name: module
        for name, module in named_norms.items()
        if name not in EXPECTED_INACTIVE_NORMALIZATION_SITES
    }
    if len(active_norms) != 73 or any(
        not bool(module.initialized) for module in active_norms.values()
    ):
        raise RuntimeError("active normalization initialization differs")
    compiler_active_modules: list[RationalNorm] = []
    for block in model.vision.blocks.blocks:
        compiler_active_modules.extend(
            module for module in block.modules() if isinstance(module, RationalNorm)
        )
    for block in model.backbone.blocks:
        compiler_active_modules.extend(
            module for module in block.modules() if isinstance(module, RationalNorm)
        )
    compiler_active_modules.append(model.norm_out)
    compiler_active_ids = {id(module) for module in compiler_active_modules}
    loaded_active_ids = {id(module) for module in active_norms.values()}
    compiler_snapshot = _active_norm_snapshot(model)
    compiler_snapshot_sites = set()
    for name, _value in compiler_snapshot:
        if not name.endswith(".initialized"):
            continue
        site = name.rsplit(".", 1)[0]
        if site.startswith("vision.blocks."):
            site = site.replace("vision.blocks.", "vision.blocks.blocks.", 1)
        compiler_snapshot_sites.add(site)
    if (
        len(compiler_active_modules) != 73
        or len(compiler_active_ids) != 73
        or compiler_active_ids != loaded_active_ids
        or len(compiler_snapshot) != 73 * 4
        or compiler_snapshot_sites != set(active_norms)
    ):
        raise RuntimeError("compiler active normalization module identity differs")
    if (
        bool(model.vision.norm_out.initialized)
        or float(model.vision.norm_out.running_ms) != 1.0
    ):
        raise RuntimeError("inactive vision normalization state differs")
    if len(named_norms) != EXPECTED_NORMALIZATION_SITE_COUNT:
        raise RuntimeError("loaded model normalization module count differs")
    return model, {
        "checkpoint_sha256": _sha256(CHECKPOINT),
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "state_key_count": len(state),
        "parameter_count": parameter_count,
        "normalization_site_count": len(named_norms),
        "active_normalization_site_count": len(active_norms),
        "compiler_active_normalization_site_count": len(compiler_active_modules),
        "compiler_active_snapshot_site_count": len(compiler_snapshot_sites),
        "compiler_active_set_equals_loaded_active_set": True,
        "normalization_buffer_inventory": normalization_inventory,
        "config_identity": config_identity,
        "vocab_size": 26,
    }


def _replay_panel(model: ChiVLA, samples: int = 2):
    generator = torch.Generator().manual_seed(20260905)
    images = torch.randn(
        samples, 3, 64, 64, dtype=DTYPE, generator=generator
    ) * 0.035
    instructions = torch.randint(
        0, 26, (samples, 32), dtype=torch.long, generator=generator
    )
    state = torch.randn(samples, 8, dtype=DTYPE, generator=generator) * 0.11
    embodiments = torch.zeros(samples, dtype=torch.long)
    physical = physical_batch_from_model_inputs(
        model, images, instructions, state, embodiments
    )
    panel_digest = hashlib.sha256()
    for value in (images, instructions, state, embodiments):
        tensor = value.detach().cpu().contiguous()
        panel_digest.update(str(tensor.dtype).encode() + b"\0")
        panel_digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        panel_digest.update(tensor.numpy().tobytes())
    return physical, {
        "seed": 20260905,
        "samples": samples,
        "sha256": panel_digest.hexdigest(),
    }


def _algorithm1_summary(canonical: Any) -> tuple[dict[str, Any], dict[str, bool], set[str]]:
    steps = canonical.steps
    performed = tuple(
        step.per_step_function_replay_error
        for step in steps
        if step.per_step_function_replay_performed
    )
    streamed = tuple(step for step in steps if step.method == DIRECT_RQ_METHOD)
    maximum_local = max(
        (step.local_scaled_reconstruction_relative_error for step in steps),
        default=0.0,
    )
    maximum_absorption = max(
        (step.maximum_absorption_scaled_relative_error for step in steps),
        default=0.0,
    )
    expected_columns = sum(int(step.unfolding_shape[1]) for step in steps)
    expected_streamed_replays = sum(
        step.direct_q_provenance_stage_count for step in streamed
    )
    normal_form = validate_canonical_exponent_normal_form(canonical.network)
    summary = {
        "step_count": len(steps),
        "factorization_methods": sorted({step.method for step in steps}),
        "final_projective_replay_relative_error": canonical.final_projective_replay_error,
        "full_projective_replay_evaluations": canonical.full_projective_replay_evaluations,
        "per_step_replays_performed": len(performed),
        "maximum_local_scaled_reconstruction_relative_error": maximum_local,
        "maximum_absorption_scaled_relative_error": maximum_absorption,
        "push_ledger": [
            sum(step.parent_occurrences_pushed for step in steps),
            sum(step.expected_parent_occurrences for step in steps),
        ],
        "scale_sensitive_occurrence_ledger": [
            sum(step.scale_sensitive_occurrences_verified for step in steps),
            sum(step.parent_occurrences_pushed for step in steps),
        ],
        "canonical_exponent_normal_form": normal_form,
        "direct_q_provenance": {
            "verified_steps": sum(step.direct_q_provenance_verified for step in steps),
            "expected_steps": len(steps),
            "streamed_steps": len(streamed),
            "columns_compared": sum(step.direct_q_columns_compared for step in steps),
            "expected_columns": expected_columns,
            "streamed_replay_calls": canonical.telemetry.streamed_direct_q_replay_qr_factorizations,
            "expected_streamed_replay_calls": expected_streamed_replays,
            "maximum_compact_q_relative_error": max(
                (step.direct_q_compact_relative_error for step in steps), default=0.0
            ),
            "maximum_reconstruction_relative_error": max(
                (step.direct_q_reconstruction_relative_error for step in steps), default=0.0
            ),
        },
        "telemetry": telemetry_dict(canonical.telemetry),
    }
    gates = {
        "direct_methods_only": all(
            step.method in {DIRECT_RQ_METHOD, UNARY_RQ_METHOD, RECTANGULAR_RQ_METHOD}
            for step in steps
        ),
        "final_replay": math.isfinite(canonical.final_projective_replay_error)
        and canonical.final_projective_replay_error < REPLAY_LIMIT,
        "production_replay_schedule": not performed
        and canonical.full_projective_replay_evaluations == 2,
        "local_scale_certificates": maximum_local < LOCAL_SCALE_CERTIFICATION_TOLERANCE
        and maximum_absorption < LOCAL_SCALE_CERTIFICATION_TOLERANCE
        and all(
            step.local_reconstruction_exponent_delta == 0
            and step.maximum_absorption_exponent_delta == 0
            and step.scale_sensitive_occurrences_verified == step.parent_occurrences_pushed
            for step in steps
        ),
        "canonical_exponent_normal_form": normal_form["all_core_exponents_zero"]
        and normal_form["head_exponent_matches_ledger"],
        "direct_q_provenance_complete": all(
            step.direct_q_provenance_method
            and step.direct_q_provenance_verified
            and step.direct_q_columns_compared == int(step.unfolding_shape[1])
            and step.direct_q_compact_relative_error <= DIRECT_Q_PROVENANCE_TOLERANCE
            and step.direct_q_reconstruction_relative_error <= DIRECT_Q_PROVENANCE_TOLERANCE
            and step.direct_q_conditioning_accepted
            for step in steps
        ),
        "streamed_direct_q_ledger": all(
            step.direct_q_provenance_method == STREAMED_DIRECT_Q_PROVENANCE_METHOD
            and step.direct_q_provenance_stage_count
            == step.direct_q_provenance_column_blocks_compared
            == step.streamed_column_blocks
            and (step.direct_q_retained_transition_elements > 0)
            == (step.direct_q_provenance_stage_count > 1)
            for step in streamed
        ),
        "direct_q_telemetry_complete": canonical.telemetry.direct_q_provenance_certificates
        == len(steps)
        and canonical.telemetry.streamed_direct_q_provenance_certificates == len(streamed)
        and canonical.telemetry.direct_q_columns_compared == expected_columns
        and canonical.telemetry.streamed_direct_q_replay_qr_factorizations
        == expected_streamed_replays,
        "every_factor_occurrence_pushed": all(
            step.parent_occurrences_pushed == step.expected_parent_occurrences
            for step in steps
        ),
        "rank_deficiency_route_is_shape_only": all(
            step.direct_q_provenance_method == STREAMED_DIRECT_Q_PROVENANCE_METHOD
            or step.direct_q_conditioning_threshold == 0.0
            for step in steps
        ),
    }
    calls = {DIRECT_RQ_RUNTIME_CALL}
    if streamed:
        calls.update({STREAMED_QR_RUNTIME_CALL, TRIANGULAR_RUNTIME_CALL})
    return summary, gates, calls


def _postorder_uids(root: Any) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    stack: list[tuple[Any, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        identity = id(node)
        if expanded:
            order.append(int(node.uid))
            continue
        if identity in seen:
            continue
        seen.add(identity)
        stack.append((node, True))
        for child in reversed(node.children):
            if id(child) not in seen:
                stack.append((child, False))
    return order


def _progress(completed: int, total: int, started: float, record: Any, base: Mapping[str, Any]) -> None:
    output = PROGRESS_DIRECTORY / f"algorithm1_step_{completed:07d}_of_{total:07d}.json"
    payload = {
        "schema": "xvla_capable_linear_b1c0_direct_odt_progress_v1",
        **dict(base),
        "completed_steps": completed,
        "total_steps": total,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mb": _peak_rss_mb(),
        "last_step": {
            "uid": record.uid,
            "label": record.label,
            "method": record.method,
            "unfolding_shape": list(record.unfolding_shape),
            "direct_q_provenance_method": record.direct_q_provenance_method,
            "parent_occurrence_ledger": [
                record.parent_occurrences_pushed,
                record.expected_parent_occurrences,
            ],
        },
        "exact_global_decomposability_certified": False,
        "compression_claimed": False,
    }
    _publish(output, payload)


def _preflight_mode() -> None:
    _assert_runtime()
    manifest = _assert_sources()
    inputs = _verify_fixed_inputs()
    capability = _capability_attestation()
    unit = _unit_attestation()
    r7_control = _r7_control_attestation()
    direct_call_audit = audit_algorithm1_factorization_calls()
    if direct_call_audit.get("prohibited_calls_found"):
        raise RuntimeError("Algorithm 1 direct-call audit did not close")
    model, checkpoint = _load_model()
    state_before = _state_digest(model)
    physical, replay_panel = _replay_panel(model, samples=1)
    baseline_actions = source_full_vla_output(model, physical)
    inactive_norm_counter, inactive_norm_handle = (
        _arm_inactive_vision_norm_sentinel(model)
    )
    active_norm_calls, active_norm_handles = _arm_active_norm_call_counters(model)
    try:
        actions = source_full_vla_output(model, physical)
    finally:
        _remove_hooks([*active_norm_handles, inactive_norm_handle])
    preflight_gates = {
        "source_action_shape": tuple(actions.shape) == (1, 8, 7),
        "source_actions_finite": bool(torch.isfinite(actions).all()),
        "sentinels_preserve_source_output_bitwise": torch.equal(
            actions, baseline_actions
        ),
        "source_model_unchanged": state_before == _state_digest(model),
        "checkpoint_config_identity": checkpoint.get("config_identity")
        == EXPECTED_CONFIG_IDENTITY,
        "checkpoint_parameter_count": checkpoint.get("parameter_count")
        == EXPECTED_PARAMETER_COUNT,
        "inactive_vision_norm_never_executed": inactive_norm_counter["calls"] == 0,
        "all_73_active_norms_observed_once": len(active_norm_calls) == 73
        and all(count == 1 for count in active_norm_calls.values()),
        "direct_call_audit": not direct_call_audit.get("prohibited_calls_found"),
    }
    failed = sorted(name for name, passed in preflight_gates.items() if not passed)
    if failed:
        raise RuntimeError(f"preflight checkpoint/source gates failed: {failed}")
    runtime = assert_direct_only_runtime_guard(
        direct_only_runtime_report(), exact_allowed_calls=set()
    )
    result = {
        "schema": "xvla_capable_linear_b1c0_direct_odt_preflight_v1",
        "checkpoint": checkpoint,
        "capability": capability,
        "unit_attestation": unit,
        "previous_exact_checkpoint_control": r7_control,
        "replay_panel": replay_panel,
        "action_shape": list(actions.shape),
        "actions_finite": True,
        "inactive_vision_norm_sentinel": {
            "site": "vision.norm_out",
            "armed": True,
            "call_count": inactive_norm_counter["calls"],
            "total_site_count": 74,
            "inactive_sites": ["vision.norm_out"],
            "active_site_count": len(active_norm_calls),
            "active_call_count": sum(active_norm_calls.values()),
            "every_active_site_call_count": 1,
            "source_output_bitwise_equal_without_sentinels": torch.equal(
                actions, baseline_actions
            ),
        },
        "preflight_gates": preflight_gates,
        "full_graph_compiled": False,
        "route_inventory_deferred_to_full": True,
        "exact_odt_claimed": False,
        "static_audit": _STATIC_AUDIT,
        "runtime_guard": runtime,
        "source_manifest": manifest,
        "authenticated_inputs": inputs,
        "all_gates_pass": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def _full_mode() -> None:
    _assert_runtime()
    manifest = _assert_sources()
    inputs = _verify_fixed_inputs()
    capability = _capability_attestation()
    unit = _unit_attestation()
    current_unit = _current_unit_attestation(manifest)
    r7_control = _r7_control_attestation()
    direct_call_audit = audit_algorithm1_factorization_calls()
    if direct_call_audit.get("prohibited_calls_found"):
        raise RuntimeError("Algorithm 1 direct-call audit did not close")
    model, checkpoint = _load_model()
    inactive_norm_counter, inactive_norm_handle = (
        _arm_inactive_vision_norm_sentinel(model)
    )
    active_norm_calls, active_norm_handles = _arm_active_norm_call_counters(model)
    model_state_before = _state_digest(model)
    physical, replay_panel = _replay_panel(model)
    started_total = time.perf_counter()
    started = time.perf_counter()
    oracle = compile_full_vla_projective_dag(model)
    compile_seconds = time.perf_counter() - started
    structure = full_vla_structure_statistics(oracle)
    initial_shape = implicit_shape_statistics(oracle.network)
    route_inventory = predict_direct_rq_route_inventory(oracle.network)
    if route_inventory["streamed_tall_unrepresentable_count"] != 0:
        raise RuntimeError("pre-sweep route inventory contains an unsupported shape")

    started = time.perf_counter()
    try:
        source_observable = source_full_vla_observable(oracle, physical)
        source_actions = source_full_vla_output(model, physical)
    finally:
        _remove_hooks(active_norm_handles)
    compiled_observable = evaluate_full_vla_observable(oracle, physical)
    compiled_actions = evaluate_full_vla_quotient(oracle, physical)
    source_evaluation_seconds = time.perf_counter() - started
    source_replay = {
        "observable_relative_error": _relative(compiled_observable, source_observable),
        "public_action_relative_error": _relative(compiled_actions, source_actions),
        "observable_matches_public_action_relative_error": _relative(
            source_observable.reshape_as(source_actions), source_actions
        ),
    }
    source_gates = {
        "linear_action_root": oracle.observable_kind == LINEAR_ACTION_OBSERVABLE
        and oracle.product_signs is None
        and structure["observable_kind"] == LINEAR_ACTION_OBSERVABLE
        and structure["observable_euclidean_width"] == 56
        and structure["root_projective_width"] == 57,
        "source_observable_replay": source_replay["observable_relative_error"] < REPLAY_LIMIT,
        "source_public_action_replay": source_replay["public_action_relative_error"] < REPLAY_LIMIT,
        "source_observable_matches_public_action": source_replay[
            "observable_matches_public_action_relative_error"
        ]
        < STRICT_COMPARISON_LIMIT,
        "exact_topology": structure["unique_nodes"] == 3_964_463
        and structure["edge_occurrences"] == 7_216_854
        and structure["physical_source_count"] == structure["physical_leaf_count"] == 97
        and structure["maximum_physical_source_width"] == 192
        and structure["active_pade_sites"] == 73,
        "heterogeneous_ingress_unpadded": all(
            value != model.cfg.dim for value in structure["physical_source_widths"].values()
        )
        and "embodiment" not in structure["physical_source_widths"],
        "no_dense_clone_core": initial_shape["dense_clone_nodes"] == 0,
        "route_inventory_exact": route_inventory["schema"]
        == "exact_postorder_structural_direct_rq_routes_v1"
        and route_inventory["simulated_unique_node_count"] == initial_shape["unique_nodes"]
        and route_inventory["bounded_retained_q_candidate_count"]
        + route_inventory["streamed_cp_candidate_count"]
        == initial_shape["cp_binary_nodes"]
        and route_inventory["streamed_tall_unrepresentable_count"] == 0,
        "inactive_vision_norm_never_executed": inactive_norm_counter["calls"] == 0,
        "all_73_active_norms_observed_twice": len(active_norm_calls) == 73
        and all(count == 2 for count in active_norm_calls.values()),
    }
    failed = sorted(name for name, passed in source_gates.items() if not passed)
    if failed:
        raise RuntimeError(f"source/compiler gates failed before Algorithm 1: {failed}")

    raw = physical_source_mapping(model, physical)
    progress_base = {
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "capability_successes": capability["successes"],
        "capability_trials": capability["trials"],
        "unit_attestation_sha256": unit["sha256"],
        "replay_panel": replay_panel,
        "source_replay": source_replay,
    }
    started = time.perf_counter()

    def progress_callback(_work: Any, _node: Any, _factor: Any, record: Any) -> None:
        completed = record.step + 1
        total = initial_shape["unique_nodes"]
        if completed % 100_000 == 0 or completed == total:
            _progress(completed, total, started, record, progress_base)

    canonical = canonicalize_implicit_dag_direct_rq(
        oracle.network,
        replay_inputs=raw,
        block_size=4096,
        replay_each_step=False,
        copy_network=False,
        step_callback=progress_callback,
    )
    algorithm1_seconds = time.perf_counter() - started
    canonical_shape = implicit_shape_statistics(canonical.network)
    algorithm1, algorithm1_gates, runtime_calls = _algorithm1_summary(canonical)
    algorithm1_gates["node_inventory_preserved"] = (
        canonical_shape["unique_nodes"] == initial_shape["unique_nodes"]
        and canonical_shape["edge_occurrences"] == initial_shape["edge_occurrences"]
    )
    postorder_uids = _postorder_uids(canonical.network.root)
    algorithm1_gates["child_before_parent_postorder"] = (
        [step.uid for step in canonical.steps] == postorder_uids
    )
    failed = sorted(name for name, passed in algorithm1_gates.items() if not passed)
    if failed:
        raise RuntimeError(f"Algorithm 1 gates failed before Algorithm 2: {failed}")

    bank = CompactRankBank.for_network(canonical.network)
    started = time.perf_counter()
    diagonal = diagonalize_implicit_dag_full_rank(
        canonical.network,
        replay_inputs=raw,
        copy_network=False,
        stream_pre_evd_environments=True,
        retain_eigenvalues=False,
        retain_post_evd_environments=False,
        compact_spectrum_callback=bank,
    )
    algorithms2_and3_seconds = time.perf_counter() - started
    runtime_calls.add(PRODUCTION_ALGORITHM3_RUNTIME_CALL)
    full_rank_gates = {
        **source_gates,
        **algorithm1_gates,
        "algorithm2_explicit_downstream_contraction_complete": (
            diagonal.algorithm2_child_messages == canonical_shape["edge_occurrences"]
        ),
        "algorithm3_every_node": (
            diagonal.diagonalized_node_count == canonical_shape["unique_nodes"]
        ),
        "algorithm3_every_occurrence": diagonal.pushed_parent_occurrences
        == diagonal.expected_parent_occurrences
        == canonical_shape["edge_occurrences"] + 1,
        "algorithm3_replay": math.isfinite(diagonal.replay_relative_error)
        and diagonal.replay_relative_error < REPLAY_LIMIT,
        "algorithm3_recontracted_environments_diagonal": math.isfinite(
            diagonal.maximum_recontracted_offdiagonal_ratio
        )
        and diagonal.maximum_recontracted_offdiagonal_ratio < OFFDIAGONAL_LIMIT,
        "zero_full_spectrum_retention": diagonal.eigenvalues == ()
        and not diagonal.eigenvalue_spectra_retained,
        "compact_rank_bank_complete": bank.complete,
        "model_state_unchanged": model_state_before == _state_digest(model),
        "inactive_vision_norm_still_never_executed": inactive_norm_counter["calls"]
        == 0,
    }
    assert_full_vla_norm_buffers_unchanged(oracle)
    runtime_full = assert_direct_only_runtime_guard(
        direct_only_runtime_report(), exact_allowed_calls=runtime_calls
    )
    full_rank_gates["runtime_guard_exact"] = runtime_full["prohibited_attempt_count"] == 0
    failed = sorted(name for name, passed in full_rank_gates.items() if not passed)
    if failed:
        raise RuntimeError(f"full-rank Algorithms 1-3 gates failed: {failed}")

    full_rank_certificate = {
        "schema": "xvla_capable_linear_b1c0_full_rank_direct_odt_v1",
        "claim_boundary": oracle.network.claim_boundary,
        "deployment_interface_scope": {
            "learned_neural_dag_inputs": (
                "unit-scaled image patches, categorical instruction coordinates, and "
                "normalized continuous robot state"
            ),
            "learned_neural_dag_outputs": "normalized continuous action coordinates",
            "fixed_image_rotation_and_scaling_contracted": False,
            "fixed_state_normalization_contracted": False,
            "fixed_action_denormalization_contracted": False,
            "instruction_text_tokenization_contracted": False,
            "gripper_sign_decode_contracted": False,
            "checkpoint_vision_norm_out_active_in_chivla_forward": False,
            "checkpoint_vision_classifier_head_active_in_chivla_forward": False,
            "checkpoint_inactive_modules_in_compiled_dag": False,
            "fixed_permutation_and_affine_maps_are_tensor_compatible": True,
            "gripper_sign_decode_is_external": True,
            "persisted_deployable_tensor_network_artifact": False,
        },
        "numerical_precision_boundary": {
            "checkpoint_storage_dtype": "float32",
            "exact_odt_verification_dtype": "float64",
            "exact_odt_verification_device": "cpu",
            "fresh_capability_execution_dtype": "float32",
            "fresh_capability_execution_device": "cuda",
            "float32_checkpoint_coefficients_exactly_representable_in_float64": True,
            "cross_precision_bitwise_equality_claimed": False,
        },
        "checkpoint": checkpoint,
        "inactive_vision_norm_sentinel": {
            "site": "vision.norm_out",
            "armed_through_full_rank": True,
            "call_count": inactive_norm_counter["calls"],
            "total_site_count": 74,
            "inactive_sites": ["vision.norm_out"],
            "active_site_count": len(active_norm_calls),
            "active_call_count": sum(active_norm_calls.values()),
            "every_active_site_call_count": 2,
        },
        "capability_evidence": capability,
        "bounded_clone_and_regression_attestation": unit,
        "current_combined_unit_attestation": current_unit,
        "previous_exact_checkpoint_control": r7_control,
        "dooms_reference_sha256": inputs[
            "reference/dooms_xnets_2504.02667.pdf"
        ],
        "dooms_algorithm_order": {
            "algorithm1": "child-before-parent direct local RQ, with R pushed to every occurrence",
            "algorithm2": "root-first explicit downstream environment contraction and eigendecomposition",
            "algorithm3": "resulting full-rank gauge applied to every occurrence",
        },
        "replay_panel": replay_panel,
        "source_replay": source_replay,
        "structure": structure,
        "initial_shape": initial_shape,
        "canonical_shape": canonical_shape,
        "route_inventory": route_inventory,
        "algorithm1": algorithm1,
        "algorithms2_and3": {
            "algorithm2_child_messages": diagonal.algorithm2_child_messages,
            "algorithm3_diagonalized_nodes": diagonal.diagonalized_node_count,
            "algorithm3_occurrence_ledger": [
                diagonal.pushed_parent_occurrences,
                diagonal.expected_parent_occurrences,
            ],
            "algorithm3_replay_relative_error": diagonal.replay_relative_error,
            "maximum_recontracted_offdiagonal_ratio": (
                diagonal.maximum_recontracted_offdiagonal_ratio
            ),
            "full_spectra_retained": diagonal.eigenvalue_spectra_retained,
            "post_environment_records_retained": False,
        },
        "timings_seconds": {
            "compile": compile_seconds,
            "source_evaluation": source_evaluation_seconds,
            "algorithm1": algorithm1_seconds,
            "algorithms2_and3": algorithms2_and3_seconds,
            "through_full_rank": time.perf_counter() - started_total,
        },
        "peak_rss_mb": _peak_rss_mb(),
        "full_rank_gates": full_rank_gates,
        "all_full_rank_gates_pass": True,
        "runtime_guard": runtime_full,
        "static_audit": _STATIC_AUDIT,
        "source_manifest": manifest,
        "bounded_clone_oracle_hash_joined": True,
        "canonical_direct_odt_algorithms_1_to_3_completed": True,
        "exact_learned_neural_dag_global_decomposability_certified": True,
        "historical_capability_path_association_recorded": (
            capability["successes"] == 419
        ),
        "historical_capability_sha256_bound_to_current_checkpoint": False,
        "compression_claimed": False,
    }
    _assert_sources()
    _assert_current_unit_unchanged(current_unit)
    full_rank_sha = _publish(FULL_RANK_OUTPUT, full_rank_certificate)
    try:
        _assert_current_unit_unchanged(current_unit)
    except BaseException:
        os.replace(
            FULL_RANK_OUTPUT,
            FULL_RANK_OUTPUT.with_name(f"{FULL_RANK_OUTPUT.name}.invalid.{os.getpid()}"),
        )
        raise

    compact_census = bank.finish()
    full_storage = implicit_storage_elements(diagonal.network)
    curves: dict[str, Any] = {}
    for target in COMPACT_TRACE_RETENTION_TARGETS:
        plan = bank.plan(target)
        projected_storage = projected_prefix_storage_elements(diagonal, plan)
        started = time.perf_counter()
        prefix = evaluate_diagonal_prefix_quotient(diagonal, raw, plan)
        prefix_seconds = time.perf_counter() - started
        prefix_actions = prefix.reshape_as(source_actions)
        curve: dict[str, Any] = {
            "target": target,
            "retained_bond_dimensions": sum(plan[uid] for uid in plan),
            "original_bond_dimensions": sum(plan.expected_dimension(uid) for uid in plan),
            "projected_storage_elements": projected_storage,
            "projected_storage_fraction": projected_storage / full_storage,
            "prefix_observable_relative_error": _relative(prefix, source_observable),
            "prefix_action_relative_error": _relative(prefix_actions, source_actions),
            "prefix_action_coordinate_max_absolute_error": [
                float(value)
                for value in (prefix_actions - source_actions)
                .abs()
                .amax(dim=(0, 1))
                .detach()
                .cpu()
                .tolist()
            ],
            "prefix_evaluation_seconds": prefix_seconds,
        }
        started = time.perf_counter()
        try:
            suffix = evaluate_diagonal_suffix_quotient(diagonal, raw, plan)
        except ValueError as error:
            message = str(error)
            if "denominator" not in message and "zero/nonfinite coordinates" not in message:
                raise
            curve["matched_width_trailing_control"] = {
                "valid_projective_chart": False,
                "rejection": message,
                "evaluation_seconds": time.perf_counter() - started,
            }
        else:
            curve["matched_width_trailing_control"] = {
                "valid_projective_chart": True,
                "observable_relative_error": _relative(suffix, source_observable),
                "action_relative_error": _relative(
                    suffix.reshape_as(source_actions), source_actions
                ),
                "evaluation_seconds": time.perf_counter() - started,
            }
        curves[str(target)] = curve

    physical_plan = bank.plan(PHYSICAL_TARGET)
    projected_storage = projected_prefix_storage_elements(diagonal, physical_plan)
    masked_pair = evaluate_diagonal_prefixes(diagonal, raw, physical_plan)
    masked_observable = evaluate_diagonal_prefix_quotient(diagonal, raw, physical_plan)
    masked_actions = masked_observable.reshape_as(source_actions)
    physical_phase_started = time.perf_counter()
    truncate_started = time.perf_counter()
    materialized = truncate_diagonal_prefixes(
        diagonal, physical_plan, copy_network=False
    )
    truncate_seconds = time.perf_counter() - truncate_started
    pair_started = time.perf_counter()
    materialized_pair = evaluate_projective_boundary(materialized.network, raw)
    materialized_pair_evaluation_seconds = time.perf_counter() - pair_started
    materialized_denominator = materialized_pair[:, -1]
    materialized_relative_denominator = (
        materialized_denominator.abs()
        / materialized_pair.abs().amax(dim=1).clamp_min(
            torch.finfo(materialized_pair.dtype).tiny
        )
    )
    if float(materialized_relative_denominator.min().item()) <= (
        100.0 * torch.finfo(materialized_pair.dtype).eps
    ):
        raise RuntimeError("materialized projective chart has a zero denominator")
    materialized_observable = (
        materialized_pair[:, :-1] / materialized_denominator[:, None]
    )
    materialized_actions = materialized_observable.reshape_as(source_actions)
    raw_sample0 = (
        {key: value[:1] for key, value in raw.items()}
        if isinstance(raw, dict)
        else raw[:1]
    )
    if (
        isinstance(raw_sample0, dict)
        and any(value.shape[0] != 1 for value in raw_sample0.values())
    ) or (not isinstance(raw_sample0, dict) and raw_sample0.shape[0] != 1):
        raise RuntimeError("actual TN executor input is not batch one")
    source_forward = model.forward

    def reject_source_model_fallback(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("actual TN executor attempted a source-model fallback")

    model.forward = reject_source_model_fallback
    executor_rss_before_mb = _peak_rss_mb()
    executor_started = time.perf_counter()
    try:
        executor_observable = evaluate_boundary_quotient(
            materialized.network, raw_sample0
        )
    finally:
        model.forward = source_forward
    executor_seconds = time.perf_counter() - executor_started
    executor_rss_after_mb = _peak_rss_mb()
    executor_actions = executor_observable.reshape(1, 8, 7)
    executor_output_shape = list(executor_observable.shape)
    executor_output_finite = bool(torch.isfinite(executor_observable).all())
    actual_tn_executor = {
        "executor": "evaluate_boundary_quotient",
        "network": "physically sliced canonical direct-ODT .999999999 network",
        "replay_panel_sample_index": 0,
        "batch_size": 1,
        "elapsed_seconds": executor_seconds,
        "peak_rss_before_mb": executor_rss_before_mb,
        "peak_rss_after_mb": executor_rss_after_mb,
        "peak_rss_delta_mb": max(
            0.0, executor_rss_after_mb - executor_rss_before_mb
        ),
        "projective_pair_source": (
            "not exposed by the timed quotient call; the separate physical-pair gate "
            "is reported on the parent materialization"
        ),
        "output_shape": executor_output_shape,
        "output_finite": executor_output_finite,
        "observable_relative_error_against_masked": _relative(
            executor_observable, masked_observable[:1]
        ),
        "action_relative_error_against_masked": _relative(
            executor_actions, masked_actions[:1]
        ),
        "source_model_forward_sentinel_armed": True,
        "source_model_fallback_observed": False,
        "network_is_in_place_materialized_canonical_network": (
            materialized.network is diagonal.network
        ),
        "serialized_artifact_claimed": False,
        "closed_loop_capability_claimed": False,
    }
    materialization = {
        "target": PHYSICAL_TARGET,
        "performed_after_all_masked_curves": True,
        "copy_network": False,
        "projective_pair_gauge_invariant_error_against_masked": (
            _projective_batch_relative_error(materialized_pair, masked_pair)
        ),
        "raw_projective_chart_relative_error_against_masked_diagnostic": _relative(
            materialized_pair, masked_pair
        ),
        "observable_relative_error_against_masked": _relative(
            materialized_observable, masked_observable
        ),
        "action_relative_error_against_masked": _relative(
            materialized_actions, masked_actions
        ),
        "original_storage_elements": materialized.original_storage_elements,
        "projected_storage_elements": projected_storage,
        "materialized_storage_elements": materialized.truncated_storage_elements,
        "allocated_storage_elements": materialized.allocated_storage_elements,
        "reduced_bonds": materialized.reduced_bonds,
        "expected_occurrence_slices": materialized.expected_occurrence_slices,
        "applied_occurrence_slices": materialized.applied_occurrence_slices,
        "truncate_seconds": truncate_seconds,
        "materialized_pair_evaluation_seconds": materialized_pair_evaluation_seconds,
        "physical_phase_total_seconds": time.perf_counter() - physical_phase_started,
        "actual_tn_executor_batch1": actual_tn_executor,
    }
    materialization_gates = {
        "projective_pair_matches_masked_up_to_rowwise_scale": materialization[
            "projective_pair_gauge_invariant_error_against_masked"
        ]
        < STRICT_COMPARISON_LIMIT,
        "observable_matches_masked": materialization[
            "observable_relative_error_against_masked"
        ]
        < STRICT_COMPARISON_LIMIT,
        "action_matches_masked": materialization["action_relative_error_against_masked"]
        < STRICT_COMPARISON_LIMIT,
        "storage_origin_matches": materialization["original_storage_elements"] == full_storage,
        "projected_materialized_allocated_equal": materialization[
            "projected_storage_elements"
        ]
        == materialization["materialized_storage_elements"]
        == materialization["allocated_storage_elements"],
        "strict_physical_storage_reduction": materialization[
            "materialized_storage_elements"
        ]
        < full_storage,
        "bond_reduction_nonempty": materialization["reduced_bonds"] > 0,
        "every_occurrence_sliced": materialization["expected_occurrence_slices"]
        == materialization["applied_occurrence_slices"]
        and materialization["applied_occurrence_slices"] > 0,
        "actual_tn_executor_matches_masked": actual_tn_executor[
            "observable_relative_error_against_masked"
        ]
        < STRICT_COMPARISON_LIMIT
        and actual_tn_executor["action_relative_error_against_masked"]
        < STRICT_COMPARISON_LIMIT
        and actual_tn_executor["output_shape"] == [1, 56]
        and actual_tn_executor["output_finite"] is True,
        "actual_tn_executor_no_source_fallback": actual_tn_executor[
            "source_model_forward_sentinel_armed"
        ]
        is True
        and actual_tn_executor["source_model_fallback_observed"] is False
        and actual_tn_executor[
            "network_is_in_place_materialized_canonical_network"
        ]
        is True,
        "inactive_vision_norm_never_executed_through_compression": (
            inactive_norm_counter["calls"] == 0
        ),
    }
    assert_full_vla_norm_buffers_unchanged(oracle)
    model_unchanged = model_state_before == _state_digest(model)
    runtime_final = assert_direct_only_runtime_guard(
        direct_only_runtime_report(), exact_allowed_calls=runtime_calls
    )
    result = {
        "schema": "xvla_capable_linear_b1c0_direct_odt_compression_v1",
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "full_rank_certificate": {
            "path": FULL_RANK_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": full_rank_sha,
        },
        "capability_evidence": capability,
        "bounded_clone_and_regression_attestation": unit,
        "current_combined_unit_attestation": current_unit,
        "inactive_vision_norm_sentinel": {
            "site": "vision.norm_out",
            "armed_through_full_exact_and_compression": True,
            "call_count": inactive_norm_counter["calls"],
            "total_site_count": 74,
            "inactive_sites": ["vision.norm_out"],
            "active_site_count": len(active_norm_calls),
            "active_call_count": sum(active_norm_calls.values()),
            "every_active_site_call_count": 2,
        },
        "compact_census": compact_census,
        "canonical_storage_elements": full_storage,
        "compression_curves": curves,
        "compression_curve_claim_boundary": (
            "Exact deterministic replay-panel function degradation, not closed-loop task degradation."
        ),
        "physical_materialization": materialization,
        "physical_materialization_gates": materialization_gates,
        "source_model_state_unchanged": model_unchanged,
        "runtime_guard_final": runtime_final,
        "static_audit": _STATIC_AUDIT,
        "source_manifest": _assert_sources(),
        "timings_seconds": {
            **full_rank_certificate["timings_seconds"],
            "total": time.perf_counter() - started_total,
        },
        "peak_rss_mb": _peak_rss_mb(),
        "all_full_rank_gates_pass": True,
        "all_compression_gates_pass": all(materialization_gates.values())
        and model_unchanged
        and runtime_final["prohibited_attempt_count"] == 0,
        "exact_learned_neural_dag_global_decomposability_certified_before_compression": True,
        "closed_loop_task_degradation_measured": False,
    }
    _assert_current_unit_unchanged(current_unit)
    digest = _publish(COMPRESSION_OUTPUT, result)
    try:
        _assert_current_unit_unchanged(current_unit)
    except BaseException:
        os.replace(
            COMPRESSION_OUTPUT,
            COMPRESSION_OUTPUT.with_name(
                f"{COMPRESSION_OUTPUT.name}.invalid.{os.getpid()}"
            ),
        )
        raise
    print(json.dumps({"compression_result_sha256": digest}, sort_keys=True), flush=True)
    if not result["all_compression_gates_pass"]:
        raise SystemExit(1)


def _fresh_norm_sentinel_is_exact(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("site") == "vision.norm_out"
        and value.get(
            "armed_during_sentinel_replay_and_all_rollout_policy_forwards"
        )
        is True
        and value.get("call_count") == 0
        and value.get("total_site_count") == 74
        and value.get("inactive_sites") == ["vision.norm_out"]
        and value.get("active_site_count") == 73
        and value.get("active_call_count") == 73
        and value.get("every_active_site_call_count") == 1
        and value.get("source_output_bitwise_equal_without_sentinels") is True
    )


def _fresh_aggregate_norm_sentinel_is_exact(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("site") == "vision.norm_out"
        and value.get("total_site_count") == 74
        and value.get("inactive_sites") == ["vision.norm_out"]
        and value.get("active_site_count") == 73
        and value.get("shard_count") == 4
        and value.get("all_shard_call_counts_zero") is True
        and value.get("all_shard_active_sets_observed_once") is True
    )


def _fresh_capability_attestation(manifest: dict[str, Any]) -> dict[str, Any]:
    fresh_manifest = _verify_line_manifest(
        FRESH_CAPABILITY_SOURCE_MANIFEST,
        "fresh capability source manifest",
        "scripts/run_capable_linear_fresh_capability.py",
    )
    fresh_manifest_sha256 = fresh_manifest["sha256"]
    stage_ledger = _verify_line_manifest(STAGE_LEDGER, "stage ledger")
    if stage_ledger["source_sha256"].get(
        FRESH_CAPABILITY_SOURCE_MANIFEST.relative_to(PROJECT_ROOT).as_posix()
    ) != fresh_manifest_sha256:
        raise RuntimeError("stage ledger does not bind the fresh capability manifest")
    if fresh_manifest_sha256 == manifest["sha256"]:
        raise RuntimeError("distinct ODT and simulator closures unexpectedly share a digest")
    fresh_entrypoints = (
        PROJECT_ROOT / "scripts/run_capable_linear_fresh_capability.py",
        PROJECT_ROOT / "tests/test_capable_linear_fresh_capability.py",
    )
    fresh_static_audit = audit_direct_only_launch(
        PROJECT_ROOT, fresh_entrypoints, require_direct_qr=False
    )
    if fresh_static_audit["source_sha256"] != fresh_manifest["source_sha256"]:
        raise RuntimeError("fresh simulator manifest differs from its static source closure")
    for field in (
        "prohibited_calls_found",
        "prohibited_self_overlap_sites",
        "guarded_dormant_spectral_norm_sites",
        "duplicate_top_level_definition_sites",
    ):
        if fresh_static_audit[field]:
            raise RuntimeError(f"fresh simulator static audit did not close {field}")
    smoke, smoke_sha256 = read_authenticated_json(
        FRESH_CAPABILITY_SMOKE_OUTPUT, "fresh capability smoke"
    )
    smoke_episodes = smoke.get("episodes")
    smoke_episode = (
        smoke_episodes[0]
        if isinstance(smoke_episodes, list)
        and len(smoke_episodes) == 1
        and isinstance(smoke_episodes[0], dict)
        else {}
    )
    smoke_success = (
        int(smoke_episode.get("success"))
        if type(smoke_episode.get("success")) is bool
        else -1
    )
    smoke_protocol = smoke.get("protocol")
    smoke_gates = {
        "schema": smoke.get("schema")
        == "xvla_capable_linear_b1c0_fresh_capability_smoke_v1",
        "checkpoint": smoke.get("checkpoint_sha256")
        == EXPECTED_INPUT_SHA256["inputs/capable_linear_b1c0_checkpoint.pt"],
        "training_record": smoke.get("training_record_sha256")
        == EXPECTED_INPUT_SHA256["inputs/capable_linear_training.json"],
        "config": smoke.get("checkpoint", {}).get("config_identity")
        == EXPECTED_CONFIG_IDENTITY,
        "range_and_seed": (smoke.get("task_start"), smoke.get("task_end"), smoke.get("seed"))
        == (0, 1, 0),
        "protocol": isinstance(smoke_protocol, dict)
        and smoke_protocol.get("suite") == "libero_object"
        and smoke_protocol.get("resolution") == 64
        and smoke_protocol.get("action_horizon") == 8
        and smoke_protocol.get("episodes_per_task") == 1
        and smoke_protocol.get("max_steps") == 1
        and smoke_protocol.get("settle_steps") == 10
        and smoke_protocol.get("settle_success_or_done_count") == 0
        and smoke_protocol.get("execution_horizon") == 8
        and smoke_protocol.get("terminal_transition_stops_episode") is True
        and smoke_protocol.get("cuda_matmul_allow_tf32") is False
        and smoke_protocol.get("cudnn_allow_tf32") is True
        and smoke_protocol.get("cublas_workspace_config") == ":4096:8"
        and smoke_protocol.get("deterministic_algorithms") is True
        and smoke_protocol.get("cudnn_benchmark") is False
        and smoke_protocol.get("cudnn_deterministic") is True
        and smoke_protocol.get("historical_evaluator_control_flow_reproduced")
        is False
        and smoke_protocol.get("historical_protocol_with_corrected_terminal_handling")
        is True
        and smoke_protocol.get("historical_hardware_reproduction_claimed") is False
        and smoke_protocol.get("fresh_hardware") == "NVIDIA A30",
        "task_protocol": smoke.get("task_protocol")
        == {"0": _expected_capability_task_record(0)},
        "episode": isinstance(smoke_episodes, list)
        and len(smoke_episodes) == 1
        and isinstance(smoke_episode, dict)
        and smoke_episode.get("task_index") == 0
        and smoke_episode.get("episode") == 0
        and type(smoke_episode.get("success")) is bool
        and type(smoke_episode.get("steps")) is int
        and smoke_episode.get("steps") == 1
        and type(smoke_episode.get("terminated_without_success")) is bool
        and not (
            smoke_episode.get("terminated_without_success")
            and smoke_episode.get("success")
        )
        and isinstance(smoke_episode.get("initial_state_sha256"), str)
        and len(smoke_episode.get("initial_state_sha256")) == 64
        and all(
            character in "0123456789abcdef"
            for character in smoke_episode.get("initial_state_sha256")
        )
        and isinstance(smoke_episode.get("elapsed_s"), (int, float))
        and not isinstance(smoke_episode.get("elapsed_s"), bool)
        and math.isfinite(float(smoke_episode.get("elapsed_s")))
        and smoke_episode.get("elapsed_s") >= 0,
        "counts": smoke.get("trials") == 1
        and smoke.get("successes") == smoke_success
        and smoke.get("overall") == smoke_success
        and smoke.get("per_task") == {"0": float(smoke_success)},
        "episode_digest": smoke.get("episode_identity_sha256")
        == _canonical_sha256([(0, 0)]),
        "environment": _fresh_environment_is_exact(smoke.get("environment")),
        "source": smoke.get("source_manifest_sha256") == fresh_manifest_sha256
        and smoke.get("stage_ledger_sha256") == _sha256(STAGE_LEDGER)
        and smoke.get("static_audit") == fresh_static_audit,
        "producer_checks": smoke.get("all_identity_and_protocol_gates_pass") is True
        and smoke.get("exact_checkpoint_digest_verified_before_and_after") is True
        and smoke.get("stage_ledger_verified_before_and_after") is True
        and smoke.get("smoke_only") is True,
        "simulator_boundary": smoke.get("simulator_external_to_odt") is True
        and smoke.get("external_simulator_outside_weight_only_odt_closure") is True
        and smoke.get("canonical_odt_runtime_guard_installed") is False
        and smoke.get("odt_runtime_compliance_claimed") is False,
        "inactive_norm_sentinel": _fresh_norm_sentinel_is_exact(
            smoke.get("inactive_vision_norm_sentinel")
        ),
    }
    failed_smoke = sorted(name for name, passed in smoke_gates.items() if not passed)
    if failed_smoke:
        raise RuntimeError(f"fresh capability smoke attestation failed: {failed_smoke}")
    aggregate, aggregate_sha256 = read_authenticated_json(
        FRESH_CAPABILITY_OUTPUT, "fresh capability aggregate"
    )
    aggregate_path_snapshot = {"aggregate": aggregate_sha256}
    shard_ranges = ((0, 3), (3, 6), (6, 8), (8, 10))
    aggregate_shards = aggregate.get("shards")
    shard_map = {}
    if isinstance(aggregate_shards, list):
        for record in aggregate_shards:
            if isinstance(record, dict):
                key = (record.get("task_start"), record.get("task_end"))
                if key in shard_map:
                    raise RuntimeError("fresh aggregate contains a duplicate shard")
                shard_map[key] = record
    if (
        not isinstance(aggregate_shards, list)
        or len(aggregate_shards) != len(shard_ranges)
        or not all(isinstance(record, dict) for record in aggregate_shards)
        or len(shard_map) != len(shard_ranges)
        or set(shard_map) != set(shard_ranges)
    ):
        raise RuntimeError("fresh aggregate shard inventory is not the exact four-way cover")
    all_episodes: list[dict[str, Any]] = []
    observed_shards: dict[str, str] = {}
    shard_gates: dict[str, bool] = {}
    for start, stop in shard_ranges:
        path = RESULT_DIRECTORY / f"fresh_capability_t{start}_{stop}.json"
        value, digest = read_authenticated_json(
            path, f"fresh capability shard {start}:{stop}"
        )
        record = shard_map.get((start, stop), {})
        episodes = value.get("episodes")
        expected_pairs = [
            (task_index, episode)
            for task_index in range(start, stop)
            for episode in range(50)
        ]
        observed_pairs: list[tuple[int, int]] = []
        fields_valid = isinstance(episodes, list) and len(episodes) == len(
            expected_pairs
        )
        if fields_valid:
            for episode_record in episodes:
                if not isinstance(episode_record, dict):
                    fields_valid = False
                    break
                task_index = episode_record.get("task_index")
                episode = episode_record.get("episode")
                success = episode_record.get("success")
                steps = episode_record.get("steps")
                terminated_without_success = episode_record.get(
                    "terminated_without_success"
                )
                initial_sha = episode_record.get("initial_state_sha256")
                elapsed = episode_record.get("elapsed_s")
                if (
                    type(task_index) is not int
                    or type(episode) is not int
                    or type(success) is not bool
                    or type(steps) is not int
                    or type(terminated_without_success) is not bool
                    or (terminated_without_success and success)
                    or not 0 <= steps <= 280
                    or not isinstance(initial_sha, str)
                    or len(initial_sha) != 64
                    or any(c not in "0123456789abcdef" for c in initial_sha)
                    or not isinstance(elapsed, (int, float))
                    or isinstance(elapsed, bool)
                    or not math.isfinite(float(elapsed))
                    or elapsed < 0
                ):
                    fields_valid = False
                    break
                observed_pairs.append((task_index, episode))
        expected_task_protocol = {
            str(task_index): _expected_capability_task_record(task_index)
            for task_index in range(start, stop)
        }
        successes = (
            sum(int(item["success"]) for item in episodes)
            if fields_valid and isinstance(episodes, list)
            else -1
        )
        expected_per_task = (
            {
                str(task_index): sum(
                    int(item["success"])
                    for item in episodes
                    if item["task_index"] == task_index
                )
                / 50
                for task_index in range(start, stop)
            }
            if fields_valid and isinstance(episodes, list)
            else {}
        )
        gates = {
            "schema": value.get("schema")
            == "xvla_capable_linear_b1c0_fresh_capability_shard_v1",
            "checkpoint": value.get("checkpoint_sha256")
            == EXPECTED_INPUT_SHA256[
                "inputs/capable_linear_b1c0_checkpoint.pt"
            ],
            "training_record": value.get("training_record_sha256")
            == EXPECTED_INPUT_SHA256["inputs/capable_linear_training.json"],
            "config_identity": value.get("checkpoint", {}).get("config_identity")
            == EXPECTED_CONFIG_IDENTITY,
            "range": (value.get("task_start"), value.get("task_end"))
            == (start, stop),
            "seed": value.get("seed") == 0,
            "protocol": value.get("protocol", {}).get("suite") == "libero_object"
            and value.get("protocol", {}).get("resolution") == 64
            and value.get("protocol", {}).get("action_horizon") == 8
            and value.get("protocol", {}).get("episodes_per_task") == 50
            and value.get("protocol", {}).get("max_steps") == 280
            and value.get("protocol", {}).get("settle_steps") == 10
            and value.get("protocol", {}).get("settle_success_or_done_count") == 0
            and value.get("protocol", {}).get("execution_horizon") == 8
            and value.get("protocol", {}).get("canonical_init_states") is True
            and value.get("protocol", {}).get("observation_transform")
            == "agentview_image[::-1,::-1] then same-size PIL resize"
            and value.get("protocol", {}).get("gripper_decode")
            == "+1 iff predicted coordinate > 0, else -1"
            and value.get("protocol", {}).get("matmul_precision") == "highest"
            and value.get("protocol", {}).get("cuda_matmul_allow_tf32") is False
            and value.get("protocol", {}).get("cudnn_allow_tf32") is True
            and value.get("protocol", {}).get("cublas_workspace_config") == ":4096:8"
            and value.get("protocol", {}).get("deterministic_algorithms") is True
            and value.get("protocol", {}).get("cudnn_benchmark") is False
            and value.get("protocol", {}).get("cudnn_deterministic") is True
            and value.get("protocol", {}).get("terminal_transition_stops_episode")
            is True
            and value.get("protocol", {}).get(
                "historical_protocol_with_corrected_terminal_handling"
            )
            is True
            and value.get("protocol", {}).get(
                "historical_evaluator_control_flow_reproduced"
            )
            is False
            and value.get("protocol", {}).get(
                "historical_hardware_reproduction_claimed"
            )
            is False
            and value.get("protocol", {}).get("fresh_hardware") == "NVIDIA A30",
            "task_protocol": value.get("task_protocol")
            == expected_task_protocol,
            "episodes": fields_valid and observed_pairs == expected_pairs,
            "episode_digest": value.get("episode_identity_sha256")
            == _canonical_sha256(expected_pairs),
            "counts": value.get("successes") == successes
            and value.get("trials") == len(expected_pairs)
            and value.get("per_task") == expected_per_task
            and value.get("overall") == successes / len(expected_pairs),
            "early_terminal_count": value.get("early_terminal_failures")
            == (
                sum(
                    int(item["terminated_without_success"])
                    for item in episodes
                )
                if fields_valid and isinstance(episodes, list)
                else -1
            ),
            "identity": value.get("source_manifest_sha256")
            == fresh_manifest_sha256
            and value.get("stage_ledger_sha256") == _sha256(STAGE_LEDGER)
            and value.get("static_audit") == fresh_static_audit
            and value.get("exact_checkpoint_digest_verified_before_and_after")
            is True
            and value.get("stage_ledger_verified_before_and_after") is True,
            "environment": _fresh_environment_is_exact(value.get("environment")),
            "smoke": value.get("smoke_gate_sha256") == smoke_sha256,
            "simulator_boundary": value.get("simulator_external_to_odt") is True
            and value.get("external_simulator_outside_weight_only_odt_closure")
            is True
            and value.get("canonical_odt_runtime_guard_installed") is False
            and value.get("odt_runtime_compliance_claimed") is False,
            "aggregate_record": record.get("sha256") == digest
            and record.get("successes") == successes
            and record.get("trials") == len(expected_pairs),
            "producer_self_gate": value.get("all_identity_and_protocol_gates_pass")
            is True,
            "inactive_norm_sentinel": _fresh_norm_sentinel_is_exact(
                value.get("inactive_vision_norm_sentinel")
            ),
        }
        for name, passed in gates.items():
            shard_gates[f"shard_{start}_{stop}_{name}"] = passed
        if isinstance(episodes, list):
            all_episodes.extend(episodes)
        observed_shards[f"{start}:{stop}"] = digest

    shard_paths = {
        key: RESULT_DIRECTORY / f"fresh_capability_t{key.replace(':', '_')}.json"
        for key in observed_shards
    }
    assert_physical_hashes_unchanged(shard_paths, observed_shards)
    assert_physical_hashes_unchanged(
        {"aggregate": FRESH_CAPABILITY_OUTPUT}, aggregate_path_snapshot
    )
    assert_physical_hashes_unchanged(
        {"smoke": FRESH_CAPABILITY_SMOKE_OUTPUT}, {"smoke": smoke_sha256}
    )

    expected_all_pairs = [
        (task_index, episode)
        for task_index in range(10)
        for episode in range(50)
    ]
    observed_all_pairs = [
        (item.get("task_index"), item.get("episode")) for item in all_episodes
    ]
    successes = sum(int(item.get("success") is True) for item in all_episodes)
    expected_per_task_successes = {
        str(task_index): sum(
            int(item.get("success") is True)
            for item in all_episodes
            if item.get("task_index") == task_index
        )
        for task_index in range(10)
    }
    expected_per_task = {
        key: value / 50 for key, value in expected_per_task_successes.items()
    }
    gates = {
        **shard_gates,
        "schema": aggregate.get("schema")
        == "xvla_capable_linear_b1c0_fresh_capability_aggregate_v1",
        "aggregate_protocol": aggregate.get("protocol", {}).get("suite")
        == "libero_object"
        and aggregate.get("protocol", {}).get("resolution") == 64
        and aggregate.get("protocol", {}).get("action_horizon") == 8
        and aggregate.get("protocol", {}).get(
            "settle_success_or_done_count"
        )
        == 0
        and aggregate.get("protocol", {}).get("tasks") == list(range(10))
        and aggregate.get("protocol", {}).get(
            "canonical_initial_states_per_task"
        )
        == list(range(50))
        and aggregate.get("protocol", {}).get("canonical_init_states") is True
        and aggregate.get("protocol", {}).get("episodes_per_task") == 50
        and aggregate.get("protocol", {}).get("trials") == 500
        and aggregate.get("protocol", {}).get("max_steps") == 280
        and aggregate.get("protocol", {}).get("settle_steps") == 10
        and aggregate.get("protocol", {}).get("execution_horizon") == 8
        and aggregate.get("protocol", {}).get("observation_transform")
        == "agentview_image[::-1,::-1] then same-size PIL resize"
        and aggregate.get("protocol", {}).get("gripper_decode")
        == "+1 iff predicted coordinate > 0, else -1"
        and aggregate.get("protocol", {}).get("matmul_precision") == "highest"
        and aggregate.get("protocol", {}).get("cuda_matmul_allow_tf32") is False
        and aggregate.get("protocol", {}).get("cudnn_allow_tf32") is True
        and aggregate.get("protocol", {}).get("deterministic_algorithms") is True
        and aggregate.get("protocol", {}).get("cublas_workspace_config")
        == ":4096:8"
        and aggregate.get("protocol", {}).get("cudnn_benchmark") is False
        and aggregate.get("protocol", {}).get("cudnn_deterministic") is True,
        "aggregate_terminal_and_hardware_boundary": aggregate.get("protocol", {}).get(
            "terminal_transition_stops_episode"
        )
        is True
        and aggregate.get("protocol", {}).get(
            "historical_protocol_with_corrected_terminal_handling"
        )
        is True
        and aggregate.get("protocol", {}).get(
            "historical_evaluator_control_flow_reproduced"
        )
        is False
        and aggregate.get("protocol", {}).get(
            "historical_hardware_reproduction_claimed"
        )
        is False
        and aggregate.get("protocol", {}).get("fresh_hardware") == "NVIDIA A30",
        "checkpoint": aggregate.get("checkpoint_sha256")
        == EXPECTED_INPUT_SHA256["inputs/capable_linear_b1c0_checkpoint.pt"],
        "training_record": aggregate.get("training_record_sha256")
        == EXPECTED_INPUT_SHA256["inputs/capable_linear_training.json"],
        "config_identity": aggregate.get("config_identity")
        == EXPECTED_CONFIG_IDENTITY,
        "sha256_binding": aggregate.get(
            "fresh_evaluation_sha256_bound_to_checkpoint"
        )
        is True,
        "historical_boundary": aggregate.get(
            "historical_evaluation_digest_binding_claimed"
        )
        is False,
        "complete_inventory": observed_all_pairs == expected_all_pairs
        and len(set(observed_all_pairs)) == 500,
        "aggregate_episode_inventory": aggregate.get("episodes") == all_episodes,
        "episode_digest": aggregate.get("episode_identity_sha256")
        == _canonical_sha256(expected_all_pairs),
        "counts": aggregate.get("successes") == successes
        and aggregate.get("trials") == 500
        and math.isclose(
            aggregate.get("success_rate", -1.0),
            successes / 500,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and aggregate.get("per_task_successes") == expected_per_task_successes
        and aggregate.get("per_task") == expected_per_task,
        "early_terminal_count": aggregate.get("early_terminal_failures")
        == sum(
            int(item.get("terminated_without_success") is True)
            for item in all_episodes
        ),
        "capability_floor": aggregate.get("capability_floor")
        == CAPABILITY_FLOOR
        and aggregate.get("capability_floor_passed") is True
        and successes >= 400,
        "shard_bundle": aggregate.get("shard_result_bundle_sha256")
        == _canonical_sha256(observed_shards),
        "smoke": aggregate.get("smoke_gate_sha256") == smoke_sha256,
        "environment": _fresh_environment_is_exact(aggregate.get("environment")),
        "aggregate_source": aggregate.get("source_manifest_sha256")
        == fresh_manifest_sha256
        and aggregate.get("stage_ledger_sha256") == _sha256(STAGE_LEDGER)
        and aggregate.get("static_audit") == fresh_static_audit,
        "inactive_norm_sentinel": _fresh_aggregate_norm_sentinel_is_exact(
            aggregate.get("inactive_vision_norm_sentinel")
        ),
        "aggregate_simulator_boundary": aggregate.get("simulator_external_to_odt")
        is True
        and aggregate.get("external_simulator_outside_weight_only_odt_closure")
        is True
        and aggregate.get("canonical_odt_runtime_guard_installed") is False
        and aggregate.get("odt_runtime_compliance_claimed") is False,
        "aggregate_self_gate": aggregate.get("all_gates_pass") is True
        and aggregate.get("fresh_all_a30_hash_bound_reevaluation") is True
        and aggregate.get("historical_hardware_reproduction_claimed") is False,
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    if failed:
        raise RuntimeError(f"fresh capability attestation failed: {failed}")
    return {
        "path": FRESH_CAPABILITY_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": aggregate_sha256,
        "checkpoint_sha256": aggregate["checkpoint_sha256"],
        "successes": successes,
        "trials": 500,
        "success_rate": successes / 500,
        "capability_floor": CAPABILITY_FLOOR,
        "config_identity": aggregate["config_identity"],
        "sha256_bound_to_current_checkpoint": True,
        "source_manifest_sha256": fresh_manifest_sha256,
        "source_manifest": fresh_manifest,
        "shard_sha256": observed_shards,
        "smoke_sha256": smoke_sha256,
        "gates": gates,
    }


def _composite_mode() -> None:
    _assert_runtime()
    manifest = _assert_sources()
    inputs = _verify_fixed_inputs()
    capability = _capability_attestation()
    fresh_capability = _fresh_capability_attestation(manifest)
    unit = _unit_attestation()
    current_unit = _current_unit_attestation(manifest)
    r7_control = _r7_control_attestation()
    odt_result_paths = {
        "full_rank": FULL_RANK_OUTPUT,
        "compression": COMPRESSION_OUTPUT,
    }
    odt_result_hashes = assert_physical_hashes_unchanged(odt_result_paths)
    full_rank, full_rank_read_sha = read_authenticated_json(
        FULL_RANK_OUTPUT, "full-rank certificate"
    )
    compression, compression_read_sha = read_authenticated_json(
        COMPRESSION_OUTPUT, "compression result"
    )
    if {
        "full_rank": full_rank_read_sha,
        "compression": compression_read_sha,
    } != odt_result_hashes:
        raise RuntimeError("ODT result fields and digests came from different bytes")
    assert_physical_hashes_unchanged(odt_result_paths, odt_result_hashes)
    full_rank_sha = odt_result_hashes["full_rank"]
    compression_sha = odt_result_hashes["compression"]
    gates = {
        "checkpoint_identity": full_rank.get("checkpoint", {}).get("checkpoint_sha256")
        == compression.get("checkpoint_sha256")
        == EXPECTED_INPUT_SHA256["inputs/capable_linear_b1c0_checkpoint.pt"],
        "historical_capability_artifacts_consistent": full_rank.get(
            "capability_evidence", {}
        ).get("successes")
        == compression.get("capability_evidence", {}).get("successes")
        == capability["successes"]
        == 419,
        "official_capability_protocol": capability["trials"] == 500
        and math.isclose(capability["success_rate"], 0.838, abs_tol=0.0, rel_tol=0.0),
        "fresh_capability_checkpoint_identity": fresh_capability[
            "checkpoint_sha256"
        ]
        == EXPECTED_INPUT_SHA256["inputs/capable_linear_b1c0_checkpoint.pt"],
        "fresh_and_exact_config_identity": fresh_capability["config_identity"]
        == full_rank.get("checkpoint", {}).get("config_identity")
        == EXPECTED_CONFIG_IDENTITY,
        "fresh_capability_official_protocol": fresh_capability["trials"] == 500,
        "fresh_capability_floor": fresh_capability["success_rate"]
        >= CAPABILITY_FLOOR,
        "fresh_capability_sha256_binding": fresh_capability[
            "sha256_bound_to_current_checkpoint"
        ]
        is True,
        "bounded_oracle_join": full_rank.get(
            "bounded_clone_and_regression_attestation", {}
        ).get("sha256")
        == compression.get("bounded_clone_and_regression_attestation", {}).get("sha256")
        == unit["sha256"],
        "current_unit_join": full_rank.get(
            "current_combined_unit_attestation", {}
        ).get("sha256")
        == compression.get("current_combined_unit_attestation", {}).get("sha256")
        == current_unit["sha256"]
        and full_rank.get("current_combined_unit_attestation", {}).get(
            "direct_result_sha256"
        )
        == compression.get("current_combined_unit_attestation", {}).get(
            "direct_result_sha256"
        )
        == current_unit["direct_result_sha256"],
        "full_rank_exact": full_rank.get("all_full_rank_gates_pass") is True
        and full_rank.get("canonical_direct_odt_algorithms_1_to_3_completed") is True
        and full_rank.get(
            "exact_learned_neural_dag_global_decomposability_certified"
        )
        is True,
        "exact_normalization_inventory_and_dead_site": full_rank.get(
            "checkpoint", {}
        ).get("normalization_site_count")
        == 74
        and full_rank.get("checkpoint", {}).get(
            "active_normalization_site_count"
        )
        == 73
        and full_rank.get("checkpoint", {}).get(
            "compiler_active_normalization_site_count"
        )
        == 73
        and full_rank.get("checkpoint", {}).get(
            "compiler_active_snapshot_site_count"
        )
        == 73
        and full_rank.get("checkpoint", {}).get(
            "compiler_active_set_equals_loaded_active_set"
        )
        is True
        and full_rank.get("checkpoint", {}).get(
            "normalization_buffer_inventory", {}
        ).get("site_count")
        == 74
        and full_rank.get("checkpoint", {}).get(
            "normalization_buffer_inventory", {}
        ).get("initialized_true_count")
        == 73
        and full_rank.get("checkpoint", {}).get(
            "normalization_buffer_inventory", {}
        ).get("initialized_false_sites")
        == ["vision.norm_out"]
        and full_rank.get("checkpoint", {}).get(
            "normalization_buffer_inventory", {}
        ).get("inactive_running_ms")
        == {"vision.norm_out": 1.0}
        and full_rank.get("checkpoint", {}).get(
            "normalization_buffer_inventory", {}
        ).get("all_running_ms_scalar_and_finite")
        is True
        and full_rank.get("checkpoint", {}).get(
            "normalization_buffer_inventory", {}
        ).get("all_active_running_ms_positive")
        is True
        and full_rank.get("checkpoint", {}).get(
            "normalization_buffer_inventory", {}
        ).get("all_pade_coefficients_match_frozen_defaults")
        is True
        and full_rank.get("inactive_vision_norm_sentinel", {}).get("site")
        == "vision.norm_out"
        and full_rank.get("inactive_vision_norm_sentinel", {}).get(
            "armed_through_full_rank"
        )
        is True
        and full_rank.get("inactive_vision_norm_sentinel", {}).get("call_count")
        == 0
        and full_rank.get("inactive_vision_norm_sentinel", {}).get(
            "total_site_count"
        )
        == 74
        and full_rank.get("inactive_vision_norm_sentinel", {}).get(
            "inactive_sites"
        )
        == ["vision.norm_out"]
        and full_rank.get("inactive_vision_norm_sentinel", {}).get(
            "active_site_count"
        )
        == 73
        and full_rank.get("inactive_vision_norm_sentinel", {}).get(
            "active_call_count"
        )
        == 146
        and full_rank.get("inactive_vision_norm_sentinel", {}).get(
            "every_active_site_call_count"
        )
        == 2
        and compression.get("inactive_vision_norm_sentinel", {}).get("site")
        == "vision.norm_out"
        and compression.get("inactive_vision_norm_sentinel", {}).get(
            "armed_through_full_exact_and_compression"
        )
        is True
        and compression.get("inactive_vision_norm_sentinel", {}).get("call_count")
        == 0
        and compression.get("inactive_vision_norm_sentinel", {}).get(
            "total_site_count"
        )
        == 74
        and compression.get("inactive_vision_norm_sentinel", {}).get(
            "inactive_sites"
        )
        == ["vision.norm_out"]
        and compression.get("inactive_vision_norm_sentinel", {}).get(
            "active_site_count"
        )
        == 73
        and compression.get("inactive_vision_norm_sentinel", {}).get(
            "active_call_count"
        )
        == 146
        and compression.get("inactive_vision_norm_sentinel", {}).get(
            "every_active_site_call_count"
        )
        == 2,
        "full_and_compression_source_identity": full_rank.get("source_manifest")
        == manifest
        and compression.get("source_manifest") == manifest
        and full_rank.get("static_audit") == _STATIC_AUDIT
        and compression.get("static_audit") == _STATIC_AUDIT,
        "numerical_precision_boundary": full_rank.get("numerical_precision_boundary")
        == {
            "checkpoint_storage_dtype": "float32",
            "exact_odt_verification_dtype": "float64",
            "exact_odt_verification_device": "cpu",
            "fresh_capability_execution_dtype": "float32",
            "fresh_capability_execution_device": "cuda",
            "float32_checkpoint_coefficients_exactly_representable_in_float64": True,
            "cross_precision_bitwise_equality_claimed": False,
        },
        "compression_links_full_rank": compression.get("full_rank_certificate", {}).get(
            "sha256"
        )
        == full_rank_sha,
        "compression_gates": compression.get("all_compression_gates_pass") is True,
        "functional_claim_boundary": compression.get("closed_loop_task_degradation_measured")
        is False,
        "historical_capability_not_misrepresented": capability[
            "historical_checkpoint_digest_recorded"
        ]
        is False
        and capability["sha256_bound_to_current_checkpoint"] is False
        and full_rank.get(
            "historical_capability_sha256_bound_to_current_checkpoint"
        )
        is False,
        "r7_regression_control": r7_control["canonical_odt_certified"] is True,
        "source_closure_unchanged": manifest == _PREIMPORT_MANIFEST,
    }
    runtime = assert_direct_only_runtime_guard(
        direct_only_runtime_report(), exact_allowed_calls=set()
    )
    gates["runtime_guard_exact"] = runtime["prohibited_attempt_count"] == 0
    exact_odt_attested = all(
        gates[name]
        for name in (
            "checkpoint_identity",
            "bounded_oracle_join",
            "current_unit_join",
            "full_rank_exact",
            "r7_regression_control",
            "source_closure_unchanged",
            "full_and_compression_source_identity",
            "numerical_precision_boundary",
            "exact_normalization_inventory_and_dead_site",
            "runtime_guard_exact",
        )
    )
    fresh_uncompressed_capability_attested = all(
        gates[name]
        for name in (
            "fresh_capability_checkpoint_identity",
            "fresh_and_exact_config_identity",
            "fresh_capability_official_protocol",
            "fresh_capability_floor",
            "fresh_capability_sha256_binding",
        )
    )
    compression_materialization_attested = all(
        gates[name]
        for name in (
            "checkpoint_identity",
            "full_rank_exact",
            "compression_links_full_rank",
            "compression_gates",
            "functional_claim_boundary",
            "source_closure_unchanged",
            "full_and_compression_source_identity",
            "exact_normalization_inventory_and_dead_site",
            "runtime_guard_exact",
        )
    )
    result = {
        "schema": "xvla_capable_linear_b1c0_direct_odt_composite_v1",
        "claim": (
            "The present b1c0 checkpoint's complete active 3,964,463-node learned neural DAG, "
            "from unit-scaled patch, categorical instruction, and normalized-state "
            "coordinates to normalized continuous action coordinates, "
            "completed canonical direct Algorithms 1-3, hash-joined to the bounded no-memo "
            "clone and adversarial regression suite, before any compression evaluation. "
            "A fresh 500-episode official-protocol evaluation is SHA-256-bound to "
            "those same checkpoint bytes and passes the preregistered 80 percent capability "
            "floor. Historical 419/500 records are retained only as unbound comparison. Fixed "
            "leading-spectrum functional curves and one physical materialization were then "
            "evaluated."
        ),
        "caveats": {
            "literal_full_policy_clone_expansion_performed": False,
            "arbitrary_size_or_rank_theorem_claimed": False,
            "closed_loop_task_degradation_measured": False,
            "compression_trace_targets_are_global_error_bounds": False,
            "historical_capability_sha256_bound_to_current_checkpoint": False,
            "fresh_capability_simulator_inside_weight_only_odt_closure": False,
            "compressed_policy_closed_loop_capability_measured": False,
            "fresh_capability_applies_to_uncompressed_checkpoint_only": True,
            "fixed_rollout_affines_and_image_permutation_contracted": False,
            "instruction_text_tokenization_contracted": False,
            "gripper_sign_decode_is_external": True,
            "persisted_deployable_tensor_network_artifact": False,
            "float32_cuda_and_float64_cpu_arithmetic_bitwise_equal_claimed": False,
            "checkpoint_resident_vision_norm_out_is_inactive_and_outside_chivla_forward": True,
            "checkpoint_resident_vision_classifier_head_is_outside_chivla_forward_and_compiled_dag": True,
        },
        "identities": {
            "checkpoint_sha256": inputs["inputs/capable_linear_b1c0_checkpoint.pt"],
            "unit_attestation_sha256": unit["sha256"],
            "current_unit_attestation_sha256": current_unit["sha256"],
            "current_direct_test_result_sha256": current_unit[
                "direct_result_sha256"
            ],
            "full_rank_certificate_sha256": full_rank_sha,
            "compression_result_sha256": compression_sha,
            "fresh_capability_aggregate_sha256": fresh_capability["sha256"],
            "source_manifest_sha256": manifest["sha256"],
            "fresh_capability_source_manifest_sha256": fresh_capability[
                "source_manifest_sha256"
            ],
            "dooms_reference_sha256": inputs[
                "reference/dooms_xnets_2504.02667.pdf"
            ],
        },
        "capability": capability,
        "fresh_capability": fresh_capability,
        "previous_exact_checkpoint_control": r7_control,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "current_checkpoint_learned_neural_dag_global_decomposability_attested": (
            exact_odt_attested
        ),
        "historical_capability_path_association_attested": gates[
            "historical_capability_artifacts_consistent"
        ]
        and gates["official_capability_protocol"]
        and gates["historical_capability_not_misrepresented"],
        "fresh_uncompressed_capability_attested": fresh_uncompressed_capability_attested,
        "capable_vla_learned_neural_dag_global_decomposability_attested": (
            exact_odt_attested and fresh_uncompressed_capability_attested
        ),
        "compression_materialization_attested": compression_materialization_attested,
        "compressed_policy_capability_attested": False,
        "runtime_guard": runtime,
        "static_audit": _STATIC_AUDIT,
    }
    fresh_paths = {
        "aggregate": FRESH_CAPABILITY_OUTPUT,
        "smoke": FRESH_CAPABILITY_SMOKE_OUTPUT,
        **{
            key: RESULT_DIRECTORY
            / f"fresh_capability_t{key.replace(':', '_')}.json"
            for key in fresh_capability["shard_sha256"]
        },
    }
    fresh_hashes = {
        "aggregate": fresh_capability["sha256"],
        "smoke": fresh_capability["smoke_sha256"],
        **fresh_capability["shard_sha256"],
    }
    assert_physical_hashes_unchanged(fresh_paths, fresh_hashes)
    assert_physical_hashes_unchanged(odt_result_paths, odt_result_hashes)
    _assert_current_unit_unchanged(current_unit)
    digest = _publish(COMPOSITE_OUTPUT, result)
    try:
        assert_physical_hashes_unchanged(fresh_paths, fresh_hashes)
    except BaseException:
        quarantine = COMPOSITE_OUTPUT.with_name(
            f"{COMPOSITE_OUTPUT.name}.invalid.{os.getpid()}"
        )
        os.replace(COMPOSITE_OUTPUT, quarantine)
        raise
    try:
        assert_physical_hashes_unchanged(odt_result_paths, odt_result_hashes)
        _assert_current_unit_unchanged(current_unit)
    except BaseException:
        quarantine = COMPOSITE_OUTPUT.with_name(
            f"{COMPOSITE_OUTPUT.name}.invalid.odt.{os.getpid()}"
        )
        os.replace(COMPOSITE_OUTPUT, quarantine)
        raise
    print(json.dumps({"composite_sha256": digest}, sort_keys=True), flush=True)
    if not result["all_gates_pass"]:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--mode", choices=("preflight", "full", "composite"), required=True
    )
    arguments = sys.argv[1:]
    parsed = parser.parse_args(arguments)
    count = sum(value == "--mode" or value.startswith("--mode=") for value in arguments)
    if count != 1:
        parser.error("--mode must occur exactly once")
    return parsed


def main() -> None:
    args = _parse_args()
    if args.mode == "preflight":
        _preflight_mode()
    elif args.mode == "full":
        _full_mode()
    else:
        _composite_mode()


if __name__ == "__main__":
    main()
