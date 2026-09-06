#!/usr/bin/env python3
"""Checkpoint-free scaling census for direct canonical ODT compression."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import secrets
import stat
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_SOURCE_MANIFEST = (
    PROJECT_ROOT / "athena/direct_odt_spectrum_scaling_sources.sha256"
)
_AUDIT_ONLY = sys.argv[1:] == ["--audit-only"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source_manifest(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("the pinned source manifest must be a real file")
    if path.resolve().parent != (PROJECT_ROOT / "athena").resolve(strict=True):
        raise RuntimeError("the pinned source manifest is outside athena")
    expected: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(f"malformed source manifest line {line_number}")
        digest, relative = fields
        relative = relative.lstrip("*")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise RuntimeError(f"invalid source digest on line {line_number}")
        source = PROJECT_ROOT / relative
        resolved = source.resolve(strict=True)
        if (
            source.is_symlink()
            or resolved == PROJECT_ROOT
            or PROJECT_ROOT not in resolved.parents
        ):
            raise RuntimeError("source manifest contains an unsafe path")
        normalized = resolved.relative_to(PROJECT_ROOT).as_posix()
        if normalized in expected:
            raise RuntimeError("source manifest contains a duplicate path")
        expected[normalized] = digest
    runner = Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix()
    if runner not in expected:
        raise RuntimeError("source manifest does not authenticate the scaling runner")
    actual: dict[str, str] = {}
    for relative, digest in sorted(expected.items()):
        source = PROJECT_ROOT / relative
        actual_digest = _sha256(source)
        if actual_digest != digest:
            raise RuntimeError(f"authenticated source changed: {relative}")
        actual[relative] = actual_digest
    return {
        "path": path.resolve().as_posix(),
        "sha256": _sha256(path),
        "source_count": len(actual),
        "source_sha256": actual,
    }


_PREIMPORT_MANIFEST = (
    None if _AUDIT_ONLY else _verify_source_manifest(EXPECTED_SOURCE_MANIFEST)
)

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


_ENTRYPOINTS = (
    *canonical_direct_only_entrypoints(PROJECT_ROOT, Path(__file__)),
    PROJECT_ROOT / "tests/test_direct_odt_truncation.py",
)
_STATIC_AUDIT = audit_direct_only_launch(PROJECT_ROOT, _ENTRYPOINTS)
if _STATIC_AUDIT["guarded_dormant_spectral_norm_sites"]:
    raise RuntimeError("the scaling launch closure contains a dormant blocked norm site")
if _PREIMPORT_MANIFEST is not None and (
    _PREIMPORT_MANIFEST["source_sha256"] != _STATIC_AUDIT["source_sha256"]
):
    raise RuntimeError("pinned manifest differs from the audited import closure")
if "--audit-only" in sys.argv[1:]:
    if sys.argv[1:] != ["--audit-only"]:
        raise RuntimeError("--audit-only cannot be combined with run options")
    print(json.dumps(_STATIC_AUDIT, indent=2, sort_keys=True))
    raise SystemExit(0)

_RUNTIME_AT_IMPORT = install_direct_only_runtime_guard()

import torch
import numpy as np

if sys.version_info[:2] != (3, 10):
    raise RuntimeError("the scaling runner requires the frozen Python 3.10 environment")
if np.__version__ != "1.26.4":
    raise RuntimeError("the scaling runner requires the frozen NumPy 1.26.4 environment")
if torch.__version__.split("+", 1)[0] != "2.7.1":
    raise RuntimeError("the scaling runner requires the frozen Torch 2.7.1 environment")

from xvla.models.vla import ChiVLA, VLAConfig
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
    DIRECT_RQ_METHOD,
    _projective_batch_relative_error,
    canonicalize_implicit_dag_direct_rq,
    diagonalize_implicit_dag_full_rank,
    evaluate_boundary_quotient,
    evaluate_projective_boundary,
    implicit_shape_statistics,
    predict_direct_rq_route_inventory,
)
from xvla.train.implicit_sparse_projective_odt_vla import (
    PRODUCT_COMPONENTS_OBSERVABLE,
    compile_full_vla_projective_dag,
    evaluate_full_vla_observable,
    evaluate_full_vla_quotient,
    full_vla_structure_statistics,
    physical_batch_from_model_inputs,
    physical_source_mapping,
    source_full_vla_observable,
    source_full_vla_output,
)


_SOURCE_AUDIT_AFTER_IMPORTS = audit_direct_only_launch(PROJECT_ROOT, _ENTRYPOINTS)
if _SOURCE_AUDIT_AFTER_IMPORTS != _STATIC_AUDIT:
    raise RuntimeError("the audited source closure changed while importing run code")
if _PREIMPORT_MANIFEST is None:
    raise RuntimeError("a run requires a pre-import authenticated source manifest")
if _verify_source_manifest(EXPECTED_SOURCE_MANIFEST) != _PREIMPORT_MANIFEST:
    raise RuntimeError("the source manifest changed while importing run code")


def _assert_frozen_sources_unchanged() -> dict[str, object]:
    current_audit = audit_direct_only_launch(PROJECT_ROOT, _ENTRYPOINTS)
    if current_audit != _STATIC_AUDIT:
        raise RuntimeError("the audited source closure changed during the run")
    current_manifest = _verify_source_manifest(EXPECTED_SOURCE_MANIFEST)
    if current_manifest != _PREIMPORT_MANIFEST:
        raise RuntimeError("the authenticated source closure changed during the run")
    return current_manifest


DTYPE = torch.float64
PHYSICAL_MATERIALIZATION_TARGET = 0.999


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    first = actual.reshape(-1)
    second = expected.reshape(-1)
    scale = second.norm().clamp_min(torch.finfo(second.dtype).tiny)
    return float(((first - second).norm() / scale).item())


def _config(case: str) -> VLAConfig:
    cases = {
        "tiny": dict(
            image_size=1,
            patch_size=1,
            vit_dim=2,
            vit_layers=1,
            vit_heads=1,
            vit_ffn_rank=3,
            vocab_size=2,
            max_instr_len=1,
            state_dim=1,
            n_embodiments=1,
            dim=4,
            n_layers=1,
            n_heads=1,
            ffn_rank=5,
            action_horizon=2,
            action_dim=2,
            head_rank=3,
            n_factors=2,
        ),
        "small": dict(
            image_size=2,
            patch_size=1,
            vit_dim=4,
            vit_layers=1,
            vit_heads=1,
            vit_ffn_rank=7,
            vocab_size=8,
            max_instr_len=2,
            state_dim=3,
            n_embodiments=2,
            dim=8,
            n_layers=1,
            n_heads=2,
            ffn_rank=13,
            action_horizon=2,
            action_dim=3,
            head_rank=7,
            n_factors=3,
        ),
        "medium": dict(
            image_size=4,
            patch_size=2,
            vit_dim=8,
            vit_layers=2,
            vit_heads=2,
            vit_ffn_rank=13,
            vocab_size=32,
            max_instr_len=4,
            state_dim=7,
            n_embodiments=3,
            dim=16,
            n_layers=2,
            n_heads=4,
            ffn_rank=25,
            action_horizon=4,
            action_dim=7,
            head_rank=13,
            n_factors=4,
        ),
    }
    if case not in cases:
        raise ValueError(f"unknown scaling case {case!r}")
    return VLAConfig(
        **cases[case],
        attn="bilinear",
        vit_attn="bilinear",
        ffn="bilinear",
        norm="rational",
        qk_norm="rational",
        residual=True,
        vit_residual=True,
        action_head="product",
    )


def _model_and_batch(case: str, seed: int, samples: int):
    torch.manual_seed(seed)
    model = ChiVLA(_config(case)).to(dtype=DTYPE).eval()
    with torch.no_grad():
        for index, module in enumerate(
            item for item in model.modules() if isinstance(item, RationalNorm)
        ):
            module.running_ms.fill_(0.79 + 0.017 * index)
            module.initialized.fill_(True)
            module.frozen = True
    generator = torch.Generator().manual_seed(seed + 40_000)
    images = torch.randn(
        samples,
        model.cfg.vit_config().in_chans,
        model.cfg.image_size,
        model.cfg.image_size,
        dtype=DTYPE,
        generator=generator,
    ) * 0.07
    instructions = torch.randint(
        0,
        model.cfg.vocab_size,
        (samples, model.cfg.max_instr_len),
        generator=generator,
    )
    state = torch.randn(
        samples,
        model.cfg.state_dim,
        dtype=DTYPE,
        generator=generator,
    ) * 0.11
    embodiments = torch.randint(
        0, model.cfg.n_embodiments, (samples,), generator=generator
    )
    physical = physical_batch_from_model_inputs(
        model, images, instructions, state, embodiments
    )
    return model, physical


def _decode_components(model: ChiVLA, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    head = model.product_head
    center = flat[:, : head.m]
    factor_end = head.m * (head.G + 1)
    factors = flat[:, head.m : factor_end].reshape(flat.shape[0], head.G, head.m)
    gates = flat[:, factor_end:]
    if gates.shape[1] != head.G:
        raise ValueError("all-components root has the wrong gate width")
    signs = torch.where(gates > 0, torch.ones_like(gates), -torch.ones_like(gates))
    return head.action_for_signs(center, factors, signs), signs


def _validated_output(path: Path) -> Path:
    raw = path if path.is_absolute() else PROJECT_ROOT / path
    if os.path.lexists(raw):
        raise FileExistsError(f"refusing to replace existing result {raw}")
    result_directory_path = PROJECT_ROOT / "athena/results"
    if result_directory_path.is_symlink() or not result_directory_path.is_dir():
        raise RuntimeError("the result directory must be a real directory")
    result_directory = result_directory_path.resolve(strict=True)
    resolved = raw.resolve(strict=False)
    if resolved.parent != result_directory or resolved.name in {"", ".", ".."}:
        raise ValueError("output must be a direct child of athena/results")
    return resolved


def _write(path: Path, payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    encoded_bytes = encoded.encode("utf-8")
    expected_digest = hashlib.sha256(encoded_bytes).hexdigest()
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(12)}.tmp")
    published = False
    temporary_stat: os.stat_result | None = None
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_stat = os.lstat(temporary)
        if not stat.S_ISREG(temporary_stat.st_mode):
            raise RuntimeError("exclusive result temporary is not a regular file")
        _assert_frozen_sources_unchanged()
        os.link(temporary, path, follow_symlinks=False)
        published = True
        if _sha256(path) != expected_digest:
            raise RuntimeError("published result bytes differ from the encoded payload")
        _assert_frozen_sources_unchanged()
        os.chmod(path, 0o444, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return expected_digest
    except BaseException:
        if published and temporary_stat is not None and os.path.lexists(path):
            published_stat = os.lstat(path)
            if (
                published_stat.st_dev == temporary_stat.st_dev
                and published_stat.st_ino == temporary_stat.st_ino
            ):
                path.unlink()
        raise
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--case", choices=("tiny", "small", "medium"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--post-recontract", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.block_size < 1:
        raise ValueError("samples and block size must be positive")
    args.output = _validated_output(args.output)

    result: dict[str, object] = {
        "object_kind": "direct_odt_checkpoint_free_scaling_census_v1",
        "case": args.case,
        "seed": args.seed,
        "samples": args.samples,
        "dtype": str(DTYPE),
        "frozen_runtime": {
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "executable": Path(sys.executable).resolve().as_posix(),
        },
        "static_audit": _STATIC_AUDIT,
        "runtime_guard_at_import": _RUNTIME_AT_IMPORT,
        "source_manifest_before_imports": _PREIMPORT_MANIFEST,
        "source_files": {
            path: digest
            for path, digest in _STATIC_AUDIT["source_sha256"].items()
            if path
            in {
                "xvla/train/implicit_sparse_projective_odt.py",
                "xvla/train/implicit_sparse_projective_odt_vla.py",
                "xvla/train/direct_odt_truncation.py",
                "xvla/nn/bilinear.py",
                "scripts/run_direct_odt_spectrum_scaling.py",
            }
        },
        "timings_seconds": {},
        "peak_rss_mb": {"start": _peak_rss_mb()},
    }
    started = time.perf_counter()
    model, physical = _model_and_batch(args.case, args.seed, args.samples)
    oracle = compile_full_vla_projective_dag(model)
    result["timings_seconds"]["compile"] = time.perf_counter() - started
    result["peak_rss_mb"]["after_compile"] = _peak_rss_mb()
    if oracle.observable_kind != PRODUCT_COMPONENTS_OBSERVABLE or oracle.product_signs is not None:
        raise RuntimeError("scaling census requires the all-components product root")
    result["raw_structure"] = full_vla_structure_statistics(oracle)
    result["predicted_routes"] = predict_direct_rq_route_inventory(oracle.network)
    if result["predicted_routes"]["streamed_tall_unrepresentable_count"] != 0:
        raise RuntimeError("route prediction found an unsupported local shape")

    source_components = source_full_vla_observable(oracle, physical)
    source_actions = source_full_vla_output(model, physical)
    compiled_components = evaluate_full_vla_observable(oracle, physical)
    compiled_actions = evaluate_full_vla_quotient(oracle, physical)
    source_action_flat, source_signs = _decode_components(model, source_components)
    result["source_replay"] = {
        "component_relative_error": _relative(compiled_components, source_components),
        "action_relative_error": _relative(compiled_actions, source_actions),
        "decode_crosscheck_relative_error": _relative(
            source_action_flat.reshape_as(source_actions), source_actions
        ),
    }
    if (
        result["source_replay"]["component_relative_error"] >= 3e-8
        or result["source_replay"]["action_relative_error"] >= 3e-8
        or result["source_replay"]["decode_crosscheck_relative_error"] >= 3e-12
    ):
        raise RuntimeError("source/compiler replay failed before canonicalization")

    raw = physical_source_mapping(model, physical)
    started = time.perf_counter()
    canonical = canonicalize_implicit_dag_direct_rq(
        oracle.network,
        replay_inputs=raw,
        block_size=args.block_size,
        replay_each_step=False,
        copy_network=False,
    )
    result["timings_seconds"]["algorithm1"] = time.perf_counter() - started
    result["peak_rss_mb"]["after_algorithm1"] = _peak_rss_mb()
    result["canonical_structure"] = implicit_shape_statistics(canonical.network)
    if canonical.final_projective_replay_error >= 3e-8:
        raise RuntimeError("Algorithm 1 replay failed before environment contraction")

    bank = CompactRankBank.for_network(canonical.network)
    started = time.perf_counter()
    diagonal = diagonalize_implicit_dag_full_rank(
        canonical.network,
        replay_inputs=raw,
        copy_network=False,
        stream_pre_evd_environments=True,
        retain_eigenvalues=False,
        retain_post_evd_environments=args.post_recontract,
        compact_spectrum_callback=bank,
    )
    result["timings_seconds"]["algorithms2_and3"] = time.perf_counter() - started
    result["peak_rss_mb"]["after_algorithm3"] = _peak_rss_mb()
    result["full_rank_certificate"] = {
        "algorithm1_replay_relative_error": canonical.final_projective_replay_error,
        "algorithm3_replay_relative_error": diagonal.replay_relative_error,
        "post_environment_check_performed": True,
        "post_environment_records_retained": bool(args.post_recontract),
        "post_environment_offdiagonal_ratio": (
            diagonal.maximum_recontracted_offdiagonal_ratio
        ),
        "algorithm2_child_messages": diagonal.algorithm2_child_messages,
        "diagonalized_nodes": diagonal.diagonalized_node_count,
        "occurrence_pushes": [
            diagonal.pushed_parent_occurrences,
            diagonal.expected_parent_occurrences,
        ],
        "full_spectra_retained": diagonal.eigenvalue_spectra_retained,
    }
    shape = result["canonical_structure"]
    pre_compression_gates = {
        "algorithm1_replay": canonical.final_projective_replay_error < 3e-8,
        "algorithm2_complete": diagonal.algorithm2_child_messages == shape["edge_occurrences"],
        "algorithm3_complete": (
            diagonal.diagonalized_node_count == shape["unique_nodes"]
            and diagonal.pushed_parent_occurrences
            == diagonal.expected_parent_occurrences
            == shape["edge_occurrences"] + 1
            and diagonal.replay_relative_error < 3e-8
        ),
        "algorithm3_environment_diagonal": (
            math.isfinite(diagonal.maximum_recontracted_offdiagonal_ratio)
            and diagonal.maximum_recontracted_offdiagonal_ratio < 3e-10
        ),
        "zero_spectrum_retention": (
            not diagonal.eigenvalue_spectra_retained and diagonal.eigenvalues == ()
        ),
        "compact_census_complete": bank.complete,
    }
    result["pre_compression_full_rank_gates"] = pre_compression_gates
    if not all(pre_compression_gates.values()):
        raise RuntimeError("full-rank gates failed before compression evaluation")
    result["compact_census"] = bank.finish()
    census_groups = result["compact_census"]["groups"]
    if not all(
        census_groups.get(group, {}).get("node_count", 0) > 0
        for group in ("product_center", "product_factors", "product_gates")
    ):
        raise RuntimeError("all-components census missed a product component family")
    full_storage = implicit_storage_elements(diagonal.network)
    curves: dict[str, object] = {}
    for target in COMPACT_TRACE_RETENTION_TARGETS:
        plan = bank.plan(target)
        projected_storage = projected_prefix_storage_elements(diagonal, plan)
        started = time.perf_counter()
        components = evaluate_diagonal_prefix_quotient(diagonal, raw, plan)
        seconds = time.perf_counter() - started
        actions, signs = _decode_components(model, components)
        curve: dict[str, object] = {
            "component_relative_error": _relative(components, source_components),
            "decoded_action_relative_error": _relative(actions, source_action_flat),
            "gate_sign_flip_fraction": float((signs != source_signs).double().mean().item()),
            "projected_storage_elements": projected_storage,
            "projected_storage_fraction": projected_storage / full_storage,
            "evaluation_seconds": seconds,
        }
        control_started = time.perf_counter()
        try:
            control_components = evaluate_diagonal_suffix_quotient(
                diagonal, raw, plan
            )
        except ValueError as error:
            message = str(error)
            if "denominator" not in message and "zero/nonfinite coordinates" not in message:
                raise
            curve["matched_width_trailing_control"] = {
                "valid_projective_chart": False,
                "rejection": message,
                "evaluation_seconds": time.perf_counter() - control_started,
            }
        else:
            control_actions, control_signs = _decode_components(
                model, control_components
            )
            curve["matched_width_trailing_control"] = {
                "valid_projective_chart": True,
                "component_relative_error": _relative(
                    control_components, source_components
                ),
                "decoded_action_relative_error": _relative(
                    control_actions, source_action_flat
                ),
                "gate_sign_flip_fraction": float(
                    (control_signs != source_signs).double().mean().item()
                ),
                "evaluation_seconds": time.perf_counter() - control_started,
            }
        curves[str(target)] = curve
    result["compression_curves"] = curves
    result["canonical_storage_elements"] = full_storage

    physical_plan = bank.plan(PHYSICAL_MATERIALIZATION_TARGET)
    physical_projected_storage = projected_prefix_storage_elements(
        diagonal, physical_plan
    )
    masked_pair = evaluate_diagonal_prefixes(diagonal, raw, physical_plan)
    masked_components = evaluate_diagonal_prefix_quotient(
        diagonal, raw, physical_plan
    )
    masked_actions, masked_signs = _decode_components(model, masked_components)
    materialization_started = time.perf_counter()
    materialized = truncate_diagonal_prefixes(
        diagonal,
        physical_plan,
        copy_network=False,
    )
    materialized_pair = evaluate_projective_boundary(materialized.network, raw)
    materialized_components = evaluate_boundary_quotient(materialized.network, raw)
    materialized_actions, materialized_signs = _decode_components(
        model,
        materialized_components,
    )
    physical_materialization: dict[str, object] = {
        "target": PHYSICAL_MATERIALIZATION_TARGET,
        "performed_after_all_masked_curves": True,
        "copy_network": False,
        "projective_pair_gauge_invariant_error_against_masked": (
            _projective_batch_relative_error(materialized_pair, masked_pair)
        ),
        "raw_projective_chart_relative_error_against_masked_diagnostic": _relative(
            materialized_pair,
            masked_pair,
        ),
        "component_relative_error_against_masked": _relative(
            materialized_components,
            masked_components,
        ),
        "decoded_action_relative_error_against_masked": _relative(
            materialized_actions,
            masked_actions,
        ),
        "gate_sign_mismatch_fraction_against_masked": float(
            (materialized_signs != masked_signs).double().mean().item()
        ),
        "original_storage_elements": materialized.original_storage_elements,
        "projected_storage_elements": physical_projected_storage,
        "materialized_storage_elements": materialized.truncated_storage_elements,
        "allocated_storage_elements": materialized.allocated_storage_elements,
        "reduced_bonds": materialized.reduced_bonds,
        "expected_occurrence_slices": materialized.expected_occurrence_slices,
        "applied_occurrence_slices": materialized.applied_occurrence_slices,
        "evaluation_seconds": time.perf_counter() - materialization_started,
    }
    physical_materialization_verified = (
        physical_materialization[
            "projective_pair_gauge_invariant_error_against_masked"
        ]
        < 2e-10
        and physical_materialization["component_relative_error_against_masked"] < 2e-10
        and physical_materialization["decoded_action_relative_error_against_masked"] < 2e-10
        and physical_materialization["gate_sign_mismatch_fraction_against_masked"] == 0.0
        and physical_materialization["original_storage_elements"] == full_storage
        and physical_materialization["materialized_storage_elements"]
        == physical_materialization["projected_storage_elements"]
        and physical_materialization["allocated_storage_elements"]
        == physical_materialization["projected_storage_elements"]
        and physical_materialization["materialized_storage_elements"] < full_storage
        and physical_materialization["reduced_bonds"] > 0
        and physical_materialization["expected_occurrence_slices"]
        == physical_materialization["applied_occurrence_slices"]
        and physical_materialization["applied_occurrence_slices"] > 0
    )
    result["physical_materialization"] = physical_materialization

    exact_runtime_calls = {
        DIRECT_RQ_RUNTIME_CALL,
        PRODUCTION_ALGORITHM3_RUNTIME_CALL,
    }
    if any(step.method == DIRECT_RQ_METHOD for step in canonical.steps):
        exact_runtime_calls.update({STREAMED_QR_RUNTIME_CALL, TRIANGULAR_RUNTIME_CALL})
    runtime = assert_direct_only_runtime_guard(
        direct_only_runtime_report(), exact_allowed_calls=exact_runtime_calls
    )
    result["runtime_guard_final"] = runtime
    final_manifest = _assert_frozen_sources_unchanged()
    result["source_manifest_before_result"] = final_manifest
    overlap_key = "_".join(("prohibited", "self", "overlap", "sites"))
    result["gates"] = {
        "all_components_product_root": True,
        "static_closure_clean": (
            not _STATIC_AUDIT["prohibited_calls_found"]
            and not _STATIC_AUDIT[overlap_key]
            and not _STATIC_AUDIT["guarded_dormant_spectral_norm_sites"]
        ),
        "source_components_replay": result["source_replay"]["component_relative_error"] < 3e-8,
        "source_actions_replay": result["source_replay"]["action_relative_error"] < 3e-8,
        "public_decode_crosscheck": result["source_replay"]["decode_crosscheck_relative_error"] < 3e-12,
        **pre_compression_gates,
        "product_component_census_complete": True,
        "physical_materialization_verified": physical_materialization_verified,
        "runtime_guard_clean": not runtime["prohibited_attempts"],
        "source_closure_unchanged": True,
    }
    result["all_gates_pass"] = all(result["gates"].values())
    result["peak_rss_mb"]["final"] = _peak_rss_mb()
    result["runner_source_sha256"] = _sha256(Path(__file__))
    published_digest = _write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"published_sha256={published_digest}")
    if not result["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
