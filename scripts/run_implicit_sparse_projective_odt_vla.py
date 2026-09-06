#!/usr/bin/env python3
"""Run the one-root heterogeneous canonical ODT path for an unchanged ChiVLA."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _preimport_lane(arguments: list[str]) -> str:
    """Parse the exact lane spelling before importing any project module."""

    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument == "--lane":
            if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
                raise RuntimeError("--lane requires one value")
            values.append(arguments[index + 1])
        elif argument.startswith("--lane="):
            value = argument.split("=", 1)[1]
            if not value:
                raise RuntimeError("--lane requires one value")
            values.append(value)
        elif argument.startswith("--lan"):
            raise RuntimeError("abbreviated --lane options are forbidden")
    if len(values) > 1:
        raise RuntimeError("--lane may be specified at most once")
    lane = values[0] if values else "tiny"
    if lane not in {"tiny", "production"}:
        raise RuntimeError(f"invalid --lane value: {lane!r}")
    return lane


_REQUESTED_LANE = _preimport_lane(sys.argv[1:])

EXPECTED_CHECKPOINT_SHA256 = (
    "96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9"
)
EXPECTED_SOURCE_MANIFEST = (
    PROJECT_ROOT / "athena/implicit_sparse_projective_odt_vla_sources.sha256"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_expected_source_manifest(
    manifest_path: Path = EXPECTED_SOURCE_MANIFEST,
) -> dict[str, object]:
    """Verify every frozen source using only the standard library."""

    if not manifest_path.is_file():
        raise RuntimeError(f"pinned source manifest is missing: {manifest_path}")
    expected: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        manifest_path.read_text().splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(
                f"malformed source manifest line {line_number}"
            )
        digest, relative = fields
        relative = relative.lstrip("*")
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(
                f"invalid SHA-256 on source manifest line {line_number}"
            )
        path = (PROJECT_ROOT / relative).resolve()
        if path == PROJECT_ROOT or PROJECT_ROOT not in path.parents:
            raise RuntimeError("source manifest path escapes the project root")
        normalized = path.relative_to(PROJECT_ROOT).as_posix()
        if normalized in expected:
            raise RuntimeError("source manifest contains a duplicate path")
        expected[normalized] = digest
    runner_relative = Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix()
    if runner_relative not in expected:
        raise RuntimeError("source manifest does not authenticate the runner itself")
    actual: dict[str, str] = {}
    for relative, expected_digest in sorted(expected.items()):
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"frozen source is missing: {relative}")
        actual_digest = _sha256(path)
        if actual_digest != expected_digest:
            raise RuntimeError(f"frozen source hash changed: {relative}")
        actual[relative] = actual_digest
    return {
        "path": manifest_path.resolve().as_posix(),
        "sha256": _sha256(manifest_path),
        "source_count": len(actual),
        "source_sha256": actual,
    }


_PREIMPORT_SOURCE_VERIFICATION = (
    _verify_expected_source_manifest()
    if _REQUESTED_LANE == "production"
    else None
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

_TRANSITIVE_STATIC_AUDIT = audit_direct_only_launch(
    PROJECT_ROOT,
    canonical_direct_only_entrypoints(PROJECT_ROOT, Path(__file__)),
)
if _PREIMPORT_SOURCE_VERIFICATION is not None and (
    _PREIMPORT_SOURCE_VERIFICATION["source_sha256"]
    != _TRANSITIVE_STATIC_AUDIT["source_sha256"]
):
    raise RuntimeError(
        "pinned source manifest differs from the authoritative import closure"
    )
_RUNTIME_GUARD_AT_IMPORT = install_direct_only_runtime_guard()

import torch
import torch.nn as nn

from scripts.run_production_factorized_odt_stress import production_config
from xvla.models.vla import ChiVLA, VLAConfig
from xvla.nn.normalization import RationalNorm
from xvla.train.implicit_sparse_projective_odt import (
    DIRECT_Q_PROVENANCE_TOLERANCE,
    DIRECT_RQ_METHOD,
    LOCAL_SCALE_CERTIFICATION_TOLERANCE,
    RECTANGULAR_RQ_METHOD,
    STREAMED_DIRECT_Q_PROVENANCE_METHOD,
    UNARY_RQ_METHOD,
    audit_algorithm1_factorization_calls,
    canonicalize_implicit_dag_direct_rq,
    diagonalize_implicit_dag_full_rank,
    implicit_shape_statistics,
    predict_direct_rq_route_inventory,
    reverse_implicit_environments,
    telemetry_dict,
    validate_canonical_exponent_normal_form,
)
from xvla.train.implicit_sparse_projective_odt_vla import (
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

def _state_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = expected.norm().clamp_min(torch.finfo(expected.dtype).tiny)
    return float(((actual - expected).norm() / scale).item())


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _assert_frozen_sources_unchanged() -> dict[str, object] | None:
    if _PREIMPORT_SOURCE_VERIFICATION is None:
        return None
    current = _verify_expected_source_manifest()
    if current != _PREIMPORT_SOURCE_VERIFICATION:
        raise RuntimeError("production source manifest or source bytes drifted during execution")
    if current["source_sha256"] != _TRANSITIVE_STATIC_AUDIT["source_sha256"]:
        raise RuntimeError("production source closure drifted after the static audit")
    return current


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_progress(
    args: argparse.Namespace,
    result: dict[str, object],
    gates: dict[str, bool],
    milestone: str,
) -> None:
    if args.lane != "production":
        return
    verification = _assert_frozen_sources_unchanged()
    payload = dict(result)
    payload["progress_milestone"] = milestone
    payload["completed_gates"] = dict(gates)
    payload["all_completed_gates_pass"] = all(gates.values())
    # A source replay or Algorithm-1 milestone is useful diagnostic evidence,
    # but it is never a certificate for the complete Dooms procedure.
    payload["canonical_odt_certified"] = False
    payload["all_gates_pass"] = False
    payload["certification_status"] = (
        f"{milestone}_diagnostic_passed"
        if payload["all_completed_gates_pass"]
        else "failed"
    )
    payload["source_manifest_at_milestone"] = verification
    progress_path = args.output.with_name(args.output.name + ".progress.json")
    _atomic_write_json(progress_path, payload)


def _initialize_norms(model: ChiVLA) -> None:
    with torch.no_grad():
        for index, module in enumerate(
            item for item in model.modules() if isinstance(item, RationalNorm)
        ):
            module.running_ms.fill_(0.81 + 0.027 * index)
            module.initialized.fill_(True)
            module.frozen = True


def _tiny(seed: int, samples: int, action_head: str = "linear"):
    torch.manual_seed(seed)
    model = ChiVLA(
        VLAConfig(
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
            attn="bilinear",
            vit_attn="bilinear",
            ffn="bilinear",
            norm="rational",
            qk_norm="rational",
            residual=True,
            vit_residual=True,
            action_horizon=1,
            action_dim=2,
            action_head=action_head,
            head_rank=3,
            n_factors=2,
        )
    ).to(dtype=DTYPE).eval()
    _initialize_norms(model)
    generator = torch.Generator().manual_seed(seed + 1700)
    images = torch.randn(samples, 3, 1, 1, dtype=DTYPE, generator=generator) * 0.13
    instructions = torch.randint(0, 2, (samples, 1), generator=generator)
    state = torch.randn(samples, 1, dtype=DTYPE, generator=generator) * 0.17
    embodiments = torch.zeros(samples, dtype=torch.long)
    return model, images, instructions, state, embodiments, None


def _production(args: argparse.Namespace):
    if args.checkpoint is None:
        raise ValueError("production lane requires --checkpoint")
    checkpoint_sha = _sha256(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA-256 differs from the pinned capable checkpoint")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vocab_size = int(state_dict["tok_emb.weight"].shape[0])
    model = ChiVLA(production_config(vocab_size))
    model.load_state_dict(state_dict, strict=True)
    device = torch.device(args.device)
    model = model.to(device=device, dtype=DTYPE).eval()
    generator = torch.Generator(device=device).manual_seed(args.seed + 2900)
    images = torch.randn(
        args.samples,
        model.cfg.vit_config().in_chans,
        model.cfg.image_size,
        model.cfg.image_size,
        dtype=DTYPE,
        device=device,
        generator=generator,
    ) * 0.035
    instructions = torch.randint(
        0,
        vocab_size,
        (args.samples, model.cfg.max_instr_len),
        device=device,
        generator=generator,
    )
    state = torch.randn(
        args.samples,
        model.cfg.state_dim,
        dtype=DTYPE,
        device=device,
        generator=generator,
    ) * 0.11
    embodiments = torch.zeros(args.samples, dtype=torch.long, device=device)
    return model, images, instructions, state, embodiments, checkpoint_sha


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--lane", choices=("tiny", "production"), default="tiny")
    parser.add_argument(
        "--tiny-action-head", choices=("linear", "product"), default="linear"
    )
    parser.add_argument(
        "--stage", choices=("replay", "algorithm1", "full"), default="replay"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--progress-every-steps", type=int, default=100_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.progress_every_steps < 1:
        raise ValueError("--progress-every-steps must be positive")

    direct_audit = audit_algorithm1_factorization_calls()
    if args.lane == "production" and args.tiny_action_head != "linear":
        raise ValueError("--tiny-action-head applies only to the tiny lane")
    loaded = (
        _tiny(args.seed, args.samples, args.tiny_action_head)
        if args.lane == "tiny"
        else _production(args)
    )
    model, images, instructions, state, embodiments, checkpoint_sha = loaded
    state_before = _state_digest(model)
    physical = physical_batch_from_model_inputs(
        model, images, instructions, state, embodiments
    )
    started = time.perf_counter()
    oracle = compile_full_vla_projective_dag(model)
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    compiled_observable = evaluate_full_vla_observable(oracle, physical)
    source_observable = source_full_vla_observable(oracle, physical)
    observable_error = _relative(compiled_observable, source_observable)
    compiled = evaluate_full_vla_quotient(oracle, physical)
    evaluate_seconds = time.perf_counter() - started
    source = source_full_vla_output(model, physical)
    source_error = _relative(compiled, source)
    structure = full_vla_structure_statistics(oracle)
    initial_shape = implicit_shape_statistics(oracle.network)
    route_inventory = predict_direct_rq_route_inventory(oracle.network)
    meter = telemetry_dict(oracle.telemetry)
    expected_source_count = 97 if args.lane == "production" else 3
    expected_patch_width = 192 if args.lane == "production" else 3
    expected_pade_sites = 73 if args.lane == "production" else 13
    result = {
        "schema": "xvla-full-heterogeneous-shared-dag-odt-v1",
        "claim_boundary": oracle.network.claim_boundary,
        "lane": args.lane,
        "action_head": model.cfg.action_head,
        "stage": args.stage,
        "checkpoint_sha256": checkpoint_sha,
        "source_replay_relative_error": source_error,
        "source_observable_relative_error": observable_error,
        "compile_seconds": compile_seconds,
        "evaluate_seconds": evaluate_seconds,
        "peak_rss_mb": _peak_rss_mb(),
        "structure": structure,
        "implicit_shape": initial_shape,
        "algorithm1_exact_predicted_route_inventory": route_inventory,
        "telemetry": meter,
        "direct_call_audit": direct_audit,
        "transitive_static_audit": _TRANSITIVE_STATIC_AUDIT,
        "runtime_direct_only_guard_at_import": _RUNTIME_GUARD_AT_IMPORT,
        "immutable_source_manifest": _PREIMPORT_SOURCE_VERIFICATION,
        "state_sha256_before": state_before,
    }
    widths = structure["physical_source_widths"]
    expected_root_width = oracle.observable_euclidean_width + 1
    gates = {
        "exact_unchanged_chivla_forward_replay": source_error < 3e-8,
        "exact_source_tensor_observable_replay": observable_error < 3e-8,
        "one_exact_policy_observable_root": structure["root_projective_width"]
        == expected_root_width,
        "one_leaf_per_heterogeneous_source": structure["physical_source_count"]
        == structure["physical_leaf_count"]
        == expected_source_count,
        "no_joint_width_input_padding": structure["maximum_physical_source_width"]
        == expected_patch_width
        and all(width != model.cfg.dim for width in widths.values()),
        "all_active_pade_sites_present": structure["active_pade_sites"]
        == expected_pade_sites,
        "singleton_embodiment_is_fixed": "embodiment" not in widths,
        "all_fixed_token_subgraphs_constant_folded": (
            meter["constant_unary_folds"] > 0
            and meter["constant_cp_to_constant_folds"] > 0
            and meter["constant_cp_to_unary_folds"] > 0
        ),
        "production_graph_contains_no_dense_clone_core": initial_shape[
            "dense_clone_nodes"
        ]
        == 0,
        "algorithm1_exact_route_prediction_is_internally_consistent": (
            route_inventory["schema"]
            == "exact_postorder_structural_direct_rq_routes_v1"
            and route_inventory["simulated_unique_node_count"]
            == initial_shape["unique_nodes"]
            and route_inventory["bounded_retained_q_candidate_count"]
            + route_inventory["streamed_cp_candidate_count"]
            == initial_shape["cp_binary_nodes"]
            and route_inventory[
                "bounded_retained_q_signed_storage_delta_elements"
            ]
            == route_inventory["bounded_retained_q_total_elements"]
            - route_inventory["bounded_retained_q_replaced_cp_total_elements"]
            and route_inventory[
                "bounded_retained_q_positive_storage_delta_elements"
            ]
            >= 0
            and route_inventory[
                "bounded_retained_q_maximum_unfolding_elements"
            ]
            <= route_inventory["bounded_explicit_unfolding_element_limit"]
            and route_inventory["bounded_retained_q_maximum_elements"]
            <= route_inventory["bounded_explicit_q_element_limit"]
            and route_inventory["streamed_tall_unrepresentable_count"] == 0
        ),
        "direct_ODT_module_static_audit": not direct_audit["prohibited_calls_found"],
        "transitive_direct_only_static_audit": not _TRANSITIVE_STATIC_AUDIT[
            "prohibited_calls_found"
        ],
        "pinned_capable_checkpoint": args.lane == "tiny"
        or checkpoint_sha == EXPECTED_CHECKPOINT_SHA256,
    }
    exact_runtime_calls: set[str] = set()
    if args.lane == "production":
        gates["immutable_source_manifest_matches_static_closure"] = (
            _PREIMPORT_SOURCE_VERIFICATION is not None
            and _PREIMPORT_SOURCE_VERIFICATION["source_sha256"]
            == _TRANSITIVE_STATIC_AUDIT["source_sha256"]
        )
    _write_progress(args, result, gates, "source_replay_complete")

    if args.stage != "replay":
        raw = physical_source_mapping(model, physical)
        production_algorithm = args.lane == "production"
        started = time.perf_counter()

        def algorithm1_progress_callback(_work, node, _factor, record):
            completed = record.step + 1
            total = initial_shape["unique_nodes"]
            if (
                completed % args.progress_every_steps != 0
                and completed != total
            ):
                return
            result["algorithm1_live_progress"] = {
                "completed_steps": completed,
                "total_steps": total,
                "elapsed_seconds": time.perf_counter() - started,
                "last_uid": node.uid,
                "last_label": node.label,
                "last_method": record.method,
                "last_unfolding_shape": list(record.unfolding_shape),
                "last_direct_q_provenance_method": (
                    record.direct_q_provenance_method
                ),
                "last_direct_q_columns_compared": (
                    record.direct_q_columns_compared
                ),
                "peak_rss_mb": _peak_rss_mb(),
            }
            _write_progress(
                args,
                result,
                gates,
                f"algorithm1_step_{completed}_of_{total}",
            )

        canonical = canonicalize_implicit_dag_direct_rq(
            oracle.network,
            replay_inputs=raw,
            block_size=args.block_size,
            replay_each_step=not production_algorithm,
            copy_network=not production_algorithm,
            step_callback=(
                algorithm1_progress_callback if production_algorithm else None
            ),
        )
        canonical_seconds = time.perf_counter() - started
        shape = implicit_shape_statistics(canonical.network)
        performed_step_replays = tuple(
            step.per_step_function_replay_error
            for step in canonical.steps
            if step.per_step_function_replay_performed
        )
        maximum_step_replay = (
            max(performed_step_replays) if performed_step_replays else None
        )
        maximum_local_reconstruction = max(
            (
                step.local_scaled_reconstruction_relative_error
                for step in canonical.steps
            ),
            default=0.0,
        )
        maximum_absorption_error = max(
            (
                step.maximum_absorption_scaled_relative_error
                for step in canonical.steps
            ),
            default=0.0,
        )
        canonical_normal_form = validate_canonical_exponent_normal_form(
            canonical.network
        )
        streamed_steps = tuple(
            step for step in canonical.steps if step.method == DIRECT_RQ_METHOD
        )
        exact_runtime_calls.add(DIRECT_RQ_RUNTIME_CALL)
        if streamed_steps:
            exact_runtime_calls.update(
                {STREAMED_QR_RUNTIME_CALL, TRIANGULAR_RUNTIME_CALL}
            )
        expected_direct_q_columns = sum(
            int(step.unfolding_shape[1]) for step in canonical.steps
        )
        maximum_direct_q_compact_error = max(
            (step.direct_q_compact_relative_error for step in canonical.steps),
            default=0.0,
        )
        maximum_direct_q_reconstruction_error = max(
            (
                step.direct_q_reconstruction_relative_error
                for step in canonical.steps
            ),
            default=0.0,
        )
        provenance_steps_verified = sum(
            step.direct_q_provenance_verified for step in canonical.steps
        )
        streamed_provenance_steps = sum(
            step.direct_q_provenance_method
            == STREAMED_DIRECT_Q_PROVENANCE_METHOD
            for step in canonical.steps
        )
        expected_streamed_replay_qr_calls = sum(
            step.direct_q_provenance_stage_count for step in streamed_steps
        )
        result["boundary_algorithms"] = {
            "canonical_seconds": canonical_seconds,
            "algorithm1_maximum_step_replay_relative_error": maximum_step_replay,
            "algorithm1_final_projective_replay_relative_error": (
                canonical.final_projective_replay_error
            ),
            "algorithm1_full_projective_evaluations": (
                canonical.full_projective_replay_evaluations
            ),
            "algorithm1_per_step_replays_performed": len(
                performed_step_replays
            ),
            "algorithm1_maximum_local_scaled_reconstruction_relative_error": (
                maximum_local_reconstruction
            ),
            "algorithm1_maximum_absorption_scaled_relative_error": (
                maximum_absorption_error
            ),
            "algorithm1_local_reconstruction_maximum_exponent_delta": max(
                (
                    abs(step.local_reconstruction_exponent_delta)
                    for step in canonical.steps
                ),
                default=0,
            ),
            "algorithm1_absorption_maximum_exponent_delta": max(
                (
                    abs(step.maximum_absorption_exponent_delta)
                    for step in canonical.steps
                ),
                default=0,
            ),
            "algorithm1_scale_sensitive_occurrence_ledger": [
                sum(
                    step.scale_sensitive_occurrences_verified
                    for step in canonical.steps
                ),
                sum(
                    step.parent_occurrences_pushed
                    for step in canonical.steps
                ),
            ],
            "algorithm1_canonical_exponent_normal_form": canonical_normal_form,
            "algorithm1_direct_q_provenance": {
                "verified_steps": provenance_steps_verified,
                "expected_steps": len(canonical.steps),
                "streamed_certificates": streamed_provenance_steps,
                "streamed_steps": len(streamed_steps),
                "streamed_replay_qr_calls": (
                    canonical.telemetry.streamed_direct_q_replay_qr_factorizations
                ),
                "expected_streamed_replay_qr_calls": (
                    expected_streamed_replay_qr_calls
                ),
                "columns_compared": sum(
                    step.direct_q_columns_compared for step in canonical.steps
                ),
                "expected_columns": expected_direct_q_columns,
                "maximum_compact_q_relative_error": (
                    maximum_direct_q_compact_error
                ),
                "maximum_direct_q_reconstruction_relative_error": (
                    maximum_direct_q_reconstruction_error
                ),
            },
            "algorithm1_in_place": production_algorithm,
            "factorization_methods": sorted({step.method for step in canonical.steps}),
            "rectangular_direct_rq_steps": sum(
                step.structurally_reduced for step in canonical.steps
            ),
            "canonical_telemetry": telemetry_dict(canonical.telemetry),
            "algorithm1_push_ledger": [
                sum(step.parent_occurrences_pushed for step in canonical.steps),
                sum(step.expected_parent_occurrences for step in canonical.steps),
            ],
        }
        gates.update(
            {
                "all_nodes_use_direct_reduced_RQ": all(
                    step.method
                    in {DIRECT_RQ_METHOD, UNARY_RQ_METHOD, RECTANGULAR_RQ_METHOD}
                    for step in canonical.steps
                ),
                "algorithm1_final_projective_replay": (
                    canonical.final_projective_replay_error < 3e-8
                ),
                "algorithm1_local_scale_certificates": (
                    maximum_local_reconstruction
                    < LOCAL_SCALE_CERTIFICATION_TOLERANCE
                    and maximum_absorption_error
                    < LOCAL_SCALE_CERTIFICATION_TOLERANCE
                    and all(
                        step.local_reconstruction_exponent_delta == 0
                        and step.maximum_absorption_exponent_delta == 0
                        and step.scale_sensitive_occurrences_verified
                        == step.parent_occurrences_pushed
                        for step in canonical.steps
                    )
                ),
                "algorithm1_replay_schedule_is_lane_appropriate": (
                    len(performed_step_replays) == 0
                    and canonical.full_projective_replay_evaluations == 2
                    if production_algorithm
                    else len(performed_step_replays) == len(canonical.steps)
                ),
                "algorithm1_canonical_exponent_normal_form": (
                    canonical_normal_form["all_core_exponents_zero"]
                    and canonical_normal_form["head_exponent_matches_ledger"]
                ),
                "algorithm1_every_step_has_direct_q_provenance": (
                    provenance_steps_verified == len(canonical.steps)
                    and all(
                        step.direct_q_provenance_method
                        and step.direct_q_columns_compared
                        == int(step.unfolding_shape[1])
                        and step.direct_q_compact_relative_error
                        <= DIRECT_Q_PROVENANCE_TOLERANCE
                        and step.direct_q_reconstruction_relative_error
                        <= DIRECT_Q_PROVENANCE_TOLERANCE
                        and step.direct_q_conditioning_accepted
                        for step in canonical.steps
                    )
                ),
                "algorithm1_streamed_direct_q_provenance_ledger": (
                    streamed_provenance_steps == len(streamed_steps)
                    and all(
                        step.direct_q_provenance_stage_count
                        == step.direct_q_provenance_column_blocks_compared
                        == step.streamed_column_blocks
                        and step.direct_q_minimum_pivot_to_maximum_entry
                        > step.direct_q_conditioning_threshold
                        and (
                            step.direct_q_retained_transition_elements > 0
                        )
                        == (step.direct_q_provenance_stage_count > 1)
                        for step in streamed_steps
                    )
                ),
                "algorithm1_direct_q_telemetry_ledger": (
                    canonical.telemetry.direct_q_provenance_certificates
                    == len(canonical.steps)
                    and canonical.telemetry.streamed_direct_q_provenance_certificates
                    == len(streamed_steps)
                    and canonical.telemetry.direct_q_columns_compared
                    == expected_direct_q_columns
                    and canonical.telemetry.streamed_direct_q_replay_qr_factorizations
                    == expected_streamed_replay_qr_calls
                ),
                "algorithm1_pushes_every_occurrence": all(
                    step.parent_occurrences_pushed == step.expected_parent_occurrences
                    for step in canonical.steps
                ),
                "heterogeneous_ingress_exercises_rectangular_direct_RQ": sum(
                    step.structurally_reduced for step in canonical.steps
                )
                > 0,
            }
        )
        _write_progress(args, result, gates, "algorithm1_complete")
        if args.stage == "full":
            exact_runtime_calls.add(PRODUCTION_ALGORITHM3_RUNTIME_CALL)
            if production_algorithm:
                started = time.perf_counter()
                diagonal = diagonalize_implicit_dag_full_rank(
                    canonical.network,
                    replay_inputs=raw,
                    copy_network=False,
                    stream_pre_evd_environments=True,
                    retain_eigenvalues=False,
                    retain_post_evd_environments=False,
                )
                algorithm2_seconds = None
                algorithm3_seconds = time.perf_counter() - started
                algorithm2_child_messages = diagonal.algorithm2_child_messages
            else:
                started = time.perf_counter()
                environments = reverse_implicit_environments(canonical.network)
                algorithm2_seconds = time.perf_counter() - started
                algorithm2_child_messages = sum(
                    record.child_messages_emitted for record in environments
                )
                started = time.perf_counter()
                diagonal = diagonalize_implicit_dag_full_rank(
                    canonical.network,
                    replay_inputs=raw,
                    precomputed_environments=environments,
                )
                algorithm3_seconds = time.perf_counter() - started
                del environments
            result["boundary_algorithms"].update(
                {
                    "algorithm2_seconds": algorithm2_seconds,
                    "algorithm2_streamed_inside_algorithm3": production_algorithm,
                    "algorithm2_child_messages": algorithm2_child_messages,
                    "algorithm3_seconds": algorithm3_seconds,
                    "algorithm3_replay_relative_error": diagonal.replay_relative_error,
                    "algorithm3_offdiagonal_ratio": (
                        diagonal.maximum_recontracted_offdiagonal_ratio
                    ),
                    "algorithm3_push_ledger": [
                        diagonal.pushed_parent_occurrences,
                        diagonal.expected_parent_occurrences,
                    ],
                    "algorithm3_diagonalized_node_count": (
                        diagonal.diagonalized_node_count
                    ),
                    "algorithm3_eigenvalue_spectra_retained": (
                        diagonal.eigenvalue_spectra_retained
                    ),
                }
            )
            gates.update(
                {
                    "algorithm2_emits_every_edge_message": (
                        algorithm2_child_messages == shape["edge_occurrences"]
                    ),
                    "algorithm3_projective_replay_and_diagonal": max(
                        diagonal.replay_relative_error,
                        diagonal.maximum_recontracted_offdiagonal_ratio,
                    )
                    < 3e-8,
                    "algorithm3_pushes_every_occurrence": (
                        diagonal.pushed_parent_occurrences
                        == diagonal.expected_parent_occurrences
                        == shape["edge_occurrences"] + 1
                    ),
                    "algorithm3_memory_schedule_is_lane_appropriate": (
                        diagonal.diagonalized_node_count == shape["unique_nodes"]
                        and (
                            not diagonal.eigenvalue_spectra_retained
                            if production_algorithm
                            else diagonal.eigenvalue_spectra_retained
                        )
                    ),
                }
            )

    assert_full_vla_norm_buffers_unchanged(oracle)
    state_after = _state_digest(model)
    result["state_sha256_after"] = state_after
    gates["source_model_state_unchanged"] = state_before == state_after
    runtime = assert_direct_only_runtime_guard(
        direct_only_runtime_report(),
        exact_allowed_calls=exact_runtime_calls,
    )
    result["runtime_direct_only_guard_final"] = runtime
    gates["transitive_direct_only_runtime_audit"] = True
    result["gates"] = gates
    completed_gates_pass = all(gates.values())
    full_shared_dag_completed = args.stage == "full" and completed_gates_pass
    # The independent no-memo clone oracle is a separately frozen artifact.
    # This runner deliberately does not claim formal certification without a
    # later hash-joining composite attestation.
    canonical_odt_certified = False
    result["all_completed_gates_pass"] = completed_gates_pass
    result["full_shared_dag_algorithms_1_to_3_completed"] = (
        full_shared_dag_completed
    )
    result["bounded_independent_clone_oracle_attached"] = False
    result["canonical_odt_certified"] = canonical_odt_certified
    result["certification_status"] = (
        "full_shared_dag_algorithms_1_to_3_completed_awaiting_composite_clone_attestation"
        if full_shared_dag_completed
        else (
            "algorithm1_only_diagnostic_passed"
            if args.stage == "algorithm1" and completed_gates_pass
            else (
                "source_replay_only_diagnostic_passed"
                if args.stage == "replay" and completed_gates_pass
                else "failed"
            )
        )
    )
    # Backward-compatible key, deliberately reserved for formal certification.
    result["all_gates_pass"] = canonical_odt_certified
    result["source_manifest_before_result"] = _assert_frozen_sources_unchanged()
    _atomic_write_json(args.output, result)
    try:
        _assert_frozen_sources_unchanged()
    except Exception:
        rejected = args.output.with_name(
            f"{args.output.name}.rejected-source-drift-{time.time_ns()}"
        )
        args.output.replace(rejected)
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    if not completed_gates_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
