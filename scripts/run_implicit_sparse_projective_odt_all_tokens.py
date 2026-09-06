#!/usr/bin/env python3
"""Run exact token-tuple compilation for tiny and production one-block scopes."""

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
    single_cli_option,
)

single_cli_option(sys.argv[1:], "--lane")
_TRANSITIVE_STATIC_AUDIT = audit_direct_only_launch(
    PROJECT_ROOT, canonical_direct_only_entrypoints(PROJECT_ROOT, Path(__file__))
)
_RUNTIME_GUARD_AT_IMPORT = install_direct_only_runtime_guard()

import torch
import torch.nn as nn

from scripts.run_production_factorized_odt_stress import (
    EXPECTED_CHECKPOINT_SHA256,
    production_config,
)
from xvla.models.vla import ChiVLA
from xvla.nn.block import ChiTransformerBlock
from xvla.nn.normalization import RationalNorm
from xvla.train.implicit_sparse_projective_odt import (
    BOUNDED_EXPLICIT_RQ_METHOD,
    DIRECT_RQ_METHOD,
    UNARY_RQ_METHOD,
    audit_algorithm1_factorization_calls,
    canonicalize_implicit_dag_direct_rq,
    diagonalize_implicit_dag_full_rank,
    evaluate_boundary_quotient,
    implicit_shape_statistics,
    reverse_implicit_environments,
    telemetry_dict,
)
from xvla.train.implicit_sparse_projective_odt_all_tokens import (
    ALL_TOKEN_CLAIM_BOUNDARY,
    all_token_structure_statistics,
    assert_all_token_norm_buffers_unchanged,
    audit_all_token_compiler_calls,
    compile_all_token_projective_stack,
    evaluate_all_token_quotients,
    evaluate_observable_concat_quotient,
    source_all_token_output,
)


DTYPE = torch.float64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _initialize_norms(block: ChiTransformerBlock) -> None:
    with torch.no_grad():
        for index, module in enumerate(
            item for item in block.modules() if isinstance(item, RationalNorm)
        ):
            module.running_ms.fill_(0.89 + 0.14 * index)
            module.initialized.fill_(True)
            module.frozen = True


def _tiny(seed: int):
    torch.manual_seed(seed)
    block = ChiTransformerBlock(
        dim=1,
        n_heads=1,
        ffn_rank=1,
        n_layers=1,
        causal=True,
        norm="rational",
        qk_norm="rational",
        residual=True,
    ).double().eval()
    generator = torch.Generator().manual_seed(1200 + seed)
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                module.weight.copy_(
                    0.08
                    + 0.12
                    * torch.rand(module.weight.shape, generator=generator, dtype=DTYPE)
                )
                if module.bias is not None:
                    module.bias.copy_(
                        -0.04
                        + 0.08
                        * torch.rand(module.bias.shape, generator=generator, dtype=DTYPE)
                    )
    _initialize_norms(block)
    positions = torch.tensor(((-0.035,), (0.052,)), dtype=DTYPE)
    raw = torch.tensor(
        ((-0.31, 0.14), (0.22, -0.27), (0.35, 0.19), (0.0, 0.0)),
        dtype=DTYPE,
    ).reshape(4, 2, 1)
    return (block,), positions, raw, (0, 1), None


def _production(args):
    if args.checkpoint is None:
        raise ValueError("production all-token lanes require --checkpoint")
    checkpoint_sha = _sha256(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA-256 differs from pinned capable checkpoint")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vocab_size = int(state["tok_emb.weight"].shape[0])
    model = ChiVLA(production_config(vocab_size))
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model = model.to(device=device, dtype=DTYPE).eval()
    if args.lane.startswith("vision"):
        available = model.vision.blocks.blocks
        positions = model.vision.pos_emb[0]
        if args.lane == "vision_depth2_final":
            blocks = (available[0], available[1])
            outputs = (positions.shape[0] - 1,)
        else:
            blocks = (available[0],)
            outputs = (
                tuple(range(positions.shape[0]))
                if args.lane == "vision_all"
                else (0, positions.shape[0] - 1)
            )
    else:
        available = model.backbone.blocks
        positions = model.pos_emb[0]
        if args.lane == "joint_depth2_final":
            blocks = (available[0], available[1])
            outputs = (positions.shape[0] - 1,)
        else:
            blocks = (available[0],)
            outputs = (
                tuple(range(positions.shape[0]))
                if args.lane == "joint_all"
                else (positions.shape[0] - 2, positions.shape[0] - 1)
            )
    generator = torch.Generator(device=device).manual_seed(args.seed + 9200)
    raw = torch.randn(
        args.samples,
        positions.shape[0],
        positions.shape[1],
        generator=generator,
        dtype=DTYPE,
        device=device,
    ) * 0.05
    return blocks, positions, raw, outputs, checkpoint_sha


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--lane",
        choices=(
            "tiny",
            "vision_subset",
            "joint_subset",
            "vision_all",
            "joint_all",
            "vision_depth2_final",
            "joint_depth2_final",
        ),
        default="tiny",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--stage", choices=("replay", "algorithm1", "full"), default="replay"
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    compiler_audit = audit_all_token_compiler_calls()
    direct_audit = audit_algorithm1_factorization_calls()
    if args.lane == "tiny":
        blocks, positions, raw, outputs, checkpoint_sha = _tiny(args.seed)
    else:
        blocks, positions, raw, outputs, checkpoint_sha = _production(args)
    if args.stage != "replay" and len(outputs) > 2:
        raise ValueError(
            "direct-RQ staging materializes one final boundary for at most two outputs"
        )
    state_owner = nn.ModuleList(list(blocks))
    state_before = _state_digest(state_owner)
    started = time.perf_counter()
    oracle = compile_all_token_projective_stack(
        blocks,
        positions,
        output_tokens=outputs,
        include_observable_boundary=args.lane == "tiny" or args.stage != "replay",
        maximum_materialized_observable_width=1024,
    )
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    compiled = evaluate_all_token_quotients(oracle, raw)
    evaluate_seconds = time.perf_counter() - started
    source = source_all_token_output(oracle, raw)
    source_error = _relative(compiled, source)
    structure = all_token_structure_statistics(oracle)
    counts = structure.pop("label_parent_occurrences")
    first_output = outputs[0]
    repeated_k_label = "block0.attention.head0.k1_norm.token0.projective_output"
    final_block_index = len(blocks) - 1
    repeated_q_label = (
        f"block{final_block_index}.attention.head0.q1_norm.token{first_output}.projective_output"
    )
    k_targets = range(positions.shape[0]) if len(blocks) > 1 else outputs
    expected_k_occurrences = sum(
        int(oracle.mask[target, 0].item()) for target in k_targets
    )
    expected_q_occurrences = int(oracle.mask[first_output].sum().item())
    meter = telemetry_dict(oracle.telemetry)
    result = {
        "schema": "xvla-implicit-all-token-projective-v1",
        "claim_boundary": ALL_TOKEN_CLAIM_BOUNDARY,
        "lane": args.lane,
        "stage": args.stage,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_trust_scope": (
            "synthetic_tiny_uncheckpointed"
            if args.lane == "tiny"
            else "fixed_capable_checkpoint_sha256"
        ),
        "output_tokens": list(outputs),
        "block_depth": len(blocks),
        "source_replay_relative_error": source_error,
        "compile_seconds": compile_seconds,
        "evaluate_seconds": evaluate_seconds,
        "peak_rss_mb": _peak_rss_mb(),
        "structure": structure,
        "selected_occurrence_counts": {
            repeated_k_label: counts[repeated_k_label],
            repeated_q_label: counts[repeated_q_label],
        },
        "telemetry": meter,
        "compiler_call_audit": compiler_audit,
        "direct_call_audit": direct_audit,
        "transitive_static_audit": _TRANSITIVE_STATIC_AUDIT,
        "runtime_direct_only_guard_at_import": _RUNTIME_GUARD_AT_IMPORT,
        "state_sha256_before": state_before,
    }
    gates = {
        "exact_source_replay_every_requested_token": source_error < 3e-8,
        "requested_output_count": compiled.shape[1] == len(outputs),
        "one_shared_raw_leaf_per_physical_token": structure["unique_physical_leaves"]
        == positions.shape[0],
        "shared_graph_contains_reused_nodes": max(counts.values(), default=0) > 1,
        "shared_K_occurrence_count": counts[repeated_k_label] == expected_k_occurrences,
        "shared_Q_occurrence_count": counts[repeated_q_label] == expected_q_occurrences,
        "shared_token_graph_contains_no_dense_clone_core": structure[
            "dense_clone_nodes"
        ]
        == 0,
        "one_fused_FFN_per_output_token": meter["fused_ffn_primitive_count"]
        == positions.shape[0] * (len(blocks) - 1) + len(outputs),
        "no_ephemeral_FFN_bond": meter["ephemeral_ffn_bond_diagonalizations"] == 0,
        "all_token_compiler_has_no_factorization_calls": not compiler_audit[
            "prohibited_calls_found"
        ],
        "direct_ODT_module_static_audit": not direct_audit["prohibited_calls_found"],
        "transitive_direct_only_static_audit": not _TRANSITIVE_STATIC_AUDIT[
            "prohibited_calls_found"
        ],
        "pinned_capable_checkpoint": (
            args.lane == "tiny" and checkpoint_sha is None
        )
        or checkpoint_sha == EXPECTED_CHECKPOINT_SHA256,
    }
    exact_runtime_calls: set[str] = set()

    if oracle.observable_network is not None:
        observable = evaluate_observable_concat_quotient(oracle, raw)
        observable_error = _relative(observable, compiled)
        result["observable_concat_relative_error"] = observable_error
        gates["observable_concat_matches_token_tuple"] = observable_error < 3e-8

    if args.stage != "replay":
        if oracle.observable_network is None:
            raise RuntimeError("direct-RQ stage did not build one observable boundary")
        canonical = canonicalize_implicit_dag_direct_rq(
            oracle.observable_network,
            replay_inputs=raw,
            block_size=args.block_size,
        )
        exact_runtime_calls.add(DIRECT_RQ_RUNTIME_CALL)
        if any(step.method == DIRECT_RQ_METHOD for step in canonical.steps):
            exact_runtime_calls.update(
                {STREAMED_QR_RUNTIME_CALL, TRIANGULAR_RUNTIME_CALL}
            )
        canonical_error = _relative(
            evaluate_boundary_quotient(canonical.network, raw),
            observable.reshape(raw.shape[0], -1),
        )
        shape = implicit_shape_statistics(canonical.network)
        result["boundary_algorithms"] = {
            "canonical_replay_relative_error": canonical_error,
            "factorization_methods": [step.method for step in canonical.steps],
            "algorithm1_push_ledger": [
                sum(step.parent_occurrences_pushed for step in canonical.steps),
                sum(step.expected_parent_occurrences for step in canonical.steps),
            ],
        }
        gates.update(
            {
                "all_observable_nodes_use_direct_RQ": all(
                    step.method
                    in {DIRECT_RQ_METHOD, UNARY_RQ_METHOD, BOUNDED_EXPLICIT_RQ_METHOD}
                    for step in canonical.steps
                ),
                "observable_algorithm1_replay": canonical_error < 3e-8,
                "observable_algorithm1_pushes_every_occurrence": all(
                    step.parent_occurrences_pushed == step.expected_parent_occurrences
                    for step in canonical.steps
                ),
            }
        )
        if args.stage == "full":
            exact_runtime_calls.add(PRODUCTION_ALGORITHM3_RUNTIME_CALL)
            environments = reverse_implicit_environments(canonical.network)
            diagonal = diagonalize_implicit_dag_full_rank(
                canonical.network,
                replay_inputs=raw,
            )
            result["boundary_algorithms"].update(
                {
                    "algorithm2_child_messages": sum(
                        record.child_messages_emitted for record in environments
                    ),
                    "algorithm3_replay_relative_error": diagonal.replay_relative_error,
                    "algorithm3_offdiagonal_ratio": (
                        diagonal.maximum_recontracted_offdiagonal_ratio
                    ),
                    "algorithm3_push_ledger": [
                        diagonal.pushed_parent_occurrences,
                        diagonal.expected_parent_occurrences,
                    ],
                }
            )
            gates.update(
                {
                    "observable_algorithm2_emits_every_message": sum(
                        record.child_messages_emitted for record in environments
                    )
                    == shape["edge_occurrences"],
                    "observable_algorithm3_replay_and_diagonal": max(
                        diagonal.replay_relative_error,
                        diagonal.maximum_recontracted_offdiagonal_ratio,
                    )
                    < 3e-8,
                    "observable_algorithm3_pushes_every_occurrence": (
                        diagonal.pushed_parent_occurrences
                        == diagonal.expected_parent_occurrences
                        == shape["edge_occurrences"] + 1
                    ),
                }
            )

    assert_all_token_norm_buffers_unchanged(oracle)
    state_after = _state_digest(state_owner)
    result["state_sha256_after"] = state_after
    gates["source_block_state_unchanged"] = state_before == state_after
    runtime_compliance = assert_direct_only_runtime_guard(
        direct_only_runtime_report(),
        exact_allowed_calls=exact_runtime_calls,
    )
    result["runtime_direct_only_guard_final"] = runtime_compliance
    gates["transitive_direct_only_runtime_audit"] = True
    result["gates"] = gates
    result["all_gates_pass"] = all(gates.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
