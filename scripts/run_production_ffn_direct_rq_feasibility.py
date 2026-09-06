#!/usr/bin/env python3
"""Checkpoint-backed local CP-FFN direct-RQ feasibility benchmark.

This benchmark materializes exactly one learned BilinearFFN core from the
capable production checkpoint and unfolds it as ``M[out, h * h]``, where
``h = width + 1`` includes the homogeneous coordinate.  It then computes
``M = R @ Q`` exclusively with Householder QR of ``M.T``.  It does not form a
Gram matrix and does not use SVD, polar, or eigendecomposition.

The scope is deliberately local.  This is not a shared-DAG canonicalization,
not a complete residual block, not a rational/projective traversal, and not a
whole-model ODT certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import torch

from xvla.models.vla import ChiVLA, VLAConfig
from xvla.nn.bilinear import BilinearFFN


EXPECTED_CHECKPOINT_SHA256 = (
    "96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9"
)
DTYPE = torch.float64


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)
    return float((torch.linalg.vector_norm(actual - expected) / scale).item())


def maximum_absolute_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).abs().max().item())


def production_config(vocab_size: int) -> VLAConfig:
    return VLAConfig(
        image_size=64,
        patch_size=8,
        vit_dim=192,
        vit_layers=4,
        vit_heads=8,
        vit_ffn_rank=576,
        vocab_size=vocab_size,
        max_instr_len=32,
        state_dim=8,
        n_embodiments=1,
        dim=384,
        n_layers=8,
        n_heads=12,
        ffn_rank=1152,
        attn="bilinear",
        vit_attn="bilinear",
        ffn="bilinear",
        norm="rational",
        qk_norm="rational",
        residual=True,
        vit_residual=True,
        action_horizon=8,
        action_dim=7,
        action_head="linear",
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result {args.output}")
    if args.block_index != 0:
        raise ValueError("this diagnostic is intentionally restricted to block 0")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least two so the zero-input case is retained")
    if not torch.cuda.is_available():
        raise RuntimeError("this production-width benchmark requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")

    total_started = time.perf_counter()
    checkpoint_sha256 = file_sha256(args.checkpoint)
    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise RuntimeError("checkpoint SHA-256 differs from the pinned capable checkpoint")

    load_started = time.perf_counter()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vocab_size = int(state["tok_emb.weight"].shape[0])
    model = ChiVLA(production_config(vocab_size)).eval()
    model.load_state_dict(state, strict=True)
    del state
    if args.stack == "vision":
        block = model.vision.blocks.blocks[args.block_index]
        expected_width, expected_rank = 192, 576
        source_path = "vision.blocks.blocks.0.ffn"
    else:
        block = model.backbone.blocks[args.block_index]
        expected_width, expected_rank = 384, 1152
        source_path = "backbone.blocks.blocks.0.ffn"
    ffn = block.ffn
    if not isinstance(ffn, BilinearFFN):
        raise RuntimeError("selected checkpoint primitive is not BilinearFFN")
    if ffn.dim != expected_width or ffn.out_dim != expected_width or ffn.rank != expected_rank:
        raise RuntimeError("selected checkpoint FFN shape differs from the production contract")
    if ffn.down.bias is not None:
        raise RuntimeError("local fused-core identity requires the checkpoint's bias-free down map")
    load_seconds = time.perf_counter() - load_started

    transfer_started = time.perf_counter()
    ffn = ffn.to(device=device, dtype=DTYPE).eval()
    synchronize(device)
    transfer_seconds = time.perf_counter() - transfer_started
    source_digest_before = module_digest(ffn)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    free_before, total_memory = torch.cuda.mem_get_info(device)

    materialize_started = time.perf_counter()
    fused_core = ffn.dense_core()
    unfolding = fused_core.reshape(ffn.out_dim, (ffn.dim + 1) ** 2)
    synchronize(device)
    materialize_seconds = time.perf_counter() - materialize_started

    qr_started = time.perf_counter()
    q_columns, r_upper = torch.linalg.qr(unfolding.T, mode="reduced")
    factor = r_upper.T
    quotient = q_columns.T
    synchronize(device)
    direct_rq_seconds = time.perf_counter() - qr_started

    verify_started = time.perf_counter()
    reconstructed = factor @ quotient
    identity = torch.eye(quotient.shape[0], dtype=DTYPE, device=device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    inputs = torch.randn(
        args.batch_size,
        ffn.dim,
        generator=generator,
        dtype=DTYPE,
    ).mul_(0.25).to(device)
    inputs[-1].zero_()
    homogeneous = torch.cat(
        (torch.ones(args.batch_size, 1, dtype=DTYPE, device=device), inputs),
        dim=-1,
    )
    tied_routes = torch.einsum("bi,bj->bij", homogeneous, homogeneous).reshape(
        args.batch_size, -1
    )
    module_output = ffn(inputs)
    fused_output = tied_routes @ unfolding.T
    rq_output = (tied_routes @ quotient.T) @ factor.T
    synchronize(device)
    verify_seconds = time.perf_counter() - verify_started

    reconstruction_error = relative_error(reconstructed, unfolding)
    row_isometry_error = relative_error(quotient @ quotient.T, identity)
    fused_module_error = relative_error(fused_output, module_output)
    rq_module_error = relative_error(rq_output, module_output)
    rq_fused_error = relative_error(rq_output, fused_output)
    source_digest_after = module_digest(ffn)
    free_after, _ = torch.cuda.mem_get_info(device)
    finite = all(
        bool(torch.isfinite(value).all())
        for value in (fused_core, factor, quotient, module_output, fused_output, rq_output)
    )
    gates = {
        "pinned_capable_checkpoint": checkpoint_sha256 == args.expected_checkpoint_sha256,
        "actual_production_block0_ffn": source_path.endswith("blocks.0.ffn")
        and ffn.dim == expected_width
        and ffn.rank == expected_rank,
        "bias_free_down_cp_identity": ffn.down.bias is None,
        "cuda_float64": device.type == "cuda" and unfolding.dtype == DTYPE,
        "direct_householder_rq_replay": reconstruction_error < 1e-10,
        "direct_householder_rq_row_isometry": row_isometry_error < 1e-10,
        "fused_core_matches_tied_input_module": fused_module_error < 1e-10,
        "direct_rq_matches_tied_input_module": rq_module_error < 1e-10,
        "direct_rq_matches_fused_core": rq_fused_error < 1e-10,
        "all_materialized_values_finite": finite,
        "source_ffn_unchanged": source_digest_before == source_digest_after,
    }
    result = {
        "schema": "xvla-production-ffn-direct-rq-feasibility-v1",
        "object_kind": "single_local_fused_cp_ffn_unfolding",
        "claim_boundary": (
            "Checkpoint-backed local-primitive feasibility only. This directly RQ-factorizes "
            "one fused CP-FFN unfolding with torch.linalg.qr(M.T, mode='reduced'). It is not a "
            "shared-DAG canonicalization, complete residual block, rational/projective traversal, "
            "whole-model ODT, or compression certificate."
        ),
        "factorization_method": {
            "name": "direct_householder_reduced_qr_of_transpose",
            "expression": "q_columns, r_upper = torch.linalg.qr(M.T, mode='reduced')",
            "identity": "M = r_upper.T @ q_columns.T",
            "svd_calls": 0,
            "polar_calls": 0,
            "normal_equation_or_gram_factorizations": 0,
        },
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "case": {
            "stack": args.stack,
            "block_index": args.block_index,
            "source_module_path": source_path,
            "width": ffn.dim,
            "cp_rank": ffn.rank,
            "homogeneous_width": ffn.dim + 1,
            "unfolding_shape": list(unfolding.shape),
            "factor_shape": list(factor.shape),
            "quotient_shape": list(quotient.shape),
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "materialization": {
            "fused_core_shape": list(fused_core.shape),
            "fused_core_elements": fused_core.numel(),
            "fused_core_bytes_float64": fused_core.numel() * fused_core.element_size(),
            "unfolding_elements": unfolding.numel(),
            "unfolding_bytes_float64": unfolding.numel() * unfolding.element_size(),
            "implicit_cp_outer_elements_if_materialized": ffn.rank * (ffn.dim + 1) ** 2,
        },
        "errors": {
            "rq_factorization_relative": reconstruction_error,
            "rq_row_isometry_relative": row_isometry_error,
            "fused_vs_module_tied_input_relative": fused_module_error,
            "fused_vs_module_tied_input_maximum_absolute": maximum_absolute_error(
                fused_output, module_output
            ),
            "rq_vs_module_tied_input_relative": rq_module_error,
            "rq_vs_module_tied_input_maximum_absolute": maximum_absolute_error(
                rq_output, module_output
            ),
            "rq_vs_fused_tied_input_relative": rq_fused_error,
        },
        "timings_seconds": {
            "checkpoint_and_model_load": load_seconds,
            "selected_ffn_cuda_float64_transfer": transfer_seconds,
            "fused_core_materialization": materialize_seconds,
            "direct_householder_reduced_rq": direct_rq_seconds,
            "verification": verify_seconds,
            "total": time.perf_counter() - total_started,
        },
        "memory": {
            "cpu_peak_rss_mb": peak_rss_mb(),
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_bytes": int(total_memory),
            "gpu_free_bytes_before_materialization": int(free_before),
            "gpu_free_bytes_after_verification": int(free_after),
            "gpu_baseline_allocated_bytes": baseline_allocated,
            "gpu_baseline_reserved_bytes": baseline_reserved,
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "source_ffn_sha256_before": source_digest_before,
        "source_ffn_sha256_after": source_digest_after,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256
    )
    parser.add_argument("--stack", choices=("vision", "joint"), required=True)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n")
    print(encoded)
    if not result["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
