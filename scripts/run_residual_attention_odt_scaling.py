"""Bounded dense scaling probe for the residual-attention ODT oracle.

This runner never silently exceeds a requested dense-core or environment cap.
It is separate from the frozen multiseed correctness runner so scaling probes
cannot weaken that experiment's gates.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import torch

from xvla.models.vla import ChiVLA, VLAConfig
from xvla.train.residual_attention_odt_oracle import (
    canonical_environments_mixed_arity,
    canonicalize_mixed_arity,
    compile_tiny_vla_residual_oracle,
    mixed_arity_forward,
    observation_to_polynomial_input,
    oracle_actions,
)


DTYPE = torch.float64


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = expected.norm().clamp_min(torch.finfo(expected.dtype).tiny)
    return float(((actual - expected).norm() / scale).item())


def model_for(args: argparse.Namespace) -> ChiVLA:
    torch.manual_seed(args.seed)
    cfg = VLAConfig(
        image_size=args.image_size,
        patch_size=1,
        vit_dim=args.dim,
        vit_layers=0,
        vit_heads=1,
        vocab_size=2,
        max_instr_len=1,
        state_dim=1,
        n_embodiments=2,
        dim=args.dim,
        n_layers=args.layers,
        n_heads=1,
        ffn_rank=max(3, 3 * args.dim),
        attn="bilinear",
        ffn="bilinear",
        norm="none",
        qk_norm="none",
        residual=True,
        action_horizon=1,
        action_dim=1,
        action_head="linear",
    )
    model = ChiVLA(cfg).double().eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.normal_(std=0.1)
        model.vision.patch.bias.normal_(std=0.1)
    return model


def write_result(result: dict[str, object], output: Path | None) -> None:
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=1)
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maximum-core-elements", type=int, default=20_000_000)
    parser.add_argument("--maximum-environment-elements", type=int, default=20_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ("image_size", "dim", "layers", "maximum_core_elements", "maximum_environment_elements"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    token_count = args.image_size**2 + 5
    homogeneous_dimension = 1 + token_count * args.dim
    attention_core_elements = homogeneous_dimension**6
    ffn_core_elements = homogeneous_dimension**3
    symmetric_attention_coordinates = math.comb(homogeneous_dimension + 4, 5)
    legacy_environment_identity_elements = homogeneous_dimension**8
    environment_flat_context_width = homogeneous_dimension**4
    environment_fast_path_materialized_elements = homogeneous_dimension**2
    result: dict[str, object] = {
        "object_kind": "bounded_dense_residual_attention_odt_scaling_probe",
        "scope": "vit_layers=0 patch front-end plus residual joint backbone",
        "seed": args.seed,
        "image_size": args.image_size,
        "dim": args.dim,
        "layers": args.layers,
        "token_count": token_count,
        "homogeneous_state_dimension": homogeneous_dimension,
        "arities": [5, 2] * args.layers,
        "attention_core_elements_each": attention_core_elements,
        "ffn_core_elements_each": ffn_core_elements,
        "total_materialized_core_elements": args.layers * (
            attention_core_elements + ffn_core_elements
        ),
        "symmetric_attention_coordinates_per_output": symmetric_attention_coordinates,
        "legacy_environment_identity_elements_each_role": legacy_environment_identity_elements,
        "environment_flat_context_width": environment_flat_context_width,
        "environment_fast_path_materialized_elements_each_role": environment_fast_path_materialized_elements,
        "environment_contraction": "identity-sibling flattened-context fast path",
        "maximum_core_elements": args.maximum_core_elements,
        "maximum_environment_elements": args.maximum_environment_elements,
        "normalization_object": "identity polynomial",
        "rational_checkpoint_note": (
            "Padé checkpoints require pair-valued projective composition and are not "
            "represented by this polynomial dense scaling probe."
        ),
        "timings_seconds": {},
        "peak_rss_mb": {"start": peak_rss_mb()},
    }
    if attention_core_elements > args.maximum_core_elements:
        result.update(
            status="core_cap_rejected",
            rejection=(
                f"attention core needs {attention_core_elements} elements, above "
                f"cap {args.maximum_core_elements}"
            ),
        )
        write_result(result, args.output)
        return

    model = model_for(args)
    start = time.perf_counter()
    oracle = compile_tiny_vla_residual_oracle(
        model, maximum_elements=args.maximum_core_elements
    )
    result["timings_seconds"]["compile"] = time.perf_counter() - start
    result["peak_rss_mb"]["after_compile"] = peak_rss_mb()

    generator = torch.Generator().manual_seed(100_000 + args.seed)
    img = torch.randn(2, 3, args.image_size, args.image_size, generator=generator, dtype=DTYPE)
    instr = torch.randint(0, 2, (2, 1), generator=generator)
    state = torch.randn(2, 1, generator=generator, dtype=DTYPE)
    embodiment = torch.randint(0, 2, (2,), generator=generator)
    raw = observation_to_polynomial_input(model, img, instr, state, embodiment)
    direct = model(img, instr, state, embodiment)[0]
    compiled = oracle_actions(oracle, raw)
    result["direct_compiler_relative_error"] = relative_error(compiled, direct)

    start = time.perf_counter()
    canonical = canonicalize_mixed_arity(oracle.network)
    result["timings_seconds"]["canonical_rq"] = time.perf_counter() - start
    result["peak_rss_mb"]["after_canonical_rq"] = peak_rss_mb()
    canonical_output = mixed_arity_forward(canonical.network, raw).reshape_as(direct)
    result["canonical_replay_relative_error"] = relative_error(canonical_output, direct)
    result["maximum_isometry_error"] = max(canonical.isometry_errors)
    result["maximum_factorization_error"] = max(canonical.factorization_errors)
    result["raw_bond_dims"] = oracle.network.bond_dims
    result["canonical_bond_dims"] = canonical.network.bond_dims

    if environment_fast_path_materialized_elements > args.maximum_environment_elements:
        result["environment_status"] = "environment_cap_rejected"
        result["environment_rejection"] = (
            "the identity-sibling fast-path result needs "
            f"{environment_fast_path_materialized_elements} elements, "
            f"above cap {args.maximum_environment_elements}"
        )
        result["status"] = "canonicalized_environment_skipped_at_cap"
    else:
        start = time.perf_counter()
        grams = canonical_environments_mixed_arity(canonical.network)
        result["timings_seconds"]["canonical_environments"] = time.perf_counter() - start
        result["peak_rss_mb"]["after_environments"] = peak_rss_mb()
        result["environment_status"] = "complete"
        result["gram_dimensions"] = [gram.shape[0] for gram in grams]
        result["gram_trace"] = [float(torch.trace(gram).item()) for gram in grams]
        result["status"] = "complete"
    result["gates"] = {
        "direct_replay": result["direct_compiler_relative_error"] < 1e-8,
        "canonical_replay": result["canonical_replay_relative_error"] < 1e-7,
        "row_isometry": result["maximum_isometry_error"] < 1e-8,
        "factorization": result["maximum_factorization_error"] < 1e-8,
        "core_cap_respected": attention_core_elements <= args.maximum_core_elements,
        "environment_cap_respected": (
            environment_fast_path_materialized_elements <= args.maximum_environment_elements
            if result["environment_status"] == "complete"
            else True
        ),
    }
    result["all_executed_gates_pass"] = all(result["gates"].values())
    write_result(result, args.output)
    if not result["all_executed_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
