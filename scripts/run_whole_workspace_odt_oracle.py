"""Multiseed exact oracle for one ChiViT block plus one joint VLA block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from xvla.models.vla import ChiVLA, VLAConfig
from xvla.train.residual_attention_odt_oracle import (
    canonical_environments_mixed_arity,
    canonicalize_mixed_arity,
    compile_tiny_vla_whole_workspace_oracle,
    gauge_bond,
    mixed_arity_forward,
    observation_to_polynomial_input,
    one_leg_gauge_negative_control,
    oracle_actions,
)


DTYPE = torch.float64


def model_for(seed: int) -> ChiVLA:
    torch.manual_seed(seed)
    cfg = VLAConfig(
        image_size=1,
        patch_size=1,
        vit_dim=1,
        vit_layers=1,
        vit_heads=1,
        vocab_size=2,
        max_instr_len=1,
        state_dim=1,
        n_embodiments=2,
        dim=1,
        n_layers=1,
        n_heads=1,
        ffn_rank=3,
        vit_ffn_rank=3,
        attn="bilinear",
        vit_attn="bilinear",
        ffn="bilinear",
        norm="none",
        qk_norm="none",
        residual=True,
        vit_residual=True,
        action_horizon=1,
        action_dim=1,
        action_head="linear",
    )
    model = ChiVLA(cfg).double().eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.normal_(std=0.2)
        model.vision.patch.bias.normal_(std=0.2)
    return model


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        ((actual - expected).norm() / expected.norm().clamp_min(torch.finfo(expected.dtype).tiny)).item()
    )


def spectrum(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.eigvalsh(0.5 * (matrix + matrix.T))


def one_seed(seed: int, maximum_elements: int) -> dict[str, object]:
    model = model_for(seed)
    oracle = compile_tiny_vla_whole_workspace_oracle(
        model, maximum_elements=maximum_elements
    )
    generator = torch.Generator().manual_seed(30_000 + seed)
    img = torch.randn(4, 3, 1, 1, generator=generator, dtype=DTYPE)
    instr = torch.randint(0, 2, (4, 1), generator=generator)
    state = torch.randn(4, 1, generator=generator, dtype=DTYPE)
    embodiment = torch.randint(0, 2, (4,), generator=generator)
    raw = observation_to_polynomial_input(model, img, instr, state, embodiment)
    direct = model(img, instr, state, embodiment)[0]
    compiled = oracle_actions(oracle, raw)
    canonical = canonicalize_mixed_arity(oracle.network)
    canonical_output = mixed_arity_forward(canonical.network, raw).reshape_as(direct)
    reference_grams = canonical_environments_mixed_arity(canonical.network)
    gauge_replay_errors = []
    gauge_canonical_errors = []
    spectrum_errors = []
    for bond, dimension in enumerate(oracle.network.bond_dims):
        random = torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
        orthogonal, _ = torch.linalg.qr(random)
        transform = orthogonal @ torch.diag(
            torch.linspace(0.65, 1.75, dimension, dtype=DTYPE)
        ) @ orthogonal.T
        gauged = gauge_bond(oracle.network, bond, transform)
        gauged_output = mixed_arity_forward(gauged, raw).reshape_as(direct)
        gauge_replay_errors.append(relative_error(gauged_output, direct))
        gauged_canonical = canonicalize_mixed_arity(gauged).network
        gauge_canonical_errors.append(
            relative_error(mixed_arity_forward(gauged_canonical, raw).reshape_as(direct), direct)
        )
        candidate_grams = canonical_environments_mixed_arity(gauged_canonical)
        spectrum_errors.append(
            max(
                relative_error(spectrum(candidate), spectrum(reference))
                for reference, candidate in zip(reference_grams, candidate_grams)
            )
        )
    dimension = oracle.network.bond_dims[0]
    bad_transform = torch.diag(torch.linspace(0.5, 1.9, dimension, dtype=DTYPE))
    incorrect = one_leg_gauge_negative_control(oracle.network, 0, bad_transform)
    one_leg_error = relative_error(
        mixed_arity_forward(incorrect, raw).reshape_as(direct), direct
    )
    return {
        "seed": seed,
        "arities": oracle.network.arities,
        "raw_bond_dims": oracle.network.bond_dims,
        "canonical_bond_dims": canonical.network.bond_dims,
        "direct_compiler_relative_error": relative_error(compiled, direct),
        "canonical_replay_relative_error": relative_error(canonical_output, direct),
        "maximum_all_bond_gauge_replay_relative_error": max(gauge_replay_errors),
        "maximum_all_bond_gauged_canonical_replay_relative_error": max(gauge_canonical_errors),
        "maximum_canonical_spectrum_gauge_relative_error": max(spectrum_errors),
        "maximum_isometry_error": max(canonical.isometry_errors),
        "maximum_factorization_error": max(canonical.factorization_errors),
        "maximum_symmetry_error": max(canonical.symmetry_errors),
        "one_leg_negative_control_relative_error": one_leg_error,
        "image_column_energy": float(oracle.network.embedding[1:, 1:4].norm().item()),
        "vision_residual": bool(model.vision.blocks.blocks[0].residual),
        "joint_residual": bool(model.backbone.blocks[0].residual),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--maximum-elements", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seeds <= 0 or args.maximum_elements <= 0:
        raise ValueError("seeds and maximum-elements must be positive")
    rows = [one_seed(seed, args.maximum_elements) for seed in range(args.seeds)]
    error_keys = (
        "direct_compiler_relative_error",
        "canonical_replay_relative_error",
        "maximum_all_bond_gauge_replay_relative_error",
        "maximum_all_bond_gauged_canonical_replay_relative_error",
        "maximum_canonical_spectrum_gauge_relative_error",
        "maximum_isometry_error",
        "maximum_factorization_error",
        "maximum_symmetry_error",
    )
    maxima = {key: max(float(row[key]) for row in rows) for key in error_keys}
    gates = {
        "full_pixels_to_actions_replay": maxima["direct_compiler_relative_error"] < 1e-9,
        "canonical_replay": maxima["canonical_replay_relative_error"] < 1e-8,
        "all_bond_gauge_replay": maxima["maximum_all_bond_gauge_replay_relative_error"] < 1e-8,
        "canonical_spectrum_gauge": maxima["maximum_canonical_spectrum_gauge_relative_error"] < 1e-6,
        "isometry": maxima["maximum_isometry_error"] < 1e-9,
        "factorization": maxima["maximum_factorization_error"] < 1e-9,
        "topology": all(tuple(row["arities"]) == (5, 2, 1, 5, 2) for row in rows),
        "vision_and_joint_residuals": all(
            bool(row["vision_residual"]) and bool(row["joint_residual"]) for row in rows
        ),
        "vision_path_present": min(float(row["image_column_energy"]) for row in rows) > 0,
        "one_leg_push_fails": min(
            float(row["one_leg_negative_control_relative_error"]) for row in rows
        ) > 1e-8,
    }
    result = {
        "object_kind": "whole_workspace_one_vit_one_joint_canonical_odt_oracle",
        "scope": (
            "Tiny float64 ChiVLA with one real bidirectional ChiViT residual block, exact "
            "visual projection bridge, and one real causal joint residual block."
        ),
        "claim_boundary": (
            "Exact polynomial canonical ODT for identity/scalar-folded normalization, "
            "not yet a Padé rational canonical ODT or a production-width materialization."
        ),
        "seeds": args.seeds,
        "maxima": maxima,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "rows": rows,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    if not result["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
