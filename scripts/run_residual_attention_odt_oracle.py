"""Run the dense residual-attention ODT oracle and emit machine-readable results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from xvla.models.vla import ChiVLA, VLAConfig
from xvla.nn.attention import causal_mask
from xvla.nn.block import ChiTransformerBlock
from xvla.train.residual_attention_odt_oracle import (
    POLYNOMIAL_CLAIM_BOUNDARY,
    RATIONAL_INTERFACE_GAP,
    canonical_environments_mixed_arity,
    canonicalize_mixed_arity,
    compile_tiny_vla_residual_oracle,
    gauge_bond,
    mixed_arity_forward,
    observation_to_polynomial_input,
    oracle_actions,
    one_leg_gauge_negative_control,
    replay_rational_residual_block,
)


DTYPE = torch.float64


def tiny_model(seed: int, layers: int) -> ChiVLA:
    torch.manual_seed(seed)
    cfg = VLAConfig(
        image_size=1,
        patch_size=1,
        vit_dim=1,
        vit_layers=0,
        vit_heads=1,
        vocab_size=2,
        max_instr_len=1,
        state_dim=1,
        n_embodiments=2,
        dim=1,
        n_layers=layers,
        n_heads=1,
        ffn_rank=3,
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
                module.bias.normal_(std=0.2)
        model.vision.patch.bias.normal_(std=0.2)
    return model


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = expected.norm().clamp_min(torch.finfo(expected.dtype).tiny)
    return float(((actual - expected).norm() / scale).item())


def sorted_spectrum(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.eigvalsh(0.5 * (matrix + matrix.T)).flip(0)


def one_seed(seed: int, layers: int) -> dict[str, object]:
    generator = torch.Generator().manual_seed(10_000 + seed)
    model = tiny_model(seed, layers)
    oracle = compile_tiny_vla_residual_oracle(model)
    img = torch.randn(5, 3, 1, 1, generator=generator, dtype=DTYPE)
    instr = torch.randint(0, 2, (5, 1), generator=generator)
    state = torch.randn(5, 1, generator=generator, dtype=DTYPE)
    embodiment = torch.randint(0, 2, (5,), generator=generator)
    raw = observation_to_polynomial_input(model, img, instr, state, embodiment)
    direct = model(img, instr, state, embodiment)[0]
    compiled = oracle_actions(oracle, raw)
    canonical = canonicalize_mixed_arity(oracle.network)
    canonical_output = mixed_arity_forward(canonical.network, raw).reshape_as(direct)

    base_grams = canonical_environments_mixed_arity(canonical.network)
    gauge_replay_errors = []
    gauge_canonical_replay_errors = []
    spectrum_errors = []
    for bond, dimension in enumerate(oracle.network.bond_dims):
        random = torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
        orthogonal, _ = torch.linalg.qr(random)
        scales = torch.linspace(0.6, 1.8, dimension, dtype=DTYPE)
        transform = orthogonal @ torch.diag(scales) @ orthogonal.T
        gauged = gauge_bond(oracle.network, bond, transform)
        gauged_output = mixed_arity_forward(gauged, raw).reshape_as(direct)
        gauged_canonical = canonicalize_mixed_arity(gauged)
        gauged_canonical_output = mixed_arity_forward(gauged_canonical.network, raw).reshape_as(direct)
        gauge_grams = canonical_environments_mixed_arity(gauged_canonical.network)
        gauge_replay_errors.append(relative_error(gauged_output, direct))
        gauge_canonical_replay_errors.append(relative_error(gauged_canonical_output, direct))
        spectrum_errors.append(
            max(
                relative_error(sorted_spectrum(left), sorted_spectrum(right))
                for left, right in zip(base_grams, gauge_grams)
            )
        )
    dimension = oracle.network.bond_dims[0]
    bad_transform = torch.diag(torch.linspace(0.55, 1.85, dimension, dtype=DTYPE))
    one_leg = one_leg_gauge_negative_control(oracle.network, 0, bad_transform)
    one_leg_error = relative_error(
        mixed_arity_forward(one_leg, raw).reshape_as(direct), direct
    )
    image_column_count = 3 * model.cfg.image_size * model.cfg.image_size
    image_column_energy = float(oracle.network.embedding[1:, 1 : 1 + image_column_count].norm().item())
    return {
        "seed": seed,
        "direct_compiler_relative_error": relative_error(compiled, direct),
        "canonical_replay_relative_error": relative_error(canonical_output, direct),
        "gauge_replay_relative_error": max(gauge_replay_errors),
        "gauge_canonical_replay_relative_error": max(gauge_canonical_replay_errors),
        "canonical_spectrum_gauge_relative_error": max(spectrum_errors),
        "one_leg_negative_control_relative_error": one_leg_error,
        "maximum_isometry_error": max(canonical.isometry_errors),
        "maximum_factorization_error": max(canonical.factorization_errors),
        "maximum_symmetry_error": max(canonical.symmetry_errors),
        "raw_bond_dims": oracle.network.bond_dims,
        "canonical_bond_dims": canonical.network.bond_dims,
        "arities": oracle.network.arities,
        "image_column_energy": image_column_energy,
    }


def rational_replay(seed: int) -> dict[str, object]:
    torch.manual_seed(20_000 + seed)
    block = ChiTransformerBlock(
        dim=2,
        n_heads=1,
        n_layers=1,
        causal=True,
        norm="rational",
        qk_norm="rational",
        attn="bilinear",
        ffn="bilinear",
        residual=True,
    ).double().eval()
    rational_norms = [module for module in block.modules() if module.__class__.__name__ == "RationalNorm"]
    with torch.no_grad():
        for norm in rational_norms:
            norm.running_ms.fill_(1.1 + 0.1 * torch.rand((), dtype=DTYPE))
            norm.initialized.fill_(True)
        for module in block.modules():
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.normal_(std=0.1)
    tokens = torch.randn(2, 2, 2, dtype=DTYPE)
    mask = causal_mask(2, dtype=DTYPE)
    direct = block(tokens, mask=mask, method="explicit")
    replay = replay_rational_residual_block(block, tokens, mask, maximum_routes=8)
    return {
        "relative_error": relative_error(replay.value, direct),
        "denominator_ledger_entries": len(replay.denominator_ledger),
        "object_kind": replay.object_kind,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.layers <= 0:
        raise ValueError("--layers must be positive")
    rows = [one_seed(seed, args.layers) for seed in range(args.seeds)]
    rational = rational_replay(0)
    numeric_keys = (
        "direct_compiler_relative_error",
        "canonical_replay_relative_error",
        "gauge_replay_relative_error",
        "gauge_canonical_replay_relative_error",
        "canonical_spectrum_gauge_relative_error",
        "maximum_isometry_error",
        "maximum_factorization_error",
        "maximum_symmetry_error",
    )
    maxima = {key: max(float(row[key]) for row in rows) for key in numeric_keys}
    gates = {
        "polynomial_forward": maxima["direct_compiler_relative_error"] < 2e-11,
        "canonical_replay": maxima["canonical_replay_relative_error"] < 2e-10,
        "nonorthogonal_gauge_replay": maxima["gauge_replay_relative_error"] < 2e-10,
        "canonical_spectrum_gauge": maxima["canonical_spectrum_gauge_relative_error"] < 2e-8,
        "row_isometry": maxima["maximum_isometry_error"] < 2e-10,
        "vision_path_present": min(float(row["image_column_energy"]) for row in rows) > 0,
        "residual_arities": all(
            tuple(row["arities"]) == (5, 2) * args.layers for row in rows
        ),
        "one_leg_push_fails": min(
            float(row["one_leg_negative_control_relative_error"]) for row in rows
        ) > 1e-8,
        "rational_staged_replay": float(rational["relative_error"]) < 5e-10,
    }
    result = {
        "object_kind": "tiny_vla_residual_attention_canonical_odt_oracle",
        "claim_boundary": POLYNOMIAL_CLAIM_BOUNDARY,
        "rational_interface_gap": RATIONAL_INTERFACE_GAP,
        "dtype": "float64",
        "seeds": args.seeds,
        "layers": args.layers,
        "maxima": maxima,
        "rational_replay": rational,
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
