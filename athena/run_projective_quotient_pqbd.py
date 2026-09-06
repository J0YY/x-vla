#!/usr/bin/env python3
"""Athena gate for Projective Quotient Balanced Decomposition (PQBD)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
from pathlib import Path
from typing import Any

import torch

from athena.run_xvla_experiment import make_config
from xvla.models.vla import ChiVLA
from xvla.nn.attention import BilinearAttention
from xvla.nn.normalization import RationalNorm
from xvla.train.projective_quotient import (
    PQBD_CLAIM_BOUNDARY,
    PQBD_OBJECT_KIND,
    projective_output_pullback,
    projective_quotient_balanced_decomposition,
    projective_rescale,
    projective_value,
    quotient_metric_pullback,
    rational_ffn_residual_projective,
    rational_norm_projective,
    stack_projective_rows,
    tiny_rational_attention_projective_oracle,
)


SCHEMA = "xvla-projective-quotient-pqbd-v1"
EXPECTED_CHECKPOINT_NAME = "ckpt_linear_rat_vit_s0_v2.pt"
EXPECTED_CHECKPOINT_SHA256 = "96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9"
EXPECTED_RUNTIME = {
    "python": "3.10.19",
    "torch": "2.7.1+cu126",
    "cuda": "12.6",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maximum_absolute(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def quotient_trials(trials: int) -> dict[str, float]:
    maximum_jacobian_error = 0.0
    maximum_expansion_error = 0.0
    maximum_projective_chain_error = 0.0
    maximum_radial_error = 0.0
    maximum_gauge_error = 0.0
    maximum_spectrum_gauge_error = 0.0
    minimum_cross_term_contribution = float("inf")
    for seed in range(trials):
        generator = torch.Generator().manual_seed(10_000 + seed)
        n_out = 3 + seed % 3
        n_in = 4 + seed % 4
        x = torch.randn(n_in, generator=generator, dtype=torch.float64, requires_grad=True)
        a = torch.randn(n_out, n_in, generator=generator, dtype=torch.float64)
        b = torch.randn(n_out, n_in, generator=generator, dtype=torch.float64)

        def numerator_fn(value: torch.Tensor) -> torch.Tensor:
            return a @ value + 0.2 * (b @ value).square()

        def denominator_fn(value: torch.Tensor) -> torch.Tensor:
            return 2.0 + 0.1 * value.square().sum()

        numerator = numerator_fn(x)
        denominator = denominator_fn(x)
        numerator_jacobian = torch.autograd.functional.jacobian(numerator_fn, x)
        denominator_gradient = torch.autograd.grad(denominator, x)[0]
        raw_metric = torch.randn(n_out, n_out, generator=generator, dtype=torch.float64)
        output_metric = raw_metric @ raw_metric.T
        result = quotient_metric_pullback(
            numerator.detach(),
            denominator.detach(),
            numerator_jacobian,
            denominator_gradient,
            output_metric,
        )
        autograd_jacobian = torch.autograd.functional.jacobian(
            lambda value: numerator_fn(value) / denominator_fn(value), x
        )
        maximum_jacobian_error = max(
            maximum_jacobian_error,
            maximum_absolute(result.jacobian, autograd_jacobian),
        )
        direct = result.jacobian.T @ output_metric @ result.jacobian
        maximum_expansion_error = max(
            maximum_expansion_error, maximum_absolute(result.metric, direct)
        )
        without_cross_terms = result.numerator_term + result.denominator_term
        minimum_cross_term_contribution = min(
            minimum_cross_term_contribution,
            maximum_absolute(without_cross_terms, direct),
        )
        projective = projective_output_pullback(
            numerator.detach(), denominator.detach(), output_metric
        )
        projective_chain = torch.cat(
            (numerator_jacobian, denominator_gradient[None, :]), dim=0
        )
        maximum_projective_chain_error = max(
            maximum_projective_chain_error,
            maximum_absolute(
                projective_chain.T @ projective.metric @ projective_chain,
                result.metric,
            ),
        )
        maximum_radial_error = max(
            maximum_radial_error,
            float(projective.radial_residual.abs().max().item()),
        )

        gauge = torch.tensor(-0.8 - 0.01 * seed, dtype=torch.float64)
        gauge_gradient = torch.randn(n_in, generator=generator, dtype=torch.float64) * 0.1
        transformed = quotient_metric_pullback(
            gauge * numerator.detach(),
            gauge * denominator.detach(),
            gauge * numerator_jacobian + numerator.detach()[:, None] * gauge_gradient[None, :],
            gauge * denominator_gradient + denominator.detach() * gauge_gradient,
            output_metric,
        )
        maximum_gauge_error = max(
            maximum_gauge_error,
            maximum_absolute(transformed.metric, result.metric),
        )

        c_factor = torch.randn(n_in, n_in, generator=generator, dtype=torch.float64)
        upstream = c_factor @ c_factor.T + 0.2 * torch.eye(n_in, dtype=torch.float64)
        base_decomposition = projective_quotient_balanced_decomposition(
            upstream,
            numerator.detach(),
            denominator.detach(),
            numerator_jacobian,
            denominator_gradient,
            output_metric,
        )
        transformed_decomposition = projective_quotient_balanced_decomposition(
            upstream,
            gauge * numerator.detach(),
            gauge * denominator.detach(),
            gauge * numerator_jacobian + numerator.detach()[:, None] * gauge_gradient[None, :],
            gauge * denominator_gradient + denominator.detach() * gauge_gradient,
            output_metric,
        )
        maximum_spectrum_gauge_error = max(
            maximum_spectrum_gauge_error,
            maximum_absolute(
                base_decomposition.eigenvalues,
                transformed_decomposition.eigenvalues,
            ),
        )
    return {
        "trials": trials,
        "maximum_quotient_jacobian_vs_autograd_error": maximum_jacobian_error,
        "maximum_expanded_vs_direct_pullback_error": maximum_expansion_error,
        "maximum_projective_chain_error": maximum_projective_chain_error,
        "maximum_radial_null_residual": maximum_radial_error,
        "maximum_nonconstant_gauge_pullback_error": maximum_gauge_error,
        "maximum_nonconstant_gauge_spectrum_error": maximum_spectrum_gauge_error,
        "minimum_error_when_both_cross_terms_are_omitted": minimum_cross_term_contribution,
    }


def pade_trials(trials: int) -> dict[str, Any]:
    maximum_error = 0.0
    minimum_denominator = float("inf")
    ledgers = []
    for degree in (1, 2, 3):
        coefficients = {}
        if degree != 2:
            coefficients = {
                "pade_numerator": tuple(
                    1.0 / (index + 1) for index in range(degree + 1)
                ),
                "pade_denominator": (1.0,) + (0.125,) * degree,
            }
        module = RationalNorm(
            variant="pade", deg=degree, **coefficients
        ).double().eval()
        module.running_ms.fill_(1.7)
        module.initialized.fill_(True)
        for seed in range(trials):
            generator = torch.Generator().manual_seed(20_000 + 100 * degree + seed)
            x = torch.randn(7, 11, generator=generator, dtype=torch.float64)
            x = x * (0.15 + 0.3 * (seed % 9))
            pair = rational_norm_projective(x, module, label=f"pade_degree_{degree}")
            maximum_error = max(
                maximum_error,
                maximum_absolute(
                    projective_value(pair, minimum_absolute_denominator=1e-8),
                    module(x),
                ),
            )
            minimum_denominator = min(
                minimum_denominator,
                pair.denominator_ledger[-1].observed_minimum_absolute_denominator,
            )
            if seed == 0:
                ledgers.append(
                    {
                        "degree": degree,
                        "numerator_degree_upper_bound": pair.numerator_degree_upper_bound,
                        "denominator_degree_upper_bound": pair.denominator_degree_upper_bound,
                        "entry": pair.denominator_ledger[-1].__dict__,
                    }
                )
    return {
        "trials_per_degree": trials,
        "maximum_exact_pair_replay_error": maximum_error,
        "minimum_observed_absolute_denominator": minimum_denominator,
        "degree_ledgers": ledgers,
    }


def attention_trials(trials: int) -> dict[str, Any]:
    maximum_error = 0.0
    maximum_gauge_error = 0.0
    maximum_degree = 0
    minimum_denominator = float("inf")
    for seed in range(trials):
        torch.manual_seed(30_000 + seed)
        causal = bool(seed % 2)
        module = BilinearAttention(
            4,
            2,
            causal=causal,
            qk_norm="rational",
            row_scale="invsqrt" if seed % 3 else "inv",
            score_scale="d_h2",
        ).double().eval()
        with torch.no_grad():
            for norm in (module.rn_q1, module.rn_k1, module.rn_q2, module.rn_k2):
                norm.running_ms.fill_(1.1 + 0.05 * seed)
                norm.initialized.fill_(True)
            for branch in (module.wq1, module.wk1, module.wq2, module.wk2, module.wv, module.wo):
                branch.bias.normal_()
        tokens = torch.randn(2, 3, 4, dtype=torch.float64)
        if causal:
            mask = None
        else:
            mask = torch.tensor(
                [[1, 0, 1], [1, 1, 0], [0, 0, 0]], dtype=torch.float64
            )
        rows = tiny_rational_attention_projective_oracle(
            module, tokens, mask, maximum_routes=32
        )
        compiled = stack_projective_rows(rows, minimum_absolute_denominator=1e-8)
        direct = module(tokens, mask=mask, method="explicit")
        maximum_error = max(maximum_error, maximum_absolute(compiled, direct))
        gauged = tuple(
            projective_rescale(row, -2.0 - query, minimum_absolute_scale=1e-8)
            for query, row in enumerate(rows)
        )
        maximum_gauge_error = max(
            maximum_gauge_error,
            maximum_absolute(
                stack_projective_rows(gauged, minimum_absolute_denominator=1e-8),
                compiled,
            ),
        )
        maximum_degree = max(
            maximum_degree,
            max(row.numerator_degree_upper_bound for row in rows),
        )
        minimum_denominator = min(
            minimum_denominator,
            min(float(row.denominator.abs().min().item()) for row in rows),
        )
    return {
        "trials": trials,
        "maximum_oracle_vs_explicit_error": maximum_error,
        "maximum_constant_gauge_error": maximum_gauge_error,
        "maximum_conservative_numerator_degree": maximum_degree,
        "minimum_observed_absolute_final_denominator": minimum_denominator,
        "scope": "tiny fixed-mask self-attention oracle only",
    }


def checkpoint_ffn_replay(checkpoint: Path) -> dict[str, Any]:
    if checkpoint.name != EXPECTED_CHECKPOINT_NAME:
        raise RuntimeError("checkpoint filename differs from the pinned Stage 8 identity")
    checkpoint_sha = file_sha256(checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA-256 differs from the pinned Stage 8 identity")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "tok_emb.weight" not in state:
        raise RuntimeError("checkpoint does not have the expected χ-VLA state dictionary")
    vocab_size = int(state["tok_emb.weight"].shape[0])
    config = make_config("chi", vocab_size, 8, 7, 64, 8, "vit")
    model = ChiVLA(config).cuda().eval()
    model.load_state_dict(state, strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise RuntimeError("checkpoint contains nonfinite tensors")
    block = model.backbone.blocks[0]
    if not isinstance(block.rbn_ffn, RationalNorm) or block.rbn_ffn.variant != "pade":
        raise RuntimeError("pinned checkpoint first backbone FFN is not Padé-normalized")
    if not bool(block.rbn_ffn.initialized):
        raise RuntimeError("active checkpoint FFN RationalNorm is uninitialized")

    captured: list[torch.Tensor] = []
    handle = block.rbn_ffn.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    generator = torch.Generator(device="cuda").manual_seed(40_000)
    inputs = {
        "img": torch.randn(1, 3, 64, 64, generator=generator, device="cuda"),
        "instr_ids": torch.randint(0, vocab_size, (1, 32), generator=generator, device="cuda"),
        "state": torch.randn(1, 8, generator=generator, device="cuda"),
        "embodiment_id": torch.zeros(1, dtype=torch.int64, device="cuda"),
    }
    before = {
        name: value.detach().clone()
        for name, value in block.rbn_ffn.named_buffers(recurse=False)
    }
    try:
        with torch.no_grad():
            output, _ = model(**inputs)
    finally:
        handle.remove()
    after = dict(block.rbn_ffn.named_buffers(recurse=False))
    if len(captured) != 1:
        raise RuntimeError("the selected active FFN normalization did not execute exactly once")
    buffers_unchanged = all(torch.equal(value, after[name]) for name, value in before.items())
    if not buffers_unchanged:
        raise RuntimeError("eval-mode RationalNorm buffers changed during replay")
    bond = captured[0]
    pair = rational_ffn_residual_projective(
        bond,
        block.rbn_ffn,
        block.ffn,
        block.ffn_gain,
        label="backbone.blocks.0.ffn_residual",
        minimum_absolute_denominator=1e-8,
    )
    with torch.no_grad():
        direct = bond + block.ffn_gain * block.ffn(block.rbn_ffn(bond))
        compiled = projective_value(pair, minimum_absolute_denominator=1e-8)
    return {
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "hook_path": "backbone.blocks.0.rbn_ffn",
        "hook_execution_count": len(captured),
        "hook_input_shape": list(bond.shape),
        "valid_model_output_shape": list(output.shape),
        "valid_model_output_finite": bool(torch.isfinite(output).all()),
        "normalization_buffers_unchanged": buffers_unchanged,
        "maximum_exact_pair_replay_error": maximum_absolute(compiled, direct),
        "numerator_degree_upper_bound": pair.numerator_degree_upper_bound,
        "denominator_degree_upper_bound": pair.denominator_degree_upper_bound,
        "denominator_ledger_entries": len(pair.denominator_ledger),
        "minimum_observed_absolute_denominator": min(
            entry.observed_minimum_absolute_denominator
            for entry in pair.denominator_ledger
        ),
        "input_scope": (
            "checkpoint weights and the real end-to-end χ-VLA execution path, "
            "using deterministic synthetic model-valid inputs rather than a LIBERO behavior panel"
        ),
    }


def negative_controls() -> dict[str, bool]:
    outcomes: dict[str, bool] = {}
    try:
        quotient_metric_pullback(
            torch.ones(2, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64),
        )
    except ValueError:
        outcomes["zero_denominator_rejected"] = True
    try:
        projective_quotient_balanced_decomposition(
            torch.eye(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            -torch.eye(2, dtype=torch.float64),
        )
    except ValueError:
        outcomes["indefinite_output_metric_rejected"] = True
    try:
        module = RationalNorm(variant="pade").double().train()
        module.initialized.fill_(True)
        rational_norm_projective(torch.ones(2, 3, dtype=torch.float64), module)
    except ValueError:
        outcomes["mutable_training_norm_rejected"] = True
    try:
        raw_attention = BilinearAttention(4, 2, qk_norm="none").double().eval()
        tiny_rational_attention_projective_oracle(
            raw_attention, torch.ones(1, 2, 4, dtype=torch.float64)
        )
    except ValueError:
        outcomes["nonrational_attention_rejected"] = True
    expected = {
        "zero_denominator_rejected",
        "indefinite_output_metric_rejected",
        "mutable_training_norm_rejected",
        "nonrational_attention_rejected",
    }
    if set(outcomes) != expected or not all(outcomes.values()):
        raise RuntimeError("one or more PQBD negative controls failed open")
    return outcomes


def runtime_identity() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_tree_sha256": os.environ.get("XVLA_SOURCE_TREE_SHA256"),
        "pip_freeze_sha256": os.environ.get("XVLA_PIP_FREEZE_SHA256"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--quotient-trials", type=int, default=64)
    parser.add_argument("--pade-trials", type=int, default=32)
    parser.add_argument("--attention-trials", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("athena/results/projective_quotient_pqbd_v1.json"),
    )
    args = parser.parse_args()
    if min(args.quotient_trials, args.pade_trials, args.attention_trials) <= 0:
        raise ValueError("all trial counts must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("the Stage 8 checkpoint gate requires an Athena CUDA compute node")
    runtime = runtime_identity()
    for key, expected in EXPECTED_RUNTIME.items():
        if runtime[key] != expected:
            raise RuntimeError(
                f"runtime {key} differs from frozen protocol: {runtime[key]!r} != {expected!r}"
            )
    if not runtime["source_tree_sha256"] or not runtime["pip_freeze_sha256"]:
        raise RuntimeError("source-tree and environment provenance hashes are required")

    quotient = quotient_trials(args.quotient_trials)
    pade = pade_trials(args.pade_trials)
    attention = attention_trials(args.attention_trials)
    checkpoint = checkpoint_ffn_replay(args.checkpoint)
    negatives = negative_controls()
    gates = {
        "quotient_jacobian": quotient["maximum_quotient_jacobian_vs_autograd_error"] <= 2e-11,
        "quotient_expansion": quotient["maximum_expanded_vs_direct_pullback_error"] <= 2e-11,
        "quotient_cross_terms_material": quotient["minimum_error_when_both_cross_terms_are_omitted"] > 1e-8,
        "projective_chain": quotient["maximum_projective_chain_error"] <= 2e-11,
        "radial_null": quotient["maximum_radial_null_residual"] <= 2e-11,
        "nonconstant_gauge_pullback": quotient["maximum_nonconstant_gauge_pullback_error"] <= 2e-11,
        "nonconstant_gauge_spectrum": quotient["maximum_nonconstant_gauge_spectrum_error"] <= 2e-9,
        "pade_exact_pair": pade["maximum_exact_pair_replay_error"] <= 2e-11,
        "pade_denominator_observed_safe": pade["minimum_observed_absolute_denominator"] > 1e-8,
        "tiny_attention_exact_pair": attention["maximum_oracle_vs_explicit_error"] <= 2e-8,
        "tiny_attention_gauge": attention["maximum_constant_gauge_error"] <= 2e-10,
        "checkpoint_ffn_exact_pair": checkpoint["maximum_exact_pair_replay_error"] <= 3e-5,
        "checkpoint_ffn_degree_ledger": (
            checkpoint["numerator_degree_upper_bound"] == 10
            and checkpoint["denominator_degree_upper_bound"] == 8
        ),
        "checkpoint_ffn_denominator_observed_safe": checkpoint["minimum_observed_absolute_denominator"] > 1e-8,
        "checkpoint_active_path": checkpoint["hook_execution_count"] == 1,
        "checkpoint_buffers_immutable": checkpoint["normalization_buffers_unchanged"],
        "negative_controls": all(negatives.values()),
    }
    passed = all(gates.values())
    payload = {
        "schema": SCHEMA,
        "passed": passed,
        "object_kind": PQBD_OBJECT_KIND,
        "runtime": runtime,
        "gates": gates,
        "quotient_rule": quotient,
        "pade_rational_norm": pade,
        "tiny_attention_oracle": attention,
        "checkpoint_ffn_residual_replay": checkpoint,
        "negative_controls": negatives,
        "claim_boundary": PQBD_CLAIM_BOUNDARY,
        "additional_limitations": [
            "Denominator ledger bounds are observed on tested tensors and are not global domain certificates.",
            "The checkpoint FFN replay uses a deterministic synthetic model-valid input, not a held-out LIBERO trajectory.",
            "The attention compiler is a tiny exact oracle whose common denominators grow rapidly with route count.",
            "No result in this artifact establishes global VLA interpretability or global projective compilation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError(f"PQBD acceptance gate failed: {gates}")
    print(json.dumps({"output": args.output.as_posix(), "passed": passed, "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
