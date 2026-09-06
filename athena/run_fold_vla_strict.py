#!/usr/bin/env python3
"""Athena gate for strict scalar χ-VLA calibration, fold, and reload."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import socket
from dataclasses import asdict
from pathlib import Path
from statistics import median

import torch

from xvla.models.vla import ChiVLA, VLAConfig
from xvla.nn.normalization import RmsBatchNorm
from xvla.train.calibrate import calibrate_rbn_sequential
from xvla.train.fold_vla import fold_vla, remaining_rbn_paths, verify_vla_fold
from xvla.train.fold_vla import load_strict_vla_export_package, make_strict_vla_export_package
from xvla.nn.attention import BilinearAttention
from xvla.train.complete_attention_odt import compile_bilinear_attention, evaluate_compiled_attention


def config() -> VLAConfig:
    return VLAConfig(
        image_size=8,
        patch_size=4,
        vit_dim=4,
        vit_layers=1,
        vit_heads=2,
        vocab_size=16,
        max_instr_len=3,
        state_dim=2,
        n_embodiments=2,
        dim=4,
        n_layers=1,
        n_heads=2,
        ffn_rank=7,
        vit_ffn_rank=7,
        norm="scalar_rbn",
        qk_norm="scalar_rbn",
        action_horizon=2,
        action_dim=3,
        action_head="linear",
    )


def batch(seed: int, dtype: torch.dtype, batch_size: int = 4) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "img": torch.randn(batch_size, 3, 8, 8, generator=generator, dtype=dtype),
        "instr_ids": torch.randint(0, 16, (batch_size, 3), generator=generator),
        "state": torch.randn(batch_size, 2, generator=generator, dtype=dtype),
        "embodiment_id": torch.randint(0, 2, (batch_size,), generator=generator),
    }


def run_seed(seed: int, dtype: torch.dtype, artifact_dir: Path) -> dict[str, object]:
    torch.manual_seed(seed)
    model = ChiVLA(config()).to(dtype=dtype).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)) and module.bias is not None:
                module.bias.normal_()
    initial_action_weight = model.action_head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
    training_losses = []
    finite_gradients = True
    model.train()
    for index in range(4):
        data = batch(50_000 + 100 * seed + index, dtype)
        target_generator = torch.Generator().manual_seed(75_000 + 100 * seed + index)
        target = torch.randn(
            data["img"].shape[0], config().action_horizon, config().action_dim,
            generator=target_generator, dtype=dtype,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(**data)
        loss = torch.nn.functional.mse_loss(prediction, target)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("synthetic strict VLA training loss became nonfinite")
        loss.backward()
        finite_gradients = finite_gradients and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        training_losses.append(float(loss.detach().item()))
    model.eval()
    action_weight_change = float(
        torch.linalg.vector_norm(model.action_head.weight - initial_action_weight).item()
    )
    calibration = calibrate_rbn_sequential(
        model,
        lambda index: batch(100_000 + 100 * seed + index, dtype),
        lambda calibrated, data: calibrated(**data),
        iters=16,
        excluded_paths=("vision.norm_out",),
    )
    validation = batch(200_000 + seed, dtype)
    result = verify_vla_fold(model, validation)
    folded = result.model
    restored = ChiVLA(copy.deepcopy(folded.cfg)).to(dtype=dtype).eval()
    restored.load_state_dict(folded.state_dict(), strict=True)
    expected, _ = folded(**validation)
    actual, _ = restored(**validation)
    reload_error = float((actual - expected).abs().max().item())
    dtype_name = str(dtype).removeprefix("torch.")
    checkpoint_path = artifact_dir / f"source_{dtype_name}_seed{seed}.pt"
    package_path = artifact_dir / f"export_{dtype_name}_seed{seed}.pt"
    if checkpoint_path.exists() or package_path.exists():
        raise FileExistsError("refusing to overwrite strict VLA gate artifacts")
    torch.save(
        {
            "config": asdict(model.cfg),
            "state_dict": model.state_dict(),
            "frozen_rbn_paths": [
                name
                for name, module in model.named_modules()
                if isinstance(module, RmsBatchNorm) and module.frozen
            ],
        },
        checkpoint_path,
    )
    source_checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    loaded_source = ChiVLA(VLAConfig(**loaded_checkpoint["config"])).to(dtype=dtype).eval()
    loaded_source.load_state_dict(loaded_checkpoint["state_dict"], strict=True)
    frozen_paths = set(loaded_checkpoint["frozen_rbn_paths"])
    for name, module in loaded_source.named_modules():
        if isinstance(module, RmsBatchNorm) and name in frozen_paths:
            module.frozen = True
            module.calibrating = False
    package = make_strict_vla_export_package(
        loaded_source, source_checkpoint_sha256=source_checkpoint_sha256
    )
    torch.save(package, package_path)
    package_sha256 = hashlib.sha256(package_path.read_bytes()).hexdigest()
    disk_roundtrip = torch.load(package_path, map_location="cpu", weights_only=False)
    packaged = load_strict_vla_export_package(disk_roundtrip)
    packaged_output, _ = packaged(**validation)
    package_error = float((packaged_output - expected).abs().max().item())

    compiler_errors = []
    for name, attention in (
        [(f"vision.{index}", block.attn) for index, block in enumerate(folded.vision.blocks.blocks)]
        + [(f"backbone.{index}", block.attn) for index, block in enumerate(folded.backbone.blocks)]
    ):
        positions = 4 if name.startswith("vision") else folded.seq_len
        compiled = compile_bilinear_attention(attention, n_query=positions)
        tokens = torch.randn(2, positions, folded.cfg.dim if name.startswith("backbone") else folded.cfg.vit_dim, dtype=dtype)
        compiler_output = evaluate_compiled_attention(compiled, tokens)
        direct_output = attention(tokens)
        compiler_errors.append(float((compiler_output - direct_output).abs().max().item()))

    vision_tokens = torch.randn(2, 4, folded.cfg.vit_dim, dtype=dtype)
    vision_attn = folded.vision.blocks.blocks[0].attn
    vision_execution_error = float((vision_attn(vision_tokens, method="explicit") - vision_attn(vision_tokens, method="khatri_rao")).abs().max().item())
    joint_tokens = torch.randn(2, folded.seq_len, folded.cfg.dim, dtype=dtype)
    joint_attn = folded.backbone.blocks[0].attn
    joint_execution_error = float((joint_attn(joint_tokens, method="explicit") - joint_attn(joint_tokens, method="causal_scan")).abs().max().item())
    deployed_norms = [
        name for name in remaining_rbn_paths(model) if name != "vision.norm_out"
    ]
    return {
        "seed": seed,
        "dtype": dtype_name,
        "training_losses": training_losses,
        "finite_training_gradients": finite_gradients,
        "action_weight_change": action_weight_change,
        "source_norm_count": len(remaining_rbn_paths(model)),
        "deployed_norm_count": len(deployed_norms),
        "folded_norm_count": len(remaining_rbn_paths(folded)),
        "maximum_absolute_error": result.maximum_absolute_error,
        "maximum_relative_error": result.maximum_relative_error,
        "reload_maximum_absolute_error": reload_error,
        "package_reload_maximum_absolute_error": package_error,
        "maximum_attention_compiler_absolute_error": max(compiler_errors),
        "vision_explicit_vs_khatri_maximum_absolute_error": vision_execution_error,
        "joint_explicit_vs_scan_maximum_absolute_error": joint_execution_error,
        "source_state_sha256": package["source_state_sha256"],
        "folded_state_sha256": package["folded_state_sha256"],
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "serialized_package_sha256": package_sha256,
        "source_checkpoint_path": checkpoint_path.as_posix(),
        "serialized_package_path": package_path.as_posix(),
        "removed_dead_paths": list(result.removed_dead_paths),
        "calibration_order": [site.path for site in calibration.sites],
        "calibration_scales": [site.scale for site in calibration.sites],
        "folded_config": asdict(folded.cfg),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("athena/results/fold_vla_strict_v1.json"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("athena/results/fold_vla_strict_v1_artifacts"),
    )
    args = parser.parse_args()
    if args.seeds <= 0:
        raise ValueError("seeds must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result {args.output}")
    if args.artifact_dir.exists():
        raise FileExistsError(f"refusing to overwrite artifact directory {args.artifact_dir}")
    args.artifact_dir.mkdir(parents=True)
    rows = [
        run_seed(seed, dtype, args.artifact_dir)
        for dtype in (torch.float64, torch.float32)
        for seed in range(args.seeds)
    ]
    rows64 = [row for row in rows if row["dtype"] == "float64"]
    rows32 = [row for row in rows if row["dtype"] == "float32"]
    checks = {
        "float64_maximum_absolute_error_below_1e-10": max(float(row["maximum_absolute_error"]) for row in rows64) < 1e-10,
        "float64_maximum_relative_error_below_1e-12": max(float(row["maximum_relative_error"]) for row in rows64) < 1e-12,
        "float32_maximum_absolute_error_below_5e-5": max(float(row["maximum_absolute_error"]) for row in rows32) < 5e-5,
        "float32_maximum_relative_error_below_5e-5": max(float(row["maximum_relative_error"]) for row in rows32) < 5e-5,
        "reload_error_is_zero": max(float(row["reload_maximum_absolute_error"]) for row in rows) == 0.0,
        "package_reload_error_is_zero": max(float(row["package_reload_maximum_absolute_error"]) for row in rows) == 0.0,
        "attention_compiler_error_below_precision_gate": max(float(row["maximum_attention_compiler_absolute_error"]) for row in rows64) < 1e-10 and max(float(row["maximum_attention_compiler_absolute_error"]) for row in rows32) < 5e-5,
        "execution_path_error_below_precision_gate": max(max(float(row["vision_explicit_vs_khatri_maximum_absolute_error"]), float(row["joint_explicit_vs_scan_maximum_absolute_error"])) for row in rows64) < 1e-10 and max(max(float(row["vision_explicit_vs_khatri_maximum_absolute_error"]), float(row["joint_explicit_vs_scan_maximum_absolute_error"])) for row in rows32) < 5e-5,
        "all_folded_norm_counts_zero": all(int(row["folded_norm_count"]) == 0 for row in rows),
        "all_dead_paths_explicit": all(row["removed_dead_paths"] == ["vision.norm_out"] for row in rows),
        "all_calibration_orders_identical": len({tuple(row["calibration_order"]) for row in rows}) == 1,
        "all_synthetic_training_losses_and_gradients_finite": all(
            bool(row["finite_training_gradients"])
            and all(torch.isfinite(torch.tensor(row["training_losses"])).tolist())
            for row in rows
        ),
        "all_synthetic_training_updates_nonzero": all(float(row["action_weight_change"]) > 0 for row in rows),
    }
    manifest = Path("athena/fold_vla_strict_sources.sha256")
    payload = {
        "schema": "xvla-fold-vla-strict-v1",
        "configuration": {"seeds_per_dtype": args.seeds, "dtypes": ["float64", "float32"], "model": asdict(config())},
        "provenance": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source_manifest": manifest.as_posix(),
            "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "source_tree_sha256": os.environ.get("XVLA_SOURCE_TREE_SHA256"),
            "pip_freeze_sha256": os.environ.get("XVLA_PIP_FREEZE_SHA256"),
        },
        "frozen_acceptance_gates": {"passed": all(checks.values()), "checks": checks},
        "summary": {
            "float64_maximum_absolute_error": max(float(row["maximum_absolute_error"]) for row in rows64),
            "float64_median_absolute_error": median(float(row["maximum_absolute_error"]) for row in rows64),
            "float64_maximum_relative_error": max(float(row["maximum_relative_error"]) for row in rows64),
            "float32_maximum_absolute_error": max(float(row["maximum_absolute_error"]) for row in rows32),
            "float32_maximum_relative_error": max(float(row["maximum_relative_error"]) for row in rows32),
        },
        "per_seed": rows,
        "claim_boundary": (
            "This validates calibration and exact normalization folding for a tiny strict VLA "
            "configuration. It is a prerequisite exporter gate, not a capable trained LIBERO "
            "checkpoint and not whole-policy ODT."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"gates": payload["frozen_acceptance_gates"], "summary": payload["summary"]}, indent=2))
    print(f"wrote {args.output}")
    if not payload["frozen_acceptance_gates"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
