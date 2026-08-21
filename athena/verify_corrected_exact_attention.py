#!/usr/bin/env python3
"""Verify the corrected-input ViT exact-attention audit triplet.

This verifier intentionally uses only the Python standard library.  The three
result-file digests bind the immutable JSON payloads copied from Athena.  The
checkpoint and cache digests below were independently recomputed on Athena when
those payloads were collected.  When ``--payload-root`` is supplied, the
verifier also rehashes those large payloads directly.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


RESULTS = {
    0: {
        "filename": "exact_attention_taskmap_s0.json",
        "sha256": "d73ce14cddc94927b0a328ea50d15d3f4a7991b6070c3ed2d14a3a8f6bce5fe5",
        "checkpoint": "artifacts/ckpt_linear_rat_vit_s0_v2.pt",
        "checkpoint_sha256": "96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9",
    },
    1: {
        "filename": "exact_attention_taskmap_s1.json",
        "sha256": "673473f16e046ef7c0e3a776ba6ba8d5eadd48e557ea7a3f9cb4ec413e2047ae",
        "checkpoint": "artifacts/ckpt_linear_rat_vit_s1.pt",
        "checkpoint_sha256": "cc0780b989165a80a449c03cbb3574e2f56b2e5747a64cf32d1fde418df8ec3c",
    },
    2: {
        "filename": "exact_attention_taskmap_s2.json",
        "sha256": "c2d4905364c6601efb2e57551648d520eb1d229bfb29e8c892e3303b2938f58f",
        "checkpoint": "artifacts/ckpt_linear_rat_vit_s2.pt",
        "checkpoint_sha256": "413a770071bd8f16b924c5604f7c7a58567d2eb6c40c66aab202bcf046fec910",
    },
}

CACHE_PATH = "artifacts/libero_frames_100000_64.pkl"
CACHE_SHA256 = "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662"
METADATA = {
    "repository": "lerobot/libero_object_image",
    "revision": "e1e080d7df1d0a359dff5c86c222e047549f447f",
    "metadata_file": "meta/tasks.parquet",
    "metadata_sha256": "34caee9641ae50bb4e077de306a7d0031753757882da8b1f117e7ea36a486b42",
    "dataset_to_official_task": {
        "0": 9,
        "1": 4,
        "2": 1,
        "3": 3,
        "4": 0,
        "5": 7,
        "6": 2,
        "7": 6,
        "8": 5,
        "9": 8,
    },
    "ordering_matches_official": False,
    "language_set_matches_official": True,
}
NUMERICAL_ENVIRONMENT = {
    "torch_version": "2.7.1+cu126",
    "cuda_version": "12.6",
    "cuda_matmul_allow_tf32": True,
    "cudnn_allow_tf32": True,
    "libero_version": "0.1.0",
    "robosuite_version": "1.4.1",
    "mujoco_version": "3.5.0",
}
EVALUATION_PROTOCOL = {
    "res": 64,
    "horizon": 8,
    "num_steps_wait": 10,
    "exec_h": 8,
    "eps_per_task": 50,
    "max_steps": 280,
}
SOURCE_SHA256 = {
    "athena/run_xvla_experiment.py": "91ae342892e32b0aa019a43ff06b5809dca2a49d4be20ce9a55fbad1b15cdbb3",
    "athena/libero_dataset_metadata.py": "3e14b117ee72b010939c0fdd2c20417778b89a8691156553a0bfb2ad2d0b4edb",
    "xvla/__init__.py": "a6eeee3f1c8c9eec78d2101a0a55761c49d24c3a107d32f065fde725ee193626",
    "xvla/models/__init__.py": "142c431a5637c1d97cc1cb8d0ada64b3904dba4d0eabf79a706be141190a71ec",
    "xvla/models/lm.py": "c77279e821bddf168a975b777bfe1f9293fd317085b7cb9a9df771b6fd569aa0",
    "xvla/models/vla.py": "bc276b0328b53cf1f54b42bcd75137df5785cb8625f51d2b09671081d7c4275f",
    "xvla/models/vit.py": "111049ad4c24179e294da8cccb732e3a416223f77b42a08ed56291febe4ddf87",
    "xvla/nn/__init__.py": "2751993d3f6782f60b55f72915744c55762beb02108f05a1c550b89ceb3cfc52",
    "xvla/nn/attention.py": "4f8e49d9dc25ee292b34daf60687f80f38a8f85548ef2905b3217e66dc8c27ec",
    "xvla/nn/baselines.py": "eb4d6ba5f3aa66589991ee3221017dc59092ec9fd35255f9124812420451a8d4",
    "xvla/nn/bilinear.py": "162929529750fe719c6b4199ae316ddb45f72f52de7bbbd07d329dfb5f8287c8",
    "xvla/nn/block.py": "7555c0b7b11c2592c97bf90ec7eeca6ecd0b76f58960df78eefa984386eea969",
    "xvla/nn/flow_action.py": "1108fd31c0091ce51afa342e798ab625514bc5a8064e45b8f0ce38790d71c1aa",
    "xvla/nn/homogeneous.py": "314ba488c638da18d74de213218e310ea75aafa4fbbc92d9cb838aaa5a31701b",
    "xvla/nn/normalization.py": "67406ca22083c28223023ce1efcfc448470654da8477a535bf4a5284cf7338a9",
    "xvla/nn/product_routing.py": "20da357c875e426b2341e1950736ac6cb88700cb1654e46a4b8a2943c666c9b2",
    "xvla/nn/projector.py": "6212e35fbab973de2e79d74bd5064635db12a2c5e48c8c7df7428cbf1e31aaed",
    "xvla/nn/quantile_action.py": "6bd1ecbe5491457fb9ae0154ea566c0ce24e43f102516699f0b3941d4b497c7e",
    "xvla/train/__init__.py": "b1a7a347ca176ef4626ecc1f1d3318dcff63aa5350a9f6df7b8eee19e6058f42",
    "xvla/train/balance.py": "c444c959dfef3935bbad316f514a3175de92a7f5c1df5dea3f1a5e3e146a6208",
    "xvla/train/calibrate.py": "f40387995adeca41a94d1d2c2d524b5fa4bfcd1f1bfda3fa6f7555f2648426f7",
    "xvla/train/data.py": "2b2dd182c654979aff8905097ba577d97c82a386d6d36220dd9e2a49b142c992",
    "xvla/train/exact_odt_attention_proto.py": "048e094f384ef6064cbd02c6d5668024a49ed8a218d4c318f7846e66d47e3238",
}
SOURCE_SNAPSHOTS = {
    "xvla/models/vla.py": (
        "athena/results/exact_attention_frozen_sources/xvla_models_vla.py.b64"
    ),
    "xvla/models/vit.py": (
        "athena/results/exact_attention_frozen_sources/xvla_models_vit.py.b64"
    ),
    "xvla/nn/product_routing.py": (
        "athena/results/exact_attention_frozen_sources/xvla_nn_product_routing.py.b64"
    ),
}
RELATIVE_L2_GATE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("athena/results"),
        help="Directory containing the three frozen result JSON files.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root used to rehash the frozen source closure.",
    )
    parser.add_argument(
        "--payload-root",
        type=Path,
        help="Optional root containing the frozen checkpoint and cache payloads.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        fail(f"{label}: expected {expected!r}, got {actual!r}")


def finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label}: expected a numeric value, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        fail(f"{label}: expected a finite nonnegative value, got {value!r}")
    return result


def verify_sources(repo_root: Path) -> None:
    for relative, expected in SOURCE_SHA256.items():
        path = repo_root / relative
        if path.is_file() and sha256(path) == expected:
            continue
        snapshot_relative = SOURCE_SNAPSHOTS.get(relative)
        if snapshot_relative is None:
            fail(f"missing or stale frozen source: {path}")
        snapshot = repo_root / snapshot_relative
        if not snapshot.is_file():
            fail(f"missing frozen source snapshot: {snapshot}")
        try:
            encoded = "".join(snapshot.read_text(encoding="ascii").split())
            payload = base64.b64decode(encoded, validate=True)
        except (OSError, UnicodeError, binascii.Error) as exc:
            fail(f"invalid frozen source snapshot {snapshot}: {exc}")
        require_equal(
            bytes_sha256(payload),
            expected,
            f"snapshot source SHA-256 for {relative}",
        )


def verify_payloads(payload_root: Path) -> None:
    expected = {CACHE_PATH: CACHE_SHA256}
    expected.update(
        {
            str(row["checkpoint"]): str(row["checkpoint_sha256"])
            for row in RESULTS.values()
        }
    )
    for relative, digest in expected.items():
        path = payload_root / relative
        if not path.is_file():
            fail(f"missing frozen payload: {path}")
        require_equal(sha256(path), digest, f"payload SHA-256 for {relative}")


def expected_row_layout() -> list[tuple[str, int, int, int, int]]:
    return [
        *(('vision', block, 8, 192, 64) for block in range(4)),
        *(('joint', block, 12, 384, 107) for block in range(8)),
    ]


def verify_result(path: Path, seed: int, identity: dict[str, str]) -> dict[str, Any]:
    if not path.is_file():
        fail(f"missing result: {path}")
    require_equal(sha256(path), identity["sha256"], f"result SHA-256 for seed {seed}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"could not parse {path}: {exc}")

    for key, expected in {
        "mode": "exact_attention",
        "architecture": "chi",
        "vision_encoder": "vit",
        "suite": "libero_object",
        "training_suite": "libero_object",
        "checkpoint": identity["checkpoint"],
        "cache": CACHE_PATH,
        "seed": seed,
        "matmul_precision": "high",
        "prediction_finite": True,
        "prediction_shape": [1, 8, 7],
        "cache_task_metadata": METADATA,
        "numerical_environment": NUMERICAL_ENVIRONMENT,
        "evaluation_protocol": EVALUATION_PROTOCOL,
    }.items():
        require_equal(result.get(key), expected, f"seed {seed} field {key}")

    profile = result.get("profile")
    if not isinstance(profile, dict):
        fail(f"seed {seed} profile is not an object")
    require_equal(profile.get("parameters"), 20_137_352, f"seed {seed} parameter count")
    require_equal(profile.get("profile_iterations"), 20, f"seed {seed} profile iterations")
    require_equal(profile.get("gpu"), "NVIDIA RTX A6000", f"seed {seed} GPU")

    exact = result.get("exact_attention")
    if not isinstance(exact, dict):
        fail(f"seed {seed} exact_attention is not an object")
    rows = exact.get("rows")
    if not isinstance(rows, list):
        fail(f"seed {seed} exact_attention rows are not a list")
    layout = expected_row_layout()
    require_equal(len(rows), len(layout), f"seed {seed} row count")

    max_absolute = 0.0
    max_relative = 0.0
    heads = 0
    for index, (row, expected_layout) in enumerate(zip(rows, layout, strict=True)):
        if not isinstance(row, dict):
            fail(f"seed {seed} row {index} is not an object")
        stack, block, expected_heads, dim, sequence_length = expected_layout
        for key, expected in {
            "stack": stack,
            "block_index": block,
            "heads": expected_heads,
            "dim": dim,
            "sequence_length": sequence_length,
        }.items():
            require_equal(row.get(key), expected, f"seed {seed} row {index} field {key}")
        max_absolute = max(
            max_absolute,
            finite_nonnegative(row.get("max_abs_error"), f"seed {seed} row {index} absolute error"),
        )
        max_relative = max(
            max_relative,
            finite_nonnegative(
                row.get("relative_l2_error"), f"seed {seed} row {index} relative L2 error"
            ),
        )
        heads += expected_heads

    require_equal(exact.get("modules_audited"), len(rows), f"seed {seed} modules audited")
    require_equal(exact.get("heads_audited"), heads, f"seed {seed} heads audited")
    require_equal(exact.get("vision_modules"), 4, f"seed {seed} vision modules")
    require_equal(exact.get("joint_modules"), 8, f"seed {seed} joint modules")
    require_equal(exact.get("max_abs_error"), max_absolute, f"seed {seed} maximum absolute error")
    require_equal(
        exact.get("max_relative_l2_error"),
        max_relative,
        f"seed {seed} maximum relative L2 error",
    )
    require_equal(
        exact.get("all_modules_below_1e_minus_5"),
        all(float(row["max_abs_error"]) < 1e-5 for row in rows),
        f"seed {seed} 1e-5 flag",
    )
    require_equal(
        exact.get("all_modules_below_2e_minus_5"),
        all(float(row["max_abs_error"]) < 2e-5 for row in rows),
        f"seed {seed} 2e-5 flag",
    )
    if max_relative > RELATIVE_L2_GATE:
        fail(
            f"seed {seed} maximum relative L2 error {max_relative} exceeds "
            f"the frozen {RELATIVE_L2_GATE} gate"
        )
    return {
        "seed": seed,
        "result": str(path),
        "result_sha256": identity["sha256"],
        "checkpoint": identity["checkpoint"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "modules": len(rows),
        "architectural_heads": heads,
        "max_abs_error": max_absolute,
        "max_relative_l2_error": max_relative,
    }


def main() -> int:
    args = parse_args()
    try:
        verify_sources(args.repo_root.resolve())
        if args.payload_root is not None:
            verify_payloads(args.payload_root.resolve())
        rows = [
            verify_result(args.results_dir / identity["filename"], seed, identity)
            for seed, identity in sorted(RESULTS.items())
        ]
        summary = {
            "certificate_pass": True,
            "relative_l2_gate": RELATIVE_L2_GATE,
            "checkpoint_payloads_rehashed": args.payload_root is not None,
            "cache_path": CACHE_PATH,
            "cache_sha256": CACHE_SHA256,
            "source_file_count": len(SOURCE_SHA256),
            "packaged_source_snapshot_count": len(SOURCE_SNAPSHOTS),
            "records": rows,
            "totals": {
                "checkpoint_inputs": len(rows),
                "module_input_cases": sum(int(row["modules"]) for row in rows),
                "architectural_heads": sum(int(row["architectural_heads"]) for row in rows),
                "worst_max_abs_error": max(float(row["max_abs_error"]) for row in rows),
                "worst_max_relative_l2_error": max(
                    float(row["max_relative_l2_error"]) for row in rows
                ),
            },
            "claim_boundary": (
                "One fixed corrected cached input per checkpoint and layerwise numerical replay "
                "of individual attention modules. This is not an exact end-to-end policy "
                "decomposition."
            ),
        }
    except (OSError, ValueError) as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
