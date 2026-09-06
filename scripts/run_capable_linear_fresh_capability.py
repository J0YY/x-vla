#!/usr/bin/env python3
"""Fresh hash-bound LIBERO-Object evaluation for the b1c0 linear policy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STAGE_ROOT = Path("/work/joy/x-vla-capable-linear-b1c0-odt-v2")
SOURCE_MANIFEST = PROJECT_ROOT / "athena/capable_linear_fresh_capability_sources.sha256"
STAGE_LEDGER = PROJECT_ROOT / "athena/capable_linear_b1c0_stage.sha256"
CHECKPOINT = PROJECT_ROOT / "inputs/capable_linear_b1c0_checkpoint.pt"
TRAINING_RECORD = PROJECT_ROOT / "inputs/capable_linear_training.json"
RESULT_DIRECTORY = PROJECT_ROOT / "athena/results/capable_linear_b1c0"
CHECKPOINT_SHA256 = "b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee"
TRAINING_RECORD_SHA256 = "e6c07beb2efb7e5ebe55c92fabf694d97b50b9e399f50868eeef4af455e30fa7"
SOURCE_MANIFEST_RELATIVE = "athena/capable_linear_fresh_capability_sources.sha256"
STAGE_LEDGER_RELATIVE = "athena/capable_linear_b1c0_stage.sha256"
SHARDS = ((0, 3), (3, 6), (6, 8), (8, 10))
EPISODES_PER_TASK = 50
MAX_STEPS = 280
SETTLE_STEPS = 10
EXECUTION_HORIZON = 8
RESOLUTION = 64
ACTION_HORIZON = 8
ACTION_DIM = 7
STATE_DIM = 8
CAPABILITY_FLOOR = 0.80
EXPECTED_PARAMETER_COUNT = 20_137_352
EXPECTED_CHECKPOINT_STATE_COUNT = 540
EXPECTED_NORMALIZATION_SITE_COUNT = 74
EXPECTED_INACTIVE_NORMALIZATION_SITES = ("vision.norm_out",)
EXPECTED_PADE_NUMERATOR = (
    5.3299665451049805,
    7.535519123077393,
    0.32727372646331787,
)
EXPECTED_PADE_DENOMINATOR = (
    1.0,
    9.648269653320312,
    2.606637477874756,
)
EXPECTED_CHECKPOINT_TENSOR_SHAPES = {
    "tok_emb.weight": (26, 384),
    "action_head.weight": (7, 384),
    "action_head.bias": (7,),
}
EXPECTED_GPU_NAME = "NVIDIA A30"
EXPECTED_CUDA_VERSION = "12.6"
EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXPECTED_CONFIG_IDENTITY = {
    "image_size": 64,
    "patch_size": 8,
    "vit_dim": 192,
    "vit_layers": 4,
    "vit_heads": 8,
    "vit_ffn_rank": 576,
    "vocab_size": 26,
    "max_instr_len": 32,
    "state_dim": 8,
    "n_embodiments": 1,
    "dim": 384,
    "n_layers": 8,
    "n_heads": 12,
    "ffn_rank": 1152,
    "attn": "bilinear",
    "vit_attn": "bilinear",
    "ffn": "bilinear",
    "norm": "rational",
    "qk_norm": "rational",
    "residual": True,
    "vit_residual": True,
    "action_horizon": 8,
    "action_dim": 7,
    "action_head": "linear",
}
EXPECTED_VERSIONS = {
    "python": "3.10.19",
    "numpy": "1.26.4",
    "torch": "2.7.1+cu126",
    "libero": "0.1.0",
    "robosuite": "1.4.1",
    "mujoco": "3.5.0",
    "pillow": "12.1.1",
}
EXPECTED_VOCAB = {
    "<pad>": 0,
    "<bos>": 1,
    "alphabet": 2,
    "and": 3,
    "basket": 4,
    "bbq": 5,
    "butter": 6,
    "cheese": 7,
    "chocolate": 8,
    "cream": 9,
    "dressing": 10,
    "in": 11,
    "it": 12,
    "juice": 13,
    "ketchup": 14,
    "milk": 15,
    "orange": 16,
    "pick": 17,
    "place": 18,
    "pudding": 19,
    "salad": 20,
    "sauce": 21,
    "soup": 22,
    "the": 23,
    "tomato": 24,
    "up": 25,
}
EXPECTED_NORMALIZATION = {
    "action_mean": [
        0.0689610093832016,
        0.14728403091430664,
        -0.05295417085289955,
        0.0016317926347255707,
        0.006103633902966976,
        -0.014664511196315289,
        0.1320992261171341,
    ],
    "action_std": [
        0.2726696729660034,
        0.4459349811077118,
        0.4579801559448242,
        0.024709034711122513,
        0.04944734275341034,
        0.0426940843462944,
        0.9903656244277954,
    ],
    "state_mean": [
        -0.03134441003203392,
        -0.023217646405100822,
        0.20341111719608307,
        3.1097357273101807,
        -0.20446741580963135,
        -0.10824166983366013,
        0.029097234830260277,
        -0.030324405059218407,
    ],
    "state_std": [
        0.06831111013889313,
        0.16870158910751343,
        0.07987862080335617,
        0.08342526108026505,
        0.3292630910873413,
        0.20355737209320068,
        0.009620374999940395,
        0.009267615154385567,
    ],
}
EXPECTED_TASK_PROTOCOL = {
    0: ("pick up the alphabet soup and place it in the basket", "pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl", "df088984da13131f8332ee0f13a7896c6a97afd02ee5007a42e8fc5e0893571e", "91eed35ff60c2b791ba1d80996e359177a9202d81358e9fddb42ccafffa3f34e"),
    1: ("pick up the cream cheese and place it in the basket", "pick_up_the_cream_cheese_and_place_it_in_the_basket.bddl", "7019f37ee158d67a21338a6df0c441dd8f84979b946b5e20bb49463ef0508ea2", "6443945d386a843b9998caeee65f3093f50e4ab65e913e7102aa39eaee7edd59"),
    2: ("pick up the salad dressing and place it in the basket", "pick_up_the_salad_dressing_and_place_it_in_the_basket.bddl", "024978e9e8f43a49b49d0965f4f426d364ce056307f0fcf5733dc4682fac437a", "7f365994bb15f09bed7e4611603d8b5e4a3576f7fccb827241f6ad60cb8b3e3f"),
    3: ("pick up the bbq sauce and place it in the basket", "pick_up_the_bbq_sauce_and_place_it_in_the_basket.bddl", "8a7e37b76fce5621e649e260dbcfcec7ce2f9a055ca09a4f1ebaf5ca74a739c2", "b186084cc72f9cf8c07ca98c18542f4f0269b23f53bf98ac8474d1cc5b642702"),
    4: ("pick up the ketchup and place it in the basket", "pick_up_the_ketchup_and_place_it_in_the_basket.bddl", "4a4e545483a3fe30cf0ef3dc05bfe5e6ad5e37c94b4376dfa73fe1b60e75b319", "8a035ce260650b525cbdec044e777f2625e689aca72dd737983c4a61539d5ab8"),
    5: ("pick up the tomato sauce and place it in the basket", "pick_up_the_tomato_sauce_and_place_it_in_the_basket.bddl", "b1f4bb69d256a05f693838de46182bb0d3de0de68eec350df1b5b1e76f6c136d", "2015e882bad1d77efd5b973a070fd6bd4cf524f35c36497f5060b1acc7b59dcb"),
    6: ("pick up the butter and place it in the basket", "pick_up_the_butter_and_place_it_in_the_basket.bddl", "5d8053b40ff33246fdbcc38548db84032c6bb647b4b88b6646d9cf6b73a5006c", "981f9d28ca49331ad5faf46ff765233fe7d4fa39a3f35dd332104f900c8cc766"),
    7: ("pick up the milk and place it in the basket", "pick_up_the_milk_and_place_it_in_the_basket.bddl", "9910aabf6717e8ba3e24a3f8500d9bce9ee075346ea6961e8fd766f1257d2996", "f78700be5769be90421173e1689e458b65f4d791855aedc3d2dedc63e3b7a0aa"),
    8: ("pick up the chocolate pudding and place it in the basket", "pick_up_the_chocolate_pudding_and_place_it_in_the_basket.bddl", "674ceabc400b7b16d46e8eaf678709d4a633235c7e0a80233775044fe38b19b0", "54593bb836dc12d321d9e7971e54034bbe01c02929cf0c09cb93b75038bedf5a"),
    9: ("pick up the orange juice and place it in the basket", "pick_up_the_orange_juice_and_place_it_in_the_basket.bddl", "6298533e7bcfb83e40779a77fda39216e8cd53f22bf1af6388585dad0abb0b50", "c198880f92818df55d0a8e53ab739bbfd36acfab236a276c7ac72a27a27baf0d"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or not value.parts or any(part in {"", ".", ".."} for part in value.parts):
        raise RuntimeError(f"unsafe authenticated path {relative!r}")
    cursor = root
    for part in value.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(f"authenticated path traverses a link: {relative}")
    resolved = cursor.resolve(strict=True)
    if resolved == root or root not in resolved.parents or not resolved.is_file():
        raise RuntimeError(f"authenticated path escapes or is nonphysical: {relative}")
    return resolved


def _read_line_manifest(path: Path, root: Path, label: str) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or nonphysical")
    result: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(f"malformed {label} line {number}")
        digest, relative = fields
        relative = relative.lstrip("*")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError(f"invalid digest in {label} line {number}")
        if relative in result:
            raise RuntimeError(f"duplicate {label} member {relative}")
        member = _safe_member(root, relative)
        if _sha256(member) != digest:
            raise RuntimeError(f"{label} byte mismatch: {relative}")
        result[relative] = digest
    return result


def _preimport_verification() -> dict[str, Any]:
    if PROJECT_ROOT != EXPECTED_STAGE_ROOT:
        raise RuntimeError(f"fresh capability must run from {EXPECTED_STAGE_ROOT}")
    source = _read_line_manifest(SOURCE_MANIFEST, PROJECT_ROOT, "source manifest")
    ledger = _read_line_manifest(STAGE_LEDGER, PROJECT_ROOT, "stage ledger")
    required = {
        SOURCE_MANIFEST_RELATIVE,
        "scripts/run_capable_linear_fresh_capability.py",
        "inputs/capable_linear_b1c0_checkpoint.pt",
        "inputs/capable_linear_training.json",
    }
    if not required.issubset(ledger):
        raise RuntimeError("stage ledger omits a fresh-capability authority or input")
    if ledger[SOURCE_MANIFEST_RELATIVE] != _sha256(SOURCE_MANIFEST):
        raise RuntimeError("stage ledger does not bind the source manifest")
    for relative, digest in source.items():
        if ledger.get(relative) != digest:
            raise RuntimeError(f"stage ledger does not bind source {relative}")
    if ledger["inputs/capable_linear_b1c0_checkpoint.pt"] != CHECKPOINT_SHA256:
        raise RuntimeError("stage ledger checkpoint identity differs")
    if ledger["inputs/capable_linear_training.json"] != TRAINING_RECORD_SHA256:
        raise RuntimeError("stage ledger inference-auxiliary identity differs")
    return {
        "source_manifest_sha256": _sha256(SOURCE_MANIFEST),
        "stage_ledger_sha256": _sha256(STAGE_LEDGER),
        "source_sha256": source,
        "stage_sha256": ledger,
    }


_RUNNING_AS_ENTRYPOINT = __name__ == "__main__"
_AUDIT_ONLY = _RUNNING_AS_ENTRYPOINT and sys.argv[1:] == ["--audit-only"]
_PREIMPORT = (
    None
    if not _RUNNING_AS_ENTRYPOINT or _AUDIT_ONLY
    else _preimport_verification()
)
sys.path[:] = [str(PROJECT_ROOT)] + [
    entry for entry in sys.path if entry != str(PROJECT_ROOT)
]
sys.dont_write_bytecode = True

from scripts.odt_direct_only_compliance import (
    audit_direct_only_launch,
)
from scripts.capable_linear_artifact_integrity import (
    assert_physical_hashes_unchanged,
    read_authenticated_json,
    require_rational_norm_buffer_inventory,
)

if Path(sys.modules["scripts"].__file__).resolve() != (
    PROJECT_ROOT / "scripts/__init__.py"
).resolve() or Path(
    sys.modules["scripts.odt_direct_only_compliance"].__file__
).resolve() != (
    PROJECT_ROOT / "scripts/odt_direct_only_compliance.py"
).resolve():
    raise RuntimeError("fresh capability imported a shadowed compliance authority")


_ENTRYPOINTS = (
    Path(__file__),
    PROJECT_ROOT / "tests/test_capable_linear_fresh_capability.py",
)
_STATIC_AUDIT = audit_direct_only_launch(
    PROJECT_ROOT, _ENTRYPOINTS, require_direct_qr=False
)
for _field in (
    "prohibited_calls_found",
    "prohibited_self_overlap_sites",
    "guarded_dormant_spectral_norm_sites",
    "duplicate_top_level_definition_sites",
):
    if _STATIC_AUDIT[_field]:
        raise RuntimeError(f"fresh capability static audit did not close {_field}")
if _PREIMPORT is not None:
    if _PREIMPORT["source_sha256"] != _STATIC_AUDIT["source_sha256"]:
        raise RuntimeError("fresh capability source manifest differs from audited closure")
if _RUNNING_AS_ENTRYPOINT and _AUDIT_ONLY:
    print(json.dumps(_STATIC_AUDIT, indent=2, sort_keys=True))
    raise SystemExit(0)

import numpy as np
import torch
from PIL import Image

from xvla.models.vla import ChiVLA, VLAConfig
from xvla.nn.normalization import RationalNorm


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or nonphysical")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _publish(path: Path, payload: Mapping[str, Any]) -> str:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing existing output {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    linked = False
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        path.chmod(0o444)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_mode & 0o222
            or _sha256(path) != digest
        ):
            raise RuntimeError("published capability output differs")
        return digest
    except BaseException:
        if linked and os.path.lexists(path):
            path.unlink()
        raise
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def _fixed_sources() -> dict[str, Any]:
    manifest = _read_line_manifest(SOURCE_MANIFEST, PROJECT_ROOT, "source manifest")
    ledger = _read_line_manifest(STAGE_LEDGER, PROJECT_ROOT, "stage ledger")
    if manifest != _PREIMPORT["source_sha256"] or ledger != _PREIMPORT["stage_sha256"]:
        raise RuntimeError("authenticated source or stage identity changed")
    audit = audit_direct_only_launch(
        PROJECT_ROOT, _ENTRYPOINTS, require_direct_qr=False
    )
    if audit["source_sha256"] != _STATIC_AUDIT["source_sha256"]:
        raise RuntimeError("fresh capability transitive source closure changed")
    for field in (
        "prohibited_calls_found",
        "prohibited_self_overlap_sites",
        "guarded_dormant_spectral_norm_sites",
        "duplicate_top_level_definition_sites",
    ):
        if audit[field]:
            raise RuntimeError(f"fresh capability final static audit failed {field}")
    return {"manifest": manifest, "ledger": ledger, "audit": audit}


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _numerical_environment(require_gpu: bool) -> dict[str, Any]:
    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "libero": _installed_version("libero"),
        "robosuite": _installed_version("robosuite"),
        "mujoco": _installed_version("mujoco"),
        "pillow": _installed_version("pillow"),
    }
    if observed != EXPECTED_VERSIONS:
        raise RuntimeError(f"frozen numerical environment differs: {observed}")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    if require_gpu and gpu != EXPECTED_GPU_NAME:
        raise RuntimeError(f"fresh capability requires {EXPECTED_GPU_NAME}, got {gpu!r}")
    result = {
        **observed,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    if require_gpu and (
        result["matmul_precision"] != "highest"
        or result["cuda"] != EXPECTED_CUDA_VERSION
        or result["cuda_matmul_allow_tf32"] is not False
        or result["cudnn_allow_tf32"] is not True
        or result["mujoco_gl"] != "egl"
        or result["cublas_workspace_config"]
        != EXPECTED_CUBLAS_WORKSPACE_CONFIG
        or result["deterministic_algorithms"] is not True
        or result["cudnn_benchmark"] is not False
        or result["cudnn_deterministic"] is not True
    ):
        raise RuntimeError(f"frozen numerical backend differs: {result}")
    return result


def _environment_record_is_exact(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and all(value.get(key) == expected for key, expected in EXPECTED_VERSIONS.items())
        and value.get("cuda") == EXPECTED_CUDA_VERSION
        and value.get("gpu") == EXPECTED_GPU_NAME
        and value.get("mujoco_gl") == "egl"
        and value.get("matmul_precision") == "highest"
        and value.get("cuda_matmul_allow_tf32") is False
        and value.get("cudnn_allow_tf32") is True
        and value.get("cublas_workspace_config")
        == EXPECTED_CUBLAS_WORKSPACE_CONFIG
        and value.get("deterministic_algorithms") is True
        and value.get("cudnn_benchmark") is False
        and value.get("cudnn_deterministic") is True
    )


def _training_record() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if _sha256(TRAINING_RECORD) != TRAINING_RECORD_SHA256:
        raise RuntimeError("frozen inference-auxiliary record changed")
    value = _read_object(TRAINING_RECORD, "inference-auxiliary record")
    fixed = {
        "architecture": value.get("architecture") == "chi",
        "vision_encoder": value.get("vision_encoder") == "vit",
        "suite": value.get("suite") == "libero_object",
        "seed": value.get("seed") == 0,
        "steps": value.get("steps") == 40_000,
        "res": value.get("res") == RESOLUTION,
        "horizon": value.get("horizon") == ACTION_HORIZON,
        "samples": value.get("samples") == 63_352,
        "parameters": value.get("parameters") == EXPECTED_PARAMETER_COUNT,
        "checkpoint_path": value.get("checkpoint")
        == "artifacts/ckpt_rational_norm_ablation_v1_rational_s0.pt",
        "cache_sha": value.get("cache_sha256")
        == "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662",
        "vocab": value.get("vocab") == EXPECTED_VOCAB,
        "vocab_sha": value.get("vocab_sha256")
        == _canonical_sha256(EXPECTED_VOCAB),
        "normalization": value.get("normalization") == EXPECTED_NORMALIZATION,
        "mapping_remapped": value.get("task_metadata", {}).get("ordering_matches_official")
        is False,
        "language_set": value.get("task_metadata", {}).get("language_set_matches_official")
        is True,
    }
    failed = sorted(name for name, passed in fixed.items() if not passed)
    if failed:
        raise RuntimeError(f"inference-auxiliary record failed: {failed}")
    normalization = {
        key: np.asarray(EXPECTED_NORMALIZATION[key], dtype=np.float32)
        for key in ("action_mean", "action_std", "state_mean", "state_std")
    }
    return value, normalization


def _encode(text: str, length: int = 32) -> list[int]:
    identifiers = [1] + [
        EXPECTED_VOCAB.get(word, 0)
        for word in text.lower().replace(".", "").split()
    ]
    return (identifiers[:length] + [0] * max(0, length - len(identifiers)))[:length]


def _config() -> VLAConfig:
    return VLAConfig(
        image_size=RESOLUTION,
        patch_size=8,
        vit_dim=192,
        vit_layers=4,
        vit_heads=8,
        vit_ffn_rank=576,
        vocab_size=len(EXPECTED_VOCAB),
        max_instr_len=32,
        state_dim=STATE_DIM,
        n_embodiments=1,
        dim=384,
        n_layers=8,
        n_heads=12,
        ffn_rank=1152,
        action_horizon=ACTION_HORIZON,
        action_dim=ACTION_DIM,
        action_head="linear",
        vision_encoder="vit",
        attn="bilinear",
        vit_attn="bilinear",
        ffn="bilinear",
        norm="rational",
        qk_norm="rational",
        residual=True,
        vit_residual=True,
    )


def _config_identity(config: VLAConfig) -> dict[str, Any]:
    return {key: getattr(config, key) for key in EXPECTED_CONFIG_IDENTITY}


def _arm_inactive_vision_norm_sentinel(
    model: ChiVLA,
) -> tuple[dict[str, int], Any]:
    counter = {"calls": 0}

    def reject_inactive_site(_module: Any, _inputs: Any) -> None:
        counter["calls"] += 1
        raise RuntimeError("inactive vision normalization site was executed")

    handle = model.vision.norm_out.register_forward_pre_hook(reject_inactive_site)
    return counter, handle


def _arm_active_norm_call_counters(
    model: ChiVLA,
) -> tuple[dict[str, int], list[Any]]:
    counters: dict[str, int] = {}
    handles: list[Any] = []
    for name, module in model.named_modules():
        if not isinstance(module, RationalNorm) or name in EXPECTED_INACTIVE_NORMALIZATION_SITES:
            continue
        counters[name] = 0

        def observe(_module: Any, _inputs: Any, *, site: str = name) -> None:
            counters[site] += 1

        handles.append(module.register_forward_pre_hook(observe))
    if len(counters) != 73:
        raise RuntimeError("active normalization sentinel inventory differs")
    return counters, handles


def _remove_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def _inactive_norm_sentinel_is_exact(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("site") == "vision.norm_out"
        and value.get(
            "armed_during_sentinel_replay_and_all_rollout_policy_forwards"
        )
        is True
        and value.get("call_count") == 0
        and value.get("total_site_count") == 74
        and value.get("inactive_sites") == ["vision.norm_out"]
        and value.get("active_site_count") == 73
        and value.get("active_call_count") == 73
        and value.get("every_active_site_call_count") == 1
        and value.get("source_output_bitwise_equal_without_sentinels") is True
    )


def _load_model() -> tuple[ChiVLA, dict[str, Any]]:
    if CHECKPOINT.is_symlink() or not CHECKPOINT.is_file():
        raise RuntimeError("checkpoint is missing or nonphysical")
    if CHECKPOINT.stat().st_size != 80_752_109 or _sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint byte identity differs")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or len(state) != EXPECTED_CHECKPOINT_STATE_COUNT:
        raise RuntimeError("checkpoint tensor inventory differs")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise RuntimeError("checkpoint is not a string-to-tensor mapping")
    if any(
        name not in state or tuple(state[name].shape) != expected
        for name, expected in EXPECTED_CHECKPOINT_TENSOR_SHAPES.items()
    ):
        raise RuntimeError("checkpoint tensor-shape identity differs")
    normalization_inventory = require_rational_norm_buffer_inventory(
        state,
        expected_count=EXPECTED_NORMALIZATION_SITE_COUNT,
        expected_inactive_sites=EXPECTED_INACTIVE_NORMALIZATION_SITES,
        expected_inactive_running_ms={"vision.norm_out": 1.0},
        expected_pade_numerator=EXPECTED_PADE_NUMERATOR,
        expected_pade_denominator=EXPECTED_PADE_DENOMINATOR,
    )
    config = _config()
    config_identity = _config_identity(config)
    if config_identity != EXPECTED_CONFIG_IDENTITY:
        raise RuntimeError("fresh evaluator config identity differs")
    model = ChiVLA(config)
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    if model.num_params() != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("loaded model parameter count differs")
    named_norms = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, RationalNorm)
    }
    checkpoint_norm_sites = {
        name[: -len(".initialized")]
        for name in state
        if name.endswith(".initialized")
    }
    if set(named_norms) != checkpoint_norm_sites:
        raise RuntimeError("loaded normalization module inventory differs")
    active_norms = {
        name: module
        for name, module in named_norms.items()
        if name not in EXPECTED_INACTIVE_NORMALIZATION_SITES
    }
    if len(active_norms) != 73 or any(
        not bool(module.initialized) for module in active_norms.values()
    ):
        raise RuntimeError("active normalization initialization differs")
    if (
        len(named_norms) != EXPECTED_NORMALIZATION_SITE_COUNT
        or bool(model.vision.norm_out.initialized)
        or float(model.vision.norm_out.running_ms) != 1.0
    ):
        raise RuntimeError("inactive vision normalization state differs")
    return model, {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "state_count": len(state),
        "state_key_sha256": _canonical_sha256(sorted(state)),
        "normalization_site_count": len(named_norms),
        "active_normalization_site_count": len(active_norms),
        "normalization_buffer_inventory": normalization_inventory,
        "parameters": model.num_params(),
        "config_identity": config_identity,
    }


def _official_init_states(suite: Any, task_index: int) -> Any:
    original_load = torch.load

    def trusted_packaged_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = trusted_packaged_load
    try:
        states = suite.get_task_init_states(task_index)
    except Exception as exc:
        raise RuntimeError(f"official packaged states failed for task {task_index}") from exc
    finally:
        torch.load = original_load
    if states is None or len(states) != EPISODES_PER_TASK:
        raise RuntimeError(f"task {task_index} does not have exactly 50 packaged states")
    state_hashes = [_array_sha256(np.asarray(state)) for state in states]
    if len(set(state_hashes)) != EPISODES_PER_TASK:
        raise RuntimeError(f"task {task_index} packaged states are not unique")
    return states


def _robot_state(observation: dict[str, Any]) -> np.ndarray:
    from robosuite.utils.transform_utils import quat2axisangle

    value = np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float64),
            quat2axisangle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64),
        )
    ).astype(np.float32)
    if value.size < STATE_DIM:
        value = np.pad(value, (0, STATE_DIM - value.size))
    value = value[:STATE_DIM]
    if value.shape != (STATE_DIM,) or not np.isfinite(value).all():
        raise RuntimeError("robot state is malformed")
    return value


@torch.inference_mode()
def _predict(
    model: ChiVLA,
    observation: dict[str, Any],
    instruction: torch.Tensor,
    normalization: dict[str, np.ndarray],
) -> np.ndarray:
    rotated = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    image_array = np.asarray(Image.fromarray(rotated).resize((RESOLUTION, RESOLUTION))).copy()
    image = torch.from_numpy(image_array).permute(2, 0, 1).float().div(255).unsqueeze(0).cuda()
    raw_state = _robot_state(observation)
    state = torch.tensor(
        (raw_state - normalization["state_mean"]) / normalization["state_std"],
        dtype=torch.float32,
        device="cuda",
    ).unsqueeze(0)
    embodiment = torch.zeros(1, dtype=torch.long, device="cuda")
    prediction, loss = model(image, instruction, state, embodiment)
    if loss is not None or prediction.shape != (1, ACTION_HORIZON, ACTION_DIM):
        raise RuntimeError("public policy forward contract differs")
    action_mean = torch.tensor(normalization["action_mean"], dtype=torch.float32, device="cuda")
    action_std = torch.tensor(normalization["action_std"], dtype=torch.float32, device="cuda")
    output = (prediction[0] * action_std + action_mean).float().cpu().numpy()
    if output.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(output).all():
        raise RuntimeError("policy action is malformed")
    return output


def _settle_environment(
    simulation: Any, observation: dict[str, Any], dummy_action: list[float]
) -> dict[str, Any]:
    for _ in range(SETTLE_STEPS):
        observation, reward, done, _ = simulation.step(dummy_action)
        if bool(done) or bool(reward > 0):
            raise RuntimeError("environment terminated or succeeded during settling")
    return observation


def _execute_episode(
    simulation: Any,
    model: ChiVLA,
    observation: dict[str, Any],
    instruction: torch.Tensor,
    normalization: dict[str, np.ndarray],
    max_steps: int,
) -> tuple[bool, int, bool]:
    success = False
    terminated = False
    steps = 0
    while steps < max_steps and not success and not terminated:
        chunk = _predict(model, observation, instruction, normalization)
        for offset in range(EXECUTION_HORIZON):
            action = chunk[offset].copy()
            action[-1] = 1.0 if action[-1] > 0 else -1.0
            observation, reward, done, _ = simulation.step(action.tolist())
            steps += 1
            success = bool(reward > 0)
            terminated = bool(done)
            if terminated or success or steps >= max_steps:
                break
    return success, steps, terminated


def _task_record(task: Any, bddl_path: Path, states: Any) -> dict[str, Any]:
    return {
        "language": str(task.language),
        "problem_folder": str(task.problem_folder),
        "bddl_file": str(task.bddl_file),
        "bddl_sha256": _sha256(bddl_path),
        "init_state_count": len(states),
        "init_states_sha256": _array_sha256(states),
    }


def _expected_task_record(task_index: int) -> dict[str, Any]:
    language, bddl_file, bddl_sha, states_sha = EXPECTED_TASK_PROTOCOL[task_index]
    return {
        "language": language,
        "problem_folder": "libero_object",
        "bddl_file": bddl_file,
        "bddl_sha256": bddl_sha,
        "init_state_count": EPISODES_PER_TASK,
        "init_states_sha256": states_sha,
    }


def _require_expected_task_record(
    task_index: int, observed: Mapping[str, Any]
) -> None:
    if dict(observed) != _expected_task_record(task_index):
        raise RuntimeError(f"task {task_index} simulator protocol differs: {observed}")


def _episode_inventory_is_exact(
    records: Any,
    task_start: int,
    task_end: int,
    episodes_per_task: int,
    max_steps: int,
) -> bool:
    expected = [
        (task, episode)
        for task in range(task_start, task_end)
        for episode in range(episodes_per_task)
    ]
    if not isinstance(records, list) or len(records) != len(expected):
        return False
    observed: list[tuple[int, int]] = []
    for record in records:
        if not isinstance(record, dict):
            return False
        task = record.get("task_index")
        episode = record.get("episode")
        success = record.get("success")
        steps = record.get("steps")
        terminated_without_success = record.get("terminated_without_success")
        if (
            type(task) is not int
            or type(episode) is not int
            or type(success) is not bool
            or type(steps) is not int
            or type(terminated_without_success) is not bool
            or (terminated_without_success and success)
            or not 0 <= steps <= max_steps
        ):
            return False
        observed.append((task, episode))
    return observed == expected and len(set(observed)) == len(expected)


def _shard_output(start: int, stop: int) -> Path:
    return RESULT_DIRECTORY / f"fresh_capability_t{start}_{stop}.json"


def _validate_smoke() -> str:
    path = RESULT_DIRECTORY / "fresh_capability_smoke.json"
    value, digest = read_authenticated_json(path, "fresh capability smoke")
    episodes = value.get("episodes")
    gates = {
        "schema": value.get("schema")
        == "xvla_capable_linear_b1c0_fresh_capability_smoke_v1",
        "checkpoint": value.get("checkpoint_sha256") == CHECKPOINT_SHA256,
        "training_record": value.get("training_record_sha256")
        == TRAINING_RECORD_SHA256,
        "config_identity": value.get("checkpoint", {}).get("config_identity")
        == EXPECTED_CONFIG_IDENTITY,
        "range": (value.get("task_start"), value.get("task_end")) == (0, 1),
        "seed": value.get("seed") == 0,
        "protocol": value.get("protocol", {}).get("suite") == "libero_object"
        and value.get("protocol", {}).get("resolution") == RESOLUTION
        and value.get("protocol", {}).get("action_horizon") == ACTION_HORIZON
        and value.get("protocol", {}).get("execution_horizon") == EXECUTION_HORIZON
        and value.get("protocol", {}).get("settle_steps") == SETTLE_STEPS
        and value.get("protocol", {}).get("canonical_init_states") is True
        and value.get("protocol", {}).get("episodes_per_task") == 1
        and value.get("protocol", {}).get("max_steps") == 1
        and value.get("protocol", {}).get("settle_success_or_done_count") == 0
        and value.get("protocol", {}).get("observation_transform")
        == "agentview_image[::-1,::-1] then same-size PIL resize"
        and value.get("protocol", {}).get("gripper_decode")
        == "+1 iff predicted coordinate > 0, else -1"
        and value.get("protocol", {}).get("matmul_precision") == "highest"
        and value.get("protocol", {}).get("cuda_matmul_allow_tf32") is False
        and value.get("protocol", {}).get("cudnn_allow_tf32") is True
        and value.get("protocol", {}).get("deterministic_algorithms") is True
        and value.get("protocol", {}).get("cudnn_benchmark") is False
        and value.get("protocol", {}).get("cudnn_deterministic") is True
        and value.get("protocol", {}).get("cublas_workspace_config")
        == EXPECTED_CUBLAS_WORKSPACE_CONFIG
        and value.get("protocol", {}).get("terminal_transition_stops_episode")
        is True
        and value.get("protocol", {}).get(
            "historical_evaluator_control_flow_reproduced"
        )
        is False
        and value.get("protocol", {}).get(
            "historical_protocol_with_corrected_terminal_handling"
        )
        is True
        and value.get("protocol", {}).get(
            "historical_hardware_reproduction_claimed"
        )
        is False
        and value.get("protocol", {}).get("fresh_hardware") == EXPECTED_GPU_NAME,
        "task_protocol": value.get("task_protocol")
        == {"0": _expected_task_record(0)},
        "episode": isinstance(episodes, list)
        and len(episodes) == 1
        and isinstance(episodes[0], dict)
        and episodes[0].get("task_index") == 0
        and episodes[0].get("episode") == 0
        and type(episodes[0].get("success")) is bool
        and type(episodes[0].get("steps")) is int
        and episodes[0].get("steps") == 1
        and type(episodes[0].get("terminated_without_success")) is bool,
        "identity": value.get("source_manifest_sha256")
        == _PREIMPORT["source_manifest_sha256"]
        and value.get("stage_ledger_sha256") == _PREIMPORT["stage_ledger_sha256"],
        "boundary": value.get("simulator_external_to_odt") is True
        and value.get("external_simulator_outside_weight_only_odt_closure") is True
        and value.get("canonical_odt_runtime_guard_installed") is False
        and value.get("odt_runtime_compliance_claimed") is False,
        "environment": _environment_record_is_exact(value.get("environment")),
        "static_audit": value.get("static_audit") == _STATIC_AUDIT,
        "inactive_norm_sentinel": _inactive_norm_sentinel_is_exact(
            value.get("inactive_vision_norm_sentinel")
        ),
        "self_gate": value.get("all_identity_and_protocol_gates_pass") is True
        and value.get("smoke_only") is True
        and value.get("exact_checkpoint_digest_verified_before_and_after") is True
        and value.get("stage_ledger_verified_before_and_after") is True,
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    if failed:
        raise RuntimeError(f"fresh capability smoke failed: {failed}")
    assert_physical_hashes_unchanged({"smoke": path}, {"smoke": digest})
    return digest


def _run_evaluation(
    mode: str, start: int, stop: int
) -> tuple[dict[str, Any], Path]:
    smoke = mode == "smoke"
    if smoke:
        if (start, stop) != (0, 1):
            raise RuntimeError("smoke range differs")
        episodes_per_task = 1
        run_max_steps = 1
        output = RESULT_DIRECTORY / "fresh_capability_smoke.json"
    else:
        if mode != "shard" or (start, stop) not in SHARDS:
            raise RuntimeError("task range is not a preregistered shard")
        episodes_per_task = EPISODES_PER_TASK
        run_max_steps = MAX_STEPS
        output = _shard_output(start, stop)
    smoke_sha = None if smoke else _validate_smoke()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing existing shard output {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("fresh capability requires CUDA")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if (
        torch.backends.cuda.matmul.allow_tf32 is not False
        or torch.backends.cudnn.allow_tf32 is not True
        or not torch.are_deterministic_algorithms_enabled()
        or torch.backends.cudnn.benchmark is not False
        or torch.backends.cudnn.deterministic is not True
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != EXPECTED_CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError("frozen TF32 backend flags were not applied")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    environment = _numerical_environment(require_gpu=True)
    training, normalization = _training_record()
    model, checkpoint = _load_model()
    synthetic_image = torch.zeros(1, 3, RESOLUTION, RESOLUTION, device="cuda")
    synthetic_instruction = torch.tensor([_encode(EXPECTED_TASK_PROTOCOL[0][0])], dtype=torch.long, device="cuda")
    synthetic_state = torch.zeros(1, STATE_DIM, device="cuda")
    synthetic_embodiment = torch.zeros(1, dtype=torch.long, device="cuda")
    with torch.inference_mode():
        synthetic_baseline, synthetic_baseline_loss = model(
            synthetic_image,
            synthetic_instruction,
            synthetic_state,
            synthetic_embodiment,
        )
    inactive_norm_counter, inactive_norm_handle = (
        _arm_inactive_vision_norm_sentinel(model)
    )
    active_norm_calls, active_norm_handles = _arm_active_norm_call_counters(model)
    try:
        with torch.inference_mode():
            synthetic_output, synthetic_loss = model(
                synthetic_image,
                synthetic_instruction,
                synthetic_state,
                synthetic_embodiment,
            )
    finally:
        _remove_hooks(active_norm_handles)
    if (
        synthetic_baseline_loss is not None
        or synthetic_loss is not None
        or tuple(synthetic_output.shape) != (1, ACTION_HORIZON, ACTION_DIM)
        or not bool(torch.isfinite(synthetic_output).all())
        or not torch.equal(synthetic_output, synthetic_baseline)
        or inactive_norm_counter["calls"] != 0
        or len(active_norm_calls) != 73
        or any(count != 1 for count in active_norm_calls.values())
    ):
        raise RuntimeError("checkpoint synthetic replay failed")
    normalization_sentinel = {
        "site": "vision.norm_out",
        "armed_during_sentinel_replay_and_all_rollout_policy_forwards": True,
        "call_count": inactive_norm_counter["calls"],
        "total_site_count": 74,
        "inactive_sites": ["vision.norm_out"],
        "active_site_count": len(active_norm_calls),
        "active_call_count": sum(active_norm_calls.values()),
        "every_active_site_call_count": 1,
        "source_output_bitwise_equal_without_sentinels": torch.equal(
            synthetic_output, synthetic_baseline
        ),
    }

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    if suite.n_tasks != 10:
        raise RuntimeError("LIBERO-Object task count differs")
    observed_languages = {index: str(suite.get_task(index).language) for index in range(10)}
    expected_languages = {index: value[0] for index, value in EXPECTED_TASK_PROTOCOL.items()}
    if observed_languages != expected_languages:
        raise RuntimeError("official LIBERO-Object task ordering differs")

    dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    records: list[dict[str, Any]] = []
    task_protocol: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for task_index in range(start, stop):
        task = suite.get_task(task_index)
        bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        states = _official_init_states(suite, task_index)
        observed_task = _task_record(task, bddl_path, states)
        _require_expected_task_record(task_index, observed_task)
        task_protocol[str(task_index)] = observed_task
        instruction = torch.tensor([_encode(str(task.language))], dtype=torch.long, device="cuda")
        simulation = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=RESOLUTION,
            camera_widths=RESOLUTION,
        )
        try:
            for episode in range(episodes_per_task):
                simulation.seed(task_index * 100 + episode)
                observation = simulation.reset()
                observation = simulation.set_init_state(states[episode])
                observation = _settle_environment(
                    simulation, observation, dummy_action
                )
                episode_started = time.perf_counter()
                success, steps, terminated = _execute_episode(
                    simulation,
                    model,
                    observation,
                    instruction,
                    normalization,
                    run_max_steps,
                )
                record = {
                    "task_index": task_index,
                    "episode": episode,
                    "initial_state_sha256": _array_sha256(states[episode]),
                    "success": success,
                    "steps": steps,
                    "terminated_without_success": terminated and not success,
                    "elapsed_s": time.perf_counter() - episode_started,
                }
                records.append(record)
                print("EPISODE", json.dumps(record, sort_keys=True), flush=True)
        finally:
            simulation.close()

    expected_pairs = [
        (task, episode)
        for task in range(start, stop)
        for episode in range(episodes_per_task)
    ]
    observed_pairs = [(record["task_index"], record["episode"]) for record in records]
    if not _episode_inventory_is_exact(
        records, start, stop, episodes_per_task, run_max_steps
    ) or observed_pairs != expected_pairs:
        raise RuntimeError("fresh shard episode inventory differs")
    if any(
        type(record["success"]) is not bool
        or type(record["steps"]) is not int
        or type(record["terminated_without_success"]) is not bool
        or not 0 <= record["steps"] <= run_max_steps
        for record in records
    ):
        raise RuntimeError("fresh shard episode field is malformed")
    successes = sum(int(record["success"]) for record in records)
    sources = _fixed_sources()
    if _sha256(CHECKPOINT) != CHECKPOINT_SHA256 or _sha256(TRAINING_RECORD) != TRAINING_RECORD_SHA256:
        raise RuntimeError("checkpoint or inference auxiliaries changed during evaluation")
    if _numerical_environment(require_gpu=True) != environment:
        raise RuntimeError("frozen simulator or numerical environment changed during evaluation")
    if inactive_norm_counter["calls"] != 0:
        raise RuntimeError("inactive vision normalization site executed during rollout")
    inactive_norm_handle.remove()
    payload = {
        "schema": (
            "xvla_capable_linear_b1c0_fresh_capability_smoke_v1"
            if smoke
            else "xvla_capable_linear_b1c0_fresh_capability_shard_v1"
        ),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint": checkpoint,
        "training_record_sha256": TRAINING_RECORD_SHA256,
        "training_record_used_as_fixed_inference_auxiliary": True,
        "historical_training_to_checkpoint_digest_link_claimed": False,
        "task_start": start,
        "task_end": stop,
        "seed": 0,
        "protocol": {
            "suite": "libero_object",
            "resolution": RESOLUTION,
            "action_horizon": ACTION_HORIZON,
            "execution_horizon": EXECUTION_HORIZON,
            "settle_steps": SETTLE_STEPS,
            "settle_success_or_done_count": 0,
            "episodes_per_task": episodes_per_task,
            "max_steps": run_max_steps,
            "canonical_init_states": True,
            "observation_transform": "agentview_image[::-1,::-1] then same-size PIL resize",
            "gripper_decode": "+1 iff predicted coordinate > 0, else -1",
            "matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": True,
            "cublas_workspace_config": EXPECTED_CUBLAS_WORKSPACE_CONFIG,
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "historical_evaluator_control_flow_reproduced": False,
            "historical_protocol_with_corrected_terminal_handling": True,
            "terminal_transition_stops_episode": True,
            "historical_hardware_reproduction_claimed": False,
            "fresh_hardware": "NVIDIA A30",
        },
        "task_protocol": task_protocol,
        "episodes": records,
        "episode_identity_sha256": _canonical_sha256(observed_pairs),
        "successes": successes,
        "trials": len(records),
        "overall": successes / len(records),
        "per_task": {
            str(task): sum(int(record["success"]) for record in records if record["task_index"] == task)
            / episodes_per_task
            for task in range(start, stop)
        },
        "early_terminal_failures": sum(
            int(record["terminated_without_success"]) for record in records
        ),
        "synthetic_output_finite": True,
        "inactive_vision_norm_sentinel": normalization_sentinel,
        "environment": environment,
        "source_manifest_sha256": _PREIMPORT["source_manifest_sha256"],
        "stage_ledger_sha256": _PREIMPORT["stage_ledger_sha256"],
        "static_audit": sources["audit"],
        "canonical_odt_runtime_guard_installed": False,
        "odt_runtime_compliance_claimed": False,
        "simulator_external_to_odt": True,
        "external_simulator_outside_weight_only_odt_closure": True,
        "exact_checkpoint_digest_verified_before_and_after": True,
        "stage_ledger_verified_before_and_after": True,
        "external_simulator_part_of_fresh_capability_measurement": True,
        "historical_hardware_reproduction_claimed": False,
        "canonical_odt_numerical_operations_performed": False,
        "smoke_only": smoke,
        "smoke_gate_sha256": smoke_sha,
        "all_identity_and_protocol_gates_pass": True,
        "elapsed_s": time.perf_counter() - started,
    }
    return payload, output


def _validate_shard(
    path: Path, start: int, stop: int
) -> tuple[dict[str, Any], dict[str, Any], str]:
    value, digest = read_authenticated_json(path, f"fresh shard {start}:{stop}")
    expected_pairs = [(task, episode) for task in range(start, stop) for episode in range(50)]
    episodes = value.get("episodes")
    observed_pairs: list[tuple[int, int]] = []
    valid_episodes = isinstance(episodes, list) and len(episodes) == len(expected_pairs)
    if valid_episodes:
        for record in episodes:
            if not isinstance(record, dict):
                valid_episodes = False
                break
            pair = (record.get("task_index"), record.get("episode"))
            observed_pairs.append(pair)
            initial_sha = record.get("initial_state_sha256")
            elapsed = record.get("elapsed_s")
            if (
                type(pair[0]) is not int
                or type(pair[1]) is not int
                or type(record.get("success")) is not bool
                or type(record.get("steps")) is not int
                or type(record.get("terminated_without_success")) is not bool
                or (
                    record.get("terminated_without_success") is True
                    and record.get("success") is True
                )
                or not 0 <= record["steps"] <= MAX_STEPS
                or not isinstance(initial_sha, str)
                or len(initial_sha) != 64
                or any(character not in "0123456789abcdef" for character in initial_sha)
                or not isinstance(elapsed, (int, float))
                or isinstance(elapsed, bool)
                or not math.isfinite(float(elapsed))
                or elapsed < 0
            ):
                valid_episodes = False
    task_protocol = value.get("task_protocol")
    expected_protocol = {str(task): _expected_task_record(task) for task in range(start, stop)}
    recomputed_successes = (
        sum(int(record["success"]) for record in episodes)
        if valid_episodes and isinstance(episodes, list)
        else -1
    )
    expected_per_task = (
        {
            str(task): sum(
                int(record["success"])
                for record in episodes
                if record["task_index"] == task
            )
            / 50
            for task in range(start, stop)
        }
        if valid_episodes and isinstance(episodes, list)
        else {}
    )
    gates = {
        "schema": value.get("schema") == "xvla_capable_linear_b1c0_fresh_capability_shard_v1",
        "checkpoint": value.get("checkpoint_sha256") == CHECKPOINT_SHA256,
        "config_identity": value.get("checkpoint", {}).get("config_identity")
        == EXPECTED_CONFIG_IDENTITY,
        "training_record": value.get("training_record_sha256") == TRAINING_RECORD_SHA256,
        "range": (value.get("task_start"), value.get("task_end")) == (start, stop),
        "seed": value.get("seed") == 0,
        "protocol": value.get("protocol", {}).get("suite") == "libero_object"
        and value.get("protocol", {}).get("resolution") == RESOLUTION
        and value.get("protocol", {}).get("action_horizon") == ACTION_HORIZON
        and value.get("protocol", {}).get("episodes_per_task") == 50
        and value.get("protocol", {}).get("max_steps") == MAX_STEPS
        and value.get("protocol", {}).get("settle_steps") == SETTLE_STEPS
        and value.get("protocol", {}).get("settle_success_or_done_count") == 0
        and value.get("protocol", {}).get("execution_horizon") == EXECUTION_HORIZON
        and value.get("protocol", {}).get("canonical_init_states") is True
        and value.get("protocol", {}).get("observation_transform")
        == "agentview_image[::-1,::-1] then same-size PIL resize"
        and value.get("protocol", {}).get("gripper_decode")
        == "+1 iff predicted coordinate > 0, else -1"
        and value.get("protocol", {}).get("matmul_precision") == "highest"
        and value.get("protocol", {}).get("cuda_matmul_allow_tf32") is False
        and value.get("protocol", {}).get("cudnn_allow_tf32") is True
        and value.get("protocol", {}).get("deterministic_algorithms") is True
        and value.get("protocol", {}).get("cudnn_benchmark") is False
        and value.get("protocol", {}).get("cudnn_deterministic") is True
        and value.get("protocol", {}).get("cublas_workspace_config")
        == EXPECTED_CUBLAS_WORKSPACE_CONFIG
        and value.get("protocol", {}).get("terminal_transition_stops_episode")
        is True
        and value.get("protocol", {}).get(
            "historical_protocol_with_corrected_terminal_handling"
        )
        is True
        and value.get("protocol", {}).get(
            "historical_evaluator_control_flow_reproduced"
        )
        is False
        and value.get("protocol", {}).get(
            "historical_hardware_reproduction_claimed"
        )
        is False
        and value.get("protocol", {}).get("fresh_hardware") == EXPECTED_GPU_NAME,
        "task_protocol": task_protocol == expected_protocol,
        "episodes": valid_episodes
        and _episode_inventory_is_exact(episodes, start, stop, 50, MAX_STEPS)
        and observed_pairs == expected_pairs,
        "episode_digest": value.get("episode_identity_sha256") == _canonical_sha256(expected_pairs),
        "trial_count": value.get("trials") == len(expected_pairs),
        "success_count": type(value.get("successes")) is int
        and value.get("successes") == recomputed_successes,
        "overall": isinstance(value.get("overall"), (int, float))
        and not isinstance(value.get("overall"), bool)
        and value.get("overall") == recomputed_successes / len(expected_pairs),
        "per_task": value.get("per_task") == expected_per_task,
        "early_terminal_count": value.get("early_terminal_failures")
        == (
            sum(
                int(record["terminated_without_success"])
                for record in episodes
            )
            if valid_episodes and isinstance(episodes, list)
            else -1
        ),
        "smoke_gate": value.get("smoke_gate_sha256") == _validate_smoke(),
        "identity_gates": value.get("all_identity_and_protocol_gates_pass") is True,
        "checkpoint_twice": value.get("exact_checkpoint_digest_verified_before_and_after") is True,
        "ledger_twice": value.get("stage_ledger_verified_before_and_after") is True,
        "simulator_boundary": value.get("simulator_external_to_odt") is True
        and value.get("external_simulator_outside_weight_only_odt_closure") is True
        and value.get("canonical_odt_runtime_guard_installed") is False
        and value.get("odt_runtime_compliance_claimed") is False,
        "environment": _environment_record_is_exact(value.get("environment")),
        "static_audit": value.get("static_audit") == _STATIC_AUDIT,
        "inactive_norm_sentinel": _inactive_norm_sentinel_is_exact(
            value.get("inactive_vision_norm_sentinel")
        ),
    }
    label = f"shard_{start}_{stop}"
    assert_physical_hashes_unchanged({label: path}, {label: digest})
    return value, gates, digest


def _aggregate() -> tuple[dict[str, Any], Path]:
    output = RESULT_DIRECTORY / "fresh_capability_aggregate.json"
    if os.path.lexists(output):
        raise FileExistsError(f"refusing existing aggregate output {output}")
    all_episodes: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    shard_environments: list[dict[str, Any]] = []
    shard_norm_sentinels: list[dict[str, Any]] = []
    gates: dict[str, bool] = {}
    for start, stop in SHARDS:
        path = _shard_output(start, stop)
        value, shard_gates, shard_digest = _validate_shard(path, start, stop)
        for name, passed in shard_gates.items():
            gates[f"shard_{start}_{stop}_{name}"] = passed
        shard_records.append({
            "task_start": start,
            "task_end": stop,
            "sha256": shard_digest,
            "successes": value.get("successes"),
            "trials": value.get("trials"),
        })
        if isinstance(value.get("episodes"), list):
            all_episodes.extend(value["episodes"])
        if isinstance(value.get("environment"), dict):
            shard_environments.append(value["environment"])
        if isinstance(value.get("inactive_vision_norm_sentinel"), dict):
            shard_norm_sentinels.append(value["inactive_vision_norm_sentinel"])
    expected_pairs = [(task, episode) for task in range(10) for episode in range(50)]
    observed_pairs = [(record.get("task_index"), record.get("episode")) for record in all_episodes]
    successes = sum(int(record.get("success") is True) for record in all_episodes)
    per_task_successes = {
        str(task): sum(
            int(record.get("success") is True)
            for record in all_episodes
            if record.get("task_index") == task
        )
        for task in range(10)
    }
    per_task = {
        task: value / EPISODES_PER_TASK
        for task, value in per_task_successes.items()
    }
    early_terminal_failures = sum(
        int(record.get("terminated_without_success") is True)
        for record in all_episodes
    )
    gates.update({
        "complete_episode_inventory": _episode_inventory_is_exact(
            all_episodes, 0, 10, 50, MAX_STEPS
        )
        and observed_pairs == expected_pairs,
        "exactly_500_trials": len(all_episodes) == 500,
        "success_count_bounded": 0 <= successes <= 500,
        "per_task_counts_complete": set(per_task_successes) == {
            str(task) for task in range(10)
        }
        and sum(per_task_successes.values()) == successes
        and all(0 <= value <= EPISODES_PER_TASK for value in per_task_successes.values()),
        "early_terminal_failures_bounded": 0 <= early_terminal_failures <= 500 - successes,
        "identical_pinned_shard_environments": len(shard_environments) == len(SHARDS)
        and all(
            _environment_record_is_exact(environment)
            and environment == shard_environments[0]
            for environment in shard_environments
        ),
        "inactive_vision_norm_never_executed": len(shard_norm_sentinels)
        == len(SHARDS)
        and all(
            _inactive_norm_sentinel_is_exact(sentinel)
            for sentinel in shard_norm_sentinels
        ),
        "capability_floor_80_percent": successes >= 400,
        "checkpoint_current": _sha256(CHECKPOINT) == CHECKPOINT_SHA256,
        "training_record_current": _sha256(TRAINING_RECORD) == TRAINING_RECORD_SHA256,
    })
    sources = _fixed_sources()
    payload = {
        "schema": "xvla_capable_linear_b1c0_fresh_capability_aggregate_v1",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "training_record_sha256": TRAINING_RECORD_SHA256,
        "fresh_evaluation_sha256_bound_to_checkpoint": True,
        "config_identity": EXPECTED_CONFIG_IDENTITY,
        "historical_evaluation_digest_binding_claimed": False,
        "protocol": {
            "suite": "libero_object",
            "resolution": RESOLUTION,
            "action_horizon": ACTION_HORIZON,
            "tasks": list(range(10)),
            "canonical_initial_states_per_task": list(range(50)),
            "canonical_init_states": True,
            "episodes_per_task": 50,
            "trials": 500,
            "max_steps": MAX_STEPS,
            "settle_steps": SETTLE_STEPS,
            "settle_success_or_done_count": 0,
            "execution_horizon": EXECUTION_HORIZON,
            "observation_transform": "agentview_image[::-1,::-1] then same-size PIL resize",
            "gripper_decode": "+1 iff predicted coordinate > 0, else -1",
            "matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": True,
            "deterministic_algorithms": True,
            "cublas_workspace_config": EXPECTED_CUBLAS_WORKSPACE_CONFIG,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "historical_evaluator_control_flow_reproduced": False,
            "historical_protocol_with_corrected_terminal_handling": True,
            "terminal_transition_stops_episode": True,
            "historical_hardware_reproduction_claimed": False,
            "fresh_hardware": EXPECTED_GPU_NAME,
        },
        "shards": shard_records,
        "inactive_vision_norm_sentinel": {
            "site": "vision.norm_out",
            "total_site_count": 74,
            "inactive_sites": ["vision.norm_out"],
            "active_site_count": 73,
            "shard_count": len(shard_norm_sentinels),
            "all_shard_call_counts_zero": len(shard_norm_sentinels) == len(SHARDS)
            and all(sentinel.get("call_count") == 0 for sentinel in shard_norm_sentinels),
            "all_shard_active_sets_observed_once": len(shard_norm_sentinels)
            == len(SHARDS)
            and all(
                sentinel.get("active_call_count") == 73
                and sentinel.get("every_active_site_call_count") == 1
                for sentinel in shard_norm_sentinels
            ),
        },
        "environment": (
            shard_environments[0]
            if len(shard_environments) == len(SHARDS)
            else None
        ),
        "smoke_gate_sha256": _validate_smoke(),
        "shard_result_bundle_sha256": _canonical_sha256({
            f"{record['task_start']}:{record['task_end']}": record["sha256"]
            for record in shard_records
        }),
        "successes": successes,
        "trials": len(all_episodes),
        "success_rate": successes / len(all_episodes) if all_episodes else None,
        "per_task_successes": per_task_successes,
        "per_task": per_task,
        "episodes": all_episodes,
        "early_terminal_failures": early_terminal_failures,
        "capability_floor": CAPABILITY_FLOOR,
        "capability_floor_passed": successes >= 400 and len(all_episodes) == 500,
        "episode_identity_sha256": _canonical_sha256(observed_pairs),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "source_manifest_sha256": _PREIMPORT["source_manifest_sha256"],
        "stage_ledger_sha256": _PREIMPORT["stage_ledger_sha256"],
        "static_audit": sources["audit"],
        "canonical_odt_runtime_guard_installed": False,
        "odt_runtime_compliance_claimed": False,
        "simulator_external_to_odt": True,
        "external_simulator_outside_weight_only_odt_closure": True,
        "historical_hardware_reproduction_claimed": False,
        "fresh_all_a30_hash_bound_reevaluation": True,
        "historical_hardware_was_mixed_and_is_comparison_only": True,
        "determinism_claim_scope": (
            "Deterministic only under the pinned A30, software, CUDA, and backend protocol; "
            "no cross-host or cross-driver bitwise reproducibility is claimed."
        ),
        "claim": (
            "Fresh closed-loop evaluation of the exact staged b1c0 checkpoint on all 500 "
            "official LIBERO-Object task and packaged-initial-state pairs, deterministic "
            "under the pinned A30, software, CUDA, and backend protocol."
        ),
    }
    return payload, output


def _current_shard_hashes() -> dict[str, str]:
    paths = {
        f"{start}:{stop}": _shard_output(start, stop)
        for start, stop in SHARDS
    }
    return assert_physical_hashes_unchanged(paths)


def _assert_aggregate_shards_unchanged(
    payload: Mapping[str, Any], expected: Mapping[str, str] | None = None
) -> dict[str, str]:
    records = payload.get("shards")
    if not isinstance(records, list):
        raise RuntimeError("aggregate payload omits its shard inventory")
    recorded: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("aggregate shard record is malformed")
        key = f"{record.get('task_start')}:{record.get('task_end')}"
        digest = record.get("sha256")
        if key in recorded or not isinstance(digest, str):
            raise RuntimeError("aggregate shard inventory is duplicate or malformed")
        recorded[key] = digest
    observed = _current_shard_hashes()
    if observed != recorded or (expected is not None and observed != dict(expected)):
        raise RuntimeError("fresh shard bytes changed across aggregate publication")
    return observed


def _assert_aggregate_smoke_unchanged(
    payload: Mapping[str, Any], expected: Mapping[str, str] | None = None
) -> dict[str, str]:
    digest = payload.get("smoke_gate_sha256")
    if not isinstance(digest, str):
        raise RuntimeError("aggregate payload omits its smoke identity")
    observed = assert_physical_hashes_unchanged(
        {"smoke": RESULT_DIRECTORY / "fresh_capability_smoke.json"}
    )
    if observed != {"smoke": digest} or (
        expected is not None and observed != dict(expected)
    ):
        raise RuntimeError("fresh smoke bytes changed across aggregate publication")
    return observed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", choices=("smoke", "shard", "aggregate"), required=True)
    parser.add_argument("--task-start", type=int)
    parser.add_argument("--task-end", type=int)
    arguments = sys.argv[1:]
    parsed = parser.parse_args(arguments)
    for option in ("--mode", "--task-start", "--task-end"):
        count = sum(value == option or value.startswith(f"{option}=") for value in arguments)
        if option == "--mode" and count != 1:
            parser.error("--mode must occur exactly once")
        if option != "--mode" and count > 1:
            parser.error(f"{option} may occur at most once")
    if parsed.mode == "smoke" and (parsed.task_start is not None or parsed.task_end is not None):
        parser.error("smoke mode accepts no task range")
    if parsed.mode == "shard" and (parsed.task_start, parsed.task_end) not in SHARDS:
        parser.error("shard mode requires one exact preregistered task range")
    if parsed.mode == "aggregate" and (parsed.task_start is not None or parsed.task_end is not None):
        parser.error("aggregate mode accepts no task range")
    return parsed


def main() -> None:
    args = _parse_args()
    if args.mode == "smoke":
        payload, output = _run_evaluation("smoke", 0, 1)
    elif args.mode == "shard":
        payload, output = _run_evaluation(
            "shard", args.task_start, args.task_end
        )
    else:
        payload, output = _aggregate()
    aggregate_inputs = None
    aggregate_smoke = None
    if args.mode == "aggregate":
        aggregate_inputs = _assert_aggregate_shards_unchanged(payload)
        aggregate_smoke = _assert_aggregate_smoke_unchanged(payload)
    digest = _publish(output, payload)
    try:
        sources = _fixed_sources()
        if not sources:
            raise RuntimeError("post-publication capability verification failed")
        if aggregate_inputs is not None:
            _assert_aggregate_shards_unchanged(payload, aggregate_inputs)
        if aggregate_smoke is not None:
            _assert_aggregate_smoke_unchanged(payload, aggregate_smoke)
        if _sha256(output) != digest or output.stat().st_mode & 0o222:
            raise RuntimeError("published capability artifact is not immutable")
    except BaseException:
        quarantine = output.with_name(f"{output.name}.invalid.{os.getpid()}")
        os.replace(output, quarantine)
        raise
    print(json.dumps({"output": output.as_posix(), "sha256": digest}, sort_keys=True), flush=True)
    if payload.get("all_gates_pass") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
