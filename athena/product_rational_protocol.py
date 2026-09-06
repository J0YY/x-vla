#!/usr/bin/env python3
"""Frozen protocol helpers for the ProductRoutingHead + Padé-rational VLA lane.

This module deliberately contains only training/evaluation provenance, configuration,
and validation code.  Canonical ODT is a later consumer of the deployment checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    import torch
    from xvla.models.vla import ChiVLA, VLAConfig


SCHEMA = "xvla_product_pade_rational_vla_v2"
RECIPE_VERSION = "product_pade_rational_object_40k_v2_postema_anchor"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant is forbidden: {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Nonfinite JSON float is forbidden")
    return parsed
SUITE = "libero_object"
SEEDS = (0, 1, 2)
SHARDS = ((0, 3), (3, 6), (6, 8), (8, 10))
DEFAULT_RUN_ROOT = Path("athena/results/product_pade_rational_v2")
FROZEN_STAGE_ROOT = Path("/work/joy/x-vla-product-pade-rational-v2")

CACHE_PATH = Path("artifacts/libero_frames_100000_64.pkl")
CACHE_SHA256 = "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662"
CACHE_FRAME_COUNT = 66984
CACHE_SAMPLE_COUNT = 63352
DATASET_REPOSITORY = "lerobot/libero_object_image"
DATASET_REVISION = "e1e080d7df1d0a359dff5c86c222e047549f447f"
DATASET_METADATA_SHA256 = "34caee9641ae50bb4e077de306a7d0031753757882da8b1f117e7ea36a486b42"
PROVENANCE_PATH = Path("athena/results/cache_provenance_libero_object.json")
PROVENANCE_SHA256 = "1e3ed7eaeef317a221bb6649ea75ef68ff924b57797682e853b4bd90a138f4c4"
CANONICAL_CACHE_CONTENT_SHA256 = "01bc724b9bf8c158b983e34b82bbb40dce9dba2cb86dd9be2a5bf8910be341c7"
DATASET_TO_OFFICIAL_TASK = {
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
}
OFFICIAL_TASK_LANGUAGES = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    2: "pick up the salad dressing and place it in the basket",
    3: "pick up the bbq sauce and place it in the basket",
    4: "pick up the ketchup and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
    7: "pick up the milk and place it in the basket",
    8: "pick up the chocolate pudding and place it in the basket",
    9: "pick up the orange juice and place it in the basket",
}
OFFICIAL_TASK_LANGUAGES_SHA256 = (
    "73bf8b4fae43a1940b48fb5d3a7227703d5d1bb1b3ae4efc2108460b66bf5c36"
)
VOCAB_SHA256 = "1e353b073134c6a1e62ca0c2ee0e9ad7204a007fd343b1bc6e09075ec0a8f2fe"
OFFICIAL_INIT_STATE_ROW_HASH_LIST_SHA256 = {
    0: "9ef09c00e35ed5beda072328143a3b6ce962c58b714c2c53c2711c29066dbf7d",
    1: "09d76bb222136707ddf9cf3d688ac3fa456d60818ac01067bbb167f1f08e2ae7",
    2: "2194addd7641a43d3ca751d84410a3b4229e6ac0da596970d188e17a968def2d",
    3: "2dc2f081b68d62e5d4a714886939b4f191b922516f3df107acd00d2837b09ec8",
    4: "703790cfc208449cd8dc2503125d40c6a2d4d247a58c600ede13f4a785f41a60",
    5: "69ddf8dd1494fc054659e765962da0ba3a519692abcb71939d06cc58e8dccb96",
    6: "8464f4f0f67c056268eb1894230e457558bdd0714780ed2c35973545ba3eeeec",
    7: "56a550b5aeb73f009cce0076d78e001f30122d8b303e889faa5c58397831bef4",
    8: "75083bc1ac59459eb8fe0e54018ed1fb79d614ee54560a94a677801627ad0502",
    9: "4117ca940b7388a9a2abe308722c921c71462050a8cc2b44394a6c98467ed445",
}
EXPECTED_TASK_PROTOCOL = {
    0: {
        "language": OFFICIAL_TASK_LANGUAGES[0],
        "problem_folder": "libero_object",
        "bddl_file": "pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl",
        "bddl_sha256": "df088984da13131f8332ee0f13a7896c6a97afd02ee5007a42e8fc5e0893571e",
        "init_state_count": 50,
        "init_states_sha256": "91eed35ff60c2b791ba1d80996e359177a9202d81358e9fddb42ccafffa3f34e",
    },
    1: {
        "language": OFFICIAL_TASK_LANGUAGES[1],
        "problem_folder": "libero_object",
        "bddl_file": "pick_up_the_cream_cheese_and_place_it_in_the_basket.bddl",
        "bddl_sha256": "7019f37ee158d67a21338a6df0c441dd8f84979b946b5e20bb49463ef0508ea2",
        "init_state_count": 50,
        "init_states_sha256": "6443945d386a843b9998caeee65f3093f50e4ab65e913e7102aa39eaee7edd59",
    },
    2: {
        "language": OFFICIAL_TASK_LANGUAGES[2],
        "problem_folder": "libero_object",
        "bddl_file": "pick_up_the_salad_dressing_and_place_it_in_the_basket.bddl",
        "bddl_sha256": "024978e9e8f43a49b49d0965f4f426d364ce056307f0fcf5733dc4682fac437a",
        "init_state_count": 50,
        "init_states_sha256": "7f365994bb15f09bed7e4611603d8b5e4a3576f7fccb827241f6ad60cb8b3e3f",
    },
    3: {
        "language": OFFICIAL_TASK_LANGUAGES[3],
        "problem_folder": "libero_object",
        "bddl_file": "pick_up_the_bbq_sauce_and_place_it_in_the_basket.bddl",
        "bddl_sha256": "8a7e37b76fce5621e649e260dbcfcec7ce2f9a055ca09a4f1ebaf5ca74a739c2",
        "init_state_count": 50,
        "init_states_sha256": "b186084cc72f9cf8c07ca98c18542f4f0269b23f53bf98ac8474d1cc5b642702",
    },
    4: {
        "language": OFFICIAL_TASK_LANGUAGES[4],
        "problem_folder": "libero_object",
        "bddl_file": "pick_up_the_ketchup_and_place_it_in_the_basket.bddl",
        "bddl_sha256": "4a4e545483a3fe30cf0ef3dc05bfe5e6ad5e37c94b4376dfa73fe1b60e75b319",
        "init_state_count": 50,
        "init_states_sha256": "8a035ce260650b525cbdec044e777f2625e689aca72dd737983c4a61539d5ab8",
    },
    5: {
        "language": OFFICIAL_TASK_LANGUAGES[5],
        "problem_folder": "libero_object",
        "bddl_file": "pick_up_the_tomato_sauce_and_place_it_in_the_basket.bddl",
        "bddl_sha256": "b1f4bb69d256a05f693838de46182bb0d3de0de68eec350df1b5b1e76f6c136d",
        "init_state_count": 50,
        "init_states_sha256": "2015e882bad1d77efd5b973a070fd6bd4cf524f35c36497f5060b1acc7b59dcb",
    },
    6: {
        "language": OFFICIAL_TASK_LANGUAGES[6],
        "problem_folder": "libero_object",
        "bddl_file": "pick_up_the_butter_and_place_it_in_the_basket.bddl",
        "bddl_sha256": "5d8053b40ff33246fdbcc38548db84032c6bb647b4b88b6646d9cf6b73a5006c",
        "init_state_count": 50,
        "init_states_sha256": "981f9d28ca49331ad5faf46ff765233fe7d4fa39a3f35dd332104f900c8cc766",
    },
    7: {
        "language": OFFICIAL_TASK_LANGUAGES[7],
        "problem_folder": "libero_object",
        "bddl_file": "pick_up_the_milk_and_place_it_in_the_basket.bddl",
        "bddl_sha256": "9910aabf6717e8ba3e24a3f8500d9bce9ee075346ea6961e8fd766f1257d2996",
        "init_state_count": 50,
        "init_states_sha256": "f78700be5769be90421173e1689e458b65f4d791855aedc3d2dedc63e3b7a0aa",
    },
    8: {
        "language": OFFICIAL_TASK_LANGUAGES[8],
        "problem_folder": "libero_object",
        "bddl_file": "pick_up_the_chocolate_pudding_and_place_it_in_the_basket.bddl",
        "bddl_sha256": "674ceabc400b7b16d46e8eaf678709d4a633235c7e0a80233775044fe38b19b0",
        "init_state_count": 50,
        "init_states_sha256": "54593bb836dc12d321d9e7971e54034bbe01c02929cf0c09cb93b75038bedf5a",
    },
    9: {
        "language": OFFICIAL_TASK_LANGUAGES[9],
        "problem_folder": "libero_object",
        "bddl_file": "pick_up_the_orange_juice_and_place_it_in_the_basket.bddl",
        "bddl_sha256": "6298533e7bcfb83e40779a77fda39216e8cd53f22bf1af6388585dad0abb0b50",
        "init_state_count": 50,
        "init_states_sha256": "c198880f92818df55d0a8e53ab739bbfd36acfab236a276c7ac72a27a27baf0d",
    },
}

RESOLUTION = 64
PATCH_SIZE = 8
VIT_DIM = 192
VIT_LAYERS = 4
VIT_HEADS = 8
VIT_FFN_RANK = 576
MODEL_DIM = 384
MODEL_LAYERS = 8
MODEL_HEADS = 12
MODEL_FFN_RANK = 1152
ACTION_HORIZON = 8
ACTION_DIM = 7
STATE_DIM = 8
N_FACTORS = 4
HEAD_RANK = 1152
MAX_INSTRUCTION_LENGTH = 32

FULL_STEPS = 40000
SMOKE_STEPS = 10
FULL_BATCH_SIZE = 256
SMOKE_BATCH_SIZE = 256
LEARNING_RATE = 8e-4
MIN_LEARNING_RATE_RATIO = 0.1
WARMUP_FRACTION = 0.05
EMA_DECAY = 0.999
WEIGHT_DECAY = 0.05
GRADIENT_CLIP_NORM = 1.0
FINAL_CALIBRATION_PASSES = 4
FINAL_CALIBRATION_BATCH_SIZE = 256
FINAL_CALIBRATION_AUDIT_MAX_RELATIVE_CHANGE = 0.05
PRECALIBRATION_ARTIFACT_ROLE = "training_only_precalibration_ema_state"
PRECALIBRATION_SCHEMA = f"{SCHEMA}_{PRECALIBRATION_ARTIFACT_ROLE}_v1"
DEPLOYMENT_COMPLETION_SCHEMA = f"{SCHEMA}_deployment_completion_v1"
CALIBRATION_RESUME_PROOF_SCHEMA = (
    f"{SCHEMA}_same_allocation_calibration_resume_proof_v1"
)
CALIBRATION_INITIALIZER_LABEL = (
    "training_time_activation_ema_running_ms_retained_after_final_parameter_ema_swap"
)
CALIBRATION_RECOVERY_SCOPE = "same_source_manifest_transient_retry_only"
PRODUCT_COMPONENT_SITE_NAMES = (
    "product_center",
    *(f"product_factor_{index}" for index in range(N_FACTORS)),
    *(f"product_gate_{index}" for index in range(N_FACTORS)),
)

EVALUATION_EPISODES_PER_TASK = 50
EVALUATION_MAX_STEPS = 280
EVALUATION_SETTLE_STEPS = 10
EVALUATION_EXEC_HORIZON = 8
EVALUATION_MATMUL_PRECISION = "highest"
EXPECTED_EVALUATION_ENVIRONMENT = {
    "gpu": "Quadro RTX 6000",
    "inference_dtype": "float32",
    "matmul_precision": EVALUATION_MATMUL_PRECISION,
    "python": "3.10.19",
    "torch": "2.7.1+cu126",
    "cuda": "12.6",
    "numpy": "1.26.4",
    "libero": "0.1.0",
    "robosuite": "1.4.1",
    "mujoco": "3.5.0",
    "mujoco_gl": "egl",
}
PRIMARY_CAPABILITY_SEED = 0
PRIMARY_CAPABILITY_FLOOR = 0.80

EXPECTED_PADE_SITE_COUNT = 73
EXPECTED_TOTAL_PADE_MODULE_COUNT = 74
PRODUCT_RUNNING_MS_IDENTITY_FLOOR = 1e-12
DEPLOYMENT_CLAIM_BOUNDARY = (
    "The deployment checkpoint contains the unchanged trained vision tower, joint "
    "backbone, and ProductRoutingHead center, every factor, and every gate, with "
    "Padé-compatible rational normalization at all active sites. The sole "
    "out-of-graph policy operation is the fixed deterministic boundary "
    "sign = +1 iff gate_logit > 0, else -1, followed by the signed factor sum."
)
CAPABILITY_SIMULATOR_BOUNDARY = (
    "This process is an official closed-loop capability measurement, not an ODT "
    "numerical process. The external LIBERO, robosuite, and MuJoCo simulator and "
    "controller are outside the authenticated weight-only ODT source closure. Only "
    "the separately launched canonical ODT process may certify direct-only Algorithms "
    "1 through 3. Simulator computations and state never enter that weight-only proof."
)

# Exact Python closure for the isolated lane. The static direct-only audit must return
# this precise mapping, not a subset. Non-Python immutable inputs and launch wrappers
# are authenticated separately in the same manifest.
SOURCE_CLOSURE = (
    "athena/__init__.py",
    "athena/aggregate_product_rational_results.py",
    "athena/eval_product_rational_checkpoint.py",
    "athena/prepare_product_rational_launch.py",
    "athena/product_rational_protocol.py",
    "athena/stage_product_rational_launch.py",
    "athena/train_product_rational_checkpoint.py",
    "scripts/__init__.py",
    "scripts/odt_direct_only_compliance.py",
    "tests/test_product_rational_production_lane.py",
    "xvla/__init__.py",
    "xvla/models/__init__.py",
    "xvla/models/lm.py",
    "xvla/models/vit.py",
    "xvla/models/vla.py",
    "xvla/nn/__init__.py",
    "xvla/nn/attention.py",
    "xvla/nn/baselines.py",
    "xvla/nn/bilinear.py",
    "xvla/nn/block.py",
    "xvla/nn/flow_action.py",
    "xvla/nn/homogeneous.py",
    "xvla/nn/normalization.py",
    "xvla/nn/product_routing.py",
    "xvla/nn/projector.py",
    "xvla/nn/quantile_action.py",
)
AUTHENTICATED_INPUTS = (
    "athena/results/cache_provenance_libero_object.json",
)
LAUNCH_CLOSURE = (
    "athena/slurm_product_rational_aggregate.sbatch",
    "athena/slurm_product_rational_eval.sbatch",
    "athena/slurm_product_rational_preflight.sbatch",
    "athena/slurm_product_rational_train.sbatch",
    "athena/submit_product_rational_v2.sh",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_type_exact_equal(observed: Any, expected: Any) -> bool:
    """Compare JSON-domain values without bool/int or int/float aliasing."""
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            json_type_exact_equal(observed[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            json_type_exact_equal(left, right)
            for left, right in zip(observed, expected)
        )
    if isinstance(expected, float):
        return math.isfinite(observed) and math.isfinite(expected) and observed == expected
    return observed == expected


_STATIC_AUDIT_KEYS = {
    "scope",
    "entrypoints",
    "source_sha256",
    "source_count",
    "direct_qr_call_sites",
    "direct_qr_required",
    "guarded_dormant_spectral_norm_sites",
    "duplicate_top_level_definition_sites",
    "prohibited_self_overlap_sites",
    "prohibited_calls_found",
    "call_site_count",
}


def static_audit_record_is_closed(
    report: Any,
    *,
    expected_source_closure: Mapping[str, str],
    direct_qr_required: bool,
    direct_qr_call_sites: int,
) -> bool:
    """Validate the complete static-audit record with JSON-exact types."""
    return (
        isinstance(report, Mapping)
        and set(report) == _STATIC_AUDIT_KEYS
        and report.get("scope") == "transitive_local_import_closure"
        and json_type_exact_equal(
            report.get("entrypoints"), sorted(expected_source_closure)
        )
        and json_type_exact_equal(
            report.get("source_sha256"), dict(expected_source_closure)
        )
        and type(report.get("source_count")) is int
        and report.get("source_count") == len(expected_source_closure)
        and type(report.get("direct_qr_call_sites")) is int
        and report.get("direct_qr_call_sites") == direct_qr_call_sites
        and report.get("direct_qr_required") is direct_qr_required
        and json_type_exact_equal(report.get("prohibited_calls_found"), [])
        and json_type_exact_equal(report.get("prohibited_self_overlap_sites"), [])
        and json_type_exact_equal(
            report.get("guarded_dormant_spectral_norm_sites"), []
        )
        and json_type_exact_equal(
            report.get("duplicate_top_level_definition_sites"), []
        )
        and type(report.get("call_site_count")) is int
        and report.get("call_site_count") >= direct_qr_call_sites
    )


def runtime_guard_record_is_closed(
    report: Any,
    *,
    expected_entrypoints: tuple[str, ...],
    expected_entrypoint_count: int,
) -> bool:
    return (
        isinstance(report, Mapping)
        and set(report)
        == {
            "installed",
            "patched_entrypoints",
            "patched_entrypoint_count",
            "allowed_call_count",
            "allowed_calls",
            "prohibited_attempt_count",
            "prohibited_attempts",
        }
        and report.get("installed") is True
        and json_type_exact_equal(
            report.get("patched_entrypoints"), sorted(expected_entrypoints)
        )
        and type(report.get("patched_entrypoint_count")) is int
        and report.get("patched_entrypoint_count") == expected_entrypoint_count
        and type(report.get("allowed_call_count")) is int
        and report.get("allowed_call_count") == 0
        and json_type_exact_equal(report.get("allowed_calls"), [])
        and type(report.get("prohibited_attempt_count")) is int
        and report.get("prohibited_attempt_count") == 0
        and json_type_exact_equal(report.get("prohibited_attempts"), [])
    )


def read_json_object_physical(path: Path, *, label: str) -> dict[str, Any]:
    """Read one physical JSON object while rejecting duplicate keys at every depth."""
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is missing or nonphysical: {path}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"Duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(),
        object_pairs_hook=reject_duplicates,
        parse_constant=_reject_json_constant,
        parse_float=_finite_json_float,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return payload


def closure_snapshot(
    paths: tuple[str, ...], root: Path | None = None
) -> dict[str, str]:
    root = repository_root() if root is None else root.resolve()
    result: dict[str, str] = {}
    for relative in paths:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Production lane file is missing or nonphysical: {relative}")
        result[relative] = file_sha256(path)
    return result


def source_snapshot(root: Path | None = None) -> dict[str, str]:
    return closure_snapshot(SOURCE_CLOSURE, root)


def source_bundle_sha256(snapshot: Mapping[str, str]) -> str:
    return canonical_sha256(dict(sorted(snapshot.items())))


def manifest_bundle_sha256(payload: Mapping[str, Any]) -> str:
    bound = {
        "schema": payload.get("schema"),
        "source_closure": payload.get("source_closure"),
        "source_bundle_sha256": payload.get("source_bundle_sha256"),
        "authenticated_inputs": payload.get("authenticated_inputs"),
        "authenticated_inputs_bundle_sha256": payload.get(
            "authenticated_inputs_bundle_sha256"
        ),
        "launch_closure": payload.get("launch_closure"),
        "launch_bundle_sha256": payload.get("launch_bundle_sha256"),
    }
    return canonical_sha256(bound)


def write_source_manifest_exclusive(path: Path, root: Path | None = None) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite source manifest {path}")
    sources = source_snapshot(root)
    inputs = closure_snapshot(AUTHENTICATED_INPUTS, root)
    launch = closure_snapshot(LAUNCH_CLOSURE, root)
    payload = {
        "schema": f"{SCHEMA}_source_manifest",
        "source_closure": sources,
        "source_bundle_sha256": source_bundle_sha256(sources),
        "authenticated_inputs": inputs,
        "authenticated_inputs_bundle_sha256": source_bundle_sha256(inputs),
        "launch_closure": launch,
        "launch_bundle_sha256": source_bundle_sha256(launch),
    }
    payload["manifest_bundle_sha256"] = manifest_bundle_sha256(payload)
    write_json_exclusive(path, payload)
    os.chmod(path, 0o444)
    return payload


def load_and_verify_source_manifest(
    path: Path, root: Path | None = None
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Source manifest is missing or nonphysical: {path}")
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"Duplicate source-manifest key {key!r}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(),
        object_pairs_hook=reject_duplicates,
        parse_constant=_reject_json_constant,
        parse_float=_finite_json_float,
    )
    if payload.get("schema") != f"{SCHEMA}_source_manifest":
        raise RuntimeError("Source manifest schema differs")
    sections = (
        ("source_closure", SOURCE_CLOSURE, "source_bundle_sha256"),
        (
            "authenticated_inputs",
            AUTHENTICATED_INPUTS,
            "authenticated_inputs_bundle_sha256",
        ),
        ("launch_closure", LAUNCH_CLOSURE, "launch_bundle_sha256"),
    )
    for section, frozen_paths, bundle_key in sections:
        expected = payload.get(section)
        if not isinstance(expected, dict) or tuple(sorted(expected)) != tuple(
            sorted(frozen_paths)
        ):
            raise RuntimeError(f"Source manifest {section} differs from the frozen lane")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in expected.values()
        ):
            raise RuntimeError(f"Source manifest {section} has an invalid SHA-256")
        if payload.get(bundle_key) != source_bundle_sha256(expected):
            raise RuntimeError(f"Source manifest {bundle_key} is invalid")
        observed = closure_snapshot(frozen_paths, root)
        if observed != expected:
            changed = {
                key: {"expected": expected.get(key), "observed": observed.get(key)}
                for key in sorted(set(expected) | set(observed))
                if expected.get(key) != observed.get(key)
            }
            raise RuntimeError(f"Production {section} drift: {changed}")
    if payload.get("manifest_bundle_sha256") != manifest_bundle_sha256(payload):
        raise RuntimeError("Source manifest aggregate bundle digest is invalid")
    return payload


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Refusing stale temporary output {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def publish_checkpoint_exclusive(path: Path, state: Mapping[str, torch.Tensor]) -> None:
    import torch

    if not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise RuntimeError("Deployment checkpoint must be a nonempty tensor state mapping")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Refusing stale temporary output {temporary}")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(state), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    """Durably commit a newly linked immutable artifact directory entry."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor keys, dtypes, shapes, and exact contiguous CPU bytes."""

    import torch

    if not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise RuntimeError("Training state must be a nonempty tensor mapping")
    digest = hashlib.sha256()

    def update(part: bytes) -> None:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)

    for key in sorted(state):
        value = state[key].detach().cpu()
        if value.layout != torch.strided:
            raise RuntimeError(f"Training state tensor {key} is not strided")
        contiguous = value.contiguous()
        raw = contiguous.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        update(key.encode("utf-8"))
        update(str(contiguous.dtype).encode("ascii"))
        update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
        update(raw)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{label} is not a lowercase SHA-256")
    return value


def _precalibration_payload_bound_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "schema",
            "artifact_role",
            "deployment_eligible",
            "seed",
            "mode",
            "step_count",
            "source_manifest_sha256",
            "manifest_bundle_sha256",
            "source_bundle_sha256",
            "training_config_sha256",
            "recovery_scope",
            "cross_version_recovery_supported",
            "training_state_sha256",
            "training_state_key_count",
            "training_state_keys_sha256",
            "post_ema_state_validation",
            "training_metrics",
        )
    }


def validate_training_only_precalibration_payload(
    payload: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_mode: str,
    expected_step_count: int,
    expected_source_manifest_sha256: str,
    expected_manifest_bundle_sha256: str,
    expected_source_bundle_sha256: str,
    expected_training_config_sha256: str,
) -> Mapping[str, torch.Tensor]:
    """Validate a recovery artifact against authority supplied outside that artifact."""

    import torch

    exact_keys = {
        "schema",
        "artifact_role",
        "deployment_eligible",
        "seed",
        "mode",
        "step_count",
        "source_manifest_sha256",
        "manifest_bundle_sha256",
        "source_bundle_sha256",
        "training_config_sha256",
        "recovery_scope",
        "cross_version_recovery_supported",
        "training_state",
        "training_state_sha256",
        "training_state_key_count",
        "training_state_keys_sha256",
        "post_ema_state_validation",
        "training_metrics",
        "artifact_bundle_sha256",
    }
    if set(payload) != exact_keys:
        raise RuntimeError("Training-only pre-calibration artifact fields differ")
    fixed = {
        "schema": payload.get("schema") == PRECALIBRATION_SCHEMA,
        "artifact_role": payload.get("artifact_role") == PRECALIBRATION_ARTIFACT_ROLE,
        "deployment_ineligible": payload.get("deployment_eligible") is False,
        "seed": type(payload.get("seed")) is int
        and payload.get("seed") == expected_seed,
        "mode": payload.get("mode") == expected_mode,
        "step_count": type(payload.get("step_count")) is int
        and payload.get("step_count") == expected_step_count,
        "source_manifest": payload.get("source_manifest_sha256")
        == expected_source_manifest_sha256,
        "manifest_bundle": payload.get("manifest_bundle_sha256")
        == expected_manifest_bundle_sha256,
        "source_bundle": payload.get("source_bundle_sha256")
        == expected_source_bundle_sha256,
        "training_config": payload.get("training_config_sha256")
        == expected_training_config_sha256,
        "recovery_scope": payload.get("recovery_scope")
        == CALIBRATION_RECOVERY_SCOPE,
        "cross_version_recovery_disabled": payload.get(
            "cross_version_recovery_supported"
        )
        is False,
    }
    failed = sorted(key for key, passed in fixed.items() if not passed)
    if failed:
        raise RuntimeError(f"Training-only pre-calibration authority differs: {failed}")
    for label in (
        "source_manifest_sha256",
        "manifest_bundle_sha256",
        "source_bundle_sha256",
        "training_config_sha256",
        "training_state_sha256",
        "training_state_keys_sha256",
        "artifact_bundle_sha256",
    ):
        _require_sha256(payload.get(label), label=label)
    state = payload.get("training_state")
    if not isinstance(state, dict) or not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise RuntimeError("Training-only pre-calibration state mapping is invalid")
    nonfinite = [
        key
        for key, value in state.items()
        if (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all())
    ]
    if nonfinite:
        raise RuntimeError(
            f"Training-only pre-calibration state contains non-finite tensors: {nonfinite}"
        )
    if (
        type(payload.get("training_state_key_count")) is not int
        or payload.get("training_state_key_count") != len(state)
    ):
        raise RuntimeError("Training-only pre-calibration state key count differs")
    if payload.get("training_state_keys_sha256") != canonical_sha256(sorted(state)):
        raise RuntimeError("Training-only pre-calibration state key digest differs")
    if payload.get("training_state_sha256") != tensor_state_sha256(state):
        raise RuntimeError("Training-only pre-calibration tensor state digest differs")
    validation = payload.get("post_ema_state_validation")
    if (
        not isinstance(validation, dict)
        or validation.get("all_parameter_and_buffer_tensors_finite") is not True
        or validation.get("all_active_initializers_finite_positive_initialized") is not True
    ):
        raise RuntimeError("Training-only pre-calibration post-EMA validation did not close")
    initializer_values = validation.get("initializer_running_ms")
    if (
        validation.get("initializer") != CALIBRATION_INITIALIZER_LABEL
        or not isinstance(initializer_values, dict)
        or len(initializer_values) != EXPECTED_PADE_SITE_COUNT
        or len(set(initializer_values)) != len(initializer_values)
        or validation.get("initializer_running_ms_sha256")
        != canonical_sha256(initializer_values)
    ):
        raise RuntimeError("Training-only pre-calibration initializer inventory differs")
    for site, expected_value in initializer_values.items():
        running_key = f"{site}.running_ms"
        initialized_key = f"{site}.initialized"
        if (
            not isinstance(site, str)
            or isinstance(expected_value, bool)
            or not isinstance(expected_value, (int, float))
            or not math.isfinite(float(expected_value))
            or float(expected_value) < PRODUCT_RUNNING_MS_IDENTITY_FLOOR
            or running_key not in state
            or initialized_key not in state
            or state[running_key].numel() != 1
            or float(state[running_key].item()) != float(expected_value)
            or state[initialized_key].numel() != 1
            or not bool(state[initialized_key].item())
        ):
            raise RuntimeError(
                f"Training-only pre-calibration initializer state differs at {site!r}"
            )
    metrics = payload.get("training_metrics")
    if not isinstance(metrics, dict) or set(metrics) != {
        "completed_steps",
        "last_loss",
        "training_elapsed_s",
        "peak_allocated_gb",
        "peak_reserved_gb",
    }:
        raise RuntimeError("Training-only pre-calibration metrics differ")
    if (
        type(metrics.get("completed_steps")) is not int
        or metrics.get("completed_steps") != expected_step_count
        or any(
            isinstance(metrics.get(key), bool)
            or not isinstance(metrics.get(key), (int, float))
            or not math.isfinite(float(metrics[key]))
            for key in (
                "last_loss",
                "training_elapsed_s",
                "peak_allocated_gb",
                "peak_reserved_gb",
            )
        )
        or float(metrics["training_elapsed_s"]) <= 0.0
        or float(metrics["peak_allocated_gb"]) < 0.0
        or float(metrics["peak_reserved_gb"]) < 0.0
    ):
        raise RuntimeError("Training-only pre-calibration metrics are invalid")
    if payload.get("artifact_bundle_sha256") != canonical_sha256(
        _precalibration_payload_bound_fields(payload)
    ):
        raise RuntimeError("Training-only pre-calibration artifact bundle digest differs")
    return state


def publish_training_only_precalibration_state_exclusive(
    path: Path,
    state: Mapping[str, torch.Tensor],
    *,
    seed: int,
    mode: str,
    step_count: int,
    source_manifest_sha256: str,
    manifest_bundle_sha256: str,
    source_bundle_sha256: str,
    training_config_sha256: str,
    post_ema_state_validation: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish one source-bound, deployment-ineligible recovery state."""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise RuntimeError("Training-only pre-calibration output directory is a link")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Refusing stale temporary output {temporary}")
    cpu_state = {key: value.detach().cpu().clone() for key, value in state.items()}
    payload: dict[str, Any] = {
        "schema": PRECALIBRATION_SCHEMA,
        "artifact_role": PRECALIBRATION_ARTIFACT_ROLE,
        "deployment_eligible": False,
        "seed": seed,
        "mode": mode,
        "step_count": step_count,
        "source_manifest_sha256": source_manifest_sha256,
        "manifest_bundle_sha256": manifest_bundle_sha256,
        "source_bundle_sha256": source_bundle_sha256,
        "training_config_sha256": training_config_sha256,
        "recovery_scope": CALIBRATION_RECOVERY_SCOPE,
        "cross_version_recovery_supported": False,
        "training_state": cpu_state,
        "training_state_sha256": tensor_state_sha256(cpu_state),
        "training_state_key_count": len(cpu_state),
        "training_state_keys_sha256": canonical_sha256(sorted(cpu_state)),
        "post_ema_state_validation": dict(post_ema_state_validation),
        "training_metrics": dict(training_metrics),
    }
    payload["artifact_bundle_sha256"] = canonical_sha256(
        _precalibration_payload_bound_fields(payload)
    )
    validate_training_only_precalibration_payload(
        payload,
        expected_seed=seed,
        expected_mode=mode,
        expected_step_count=step_count,
        expected_source_manifest_sha256=source_manifest_sha256,
        expected_manifest_bundle_sha256=manifest_bundle_sha256,
        expected_source_bundle_sha256=source_bundle_sha256,
        expected_training_config_sha256=training_config_sha256,
    )
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        artifact_sha256 = file_sha256(temporary)
        loaded = load_training_only_precalibration_state(
            temporary,
            expected_file_sha256=artifact_sha256,
            expected_seed=seed,
            expected_mode=mode,
            expected_step_count=step_count,
            expected_source_manifest_sha256=source_manifest_sha256,
            expected_manifest_bundle_sha256=manifest_bundle_sha256,
            expected_source_bundle_sha256=source_bundle_sha256,
            expected_training_config_sha256=training_config_sha256,
        )
        if tensor_state_sha256(loaded["training_state"]) != payload["training_state_sha256"]:
            raise RuntimeError("Serialized training-only pre-calibration state changed")
        temporary.chmod(0o444)
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": path.as_posix(),
        "sha256": artifact_sha256,
        "schema": PRECALIBRATION_SCHEMA,
        "artifact_role": PRECALIBRATION_ARTIFACT_ROLE,
        "deployment_eligible": False,
        "recovery_scope": CALIBRATION_RECOVERY_SCOPE,
        "cross_version_recovery_supported": False,
        "artifact_bundle_sha256": payload["artifact_bundle_sha256"],
        "training_state_sha256": payload["training_state_sha256"],
        "training_state_key_count": payload["training_state_key_count"],
        "training_state_keys_sha256": payload["training_state_keys_sha256"],
        "training_metrics": payload["training_metrics"],
    }


def load_training_only_precalibration_state(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_seed: int,
    expected_mode: str,
    expected_step_count: int,
    expected_source_manifest_sha256: str,
    expected_manifest_bundle_sha256: str,
    expected_source_bundle_sha256: str,
    expected_training_config_sha256: str,
) -> dict[str, Any]:
    """Load a recovery state only when an external SHA and all authorities agree."""

    import torch

    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Training-only pre-calibration artifact is missing or nonphysical: {path}")
    _require_sha256(expected_file_sha256, label="expected_file_sha256")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("Training-only pre-calibration artifact is not regular")
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != expected_file_sha256:
            raise RuntimeError("Training-only pre-calibration artifact file digest differs")
        handle.seek(0)
        payload = torch.load(handle, map_location="cpu", weights_only=True)
        after = os.fstat(handle.fileno())
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise RuntimeError(
            "Training-only pre-calibration artifact changed during authenticated load"
        )
    if not isinstance(payload, dict):
        raise RuntimeError("Training-only pre-calibration artifact is not a mapping")
    validate_training_only_precalibration_payload(
        payload,
        expected_seed=expected_seed,
        expected_mode=expected_mode,
        expected_step_count=expected_step_count,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_manifest_bundle_sha256=expected_manifest_bundle_sha256,
        expected_source_bundle_sha256=expected_source_bundle_sha256,
        expected_training_config_sha256=expected_training_config_sha256,
    )
    return payload


def build_vocab(tasks: Mapping[int, str]) -> tuple[dict[str, int], Any]:
    if sorted(tasks.values()) != sorted(OFFICIAL_TASK_LANGUAGES.values()):
        raise RuntimeError("Vocabulary task languages differ from the frozen Object suite")
    words: set[str] = set()
    for language in tasks.values():
        words.update(language.lower().replace(".", "").split())
    vocab = {"<pad>": 0, "<bos>": 1}
    for word in sorted(words):
        vocab[word] = len(vocab)
    if canonical_sha256(vocab) != VOCAB_SHA256:
        raise RuntimeError("Frozen Object vocabulary SHA-256 differs")
    return vocab, build_encoder(vocab)


def build_encoder(vocab: Mapping[str, int]):
    def encode(text: str, length: int = MAX_INSTRUCTION_LENGTH) -> list[int]:
        ids = [1] + [
            int(vocab.get(word, 0))
            for word in text.lower().replace(".", "").split()
        ]
        return (ids[:length] + [0] * max(0, length - len(ids)))[:length]

    return encode


def load_suite():
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[SUITE]()


def task_languages(suite: Any) -> dict[int, str]:
    tasks = {index: str(suite.get_task(index).language) for index in range(suite.n_tasks)}
    if tasks != OFFICIAL_TASK_LANGUAGES:
        raise RuntimeError(f"Official Object tasks differ from the frozen map: {tasks}")
    if canonical_sha256(tasks) != OFFICIAL_TASK_LANGUAGES_SHA256:
        raise RuntimeError("Official Object task-language SHA-256 differs")
    return tasks


def validate_task_mapping(
    official_tasks: Mapping[int, str],
) -> tuple[dict[int, str], dict[str, Any]]:
    if dict(official_tasks) != OFFICIAL_TASK_LANGUAGES:
        raise RuntimeError("Task remap input differs from the frozen official task map")
    if canonical_sha256(dict(official_tasks)) != OFFICIAL_TASK_LANGUAGES_SHA256:
        raise RuntimeError("Task remap input SHA-256 differs")
    provenance_path = repository_root() / PROVENANCE_PATH
    if not provenance_path.is_file() or provenance_path.is_symlink():
        raise RuntimeError("Pinned cache provenance is missing or nonphysical")
    if file_sha256(provenance_path) != PROVENANCE_SHA256:
        raise RuntimeError("Pinned cache provenance SHA-256 differs")
    payload = read_json_object_physical(
        provenance_path, label="Pinned cache provenance"
    )
    cache = payload.get("cache", {})
    metadata = payload.get("metadata", {})
    comparison = payload.get("comparison", {})
    source = payload.get("source", {})
    required = {
        "schema": payload.get("schema") == "xvla-cache-provenance-v1",
        "suite": payload.get("suite") == SUITE,
        "verified": payload.get("verified") is True,
        "cache_path": cache.get("path") == CACHE_PATH.as_posix(),
        "cache_sha256": cache.get("sha256") == CACHE_SHA256,
        "cache_frames": cache.get("frames") == CACHE_FRAME_COUNT,
        "cache_resolution": cache.get("resolution") == RESOLUTION,
        "canonical_cache_content": cache.get("canonical_content_sha256")
        == CANONICAL_CACHE_CONTENT_SHA256,
        "metadata_repository": metadata.get("repository") == DATASET_REPOSITORY,
        "metadata_revision": metadata.get("revision") == DATASET_REVISION,
        "metadata_file": metadata.get("file") == "meta/tasks.parquet",
        "metadata_sha256": metadata.get("sha256") == DATASET_METADATA_SHA256,
        "task_mapping": metadata.get("dataset_to_official_task")
        == DATASET_TO_OFFICIAL_TASK,
        "language_set": metadata.get("language_set_matches_official") is True,
        "compared_frames": comparison.get("cached_frames_compared") == CACHE_FRAME_COUNT,
        "zero_mismatches": bool(comparison.get("mismatch_counts"))
        and all(value == 0 for value in comparison.get("mismatch_counts", {}).values()),
        "no_mismatch_examples": comparison.get("mismatch_examples") == [],
        "source_content_identity": source.get("content_hashes_match") is True,
        "source_canonical_content": source.get("canonical_content_sha256")
        == CANONICAL_CACHE_CONTENT_SHA256,
    }
    failed = sorted(key for key, value in required.items() if not value)
    if failed:
        raise RuntimeError(f"Pinned cache provenance failed gates: {failed}")
    dataset_tasks = {
        dataset_index: str(official_tasks[DATASET_TO_OFFICIAL_TASK[str(dataset_index)]])
        for dataset_index in range(10)
    }
    expected_record = expected_task_mapping_record()
    observed_record = {
        "path": PROVENANCE_PATH.as_posix(),
        "sha256": PROVENANCE_SHA256,
        "dataset_to_official_task": DATASET_TO_OFFICIAL_TASK,
        "dataset_tasks_derived_from_official_languages": True,
        "official_task_languages_sha256": OFFICIAL_TASK_LANGUAGES_SHA256,
        "vocab_sha256": VOCAB_SHA256,
        "validation_gates": required,
    }
    if observed_record != expected_record:
        raise RuntimeError("Pinned task-mapping record differs from its frozen authority")
    return dataset_tasks, observed_record


def expected_task_mapping_record() -> dict[str, Any]:
    """Return the immutable task-remap record without reading ambient metadata.

    The producer still authenticates and reads the provenance artifact in
    :func:`validate_task_mapping`.  This pure representation lets independent
    downstream consumers reconstruct the exact expected evaluation field from
    already-authenticated constants instead of trusting a producer copy.
    """

    return {
        "path": PROVENANCE_PATH.as_posix(),
        "sha256": PROVENANCE_SHA256,
        "dataset_to_official_task": DATASET_TO_OFFICIAL_TASK,
        "dataset_tasks_derived_from_official_languages": True,
        "official_task_languages_sha256": OFFICIAL_TASK_LANGUAGES_SHA256,
        "vocab_sha256": VOCAB_SHA256,
        "validation_gates": {
            "schema": True,
            "suite": True,
            "verified": True,
            "cache_path": True,
            "cache_sha256": True,
            "cache_frames": True,
            "cache_resolution": True,
            "canonical_cache_content": True,
            "metadata_repository": True,
            "metadata_revision": True,
            "metadata_file": True,
            "metadata_sha256": True,
            "task_mapping": True,
            "language_set": True,
            "compared_frames": True,
            "zero_mismatches": True,
            "no_mismatch_examples": True,
            "source_content_identity": True,
            "source_canonical_content": True,
        },
    }


def make_product_config(vocab_size: int, *, deployment: bool) -> VLAConfig:
    from xvla.models.vla import VLAConfig

    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size < 3:
        raise ValueError(f"Invalid vocabulary size {vocab_size!r}")
    return VLAConfig(
        image_size=RESOLUTION,
        patch_size=PATCH_SIZE,
        vit_dim=VIT_DIM,
        vit_layers=VIT_LAYERS,
        vit_heads=VIT_HEADS,
        vit_ffn_rank=VIT_FFN_RANK,
        vocab_size=vocab_size,
        max_instr_len=MAX_INSTRUCTION_LENGTH,
        state_dim=STATE_DIM,
        n_embodiments=1,
        dim=MODEL_DIM,
        n_layers=MODEL_LAYERS,
        n_heads=MODEL_HEADS,
        ffn_rank=MODEL_FFN_RANK,
        attn="bilinear",
        ffn="bilinear",
        norm="rational",
        qk_norm="rational",
        residual=True,
        action_horizon=ACTION_HORIZON,
        action_dim=ACTION_DIM,
        action_head="product",
        head_rank=HEAD_RANK,
        n_factors=N_FACTORS,
        vision_encoder="vit",
        distill_teacher=not deployment,
        teacher_weight=1.0,
        lambda_distill=0.0 if deployment else 1.0,
        curriculum=not deployment,
        distill_tau0=1.0,
        distill_tau1=0.02,
        product_st=False,
        value_head=False,
    )


def config_record(config: VLAConfig) -> dict[str, Any]:
    payload = asdict(config)
    return {"values": payload, "sha256": canonical_sha256(payload)}


def install_product_running_ms_fail_closed_guards(model: ChiVLA) -> dict[str, Any]:
    """Reject invalid anchors before RationalNorm's historical floor can act."""

    import torch

    from xvla.nn.normalization import RationalNorm

    guarded: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, RationalNorm):
            continue
        if not getattr(module, "_product_running_ms_fail_closed_guard", False):
            def require_identity_floor(_module, _inputs, *, site_name=name):
                if _module.training and not _module.frozen:
                    return
                running_ms = _module.running_ms
                unchanged = (
                    id(running_ms)
                    == _module._product_running_ms_validated_tensor_identity
                    and running_ms._version
                    == _module._product_running_ms_validated_tensor_version
                )
                if not unchanged:
                    finite = (
                        bool(torch.isfinite(running_ms).all())
                        if running_ms.numel() == 1
                        else False
                    )
                    value = (
                        float(running_ms.item())
                        if running_ms.numel() == 1 and finite
                        else None
                    )
                    raise RuntimeError(
                        "Product RationalNorm running_ms failed before forward: "
                        f"site={site_name!r} unchanged_since_validation={unchanged} "
                        f"finite={finite} value={value!r} "
                        f"required_min={PRODUCT_RUNNING_MS_IDENTITY_FLOOR}"
                    )

            module.register_forward_pre_hook(require_identity_floor)
            module._product_running_ms_fail_closed_guard = True
        valid = (
            module.running_ms.numel() == 1
            and bool(torch.isfinite(module.running_ms).all())
            and float(module.running_ms.item()) >= PRODUCT_RUNNING_MS_IDENTITY_FLOOR
        )
        if not valid:
            finite = (
                bool(torch.isfinite(module.running_ms).all())
                if module.running_ms.numel() == 1
                else False
            )
            value = (
                float(module.running_ms.item())
                if module.running_ms.numel() == 1 and finite
                else None
            )
            raise RuntimeError(
                "Product RationalNorm has an invalid running mean-square during "
                f"validation: site={name!r} finite={finite} value={value!r} "
                f"required_min={PRODUCT_RUNNING_MS_IDENTITY_FLOOR}"
            )
        module._product_running_ms_validated_tensor_identity = id(module.running_ms)
        module._product_running_ms_validated_tensor_version = module.running_ms._version
        guarded.append(name)
    if len(guarded) != EXPECTED_TOTAL_PADE_MODULE_COUNT or len(set(guarded)) != len(
        guarded
    ):
        raise RuntimeError(f"Product RationalNorm guard inventory differs: {guarded}")
    return {
        "guarded_module_count": len(guarded),
        "running_ms_required_min": PRODUCT_RUNNING_MS_IDENTITY_FLOOR,
        "historical_floor_is_identity_for_every_accepted_forward": True,
        "invalid_anchor_fails_before_normalization": True,
        "accepted_eval_forward_uses_host_only_identity_and_version_check": True,
        "training_updates_use_positive_mean_square_convex_update_invariant": True,
    }


def validate_product_model(
    model: ChiVLA,
    *,
    deployment: bool,
    require_initialized_norms: bool,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    from xvla.models.vit import ChiViT
    from xvla.nn.attention import BilinearAttention
    from xvla.nn.bilinear import BilinearFFN
    from xvla.nn.block import ChiTransformerBlock
    from xvla.nn.normalization import RationalNorm
    from xvla.nn.product_routing import ProductRoutingHead

    cfg = model.cfg
    running_ms_guard = install_product_running_ms_fail_closed_guards(model)
    expected = make_product_config(cfg.vocab_size, deployment=deployment)
    if asdict(cfg) != asdict(expected):
        raise RuntimeError("Product VLA config differs from the frozen recipe")
    if not hasattr(model, "product_head") or hasattr(model, "action_head"):
        raise RuntimeError("Model does not expose only the deployed ProductRoutingHead")
    head = model.product_head
    if (
        not isinstance(head, ProductRoutingHead)
        or head.dim != MODEL_DIM
        or head.action_dim != ACTION_DIM
        or head.horizon != ACTION_HORIZON
        or head.m != ACTION_HORIZON * ACTION_DIM
        or head.G != N_FACTORS
        or head.center.rank != HEAD_RANK
        or len(head.factors) != N_FACTORS
        or len(head.gates) != N_FACTORS
        or any(module.rank != HEAD_RANK for module in (*head.factors, *head.gates))
    ):
        raise RuntimeError("ProductRoutingHead geometry differs")
    if deployment:
        if hasattr(model, "teacher_head") or hasattr(model, "value_head"):
            raise RuntimeError("Deployment model retains a training-only head")
    elif not hasattr(model, "teacher_head") or hasattr(model, "value_head"):
        raise RuntimeError("Training model has the wrong auxiliary-head topology")
    if (
        not isinstance(model.vision, ChiViT)
        or hasattr(model, "vision2")
        or hasattr(model, "projector")
        or not isinstance(model.vis_proj, nn.Linear)
        or model.vis_proj.in_features != VIT_DIM
        or model.vis_proj.out_features != MODEL_DIM
        or model.vision.patch.in_channels != 3
        or model.vision.patch.out_channels != VIT_DIM
        or tuple(model.vision.patch.kernel_size) != (PATCH_SIZE, PATCH_SIZE)
        or tuple(model.vision.patch.stride) != (PATCH_SIZE, PATCH_SIZE)
        or len(model.vision.blocks.blocks) != VIT_LAYERS
        or len(model.backbone.blocks) != MODEL_LAYERS
        or model.vision.blocks.dim != VIT_DIM
        or model.vision.blocks.causal
        or not model.vision.blocks.residual
        or model.backbone.dim != MODEL_DIM
        or not model.backbone.causal
        or not model.backbone.residual
    ):
        raise RuntimeError("Vision or joint topology differs from the frozen capable architecture")
    tower_specs = (
        ("vision", model.vision.blocks.blocks, VIT_DIM, VIT_HEADS, VIT_FFN_RANK),
        ("joint", model.backbone.blocks, MODEL_DIM, MODEL_HEADS, MODEL_FFN_RANK),
    )
    for tower, blocks, dim, heads, rank in tower_specs:
        for block_index, block in enumerate(blocks):
            if (
                not isinstance(block, ChiTransformerBlock)
                or not block.residual
                or not isinstance(block.attn, BilinearAttention)
                or block.attn.dim != dim
                or block.attn.n_heads != heads
                or block.attn.qk_norm != "rational"
                or not isinstance(block.ffn, BilinearFFN)
                or block.ffn.dim != dim
                or block.ffn.out_dim != dim
                or block.ffn.rank != rank
            ):
                raise RuntimeError(
                    f"{tower} block {block_index} is remapped, linearized, or non-residual"
                )
    forbidden_architecture_attributes = [
        name
        for name, module in model.named_modules()
        if hasattr(module, "const_fill")
    ]
    if forbidden_architecture_attributes:
        raise RuntimeError(
            "Frozen Product VLA contains an unauthorized constant-fill route: "
            f"{forbidden_architecture_attributes}"
        )
    named_norms = active_rational_norms(model)
    norms = [module for _, module in named_norms]
    for block in (*model.vision.blocks.blocks, *model.backbone.blocks):
        block_norms = [
            module for module in block.modules() if isinstance(module, RationalNorm)
        ]
        if len(block_norms) != 6:
            raise RuntimeError("A deployed transformer block does not expose six Padé sites")
    if not isinstance(model.norm_out, RationalNorm):
        raise RuntimeError("The deployed final policy norm is not RationalNorm")
    if len(norms) != EXPECTED_PADE_SITE_COUNT:
        raise RuntimeError(
            f"Padé-compatible rational site count {len(norms)} != {EXPECTED_PADE_SITE_COUNT}"
        )
    reference = RationalNorm(variant="pade", deg=2)
    reference_pa = reference.pa.to(norms[0].pa.device)
    reference_pb = reference.pb.to(norms[0].pb.device)
    for index, norm in enumerate(norms):
        if norm.variant != "pade" or norm.deg != 2:
            raise RuntimeError(f"Normalization site {index} is not degree-two Padé")
        if norm.pa.shape != (3,) or norm.pb.shape != (3,):
            raise RuntimeError(f"Normalization site {index} has wrong Padé buffers")
        if not bool(torch.isfinite(norm.pa).all() and torch.isfinite(norm.pb).all()):
            raise RuntimeError(f"Normalization site {index} has non-finite Padé buffers")
        if not torch.equal(norm.pa, reference_pa.to(norm.pa.dtype)) or not torch.equal(
            norm.pb, reference_pb.to(norm.pb.dtype)
        ):
            raise RuntimeError(f"Normalization site {index} changed the frozen Padé coefficients")
        if not bool((norm.pb > 0).all()):
            raise RuntimeError(f"Normalization site {index} has a nonpositive Padé denominator")
        if (
            not bool(torch.isfinite(norm.running_ms))
            or float(norm.running_ms) < PRODUCT_RUNNING_MS_IDENTITY_FLOOR
        ):
            raise RuntimeError(f"Normalization site {index} has an invalid running mean-square")
        if require_initialized_norms and not bool(norm.initialized):
            raise RuntimeError(f"Normalization site {index} was never initialized")
        if require_initialized_norms and not norm.frozen:
            raise RuntimeError(f"Normalization site {index} is not frozen for deployment")
    return {
        "pade_compatible_rational_norm": True,
        "pade_site_count": len(norms),
        "pade_degree": 2,
        "pade_numerator": [float(value) for value in reference.pa.tolist()],
        "pade_denominator": [float(value) for value in reference.pb.tolist()],
        "pade_denominator_domain": "v >= 0",
        "pade_denominator_strict_lower_bound": float(reference.pb[0]),
        "pade_denominator_strictly_positive_coefficients": True,
        "product_factors": head.G,
        "product_modes": 2 ** head.G,
        "product_component_count": 1 + 2 * head.G,
        "product_component_root_euclidean_width": head.m * (head.G + 1) + head.G,
        "product_component_root_projective_width": head.m * (head.G + 1) + head.G + 1,
        "deterministic_sign_rule": "+1 iff gate_logit > 0, else -1",
        "training_only_teacher_present": hasattr(model, "teacher_head"),
        "vision_encoder_class": type(model.vision).__name__,
        "vision_block_count": len(model.vision.blocks.blocks),
        "joint_block_count": len(model.backbone.blocks),
        "all_vision_and_joint_blocks_residual": True,
        "all_attention_blocks_bilinear": True,
        "all_ffn_blocks_bilinear": True,
        "vision_projection": [VIT_DIM, MODEL_DIM],
        "constant_fill_routes": [],
        "linear_action_fallback_present": hasattr(model, "action_head"),
        "running_ms_fail_closed_guard": running_ms_guard,
    }


def active_rational_norms(model: ChiVLA) -> tuple[tuple[str, Any], ...]:
    """Return the 73 live normalization sites in deterministic forward order."""

    from xvla.nn.normalization import RationalNorm

    names_by_id = {id(module): name for name, module in model.named_modules()}
    ordered: list[tuple[str, Any]] = []
    for block in (*model.vision.blocks.blocks, *model.backbone.blocks):
        for module in block.modules():
            if isinstance(module, RationalNorm):
                ordered.append((names_by_id[id(module)], module))
    if not isinstance(model.norm_out, RationalNorm):
        raise RuntimeError("The final live policy norm is not Padé-compatible rational")
    ordered.append((names_by_id[id(model.norm_out)], model.norm_out))
    if len(ordered) != EXPECTED_PADE_SITE_COUNT:
        raise RuntimeError(
            f"Live Padé-compatible rational site count {len(ordered)} "
            f"!= {EXPECTED_PADE_SITE_COUNT}"
        )
    if len({name for name, _ in ordered}) != len(ordered):
        raise RuntimeError("Live Padé-compatible rational site names are not unique")
    return tuple(ordered)


def active_running_ms_values(model: ChiVLA) -> dict[str, float]:
    """Return exact scalar anchors for every live forward-reachable Padé site."""

    import torch

    values: dict[str, float] = {}
    for name, norm in active_rational_norms(model):
        if (
            norm.running_ms.numel() != 1
            or not bool(torch.isfinite(norm.running_ms).all())
            or not bool(norm.initialized.item())
        ):
            raise RuntimeError(f"Live running_ms value is invalid at site {name!r}")
        value = float(norm.running_ms.item())
        if value < PRODUCT_RUNNING_MS_IDENTITY_FLOOR:
            raise RuntimeError(f"Live running_ms value is below the identity floor at {name!r}")
        values[name] = value
    return values


def calibration_state_transition_proof(
    precalibration_state: Mapping[str, torch.Tensor],
    calibrated_training_state: Mapping[str, torch.Tensor],
    deployment_state: Mapping[str, torch.Tensor],
    *,
    active_site_names: tuple[str, ...],
) -> dict[str, Any]:
    """Prove calibration changed only live running_ms buffers, bit for bit."""

    import torch

    mappings = {
        "precalibration": precalibration_state,
        "calibrated_training": calibrated_training_state,
        "deployment": deployment_state,
    }
    for label, state in mappings.items():
        if not state or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state.items()
        ):
            raise RuntimeError(f"Calibration {label} state mapping is invalid")
    if (
        len(active_site_names) != EXPECTED_PADE_SITE_COUNT
        or len(set(active_site_names)) != len(active_site_names)
    ):
        raise RuntimeError("Calibration state-transition site inventory differs")
    mutable_keys = tuple(f"{name}.running_ms" for name in active_site_names)
    initialized_keys = tuple(f"{name}.initialized" for name in active_site_names)
    training_keys = tuple(sorted(precalibration_state))
    if tuple(sorted(calibrated_training_state)) != training_keys:
        raise RuntimeError("Calibration changed the training state-key mapping")
    if any(
        key not in precalibration_state or key not in calibrated_training_state
        for key in (*mutable_keys, *initialized_keys)
    ):
        raise RuntimeError("Calibration state is missing an active Padé buffer")

    removed_keys = tuple(sorted(set(training_keys) - set(deployment_state)))
    if removed_keys != ("teacher_head.bias", "teacher_head.weight"):
        raise RuntimeError(
            f"Deployment removed an unauthorized training state entry: {removed_keys}"
        )
    expected_deployment_keys = tuple(
        key for key in training_keys if key not in removed_keys
    )
    if tuple(sorted(deployment_state)) != expected_deployment_keys:
        raise RuntimeError("Deployment state-key mapping differs from training state")

    changed_training_keys = tuple(
        key
        for key in training_keys
        if not torch.equal(precalibration_state[key], calibrated_training_state[key])
    )
    unauthorized_changes = tuple(
        key for key in changed_training_keys if key not in mutable_keys
    )
    if unauthorized_changes:
        raise RuntimeError(
            "Calibration mutated non-running_ms state entries: "
            f"{unauthorized_changes}"
        )
    if any(
        not torch.equal(calibrated_training_state[key], deployment_state[key])
        for key in expected_deployment_keys
    ):
        raise RuntimeError("Deployment export changed a calibrated state tensor")
    if any(
        precalibration_state[key].numel() != 1
        or calibrated_training_state[key].numel() != 1
        or not bool(precalibration_state[key].item())
        or not bool(calibrated_training_state[key].item())
        or not torch.equal(precalibration_state[key], calibrated_training_state[key])
        for key in initialized_keys
    ):
        raise RuntimeError("Calibration changed or cleared an active initialized buffer")

    dormant_keys = (
        "vision.norm_out.running_ms",
        "vision.norm_out.initialized",
    )
    if any(
        key not in precalibration_state
        or key not in calibrated_training_state
        or key not in deployment_state
        or not torch.equal(precalibration_state[key], calibrated_training_state[key])
        or not torch.equal(precalibration_state[key], deployment_state[key])
        for key in dormant_keys
    ):
        raise RuntimeError("Calibration changed the dormant vision.norm_out state")

    changed_deployment_keys = tuple(
        key
        for key in expected_deployment_keys
        if not torch.equal(precalibration_state[key], deployment_state[key])
    )
    if changed_deployment_keys != changed_training_keys:
        raise RuntimeError(
            "Pre-calibration-to-deployment changed-key inventory differs from training"
        )
    consumer_view = {
        "precalibration_projected_state_sha256": tensor_state_sha256(
            {
                key: precalibration_state[key]
                for key in expected_deployment_keys
            }
        ),
        "deployment_state_sha256": tensor_state_sha256(deployment_state),
        "deployment_state_key_count": len(deployment_state),
        "deployment_state_keys_sha256": canonical_sha256(
            list(expected_deployment_keys)
        ),
        "removed_training_only_state_keys": list(removed_keys),
        "authorized_mutable_state_keys": list(mutable_keys),
        "authorized_mutable_state_keys_sha256": canonical_sha256(list(mutable_keys)),
        "changed_state_keys": list(changed_deployment_keys),
        "changed_state_keys_sha256": canonical_sha256(list(changed_deployment_keys)),
        "all_shared_noncalibration_state_bitwise_unchanged": True,
        "all_active_initialized_buffers_true_and_unchanged": True,
        "dormant_vision_norm_out_state_bitwise_unchanged": True,
    }
    return {
        "schema": f"{SCHEMA}_calibration_state_transition_v1",
        "precalibration_training_state_sha256": tensor_state_sha256(
            precalibration_state
        ),
        "calibrated_training_state_sha256": tensor_state_sha256(
            calibrated_training_state
        ),
        "training_state_key_count": len(training_keys),
        "training_state_keys_sha256": canonical_sha256(list(training_keys)),
        "changed_training_state_keys": list(changed_training_keys),
        "changed_training_state_keys_sha256": canonical_sha256(
            list(changed_training_keys)
        ),
        "only_active_running_ms_changed_during_calibration": True,
        "all_learned_parameters_bitwise_unchanged_during_calibration": True,
        "all_noncalibration_buffers_bitwise_unchanged_during_calibration": True,
        "deployment_removed_only_training_teacher": True,
        "deployment_equals_calibrated_training_state_on_every_retained_key": True,
        "precalibration_to_deployment": consumer_view,
    }


def validate_precalibration_to_deployment_transition(
    precalibration_state: Mapping[str, torch.Tensor],
    deployment_state: Mapping[str, torch.Tensor],
    *,
    active_site_names: tuple[str, ...],
    claimed_proof: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently recompute the deploy-visible part of a transition proof."""

    import torch

    if (
        len(active_site_names) != EXPECTED_PADE_SITE_COUNT
        or len(set(active_site_names)) != len(active_site_names)
    ):
        raise RuntimeError("Deployment transition site inventory differs")
    if not precalibration_state or not deployment_state:
        raise RuntimeError("Deployment transition state mapping is empty")
    removed_keys = tuple(sorted(set(precalibration_state) - set(deployment_state)))
    if removed_keys != ("teacher_head.bias", "teacher_head.weight"):
        raise RuntimeError("Deployment transition removed-state inventory differs")
    expected_deployment_keys = tuple(
        key for key in sorted(precalibration_state) if key not in removed_keys
    )
    if tuple(sorted(deployment_state)) != expected_deployment_keys:
        raise RuntimeError("Deployment transition state-key mapping differs")
    mutable_keys = tuple(f"{name}.running_ms" for name in active_site_names)
    initialized_keys = tuple(f"{name}.initialized" for name in active_site_names)
    changed_keys = tuple(
        key
        for key in expected_deployment_keys
        if not torch.equal(precalibration_state[key], deployment_state[key])
    )
    unauthorized = tuple(key for key in changed_keys if key not in mutable_keys)
    if unauthorized:
        raise RuntimeError(
            f"Deployment transition changed unauthorized state: {unauthorized}"
        )
    if any(
        key not in precalibration_state
        or key not in deployment_state
        or precalibration_state[key].numel() != 1
        or deployment_state[key].numel() != 1
        or not bool(precalibration_state[key].item())
        or not bool(deployment_state[key].item())
        or not torch.equal(precalibration_state[key], deployment_state[key])
        for key in initialized_keys
    ):
        raise RuntimeError("Deployment transition changed an initialized buffer")
    dormant_keys = (
        "vision.norm_out.running_ms",
        "vision.norm_out.initialized",
    )
    if any(
        key not in precalibration_state
        or key not in deployment_state
        or not torch.equal(precalibration_state[key], deployment_state[key])
        for key in dormant_keys
    ):
        raise RuntimeError("Deployment transition changed dormant vision.norm_out state")
    recomputed = {
        "precalibration_projected_state_sha256": tensor_state_sha256(
            {
                key: precalibration_state[key]
                for key in expected_deployment_keys
            }
        ),
        "deployment_state_sha256": tensor_state_sha256(deployment_state),
        "deployment_state_key_count": len(deployment_state),
        "deployment_state_keys_sha256": canonical_sha256(
            list(expected_deployment_keys)
        ),
        "removed_training_only_state_keys": list(removed_keys),
        "authorized_mutable_state_keys": list(mutable_keys),
        "authorized_mutable_state_keys_sha256": canonical_sha256(list(mutable_keys)),
        "changed_state_keys": list(changed_keys),
        "changed_state_keys_sha256": canonical_sha256(list(changed_keys)),
        "all_shared_noncalibration_state_bitwise_unchanged": True,
        "all_active_initialized_buffers_true_and_unchanged": True,
        "dormant_vision_norm_out_state_bitwise_unchanged": True,
    }
    expected_claim_keys = {
        "schema",
        "precalibration_training_state_sha256",
        "calibrated_training_state_sha256",
        "training_state_key_count",
        "training_state_keys_sha256",
        "changed_training_state_keys",
        "changed_training_state_keys_sha256",
        "only_active_running_ms_changed_during_calibration",
        "all_learned_parameters_bitwise_unchanged_during_calibration",
        "all_noncalibration_buffers_bitwise_unchanged_during_calibration",
        "deployment_removed_only_training_teacher",
        "deployment_equals_calibrated_training_state_on_every_retained_key",
        "precalibration_to_deployment",
    }
    reconstructed_training_state = {
        key: (
            precalibration_state[key]
            if key in removed_keys
            else deployment_state[key]
        )
        for key in sorted(precalibration_state)
    }
    if (
        set(claimed_proof) != expected_claim_keys
        or claimed_proof.get("schema")
        != f"{SCHEMA}_calibration_state_transition_v1"
        or claimed_proof.get("precalibration_training_state_sha256")
        != tensor_state_sha256(precalibration_state)
        or claimed_proof.get("calibrated_training_state_sha256")
        != tensor_state_sha256(reconstructed_training_state)
        or claimed_proof.get("training_state_key_count")
        != len(precalibration_state)
        or claimed_proof.get("training_state_keys_sha256")
        != canonical_sha256(sorted(precalibration_state))
        or claimed_proof.get("changed_training_state_keys")
        != list(changed_keys)
        or claimed_proof.get("changed_training_state_keys_sha256")
        != canonical_sha256(list(changed_keys))
        or claimed_proof.get("only_active_running_ms_changed_during_calibration")
        is not True
        or claimed_proof.get(
            "all_learned_parameters_bitwise_unchanged_during_calibration"
        )
        is not True
        or claimed_proof.get(
            "all_noncalibration_buffers_bitwise_unchanged_during_calibration"
        )
        is not True
        or claimed_proof.get("deployment_removed_only_training_teacher") is not True
        or claimed_proof.get(
            "deployment_equals_calibrated_training_state_on_every_retained_key"
        )
        is not True
        or claimed_proof.get("precalibration_to_deployment") != recomputed
    ):
        raise RuntimeError("Independent calibration state-transition proof differs")
    return recomputed


def validate_simultaneous_calibration_attestation(
    calibration: Mapping[str, Any],
    *,
    initial_values: Mapping[str, float],
    final_values: Mapping[str, float],
    sample_count: int,
) -> dict[str, Any]:
    """Independently validate exact site coverage, passes, anchors, and final scales."""

    exact_calibration_keys = {
        "method",
        "under_final_ema_parameters",
        "initializer",
        "initializer_running_ms_sha256",
        "initializer_running_ms",
        "initializer_min",
        "initializer_max",
        "running_ms_required_min",
        "historical_floor_is_identity_for_every_accepted_forward",
        "old_training_buffers_discarded",
        "all_sites_frozen_before_first_pass",
        "simultaneous_snapshot_commit",
        "commit_protocol",
        "commit_exception_rollback",
        "intra_pass_exception_rollback_source",
        "process_level_retry_source",
        "recovery_scope",
        "cross_version_recovery_supported",
        "no_value_changing_clamp_or_fallback_on_accepted_path",
        "invalid_anchor_rejected_before_frozen_forward",
        "initializer_and_every_candidate_floor_verified_before_use_or_commit",
        "observation_copy_dtype",
        "observation_copy_before_square",
        "accumulator_dtype",
        "update_passes",
        "final_no_commit_audit_pass",
        "audit_max_relative_change",
        "audit_observed_max_relative_change",
        "batch_size",
        "sample_count_per_pass",
        "live_site_count",
        "site_names",
        "site_names_sha256",
        "site_call_count",
        "product_component_site_names",
        "product_component_site_names_sha256",
        "product_component_call_count",
        "raw_product_components_finite_every_batch",
        "expected_calls_per_site",
        "pass_records",
        "final_running_ms_sha256",
        "final_running_ms",
        "exact_fixed_point_convergence_claimed",
        "all_initialized_and_frozen",
    }
    if not isinstance(calibration, Mapping) or set(calibration) != exact_calibration_keys:
        raise RuntimeError("Calibration attestation top-level fields differ")

    names = tuple(initial_values)
    if (
        len(names) != EXPECTED_PADE_SITE_COUNT
        or tuple(final_values) != names
        or len(set(names)) != len(names)
        or sample_count <= 0
    ):
        raise RuntimeError("Calibration attestation site or sample inventory differs")
    for label, values in (("initial", initial_values), ("final", final_values)):
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < PRODUCT_RUNNING_MS_IDENTITY_FLOOR
            for value in values.values()
        ):
            raise RuntimeError(f"Calibration attestation {label} values are invalid")
    batch_count = (sample_count + FINAL_CALIBRATION_BATCH_SIZE - 1) // (
        FINAL_CALIBRATION_BATCH_SIZE
    )
    expected_calls = (FINAL_CALIBRATION_PASSES + 1) * batch_count
    site_calls = calibration.get("site_call_count")
    product_component_calls = calibration.get("product_component_call_count")
    pass_records = calibration.get("pass_records")
    fixed = {
        "method": calibration.get("method")
        == "trained_anchor_fixed_pass_simultaneous_end_to_end",
        "final_ema": calibration.get("under_final_ema_parameters") is True,
        "initializer": calibration.get("initializer") == CALIBRATION_INITIALIZER_LABEL,
        "initializer_digest": calibration.get("initializer_running_ms_sha256")
        == canonical_sha256(dict(initial_values)),
        "initializer_values": json_type_exact_equal(
            calibration.get("initializer_running_ms"), dict(initial_values)
        ),
        "initializer_min": type(calibration.get("initializer_min")) is float
        and calibration.get("initializer_min")
        == min(float(value) for value in initial_values.values()),
        "initializer_max": type(calibration.get("initializer_max")) is float
        and calibration.get("initializer_max")
        == max(float(value) for value in initial_values.values()),
        "running_ms_floor": type(calibration.get("running_ms_required_min")) is float
        and calibration.get("running_ms_required_min")
        == PRODUCT_RUNNING_MS_IDENTITY_FLOOR,
        "historical_floor_identity": calibration.get(
            "historical_floor_is_identity_for_every_accepted_forward"
        )
        is True,
        "training_buffers_retained": calibration.get("old_training_buffers_discarded")
        is False,
        "frozen_before_first": calibration.get("all_sites_frozen_before_first_pass")
        is True,
        "simultaneous": calibration.get("simultaneous_snapshot_commit") is True,
        "commit_protocol": calibration.get("commit_protocol")
        == "all_candidates_validated_before_sequential_commit_with_exception_rollback",
        "commit_exception_rollback": calibration.get("commit_exception_rollback")
        is True,
        "intra_pass_rollback": calibration.get(
            "intra_pass_exception_rollback_source"
        )
        == "in_memory_precommit_running_ms_snapshot",
        "process_retry_source": calibration.get("process_level_retry_source")
        == "immutable_training_only_precalibration_ema_state",
        "recovery_scope": calibration.get("recovery_scope")
        == CALIBRATION_RECOVERY_SCOPE,
        "cross_version_disabled": calibration.get("cross_version_recovery_supported")
        is False,
        "f64_copy": calibration.get("observation_copy_dtype") == "float64",
        "copy_before_square": calibration.get("observation_copy_before_square") is True,
        "f64_accumulator": calibration.get("accumulator_dtype") == "float64",
        "accepted_path_no_value_changing_floor": calibration.get(
            "no_value_changing_clamp_or_fallback_on_accepted_path"
        )
        is True,
        "invalid_anchor_rejected": calibration.get(
            "invalid_anchor_rejected_before_frozen_forward"
        )
        is True,
        "candidate_floor_validation": calibration.get(
            "initializer_and_every_candidate_floor_verified_before_use_or_commit"
        )
        is True,
        "update_passes": type(calibration.get("update_passes")) is int
        and calibration.get("update_passes")
        == FINAL_CALIBRATION_PASSES,
        "audit_pass": calibration.get("final_no_commit_audit_pass") is True,
        "audit_threshold": type(calibration.get("audit_max_relative_change")) is float
        and calibration.get("audit_max_relative_change")
        == FINAL_CALIBRATION_AUDIT_MAX_RELATIVE_CHANGE,
        "batch_size": type(calibration.get("batch_size")) is int
        and calibration.get("batch_size") == FINAL_CALIBRATION_BATCH_SIZE,
        "sample_count": type(calibration.get("sample_count_per_pass")) is int
        and calibration.get("sample_count_per_pass") == sample_count,
        "live_site_count": type(calibration.get("live_site_count")) is int
        and calibration.get("live_site_count")
        == EXPECTED_PADE_SITE_COUNT,
        "site_names": json_type_exact_equal(calibration.get("site_names"), list(names)),
        "site_names_digest": calibration.get("site_names_sha256")
        == canonical_sha256(list(names)),
        "expected_calls": type(calibration.get("expected_calls_per_site")) is int
        and calibration.get("expected_calls_per_site") == expected_calls,
        "site_calls": isinstance(site_calls, dict)
        and tuple(site_calls) == names
        and all(type(value) is int for value in site_calls.values())
        and set(site_calls.values()) == {expected_calls},
        "product_component_names": calibration.get(
            "product_component_site_names"
        )
        is not None
        and json_type_exact_equal(
            calibration.get("product_component_site_names"),
            list(PRODUCT_COMPONENT_SITE_NAMES),
        ),
        "product_component_names_digest": calibration.get(
            "product_component_site_names_sha256"
        )
        == canonical_sha256(list(PRODUCT_COMPONENT_SITE_NAMES)),
        "product_component_calls": isinstance(product_component_calls, dict)
        and tuple(product_component_calls) == PRODUCT_COMPONENT_SITE_NAMES
        and all(type(value) is int for value in product_component_calls.values())
        and set(product_component_calls.values()) == {expected_calls},
        "raw_product_components_finite": calibration.get(
            "raw_product_components_finite_every_batch"
        )
        is True,
        "pass_count": isinstance(pass_records, list)
        and len(pass_records) == FINAL_CALIBRATION_PASSES + 1,
        "final_digest": calibration.get("final_running_ms_sha256")
        == canonical_sha256(dict(final_values)),
        "final_values": json_type_exact_equal(
            calibration.get("final_running_ms"), dict(final_values)
        ),
        "all_initialized_frozen": calibration.get("all_initialized_and_frozen") is True,
        "not_exact_fixed_point": calibration.get("exact_fixed_point_convergence_claimed")
        is False,
    }
    failed = sorted(key for key, passed in fixed.items() if not passed)
    if failed:
        raise RuntimeError(f"Calibration attestation fixed gates failed: {failed}")
    current_values = dict(initial_values)
    for pass_index, record in enumerate(pass_records):
        expected_keys = {
            "pass_index",
            "audit_only_no_commit",
            "sample_count",
            "batch_count",
            "max_relative_scale_change",
            "worst_relative_change_site",
            "observation_copy_dtype",
            "observation_copy_before_square",
            "accumulator_dtype",
            "candidate_running_ms",
            "candidate_running_ms_sha256",
            "relative_change_by_site",
            "relative_change_by_site_sha256",
            "site_call_count",
            "product_component_call_count",
            "committed_running_ms_sha256",
        }
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise RuntimeError(f"Calibration pass {pass_index} fields differ")
        candidates = record["candidate_running_ms"]
        relatives = record["relative_change_by_site"]
        if (
            type(record["pass_index"]) is not int
            or record["pass_index"] != pass_index
            or record["audit_only_no_commit"]
            is not (pass_index == FINAL_CALIBRATION_PASSES)
            or type(record["sample_count"]) is not int
            or record["sample_count"] != sample_count
            or type(record["batch_count"]) is not int
            or record["batch_count"] != batch_count
            or record["observation_copy_dtype"] != "float64"
            or record["observation_copy_before_square"] is not True
            or record["accumulator_dtype"] != "float64"
            or not isinstance(candidates, dict)
            or tuple(candidates) != names
            or not isinstance(relatives, dict)
            or tuple(relatives) != names
            or not isinstance(record["site_call_count"], dict)
            or tuple(record["site_call_count"]) != names
            or any(
                type(value) is not int
                for value in record["site_call_count"].values()
            )
            or set(record["site_call_count"].values()) != {batch_count}
            or not isinstance(record["product_component_call_count"], dict)
            or tuple(record["product_component_call_count"])
            != PRODUCT_COMPONENT_SITE_NAMES
            or any(
                type(value) is not int
                for value in record["product_component_call_count"].values()
            )
            or set(record["product_component_call_count"].values())
            != {batch_count}
        ):
            raise RuntimeError(f"Calibration pass {pass_index} inventory differs")
        if any(
            type(value) is not float
            or not math.isfinite(float(value))
            or float(value) < PRODUCT_RUNNING_MS_IDENTITY_FLOOR
            for value in candidates.values()
        ):
            raise RuntimeError(f"Calibration pass {pass_index} candidate differs")
        if any(
            type(value) is not float
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in relatives.values()
        ):
            raise RuntimeError(f"Calibration pass {pass_index} residual differs")
        if record["candidate_running_ms_sha256"] != canonical_sha256(candidates):
            raise RuntimeError(f"Calibration pass {pass_index} candidate digest differs")
        if record["relative_change_by_site_sha256"] != canonical_sha256(relatives):
            raise RuntimeError(f"Calibration pass {pass_index} residual digest differs")
        recomputed_relatives = {
            name: abs(float(candidates[name]) - float(current_values[name]))
            / abs(float(current_values[name]))
            for name in names
        }
        if not json_type_exact_equal(relatives, recomputed_relatives):
            raise RuntimeError(f"Calibration pass {pass_index} residual arithmetic differs")
        maximum = max(float(value) for value in relatives.values())
        worst_site = max(relatives, key=relatives.__getitem__)
        if (
            type(record["max_relative_scale_change"]) is not float
            or record["max_relative_scale_change"] != maximum
            or record["worst_relative_change_site"] != worst_site
        ):
            raise RuntimeError(f"Calibration pass {pass_index} maximum residual differs")
        if pass_index < FINAL_CALIBRATION_PASSES:
            current_values = {name: float(candidates[name]) for name in names}
        if record["committed_running_ms_sha256"] != canonical_sha256(current_values):
            raise RuntimeError(f"Calibration pass {pass_index} committed digest differs")
    if not json_type_exact_equal(
        current_values, {name: float(final_values[name]) for name in names}
    ):
        raise RuntimeError("Calibration final values differ from the last committed pass")
    audit_residual = pass_records[-1]["max_relative_scale_change"]
    if (
        type(calibration.get("audit_observed_max_relative_change")) is not float
        or calibration.get("audit_observed_max_relative_change") != audit_residual
        or audit_residual > FINAL_CALIBRATION_AUDIT_MAX_RELATIVE_CHANGE
    ):
        raise RuntimeError("Calibration fixed-pass audit residual failed")
    return {
        "validated": True,
        "site_count": len(names),
        "expected_calls_per_site": expected_calls,
        "pass_count": len(pass_records),
        "initializer_running_ms_sha256": canonical_sha256(dict(initial_values)),
        "final_running_ms_sha256": canonical_sha256(dict(final_values)),
        "audit_observed_max_relative_change": audit_residual,
    }


def deployment_state_from_training(
    model: ChiVLA,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    import torch

    validate_product_model(model, deployment=False, require_initialized_norms=False)
    source = model.state_dict()
    stripped = tuple(sorted(key for key in source if key.startswith("teacher_head.")))
    if stripped != ("teacher_head.bias", "teacher_head.weight"):
        raise RuntimeError(f"Unexpected training-only state keys: {stripped}")
    forbidden_prefixes = ("value_head.",)
    if any(key.startswith(forbidden_prefixes) for key in source):
        raise RuntimeError("Training checkpoint unexpectedly contains a value head")
    deployment = {
        key: value.detach().clone()
        for key, value in source.items()
        if key not in stripped
    }
    if len(deployment) + len(stripped) != len(source):
        raise RuntimeError("Deployment export removed an unapproved state entry")
    return deployment, stripped


def strict_load_deployment_state(
    model: ChiVLA, state: Mapping[str, torch.Tensor]
) -> None:
    from xvla.nn.normalization import RationalNorm

    validate_product_model(model, deployment=True, require_initialized_norms=False)
    if (
        "artifact_role" in state
        or "deployment_eligible" in state
        or "training_state" in state
    ):
        raise RuntimeError(
            "Deployment state is a training-only pre-calibration artifact envelope"
        )
    rejected = sorted(
        key for key in state if key.startswith("teacher_head.") or key.startswith("value_head.")
    )
    if rejected:
        raise RuntimeError(f"Deployment state contains training-only keys: {rejected}")
    incompatibility = model.load_state_dict(dict(state), strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"Strict deployment load failed: {incompatibility}")
    model.eval()
    for module in model.modules():
        if isinstance(module, RationalNorm):
            module.freeze()
    install_product_running_ms_fail_closed_guards(model)


def checkpoint_path(run_root: Path, seed: int, smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    return run_root / "checkpoints" / f"product_pade_rational_s{seed}{suffix}.pt"


def metadata_path(run_root: Path, seed: int, smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    return run_root / "metadata" / f"product_pade_rational_s{seed}{suffix}.json"


def training_result_path(run_root: Path, seed: int, smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    return run_root / "results" / f"train_product_pade_rational_s{seed}{suffix}.json"


def precalibration_state_path(run_root: Path, seed: int, smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    return (
        run_root
        / "training_only"
        / f"precalibration_ema_product_pade_rational_s{seed}{suffix}.pt"
    )


def deployment_completion_path(run_root: Path, seed: int, smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    return (
        run_root
        / "gates"
        / f"product_pade_rational_s{seed}{suffix}_deployment_complete.json"
    )


def calibration_resume_proof_path(run_root: Path) -> Path:
    return (
        run_root
        / "gates"
        / "product_pade_rational_smoke_s0_calibration_resume_proof.json"
    )


def validate_same_allocation_calibration_resume_proof(
    payload: Mapping[str, Any],
    *,
    expected_source_manifest_sha256: str,
    expected_manifest_bundle_sha256: str,
    expected_preflight_sha256: str,
    expected_source_closure: Mapping[str, str],
    expected_static_audit: Mapping[str, Any],
    expected_checkpoint_sha256: str,
    expected_metadata_sha256: str,
    expected_training_result_sha256: str,
    expected_precalibration_sha256: str,
    expected_precalibration_training_state_sha256: str,
    expected_completion_sha256: str,
    expected_completion_bundle_sha256: str,
    expected_training_environment: Mapping[str, Any],
    expected_calibration_attestation: Mapping[str, Any],
    expected_state_transition: Mapping[str, Any],
    expected_deployment_equivalence: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly validate the distinct real smoke recovery replay artifact."""

    from scripts.odt_direct_only_compliance import (
        EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT,
        EXPECTED_RUNTIME_GUARD_ENTRYPOINTS,
    )

    exact_keys = {
        "schema",
        "seed",
        "mode",
        "fresh_deployment_completion_sha256",
        "fresh_deployment_completion_bundle_sha256",
        "fresh_checkpoint_sha256",
        "fresh_metadata_sha256",
        "fresh_training_result_sha256",
        "precalibration_ema_state_sha256",
        "precalibration_training_state_sha256",
        "external_sha_authority_supplied",
        "optimizer_steps_executed",
        "same_slurm_gpu_allocation_as_fresh_smoke",
        "allocation",
        "calibration_elapsed_s",
        "fresh_and_resume_calibration_report_bitwise_equal",
        "fresh_and_resume_final_running_ms_bitwise_equal",
        "fresh_and_resume_deployment_state_bitwise_equal",
        "fresh_and_resume_raw_components_and_actions_bitwise_equal",
        "fresh_output_tensor_sha256",
        "resume_output_tensor_sha256",
        "calibration_attestation",
        "calibration_state_transition",
        "deployment_equivalence",
        "removed_training_only_keys",
        "source_manifest_sha256",
        "manifest_bundle_sha256",
        "preflight_certificate_sha256",
        "source_snapshot_start",
        "source_snapshot_end",
        "transitive_direct_only_static_audit",
        "runtime_direct_only_guard",
        "recovery_scope",
        "cross_version_recovery_supported",
    }
    if not isinstance(payload, Mapping) or set(payload) != exact_keys:
        raise RuntimeError("Calibration resume proof fields differ")
    for label, value in (
        ("source_manifest_sha256", expected_source_manifest_sha256),
        ("manifest_bundle_sha256", expected_manifest_bundle_sha256),
        ("preflight_sha256", expected_preflight_sha256),
        ("checkpoint_sha256", expected_checkpoint_sha256),
        ("metadata_sha256", expected_metadata_sha256),
        ("training_result_sha256", expected_training_result_sha256),
        ("precalibration_sha256", expected_precalibration_sha256),
        (
            "precalibration_training_state_sha256",
            expected_precalibration_training_state_sha256,
        ),
        ("completion_sha256", expected_completion_sha256),
        ("completion_bundle_sha256", expected_completion_bundle_sha256),
    ):
        _require_sha256(value, label=label)
    for label in (
        "fresh_deployment_completion_sha256",
        "fresh_deployment_completion_bundle_sha256",
        "fresh_checkpoint_sha256",
        "fresh_metadata_sha256",
        "fresh_training_result_sha256",
        "precalibration_ema_state_sha256",
        "precalibration_training_state_sha256",
        "fresh_output_tensor_sha256",
        "resume_output_tensor_sha256",
        "source_manifest_sha256",
        "manifest_bundle_sha256",
        "preflight_certificate_sha256",
    ):
        _require_sha256(payload.get(label), label=label)
    allocation = payload.get("allocation")
    allocation_keys = {
        "gpu",
        "compute_capability",
        "slurm_job_id",
        "slurm_job_gpus",
        "cuda_visible_devices",
    }
    static = payload.get("transitive_direct_only_static_audit")
    runtime = payload.get("runtime_direct_only_guard")
    replay = payload.get("deployment_equivalence")
    bitwise = replay.get("bitwise_checks", {}) if isinstance(replay, Mapping) else {}
    elapsed = payload.get("calibration_elapsed_s")
    conditions = {
        "schema": payload.get("schema") == CALIBRATION_RESUME_PROOF_SCHEMA,
        "identity": type(payload.get("seed")) is int
        and payload.get("seed") == 0
        and payload.get("mode") == "smoke",
        "artifact_hashes": payload.get("fresh_checkpoint_sha256")
        == expected_checkpoint_sha256
        and payload.get("fresh_metadata_sha256") == expected_metadata_sha256
        and payload.get("fresh_training_result_sha256")
        == expected_training_result_sha256
        and payload.get("precalibration_ema_state_sha256")
        == expected_precalibration_sha256
        and payload.get("precalibration_training_state_sha256")
        == expected_precalibration_training_state_sha256
        and payload.get("fresh_deployment_completion_sha256")
        == expected_completion_sha256
        and payload.get("fresh_deployment_completion_bundle_sha256")
        == expected_completion_bundle_sha256,
        "external_sha_zero_training": payload.get(
            "external_sha_authority_supplied"
        )
        is True
        and type(payload.get("optimizer_steps_executed")) is int
        and payload.get("optimizer_steps_executed") == 0,
        "allocation": payload.get("same_slurm_gpu_allocation_as_fresh_smoke")
        is True
        and isinstance(allocation, Mapping)
        and set(allocation) == allocation_keys
        and allocation.get("gpu") == "NVIDIA RTX A6000"
        and json_type_exact_equal(allocation.get("compute_capability"), [8, 6])
        and all(
            isinstance(allocation.get(key), str) and bool(allocation.get(key))
            for key in ("slurm_job_id", "slurm_job_gpus", "cuda_visible_devices")
        )
        and all(
            json_type_exact_equal(
                allocation.get(key), expected_training_environment.get(key)
            )
            for key in allocation_keys
        ),
        "calibration_elapsed": isinstance(elapsed, (int, float))
        and not isinstance(elapsed, bool)
        and math.isfinite(float(elapsed))
        and float(elapsed) > 0.0,
        "exact_claims": payload.get(
            "fresh_and_resume_calibration_report_bitwise_equal"
        )
        is True
        and payload.get("fresh_and_resume_final_running_ms_bitwise_equal") is True
        and payload.get("fresh_and_resume_deployment_state_bitwise_equal") is True
        and payload.get(
            "fresh_and_resume_raw_components_and_actions_bitwise_equal"
        )
        is True,
        "output_replay": isinstance(replay, Mapping)
        and json_type_exact_equal(replay, expected_deployment_equivalence)
        and isinstance(bitwise, Mapping)
        and bool(bitwise)
        and all(value is True for value in bitwise.values())
        and payload.get("fresh_output_tensor_sha256")
        == payload.get("resume_output_tensor_sha256")
        == replay.get("output_tensor_sha256"),
        "attestations": json_type_exact_equal(
            payload.get("calibration_attestation"),
            expected_calibration_attestation,
        )
        and json_type_exact_equal(
            payload.get("calibration_state_transition"), expected_state_transition
        ),
        "deployment_boundary": json_type_exact_equal(
            payload.get("removed_training_only_keys"),
            ["teacher_head.bias", "teacher_head.weight"],
        ),
        "source": payload.get("source_manifest_sha256")
        == expected_source_manifest_sha256
        and payload.get("manifest_bundle_sha256")
        == expected_manifest_bundle_sha256
        and payload.get("preflight_certificate_sha256")
        == expected_preflight_sha256
        and json_type_exact_equal(
            payload.get("source_snapshot_start"), dict(expected_source_closure)
        )
        and json_type_exact_equal(
            payload.get("source_snapshot_end"), dict(expected_source_closure)
        ),
        "static": isinstance(static, Mapping)
        and set(static)
        == {
            "scope",
            "entrypoints",
            "source_sha256",
            "source_count",
            "direct_qr_call_sites",
            "direct_qr_required",
            "guarded_dormant_spectral_norm_sites",
            "duplicate_top_level_definition_sites",
            "prohibited_self_overlap_sites",
            "prohibited_calls_found",
            "call_site_count",
        }
        and json_type_exact_equal(static, dict(expected_static_audit))
        and static.get("scope") == "transitive_local_import_closure"
        and json_type_exact_equal(
            static.get("entrypoints"), sorted(expected_source_closure)
        )
        and json_type_exact_equal(
            static.get("source_sha256"), dict(expected_source_closure)
        )
        and type(static.get("source_count")) is int
        and static.get("source_count") == len(expected_source_closure)
        and type(static.get("direct_qr_call_sites")) is int
        and static.get("direct_qr_call_sites") == 0
        and static.get("direct_qr_required") is False
        and json_type_exact_equal(static.get("prohibited_calls_found"), [])
        and json_type_exact_equal(static.get("prohibited_self_overlap_sites"), [])
        and json_type_exact_equal(static.get("guarded_dormant_spectral_norm_sites"), [])
        and json_type_exact_equal(static.get("duplicate_top_level_definition_sites"), [])
        and type(static.get("call_site_count")) is int
        and static.get("call_site_count") >= 0,
        "runtime": isinstance(runtime, Mapping)
        and set(runtime)
        == {
            "installed",
            "patched_entrypoints",
            "patched_entrypoint_count",
            "allowed_call_count",
            "allowed_calls",
            "prohibited_attempt_count",
            "prohibited_attempts",
        }
        and runtime.get("installed") is True
        and json_type_exact_equal(
            runtime.get("patched_entrypoints"),
            sorted(EXPECTED_RUNTIME_GUARD_ENTRYPOINTS),
        )
        and type(runtime.get("patched_entrypoint_count")) is int
        and runtime.get("patched_entrypoint_count")
        == EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT
        and type(runtime.get("allowed_call_count")) is int
        and runtime.get("allowed_call_count") == 0
        and json_type_exact_equal(runtime.get("allowed_calls"), [])
        and type(runtime.get("prohibited_attempt_count")) is int
        and runtime.get("prohibited_attempt_count") == 0
        and json_type_exact_equal(runtime.get("prohibited_attempts"), []),
        "scope": payload.get("recovery_scope") == CALIBRATION_RECOVERY_SCOPE
        and payload.get("cross_version_recovery_supported") is False,
    }
    failed = sorted(name for name, passed in conditions.items() if not passed)
    if failed:
        raise RuntimeError(f"Calibration resume proof failed strict gates: {failed}")
    return {"validated": True, "conditions": conditions}


def source_manifest_path(run_root: Path) -> Path:
    return run_root / "manifest" / "product_pade_rational_source_manifest.json"


def preflight_result_path(run_root: Path) -> Path:
    return run_root / "gates" / "product_pade_rational_preflight.json"


def evaluation_result_path(
    run_root: Path, seed: int, task_start: int, task_end: int, smoke: bool
) -> Path:
    suffix = "_smoke" if smoke else ""
    return (
        run_root
        / "results"
        / f"eval_product_pade_rational_s{seed}_t{task_start}_{task_end}{suffix}.json"
    )


def aggregate_result_path(run_root: Path) -> Path:
    return run_root / "results" / "product_pade_rational_three_seed_aggregate.json"


def seed0_capability_gate_path(run_root: Path) -> Path:
    return run_root / "gates" / "product_pade_rational_seed0_capability_gate.json"


def _deployment_completion_bound_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "schema",
            "complete",
            "seed",
            "mode",
            "checkpoint",
            "checkpoint_sha256",
            "metadata",
            "metadata_sha256",
            "training_result",
            "training_result_sha256",
            "precalibration_ema_state",
            "precalibration_ema_state_sha256",
            "source_manifest_sha256",
            "manifest_bundle_sha256",
        )
    }


def publish_deployment_completion_exclusive(
    path: Path,
    *,
    seed: int,
    mode: str,
    checkpoint: Path,
    metadata: Path,
    training_result: Path,
    precalibration_ema_state: Path,
    source_manifest_sha256: str,
    manifest_bundle_sha256: str,
) -> dict[str, Any]:
    """Publish the final marker only after every immutable seed artifact exists."""

    artifact_paths = {
        "checkpoint": checkpoint,
        "metadata": metadata,
        "training_result": training_result,
        "precalibration_ema_state": precalibration_ema_state,
    }
    for label, artifact in artifact_paths.items():
        if artifact.is_symlink() or not artifact.is_file():
            raise RuntimeError(f"Cannot complete seed with missing {label}: {artifact}")
    payload: dict[str, Any] = {
        "schema": DEPLOYMENT_COMPLETION_SCHEMA,
        "complete": True,
        "seed": seed,
        "mode": mode,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint),
        "metadata": metadata.as_posix(),
        "metadata_sha256": file_sha256(metadata),
        "training_result": training_result.as_posix(),
        "training_result_sha256": file_sha256(training_result),
        "precalibration_ema_state": precalibration_ema_state.as_posix(),
        "precalibration_ema_state_sha256": file_sha256(precalibration_ema_state),
        "source_manifest_sha256": _require_sha256(
            source_manifest_sha256, label="source_manifest_sha256"
        ),
        "manifest_bundle_sha256": _require_sha256(
            manifest_bundle_sha256, label="manifest_bundle_sha256"
        ),
    }
    payload["artifact_bundle_sha256"] = canonical_sha256(
        _deployment_completion_bound_fields(payload)
    )
    write_json_exclusive(path, payload)
    os.chmod(path, 0o444)
    return payload


def load_and_validate_deployment_completion(
    path: Path,
    *,
    expected_seed: int,
    expected_mode: str,
    expected_checkpoint: Path,
    expected_metadata: Path,
    expected_training_result: Path,
    expected_precalibration_ema_state: Path,
    expected_source_manifest_sha256: str,
    expected_manifest_bundle_sha256: str,
) -> dict[str, Any]:
    """Reject partial publication and bind every completed deployment artifact."""

    payload = read_json_object_physical(path, label="Deployment completion marker")
    exact_keys = {
        "schema",
        "complete",
        "seed",
        "mode",
        "checkpoint",
        "checkpoint_sha256",
        "metadata",
        "metadata_sha256",
        "training_result",
        "training_result_sha256",
        "precalibration_ema_state",
        "precalibration_ema_state_sha256",
        "source_manifest_sha256",
        "manifest_bundle_sha256",
        "artifact_bundle_sha256",
    }
    if set(payload) != exact_keys:
        raise RuntimeError("Deployment completion marker fields differ")
    expected_paths = {
        "checkpoint": expected_checkpoint,
        "metadata": expected_metadata,
        "training_result": expected_training_result,
        "precalibration_ema_state": expected_precalibration_ema_state,
    }
    conditions = {
        "schema": payload.get("schema") == DEPLOYMENT_COMPLETION_SCHEMA,
        "complete": payload.get("complete") is True,
        "seed": type(payload.get("seed")) is int
        and payload.get("seed") == expected_seed,
        "mode": payload.get("mode") == expected_mode,
        "source_manifest": payload.get("source_manifest_sha256")
        == expected_source_manifest_sha256,
        "manifest_bundle": payload.get("manifest_bundle_sha256")
        == expected_manifest_bundle_sha256,
        "bundle": payload.get("artifact_bundle_sha256")
        == canonical_sha256(_deployment_completion_bound_fields(payload)),
    }
    for label, artifact in expected_paths.items():
        conditions[f"{label}_path"] = payload.get(label) == artifact.as_posix()
        conditions[f"{label}_physical"] = artifact.is_file() and not artifact.is_symlink()
        conditions[f"{label}_sha256"] = (
            artifact.is_file()
            and not artifact.is_symlink()
            and payload.get(f"{label}_sha256") == file_sha256(artifact)
        )
    failed = sorted(key for key, passed in conditions.items() if not passed)
    if failed:
        raise RuntimeError(f"Deployment completion marker failed gates: {failed}")
    return payload


def assert_outputs_absent(run_root: Path, *, include_smoke: bool = False) -> None:
    paths: list[Path] = [
        aggregate_result_path(run_root),
        seed0_capability_gate_path(run_root),
        calibration_resume_proof_path(run_root),
    ]
    modes = (False, True) if include_smoke else (False,)
    for smoke in modes:
        for seed in SEEDS:
            paths.extend(
                (
                    checkpoint_path(run_root, seed, smoke),
                    metadata_path(run_root, seed, smoke),
                    training_result_path(run_root, seed, smoke),
                    precalibration_state_path(run_root, seed, smoke),
                    deployment_completion_path(run_root, seed, smoke),
                )
            )
            shards = ((0, 1),) if smoke else SHARDS
            paths.extend(
                evaluation_result_path(run_root, seed, start, end, smoke)
                for start, end in shards
            )
    collisions = [str(path) for path in paths if path.exists() or path.is_symlink()]
    if collisions:
        raise FileExistsError(f"Refusing existing product-lane outputs: {collisions}")


def validate_seed(seed: int) -> None:
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}, got {seed}")


def training_recipe(smoke: bool) -> dict[str, Any]:
    return {
        "steps": SMOKE_STEPS if smoke else FULL_STEPS,
        "batch_size": SMOKE_BATCH_SIZE if smoke else FULL_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "minimum_learning_rate_ratio": MIN_LEARNING_RATE_RATIO,
        "warmup_fraction": WARMUP_FRACTION,
        "optimizer": "AdamW",
        "optimizer_betas": [0.9, 0.95],
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "ema_decay": EMA_DECAY,
        "post_ema_simultaneous_rational_calibration": {
            "update_passes": FINAL_CALIBRATION_PASSES,
            "final_no_commit_audit_pass": True,
            "audit_max_relative_change": FINAL_CALIBRATION_AUDIT_MAX_RELATIVE_CHANGE,
            "batch_size": FINAL_CALIBRATION_BATCH_SIZE,
            "full_cache_in_fixed_order": True,
            "all_live_sites_updated_together_after_each_pass": True,
            "commit_protocol": (
                "all_candidates_validated_before_sequential_commit_with_exception_rollback"
            ),
            "commit_exception_rollback": True,
            "intra_pass_exception_rollback_source": (
                "in_memory_precommit_running_ms_snapshot"
            ),
            "process_level_retry_source": (
                "immutable_training_only_precalibration_ema_state"
            ),
            "initializer": CALIBRATION_INITIALIZER_LABEL,
            "old_training_buffers_discarded": False,
            "all_sites_frozen_before_first_pass": True,
            "observation_copy_dtype": "float64",
            "observation_copy_before_square": True,
            "accumulator_dtype": "float64",
            "no_value_changing_clamp_or_fallback_on_accepted_path": True,
            "invalid_anchor_rejected_before_frozen_forward": True,
            "initializer_and_every_candidate_floor_verified_before_use_or_commit": True,
            "training_only_precalibration_ema_state_published_exclusively": True,
            "calibration_only_resume_requires_external_artifact_sha256": True,
            "recovery_scope": CALIBRATION_RECOVERY_SCOPE,
            "cross_version_recovery_supported": False,
            "convergence_claim": (
                "four_fixed_simultaneous_update_passes_then_no_commit_audit_at_most_5_percent"
            ),
            "exact_fixed_point_convergence_claimed": False,
        },
        "autocast_dtype": "bfloat16",
        "matmul_precision": "high",
        "curriculum": {
            "enabled": True,
            "temperature_start": 1.0,
            "temperature_end": 0.02,
        },
        "distillation": {
            "training_only_linear_teacher": True,
            "teacher_loss_weight": 1.0,
            "center_anchor_start_weight": 1.0,
            "center_anchor_schedule": "lambda * (1 - progress)",
        },
    }


def evaluation_protocol(smoke: bool, task_start: int, task_end: int) -> dict[str, Any]:
    if smoke:
        if (task_start, task_end) != (0, 1):
            raise ValueError("Smoke evaluation is fixed to task range [0, 1)")
        episodes, max_steps = 1, 8
    else:
        if (task_start, task_end) not in SHARDS:
            raise ValueError(f"Full evaluation requires a frozen shard, got {(task_start, task_end)}")
        episodes, max_steps = EVALUATION_EPISODES_PER_TASK, EVALUATION_MAX_STEPS
    return {
        "suite": SUITE,
        "task_start": task_start,
        "task_end": task_end,
        "episodes_per_task": episodes,
        "max_steps": max_steps,
        "settle_steps": EVALUATION_SETTLE_STEPS,
        "settle_gate": (
            "every dummy step must return a finite observation and reward, "
            "with reward <= 0 and done false"
        ),
        "execution_horizon": EVALUATION_EXEC_HORIZON,
        "resolution": RESOLUTION,
        "canonical_packaged_initial_states": True,
        "require_unique_initial_states": True,
        "observation_transform": "rotate_180_degrees",
        "gripper_boundary": "+1 iff predicted_gripper > 0, else -1",
        "early_stop_gate": (
            "one or more policy actions; stopping before max_steps requires "
            "success or done"
        ),
        "matmul_precision": EVALUATION_MATMUL_PRECISION,
    }


def assert_cache_identity() -> dict[str, Any]:
    root = repository_root()
    path = root / CACHE_PATH
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Pinned cache is missing or nonphysical: {path}")
    observed = file_sha256(path)
    if observed != CACHE_SHA256:
        raise RuntimeError(f"Pinned cache SHA-256 differs: {observed}")
    return {
        "path": CACHE_PATH.as_posix(),
        "sha256": observed,
        "frame_count": CACHE_FRAME_COUNT,
        "sample_count": CACHE_SAMPLE_COUNT,
    }


def read_training_metadata(path: Path) -> dict[str, Any]:
    payload = read_json_object_physical(path, label="Training metadata")
    if payload.get("schema") != SCHEMA or payload.get("recipe_version") != RECIPE_VERSION:
        raise RuntimeError("Training metadata schema or recipe differs")
    return payload


def validate_preflight_certificate(
    path: Path, manifest_path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Preflight certificate is missing or nonphysical: {path}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"Duplicate preflight-certificate key {key!r}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(),
        object_pairs_hook=reject_duplicates,
        parse_constant=_reject_json_constant,
        parse_float=_finite_json_float,
    )
    from scripts.odt_direct_only_compliance import (
        EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT,
        EXPECTED_RUNTIME_GUARD_ENTRYPOINTS,
    )

    stored_static = payload.get("postimport_transitive_direct_only_static_audit", {})
    stored_import_guard = payload.get("runtime_direct_only_guard_at_import", {})
    stored_final_guard = payload.get("runtime_direct_only_guard_final", {})
    required = {
        "schema": payload.get("schema") == f"{SCHEMA}_preflight",
        "ready": payload.get("ready") is True,
        "mode": payload.get("mode") == "smoke",
        "seeds": json_type_exact_equal(payload.get("seeds"), list(SEEDS)),
        "cache_sha": payload.get("cache", {}).get("sha256") == CACHE_SHA256,
        "cache_frames": type(
            payload.get("cache_structure", {}).get("frame_count")
        )
        is int
        and payload.get("cache_structure", {}).get("frame_count")
        == CACHE_FRAME_COUNT,
        "cache_samples": type(
            payload.get("cache_structure", {}).get("sample_count")
        )
        is int
        and payload.get("cache_structure", {}).get("sample_count")
        == CACHE_SAMPLE_COUNT,
        "task_mapping": json_type_exact_equal(
            payload.get("task_metadata", {}).get("dataset_to_official_task"),
            DATASET_TO_OFFICIAL_TASK,
        ),
        "official_tasks": json_type_exact_equal(
            payload.get("official_tasks"),
            {str(key): value for key, value in OFFICIAL_TASK_LANGUAGES.items()},
        ),
        "vocab_sha": payload.get("vocab_sha256") == VOCAB_SHA256,
        "pade_sites": type(
            payload.get("deployment_topology", {}).get("pade_site_count")
        )
        is int
        and payload.get("deployment_topology", {}).get("pade_site_count")
        == EXPECTED_PADE_SITE_COUNT,
        "product_factors": type(
            payload.get("deployment_topology", {}).get("product_factors")
        )
        is int
        and payload.get("deployment_topology", {}).get("product_factors")
        == N_FACTORS,
        "product_root_width": type(
            payload.get("deployment_topology", {}).get(
                "product_component_root_projective_width"
            )
        )
        is int
        and payload.get("deployment_topology", {}).get(
            "product_component_root_projective_width"
        )
        == 285,
        "export_exact": payload.get("deployment_export", {}).get(
            "retained_state_bitwise_equal"
        )
        is True,
        "manifest_sha": payload.get("source_manifest", {}).get("sha256")
        == file_sha256(manifest_path),
        "manifest_source_bundle": payload.get("source_manifest", {}).get(
            "bundle_sha256"
        )
        == manifest.get("source_bundle_sha256"),
        "manifest_input_bundle": payload.get("source_manifest", {}).get(
            "authenticated_inputs_bundle_sha256"
        )
        == manifest.get("authenticated_inputs_bundle_sha256"),
        "manifest_launch_bundle": payload.get("source_manifest", {}).get(
            "launch_bundle_sha256"
        )
        == manifest.get("launch_bundle_sha256"),
        "manifest_bundle": payload.get("source_manifest", {}).get(
            "manifest_bundle_sha256"
        )
        == manifest.get("manifest_bundle_sha256"),
        "static_source": static_audit_record_is_closed(
            stored_static,
            expected_source_closure=manifest["source_closure"],
            direct_qr_required=False,
            direct_qr_call_sites=0,
        ),
        "static_calls": json_type_exact_equal(
            stored_static.get("prohibited_calls_found"), []
        ),
        "static_overlap": json_type_exact_equal(
            stored_static.get("prohibited_self_overlap_sites"), []
        ),
        "static_dormant": json_type_exact_equal(
            stored_static.get("guarded_dormant_spectral_norm_sites"), []
        )
        and json_type_exact_equal(
            stored_static.get("duplicate_top_level_definition_sites"), []
        ),
        "import_guard": runtime_guard_record_is_closed(
            stored_import_guard,
            expected_entrypoints=EXPECTED_RUNTIME_GUARD_ENTRYPOINTS,
            expected_entrypoint_count=EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT,
        ),
        "final_guard": runtime_guard_record_is_closed(
            stored_final_guard,
            expected_entrypoints=EXPECTED_RUNTIME_GUARD_ENTRYPOINTS,
            expected_entrypoint_count=EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT,
        ),
        "claim_boundary": payload.get("claim_boundary") == DEPLOYMENT_CLAIM_BOUNDARY,
    }
    failed = sorted(key for key, value in required.items() if not value)
    if failed:
        raise RuntimeError(f"Preflight certificate failed frozen gates: {failed}")
    return {
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "validation_gates": required,
    }
