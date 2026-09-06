#!/usr/bin/env python3
"""Frozen identities and exact BDDL rewriting for the target-swap certificate."""

from __future__ import annotations

import copy
import base64
import hashlib
import json
import re
import subprocess
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


SCHEMA_MANIFEST = "xvla-counterfactual-target-swap-manifest-v1"
SCHEMA_RUN = "xvla-counterfactual-target-swap-run-v1"
SCHEMA_SUMMARY = "xvla-counterfactual-target-swap-summary-v1"
SPECIFICITY_SUMMARY_SCHEMA = "xvla-local-instruction-specificity-summary-v2"

CHECKPOINTS = {
    0: {
        "path": "artifacts/ckpt_linear_rat_vit_s0_v2.pt",
        "sha256": "96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9",
    },
    1: {
        "path": "artifacts/ckpt_linear_rat_vit_s1.pt",
        "sha256": "cc0780b989165a80a449c03cbb3574e2f56b2e5747a64cf32d1fde418df8ec3c",
    },
    2: {
        "path": "artifacts/ckpt_linear_rat_vit_s2.pt",
        "sha256": "413a770071bd8f16b924c5604f7c7a58567d2eb6c40c66aab202bcf046fec910",
    },
}

CACHE = {
    "path": "artifacts/libero_frames_100000_64.pkl",
    "sha256": "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662",
    "frames": 66984,
    "samples": 63352,
}
PROVENANCE = {
    "path": "results/cache_provenance_libero_object.json",
    "job_id": "830988",
    "repository": "lerobot/libero_object_image",
    "revision": "e1e080d7df1d0a359dff5c86c222e047549f447f",
    "metadata_sha256": "34caee9641ae50bb4e077de306a7d0031753757882da8b1f117e7ea36a486b42",
    "canonical_content_sha256": "01bc724b9bf8c158b983e34b82bbb40dce9dba2cb86dd9be2a5bf8910be341c7",
}
SPECIFICITY = {
    "path": "results/local_instruction_specificity_v2_summary.json",
    "job_id": "831016",
}
MANIFEST_PATH = "artifacts/counterfactual_target_swap_v1_manifest.json"
BDDL_DIRECTORY = "artifacts/counterfactual_target_swap_v1_bddl"
SMOKE_RESULT_PATH = "results/counterfactual_target_swap_v1_smoke.json"
SUMMARY_RESULT_PATH = "results/counterfactual_target_swap_v1_summary.json"

PROMPTS = {
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
TARGETS = {index: text[len("pick up the ") : -len(" and place it in the basket")] for index, text in PROMPTS.items()}
BDDL_LANGUAGES = {index: f"Pick the {target} and place it in the basket" for index, target in TARGETS.items()}
OBJECT_SYMBOLS = {index: target.replace(" ", "_") + "_1" for index, target in TARGETS.items()}
RECEPTACLE_SYMBOL = "basket_1"

ORIGINAL_BDDL = {
    0: ("pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl", "df088984da13131f8332ee0f13a7896c6a97afd02ee5007a42e8fc5e0893571e"),
    1: ("pick_up_the_cream_cheese_and_place_it_in_the_basket.bddl", "7019f37ee158d67a21338a6df0c441dd8f84979b946b5e20bb49463ef0508ea2"),
    2: ("pick_up_the_salad_dressing_and_place_it_in_the_basket.bddl", "024978e9e8f43a49b49d0965f4f426d364ce056307f0fcf5733dc4682fac437a"),
    3: ("pick_up_the_bbq_sauce_and_place_it_in_the_basket.bddl", "8a7e37b76fce5621e649e260dbcfcec7ce2f9a055ca09a4f1ebaf5ca74a739c2"),
    4: ("pick_up_the_ketchup_and_place_it_in_the_basket.bddl", "4a4e545483a3fe30cf0ef3dc05bfe5e6ad5e37c94b4376dfa73fe1b60e75b319"),
    5: ("pick_up_the_tomato_sauce_and_place_it_in_the_basket.bddl", "b1f4bb69d256a05f693838de46182bb0d3de0de68eec350df1b5b1e76f6c136d"),
    6: ("pick_up_the_butter_and_place_it_in_the_basket.bddl", "5d8053b40ff33246fdbcc38548db84032c6bb647b4b88b6646d9cf6b73a5006c"),
    7: ("pick_up_the_milk_and_place_it_in_the_basket.bddl", "9910aabf6717e8ba3e24a3f8500d9bce9ee075346ea6961e8fd766f1257d2996"),
    8: ("pick_up_the_chocolate_pudding_and_place_it_in_the_basket.bddl", "674ceabc400b7b16d46e8eaf678709d4a633235c7e0a80233775044fe38b19b0"),
    9: ("pick_up_the_orange_juice_and_place_it_in_the_basket.bddl", "6298533e7bcfb83e40779a77fda39216e8cd53f22bf1af6388585dad0abb0b50"),
}

PROTOCOL = {
    "suite": "libero_object",
    "task_indices": list(range(10)),
    "canonical_episode_start": 40,
    "canonical_episode_end_exclusive": 50,
    "distractors_per_task": 5,
    "checkpoint_seeds": [0, 1, 2],
    "counterfactual_conditions": ["matching_counterfactual_prompt", "original_prompt"],
    "control_condition": "original_goal_with_original_prompt",
    "settle_steps": 10,
    "action_horizon": 8,
    "execution_horizon": 8,
    "max_steps": 280,
    "resolution": 64,
    "matmul_precision": "highest",
    "deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cublas_workspace_config": ":4096:8",
    "gpu_family": "a6000",
    "gpu_name": "NVIDIA RTX A6000",
    "gpu_compute_capability": [8, 6],
    "runtime_versions": {
        "python": "3.10.19",
        "torch": "2.7.1",
        "torch_build": "2.7.1+cu126",
        "cuda": "12.6",
        "numpy": "1.26.4",
        "pillow": "12.1.1",
        "mujoco": "3.5.0",
        "robosuite": "1.4.1",
    },
    "task_shards": [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10]],
    "paired_counterfactual_trials_per_checkpoint": 500,
    "counterfactual_rollouts_per_checkpoint": 1000,
    "original_goal_controls_per_checkpoint": 100,
    "total_rollouts_per_checkpoint": 1100,
    "bootstrap_draws": 20000,
    "bootstrap_seed": 2026082601,
}

LIBERO_RUNTIME = {
    "repository_root": "/athenahomes/joy/LIBERO",
    "package_root": "/athenahomes/joy/LIBERO/libero",
    "git_commit": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
    "git_status_porcelain": "?? libero/__init__.py\n",
    "python_file_count": 83,
    "python_tree_sha256": "b11dfeed8e4430a392a38a2dcc5c21ac186fb0345d7de2979cd53e4af9fb0b55",
}

FROZEN_GATES = {
    "minimum_matching_counterfactual_task_macro_success": 0.25,
    "minimum_pooled_paired_advantage": 0.15,
    "paired_advantage_must_be_positive_every_checkpoint": True,
    "minimum_positive_pooled_task_advantages": 8,
    "task_checkpoint_stratified_bootstrap_lower_strictly_above": 0.0,
    "minimum_original_goal_control_success_every_checkpoint": 0.60,
    "all_three_checkpoints_must_pass": True,
}

LOCAL_SPECIFICITY_PROTOCOL = {
    "suite": "libero_object",
    "task_start": 0,
    "task_end": 10,
    "primary_episode_start": 10,
    "primary_eps_per_task": 40,
    "smoke_episode_start": 0,
    "smoke_eps_per_task": 1,
    "settle_steps": 10,
    "action_horizon": 8,
    "resolution": 64,
    "matmul_precision": "highest",
    "language_conditions": "ten official Object prompts plus empty/BOS-only",
    "scene_objects": "one official target plus five co-present distractors",
    "primary_metric": "mean over five distractors of 0.5 * (delta_A-delta_B) dot (unit_A-unit_B)",
    "control_metric": "for each of five target-distractor axes, rank its semantic score against 12 ordered contrasts among four truly absent-object prompts, then average ranks",
}
LOCAL_SPECIFICITY_GATES = {
    "completed_trials_per_checkpoint": 400,
    "primary_canonical_episode_start": 10,
    "primary_canonical_episode_end_exclusive": 50,
    "minimum_positive_task_means": 8,
    "semantic_mean_hierarchical_resampling_lower_strictly_above_m": 0.0,
    "minimum_mean_specificity_rank": 0.75,
    "specificity_rank_hierarchical_resampling_lower_strictly_above": 0.50,
    "hierarchical_resampling_draws": 20000,
    "all_three_checkpoints_must_pass": True,
}

SOURCE_PATHS = (
    "athena/counterfactual_target_swap_common.py",
    "athena/libero_dataset_metadata.py",
    "athena/run_local_instruction_specificity.py",
    "athena/run_xvla_experiment.py",
    "athena/summarize_local_instruction_specificity.py",
    "xvla/__init__.py",
    "xvla/models/__init__.py",
    "xvla/models/lm.py",
    "xvla/models/vla.py",
    "xvla/models/vit.py",
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

FROZEN_IMPORTED_SOURCE_SHA256 = {
    "athena/libero_dataset_metadata.py": "3e14b117ee72b010939c0fdd2c20417778b89a8691156553a0bfb2ad2d0b4edb",
    "athena/run_local_instruction_specificity.py": "9ded0920d955e3f502379b77c5782fa232ff20c0978298ec190041980ba93d93",
    "athena/run_xvla_experiment.py": "91ae342892e32b0aa019a43ff06b5809dca2a49d4be20ce9a55fbad1b15cdbb3",
    "athena/summarize_local_instruction_specificity.py": "d61cde37abedb6296791316b23be53eea841a8ad6289e2ab3b3d8188ac3c8ec7",
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
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_array(array: Any) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def encode_array(array: Any) -> dict[str, Any]:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "sha256": hash_array(contiguous),
        "codec": "zlib+base64",
        "data": base64.b64encode(zlib.compress(contiguous.tobytes(), level=9)).decode("ascii"),
    }


def decode_array(value: dict[str, Any]) -> Any:
    import numpy as np

    if value.get("codec") != "zlib+base64":
        raise RuntimeError("Unsupported array codec")
    dtype = np.dtype(str(value["dtype"]))
    shape = tuple(int(item) for item in value["shape"])
    raw = zlib.decompress(base64.b64decode(value["data"], validate=True))
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw) != expected_bytes:
        raise RuntimeError("Encoded array byte count differs from dtype and shape")
    array = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if hash_array(array) != value.get("sha256"):
        raise RuntimeError("Encoded array digest does not reproduce")
    return array


def source_hashes(repository_root: Path, extra: tuple[str, ...] = ()) -> dict[str, str]:
    paths = tuple(dict.fromkeys((*SOURCE_PATHS, *extra)))
    result = {}
    for relative in paths:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source is absent: {path}")
        result[relative] = file_sha256(path)
        if relative in FROZEN_IMPORTED_SOURCE_SHA256 and result[relative] != FROZEN_IMPORTED_SOURCE_SHA256[relative]:
            raise RuntimeError(f"Imported source differs from the prospective frozen hash: {relative}")
    return result


def python_tree_identity(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    paths = sorted(root.rglob("*.py"))
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return {"python_file_count": len(paths), "python_tree_sha256": digest.hexdigest()}


def validate_libero_runtime() -> dict[str, Any]:
    repository_root = Path(LIBERO_RUNTIME["repository_root"])
    package_root = Path(LIBERO_RUNTIME["package_root"])
    if not repository_root.is_dir() or not package_root.is_dir():
        raise RuntimeError("Frozen LIBERO source tree is absent")
    tree = python_tree_identity(package_root)
    commit = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    observed = {
        "repository_root": str(repository_root),
        "package_root": str(package_root),
        "git_commit": commit,
        "git_status_porcelain": status,
        **tree,
    }
    if observed != LIBERO_RUNTIME:
        raise RuntimeError(f"LIBERO runtime source identity differs: {observed}")
    return observed


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_specificity_summary(path: Path) -> dict[str, Any]:
    """Refuse continuation unless the prospective prerequisite passed in full."""
    result = load_json(path)
    repository_root = path.resolve().parents[1]
    errors = []
    if result.get("schema") != SPECIFICITY_SUMMARY_SCHEMA:
        errors.append("wrong summary schema")
    if result.get("protocol") != LOCAL_SPECIFICITY_PROTOCOL:
        errors.append("specificity protocol differs from the frozen v2 protocol")
    if result.get("frozen_gates") != LOCAL_SPECIFICITY_GATES:
        errors.append("specificity gates differ from the frozen v2 gates")
    if result.get("cache_sha256") != CACHE["sha256"]:
        errors.append("specificity cache identity differs")
    if str(result.get("provenance_job_id")) != PROVENANCE["job_id"]:
        errors.append("specificity provenance job differs")
    if result.get("identity_validated") is not True:
        errors.append("specificity identities were not validated")
    expected_summary_source = repository_root / "athena/summarize_local_instruction_specificity.py"
    if not expected_summary_source.is_file() or result.get("summary_source_sha256") != file_sha256(expected_summary_source):
        errors.append("specificity summary source identity is stale")
    provenance_path = repository_root / PROVENANCE["path"]
    if not provenance_path.is_file() or result.get("provenance_result_sha256") != file_sha256(provenance_path):
        errors.append("specificity provenance result identity is stale")
    checkpoint_results = result.get("checkpoint_results")
    if not isinstance(checkpoint_results, dict) or set(checkpoint_results) != {"0", "1", "2"}:
        errors.append("specificity checkpoint set is incomplete")
    else:
        expected_local_runner = repository_root / "athena/run_local_instruction_specificity.py"
        expected_local_runner_sha = file_sha256(expected_local_runner) if expected_local_runner.is_file() else None
        local_imported_paths = (
            "athena/run_xvla_experiment.py",
            "xvla/models/vla.py",
            "xvla/models/vit.py",
            "xvla/nn/attention.py",
            "xvla/nn/normalization.py",
        )
        expected_local_imports = {
            relative: file_sha256(repository_root / relative)
            for relative in local_imported_paths
            if (repository_root / relative).is_file()
        }
        if expected_local_runner_sha is None or len(expected_local_imports) != len(local_imported_paths):
            errors.append("specificity runner or imported source closure is absent")
        for seed in range(3):
            row = checkpoint_results[str(seed)]
            expected = CHECKPOINTS[seed]
            if Path(str(row.get("checkpoint", ""))).name != Path(expected["path"]).name:
                errors.append(f"specificity checkpoint basename differs for seed {seed}")
            if row.get("checkpoint_sha256") != expected["sha256"]:
                errors.append(f"specificity checkpoint SHA differs for seed {seed}")
            if int(row.get("trials", -1)) != 400:
                errors.append(f"specificity trial count differs for seed {seed}")
            checks = row.get("gate_checks")
            expected_check_names = {
                "semantic_score_resampling_lower_above_zero",
                "at_least_eight_positive_task_means",
                "mean_specificity_rank_at_least_0p75",
                "specificity_rank_resampling_lower_above_0p50",
            }
            if not isinstance(checks, dict) or set(checks) != expected_check_names or not all(value is True for value in checks.values()):
                errors.append(f"specificity component gate failed for seed {seed}")
            if row.get("passes_frozen_gate") is not True:
                errors.append(f"specificity checkpoint gate failed for seed {seed}")
            result_path = Path(str(row.get("result", "")))
            if not result_path.is_absolute():
                result_path = repository_root / result_path
            if not result_path.is_file() or row.get("result_sha256") != file_sha256(result_path):
                errors.append(f"specificity full-result identity is stale for seed {seed}")
                continue
            raw = load_json(result_path)
            if raw.get("schema") != "xvla-local-instruction-specificity-v2" or raw.get("mode") != "full" or raw.get("protocol") != LOCAL_SPECIFICITY_PROTOCOL:
                errors.append(f"specificity raw result protocol differs for seed {seed}")
                continue
            raw_identity = raw.get("identity", {})
            if (
                int(raw_identity.get("checkpoint_seed", -1)) != seed
                or raw_identity.get("checkpoint_sha256") != expected["sha256"]
                or raw_identity.get("cache_sha256") != CACHE["sha256"]
                or raw_identity.get("provenance_result_sha256") != result.get("provenance_result_sha256")
                or raw_identity.get("runner_source_sha256") != expected_local_runner_sha
                or raw_identity.get("imported_source_sha256") != expected_local_imports
            ):
                errors.append(f"specificity raw result source or artifact identity differs for seed {seed}")
            raw_evaluation = raw.get("evaluation", {})
            numerical = raw_evaluation.get("numerical_environment", {})
            if (
                raw_evaluation.get("matmul_precision") != "highest"
                or numerical.get("torch_version") != PROTOCOL["runtime_versions"]["torch_build"]
                or numerical.get("cuda_version") != PROTOCOL["runtime_versions"]["cuda"]
                or numerical.get("numpy_version") != PROTOCOL["runtime_versions"]["numpy"]
                or numerical.get("pillow_version") != PROTOCOL["runtime_versions"]["pillow"]
                or not isinstance(numerical.get("gpu"), str)
                or not numerical.get("gpu")
            ):
                errors.append(f"specificity raw result numerical runtime differs for seed {seed}")
    independently_reproduced = bool(
        isinstance(checkpoint_results, dict)
        and set(checkpoint_results) == {"0", "1", "2"}
        and all(checkpoint_results[str(seed)].get("passes_frozen_gate") is True for seed in range(3))
    )
    if result.get("overall_pass") is not True or result.get("claim_eligible") is not True:
        errors.append("specificity overall claim gate did not pass")
    if result.get("overall_pass") is not independently_reproduced:
        errors.append("specificity overall pass does not reproduce from checkpoints")
    if errors:
        raise RuntimeError("Target-swap prerequisite failed: " + ", ".join(errors))
    return result


@dataclass
class Token:
    text: str
    start: int
    end: int


@dataclass
class Node:
    value: str | None
    children: list["Node"]
    token: Token | None

    def plain(self) -> Any:
        return self.value if self.value is not None else [child.plain() for child in self.children]


TOKEN_PATTERN = re.compile(r"\(|\)|[^\s()]+")


def tokens(text: str) -> list[Token]:
    return [Token(match.group(), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def parse_bddl(text: str) -> Node:
    stream = tokens(text)
    index = 0

    def parse_one() -> Node:
        nonlocal index
        if index >= len(stream):
            raise RuntimeError("Unexpected end of BDDL")
        token = stream[index]
        index += 1
        if token.text == "(":
            children = []
            while index < len(stream) and stream[index].text != ")":
                children.append(parse_one())
            if index >= len(stream):
                raise RuntimeError("Unbalanced BDDL opening parenthesis")
            index += 1
            return Node(None, children, None)
        if token.text == ")":
            raise RuntimeError("Unexpected BDDL closing parenthesis")
        return Node(token.text, [], token)

    root = parse_one()
    if index != len(stream):
        raise RuntimeError("BDDL contains multiple root forms")
    return root


def _form(root: Node, name: str) -> Node:
    matches = []
    for child in root.children:
        if child.value is None and child.children and child.children[0].value == name:
            matches.append(child)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} form, found {len(matches)}")
    return matches[0]


def _problem_forms(root: Node) -> Node:
    if root.value is not None or len(root.children) < 3 or root.children[0].value != "define":
        raise RuntimeError("BDDL root is not a define form")
    return root


def _validate_fixed_shape(root: Node, task_index: int) -> dict[str, Any]:
    root = _problem_forms(root)
    language = _form(root, ":language")
    objects = _form(root, ":objects")
    interest = _form(root, ":obj_of_interest")
    goal = _form(root, ":goal")
    observed_language = " ".join(str(child.value) for child in language.children[1:])
    expected_bddl_language = BDDL_LANGUAGES[task_index]
    if observed_language != expected_bddl_language:
        raise RuntimeError(f"Task {task_index} BDDL language is unexpected: {observed_language}")
    object_tokens = [child.value for child in objects.children[1:]]
    if len(object_tokens) % 3 or any(object_tokens[offset + 1] != "-" for offset in range(0, len(object_tokens), 3)):
        raise RuntimeError("BDDL object declarations are not name-dash-type triples")
    declarations = {str(object_tokens[offset]): str(object_tokens[offset + 2]) for offset in range(0, len(object_tokens), 3)}
    expected_target = OBJECT_SYMBOLS[task_index]
    if declarations.get(expected_target) != expected_target.removesuffix("_1"):
        raise RuntimeError("Official target declaration is absent")
    if declarations.get("basket_1") != "basket":
        raise RuntimeError("Basket declaration is absent")
    groceries = {symbol for symbol in declarations if symbol in set(OBJECT_SYMBOLS.values())}
    if len(groceries) != 6 or expected_target not in groceries:
        raise RuntimeError(f"Expected six known grocery objects, found {sorted(groceries)}")
    if len(interest.children) != 3 or interest.children[1].value != expected_target or interest.children[2].value != "basket_1":
        raise RuntimeError("Object-of-interest form is not target plus basket")
    expected_goal = [":goal", ["And", ["In", expected_target, "basket_1_contain_region"]]]
    if goal.plain() != expected_goal:
        raise RuntimeError(f"Unexpected Object goal form: {goal.plain()}")
    return {
        "language": language,
        "objects": objects,
        "interest": interest,
        "goal": goal,
        "groceries": groceries,
        "declarations": declarations,
    }


def inspect_original_bddl(text: str, task_index: int) -> dict[str, Any]:
    info = _validate_fixed_shape(parse_bddl(text), task_index)
    prompt_by_symbol = {symbol: prompt for prompt, symbol in OBJECT_SYMBOLS.items()}
    present = sorted(prompt_by_symbol[symbol] for symbol in info["groceries"])
    if task_index not in present or len(present) != 6:
        raise RuntimeError("Present prompt mapping is invalid")
    return {
        "target_prompt_id": task_index,
        "present_prompt_ids": present,
        "distractor_prompt_ids": [prompt for prompt in present if prompt != task_index],
        "object_symbols": {str(prompt): OBJECT_SYMBOLS[prompt] for prompt in present},
    }


def rewrite_bddl(text: str, task_index: int, distractor_prompt_id: int) -> tuple[str, dict[str, Any]]:
    """Rewrite exactly three semantic slots and prove that no other AST node changed."""
    if distractor_prompt_id == task_index:
        raise ValueError("Counterfactual target must differ from the original target")
    root = parse_bddl(text)
    info = _validate_fixed_shape(root, task_index)
    mapping = inspect_original_bddl(text, task_index)
    if distractor_prompt_id not in mapping["distractor_prompt_ids"]:
        raise ValueError("Counterfactual target is not a co-present distractor")
    language_nodes = info["language"].children[1:]
    target_words = TARGETS[task_index].split()
    target_start_index = 2
    target_end_index = target_start_index + len(target_words)
    if (
        not language_nodes
        or [node.value for node in language_nodes[target_start_index:target_end_index]] != target_words
        or any(node.token is None for node in language_nodes)
    ):
        raise RuntimeError("Language token spans are unavailable")
    language_target_nodes = language_nodes[target_start_index:target_end_index]
    interest_node = info["interest"].children[1]
    goal_node = info["goal"].children[1].children[1].children[1]
    if interest_node.token is None or goal_node.token is None:
        raise RuntimeError("Target token spans are unavailable")
    replacements = [
        (language_target_nodes[0].token.start, language_target_nodes[-1].token.end, TARGETS[distractor_prompt_id]),
        (interest_node.token.start, interest_node.token.end, OBJECT_SYMBOLS[distractor_prompt_id]),
        (goal_node.token.start, goal_node.token.end, OBJECT_SYMBOLS[distractor_prompt_id]),
    ]
    rewritten = text
    for start, end, value in sorted(replacements, reverse=True):
        rewritten = rewritten[:start] + value + rewritten[end:]

    expected_root = parse_bddl(rewritten)
    # Compare all top-level forms, allowing changes only in the three declared slots.
    expected_parsed = copy.deepcopy(root)
    expected_language = _form(expected_parsed, ":language")
    expected_language.children = [expected_language.children[0]] + [Node(word, [], None) for word in BDDL_LANGUAGES[distractor_prompt_id].split()]
    _form(expected_parsed, ":obj_of_interest").children[1].value = OBJECT_SYMBOLS[distractor_prompt_id]
    _form(expected_parsed, ":goal").children[1].children[1].children[1].value = OBJECT_SYMBOLS[distractor_prompt_id]
    if expected_root.plain() != expected_parsed.plain():
        raise RuntimeError("Rewritten BDDL differs outside the three frozen semantic slots")
    rewritten_info = _form(expected_root, ":goal").plain()
    if rewritten_info != [":goal", ["And", ["In", OBJECT_SYMBOLS[distractor_prompt_id], "basket_1_contain_region"]]]:
        raise RuntimeError("Rewritten BDDL goal did not validate")
    if _form(expected_root, ":init").plain() != _form(root, ":init").plain():
        raise RuntimeError("Rewritten BDDL altered the initial state")
    if _form(expected_root, ":regions").plain() != _form(root, ":regions").plain():
        raise RuntimeError("Rewritten BDDL altered regions")
    if _form(expected_root, ":objects").plain() != _form(root, ":objects").plain():
        raise RuntimeError("Rewritten BDDL altered object declarations")
    return rewritten, {
        **mapping,
        "counterfactual_prompt_id": distractor_prompt_id,
        "counterfactual_object_symbol": OBJECT_SYMBOLS[distractor_prompt_id],
        "changed_semantic_slots": [":language[target phrase]", ":obj_of_interest[0]", ":goal/And/In[0]"],
        "ast_exactly_validated": True,
        "init_regions_objects_unchanged": True,
    }


def validate_hex_digest(value: Any, label: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise RuntimeError(f"Malformed SHA256 for {label}")
    return text


def iter_mapping_rows(manifest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    rows = manifest.get("mappings")
    if not isinstance(rows, list):
        raise RuntimeError("Manifest mappings must be a list")
    yield from rows


def validate_manifest(path: Path, repository_root: Path) -> dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("schema") != SCHEMA_MANIFEST or manifest.get("protocol") != PROTOCOL:
        raise RuntimeError("Target-swap manifest schema or protocol is invalid")
    if manifest.get("frozen_gates") != FROZEN_GATES:
        raise RuntimeError("Target-swap manifest gates differ")
    expected_manifest_checkpoints = {str(seed): identity for seed, identity in CHECKPOINTS.items()}
    if manifest.get("checkpoints") != expected_manifest_checkpoints:
        raise RuntimeError("Target-swap manifest checkpoint declarations differ")
    if manifest.get("mapping_count") != 50:
        raise RuntimeError("Target-swap manifest mapping count declaration differs")
    if manifest.get("specificity_gate_validated") is not True or manifest.get("specificity_gate_fields") != {
        "identity_validated": True,
        "overall_pass": True,
        "claim_eligible": True,
        "all_checkpoint_gates_pass": True,
    }:
        raise RuntimeError("Target-swap manifest prerequisite gate declaration differs")
    cache_identity = manifest.get("cache", {})
    if any(cache_identity.get(key) != value for key, value in CACHE.items()) or cache_identity.get("live_sha256") != CACHE["sha256"]:
        raise RuntimeError("Target-swap manifest cache identity differs")
    provenance_identity = manifest.get("provenance", {})
    if any(provenance_identity.get(key) != value for key, value in PROVENANCE.items()) or provenance_identity.get("verified") is not True:
        raise RuntimeError("Target-swap manifest provenance declaration differs")
    provenance_path = repository_root / PROVENANCE["path"]
    if not provenance_path.is_file() or provenance_identity.get("live_sha256") != file_sha256(provenance_path):
        raise RuntimeError("Target-swap manifest provenance JSON is stale")
    expected_sources = source_hashes(
        repository_root,
        ("athena/preflight_counterfactual_target_swap.py",),
    )
    if manifest.get("source_sha256_start") != expected_sources or manifest.get("source_sha256_end") != expected_sources:
        raise RuntimeError("Target-swap manifest source identities are stale")
    live_libero = validate_libero_runtime()
    if manifest.get("libero_runtime_start") != live_libero or manifest.get("libero_runtime_end") != live_libero:
        raise RuntimeError("Target-swap manifest LIBERO source identity is stale")
    if manifest.get("specificity_summary_sha256") != file_sha256(repository_root / SPECIFICITY["path"]):
        raise RuntimeError("Specificity summary changed after target-swap preflight")
    validate_specificity_summary(repository_root / SPECIFICITY["path"])
    rows = list(iter_mapping_rows(manifest))
    if len(rows) != 50:
        raise RuntimeError(f"Manifest has {len(rows)} mappings, expected 50")
    seen = set()
    for row in rows:
        task = int(row["task_index"])
        distractor = int(row["counterfactual_prompt_id"])
        key = (task, distractor)
        if key in seen:
            raise RuntimeError(f"Duplicate manifest mapping {key}")
        seen.add(key)
        original_path = Path(row["original_bddl_path"])
        rewritten_path = Path(row["rewritten_bddl_path"])
        if not original_path.is_file() or not rewritten_path.is_file():
            raise FileNotFoundError(f"Manifest BDDL path is absent for {key}")
        expected_name, expected_sha = ORIGINAL_BDDL[task]
        if original_path.name != expected_name or file_sha256(original_path) != expected_sha:
            raise RuntimeError(f"Original BDDL identity differs for task {task}")
        if row["original_bddl_sha256"] != expected_sha:
            raise RuntimeError(f"Manifest original BDDL SHA differs for task {task}")
        rewritten_sha = file_sha256(rewritten_path)
        if rewritten_sha != row["rewritten_bddl_sha256"]:
            raise RuntimeError(f"Rewritten BDDL SHA differs for {key}")
        expected_text, expected_metadata = rewrite_bddl(original_path.read_text(), task, distractor)
        if rewritten_path.read_text() != expected_text or row["rewrite_validation"] != expected_metadata:
            raise RuntimeError(f"Rewritten BDDL does not reproduce exactly for {key}")
        validate_hex_digest(rewritten_sha, f"rewritten BDDL {key}")
    expected = {
        (task, distractor)
        for task in range(10)
        for distractor in inspect_original_bddl(
            Path(next(row["original_bddl_path"] for row in rows if int(row["task_index"]) == task)).read_text(),
            task,
        )["distractor_prompt_ids"]
    }
    if seen != expected:
        raise RuntimeError("Manifest does not cover every task and co-present distractor")
    return manifest
