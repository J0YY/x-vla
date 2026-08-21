#!/usr/bin/env python3
"""Verify the frozen instruction-embedding-path ablation with the standard library."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import struct
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any


ARTIFACT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = ARTIFACT_DIR.parents[1]
CONDITIONS = ("full_lexical_embeddings", "zeroed_lexical_embeddings")
SHARDS = ((0, 2), (2, 4), (4, 6), (6, 8), (8, 10))
RAW_RE = re.compile(r"instruction_embedding_ablation_v1_s([0-2])_t(0|2|4|6|8)_(2|4|6|8|10)\.json")
EXPECTED_PACKAGE_MANIFEST_SHA256 = "f63ac60ecb0516ae310e8215507074b103bf09a3c48765cb6ea2c107505e0162"
EXPECTED_PREFLIGHT_SHA256 = "a91ce8bfc4b296e4a84503db63ad2b34f1c6509a9fb5f3fad74cbfd2bafb748d"
EXPECTED_ZERO_OUTPUT_SHA256 = "309575452b0ebda5b7cf9288f56e15ec5d3894b1c7faca010665ea58a6d29b24"
BOOTSTRAP_SEED = 2026082701
BOOTSTRAP_DRAWS = 20000
PCG64_INITIAL_STATE = 16709343118966810781669210904088756485
PCG64_INCREMENT = 158727911587790572230967259974399590507
PCG64_MULTIPLIER = 47026247687942121848144207491837523525
PCG64_FIRST_RAW = 15274225602273792823
EXPECTED_PROMPTS = (
    "pick up the alphabet soup and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the orange juice and place it in the basket",
)
EXPECTED_INSTRUCTION_IDS = (
    (1, 17, 25, 23, 2, 22, 3, 18, 12, 11, 23, 4),
    (1, 17, 25, 23, 9, 7, 3, 18, 12, 11, 23, 4),
    (1, 17, 25, 23, 20, 10, 3, 18, 12, 11, 23, 4),
    (1, 17, 25, 23, 5, 21, 3, 18, 12, 11, 23, 4),
    (1, 17, 25, 23, 14, 3, 18, 12, 11, 23, 4),
    (1, 17, 25, 23, 24, 21, 3, 18, 12, 11, 23, 4),
    (1, 17, 25, 23, 6, 3, 18, 12, 11, 23, 4),
    (1, 17, 25, 23, 15, 3, 18, 12, 11, 23, 4),
    (1, 17, 25, 23, 8, 19, 3, 18, 12, 11, 23, 4),
    (1, 17, 25, 23, 16, 13, 3, 18, 12, 11, 23, 4),
)
EXPECTED_INSTRUCTION_ID_SHA256 = (
    "9da4fd921a97a862ec003cc2667acd680e49f8b7b4ef76ed62503e1d679cf969",
    "ab12ea186efc54dc7e07324eaa0c4937708016a7bd889e67e0b719cadc3e4d42",
    "4b0f2a1cdc18214f06d0f29f96b32a2e79c4a1a3ecd72697ddf76766146aae20",
    "ebf52dd579dd5edbbdbdb080af6f34de5b83dc2971ca40fb8696858fbdeb89de",
    "2140993f7f55a85eab085b09d7794dccc3416b6f1892f014a90920c6e40ae0f7",
    "82d6be799b41e70f77351122d5d13cfa200db40feb73d63d3902c171b88ab4aa",
    "ff242138852207070d3d6948b6b25dd9573b53708011513bcc7af281a181309e",
    "0974c1b07f045b087f531c1e984dd673ed0e91160c551761181e558c2e27bc5b",
    "95d68b20e594bf6e8fd843f3a395139353863303ceecb0d9f58ab7c3c582eb66",
    "aa7811556d9d6eef4e2e4952f99e7f9d6014aaa4d86b3716425f6e14d420ba24",
)


class VerificationError(RuntimeError):
    """Raised when committed evidence differs from the frozen artifact."""


def require(value: bool, message: str) -> None:
    if not value:
        raise VerificationError(message)


def reject_constant(value: str) -> None:
    raise VerificationError(f"non-finite JSON constant {value}")


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path}: top-level value must be an object")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise VerificationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def product(values: list[int]) -> int:
    total = 1
    for value in values:
        require(type(value) is int and value >= 0, "array shape is malformed")
        total *= value
    return total


def decode_array(value: dict[str, Any]) -> tuple[list[float | bool], list[int]]:
    require(value.get("codec") == "zlib+base64", "array codec differs")
    shape = value.get("shape")
    require(isinstance(shape, list), "array shape is absent")
    try:
        raw = zlib.decompress(base64.b64decode(value["data"], validate=True))
    except Exception as exc:
        raise VerificationError(f"cannot decode array: {exc}") from exc
    dtype = value.get("dtype")
    formats = {"<f8": ("<d", 8), "<f4": ("<f", 4), "|b1": ("<?", 1)}
    require(dtype in formats, f"unsupported array dtype {dtype!r}")
    fmt, width = formats[dtype]
    count = product(shape)
    require(len(raw) == count * width, "array byte count differs from shape")
    digest = hashlib.sha256()
    digest.update(dtype.encode("ascii"))
    digest.update(struct.pack(f"<{len(shape)}q", *shape))
    digest.update(raw)
    require(digest.hexdigest() == value.get("sha256"), "array digest differs")
    return [item[0] for item in struct.iter_unpack(fmt, raw)], shape


def validate_rollout(rollout: dict[str, Any]) -> bool:
    steps = rollout.get("steps")
    require(type(steps) is int and 1 <= steps <= 280, "rollout step count differs")
    actions, action_shape = decode_array(rollout["executed_actions"])
    rewards, reward_shape = decode_array(rollout["rewards"])
    dones, done_shape = decode_array(rollout["dones"])
    require(action_shape == [steps, 7], "executed-action shape differs")
    require(reward_shape == [steps] and done_shape == [steps], "reward or done shape differs")
    require(all(math.isfinite(float(x)) for x in actions + rewards), "non-finite trajectory value")
    require(all(type(x) is bool for x in dones), "done array is not boolean")
    success = any(float(reward) > 0.0 for reward in rewards)
    require(rollout.get("success") is success, "success does not reproduce from rewards")
    success_step = next((index + 1 for index, reward in enumerate(rewards) if float(reward) > 0.0), None)
    require(rollout.get("success_step") == success_step, "success step does not reproduce")

    rebuilt: list[float] = []
    expected_start = 0
    for chunk in rollout.get("chunks", []):
        require(chunk.get("start_step") == expected_start, "action chunks are not contiguous")
        prediction, shape = decode_array(chunk["prediction"])
        require(shape == [8, 7], "action-chunk shape differs")
        rows = chunk.get("executed_rows")
        require(type(rows) is int and 1 <= rows <= 8, "executed chunk-row count differs")
        for row in range(rows):
            values = [float(x) for x in prediction[row * 7 : (row + 1) * 7]]
            values[-1] = 1.0 if values[-1] > 0.0 else -1.0
            rebuilt.extend(values)
        expected_start += rows
    require(expected_start == steps, "chunks do not cover the trajectory")
    require(rebuilt == [float(x) for x in actions], "executed actions do not reproduce from chunks")
    return success


def exact_one_sided_p(full_only: int, zeroed_only: int) -> float:
    discordant = full_only + zeroed_only
    return sum(math.comb(discordant, k) for k in range(full_only, discordant + 1)) / (2**discordant)


def counts(records: list[dict[str, int | bool]]) -> dict[str, float | int]:
    full = sum(int(row["full"]) for row in records)
    zeroed = sum(int(row["zeroed"]) for row in records)
    full_only = sum(int(bool(row["full"]) and not bool(row["zeroed"])) for row in records)
    zeroed_only = sum(int(bool(row["zeroed"]) and not bool(row["full"])) for row in records)
    n = len(records)
    return {
        "pairs": n,
        "full_successes": full,
        "zeroed_successes": zeroed,
        "full_rate": full / n,
        "zeroed_rate": zeroed / n,
        "gap": (full - zeroed) / n,
        "full_only": full_only,
        "zeroed_only": zeroed_only,
        "one_sided_exact_p": exact_one_sided_p(full_only, zeroed_only),
    }


def close(actual: Any, expected: Any) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-15)


class FrozenPCG64:
    """Minimal NumPy-1.26 PCG64 stream for the prospectively frozen seed."""

    def __init__(self, seed: int) -> None:
        require(seed == BOOTSTRAP_SEED, "unsupported bootstrap seed")
        self.state = PCG64_INITIAL_STATE
        self.has_uint32 = False
        self.buffered_uint32 = 0

    def random_raw(self) -> int:
        mask128 = (1 << 128) - 1
        mask64 = (1 << 64) - 1
        self.state = (self.state * PCG64_MULTIPLIER + PCG64_INCREMENT) & mask128
        xorshifted = ((self.state >> 64) ^ self.state) & mask64
        rotation = self.state >> 122
        return ((xorshifted >> rotation) | (xorshifted << ((-rotation) & 63))) & mask64

    def uint32(self) -> int:
        if self.has_uint32:
            self.has_uint32 = False
            return self.buffered_uint32
        word = self.random_raw()
        self.buffered_uint32 = word >> 32
        self.has_uint32 = True
        return word & ((1 << 32) - 1)

    def bounded_uint32(self, upper_exclusive: int) -> int:
        require(type(upper_exclusive) is int and 0 < upper_exclusive <= 1 << 32, "invalid bounded integer range")
        mask32 = (1 << 32) - 1
        threshold = ((1 << 32) - upper_exclusive) % upper_exclusive
        while True:
            product_value = self.uint32() * upper_exclusive
            if product_value & mask32 >= threshold:
                return product_value >> 32


def linear_quantile(values: list[float], probability: float) -> float:
    require(values and 0.0 <= probability <= 1.0, "invalid quantile input")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def task_stratified_state_cluster_bootstrap(
    records: list[dict[str, int | bool]], draws: int, seed: int
) -> list[float]:
    require(draws == BOOTSTRAP_DRAWS, "bootstrap draw count differs")
    by_state: dict[tuple[int, int], list[dict[str, int | bool]]] = defaultdict(list)
    for row in records:
        by_state[(int(row["task"]), int(row["episode"]))].append(row)
    expected_states = {(task, episode) for task in range(10) for episode in range(30, 40)}
    require(set(by_state) == expected_states, "bootstrap state matrix differs")
    require(all(len(rows) == 3 for rows in by_state.values()), "bootstrap checkpoint clusters differ")
    stream_check = FrozenPCG64(seed)
    require(stream_check.random_raw() == PCG64_FIRST_RAW, "frozen PCG64 stream differs")
    rng = FrozenPCG64(seed)
    means: list[float] = []
    for _ in range(draws):
        total = 0
        count = 0
        for task in range(10):
            for _ in range(10):
                episode = 30 + rng.bounded_uint32(10)
                for row in by_state[(task, episode)]:
                    total += int(row["full"]) - int(row["zeroed"])
                    count += 1
        require(count == 300, "bootstrap draw did not retain 300 checkpoint outcomes")
        means.append(total / count)
    return [linear_quantile(means, 0.025), linear_quantile(means, 0.975)]


def expected_instruction_ids(task: int) -> list[int]:
    return [*EXPECTED_INSTRUCTION_IDS[task], *([0] * (32 - len(EXPECTED_INSTRUCTION_IDS[task])))]


def expected_condition_order(seed: int, task: int, episode: int) -> list[str]:
    payload = f"instruction-embedding-ablation-v1-order|{seed}|{task}|{episode}".encode()
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2
    return list(CONDITIONS if index == 0 else CONDITIONS[::-1])


def validate_identity(
    identity: dict[str, Any],
    seed: int,
    preflight: dict[str, Any],
    label: str,
) -> tuple[str, dict[str, str]]:
    checkpoint = preflight["checkpoints"][str(seed)]
    cache = preflight["cache"]
    provenance = preflight["provenance"]
    frozen_source = preflight["source_sha256_start"]
    require(identity.get("checkpoint_seed") == seed, f"{label}: checkpoint seed differs")
    require(identity.get("checkpoint") == checkpoint["path"], f"{label}: checkpoint path differs")
    require(identity.get("checkpoint_sha256") == checkpoint["sha256"], f"{label}: checkpoint SHA differs")
    require(identity.get("cache") == cache["path"], f"{label}: cache path differs")
    require(identity.get("cache_sha256") == cache["sha256"] == cache["live_sha256"], f"{label}: cache SHA differs")
    require(identity.get("manifest") == "artifacts/instruction_embedding_ablation_v1_manifest.json", f"{label}: manifest path differs")
    require(identity.get("manifest_sha256") == EXPECTED_PREFLIGHT_SHA256, f"{label}: manifest SHA differs")
    require(identity.get("provenance_job_id") == provenance["job_id"], f"{label}: provenance job differs")
    require(identity.get("provenance_result") == provenance["path"], f"{label}: provenance path differs")
    require(identity.get("provenance_result_sha256") == provenance["live_sha256"], f"{label}: provenance SHA differs")
    require(identity.get("source_sha256_start") == frozen_source, f"{label}: source start differs from preflight")
    require(identity.get("source_sha256_end") == frozen_source, f"{label}: source end differs from preflight")
    model_state = identity.get("model_state_sha256_start")
    require(isinstance(model_state, str) and re.fullmatch(r"[0-9a-f]{64}", model_state) is not None, f"{label}: model-state SHA is malformed")
    require(identity.get("model_state_sha256_end") == model_state, f"{label}: model state changed")
    protected = identity.get("protected_component_sha256_start")
    require(
        isinstance(protected, dict)
        and set(protected)
        == {
            "action_queries",
            "joint_position_embeddings",
            "learned_common_bos",
            "state_projection_bias",
            "state_projection_weight",
            "token_embedding_weights",
            "vision_position_embeddings",
        }
        and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None for value in protected.values()),
        f"{label}: protected-component identity is malformed",
    )
    require(identity.get("protected_component_sha256_end") == protected, f"{label}: protected model component changed")
    return model_state, protected


def validate_condition_metadata(
    value: dict[str, Any],
    mapping: dict[str, Any],
    task: int,
    label: str,
    rollout: dict[str, Any] | None = None,
) -> None:
    ids = expected_instruction_ids(task)
    ids_sha = EXPECTED_INSTRUCTION_ID_SHA256[task]
    require(value.get("prompt_id") == task, f"{label}: prompt ID differs")
    require(value.get("prompt_text") == EXPECTED_PROMPTS[task] == mapping["task_language"], f"{label}: prompt text differs")
    require(value.get("instruction_ids") == ids, f"{label}: instruction IDs differ")
    contract = value.get("input_contract")
    require(isinstance(contract, dict), f"{label}: input contract is absent")
    for field in (
        "action_queries_preserved",
        "common_learned_bos_preserved",
        "embodiment_path_preserved",
        "positions_preserved",
        "sequence_length_preserved",
        "state_path_preserved",
        "vision_path_preserved",
    ):
        require(contract.get(field) is True, f"{label}: {field} is not preserved")
    require(contract.get("instruction_id_shape") == [1, 32], f"{label}: instruction-ID shape differs")
    require(contract.get("instruction_ids_sha256") == ids_sha, f"{label}: instruction-ID SHA differs")
    require(contract.get("model_image_sha256") == mapping["settled_model_image_sha256"], f"{label}: model image differs from preflight")
    require(contract.get("model_robot_state_sha256") == mapping["settled_model_state_sha256"], f"{label}: model state differs from preflight")
    require(contract.get("physical_input_sha256") == mapping["settled_physical_input_sha256"], f"{label}: physical input differs from preflight")
    if rollout is None:
        return
    require(rollout.get("prompt_id") == task, f"{label}: rollout prompt ID differs")
    require(rollout.get("instruction_ids") == ids, f"{label}: rollout instruction IDs differ")
    start_physical = rollout.get("start_physical")
    require(isinstance(start_physical, dict), f"{label}: start-physical record is absent")
    components = start_physical.get("component_sha256")
    require(isinstance(components, dict), f"{label}: start components are absent")
    require(components.get("model_image") == mapping["settled_model_image_sha256"], f"{label}: rollout image differs from preflight")
    require(components.get("model_robot_state") == mapping["settled_model_state_sha256"], f"{label}: rollout state differs from preflight")
    require(components.get("mujoco_state") == mapping["settled_component_sha256"]["simulator_state"], f"{label}: simulator state differs from preflight")


def validate_hook_audit(
    audit: dict[str, Any],
    condition: str,
    expected_calls: int,
    ids_sha: str,
    protected: dict[str, str],
    label: str,
) -> None:
    require(audit.get("preexisting_hook_count") == 0 and audit.get("post_context_hook_count") == 0, f"{label}: hook lifecycle differs")
    require(audit.get("protected_component_sha256_before") == protected, f"{label}: pre-hook protected identity differs")
    require(audit.get("protected_component_sha256_after") == protected, f"{label}: post-hook protected identity differs")
    calls = audit.get("calls")
    require(isinstance(calls, list), f"{label}: hook call records are absent")
    if condition == CONDITIONS[0]:
        require(audit.get("enabled") is False, f"{label}: full-path hook was enabled")
        require(audit.get("mechanism") == "deployed_model_unmodified", f"{label}: full-path mechanism differs")
        require(audit.get("hook_calls") == 0 and calls == [], f"{label}: full-path hook calls differ")
        require(audit.get("all_returned_outputs_exactly_zero") is None, f"{label}: full-path zero-output flag differs")
        return

    require(audit.get("enabled") is True, f"{label}: ablation hook was not enabled")
    require(
        audit.get("mechanism") == "scoped_post_lookup_forward_hook_returning_torch_zeros_like",
        f"{label}: ablation mechanism differs",
    )
    require(audit.get("hook_calls") == expected_calls, f"{label}: ablation hook count differs")
    require(len(calls) == expected_calls, f"{label}: ablation call-record count differs")
    require(audit.get("instruction_token_positions_zeroed_per_call") == 32, f"{label}: zeroed-position count differs")
    require(audit.get("all_returned_outputs_exactly_zero") is True, f"{label}: ablation output was not exactly zero")
    pre_output_identity: tuple[str, float] | None = None
    for index, call in enumerate(calls):
        require(call.get("call_index") == index, f"{label}: hook call index differs")
        require(call.get("input_ids_sha256") == ids_sha, f"{label}: hook input-ID SHA differs")
        require(call.get("input_ids_shape") == [1, 32], f"{label}: hook input-ID shape differs")
        require(call.get("pre_ablation_output_dtype") == "torch.float32", f"{label}: pre-ablation dtype differs")
        require(call.get("pre_ablation_output_shape") == [1, 32, 384], f"{label}: pre-ablation shape differs")
        pre_sha = call.get("pre_ablation_output_sha256")
        pre_max = call.get("pre_ablation_output_max_abs")
        require(isinstance(pre_sha, str) and re.fullmatch(r"[0-9a-f]{64}", pre_sha) is not None, f"{label}: pre-ablation output SHA is malformed")
        require(isinstance(pre_max, (int, float)) and math.isfinite(float(pre_max)) and float(pre_max) > 0.0, f"{label}: pre-ablation output magnitude differs")
        if pre_output_identity is None:
            pre_output_identity = (pre_sha, float(pre_max))
        require((pre_sha, float(pre_max)) == pre_output_identity, f"{label}: learned lookup output changed between calls")
        require(call.get("zeroed_token_positions") == 32, f"{label}: call did not zero 32 positions")
        require(call.get("returned_nonzero_elements") == 0, f"{label}: returned output contains nonzeros")
        require(call.get("returned_output_max_abs") == 0.0, f"{label}: returned output magnitude is nonzero")
        require(call.get("returned_output_sha256") == EXPECTED_ZERO_OUTPUT_SHA256, f"{label}: returned zero-tensor SHA differs")


def verify(root: Path) -> dict[str, Any]:
    require(
        file_sha256(ARTIFACT_DIR / "manifest.json") == EXPECTED_PACKAGE_MANIFEST_SHA256,
        "artifact trust-root manifest differs",
    )
    package = load_json(ARTIFACT_DIR / "manifest.json")
    require(package.get("schema") == "anonymous-instruction-embedding-ablation-artifact-v1", "artifact schema differs")
    files = package.get("files")
    require(isinstance(files, dict) and len(files) == 18, "artifact file manifest differs")
    for relative, digest in files.items():
        require(".." not in Path(relative).parts and not Path(relative).is_absolute(), "unsafe artifact path")
        require(file_sha256(root / relative) == digest, f"file digest differs: {relative}")

    preflight_path = root / "athena/results/instruction_embedding_ablation_v1_manifest.json"
    require(file_sha256(preflight_path) == EXPECTED_PREFLIGHT_SHA256, "preflight trust-root digest differs")
    preflight = load_json(preflight_path)
    smoke = load_json(root / "athena/results/instruction_embedding_ablation_v1_smoke.json")
    summary = load_json(root / "athena/results/instruction_embedding_ablation_v1_summary.json")
    require(preflight.get("schema") == "xvla-instruction-embedding-ablation-manifest-v1", "preflight schema differs")
    require(preflight.get("mapping_count") == 100 and len(preflight.get("mappings", [])) == 100, "preflight matrix differs")
    require(preflight.get("source_sha256_start") == preflight.get("source_sha256_end"), "preflight source closure changed")
    require(preflight.get("provenance", {}).get("verified") is True, "preflight cache provenance was not verified")
    mappings = {(row.get("task_index"), row.get("episode")): row for row in preflight["mappings"]}
    require(len(mappings) == 100, "preflight contains duplicate state keys")
    require(set(mappings) == {(task, episode) for task in range(10) for episode in range(30, 40)}, "preflight state keys differ")
    for (task, episode), mapping in mappings.items():
        require(mapping.get("task_language") == EXPECTED_PROMPTS[task], "preflight task language differs")
        require(mapping.get("instruction_token_count") == 32, "preflight instruction-token count differs")
        require(
            mapping.get("init_state_index") == episode and mapping.get("reset_seed") == 100 * task + episode,
            "preflight state index differs",
        )

    protocol = preflight.get("protocol")
    gates_spec = preflight.get("frozen_gates")
    require(smoke.get("schema") == "xvla-instruction-embedding-ablation-run-v1" and smoke.get("mode") == "strict_smoke", "smoke identity differs")
    require(smoke.get("protocol") == protocol and smoke.get("frozen_gates") == gates_spec, "smoke protocol or gates differ")
    smoke_model_identity = validate_identity(smoke.get("identity", {}), 0, preflight, "smoke")
    require(len(smoke.get("smoke_rows", [])) == 10, "smoke task coverage differs")
    smoke_tasks: set[int] = set()
    for row in smoke["smoke_rows"]:
        task, episode = row.get("task_index"), row.get("episode")
        require(type(task) is int and task not in smoke_tasks and 0 <= task < 10 and episode == 30, "smoke state coverage differs")
        smoke_tasks.add(task)
        mapping = mappings[(task, episode)]
        require(row.get("physical_input_sha256") == mapping["settled_physical_input_sha256"], "smoke physical input differs from preflight")
        require(row.get("condition_order") == expected_condition_order(0, task, episode), "smoke condition order differs")
        require(set(row.get("conditions", {})) == set(CONDITIONS), "smoke condition set differs")
        for condition, value in row["conditions"].items():
            label = f"smoke task {task} {condition}"
            validate_condition_metadata(value, mapping, task, label)
            action_chunk, action_shape = decode_array(value["action_chunk"])
            require(action_shape == [8, 7] and all(math.isfinite(float(x)) for x in action_chunk), f"{label}: action chunk differs")
            audit = value.get("ablation_audit", {})
            validate_hook_audit(audit, condition, 1, EXPECTED_INSTRUCTION_ID_SHA256[task], smoke_model_identity[1], label)

    raw_paths = [root / path for path in files if RAW_RE.fullmatch(Path(path).name)]
    require(len(raw_paths) == 15, "raw shard count differs")
    records: list[dict[str, int | bool]] = []
    observed: set[tuple[int, int, int]] = set()
    model_identity_by_seed: dict[int, tuple[str, dict[str, str]]] = {}
    for path in sorted(raw_paths):
        match = RAW_RE.fullmatch(path.name)
        require(match is not None, f"raw filename differs: {path.name}")
        seed, start, end = map(int, match.groups())
        require((start, end) in SHARDS, f"raw shard bounds differ: {path.name}")
        result = load_json(path)
        require(result.get("schema") == "xvla-instruction-embedding-ablation-run-v1" and result.get("mode") == "full", "raw identity differs")
        require(result.get("protocol") == protocol and result.get("frozen_gates") == gates_spec, "raw protocol or gates differ")
        identity = result.get("identity", {})
        model_identity = validate_identity(identity, seed, preflight, path.name)
        if seed in model_identity_by_seed:
            require(model_identity_by_seed[seed] == model_identity, f"{path.name}: model identity differs across shards")
        else:
            model_identity_by_seed[seed] = model_identity
        if seed == 0:
            require(model_identity == smoke_model_identity, f"{path.name}: model identity differs from smoke")
        evaluation = result.get("evaluation", {})
        require((evaluation.get("task_start"), evaluation.get("task_end")) == (start, end), "raw evaluation bounds differ")
        require(evaluation.get("episode_start") == 30 and evaluation.get("eps_per_task") == 10, "raw episode partition differs")
        rows = result.get("checkpoint_state_pairs", [])
        require(len(rows) == 20, "raw shard row count differs")
        for row in rows:
            task, episode = row.get("task_index"), row.get("episode")
            key = (seed, task, episode)
            require(type(task) is int and type(episode) is int, "raw state key is malformed")
            require(key not in observed and start <= task < end and 30 <= episode < 40, "duplicate or out-of-range row")
            observed.add(key)
            mapping = mappings[(task, episode)]
            require(row.get("condition_order") == expected_condition_order(seed, task, episode), "raw condition order differs")
            require(row.get("init_state_index") == mapping["init_state_index"], "raw init-state index differs from preflight")
            require(row.get("init_state_sha256") == mapping["init_state_sha256"], "raw init-state SHA differs from preflight")
            require(row.get("original_bddl_sha256") == mapping["original_bddl_sha256"], "raw BDDL SHA differs from preflight")
            require(row.get("physical_input_sha256") == mapping["settled_physical_input_sha256"], "raw physical input differs from preflight")
            require(row.get("reset_seed") == mapping["reset_seed"], "raw reset seed differs from preflight")
            require(set(row.get("conditions", {})) == set(CONDITIONS), "raw condition set differs")
            starts: list[Any] = []
            contracts: list[Any] = []
            prompts: list[Any] = []
            outcome: dict[str, int | bool] = {"seed": seed, "task": task, "episode": episode}
            for condition, value in row["conditions"].items():
                rollout = value.get("rollout", {})
                audit = value.get("ablation_audit", {})
                label = f"seed {seed} task {task} episode {episode} {condition}"
                validate_condition_metadata(value, mapping, task, label, rollout)
                starts.append(rollout.get("start_physical"))
                contracts.append(value.get("input_contract"))
                prompts.append((value.get("prompt_id"), value.get("prompt_text"), value.get("instruction_ids")))
                validate_hook_audit(
                    audit,
                    condition,
                    len(rollout.get("chunks", [])),
                    EXPECTED_INSTRUCTION_ID_SHA256[task],
                    model_identity[1],
                    label,
                )
                outcome["full" if condition == CONDITIONS[0] else "zeroed"] = validate_rollout(rollout)
            require(starts[0] == starts[1] and contracts[0] == contracts[1] and prompts[0] == prompts[1], "paired inputs differ")
            records.append(outcome)

    expected_keys = {(seed, task, episode) for seed in range(3) for task in range(10) for episode in range(30, 40)}
    require(observed == expected_keys and len(records) == 300, "complete paired matrix is absent")
    pooled = counts(records)
    by_checkpoint = [counts([row for row in records if row["seed"] == seed]) for seed in range(3)]
    by_task = [counts([row for row in records if row["task"] == task]) for task in range(10)]
    expected = package["expected"]
    require(pooled["full_successes"] == expected["full_successes"], "full success total differs")
    require(pooled["zeroed_successes"] == expected["zeroed_successes"], "zeroed success total differs")
    require(pooled["full_only"] == expected["full_only"] and pooled["zeroed_only"] == expected["zeroed_only"], "discordance totals differ")
    require(close(pooled["gap"], expected["paired_gap"]), "paired gap differs")
    require(close(pooled["one_sided_exact_p"], expected["one_sided_exact_p"]), "exact paired p-value differs")
    require([row["full_successes"] for row in by_checkpoint] == expected["checkpoint_full_successes"], "checkpoint full totals differ")
    require([row["zeroed_successes"] for row in by_checkpoint] == expected["checkpoint_zeroed_successes"], "checkpoint zeroed totals differ")
    require(all(close(row["gap"], value) for row, value in zip(by_task, expected["task_gaps"])), "task gaps differ")

    by_state: dict[tuple[int, int], list[dict[str, int | bool]]] = defaultdict(list)
    for row in records:
        by_state[(int(row["task"]), int(row["episode"]))].append(row)
    task_support_minima = []
    for task in range(10):
        state_sums = [sum(int(row["full"]) - int(row["zeroed"]) for row in by_state[(task, episode)]) for episode in range(30, 40)]
        task_support_minima.append(min(state_sums))
    support_lower = sum(10 * value for value in task_support_minima) / 300
    require(close(support_lower, expected["bootstrap_support_lower_bound"]), "bootstrap support lower bound differs")
    require(support_lower > 0.0, "bootstrap support permits a nonpositive draw")
    require(gates_spec.get("task_stratified_state_cluster_bootstrap_draws") == BOOTSTRAP_DRAWS, "frozen bootstrap draws differ")
    require(gates_spec.get("task_stratified_state_cluster_bootstrap_seed") == BOOTSTRAP_SEED, "frozen bootstrap seed differs")
    bootstrap_interval = task_stratified_state_cluster_bootstrap(records, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    require(
        len(expected["bootstrap_interval"]) == 2
        and all(close(a, b) for a, b in zip(bootstrap_interval, expected["bootstrap_interval"])),
        "recomputed bootstrap interval differs",
    )

    require(summary.get("schema") == "xvla-instruction-embedding-ablation-summary-v1", "summary schema differs")
    require(summary.get("identity_validated") is True and summary.get("overall_pass") is True and summary.get("claim_eligible") is True, "strict summary did not pass")
    require(all(summary.get("gates", {}).values()) and len(summary.get("gates", {})) == 8, "summary gate set differs")
    summary_interval = summary.get("task_stratified_state_cluster_bootstrap_interval", [])
    require(
        len(summary_interval) == 2 and all(close(a, b) for a, b in zip(summary_interval, bootstrap_interval)),
        "summary bootstrap interval differs",
    )
    require(close(summary["pooled"]["paired_success_gap"], pooled["gap"]), "summary pooled gap differs")
    require(summary["pooled"]["full_only"] == pooled["full_only"] and summary["pooled"]["zeroed_only"] == pooled["zeroed_only"], "summary discordances differ")

    gates = {
        "full_pooled_at_least_70_percent": float(pooled["full_rate"]) >= 0.70,
        "full_every_checkpoint_at_least_60_percent": all(float(row["full_rate"]) >= 0.60 for row in by_checkpoint),
        "pooled_gap_at_least_20_points": float(pooled["gap"]) >= 0.20,
        "gap_positive_every_checkpoint": all(float(row["gap"]) > 0.0 for row in by_checkpoint),
        "one_sided_exact_p_at_most_0p01": float(pooled["one_sided_exact_p"]) <= 0.01,
        "at_least_eight_positive_tasks": sum(float(row["gap"]) > 0.0 for row in by_task) >= 8,
        "no_negative_task": all(float(row["gap"]) >= 0.0 for row in by_task),
        "bootstrap_lower_above_zero": bootstrap_interval[0] > 0.0 and support_lower > 0.0,
    }
    require(all(gates.values()), "a recomputed frozen gate failed")
    return {
        "verified": True,
        "pooled": pooled,
        "by_checkpoint": by_checkpoint,
        "by_task": by_task,
        "bootstrap_interval": bootstrap_interval,
        "bootstrap_support_lower_bound": support_lower,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(args.root.resolve())
    except VerificationError as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        pooled = result["pooled"]
        print("Instruction-embedding-path ablation artifact: VERIFIED")
        print(f"  full path: {pooled['full_successes']}/300 ({100 * pooled['full_rate']:.1f}%)")
        print(f"  zeroed path: {pooled['zeroed_successes']}/300 ({100 * pooled['zeroed_rate']:.1f}%)")
        print(f"  paired gap: {100 * pooled['gap']:.1f} percentage points")
        print(f"  one-sided exact p: {pooled['one_sided_exact_p']:.3e}")
        low, high = result["bootstrap_interval"]
        print(f"  state-cluster bootstrap: [{100 * low:.1f}, {100 * high:.1f}] points")
        print("  frozen gates: 8/8 passed")


if __name__ == "__main__":
    main()
