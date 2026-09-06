from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import pytest

from scripts import run_capable_linear_rtx6000_capability as lane


def _valid_observation() -> dict[str, object]:
    return {
        "agentview_image": lane.np.zeros(
            (lane.RESOLUTION, lane.RESOLUTION, 3), dtype=lane.np.uint8
        ),
        "robot0_eef_pos": lane.np.zeros(3, dtype=lane.np.float64),
        "robot0_eef_quat": lane.np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=lane.np.float64
        ),
        "robot0_gripper_qpos": lane.np.zeros(2, dtype=lane.np.float64),
    }


def _parse(monkeypatch: pytest.MonkeyPatch, *arguments: str):
    monkeypatch.setattr(sys, "argv", ["runner", *arguments])
    return lane._parse_args()


def test_duplicate_cli_flags_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        _parse(
            monkeypatch,
            "--mode",
            "shard",
            "--mode",
            "shard",
            "--task-start",
            "0",
            "--task-end",
            "3",
        )


def test_arbitrary_shard_range_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        _parse(
            monkeypatch,
            "--mode",
            "shard",
            "--task-start",
            "0",
            "--task-end",
            "2",
        )


@pytest.mark.parametrize("mode", ["smoke", "aggregate"])
def test_nonshard_modes_reject_ranges(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    with pytest.raises(SystemExit):
        _parse(
            monkeypatch,
            "--mode",
            mode,
            "--task-start",
            "0",
            "--task-end",
            "1",
        )


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"value": 1, "value": 2}\n')
    with pytest.raises(RuntimeError, match="duplicate key"):
        lane._read_object(path, "fixture")


def test_publication_is_exclusive_physical_and_read_only(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    digest = lane._publish(output, {"ok": True})
    assert lane._sha256(output) == digest
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        lane._publish(output, {"ok": False})
    link = tmp_path / "link.json"
    link.symlink_to(tmp_path / "missing.json")
    with pytest.raises(FileExistsError):
        lane._publish(link, {"ok": True})


def test_episode_inventory_rejects_incomplete_and_duplicate_pairs() -> None:
    records = [
        {
            "task_index": task,
            "episode": episode,
            "initial_state_sha256": f"{task * 50 + episode:064x}",
            "success": False,
            "steps": 280,
            "terminated_without_success": False,
            "elapsed_s": 0.1,
        }
        for task in range(10)
        for episode in range(50)
    ]
    assert lane._episode_inventory_is_exact(records, 0, 10, 50, 280)
    assert not lane._episode_inventory_is_exact(records[:-1], 0, 10, 50, 280)
    duplicated = list(records)
    duplicated[-1] = dict(duplicated[-2])
    assert not lane._episode_inventory_is_exact(duplicated, 0, 10, 50, 280)


@pytest.mark.parametrize(
    "mutation",
    ("zero_steps", "early_nonterminal_stop", "extra_field", "elapsed_int_alias"),
)
def test_episode_inventory_rejects_impossible_step_semantics(mutation: str) -> None:
    records = [
        {
            "task_index": task,
            "episode": episode,
            "initial_state_sha256": f"{task * 50 + episode:064x}",
            "success": False,
            "steps": 280,
            "terminated_without_success": False,
            "elapsed_s": 0.1,
        }
        for task in range(10)
        for episode in range(50)
    ]
    if mutation == "extra_field":
        records[0]["unexpected"] = None
    elif mutation == "elapsed_int_alias":
        records[0]["elapsed_s"] = 0
    else:
        records[0]["steps"] = 0 if mutation == "zero_steps" else 279
    assert not lane._episode_inventory_is_exact(records, 0, 10, 50, 280)


def test_initial_state_row_bundle_and_episode_digest_reject_row_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            "task_index": 0,
            "episode": episode,
            "initial_state_sha256": f"{episode:064x}",
            "success": False,
            "steps": 280,
            "terminated_without_success": False,
            "elapsed_s": 0.1,
        }
        for episode in range(50)
    ]
    expected_bundle = lane._canonical_sha256(
        [record["initial_state_sha256"] for record in records]
    )
    monkeypatch.setitem(
        lane.EXPECTED_INITIAL_STATE_ROW_BUNDLE_SHA256, 0, expected_bundle
    )
    assert lane._initial_state_row_bundles_are_exact(records, 0, 1, 50)
    identity_digest = lane._canonical_sha256(lane._episode_identity_rows(records))
    records[0]["initial_state_sha256"] = "f" * 64
    assert not lane._initial_state_row_bundles_are_exact(records, 0, 1, 50)
    assert lane._canonical_sha256(lane._episode_identity_rows(records)) != identity_digest


def test_capability_nested_record_schemas_reject_unknown_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_expected_environment(monkeypatch)
    environment = lane._numerical_environment(require_gpu=True)
    assert lane._environment_record_is_exact(environment)
    environment["unexpected"] = None
    assert not lane._environment_record_is_exact(environment)

    sentinel = {
        "site": "vision.norm_out",
        "armed_during_sentinel_replay_and_all_rollout_policy_forwards": True,
        "call_count": 0,
        "total_site_count": 74,
        "inactive_sites": ["vision.norm_out"],
        "active_site_count": 73,
        "active_call_count": 73,
        "every_active_site_call_count": 1,
        "source_output_bitwise_equal_without_sentinels": True,
    }
    assert lane._inactive_norm_sentinel_is_exact(sentinel)
    sentinel["call_count"] = False
    assert not lane._inactive_norm_sentinel_is_exact(sentinel)
    sentinel["call_count"] = 0
    sentinel["unexpected"] = None
    assert not lane._inactive_norm_sentinel_is_exact(sentinel)


@pytest.mark.parametrize(
    ("path", "alias"),
    (
        (("resolution",), 64.0),
        (("settle_success_or_done_count",), False),
        (("episodes_per_task",), True),
        (("canonical_init_states",), 1),
    ),
)
def test_rollout_protocol_rejects_numeric_boolean_aliases(
    path: tuple[str, ...], alias: object
) -> None:
    protocol = lane._expected_rollout_protocol(1, 1)
    assert lane._rollout_protocol_is_exact(protocol, 1, 1)
    protocol[path[0]] = alias
    assert not lane._rollout_protocol_is_exact(protocol, 1, 1)


def test_recursive_authority_comparison_is_type_exact() -> None:
    authority = {"zero": 0, "one": 1, "ratio": 1.0, "flag": True}
    assert lane._type_exact_equal(authority, dict(authority))
    for field, alias in (
        ("zero", False),
        ("one", True),
        ("ratio", 1),
        ("flag", 1),
    ):
        mutated = dict(authority)
        mutated[field] = alias
        assert not lane._type_exact_equal(mutated, authority)


@pytest.mark.parametrize("field", ("settle", "policy"))
def test_transition_counts_reject_boolean_alias(field: str) -> None:
    records = [
        {
            "task_index": 0,
            "episode": 0,
            "initial_state_sha256": "0" * 64,
            "success": True,
            "steps": 1,
            "terminated_without_success": False,
            "elapsed_s": 0.1,
        }
    ]
    counts = {"settle": 10, "policy": 1}
    assert lane._transition_counts_are_exact(counts, records, expected_settle=10)
    counts[field] = True if counts[field] == 1 else False
    assert not lane._transition_counts_are_exact(
        counts, records, expected_settle=10
    )


def test_task_protocol_detects_digest_drift() -> None:
    expected = lane._expected_task_record(0)
    drifted = dict(expected)
    drifted["init_states_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="simulator protocol differs"):
        lane._require_expected_task_record(0, drifted)


def test_capability_config_identity_is_fully_explicit() -> None:
    config = lane._config()
    assert lane._config_identity(config) == lane.EXPECTED_CONFIG_IDENTITY
    config.vit_residual = None
    assert lane._config_identity(config) != lane.EXPECTED_CONFIG_IDENTITY


def test_checkpoint_tensor_shapes_match_per_token_linear_head_contract() -> None:
    expected = {
        "tok_emb.weight": (26, 384),
        "action_head.weight": (7, 384),
        "action_head.bias": (7,),
    }
    assert lane.EXPECTED_CHECKPOINT_TENSOR_SHAPES == expected
    assert lane.EXPECTED_PARAMETER_COUNT == 20_137_352


def test_inactive_norm_sentinel_raises_on_dead_site_execution() -> None:
    class ToyModel(lane.torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision = lane.torch.nn.Module()
            self.vision.norm_out = lane.torch.nn.Identity()

    model = ToyModel()
    counter, handle = lane._arm_inactive_vision_norm_sentinel(model)
    try:
        assert counter == {"calls": 0}
        lane.torch.nn.Identity()(lane.torch.ones(1))
        assert counter == {"calls": 0}
        with pytest.raises(
            RuntimeError, match="inactive vision normalization site was executed"
        ):
            model.vision.norm_out(lane.torch.ones(1))
        assert counter == {"calls": 1}
    finally:
        handle.remove()


def _install_expected_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        lane.platform, "python_version", lambda: lane.EXPECTED_VERSIONS["python"]
    )
    monkeypatch.setattr(lane.np, "__version__", lane.EXPECTED_VERSIONS["numpy"])
    monkeypatch.setattr(lane.torch, "__version__", lane.EXPECTED_VERSIONS["torch"])
    monkeypatch.setattr(
        lane,
        "_installed_version",
        lambda name: lane.EXPECTED_VERSIONS[name],
    )
    monkeypatch.setattr(lane.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(lane.torch.version, "cuda", lane.EXPECTED_CUDA_VERSION)
    monkeypatch.setattr(
        lane.torch.cuda, "get_device_name", lambda _index: lane.EXPECTED_GPU_NAME
    )
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("SLURMD_NODENAME", lane.EXPECTED_SLURM_NODE)
    monkeypatch.setenv(
        "CUBLAS_WORKSPACE_CONFIG", lane.EXPECTED_CUBLAS_WORKSPACE_CONFIG
    )
    monkeypatch.setattr(
        lane.torch, "get_float32_matmul_precision", lambda: "highest"
    )
    monkeypatch.setattr(lane.torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(lane.torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(
        lane.torch, "are_deterministic_algorithms_enabled", lambda: True
    )
    monkeypatch.setattr(lane.torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(lane.torch.backends.cudnn, "deterministic", True)


@pytest.mark.parametrize("component", sorted(lane.EXPECTED_VERSIONS))
def test_environment_gate_rejects_each_version_drift(
    monkeypatch: pytest.MonkeyPatch, component: str
) -> None:
    _install_expected_environment(monkeypatch)
    if component == "python":
        monkeypatch.setattr(lane.platform, "python_version", lambda: "0.0.0")
    elif component == "numpy":
        monkeypatch.setattr(lane.np, "__version__", "0.0.0")
    elif component == "torch":
        monkeypatch.setattr(lane.torch, "__version__", "0.0.0")
    else:
        monkeypatch.setattr(
            lane,
            "_installed_version",
            lambda name: "0.0.0" if name == component else lane.EXPECTED_VERSIONS[name],
        )
    with pytest.raises(RuntimeError, match="environment differs"):
        lane._numerical_environment(require_gpu=False)


def test_environment_gate_rejects_gpu_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_expected_environment(monkeypatch)
    monkeypatch.setattr(lane.torch.cuda, "get_device_name", lambda _index: "wrong")
    with pytest.raises(RuntimeError, match="requires Quadro RTX 6000"):
        lane._numerical_environment(require_gpu=True)


def test_environment_gate_rejects_node_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_expected_environment(monkeypatch)
    monkeypatch.setenv("SLURMD_NODENAME", "c2-g8-06")
    with pytest.raises(RuntimeError, match="backend differs"):
        lane._numerical_environment(require_gpu=True)


def test_environment_gate_accepts_exact_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_expected_environment(monkeypatch)
    observed = lane._numerical_environment(require_gpu=True)
    assert observed["cuda_matmul_allow_tf32"] is False
    assert observed["cudnn_allow_tf32"] is True


def test_torch_version_subclass_is_normalized_to_literal_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_version_type = type(lane.torch.__version__)
    assert torch_version_type is not str
    version = torch_version_type(lane.EXPECTED_VERSIONS["torch"])
    assert type(version) is torch_version_type
    assert str(version) == lane.EXPECTED_VERSIONS["torch"]
    _install_expected_environment(monkeypatch)
    monkeypatch.setattr(lane.torch, "__version__", version)
    observed = lane._numerical_environment(require_gpu=False)
    assert type(observed["torch"]) is str
    assert observed["torch"] == lane.EXPECTED_VERSIONS["torch"]


@pytest.mark.parametrize(
    "component",
    (
        "python",
        "numpy",
        "libero",
        "cuda",
        "gpu",
        "slurm_node",
        "mujoco_gl",
        "matmul_precision",
        "cublas_workspace_config",
    ),
)
def test_environment_gate_rejects_equal_string_subclasses(
    monkeypatch: pytest.MonkeyPatch, component: str
) -> None:
    class StringSubclass(str):
        pass

    _install_expected_environment(monkeypatch)
    if component == "python":
        monkeypatch.setattr(
            lane.platform,
            "python_version",
            lambda: StringSubclass(lane.EXPECTED_VERSIONS["python"]),
        )
    elif component == "numpy":
        monkeypatch.setattr(
            lane.np,
            "__version__",
            StringSubclass(lane.EXPECTED_VERSIONS["numpy"]),
        )
    elif component == "libero":
        monkeypatch.setattr(
            lane,
            "_installed_version",
            lambda name: StringSubclass(lane.EXPECTED_VERSIONS[name])
            if name == "libero"
            else lane.EXPECTED_VERSIONS[name],
        )
    elif component == "cuda":
        monkeypatch.setattr(
            lane.torch.version,
            "cuda",
            StringSubclass(lane.EXPECTED_CUDA_VERSION),
        )
    elif component == "gpu":
        monkeypatch.setattr(
            lane.torch.cuda,
            "get_device_name",
            lambda _index: StringSubclass(lane.EXPECTED_GPU_NAME),
        )
    elif component == "matmul_precision":
        monkeypatch.setattr(
            lane.torch,
            "get_float32_matmul_precision",
            lambda: StringSubclass("highest"),
        )
    else:
        environment = dict(lane.os.environ)
        environment.update(
            {
                "SLURMD_NODENAME": lane.EXPECTED_SLURM_NODE,
                "MUJOCO_GL": "egl",
                "CUBLAS_WORKSPACE_CONFIG": lane.EXPECTED_CUBLAS_WORKSPACE_CONFIG,
            }
        )
        key = {
            "slurm_node": "SLURMD_NODENAME",
            "mujoco_gl": "MUJOCO_GL",
            "cublas_workspace_config": "CUBLAS_WORKSPACE_CONFIG",
        }[component]
        environment[key] = StringSubclass(environment[key])
        monkeypatch.setattr(lane.os, "environ", environment)
    with pytest.raises(RuntimeError, match="(environment|record types) differ"):
        lane._numerical_environment(require_gpu=True)


def test_environment_record_rejects_non_json_and_scalar_type_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_expected_environment(monkeypatch)
    record = lane._numerical_environment(require_gpu=True)
    assert lane._environment_record_is_exact(record)
    for key, value in record.items():
        non_json = dict(record)
        non_json[key] = object()
        assert not lane._environment_record_is_exact(non_json), key
        alias = dict(record)
        if type(value) is str:
            class StringSubclass(str):
                pass

            alias[key] = StringSubclass(value)
        elif type(value) is bool:
            alias[key] = int(value)
        else:
            raise AssertionError(f"uncovered environment value type for {key}")
        assert not lane._environment_record_is_exact(alias), key


@pytest.mark.parametrize(
    "component",
    (
        "cuda",
        "mujoco_gl",
        "matmul_precision",
        "cuda_tf32",
        "cudnn_tf32",
        "cublas_workspace",
        "deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
    ),
)
def test_environment_gate_rejects_each_backend_drift(
    monkeypatch: pytest.MonkeyPatch, component: str
) -> None:
    _install_expected_environment(monkeypatch)
    if component == "cuda":
        monkeypatch.setattr(lane.torch.version, "cuda", "0.0")
    elif component == "mujoco_gl":
        monkeypatch.setenv("MUJOCO_GL", "wrong")
    elif component == "matmul_precision":
        monkeypatch.setattr(
            lane.torch, "get_float32_matmul_precision", lambda: "medium"
        )
    elif component == "cuda_tf32":
        monkeypatch.setattr(lane.torch.backends.cuda.matmul, "allow_tf32", True)
    elif component == "cudnn_tf32":
        monkeypatch.setattr(lane.torch.backends.cudnn, "allow_tf32", False)
    elif component == "cublas_workspace":
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "wrong")
    elif component == "deterministic_algorithms":
        monkeypatch.setattr(
            lane.torch, "are_deterministic_algorithms_enabled", lambda: False
        )
    elif component == "cudnn_benchmark":
        monkeypatch.setattr(lane.torch.backends.cudnn, "benchmark", True)
    else:
        monkeypatch.setattr(lane.torch.backends.cudnn, "deterministic", False)
    with pytest.raises(RuntimeError, match="backend differs"):
        lane._numerical_environment(require_gpu=True)


def test_terminal_transition_stops_before_another_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"predict": 0, "step": 0}
    executed_actions: list[list[float]] = []

    class Simulation:
        def step(self, action):
            calls["step"] += 1
            executed_actions.append(action)
            return _valid_observation(), 0.0, True, {}

    def predict(*_args, **_kwargs):
        calls["predict"] += 1
        return lane.np.zeros((lane.ACTION_HORIZON, lane.ACTION_DIM), dtype="float32")

    monkeypatch.setattr(lane, "_predict", predict)
    success, steps, terminated = lane._execute_episode(
        Simulation(), object(), _valid_observation(), object(), {}, lane.MAX_STEPS
    )
    assert (success, steps, terminated) == (False, 1, True)
    assert calls == {"predict": 1, "step": 1}
    assert executed_actions[0][:-1] == [0.0] * (lane.ACTION_DIM - 1)
    assert executed_actions[0][-1] == -1.0


def test_transition_normalizes_numpy_bool_to_literal_bool() -> None:
    for value in (lane.np.bool_(False), lane.np.bool_(True)):
        _observation, reward, done = lane._validated_transition(
            (_valid_observation(), 0.0, value, {}), "test transition"
        )
        assert type(reward) is float
        assert type(done) is bool
        assert done is bool(value)


@pytest.mark.parametrize(
    "value",
    (
        pytest.param(0, id="integer-zero"),
        pytest.param(1, id="integer-one"),
        pytest.param(lane.np.array(False), id="scalar-array"),
        pytest.param(lane.np.array([False]), id="vector-array"),
    ),
)
def test_transition_rejects_non_boolean_done_aliases(value: object) -> None:
    with pytest.raises(RuntimeError, match="done flag is not a literal boolean"):
        lane._validated_transition(
            (_valid_observation(), 0.0, value, {}), "test transition"
        )


@pytest.mark.parametrize(
    "mutation",
    ("done", "success", "nan_reward", "invalid_observation"),
)
def test_settle_rejects_invalid_or_terminal_transition(mutation: str) -> None:
    class Simulation:
        def step(self, _action):
            observation = _valid_observation()
            reward: object = 0.0
            done: object = False
            if mutation == "done":
                done = True
            elif mutation == "success":
                reward = 1.0
            elif mutation == "nan_reward":
                reward = float("nan")
            else:
                observation["robot0_eef_pos"][0] = lane.np.nan
            return observation, reward, done, {}

    with pytest.raises(RuntimeError):
        lane._settle_environment(
            Simulation(), _valid_observation(), [0.0] * lane.ACTION_DIM
        )


def test_settle_records_every_validated_transition() -> None:
    calls = 0

    class Simulation:
        def step(self, _action):
            nonlocal calls
            calls += 1
            return _valid_observation(), 0.0, False, {}

    observation, validated = lane._settle_environment(
        Simulation(), _valid_observation(), [0.0] * lane.ACTION_DIM
    )
    assert validated == calls == lane.SETTLE_STEPS
    assert observation.keys() == _valid_observation().keys()


@pytest.mark.parametrize(
    "field,mutation",
    [
        (field, mutation)
        for field in lane._REQUIRED_OBSERVATION_FIELDS
        for mutation in ("missing", "nan", "inf", "shape", "dtype")
    ],
)
def test_observation_contract_rejects_each_field_drift(
    field: str, mutation: str
) -> None:
    observation = _valid_observation()
    if mutation == "missing":
        observation.pop(field)
    elif mutation in {"nan", "inf"}:
        value = observation[field].astype(lane.np.float64)
        value.flat[0] = lane.np.nan if mutation == "nan" else lane.np.inf
        observation[field] = value
    elif mutation == "shape":
        observation[field] = observation[field][:-1]
    else:
        observation[field] = observation[field].astype(
            lane.np.float32 if field != "agentview_image" else lane.np.int16
        )
    with pytest.raises(RuntimeError, match="observation"):
        lane._validate_policy_observation(observation, "test")


@pytest.mark.parametrize(
    "mutation", ("nan_reward", "nonscalar_reward", "nonliteral_done", "nonfinite_final")
)
def test_policy_transition_rejects_invalid_final_step(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def predict(*_args, **_kwargs):
        return lane.np.zeros(
            (lane.ACTION_HORIZON, lane.ACTION_DIM), dtype=lane.np.float32
        )

    class Simulation:
        def step(self, _action):
            observation = _valid_observation()
            reward: object = 0.0
            done: object = False
            if mutation == "nan_reward":
                reward = float("nan")
            elif mutation == "nonscalar_reward":
                reward = [0.0]
            elif mutation == "nonliteral_done":
                done = 0
            else:
                observation["agentview_image"] = observation[
                    "agentview_image"
                ].astype(lane.np.float64)
                observation["agentview_image"][0, 0, 0] = lane.np.inf
            return observation, reward, done, {}

    monkeypatch.setattr(lane, "_predict", predict)
    with pytest.raises(RuntimeError):
        lane._execute_episode(
            Simulation(), object(), _valid_observation(), object(), {}, 1
        )


def test_aggregate_input_recheck_detects_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lane, "RESULT_DIRECTORY", tmp_path)
    records = []
    for start, stop in lane.SHARDS:
        path = lane._shard_output(start, stop)
        path.write_bytes(f"{start}:{stop}".encode())
        records.append(
            {
                "task_start": start,
                "task_end": stop,
                "sha256": lane._sha256(path),
            }
        )
    payload = {"shards": records}
    first = lane._assert_aggregate_shards_unchanged(payload)
    lane._shard_output(0, 3).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed across"):
        lane._assert_aggregate_shards_unchanged(payload, first)


def test_aggregate_smoke_recheck_detects_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lane, "RESULT_DIRECTORY", tmp_path)
    smoke = tmp_path / "rtx6000_capability_smoke.json"
    smoke.write_bytes(b"smoke-v1")
    payload = {"smoke_gate_sha256": lane._sha256(smoke)}
    first = lane._assert_aggregate_smoke_unchanged(payload)
    smoke.write_bytes(b"smoke-v2")
    with pytest.raises(RuntimeError, match="changed across"):
        lane._assert_aggregate_smoke_unchanged(payload, first)


def test_failed_capability_floor_publishes_then_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lane, "RESULT_DIRECTORY", tmp_path)
    shard_records = []
    for start, stop in lane.SHARDS:
        path = lane._shard_output(start, stop)
        path.write_bytes(f"{start}:{stop}".encode())
        shard_records.append(
            {
                "task_start": start,
                "task_end": stop,
                "sha256": lane._sha256(path),
            }
        )
    output = tmp_path / "failed.json"
    payload = {
        "shards": shard_records,
        "smoke_gate_sha256": "0" * 64,
        "all_gates_pass": False,
        "capability_floor_passed": False,
    }
    smoke = tmp_path / "rtx6000_capability_smoke.json"
    smoke.write_bytes(b"smoke")
    payload["smoke_gate_sha256"] = lane._sha256(smoke)
    monkeypatch.setattr(lane, "_aggregate", lambda: (payload, output))
    monkeypatch.setattr(lane, "_fixed_sources", lambda: {"verified": True})
    monkeypatch.setattr(sys, "argv", ["runner", "--mode", "aggregate"])
    with pytest.raises(SystemExit) as error:
        lane.main()
    assert error.value.code == 1
    assert output.is_file() and not output.is_symlink()
    assert output.stat().st_mode & 0o222 == 0
    assert json.loads(output.read_text())["capability_floor_passed"] is False


def _manifest_entries(path: Path) -> dict[str, str]:
    return {
        fields[1].lstrip("*"): fields[0]
        for raw in path.read_text().splitlines()
        if (line := raw.strip()) and not line.startswith("#")
        for fields in (line.split(maxsplit=1),)
    }


def test_shared_model_closure_is_exact_frozen_b1c_intersection() -> None:
    frozen = lane.PINNED_B1C_CAPABILITY_MANIFEST
    if lane.PROJECT_ROOT != lane.EXPECTED_STAGE_ROOT:
        frozen = lane.PROJECT_ROOT / "athena/capable_linear_fresh_capability_sources.sha256"
    current = _manifest_entries(lane.SOURCE_MANIFEST)
    b1c = _manifest_entries(frozen)
    assert lane._shared_source_identity_is_exact(current, b1c)
    assert {
        relative: current[relative]
        for relative in lane.EXPECTED_FROZEN_SHARED_SHA256
    } == lane.EXPECTED_FROZEN_SHARED_SHA256


def test_all_capability_output_schemas_match_producer_ast() -> None:
    tree = ast.parse(Path(lane.__file__).read_text())
    top_level_function_positions = {
        node.name: index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.FunctionDef)
    }
    assert top_level_function_positions["_type_exact_equal"] < (
        top_level_function_positions["_preimport_verification"]
    )

    def function(name: str) -> ast.FunctionDef:
        return next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

    def assigned_dict(owner: ast.FunctionDef, name: str) -> ast.Dict:
        return next(
            node.value
            for node in ast.walk(owner)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
            and isinstance(node.value, ast.Dict)
        )

    rollout_payload = assigned_dict(function("_run_evaluation"), "payload")
    rollout_fields = {
        ast.literal_eval(key): value
        for key, value in zip(rollout_payload.keys, rollout_payload.values)
    }
    assert set(rollout_fields) == lane.ROLLOUT_RESULT_KEYS
    assert isinstance(rollout_fields["protocol"], ast.Call)
    assert isinstance(rollout_fields["protocol"].func, ast.Name)
    assert rollout_fields["protocol"].func.id == "_expected_rollout_protocol"
    assert set(lane._expected_rollout_protocol(1, 1)) == lane.ROLLOUT_PROTOCOL_KEYS
    episode_record = assigned_dict(function("_run_evaluation"), "record")
    assert {ast.literal_eval(key) for key in episode_record.keys} == (
        lane.EPISODE_RECORD_KEYS
    )

    aggregate_payload = assigned_dict(function("_aggregate"), "payload")
    aggregate_fields = {
        ast.literal_eval(key): value
        for key, value in zip(aggregate_payload.keys, aggregate_payload.values)
    }
    assert set(aggregate_fields) == lane.AGGREGATE_RESULT_KEYS
    assert isinstance(aggregate_fields["protocol"], ast.Call)
    assert isinstance(aggregate_fields["protocol"].func, ast.Name)
    assert aggregate_fields["protocol"].func.id == "_expected_aggregate_protocol"
    assert set(lane._expected_aggregate_protocol()) == lane.AGGREGATE_PROTOCOL_KEYS
    aggregate_sentinel = aggregate_fields["inactive_vision_norm_sentinel"]
    assert isinstance(aggregate_sentinel, ast.Dict)
    assert {ast.literal_eval(key) for key in aggregate_sentinel.keys} == (
        lane.AGGREGATE_SENTINEL_KEYS
    )

    load_function = function("_load_model")
    load_return = next(
        node.value
        for node in ast.walk(load_function)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Tuple)
        and len(node.value.elts) == 2
        and isinstance(node.value.elts[1], ast.Dict)
    )
    checkpoint = load_return.elts[1]
    assert isinstance(checkpoint, ast.Dict)
    assert {ast.literal_eval(key) for key in checkpoint.keys} == (
        lane.CHECKPOINT_RECORD_KEYS
    )
