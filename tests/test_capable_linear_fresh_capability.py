from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import pytest

from scripts import run_capable_linear_fresh_capability as lane


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
            "success": False,
            "steps": 1,
            "terminated_without_success": False,
        }
        for task in range(10)
        for episode in range(50)
    ]
    assert lane._episode_inventory_is_exact(records, 0, 10, 50, 280)
    assert not lane._episode_inventory_is_exact(records[:-1], 0, 10, 50, 280)
    duplicated = list(records)
    duplicated[-1] = dict(duplicated[-2])
    assert not lane._episode_inventory_is_exact(duplicated, 0, 10, 50, 280)


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
    exact_runner = ast.parse(
        (lane.PROJECT_ROOT / "scripts/run_capable_linear_direct_odt_spectrum.py").read_text()
    )
    assignments = {
        target.id: ast.literal_eval(node.value)
        for node in exact_runner.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id
        in {"EXPECTED_CHECKPOINT_TENSOR_SHAPES", "EXPECTED_PARAMETER_COUNT"}
    }
    assert assignments == {
        "EXPECTED_CHECKPOINT_TENSOR_SHAPES": expected,
        "EXPECTED_PARAMETER_COUNT": lane.EXPECTED_PARAMETER_COUNT,
    }


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
    with pytest.raises(RuntimeError, match="requires NVIDIA A30"):
        lane._numerical_environment(require_gpu=True)


def test_environment_gate_accepts_exact_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_expected_environment(monkeypatch)
    observed = lane._numerical_environment(require_gpu=True)
    assert observed["cuda_matmul_allow_tf32"] is False
    assert observed["cudnn_allow_tf32"] is True


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
            return {"observation": calls["step"]}, 0.0, True, {}

    def predict(*_args, **_kwargs):
        calls["predict"] += 1
        return lane.np.zeros((lane.ACTION_HORIZON, lane.ACTION_DIM), dtype="float32")

    monkeypatch.setattr(lane, "_predict", predict)
    success, steps, terminated = lane._execute_episode(
        Simulation(), object(), {}, object(), {}, lane.MAX_STEPS
    )
    assert (success, steps, terminated) == (False, 1, True)
    assert calls == {"predict": 1, "step": 1}
    assert executed_actions[0][:-1] == [0.0] * (lane.ACTION_DIM - 1)
    assert executed_actions[0][-1] == -1.0


def test_settle_rejects_early_terminal_state() -> None:
    class Simulation:
        def step(self, _action):
            return {}, 0.0, True, {}

    with pytest.raises(RuntimeError, match="during settling"):
        lane._settle_environment(Simulation(), {}, [0.0] * lane.ACTION_DIM)


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
    smoke = tmp_path / "fresh_capability_smoke.json"
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
    smoke = tmp_path / "fresh_capability_smoke.json"
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
