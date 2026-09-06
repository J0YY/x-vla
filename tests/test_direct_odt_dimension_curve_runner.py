"""The exporter handoff must not bypass or relabel full-rank certification."""

import importlib.util
import json
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_direct_odt_dimension_curve_v1.py"
if not RUNNER.is_file():
    RUNNER = ROOT / "athena/run_direct_odt_dimension_curve_v1.py"
SPEC = importlib.util.spec_from_file_location("curve_runner_under_test", RUNNER)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeTensor:
    def __getitem__(self, _item):
        return self

    def detach(self):
        return self

    def clone(self):
        return self


def fake_base(path, gates=True, write=True, checkpoint=runner.CHECKPOINT_SHA256):
    class Bank:
        def finish(self):
            return {"node_count": 7}

    base = types.SimpleNamespace(CompactRankBank=Bank)
    base.diagonalize_implicit_dag_full_rank = lambda *args, **kwargs: "diagonal"

    def full():
        base.diagonalize_implicit_dag_full_rank(replay_inputs={"input": FakeTensor()})
        if write:
            path.write_text(json.dumps({"all_full_rank_gates_pass": True,
                "full_rank_gates": {"replay": gates},
                "schema": "xvla_capable_linear_b1c0_full_rank_direct_odt_v1",
                "canonical_direct_odt_algorithms_1_to_3_completed": True,
                "runtime_guard": {"prohibited_attempt_count": 0},
                "checkpoint": {"checkpoint_sha256": checkpoint}}))
            path.chmod(0o444)
        base.CompactRankBank().finish()
        raise AssertionError("legacy compression code must never execute")

    base._full_mode = full
    return base


def test_handoff_after_certificate_restores_hooks(tmp_path):
    path = tmp_path / "certificate.json"
    base = fake_base(path)
    original = (base.CompactRankBank, base.diagonalize_implicit_dag_full_rank)
    result = runner.certified_handoff(base, path)
    assert result["diagonal"] == "diagonal"
    assert result["certificate_sha256"] == runner.sha256(path)
    assert (base.CompactRankBank, base.diagonalize_implicit_dag_full_rank) == original


@pytest.mark.parametrize("kwargs", [{"gates": False}, {"write": False}, {"checkpoint": "0" * 64}])
def test_handoff_fails_closed(tmp_path, kwargs):
    path = tmp_path / "certificate.json"
    base = fake_base(path, **kwargs)
    original = (base.CompactRankBank, base.diagonalize_implicit_dag_full_rank)
    with pytest.raises(RuntimeError):
        runner.certified_handoff(base, path)
    assert (base.CompactRankBank, base.diagonalize_implicit_dag_full_rank) == original


def test_driver_exception_not_swallowed(tmp_path):
    base = fake_base(tmp_path / "certificate.json")
    def failure():
        raise ArithmeticError("failed real gate")
    base._full_mode = failure
    with pytest.raises(ArithmeticError):
        runner.certified_handoff(base, tmp_path / "certificate.json")


def test_writable_certificate_rejected(tmp_path):
    path = tmp_path / "certificate.json"
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="immutable"):
        runner.read_certificate(path)


def test_symlink_certificate_rejected(tmp_path):
    path = tmp_path / "certificate.json"
    path.write_text("{}")
    path.chmod(0o444)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(RuntimeError, match="immutable"):
        runner.read_certificate(link)


def test_more_than_one_diagonalization_rejected(tmp_path):
    path = tmp_path / "certificate.json"
    base = fake_base(path)
    def twice():
        base.diagonalize_implicit_dag_full_rank(replay_inputs={"input": FakeTensor()})
        base.diagonalize_implicit_dag_full_rank(replay_inputs={"input": FakeTensor()})
    base._full_mode = twice
    with pytest.raises(RuntimeError, match="more than one"):
        runner.certified_handoff(base, path)


def test_finish_before_diagonalization_rejected(tmp_path):
    path = tmp_path / "certificate.json"
    base = fake_base(path)
    base._full_mode = lambda: base.CompactRankBank().finish()
    with pytest.raises(RuntimeError, match="precede"):
        runner.certified_handoff(base, path)


def test_import_name_collision_rejected():
    with pytest.raises(RuntimeError, match="already imported"):
        runner.load_module("sys", RUNNER)
