"""Fail-closed tests for the reported-specification SVHN reproduction lane."""

from __future__ import annotations

import copy
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from athena import dooms_svhn_reproduction as reproduction
from athena import launch_dooms_svhn_reproduction as launcher
from xvla.models.dooms_xnet import DoomsReportedChiNetConfig
from xvla.nn.normalization import RmsBatchNorm


def test_frozen_reproduction_identity_and_architecture_are_exact():
    assert reproduction.SCHEMA == "xvla-dooms-chi-net-svhn-reported-spec-reproduction-v1"
    assert reproduction.SEEDS == (0, 1, 2, 3, 4)
    assert reproduction.TRAIN_BATCH_SIZE == 2048
    assert reproduction.EPSILON_GRID == (
        0.0,
        0.0025,
        0.005,
        0.01,
        0.015,
        0.02,
        0.03,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.50,
    )
    model = reproduction.exact_model()
    assert model.cfg == DoomsReportedChiNetConfig()
    assert model.num_params() == reproduction.EXPECTED_PARAMETER_COUNT == 659_722
    assert [tuple(layer.left.weight.shape) for layer in model.layers] == [(256, 256)] * 3
    assert [tuple(layer.right.weight.shape) for layer in model.layers] == [(256, 256)] * 3
    assert all(not hasattr(layer, "down") for layer in model.layers)


def test_grayscale_formula_shape_scale_and_coefficients_are_literal():
    rgb = torch.zeros(3, 3, 32, 32, dtype=torch.uint8)
    rgb[0, 0].fill_(255)
    rgb[1, 1].fill_(255)
    rgb[2, 2].fill_(255)
    gray = reproduction.grayscale(rgb)
    assert gray.shape == (3, 1, 32, 32)
    assert gray.dtype == torch.float32
    assert torch.allclose(
        gray[:, 0, 0, 0],
        torch.tensor([0.2989, 0.5870, 0.1140]),
        atol=1e-7,
        rtol=0,
    )
    with pytest.raises(ValueError, match="shape"):
        reproduction.grayscale(torch.zeros(2, 1, 32, 32))


def test_epsilon_rank_schedule_uses_corrected_trace_tail_budget_and_clusters():
    values_a = torch.tensor([9.0, 4.0, 4.0, 1.0], dtype=torch.float64)
    values_b = torch.tensor([10.0, 5.0, 2.0, 1.0], dtype=torch.float64)
    vectors = torch.eye(4, dtype=torch.float64)
    systems = ((values_a, vectors), (values_b, vectors))
    schedule = reproduction.epsilon_rank_schedule(
        systems, full_norm_squared=18.0, epsilon_grid=(0.0, 0.5, 1.0)
    )
    assert schedule[0]["rank_tuple"] == [4, 4]
    assert [item["per_projection_tail_budget"] for item in schedule] == pytest.approx(
        [0.0, 1.5, 6.0]
    )
    # At epsilon=.5, rank 3 meets the budget. At epsilon=1, the first
    # spectrum's nominal rank 2 cuts the repeated eigenvalue cluster and must
    # be completed upward to rank 3.
    assert schedule[1]["rank_tuple"] == [3, 3]
    assert schedule[2]["rank_tuple"] == [3, 2]
    assert all(
        all(new <= old for new, old in zip(right["rank_tuple"], left["rank_tuple"]))
        for left, right in zip(schedule, schedule[1:])
    )


def test_zero_epsilon_forces_full_rank_even_with_numerical_null_eigenvalues():
    values = torch.tensor([3.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    vectors = torch.eye(4, dtype=torch.float64)
    systems = tuple((values, vectors) for _ in range(4))
    corrected = reproduction.epsilon_rank_schedule(systems, 4.0, epsilon_grid=(0.0,))[0]
    assert corrected["rank_tuple"] == [4, 4, 4, 4]
    printed = reproduction.paper_printed_rank_allocation(systems)["schedules"][0]
    assert printed["rank_tuple"] == [4, 4, 4, 4]
    assert printed["minimal_printed_rule_rank_tuple"] == [2, 2, 2, 2]
    assert printed["evaluated_cluster_complete_rank_tuple"] == [4, 4, 4, 4]


def test_checkpoint_round_trip_persists_python_rbn_runtime_state(tmp_path):
    model = reproduction.exact_model()
    for index, norm in enumerate(model.norms):
        assert isinstance(norm, RmsBatchNorm)
        norm.initialized.fill_(True)
        norm.running_rms.fill_(1.0 + index)
        norm.freeze()
    path = tmp_path / "calibrated.pt"
    torch.save(reproduction.checkpoint_payload(model, 2, "calibrated_final"), path)
    loaded = reproduction.load_checkpoint(path, 2, "calibrated_final", device="cpu")
    assert all(norm.frozen for norm in loaded.norms)
    assert all(not norm.calibrating for norm in loaded.norms)
    assert [float(norm.running_rms) for norm in loaded.norms] == [1.0, 2.0, 3.0]


def test_checkpoint_loader_rejects_extra_fields_and_wrong_stage_state(tmp_path):
    model = reproduction.exact_model()
    for norm in model.norms:
        norm.initialized.fill_(True)
    payload = reproduction.checkpoint_payload(model, 0, "raw_final_epoch")
    path = tmp_path / "raw.pt"
    torch.save(payload, path)
    loaded = reproduction.load_checkpoint(path, 0, "raw_final_epoch", device="cpu")
    assert all(not norm.frozen and not norm.calibrating for norm in loaded.norms)

    payload["unexpected"] = True
    extra = tmp_path / "extra.pt"
    torch.save(payload, extra)
    with pytest.raises(RuntimeError, match="inventory"):
        reproduction.load_checkpoint(extra, 0, "raw_final_epoch", device="cpu")

    payload.pop("unexpected")
    payload["rbn_runtime_state"][0]["frozen"] = True
    wrong = tmp_path / "wrong.pt"
    torch.save(payload, wrong)
    with pytest.raises(RuntimeError, match="does not match its stage"):
        reproduction.load_checkpoint(wrong, 0, "raw_final_epoch", device="cpu")


def test_calibration_transition_allows_only_rbn_calibration_state():
    raw = reproduction.exact_model()
    calibrated = copy.deepcopy(raw)
    for raw_norm, calibrated_norm in zip(raw.norms, calibrated.norms):
        raw_norm.initialized.fill_(True)
        calibrated_norm.initialized.fill_(True)
        calibrated_norm.running_rms.add_(0.5)
        calibrated_norm._calib_sum.add_(3.0)
        calibrated_norm._calib_count.add_(2.0)
        calibrated_norm.freeze()
    reproduction.validate_calibrated_transition(raw, calibrated)
    with torch.no_grad():
        calibrated.head.weight[0, 0].add_(1.0)
    with pytest.raises(RuntimeError, match="learned"):
        reproduction.validate_calibrated_transition(raw, calibrated)


def _allocation_record(systems, full_norm_squared):
    return reproduction.corrected_rank_allocation(systems, full_norm_squared)


def test_rank_allocation_validator_recomputes_and_rejects_tampering():
    values = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64)
    vectors = torch.eye(4, dtype=torch.float64)
    systems = tuple((values, vectors) for _ in range(4))
    record = _allocation_record(systems, full_norm_squared=10.0)
    validated = reproduction.validate_rank_allocation(record, systems, 10.0)
    assert len(validated) == len(reproduction.EPSILON_GRID)

    bad_rank = copy.deepcopy(record)
    bad_rank["schedules"][0]["rank_tuple"][0] = True
    with pytest.raises(RuntimeError, match="rank tuple"):
        reproduction.validate_rank_allocation(bad_rank, systems, 10.0)

    bad_budget = copy.deepcopy(record)
    bad_budget["schedules"][3]["per_projection_tail_budget"] += 1.0
    with pytest.raises(RuntimeError, match="budget"):
        reproduction.validate_rank_allocation(bad_budget, systems, 10.0)

    printed = reproduction.paper_printed_rank_allocation(systems)
    assert len(reproduction.validate_paper_printed_rank_allocation(printed, systems)) == 15
    bad_printed = copy.deepcopy(printed)
    bad_printed["schedules"][1]["rank_tuple"][0] -= 1
    with pytest.raises(RuntimeError, match="rank"):
        reproduction.validate_paper_printed_rank_allocation(bad_printed, systems)

    uniform = reproduction.uniform_dimension_rank_sweep(systems)
    assert len(reproduction.validate_uniform_dimension_sweep(uniform, systems)) == len(
        reproduction.REMOVAL_GRID
    )


def test_evaluation_plan_has_distinct_full_row_and_every_schedule():
    schedules = tuple({"epsilon": epsilon, "rank_tuple": [257] * 4} for epsilon in reproduction.EPSILON_GRID)
    printed = tuple({"epsilon": epsilon, "rank_tuple": [257] * 4} for epsilon in reproduction.EPSILON_GRID)
    uniform = tuple(
        {"target_fraction_dimensions_removed": fraction, "rank_tuple": [257] * 4}
        for fraction in reproduction.REMOVAL_GRID
    )
    rows = reproduction.build_evaluation_plan(schedules, printed, uniform)
    assert len(rows) == 44
    assert rows[0] == {
        "row_index": 0,
        "kind": "uncompressed_full",
        "family_code": 0,
        "schedule_index": -1,
        "rank_tuple": [257] * 4,
    }
    assert rows[1]["schedule_index"] == 0
    assert rows[1]["kind"] == "paper_printed_gram_frobenius"
    assert rows[1]["epsilon"] == 0.0
    assert rows[16]["kind"] == "corrected_trace_hsvd"
    assert rows[31]["kind"] == "uniform_dimension_sweep"
    assert rows[-1]["schedule_index"] == len(reproduction.REMOVAL_GRID) - 1


def test_strict_json_rejects_duplicate_keys_and_nonstandard_numbers(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}')
    with pytest.raises(ValueError, match="duplicate"):
        reproduction.strict_json(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a": NaN}')
    with pytest.raises(ValueError, match="nonstandard"):
        reproduction.strict_json(nonfinite)


def test_replay_inventory_is_exact_and_nonvacuous():
    tensor = torch.tensor([1.0], dtype=torch.float64)
    metrics = reproduction.replay_metrics(tensor, tensor)
    reproduction.validate_replay_inventory({"same": metrics}, {"same"}, "unit")
    with pytest.raises(RuntimeError, match="inventory"):
        reproduction.validate_replay_inventory({}, {"same"}, "unit")
    forged = copy.deepcopy(metrics)
    forged["passed"] = False
    with pytest.raises(RuntimeError, match="did not pass"):
        reproduction.validate_replay_inventory({"same": forged}, {"same"}, "unit")


def test_array_runtime_binding_uses_master_and_task_not_element_job_id():
    launch = {"jobs": {"train": "100"}}
    runtime = {
        "python": "3.11.0",
        "python_executable_invoked": "/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python",
        "python_executable_resolved": "/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python3.11",
        "platform": "Linux",
        "packages": {"torch": "x", "torchvision": "x", "numpy": "x", "scipy": "x"},
        "cuda": "12.1",
        "cudnn": 8900,
        "gpu": "NVIDIA RTX A6000",
        "slurm_job_id": "237",
        "slurm_array_job_id": "100",
        "slurm_array_task_id": "3",
    }
    reproduction.validate_runtime_binding(runtime, launch, "train", 3)
    wrong = dict(runtime, slurm_array_task_id="2")
    with pytest.raises(RuntimeError, match="array-task"):
        reproduction.validate_runtime_binding(wrong, launch, "train", 3)
    incomplete = {
        "slurm_job_id": "237",
        "slurm_array_job_id": "100",
        "slurm_array_task_id": "3",
    }
    with pytest.raises(RuntimeError, match="malformed"):
        reproduction.validate_runtime_binding(incomplete, launch, "train", 3)


def test_software_fingerprint_records_but_does_not_equate_partition_platforms():
    base = {
        "python": "3.11.0",
        "python_executable_invoked": "/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python",
        "python_executable_resolved": "/athenahomes/joy/miniconda3/envs/safesae-openvla/bin/python3.11",
        "platform": "compute-kernel",
        "packages": {"torch": "x", "torchvision": "x", "numpy": "x", "scipy": "x"},
        "cuda": "12.1",
        "cudnn": 8900,
        "gpu": None,
        "slurm_job_id": "1",
        "slurm_array_job_id": None,
        "slurm_array_task_id": None,
    }
    other = dict(base, platform="gpu-kernel", slurm_job_id="2")
    reproduction.validate_software_match(base, other, "cross-partition test")
    with pytest.raises(RuntimeError, match="software environment"):
        reproduction.validate_software_match(
            base, {**other, "packages": {**other["packages"], "numpy": "changed"}}, "test"
        )


def test_artifact_capture_detects_consumer_side_change(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"before")
    captured = reproduction.capture_artifact(path, tmp_path, "input")
    reproduction.require_artifact_unchanged(path, captured, tmp_path, "input")
    path.write_bytes(b"after")
    with pytest.raises(RuntimeError, match="changed while"):
        reproduction.require_artifact_unchanged(path, captured, tmp_path, "input")


def _launcher_tree(tmp_path):
    root = tmp_path / "run"
    data = tmp_path / "data"
    root.mkdir()
    data.mkdir()
    (root / "frozen.py").write_text("x = 1\n")
    return root, data


def test_launcher_writes_dag_before_release_and_uses_terminal_dependencies(
    tmp_path, monkeypatch,
):
    root, data = _launcher_tree(tmp_path)
    monkeypatch.setattr(launcher, "SOURCE_PATHS", ("frozen.py",))
    submitted = []
    events = []

    def fake_submit(*args, **kwargs):
        submitted.append((args, kwargs))
        return str(100 + len(submitted))

    def fake_run(command, check):
        events.append((list(command), (root / "results" / "launch.json").exists(), check))

    monkeypatch.setattr(launcher, "submit", fake_submit)
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "launch", "--run-root", str(root), "--data-root", str(data)
    ])
    launcher.main()
    launch = json.loads((root / "results" / "launch.json").read_text())
    assert launch["dag_schema"] == reproduction.DAG_SCHEMA
    assert all(value.startswith("afterany:") for value in launch["dependencies"].values())
    assert launch["node_order"][-2:] == ["verify_eval", "summary"]
    assert reproduction.validate_launch(root.resolve(), data.resolve()) == launch
    assert submitted[0][1]["hold"] is True
    assert [call[0][1] for call in submitted] == [
        "dooms-unit", "dooms-prefetch", "dooms-feasibility", "dooms-train",
        "dooms-calibrate", "dooms-odt", "dooms-eval", "dooms-verify-eval",
        "dooms-summary",
    ]
    assert [call[1].get("array") for call in submitted] == [
        None, None, None, "0-4", "0-4", "0-4", "0-4", "0-4", None,
    ]
    assert [call[1].get("dependency") for call in submitted[1:]] == [
        "afterany:101", "afterany:102", "afterany:103", "afterany:104",
        "afterany:105", "afterany:106", "afterany:107", "afterany:108",
    ]
    assert events == [(["scontrol", "release", "101"], True, True)]


def test_launcher_cancels_every_submitted_job_on_transaction_failure(tmp_path, monkeypatch):
    root, data = _launcher_tree(tmp_path)
    monkeypatch.setattr(launcher, "SOURCE_PATHS", ("frozen.py",))
    events = []
    counter = {"value": 0}

    def fake_submit(*args, **kwargs):
        counter["value"] += 1
        if counter["value"] == 3:
            raise RuntimeError("submission failed")
        return str(200 + counter["value"])

    def fake_run(command, check):
        events.append((list(command), check))

    monkeypatch.setattr(launcher, "submit", fake_submit)
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "launch", "--run-root", str(root), "--data-root", str(data)
    ])
    with pytest.raises(RuntimeError, match="submission failed"):
        launcher.main()
    assert events == [(["scancel", "201", "202"], False)]
    receipt = tmp_path / "run.launch_failure.json"
    assert json.loads(receipt.read_text())["submitted_jobs"] == {
        "unit": "201", "prefetch": "202"
    }


def test_launcher_rolls_back_manifest_if_freeze_fails_before_submission(tmp_path, monkeypatch):
    root, data = _launcher_tree(tmp_path)
    monkeypatch.setattr(launcher, "SOURCE_PATHS", ("frozen.py",))
    real_chmod = launcher.os.chmod

    def fail_source_chmod(path, mode):
        if Path(path).name == "frozen.py":
            raise OSError("chmod failed")
        return real_chmod(path, mode)

    monkeypatch.setattr(launcher.os, "chmod", fail_source_chmod)
    monkeypatch.setattr(sys, "argv", [
        "launch", "--run-root", str(root), "--data-root", str(data)
    ])
    with pytest.raises(OSError, match="chmod failed"):
        launcher.main()
    assert not (root / "source_manifest.sha256").exists()
    receipt = tmp_path / "run.launch_failure.json"
    assert json.loads(receipt.read_text())["submitted_jobs"] == {}


def _write_bytes(path, payload=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_odt_and_evaluation_verification_records_bind_every_artifact(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    _write_bytes(tmp_path / "source_manifest.sha256", b"manifest")
    seed = 0
    paths = {
        "odt_json": results / "paper_seed_0_odt.json",
        "odt_matrix_artifact": results / "paper_seed_0_odt_matrices.npz",
        "calibrated_checkpoint": results / "paper_seed_0_calibrated.pt",
        "calibration_json": results / "paper_seed_0_calibrate.json",
        "evaluation_json": results / "paper_seed_0_eval.json",
        "prediction_artifact": results / "paper_seed_0_test_predictions.npz",
        "consumed_odt_verification": results / "paper_seed_0_odt_verification.json",
    }
    for index, path in enumerate(paths.values()):
        _write_bytes(path, f"artifact-{index}".encode())
    runtime = {"opaque": True}
    monkeypatch.setattr(reproduction, "validate_runtime_binding", lambda *args: None)
    odt_record = {
        "schema": reproduction.SCHEMA,
        "stage": "odt_verification",
        "seed": seed,
        **{key: reproduction.artifact(paths[key]) for key in (
            "odt_json", "odt_matrix_artifact", "calibrated_checkpoint", "calibration_json"
        )},
        "validated_bonds": 4,
        "validated_schedule_counts": [15, 15, 13],
        "source_manifest_sha256": reproduction.sha256(tmp_path / "source_manifest.sha256"),
        "runtime": runtime,
    }
    (results / "paper_seed_0_odt_verification.json").write_text(json.dumps(odt_record))
    reproduction.validate_odt_verification(tmp_path, {}, seed)

    eval_record = {
        "schema": reproduction.SCHEMA,
        "stage": "eval_verification",
        "seed": seed,
        **{key: reproduction.artifact(paths[key]) for key in (
            "evaluation_json", "prediction_artifact", "odt_json", "odt_matrix_artifact",
            "consumed_odt_verification",
        )},
        "validated_rows": 44,
        "fresh_prediction_exact_match": True,
        "fresh_discrete_metric_exact_match": True,
        "fresh_loss_match_tolerance": {"relative": 1e-9, "absolute": 1e-12},
        "source_manifest_sha256": reproduction.sha256(tmp_path / "source_manifest.sha256"),
        "runtime": runtime,
    }
    (results / "paper_seed_0_eval_verification.json").write_text(json.dumps(eval_record))
    reproduction.validate_eval_verification(tmp_path, {}, seed)
    paths["odt_matrix_artifact"].write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="artifact chain"):
        reproduction.validate_odt_verification(tmp_path, {}, seed)
    with pytest.raises(RuntimeError, match="artifact chain"):
        reproduction.validate_eval_verification(tmp_path, {}, seed)


def test_evaluation_record_validates_all_44_rows_and_rejects_mapping_tampering(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    seed = 0
    odt_verification = results / "paper_seed_0_odt_verification.json"
    odt_verification.write_text("{}")
    schedules = tuple(
        {"epsilon": epsilon, "rank_tuple": [257] * 4}
        for epsilon in reproduction.EPSILON_GRID
    )
    printed = tuple(
        {"epsilon": epsilon, "rank_tuple": [257] * 4}
        for epsilon in reproduction.EPSILON_GRID
    )
    uniform = tuple(
        {"target_fraction_dimensions_removed": fraction, "rank_tuple": [257] * 4}
        for fraction in reproduction.REMOVAL_GRID
    )
    plan = reproduction.build_evaluation_plan(schedules, printed, uniform)
    count = reproduction.EXPECTED_SPLIT_LENGTHS["test"]
    labels = np.zeros(count, dtype=np.int64)
    predictions = np.zeros((44, count), dtype=np.int64)
    family_codes = np.asarray([row["family_code"] for row in plan], dtype=np.int64)
    schedule_indices = np.asarray([row["schedule_index"] for row in plan], dtype=np.int64)
    rank_tuples = np.asarray([row["rank_tuple"] for row in plan], dtype=np.int64)
    prediction_path = results / "paper_seed_0_test_predictions.npz"
    reproduction.atomic_npz(prediction_path, {
        "predictions": predictions,
        "labels": labels,
        "family_codes": family_codes,
        "schedule_indices": schedule_indices,
        "rank_tuples": rank_tuples,
    })
    metrics = {"loss_sum": 1.0, "correct": count, "count": count, "accuracy": 1.0}
    rows = [{**row, **metrics} for row in plan]
    metadata = {
        "labels": {"dtype": "int64", "shape": [count]},
        "family_codes": {"dtype": "int64", "shape": [44]},
        "predictions": {"dtype": "int64", "shape": [44, count]},
        "rank_tuples": {"dtype": "int64", "shape": [44, 4]},
        "schedule_indices": {"dtype": "int64", "shape": [44]},
    }
    evaluate = {
        "schema": reproduction.SCHEMA,
        "stage": "eval",
        "seed": seed,
        "official_test_access": True,
        "evaluation_rows": rows,
        "uncompressed_full": rows[0],
        "paper_printed_curve": [
            {**row, **schedule, **metrics}
            for row, schedule in zip(plan[1:16], printed)
        ],
        "corrected_epsilon_curve": [
            {**row, **schedule, **metrics}
            for row, schedule in zip(plan[16:31], schedules)
        ],
        "uniform_dimension_curve": [
            {**row, **schedule, **metrics}
            for row, schedule in zip(plan[31:44], uniform)
        ],
        "prediction_artifact": {
            **reproduction.artifact(prediction_path),
            "allow_pickle": False,
            "keys": metadata,
            "predictions_sha256": reproduction.hashlib.sha256(predictions.tobytes()).hexdigest(),
            "labels_sha256": reproduction.hashlib.sha256(labels.tobytes()).hexdigest(),
        },
        "odt_json": {"unused": True},
        "odt_matrix_artifact": {"unused": True},
        "consumed_odt_verification": reproduction.artifact(odt_verification),
        "paper_reported_accuracy_comparison_only": 0.854,
        "source_manifest_sha256": "unused",
        "runtime": {"unused": True},
    }
    derived, _, stored = reproduction.validate_evaluation_record(
        tmp_path, seed, evaluate, schedules, printed, uniform, labels
    )
    assert len(derived) == 44 and np.array_equal(stored, predictions)
    evaluate["evaluation_rows"][31]["family_code"] = 2
    with pytest.raises(RuntimeError, match="descriptor"):
        reproduction.validate_evaluation_record(
            tmp_path, seed, evaluate, schedules, printed, uniform, labels
        )


def test_verify_eval_is_a_seeded_cli_stage_with_explicit_dispatch():
    source = inspect.getsource(reproduction.main)
    assert '"verify_eval"' in source
    assert "stage_verify_eval(run_root, data_root, args.seed)" in source


def test_cosine_learning_rate_trace_is_fully_reconstructible():
    trace = reproduction.expected_cosine_learning_rate_trace(32)
    assert trace.shape == (32,)
    assert trace.dtype == np.float64
    assert trace[0] == 0.001
    assert np.all(trace[1:] < trace[:-1])
    assert trace[-1] > 0.0


def test_atomic_npz_round_trip_and_overwrite_refusal(tmp_path):
    path = tmp_path / "arrays.npz"
    arrays = {"x": np.arange(6, dtype=np.float64).reshape(2, 3)}
    reproduction.atomic_npz(path, arrays)
    with np.load(path, allow_pickle=False) as bundle:
        assert np.array_equal(bundle["x"], arrays["x"])
    with pytest.raises(RuntimeError, match="overwrite"):
        reproduction.atomic_npz(path, arrays)


def test_pre_eval_stages_do_not_construct_the_test_split():
    for function in (
        reproduction.stage_train,
        reproduction.stage_calibrate,
        reproduction.stage_odt,
        reproduction.stage_feasibility,
    ):
        source = inspect.getsource(function)
        assert 'split="test"' not in source
    assert 'split="test"' in inspect.getsource(reproduction.stage_eval)


def test_every_unpublished_choice_is_explicitly_named():
    assert set(reproduction.ASSUMPTIONS) == {
        "grayscale",
        "input_noise_norm_0p3",
        "noise_scope",
        "noise_rng",
        "bilinear_initialization",
        "embedding_head_initialization",
        "biases",
        "normalization_placement",
        "normalization_definition",
        "calibration",
        "adamw",
        "cosine",
        "remainder",
        "precision",
        "softmax",
    }
    assert set(reproduction.SVHN_FILES) == {
        "train_32x32.mat",
        "extra_32x32.mat",
        "test_32x32.mat",
    }


def test_source_manifest_inventory_covers_every_xvla_python_file():
    repository = Path(reproduction.__file__).resolve().parents[1]
    all_xvla = {
        str(path.relative_to(repository)) for path in (repository / "xvla").glob("**/*.py")
    }
    assert all_xvla <= set(reproduction.SOURCE_PATHS)
    assert "tests/test_canonical_odt.py" in reproduction.SOURCE_PATHS
    assert "pyproject.toml" in reproduction.SOURCE_PATHS
    assert "tmp/pdfs/dooms-xnets-2504.02667.pdf" in reproduction.SOURCE_PATHS
