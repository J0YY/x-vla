"""Independent, fail-closed analysis of the paired 80-episode FFN pilot."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from research.odt_reference.run_tests import COUNTS, PROHIBITED, install_guards


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("native", "fullrank", "leading_k174", "anchored_k174_s0")
SMOKE_SHA = "044d10e663805900d2e24fd0f735e54dc99fa4ab91b5c730e9c3b9f0113bfb48"
OFFLINE_SHA = "083d5f3d9e9ec1262cd4d9cbe62ec8a759e4768971bf7b5fc801bec991ffc73b"
TRAINING_SHA = "e6c07beb2efb7e5ebe55c92fabf694d97b50b9e399f50868eeef4af455e30fa7"
CACHE_SHA = "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662"
CHECKPOINT_SHA = "b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee"
CONTROLLER_SHA = "e89c2b9efe953086859a5c9f37fcbca0fef1fcf9dd9f8c301fe94ce61aaf2687"
BOOTSTRAP_SEED = 2026090704
BOOTSTRAP_REPLICATES = 4000
MAX_STEPS = 280
HORIZON = 8


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def require_hash(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or digest(path) != expected:
        raise ValueError(f"artifact identity differs: {path}")


def read_json(path, expected=None):
    if expected is not None:
        require_hash(path, expected)
    elif Path(path).is_symlink() or not Path(path).is_file():
        raise ValueError(f"regular JSON artifact required: {path}")
    return json.loads(Path(path).read_text())


def array_digest(value):
    value = np.ascontiguousarray(value)
    result = hashlib.sha256(str(value.dtype).encode() + b"\0" + json.dumps(list(value.shape)).encode() + b"\0")
    result.update(value.tobytes())
    return result.hexdigest()


def audit_analysis():
    files = (Path(__file__), Path(__file__).with_name("test_odt_rollout_summary_v1.py"),
             ROOT / "research/odt_reference/run_tests.py")
    imports = {"argparse", "ast", "hashlib", "json", "math", "numpy", "unittest"}
    modules = {"pathlib", "research.odt_reference.run_tests", "scripts", "unittest.mock"}
    hashes = {}
    for path in files:
        hashes[str(path.relative_to(ROOT))] = digest(path)
        for node in ast.walk(ast.parse(path.read_text())):
            allowed = imports | ({"subprocess", "sys", "tempfile"} if path.name.startswith("test_") else set())
            if isinstance(node, ast.Import) and any(item.name not in allowed for item in node.names):
                raise ValueError("unreviewed rollout analysis import")
            if isinstance(node, ast.ImportFrom) and (node.module not in modules or
                    (node.module == "scripts" and any(item.name != "odt_rollout_summary_v1" for item in node.names))):
                raise ValueError("unreviewed rollout analysis dependency")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in PROHIBITED | {"qr", "eigh"}:
                raise ValueError("rollout analysis cannot execute numerical factorization")
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                raise ValueError("rollout analysis cannot perform tensor contractions")
    return hashes


def check_source_hashes(sources, base):
    if not isinstance(sources, dict) or not sources:
        raise ValueError("empty source closure")
    for name, expected in sources.items():
        path = base / name
        if path.resolve() != path.absolute() or base.resolve() not in path.resolve().parents:
            raise ValueError("source path leaves its authenticated root")
        require_hash(path, expected)


def validate_sources(sources):
    local = sources["local"]
    if set(local) != {"__init__.py", "guards.py", "worker.py", "test_controller.py", "test_worker.py", "tasks.json", "controller.py"}:
        raise ValueError("rollout source inventory differs")
    if local["controller.py"] != CONTROLLER_SHA:
        raise ValueError("controller byte identity differs")
    base = ROOT / "research/odt_ffn_rollout_v1"
    check_source_hashes(local, base)
    check_source_hashes(sources["fixtures"], base / "fixtures")
    for key, version in (("native_source_audit", "v1"), ("offline_source_audit", "v2")):
        report = sources[key]
        if (report.get("scope") != "transitive_local_import_closure"
                or report.get("entrypoints") != [f"research/odt_ffn_native_{version}/native.py", f"research/odt_ffn_native_{version}/test_native.py"]
                or report.get("source_count") != len(report.get("source_sha256", {}))):
            raise ValueError("native closure schema differs")
        for field in ("prohibited_calls_found", "prohibited_self_overlap_sites",
                      "guarded_dormant_spectral_norm_sites", "duplicate_top_level_definition_sites"):
            if report.get(field) != []:
                raise ValueError("native source audit did not pass")
        check_source_hashes(report["source_sha256"], ROOT)
    return read_json(base / "tasks.json", local["tasks.json"])


def validate_guards(value, smoke=None):
    native, controller, qr = (value[key] for key in ("native", "controller", "controller_qr"))
    if (native.get("installed") is not True or native.get("profile") != "legacy87"
            or native.get("patched_entrypoint_count") != 87 or len(set(native.get("patched_entrypoints", []))) != 87
            or native.get("prohibited_attempt_count") != 0 or native.get("prohibited_attempts") != []
            or native.get("allowed_call_count") != 0 or native.get("allowed_calls") != []
            or controller.get("prohibited_calls") != 0 or controller.get("matrix_spectral_norm_calls") != 0
            or qr.get("installed") is not True or qr.get("prohibited_qr_callers") != 0
            or type(qr.get("qr_calls")) is not int or qr["qr_calls"] <= 0
            or value.get("composition") != "controllerfirst,legacy87last,privatecontroller-onlyNumPyQRwrapper"):
        raise ValueError("rollout direct-only runtime receipt differs")
    if smoke is not None:
        for key in ("native", "controller", "controller_only_verified_entrypoints", "composition"):
            if value[key] != smoke[key]:
                raise ValueError("worker guard surface differs from authenticated smoke")


def validate_counts(counts):
    integers = ("calls", "direct_qr_systems", "rhs_residual_warning_systems", "componentwise_residual_warning_systems")
    floats = ("largest_relative_residual", "largest_componentwise_backward_error", "largest_normwise_backward_error")
    if set(counts) != set(integers + floats):
        raise ValueError("controller count schema differs")
    if any(type(counts[key]) is not int or counts[key] < 0 for key in integers):
        raise ValueError("controller counts are malformed")
    if counts["calls"] <= 0 or counts["direct_qr_systems"] != 3 * counts["calls"]:
        raise ValueError("controller must use three direct QR systems per call")
    if any(type(counts[key]) not in (float, int) or not math.isfinite(counts[key]) or counts[key] < 0 for key in floats):
        raise ValueError("controller diagnostic is nonfinite or negative")
    if counts["largest_normwise_backward_error"] > 1e-12:
        raise ValueError("controller normwise gate failed")
    if any(counts[key] > counts["direct_qr_systems"] for key in integers[2:]):
        raise ValueError("controller warning count exceeds executed systems")


def validate_contract(receipt, smoke, authority, *, is_smoke=False):
    if (receipt.get("schema") != ("odt-ffn-rollout-smoke-v1" if is_smoke else "odt-ffn-rollout-worker-v1")
            or receipt.get("measurement_complete") is not True or receipt.get("offline_sha256") != OFFLINE_SHA
            or receipt.get("full_policy_odt") is not False or receipt.get("pilot_only") is not True
            or receipt.get("source_sha256") != smoke["source_sha256"]
            or receipt.get("model") != smoke["model"] or receipt["model"].get("checkpoint_sha256") != CHECKPOINT_SHA
            or receipt.get("controller") != smoke["controller"]
            or receipt.get("variants_manifest_sha256") != smoke["variants_manifest_sha256"]
            or receipt.get("environment") != smoke["environment"]):
        raise ValueError("worker and authenticated smoke contracts differ")
    environment = receipt["environment"]
    if any(environment.get(key) != expected for key, expected in authority["EXPECTED_VERSIONS"].items()):
        raise ValueError("rollout numerical environment versions differ")
    if (environment.get("tf32") is not False or environment.get("deterministic_algorithms") is not True
            or environment.get("threads") != 4 or environment.get("device") != "cuda"
            or environment.get("cublas_workspace_config") != ":4096:8"):
        raise ValueError("rollout deterministic environment contract differs")
    controller = receipt["controller"]
    if (controller.get("schema") != "xvla_modal_direct_qr_constrained_dynamics_controller_v3"
            or controller.get("method") != "direct_householder_qr_of_mass_jacobian_saddle_system"
            or controller.get("unconditional_same_factor_refinement_steps") != 2
            or controller.get("normwise_backward_error_gate") != 1e-12
            or controller.get("singular_policy") != "fail_closed_no_fallback"):
        raise ValueError("direct QR controller contract differs")
    validate_guards(receipt["guards"], None if is_smoke else smoke["guards"])


def load_training(path):
    value = read_json(path, TRAINING_SHA)
    if (value.get("cache_sha256") != CACHE_SHA or value.get("norm") not in (None, "rational")
            or value.get("suite") != "libero_object" or value.get("res") != 64 or len(value.get("vocab", {})) != 26):
        raise ValueError("fixed training auxiliary contract differs")
    normalization = {key: np.asarray(value["normalization"][key], dtype=np.float32)
                     for key in ("action_mean", "action_std", "state_mean", "state_std")}
    for key, array in normalization.items():
        if (array.shape != ((7,) if key.startswith("action") else (8,)) or not np.isfinite(array).all()
                or (key.endswith("std") and np.any(array <= 0))):
            raise ValueError("fixed normalization schema differs")
    return normalization


def validate_episode(arrays, record, normalization, authority, *, arm, task, episode):
    if (record.get("arm") != arm or type(record.get("task_index")) is not int or record["task_index"] != task
            or type(record.get("episode")) is not int or record["episode"] != episode
            or type(record.get("seed")) is not int or record["seed"] != task * 100 + episode
            or record.get("measurement_complete") is not True or record.get("failure") is not None
            or type(record.get("success")) is not bool or type(record.get("terminated_without_success")) is not bool):
        raise ValueError("episode identity or completion differs")
    steps = record["steps"]
    if type(steps) is not int or not 1 <= steps <= MAX_STEPS:
        raise ValueError("episode step count differs")
    for key in ("elapsed_s", "setup_elapsed_s"):
        if type(record.get(key)) not in (int, float) or not math.isfinite(record[key]) or record[key] <= 0:
            raise ValueError("episode timing is not finite positive")
    chunks = (steps + HORIZON - 1) // HORIZON
    schema = {"rgb": ((steps, 64, 64, 3), np.uint8), "states_raw": ((steps, 8), np.float32),
        "actions_executed": ((steps, 7), np.float32), "actions_normalized": ((steps, 7), np.float32),
        "rewards": ((steps,), np.float64), "terminals": ((steps,), np.bool_),
        "prediction_steps": ((chunks,), np.int64), "action_chunks_normalized": ((chunks, 8, 7), np.float32),
        "action_chunks_denormalized": ((chunks, 8, 7), np.float32), "executed_step_action_valid": ((steps,), np.bool_)}
    if set(arrays) != set(schema) | {"initial_state"}:
        raise ValueError("episode array inventory differs")
    for key, (shape, dtype) in schema.items():
        value = arrays[key]
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"episode array schema differs: {key}")
    if not np.all(arrays["executed_step_action_valid"]):
        raise ValueError("an executed action was invalid")
    initial = arrays["initial_state"]
    if (initial.dtype != np.float64 or initial.ndim != 1 or not initial.size or not np.isfinite(initial).all()
            or array_digest(initial) != record.get("initial_state_sha256")):
        raise ValueError("initial-state array identity differs")
    expected = authority["EXPECTED_TASK_PROTOCOL"][str(task)]
    task_protocol = record["task_protocol"]
    if (set(task_protocol) != {"task_index", "language", "bddl_sha256", "init_states_array_sha256", "packaged_file"}
            or task_protocol["task_index"] != task or task_protocol["language"] != expected[0]
            or task_protocol["bddl_sha256"] != expected[2] or task_protocol["init_states_array_sha256"] != expected[3]
            or set(task_protocol["packaged_file"]) != {"path", "sha256"}
            or not isinstance(task_protocol["packaged_file"]["path"], str)
            or not isinstance(task_protocol["packaged_file"]["sha256"], str)
            or len(task_protocol["packaged_file"]["sha256"]) != 64):
        raise ValueError("official task/initial-state protocol differs")
    if not np.array_equal(arrays["prediction_steps"], np.arange(0, steps, HORIZON, dtype=np.int64)):
        raise ValueError("policy action chunk schedule differs")
    normalized = arrays["action_chunks_normalized"]
    denormalized = normalized * normalization["action_std"] + normalization["action_mean"]
    if not np.array_equal(denormalized, arrays["action_chunks_denormalized"]):
        raise ValueError("saved chunk denormalization differs")
    expected_actions = denormalized.reshape(-1, 7)[:steps].copy()
    expected_actions[:, -1] = np.where(expected_actions[:, -1] > 0, 1., -1.)
    if (not np.array_equal(arrays["actions_executed"], expected_actions)
            or not np.array_equal(arrays["actions_normalized"], normalized.reshape(-1, 7)[:steps])):
        raise ValueError("executed action differs from its saved policy chunk")
    positive = arrays["rewards"] > 0
    terminals = arrays["terminals"]
    success = bool(np.any(positive))
    if (np.any(positive[:-1]) or np.any(terminals[:-1]) or success != record["success"]
            or bool(terminals[-1] and not success) != record["terminated_without_success"]
            or (not success and not terminals[-1] and steps != MAX_STEPS)):
        raise ValueError("success/termination/horizon differs from actual transitions")
    validate_counts(record["controller_counts"])
    return {"success": success, "steps": steps, "initial_state_sha256": array_digest(initial),
            "first_rgb_sha256": array_digest(arrays["rgb"][0]), "first_state_sha256": array_digest(arrays["states_raw"][0])}


def wilson(successes, count):
    if type(successes) is not int or type(count) is not int or not 0 <= successes <= count or count <= 0:
        raise ValueError("Wilson requires valid success counts")
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return {"lower": max(0., center - radius), "upper": min(1., center + radius)}


def bootstrap_design(repetitions=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    weights = np.zeros((repetitions, 20), dtype=np.int64)
    for task in range(10):
        draw = rng.integers(0, 2, size=(repetitions, 2))
        weights[:, task * 2] = np.sum(draw == 0, axis=1)
        weights[:, task * 2 + 1] = np.sum(draw == 1, axis=1)
    return weights


def paired(a, b, weights):
    if a.dtype != np.bool_ or b.dtype != np.bool_ or a.shape != (20,) or b.shape != (20,):
        raise ValueError("paired success requires all20 literal boolean outcomes")
    a_only, b_only = int(np.sum(a & ~b)), int(np.sum(b & ~a))
    discordant = a_only + b_only
    p_value = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1)) / (2 ** discordant))
    delta = a.astype(np.float64) - b.astype(np.float64)
    draws = np.sum(weights * delta[None, :], axis=1) / 20
    return {"a_only_success": a_only, "b_only_success": b_only, "both_success": int(np.sum(a & b)),
        "both_failure": int(np.sum(~a & ~b)), "discordant_pairs": discordant,
        "exact_mcnemar_two_sided_p": p_value, "success_difference_a_minus_b": float(np.mean(delta)),
        "paired_task_stratified_interval95": {"lower": float(np.percentile(draws, 2.5)), "upper": float(np.percentile(draws, 97.5))},
        "paired_initial_states": 20, "finite_bootstrap_replicates": len(draws)}


def protocol_expected(smoke):
    return {"mode": "worker", "source_sha256": smoke["source_sha256"], "offline_sha256": OFFLINE_SHA,
        "arms": list(ARMS), "max_steps": 280, "settle_steps": 10, "action_horizon": 8,
        "tasks": 10, "episodes_per_task": 2, "image": "rot180+PILdefaultresize64+float32/255",
        "gripper": "positive=>1,otherwise-1", "all_arms_same_controller": True, "full_policy_odt": False}


def matched_identity(result, prior=None):
    identity = {field: result[field] for field in ("initial_state_sha256", "first_rgb_sha256", "first_state_sha256")}
    if prior is not None and prior != identity:
        raise ValueError("paired arms have different initial states or first policy observations")
    return identity


def summarize(rollout, training, output):
    sources = audit_analysis()
    smoke = read_json(rollout / "smoke/receipt.json", SMOKE_SHA)
    authority = validate_sources(smoke["source_sha256"])
    validate_contract(smoke, smoke, authority, is_smoke=True)
    if (smoke.get("timing", {}).get("passed") is not True or len(smoke.get("records", [])) != 4
            or {record.get("arm") for record in smoke["records"]} != set(ARMS)
            or any(record.get("measurement_complete") is not True or record.get("failure") is not None for record in smoke["records"])):
        raise ValueError("authenticated smoke is not complete")
    for record in smoke["records"]:
        validate_counts(record["controller_counts"])
    normalization = load_training(training)
    outcomes = np.zeros((4, 20), dtype=bool)
    step_counts = np.zeros((4, 20), dtype=np.int64)
    evidence, episode_records, matching, task_protocols = {}, [], {}, {}
    maximum_normwise = 0.
    for index in range(40):
        arm_index, task = divmod(index, 10)
        arm = ARMS[arm_index]
        directory = rollout / "results" / f"task_{index}"
        receipt_path = directory / "receipt.json"
        receipt = read_json(receipt_path)
        validate_contract(receipt, smoke, authority)
        if read_json(directory / "protocol.json") != protocol_expected(smoke):
            raise ValueError("worker publication protocol differs")
        records = receipt.get("records", [])
        if len(records) != 2 or [record.get("episode") for record in records] != [0, 1]:
            raise ValueError("worker must complete both declared initial states")
        expected_files = {"receipt.json", "protocol.json"} | {
            f"{arm}_task{task}_episode{episode}{suffix}" for episode in (0, 1) for suffix in (".json", ".npz")}
        if {path.name for path in directory.iterdir()} != expected_files:
            raise ValueError("unexpected or missing worker artifacts, including failure receipts")
        evidence[f"results/task_{index}/receipt.json"] = digest(receipt_path)
        evidence[f"results/task_{index}/protocol.json"] = digest(directory / "protocol.json")
        prior_counts = None
        for episode, record in enumerate(records):
            name = f"{arm}_task{task}_episode{episode}"
            if read_json(directory / (name + ".json")) != record:
                raise ValueError("episode JSON publication differs from worker receipt")
            array_path = directory / (name + ".npz")
            require_hash(array_path, record["arrays_sha256"])
            with np.load(array_path, allow_pickle=False) as saved:
                arrays = {key: saved[key] for key in saved.files}
            result = validate_episode(arrays, record, normalization, authority, arm=arm, task=task, episode=episode)
            key = (task, episode)
            matching[key] = matched_identity(result, matching.get(key))
            if task in task_protocols and task_protocols[task] != record["task_protocol"]:
                raise ValueError("paired arms used different official task packages")
            task_protocols[task] = record["task_protocol"]
            counts = record["controller_counts"]
            if prior_counts is not None and any(counts[key] < prior_counts[key] for key in counts):
                raise ValueError("cumulative controller ledger went backwards")
            prior_counts = counts
            maximum_normwise = max(maximum_normwise, counts["largest_normwise_backward_error"])
            outcomes[arm_index, task * 2 + episode] = result["success"]
            step_counts[arm_index, task * 2 + episode] = result["steps"]
            evidence[f"results/task_{index}/{name}.npz"] = record["arrays_sha256"]
            evidence[f"results/task_{index}/{name}.json"] = digest(directory / (name + ".json"))
            episode_records.append({"arm": arm, "task_index": task, "episode": episode, **result,
                                    "controller_counts": counts, "arrays_sha256": record["arrays_sha256"]})
        if receipt["guards"]["controller_qr"]["qr_calls"] != prior_counts["direct_qr_systems"]:
            raise ValueError("controller QR wrapper count differs from executed systems")
    if len(matching) != 20 or any(matching[(task, 0)]["initial_state_sha256"] == matching[(task, 1)]["initial_state_sha256"] for task in range(10)):
        raise ValueError("pilot initial states are missing or duplicated")
    weights = bootstrap_design()
    arm_summaries, comparisons = [], []
    for index, arm in enumerate(ARMS):
        successes = int(np.sum(outcomes[index]))
        arm_summaries.append({"arm": arm, "successes": successes, "episodes": 20, "success_rate": successes / 20,
            "wilson95": wilson(successes, 20), "mean_executed_steps": float(np.mean(step_counts[index])),
            "by_task": [{"task_index": task, "successes": int(np.sum(outcomes[index, task * 2:task * 2 + 2])), "episodes": 2}
                        for task in range(10)]})
        for other in range(index):
            comparisons.append({"a": arm, "b": ARMS[other], **paired(outcomes[index], outcomes[other], weights)})
    if audit_analysis() != sources or any(COUNTS.values()):
        raise ValueError("analysis source drift or numerical kernel call")
    for name, expected in evidence.items():
        require_hash(rollout / name, expected)
    require_hash(rollout / "smoke/receipt.json", SMOKE_SHA)
    require_hash(training, TRAINING_SHA)
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "paired_episodes.npz", successes=outcomes, executed_steps=step_counts, arm_names=np.asarray(ARMS),
             task_ids=np.repeat(np.arange(10, dtype=np.int64), 2), episode_ids=np.tile(np.array([0, 1], dtype=np.int64), 10))
    report = {"schema": "odt-rollout-summary-v1", "complete": True, "episodes": 80, "paired_initial_states": 20,
        "all_measurements_complete_zero_failures": True, "arms": arm_summaries, "paired_comparisons": comparisons,
        "episode_records": episode_records, "matched_first_observations_all_arms": True,
        "largest_normwise_controller_backward_error": maximum_normwise, "controller_gate": 1e-12,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "unit": "initial-state episode",
            "strata": "task", "episodes_per_stratum": 2, "same_resampling_for_all_arms": True,
            "interval": "paired percentile95, pointwise without multiplicity adjustment"},
        "success_interval": "Wilson95 descriptive binomial interval; fixed tasks have only two initial states each",
        "mcnemar": "exact two-sided conditional binomial probability on discordant pairs, unadjusted",
        "source_sha256": sources, "rollout_source_sha256": smoke["source_sha256"], "artifact_sha256": evidence,
        "smoke_sha256": SMOKE_SHA, "offline_sha256": OFFLINE_SHA, "training_sha256": TRAINING_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA, "variants_manifest_sha256": smoke["variants_manifest_sha256"],
        "paired_arrays_sha256": digest(output / "paired_episodes.npz"), "environment": smoke["environment"],
        "controller": smoke["controller"], "analysis_kernels": dict(COUNTS), "pilot_only": True, "full_policy_odt": False,
        "limitations": "20 matched initial states, two per task, one checkpoint and one random retained-basis seed; broad sampling uncertainty; no success noninferiority or policy-training-seed claim"}
    with (output / "summary.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main():
    audit_analysis()
    install_guards()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rollout", "training", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.rollout, args.training, args.output)
    print(json.dumps({"complete": result["complete"], "episodes": result["episodes"], "kernels": result["analysis_kernels"]}))


if __name__ == "__main__":
    main()
