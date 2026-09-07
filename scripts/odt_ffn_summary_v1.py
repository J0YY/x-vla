"""Independent authenticated FFN action analysis, without numerical decomposition."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from research.odt_reference.run_tests import COUNTS, PROHIBITED, install_guards


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA = "b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee"
PRODUCER_SHA = "d95301a94a3708bdb5f77bd7c9ae2d0896b4795670f1c80c5f6cc0948e1dc790"
ACTIONS_SHA = "083d5f3d9e9ec1262cd4d9cbe62ec8a759e4768971bf7b5fc801bec991ffc73b"
TRAINING_SHA = "e6c07beb2efb7e5ebe55c92fabf694d97b50b9e399f50868eeef4af455e30fa7"
CACHE_SHA = "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662"
NATIVE_GATE = {"absolute": 2e-4, "scaled": 2e-5}
RANKS = (192, 184, 174, 154)
SEEDS = (0, 1, 2)
BOOTSTRAP_SEED = 2026090703
BOOTSTRAP_REPLICATES = 4000
METRIC_ATOL = 1e-12
METRIC_RTOL = 1e-10


def conditions():
    result = [{"name": "fullrank", "rank": 193, "seed": None, "construction": "leading"}]
    for rank in RANKS:
        result.append({"name": f"leading_k{rank}", "rank": rank, "seed": None, "construction": "leading"})
        result.extend({"name": f"anchored_k{rank}_s{seed}", "rank": rank, "seed": seed,
                       "construction": "zero_input_anchored_direct_qr"} for seed in SEEDS)
    return result


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require_hash(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or digest(path) != expected:
        raise ValueError(f"artifact identity differs: {path}")


def read_json(path, expected=None):
    if expected is not None:
        require_hash(path, expected)
    return json.loads(Path(path).read_text())


def read_arrays(path, expected):
    require_hash(path, expected)
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def audit_analysis():
    """Only array arithmetic, reductions and random resampling are admitted."""
    files = (Path(__file__), Path(__file__).with_name("test_odt_ffn_summary_v1.py"),
             ROOT / "research/odt_reference/run_tests.py")
    imports = {"argparse", "ast", "hashlib", "json", "math", "numpy", "unittest"}
    modules = {"pathlib", "research.odt_reference.run_tests", "scripts", "scripts.odt_ffn_summary_v1", "unittest.mock"}
    hashes = {}
    for path in files:
        hashes[str(path.relative_to(ROOT))] = digest(path)
        for node in ast.walk(ast.parse(path.read_text())):
            allowed = imports | ({"subprocess", "sys", "tempfile"} if path.name.startswith("test_") else set())
            if isinstance(node, ast.Import) and any(x.name not in allowed for x in node.names):
                raise ValueError("unreviewed FFN analysis import")
            if isinstance(node, ast.ImportFrom) and node.module not in modules:
                raise ValueError("unreviewed FFN analysis dependency")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in PROHIBITED | {"qr", "eigh"}:
                raise ValueError("FFN analysis must not execute a numerical factorization")
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                raise ValueError("FFN analysis must not perform tensor contractions")
    return hashes


def verify(actual, expected, label="metric"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"receipt schema differs: {label}")
        for key in expected:
            verify(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"receipt list differs: {label}")
        for index, item in enumerate(expected):
            verify(actual[index], item, f"{label}[{index}]")
    elif isinstance(expected, float):
        if (type(actual) not in (int, float) or not math.isfinite(actual)
                or abs(actual - expected) > METRIC_ATOL + METRIC_RTOL * abs(expected)):
            raise ValueError(f"receipt value differs: {label}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"receipt exact value differs: {label}")


def array_hash(value):
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)},
                        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result = hashlib.sha256(header)
    result.update(np.ascontiguousarray(value).tobytes())
    return result.hexdigest()


def check_sources(sources, base=ROOT):
    if not isinstance(sources, dict) or not sources:
        raise ValueError("empty source closure")
    for name, expected in sources.items():
        path = base / name
        if path.resolve() != path.absolute() or base.resolve() not in path.resolve().parents:
            raise ValueError("source path leaves its authenticated root")
        require_hash(path, expected)


def validate_source_report(report):
    if (report.get("scope") != "transitive_local_import_closure"
            or report.get("entrypoints") != ["research/odt_ffn_native_v2/native.py", "research/odt_ffn_native_v2/test_native.py"]
            or report.get("source_count") != len(report.get("source_sha256", {}))
            or report.get("direct_qr_required") is not False or report.get("direct_qr_call_sites") != 0):
        raise ValueError("native v2 source contract differs")
    for key in ("prohibited_calls_found", "prohibited_self_overlap_sites",
                "guarded_dormant_spectral_norm_sites", "duplicate_top_level_definition_sites"):
        if report.get(key) != []:
            raise ValueError("native v2 source audit did not pass")
    check_sources(report["source_sha256"])


def validate_ids(arrays, records, *, prefix="", count=160):
    ids = []
    for field in ("task", "episode", "frame"):
        value = arrays[prefix + f"image_{field}_ids"]
        if value.dtype != np.int64 or value.shape != (count,):
            raise ValueError("frame identity array schema differs")
        ids.append(value)
    rows = list(zip(*(value.tolist() for value in ids)))
    if len(set(rows)) != count or len(records) != 2 * count:
        raise ValueError("duplicate or missing frame identities")
    observed = set()
    for row in records:
        if any(type(row.get(key)) is not int for key in ("image_index", "task_id", "episode_id", "frame_id", "pair_id")):
            raise ValueError("panel record identity schema differs")
        index, pair = row["image_index"], row["pair_id"]
        if not 0 <= index < count or pair not in (0, 1) or (index, pair) in observed:
            raise ValueError("panel pair index differs")
        if tuple(row[key] for key in ("task_id", "episode_id", "frame_id")) != rows[index]:
            raise ValueError("panel JSON and frame IDs disagree")
        if row["pair_positions"] != [[27, 28], [0, 63]][pair]:
            raise ValueError("frozen pair positions differ")
        observed.add((index, pair))
    clusters = {(task, episode) for task, episode, _ in rows}
    for task in range(10):
        selected = [key for key in clusters if key[0] == task]
        if len(selected) != count // 20 or any(sum((t, e) == key for t, e, _ in rows) != 2 for key in selected):
            raise ValueError("panel must contain two frames per episode with balanced tasks")
    if len(clusters) != count // 2:
        raise ValueError("unexpected panel task or episode")
    return rows, clusters


def load_panel(panel_path, metadata_path, protocol):
    metadata = read_json(metadata_path, protocol["panel_metadata_sha256"])
    if (metadata.get("schema") != "odt-real-panel-v1" or metadata.get("passed") is not True
            or metadata.get("checkpoint_sha256") != CHECKPOINT_SHA
            or metadata.get("training_sha256") != TRAINING_SHA or metadata.get("cache_sha256") != CACHE_SHA
            or metadata.get("panel_sha256") != protocol["panel_sha256"]
            or metadata.get("freeze_sha256") != protocol["freeze_sha256"]):
        raise ValueError("panel provenance differs")
    freeze = read_json(metadata_path.parent / "panel_freeze.json", metadata["freeze_sha256"])
    for key in ("records", "development_records"):
        if metadata[key] != freeze[key]:
            raise ValueError("panel was not frozen before native outputs")
    panel = read_arrays(panel_path, metadata["panel_sha256"])
    if set(panel) != set(metadata["arrays"]):
        raise ValueError("panel array inventory differs")
    for name, value in panel.items():
        expected = {"shape": list(value.shape), "dtype": value.dtype.str, "sha256": array_hash(value)}
        if value.dtype.hasobject or metadata["arrays"][name] != expected:
            raise ValueError("panel array content contract differs")
    rows, confirmation = validate_ids(panel, metadata["records"])
    _, development = validate_ids(panel, metadata["development_records"], prefix="dev_", count=40)
    if confirmation & development:
        raise ValueError("development and confirmation episodes overlap")
    for key in ("normalization__action_mean", "normalization__action_std"):
        value = panel[key]
        if value.dtype != np.float32 or value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError("native action normalization differs")
    if np.any(panel["normalization__action_std"] <= 0):
        raise ValueError("action scale is not positive")
    return panel, rows


def validate_prediction(values, panel, *, prefix="", count=160, confirmation=True):
    required = {"actions_normalized", "actions", "stage_output", "valid", "attempted", "status_code", "stage_token_valid", "latency_s_per_frame"}
    if confirmation:
        required |= {"action_delta", "gripper_sign_disagreement", "task_ids", "episode_ids", "frame_ids"}
    if set(values) != required:
        raise ValueError("action result array inventory differs")
    for key, shape, dtype in (("actions", (count, 8, 7), np.float32),
            ("actions_normalized", (count, 8, 7), np.float32), ("stage_output", (count, 64, 192), np.float32),
            ("latency_s_per_frame", (count,), np.float64)):
        value = values[key]
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"valid action/stage array schema differs: {key}")
    for key, shape, dtype in (("valid", (count,), np.bool_), ("attempted", (count,), np.bool_),
            ("stage_token_valid", (count, 64), np.bool_), ("status_code", (count,), np.int8)):
        if values[key].shape != shape or values[key].dtype != dtype or not np.all(values[key] == 1):
            raise ValueError("strict complete zero-invalid FFN analysis admission failed")
    if np.any(values["latency_s_per_frame"] < 0):
        raise ValueError("negative execution latency")
    denormalized = values["actions_normalized"] * panel["normalization__action_std"] + panel["normalization__action_mean"]
    if not np.array_equal(denormalized, values["actions"]):
        raise ValueError("action denormalization differs")
    if confirmation:
        for field in ("task", "episode", "frame"):
            ids = values[field + "_ids"]
            if ids.dtype != np.int64 or not np.array_equal(ids, panel["image_" + field + "_ids"]):
                raise ValueError("result frame identities differ from frozen panel")


def parity(actual, reference):
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("parity requires finite equal shapes")
    maximum = float(np.max(np.abs(actual.astype(np.float64) - reference.astype(np.float64))))
    scale = max(1., float(np.max(np.abs(reference.astype(np.float64)))))
    return {"passed": maximum <= NATIVE_GATE["absolute"] and maximum / scale <= NATIVE_GATE["scaled"],
            "finite": True, "maximum_absolute_error": maximum, "scaled_error": maximum / scale,
            "scale": scale, "tolerance": NATIVE_GATE}


def validate_double_receipt(value):
    if not isinstance(value, dict) or set(value) != {"passed", "finite", "maximum_absolute_error", "scaled_error", "scale", "tolerance"}:
        raise ValueError("double parity receipt schema differs")
    maximum, scaled, scale = (value[key] for key in ("maximum_absolute_error", "scaled_error", "scale"))
    if (any(type(x) not in (float, int) or not math.isfinite(x) for x in (maximum, scaled, scale))
            or maximum < 0 or scaled < 0 or scale < 1e-12 or scaled > 1e-10
            or value["passed"] is not True or value["finite"] is not True or value["tolerance"] != 1e-10):
        raise ValueError("double parity receipt did not pass")
    verify(scaled, maximum / scale, "double scaled error")


def recompute(result, reference):
    """Independent array reductions reproduce the accepted zero-invalid receipt."""
    count = len(result["actions"])
    delta = result["actions"].astype(np.float64) - reference["actions"].astype(np.float64)
    disagreement = np.sign(result["actions"][:, :, -1]) != np.sign(reference["actions"][:, :, -1])
    metrics = {"frames": count, "attempted_invalid_frames": 0, "execution_failed_frames": 0,
        "denominator_invalid_frames": 0, "batch_aborted_frames": 0, "attempted_unavailable_action_frames": 0,
        "not_attempted_frames": 0, "attempted_frames": count, "validity_measurement_complete": True,
        "incomplete": False, "invalid_probability_if_fully_attempted": 0., "jointly_valid_frames": count,
        "error_statistics_conditional_on_joint_validity": True}
    for label, part in (("translation", slice(0, 3)), ("rotation", slice(3, 6)), ("gripper", slice(6, 7)), ("all", slice(0, 7))):
        value = delta[:, :, part]
        frame_rmse = np.sqrt(np.mean(value * value, axis=(1, 2)))
        metrics[label] = {"rmse": float(np.sqrt(np.mean(value * value))),
            "p95_frame_rmse": float(np.percentile(frame_rmse, 95)), "maximum_absolute_error": float(np.max(np.abs(value)))}
    metrics["gripper_sign_disagreement_probability"] = float(np.mean(disagreement))
    return metrics, delta, disagreement


def descriptive(result, reference):
    metrics, delta, _ = recompute(result, reference)
    stage = result["stage_output"].astype(np.float64) - reference["stage_output"].astype(np.float64)
    metrics.update({"action_mse_all7": float(np.mean(delta * delta)),
        "per_component_action_rmse": np.sqrt(np.mean(delta * delta, axis=(0, 1))).tolist(),
        "executed_gripper_sign_disagreement_probability": float(np.mean(
            (result["actions"][:, :, -1] > 0) != (reference["actions"][:, :, -1] > 0))),
        "stage_output_rmse": float(np.sqrt(np.mean(stage * stage))),
        "stage_output_maximum_absolute_error": float(np.max(np.abs(stage)))})
    return metrics, np.mean(delta * delta, axis=(1, 2))


def bootstrap_design(rows, repetitions=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    keys = sorted({(task, episode) for task, episode, _ in rows})
    if len(rows) != 160 or len(keys) != 80:
        raise ValueError("bootstrap requires 160 frames in 80 episodes")
    for task in range(10):
        task_keys = [key for key in keys if key[0] == task]
        if len(task_keys) != 8 or any(sum((t, e) == key for t, e, _ in rows) != 2 for key in task_keys):
            raise ValueError("bootstrap requires eight episodes per task")
    mapping = {key: index for index, key in enumerate(keys)}
    cluster = np.array([mapping[(task, episode)] for task, episode, _ in rows], dtype=np.int64)
    weights = np.zeros((repetitions, 80), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for task in range(10):
        positions = [index for index, key in enumerate(keys) if key[0] == task]
        draw = rng.integers(0, 8, size=(repetitions, 8))
        for column, index in enumerate(positions):
            weights[:, index] = np.sum(draw == column, axis=1)
    return cluster, weights


def paired_bootstrap(a, b, design):
    cluster, weights = design
    if a.shape != (160,) or b.shape != (160,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired endpoint requires every finite frame")
    delta = a - b
    episode_mean = np.bincount(cluster, weights=delta, minlength=80) / 2
    samples = np.sum(weights * episode_mean[None, :], axis=1) / 80
    return {"action_mse_difference_leading_minus_anchored": float(np.mean(delta)),
        "interval95": {"lower": float(np.percentile(samples, 2.5)), "upper": float(np.percentile(samples, 97.5))},
        "jointly_valid_frames": 160, "episode_clusters": 80, "finite_bootstrap_replicates": len(samples)}


def summarize(actions, panel_path, metadata_path, variants, output):
    sources = audit_analysis()
    manifest = read_json(actions / "manifest.json", ACTIONS_SHA)
    names = ["native"] + [item["name"] for item in conditions()]
    if (manifest.get("schema") != "odt-ffn-native-actions-v2" or manifest.get("failure_accounting_version") != 2
            or manifest.get("workflow_completed") is not True or manifest.get("all_arms_completed") is not True
            or manifest.get("failed_or_stopped_arms") != [] or manifest.get("not_attempted_arms") != []
            or set(manifest.get("outcomes", {})) != set(names) or manifest["model"]["checkpoint_sha256"] != CHECKPOINT_SHA
            or manifest.get("simulator_launched") is not False or manifest.get("full_policy_odt") is not False):
        raise ValueError("strict complete native v2 manifest admission failed")
    validate_source_report(manifest["source_sha256"])
    runtime = manifest["runtime_guard"]
    if (runtime.get("installed") is not True or runtime.get("profile") != "legacy87"
            or runtime.get("patched_entrypoint_count") != 87 or len(set(runtime.get("patched_entrypoints", []))) != 87
            or runtime.get("prohibited_attempt_count") != 0 or runtime.get("prohibited_attempts") != []
            or runtime.get("allowed_call_count") != 0 or runtime.get("allowed_calls") != []):
        raise ValueError("native action runtime guard receipt differs")
    if manifest.get("status_codes") != {"not_attempted": 0, "valid": 1, "denominator_invalid": 2, "execution_failure": 3, "batch_aborted": 4}:
        raise ValueError("native action status schema differs")
    publication = read_json(actions / "protocol.json")
    for key in ("variants_manifest_sha256", "panel_sha256", "source_sha256"):
        if publication.get(key) != manifest[key]:
            raise ValueError("native action publication binding differs")
    if (publication.get("failure_accounting_version") != 2 or publication.get("conditions") != names
            or publication.get("score_all160_confirmation_frames") is not True
            or publication.get("native_action_gate") != NATIVE_GATE or publication.get("double_gate") != 1e-10):
        raise ValueError("native action publication contract differs")
    variant_manifest = read_json(variants / "manifest.json", manifest["variants_manifest_sha256"])
    if (variant_manifest.get("ready") is not True or variant_manifest.get("schema") != "odt-ffn-native-variants-v1"
            or set(variant_manifest.get("artifacts", {})) != set(names[1:])
            or variant_manifest.get("kernels", {}).get("prohibited_attempts") != 0):
        raise ValueError("physical variants are not accepted")
    protocol = read_json(variants / "protocol.json", variant_manifest["protocol_sha256"])
    if (protocol.get("schema") != "odt-ffn-native-protocol-v1" or protocol.get("conditions") != conditions()
            or protocol.get("receipt_sha256") != PRODUCER_SHA or protocol.get("checkpoint_sha256") != CHECKPOINT_SHA
            or protocol.get("native_action_gate") != NATIVE_GATE or protocol.get("double_gate") != 1e-10
            or protocol.get("cut_node") != 4 or protocol.get("confirmation_frames") != 160
            or protocol.get("development_frames") != 40 or protocol.get("panel_sha256") != manifest["panel_sha256"]):
        raise ValueError("physical variant protocol differs")
    source = protocol["source_sha256"]
    check_sources(source["bridge_reference"], ROOT / "research/odt_ffn_native_v1")
    check_sources(source["producer"]["local"], ROOT / "research/odt_ffn_v1")
    check_sources(source["producer"]["reference"], ROOT / "research/odt_reference")
    require_hash(variants / "dev_inputs.npz", variant_manifest["dev_inputs_sha256"])
    for condition in conditions():
        artifact = variant_manifest["artifacts"][condition["name"]]
        directory = variants / condition["name"]
        for filename, expected in (("graph.json", artifact["graph_hashes"]["graph_sha256"]),
                ("arrays.npz", artifact["graph_hashes"]["arrays_sha256"]), ("dev_reference.npz", artifact["dev_reference_sha256"])):
            require_hash(directory / filename, expected)
        if artifact["removed_dimensions"] != 193 - condition["rank"] or artifact["eligible_dimensions"] != 390:
            raise ValueError("single-cut rank budget differs")
    panel, rows = load_panel(panel_path, metadata_path, protocol)
    loaded, development, summaries, frame_mses = {}, {}, [], []
    for name in names:
        receipt = manifest["outcomes"][name]
        if (read_json(actions / (name + ".json")) != receipt or receipt.get("completed") is not True
                or receipt.get("failure") is not None
                or receipt.get("compared_against") != ("native" if name in ("native", "fullrank") else "fullrank")):
            raise ValueError("arm publication/reference differs")
        values = read_arrays(actions / (name + ".npz"), receipt["array_sha256"])
        dev = read_arrays(actions / (name + "_development.npz"), receipt["development_sha256"])
        validate_prediction(values, panel)
        validate_prediction(dev, panel, prefix="dev_", count=40, confirmation=False)
        loaded[name], development[name] = values, dev
        reference = loaded[receipt["compared_against"]]
        measured, delta, disagreement = recompute(values, reference)
        verify(receipt["metrics"], measured, name)
        if (values["action_delta"].dtype != np.float64 or values["action_delta"].shape != delta.shape
                or not np.array_equal(values["action_delta"], delta)
                or values["gripper_sign_disagreement"].dtype != np.bool_
                or not np.array_equal(values["gripper_sign_disagreement"], disagreement)):
            raise ValueError("stored per-frame action errors differ")
        verify(receipt["mean_latency_s_per_frame"], float(np.mean(values["latency_s_per_frame"])), "latency")
        if name != "native":
            validate_double_receipt(receipt["double_dev_parity"])
        if name in ("native", "fullrank"):
            for key, actual, expected in (
                    ("native_action_dev_parity", dev["actions_normalized"], development["native"]["actions_normalized"]),
                    ("native_denormalized_action_dev_parity", dev["actions"], development["native"]["actions"]),
                    ("native_action_confirmation_parity", values["actions_normalized"],
                     panel["image_actions_normalized"] if name == "native" else loaded["native"]["actions_normalized"])):
                gate = parity(actual, expected)
                if not gate["passed"]:
                    raise ValueError("independently recomputed native parity failed")
                verify(receipt[key], gate, key)
            if name == "fullrank":
                gate = parity(values["actions"], loaded["native"]["actions"])
                if not gate["passed"]:
                    raise ValueError("fullrank denormalized action parity failed")
                verify(receipt["native_denormalized_confirmation_parity"], gate)
                validate_double_receipt(receipt["fullrank_source_double_parity"])
    for prefix, values in (("", loaded["native"]), ("dev_", development["native"])):
        for key, panel_key in (("actions_normalized", "image_actions_normalized"), ("actions", "image_actions"), ("stage_output", "image_output64")):
            if not parity(values[key], panel[prefix + panel_key])["passed"]:
                raise ValueError("native replay versus captured panel differs")
    for name in names:
        primary, mse = descriptive(loaded[name], loaded["fullrank"])
        native, _ = descriptive(loaded[name], loaded["native"])
        frame_mses.append(mse)
        task_metrics = [{"task_id": task, "frames": 16, "episodes": 8,
                         "action_mse_all7_vs_fullrank": float(np.mean(mse[panel["image_task_ids"] == task]))}
                        for task in range(10)]
        summaries.append({"name": name, "primary_reference": "fullrank", "versus_fullrank": primary,
                          "versus_native_descriptive": native, "by_task": task_metrics})
    design = bootstrap_design(rows)
    mse_by_name = dict(zip(names, frame_mses))
    contrasts, seed_summaries = [], []
    for rank in RANKS:
        basis_values = []
        for seed in SEEDS:
            leading, anchored = f"leading_k{rank}", f"anchored_k{rank}_s{seed}"
            contrast = paired_bootstrap(mse_by_name[leading], mse_by_name[anchored], design)
            contrasts.append({"leading": leading, "anchored": anchored, "rank": rank, "basis_seed": seed,
                              "reference": "fullrank", **contrast})
            basis_values.append(float(np.mean(mse_by_name[anchored])))
        seed_summaries.append({"rank": rank, "per_basis_seed_action_mse_vs_fullrank": dict(zip(map(str, SEEDS), basis_values)),
            "arithmetic_mean": float(np.mean(basis_values)), "minimum": min(basis_values), "maximum": max(basis_values),
            "scope": "three random retained-basis seeds, not policy training seeds"})
    if audit_analysis() != sources or any(COUNTS.values()):
        raise ValueError("analysis source drift or nonzero numerical kernel ledger")
    require_hash(actions / "manifest.json", ACTIONS_SHA)
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "per_frame.npz", action_mse_vs_fullrank=np.asarray(frame_mses),
             arm_names=np.asarray(names), task_ids=panel["image_task_ids"], episode_ids=panel["image_episode_ids"], frame_ids=panel["image_frame_ids"])
    report = {"schema": "odt-ffn-summary-v1", "complete": True, "arms": summaries, "contrasts": contrasts,
        "basis_seed_descriptive": seed_summaries, "frames": 160, "episodes": 80, "tasks": 10,
        "all_18_arms_complete_zero_invalid": True, "primary_endpoint": "mean squared denormalized all7 action error across8 chunk positions versus fullrank replacement",
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
            "unit": "episode", "strata": "task", "interval": "paired percentile95, pointwise without multiplicity adjustment",
            "same_resampled_episodes_for_all_arms": True},
        "source_sha256": sources, "actions_manifest_sha256": ACTIONS_SHA, "producer_receipt_sha256": PRODUCER_SHA,
        "variants_manifest_sha256": manifest["variants_manifest_sha256"], "panel_sha256": protocol["panel_sha256"],
        "panel_metadata_sha256": protocol["panel_metadata_sha256"], "checkpoint_sha256": CHECKPOINT_SHA,
        "per_frame_sha256": digest(output / "per_frame.npz"), "kernels": dict(COUNTS),
        "receipt_comparison_tolerance": {"absolute": METRIC_ATOL, "relative": METRIC_RTOL},
        "double_gate_scope": "authenticated accepted receipts checked algebraically; GPU projective outputs are not re-executed by analysis",
        "simulator_launched": False, "full_policy_odt": False,
        "interpretation": "offline local first-block FFN intervention; no closed-loop capability or training-seed inference"}
    with (output / "summary.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main():
    audit_analysis()
    install_guards()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("actions", "panel", "panel-metadata", "variants", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.actions, args.panel, args.panel_metadata, args.variants, args.output)
    print(json.dumps({"complete": report["complete"], "arms": len(report["arms"]), "kernels": report["kernels"]}))


if __name__ == "__main__":
    main()
