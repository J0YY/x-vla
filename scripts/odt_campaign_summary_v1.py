"""Independent, hash-checked paired analysis of the frozen real-input campaign."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from research.odt_reference.run_tests import install_guards, COUNTS, PROHIBITED


BOOTSTRAP_SEED = 2026090702
BOOTSTRAP_REPLICATES = 4000
METRIC_ATOL = 1e-12
METRIC_RTOL = 1e-10
VERSION = "odt-campaign-v1"
PRODUCER_RECEIPT_SHA = "bc961a3d700e94a15341aa2700eb85ca638cda00d80a2c0e7f6d803b14753a3d"
CHECKPOINT_SHA = "b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee"
ARTIFACTS = {"protocol.json", "inputs.npz", "records.json", "bases.npz", "spectral_diagnostics.json",
             "anchor_diagnostics.json", "baseline.npz"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def authenticated_json(path, expected=None):
    if expected is not None and digest(path) != expected:
        raise ValueError(f"artifact identity differs: {path}")
    return json.loads(Path(path).read_text())


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def audit_analysis():
    """Audit this independent analysis and its tests before numerical setup."""
    here = Path(__file__).resolve()
    files = (here, here.with_name("test_odt_campaign_summary_v1.py"),
             here.parents[1]/"research/odt_reference/run_tests.py")
    imports = {"argparse", "ast", "hashlib", "json", "math", "numpy", "unittest"}
    hashes = {}
    for path in files:
        hashes[str(path.relative_to(here.parents[1]))] = digest(path)
        for node in ast.walk(ast.parse(path.read_text())):
            allowed = imports | ({"subprocess", "sys", "tempfile"} if path.name.startswith("test_") else set())
            if isinstance(node, ast.Import) and any(name.name not in allowed for name in node.names):
                raise ValueError("unreviewed analysis import")
            if isinstance(node, ast.ImportFrom) and node.module not in {
                    "pathlib", "research.odt_reference.run_tests", "scripts.odt_campaign_summary_v1"}:
                raise ValueError("unreviewed analysis dependency")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in PROHIBITED:
                raise ValueError("prohibited analysis numerical route")
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                for a, b in ((node.left, node.right), (node.right, node.left)):
                    if isinstance(b, ast.Attribute) and b.attr == "T" and ast.dump(a) == ast.dump(b.value):
                        raise ValueError("prohibited analysis self-overlap")
    return hashes


def validate_predictions(values, *, shape=(576, 384), baseline=False):
    """No invalid-row imputation or unchecked standalone/partial conditions."""
    decoded, valid = values["decoded"], values["valid"]
    if (decoded.dtype != np.float64 or decoded.shape != shape
            or valid.dtype != np.bool_ or valid.shape != (shape[0],)):
        raise ValueError("prediction decoded/valid schema differs")
    if not np.isfinite(decoded[valid]).all() or not np.isnan(decoded[~valid]).all():
        raise ValueError("valid predictions must be finite, invalid predictions must remain NaN")
    projective, code, margin = (values[key] for key in ("projective", "failure_code", "relative_denominator"))
    if (projective.dtype != np.float64 or projective.shape != (shape[0], shape[1]+1)
            or not np.isfinite(projective).all() or code.dtype != np.int8 or code.shape != valid.shape
            or margin.dtype != np.float64 or margin.shape != valid.shape or not np.isfinite(margin).all()
            or not np.all(np.isin(code, [0, 1, 2])) or not np.array_equal(valid, code == 0)):
        raise ValueError("prediction projective/validity schema differs")
    if not np.array_equal(margin, np.abs(projective[:, -1])):
        raise ValueError("stored chart margin differs from projective denominator")
    size = np.max(np.abs(projective), axis=1)
    expected_code = np.where(size == 0, 2, np.where(margin <= 1e-12, 1, 0)).astype(np.int8)
    if (not np.array_equal(code, expected_code) or np.any((size != 0) & (size != 1))
            or np.any((margin < 0) | (margin > 1))):
        raise ValueError("stored chart classification differs from frozen gate")
    if np.any(valid):
        quotient = projective[valid, :-1]/projective[valid, -1:]
        if not np.isfinite(quotient).all() or not np.allclose(decoded[valid], quotient, rtol=METRIC_RTOL, atol=METRIC_ATOL):
            raise ValueError("decoded predictions differ from projective quotient")
    if baseline:
        if not np.all(valid):
            raise ValueError("full-rank baseline contains invalid rows")
        expected = values["expected"]
        if expected.dtype != np.float64 or expected.shape != shape or not np.isfinite(expected).all():
            raise ValueError("independent full-rank expected schema differs")
        error = float(np.max(np.abs(decoded-expected)))/max(float(np.max(np.abs(expected))), 1e-12)
        if not math.isfinite(error) or error > 1e-10:
            raise ValueError("independent full-rank replay gate differs")


def recompute_metrics(predictions, baseline, selection):
    """Compute endpoints from row arrays, independent of consumer summaries."""
    valid, reference_valid = predictions["valid"], baseline["valid"]
    chosen = np.flatnonzero(selection)
    if not len(chosen):
        raise ValueError("nonempty metric selection required")
    paired = chosen[valid[chosen] & reference_valid[chosen]]
    difference = predictions["decoded"][paired]-baseline["decoded"][paired]
    with np.errstate(over="raise", invalid="raise"):
        mse = np.sum(difference*difference, axis=1)/predictions["decoded"].shape[1]
        rms = float(np.sqrt(np.sum(mse)/len(paired))) if len(paired) else None
        source = baseline["decoded"][chosen[reference_valid[chosen]]]
        source_rms = float(np.sqrt(np.sum(source*source)/source.size)) if source.size else None
    return {"input_count": len(chosen), "invalid_count": int(np.count_nonzero(~valid[chosen])),
            "invalid_probability_unconditional": float(np.count_nonzero(~valid[chosen])/len(chosen)),
            "reference_invalid_count": int(np.count_nonzero(~reference_valid[chosen])),
            "jointly_valid_count": len(paired), "decoded_errors_conditional_on_joint_validity": True,
            "rmse": rms, "source_output_rms": source_rms,
            "rmse_over_source_rms": rms/source_rms if rms is not None and source_rms else None,
            "p95_per_example_rmse": float(np.percentile(np.sqrt(mse), 95)) if len(mse) else None,
            "maximum_per_example_rmse": float(np.max(np.sqrt(mse))) if len(mse) else None,
            "maximum_absolute_error": float(np.max(np.abs(difference))) if difference.size else None}


def recompute_all_metrics(predictions, baseline, records):
    metrics = {"all": recompute_metrics(predictions, baseline, np.ones(len(records), dtype=bool))}
    for name in ("real", "gaussian"):
        metrics[name] = recompute_metrics(predictions, baseline, np.array([r["panel"] == name for r in records]))
    tasks = []
    for task_id in sorted({r["task_id"] for r in records if r["panel"] == "real"}):
        selection = np.array([r["panel"] == "real" and r["task_id"] == task_id for r in records])
        tasks.append({"task_id": task_id, **recompute_metrics(predictions, baseline, selection)})
    return metrics, tasks


def verify_metric_receipt(actual, expected, *, label="metrics"):
    """Counts, flags, absent endpoints and keys are exact; floats are tolerant."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"receipt metric schema differs: {label}")
        for key in expected:
            verify_metric_receipt(actual[key], expected[key], label=f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"receipt metric list differs: {label}")
        for i, value in enumerate(expected):
            verify_metric_receipt(actual[i], value, label=f"{label}[{i}]")
    elif isinstance(expected, float):
        if (type(actual) not in (int, float) or not math.isfinite(actual)
                or abs(actual-expected) > METRIC_ATOL+METRIC_RTOL*abs(expected)):
            raise ValueError(f"receipt metric value differs: {label}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"receipt metric exact value differs: {label}")


def validate_records(records):
    if len(records) != 576 or any(r.get("panel") != "real" for r in records[:320]) or any(
            r.get("panel") != "gaussian" for r in records[320:]):
        raise ValueError("frozen real/Gaussian panel row ordering differs")
    episodes = {}
    for row in records[:320]:
        if any(type(row.get(key)) is not int for key in ("task_id", "episode_id", "frame_id", "pair_id")):
            raise ValueError("integer real panel identity required")
        key = (row["task_id"], row["episode_id"])
        pair = (row["frame_id"], row["pair_id"])
        if pair in episodes.setdefault(key, set()):
            raise ValueError("duplicate real panel row identity")
        episodes[key].add(pair)
    for task_id in range(10):
        keys = [key for key in episodes if key[0] == task_id]
        if len(keys) != 8:
            raise ValueError("task-stratified confirmation episode balance differs")
        for key in keys:
            frames = {frame for frame, _ in episodes[key]}
            if len(frames) != 2 or episodes[key] != {(f, p) for f in frames for p in (0, 1)}:
                raise ValueError("episode must retain its two frames and both pairs")
    if len(episodes) != 80:
        raise ValueError("unexpected real task or episode")
    if any(row.get("task_id") != "gaussian" or any(row.get(key) != i for key in
            ("episode_id", "frame_id", "pair_id")) for i, row in enumerate(records[320:])):
        raise ValueError("frozen Gaussian row identities differ")


def descriptive_seed_summaries(conditions):
    groups = {}
    for item in conditions:
        condition = item["condition"]
        if condition["construction"] != "zero_input_anchored_direct_qr":
            continue
        key = condition["rank_sha256"]
        group = groups.setdefault(key, {"rank_sha256": key, "aliases": condition["aliases"], "seeds": {}})
        if condition["seed"] in group["seeds"]:
            raise ValueError("duplicate retained-space seed for one rank tuple")
        group["seeds"][condition["seed"]] = item
    summaries = []
    for key, group in sorted(groups.items()):
        available = sorted(group["seeds"])
        complete = available == [0, 1, 2]
        panels = {}
        for panel in ("real", "gaussian"):
            entries = {}
            for metric in ("rmse", "rmse_over_source_rms", "invalid_probability_unconditional", "jointly_valid_count"):
                values = [group["seeds"][seed]["metrics"][panel][metric] for seed in available]
                defined = all(value is not None for value in values)
                entries[metric] = {"per_basis_seed": {str(seed): value for seed, value in zip(available, values)},
                    "arithmetic_mean": float(sum(values)/3) if complete and defined else None,
                    "minimum": min(values) if complete and defined else None,
                    "maximum": max(values) if complete and defined else None,
                    "defined_seed_count": sum(value is not None for value in values)}
            panels[panel] = entries
        summaries.append({"rank_sha256": key, "aliases": group["aliases"], "available_seeds": available,
                          "complete_three_basis_seeds": complete, "panels": panels,
                          "scope": "descriptive random retained-basis seeds, not policy-training seeds",
                          "rmse_interpretation": "each seed's RMSE is conditional on that seed's joint validity with the source; rows are not pooled"})
    return summaries


def cluster_design(records, *, repetitions=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """Resample episodes within each fixed task, carrying every related row."""
    real_indices = np.array([i for i, row in enumerate(records) if row["panel"] == "real"], dtype=int)
    keys = sorted({(records[i]["task_id"], records[i]["episode_id"]) for i in real_indices})
    if not keys or repetitions < 1:
        raise ValueError("nonempty real episode design and positive replicates required")
    index = {key: i for i, key in enumerate(keys)}
    row_clusters = np.array([index[(records[i]["task_id"], records[i]["episode_id"])] for i in real_indices])
    rng = np.random.default_rng(seed)
    weights = np.zeros((repetitions, len(keys)), dtype=np.int64)
    for task in sorted({key[0] for key in keys}):
        positions = np.array([i for i, key in enumerate(keys) if key[0] == task])
        draws = rng.integers(0, len(positions), size=(repetitions, len(positions)))
        for column, position in enumerate(positions):
            weights[:, position] = np.sum(draws == column, axis=1)
    return real_indices, row_clusters, keys, weights


def interval(samples):
    finite = np.asarray(samples)[np.isfinite(samples)]
    return {"lower": float(np.percentile(finite, 2.5)) if len(finite) else None,
            "upper": float(np.percentile(finite, 97.5)) if len(finite) else None,
            "finite_bootstrap_replicates": len(finite), "total_bootstrap_replicates": len(samples)}


def paired_comparison(a, b, baseline, design):
    indices, row_cluster, keys, weights = design
    av, bv, rv = (value["valid"][indices] for value in (a, b, baseline))
    joint = av & bv & rv
    ad, bd, rd = (value["decoded"][indices] for value in (a, b, baseline))
    if (not np.isfinite(ad[av]).all() or not np.isfinite(bd[bv]).all()
            or not np.isfinite(rd[rv]).all()):
        raise ValueError("valid rows contain nonfinite predictions")
    delta = np.zeros(len(indices))
    delta[joint] = np.mean((ad[joint] - rd[joint])**2, axis=1) - np.mean((bd[joint] - rd[joint])**2, axis=1)
    sums = np.bincount(row_cluster, weights=delta, minlength=len(keys))
    counts = np.bincount(row_cluster, weights=joint.astype(float), minlength=len(keys))
    sample_counts = weights @ counts
    samples = np.full(len(weights), np.nan)
    usable = sample_counts > 0
    samples[usable] = (weights[usable] @ sums) / sample_counts[usable]
    invalid_delta = (~av).astype(float) - (~bv).astype(float)
    invalid_sums = np.bincount(row_cluster, weights=invalid_delta, minlength=len(keys))
    all_counts = np.bincount(row_cluster, minlength=len(keys))
    invalid_samples = (weights @ invalid_sums) / (weights @ all_counts)
    return {"real_input_count": len(indices), "episode_clusters": len(keys),
            "jointly_valid_count": int(np.sum(joint)),
            "a_invalid_count": int(np.sum(~av)), "b_invalid_count": int(np.sum(~bv)),
            "mse_difference_a_minus_b_conditional_on_joint_validity": float(np.mean(delta[joint])) if np.any(joint) else None,
            "mse_difference_interval95": interval(samples),
            "invalid_probability_difference_a_minus_b": float(np.mean(invalid_delta)),
            "invalid_probability_difference_interval95": interval(invalid_samples)}


def summarize(campaign, results, output, *, allow_partial=False):
    sources = audit_analysis()
    manifest = authenticated_json(campaign / "manifest.json")
    if (manifest.get("ready") is not True or manifest.get("version") != VERSION
            or set(manifest.get("artifacts", {})) != ARTIFACTS):
        raise ValueError("campaign is not accepted for consumer execution")
    for name, expected in manifest["artifacts"].items():
        if digest(campaign / name) != expected:
            raise ValueError(f"campaign artifact differs: {name}")
    protocol = authenticated_json(campaign / "protocol.json")
    if (protocol.get("version") != VERSION or protocol.get("producer_receipt_sha256") != PRODUCER_RECEIPT_SHA
            or protocol.get("checkpoint_sha256") != CHECKPOINT_SHA
            or protocol.get("global_binary_exponent") != -12910
            or protocol.get("environment_binary_exponent") != -25820
            or protocol.get("original_dimensions") != 9274
            or protocol.get("minimum_internal_rank") != 1
            or protocol.get("ingress_and_root_full_rank") is not True
            or protocol.get("inputs_sha256") != manifest["artifacts"]["inputs.npz"]
            or protocol.get("records_sha256") != manifest["artifacts"]["records.json"]):
        raise ValueError("frozen producer/protocol identity differs")
    planned = protocol["conditions"]
    if not planned or manifest.get("condition_count") != len(planned):
        raise ValueError("frozen condition count differs")
    seen = set()
    for index, condition in enumerate(planned):
        ranks = condition.get("ranks", [])
        identity = {key: condition.get(key) for key in ("ranks", "construction", "seed")}
        if (condition.get("task_index") != index or len(ranks) != 453
                or any(type(rank) is not int or rank < 1 for rank in ranks)
                or condition.get("rank_sha256") != json_digest(ranks)
                or condition.get("condition_sha256") != json_digest(identity)
                or condition["condition_sha256"] in seen
                or condition.get("original_dimensions") != 9274
                or condition.get("construction") not in ("leading", "zero_input_anchored_direct_qr")
                or (condition["construction"] == "leading" and condition["seed"] is not None)
                or (condition["construction"] == "zero_input_anchored_direct_qr" and
                    (type(condition["seed"]) is not int or condition["seed"] not in (0, 1, 2)))):
            raise ValueError("frozen condition schema or identity differs")
        seen.add(condition["condition_sha256"])
    if (planned[0]["removed_dimensions"] != 0 or planned[0]["construction"] != "leading"
            or planned[0]["aliases"] != [{"policy": "full", "budget": 0}]):
        raise ValueError("full-rank condition identity differs")
    widths = planned[0]["ranks"]
    for condition in planned:
        removed = sum(a-b for a, b in zip(widths, condition["ranks"]))
        if (any(a < b for a, b in zip(widths, condition["ranks"]))
                or type(condition["removed_dimensions"]) is not int or removed != condition["removed_dimensions"]
                or not condition.get("aliases")
                or any(alias["budget"] != removed for alias in condition["aliases"])):
            raise ValueError("frozen rank budget or aliases differ")
    if any(type(definition.get(key)) is not int or not 0 <= definition[key] < len(planned)
           for definition in protocol["contrasts"] for key in ("a", "b")):
        raise ValueError("frozen contrast references an absent condition")
    records = authenticated_json(campaign / "records.json")
    validate_records(records)
    with np.load(campaign / "baseline.npz", allow_pickle=False) as data:
        baseline = {name: data[name] for name in ("decoded", "valid", "projective", "failure_code", "relative_denominator", "expected")}
    validate_predictions(baseline, baseline=True)
    design = cluster_design(records)
    if len(design[0]) != 320 or len(design[2]) != 80:
        raise ValueError("frozen real panel episode counts differ")
    loaded, conditions, evidence, missing = {}, [], {}, []
    for condition in planned:
        index = condition["task_index"]
        directory = results / f"task_{index}"
        path = directory / "result.json"
        if not path.is_file():
            missing.append(index)
            continue
        receipt = authenticated_json(path)
        if receipt.get("condition") != condition:
            raise ValueError(f"condition receipt identity differs: {index}")
        required_receipt = {
            "completed": True, "version": protocol["version"],
            "campaign_manifest_sha256": digest(campaign / "manifest.json"),
            "protocol_sha256": manifest["artifacts"]["protocol.json"],
            "basis_artifact_sha256": manifest["artifacts"]["bases.npz"],
            "source_sha256": protocol["sources"],
            "producer_receipt_sha256": protocol["producer_receipt_sha256"],
            "global_binary_exponent": protocol["global_binary_exponent"],
            "environment_binary_exponent": protocol["environment_binary_exponent"],
        }
        if (any(receipt.get(key) != value for key, value in required_receipt.items())
                or receipt.get("kernels", {}).get("prohibited_attempts") != 0):
            raise ValueError(f"result belongs to a different or failed execution contract: {index}")
        predictions = directory / "predictions.npz"
        if digest(predictions) != receipt["predictions_sha256"]:
            raise ValueError(f"predictions hash differs: {index}")
        physical = receipt["physical_artifact"]
        for filename, key in (("graph.json", "graph_sha256"), ("arrays.npz", "arrays_sha256")):
            if digest(directory / filename) != physical[key]:
                raise ValueError(f"physical artifact hash differs: {index}:{filename}")
        with np.load(predictions, allow_pickle=False) as data:
            row = {name: data[name] for name in ("decoded", "valid", "projective", "failure_code", "relative_denominator", "squared_error")}
        validate_predictions(row)
        mse = row["squared_error"]
        joint = row["valid"] & baseline["valid"]
        if (mse.dtype != np.float64 or mse.shape != row["valid"].shape or not np.isfinite(mse[joint]).all()
                or not np.isnan(mse[~joint]).all()):
            raise ValueError("stored per-example squared-error schema differs")
        squared = np.sum((row["decoded"][joint]-baseline["decoded"][joint])**2, axis=1)/row["decoded"].shape[1]
        if not np.allclose(mse[joint], squared, rtol=METRIC_RTOL, atol=METRIC_ATOL):
            raise ValueError("stored per-example squared errors differ")
        metrics, task_metrics = recompute_all_metrics(row, baseline, records)
        verify_metric_receipt(receipt["metrics"], metrics)
        verify_metric_receipt(receipt["task_metrics"], task_metrics, label="task_metrics")
        loaded[index] = row
        conditions.append({"task_index": index, "condition": condition, "metrics": metrics,
                           "task_metrics": task_metrics, "receipt_metrics_independently_verified": True})
        evidence[str(index)] = {"result_sha256": digest(path), "predictions_sha256": digest(predictions),
                                "physical_artifact": physical}
    if missing and not allow_partial:
        raise ValueError(f"missing frozen conditions: {missing}")
    contrasts = []
    for definition in protocol["contrasts"]:
        if definition["a"] in loaded and definition["b"] in loaded:
            contrasts.append({**definition, **paired_comparison(loaded[definition["a"]],
                              loaded[definition["b"]], baseline, design)})
    summary = {"complete": not missing, "missing_conditions": missing, "condition_count": len(conditions),
        "planned_condition_count": len(protocol["conditions"]), "conditions": conditions, "contrasts": contrasts,
        "campaign_manifest_sha256": digest(campaign / "manifest.json"),
        "protocol_sha256": digest(campaign / "protocol.json"), "evidence": evidence,
        "analysis_source_sha256": sources, "numpy": np.__version__,
        "metric_receipt_tolerances": {"absolute": METRIC_ATOL, "relative": METRIC_RTOL,
            "counts_flags_nulls_and_schema": "exact"},
        "basis_seed_summaries": descriptive_seed_summaries(conditions),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
            "method": "resample episode clusters within each fixed task; paired errors only on joint validity",
            "interval": "95% percentile, descriptive and not adjusted for multiple comparisons",
            "scope": "conditional on one trained checkpoint and these fixed tasks"},
        "source_scope": protocol["scope"], "context_removal_error": manifest["context_removal_error"],
        "new_full_rank_replay_error": manifest["new_full_rank_replay_error"],
        "native_two_token_parity": manifest["native_two_token_parity"], "kernels": dict(COUNTS)}
    if COUNTS["prohibited_attempts"] or audit_analysis() != sources:
        raise ValueError("prohibited numerical attempt in analysis")
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    lines = ["**ODT real-input campaign: paired confirmation results**", "",
        f"Completed {len(conditions)}/{len(protocol['conditions'])} frozen conditions. "
        "One trained checkpoint, real inputs from its training-data source, two-token operator.", "",
        "Each RMSE uses the condition's jointly valid rows with the source. Invalidity uses every frozen input.", "",
        "| ID | Allocation / removed dimensions | Retained-space construction | Real RMSE | Real invalid | Gaussian RMSE |", 
        "| --- | --- | --- | ---: | ---: | ---: |"]
    def number(value):
        return "undefined" if value is None else f"{value:.6g}"
    for item in conditions:
        c, m = item["condition"], item["metrics"]
        aliases = ", ".join(f"{a['policy']} K={a['budget']}" for a in c["aliases"])
        construction = "leading" if c["construction"] == "leading" else f"zero-input anchored QR, seed {c['seed']}"
        lines.append(f"| {item['task_index']} | {aliases} | {construction} | {number(m['real']['rmse'])} | "
                     f"{m['real']['invalid_count']}/{m['real']['input_count']} | {number(m['gaussian']['rmse'])} |")
    lines += ["", "Paired real-input contrasts use a task-stratified episode bootstrap. "
        "Negative squared-error differences favor the first condition. Invalidity is reported separately. "
        "Intervals are descriptive and are not adjusted for multiple comparisons.", "",
        "| Contrast | Jointly valid | Paired MSE difference | 95% interval | Invalidity difference |",
        "| --- | ---: | ---: | --- | ---: |"]
    for row in contrasts:
        ci = row["mse_difference_interval95"]
        lines.append(f"| {row['name']} | {row['jointly_valid_count']} | "
            f"{number(row['mse_difference_a_minus_b_conditional_on_joint_validity'])} | "
            f"[{number(ci['lower'])}, {number(ci['upper'])}] | "
            f"{number(row['invalid_probability_difference_a_minus_b'])} |")
    lines += ["", "Random-control seed summaries describe the three retained-basis seeds. "
              "They do not measure policy-training-seed uncertainty. RMSE means summarize each seed's conditional endpoint.", "",
              "| Allocation | Available basis seeds | Mean real RMSE | Real RMSE range | Mean real invalidity |",
              "| --- | --- | ---: | --- | ---: |"]
    for group in summary["basis_seed_summaries"]:
        aliases = ", ".join(f"{a['policy']} K={a['budget']}" for a in group["aliases"])
        rms, invalid = group["panels"]["real"]["rmse"], group["panels"]["real"]["invalid_probability_unconditional"]
        lines.append(f"| {aliases} | {group['available_seeds']} | {number(rms['arithmetic_mean'])} | "
                     f"[{number(rms['minimum'])}, {number(rms['maximum'])}] | {number(invalid['arithmetic_mean'])} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": summary["complete"], "conditions": len(conditions),
                      "contrasts": len(contrasts), "output": str(output)}, sort_keys=True))


def main():
    audit_analysis()
    install_guards()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summarize(args.campaign, args.results, args.output, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
