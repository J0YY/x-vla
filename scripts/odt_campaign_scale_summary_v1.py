"""Authenticated fixed-lift scale diagnostics for the completed primary 25 arms."""
import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np

from research.odt_reference.run_tests import COUNTS, PROHIBITED, install_guards
from scripts.odt_campaign_summary_v1 import (
    ARTIFACTS, CHECKPOINT_SHA, PRODUCER_RECEIPT_SHA, VERSION, audit_analysis,
    authenticated_json, digest, validate_predictions, validate_records,
)

SCALE_KEYS = ("output_log2_scale", "log2_abs_denominator", "log2_numerator_max_abs", "denominator_sign")
CHART_KEYS = ("decoded", "valid", "projective", "failure_code", "relative_denominator")
LOG_ATOL = 2e-9


def source_audit():
    here = Path(__file__).resolve()
    files = (here, here.with_name("test_odt_campaign_scale_summary_v1.py"))
    imports = {"argparse", "ast", "json", "math", "numpy", "unittest", "subprocess", "sys", "tempfile"}
    modules = {"pathlib", "research.odt_reference.run_tests", "scripts.odt_campaign_summary_v1",
               "scripts.odt_campaign_scale_summary_v1", "research.odt_campaign_v1.campaign"}
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import) and any(x.name not in imports for x in node.names):
                raise ValueError("unreviewed scale-summary import")
            if isinstance(node, ast.ImportFrom) and node.module not in modules:
                raise ValueError("unreviewed scale-summary dependency")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in PROHIBITED | {"qr", "eigh", "exp", "exp2", "ldexp"}:
                raise ValueError("prohibited scale-summary numerical call")
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.MatMult, ast.Pow)):
                raise ValueError("scale summary needs no tensor contraction or exponentiation")
    # Only the audit is called, never a graph constructor or numerical evaluator.
    from research.odt_campaign_v1.campaign import audit_campaign
    return {"local": {str(path.relative_to(here.parents[1])): digest(path) for path in files},
            "paired_analysis": audit_analysis(), "consumer": audit_campaign()}


def assert_guards():
    if (COUNTS != {"qr": 0, "eigh": 0, "prohibited_attempts": 0}
            or np.linalg.qr.__name__ != "counted" or np.linalg.eigh.__name__ != "counted"
            or any(getattr(namespace, name).__name__ != "reject" for namespace in (np, np.linalg)
                   for name in PROHIBITED if hasattr(namespace, name))):
        raise ValueError("scale-summary runtime guards differ or numerical factorization occurred")


def extended_equal(actual, expected, label):
    if (actual.dtype != np.float64 or actual.shape != expected.shape
            or not np.array_equal(np.isnan(actual), np.isnan(expected))
            or not np.array_equal(np.isposinf(actual), np.isposinf(expected))
            or not np.array_equal(np.isneginf(actual), np.isneginf(expected))):
        raise ValueError(f"extended logarithm schema differs: {label}")
    finite = np.isfinite(expected)
    if np.any(np.abs(actual[finite]-expected[finite]) > LOG_ATOL):
        raise ValueError(f"logarithm replay differs: {label}")


def validate_scales(values):
    rows = len(values["valid"])
    scale = values["output_log2_scale"]
    zero = np.max(np.abs(values["projective"]), axis=1) == 0
    if (scale.dtype != np.float64 or scale.shape != (rows,) or np.isnan(scale).any()
            or np.isposinf(scale).any() or not np.array_equal(np.isneginf(scale), zero)):
        raise ValueError("nonfinite or inconsistent output-scale ledger")
    sign = values["denominator_sign"]
    expected_sign = np.sign(values["projective"][:, -1]).astype(np.int8)
    if sign.dtype != np.int8 or sign.shape != (rows,) or not np.array_equal(sign, expected_sign):
        raise ValueError("denominator sign differs from normalized coordinates")
    with np.errstate(divide="ignore", invalid="raise"):
        expected_d = scale + np.log2(np.abs(values["projective"][:, -1]))
        expected_n = scale + np.log2(np.max(np.abs(values["projective"][:, :-1]), axis=1))
    extended_equal(values["log2_abs_denominator"], expected_d, "absolute denominator")
    extended_equal(values["log2_numerator_max_abs"], expected_n, "maximum numerator")


def log_difference(current, baseline):
    """Exact-zero log ratios retain +/- infinity; zero over zero is undefined."""
    with np.errstate(invalid="ignore", over="raise"):
        return current-baseline


def distribution(values):
    finite = values[np.isfinite(values)]
    return {"count": int(values.size), "finite_count": int(finite.size),
            "negative_infinity_count": int(np.count_nonzero(np.isneginf(values))),
            "positive_infinity_count": int(np.count_nonzero(np.isposinf(values))),
            "undefined_count": int(np.count_nonzero(np.isnan(values))),
            "finite_minimum": float(np.min(finite)) if len(finite) else None,
            "finite_p05": float(np.percentile(finite, 5)) if len(finite) else None,
            "finite_median": float(np.median(finite)) if len(finite) else None,
            "finite_p95": float(np.percentile(finite, 95)) if len(finite) else None,
            "finite_maximum": float(np.max(finite)) if len(finite) else None}


def chart_log_amplitude(values):
    p = values["projective"]
    with np.errstate(divide="ignore", invalid="ignore"):
        amplitude = np.log2(np.max(np.abs(p[:, :-1]), axis=1))-np.log2(np.abs(p[:, -1]))
    has_denominator = p[:, -1] != 0
    with np.errstate(divide="ignore", invalid="raise"):
        expected_margin = -np.log2(values["relative_denominator"][has_denominator])
    if np.any(np.abs(expected_margin-np.maximum(0., amplitude[has_denominator])) > LOG_ATOL):
        raise ValueError("chart/output-amplitude identity differs")
    return amplitude


def validate_cross_and_deltas(current, baseline):
    for name in ("log2_abs_denominator", "log2_numerator_max_abs"):
        extended_equal(current["delta_"+name], log_difference(current[name], baseline[name]), "delta "+name)
    cp, bp = current["projective"], baseline["projective"]
    cross = cp[:, :-1]*bp[:, -1:] - bp[:, :-1]*cp[:, -1:]
    stored = current["cross_product_normalized_mantissa"]
    if stored.dtype != np.float64 or stored.shape != cross.shape or not np.array_equal(stored, cross):
        raise ValueError("cross-product normalized mantissa differs")
    with np.errstate(divide="ignore", invalid="raise"):
        expected = np.log2(np.max(np.abs(cross), axis=1))+current["output_log2_scale"]+baseline["output_log2_scale"]
    extended_equal(current["log2_cross_product_numerator_max_abs"], expected, "cross-product fixed-lift log")


def panel_summary(current, baseline, selection):
    sign, source_sign = current["denominator_sign"][selection], baseline["denominator_sign"][selection]
    amp, base_amp = chart_log_amplitude(current), chart_log_amplitude(baseline)
    nonzero = (sign != 0) & (source_sign != 0)
    return {"input_count": int(np.count_nonzero(selection)),
            "invalid_count": int(np.count_nonzero(~current["valid"][selection])),
            "jointly_valid_count": int(np.count_nonzero(current["valid"][selection] & baseline["valid"][selection])),
            "delta_log2_abs_denominator": distribution(log_difference(current["log2_abs_denominator"], baseline["log2_abs_denominator"])[selection]),
            "delta_log2_numerator_max_abs": distribution(log_difference(current["log2_numerator_max_abs"], baseline["log2_numerator_max_abs"])[selection]),
            "fixed_lift_log2_abs_denominator": distribution(current["log2_abs_denominator"][selection]),
            "fixed_lift_log2_numerator_max_abs": distribution(current["log2_numerator_max_abs"][selection]),
            "cross_product_fixed_lift_log2_max_abs": distribution(current["log2_cross_product_numerator_max_abs"][selection]),
            "denominator_sign": {"opposite_nonzero": int(np.count_nonzero(nonzero & (sign != source_sign))),
                "same_nonzero": int(np.count_nonzero(nonzero & (sign == source_sign))),
                "source_nonzero_to_zero": int(np.count_nonzero((source_sign != 0) & (sign == 0))),
                "source_zero_to_nonzero": int(np.count_nonzero((source_sign == 0) & (sign != 0))),
                "both_zero": int(np.count_nonzero((source_sign == 0) & (sign == 0)))},
            "chart_margin": distribution(current["relative_denominator"][selection]),
            "log2_output_max_abs_from_chart": distribution(amp[selection]),
            "delta_log2_output_max_abs_from_chart": distribution(log_difference(amp, base_amp)[selection]),
            "decoded_valid_log2_output_max_abs": distribution(amp[selection & current["valid"]])}


def allocation_summary(condition, cutoffs, nodes, widths, spectra, diagnostics):
    ranks = condition["ranks"]
    cut = {i for i, (width, rank) in enumerate(zip(widths, ranks)) if rank < width}
    if len(cutoffs) != len(cut) or {c["node"] for c in cutoffs} != cut:
        raise ValueError("cutoff inventory differs from frozen ranks")
    families = {}
    for item in cutoffs:
        i = item["node"]
        width, rank = widths[i], ranks[i]
        if not nodes[i]["children"] or nodes[i]["source"] is not None or i == len(widths)-1:
            raise ValueError("ingress or root unexpectedly cut")
        values = spectra[i]
        if values.dtype != np.float64 or values.shape != (width,) or not np.isfinite(values).all():
            raise ValueError("cutoff spectrum schema differs")
        gap = float(values[rank-1]-values[rank])
        unresolved = gap <= 1e-8*max(float(np.max(np.abs(values))), 1e-12)
        expected = {"node": i, "rank": rank, "width": width, "eigengap": gap,
                    "unresolved": unresolved, "eigen_equation_residual": diagnostics[i]["eigen_equation_residual"],
                    "tail_cost_stored_environment_units": float(np.sum(values[rank:]))}
        if item != expected or type(item["unresolved"]) is not bool:
            raise ValueError("cutoff receipt differs from authenticated spectrum")
        family = "wide" if width != 2 else "moment_pade" if nodes[i]["name"].endswith((":moment", ":pade")) else "attention_scalar"
        totals = families.setdefault(family, {"cut_bonds": 0, "removed_dimensions": 0, "unresolved_cutoffs": 0,
            "tail_cost_stored_environment_units": 0.})
        totals["cut_bonds"] += 1
        totals["removed_dimensions"] += width-rank
        totals["unresolved_cutoffs"] += int(unresolved)
        totals["tail_cost_stored_environment_units"] += item["tail_cost_stored_environment_units"]
    removed = sum(value["removed_dimensions"] for value in families.values())
    if removed != condition["removed_dimensions"]:
        raise ValueError("bond-family allocation does not match frozen removal budget")
    return {"cut_bonds": len(cut), "removed_dimensions": removed,
            "unresolved_cutoffs": sum(item["unresolved_cutoffs"] for item in families.values()),
            "by_bond_family": families,
            "scope": "cutoff gaps describe the allocated leading spectral boundaries, including for matched random retained spaces"}


def summarize(campaign, results, paired_summary, paired_sha, output):
    assert_guards()
    sources = source_audit()
    paired = authenticated_json(paired_summary, paired_sha)
    if (paired.get("complete") is not True or paired.get("missing_conditions") != []
            or paired.get("condition_count") != 25 or paired.get("planned_condition_count") != 25
            or paired.get("analysis_source_sha256") != sources["paired_analysis"]
            or paired.get("kernels") != {"qr": 0, "eigh": 0, "prohibited_attempts": 0}):
        raise ValueError("complete audited primary25 paired summary required")
    manifest = authenticated_json(campaign/"manifest.json", paired["campaign_manifest_sha256"])
    if manifest.get("ready") is not True or manifest.get("version") != VERSION or set(manifest["artifacts"]) != ARTIFACTS:
        raise ValueError("campaign manifest schema differs")
    for name, expected in manifest["artifacts"].items():
        if digest(campaign/name) != expected:
            raise ValueError("campaign artifact hash differs: "+name)
    protocol = authenticated_json(campaign/"protocol.json", paired["protocol_sha256"])
    if (protocol["sources"] != sources["consumer"] or protocol["checkpoint_sha256"] != CHECKPOINT_SHA
            or protocol["producer_receipt_sha256"] != PRODUCER_RECEIPT_SHA
            or protocol["global_binary_exponent"] != -12910 or protocol["environment_binary_exponent"] != -25820
            or len(protocol["conditions"]) != 25 or manifest["condition_count"] != 25):
        raise ValueError("audited campaign source or fixed-lift contract differs")
    records = authenticated_json(campaign/"records.json")
    validate_records(records)
    with np.load(campaign/"baseline.npz", allow_pickle=False) as arrays:
        baseline = {name: arrays[name] for name in (*CHART_KEYS, *SCALE_KEYS, "expected")}
    validate_predictions(baseline, baseline=True)
    validate_scales(baseline)
    widths = protocol["conditions"][0]["ranks"]
    diagnostics = authenticated_json(campaign/"spectral_diagnostics.json")
    with np.load(campaign/"bases.npz", allow_pickle=False) as arrays:
        spectra = [arrays[f"spectrum_{i}"] for i in range(len(widths))]
    if len(widths) != 453 or len(diagnostics) != len(widths):
        raise ValueError("frozen bond count differs")
    selections = {panel: np.array([row["panel"] == panel for row in records]) for panel in ("real", "gaussian")}
    conditions, evidence, base_nodes = [], {}, None
    for index, condition in enumerate(protocol["conditions"]):
        directory = results/f"task_{index}"
        authenticated = paired["evidence"][str(index)]
        receipt = authenticated_json(directory/"result.json", authenticated["result_sha256"])
        if (condition != paired["conditions"][index]["condition"] or receipt["condition"] != condition
                or condition["task_index"] != index or receipt.get("completed") is not True
                or receipt.get("fixed_lift_scale_diagnostic") is not True
                or receipt["campaign_manifest_sha256"] != paired["campaign_manifest_sha256"]
                or receipt["source_sha256"] != protocol["sources"]
                or receipt["global_binary_exponent"] != -12910 or receipt["environment_binary_exponent"] != -25820
                or receipt["predictions_sha256"] != authenticated["predictions_sha256"]
                or receipt["physical_artifact"] != authenticated["physical_artifact"]):
            raise ValueError("result identity differs from authenticated paired analysis")
        for name, expected in (("predictions.npz", receipt["predictions_sha256"]),
                               ("graph.json", receipt["physical_artifact"]["graph_sha256"]),
                               ("arrays.npz", receipt["physical_artifact"]["arrays_sha256"])):
            if digest(directory/name) != expected:
                raise ValueError("condition artifact hash differs: "+name)
        graph = authenticated_json(directory/"graph.json")
        nodes = graph["nodes"]
        if (graph["global_binary_exponent"] != -12910 or len(nodes) != len(widths)
                or graph["arrays_sha256"] != receipt["physical_artifact"]["arrays_sha256"]
                or [n["shape"][0] for n in nodes] != condition["ranks"]):
            raise ValueError("physical graph allocation schema differs")
        structure = [{key: node[key] for key in ("name", "children", "source")} for node in nodes]
        if base_nodes is None:
            base_nodes = structure
        elif structure != base_nodes:
            raise ValueError("physical graph node identities changed")
        with np.load(directory/"predictions.npz", allow_pickle=False) as arrays:
            current = {name: arrays[name] for name in (*CHART_KEYS, *SCALE_KEYS,
                "delta_log2_abs_denominator", "delta_log2_numerator_max_abs",
                "cross_product_normalized_mantissa", "log2_cross_product_numerator_max_abs")}
        validate_predictions(current)
        validate_scales(current)
        validate_cross_and_deltas(current, baseline)
        conditions.append({"task_index": index, "condition": condition,
            "panels": {panel: panel_summary(current, baseline, selection) for panel, selection in selections.items()},
            "allocation": allocation_summary(condition, receipt["cutoffs"], nodes, widths, spectra, diagnostics),
            "new_cutoffs_independently_clone_checked": receipt["new_cutoffs_independently_clone_checked"]})
        evidence[str(index)] = authenticated
    summary = {"schema": "odt-campaign-fixed-lift-scale-summary-v1", "complete": True,
        "condition_count": len(conditions), "conditions": conditions, "evidence": evidence,
        "baseline_panels": {panel: {key: distribution(baseline[key][selection]) for key in SCALE_KEYS[:3]}
                            for panel, selection in selections.items()},
        "analysis_source_sha256": sources, "paired_summary_sha256": paired_sha,
        "campaign_manifest_sha256": paired["campaign_manifest_sha256"],
        "baseline_sha256": manifest["artifacts"]["baseline.npz"], "numpy": np.__version__,
        "log_replay_absolute_tolerance": LOG_ATOL,
        "interpretation": {"fixed_lift": "representation-specific logarithmic scales with global exponent -12910; never exponentiated",
            "chart": "margin = 1/max(1,max_abs(decoded)); an output-amplitude statistic, not pole or conditioning evidence",
            "zeros": "log magnitudes may be -infinity for exact zeros; log ratios may be +/-infinity; zero/zero ratios are undefined",
            "conditioning": "no perturbation probe was executed; neither absolute denominator size nor amplitude establishes a pole",
            "cutoffs": "unresolved leading spectral gaps are counted per condition, not distinct eigenvalue claims across repeated conditions",
            "validity": "fixed-lift summaries include all rows; decoded amplitude is also separately summarized only on valid rows"}}
    assert_guards()
    if source_audit() != sources:
        raise ValueError("scale-summary sources changed during analysis")
    summary["kernels"] = dict(COUNTS)
    output.mkdir(parents=True, exist_ok=False)
    (output/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+"\n")
    lines = ["**Fixed-lift scale diagnostics, primary 25 conditions**", "",
             summary["interpretation"]["chart"]+". "+summary["interpretation"]["conditioning"]+".", "",
             "Table entries are medians over finite log differences. JSON retains all nonfinite and undefined counts.", "",
             "| ID | Panel | Median Δlog2 abs D | Median Δlog2 max abs N | Opposite denominator signs | Unresolved cutoffs |", 
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for item in conditions:
        for panel, values in item["panels"].items():
            d, n = (values[key]["finite_median"] for key in ("delta_log2_abs_denominator", "delta_log2_numerator_max_abs"))
            lines.append(f"| {item['task_index']} | {panel} | {d} | {n} | {values['denominator_sign']['opposite_nonzero']} | {item['allocation']['unresolved_cutoffs']} |")
    (output/"summary.md").write_text("\n".join(lines)+"\n")
    return summary


def main():
    install_guards()
    source_audit()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--paired-summary", type=Path, required=True)
    parser.add_argument("--paired-summary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.campaign, args.results, args.paired_summary, args.paired_summary_sha256, args.output)
    print(json.dumps({"complete": result["complete"], "condition_count": result["condition_count"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
