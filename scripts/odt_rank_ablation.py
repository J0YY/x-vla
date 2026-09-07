"""Separate consumer of immutable v5 ODT evidence. No new decomposition claim."""
import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from research.odt_reference.run_tests import install_guards, audit, COUNTS, PROHIBITED
from research.odt_reference import run_curve as original
from research.odt_reference.curve import RankPlan, _Event, rank_schedule, physical_variant
from research.odt_reference.shared_dag import apply_bases, occurrence_environment_sums


RECEIPT_SHA = "bc961a3d700e94a15341aa2700eb85ca638cda00d80a2c0e7f6d803b14753a3d"
TASKS = (tuple(("original", p) for p in (0, 1, 5, 10, 20, 23, 24, 25, 26, 30))
         + tuple((policy, p) for policy in ("protect_all", "protect_norm")
                 for p in (30, 40, 50, 60, 70, 80))
         + (("only_all", None), ("only_norm", None), ("only_score", None),
            ("restore_all", 30), ("restore_norm", 30)))


def family(plan, nodes, kind):
    narrow = tuple(i for i in plan.eligible if plan.widths[i] == 2)
    norm = tuple(i for i in narrow if nodes[i].name.endswith((":moment", ":pade")))
    if kind == "all":
        return narrow
    if kind == "norm":
        return norm
    if kind == "score":
        return tuple(i for i in narrow if i not in norm)
    raise ValueError("unknown declared bond family")


def protected_plan(canonical, percent, protected=()):
    """Filter removal events, retaining ALL eligible widths in the denominator."""
    base = rank_schedule(canonical, percents=(percent,)).plans[0]
    protected = tuple(protected)
    if (len(set(protected)) != len(protected)
            or any(type(i) is not int or i not in base.eligible for i in protected)):
        raise ValueError("unique eligible protected indices required")
    events = sorted(_Event(2*j-1, 2*base.widths[i], i)
                    for i in base.eligible if i not in protected
                    for j in range(1, base.widths[i]))
    if base.removed_dimensions > len(events):
        raise ValueError("protected budget is unattainable")
    ranks = list(base.widths)
    for event in events[:base.removed_dimensions]:
        ranks[event.node] -= 1
    return RankPlan(percent, tuple(ranks), base.widths, base.eligible, base.signature,
                    base.original_dimensions, base.removed_dimensions)


def allocation(canonical, policy, percent):
    base = rank_schedule(canonical, percents=(0,)).plans[0]
    if policy == "original":
        return rank_schedule(canonical, percents=(percent,)).plans[0], ()
    action, kind = policy.split("_", 1)
    selected = family(base, canonical.nodes, kind)
    if action == "protect":
        return protected_plan(canonical, percent, selected), selected
    if action == "only":
        if percent is not None:
            raise ValueError("family-only ablation has no requested percent")
        ranks = list(base.widths)
        for i in selected:
            ranks[i] = 1
    elif action == "restore":
        ranks = list(rank_schedule(canonical, percents=(percent,)).plans[0].ranks)
        for i in selected:
            ranks[i] = base.widths[i]
    else:
        raise ValueError("unknown declared allocation policy")
    removed = sum(base.widths[i]-ranks[i] for i in base.eligible)
    # -1 means a causal intervention, not a requested percentage budget.
    plan = RankPlan(-1, tuple(ranks), base.widths, base.eligible, base.signature,
                    base.original_dimensions, removed)
    return plan, selected


def consumer_audit():
    """Audit this boundary AND its tests, plus the unchanged producer closure."""
    path = Path(__file__)
    test_path = path.with_name("test_odt_rank_ablation.py")
    allowed = {"argparse", "ast", "hashlib", "json", "time", "numpy", "unittest", "tempfile"}
    for source in (path, test_path):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import) and any(n.name not in allowed for n in node.names):
                raise ValueError("unexpected consumer/test import")
            if isinstance(node, ast.ImportFrom) and node.module not in {
                    "pathlib", "fractions", "scripts.odt_rank_ablation",
                    "research.odt_reference.run_tests", "research.odt_reference",
                    "research.odt_reference.curve", "research.odt_reference.shared_dag"}:
                raise ValueError("unexpected consumer/test dependency")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in PROHIBITED:
                raise ValueError("prohibited consumer/test call")
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                for a, b in ((node.left, node.right), (node.right, node.left)):
                    if isinstance(b, ast.Attribute) and b.attr == "T" and ast.dump(a) == ast.dump(b.value):
                        raise ValueError("prohibited self-overlap")
    return {"consumer_sha256": original.digest(path), "test_sha256": original.digest(test_path),
            "producer_sources": audit(original.SOURCES)}


def load_accepted(producer, sources):
    if original.digest(producer / "accepted.json") != RECEIPT_SHA:
        raise ValueError("immutable accepted producer identity mismatch")
    receipt = json.loads((producer / "accepted.json").read_text())
    if (receipt["accepted"] is not True or receipt["source_sha256"] != sources["producer_sources"]
            or receipt["checkpoint_sha256"] != original.CHECKPOINT_SHA
            or receipt["tolerances"] != original.TOLERANCES or receipt["numpy"] != np.__version__
            or original.digest(producer / "panel.npz") != receipt["panel_sha256"]):
        raise ValueError("accepted producer/source/panel mismatch")
    exponent = receipt["global_binary_exponent"]
    if type(exponent) is not int or receipt["environment_binary_exponent"] != 2*exponent:
        raise ValueError("scale ledger mismatch")
    graph, bases = original.load_graph(producer / "canonical", receipt["canonical_hashes"],
                                       global_exponent=exponent)
    base = rank_schedule(graph, percents=(0,)).plans[0]
    if (base.original_dimensions != 9274 or len(base.eligible) != 450 or len(graph.nodes) != 453
            or len(family(base, graph.nodes, "all")) != 232
            or len(family(base, graph.nodes, "norm")) != 136
            or any(graph.nodes[i].name.rsplit(":", 1)[-1] not in ("dot1", "dot2", "score")
                   for i in family(base, graph.nodes, "score"))):
        raise ValueError("frozen trained topology differs")
    with np.load(producer / "panel.npz", allow_pickle=False) as panel:
        raw, expected = panel["raw"], panel["expected"]
    return receipt, graph, bases, raw, expected


def run(producer, output, task_index):
    started, sources = time.monotonic(), consumer_audit()
    receipt, canonical, bases, raw, expected = load_accepted(producer, sources)
    policy, percent = TASKS[task_index]
    plan, selected = allocation(canonical, policy, percent)
    # Additional cutoffs are not retroactively part of the old clone certificate.
    # Check their finite eigen-equations and flag unresolved gaps explicitly.
    residuals, unresolved = [], []
    for i, (env, basis) in enumerate(zip(occurrence_environment_sums(canonical), bases)):
        diagonal = basis.T @ env @ basis
        values = np.diag(diagonal)
        residual = env @ basis - basis * values
        scale = max(float(np.max(np.abs(env))), 1e-12)
        error = float(np.max(np.abs(residual))) / scale
        if not np.isfinite(error) or error > original.TOLERANCES["subspace"]:
            raise ValueError("reused basis eigen-equation failed")
        residuals.append(error)
        k = plan.ranks[i]
        if k < len(values) and values[k-1]-values[k] <= 1e-8*max(float(np.max(np.abs(values))), 1e-12):
            unresolved.append(i)
    variant = physical_variant(canonical, bases, plan)
    hashes = original.save_graph(output, variant.graph, global_exponent=receipt["global_binary_exponent"])
    reloaded, _ = original.load_graph(output, hashes, global_exponent=receipt["global_binary_exponent"])
    inputs = original.homogeneous(raw)
    physical = original.checked_chart(reloaded, inputs)
    masked = original.checked_chart(apply_bases(canonical, variant.full_bases), inputs, plan.ranks)
    if physical.valid_rows != masked.valid_rows or physical.failure_reasons != masked.failure_reasons:
        raise ValueError("physical/mask chart classifications differ")
    equivalence = original.close(physical.projective, masked.projective,
                                 original.TOLERANCES["coordinates"], "physical/mask")
    valid = np.array(physical.valid_rows, dtype=bool)
    decoded = physical.projective[valid, :-1] / physical.projective[valid, -1:]
    delta = decoded - expected[valid]
    if not np.isfinite(delta).all():
        raise ValueError("nonfinite decoded error")
    full_rank_error = (original.close(decoded, expected, original.TOLERANCES["replay"],
                                     "consumer full-rank replay")
                       if plan.removed_dimensions == 0 else None)
    if consumer_audit() != sources or COUNTS["prohibited_attempts"]:
        raise ValueError("consumer source changed or prohibited route attempted")
    result = {"scope": original.SCOPE, "full_policy": False, "libero_evaluated": False,
              "posthoc_allocation_diagnostic": True, "synthetic_inputs": True,
              "task_index": task_index, "policy": policy, "requested_percent": percent,
              "budget_matched": policy == "original" or policy.startswith("protect_"),
              "selected_family_nodes": selected, "ranks": plan.ranks,
              "original_dimensions": plan.original_dimensions, "removed_dimensions": plan.removed_dimensions,
              "actual_removed_fraction": plan.actual_fraction, "kept_dimensions": plan.retained_dimensions,
              "width2_cuts": [i for i in plan.eligible if plan.widths[i] == 2 and plan.ranks[i] == 1],
              "physical_artifact": hashes, "producer_receipt_sha256": RECEIPT_SHA,
              "checkpoint_sha256": receipt["checkpoint_sha256"], "panel_sha256": receipt["panel_sha256"],
              "global_binary_exponent": receipt["global_binary_exponent"],
              "new_cutoffs_independently_clone_checked": False,
              "new_cutoff_unresolved_nodes": unresolved, "basis_eigen_residual_max": max(residuals),
              "rank_mask_equivalence": equivalence, "valid_inputs": int(np.sum(valid)),
              "full_rank_replay_error": full_rank_error,
              "input_count": len(raw), "failure_reasons": physical.failure_reasons,
              "relative_denominators": physical.relative_denominators,
              "errors_on_valid_inputs_only": {
                  "rmse": float(np.sqrt(np.mean(delta*delta))) if delta.size else None,
                  "maximum_absolute_error": float(np.max(np.abs(delta))) if delta.size else None},
              "original_output_rms": float(np.sqrt(np.mean(expected*expected))),
              "elapsed_seconds": time.monotonic()-started, "sources": sources, "kernels": dict(COUNTS)}
    original.write_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    install_guards()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-index", type=int, choices=range(len(TASKS)), required=True)
    args = parser.parse_args()
    run(args.producer, args.output, args.task_index)
