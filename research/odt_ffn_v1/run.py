"""Immutable producer for a local residual FFN, with full clone acceptance."""
import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from research.odt_reference.run_tests import audit, install_guards, COUNTS, PROHIBITED
from research.odt_reference import run_curve as reference


LOCAL_FILES = ("__init__.py", "core.py", "run.py", "test_ffn.py")
SCOPE = "local first-vision-block residual FFN, trained width192 CP576, applied tokenwise; not full-policy ODT"


def source_audit():
    here = Path(__file__).resolve().parent
    digests = {}
    allowed_imports = {"argparse", "ast", "hashlib", "json", "time", "numpy", "unittest"}
    allowed_from = {"pathlib", "research.odt_reference", "research.odt_reference.run_tests",
        "research.odt_reference.block", "research.odt_reference.shared_dag",
        "research.odt_reference.weights", "research.odt_reference.checkpoint",
        "research.odt_reference.clone_passes", "research.odt_reference.scaled",
        "research.odt_reference.dooms", "research.odt_reference.curve",
        "research.odt_ffn_v1.core", "research.odt_ffn_v1.run"}
    for name in LOCAL_FILES:
        path = here / name
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import) and any(x.name not in allowed_imports for x in node.names):
                raise ValueError(f"unreviewed FFN dependency {name}:{node.lineno}")
            if isinstance(node, ast.ImportFrom) and node.module not in allowed_from:
                raise ValueError(f"unreviewed FFN dependency {name}:{node.lineno}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in PROHIBITED:
                raise ValueError(f"prohibited FFN route {name}:{node.lineno}")
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                for a, b in ((node.left, node.right), (node.right, node.left)):
                    if isinstance(b, ast.Attribute) and b.attr == "T" and ast.dump(a) == ast.dump(b.value):
                        raise ValueError("prohibited self-overlap")
    return {"local": digests, "reference": audit(reference.SOURCES)}


def independent_cut_checks(raw_graph, bases, ranks):
    """Fresh explicit-clone environment, no shared environment reused here."""
    from research.odt_reference.scaled import canonicalize_scaled_clones
    from research.odt_reference.clone_passes import aggregate_environments
    from research.odt_reference.dooms import eigenspaces
    clones, exponent = canonicalize_scaled_clones(raw_graph, maximum_nodes=100,
                                                   maximum_bytes=2 * 1024**3)
    environment = aggregate_environments(clones)[4]
    oracle = eigenspaces((environment,))[0]
    values = np.diag(oracle.T @ environment @ oracle)
    rows = []
    for rank in ranks:
        gap = float(values[rank - 1] - values[rank])
        threshold = 1e-8 * max(float(np.max(np.abs(values))), 1e-12)
        candidate = bases[4][:, :rank]
        residual = environment @ candidate - candidate * values[:rank]
        magnitude = float(np.sqrt(np.sum(residual * residual)))
        resolved = gap > threshold
        bound = magnitude / gap if resolved else None
        if resolved and (not np.isfinite(bound) or bound > reference.TOLERANCES["subspace"]):
            raise ValueError("independent FFN retained-space check failed")
        rows.append({"rank": rank, "gap": gap, "resolved": resolved,
                     "frobenius_eigen_residual": magnitude, "leakage_bound": bound})
    return exponent, rows


def prepare(checkpoint, output, sources):
    from research.odt_reference.checkpoint import load_checkpoint
    from research.odt_reference.block import evaluate_block
    from research.odt_reference.clone_passes import clone_node_count
    from research.odt_reference.shared_dag import occurrence_counts
    from research.odt_ffn_v1.core import compile_ffn, ffn_forward
    if reference.digest(checkpoint) != reference.CHECKPOINT_SHA:
        raise ValueError("checkpoint identity mismatch")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def progress(record):
        record = dict(record, elapsed_seconds=time.monotonic() - started, kernels=dict(COUNTS))
        reference.write_json(output / "progress.json", record)
        print(json.dumps(record, allow_nan=False, sort_keys=True), flush=True)
    weights = load_checkpoint(checkpoint, "vision.blocks.blocks.0.")
    if weights["ffn.left.weight"].shape != (576, 192):
        raise ValueError("trained FFN width changed")
    raw = np.random.default_rng(20260907).normal(size=(32, 1, 192)) * .2
    raw[0] = 0.
    expected = ffn_forward(raw, weights)
    np.savez(output / "panel.npz", raw=raw, expected=expected)
    np.savez(output / "weights.npz", **weights)
    progress({"phase": "compile"})
    graph = compile_ffn(weights)
    count = clone_node_count(graph, maximum_nodes=100, maximum_bytes=2 * 1024**3)
    if count != 21 or tuple(occurrence_counts(graph)) != (11, 4, 2, 2, 1, 1):
        raise ValueError("FFN clone topology mismatch")
    replay = reference.close(evaluate_block(graph, raw), expected,
                              reference.TOLERANCES["replay"], "FFN raw-factor replay")
    preflight_exp, preflight = reference.shared_preflight(graph, reference.homogeneous(raw), expected)
    progress({"phase": "clone_acceptance", "raw_replay": replay, "preflight": preflight})
    canonical, bases, schedule, gates = reference.clone_gate(graph, raw, expected, progress)
    if gates["global_binary_exponent"] != preflight_exp:
        raise ValueError("FFN preflight scale differs")
    if (schedule.canonical_internal_dimensions != 390 or schedule.eligible_bonds != 4
            or canonical.nodes[4].core.shape[0] != 193):
        raise ValueError("FFN canonical dimensions differ")
    cut_exponent, cuts = independent_cut_checks(graph, bases, (192, 184, 174, 154))
    if cut_exponent != preflight_exp:
        raise ValueError("independent FFN cutoff scale differs")
    hashes = reference.save_graph(output / "canonical", canonical, bases, global_exponent=preflight_exp)
    if source_audit() != sources or COUNTS["prohibited_attempts"]:
        raise ValueError("FFN source drift or prohibited numerical attempt")
    receipt = {"accepted": True, "scope": SCOPE, "full_policy_odt": False,
        "native_policy_replay_completed": False, "checkpoint_sha256": reference.CHECKPOINT_SHA,
        "source_sha256": sources, "numpy": np.__version__, "canonical_hashes": hashes,
        "panel_sha256": reference.digest(output / "panel.npz"),
        "weights_sha256": reference.digest(output / "weights.npz"),
        "global_binary_exponent": preflight_exp, "environment_binary_exponent": 2 * preflight_exp,
        "gates": gates, "cut_checks": cuts, "raw_replay": replay,
        "canonical_internal_dimensions": 390, "eligible_bonds": 4, "clone_count": count,
        "unique_nodes": len(canonical.nodes), "tolerances": reference.TOLERANCES,
        "input_seed": 20260907, "input_count": len(raw), "includes_zero_input": True,
        "kernels": dict(COUNTS), "elapsed_seconds": time.monotonic() - started}
    reference.write_json(output / "accepted.json", receipt)
    print(json.dumps(receipt, allow_nan=False, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = source_audit()
    install_guards()
    prepare(args.checkpoint, args.output, sources)


if __name__ == "__main__":
    main()
