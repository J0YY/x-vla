"""Guarded trained two-token BLOCK curve, not a full-policy or LIBERO run.

One independent-clone producer publishes an acceptance receipt last. Separate
workers consume that immutable canonical graph and physically remove directions.
Compilation, persistence and scheduling are outside the mathematical reference.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from research.odt_reference.run_tests import audit, install_guards, COUNTS
from research.odt_reference.shared_dag import (
    Graph, Node, canonical_step, absorb_axis, occurrence_environment_sums, apply_bases, occurrence_counts,
)
from research.odt_reference.clone_oracle import walk_clones
from research.odt_reference.clone_passes import (
    clone_node_count, aggregate_environments, apply_common_bases,
)
from research.odt_reference.dooms import eigenspaces
from research.odt_reference.curve import rank_schedule, physical_variant, masked_evaluate
from research.odt_reference.scaled import balanced, canonicalize_scaled_clones


SOURCES = ("run_curve.py", "checkpoint.py", "block.py", "block_oracle.py", "weights.py",
           "dooms.py", "shared_dag.py", "clone_oracle.py", "clone_passes.py", "curve.py", "scaled.py")
CHECKPOINT_SHA = "b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee"
TOLERANCES = {"coordinates": 3e-10, "replay": 1e-10, "environment": 3e-10,
              "off_diagonal": 3e-10, "subspace": 2e-8, "denominator_margin": 1e-12}
SCOPE = "unchanged trained width192/head8/rank576, joint two-token vision block, six Pade norms; synthetic inputs; NOT full policy or LIBERO"


def digest(path):
    state = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            state.update(chunk)
    return state.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def close(actual, expected, tolerance, label):
    """Finite, global-max-scaled error, chunked to bound comparison workspace."""
    if actual.shape != expected.shape:
        raise ValueError(f"{label}: shape mismatch")
    # Compare every entry even in the common bitwise-identical clone case.
    # Infinities can compare equal, so exact equality still requires finiteness.
    if np.array_equal(actual, expected):
        if not np.isfinite(actual).all():
            raise ValueError(f"{label}: nonfinite comparison")
        return 0.
    a, b, error, scale = actual.reshape(-1), expected.reshape(-1), 0., 0.
    for offset in range(0, a.size, 131072):
        x, y = a[offset:offset + 131072], b[offset:offset + 131072]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"{label}: nonfinite comparison")
        error = max(error, float(np.max(np.abs(x - y))))
        scale = max(scale, float(np.max(np.abs(y))))
    scaled = error / max(scale, 1e-12)
    if not np.isfinite(scaled) or scaled > tolerance:
        raise ValueError(f"{label}: scaled error {scaled} exceeds {tolerance}")
    return scaled


def save_graph(directory, graph, bases=(), *, global_exponent=0):
    if type(global_exponent) is not int:
        raise ValueError("integer global binary exponent required")
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {f"core_{i}": n.core for i, n in enumerate(graph.nodes)}
    arrays.update({f"basis_{i}": b for i, b in enumerate(bases)})
    arrays["head"] = graph.head
    np.savez(directory / "arrays.npz", **arrays)
    metadata = {"canonical": graph.canonical, "basis_count": len(bases),
                "global_binary_exponent": global_exponent,
                "nodes": [{"name": n.name, "children": n.children, "source": n.source,
                           "shape": list(n.core.shape)} for n in graph.nodes],
                "arrays_sha256": digest(directory / "arrays.npz")}
    write_json(directory / "graph.json", metadata)
    return {"arrays_sha256": metadata["arrays_sha256"],
            "graph_sha256": digest(directory / "graph.json")}


def load_graph(directory, hashes, *, global_exponent=0):
    for name, key in (("graph.json", "graph_sha256"), ("arrays.npz", "arrays_sha256")):
        if digest(directory / name) != hashes[key]:
            raise ValueError("graph artifact identity mismatch")
    metadata = json.loads((directory / "graph.json").read_text())
    if (type(metadata["global_binary_exponent"]) is not int
            or metadata["global_binary_exponent"] != global_exponent):
        raise ValueError("global tensor scale identity mismatch")
    if metadata["arrays_sha256"] != hashes["arrays_sha256"]:
        raise ValueError("graph metadata identity mismatch")
    with np.load(directory / "arrays.npz", allow_pickle=False) as data:
        expected = {"head"} | {f"core_{i}" for i in range(len(metadata["nodes"]))}
        expected |= {f"basis_{i}" for i in range(metadata["basis_count"])}
        if set(data.files) != expected:
            raise ValueError("unexpected graph arrays")
        nodes = []
        for i, description in enumerate(metadata["nodes"]):
            core = data[f"core_{i}"]
            if core.dtype != np.float64 or list(core.shape) != description["shape"]:
                raise ValueError("graph core shape/dtype mismatch")
            nodes.append(Node(description["name"], core, tuple(description["children"]), description["source"]))
        graph = Graph(nodes, data["head"], metadata["canonical"])
        bases = tuple(data[f"basis_{i}"] for i in range(metadata["basis_count"]))
    return graph, bases


def homogeneous(raw):
    return {f"token{t}": np.concatenate((raw[:, t], np.ones((len(raw), 1))), axis=1)
            for t in range(raw.shape[1])}


def checked_chart(graph, inputs, ranks=None):
    chart = masked_evaluate(graph, inputs, ranks,
                            denominator_margin=TOLERANCES["denominator_margin"])
    if "nonfinite_contraction" in chart.failure_reasons:
        raise ValueError("nonfinite execution, not a policy error measurement")
    return chart


def full_rank_replay(graph, inputs, expected, label):
    chart = checked_chart(graph, inputs)
    if chart.decoded is None:
        raise ValueError(f"{label}: invalid full-rank chart, reasons={chart.failure_reasons}, "
                         f"margins={chart.relative_denominators}, head_amax={float(np.max(np.abs(graph.head)))}")
    return close(chart.decoded, expected, TOLERANCES["replay"], label)


def scaled_shared_step(graph, index, multiplicity):
    """Move one core's exact binary scale to the COMPLETE-tensor ledger."""
    graph.nodes[index].core, shift = balanced(graph.nodes[index].core)
    scale = max(float(np.max(np.abs(graph.nodes[index].core))), 1e-12)
    record = canonical_step(graph, index)
    error = record["reconstruction_max_error"] / scale
    if not np.isfinite(error) or error > TOLERANCES["coordinates"]:
        raise ValueError("local direct-QR reconstruction failed")
    return int(multiplicity) * shift, error


def shared_preflight(graph, inputs, expected):
    """Necessary finite-arithmetic gates only, NOT independent acceptance."""
    preflight, exponent = graph.copy(), 0
    for index, multiplicity in enumerate(occurrence_counts(graph)):
        delta, _ = scaled_shared_step(preflight, index, multiplicity)
        exponent += delta
    preflight.head, shift = balanced(preflight.head)
    exponent += shift
    preflight.canonical = True
    canonical_error = full_rank_replay(preflight, inputs, expected, "shared-only canonical preflight")
    bases = eigenspaces(occurrence_environment_sums(preflight))
    gauged = apply_bases(preflight, bases)
    gauge_error = full_rank_replay(gauged, inputs, expected, "shared-only gauged preflight")
    off_diagonal = max(close(e, np.diag(np.diag(e)), TOLERANCES["off_diagonal"], "preflight off-diagonal")
                       for e in occurrence_environment_sums(gauged))
    return exponent, {"canonical_replay": canonical_error, "gauged_replay": gauge_error,
                      "post_gauge_off_diagonal": off_diagonal}


def clone_gate(original, raw, expected, progress):
    """Every-origin core equality certifies identical ordered clone networks.

    Trained tensors are too large to expand into all physical coefficients.
    Literal complete-coefficient checks therefore remain in the tiny test suite.
    """
    shared, started, errors = original.copy(), time.monotonic(), {}
    multiplicities, global_exponent = occurrence_counts(original), 0

    def compare(tree, index, phase):
        worst = close(shared.head, tree.head, TOLERANCES["coordinates"], "clone head")
        for node in walk_clones(tree):
            worst = max(worst, close(shared.nodes[node.origin].core, node.core,
                                     TOLERANCES["coordinates"], "clone core"))
        errors[phase] = max(errors.get(phase, 0.), worst)
        progress({"phase": phase, "completed_origins": index + 1, "total_origins": len(shared.nodes),
                  "global_binary_exponent": global_exponent,
                  "elapsed_seconds": time.monotonic() - started, "maximum_errors": errors})

    def qr_step(tree, index, independent_exponent):
        nonlocal global_exponent
        delta, error = scaled_shared_step(shared, index, multiplicities[index])
        global_exponent += delta
        if global_exponent != independent_exponent:
            raise ValueError("independent complete-tensor scale ledger mismatch")
        errors["local_qr"] = max(errors.get("local_qr", 0.), error)
        compare(tree, index, "every_qr_step")

    clones, independent_exponent = canonicalize_scaled_clones(original, maximum_nodes=100_000,
                                     maximum_bytes=32 * 1024**3, after_step=qr_step)
    shared.head, head_shift = balanced(shared.head)
    global_exponent += head_shift
    if global_exponent != independent_exponent:
        raise ValueError("independent head scale ledger mismatch")
    close(shared.head, clones.tree.head, TOLERANCES["coordinates"], "balanced clone head")
    shared.canonical = True
    inputs = homogeneous(raw)
    errors["canonical_replay"] = full_rank_replay(shared, inputs, expected, "canonical replay")
    progress({"phase": "independent_occurrence_environments", "elapsed_seconds": time.monotonic() - started})
    environments, independent = occurrence_environment_sums(shared), aggregate_environments(clones)
    errors["environment"] = max(close(a, b, TOLERANCES["environment"], "environment")
                                for a, b in zip(environments, independent))
    bases = eigenspaces(environments)
    independent_bases = eigenspaces(independent)
    schedule = rank_schedule(shared, original)
    checked, degenerate = 0, 0
    for i, (environment, basis, oracle) in enumerate(zip(independent, bases, independent_bases)):
        diagonal = oracle.T @ environment @ oracle
        values = np.diag(diagonal)
        for rank in sorted({plan.ranks[i] for plan in schedule.plans}):
            if rank == len(values):
                continue
            if values[rank-1] - values[rank] <= 1e-8 * max(float(np.max(np.abs(values))), 1e-12):
                degenerate += 1
                continue
            # Compare against the independent environment's leading eigenspace
            # through its eigen-equation, not a solve or an overlap matrix.
            # At a resolved cutoff, residual / separation bounds leakage into
            # the excluded invariant subspace (Frobenius residual bound).
            candidate = basis[:, :rank]
            residual = environment @ candidate - candidate * values[:rank]
            bound = float(np.sqrt(np.sum(residual * residual))) / (values[rank-1] - values[rank])
            if not np.isfinite(bound) or bound > TOLERANCES["subspace"]:
                raise ValueError("independent retained-subspace residual bound failed")
            checked += 1
    canonical = shared.copy()

    def gauge_step(tree, index):
        node, basis = shared.nodes[index], bases[index]
        node.core = np.tensordot(basis.T, node.core, axes=(1, 0))
        for parent in shared.nodes:
            for role, child in enumerate(parent.children):
                if child == index:
                    parent.core = absorb_axis(parent.core, role, basis)
        if index == len(shared.nodes) - 1:
            shared.head = shared.head @ basis
        compare(tree, index, "every_gauge_step")

    apply_common_bases(clones, bases, after_step=gauge_step)
    shared.canonical = True
    off_diagonal = []
    for environment in occurrence_environment_sums(shared):
        off_diagonal.append(close(environment, np.diag(np.diag(environment)),
                                  TOLERANCES["off_diagonal"], "post-gauge environment"))
    errors["post_gauge_off_diagonal"] = max(off_diagonal)
    errors["gauged_replay"] = full_rank_replay(shared, inputs, expected, "gauged replay")
    return canonical, bases, schedule, {"maximum_errors": errors, "subspace_checks": checked,
                                      "global_binary_exponent": global_exponent,
                                      "degenerate_cutoffs_not_uniquely_identifiable": degenerate}


def prepare(output, checkpoint, sources):
    from research.odt_reference.checkpoint import load_checkpoint
    from research.odt_reference.block import BlockSpec, compile_block, evaluate_block
    from research.odt_reference.block_oracle import block_forward
    if digest(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("checkpoint identity mismatch")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def progress(value):
        value = dict(value, campaign_seconds=time.monotonic() - started, scope=SCOPE, kernels=dict(COUNTS))
        write_json(output / "progress.json", value)
        print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)
    progress({"phase": "compile"})
    weights = load_checkpoint(checkpoint, "vision.blocks.blocks.0.")
    if weights["ffn.left.weight"].shape != (576, 192):
        raise ValueError("trained block shape mismatch")
    raw = np.random.default_rng(20260906).normal(size=(64, 2, 192)) * .2
    mask, spec = np.ones((2, 2)), BlockSpec(8)
    expected = block_forward(raw, weights, n_heads=8, mask=mask, eps=spec.eps).reshape(len(raw), -1)
    np.savez(output / "panel.npz", raw=raw, expected=expected)
    graph = compile_block(weights, spec=spec, mask=mask, maximum_elements=16_000_000)
    count = clone_node_count(graph, maximum_nodes=100_000, maximum_bytes=32 * 1024**3)
    replay = close(evaluate_block(graph, raw), expected, TOLERANCES["replay"], "independent export replay")
    full_rank_replay(graph, homogeneous(raw), expected, "original chart preflight")
    # Cheap necessary preflight, never a substitute for the clone acceptance.
    preflight_exponent, preflight_errors = shared_preflight(graph, homogeneous(raw), expected)
    progress({"phase": "shared_only_preflight_passed_not_acceptance", "errors": preflight_errors,
              "global_binary_exponent": preflight_exponent})
    progress({"phase": "clone_qr", "clone_count": count, "export_replay_error": replay})
    canonical, bases, schedule, gates = clone_gate(graph, raw, expected, progress)
    exponent = gates["global_binary_exponent"]
    if exponent != preflight_exponent:
        raise ValueError("preflight and accepted scale ledgers differ")
    hashes = save_graph(output / "canonical", canonical, bases, global_exponent=exponent)
    if audit(SOURCES) != sources or COUNTS["prohibited_attempts"]:
        raise ValueError("source changed or prohibited route attempted")
    receipt = {"accepted": True, "scope": SCOPE, "full_policy": False, "libero_evaluated": False,
               "checkpoint_sha256": CHECKPOINT_SHA, "source_sha256": sources, "numpy": np.__version__,
               "canonical_hashes": hashes, "panel_sha256": digest(output / "panel.npz"),
               "input_seed": 20260906, "input_count": len(raw), "synthetic_inputs": True,
               "tolerances": TOLERANCES, "gates": gates, "kernels": dict(COUNTS),
               "global_binary_exponent": exponent, "environment_binary_exponent": 2 * exponent,
               "tensor_representation": "original ordered tensor = 2**global_binary_exponent times stored graph",
               "elapsed_seconds": time.monotonic() - started,
               "raw_internal_dimensions": schedule.raw_internal_dimensions,
               "canonical_internal_dimensions": schedule.canonical_internal_dimensions,
               "eligible_bonds": schedule.eligible_bonds, "clone_count": count,
               "dimension_axis": "unique nonleaf/nonroot bond directions, post shape-only direct QR"}
    write_json(output / "accepted.json", receipt)
    write_json(output / "baseline.json", dict(receipt, percent=0, valid_inputs=len(raw),
                                              maximum_error=gates["maximum_errors"]["gauged_replay"]))


def evaluate(output, percent, sources):
    receipt = json.loads((output / "accepted.json").read_text())
    if (receipt.get("accepted") is not True or receipt["source_sha256"] != sources
            or receipt["checkpoint_sha256"] != CHECKPOINT_SHA or receipt["tolerances"] != TOLERANCES
            or receipt["numpy"] != np.__version__ or digest(output / "panel.npz") != receipt["panel_sha256"]):
        raise ValueError("acceptance receipt mismatch")
    exponent = receipt["global_binary_exponent"]
    if type(exponent) is not int or receipt["environment_binary_exponent"] != 2 * exponent:
        raise ValueError("tensor/environment scale ledger mismatch")
    canonical, bases = load_graph(output / "canonical", receipt["canonical_hashes"], global_exponent=exponent)
    plan = next(p for p in rank_schedule(canonical).plans if p.percent == percent)
    with np.load(output / "panel.npz", allow_pickle=False) as data:
        raw, expected = data["raw"], data["expected"]
    inputs = homogeneous(raw)
    controls = [("leading", 0)]
    if percent in (50, 70):
        controls += [("trailing", 0)] + [("random", seed) for seed in (0, 1, 2)]
    for mode, seed in controls:
        started = time.monotonic()
        variant = physical_variant(canonical, bases, plan, mode, seed=seed)
        destination = output / f"p{percent}_{mode}_{seed}"
        hashes = save_graph(destination, variant.graph, global_exponent=exponent)
        reloaded, _ = load_graph(destination, hashes, global_exponent=exponent)
        physical = checked_chart(reloaded, inputs)
        masked = checked_chart(apply_bases(canonical, variant.full_bases), inputs, plan.ranks)
        if physical.valid_rows != masked.valid_rows or physical.failure_reasons != masked.failure_reasons:
            raise ValueError("physical/masked denominator classifications differ")
        equivalence = close(physical.projective, masked.projective,
                            TOLERANCES["coordinates"], "physical versus mask")
        valid = np.array(physical.valid_rows, dtype=bool)
        delta = physical.projective[valid, :-1] / physical.projective[valid, -1:] - expected[valid]
        if not np.isfinite(delta).all():
            raise ValueError("nonfinite decoded error")
        metrics = {"maximum_absolute_error": float(np.max(np.abs(delta))) if delta.size else None,
                   "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else None}
        if audit(SOURCES) != sources or COUNTS["prohibited_attempts"]:
            raise ValueError("source changed or prohibited route attempted")
        result = {"scope": SCOPE, "accepted_producer_sha256": digest(output / "accepted.json"),
                  "percent_requested": percent, "mode": mode, "seed": seed, "rank_mask_equivalence": equivalence,
                  "raw_internal_dimensions": receipt["raw_internal_dimensions"],
                  "canonical_internal_dimensions": plan.original_dimensions,
                  "kept_internal_dimensions": plan.retained_dimensions,
                  "post_qr_dimensions_removed_fraction": plan.actual_fraction,
                  "raw_dimensions_removed_fraction": 1 - plan.retained_dimensions / receipt["raw_internal_dimensions"],
                  "ranks": plan.ranks, "physical_artifact": hashes,
                  "global_binary_exponent": exponent,
                  "valid_inputs": int(np.sum(valid)), "input_count": len(raw),
                  "failure_reasons": physical.failure_reasons,
                  "relative_denominators": physical.relative_denominators,
                  "errors_on_valid_inputs_only": metrics, "elapsed_seconds": time.monotonic() - started,
                  "source_sha256": sources, "kernels": dict(COUNTS)}
        write_json(destination / "result.json", result)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--percent", type=int, choices=(30, 40, 50, 60, 70, 80))
    args = parser.parse_args()
    sources = audit(SOURCES)
    install_guards()
    if np.__version__ != "2.2.6":
        raise ValueError("pinned NumPy 2.2.6 required")
    if args.phase == "prepare":
        if args.checkpoint is None:
            parser.error("prepare requires --checkpoint")
        prepare(args.output, args.checkpoint, sources)
    else:
        if args.percent is None:
            parser.error("evaluate requires --percent")
        evaluate(args.output, args.percent, sources)


if __name__ == "__main__":
    main()
