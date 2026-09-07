"""Freeze and execute matched ODT consumers without changing accepted evidence.

Run ``python -m research.odt_campaign_v1.campaign prepare --help`` or ``task
--help``. All numerical constructors run under the reference runtime guards.
"""
import argparse
import ast
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from research.odt_reference.run_tests import COUNTS, PROHIBITED, install_guards
from research.odt_reference import run_curve as original
from research.odt_reference.block_oracle import block_forward
from research.odt_reference.curve import ChartEvaluation, RankPlan, _Event, rank_schedule, physical_variant
from research.odt_reference.shared_dag import apply_bases, occurrence_environment_sums
from scripts.odt_rank_ablation import consumer_audit, family, load_accepted


VERSION = "odt-campaign-v1"
BUDGETS = (93, 464, 927, 2782)
RANDOM_SEEDS = (0, 1, 2)
GAUSSIAN_SEED = 20260907
GAUSSIAN_COUNT = 256
POLICIES = ("W", "WN", "S", "SN")
ENVIRONMENT_EXPONENT = -25820
NATIVE_TRAINING_SHA = "e6c07beb2efb7e5ebe55c92fabf694d97b50b9e399f50868eeef4af455e30fa7"
NATIVE_CACHE_SHA = "053cf7e392054c4bc1ac0ea280828c3baf7f02a43e2feee22f27734956575662"
NATIVE_BLOCK_CONFIG = {"n_heads": 8, "eps": 1e-6, "width": 192, "tokens": 2,
                       "mask": [[1, 1], [1, 1]], "causal": False}
NATIVE_MANIFEST_TOLERANCES = {"absolute": 2e-4, "scaled": 2e-5,
                              "scale": "max(1,max(abs(reference)))", "require_both": True}
NATIVE_TOLERANCES = {"maximum_absolute_error": 2e-4, "scaled_error": 2e-5}
SPECTRAL_ROUNDOFF = 2e-12
FAILURE_CODES = {None: 0, "denominator_margin": 1, "zero_projective_row": 2,
                 "nonfinite_contraction": 3, "nonfinite_decoded_output": 4}


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def audit_campaign():
    """Bounded transitive source audit, including tests and the old consumer."""
    root = Path(__file__).resolve().parents[2]
    paths = (Path(__file__).with_name("__init__.py"), Path(__file__),
             root / "scripts/test_odt_campaign_v1.py")
    imports = {"argparse", "ast", "hashlib", "json", "math", "time", "numpy", "unittest", "tempfile"}
    from_imports = {"pathlib", "fractions", "research.odt_reference.run_tests",
                    "research.odt_reference", "research.odt_reference.block_oracle",
                    "research.odt_reference.curve", "research.odt_reference.shared_dag",
                    "scripts.odt_rank_ablation", "research.odt_campaign_v1.campaign"}
    hashes = {}
    for path in paths:
        tree = ast.parse(path.read_text())
        hashes[str(path.relative_to(root))] = original.digest(path)
        for node in ast.walk(tree):
            allowed = imports | ({"subprocess", "sys"} if path.name == "test_odt_campaign_v1.py" else set())
            if isinstance(node, ast.Import) and any(n.name not in allowed for n in node.names):
                raise ValueError("unreviewed campaign import")
            if isinstance(node, ast.ImportFrom) and node.module not in from_imports:
                raise ValueError("unreviewed campaign dependency")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in PROHIBITED:
                raise ValueError("prohibited campaign call")
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                for a, b in ((node.left, node.right), (node.right, node.left)):
                    if isinstance(b, ast.Attribute) and b.attr == "T" and ast.dump(a) == ast.dump(b.value):
                        raise ValueError("prohibited campaign self-overlap")
    return {"campaign_sources": hashes, "accepted_consumer": consumer_audit()}


def check_runtime():
    if np.__version__ != "2.2.6":
        raise ValueError("accepted reference runtime requires NumPy 2.2.6")
    if COUNTS["prohibited_attempts"]:
        raise ValueError("prohibited numerical route previously attempted")


def spectra_from_environments(graph, bases, *, environment_exponent):
    """Common-scale occurrence spectra with a declared negative-roundoff rule.

    Tiny negatives in [-2e-12 * maxabs(H), 0] are clipped to zero. Material
    negative values, unordered accepted bases, and failed eigen-equations stop
    allocation. Environments are never normalized or divided by occurrences.
    """
    if type(environment_exponent) is not int or environment_exponent != ENVIRONMENT_EXPONENT:
        raise ValueError("common environment scale ledger differs")
    if len(bases) != len(graph.nodes):
        raise ValueError("one accepted basis per node required")
    spectra, diagnostics = [], []
    for i, (environment, basis) in enumerate(zip(occurrence_environment_sums(graph), bases)):
        scale = float(np.max(np.abs(environment)))
        if not math.isfinite(scale):
            raise ValueError(f"nonfinite environment scale at node {i}")
        if basis.shape != environment.shape or not np.isfinite(basis).all():
            raise ValueError("accepted basis dimensions or finiteness differ")
        values = np.diag(basis.T @ environment @ basis).copy()
        residual = environment @ basis - basis * values
        error = float(np.max(np.abs(residual))) / max(scale, 1e-12)
        if not math.isfinite(error) or error > original.TOLERANCES["subspace"]:
            raise ValueError(f"accepted basis eigen-equation failed at node {i}")
        bound = SPECTRAL_ROUNDOFF * scale
        if not np.isfinite(values).all() or float(np.min(values)) < -bound:
            raise ValueError(f"material indefinite environment at node {i}")
        if np.any(np.diff(values) > bound):
            raise ValueError(f"accepted spectral prefix is unordered at node {i}")
        tiny_negative = int(np.sum(values < 0))
        minimum_raw = float(np.min(values))
        values = np.maximum(values, 0.)
        # QR/EVD roundoff may invert nearly tied entries. Costs use a monotone
        # envelope, retaining the accepted prefix order and deterministic ties.
        # Only differences already within the declared bound may be adjusted.
        adjusted = np.maximum.accumulate(values[::-1])[::-1]
        if np.max(adjusted - values) > bound:
            raise ValueError("spectral monotonic correction exceeded roundoff bound")
        spectra.append(adjusted)
        diagnostics.append({"node": i, "eigen_equation_residual": error,
                            "environment_max_abs": scale, "minimum_raw_eigenvalue": minimum_raw,
                            "clipped_negative_count": tiny_negative,
                            "monotonic_adjustment_max": float(np.max(adjusted-values))})
    return tuple(spectra), tuple(diagnostics)


def allocation(graph, spectra, policy, budget):
    """Exactly K tail cuts, with one common 9274-dimensional denominator."""
    base = rank_schedule(graph, percents=(0,)).plans[0]
    if policy not in POLICIES or type(budget) is not int or budget < 0:
        raise ValueError("declared policy and nonnegative integer budget required")
    protected = family(base, graph.nodes, "norm") if policy.endswith("N") else ()
    eligible = tuple(i for i in base.eligible if i not in protected)
    if budget > sum(base.widths[i]-1 for i in eligible):
        raise ValueError("protected budget is unattainable")
    ranks = list(base.widths)
    if policy.startswith("W"):
        events = sorted(_Event(2*j-1, 2*base.widths[i], i)
                        for i in eligible for j in range(1, base.widths[i]))
        for event in events[:budget]:
            ranks[event.node] -= 1
    else:
        if len(spectra) != len(base.widths):
            raise ValueError("one full ordered spectrum per node required")
        for spectrum, width in zip(spectra, base.widths):
            if (spectrum.shape != (width,) or not np.isfinite(spectrum).all()
                    or np.any(spectrum < 0) or np.any(np.diff(spectrum) > 0)):
                raise ValueError("nonnegative descending spectra required")
        # Larger column indices are removed first for equal costs on one node.
        # Across nodes use origin index. No rank-zero event exists.
        events = sorted((float(spectra[i][column]), i, -column)
                        for i in eligible for column in range(1, base.widths[i]))
        for _, i, negative_column in events[:budget]:
            if -negative_column != ranks[i]-1:
                raise ValueError("spectral tail event violated prefix admissibility")
            ranks[i] -= 1
    plan = RankPlan(-1, tuple(ranks), base.widths, base.eligible, base.signature,
                    base.original_dimensions, budget)
    if sum(base.widths[i]-plan.ranks[i] for i in base.eligible) != budget:
        raise ValueError("allocation did not match exact removal budget")
    return plan


def freeze_conditions(graph, spectra, budgets=BUDGETS, random_seeds=RANDOM_SEEDS):
    """Deduplicate complete ranks + construction + seed, preserving aliases."""
    zero = allocation(graph, spectra, "W", 0)
    conditions, lookup = [], {}

    def add(plan, construction, seed, alias):
        identity = {"ranks": plan.ranks, "construction": construction, "seed": seed}
        key = json_digest(identity)
        if key in lookup:
            conditions[lookup[key]]["aliases"].append(alias)
            return lookup[key]
        index = len(conditions)
        lookup[key] = index
        conditions.append({"task_index": index, "condition_sha256": key,
                           **identity, "aliases": [alias], "removed_dimensions": plan.removed_dimensions,
                           "original_dimensions": plan.original_dimensions,
                           "rank_sha256": json_digest(plan.ranks)})
        return index

    add(zero, "leading", None, {"policy": "full", "budget": 0})
    for policy in POLICIES:
        for budget in budgets:
            plan = allocation(graph, spectra, policy, budget)
            add(plan, "leading", None, {"policy": policy, "budget": budget})
            if budget in (927, 2782):
                for seed in random_seeds:
                    add(plan, "zero_input_anchored_direct_qr", seed, {"policy": policy, "budget": budget})
    aliases = {(alias["policy"], alias["budget"], condition["construction"], condition["seed"]):
               condition["task_index"] for condition in conditions for alias in condition["aliases"]}
    contrasts = []
    for budget in budgets:
        for a, b in (("S", "W"), ("SN", "WN"), ("WN", "W"), ("SN", "S")):
            contrasts.append({"name": f"{a}-versus-{b}-K{budget}",
                              "a": aliases[(a, budget, "leading", None)],
                              "b": aliases[(b, budget, "leading", None)],
                              "comparison": "leading_allocation"})
        if budget in (927, 2782):
            for policy in POLICIES:
                for seed in random_seeds:
                    contrasts.append({"name": f"{policy}-leading-versus-anchored{seed}-K{budget}",
                                      "a": aliases[(policy, budget, "leading", None)],
                                      "b": aliases[(policy, budget, "zero_input_anchored_direct_qr", seed)],
                                      "comparison": "identical_per_bond_ranks"})
    return conditions, contrasts


def zero_activations(graph):
    """Physical zero means the last ingress coordinate is one, never zero."""
    values = []
    for i, node in enumerate(graph.nodes):
        if not node.children:
            raw = np.zeros(node.core.shape[1])
            raw[-1] = 1.
            value = node.core @ raw
        elif len(node.children) == 1:
            value = node.core @ values[node.children[0]]
        else:
            value = np.einsum("oij,i,j->o", node.core,
                              values[node.children[0]], values[node.children[1]])
        size = float(np.max(np.abs(value)))
        if not math.isfinite(size) or size == 0:
            raise ValueError(f"zero-input anchor is zero or nonfinite at node {i}, shape={value.shape}")
        values.append(value / size)
    return tuple(values)


def anchored_bases(graph, spectral_bases, seed):
    """Retain the full direct Q, including its Householder completion."""
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer seed required")
    plan = rank_schedule(graph, percents=(0,)).plans[0]
    anchors = zero_activations(graph)
    rng, controls, diagnostics = np.random.default_rng(seed), [], []
    for i, (width, basis, anchor) in enumerate(zip(plan.widths, spectral_bases, anchors)):
        if i not in plan.eligible:
            controls.append(np.array(basis, copy=True))
            continue
        columns = rng.normal(size=(width, width))
        columns[:, 0] = anchor
        q, triangular = np.linalg.qr(columns, mode="reduced")
        reconstruction = original.close(q @ triangular, columns, original.TOLERANCES["coordinates"],
                                         "anchored direct-QR reconstruction")
        anchor_error = original.close(q[:, 0]*triangular[0, 0], anchor,
                                       original.TOLERANCES["coordinates"], "retained zero-input anchor")
        if q.shape != (width, width) or not np.isfinite(q).all():
            raise ValueError("full direct-Q completion was not retained")
        controls.append(q)
        diagnostics.append({"node": i, "qr_reconstruction": reconstruction, "anchor_error": anchor_error})
    return tuple(controls), tuple(diagnostics)


def scaled_chart(graph, inputs, *, global_exponent=0):
    """Reference contractions plus a per-sample fixed-lift log2 scale ledger.

    Every repeated child contributes its scale once per syntactic occurrence.
    No amplitude is reconstructed by exponentiating the large global scale.
    """
    if type(global_exponent) is not int:
        raise ValueError("integer complete-tensor exponent required")
    sources = {n.source: n.core.shape[1] for n in graph.nodes if not n.children}
    if set(inputs) != set(sources):
        raise ValueError("physical source keys differ")
    batches = {key: np.asarray(value) for key, value in inputs.items()}
    if any(value.ndim != 2 or value.shape[1] != sources[key] or not len(value)
           or not np.isfinite(value).all() or np.iscomplexobj(value) for key, value in batches.items()):
        raise ValueError("finite homogeneous input batches required")
    count = len(next(iter(batches.values())))
    if any(len(value) != count for value in batches.values()):
        raise ValueError("equal input batch sizes required")
    rows, valid, reasons, margins, log_scales = [], [], [], [], []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for sample in range(count):
            values, scales = [], []
            for node in graph.nodes:
                if not node.children:
                    value = node.core @ batches[node.source][sample]
                    inherited = 0.
                elif len(node.children) == 1:
                    value = node.core @ values[node.children[0]]
                    inherited = scales[node.children[0]]
                else:
                    value = np.einsum("oij,i,j->o", node.core,
                                      values[node.children[0]], values[node.children[1]])
                    inherited = sum(scales[child] for child in node.children)
                if not np.isfinite(value).all():
                    raise ValueError("nonfinite contraction stops the affected lane")
                size = float(np.max(np.abs(value)))
                values.append(value / size if size > 0 else value)
                scales.append(inherited + math.log2(size) if size > 0 else -math.inf)
            output = graph.head @ values[-1]
            if not np.isfinite(output).all():
                raise ValueError("nonfinite head contraction stops the affected lane")
            size = float(np.max(np.abs(output)))
            output = output / size if size > 0 else output
            margin = float(abs(output[-1])) if size > 0 else 0.
            reason = ("zero_projective_row" if size == 0 else "denominator_margin"
                      if margin <= original.TOLERANCES["denominator_margin"] else None)
            rows.append(output)
            valid.append(reason is None)
            margins.append(margin)
            reasons.append(reason)
            log_scales.append(scales[-1] + math.log2(size) + global_exponent if size > 0 else -math.inf)
        projective, log_scale = np.array(rows), np.array(log_scales)
        decoded = projective[:, :-1] / projective[:, -1:] if all(valid) else None
        if decoded is not None and not np.isfinite(decoded).all():
            raise ValueError("nonfinite decoded execution stops the affected lane")
        diagnostic = {"output_log2_scale": log_scale,
                      "log2_abs_denominator": log_scale + np.log2(np.abs(projective[:, -1])),
                      "log2_numerator_max_abs": log_scale + np.log2(np.max(np.abs(projective[:, :-1]), axis=1)),
                      "denominator_sign": np.sign(projective[:, -1]).astype(np.int8)}
    return ChartEvaluation(projective, tuple(valid), tuple(margins), tuple(reasons), decoded), diagnostic


def chart_arrays(chart):
    valid = np.array(chart.valid_rows, dtype=bool)
    decoded = np.full((len(valid), chart.projective.shape[1]-1), np.nan)
    decoded[valid] = chart.projective[valid, :-1] / chart.projective[valid, -1:]
    if not np.isfinite(decoded[valid]).all():
        raise ValueError("nonfinite decoded valid output")
    return {"projective": chart.projective, "decoded": decoded, "valid": valid,
            "failure_code": np.array([FAILURE_CODES[x] for x in chart.failure_reasons], dtype=np.int8),
            "relative_denominator": np.array(chart.relative_denominators)}


def error_metrics(decoded, valid, reference, reference_valid, selection=None):
    """Unconditional invalidity and explicitly conditional decoded errors."""
    selection = np.ones(len(valid), dtype=bool) if selection is None else np.asarray(selection, dtype=bool)
    if selection.shape != valid.shape or not np.any(selection):
        raise ValueError("nonempty aligned selection required")
    joint = selection & valid & reference_valid
    delta = decoded[joint] - reference[joint]
    if not np.isfinite(delta).all():
        raise ValueError("nonfinite jointly valid error")
    mse = np.mean(delta*delta, axis=1)
    rmse = float(np.sqrt(np.mean(mse))) if len(mse) else None
    source_rows = selection & reference_valid
    source_rms = float(np.sqrt(np.mean(reference[source_rows]**2))) if np.any(source_rows) else None
    return {"input_count": int(np.sum(selection)), "invalid_count": int(np.sum(selection & ~valid)),
            "invalid_probability_unconditional": float(np.mean(~valid[selection])),
            "reference_invalid_count": int(np.sum(selection & ~reference_valid)),
            "jointly_valid_count": int(np.sum(joint)), "decoded_errors_conditional_on_joint_validity": True,
            "rmse": rmse, "source_output_rms": source_rms,
            "rmse_over_source_rms": rmse/source_rms if rmse is not None and source_rms else None,
            "p95_per_example_rmse": float(np.percentile(np.sqrt(mse), 95)) if len(mse) else None,
            "maximum_per_example_rmse": float(np.max(np.sqrt(mse))) if len(mse) else None,
            "maximum_absolute_error": float(np.max(np.abs(delta))) if delta.size else None}


def cutoff_diagnostics(condition, spectra, diagnostics):
    result = []
    for i, rank in enumerate(condition["ranks"]):
        values = spectra[i]
        if rank == len(values):
            continue
        gap = float(values[rank-1]-values[rank])
        scale = max(float(np.max(np.abs(values))), 1e-12)
        result.append({"node": i, "rank": rank, "width": len(values), "eigengap": gap,
                       "unresolved": gap <= 1e-8*scale,
                       "eigen_equation_residual": diagnostics[i]["eigen_equation_residual"],
                       "tail_cost_stored_environment_units": float(np.sum(values[rank:]))})
    return result


def validate_record_partitions(records, development_records):
    """Retain the frozen 80 confirmation episode clusters and 20 dev clusters."""
    required = {"task_id", "episode_id", "frame_id", "pair_id", "pair_positions"}
    episode_sets = []
    for rows, expected_episodes in ((records, 8), (development_records, 2)):
        if len(rows) != 10*expected_episodes*4:
            raise ValueError("native confirmation/development row count differs")
        groups, identities = {}, set()
        for record in rows:
            if not required <= set(record):
                raise ValueError("aligned task/episode/frame/pair records required")
            if any(type(record[key]) is not int for key in ("task_id", "episode_id", "frame_id", "pair_id")):
                raise ValueError("integer native row identities required")
            task_id, episode, frame, pair = (record[key] for key in ("task_id", "episode_id", "frame_id", "pair_id"))
            if not 0 <= task_id < 10 or episode < 0 or frame < 0 or pair not in (0, 1):
                raise ValueError("native row identity outside declared panel")
            if record["pair_positions"] != ([27, 28] if pair == 0 else [0, 63]):
                raise ValueError("predeclared adjacent/distant token pair differs")
            identity = (task_id, episode, frame, pair)
            if identity in identities:
                raise ValueError("duplicate real panel identity")
            identities.add(identity)
            frames = groups.setdefault((task_id, episode), {})
            frames.setdefault(frame, set()).add(pair)
        if any(len(frames) != 2 or any(pairs != {0, 1} for pairs in frames.values()) for frames in groups.values()):
            raise ValueError("each episode requires two frames and both token pairs")
        if any(sum(t == task_id for t, _ in groups) != expected_episodes for task_id in range(10)):
            raise ValueError("native panel task/episode balance differs")
        episode_sets.append(set(groups))
    if episode_sets[0] & episode_sets[1]:
        raise ValueError("development and confirmation episodes overlap")


def validate_native_panel(path, metadata_path):
    metadata = json.loads(metadata_path.read_text())
    if metadata["panel_sha256"] != original.digest(path) or metadata["checkpoint_sha256"] != original.CHECKPOINT_SHA:
        raise ValueError("native panel authentication mismatch")
    if metadata.get("passed") is not True or metadata.get("schema") != "odt-real-panel-v1":
        raise ValueError("native bridge producer did not pass")
    if (metadata.get("training_sha256") != NATIVE_TRAINING_SHA
            or metadata.get("cache_sha256") != NATIVE_CACHE_SHA
            or metadata.get("block_config") != NATIVE_BLOCK_CONFIG
            or metadata.get("native_parity_tolerances") != NATIVE_MANIFEST_TOLERANCES
            or metadata.get("environment", {}).get("numpy") != "1.26.4"
            or metadata.get("environment", {}).get("torch") != "2.7.1+cu126"):
        raise ValueError("native producer configuration or runtime differs")
    required_parity = {"native_two_vs_numpy", "native64_vs_numpy", "native_post_attention_vs_numpy", "native_two_vs_pair_mask64"}
    for split in ("development", "confirmation"):
        parity = metadata.get("parity", {}).get(split, {})
        if not required_parity <= set(parity) or any(parity[key].get("passed") is not True for key in required_parity):
            raise ValueError("native two-token/context bridge gate failed")
    if metadata.get("freeze_file") != "panel_freeze.json":
        raise ValueError("native identity freeze filename differs")
    freeze_path = metadata_path.parent/"panel_freeze.json"
    if original.digest(freeze_path) != metadata["freeze_sha256"]:
        raise ValueError("native identity freeze authentication differs")
    freeze = json.loads(freeze_path.read_text())
    if (freeze.get("schema") != "odt-real-panel-freeze-v1"
            or any(freeze.get(key) != metadata[key] for key in
                   ("checkpoint_sha256", "training_sha256", "cache_sha256", "native_parity_tolerances",
                    "records", "development_records"))):
        raise ValueError("native identity freeze differs from final manifest")
    records = metadata["records"]
    validate_record_partitions(records, metadata["development_records"])
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    raw = arrays["real_inputs"]
    if raw.dtype != np.float64 or raw.shape != (320, 2, 192) or not np.isfinite(raw).all():
        raise ValueError("native real panel must be finite float64 [320,2,192]")
    for prefix, rows in (("", records), ("dev_", metadata["development_records"])):
        for field in ("task", "episode", "frame", "pair"):
            values = arrays[prefix+field+"_ids"]
            if values.dtype != np.int64 or not np.array_equal(values, [r[field+"_id"] for r in rows]):
                raise ValueError("NPZ and JSON native row identities differ")
        pairs = arrays[prefix+"pair_positions"]
        if pairs.dtype != np.int64 or not np.array_equal(pairs, [r["pair_positions"] for r in rows]):
            raise ValueError("NPZ and JSON token pair coordinates differ")
    for key in ("native_two_token", "native_context_rows"):
        if arrays[key].shape not in ((320, 2, 192), (320, 384)) or not np.isfinite(arrays[key]).all():
            raise ValueError("native output panel shape or finiteness differs")
        arrays[key] = arrays[key].reshape(320, 384)
    weights = {name[len("weight__"):]: value for name, value in arrays.items() if name.startswith("weight__")}
    if not weights or weights["ffn.left.weight"].shape != (576, 192):
        raise ValueError("authenticated first-block weight boundary absent")
    return arrays, metadata, weights


def native_parity(native, expected):
    maximum = float(np.max(np.abs(native-expected)))
    scaled = maximum / max(1., float(np.max(np.abs(expected))))
    if (not math.isfinite(maximum) or maximum > NATIVE_TOLERANCES["maximum_absolute_error"]
            or scaled > NATIVE_TOLERANCES["scaled_error"]):
        raise ValueError(f"predeclared native float32 parity gate failed: absolute={maximum}, scaled={scaled}")
    return {"maximum_absolute_error": maximum, "scaled_error": scaled,
            "tolerances": NATIVE_TOLERANCES}


def prepare(producer, real_panel, real_metadata, output):
    started, sources = time.monotonic(), audit_campaign()
    check_runtime()
    receipt, canonical, spectral_bases, old_raw, old_expected = load_accepted(producer, sources["accepted_consumer"])
    native, metadata, weights = validate_native_panel(real_panel, real_metadata)
    spectra, spectral_diagnostics = spectra_from_environments(canonical, spectral_bases,
                                         environment_exponent=receipt["environment_binary_exponent"])
    conditions, contrasts = freeze_conditions(canonical, spectra)
    synthetic = np.random.default_rng(GAUSSIAN_SEED).normal(size=(GAUSSIAN_COUNT, 2, 192)) * .2
    raw = np.concatenate((native["real_inputs"], synthetic), axis=0)
    records = [dict(record, panel="real") for record in metadata["records"]]
    records += [{"panel": "gaussian", "task_id": "gaussian", "episode_id": i,
                 "frame_id": i, "pair_id": i} for i in range(GAUSSIAN_COUNT)]
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "inputs.npz", raw=raw)
    original.write_json(output / "records.json", records)
    protocol = {"version": VERSION, "sources": sources, "producer_receipt_sha256": original.digest(producer/"accepted.json"),
                "checkpoint_sha256": receipt["checkpoint_sha256"], "global_binary_exponent": receipt["global_binary_exponent"],
                "environment_binary_exponent": receipt["environment_binary_exponent"],
                "real_panel_sha256": original.digest(real_panel), "real_metadata_sha256": original.digest(real_metadata),
                "inputs_sha256": original.digest(output/"inputs.npz"), "records_sha256": original.digest(output/"records.json"),
                "gaussian_seed": GAUSSIAN_SEED, "gaussian_count": GAUSSIAN_COUNT,
                "gaussian_standard_deviation": .2, "native_parity_tolerances": NATIVE_TOLERANCES,
                "conditions": conditions, "contrasts": contrasts, "original_dimensions": 9274,
                "minimum_internal_rank": 1, "ingress_and_root_full_rank": True,
                "spectral_roundoff_relative_bound": SPECTRAL_ROUNDOFF,
                "spectral_roundoff_rule": "clip tiny negative values, then bounded descending monotone envelope",
                "spectral_cost_interpretation": "sum of independent occurrence-cut coefficient tail costs, not simultaneous decoded error",
                "uncertainty_unit": "task-stratified episode cluster for real data",
                "chart_margin_interpretation": "output-amplitude statistic, not a pole or conditioning certificate",
                "scope": "accepted first vision block with joint two-token context; real training-source activations plus Gaussian inputs",
                "full_policy": False, "libero_rollouts": False}
    # This file freezes input identities, all ranks, seeds and contrasts BEFORE
    # evaluating any reduced graph or using confirmation outputs.
    original.write_json(output / "protocol.json", protocol)
    print(json.dumps({"phase": "protocol_frozen", "conditions": len(conditions),
                      "protocol_sha256": original.digest(output/"protocol.json")}), flush=True)
    basis_arrays = {f"spectral_{i}": basis for i, basis in enumerate(spectral_bases)}
    basis_arrays.update({f"spectrum_{i}": values for i, values in enumerate(spectra)})
    anchor_diagnostics = {}
    for seed in RANDOM_SEEDS:
        controls, diagnostic = anchored_bases(canonical, spectral_bases, seed)
        basis_arrays.update({f"anchor_{seed}_{i}": basis for i, basis in enumerate(controls)})
        anchor_diagnostics[str(seed)] = diagnostic
    np.savez(output/"bases.npz", **basis_arrays)
    original.write_json(output/"spectral_diagnostics.json", spectral_diagnostics)
    original.write_json(output/"anchor_diagnostics.json", anchor_diagnostics)
    # Independent raw-weight forward includes all new Gaussian and real inputs.
    expected = block_forward(raw, weights, n_heads=8, mask=np.ones((2, 2)), eps=1e-6).reshape(len(raw), -1)
    native_error = native_parity(native["native_two_token"], expected[:320])
    baseline, scales = scaled_chart(canonical, original.homogeneous(raw),
                                    global_exponent=receipt["global_binary_exponent"])
    if baseline.decoded is None:
        raise ValueError("real or new Gaussian full-rank chart failed closed")
    full_rank_error = original.close(baseline.decoded, expected, original.TOLERANCES["replay"], "new full-rank panel replay")
    # Small original panel verifies the immutable producer independently.
    old_error = original.full_rank_replay(canonical, original.homogeneous(old_raw), old_expected, "old panel reproduction")
    reference_chart = original.checked_chart(canonical, original.homogeneous(raw))
    original.close(baseline.projective, reference_chart.projective, original.TOLERANCES["coordinates"], "scale-ledger evaluator replay")
    if baseline.valid_rows != reference_chart.valid_rows or baseline.failure_reasons != reference_chart.failure_reasons:
        raise ValueError("scale-ledger evaluator validity differs")
    arrays = dict(chart_arrays(baseline), **scales, expected=expected,
                  native_two_token=native["native_two_token"], native_context_rows=native["native_context_rows"])
    np.savez(output/"baseline.npz", **arrays)
    context_metrics = error_metrics(native["native_two_token"], np.ones(320, dtype=bool),
                                   native["native_context_rows"], np.ones(320, dtype=bool))
    if audit_campaign() != sources or COUNTS["prohibited_attempts"]:
        raise ValueError("source changed or prohibited numerical route attempted")
    artifacts = {name: original.digest(output/name) for name in
                 ("protocol.json", "inputs.npz", "records.json", "bases.npz", "spectral_diagnostics.json",
                  "anchor_diagnostics.json", "baseline.npz")}
    manifest = {"ready": True, "version": VERSION, "artifacts": artifacts,
                "native_two_token_parity": native_error, "new_full_rank_replay_error": full_rank_error,
                "original_panel_reproduction_error": old_error, "context_removal_error": context_metrics,
                "condition_count": len(conditions), "elapsed_seconds": time.monotonic()-started, "kernels": dict(COUNTS)}
    original.write_json(output/"manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False), flush=True)


def load_campaign(path):
    manifest = json.loads((path/"manifest.json").read_text())
    if manifest.get("ready") is not True or manifest.get("version") != VERSION:
        raise ValueError("frozen campaign is not ready")
    required = {"protocol.json", "inputs.npz", "records.json", "bases.npz", "spectral_diagnostics.json",
                "anchor_diagnostics.json", "baseline.npz"}
    if set(manifest["artifacts"]) != required:
        raise ValueError("frozen campaign artifact closure differs")
    for name, expected in manifest["artifacts"].items():
        if original.digest(path/name) != expected:
            raise ValueError("frozen campaign artifact hash mismatch")
    protocol = json.loads((path/"protocol.json").read_text())
    if protocol["sources"] != audit_campaign():
        raise ValueError("frozen consumer source closure changed")
    if (protocol["inputs_sha256"] != manifest["artifacts"]["inputs.npz"]
            or protocol["records_sha256"] != manifest["artifacts"]["records.json"]):
        raise ValueError("protocol input identity differs")
    return manifest, protocol


def task(producer, campaign, output, task_index):
    started = time.monotonic()
    check_runtime()
    manifest, protocol = load_campaign(campaign)
    sources = protocol["sources"]
    receipt, canonical, accepted_bases, _, _ = load_accepted(producer, sources["accepted_consumer"])
    if (protocol["producer_receipt_sha256"] != original.digest(producer/"accepted.json")
            or protocol["global_binary_exponent"] != receipt["global_binary_exponent"]
            or protocol["environment_binary_exponent"] != receipt["environment_binary_exponent"]):
        raise ValueError("producer identity or scale ledger differs")
    if type(task_index) is not int or not 0 <= task_index < len(protocol["conditions"]):
        raise ValueError("frozen task index out of range")
    condition = protocol["conditions"][task_index]
    if (condition["task_index"] != task_index
            or condition["rank_sha256"] != json_digest(condition["ranks"])
            or condition["condition_sha256"] != json_digest({key: condition[key] for key in ("ranks", "construction", "seed")})):
        raise ValueError("condition identity mismatch")
    with np.load(campaign/"inputs.npz", allow_pickle=False) as saved:
        raw = saved["raw"]
    with np.load(campaign/"baseline.npz", allow_pickle=False) as saved:
        baseline = {key: saved[key] for key in saved.files}
    with np.load(campaign/"bases.npz", allow_pickle=False) as saved:
        prefix = "spectral" if condition["construction"] == "leading" else f"anchor_{condition['seed']}"
        full_bases = tuple(saved[f"{prefix}_{i}"] for i in range(len(canonical.nodes)))
        spectra = tuple(saved[f"spectrum_{i}"] for i in range(len(canonical.nodes)))
        for i, accepted in enumerate(accepted_bases):
            if not np.array_equal(accepted, saved[f"spectral_{i}"]):
                raise ValueError("accepted basis identity changed")
    base = rank_schedule(canonical, percents=(0,)).plans[0]
    plan = RankPlan(-1, tuple(condition["ranks"]), base.widths, base.eligible, base.signature,
                    condition["original_dimensions"], condition["removed_dimensions"])
    variant = physical_variant(canonical, full_bases, plan)
    zero = original.homogeneous(np.zeros((1, 2, 192)))
    zero_replay = None
    if condition["construction"] == "zero_input_anchored_direct_qr":
        full_zero = original.checked_chart(canonical, zero)
        cut_zero = original.checked_chart(variant.graph, zero)
        if full_zero.decoded is None or cut_zero.decoded is None:
            raise ValueError("zero-input whole-graph control replay chart failed")
        zero_replay = original.close(cut_zero.decoded, full_zero.decoded,
                                      original.TOLERANCES["replay"], "zero-input whole-graph replay")
    hashes = original.save_graph(output, variant.graph, global_exponent=receipt["global_binary_exponent"])
    reloaded, _ = original.load_graph(output, hashes, global_exponent=receipt["global_binary_exponent"])
    inputs = original.homogeneous(raw)
    physical, scales = scaled_chart(reloaded, inputs, global_exponent=receipt["global_binary_exponent"])
    masked = original.checked_chart(apply_bases(canonical, variant.full_bases), inputs, plan.ranks)
    if physical.valid_rows != masked.valid_rows or physical.failure_reasons != masked.failure_reasons:
        raise ValueError("physical/masked validity classifications differ")
    equivalence = original.close(physical.projective, masked.projective,
                                 original.TOLERANCES["coordinates"], "physical/mask every-occurrence replay")
    arrays = dict(chart_arrays(physical), **scales)
    joint = arrays["valid"] & baseline["valid"]
    arrays["squared_error"] = np.full(len(raw), np.nan)
    arrays["squared_error"][joint] = np.mean((arrays["decoded"][joint]-baseline["decoded"][joint])**2, axis=1)
    cross = arrays["projective"][:, :-1]*baseline["projective"][:, -1:] - baseline["projective"][:, :-1]*arrays["projective"][:, -1:]
    cross_size = np.max(np.abs(cross), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        arrays["log2_cross_product_numerator_max_abs"] = np.log2(cross_size) + scales["output_log2_scale"] + baseline["output_log2_scale"]
        arrays["delta_log2_abs_denominator"] = scales["log2_abs_denominator"] - baseline["log2_abs_denominator"]
        arrays["delta_log2_numerator_max_abs"] = scales["log2_numerator_max_abs"] - baseline["log2_numerator_max_abs"]
    arrays["cross_product_normalized_mantissa"] = cross
    np.savez(output/"predictions.npz", **arrays)
    records = json.loads((campaign/"records.json").read_text())
    metrics = {"all": error_metrics(arrays["decoded"], arrays["valid"], baseline["decoded"], baseline["valid"])}
    for name in ("real", "gaussian"):
        selection = np.array([record["panel"] == name for record in records])
        metrics[name] = error_metrics(arrays["decoded"], arrays["valid"], baseline["decoded"], baseline["valid"], selection)
    task_metrics = []
    real_tasks = sorted({json.dumps(record["task_id"]) for record in records if record["panel"] == "real"})
    for task_id in real_tasks:
        selection = np.array([record["panel"] == "real" and json.dumps(record["task_id"]) == task_id for record in records])
        task_metrics.append({"task_id": json.loads(task_id),
                             **error_metrics(arrays["decoded"], arrays["valid"], baseline["decoded"], baseline["valid"], selection)})
    full_rank_error = original.close(arrays["decoded"], baseline["decoded"], original.TOLERANCES["replay"],
                                     "full-rank task replay") if plan.removed_dimensions == 0 else None
    diagnostics = json.loads((campaign/"spectral_diagnostics.json").read_text())
    if audit_campaign() != sources or COUNTS["prohibited_attempts"]:
        raise ValueError("source changed or prohibited numerical route attempted")
    result = {"completed": True, "version": VERSION, "condition": condition, "metrics": metrics,
              "task_metrics": task_metrics, "cutoffs": cutoff_diagnostics(condition, spectra, diagnostics),
              "new_cutoffs_independently_clone_checked": False,
              "zero_input_whole_graph_replay_error": zero_replay, "physical_mask_equivalence": equivalence,
              "full_rank_replay_error": full_rank_error,
              "physical_artifact": hashes, "predictions_sha256": original.digest(output/"predictions.npz"),
              "campaign_manifest_sha256": original.digest(campaign/"manifest.json"),
              "protocol_sha256": manifest["artifacts"]["protocol.json"],
              "basis_artifact_sha256": manifest["artifacts"]["bases.npz"], "source_sha256": sources,
              "producer_receipt_sha256": protocol["producer_receipt_sha256"],
              "global_binary_exponent": receipt["global_binary_exponent"],
              "environment_binary_exponent": receipt["environment_binary_exponent"],
              "failure_codes": {str(key): value for key, value in FAILURE_CODES.items()},
              "fixed_lift_scale_diagnostic": True, "full_policy": False, "libero_rollouts": False,
              "elapsed_seconds": time.monotonic()-started, "kernels": dict(COUNTS)}
    original.write_json(output/"result.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def main():
    # Both ``python -m ...`` and direct script entry audit then guard before
    # argument dispatch or any numerical constructor. Library callers and tests
    # must explicitly install their surrounding runtime guard before calling.
    audit_campaign()
    install_guards()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--producer", type=Path, required=True)
    prep.add_argument("--real-panel", type=Path, required=True)
    prep.add_argument("--real-metadata", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    worker = commands.add_parser("task")
    worker.add_argument("--producer", type=Path, required=True)
    worker.add_argument("--campaign", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--task-index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.producer, args.real_panel, args.real_metadata, args.output)
    else:
        task(args.producer, args.campaign, args.output, args.task_index)


if __name__ == "__main__":
    main()
