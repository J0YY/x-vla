#!/usr/bin/env python3
"""Small, guarded timing study of canonical QR and contracted-environment EVD."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.odt_direct_only_compliance import (
    assert_direct_only_runtime_guard,
    audit_direct_only_launch,
    direct_only_runtime_report,
    install_direct_only_runtime_guard,
)

AUDIT = audit_direct_only_launch(ROOT, (Path(__file__),))
for field in ("prohibited_calls_found", "prohibited_self_overlap_sites", "guarded_dormant_spectral_norm_sites", "duplicate_top_level_definition_sites"):
    if AUDIT[field]:
        raise RuntimeError(f"canonical timing source audit rejected {field}")
install_direct_only_runtime_guard()

import torch

from xvla.train.implicit_sparse_projective_odt import (
    ImplicitProjectiveDAG,
    MaterializationTelemetry,
    PhysicalSourceSpec,
    _Builder,
    _positive_diagonal_direct_rq_rows,
    canonicalize_implicit_dag_direct_rq,
    diagonalize_implicit_dag_full_rank,
)


def _motif():
    generator = torch.Generator().manual_seed(291)
    dtype = torch.float64
    builder = _Builder(torch.empty((), dtype=dtype), MaterializationTelemetry())
    source = builder.physical_pair(PhysicalSourceSpec("x", "state", 384))
    matrix = torch.eye(385, dtype=dtype) + 0.01 * torch.randn(385, 385, generator=generator, dtype=dtype)
    shared = builder.unary("internal.385", matrix, source)
    root = builder.unary("root", torch.randn(2, 385, generator=generator, dtype=dtype), shared)
    network = ImplicitProjectiveDAG(root=root, head=torch.eye(2, dtype=dtype), head_binary_exponent=0, token_count=1, feature_dimension=384, selected_token=0, mask=torch.ones(1, 1, dtype=dtype), physical_sources=builder.physical_source_specs)
    raw = {"x": torch.randn(1, 384, generator=generator, dtype=dtype)}
    return network, raw


def _seconds(function, repetitions):
    values = []
    for _ in range(3):
        started = time.perf_counter()
        for _ in range(repetitions):
            function()
        values.append((time.perf_counter() - started) / repetitions)
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    generator = torch.Generator().manual_seed(118)
    cases = []
    for rows, columns, repetitions in ((1, 1, 128), (2, 3, 128), (385, 386, 4)):
        matrix = torch.randn(rows, columns, generator=generator, dtype=torch.float64)
        if rows == 385:
            matrix[-1] = matrix[0] + matrix[1]
        cases.append((rows, columns, repetitions, matrix))
    records = []
    for threads in (1, 2, 4, 8, 16):
        torch.set_num_threads(threads)
        record = {"threads": threads, "direct_rq": []}
        for rows, columns, repetitions, matrix in cases:
            factor, q_rows = _positive_diagonal_direct_rq_rows(matrix)
            reconstruction = factor @ q_rows
            error = float((reconstruction - matrix).abs().max().item())
            if error > 2e-10:
                raise RuntimeError("direct QR reconstruction failed")
            elapsed = _seconds(lambda: _positive_diagonal_direct_rq_rows(matrix), repetitions)
            record["direct_rq"].append({"shape": [rows, columns], "seconds_per_factorization": elapsed, "reconstruction_max_error": error})
        network, raw = _motif()
        started = time.perf_counter()
        canonical = canonicalize_implicit_dag_direct_rq(network, replay_inputs=raw, block_size=4096, copy_network=False)
        record["motif_algorithm1_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        diagonal = diagonalize_implicit_dag_full_rank(canonical.network, replay_inputs=raw, copy_network=False, retain_eigenvalues=False, retain_post_evd_environments=False, stream_pre_evd_environments=True)
        record["motif_algorithms2_and3_seconds"] = time.perf_counter() - started
        record["motif_replay_error"] = diagonal.replay_relative_error
        if diagonal.replay_relative_error > 2e-10:
            raise RuntimeError("contracted-environment motif replay failed")
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    runtime = assert_direct_only_runtime_guard(direct_only_runtime_report())
    if audit_direct_only_launch(ROOT, (Path(__file__),)) != AUDIT:
        raise RuntimeError("benchmark sources changed")
    result = {"schema": "xvla_canonical_thread_benchmark_v1", "records": records, "static_audit": AUDIT, "runtime_guard": runtime, "scope": "microbenchmark only, full-sweep timing not inferred automatically"}
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
