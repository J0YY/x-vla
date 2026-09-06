#!/usr/bin/env python3
"""Guarded correctness and bounded timing harness for the compiled DAG prototype."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

from scripts.odt_direct_only_compliance import (
    assert_direct_only_runtime_guard,
    audit_direct_only_launch,
    direct_only_runtime_report,
    install_direct_only_runtime_guard,
)


ENTRYPOINTS = (
    Path(__file__),
    ROOT / "tests/test_implicit_projective_dag_mapped.py",
)
STATIC_AUDIT = audit_direct_only_launch(ROOT, ENTRYPOINTS)
for field in (
    "prohibited_calls_found",
    "prohibited_self_overlap_sites",
    "guarded_dormant_spectral_norm_sites",
    "duplicate_top_level_definition_sites",
):
    if STATIC_AUDIT[field]:
        raise RuntimeError(f"compiled executor static audit rejected {field}")
install_direct_only_runtime_guard()

import pytest
import torch

from xvla.train.implicit_projective_dag_mapped import (
    open_mapped_implicit_projective_dag_artifact,
)
from xvla.train.implicit_projective_dag_numba import (
    open_compiled_mapped_implicit_projective_dag_artifact,
)
from xvla.train.implicit_projective_dag_artifact import (
    export_implicit_projective_dag_artifact,
)
from xvla.train.implicit_sparse_projective_odt import (
    ImplicitNode,
    ImplicitProjectiveDAG,
    PhysicalSourceSpec,
    ReducedQRBinaryCore,
    UnaryCore,
)


DTYPE = torch.float64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dispatch_network(requested_nodes: int) -> tuple[ImplicitProjectiveDAG, int]:
    leaves = max(2, (requested_nodes + 2) // 3)
    level = [
        ImplicitNode(
            uid=index,
            label=f"benchmark.leaf{index}",
            core=UnaryCore(torch.eye(2, dtype=DTYPE), "physical"),
            physical_token=0,
        )
        for index in range(leaves)
    ]
    index = leaves
    while len(level) > 1:
        next_level = []
        for offset in range(0, len(level), 2):
            if offset + 1 == len(level):
                next_level.append(level[offset])
                continue
            binary = ImplicitNode(
                uid=index,
                label=f"benchmark.binary{index}",
                core=ReducedQRBinaryCore(
                    torch.tensor(
                        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                        dtype=DTYPE,
                    ),
                    left_dimension=2,
                    right_dimension=2,
                    kind="benchmark_identity_rows",
                ),
                children=(level[offset], level[offset + 1]),
            )
            index += 1
            unary = ImplicitNode(
                uid=index,
                label=f"benchmark.unary{index}",
                core=UnaryCore(torch.eye(2, dtype=DTYPE), "benchmark_identity"),
                children=(binary,),
            )
            index += 1
            next_level.append(unary)
        level = next_level
    return (
        ImplicitProjectiveDAG(
            root=level[0],
            head=torch.eye(2, dtype=DTYPE),
            head_binary_exponent=0,
            token_count=1,
            feature_dimension=1,
            selected_token=0,
            mask=torch.ones(1, 1, dtype=DTYPE),
            claim_boundary="Synthetic dispatch timing only.",
            algorithm1_direct_rq_complete=True,
            algorithm1_scale_ledger_complete=True,
            algorithm1_certified_head_binary_exponent=0,
        ),
        index,
    )


def _wide_network() -> ImplicitProjectiveDAG:
    generator = torch.Generator().manual_seed(385386)
    left = ImplicitNode(
        uid=1,
        label="wide.left",
        core=UnaryCore(torch.eye(2, dtype=DTYPE), "physical"),
        physical_source_key="left",
    )
    right = ImplicitNode(
        uid=2,
        label="wide.right",
        core=UnaryCore(torch.eye(193, dtype=DTYPE), "physical"),
        physical_source_key="right",
    )
    root = ImplicitNode(
        uid=3,
        label="wide.retained_q",
        core=ReducedQRBinaryCore(
            torch.randn(385, 386, generator=generator, dtype=DTYPE) / 64.0,
            left_dimension=2,
            right_dimension=193,
            kind="bounded_direct_q",
        ),
        children=(left, right),
    )
    return ImplicitProjectiveDAG(
        root=root,
        head=torch.randn(57, 385, generator=generator, dtype=DTYPE) / 32.0,
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=1,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=DTYPE),
        physical_sources=(
            PhysicalSourceSpec("left", "wide.left", 1),
            PhysicalSourceSpec("right", "wide.right", 192),
        ),
        claim_boundary="One production-shape retained-Q timing fixture.",
        algorithm1_direct_rq_complete=True,
        algorithm1_scale_ledger_complete=True,
        algorithm1_certified_head_binary_exponent=0,
    )


def _relative_projective(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_scale = actual.abs().amax(dim=1, keepdim=True)
    expected_scale = expected.abs().amax(dim=1, keepdim=True)
    actual = actual / actual_scale
    expected = expected / expected_scale
    pivot = expected.abs().argmax(dim=1, keepdim=True)
    sign = torch.sign(actual.gather(1, pivot) * expected.gather(1, pivot))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return float((actual * sign - expected).abs().amax().item())


def _timed(function: object, raw: object) -> tuple[torch.Tensor, float]:
    started = time.perf_counter()
    result = function(raw)
    return result, time.perf_counter() - started


def _benchmark_one(
    root: Path,
    network: ImplicitProjectiveDAG,
    inputs: dict[int, object],
) -> dict[str, object]:
    receipt = export_implicit_projective_dag_artifact(network, root)
    ordinary_results: dict[int, torch.Tensor] = {}
    ordinary_seconds: dict[int, float] = {}
    with open_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as ordinary:
        for batch, raw in inputs.items():
            result, elapsed = _timed(ordinary.evaluate_projective_boundary, raw)
            ordinary_results[batch] = result
            ordinary_seconds[batch] = elapsed

    bind_started = time.perf_counter()
    with open_compiled_mapped_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=receipt.manifest_sha256,
    ) as compiled:
        bind_seconds = time.perf_counter() - bind_started
        first_batch = min(inputs)
        _warm, compilation_seconds = _timed(
            compiled.evaluate_projective_boundary,
            inputs[first_batch],
        )
        compiled_seconds: dict[int, float] = {}
        errors: dict[int, float] = {}
        for batch, raw in inputs.items():
            result, elapsed = _timed(compiled.evaluate_projective_boundary, raw)
            compiled_seconds[batch] = elapsed
            errors[batch] = _relative_projective(result, ordinary_results[batch])
        arena = {
            **compiled.arena_receipt.__dict__,
            "batch20_bytes": compiled.arena_bytes_for_batch(20),
        }
    if max(errors.values()) > 2e-10:
        raise RuntimeError(f"compiled executor differs from mapped oracle: {errors}")
    return {
        "manifest_sha256": receipt.manifest_sha256,
        "raw_byte_count": receipt.raw_byte_count,
        "bind_seconds": bind_seconds,
        "compilation_seconds": compilation_seconds,
        "ordinary_seconds": ordinary_seconds,
        "compiled_seconds": compiled_seconds,
        "compiled_speedup": {
            batch: ordinary_seconds[batch] / compiled_seconds[batch]
            for batch in inputs
        },
        "maximum_projective_error": max(errors.values()),
        "arena": arena,
    }


def main() -> None:
    torch.set_num_threads(16)
    test_arguments = [
        str(ROOT / "tests/test_implicit_projective_dag_mapped.py"),
        "-q",
        "-p",
        "no:cacheprovider",
        "-k",
        (
            "compiled_executor or "
            "all_six_physical_prefixes_match_their_original_mask_outputs_mapped"
        ),
    ]
    test_started = time.perf_counter()
    test_code = pytest.main(test_arguments)
    test_seconds = time.perf_counter() - test_started
    if test_code:
        raise RuntimeError(f"compiled executor tests failed with exit {test_code}")

    temporary = Path(tempfile.mkdtemp(prefix="xvla_compiled_mapped_"))
    dispatch, dispatch_nodes = _dispatch_network(10_000)
    dispatch_inputs = {
        batch: torch.ones(batch, 1, 1, dtype=DTYPE)
        for batch in (1, 20)
    }
    wide = _wide_network()
    generator = torch.Generator().manual_seed(441)
    wide_inputs = {
        batch: {
            "left": torch.randn(batch, 1, generator=generator, dtype=DTYPE),
            "right": torch.randn(batch, 192, generator=generator, dtype=DTYPE),
        }
        for batch in (1, 20)
    }
    result = {
        "schema": "xvla_compiled_mapped_executor_bounded_benchmark_v1",
        "tests": {"exit_code": test_code, "elapsed_seconds": test_seconds},
        "dispatch_10k": {
            "node_count": dispatch_nodes,
            **_benchmark_one(temporary / "dispatch", dispatch, dispatch_inputs),
        },
        "retained_q_385x386": _benchmark_one(
            temporary / "wide",
            wide,
            wide_inputs,
        ),
        "static_audit": STATIC_AUDIT,
        "runtime_guard": assert_direct_only_runtime_guard(
            direct_only_runtime_report()
        ),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "source_sha256": {
            "runner": _sha256(Path(__file__)),
            "compiled_executor": _sha256(
                ROOT / "xvla/train/implicit_projective_dag_numba.py"
            ),
            "mapped_oracle": _sha256(
                ROOT / "xvla/train/implicit_projective_dag_mapped.py"
            ),
            "tests": _sha256(ROOT / "tests/test_implicit_projective_dag_mapped.py"),
        },
    }
    if audit_direct_only_launch(ROOT, ENTRYPOINTS) != STATIC_AUDIT:
        raise RuntimeError("compiled executor sources changed during benchmark")
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
