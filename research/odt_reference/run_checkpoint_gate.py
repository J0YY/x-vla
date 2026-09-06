"""Local unchanged-width trained-block export/replay and clone admission gate.

This is NOT a trained-block ODT success claim. Full ODT needs the independent
clone gate to admit the representation first. No cluster or model imports.
"""
import hashlib
import json
from pathlib import Path
from dataclasses import asdict

import numpy as np

from research.odt_reference.run_tests import audit, install_guards, COUNTS


if __name__ == "__main__":
    sources = audit(("run_checkpoint_gate.py", "checkpoint.py", "block.py", "block_oracle.py",
                     "weights.py", "dooms.py", "shared_dag.py", "clone_passes.py", "clone_oracle.py"))
    install_guards()
    from research.odt_reference.checkpoint import load_checkpoint
    from research.odt_reference.block import BlockSpec, compile_block, evaluate_block
    from research.odt_reference.block_oracle import block_forward
    from research.odt_reference.clone_passes import clone_node_count

    path = Path("tmp/odt_modal_curve_v1/inputs/capable_linear_b1c0_checkpoint.pt")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != "b1c0dfce86ee90b45e30367603a7cc4d88c02f9056e836b19656179bf74ea3ee":
        raise ValueError("checkpoint identity mismatch")
    # Source config: scripts/run_capable_linear_fresh_capability.py::_config.
    # Vision is noncausal, width192, heads8, rank576, both residuals, Padé norms.
    spec = BlockSpec(8)
    weights = load_checkpoint(path, "vision.blocks.blocks.0.")
    if weights["ffn.left.weight"].shape != (576, 192):
        raise ValueError("unchanged trained vision-block dimensions disagree")
    raw = np.random.default_rng(20260906).normal(size=(2, 2, 192)) * 0.2
    mask = np.ones((2, 2))
    expected = block_forward(raw, weights, n_heads=spec.n_heads, mask=mask, eps=spec.eps)
    print("Compiling unchanged width192/head8/rank576 joint two-token block", flush=True)
    graph = compile_block(weights, spec=spec, mask=mask, maximum_elements=16_000_000)
    actual = evaluate_block(graph, raw).reshape(expected.shape)
    error = float(np.max(np.abs(actual - expected)))
    scale = max(float(np.max(np.abs(expected))), 1e-12)
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-10)
    admitted, admission = True, None
    try:
        count = clone_node_count(graph)
    except ValueError as exc:
        admitted, admission, count = False, str(exc), None
    byte_sizes = []
    for node in graph.nodes:
        byte_sizes.append(node.core.nbytes + sum(byte_sizes[child] for child in node.children))
    print(json.dumps({"checkpoint_sha256": digest, "source_prefix": "vision.blocks.blocks.0.",
        "source_config": asdict(spec), "input_shape": list(raw.shape), "joint_output": True,
        "synthetic_inputs": True, "trained_weights_unchanged": True, "norm_sites": 6,
        "nodes": len(graph.nodes), "maximum_core_elements": max(n.core.size for n in graph.nodes),
        "clone_tensor_bytes": byte_sizes[-1] + graph.head.nbytes, "clone_admitted": admitted,
        "clone_count": count, "clone_admission_message": admission,
        "export_replay_passed": True, "maximum_absolute_error": error, "maximum_relative_error": error / scale,
        "odt_completed": False, "kernels": COUNTS, "source_sha256": sources}, sort_keys=True))
