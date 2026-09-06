"""Additive persistence contract for physically truncated ODT descendants.

The binary packing and authentication are identical to the archival DAG codec.
The distinct schema explicitly records that the stored cores are prefixes, not
new canonical Q cores.  The ladder receipt binds each manifest to its original
full-rank certificate and every-occurrence prefix proof.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from xvla.train import implicit_projective_dag_artifact as canonical_codec
from xvla.train.odt_engine_v2.graph import (
    _validate_network,
    _walk_unique,
)
from xvla.train.odt_engine_v2.types import (
    CPBinaryCore,
    ImplicitProjectiveDAG,
)


ARTIFACT_SCHEMA = "xvla_implicit_projective_dag_prefix_artifact_v1"


def _require_prefix_normal_form(network: ImplicitProjectiveDAG) -> dict[str, Any]:
    _validate_network(network)
    if (
        network.algorithm1_direct_rq_complete is not False
        or network.algorithm1_scale_ledger_complete is not False
        or network.algorithm1_certified_head_binary_exponent is not None
    ):
        raise ValueError("a prefix artifact must explicitly clear its full-rank canonical flags")
    nodes = _walk_unique(network.root)
    for node in nodes:
        if node.core.binary_exponent != 0:
            raise ValueError("a canonical descendant must retain zero local binary exponents")
        if isinstance(node.core, CPBinaryCore) and node.core.direct_q_provenance is not None:
            raise ValueError("a sliced CP core must not claim unsliced direct-Q provenance")
    return {"core_count": len(nodes), "all_core_exponents_zero": True, "full_rank_canonical_claimed": False}


def _codec() -> Any:
    # A private module instance prevents weakening or mutating the canonical
    # codec's validators for any existing consumer.  Only its explicit schema
    # identity and normal-form authority differ for descendant artifacts.
    name = __name__ + "._descendant_codec"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    source = Path(canonical_codec.__file__).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the authenticated DAG binary codec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        module.ARTIFACT_SCHEMA = ARTIFACT_SCHEMA
        module.validate_canonical_exponent_normal_form = _require_prefix_normal_form
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def export_implicit_projective_prefix_artifact(network: ImplicitProjectiveDAG, destination: str | Path, *, shard_size_bytes: int = canonical_codec.MAXIMUM_SHARD_BYTES) -> Any:
    return _codec().export_implicit_projective_dag_artifact(network, destination, shard_size_bytes=shard_size_bytes)


def load_implicit_projective_prefix_artifact(source: str | Path, *, expected_manifest_sha256: str) -> ImplicitProjectiveDAG:
    return _codec().load_implicit_projective_dag_artifact(source, expected_manifest_sha256=expected_manifest_sha256)
