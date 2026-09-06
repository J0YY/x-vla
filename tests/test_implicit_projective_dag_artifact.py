from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from xvla.train.implicit_projective_dag_artifact import (
    ARTIFACT_SCHEMA,
    MANIFEST_NAME,
    MAXIMUM_SHARD_BYTES,
    export_implicit_projective_dag_artifact,
    load_implicit_projective_dag_artifact,
)
from xvla.train.implicit_sparse_projective_odt import (
    CPBinaryCore,
    DirectQProvenance,
    ImplicitNode,
    ImplicitProjectiveDAG,
    PhysicalSourceSpec,
    ReducedQRBinaryCore,
    UnaryCore,
    evaluate_projective_boundary,
)


DTYPE = torch.float64


def _network() -> ImplicitProjectiveDAG:
    leaf_a = ImplicitNode(
        uid=10,
        label="leaf.a",
        core=UnaryCore(
            torch.tensor(
                [[1.0, -0.25, 0.5], [0.125, 0.75, -0.5]], dtype=DTYPE
            ),
            "physical_a",
        ),
        physical_source_key="a",
    )
    leaf_b = ImplicitNode(
        uid=11,
        label="leaf.b",
        core=UnaryCore(
            torch.tensor(
                [[0.5, 1.0], [-0.75, 0.25], [0.125, -0.5]], dtype=DTYPE
            ),
            "physical_b",
        ),
        physical_source_key="b",
    )
    provenance = DirectQProvenance(
        method="test_direct_q_replay",
        verified=True,
        stage_count=1,
        column_blocks_compared=1,
        columns_compared=6,
        compact_q_relative_error=0.0,
        direct_q_reconstruction_relative_error=0.0,
        retained_transition_elements=0,
        minimum_pivot_to_maximum_entry=1.0,
        conditioning_threshold=0.0,
        conditioning_accepted=True,
    )
    binary = ImplicitNode(
        uid=20,
        label="binary.cp",
        core=CPBinaryCore(
            output_factor=torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.5, -0.25], [0.125, 0.75]],
                dtype=DTYPE,
            ),
            left_factor=torch.tensor(
                [[1.0, 0.25], [-0.5, 1.0]], dtype=DTYPE
            ),
            right_factor=torch.tensor(
                [[0.75, -0.25, 0.5], [0.125, 1.0, -0.5]], dtype=DTYPE
            ),
            kind="canonical_cp",
            direct_q_provenance=provenance,
        ),
        children=(leaf_a, leaf_b),
    )
    shared = ImplicitNode(
        uid=21,
        label="shared.unary",
        core=UnaryCore(torch.eye(4, dtype=DTYPE), "shared_identity"),
        children=(binary,),
        origin_uid=120,
    )
    root = ImplicitNode(
        uid=30,
        label="root.reduced_q",
        core=ReducedQRBinaryCore(
            q_rows=torch.tensor(
                [
                    [
                        0.25,
                        -0.5,
                        0.75,
                        0.125,
                        -0.25,
                        0.375,
                        0.5,
                        -0.125,
                        0.625,
                        0.25,
                        -0.375,
                        0.5,
                        -0.75,
                        0.125,
                        0.25,
                        1.0,
                    ],
                    [
                        -0.125,
                        0.25,
                        0.5,
                        -0.75,
                        0.375,
                        0.625,
                        -0.25,
                        0.125,
                        0.5,
                        -0.375,
                        0.25,
                        0.75,
                        0.125,
                        -0.5,
                        1.0,
                        0.25,
                    ],
                ],
                dtype=DTYPE,
            ),
            left_dimension=4,
            right_dimension=4,
            kind="bounded_direct_q",
        ),
        children=(shared, shared),
    )
    return ImplicitProjectiveDAG(
        root=root,
        head=torch.tensor(
            [[1.0, 0.25], [-0.5, 1.0], [0.75, 0.5]], dtype=DTYPE
        ),
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=1,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=DTYPE),
        claim_boundary="Canonical persistence unit fixture.",
        algorithm1_direct_rq_complete=True,
        physical_sources=(
            PhysicalSourceSpec("a", "source.a", 2),
            PhysicalSourceSpec("b", "source.b", 1),
        ),
        algorithm1_scale_ledger_complete=True,
        algorithm1_certified_head_binary_exponent=0,
    )


def _raw_inputs() -> dict[str, torch.Tensor]:
    return {
        "a": torch.tensor([[0.25, -0.5], [-0.125, 0.75]], dtype=DTYPE),
        "b": torch.tensor([[0.375], [-0.625]], dtype=DTYPE),
    }


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _read_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / MANIFEST_NAME).read_text())


def _manifest_sha256(root: Path) -> str:
    return hashlib.sha256((root / MANIFEST_NAME).read_bytes()).hexdigest()


def _load_authorized(root: Path) -> ImplicitProjectiveDAG:
    return load_implicit_projective_dag_artifact(
        root,
        expected_manifest_sha256=_manifest_sha256(root),
    )


def _make_mutable_copy(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    destination.chmod(0o755)
    for path in destination.iterdir():
        path.chmod(0o644)
    return destination


def _freeze(root: Path) -> None:
    for path in root.iterdir():
        path.chmod(0o444)
    root.chmod(0o555)


def _rewrite_manifest(root: Path, value: dict[str, Any]) -> None:
    (root / MANIFEST_NAME).write_bytes(_canonical_bytes(value))


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(root.iterdir())}


def _swap_equal_sized_tensor_records_and_payloads(
    root: Path,
    manifest: dict[str, Any],
    first_index: int,
    second_index: int,
) -> None:
    tensors = manifest["tensors"]
    assert second_index == first_index + 1
    assert tensors[first_index]["byte_count"] == tensors[second_index]["byte_count"]
    shard_bytes = b"".join(
        (root / shard["name"]).read_bytes() for shard in manifest["shards"]
    )
    first_start = sum(
        record["byte_count"] for record in tensors[:first_index]
    )
    second_start = first_start + tensors[first_index]["byte_count"]
    count = tensors[first_index]["byte_count"]
    first_raw = shard_bytes[first_start : first_start + count]
    second_raw = shard_bytes[second_start : second_start + count]
    mutated_bytes = bytearray(shard_bytes)
    mutated_bytes[first_start : first_start + count] = second_raw
    mutated_bytes[second_start : second_start + count] = first_raw

    cursor = 0
    for shard in manifest["shards"]:
        stop = cursor + shard["byte_count"]
        raw = bytes(mutated_bytes[cursor:stop])
        (root / shard["name"]).write_bytes(raw)
        shard["sha256"] = hashlib.sha256(raw).hexdigest()
        cursor = stop

    first_record = tensors[first_index]
    second_record = tensors[second_index]
    first_segments = first_record["segments"]
    second_segments = second_record["segments"]
    tensors[first_index], tensors[second_index] = second_record, first_record
    tensors[first_index]["index"] = first_index
    tensors[second_index]["index"] = second_index
    tensors[first_index]["segments"] = first_segments
    tensors[second_index]["segments"] = second_segments


def test_round_trip_is_byte_stable_and_preserves_shared_execution(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    network = _network()
    expected = evaluate_projective_boundary(network, _raw_inputs())
    receipt = export_implicit_projective_dag_artifact(
        network, first, shard_size_bytes=64
    )
    loaded = load_implicit_projective_dag_artifact(
        first,
        expected_manifest_sha256=receipt.manifest_sha256,
    )
    observed = evaluate_projective_boundary(loaded, _raw_inputs())
    assert torch.equal(observed, expected)
    assert loaded.root.children[0] is loaded.root.children[1]
    second_receipt = export_implicit_projective_dag_artifact(
        loaded, second, shard_size_bytes=64
    )
    assert _artifact_bytes(first) == _artifact_bytes(second)
    assert receipt.manifest_sha256 == second_receipt.manifest_sha256
    assert receipt.shard_sha256 == second_receipt.shard_sha256

    manifest_bytes = (first / MANIFEST_NAME).read_bytes()
    manifest = _read_manifest(first)
    assert manifest_bytes == _canonical_bytes(manifest)
    assert manifest["schema"] == ARTIFACT_SCHEMA
    assert manifest["graph"]["order"] == "depth_first_child_order_postorder"
    assert manifest["graph"]["root_index"] == manifest["graph"]["node_count"] - 1
    assert manifest["graph"]["incoming_refcount"][-2] == 2
    assert len(manifest["shards"]) > 1
    assert all(item["byte_count"] <= 64 for item in manifest["shards"])
    assert receipt.raw_byte_count == sum(
        item["byte_count"] for item in manifest["shards"]
    )


def test_export_is_exclusive_and_failed_validation_publishes_nothing(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact"
    receipt = export_implicit_projective_dag_artifact(_network(), destination)
    before = _artifact_bytes(destination)
    with pytest.raises(FileExistsError):
        export_implicit_projective_dag_artifact(_network(), destination)
    assert _artifact_bytes(destination) == before
    assert receipt.manifest_sha256 == hashlib.sha256(
        (destination / MANIFEST_NAME).read_bytes()
    ).hexdigest()

    invalid = _network()
    invalid.head[0, 0] = torch.inf
    failed_destination = tmp_path / "failed"
    with pytest.raises(ValueError, match="finite"):
        export_implicit_projective_dag_artifact(invalid, failed_destination)
    assert not os.path.lexists(failed_destination)

    with pytest.raises(ValueError, match="aligned"):
        export_implicit_projective_dag_artifact(
            _network(), tmp_path / "oversized", shard_size_bytes=MAXIMUM_SHARD_BYTES + 8
        )
    assert not os.path.lexists(tmp_path / "oversized")

    for index, invalid_size in enumerate((8, 120, True)):
        invalid_destination = tmp_path / f"misaligned-{index}"
        with pytest.raises(ValueError, match="64-byte aligned"):
            export_implicit_projective_dag_artifact(
                _network(), invalid_destination, shard_size_bytes=invalid_size
            )
        assert not os.path.lexists(invalid_destination)


@pytest.mark.parametrize("damage", ("byte", "truncate"))
def test_shard_damage_is_rejected(tmp_path: Path, damage: str) -> None:
    source = tmp_path / "source"
    export_implicit_projective_dag_artifact(_network(), source, shard_size_bytes=64)
    damaged = _make_mutable_copy(source, tmp_path / damage)
    shard = sorted(damaged.glob("*.f64le"))[0]
    raw = bytearray(shard.read_bytes())
    if damage == "byte":
        raw[0] ^= 1
    else:
        del raw[-8:]
    shard.write_bytes(raw)
    _freeze(damaged)
    with pytest.raises(ValueError, match="shard"):
        _load_authorized(damaged)


def test_extra_file_and_symbolic_member_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    export_implicit_projective_dag_artifact(_network(), source, shard_size_bytes=64)

    extra = _make_mutable_copy(source, tmp_path / "extra")
    (extra / "unexpected.bin").write_bytes(b"unexpected")
    _freeze(extra)
    with pytest.raises(ValueError, match="missing or extra"):
        _load_authorized(extra)

    linked = _make_mutable_copy(source, tmp_path / "linked")
    first_shard = sorted(linked.glob("*.f64le"))[0]
    first_shard.unlink()
    first_shard.symlink_to(source / first_shard.name)
    _freeze(linked)
    with pytest.raises(ValueError, match="physical"):
        _load_authorized(linked)


def _wrong_dtype(value: dict[str, Any]) -> None:
    value["encoding"]["dtype"] = "float32"


def _wrong_shape(value: dict[str, Any]) -> None:
    value["tensors"][0]["shape"][0] += 1


def _wrong_exponent(value: dict[str, Any]) -> None:
    value["graph"]["core_binary_exponent"][0] = 1


def _wrong_reachability(value: dict[str, Any]) -> None:
    graph = value["graph"]
    graph["child_indices"][1] = 0
    graph["incoming_refcount"] = [2, 0, 1, 2, 0]


def _wrong_refcount(value: dict[str, Any]) -> None:
    value["graph"]["incoming_refcount"][0] += 1


def _wrong_registry(value: dict[str, Any]) -> None:
    del value["physical_sources"][1]


@pytest.mark.parametrize(
    "mutation,match",
    (
        (_wrong_dtype, "encoding"),
        (_wrong_shape, "byte count"),
        (_wrong_exponent, "canonical core exponent"),
        (_wrong_reachability, "unreachable"),
        (_wrong_refcount, "refcounts"),
        (_wrong_registry, "source"),
    ),
)
def test_manifest_semantic_mutations_are_rejected(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    source = tmp_path / "source"
    export_implicit_projective_dag_artifact(_network(), source, shard_size_bytes=64)
    damaged = _make_mutable_copy(source, tmp_path / mutation.__name__)
    manifest = _read_manifest(damaged)
    mutation(manifest)
    _rewrite_manifest(damaged, manifest)
    _freeze(damaged)
    with pytest.raises(ValueError, match=match):
        _load_authorized(damaged)


def test_noncanonical_and_ambiguous_manifests_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    export_implicit_projective_dag_artifact(_network(), source, shard_size_bytes=64)

    pretty = _make_mutable_copy(source, tmp_path / "pretty")
    manifest = _read_manifest(pretty)
    (pretty / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    _freeze(pretty)
    with pytest.raises(ValueError, match="canonical"):
        _load_authorized(pretty)

    duplicate = _make_mutable_copy(source, tmp_path / "duplicate")
    encoded = (duplicate / MANIFEST_NAME).read_text()
    assert encoded.startswith('{"encoding":')
    (duplicate / MANIFEST_NAME).write_text(
        '{"encoding":{},' + encoded[1:]
    )
    _freeze(duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        _load_authorized(duplicate)


def test_authenticated_nonfinite_tensor_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    export_implicit_projective_dag_artifact(_network(), source, shard_size_bytes=64)
    damaged = _make_mutable_copy(source, tmp_path / "nonfinite")
    manifest = _read_manifest(damaged)
    shard = damaged / manifest["shards"][0]["name"]
    raw = bytearray(shard.read_bytes())
    raw[:8] = struct.pack("<d", float("inf"))
    shard.write_bytes(raw)
    manifest["shards"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    _rewrite_manifest(damaged, manifest)
    _freeze(damaged)
    with pytest.raises(ValueError, match="nonfinite"):
        _load_authorized(damaged)


def test_incomplete_publication_and_writable_members_are_rejected(
    tmp_path: Path,
) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir(mode=0o555)
    with pytest.raises(FileNotFoundError):
        load_implicit_projective_dag_artifact(
            incomplete,
            expected_manifest_sha256="0" * 64,
        )

    source = tmp_path / "source"
    receipt = export_implicit_projective_dag_artifact(_network(), source)
    source.chmod(0o755)
    with pytest.raises(ValueError, match="immutable"):
        load_implicit_projective_dag_artifact(
            source,
            expected_manifest_sha256=receipt.manifest_sha256,
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("verified", 1),
        ("conditioning_accepted", 1),
        ("compact_q_relative_error", 0),
        ("direct_q_reconstruction_relative_error", 0),
        ("minimum_pivot_to_maximum_entry", 1),
        ("conditioning_threshold", 0),
    ),
)
def test_export_rejects_nonexact_provenance_primitives(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    network = _network()
    cp_core = network.root.children[0].children[0].core
    assert isinstance(cp_core, CPBinaryCore)
    assert cp_core.direct_q_provenance is not None
    cp_core.direct_q_provenance = replace(
        cp_core.direct_q_provenance,
        **{field: value},
    )
    destination = tmp_path / field
    with pytest.raises(ValueError, match="provenance"):
        export_implicit_projective_dag_artifact(network, destination)
    assert not os.path.lexists(destination)


def test_export_rejects_noninteger_dimensions_and_source_widths(
    tmp_path: Path,
) -> None:
    dimension_network = _network()
    reduced_core = dimension_network.root.core
    assert isinstance(reduced_core, ReducedQRBinaryCore)
    reduced_core.left_dimension = 4.0
    with pytest.raises(ValueError, match="left dimension"):
        export_implicit_projective_dag_artifact(
            dimension_network,
            tmp_path / "dimension",
        )
    assert not os.path.lexists(tmp_path / "dimension")

    source_network = _network()
    object.__setattr__(source_network.physical_sources[0], "width", 2.0)
    with pytest.raises(ValueError, match="physical source width"):
        export_implicit_projective_dag_artifact(
            source_network,
            tmp_path / "source-width",
        )
    assert not os.path.lexists(tmp_path / "source-width")


def test_noncanonical_topological_permutation_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    export_implicit_projective_dag_artifact(_network(), source, shard_size_bytes=64)
    permuted = _make_mutable_copy(source, tmp_path / "permuted")
    manifest = _read_manifest(permuted)
    graph = manifest["graph"]
    node_count = graph["node_count"]
    assert node_count == 5
    old_for_new = [1, 0, 2, 3, 4]
    old_to_new = {old: new for new, old in enumerate(old_for_new)}
    node_columns = (
        "uid",
        "label",
        "origin_uid",
        "physical_token",
        "physical_source_key",
        "core_type",
        "core_kind",
        "core_binary_exponent",
        "direct_q_provenance",
        "matrix_tensor",
        "output_factor_tensor",
        "left_factor_tensor",
        "right_factor_tensor",
        "q_rows_tensor",
        "left_dimension",
        "right_dimension",
        "incoming_refcount",
    )
    for name in node_columns:
        original = graph[name]
        graph[name] = [original[index] for index in old_for_new]
    graph["child_indices"] = [
        old_to_new[index] for index in graph["child_indices"]
    ]
    _swap_equal_sized_tensor_records_and_payloads(permuted, manifest, 2, 3)
    graph["matrix_tensor"][0] = 2
    graph["matrix_tensor"][1] = 3
    for index in (0, 1):
        tensor_index = graph["matrix_tensor"][index]
        manifest["tensors"][tensor_index]["name"] = f"node[{index}].matrix"
    _rewrite_manifest(permuted, manifest)
    _freeze(permuted)
    with pytest.raises(ValueError, match="depth-first postorder"):
        _load_authorized(permuted)


def test_noncanonical_tensor_table_permutation_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    export_implicit_projective_dag_artifact(_network(), source, shard_size_bytes=64)
    permuted = _make_mutable_copy(source, tmp_path / "tensor-permuted")
    manifest = _read_manifest(permuted)
    tensors = manifest["tensors"]
    assert tensors[2]["byte_count"] == tensors[3]["byte_count"] == 48
    _swap_equal_sized_tensor_records_and_payloads(permuted, manifest, 2, 3)
    graph = manifest["graph"]
    graph["matrix_tensor"][0] = 3
    graph["matrix_tensor"][1] = 2
    _rewrite_manifest(permuted, manifest)
    _freeze(permuted)
    with pytest.raises(ValueError, match="canonical order"):
        _load_authorized(permuted)


def test_manifest_digest_is_a_required_external_root_of_trust(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    receipt = export_implicit_projective_dag_artifact(
        _network(), source, shard_size_bytes=64
    )
    with pytest.raises(TypeError):
        load_implicit_projective_dag_artifact(source)

    tampered = _make_mutable_copy(source, tmp_path / "finite-tamper")
    manifest = _read_manifest(tampered)
    shard = tampered / manifest["shards"][0]["name"]
    raw = bytearray(shard.read_bytes())
    raw[:8] = struct.pack("<d", 1.125)
    shard.write_bytes(raw)
    manifest["shards"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    _rewrite_manifest(tampered, manifest)
    _freeze(tampered)
    with pytest.raises(ValueError, match="manifest digest"):
        load_implicit_projective_dag_artifact(
            tampered,
            expected_manifest_sha256=receipt.manifest_sha256,
        )

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_implicit_projective_dag_artifact(
            source,
            expected_manifest_sha256=True,
        )


def test_symlink_parent_and_hardlinked_member_are_rejected(tmp_path: Path) -> None:
    physical_parent = tmp_path / "physical-parent"
    physical_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(physical_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="physical directory"):
        export_implicit_projective_dag_artifact(
            _network(),
            linked_parent / "artifact",
        )
    assert not (physical_parent / "artifact").exists()

    source = tmp_path / "source"
    receipt = export_implicit_projective_dag_artifact(
        _network(), source, shard_size_bytes=64
    )
    shard = sorted(source.glob("*.f64le"))[0]
    os.link(shard, tmp_path / "outside-hardlink.f64le")
    with pytest.raises(ValueError, match="hard-linked"):
        load_implicit_projective_dag_artifact(
            source,
            expected_manifest_sha256=receipt.manifest_sha256,
        )


def test_post_read_member_mutation_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xvla.train import implicit_projective_dag_artifact as artifact_module

    source = tmp_path / "source"
    receipt = export_implicit_projective_dag_artifact(
        _network(), source, shard_size_bytes=64
    )
    shard = sorted(source.glob("*.f64le"))[0]
    decode = artifact_module._decode_tensors

    def decode_then_mutate(
        tensor_records: Any,
    ) -> tuple[tuple[str, torch.Tensor], ...]:
        result = decode(tensor_records)
        shard.chmod(0o644)
        raw = bytearray(shard.read_bytes())
        raw[0] ^= 1
        shard.write_bytes(raw)
        shard.chmod(0o444)
        return result

    monkeypatch.setattr(artifact_module, "_decode_tensors", decode_then_mutate)
    with pytest.raises(ValueError, match="changed during loading"):
        load_implicit_projective_dag_artifact(
            source,
            expected_manifest_sha256=receipt.manifest_sha256,
        )
