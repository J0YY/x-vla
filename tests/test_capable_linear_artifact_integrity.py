from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from scripts.capable_linear_artifact_integrity import (
    assert_physical_hashes_unchanged,
    read_authenticated_json,
    require_rational_norm_buffer_inventory,
)


def test_attestation_input_mutation_is_rejected(tmp_path: Path) -> None:
    full = tmp_path / "full.json"
    compression = tmp_path / "compression.json"
    full.write_bytes(b"full-v1")
    compression.write_bytes(b"compression-v1")
    paths = {"full_rank": full, "compression": compression}
    snapshot = assert_physical_hashes_unchanged(paths)
    compression.write_bytes(b"compression-v2")
    with pytest.raises(RuntimeError, match="changed across publication"):
        assert_physical_hashes_unchanged(paths, snapshot)


def test_json_fields_and_digest_are_bound_to_the_same_read_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    first = b'{"generation":1,"ok":true}\n'
    artifact.write_bytes(first)
    value, digest = read_authenticated_json(artifact, "fixture")
    assert value == {"generation": 1, "ok": True}
    assert digest == hashlib.sha256(first).hexdigest()
    snapshot = {"artifact": digest}
    artifact.write_bytes(b'{"generation":2,"ok":true}\n')
    with pytest.raises(RuntimeError, match="changed across publication"):
        assert_physical_hashes_unchanged({"artifact": artifact}, snapshot)


@pytest.mark.parametrize(
    "encoded",
    (
        b'{"value":1,"value":2}\n',
        b'{"value":NaN}\n',
    ),
)
def test_authenticated_json_rejects_ambiguous_values(
    tmp_path: Path, encoded: bytes
) -> None:
    artifact = tmp_path / "ambiguous.json"
    artifact.write_bytes(encoded)
    with pytest.raises(RuntimeError):
        read_authenticated_json(artifact, "fixture")


def test_authenticated_rehash_rejects_symbolic_link(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b'{"ok":true}\n')
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(RuntimeError, match="nonphysical"):
        assert_physical_hashes_unchanged({"artifact": link})


def test_current_unit_dependency_is_full_only_not_concurrent_preflight() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_capable_linear_direct_odt_spectrum.py"
    )
    module = ast.parse(source.read_text())
    functions = {
        node.name: node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    preflight = functions["_preflight_mode"]
    full = functions["_full_mode"]
    current_unit = functions["_current_unit_attestation"]

    def referenced_names(node: ast.AST) -> set[str]:
        return {
            child.id for child in ast.walk(node) if isinstance(child, ast.Name)
        }

    preflight_names = referenced_names(preflight)
    full_names = referenced_names(full)
    assert "_current_unit_attestation" not in preflight_names
    assert "CURRENT_UNIT_ATTESTATION" not in preflight_names
    assert "CURRENT_DIRECT_TEST_RESULT" not in preflight_names
    assert "compile_full_vla_projective_dag" not in preflight_names
    assert "predict_direct_rq_route_inventory" not in preflight_names
    assert "_current_unit_attestation" in full_names
    assert "current_unit" in full_names
    assert "_arm_inactive_vision_norm_sentinel" in preflight_names
    assert "_arm_inactive_vision_norm_sentinel" in full_names
    full_remove_calls = [
        node
        for node in ast.walk(full)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_remove_hooks"
    ]
    assert len(full_remove_calls) == 1
    assert isinstance(full_remove_calls[0].args[0], ast.Name)
    assert full_remove_calls[0].args[0].id == "active_norm_handles"
    full_text = ast.unparse(full)
    assert "armed_through_full_rank" in full_text
    assert "armed_through_full_exact_and_compression" in full_text
    expected_gate_assignments = [
        child
        for child in ast.walk(current_unit)
        if isinstance(child, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "expected_unit_gate_names"
            for target in child.targets
        )
    ]
    assert len(expected_gate_assignments) == 1
    expected_gate_literal = expected_gate_assignments[0].value
    assert isinstance(expected_gate_literal, ast.Set)
    expected_gate_names = {
        element.value
        for element in expected_gate_literal.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }

    combined_source = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_capable_linear_b1c0_tests.py"
    )
    combined_module = ast.parse(combined_source.read_text())
    combined_main = next(
        node
        for node in combined_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    emitted_gate_assignments = [
        child
        for child in ast.walk(combined_main)
        if isinstance(child, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "gates"
            for target in child.targets
        )
    ]
    assert len(emitted_gate_assignments) == 1
    emitted_gate_literal = emitted_gate_assignments[0].value
    assert isinstance(emitted_gate_literal, ast.Dict)
    emitted_gate_names = {
        key.value
        for key in emitted_gate_literal.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert expected_gate_names == emitted_gate_names
    assert "norm_inventory_and_dead_site_tests_present" in expected_gate_names


def test_checkpoint_norm_inventory_allows_only_the_exact_dead_vision_site() -> None:
    numerator = (
        5.3299665451049805,
        7.535519123077393,
        0.32727372646331787,
    )
    denominator = (1.0, 9.648269653320312, 2.606637477874756)
    state: dict[str, object] = {}
    for index in range(73):
        state[f"active.{index}.initialized"] = True
        state[f"active.{index}.running_ms"] = 0.75 + index / 1000
        state[f"active.{index}.pa"] = numerator
        state[f"active.{index}.pb"] = denominator
    state["vision.norm_out.initialized"] = False
    state["vision.norm_out.running_ms"] = 1.0
    state["vision.norm_out.pa"] = numerator
    state["vision.norm_out.pb"] = denominator

    arguments = {
        "expected_count": 74,
        "expected_inactive_sites": ("vision.norm_out",),
        "expected_inactive_running_ms": {"vision.norm_out": 1.0},
        "expected_pade_numerator": numerator,
        "expected_pade_denominator": denominator,
    }

    report = require_rational_norm_buffer_inventory(
        state,
        **arguments,
    )
    assert report["initialized_true_count"] == 73
    assert report["initialized_false_sites"] == ["vision.norm_out"]

    active_false = dict(state)
    active_false["active.7.initialized"] = False
    with pytest.raises(RuntimeError, match="inactive-site mask"):
        require_rational_norm_buffer_inventory(
            active_false,
            **arguments,
        )

    fabricated_dead = dict(state)
    fabricated_dead["vision.norm_out.initialized"] = True
    with pytest.raises(RuntimeError, match="inactive-site mask"):
        require_rational_norm_buffer_inventory(
            fabricated_dead,
            **arguments,
        )

    changed_dead_scale = dict(state)
    changed_dead_scale["vision.norm_out.running_ms"] = 1.25
    with pytest.raises(RuntimeError, match="inactive running statistic"):
        require_rational_norm_buffer_inventory(
            changed_dead_scale,
            **arguments,
        )

    nonfinite_active = dict(state)
    nonfinite_active["active.11.running_ms"] = float("nan")
    with pytest.raises(RuntimeError, match="nonfinite"):
        require_rational_norm_buffer_inventory(
            nonfinite_active,
            **arguments,
        )

    nonpositive_active = dict(state)
    nonpositive_active["active.11.running_ms"] = 0.0
    with pytest.raises(RuntimeError, match="not positive"):
        require_rational_norm_buffer_inventory(nonpositive_active, **arguments)

    changed_numerator = dict(state)
    changed_numerator["vision.norm_out.pa"] = (0.0, *numerator[1:])
    with pytest.raises(RuntimeError, match=r"\.pa"):
        require_rational_norm_buffer_inventory(changed_numerator, **arguments)

    changed_denominator = dict(state)
    changed_denominator["vision.norm_out.pb"] = (*denominator[:-1], 0.0)
    with pytest.raises(RuntimeError, match=r"\.pb"):
        require_rational_norm_buffer_inventory(changed_denominator, **arguments)

    project_root = Path(__file__).resolve().parents[1]
    loader_sources = {
        "exact": project_root / "scripts/run_capable_linear_direct_odt_spectrum.py",
        "capability": project_root / "scripts/run_capable_linear_fresh_capability.py",
    }
    expected_keywords = {
        "expected_count",
        "expected_inactive_sites",
        "expected_inactive_running_ms",
        "expected_pade_numerator",
        "expected_pade_denominator",
    }
    for label, source in loader_sources.items():
        module = ast.parse(source.read_text())
        functions = {
            node.name: node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
        }
        loader = functions["_load_model"]
        calls = [
            node
            for node in ast.walk(loader)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "require_rational_norm_buffer_inventory"
        ]
        assert len(calls) == 1, label
        assert {keyword.arg for keyword in calls[0].keywords} == expected_keywords
        assert "EXPECTED_NORMALIZATION_SITE_COUNT" in {
            node.id for node in ast.walk(loader) if isinstance(node, ast.Name)
        }
