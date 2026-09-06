from __future__ import annotations

import ast
import copy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import athena
import athena.product_rational_exact_odt_protocol as protocol
import athena.product_rational_protocol as capability_protocol
import athena.run_product_rational_exact_odt as runner
from scripts.odt_direct_only_compliance import audit_direct_only_launch
from xvla.models.vla import ChiVLA, VLAConfig
from xvla.nn.normalization import RationalNorm
from xvla.train.direct_odt_truncation import (
    CompactRankBank,
    evaluate_diagonal_prefix_quotient,
    evaluate_diagonal_prefixes,
    implicit_storage_elements,
    projected_prefix_storage_elements,
    truncate_diagonal_prefixes,
)
from xvla.train.implicit_sparse_projective_odt import (
    _projective_batch_relative_error,
    canonicalize_implicit_dag_direct_rq,
    diagonalize_implicit_dag_full_rank,
    evaluate_boundary_quotient,
    evaluate_projective_boundary,
    implicit_shape_statistics,
    predict_direct_rq_route_inventory,
)
from xvla.train.implicit_sparse_projective_odt_vla import (
    PRODUCT_COMPONENTS_OBSERVABLE,
    compile_full_vla_projective_dag,
    evaluate_full_vla_observable,
    physical_batch_from_model_inputs,
    physical_source_mapping,
    source_full_vla_observable,
    source_full_vla_output,
)


ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64


def _tiny_product() -> ChiVLA:
    torch.manual_seed(20260905)
    model = ChiVLA(
        VLAConfig(
            image_size=1,
            patch_size=1,
            vit_dim=2,
            vit_layers=1,
            vit_heads=1,
            vit_ffn_rank=3,
            vocab_size=3,
            max_instr_len=1,
            state_dim=1,
            n_embodiments=1,
            dim=4,
            n_layers=1,
            n_heads=1,
            ffn_rank=5,
            attn="bilinear",
            vit_attn="bilinear",
            ffn="bilinear",
            norm="rational",
            qk_norm="rational",
            residual=True,
            vit_residual=True,
            action_horizon=1,
            action_dim=2,
            action_head="product",
            head_rank=3,
            n_factors=2,
        )
    ).to(dtype=DTYPE).eval()
    with torch.no_grad():
        for index, module in enumerate(
            item for item in model.modules() if isinstance(item, RationalNorm)
        ):
            module.running_ms.fill_(0.8 + 0.01 * index)
            module.initialized.fill_(True)
            module.frozen = True
    return model


def _tiny_panel(model: ChiVLA):
    images = torch.tensor(
        [
            [[[0.1]], [[-0.2]], [[0.3]]],
            [[[-0.4]], [[0.2]], [[0.05]]],
        ],
        dtype=DTYPE,
    )
    instructions = torch.tensor(((1,), (2,)), dtype=torch.long)
    state = torch.tensor(((0.15,), (-0.11,)), dtype=DTYPE)
    embodiment = torch.zeros(2, dtype=torch.long)
    return physical_batch_from_model_inputs(
        model, images, instructions, state, embodiment
    )


def _synthetic_full_rank_certificate(monkeypatch):
    """Build the smallest internally consistent production-certificate record."""

    manifest_sha = "f" * 64
    static = {
        "scope": "transitive_local_import_closure",
        "entrypoints": ["athena/run_product_rational_exact_odt.py"],
        "source_sha256": {"athena/run_product_rational_exact_odt.py": "a" * 64},
        "source_count": 1,
        "direct_qr_call_sites": 4,
        "direct_qr_required": True,
        "guarded_dormant_spectral_norm_sites": [],
        "duplicate_top_level_definition_sites": [],
        "prohibited_self_overlap_sites": [],
        "prohibited_calls_found": [],
        "call_site_count": 4,
    }
    monkeypatch.setattr(runner, "_TRANSITIVE_STATIC_AUDIT", static)
    monkeypatch.setattr(runner, "file_sha256", lambda _path: manifest_sha)
    inputs = {
        "checkpoint_sha256": "1" * 64,
        "metadata_sha256": "2" * 64,
        "training_result_sha256": "3" * 64,
        "smoke_calibration_resume_proof_sha256": "4" * 64,
    }
    manifest = {"manifest_bundle_sha256": "5" * 64}
    preflight = {"path": "/frozen/preflight.json", "sha256": "6" * 64}
    clone = {"path": "/frozen/clone.json", "sha256": "7" * 64}
    lane_tests = {"path": "/frozen/lane.json", "sha256": "8" * 64}
    initial_shape = {
        "unique_nodes": 4,
        "edge_occurrences": 3,
        "cp_binary_nodes": 3,
        "reduced_q_binary_nodes": 0,
        "dense_clone_nodes": 0,
        "unary_nodes": 1,
        "heterogeneous_physical_sources": 97,
        "maximum_local_bond_dimension": 8,
        "maximum_cp_rank": 4,
        "maximum_reduced_q_elements": 0,
        "fused_token_feature_dimension": 384,
    }
    canonical_shape = {
        **initial_shape,
        "cp_binary_nodes": 1,
        "reduced_q_binary_nodes": 2,
        "maximum_reduced_q_elements": 64,
    }
    widths = {f"source_{index}": 1 for index in range(97)}
    widths["source_0"] = 192
    structure = {
        "unique_nodes": 4,
        "edge_occurrences": 3,
        "physical_source_count": 97,
        "physical_leaf_count": 97,
        "physical_source_widths": widths,
        "maximum_physical_source_width": 192,
        "root_projective_width": 285,
        "observable_kind": PRODUCT_COMPONENTS_OBSERVABLE,
        "observable_euclidean_width": 284,
        "product_signs": None,
        "active_pade_sites": capability_protocol.EXPECTED_PADE_SITE_COUNT,
    }
    route = {
        "schema": "exact_postorder_structural_direct_rq_routes_v1",
        "bounded_explicit_unfolding_element_limit": 1_000,
        "bounded_explicit_q_element_limit": 1_000,
        "bounded_retained_q_candidate_count": 2,
        "bounded_retained_q_total_elements": 100,
        "bounded_retained_q_replaced_cp_total_elements": 80,
        "bounded_retained_q_signed_storage_delta_elements": 20,
        "bounded_retained_q_positive_storage_delta_elements": 20,
        "bounded_retained_q_maximum_unfolding_elements": 40,
        "bounded_retained_q_maximum_elements": 50,
        "streamed_cp_candidate_count": 1,
        "streamed_tall_unrepresentable_count": 0,
        "streamed_tall_unrepresentable_shape_counts": {},
        "bounded_retained_q_shape_counts": {"2x2": 2},
        "nodes_with_propagated_input_shape_change": 0,
        "predicted_root_output_dimension": 285,
        "simulated_unique_node_count": 4,
    }
    postorder_sha = runner._canonical_sha256([0, 1, 2, 3])
    provenance = {
        "verified_steps": 4,
        "expected_steps": 4,
        "streamed_certificates": 1,
        "streamed_steps": 1,
        "columns_compared": 10,
        "expected_columns": 10,
        "streamed_replay_calls": 2,
        "expected_streamed_replay_calls": 2,
        "maximum_compact_q_relative_error": 0.0,
        "maximum_reconstruction_relative_error": 0.0,
    }
    telemetry = {key: 0 for key in runner._TELEMETRY_KEYS}
    telemetry.update(
        {
            "maximum_persistent_tensor_elements": 100,
            "maximum_temporary_tensor_elements": 50,
            "maximum_temporary_tensor_order": 2,
            "local_direct_rq_factorizations": 4,
            "householder_qr_kernel_calls": 7,
            "fused_ffn_primitive_count": 1,
            "rectangular_direct_rq_factorizations": 2,
            "maximum_rectangular_q_elements": 64,
            "bounded_explicit_direct_rq_factorizations": 2,
            "maximum_bounded_explicit_q_elements": 64,
            "direct_q_provenance_certificates": 4,
            "streamed_direct_q_provenance_certificates": 1,
            "streamed_direct_q_replay_qr_factorizations": 2,
            "direct_q_columns_compared": 10,
            "maximum_direct_q_transition_elements": 8,
            "maximum_direct_q_panel_elements": 16,
            "maximum_direct_q_compact_relative_error": 0.0,
            "maximum_direct_q_reconstruction_relative_error": 0.0,
        }
    )
    algorithm1 = {
        "step_count": 4,
        "factorization_methods": sorted(
            {
                runner.DIRECT_RQ_METHOD,
                runner.RECTANGULAR_RQ_METHOD,
                runner.UNARY_RQ_METHOD,
            }
        ),
        "factorization_method_counts": {
            runner.DIRECT_RQ_METHOD: 1,
            runner.RECTANGULAR_RQ_METHOD: 2,
            runner.UNARY_RQ_METHOD: 1,
        },
        "canonical_postorder_uid_sha256": postorder_sha,
        "algorithm1_step_uid_sha256": postorder_sha,
        "child_parent_order_violation_count": 0,
        "final_projective_replay_relative_error": 0.0,
        "full_projective_replay_evaluations": 2,
        "per_step_replays_performed": 0,
        "maximum_local_scaled_reconstruction_relative_error": 0.0,
        "maximum_absorption_scaled_relative_error": 0.0,
        "maximum_local_reconstruction_exponent_delta": 0,
        "maximum_absorption_exponent_delta": 0,
        "push_ledger": [4, 4],
        "scale_sensitive_occurrence_ledger": [4, 4],
        "canonical_exponent_normal_form": {
            "core_count": 4,
            "all_core_exponents_zero": True,
            "head_binary_exponent": 0,
            "head_exponent_matches_ledger": True,
            "cp_direct_q_certificates": 1,
            "cp_direct_q_columns": 2,
            "streamed_compact_cp_direct_q_certificates": 1,
            "streamed_compact_cp_direct_q_columns": 2,
            "all_streamed_compact_cp_q_has_direct_q_provenance": True,
        },
        "direct_q_provenance": provenance,
        "telemetry": telemetry,
    }
    algorithms23 = {
        "explicit_downstream_contraction": True,
        "streamed_root_first": True,
        "child_messages": 3,
        "eigenbasis_order": "descending environment eigenvalue",
        "diagonalized_nodes": 4,
        "occurrence_push_ledger": [4, 4],
        "replay_relative_error": 0.0,
        "maximum_recontracted_offdiagonal_ratio": 0.0,
        "full_spectra_retained": False,
        "post_environment_records_retained": False,
    }
    operation_counts = {
        "direct_rq_householder_qr_calls": 3,
        "streamed_tsqr_householder_qr_calls": 4,
        "triangular_solve_calls": 1,
        "environment_evd_calls": 4,
        "total_controlled_calls": 12,
    }
    allowed_calls = sorted(
        {
            runner.DIRECT_RQ_RUNTIME_CALL,
            runner.STREAMED_QR_RUNTIME_CALL,
            runner.TRIANGULAR_RUNTIME_CALL,
            runner.PRODUCTION_ALGORITHM3_RUNTIME_CALL,
        }
    )
    runtime = {
        "installed": True,
        "patched_entrypoints": sorted(runner.EXPECTED_RUNTIME_GUARD_ENTRYPOINTS),
        "patched_entrypoint_count": runner.EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT,
        "allowed_call_count": 12,
        "allowed_calls": allowed_calls,
        "prohibited_attempt_count": 0,
        "prohibited_attempts": [],
    }
    panel = {
        "kind": "deterministic_functional_replay_not_task_degradation",
        "seed": runner.SYNTHETIC_REPLAY_SEED,
        "official_task_indices": list(runner.SYNTHETIC_REPLAY_TASKS),
        "instruction_token_sha256": "9" * 64,
        "image_sha256": "a" * 64,
        "state_sha256": "b" * 64,
        "embodiment": 0,
        "batch_size": len(runner.SYNTHETIC_REPLAY_TASKS),
    }
    panel["panel_sha256"] = runner._canonical_sha256(panel)
    payload = {
        "schema": f"{runner.SCHEMA}_algorithm3_full_rank_complete_before_compression",
        "checkpoint_sha256": inputs["checkpoint_sha256"],
        "metadata_sha256": inputs["metadata_sha256"],
        "training_result_sha256": inputs["training_result_sha256"],
        "smoke_calibration_resume_proof_sha256": inputs[
            "smoke_calibration_resume_proof_sha256"
        ],
        "preflight": preflight,
        "bounded_clone_oracle": clone,
        "exact_lane_tests": lane_tests,
        "dooms_reference": {
            "sha256": runner.DOOMS_REFERENCE_SHA256,
            "orientation": runner.DOOMS_ORIENTATION,
        },
        "replay_panel": panel,
        "source_replay": {
            "component_relative_error": 0.0,
            "public_action_relative_error": 0.0,
            "source_decode_relative_error": 0.0,
            "source_decode_bitwise_equal": True,
            "source_positive_sign_fraction": 0.5,
            "source_zero_gate_count": 0,
        },
        "structure": structure,
        "initial_shape": initial_shape,
        "canonical_shape": canonical_shape,
        "route_inventory": route,
        "algorithm1": algorithm1,
        "algorithm2_and3": algorithms23,
        "numerical_execution": {
            "runtime": runner._runtime_environment(),
            "model_parameter_dtypes": ["torch.float64"],
            "model_floating_buffer_dtypes": ["torch.float64"],
            "physical_source_dtypes": ["torch.float64"],
            "running_ms_guard_revalidated_after_float64_conversion": True,
            "controlled_operation_counts": operation_counts,
        },
        "timings_seconds": {
            "compile": 1.0,
            "source_evaluation": 1.0,
            "algorithm1": 1.0,
            "algorithms2_and3": 1.0,
            "through_full_rank": 4.0,
        },
        "peak_rss_mb": 10.0,
        "full_rank_gates": {key: True for key in runner._FULL_RANK_GATE_KEYS},
        "all_full_rank_gates_pass": True,
        "runtime_guard": runtime,
        "static_audit": static,
        "manifest_sha256": manifest_sha,
        "manifest_bundle_sha256": manifest["manifest_bundle_sha256"],
        "bounded_clone_oracle_hash_joined": True,
        "exact_lane_test_hash_joined": True,
        "canonical_direct_odt_algorithms_1_to_3_completed": True,
        "exact_global_decomposability_certified": True,
        "compression_claimed": False,
        "capability_claimed": False,
        "capable_global_decomposable_vla_claimed": False,
    }
    arguments = {
        "inputs": inputs,
        "manifest": manifest,
        "preflight": preflight,
        "clone": clone,
        "lane_tests": lane_tests,
    }
    return payload, arguments


def _synthetic_compression_result(
    full_rank: dict,
    arguments: dict,
    full_rank_validation: dict,
):
    target_keys = [str(value) for value in runner.COMPACT_TRACE_RETENTION_TARGETS]
    floor_keys = [str(value) for value in runner.COMPACT_RELATIVE_VALUE_FLOORS]

    def group(nodes: int, dimension: int) -> dict:
        return {
            "node_count": nodes,
            "total_dimension": dimension,
            "trace_retained_dimensions": {
                key: dimension for key in target_keys
            },
            "relative_floor_dimensions": {
                key: dimension for key in floor_keys
            },
            "trace_tail_sums": {
                key: {
                    "mantissa": 0.0,
                    "binary_exponent": 0,
                    "log2_value": None,
                }
                for key in target_keys
            },
            "zero_trace_nodes": 0,
            "tie_extended_nodes": {key: 0 for key in target_keys},
        }

    groups = {
        "product_center": group(1, 10),
        "product_factors": group(2, 20),
        "product_gates": group(1, 5),
    }
    census = {
        "node_count": 4,
        "full_spectra_retained": False,
        "tail_quantity": (
            "sum of discarded Algorithm 2 environment eigenvalues, with "
            "the environment binary scale restored"
        ),
        "simultaneous_multibond_error_bound_claimed": False,
        "trace_targets": list(runner.COMPACT_TRACE_RETENTION_TARGETS),
        "relative_value_floors": list(runner.COMPACT_RELATIVE_VALUE_FLOORS),
        "overall": group(4, 35),
        "groups": groups,
    }
    metrics = {
        "component_relative_error": 0.0,
        "decoded_action_relative_error": 0.0,
        "decoded_action_max_absolute_error": 0.0,
        "decoded_action_coordinate_max_absolute_error": [
            0.0 for _ in range(capability_protocol.ACTION_DIM)
        ],
        "gate_sign_flip_count": 0,
        "gate_sign_flip_fraction": 0.0,
    }
    curves = {
        str(target): {
            "target": target,
            "retained_bond_dimensions": 4,
            "original_bond_dimensions": 8,
            "projected_storage_elements": 80,
            "projected_storage_fraction": 0.8,
            "prefix_metrics": copy.deepcopy(metrics),
            "prefix_evaluation_seconds": 1.0,
            "matched_width_trailing_control": {
                "valid_projective_chart": False,
                "rejection": "zero/nonfinite coordinates",
                "evaluation_seconds": 1.0,
            },
        }
        for target in runner.COMPACT_TRACE_RETENTION_TARGETS
    }
    material = {
        "target": runner.PHYSICAL_MATERIALIZATION_TARGET,
        "performed_after_all_masked_curves": True,
        "copy_network": False,
        "executor_kind": "materialized_implicit_projective_dag",
        "materialized_tensor_network_executed": True,
        "source_model_forward_fallback_forbidden": True,
        "source_model_forward_calls_during_materialized_execution": 0,
        "projective_pair_relative_error_against_masked": 1.0,
        "projective_pair_gauge_invariant_relative_error_against_masked": 0.0,
        "component_relative_error_against_masked": 0.0,
        "decoded_action_relative_error_against_masked": 0.0,
        "decoded_action_coordinate_max_absolute_error_against_masked": [
            0.0 for _ in range(capability_protocol.ACTION_DIM)
        ],
        "gate_sign_mismatch_count_against_masked": 0,
        "original_storage_elements": 100,
        "projected_storage_elements": 80,
        "materialized_storage_elements": 80,
        "allocated_storage_elements": 80,
        "reduced_bonds": 1,
        "expected_occurrence_slices": 2,
        "applied_occurrence_slices": 2,
        "elapsed_seconds": 1.0,
    }
    full_rank_sha = "c" * 64
    result = {
        "schema": f"{runner.SCHEMA}_full_with_same_sweep_compression",
        "checkpoint_sha256": arguments["inputs"]["checkpoint_sha256"],
        "smoke_calibration_resume_proof_sha256": arguments["inputs"][
            "smoke_calibration_resume_proof_sha256"
        ],
        "full_rank_certificate": {
            "path": runner.full_rank_progress_path().as_posix(),
            "sha256": full_rank_sha,
        },
        "bounded_clone_oracle": arguments["clone"],
        "exact_lane_tests": arguments["lane_tests"],
        "preflight": arguments["preflight"],
        "replay_panel": full_rank["replay_panel"],
        "full_rank_gates": full_rank["full_rank_gates"],
        "all_full_rank_gates_pass": True,
        "compact_census": census,
        "canonical_storage_elements": 100,
        "compression_curves": curves,
        "compression_curves_are_functional_not_task_degradation": True,
        "physical_materialization": material,
        "physical_materialization_gates": {
            key: True for key in runner._MATERIALIZATION_GATE_KEYS
        },
        "all_compression_gates_pass": True,
        "timings_seconds": {**full_rank["timings_seconds"], "total": 10.0},
        "peak_rss_mb": 10.0,
        "runtime_guard_final": full_rank["runtime_guard"],
        "static_audit": full_rank["static_audit"],
        "manifest_sha256": full_rank["manifest_sha256"],
        "manifest_bundle_sha256": full_rank["manifest_bundle_sha256"],
        "canonical_direct_odt_algorithms_1_to_3_completed": True,
        "bounded_clone_oracle_hash_joined": True,
        "exact_lane_test_hash_joined": True,
        "exact_global_decomposability_certified": True,
        "compression_claimed_after_full_rank_certificate": True,
        "capability_claimed": False,
        "capable_global_decomposable_vla_claimed": False,
    }
    validation_arguments = {
        "full_rank": full_rank,
        "full_rank_sha256": full_rank_sha,
        "full_rank_validation": full_rank_validation,
        **arguments,
    }
    return result, validation_arguments


def test_exact_protocol_and_runner_duplicate_the_same_complete_tuples():
    assert runner._EXPECTED_SOURCE_CLOSURE == protocol.SOURCE_CLOSURE
    assert runner._EXPECTED_AUTHENTICATED_INPUTS == protocol.AUTHENTICATED_INPUTS
    assert runner._EXPECTED_LAUNCH_CLOSURE == protocol.LAUNCH_CLOSURE
    assert "athena/__init__.py" in protocol.SOURCE_CLOSURE
    assert Path(athena.__file__).resolve() == (ROOT / "athena/__init__.py").resolve()
    assert "tests/test_direct_odt_truncation.py" in protocol.SOURCE_CLOSURE
    assert "tests/test_direct_odt_clone_reference.py" in protocol.SOURCE_CLOSURE
    assert "reference/dooms_xnets_2504.02667.pdf" in protocol.AUTHENTICATED_INPUTS
    assert protocol.ODT_SOURCE_OVERRIDES == (
        "scripts/odt_direct_only_compliance.py",
    )


def test_exact_python_and_launch_closure_are_clean_and_complete():
    report = audit_direct_only_launch(
        ROOT,
        tuple(ROOT / relative for relative in protocol.SOURCE_CLOSURE),
    )
    assert tuple(sorted(report["source_sha256"])) == tuple(
        sorted(protocol.SOURCE_CLOSURE)
    )
    assert report["prohibited_calls_found"] == []
    assert report["prohibited_self_overlap_sites"] == []
    assert report["guarded_dormant_spectral_norm_sites"] == []
    blocked = (
        "s" + "vd",
        "g" + "ram",
        "p" + "olar",
        "c" + "ovariance",
        "normal" + "-equation",
        "normal" + " equation",
        "p" + "inv",
        "l" + "stsq",
    )
    for relative in protocol.LAUNCH_CLOSURE:
        text = (ROOT / relative).read_text().lower()
        assert not [token for token in blocked if token in text]


def test_reference_identity_and_orientation_record_are_fixed():
    reference = ROOT / "tmp/pdfs/dooms-xnets-2504.02667.pdf"
    if reference.is_file():
        assert protocol.file_sha256(reference) == protocol.DOOMS_REFERENCE_SHA256
    assert runner.DOOMS_ORIENTATION == {
        "reference_algorithm1_order": "child-before-parent postorder",
        "reference_algorithm1_core_update": "replace each local unfolding by direct reduced-RQ Q",
        "reference_algorithm1_factor_update": "push R into every connected parent-input occurrence or the root head",
        "reference_algorithm2_order": "root-first downstream environment contraction from the output head",
        "reference_algorithm3_output_update": "apply each ordered eigenbasis transpose on the bond output",
        "reference_algorithm3_input_update": "apply the same eigenbasis on every connected parent-input occurrence or the root head",
    }
    assert type(runner._runtime_environment()["torch"]) is str


def test_tiny_product_all_components_runs_direct_algorithms_and_compression():
    model = _tiny_product()
    physical = _tiny_panel(model)
    oracle = compile_full_vla_projective_dag(model)
    assert oracle.observable_kind == PRODUCT_COMPONENTS_OBSERVABLE
    assert oracle.product_signs is None
    raw = physical_source_mapping(model, physical)
    source_components = source_full_vla_observable(oracle, physical)
    source_actions = source_full_vla_output(model, physical)
    compiled = evaluate_full_vla_observable(oracle, physical)
    assert runner._relative(compiled, source_components) < 3e-10
    decoded, signs = runner._decode_components(model, source_components)
    assert runner._relative(decoded, source_actions) < 3e-12
    assert bool(((signs == -1) | (signs == 1)).all())
    routes = predict_direct_rq_route_inventory(oracle.network)
    assert routes["streamed_tall_unrepresentable_count"] == 0
    canonical = canonicalize_implicit_dag_direct_rq(
        oracle.network,
        replay_inputs=raw,
        block_size=64,
        replay_each_step=True,
    )
    shape = implicit_shape_statistics(canonical.network)
    bank = CompactRankBank.for_network(canonical.network)
    diagonal = diagonalize_implicit_dag_full_rank(
        canonical.network,
        replay_inputs=raw,
        retain_eigenvalues=False,
        retain_post_evd_environments=False,
        compact_spectrum_callback=bank,
    )
    assert bank.complete
    assert diagonal.eigenvalues == ()
    assert not diagonal.eigenvalue_spectra_retained
    assert all(runner._dooms_order_gates(canonical, diagonal, shape).values())
    census = bank.finish()
    assert all(
        census["groups"][name]["node_count"] > 0
        for name in ("product_center", "product_factors", "product_gates")
    )
    plan = bank.plan(protocol.PHYSICAL_MATERIALIZATION_TARGET)
    projected = projected_prefix_storage_elements(diagonal, plan)
    masked_pair = evaluate_diagonal_prefixes(diagonal, raw, plan)
    masked_components = evaluate_diagonal_prefix_quotient(diagonal, raw, plan)
    materialized = truncate_diagonal_prefixes(diagonal, plan)
    replay_pair = evaluate_projective_boundary(materialized.network, raw)
    replay_components = evaluate_boundary_quotient(materialized.network, raw)
    assert _projective_batch_relative_error(replay_pair, masked_pair) < 2e-10
    assert runner._relative(replay_components, masked_components) < 2e-10
    assert materialized.truncated_storage_elements == projected
    assert materialized.allocated_storage_elements == projected
    assert materialized.truncated_storage_elements < implicit_storage_elements(diagonal.network)


def test_projective_pair_gate_ignores_only_nonzero_rowwise_scale():
    expected = torch.tensor(
        ((1.0, 2.0, 1.0), (-2.0, 5.0, 3.0), (0.4, -0.7, 2.0)),
        dtype=DTYPE,
    )
    scaled = expected * torch.tensor((0.5, -3.0, 8.0), dtype=DTYPE)[:, None]
    assert _projective_batch_relative_error(scaled, expected) < 1e-15
    assert runner._relative(scaled, expected) > 0.1
    source = (ROOT / "athena/run_product_rational_exact_odt.py").read_text()
    tree = ast.parse(source)
    materialization_gate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and any(
            isinstance(key, ast.Constant)
            and key.value == "projective_pair_matches_masked_up_to_rowwise_scale"
            for key in node.keys
        )
    )
    rendered = ast.unparse(materialization_gate)
    assert "projective_pair_gauge_invariant_relative_error_against_masked" in rendered
    assert "projective_pair_relative_error_against_masked" not in rendered


def test_orientation_and_occurrence_mutations_fail_their_gates():
    leaf = SimpleNamespace(uid=0, children=())
    root = SimpleNamespace(uid=1, children=(leaf,))
    records = (
        SimpleNamespace(
            uid=0,
            direct_q_provenance_verified=True,
            parent_occurrences_pushed=1,
            expected_parent_occurrences=1,
        ),
        SimpleNamespace(
            uid=1,
            direct_q_provenance_verified=True,
            parent_occurrences_pushed=1,
            expected_parent_occurrences=1,
        ),
    )
    canonical = SimpleNamespace(
        network=SimpleNamespace(root=root),
        steps=records,
    )
    diagonal = SimpleNamespace(
        algorithm2_child_messages=1,
        diagonalized_node_count=2,
        pushed_parent_occurrences=2,
        expected_parent_occurrences=2,
    )
    shape = {"unique_nodes": 2, "edge_occurrences": 1}
    assert all(runner._dooms_order_gates(canonical, diagonal, shape).values())
    reversed_canonical = SimpleNamespace(
        network=canonical.network, steps=tuple(reversed(records))
    )
    assert not runner._dooms_order_gates(reversed_canonical, diagonal, shape)[
        "algorithm1_child_before_parent"
    ]
    missing_push = SimpleNamespace(
        network=canonical.network,
        steps=(
            records[0],
            SimpleNamespace(
                uid=1,
                direct_q_provenance_verified=True,
                parent_occurrences_pushed=0,
                expected_parent_occurrences=1,
            ),
        ),
    )
    assert not runner._dooms_order_gates(missing_push, diagonal, shape)[
        "algorithm1_r_reaches_every_occurrence"
    ]


def test_full_rank_consumer_reconstructs_every_algorithm_and_runtime_counter(
    monkeypatch,
):
    payload, arguments = _synthetic_full_rank_certificate(monkeypatch)
    validated = runner._validate_full_rank_certificate_payload(payload, **arguments)
    assert validated["validated"] is True
    assert validated["runtime_call_count"] == 12

    mutations = (
        ("top-level schema", ("unreviewed",), True),
        ("Algorithm 1 method route", ("route_inventory", "streamed_cp_candidate_count"), 2),
        ("Algorithm 1 order", ("algorithm1", "algorithm1_step_uid_sha256"), "0" * 64),
        ("Algorithm 1 order count", ("algorithm1", "child_parent_order_violation_count"), 1),
        ("Algorithm 1 reconstruction", ("algorithm1", "final_projective_replay_relative_error"), 1.0),
        ("Algorithm 1 every occurrence", ("algorithm1", "push_ledger"), [3, 4]),
        (
            "Algorithm 1 direct provenance",
            ("algorithm1", "direct_q_provenance", "verified_steps"),
            3,
        ),
        (
            "Algorithm 1 normal form",
            ("algorithm1", "canonical_exponent_normal_form", "all_core_exponents_zero"),
            False,
        ),
        (
            "Algorithm 1 QR telemetry",
            ("algorithm1", "telemetry", "householder_qr_kernel_calls"),
            6,
        ),
        (
            "Algorithm 1 replay telemetry",
            (
                "algorithm1",
                "telemetry",
                "streamed_direct_q_replay_qr_factorizations",
            ),
            1,
        ),
        (
            "Algorithm 2 contraction",
            ("algorithm2_and3", "explicit_downstream_contraction"),
            False,
        ),
        ("Algorithm 2 messages", ("algorithm2_and3", "child_messages"), 2),
        ("Algorithm 3 EVD", ("algorithm2_and3", "diagonalized_nodes"), 3),
        (
            "Algorithm 3 occurrence gauge",
            ("algorithm2_and3", "occurrence_push_ledger"),
            [3, 4],
        ),
        ("Algorithm 3 replay", ("algorithm2_and3", "replay_relative_error"), 1.0),
        (
            "Algorithm 3 diagonality",
            ("algorithm2_and3", "maximum_recontracted_offdiagonal_ratio"),
            1.0,
        ),
        (
            "Algorithm 3 spectrum retention",
            ("algorithm2_and3", "full_spectra_retained"),
            True,
        ),
        (
            "float64 boundary",
            ("numerical_execution", "physical_source_dtypes"),
            ["torch.float32"],
        ),
        (
            "post-conversion running-ms guard",
            (
                "numerical_execution",
                "running_ms_guard_revalidated_after_float64_conversion",
            ),
            False,
        ),
        (
            "direct RQ QR operation count",
            (
                "numerical_execution",
                "controlled_operation_counts",
                "direct_rq_householder_qr_calls",
            ),
            4,
        ),
        (
            "streamed QR operation count",
            (
                "numerical_execution",
                "controlled_operation_counts",
                "streamed_tsqr_householder_qr_calls",
            ),
            3,
        ),
        (
            "triangular operation count",
            (
                "numerical_execution",
                "controlled_operation_counts",
                "triangular_solve_calls",
            ),
            2,
        ),
        (
            "environment EVD operation count",
            (
                "numerical_execution",
                "controlled_operation_counts",
                "environment_evd_calls",
            ),
            5,
        ),
        (
            "total controlled operation count",
            (
                "numerical_execution",
                "controlled_operation_counts",
                "total_controlled_calls",
            ),
            13,
        ),
        ("runtime call count", ("runtime_guard", "allowed_call_count"), 11),
        ("runtime call identities", ("runtime_guard", "allowed_calls"), []),
        (
            "runtime patched-count type alias",
            ("runtime_guard", "patched_entrypoint_count"),
            True,
        ),
        (
            "runtime zero type alias",
            ("runtime_guard", "prohibited_attempt_count"),
            False,
        ),
        ("static authority", ("static_audit", "source_count"), 2),
        (
            "static zero type alias",
            ("static_audit", "direct_qr_call_sites"),
            False,
        ),
        ("replay integer type alias", ("replay_panel", "seed"), True),
        (
            "source replay integer type alias",
            ("source_replay", "source_zero_gate_count"),
            False,
        ),
        ("checkpoint authority", ("checkpoint_sha256",), "0" * 64),
        (
            "Dooms orientation",
            ("dooms_reference", "orientation", "reference_algorithm2_order"),
            "leaf-first",
        ),
        ("bounded clone", ("bounded_clone_oracle", "sha256"), "0" * 64),
        ("literal full-rank gate", ("full_rank_gates", next(iter(runner._FULL_RANK_GATE_KEYS))), "true"),
        ("literal claim boolean", ("all_full_rank_gates_pass",), 1),
        ("finite timing", ("timings_seconds", "algorithm1"), float("nan")),
    )
    for label, path, replacement in mutations:
        candidate = copy.deepcopy(payload)
        cursor = candidate
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = replacement
        with pytest.raises((RuntimeError, ValueError), match=".+"):
            runner._validate_full_rank_certificate_payload(candidate, **arguments)

    for field in (
        "direct_rq_householder_qr_calls",
        "streamed_tsqr_householder_qr_calls",
        "triangular_solve_calls",
        "environment_evd_calls",
        "total_controlled_calls",
    ):
        for delta in (-1, 1):
            candidate = copy.deepcopy(payload)
            candidate["numerical_execution"]["controlled_operation_counts"][field] += delta
            with pytest.raises(RuntimeError, match="operation ledger"):
                runner._validate_full_rank_certificate_payload(candidate, **arguments)


def test_compression_consumer_reconstructs_curves_materialization_and_authorities(
    monkeypatch,
):
    full_rank, full_rank_arguments = _synthetic_full_rank_certificate(monkeypatch)
    full_rank_validation = runner._validate_full_rank_certificate_payload(
        full_rank, **full_rank_arguments
    )
    payload, arguments = _synthetic_compression_result(
        full_rank, full_rank_arguments, full_rank_validation
    )
    validated = runner._validate_compression_result_payload(payload, **arguments)
    assert validated == {
        "validated": True,
        "physical_materialization_gates": {
            key: True for key in runner._MATERIALIZATION_GATE_KEYS
        },
        "canonical_storage_elements": 100,
    }

    target = str(runner.COMPACT_TRACE_RETENTION_TARGETS[0])
    gate = next(iter(runner._MATERIALIZATION_GATE_KEYS))
    full_rank_gate = next(iter(runner._FULL_RANK_GATE_KEYS))
    mutations = (
        ("top-level exact schema", ("unreviewed",), True),
        ("full-rank physical path", ("full_rank_certificate", "path"), "/wrong"),
        ("full-rank physical digest", ("full_rank_certificate", "sha256"), "0" * 64),
        ("replay-panel hash join", ("replay_panel", "panel_sha256"), "0" * 64),
        ("bounded-clone authority", ("bounded_clone_oracle", "path"), "/wrong"),
        ("lane-test authority", ("exact_lane_tests", "sha256"), "0" * 64),
        ("preflight authority", ("preflight", "sha256"), "0" * 64),
        ("full-rank literal gate copy", ("full_rank_gates", full_rank_gate), 1),
        ("full-rank closing boolean", ("all_full_rank_gates_pass",), 1),
        ("compact spectra retention", ("compact_census", "full_spectra_retained"), True),
        ("compact node type alias", ("compact_census", "node_count"), True),
        (
            "compact group coverage",
            ("compact_census", "groups", "product_gates", "node_count"),
            0,
        ),
        (
            "compact retained dimension",
            (
                "compact_census",
                "groups",
                "product_center",
                "trace_retained_dimensions",
                target,
            ),
            11,
        ),
        (
            "compact tail numeric type alias",
            (
                "compact_census",
                "groups",
                "product_center",
                "trace_tail_sums",
                target,
                "mantissa",
            ),
            0,
        ),
        ("curve target", ("compression_curves", target, "target"), "0.9"),
        (
            "curve projected storage arithmetic",
            ("compression_curves", target, "projected_storage_elements"),
            81,
        ),
        (
            "curve storage fraction type",
            ("compression_curves", target, "projected_storage_fraction"),
            0,
        ),
        (
            "prefix metric type alias",
            (
                "compression_curves",
                target,
                "prefix_metrics",
                "component_relative_error",
            ),
            0,
        ),
        (
            "prefix gate-sign arithmetic",
            (
                "compression_curves",
                target,
                "prefix_metrics",
                "gate_sign_flip_count",
            ),
            1,
        ),
        (
            "trailing-control literal validity",
            (
                "compression_curves",
                target,
                "matched_width_trailing_control",
                "valid_projective_chart",
            ),
            "false",
        ),
        (
            "materialization target",
            ("physical_materialization", "target"),
            0.5,
        ),
        (
            "materialized executor identity",
            ("physical_materialization", "executor_kind"),
            "source_model",
        ),
        (
            "materialized executor execution",
            ("physical_materialization", "materialized_tensor_network_executed"),
            False,
        ),
        (
            "source-model fallback prohibition",
            (
                "physical_materialization",
                "source_model_forward_fallback_forbidden",
            ),
            False,
        ),
        (
            "source-model fallback call count",
            (
                "physical_materialization",
                "source_model_forward_calls_during_materialized_execution",
            ),
            1,
        ),
        (
            "physical gauge-invariant replay",
            (
                "physical_materialization",
                "projective_pair_gauge_invariant_relative_error_against_masked",
            ),
            1.0,
        ),
        (
            "physical component replay",
            ("physical_materialization", "component_relative_error_against_masked"),
            1.0,
        ),
        (
            "physical action replay",
            (
                "physical_materialization",
                "decoded_action_relative_error_against_masked",
            ),
            1.0,
        ),
        (
            "physical sign replay",
            ("physical_materialization", "gate_sign_mismatch_count_against_masked"),
            1,
        ),
        (
            "physical storage origin",
            ("physical_materialization", "original_storage_elements"),
            101,
        ),
        (
            "physical projected storage",
            ("physical_materialization", "projected_storage_elements"),
            79,
        ),
        (
            "physical materialized storage",
            ("physical_materialization", "materialized_storage_elements"),
            79,
        ),
        (
            "physical allocated storage",
            ("physical_materialization", "allocated_storage_elements"),
            79,
        ),
        (
            "physical nonempty bond reduction",
            ("physical_materialization", "reduced_bonds"),
            0,
        ),
        (
            "physical expected occurrence slicing",
            ("physical_materialization", "expected_occurrence_slices"),
            3,
        ),
        (
            "physical applied occurrence slicing",
            ("physical_materialization", "applied_occurrence_slices"),
            1,
        ),
        ("literal materialization gate", ("physical_materialization_gates", gate), "true"),
        ("literal compression close", ("all_compression_gates_pass",), 1),
        ("preserved exact timing type", ("timings_seconds", "algorithm1"), 1),
        ("final runtime count", ("runtime_guard_final", "allowed_call_count"), 13),
        (
            "final runtime zero type alias",
            ("runtime_guard_final", "prohibited_attempt_count"),
            False,
        ),
        ("static source-count type alias", ("static_audit", "source_count"), True),
        ("manifest authority", ("manifest_bundle_sha256",), "0" * 64),
        (
            "functional-only compression boundary",
            ("compression_curves_are_functional_not_task_degradation",),
            1,
        ),
        ("compression ordering claim", ("compression_claimed_after_full_rank_certificate",), 1),
        ("capability nonclaim", ("capability_claimed",), 0),
    )
    for label, path, replacement in mutations:
        candidate = copy.deepcopy(payload)
        cursor = candidate
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = replacement
        with pytest.raises((RuntimeError, ValueError), match=".+"):
            runner._validate_compression_result_payload(candidate, **arguments)


def test_zero_product_gate_decodes_to_negative_sign():
    model = _tiny_product()
    width = model.product_head.m * (model.product_head.G + 1) + model.product_head.G
    components = torch.zeros(1, width, dtype=DTYPE)
    actions, signs = runner._decode_components(model, components)
    assert torch.equal(signs, -torch.ones_like(signs))
    assert actions.shape == (1, model.cfg.action_horizon, model.cfg.action_dim)


def test_cli_authority_rejects_duplicates_and_abbreviations(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--mode",
            "preflight",
            "--mode",
            "full",
            "--source-manifest",
            protocol.MANIFEST_PATH.as_posix(),
        ],
    )
    with pytest.raises(SystemExit):
        runner.parse_args()
    with pytest.raises(RuntimeError):
        runner._single_preimport_value(
            ["--source-m", protocol.MANIFEST_PATH.as_posix()],
            "--source-manifest",
        )


def test_exact_output_publication_rejects_broken_link(tmp_path: Path):
    output = tmp_path / "artifact.json"
    output.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        protocol.publish_json_exclusive(output, {"ok": True})


def test_launch_shells_parse_and_stage_has_no_dynamic_global_rebinding():
    for relative in protocol.LAUNCH_CLOSURE:
        subprocess.run(["/bin/bash", "-n", str(ROOT / relative)], check=True)
    stage_source = (ROOT / "athena/stage_product_rational_exact_odt.py").read_text()
    assert "__globals__" not in stage_source
    assert "ODT_SOURCE_OVERRIDES" in stage_source


def test_delayed_stage_and_full_are_bound_to_prestage_and_lane_test_authorities():
    stage_source = (ROOT / "athena/stage_product_rational_exact_odt.py").read_text()
    stage_job = (ROOT / "athena/slurm_product_rational_odt_stage.sbatch").read_text()
    full_job = (ROOT / "athena/slurm_product_rational_odt_full.sbatch").read_text()
    submit = (ROOT / "athena/submit_product_rational_exact_odt.sh").read_text()
    full = (ROOT / "athena/run_product_rational_exact_odt.py").read_text()
    assert stage_source.index("_PREIMPORT_PRESTAGE = _verify_prestage_preimport()") < (
        stage_source.index("from athena.product_rational_exact_odt_protocol import")
    )
    assert protocol.ODT_PRESTAGE_ROOT == Path(
        "/work/joy/x-vla-product-pade-rational-odt-prestage-v3"
    )
    assert protocol.ODT_STAGE_ROOT == Path(
        "/work/joy/x-vla-product-pade-rational-odt-v3"
    )
    assert protocol.PRESTAGE_SCHEMA.endswith("_prestage_manifest_v3")
    assert 'prestage_root=/work/joy/x-vla-product-pade-rational-odt-prestage-v3' in stage_job
    assert 'cd "$prestage_root"' in stage_job
    assert "#SBATCH --partition=low-prio-gpu" in full_job
    assert "#SBATCH --qos=normal" in full_job
    assert "#SBATCH --nodelist=c2-g8-07" in full_job
    assert "#SBATCH --nodelist=c2-g8-05" not in full_job
    assert "#SBATCH --cpus-per-task=16" in full_job
    assert "#SBATCH --mem=600G" in full_job
    assert "#SBATCH --time=24:00:00" in full_job
    assert "#SBATCH --gres=" not in full_job
    assert runner._runtime_environment()["device"] == "cpu"
    assert "lane_tests_job=$(submit_job" in submit
    assert 'afterok:${preflight_job}:${oracle_job}:${lane_tests_job}' in submit
    assert 'elif args.mode == "lane-tests":' in full
    assert "lane_tests = _validated_lane_tests()" in full
    assert "inputs/odt_prestage_manifest.json" in protocol.AUTHENTICATED_INPUTS
    assert "inputs/odt_prestage_ledger.json" in protocol.AUTHENTICATED_INPUTS
    assert "inputs/seed0_deployment_completion.json" in protocol.AUTHENTICATED_INPUTS
    assert "inputs/seed0_precalibration_ema_state.pt" in protocol.AUTHENTICATED_INPUTS
    assert "inputs/smoke_calibration_resume_proof.json" in protocol.AUTHENTICATED_INPUTS
    assert "SCHEMA as CAPABILITY_SCHEMA" in stage_source
    assert "xvla_product_pade_rational_vla_v1" not in stage_source
    assert "_validate_smoke_calibration_resume_proof" in stage_source
    assert "smoke_completion_path = deployment_completion_path" in stage_source
    assert 'proof.get("fresh_deployment_completion_sha256")' in stage_source
    assert '"smoke_resume_proof_hash_joined"' in full


def test_strict_load_float64_conversion_revalidates_guard_before_real_forward():
    source = ChiVLA(capability_protocol.make_product_config(26, deployment=True))
    with torch.no_grad():
        for _, norm in capability_protocol.active_rational_norms(source):
            norm.running_ms.fill_(1.25)
            norm.initialized.fill_(True)
    state = source.state_dict()
    loaded = ChiVLA(capability_protocol.make_product_config(26, deployment=True))
    capability_protocol.strict_load_deployment_state(loaded, state)
    loaded = loaded.to(device="cpu", dtype=torch.float64).eval()
    topology = capability_protocol.validate_product_model(
        loaded, deployment=True, require_initialized_norms=True
    )
    assert topology["running_ms_fail_closed_guard"]["guarded_module_count"] == 74
    image = torch.zeros(
        1,
        3,
        capability_protocol.RESOLUTION,
        capability_protocol.RESOLUTION,
        dtype=torch.float64,
    )
    instruction = torch.zeros(
        1, capability_protocol.MAX_INSTRUCTION_LENGTH, dtype=torch.long
    )
    state_input = torch.zeros(1, capability_protocol.STATE_DIM, dtype=torch.float64)
    embodiment = torch.zeros(1, dtype=torch.long)
    action, _ = loaded(image, instruction, state_input, embodiment)
    assert bool(torch.isfinite(action).all())


def test_exact_lane_modules_have_no_duplicate_top_level_definitions():
    for relative in (
        "athena/product_rational_exact_odt_protocol.py",
        "athena/prepare_product_rational_exact_odt_prestage.py",
        "athena/stage_product_rational_exact_odt.py",
        "athena/run_product_rational_exact_odt.py",
        "xvla/train/direct_odt_truncation.py",
    ):
        tree = ast.parse((ROOT / relative).read_text())
        definitions: dict[str, list[int]] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions.setdefault(node.name, []).append(node.lineno)
        assert {
            name: lines for name, lines in definitions.items() if len(lines) > 1
        } == {}
