#!/usr/bin/env python3
"""Athena certificate for the planted six-role typed VLA benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
from pathlib import Path
from statistics import mean, median

import torch

from xvla.train.generalized_metric_odt import forward_metric, role_environment
from xvla.train.planted_typed_vla import (
    CLAIM_BOUNDARY,
    ROLE_NAMES,
    SemanticCounterfactualPanel,
    TypedCPPolicy,
    analysis_covectors_from_state_directions,
    apply_hidden_gauge,
    balanced_role_odt,
    counterfactual_swap_error,
    dense_coefficient_core,
    dense_forward,
    dense_role_environment_oracle,
    empirical_input_metrics,
    factorized_forward_metric,
    factorized_role_environment,
    gauge_panel,
    intervention_effects,
    make_discovery_panel,
    make_planted_policy,
    make_random_gauges,
    make_semantic_counterfactual_panel,
    map_state_directions_to_canonical,
    action_query_sign_equivariance_error,
    ordered_component_recovery,
    planted_input_metrics,
    subspace_recovery,
    transform_input_metrics,
    transform_inputs,
)


DTYPE = torch.float64
SEEDS = (0, 1, 2, 3, 4)
ROLE_DIM = 12
OUTPUT_DIM = 7
EVALUATION_PAIRS = 30
HAAR_CONTROLS = 32
SOURCE_PATHS = (
    "xvla/train/generalized_metric_odt.py",
    "xvla/train/planted_typed_vla.py",
    "tests/test_generalized_metric_odt.py",
    "tests/test_planted_typed_vla.py",
    "athena/run_planted_typed_vla.py",
    "athena/PLANTED_TYPED_VLA_V1_PROTOCOL.md",
    "athena/slurm_planted_typed_vla.sbatch",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    return {path: sha256_file(root / path) for path in SOURCE_PATHS}


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)
    return float((torch.linalg.vector_norm(actual - expected) / denominator).item())


def make_tiny_oracle_policy(seed: int) -> TypedCPPolicy:
    generator = torch.Generator().manual_seed(seed)
    factors = tuple(
        torch.linalg.qr(
            torch.randn(2, 2, generator=generator, dtype=DTYPE), mode="reduced"
        ).Q
        for _ in ROLE_NAMES
    )
    output = torch.randn(3, 2, generator=generator, dtype=DTYPE)
    return TypedCPPolicy(
        factors,
        output,
        mechanism_names=("oracle_0", "oracle_1"),
    )


def tiny_dense_oracle() -> dict[str, object]:
    policy = make_tiny_oracle_policy(90210)
    generator = torch.Generator().manual_seed(90211)
    inputs = tuple(
        torch.randn(23, dimension, generator=generator, dtype=DTYPE)
        for dimension in policy.role_dims
    )
    metrics = []
    for dimension in policy.role_dims:
        factor = torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
        metrics.append(factor @ factor.T + 0.3 * torch.eye(dimension, dtype=DTYPE))
    output_factor = torch.randn(
        policy.output_dim, policy.output_dim, generator=generator, dtype=DTYPE
    )
    downstream = output_factor @ output_factor.T
    dense_core = dense_coefficient_core(policy)
    forward_errors = {
        "function_relative_error": relative_error(policy(inputs), dense_forward(policy, inputs)),
        "metric_relative_error": relative_error(
            factorized_forward_metric(policy, metrics), forward_metric(dense_core, metrics)
        ),
    }
    role_errors = []
    for role in range(len(ROLE_NAMES)):
        actual = factorized_role_environment(policy, downstream, metrics, role)
        explicit = dense_role_environment_oracle(policy, downstream, metrics, role)
        generic = role_environment(dense_core, downstream, metrics, role)
        role_errors.append(
            {
                "role": ROLE_NAMES[role],
                "explicit_relative_error": relative_error(actual, explicit),
                "generic_relative_error": relative_error(actual, generic),
            }
        )
    return {**forward_errors, "roles": role_errors}


def energy_suppression(effects: dict[str, torch.Tensor]) -> float:
    baseline = effects["baseline_target_energy"].clamp_min(torch.finfo(DTYPE).tiny)
    remaining = effects["intervened_target_energy"]
    return float((1 - remaining / baseline).mean().item())


def off_target_fraction(effects: dict[str, torch.Tensor]) -> float:
    baseline_energy = effects["baseline"].square().sum(1).clamp_min(
        torch.finfo(DTYPE).tiny
    )
    return float((effects["off_target_energy"] / baseline_energy).mean().item())


def gauged_semantic_panel(
    canonical: SemanticCounterfactualPanel,
    gauges,
    *,
    namespace: str,
) -> SemanticCounterfactualPanel:
    return SemanticCounterfactualPanel(
        panel=gauge_panel(canonical.panel, gauges, namespace=namespace),
        pair_ids=canonical.pair_ids,
        instruction_labels=canonical.instruction_labels,
        active_components=canonical.active_components,
    )


def role_controls(
    *,
    policy: TypedCPPolicy,
    gauged_policy: TypedCPPolicy,
    gauges,
    gauged_metrics: tuple[torch.Tensor, ...],
    semantic: SemanticCounterfactualPanel,
    role: int,
    seed: int,
    recovered_state: torch.Tensor,
    recovered_analysis: torch.Tensor,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    rank = policy.rank
    gauge = gauges.matrices[role]
    mapped_state = map_state_directions_to_canonical(recovered_state, gauge)
    selected_effects = intervention_effects(
        gauged_policy,
        semantic,
        role=role,
        upstream_metric=gauged_metrics[role],
        intervention_factor=recovered_analysis,
    )

    covariance_values, covariance_vectors = torch.linalg.eigh(gauged_metrics[role])
    pca_state = covariance_vectors[:, torch.argsort(covariance_values, descending=True)[:rank]]
    pca_analysis = analysis_covectors_from_state_directions(
        gauged_metrics[role], pca_state
    )
    pca_effects = intervention_effects(
        gauged_policy,
        semantic,
        role=role,
        upstream_metric=gauged_metrics[role],
        intervention_factor=pca_analysis,
    )

    environment = factorized_role_environment(
        gauged_policy,
        torch.eye(OUTPUT_DIM, dtype=DTYPE),
        gauged_metrics,
        role,
    )
    environment_values, environment_vectors = torch.linalg.eigh(environment)
    output_only_state = environment_vectors[:, torch.argsort(environment_values, descending=True)[:rank]]
    output_only = analysis_covectors_from_state_directions(
        gauged_metrics[role], output_only_state
    )
    output_only_effects = intervention_effects(
        gauged_policy,
        semantic,
        role=role,
        upstream_metric=gauged_metrics[role],
        intervention_factor=output_only,
    )

    generator = torch.Generator().manual_seed(1_000_000 + 1000 * seed + role)
    haar_suppression = []
    haar_ordered = []
    ambient_suppression = []
    for _ in range(HAAR_CONTROLS):
        rotation = torch.linalg.qr(
            torch.randn(rank, rank, generator=generator, dtype=DTYPE), mode="reduced"
        ).Q
        rotated = recovered_analysis @ rotation
        rotated_effects = intervention_effects(
            gauged_policy,
            semantic,
            role=role,
            upstream_metric=gauged_metrics[role],
            intervention_factor=rotated,
        )
        haar_suppression.append(energy_suppression(rotated_effects))
        haar_ordered.append(
            ordered_component_recovery(gauged_policy.role_factors[role], rotated)
        )
        ambient = torch.linalg.qr(
            torch.randn(ROLE_DIM, rank, generator=generator, dtype=DTYPE), mode="reduced"
        ).Q
        ambient_effects = intervention_effects(
            gauged_policy,
            semantic,
            role=role,
            upstream_metric=gauged_metrics[role],
            intervention_factor=ambient,
        )
        ambient_suppression.append(energy_suppression(ambient_effects))

    mapped_pca = map_state_directions_to_canonical(pca_state, gauge)
    canonical_covariance = torch.linalg.solve(
        gauge, torch.linalg.solve(gauge, gauged_metrics[role]).T
    ).T
    canonical_values, canonical_vectors = torch.linalg.eigh(
        0.5 * (canonical_covariance + canonical_covariance.T)
    )
    canonical_pca = canonical_vectors[
        :, torch.argsort(canonical_values, descending=True)[:rank]
    ]
    output_only_as_state = map_state_directions_to_canonical(output_only_state, gauge)
    result = {
        "role": ROLE_NAMES[role],
        "balanced_subspace_recovery": subspace_recovery(
            policy.role_factors[role], mapped_state
        ),
        "balanced_ordered_recovery": ordered_component_recovery(
            policy.role_factors[role], mapped_state
        ),
        "analysis_covector_subspace_recovery": subspace_recovery(
            gauged_policy.role_factors[role], recovered_analysis
        ),
        "selected_target_energy_suppression": energy_suppression(selected_effects),
        "selected_off_target_energy_fraction": off_target_fraction(selected_effects),
        "pca_subspace_recovery": subspace_recovery(
            policy.role_factors[role], mapped_pca
        ),
        "canonical_pca_subspace_recovery": subspace_recovery(
            policy.role_factors[role], canonical_pca
        ),
        "pca_target_energy_suppression": energy_suppression(pca_effects),
        "output_only_naive_state_subspace_recovery": subspace_recovery(
            policy.role_factors[role], output_only_as_state
        ),
        "output_only_target_energy_suppression": energy_suppression(output_only_effects),
        "same_support_haar_subspace_recovery": subspace_recovery(
            recovered_analysis, recovered_analysis @ torch.linalg.qr(
                torch.randn(rank, rank, generator=generator, dtype=DTYPE), mode="reduced"
            ).Q
        ),
        "same_support_haar_target_suppression_median": median(haar_suppression),
        "same_support_haar_target_suppression_maximum": max(haar_suppression),
        "same_support_haar_ordered_recovery_median": median(haar_ordered),
        "ambient_random_target_suppression_median": median(ambient_suppression),
        "control_draws": HAAR_CONTROLS,
    }
    arrays = {
        f"role_{role}_gauge": gauge,
        f"role_{role}_upstream_metric": gauged_metrics[role],
        f"role_{role}_balanced_state_directions": recovered_state,
        f"role_{role}_recovered_analysis_covectors": recovered_analysis,
        f"role_{role}_planted_factor": gauged_policy.role_factors[role],
    }
    return result, arrays


def run_seed(seed: int) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    policy = make_planted_policy(seed, role_dim=ROLE_DIM, output_dim=OUTPUT_DIM, dtype=DTYPE)
    canonical_metrics = planted_input_metrics(
        policy, signal_variance=1.0, nuisance_variance=9.0
    )
    discovery = make_discovery_panel(
        policy, canonical_metrics, namespace=f"discovery-s{seed}"
    )
    empirical_metrics = empirical_input_metrics(discovery)
    covariance_error = max(
        relative_error(actual, expected)
        for actual, expected in zip(empirical_metrics, canonical_metrics)
    )
    semantic = make_semantic_counterfactual_panel(
        policy,
        20_000 + seed,
        pairs=EVALUATION_PAIRS,
        namespace=f"evaluation-s{seed}",
    )
    if not set(discovery.sample_ids).isdisjoint(semantic.panel.sample_ids):
        raise RuntimeError("discovery and evaluation sample identities overlap")

    gauges = make_random_gauges(30_000 + seed, policy.role_dims, dtype=DTYPE)
    gauged_policy = apply_hidden_gauge(policy, gauges)
    gauged_metrics = transform_input_metrics(empirical_metrics, gauges)
    transformed_discovery = transform_inputs(discovery.role_inputs, gauges)
    transformed_semantic = gauged_semantic_panel(
        semantic, gauges, namespace=f"evaluation-gauged-s{seed}"
    )
    function_error = relative_error(
        gauged_policy(transformed_discovery), policy(discovery.role_inputs)
    )
    semantic_function_error = relative_error(
        gauged_policy(transformed_semantic.panel.role_inputs),
        policy(semantic.panel.role_inputs),
    )
    counterfactual_error = float(
        counterfactual_swap_error(gauged_policy, transformed_semantic).max().item()
    )
    sign_equivariance_error = float(
        action_query_sign_equivariance_error(gauged_policy, transformed_semantic).max().item()
    )
    downstream = torch.eye(OUTPUT_DIM, dtype=DTYPE)
    role_rows = []
    arrays: dict[str, torch.Tensor] = {
        "output_factor": policy.output_factor,
    }
    environment_transport_errors = []
    for role in range(len(ROLE_NAMES)):
        canonical_environment = factorized_role_environment(
            policy, downstream, empirical_metrics, role
        )
        gauged_environment = factorized_role_environment(
            gauged_policy, downstream, gauged_metrics, role
        )
        inverse = torch.linalg.inv(gauges.matrices[role])
        expected_environment = inverse.T @ canonical_environment @ inverse
        environment_transport_errors.append(
            relative_error(gauged_environment, expected_environment)
        )
        system = balanced_role_odt(
            gauged_policy,
            downstream,
            gauged_metrics,
            role,
            support_rtol=1e-13,
        )
        if system.eigenvalues.shape[0] != ROLE_DIM:
            raise RuntimeError("balanced ODT unexpectedly dropped positive covariance support")
        state = system.state_directions[:, : policy.rank]
        analysis = analysis_covectors_from_state_directions(
            gauged_metrics[role], state
        )
        row, role_arrays = role_controls(
            policy=policy,
            gauged_policy=gauged_policy,
            gauges=gauges,
            gauged_metrics=gauged_metrics,
            semantic=transformed_semantic,
            role=role,
            seed=seed,
            recovered_state=state,
            recovered_analysis=analysis,
        )
        row["balanced_eigenvalues"] = system.eigenvalues.tolist()
        row["environment_gauge_transport_relative_error"] = environment_transport_errors[-1]
        role_rows.append(row)
        arrays.update(role_arrays)

    return (
        {
            "seed": seed,
            "discovery_namespace": discovery.namespace,
            "evaluation_namespace": semantic.panel.namespace,
            "discovery_samples": len(discovery.sample_ids),
            "evaluation_pairs": EVALUATION_PAIRS,
            "discovery_evaluation_ids_disjoint": True,
            "discovery_covariance_max_relative_error": covariance_error,
            "gauge_function_relative_error": function_error,
            "gauge_semantic_function_relative_error": semantic_function_error,
            "counterfactual_swap_max_l2_error": counterfactual_error,
            "action_query_sign_equivariance_max_l2_error": sign_equivariance_error,
            "gauge_condition_numbers": [
                float(torch.linalg.cond(matrix).item()) for matrix in gauges.matrices
            ],
            "role_results": role_rows,
        },
        arrays,
    )


def save_tensor_archive(path: Path, arrays: dict[str, torch.Tensor]) -> str:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{name: value.detach().cpu().numpy() for name, value in sorted(arrays.items())},
    )
    with np.load(path, allow_pickle=False) as restored:
        if set(restored.files) != set(arrays):
            raise RuntimeError("tensor archive keys changed during persistence")
        for name, value in arrays.items():
            expected = value.detach().cpu().numpy()
            if not np.array_equal(restored[name], expected):
                raise RuntimeError(f"tensor archive changed {name} during persistence")
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("athena/results/planted_typed_vla_v1.json"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("athena/artifacts/planted_typed_vla_v1"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    start_hashes = source_hashes(root)
    dense = tiny_dense_oracle()
    rows = []
    artifact_records = []
    for seed in SEEDS:
        row, arrays = run_seed(seed)
        artifact_path = args.artifact_dir / f"planted_typed_vla_v1_s{seed}.npz"
        artifact_records.append(
            {
                "seed": seed,
                "path": str(artifact_path),
                "sha256": save_tensor_archive(artifact_path, arrays),
            }
        )
        rows.append(row)
        print(json.dumps({"seed_complete": seed, "summary": row}, sort_keys=True), flush=True)

    role_rows = [role for row in rows for role in row["role_results"]]
    gates = {
        "dense_function_relative_error_below_1e_11": dense["function_relative_error"] < 1e-11,
        "dense_metric_relative_error_below_1e_11": dense["metric_relative_error"] < 1e-11,
        "all_dense_role_environment_errors_below_1e_11": all(
            max(role["explicit_relative_error"], role["generic_relative_error"]) < 1e-11
            for role in dense["roles"]
        ),
        "all_discovery_covariance_errors_below_1e_11": all(
            row["discovery_covariance_max_relative_error"] < 1e-11 for row in rows
        ),
        "all_gauge_function_errors_below_1e_10": all(
            max(row["gauge_function_relative_error"], row["gauge_semantic_function_relative_error"]) < 1e-10
            for row in rows
        ),
        "all_environment_transport_errors_below_1e_10": all(
            role["environment_gauge_transport_relative_error"] < 1e-10 for role in role_rows
        ),
        "all_planted_subspace_recovery_at_least_0p95": all(
            role["balanced_subspace_recovery"] >= 0.95 for role in role_rows
        ),
        "all_ordered_component_recovery_at_least_0p95": all(
            role["balanced_ordered_recovery"] >= 0.95 for role in role_rows
        ),
        "all_counterfactual_swap_errors_below_1e_10": all(
            row["counterfactual_swap_max_l2_error"] < 1e-10 for row in rows
        ),
        "all_action_query_sign_equivariance_errors_below_1e_10": all(
            row["action_query_sign_equivariance_max_l2_error"] < 1e-10 for row in rows
        ),
        "all_targeted_suppression_at_least_0p99": all(
            role["selected_target_energy_suppression"] >= 0.99 for role in role_rows
        ),
        "all_off_target_energy_fractions_below_1e_10": all(
            role["selected_off_target_energy_fraction"] < 1e-10 for role in role_rows
        ),
        "selected_beats_same_support_haar_median_every_role": all(
            role["selected_target_energy_suppression"]
            > role["same_support_haar_target_suppression_median"]
            for role in role_rows
        ),
        "canonical_activation_pca_is_null_support_control_every_role": all(
            role["canonical_pca_subspace_recovery"] < 0.05 for role in role_rows
        ),
        "balanced_beats_gauged_activation_pca_every_role": all(
            role["balanced_subspace_recovery"] > role["pca_subspace_recovery"]
            for role in role_rows
        ),
        "all_panels_episode_analog_disjoint": all(
            row["discovery_evaluation_ids_disjoint"] for row in rows
        ),
        "five_seed_requirement_met": tuple(row["seed"] for row in rows) == SEEDS,
    }
    gates["all_primary_gates_pass"] = all(gates.values())
    end_hashes = source_hashes(root)
    if end_hashes != start_hashes:
        raise RuntimeError("source closure changed during execution")
    payload = {
        "schema": "xvla-planted-typed-vla-v1",
        "status": "pass" if gates["all_primary_gates_pass"] else "fail",
        "frozen_configuration": {
            "seeds": list(SEEDS),
            "role_names": list(ROLE_NAMES),
            "role_dim": ROLE_DIM,
            "output_dim": OUTPUT_DIM,
            "rank": 6,
            "evaluation_pairs_per_seed": EVALUATION_PAIRS,
            "same_support_haar_controls_per_role": HAAR_CONTROLS,
            "dtype": str(DTYPE),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "xvla_python_tree_sha256": os.environ.get("XVLA_SOURCE_TREE_SHA256"),
            "pip_freeze_sha256": os.environ.get("XVLA_PIP_FREEZE_SHA256"),
        },
        "dense_order_seven_oracle": dense,
        "seed_results": rows,
        "aggregate": {
            "minimum_balanced_subspace_recovery": min(
                role["balanced_subspace_recovery"] for role in role_rows
            ),
            "minimum_balanced_ordered_recovery": min(
                role["balanced_ordered_recovery"] for role in role_rows
            ),
            "minimum_selected_target_suppression": min(
                role["selected_target_energy_suppression"] for role in role_rows
            ),
            "maximum_selected_off_target_fraction": max(
                role["selected_off_target_energy_fraction"] for role in role_rows
            ),
            "mean_same_support_haar_median_suppression": mean(
                role["same_support_haar_target_suppression_median"] for role in role_rows
            ),
            "mean_pca_subspace_recovery": mean(
                role["pca_subspace_recovery"] for role in role_rows
            ),
        },
        "gates": gates,
        "artifacts": artifact_records,
        "source_hashes": start_hashes,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": payload["status"], "gates": gates}, sort_keys=True))
    if not gates["all_primary_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
