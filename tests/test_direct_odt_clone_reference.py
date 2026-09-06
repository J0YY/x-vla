from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import xvla.train.implicit_sparse_projective_odt as implicit_odt
from xvla.nn.block import ChiTransformerBlock
from xvla.nn.normalization import RationalNorm
from xvla.nn.product_routing import ProductRoutingHead
from xvla.train.direct_odt_clone_reference import (
    canonicalize_clone_reference_direct_rq,
    compare_dense_occurrence_networks,
    compare_production_algorithm1_steps,
    compare_production_eigenvalues,
    contract_and_aggregate_clone_environments,
    contract_clone_environments,
    diagonalize_shared_and_explicit_clone_independently as diagonalize_clone_reference,
    physical_relative_error,
    projective_relative_error,
    reference_evaluate_homogeneous_scaled,
    reference_evaluate_projective,
    reference_evaluate_quotient,
    ReferenceScaledMatrix,
    _sum_scaled_messages,
    scaled_batch_relative_error,
    scaled_matrix_relative_error,
    unfold_occurrences_no_memo,
)
from xvla.train.implicit_sparse_projective_odt import (
    ImplicitNode,
    ImplicitProjectiveDAG,
    MaterializationTelemetry,
    PhysicalSourceSpec,
    ScaledMatrix,
    _Builder,
    _add_scaled_matrices,
    _add_pair_vectors,
    _affine_pair,
    _pade_norm_pair,
    _scale_pair_vector,
    canonicalize_implicit_dag_direct_rq,
    compile_implicit_projective_block_boundary,
    diagonalize_implicit_dag_full_rank,
    reverse_implicit_environments,
)
from xvla.train.implicit_sparse_projective_odt_vla import (
    _balanced_pair_concat,
    _bilinear_ffn_output_pair,
)


DTYPE = torch.float64
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SOURCE = PROJECT_ROOT / "xvla/train/direct_odt_clone_reference.py"


def _assert_algorithm1_differential(
    network: ImplicitProjectiveDAG,
    raw: torch.Tensor | dict[str, torch.Tensor],
    *,
    tolerance: float,
    block_size: int = 32,
):
    initial_tree = unfold_occurrences_no_memo(network)
    initial_homogeneous = reference_evaluate_homogeneous_scaled(initial_tree, raw)
    initial_projective = reference_evaluate_projective(initial_tree, raw)

    reference = canonicalize_clone_reference_direct_rq(network, raw)
    production_snapshots: list[dict[str, object]] = []

    def capture_step(work_network, canonicalized_node, factor, record) -> None:
        snapshot = unfold_occurrences_no_memo(work_network)
        production_snapshots.append(
            {
                "node_uid": canonicalized_node.uid,
                "factor": ReferenceScaledMatrix(
                    factor.mantissa.clone(), int(factor.binary_exponent)
                ),
                "record": record,
                "network": snapshot,
                "homogeneous": reference_evaluate_homogeneous_scaled(snapshot, raw),
                "projective": reference_evaluate_projective(snapshot, raw),
            }
        )

    try:
        production = canonicalize_implicit_dag_direct_rq(
            network, block_size=block_size, step_callback=capture_step
        )
    except RuntimeError as error:
        next_label = (
            reference.steps[len(production_snapshots)].label
            if len(production_snapshots) < len(reference.steps)
            else "after-final-step"
        )
        raise RuntimeError(
            f"production Algorithm 1 failed before reference step {next_label!r}"
        ) from error
    comparisons = compare_production_algorithm1_steps(production, reference)

    assert comparisons
    assert (
        len(comparisons)
        == len(reference.steps)
        == len(production.steps)
        == len(production_snapshots)
    )
    for comparison, reference_step, production_snapshot in zip(
        comparisons, reference.steps, production_snapshots
    ):
        assert comparison.step == reference_step.step
        assert comparison.origin_uid == reference_step.origin_uid
        assert comparison.label == reference_step.label
        assert comparison.maximum_q_core_relative_error < tolerance
        assert comparison.production_factorization_relative_error < tolerance
        assert comparison.reference_factorization_relative_error < tolerance
        assert comparison.production_pushes_complete
        assert comparison.reference_pushes_complete
        assert reference_step.occurrence_count >= 1
        assert reference_step.completed_factor_pushes == reference_step.expected_factor_pushes
        assert reference_step.omitted_factor_pushes == 0
        assert reference_step.projective_replay_relative_error < tolerance
        assert production_snapshot["node_uid"] == reference_step.origin_uid
        production_record = production_snapshot["record"]
        assert production_record.step == reference_step.step
        assert production_record.uid == reference_step.origin_uid
        production_factor = production_snapshot["factor"]
        for reference_factor in reference_step.factors:
            assert (
                scaled_matrix_relative_error(production_factor, reference_factor)
                < tolerance
            )
        network_comparison = compare_dense_occurrence_networks(
            production_snapshot["network"], reference_step.network_snapshot
        )
        assert network_comparison.maximum_core_relative_error < tolerance
        assert network_comparison.head_relative_error < tolerance
        assert (
            scaled_batch_relative_error(
                production_snapshot["homogeneous"],
                reference_step.homogeneous_output,
            )
            < tolerance
        )
        assert (
            projective_relative_error(
                production_snapshot["projective"],
                reference_step.projective_coordinates,
            )
            < tolerance
        )
        assert (
            projective_relative_error(
                reference_step.projective_coordinates, initial_projective
            )
            < tolerance
        )

    production_tree = unfold_occurrences_no_memo(production.network)
    production_homogeneous = reference_evaluate_homogeneous_scaled(
        production_tree, raw
    )
    reference_homogeneous = reference_evaluate_homogeneous_scaled(
        reference.network, raw
    )
    assert scaled_batch_relative_error(reference_homogeneous, initial_homogeneous) < tolerance
    assert scaled_batch_relative_error(production_homogeneous, initial_homogeneous) < tolerance
    assert scaled_batch_relative_error(production_homogeneous, reference_homogeneous) < tolerance
    assert (
        projective_relative_error(
            reference_evaluate_projective(production_tree, raw),
            reference_evaluate_projective(reference.network, raw),
        )
        < tolerance
    )
    return production, reference


def _assert_algorithms2_and3_differential(
    production,
    reference,
    raw: torch.Tensor | dict[str, torch.Tensor],
    *,
    tolerance: float,
):
    aggregates = contract_and_aggregate_clone_environments(reference.network)
    assert set(aggregates) == {uid for uid, _ in reference.origin_schedule}
    production_environments = {
        record.uid: ReferenceScaledMatrix(
            record.mantissa, int(record.binary_exponent)
        )
        for record in reverse_implicit_environments(production.network)
    }
    assert set(production_environments) == set(aggregates)
    for origin_uid, expected_environment in aggregates.items():
        assert (
            scaled_matrix_relative_error(
                production_environments[origin_uid], expected_environment
            )
            < tolerance
        )

    reference_diagonal = diagonalize_clone_reference(reference, raw)
    production_diagonal = diagonalize_implicit_dag_full_rank(production.network)
    eigenvalue_errors = compare_production_eigenvalues(
        production_diagonal, reference_diagonal, reference.origin_schedule
    )
    assert set(eigenvalue_errors) == set(aggregates)
    assert max(eigenvalue_errors.values(), default=0.0) < tolerance

    assert reference_diagonal.replay_relative_error < tolerance
    assert reference_diagonal.maximum_recontracted_offdiagonal_ratio < tolerance
    assert (
        reference_diagonal.completed_gauge_pushes
        == reference_diagonal.expected_gauge_pushes
    )
    assert reference_diagonal.omitted_gauge_pushes == 0
    assert production_diagonal.replay_relative_error == 0.0
    assert production_diagonal.maximum_recontracted_offdiagonal_ratio < tolerance
    assert (
        production_diagonal.pushed_parent_occurrences
        == production_diagonal.expected_parent_occurrences
    )

    production_tree = unfold_occurrences_no_memo(production_diagonal.network)
    production_pair = reference_evaluate_projective(production_tree, raw)
    assert (
        projective_relative_error(
            production_pair, reference_diagonal.final_projective_coordinates
        )
        < tolerance
    )
    assert (
        projective_relative_error(
            production_pair, reference.initial_projective_coordinates
        )
        < tolerance
    )
    return production_diagonal, reference_diagonal


def _six_pade_block(seed: int = 11):
    torch.manual_seed(seed)
    block = ChiTransformerBlock(
        dim=1,
        n_heads=1,
        ffn_rank=1,
        n_layers=1,
        causal=True,
        norm="rational",
        qk_norm="rational",
        residual=True,
    ).double().eval()
    generator = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                module.weight.copy_(
                    0.13
                    + 0.09
                    * torch.rand(
                        module.weight.shape, generator=generator, dtype=DTYPE
                    )
                )
                if module.bias is not None:
                    module.bias.copy_(
                        -0.04
                        + 0.08
                        * torch.rand(
                            module.bias.shape, generator=generator, dtype=DTYPE
                        )
                    )
        for index, module in enumerate(
            item for item in block.modules() if isinstance(item, RationalNorm)
        ):
            module.running_ms.fill_(0.83 + 0.07 * index)
            module.initialized.fill_(True)
            module.frozen = True
    positions = torch.tensor(((-0.03,), (0.05,)), dtype=DTYPE)
    oracle = compile_implicit_projective_block_boundary(block, positions)
    raw = torch.tensor(
        (((-0.31,), (0.17,)), ((0.29,), (-0.11,)), ((0.0,), (0.0,))),
        dtype=DTYPE,
    )
    return oracle.network, raw


def _heterogeneous_terminal_policy(seed: int = 23):
    torch.manual_seed(seed)
    like = torch.empty((), dtype=DTYPE)
    builder = _Builder(like, MaterializationTelemetry())
    vision = builder.physical_pair(
        PhysicalSourceSpec("vision", "input.vision_boundary", 1)
    )
    instruction = builder.physical_pair(
        PhysicalSourceSpec("instruction", "input.instruction_boundary", 1)
    )
    state = builder.physical_pair(
        PhysicalSourceSpec("state", "input.state_boundary", 1)
    )

    visual_hidden = _affine_pair(
        builder,
        vision,
        torch.tensor(((1.23,), (-0.21,)), dtype=DTYPE),
        torch.tensor((0.08, -0.03), dtype=DTYPE),
        "cross_boundary.visual_projection",
    )
    language_hidden = _affine_pair(
        builder,
        instruction,
        torch.tensor(((-0.17,), (1.17,)), dtype=DTYPE),
        torch.tensor((0.02, 0.05), dtype=DTYPE),
        "cross_boundary.instruction_projection",
    )
    state_hidden = _affine_pair(
        builder,
        state,
        torch.tensor(((1.31,), (0.47,)), dtype=DTYPE),
        torch.tensor((-0.06, 0.01), dtype=DTYPE),
        "cross_boundary.state_projection",
    )
    fused = _add_pair_vectors(
        builder, visual_hidden, language_hidden, "cross_boundary.vision_language"
    )
    fused = _add_pair_vectors(
        builder, fused, state_hidden, "cross_boundary.state_fusion"
    )
    query0 = _affine_pair(
        builder,
        fused,
        torch.tensor(((1.21, -0.12), (0.19, 0.62)), dtype=DTYPE),
        torch.tensor((0.03, -0.02), dtype=DTYPE),
        "policy.query0",
    )
    final_norm = RationalNorm(variant="pade").to(dtype=DTYPE).eval()
    final_norm.running_ms.fill_(0.91)
    final_norm.initialized.fill_(True)
    final_norm.frozen = True
    normalized0 = _pade_norm_pair(
        builder, query0, final_norm, "policy.final_norm.action0"
    )
    pooled = _scale_pair_vector(
        builder,
        normalized0,
        1.0,
        "policy.product.action_query_mean",
    )

    head = ProductRoutingHead(
        dim=2, action_dim=1, horizon=1, n_factors=1, rank=2
    ).to(dtype=DTYPE).eval()
    head_generator = torch.Generator().manual_seed(4000 + seed)
    with torch.no_grad():
        for module in head.modules():
            if isinstance(module, nn.Linear):
                module.weight.copy_(
                    1.09
                    + 0.16
                    * torch.rand(
                        module.weight.shape,
                        generator=head_generator,
                        dtype=DTYPE,
                    )
                )
                if module.bias is not None:
                    module.bias.copy_(
                        -0.03
                        + 0.06
                        * torch.rand(
                            module.bias.shape,
                            generator=head_generator,
                            dtype=DTYPE,
                        )
                    )
    center = _bilinear_ffn_output_pair(
        builder, pooled, head.center, "policy.product.center"
    )
    factor = _bilinear_ffn_output_pair(
        builder, pooled, head.factors[0], "policy.product.factor0"
    )
    gate = _bilinear_ffn_output_pair(
        builder, pooled, head.gates[0], "policy.product.gate0"
    )
    root = _balanced_pair_concat(
        builder,
        (center, factor, gate),
        "policy.product.component_concat",
    )
    network = ImplicitProjectiveDAG(
        root=root,
        head=torch.eye(4, dtype=DTYPE),
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=1,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=DTYPE),
        claim_boundary=(
            "Bounded heterogeneous vision, instruction, and state boundary through "
            "final action normalization, action-query pooling, and product head."
        ),
        physical_sources=builder.physical_source_specs,
    )
    raw = {
        "vision": torch.tensor(((-0.28,), (0.16,)), dtype=DTYPE),
        "instruction": torch.tensor(((0.37,), (-0.19,)), dtype=DTYPE),
        "state": torch.tensor(((0.11,), (0.41,)), dtype=DTYPE),
    }
    return network, raw


def _rank_deficient_shared_network(seed: int = 31):
    generator = torch.Generator().manual_seed(seed)
    like = torch.empty((), dtype=DTYPE)
    builder = _Builder(like, MaterializationTelemetry())
    x = builder.physical_pair(PhysicalSourceSpec("x", "input.x", 1))
    y = builder.physical_pair(PhysicalSourceSpec("y", "input.y", 1))

    left = torch.zeros(4, 2, dtype=DTYPE)
    right = torch.zeros(4, 2, dtype=DTYPE)
    for atom in range(4):
        left[atom, atom // 2] = 1.0
        right[atom, atom % 2] = 1.0
    coefficient = 0.25 + torch.rand(3, generator=generator, dtype=DTYPE)
    output = torch.zeros(3, 4, dtype=DTYPE)
    output[0, 0] = coefficient[0]
    output[0, 1] = coefficient[1]
    output[0, 2] = coefficient[1]
    output[0, 3] = coefficient[2]
    output[1] = output[0]
    output[2, 3] = 1.0
    shared = builder.cp(
        "rank_deficient.shared_child",
        output,
        left,
        right,
        (x, x),
        kind="bounded_exact_dependent_shared_child",
    )
    assert isinstance(shared, ImplicitNode)

    repeated_output = torch.tensor(
        ((0.61, -0.27, 0.0), (0.0, 0.0, 1.0)), dtype=DTYPE
    )
    repeated_factors = torch.eye(3, dtype=DTYPE)
    repeated = builder.cp(
        "adversary.repeated_both_roles",
        repeated_output,
        repeated_factors,
        repeated_factors,
        (shared, shared),
        kind="shared_child_in_both_binary_roles",
    )
    projected = builder.unary(
        "adversary.distinct_parent",
        torch.tensor(((0.38, 0.14, 0.09), (0.0, 0.0, 1.0)), dtype=DTYPE),
        shared,
        kind="second_distinct_parent_of_shared_child",
    )
    second_branch = _add_pair_vectors(
        builder, projected, y, "adversary.distinct_parent_plus_y"
    )
    root = _add_pair_vectors(
        builder, repeated, second_branch, "adversary.observable_sum"
    )
    assert isinstance(root, ImplicitNode)
    network = ImplicitProjectiveDAG(
        root=root,
        head=torch.eye(2, dtype=DTYPE),
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=1,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=DTYPE),
        physical_sources=builder.physical_source_specs,
    )
    raw = {
        "x": torch.tensor(((-0.41,), (0.23,), (0.57,)), dtype=DTYPE),
        "y": torch.tensor(((0.17,), (-0.29,), (0.08,)), dtype=DTYPE),
    }
    return network, raw


def _seeded_shared_dag(seed: int):
    generator = torch.Generator().manual_seed(8000 + seed)
    like = torch.empty((), dtype=DTYPE)
    builder = _Builder(like, MaterializationTelemetry())
    x = builder.physical_pair(PhysicalSourceSpec("x", "family.x", 1))
    y = builder.physical_pair(PhysicalSourceSpec("y", "family.y", 1))
    affine = torch.tensor(
        (
            (
                0.55 + 0.2 * torch.rand((), generator=generator, dtype=DTYPE),
                -0.12 + 0.24 * torch.rand((), generator=generator, dtype=DTYPE),
            ),
            (0.0, 1.0),
        ),
        dtype=DTYPE,
    )
    shared = builder.unary(
        f"family{seed}.shared", affine, x, kind="seeded_shared_affine"
    )
    factors = torch.eye(2, dtype=DTYPE)
    square = builder.cp(
        f"family{seed}.shared_in_both_roles",
        torch.eye(2, dtype=DTYPE),
        factors,
        factors,
        (shared, shared),
        kind="seeded_repeated_binary_child",
    )
    projected = builder.unary(
        f"family{seed}.second_parent",
        torch.tensor(
            (
                (
                    0.31 + 0.2 * torch.rand((), generator=generator, dtype=DTYPE),
                    -0.07,
                ),
                (0.0, 1.0),
            ),
            dtype=DTYPE,
        ),
        shared,
        kind="seeded_distinct_parent",
    )
    mixed = _add_pair_vectors(builder, projected, y, f"family{seed}.mix_y")
    root = _add_pair_vectors(builder, square, mixed, f"family{seed}.root")
    assert isinstance(root, ImplicitNode)
    network = ImplicitProjectiveDAG(
        root=root,
        head=torch.tensor(((1.0, 0.13), (-0.09, 1.0)), dtype=DTYPE),
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=1,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=DTYPE),
        physical_sources=builder.physical_source_specs,
    )
    raw = {
        "x": torch.tensor(((-0.33,), (0.21,), (0.49,)), dtype=DTYPE),
        "y": torch.tensor(((0.14,), (-0.26,), (0.37,)), dtype=DTYPE),
    }
    return network, raw, shared.uid


def _forced_streamed_shared_network():
    like = torch.empty((), dtype=DTYPE)
    builder = _Builder(like, MaterializationTelemetry())
    x = builder.physical_pair(PhysicalSourceSpec("x", "forced_stream.x", 5))

    left = torch.zeros(36, 6, dtype=DTYPE)
    right = torch.zeros(36, 6, dtype=DTYPE)
    for atom in range(36):
        left[atom, atom // 6] = 1.0
        right[atom, atom % 6] = 1.0
    output = torch.empty(5, 36, dtype=DTYPE)
    for row in range(5):
        for column in range(36):
            residue = ((row + 2) * (column + 3)) % 17
            output[row, column] = 0.0125 * (residue - 8)
    output[:, :5] += torch.tensor(
        (
            (1.25, 0.12, -0.08, 0.05, 0.03),
            (0.17, 1.125, 0.09, -0.06, 0.04),
            (-0.11, 0.14, 0.875, 0.07, -0.05),
            (0.06, -0.10, 0.16, 1.0625, 0.13),
            (0.09, 0.04, -0.07, 0.12, 1.0),
        ),
        dtype=DTYPE,
    )
    # Keep the homogeneous coordinate safely away from zero while every row
    # still overlaps every streamed panel.  The dense, nonorthogonal panels
    # make TSQR suffix order and transpose mistakes observable to the clone.
    output[4, 35] += 2.0
    shared = builder.cp(
        "forced_stream.shared_cp",
        output,
        left,
        right,
        (x, x),
        kind="full_rank_forced_streamed_shared_cp",
    )
    assert isinstance(shared, ImplicitNode)

    repeated = _add_pair_vectors(
        builder,
        shared,
        shared,
        "forced_stream.repeated_both_roles",
    )
    projected = _affine_pair(
        builder,
        shared,
        torch.tensor(
            (
                (1.1, 0.08, 0.0, -0.03),
                (-0.04, 0.95, 0.06, 0.0),
                (0.0, -0.07, 1.05, 0.05),
                (0.02, 0.0, -0.05, 0.9),
            ),
            dtype=DTYPE,
        ),
        torch.tensor((0.03, -0.02, 0.01, 0.04), dtype=DTYPE),
        "forced_stream.distinct_parent",
    )
    root = _add_pair_vectors(
        builder,
        repeated,
        projected,
        "forced_stream.root",
    )
    assert isinstance(root, ImplicitNode)
    network = ImplicitProjectiveDAG(
        root=root,
        head=torch.eye(5, dtype=DTYPE),
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=5,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=DTYPE),
        physical_sources=builder.physical_source_specs,
    )
    raw = {
        "x": torch.tensor(
            (
                (-0.31, 0.17, 0.23, -0.09, 0.11),
                (0.19, -0.27, 0.05, 0.32, -0.14),
                (0.41, 0.08, -0.22, 0.13, 0.29),
            ),
            dtype=DTYPE,
        )
    }
    return network, raw, shared.uid


def _bounded_rank_deficient_385x386_shared_network(
    *, consumer_count: int = 3
):
    """Reproduce the production failure shape with an exact null R pivot."""

    like = torch.empty((), dtype=DTYPE)
    builder = _Builder(like, MaterializationTelemetry())
    raw_left = builder.physical_pair(
        PhysicalSourceSpec("left", "bounded_rank_deficient.left", 192)
    )
    right_value = builder.physical_pair(
        PhysicalSourceSpec("right", "bounded_rank_deficient.right", 1)
    )

    rank = 385
    left_projection = torch.zeros(rank, 193, dtype=DTYPE)
    right = torch.zeros(rank, 2, dtype=DTYPE)
    # The first 383 CP atoms are distinct.  Atom 383 duplicates atom 0,
    # creating an exact null R pivot, while atom 384 selects the homogeneous
    # input column at flattened index 385.  This preserves a nonzero quotient
    # denominator and matches the failing production input geometry (193, 2).
    flattened_columns = (*range(383), 0, 385)
    for atom, column in enumerate(flattened_columns):
        left_projection[atom, column // 2] = 1.0
        right[atom, column % 2] = 1.0
    left_value = builder.unary(
        "bounded_rank_deficient.project_left_to_385",
        left_projection,
        raw_left,
        kind="production_path_tall_projective_input",
    )
    assert isinstance(left_value, ImplicitNode)

    output = torch.eye(385, dtype=DTYPE)
    shared = builder.cp(
        "joint.block0.pre_attention.token0.projective_output",
        output,
        torch.eye(rank, dtype=DTYPE),
        right,
        (left_value, right_value),
        kind="sparse_projective_pade_vector_output",
    )
    assert isinstance(shared, ImplicitNode)

    projections = []
    for index in range(consumer_count):
        coordinate = index % 3
        scale = (1.0, -0.75, 0.625)[coordinate]
        bias = (0.03, -0.02, 0.01)[coordinate]
        weight = torch.zeros(1, 384, dtype=DTYPE)
        weight[0, coordinate] = scale
        projected = _affine_pair(
            builder,
            shared,
            weight,
            torch.tensor((bias,), dtype=DTYPE),
            f"bounded_rank_deficient.parent{index}",
        )
        assert isinstance(projected, ImplicitNode)
        projections.append(projected)
    root = projections[0]
    for index, projected in enumerate(projections[1:], start=1):
        root = _add_pair_vectors(
            builder,
            root,
            projected,
            f"bounded_rank_deficient.sum{index}",
        )
    assert isinstance(root, ImplicitNode)
    network = ImplicitProjectiveDAG(
        root=root,
        head=torch.eye(2, dtype=DTYPE),
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=1,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=DTYPE),
        physical_sources=builder.physical_source_specs,
    )
    left_sample = torch.linspace(-0.31, 0.29, 192, dtype=DTYPE)
    raw = {
        "left": torch.stack((left_sample, left_sample.flip(0))),
        "right": torch.tensor(((-0.27,), (0.19,)), dtype=DTYPE),
    }
    return network, raw, shared.uid


def _shared_edge_occurrences(root: ImplicitNode) -> int:
    completed: set[int] = set()
    active: set[int] = set()
    total = 0

    def visit(node: ImplicitNode) -> None:
        nonlocal total
        identity = id(node)
        if identity in active:
            raise ValueError("test graph contains a cycle")
        if identity in completed:
            return
        active.add(identity)
        total += len(node.children)
        for child in node.children:
            visit(child)
        active.remove(identity)
        completed.add(identity)

    visit(root)
    return total


def test_reference_source_is_independent_and_stage_restricted():
    tree = ast.parse(REFERENCE_SOURCE.read_text(), filename=str(REFERENCE_SOURCE))
    imported_from_production: set[str] = set()
    numerical_calls: list[tuple[str, str]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.functions: list[str] = []

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module == "xvla.train.implicit_sparse_projective_odt":
                imported_from_production.update(alias.name for alias in node.names)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute):
                terminal = node.func.attr
                if terminal in {"qr", "eigh"}:
                    numerical_calls.append(
                        (self.functions[-1] if self.functions else "<module>", terminal)
                    )
            self.generic_visit(node)

    Visitor().visit(tree)
    assert imported_from_production == {
        "CPBinaryCore",
        "DenseCloneCore",
        "ImplicitNode",
        "ImplicitProjectiveDAG",
        "ReducedQRBinaryCore",
        "UnaryCore",
    }
    assert numerical_calls == [
        ("_independent_dense_clone_rq", "qr"),
        ("diagonalize_shared_and_explicit_clone_independently", "eigh"),
    ]
    with pytest.raises(ValueError, match="coordinate matrices"):
        projective_relative_error(
            torch.ones(2, 1, dtype=DTYPE), torch.ones(2, 1, dtype=DTYPE)
        )


def test_unchanged_six_pade_attention_ffn_residual_block_matches_every_step():
    network, raw = _six_pade_block()
    production, reference = _assert_algorithm1_differential(
        network, raw, tolerance=4e-8
    )
    assert len(reference.steps) > 20
    assert any(step.occurrence_count > 1 for step in reference.steps)
    _assert_algorithms2_and3_differential(
        production, reference, raw, tolerance=5e-8
    )


def test_heterogeneous_final_norm_pooling_and_product_head_matches_every_step():
    network, raw = _heterogeneous_terminal_policy()
    production, reference = _assert_algorithm1_differential(
        network, raw, tolerance=5e-8
    )
    labels = {step.label for step in reference.steps}
    assert any("policy.final_norm" in label for label in labels)
    assert "policy.product.action_query_mean" in labels
    assert "policy.product.center" in labels
    assert "policy.product.factor0" in labels
    assert "policy.product.gate0" in labels
    assert len(network.physical_sources) == 3
    _assert_algorithms2_and3_differential(
        production, reference, raw, tolerance=7e-8
    )


def test_rank_deficient_shared_child_matches_and_missing_occurrences_fail():
    network, raw = _rank_deficient_shared_network()
    production, reference = _assert_algorithm1_differential(
        network, raw, tolerance=3e-10
    )
    shared_step = next(
        step
        for step in reference.steps
        if step.label == "rank_deficient.shared_child"
    )
    assert shared_step.occurrence_count == 3
    assert not shared_step.literal_q_chart_resolved
    assert shared_step.unfolding_shape == (3, 4)
    _, reference_diagonal = _assert_algorithms2_and3_differential(
        production, reference, raw, tolerance=2e-9
    )

    broken_reference_r = canonicalize_clone_reference_direct_rq(
        network,
        raw,
        omit_factor_occurrence=("rank_deficient.shared_child", 0),
    )
    assert broken_reference_r.steps[-1].projective_replay_relative_error > 1e-6
    assert sum(step.omitted_factor_pushes for step in broken_reference_r.steps) == 1

    broken_production_r = canonicalize_implicit_dag_direct_rq(
        network,
        block_size=32,
        omit_parent_push=("rank_deficient.shared_child", 0),
    )
    broken_production_tree = unfold_occurrences_no_memo(
        broken_production_r.network
    )
    assert (
        projective_relative_error(
            reference_evaluate_projective(broken_production_tree, raw),
            reference.initial_projective_coordinates,
        )
        > 1e-6
    )

    broken_gauge = diagonalize_clone_reference(
        reference,
        raw,
        omit_gauge_occurrence=("rank_deficient.shared_child", 0),
    )
    assert broken_gauge.omitted_gauge_pushes == 1
    assert (
        broken_gauge.completed_gauge_pushes
        == broken_gauge.expected_gauge_pushes - 1
    )
    assert broken_gauge.replay_relative_error > 1e-6
    assert reference_diagonal.omitted_gauge_pushes == 0


@pytest.mark.parametrize("seed", range(4))
def test_seeded_adversarial_shared_dag_family(seed: int):
    network, raw, shared_uid = _seeded_shared_dag(seed)
    production, reference = _assert_algorithm1_differential(
        network, raw, tolerance=2e-10
    )
    shared_step = next(
        step for step in reference.steps if step.origin_uid == shared_uid
    )
    assert shared_step.occurrence_count == 3
    production_diagonal, reference_diagonal = (
        _assert_algorithms2_and3_differential(
            production, reference, raw, tolerance=2e-9
        )
    )
    production_actions = reference_evaluate_quotient(
        unfold_occurrences_no_memo(production_diagonal.network), raw
    )
    reference_actions = reference_evaluate_quotient(
        reference_diagonal.network, raw
    )
    assert physical_relative_error(production_actions, reference_actions) < 2e-9


def test_forced_streamed_shared_cp_matches_independent_clone_after_every_step(
    monkeypatch: pytest.MonkeyPatch,
):
    target_unfolding_elements = 5 * 36
    monkeypatch.setattr(
        implicit_odt,
        "MAXIMUM_BOUNDED_EXPLICIT_UNFOLDING_ELEMENTS",
        target_unfolding_elements - 1,
    )
    monkeypatch.setattr(
        implicit_odt,
        "MAXIMUM_BOUNDED_EXPLICIT_Q_ELEMENTS",
        target_unfolding_elements - 1,
    )
    for block_size, expected_stages in ((5, 8), (7, 6), (11, 4)):
        assert 36 % block_size
        assert expected_stages == (36 + block_size - 1) // block_size
        network, raw, shared_uid = _forced_streamed_shared_network()
        production, reference = _assert_algorithm1_differential(
            network,
            raw,
            tolerance=3e-10,
            block_size=block_size,
        )

        streamed_steps = [
            step
            for step in production.steps
            if step.method == implicit_odt.DIRECT_RQ_METHOD
        ]
        assert [step.uid for step in streamed_steps] == [shared_uid]
        production_step = streamed_steps[0]
        reference_step = next(
            step for step in reference.steps if step.origin_uid == shared_uid
        )
        assert (
            production_step.label
            == reference_step.label
            == "forced_stream.shared_cp"
        )
        assert production_step.unfolding_shape == reference_step.unfolding_shape == (
            5,
            36,
        )
        assert production_step.streamed_column_blocks == expected_stages
        assert production_step.direct_q_provenance_method == (
            implicit_odt.STREAMED_DIRECT_Q_PROVENANCE_METHOD
        )
        assert production_step.direct_q_provenance_verified
        assert production_step.direct_q_provenance_stage_count == expected_stages
        assert (
            production_step.direct_q_provenance_column_blocks_compared
            == expected_stages
        )
        assert production_step.direct_q_columns_compared == 36
        assert production_step.direct_q_retained_transition_elements == (
            (expected_stages - 1) * 5 * 5
        )
        assert production_step.direct_q_conditioning_accepted
        assert production_step.direct_q_minimum_pivot_to_maximum_entry > (
            production_step.direct_q_conditioning_threshold
        )
        assert production_step.direct_q_compact_relative_error < 3e-10
        assert production_step.direct_q_reconstruction_relative_error < 3e-10
        assert production_step.local_scaled_reconstruction_relative_error < 3e-10
        assert production_step.local_reconstruction_exponent_delta == 0
        assert production_step.expected_parent_occurrences == 3
        assert production_step.parent_occurrences_pushed == 3
        assert production_step.scale_sensitive_occurrences_verified == 3
        assert reference_step.occurrence_count == 3
        assert reference_step.expected_factor_pushes == 3
        assert reference_step.completed_factor_pushes == 3
        assert reference_step.omitted_factor_pushes == 0
        assert production.telemetry.streamed_direct_q_provenance_certificates == 1

        production_diagonal, reference_diagonal = (
            _assert_algorithms2_and3_differential(
                production,
                reference,
                raw,
                tolerance=2e-9,
            )
        )
        assert production_diagonal.diagonalized_node_count == len(
            reference.origin_schedule
        )
        assert (
            production_diagonal.pushed_parent_occurrences
            == production_diagonal.expected_parent_occurrences
        )
        assert (
            reference_diagonal.completed_gauge_pushes
            == reference_diagonal.expected_gauge_pushes
        )
        production_actions = reference_evaluate_quotient(
            unfold_occurrences_no_memo(production_diagonal.network), raw
        )
        reference_actions = reference_evaluate_quotient(
            reference_diagonal.network, raw
        )
        assert physical_relative_error(production_actions, reference_actions) < 2e-9


def test_bounded_rank_deficient_385x386_retains_direct_q_and_matches_clone_every_step(
    monkeypatch: pytest.MonkeyPatch,
):
    streamed_shapes: list[tuple[int, int]] = []
    original_streamed = implicit_odt._direct_rq_cp_streamed

    def record_streamed(core, telemetry, *, block_size):
        streamed_shapes.append(
            (core.output_dimension, core.input_dimensions[0] * core.input_dimensions[1])
        )
        return original_streamed(core, telemetry, block_size=block_size)

    monkeypatch.setattr(implicit_odt, "_direct_rq_cp_streamed", record_streamed)
    network, raw, shared_uid = _bounded_rank_deficient_385x386_shared_network()
    source_node = next(
        node for node in implicit_odt._walk_unique(network.root) if node.uid == shared_uid
    )
    assert isinstance(source_node.core, implicit_odt.CPBinaryCore)
    assert source_node.core.input_dimensions == (385, 2)
    assert source_node.core.output_dimension * 385 * 2 == 296_450
    retained_q_elements = 385 * 386
    source_cp_elements = sum(
        factor.numel()
        for factor in (
            source_node.core.output_factor,
            source_node.core.left_factor,
            source_node.core.right_factor,
        )
    )
    assert retained_q_elements == 148_610
    assert source_cp_elements == 297_220
    assert retained_q_elements <= implicit_odt.MAXIMUM_BOUNDED_EXPLICIT_Q_ELEMENTS
    assert implicit_odt._bounded_explicit_q_allowed_by_structure(source_node.core)
    route_inventory = implicit_odt.predict_direct_rq_route_inventory(network)
    assert route_inventory == {
        "schema": "exact_postorder_structural_direct_rq_routes_v1",
        "bounded_explicit_unfolding_element_limit": 4_000_000,
        "bounded_explicit_q_element_limit": 4_000_000,
        "bounded_retained_q_candidate_count": 3,
        "bounded_retained_q_total_elements": 148_626,
        "bounded_retained_q_replaced_cp_total_elements": 223_336,
        "bounded_retained_q_signed_storage_delta_elements": -74_710,
        "bounded_retained_q_positive_storage_delta_elements": 0,
        "bounded_retained_q_maximum_unfolding_elements": 148_610,
        "bounded_retained_q_maximum_elements": 148_610,
        "streamed_cp_candidate_count": 0,
        "streamed_tall_unrepresentable_count": 0,
        "streamed_tall_unrepresentable_shape_counts": {},
        "bounded_retained_q_shape_counts": {"2x4": 2, "385x386": 1},
        "nodes_with_propagated_input_shape_change": 1,
        "predicted_root_output_dimension": 2,
        "simulated_unique_node_count": 9,
    }
    production, reference = _assert_algorithm1_differential(
        network,
        raw,
        tolerance=3e-10,
        block_size=4096,
    )
    production_step = next(step for step in production.steps if step.uid == shared_uid)
    reference_step = next(
        step for step in reference.steps if step.origin_uid == shared_uid
    )
    assert production_step.unfolding_shape == reference_step.unfolding_shape == (
        385,
        386,
    )
    assert production_step.method == implicit_odt.BOUNDED_EXPLICIT_RQ_METHOD
    assert production_step.streamed_column_blocks == 1
    assert not production_step.full_row_rank
    assert not production_step.literal_q_chart_resolved
    assert production_step.minimum_diagonal_to_maximum_entry == 0.0
    assert production_step.direct_q_provenance_method == (
        f"retained_explicit_q::{implicit_odt.BOUNDED_EXPLICIT_RQ_METHOD}"
    )
    assert production_step.direct_q_provenance_verified
    assert production_step.direct_q_provenance_stage_count == 1
    assert production_step.direct_q_columns_compared == 386
    assert production_step.direct_q_compact_relative_error == 0.0
    assert production_step.direct_q_reconstruction_relative_error < 3e-10
    assert production_step.local_scaled_reconstruction_relative_error < 3e-10
    assert production_step.local_reconstruction_exponent_delta == 0
    assert production_step.expected_parent_occurrences == 3
    assert production_step.parent_occurrences_pushed == 3
    assert production_step.scale_sensitive_occurrences_verified == 3
    assert reference_step.occurrence_count == 3
    assert reference_step.expected_factor_pushes == 3
    assert reference_step.completed_factor_pushes == 3
    assert reference_step.omitted_factor_pushes == 0
    assert reference_step.minimum_diagonal_to_maximum_entry == 0.0
    assert not reference_step.literal_q_chart_resolved

    canonical_node = next(
        node
        for node in implicit_odt._walk_unique(production.network.root)
        if node.uid == shared_uid
    )
    assert isinstance(canonical_node.core, implicit_odt.ReducedQRBinaryCore)
    assert canonical_node.core.q_rows.shape == (385, 386)
    assert canonical_node.core.output_dimension == 385

    production_diagonal, reference_diagonal = (
        _assert_algorithms2_and3_differential(
            production,
            reference,
            raw,
            tolerance=2e-9,
        )
    )
    production_actions = reference_evaluate_quotient(
        unfold_occurrences_no_memo(production_diagonal.network), raw
    )
    reference_actions = reference_evaluate_quotient(
        reference_diagonal.network, raw
    )
    assert physical_relative_error(production_actions, reference_actions) < 2e-9

    # Rebuild the exact production occurrence count without retaining the
    # no-memo snapshots of all 60 copies.  The three-occurrence graph above is
    # the full independent step-by-step/A2/A3 oracle; this second lane isolates
    # the production failure's 60-edge absorption ledger and function replay.
    sixty_network, sixty_raw, sixty_shared_uid = (
        _bounded_rank_deficient_385x386_shared_network(consumer_count=60)
    )
    sixty_shared = next(
        node
        for node in implicit_odt._walk_unique(sixty_network.root)
        if node.uid == sixty_shared_uid
    )
    parents = implicit_odt._parent_occurrences(sixty_network.root)
    assert len(parents[id(sixty_shared)]) == 60
    before = implicit_odt.evaluate_projective_boundary(sixty_network, sixty_raw)
    sixty_canonical = canonicalize_implicit_dag_direct_rq(
        sixty_network,
        block_size=4096,
        replay_inputs=sixty_raw,
        replay_each_step=False,
    )
    sixty_step = next(
        step for step in sixty_canonical.steps if step.uid == sixty_shared_uid
    )
    assert sixty_step.method == implicit_odt.BOUNDED_EXPLICIT_RQ_METHOD
    assert sixty_step.unfolding_shape == (385, 386)
    assert not sixty_step.full_row_rank
    assert not sixty_step.literal_q_chart_resolved
    assert sixty_step.expected_parent_occurrences == 60
    assert sixty_step.parent_occurrences_pushed == 60
    assert sixty_step.scale_sensitive_occurrences_verified == 60
    assert sixty_step.maximum_absorption_scaled_relative_error < 3e-10
    assert sixty_step.maximum_absorption_exponent_delta == 0
    assert sixty_canonical.full_projective_replay_evaluations == 2
    assert sixty_canonical.final_projective_replay_error < 3e-10
    after = implicit_odt.evaluate_projective_boundary(
        sixty_canonical.network, sixty_raw
    )
    assert implicit_odt._projective_batch_relative_error(after, before) < 3e-10
    assert streamed_shapes == []

    # The correctness boundary is the declared direct-Q storage bound, not
    # whether retaining Q happens to shrink this particular CP encoding.
    # This attention-concat geometry is bounded but its retained Q is larger
    # than its CP factors, and an exact null row still requires the completion.
    output = torch.eye(97, dtype=DTYPE)
    output[-1].zero_()
    left = torch.zeros(97, 33, dtype=DTYPE)
    right = torch.zeros(97, 65, dtype=DTYPE)
    for atom in range(97):
        left[atom, atom % 33] = 1.0
        right[atom, atom % 65] = 1.0
    attention_concat = implicit_odt.CPBinaryCore(
        output,
        left,
        right,
        "bounded_attention_concat_rank_deficient_control",
    )
    attention_q_elements = 97 * 2_145
    attention_cp_elements = sum(
        factor.numel()
        for factor in (
            attention_concat.output_factor,
            attention_concat.left_factor,
            attention_concat.right_factor,
        )
    )
    assert attention_q_elements == 208_065
    assert attention_cp_elements == 18_915
    assert attention_q_elements > attention_cp_elements
    assert implicit_odt._bounded_explicit_q_allowed_by_structure(attention_concat)
    factor, retained_q, attention_diagnostics = implicit_odt._direct_rq_core(
        attention_concat,
        MaterializationTelemetry(),
        block_size=4_096,
    )
    assert attention_diagnostics.method == implicit_odt.BOUNDED_EXPLICIT_RQ_METHOD
    assert attention_diagnostics.unfolding_shape == (97, 2_145)
    assert not attention_diagnostics.full_row_rank
    assert not attention_diagnostics.literal_q_chart_resolved
    assert isinstance(retained_q, implicit_odt.ReducedQRBinaryCore)
    assert retained_q.q_rows.shape == (97, 2_145)
    attention_error, attention_exponent_delta = (
        implicit_odt._local_direct_rq_scaled_reconstruction_audit(
            attention_concat,
            factor,
            retained_q,
            block_size=4_096,
            telemetry=MaterializationTelemetry(),
            factor_normalization_binary_exponent=(
                attention_diagnostics.factor_normalization_binary_exponent
            ),
        )
    )
    assert attention_error < 3e-10
    assert attention_exponent_delta == 0
    assert streamed_shapes == []


def test_over_bound_large_deficient_unfolding_fails_closed():
    columns = 2_000_001
    output = torch.tensor(((1.0,), (0.0,)), dtype=DTYPE)
    left = torch.zeros(1, columns, dtype=DTYPE)
    left[0, 0] = 1.0
    right = torch.ones(1, 1, dtype=DTYPE)
    core = implicit_odt.CPBinaryCore(
        output,
        left,
        right,
        "over_bound_rank_deficient_control",
    )
    unfolding_elements = 2 * columns
    retained_q_elements = 2 * columns
    assert unfolding_elements == retained_q_elements == 4_000_002
    assert unfolding_elements > implicit_odt.MAXIMUM_BOUNDED_EXPLICIT_Q_ELEMENTS
    assert not implicit_odt._bounded_explicit_q_allowed_by_structure(core)
    meter = MaterializationTelemetry()
    with pytest.raises(
        ValueError,
        match=r"conditioning gate rejected.*no alternate factorization.*shape=2x2000001",
    ):
        implicit_odt._direct_rq_core(core, meter, block_size=columns)
    assert meter.local_direct_rq_factorizations == 1
    assert meter.householder_qr_kernel_calls == 1
    assert meter.bounded_explicit_direct_rq_factorizations == 0
    assert meter.direct_q_provenance_certificates == 0

    # A tall unfolding beyond the same source bound cannot retain its reduced
    # Q either.  It must report the exact shape before any QR call, with no
    # attempt to synthesize an alternative representation.
    tall_rows = 2_000_001
    tall_core = implicit_odt.CPBinaryCore(
        torch.cat(
            (
                torch.ones(1, 1, dtype=DTYPE),
                torch.zeros(tall_rows - 1, 1, dtype=DTYPE),
            ),
            dim=0,
        ),
        torch.ones(1, 2, dtype=DTYPE),
        torch.ones(1, 1, dtype=DTYPE),
        "over_bound_tall_rank_deficient_control",
    )
    assert not implicit_odt._bounded_explicit_q_allowed_by_structure(tall_core)
    tall_meter = MaterializationTelemetry()
    with pytest.raises(
        ValueError,
        match=r"fewer columns than rows.*shape=2000001x2",
    ):
        implicit_odt._direct_rq_core(tall_core, tall_meter, block_size=4_096)
    assert tall_meter.local_direct_rq_factorizations == 0
    assert tall_meter.householder_qr_kernel_calls == 0
    assert tall_meter.bounded_explicit_direct_rq_factorizations == 0
    assert tall_meter.direct_q_provenance_certificates == 0


def test_rank_zero_direct_rq_reconstruction_and_homogeneous_replay():
    like = torch.empty((), dtype=DTYPE)
    builder = _Builder(like, MaterializationTelemetry())
    physical = builder.physical_pair(
        PhysicalSourceSpec("zero", "rank_zero.source", 1)
    )
    zero = builder.unary(
        "rank_zero.branch",
        torch.zeros(2, 2, dtype=DTYPE),
        physical,
        kind="rank_zero_unary_control",
    )
    assert isinstance(zero, ImplicitNode)
    network = ImplicitProjectiveDAG(
        root=zero,
        head=torch.eye(2, dtype=DTYPE),
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=1,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=DTYPE),
        physical_sources=builder.physical_source_specs,
    )
    raw = {"zero": torch.tensor(((-0.3,), (0.2,)), dtype=DTYPE)}

    def raw_homogeneous_boundary(candidate: ImplicitProjectiveDAG):
        memo: dict[int, torch.Tensor] = {}

        def evaluate(node: ImplicitNode):
            cached = memo.get(id(node))
            if cached is not None:
                return cached
            if node.children:
                value = implicit_odt._core_apply(
                    node.core, tuple(evaluate(child) for child in node.children)
                )
            else:
                value = implicit_odt._core_apply(
                    node.core,
                    (
                        torch.cat(
                            (
                                raw[node.physical_source_key],
                                torch.ones(2, 1, dtype=DTYPE),
                            ),
                            dim=1,
                        ),
                    ),
                )
            value = torch.ldexp(
                value,
                torch.tensor(node.core.binary_exponent, dtype=torch.int64),
            )
            memo[id(node)] = value
            return value

        result = evaluate(candidate.root) @ candidate.head.T
        return torch.ldexp(
            result,
            torch.tensor(candidate.head_binary_exponent, dtype=torch.int64),
        )

    before = raw_homogeneous_boundary(network)
    assert not bool((before != 0).any())
    with pytest.raises(ValueError, match="zero/nonfinite coordinates"):
        implicit_odt.evaluate_scaled_boundary(network, raw)

    meter = MaterializationTelemetry()
    factor, direct_q, diagnostics = implicit_odt._direct_rq_core(
        zero.core.clone(), meter, block_size=8
    )
    local_error, exponent_delta = (
        implicit_odt._local_direct_rq_scaled_reconstruction_audit(
            zero.core,
            factor,
            direct_q,
            block_size=8,
            telemetry=meter,
            factor_normalization_binary_exponent=(
                diagnostics.factor_normalization_binary_exponent
            ),
        )
    )
    assert local_error == 0.0
    assert exponent_delta == 0
    corrupted_q = direct_q.clone()
    corrupted_q.binary_exponent = 1
    with pytest.raises(ValueError, match="inconsistent exponent ledger"):
        implicit_odt._local_direct_rq_scaled_reconstruction_audit(
            zero.core,
            factor,
            corrupted_q,
            block_size=8,
            telemetry=MaterializationTelemetry(),
            factor_normalization_binary_exponent=(
                diagnostics.factor_normalization_binary_exponent
            ),
        )

    canonical = canonicalize_implicit_dag_direct_rq(
        network,
        block_size=8,
        replay_each_step=False,
    )
    zero_step = next(step for step in canonical.steps if step.label == "rank_zero.branch")
    assert zero_step.local_scaled_reconstruction_relative_error == 0.0
    assert zero_step.local_reconstruction_exponent_delta == 0
    assert not zero_step.full_row_rank
    assert not zero_step.literal_q_chart_resolved
    after = raw_homogeneous_boundary(canonical.network)
    assert torch.equal(after, before)
    assert not bool((after != 0).any())
    with pytest.raises(ValueError, match="zero/nonfinite coordinates"):
        implicit_odt.evaluate_scaled_boundary(canonical.network, raw)
    with pytest.raises(ValueError, match="zero/nonfinite coordinates"):
        implicit_odt.evaluate_projective_boundary(canonical.network, raw)


def test_streamed_production_algorithms2_and3_match_separated_clone_reference():
    network, raw, _ = _seeded_shared_dag(17)
    production, reference = _assert_algorithm1_differential(
        network, raw, tolerance=2e-10
    )
    reference_diagonal = diagonalize_clone_reference(reference, raw)
    occurrence_environments = contract_clone_environments(reference.network)
    assert sum(
        record.child_messages_emitted for record in occurrence_environments
    ) == len(occurrence_environments) - 1
    aggregates = contract_and_aggregate_clone_environments(reference.network)
    separated_production_environments = {
        record.uid: ReferenceScaledMatrix(
            record.mantissa, int(record.binary_exponent)
        )
        for record in reverse_implicit_environments(production.network)
    }
    assert set(separated_production_environments) == set(aggregates)
    for origin_uid, expected_environment in aggregates.items():
        assert (
            scaled_matrix_relative_error(
                separated_production_environments[origin_uid],
                expected_environment,
            )
            < 2e-9
        )

    streamed = diagonalize_implicit_dag_full_rank(
        production.network,
        stream_pre_evd_environments=True,
        retain_eigenvalues=True,
        retain_post_evd_environments=False,
    )
    root_first_schedule = tuple(reversed(reference.origin_schedule))
    errors = compare_production_eigenvalues(
        streamed, reference_diagonal, root_first_schedule
    )
    assert max(errors.values(), default=0.0) < 2e-9
    assert streamed.algorithm2_child_messages == _shared_edge_occurrences(
        production.network.root
    )
    assert streamed.diagonalized_node_count == len(reference.origin_schedule)
    assert streamed.eigenvalue_spectra_retained
    assert (
        streamed.pushed_parent_occurrences
        == streamed.expected_parent_occurrences
    )
    assert streamed.maximum_recontracted_offdiagonal_ratio < 2e-9

    canonical_tree = unfold_occurrences_no_memo(production.network)
    streamed_tree = unfold_occurrences_no_memo(streamed.network)
    canonical_homogeneous = reference_evaluate_homogeneous_scaled(
        canonical_tree, raw
    )
    streamed_homogeneous = reference_evaluate_homogeneous_scaled(
        streamed_tree, raw
    )
    assert (
        scaled_batch_relative_error(streamed_homogeneous, canonical_homogeneous)
        < 2e-9
    )
    streamed_pair = reference_evaluate_projective(streamed_tree, raw)
    assert (
        projective_relative_error(
            streamed_pair, reference_diagonal.final_projective_coordinates
        )
        < 2e-9
    )
    assert (
        projective_relative_error(
            streamed_pair, reference.initial_projective_coordinates
        )
        < 2e-9
    )


def test_shared_zero_route_environment_uses_only_nonzero_scale_center():
    nonzero = torch.tensor(((0.5, 0.125), (0.125, 0.25)), dtype=DTYPE)
    zero = torch.zeros_like(nonzero)
    production = _add_scaled_matrices(
        (ScaledMatrix(nonzero, -5000), ScaledMatrix(zero, 7000))
    )
    reference = _sum_scaled_messages(
        (
            ReferenceScaledMatrix(nonzero, -5000),
            ReferenceScaledMatrix(zero, 7000),
        )
    )
    production_reference = ReferenceScaledMatrix(
        production.mantissa, int(production.binary_exponent)
    )
    assert reference.binary_exponent == production.binary_exponent == -5000
    assert scaled_matrix_relative_error(production_reference, reference) < 2e-12
    assert bool((reference.mantissa != 0).any())

    production_zero = _add_scaled_matrices((ScaledMatrix(zero, 7000),))
    reference_zero = _sum_scaled_messages((ReferenceScaledMatrix(zero, 7000),))
    assert production_zero.binary_exponent == reference_zero.binary_exponent == 0
    assert not bool((production_zero.mantissa != 0).any())
    assert not bool((reference_zero.mantissa != 0).any())


def test_two_nonzero_shared_route_exponent_gap_fails_closed():
    larger = torch.tensor(((0.5, 0.125), (0.125, 0.25)), dtype=DTYPE)
    smaller = torch.tensor(((0.25, 0.0), (0.0, 0.125)), dtype=DTYPE)
    with pytest.raises(ValueError, match="nonzero.*underflowed"):
        _sum_scaled_messages(
            (
                ReferenceScaledMatrix(larger, 0),
                ReferenceScaledMatrix(smaller, -5000),
            )
        )
    with pytest.raises(ValueError, match="nonzero.*underflowed"):
        _add_scaled_matrices(
            (ScaledMatrix(larger, 0), ScaledMatrix(smaller, -5000))
        )


def test_no_memo_expansion_fails_closed_at_occurrence_limit():
    network, _, _ = _seeded_shared_dag(0)
    with pytest.raises(ValueError, match="occurrence expansion exceeds"):
        unfold_occurrences_no_memo(network, maximum_occurrences=4)
