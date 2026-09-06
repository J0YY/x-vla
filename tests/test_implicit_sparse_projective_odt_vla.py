import torch

from xvla.models.vla import ChiVLA, VLAConfig
from xvla.nn.product_routing import ProductRoutingHead
from xvla.nn.normalization import RationalNorm
from xvla.train.implicit_sparse_projective_odt import (
    ImplicitProjectiveDAG,
    MaterializationTelemetry,
    PhysicalSourceSpec,
    _Builder,
    _validate_network,
    canonicalize_implicit_dag_direct_rq,
    canonicalize_with_explicit_clone_step_trace,
    compare_shared_and_explicit_clone_environments,
    diagonalize_implicit_dag_full_rank,
    diagonalize_shared_and_explicit_clone_independently,
    evaluate_boundary_quotient,
    implicit_shape_statistics,
    reverse_implicit_environments,
)
import xvla.train.implicit_sparse_projective_odt_vla as vla_odt
from xvla.train.implicit_sparse_projective_odt_vla import (
    PRODUCT_COMPONENTS_OBSERVABLE,
    PRODUCT_FIXED_ROUTE_OBSERVABLE,
    assert_full_vla_norm_buffers_unchanged,
    compile_full_vla_projective_dag,
    evaluate_full_vla_observable,
    evaluate_full_vla_quotient,
    full_vla_layout,
    full_vla_structure_statistics,
    patchize_images,
    physical_batch_from_model_inputs,
    physical_source_mapping,
    source_full_vla_observable,
    source_full_vla_output,
    unpatchize_images,
    validate_physical_batch,
)


DTYPE = torch.float64


def _tiny_full_vla(
    seed: int = 17,
    embodiments: int = 2,
    action_head: str = "linear",
) -> ChiVLA:
    torch.manual_seed(seed)
    model = ChiVLA(
        VLAConfig(
            image_size=2,
            patch_size=1,
            vit_dim=2,
            vit_layers=2,
            vit_heads=1,
            vit_ffn_rank=3,
            vocab_size=3,
            max_instr_len=2,
            state_dim=2,
            n_embodiments=embodiments,
            dim=2,
            n_layers=2,
            n_heads=1,
            ffn_rank=3,
            attn="bilinear",
            vit_attn="bilinear",
            ffn="bilinear",
            norm="rational",
            qk_norm="rational",
            residual=True,
            vit_residual=True,
            action_horizon=2,
            action_dim=2,
            action_head=action_head,
            head_rank=3,
            n_factors=2,
        )
    ).to(dtype=DTYPE).eval()
    with torch.no_grad():
        for index, module in enumerate(
            item for item in model.modules() if isinstance(item, RationalNorm)
        ):
            module.running_ms.fill_(0.79 + 0.031 * index)
            module.initialized.fill_(True)
            module.frozen = True
    return model


def test_patch_round_trip_matches_conv2d_order_exactly():
    model = _tiny_full_vla()
    images = torch.arange(2 * 3 * 2 * 2, dtype=DTYPE).reshape(2, 3, 2, 2) / 13.0
    patches = patchize_images(model, images)
    assert patches.shape == (2, 4, 3)
    assert torch.equal(unpatchize_images(model, patches), images)

    with torch.no_grad():
        projected_from_patches = patches @ model.vision.patch.weight.reshape(2, 3).T
        projected_from_patches = projected_from_patches + model.vision.patch.bias
        projected_from_conv = model.vision.patch(images).flatten(2).transpose(1, 2)
    assert float((projected_from_patches - projected_from_conv).abs().max()) < 1e-15


def test_heterogeneous_source_round_trip_uses_unchanged_chivla_forward():
    model = _tiny_full_vla()
    images = torch.linspace(-0.31, 0.37, 2 * 3 * 2 * 2, dtype=DTYPE).reshape(2, 3, 2, 2)
    instruction_ids = torch.tensor(((0, 2), (2, 1)), dtype=torch.long)
    state = torch.tensor(((0.13, -0.27), (-0.19, 0.41)), dtype=DTYPE)
    embodiment_ids = torch.tensor((0, 1), dtype=torch.long)
    physical = physical_batch_from_model_inputs(
        model, images, instruction_ids, state, embodiment_ids
    )
    validate_physical_batch(model, physical)
    direct, _ = model(images, instruction_ids, state, embodiment_ids)
    replay = source_full_vla_output(model, physical)
    assert torch.equal(replay, direct)

    layout = full_vla_layout(model)
    assert layout.physical_source_count == 8
    assert layout.joint_token_count == 11
    assert layout.action_start == 9
    assert layout.observable_width == 4
    assert physical.patches.shape[-1] == 3
    assert physical.instruction_one_hot.shape[-1] == 3
    assert physical.state.shape[-1] == 2
    assert physical.embodiment_one_hot.shape[-1] == 2


def test_categorical_source_oracle_fails_closed_outside_one_hot_domain():
    model = _tiny_full_vla()
    images = torch.zeros(1, 3, 2, 2, dtype=DTYPE)
    physical = physical_batch_from_model_inputs(
        model,
        images,
        torch.tensor(((0, 1),), dtype=torch.long),
        torch.zeros(1, 2, dtype=DTYPE),
        torch.tensor((1,), dtype=torch.long),
    )
    bad = physical.instruction_one_hot.clone()
    bad[0, 0] = torch.tensor((0.5, 0.5, 0.0), dtype=DTYPE)
    invalid = type(physical)(physical.patches, bad, physical.state, physical.embodiment_one_hot)
    try:
        validate_physical_batch(model, invalid)
    except ValueError as error:
        assert "exactly binary" in str(error)
    else:
        raise AssertionError("non-categorical instruction coordinates were silently accepted")


def test_layout_rejects_a_changed_deployed_topology():
    model = _tiny_full_vla()
    model.cfg.action_head = "product"
    try:
        full_vla_layout(model)
    except ValueError as error:
        assert "product action-head configuration differs" in str(error)
    else:
        raise AssertionError("changed action-head topology was silently accepted")


def test_singleton_embodiment_is_a_checkpoint_constant_not_a_physical_leaf():
    model = _tiny_full_vla(embodiments=1)
    physical = physical_batch_from_model_inputs(
        model,
        torch.zeros(1, 3, 2, 2, dtype=DTYPE),
        torch.tensor(((0, 1),), dtype=torch.long),
        torch.zeros(1, 2, dtype=DTYPE),
        torch.tensor((0,), dtype=torch.long),
    )
    layout = full_vla_layout(model)
    mapping = physical_source_mapping(model, physical)
    assert layout.embodiment_is_fixed
    assert layout.physical_source_count == 7
    assert len(mapping) == 7
    assert "embodiment" not in mapping


def test_tiny_full_policy_is_one_exact_shared_dag_with_heterogeneous_leaves():
    model = _tiny_full_vla(embodiments=1)
    images = torch.linspace(-0.23, 0.29, 2 * 3 * 2 * 2, dtype=DTYPE).reshape(2, 3, 2, 2)
    instruction_ids = torch.tensor(((0, 2), (2, 1)), dtype=torch.long)
    state = torch.tensor(((0.17, -0.31), (-0.11, 0.37)), dtype=DTYPE)
    embodiment_ids = torch.zeros(2, dtype=torch.long)
    physical = physical_batch_from_model_inputs(
        model, images, instruction_ids, state, embodiment_ids
    )
    oracle = compile_full_vla_projective_dag(model)
    compiled = evaluate_full_vla_quotient(oracle, physical)
    source = source_full_vla_output(model, physical)
    relative = float(
        ((compiled - source).norm() / source.norm().clamp_min(torch.finfo(DTYPE).tiny)).item()
    )
    assert relative < 3e-9
    assert compiled.shape == source.shape == (2, 2, 2)

    stats = full_vla_structure_statistics(oracle)
    assert stats["physical_source_count"] == stats["physical_leaf_count"] == 7
    assert stats["physical_source_widths"] == {
        "image.patch0": 3,
        "image.patch1": 3,
        "image.patch2": 3,
        "image.patch3": 3,
        "instruction.token0": 3,
        "instruction.token1": 3,
        "state": 2,
    }
    assert stats["root_projective_width"] == 5
    assert stats["active_pade_sites"] == 25
    assert oracle.network.claim_boundary
    assert_full_vla_norm_buffers_unchanged(oracle)


def _product_physical_batch(model: ChiVLA):
    images = torch.linspace(-0.19, 0.33, 2 * 3 * 2 * 2, dtype=DTYPE).reshape(
        2, 3, 2, 2
    )
    instruction_ids = torch.tensor(((0, 2), (1, 0)), dtype=torch.long)
    state = torch.tensor(((0.21, -0.29), (-0.15, 0.35)), dtype=DTYPE)
    embodiment_ids = torch.zeros(2, dtype=torch.long)
    return physical_batch_from_model_inputs(
        model, images, instruction_ids, state, embodiment_ids
    )


def _minimal_product_vla(seed: int = 29) -> ChiVLA:
    torch.manual_seed(seed)
    model = ChiVLA(
        VLAConfig(
            image_size=1,
            patch_size=1,
            vit_dim=2,
            vit_layers=1,
            vit_heads=1,
            vit_ffn_rank=3,
            vocab_size=2,
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
            action_horizon=2,
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
            module.running_ms.fill_(0.83 + 0.023 * index)
            module.initialized.fill_(True)
            module.frozen = True
    return model


def test_product_head_component_root_replays_every_expert_gate_and_public_decode():
    model = _tiny_full_vla(embodiments=1, action_head="product")
    physical = _product_physical_batch(model)
    oracle = compile_full_vla_projective_dag(model)

    compiled_components = evaluate_full_vla_observable(oracle, physical)
    source_components = source_full_vla_observable(oracle, physical)
    component_relative = float(
        (
            (compiled_components - source_components).norm()
            / source_components.norm().clamp_min(torch.finfo(DTYPE).tiny)
        ).item()
    )
    compiled_actions = evaluate_full_vla_quotient(oracle, physical)
    source_actions = source_full_vla_output(model, physical)
    action_relative = float(
        (
            (compiled_actions - source_actions).norm()
            / source_actions.norm().clamp_min(torch.finfo(DTYPE).tiny)
        ).item()
    )

    assert component_relative < 3e-9
    assert action_relative < 3e-9
    assert oracle.observable_kind == PRODUCT_COMPONENTS_OBSERVABLE
    assert oracle.product_signs is None
    assert compiled_components.shape == (2, 14)
    assert oracle.network.root.output_dimension == 15
    assert "every BilinearFFN center, factor, and gate" in oracle.network.claim_boundary
    assert_full_vla_norm_buffers_unchanged(oracle)


def test_product_head_fixed_external_sign_route_is_one_action_root():
    model = _tiny_full_vla(embodiments=1, action_head="product")
    physical = _product_physical_batch(model)
    signs = torch.tensor((-1.0, 1.0), dtype=DTYPE)
    oracle = compile_full_vla_projective_dag(model, product_signs=signs)
    compiled = evaluate_full_vla_quotient(oracle, physical)
    source = source_full_vla_observable(oracle, physical).reshape(2, 2, 2)
    relative = float(
        ((compiled - source).norm() / source.norm().clamp_min(torch.finfo(DTYPE).tiny)).item()
    )

    assert relative < 3e-9
    assert oracle.observable_kind == PRODUCT_FIXED_ROUTE_OBSERVABLE
    assert oracle.product_signs == (-1.0, 1.0)
    assert oracle.observable_euclidean_width == 4
    assert oracle.network.root.output_dimension == 5
    assert compiled.shape == (2, 2, 2)
    assert "externally fixed sign route" in oracle.network.claim_boundary


def test_product_head_fixed_route_rejects_non_sign_coordinates():
    model = _tiny_full_vla(embodiments=1, action_head="product")
    try:
        compile_full_vla_projective_dag(model, product_signs=(1.0, 0.0))
    except ValueError as error:
        assert "exact -1 or +1" in str(error)
    else:
        raise AssertionError("non-sign product route was silently accepted")


def test_minimal_product_shared_dag_runs_direct_algorithms1_to3_with_step_replay():
    model = _minimal_product_vla()
    physical = physical_batch_from_model_inputs(
        model,
        torch.tensor([[[[0.13]], [[-0.17]], [[0.29]]]], dtype=DTYPE),
        torch.tensor(((1,),), dtype=torch.long),
        torch.tensor(((0.31,),), dtype=DTYPE),
        torch.zeros(1, dtype=torch.long),
    )
    oracle = compile_full_vla_projective_dag(model)
    raw = physical_source_mapping(model, physical)
    canonical = canonicalize_implicit_dag_direct_rq(
        oracle.network, replay_inputs=raw, block_size=64
    )
    assert max(
        step.per_step_function_replay_error for step in canonical.steps
    ) < 3e-8
    assert all(
        step.parent_occurrences_pushed == step.expected_parent_occurrences
        for step in canonical.steps
    )

    shape = implicit_shape_statistics(canonical.network)
    environments = reverse_implicit_environments(canonical.network)
    assert sum(record.child_messages_emitted for record in environments) == shape[
        "edge_occurrences"
    ]
    diagonal = diagonalize_implicit_dag_full_rank(
        canonical.network, replay_inputs=raw
    )
    assert max(
        diagonal.replay_relative_error,
        diagonal.maximum_recontracted_offdiagonal_ratio,
    ) < 3e-8
    assert (
        diagonal.pushed_parent_occurrences
        == diagonal.expected_parent_occurrences
        == shape["edge_occurrences"] + 1
    )
    streamed = diagonalize_implicit_dag_full_rank(
        canonical.network,
        replay_inputs=raw,
        stream_pre_evd_environments=True,
        retain_eigenvalues=False,
        retain_post_evd_environments=False,
    )
    assert not streamed.eigenvalue_spectra_retained
    assert streamed.algorithm2_child_messages == shape["edge_occurrences"]
    assert streamed.diagonalized_node_count == shape["unique_nodes"]
    assert max(
        streamed.replay_relative_error,
        streamed.maximum_recontracted_offdiagonal_ratio,
    ) < 3e-8


def test_product_head_subgraph_matches_no_memo_clone_at_every_canonical_step():
    torch.manual_seed(41)
    head = ProductRoutingHead(
        dim=1, action_dim=1, horizon=1, n_factors=1, rank=1
    ).to(dtype=DTYPE).eval()
    like = head.center.down.weight
    builder = _Builder(like, MaterializationTelemetry())
    pooled = builder.physical_pair(
        PhysicalSourceSpec("pooled", "input.product_pooled_state", 1)
    )
    center = vla_odt._bilinear_ffn_output_pair(
        builder, pooled, head.center, "oracle.product.center"
    )
    factor = vla_odt._bilinear_ffn_output_pair(
        builder, pooled, head.factors[0], "oracle.product.factor0"
    )
    gate = vla_odt._bilinear_ffn_output_pair(
        builder, pooled, head.gates[0], "oracle.product.gate0"
    )
    root = vla_odt._balanced_pair_concat(
        builder,
        (center, factor, gate),
        "oracle.product.component_concat",
    )
    network = ImplicitProjectiveDAG(
        root=root,
        head=torch.eye(4, dtype=DTYPE),
        head_binary_exponent=0,
        token_count=1,
        feature_dimension=1,
        selected_token=0,
        mask=torch.ones(1, 1, dtype=DTYPE),
        claim_boundary="Bounded pooled ProductRoutingHead clone oracle only.",
        physical_sources=builder.physical_source_specs,
    )
    _validate_network(network)
    raw = {"pooled": torch.tensor([[0.37], [-0.23]], dtype=DTYPE)}
    observed = evaluate_boundary_quotient(network, raw)
    with torch.no_grad():
        source_center, source_factors, source_gates = head(raw["pooled"])
        source = torch.cat(
            (
                source_center,
                source_factors.reshape(raw["pooled"].shape[0], -1),
                source_gates,
            ),
            dim=-1,
        )
    assert float(
        ((observed - source).norm() / source.norm().clamp_min(torch.finfo(DTYPE).tiny)).item()
    ) < 3e-12

    trace = canonicalize_with_explicit_clone_step_trace(
        network, raw, block_size=32
    )
    for record in trace.records:
        assert record.factor_relative_error < 3e-10
        assert record.q_core_relative_error < 3e-10
        assert record.transformed_parent_occurrence_relative_error < 3e-10
        assert record.shared_function_replay_error < 3e-10
        assert record.clone_function_replay_error < 3e-10
        assert record.shared_clone_function_relative_error < 3e-10
        assert record.supported_reconstruction_relative_error < 3e-10
        assert record.boundary_head_relative_error < 3e-10
        assert (
            record.literal_rq_occurrences_compared
            + record.null_gauge_occurrences_checked_by_reconstruction
            > 0
        )
        assert (
            record.expected_shared_parent_occurrences
            == record.pushed_shared_parent_occurrences
        )
        assert (
            record.expected_clone_parent_occurrences
            == record.pushed_clone_parent_occurrences
        )

    environments = compare_shared_and_explicit_clone_environments(
        trace.shared_network, trace.explicit_clone_network
    )
    assert environments["maximum_aggregate_relative_error"] < 3e-10
    diagonal = diagonalize_shared_and_explicit_clone_independently(
        trace.shared_network, trace.explicit_clone_network, raw
    )
    assert diagonal.pre_evd_aggregate_environment_relative_error < 3e-10
    assert diagonal.eigenvalue_relative_error < 3e-10
    assert max(
        diagonal.shared_replay_relative_error,
        diagonal.clone_replay_relative_error,
        diagonal.shared_clone_relative_error,
        diagonal.maximum_shared_aggregate_offdiagonal_ratio,
        diagonal.maximum_clone_aggregate_offdiagonal_ratio,
    ) < 3e-10
    assert (
        diagonal.pushed_shared_parent_occurrences
        == diagonal.expected_shared_parent_occurrences
    )
    assert (
        diagonal.pushed_clone_parent_occurrences
        == diagonal.expected_clone_parent_occurrences
    )

    broken = canonicalize_implicit_dag_direct_rq(
        network,
        replay_inputs=raw,
        block_size=32,
        omit_parent_push=("oracle.product.center", 0),
    )
    assert max(
        step.per_step_function_replay_error for step in broken.steps
    ) > 1e-6
