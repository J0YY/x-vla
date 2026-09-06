"""Exact planted six-role CP policy for validating typed VLA ODT.

The benchmark is deliberately synthetic.  It is a positive control for the
mathematics and intervention protocol, not evidence that a trained VLA has a
human-nameable mechanism.  Its six independent input occurrences are

    instruction, object_a, object_b, robot_state, phase, action_query.

For CP factors ``A_0, ..., A_5`` and output factor ``B``, the policy is

    y = B ((A_0^T x_0) * ... * (A_5^T x_5)).

All metric recursions below contract this one joint coefficient tensor.  The
roles are never tied, even when their ambient dimensions happen to match.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import ascii_lowercase
from typing import Sequence

import torch

from xvla.train.generalized_metric_odt import BalancedMetricODT, balanced_metric_odt


Tensor = torch.Tensor
ROLE_NAMES = (
    "instruction",
    "object_a",
    "object_b",
    "robot_state",
    "phase",
    "action_query",
)
MECHANISM_NAMES = (
    "reach_a",
    "reach_b",
    "grasp_a",
    "grasp_b",
    "transport_a",
    "transport_b",
)
_EINSUM_INPUT_LABELS = tuple(ascii_lowercase[: len(ROLE_NAMES)])


def _real_finite(value: Tensor, name: str) -> Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(f"{name} must be a real floating-point tensor")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _symmetric_psd(value: Tensor, dimension: int, name: str) -> Tensor:
    tensor = _real_finite(value, name)
    if tensor.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    scale = max(float(tensor.abs().max().item()), torch.finfo(tensor.dtype).tiny)
    tolerance = max(
        100.0 * torch.finfo(tensor.dtype).eps * dimension * scale,
        1e-12 * scale,
    )
    if float((tensor - tensor.T).abs().max().item()) > tolerance:
        raise ValueError(f"{name} must be symmetric")
    symmetric = 0.5 * (tensor + tensor.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    if float(eigenvalues[0].item()) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    if float(eigenvalues[0].item()) < 0:
        symmetric = (
            eigenvectors * eigenvalues.clamp_min(0).unsqueeze(0)
        ) @ eigenvectors.T
    return 0.5 * (symmetric + symmetric.T)


@dataclass(frozen=True)
class TypedCPPolicy:
    """A real finite CP map with exactly six typed input occurrences."""

    role_factors: tuple[Tensor, ...]
    output_factor: Tensor
    role_names: tuple[str, ...] = ROLE_NAMES
    mechanism_names: tuple[str, ...] = MECHANISM_NAMES

    def __post_init__(self) -> None:
        if len(self.role_factors) != len(ROLE_NAMES):
            raise ValueError("TypedCPPolicy requires exactly six role factors")
        if tuple(self.role_names) != ROLE_NAMES:
            raise ValueError("role_names must equal the frozen six-role schema")
        output = _real_finite(self.output_factor, "output_factor")
        if output.ndim != 2 or min(output.shape) <= 0:
            raise ValueError("output_factor must be a nonempty matrix")
        rank = output.shape[1]
        if len(self.mechanism_names) != rank or len(set(self.mechanism_names)) != rank:
            raise ValueError("mechanism_names must uniquely name every CP component")
        for index, factor in enumerate(self.role_factors):
            checked = _real_finite(factor, f"role_factors[{index}]")
            if checked.ndim != 2 or checked.shape[1] != rank:
                raise ValueError("every role factor must have shape (dimension, CP rank)")
            if checked.dtype != output.dtype or checked.device != output.device:
                raise ValueError("all factors must share dtype and device")
            if int(torch.linalg.matrix_rank(checked).item()) != rank:
                raise ValueError("every planted role factor must have full column rank")

    @property
    def rank(self) -> int:
        return self.output_factor.shape[1]

    @property
    def output_dim(self) -> int:
        return self.output_factor.shape[0]

    @property
    def role_dims(self) -> tuple[int, ...]:
        return tuple(factor.shape[0] for factor in self.role_factors)

    def __call__(self, role_inputs: Sequence[Tensor]) -> Tensor:
        return factorized_forward(self, role_inputs)


@dataclass(frozen=True)
class GaugeTransform:
    """Invertible state-coordinate maps ``x'_a = G_a x_a``."""

    matrices: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if len(self.matrices) != len(ROLE_NAMES):
            raise ValueError("one gauge matrix is required for every typed role")
        for index, matrix in enumerate(self.matrices):
            checked = _real_finite(matrix, f"matrices[{index}]")
            if checked.ndim != 2 or checked.shape[0] != checked.shape[1]:
                raise ValueError("every gauge must be square")
            if int(torch.linalg.matrix_rank(checked).item()) != checked.shape[0]:
                raise ValueError("every gauge must be invertible")


@dataclass(frozen=True)
class TypedInputPanel:
    """A provenance-bearing collection of six role inputs."""

    namespace: str
    sample_ids: tuple[str, ...]
    role_inputs: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if not self.namespace:
            raise ValueError("namespace must be nonempty")
        if len(self.role_inputs) != len(ROLE_NAMES):
            raise ValueError("panel requires exactly six role tensors")
        if not self.sample_ids or len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be nonempty and unique")
        count = len(self.sample_ids)
        reference = _real_finite(self.role_inputs[0], "role_inputs[0]")
        for index, values in enumerate(self.role_inputs):
            checked = _real_finite(values, f"role_inputs[{index}]")
            if checked.ndim != 2 or checked.shape[0] != count:
                raise ValueError("role inputs must have shape (samples, role dimension)")
            if checked.dtype != reference.dtype or checked.device != reference.device:
                raise ValueError("all panel tensors must share dtype and device")


@dataclass(frozen=True)
class SemanticCounterfactualPanel:
    """Paired A/B instructions with all non-instruction roles held fixed."""

    panel: TypedInputPanel
    pair_ids: tuple[str, ...]
    instruction_labels: tuple[str, ...]
    active_components: tuple[int, ...]

    def __post_init__(self) -> None:
        count = len(self.panel.sample_ids)
        if count % 2 or len(self.pair_ids) != count or len(self.instruction_labels) != count:
            raise ValueError("counterfactual panel must contain adjacent A/B pairs")
        if len(self.active_components) != count:
            raise ValueError("one active component is required per sample")
        if len(set(self.pair_ids[::2])) != count // 2:
            raise ValueError("counterfactual pair IDs must be unique")
        for index in range(0, count, 2):
            if self.pair_ids[index] != self.pair_ids[index + 1]:
                raise ValueError("adjacent counterfactual rows must share pair_ids")
            if self.instruction_labels[index : index + 2] != ("a", "b"):
                raise ValueError("each counterfactual pair must be ordered A then B")
            left = self.active_components[index]
            right = self.active_components[index + 1]
            if left < 0 or right >= len(MECHANISM_NAMES) or left % 2 or right != left + 1:
                raise ValueError("A/B active component parity is invalid")
            if any(
                not torch.equal(values[index], values[index + 1])
                for values in self.panel.role_inputs[1:]
            ):
                raise ValueError("non-instruction roles must be bitwise fixed within each pair")


def _validated_policy_inputs(
    policy: TypedCPPolicy, role_inputs: Sequence[Tensor]
) -> tuple[Tensor, ...]:
    if len(role_inputs) != len(ROLE_NAMES):
        raise ValueError("one independent input is required for every typed role")
    checked: list[Tensor] = []
    leading_shape: torch.Size | None = None
    for index, (value, dimension) in enumerate(zip(role_inputs, policy.role_dims)):
        tensor = _real_finite(value, f"role_inputs[{index}]")
        if tensor.shape[-1:] != (dimension,):
            raise ValueError(f"role_inputs[{index}] must end in dimension {dimension}")
        if tensor.dtype != policy.output_factor.dtype or tensor.device != policy.output_factor.device:
            raise ValueError("inputs and factors must share dtype and device")
        if leading_shape is None:
            leading_shape = tensor.shape[:-1]
        elif tensor.shape[:-1] != leading_shape:
            raise ValueError("all role inputs must have identical leading shapes")
        checked.append(tensor)
    return tuple(checked)


@torch.no_grad()
def factorized_forward(policy: TypedCPPolicy, role_inputs: Sequence[Tensor]) -> Tensor:
    """Evaluate the exact CP policy without materializing its dense core."""

    inputs = _validated_policy_inputs(policy, role_inputs)
    scores = [value @ factor for value, factor in zip(inputs, policy.role_factors)]
    product = torch.stack(scores).prod(dim=0)
    return product @ policy.output_factor.T


@torch.no_grad()
def dense_coefficient_core(policy: TypedCPPolicy) -> Tensor:
    """Materialize the order-seven coefficient tensor for tiny oracle checks."""

    equation = "or," + ",".join(
        f"{label}r" for label in _EINSUM_INPUT_LABELS
    ) + "->o" + "".join(_EINSUM_INPUT_LABELS)
    return torch.einsum(equation, policy.output_factor, *policy.role_factors)


@torch.no_grad()
def dense_forward(policy: TypedCPPolicy, role_inputs: Sequence[Tensor]) -> Tensor:
    """Evaluate by an independently materialized dense order-seven core."""

    inputs = _validated_policy_inputs(policy, role_inputs)
    core = dense_coefficient_core(policy)
    leading = inputs[0].shape[:-1]
    flat_inputs = [value.reshape(-1, value.shape[-1]) for value in inputs]
    equation = "o" + "".join(_EINSUM_INPUT_LABELS) + "," + ",".join(
        f"n{label}" for label in _EINSUM_INPUT_LABELS
    ) + "->no"
    output = torch.einsum(equation, core, *flat_inputs)
    return output.reshape(*leading, policy.output_dim)


def _validated_metrics(
    policy: TypedCPPolicy, input_metrics: Sequence[Tensor]
) -> tuple[Tensor, ...]:
    if len(input_metrics) != len(ROLE_NAMES):
        raise ValueError("one input metric is required for every typed role")
    checked = []
    for index, (metric, dimension) in enumerate(zip(input_metrics, policy.role_dims)):
        value = _symmetric_psd(metric, dimension, f"input_metrics[{index}]")
        if value.dtype != policy.output_factor.dtype or value.device != policy.output_factor.device:
            raise ValueError("metrics and policy must share dtype and device")
        checked.append(value)
    return tuple(checked)


@torch.no_grad()
def factorized_forward_metric(
    policy: TypedCPPolicy, input_metrics: Sequence[Tensor]
) -> Tensor:
    """Propagate six independent coefficient metrics through the CP map."""

    metrics = _validated_metrics(policy, input_metrics)
    component_metric = policy.output_factor.new_ones(policy.rank, policy.rank)
    for factor, metric in zip(policy.role_factors, metrics):
        component_metric = component_metric * (factor.T @ metric @ factor)
    result = policy.output_factor @ component_metric @ policy.output_factor.T
    return 0.5 * (result + result.T)


@torch.no_grad()
def factorized_role_environment(
    policy: TypedCPPolicy,
    downstream_metric: Tensor,
    input_metrics: Sequence[Tensor],
    role: int,
) -> Tensor:
    """Pull an action metric to one of the six independent role occurrences."""

    metrics = _validated_metrics(policy, input_metrics)
    if isinstance(role, bool) or not isinstance(role, int) or not 0 <= role < len(ROLE_NAMES):
        raise ValueError("role must index one of the six typed occurrences")
    downstream = _symmetric_psd(
        downstream_metric, policy.output_dim, "downstream_metric"
    )
    if downstream.dtype != policy.output_factor.dtype or downstream.device != policy.output_factor.device:
        raise ValueError("downstream metric and policy must share dtype and device")
    component_metric = policy.output_factor.T @ downstream @ policy.output_factor
    for index, (factor, metric) in enumerate(zip(policy.role_factors, metrics)):
        if index != role:
            component_metric = component_metric * (factor.T @ metric @ factor)
    factor = policy.role_factors[role]
    result = factor @ component_metric @ factor.T
    return 0.5 * (result + result.T)


@torch.no_grad()
def dense_role_environment_oracle(
    policy: TypedCPPolicy,
    downstream_metric: Tensor,
    input_metrics: Sequence[Tensor],
    role: int,
) -> Tensor:
    """Independent dense-core role environment used only for small validation."""

    metrics = _validated_metrics(policy, input_metrics)
    downstream = _symmetric_psd(
        downstream_metric, policy.output_dim, "downstream_metric"
    )
    if isinstance(role, bool) or not isinstance(role, int) or not 0 <= role < len(ROLE_NAMES):
        raise ValueError("role must index one of the six typed occurrences")
    core = dense_coefficient_core(policy)
    input_axis = role + 1
    other_axes = [axis for axis in range(1, core.ndim) if axis != input_axis]
    arranged = core.permute(0, input_axis, *other_axes)
    rows = arranged.reshape(policy.output_dim, policy.role_dims[role], -1)
    context = core.new_ones(1, 1)
    for axis in other_axes:
        context = torch.kron(context, metrics[axis - 1])
    result = torch.einsum("oia,op,pjb,ab->ij", rows, downstream, rows, context)
    return 0.5 * (result + result.T)


@torch.no_grad()
def balanced_role_odt(
    policy: TypedCPPolicy,
    downstream_metric: Tensor,
    input_metrics: Sequence[Tensor],
    role: int,
    *,
    support_rtol: float = 1e-12,
) -> BalancedMetricODT:
    """Return balanced ODT on one typed role's reachable support."""

    metrics = _validated_metrics(policy, input_metrics)
    environment = factorized_role_environment(
        policy, downstream_metric, metrics, role
    )
    return balanced_metric_odt(
        metrics[role], environment, support_rtol=support_rtol
    )


@torch.no_grad()
def apply_hidden_gauge(
    policy: TypedCPPolicy, gauges: GaugeTransform
) -> TypedCPPolicy:
    """Rewrite factors for ``x'_a = G_a x_a`` with exact function preservation."""

    if tuple(matrix.shape[0] for matrix in gauges.matrices) != policy.role_dims:
        raise ValueError("gauge dimensions must match policy role dimensions")
    transformed = []
    for factor, matrix in zip(policy.role_factors, gauges.matrices):
        if matrix.dtype != factor.dtype or matrix.device != factor.device:
            raise ValueError("gauges and policy must share dtype and device")
        transformed.append(torch.linalg.solve(matrix.T, factor))
    return TypedCPPolicy(
        role_factors=tuple(transformed),
        output_factor=policy.output_factor.clone(),
        role_names=policy.role_names,
        mechanism_names=policy.mechanism_names,
    )


@torch.no_grad()
def transform_inputs(
    role_inputs: Sequence[Tensor], gauges: GaugeTransform
) -> tuple[Tensor, ...]:
    """Transform row-vector input batches according to ``x'_a = G_a x_a``."""

    if len(role_inputs) != len(ROLE_NAMES):
        raise ValueError("one input is required for every typed role")
    result = []
    for index, (value, matrix) in enumerate(zip(role_inputs, gauges.matrices)):
        tensor = _real_finite(value, f"role_inputs[{index}]")
        if tensor.shape[-1] != matrix.shape[1]:
            raise ValueError("input and gauge dimensions do not match")
        if tensor.dtype != matrix.dtype or tensor.device != matrix.device:
            raise ValueError("inputs and gauges must share dtype and device")
        result.append(tensor @ matrix.T)
    return tuple(result)


@torch.no_grad()
def transform_input_metrics(
    input_metrics: Sequence[Tensor], gauges: GaugeTransform
) -> tuple[Tensor, ...]:
    """Apply covariance transport ``C'_a = G_a C_a G_a^T``."""

    if len(input_metrics) != len(ROLE_NAMES):
        raise ValueError("one metric is required for every typed role")
    result = []
    for index, (metric, matrix) in enumerate(zip(input_metrics, gauges.matrices)):
        value = _symmetric_psd(metric, matrix.shape[1], f"input_metrics[{index}]")
        if value.dtype != matrix.dtype or value.device != matrix.device:
            raise ValueError("metrics and gauges must share dtype and device")
        transformed = matrix @ value @ matrix.T
        result.append(0.5 * (transformed + transformed.T))
    return tuple(result)


@torch.no_grad()
def map_state_directions_to_canonical(
    state_directions: Tensor, gauge: Tensor
) -> Tensor:
    """Map state directions from primed coordinates back to canonical coordinates."""

    directions = _real_finite(state_directions, "state_directions")
    matrix = _real_finite(gauge, "gauge")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gauge must be square")
    if directions.ndim != 2 or directions.shape[0] != matrix.shape[0]:
        raise ValueError("state directions and gauge dimensions do not match")
    return torch.linalg.solve(matrix, directions)


@torch.no_grad()
def analysis_covectors_from_state_directions(
    upstream_metric: Tensor, state_directions: Tensor
) -> Tensor:
    """Convert balanced state directions into dual analysis covectors.

    For ``S = C A`` this returns the corresponding columns of ``A`` up to
    nonzero component scaling.  It is the coordinate-covariant object needed
    by score-zeroing interventions.  A Euclidean interpretation of ``S`` as a
    covector would not survive a non-orthogonal gauge.
    """

    directions = _real_finite(state_directions, "state_directions")
    if directions.ndim != 2 or directions.shape[1] == 0:
        raise ValueError("state_directions must be a nonempty matrix")
    metric = _symmetric_psd(
        upstream_metric, directions.shape[0], "upstream_metric"
    )
    if metric.dtype != directions.dtype or metric.device != directions.device:
        raise ValueError("metric and directions must share dtype and device")
    if int(torch.linalg.matrix_rank(metric).item()) != metric.shape[0]:
        raise ValueError("upstream_metric must be positive definite for dual recovery")
    return torch.linalg.solve(metric, directions)


@torch.no_grad()
def component_zeroing_projector(
    factor: Tensor,
    component: int,
    upstream_metric: Tensor,
) -> Tensor:
    """Return the column-vector projector that zeros one CP score."""

    planted = _real_finite(factor, "factor")
    if planted.ndim != 2:
        raise ValueError("factor must be a matrix")
    if not 0 <= component < planted.shape[1]:
        raise ValueError("component is out of range")
    metric = _symmetric_psd(upstream_metric, planted.shape[0], "upstream_metric")
    if metric.dtype != planted.dtype or metric.device != planted.device:
        raise ValueError("upstream_metric and factor must share dtype and device")
    gram = planted.T @ metric @ planted
    if int(torch.linalg.matrix_rank(gram).item()) != gram.shape[0]:
        raise ValueError("factor must have full column rank")
    dual = metric @ planted @ torch.linalg.inv(gram)
    return torch.eye(planted.shape[0], dtype=planted.dtype, device=planted.device) - (
        dual[:, component : component + 1] @ planted[:, component : component + 1].T
    )


@torch.no_grad()
def component_zeroing_intervention(
    inputs: Tensor,
    factor: Tensor,
    component: int,
    upstream_metric: Tensor,
) -> Tensor:
    """Set one CP score to zero with a coordinate-covariant C-dual edit."""

    values = _real_finite(inputs, "inputs")
    planted = _real_finite(factor, "factor")
    if planted.ndim != 2 or values.shape[-1] != planted.shape[0]:
        raise ValueError("inputs and factor dimensions do not match")
    projector = component_zeroing_projector(planted, component, upstream_metric)
    return values @ projector.T


@torch.no_grad()
def subspace_recovery(reference: Tensor, estimate: Tensor) -> float:
    """Mean squared canonical correlation, in ``[0, 1]``, for equal-rank spans."""

    left = _real_finite(reference, "reference")
    right = _real_finite(estimate, "estimate")
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("reference and estimate must be same-shaped matrices")
    rank = left.shape[1]
    if rank == 0:
        raise ValueError("subspaces must have positive rank")
    if int(torch.linalg.matrix_rank(left).item()) != rank or int(torch.linalg.matrix_rank(right).item()) != rank:
        raise ValueError("reference and estimate must have full column rank")
    q_left = torch.linalg.qr(left, mode="reduced").Q
    q_right = torch.linalg.qr(right, mode="reduced").Q
    return float(torch.linalg.matrix_norm(q_left.T @ q_right).square().item() / rank)


@torch.no_grad()
def ordered_component_recovery(reference: Tensor, estimate: Tensor) -> float:
    """Mean squared cosine at the frozen CP component ordering."""

    left = _real_finite(reference, "reference")
    right = _real_finite(estimate, "estimate")
    if left.ndim != 2 or right.shape != left.shape or left.shape[1] == 0:
        raise ValueError("reference and estimate must be equal nonempty matrices")
    left = left / torch.linalg.vector_norm(left, dim=0).clamp_min(
        torch.finfo(left.dtype).tiny
    )
    right = right / torch.linalg.vector_norm(right, dim=0).clamp_min(
        torch.finfo(right.dtype).tiny
    )
    return float((left.mul(right).sum(0).square().mean()).item())


def _orthonormal_columns(
    generator: torch.Generator, rows: int, columns: int, dtype: torch.dtype
) -> Tensor:
    if rows < columns:
        raise ValueError("ambient dimensions must be at least the planted rank")
    values = torch.randn(rows, columns, generator=generator, dtype=dtype)
    return torch.linalg.qr(values, mode="reduced").Q


@torch.no_grad()
def make_planted_policy(
    seed: int,
    *,
    role_dim: int = 12,
    output_dim: int = 7,
    dtype: torch.dtype = torch.float64,
) -> TypedCPPolicy:
    """Create a canonical rank-six planted policy with a frozen strength order."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    rank = len(MECHANISM_NAMES)
    if role_dim < rank or output_dim < rank:
        raise ValueError("role_dim and output_dim must be at least six")
    generator = torch.Generator().manual_seed(seed)
    role_factors = tuple(
        _orthonormal_columns(generator, role_dim, rank, dtype)
        for _ in ROLE_NAMES
    )
    output_basis = _orthonormal_columns(generator, output_dim, rank, dtype)
    strengths = torch.tensor((9.0, 7.0, 5.0, 4.0, 3.0, 2.0), dtype=dtype)
    return TypedCPPolicy(
        role_factors=role_factors,
        output_factor=output_basis * strengths.unsqueeze(0),
    )


@torch.no_grad()
def make_random_gauges(
    seed: int,
    role_dims: Sequence[int],
    *,
    dtype: torch.dtype = torch.float64,
    minimum_scale: float = 0.35,
    maximum_scale: float = 2.8,
) -> GaugeTransform:
    """Generate reproducible, non-orthogonal, well-conditioned hidden gauges."""

    if len(role_dims) != len(ROLE_NAMES) or any(dimension <= 0 for dimension in role_dims):
        raise ValueError("role_dims must contain six positive dimensions")
    if not 0 < minimum_scale <= maximum_scale:
        raise ValueError("gauge scales must satisfy 0 < minimum <= maximum")
    generator = torch.Generator().manual_seed(seed)
    matrices = []
    for dimension in role_dims:
        left = _orthonormal_columns(generator, dimension, dimension, dtype)
        right = _orthonormal_columns(generator, dimension, dimension, dtype)
        scales = torch.logspace(
            torch.log10(torch.tensor(minimum_scale, dtype=dtype)).item(),
            torch.log10(torch.tensor(maximum_scale, dtype=dtype)).item(),
            dimension,
            dtype=dtype,
        )
        matrices.append((left * scales.unsqueeze(0)) @ right.T)
    return GaugeTransform(tuple(matrices))


@torch.no_grad()
def planted_input_metrics(
    policy: TypedCPPolicy,
    *,
    signal_variance: float = 1.0,
    nuisance_variance: float = 9.0,
) -> tuple[Tensor, ...]:
    """Return canonical SPD role metrics with high-variance causal-null nuisance."""

    if not 0 < signal_variance < nuisance_variance:
        raise ValueError("require 0 < signal_variance < nuisance_variance")
    result = []
    for factor in policy.role_factors:
        support = factor @ factor.T
        identity = torch.eye(factor.shape[0], dtype=factor.dtype, device=factor.device)
        metric = signal_variance * support + nuisance_variance * (identity - support)
        result.append(0.5 * (metric + metric.T))
    return tuple(result)


def _covariance_design(metric: Tensor) -> Tensor:
    values, vectors = torch.linalg.eigh(metric)
    if not bool((values > 0).all()):
        raise ValueError("discovery metrics must be positive definite")
    root = vectors * values.sqrt().unsqueeze(0)
    dimension = metric.shape[0]
    return torch.cat((root.T, -root.T), dim=0) * (dimension**0.5)


@torch.no_grad()
def make_discovery_panel(
    policy: TypedCPPolicy,
    input_metrics: Sequence[Tensor],
    *,
    namespace: str,
) -> TypedInputPanel:
    """Create a zero-mean finite panel whose population covariances are exact."""

    metrics = _validated_metrics(policy, input_metrics)
    if len(set(policy.role_dims)) != 1:
        raise ValueError("the exact paired covariance design requires equal role dimensions")
    role_inputs = tuple(_covariance_design(metric) for metric in metrics)
    count = role_inputs[0].shape[0]
    sample_ids = tuple(f"{namespace}:covariance:{index:04d}" for index in range(count))
    return TypedInputPanel(namespace, sample_ids, role_inputs)


@torch.no_grad()
def empirical_input_metrics(panel: TypedInputPanel) -> tuple[Tensor, ...]:
    """Return population covariances after independently centering every role."""

    result = []
    for values in panel.role_inputs:
        centered = values - values.mean(0)
        covariance = centered.T @ centered / values.shape[0]
        result.append(0.5 * (covariance + covariance.T))
    return tuple(result)


def _synthesize_scores(factor: Tensor, scores: Tensor) -> Tensor:
    gram = factor.T @ factor
    dual = factor @ torch.linalg.inv(gram)
    return scores @ dual.T


@torch.no_grad()
def make_semantic_counterfactual_panel(
    policy: TypedCPPolicy,
    seed: int,
    *,
    pairs: int = 18,
    namespace: str,
) -> SemanticCounterfactualPanel:
    """Build paired instruction swaps with phase and scene held exactly fixed."""

    if pairs <= 0:
        raise ValueError("pairs must be positive")
    if policy.rank != len(MECHANISM_NAMES):
        raise ValueError("semantic generator requires the frozen rank-six mechanism schema")
    generator = torch.Generator().manual_seed(seed)
    rank = policy.rank
    score_rows = [[] for _ in ROLE_NAMES]
    sample_ids: list[str] = []
    pair_ids: list[str] = []
    instruction_labels: list[str] = []
    active_components: list[int] = []
    instruction_a = policy.output_factor.new_tensor((1, 0, 1, 0, 1, 0))
    instruction_b = policy.output_factor.new_tensor((0, 1, 0, 1, 0, 1))
    for pair_index in range(pairs):
        phase_index = pair_index % 3
        # Keep all three same-target mechanisms co-active. The selected phase
        # has unit gain and the two background phases have smaller nonzero
        # gain, so off-target leakage is measurable rather than structural zero.
        phase = policy.output_factor.new_full((rank,), 0.25)
        phase[2 * phase_index : 2 * phase_index + 2] = 1
        object_a = policy.output_factor.new_ones(rank)
        object_b = policy.output_factor.new_ones(rank)
        amplitudes = 0.6 + 0.8 * torch.rand(rank, generator=generator, dtype=policy.output_factor.dtype)
        object_a[0::2] = amplitudes[0::2]
        object_b[1::2] = amplitudes[1::2]
        robot_state = 0.7 + 0.6 * torch.rand(rank, generator=generator, dtype=policy.output_factor.dtype)
        action_query = 0.8 + 0.4 * torch.rand(rank, generator=generator, dtype=policy.output_factor.dtype)
        pair_id = f"{namespace}:pair:{pair_index:04d}"
        for label, instruction, offset in (
            ("a", instruction_a, 0),
            ("b", instruction_b, 1),
        ):
            score_rows[0].append(instruction)
            score_rows[1].append(object_a)
            score_rows[2].append(object_b)
            score_rows[3].append(robot_state)
            score_rows[4].append(phase)
            score_rows[5].append(action_query)
            sample_ids.append(f"{pair_id}:instruction_{label}")
            pair_ids.append(pair_id)
            instruction_labels.append(label)
            active_components.append(2 * phase_index + offset)
    role_inputs = tuple(
        _synthesize_scores(factor, torch.stack(rows))
        for factor, rows in zip(policy.role_factors, score_rows)
    )
    panel = TypedInputPanel(namespace, tuple(sample_ids), role_inputs)
    return SemanticCounterfactualPanel(
        panel=panel,
        pair_ids=tuple(pair_ids),
        instruction_labels=tuple(instruction_labels),
        active_components=tuple(active_components),
    )


@torch.no_grad()
def gauge_panel(panel: TypedInputPanel, gauges: GaugeTransform, *, namespace: str) -> TypedInputPanel:
    """Transport a panel and assign a new explicit namespace."""

    transformed = transform_inputs(panel.role_inputs, gauges)
    sample_ids = tuple(f"{namespace}:{index:04d}" for index in range(len(panel.sample_ids)))
    return TypedInputPanel(namespace, sample_ids, transformed)


@torch.no_grad()
def intervention_effects(
    policy: TypedCPPolicy,
    panel: SemanticCounterfactualPanel,
    *,
    role: int,
    upstream_metric: Tensor,
    intervention_factor: Tensor | None = None,
) -> dict[str, Tensor]:
    """Apply the row's active-component intervention and report target/off-target energy."""

    if isinstance(role, bool) or not isinstance(role, int) or not 0 <= role < len(ROLE_NAMES):
        raise ValueError("role must index one of the six typed occurrences")
    factor = policy.role_factors[role] if intervention_factor is None else _real_finite(
        intervention_factor, "intervention_factor"
    )
    if factor.shape != policy.role_factors[role].shape:
        raise ValueError("intervention_factor must match the selected policy factor shape")
    if factor.dtype != policy.output_factor.dtype or factor.device != policy.output_factor.device:
        raise ValueError("intervention_factor and policy must share dtype and device")
    baseline = policy(panel.panel.role_inputs)
    intervened_rows = []
    for row, component in enumerate(panel.active_components):
        inputs = [value[row : row + 1].clone() for value in panel.panel.role_inputs]
        inputs[role] = component_zeroing_intervention(
            inputs[role], factor, component, upstream_metric
        )
        intervened_rows.append(policy(inputs).squeeze(0))
    intervened = torch.stack(intervened_rows)
    delta = baseline - intervened
    target_energy = []
    off_target_energy = []
    baseline_target_energy = []
    intervened_target_energy = []
    for row, component in enumerate(panel.active_components):
        direction = policy.output_factor[:, component]
        unit = direction / torch.linalg.vector_norm(direction)
        target = torch.dot(delta[row], unit) * unit
        off_target = delta[row] - target
        target_energy.append(target.square().sum())
        off_target_energy.append(off_target.square().sum())
        baseline_target_energy.append(torch.dot(baseline[row], unit).square())
        intervened_target_energy.append(torch.dot(intervened[row], unit).square())
    return {
        "baseline": baseline,
        "intervened": intervened,
        "delta": delta,
        "target_energy": torch.stack(target_energy),
        "off_target_energy": torch.stack(off_target_energy),
        "baseline_target_energy": torch.stack(baseline_target_energy),
        "intervened_target_energy": torch.stack(intervened_target_energy),
    }


@torch.no_grad()
def counterfactual_swap_error(
    policy: TypedCPPolicy, panel: SemanticCounterfactualPanel
) -> Tensor:
    """Swap only instruction rows and compare with the paired observed outputs."""

    outputs = policy(panel.panel.role_inputs)
    swapped_inputs = list(panel.panel.role_inputs)
    permutation = torch.arange(outputs.shape[0], device=outputs.device).reshape(-1, 2).flip(1).reshape(-1)
    swapped_inputs[0] = swapped_inputs[0][permutation]
    swapped = policy(swapped_inputs)
    expected = outputs[permutation]
    return torch.linalg.vector_norm(swapped - expected, dim=-1)


@torch.no_grad()
def action_query_sign_equivariance_error(
    policy: TypedCPPolicy, panel: SemanticCounterfactualPanel
) -> Tensor:
    """Verify odd sign equivariance in the independent action-query occurrence."""

    positive = policy(panel.panel.role_inputs)
    negative_inputs = list(panel.panel.role_inputs)
    negative_inputs[5] = -negative_inputs[5]
    negative = policy(negative_inputs)
    return torch.linalg.vector_norm(positive + negative, dim=-1)


# Compatibility only. This identity is not evidence of fixed-context
# multimodality and new artifacts must use the explicit sign-equivariance name.
multimodal_query_branch_error = action_query_sign_equivariance_error


CLAIM_BOUNDARY = (
    "A pass validates exact six-occurrence CP metric contraction, coordinate-gauge covariance, "
    "and selective recovery of mechanisms deliberately planted in a synthetic policy. It does "
    "not establish global ODT for attention, discover a mechanism in a trained VLA, demonstrate "
    "human-nameable features, or establish closed-loop robotic competence."
)
