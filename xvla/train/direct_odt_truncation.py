"""Post-certificate truncation for a directly canonicalized projective DAG.

The routines in this module never construct a new basis.  They consume the
ordered bases already produced by Algorithm 3, keep leading coordinates, and
apply that choice to every syntactic occurrence of each shared bond.
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch

from xvla.train.odt_engine_v2.constants import (
    COMPACT_RELATIVE_VALUE_FLOORS,
    COMPACT_TRACE_RETENTION_TARGETS,
)
from xvla.train.odt_engine_v2.graph import (
    _clone_network,
    _parent_occurrences,
    _prepare_physical_input,
    _validate_network,
    _walk_unique,
)
from xvla.train.odt_engine_v2.ops import (
    _core_apply,
    _normalize_projective_batch,
)
from xvla.train.odt_engine_v2.types import (
    CPBinaryCore,
    CompactSpectrumRecord,
    DenseCloneCore,
    DiagonalImplicitDAG,
    ImplicitCore,
    ImplicitNode,
    ImplicitProjectiveDAG,
    RawPhysicalInput,
    ReducedQRBinaryCore,
    UnaryCore,
)


Tensor = torch.Tensor
_MISSING_RANK = (1 << 32) - 1
_CERTIFICATE_REPLAY_LIMIT = 1e-10
_CERTIFICATE_OFFDIAGONAL_LIMIT = 3e-10


@dataclass(frozen=True)
class ScaledPositiveSummary:
    mantissa: float
    binary_exponent: int

    @property
    def log2_value(self) -> float:
        if self.mantissa == 0.0:
            return -math.inf
        return math.log2(self.mantissa) + float(self.binary_exponent)


class _ScaledPositiveAccumulator:
    def __init__(self) -> None:
        self.mantissa = 0.0
        self.binary_exponent = 0

    def add(self, mantissa: float, binary_exponent: int) -> None:
        if not math.isfinite(mantissa) or mantissa < 0.0:
            raise ValueError("scaled positive contribution must be finite and nonnegative")
        if mantissa == 0.0:
            return
        incoming_mantissa, incoming_shift = math.frexp(mantissa)
        incoming_exponent = int(binary_exponent) + incoming_shift
        if self.mantissa == 0.0:
            self.mantissa = incoming_mantissa
            self.binary_exponent = incoming_exponent
            return
        common_exponent = max(self.binary_exponent, incoming_exponent)
        combined = math.ldexp(
            self.mantissa, self.binary_exponent - common_exponent
        ) + math.ldexp(incoming_mantissa, incoming_exponent - common_exponent)
        self.mantissa, shift = math.frexp(combined)
        self.binary_exponent = common_exponent + shift

    def finish(self) -> ScaledPositiveSummary:
        return ScaledPositiveSummary(self.mantissa, self.binary_exponent)


def _label_group(label: str) -> str:
    if label.startswith("policy.product.center"):
        return "product_center"
    if label.startswith("policy.product.factor"):
        return "product_factors"
    if label.startswith("policy.product.gate"):
        return "product_gates"
    if label.startswith("policy.product"):
        return "product_other"
    for stem in ("vision.block", "joint.block"):
        if label.startswith(stem):
            remainder = label[len(stem) :]
            digits = remainder[: len(remainder) - len(remainder.lstrip("0123456789"))]
            if digits:
                return f"{stem}{digits}"
    if label.startswith(("image.", "vision.patch")):
        return "image_ingress"
    if label.startswith(("instruction.", "joint.instruction")):
        return "instruction_ingress"
    if label.startswith(("state.", "joint.state")):
        return "state_ingress"
    if label.startswith("joint.action"):
        return "action_queries"
    first = label.split(".", 1)[0]
    if first in {"vision", "language", "instruction", "state", "joint", "policy"}:
        return first
    return "other"


@dataclass
class _GroupAccumulator:
    node_count: int
    total_dimension: int
    retained_dimensions: list[int]
    floor_dimensions: list[int]
    tail_sums: list[_ScaledPositiveAccumulator]
    zero_trace_nodes: int
    tie_extended_nodes: list[int]

    @classmethod
    def empty(cls) -> "_GroupAccumulator":
        return cls(
            node_count=0,
            total_dimension=0,
            retained_dimensions=[0] * len(COMPACT_TRACE_RETENTION_TARGETS),
            floor_dimensions=[0] * len(COMPACT_RELATIVE_VALUE_FLOORS),
            tail_sums=[
                _ScaledPositiveAccumulator()
                for _ in COMPACT_TRACE_RETENTION_TARGETS
            ],
            zero_trace_nodes=0,
            tie_extended_nodes=[0] * len(COMPACT_TRACE_RETENTION_TARGETS),
        )

    def observe(self, record: CompactSpectrumRecord) -> None:
        self.node_count += 1
        self.total_dimension += record.dimension
        self.zero_trace_nodes += int(record.zero_trace)
        for index, rank in enumerate(record.trace_retention_ranks):
            self.retained_dimensions[index] += rank
            self.tie_extended_nodes[index] += int(record.tie_extensions[index] > 0)
            self.tail_sums[index].add(
                record.trace_tail_mantissas[index],
                record.environment_binary_exponent,
            )
        for index, rank in enumerate(record.relative_floor_ranks):
            self.floor_dimensions[index] += rank

    def result(self) -> dict[str, object]:
        return {
            "node_count": self.node_count,
            "total_dimension": self.total_dimension,
            "trace_retained_dimensions": dict(
                zip(
                    (str(value) for value in COMPACT_TRACE_RETENTION_TARGETS),
                    self.retained_dimensions,
                )
            ),
            "relative_floor_dimensions": dict(
                zip(
                    (str(value) for value in COMPACT_RELATIVE_VALUE_FLOORS),
                    self.floor_dimensions,
                )
            ),
            "trace_tail_sums": {
                str(target): {
                    "mantissa": summary.mantissa,
                    "binary_exponent": summary.binary_exponent,
                    "log2_value": summary.log2_value,
                }
                for target, summary in zip(
                    COMPACT_TRACE_RETENTION_TARGETS,
                    (item.finish() for item in self.tail_sums),
                )
            },
            "zero_trace_nodes": self.zero_trace_nodes,
            "tie_extended_nodes": dict(
                zip(
                    (str(value) for value in COMPACT_TRACE_RETENTION_TARGETS),
                    self.tie_extended_nodes,
                )
            ),
        }


class CompactRankPlan(Mapping[int, int]):
    """Read-only leading-coordinate plan backed by a dense integer array."""

    def __init__(
        self,
        ranks: array,
        dimensions: array,
        expected: bytearray,
        target: float,
        node_count: int,
        network_identity: int | None,
    ) -> None:
        self._ranks = ranks
        self._dimensions = dimensions
        self._expected = expected
        self.target = float(target)
        self._node_count = int(node_count)
        self._network_identity = network_identity

    def __getitem__(self, uid: int) -> int:
        if not isinstance(uid, int) or not 0 <= uid < len(self._expected):
            raise KeyError(uid)
        if not self._expected[uid]:
            raise KeyError(uid)
        value = int(self._ranks[uid])
        if value == _MISSING_RANK:
            raise RuntimeError("compact rank plan was requested before completion")
        return value

    def __iter__(self) -> Iterator[int]:
        return (uid for uid, flag in enumerate(self._expected) if flag)

    def __len__(self) -> int:
        return self._node_count

    def expected_dimension(self, uid: int) -> int:
        _ = self[uid]
        return int(self._dimensions[uid])


class CompactRankBank:
    """Zero-spectrum-retention callback and compact rank-plan store."""

    def __init__(
        self,
        expected_uids: Sequence[int],
        *,
        network_identity: int | None = None,
    ) -> None:
        uids = tuple(int(value) for value in expected_uids)
        if not uids or len(set(uids)) != len(uids) or min(uids) < 0:
            raise ValueError("expected node identifiers must be unique nonnegative integers")
        maximum = max(uids)
        if maximum >= 2 * len(uids) + 1024:
            raise ValueError("node identifiers are too sparse for compact rank storage")
        self._expected = bytearray(maximum + 1)
        for uid in uids:
            self._expected[uid] = 1
        self._seen = bytearray(maximum + 1)
        self._dimensions = array("I", [_MISSING_RANK]) * (maximum + 1)
        self._ranks = tuple(
            array("I", [_MISSING_RANK]) * (maximum + 1)
            for _ in COMPACT_TRACE_RETENTION_TARGETS
        )
        self._node_count = len(uids)
        self._network_identity = network_identity
        self._bound_by_sweep = False
        self._observed = 0
        self._overall = _GroupAccumulator.empty()
        self._groups: dict[str, _GroupAccumulator] = {}

    @classmethod
    def for_network(cls, network: ImplicitProjectiveDAG) -> "CompactRankBank":
        return cls(
            tuple(node.uid for node in _walk_unique(network.root)),
            network_identity=id(network),
        )

    def __call__(self, record: CompactSpectrumRecord) -> None:
        uid = record.uid
        if not 0 <= uid < len(self._expected) or not self._expected[uid]:
            raise ValueError("compact spectrum callback received an unknown node")
        if self._seen[uid]:
            raise ValueError("compact spectrum callback received a node twice")
        if len(record.trace_retention_ranks) != len(self._ranks):
            raise ValueError("compact spectrum callback schedule drifted")
        if len(record.relative_floor_ranks) != len(COMPACT_RELATIVE_VALUE_FLOORS):
            raise ValueError("compact relative-floor schedule drifted")
        if not all(1 <= rank <= record.dimension for rank in record.trace_retention_ranks):
            raise ValueError("compact spectrum callback emitted an invalid rank")
        self._seen[uid] = 1
        self._dimensions[uid] = record.dimension
        for values, rank in zip(self._ranks, record.trace_retention_ranks):
            values[uid] = rank
        self._observed += 1
        self._overall.observe(record)
        group = _label_group(record.label)
        self._groups.setdefault(group, _GroupAccumulator.empty()).observe(record)

    def bind_network(self, network: ImplicitProjectiveDAG) -> None:
        """Bind plans to the exact network instance mutated by Algorithm 3."""

        if self._observed or self._bound_by_sweep:
            raise RuntimeError("compact rank bank cannot be rebound")
        actual = {node.uid for node in _walk_unique(network.root)}
        expected = {uid for uid, flag in enumerate(self._expected) if flag}
        if actual != expected:
            raise ValueError("compact rank bank node inventory drifted before Algorithm 3")
        self._network_identity = id(network)
        self._bound_by_sweep = True

    @property
    def complete(self) -> bool:
        return self._observed == self._node_count

    def finish(self) -> dict[str, object]:
        if not self.complete:
            missing = self._node_count - self._observed
            raise RuntimeError(f"compact spectrum callback is missing {missing} nodes")
        return {
            "node_count": self._node_count,
            "full_spectra_retained": False,
            "tail_quantity": (
                "sum of discarded Algorithm 2 environment eigenvalues, with "
                "the environment binary scale restored"
            ),
            "simultaneous_multibond_error_bound_claimed": False,
            "trace_targets": list(COMPACT_TRACE_RETENTION_TARGETS),
            "relative_value_floors": list(COMPACT_RELATIVE_VALUE_FLOORS),
            "overall": self._overall.result(),
            "groups": {
                key: self._groups[key].result() for key in sorted(self._groups)
            },
        }

    def plan(self, target: float) -> CompactRankPlan:
        if not self.complete:
            raise RuntimeError("compact rank plan is unavailable before callback completion")
        try:
            index = COMPACT_TRACE_RETENTION_TARGETS.index(float(target))
        except ValueError as error:
            raise KeyError(f"unknown fixed trace-retention target {target}") from error
        return CompactRankPlan(
            self._ranks[index],
            self._dimensions,
            self._expected,
            float(target),
            self._node_count,
            self._network_identity,
        )


def _certified_network(diagonal: DiagonalImplicitDAG) -> ImplicitProjectiveDAG:
    network = diagonal.network
    _validate_network(network)
    nodes = _walk_unique(network.root)
    edge_occurrences = sum(len(node.children) for node in nodes)
    if not network.algorithm1_direct_rq_complete:
        raise ValueError("prefix truncation requires a completed direct Algorithm 1 sweep")
    if diagonal.diagonalized_node_count != len(nodes):
        raise ValueError("prefix truncation requires every bond to complete Algorithm 3")
    if diagonal.algorithm2_child_messages != edge_occurrences:
        raise ValueError("prefix truncation requires a complete Algorithm 2 contraction")
    if diagonal.pushed_parent_occurrences != diagonal.expected_parent_occurrences:
        raise ValueError("prefix truncation requires every Algorithm 3 occurrence push")
    if (
        not math.isfinite(diagonal.replay_relative_error)
        or diagonal.replay_relative_error > _CERTIFICATE_REPLAY_LIMIT
    ):
        raise ValueError("prefix truncation requires an exact full-rank replay certificate")
    if (
        not math.isfinite(diagonal.maximum_recontracted_offdiagonal_ratio)
        or diagonal.maximum_recontracted_offdiagonal_ratio
        > _CERTIFICATE_OFFDIAGONAL_LIMIT
    ):
        raise ValueError("prefix truncation requires diagonal Algorithm 3 environments")
    return network


def _validate_plan(
    network: ImplicitProjectiveDAG,
    plan: Mapping[int, int],
    *,
    enforce_network_binding: bool = True,
) -> tuple[ImplicitNode, ...]:
    nodes = _walk_unique(network.root)
    if not isinstance(plan, CompactRankPlan):
        known = {node.uid for node in nodes}
        unknown = set(plan) - known
        if unknown:
            raise ValueError("prefix plan contains unknown node identifiers")
    else:
        if (
            enforce_network_binding
            and plan._network_identity is not None
            and plan._network_identity != id(network)
        ):
            raise ValueError("compact prefix plan belongs to a different network instance")
        if len(plan) != len(nodes):
            raise ValueError("compact prefix plan does not cover this network")
    for node in nodes:
        if isinstance(plan, CompactRankPlan):
            raw_rank = plan[node.uid]
            if plan.expected_dimension(node.uid) != node.output_dimension:
                raise ValueError("compact prefix plan belongs to a different network shape")
        else:
            raw_rank = plan[node.uid] if node.uid in plan else node.output_dimension
        if isinstance(raw_rank, bool) or not isinstance(raw_rank, int):
            raise ValueError("prefix plan ranks must be integers")
        rank = int(raw_rank)
        if not 1 <= rank <= node.output_dimension:
            raise ValueError("prefix plan rank is outside its bond dimension")
    return nodes


def _rank_for(plan: Mapping[int, int], node: ImplicitNode) -> int:
    if isinstance(plan, CompactRankPlan):
        return int(plan[node.uid])
    return int(plan[node.uid]) if node.uid in plan else node.output_dimension


def _core_element_count(core: ImplicitCore) -> int:
    if isinstance(core, UnaryCore):
        return int(core.matrix.numel())
    if isinstance(core, CPBinaryCore):
        return int(
            core.output_factor.numel()
            + core.left_factor.numel()
            + core.right_factor.numel()
        )
    if isinstance(core, ReducedQRBinaryCore):
        return int(core.q_rows.numel())
    raise TypeError("production compression does not accept an explicit clone core")


def implicit_storage_elements(network: ImplicitProjectiveDAG) -> int:
    """Count tensor entries in the unique DAG cores and boundary head."""

    _validate_network(network)
    return int(network.head.numel()) + sum(
        _core_element_count(node.core) for node in _walk_unique(network.root)
    )


def implicit_allocated_storage_elements(network: ImplicitProjectiveDAG) -> int:
    """Count allocated tensor elements, rejecting hidden slice backing storage."""

    _validate_network(network)
    tensors: list[tuple[str, Tensor]] = [("boundary_head", network.head)]
    for node in _walk_unique(network.root):
        core = node.core
        if isinstance(core, UnaryCore):
            tensors.append((f"{node.label}.matrix", core.matrix))
        elif isinstance(core, CPBinaryCore):
            tensors.extend(
                (
                    (f"{node.label}.output_factor", core.output_factor),
                    (f"{node.label}.left_factor", core.left_factor),
                    (f"{node.label}.right_factor", core.right_factor),
                )
            )
        elif isinstance(core, ReducedQRBinaryCore):
            tensors.append((f"{node.label}.q_rows", core.q_rows))
        else:
            raise TypeError("production compression does not accept an explicit clone core")
    total = 0
    seen: set[tuple[str, int | None, int, int]] = set()
    for label, tensor in tensors:
        storage = tensor.untyped_storage()
        allocated_bytes = int(storage.nbytes())
        expected_bytes = int(tensor.numel()) * int(tensor.element_size())
        if tensor.storage_offset() != 0 or allocated_bytes != expected_bytes:
            raise RuntimeError(
                "compressed tensor retains hidden backing storage: "
                f"{label}, shape={tuple(tensor.shape)}, offset={tensor.storage_offset()}, "
                f"allocated_bytes={allocated_bytes}, expected_bytes={expected_bytes}"
            )
        identity = (
            tensor.device.type,
            tensor.device.index,
            int(storage.data_ptr()),
            allocated_bytes,
        )
        if identity not in seen:
            seen.add(identity)
            total += allocated_bytes // int(tensor.element_size())
    return total


def projected_prefix_storage_elements(
    diagonal: DiagonalImplicitDAG,
    plan: Mapping[int, int],
) -> int:
    """Count entries after a proposed structural leading-coordinate slice."""

    network = _certified_network(diagonal)
    nodes = _validate_plan(network, plan)
    total = int(network.head.shape[0]) * _rank_for(plan, network.root)
    for node in nodes:
        output_rank = _rank_for(plan, node)
        input_ranks = tuple(_rank_for(plan, child) for child in node.children)
        core = node.core
        if isinstance(core, UnaryCore):
            input_rank = input_ranks[0] if input_ranks else core.input_dimensions[0]
            total += output_rank * input_rank
        elif isinstance(core, CPBinaryCore):
            total += core.cp_rank * (output_rank + sum(input_ranks))
        elif isinstance(core, ReducedQRBinaryCore):
            total += output_rank * math.prod(input_ranks)
        else:
            raise TypeError("production compression does not accept an explicit clone core")
    return int(total)


def _compact_copy(value: Tensor) -> Tensor:
    result = torch.empty(tuple(value.shape), dtype=value.dtype, device=value.device)
    result.copy_(value)
    return result


def _slice_output(core: ImplicitCore, rank: int) -> ImplicitCore:
    if isinstance(core, UnaryCore):
        return UnaryCore(_compact_copy(core.matrix[:rank]), core.kind, core.binary_exponent)
    if isinstance(core, CPBinaryCore):
        return CPBinaryCore(
            _compact_copy(core.output_factor[:rank]),
            core.left_factor,
            core.right_factor,
            core.kind,
            core.binary_exponent,
            None,
        )
    if isinstance(core, ReducedQRBinaryCore):
        return ReducedQRBinaryCore(
            _compact_copy(core.q_rows[:rank]),
            core.left_dimension,
            core.right_dimension,
            core.kind,
            core.binary_exponent,
        )
    raise TypeError("production compression does not accept an explicit clone core")


def _slice_input(core: ImplicitCore, role: int, rank: int) -> ImplicitCore:
    if isinstance(core, UnaryCore):
        if role != 0:
            raise ValueError("unary core has only one input occurrence")
        return UnaryCore(_compact_copy(core.matrix[:, :rank]), core.kind, core.binary_exponent)
    if isinstance(core, CPBinaryCore):
        if role not in (0, 1):
            raise ValueError("binary core input occurrence is invalid")
        return CPBinaryCore(
            core.output_factor,
            _compact_copy(core.left_factor[:, :rank]) if role == 0 else core.left_factor,
            _compact_copy(core.right_factor[:, :rank]) if role == 1 else core.right_factor,
            core.kind,
            core.binary_exponent,
            None,
        )
    if isinstance(core, ReducedQRBinaryCore):
        if role not in (0, 1):
            raise ValueError("binary core input occurrence is invalid")
        tensor = core.q_rows.reshape(
            core.output_dimension, core.left_dimension, core.right_dimension
        )
        if role == 0:
            tensor = tensor[:, :rank, :]
        else:
            tensor = tensor[:, :, :rank]
        return ReducedQRBinaryCore(
            tensor.contiguous().reshape(tensor.shape[0], -1),
            int(tensor.shape[1]),
            int(tensor.shape[2]),
            core.kind,
            core.binary_exponent,
        )
    raise TypeError("production compression does not accept an explicit clone core")


def _compact_core_storage(core: ImplicitCore) -> ImplicitCore:
    if isinstance(core, UnaryCore):
        return UnaryCore(_compact_copy(core.matrix), core.kind, core.binary_exponent)
    if isinstance(core, CPBinaryCore):
        return CPBinaryCore(
            _compact_copy(core.output_factor),
            _compact_copy(core.left_factor),
            _compact_copy(core.right_factor),
            core.kind,
            core.binary_exponent,
            None,
        )
    if isinstance(core, ReducedQRBinaryCore):
        return ReducedQRBinaryCore(
            _compact_copy(core.q_rows),
            core.left_dimension,
            core.right_dimension,
            core.kind,
            core.binary_exponent,
        )
    raise TypeError("production compression does not accept an explicit clone core")


@dataclass(frozen=True)
class PrefixTruncationResult:
    network: ImplicitProjectiveDAG
    original_storage_elements: int
    truncated_storage_elements: int
    allocated_storage_elements: int
    reduced_bonds: int
    original_bond_dimensions: int
    retained_bond_dimensions: int
    expected_occurrence_slices: int
    applied_occurrence_slices: int


@torch.no_grad()
def truncate_diagonal_prefixes(
    diagonal: DiagonalImplicitDAG,
    plan: Mapping[int, int],
    *,
    copy_network: bool = True,
) -> PrefixTruncationResult:
    """Physically slice each selected bond and every one of its occurrences."""

    source = _certified_network(diagonal)
    _validate_plan(source, plan)
    original_storage = implicit_storage_elements(source)
    projected_storage = projected_prefix_storage_elements(diagonal, plan)
    work = _clone_network(source, unfold=False) if copy_network else source
    nodes = _validate_plan(
        work,
        plan,
        enforce_network_binding=not copy_network,
    )
    parents = _parent_occurrences(work.root)
    expected = 0
    applied = 0
    reduced = 0
    original_dimension_sum = 0
    retained_dimension_sum = 0
    for node in reversed(nodes):
        original_dimension = node.output_dimension
        rank = _rank_for(plan, node)
        original_dimension_sum += original_dimension
        retained_dimension_sum += rank
        if rank == original_dimension:
            continue
        reduced += 1
        node.core = _slice_output(node.core, rank)
        occurrences = parents.get(id(node), [])
        if occurrences:
            expected += len(occurrences)
            for parent, role in occurrences:
                parent.core = _slice_input(parent.core, role, rank)
                applied += 1
        else:
            if node is not work.root:
                raise ValueError("a non-root truncated bond has no parent occurrence")
            expected += 1
            work.head = _compact_copy(work.head[:, :rank])
            applied += 1
    if applied != expected:
        raise RuntimeError("prefix truncation did not slice every shared occurrence")
    work.algorithm1_direct_rq_complete = False
    work.algorithm1_scale_ledger_complete = False
    work.algorithm1_certified_head_binary_exponent = None
    work.head = _compact_copy(work.head)
    for node in nodes:
        node.core = _compact_core_storage(node.core)
    _validate_network(work)
    truncated_storage = implicit_storage_elements(work)
    if truncated_storage != projected_storage:
        raise RuntimeError("projected and materialized prefix storage disagree")
    allocated_storage = implicit_allocated_storage_elements(work)
    if allocated_storage != projected_storage:
        raise RuntimeError("allocated and projected prefix storage disagree")
    return PrefixTruncationResult(
        network=work,
        original_storage_elements=original_storage,
        truncated_storage_elements=truncated_storage,
        allocated_storage_elements=allocated_storage,
        reduced_bonds=reduced,
        original_bond_dimensions=original_dimension_sum,
        retained_bond_dimensions=retained_dimension_sum,
        expected_occurrence_slices=expected,
        applied_occurrence_slices=applied,
    )


@torch.no_grad()
def _evaluate_diagonal_intervals(
    diagonal: DiagonalImplicitDAG,
    raw_input: RawPhysicalInput,
    plan: Mapping[int, int],
    *,
    leading: bool,
) -> Tensor:
    network = _certified_network(diagonal)
    _validate_plan(network, plan)
    batch_size, prepared = _prepare_physical_input(network, raw_input)
    memo: dict[int, Tensor] = {}

    def evaluate(node: ImplicitNode) -> Tensor:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        if node.children:
            value = _core_apply(node.core, tuple(evaluate(child) for child in node.children))
        else:
            if node.physical_source_key is not None:
                if not isinstance(prepared, dict):
                    raise ValueError("heterogeneous leaf received homogeneous input")
                raw_value = prepared[node.physical_source_key]
            else:
                if node.physical_token is None or not isinstance(prepared, Tensor):
                    raise ValueError("implicit leaf has no physical token")
                raw_value = prepared[:, node.physical_token]
            physical = torch.cat(
                (raw_value, raw_value.new_ones(batch_size, 1)), dim=1
            )
            value = _core_apply(node.core, (physical,))
        rank = _rank_for(plan, node)
        if rank < node.output_dimension:
            value = value.clone()
            if leading:
                value[:, rank:] = 0
            else:
                value[:, : node.output_dimension - rank] = 0
        result = _normalize_projective_batch(value)
        memo[id(node)] = result
        return result

    root = evaluate(network.root)
    return _normalize_projective_batch(root @ network.head.T)


@torch.no_grad()
def evaluate_diagonal_prefixes(
    diagonal: DiagonalImplicitDAG,
    raw_input: RawPhysicalInput,
    plan: Mapping[int, int],
) -> Tensor:
    """Evaluate all leading-coordinate projections without copying the DAG."""

    return _evaluate_diagonal_intervals(diagonal, raw_input, plan, leading=True)


@torch.no_grad()
def evaluate_diagonal_suffixes(
    diagonal: DiagonalImplicitDAG,
    raw_input: RawPhysicalInput,
    plan: Mapping[int, int],
) -> Tensor:
    """Evaluate matched-width trailing coordinates as a negative control."""

    return _evaluate_diagonal_intervals(diagonal, raw_input, plan, leading=False)


@torch.no_grad()
def evaluate_diagonal_prefix_quotient(
    diagonal: DiagonalImplicitDAG,
    raw_input: RawPhysicalInput,
    plan: Mapping[int, int],
) -> Tensor:
    pair = evaluate_diagonal_prefixes(diagonal, raw_input, plan)
    denominator = pair[:, -1]
    relative = denominator.abs() / pair.abs().amax(dim=1).clamp_min(
        torch.finfo(pair.dtype).tiny
    )
    if float(relative.min().item()) <= 100.0 * torch.finfo(pair.dtype).eps:
        raise ValueError("prefix evaluation encountered a numerically zero denominator")
    return pair[:, :-1] / denominator[:, None]


@torch.no_grad()
def evaluate_diagonal_suffix_quotient(
    diagonal: DiagonalImplicitDAG,
    raw_input: RawPhysicalInput,
    plan: Mapping[int, int],
) -> Tensor:
    pair = evaluate_diagonal_suffixes(diagonal, raw_input, plan)
    denominator = pair[:, -1]
    relative = denominator.abs() / pair.abs().amax(dim=1).clamp_min(
        torch.finfo(pair.dtype).tiny
    )
    if float(relative.min().item()) <= 100.0 * torch.finfo(pair.dtype).eps:
        raise ValueError("suffix evaluation encountered a numerically zero denominator")
    return pair[:, :-1] / denominator[:, None]


__all__ = [
    "CompactRankBank",
    "CompactRankPlan",
    "PrefixTruncationResult",
    "ScaledPositiveSummary",
    "evaluate_diagonal_prefix_quotient",
    "evaluate_diagonal_prefixes",
    "evaluate_diagonal_suffix_quotient",
    "evaluate_diagonal_suffixes",
    "implicit_allocated_storage_elements",
    "implicit_storage_elements",
    "projected_prefix_storage_elements",
    "truncate_diagonal_prefixes",
]
