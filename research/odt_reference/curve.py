"""Physical direction curves for one fixed, QR-canonical reference graph.

The x-axis counts unique nonleaf/nonroot output bonds after shape-only QR.
Each budget removes an integer number of directions, never storage elements.
Leading, trailing and direct-QR random controls use exactly the same ranks.
These helpers do not attest checkpoint identity, capability or clone agreement.
"""
from dataclasses import dataclass
import math

import numpy as np

from research.odt_reference.dooms import rq
from research.odt_reference.shared_dag import apply_bases


def _signature(graph):
    return tuple((node.name, node.children, node.source, node.core.shape)
                 for node in graph.nodes)


@dataclass(frozen=True)
class RankPlan:
    percent: int
    ranks: tuple
    widths: tuple
    eligible: tuple
    signature: tuple
    original_dimensions: int
    removed_dimensions: int

    @property
    def actual_fraction(self):
        return self.removed_dimensions / self.original_dimensions

    @property
    def retained_dimensions(self):
        return self.original_dimensions - self.removed_dimensions


@dataclass(frozen=True)
class DimensionSchedule:
    plans: tuple
    raw_internal_dimensions: int
    canonical_internal_dimensions: int
    raw_all_dimensions: int
    canonical_all_dimensions: int
    eligible_bonds: int
    minimum_rank: int = 1


@dataclass(frozen=True)
class _Event:
    numerator: int
    denominator: int
    node: int

    def __lt__(self, other):
        left, right = self.numerator * other.denominator, other.numerator * self.denominator
        return left < right if left != right else self.node < other.node


def rank_schedule(canonical, raw_graph=None, percents=(0, 30, 40, 50, 60, 70, 80)):
    """Freeze widths and exact nested integer budgets, without inspecting data.

    Remove direction j from width d at threshold (2j-1)/(2d). Exact integer
    cross-products order thresholds, and node order breaks simultaneous events.
    Percentage budgets round to nearest integer, with half-to-even ties.
    """
    if not canonical.canonical:
        raise ValueError("rank scheduling requires an accepted canonical graph")
    percents = tuple(percents)
    if (not percents or any(type(p) is not int or not 0 <= p < 100 for p in percents)
            or tuple(sorted(set(percents))) != percents):
        raise ValueError("percentages must be unique ascending integers in [0,100)")
    widths = tuple(node.core.shape[0] for node in canonical.nodes)
    eligible = tuple(i for i, node in enumerate(canonical.nodes)
                     if node.children and i != len(widths) - 1)
    if not eligible:
        raise ValueError("no eligible internal bonds")
    total = sum(widths[i] for i in eligible)
    event_count = total - len(eligible)
    if event_count > 2_000_000:
        raise ValueError("bounded reference rank schedule exceeds two million events")
    events = sorted(_Event(2 * j - 1, 2 * widths[i], i)
                    for i in eligible for j in range(1, widths[i]))
    ranks, used, plans = list(widths), 0, []
    for percent in percents:
        budget, remainder = divmod(total * percent, 100)
        budget += int(remainder > 50 or (remainder == 50 and budget % 2 == 1))
        if budget > len(events):
            raise ValueError("requested dimension budget is unattainable with rank at least one")
        while used < budget:
            ranks[events[used].node] -= 1
            used += 1
        plans.append(RankPlan(percent, tuple(ranks), widths, eligible,
                              _signature(canonical), total, budget))
    raw = canonical if raw_graph is None else raw_graph
    if (len(raw.nodes) != len(canonical.nodes)
            or any((a.name, a.children, a.source) != (b.name, b.children, b.source)
                   for a, b in zip(raw.nodes, canonical.nodes))):
        raise ValueError("raw and canonical graph topology differs")
    raw_widths = tuple(node.core.shape[0] for node in raw.nodes)
    return DimensionSchedule(tuple(plans), sum(raw_widths[i] for i in eligible),
                             total, sum(raw_widths), sum(widths), len(eligible))


@dataclass(frozen=True)
class PhysicalVariant:
    graph: object
    full_bases: tuple
    selected_bases: tuple
    plan: RankPlan
    mode: str
    seed: int


def physical_variant(canonical, full_bases, plan, mode="leading", *, seed=0):
    """Actually contract retained columns into producers and EVERY consumer.

    Reusing a seed freezes one random full basis across the entire rank ladder.
    Full-width ingress/root bases remain the supplied spectral bases in all
    controls. No basis is selected using replay inputs or desired outcomes.
    """
    if not canonical.canonical or not isinstance(plan, RankPlan) or plan.signature != _signature(canonical):
        raise ValueError("rank plan must match this canonical graph")
    widths = tuple(node.core.shape[0] for node in canonical.nodes)
    eligible = tuple(i for i, node in enumerate(canonical.nodes)
                     if node.children and i != len(widths) - 1)
    if (plan.widths != widths or plan.eligible != eligible or len(plan.ranks) != len(widths)
            or any(type(r) is not int or not 1 <= r <= w for r, w in zip(plan.ranks, widths))
            or any(plan.ranks[i] != w for i, w in enumerate(widths) if i not in eligible)
            or plan.original_dimensions != sum(widths[i] for i in eligible)
            or plan.removed_dimensions != sum(widths[i] - plan.ranks[i] for i in eligible)):
        raise ValueError("rank plan dimensions or budget are inconsistent")
    if mode not in ("leading", "trailing", "random") or type(seed) is not int or seed < 0:
        raise ValueError("explicit leading/trailing/random mode and nonnegative integer seed required")
    full_bases = tuple(np.asarray(value) for value in full_bases)
    if len(full_bases) != len(plan.widths):
        raise ValueError("one full spectral basis required per node")
    for basis, width in zip(full_bases, plan.widths):
        if basis.shape != (width, width) or np.iscomplexobj(basis) or not np.isfinite(basis).all():
            raise ValueError("finite real square full basis required")
    rng, eligible = np.random.default_rng(seed), set(plan.eligible)
    controls = []
    for index, (basis, width) in enumerate(zip(full_bases, plan.widths)):
        if index in eligible and mode == "trailing":
            basis = basis[:, ::-1]
        elif index in eligible and mode == "random":
            _, rows = rq(rng.normal(size=(width, width)))
            basis = rows.T
        controls.append(np.array(basis, copy=True))
    selected = tuple(np.array(basis[:, :rank], copy=True)
                     for basis, rank in zip(controls, plan.ranks))
    result = apply_bases(canonical, selected)
    for index, node in enumerate(result.nodes):
        if not np.isfinite(node.core).all():
            raise RuntimeError("physical contraction produced nonfinite core entries")
        expected = (plan.ranks[index],) + tuple(plan.ranks[c] for c in node.children)
        if node.children and node.core.shape != expected:
            raise RuntimeError("physical contraction did not narrow every incident axis")
        if not node.children and node.core.shape[0] != plan.ranks[index]:
            raise RuntimeError("physical leaf dimension differs")
    if not np.isfinite(result.head).all():
        raise RuntimeError("physical contraction produced a nonfinite head")
    return PhysicalVariant(result, tuple(controls), selected, plan, mode, seed)


@dataclass(frozen=True)
class ChartEvaluation:
    projective: np.ndarray
    valid_rows: tuple
    relative_denominators: tuple
    failure_reasons: tuple
    decoded: object


def masked_evaluate(graph, inputs, ranks=None, *, denominator_margin=1e-12):
    """Independent gauge-space zero masks, with honest zero/invalid charts.

    Inputs map physical-source names to homogeneous vectors or batches. The
    returned decoded array is None if ANY row is invalid. Exact zero internal
    vectors propagate as zero, never as fabricated homogeneous constants.
    Use no ranks to execute an already physically narrowed graph.
    """
    if (isinstance(denominator_margin, bool) or not isinstance(denominator_margin, (int, float))
            or not math.isfinite(denominator_margin) or not 0 <= denominator_margin < 1):
        raise ValueError("finite denominator margin in [0,1) required")
    widths = tuple(node.core.shape[0] for node in graph.nodes)
    ranks = widths if ranks is None else tuple(ranks)
    if len(ranks) != len(widths) or any(type(r) is not int or not 1 <= r <= d for r, d in zip(ranks, widths)):
        raise ValueError("one valid positive rank required per node")
    sources = {node.source: node.core.shape[1] for node in graph.nodes if not node.children}
    if not isinstance(inputs, dict) or set(inputs) != set(sources):
        raise ValueError("physical source keys differ")
    batches = {}
    for name, width in sources.items():
        value = np.asarray(inputs[name])
        if value.ndim == 1:
            value = value[None]
        if (np.iscomplexobj(value) or value.ndim != 2 or value.shape[1] != width
                or not value.shape[0] or not np.isfinite(value).all()):
            raise ValueError("finite real homogeneous physical input batches required")
        batches[name] = value
    if len({len(value) for value in batches.values()}) != 1 or graph.head.shape[0] < 2:
        raise ValueError("equal input batches and projective output width at least two required")
    rows, valid, margins, failures = [], [], [], []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for sample in range(len(next(iter(batches.values())))):
            values, finite = [], True
            for index, node in enumerate(graph.nodes):
                if not node.children:
                    value = node.core @ batches[node.source][sample]
                elif len(node.children) == 1:
                    value = node.core @ values[node.children[0]]
                else:
                    value = np.einsum("oij,i,j->o", node.core,
                                      values[node.children[0]], values[node.children[1]])
                finite = finite and bool(np.isfinite(value).all())
                value[ranks[index]:] = 0.
                size = float(np.max(np.abs(value)))
                values.append(value / size if math.isfinite(size) and size > 0 else value)
            output = graph.head @ values[-1]
            finite = finite and bool(np.isfinite(output).all())
            size = float(np.max(np.abs(output)))
            output = output / size if finite and size > 0 else output
            margin = float(abs(output[-1])) if finite and size > 0 else (0. if finite else None)
            reason = ("nonfinite_contraction" if not finite else "zero_projective_row" if size == 0
                      else "denominator_margin" if margin <= denominator_margin else None)
            rows.append(output)
            valid.append(reason is None)
            margins.append(margin)
            failures.append(reason)
        projective = np.array(rows)
        decoded = projective[:, :-1] / projective[:, -1:] if all(valid) else None
        if decoded is not None and not np.isfinite(decoded).all():
            for index, row in enumerate(decoded):
                if not np.isfinite(row).all():
                    valid[index], failures[index] = False, "nonfinite_decoded_output"
            decoded = None
    return ChartEvaluation(projective, tuple(valid), tuple(margins), tuple(failures), decoded)
