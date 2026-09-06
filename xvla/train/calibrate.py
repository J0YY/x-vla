"""RmsBatchNorm calibration (spec §7.5 / Stage 6).

After (or during) training the EMA running-RMS estimates are unreliable — early
training contaminates them — so before any faithful eval or tensor-network
export we re-estimate every RBN's scalar constant fresh on a representative
calibration set, then freeze. The deployed graph divides by these frozen
constants (folded into neighbours), so eval here faithfully mirrors inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn

from xvla.nn.normalization import RmsBatchNorm


@dataclass(frozen=True)
class SequentialCalibrationSite:
    """One live scalar normalization calibrated in execution order."""

    path: str
    scale: float
    batches: int


@dataclass(frozen=True)
class SequentialCalibrationReport:
    """Auditable record of a topological scalar-RBN calibration pass."""

    sites: tuple[SequentialCalibrationSite, ...]
    excluded_paths: tuple[str, ...]


@torch.no_grad()
def calibrate_rbn(model: nn.Module, batch_iter, iters: int = 50) -> None:
    """Freshly estimate every RmsBatchNorm running constant.

    Args:
        model: model containing RmsBatchNorm modules.
        batch_iter: callable returning an ``(x, y)`` batch each call.
        iters: number of calibration batches to average over.
    """
    rbns = [m for m in model.modules() if isinstance(m, RmsBatchNorm)]
    if not rbns:
        return
    was_training = model.training
    model.eval()  # disable dropout etc; RBN.calibrating overrides eval behaviour
    for m in rbns:
        m.start_calibration()
    for _ in range(iters):
        x, y = batch_iter()
        model(x)
    for m in rbns:
        m.finish_calibration()
    if was_training:
        model.train()
    # Re-open EMA updates so training can resume after a mid-run calibration eval.
    for m in rbns:
        m.frozen = False


@torch.no_grad()
def calibrate_rbn_sequential(
    model: nn.Module,
    batch_at: Callable[[int], Any],
    forward_batch: Callable[[nn.Module, Any], Any],
    *,
    iters: int = 50,
    excluded_paths: Sequence[str] = (),
) -> SequentialCalibrationReport:
    """Calibrate deployed RBNs one at a time in module execution order.

    Simultaneously putting every RBN in calibration mode makes downstream
    sites observe input-dependent batch normalization even though deployment
    uses the final frozen upstream constants. This routine avoids that
    mismatch. Earlier sites are frozen before later sites are measured. The
    same indexed calibration batches can be replayed at every site through
    ``batch_at(index)``.

    ``excluded_paths`` is only for modules proven absent from the deployed
    graph. Exclusion is exact-name based and every requested path must exist.
    The routine intentionally leaves all live sites initialized and frozen,
    which is the state required by strict export.
    """

    if iters <= 0:
        raise ValueError("iters must be positive")
    excluded = tuple(excluded_paths)
    if len(set(excluded)) != len(excluded):
        raise ValueError("excluded_paths contains duplicates")
    named = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, RmsBatchNorm)
    }
    missing = sorted(set(excluded) - set(named))
    if missing:
        raise ValueError(f"excluded RBN paths do not exist: {missing}")
    live_by_name = {name: module for name, module in named.items() if name not in excluded}
    live = list(live_by_name.items())
    model.eval()
    for _, module in live:
        module.calibrating = False
        module.frozen = True

    # Discover the actual first-execution order. Registration order is not an
    # execution-order guarantee for a generic nn.Module.
    observed: list[str] = []
    handles = []
    for path, module in named.items():
        handles.append(module.register_forward_pre_hook(lambda _module, _inputs, name=path: observed.append(name)))
    try:
        forward_batch(model, batch_at(0))
    finally:
        for handle in handles:
            handle.remove()
    executed_excluded = sorted(set(observed) & set(excluded))
    if executed_excluded:
        raise ValueError(f"excluded RBN executed during calibration dry run: {executed_excluded}")
    duplicates = sorted({path for path in observed if observed.count(path) > 1})
    if duplicates:
        raise ValueError(f"RBN sites execute more than once per VLA forward: {duplicates}")
    missing_live = sorted(set(live_by_name) - set(observed))
    if missing_live:
        raise ValueError(
            "live RBN sites were not observed during calibration dry run: "
            f"{missing_live}; prove them dead and exclude them explicitly"
        )
    if len(observed) != len(live_by_name):
        raise ValueError("RBN execution order is conditional or ambiguous")
    if not live_by_name:
        return SequentialCalibrationReport(sites=(), excluded_paths=excluded)
    live = [(path, live_by_name[path]) for path in observed]

    records: list[SequentialCalibrationSite] = []
    for path, module in live:
        module.start_calibration()
        try:
            for index in range(iters):
                forward_batch(model, batch_at(index))
        finally:
            module.finish_calibration()
        if int(module._calib_count.item()) != iters:
            raise RuntimeError(
                f"{path} executed {int(module._calib_count.item())} times, expected {iters}"
            )
        scale = module.scale.detach()
        if not bool(torch.isfinite(scale)) or float(scale.item()) <= 0:
            raise ValueError(f"{path} produced an invalid calibration scale")
        records.append(
            SequentialCalibrationSite(path=path, scale=float(scale.cpu().item()), batches=iters)
        )

    return SequentialCalibrationReport(sites=tuple(records), excluded_paths=excluded)
