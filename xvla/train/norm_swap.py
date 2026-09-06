"""Swap a trained model's normalization to the foldable scalar norm, safely.

ODT needs the deployed map to be a *polynomial*. Two of the norm modes this repo
trains with are not:

* ``per_token`` (:class:`~xvla.nn.normalization.PerTokenRmsNorm`) divides by a
  genuine per-instance ``sqrt``, which is not even rational.
* ``rational`` (:class:`~xvla.nn.normalization.RationalNorm`, Pade variant) is a
  true rational function ``P(x)/Q(x)``. It *folds*, but it is not ODT-compatible:
  the mean-square is taken over the whole bond, so ``P/Q`` depends on exactly the
  coordinates a bond truncation is about to discard. After truncation the map is
  no longer a function of the retained subspace, and the Gram tail does not bound
  the error. ``canonical_odt.export_homogeneous_network`` refuses it for this
  reason, accepting only ``RmsBatchNorm`` or ``Identity``.

So a decomposable policy has to run on ``scalar_rbn``, a frozen scalar divide
that folds exactly into its neighbours. The cost of that change has never been
measured closed-loop in this repo: the ``product_strict`` run that was meant to
measure it never returned a result.

This module performs the swap and, critically, *proves it did not silently lose
weights*. No norm module in :mod:`xvla.nn.normalization` has learned parameters,
only buffers, so the swap is weight-preserving by construction. That is exactly
the situation where a ``strict=False`` load quietly drops a real tensor and the
resulting capability number is garbage, so the transfer is validated rather than
trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from xvla.nn.normalization import HomotopyNorm, PerTokenRmsNorm, RationalNorm, RmsBatchNorm

# Buffer leaf-names owned by each norm implementation. A key that is missing or
# unexpected is only forgivable if its leaf name is one of these.
_NORM_BUFFER_LEAVES = frozenset({
    "running_rms", "running_ms", "initialized", "kappa", "last_tail",
    "_calib_sum", "_calib_count", "pa", "pb",
})

_SCALAR_NORM_TYPES = (RmsBatchNorm,)
_NONPOLYNOMIAL_NORM_TYPES = (PerTokenRmsNorm, RationalNorm, HomotopyNorm)


@dataclass
class NormSwapReport:
    """Auditable record of a normalization swap."""

    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()
    transferred_tensors: int = 0
    transferred_elements: int = 0
    scalar_norm_sites: tuple[str, ...] = ()
    remaining_nonpolynomial_sites: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return dict(
            missing_keys=list(self.missing_keys),
            unexpected_keys=list(self.unexpected_keys),
            transferred_tensors=self.transferred_tensors,
            transferred_elements=self.transferred_elements,
            scalar_norm_sites=list(self.scalar_norm_sites),
            remaining_nonpolynomial_sites=list(self.remaining_nonpolynomial_sites),
        )


def _leaf(key: str) -> str:
    return key.rsplit(".", 1)[-1]


def swap_norm_weights(dst: nn.Module, src_state: dict, *, require_polynomial: bool = True
                      ) -> NormSwapReport:
    """Load ``src_state`` into ``dst`` and prove only norm buffers differed.

    Args:
        dst: freshly constructed model with the target norm mode.
        src_state: ``state_dict`` of the trained source model.
        require_polynomial: if True, raise unless every surviving norm site in
            ``dst`` is ODT-compatible (a scalar ``RmsBatchNorm`` or an identity).

    Raises:
        RuntimeError: if any missing or unexpected key is NOT a norm buffer, i.e.
            a real weight would have been silently dropped or ignored. Also raised
            when ``require_polynomial`` and a non-polynomial norm survives.
    """
    result = dst.load_state_dict(src_state, strict=False)
    missing = tuple(result.missing_keys)
    unexpected = tuple(result.unexpected_keys)

    bad_missing = [k for k in missing if _leaf(k) not in _NORM_BUFFER_LEAVES]
    bad_unexpected = [k for k in unexpected if _leaf(k) not in _NORM_BUFFER_LEAVES]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "normalization swap would drop non-norm tensors, refusing. "
            f"unfilled weights={bad_missing} ignored source weights={bad_unexpected}"
        )

    dst_keys = set(dst.state_dict())
    transferred = [k for k in src_state if k in dst_keys and _leaf(k) not in _NORM_BUFFER_LEAVES]
    n_elem = sum(src_state[k].numel() for k in transferred)

    scalar_sites, nonpoly_sites = [], []
    for name, mod in dst.named_modules():
        if isinstance(mod, _SCALAR_NORM_TYPES):
            scalar_sites.append(name)
        elif isinstance(mod, _NONPOLYNOMIAL_NORM_TYPES):
            nonpoly_sites.append(name)
    if require_polynomial and nonpoly_sites:
        raise RuntimeError(
            "target model still contains non-polynomial normalization, so it is not "
            f"ODT-compatible: {nonpoly_sites}"
        )

    return NormSwapReport(
        missing_keys=missing, unexpected_keys=unexpected,
        transferred_tensors=len(transferred), transferred_elements=n_elem,
        scalar_norm_sites=tuple(scalar_sites),
        remaining_nonpolynomial_sites=tuple(nonpoly_sites),
    )


@dataclass
class CalibrationReport:
    """Frozen scalar per exercised site, plus the sites the forward pass never hit."""

    frozen: dict[str, float] = field(default_factory=dict)
    unexercised: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return dict(frozen=dict(self.frozen), unexercised=list(self.unexercised))


@torch.no_grad()
def calibrate_scalar_norms(model: nn.Module, batch_iter, iters: int = 50) -> CalibrationReport:
    """Fresh-average calibration of every :class:`RmsBatchNorm`, then freeze.

    Uses the ``calibrating`` path (a plain mean of batch RMS, no EMA
    contamination), which is what spec 7.5 requires before an export.

    ``batch_iter`` is a callable returning a tuple of positional args for
    ``model(...)``, so the caller owns batching and device placement.

    Sites the forward pass never reaches are reported in ``unexercised`` and left
    at their loaded value rather than being given a fabricated constant. This is
    not hypothetical: ``ChiVLA._visual_tokens`` calls ``ChiViT.features()`` with
    ``apply_final_norm=False``, so ``vision.norm_out`` is dead in the VLA path.
    An export that folds such a site would be folding an uncalibrated ``1.0``, so
    the caller has to see it. Raises only if *nothing* was exercised, which means
    the batch or the model is wrong.
    """
    sites = [(n, m) for n, m in model.named_modules() if isinstance(m, RmsBatchNorm)]
    if not sites:
        return CalibrationReport()
    was_training = model.training
    model.train()
    for _, m in sites:
        m.calibrating, m.frozen = True, False
        m._calib_sum.zero_()
        m._calib_count.zero_()
    for _ in range(iters):
        model(*batch_iter())

    frozen, unexercised = {}, []
    for name, m in sites:
        if float(m._calib_count) <= 0:
            unexercised.append(name)
            m.calibrating = False
            continue
        c = float(m._calib_sum) / float(m._calib_count)
        m.running_rms.fill_(c)
        m.initialized.fill_(True)
        m.calibrating, m.frozen = False, True
        frozen[name] = c
    if not frozen:
        raise RuntimeError(
            f"calibration exercised none of the {len(sites)} scalar norm sites; "
            "check iters>0 and that batch_iter drives a real forward pass"
        )
    if not was_training:
        model.eval()
    return CalibrationReport(frozen=frozen, unexercised=tuple(unexercised))


@torch.no_grad()
def tail_ratio_report(model: nn.Module, batch_iter, iters: int = 10) -> dict[str, dict]:
    """Fold-safety diagnostic: ``max_i(perTokenRMS_i / c)`` at every scalar site.

    A frozen scalar divide equals the per-token divide it replaced only where this
    ratio is near 1. Large values are exactly where swapping the norm changes the
    function, so this predicts *before any rollout* whether the swap is safe, and
    localises the damage when it is not. This is the same quantity
    :class:`~xvla.nn.normalization.HomotopyNorm` tracks as ``last_tail``.
    """
    sites = [(n, m) for n, m in model.named_modules() if isinstance(m, RmsBatchNorm)]
    stats = {n: dict(max_ratio=0.0, mean_ratio=0.0, n_batches=0) for n, _ in sites}
    handles = []

    def make_hook(name: str, mod: RmsBatchNorm):
        def hook(_m, args, _out):
            x = args[0]
            c = float(mod.running_rms) + mod.eps
            pt = torch.sqrt(x.float().pow(2).mean(dim=-1).clamp_min(1e-12)) / c
            s = stats[name]
            s["max_ratio"] = max(s["max_ratio"], float(pt.max()))
            s["mean_ratio"] += float(pt.mean())
            s["n_batches"] += 1
        return hook

    for name, mod in sites:
        handles.append(mod.register_forward_hook(make_hook(name, mod)))
    was_training = model.training
    model.eval()
    try:
        for _ in range(iters):
            model(*batch_iter())
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()
    for s in stats.values():
        if s["n_batches"]:
            s["mean_ratio"] /= s["n_batches"]
    return stats
