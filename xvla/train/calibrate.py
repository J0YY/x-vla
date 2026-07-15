"""RmsBatchNorm calibration (spec §7.5 / Stage 6).

After (or during) training the EMA running-RMS estimates are unreliable — early
training contaminates them — so before any faithful eval or tensor-network
export we re-estimate every RBN's scalar constant fresh on a representative
calibration set, then freeze. The deployed graph divides by these frozen
constants (folded into neighbours), so eval here faithfully mirrors inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xvla.nn.normalization import RmsBatchNorm


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
