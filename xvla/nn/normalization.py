"""Foldable RmsBatchNorm (spec §7).

Unlike LayerNorm/RMSNorm, this normalization divides by a *scalar* running RMS
magnitude estimated over the batch and tokens — never by a per-sample,
input-dependent statistic. After training the scalar is frozen and folded into
the neighbouring projection matrices, leaving **no division** in the exported
inference graph::

    c_t = ρ·c_{t-1} + (1-ρ)·sqrt( E_{batch,tokens}[ ||x||₂² / d ] )
    RBN(x) = x / (c_t + ε)

Folding (spec §7.2–7.3): because ``c`` is a scalar constant, ``RBN`` followed by
a linear map ``W`` is exactly ``W' = W / c`` with the module deleted. For a
bilinear branch, fold into the two input projections ``L' = L/c, R' = R/c``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RmsBatchNorm(nn.Module):
    """Scalar running-RMS normalization that folds into adjacent weights.

    Args:
        momentum: EMA decay ρ for the running RMS (spec recommends 0.99–0.999).
        eps: numerical floor ε.
    """

    def __init__(self, momentum: float = 0.99, eps: float = 1e-6):
        super().__init__()
        if not 0.0 < momentum < 1.0:
            raise ValueError(f"momentum must be in (0, 1), got {momentum}")
        self.momentum = momentum
        self.eps = eps
        # Scalar running RMS. Initialised to 1 so early forward passes are ~identity.
        self.register_buffer("running_rms", torch.ones(()))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        # When frozen (post-calibration) we never update the running statistic.
        self.frozen: bool = False
        # Fresh-average calibration state (spec §7.5), separate from the EMA.
        self.calibrating: bool = False
        self.register_buffer("_calib_sum", torch.zeros(()))
        self.register_buffer("_calib_count", torch.zeros(()))

    @staticmethod
    def _batch_rms(x: torch.Tensor) -> torch.Tensor:
        """sqrt( mean over all-but-last dims of (||x||₂²/d) ) as a scalar.

        Equivalent to the RMS over every element, computed in fp32 for stability.
        """
        return torch.sqrt(x.float().pow(2).mean().clamp_min(1e-12))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BatchNorm semantics: normalize by the *batch* RMS during training so
        # activations stay unit-scale as weights move (dividing by the lagging
        # EMA, as spec §7.1 literally reads, does not — it de-stabilizes the
        # degree-5 stack). The EMA running_rms is tracked for eval and is the
        # frozen constant folded into neighbours at export (spec §7.2–7.5), so
        # the deployed graph still contains no input-dependent division.
        if self.calibrating:
            # Fresh-average calibration (spec §7.5): accumulate a simple mean of
            # the batch RMS (no EMA contamination), still normalizing by the
            # batch so downstream layers see the calibration distribution.
            batch_rms = self._batch_rms(x)
            with torch.no_grad():
                self._calib_sum += batch_rms
                self._calib_count += 1
            return x / (batch_rms.to(x.dtype) + self.eps)
        if self.training and not self.frozen:
            batch_rms = self._batch_rms(x)
            with torch.no_grad():
                if not bool(self.initialized):
                    self.running_rms.copy_(batch_rms)
                    self.initialized.fill_(True)
                else:
                    self.running_rms.mul_(self.momentum).add_(
                        (1.0 - self.momentum) * batch_rms
                    )
            return x / (batch_rms.to(x.dtype) + self.eps)
        return x / (self.running_rms.to(x.dtype) + self.eps)

    @torch.no_grad()
    def calibrate(self, x: torch.Tensor) -> None:
        """Accumulate the running RMS on a calibration batch (spec §7.5).

        Uses the same EMA rule; call repeatedly over a representative
        calibration set with the module in eval() before freezing.
        """
        batch_rms = self._batch_rms(x)
        if not bool(self.initialized):
            self.running_rms.copy_(batch_rms)
            self.initialized.fill_(True)
        else:
            self.running_rms.mul_(self.momentum).add_(
                (1.0 - self.momentum) * batch_rms
            )

    @torch.no_grad()
    def start_calibration(self) -> None:
        """Begin a fresh running-RMS estimate (spec §7.5 step 2)."""
        self.calibrating = True
        self._calib_sum.zero_()
        self._calib_count.zero_()

    @torch.no_grad()
    def finish_calibration(self) -> None:
        """Set running_rms to the calibration average and freeze (spec §7.5)."""
        if float(self._calib_count) > 0:
            self.running_rms.copy_(self._calib_sum / self._calib_count)
            self.initialized.fill_(True)
        self.calibrating = False
        self.frozen = True

    @torch.no_grad()
    def freeze(self) -> None:
        """Stop updating the running statistic (prepare for folding)."""
        self.frozen = True

    @property
    def scale(self) -> torch.Tensor:
        """The scalar divisor ``c + ε`` folded into neighbours at export."""
        return self.running_rms + self.eps


class PerTokenRmsNorm(nn.Module):
    """Stateless per-token RMS normalization over the last dim (no learned gain).

    This is the *non-strict* stability control (spec §18 "per-instance L2
    normalization") — identical in train and eval, so it avoids the calibration
    mismatch that destabilizes the foldable scalar :class:`RmsBatchNorm`. It is
    NOT tensor-network-foldable (input-dependent division); use it to establish
    a stable, learning model, then migrate to ``RmsBatchNorm`` for the strict,
    exportable model.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True).to(x.dtype) + self.eps)


class HomotopyNorm(nn.Module):
    """Interpolates per-token ↔ foldable-scalar norm by a knob ``κ ∈ [0,1]``.

        n(x) = x / ( (1−κ)·perTokenRMS(x) + κ·c )      c = frozen scalar

    At ``κ=0`` this is exactly :class:`PerTokenRmsNorm` (stable, non-strict); at
    ``κ=1`` it is exactly ``x/c`` — the foldable scalar :class:`RmsBatchNorm`.
    Training starts at κ=0 (the stable per-token model, so no separate teacher is
    needed) and anneals κ→1; the gate advances κ only where the export is safe.

    ``last_tail`` records max_i(perTokenRMS_i / c) over the last forward — the
    tail ratio that governs whether κ=1 is safe here (κ=1 ≡ per-token iff this is
    ≈1 for every instance). At κ=1 the module is a fixed scalar divide and folds
    into adjacent weights exactly like :class:`RmsBatchNorm` (see ``fold.py``).
    """

    def __init__(self, momentum: float = 0.99, eps: float = 1e-6, kappa: float = 0.0):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_rms", torch.ones(()))
        self.register_buffer("kappa", torch.tensor(float(kappa)))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("last_tail", torch.ones(()))
        self.register_buffer("_calib_sum", torch.zeros(()))
        self.register_buffer("_calib_count", torch.zeros(()))
        self.frozen = False
        self.calibrating = False

    def _pertoken(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pt = self._pertoken(x)  # (..., 1) per-token divisor
        batch_scalar = pt.mean()
        if self.calibrating:
            with torch.no_grad():
                self._calib_sum += batch_scalar.float()
                self._calib_count += 1
        elif self.training and not self.frozen:
            with torch.no_grad():
                if not bool(self.initialized):
                    self.running_rms.copy_(batch_scalar); self.initialized.fill_(True)
                else:
                    self.running_rms.mul_(self.momentum).add_((1 - self.momentum) * batch_scalar)
        c = self.running_rms + self.eps
        with torch.no_grad():
            self.last_tail.copy_((pt.squeeze(-1) / c).max())
        k = self.kappa
        divisor = (1.0 - k) * pt + k * c.to(x.dtype)
        return x / divisor

    @torch.no_grad()
    def start_calibration(self):
        self.calibrating = True; self._calib_sum.zero_(); self._calib_count.zero_()

    @torch.no_grad()
    def finish_calibration(self):
        if float(self._calib_count) > 0:
            self.running_rms.copy_(self._calib_sum / self._calib_count)
            self.initialized.fill_(True)
        self.calibrating = False

    @torch.no_grad()
    def set_kappa(self, value: float):
        self.kappa.fill_(float(max(0.0, min(1.0, value))))

    @property
    def scale(self) -> torch.Tensor:
        """Foldable divisor at κ=1 (spec fold path). Valid only when κ==1."""
        return self.running_rms + self.eps


def make_norm(mode: str, momentum: float = 0.99, eps: float = 1e-6) -> nn.Module:
    """Factory for a normalization site.

    ``"per_token"`` → :class:`PerTokenRmsNorm` (stable, non-strict);
    ``"scalar_rbn"`` → :class:`RmsBatchNorm` (foldable, strict);
    ``"homotopy"`` → :class:`HomotopyNorm` (per-token↔scalar knob, κ=0 init);
    ``"none"`` → identity.
    """
    if mode == "per_token":
        return PerTokenRmsNorm(eps=eps)
    if mode == "scalar_rbn":
        return RmsBatchNorm(momentum=momentum, eps=eps)
    if mode == "homotopy":
        return HomotopyNorm(momentum=momentum, eps=eps)
    if mode == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm mode {mode!r}")
