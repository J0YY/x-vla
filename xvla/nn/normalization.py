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


# Checked directly against all 74 Padé numerator/denominator buffer pairs in
# ``ckpt_linear_rat_vit_s0_v2.pt`` (SHA-256
# 96f11093701d6b52deefb50b7921b46e2c987e5b9dbce947997882f7c59da6c9).
# Keeping the rounded float32 values here preserves the checkpoint's exact IEEE
# bit patterns while ensuring model construction performs no numerical fit.
_DEFAULT_PADE_NUMERATOR = (
    5.3299665451049805,
    7.535519123077393,
    0.32727372646331787,
)
_DEFAULT_PADE_DENOMINATOR = (
    1.0,
    9.648269653320312,
    2.606637477874756,
)


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


class RationalNorm(nn.Module):
    """Per-instance normalization that stays inside a FOLDABLE tensor-network class.

    The strict foldable model used a frozen SCALAR (RmsBatchNorm), which cannot capture
    per-instance magnitude variation (a real capability tax). This recovers exact
    per-instance normalization while remaining foldable, via the rational-TN result
    (DEVLOG cont.31 / rational_norm_proto.py):

      * ``variant="meansq"`` : n(x) = x / (mean(x²)+ε).  Exactly RATIONAL (no √) →
        folds projectively to P(x)/Q(x).  But it inverts magnitude (out-RMS ∝ 1/in-RMS),
        so it is a different, more aggressive normalization than RMSNorm.
      * ``variant="nr_rsqrt"`` (default) : x · y, where y approximates 1/√(mean(x²)) by
        ``nr_steps`` Newton-Raphson iterations  y ← y·(1.5 − 0.5·ms·y²)  from a FROZEN
        constant initial guess y₀ = 1/√s₀ (s₀ = running mean-square, EMA in train, frozen
        at export).  Each step is polynomial in ms (hence in x), so the map folds as an
        ordinary polynomial TN; NR converges quadratically to true 1/√, so it BEHAVES like
        RMSNorm (unit-RMS) — capability should match, unlike meansq or the frozen scalar.
        Accurate across the measured thin activation tail (ρ≈2.8, DEVLOG Finding 18).

    decode/export note: with s₀ frozen the output is a fixed polynomial in x → foldable;
    ``running_ms`` is the only state, mirroring RmsBatchNorm's running scale.
    """

    def __init__(self, variant: str = "pade", nr_steps: int = 2,
                 momentum: float = 0.99, eps: float = 1e-6,
                 v_lo: float = 0.1, v_hi: float = 10.0, deg: int = 2,
                 pade_numerator=None, pade_denominator=None):
        super().__init__()
        if variant not in ("nr_rsqrt", "meansq", "pade"):
            raise ValueError(f"RationalNorm variant must be 'pade', 'nr_rsqrt' or 'meansq', got {variant!r}")
        if not isinstance(deg, int) or isinstance(deg, bool) or deg < 0:
            raise ValueError(f"Padé degree must be a nonnegative integer, got {deg!r}")
        self.variant = variant
        self.nr_steps = nr_steps
        self.momentum = momentum
        self.eps = eps
        self.deg = deg
        self.register_buffer("running_ms", torch.ones(()))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.frozen = False
        if variant == "pade":
            supplied = pade_numerator is not None or pade_denominator is not None
            if supplied and (pade_numerator is None or pade_denominator is None):
                raise ValueError(
                    "custom Padé coefficients require both numerator and denominator"
                )
            if supplied:
                numerator = torch.as_tensor(pade_numerator, dtype=torch.float32)
                denominator = torch.as_tensor(pade_denominator, dtype=torch.float32)
            elif deg == 2 and float(v_lo) == 0.1 and float(v_hi) == 10.0:
                numerator = torch.tensor(_DEFAULT_PADE_NUMERATOR, dtype=torch.float32)
                denominator = torch.tensor(_DEFAULT_PADE_DENOMINATOR, dtype=torch.float32)
            else:
                raise ValueError(
                    "no checked-in Padé coefficients exist for this configuration; "
                    "supply explicit pade_numerator and pade_denominator values"
                )
            expected = deg + 1
            if numerator.ndim != 1 or denominator.ndim != 1:
                raise ValueError("Padé coefficients must be one-dimensional")
            if numerator.numel() != expected or denominator.numel() != expected:
                raise ValueError(f"degree-{deg} Padé coefficients must have {expected} entries")
            if not bool(torch.isfinite(numerator).all() and torch.isfinite(denominator).all()):
                raise ValueError("Padé coefficients must be finite")
            if float(denominator[0]) == 0.0:
                raise ValueError("Padé denominator constant coefficient must be nonzero")
            self.register_buffer("pa", numerator.clone())
            self.register_buffer("pb", denominator.clone())
        elif pade_numerator is not None or pade_denominator is not None:
            raise ValueError("Padé coefficients require variant='pade'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve float64 for algebraic reference/certification runs while
        # retaining the historical float32 compute path for deployed fp32 and
        # lower-precision inputs.
        xf = x if x.dtype in (torch.float32, torch.float64) else x.float()
        ms = xf.pow(2).mean(dim=-1, keepdim=True) + self.eps          # (..., 1) per-token mean-square
        if self.variant == "meansq":
            return (xf / ms).to(x.dtype)
        # running mean-square scale s₀ (EMA in train, frozen at export → a constant)
        if self.training and not self.frozen:
            with torch.no_grad():
                bm = ms.mean()
                if not bool(self.initialized):
                    self.running_ms.copy_(bm); self.initialized.fill_(True)
                else:
                    self.running_ms.mul_(self.momentum).add_((1.0 - self.momentum) * bm)
        s0 = self.running_ms.clamp_min(1e-12)
        if self.variant == "pade":
            v = ms / s0                                               # normalized mean-square
            P = sum(self.pa[k] * v ** k for k in range(self.deg + 1))
            Q = sum(self.pb[k] * v ** k for k in range(self.deg + 1))
            inv_sqrt = (P / Q) * torch.rsqrt(s0)                      # ≈ 1/√ms, rational, bounded
            return (xf * inv_sqrt).to(x.dtype)
        # nr_rsqrt: polynomial (Newton) approx of 1/√ms — accurate only near s₀ (tail diverges)
        y = torch.rsqrt(s0).expand_as(ms).clone()
        for _ in range(self.nr_steps):
            y = y * (1.5 - 0.5 * ms * y * y)
        return (xf * y).to(x.dtype)

    @torch.no_grad()
    def freeze(self) -> None:
        self.frozen = True

    @property
    def scale(self) -> torch.Tensor:
        """Frozen scale s₀ (the NR initial-guess anchor); the fold uses the polynomial in ms."""
        return self.running_ms + self.eps


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
    if mode in ("rational", "pade"):
        return RationalNorm(variant="pade", momentum=momentum, eps=eps)
    if mode == "nr_rsqrt":
        return RationalNorm(variant="nr_rsqrt", momentum=momentum, eps=eps)
    if mode == "meansq":
        return RationalNorm(variant="meansq", momentum=momentum, eps=eps)
    if mode == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm mode {mode!r}")
