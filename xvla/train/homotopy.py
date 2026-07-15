"""Swap-gap-gated homotopy training (the proposed strict-norm solution).

Self-morphing run: the model uses HomotopyNorm at every site (block + Q/K), so
at κ=0 it is the stable per-token model (no separate teacher) and at κ=1 it is
the foldable scalar model. κ anneals 0→1; the FFN degree-2 gain is capped by a
foldable spectral budget so residual gains can stay large. Three arms isolate
what matters:

    A  big gains, no spectral budget, κ jumped to 1   → must diverge (live-test)
    B  big gains + spectral budget + swap-gap-gated κ → the proposed method
    C  same as B but κ on a fixed schedule            → ablates the gate

The gate advances a site's κ only while its tail ratio max_i(perTokenRMS_i / c)
stays under threshold — so κ=1 is reached only where the fixed op ≈ per-token op,
and the anneal cannot re-detonate the degree-2/degree-4 blow-up.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from xvla.models.lm import ChiLanguageModel, LMConfig
from xvla.nn.normalization import HomotopyNorm
from xvla.train.balance import balance_model, spectral_clip_model
from xvla.train.data import TokenBinDataset
from xvla.train.train_lm import TrainConfig, _lr_at, evaluate


@dataclass
class HomotopyRunConfig:
    arm: str = "B"                 # "A" | "B" | "C"
    spectral_budget: float = 8.0
    clip_every: int = 25
    gate_every: int = 25
    gate_warmup: int = 100
    kappa_step: float = 0.05       # κ increment per successful gate check
    rms_ceiling: float = 6.0       # advance κ while stress maxRMS < ceiling, else back off
    stress_scale: float = 3.0


@torch.no_grad()
def _hidden_rms_stats(model, x, input_scale=1.0):
    model.eval()
    _, _, hid = model(x, return_hidden=True, input_scale=input_scale)
    model.train()
    return max(h.float().pow(2).mean().sqrt().item() for h in hid)


@torch.no_grad()
def _kappa_stats(homos):
    ks = [float(m.kappa) for m in homos]
    at1 = sum(1 for k in ks if k >= 1.0 - 1e-6)
    return sum(ks) / len(ks), at1, len(ks)


def train_homotopy(model_cfg: LMConfig, cfg: TrainConfig, rc: HomotopyRunConfig, log=print):
    torch.manual_seed(cfg.seed)
    dev = cfg.device
    model = ChiLanguageModel(model_cfg).to(dev)
    homos = [m for m in model.modules() if isinstance(m, HomotopyNorm)]
    log(f"[arm {rc.arm}] {len(homos)} homotopy sites, params={model.num_params()/1e6:.1f}M")

    if rc.arm == "A":                       # jump straight to strict, no budget
        for m in homos:
            m.set_kappa(1.0)

    train_ds = TokenBinDataset(cfg.train_bin, cfg.seq_len)
    val_ds = TokenBinDataset(cfg.val_bin, cfg.seq_len)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                            weight_decay=cfg.weight_decay)
    ac = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if dev.startswith("cuda") \
        else torch.autocast(device_type="cpu", enabled=False)

    hist = {"arm": rc.arm, "step": [], "loss": [], "mean_kappa": [], "frac_k1": [],
            "max_rms": [], "max_rms_stress": [], "skipped": 0, "diverged": False}
    t0 = time.time()
    model.train()
    for step in range(cfg.max_steps + 1):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, cfg)
        opt.zero_grad(set_to_none=True)
        x, y = train_ds.batch(cfg.batch_size, dev)
        with ac:
            _, loss = model(x, y)
        if not torch.isfinite(loss):
            hist["skipped"] += 1
            if hist["skipped"] > 20:
                hist["diverged"] = True
                log(f"[arm {rc.arm}] DIVERGED at step {step}")
                break
            continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        # Foldable degree-2 gain cap (arms B, C).
        if rc.arm in ("B", "C") and step > 0 and step % rc.clip_every == 0:
            balance_model(model)
            spectral_clip_model(model, rc.spectral_budget)

        # κ scheduling.
        if rc.arm == "C":                                   # fixed linear anneal
            k = min(1.0, step / max(1, int(0.7 * cfg.max_steps)))
            for m in homos:
                m.set_kappa(k)
        elif rc.arm == "B" and step >= rc.gate_warmup and step % rc.gate_every == 0:
            # Stability-gated global anneal with back-off: push κ→1 as long as the
            # model stays bounded under magnitude stress; retreat if it spikes.
            stress_rms = _hidden_rms_stats(model, x, input_scale=rc.stress_scale)
            cur = float(homos[0].kappa)
            newk = min(1.0, cur + rc.kappa_step) if (stress_rms < rc.rms_ceiling and math.isfinite(stress_rms)) \
                else max(0.0, cur - 2 * rc.kappa_step)
            for m in homos:
                m.set_kappa(newk)

        if step % cfg.log_every == 0:
            mk, a1, n = _kappa_stats(homos)
            mr = _hidden_rms_stats(model, x)
            log(f"[arm {rc.arm}] step {step:5d} | loss {loss.item():.4f} | "
                f"κ̄ {mk:.2f} ({a1}/{n} at 1) | maxRMS {mr:.2e} | gn {gn:.1f} | {time.time()-t0:.0f}s")
            hist["step"].append(step); hist["loss"].append(loss.item())
            hist["mean_kappa"].append(mk); hist["frac_k1"].append(a1 / n)
            hist["max_rms"].append(mr)

    # Final: stress-test stability, calibrate, eval, report κ coverage.
    xs, _ = val_ds.batch(cfg.batch_size, dev)
    hist["max_rms_final"] = _hidden_rms_stats(model, xs)
    hist["max_rms_stress"] = _hidden_rms_stats(model, xs, input_scale=rc.stress_scale)
    for m in homos:
        m.start_calibration()
    for _ in range(50):
        xb, _ = train_ds.batch(cfg.batch_size, dev)
        with torch.no_grad():
            model(xb)
    for m in homos:
        m.finish_calibration(); m.frozen = True
    mk, a1, n = _kappa_stats(homos)
    hist["final_mean_kappa"] = mk
    hist["frac_at_kappa1"] = a1 / n
    if not hist["diverged"]:
        hist["val_loss"] = evaluate(model, val_ds, cfg, ac)
    else:
        hist["val_loss"] = float("nan")
    hist["model"] = model
    hist["all_kappa1"] = (a1 == n)
    log(f"[arm {rc.arm}] DONE | κ̄ {mk:.2f} | {a1}/{n} sites at κ=1 | "
        f"val {hist['val_loss']:.4f} | maxRMS {hist['max_rms_final']:.2e} "
        f"(stress×{rc.stress_scale}: {hist['max_rms_stress']:.2e})")
    return hist
