"""Distill a per-token teacher into a strict scalar-norm student (spec §A/§7.5).

The working M1 model uses per-token RMS-norm (input-dependent → not tensor-pure).
The strict target uses *scalar* RmsBatchNorm everywhere: once calibrated and
frozen, every normalization is a fixed scalar rescaling (spec §1) with no
input-dependent division (§19 satisfied) — but it is unstable when trained from
scratch (DEVLOG Bug 4), because a fixed per-layer scalar cannot renormalize
per-instance residual-stream magnitude through degree-2 FFNs.

Strategy: warm-start the student from the teacher (identical weight shapes; only
the norm modules differ), then distill so the student learns a scale-consistent
residual stream a fixed scalar can handle:

    L = λ_kl · KL(teacher ‖ student) + λ_feat · Σ_ℓ MSE(hʲ_student, hʲ_teacher)
        + λ_ce · CE(student, targets)

Feature matching is the load-bearing term: it forces the student's hidden
activations to track the teacher's (well-scaled, per-token-normalized) stream,
teaching the scale-consistency that makes a frozen scalar norm viable.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from xvla.models.lm import ChiLanguageModel, LMConfig
from xvla.train.balance import balance_model
from xvla.train.calibrate import calibrate_rbn
from xvla.train.data import TokenBinDataset
from xvla.train.train_lm import TrainConfig, _lr_at, evaluate


@dataclass
class DistillConfig:
    lambda_kl: float = 1.0
    lambda_feat: float = 1.0
    lambda_ce: float = 0.1
    temperature: float = 2.0
    # Warm-starting the student's attention Q/K from a per-token teacher gives
    # scores tuned for per-token bounding; under scalar norm they're unbounded
    # and the forward explodes. Train from scratch instead (fresh small Q/K keep
    # scalar-normed, /d_h²-scaled scores bounded); distillation supplies the
    # teacher's function and constant stream magnitude.
    warm_start: bool = False


def _copy_shared_weights(teacher: ChiLanguageModel, student: ChiLanguageModel) -> int:
    """Copy every parameter/buffer with a matching name+shape (skips norm modules
    that differ between per-token and scalar variants). Returns count copied."""
    tsd = teacher.state_dict()
    ssd = student.state_dict()
    copied = 0
    for k, v in tsd.items():
        if k in ssd and ssd[k].shape == v.shape:
            ssd[k].copy_(v)
            copied += 1
    student.load_state_dict(ssd)
    return copied


def distill_strict(
    teacher: ChiLanguageModel,
    student_cfg: LMConfig,
    cfg: TrainConfig,
    dcfg: DistillConfig,
    log=print,
) -> dict:
    device = cfg.device
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = ChiLanguageModel(student_cfg).to(device)
    if dcfg.warm_start:
        n = _copy_shared_weights(teacher, student)
        log(f"warm-started student: copied {n} tensors from teacher")

    train_ds = TokenBinDataset(cfg.train_bin, cfg.seq_len)
    val_ds = TokenBinDataset(cfg.val_bin, cfg.seq_len)

    decay = [p for p in student.parameters() if p.dim() >= 2 and p.requires_grad]
    nodecay = [p for p in student.parameters() if p.dim() < 2 and p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
    )

    use_amp = cfg.dtype == "bfloat16" and device.startswith("cuda")
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_amp else torch.autocast(device_type="cpu", enabled=False))
    T = dcfg.temperature

    history = {"step": [], "loss": [], "val_loss": [], "val_step": [], "tag": "distill-strict"}
    t0 = time.time()
    student.train()
    for step in range(cfg.max_steps + 1):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, cfg)
        opt.zero_grad(set_to_none=True)

        x, y = train_ds.batch(cfg.batch_size, device)
        with torch.no_grad(), autocast_ctx:
            t_logits, _, t_hid = teacher(x, return_hidden=True)
        with autocast_ctx:
            s_logits, ce, s_hid = student(x, targets=y, return_hidden=True)
            # Logit KL (temperature-scaled). Flatten to (B·N, V) so batchmean
            # averages over all token distributions, not just the batch dim
            # (otherwise KL and its gradient are N× too large → weight blow-up).
            V = s_logits.size(-1)
            kl = F.kl_div(
                F.log_softmax(s_logits.float().reshape(-1, V) / T, dim=-1),
                F.log_softmax(t_logits.float().reshape(-1, V) / T, dim=-1),
                log_target=True, reduction="batchmean",
            ) * (T * T)
            # Feature matching, scale-normalized per layer by the teacher RMS.
            feat = s_logits.new_zeros(())
            for hs, ht in zip(s_hid, t_hid):
                denom = ht.float().pow(2).mean().clamp_min(1e-6)
                feat = feat + (hs.float() - ht.float()).pow(2).mean() / denom
            feat = feat / max(1, len(s_hid))
            loss = dcfg.lambda_kl * kl + dcfg.lambda_feat * feat + dcfg.lambda_ce * ce
        # Skip non-finite steps so one activation spike can't corrupt the weights
        # (spec §20 norm-instability guard). Track the skip rate as a signal.
        if not torch.isfinite(loss):
            history.setdefault("skipped", 0)
            history["skipped"] += 1
            continue
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        if not torch.isfinite(gnorm):
            opt.zero_grad(set_to_none=True)
            history.setdefault("skipped", 0)
            history["skipped"] += 1
            continue
        opt.step()

        if cfg.balance_every and step > 0 and step % cfg.balance_every == 0:
            balance_model(student)

        if step % cfg.log_every == 0:
            log(f"[distill] step {step:5d} | loss {loss.item():.4f} "
                f"(kl {kl.item():.3f} feat {feat.item():.3f} ce {ce.item():.3f}) "
                f"| gnorm {gnorm:.2f} | {time.time()-t0:.1f}s")
            history["step"].append(step); history["loss"].append(loss.item())

        if cfg.eval_every and step % cfg.eval_every == 0:
            # Calibrate scalar RBNs fresh, then eval the frozen (deployable) graph.
            calibrate_rbn(student, lambda: train_ds.batch(cfg.batch_size, device),
                          iters=cfg.calib_iters)
            vl = evaluate(student, val_ds, cfg, autocast_ctx)
            finite = math.isfinite(vl)
            log(f"[distill] step {step:5d} | STRICT VAL {vl:.4f} "
                f"| ppl {math.exp(min(vl,20)):.2f} | finite={finite}")
            history["val_step"].append(step); history["val_loss"].append(vl)

    history["final_val_loss"] = history["val_loss"][-1] if history["val_loss"] else None
    history["best_val_loss"] = min([v for v in history["val_loss"] if math.isfinite(v)],
                                   default=float("nan"))
    history["student"] = student
    return history
