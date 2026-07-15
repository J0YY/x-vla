"""Stage-1 language pretraining loop + four-way ablation driver (spec §13.3, §14).

Optimizer/schedule follow spec §13.3: AdamW, bf16 autocast, grad clip 1.0,
warmup 2–5% + cosine decay, weight decay sweepable, no dropout, periodic CP
factor balancing, and activation/factor-norm monitoring hooks.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from xvla.models.lm import ChiLanguageModel, LMConfig
from xvla.train.balance import balance_model
from xvla.train.calibrate import calibrate_rbn
from xvla.train.data import TokenBinDataset


@dataclass
class TrainConfig:
    train_bin: str
    val_bin: str
    seq_len: int = 512
    batch_size: int = 32
    grad_accum: int = 1
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_frac: float = 0.03
    max_steps: int = 2000
    balance_every: int = 100
    eval_every: int = 200
    eval_iters: int = 40
    calib_iters: int = 50
    log_every: int = 20
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 1337


def _lr_at(step: int, cfg: TrainConfig) -> float:
    warmup = max(1, int(cfg.warmup_frac * cfg.max_steps))
    if step < warmup:
        return cfg.lr * step / warmup
    if step >= cfg.max_steps:
        return cfg.lr * cfg.min_lr_ratio
    prog = (step - warmup) / max(1, cfg.max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * prog))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * coeff)


@torch.no_grad()
def evaluate(model, dataset, cfg: TrainConfig, autocast_ctx) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = dataset.batch(cfg.batch_size, cfg.device)
        with autocast_ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def train_lm(model_cfg: LMConfig, cfg: TrainConfig, log=print) -> dict:
    torch.manual_seed(cfg.seed)
    device = cfg.device
    model = ChiLanguageModel(model_cfg).to(device)
    log(f"[{model_cfg.tag}] params={model.num_params()/1e6:.1f}M")

    train_ds = TokenBinDataset(cfg.train_bin, cfg.seq_len)
    val_ds = TokenBinDataset(cfg.val_bin, cfg.seq_len)

    # AdamW with decay only on 2D+ weights (spec §13.3).
    decay = [p for p in model.parameters() if p.dim() >= 2 and p.requires_grad]
    nodecay = [p for p in model.parameters() if p.dim() < 2 and p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
    )

    use_amp = cfg.dtype in ("bfloat16", "float16") and device.startswith("cuda")
    amp_dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp else torch.autocast(device_type="cpu", enabled=False)
    )

    history = {"step": [], "train_loss": [], "val_loss": [], "val_step": [],
               "imbalance": [], "tag": model_cfg.tag}
    t0 = time.time()
    model.train()
    for step in range(cfg.max_steps + 1):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, cfg)

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum):
            x, y = train_ds.batch(cfg.batch_size, device)
            with autocast_ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum
            loss.backward()
            accum_loss += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if cfg.balance_every and step > 0 and step % cfg.balance_every == 0:
            imb = balance_model(model)
            history["imbalance"].append((step, imb))

        if step % cfg.log_every == 0:
            dt = time.time() - t0
            log(f"[{model_cfg.tag}] step {step:5d} | loss {accum_loss:.4f} "
                f"| lr {opt.param_groups[0]['lr']:.2e} | gnorm {grad_norm:.2f} "
                f"| {dt:.1f}s")
            history["step"].append(step)
            history["train_loss"].append(accum_loss)

        if cfg.eval_every and step % cfg.eval_every == 0:
            # Fresh RBN calibration so eval mirrors the folded/deployed graph
            # (spec §7.5) — the training EMA is unreliable, esp. early on.
            calibrate_rbn(model, lambda: train_ds.batch(cfg.batch_size, device),
                          iters=cfg.calib_iters)
            vl = evaluate(model, val_ds, cfg, autocast_ctx)
            log(f"[{model_cfg.tag}] step {step:5d} | VAL loss {vl:.4f} "
                f"| ppl {math.exp(min(vl, 20)):.2f}")
            history["val_step"].append(step)
            history["val_loss"].append(vl)

    history["final_val_loss"] = history["val_loss"][-1] if history["val_loss"] else None
    # Best (min) val is the fair capability metric — final val is confounded by
    # overfitting on small corpora (tiny-shakespeare).
    history["best_val_loss"] = min(history["val_loss"]) if history["val_loss"] else None
    history["params_M"] = model.num_params() / 1e6
    return history


def four_way_ablation(base_cfg: LMConfig, cfg: TrainConfig, log=print) -> dict:
    """Run the spec §14 Stage-1 four-way ablation and return per-config history."""
    from dataclasses import replace
    results = {}
    for attn in ("softmax", "bilinear"):
        for ffn in ("swiglu", "bilinear"):
            mc = replace(base_cfg, attn=attn, ffn=ffn)
            log(f"\n===== ablation: attn={attn} ffn={ffn} =====")
            results[mc.tag] = train_lm(mc, cfg, log=log)
    log("\n===== ablation summary =====")
    for tag, h in results.items():
        log(f"  {tag:20s}  best_val={h['best_val_loss']:.4f}  "
            f"final_val={h['final_val_loss']:.4f}  params={h['params_M']:.1f}M")
    return results
