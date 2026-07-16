"""χ-ViT image-classification trainer (Milestone 2).

Generic loop over a torchvision-style ``(images, labels)`` loader. Softmax +
cross-entropy live only in the loss; the χ-ViT feature graph stays tensor-pure.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from xvla.models.vit import ChiViT, ViTConfig
from xvla.train.balance import balance_model


@dataclass
class ViTTrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    warmup_frac: float = 0.05
    balance_every: int = 100
    log_every: int = 100
    device: str = "cuda"
    dtype: str = "bfloat16"


def _lr_at(step, total, cfg: ViTTrainConfig):
    warm = max(1, int(cfg.warmup_frac * total))
    if step < warm:
        return cfg.lr * step / warm
    prog = (step - warm) / max(1, total - warm)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


@torch.no_grad()
def evaluate_acc(model, loader, device, autocast_ctx):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast_ctx:
            logits, _ = model(imgs)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.numel()
    model.train()
    return correct / total


def train_vit(model_cfg: ViTConfig, cfg: ViTTrainConfig, train_loader, val_loader, log=print):
    dev = cfg.device
    model = ChiViT(model_cfg).to(dev)
    log(f"[χ-ViT {model_cfg.norm}] params={model.num_params()/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
                            weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    use_amp = cfg.dtype == "bfloat16" and dev.startswith("cuda")
    ac = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp \
        else torch.autocast(device_type="cpu", enabled=False)

    hist = {"acc": [], "epoch": []}
    step = 0
    t0 = time.time()
    model.train()
    for epoch in range(cfg.epochs):
        for imgs, labels in train_loader:
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, total_steps, cfg)
            imgs, labels = imgs.to(dev), labels.to(dev)
            opt.zero_grad(set_to_none=True)
            with ac:
                _, loss = model(imgs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if cfg.balance_every and step > 0 and step % cfg.balance_every == 0:
                balance_model(model)
            if step % cfg.log_every == 0:
                log(f"  ep{epoch} step{step} loss {loss.item():.3f} lr {opt.param_groups[0]['lr']:.1e} {time.time()-t0:.0f}s")
            step += 1
        acc = evaluate_acc(model, val_loader, dev, ac)
        hist["acc"].append(acc); hist["epoch"].append(epoch)
        log(f"[χ-ViT] epoch {epoch} | val acc {acc:.4f}")
    hist["best_acc"] = max(hist["acc"]) if hist["acc"] else 0.0
    hist["final_acc"] = hist["acc"][-1] if hist["acc"] else 0.0
    hist["model"] = model
    return hist
